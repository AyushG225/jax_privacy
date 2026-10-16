# Copyright 2026 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Augmented gradient transformations for differentially private training.

Gradient-based DP training algorithms may need to specify pre-processing of the
per-example gradients before clipping and noising happens. Because this is
tightly linked to the optimizer, we provide an
:class:`~jax_privacy.optimizers.AugmentedGradientTransformation` that provides a
pre-processing function (the ``pre_clipping_transform`` that can be passed into
:func:`~jax_privacy.clipping.clipped_grad`) bound together with the core
optimizer update.

The primary use case is the "scale-then-privatize" technique from:
  Ganesh, McMahan, Thakurta. "On Design Principles for Private Adaptive
  Optimizers." arXiv:2507.01129.

Example Usage (clipping but no noise):
  >>> from jax_privacy import clipped_grad, noise_addition
  >>> import optax
  >>> import jax.numpy as jnp
  >>> loss_fn = lambda params, batch: 0.5 * jnp.mean((params - batch) ** 2)
  >>> optimizer = scale_then_privatize(optax.adamw(1e-3))
  >>> params = jnp.ones(3)
  >>> data = jnp.ones((10, 3))
  >>> state = optimizer.init(params)
  >>> noise_multiplier = 0.0
  >>> noise_state = noise_addition.gaussian_privatizer(stddev=0.0).init(params)
  >>> for _ in range(5):
  ...   grad_fn = clipped_grad(
  ...     loss_fn,
  ...     l2_clip_norm=1,
  ...     pre_clipping_transform=optimizer.pre_clipping_transform(state)
  ...   )
  ...   stddev = grad_fn.sensitivity() * noise_multiplier
  ...   noise_fn = noise_addition.gaussian_privatizer(stddev=stddev)
  ...   clipped_grads = grad_fn(params, data)
  ...   noisy_grads, noise_state = noise_fn.update(clipped_grads, noise_state)
  ...   updates, state = optimizer.update(noisy_grads, state, params)
  ...   params = optax.apply_updates(params, updates)
"""

from typing import Callable, NamedTuple, Protocol

import jax
from jax import numpy as jnp
import optax


class PreClippingTransform(Protocol):
  """A function that applies a transformation to a pytree of updates."""

  def __call__(
      self, updates: optax.Updates, inverse: bool = False
  ) -> optax.Updates:
    ...


class AugmentedGradientTransformation(NamedTuple):
  """A gradient transformation augmented with a pre-clipping transform.

  This extends the standard optax.GradientTransformation interface with a
  ``pre_clipping_transform`` field that, given the current optimizer state,
  returns a ``pre_clipping_transform`` specifying how to transform per-example
  gradients before and after the clipping/noising step.

  The ``update`` function expects to receive gradients that have already been
  transformed by ``pre_clipping_transform(...)``, clipped, summed,
  and noised. It will internally apply the inverse transform before delegating
  to the base optimizer's update. See the module docstring for an example usage.

  Attributes:
    init: Initializes the optimizer state given initial parameters. Matches the
      optax.GradientTransformation.init API: init(params) -> state
    update: Computes parameter updates from noisy gradients. The noisy gradients
      should be in the *scaled* space (transform -> clip -> aggregate -> noise).
      This function applies the inverse transform internally before calling the
      base optimizer's update: update(updates, state, params=None) -> (updates,
      new_state)
    pre_clipping_transform: Given the current optimizer state, returns a
      ``pre_clipping_transform`` function intended to be used with
      :func:`~jax_privacy.clipping.clipped_grad`. It consumes a pytree with
      structure matching the parameters and returns a transformed pytree. The
      transformed pytree may or may not have the same structure. The ``update``
      function is responsible for mapping the input back to the original
      structure.
  """

  init: Callable[[optax.Params], optax.OptState]
  update: Callable[..., tuple[optax.Updates, optax.OptState]]
  pre_clipping_transform: Callable[[optax.OptState], PreClippingTransform]


def _find_adaptive_state(state: optax.OptState) -> optax.Updates:
  """Recursively searches for a recognized adaptive optimizer state."""
  # These are all adaptive optimizers where the square root of the second
  # moments of gradients is the appropriate scaling.
  if isinstance(state, optax.ScaleByAdamState):
    return state.nu
  elif isinstance(state, optax.ScaleByAmsgradState):
    return state.nu_max
  elif isinstance(state, optax.ScaleByRmsState):
    return state.nu
  elif isinstance(state, optax.ScaleByRssState):
    return state.sum_of_squares

  # If the state is a tuple/list (e.g., from optax.chain), search recursively.
  if isinstance(state, (tuple, list)):
    for sub_state in state:
      result = _find_adaptive_state(sub_state)
      if result is not None:
        return result

  raise ValueError(
      f'Could not find an adaptive optimizer state in {type(state)}.'
      ' scale_then_privatize requires an adaptive optimizer (e.g.,'
      ' optax.adam, optax.adamw, optax.rmsprop, optax.adagrad). If you are'
      ' using a custom adaptive optimizer, pass a custom'
      ' ``extract_preconditioner_from_state_fn`` function to'
      ' scale_then_privatize.'
  )


def scale_then_privatize(
    base_optimizer: optax.GradientTransformation,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    extract_preconditioner_from_state_fn: (
        Callable[[optax.OptState], optax.Updates] | None
    ) = None,
) -> AugmentedGradientTransformation:
  r"""Constructs an AugmentedGradientTransformation for scale-then-privatize.

  This implements Algorithm 8 from Ganesh, McMahan, Thakurta (2507.01129).
  The key idea is to use the optimizer's second-moment estimate :math:`v_{t-1}`
  from the previous step to define a non-isotropic geometry for clipping and
  noising per-example gradients. Specifically:

  .. math::

    s_t = 1 / (\sqrt{v_{t-1} + \text{eps\_root}} + \text{eps})

  Before clipping, each per-example gradient :math:`g` is transformed to
  :math:`s_t \odot g`. After clipping + aggregation + noise addition, the
  ``update`` function applies the inverse (divides by :math:`s_t`) before
  passing to the base optimizer's update.

  A large ``eps`` or ``eps_root`` passed here (but not in the adaptive
  optimizer's scaling) will cause all coordinates to be scaled
  nearly-identically, effectively retrieving no pre-clipping transform.
  ``eps`` or ``eps_root`` matching the adaptive optimizer may add large noise
  in coordinates where the gradient is small. Ideally, this parameter should
  be tuned to tradeoff between these two regimes.

  Args:
    base_optimizer: A standard :class:`optax.GradientTransformation`, typically
      an adaptive optimizer like ``optax.adamw(...)``, ``optax.adam(...)``, or
      any chained transformation containing a ``scale_by_adam`` (or similar)
      component.
    eps: A small constant added to the denominator outside the square root when
      computing the scaling vector :math:`s_t`. Analogous to the ``eps``
      parameter in Adam. This also acts as a stability constant to prevent
      excessively large scaling in coordinates where :math:`v` is very small.
      Corresponds to :math:`\varepsilon_{s_1}` in Algorithm 8 of the paper. See
      the note above on tuning this parameter.
    eps_root: A small constant added to :math:`v` inside the square root,
      analogous to ``eps_root`` in :func:`optax.scale_by_adam`. See the note
      above on tuning this parameter.
    extract_preconditioner_from_state_fn: A function that takes the optimizer
      state and returns the second-moment estimate (:math:`v`) pytree. If
      ``None``, uses a default implementation that handles common optax adaptive
      optimizers (Adam, AMSGrad, RMSProp, AdaGrad).

  Returns:
    An :class:`~jax_privacy.optimizers.AugmentedGradientTransformation` for the
    scale-then-privatize technique.
  """
  if extract_preconditioner_from_state_fn is None:
    extract_preconditioner_from_state_fn = _find_adaptive_state

  def pre_clipping_transform(state, inverse=False):
    """Extracts ν̂ from the optimizer state and builds the scaling transform."""
    # Compute the scaling vector: s = 1 / (sqrt(ν + eps_root) + eps).
    # It is the same formula used in Adam for per-coordinate learning rates.
    nu = extract_preconditioner_from_state_fn(state)
    # TODO: b/415360727 - Investigate+improve numerical stability when v = 0.
    scaling = jax.tree.map(lambda v: 1.0 / (jnp.sqrt(v + eps_root) + eps), nu)
    scale_fn = jnp.divide if inverse else jnp.multiply
    return lambda updates: jax.tree.map(
        lambda u, s: jnp.astype(scale_fn(u, s), u.dtype), updates, scaling
    )

  def update(updates, state, params, **extra_args):
    """Applies the inverse scaling transform, then the base optimizer update."""
    unscaled_updates = pre_clipping_transform(state, inverse=True)(updates)
    return base_optimizer.update(unscaled_updates, state, params, **extra_args)

  return AugmentedGradientTransformation(
      init=base_optimizer.init,
      update=update,
      pre_clipping_transform=pre_clipping_transform,
  )


class _DP2RMSPropState(NamedTuple):
  """Optimizer state for DP2-RMSProp."""

  count: jax.Array
  grad_accumulator: optax.Updates
  nu: optax.Updates
  inner_state: optax.OptState


def dp2_rmsprop(
    sgd_learning_rate: float,
    rmsprop_learning_rate: float,
    interval: int,
    sgd_clip_norm: float,
    rmsprop_clip_norm: float,
    decay: float = 0.9,
    eps: float = 1e-8,
    rescale_to_unit_norm: bool = True,
    *,
    inner_optimizer: optax.GradientTransformation | None = None,
) -> AugmentedGradientTransformation:
  r"""Constructs an AugmentedGradientTransformation for DP2-RMSProp.

  This implements Algorithm 1 from Li et al. (2212.00309). It alternates
  ``interval`` private SGD steps with ``interval`` private RMSProp steps. At
  each SGD-to-RMSProp transition, the average private SGD gradient
  :math:`\bar{g}` updates the delayed second moment:

  .. math::

    v \leftarrow \text{decay} \cdot v + (1 - \text{decay}) \bar{g}^{2}

  Before clipping, SGD gradients are unchanged; RMSProp gradients are scaled
  by ``sgd_clip_norm / rmsprop_clip_norm`` and divided by
  :math:`\sqrt{v} + \text{eps}`. After clipping, aggregation, and noising,
  ``update`` undoes only the clipping scale, preserving the preconditioning.
  It delegates to the inner optimizer and scales its updates by the phase
  learning rate. Inner state persists across phases.

  Based on the authors' Apache-2.0 JAX implementation:
  https://github.com/kenziyuliu/DP2.

  Example Usage:
    >>> optimizer = dp2_rmsprop(0.1, 0.01, 2, 1.0, 0.5)
    >>> state = optimizer.init(jnp.zeros(1))
    >>> updates, _ = optimizer.update(jnp.ones(1), state)
    >>> bool(jnp.allclose(updates, jnp.array([-0.1])))
    True

  Args:
    sgd_learning_rate: Nonnegative multiplier on inner updates in SGD phases.
    rmsprop_learning_rate: Nonnegative multiplier on inner updates in RMSProp
      phases.
    interval: Positive integer number of steps in each phase.
    sgd_clip_norm: Positive SGD clipping norm. Must match the enclosing
      configuration's ``l2_clip_norm``.
    rmsprop_clip_norm: Positive clipping norm for preconditioned gradients.
    decay: Second-moment decay in ``[0, 1)``.
    eps: Positive stability constant in the RMSProp denominator.
    rescale_to_unit_norm: Must match the enclosing clipping configuration's
      ``rescale_to_unit_norm``.
    inner_optimizer: An :class:`optax.GradientTransformation`, defaulting to
      ``optax.sgd(1.0)``. Its learning rate multiplies the phase rate; adaptive
      inner optimizers change standard DP2-RMSProp. Receives ``params`` and
      extra update arguments.

  Returns:
    An :class:`~jax_privacy.optimizers.AugmentedGradientTransformation` for
    DP2-RMSProp.

  Raises:
    TypeError: If ``interval`` is not a Python integer.
    ValueError: If a hyperparameter is outside its allowed range.
  """
  if isinstance(interval, bool) or not isinstance(interval, int):
    raise TypeError('interval must be a Python integer.')
  if interval <= 0:
    raise ValueError('interval must be positive.')
  if sgd_learning_rate < 0 or rmsprop_learning_rate < 0:
    raise ValueError('learning rates must be nonnegative.')
  if sgd_clip_norm <= 0 or rmsprop_clip_norm <= 0:
    raise ValueError('clipping norms must be positive.')
  if eps <= 0:
    raise ValueError('eps must be positive.')
  if not 0 <= decay < 1:
    raise ValueError('decay must be in [0, 1).')

  if inner_optimizer is None:
    inner_optimizer = optax.sgd(1.0)
  inner_optimizer = optax.with_extra_args_support(inner_optimizer)

  cycle_length = 2 * interval
  clip_ratio = sgd_clip_norm / rmsprop_clip_norm
  sgd_decode_scale = sgd_clip_norm if rescale_to_unit_norm else 1.0
  rmsprop_decode_scale = (
      rmsprop_clip_norm
      if rescale_to_unit_norm
      else rmsprop_clip_norm / sgd_clip_norm
  )

  def init_fn(params):
    zeros = jax.tree.map(jnp.zeros_like, params)
    return _DP2RMSPropState(
        count=jnp.zeros([], dtype=jnp.int32),
        grad_accumulator=zeros,
        nu=zeros,
        inner_state=inner_optimizer.init(params),
    )

  def pre_clipping_transform(state):
    use_sgd = state.count % cycle_length < interval

    def rmsprop_transform(gradients):
      return jax.tree.map(
          lambda g, v: jnp.astype(
              clip_ratio * g / (jnp.sqrt(v) + eps), g.dtype
          ),
          gradients,
          state.nu,
      )

    def transform(gradients):
      return jax.lax.cond(use_sgd, lambda x: x, rmsprop_transform, gradients)

    return transform

  def update_fn(updates, state, params=None, **extra_args):
    cycle_step = state.count % cycle_length
    use_sgd = cycle_step < interval
    last_sgd_step = cycle_step == interval - 1

    decode_scale = jnp.where(use_sgd, sgd_decode_scale, rmsprop_decode_scale)
    decoded_grad = jax.tree.map(
        lambda g: jnp.astype(decode_scale * g, g.dtype), updates
    )

    accumulator = jax.lax.cond(
        use_sgd,
        lambda acc: jax.tree.map(jnp.add, acc, decoded_grad),
        lambda acc: acc,
        state.grad_accumulator,
    )

    def finish_sgd_phase(accumulator):
      average = jax.tree.map(lambda g: g / interval, accumulator)
      new_nu = jax.tree.map(
          lambda v, g: decay * v + (1.0 - decay) * jnp.square(g),
          state.nu,
          average,
      )
      return new_nu, jax.tree.map(jnp.zeros_like, accumulator)

    new_nu, new_accumulator = jax.lax.cond(
        last_sgd_step,
        finish_sgd_phase,
        lambda acc: (state.nu, acc),
        accumulator,
    )

    inner_updates, new_inner_state = inner_optimizer.update(
        decoded_grad, state.inner_state, params, **extra_args
    )
    learning_rate = jnp.where(use_sgd, sgd_learning_rate, rmsprop_learning_rate)
    parameter_updates = jax.tree.map(
        lambda u: jnp.astype(learning_rate * u, u.dtype), inner_updates
    )
    new_state = _DP2RMSPropState(
        count=state.count + 1,
        grad_accumulator=new_accumulator,
        nu=new_nu,
        inner_state=new_inner_state,
    )
    return parameter_updates, new_state

  return AugmentedGradientTransformation(
      init=init_fn,
      update=update_fn,
      pre_clipping_transform=pre_clipping_transform,
  )


def as_augmented_optimizer(
    optimizer: AugmentedGradientTransformation | optax.GradientTransformation,
) -> AugmentedGradientTransformation:
  """Wraps a plain Optax optimizer with an identity pre-clipping transform."""
  if isinstance(optimizer, AugmentedGradientTransformation):
    return optimizer
  return AugmentedGradientTransformation(
      init=optimizer.init,
      update=optimizer.update,
      pre_clipping_transform=lambda opt_state: lambda x: x,
  )
