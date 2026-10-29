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

r"""GeoClip: geometry-aware clipping for DP-SGD.

Implements GeoClip (Algorithm 1) from:
  Gilani et al. "Geometry-Aware Clipping for DP-SGD." NeurIPS 2025.
  arXiv:2506.06549.

Each per-example gradient :math:`g` is centered by a running mean :math:`a`
and rescaled inside the top-``rank`` principal subspace :math:`U` of past
released gradients before it is clipped, so that clipping and noise follow the
shape of the gradient distribution rather than a sphere. The mean and the
covariance behind :math:`U` are estimated from the privatized gradients only,
so they cost no extra privacy budget.

Example Usage (clipping but no noise):
  >>> from jax_privacy import clipped_grad
  >>> import optax
  >>> import jax.numpy as jnp
  >>> loss_fn = lambda params, batch: 0.5 * jnp.mean((params - batch) ** 2)
  >>> optimizer = geoclip(optax.sgd(0.1), rank=2, expected_batch_size=10)
  >>> params = jnp.ones(3)
  >>> data = jnp.zeros((10, 3))
  >>> state = optimizer.init(params)
  >>> grad_fn = clipped_grad(
  ...     loss_fn,
  ...     l2_clip_norm=1.0,
  ...     pre_clipping_transform=optimizer.pre_clipping_transform(state),
  ... )
  >>> updates, state = optimizer.update(grad_fn(params, data), state, params)
  >>> updates.shape
  (3,)
  >>> int(state.count)
  1
"""

from typing import NamedTuple

import jax
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree
from jax_privacy import optimizers
import optax


class GeoClipState(NamedTuple):
  r"""State for :func:`geoclip`.

  Attributes:
    mean: Running mean :math:`a` of released gradients. Same structure as the
      parameters.
    cov: A ``(d, d)`` running estimate :math:`\Sigma` of the per-example
      gradient covariance, where ``d`` is the total number of parameters.
    eigvecs: A ``(d, rank)`` matrix :math:`U` with orthonormal columns: the
      top-``rank`` eigenvectors of ``cov``.
    eigvals: A ``(rank,)`` vector :math:`\lambda` of the matching eigenvalues,
      clamped to ``[h1, h2]``.
    count: Number of statistics updates so far. While it is zero the
      pre-clipping transform is the identity, so a step is plain DP-SGD.
    inner_state: State of the base optimizer.
  """

  mean: optax.Updates
  cov: jax.Array
  eigvecs: jax.Array
  eigvals: jax.Array
  count: jax.Array
  inner_state: optax.OptState


def _subspace_scales(eigvals: jax.Array) -> jax.Array:
  r"""Returns the per-direction scales :math:`s = c \lambda^{-1/4}`.

  Here :math:`c = (\sum_i \sqrt{\lambda_i})^{-1/2}`, i.e. the paper's
  :math:`\gamma = 1`.
  """
  c = jnp.sqrt(1.0 / jnp.sum(jnp.sqrt(eigvals)))
  return c * eigvals**-0.25


def geoclip(
    base_optimizer: optax.GradientTransformation,
    rank: int,
    expected_batch_size: float,
    *,
    normalize_by: float = 1.0,
    beta1: float = 0.99,
    beta2: float = 0.999,
    h1: float = 1e-15,
    h2: float = 10.0,
) -> optimizers.AugmentedGradientTransformation:
  r"""Constructs an AugmentedGradientTransformation for GeoClip.

  Before clipping, each flattened per-example gradient :math:`g` is mapped to

  .. math::

    \omega = U \operatorname{diag}(s) U^\top (g - a),
    \qquad s = c \lambda^{-1/4},

  where :math:`a, U, \lambda` are read from :class:`GeoClipState` and the
  constant :math:`c` sets the clip region, so ``clipped_grad`` should use
  ``l2_clip_norm=1.0``. The component of :math:`g - a` outside the span of
  :math:`U` is dropped, as in the paper.

  After clipping, summing and noising, ``update`` applies the inverse

  .. math::

    \tilde g = U \operatorname{diag}(1/s) U^\top \tilde\omega + (B / n) a,

  where :math:`B` is ``expected_batch_size`` and :math:`n` is
  ``normalize_by``, then calls the base optimizer. Projecting onto :math:`U`
  also discards noise outside the subspace. While ``count`` is zero both maps
  are the identity.

  ``update`` then refreshes the statistics from the batch-average gradient
  :math:`\bar g = (n / B)\,\tilde g`, as in Algorithm 1 of the paper:

  .. math::

    a \leftarrow \beta_1 a + (1 - \beta_1) \bar g, \qquad
    \Sigma \leftarrow \beta_2 \Sigma
      + (1 - \beta_2)\, B\, (\bar g - a)(\bar g - a)^\top,

  with :math:`a` on the right of the second line already updated, and
  :math:`\Sigma` starting at the identity. The factor :math:`B` turns the
  covariance of a batch average into a per-example covariance (paper eq. 20).
  :math:`U` and :math:`\lambda` are the top-``rank`` eigenpairs of
  :math:`\Sigma`, with :math:`\lambda` clamped to ``[h1, h2]``.

  The state holds the full ``(d, d)`` covariance and ``update`` computes its
  eigendecomposition, so memory is :math:`O(d^2)` and each step costs
  :math:`O(d^3)`. This suits small models such as linear and last-layer
  fine-tuning; the paper's streaming low-rank variant (Algorithm 2) is not
  implemented here.

  Args:
    base_optimizer: The optimizer that receives the un-transformed gradient.
    rank: Number of principal directions :math:`k` to keep.
    expected_batch_size: Expected number of examples per step, :math:`B`.
    normalize_by: The ``normalize_by`` passed to ``clipped_grad``, :math:`n`.
      Use the default 1.0 when ``update`` receives the noisy clipped sum, and
      :math:`B` when it receives the batch average.
    beta1: Decay of the running mean.
    beta2: Decay of the running covariance.
    h1: Lower clamp on the eigenvalues; keeps :math:`\lambda^{-1/4}` finite.
    h2: Upper clamp on the eigenvalues. The paper tunes it in ``{1, 10}``.

  Returns:
    An :class:`~jax_privacy.optimizers.AugmentedGradientTransformation` for
    GeoClip.

  Raises:
    ValueError: If a hyperparameter is out of range.
  """
  if expected_batch_size <= 0:
    raise ValueError(
        f'expected_batch_size must be positive, got {expected_batch_size}.'
    )
  if normalize_by <= 0:
    raise ValueError(f'normalize_by must be positive, got {normalize_by}.')
  if not (0 <= beta1 < 1 and 0 <= beta2 < 1):
    raise ValueError(f'Need beta1, beta2 in [0, 1), got {beta1}, {beta2}.')
  if not 0 < h1 <= h2:
    raise ValueError(f'Need 0 < h1 <= h2, got h1={h1}, h2={h2}.')
  mean_scale = expected_batch_size / normalize_by

  def init(params):
    flat_params, _ = ravel_pytree(params)
    d = flat_params.size
    if not 1 <= rank <= d:
      raise ValueError(
          f'rank must be in [1, {d}] (the number of parameters), got {rank}.'
      )
    # Algorithm 1 starts from Sigma = I. Its top eigenvectors are taken to be
    # the first standard basis vectors, as in the paper's Algorithm 3. The two
    # are separate arrays: with rank == d a slice of cov could share its
    # buffer, which breaks callers that donate the state (e.g. DPTrainer).
    return GeoClipState(
        mean=optax.tree.zeros_like(params),
        cov=jnp.eye(d, dtype=flat_params.dtype),
        eigvecs=jnp.eye(d, rank, dtype=flat_params.dtype),
        eigvals=jnp.ones((rank,), flat_params.dtype),
        count=jnp.zeros([], jnp.int32),
        inner_state=base_optimizer.init(params),
    )

  def pre_clipping_transform(state, inverse=False):
    """Builds the GeoClip map (or its inverse) from the current state."""
    flat_mean, _ = ravel_pytree(state.mean)
    u = state.eigvecs
    scales = _subspace_scales(state.eigvals)

    def transform(updates):
      flat, unravel = ravel_pytree(updates)
      if inverse:
        out = u @ ((u.T @ flat) / scales) + mean_scale * flat_mean
      else:
        out = u @ (scales * (u.T @ (flat - flat_mean)))
      return unravel(jnp.where(state.count == 0, flat, out))

    return transform

  def update(updates, state, params=None, **extra_args):
    """Applies the inverse transform, the base update, then Algorithm 1."""
    unscaled_updates = pre_clipping_transform(state, inverse=True)(updates)
    new_updates, inner_state = base_optimizer.update(
        unscaled_updates, state.inner_state, params, **extra_args
    )

    flat, unravel = ravel_pytree(unscaled_updates)
    avg_grad = flat / mean_scale
    flat_mean, _ = ravel_pytree(state.mean)
    mean = beta1 * flat_mean + (1 - beta1) * avg_grad
    centered = avg_grad - mean
    cov = beta2 * state.cov + (1 - beta2) * expected_batch_size * jnp.outer(
        centered, centered
    )
    eigvals, eigvecs = jnp.linalg.eigh(cov)  # Ascending order.
    new_state = GeoClipState(
        mean=unravel(mean),
        cov=cov,
        eigvecs=eigvecs[:, -rank:],
        eigvals=jnp.clip(eigvals[-rank:], h1, h2),
        count=state.count + 1,
        inner_state=inner_state,
    )
    return new_updates, new_state

  return optimizers.AugmentedGradientTransformation(
      init=init,
      update=update,
      pre_clipping_transform=pre_clipping_transform,
  )
