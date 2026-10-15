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

Implements the pre-clipping transform from:
  Gilani et al. "Geometry-Aware Clipping for DP-SGD." NeurIPS 2025.
  arXiv:2506.06549.

Each per-example gradient :math:`g` is centered by a running mean :math:`a`
and rescaled inside the top-``rank`` principal subspace :math:`U` of past
released gradients before it is clipped, so that clipping and noise follow the
shape of the gradient distribution rather than a sphere.

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
    eigvecs: A ``(d, rank)`` matrix :math:`U` with orthonormal columns spanning
      the top-``rank`` directions of the gradient distribution, where ``d`` is
      the total number of parameters.
    eigvals: A ``(rank,)`` vector :math:`\lambda` of gradient variances along
      the columns of ``eigvecs``.
    count: Number of statistics updates so far. While it is zero the
      pre-clipping transform is the identity, so a step is plain DP-SGD.
    inner_state: State of the base optimizer.
  """

  mean: optax.Updates
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
    expected_batch_size: int,
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

    \tilde g = U \operatorname{diag}(1/s) U^\top \tilde\omega + B a,

  where :math:`B` is ``expected_batch_size``, before calling the base
  optimizer. Projecting onto :math:`U` also discards noise outside the
  subspace. While ``count`` is zero both maps are the identity.

  This version does not yet update ``mean``, ``eigvecs``, ``eigvals`` or
  ``count``, so training with it is equivalent to DP-SGD.

  Args:
    base_optimizer: The optimizer that receives the un-transformed gradient.
    rank: Number of principal directions :math:`k` to keep.
    expected_batch_size: Expected number of examples summed per step,
      :math:`B`. Used to add back the mean subtracted from each example. If
      the clipped sum is divided by ``normalize_by``, pass the expected batch
      size divided by ``normalize_by``.

  Returns:
    An :class:`~jax_privacy.optimizers.AugmentedGradientTransformation` for
    GeoClip.
  """

  def init(params):
    flat_params, _ = ravel_pytree(params)
    if not 1 <= rank <= flat_params.size:
      raise ValueError(
          f'rank must be in [1, {flat_params.size}] (the number of'
          f' parameters), got {rank}.'
      )
    return GeoClipState(
        mean=optax.tree.zeros_like(params),
        eigvecs=jnp.zeros((flat_params.size, rank), flat_params.dtype),
        # Unused while count == 0; ones keep the scales finite.
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
        out = u @ ((u.T @ flat) / scales) + expected_batch_size * flat_mean
      else:
        out = u @ (scales * (u.T @ (flat - flat_mean)))
      return unravel(jnp.where(state.count == 0, flat, out))

    return transform

  def update(updates, state, params=None, **extra_args):
    """Applies the inverse transform, then the base optimizer update."""
    unscaled_updates = pre_clipping_transform(state, inverse=True)(updates)
    new_updates, inner_state = base_optimizer.update(
        unscaled_updates, state.inner_state, params, **extra_args
    )
    # TODO: Update mean, eigvecs, eigvals and count from
    # unscaled_updates / expected_batch_size (paper Algorithm 2).
    return new_updates, state._replace(inner_state=inner_state)

  return optimizers.AugmentedGradientTransformation(
      init=init,
      update=update,
      pre_clipping_transform=pre_clipping_transform,
  )
