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
from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
from jax_privacy import clipping
from jax_privacy import noise_addition
from jax_privacy import optimizers
from jax_privacy.experimental import geoclip
import optax

_RANK = 2
_BATCH_SIZE = 4
_DATA = jnp.array([
    [1.0, 2.0, 0.0, 5.0],
    [-3.0, 1.0, 2.0, 0.0],
    [0.5, 0.5, 4.0, -2.0],
    [2.0, -1.0, 1.0, 3.0],
])


def _init_params():
  """Pytree params with mixed shapes; 4 parameters in total."""
  return {'w': jnp.array([1.0, -2.0, 0.5]), 'b': jnp.array(0.3)}


def _loss(params, batch):
  """Squared error of a linear model; each row of batch is (x0, x1, x2, y)."""
  pred = batch[:, :3] @ params['w'] + params['b']
  return 0.5 * jnp.mean((pred - batch[:, 3]) ** 2)


def _random_state(optimizer, params, key):
  """A non-identity GeoClipState with orthonormal eigvecs and nonzero mean."""
  state = optimizer.init(params)
  flat_params, unravel = ravel_pytree(params)
  k1, k2, k3 = jax.random.split(key, 3)
  eigvecs, _ = jnp.linalg.qr(jax.random.normal(k1, (flat_params.size, _RANK)))
  return state._replace(
      mean=unravel(jax.random.normal(k2, flat_params.shape)),
      eigvecs=eigvecs,
      eigvals=jax.random.uniform(k3, (_RANK,), minval=0.1, maxval=10.0),
      count=jnp.ones([], jnp.int32),
  )


class GeoClipTest(parameterized.TestCase):

  def test_output_matches_params_structure(self):
    """Transform and update outputs have the same structure as params."""
    params = _init_params()
    optimizer = geoclip.geoclip(optax.sgd(0.1), _RANK, _BATCH_SIZE)
    state = _random_state(optimizer, params, jax.random.key(0))
    chex.assert_shape(state.cov, (4, 4))
    chex.assert_shape(state.eigvecs, (4, _RANK))
    chex.assert_shape(state.eigvals, (_RANK,))

    grads = jax.tree.map(jnp.ones_like, params)
    transformed = optimizer.pre_clipping_transform(state)(grads)
    chex.assert_trees_all_equal_shapes_and_dtypes(transformed, params)
    updates, new_state = optimizer.update(transformed, state, params)
    chex.assert_trees_all_equal_shapes_and_dtypes(updates, params)
    chex.assert_trees_all_equal_shapes_and_dtypes(new_state, state)

  def test_init_is_identity_covariance(self):
    """Algorithm 1 starts from Sigma = I and the first basis vectors."""
    state = geoclip.geoclip(optax.sgd(0.1), _RANK, _BATCH_SIZE).init(
        _init_params()
    )
    chex.assert_trees_all_close(state.cov, jnp.eye(4))
    chex.assert_trees_all_close(state.eigvecs, jnp.eye(4)[:, :_RANK])
    chex.assert_trees_all_close(state.eigvals, jnp.ones(_RANK))
    self.assertEqual(int(state.count), 0)

  @parameterized.parameters(dict(rank=_RANK), dict(rank=4))
  def test_state_can_be_donated(self, rank):
    """State leaves don't share buffers, so DPTrainer can donate the state."""
    params = _init_params()
    optimizer = geoclip.geoclip(optax.sgd(0.1), rank, _BATCH_SIZE)
    step = jax.jit(
        lambda s: optimizer.update(params, s, params)[1], donate_argnums=0
    )
    state = step(step(optimizer.init(params)))
    self.assertEqual(int(state.count), 2)

  @parameterized.parameters(dict(use_jit=False), dict(use_jit=True))
  def test_first_step_matches_dp_sgd(self, use_jit):
    """From init (count == 0), one noisy clipped step is plain DP-SGD."""

    def train(optimizer):
      privatizer = noise_addition.gaussian_privatizer(
          stddev=0.5, prng_key=jax.random.key(42)
      )

      def step(params, opt_state, noise_state):
        grad_fn = clipping.clipped_grad(
            _loss,
            l2_clip_norm=1.0,
            batch_argnums=1,
            pre_clipping_transform=optimizer.pre_clipping_transform(opt_state),
        )
        noisy_grads, noise_state = privatizer.update(
            grad_fn(params, _DATA), noise_state
        )
        updates, opt_state = optimizer.update(noisy_grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, noise_state

      if use_jit:
        step = jax.jit(step)
      params = _init_params()
      params, _, _ = step(
          params, optimizer.init(params), privatizer.init(params)
      )
      return params

    dp_sgd = optimizers.as_augmented_optimizer(optax.sgd(0.1))
    gc = geoclip.geoclip(optax.sgd(0.1), _RANK, _BATCH_SIZE)
    chex.assert_trees_all_close(train(gc), train(dp_sgd), atol=1e-6)

  def test_statistics_update_matches_algorithm_1(self):
    """One update from init: running mean, covariance and top eigenpairs."""
    beta1, beta2 = 0.9, 0.95
    optimizer = geoclip.geoclip(
        optax.sgd(0.1), _RANK, _BATCH_SIZE, beta1=beta1, beta2=beta2, h2=1e6
    )
    params = _init_params()
    state = optimizer.init(params)
    noisy_sum = {'w': jnp.array([2.0, -1.0, 0.5]), 'b': jnp.array(4.0)}
    _, state = optimizer.update(noisy_sum, state, params)

    # count == 0, so the inverse is the identity; the batch average is
    # noisy_sum / B.
    avg_grad, _ = ravel_pytree(noisy_sum)
    avg_grad = avg_grad / _BATCH_SIZE
    mean = (1 - beta1) * avg_grad
    centered = avg_grad - mean
    cov = beta2 * jnp.eye(4) + (1 - beta2) * _BATCH_SIZE * jnp.outer(
        centered, centered
    )
    chex.assert_trees_all_close(ravel_pytree(state.mean)[0], mean, atol=1e-6)
    chex.assert_trees_all_close(state.cov, cov, atol=1e-6)
    self.assertEqual(int(state.count), 1)
    # eigvecs are orthonormal top-rank eigenvectors of cov.
    u, lam = state.eigvecs, state.eigvals
    chex.assert_trees_all_close(u.T @ u, jnp.eye(_RANK), atol=1e-5)
    chex.assert_trees_all_close(cov @ u, u * lam, atol=1e-4)
    chex.assert_trees_all_close(lam, jnp.linalg.eigvalsh(cov)[-_RANK:])

  def test_eigenvalues_are_clamped(self):
    """Eigenvalues stay in [h1, h2], so the next transform stays finite."""
    optimizer = geoclip.geoclip(optax.sgd(0.1), 4, _BATCH_SIZE, h1=0.5, h2=2.0)
    params = _init_params()
    huge = {'w': jnp.array([1e3, -1e3, 5e2]), 'b': jnp.array(1e3)}
    _, state = optimizer.update(huge, optimizer.init(params), params)
    eigvals = state.eigvals
    self.assertTrue(bool(jnp.all((eigvals >= 0.5) & (eigvals <= 2.0))))
    transformed = optimizer.pre_clipping_transform(state)(huge)
    chex.assert_tree_all_finite(transformed)

  def test_clipping_with_count_positive(self):
    """With count > 0, clipping happens on the transformed gradients."""
    params = _init_params()
    optimizer = geoclip.geoclip(optax.sgd(0.1), 4, _BATCH_SIZE)
    state = optimizer.init(params)
    for scale in (1.0, 3.0):  # Two updates move the state off the identity.
      grads = jax.tree.map(lambda x, s=scale: s * jnp.ones_like(x), params)
      _, state = optimizer.update(grads, state, params)
    self.assertEqual(int(state.count), 2)

    transform = optimizer.pre_clipping_transform(state)
    clipped_sum = clipping.clipped_grad(
        _loss,
        l2_clip_norm=1.0,
        batch_argnums=1,
        pre_clipping_transform=transform,
    )(params, _DATA)

    def clip_one(row):
      g = jax.grad(_loss)(params, row[None])
      flat, unravel = ravel_pytree(transform(g))
      return unravel(flat / jnp.maximum(1.0, jnp.linalg.norm(flat)))

    expected = jax.tree.map(lambda x: x.sum(0), jax.vmap(clip_one)(_DATA))
    chex.assert_trees_all_close(clipped_sum, expected, atol=1e-5)

  def test_normalize_by_matches_sum(self):
    """Receiving the batch average with normalize_by=B equals the sum."""
    params = _init_params()
    noisy_sum = {'w': jnp.array([2.0, -1.0, 0.5]), 'b': jnp.array(4.0)}
    noisy_avg = optax.tree.scale(1.0 / _BATCH_SIZE, noisy_sum)
    summed = geoclip.geoclip(optax.sgd(1.0), _RANK, _BATCH_SIZE)
    averaged = geoclip.geoclip(
        optax.sgd(1.0), _RANK, _BATCH_SIZE, normalize_by=_BATCH_SIZE
    )
    s_state = _random_state(summed, params, jax.random.key(5))
    a_state = s_state  # Same statistics for both.
    s_updates, s_state = summed.update(noisy_sum, s_state, params)
    a_updates, a_state = averaged.update(noisy_avg, a_state, params)
    chex.assert_trees_all_close(
        s_updates, optax.tree.scale(_BATCH_SIZE, a_updates), atol=1e-5
    )
    chex.assert_trees_all_close(s_state.mean, a_state.mean, atol=1e-6)
    chex.assert_trees_all_close(s_state.cov, a_state.cov, atol=1e-5)

  def test_round_trip_recovers_gradients_in_span(self):
    """Transform, sum, invert: exact when each g - a is in span(U)."""
    params = _init_params()
    optimizer = geoclip.geoclip(optax.sgd(1.0), _RANK, _BATCH_SIZE)
    state = _random_state(optimizer, params, jax.random.key(1))
    flat_mean, unravel = ravel_pytree(state.mean)
    coeffs = jax.random.normal(jax.random.key(2), (_BATCH_SIZE, _RANK))
    grads = jax.vmap(unravel)(flat_mean + coeffs @ state.eigvecs.T)

    transformed = jax.vmap(optimizer.pre_clipping_transform(state))(grads)
    summed = jax.tree.map(lambda x: x.sum(0), transformed)
    expected = jax.tree.map(lambda x: x.sum(0), grads)

    recovered = optimizer.pre_clipping_transform(state, inverse=True)(summed)
    chex.assert_trees_all_close(recovered, expected, atol=1e-5)
    # update applies the same inverse; SGD with lr 1 returns its negation.
    updates, _ = optimizer.update(summed, state, params)
    chex.assert_trees_all_close(
        updates, optax.tree.scale(-1.0, expected), atol=1e-5
    )

  def test_component_outside_span_is_dropped(self):
    """Paper-faithful default: the inverse returns a + U U^T (g - a)."""
    params = _init_params()
    optimizer = geoclip.geoclip(optax.sgd(1.0), _RANK, expected_batch_size=1)
    state = _random_state(optimizer, params, jax.random.key(3))
    flat_mean, unravel = ravel_pytree(state.mean)
    g = jax.random.normal(jax.random.key(4), flat_mean.shape)

    transformed = optimizer.pre_clipping_transform(state)(unravel(g))
    recovered = optimizer.pre_clipping_transform(state, inverse=True)(
        transformed
    )
    u = state.eigvecs
    expected = flat_mean + u @ (u.T @ (g - flat_mean))
    chex.assert_trees_all_close(recovered, unravel(expected), atol=1e-5)

  def test_rank_larger_than_num_params_raises(self):
    optimizer = geoclip.geoclip(optax.sgd(0.1), rank=5, expected_batch_size=4)
    with self.assertRaisesRegex(ValueError, 'rank'):
      optimizer.init(_init_params())

  @parameterized.parameters(
      dict(kwargs=dict(expected_batch_size=0), match='expected_batch_size'),
      dict(kwargs=dict(normalize_by=0.0), match='normalize_by'),
      dict(kwargs=dict(beta1=1.0), match='beta1'),
      dict(kwargs=dict(beta2=-0.1), match='beta2'),
      dict(kwargs=dict(h1=0.0), match='h1'),
      dict(kwargs=dict(h1=2.0, h2=1.0), match='h1'),
  )
  def test_invalid_hyperparameters_raise(self, kwargs, match):
    args = dict(rank=_RANK, expected_batch_size=_BATCH_SIZE) | kwargs
    with self.assertRaisesRegex(ValueError, match):
      geoclip.geoclip(optax.sgd(0.1), **args)


if __name__ == '__main__':
  absltest.main()
