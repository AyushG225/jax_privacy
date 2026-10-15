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
    chex.assert_shape(state.eigvecs, (4, _RANK))
    chex.assert_shape(state.eigvals, (_RANK,))

    grads = jax.tree.map(jnp.ones_like, params)
    transformed = optimizer.pre_clipping_transform(state)(grads)
    chex.assert_trees_all_equal_shapes_and_dtypes(transformed, params)
    updates, _ = optimizer.update(transformed, state, params)
    chex.assert_trees_all_equal_shapes_and_dtypes(updates, params)

  @parameterized.parameters(dict(use_jit=False), dict(use_jit=True))
  def test_identity_state_matches_dp_sgd(self, use_jit):
    """From init (count == 0), noisy clipped training matches plain DP-SGD."""
    data = jnp.array([
        [1.0, 2.0, 0.0, 5.0],
        [-3.0, 1.0, 2.0, 0.0],
        [0.5, 0.5, 4.0, -2.0],
        [2.0, -1.0, 1.0, 3.0],
    ])

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
            grad_fn(params, data), noise_state
        )
        updates, opt_state = optimizer.update(noisy_grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, noise_state

      if use_jit:
        step = jax.jit(step)
      params = _init_params()
      opt_state = optimizer.init(params)
      noise_state = privatizer.init(params)
      for _ in range(3):
        params, opt_state, noise_state = step(params, opt_state, noise_state)
      return params

    dp_sgd = optimizers.as_augmented_optimizer(optax.sgd(0.1))
    gc = geoclip.geoclip(optax.sgd(0.1), _RANK, _BATCH_SIZE)
    chex.assert_trees_all_close(train(gc), train(dp_sgd), atol=1e-6)

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


if __name__ == '__main__':
  absltest.main()
