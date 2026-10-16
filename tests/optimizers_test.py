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
import jax.numpy as jnp
from jax_privacy import clipping
from jax_privacy import optimizers
import optax


def _simple_loss(params, x):
  """Per-example quadratic loss: 0.5 * mean((params - x)^2)."""
  return 0.5 * jnp.mean((params - x) ** 2)


def _init_params():
  """Simple 1D params for tests."""
  return jnp.array([1.0, 2.0, 3.0])


class FindAdaptiveStateTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(optimizer=optax.adam(1.0)),
      dict(optimizer=optax.adamw(1.0)),
      dict(optimizer=optax.rmsprop(1.0)),
      dict(optimizer=optax.adagrad(1.0)),
      dict(optimizer=optax.amsgrad(1.0)),
  )
  def test_finds_adaptive_state(self, optimizer):
    """Should extract nu from a ScaleByRmsState."""
    params = {'w': jnp.ones(3)}
    state = optimizer.init(params)
    result = optimizers._find_adaptive_state(state)
    chex.assert_trees_all_equal_shapes_and_dtypes(result, params)

  def test_raises_on_sgd_state(self):
    """Should raise ValueError for non-adaptive optimizer (SGD)."""
    opt = optax.sgd(1e-3)
    params = _init_params()
    state = opt.init(params)
    with self.assertRaisesRegex(ValueError, 'adaptive optimizer'):
      optimizers._find_adaptive_state(state)

  def test_raises_on_empty_tuple(self):
    """Should raise ValueError for an empty tuple."""
    with self.assertRaisesRegex(ValueError, 'adaptive optimizer'):
      optimizers._find_adaptive_state(())


class ScaleThenPrivatizeTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(opt_fn=lambda: optax.adam(1e-3)),
      dict(opt_fn=lambda: optax.adamw(1e-3)),
  )
  def test_init_matches_base_optimizer(self, opt_fn):
    """Init should produce the same state as the base optimizer."""
    base_opt = opt_fn()
    augmented = optimizers.scale_then_privatize(base_opt)
    params = _init_params()
    base_state = base_opt.init(params)
    aug_state = augmented.init(params)
    chex.assert_trees_all_close(aug_state, base_state)

  def test_scaling_is_non_isotropic(self):
    """The scaling should be different across coordinates with different nu."""
    # Use a custom preconditioner to directly control nu and verify
    # that scaling is non-isotropic (different per coordinate).
    base_opt = optax.adam(1e-3)
    custom_nu = jnp.array([1.0, 10000.0])
    augmented = optimizers.scale_then_privatize(
        base_opt, extract_preconditioner_from_state_fn=lambda state: custom_nu
    )
    params = jnp.array([0.0, 0.0])
    state = augmented.init(params)

    pct = augmented.pre_clipping_transform(state)
    g = jnp.array([1.0, 1.0])
    scaled = pct(g)

    # s = 1/(sqrt(nu) + eps), so coordinate with larger nu gets scaled more.
    # scaled[0] = 1/(sqrt(1)+eps) ≈ 0.5, scaled[1] = 1/(sqrt(10000)+eps) ≈ 0.01
    self.assertGreater(float(scaled[0]), float(scaled[1]))

  def test_update_applies_inverse(self):
    """update should apply inverse transform before base optimizer update."""
    base_opt = optax.adam(1e-3)
    augmented = optimizers.scale_then_privatize(base_opt)
    params = _init_params()
    state = augmented.init(params)

    # Run a step with the base optimizer to populate nu.
    grad = jnp.array([0.1, 0.2, 0.3])
    _, state = base_opt.update(grad, state, params)

    # Now, if we forward-transform a gradient and pass it to augmented.update,
    # the update should produce the same result as calling base_opt.update
    # with the original gradient (since update applies inverse internally).
    pct = augmented.pre_clipping_transform(state)
    scaled_grad = pct(grad)
    aug_updates, _ = augmented.update(scaled_grad, state, params)
    base_updates, _ = base_opt.update(grad, state, params)
    chex.assert_trees_all_close(aug_updates, base_updates, atol=1e-6)

  def test_custom_preconditioner_fn(self):
    """Should use a custom preconditioner function when provided."""
    base_opt = optax.sgd(1e-3)  # SGD has no adaptive state.
    params = _init_params()

    # Custom extractor that returns a constant nu.
    custom_nu = jnp.array([4.0, 9.0, 16.0])
    custom_fn = lambda state: custom_nu  # pylint: disable=unnecessary-lambda

    augmented = optimizers.scale_then_privatize(
        base_opt, extract_preconditioner_from_state_fn=custom_fn
    )
    state = augmented.init(params)

    pct = augmented.pre_clipping_transform(state)
    g = jnp.array([1.0, 1.0, 1.0])
    scaled = pct(g)

    # s = 1 / (sqrt(nu) + eps) ≈ [1/2, 1/3, 1/4] for eps ≈ 0.
    expected = g / (jnp.sqrt(custom_nu) + 1e-8)
    chex.assert_trees_all_close(scaled, expected, atol=1e-6)

  def test_raises_on_sgd_without_custom_fn(self):
    """Should raise ValueError when using SGD without custom extractor."""
    base_opt = optax.sgd(1e-3)
    augmented = optimizers.scale_then_privatize(base_opt)
    params = _init_params()
    state = augmented.init(params)

    with self.assertRaisesRegex(ValueError, 'adaptive optimizer'):
      augmented.pre_clipping_transform(state)

  def test_eps_root_affects_scaling(self):
    """eps_root should affect the scaling computation."""
    base_opt = optax.adam(1e-3, b1=0.0, b2=1.0)
    params = jnp.array([0.0])
    state = base_opt.init(params)
    grad = jnp.array([1.0])
    _, state = base_opt.update(grad, state, params)

    aug_no_eps_root = optimizers.scale_then_privatize(base_opt, eps_root=0.0)
    aug_with_eps_root = optimizers.scale_then_privatize(base_opt, eps_root=1.0)

    pct_no = aug_no_eps_root.pre_clipping_transform(state)
    pct_with = aug_with_eps_root.pre_clipping_transform(state)

    g = jnp.array([1.0])
    scaled_no = pct_no(g)
    scaled_with = pct_with(g)

    # With eps_root > 0, the scaling should be smaller (more regularized).
    self.assertGreater(float(scaled_no[0]), float(scaled_with[0]))

  def test_jit_compatible(self):
    """The full pipeline should work under jax.jit."""
    base_opt = optax.adamw(1e-3)
    augmented = optimizers.scale_then_privatize(base_opt)
    params = _init_params()
    state = augmented.init(params)

    @jax.jit
    def step(params, state, grad):
      pct = augmented.pre_clipping_transform(state)
      scaled = pct(grad)
      updates, new_state = augmented.update(scaled, state, params)
      new_params = optax.apply_updates(params, updates)
      return new_params, new_state

    grad = jnp.array([0.1, 0.2, 0.3])
    new_params, _ = step(params, state, grad)
    # Check it produced valid output.
    chex.assert_shape(new_params, params.shape)
    self.assertFalse(jnp.any(jnp.isnan(new_params)))


class ScaleThenPrivatizeE2ETest(parameterized.TestCase):
  """End-to-end test integrating scale_then_privatize with clipped_grad."""

  @parameterized.parameters(dict(use_jit=False), dict(use_jit=True))
  def test_augmented_matches_base_with_no_clipping(self, use_jit):
    """With l2_clip_norm=inf, augmented training should match base training."""
    params = jnp.array([3.0, 1.0, -2.0])
    data = jnp.array([[1.0, 0.0, 0.5], [4.0, -1.0, 2.0], [7.0, 3.0, -1.0]])
    num_steps = 5

    base_optimizer = optax.adamw(1e-2)
    scale_first_optimizer = optimizers.scale_then_privatize(base_optimizer)

    def base_train_step(params, opt_state, data):
      grad_fn = clipping.clipped_grad(
          _simple_loss,
          argnums=0,
          batch_argnums=1,
          l2_clip_norm=jnp.inf,
      )
      grads = grad_fn(params, data)
      updates, new_state = base_optimizer.update(grads, opt_state, params)
      new_params = optax.apply_updates(params, updates)
      return new_params, new_state

    def augmented_train_step(params, opt_state, data):
      transform_fn = scale_first_optimizer.pre_clipping_transform(opt_state)
      grad_fn = clipping.clipped_grad(
          _simple_loss,
          argnums=0,
          batch_argnums=1,
          l2_clip_norm=jnp.inf,
          pre_clipping_transform=transform_fn,
      )
      grad = grad_fn(params, data)
      updates, new_state = scale_first_optimizer.update(grad, opt_state, params)
      new_params = optax.apply_updates(params, updates)
      return new_params, new_state

    if use_jit:
      base_train_step = jax.jit(base_train_step)
      augmented_train_step = jax.jit(augmented_train_step)

    base_params = params
    base_state = base_optimizer.init(params)
    aug_params = params
    aug_state = scale_first_optimizer.init(params)

    for _ in range(num_steps):
      base_params, base_state = base_train_step(base_params, base_state, data)
      aug_params, aug_state = augmented_train_step(aug_params, aug_state, data)

    chex.assert_trees_all_close(aug_params, base_params, atol=1e-5)


class DP2RMSPropTest(parameterized.TestCase):

  def test_init_creates_zero_state_matching_nested_params(self):
    params = {
        'layer': {'w': jnp.ones((2, 3), dtype=jnp.float32)},
        'bias': jnp.ones(3, dtype=jnp.float16),
    }
    optimizer = optimizers.dp2_rmsprop(0.1, 0.01, 2, 1.0, 0.5)

    state = optimizer.init(params)

    self.assertEqual(int(state.count), 0)
    chex.assert_trees_all_equal_shapes_and_dtypes(
        state.grad_accumulator, params
    )
    chex.assert_trees_all_equal_shapes_and_dtypes(state.nu, params)
    chex.assert_trees_all_close(
        state.grad_accumulator, jax.tree.map(jnp.zeros_like, params)
    )
    chex.assert_trees_all_close(state.nu, jax.tree.map(jnp.zeros_like, params))

  def test_alternates_equal_length_sgd_and_rmsprop_phases(self):
    optimizer = optimizers.dp2_rmsprop(
        0.1,
        0.01,
        interval=2,
        sgd_clip_norm=2.0,
        rmsprop_clip_norm=1.0,
        eps=1.0,
        rescale_to_unit_norm=False,
    )
    grad = jnp.ones(2)
    state = optimizer.init(grad)
    is_identity = []

    for _ in range(6):
      transformed = optimizer.pre_clipping_transform(state)(grad)
      is_identity.append(bool(jnp.allclose(transformed, grad)))
      _, state = optimizer.update(jnp.zeros_like(grad), state)

    self.assertSequenceEqual(
        is_identity, [True, True, False, False, True, True]
    )

  def test_updates_delayed_moment_and_resets_accumulator(self):
    optimizer = optimizers.dp2_rmsprop(
        0.1,
        0.01,
        interval=2,
        sgd_clip_norm=1.0,
        rmsprop_clip_norm=1.0,
        decay=0.5,
        rescale_to_unit_norm=False,
    )
    state = optimizer.init(jnp.zeros(1))

    _, state = optimizer.update(jnp.array([2.0]), state)
    chex.assert_trees_all_close(state.grad_accumulator, jnp.array([2.0]))
    chex.assert_trees_all_close(state.nu, jnp.array([0.0]))

    _, state = optimizer.update(jnp.array([4.0]), state)
    chex.assert_trees_all_close(state.grad_accumulator, jnp.array([0.0]))
    # decay * 0 + (1 - decay) * ((2 + 4) / 2)^2 = 4.5.
    chex.assert_trees_all_close(state.nu, jnp.array([4.5]))

    _, state = optimizer.update(jnp.array([7.0]), state)
    chex.assert_trees_all_close(state.nu, jnp.array([4.5]))
    _, state = optimizer.update(jnp.array([9.0]), state)
    chex.assert_trees_all_close(state.nu, jnp.array([4.5]))

  @parameterized.parameters(False, True)
  def test_two_clip_norm_transform_is_equivalent(self, rescale_to_unit_norm):
    sgd_clip_norm = 2.0
    rmsprop_clip_norm = 1.0
    eps = 0.25
    optimizer = optimizers.dp2_rmsprop(
        0.1,
        0.01,
        interval=2,
        sgd_clip_norm=sgd_clip_norm,
        rmsprop_clip_norm=rmsprop_clip_norm,
        eps=eps,
        rescale_to_unit_norm=rescale_to_unit_norm,
    )
    grad = jnp.array([6.0, 8.0])
    state = optimizer.init(grad)._replace(
        count=jnp.array(2, dtype=jnp.int32), nu=jnp.array([1.0, 9.0])
    )

    transformed = optimizer.pre_clipping_transform(state)(grad)
    clipped_with_sgd_norm, _ = clipping.clip_pytree(
        transformed,
        sgd_clip_norm,
        rescale_to_unit_norm=rescale_to_unit_norm,
    )
    decode_scale = (
        rmsprop_clip_norm
        if rescale_to_unit_norm
        else rmsprop_clip_norm / sgd_clip_norm
    )
    actual = clipped_with_sgd_norm * decode_scale

    preconditioned = grad / (jnp.sqrt(state.nu) + eps)
    expected, _ = clipping.clip_pytree(preconditioned, rmsprop_clip_norm)
    chex.assert_trees_all_close(actual, expected, atol=1e-6)

  def test_uses_phase_learning_rates_without_undoing_preconditioner(self):
    optimizer = optimizers.dp2_rmsprop(
        sgd_learning_rate=0.1,
        rmsprop_learning_rate=0.01,
        interval=1,
        sgd_clip_norm=2.0,
        rmsprop_clip_norm=1.0,
        decay=0.0,
        eps=1e-6,
        rescale_to_unit_norm=False,
    )
    params = jnp.zeros(1)
    state = optimizer.init(params)

    sgd_updates, state = optimizer.update(jnp.array([2.0]), state, params)
    chex.assert_trees_all_close(sgd_updates, jnp.array([-0.2]))
    chex.assert_trees_all_close(state.nu, jnp.array([4.0]))

    transformed = optimizer.pre_clipping_transform(state)(jnp.array([6.0]))
    rmsprop_updates, _ = optimizer.update(transformed, state, params)
    # The transformed gradient decodes to 6 / (sqrt(4) + eps), not back to 6.
    chex.assert_trees_all_close(rmsprop_updates, jnp.array([-0.03]), atol=1e-6)

  def test_jit_across_phase_boundaries(self):
    optimizer = optimizers.dp2_rmsprop(
        0.1, 0.01, 2, 2.0, 1.0, rescale_to_unit_norm=False
    )
    state = optimizer.init(jnp.zeros(2))

    @jax.jit
    def run_steps(state, grads):
      def step(state, grad):
        transformed = optimizer.pre_clipping_transform(state)(grad)
        updates, state = optimizer.update(transformed, state)
        return state, updates

      return jax.lax.scan(step, state, grads)

    state, updates = run_steps(state, jnp.ones((5, 2)))

    self.assertEqual(int(state.count), 5)
    chex.assert_shape(updates, (5, 2))
    self.assertTrue(bool(jnp.all(jnp.isfinite(updates))))

  @parameterized.parameters(False, True)
  def test_default_inner_optimizer_preserves_updates(
      self, rescale_to_unit_norm
  ):
    optimizer = optimizers.dp2_rmsprop(
        0.1,
        0.02,
        2,
        2.0,
        1.0,
        rescale_to_unit_norm=rescale_to_unit_norm,
    )
    params = {'layer': {'w': jnp.zeros(2)}, 'bias': jnp.zeros(1)}
    gradients = jax.tree.map(jnp.ones_like, params)
    state = optimizer.init(params)

    for step in range(6):
      use_sgd = step % 4 < 2
      transformed = optimizer.pre_clipping_transform(state)(gradients)
      clipped, _ = clipping.clip_pytree(
          transformed, 2.0, rescale_to_unit_norm=rescale_to_unit_norm
      )
      noisy_grad = jax.tree.map(lambda g: g + 0.1, clipped)
      clip_norm = 2.0 if use_sgd else 1.0
      decode_scale = clip_norm if rescale_to_unit_norm else clip_norm / 2.0
      learning_rate = 0.1 if use_sgd else 0.02
      expected = optax.tree.scale(-learning_rate * decode_scale, noisy_grad)

      updates, state = optimizer.update(noisy_grad, state, params)

      chex.assert_trees_all_close(updates, expected, atol=1e-6)
      chex.assert_trees_all_equal_shapes_and_dtypes(updates, params)

  @parameterized.parameters(False, True)
  def test_inner_momentum_and_schedule_persist_across_phases(self, nesterov):
    inner_optimizer = optax.sgd(
        lambda count: 1.0 / (count + 1), momentum=0.5, nesterov=nesterov
    )
    optimizer = optimizers.dp2_rmsprop(
        0.1,
        0.02,
        2,
        1.0,
        1.0,
        decay=0.5,
        inner_optimizer=inner_optimizer,
    )
    state = optimizer.init(jnp.zeros(1))
    chex.assert_trees_all_close(
        state.inner_state, inner_optimizer.init(jnp.zeros(1))
    )
    update_fn = jax.jit(optimizer.update)
    trace = 0.0

    for step, grad in enumerate([2.0, 4.0, 3.0, 5.0, 6.0, 8.0]):
      trace = grad + 0.5 * trace
      direction = grad + 0.5 * trace if nesterov else trace
      learning_rate = 0.1 if step % 4 < 2 else 0.02
      expected = -learning_rate * direction / (step + 1)

      updates, state = update_fn(jnp.array([grad]), state)

      chex.assert_trees_all_close(updates, jnp.array([expected]), atol=1e-6)

    # The delayed moment uses SGD gradients (averages 3 and 7), before momentum
    # or learning rates: 0.5 * (0.5 * 3^2) + 0.5 * 7^2 = 26.75.
    chex.assert_trees_all_close(state.nu, jnp.array([26.75]))
    chex.assert_trees_all_close(state.grad_accumulator, jnp.zeros(1))
    self.assertEqual(int(state.count), 6)

  def test_inner_optimizer_receives_params_for_weight_decay(self):
    inner_optimizer = optax.chain(
        optax.add_decayed_weights(0.2), optax.sgd(1.0)
    )
    optimizer = optimizers.dp2_rmsprop(
        0.1, 0.02, 1, 1.0, 1.0, inner_optimizer=inner_optimizer
    )
    params = {'w': jnp.array([2.0, 3.0])}
    gradients = {'w': jnp.array([0.5, 1.0])}
    state = optimizer.init(params)
    update_fn = jax.jit(optimizer.update)

    for learning_rate in [0.1, 0.02]:
      expected = {'w': -learning_rate * (gradients['w'] + 0.2 * params['w'])}

      updates, state = update_fn(gradients, state, params)

      chex.assert_trees_all_close(updates, expected, atol=1e-6)
      params = optax.apply_updates(params, updates)

  def test_inner_optimizer_receives_extra_arguments(self):
    def inner_update(updates, state, params=None, *, gain):
      del params
      return jax.tree.map(lambda g: -gain * g, updates), state + 1

    inner_optimizer = optax.GradientTransformationExtraArgs(
        init=lambda _: jnp.zeros([], dtype=jnp.int32),
        update=inner_update,
    )
    optimizer = optimizers.dp2_rmsprop(
        0.1,
        0.02,
        1,
        1.0,
        1.0,
        decay=0.0,
        inner_optimizer=inner_optimizer,
    )
    state = optimizer.init(jnp.zeros(1))
    update_fn = jax.jit(optimizer.update)

    for step, learning_rate in enumerate([0.1, 0.02]):
      updates, state = update_fn(jnp.ones(1), state, gain=jnp.array(3.0))

      chex.assert_trees_all_close(
          updates, jnp.array([-learning_rate * 3.0]), atol=1e-6
      )
      chex.assert_trees_all_close(state.nu, jnp.ones(1))
      self.assertEqual(int(state.inner_state), step + 1)

  def test_plain_inner_transformation_ignores_unused_extra_arguments(self):
    optimizer = optimizers.dp2_rmsprop(
        0.1, 0.02, 1, 1.0, 1.0, inner_optimizer=optax.scale(-1.0)
    )
    state = optimizer.init(jnp.zeros(1))

    updates, _ = optimizer.update(jnp.ones(1), state, unused=True)

    chex.assert_trees_all_close(updates, jnp.array([-0.1]))

  @parameterized.named_parameters(
      ('interval_type', {'interval': 1.5}, TypeError, 'Python integer'),
      ('zero_interval', {'interval': 0}, ValueError, 'positive'),
      (
          'negative_sgd_lr',
          {'sgd_learning_rate': -0.1},
          ValueError,
          'nonnegative',
      ),
      (
          'negative_rmsprop_lr',
          {'rmsprop_learning_rate': -0.1},
          ValueError,
          'nonnegative',
      ),
      ('zero_sgd_clip', {'sgd_clip_norm': 0.0}, ValueError, 'positive'),
      ('zero_rmsprop_clip', {'rmsprop_clip_norm': 0.0}, ValueError, 'positive'),
      ('zero_eps', {'eps': 0.0}, ValueError, 'positive'),
      ('negative_decay', {'decay': -0.1}, ValueError, r'\[0, 1\)'),
      ('unit_decay', {'decay': 1.0}, ValueError, r'\[0, 1\)'),
  )
  def test_rejects_invalid_hyperparameters(self, overrides, error_type, regex):
    kwargs = dict(
        sgd_learning_rate=0.1,
        rmsprop_learning_rate=0.01,
        interval=2,
        sgd_clip_norm=1.0,
        rmsprop_clip_norm=0.5,
    )
    kwargs.update(overrides)

    with self.assertRaisesRegex(error_type, regex):
      optimizers.dp2_rmsprop(**kwargs)


class MiscellaneousTest(parameterized.TestCase):

  def test_as_augmented_optimizer(self):
    """Should convert a GradientTransformation to an AugmentedOptimizer."""
    optimizer = optax.adam(1e-3)
    augmented = optimizers.as_augmented_optimizer(optimizer)
    self.assertIsInstance(augmented, optimizers.AugmentedGradientTransformation)
    augmented_v2 = optimizers.as_augmented_optimizer(augmented)
    self.assertIs(augmented, augmented_v2)


if __name__ == '__main__':
  absltest.main()
