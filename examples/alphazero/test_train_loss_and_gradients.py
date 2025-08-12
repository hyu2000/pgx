#!/usr/bin/env python3
"""
Test script to verify loss calculation and gradient updates from train.py work correctly.
This ensures the Haiku to Equinox migration is working for the core training loop.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import pgx
import pytest
from typing import NamedTuple

# Import from train.py
import sys
import os
sys.path.append(os.path.dirname(__file__))

from network import AZNet
from examples.alphazero.config import Config


class Sample(NamedTuple):
    obs: jnp.ndarray
    policy_tgt: jnp.ndarray
    value_tgt: jnp.ndarray
    mask: jnp.ndarray


def forward_fn(model, state, x):
    """Forward pass with Equinox model - from train.py"""
    return model(x, state)


def loss_fn(model, bn_state, samples: Sample):
    """Loss function from train.py"""
    (logits, value), _ = forward_fn(model, bn_state, samples.obs)

    policy_loss = optax.softmax_cross_entropy(logits, samples.policy_tgt)
    policy_loss = jnp.mean(policy_loss)

    value_loss = optax.l2_loss(value, samples.value_tgt)
    value_loss = jnp.mean(value_loss * samples.mask)

    return policy_loss + value_loss, (policy_loss, value_loss)


def create_model(key, input_channels, spatial_size, env):
    """Create model helper from train.py"""
    config = Config(
        env_id="go_5x5C2",
        num_channels=32,  # Smaller for testing
        num_layers=2,     # Smaller for testing
        resnet_v2=True
    )
    return AZNet(
        num_actions=env.num_actions,
        input_channels=input_channels,
        num_channels=config.num_channels,
        num_blocks=config.num_layers,
        resnet_v2=config.resnet_v2,
        spatial_size=spatial_size,
        key=key
    )


def test_loss_calculation():
    """Test that loss calculation works correctly."""
    # Setup environment and model
    env = pgx.make("go_5x5C2")
    
    # Create dummy batch data
    batch_size = 4
    dummy_state = env.init(jax.random.PRNGKey(0))
    dummy_obs = jax.vmap(lambda _: dummy_state.observation)(jnp.arange(batch_size))
    
    input_channels = dummy_obs.shape[-1]
    spatial_size = dummy_obs.shape[1] * dummy_obs.shape[2]
    
    # Create model
    model_key = jax.random.PRNGKey(42)
    model, bn_state = eqx.nn.make_with_state(
        lambda key: create_model(key, input_channels, spatial_size, env)
    )(model_key)
    
    # Create dummy training sample
    dummy_policy = jax.nn.softmax(jax.random.normal(jax.random.PRNGKey(1), (batch_size, env.num_actions)))
    dummy_values = jax.random.uniform(jax.random.PRNGKey(2), (batch_size,), minval=-1, maxval=1)
    dummy_mask = jnp.ones((batch_size,))
    
    sample = Sample(
        obs=dummy_obs,
        policy_tgt=dummy_policy,
        value_tgt=dummy_values,
        mask=dummy_mask
    )
    
    # Test loss calculation
    total_loss, (policy_loss, value_loss) = loss_fn(model, bn_state, sample)
    
    # Assertions
    assert jnp.isfinite(total_loss), "Total loss should be finite"
    assert jnp.isfinite(policy_loss), "Policy loss should be finite" 
    assert jnp.isfinite(value_loss), "Value loss should be finite"
    assert total_loss > 0, "Total loss should be positive"
    assert policy_loss > 0, "Policy loss should be positive"
    assert value_loss >= 0, "Value loss should be non-negative"
    
    print(f"✓ Loss calculation works: total={total_loss:.4f}, policy={policy_loss:.4f}, value={value_loss:.4f}")


def test_gradient_computation():
    """Test that gradients can be computed and applied."""
    # Setup
    env = pgx.make("go_5x5C2")
    batch_size = 4
    dummy_state = env.init(jax.random.PRNGKey(0))
    dummy_obs = jax.vmap(lambda _: dummy_state.observation)(jnp.arange(batch_size))
    
    input_channels = dummy_obs.shape[-1]
    spatial_size = dummy_obs.shape[1] * dummy_obs.shape[2]
    
    # Create model and optimizer
    model_key = jax.random.PRNGKey(42)
    model, bn_state = eqx.nn.make_with_state(
        lambda key: create_model(key, input_channels, spatial_size, env)
    )(model_key)
    
    optimizer = optax.adam(0.001)
    # Filter model to only trainable parameters (exclude axis_name strings)
    trainable_model = eqx.filter(model, eqx.is_array)
    opt_state = optimizer.init(trainable_model)
    
    # Create sample
    dummy_policy = jax.nn.softmax(jax.random.normal(jax.random.PRNGKey(1), (batch_size, env.num_actions)))
    dummy_values = jax.random.uniform(jax.random.PRNGKey(2), (batch_size,), minval=-1, maxval=1)
    dummy_mask = jnp.ones((batch_size,))
    
    sample = Sample(
        obs=dummy_obs,
        policy_tgt=dummy_policy,
        value_tgt=dummy_values,
        mask=dummy_mask
    )
    
    # Compute gradients with respect to trainable parameters only
    def loss_with_trainable_only(trainable_params, bn_state, sample):
        # Reconstruct full model from trainable and non-trainable parts
        full_model = eqx.combine(trainable_params, eqx.filter(model, lambda x: not eqx.is_array(x)))
        return loss_fn(full_model, bn_state, sample)
    
    grads, (policy_loss, value_loss) = jax.grad(loss_with_trainable_only, has_aux=True)(trainable_model, bn_state, sample)
    
    # Check gradients are finite
    grad_tree_finite = jax.tree_util.tree_map(lambda x: jnp.all(jnp.isfinite(x)), grads)
    # Check if all gradients are finite (manual tree_all)
    grad_finite_flat = jax.tree_util.tree_leaves(grad_tree_finite)
    all_grads_finite = all(grad_finite_flat)
    assert all_grads_finite, "All gradients should be finite"
    
    # Apply gradients
    updates, new_opt_state = optimizer.update(grads, opt_state, trainable_model)
    new_trainable_model = optax.apply_updates(trainable_model, updates)
    # Reconstruct the full model
    new_model = eqx.combine(new_trainable_model, eqx.filter(model, lambda x: not eqx.is_array(x)))
    
    # Verify parameters changed
    def params_changed(old, new):
        return jax.tree_util.tree_map(lambda x, y: not jnp.allclose(x, y, atol=1e-8), old, new)
    
    param_changes = params_changed(trainable_model, new_trainable_model)
    # Check if any parameters changed (manual tree_any)
    changes_flat = jax.tree_util.tree_leaves(param_changes)
    some_params_changed = any(changes_flat)
    assert some_params_changed, "Some parameters should have changed after gradient update"
    
    print(f"✓ Gradient computation and update works: policy_loss={policy_loss:.4f}, value_loss={value_loss:.4f}")


def test_training_step():
    """Test a complete training step similar to train.py"""
    env = pgx.make("go_5x5C2")
    batch_size = 4
    
    # Setup model
    dummy_state = env.init(jax.random.PRNGKey(0))
    dummy_obs = jax.vmap(lambda _: dummy_state.observation)(jnp.arange(batch_size))
    
    input_channels = dummy_obs.shape[-1]
    spatial_size = dummy_obs.shape[1] * dummy_obs.shape[2]
    
    model_key = jax.random.PRNGKey(42)
    model, bn_state = eqx.nn.make_with_state(
        lambda key: create_model(key, input_channels, spatial_size, env)
    )(model_key)
    
    # Setup optimizer
    lr_schedule = optax.exponential_decay(
        init_value=0.01,
        transition_steps=100,
        decay_rate=0.5,
        staircase=True
    )
    optimizer = optax.adam(lr_schedule)
    trainable_model = eqx.filter(model, eqx.is_array)
    opt_state = optimizer.init(trainable_model)
    
    def train_step(model, bn_state, opt_state, sample):
        """Single training step"""
        trainable_params = eqx.filter(model, eqx.is_array)
        non_trainable = eqx.filter(model, lambda x: not eqx.is_array(x))
        
        def loss_with_trainable(params, bn_state, sample):
            full_model = eqx.combine(params, non_trainable)
            return loss_fn(full_model, bn_state, sample)
        
        grads, (policy_loss, value_loss) = jax.grad(loss_with_trainable, has_aux=True)(trainable_params, bn_state, sample)
        updates, new_opt_state = optimizer.update(grads, opt_state, trainable_params)
        new_trainable = optax.apply_updates(trainable_params, updates)
        new_model = eqx.combine(new_trainable, non_trainable)
        return new_model, bn_state, new_opt_state, policy_loss, value_loss
    
    # Create training sample
    dummy_policy = jax.nn.softmax(jax.random.normal(jax.random.PRNGKey(1), (batch_size, env.num_actions)))
    dummy_values = jax.random.uniform(jax.random.PRNGKey(2), (batch_size,), minval=-1, maxval=1)
    dummy_mask = jnp.ones((batch_size,))
    
    sample = Sample(
        obs=dummy_obs,
        policy_tgt=dummy_policy,
        value_tgt=dummy_values,
        mask=dummy_mask
    )
    
    # Run training step
    initial_loss, _ = loss_fn(model, bn_state, sample)
    new_model, new_bn_state, new_opt_state, policy_loss, value_loss = train_step(model, bn_state, opt_state, sample)
    
    # Verify training worked
    assert jnp.isfinite(policy_loss), "Policy loss should be finite"
    assert jnp.isfinite(value_loss), "Value loss should be finite"
    
    # Verify model parameters changed
    def model_changed(old, new):
        return jax.tree_util.tree_map(lambda x, y: not jnp.allclose(x, y, atol=1e-8), old, new)
    
    old_trainable = eqx.filter(model, eqx.is_array)
    new_trainable = eqx.filter(new_model, eqx.is_array)
    changes = model_changed(old_trainable, new_trainable)
    # Check if any parameters changed (manual tree_any)
    changes_flat = jax.tree_util.tree_leaves(changes)
    model_changed_any = any(changes_flat)
    assert model_changed_any, "Model should have changed after training step"
    
    print(f"✓ Complete training step works: initial_loss={initial_loss:.4f}, final policy_loss={policy_loss:.4f}, value_loss={value_loss:.4f}")


def test_forward_pass_batching():
    """Test that batched forward passes work correctly."""
    env = pgx.make("go_5x5C2")
    
    # Test different batch sizes
    for batch_size in [1, 2, 4, 8]:
        dummy_state = env.init(jax.random.PRNGKey(0))
        dummy_obs = jax.vmap(lambda _: dummy_state.observation)(jnp.arange(batch_size))
        
        input_channels = dummy_obs.shape[-1]
        spatial_size = dummy_obs.shape[1] * dummy_obs.shape[2]
        
        model_key = jax.random.PRNGKey(42)
        model, bn_state = eqx.nn.make_with_state(
            lambda key: create_model(key, input_channels, spatial_size, env)
        )(model_key)
        
        # Test forward pass
        (logits, values), _ = forward_fn(model, bn_state, dummy_obs)
        
        # Check shapes
        assert logits.shape == (batch_size, env.num_actions), f"Logits shape mismatch for batch_size={batch_size}"
        assert values.shape == (batch_size,), f"Values shape mismatch for batch_size={batch_size}"
        
        # Check all outputs are finite
        assert jnp.all(jnp.isfinite(logits)), f"All logits should be finite for batch_size={batch_size}"
        assert jnp.all(jnp.isfinite(values)), f"All values should be finite for batch_size={batch_size}"
        
    print("✓ Batched forward passes work correctly for various batch sizes")


def test_inference_mode():
    """Test that inference mode works as expected for evaluation."""
    env = pgx.make("go_5x5C2")
    batch_size = 2
    
    dummy_state = env.init(jax.random.PRNGKey(0))
    dummy_obs = jax.vmap(lambda _: dummy_state.observation)(jnp.arange(batch_size))
    
    input_channels = dummy_obs.shape[-1]
    spatial_size = dummy_obs.shape[1] * dummy_obs.shape[2]
    
    model_key = jax.random.PRNGKey(42)
    model, bn_state = eqx.nn.make_with_state(
        lambda key: create_model(key, input_channels, spatial_size, env)
    )(model_key)
    
    # Test training mode
    (train_logits, train_values), train_new_state = forward_fn(model, bn_state, dummy_obs)
    
    # Test inference mode
    inference_model = eqx.nn.inference_mode(model)
    (inf_logits, inf_values), inf_new_state = forward_fn(inference_model, bn_state, dummy_obs)
    
    # Outputs should be finite in both modes
    assert jnp.all(jnp.isfinite(train_logits)), "Training mode logits should be finite"
    assert jnp.all(jnp.isfinite(train_values)), "Training mode values should be finite"
    assert jnp.all(jnp.isfinite(inf_logits)), "Inference mode logits should be finite"
    assert jnp.all(jnp.isfinite(inf_values)), "Inference mode values should be finite"
    
    # Shapes should match
    assert train_logits.shape == inf_logits.shape, "Logits shapes should match between modes"
    assert train_values.shape == inf_values.shape, "Values shapes should match between modes"
    
    print("✓ Inference mode works correctly")


if __name__ == "__main__":
    print("=" * 80)
    print("Testing Train.py Loss Calculation and Gradient Updates")
    print("=" * 80)
    
    try:
        test_loss_calculation()
        test_gradient_computation()
        test_training_step()
        test_forward_pass_batching()
        test_inference_mode()
        
        print("\n" + "=" * 80)
        print("✅ All tests passed! Train.py migration is working correctly.")
        print("=" * 80)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise