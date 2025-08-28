"""Equinox implementation of AlphaZero network architecture"""
import pickle

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional, Tuple


class BlockV1(eqx.Module):
    """ResNet v1 block - activation before batch norm"""
    conv1: eqx.nn.Conv2d
    bn1: eqx.nn.BatchNorm
    conv2: eqx.nn.Conv2d
    bn2: eqx.nn.BatchNorm

    def __init__(self, num_channels: int, key: Optional[jax.random.PRNGKey] = None):
        if key is None:
            key = jax.random.PRNGKey(0)
        key1, key2 = jax.random.split(key, 2)
        
        self.conv1 = eqx.nn.Conv2d(num_channels, num_channels, 3, padding=1, key=key1)
        self.bn1 = eqx.nn.BatchNorm(num_channels, axis_name="batch", mode="batch")
        self.conv2 = eqx.nn.Conv2d(num_channels, num_channels, 3, padding=1, key=key2)
        self.bn2 = eqx.nn.BatchNorm(num_channels, axis_name="batch", mode="batch")

    def __call__(self, x: jnp.ndarray, state: eqx.nn.State) -> tuple[jnp.ndarray, eqx.nn.State]:
        identity = x
        
        x = self.conv1(x)
        x, state = self.bn1(x, state)
        x = jax.nn.relu(x)
        
        x = self.conv2(x)
        x, state = self.bn2(x, state)
        
        x = jax.nn.relu(x + identity)
        return x, state


class BlockV2(eqx.Module):
    """ResNet v2 block - batch norm before activation"""
    bn1: eqx.nn.BatchNorm
    conv1: eqx.nn.Conv2d
    bn2: eqx.nn.BatchNorm
    conv2: eqx.nn.Conv2d

    def __init__(self, num_channels: int, key: Optional[jax.random.PRNGKey] = None):
        if key is None:
            key = jax.random.PRNGKey(0)
        key1, key2 = jax.random.split(key, 2)
        
        self.bn1 = eqx.nn.BatchNorm(num_channels, axis_name="batch", mode="batch")
        self.conv1 = eqx.nn.Conv2d(num_channels, num_channels, 3, padding=1, key=key1)
        self.bn2 = eqx.nn.BatchNorm(num_channels, axis_name="batch", mode="batch")
        self.conv2 = eqx.nn.Conv2d(num_channels, num_channels, 3, padding=1, key=key2)

    def __call__(self, x: jnp.ndarray, state: eqx.nn.State) -> tuple[jnp.ndarray, eqx.nn.State]:
        identity = x
        
        x, state = self.bn1(x, state)
        x = jax.nn.relu(x)
        x = self.conv1(x)
        
        x, state = self.bn2(x, state)
        x = jax.nn.relu(x)
        x = self.conv2(x)
        
        return x + identity, state


class AZNet(eqx.Module):
    """AlphaZero NN architecture implemented in Equinox."""
    
    initial_conv: eqx.nn.Conv2d
    initial_bn: Optional[eqx.nn.BatchNorm]
    blocks: list
    final_bn: Optional[eqx.nn.BatchNorm]
    
    # Policy head
    policy_conv: eqx.nn.Conv2d
    policy_bn: eqx.nn.BatchNorm
    policy_linear: eqx.nn.Linear
    
    # Value head
    value_conv: eqx.nn.Conv2d
    value_bn: eqx.nn.BatchNorm
    value_linear1: eqx.nn.Linear
    value_linear2: eqx.nn.Linear
    
    num_actions: int
    num_channels: int
    num_blocks: int
    spatial_size: int  # Height * width of board
    resnet_v2: bool

    def __init__(
        self,
        num_actions: int,
        input_channels: int = 1,
        num_channels: int = 64,
        num_blocks: int = 5,
        resnet_v2: bool = True,
        spatial_size: int = 25,  # Default for 5x5 board 
        key: Optional[jax.random.PRNGKey] = None
    ):
        if key is None:
            key = jax.random.PRNGKey(0)
            
        self.num_actions = num_actions
        self.num_channels = num_channels
        self.num_blocks = num_blocks
        self.resnet_v2 = resnet_v2
        self.spatial_size = spatial_size
        
        keys = jax.random.split(key, num_blocks + 6)
        
        # Initial conv layer
        self.initial_conv = eqx.nn.Conv2d(
            input_channels, num_channels, 3, padding=1, key=keys[0]
        )
        
        # BatchNorm for ResNet v1 (after initial conv) 
        self.initial_bn = None if resnet_v2 else eqx.nn.BatchNorm(num_channels, axis_name="batch", mode="batch")
        
        # Residual blocks
        block_cls = BlockV2 if resnet_v2 else BlockV1
        self.blocks = [
            block_cls(num_channels, key=keys[i+1]) 
            for i in range(num_blocks)
        ]
        
        # Final BatchNorm for ResNet v2 (after all blocks)
        self.final_bn = eqx.nn.BatchNorm(num_channels, axis_name="batch", mode="batch") if resnet_v2 else None
        
        # Policy head
        self.policy_conv = eqx.nn.Conv2d(num_channels, 2, 1, key=keys[num_blocks+1])
        self.policy_bn = eqx.nn.BatchNorm(2, axis_name="batch", mode="batch")
        self.policy_linear = eqx.nn.Linear(2 * spatial_size, num_actions, key=keys[num_blocks+2])
        
        # Value head  
        self.value_conv = eqx.nn.Conv2d(num_channels, 1, 1, key=keys[num_blocks+3])
        self.value_bn = eqx.nn.BatchNorm(1, axis_name="batch", mode="batch")
        self.value_linear1 = eqx.nn.Linear(spatial_size, num_channels, key=keys[num_blocks+4])
        self.value_linear2 = eqx.nn.Linear(num_channels, 1, key=keys[num_blocks+5])

    def __call__(self, x: jnp.ndarray, state: eqx.nn.State) -> tuple[tuple[jnp.ndarray, jnp.ndarray], eqx.nn.State]:
        """Forward pass returning (logits, value) and updated state."""
        x = x.astype(jnp.float32)
        
        # Ensure x has channel-first format: (batch, channels, height, width)
        assert x.ndim == 4
        x = jnp.transpose(x, (0, 3, 1, 2))
        # elif x.ndim == 3:  # Add channel dimension
        #     x = x[..., None]  # Add channel as last dim
        #     x = jnp.transpose(x, (0, 3, 1, 2))  # Move to channel-first
        
        # Equinox Conv2d expects unbatched inputs, so we need to vmap over the batch dimension
        def single_forward(single_x, single_state):
            # Initial convolution
            x = self.initial_conv(single_x)
            
            # Initial batch norm for ResNet v1
            if not self.resnet_v2:
                x, single_state = self.initial_bn(x, single_state)
                x = jax.nn.relu(x)
            
            # Residual blocks
            for block in self.blocks:
                x, single_state = block(x, single_state)
            
            # Final batch norm for ResNet v2
            if self.resnet_v2:
                x, single_state = self.final_bn(x, single_state)
                x = jax.nn.relu(x)
            
            # Policy head
            logits = self.policy_conv(x)
            logits, single_state = self.policy_bn(logits, single_state)
            logits = jax.nn.relu(logits)
            # Flatten spatial dimensions: (channels, h, w) -> (channels*h*w)
            logits_flat = logits.reshape(-1)
            logits = self.policy_linear(logits_flat)
            
            # Value head
            v = self.value_conv(x)
            v, single_state = self.value_bn(v, single_state)
            v = jax.nn.relu(v)
            # Flatten spatial dimensions
            v_flat = v.reshape(-1)
            v = self.value_linear1(v_flat)
            v = jax.nn.relu(v)
            v = self.value_linear2(v)
            v = jnp.tanh(v)
            v = v.reshape(())  # Remove last dimension to make scalar
            
            return (logits, v), single_state
        
        # Apply over batch dimension with axis_name for BatchNorm
        (logits_batch, values_batch), new_state = jax.vmap(
            single_forward,
            in_axes=(0, None), 
            out_axes=(0, None),
            axis_name="batch"
        )(x, state)
        
        return (logits_batch, values_batch), new_state


def create_model(env, config, key) -> Tuple:
    """Create an Equinox model, initialize """
    dummy_state = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(0), 2))
    dummy_input = dummy_state.observation
    input_channels = dummy_input.shape[-1]  # Last dimension is channels
    spatial_size = dummy_input.shape[1] * dummy_input.shape[2]  # Height * width
    return eqx.nn.make_with_state(AZNet)(
        num_actions=env.num_actions,
        input_channels=input_channels,
        num_channels=config.num_channels,
        num_blocks=config.num_layers,
        resnet_v2=config.resnet_v2,
        spatial_size=spatial_size,
        key=key
    )


def load_from_ckpt(fpath: str) -> Tuple:
    with open(fpath, "rb") as f:
        d = pickle.load(f)
    model_params, model_state = d["model"]

    def forward_fn(model, state, x):
        return model(x, state)

    return forward_fn, model_params, model_state
