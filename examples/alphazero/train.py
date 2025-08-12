# Copyright 2023 The Pgx Authors. All Rights Reserved.
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

import datetime
import os
import pickle
from zoneinfo import ZoneInfo

import time
from functools import partial
from typing import NamedTuple
import platform

import equinox as eqx
import jax
import jax.numpy as jnp
import mctx
import optax
import pgx
import wandb
from omegaconf import OmegaConf

from examples.alphazero.config import Config
from examples.alphazero.mctx_search import make_recurrent_fn
from pgx.experimental import auto_reset

from network import AZNet

devices = jax.local_devices()
num_devices = len(devices)

# python train.py env_id=go_5x5C2 max_num_iters=200
conf_dict = OmegaConf.from_cli()
config: Config = Config(**conf_dict)
print(config)

env = pgx.make(config.env_id)

CHECKPOINT_DIR = '/Users/hyu/PycharmProjects/pgx/examples/alphazero/checkpoints' if platform.system() == 'Darwin' else '/content/drive/MyDrive/dlgo/pgx'
assert(os.path.isdir(CHECKPOINT_DIR))
baseline_id = 'go_5x5C2_250722-193343/000200'
baseline = pgx.make_baseline_model(config.env_id + "_v0", f'{CHECKPOINT_DIR}/{baseline_id}.ckpt')


def create_model(key, input_channels, spatial_size):
    """Create an Equinox model."""
    return AZNet(
        num_actions=env.num_actions,
        input_channels=input_channels,
        num_channels=config.num_channels,
        num_blocks=config.num_layers,
        resnet_v2=config.resnet_v2,
        spatial_size=spatial_size,
        key=key
    )

def forward_fn(model, state, x):
    """Forward pass with Equinox model."""
    return model(x, state)


lr_schedule_exp = optax.exponential_decay(
    init_value=config.learning_rate,
    transition_steps=config.lr_decay_steps,
    decay_rate=0.5,  # This gives you the same 2^(-e) behavior
    staircase=True   # This gives you the floor behavior
)
lr_schedule = optax.cosine_decay_schedule(
    config.learning_rate,
    decay_steps=config.lr_decay_steps,
    alpha=0.2
)
optimizer = optax.adam(lr_schedule_exp)
# optimizer = optax.chain(
#     optax.add_decayed_weights(config.weight_decay),
#     optax.sgd(lr_schedule, momentum=0.9),
# )


"""
def recurrent_fn(model, rng_key: jnp.ndarray, action: jnp.ndarray, state: pgx.State):
    # model: params
    # state: embedding
    del rng_key
    model_params, model_state = model

    current_player = state.current_player
    state = jax.vmap(env.step)(state, action)

    (logits, value), _ = forward.apply(model_params, model_state, state.observation, is_eval=True)
    # mask invalid actions
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)

    reward = state.rewards[jnp.arange(state.rewards.shape[0]), current_player]
    value = jnp.where(state.terminated, 0.0, value)
    discount = -1.0 * jnp.ones_like(value)
    discount = jnp.where(state.terminated, 0.0, discount)

    recurrent_fn_output = mctx.RecurrentFnOutput(
        reward=reward,
        discount=discount,
        prior_logits=logits,
        value=value,
    )
    return recurrent_fn_output, state
"""
recurrent_fn = make_recurrent_fn(forward_fn, env.step)


class SelfplayOutput(NamedTuple):
    obs: jnp.ndarray
    reward: jnp.ndarray
    terminated: jnp.ndarray
    action_weights: jnp.ndarray
    discount: jnp.ndarray


@jax.pmap
def selfplay(trainable_params, bn_state, rng_key: jnp.ndarray) -> SelfplayOutput:
    # Reconstruct full model from trainable and non-trainable parts
    model = eqx.combine(trainable_params, non_trainable_model)
    batch_size = config.selfplay_batch_size // num_devices

    def step_fn(state, key) -> SelfplayOutput:
        """ state: simultaneous games (batch_size)
        """
        key1, key2 = jax.random.split(key)
        observation = state.observation

        # Use inference mode for self-play evaluation
        inference_model = eqx.nn.inference_mode(model)
        (logits, value), _ = forward_fn(inference_model, bn_state, state.observation)
        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)

        policy_output = mctx.gumbel_muzero_policy(
            params=(model, bn_state),
            rng_key=key1,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=config.num_simulations,
            invalid_actions=~state.legal_action_mask,
            qtransform=partial(mctx.qtransform_completed_by_mix_value, rescale_values=False),
            gumbel_scale=1.0,
        )
        actor = state.current_player
        keys = jax.random.split(key2, batch_size)
        state = jax.vmap(auto_reset(env.step, env.init))(state, policy_output.action, keys)
        discount = -1.0 * jnp.ones_like(value)
        discount = jnp.where(state.terminated, 0.0, discount)
        return state, SelfplayOutput(
            obs=observation,   # obs is from the perspective of current player too
            action_weights=policy_output.action_weights,
            reward=state.rewards[jnp.arange(state.rewards.shape[0]), actor],  # reward from the perspective of current player
            terminated=state.terminated,
            discount=discount,
        )

    # Run selfplay for max_num_steps by batch
    rng_key, sub_key = jax.random.split(rng_key)
    keys = jax.random.split(sub_key, batch_size)
    state = jax.vmap(env.init)(keys)
    key_seq = jax.random.split(rng_key, config.max_num_steps)
    _, data = jax.lax.scan(step_fn, state, key_seq)

    return data  # data.[field]: (time, batch, ...)


class Sample(NamedTuple):
    obs: jnp.ndarray
    policy_tgt: jnp.ndarray
    value_tgt: jnp.ndarray
    mask: jnp.ndarray


@jax.pmap
def compute_loss_input(data: SelfplayOutput) -> Sample:
    batch_size = config.selfplay_batch_size // num_devices
    # If episode is truncated, there is no value target
    # auto-reset: the next init state kept the previous terminated/truncated/rewards
    # So when we compute value loss, we need to mask it (value_mask=0 means not using it)
    value_mask = jnp.cumsum(data.terminated[::-1, :], axis=0)[::-1, :] >= 1

    # Compute value target
    # discount=-1 except 0 for terminated
    # Be aware of off-by-1 error: rewards are stored at the next init state due to auto-reset, but next init-state shouldn't get that reward
    # The bug affects value_tgt for init states
    def body_fn(carry, i):
        ix = config.max_num_steps - i - 1
        v = -1 * carry
        carry = data.reward[ix] + data.discount[ix] * carry
        return carry, v

    _, value_tgt = jax.lax.scan(
        body_fn,
        jnp.zeros(batch_size),
        jnp.arange(config.max_num_steps),
    )
    value_tgt = value_tgt[::-1, :]

    return Sample(
        obs=data.obs,
        policy_tgt=data.action_weights,
        value_tgt=value_tgt,
        mask=value_mask,
    )


def loss_fn(model, bn_state, samples: Sample):
    (logits, value), _ = forward_fn(model, bn_state, samples.obs)

    policy_loss = optax.softmax_cross_entropy(logits, samples.policy_tgt)
    policy_loss = jnp.mean(policy_loss)

    value_loss = optax.l2_loss(value, samples.value_tgt)
    value_loss = jnp.mean(value_loss * samples.mask)  # mask if the episode is truncated

    return policy_loss + value_loss, (policy_loss, value_loss)


@partial(jax.pmap, axis_name="i")
def train(trainable_params, bn_state, opt_state, data: Sample):
    # Define loss function for trainable params only
    def loss_with_trainable(params, bn_state, data):
        full_model = eqx.combine(params, non_trainable_model)
        return loss_fn(full_model, bn_state, data)
    
    grads, (policy_loss, value_loss) = jax.grad(loss_with_trainable, has_aux=True)(
        trainable_params, bn_state, data
    )
    grads = jax.lax.pmean(grads, axis_name="i")
    updates, opt_state = optimizer.update(grads, opt_state, trainable_params)
    new_trainable = optax.apply_updates(trainable_params, updates)
    return new_trainable, bn_state, opt_state, policy_loss, value_loss


@jax.pmap
def evaluate(rng_key, my_trainable, my_bn_state):
    # Reconstruct full model
    my_model = eqx.combine(my_trainable, non_trainable_model)
    """A simplified evaluation by sampling. Only for debugging. 
    Please use MCTS and run tournaments for serious evaluation."""
    my_player = 0

    key, subkey = jax.random.split(rng_key)
    batch_size = config.eval_batch_size // num_devices
    keys = jax.random.split(subkey, batch_size)
    state = jax.vmap(env.init)(keys)

    def body_fn(val):
        key, state, R = val
        inference_model = eqx.nn.inference_mode(my_model)
        (my_logits, _), _ = forward_fn(inference_model, my_bn_state, state.observation)
        opp_logits, _ = baseline(state.observation)
        is_my_turn = (state.current_player == my_player).reshape((-1, 1))
        logits = jnp.where(is_my_turn, my_logits, opp_logits)
        key, subkey = jax.random.split(key)
        action = jax.random.categorical(subkey, logits, axis=-1)
        state = jax.vmap(env.step)(state, action)
        R = R + state.rewards[jnp.arange(batch_size), my_player]
        return (key, state, R)

    _, _, R = jax.lax.while_loop(
        lambda x: ~(x[1].terminated.all()), body_fn, (key, state, jnp.zeros(batch_size))
    )
    return R


def main():
    global non_trainable_model  # Make it global so functions can access it
    
    wandb.init(project="pgx-az", config=config.model_dump())

    # Initialize model and opt_state
    dummy_state = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(0), 2))
    dummy_input = dummy_state.observation
    input_channels = dummy_input.shape[-1]  # Last dimension is channels
    spatial_size = dummy_input.shape[1] * dummy_input.shape[2]  # Height * width
    
    # Create Equinox model with proper state handling
    model_key = jax.random.PRNGKey(0)
    model, bn_state = eqx.nn.make_with_state(create_model)(model_key, input_channels, spatial_size)
    
    # Test forward pass to get shapes right
    test_output, _ = forward_fn(model, bn_state, dummy_input)
    
    # For the optimizer we only need the trainable parameters (exclude axis_name strings)
    trainable_model = eqx.filter(model, eqx.is_array)
    opt_state = optimizer.init(trainable_model)
    
    # Store non-trainable parts separately (they don't need replication)
    non_trainable_model = eqx.filter(model, lambda x: not eqx.is_array(x))
    
    # Replicate only the trainable parts, bn_state, and opt_state
    trainable_replicated, bn_state, opt_state = jax.device_put_replicated(
        (trainable_model, bn_state, opt_state), devices
    )
    
    # Store the trainable and non-trainable parts for use in functions
    # We'll reconstruct the model within each pmap function

    # Prepare checkpoint dir
    now = datetime.datetime.now(tz=ZoneInfo("America/New_York"))
    now = now.strftime("%y%m%d-%H%M%S")
    ckpt_dir = os.path.join(CHECKPOINT_DIR, f"{config.env_id}_{now}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Initialize logging dict
    iteration: int = 0
    hours: float = 0.0
    frames: int = 0
    train_steps: int = 0
    log = {"iteration": iteration, "hours": hours, "frames": frames, "train_steps": train_steps}

    rng_key = jax.random.PRNGKey(config.seed)
    while True:
        if (1 + iteration) % config.eval_interval == 0:
            # Evaluation
            rng_key, subkey = jax.random.split(rng_key)
            keys = jax.random.split(subkey, num_devices)
            R = evaluate(keys, trainable_replicated, bn_state)
            log.update(
                {
                    # f"eval/vs_baseline/avg_R": R.mean().item(),
                    f"eval/vs_baseline/win_rate": ((R == 1).sum() / R.size).item(),
                    f"eval/vs_baseline/draw_rate": ((R == 0).sum() / R.size).item(),
                    # f"eval/vs_baseline/lose_rate": ((R == -1).sum() / R.size).item(),
                }
            )

        if iteration % config.checkpoint_interval == 0:
            # Store checkpoints - extract device 0 from arrays only
            def extract_device_0(x):
                if hasattr(x, '__getitem__') and hasattr(x, 'shape'):
                    return x[0]
                else:
                    return x
            
            trainable_0, bn_state_0, opt_state_0 = jax.tree_util.tree_map(extract_device_0, (trainable_replicated, bn_state, opt_state))
            # Reconstruct full model for checkpointing
            model_0 = eqx.combine(trainable_0, non_trainable_model)
            print(f'checkpointing to {ckpt_dir}/{iteration}')
            with open(os.path.join(ckpt_dir, f"{iteration:06d}.ckpt"), "wb") as f:
                dic = {
                    "config": config,
                    "rng_key": rng_key,
                    "model": jax.device_get(model_0),
                    "bn_state": jax.device_get(bn_state_0),
                    "opt_state": jax.device_get(opt_state_0),
                    "iteration": iteration,
                    "frames": frames,
                    "hours": hours,
                    "pgx.__version__": pgx.__version__,
                    "env_id": env.id,
                    "env_version": env.version,
                }
                pickle.dump(dic, f)

        print(log)
        wandb.log(log)

        if iteration >= config.max_num_iters:
            break

        iteration += 1
        log = {"iteration": iteration}
        st = time.time()

        # Selfplay
        rng_key, subkey = jax.random.split(rng_key)
        keys = jax.random.split(subkey, num_devices)
        data: SelfplayOutput = selfplay(trainable_replicated, bn_state, keys)
        samples: Sample = compute_loss_input(data)

        # Shuffle samples and make minibatches
        samples = jax.device_get(samples)  # (#devices, batch, max_num_steps, ...)
        frames_cur_iter = samples.obs.shape[0] * samples.obs.shape[1] * samples.obs.shape[2]
        samples = jax.tree_util.tree_map(lambda x: x.reshape((-1, *x.shape[3:])), samples)
        rng_key, subkey = jax.random.split(rng_key)
        ixs = jax.random.permutation(subkey, jnp.arange(samples.obs.shape[0]))
        samples = jax.tree_util.tree_map(lambda x: x[ixs], samples)  # shuffle
        num_updates = samples.obs.shape[0] // config.training_batch_size
        frames += frames_cur_iter
        train_steps += num_updates
        minibatches = jax.tree_util.tree_map(
            lambda x: x.reshape((num_updates, num_devices, -1) + x.shape[1:]), samples
        )

        # Training
        policy_losses, value_losses = [], []
        for i in range(num_updates):
            minibatch: Sample = jax.tree_util.tree_map(lambda x: x[i], minibatches)
            trainable_replicated, bn_state, opt_state, policy_loss, value_loss = train(trainable_replicated, bn_state, opt_state, minibatch)
            policy_losses.append(policy_loss.mean().item())
            value_losses.append(value_loss.mean().item())
        policy_loss = sum(policy_losses) / len(policy_losses)
        value_loss = sum(value_losses) / len(value_losses)

        et = time.time()
        hours += (et - st) / 3600
        log.update(
            {
                "train/policy_loss": policy_loss,
                "train/value_loss": value_loss,
                "hours": hours,
                "frames": frames,
                "train_steps": train_steps
            }
        )


if __name__ == "__main__":
    main()