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

import chex
import cloudpickle as pickle
from zoneinfo import ZoneInfo

import time
from functools import partial
from typing import NamedTuple
import platform

import jax
import jax.numpy as jnp
import equinox as eqx
import mctx
import optax
import pgx
import wandb
from omegaconf import OmegaConf

from examples.alphazero.config import Config
from pgx.experimental import auto_reset
from examples.alphazero.network import create_model, load_from_ckpt, get_batch_forward_fn, batch_forward_to_policy
from examples.alphazero import mctx_search
from examples.alphazero import train_lib

devices = jax.local_devices()
num_devices = len(devices)

conf_dict = OmegaConf.from_cli()
config: Config = Config(**conf_dict)
print(config)

env = pgx.make(config.env_id)


def get_batch_fwd_mcts_for_model(model, num_simulations: int):
    model_params, model_state = model
    model_params = eqx.nn.inference_mode(model_params)
    batch_forward = get_batch_forward_fn(model_params, model_state)

    batch_policy = batch_forward_to_policy(batch_forward)
    batch_mcts_policy = mctx_search.batch_fwd_mcts_to_policy(mctx_search.get_batch_fwd_mcts(
        batch_forward, env.step, num_simulations=num_simulations))
    return batch_policy, batch_mcts_policy


CHECKPOINT_DIR = '/Users/hyu/PycharmProjects/pgx/examples/alphazero/checkpoints' if platform.system() == 'Darwin' else '/content/drive/MyDrive/dlgo/pgx'
assert(os.path.isdir(CHECKPOINT_DIR))
baseline_id = config.baseline
baseline_model = load_from_ckpt(f'{CHECKPOINT_DIR}/{baseline_id}.ckpt')
baseline_raw, baseline_mcts = get_batch_fwd_mcts_for_model(baseline_model, config.num_simulations)


lr_schedule_exp = optax.exponential_decay(
    init_value=config.learning_rate,
    transition_steps=config.lr_decay_steps,
    decay_rate=0.5,  # This gives you the same 2^(-e) behavior
    staircase=True   # This gives you the floor behavior
)
lr_schedule_cos = optax.cosine_decay_schedule(
    config.learning_rate,
    decay_steps=config.lr_decay_steps,
    alpha=0.005
)
optimizer = optax.adam(lr_schedule_cos)


def loss_fn(model_params, model_state, samples: train_lib.Sample):
    (logits, value), model_state = eqx.filter_vmap(
        model_params, in_axes=(0, None), out_axes=(0, None), axis_name="batch"
    )(samples.obs, model_state)

    policy_loss = optax.softmax_cross_entropy(logits, samples.policy_tgt)
    policy_loss = jnp.mean(policy_loss)

    value_loss = optax.l2_loss(value, samples.value_tgt)
    value_loss = jnp.mean(value_loss * samples.mask)  # mask if the episode is truncated

    return policy_loss + value_loss, (model_state, policy_loss, value_loss)


def shuffle_and_batch(samples: train_lib.Sample, batch_size: int, rng_key) -> (train_lib.Sample, int):
    """ Shuffle samples and make minibatches
    """
    ixs = jax.random.permutation(rng_key, jnp.arange(samples.obs.shape[0]))
    samples = jax.tree_util.tree_map(lambda x: x[ixs], samples)  # shuffle
    num_updates = samples.obs.shape[0] // batch_size
    # TODO we could shave samples so that reshape will always succeed
    minibatches = jax.tree_util.tree_map(lambda x: x.reshape((num_updates, -1) + x.shape[1:]), samples)
    return minibatches, num_updates


@eqx.filter_jit
def train(model, opt_state, data: train_lib.Sample):
    model_params, model_state = model
    grads, (model_state, policy_loss, value_loss) = eqx.filter_grad(loss_fn, has_aux=True)(
        model_params, model_state, data
    )
    updates, opt_state = optimizer.update(grads, opt_state)
    model_params = eqx.apply_updates(model_params, updates)
    model = (model_params, model_state)
    return model, opt_state, policy_loss, value_loss


def main():
    wandb.init(project=config.wandb_project, config=config.model_dump())

    rng_key = jax.random.key(config.seed)
    # Initialize model and opt_state
    rng_key, model_key = jax.random.split(rng_key)
    if config.init_model:
        print(f'loading init_model from {config.init_model}')
        init_model, state = load_from_ckpt(f'{CHECKPOINT_DIR}/{config.init_model}.ckpt')
    else:
        print(f'initialize model from random')
        init_model, state = create_model(env, config, key=model_key)
    model = (init_model, state)
    opt_state = optimizer.init(eqx.filter(init_model, eqx.is_array))

    # Prepare checkpoint dir
    now = datetime.datetime.now(tz=ZoneInfo("America/New_York"))
    now = now.strftime("%y%m%d-%H%M%S")
    ckpt_dir = os.path.join(CHECKPOINT_DIR, f"{config.env_id}_{now}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Initialize logging dict
    iteration: int = 0
    hours: float = 0.0
    frames: int = 0
    grad_steps: int = 0
    log = {"iteration": iteration, "hours": hours, "frames": frames, "grad_steps": grad_steps}

    while True:
        if (1 + iteration) % config.eval_interval == 0:
            # Evaluation
            rng_key, subkey, subkey2 = jax.random.split(rng_key, 3)
            batch_forward_mcts = batch_mcts1
            # R, records = train_lib.evaluate(env, subkey, config.eval_batch_size, batch_raw_policy, baseline_raw)
            R_mcts, records = train_lib.evaluate(env, subkey2, config.eval_batch_size, batch_forward_mcts, baseline_mcts)
            log.update(
                {
                    # f"eval/vs_baseline/avg_R": R.mean().item(),
                    f"eval/vs_baseline/win_rate": ((R == 1).sum() / R.size).item(),
                    f"eval/vs_mcts/win_rate": ((R_mcts == 1).sum() / R_mcts.size).item(),
                    f"eval/vs_baseline/draw_rate": ((R == 0).sum() / R.size).item(),
                    # f"eval/vs_baseline/lose_rate": ((R == -1).sum() / R.size).item(),
                }
            )

        if iteration % config.checkpoint_interval == 0:
            # Store checkpoints
            # model_0, opt_state_0 = jax.tree_util.tree_map(lambda x: x[0], (train_model, opt_state))
            # model_0, opt_state_0 = eqx.filter((model, opt_state), eqx.is_array)
            model_0, opt_state_0 = model, opt_state
            with open(os.path.join(ckpt_dir, f"{iteration:06d}.ckpt"), "wb") as f:
                dic = {
                    "config": config,
                    "rng_key": rng_key,
                    "model": jax.device_get(model_0),
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
        # data: train_lib.SelfplayOutput = train_lib.selfplay(env, model, config.selfplay_batch_size, config, subkey)
        batch_mcts1 = get_batch_fwd_mcts_for_model(model, num_simulations=config.num_simulations)
        data: train_lib.SelfplayOutput = train_lib.pairplay(env, batch_mcts1, batch_mcts1, config.selfplay_batch_size, config, subkey)
        samples: train_lib.Sample = train_lib.compute_loss_input(data)

        # samples = jax.device_get(samples)  # (#devices, max_num_steps, batch, ...)
        frames += samples.obs.shape[0] * samples.obs.shape[1]
        samples = jax.tree_util.tree_map(lambda x: x.reshape((-1, *x.shape[2:])), samples)
        chex.assert_rank([samples.value_tgt, samples.policy_tgt], [1, 2])
        rng_key, subkey = jax.random.split(rng_key)
        minibatches, num_updates = shuffle_and_batch(samples, config.training_batch_size, subkey)
        grad_steps += num_updates

        # Training
        policy_losses, value_losses = [], []
        for i in range(num_updates):
            minibatch: train_lib.Sample = jax.tree_util.tree_map(lambda x: x[i], minibatches)
            model, opt_state, policy_loss, value_loss = train(model, opt_state, minibatch)
            policy_losses.append(policy_loss.mean().item())
            value_losses.append(value_loss.mean().item())

        policy_loss = sum(policy_losses) / len(policy_losses)
        value_loss  = sum( value_losses) / len( value_losses)

        et = time.time()
        hours += (et - st) / 3600
        log.update(
            {
                "train/policy_loss": policy_loss,
                "train/value_loss": value_loss,
                "hours": hours,
                "frames": frames,
                "grad_steps": grad_steps
            }
        )


if __name__ == "__main__":
    main()
