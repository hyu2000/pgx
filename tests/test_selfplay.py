import platform

import chex
import jax
import jax.numpy as jnp
import equinox as eqx
import pgx
from examples.alphazero.config import Config
from examples.alphazero.network import load_from_ckpt, get_batch_forward_fn
from examples.alphazero.train_lib import selfplay, compute_loss_input, pairplay
from examples.alphazero import mctx_search


def load_go5_checkpoint_eqx(fpath = None):
    if not fpath:
        fpath = f'go_5x5C2_250903-143719/000210.ckpt'
        fpath = 'go_5x5C2_250906-125418/000075.ckpt'  # mini-batch new baseline
    if not fpath.startswith('/'):
        CHECKPOINT_DIR = '/Users/hyu/PycharmProjects/pgx/examples/alphazero/checkpoints' if platform.system() == 'Darwin' else '/content/drive/MyDrive/dlgo/pgx'
        fpath = f'{CHECKPOINT_DIR}/{fpath}'
    if not fpath.endswith('.ckpt'):
        fpath = f'{fpath}.ckpt'
    model_params, model_state = load_from_ckpt(fpath)
    model_params = eqx.nn.inference_mode(model_params)
    batch_forward = get_batch_forward_fn(model_params, model_state)
    return batch_forward, model_params, model_state


def test_hashable_config():
    """ to use config as an arg of a jitted function, it needs to be hashable """
    config = Config(num_simulations=2)
    print(config.num_simulations)
    d = config.model_dump()
    print(d['num_simulations'])

    # hashable when frozen=True
    config_set = {config}


def test_selfplay():
    env = pgx.make("go_5x5C2")
    key = jax.random.PRNGKey(0)

    config = Config()
    num_games = 4
    batch_forward1, model_param, model_state = load_go5_checkpoint_eqx('go_5x5C2_250909-160146/000140.ckpt')

    data = selfplay(env, (model_param, model_state), num_games, config, key)
    print(data.terminated.shape)
    chex.assert_shape(data.terminated, (config.max_num_steps, num_games))
    chex.assert_equal_shape([data.reward, data.discount, data.terminated])
    print('terminated', data.terminated.sum(axis=0))
    print('reward abs(sum)=', jnp.abs(data.reward).sum(axis=0), 'sum=', data.reward.sum(axis=0))

    samples = compute_loss_input(data)
    print(samples.value_tgt)


def test_pairplay():
    env = pgx.make("go_5x5C2")
    key = jax.random.PRNGKey(0)

    config = Config()
    num_games = 4
    batch_forward1, model_param, model_state = load_go5_checkpoint_eqx('go_5x5C2_250909-160146/000140.ckpt')
    batch_mcts1 = mctx_search.get_batch_fwd_mcts(batch_forward1, env.step, 2)

    data = pairplay(env, batch_mcts1, batch_mcts1, num_games, config, key)
    print(data.terminated.shape)
    chex.assert_shape(data.terminated, (config.max_num_steps, num_games))
    chex.assert_equal_shape([data.reward, data.discount, data.terminated])
    print('terminated', data.terminated.sum(axis=0))
    print('reward abs(sum)=', jnp.abs(data.reward).sum(axis=0), 'sum=', data.reward.sum(axis=0))

    samples = compute_loss_input(data)
    print(samples.value_tgt.shape)
    print(samples.actor.shape)
