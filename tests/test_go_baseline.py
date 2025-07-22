import pickle

import jax
import jax.numpy as jnp
import pgx

import haiku as hk

from examples.alphazero.network import AZNet
from examples.alphazero.config import Config

#from IPython.display import *

print(pgx.__version__)
print(jax.__version__)
print(hk.__version__)


def test_load_checkpoint():
    local_dir = '/Users/hyu/PycharmProjects/pgx/examples/alphazero/checkpoints/go_5x5_20250722021439'
    with open(f'{local_dir}/000000.ckpt', 'rb') as f:
        d = pickle.load(f)
        print(d.keys())


def sample_legal_action(rng_key, logits, legal_mask):
    """ as logits get sharper, there are less diversity
    Might help to increase batch-size
    """
    masked_logits = jnp.where(legal_mask, logits, -jnp.inf)
    return jax.random.categorical(rng_key, logits=masked_logits, axis=-1)


def test_run_game():
    """ runs on jax cpu! jax-metal 0.1.1 erred out
    """
    env_id = "go_5x5C2"
    model_id = f"{env_id}_v0"
    rng_key = jax.random.PRNGKey(1)

    env = pgx.make(env_id)
    # model is a function: model(state.observation)
    model = pgx.make_baseline_model(model_id,
                                    download_dir='/Users/hyu/PycharmProjects/pgx/examples/alphazero/checkpoints/go_5x5_20250722113749/000100.ckpt')

    init_fn = jax.jit(jax.vmap(env.init))
    step_fn = jax.jit(jax.vmap(env.step))

    states = []
    batch_size = 10
    rng_key, key2 = jax.random.split(rng_key)
    keys = jax.random.split(key2, batch_size)
    state = init_fn(keys)
    states.append(state)
    assert len(state.observation) == batch_size
    while not (state.terminated | state.truncated).all():
        logits, value = model(state.observation)
        # action = logits.argmax(axis=-1)
        rng_key, key2 = jax.random.split(rng_key)
        action = sample_legal_action(key2, logits, state.legal_action_mask)
        state = step_fn(state, action)
        states.append(state)

    pgx.save_svg_animation(states, f"{env_id}.svg", frame_duration_seconds=1)


def forward_fn(x, is_eval=True):
    net = AZNet(
        num_actions=26,
        num_channels=4,
        num_blocks=3,
        resnet_v2=True,
    )
    policy_out, value_out = net(x, is_training=not is_eval, test_local_stats=False)
    return policy_out, value_out


def test_play_random_model():
    """ random play on go5CX2 """
    env_id = "go_5x5C2"

    env = pgx.make(env_id)

    # random init a model
    forward = hk.without_apply_rng(hk.transform_with_state(forward_fn))
    dummy_state = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(1), 2))
    dummy_input = dummy_state.observation
    # is_eval needs to be False for BatchNorm to initialize
    model = forward.init(jax.random.PRNGKey(0), dummy_input, is_eval=False)  # (params, state)

    def apply(obs):
        (logits, value), _ = forward.apply(model[0], model[1], obs, is_eval=True)
        return logits, value

    # run games
    init_fn = jax.jit(jax.vmap(env.init))
    step_fn = jax.jit(jax.vmap(env.step))

    states = []
    batch_size = 3
    keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
    state = init_fn(keys)
    states.append(state)
    while not (state.terminated | state.truncated).all():
        logits, value = apply(state.observation)
        action = logits.argmax(axis=-1)
        state = step_fn(state, action)
        states.append(state)

    print('Total #states =', len(states))
    pgx.save_svg_animation(states, f"{env_id}.svg", frame_duration_seconds=1)
