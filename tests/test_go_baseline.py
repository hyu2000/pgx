import pickle
import platform

import jax
import jax.numpy as jnp
import pgx
import equinox as eqx
# import haiku as hk

from examples.alphazero.network import AZNet, create_model, load_from_ckpt
from examples.alphazero import mctx_search
from pgx._src.baseline import init_az_random_model

#from IPython.display import *

print(pgx.__version__)
print(jax.__version__)


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


def load_go5_checkpoint_hk():
    env_id = "go_5x5C2"
    model_id = f"{env_id}_v0"
    # model is a function: model(state.observation)
    CHECKPOINT_DIR = '/Users/hyu/PycharmProjects/pgx/examples/alphazero/checkpoints' if platform.system() == 'Darwin' else '/content/drive/MyDrive/dlgo/pgx'
    model = pgx.make_baseline_model(model_id,
                                    download_dir=f'{CHECKPOINT_DIR}/go_5x5C2_250722-193343/000200.ckpt')
    # model is apply(model_params)
    return model


def test_run_game_mctx_hk():
    env_id = "go_5x5C2"
    rng_key = jax.random.PRNGKey(1)
    env = pgx.make(env_id)

    from pgx._src.baseline import load_hk_baseline_model
    CHECKPOINT_DIR = '/Users/hyu/PycharmProjects/pgx/examples/alphazero/checkpoints' if platform.system() == 'Darwin' else '/content/drive/MyDrive/dlgo/pgx'
    fpath = f'{CHECKPOINT_DIR}/go_5x5C2_250722-193343/000200.ckpt'
    # fpath = f'{CHECKPOINT_DIR}/go_5x5C2_250827-130633/000005.ckpt'
    model_apply, model_param, model_state = load_hk_baseline_model(fpath)
    model = (model_param, model_state)

    init_fn = jax.jit(jax.vmap(env.init))
    step_fn = jax.jit(jax.vmap(env.step))
    recur_fn = mctx_search.make_recurrent_fn(model_apply, env.step)

    states = []
    batch_size = 10
    rng_key, key2 = jax.random.split(rng_key)
    keys = jax.random.split(key2, batch_size)
    state = init_fn(keys)
    states.append(state)
    assert len(state.observation) == batch_size
    while not (state.terminated | state.truncated).all():
        # logits, value = model_apply(state.observation)
        rng_key, key2 = jax.random.split(rng_key)
        policy_output = mctx_search.improve_policy_with_mcts(model_apply, recur_fn, model, state, key2, num_simulations=2)
        action = policy_output.action
        state = step_fn(state, action)
        states.append(state)

    pgx.save_svg_animation(states, f"{env_id}.svg", frame_duration_seconds=1)


def load_go5_checkpoint_eqx():
    CHECKPOINT_DIR = '/Users/hyu/PycharmProjects/pgx/examples/alphazero/checkpoints' if platform.system() == 'Darwin' else '/content/drive/MyDrive/dlgo/pgx'
    fpath = f'{CHECKPOINT_DIR}/go_5x5C2_250831-215449/000005.ckpt'
    return load_from_ckpt(fpath)


def test_run_game_mctx_eqx():
    env_id = "go_5x5C2"
    rng_key = jax.random.PRNGKey(1)
    env = pgx.make(env_id)

    model_apply, model_param, model_state = load_go5_checkpoint_eqx()
    model_param = eqx.nn.inference_mode(model_param)
    model = (model_param, model_state)

    init_fn = jax.jit(jax.vmap(env.init))
    step_fn = jax.jit(jax.vmap(env.step))
    recur_fn = mctx_search.make_recurrent_fn(model_apply, env.step)

    history = []
    batch_size = 2
    rng_key, key2 = jax.random.split(rng_key)
    keys = jax.random.split(key2, batch_size)
    state = init_fn(keys)
    history.append(state)
    assert len(state.observation) == batch_size
    while not (state.terminated | state.truncated).all():
        (logits, value), _ = model_param(state.observation, model_state)
        rng_key, key2 = jax.random.split(rng_key)
        policy_output = mctx_search.improve_policy_with_mcts(model_apply, recur_fn, model, state, key2, num_simulations=2)
        action = policy_output.action
        state = step_fn(state, action)
        history.append(state)

    pgx.save_svg_animation(history, f"{env_id}.svg", frame_duration_seconds=1)


def test_run_game_raw_policy():
    """ runs on jax cpu! jax-metal 0.1.1 erred out
    """
    env_id = "go_5x5C2"
    rng_key = jax.random.PRNGKey(1)

    env = pgx.make(env_id)
    rng_key, key2 = jax.random.split(rng_key)
    model_apply, model_param, model_state = load_go5_checkpoint_eqx()
    # model = init_az_random_model(env, key2)

    init_fn = jax.jit(jax.vmap(env.init))
    step_fn = jax.jit(jax.vmap(env.step))

    states = []
    batch_size = 2
    rng_key, key2 = jax.random.split(rng_key)
    keys = jax.random.split(key2, batch_size)
    state = init_fn(keys)
    states.append(state)
    assert len(state.observation) == batch_size
    while not (state.terminated | state.truncated).all():
        (logits, value), _ = model_param(state.observation, model_state)
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


def test_init_save():
    """ """
    env = pgx.make("go_5x5C2")
    key = jax.random.PRNGKey(0)
    if True:
        from examples.alphazero.config import Config
        config = Config()
        model_params, model_state = create_model(env, config, key=key)
        print(type(model_state))
    else:
        model_apply, model_params, model_state = load_go5_checkpoint_eqx()

    # do some inference
    batch_size = 2
    keys = jax.random.split(key, batch_size)
    init_fn = jax.jit(jax.vmap(env.init))
    state = init_fn(keys)
    print(state.observation.shape)
    inference_model = eqx.nn.inference_mode(model_params)
    # (logits, value), _ = inference_model(state.observation, model_state)
    # print(value)
    forward_fn = eqx.filter_jit(eqx.filter_vmap(inference_model,
                                                in_axes=(0, None), out_axes=(0, None), axis_name="batch"))
    (logits, value), _ = forward_fn(state.observation, model_state)
    print(value)



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
