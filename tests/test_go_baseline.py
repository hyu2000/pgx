import pickle
import platform
from typing import Iterable

import jax
import jax.numpy as jnp
import pgx
import equinox as eqx
# import haiku as hk

from examples.alphazero.network import AZNet, create_model, load_from_ckpt, get_batch_forward_fn, batch_forward_to_policy
from examples.alphazero import mctx_search
from examples.alphazero import train_util
from examples.alphazero.train_util import format_game_records

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


def test_load_colab_ckpt():
    """ debug why we cannot run colab ckpt model locally """
    CHECKPOINT_DIR = '/Users/hyu/PycharmProjects/pgx/examples/alphazero/checkpoints' if platform.system() == 'Darwin' else '/content/drive/MyDrive/dlgo/pgx'
    fpath = f'{CHECKPOINT_DIR}/go_5x5C2_250906-125418/000075.ckpt'
    # fpath = f'{CHECKPOINT_DIR}/go_5x5C2_250902-163535/000010.ckpt'

    key = jax.random.PRNGKey(0)
    with open(fpath, "rb") as f:
        d = pickle.load(f)
    env = pgx.make(d['env_id'])
    config = d['config']
    init_model = create_model(env, config, key=key)
    _, static = eqx.partition(init_model, eqx.is_array)

    model_arr, _ = eqx.partition(d['model'], eqx.is_array)
    model_params, model_state = eqx.combine(model_arr, static)
    batch_forward = get_batch_forward_fn(model_params, model_state)
    batch_policy = batch_forward_to_policy(batch_forward)

    batch_size = 2
    keys = jax.random.split(key, batch_size)
    init_fn = jax.jit(jax.vmap(env.init))
    state = init_fn(keys)
    print(state.observation.shape)
    (logits, values), _ = batch_forward(state.observation)
    print(values)
    action = batch_policy(state, key)
    print(action)


def test_run_game_mctx_eqx():
    env_id = "go_5x5C2"
    rng_key = jax.random.PRNGKey(1)
    env = pgx.make(env_id)

    batch_forward, model_param, model_state = load_go5_checkpoint_eqx('go_5x5C2_250909-160146/000100.ckpt')

    init_fn = jax.jit(jax.vmap(env.init))
    step_fn = jax.jit(jax.vmap(env.step))
    batch_fwd_mcts = mctx_search.get_batch_fwd_mcts(batch_forward, env.step, num_simulations=32)

    history = []
    batch_size = 5
    rng_key, key2 = jax.random.split(rng_key)
    keys = jax.random.split(key2, batch_size)
    state = init_fn(keys)
    history.append(state)
    assert len(state.observation) == batch_size
    while not (state.terminated | state.truncated).all():
        # (logits, value), _ = model_param(state.observation, model_state)
        rng_key, key2 = jax.random.split(rng_key)
        # policy_output = mctx_search.improve_policy_with_mcts(batch_forward, recur_fn, model, state, key2, num_simulations=32)
        policy_output = batch_fwd_mcts(state, key2)
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
    batch_forward, model_param, model_state = load_go5_checkpoint_eqx()
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
        (logits, value), _ = batch_forward(state.observation)
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
    if False:
        from examples.alphazero.config import Config
        config = Config()
        model_params, model_state = create_model(env, config, key=key)
        print(type(model_state))
    else:
        batch_forward, model_params, model_state = load_go5_checkpoint_eqx()

    # do some inference
    batch_size = 2
    keys = jax.random.split(key, batch_size)
    init_fn = jax.jit(jax.vmap(env.init))
    state = init_fn(keys)
    print(state.observation.shape)
    # inference_model = eqx.nn.inference_mode(model_params)
    # (logits, value), _ = inference_model(state.observation, model_state)
    # print(value)
    # forward_fn = eqx.filter_jit(eqx.filter_vmap(inference_model,
    #                                             in_axes=(0, None), out_axes=(0, None), axis_name="batch"))
    (logits, value), _ = batch_forward(state.observation)
    print(value)


def test_myeqx_net():
    from examples.alphazero.network_myeqx import create_model
    from examples.alphazero.config import Config

    env = pgx.make("go_5x5C2")
    key = jax.random.PRNGKey(0)
    config = Config()
    model_params, model_state = create_model(env, config, key=key)
    print(model_params)

    batch_size = 2
    keys = jax.random.split(key, batch_size)
    init_fn = jax.jit(jax.vmap(env.init))
    state = init_fn(keys)
    print(state.observation.shape)

    (logits, value), _ = model_params(state.observation, model_state)
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


def test_mcts_policy():
    env = pgx.make("go_5x5C2")
    key = jax.random.PRNGKey(0)

    batch_size = 2
    key, key2 = jax.random.split(key, 2)
    keys = jax.random.split(key, batch_size)
    init_fn = jax.jit(jax.vmap(env.init))
    step_fn = jax.jit(jax.vmap(env.step))
    state = init_fn(keys)

    batch_forward, model_params, model_state = load_go5_checkpoint_eqx('go_5x5C2_250903-143719/000050.ckpt')
    batch_forward_mcts = mctx_search.get_batch_fwd_mcts(batch_forward, env.step, num_simulations=32)
    for i in range(5):
        policy_output = batch_forward_mcts(state, key)
        print(policy_output.action)
        state = step_fn(state, policy_output.action)
        # print(state.observation.shape)


def test_eval_vs_baseline():
    """
# gen100 wrate against baseline (glorius-yogurt): 69% (policy sampling), 44% (#simu=32)
mctx is deterministic:
both num_simulations=1:  Total 64 games, win-rate= 0.421875
both num_simulations=32: Total 64 games, win-rate= 0.4375
policy-only is 0.32 ~ 0.40

re-run evaluate under same condition as in train, except for 64 games:
       gen:            30       40       50         60      70      80        90      100
raw policy wrates:  ['0.375', '0.688', '0.734', '0.734', '0.688', '0.594', '0.406', '0.328']
mcts wrates:        ['0.312', '0.906', '0.844', '0.719', '0.703', '0.562', '0.656', '0.438']

against fixed baseline #simu=1:
num_simulations=16: Total 64 games, win-rate= 0.671875
num_simulations=32: Total 64 games, win-rate= 0.65625
num_simulations=64: Total 64 games, win-rate= 0.78125
    """
    env = pgx.make("go_5x5C2")
    key = jax.random.PRNGKey(0)

    num_simulations = 32
    num_eval_games = 128
    batch_forward2, _, _ = load_go5_checkpoint_eqx('go_5x5C2_250906-125418/000075.ckpt')
    batch_policy2 = batch_forward_to_policy(batch_forward2)
    batch_forward_mcts2 = mctx_search.batch_fwd_mcts_to_policy(
        mctx_search.get_batch_fwd_mcts(batch_forward2, env.step, num_simulations=num_simulations))
    wrates_raw, wrates_mcts = [], []
    for gen1 in range(100, 150, 10):
        player_names = [f'gen{gen1}', f'baseline']
        batch_forward1, _, _ = load_go5_checkpoint_eqx(f'go_5x5C2_250909-160146/{gen1:06d}.ckpt')
        batch_policy1 = batch_forward_to_policy(batch_forward1)
        batch_forward_mcts1 = mctx_search.batch_fwd_mcts_to_policy(
            mctx_search.get_batch_fwd_mcts(batch_forward1, env.step, num_simulations=num_simulations))
        R, game_records = train_util.evaluate(env, key, num_eval_games, batch_policy1, batch_policy2)
        wrate = (1 + sum(R) / len(R)) * 0.5
        print(f'{gen1=}  raw: Total {len(R)} games, win-rate={wrate}')
        wrates_raw.append(wrate)
        R, game_records = train_util.evaluate(env, key, num_eval_games, batch_forward_mcts1, batch_forward_mcts2)
        wrate = (1 + sum(R) / len(R)) * 0.5
        print(f'{gen1=} mcts: Total {len(R)} games, win-rate=', (1 + sum(R) / len(R)) * 0.5)
        wrates_mcts.append(wrate)

        unique_games, counts = jnp.unique(game_records, axis=0, return_counts=True)
        print(counts)
        games_formatted = format_game_records(env, unique_games, player_names=player_names)
        print('\n'.join(games_formatted))
        print()

    print('raw policy wrates: ', [f'{x:.3f}' for x in wrates_raw])
    print('mcts wrates: ', [f'{x:.3f}' for x in wrates_mcts])


def show_game_records(game_records: jnp.array, env, player_names: Iterable[str] = None):
    unique_games, counts = jnp.unique(game_records, axis=0, return_counts=True)
    print(counts)
    games_formatted = format_game_records(env, unique_games, player_names=player_names)
    print('\n'.join(games_formatted))


def test_eval_cohort():
    """ examine ckpts, keeping a cohort of top models """
    env = pgx.make("go_5x5C2")
    key = jax.random.PRNGKey(0)

    RUN_ID = 'go_5x5C2_250917-210117'
    num_simulations = 32
    num_eval_games = 128

    cohort = [
        ('baseline',    'go_5x5C2_250906-125418/000075'),
        ('0909gen90',   'go_5x5C2_250909-160146/000090'),
        ('0909gen140',  'go_5x5C2_250909-160146/000140'),
    ]
    top_k = {}
    for short_name, model_id in cohort:
        batch_forward1, _, _ = load_go5_checkpoint_eqx(model_id)
        batch_forward_mcts1 = mctx_search.batch_fwd_mcts_to_policy(
            mctx_search.get_batch_fwd_mcts(batch_forward1, env.step, num_simulations=num_simulations))
        top_k[short_name] = batch_forward_mcts1

    for i_gen in range(0, 110, 10):
        #
        cur_player = f'{RUN_ID}/{i_gen:06d}'
        print(f'\nEvaluating {cur_player}: {num_simulations=}')
        batch_forward1, _, _ = load_go5_checkpoint_eqx(f'{cur_player}')
        batch_forward_mcts1 = mctx_search.batch_fwd_mcts_to_policy(
            mctx_search.get_batch_fwd_mcts(batch_forward1, env.step, num_simulations=num_simulations))
        for opponent_name, opponent in top_k.items():
            player_names = (f'gen{i_gen}', opponent_name)
            R, game_records = train_util.evaluate(env, key, num_eval_games, batch_forward_mcts1, opponent)
            print(f'eval {player_names}: total {len(R)} games, win-rate=', (1 + sum(R) / len(R)) * 0.5)
            show_game_records(game_records, env, player_names)
        # add/replace prev model
        top_k[f'prev-10'] = batch_forward_mcts1


def test_extract_selfplay_records():
    """ evaluate() on the same model """
    env = pgx.make("go_5x5C2")
    key = jax.random.PRNGKey(0)

    RUN_ID = 'go_5x5C2_250909-160146'
    num_simulations = 32
    num_eval_games = 16
    for i_gen in range(10, 110, 10):
        player_names = [f'gen{i_gen}', f'gen{i_gen}']
        print(f'Running eval on {player_names}: {num_simulations=}')
        batch_forward1, _, _ = load_go5_checkpoint_eqx(f'{RUN_ID}/{i_gen:06d}.ckpt')
        batch_forward_mcts1 = mctx_search.batch_fwd_mcts_to_policy(
            mctx_search.get_batch_fwd_mcts(batch_forward1, env.step, num_simulations=num_simulations))
        batch_policy1 = batch_forward_to_policy(batch_forward1)
        R, game_records = train_util.evaluate(env, key, num_eval_games, batch_forward_mcts1, batch_forward_mcts1)
        print(f'selfplay {i_gen}: total {len(R)} games, win-rate=', (1 + sum(R) / len(R)) * 0.5)

        show_game_records(game_records, env, player_names)


def test_run_eval():
    env = pgx.make("go_5x5C2")
    key = jax.random.PRNGKey(0)

    num_simulations = 32
    num_eval_games = 8
    batch_forward1, _, _ = load_go5_checkpoint_eqx('go_5x5C2_250909-160146/000140.ckpt')
    # batch_forward1, _, _ = load_go5_checkpoint_eqx('go_5x5C2_250907-093737/000050.ckpt')
    batch_forward_mcts1 = mctx_search.batch_fwd_mcts_to_policy(
        mctx_search.get_batch_fwd_mcts(batch_forward1, env.step, num_simulations=num_simulations))
    batch_policy1 = batch_forward_to_policy(batch_forward1)
    batch_forward2, _, _ = load_go5_checkpoint_eqx('go_5x5C2_250906-125418/000075.ckpt')
    # batch_forward2, _, _ = load_go5_checkpoint_eqx('go_5x5C2_250909-150643/000100.ckpt')
    batch_forward_mcts2 = mctx_search.batch_fwd_mcts_to_policy(
        mctx_search.get_batch_fwd_mcts(batch_forward2, env.step, num_simulations=num_simulations))
    batch_policy2 = batch_forward_to_policy(batch_forward2)
    R, game_records = train_util.evaluate(env, key, num_eval_games, batch_forward_mcts1, batch_forward_mcts2)
    print(f'Total {len(R)} games, win-rate=', (1 + sum(R) / len(R)) * 0.5)

    show_game_records(game_records, env, player_names=None)