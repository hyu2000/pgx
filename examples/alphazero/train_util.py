from functools import partial
from typing import Optional, List, Iterable, NamedTuple

import chex
import jax
import jax.numpy as jnp
import equinox as eqx
import mctx

import pgx
from examples.alphazero.config import Config
from pgx.experimental import coords, auto_reset


class SelfplayOutput(NamedTuple):
    actor: jnp.ndarray  # more for pairplay
    obs: jnp.ndarray
    reward: jnp.ndarray
    terminated: jnp.ndarray
    action_weights: jnp.ndarray
    discount: jnp.ndarray


@eqx.filter_jit
def pairplay(env, batch_mcts1, batch_mcts2, num_games: int, config: Config, rng_key) -> SelfplayOutput:
    """ """
    policy1_player = 0  # model1 is player0

    def step_fn(state, key):
        observation = state.observation
        actor = state.current_player

        key, subkey1, subkey2 = jax.random.split(key, 3)
        policy_output1 = batch_mcts1(state, subkey1)
        policy_output2 = batch_mcts2(state, subkey2)
        is_my_turn = actor == policy1_player
        chex.assert_rank(is_my_turn, 1)
        chex.assert_equal_shape([is_my_turn, policy_output1.action])

        action = jnp.where(is_my_turn, policy_output1.action, policy_output2.action)
        action_weights = jnp.where(is_my_turn.reshape((-1, 1)), policy_output1.action_weights, policy_output2.action_weights)

        keys = jax.random.split(key, batch_size)
        state = jax.vmap(auto_reset(env.step, env.init))(state, action, keys)
        discount = -1.0 * jnp.ones_like(state.terminated)
        discount = jnp.where(state.terminated, 0.0, discount)
        return state, SelfplayOutput(
            actor=actor,
            obs=observation,   # obs is from the perspective of current player too
            action_weights=action_weights,
            reward=state.rewards[jnp.arange(state.rewards.shape[0]), actor],  # reward from the perspective of current player
            terminated=state.terminated,
            discount=discount,
        )

    # Run selfplay for max_num_steps by batch
    batch_size = num_games
    rng_key, sub_key = jax.random.split(rng_key)
    keys = jax.random.split(sub_key, batch_size)
    state = jax.vmap(env.init)(keys)
    key_seq = jax.random.split(rng_key, config.max_num_steps)
    _, data = jax.lax.scan(step_fn, state, key_seq)

    chex.assert_shape(data.terminated, (config.max_num_steps, num_games))
    return data  # data.[field]: (time_step, game#, ...)


@eqx.filter_jit
def selfplay(env, model, num_games: int, config: Config, rng_key) -> SelfplayOutput:
    model_params, model_state = model
    model_params = eqx.nn.inference_mode(model_params)
    model = (model_params, model_state)
    arr, static = eqx.partition(model, eqx.is_array)

    def recurrent_fn(model, rng_key: jnp.ndarray, action: jnp.ndarray, state: pgx.State):
        del rng_key
        model = eqx.combine(model, static)
        model_params, model_state = model

        current_player = state.current_player
        state = jax.vmap(env.step)(state, action)

        (logits, value), _ = eqx.filter_vmap(model_params, in_axes=(0, None), out_axes=(0, None), axis_name="batch")(
            state.observation, model_state
        )
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

    def step_fn(state, key) -> SelfplayOutput:
        key1, key2 = jax.random.split(key)
        observation = state.observation

        (logits, value), _ = eqx.filter_vmap(model_params, in_axes=(0, None), out_axes=(0, None), axis_name="batch")(
            state.observation, model_state
        )
        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)

        policy_output = mctx.gumbel_muzero_policy(
            params=arr,
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
    batch_size = num_games
    rng_key, sub_key = jax.random.split(rng_key)
    keys = jax.random.split(sub_key, batch_size)
    state = jax.vmap(env.init)(keys)
    key_seq = jax.random.split(rng_key, config.max_num_steps)
    _, data = jax.lax.scan(step_fn, state, key_seq)

    chex.assert_shape(data.terminated, (config.max_num_steps, num_games))
    return data  # data.[field]: (time_step, game#, ...)


class Sample(NamedTuple):
    actor: jnp.ndarray
    obs: jnp.ndarray
    policy_tgt: jnp.ndarray
    value_tgt: jnp.ndarray
    mask: jnp.ndarray


@jax.jit
def compute_loss_input(data: SelfplayOutput) -> Sample:
    # batch_size = config.selfplay_batch_size // num_devices
    max_num_steps, batch_size = data.terminated.shape
    # If episode is truncated, there is no value target
    # auto-reset: only final state is marked as terminated. later states are reset
    # So when we compute value loss, we need to mask it (value_mask=0 means not using it)
    value_mask = jnp.cumsum(data.terminated[::-1, :], axis=0)[::-1, :] >= 1

    # Compute value target
    # discount=-1 except 0 for terminated
    def body_fn(carry, i):
        ix = max_num_steps - i - 1
        v = data.reward[ix] + data.discount[ix] * carry
        return v, v

    _, value_tgt = jax.lax.scan(
        body_fn,
        jnp.zeros(batch_size),
        jnp.arange(max_num_steps),
    )
    value_tgt = value_tgt[::-1, :]

    return Sample(
        actor=data.actor,
        obs=data.obs,
        policy_tgt=data.action_weights,
        value_tgt=value_tgt,
        mask=value_mask,
    )


@eqx.filter_jit
def evaluate(env, rng_key, num_games, batch_policy1, batch_policy2):
    """
    """
    policy1_player = 0  # starting player is randomized upon env.init

    key, subkey = jax.random.split(rng_key)
    batch_size = num_games
    keys = jax.random.split(subkey, batch_size)
    state = jax.vmap(env.init)(keys)

    def body_fn(val):
        key, state, R, action_history = val

        key, subkey1, subkey2 = jax.random.split(key, 3)
        policy_output1 = batch_policy1(state, subkey1)
        policy_output2 = batch_policy2(state, subkey2)
        is_my_turn = state.current_player == policy1_player  #).reshape((-1, 1))
        step_count = state._step_count.max()  # need a single int; max() since some games may've ended
        # policy_output.action_weights   is action guaranteed to be the argmax?
        action = jnp.where(is_my_turn, policy_output1, policy_output2)
        state = jax.vmap(env.step)(state, action)
        R = R + state.rewards[jnp.arange(batch_size), policy1_player]
        action_history = action_history.at[:, step_count].set(action)
        return (key, state, R, action_history)

    game_record = jnp.ones((batch_size, env._game.max_termination_steps), dtype=jnp.int8) * -1
    _, _, R, game_record = jax.lax.while_loop(lambda x: ~(x[1].terminated.all()), body_fn,
                                                 (key, state, jnp.zeros(batch_size), game_record))
    if env._open_move:
        game_record = jnp.insert(game_record, 0, env._open_move, axis=1)
    # add meta data: which player started the game (as white in Go5C2), policy1_player (0) win/lose
    game_record = jnp.insert(game_record, 0, state.current_player, axis=1)
    game_record = jnp.insert(game_record, 1, R.astype(int), axis=1)
    return R, game_record


def convert_to_black_view(player_to_start, player0_reward, open_move: Optional[int]):
    """ convert game result from player-centric view to color-centric view

    player_to_start: 0/1
    player0_reward: 1/-1
    """
    black_player_id = player_to_start
    if open_move is not None:
        black_player_id = 1 - player_to_start

    if player0_reward == 0:
        return 'B+T', black_player_id

    black_reward = player0_reward
    if player_to_start != 0:
        black_reward *= -1
    if open_move is not None:
        black_reward *= -1
    return ('B+R' if black_reward > 0 else 'W+R'), black_player_id


def format_game_records(env, game_records: jnp.array, sgf: bool=False, player_names: Iterable[str] = None):
    open_move = env._open_move
    records = []
    for game in game_records:
        game_result, black_player_id = convert_to_black_view(game[0], game[1], open_move)
        moves_str = ' '.join([coords.arr_to_gtp(game[2:], sgf=sgf)])

        white_player_id = 1 - black_player_id
        if player_names:
            black_player_name, white_player_name = player_names[black_player_id], player_names[white_player_id]
        else:
            black_player_name, white_player_name = black_player_id, white_player_id
        s = f'{black_player_name} {white_player_name} {game_result} {moves_str}'
        records.append(s)
    return records
