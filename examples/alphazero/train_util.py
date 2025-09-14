from typing import Optional, List, Iterable

import jax
import jax.numpy as jnp
import equinox as eqx

from pgx.experimental import coords


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
