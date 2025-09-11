import jax
import jax.numpy as jnp
import equinox as eqx


@eqx.filter_jit
def evaluate(env, rng_key, num_games, batch_policy1, batch_policy2):
    """
    """
    my_player = 0  # player is randomized at env.init

    key, subkey = jax.random.split(rng_key)
    batch_size = num_games
    keys = jax.random.split(subkey, batch_size)
    state = jax.vmap(env.init)(keys)

    def body_fn(val):
        key, state, R, action_history = val

        key, subkey1, subkey2 = jax.random.split(key, 3)
        policy_output1 = batch_policy1(state, subkey1)
        policy_output2 = batch_policy2(state, subkey2)
        is_my_turn = state.current_player == my_player  #).reshape((-1, 1))
        step_count = state._step_count[0]  # need a single int!
        # policy_output.action_weights   is action guaranteed to be the argmax?
        action = jnp.where(is_my_turn, policy_output1, policy_output2)
        state = jax.vmap(env.step)(state, action)
        R = R + state.rewards[jnp.arange(batch_size), my_player]
        action_history = action_history.at[:, step_count].set(action)
        return (key, state, R, action_history)

    action_history_init = jnp.ones((batch_size, env._game.max_termination_steps), dtype=jnp.int8) * -1
    _, _, R, action_history = jax.lax.while_loop(lambda x: ~(x[1].terminated.all()), body_fn,
                                                 (key, state, jnp.zeros(batch_size), action_history_init))
    if env._open_move:
        action_history = jnp.insert(action_history, 0, env._open_move, axis=1)
    return R, action_history
