""" encapsulate mctx """
import jax
import jax.numpy as jnp
import mctx
import pgx


def make_recurrent_fn(forward_apply, env_step):
    """ forward_apply is not bound w/ params
    """

    def recurrent_fn(model, rng_key: jnp.ndarray, action: jnp.ndarray, state: pgx.State):
        """
        seems model cannot be Any (but a0jax uses eqx.Module); state can be Any
        state/action: array
        """
        # model: params
        # state: embedding
        del rng_key
        model_params, model_state = model

        current_player = state.current_player
        state = jax.vmap(env_step)(state, action)

        (logits, value), _ = forward_apply(model_params, model_state, state.observation, is_eval=True)
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

    return recurrent_fn


def improve_policy_with_mcts(forward_apply, recurrent_fn, model, state, rng_key, num_simulations: int):
    model_params, model_state = model
    (logits, value), _ = forward_apply(
        model_params, model_state, state.observation, is_eval=True
    )
    root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)

    policy_output = mctx.gumbel_muzero_policy(
        params=model,
        rng_key=rng_key,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=num_simulations,
        invalid_actions=~state.legal_action_mask,
        qtransform=mctx.qtransform_completed_by_mix_value,
        gumbel_scale=1.0,
    )
    return policy_output
