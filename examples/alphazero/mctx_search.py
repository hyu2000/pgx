""" encapsulate mctx """
from functools import partial

import jax
import jax.numpy as jnp
import mctx
import pgx
import equinox as eqx


def make_recurrent_fn(forward_fn, env_step):
    """ forward_fn: Equinox forward function (model_param, model_state, obs) -> ((logits, value), new_state)
    """

    def recurrent_fn(inference_model, rng_key: jnp.ndarray, action: jnp.ndarray, state: pgx.State):
        """ per mctx spec
        """
        del rng_key
        # model_param, model_state = inference_model

        current_player = state.current_player
        state = jax.vmap(env_step)(state, action)

        (logits, value), _ = forward_fn(state.observation)
        
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


@partial(eqx.filter_jit)  #, static_argnums=[0, 1, 5])
# if we jit this func, num_simulation needs to be marked as static. If instead we want to jit its callers,
# how do we mark this static_arg? In a big project, there are lots of args sprinkled around that are configs
def improve_policy_with_mcts(forward_apply, recurrent_fn, model, state, rng_key, num_simulations: int):
    # model_params, model_state = model
    (logits, value), _ = forward_apply(state.observation)
    root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)

    policy_output = mctx.gumbel_muzero_policy(
        params=model,
        rng_key=rng_key,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=num_simulations,
        invalid_actions=~state.legal_action_mask,
        qtransform=partial(
            mctx.qtransform_completed_by_mix_value,
            rescale_values=False),
        max_num_considered_actions=16,  # default=16
        gumbel_scale=1.0,
    )
    return policy_output


def get_batch_fwd_mcts(batch_forward, batch_env_step, num_simulation: int = 32):
    """ return a new batch_fwd function, which has a simple signature:
    - params: env state, plus an extra rng_key (for mctx)
    - returns mctx.PolicyOutput
    """
    recur_fn = make_recurrent_fn(batch_forward, batch_env_step)

    def batch_fwd_mcts(state, rng_key):
        policy_output = improve_policy_with_mcts(batch_forward, recur_fn, None, state, rng_key, num_simulations=num_simulation)
        return policy_output

    return batch_fwd_mcts


def batch_fwd_mcts_to_policy(batch_fwd_mcts):
    def policy_fn(state, rng_key):
        return batch_fwd_mcts(state, rng_key).action
    return policy_fn