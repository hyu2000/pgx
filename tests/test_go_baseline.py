import jax
import jax.numpy as jnp
import pgx

import haiku as hk
#from IPython.display import *

print(pgx.__version__)
print(hk.__version__)


def test_run_game():
    """ runs on jax cpu! jax-metal 0.1.1 erred out """
    env_id = "go_9x9"
    model_id = "go_9x9_v0"

    env = pgx.make(env_id)
    model = pgx.make_baseline_model(model_id)

    init_fn = jax.jit(jax.vmap(env.init))
    step_fn = jax.jit(jax.vmap(env.step))

    states = []
    batch_size = 1
    keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
    state = init_fn(keys)
    states.append(state)
    while not (state.terminated | state.truncated).all():
        logits, value = model(state.observation)
        action = logits.argmax(axis=-1)
        state = step_fn(state, action)
        states.append(state)

    pgx.save_svg_animation(states, f"{env_id}.svg", frame_duration_seconds=1)
