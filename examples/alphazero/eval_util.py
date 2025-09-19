import platform
from typing import Optional, List, Iterable, NamedTuple, Any, Dict
from dataclasses import dataclass
import chex
import jax
import jax.numpy as jnp
import equinox as eqx
import pgx
from examples.alphazero import mctx_search
from examples.alphazero.network import load_from_ckpt, get_batch_forward_fn


@dataclass
class ModelPolicy:
    name: str
    model_id: str
    model: tuple

    batch_forward: Any
    batch_mcts: Any = None
    batch_mcts_policy: Any = None


def load_checkpoint(fpath: str, CHECKPOINT_DIR: str):
    if not fpath.startswith('/'):
        fpath = f'{CHECKPOINT_DIR}/{fpath}'
    if not fpath.endswith('.ckpt'):
        fpath = f'{fpath}.ckpt'
    model_params, model_state = load_from_ckpt(fpath)
    model_params = eqx.nn.inference_mode(model_params)
    batch_forward = get_batch_forward_fn(model_params, model_state)
    return batch_forward, model_params, model_state


def load_cohort(cohort: Dict[str, str], CHECKPOINT_DIR: str) -> Dict[str, ModelPolicy]:
    d = {}
    for name, model_id in cohort.items():
        batch_fwd, model_param, model_state = load_checkpoint(model_id, CHECKPOINT_DIR)
        d[name] = ModelPolicy(name, model_id, (model_param, model_state), batch_fwd, None, None)
    return d


def fill_in_batch_mcts(cohort: Dict[str, ModelPolicy], env, num_simulations: int = 32):
    for mp in cohort.values():
        assert mp.batch_forward is not None
        mp.batch_mcts = mctx_search.get_batch_fwd_mcts(
            mp.batch_forward, env.step, num_simulations=num_simulations
        )
        mp.batch_mcts_policy = mctx_search.batch_fwd_mcts_to_policy(mp.batch_mcts)
