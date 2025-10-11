import os.path

import itertools
from collections import defaultdict

import pickle
import platform
from typing import Iterable

import jax
import jax.numpy as jnp
import pgx
import equinox as eqx

from examples.alphazero.eval_util import load_cohort, fill_in_batch_mcts, ModelPolicy

from examples.alphazero.network import AZNet, create_model, load_from_ckpt, get_batch_forward_fn, batch_forward_to_policy
from examples.alphazero import mctx_search
from examples.alphazero import train_lib
from examples.alphazero.train_lib import show_game_records

CHECKPOINT_DIR = '/Users/hyu/PycharmProjects/pgx/examples/alphazero/checkpoints' if platform.system() == 'Darwin' else '/content/drive/MyDrive/dlgo/pgx'


def load_go5_checkpoint_eqx(fpath = None):
    if not fpath:
        fpath = f'go_5x5C2_250903-143719/000210.ckpt'
        fpath = 'go_5x5C2_250906-125418/000075.ckpt'  # mini-batch new baseline
    if not fpath.startswith('/'):
        fpath = f'{CHECKPOINT_DIR}/{fpath}'
    if not fpath.endswith('.ckpt'):
        fpath = f'{fpath}.ckpt'
    model_params, model_state = load_from_ckpt(fpath)
    model_params = eqx.nn.inference_mode(model_params)
    batch_forward = get_batch_forward_fn(model_params, model_state)
    return batch_forward, model_params, model_state


class PayoffTable:
    """ a global payoff table between different models
    It's anti-symmetric.

    agent-id: <run-id>/<gen>#<num_simu>,  e.g. go_5x5C2_250919-145526/000100#32
    (agent1, agent2): num_games, wrate

    """
    def __init__(self, fname: str):
        self._fname = fname
        if os.path.isfile(self._fname):
            self.load()
        else:
            self._num_wins = defaultdict(PayoffTable.default_0)
            self._num_games = defaultdict(PayoffTable.default_0)

    @staticmethod
    def default_0():
        return 0

    def load(self):
        with open(self._fname, 'rb') as f:
            self._num_games, self._num_wins = pickle.load(f)

    def save(self):
        with open(self._fname, 'wb') as f:
            pickle.dump((self._num_games, self._num_wins), f)

    def match_record(self, agent1: str, agent2: str):
        """ """
        if agent1 > agent2:
            agent1, agent2 = agent2, agent1
        if (agent1, agent2) not in self._num_games:
            return None
        return self._num_games[agent1, agent2], self._num_wins[agent1, agent2]

    def add_record(self, agent1, agent2, num_games, num_wins):
        if agent1 > agent2:
            agent1, agent2, num_wins = agent2, agent1, num_games - num_wins
        self._num_wins[agent1, agent2] += num_wins
        self._num_games[agent1, agent2] += num_games

    def show(self):
        pass


def test_payoff():
    payoff = PayoffTable('/tmp/payoff.pkl')
    assert payoff.match_record('a', 'b') == (0, 0)
    payoff.add_record('a', 'b', 6, 4)
    assert payoff.match_record('a', 'b') == (6, 4)
    payoff.save()

    payoff2 = PayoffTable('/tmp/payoff.pkl')
    assert payoff2.match_record('a', 'b') == (6, 4)
    payoff2.add_record('b', 'a', 1, 1)
    assert payoff2.match_record('a', 'b') == (7, 4)


def test_show_payoff():
    payoff_table = PayoffTable(f'{CHECKPOINT_DIR}/payoff.pkl')
    print(payoff_table._num_games)
    print(payoff_table._num_wins)


def test_eval_gens():
    """ eval ckpts against a cohort of top models """
    env = pgx.make("go_5x5C2")
    key = jax.random.PRNGKey(0)

    RUN_ID = 'go_5x5C2_250919-083857'
    RUN_ID = 'go_5x5C2_250917-210117'  # pure self-play, gen 0 -> 105
    num_simulations = 32
    num_eval_games = 128

    population_def = {}
    for model_id in {
        'baseline': 'go_5x5C2_250906-125418/000075',
        # '0909gen90': 'go_5x5C2_250909-160146/000090',
        '0909gen140': 'go_5x5C2_250909-160146/000140',
        # '0917gen100': 'go_5x5C2_250917-210117/000100'
    }.values():
        population_def[model_id] = model_id
    for i_gen in range(10, 105, 10):
        model_id = f'{RUN_ID}/{i_gen:06d}'
        population_def[model_id] = model_id

    population = load_cohort(population_def, CHECKPOINT_DIR)
    fill_in_batch_mcts(population, env, num_simulations)

    payoff_table = PayoffTable(f'{CHECKPOINT_DIR}/payoff.pkl')
    for player1, player2 in itertools.combinations(population.values(), 2):
        if payoff_table.match_record(player1.model_id, player2.model_id) is not None:
            continue
        player_names = (player1.name, player2.name)
        R, game_records = train_lib.evaluate(env, key, num_eval_games, player1.batch_mcts_policy, player2.batch_mcts_policy)
        wrate = (1 + sum(R) / len(R)) * 0.5

        print(f'eval {player_names}: total {len(R)} games, win-rate=', wrate)
        show_game_records(game_records, env, player_names)

        payoff_table.add_record(player1.name, player2.name, len(R), sum(R > 0))

    payoff_table.save()



