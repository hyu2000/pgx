import os.path

import itertools
from collections import defaultdict

import numpy as np
import pandas as pd
import pickle
import platform
from typing import Iterable, List, Dict

import jax
import jax.numpy as jnp
import pgx
import equinox as eqx

from examples.alphazero.eval_util import load_cohort, fill_in_batch_mcts, ModelPolicy
from examples.alphazero.network import AZNet, create_model, load_from_ckpt, get_batch_forward_fn, batch_forward_to_policy
from examples.alphazero import mctx_search
from examples.alphazero import train_lib
from examples.alphazero.train_lib import show_game_records

from open_spiel.python.egt import alpharank
from open_spiel.python.egt import utils as egt_util
from open_spiel.python.egt import heuristic_payoff_table

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', '{:.2f}'.format)

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
        flipped = False
        if agent1 > agent2:
            agent1, agent2 = agent2, agent1
            flipped = True
        if (agent1, agent2) not in self._num_games:
            return None
        num_games, num_wins = self._num_games[agent1, agent2], self._num_wins[agent1, agent2]
        return num_games, (num_games - num_wins) if flipped else num_wins

    def add_record(self, agent1, agent2, num_games, num_wins):
        if agent1 > agent2:
            agent1, agent2, num_wins = agent2, agent1, num_games - num_wins
        self._num_wins[agent1, agent2] += num_wins
        self._num_games[agent1, agent2] += num_games

    def show(self):
        df = self.get_wrates()
        print(df)

    def get_agents(self) -> List[str]:
        pairs = self._num_wins.keys()
        ids = sorted(set(p[0] for p in pairs).union(set(p[1] for p in pairs)))
        return ids

    def get_wrates(self, agents: List[str]) -> pd.DataFrame:
        assert len(self._num_wins) == len(self._num_games)
        agents = sorted(agents)

        wrates = np.full((len(agents), len(agents)), np.nan)
        for i1, agent1 in enumerate(agents):
            for i2, agent2 in enumerate(agents):
                if agent1 >= agent2:
                    continue
                if (agent1, agent2) not in self._num_games:
                    continue
                wrate = self._num_wins[agent1, agent2] / self._num_games[agent1, agent2]
                wrates[i1, i2] = wrate
                wrates[i2, i1] = 1 - wrate

        np.fill_diagonal(wrates, 0.5)
        df_wrates = pd.DataFrame(wrates, index=agents, columns=agents)
        return df_wrates

    @staticmethod
    def shorten_names(df: pd.DataFrame, substr_to_remove: str = None):
        def shorten_id(model_id: str):
            s = model_id
            if substr_to_remove:
                s = s.replace(substr_to_remove, '')
            s = s.replace('go_5x5C2_', '')
            s = s.replace('000', '')
            return s

        long2short = {s: shorten_id(s) for s in df.index}
        return df.rename(index=long2short, columns=long2short)


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
    agents_all = payoff_table.get_agents()
    df = payoff_table.get_wrates(agents_all)

    def shorten_id(model_id: str):
        # s = model_id.replace('go_5x5C2_250917-210117/', '')
        # s = s.replace('go_5x5C2_250906-125418/000075', '0906')
        # s = s.replace('go_5x5C2_250909-160146/000140', '0909')
        s = model_id.replace('go_5x5C2_', '')
        s = s.replace('/000', '/')
        return s

    long2short = {s: shorten_id(s) for s in df.index}
    df = df.rename(index=long2short, columns=long2short)

    print('win rates')
    print(df)


def test_alpharank():
    payoff_table = PayoffTable(f'{CHECKPOINT_DIR}/payoff.pkl')
    agents_all = payoff_table.get_agents()

    # population_def = get_custom_population()
    # agents = population_def.values()
    agents = [x for x in agents_all if x.startswith('go_5x5C2_250909-160146')]
    df = payoff_table.get_wrates(agents)
    df = payoff_table.shorten_names(df, 'go_5x5C2_250909-160146')
    print('wrates - 0.5:\n', df - 0.5)

    payoff_tables = heuristic_payoff_table.from_matrix_game(df.values)
    is_symmetric_game, payoff_tables = egt_util.is_symmetric_matrix_game([payoff_tables, payoff_tables])
    assert is_symmetric_game

    # alpharank.print_results(payoff_tables, payoffs_are_hpt_format)
    (rhos, rho_m, pi, num_profiles, num_strats_per_population) = alpharank.compute(
        payoff_tables, alpha=1e2
    )
    alpharank.print_results(payoff_tables, True, pi=pi)
    sdist = pd.Series(pi, index=df.index)
    print(sdist)


def get_population_gens():
    """ eval ckpts against a cohort of top models
    22m for a population of 12

seems monotonic:
go_5x5C2_250917-210117: gen10 to gen100
go_5x5C2_250917-210117/000080   0.95
go_5x5C2_250917-210117/000090   0.02
go_5x5C2_250917-210117/000100   0.03

'go_5x5C2_250909-160146': gen100 to gen150
with 250917-210117/100, it dominates
w/o it: a little cyclic
250906-125418/075   0.19
100                 0.09
110                 0.10
120                 0.07
130                 0.01
140                 0.14
150                 0.41
w/o both, 150 dominates
ran from gen50 to gen150: 140/150 dominates
    """
    RUN_ID = 'go_5x5C2_250919-083857'
    RUN_ID = 'go_5x5C2_250917-210117'  # pure self-play, gen 0 -> 105
    RUN_ID = 'go_5x5C2_250909-160146'  # all the way to gen150

    population_def = {}
    for model_id in {
        'baseline': 'go_5x5C2_250906-125418/000075',
        # '0909gen90': 'go_5x5C2_250909-160146/000090',
        # '0909gen140': 'go_5x5C2_250909-160146/000140',
        '0917gen100': 'go_5x5C2_250917-210117/000100'
    }.values():
        population_def[model_id] = model_id
    for i_gen in range(10, 155, 20):
        model_id = f'{RUN_ID}/{i_gen:06d}'
        population_def[model_id] = model_id

    return population_def


def get_custom_population():
    """ top models from various runs: very cyclic!
wrates - 0.5:
                06-125418/075  09-160146/140  17-210117/100  20-142953/050  20-204024/100  22-151231/060
06-125418/075           0.00           0.26          -0.12           0.05          -0.07           0.03
09-160146/140          -0.26           0.00          -0.03           0.00           0.13           0.00
17-210117/100           0.12           0.03           0.00          -0.20          -0.09          -0.16
20-142953/050          -0.05           0.00           0.20           0.00           0.20           0.13  <-- almost the top model
20-204024/100           0.07          -0.13           0.09          -0.20           0.00           0.02
22-151231/060          -0.03           0.00           0.16          -0.13          -0.02           0.00
Stationary distribution (pi):
250906-125418/075   0.23
250909-160146/140   0.10
250917-210117/100   0.11
250920-142953/050   0.33
250920-204024/100   0.19
250922-151231/060   0.04
    """
    population = [
        'go_5x5C2_250906-125418/000075',  # baseline
        'go_5x5C2_250909-160146/000140',  # all the way to gen150
        'go_5x5C2_250917-210117/000100',  # pure self-play, gen 0 -> 105
        'go_5x5C2_250920-142953/000050',  # pairplay w/ 0917gen100,
        'go_5x5C2_250920-204024/000100',  # minibatch -> 128
        'go_5x5C2_250922-151231/000060',  # pairplay w/ 0917gen100, mini-batch 64
    ]
    population_def = {x: x for x in population}
    return population_def


def get_agents_matching(payoff_table: PayoffTable, prefix: str):
    agents_all = payoff_table.get_agents()
    agents = [x for x in agents_all if x.startswith(prefix)]
    return {x: x for x in agents}


def test_get_agents():
    payoff_table = PayoffTable(f'{CHECKPOINT_DIR}/payoff.pkl')
    p_def = get_agents_matching(payoff_table, 'go_5x5C2_250909')
    print(p_def.values())


def test_eval_population():
    """ run pair-wise eval on population, save results to payoff table

pytest -s tests/test_go_gamescape.py::test_eval_population 2>&1 | tee /content/drive/MyDrive/dlgo/pgx/eval-gs0909-all.log
    """
    env = pgx.make("go_5x5C2")
    key = jax.random.PRNGKey(0)
    num_simulations = 32
    num_eval_games = 128

    payoff_table = PayoffTable(f'{CHECKPOINT_DIR}/payoff.pkl')
    population_def = get_agents_matching(payoff_table, 'go_5x5C2_250909')
    # population_def = get_population_gens()
    # population_def = get_custom_population()
    print('population: ', population_def.keys())

    population = load_cohort(population_def, CHECKPOINT_DIR)
    fill_in_batch_mcts(population, env, num_simulations)

    for player1, player2 in itertools.combinations(population.values(), 2):
        if payoff_table.match_record(player1.model_id, player2.model_id) is not None:
            print(f'Skipping {player1.model_id} vs {player2.model_id}, already in payoff table')
            continue
        player_names = (player1.name, player2.name)
        R, game_records = train_lib.evaluate(env, key, num_eval_games, player1.batch_mcts_policy, player2.batch_mcts_policy)
        wrate = (1 + sum(R) / len(R)) * 0.5

        print(f'eval {player_names}: total {len(R)} games, win-rate=', wrate)
        show_game_records(game_records, env, player_names)

        payoff_table.add_record(player1.name, player2.name, len(R), sum(R > 0))

    payoff_table.save()



