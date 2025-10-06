import numpy as np
from open_spiel.python.algorithms import fictitious_play
from open_spiel.python.egt import alpharank
from open_spiel.python.egt import alpharank_visualizer
from open_spiel.python.egt import utils
from open_spiel.python.egt import heuristic_payoff_table
import pyspiel
import pandas as pd


def test_rps_example():
    # Load the game
    game = pyspiel.load_matrix_game("matrix_rps")
    payoff_tables = utils.game_payoffs_array(game)
    print('payoff tables:\n', payoff_tables.shape)
    print(payoff_tables)
    # affine transform doesn't seem to matter
    payoff_tables = payoff_tables * 0.5 + 1

    # Convert to heuristic payoff tables
    payoff_tables = [heuristic_payoff_table.from_matrix_game(payoff_tables[0]),
                     heuristic_payoff_table.from_matrix_game(payoff_tables[1].T)]

    # Check if the game is symmetric (i.e., players have identical strategy sets
    # and payoff tables) and return only a single-player’s payoff table if so.
    # This ensures Alpha-Rank automatically computes rankings based on the
    # single-population dynamics.
    _, payoff_tables = utils.is_symmetric_matrix_game(payoff_tables)

    # Compute Alpha-Rank
    (rhos, rho_m, pi, num_profiles, num_strats_per_population) = alpharank.compute(
        payoff_tables, alpha=1e2
    )

    # Report results
    payoffs_are_hpt_format = True
    alpharank.print_results(payoff_tables, payoffs_are_hpt_format, pi=pi)


def get_table_9_AG():
    """ table 9 from AlaphGo 2016 paper """
    names = 'rvp,vp,rp,rv,r,v,p'.split(',')
    df_winrate_pctg = pd.DataFrame([
        [-1,  1, 5, 0, 0, 0, 0],
        [99, -1, 61, 35, 6, 0, 1],
        [95, 39, -1, 13, 0, 0, 4],
        [100, 65, 87, -1, 0, 29, 48],
        [100, 94, 100, 100, -1, 78, 78],
        [100, 100, 100, 71, 22, -1, 30],
        [100,  99,  96, 52, 22, 70, -1]],
        index=names,
        columns=names
    )
    df_winrate_pctg.iloc[range(7), range(7)] = 50
    df = df_winrate_pctg
    return df


def test_AlphaGo_data():
    df = get_table_9_AG()
    print()
    print(df)
    df_total = df + df.T
    assert all(df_total == 100)
    print(df.iloc[range(7), range(7)])


def test_AG_example():
    df = get_table_9_AG()
    payoff_tables = heuristic_payoff_table.from_matrix_game(df)
    print(payoff_tables)
