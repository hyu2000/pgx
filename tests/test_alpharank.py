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
    is_symmetric_game, payoff_tables = utils.is_symmetric_matrix_game(payoff_tables)
    assert is_symmetric_game

    # Compute Alpha-Rank
    (rhos, rho_m, pi, num_profiles, num_strats_per_population) = alpharank.compute(
        payoff_tables, alpha=1e2
    )

    # Report results
    payoffs_are_hpt_format = True
    alpharank.print_results(payoff_tables, payoffs_are_hpt_format, pi=pi)


def get_table_9_AG():
    """ table 9 from AlaphGo 2016 paper """
    names = 'rvp,vp,rp,rv,r,v,p,CS,ZN,PC,FG,GG,CS4,ZN4,PC4'.split(',')
    num_agents = len(names)
    arr_table_9 = np.array([
        [-1,  1, 5, 0, 0, 0, 0],
        [99, -1, 61, 35, 6, 0, 1],
        [95, 39, -1, 13, 0, 0, 4],
        [100, 65, 87, -1, 0, 29, 48],
        [100, 94, 100, 100, -1, 78, 78],
        [100, 100, 100, 71, 22, -1, 30],
        [100,  99,  96, 52, 22, 70, -1],
        [100,  74, 98, 80, 5, 36, 8],
        [99, 84, 98, 92, 6, 40, 100],
        [100, 99, 100, 98, 78, 87, 55],
        [100, 99, 100, 100, 78, 100, 65],
        [100, 100, 100, 100, 99, 67, 99],
        [77, 12, 53, 15, 0, 0, 0],
        [86, 25, 67, 14, 0, 0, np.nan],
        [99, 82, 98, 89, 32, 13, 35],
    ])
    N, M = arr_table_9.shape
    assert N == num_agents
    arr_full = np.zeros((num_agents, num_agents))
    arr_full[:, :arr_table_9.shape[1]] = arr_table_9
    # we don't have matches among non-AG agents
    arr_full[M:, M:] = np.nan
    arr_full = np.tril(arr_full)
    arr_full = arr_full + (np.tril(100 * np.ones((num_agents, num_agents))) - arr_full).T
    # payoff matrix should be row-centric, rather than column-centric as in the original table
    arr_full = arr_full.T
    np.fill_diagonal(arr_full, 50)

    df_winrate_pctg = pd.DataFrame(arr_full,
        index=names,
        columns=names
    )
    return df_winrate_pctg


def test_AlphaGo_data():
    df = get_table_9_AG()
    print()
    print(df)
    # verify matrix is anti-symmetry
    df_total = df + df.T
    assert all(df_total == 100)
    print(df.iloc[range(7), range(7)])

    strats_of_interest = 'rvp,vp,rp'.split(',')
    print(df.loc[strats_of_interest, strats_of_interest])


def test_AG_examples():
    df = get_table_9_AG()  # - 50
    strats_of_interest = 'rvp,vp,rp'.split(',')   # rvp dominates
    strats_of_interest = 'v,p,ZN'.split(',')  # circular
    df = df.loc[strats_of_interest, strats_of_interest]
    print()
    print(df)
    payoff_tables = heuristic_payoff_table.from_matrix_game(df.values)
    print(payoff_tables)
    is_symmetric_game, payoff_tables = utils.is_symmetric_matrix_game([payoff_tables, payoff_tables])
    assert is_symmetric_game
    print(type(payoff_tables[0]))

    payoffs_are_hpt_format = True
    # alpharank.print_results(payoff_tables, payoffs_are_hpt_format)

    (rhos, rho_m, pi, num_profiles, num_strats_per_population) = alpharank.compute(
        payoff_tables, alpha=1
    )

    alpharank.print_results(payoff_tables, payoffs_are_hpt_format, pi=pi)

