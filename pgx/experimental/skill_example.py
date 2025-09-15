import random
from typing import Dict

import math
import trueskill
from statistics import NormalDist
from collections import defaultdict, Counter

# Setup TrueSkill environment
env = trueskill.TrueSkill(draw_probability=0.0)

# Global ratings
ratings = {}  # type: Dict[str, trueskill.Rating]
match_history = Counter()


# handy z lookup, falls back to 1.96 if SciPy not available or user picks odd conf
Z_LOOKUP = {0.90: 1.6448536269514722,
            0.95: 1.959963984540054,
            0.99: 2.5758293035489004}

def z_for_conf(conf):
    try:
        from scipy.stats import norm
        return norm.ppf((1 + conf) / 2)
    except Exception:
        return Z_LOOKUP.get(conf, 1.96)

def wilson_interval(wins, n, conf=0.95):
    """Return (low, high) Wilson score interval for wins/n."""
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    z = z_for_conf(conf)
    denom = 1 + (z * z) / n
    center = (p + (z * z) / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) / n) + (z * z) / (4 * n * n))) / denom
    return max(0.0, center - half), min(1.0, center + half)


def add_model(name):
    """Add a new model with default rating"""
    ratings[name] = env.Rating()


def play_game(model_a, model_b) -> bool:
    """
    Simulate a stochastic outcome between two models.
    Replace with your actual game runner.

    :returns True if a wins
    """
    # Example: assign fixed underlying "true strengths"
    strengths = {"anchor_easy": 0.3, "anchor_mid": 0.5, "anchor_strong": 0.7}
    sa = strengths.get(model_a, 0.6)  # default new models start around 0.6
    sb = strengths.get(model_b, 0.6)
    prob_a_wins = sa / (sa + sb)
    return random.random() < prob_a_wins


def update_rating_1vs1(model_a, model_b, win_a: bool):
    if win_a:
        model_winner, model_loser = model_a, model_b
    else:
        model_winner, model_loser = model_b, model_a
    ratings[model_winner], ratings[model_loser] = env.rate_1vs1(ratings[model_winner], ratings[model_loser])
    match_history[(model_winner, model_loser)] += 1


def adaptive_match(model_a, model_b, max_games=50, min_games=10, conf=0.95):
    """Run adaptive evaluation between two models."""
    wins_a, wins_b = 0, 0
    games_played = 0
    ci = None

    while games_played < max_games:
        games_played += 1
        a_wins = play_game(model_a, model_b)

        update_rating_1vs1(model_a, model_b, a_wins)
        if a_wins:
            wins_a += 1
        else:
            wins_b += 1

        if games_played >= min_games:
            total = wins_a + wins_b
            winrate_a = wins_a / total
            # ci = NormalDist.from_samples([1] * wins_a + [0] * wins_b).confidence_interval(conf)
            ci_low, ci_high = wilson_interval(wins_a, total, conf=conf)
            ci = (f'{ci_low:.2f}', f'{ci_high:.2f}')

            if ci_low > 0.5:
                return f"{model_a} stronger", winrate_a, games_played, ci
            elif ci_high < 0.5:
                return f"{model_b} stronger", winrate_a, games_played, ci

    return "Unclear", wins_a / (wins_a + wins_b), games_played, ci


def evaluate_new_model(new_model, predecessor, anchors):
    """Evaluate new model against predecessor + anchors"""
    results = {}
    # vs predecessor
    results[predecessor] = adaptive_match(new_model, predecessor)
    # vs anchors
    for anchor in anchors:
        results[anchor] = adaptive_match(new_model, anchor)
    return results

def leaderboard():
    """Return models sorted by TrueSkill mean"""
    return sorted(ratings.items(), key=lambda x: env.expose(x[1]), reverse=True)


def show_leaderboard():
    for name, rating in leaderboard():
        print(f"{name}: {rating}")


def test_simple():
    add_model("anchor_easy")
    add_model("anchor_mid")
    add_model("anchor_strong")

    show_leaderboard()
    for i in range(20):
        win_a = play_game('anchor_easy', 'anchor_mid')
        update_rating_1vs1('anchor_easy', 'anchor_mid', win_a)
    show_leaderboard()


def test_pairwise():
    """ same as Elo, trueskill assumes a transitive skill ladder
    Guess it'll be confused by circular players
    """
    ratings['easy'] = env.Rating(20.926, sigma=2.173)
    ratings['mid']  = env.Rating(mu=26.533, sigma=2.022)
    add_model('model1')
    add_model('model2')
    for i in range(10):
        update_rating_1vs1('model1', 'easy', True)
        update_rating_1vs1('model2', 'mid', True)
    show_leaderboard()


def test_main_example():
    # Add anchor models
    add_model("anchor_easy")
    add_model("anchor_mid")
    add_model("anchor_strong")

    # Add first model
    add_model("model_1")

    # Evaluate model_1 vs anchors
    print("Evaluating model_1:")
    print(evaluate_new_model("model_1", "anchor_easy", ["anchor_mid", "anchor_strong"]))

    # Add model_2 (trained successor of model_1)
    add_model("model_2")

    print("Evaluating model_2:")
    print(evaluate_new_model("model_2", "model_1", ["anchor_mid", "anchor_strong"]))

    # Show leaderboard
    print("\nLeaderboard:")
    show_leaderboard()

    print(match_history)