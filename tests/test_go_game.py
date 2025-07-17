from pgx._src.games.go import Game, GameState


def test_simple():
    game = Game(5, 0.5)
    state0 = game.init()
    assert state0.color == 0  # color is just step_count % 2
    state1 = game.step(state0, 1)
    assert state1.color == 1
    assert state1.step_count == 1
    state2 = game.step(state1, 2)
    assert state2.step_count == 2

    assert(sum(game.legal_action_mask(state0)) == 26)
    assert(sum(game.legal_action_mask(state1)) == 25)

    rewards = [game.rewards(s) for s in (state1, state0, state2)]
    print(rewards)
    assert(all(x[0] == 0 and x[1] == 0 for x in rewards))

    statef = state2
    statef = game.step(statef, 25)
    statef = game.step(statef, 25)
    reward = game.rewards(statef)
    # white wins on komi
    print(reward)
    assert(reward[0] == -1 and reward[1] == 1)


def test_observation():
    game = Game(5, 0.5)
    state0 = game.init()
    state1 = game.step(state0, 1)
    assert state1.color == 1  # white
    obs = game.observe(state1)
    obs0 = game.observe(state1, color=0)
    assert (obs != obs0).any()
