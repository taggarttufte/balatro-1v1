"""Same seed + same players -> identical N x N matrices and identical final game
state_signature()s.  Covers both fan-out methods and both a scripted and a random-legal
population (the random player is the one most likely to leak state across runs if
``reset()`` were not honoured)."""
import numpy as np

from ..players import default_population, RandomLegalPlayer, ScriptedPlayerAdapter, OPENER
from ..runner import Tournament


def _run(seed, n, players, life_rule, max_ante, fanout):
    t = Tournament(seed=seed, n_agents=n, players=players, life_rule=life_rule,
                    max_ante=max_ante, fanout=fanout)
    res = t.run()
    return res, t._last_games


def _assert_same_result(res1, games1, res2, games2):
    assert res1.seed == res2.seed
    assert len(res1.ante_matrices) == len(res2.ante_matrices)
    for m1, m2 in zip(res1.ante_matrices, res2.ante_matrices):
        assert m1.ante == m2.ante
        assert np.array_equal(m1.scores, m2.scores, equal_nan=True)
        assert np.array_equal(m1.outcome, m2.outcome, equal_nan=True)
        assert np.array_equal(m1.log_margin, m2.log_margin, equal_nan=True)
        assert np.array_equal(m1.rank, m2.rank, equal_nan=True)
        assert m1.losers == m2.losers
        assert m1.tie_fraction == m2.tie_fraction or (
            np.isnan(m1.tie_fraction) and np.isnan(m2.tie_fraction))
    assert res1.final_lives == res2.final_lives
    assert res1.alive_at_end == res2.alive_at_end
    for g1, g2 in zip(games1, games2):
        assert g1.state_signature() == g2.state_signature()


def test_determinism_scripted_population_clone_fanout():
    n = 8
    p1 = default_population(n, base_seed=11)
    p2 = default_population(n, base_seed=11)
    res1, g1 = _run("7I4M53DL", n, p1, "paired", 6, "clone")
    res2, g2 = _run("7I4M53DL", n, p2, "paired", 6, "clone")
    _assert_same_result(res1, g1, res2, g2)


def test_determinism_scripted_population_construct_fanout():
    n = 6
    p1 = default_population(n, base_seed=3)
    p2 = default_population(n, base_seed=3)
    res1, g1 = _run("ALEEB", n, p1, "median", 5, "construct")
    res2, g2 = _run("ALEEB", n, p2, "median", 5, "construct")
    _assert_same_result(res1, g1, res2, g2)


def test_determinism_random_legal_population():
    n = 10
    p1 = [RandomLegalPlayer(seed=i) for i in range(n)]
    p2 = [RandomLegalPlayer(seed=i) for i in range(n)]
    res1, g1 = _run("7I4M53DL", n, p1, "none", 5, "clone")
    res2, g2 = _run("7I4M53DL", n, p2, "none", 5, "clone")
    _assert_same_result(res1, g1, res2, g2)


def test_determinism_survives_reusing_the_same_player_objects():
    """Tournament.run() must reset() every player, so running the SAME Tournament twice (or
    two Tournaments sharing player objects) reproduces the same result -- RandomLegalPlayer
    is the one that mutates internal state (its RNG) across a run."""
    n = 5
    players = [RandomLegalPlayer(seed=i) for i in range(n)] + \
        [ScriptedPlayerAdapter(OPENER) for _ in range(0)]
    t = Tournament(seed="7I4M53DL", n_agents=n, players=players, life_rule="none", max_ante=4)
    res_a = t.run()
    res_b = t.run()   # same Tournament object, same player objects, second call
    for ma, mb in zip(res_a.ante_matrices, res_b.ante_matrices):
        assert np.array_equal(ma.scores, mb.scores, equal_nan=True)
