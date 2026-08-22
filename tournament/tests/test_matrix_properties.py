"""Matrix algebra properties, on synthetic scores AND on a real (small) Tournament run.
Antisymmetry, zero diagonal, tie <-> outcome==0 symmetry, rank consistent with scores."""
import math

import numpy as np

from ..matrix import outcome_matrix, log_margin_matrix, population_rank, tie_fraction, AnteMatrix
from ..players import default_population
from ..runner import Tournament


def test_outcome_matrix_synthetic():
    scores = np.array([10.0, 20.0, 20.0, np.nan, 5.0])
    m = outcome_matrix(scores)
    assert m.shape == (5, 5)
    # zero diagonal
    assert np.all(np.diag(m) == 0)
    # antisymmetric
    assert np.array_equal(m, -m.T)
    # 0 beats 4 (10 > 5): row0,col4 = +1
    assert m[0, 4] == 1
    assert m[4, 0] == -1
    # 1 ties 2 (both 20)
    assert m[1, 2] == 0
    assert m[2, 1] == 0
    # anyone vs the NaN agent (index 3) is 0 both ways
    assert m[0, 3] == 0 and m[3, 0] == 0


def test_log_margin_matrix_synthetic():
    scores = np.array([10.0, 20.0, np.nan])
    m = log_margin_matrix(scores)
    assert np.all(np.diag(m) == 0)
    assert np.allclose(m, -m.T, equal_nan=True)
    assert math.isnan(m[0, 2]) and math.isnan(m[2, 0])
    assert m[0, 1] < 0 < m[1, 0]   # agent 0 scored less -> negative log margin vs agent 1


def test_population_rank_ties_share_average_rank():
    scores = np.array([100.0, 50.0, 50.0, 10.0, np.nan])
    r = population_rank(scores)
    assert r[0] == 1.0            # sole highest
    assert r[1] == r[2] == 2.5    # tied for 2nd/3rd -> average rank 2.5
    assert r[3] == 4.0
    assert math.isnan(r[4])       # absent agent has no rank


def test_tie_fraction_all_distinct_vs_all_identical():
    distinct = np.array([1.0, 2.0, 3.0, 4.0])
    identical = np.array([7.0, 7.0, 7.0, 7.0])
    out_distinct = outcome_matrix(distinct)
    out_identical = outcome_matrix(identical)
    mask = np.array([True, True, True, True])
    assert tie_fraction(out_distinct, mask) == 0.0
    assert tie_fraction(out_identical, mask) == 1.0


def test_tie_fraction_nan_with_fewer_than_two_present():
    out = outcome_matrix(np.array([1.0, np.nan, np.nan]))
    mask = np.array([True, False, False])
    assert math.isnan(tie_fraction(out, mask))


def test_ante_matrix_build_matches_manual_scores():
    scores_by_agent = {0: 100, 1: 50, 2: 50, 3: 10}
    am = AnteMatrix.build(ante=3, n_agents=5, scores_by_agent=scores_by_agent, losers={3})
    assert am.ante == 3
    assert am.losers == [3]
    assert am.stats["n_present"] == 4
    assert am.outcome[0, 1] == 1 and am.outcome[1, 0] == -1
    assert am.outcome[1, 2] == 0   # tie
    assert math.isnan(am.scores[4])   # agent 4 never appeared


def _real_ante_matrix():
    n = 8
    players = default_population(n, base_seed=42)
    t = Tournament(seed="7I4M53DL", n_agents=n, players=players, life_rule="none", max_ante=4)
    res = t.run()
    return res.ante_matrices


def test_properties_hold_on_a_real_run():
    matrices = _real_ante_matrix()
    assert matrices, "expected at least one Nemesis to be played"
    for m in matrices:
        n = m.n_agents
        assert np.array_equal(m.outcome, -m.outcome.T)
        assert np.all(np.diag(m.outcome) == 0)
        assert np.all(np.diag(m.log_margin) == 0)
        assert np.allclose(m.log_margin, -m.log_margin.T, equal_nan=True)
        present = ~np.isnan(m.scores)
        # rank consistency: for every present pair, the strictly-higher score has the
        # strictly-lower (better) rank number
        idx = np.where(present)[0]
        for i in idx:
            for j in idx:
                if m.scores[i] > m.scores[j]:
                    assert m.rank[i] <= m.rank[j]
                if m.scores[i] == m.scores[j]:
                    assert m.rank[i] == m.rank[j]
        assert 0.0 <= m.tie_fraction <= 1.0 or math.isnan(m.tie_fraction)
