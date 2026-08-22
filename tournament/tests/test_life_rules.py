"""The three life_rule policies (brief §0.3): pure function tests against fabricated scores
(fast, exhaustive) + one end-to-end integration check per rule that a real Tournament run
actually applies the intended lives.  "A tie costs nobody" and "paired: exactly one life per
lost pairing" are asserted explicitly."""
import statistics

import pytest

from ..runner import Tournament, _pairing
from ..players import ScriptedPlayer, ScriptedPlayerAdapter, OPENER


class _Dummy:
    """A bare object carrying only what Tournament._decide_losers reads, so the pure
    decision logic can be tested without constructing engine games."""
    _decide_losers = Tournament._decide_losers

    def __init__(self, n_agents, life_rule):
        self.n_agents = n_agents
        self.life_rule = life_rule
        self._pairs, self._singles = _pairing(n_agents)


def test_none_rule_never_produces_losers():
    d = _Dummy(6, "none")
    scores = {i: i * 7 for i in range(6)}
    assert d._decide_losers(list(range(6)), scores, 0) == set()
    # even a totally lopsided score spread loses nobody
    scores2 = {0: 0, 1: 1_000_000}
    assert _Dummy(2, "none")._decide_losers([0, 1], scores2, 0) == set()


def test_median_rule_strictly_below_median_loses_ties_at_median_survive():
    d = _Dummy(5, "median")
    scores = {0: 100, 1: 50, 2: 50, 3: 50, 4: 10}   # sorted: 10,50,50,50,100 -> median=50
    losers = d._decide_losers([0, 1, 2, 3, 4], scores, 0)
    assert losers == {4}    # only strictly below the median
    assert statistics.median(scores.values()) == 50


def test_median_rule_all_equal_scores_costs_nobody():
    d = _Dummy(4, "median")
    scores = {i: 42 for i in range(4)}
    assert d._decide_losers(list(range(4)), scores, 0) == set()


def test_paired_rule_exactly_one_loser_per_decided_pairing_tie_costs_nobody():
    d = _Dummy(6, "paired")
    assert d._pairs == [(0, 1), (2, 3), (4, 5)]
    assert d._singles == []
    scores = {0: 10, 1: 20, 2: 5, 3: 5, 4: 99, 5: 1}
    losers = d._decide_losers(list(range(6)), scores, 0)
    # (0,1): 0 loses.  (2,3): tie, nobody.  (4,5): 5 loses.
    assert losers == {0, 5}
    for i, j in d._pairs:
        assert not ({i, j} <= losers)   # never BOTH members of one pairing


def test_paired_rule_skips_a_pairing_when_one_side_already_died():
    """Agent 1 died in an earlier round (dropped from `alive`/`scores`); agent 0's fixed
    pairing has no opponent to compare against this round, so agent 0 gets a "bye" (no life
    change from pairing) -- the OTHER pairing (2,3), both still alive, resolves normally."""
    d = _Dummy(4, "paired")
    alive = [0, 2, 3]              # agent 1 already eliminated
    scores = {0: 50, 2: 5, 3: 999}  # agent 1 absent
    losers = d._decide_losers(alive, scores, 0)
    assert losers == {2}           # (0,1) unresolved: 1 absent -> no comparison; (2,3): 2 loses


def test_paired_rule_odd_agent_out_rotates_deterministically():
    """Isolate the rotating comparison from the fixed pairings by making both fixed pairs
    tie every round (0==1, 2==3): any loser that appears can only have come from agent 4's
    rotating comparison, whose partner must then cycle 0,1,2,3 over four rounds.  (Note: the
    rotating partner is drawn from ALL other agents, so on some rounds it coincides with an
    agent that ALSO has its own fixed pairing that round -- documented in TOURNAMENT_NOTES.md,
    not something this isolated setup needs to worry about.)"""
    d = _Dummy(5, "paired")
    assert d._pairs == [(0, 1), (2, 3)]
    assert d._singles == [4]
    scores = {0: 50, 1: 50, 2: 5, 3: 5, 4: 100}
    seen_partners = []
    for round_idx in range(4):
        losers = d._decide_losers([0, 1, 2, 3, 4], scores, round_idx)
        assert len(losers) == 1   # the fixed pairs tie; only the rotating comparison decides
        seen_partners.append(losers)
    # four different rotating partners over four rounds (agents 0,1,2,3 each exactly once)
    assert seen_partners == [{0}, {1}, {2}, {3}]


def test_paired_rule_single_agent_does_not_crash():
    d = _Dummy(1, "paired")
    assert d._decide_losers([0], {0: 5}, 0) == set()


def test_unknown_life_rule_rejected_at_construction():
    with pytest.raises(ValueError):
        Tournament(seed="7I4M53DL", n_agents=2,
                   players=[ScriptedPlayerAdapter(OPENER), ScriptedPlayerAdapter(OPENER)],
                   life_rule="bogus")


# ── integration: a real Tournament actually applies the rule ──────────────────────────

def test_integration_paired_rule_kills_the_weaker_of_a_pair():
    """``debug_win_regular=True`` clears every non-Nemesis blind for free (MLB_NOTES.md's
    harness hook, touches no stream) so BOTH players survive Small/Big/ante-1-Boss
    regardless of skill, isolating the paired life mechanism to what it is meant to measure:
    who plays the Nemesis itself better.  "weak" (first legal one-card play, no discard
    optimisation) should lose every one of these Nemeses to "greedy"."""
    weak = ScriptedPlayerAdapter(ScriptedPlayer(name="weak", hand="weak", debug_win_regular=True))
    strong = ScriptedPlayerAdapter(ScriptedPlayer(name="strong", hand="greedy", debug_win_regular=True))
    t = Tournament(seed="7I4M53DL", n_agents=2, players=[weak, strong], life_rule="paired",
                   max_ante=8, lives=20)
    res = t.run()
    assert len(res.ante_matrices) == 7   # antes 2..8, both survive the regular blinds for free
    assert all(m.losers == [0] for m in res.ante_matrices)   # weak loses every single Nemesis
    assert res.final_lives[0] == 20 - len(res.ante_matrices)
    assert res.final_lives[1] == 20


def test_integration_none_rule_nobody_ever_dies_and_reaches_max_ante():
    n = 5
    players = [ScriptedPlayerAdapter(ScriptedPlayer(name=f"p{i}", hand="weak"))
               for i in range(n)]
    t = Tournament(seed="7I4M53DL", n_agents=n, players=players, life_rule="none", max_ante=6)
    res = t.run()
    assert len(res.ante_matrices) == 5   # antes 2..6
    assert res.alive_at_end == list(range(n))
    assert all(l > 0 for l in res.final_lives)
    assert all(m.losers == [] for m in res.ante_matrices)
