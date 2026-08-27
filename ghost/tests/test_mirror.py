"""test_mirror.py — the solo-MLB mirror with an external (scripted "human") opponent.

Real engine, real EVPlayer (ev:fast), oracle seed — no IPC involved.  The scripted human
here just picks finals relative to the agent's, which exercises every resolution branch:
human loses, agent loses (+ comeback money), exact tie (nobody), and agent death.
"""
from __future__ import annotations

import pytest

from .._bootstrap import State
from ..mirror import MirrorAgent, MirrorDead

SEED = "7I4M53DL"       # oracle ground-truth seed (live-confirmed in the real game)


@pytest.fixture(scope="module")
def first_round():
    m = MirrorAgent(SEED)
    rec = m.advance_to_nemesis()
    return m, rec


def test_first_nemesis_shape(first_round):
    m, rec = first_round
    assert rec["ante"] >= 2                             # MLB_PVP_START_ROUND
    assert rec["hands"], "agent played no hands at its Nemesis"
    scores = [h["score"] for h in rec["hands"]]
    assert scores == sorted(scores)                     # cumulative round score
    assert rec["final"] == scores[-1] > 0
    assert m.game.state is State.PVP_WAIT               # parked until resolve()
    assert m.awaiting_final and not m.dead


def test_double_advance_refused(first_round):
    m, _ = first_round
    with pytest.raises(RuntimeError, match="resolve"):
        m.advance_to_nemesis()


def test_full_match_all_branches():
    m = MirrorAgent(SEED)

    # round 1: human falls short -> human loses the round, agent keeps its lives
    # (human_lives is the POST-resolution value the mod reports)
    rec = m.advance_to_nemesis()
    res = m.resolve(rec["final"] - 1, human_lives=3)
    assert res["loser"] == "human"
    assert m.lives == 4 and m.opp.lives == 3
    assert m.pvp_log[-1] == (rec["ante"], 1, rec["final"], rec["final"] - 1)

    # round 2: exact tie -> nobody loses (server rule, not the ghost-mode >=)
    rec2 = m.advance_to_nemesis()
    assert rec2["ante"] > rec["ante"]
    res2 = m.resolve(rec2["final"])
    assert res2["loser"] is None
    assert m.lives == 4 and m.opp.lives == 3

    # round 3: human wins -> agent loses a life AND banks comeback money in the same
    # cash out ($5 Nemesis reward is paid win or lose, so the delta must exceed it)
    rec3 = m.advance_to_nemesis()
    money_before = m.money
    res3 = m.resolve(rec3["final"] + 1)
    assert res3["loser"] == "agent"
    assert m.lives == 3
    assert m.money - money_before >= 5 + 4              # $5 nemesis + $4 x 1 comeback

    # grind the agent down: it must die cleanly at 0 lives
    while not m.dead:
        rec_n = m.advance_to_nemesis()
        m.resolve(rec_n["final"] + 1)
    assert m.lives == 0
    with pytest.raises(MirrorDead):
        m.advance_to_nemesis()


def test_deterministic_in_seed_and_spec():
    a = MirrorAgent(SEED).advance_to_nemesis()
    b = MirrorAgent(SEED).advance_to_nemesis()
    assert a == b


def test_resolve_without_pending_refused():
    m = MirrorAgent(SEED)
    with pytest.raises(RuntimeError, match="pending"):
        m.resolve(100)
