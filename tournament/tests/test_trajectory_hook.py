"""
`Tournament`'s trajectory hooks (Phase 4 W2, for W3's `replay`; REPLAY_NOTES.md §2.3).

The runner is the only place that knows every mutation an agent's game undergoes:
the agent's own actions, the no-progress guard's forced action, `_cash_out`'s advance, and
the cross-agent life rule — which is the ONE life lost without a `step()` and therefore the
one thing a replay could not reconstruct unless the runner says so.

So there are three hooks, and this file pins their contract:

    on_fanout(games, seed_str)             once, the instant the N games exist
    on_step(agent_idx, game, action)       AFTER every step this module performs
    on_agent_done(agent_idx, game, reason) exactly once per agent, and for an eliminated
                                           agent BEFORE `state = GAME_OVER` (an out-of-band
                                           mutation no replay reproduces)

`test_replays_a_logged_tournament` is the end-to-end proof: a real trajectory log written
through these hooks replays bit-exactly.
"""
from __future__ import annotations

import pytest

from tournament.bootstrap import BalatroGame, State
from tournament.players import default_population, make_scripted
from tournament.runner import OP_LOSE_LIFE, Tournament

SEED = "7I4M53DL"


def _population(n):
    # "median" + identical-ish players guarantees somebody loses lives every round, so the
    # synthetic op is actually exercised.
    return [make_scripted(name=f"s{i}", hand="greedy", rerolls_per_visit=i % 3,
                          buy_slot0=bool(i % 2)) for i in range(n)]


def _run(n=4, life_rule="median", max_ante=4, **hooks):
    t = Tournament(seed=SEED, n_agents=n, players=_population(n), life_rule=life_rule,
                   max_ante=max_ante, **hooks)
    return t, t.run()


def test_op_lose_life_matches_the_replay_packages_constant():
    """Duplicated as a literal so `tournament` keeps no dependency on `replay`; if the
    two ever drift, every tournament trajectory silently stops replaying."""
    import sys
    from pathlib import Path
    mp_root = str(Path(__file__).resolve().parents[2])
    if mp_root not in sys.path:
        sys.path.insert(0, mp_root)
    util = pytest.importorskip("replay._util", reason="W3's replay has not landed")
    assert OP_LOSE_LIFE == util.OP_LOSE_LIFE


def test_on_fanout_sees_the_games_before_anybody_acts():
    seen = {}

    def on_fanout(games, seed_str):
        seen["n"] = len(games)
        seen["seed"] = seed_str
        seen["states"] = [g.state for g in games]
        seen["sigs"] = {g.state_signature() for g in games}

    _run(n=4, on_fanout=on_fanout)
    assert seen["n"] == 4 and seen["seed"] == SEED
    assert all(s == State.BLIND_SELECT for s in seen["states"])
    assert len(seen["sigs"]) == 1, "same seed, same start, nobody has acted yet"


def test_on_step_fires_after_every_step_including_the_synthetic_life_loss():
    ops: list = []
    _run(n=4, on_step=lambda i, g, a: ops.append((i, a["type"], g.lives)))

    assert ops
    assert any(t == OP_LOSE_LIFE for _, t, _ in ops), "the life rule must be recorded"
    assert any(t == "advance" for _, t, _ in ops), "_cash_out's advance is a real action"
    # The synthetic op is emitted AFTER `lose_life()`, so the lives it reports are the new
    # ones — that is what a replay reproduces.
    for i, t, lives in ops:
        if t == OP_LOSE_LIFE:
            assert lives >= 0


def test_on_agent_done_fires_exactly_once_per_agent():
    calls: list = []
    _run(n=5, on_agent_done=lambda i, g, why: calls.append((i, why)))
    idxs = [i for i, _ in calls]
    assert sorted(idxs) == list(range(5))
    assert all(why in ("died", "eliminated", "finished") for _, why in calls)


def test_an_eliminated_agent_is_finished_before_the_out_of_band_game_over():
    """`Tournament.run` force-sets `State.GAME_OVER` on an agent the life rule eliminated.
    A trajectory log has to take its final signature BEFORE that, or replay diverges on the
    last line."""
    states: dict = {}
    _run(n=4, life_rule="median", max_ante=8,
         on_agent_done=lambda i, g, why: states.setdefault(i, (why, g.state, g.lives)))
    eliminated = [(i, v) for i, v in states.items() if v[0] == "eliminated"]
    if not eliminated:
        pytest.skip("no agent was eliminated by the life rule on this seed")
    for _, (_, state, lives) in eliminated:
        assert state != State.GAME_OVER
        assert lives == 0


def test_hooks_do_not_change_the_run():
    import math

    def sig(**hooks):
        t, r = _run(n=4, **hooks)
        scores = [[None if math.isnan(v) else float(v) for v in m.scores]
                  for m in r.ante_matrices]          # NaN != NaN; an absent agent is None
        return ([g.state_signature() for g in t._last_games], r.final_lives, scores)

    assert sig() == sig(on_step=lambda *a: None, on_agent_done=lambda *a: None,
                        on_fanout=lambda *a: None)


def test_replays_a_logged_tournament(tmp_path):
    """End to end: write a real trajectory log through the hooks, replay every line."""
    import sys
    from pathlib import Path
    mp_root = str(Path(__file__).resolve().parents[2])
    if mp_root not in sys.path:
        sys.path.insert(0, mp_root)
    log_mod = pytest.importorskip("replay.log", reason="W3's replay has not landed")
    replay_mod = pytest.importorskip("replay.replay")

    path = tmp_path / "tournament.jsonl"
    loggers = {i: log_mod.TrajectoryLogger(str(path), sig_every=5) for i in range(4)}

    def on_fanout(games, seed_str):
        for i, lg in loggers.items():
            lg.begin(games[i], {"agent": i})

    def on_step(i, game, action):
        loggers[i].step(game, action)

    def on_done(i, game, why):
        loggers.pop(i).end(game, {"reason": why})

    _run(n=4, life_rule="median", max_ante=6, on_fanout=on_fanout, on_step=on_step,
         on_agent_done=on_done)
    assert not loggers

    import json
    lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 4
    for line in lines:
        rebuilt = replay_mod.replay_line(line)          # raises on any divergence
        assert rebuilt.lives == line["final_state"]["lives"]
