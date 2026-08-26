"""Unit tests for decide.py -- Row shape, sort order, side-effect freedom, and a fast
in-suite timing sanity check (the full >=300-state gate-3 benchmark is a separate script,
reported in STATS_NOTES.md, so the committed suite stays well under the 60s budget)."""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import BalatroGame, State

import decide

_SCRIPTS_DIR = str(Path(_bootstrap.MP_ROOT) / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
import mlb_match_demo as D  # noqa: E402


class _SoloShim:
    __slots__ = ("games",)

    def __init__(self, game):
        self.games = [game]


def _policy_step(game, policy_fn):
    acts = game.legal_actions()
    if not acts:
        return {"type": "advance"}
    return policy_fn(_SoloShim(game), 0, acts)


def _collect_states(seed: str, max_states: int = 6, max_steps: int = 6000):
    """Drive one seed with a pack-opening, rerolling scripted player and return every
    SHOP / BOOSTER_OPEN state reached (cloned so later steps cannot mutate them), up to
    ``max_states``."""
    spec = D.ScriptedPlayer(name="w4-test", hand="greedy", rerolls_per_visit=1,
                            buy_slot0=True, open_pack_slot=0, buy_voucher=True)
    policy = D.make_policy(spec)
    game = BalatroGame(seed=seed, deck_key="b_red", stake=1, ruleset="mlb")
    out = []
    last_state = None
    steps = 0
    while game.state != State.GAME_OVER and len(out) < max_states and steps < max_steps:
        if game.state != last_state and game.state in (State.SHOP, State.BOOSTER_OPEN):
            out.append(game.clone())
        last_state = game.state
        game.step(_policy_step(game, policy))
        steps += 1
    return out


def test_decision_table_empty_outside_shop_and_booster_open():
    g = BalatroGame(seed="11111111", deck_key="b_red", stake=1, ruleset="mlb")
    assert g.state == State.BLIND_SELECT
    assert decide.decision_table(g) == []


def test_row_shape_and_sorted_by_net_ev():
    states = _collect_states("11111111", max_states=3)
    assert states, "fixture found no SHOP/BOOSTER_OPEN state -- driver regressed"
    for g in states:
        rows = decide.decision_table(g)
        assert rows, f"expected at least one row (leave/skip is always legal) in {g.state}"
        for r in rows:
            assert isinstance(r.action, dict) and "type" in r.action
            assert isinstance(r.kind, str) and r.kind
            assert isinstance(r.label, str) and r.label
            for field in (r.p_hit, r.hit_value, r.cost, r.interest_loss, r.true_cost, r.urgency, r.net_ev):
                assert isinstance(field, float)
            assert 0.0 <= r.p_hit <= 1.0 + 1e-9
            assert 0.0 <= r.urgency <= 1.0 + 1e-9
            assert abs(r.true_cost - (r.cost + r.interest_loss)) < 1e-6
        net_evs = [r.net_ev for r in rows]
        assert net_evs == sorted(net_evs, reverse=True)


def test_shop_rows_have_a_leave_baseline():
    states = _collect_states("11111111", max_states=3)
    shop_states = [g for g in states if g.state == State.SHOP]
    assert shop_states
    for g in shop_states:
        rows = decide.decision_table(g)
        kinds = {r.kind for r in rows}
        assert "leave" in kinds
        leave_rows = [r for r in rows if r.kind == "leave"]
        assert len(leave_rows) == 1
        assert leave_rows[0].net_ev == 0.0


def test_booster_open_rows_have_a_skip_baseline():
    states = _collect_states("15H9Z3IY", max_states=6)
    booster_states = [g for g in states if g.state == State.BOOSTER_OPEN]
    if not booster_states:
        return   # this seed's scripted run never opened a pack -- not this test's job to force it
    for g in booster_states:
        rows = decide.decision_table(g)
        kinds = {r.kind for r in rows}
        assert "skip_pack" in kinds
        assert all(r.kind in ("skip_pack", "pick") for r in rows)


def test_decision_table_side_effect_free():
    states = _collect_states("1558AXDL", max_states=4)
    for g in states:
        sig_before = g.state_signature()
        decide.decision_table(g)
        assert g.state_signature() == sig_before


def test_decision_table_deterministic_across_calls():
    states = _collect_states("1KV4W6YS", max_states=2)
    for g in states:
        rows1 = decide.decision_table(g)
        rows2 = decide.decision_table(g)
        assert [(r.kind, r.action, round(r.net_ev, 9)) for r in rows1] == \
               [(r.kind, r.action, round(r.net_ev, 9)) for r in rows2]


def test_decision_table_timing_sanity():
    """Fast in-suite check (small N); the >=300-state gate-3 benchmark lives in
    STATS_NOTES.md (produced by stats/bench_decide.py, not run on every ``pytest``)."""
    states = []
    for seed in ("11111111", "1558AXDL", "15H9Z3IY", "1KV4W6YS", "1MD1YZ9T"):
        states.extend(_collect_states(seed, max_states=6))
    assert len(states) >= 10
    times = []
    for g in states:
        t0 = time.perf_counter()
        decide.decision_table(g)
        times.append((time.perf_counter() - t0) * 1000.0)
    mean_ms = statistics.fmean(times)
    p95_ms = sorted(times)[int(0.95 * (len(times) - 1))]
    assert mean_ms < 50.0, f"mean {mean_ms:.2f}ms over {len(times)} states exceeds the gate-3 budget"
    assert p95_ms < 100.0, f"p95 {p95_ms:.2f}ms is far outside the gate-3 budget"


def test_kind_enum_matches_row_kinds_seen():
    """Every ``kind`` this module ever emits is one the brief lists, PLUS the documented
    ``use_consumable`` extension (see STATS_NOTES.md "deviations from the interface")."""
    allowed = {"buy_joker", "buy_consumable", "buy_card", "buy_voucher", "buy_pack",
              "reroll", "pick", "skip_pack", "leave", "sell", "use_consumable"}
    states = _collect_states("11111111", max_states=6)
    seen = set()
    for g in states:
        for r in decide.decision_table(g):
            seen.add(r.kind)
    assert seen <= allowed
