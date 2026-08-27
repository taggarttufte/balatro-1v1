"""test_verify.py — the harness's own machinery.

Two things are being pinned here and they matter more than the individual numbers:

1. **The accept rule bites.**  Each gate is exercised in isolation with a synthetic
   Measurement, and the negative controls are driven end to end through the real engine.
2. **The measurement is marginal.**  A construction that would double-count is shown to
   measure zero.
"""
from __future__ import annotations

import pytest

import registry as R
import verify as V
import run_poc as P


# ── the accept rule, gate by gate ───────────────────────────────────────────────────

def _m(**kw):
    base = dict(key="k", mode="round_end_paired", n=4, predicted=1.0, measured=1.0,
                ci=0.0, unit="dollars", fired=1, residual=0.0)
    base.update(kw)
    return V.Measurement(**base)


def test_a_clean_measurement_accepts():
    assert _m().accept


def test_the_ci_gate_rejects_a_residual_outside_its_band():
    assert not _m(predicted=2.0, measured=1.0, residual=1.0, ci=0.1).within_ci
    assert _m(predicted=2.0, measured=1.0, residual=1.0, ci=2.0).within_ci


def test_the_band_gate_rejects_a_scale_error_the_ci_would_swallow():
    """The load-bearing case: a 3x error inflates the residual SPREAD as fast as it inflates
    the residual, so a residual CI alone can accept it.  The band gate is not redundant."""
    m = _m(predicted=9.0, measured=3.0, residual=6.0, ci=6.2)
    assert m.within_ci, "precondition: this is exactly the case the CI cannot see"
    assert not m.within_band
    assert not m.accept and "band" in m.reason


def test_the_band_gate_is_zero_aware_which_is_what_catches_a_double_count():
    assert not _m(predicted=4.0, measured=0.0, residual=4.0).within_band
    assert _m(predicted=0.0, measured=0.0, residual=0.0).within_band


def test_the_band_gate_rejects_a_sign_error():
    assert not _m(predicted=5.0, measured=-5.0, residual=10.0, ci=99.0).within_band


def test_the_band_edges():
    assert _m(predicted=2.0, measured=1.0, residual=1.0, ci=9.0).within_band       # exactly 2x
    assert _m(predicted=0.5, measured=1.0, residual=-0.5, ci=9.0).within_band      # exactly 1/2x
    assert not _m(predicted=2.01, measured=1.0, residual=1.01, ci=9.0).within_band


def test_the_reachability_gate_rejects_a_number_nothing_produced():
    """A predictor of 4 that matched because the hook never ran is not a verified entry —
    the A1 lesson.  A zero claim with a zero measurement needs no firing."""
    assert not _m(predicted=4.0, measured=4.0, fired=0).reachable
    assert _m(predicted=0.0, measured=0.0, fired=0).reachable
    assert _m(predicted=4.0, measured=4.0, fired=0, needs_fire=False).reachable


def test_the_exactness_gate_applies_only_to_deterministic_entries():
    assert not _m(exact=False, exactness_required=True).accept
    assert _m(exact=False, exactness_required=False).accept


def test_an_unscored_measurement_is_info_not_a_reject():
    m = _m(predicted=float("nan"), measured=0.0, fired=0, scored=False)
    assert m.accept and "info" in m.reason and "INFO" in m.row()


# ── the reachability probe ──────────────────────────────────────────────────────────

def test_the_probe_counts_hooks_and_restores_the_registry():
    from balatro_sim.jokers.base import JOKER_REGISTRY
    real = JOKER_REGISTRY["j_cloud_9"]
    g = V.in_blind()
    V.set_hand(g, P._HAND)
    g.jokers = []
    V.set_deck_nines(g, 4)
    V.add_joker(g, "j_cloud_9")
    with V.reach_probe("j_cloud_9") as rec:
        assert JOKER_REGISTRY["j_cloud_9"] is not real, "the probe must be installed"
        g._end_round()
    assert JOKER_REGISTRY["j_cloud_9"] is real, "the probe must be uninstalled"
    assert rec.calls.get("on_round_end") == 1
    assert rec.money_written == 4.0


def test_the_probe_is_transparent_to_the_engine():
    """Probing must not change a single dollar — otherwise every measurement is of the
    probe.  Same construction, with and without the instrument."""
    def build():
        g = V.in_blind()
        V.set_hand(g, P._HAND)
        g.jokers = []
        V.set_deck_nines(g, 5)
        V.add_joker(g, "j_cloud_9")
        return g
    a = build()
    with V.reach_probe("j_cloud_9"):
        a._end_round()
    b = build()
    b._end_round()
    assert a.dollars == b.dollars


def test_the_probe_does_not_weaken_the_double_registration_guard():
    """jokers/base.py's _JokerRegistry refuses a second __setitem__ for a key.  The probe
    goes around it deliberately (dict.__setitem__) and must leave the guard intact."""
    from balatro_sim.jokers.base import JOKER_REGISTRY
    with V.reach_probe("j_cloud_9"):
        pass
    with pytest.raises(KeyError):
        JOKER_REGISTRY["j_cloud_9"] = object()


# ── the marginal rule, demonstrated ─────────────────────────────────────────────────

def test_an_already_priced_scoring_effect_measures_zero_dollars():
    """The +4 Mult of j_joker is priced by the scorer.  Its MARGINAL end-of-round dollars
    are exactly $0, which is what makes the double-count control fail."""
    e = R.NEGATIVE_CONTROLS["j_joker__doublecount"]
    m = V.measure_round_end(e, P.joker_doublecount_scenarios())
    assert m.measured == 0.0
    assert not m.accept


def test_negative_control_one_is_rejected_by_the_band():
    m = V.measure_round_end(R.NEGATIVE_CONTROLS["j_cloud_9__x3"], P.cloud9_scenarios())
    assert not m.accept
    assert "band" in m.reason
    assert m.fired > 0, "it must be rejected for being WRONG, not for being unreachable"


def test_negative_control_two_is_rejected_and_diagnosed_as_unreachable_too():
    """j_joker has no end-of-round hook at all, so the probe's verdict is the sharper
    diagnosis: the entry does not merely have the wrong number, it prices a row that does
    not exist."""
    m = V.measure_round_end(R.NEGATIVE_CONTROLS["j_joker__doublecount"],
                            P.joker_doublecount_scenarios())
    assert not m.accept
    assert "unreachable" in m.reason and "band" in m.reason


def test_the_controls_and_the_truth_go_through_the_identical_measurement():
    """Cloud 9 and Cloud-9-x3 are scored by the same call on the same scenarios.  If the
    controls were rejected by a different code path they would prove nothing."""
    good = V.measure_round_end(R.REGISTRY["j_cloud_9"], P.cloud9_scenarios())
    bad = V.measure_round_end(R.NEGATIVE_CONTROLS["j_cloud_9__x3"], P.cloud9_scenarios())
    assert good.measured == bad.measured        # same measurement
    assert good.accept and not bad.accept       # different verdict


# ── the deterministic entries, end to end through the real engine ───────────────────

def test_cloud_9_accepts_exactly():
    m = V.measure_round_end(R.REGISTRY["j_cloud_9"], P.cloud9_scenarios())
    assert m.accept and m.exact and m.fired == 4


def test_rocket_accepts_exactly_including_the_boss_ordering():
    m = V.measure_round_end(R.REGISTRY["j_rocket"], P.rocket_scenarios())
    assert m.accept and m.exact
    boss = [r for r in m.scenarios if r["scenario"].startswith("boss=1")]
    assert boss and all(r["measured"] == r["predicted"] for r in boss)
    # the claim that makes the entry non-trivial: a boss round pays base + 2
    b1 = next(r for r in m.scenarios if r["scenario"] == "boss=1,bonus=1")
    n1 = next(r for r in m.scenarios if r["scenario"] == "boss=0,bonus=1")
    assert b1["measured"] - n1["measured"] == 2


def test_hermit_accepts_exactly_and_caps_the_gain():
    m = V.measure_use(R.REGISTRY["c_hermit"], P.hermit_scenarios())
    assert m.accept and m.exact
    big = next(r for r in m.scenarios if r["scenario"] == "$60")
    assert big["measured"] == 20.0 and big["dollars_after"] == 80


def test_seed_money_accepts_exactly_and_is_a_step_function():
    m = V.measure_round_end(R.REGISTRY["v_seed_money"], P.seed_money_scenarios())
    assert m.accept and m.exact
    vals = {r["scenario"]: r["measured"] for r in m.scenarios}
    assert vals["$0"] == 0.0 and vals["$27"] == 0.0
    assert vals["$40"] == 3.0 and vals["$55"] == 5.0 and vals["$80"] == 5.0


def test_satellite_accepts_exactly_on_both_scenario_families():
    """**Flipped by W-FIX (2026-08-26).**  The entry was always faithful to the Lua; the
    ENGINE was not, and the two scenario families split cleanly — every "planets used
    BEFORE the purchase" case measured $0 against a predicted $1-$3, every "used after"
    case was exact.  That split is what turned a reject into a bug report (POC_NOTES §3.1),
    and it is the reason this test now asserts acceptance on the union: the engine reads
    the run-global planet tally, so purchase timing no longer changes the payout.

    The `before` family is retained deliberately — it is the discriminating half, and a
    regression would show up here as the same clean split."""
    m = V.measure_round_end(R.REGISTRY["j_satellite"],
                            P.satellite_scenarios("before") + P.satellite_scenarios("after"))
    assert m.accept and m.exact
    before = [r for r in m.scenarios if "before" in r["scenario"]]
    after = [r for r in m.scenarios if "after" in r["scenario"]]
    assert before and after
    for r in before + after:
        assert r["measured"] == r["predicted"], r["scenario"]


# ── the player-cache leak the harness has to defend against (POC_NOTES §3.5) ───────

def test_the_board_ratio_cache_key_ignores_planet_levels():
    """``hand._board_sig`` deliberately omits planet levels and the exact deck composition,
    but ``board_ratio`` samples real hands at the run's real levels — so two states that
    differ ONLY in an omitted field share a cache entry and the first one computed wins.

    The KEY is unchanged and this test still holds; what W-FIX (2026-08-26) changed is the
    SCOPE.  ``board_ratio`` now memoises into a caller-supplied dict and ``EVPlayer`` owns
    one per instance (cleared by ``reset()``), so the approximation stays inside a run —
    where it is deterministic given the seed — instead of crossing between runs that share
    a worker process.  ``hand._RATIO_CACHE``, exercised below, is the fallback for
    module-level callers like this test."""
    import hand as H
    a = V.in_blind()
    V.add_joker(a, "j_joker")
    b = V.in_blind()
    V.add_joker(b, "j_joker")
    b.planet_levels["Pair"] = b.planet_levels.get("Pair", 1) + 6      # a very different board
    assert H._board_sig(a) == H._board_sig(b), "the key cannot tell these two apart"

    H._RATIO_CACHE.clear()
    ra = H.board_ratio(a)                       # computed cold
    rb_warm = H.board_ratio(b)                  # served from a's entry
    assert rb_warm == ra, "b was served a's number"
    H._RATIO_CACHE.clear()
    rb_cold = H.board_ratio(b)                  # what b would have got in a cold process
    assert rb_cold != rb_warm, "and the cold answer differs — that is the leak"


def test_the_harness_resets_the_player_caches():
    import hand as H
    H._RATIO_CACHE["poisoned"] = 999.0
    H._MODEL_CACHE["poisoned"] = object()
    V.reset_player_caches()
    assert "poisoned" not in H._RATIO_CACHE and "poisoned" not in H._MODEL_CACHE


def test_a_trajectory_is_independent_of_what_ran_before_it():
    """The property the harness needs and the raw player does not have."""
    import hand as H
    H._RATIO_CACHE.clear()
    cold = V.trajectory("11111111", "j_green_joker", 8)
    V.trajectory("1DBCEO2Z", "j_ice_cream", 8)          # poison the process
    warm = V.trajectory("11111111", "j_green_joker", 8)
    assert (warm.value, warm.hands, warm.discards) == (cold.value, cold.hands, cold.discards)


# ── the policy-conditional mode (slow-ish: real ev:fast play) ──────────────────────

def test_a_trajectory_measures_the_policy_rates_a_lua_reading_cannot_supply():
    r = V.trajectory("11111111", "j_green_joker", 8)
    assert r.reached and r.hands == 8
    assert r.discards >= 0 and 0 <= r.face_hands <= r.hands
    assert r.value >= 0


def test_a_trajectory_records_the_joker_being_sold_rather_than_averaging_it_in():
    """The shop rules see no immediate strength on a scaling joker and sometimes sell it.
    That is the blind spot under study, so it has to be reported, not absorbed."""
    m = V.measure_trajectory(R.REGISTRY["j_ice_cream"],
                             P._seeds_for_test(6), 6, workers=2)
    assert m.extra["seeds_run"] == 6
    assert m.extra["seeds_scored"] <= 6
    assert 0.0 <= m.extra["survival_rate"] <= 1.0
    assert m.extra["sign_of_delta_ok"] is True, "Ice Cream must come out NEGATIVE"
    assert m.extra["delta"] < 0


def test_ice_cream_trajectory_is_the_deterministic_decay():
    m = V.measure_trajectory(R.REGISTRY["j_ice_cream"], P._seeds_for_test(4), 5, workers=2)
    for row in m.scenarios:
        assert row["predicted"] == 100.0 - 5.0 * row["hands"]


def test_green_joker_trajectory_uses_each_seeds_own_discard_count():
    """No rate is fitted for Green Joker — the predictor is fed the seed's own realized
    hands and discards, so the residual is the mechanism's error and nothing else."""
    m = V.measure_trajectory(R.REGISTRY["j_green_joker"], P._seeds_for_test(4), 6, workers=2)
    for row in m.scenarios:
        assert row["predicted"] == max(0.0, row["hands"] - row["discards"])


# ── the empirical fallback ─────────────────────────────────────────────────────────

def test_a_rejected_entry_yields_a_usable_measured_constant():
    m = V.measure_round_end(R.NEGATIVE_CONTROLS["j_cloud_9__x3"], P.cloud9_scenarios())
    fb = V.empirical_fallback(m)
    assert fb["supersedes_predict"] is True
    assert fb["value"] == m.measured and fb["unit"] == "dollars"
    assert "verify.py" in fb["source"]
