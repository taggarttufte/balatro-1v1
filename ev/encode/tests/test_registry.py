"""test_registry.py — the registry format is the fleet's contract, so it is pinned here:
every entry is auditable (a Lua citation, an assumption list, a provenance note), every
predictor is a pure function of its summary, and the negative controls are declared as such.
"""
from __future__ import annotations

import pytest

import registry as R


def test_every_entry_is_auditable():
    for e in R.entries():
        assert e.lua, f"{e.key}: no Lua citation"
        assert e.assumptions, f"{e.key}: no assumptions"
        assert e.generated_by, f"{e.key}: no provenance"
        assert e.kind in R.KINDS and e.tier in R.TIERS and e.unit in R.UNITS


def test_the_poc_set_is_the_eight_items_the_brief_names():
    assert set(R.REGISTRY) == {
        "j_cloud_9", "j_rocket", "j_satellite",            # econ, trigger-targeted
        "j_ride_the_bus", "j_green_joker", "j_ice_cream",  # policy-conditional scaling
        "c_hermit",                                        # consumable
        "v_seed_money",                                    # voucher
    }


def test_the_three_tiers_are_all_represented():
    tiers = {e.tier for e in R.entries(include_controls=False)}
    assert "deterministic" in tiers and "policy_conditional" in tiers


def test_negative_controls_are_declared_and_separate():
    assert set(R.NEGATIVE_CONTROLS) == {"j_cloud_9__x3", "j_joker__doublecount"}
    assert all(e.expect_reject for e in R.NEGATIVE_CONTROLS.values())
    assert not any(e.expect_reject for e in R.REGISTRY.values())
    assert set(R.ALL_ENTRIES) == set(R.REGISTRY) | set(R.NEGATIVE_CONTROLS)


def test_controls_probe_the_real_engine_key():
    """A control names a fake registry key but lies about a real joker.  If the
    reachability probe followed the fake key, every control would be rejected as
    'unreachable' for a trivial reason and the controls would prove nothing."""
    assert R.NEGATIVE_CONTROLS["j_cloud_9__x3"].engine_key == "j_cloud_9"
    assert R.NEGATIVE_CONTROLS["j_joker__doublecount"].engine_key == "j_joker"
    assert R.REGISTRY["j_cloud_9"].engine_key == "j_cloud_9"


def test_entry_rejects_a_malformed_definition():
    ok = dict(key="k", name="n", kind="round_econ", tier="deterministic", unit="dollars",
              predict=lambda s: 0.0, modes=("round_end_paired",), assumptions=("a",),
              lua=("x.lua:1",), generated_by="t")
    R.Entry(**ok)                                   # the happy path builds
    for bad in ({"kind": "nope"}, {"tier": "nope"}, {"unit": "nope"},
                {"modes": ("nope",)}, {"lua": ()}, {"assumptions": ()},
                {"sign_of_delta": 7}):
        with pytest.raises(ValueError):
            R.Entry(**{**ok, **bad})


# ── the closed forms, checked against hand arithmetic (no engine involved) ──────────

def test_cloud_9_is_a_dollar_a_nine():
    p = R.REGISTRY["j_cloud_9"].predict
    assert [p({"deck_nines": n}) for n in (0, 1, 4, 7)] == [0.0, 1.0, 4.0, 7.0]


def test_rocket_pays_the_upgraded_figure_on_the_boss_round():
    """The +$2 lands in end_round() (state_events.lua:101) before evaluate_round reads
    calculate_dollar_bonus (:1174) — so a boss round pays base+2, not base."""
    p = R.REGISTRY["j_rocket"].predict
    assert p({"rocket_dollars": 1, "blind_is_boss": False}) == 1.0
    assert p({"rocket_dollars": 1, "blind_is_boss": True}) == 3.0
    assert p({"rocket_dollars": 5, "blind_is_boss": True}) == 7.0


def test_satellite_counts_unique_planets_only():
    p = R.REGISTRY["j_satellite"].predict
    assert p({"unique_planets_used": 0}) == 0.0
    assert p({"unique_planets_used": 3}) == 3.0


def test_hermit_caps_the_gain_not_the_balance():
    p = R.REGISTRY["c_hermit"].predict
    assert p({"dollars": 0}) == 0.0
    assert p({"dollars": 14}) == 14.0
    assert p({"dollars": 20}) == 20.0
    assert p({"dollars": 60}) == 20.0          # gain capped at $20 -> ends on $80
    assert p({"dollars": -5}) == 0.0           # math.max(0, ...) — never doubles a debt


def test_seed_money_is_a_step_function_of_money_held():
    """The shop rules give every voucher a flat +0.02.  This is the shape they miss: zero
    below $30 and $5/round at $50+."""
    p = R.REGISTRY["v_seed_money"].predict
    assert p({"dollars": 0}) == 0.0
    assert p({"dollars": 24}) == 0.0           # base cap $25 not reached
    assert p({"dollars": 27}) == 0.0
    assert p({"dollars": 30}) == 1.0
    assert p({"dollars": 40}) == 3.0
    assert p({"dollars": 55}) == 5.0
    assert p({"dollars": 500}) == 5.0          # both caps saturated


def test_ride_the_bus_matches_a_brute_force_expectation():
    """The closed form against an exhaustive enumeration of face/no-face sequences."""
    from itertools import product
    p = R.REGISTRY["j_ride_the_bus"].predict
    for rate in (0.0, 0.2, 0.35, 0.5, 1.0):
        for n in (1, 2, 5, 8):
            brute = 0.0
            for seq in product((0, 1), repeat=n):          # 1 = the hand had a face
                pr = 1.0
                m = 0.0
                for f in seq:
                    pr *= rate if f else (1 - rate)
                    m = 0.0 if f else m + 1
                brute += pr * m
            got = p({"hands_ahead": n, "rtb_mult": 0.0, "face_hand_rate": rate})
            assert got == pytest.approx(brute, abs=1e-9), (rate, n)


def test_ride_the_bus_degenerate_rates():
    p = R.REGISTRY["j_ride_the_bus"].predict
    assert p({"hands_ahead": 10, "rtb_mult": 0.0, "face_hand_rate": 0.0}) == 10.0
    assert p({"hands_ahead": 10, "rtb_mult": 3.0, "face_hand_rate": 1.0}) == 0.0
    assert p({"hands_ahead": 0, "rtb_mult": 4.0, "face_hand_rate": 0.3}) == 4.0


def test_green_joker_is_the_floored_random_walk_and_is_biased_low():
    p = R.REGISTRY["j_green_joker"].predict
    assert p({"green_mult": 0, "hands_ahead": 12, "discards_ahead": 4}) == 8.0
    assert p({"green_mult": 0, "hands_ahead": 4, "discards_ahead": 12}) == 0.0
    # the documented bias: the true value is >= this whenever the walk touches the floor
    assert p({"green_mult": 0, "hands_ahead": 6, "discards_ahead": 6}) == 0.0


def test_ice_cream_is_a_negative_scaler():
    e = R.REGISTRY["j_ice_cream"]
    assert e.sign_of_delta == -1
    vals = [e.predict({"ice_chips": 100, "hands_ahead": n}) for n in (0, 1, 12, 20, 40)]
    assert vals == [100.0, 95.0, 40.0, 0.0, 0.0]
    assert all(b <= a for a, b in zip(vals, vals[1:])), "a decay predictor must not rise"


def test_only_the_scaling_entries_claim_a_sign():
    for e in R.entries(include_controls=False):
        if e.tier == "policy_conditional":
            assert e.sign_of_delta != 0, f"{e.key}: a scaling entry must claim a direction"
        else:
            assert e.sign_of_delta == 0


# ── the controls really are wrong, at the level of arithmetic ──────────────────────

def test_control_one_is_three_times_the_truth():
    truth = R.REGISTRY["j_cloud_9"].predict
    lie = R.NEGATIVE_CONTROLS["j_cloud_9__x3"].predict
    for n in (1, 4, 7):
        assert lie({"deck_nines": n}) == 3 * truth({"deck_nines": n})


def test_control_two_prices_a_scoring_effect_as_dollars():
    lie = R.NEGATIVE_CONTROLS["j_joker__doublecount"]
    assert lie.kind == "round_econ" and lie.unit == "dollars"
    assert lie.predict({}) == 4.0                      # j_joker's +4 MULT, mispriced as $4
