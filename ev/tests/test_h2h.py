"""test_h2h.py -- one seed x two seatings with fast players completes, and the result dict's
JSON schema matches ADVISOR_NOTES.md's documented shape (Phase 5 rev 2, W6)."""
from __future__ import annotations

import json

import pytest

import _bootstrap  # noqa: F401

import h2h


TRIAL_FIELDS = {
    "seed", "seating", "steps", "done", "seconds", "a_win", "lives_a", "lives_b",
    "lives_margin_a", "final_ante_a", "final_ante_b", "final_money_a", "final_money_b",
    "nem_wins_a", "nem_wins_b", "nem_total",
}
SUMMARY_FIELDS = {
    "n_trials", "n_decided", "a_wins", "b_wins", "undecided", "win_rate_a",
    "mean_final_ante_a", "mean_final_ante_b", "mean_lives_margin_a", "nemesis_win_rate_a",
    "mean_seconds_per_match",
}
RESULT_FIELDS = {
    "spec_a", "spec_b", "seeds", "n_seeds", "sims", "checkpoint", "lives", "max_steps",
    "deck_key", "stake", "procs", "seed_base", "wall_clock_s", "trials", "summary",
}
WIN_CI_FIELDS = {"point", "lo", "hi", "n"}


def test_one_seed_two_seatings_fast_players_completes():
    result = h2h.run_h2h("ev:fast", "ev:fast", ["11111111"], procs=0, seed_base=1)

    assert RESULT_FIELDS <= set(result)
    assert result["n_seeds"] == 1
    assert result["seeds"] == ["11111111"]
    assert len(result["trials"]) == 2                 # one seed -> two seatings
    assert {t["seating"] for t in result["trials"]} == {0, 1}

    for t in result["trials"]:
        assert TRIAL_FIELDS <= set(t)
        assert t["seed"] == "11111111"
        assert t["done"] is True                      # ev:fast vs ev:fast finishes well inside 100k steps
        assert t["a_win"] in (True, False)
        assert t["nem_total"] >= 0
        assert t["nem_wins_a"] + t["nem_wins_b"] <= t["nem_total"]

    s = result["summary"]
    assert SUMMARY_FIELDS <= set(s)
    assert WIN_CI_FIELDS <= set(s["win_rate_a"])
    assert s["n_trials"] == 2
    assert s["a_wins"] + s["b_wins"] == s["n_decided"]

    # the whole result is JSON-serializable (what write_report actually does)
    json.dumps(result, default=float)


def test_seatings_are_mirrors_for_an_identical_matchup():
    """ev:fast vs ev:fast is deterministic and identical on both sides (epsilon=0 -- no
    dependence on the per-player `seed` field), so seating 0 and seating 1 must be exact
    mirror images of the SAME underlying trajectory."""
    result = h2h.run_h2h("ev:fast", "ev:fast", ["1558AXDL"], procs=0, seed_base=0)
    t0 = next(t for t in result["trials"] if t["seating"] == 0)
    t1 = next(t for t in result["trials"] if t["seating"] == 1)
    assert t0["lives_a"] == t1["lives_b"]
    assert t0["lives_b"] == t1["lives_a"]
    assert t0["final_ante_a"] == t1["final_ante_b"]
    assert t0["steps"] == t1["steps"]
    assert t0["a_win"] != t1["a_win"], "identical policies on both seats: the winning SEAT " \
        "index is the same trajectory in both matches, so relabelling A/B must flip a_win"


def test_build_player_ev_specs():
    pol_fast, obj_fast = h2h.build_player("ev:fast", 0)
    pol_full, obj_full = h2h.build_player("ev:full", 0)
    pol_stats, obj_stats = h2h.build_player("ev:full+stats", 0)
    assert obj_fast.budget == "fast" and obj_fast.stats is None
    assert obj_full.budget == "full" and obj_full.stats is None
    assert obj_stats.budget == "full" and obj_stats.stats is not None
    assert callable(pol_fast) and callable(pol_full) and callable(pol_stats)


def test_build_player_scripted_spec():
    pol, obj = h2h.build_player("scripted:hand=greedy", 0)
    assert obj is None
    assert callable(pol)


def test_build_player_unknown_spec_raises():
    with pytest.raises(ValueError):
        h2h.build_player("nonsense:foo", 0)


def test_write_report_round_trips(tmp_path):
    result = h2h.run_h2h("ev:fast", "ev:fast", ["11111111"], procs=0)
    out_json = tmp_path / "r.json"
    out_md = tmp_path / "r.md"
    h2h.write_report(result, str(out_json), str(out_md))
    loaded = json.loads(out_json.read_text(encoding="utf-8"))
    assert loaded["spec_a"] == "ev:fast"
    assert len(loaded["trials"]) == 2
    md = out_md.read_text(encoding="utf-8")
    assert "H2H: ev:fast vs ev:fast" in md
    assert "Per-trial" in md
