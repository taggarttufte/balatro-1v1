"""Tests for eval/eval_harness.py: determinism, CI sanity (player vs itself), and report
schema round-trip, across all three modes.  Run: python -m pytest eval/tests -q (repo root).
"""
from __future__ import annotations

import json

import pytest

import common as C
import eval_harness as EH

SMALL_SEEDS = ["7I4M53DL", "ALEEB", "11111111", "1558AXDL", "15H9Z3IY"]
FAST = dict(n_boot=100)


# ============================================================================ determinism

@pytest.mark.parametrize("mode", ["sp_vanilla", "sp_mlb"])
def test_evaluate_is_deterministic(mode):
    r1 = EH.evaluate(mode, "scripted:hand=greedy,reroll=1,buy=1", SMALL_SEEDS, **FAST)
    r2 = EH.evaluate(mode, "scripted:hand=greedy,reroll=1,buy=1", SMALL_SEEDS, **FAST)
    assert r1["per_seed"] == r2["per_seed"]


def test_evaluate_1v1_is_deterministic():
    r1 = EH.evaluate("1v1", "scripted:hand=greedy,reroll=1,buy=1", SMALL_SEEDS,
                     reference_spec="scripted:hand=weak", **FAST)
    r2 = EH.evaluate("1v1", "scripted:hand=greedy,reroll=1,buy=1", SMALL_SEEDS,
                     reference_spec="scripted:hand=weak", **FAST)
    assert r1["per_seed"] == r2["per_seed"]


# ============================================================================ mode sanity (each mode runs, expected keys present)

def test_sp_vanilla_mode_shape():
    r = EH.evaluate("sp_vanilla", "scripted:hand=greedy,buy=1,pack=0", SMALL_SEEDS, **FAST)
    assert r["mode"] == "sp_vanilla"
    assert r["n_seeds"] == len(SMALL_SEEDS)
    assert "win_rate" in r["summary"]
    for row in r["per_seed"]:
        for k in ("won", "furthest_ante", "furthest_blind", "final_ante", "final_money"):
            assert k in row


def test_sp_mlb_mode_shape():
    r = EH.evaluate("sp_mlb", "scripted:hand=greedy,reroll=1,buy=1", SMALL_SEEDS, max_antes=6, **FAST)
    assert r["mode"] == "sp_mlb"
    for row in r["per_seed"]:
        for k in ("furthest_ante", "lives_lost", "final_lives", "money_curve", "nemesis_log"):
            assert k in row


def test_1v1_mode_shape():
    r = EH.evaluate("1v1", "scripted:hand=greedy,reroll=1,buy=1", SMALL_SEEDS,
                    reference_spec="scripted:hand=weak", **FAST)
    assert r["mode"] == "1v1"
    assert "win_rate" in r["summary"]
    for row in r["per_seed"]:
        for k in ("winner", "lives_margin", "pvp_log", "mean_log_score_margin"):
            assert k in row


def test_1v1_requires_reference():
    with pytest.raises(ValueError):
        EH.evaluate("1v1", "scripted:hand=greedy", SMALL_SEEDS)


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        EH.evaluate("not_a_mode", "scripted:hand=greedy", SMALL_SEEDS)


# ============================================================================ CI sanity: player vs itself

@pytest.mark.parametrize("mode,kwargs", [
    ("sp_vanilla", {}),
    ("sp_mlb", {"max_antes": 6}),
])
def test_compare_self_vs_self_ci_contains_zero(mode, kwargs):
    spec = "scripted:hand=greedy,reroll=1,buy=1"
    ra = EH.evaluate(mode, spec, SMALL_SEEDS, **kwargs, **FAST)
    rb = EH.evaluate(mode, spec, SMALL_SEEDS, **kwargs, **FAST)
    cmp = EH.compare(ra, rb, n_boot=200)
    assert cmp["n_paired_seeds"] == len(SMALL_SEEDS)
    assert cmp["diffs"], "expected at least one numeric field to compare"
    for field, d in cmp["diffs"].items():
        assert d["lo"] <= 0.0 <= d["hi"], f"{field}: CI [{d['lo']},{d['hi']}] does not contain 0"
        assert d["point"] == 0.0


def test_compare_1v1_self_vs_self_ci_contains_zero():
    spec = "scripted:hand=greedy,reroll=1,buy=1"
    ref = "scripted:hand=weak"
    ra = EH.evaluate("1v1", spec, SMALL_SEEDS, reference_spec=ref, **FAST)
    rb = EH.evaluate("1v1", spec, SMALL_SEEDS, reference_spec=ref, **FAST)
    cmp = EH.compare(ra, rb, n_boot=200)
    for field, d in cmp["diffs"].items():
        assert d["lo"] <= 0.0 <= d["hi"]


def test_compare_rejects_mismatched_modes():
    spec = "scripted:hand=greedy"
    ra = EH.evaluate("sp_vanilla", spec, SMALL_SEEDS, **FAST)
    rb = EH.evaluate("sp_mlb", spec, SMALL_SEEDS, **FAST)
    with pytest.raises(ValueError):
        EH.compare(ra, rb)


def test_compare_detects_a_real_difference():
    """Sanity check on the OTHER side: two genuinely different players should NOT all have
    zero-width, zero-centered CIs (guards against a --compare that accidentally always
    reports 0, e.g. from comparing a report to itself by mistake)."""
    ra = EH.evaluate("sp_vanilla", "scripted:hand=greedy,buy=1,pack=0", SMALL_SEEDS, **FAST)
    rb = EH.evaluate("sp_vanilla", "scripted:hand=weak", SMALL_SEEDS, **FAST)
    cmp = EH.compare(ra, rb, n_boot=200)
    assert any(d["point"] != 0.0 for d in cmp["diffs"].values())


# ============================================================================ report schema round-trip

@pytest.mark.parametrize("mode,kwargs", [
    ("sp_vanilla", {}),
    ("sp_mlb", {"max_antes": 6}),
])
def test_report_json_round_trip(tmp_path, mode, kwargs):
    r = EH.evaluate(mode, "scripted:hand=greedy,reroll=1,buy=1", SMALL_SEEDS, **kwargs, **FAST)
    path = tmp_path / "report.json"
    EH._dump(r, str(path))
    loaded = EH._load(str(path))
    assert loaded["mode"] == r["mode"]
    assert loaded["seeds"] == r["seeds"]
    assert len(loaded["per_seed"]) == len(r["per_seed"])
    assert set(loaded["summary"]) == set(r["summary"])
    # and the round-tripped report is itself comparable
    cmp = EH.compare(loaded, r, n_boot=100)
    for d in cmp["diffs"].values():
        assert d["point"] == 0.0


def test_1v1_report_json_round_trip(tmp_path):
    r = EH.evaluate("1v1", "scripted:hand=greedy,reroll=1,buy=1", SMALL_SEEDS,
                    reference_spec="scripted:hand=weak", **FAST)
    path = tmp_path / "report_1v1.json"
    EH._dump(r, str(path))
    loaded = EH._load(str(path))
    assert loaded["reference"] == r["reference"]
    assert loaded["per_seed"][0]["pvp_log"] == [list(x) for x in r["per_seed"][0]["pvp_log"]] \
        or loaded["per_seed"][0]["pvp_log"] == r["per_seed"][0]["pvp_log"]


# ============================================================================ CLI smoke

def test_cli_runs_and_writes_json(tmp_path):
    out = tmp_path / "cli_out.json"
    rc = EH.main(["--mode", "sp_vanilla", "--player", "scripted:hand=greedy",
                 "--seeds", ",".join(SMALL_SEEDS), "--n-boot", "50", "--out", str(out), "--quiet"])
    assert rc == 0
    assert out.is_file()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["mode"] == "sp_vanilla"


def test_cli_compare_runs(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    for p in (a, b):
        EH.main(["--mode", "sp_vanilla", "--player", "scripted:hand=greedy",
                "--seeds", ",".join(SMALL_SEEDS), "--n-boot", "50", "--out", str(p), "--quiet"])
    out = tmp_path / "cmp.json"
    rc = EH.main(["--compare", str(a), str(b), "--out", str(out), "--quiet"])
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["n_paired_seeds"] == len(SMALL_SEEDS)
