"""Tests for eval/rho_decay.py: rho(h) == 1 with no perturbation, rho decreasing with a
larger perturbation, perturbation validation, and CLI/JSON round-trip.  Run:
python -m pytest eval/tests -q (repo root).

NB these run the real engine (no mocks) -- kept to small-ish seed counts / low n_boot to stay
fast; the FULL measurement (126+ seeds, n_boot=2000) is run separately by
`python -m eval.rho_decay --all` and written to results/ (see EVAL_NOTES.md).
"""
from __future__ import annotations

import json

import common as C
import rho_decay as R

SEEDS_8 = C.DEFAULT_SEEDS[:8]
SEEDS_24 = C.DEFAULT_SEEDS[:24]


# ============================================================================ perturbation registry / validation

def test_perturbations_registry_has_three_entries():
    assert set(R.PERTURBATIONS) == {"buy_slot0", "reroll_once", "skip_small"}


def test_unknown_perturbation_raises():
    import pytest
    with pytest.raises(ValueError):
        R.make_perturbed_game("7I4M53DL", "not_a_real_perturbation", "A")


# ============================================================================ determinism

def test_measure_rho_is_deterministic():
    r1 = R.measure_rho("buy_slot0", SEEDS_8, [1, 2], n_boot=100, ci_seed=0)
    r2 = R.measure_rho("buy_slot0", SEEDS_8, [1, 2], n_boot=100, ci_seed=0)
    assert r1["per_seed"] == r2["per_seed"]
    for h in ("1", "2"):
        assert (r1["per_horizon"][h]["metrics"]["log_score"]["pearson"]["point"]
                == r2["per_horizon"][h]["metrics"]["log_score"]["pearson"]["point"])


def test_make_perturbed_game_is_deterministic():
    g1, _ = R.make_perturbed_game("7I4M53DL", "buy_slot0", "A")
    g2, _ = R.make_perturbed_game("7I4M53DL", "buy_slot0", "A")
    assert g1.state_signature() == g2.state_signature()


# ============================================================================ rho(h) == 1 with no perturbation

def test_rho_is_one_with_no_perturbation():
    """Both arms get the IDENTICAL policy with no divergence at all ('none') -- the two arms
    are byte-identical plays of the same seed, so every outcome variable must correlate
    perfectly (and in fact be numerically equal) at every horizon."""
    result = R.measure_rho("none", SEEDS_8, [1, 2, 4], n_boot=50, ci_seed=0)
    for h in (1, 2, 4):
        row = result["per_horizon"][str(h)]
        for var in ("log_score", "money", "lives_lost"):
            pear = row["metrics"][var]["pearson"]["point"]
            assert pear == 1.0 or abs(pear - 1.0) < 1e-9, f"h={h} {var}: pearson={pear}"
        # and the raw per-seed values are literally identical, not just correlated
        for r in result["per_seed"]:
            a, b = r["A"][h], r["B"][h]
            assert a["score"] == b["score"]
            assert a["dollars"] == b["dollars"]
            assert a["cum_lives_lost"] == b["cum_lives_lost"]


# ============================================================================ rho decreases with a larger perturbation

def test_rho_lower_for_a_larger_perturbation():
    """buy_slot0 (spend on one item) is the smallest of the three registered perturbations;
    reroll_once (redraw the entire ante-1 shelf) is empirically the largest -- see
    EVAL_NOTES.md "what changed vs the design doc's guesses" for the full comparison over all
    126+ seeds.  Uses log-score (the primary outcome) at every horizon; allow noise by using
    enough seeds (24) and a comfortable margin, per the Phase 3 brief."""
    small = R.measure_rho("buy_slot0", SEEDS_24, [1, 2, 4], n_boot=200, ci_seed=0)
    large = R.measure_rho("reroll_once", SEEDS_24, [1, 2, 4], n_boot=200, ci_seed=0)
    for h in (1, 2, 4):
        r_small = small["per_horizon"][str(h)]["metrics"]["log_score"]["pearson"]["point"]
        r_large = large["per_horizon"][str(h)]["metrics"]["log_score"]["pearson"]["point"]
        assert r_large <= r_small + 0.05, (
            f"h={h}: expected the larger perturbation (reroll_once={r_large:.3f}) to correlate "
            f"no higher than the smaller one (buy_slot0={r_small:.3f}), modulo noise")


# ============================================================================ CLI / JSON round-trip

def test_cli_writes_expected_json(tmp_path):
    out = tmp_path / "rho.json"
    rc = R.main(["--perturbation", "buy_slot0", "--horizons", "1,2",
                "--seeds", ",".join(SEEDS_8), "--n-boot", "50", "--out", str(out), "--quiet"])
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["perturbation"] == "buy_slot0"
    assert data["horizons"] == [1, 2]
    assert set(data["per_horizon"]) == {"1", "2"}
    for h in ("1", "2"):
        for var in ("log_score", "money", "lives_lost"):
            m = data["per_horizon"][h]["metrics"][var]
            assert "pearson" in m and "spearman" in m and "variance_reduction_factor" in m


def test_list_perturbations_cli(capsys):
    rc = R.main(["--list-perturbations"])
    assert rc == 0
    out = capsys.readouterr().out
    for name in R.PERTURBATIONS:
        assert name in out


def test_make_extra_seeds_deterministic_and_well_formed():
    seeds1 = R.make_extra_seeds(10, rng_seed=42)
    seeds2 = R.make_extra_seeds(10, rng_seed=42)
    assert seeds1 == seeds2
    assert len(seeds1) == 10
    for s in seeds1:
        assert len(s) == 8
        assert all(ch in R.SEED_ALPHABET for ch in s)
