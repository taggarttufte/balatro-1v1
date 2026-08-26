"""Unit tests for eval/common.py: player-spec parsing, the solo-driving shim, statistics
helpers, and the phantom-Nemesis target functions.  Run: python -m pytest eval/tests -q
(repo root)."""
from __future__ import annotations

import math

import pytest

import common as C


# ============================================================================ player-spec parsing

def test_scripted_spec_defaults():
    label, spec = C.parse_player_spec("scripted:")
    assert isinstance(spec, C.ScriptedPlayer)
    assert spec.hand == "greedy"
    assert spec.buy_slot0 is False


def test_scripted_spec_aliases_and_types():
    label, spec = C.parse_player_spec("scripted:reroll=1,buy=1,pack=0,hand=weak")
    assert spec.rerolls_per_visit == 1
    assert spec.buy_slot0 is True
    assert spec.open_pack_slot == 0
    assert spec.hand == "weak"


def test_scripted_spec_unknown_field_raises():
    with pytest.raises(ValueError):
        C.parse_player_spec("scripted:not_a_real_field=1")


def test_checkpoint_spec_builds_an_mp_agent_player():
    """Phase 4 close: ``checkpoint:`` wires to ``agent``'s ``make_player``. A missing path
    surfaces as the loader's own error (FileNotFoundError), not NotImplementedError; an empty
    path gives a cold-start player that satisfies the ``Player`` protocol."""
    pytest.importorskip("torch")
    with pytest.raises(FileNotFoundError):
        C.parse_player_spec("checkpoint:/definitely/not/here.pt")
    label, player = C.parse_player_spec("checkpoint:,sims=4,device=cpu")
    assert label.startswith("checkpoint:")
    assert callable(getattr(player, "act", None)) and callable(getattr(player, "reset", None))


def test_unknown_spec_kind_raises_value_error():
    with pytest.raises(ValueError):
        C.parse_player_spec("not_a_kind:whatever")


def test_make_player_policy_scripted_is_callable():
    label, policy = C.make_player_policy("scripted:hand=greedy")
    assert label == "scripted:hand=greedy"
    game = C.BalatroGame(seed="7I4M53DL", ruleset="vanilla")
    action = C.solo_policy_step(game, policy)
    assert isinstance(action, dict) and "type" in action


# ============================================================================ SoloShim / drivers wiring

def test_solo_policy_step_matches_manual_shim():
    game = C.BalatroGame(seed="ALEEB", ruleset="vanilla")
    _, policy = C.make_player_policy("scripted:hand=greedy")
    shim = C.SoloShim(games=[game])
    expected = policy(shim, 0, game.legal_actions())
    got = C.solo_policy_step(game, policy)
    assert got == expected


def test_play_sp_vanilla_terminates_and_reports_expected_fields():
    _, policy = C.make_player_policy("scripted:hand=greedy,buy=1,pack=0")
    r = C.play_sp_vanilla("7I4M53DL", policy, max_steps=5000)
    for k in ("seed", "won", "furthest_ante", "furthest_blind", "final_ante", "final_money", "steps"):
        assert k in r
    assert r["seed"] == "7I4M53DL"
    assert isinstance(r["won"], bool)


def test_play_sp_mlb_lives_lost_matches_observed_delta():
    """Regression: lives_lost must count EVERY source of life loss (regular-blind fails,
    deck-outs) not just the Nemesis-vs-target comparison this driver makes directly."""
    _, policy = C.make_player_policy("scripted:hand=greedy,reroll=1,buy=1")
    r = C.play_sp_mlb("7I4M53DL", policy, lives=4, max_antes=6, max_steps=5000)
    assert r["lives_lost"] == 4 - r["final_lives"]
    assert 0 <= r["final_lives"] <= 4


def test_play_1v1_runs_a_real_match():
    _, pa = C.make_player_policy("scripted:hand=greedy,reroll=1,buy=1")
    _, pb = C.make_player_policy("scripted:hand=weak")
    r = C.play_1v1("7I4M53DL", pa, pb, max_steps=5000)
    assert r["winner"] in (0, 1)
    assert r["done"] is True
    assert r["lives"][r["winner"]] > 0
    assert r["lives"][1 - r["winner"]] == 0


# ============================================================================ target functions

def test_own_big_blind_target_defaults_to_zero_before_any_big_blind():
    target = C.own_big_blind_target(k=1.0)
    game = C.BalatroGame(seed="7I4M53DL", ruleset="mlb")
    assert target(game, {}) == 0
    assert target(game, {game.ante: 500}) == 500


def test_own_big_blind_target_scales_by_k():
    target = C.own_big_blind_target(k=2.0)
    game = C.BalatroGame(seed="7I4M53DL", ruleset="mlb")
    assert target(game, {game.ante: 100}) == 200


def test_external_vanilla_big_blind_target_is_seed_independent_given_ante():
    gA = C.BalatroGame(seed="7I4M53DL", ruleset="mlb")
    gB = C.BalatroGame(seed="ALEEB", ruleset="mlb")
    assert C.external_vanilla_big_blind_target(gA) == C.external_vanilla_big_blind_target(gB)
    assert C.external_vanilla_big_blind_target(gA) > 0


# ============================================================================ statistics helpers

def test_bootstrap_ci_mean_of_constant_is_exact():
    ci = C.bootstrap_ci([5.0] * 10, n_boot=200)
    assert ci["point"] == 5.0
    assert ci["lo"] == 5.0
    assert ci["hi"] == 5.0


def test_bootstrap_ci_contains_true_mean_typically():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    ci = C.bootstrap_ci(vals, n_boot=1000, seed=1)
    assert ci["lo"] <= ci["point"] <= ci["hi"]
    assert ci["lo"] < 4.5 < ci["hi"]


def test_paired_bootstrap_ci_zero_for_identical_sequences():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ci = C.paired_bootstrap_ci(xs, xs, n_boot=200)
    assert ci["point"] == 0.0
    assert ci["lo"] == 0.0 and ci["hi"] == 0.0


def test_pearson_r_perfect_positive_and_negative():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [2.0, 4.0, 6.0, 8.0, 10.0]
    assert C.pearson_r(xs, ys) == pytest.approx(1.0)
    zs = [-1.0, -2.0, -3.0, -4.0, -5.0]
    assert C.pearson_r(xs, zs) == pytest.approx(-1.0)


def test_pearson_r_constant_input_is_nan():
    assert math.isnan(C.pearson_r([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))


def test_spearman_r_monotone_nonlinear_is_one():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [1.0, 4.0, 9.0, 16.0, 25.0]     # monotone but nonlinear -- pearson < 1, spearman == 1
    assert C.spearman_r(xs, ys) == pytest.approx(1.0)
    assert C.pearson_r(xs, ys) < 1.0


def test_bootstrap_corr_ci_shape():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    ys = [1.1, 2.2, 2.9, 4.3, 4.8, 6.1]
    ci = C.bootstrap_corr_ci(xs, ys, "pearson", n_boot=300, seed=0)
    assert ci["lo"] <= ci["point"] <= ci["hi"]
    assert ci["n"] == 6


def test_unpaired_control_variance_ge_zero():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    ys = [1.1, 2.2, 2.9, 4.3, 4.8, 6.1]
    v = C.unpaired_control_variance(xs, ys, n_perm=200, seed=0)
    assert v >= 0


def test_sample_size_per_arm_paired_lt_unpaired_when_rho_positive():
    r = C.sample_size_per_arm(0.5, rho=0.8)
    assert r["n_paired"] < r["n_unpaired"]
    assert r["variance_reduction_factor"] == pytest.approx(5.0)
    r0 = C.sample_size_per_arm(0.5, rho=0.0)
    assert r0["n_paired"] == r0["n_unpaired"]
