"""Tests for mp/eval/targets.py: vanilla_boss_target's chip-formula pin, the
scaled_own_big_blind / table_target factories, the get_target registry, and the module's
"engine-only, no mp.eval heavy imports" claim (subprocess-isolated). Run:
python -m pytest mp/eval/tests -q (repo root)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import common as C
import targets as T

from balatro_sim.constants import blind_base_chips, get_blind_amount
from balatro_sim.decks import deck_spec
from balatro_sim.game import BOSS_CHIP_MULT
from tournament.runner import Tournament
from tournament.players import default_population

_REPO_ROOT = Path(__file__).resolve().parents[3]   # .../balatro-rl (mp/eval/tests -> ^3)

WHITE = 1
THREE_DECKS = ("b_red", "b_checkered", "b_plasma")


# ============================================================================ vanilla_boss_target

@pytest.mark.parametrize("ante", list(range(1, 13)))
@pytest.mark.parametrize("deck", THREE_DECKS)
def test_vanilla_boss_target_matches_direct_formula(ante, deck):
    """Pin: vanilla_boss_target must equal int(blind_base_chips(ante, 2, scaling) *
    ante_scaling) -- game.py:642's own composition, before any boss-specific multiplier."""
    scaling = 1   # White stake
    ante_scale = deck_spec(deck).ante_scaling
    expected = int(blind_base_chips(ante, 2, scaling) * ante_scale)
    assert T.vanilla_boss_target(ante, deck, WHITE) == expected


def test_plasma_is_exactly_2x_red_every_ante_1_to_12():
    for ante in range(1, 13):
        red = T.vanilla_boss_target(ante, "b_red", WHITE)
        plasma = T.vanilla_boss_target(ante, "b_plasma", WHITE)
        assert plasma == 2 * red, ante


def test_checkered_equals_red_every_ante_1_to_12():
    """Checkered's ante_scaling is 1 (same as Red) -- the target itself only differs from
    Red's when a deck changes ante_scaling; composition differences (Checkered's swapped
    suits) show up in ACHIEVED SCORE against a fixed target, not in the target itself --
    that is exactly the thing transfer_spread.py measures."""
    for ante in range(1, 13):
        assert T.vanilla_boss_target(ante, "b_checkered", WHITE) == T.vanilla_boss_target(ante, "b_red", WHITE)


def test_ante_beyond_8_uses_the_endless_formula():
    """ante > 8 must not just repeat the ante-8 value -- it must follow
    constants.get_blind_amount's endless formula (composed with BLIND_MULT[2]=2.0)."""
    a9 = T.vanilla_boss_target(9, "b_red", WHITE)
    a8 = T.vanilla_boss_target(8, "b_red", WHITE)
    assert a9 != a8
    assert a9 == int(get_blind_amount(9, 1) * 2.0)
    a12 = T.vanilla_boss_target(12, "b_red", WHITE)
    assert a12 == int(get_blind_amount(12, 1) * 2.0)
    assert a12 > a9 > a8


def test_higher_stake_scaling_raises_the_target():
    # stake_green (level 3) has blind_scaling=2, stake_purple (level 6) has scaling=3.
    white = T.vanilla_boss_target(4, "b_red", 1)
    green = T.vanilla_boss_target(4, "b_red", 3)
    purple = T.vanilla_boss_target(4, "b_red", 6)
    assert white < green < purple


def _at_ante1_boss_select(g) -> bool:
    return g.state == C.State.BLIND_SELECT and g.ante == 1 and g.blind_idx == 2


def test_vanilla_boss_target_matches_a_live_vanilla_boss_with_no_special_multiplier():
    """Integration check: for a real BalatroGame's ante-1 Boss whose boss_key carries no
    BOSS_CHIP_MULT entry (mult == 1.0, the common case -- 2/3 of the three exceptions are
    MLB_BANNED_BLINDS and never drawn under ruleset='mlb' anyway), the live
    current_blind.chips_target must equal vanilla_boss_target(1, deck, stake) exactly. Drive
    Small + Big with a greedy scripted policy first (debug_win_blind only fires mid-hand)."""
    _label, policy = C.make_player_policy("scripted:hand=greedy")
    checked_any = False
    for seed in C.DEFAULT_SEEDS[:30]:
        g = C.BalatroGame(seed=seed, deck_key="b_red", stake=1, ruleset="vanilla")
        C.run_until(g, policy, _at_ante1_boss_select, max_steps=2000)
        if not _at_ante1_boss_select(g):
            continue   # this seed's greedy policy lost Small/Big under vanilla's single-life rule
        if BOSS_CHIP_MULT.get(g.boss_blind, 1.0) == 1.0:
            assert g.current_blind.chips_target == T.vanilla_boss_target(1, "b_red", 1)
            checked_any = True
    assert checked_any, "no ground-truth seed had an unmultiplied ante-1 boss to check against"


# ============================================================================ scaled_own_big_blind

def test_scaled_own_big_blind_defaults_to_zero_before_any_big_blind():
    target = T.scaled_own_big_blind(k=1.0)
    game = C.BalatroGame(seed="7I4M53DL", ruleset="mlb")
    assert target(game, {}) == 0
    assert target(game, None) == 0
    assert target(game, {game.ante: 500}) == 500


def test_scaled_own_big_blind_scales_by_k_and_matches_common_own_big_blind_target():
    """This module deliberately DUPLICATES eval/common.py::own_big_blind_target (so
    mp/agent never has to import the heavier common.py) -- pin that the two are numerically
    identical for the same (game, big_blind)."""
    game = C.BalatroGame(seed="7I4M53DL", ruleset="mlb")
    big_blind = {game.ante: 321}
    for k in (0.5, 1.0, 2.0):
        assert T.scaled_own_big_blind(k=k)(game, big_blind) == C.own_big_blind_target(k=k)(game, big_blind)


# ============================================================================ table_target

def _write_summary_jsonl(path: Path, rows: list) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_table_target_reads_median_by_default(tmp_path):
    p = tmp_path / "summary.jsonl"
    _write_summary_jsonl(p, [
        {"ante": 2, "quantiles": {"0.5": 1000.0, "0.9": 2000.0}},
        {"ante": 3, "quantiles": {"0.5": 1500.0, "0.9": 3000.0}},
    ])
    target = T.table_target(p)
    game_a2 = type("G", (), {"ante": 2})()
    game_a3 = type("G", (), {"ante": 3})()
    assert target(game_a2) == 1000
    assert target(game_a3) == 1500


def test_table_target_quantile_configurable(tmp_path):
    p = tmp_path / "summary.jsonl"
    _write_summary_jsonl(p, [{"ante": 2, "quantiles": {"0.5": 1000.0, "0.9": 2000.0}}])
    target = T.table_target(p, quantile=0.9)
    game = type("G", (), {"ante": 2})()
    assert target(game) == 2000


def test_table_target_accepts_a_directory_containing_summary_jsonl(tmp_path):
    _write_summary_jsonl(tmp_path / "summary.jsonl", [{"ante": 2, "quantiles": {"0.5": 777.0}}])
    target = T.table_target(tmp_path)   # directory, not the file itself
    game = type("G", (), {"ante": 2})()
    assert target(game) == 777


def test_table_target_missing_quantile_raises_at_construction(tmp_path):
    p = tmp_path / "summary.jsonl"
    _write_summary_jsonl(p, [{"ante": 2, "quantiles": {"0.5": 1000.0}}])
    with pytest.raises(KeyError):
        T.table_target(p, quantile=0.9)   # fails fast, not on first call


def test_table_target_fallback_nearest_below(tmp_path):
    p = tmp_path / "summary.jsonl"
    _write_summary_jsonl(p, [
        {"ante": 2, "quantiles": {"0.5": 1000.0}},
        {"ante": 5, "quantiles": {"0.5": 4000.0}},
    ])
    target = T.table_target(p)   # default fallback="nearest_below"
    game_a4 = type("G", (), {"ante": 4})()
    assert target(game_a4) == 1000   # nearest tabulated ante <= 4 is 2
    game_a1 = type("G", (), {"ante": 1})()
    with pytest.raises(KeyError):
        target(game_a1)   # below every tabulated ante


def test_table_target_fallback_error_mode(tmp_path):
    p = tmp_path / "summary.jsonl"
    _write_summary_jsonl(p, [{"ante": 2, "quantiles": {"0.5": 1000.0}}])
    target = T.table_target(p, fallback="error")
    game_a3 = type("G", (), {"ante": 3})()
    with pytest.raises(KeyError):
        target(game_a3)


def test_table_target_bad_fallback_value_raises():
    with pytest.raises(ValueError):
        T.table_target("unused", fallback="not_a_real_mode")


def test_table_target_no_such_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        T.table_target(tmp_path / "does_not_exist.jsonl")


def test_table_target_against_a_real_tournament_run(tmp_path):
    """End-to-end: write_run's actual summary.jsonl format, read back through
    table_target, values match matrix.score_distribution's own quantiles exactly."""
    players = default_population(6, base_seed=0)
    res = Tournament(seed="7I4M53DL", n_agents=6, players=players, deck_key="b_red",
                     stake=1, life_rule="none", max_ante=3, out_dir=str(tmp_path)).run()
    target = T.table_target(tmp_path, quantile=0.5)
    for m in res.ante_matrices:
        game = type("G", (), {"ante": m.ante})()
        assert target(game) == int(round(m.stats["quantiles"]["0.5"]))


# ============================================================================ registry

def test_get_target_vanilla_boss():
    target = T.get_target("vanilla_boss")
    game = type("G", (), {"ante": 3, "deck_key": "b_plasma", "stake": 1})()
    assert target(game, {}) == T.vanilla_boss_target(3, "b_plasma", 1)


def test_get_target_own_big_blind_with_kwargs():
    target = T.get_target("own_big_blind", k=2.0)
    game = type("G", (), {"ante": 4})()
    assert target(game, {4: 10}) == 20


def test_get_target_table_with_kwargs(tmp_path):
    _write_summary_jsonl(tmp_path / "summary.jsonl", [{"ante": 2, "quantiles": {"0.5": 999.0}}])
    target = T.get_target("table", path=tmp_path, quantile=0.5)
    game = type("G", (), {"ante": 2})()
    assert target(game) == 999


def test_get_target_unknown_name_raises():
    with pytest.raises(ValueError):
        T.get_target("not_a_real_target")


# ============================================================================ "engine-only deps" claim

def test_targets_module_avoids_heavy_mp_eval_imports_when_imported_alone():
    """Import targets.py in a FRESH interpreter with only mp/eval on sys.path (mirroring
    how mp/agent would import it) and confirm it never pulled in mlb_match_demo,
    oracle.parity_check or mp.rng.generate/pools -- eval/common.py's heavier bootstrap
    chain, and precisely what the Phase 4 brief's "engine-only deps" asks this module to
    avoid so mp/agent can import it cheaply."""
    script = (
        "import sys; "
        "sys.path.insert(0, 'mp/eval'); "
        "import targets; "
        "forbidden = ['mlb_match_demo', 'oracle.parity_check', 'rng.generate', 'rng.pools', 'torch']; "
        "leaked = [m for m in forbidden if m in sys.modules]; "
        "assert not leaked, leaked; "
        "assert targets.vanilla_boss_target(1, 'b_red', 1) == 600; "
        "print('OK')"
    )
    proc = subprocess.run([sys.executable, "-c", script], cwd=str(_REPO_ROOT),
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "OK" in proc.stdout
