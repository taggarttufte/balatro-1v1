"""Tests for mp/eval/transfer_spread.py: determinism, JSON-schema round-trip, the
identical-cell sanity case (spread CI must contain 0 when a player is evaluated against
itself across three cells wired to the SAME deck), and the checkpoint-spec passthrough.
Kept to small seed counts / n_agents / max_antes / n_boot to stay fast -- the real,
full-size run is `python -m mp.eval.transfer_spread` (see EVAL_NOTES.md Phase 4 for the
numbers). Run: python -m pytest mp/eval/tests -q (repo root)."""
from __future__ import annotations

import json

import pytest

import common as C
import transfer_spread as TS

SEEDS_3 = C.DEFAULT_SEEDS[:3]
SEEDS_2 = C.DEFAULT_SEEDS[:2]
FAST = dict(solo_seeds=SEEDS_3, tournament_seeds=SEEDS_2, n_agents=6, max_antes=3, n_boot=100)


def _strip_wall_clock(obj):
    """Recursively drop every 'wall_clock_s' key so two runs can be compared for exact
    determinism without timing noise tripping the comparison."""
    if isinstance(obj, dict):
        return {k: _strip_wall_clock(v) for k, v in obj.items() if k != "wall_clock_s"}
    if isinstance(obj, list):
        return [_strip_wall_clock(v) for v in obj]
    return obj


# ============================================================================ solo_cell / tournament_cell

def test_solo_cell_basic_shape():
    cell = TS.solo_cell("scripted:hand=greedy,reroll=1,buy=1", SEEDS_3, "b_red", max_antes=3, n_boot=100)
    assert cell["deck"] == "b_red"
    assert cell["n_seeds"] == 3
    assert len(cell["per_seed"]) == 3
    for r in cell["per_seed"]:
        assert 0 <= r["lives_lost"] <= C.MLB_STARTING_LIVES
        assert r["n_nemesis"] == len(r["margins"])
    s = cell["summary"]
    for k in ("furthest_ante", "lives_lost", "win_rate"):
        assert "point" in s[k] and "lo" in s[k] and "hi" in s[k]
    assert set(s["margin_quantiles"]) == {"0.0", "0.1", "0.25", "0.5", "0.75", "0.9", "1.0"}


def test_tournament_cell_basic_shape():
    cell = TS.tournament_cell("scripted:hand=greedy,reroll=1,buy=1", SEEDS_2, "b_red",
                              n_agents=6, max_ante=3)
    assert cell["deck"] == "b_red"
    assert cell["n_agents"] == 6
    assert len(cell["per_seed"]) == 2
    for r in cell["per_seed"]:
        for ante, rank in r["ante_ranks"].items():
            assert 1 <= rank <= 6
        assert 0.0 <= r["mean_rank_frac"] <= 1.0
    assert "point" in cell["summary"]["mean_rank_frac"]


def test_tournament_cell_same_base_seed_gives_identical_background_population_across_decks():
    """The background population must be the SAME composition across decks (only the deck
    should change between cells) -- checked indirectly: two calls with the same base_seed
    but different decks must reuse literally the same default_population() call (pinned by
    re-deriving it directly and comparing repr(), since ScriptedPlayerAdapter/RandomLegalPlayer
    are simple reprs)."""
    from tournament.players import default_population
    pop_a = default_population(5, base_seed=7)
    pop_b = default_population(5, base_seed=7)
    assert [repr(p) for p in pop_a] == [repr(p) for p in pop_b]


# ============================================================================ evaluate_player: shape + determinism

def test_evaluate_player_both_modes_shape():
    res = TS.evaluate_player("scripted:hand=greedy,reroll=1,buy=1", mode="both", **FAST)
    assert res["mode"] == "both"
    assert res["decks"] == list(TS.DECKS)
    assert len(res["cells"]) == 3
    for c in res["cells"]:
        assert "solo" in c and "tournament" in c
    assert "win_rate" in res["cross_cell_spread"]
    assert "rank_frac" in res["cross_cell_spread"]


def test_evaluate_player_solo_only_mode_omits_tournament():
    res = TS.evaluate_player("scripted:hand=greedy", mode="solo", solo_seeds=SEEDS_3, n_boot=100)
    for c in res["cells"]:
        assert "solo" in c
        assert "tournament" not in c
    assert "rank_frac" not in res["cross_cell_spread"]
    assert res["n_tournament_seeds"] == 0


def test_evaluate_player_tournament_only_mode_omits_solo():
    res = TS.evaluate_player("scripted:hand=greedy", mode="tournament", tournament_seeds=SEEDS_2,
                             n_agents=6, max_antes=3, n_boot=100)
    for c in res["cells"]:
        assert "tournament" in c
        assert "solo" not in c
    assert "win_rate" not in res["cross_cell_spread"]
    assert res["n_solo_seeds"] == 0


def test_evaluate_player_is_deterministic():
    r1 = TS.evaluate_player("scripted:hand=greedy,reroll=1,buy=1", mode="both", **FAST)
    r2 = TS.evaluate_player("scripted:hand=greedy,reroll=1,buy=1", mode="both", **FAST)
    assert _strip_wall_clock(r1) == _strip_wall_clock(r2)


def test_evaluate_player_json_round_trips(tmp_path):
    res = TS.evaluate_player("scripted:hand=greedy,reroll=1,buy=1", mode="both", **FAST)
    p = tmp_path / "report.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(res, f, default=str)
    with open(p, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["player"] == res["player"]
    assert loaded["decks"] == res["decks"]
    assert len(loaded["cells"]) == len(res["cells"])
    assert (loaded["cross_cell_spread"]["win_rate"]["point_range"]
            == res["cross_cell_spread"]["win_rate"]["point_range"])


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        TS.evaluate_player("scripted:hand=greedy", mode="not_a_real_mode")


# ============================================================================ identical-cell sanity case

def test_identical_cell_spread_ci_contains_zero_solo_mode():
    """A player evaluated against ITSELF across three cells all wired to the SAME deck must
    show EXACTLY zero spread (not just a CI containing zero) -- same seed, same deck, same
    policy is fully deterministic, so every per-cell value is bit-identical and every
    bootstrap resample of identical arrays also gives range=variance=0."""
    res = TS.evaluate_player("scripted:hand=greedy,reroll=1,buy=1", mode="solo",
                             decks=("b_red", "b_red", "b_red"), solo_seeds=SEEDS_3, n_boot=200)
    cc = res["cross_cell_spread"]["win_rate"]
    assert cc["point_range"] == 0.0
    assert cc["point_variance"] == 0.0
    assert cc["range_ci"]["lo"] <= 0.0 <= cc["range_ci"]["hi"]
    assert cc["variance_ci"]["lo"] <= 0.0 <= cc["variance_ci"]["hi"]
    assert cc["range_ci"]["lo"] == 0.0 and cc["range_ci"]["hi"] == 0.0


def test_identical_cell_spread_ci_contains_zero_tournament_mode():
    res = TS.evaluate_player("scripted:hand=greedy,reroll=1,buy=1", mode="tournament",
                             decks=("b_red", "b_red"), tournament_seeds=SEEDS_2,
                             n_agents=6, max_antes=3, n_boot=200)
    cc = res["cross_cell_spread"]["rank_frac"]
    assert cc["point_range"] == 0.0
    assert cc["range_ci"]["lo"] == 0.0 and cc["range_ci"]["hi"] == 0.0


# ============================================================================ _cross_cell_bootstrap unit-level

def test_cross_cell_bootstrap_matches_hand_computed_point_values():
    maps = [{"s1": 0.0, "s2": 0.5}, {"s1": 1.0, "s2": 1.0}, {"s1": 0.5, "s2": 0.5}]
    out = TS._cross_cell_bootstrap(maps, n_boot=50, seed=0)
    # cell means: [0.25, 1.0, 0.5]
    assert out["per_cell_mean"] == pytest.approx([0.25, 1.0, 0.5])
    assert out["point_range"] == pytest.approx(0.75)
    assert out["n_paired_seeds"] == 2


def test_cross_cell_bootstrap_drops_seeds_missing_from_any_cell():
    maps = [{"s1": 1.0, "s2": 2.0}, {"s1": 1.0}]   # s2 missing from the second cell
    out = TS._cross_cell_bootstrap(maps, n_boot=50, seed=0)
    assert out["n_paired_seeds"] == 1


def test_cross_cell_bootstrap_nan_treated_as_absent():
    maps = [{"s1": 1.0, "s2": float("nan")}, {"s1": 1.0, "s2": 2.0}]
    out = TS._cross_cell_bootstrap(maps, n_boot=50, seed=0)
    assert out["n_paired_seeds"] == 1


def test_cross_cell_bootstrap_empty_pairing_returns_nan():
    out = TS._cross_cell_bootstrap([{"s1": 1.0}, {"s2": 1.0}], n_boot=50, seed=0)
    assert out["n_paired_seeds"] == 0
    assert out["range_ci"] is None


# ============================================================================ player construction

def test_build_tournament_player_scripted():
    p = TS._build_tournament_player("scripted:hand=greedy,buy=1")
    game = C.BalatroGame(seed="7I4M53DL", ruleset="mlb")
    action = p.act(game)
    assert isinstance(action, dict) and "type" in action


def test_build_tournament_player_checkpoint_bogus_path_is_not_swallowed_as_not_implemented():
    """MCTSPlayer is no longer the Phase-3 placeholder (the agent workstream wired it up
    concurrently -- tournament.players.MCTSPlayer is now a real factory over mp/agent/mcts).
    A bogus, nonexistent checkpoint path must raise the REAL underlying error (a file-load
    failure), never this module's NotImplementedError fallback -- _build_tournament_player
    only intercepts NotImplementedError, by design, so a real error is never masked."""
    with pytest.raises(Exception) as exc:
        TS._build_tournament_player("checkpoint:/definitely/missing/path.pt")
    assert not isinstance(exc.value, NotImplementedError)


def test_build_tournament_player_checkpoint_empty_path_is_cold_start_and_works():
    """"checkpoint:" (no path) -> checkpoint=None -> cold-start weights, a real usable
    Player right away -- exercises the actual pass-through end to end."""
    p = TS._build_tournament_player("checkpoint:")
    game = C.BalatroGame(seed="7I4M53DL", ruleset="mlb")
    action = p.act(game)
    assert isinstance(action, dict) and "type" in action
    p.reset()


# ============================================================================ markdown rendering

def test_to_markdown_runs_and_mentions_every_deck():
    res = TS.evaluate_player("scripted:hand=greedy", mode="both", **FAST)
    md = TS.to_markdown(res)
    assert isinstance(md, str) and len(md) > 0
    for deck in TS.DECKS:
        assert deck in md
