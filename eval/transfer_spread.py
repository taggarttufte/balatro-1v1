"""
eval/transfer_spread.py -- Phase 4 exit-gate item 4 / the campaign's DECISION GATE
(MP_SELFPLAY_ASSESSMENT_2026-08.md's "layer 1" transfer prior; MP_CAMPAIGN_PLAN_2026-08.md's
Phase 4: "Run ... the three-deck transfer spread (Red / Checkered / Plasma at White stake).
That's the decision gate from the assessment -- if a Red-trained policy collapses on Plasma,
you've learned the shape of the problem before building the other 12 decks and 7 stakes.").

Evaluates ONE player spec on the SAME seed(s) across three cells -- Red (`b_red`, baseline),
Checkered (`b_checkered`, a "composition change": LOW transfer prior per the assessment) and
Plasma (`b_plasma`, "a different game": `ante_scaling=2`, chips/mult balance at
`final_scoring_step`, LOWEST transfer prior, "expect this to be the hole") -- at White stake,
in two modes:

  (a) SP-MLB-solo (`eval/common.py::play_sp_mlb`) against an EXTERNAL target -- default
      `targets.vanilla_boss_target_fn()`, NOT `own_big_blind_target`: a calibration-free
      external cost is exactly what Phase 4 decision 4 requires (a free Nemesis is
      degenerate -- CAMPAIGN_LOG.md's 2026-08-22 07:35 overnight readout). Reports furthest
      ante, lives lost, per-Nemesis margin (`score - target`, pooled across seeds AND antes
      within a cell) quantiles, and win rate (fraction of Nemesis rounds with
      `score >= target`). Run over the 126 ground-truth seeds + `--n-extra-seeds` synthetic
      ones (`rho_decay.make_extra_seeds` -- cheap, this mode runs ~100-150 seeds/deck in a
      few seconds).

  (b) Tournament (`tournament.runner.Tournament(n=32, life_rule="none", max_ante=8)`): the
      evaluated player is mixed into a heterogeneous scripted+random population
      (`tournament.players.default_population`, held IDENTICAL across the three decks via a
      fixed `base_seed` so the only thing that changes between cells is the deck) at a FIXED
      index (0), and its population RANK (1 = best; `tournament.matrix.population_rank`) is
      read off at every ante it is present for -- `life_rule="none"` uses a lives sentinel
      so every agent is present at every ante (TOURNAMENT_NOTES.md S3). Reported as
      `rank_frac = (rank - 1) / (n_agents - 1)` in [0, 1], 0 = best. This mode is much more
      expensive per seed (~5-7s/seed/deck at n=32, max_ante=8 on this machine) than (a), so
      it defaults to a SMALLER seed subset (`--tournament-n-seeds`, default 16) -- a
      deliberate compute/statistics trade documented in EVAL_NOTES.md, not an oversight.

PAIRED BY SEED: both modes reuse the SAME seed value across all three decks, so a per-seed
outcome difference between cells is (to the extent the game is deterministic, which Phase 1
guarantees) a DECK effect, not seed noise -- with one caveat: the RNG stream KEYS ('shuffle'
/ 'shop' / 'boss' / 'tag' / ...) are identical across decks on one seed (Phase 1
stream-independence invariant), but what a deck's own generation-side hooks draw FROM, or
how an engine constant is scaled, differs per deck:
  - Checkered's post-creation suit swap (`decks.py::creation_order`) changes the physical
    52-card order the 'shuffle' stream indexes into, so THE SAME 'shuffle' draws deal
    different cards than Red on the same seed (this is the "composition change" the
    assessment flags as LOW-transfer).
  - Plasma's `ante_scaling=2` doubles every Nemesis's `chips_target` (this module's own
    `targets.vanilla_boss_target` bakes that in automatically) and its `plasma=True` scoring
    balance changes what a FIXED hand is worth (`scoring.score_hand`, `engine/DECKS_NOTES.md`
    S2) -- neither touches which cards are dealt.
  - Both decks otherwise draw from the identical shop/pack/voucher/boss/tag streams a Red
    run on the same seed would.

Cross-cell spread: for each of {win_rate (mode a), rank_frac (mode b)}, the per-cell POINT
means plus a bootstrap CI on the SPREAD (range and population variance across the cells),
computed by resampling the PAIRED seed set (the same resampled seed indices applied to every
cell each replicate, since all cells share one seed list) and recomputing each cell's mean +
the resulting range/variance on every replicate -- this is what answers "does the spread we
measured reflect a real deck effect, or seed noise," not just "what is the spread." See
`_cross_cell_bootstrap`.

    python -m eval.transfer_spread --player "scripted:hand=greedy,reroll=1,buy=1" \\
        --mode both --out results/transfer_spread_greedy_reroll1_buy1.json
    python -m eval.transfer_spread --player "scripted:hand=greedy" --mode solo \\
        --n-extra-seeds 24 --out results/transfer_spread_greedy.json
    python -m eval.transfer_spread --player "scripted:hand=greedy" --mode tournament \\
        --tournament-n-seeds 16 --out results/transfer_spread_greedy_tournament.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

_HERE = Path(__file__).resolve().parent      # eval
_MP_ROOT = _HERE.parent                       # repo root
for _p in (str(_MP_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import common as C           # noqa: E402  (fork-guarded engine bootstrap, drivers, stats)
import targets as T          # noqa: E402  (per-ante external Nemesis targets)
import rho_decay as RD       # noqa: E402  (make_extra_seeds -- synthetic seed generator)
from tournament.runner import Tournament                       # noqa: E402
from tournament.players import (                                # noqa: E402
    default_population, ScriptedPlayerAdapter, MCTSPlayer,
)

DECKS = ("b_red", "b_checkered", "b_plasma")
WHITE_STAKE = 1
_QUANTILES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)

__all__ = [
    "DECKS", "WHITE_STAKE", "solo_cell", "tournament_cell", "evaluate_player",
    "to_markdown", "main",
]


# ============================================================================ player construction

def _build_tournament_player(player_spec: str):
    """`player_spec` -> a `tournament.players.Player` (`.act(game) -> dict`, optional
    `.reset()`). ``"scripted:..."`` wraps `common.parse_player_spec`'s `ScriptedPlayer` in
    `ScriptedPlayerAdapter`. ``"checkpoint:<path>"`` is passed through to
    `MCTSPlayer(checkpoint=<path>)` -- if the agent workstream has wired it up by hand-off
    this just works (confirmed already live: `tournament.players.MCTSPlayer` is now a real
    factory over `agent/mcts`, not the Phase-3 placeholder); ``"checkpoint:"`` with no
    path gives `checkpoint=None` (cold-start weights, per `MCTSPlayer`'s own docstring) --
    useful for evaluating an untrained net's transfer spread as a baseline. If a future
    rollback ever reintroduces the placeholder (`NotImplementedError` unconditionally,
    "W1/W3 plug in here"), it is re-raised here with the exact expected call spelled out;
    any OTHER exception (e.g. a real checkpoint path that does not exist) is not
    intercepted and propagates as-is."""
    kind, _, body = player_spec.partition(":")
    if kind.strip().lower() == "checkpoint":
        checkpoint = body.strip() or None   # "checkpoint:" with no path -> cold-start weights
        try:
            return MCTSPlayer(checkpoint=checkpoint)
        except NotImplementedError:
            raise NotImplementedError(
                "transfer_spread's tournament mode expects "
                "tournament.players.MCTSPlayer(checkpoint=<path>) -> Player once the agent "
                f"workstream wires it up; got spec {player_spec!r}. MCTSPlayer is still the "
                "Phase-3 placeholder ('W1/W3 plug in here', tournament/players.py)."
            ) from None
    _label, spec = C.parse_player_spec(player_spec)
    return ScriptedPlayerAdapter(spec)


# ============================================================================ mode (a): SP-MLB-solo

def _pooled_quantiles(values: Sequence[float]) -> dict:
    if not values:
        return {str(q): None for q in _QUANTILES}
    vals = sorted(values)
    n = len(vals)
    out = {}
    for q in _QUANTILES:
        idx = min(n - 1, max(0, round(q * (n - 1))))
        out[str(q)] = vals[idx]
    return out


def solo_cell(player_spec: str, seeds: Sequence[str], deck_key: str, stake=WHITE_STAKE,
              max_antes: int = 8, lives: Optional[int] = None, target_name: str = "vanilla_boss",
              target_kwargs: Optional[dict] = None, n_boot: int = 2000, ci_seed: int = 0) -> dict:
    """Mode (a) for one (player, deck) cell: `play_sp_mlb` over `seeds` against
    `targets.get_target(target_name, **target_kwargs)`. Per-seed record: `furthest_ante`,
    `lives_lost`, `n_nemesis` (Nemesis rounds actually reached), `win_rate` (fraction of
    those with `score >= target`; `NaN` if the seed's run never reached a Nemesis --
    excluded from the summary, not counted as a loss), pooled per-Nemesis `margins`
    (`score - target`), `ended_early_engine_gap` (the known frozen-engine boss-rejection gap,
    `eval/common.py::play_sp_mlb` docstring / EVAL_NOTES.md S7 -- flagged, not miscounted)."""
    label = player_spec
    _label, policy = C.make_player_policy(player_spec)
    target_kwargs = dict(target_kwargs or {})
    target_fn = T.get_target(target_name, **target_kwargs)
    lives = C.MLB_STARTING_LIVES if lives is None else lives
    per_seed = []
    t0 = time.time()
    for seed in seeds:
        r = C.play_sp_mlb(seed, policy, deck_key=deck_key, stake=stake, lives=lives,
                           max_antes=max_antes, target_fn=target_fn)
        nlog = r["nemesis_log"]
        n_nem = len(nlog)
        wins = sum(1 for e in nlog if not e["life_lost"])
        win_rate = (wins / n_nem) if n_nem else float("nan")
        margins = [e["margin"] for e in nlog]
        per_seed.append({
            "seed": r["seed"], "furthest_ante": r["furthest_ante"], "lives_lost": r["lives_lost"],
            "final_lives": r["final_lives"], "n_nemesis": n_nem, "win_rate": win_rate,
            "margins": margins, "ended_early_engine_gap": r["ended_early_engine_gap"],
        })
    wall_s = time.time() - t0

    furthest = [r["furthest_ante"] for r in per_seed]
    lives_lost = [r["lives_lost"] for r in per_seed]
    win_rates = [r["win_rate"] for r in per_seed if r["win_rate"] == r["win_rate"]]
    pooled_margins = [m for r in per_seed for m in r["margins"]]
    summary = {
        "furthest_ante": C.bootstrap_ci(furthest, n_boot=n_boot, seed=ci_seed),
        "lives_lost": C.bootstrap_ci(lives_lost, n_boot=n_boot, seed=ci_seed),
        "win_rate": C.bootstrap_ci(win_rates, n_boot=n_boot, seed=ci_seed),
        "n_nemesis_total": len(pooled_margins),
        "margin_quantiles": _pooled_quantiles(pooled_margins),
        "engine_gap_seeds": sum(1 for r in per_seed if r["ended_early_engine_gap"]),
    }
    return {
        "deck": deck_key, "player": label, "target_fn": getattr(target_fn, "__name__", target_name),
        "n_seeds": len(seeds), "lives": lives, "max_antes": max_antes,
        "wall_clock_s": wall_s, "per_seed": per_seed, "summary": summary,
    }


# ============================================================================ mode (b): tournament

def tournament_cell(player_spec: str, seeds: Sequence[str], deck_key: str, n_agents: int = 32,
                     max_ante: int = 8, base_seed: int = 0, stake=WHITE_STAKE) -> dict:
    """Mode (b) for one (player, deck) cell: the evaluated player at population index 0 +
    `default_population(n_agents - 1, base_seed=base_seed)` (IDENTICAL composition across
    every deck this is called for, since `base_seed` is fixed by the caller -- only the deck
    changes), `Tournament(life_rule="none", max_ante=max_ante)` per seed. Per-seed record:
    `ante_ranks` (`{ante: rank}`, 1 = best; only antes the evaluated player was present for --
    `life_rule="none"` means this should be every ante 2..max_ante), `mean_rank_frac`
    (`(rank - 1) / (n_agents - 1)` averaged over those antes, 0 = best / 1 = worst,
    comparable across cells since `n_agents` is fixed)."""
    label = player_spec
    eval_player = _build_tournament_player(player_spec)
    background = default_population(n_agents - 1, base_seed=base_seed)
    players = [eval_player] + background
    per_seed = []
    t0 = time.time()
    for seed in seeds:
        res = Tournament(seed=seed, n_agents=n_agents, players=players, deck_key=deck_key,
                          stake=stake, life_rule="none", max_ante=max_ante).run()
        ante_ranks = {}
        for m in res.ante_matrices:
            r = float(m.rank[0])
            if r == r:   # not NaN -- agent 0 present this ante
                ante_ranks[m.ante] = r
        rank_fracs = [(r - 1.0) / (n_agents - 1.0) for r in ante_ranks.values()]
        mean_rank_frac = statistics.fmean(rank_fracs) if rank_fracs else float("nan")
        per_seed.append({
            "seed": res.seed, "ante_ranks": ante_ranks, "n_antes_present": len(ante_ranks),
            "mean_rank_frac": mean_rank_frac,
        })
    wall_s = time.time() - t0

    mean_rank_fracs = [r["mean_rank_frac"] for r in per_seed if r["mean_rank_frac"] == r["mean_rank_frac"]]
    summary = {"mean_rank_frac": C.bootstrap_ci(mean_rank_fracs)}
    return {
        "deck": deck_key, "player": label, "n_agents": n_agents, "max_ante": max_ante,
        "base_seed": base_seed, "n_seeds": len(seeds),
        "wall_clock_s": wall_s, "per_seed": per_seed, "summary": summary,
    }


# ============================================================================ cross-cell spread

def _cross_cell_bootstrap(cell_seed_maps: Sequence[dict], n_boot: int = 2000, seed: int = 0) -> dict:
    """`cell_seed_maps`: one `{seed: scalar}` map per cell, SAME ORDER as `decks` (may
    repeat a deck -- the identical-cell sanity case deliberately does). Only seeds present
    (non-NaN) in EVERY cell are used (paired). Point range/variance are computed directly
    from the per-cell means; the CI resamples the shared seed set -- the SAME resampled
    indices applied to every cell each replicate, preserving the pairing -- and recomputes
    each cell's mean and the resulting range/variance on every replicate."""
    n_cells = len(cell_seed_maps)
    common_seeds = None
    for m in cell_seed_maps:
        s = set(k for k, v in m.items() if v == v)
        common_seeds = s if common_seeds is None else (common_seeds & s)
    common_seeds = sorted(common_seeds or [])
    n = len(common_seeds)
    if n_cells < 2 or n == 0:
        return {"n_paired_seeds": n, "per_cell_mean": [None] * n_cells,
                "point_range": float("nan"), "point_variance": float("nan"),
                "range_ci": None, "variance_ci": None}
    arrays = [[m[s] for s in common_seeds] for m in cell_seed_maps]

    def _cell_means(idxs) -> list:
        return [statistics.fmean(arr[i] for i in idxs) for arr in arrays]

    all_idx = range(n)
    point_means = _cell_means(all_idx)
    point_range = max(point_means) - min(point_means)
    point_var = statistics.pvariance(point_means) if n_cells > 1 else 0.0

    if n < 2:
        return {"n_paired_seeds": n, "per_cell_mean": point_means, "point_range": point_range,
                "point_variance": point_var, "range_ci": {"lo": point_range, "hi": point_range},
                "variance_ci": {"lo": point_var, "hi": point_var}}

    rng = random.Random(seed)
    range_boot, var_boot = [], []
    for _ in range(n_boot):
        idxs = [rng.randrange(n) for _ in range(n)]
        means = _cell_means(idxs)
        range_boot.append(max(means) - min(means))
        var_boot.append(statistics.pvariance(means) if n_cells > 1 else 0.0)
    range_boot.sort()
    var_boot.sort()
    lo_i = int(0.025 * n_boot)
    hi_i = min(n_boot - 1, int(0.975 * n_boot) - 1)
    return {
        "n_paired_seeds": n, "per_cell_mean": point_means,
        "point_range": point_range, "point_variance": point_var,
        "range_ci": {"lo": range_boot[max(0, lo_i)], "hi": range_boot[hi_i]},
        "variance_ci": {"lo": var_boot[max(0, lo_i)], "hi": var_boot[hi_i]},
    }


# ============================================================================ top level

def evaluate_player(player_spec: str, mode: str = "both", decks: Sequence[str] = DECKS,
                     stake=WHITE_STAKE, solo_seeds: Optional[Sequence[str]] = None,
                     n_extra_seeds: int = 24, max_antes: int = 8, lives: Optional[int] = None,
                     target_name: str = "vanilla_boss", target_kwargs: Optional[dict] = None,
                     tournament_seeds: Optional[Sequence[str]] = None, tournament_n_seeds: int = 16,
                     n_agents: int = 32, base_seed: int = 0, n_boot: int = 2000,
                     ci_seed: int = 0, extra_seed_rng: int = 20260821) -> dict:
    """Evaluate `player_spec` on every deck in `decks` (default: Red/Checkered/Plasma,
    White stake), paired by seed. `mode`: "solo" (a only), "tournament" (b only), "both"
    (default). `decks` MAY repeat a deck key (the identical-cell sanity case)."""
    if mode not in ("solo", "tournament", "both"):
        raise ValueError(f"mode must be 'solo'/'tournament'/'both', got {mode!r}")
    decks = list(decks)
    if solo_seeds is None:
        solo_seeds = list(C.DEFAULT_SEEDS)
        if n_extra_seeds:
            solo_seeds = solo_seeds + RD.make_extra_seeds(n_extra_seeds, rng_seed=extra_seed_rng)
    else:
        solo_seeds = list(solo_seeds)
    if tournament_seeds is None:
        tournament_seeds = solo_seeds[:tournament_n_seeds]
    else:
        tournament_seeds = list(tournament_seeds)

    cells = []
    t0 = time.time()
    for deck in decks:
        cell = {"deck": deck}
        if mode in ("solo", "both"):
            cell["solo"] = solo_cell(player_spec, solo_seeds, deck, stake=stake, max_antes=max_antes,
                                     lives=lives, target_name=target_name, target_kwargs=target_kwargs,
                                     n_boot=n_boot, ci_seed=ci_seed)
        if mode in ("tournament", "both"):
            cell["tournament"] = tournament_cell(player_spec, tournament_seeds, deck, n_agents=n_agents,
                                                 max_ante=max_antes, base_seed=base_seed, stake=stake)
        cells.append(cell)
    wall_s = time.time() - t0

    cross = {}
    if mode in ("solo", "both"):
        maps = [{r["seed"]: r["win_rate"] for r in c["solo"]["per_seed"]} for c in cells]
        cross["win_rate"] = _cross_cell_bootstrap(maps, n_boot=n_boot, seed=ci_seed)
    if mode in ("tournament", "both"):
        maps = [{r["seed"]: r["mean_rank_frac"] for r in c["tournament"]["per_seed"]} for c in cells]
        cross["rank_frac"] = _cross_cell_bootstrap(maps, n_boot=n_boot, seed=ci_seed)

    return {
        "player": player_spec, "mode": mode, "decks": decks, "stake": stake,
        "n_solo_seeds": len(solo_seeds) if mode in ("solo", "both") else 0,
        "n_tournament_seeds": len(tournament_seeds) if mode in ("tournament", "both") else 0,
        "n_agents": n_agents, "max_antes": max_antes, "target_fn": target_name,
        "wall_clock_s": wall_s,
        "cells": cells,
        "cross_cell_spread": cross,
    }


# ============================================================================ markdown report

def _fmt_ci(d: Optional[dict], nd: int = 3) -> str:
    if d is None:
        return "n/a"
    return f"{d['point']:.{nd}f} [{d['lo']:.{nd}f},{d['hi']:.{nd}f}]"


def to_markdown(result: dict) -> str:
    lines = [f"# transfer spread -- `{result['player']}`", ""]
    lines.append(f"mode={result['mode']}  stake={result['stake']}  max_antes={result['max_antes']}  "
                f"target={result['target_fn']}  n_solo_seeds={result['n_solo_seeds']}  "
                f"n_tournament_seeds={result['n_tournament_seeds']}  n_agents={result['n_agents']}  "
                f"wall_clock={result['wall_clock_s']:.1f}s")
    lines.append("")

    if result["mode"] in ("solo", "both"):
        lines.append("## mode (a): SP-MLB-solo vs external target")
        lines.append("")
        lines.append("| deck | furthest ante | lives lost | win rate | margin p10 | p50 | p90 |")
        lines.append("|---|---|---|---|---|---|---|")
        for c in result["cells"]:
            s = c["solo"]["summary"]
            mq = s["margin_quantiles"]
            lines.append(f"| {c['deck']} | {_fmt_ci(s['furthest_ante'], 2)} | "
                        f"{_fmt_ci(s['lives_lost'], 2)} | {_fmt_ci(s['win_rate'])} | "
                        f"{mq.get('0.1')} | {mq.get('0.5')} | {mq.get('0.9')} |")
        lines.append("")

    if result["mode"] in ("tournament", "both"):
        lines.append("## mode (b): tournament population rank")
        lines.append("")
        lines.append("| deck | mean rank_frac (0=best, 1=worst) |")
        lines.append("|---|---|")
        for c in result["cells"]:
            lines.append(f"| {c['deck']} | {_fmt_ci(c['tournament']['summary']['mean_rank_frac'])} |")
        lines.append("")

    lines.append("## cross-cell spread (bootstrap over paired seeds)")
    lines.append("")
    deck_header = " | ".join(c["deck"] for c in result["cells"])
    lines.append(f"| metric | {deck_header} | range [95% CI] | variance [95% CI] | n_paired_seeds |")
    lines.append("|---|" + "---|" * len(result["cells"]) + "---|---|---|")
    for key, nd in (("win_rate", 3), ("rank_frac", 3)):
        if key not in result["cross_cell_spread"]:
            continue
        cc = result["cross_cell_spread"][key]
        means = cc["per_cell_mean"]
        mean_cells = " | ".join("n/a" if v is None else f"{v:.{nd}f}" for v in means)
        rng = cc["range_ci"]
        var = cc["variance_ci"]
        rng_s = "n/a" if rng is None else f"{cc['point_range']:.{nd}f} [{rng['lo']:.{nd}f},{rng['hi']:.{nd}f}]"
        var_s = "n/a" if var is None else f"{cc['point_variance']:.{nd + 1}f} [{var['lo']:.{nd + 1}f},{var['hi']:.{nd + 1}f}]"
        lines.append(f"| {key} | {mean_cells} | {rng_s} | {var_s} | {cc['n_paired_seeds']} |")
    lines.append("")
    return "\n".join(lines)


# ============================================================================ CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--player", required=True, help="e.g. 'scripted:hand=greedy,reroll=1,buy=1'")
    ap.add_argument("--mode", choices=("solo", "tournament", "both"), default="both")
    ap.add_argument("--decks", default=",".join(DECKS), help="comma-separated deck keys (may repeat)")
    ap.add_argument("--stake", default="1")
    ap.add_argument("--max-antes", type=int, default=8)
    ap.add_argument("--lives", type=int, default=None, help="solo mode only; default MLB_STARTING_LIVES")
    ap.add_argument("--target", default="vanilla_boss", help="targets.py registry name")
    ap.add_argument("--solo-seeds", default=None, help="comma-separated; default 126 ground-truth + --n-extra-seeds")
    ap.add_argument("--n-extra-seeds", type=int, default=24)
    ap.add_argument("--tournament-seeds", default=None, help="comma-separated; default: first --tournament-n-seeds of the solo seed list")
    ap.add_argument("--tournament-n-seeds", type=int, default=16)
    ap.add_argument("--n-agents", type=int, default=32)
    ap.add_argument("--base-seed", type=int, default=0, help="tournament background-population seed")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", required=True, help="JSON report path; a sibling .md is also written")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    stake = int(args.stake) if args.stake.isdigit() else args.stake
    decks = tuple(d.strip() for d in args.decks.split(",") if d.strip())
    solo_seeds = [s.strip() for s in args.solo_seeds.split(",") if s.strip()] if args.solo_seeds else None
    tournament_seeds = ([s.strip() for s in args.tournament_seeds.split(",") if s.strip()]
                        if args.tournament_seeds else None)

    result = evaluate_player(
        args.player, mode=args.mode, decks=decks, stake=stake, solo_seeds=solo_seeds,
        n_extra_seeds=args.n_extra_seeds, max_antes=args.max_antes, lives=args.lives,
        target_name=args.target, tournament_seeds=tournament_seeds,
        tournament_n_seeds=args.tournament_n_seeds, n_agents=args.n_agents,
        base_seed=args.base_seed, n_boot=args.n_boot,
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, default=str)
    md_path = os.path.splitext(args.out)[0] + ".md"
    md = to_markdown(result)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    if not args.quiet:
        print(md)
        print(f"-> {args.out}\n-> {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
