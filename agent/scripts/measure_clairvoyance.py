"""
measure_clairvoyance.py — Phase 5 W2 gate 1: how much of `real1`'s play was reading the
future.

Phase 4's MCTS (`mcts/search.py` + `mcts/player.py::MCTSPlayer`) is CLAIRVOYANT: every
simulation clones the true game (`BalatroGame.clone()`), which copies the keyed RNG and
draw-pile order verbatim, so every simulated future saw the actual future draws, reroll
results, pack contents and probability rolls. `mp/engine/balatro_sim/game.py`'s
`clone_determinized(seed)` (DETERMINIZE_NOTES.md) fixes the primitive; `mcts/determinize.py`
wires it into a `DeterminizedMCTSPlayer` without touching search.py/player.py.

This script measures the gap on `real1/latest.pt` (106-gen MCTS checkpoint, the baseline
Phase 5 replaces) with the exact search hyperparameters it was TRAINED with
(`mp/agent/runs/real1.sh` Stage B: sims=40, encoder=set, heuristic_prior=0.4,
heuristic_tau=0.35, max_hand_candidates=32, strategy=gumbel):

  (a) OUTCOME table: the clairvoyant player and the determinized player each play an
      INDEPENDENT full vanilla game (max ante 8) on the same seed. Paired-by-seed
      bootstrap CI on the difference in: final ante, ante-1/2/3 clear rate, blinds
      cleared, final $.
  (b) DISAGREEMENT table: drive the CLAIRVOYANT trajectory (its own actual choices); at
      every REAL decision point (>1 legal action — forced/single-legal states are
      skipped, there is nothing to disagree about) also ask a determinized player
      (same `sims`, fresh per probe) what it would choose from that same true state.
      Agreement rate broken down by `action["type"]`.
  Wall-clock per decision is recorded for both players in both (a) and (b).

Seeds parallelize over `multiprocessing` (one worker per seed: both players' full
outcome games + the disagreement walk, so a worker's checkpoint load is amortised over
~2x max_ante x 3 games' worth of decisions). `--processes` bounds concurrency; each
worker pins `torch.set_num_threads(1)` (eval_checkpoint.py's finding: an unpinned
single-leaf net actively hurts on a box running many workers).

    python mp/agent/scripts/measure_clairvoyance.py \\
        --checkpoint mp/agent/runs/real1/latest.pt \\
        --n-seeds 30 --sims 40 --processes 16 \\
        --out-json mp/results/clairvoyance_2026-08-23.json \\
        --out-md   mp/results/clairvoyance_2026-08-23.md

Smoke test (fast, proves the pipeline without the full job):

    python mp/agent/scripts/measure_clairvoyance.py --n-seeds 2 --sims 8 --processes 2 \\
        --out-json mp/results/clairvoyance_smoke.json --out-md mp/results/clairvoyance_smoke.md
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (sys.path + fork guard: mp/agent, mp/engine)

_MP_ROOT = Path(__file__).resolve().parents[2]
if str(_MP_ROOT / "eval") not in sys.path:
    sys.path.insert(0, str(_MP_ROOT / "eval"))

import common as C  # noqa: E402  (mp/eval/common.py — read-only: DEFAULT_SEEDS, bootstrap_ci)

REAL1_FLAGS = dict(
    encoder="set", strategy="gumbel", heuristic_prior=0.4, heuristic_tau=0.35,
    max_hand_candidates=32, heuristic_exact_top=8, heuristic_discard_bias=1.0,
)
ACTION_TYPES = ["play", "discard", "skip_blind", "play_blind", "buy", "reroll",
               "pick_booster", "use_consumable", "leave_shop", "sell", "other"]
_BLIND_ORDER = {"Small": 0, "Big": 1, "Boss": 2}


def _bucket(action_type: str) -> str:
    return "sell" if action_type == "sell_joker" else (
        action_type if action_type in ACTION_TYPES else "other")


def _furthest_key(game) -> tuple:
    return (game.ante, _BLIND_ORDER.get(game.current_blind.kind, 0))


def _build_players(checkpoint, sims, device, seed, determinize_seed, determinize_mode):
    """One clairvoyant `MCTSPlayer` + one `DeterminizedMCTSPlayer`, identical hyperparameters
    otherwise (the ONLY variable this measurement isolates is what the search can see)."""
    from mcts.player import make_player
    from mcts.determinize import make_determinized_player
    clair = make_player(checkpoint=checkpoint, sims=sims, device=device, seed=seed,
                        reuse=True, **REAL1_FLAGS)
    det = make_determinized_player(checkpoint=checkpoint, sims=sims, device=device, seed=seed,
                                   determinize_seed=determinize_seed, mode=determinize_mode,
                                   reuse=True, **REAL1_FLAGS)
    return clair, det


def _play_outcome(player, seed: str, deck_key: str, stake, max_steps: int) -> dict:
    """One full vanilla game (to GAME_OVER — win at ante 9 or a lost blind). Timing is
    over REAL decisions only (>1 legal action); forced/single-legal steps are free."""
    from balatro_sim.game import BalatroGame, State
    game = BalatroGame(seed=seed, deck_key=deck_key, stake=stake, ruleset="vanilla")
    player.reset()
    furthest = _furthest_key(game)
    decision_times = []
    steps = 0
    while game.state != State.GAME_OVER and steps < max_steps:
        legal = game.legal_actions()
        t0 = time.perf_counter()
        a = player.act(game)
        dt = time.perf_counter() - t0
        if len(legal) > 1:
            decision_times.append(dt)
        if a is None:
            a = {"type": "advance"}
        game.step(a)
        steps += 1
        key = _furthest_key(game)
        if key > furthest:
            furthest = key
    return {
        "seed": game.seed_str,
        "won": bool(game._obs().won),
        "final_ante": game.ante,
        "furthest_ante": furthest[0],
        "furthest_blind_idx": furthest[1],
        "blinds_cleared": max(0, 3 * (furthest[0] - 1) + furthest[1]),
        "final_money": game.dollars,
        "steps": steps,
        "n_decisions": len(decision_times),
        "mean_decision_s": statistics.fmean(decision_times) if decision_times else 0.0,
        "total_decision_s": sum(decision_times),
    }


def _walk_disagreement(clair_player, det_player_factory, seed: str, deck_key: str, stake,
                       max_steps: int) -> list:
    """Drive the CLAIRVOYANT trajectory; at every REAL decision (>1 legal action) also ask
    a FRESH determinized player (own factory call — no cache carried between unrelated
    probes) the same question from the SAME true state. Neither player's `.act()` mutates
    `game` (see mcts/determinize.py / mcts/player.py), so both can be asked in either
    order without cloning defensively."""
    from balatro_sim.game import BalatroGame, State
    from mcts.action import action_key
    game = BalatroGame(seed=seed, deck_key=deck_key, stake=stake, ruleset="vanilla")
    clair_player.reset()
    records = []
    steps = 0
    while game.state != State.GAME_OVER and steps < max_steps:
        legal = game.legal_actions()
        if len(legal) <= 1:
            a = clair_player.act(game)
            game.step(a if a is not None else {"type": "advance"})
            steps += 1
            continue
        t0 = time.perf_counter()
        a_clair = clair_player.act(game)
        t_clair = time.perf_counter() - t0
        det_player = det_player_factory()
        t0 = time.perf_counter()
        a_det = det_player.act(game)
        t_det = time.perf_counter() - t0
        records.append({
            "type": _bucket((a_clair or {}).get("type", "other")),
            "agree": (a_det is not None and action_key(a_clair) == action_key(a_det)),
            "t_clair_s": t_clair, "t_det_s": t_det,
        })
        game.step(a_clair if a_clair is not None else {"type": "advance"})
        steps += 1
    return records


def _worker(args) -> dict:
    (seed, checkpoint, sims, device, deck_key, stake, max_steps,
     determinize_seed_base, determinize_mode) = args
    import torch
    torch.set_num_threads(1)
    det_seed = None if determinize_seed_base is None else (determinize_seed_base ^ hash(seed)) & ((1 << 40) - 1)

    clair, det = _build_players(checkpoint, sims, device, seed=0,
                                determinize_seed=det_seed, determinize_mode=determinize_mode)
    t0 = time.time()
    outcome_clair = _play_outcome(clair, seed, deck_key, stake, max_steps)
    t_clair_wall = time.time() - t0

    t0 = time.time()
    outcome_det = _play_outcome(det, seed, deck_key, stake, max_steps)
    t_det_wall = time.time() - t0

    # A fresh determinized player per disagreement probe (own factory call — see
    # `_walk_disagreement`'s docstring); reuse the clairvoyant player already built above
    # so its trajectory is the one actually taken.
    def _det_factory():
        _, d = _build_players(checkpoint, sims, device, seed=0,
                              determinize_seed=det_seed, determinize_mode=determinize_mode)
        return d

    clair2, _ = _build_players(checkpoint, sims, device, seed=0,
                               determinize_seed=det_seed, determinize_mode=determinize_mode)
    t0 = time.time()
    disagreement = _walk_disagreement(clair2, _det_factory, seed, deck_key, stake, max_steps)
    t_disagree_wall = time.time() - t0

    return {
        "seed": seed,
        "outcome_clairvoyant": outcome_clair,
        "outcome_determinized": outcome_det,
        "disagreement": disagreement,
        "wall_s": {"clairvoyant_game": t_clair_wall, "determinized_game": t_det_wall,
                  "disagreement_walk": t_disagree_wall},
    }


# ══════════════════════════════════════════════════════════════════════════ aggregation / report

def _aggregate(per_seed: list, n_boot: int) -> dict:
    fields = ["final_ante", "blinds_cleared", "final_money", "mean_decision_s"]
    out = {"outcome": {}, "disagreement": {}}
    for arm in ("clairvoyant", "determinized"):
        vals = {f: [r[f"outcome_{arm}"][f] for r in per_seed] for f in fields}
        vals["ante1_clear"] = [1.0 if r[f"outcome_{arm}"]["furthest_ante"] >= 2 else 0.0 for r in per_seed]
        vals["ante2_clear"] = [1.0 if r[f"outcome_{arm}"]["furthest_ante"] >= 3 else 0.0 for r in per_seed]
        vals["ante3_clear"] = [1.0 if r[f"outcome_{arm}"]["furthest_ante"] >= 4 else 0.0 for r in per_seed]
        out["outcome"][arm] = {k: C.bootstrap_ci(v, n_boot=n_boot, seed=0) for k, v in vals.items()}
    diff_fields = fields + ["ante1_clear", "ante2_clear", "ante3_clear"]
    # raw per-seed vectors for paired diffs (clairvoyant - determinized)
    raw = {"clairvoyant": {}, "determinized": {}}
    for arm in ("clairvoyant", "determinized"):
        raw[arm]["final_ante"] = [r[f"outcome_{arm}"]["final_ante"] for r in per_seed]
        raw[arm]["blinds_cleared"] = [r[f"outcome_{arm}"]["blinds_cleared"] for r in per_seed]
        raw[arm]["final_money"] = [r[f"outcome_{arm}"]["final_money"] for r in per_seed]
        raw[arm]["mean_decision_s"] = [r[f"outcome_{arm}"]["mean_decision_s"] for r in per_seed]
        raw[arm]["ante1_clear"] = [1.0 if r[f"outcome_{arm}"]["furthest_ante"] >= 2 else 0.0 for r in per_seed]
        raw[arm]["ante2_clear"] = [1.0 if r[f"outcome_{arm}"]["furthest_ante"] >= 3 else 0.0 for r in per_seed]
        raw[arm]["ante3_clear"] = [1.0 if r[f"outcome_{arm}"]["furthest_ante"] >= 4 else 0.0 for r in per_seed]
    diffs = {f: C.paired_bootstrap_ci(raw["clairvoyant"][f], raw["determinized"][f], n_boot=n_boot, seed=0)
            for f in diff_fields}
    out["outcome_diff_clairvoyant_minus_determinized"] = diffs

    # disagreement: overall + by action type
    all_recs = [rec for r in per_seed for rec in r["disagreement"]]
    out["disagreement"]["n_probes"] = len(all_recs)
    out["disagreement"]["overall_agree_rate"] = (
        C.bootstrap_ci([1.0 if rec["agree"] else 0.0 for rec in all_recs], n_boot=n_boot, seed=0)
        if all_recs else {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0})
    by_type = {}
    for t in ACTION_TYPES:
        recs_t = [rec for rec in all_recs if rec["type"] == t]
        if not recs_t:
            continue
        by_type[t] = {
            "n": len(recs_t),
            "agree_rate": C.bootstrap_ci([1.0 if r["agree"] else 0.0 for r in recs_t],
                                         n_boot=n_boot, seed=0),
        }
    out["disagreement"]["by_type"] = by_type
    out["disagreement"]["mean_t_clair_s"] = (statistics.fmean(r["t_clair_s"] for r in all_recs)
                                             if all_recs else 0.0)
    out["disagreement"]["mean_t_det_s"] = (statistics.fmean(r["t_det_s"] for r in all_recs)
                                           if all_recs else 0.0)
    return out


def _write_markdown(path: str, meta: dict, agg: dict, per_seed: list) -> None:
    lines = []
    lines.append(f"# Clairvoyance measurement — {meta['checkpoint']}")
    lines.append("")
    lines.append(f"Seeds: {meta['n_seeds']} ({', '.join(meta['seeds'][:8])}"
                 f"{'...' if meta['n_seeds'] > 8 else ''}) · sims={meta['sims']} · "
                 f"determinize_mode={meta['determinize_mode']} · ruleset=vanilla · max wall {meta['wall_s']:.0f}s")
    lines.append("")
    lines.append("## (a) Outcome table — clairvoyant vs determinized, paired by seed")
    lines.append("")
    lines.append("| field | clairvoyant | determinized | diff (clair - det) | 95% CI |")
    lines.append("|---|---|---|---|---|")
    field_labels = [("final_ante", "mean final ante"), ("blinds_cleared", "mean blinds cleared"),
                    ("ante1_clear", "ante-1 clear rate"), ("ante2_clear", "ante-2 clear rate"),
                    ("ante3_clear", "ante-3 clear rate"), ("final_money", "mean final $"),
                    ("mean_decision_s", "mean s/decision")]
    for key, label in field_labels:
        c = agg["outcome"]["clairvoyant"][key]
        d = agg["outcome"]["determinized"][key]
        diff = agg["outcome_diff_clairvoyant_minus_determinized"][key]
        lines.append(f"| {label} | {c['point']:.3f} | {d['point']:.3f} | {diff['point']:+.3f} "
                     f"| [{diff['lo']:+.3f}, {diff['hi']:+.3f}] |")
    lines.append("")
    lines.append("## (b) Disagreement table — determinized vs clairvoyant's own trajectory")
    lines.append("")
    dis = agg["disagreement"]
    lines.append(f"{dis['n_probes']} real decision points probed (>1 legal action). "
                f"Overall agreement: {dis['overall_agree_rate']['point']:.3f} "
                f"[{dis['overall_agree_rate']['lo']:.3f}, {dis['overall_agree_rate']['hi']:.3f}]")
    lines.append("")
    lines.append("| action type | n | agreement rate | 95% CI |")
    lines.append("|---|---|---|---|")
    for t, d in sorted(dis["by_type"].items(), key=lambda kv: -kv[1]["n"]):
        r = d["agree_rate"]
        lines.append(f"| {t} | {d['n']} | {r['point']:.3f} | [{r['lo']:.3f}, {r['hi']:.3f}] |")
    lines.append("")
    lines.append(f"Mean wall-clock per probed decision: clairvoyant {dis['mean_t_clair_s']*1000:.1f} ms, "
                f"determinized {dis['mean_t_det_s']*1000:.1f} ms.")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("(fill in by hand after a full run: point at the outcome diff and the "
                f"agreement-by-type breakdown above)")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=str(_MP_ROOT / "agent" / "runs" / "real1" / "latest.pt"))
    ap.add_argument("--n-seeds", type=int, default=30)
    ap.add_argument("--seeds", default=None, help="comma-separated; overrides --n-seeds")
    ap.add_argument("--sims", type=int, default=40)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--deck", default="b_red")
    ap.add_argument("--stake", default="1")
    ap.add_argument("--max-steps", type=int, default=20_000)
    ap.add_argument("--determinize-mode", choices=["per_sim", "per_search"], default="per_sim")
    ap.add_argument("--determinize-seed-base", type=int, default=0,
                    help="base seed for the determinize world stream, XORed per-seed for "
                         "reproducibility; pass -1 for fresh (secrets) randomness")
    ap.add_argument("--processes", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args(argv)

    seeds = ([s.strip() for s in args.seeds.split(",") if s.strip()] if args.seeds
             else list(C.DEFAULT_SEEDS)[:args.n_seeds])
    stake = int(args.stake) if str(args.stake).isdigit() else args.stake
    det_base = None if args.determinize_seed_base == -1 else args.determinize_seed_base
    n_proc = args.processes or min(len(seeds), os.cpu_count() or 4)

    tasks = [(s, args.checkpoint, args.sims, args.device, args.deck, stake, args.max_steps,
             det_base, args.determinize_mode) for s in seeds]

    t0 = time.time()
    if n_proc <= 1:
        per_seed = [_worker(t) for t in tasks]
    else:
        with mp.Pool(processes=n_proc) as pool:
            per_seed = pool.map(_worker, tasks)
    wall_s = time.time() - t0

    agg = _aggregate(per_seed, args.n_boot)
    meta = {"checkpoint": os.path.abspath(args.checkpoint), "n_seeds": len(seeds), "seeds": seeds,
           "sims": args.sims, "determinize_mode": args.determinize_mode,
           "determinize_seed_base": det_base, "processes": n_proc, "wall_s": wall_s,
           "real1_flags": REAL1_FLAGS}

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "aggregate": agg, "per_seed": per_seed}, f, indent=1, default=str)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_md)) or ".", exist_ok=True)
    _write_markdown(args.out_md, meta, agg, per_seed)

    print(f"{len(seeds)} seeds, {n_proc} processes, {wall_s:.1f}s wall -> {args.out_json} / {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
