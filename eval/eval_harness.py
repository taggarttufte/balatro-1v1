"""
mp/eval/eval_harness.py -- evaluate a Player over a fixed seed list in three modes and produce
a JSON report with bootstrap CIs; ``--compare`` gives a paired-by-seed difference with CI
(common random numbers -- see mp/eval/EVAL_NOTES.md).

Modes:
  sp_vanilla  BalatroGame(ruleset="vanilla") to GAME_OVER.  win / furthest blind / final ante /
              final money.
  sp_mlb      BalatroGame(ruleset="mlb") solo, every Nemesis played to hand-exhaustion against
              a fixed EXTERNAL target function (default: the agent's own Big-Blind score that
              same ante x k=1, see common.own_big_blind_target -- "the default SP-MLB target"
              in EVAL_NOTES.md).  furthest ante / lives lost / money curve / per-Nemesis score.
  1v1         MLBMatch vs a --reference player (canonical alternation, Nemesis-to-exhaustion is
              the engine's own server rule).  win rate / lives margin / per-Nemesis log-score
              margin.

    python -m mp.eval.eval_harness --mode sp_vanilla --player scripted:hand=greedy,buy=1,pack=0 \\
        --out mp/results/demo_vanilla.json
    python -m mp.eval.eval_harness --mode sp_mlb --player scripted:reroll=1,buy=1 \\
        --out mp/results/demo_sp_mlb.json
    python -m mp.eval.eval_harness --mode 1v1 --player scripted:reroll=1,buy=1 \\
        --reference scripted:hand=weak --out mp/results/demo_1v1.json
    python -m mp.eval.eval_harness --compare mp/results/a.json mp/results/b.json \\
        --out mp/results/compare_a_b.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import common as C  # noqa: E402

MODES = ("sp_vanilla", "sp_mlb", "1v1")


def _numeric_fields(record: dict) -> dict:
    """Top-level int/float/bool fields of a per-seed result record -- what ``--compare`` pairs
    (lists/dicts like ``money_curve`` / ``nemesis_log`` / ``pvp_log`` are left out, they are not
    scalar per-seed outcomes)."""
    out = {}
    for k, v in record.items():
        if isinstance(v, bool):
            out[k] = float(v)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = float(v)
    return out


def evaluate(mode: str, player_spec: str, seeds, reference_spec: Optional[str] = None,
            deck_key: str = "b_red", stake=1, lives: int = C.MLB_STARTING_LIVES,
            max_antes: int = 8, n_boot: int = 2000, ci_seed: int = 0,
            target_k: float = 1.0) -> dict:
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r} (want one of {MODES})")
    label, policy = C.make_player_policy(player_spec)
    ref_label = ref_policy = None
    if mode == "1v1":
        if reference_spec is None:
            raise ValueError("--mode 1v1 requires --reference")
        ref_label, ref_policy = C.make_player_policy(reference_spec)
    target_fn = C.own_big_blind_target(k=target_k) if mode == "sp_mlb" else None

    per_seed = []
    t0 = time.time()
    for seed in seeds:
        if mode == "sp_vanilla":
            per_seed.append(C.play_sp_vanilla(seed, policy, deck_key=deck_key, stake=stake))
        elif mode == "sp_mlb":
            per_seed.append(C.play_sp_mlb(seed, policy, deck_key=deck_key, stake=stake,
                                          lives=lives, max_antes=max_antes, target_fn=target_fn))
        else:  # 1v1
            per_seed.append(C.play_1v1(seed, policy, ref_policy, deck_key=deck_key, stake=stake, lives=lives))
    wall_s = time.time() - t0

    summary = {}
    for field in sorted({k for r in per_seed for k, v in r.items()
                        if isinstance(v, (int, float)) and not isinstance(v, bool)}):
        vals = [r[field] for r in per_seed if field in r]
        summary[field] = C.bootstrap_ci(vals, n_boot=n_boot, seed=ci_seed)
    if mode == "sp_vanilla":
        summary["win_rate"] = C.bootstrap_ci([1.0 if r["won"] else 0.0 for r in per_seed],
                                             n_boot=n_boot, seed=ci_seed)
    if mode == "1v1":
        summary["win_rate"] = C.bootstrap_ci([1.0 if r["winner"] == 0 else 0.0 for r in per_seed],
                                             n_boot=n_boot, seed=ci_seed)

    return {
        "mode": mode, "player": label, "reference": ref_label,
        "deck": deck_key, "stake": stake, "lives": lives, "max_antes": max_antes,
        "target_fn": (f"own_big_blind_target(k={target_k})" if mode == "sp_mlb" else None),
        "seeds": list(seeds), "n_seeds": len(seeds), "wall_clock_s": wall_s,
        "per_seed": per_seed, "summary": summary,
    }


def compare(report_a: dict, report_b: dict, n_boot: int = 2000, ci_seed: int = 0) -> dict:
    """Paired-by-seed difference (A - B) with bootstrap CI over every scalar metric both
    reports share.  Common random numbers is the point: seeds are matched by value, and a
    seed missing from either report is dropped (with a note)."""
    if report_a["mode"] != report_b["mode"]:
        raise ValueError(f"--compare needs the same mode: {report_a['mode']!r} vs {report_b['mode']!r}")
    by_seed_a = {r["seed"]: r for r in report_a["per_seed"]}
    by_seed_b = {r["seed"]: r for r in report_b["per_seed"]}
    common_seeds = [s for s in report_a["seeds"] if s in by_seed_a and s in by_seed_b]
    dropped = (set(report_a["seeds"]) | set(report_b["seeds"])) - set(common_seeds)

    fields_a = set()
    fields_b = set()
    for s in common_seeds:
        fields_a |= set(_numeric_fields(by_seed_a[s]))
        fields_b |= set(_numeric_fields(by_seed_b[s]))
    fields = sorted(fields_a & fields_b)

    diffs = {}
    for field in fields:
        xs = [_numeric_fields(by_seed_a[s])[field] for s in common_seeds if field in _numeric_fields(by_seed_a[s])]
        ys = [_numeric_fields(by_seed_b[s])[field] for s in common_seeds if field in _numeric_fields(by_seed_b[s])]
        if len(xs) != len(ys) or len(xs) < 1:
            continue
        diffs[field] = C.paired_bootstrap_ci(xs, ys, n_boot=n_boot, seed=ci_seed)
        diffs[field]["mean_a"] = sum(xs) / len(xs)
        diffs[field]["mean_b"] = sum(ys) / len(ys)

    return {
        "mode": report_a["mode"],
        "player_a": report_a["player"], "player_b": report_b["player"],
        "n_paired_seeds": len(common_seeds), "dropped_seeds": sorted(dropped),
        "diffs": diffs,
    }


# ============================================================================ CLI

def _load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dump(obj: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=str)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=MODES)
    ap.add_argument("--player", help="e.g. 'scripted:hand=greedy,reroll=1,buy=1,pack=0' or 'checkpoint:<path>'")
    ap.add_argument("--reference", help="1v1 mode only: the opponent's player spec")
    ap.add_argument("--seeds", default=None, help="comma-separated; default: all 126 ground-truth seeds")
    ap.add_argument("--deck", default="b_red")
    ap.add_argument("--stake", default="1")
    ap.add_argument("--lives", type=int, default=C.MLB_STARTING_LIVES)
    ap.add_argument("--max-antes", type=int, default=8, help="sp_mlb only: harness cutoff (MLB is endless)")
    ap.add_argument("--target-k", type=float, default=1.0,
                   help="sp_mlb only: Nemesis target = k x the agent's own Big-Blind score that ante (default k=1)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", required=False, help="JSON report path (or the --compare output path)")
    ap.add_argument("--name", default=None, help="informational only; not required for the JSON schema")
    ap.add_argument("--compare", nargs=2, metavar=("A_JSON", "B_JSON"), default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if args.compare:
        ra, rb = _load(args.compare[0]), _load(args.compare[1])
        result = compare(ra, rb, n_boot=args.n_boot)
        out = args.out or str(C.RESULTS_DIR / "compare.json")
        _dump(result, out)
        if not args.quiet:
            print(f"compare {args.compare[0]} vs {args.compare[1]} ({result['n_paired_seeds']} paired seeds)")
            for field, d in result["diffs"].items():
                print(f"  {field}: A-B point={d['point']:.4g} CI=[{d['lo']:.4g},{d['hi']:.4g}] "
                     f"(mean_a={d['mean_a']:.4g} mean_b={d['mean_b']:.4g})")
            print(f"  -> {out}")
        return 0

    if not args.mode or not args.player:
        ap.error("either --compare A B, or --mode + --player, is required")
    stake = int(args.stake) if args.stake.isdigit() else args.stake
    seeds = ([s.strip() for s in args.seeds.split(",") if s.strip()] if args.seeds else list(C.DEFAULT_SEEDS))
    result = evaluate(args.mode, args.player, seeds, reference_spec=args.reference, deck_key=args.deck,
                      stake=stake, lives=args.lives, max_antes=args.max_antes, n_boot=args.n_boot,
                      target_k=args.target_k)
    out = args.out or str(C.RESULTS_DIR / f"{args.name or args.mode}.json")
    _dump(result, out)
    if not args.quiet:
        print(f"{args.mode}  player={result['player']}  reference={result['reference']}  "
             f"N={result['n_seeds']}  ({result['wall_clock_s']:.1f}s)")
        for field, s in result["summary"].items():
            print(f"  {field}: {s['point']:.4g} [{s['lo']:.4g},{s['hi']:.4g}] n={s['n']}")
        print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
