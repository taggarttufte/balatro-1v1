"""
eval_checkpoint.py — evaluate a `train_cold` / `train_mlb` checkpoint with `eval`'s
harness and write a report in `eval/eval_harness.py`'s own JSON schema, so

    python -m eval.eval_harness --compare A.json B.json --out cmp.json

pairs the two by seed exactly as it does for scripted players.

Why this lives in `agent` and not in `eval`
------------------------------------------------
`eval/common.py::parse_player_spec` raises `NotImplementedError` for `checkpoint:` and
says, verbatim, that `agent` owns the checkpoint loader. `eval/**` is frozen for
Phase 4 W1, so rather than edit it this script imports the harness's own drivers
(`play_sp_vanilla` / `play_sp_mlb`) and its bootstrap CI, and emits the same record shape.
The comparison is then done by the frozen `--compare` path, unmodified.

    python agent/scripts/eval_checkpoint.py --checkpoint runs/x/latest.pt \\
        --mode sp_mlb --sims 60 --device cuda --out results/x.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (sys.path + fork guard)

_MP_ROOT = Path(__file__).resolve().parents[2]
if str(_MP_ROOT / "eval") not in sys.path:
    sys.path.insert(0, str(_MP_ROOT / "eval"))

import torch  # noqa: E402

import common as C  # noqa: E402  (eval/common.py — read-only)

from mcts.player import make_player  # noqa: E402


def build_report(checkpoint: str, mode: str, seeds, sims: int, device: str,
                 deck_key: str, stake, lives: int, max_antes: int, n_boot: int,
                 target_k: float, label: str, leaf_batch: int, strategy: str,
                 max_steps: int = 200_000, threads: int = 0) -> dict:
    # A small net evaluated one leaf at a time does not benefit from torch's default
    # intra-op parallelism, and on a box already running training it actively hurts:
    # measured 20 873 s of user CPU for 30 min of wall clock (~12 threads spinning).
    if threads:
        torch.set_num_threads(threads)
    player = make_player(checkpoint, sims=sims, device=device, strategy=strategy,
                         reuse=True, leaf_batch=leaf_batch, seed=0)
    policy = C.adapt_player(player)
    target_fn = C.own_big_blind_target(k=target_k) if mode == "sp_mlb" else None

    per_seed = []
    t0 = time.time()
    for seed in seeds:
        player.reset()
        if mode == "sp_vanilla":
            per_seed.append(C.play_sp_vanilla(seed, policy, deck_key=deck_key, stake=stake))
        elif mode == "sp_mlb":
            per_seed.append(C.play_sp_mlb(seed, policy, deck_key=deck_key, stake=stake,
                                          lives=lives, max_antes=max_antes,
                                          target_fn=target_fn, max_steps=max_steps))
        else:
            raise SystemExit(f"eval_checkpoint supports sp_vanilla / sp_mlb, not {mode!r}")
    wall = time.time() - t0

    summary = {}
    for field in sorted({k for r in per_seed for k, v in r.items()
                         if isinstance(v, (int, float)) and not isinstance(v, bool)}):
        vals = [r[field] for r in per_seed if field in r]
        summary[field] = C.bootstrap_ci(vals, n_boot=n_boot, seed=0)
    if mode == "sp_vanilla":
        summary["win_rate"] = C.bootstrap_ci([1.0 if r["won"] else 0.0 for r in per_seed],
                                             n_boot=n_boot, seed=0)
    return {
        "mode": mode, "player": label, "reference": None,
        "deck": deck_key, "stake": stake, "lives": lives, "max_antes": max_antes,
        "target_fn": (f"own_big_blind_target(k={target_k})" if mode == "sp_mlb" else None),
        "seeds": list(seeds), "n_seeds": len(seeds), "wall_clock_s": wall,
        "per_seed": per_seed, "summary": summary,
        "checkpoint": os.path.abspath(checkpoint), "sims": sims, "device": device,
        "max_steps": max_steps,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mode", choices=["sp_vanilla", "sp_mlb"], default="sp_mlb")
    ap.add_argument("--seeds", default=None, help="comma-separated; default: all 126")
    ap.add_argument("--n-seeds", type=int, default=None, help="use the first N default seeds")
    ap.add_argument("--sims", type=int, default=60)
    ap.add_argument("--leaf-batch", type=int, default=16)
    ap.add_argument("--strategy", choices=["gumbel", "puct"], default="gumbel")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--deck", default="b_red")
    ap.add_argument("--stake", default="1")
    ap.add_argument("--lives", type=int, default=C.MLB_STARTING_LIVES)
    ap.add_argument("--max-antes", type=int, default=8)
    ap.add_argument("--target-k", type=float, default=1.0)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--max-steps", type=int, default=200_000,
                    help="engine steps per seed before the harness truncates the episode "
                         "(sp_mlb only). Lower it to bound a pathological seed.")
    ap.add_argument("--threads", type=int, default=1,
                    help="torch intra-op threads; 1 is fastest for single-leaf inference "
                         "and avoids oversubscribing a box that is also training. 0 = leave "
                         "torch's default.")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    seeds = ([s.strip() for s in args.seeds.split(",") if s.strip()] if args.seeds
             else list(C.DEFAULT_SEEDS))
    if args.n_seeds:
        seeds = seeds[:args.n_seeds]
    stake = int(args.stake) if str(args.stake).isdigit() else args.stake

    report = build_report(
        args.checkpoint, args.mode, seeds, args.sims, args.device, args.deck, stake,
        args.lives, args.max_antes, args.n_boot, args.target_k,
        args.name or f"checkpoint:{args.checkpoint}", args.leaf_batch, args.strategy,
        args.max_steps, args.threads)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, default=str)
    print(f"{report['player']}: {len(seeds)} seeds in {report['wall_clock_s']:.0f}s "
          f"-> {args.out}")
    for field in sorted(report["summary"]):
        ci = report["summary"][field]
        mean = ci.get("mean", ci.get("point"))
        print(f"  {field:<24}{mean:>10.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
