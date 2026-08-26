"""
bench_parallel.py — how much does the box actually give us?

ONE command, run it when the machine is free:

    python agent/benchmarks/bench_parallel.py

It plays ONE generation of the real run's Stage B configuration (N=16 agents, 8 current +
4 scripted anchors + 4 past-self seats, set encoder, 40 sims, W0's heuristic prior on,
leaf_batch 1, skip cap 1) under each arm and reports sims/s, leaf evaluations/s and
generation wall clock:

    serial                the existing single-process path (`MLBTrainer`) — the baseline
    workers x {cpu,cuda}  N worker processes feeding ONE shared batched evaluator
    workers x local       N worker processes each running the net on its own core
                          (--include-local; the control arm for "is the shared evaluator
                          worth its transport on THIS box")

Nothing here touches `runs/real1/`.  The net is cold unless `--init <checkpoint>` is given;
throughput is dominated by search cost, not by what the weights say, and a cold net keeps
the benchmark reproducible.  Every arm plays the SAME seed, so the arms are comparable and
a difference in sims/s is a difference in throughput rather than in how far the agents got.

    --workers 1,4,8,12,16     worker counts to sweep (0 = the serial baseline)
    --devices cpu,cuda        evaluator devices to sweep
    --include-local           add the per-worker-net control arm
    --seeds-per-gen 1         tournaments per generation (the real run uses 2)
    --max-ante 8              the real run's horizon
    --max-wait-ms 0           evaluator batching budget (0 = pure opportunistic drain)
    --out <path>              JSON (default: benchmarks/bench_parallel_<date>.json)
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import _bootstrap  # noqa: E402,F401  (sys.path + fork guard)

HERE = Path(__file__).resolve().parent


# ══════════════════════════════════════════════════════════════════════ the arms

def build_configs(args):
    from train import MLBTrainConfig, TrainConfig

    cfg = TrainConfig(
        seed=args.seed, sims=args.sims, encoder="set", device="cpu", ruleset="mlb",
        deck_key="b_red", stake=1, lives=4, max_decisions=1500, max_antes=args.max_ante,
        batch_size=32, lr=1e-3, buffer_capacity=20_000, min_buffer=128,
        heuristic_prior=args.heuristic_prior, heuristic_tau=0.35,
        heuristic_prior_anneal="clear:0.5", heuristic_prior_floor=0.1,
        max_hand_candidates=32,
    )
    mlb = MLBTrainConfig(
        objective="tournament", n_agents=args.n_agents, m_current=args.n_agents // 2,
        p_history=4, seeds_per_generation=args.seeds_per_gen, max_ante=args.max_ante,
        life_rule="paired", anchor_frac=0.25, sims_budgets=(1.0, 0.5, 1.5),
        leaf_batch=1, reuse=True, max_skips_per_ante=1,
        skip_cap_anneal_clear_rate=0.5, train_steps=128, max_train_steps=128,
    )
    return cfg, mlb


def run_arm(args, workers: int, device: str, work_dir: Path) -> dict:
    """One generation under one arm.  Returns the row for the table."""
    from train import MLBTrainer
    from train.parallel import ParallelMLBTrainer
    from parallel.pool import PoolConfig

    cfg, mlb = build_configs(args)
    if workers <= 0:
        trainer = MLBTrainer(cfg, mlb)
    else:
        trainer = ParallelMLBTrainer(
            cfg, mlb, work_dir=str(work_dir),
            pool_cfg=PoolConfig(n_workers=workers, evaluator_device=device,
                                max_wait_ms=args.max_wait_ms,
                                request_mb=args.arena_mb))
    if args.init:
        from train_mlb import init_weights                   # noqa: WPS433
        init_weights(trainer, args.init, cfg.device)

    t0 = time.perf_counter()
    try:
        m = trainer.run_generation()
    finally:
        close = getattr(trainer, "close", None)
        if close is not None:
            close()
    wall = time.perf_counter() - t0

    extra = getattr(trainer, "extra_metrics", {})
    row = {
        "workers": workers, "device": ("serial" if workers <= 0 else device),
        "wall_s": round(wall, 2),
        "sims_per_s": round(m.sims_per_s, 1),
        "sims_per_s_per_worker": round(m.sims_per_s / max(1, workers), 1),
        "leaf_evals_per_s": (None if math.isnan(m.leaf_evals_per_s)
                             else round(m.leaf_evals_per_s, 1)),
        "searches": m.searches, "samples": m.n_samples, "episodes": m.episodes,
        "mean_ante": round(m.mean_ante_reached, 2),
        "eval_mean_batch": extra.get("eval_mean_batch"),
        "eval_forward_s": extra.get("eval_forward_s"),
        "eval_idle_s": extra.get("eval_idle_s"),
        "worker_wait_s_mean": extra.get("worker_wait_s_mean"),
        "dead_workers": extra.get("dead_workers", []),
    }
    return row


# ══════════════════════════════════════════════════════════════════════ reporting

def table(rows: list, baseline: float) -> str:
    head = (f"| {'arm':>16} | {'wall s':>7} | {'sims/s':>8} | {'per wkr':>7} | "
            f"{'leaf/s':>8} | {'batch':>6} | {'fwd s':>6} | {'wait s':>6} | {'x':>5} |")
    sep = "|" + "|".join("-" * len(c) for c in head.split("|")[1:-1]) + "|"
    lines = [head, sep]
    for r in rows:
        arm = ("serial" if r["workers"] <= 0 else f"{r['workers']}w {r['device']}")
        speed = (r["sims_per_s"] / baseline) if baseline else float("nan")
        lines.append(
            f"| {arm:>16} | {r['wall_s']:>7.1f} | {r['sims_per_s']:>8.0f} | "
            f"{r['sims_per_s_per_worker']:>7.0f} | "
            f"{(r['leaf_evals_per_s'] or 0):>8.0f} | "
            f"{(r['eval_mean_batch'] or 0):>6.2f} | "
            f"{(r['eval_forward_s'] or 0):>6.1f} | "
            f"{(r['worker_wait_s_mean'] or 0):>6.1f} | {speed:>5.2f} |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", default="1,4,8,12,16",
                    help="comma-separated worker counts (0 = the serial baseline)")
    ap.add_argument("--devices", default="cpu,cuda", help="evaluator devices to sweep")
    ap.add_argument("--include-local", action="store_true",
                    help="add the per-worker-net control arm")
    ap.add_argument("--no-serial", action="store_true", help="skip the serial baseline")
    ap.add_argument("--n-agents", type=int, default=16)
    ap.add_argument("--sims", type=int, default=40)
    ap.add_argument("--seeds-per-gen", type=int, default=1)
    ap.add_argument("--max-ante", type=int, default=8)
    ap.add_argument("--heuristic-prior", type=float, default=0.4)
    ap.add_argument("--max-wait-ms", type=float, default=0.0)
    ap.add_argument("--arena-mb", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=None,
                    help="load weights from a checkpoint (NOT the live run's — copy it "
                         "first if you want one)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch

    # The "local" arm caches the generation's weights on disk; keep that out of the repo.
    work_dir = Path(tempfile.mkdtemp(prefix="bench_parallel_"))
    counts = [int(x) for x in str(args.workers).split(",") if x.strip()]
    devices = [d.strip() for d in str(args.devices).split(",") if d.strip()]
    if "cuda" in devices and not torch.cuda.is_available():
        print("! CUDA is not available; dropping the cuda arms", flush=True)
        devices = [d for d in devices if d != "cuda"]
    if args.include_local:
        devices.append("local")

    arms: list = []
    if not args.no_serial:
        arms.append((0, "serial"))
    for device in devices:
        for w in counts:
            if w > 0:
                arms.append((w, device))

    print(f"=== bench_parallel: {len(arms)} arms, N={args.n_agents}, sims={args.sims}, "
          f"encoder=set, heuristic prior {args.heuristic_prior}, "
          f"{args.seeds_per_gen} seed(s)/gen, max_ante {args.max_ante} ===", flush=True)
    rows: list = []
    for workers, device in arms:
        label = "serial" if workers <= 0 else f"{workers} workers, evaluator {device}"
        print(f"  [{label}] ...", end="", flush=True)
        t0 = time.perf_counter()
        try:
            row = run_arm(args, workers, device, work_dir)
        except BaseException as exc:                          # noqa: BLE001
            print(f" FAILED: {type(exc).__name__}: {exc}", flush=True)
            rows.append({"workers": workers, "device": device, "error": repr(exc),
                         "wall_s": round(time.perf_counter() - t0, 2),
                         "sims_per_s": 0.0, "sims_per_s_per_worker": 0.0,
                         "leaf_evals_per_s": None})
            continue
        rows.append(row)
        print(f" {row['wall_s']:.1f} s, {row['sims_per_s']:.0f} sims/s", flush=True)

    baseline = next((r["sims_per_s"] for r in rows if r["workers"] <= 0), 0.0)
    print()
    print(table(rows, baseline))
    out = Path(args.out) if args.out else HERE / f"bench_parallel_{date.today()}.json"
    out.write_text(json.dumps({
        "date": str(date.today()), "args": vars(args), "rows": rows,
        "baseline_sims_per_s": baseline,
        "torch": torch.__version__,
        "cuda": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
    }, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out}")
    shutil.rmtree(work_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
