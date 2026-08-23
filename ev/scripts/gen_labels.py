"""
gen_labels.py — the label campaign driver (Phase 5 rev 2, W5).

    python mp/ev/scripts/gen_labels.py --run-dir mp/ev/runs/labels_a --seeds default+random:400 \
        --n-states 12 --n-rollouts 8 --workers 16 --minutes 600
    touch mp/ev/runs/labels_a/PAUSE          # stops submitting; in-flight jobs finish; flush
    python mp/ev/scripts/gen_labels.py --run-dir mp/ev/runs/labels_a --seeds default+random:400  # resumes

One pool job = one seed (``labels.label_job``): self-play → stratified snapshots → labels for
both perspectives → encoded rows.  Rows are buffered in the main process and flushed to
``<run-dir>/shards/shard_NNNN.npz`` every ``--flush-jobs`` jobs (or ``--shard-rows`` rows);
a seed is recorded in ``<run-dir>/done.ids`` only AFTER the shard holding its rows is on
disk, so a crash loses at most one unflushed buffer and never marks rows done that were
not saved.  Restarting with the same ``--run-dir`` and seed list skips the recorded seeds.

``--symmetry-jobs N``: the first N seeds also label player 1 from an independent rollout
set (``independent_perspectives``), for the sum-to-one check in the summary.

Outputs: ``<run-dir>/gen.jsonl`` (config / flush / summary), console one line per flush,
``mp/results/labels_<name>.json`` (dataset summary: label mean/sd by kind and ante, CI
widths, truncation fraction, symmetry check, throughput).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import string
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent          # mp/ev/scripts
EV_ROOT = HERE.parent                            # mp/ev
MP_ROOT = EV_ROOT.parent                         # mp
for _p in (str(EV_ROOT), str(MP_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _bootstrap  # noqa: E402,F401
import labels as L  # noqa: E402
import dataset as DS  # noqa: E402
import workers as W  # noqa: E402

RESULTS_DIR = MP_ROOT / "results"
PAUSE_FILE = "PAUSE"


def parse_seeds(spec: str, rng_seed: int = 0) -> list:
    """``default`` (the 126 ground-truth seeds) | ``random:N`` | ``file:<path>`` | a comma
    list | ``+``-joined combinations, e.g. ``default+random:400``."""
    out: list = []
    rng = random.Random(rng_seed)
    alphabet = string.ascii_uppercase + string.digits
    for part in spec.split("+"):
        part = part.strip()
        if not part:
            continue
        if part == "default":
            from eval.common import DEFAULT_SEEDS
            out.extend(DEFAULT_SEEDS)
        elif part.startswith("random:"):
            n = int(part.split(":", 1)[1])
            out.extend("".join(rng.choice(alphabet) for _ in range(8)) for _ in range(n))
        elif part.startswith("file:"):
            out.extend(l.strip() for l in Path(part[5:]).read_text(encoding="utf-8").splitlines() if l.strip())
        else:
            out.extend(s.strip() for s in part.split(",") if s.strip())
    seen = set()
    uniq = []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="label campaign driver (W5)")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--seeds", default="default+random:400")
    ap.add_argument("--seed-rng", type=int, default=0, help="for random: seeds")
    ap.add_argument("--n-states", type=int, default=12, help="snapshots per self-play match")
    ap.add_argument("--n-rollouts", type=int, default=8)
    ap.add_argument("--policy", default="auto", choices=["auto", "ev", "scripted"])
    ap.add_argument("--budget", default="fast")
    ap.add_argument("--epsilon-selfplay", type=float, default=0.1)
    ap.add_argument("--epsilon-rollout", type=float, default=0.02)
    ap.add_argument("--encoder", default="auto", choices=["auto", "v2", "dummy"])
    ap.add_argument("--max-ante", type=int, default=L.DEFAULT_MAX_ANTE)
    ap.add_argument("--deck", default="b_red")
    ap.add_argument("--stake", type=int, default=1)
    ap.add_argument("--lives", type=int, default=4)
    ap.add_argument("--policy-seed", type=int, default=0)
    ap.add_argument("--rollout-seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-jobs", type=int, default=None)
    ap.add_argument("--minutes", type=float, default=None)
    ap.add_argument("--flush-jobs", type=int, default=16, help="flush a shard every N jobs")
    ap.add_argument("--shard-rows", type=int, default=4000, help="...or when the buffer holds N rows")
    ap.add_argument("--symmetry-jobs", type=int, default=0)
    ap.add_argument("--allow-clairvoyant", action="store_true", help="plumbing only (no W2)")
    ap.add_argument("--name", default=None, help="results file name (default: run-dir name)")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = Path(args.run_dir)
    shards_dir = run_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    done_path = run_dir / "done.ids"
    pause_path = run_dir / PAUSE_FILE
    log_path = run_dir / "gen.jsonl"
    name = args.name or run_dir.name
    seeds = parse_seeds(args.seeds, args.seed_rng)
    done = W.load_done_ids(done_path)
    if pause_path.exists():
        pause_path.unlink()

    base = {"n_states": args.n_states, "n_rollouts": args.n_rollouts, "policy": args.policy,
            "budget": args.budget, "epsilon_selfplay": args.epsilon_selfplay,
            "epsilon_rollout": args.epsilon_rollout, "encoder": args.encoder, "max_ante": args.max_ante,
            "deck_key": args.deck, "stake": args.stake, "lives": args.lives,
            "policy_seed": args.policy_seed, "rollout_seed": args.rollout_seed,
            "allow_clairvoyant": args.allow_clairvoyant}

    def jobs():
        for i, s in enumerate(seeds):
            if s in done:
                continue
            payload = dict(base, seed=s)
            if i < args.symmetry_jobs:
                payload["independent_perspectives"] = True
            yield s, payload

    lf = open(log_path, "a", encoding="utf-8")

    def emit(rec: dict) -> None:
        lf.write(json.dumps(rec, default=str) + "\n")
        lf.flush()

    existing = sorted(shards_dir.glob("shard_*.npz"))
    state = {"buffer": [], "buffer_ids": [], "n_shards": len(existing), "rows": 0, "jobs": 0,
             "t0": time.perf_counter(), "timing": {"selfplay_s": 0.0, "label_s": 0.0, "n_rollouts": 0,
                                                  "rollout_decisions": 0, "rollout_policy_s": 0.0,
                                                  "n_snapshots": 0, "total_s": 0.0}}

    def flush(why: str) -> None:
        if not state["buffer"]:
            return
        path = shards_dir / f"shard_{state['n_shards']:04d}.npz"
        DS.save_shard(path, state["buffer"])
        for jid in state["buffer_ids"]:
            W.mark_done(done_path, jid)
            done.add(jid)
        n = len(state["buffer"])
        state["n_shards"] += 1
        state["rows"] += n
        el = time.perf_counter() - state["t0"]
        tm = state["timing"]
        rec = {"kind": "flush", "why": why, "shard": str(path), "rows": n, "jobs": len(state["buffer_ids"]),
               "rows_total": state["rows"], "jobs_total": state["jobs"], "elapsed_s": el,
               "labels_per_min": 60 * state["rows"] / el if el else 0.0,
               "ms_per_rollout": 1000 * tm["label_s"] / tm["n_rollouts"] if tm["n_rollouts"] else 0.0,
               "decisions_per_rollout": tm["rollout_decisions"] / tm["n_rollouts"] if tm["n_rollouts"] else 0.0,
               "policy_frac": tm["rollout_policy_s"] / tm["label_s"] if tm["label_s"] else 0.0,
               "selfplay_frac": tm["selfplay_s"] / tm["total_s"] if tm["total_s"] else 0.0}
        emit(rec)
        print(f"  [flush:{why}] {path.name} +{n} rows ({state['rows']} total, {state['jobs']} jobs, "
              f"{rec['labels_per_min']:.1f} labels/min, {rec['ms_per_rollout']:.0f} ms/rollout, "
              f"{rec['decisions_per_rollout']:.0f} dec/rollout, policy {100 * rec['policy_frac']:.0f}%)",
              flush=True)
        state["buffer"] = []
        state["buffer_ids"] = []

    def on_result(job_id, result) -> None:
        rows = L.rows_from_result(result)
        state["buffer"].extend(rows)
        state["buffer_ids"].append(job_id)
        state["jobs"] += 1
        for k, v in result["timing"].items():
            if k in state["timing"]:
                state["timing"][k] += v
        if len(state["buffer"]) >= args.shard_rows:
            flush("rows")

    def on_checkpoint(summary) -> None:
        flush("jobs")

    emit({"kind": "config", "timestamp": datetime.now().isoformat(timespec="seconds"), "args": vars(args),
          "payload_base": base, "n_seeds": len(seeds), "n_done_before": len(done),
          "has_determinize": L.has_determinize(), "has_ev_player": L.has_ev_player(),
          "has_encoder_v2": L.has_encoder_v2()})
    print(f"=== label campaign {name}: {len(seeds)} seeds ({len(done)} done), {args.workers} workers, "
          f"{args.n_states} states x {args.n_rollouts} rollouts, policy={args.policy} "
          f"(ev={L.has_ev_player()}), encoder={args.encoder} (v2={L.has_encoder_v2()}), "
          f"determinize={L.has_determinize()} ===")
    print(f"  run dir: {run_dir}   pause: touch {pause_path}")
    summary = W.run_pool(L.label_job, jobs(), n_workers=args.workers, on_result=on_result,
                         pause_file=pause_path, checkpoint_every=args.flush_jobs, on_checkpoint=on_checkpoint,
                         state_path=None, max_jobs=args.max_jobs,
                         deadline_s=(args.minutes * 60 if args.minutes else None))
    flush("exit")
    # dataset summary
    ds = DS.LabelDataset.load(shards_dir)
    dsum = ds.summary()
    sym = None
    if len(ds):
        pairs = {}
        for i, m in enumerate(ds.meta):
            if m.get("independent"):
                pairs.setdefault((m["seed"], m["step"]), {})[m["player"]] = float(ds.y[i])
        sums = [v[0] + v[1] for v in pairs.values() if 0 in v and 1 in v]
        if sums:
            import statistics
            sym = {"n_pairs": len(sums), "mean_y0_plus_y1": statistics.fmean(sums),
                   "sd": statistics.pstdev(sums) if len(sums) > 1 else 0.0}
    holdout = sorted(s for s in ds.seeds() if DS.seed_in_holdout(s, 0.1))
    out = {"name": name, "run_dir": str(run_dir), "timestamp": datetime.now().isoformat(timespec="seconds"),
           "pool": summary.as_dict(), "dataset": dsum, "symmetry_check": sym,
           "holdout_seeds_at_0.1": holdout, "n_holdout_rows_at_0.1": int(sum(
               1 for s in ds.columns["seed"].tolist() if DS.seed_in_holdout(s, 0.1))) if len(ds) else 0,
           "timing_totals": state["timing"], "config": base, "seeds_spec": args.seeds}
    emit({"kind": "summary", **out})
    lf.close()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    res_path = RESULTS_DIR / f"labels_{name}.json"
    res_path.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    why = "paused" if summary.paused else ("interrupted" if summary.interrupted else
                                            ("exhausted" if summary.exhausted else "deadline/max_jobs"))
    print(f"=== stopped ({why}): {summary.done} jobs ({summary.failed} failed), {len(ds)} rows in "
          f"{state['n_shards']} shards, {summary.jobs_per_min:.1f} jobs/min; summary -> {res_path}")
    if sym:
        print(f"  symmetry check: mean(y0+y1) = {sym['mean_y0_plus_y1']:.3f} over {sym['n_pairs']} pairs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
