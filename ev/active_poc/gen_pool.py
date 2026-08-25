"""gen_pool.py — the candidate pool: fresh self-play states, encoded, with a cheap probe label.

    python mp/ev/active_poc/gen_pool.py --run-dir mp/ev/runs/active_poc/pool \
        --seeds 600 --workers 8 --n-probe 2 --minutes 50

One job = one FRESH seed (``active_poc.jobs.pool_job``): the same self-play + stratified
snapshot machinery the 51k corpus used (``labels.sample_states``, corpus config verbatim),
both perspectives encoded, plus an ``n_probe``-rollout label used only as the error proxy's
noisy reference.  Seeds are drawn to avoid BOTH the existing corpus seeds and the standard
evaluation holdout, so nothing here can leak into the holdout.

Rows are written as ordinary label shards (``dataset.save_shard``) with ``y`` = the probe
label and ``n_rollouts`` = ``n_probe``, so the whole pool loads through ``LabelDataset`` and
carries its obs, kinds, antes and ``obs_fp`` with it.  Crash-safe like ``gen_labels.py``: a
seed enters ``done.ids`` only after the shard holding its rows is on disk.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
EV_ROOT = HERE.parent
MP_ROOT = EV_ROOT.parent
for _p in (str(EV_ROOT), str(MP_ROOT), str(EV_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _bootstrap  # noqa: E402,F401
import dataset as DS  # noqa: E402
import workers as W  # noqa: E402
from dataset import LabelRow  # noqa: E402

from active_poc import corpus as C  # noqa: E402
from active_poc.jobs import CORPUS_CONFIG, pool_job  # noqa: E402  (module-level = picklable)

PAUSE_FILE = "PAUSE"


def corpus_seeds() -> list:
    """The exact seed list the 51k campaign used (``default+random:2000``, seed-rng 0)."""
    from gen_labels import parse_seeds
    return parse_seeds("default+random:2000", 0)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", default="mp/ev/runs/active_poc/pool")
    ap.add_argument("--seeds", type=int, default=600, help="number of FRESH seeds")
    ap.add_argument("--seed-rng", type=int, default=20260825)
    ap.add_argument("--n-probe", type=int, default=2, help="rollouts for the cheap probe label")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--minutes", type=float, default=None, help="wall-clock deadline")
    ap.add_argument("--flush-jobs", type=int, default=16)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers > 8:
        raise SystemExit("ops cap: at most 8 label workers")
    run_dir = Path(args.run_dir)
    shards_dir = run_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    done_path = run_dir / "done.ids"
    pause_path = run_dir / PAUSE_FILE
    if pause_path.exists():
        pause_path.unlink()

    existing = corpus_seeds()
    seeds = C.fresh_seeds(args.seeds, rng_seed=args.seed_rng, exclude=existing)
    done = W.load_done_ids(done_path)

    lf = open(run_dir / "gen.jsonl", "a", encoding="utf-8")

    def emit(rec: dict) -> None:
        lf.write(json.dumps(rec, default=str) + "\n")
        lf.flush()

    state = {"buffer": [], "ids": [], "n_shards": len(sorted(shards_dir.glob("shard_*.npz"))),
             "rows": 0, "jobs": 0, "t0": time.perf_counter(),
             "timing": {"selfplay_s": 0.0, "label_s": 0.0, "n_rollouts": 0, "total_s": 0.0}}

    def flush(why: str) -> None:
        if not state["buffer"]:
            return
        path = shards_dir / f"shard_{state['n_shards']:04d}.npz"
        DS.save_shard(path, state["buffer"])
        for jid in state["ids"]:
            W.mark_done(done_path, jid)
            done.add(jid)
        n = len(state["buffer"])
        state["n_shards"] += 1
        state["rows"] += n
        el = time.perf_counter() - state["t0"]
        tm = state["timing"]
        rec = {"kind": "flush", "why": why, "shard": path.name, "rows": n, "rows_total": state["rows"],
               "jobs_total": state["jobs"], "elapsed_s": el,
               "rows_per_min": 60 * state["rows"] / el if el else 0.0,
               "s_per_rollout": tm["label_s"] / tm["n_rollouts"] if tm["n_rollouts"] else 0.0}
        emit(rec)
        print(f"  [flush:{why}] {path.name} +{n} rows ({state['rows']} total, {state['jobs']} jobs, "
              f"{rec['rows_per_min']:.0f} rows/min, {rec['s_per_rollout']:.2f} s/rollout)", flush=True)
        state["buffer"] = []
        state["ids"] = []

    def on_result(job_id, result) -> None:
        for r in result["rows"]:
            state["buffer"].append(LabelRow(r["obs"], float(r["y_probe"]), r["meta"]))
        state["ids"].append(job_id)
        state["jobs"] += 1
        for k, v in result["timing"].items():
            if k in state["timing"]:
                state["timing"][k] += v

    def jobs():
        for s in seeds:
            if s in done:
                continue
            yield s, {"seed": s, "n_probe": args.n_probe}

    emit({"kind": "config", "timestamp": datetime.now().isoformat(timespec="seconds"),
          "args": vars(args), "corpus_config": CORPUS_CONFIG, "n_seeds": len(seeds),
          "n_done_before": len(done)})
    print(f"=== candidate pool: {len(seeds)} fresh seeds ({len(done)} done), {args.workers} workers, "
          f"{CORPUS_CONFIG['n_states']} states x {args.n_probe} probe rollouts ===", flush=True)
    print(f"  run dir: {run_dir}   pause: touch {pause_path}", flush=True)

    summary = W.run_pool(pool_job, jobs(), n_workers=args.workers, on_result=on_result,
                         pause_file=pause_path, checkpoint_every=args.flush_jobs,
                         on_checkpoint=lambda s: flush("jobs"), state_path=None,
                         deadline_s=(args.minutes * 60 if args.minutes else None))
    flush("exit")
    ds = DS.LabelDataset.load(shards_dir)
    out = {"pool": summary.as_dict(), "n_rows": len(ds), "n_states": len(ds) // 2,
           "n_seeds": len(ds.seeds()), "by_kind": C.kind_counts(ds),
           "timing": state["timing"], "n_probe": args.n_probe}
    emit({"kind": "summary", **out})
    lf.close()
    print(f"=== pool done: {summary.done} jobs ({summary.failed} failed), {len(ds)} rows / "
          f"{len(ds) // 2} states in {state['n_shards']} shards, "
          f"{60 * len(ds) / max(summary.elapsed_s, 1e-9):.0f} rows/min ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
