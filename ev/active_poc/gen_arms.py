"""gen_arms.py — label the UNION of the three arms' selected states with the standard pipeline.

    python mp/ev/active_poc/gen_arms.py --workers 8 --minutes 95

One job = one seed carrying all of that seed's selected states (``active_poc.jobs.arm_job``):
``labels.sample_states`` re-derives the snapshots, and each selected state is labelled by
``labels.label_both`` at ``n_rollouts=8`` — the standard label definition, unchanged.

Why one pass over the union rather than three passes: a state chosen by two arms is labelled
ONCE and shared, which is both cheaper and a tighter control (an overlapping state cannot
differ between arms through label noise).  Seeds are shuffled deterministically so that if
the deadline cuts the run short, what is missing is an unbiased subsample of EVERY arm rather
than a suffix of one; the final stage then trims all arms to a common size.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
EV_ROOT = HERE.parent
MP_ROOT = EV_ROOT.parent
for _p in (str(EV_ROOT), str(MP_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _bootstrap  # noqa: E402,F401
import dataset as DS  # noqa: E402
import workers as W  # noqa: E402
from dataset import LabelRow  # noqa: E402

from active_poc import corpus as C  # noqa: E402
from active_poc.jobs import CORPUS_CONFIG, arm_job  # noqa: E402  (module-level = picklable)

PAUSE_FILE = "PAUSE"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="mp/ev/runs/active_poc")
    ap.add_argument("--run-dir", default="mp/ev/runs/active_poc/arms")
    ap.add_argument("--pool-done", default="mp/ev/runs/active_poc/pool/done.ids",
                    help="the pool's job ids = the RAW seed strings (see the canonicalisation note)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--minutes", type=float, default=None)
    ap.add_argument("--flush-jobs", type=int, default=16)
    ap.add_argument("--job-rng", type=int, default=777)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers > 8:
        raise SystemExit("ops cap: at most 8 label workers")
    out = Path(args.out)
    spec = json.loads((out / "arms.json").read_text(encoding="utf-8"))
    run_dir = Path(args.run_dir)
    shards_dir = run_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    done_path = run_dir / "done.ids"
    pause_path = run_dir / PAUSE_FILE
    if pause_path.exists():
        pause_path.unlink()

    by_seed = defaultdict(list)
    for states in spec["arms"].values():
        for s, t in states:
            by_seed[str(s)].append(int(t))
    by_seed = {s: sorted(set(ts)) for s, ts in by_seed.items()}
    n_states = sum(len(v) for v in by_seed.values())

    # The snapshot reservoir is seeded from the seed STRING the job was given, but a shard
    # records the engine's CANONICAL seed ('0' -> 'O').  The pool was driven with the raw
    # strings, so arm_job must be given those same raw strings or it re-derives a different
    # snapshot set (and every state of a seed containing a '0' would fail the fingerprint
    # check).  The pool's done.ids are exactly those raw job ids.
    raw_by_canon = {C.canonical_seed(r): r for r in W.load_done_ids(args.pool_done)}
    unmapped = [s for s in by_seed if s not in raw_by_canon]
    if unmapped:
        print(f"  WARNING {len(unmapped)} selected seeds absent from {args.pool_done} "
              f"(using the canonical form): {unmapped[:4]}", flush=True)
    n_remapped = sum(1 for s in by_seed if raw_by_canon.get(s, s) != s)
    print(f"  raw-seed map: {len(raw_by_canon)} pool seeds, {n_remapped} of the "
          f"{len(by_seed)} selected seeds needed de-canonicalising", flush=True)

    order = sorted(by_seed)
    random.Random(args.job_rng).shuffle(order)      # unbiased partial completion
    done = W.load_done_ids(done_path)

    lf = open(run_dir / "gen.jsonl", "a", encoding="utf-8")

    def emit(rec: dict) -> None:
        lf.write(json.dumps(rec, default=str) + "\n")
        lf.flush()

    state = {"buffer": [], "ids": [], "n_shards": len(sorted(shards_dir.glob("shard_*.npz"))),
             "rows": 0, "jobs": 0, "states": 0, "t0": time.perf_counter(),
             "missing": [], "mismatched": [],
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
        rec = {"kind": "flush", "why": why, "shard": path.name, "rows": n,
               "rows_total": state["rows"], "states_total": state["states"],
               "jobs_total": state["jobs"], "elapsed_s": el,
               "labels_per_min": 60 * state["rows"] / el if el else 0.0,
               "s_per_rollout": tm["label_s"] / tm["n_rollouts"] if tm["n_rollouts"] else 0.0}
        emit(rec)
        eta = (n_states - state["states"]) / max(state["states"] / el, 1e-9) / 60 if state["states"] else 0
        print(f"  [flush:{why}] {path.name} +{n} rows ({state['rows']} rows / {state['states']} of "
              f"{n_states} states, {rec['labels_per_min']:.0f} labels/min, "
              f"{rec['s_per_rollout']:.2f} s/rollout, eta {eta:.0f} min)", flush=True)
        state["buffer"] = []
        state["ids"] = []

    def on_result(job_id, result) -> None:
        for r in result["rows"]:
            state["buffer"].append(LabelRow(r["obs"], float(r["y"]), r["meta"]))
        state["ids"].append(job_id)
        state["jobs"] += 1
        state["states"] += result["timing"]["n_states"]
        state["missing"].extend([[job_id, s] for s in result.get("missing_steps", [])])
        state["mismatched"].extend([[job_id, s] for s in result.get("mismatched_steps", [])])
        for k, v in result["timing"].items():
            if k in state["timing"]:
                state["timing"][k] += v

    def jobs():
        for s in order:
            if s in done:
                continue
            steps = by_seed[s]
            fps = {str(t): spec["fps"].get(f"{s}|{t}") for t in steps}
            yield s, {"seed": raw_by_canon.get(s, s), "steps": steps,
                      "fps": {k: v for k, v in fps.items() if v}}

    emit({"kind": "config", "timestamp": datetime.now().isoformat(timespec="seconds"),
          "args": vars(args), "corpus_config": CORPUS_CONFIG, "n_seeds": len(order),
          "n_states": n_states, "n_done_before": len(done)})
    print(f"=== arm labelling: {n_states} union states over {len(order)} seeds "
          f"({len(done)} seeds done), {args.workers} workers, n_rollouts="
          f"{CORPUS_CONFIG['n_rollouts']} ===", flush=True)
    print(f"  run dir: {run_dir}   pause: touch {pause_path}", flush=True)

    summary = W.run_pool(arm_job, jobs(), n_workers=args.workers, on_result=on_result,
                         pause_file=pause_path, checkpoint_every=args.flush_jobs,
                         on_checkpoint=lambda s: flush("jobs"), state_path=None,
                         deadline_s=(args.minutes * 60 if args.minutes else None))
    flush("exit")
    ds = DS.LabelDataset.load(shards_dir)
    res = {"pool": summary.as_dict(), "n_rows": len(ds), "n_states": len(ds) // 2,
           "n_seeds": len(ds.seeds()), "union_states_requested": n_states,
           "missing": state["missing"], "mismatched": state["mismatched"],
           "timing": state["timing"]}
    emit({"kind": "summary", **res})
    lf.close()
    print(f"=== arms done: {summary.done} jobs ({summary.failed} failed), {len(ds)} rows / "
          f"{len(ds) // 2} of {n_states} states, "
          f"{60 * len(ds) / max(summary.elapsed_s, 1e-9):.0f} labels/min; "
          f"missing {len(state['missing'])}, mismatched {len(state['mismatched'])} ===", flush=True)
    if state["mismatched"]:
        print(f"  WARNING mismatched (reconstruction drift): {state['mismatched'][:5]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
