"""
workers.py — a resumable multiprocessing pool over PURE functions (Phase 5 rev 2, W5).

Built for the label generator: thousands of independent jobs (one seed each) that need no
network, no shared state and no lockstep — just CPU.  Nothing from Phase 5 rev 1's
``mp/agent/parallel`` (shared-memory leaf transport to a batched evaluator) is needed here,
so that package is left untouched (see TRAINV_NOTES.md).

    summary = run_pool(fn, jobs, n_workers=16, on_result=store, state_path=run/"pool.state",
                       pause_file=run/"PAUSE", checkpoint_every=25)

* ``fn(payload) -> result`` must be a module-level, picklable function (Windows spawns
  workers; they re-import the module).  Every worker runs ``import _bootstrap`` first (the
  fork-guarded engine import) via the initializer, with ``OMP/MKL_NUM_THREADS=1``.
* ``jobs`` is an iterable of ``(job_id, payload)``; ids are strings (or ints), unique.
* ``on_result(job_id, result)`` runs in the MAIN process, in completion order.  It should
  persist the result itself; after it returns the id is appended to ``state_path`` (one id
  per line, append-only) — a restart with the same ``state_path`` skips those ids.  The
  window between ``on_result`` and the append is the only place a crash can cause a job to
  be redone (so make ``on_result`` idempotent per id, or accept a rare duplicate).
* ``pause_file`` is checked before every submission: once it exists no new job is
  submitted, the in-flight ones finish (they are short), ``on_result`` sees them, and
  ``run_pool`` returns with ``summary.paused = True``.  Delete the file and call again to
  resume.  Ctrl+C takes the same path.
* ``checkpoint_every``: after that many completed jobs ``on_checkpoint(summary)`` is called
  (the label generator flushes a shard there).
* ``max_in_flight`` bounds the queue so a pause is honoured within ~one job per worker.
* A job that raises is logged, counted in ``failed`` and NOT marked done (``retry_failed``
  decides whether a later run retries it); the pool keeps going.

``python mp/ev/workers.py --bench --workers 4`` runs a few seconds of dummy jobs and prints
jobs/min and per-job latency.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
MP_ROOT = os.path.dirname(HERE)

__all__ = ["run_pool", "PoolSummary", "load_done_ids", "mark_done", "worker_init", "bench_job"]


# ── persistence of completed ids ─────────────────────────────────────────────────

def load_done_ids(state_path) -> set:
    """Ids recorded by ``mark_done`` (one per line; blank lines ignored)."""
    p = Path(state_path) if state_path is not None else None
    if p is None or not p.exists():
        return set()
    with open(p, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def mark_done(state_path, job_id) -> None:
    if state_path is None:
        return
    p = Path(state_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(f"{job_id}\n")
        f.flush()


# ── worker side ───────────────────────────────────────────────────────────────────

def worker_init(extra_paths: Optional[list] = None, threads: int = 1) -> None:
    """Runs once per worker process: single-threaded BLAS, sys.path, the fork guard."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(threads)
    for p in [HERE, MP_ROOT] + list(extra_paths or []):
        if p not in sys.path:
            sys.path.insert(0, p)
    import _bootstrap  # noqa: F401  (fork-guarded engine import; raises loudly if wrong)
    try:
        import torch  # noqa: F401
        torch.set_num_threads(threads)
    except Exception:
        pass


def _run_job(fn, job_id, payload):
    """The function actually submitted: catches everything so one bad job cannot kill the
    pool, and returns ``(job_id, ok, result_or_traceback, seconds)``."""
    t0 = time.perf_counter()
    try:
        out = fn(payload)
        return job_id, True, out, time.perf_counter() - t0
    except BaseException as e:                       # noqa: BLE001 — the worker must survive
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        return job_id, False, tb, time.perf_counter() - t0


def bench_job(payload) -> dict:
    """Dummy job for the throughput benchmark: a few ms of engine work (construct a match,
    take ``steps`` scripted steps) so the number means something about real workers."""
    import _bootstrap
    from _bootstrap import MLBMatch
    m = MLBMatch(seed=payload.get("seed", "7I4M53DL"))
    n = int(payload.get("steps", 20))
    k = 0
    while not m.done and k < n:
        p = m.current_player()
        acts = m.legal_actions(p)
        m.step(p, acts[0])
        k += 1
    return {"steps": k, "pid": os.getpid()}


# ── the pool ──────────────────────────────────────────────────────────────────────

@dataclass
class PoolSummary:
    submitted: int = 0
    done: int = 0
    failed: int = 0
    skipped: int = 0
    elapsed_s: float = 0.0
    job_seconds: float = 0.0            # sum of per-job wall time inside the workers
    paused: bool = False
    interrupted: bool = False
    exhausted: bool = False             # every job was submitted and finished
    failed_ids: list = field(default_factory=list)
    n_workers: int = 0

    @property
    def jobs_per_min(self) -> float:
        return 60.0 * self.done / self.elapsed_s if self.elapsed_s > 0 else 0.0

    @property
    def mean_job_s(self) -> float:
        return self.job_seconds / self.done if self.done else 0.0

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["jobs_per_min"] = self.jobs_per_min
        d["mean_job_s"] = self.mean_job_s
        return d


def run_pool(fn: Callable, jobs: Iterable, *, n_workers: int = 16,
             on_result: Optional[Callable] = None, pause_file=None, checkpoint_every: int = 0,
             on_checkpoint: Optional[Callable] = None, state_path=None,
             retry_failed: bool = True, max_in_flight: Optional[int] = None,
             log: Optional[Callable] = print, extra_paths: Optional[list] = None,
             threads_per_worker: int = 1, poll_s: float = 0.5,
             max_jobs: Optional[int] = None, deadline_s: Optional[float] = None,
             context: str = "spawn") -> PoolSummary:
    """See the module docstring.  ``n_workers <= 0`` runs every job inline (no processes —
    handy for tests and debugging)."""
    done_ids = load_done_ids(state_path)
    summary = PoolSummary(n_workers=max(int(n_workers), 0))
    pause_path = Path(pause_file) if pause_file is not None else None
    t_start = time.perf_counter()
    max_in_flight = max_in_flight or max(2 * max(n_workers, 1), 2)

    def paused() -> bool:
        return pause_path is not None and pause_path.exists()

    def over_deadline() -> bool:
        return deadline_s is not None and (time.perf_counter() - t_start) >= deadline_s

    def handle(job_id, ok, result, secs) -> None:
        summary.job_seconds += secs
        if ok:
            if on_result is not None:
                on_result(job_id, result)
            mark_done(state_path, job_id)
            done_ids.add(str(job_id))
            summary.done += 1
            if checkpoint_every and on_checkpoint is not None and summary.done % checkpoint_every == 0:
                summary.elapsed_s = time.perf_counter() - t_start
                on_checkpoint(summary)
        else:
            summary.failed += 1
            summary.failed_ids.append(str(job_id))
            if log is not None:
                log(f"[pool] job {job_id} FAILED after {secs:.1f}s:\n{result}")

    cut = {"max_jobs": False}

    def pending_jobs():
        n = 0
        for job_id, payload in jobs:
            if max_jobs is not None and n >= max_jobs:
                cut["max_jobs"] = True
                return
            if str(job_id) in done_ids:
                summary.skipped += 1
                continue
            n += 1
            yield job_id, payload

    it = pending_jobs()

    # ── inline mode ──
    if n_workers <= 0:
        try:
            for job_id, payload in it:
                if paused():
                    summary.paused = True
                    break
                if over_deadline():
                    break
                summary.submitted += 1
                handle(*_run_job(fn, job_id, payload))
            else:
                summary.exhausted = not cut["max_jobs"]
        except KeyboardInterrupt:
            summary.interrupted = True
        summary.elapsed_s = time.perf_counter() - t_start
        return summary

    # ── process pool ──
    import queue as _queue
    ctx = mp.get_context(context)
    pool = ctx.Pool(processes=n_workers, initializer=worker_init,
                    initargs=(list(extra_paths or []), threads_per_worker))
    results: "_queue.Queue" = _queue.Queue()
    in_flight = 0
    try:
        source_empty = False
        while True:
            # submit until the window is full
            while not source_empty and in_flight < max_in_flight:
                if paused():
                    summary.paused = True
                    source_empty = True
                    break
                if over_deadline():
                    source_empty = True
                    break
                try:
                    job_id, payload = next(it)
                except StopIteration:
                    source_empty = True
                    summary.exhausted = not cut["max_jobs"]
                    break
                summary.submitted += 1
                in_flight += 1
                pool.apply_async(_run_job, (fn, job_id, payload), callback=results.put,
                                 error_callback=lambda e, _j=job_id: results.put((_j, False, repr(e), 0.0)))
            if in_flight == 0:
                break
            # block until something finishes (the callback runs in the pool's result thread)
            try:
                item = results.get(timeout=poll_s)
            except _queue.Empty:
                continue
            in_flight -= 1
            handle(*item)
    except KeyboardInterrupt:
        summary.interrupted = True
        if log is not None:
            log("[pool] interrupted: terminating workers (in-flight jobs are lost, not marked done)")
        pool.terminate()
        pool.join()
        summary.elapsed_s = time.perf_counter() - t_start
        return summary
    finally:
        try:
            pool.close()
            pool.join()
        except Exception:
            pool.terminate()
    summary.elapsed_s = time.perf_counter() - t_start
    if summary.exhausted and summary.paused:
        summary.exhausted = False
    return summary


# ── benchmark CLI ─────────────────────────────────────────────────────────────────

def _bench(n_workers: int, seconds: float, steps: int) -> PoolSummary:
    jobs = ((f"bench-{i}", {"seed": "7I4M53DL", "steps": steps}) for i in range(10_000_000))
    t = time.perf_counter()
    s = run_pool(bench_job, jobs, n_workers=n_workers, deadline_s=seconds, log=print)
    s.elapsed_s = time.perf_counter() - t
    print(f"workers={n_workers} steps/job={steps}: {s.done} jobs in {s.elapsed_s:.1f}s = "
          f"{s.jobs_per_min:.0f} jobs/min, mean job {1000 * s.mean_job_s:.1f} ms in-worker "
          f"(pool overhead {1000 * (s.elapsed_s / max(s.done, 1) * n_workers - s.mean_job_s):.1f} ms/job)")
    return s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--steps", type=int, default=20, help="engine steps per dummy job")
    args = ap.parse_args(argv)
    if args.bench:
        _bench(args.workers, args.seconds, args.steps)
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
