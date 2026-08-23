"""test_workers.py — the resumable pool (W5).  Process tests use <= 2 workers and run in
seconds (Tagg is using the machine)."""
from __future__ import annotations

import json
import os
import time

import pytest

import workers as W

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _jobs(n, steps=5):
    return [(f"j{i}", {"seed": "7I4M53DL", "steps": steps}) for i in range(n)]


# ── inline mode (no processes) ──

def test_inline_runs_everything_and_records_state(tmp_path):
    state = tmp_path / "pool.state"
    got = {}
    s = W.run_pool(W.bench_job, _jobs(6), n_workers=0, on_result=lambda j, r: got.__setitem__(j, r),
                   state_path=state, log=None)
    assert s.done == 6 and s.failed == 0 and s.exhausted and not s.paused
    assert set(got) == {f"j{i}" for i in range(6)}
    assert all(r["steps"] == 5 for r in got.values())
    assert W.load_done_ids(state) == set(got)
    # a second run with the same state skips everything
    s2 = W.run_pool(W.bench_job, _jobs(6), n_workers=0, state_path=state, log=None)
    assert s2.done == 0 and s2.skipped == 6 and s2.exhausted


def _boom(payload):
    if payload.get("boom"):
        raise RuntimeError("boom")
    return payload["v"]


def test_inline_failures_are_isolated_and_not_marked_done(tmp_path):
    state = tmp_path / "pool.state"
    jobs = [("a", {"v": 1}), ("b", {"boom": True}), ("c", {"v": 3})]
    logs = []
    s = W.run_pool(_boom, jobs, n_workers=0, state_path=state, log=logs.append)
    assert s.done == 2 and s.failed == 1 and s.failed_ids == ["b"]
    assert W.load_done_ids(state) == {"a", "c"}
    assert any("boom" in m for m in logs)
    # the failed one is retried on the next run
    s2 = W.run_pool(_boom, jobs, n_workers=0, state_path=state, log=None)
    assert s2.skipped == 2 and s2.failed == 1


def test_inline_pause_file_stops_between_jobs(tmp_path):
    pause = tmp_path / "PAUSE"
    state = tmp_path / "pool.state"
    count = {"n": 0}

    def on_result(j, r):
        count["n"] += 1
        if count["n"] == 3:
            pause.write_text("")

    s = W.run_pool(W.bench_job, _jobs(10), n_workers=0, on_result=on_result, state_path=state,
                   pause_file=pause, log=None)
    assert s.paused and s.done == 3 and not s.exhausted
    pause.unlink()
    s2 = W.run_pool(W.bench_job, _jobs(10), n_workers=0, state_path=state, pause_file=pause, log=None)
    assert s2.done == 7 and s2.skipped == 3 and s2.exhausted


def test_checkpoint_callback_and_max_jobs(tmp_path):
    ticks = []
    s = W.run_pool(W.bench_job, _jobs(9), n_workers=0, checkpoint_every=4,
                   on_checkpoint=lambda sm: ticks.append(sm.done), max_jobs=8, log=None)
    assert s.done == 8 and ticks == [4, 8]


# ── real processes (spawn), 2 workers ──

def test_pool_processes_run_jobs_and_resume(tmp_path):
    state = tmp_path / "pool.state"
    pause = tmp_path / "PAUSE"
    got = {}
    t = time.perf_counter()
    s = W.run_pool(W.bench_job, _jobs(12, steps=10), n_workers=2, on_result=lambda j, r: got.__setitem__(j, r),
                   state_path=state, pause_file=pause, log=None)
    dt = time.perf_counter() - t
    assert s.done == 12 and s.failed == 0 and s.exhausted
    assert len({r["pid"] for r in got.values()}) >= 1
    assert all(r["pid"] != os.getpid() for r in got.values())     # really ran elsewhere
    assert W.load_done_ids(state) == set(got)
    assert dt < 60
    s2 = W.run_pool(W.bench_job, _jobs(12, steps=10), n_workers=2, state_path=state, log=None)
    assert s2.done == 0 and s2.skipped == 12


def test_pool_processes_failure_isolated():
    jobs = [("a", {"v": 1}), ("b", {"boom": True}), ("c", {"v": 3})]
    logs = []
    s = W.run_pool(_boom, jobs, n_workers=2, log=logs.append)
    assert s.done == 2 and s.failed == 1 and s.failed_ids == ["b"]
    assert any("RuntimeError: boom" in m for m in logs)


def test_pool_processes_deadline_and_summary_fields():
    jobs = ((f"j{i}", {"seed": "7I4M53DL", "steps": 10}) for i in range(100_000))
    s = W.run_pool(W.bench_job, jobs, n_workers=2, deadline_s=2.0, log=None)
    assert s.done > 0 and not s.exhausted
    d = s.as_dict()
    assert d["jobs_per_min"] > 0 and d["mean_job_s"] > 0
    json.dumps(d)
