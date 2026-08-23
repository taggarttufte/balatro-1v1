"""
evaluator.py — the ONE net server every worker's leaves go through.

It runs as a daemon thread in the **main** process, not as a process of its own.  Two
reasons, both measured or structural:

* **The weights are already there.** The trainer owns the net; a thread in the same
  process can mirror it with one ``load_state_dict`` per generation (2.4M params, ~10 MB,
  ~8 ms) instead of shipping it over a pipe.  The brief's "broadcast the weights to the
  evaluator after each training step" is therefore literally one call
  (:meth:`BatchEvaluator.sync_weights`), and it cannot go stale unnoticed —
  ``tests/test_parallel.py::test_sync_weights_changes_what_workers_see`` pins it.
* **The main process has nothing else to do while a generation plays.** Play and train are
  strictly sequential in ``MLBTrainer.run_generation``, so the "spare" core the evaluator
  thread uses is spare by construction.  Torch releases the GIL for the forward pass and
  for the host<->device copies, and the numpy stacking around it is the only Python that
  competes with the main thread's ``Queue.get``.

Moving it to a separate process later is a small change: everything it touches
(:class:`~.channel.EvaluatorChannel`, the arenas, the queue) already crosses process
boundaries.  What would have to be added is the weight broadcast.

Per-policy batching
-------------------
A generation's population is the live net plus up to ``p_history`` past-self checkpoints,
and they are different nets, so leaves are grouped by ``policy_id`` and each group gets its
own forward pass.  The live net is the group that matters: it holds every sample-producing
seat (``m_current``, 8 of 16 in the real run), so it is 8-of-12 net-driven leaves per round
and the one that fills a batch.  The past-self seats submit one leaf each and cost one
small forward pass each; anchors are scripted and never appear here at all.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Optional

import numpy as np
import torch

from .forward import forward_leaves


class EvaluatorStats:
    """Where the evaluator's wall clock went.  Read by the benchmark and logged per
    generation; nothing in the serving loop reads it back."""

    __slots__ = ("batches", "forwards", "leaves", "forward_seconds", "read_seconds",
                 "idle_seconds", "batch_sizes", "started")

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.batches = 0
        self.forwards = 0
        self.leaves = 0
        self.forward_seconds = 0.0
        self.read_seconds = 0.0
        self.idle_seconds = 0.0
        self.batch_sizes = []
        self.started = time.perf_counter()

    @property
    def mean_batch(self) -> float:
        return float(np.mean(self.batch_sizes)) if self.batch_sizes else 0.0

    def as_dict(self) -> dict:
        return {"eval_batches": self.batches, "eval_forwards": self.forwards,
                "eval_leaves": self.leaves,
                "eval_mean_batch": round(self.mean_batch, 2),
                "eval_forward_s": round(self.forward_seconds, 3),
                "eval_read_s": round(self.read_seconds, 3),
                "eval_idle_s": round(self.idle_seconds, 3)}


class BatchEvaluator:
    """Serves leaf evaluations for every worker, batching across them.

    ``models`` maps ``policy_id -> torch.nn.Module`` already on ``device`` and in
    ``eval()`` mode.  ``policy_id`` 0 is the live net by convention (``LIVE_POLICY_ID``).
    """

    LIVE_POLICY_ID = 0

    def __init__(self, channel, layout, *, is_set: bool, device: str = "cpu",
                 max_action_rows: int = 0, poll_seconds: float = 0.005,
                 max_leaves_per_batch: int = 4096):
        self.channel = channel
        self.layout = layout
        self.is_set = bool(is_set)
        self.device = torch.device(device)
        self.max_action_rows = int(max_action_rows)
        self.poll_seconds = float(poll_seconds)
        self.max_leaves_per_batch = int(max_leaves_per_batch)
        self.models: dict = {}
        self.stats = EvaluatorStats()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.error: Optional[BaseException] = None

    # ── models ───────────────────────────────────────────────────────────────

    def set_model(self, policy_id: int, module) -> None:
        with self._lock:
            self.models[int(policy_id)] = module.to(self.device).eval()

    def sync_weights(self, net, policy_id: int = LIVE_POLICY_ID) -> None:
        """Broadcast the trainer's current weights to the evaluator's mirror of the live
        net.  Called once per generation, AFTER the training step that produced them and
        BEFORE any worker plays with them; a no-op mirror (same module object) is still
        given the call so the contract does not depend on the device configuration."""
        with self._lock:
            mirror = self.models.get(int(policy_id))
            if mirror is None:                       # pragma: no cover - set_model first
                raise KeyError(f"no model registered for policy {policy_id}")
            if mirror is net:
                return
            mirror.load_state_dict(net.state_dict())
            mirror.eval()

    # ── the serving loop ─────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread is not None:                 # pragma: no cover - defensive
            raise RuntimeError("evaluator already started")
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="mp-evaluator", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        t, self._thread = self._thread, None
        if t is not None:
            t.join(timeout=timeout)

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                self.serve_once()
        except BaseException as exc:                  # noqa: BLE001 - reported to the pool
            self.error = exc

    def serve_once(self) -> int:
        """One round: collect, forward per policy, reply.  Returns the leaves served."""
        t0 = time.perf_counter()
        subs = self.channel.collect(timeout=self.poll_seconds,
                                    max_leaves=self.max_leaves_per_batch)
        if not subs:
            self.stats.idle_seconds += time.perf_counter() - t0
            return 0
        self.stats.batches += 1

        by_policy: dict = defaultdict(list)
        for sub in subs:
            by_policy[sub.policy_id].append(sub)

        served = 0
        for policy_id, group in by_policy.items():
            t_read = time.perf_counter()
            obs_all: list = []
            acts_all: list = []
            spans: list = []
            for sub in group:
                obs, acts = self.channel.read(sub, self.layout)
                spans.append((sub, len(obs_all), len(obs)))
                obs_all.extend(obs)
                acts_all.extend(acts)
            self.stats.read_seconds += time.perf_counter() - t_read

            with self._lock:
                model = self.models.get(policy_id)
            if model is None:                        # pragma: no cover - protocol error
                raise KeyError(f"worker asked for policy {policy_id}, which is not loaded")

            t_fwd = time.perf_counter()
            probs, values = forward_leaves(model, obs_all, acts_all, is_set=self.is_set,
                                           device=self.device,
                                           max_action_rows=self.max_action_rows)
            self.stats.forward_seconds += time.perf_counter() - t_fwd
            self.stats.forwards += 1
            self.stats.leaves += len(obs_all)
            self.stats.batch_sizes.append(len(obs_all))
            served += len(obs_all)

            for sub, at, n in spans:
                self.channel.write(sub, probs[at:at + n], values[at:at + n])
                self.channel.signal(sub)
        return served

    # ── teardown ─────────────────────────────────────────────────────────────

    def release_workers(self, worker_ids) -> None:
        """Unblock any worker still waiting on a reply (used when tearing down after a
        crash, so nobody sits in ``conn.poll`` until its timeout)."""
        for w in worker_ids:
            self.channel.shutdown_signal(w)


__all__ = ["BatchEvaluator", "EvaluatorStats"]
