"""
channel.py — the worker <-> evaluator transport.

One request arena and one reply arena per worker, both ``multiprocessing.shared_memory``
blocks; a single ``Queue`` from all workers to the evaluator carrying nothing but offsets;
one ``Connection`` per worker carrying nothing but a round id.

    worker  w                                  evaluator (main process, one thread)
    --------------------------------------     --------------------------------------
    pack B leaves into REQ[w]                   drain leaf_q  (blocking with a timeout)
    leaf_q.put((w, round, policy, metas))  -->  read the leaves as numpy VIEWS into REQ[*]
    conn[w].recv_bytes()  (blocks)              one forward pass per policy id
                                           <--  write probs+values into REP[w]
    read REP[w], build the priors                conn[w].send_bytes(round)

Why a queue for the offsets and shared memory for the payload: the payload is ~127 KB per
leaf and the offsets are 16 bytes per leaf (layout.py has the measurements).  Pickling the
offsets costs nothing and gives a ready-made multi-producer / single-consumer channel with
a blocking ``get(timeout=...)``; pickling the payload would be ~370 MB/s of copying at 16
workers.

Batching policy (the brief's "never block the evaluator's batch on one slow tree"): the
evaluator blocks only until the FIRST submission of a round arrives, then drains whatever
else is already queued and forwards that.  No worker is ever waited for by name, so a
worker stuck in a long ``game.step()`` cannot hold anybody up; and because a forward pass
takes time, the queue refills while it runs, which makes the batch size self-balancing
under load.  ``max_wait_ms`` (default 0) additionally lets the evaluator keep draining for
a fixed budget after the first arrival, to trade a little latency for a bigger batch; the
benchmark sweeps it.

Windows notes
-------------
* ``spawn`` only: every object here is created in the parent and handed to the child as a
  ``Process`` argument (``Queue`` and ``Connection`` are picklable that way; they are NOT
  picklable through a queue).
* A ``SharedMemory`` block lives as long as at least one handle is open, and the parent
  holds one for the whole run, so a worker that dies cannot take the arena with it.
* ``SharedMemory.unlink()`` is a no-op on Windows; ``close()`` on every handle is what
  actually releases the mapping.  ``ArenaSet.close()`` does both so the code is portable.
"""
from __future__ import annotations

import struct
import time
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Optional, Sequence

import numpy as np

#: Request arena per worker.  A 436-action set-encoder leaf is ~127 KB, so 16 MB holds 128
#: leaves in flight -- 8x the largest sensible per-worker agent count.
DEFAULT_REQUEST_BYTES = 16 << 20
#: Reply arena per worker: (1 value + n_actions priors) float32 per leaf, ~1.7 KB each.
DEFAULT_REPLY_BYTES = 4 << 20

_SHUTDOWN_ROUND = -1
_ROUND_FMT = "<q"


@dataclass(frozen=True)
class ArenaHandle:
    """What a worker needs to attach to its own two arenas (picklable, no handles)."""
    worker_id: int
    request_name: str
    reply_name: str
    request_bytes: int
    reply_bytes: int


class ArenaSet:
    """The parent's handles on every worker's arenas.  Create once per run."""

    def __init__(self, n_workers: int, request_bytes: int = DEFAULT_REQUEST_BYTES,
                 reply_bytes: int = DEFAULT_REPLY_BYTES):
        self.n_workers = int(n_workers)
        self.request_bytes = int(request_bytes)
        self.reply_bytes = int(reply_bytes)
        self._req = [shared_memory.SharedMemory(create=True, size=self.request_bytes)
                     for _ in range(self.n_workers)]
        self._rep = [shared_memory.SharedMemory(create=True, size=self.reply_bytes)
                     for _ in range(self.n_workers)]
        self.req_views = [np.ndarray((self.request_bytes,), dtype=np.uint8, buffer=s.buf)
                          for s in self._req]
        self.rep_views = [np.ndarray((self.reply_bytes // 4,), dtype=np.float32,
                                     buffer=s.buf) for s in self._rep]

    def handle(self, worker_id: int) -> ArenaHandle:
        return ArenaHandle(worker_id=worker_id, request_name=self._req[worker_id].name,
                           reply_name=self._rep[worker_id].name,
                           request_bytes=self.request_bytes, reply_bytes=self.reply_bytes)

    def close(self) -> None:
        self.req_views = []
        self.rep_views = []
        for s in self._req + self._rep:
            try:
                s.close()
            except Exception:                        # pragma: no cover - teardown
                pass
            try:
                s.unlink()                           # no-op on Windows
            except Exception:                        # pragma: no cover - teardown
                pass
        self._req, self._rep = [], []


class WorkerArena:
    """The child's side: attach by name, keep the two numpy views."""

    def __init__(self, handle: ArenaHandle):
        self.worker_id = handle.worker_id
        self._req = shared_memory.SharedMemory(name=handle.request_name)
        self._rep = shared_memory.SharedMemory(name=handle.reply_name)
        self.request = np.ndarray((handle.request_bytes,), dtype=np.uint8, buffer=self._req.buf)
        self.reply = np.ndarray((handle.reply_bytes // 4,), dtype=np.float32,
                                buffer=self._rep.buf)
        self.request_bytes = handle.request_bytes
        self.reply_floats = handle.reply_bytes // 4

    def close(self) -> None:
        self.request = None
        self.reply = None
        for s in (self._req, self._rep):
            try:
                s.close()
            except Exception:                        # pragma: no cover - teardown
                pass


# ── worker side ────────────────────────────────────────────────────────────────

class WorkerChannel:
    """Submit a homogeneous batch of leaves and block until the evaluator answers.

    "Homogeneous" = one ``policy_id`` per submission, which is free: a worker runs one
    ``BatchedSearch`` per distinct policy in its slice of the population (the live net and
    each past-self checkpoint are different nets), so every ``evaluate_many`` call it makes
    already belongs to exactly one of them.
    """

    def __init__(self, arena: WorkerArena, leaf_q, reply_conn, poll_seconds: float = 5.0):
        self.arena = arena
        self.leaf_q = leaf_q
        self.conn = reply_conn
        self.poll_seconds = float(poll_seconds)
        self.round = 0
        self.submissions = 0
        self.leaves = 0
        self.wait_seconds = 0.0
        self.pack_seconds = 0.0

    def submit_and_wait(self, policy_id: int, metas: Sequence[tuple]) -> int:
        """``metas`` is ``[(n_actions, offset), ...]`` already written into the arena."""
        self.round += 1
        self.submissions += 1
        self.leaves += len(metas)
        self.leaf_q.put((self.arena.worker_id, self.round, int(policy_id), tuple(metas)))
        t0 = time.perf_counter()
        while True:
            if self.conn.poll(self.poll_seconds):
                got = struct.unpack(_ROUND_FMT, self.conn.recv_bytes(8))[0]
                if got == _SHUTDOWN_ROUND:
                    raise EvaluatorGone("evaluator shut down while a batch was in flight")
                if got != self.round:                # pragma: no cover - protocol error
                    raise RuntimeError(
                        f"worker {self.arena.worker_id} expected round {self.round}, "
                        f"got {got}: the reply channel is out of step")
                self.wait_seconds += time.perf_counter() - t0
                return got
            raise EvaluatorGone(
                f"no reply from the evaluator for {self.poll_seconds:.0f}s "
                f"(worker {self.arena.worker_id}, round {self.round})")


class EvaluatorGone(RuntimeError):
    """The evaluator stopped answering.  A worker turns this into a clean exit so the
    tournament can mark its agents crashed rather than hanging."""


# ── evaluator side ─────────────────────────────────────────────────────────────

class Submission:
    """One worker's request as the evaluator sees it."""

    __slots__ = ("worker_id", "round", "policy_id", "metas")

    def __init__(self, worker_id: int, round_id: int, policy_id: int, metas):
        self.worker_id = worker_id
        self.round = round_id
        self.policy_id = policy_id
        self.metas = metas

    @property
    def n_leaves(self) -> int:
        return len(self.metas)


class EvaluatorChannel:
    """The parent's side: collect submissions, hand back replies."""

    def __init__(self, arenas: ArenaSet, leaf_q, reply_conns: Sequence,
                 max_wait_ms: float = 0.0):
        self.arenas = arenas
        self.leaf_q = leaf_q
        self.conns = list(reply_conns)
        self.max_wait_ms = float(max_wait_ms)

    def collect(self, timeout: float = 0.005, max_leaves: int = 4096) -> list:
        """Block up to ``timeout`` for the first submission, then drain.

        Returns ``[]`` when nothing arrived, which is how the serving loop notices a
        shutdown request or a dead worker.
        """
        import queue as _queue

        out: list = []
        try:
            first = self.leaf_q.get(timeout=timeout)
        except _queue.Empty:
            return out
        out.append(Submission(*first))
        n = out[0].n_leaves

        deadline = (time.perf_counter() + self.max_wait_ms / 1000.0
                    if self.max_wait_ms > 0 else None)
        while n < max_leaves:
            try:
                item = self.leaf_q.get_nowait()
            except _queue.Empty:
                if deadline is None or time.perf_counter() >= deadline:
                    break
                time.sleep(0)                       # yield; the queue is filled by workers
                continue
            sub = Submission(*item)
            out.append(sub)
            n += sub.n_leaves
        return out

    def read(self, sub: Submission, layout) -> tuple:
        """``(obs_list, acts_list)`` as views into the submitting worker's arena."""
        buf = self.arenas.req_views[sub.worker_id]
        obs_list, acts_list = [], []
        for n_actions, offset in sub.metas:
            obs, acts = layout.unpack(buf, int(offset), int(n_actions))
            obs_list.append(obs)
            acts_list.append(acts)
        return obs_list, acts_list

    def write(self, sub: Submission, probs: Sequence, values) -> None:
        """Values first (one float per leaf), then the priors, in submission order."""
        rep = self.arenas.rep_views[sub.worker_id]
        b = len(sub.metas)
        rep[:b] = values
        at = b
        for p in probs:
            rep[at:at + p.shape[0]] = p
            at += p.shape[0]

    def signal(self, sub: Submission) -> None:
        self.conns[sub.worker_id].send_bytes(struct.pack(_ROUND_FMT, sub.round))

    def shutdown_signal(self, worker_id: int) -> None:
        try:
            self.conns[worker_id].send_bytes(struct.pack(_ROUND_FMT, _SHUTDOWN_ROUND))
        except (BrokenPipeError, OSError):            # pragma: no cover - teardown race
            pass


def reply_offsets(metas: Sequence[tuple]) -> list:
    """Where each leaf's priors start in the reply arena (values occupy ``[0, B)``)."""
    at = len(metas)
    out = []
    for n_actions, _offset in metas:
        out.append(at)
        at += int(n_actions)
    return out


__all__ = [
    "ArenaHandle", "ArenaSet", "WorkerArena", "WorkerChannel", "EvaluatorChannel",
    "Submission", "EvaluatorGone", "reply_offsets",
    "DEFAULT_REQUEST_BYTES", "DEFAULT_REPLY_BYTES",
]
