"""
pool.py — the main process's side: N workers, one evaluator, one tournament driver.

    WorkerPool      spawn / command / collect / bury.  Owns the arenas, the leaf queue,
                    the reply pipes and (in "remote" mode) the evaluator thread.
    MPDriver        a ``tournament.parallel.TournamentDriver`` that fans every driver call
                    out to the workers that own the agents in question.
    partition_agents  which agent goes to which worker.

Ordering and determinism
------------------------
Everything cross-agent stays here and stays in the same order ``Tournament.run`` does it
(that is ``ParallelTournament``'s job).  What this module has to guarantee on top is that
the *inputs* to those decisions do not depend on how many workers there are:

* the tournament seeds come from the trainer's generator in the main process;
* the seed STRING is resolved here, once, and handed to every worker, so N workers
  constructing 16 games between them build the same 16 games one process would have
  (``clone_games``/``construct_games`` are pinned equal by ``tests/test_fanout.py``);
* each agent's search rng is ``default_rng(member.seed)`` — a function of the population,
  not of the worker;
* collected samples are re-ordered by ``(seed index, agent index, decision index)`` before
  they reach the buffer, so the buffer sees one stream whatever the partition was.

Worker death
------------
A worker that dies takes its agents out of the tournament, not the run.  Every call checks
liveness while it waits; a dead worker's agents report ``status="crashed"``, which
``ParallelTournament`` handles exactly like a death (they leave the population, the
matrix is built from whoever is left).  The dead worker is NOT restarted mid-generation —
its games and trees are gone and re-creating them would be a different experiment — but
:meth:`WorkerPool.respawn_dead` brings it back for the next generation.
"""
from __future__ import annotations

import multiprocessing as mp
import queue as _queue
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from .channel import ArenaSet, EvaluatorChannel
from .protocol import (
    CRASH_EXIT_CODE, OP_APPLY, OP_CRASH, OP_DRIVE, OP_FANOUT, OP_SHUTDOWN,
    OP_SUMMARIZE, TournamentSetup, WorkerSpec,
)
from .worker import worker_main

AGENT_ROOT = Path(__file__).resolve().parents[1]
ENGINE_ROOT = AGENT_ROOT.parent / "engine"
MP_ROOT = AGENT_ROOT.parent
ROOTS = (str(ENGINE_ROOT), str(AGENT_ROOT), str(MP_ROOT))

#: How long a single pool call may take before it is declared hung.  A drive to the next
#: Nemesis at 40 sims is seconds; 10 minutes is "something is very wrong", not "slow".
DEFAULT_CALL_TIMEOUT = 600.0


@dataclass
class PoolConfig:
    n_workers: int = 4
    #: "cpu" / "cuda" -> one shared evaluator on that device.  "local" -> no evaluator;
    #: every worker loads the generation's weights and runs the net itself (the control
    #: arm the benchmark needs to answer "is the shared evaluator worth it on this box").
    evaluator_device: str = "cpu"
    torch_threads: int = 1
    max_wait_ms: float = 0.0
    request_mb: int = 16
    reply_mb: int = 4
    max_action_rows: int = 0
    #: "local" mode only: where a worker runs its OWN net.  Always the CPU — 16 CUDA
    #: contexts on one card is slower than one, and the whole point of the local arm is
    #: "N independent CPU nets vs one shared batched one".
    worker_device: str = "cpu"
    call_timeout: float = DEFAULT_CALL_TIMEOUT
    worker_poll_seconds: float = 120.0
    evaluator_poll_seconds: float = 0.002

    @property
    def mode(self) -> str:
        return "local" if self.evaluator_device == "local" else "remote"


# ═══════════════════════════════════════════════════════════════════ partitioning

def partition_agents(members: Sequence, n_workers: int) -> list:
    """Which agents each worker owns: ``[[idx, ...], ...]``, one list per worker.

    Two things are being balanced.  **Cost**: an ante is a barrier — every worker's agents
    must reach the Nemesis before the matrix can be built — so a worker holding four
    scripted anchors (which never search) while another holds four 60-sim seats wastes
    three quarters of a core.  Agents are therefore dealt in descending ``sims`` order, so
    the expensive seats spread out first and the free ones fill the gaps.  **Batching**:
    within a worker, seats sharing a net batch into one ``BatchedSearch``; because the
    ``m_current`` seats all share the live net and are also the most expensive, dealing by
    cost keeps them together anyway.

    Deterministic in ``(members, n_workers)`` — the same partition every generation, and a
    partition a test can assert on.
    """
    n_workers = max(1, int(n_workers))
    order = sorted(members, key=lambda m: (-int(getattr(m, "sims", 0)), int(m.idx)))
    buckets: list = [[] for _ in range(n_workers)]
    load = [0] * n_workers
    for m in order:
        w = min(range(n_workers), key=lambda k: (load[k], len(buckets[k]), k))
        buckets[w].append(int(m.idx))
        load[w] += max(1, int(getattr(m, "sims", 0)))
    return [sorted(b) for b in buckets]


# ═══════════════════════════════════════════════════════════════════ the pool

class WorkerPool:
    """Spawns and drives the workers; owns the evaluator."""

    def __init__(self, cfg: PoolConfig, layout, *, encoder: str = "set",
                 caps: Optional[dict] = None, is_set: bool = True,
                 hand_type_features: bool = True):
        self.cfg = cfg
        self.layout = layout
        self.encoder = encoder
        self.caps = caps
        self.is_set = bool(is_set)
        self.hand_type_features = bool(hand_type_features)
        self.ctx = mp.get_context("spawn")
        self.n = max(1, int(cfg.n_workers))
        self.procs: list = []
        self.cmd_qs: list = []
        self.res_q = None
        self.leaf_q = None
        self.arenas: Optional[ArenaSet] = None
        self.conns: list = []            # parent's SEND ends
        self.child_conns: list = []      # children's RECV ends (kept so they stay open)
        self.evaluator = None
        self.channel: Optional[EvaluatorChannel] = None
        self.live: set = set()
        self.dead: dict = {}
        self._seq = 0
        self.stale = 0
        self.started = False

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self.started:                                     # pragma: no cover - defensive
            raise RuntimeError("pool already started")
        for root in ROOTS:                # so the children inherit an importable sys.path
            if root not in sys.path:
                sys.path.insert(0, root)
        self.res_q = self.ctx.Queue()
        self.leaf_q = self.ctx.Queue() if self.cfg.mode == "remote" else None
        if self.cfg.mode == "remote":
            self.arenas = ArenaSet(self.n, self.cfg.request_mb << 20,
                                   self.cfg.reply_mb << 20)
        for w in range(self.n):
            recv, send = (self.ctx.Pipe(duplex=False) if self.cfg.mode == "remote"
                          else (None, None))
            self.conns.append(send)
            self.child_conns.append(recv)
            cmd_q = self.ctx.Queue()
            self.cmd_qs.append(cmd_q)
            spec = WorkerSpec(
                worker_id=w, roots=ROOTS,
                arena=(self.arenas.handle(w) if self.arenas is not None else None),
                layout=self.layout, encoder=self.encoder, caps=self.caps,
                hand_type_features=self.hand_type_features, mode=self.cfg.mode,
                device=self.cfg.worker_device,
                torch_threads=self.cfg.torch_threads,
                poll_seconds=self.cfg.worker_poll_seconds,
                max_action_rows=self.cfg.max_action_rows)
            p = self.ctx.Process(target=worker_main,
                                 args=(spec, cmd_q, self.res_q, self.leaf_q, recv),
                                 name=f"mp-worker-{w}", daemon=True)
            p.start()
            self.procs.append(p)
            self.live.add(w)
        if self.cfg.mode == "remote":
            from .evaluator import BatchEvaluator
            self.channel = EvaluatorChannel(self.arenas, self.leaf_q, self.conns,
                                            max_wait_ms=self.cfg.max_wait_ms)
            self.evaluator = BatchEvaluator(
                self.channel, self.layout, is_set=self.is_set,
                device=self.cfg.evaluator_device,
                max_action_rows=self.cfg.max_action_rows,
                poll_seconds=self.cfg.evaluator_poll_seconds)
        self.started = True

    def start_evaluator(self) -> None:
        if self.evaluator is not None and self.evaluator._thread is None:
            self.evaluator.start()

    def close(self, timeout: float = 10.0) -> None:
        for w in sorted(self.live):
            try:
                self.cmd_qs[w].put({"op": OP_SHUTDOWN, "seq": -1})
            except Exception:                                # pragma: no cover - teardown
                pass
        deadline = time.time() + timeout
        for w, p in enumerate(self.procs):
            p.join(timeout=max(0.1, deadline - time.time()))
            if p.is_alive():
                p.terminate()
                p.join(timeout=2.0)
        if self.evaluator is not None:
            self.evaluator.stop()
        for c in self.conns:
            if c is not None:
                try:
                    c.close()
                except Exception:                            # pragma: no cover - teardown
                    pass
        for c in self.child_conns:
            if c is not None:
                try:
                    c.close()
                except Exception:                            # pragma: no cover - teardown
                    pass
        if self.arenas is not None:
            self.arenas.close()
        self.started = False
        self.live = set()

    # ── commands ─────────────────────────────────────────────────────────────

    def call(self, op: str, payloads: dict, timeout: Optional[float] = None) -> dict:
        """Send one command per worker in ``payloads`` (``{worker_id: cmd dict}``) and
        wait for every live one to answer.  Returns ``{worker_id: payload}``; workers that
        died on the way are absent and recorded in ``self.dead``."""
        self._seq += 1
        seq = self._seq
        targets = {w for w in payloads if w in self.live}
        for w in sorted(targets):
            cmd = dict(payloads[w])
            cmd["op"] = op
            cmd["seq"] = seq
            self.cmd_qs[w].put(cmd)

        out: dict = {}
        pending = set(targets)
        deadline = time.time() + (self.cfg.call_timeout if timeout is None else timeout)
        while pending:
            self._check_evaluator()
            try:
                wid, got_seq, ok, payload = self.res_q.get(timeout=0.2)
            except _queue.Empty:
                self._reap(pending, out)
                if time.time() > deadline:
                    raise TimeoutError(
                        f"pool.call({op!r}) timed out with workers {sorted(pending)} "
                        "still outstanding")
                continue
            if got_seq != seq:
                # A reply from a call that had already given up on this worker (it was
                # reaped on a timeout and then answered anyway).  Drop it: acting on it
                # would attribute one command's result to another.
                self.stale += 1
                continue
            pending.discard(wid)
            if ok:
                out[wid] = payload
            else:
                self._bury(wid, f"worker error:\n{payload}")
        if targets and not out:
            # Every worker asked died on the same command.  That is a bug in the setup,
            # not a flake in one process: fail loudly rather than quietly playing a
            # tournament with nobody in it.
            raise RuntimeError(
                f"every worker died on {op!r}. First reason: "
                + next(iter(self.dead.values()), "(no reason recorded)"))
        return out

    def broadcast(self, op: str, cmd: Optional[dict] = None,
                  timeout: Optional[float] = None) -> dict:
        return self.call(op, {w: dict(cmd or {}) for w in sorted(self.live)},
                         timeout=timeout)

    def crash_worker(self, w: int) -> None:
        """TEST ONLY: make worker ``w`` die without draining, to exercise the crash path."""
        if w in self.live:
            self.cmd_qs[w].put({"op": OP_CRASH, "seq": -1})

    # ── failure handling ─────────────────────────────────────────────────────

    def _reap(self, pending: set, out: dict) -> None:
        for w in list(pending):
            p = self.procs[w]
            if not p.is_alive():
                code = p.exitcode
                pending.discard(w)
                self._bury(w, f"worker process exited with code {code}"
                              + (" (deliberate test crash)" if code == CRASH_EXIT_CODE else ""))

    def _bury(self, w: int, why: str) -> None:
        self.live.discard(w)
        self.dead[w] = why
        if self.evaluator is not None:
            self.evaluator.release_workers([w])

    def _check_evaluator(self) -> None:
        if self.evaluator is not None and self.evaluator.error is not None:
            err = self.evaluator.error
            self.evaluator.error = None
            raise RuntimeError(f"evaluator thread died: {err!r}") from err

    def respawn_dead(self) -> list:
        """Bring dead workers back for the NEXT generation.  Their arenas, queues and
        pipes are reused (the parent still holds every handle), so this is a fresh process
        against the same transport.  Returns the worker ids that came back."""
        back = []
        for w, _why in list(self.dead.items()):
            # The dead worker's unread commands are not the new one's: a leftover would be
            # answered under a stale seq and dropped, and the command it was really sent
            # would never be seen.
            _drain(self.cmd_qs[w])
            spec = WorkerSpec(
                worker_id=w, roots=ROOTS,
                arena=(self.arenas.handle(w) if self.arenas is not None else None),
                layout=self.layout, encoder=self.encoder, caps=self.caps,
                hand_type_features=self.hand_type_features, mode=self.cfg.mode,
                device=self.cfg.worker_device,
                torch_threads=self.cfg.torch_threads,
                poll_seconds=self.cfg.worker_poll_seconds,
                max_action_rows=self.cfg.max_action_rows)
            p = self.ctx.Process(target=worker_main,
                                 args=(spec, self.cmd_qs[w], self.res_q, self.leaf_q,
                                       self.child_conns[w]),
                                 name=f"mp-worker-{w}", daemon=True)
            p.start()
            self.procs[w] = p
            self.live.add(w)
            self.dead.pop(w, None)
            back.append(w)
        return back


# ═══════════════════════════════════════════════════════════════════ the driver

class MPDriver:
    """``TournamentDriver`` over a :class:`WorkerPool`.

    Holds the agent -> worker map for the generation and nothing else: every game, tree
    and logger is in a worker.  The last outcome seen for each agent is kept so an agent
    whose worker dies can still be reported with its true lives rather than a zero.
    """

    def __init__(self, pool: WorkerPool, owners: dict, n_agents: int, setup_extra: dict):
        self.pool = pool
        self.owners = dict(owners)                 # agent idx -> worker id
        self.n_agents = int(n_agents)
        self.setup_extra = dict(setup_extra)
        self._last: dict = {}
        self.crashed: set = set()

    # -- helpers ----------------------------------------------------------------------

    def _by_worker(self, indices: Sequence[int]) -> dict:
        out: dict = {}
        for i in indices:
            w = self.owners.get(int(i))
            if w is None or w not in self.pool.live:
                self.crashed.add(int(i))
                continue
            out.setdefault(w, []).append(int(i))
        return out

    def _alive_agents(self) -> list:
        return [i for i, w in self.owners.items() if w in self.pool.live]

    # -- protocol ---------------------------------------------------------------------

    def setup(self, seed, n_agents: int, deck_key: str, stake, lives: int,
              ruleset: str = "mlb", fanout: str = "clone") -> str:
        seed_str = _resolve_seed(seed, deck_key, stake, ruleset)
        setup = TournamentSetup(seed_str=seed_str, deck_key=deck_key, stake=stake,
                                lives=lives, ruleset=ruleset, fanout=fanout,
                                n_agents=n_agents, **self.setup_extra)
        self.pool.broadcast(OP_FANOUT, {"setup": setup})
        self._last = {}
        return seed_str

    def drive(self, indices: Sequence[int], max_steps: int, noop_budget: int) -> dict:
        from tournament.parallel import DriveOutcome

        by_worker = self._by_worker(indices)
        payloads = {w: {"indices": idxs, "max_steps": max_steps,
                        "noop_budget": noop_budget}
                    for w, idxs in by_worker.items()}
        results = self.pool.call(OP_DRIVE, payloads)
        out: dict = {}
        seen: set = set()
        for w, per_agent in results.items():
            for i, d in per_agent.items():
                i = int(i)
                out[i] = DriveOutcome(**d)
                self._last[i] = d
                seen.add(i)
        for i in indices:
            i = int(i)
            if i in seen:
                continue
            self.crashed.add(i)
            last = self._last.get(i, {})
            out[i] = DriveOutcome(status="crashed", steps=0, forced=0,
                                  chips=float(last.get("chips", 0.0)),
                                  lives=int(last.get("lives", 0)),
                                  ante=int(last.get("ante", 0)))
        return out

    def apply(self, ops: Sequence[tuple]) -> dict:
        grouped: dict = {}
        for op in ops:
            w = self.owners.get(int(op[1]))
            if w is None or w not in self.pool.live:
                continue
            grouped.setdefault(w, []).append(tuple(op))
        results = self.pool.call(OP_APPLY, {w: {"ops": v} for w, v in grouped.items()})
        out: dict = {}
        for per_agent in results.values():
            for i, d in per_agent.items():
                out[int(i)] = d
                self._last[int(i)] = {**self._last.get(int(i), {}), **d}
        for op in ops:
            i = int(op[1])
            if i not in out:
                last = self._last.get(i, {})
                out[i] = {"lives": int(last.get("lives", 0)),
                          "ante": int(last.get("ante", 0)),
                          "chips": float(last.get("chips", 0.0))}
        return out

    def summarize(self) -> dict:
        from tournament.parallel import AgentSummary

        results = self.pool.broadcast(OP_SUMMARIZE)
        out: dict = {}
        for per_agent in results.values():
            for i, d in per_agent.items():
                out[int(i)] = AgentSummary(**d)
        for i in range(self.n_agents):
            if i not in out:
                last = self._last.get(i, {})
                out[i] = AgentSummary(lives=int(last.get("lives", 0)),
                                      ante=int(last.get("ante", 0)), jokers=(),
                                      chips=float(last.get("chips", 0.0)), alive=False)
        return out

    def close(self) -> None:
        pass


def _drain(q) -> int:
    """Empty a queue without blocking.  Returns how many messages were thrown away."""
    n = 0
    while True:
        try:
            q.get_nowait()
        except Exception:
            return n
        n += 1


def _resolve_seed(seed, deck_key: str, stake, ruleset: str) -> str:
    """Turn whatever the trainer handed us into the exact seed STRING every worker will
    build from.  ``construct_games`` does this implicitly by pinning subsequent
    constructions to the first game's ``seed_str``; with the games spread over processes
    the pinning has to happen here instead, or a ``seed=None`` run would give each worker
    a different game."""
    if isinstance(seed, str):
        return seed
    from tournament.bootstrap import BalatroGame
    return BalatroGame(seed=seed, deck_key=deck_key, stake=stake, ruleset=ruleset).seed_str


__all__ = ["PoolConfig", "WorkerPool", "MPDriver", "partition_agents", "ROOTS"]
