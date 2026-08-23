"""
worker.py — one self-play worker process.

A worker owns a SLICE of the tournament's agents: their ``BalatroGame``s, their
``MCTSPlayer``s (trees, rngs, tree caches, W0's heuristic prior, the skip-cap filter),
their ``SampleCollector``s and their ``TrajectoryLogger``s.  It owns **no net**: every
leaf evaluation goes out through :class:`~.remote.RemotePolicy` to the shared evaluator in
the main process (``mode="remote"``).  ``mode="local"`` is the control arm — the worker
loads the generation's weights itself and runs the net on its own core — which exists
because the honest answer to "does a shared batched evaluator beat N independent CPU nets
on THIS box" is a measurement, not an opinion; ``benchmarks/bench_parallel.py`` runs both.

Windows / spawn
---------------
The child re-imports this module and unpickles a :class:`~.protocol.WorkerSpec`.  Nothing
in the spec is a closure or a handle; ``sys.path`` is re-established from the roots in the
spec before any ``mcts`` / ``train`` / ``tournament`` import, so the worker can never pick
up the repo-root ``balatro_sim`` instead of the frozen fork.  ``torch.set_num_threads(1)``
is set before the first forward pass any worker might do (local mode) — 16 workers each
spawning 16 BLAS threads is the classic way to make a 16-core box slower than one core.

Failure policy
--------------
Anything that escapes a command handler is reported to the main process as an error result
with the traceback, and the worker keeps serving; anything that escapes the SERVE LOOP
(the evaluator vanishing, a broken queue) exits the process. Either way the main process's
:class:`~.pool.WorkerPool` marks the worker's agents "crashed" for the tournament in
flight and the run continues without them — ``ParallelTournament`` treats a crash exactly
like a death.
"""
from __future__ import annotations

import os
import sys
import traceback
from typing import Optional

from .protocol import (
    CRASH_EXIT_CODE, GenerationSpec, LIVE_POLICY_ID, OP_APPLY, OP_COLLECT, OP_CRASH,
    OP_DRIVE, OP_FANOUT, OP_GENERATION, OP_SHUTDOWN, OP_STATS, OP_SUMMARIZE, TournamentSetup,
    WorkerSpec,
)


# ── entry point ────────────────────────────────────────────────────────────────

def worker_main(spec: WorkerSpec, cmd_q, res_q, leaf_q, reply_conn) -> int:
    """``Process(target=worker_main, args=(...))``.  Returns the process exit code."""
    _prepare_paths(spec.roots)
    import torch

    torch.set_num_threads(max(1, int(spec.torch_threads)))
    try:
        worker = Worker(spec, cmd_q, res_q, leaf_q, reply_conn)
    except BaseException:                                    # noqa: BLE001
        try:
            res_q.put((spec.worker_id, -1, False, traceback.format_exc()))
        except Exception:                                    # pragma: no cover - teardown
            pass
        return 1
    return worker.serve()


def _prepare_paths(roots) -> None:
    """mp/engine then mp/agent at the front (agent wins), mp/ on the end for
    ``import tournament`` — the same order ``conftest.py`` and ``scripts/_bootstrap.py``
    establish, so the fork guard in either of them passes in a worker too."""
    engine_root, agent_root, mp_root = (str(r) for r in roots)
    for root in (engine_root, agent_root):
        if root in sys.path:
            sys.path.remove(root)
        sys.path.insert(0, root)
    if mp_root not in sys.path:
        sys.path.append(mp_root)


# ── the worker ─────────────────────────────────────────────────────────────────

class Worker:
    """Serves commands until told to stop."""

    def __init__(self, spec: WorkerSpec, cmd_q, res_q, leaf_q, reply_conn):
        from .channel import WorkerArena, WorkerChannel
        from .leaf import LeafEncoder
        from .lockstep import LockstepDecider

        self.spec = spec
        self.id = spec.worker_id
        self.cmd_q = cmd_q
        self.res_q = res_q
        self.leaf = LeafEncoder(spec.encoder, caps=spec.caps,
                                hand_type_features=spec.hand_type_features)
        self.layout = spec.layout
        self.decider = LockstepDecider()
        self.arena = None
        self.channel = None
        if spec.mode == "remote":
            self.arena = WorkerArena(spec.arena)
            self.channel = WorkerChannel(self.arena, leaf_q, reply_conn,
                                         poll_seconds=spec.poll_seconds)
        self.gen: Optional[GenerationSpec] = None
        self.driver = None
        self.players: dict = {}
        self.collectors: dict = {}
        self.policies: dict = {}
        self.loggers: dict = {}
        self.records: list = []
        self._traj: Optional[dict] = None

    # ── serve loop ───────────────────────────────────────────────────────────

    def serve(self) -> int:
        while True:
            cmd = self.cmd_q.get()
            op = cmd.get("op")
            seq = cmd.get("seq", 0)
            if op == OP_SHUTDOWN:
                self._close()
                self.res_q.put((self.id, seq, True, {"op": OP_SHUTDOWN}))
                return 0
            if op == OP_CRASH:
                # Deliberate, test-only: no drain, no reply, no atexit handlers.
                os._exit(CRASH_EXIT_CODE)
            try:
                payload = self._dispatch(op, cmd)
            except _FatalWorkerError as exc:
                self.res_q.put((self.id, seq, False, f"{type(exc).__name__}: {exc}"))
                self._close()
                return 2
            except BaseException:                            # noqa: BLE001
                self.res_q.put((self.id, seq, False, traceback.format_exc()))
                continue
            self.res_q.put((self.id, seq, True, payload))

    def _dispatch(self, op: str, cmd: dict):
        if op == OP_GENERATION:
            return self._generation(cmd["spec"])
        if op == OP_FANOUT:
            return self._fanout(cmd["setup"])
        if op == OP_DRIVE:
            return self._drive(cmd["indices"], cmd["max_steps"], cmd["noop_budget"])
        if op == OP_APPLY:
            return self._apply(cmd["ops"])
        if op == OP_SUMMARIZE:
            return self._summarize()
        if op == OP_COLLECT:
            return self._collect()
        if op == OP_STATS:
            return self._stats()
        raise ValueError(f"unknown worker op {op!r}")        # pragma: no cover

    # ── generation ───────────────────────────────────────────────────────────

    def _generation(self, gen: GenerationSpec) -> dict:
        from mcts.outcome import MLBOutcome
        from train.population import instantiate

        self.gen = gen
        self.policies = {}
        outcome = MLBOutcome(starting_lives=gen.starting_lives,
                             horizon_antes=max(1, gen.horizon_antes))
        players = instantiate(
            gen.members, None, device=gen.device, encoder=gen.encoder,
            leaf_batch=gen.leaf_batch, reuse=gen.reuse, strategy=gen.strategy,
            outcome=outcome, max_skips_per_ante=gen.max_skips_per_ante,
            policy_for=self._policy_for, **(gen.heuristic or {}))
        self.players = {m.idx: p for m, p in zip(gen.members, players)}
        self.collectors = self._build_collectors(gen)
        self.records = []
        return {"agents": sorted(self.players), "policies": sorted(self.policies)}

    def _policy_for(self, member):
        """Every net-driven seat gets a policy for ITS net.  Remote mode: a
        ``RemotePolicy`` tagged with the evaluator's id for that net (several seats sharing
        a net share one object, so ``BatchedSearch`` groups them into one batch).  Local
        mode: a real ``PolicyValueFn`` built here, cached per net."""
        gen = self.gen
        policy_id = int(gen.policy_ids[member.idx])
        cached = self.policies.get(policy_id)
        if cached is not None:
            return cached
        if self.spec.mode == "remote":
            from .remote import RemotePolicy
            policy = RemotePolicy(policy_id, self.channel, self.leaf, self.layout,
                                  name=f"w{self.id}p{policy_id}")
        else:
            from mcts.player import load_policy
            path = (gen.weights_path if policy_id == LIVE_POLICY_ID
                    else gen.checkpoints[policy_id])
            policy = load_policy(path, device=self.spec.device, batched=True,
                                 encoder=gen.encoder)
        self.policies[policy_id] = policy
        return policy

    def _build_collectors(self, gen: GenerationSpec) -> dict:
        """One ``SampleCollector`` per current-net seat this worker owns, with W1's
        ``SampleBuilder`` when the run uses one."""
        import numpy as np
        from train.sample import SampleBuilder
        from train.selfplay import SampleCollector

        out: dict = {}
        if not gen.record_current:
            return out
        for m in gen.members:
            if not m.is_current:
                continue
            sample_fn = None
            if gen.sample is not None:
                rng = np.random.default_rng(
                    (int(gen.sample_seed) * 1_000_003 + int(gen.generation) * 9_176
                     + int(m.idx)) % (2 ** 63))
                sample_fn = SampleBuilder(self.leaf.encoder, rng=rng, **gen.sample)
            out[m.idx] = SampleCollector(m.idx, self.leaf.encoder, sample_fn=sample_fn,
                                         max_records=gen.max_samples_per_agent)
        return out

    # ── one tournament ───────────────────────────────────────────────────────

    def _fanout(self, setup: TournamentSetup) -> dict:
        from tournament.parallel import LocalDriver

        self._close_loggers()
        self._traj = setup.traj
        indices = sorted(self.players)
        for idx, coll in self.collectors.items():
            coll.clear()
            self.players[idx].record_hook = coll
        self.driver = LocalDriver(
            [self.players[i] for i in indices], decide_many=self.decider,
            on_step=self._on_step, on_agent_done=self._on_agent_done,
            on_fanout=self._on_fanout, indices=indices)
        self._setup = setup
        seed_str = self.driver.setup(setup.seed_str, setup.n_agents, setup.deck_key,
                                     setup.stake, setup.lives, setup.ruleset, setup.fanout)
        return {"seed": seed_str, "agents": indices}

    def _drive(self, indices, max_steps: int, noop_budget: int) -> dict:
        from .channel import EvaluatorGone

        try:
            outcomes = self.driver.drive(indices, max_steps, noop_budget)
        except EvaluatorGone as exc:
            raise _FatalWorkerError(str(exc)) from exc
        return {i: vars(o) for i, o in outcomes.items()}

    def _apply(self, ops) -> dict:
        return self.driver.apply([tuple(op) for op in ops])

    def _summarize(self) -> dict:
        return {i: vars(s) for i, s in self.driver.summarize().items()}

    def _collect(self) -> list:
        """The tournament's samples, oldest first, grouped per agent so the main process
        can re-order them deterministically regardless of worker count."""
        out = []
        for idx in sorted(self.collectors):
            coll = self.collectors[idx]
            out.append((idx, coll.records, coll.dropped))
            coll.records = []
            coll.dropped = 0
        for p in self.players.values():
            p.record_hook = None
        self._close_loggers()
        return out

    def _stats(self) -> dict:
        st = {"worker": self.id, "decider": self.decider.stats,
              "searches": sum(getattr(p, "searches", 0) for p in self.players.values()),
              "shortcuts": sum(getattr(p, "shortcuts", 0) for p in self.players.values()),
              "leaves": sum(getattr(p, "leaves", 0) for p in self.policies.values()),
              "policy_calls": sum(getattr(p, "calls", 0) for p in self.policies.values())}
        if self.channel is not None:
            st.update(submissions=self.channel.submissions, leaf_sends=self.channel.leaves,
                      wait_s=round(self.channel.wait_seconds, 4))
        return st

    # ── trajectory-logger hooks (REPLAY_NOTES §2.3) ──────────────────────────

    def _on_fanout(self, games: dict, seed_str: str) -> None:
        if not self._traj:
            return
        from replay.log import TrajectoryLogger                 # noqa: WPS433

        setup = self._setup
        # PER WORKER, never the run's shared file: N processes appending 12 KB JSON lines
        # to one handle is a corrupted line waiting to happen, and there is no cross-process
        # append lock worth taking for a diagnostic.  `train/parallel.py` folds the parts
        # into `trajectories.jsonl` at the end of the generation.
        path = _worker_traj_path(self._traj["path"], self.id)
        sig_every = int(self._traj.get("sig_every", 50))
        for idx, game in games.items():
            lg = TrajectoryLogger(path, sig_every=sig_every)
            lg.begin(game, {"source": "train_mlb", "agent": idx,
                            "is_current": bool(idx in self.collectors),
                            "n_agents": setup.n_agents, "life_rule": setup.life_rule,
                            "max_ante": setup.max_ante, "worker": self.id})
            self.loggers[idx] = lg

    def _on_step(self, idx: int, game, action) -> None:
        lg = self.loggers.get(idx)
        if lg is not None:
            lg.step(game, action)

    def _on_agent_done(self, idx: int, game, reason: str) -> None:
        lg = self.loggers.pop(idx, None)
        if lg is not None:
            lg.end(game, {"objective": "tournament", "agent": idx, "reason": reason,
                          "lives": int(getattr(game, "lives", 0)),
                          "ante": int(getattr(game, "ante", 0))})

    def _close_loggers(self) -> None:
        """A tournament that ended early (PAUSE between antes) can leave loggers open;
        closing them here keeps ``trajectories.jsonl`` a set of complete episodes."""
        while self.loggers:
            idx, lg = self.loggers.popitem()
            game = self.driver.game(idx) if self.driver is not None else None
            if game is not None:
                lg.end(game, {"objective": "tournament", "agent": idx,
                              "reason": "abandoned"})

    def _close(self) -> None:
        self._close_loggers()
        if self.arena is not None:
            self.arena.close()


def _worker_traj_path(path: str, worker_id: int) -> str:
    """``.../trajectories.jsonl`` -> ``.../trajectories.w3.jsonl``.  Must agree with
    ``train/parallel.py::ParallelMLBTrainer._send_generation``, which builds the same names
    to merge them; ``tests/test_parallel.py::test_worker_trajectory_paths_agree`` pins it."""
    from pathlib import Path

    p = Path(path)
    return str(p.with_name(f"{p.stem}.w{worker_id}{p.suffix}"))


class _FatalWorkerError(RuntimeError):
    """Something the worker cannot continue from (the evaluator went away).  Reported,
    then the process exits so the pool sees a dead worker rather than a wedged one."""


__all__ = ["worker_main", "Worker", "_worker_traj_path"]
