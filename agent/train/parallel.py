"""
parallel.py — ``MLBTrainer`` with the tournament played by N worker processes.

    ParallelMLBTrainer(cfg, mlb, pool_cfg=PoolConfig(n_workers=12, evaluator_device="cpu"))

is an ``MLBTrainer`` in every respect that touches a checkpoint.  It **subclasses** it and
overrides exactly one method — ``_play_tournament`` — so the net, the optimizer, the replay
buffer, the numpy generator, the counters, the generation counter, the opponent history,
``state_dict`` / ``load_state_dict`` / ``from_checkpoint`` and therefore the on-disk format
are literally the inherited ones.  That is the non-negotiable requirement from the brief:

    python scripts/train_mlb.py --resume runs/real1/latest.pt --workers 12

continues the live single-process run, and a checkpoint this trainer writes resumes into
the single-process one.  Nothing about the worker count is recorded in the checkpoint —
it is a property of the machine, not of the experiment (``MLBTrainConfig`` is untouched).

What a generation looks like
----------------------------
1. ``build_population`` (main) — the same pure function, so the same population.
2. Each distinct net in it gets a ``policy_id``; the evaluator loads them and the live
   net's mirror is re-synced from the weights the last training step produced.
3. ``partition_agents`` splits the seats over the workers; each worker builds ITS players
   (trees, rngs, heuristic prior, skip cap, sample collectors) and nothing else.
4. Per seed: ``ParallelTournament`` (main) drives the ante lockstep through ``MPDriver``;
   the workers play, their MCTS leaves batch across processes at the evaluator.
5. The records come back, get their value targets from the matrices (main, unchanged
   ``value_targets_from_result`` / ``assign_value_targets``), and are re-ordered by
   ``(seed, agent, decision)`` so the replay buffer sees one deterministic stream.

The one deliberate difference from the serial path
--------------------------------------------------
W1's ``SampleBuilder`` subsamples the action set with an RNG, and in the serial path that
RNG is the trainer's single shared generator.  A worker cannot draw from it, so each agent
gets its own stream seeded from ``(cfg.seed, generation, agent idx)`` — which makes the
subsampling independent of the worker COUNT, something the shared generator never was.
Consequence, stated plainly: a parallel generation is not bit-identical to the serial
generation it replaces.  It is the same experiment with a different (and more
reproducible) noise stream.  See PARALLEL_NOTES §5.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

from parallel.pool import MPDriver, PoolConfig, WorkerPool, partition_agents
from parallel.protocol import GenerationSpec, LIVE_POLICY_ID, OP_COLLECT, OP_GENERATION, OP_STATS

from .population import population_summary
from .selfplay import (
    MLBTrainer, _tournament_metrics, assign_value_targets, tournament_module,
    value_targets_from_result,
)

__all__ = ["ParallelMLBTrainer", "assign_policy_ids", "merge_trajectory_parts"]


def assign_policy_ids(members) -> tuple:
    """``({agent idx: policy_id}, {policy_id: checkpoint path})``.

    ``LIVE_POLICY_ID`` (0) is the net being trained — every ``is_current`` seat and every
    history-less fallback seat.  Each distinct past-self checkpoint gets the next id, in
    the order it first appears, so the mapping is a pure function of the population.
    Anchors are scripted and get no id at all.
    """
    by_agent: dict = {}
    by_path: dict = {}
    for m in members:
        if m.is_anchor:
            continue
        if m.checkpoint is None:
            by_agent[m.idx] = LIVE_POLICY_ID
            continue
        pid = by_path.get(m.checkpoint)
        if pid is None:
            pid = len(by_path) + 1
            by_path[m.checkpoint] = pid
        by_agent[m.idx] = pid
    return by_agent, {pid: path for path, pid in by_path.items()}


def merge_trajectory_parts(target: Path, parts: list, delete: bool = True) -> int:
    """Concatenate the per-worker trajectory files into the run's one file.

    Workers cannot share an append handle safely across processes, so each writes
    ``trajectories.w<id>.jsonl`` and the generation ends by folding them in.  One episode
    is one line, so concatenation is the whole merge; line ORDER across workers is not part
    of any contract (``replay`` reads each line independently and each carries its own
    agent index).  Returns the number of lines merged.
    """
    n = 0
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8") as out:
        for part in parts:
            p = Path(part)
            if not p.is_file():
                continue
            with open(p, "r", encoding="utf-8") as src:
                for line in src:
                    if line.strip():
                        out.write(line if line.endswith("\n") else line + "\n")
                        n += 1
            if delete:
                try:
                    p.unlink()
                except OSError:                              # pragma: no cover - Windows lock
                    pass
    return n


class ParallelMLBTrainer(MLBTrainer):
    """See the module docstring.  ``objective="external"`` falls through to the inherited
    single-process path: it is one solo game per episode, so there is nothing to batch
    across and nothing to parallelise that the worker pool would not just slow down."""

    #: Same default ``TournamentSpec`` uses; a wedged agent is a bug, not a budget.
    max_steps_per_drive = 20_000

    def __init__(self, cfg, mlb, *, pool_cfg: Optional[PoolConfig] = None,
                 traj_config: Optional[dict] = None, work_dir: Optional[str] = None):
        super().__init__(cfg, mlb)
        self.pool_cfg = pool_cfg or PoolConfig()
        #: Where the "local" mode's weights cache goes; the run directory when the script
        #: passes one, else the current directory.
        self.work_dir = Path(work_dir) if work_dir else None
        #: ``{"path": <run dir>/trajectories.jsonl, "sig_every": N}`` or None.  The serial
        #: path takes a logger FACTORY; a worker cannot be handed a closure, so the
        #: parallel path takes the two values a worker needs to build its own.
        self.traj_config = traj_config
        self.pool: Optional[WorkerPool] = None
        self.parallel_stats: dict = {}
        self._layout = None
        self._eval_models: dict = {}
        self._generation_spec_sent = -1

    # ── pool lifecycle ───────────────────────────────────────────────────────

    def ensure_pool(self) -> WorkerPool:
        if self.pool is not None:
            return self.pool
        from balatro_sim.game import BalatroGame
        from parallel.layout import LeafLayout
        from parallel.leaf import LeafEncoder

        enc = self.cold.encoder
        caps = enc.caps.as_dict() if getattr(enc, "is_set", False) else None
        probe = LeafEncoder(self.cfg.encoder, caps=caps)
        game = BalatroGame(seed="7I4M53DL", deck_key=self.cfg.deck_key,
                           stake=self.cfg.stake, ruleset="mlb")
        self._layout = LeafLayout.from_prototype(*probe.prototype(game))
        self.pool = WorkerPool(self.pool_cfg, self._layout, encoder=self.cfg.encoder,
                               caps=caps, is_set=bool(getattr(enc, "is_set", False)))
        self.pool.start()
        self.pool.start_evaluator()
        return self.pool

    def close(self) -> None:
        """Drain and stop the workers.  Safe to call twice; the PAUSE / SIGTERM path and
        the normal exit both go through it."""
        if self.pool is not None:
            self.pool.close()
            self.pool = None

    # ── the evaluator's models ───────────────────────────────────────────────

    def _mirror_net(self):
        """A private copy of the live net on the evaluator's device.  A copy, not the
        trainer's object, so that "the evaluator sees the weights the last training step
        produced" is an explicit act (``sync_weights``) rather than an aliasing accident —
        and so an evaluator on CUDA and a trainer on CPU need no special case."""
        from mcts.model import PolicyValueNet

        desc = self.net.describe()
        if desc.get("kind") == "set" or getattr(self.cold.encoder, "is_set", False):
            from mcts.model_set import SetPolicyValueNet
            mirror = SetPolicyValueNet.from_description(desc)
        else:
            mirror = PolicyValueNet.from_description(desc)
        mirror.load_state_dict(self.net.state_dict())
        return mirror

    def _register_models(self, checkpoints: dict) -> None:
        ev = self.pool.evaluator
        if ev is None:                       # "local" mode: the workers hold the nets
            return
        if LIVE_POLICY_ID not in self._eval_models:
            self._eval_models[LIVE_POLICY_ID] = self._mirror_net()
            ev.set_model(LIVE_POLICY_ID, self._eval_models[LIVE_POLICY_ID])
        # The broadcast the brief asks for: after the last training step, before play.
        ev.sync_weights(self.net, LIVE_POLICY_ID)
        for pid, path in checkpoints.items():
            if pid in self._eval_models:
                continue
            from mcts.player import load_policy
            policy = load_policy(path, device=self.pool_cfg.evaluator_device,
                                 batched=True, encoder=self.cfg.encoder)
            self._eval_models[pid] = policy.model
            ev.set_model(pid, policy.model)

    def _write_weights_file(self) -> Optional[str]:
        """"local" mode only: the generation's weights as a checkpoint the workers can
        load.  Written once per generation to the run directory (or beside the
        trajectories) and overwritten in place; it is a cache, not a record."""
        if self.pool_cfg.mode != "local":
            return None
        from .checkpoint import save_checkpoint

        base = self.work_dir or (Path(self.traj_config["path"]).parent
                                 if self.traj_config else Path("."))
        path = base / "_worker_weights.pt"
        enc = self.cold.encoder
        save_checkpoint(path, {
            "config": {"encoder": self.cfg.encoder},
            "net_desc": self.net.describe(), "encoder": self.cfg.encoder,
            "net_kind": "set" if getattr(enc, "is_set", False) else "flat",
            "encoder_caps": (enc.caps.as_dict() if getattr(enc, "is_set", False) else None),
            "model": {k: v.detach().cpu() for k, v in self.net.state_dict().items()},
        })
        return str(path)

    # ── one generation of tournaments ────────────────────────────────────────

    def _play_tournament(self, stop_check, traj_logger_factory, out_dir):
        if self.mlb.objective != "tournament":                # pragma: no cover - guarded
            return super()._play_tournament(stop_check, traj_logger_factory, out_dir)
        from tournament.parallel import ParallelTournament

        pool = self.ensure_pool()
        if pool.dead:
            back = pool.respawn_dead()
            if back:
                self._generation_spec_sent = -1               # they need the population
        members = self._build_members()
        policy_ids, checkpoints = assign_policy_ids(members)
        self._register_models(checkpoints)
        weights_path = self._write_weights_file()

        buckets = partition_agents(members, pool.n)
        owners = {idx: w for w, idxs in enumerate(buckets) for idx in idxs}
        by_idx = {m.idx: m for m in members}
        traj_paths = self._send_generation(pool, buckets, by_idx, policy_ids, checkpoints,
                                           weights_path)

        seeds = [self._episode_seed() for _ in range(self.mlb.seeds_per_generation)]
        tour_mod = tournament_module()                        # puts the repo root on sys.path
        del tour_mod

        all_records: list = []
        results: list = []
        joker_sets: list = []
        antes_reached: list = []
        tie_fractions: list = []
        lives_lost: dict = {}
        group_ranks: dict = {"current": [], "anchor": [], "history": []}
        groups = {"current": [m.idx for m in members if m.is_current],
                  "anchor": [m.idx for m in members if m.is_anchor],
                  "history": [m.idx for m in members
                              if not m.is_current and not m.is_anchor]}
        crashed: list = []
        n_played = 0
        t0 = time.perf_counter()

        for si, seed in enumerate(seeds):
            if stop_check is not None and stop_check():
                break                # PAUSE / SIGTERM: one tournament is the atomic unit
            driver = MPDriver(pool, owners, self.mlb.n_agents,
                              {"life_rule": self.mlb.life_rule,
                               "max_ante": self.mlb.max_ante,
                               "traj": self.traj_config and dict(self.traj_config)})
            tour = ParallelTournament(
                seed=seed, n_agents=self.mlb.n_agents, players=[None] * self.mlb.n_agents,
                deck_key=self.cfg.deck_key, stake=self.cfg.stake,
                life_rule=self.mlb.life_rule, max_ante=self.mlb.max_ante,
                lives=self.cfg.lives, max_steps_per_drive=self.max_steps_per_drive,
                out_dir=(str(Path(out_dir) / f"seed{si:02d}") if out_dir else None),
                driver=driver)
            result = tour.run()
            results.append(result)
            crashed.extend(getattr(result, "crashed", []))

            targets = value_targets_from_result(result, self.mlb.n_agents,
                                                self.mlb.value_blend)
            for name, idxs in groups.items():
                group_ranks[name].extend(
                    targets["rank_by_ante"][a][i]
                    for a in targets["nemesis_antes"] for i in idxs
                    if not np.isnan(targets["rank_by_ante"][a][i]))

            seed_records = self._collect_records(pool)
            assign_value_targets(seed_records, targets)
            all_records.extend(seed_records)

            for i in range(self.mlb.n_agents):
                g = tour._last_games[i]
                joker_sets.append(tuple(sorted(j.key for j in g.jokers)))
                antes_reached.append(int(g.ante))
                fl = result.final_lives[i]
                if fl is not None and self.mlb.life_rule != "none":
                    k = int(max(0, self.cfg.lives - fl))
                    lives_lost[k] = lives_lost.get(k, 0) + 1
            tie_fractions.extend([am.tie_fraction for am in result.ante_matrices
                                  if not np.isnan(am.tie_fraction)])
            n_played += 1

        wall = time.perf_counter() - t0
        stats = pool.broadcast(OP_STATS) if pool.live else {}
        searches = sum(s.get("searches", 0) for s in stats.values())
        leaves = (pool.evaluator.stats.leaves if pool.evaluator is not None
                  else sum(s.get("leaves", 0) for s in stats.values()))
        net_seats = [m for m in members if not m.is_anchor] or list(members)
        sims = sum(m.sims for m in net_seats) / len(net_seats) * max(0, searches)

        metrics = _tournament_metrics(all_records, joker_sets, lives_lost, tie_fractions,
                                      antes_reached, wall, searches, sims, leaves,
                                      episodes=n_played * self.mlb.n_agents)
        for name, vals in group_ranks.items():
            metrics[f"rank_{name}"] = float(np.mean(vals)) if vals else float("nan")
        metrics["population"] = population_summary(members)
        metrics["seeds"] = [str(s) for s in seeds]
        metrics.update(self._parallel_metrics(pool, stats, crashed, traj_paths))
        # Kept for tests and the benchmark (the serial path exposes the same thing through
        # `GenerationSamples`); not part of any checkpoint.
        self.last_results = results
        self.last_records = all_records
        return [r.sample for r in all_records], metrics

    # ── pieces ───────────────────────────────────────────────────────────────

    def _build_members(self):
        from .population import build_population
        return build_population(self.pop_cfg, self.generation, self.history,
                                base_seed=self.cfg.seed)

    def _send_generation(self, pool, buckets, by_idx, policy_ids, checkpoints,
                         weights_path) -> list:
        sample_kwargs = (None if self.cold.sample_builder is None else
                         {"k_unvisited": self.cfg.k_unvisited,
                          "subsample": bool(self.cfg.subsample)})
        traj_paths: list = []
        payloads: dict = {}
        for w, idxs in enumerate(buckets):
            if w not in pool.live:
                continue
            spec = GenerationSpec(
                generation=self.generation,
                members=tuple(by_idx[i] for i in idxs),
                policy_ids={i: policy_ids[i] for i in idxs if i in policy_ids},
                checkpoints=dict(checkpoints), weights_path=weights_path,
                encoder=self.cfg.encoder, device=self.pool_cfg.worker_device,
                leaf_batch=self.mlb.leaf_batch, reuse=self.mlb.reuse,
                strategy=self.mlb.strategy, starting_lives=self.cfg.lives,
                horizon_antes=max(1, self.mlb.max_ante),
                max_skips_per_ante=self.effective_skip_cap(),
                heuristic=self.heuristic_kwargs(), sample=sample_kwargs,
                sample_seed=int(self.cfg.seed), record_current=True,
                max_samples_per_agent=self.mlb.max_samples_per_agent)
            payloads[w] = {"spec": spec}
        pool.call(OP_GENERATION, payloads)
        if self.traj_config:
            from parallel.worker import _worker_traj_path
            traj_paths = [_worker_traj_path(self.traj_config["path"], w)
                          for w in range(pool.n)]
        return traj_paths

    def _collect_records(self, pool) -> list:
        """Every worker's records for the tournament just played, re-ordered by
        ``(agent idx, decision order)`` so the buffer's stream does not depend on which
        worker happened to answer first."""
        per_agent: dict = {}
        for chunks in pool.broadcast(OP_COLLECT).values():
            for idx, records, _dropped in chunks:
                per_agent.setdefault(int(idx), []).extend(records)
        out: list = []
        for idx in sorted(per_agent):
            out.extend(per_agent[idx])
        return out

    def _parallel_metrics(self, pool, stats, crashed, traj_paths) -> dict:
        out: dict = {"workers": pool.n, "workers_live": len(pool.live),
                     "evaluator_device": self.pool_cfg.evaluator_device,
                     "crashed_agents": sorted(set(crashed)),
                     "dead_workers": sorted(pool.dead),
                     "dead_worker_reasons": {w: str(r).strip().splitlines()[-1]
                                             for w, r in sorted(pool.dead.items())}}
        if pool.evaluator is not None:
            out.update(pool.evaluator.stats.as_dict())
            pool.evaluator.stats.reset()
        waits = [s.get("wait_s", 0.0) for s in stats.values() if "wait_s" in s]
        if waits:
            out["worker_wait_s_mean"] = round(float(np.mean(waits)), 3)
            out["worker_wait_s_max"] = round(float(np.max(waits)), 3)
        if self.traj_config and traj_paths:
            out["trajectories_merged"] = merge_trajectory_parts(
                Path(self.traj_config["path"]), traj_paths)
        return out
