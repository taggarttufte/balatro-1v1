"""
parallel.py — the lockstep form of ``Tournament.run``, and the driver seam that lets the
N agents live in other processes.

Why this file exists (Phase 5 infra item #1, CAMPAIGN_LOG 2026-08-22 19:30)
--------------------------------------------------------------------------
``runner.Tournament.run`` drives agent *i* all the way to its Nemesis before it starts
agent *i+1* (``runner._drive_to_next_nemesis``).  That is the correct *semantics* — the
agents are genuinely independent between Nemeses — but it is also the reason no two trees
ever want a leaf at the same moment, and it pins the whole tournament to one core.  Phase 4
measured the consequence: ~59% of a batched search is engine ``clone()``/``step()`` and ~13%
is per-leaf Python, all single-threaded, and the GPU only pays for itself at batch sizes a
single tree cannot produce (BATCH_NOTES §6).

This module keeps ``Tournament`` **byte-identical** (it is not edited, and it is still the
class every existing test exercises) and adds:

``drive_many``
    exactly ``runner._drive_to_next_nemesis``'s state machine, turned inside out: instead
    of a ``while True`` loop per agent, one loop over *all* the agents that still need an
    action, with the decisions taken by a caller-supplied ``decide_many``.  With the
    default ``decide_many`` (a plain list comprehension over ``player.act``) it produces
    exactly the same actions, in the same per-agent order, with the same no-progress
    guard — the ordering across agents changes and nothing else, and nothing in a
    ``BalatroGame`` or a ``Player`` is shared between agents.  ``mp/agent`` supplies the
    interesting ``decide_many``: one that collects every agent's MCTS leaf and evaluates
    them in a single forward pass.

``ParallelTournament``
    ``Tournament`` with ``run()`` overridden to talk to a **driver** instead of to games it
    owns.  Everything that has to be decided across agents — the N x N matrix, the life
    rule, who is eliminated, when the ante advances — stays here, in one process, in the
    same order ``Tournament.run`` does it.  The driver only ever gets told "drive these
    agents to their next Nemesis", "these agents lose a life", "cash these out".

``LocalDriver``
    a driver that owns the games in *this* process.  It is what makes the refactor
    testable without any multiprocessing at all: ``ParallelTournament(driver=LocalDriver(...))``
    must reproduce ``Tournament.run`` exactly (``tests/test_parallel_runner.py``).

The multiprocess driver lives in ``mp/agent/parallel/pool.py`` (it needs torch and the
agent layer; this module must keep importing with neither).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Sequence, runtime_checkable

from .bootstrap import State, MLB_PVP_START_ROUND
from .matrix import AnteMatrix, write_run
from .runner import (
    NOOP_BUDGET_DEFAULT, OP_LOSE_LIFE, Tournament, TournamentResult,
    _cash_out, _force_progress_action, clone_games, construct_games,
)

__all__ = [
    "AgentDrive", "drive_many", "serial_decide",
    "DriveOutcome", "AgentSummary", "TournamentDriver", "LocalDriver",
    "ParallelTournament", "JokerRef", "GameSummary",
]


# ═══════════════════════════════════════════════════════════ the per-agent state machine

@dataclass
class AgentDrive:
    """One agent's slice of ``runner._drive_to_next_nemesis``.

    Field for field the same bookkeeping the serial driver keeps in locals: the step
    count, the ``state_signature()`` of the last state seen, the consecutive-no-op counter
    and how many times the guard had to force progress.  ``classify()`` is the serial
    loop's two ``if``s at the top; ``apply()`` is everything after ``player.act``.
    """

    idx: int
    game: Any
    player: Any
    noop_budget: int = NOOP_BUDGET_DEFAULT
    max_steps: int = 20_000
    on_step: Optional[Callable] = None       # on_step(idx, game, action)
    steps: int = 0
    forced: int = 0
    _sig: Any = None
    _noops: int = 0

    def __post_init__(self):
        self._sig = self.game.state_signature() if self.noop_budget else None

    # -- the two stop conditions, checked BEFORE acting (serial loop, top) ------------

    def classify(self) -> Optional[str]:
        s = self.game.state
        if s == State.GAME_OVER:
            return "dead"
        if s == State.ROUND_EVAL and self.game.current_blind.is_pvp:
            return "at_nemesis"
        return None

    # -- one action + the no-progress guard (serial loop, bottom) ---------------------

    def apply(self, action: dict) -> None:
        game = self.game
        game.step(action)
        if self.on_step is not None:
            self.on_step(self.idx, game, action)      # AFTER the step: REPLAY_NOTES §2.3
        self.steps += 1
        if self.noop_budget:
            new_sig = game.state_signature()
            if new_sig == self._sig:
                self._noops += 1
                if self._noops >= self.noop_budget:
                    forced = _force_progress_action(game)
                    if forced is None:
                        raise RuntimeError(
                            f"tournament agent made no progress in state {game.state.name} "
                            f"for {self._noops} steps and no forced action exists")
                    game.step(forced)
                    if self.on_step is not None:
                        self.on_step(self.idx, game, forced)
                    self.steps += 1
                    new_sig = game.state_signature()
                    self.forced += 1
                    self._noops = 0
            else:
                self._noops = 0
            self._sig = new_sig
        if self.steps > self.max_steps:
            raise RuntimeError(
                f"tournament agent wedged in state {game.state.name} after {self.steps} steps")


def serial_decide(items: Sequence[tuple]) -> list:
    """The reference ``decide_many``: ask each player on its own, in order.

    ``items`` is a list of ``(idx, game, player)``.  This is what makes
    ``ParallelTournament`` + ``LocalDriver`` byte-identical to ``Tournament`` — the
    per-agent action sequence is untouched; only the interleaving changes.
    """
    return [player.act(game) for _idx, game, player in items]


def drive_many(drives: Sequence[AgentDrive], decide_many: Callable = serial_decide) -> dict:
    """Drive every agent in ``drives`` to its next Nemesis (or death), in lockstep.

    Returns ``{idx: (status, steps, forced)}`` with ``status`` in ``{"at_nemesis", "dead"}``
    — the same two the serial driver returns.  An agent that is ALREADY at a just-resolved
    Nemesis's ``ROUND_EVAL`` returns immediately without progress, exactly as the serial
    driver does (and for the same reason: cash it out first).
    """
    out: dict = {}
    pending = list(drives)
    while pending:
        need: list = []
        for d in pending:
            status = d.classify()
            if status is not None:
                out[d.idx] = (status, d.steps, d.forced)
            else:
                need.append(d)
        if not need:
            break
        actions = decide_many([(d.idx, d.game, d.player) for d in need])
        for d, action in zip(need, actions):
            d.apply(action)
        pending = need
    return out


# ═══════════════════════════════════════════════════════════════════ the driver seam

@dataclass
class DriveOutcome:
    """What ``TournamentDriver.drive`` reports back for one agent."""
    status: str                 # "at_nemesis" | "dead" | "crashed"
    steps: int = 0
    forced: int = 0
    chips: float = 0.0          # game.chips_scored at the Nemesis
    lives: int = 0
    ante: int = 0


@dataclass
class AgentSummary:
    """What ``TournamentDriver.summarize`` reports for one agent at the end of a run."""
    lives: int = 0
    ante: int = 0
    jokers: tuple = ()
    chips: float = 0.0
    alive: bool = False


@runtime_checkable
class TournamentDriver(Protocol):
    """Everything ``ParallelTournament`` needs from whatever owns the N games.

    Deliberately coarse: one call per ante for the driving, one per ante for the
    cross-agent mutations.  A driver may live in this process (``LocalDriver``) or hold
    the games in N worker processes (``mp/agent/parallel/pool.py::WorkerPool``); the
    tournament cannot tell the difference and neither can its result.
    """

    def setup(self, seed, n_agents: int, deck_key: str, stake, lives: int,
              ruleset: str, fanout: str) -> str:
        """Build the N games, reset the players, arm any loggers.  Returns ``seed_str``."""

    def drive(self, indices: Sequence[int], max_steps: int, noop_budget: int) -> dict:
        """``{idx: DriveOutcome}`` for every index in ``indices``."""

    def apply(self, ops: Sequence[tuple]) -> dict:
        """Perform cross-agent mutations; returns ``{idx: {"lives": int, ...}}``.

        Ops: ``("lose_life", i)``, ``("game_over", i)``, ``("cash_out", i)``,
        ``("done", i, reason)``.
        """

    def summarize(self) -> dict:
        """``{idx: AgentSummary}`` for every agent."""

    def close(self) -> None:
        """Release whatever the driver holds for this run (loggers, games)."""


class JokerRef:
    """A ``.key``-only stand-in for a joker, so ``selfplay._joker_keys`` works on a
    summary the way it works on a real game."""

    __slots__ = ("key",)

    def __init__(self, key: str):
        self.key = key

    def __repr__(self) -> str:                       # pragma: no cover - debugging aid
        return f"JokerRef({self.key!r})"


class GameSummary:
    """What ``ParallelTournament`` exposes as ``_last_games[i]`` when the real game lives
    in another process: enough of a ``BalatroGame`` for the metrics
    (``selfplay._joker_keys`` reads ``.jokers[*].key``, the generation metrics read
    ``.ante``), and nothing else — reading anything more off it should fail loudly rather
    than silently report a default."""

    __slots__ = ("ante", "jokers", "lives", "chips_scored")

    def __init__(self, summary: AgentSummary):
        self.ante = summary.ante
        self.jokers = [JokerRef(k) for k in summary.jokers]
        self.lives = summary.lives
        self.chips_scored = summary.chips


# ═══════════════════════════════════════════════════════════════════ in-process driver

class LocalDriver:
    """A ``TournamentDriver`` that owns its games right here.

    Three callers, one class:

    * ``tests/test_parallel_runner.py`` — owning all N agents, it proves the refactor is a
      refactor (``ParallelTournament`` + ``LocalDriver`` == ``Tournament``, byte for byte).
    * ``mp/agent``'s ``--workers 0`` path — same thing, but with the lockstep
      ``decide_many`` so one process still batches its trees.
    * **each worker process** — owning the SUBSET of agents it was assigned
      (``indices=[3, 7, 11]``).  Global agent indices are used throughout, so a worker's
      ops, outcomes and summaries slot straight into the main process's bookkeeping.

    ``decide_many`` defaults to ``serial_decide``; pass ``mp/agent``'s
    ``parallel.lockstep.LockstepDecider`` to batch the agents' MCTS leaves into one
    forward pass.
    """

    def __init__(self, players: Sequence, decide_many: Callable = serial_decide,
                 on_step: Optional[Callable] = None, on_agent_done: Optional[Callable] = None,
                 on_fanout: Optional[Callable] = None, indices: Optional[Sequence[int]] = None):
        players = list(players)
        self.indices = list(range(len(players))) if indices is None else list(indices)
        if len(self.indices) != len(players):
            raise ValueError(f"{len(players)} players for {len(self.indices)} indices")
        self._players = dict(zip(self.indices, players))
        self.decide_many = decide_many
        self.on_step = on_step
        self.on_agent_done = on_agent_done
        self.on_fanout = on_fanout
        self._games: dict = {}
        self._done: set = set()

    @property
    def players(self) -> list:
        return [self._players[i] for i in self.indices]

    @property
    def games(self) -> list:
        return [self._games[i] for i in self.indices]

    def game(self, idx: int):
        """The live ``BalatroGame`` for one GLOBAL agent index, or ``None`` if this
        driver does not own it (or has not fanned out yet)."""
        return self._games.get(int(idx))

    # -- protocol ---------------------------------------------------------------------

    def setup(self, seed, n_agents: int, deck_key: str, stake, lives: int,
              ruleset: str = "mlb", fanout: str = "clone") -> str:
        """``n_agents`` is the size of the WHOLE population; this driver builds only the
        ``len(self.indices)`` games it owns.  Every game on a seed starts identical (the
        two fan-out methods are pinned equal by ``tests/test_fanout.py``), so a worker
        building 4 of 16 is building the same 4 games the serial run would have."""
        for p in self._players.values():
            reset = getattr(p, "reset", None)
            if reset is not None:
                reset()
        fn = construct_games if fanout == "construct" else clone_games
        games, seed_str = fn(seed, len(self.indices), deck_key, stake, lives, ruleset)
        self._games = dict(zip(self.indices, games))
        self._done = set()
        if self.on_fanout is not None:
            self.on_fanout(self._games, seed_str)
        return seed_str

    def drive(self, indices: Sequence[int], max_steps: int, noop_budget: int) -> dict:
        mine = [i for i in indices if i in self._games]
        drives = [AgentDrive(idx=i, game=self._games[i], player=self._players[i],
                             noop_budget=noop_budget, max_steps=max_steps,
                             on_step=self.on_step)
                  for i in mine]
        raw = drive_many(drives, self.decide_many)
        out = {}
        for i in mine:
            status, steps, forced = raw[i]
            g = self._games[i]
            out[i] = DriveOutcome(status=status, steps=steps, forced=forced,
                                  chips=float(getattr(g, "chips_scored", 0.0)),
                                  lives=int(getattr(g, "lives", 0)),
                                  ante=int(getattr(g, "ante", 0)))
        return out

    def apply(self, ops: Sequence[tuple]) -> dict:
        out: dict = {}
        for op in ops:
            kind, i = op[0], op[1]
            if i not in self._games:
                continue
            g = self._games[i]
            if kind == "lose_life":
                g.lose_life()
                if self.on_step is not None:
                    # The one life lost without a step() -- REPLAY_NOTES.md §2.3.
                    self.on_step(i, g, {"type": OP_LOSE_LIFE})
            elif kind == "game_over":
                g.state = State.GAME_OVER
            elif kind == "cash_out":
                _cash_out(g, (lambda game, action, _i=i: self.on_step(_i, game, action))
                          if self.on_step is not None else None)
            elif kind == "done":
                self._finish(i, op[2])
            else:                                    # pragma: no cover - programmer error
                raise ValueError(f"unknown driver op {kind!r}")
            out[i] = {"lives": int(getattr(g, "lives", 0)),
                      "ante": int(getattr(g, "ante", 0)),
                      "chips": float(getattr(g, "chips_scored", 0.0))}
        return out

    def summarize(self) -> dict:
        out = {}
        for i, g in self._games.items():
            out[i] = AgentSummary(
                lives=int(getattr(g, "lives", 0)), ante=int(getattr(g, "ante", 0)),
                jokers=tuple(getattr(j, "key", "?") for j in getattr(g, "jokers", [])),
                chips=float(getattr(g, "chips_scored", 0.0)),
                alive=(g.state != State.GAME_OVER))
        return out

    def close(self) -> None:
        pass

    # -- internals --------------------------------------------------------------------

    def _finish(self, i: int, reason: str) -> None:
        if i in self._done:
            return
        self._done.add(i)
        if self.on_agent_done is not None:
            self.on_agent_done(i, self._games[i], reason)


# ═══════════════════════════════════════════════════════════════════ the tournament

class ParallelTournament(Tournament):
    """``Tournament`` whose N games live behind a :class:`TournamentDriver`.

    ``run()`` is a line-for-line transliteration of ``Tournament.run``: the same ante
    lockstep, the same ``_decide_losers`` (inherited, not copied), the same
    ``AnteMatrix.build``, the same order of "take the logger's final signature BEFORE the
    out-of-band GAME_OVER".  The only difference is that every mutation of a game goes
    through the driver, so the games may be anywhere.

    A driver that reports ``status="crashed"`` for an agent (a worker process died) is
    handled exactly like a death: the agent leaves the tournament, its last known lives
    are recorded, and the run continues with everybody else.  ``crashed`` is counted
    separately in ``TournamentResult.crashed`` so it can never be mistaken for a real
    game over.
    """

    def __init__(self, *args, driver: TournamentDriver, **kwargs):
        super().__init__(*args, **kwargs)
        self.driver = driver

    def run(self) -> TournamentResult:
        t0 = time.perf_counter()
        seed_str = self.driver.setup(self.seed, self.n_agents, self.deck_key, self.stake,
                                     self.lives, "mlb", self.fanout)
        alive = list(range(self.n_agents))
        final_lives: list = [None] * self.n_agents
        last_score: dict = {}
        ante_matrices: list = []
        steps_total = 0
        forced_progress = [0] * self.n_agents
        crashed: list = []
        done: set = set()

        nemesis_ante = MLB_PVP_START_ROUND
        while alive and nemesis_ante <= self.max_ante:
            outcomes = self.driver.drive(list(alive), self.max_steps_per_drive,
                                         self.noop_budget)
            dead_this_round = []
            ops: list = []
            for i in list(alive):
                o = outcomes[i]
                steps_total += o.steps
                forced_progress[i] += o.forced
                if o.status != "at_nemesis":
                    dead_this_round.append(i)
                    final_lives[i] = o.lives
                    if o.status == "crashed":
                        crashed.append(i)
                    if i not in done:
                        done.add(i)
                        ops.append(("done", i, "died" if o.status == "dead" else "crashed"))
            alive = [i for i in alive if i not in dead_this_round]
            if ops:
                self.driver.apply(ops)
            if not alive:
                break

            scores = {i: outcomes[i].chips for i in alive}
            for i, s in scores.items():
                last_score[i] = (nemesis_ante, s)
            round_idx = nemesis_ante - MLB_PVP_START_ROUND
            losers = self._decide_losers(alive, scores, round_idx)
            ante_matrices.append(
                AnteMatrix.build(nemesis_ante, self.n_agents, scores, losers=losers))

            lives_after = self.driver.apply([("lose_life", i) for i in sorted(losers)])
            newly_dead = []
            post: list = []
            for i in sorted(losers):
                if lives_after[i]["lives"] <= 0:
                    # Take the logger's final signature BEFORE the out-of-band GAME_OVER.
                    if i not in done:
                        done.add(i)
                        post.append(("done", i, "eliminated"))
                    post.append(("game_over", i))
                    newly_dead.append(i)
                    final_lives[i] = lives_after[i]["lives"]
            alive = [i for i in alive if i not in newly_dead]
            post.extend(("cash_out", i) for i in alive)
            if post:
                self.driver.apply(post)
            nemesis_ante += 1

        summaries = self.driver.summarize()
        for i in alive:
            final_lives[i] = summaries[i].lives
        tail = [("done", i, "finished") for i in range(self.n_agents) if i not in done]
        done.update(i for _, i, _ in tail)
        if tail:
            self.driver.apply(tail)

        result = TournamentResult(
            seed=seed_str, n_agents=self.n_agents, life_rule=self.life_rule,
            max_ante=self.max_ante, deck_key=self.deck_key, stake=self.stake,
            ante_matrices=ante_matrices, final_lives=final_lives, alive_at_end=list(alive),
            last_score=last_score, fanout_method=self.fanout,
            wall_clock_s=time.perf_counter() - t0, steps_total=steps_total,
            forced_progress=forced_progress,
        )
        # Not a TournamentResult field (the dataclass is W2's and stays as it is): attached
        # so a caller can see a worker crash without parsing logs.  `[]` on a clean run.
        result.crashed = sorted(crashed)
        self._last_games = [GameSummary(summaries[i]) for i in range(self.n_agents)]
        self._summaries = summaries
        if self.out_dir is not None:
            write_run(self.out_dir, result)
        return result
