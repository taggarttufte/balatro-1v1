"""
runner.py — N-agent same-seed tournament: N independent ``BalatroGame(ruleset="mlb")``
instances, one per agent, stepped by that agent's ``Player`` in ante lockstep.

Key engine fact this whole module leans on (MLB_NOTES.md §2 items 1.2e/1.3f, game.py
``lose_life`` / ``end_pvp`` / ``pvp_solo``): a ``BalatroGame`` with the default
``pvp_solo=True`` (i.e. NOT attached to an ``MLBMatch``) plays its Nemesis blind entirely on
its own — ``play_blind`` starts it immediately (no ``pvp_ready`` handshake), and the moment
the agent is out of hands the game auto-calls ``end_pvp()`` and lands in ``ROUND_EVAL`` with
``chips_scored`` intact and NO life lost.  That is exactly decision 0.2's "every agent plays
its Nemesis blind blind, to exhaustion": nobody's game ever needs to see anybody else's score
mid-hand, because MLB pays no unused-hand money at a PvP blind and the round never ends early
on chips reached.  So N agents can be driven completely independently, one at a time, through
an entire ante (Small, Big, and the Nemesis's own exhaustion) with no coordination object at
all — the ONLY point where agents need to be brought together is *after* every one of them has
finished this ante's Nemesis, to build the N x N matrix from their final scores and decide
lives (decision 0.3, ``life_rule``).  That synchronisation point is this module's "ante
lockstep": drive every alive agent to its Nemesis conclusion, snapshot scores, apply the life
rule (calling the same public ``game.lose_life()`` hook ``MLBMatch._resolve_pvp`` uses),
drop anyone who hits 0 lives, cash out the survivors, and repeat for the next ante.

See TOURNAMENT_NOTES.md for the full contract, its known gap, and the fan-out benchmark.

Phase 4 (2026-08-22, W2): ``_repair_mlb_gameover_bug`` -- the Phase 3 workaround for
``game.py``'s bl_hook / bl_eye / bl_mouth rejection branches setting GAME_OVER on hand
exhaustion regardless of ``self.mlb`` -- is GONE.  The lead fixed the engine at the Phase 3
close (those branches now route through ``_mlb_fail_round()``; engine regression test
``TestBossRejectionRespectsMLB``), so the repair was dead code that could only ever mask a
future engine bug.  ``tests/test_boss_rejection_life.py`` is the runner-level replacement:
a Hook-rejected exhaustion costs the agent one life and the run continues.
"""
from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .bootstrap import BalatroGame, State, MLB_STARTING_LIVES, MLB_PVP_START_ROUND
from .matrix import AnteMatrix, write_run

__all__ = [
    "Tournament", "TournamentResult", "construct_games", "clone_games",
    "benchmark_fanout", "FANOUT_DEFAULT", "NONE_RULE_LIVES_SENTINEL",
    "NOOP_BUDGET_DEFAULT", "OP_LOSE_LIFE",
]

# The synthetic op a trajectory log records when this module mutates a game OUTSIDE
# ``step()`` -- the cross-agent life rule below is the only such place.  Value must match
# ``replay/_util.py::OP_LOSE_LIFE`` (REPLAY_NOTES.md §2.3); duplicated as a literal rather
# than imported so ``tournament`` keeps no dependency on ``replay``, and pinned by
# ``tests/test_trajectory_hook.py``.
OP_LOSE_LIFE = "__lose_life__"

# Set from benchmark_fanout(seed="7I4M53DL", n=100) measured on this machine (see
# TOURNAMENT_NOTES.md "fan-out benchmark" for the numbers) -- clone() beat N constructions,
# so it is the default; "construct" / "clone" remain selectable and both are tested for
# identical state_signature() output.
FANOUT_DEFAULT = "clone"

# life_rule="none": "nobody dies, run to a fixed ante" (brief §0.3).  A regular (non-Nemesis)
# blind loss under MLB costs a life independently of life_rule (game.py's own
# `_mlb_fail_round` / `_mlb_check_deck_out`) and would still end an agent's run early if its
# lives hit 0 -- so under "none" every game is given a lives count no realistic scripted or
# random-legal run will exhaust, guaranteeing every agent actually reaches max_ante.  The
# Nemesis-triggered life_rule step itself is simply skipped (see Tournament._decide_losers).
NONE_RULE_LIVES_SENTINEL = 1_000_000_000


# ============================================================================ fan-out

def construct_games(seed, n: int, deck_key: str, stake, lives: int, ruleset: str = "mlb") -> tuple[list, str]:
    """N independent ``BalatroGame`` CONSTRUCTIONS on the same seed.  The first construction
    may normalise/derive ``seed`` (e.g. ``None`` -> a random seed_str); every subsequent
    construction is pinned to that exact string so all N really share one seed."""
    games: list = []
    seed_str: Optional[str] = None
    for _ in range(n):
        g = BalatroGame(seed=(seed if seed_str is None else seed_str), deck_key=deck_key,
                         stake=stake, ruleset=ruleset)
        g.lives = lives
        seed_str = g.seed_str
        games.append(g)
    return games, seed_str


def clone_games(seed, n: int, deck_key: str, stake, lives: int, ruleset: str = "mlb") -> tuple[list, str]:
    """One construction + ``clone()`` N-1 times (``BalatroGame.clone()`` is the MCTS
    snapshot machinery already exercised by the engine suite's clone-fidelity tests)."""
    base = BalatroGame(seed=seed, deck_key=deck_key, stake=stake, ruleset=ruleset)
    base.lives = lives
    games = [base] + [base.clone() for _ in range(n - 1)]
    return games, base.seed_str


def benchmark_fanout(seed: str = "7I4M53DL", n: int = 100, deck_key: str = "b_red",
                      stake=1, lives: int = MLB_STARTING_LIVES, repeats: int = 3) -> dict:
    """Best-of-``repeats`` wall clock for both fan-out methods building N fresh games, plus a
    state_signature() equality check (both methods must produce N identical initial states).
    Used once to pick FANOUT_DEFAULT and recorded in TOURNAMENT_NOTES.md; also exercised by
    ``tests/test_fanout.py``."""
    best = {}
    sigs = {}
    for name, fn in (("construct", construct_games), ("clone", clone_games)):
        best_t = math.inf
        games = None
        for _ in range(repeats):
            t0 = time.perf_counter()
            games, _ = fn(seed, n, deck_key, stake, lives)
            dt = time.perf_counter() - t0
            best_t = min(best_t, dt)
        best[name] = best_t
        sigs[name] = [g.state_signature() for g in games]
    return {"seconds": best, "signatures_equal": sigs["construct"] == sigs["clone"],
            "n": n, "seed": seed}


# ============================================================================ per-agent driving

# How many CONSECUTIVE state-preserving steps an agent may take before the driver forces
# progress.  See `_drive_to_next_nemesis` for why this exists; 0 disables the guard.
NOOP_BUDGET_DEFAULT = 8


def _force_progress_action(game) -> Optional[dict]:
    """One legal action per state that is GUARANTEED to change the state.  Used only by the
    no-progress guard below, and only after the agent has already failed to make progress
    ``noop_budget`` times in a row."""
    s = game.state
    if s == State.SHOP:
        return {"type": "leave_shop"}
    if s == State.BOOSTER_OPEN:
        return {"type": "skip_booster"}
    if s == State.BLIND_SELECT:
        return {"type": "play_blind"}
    if s == State.ROUND_EVAL:
        return {"type": "advance"}
    if s == State.SELECTING_HAND and game.hand:
        return {"type": "play", "cards": [0]}     # always spends a hand
    return None


def _drive_to_next_nemesis(game, player, max_steps: int = 20_000,
                            noop_budget: int = NOOP_BUDGET_DEFAULT, on_step=None,
                            forced: Optional[list] = None) -> tuple:
    """Step ``game`` via ``player.act`` until it is exhausted at a Nemesis (``ROUND_EVAL``
    with ``current_blind.is_pvp`` -- ``pvp_solo`` already auto-resolved it, no life lost yet)
    or the game ends (``GAME_OVER``, from a regular Small/Big/ante-1-Boss blind loss at 0
    lives).  Must never be called while ``game.state`` is ALREADY a just-resolved Nemesis's
    ``ROUND_EVAL`` -- cash that one out first (``_cash_out``) or this returns immediately
    without making progress.  Returns ``(status, n_steps)``.

    **No-progress guard** (Phase 4 W2, works around a frozen-engine gap -- see
    TRAIN_NOTES.md "needs engine change"): ``game.py``'s SHOP branch of ``legal_actions()``
    (:1433) enumerates card-targeting ``use_consumable`` actions against ``self.hand``,
    which in the SHOP still holds the PREVIOUS blind's cards, and ``_use_consumable``
    (:1854) silently no-ops when the application fails (``success = False``) instead of
    consuming the consumable or rejecting the action.  The result is a legal action that
    leaves the game bit-identical -- an infinite loop for any agent that likes it.  Observed
    with an MCTS population: one agent burned all 20 000 steps of ``max_steps_per_drive`` in
    a single shop (and, under a different noise seed, 14 000 training samples and 55 s of a
    60 s generation).  Scripted / random-legal players had never hit it, which is why Phase 3
    did not see it.

    So: ``state_signature()`` before and after each step (42 us, ~0.1% of a search-driven
    decision), and after ``noop_budget`` CONSECUTIVE steps that changed nothing, the driver
    plays ``_force_progress_action`` itself.  Zero effect on any agent that ever changes the
    state -- verified against the existing determinism / smoke suites, whose results are
    unchanged.  ``noop_budget=0`` disables it.

    How many times it fired is accumulated into the caller's ``forced`` list (a one-element
    counter) and surfaced as ``TournamentResult.forced_progress``.  It is deliberately NOT
    stored on the game: ``state_signature()`` sweeps up EVERY int/float/str/bool attribute of
    the game object (game.py:923), so attaching a diagnostic counter to it silently changes
    the run's signature and breaks trajectory replay.  Learned the hard way -- see
    ``agent/TRAIN_NOTES.md`` "Found, not fixed".
    """
    n = 0
    sig = game.state_signature() if noop_budget else None
    noops = 0
    while True:
        s = game.state
        if s == State.GAME_OVER:
            return "dead", n
        if s == State.ROUND_EVAL and game.current_blind.is_pvp:
            return "at_nemesis", n
        a = player.act(game)
        game.step(a)
        if on_step is not None:
            on_step(game, a)          # AFTER the step: REPLAY_NOTES.md §2.3
        n += 1
        if noop_budget:
            new_sig = game.state_signature()
            if new_sig == sig:
                noops += 1
                if noops >= noop_budget:
                    action = _force_progress_action(game)
                    if action is None:
                        raise RuntimeError(
                            f"tournament agent made no progress in state {game.state.name} "
                            f"for {noops} steps and no forced action exists")
                    game.step(action)
                    if on_step is not None:
                        on_step(game, action)
                    n += 1
                    new_sig = game.state_signature()
                    if forced is not None:
                        forced[0] += 1
                    noops = 0
            else:
                noops = 0
            sig = new_sig
        if n > max_steps:
            raise RuntimeError(f"tournament agent wedged in state {s.name} after {n} steps")


def _cash_out(game, on_step=None) -> None:
    """Force the single ``advance`` transition off a just-decided Nemesis's ``ROUND_EVAL``
    (pays the $5 blind reward + comeback money, ante += 1 next boss drawn, lands in SHOP).
    ``BalatroGame.step`` ignores the action's content in ``ROUND_EVAL`` (any dict advances
    it), matching ``legal_actions()``'s single dummy ``{"type": "advance"}``."""
    assert game.state == State.ROUND_EVAL
    game.step({"type": "advance"})
    if on_step is not None:
        on_step(game, {"type": "advance"})


def _pairing(n_agents: int) -> tuple[list, list]:
    """Fixed assignment for ``life_rule="paired"`` (brief §0.3): consecutive pairs
    (0,1),(2,3),...; with odd ``n_agents`` the last agent has no fixed partner and instead
    gets a ROTATING opponent, one different agent per Nemesis round, cycling through every
    other agent in index order (see ``Tournament._decide_losers``).  Note the rotating
    partner's own fixed pairing still runs that round too -- that agent can lose a life from
    either comparison in the same round; documented in TOURNAMENT_NOTES.md, not hidden."""
    pairs = [(i, i + 1) for i in range(0, n_agents - (n_agents % 2), 2)]
    singles = [n_agents - 1] if n_agents % 2 else []
    return pairs, singles


@dataclass
class TournamentResult:
    seed: str
    n_agents: int
    life_rule: str
    max_ante: int
    deck_key: str
    stake: object
    ante_matrices: list = field(default_factory=list)     # list[AnteMatrix], one per Nemesis played
    final_lives: list = field(default_factory=list)       # (n_agents,) lives at the end (0 if dead)
    alive_at_end: list = field(default_factory=list)       # agent indices never eliminated
    last_score: dict = field(default_factory=dict)         # agent_idx -> (ante, score) at its last Nemesis
    fanout_method: str = ""
    wall_clock_s: float = 0.0
    steps_total: int = 0
    #: Per agent, how many times the no-progress guard had to play an action for it.
    #: Non-zero means that agent hit the frozen-engine SHOP gap (see
    #: ``_drive_to_next_nemesis``); it is a diagnostic, not part of the outcome.
    forced_progress: list = field(default_factory=list)

    def summary_rows(self) -> list:
        """One dict per ante: {ante, n_present, mean, std, quantiles, tie_fraction, losers}."""
        return [
            {"ante": m.ante, **m.stats, "tie_fraction": m.tie_fraction, "losers": m.losers}
            for m in self.ante_matrices
        ]


class Tournament:
    """N independent MLB games on one seed, in ante lockstep (see module docstring)."""

    def __init__(self, seed, n_agents: int, players: list, deck_key: str = "b_red",
                 stake=1, life_rule: str = "paired", max_ante: int = 8,
                 lives: int = MLB_STARTING_LIVES, fanout: str = "auto",
                 max_steps_per_drive: int = 20_000, out_dir: Optional[str] = None,
                 noop_budget: int = NOOP_BUDGET_DEFAULT, on_fanout=None,
                 on_step=None, on_agent_done=None):
        if len(players) != n_agents:
            raise ValueError(f"len(players)={len(players)} != n_agents={n_agents}")
        if life_rule not in ("paired", "median", "none"):
            raise ValueError(f"life_rule must be 'paired'/'median'/'none', got {life_rule!r}")
        if max_ante < MLB_PVP_START_ROUND:
            raise ValueError(
                f"max_ante={max_ante} < MLB_PVP_START_ROUND={MLB_PVP_START_ROUND}: no "
                "Nemesis would ever be played, so there is nothing to measure")
        self.seed = seed
        self.n_agents = n_agents
        self.players = list(players)
        self.deck_key = deck_key
        self.stake = stake
        self.life_rule = life_rule
        self.max_ante = max_ante
        self.lives = NONE_RULE_LIVES_SENTINEL if life_rule == "none" else lives
        self.fanout = FANOUT_DEFAULT if fanout == "auto" else fanout
        if self.fanout not in ("construct", "clone"):
            raise ValueError(f"fanout must be 'construct'/'clone'/'auto', got {fanout!r}")
        self.max_steps_per_drive = max_steps_per_drive
        self.noop_budget = int(noop_budget)
        # Called once per run() as ``on_fanout(games, seed_str)``, immediately after the N
        # games exist and before anybody acts.  Added in Phase 4 W2 for W3's
        # ``replay.TrajectoryLogger``, whose ``begin(game, meta)`` needs the actual
        # ``BalatroGame`` -- which nothing outside this class used to be able to see until
        # ``run()`` had already finished.  Never mutate the games from here.
        self.on_fanout = on_fanout
        # Trajectory-logging hooks (Phase 4 W2, for W3's ``replay``; REPLAY_NOTES §2.3).
        # ``on_step(agent_idx, game, action)`` fires after EVERY ``game.step()`` this module
        # performs -- the agent's own actions, the no-progress guard's forced action, and
        # ``_cash_out``'s advance -- plus a synthetic ``{"type": OP_LOSE_LIFE}`` after the
        # cross-agent life rule, which is the ONE place a life is lost without a step.
        # ``on_agent_done(agent_idx, game, reason)`` fires exactly once per agent, BEFORE
        # this module force-sets ``State.GAME_OVER`` on an eliminated agent -- that
        # assignment is an out-of-band mutation no replay can reproduce, so a logger has to
        # take its final signature first.  ``reason``: "died" | "eliminated" | "finished".
        self.on_step = on_step
        self.on_agent_done = on_agent_done
        self.out_dir = Path(out_dir) if out_dir is not None else None
        self._pairs, self._singles = _pairing(n_agents)

    # -- life rule ------------------------------------------------------------------

    def _decide_losers(self, alive: list, scores: dict, round_idx: int) -> set:
        rule = self.life_rule
        if rule == "none":
            return set()
        if rule == "median":
            vals = sorted(scores.values())
            med = statistics.median(vals)
            return {i for i in alive if scores[i] < med}
        if rule == "paired":
            losers: set = set()
            for i, j in self._pairs:
                if i in scores and j in scores:
                    if scores[i] < scores[j]:
                        losers.add(i)
                    elif scores[j] < scores[i]:
                        losers.add(j)
            if self._singles:
                odd = self._singles[0]
                others = [k for k in range(self.n_agents - 1)]
                if odd in scores and others:
                    partner = others[round_idx % len(others)]
                    if partner in scores:
                        if scores[odd] < scores[partner]:
                            losers.add(odd)
                        elif scores[partner] < scores[odd]:
                            losers.add(partner)
            return losers
        raise ValueError(rule)   # unreachable, validated in __init__

    # -- fan-out ----------------------------------------------------------------------

    # -- trajectory hooks -------------------------------------------------------------

    def _step_cb(self, i: int):
        if self.on_step is None:
            return None
        return lambda game, action, _i=i: self.on_step(_i, game, action)

    def _finish(self, done: set, i: int, game, reason: str) -> None:
        if self.on_agent_done is None or i in done:
            done.add(i)
            return
        done.add(i)
        self.on_agent_done(i, game, reason)

    # -- fan-out ----------------------------------------------------------------------

    def _fanout(self) -> tuple[list, str]:
        fn = construct_games if self.fanout == "construct" else clone_games
        return fn(self.seed, self.n_agents, self.deck_key, self.stake, self.lives)

    # -- run ----------------------------------------------------------------------------

    def run(self) -> TournamentResult:
        t0 = time.perf_counter()
        for p in self.players:
            reset = getattr(p, "reset", None)
            if reset is not None:
                reset()
        games, seed_str = self._fanout()
        if self.on_fanout is not None:
            self.on_fanout(games, seed_str)
        alive = list(range(self.n_agents))
        final_lives = [None] * self.n_agents
        last_score: dict = {}
        ante_matrices: list = []
        steps_total = 0
        forced_progress = [0] * self.n_agents
        done: set = set()

        nemesis_ante = MLB_PVP_START_ROUND
        while alive and nemesis_ante <= self.max_ante:
            dead_this_round = []
            for i in list(alive):
                counter = [0]
                outcome, n_steps = _drive_to_next_nemesis(
                    games[i], self.players[i], self.max_steps_per_drive, self.noop_budget,
                    self._step_cb(i), counter)
                steps_total += n_steps
                forced_progress[i] += counter[0]
                if outcome == "dead":
                    dead_this_round.append(i)
                    final_lives[i] = games[i].lives
                    self._finish(done, i, games[i], "died")
            alive = [i for i in alive if i not in dead_this_round]
            if not alive:
                break
            scores = {i: games[i].chips_scored for i in alive}
            for i, s in scores.items():
                last_score[i] = (nemesis_ante, s)
            round_idx = nemesis_ante - MLB_PVP_START_ROUND
            losers = self._decide_losers(alive, scores, round_idx)
            am = AnteMatrix.build(nemesis_ante, self.n_agents, scores, losers=losers)
            ante_matrices.append(am)

            newly_dead = []
            for i in losers:
                games[i].lose_life()
                if self.on_step is not None:
                    # The one life lost without a step() -- REPLAY_NOTES.md §2.3.
                    self.on_step(i, games[i], {"type": OP_LOSE_LIFE})
                if games[i].lives <= 0:
                    # Take the logger's final signature BEFORE the out-of-band GAME_OVER.
                    self._finish(done, i, games[i], "eliminated")
                    games[i].state = State.GAME_OVER
                    newly_dead.append(i)
                    final_lives[i] = games[i].lives
            alive = [i for i in alive if i not in newly_dead]

            for i in alive:
                _cash_out(games[i], self._step_cb(i))
            nemesis_ante += 1

        for i in alive:
            final_lives[i] = games[i].lives
        for i in range(self.n_agents):
            self._finish(done, i, games[i], "finished")

        result = TournamentResult(
            seed=seed_str, n_agents=self.n_agents, life_rule=self.life_rule,
            max_ante=self.max_ante, deck_key=self.deck_key, stake=self.stake,
            ante_matrices=ante_matrices, final_lives=final_lives, alive_at_end=list(alive),
            last_score=last_score, fanout_method=self.fanout,
            wall_clock_s=time.perf_counter() - t0, steps_total=steps_total,
            forced_progress=forced_progress,
        )
        self._last_games = games   # kept for tests/inspection (state_signature() etc.); not part
        # of the public TournamentResult contract, mirroring how the engine itself exposes
        # underscore-prefixed fields (`_hands_played_round`) for harness use.
        if self.out_dir is not None:
            write_run(self.out_dir, result)
        return result
