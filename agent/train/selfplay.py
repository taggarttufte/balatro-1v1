"""
selfplay.py — turning games into training samples under an objective that is not degenerate.

Phase 4 W2. The overnight shakedown (CAMPAIGN_LOG 2026-08-22 07:35) proved the pipeline
learns and that the objective it was learning was worthless: under solo MLB the Nemesis is
free (`pvp_solo=True` auto-resolves it, no life lost), so the optimal policy is "skip 15/16
blinds and coast", the value target collapsed to sd 0.07, and the value head had nothing
left to say. This module supplies the two replacements the brief asks for:

    OBJECTIVE 1 (real) — the tournament. N agents on ONE seed; at every Nemesis the N x N
    matrix says where each agent stands. A current-net agent's value target is its
    POPULATION RANK in [0, 1] at the NEXT Nemesis (short horizon, MP_TRAINING_DESIGN §1)
    blended with its final standing in the match. Skipping everything now costs you the
    comparison, because everyone else is on your seed and some of them built something.

    OBJECTIVE 2 (interim, cheap) — solo play against an EXTERNAL per-ante chip target
    (`--objective external`). The driver charges the life the solo engine does not:
    `chips_scored < target(ante)` at the Nemesis -> `game.lose_life()`. Skipping both
    blinds and building nothing now loses a life, which is the whole point.

Both produce `train.Sample`s through ONE function, `make_sample()` — that is the seam W1's
set-based encoder plugs into (see §"W1 seam" below), and it is the only place in this
workstream that knows what an observation looks like.

How samples are collected without owning the play loop
------------------------------------------------------
`tournament/runner.py` drives the games; the players are handed to it. So collection rides
on `mcts.MCTSPlayer.record_hook` (added by W2, additive, `None` by default and free when
unset): every decision hands back a `mcts.Decision` (live game, legal actions, root visit
counts, chosen key) and `SampleCollector` turns it into a Sample with a placeholder value.
Values are filled in afterwards from the tournament's own N x N matrices — the same objects
`matrix.py` already builds, no second source of truth.

Value target, precisely
-----------------------
For a decision made by current-net agent `i` while `game.ante == a`:

    rank_next  = normalised population rank of agent i at the first Nemesis ante >= a
                 (1.0 = highest score in the population, 0.0 = lowest, ties averaged;
                 0.0 if agent i was already eliminated and is absent from that matrix;
                 the final standing if there is no later Nemesis)
    outcome    = normalised final standing of agent i over the whole match, ranking agents
                 by (Nemesis rounds survived, final lives, last Nemesis score)
    z          = value_blend * rank_next + (1 - value_blend) * outcome

`value_blend` defaults to **0.7** (`--value-blend`). Rationale: the short-horizon term is
the one with real variance and a short credit-assignment path — it is a per-ante comparison
against 15 other runs of the same seed — while the match term is what stops the agent from
trading the whole run for one good ante. 0.7/0.3 keeps the target dominated by the signal
that moves per decision while still ordering "won the ante, died at the next one" below
"won the ante, survived". Both terms are dense ranks in [0, 1], so neither can collapse the
way `MLBOutcome` did: with N=16 a uniform rank has sd 0.30, and the blend cannot fall below
the sd of a rank distribution unless the population itself has collapsed (which is exactly
what `tie_fraction` reports).

W1 seam — WIRED (2026-08-22)
---------------------------
`make_sample(game, legal, legal_keys, visits, encoder, z)` is the ONLY sample constructor,
and `SampleCollector.sample_fn` is the one place it is chosen. W1 published the set-encoder
contract in `SETENC_NOTES.md` §0 and built `train.sample.SampleBuilder` with exactly this
signature, so `MLBTrainer` hands `ColdTrainer.sample_builder` straight through: with
`--encoder set` (or plain `--subsample`) every sample is a subsampled `Sample` v2, and with
`--no-subsample --encoder mlb` it falls back to the v1 `make_sample` below, unchanged. The
local function stays because it is the thing tests pin the seam against, and because a
`SampleCollector` built by hand (no ColdTrainer) still has to work.
"""
from __future__ import annotations

import math
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from balatro_sim.game import BalatroGame, State
from balatro_sim.constants import MLB_STARTING_LIVES, blind_base_chips
from mcts.action_features import featurize_actions
from mcts.outcome import ExternalOutcome, MLBOutcome, is_stuck_state

#: Mirrors `tournament/runner.py::OP_LOSE_LIFE` / `replay/_util.py::OP_LOSE_LIFE` — the
#: synthetic op a trajectory log records when a driver takes a life outside `game.step()`.
OP_LOSE_LIFE = "__lose_life__"
#: The solo external-target driver's OTHER out-of-band mutation: `game.set_pvp_info(score,
#: hands)` (the `enemyInfo` relay that shows the agent what it has to beat). `mp/replay`
#: does not know this op yet — see TRAIN_NOTES §9 — so a solo trajectory logged WITH the
#: relay on replays only up to its first Nemesis. `--no-pvp-relay` turns the relay off and
#: makes solo logs fully verifiable; the tournament objective never relays at all
#: (TOURNAMENT_NOTES §2: every agent plays its Nemesis blind), so its logs always verify.
OP_SET_PVP_INFO = "__set_pvp_info__"

from .population import (
    CheckpointHistory, PopulationConfig, SkipCap, build_population, instantiate,
    population_summary, tournament_module,
)
from .trajectory import Sample

__all__ = [
    "make_sample", "SampleCollector", "RecordedDecision",
    "TournamentSpec", "run_tournament_generation", "GenerationSamples",
    "normalized_ranks", "value_targets_from_result",
    "vanilla_boss_target", "big_blind_floor", "load_target_fn", "external_outcome_for",
    "play_solo_external_episode", "SoloEpisodeResult",
    "GenerationMetrics", "tournament_module", "solo_metrics", "assign_value_targets",
    "MLBTrainConfig", "MLBTrainer",
]


# ═══════════════════════════════════════════════════════════════════ the W1 seam

def make_sample(game: BalatroGame, legal: list, legal_keys: list, visits: dict,
                encoder, z: float = 0.0) -> Sample:
    """One decision -> one `train.Sample`. **The only observation-encoding call site in
    this workstream** — swapping in W1's set encoder replaces this body.

    `visits` empty (a forced-action state that skipped the search) -> a point mass on the
    single legal action, matching `train/agent.py::SelfPlayAgent`.

    Note on tree reuse: with `reuse=True` a reused root's visit counts include the
    simulations spent on it in earlier decisions. That is MORE evidence, not stale
    evidence — `clone().step(a)` is deterministic on this engine (Phase 1 dividend, see
    BATCH_NOTES §4), so a retained subtree describes the same state it did last decision.
    The policy target is therefore over accumulated visits, deliberately.
    """
    obs = encoder(game)
    feats = featurize_actions(legal)
    n = len(legal_keys)
    if visits:
        counts = np.array([visits.get(k, 0) for k in legal_keys], dtype=np.float64)
        total = counts.sum()
        target = ((counts / total).astype(np.float32) if total > 0
                  else np.full(n, 1.0 / n, dtype=np.float32))
    else:
        target = np.full(n, 1.0 / n, dtype=np.float32)
    return Sample(obs=obs, action_features=feats, target_policy=target, z=float(z))


# ═══════════════════════════════════════════════════════════════════ collection

@dataclass
class RecordedDecision:
    """A Sample plus the bookkeeping needed to give it a value target later."""
    sample: Sample
    agent_idx: int
    ante: int
    blind_idx: int
    is_pvp: bool
    action_type: str
    skip_offered: bool
    shortcut: bool
    n_legal: int
    #: At a non-PvP `ROUND_EVAL`: True if the blind was CLEARED, False if it was failed
    #: (MLB routes a failed regular blind to `ROUND_EVAL` too, with `chips_scored` short of
    #: `chips_target`). `None` anywhere else. This is the blind-clear rate's only source.
    blind_result: Optional[bool] = None


class SampleCollector:
    """`record_hook` for one agent. Cheap: one encode + one featurize per decision, the
    same work the search already did for the root leaf (not shared — the root evaluation
    happens inside the tree and its arrays are not retained)."""

    def __init__(self, agent_idx: int, encoder, sample_fn: Optional[Callable] = None,
                 max_records: int = 0):
        self.agent_idx = agent_idx
        self.encoder = encoder
        self.sample_fn = sample_fn or make_sample
        self.max_records = int(max_records)     # 0 = unbounded
        self.dropped = 0
        self.records: list[RecordedDecision] = []

    def __call__(self, decision) -> None:
        if self.max_records and len(self.records) >= self.max_records:
            self.dropped += 1
            return
        game = decision.game
        blind = getattr(game, "current_blind", None)
        atype = decision.chosen[0] if decision.chosen else "none"
        # Read off the STATE, not off `decision.legal`: a `legal_filter` (the
        # `--max-skips-per-ante` cap) removes `skip_blind` from the candidate set, and a
        # skip rate whose denominator the cap also shrinks would always read 0/0.
        skip_offered = (game.state is State.BLIND_SELECT
                        and blind is not None
                        and getattr(blind, "kind", "") != "Boss"
                        and not getattr(blind, "is_pvp", False)
                        and not getattr(game, "pvp_ready", False))
        blind_result = None
        if (game.state is State.ROUND_EVAL and blind is not None
                and not getattr(blind, "is_pvp", False)):
            blind_result = int(game.chips_scored) >= int(getattr(blind, "chips_target", 0))
        sample = self.sample_fn(game, decision.legal, decision.legal_keys,
                                decision.visits, self.encoder, 0.0)
        self.records.append(RecordedDecision(
            sample=sample, agent_idx=self.agent_idx, ante=int(game.ante),
            blind_idx=int(getattr(game, "blind_idx", 0)),
            is_pvp=bool(getattr(blind, "is_pvp", False)),
            action_type=str(atype), skip_offered=skip_offered,
            shortcut=bool(decision.shortcut), n_legal=len(decision.legal),
            blind_result=blind_result,
        ))

    def clear(self) -> None:
        self.records = []
        self.dropped = 0


# ═══════════════════════════════════════════════════════════════════ ranks / targets

def normalized_ranks(values: Sequence[float]) -> np.ndarray:
    """[0, 1] rank of each value against the others: 1.0 = largest, 0.0 = smallest, ties
    share the average. `nan` in -> `nan` out (an absent agent has no rank). A single
    present value ranks 0.5 (there is nothing to compare it to; calling that a win or a
    loss would be a lie)."""
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape, np.nan)
    present = np.where(~np.isnan(arr))[0]
    if present.size == 0:
        return out
    if present.size == 1:
        out[present[0]] = 0.5
        return out
    vals = arr[present]
    order = np.argsort(vals, kind="stable")           # ascending: worst first
    positions = np.empty(present.size, dtype=float)
    i = 0
    m = present.size
    sorted_vals = vals[order]
    while i < m:
        j = i
        while j + 1 < m and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        positions[i:j + 1] = (i + j) / 2.0
        i = j + 1
    out[present[order]] = positions / (m - 1)
    return out


def _rank_from_matrix(am, n_agents: int) -> np.ndarray:
    """Normalise `matrix.population_rank`'s 1-is-best ranks into 1.0-is-best [0, 1].
    Uses the AnteMatrix the tournament already built, so ties are averaged exactly the
    way the tournament's own metrics average them."""
    rank = np.asarray(am.rank, dtype=float)
    present = ~np.isnan(rank)
    out = np.full(n_agents, np.nan)
    m = int(present.sum())
    if m == 0:
        return out
    if m == 1:
        out[present] = 0.5
        return out
    out[present] = 1.0 - (rank[present] - 1.0) / (m - 1)
    return out


def value_targets_from_result(result, n_agents: int, value_blend: float = 0.7) -> dict:
    """Everything the value target needs, computed once per tournament.

    Returns `{"nemesis_antes": [...], "rank_by_ante": {ante: (n,) array},
              "outcome": (n,) array, "blend": float}`.
    """
    nemesis_antes = [am.ante for am in result.ante_matrices]
    rank_by_ante = {am.ante: _rank_from_matrix(am, n_agents) for am in result.ante_matrices}

    survived = np.zeros(n_agents)
    for am in result.ante_matrices:
        survived += (~np.isnan(np.asarray(am.scores, dtype=float))).astype(float)
    lives = np.array([(v if v is not None else 0) for v in result.final_lives], dtype=float)
    lives = np.clip(lives, 0, MLB_STARTING_LIVES)      # the "none" rule's sentinel would swamp the key
    last = np.zeros(n_agents)
    for i, (_, score) in result.last_score.items():
        last[int(i)] = float(score)
    # One composite key, ranked once: rounds survived dominates, then lives, then score.
    keys = [(survived[i], lives[i], math.log1p(max(0.0, last[i]))) for i in range(n_agents)]
    # Rank tuples by mapping to a strictly-ordered scalar is fragile; rank the tuples.
    order = sorted(range(n_agents), key=lambda i: keys[i])
    positions = np.empty(n_agents, dtype=float)
    i = 0
    while i < n_agents:
        j = i
        while j + 1 < n_agents and keys[order[j + 1]] == keys[order[i]]:
            j += 1
        positions[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    outcome = positions / max(1, n_agents - 1) if n_agents > 1 else np.array([0.5])

    return {"nemesis_antes": nemesis_antes, "rank_by_ante": rank_by_ante,
            "outcome": outcome, "blend": float(value_blend)}


def _next_nemesis_ante(ante: int, nemesis_antes: Sequence[int]) -> Optional[int]:
    for a in nemesis_antes:
        if a >= ante:
            return a
    return None


def assign_value_targets(records: Sequence[RecordedDecision], targets: dict) -> None:
    """Fill `record.sample.z` in place. See the module docstring for the formula."""
    nemesis_antes = targets["nemesis_antes"]
    rank_by_ante = targets["rank_by_ante"]
    outcome = targets["outcome"]
    w = targets["blend"]
    for rec in records:
        i = rec.agent_idx
        a = _next_nemesis_ante(rec.ante, nemesis_antes)
        if a is None:
            rank_next = float(outcome[i])
        else:
            r = rank_by_ante[a][i]
            # NaN = agent i is absent from that Nemesis, i.e. it was eliminated before
            # reaching it. That is the worst possible standing, and saying so is the whole
            # reason a life has to cost something.
            rank_next = 0.0 if np.isnan(r) else float(r)
        rec.sample.z = float(w * rank_next + (1.0 - w) * float(outcome[i]))


# ═══════════════════════════════════════════════════════════════════ metrics

@dataclass
class GenerationMetrics:
    """The per-generation read-out the brief asks for. `value_target_sd` is the collapse
    detector: the overnight run sat at 0.07 and anything below `ALARM_SD` means the
    objective has gone degenerate again."""
    ALARM_SD = 0.15

    generation: int = 0
    objective: str = "tournament"
    n_samples: int = 0
    value_target_sd: float = float("nan")
    value_target_mean: float = float("nan")
    skip_rate: float = float("nan")
    skip_opportunities: int = 0
    #: Of the regular (non-Nemesis) blinds the sample-producing agents actually PLAYED, the
    #: fraction they cleared. The other half of the skip-rate story: a high skip rate with a
    #: low clear rate is a net that cannot play blinds yet; a high skip rate with a high
    #: clear rate is a strategy.
    blind_clear_rate: float = float("nan")
    blinds_played: int = 0
    distinct_joker_sets: int = 0
    joker_top5: list = field(default_factory=list)
    mean_jokers: float = float("nan")
    tie_fraction: float = float("nan")
    #: Mean normalised population rank (1.0 = best in the population) per seat GROUP,
    #: averaged over every Nemesis of the generation. `rank_current` vs `rank_anchor` is the
    #: skip-vs-build referendum: if the net's skip-everything policy were actually winning,
    #: `rank_current` would sit above `rank_anchor`.
    rank_current: float = float("nan")
    rank_anchor: float = float("nan")
    rank_history: float = float("nan")
    lives_lost: dict = field(default_factory=dict)
    max_ante_reached: int = 0
    mean_ante_reached: float = float("nan")
    decisions: int = 0
    searches: int = 0
    sims_per_s: float = float("nan")
    leaf_evals_per_s: float = float("nan")
    episodes: int = 0
    ep_per_min: float = float("nan")
    wall_clock_s: float = 0.0
    train_steps: int = 0
    policy_loss: float = float("nan")
    value_loss: float = float("nan")

    @property
    def collapsed(self) -> bool:
        """Below `ALARM_SD` the value head has nothing left to learn from.

        The threshold is calibrated for the TOURNAMENT objective, where the target is a
        dense population rank and a healthy sd is ~0.30. `--objective external` is an
        ABSOLUTE objective: when every episode fails (or every episode succeeds) its targets
        legitimately cluster, so a low sd there means "the target is mis-scaled for this
        agent", not "the loop is broken". `console_line` says which."""
        return not (self.value_target_sd > self.ALARM_SD)

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        d["collapsed"] = self.collapsed
        return d

    def console_line(self) -> str:
        alarm = ""
        if self.collapsed:
            alarm = ("  ** VALUE-TARGET COLLAPSE **" if self.objective == "tournament"
                     else "  ** target mis-scaled for this agent **")
        return (
            f"[gen {self.generation:4d}] {self.wall_clock_s:7.1f}s "
            f"eps {self.episodes:4d} ({self.ep_per_min:5.1f} ep/min) "
            f"samples {self.n_samples:6d} | "
            f"z sd {self.value_target_sd:.3f} mean {self.value_target_mean:.3f} | "
            f"skip {self.skip_rate * 100:5.1f}% clear {self.blind_clear_rate * 100:5.1f}% | "
            f"jokersets {self.distinct_joker_sets:3d} (mean {self.mean_jokers:.2f}) | "
            f"tie {self.tie_fraction:.3f} | "
            f"rank cur {self.rank_current:.3f} vs anch {self.rank_anchor:.3f} | "
            f"ante max {self.max_ante_reached} mean {self.mean_ante_reached:.2f} | "
            f"{self.sims_per_s:6.0f} sims/s | "
            f"loss p={self.policy_loss:.3f} v={self.value_loss:.4f}{alarm}"
        )


def _clear_rate(records) -> float:
    played = [r.blind_result for r in records if r.blind_result is not None]
    return (sum(played) / len(played)) if played else float("nan")


def _joker_keys(game) -> tuple:
    return tuple(sorted(getattr(j, "key", "?") for j in getattr(game, "jokers", [])))


# ═══════════════════════════════════════════════════════════════════ objective 1: tournament

@dataclass
class TournamentSpec:
    seeds: Sequence
    n_agents: int = 16
    life_rule: str = "paired"
    max_ante: int = 4
    deck_key: str = "b_red"
    stake: int = 1
    lives: int = MLB_STARTING_LIVES
    value_blend: float = 0.7
    max_steps_per_drive: int = 20_000


@dataclass
class GenerationSamples:
    samples: list
    records: list
    results: list
    joker_sets: list
    metrics: dict


def run_tournament_generation(spec: TournamentSpec, members, players, encoder, *,
                              sample_fn: Optional[Callable] = None,
                              traj_logger_factory: Optional[Callable] = None,
                              out_dir: Optional[str] = None,
                              stop_check: Optional[Callable[[], bool]] = None,
                              max_samples_per_agent: int = 0,
                              ) -> GenerationSamples:
    """Run `len(spec.seeds)` tournaments with this population and return the samples the
    CURRENT-net agents produced, already labelled.

    `players[i]` must be an `mcts.MCTSPlayer` (the tournament's `MCTSPlayer` factory
    returns one). Only members with `is_current` get a `record_hook`; past-self opponents
    play with no hook at all, which is both the correct objective (do not imitate a worse
    net) and free.
    """
    tour_mod = tournament_module()
    Tournament = tour_mod.runner.Tournament

    collectors: dict[int, SampleCollector] = {}
    for m in members:
        if m.is_current:
            collectors[m.idx] = SampleCollector(m.idx, encoder, sample_fn=sample_fn,
                                                max_records=max_samples_per_agent)

    all_records: list[RecordedDecision] = []
    results = []
    joker_sets: list[tuple] = []
    lives_lost: Counter = Counter()
    groups = {
        "current": [m.idx for m in members if m.is_current],
        "anchor": [m.idx for m in members if m.is_anchor],
        "history": [m.idx for m in members
                    if not m.is_current and not m.is_anchor],
    }
    group_ranks: dict = {k: [] for k in groups}
    tie_fractions: list[float] = []
    antes_reached: list[int] = []
    t0 = time.perf_counter()
    searches_before = sum(getattr(p, "searches", 0) for p in players)
    leaves_before = _leaf_count(players)

    n_played = 0
    for si, seed in enumerate(spec.seeds):
        if stop_check is not None and stop_check():
            break            # PAUSE / SIGTERM: one tournament is the atomic unit of play
        # ── W3 hook (mp/replay): ONE TrajectoryLogger per current-net agent. W3's logger
        # is a one-`BalatroGame`-per-episode object, and a tournament is N games, so a
        # single logger cannot describe it. `Tournament.on_fanout` (added by W2 for exactly
        # this) hands over the games the instant they exist, which is the only moment
        # `begin(game, meta)` can be called.
        loggers: dict = ({m.idx: traj_logger_factory(seed) for m in members}
                         if traj_logger_factory is not None else {})
        for idx, coll in collectors.items():
            coll.clear()
            players[idx].record_hook = coll

        def _begin(games, seed_str, _loggers=loggers):
            for idx, lg in _loggers.items():
                lg.begin(games[idx], {"source": "train_mlb", "agent": idx,
                                      "is_current": bool(members[idx].is_current),
                                      "n_agents": spec.n_agents,
                                      "life_rule": spec.life_rule,
                                      "max_ante": spec.max_ante})

        def _on_step(idx, game, action, _loggers=loggers):
            lg = _loggers.get(idx)
            if lg is not None:
                lg.step(game, action)

        def _on_done(idx, game, reason, _loggers=loggers):
            lg = _loggers.pop(idx, None)
            if lg is not None:
                lg.end(game, {"objective": "tournament", "agent": idx, "reason": reason,
                              "lives": int(getattr(game, "lives", 0)),
                              "ante": int(getattr(game, "ante", 0))})

        tour = Tournament(
            seed=seed, n_agents=spec.n_agents, players=list(players),
            deck_key=spec.deck_key, stake=spec.stake, life_rule=spec.life_rule,
            max_ante=spec.max_ante, lives=spec.lives,
            max_steps_per_drive=spec.max_steps_per_drive,
            out_dir=(str(Path(out_dir) / f"seed{si:02d}") if out_dir else None),
            on_fanout=_begin if loggers else None,
            on_step=_on_step if loggers else None,
            on_agent_done=_on_done if loggers else None,
        )
        result = tour.run()
        results.append(result)

        targets = value_targets_from_result(result, spec.n_agents, spec.value_blend)
        for name, idxs in groups.items():
            ranks = [targets["rank_by_ante"][a][i]
                     for a in targets["nemesis_antes"] for i in idxs
                     if not np.isnan(targets["rank_by_ante"][a][i])]
            group_ranks[name].extend(ranks)
        seed_records = [r for coll in collectors.values() for r in coll.records]
        assign_value_targets(seed_records, targets)
        all_records.extend(seed_records)

        games = getattr(tour, "_last_games", None) or []
        for i, g in enumerate(games):
            joker_sets.append(_joker_keys(g))
            antes_reached.append(int(getattr(g, "ante", 0)))
            fl = result.final_lives[i]
            if fl is not None and spec.life_rule != "none":
                lives_lost[int(max(0, spec.lives - fl))] += 1
        tie_fractions.extend([am.tie_fraction for am in result.ante_matrices
                              if not math.isnan(am.tie_fraction)])
        n_played += 1
        assert not loggers, "Tournament.on_agent_done must close every trajectory"

    for idx in collectors:
        players[idx].record_hook = None       # never leave a hook armed on a shared player

    wall = time.perf_counter() - t0
    searches = sum(getattr(p, "searches", 0) for p in players) - searches_before
    leaves = _leaf_count(players) - leaves_before
    # Nominal simulations: mean budget over the seats x searches actually run. `leaves`
    # (below) is the honest count of net evaluations; both are logged because tree reuse
    # makes them differ by design (a reused root spends fewer sims for the same evidence).
    net_seats = [m for m in members if not m.is_anchor] or list(members)
    sims = sum(m.sims for m in net_seats) / len(net_seats) * max(0, searches)

    samples = [r.sample for r in all_records]
    metrics = _tournament_metrics(all_records, joker_sets, lives_lost, tie_fractions,
                                  antes_reached, wall, searches, sims, leaves,
                                  episodes=n_played * spec.n_agents)
    for name, vals in group_ranks.items():
        metrics[f"rank_{name}"] = float(np.mean(vals)) if vals else float("nan")
    return GenerationSamples(samples=samples, records=all_records, results=results,
                             joker_sets=joker_sets, metrics=metrics)


def _leaf_count(players) -> int:
    """Leaves actually evaluated by the net, summed over the DISTINCT policy objects the
    population uses (several players share one policy)."""
    seen = {}
    for p in players:
        pol = getattr(p, "policy", None)
        if pol is not None:
            seen[id(pol)] = getattr(pol, "leaves", 0)
    return int(sum(seen.values()))


def _tournament_metrics(records, joker_sets, lives_lost, tie_fractions, antes_reached,
                        wall, searches, sims, leaves, episodes: int) -> dict:
    zs = [r.sample.z for r in records]
    skip_ops = [r for r in records if r.skip_offered]
    skipped = [r for r in skip_ops if r.action_type == "skip_blind"]
    counts = Counter(k for js in joker_sets for k in js)
    return {
        "n_samples": len(records),
        "value_target_sd": float(np.std(zs)) if zs else float("nan"),
        "value_target_mean": float(np.mean(zs)) if zs else float("nan"),
        "skip_rate": (len(skipped) / len(skip_ops)) if skip_ops else float("nan"),
        "skip_opportunities": len(skip_ops),
        "blind_clear_rate": _clear_rate(records),
        "blinds_played": sum(1 for r in records if r.blind_result is not None),
        "distinct_joker_sets": len({js for js in joker_sets}),
        "joker_top5": counts.most_common(5),
        "mean_jokers": (float(np.mean([len(js) for js in joker_sets]))
                        if joker_sets else float("nan")),
        "tie_fraction": float(np.mean(tie_fractions)) if tie_fractions else float("nan"),
        "lives_lost": {str(k): int(v) for k, v in sorted(lives_lost.items())},
        "max_ante_reached": max(antes_reached) if antes_reached else 0,
        "mean_ante_reached": float(np.mean(antes_reached)) if antes_reached else float("nan"),
        "decisions": len(records),
        "searches": int(searches),
        "sims_per_s": (sims / wall) if wall > 0 else float("nan"),
        # NaN, not 0, when the policy exposes no counter (the serial leaf path does not
        # go through `evaluate_many`) -- 0 would read as "the net was never called".
        "leaf_evals_per_s": ((leaves / wall) if (wall > 0 and leaves > 0)
                             else float("nan")),
        "episodes": episodes,
        "ep_per_min": (episodes / wall * 60.0) if wall > 0 else float("nan"),
        "wall_clock_s": wall,
    }


# ═══════════════════════════════════════════════════════════ objective 2: external target

def vanilla_boss_target(ante: int, deck: str = "b_red", stake: int = 1) -> int:
    """FALLBACK for W4's `mp/eval/targets.py::vanilla_boss` — what a vanilla Boss blind
    would have demanded at this ante, i.e. "score at least as much as the single-player
    game would have asked of you". Uses the engine's own `blind_base_chips` (blind_idx 2 =
    Boss) times the deck's `ante_scaling` (Plasma x2, `decks.py:56`, applied by the caller
    exactly as `game.py:642` does). `stake` maps to the blind-scaling table: White/Red = 1,
    Green+ = 2, Purple+ = 3 (`constants.BLIND_AMOUNTS_BY_SCALING`).

    Deliberately NOT boss-multiplied: the Nemesis has no boss ability, so the plain Boss
    amount is the honest "a competent single-player run clears this" bar.
    """
    scaling = 1 if stake <= 4 else (2 if stake <= 6 else 3)
    ante_scaling = 2 if deck == "b_plasma" else 1
    return int(blind_base_chips(max(1, int(ante)), 2, scaling) * ante_scaling)


def big_blind_floor(game) -> int:
    """The chip requirement this ante's vanilla BIG blind carried — the floor under any
    self-referential Nemesis target.

    Why it exists: W4's `own_big_blind` target (`k x` the agent's own Big-blind score that
    ante) is ~50/50 by construction *for an agent that plays the Big blind*. Ours does not:
    under MLB a regular blind can be SKIPPED for free, so `big_blind[ante]` stays 0, the
    mirror target is 0, and the Nemesis is free again — the identical degeneracy the
    overnight run found, reached by a different road. Measured, not hypothesised: with a raw
    mirror target a cold net skipped 77.5% of its blinds, cleared every Nemesis, and the
    value target sat at sd 0.14 / mean 0.84.

    Flooring the target at what the Big blind would have demanded removes the free lunch
    without touching W4's function: playing the Big blind (which requires scoring at least
    its target) already clears the floor, so the floor only ever bites on a skip or a fail.
    """
    ante = max(1, int(getattr(game, "ante", 1)))
    scaling = int(getattr(game, "blind_scaling", 1) or 1)
    return int(blind_base_chips(ante, 1, scaling) * int(getattr(game, "ante_scaling", 1) or 1))


def load_target_fn(kind: str = "own_big_blind", deck: str = "b_red", stake: int = 1,
                   multiplier: float = 1.0, floor_frac: float = 1.0):
    """`(game) -> chip target`, W4's shared `target_fn(game, big_blind=None) -> int`
    signature (`mp/eval/targets.py` module docstring). Prefers W4's registry — the
    campaign's one table of per-ante Nemesis targets, including the `table` target derived
    from real tournament score distributions — and falls back to the local
    `vanilla_boss_target` above, which is the same formula from the same constants.

    Returns `(fn, source)` so the run log records which table was used: a run trained
    against a different target is a different experiment.
    """
    source = "selfplay.vanilla_boss_target"
    inner = None
    try:
        tournament_module()                       # puts mp/ on sys.path
        from eval import targets as _targets      # type: ignore  # noqa: WPS433
    except Exception:                             # noqa: BLE001 — W4 may not have landed
        _targets = None
    if _targets is not None:
        # W4's registry entries take different kwargs (`vanilla_boss` wants deck/stake,
        # `own_big_blind` wants only `k`), so probe rather than assume.
        for kwargs in ({"deck_key": deck, "stake": stake}, {}):
            try:
                inner = _targets.get_target(kind, **kwargs)
                source = f"mp/eval/targets.py::get_target({kind!r})"
                break
            except TypeError:
                continue
            except Exception:                     # noqa: BLE001 — unknown kind
                break
    if inner is None:
        def inner(game, big_blind=None):          # noqa: WPS430
            return vanilla_boss_target(int(game.ante), deck, stake)

    if multiplier != 1.0:
        base = inner

        def inner(game, big_blind=None, _f=base):      # noqa: WPS430
            return int(_f(game, big_blind) * multiplier)
        source = f"{source} x{multiplier}"

    if floor_frac > 0:
        raw = inner

        def inner(game, big_blind=None, _f=raw):       # noqa: WPS430
            return max(int(_f(game, big_blind)), int(floor_frac * big_blind_floor(game)))
        source = f"{source} floor={floor_frac}xBig"

    return inner, source


def _mlb_shaped_value(game, lives: int, starting_lives: int, horizon_antes: int,
                      lives_weight: float) -> float:
    """`MLBOutcome.value` with the lives count supplied rather than read off the game —
    so the search can be shown the life the solo engine has not charged yet."""
    life_frac = min(1.0, max(0, lives) / max(1, starting_lives))
    blinds = (int(game.ante) - 1) * 3 + int(getattr(game, "blind_idx", 0))
    progress = min(1.0, blinds / max(1, 3 * horizon_antes))
    return float(min(1.0, max(0.0, lives_weight * life_frac + (1 - lives_weight) * progress)))


def external_outcome_for(target_fn, starting_lives: int = MLB_STARTING_LIVES,
                         horizon_antes: int = 8, lives_weight: float = 0.5,
                         big_blind: Optional[dict] = None):
    """The `OutcomeFn` the SEARCH uses under `--objective external`.

    `MLBOutcome` is what made the overnight run degenerate: inside the tree a Nemesis is
    free, so every line that reaches one looks equally good. This wraps it so that a leaf
    sitting at a resolved-but-not-yet-cashed-out Nemesis (`PVP_WAIT`, or `ROUND_EVAL` with
    `is_pvp`) whose `chips_scored` is under `target_fn(ante)` is valued as if the life the
    DRIVER is about to charge had already been charged. Same shape, same range, one term
    moved — so the search can see the cost of coasting.

    The target is derived from the game's ANTE (W4's `target_fn(game)` reads `game.ante` /
    `deck_key` / `stake`), never from `current_blind.chips_target`, on purpose:
    `_start_blind` zeroes a PvP blind's target (`game.py:1005`), so a subtree that crosses a
    blind start would otherwise lose the target entirely.
    """
    base = MLBOutcome(starting_lives=starting_lives, horizon_antes=horizon_antes,
                      lives_weight=lives_weight)

    def value_fn(game: BalatroGame) -> float:
        if base.is_win(game):
            return 1.0
        lives = int(getattr(game, "lives", 0))
        blind = getattr(game, "current_blind", None)
        pending = (getattr(blind, "is_pvp", False)
                   and game.state in (State.PVP_WAIT, State.ROUND_EVAL)
                   and not getattr(game, "life_lost_this_round", False)
                   and int(getattr(game, "chips_scored", 0)) < int(target_fn(game, big_blind)))
        return _mlb_shaped_value(game, lives - (1 if pending else 0), starting_lives,
                                 horizon_antes, lives_weight)

    return ExternalOutcome(value_fn=value_fn, base=base, name="external_target")


@dataclass
class SoloEpisodeResult:
    records: list
    final_ante: int
    final_lives: int
    stop_reason: str
    decisions: int
    nemesis_results: list           # [{"ante", "score", "target", "lost_life"}]
    joker_keys: tuple = ()


def play_solo_external_episode(game: BalatroGame, player, encoder, target_fn, *,
                               agent_idx: int = 0, max_decisions: int = 2000,
                               max_antes: Optional[int] = None,
                               sample_fn: Optional[Callable] = None,
                               value_blend: float = 0.7,
                               starting_lives: int = MLB_STARTING_LIVES,
                               margin_scale: float = 1.0,
                               pvp_relay: bool = True,
                               big_blind: Optional[dict] = None,
                               traj_logger=None) -> SoloEpisodeResult:
    """One solo MLB episode where the Nemesis COSTS something.

    The engine cannot charge it: with `pvp_solo=True` the Nemesis auto-resolves at hand
    exhaustion with no life lost (`game.py::end_pvp`), which is exactly the degeneracy the
    overnight run exploited. So this driver does what the tournament runner does with the
    N x N verdict — compares the final score to an external target and calls the same
    public `game.lose_life()` hook.

    Value targets mirror the tournament's shape: short-horizon (how did I do at the NEXT
    Nemesis, as a logistic of the log-margin against its target) blended with the final
    MLB-shaped outcome.
    """
    collector = SampleCollector(agent_idx, encoder, sample_fn=sample_fn)
    log = traj_logger        # W3 `mp/replay`: logged HERE, after each step, not in the
                             # record hook (which fires before the step). See REPLAY_NOTES §2.
    # The same no-progress guard the tournament runner carries, for the same frozen-engine
    # gap (a SHOP `use_consumable` that changes nothing — TRAIN_NOTES §9). A solo run wedges
    # on it just as readily; observed as an episode spending all 400 of its decisions inside
    # one ante-1 shop.
    _runner = tournament_module().runner
    force_progress, noop_budget = _runner._force_progress_action, _runner.NOOP_BUDGET_DEFAULT
    sig = game.state_signature()
    noops = 0
    player.record_hook = collector
    player.reset()
    nemesis: list[dict] = []
    stop = "max_decisions"
    decisions = 0
    base = MLBOutcome(starting_lives=starting_lives)
    # `{ante: own Big-blind chip score}` — W4's `own_big_blind` target reads this
    # (`targets.py::scaled_own_big_blind`) and it is the driver's job to fill it in before
    # that ante's Nemesis. Shared (not copied) with the caller and with the OutcomeFn, so
    # the search sees the same mirror target the driver will charge.
    bb: dict = big_blind if big_blind is not None else {}

    try:
        for _ in range(max_decisions):
            if game.state is State.GAME_OVER:
                stop = "game_over"
                break
            if max_antes is not None and game.ante > max_antes:
                stop = "max_antes"
                break
            blind = game.current_blind
            if (game.state is State.ROUND_EVAL and getattr(blind, "kind", "") == "Big"
                    and not getattr(blind, "is_pvp", False)):
                bb[int(game.ante)] = int(game.chips_scored)
            if game.state is State.ROUND_EVAL and getattr(blind, "is_pvp", False):
                target = int(target_fn(game, bb))
                score = int(game.chips_scored)
                lost = False
                if score < target:
                    lost = bool(game.lose_life())
                    if log is not None:
                        log.step(game, {"type": OP_LOSE_LIFE})
                nemesis.append({"ante": int(game.ante), "score": score,
                                "target": target, "lost_life": lost})
                if game.lives <= 0:
                    # Deliberately NOT `state = GAME_OVER`: that would be an out-of-band
                    # mutation no replay can reproduce, and the run is over either way.
                    # `final_lives == 0` is the signal; `stop_reason` says why.
                    stop = "out_of_lives"
                    break
                game.step({"type": "advance"})           # cash out
                if log is not None:
                    log.step(game, {"type": "advance"})
                sig, noops = game.state_signature(), 0
                continue
            if is_stuck_state(game):
                stop = "stuck"
                break
            if (pvp_relay and game.state is State.SELECTING_HAND
                    and getattr(blind, "is_pvp", False)):
                # `enemyInfo`: show the agent what it has to beat. Must be re-applied after
                # `play_blind` -- `_start_blind` zeroes a PvP blind's target.
                score = int(target_fn(game, bb))
                game.set_pvp_info(score, int(game.hands_left))
                if log is not None:
                    log.step(game, {"type": OP_SET_PVP_INFO, "score": score,
                                    "hands": int(game.hands_left)})
                sig = game.state_signature()      # set_pvp_info changes the state too
            action = player.act(game)
            if action is None:
                stop = "no_actions"
                break
            game.step(action)
            if log is not None:
                log.step(game, action)
            decisions += 1
            new_sig = game.state_signature()
            if new_sig == sig:
                noops += 1
                if noops >= noop_budget:
                    forced = force_progress(game)
                    if forced is None:
                        stop = "wedged"
                        break
                    game.step(forced)
                    if log is not None:
                        log.step(game, forced)
                    new_sig = game.state_signature()
                    noops = 0
            else:
                noops = 0
            sig = new_sig
    finally:
        player.record_hook = None

    final_outcome = base.value(game)
    by_ante = {n["ante"]: n for n in nemesis}
    antes = sorted(by_ante)
    for rec in collector.records:
        nxt = next((a for a in antes if a >= rec.ante), None)
        if nxt is None:
            short = final_outcome
        else:
            n = by_ante[nxt]
            margin = math.log1p(max(0, n["score"])) - math.log1p(max(0, n["target"]))
            short = 1.0 / (1.0 + math.exp(-margin / max(margin_scale, 1e-9)))
        rec.sample.z = float(value_blend * short + (1 - value_blend) * final_outcome)

    return SoloEpisodeResult(
        records=collector.records, final_ante=int(game.ante), final_lives=int(game.lives),
        stop_reason=stop, decisions=decisions, nemesis_results=nemesis,
        joker_keys=_joker_keys(game),
    )


def solo_metrics(episodes: Sequence[SoloEpisodeResult], wall: float, searches: int,
                 sims: float, leaves: int) -> dict:
    records = [r for ep in episodes for r in ep.records]
    zs = [r.sample.z for r in records]
    skip_ops = [r for r in records if r.skip_offered]
    skipped = [r for r in skip_ops if r.action_type == "skip_blind"]
    joker_sets = [ep.joker_keys for ep in episodes]
    counts = Counter(k for js in joker_sets for k in js)
    lives_lost = Counter(max(0, MLB_STARTING_LIVES - ep.final_lives) for ep in episodes)
    return {
        "n_samples": len(records),
        "value_target_sd": float(np.std(zs)) if zs else float("nan"),
        "value_target_mean": float(np.mean(zs)) if zs else float("nan"),
        "skip_rate": (len(skipped) / len(skip_ops)) if skip_ops else float("nan"),
        "skip_opportunities": len(skip_ops),
        "blind_clear_rate": _clear_rate(records),
        "blinds_played": sum(1 for r in records if r.blind_result is not None),
        "distinct_joker_sets": len(set(joker_sets)),
        "joker_top5": counts.most_common(5),
        "mean_jokers": float(np.mean([len(js) for js in joker_sets])) if joker_sets else float("nan"),
        "tie_fraction": float("nan"),          # no population: nothing to tie with
        "lives_lost": {str(k): int(v) for k, v in sorted(lives_lost.items())},
        "max_ante_reached": max((ep.final_ante for ep in episodes), default=0),
        "mean_ante_reached": (float(np.mean([ep.final_ante for ep in episodes]))
                              if episodes else float("nan")),
        "decisions": len(records),
        "searches": int(searches),
        "sims_per_s": (sims / wall) if wall > 0 else float("nan"),
        "leaf_evals_per_s": ((leaves / wall) if (wall > 0 and leaves > 0)
                             else float("nan")),
        "episodes": len(episodes),
        "ep_per_min": (len(episodes) / wall * 60.0) if wall > 0 else float("nan"),
        "wall_clock_s": wall,
    }


# ═══════════════════════════════════════════════════════════════════ the generation loop

@dataclass
class MLBTrainConfig:
    """Everything about a generation-based run that `TrainConfig` does not already cover.
    Carried in the checkpoint so `--resume` cannot silently change the experiment."""
    objective: str = "tournament"          # "tournament" | "external"
    n_agents: int = 16
    m_current: int = 8
    p_history: int = 4
    seeds_per_generation: int = 2
    max_ante: int = 4
    life_rule: str = "paired"
    value_blend: float = 0.7
    #: Training-time cap on `skip_blind` per ante for the SAMPLE-PRODUCING seats
    #: (None = uncapped, the real MLB rule). See `population.SkipCap`.
    max_skips_per_ante: Optional[int] = None
    #: Lift the cap once the rolling blind-clear rate exceeds this, or after this many
    #: generations - whichever comes first - so the final policy is trained under the
    #: real rules.
    skip_cap_anneal_clear_rate: float = 0.5
    skip_cap_anneal_generations: int = 0        # 0 = only the clear-rate criterion
    #: Scripted, never-skipping reference players as a fraction of the population. Without
    #: them the population is all one lineage and a shared bad habit is invisible to the
    #: rank target — measured: 95-98% skip rate by generation 4 of the anchor-free gate.
    anchor_frac: float = 0.25
    sims_budgets: tuple = (1.0, 0.5, 1.5)
    #: 1, not BATCH_NOTES §7.1's 16. That recommendation was measured at 500 sims/decision;
    #: at the 40-60 sims a generation-based run can afford, an interleaved A/B on this box
    #: puts L=1 / 4 / 16 at 238 / 217 / 218 sims/s on CUDA and 249 / 251 / 247 on CPU --
    #: i.e. no gain, and L>1 changes the search (virtual loss). L=1 keeps it exact.
    leaf_batch: int = 1
    reuse: bool = True
    strategy: str = "gumbel"
    noise_current: bool = True
    # external objective
    episodes_per_generation: int = 16
    # `own_big_blind` (k x the agent's OWN Big-blind score that ante) rather than
    # `vanilla_boss`: a fixed vanilla-Boss target is unreachable for a cold net, so every
    # margin saturates the logistic and the value target collapses for the SECOND time --
    # measured, see TRAIN_NOTES §9. The mirror target is ~50/50 by construction and
    # improves as the agent does (W4's `targets.py::scaled_own_big_blind`).
    target_kind: str = "own_big_blind"
    target_multiplier: float = 1.0
    #: Floor on the Nemesis target, as a multiple of this ante's vanilla BIG-blind amount.
    #: Without it a self-referential target (`own_big_blind`) is 0 whenever the agent skips
    #: the Big blind, which makes the Nemesis free again — see `big_blind_floor`.
    target_floor: float = 1.0
    margin_scale: float = 1.0
    pvp_relay: bool = True
    # optimisation
    train_steps: int = 0                   # 0 = auto (one pass over the new samples)
    max_train_steps: int = 2_000
    # Insurance against a pathological agent flooding the buffer (see TRAIN_NOTES
    # "needs engine change": one MCTS agent produced 14 338 samples in a single shop
    # before the runner's no-progress guard existed). A normal agent to ante 8 produces
    # ~150-400.
    max_samples_per_agent: int = 2_000


class MLBTrainer:
    """Generation-based training against a non-degenerate objective.

    Deliberately a thin shell over `ColdTrainer`: the net, the optimizer, the replay
    buffer, the numpy generator, the counters and the checkpoint payload are all the ones
    `train_cold` already uses, so a `train_mlb` checkpoint IS a `train_cold` checkpoint
    (`checkpoint.CHECKPOINT_KIND` unchanged) and `mcts.load_policy` can read it — which is
    how a generation's past selves get loaded — and Phase 3's bit-exact round-trip still
    holds. What is added is the generation counter, the opponent history, and a
    `state_dict` that carries both under `"mlb"`.
    """

    def __init__(self, cfg, mlb: MLBTrainConfig):
        from mcts.policy import make_policy
        from .loop import ColdTrainer

        self.cold = ColdTrainer(cfg)
        self.mlb = mlb
        self.cfg = self.cold.cfg
        self.rng = self.cold.rng
        # `make_policy(..., batched=True)`, not ColdTrainer's serial policy: `leaf_batch=16`
        # is BATCH_NOTES §7.1's fastest single-tree configuration and the tournament runner
        # drives one agent at a time, so within-tree batching is the only batching there is.
        # W1's factory picks the flat or the set implementation off the encoder, so
        # `--encoder set` needs no branch here.
        self.policy = make_policy(self.cold.net, device=cfg.device,
                                  encoder=self.cold.encoder, batched=True)
        # W1's `SampleBuilder` IS this workstream's `sample_fn` (same signature, published
        # in SETENC_NOTES §0.6): subsampled `Sample` v2 with whichever encoder is
        # configured. `None` -> the local v1 `make_sample` above.
        self.sample_fn = getattr(self.cold, "sample_builder", None) or make_sample
        self.history = CheckpointHistory(capacity=mlb.p_history)
        self.generation = 0
        self.pop_cfg = PopulationConfig(
            n_agents=mlb.n_agents, m_current=mlb.m_current, p_history=mlb.p_history,
            sims=cfg.sims, sims_budgets=tuple(mlb.sims_budgets),
            noise_current=mlb.noise_current, encoder=cfg.encoder, device=cfg.device,
            leaf_batch=mlb.leaf_batch, reuse=mlb.reuse, strategy=mlb.strategy,
            anchor_frac=mlb.anchor_frac,
        )
        self.target_fn, self.target_source = load_target_fn(
            mlb.target_kind, deck=cfg.deck_key, stake=cfg.stake,
            multiplier=mlb.target_multiplier, floor_frac=mlb.target_floor)
        self.big_blind: dict = {}          # {ante: own Big-blind score}, W4's mirror target
        self.clear_rate_ema: Optional[float] = None    # drives the skip-cap anneal
        if mlb.objective == "external":
            self.outcome = external_outcome_for(self.target_fn,
                                                starting_lives=cfg.lives,
                                                horizon_antes=max(1, mlb.max_ante),
                                                big_blind=self.big_blind)
        else:
            self.outcome = MLBOutcome(starting_lives=cfg.lives,
                                      horizon_antes=max(1, mlb.max_ante))
        self._policy_cache: dict = {}

    # ── Pieces ───────────────────────────────────────────────────────────────

    @property
    def net(self):
        return self.cold.net

    @property
    def buffer(self):
        return self.cold.buffer

    @property
    def counters(self):
        return self.cold.counters

    def _episode_seed(self) -> int:
        return int(self.rng.integers(0, 2**31 - 1))

    def build_players(self):
        members = build_population(self.pop_cfg, self.generation, self.history,
                                   base_seed=self.cfg.seed)
        players = instantiate(members, self.policy, device=self.cfg.device,
                              encoder=self.cfg.encoder, leaf_batch=self.mlb.leaf_batch,
                              reuse=self.mlb.reuse, strategy=self.mlb.strategy,
                              outcome=self.outcome, policy_cache=self._policy_cache,
                              max_skips_per_ante=self.effective_skip_cap())
        return members, players

    def effective_skip_cap(self) -> Optional[int]:
        """The cap this generation actually runs under, after annealing: None once the net
        can clear blinds by itself (rolling clear rate over the threshold) or after
        `skip_cap_anneal_generations`, whichever comes first."""
        cap = self.mlb.max_skips_per_ante
        if cap is None:
            return None
        g = self.mlb.skip_cap_anneal_generations
        if g and self.generation >= g:
            return None
        rate = self.clear_rate_ema
        if rate is not None and rate > self.mlb.skip_cap_anneal_clear_rate:
            return None
        return cap

    # ── One generation ───────────────────────────────────────────────────────

    def run_generation(self, *, stop_check: Optional[Callable[[], bool]] = None,
                       traj_logger_factory: Optional[Callable] = None,
                       out_dir: Optional[str] = None) -> GenerationMetrics:
        """Play -> label -> buffer -> train -> bump the generation counter. Checkpointing
        and the population hand-off are the caller's (`scripts/train_mlb.py`), because only
        the caller knows where checkpoints go."""
        t0 = time.perf_counter()
        if self.mlb.objective == "external":
            samples, raw = self._play_external(stop_check, traj_logger_factory)
        else:
            samples, raw = self._play_tournament(stop_check, traj_logger_factory, out_dir)

        self.buffer.extend(samples)
        self.counters.samples += len(samples)
        train = self._train(len(samples))

        fields = set(GenerationMetrics.__dataclass_fields__)
        m = GenerationMetrics(generation=self.generation, objective=self.mlb.objective,
                              **{k: v for k, v in raw.items() if k in fields})
        m.wall_clock_s = time.perf_counter() - t0
        m.ep_per_min = ((m.episodes / m.wall_clock_s * 60.0)
                        if m.wall_clock_s > 0 else float("nan"))
        m.train_steps = train["steps"]
        m.policy_loss = train["policy_loss"]
        m.value_loss = train["value_loss"]
        self.counters.episodes += m.episodes
        if not math.isnan(m.blind_clear_rate):
            self.clear_rate_ema = (m.blind_clear_rate if self.clear_rate_ema is None
                                   else 0.7 * self.clear_rate_ema + 0.3 * m.blind_clear_rate)
        self.extra_metrics = {k: v for k, v in raw.items() if k not in fields}
        self.extra_metrics["skip_cap"] = self.effective_skip_cap()
        self.extra_metrics["clear_rate_ema"] = self.clear_rate_ema
        self.generation += 1
        return m

    def _play_tournament(self, stop_check, traj_logger_factory, out_dir):
        members, players = self.build_players()
        seeds = [self._episode_seed() for _ in range(self.mlb.seeds_per_generation)]
        spec = TournamentSpec(
            seeds=seeds, n_agents=self.mlb.n_agents, life_rule=self.mlb.life_rule,
            max_ante=self.mlb.max_ante, deck_key=self.cfg.deck_key, stake=self.cfg.stake,
            lives=self.cfg.lives, value_blend=self.mlb.value_blend,
        )
        gs = run_tournament_generation(spec, members, players, self.cold.encoder,
                                       sample_fn=self.sample_fn,
                                       traj_logger_factory=traj_logger_factory,
                                       out_dir=out_dir, stop_check=stop_check,
                                       max_samples_per_agent=self.mlb.max_samples_per_agent)
        gs.metrics["population"] = population_summary(members)
        gs.metrics["seeds"] = [str(s) for s in seeds]
        return gs.samples, gs.metrics

    def _play_external(self, stop_check, traj_logger_factory):
        from mcts import MCTSConfig, MCTSPlayer
        player = MCTSPlayer(
            policy=self.policy,
            config=MCTSConfig(num_simulations=self.cfg.sims,
                              leaf_batch=self.mlb.leaf_batch),
            outcome=self.outcome, rng=self.rng, strategy=self.mlb.strategy,
            add_noise=self.mlb.noise_current, reuse=self.mlb.reuse,
            leaf_batch=self.mlb.leaf_batch, name="cur")
        episodes: list = []
        t0 = time.perf_counter()
        leaves0 = getattr(self.policy, "leaves", 0)
        for _ in range(self.mlb.episodes_per_generation):
            if stop_check is not None and stop_check():
                break                     # PAUSE / SIGTERM: an episode is the atomic unit
            seed = self._episode_seed()
            logger = traj_logger_factory(seed) if traj_logger_factory is not None else None
            game = BalatroGame(seed=seed, deck_key=self.cfg.deck_key,
                               stake=self.cfg.stake, ruleset="mlb")
            game.lives = self.cfg.lives
            self.big_blind.clear()         # per episode; the OutcomeFn holds the same dict
            if logger is not None:
                # W3 hook: one solo episode == one BalatroGame, W3's exact shape.
                logger.begin(game, {"source": "train_mlb", "objective": "external",
                                    "target": self.target_source})
            ep = play_solo_external_episode(
                game, player, self.cold.encoder, self.target_fn,
                sample_fn=self.sample_fn,
                max_decisions=self.cfg.max_decisions, max_antes=self.mlb.max_ante,
                value_blend=self.mlb.value_blend, starting_lives=self.cfg.lives,
                margin_scale=self.mlb.margin_scale, pvp_relay=self.mlb.pvp_relay,
                big_blind=self.big_blind, traj_logger=logger)
            episodes.append(ep)
            if logger is not None:
                logger.end(game, {"objective": "external", "ante": ep.final_ante,
                                  "lives": ep.final_lives, "stop": ep.stop_reason,
                                  "nemesis": ep.nemesis_results})
        wall = time.perf_counter() - t0
        searches = getattr(player, "searches", 0)
        leaves = getattr(self.policy, "leaves", 0) - leaves0
        metrics = solo_metrics(episodes, wall, searches, searches * self.cfg.sims, leaves)
        metrics["target_source"] = self.target_source
        samples = [r.sample for ep in episodes for r in ep.records]
        return samples, metrics

    def _train(self, n_new: int) -> dict:
        if len(self.buffer) < self.cfg.min_buffer or n_new == 0:
            return {"steps": 0, "policy_loss": float("nan"), "value_loss": float("nan")}
        steps = self.mlb.train_steps or max(1, n_new // max(1, self.cfg.batch_size))
        steps = int(min(steps, self.mlb.max_train_steps))
        pl, vl = [], []
        for _ in range(steps):
            batch = self.buffer.sample(self.cfg.batch_size, rng=self.rng)
            metrics = self.cold.trainer.step(batch)
            self.counters.train_steps += 1
            pl.append(metrics["policy_loss"])
            vl.append(metrics["value_loss"])
        return {"steps": steps,
                "policy_loss": float(np.mean(pl)) if pl else float("nan"),
                "value_loss": float(np.mean(vl)) if vl else float("nan")}

    # ── Checkpointing ────────────────────────────────────────────────────────

    def state_dict(self, include_buffer: Optional[bool] = None) -> dict:
        sd = self.cold.state_dict(include_buffer=include_buffer)
        sd["mlb"] = {
            "config": {k: (list(v) if isinstance(v, tuple) else v)
                       for k, v in self.mlb.__dict__.items()},
            "generation": self.generation,
            "history": self.history.state_dict(),
            "clear_rate_ema": self.clear_rate_ema,
        }
        return sd

    def load_state_dict(self, ckpt: dict, *, strict_config: bool = True) -> None:
        self.cold.load_state_dict(ckpt, strict_config=strict_config)
        mlb = ckpt.get("mlb") or {}
        self.generation = int(mlb.get("generation", 0))
        self.clear_rate_ema = mlb.get("clear_rate_ema")
        if mlb.get("history"):
            self.history.load_state_dict(mlb["history"])
            self.history.capacity = self.mlb.p_history

    @classmethod
    def from_checkpoint(cls, ckpt: dict, overrides: Optional[dict] = None,
                        mlb_overrides: Optional[dict] = None) -> "MLBTrainer":
        """Rebuild a run from a checkpoint. `overrides` may change run-shaping fields
        (device, logging) but not the ones `ColdTrainer._check_config` pins; `mlb_overrides`
        is the same idea for the generation config."""
        from .loop import TrainConfig
        cfg_fields = {f: v for f, v in ckpt["config"].items()
                      if f in TrainConfig.__dataclass_fields__}
        cfg_fields.update(overrides or {})
        mlb_fields = dict((ckpt.get("mlb") or {}).get("config") or {})
        mlb_fields = {f: v for f, v in mlb_fields.items()
                      if f in MLBTrainConfig.__dataclass_fields__}
        if "sims_budgets" in mlb_fields:
            mlb_fields["sims_budgets"] = tuple(mlb_fields["sims_budgets"])
        mlb_fields.update(mlb_overrides or {})
        trainer = cls(TrainConfig(**cfg_fields), MLBTrainConfig(**mlb_fields))
        trainer.load_state_dict(ckpt)
        return trainer
