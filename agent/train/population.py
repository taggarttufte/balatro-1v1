"""
population.py — who plays in a generation's tournaments.

Phase 4 W2. The training objective is "beat the population you are in" (MP design doc §6:
population rank -> value target, pairwise outcome -> policy). That objective is only worth
anything if the population is HETEROGENEOUS — N copies of one policy on one seed produce N
identical games, an all-ties N x N matrix, and a value target with zero variance
(`tournament/matrix.py::tie_fraction` is the degeneracy detector for exactly this).

So a generation's population is built from three sources of variation, in order of how much
signal they carry:

1. **Past selves** — the last `p` checkpoints. This is the real diversity: a genuinely
   different policy, not a perturbation of the current one. Absent in generation 0.
2. **Search budget** — the same net at different `sims`. A 60-sim agent and a 20-sim agent
   are different players (the 60-sim one is the improvement operator applied harder), and
   the brief asks for budget heterogeneity explicitly.
3. **Root noise + rng seed** — Dirichlet noise at the root, one seed per agent. The cheapest
   axis and the only one available at generation 0. Under Gumbel selection the sampled
   top-k is redrawn per decision from the agent's own rng, so distinct seeds diverge even
   with `add_noise=False`; noise is switched on for the current-net agents anyway because
   they are the ones whose games become training data and exploration is the point.

...and one that is not diversity at all but a REFERENCE:

4. **Scripted anchors** (`anchor_frac`, default 0.25). Every one of the three axes above is
   the current net or a recent version of it, so a behaviour the whole lineage shares is
   invisible to the rank target. That is not hypothetical: the first anchor-free gate run
   converged on a population of skippers — 95-98% of Small/Big blinds skipped by every seat,
   at which point skipping TIES on rank and a cold agent has no reason to risk a life
   playing a blind it cannot clear. Anchors are `mlb_match_demo.ScriptedPlayer`s, which by
   construction never skip (`BLIND_SELECT -> play_blind`, mlb_match_demo.py:126), so a
   skip-everything policy has to out-score somebody who spent the ante building. They
   produce no samples; they only shape the matrix.

**Only current-net agents produce training samples** (`PopulationMember.is_current`). Past
selves are opponents; learning from their decisions would be off-policy imitation of a
worse net. They still count in the N x N matrix, which is exactly what makes the rank
target mean something.

The whole thing is a pure function of `(n, generation, history, cfg)` so a `--resume`
rebuilds the identical population — tested in `tests/test_train_mlb.py`.

W1 note: nothing here touches the observation encoding. Swapping in the set encoder changes
`load_policy(..., encoder=)` only, which is threaded through `PopulationConfig.encoder`.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np


def tournament_module():
    """`mp/tournament` as an importable package. `mp/tournament/bootstrap.py` puts `mp/` and
    `mp/scripts` on `sys.path` itself and imports the frozen engine through
    `oracle.engine_parity.import_engine()`, which refuses to run against a different
    `balatro_sim` — since `_bootstrap.py` / `conftest.py` already put `mp/engine` first,
    that check confirms the fork rather than fighting it."""
    mp_root = str(Path(__file__).resolve().parents[2])       # mp/
    if mp_root not in sys.path:
        sys.path.append(mp_root)
    import tournament                                        # noqa: WPS433
    from tournament import matrix, players, runner            # noqa: F401,WPS433
    return tournament


@dataclass(frozen=True)
class PopulationMember:
    """One seat in the tournament. `checkpoint is None` means "the live current net"."""
    idx: int
    name: str
    is_current: bool
    sims: int
    seed: int
    checkpoint: Optional[str] = None
    generation: Optional[int] = None      # which checkpoint generation this seat plays
    add_noise: bool = False
    kind: str = "net"                     # "net" | "anchor"
    spec: Optional[tuple] = None          # anchor only: the ScriptedPlayer kwargs, sorted

    @property
    def is_anchor(self) -> bool:
        return self.kind == "anchor"

    def describe(self) -> dict:
        return {"idx": self.idx, "name": self.name, "is_current": self.is_current,
                "sims": self.sims, "seed": self.seed, "checkpoint": self.checkpoint,
                "generation": self.generation, "add_noise": self.add_noise,
                "kind": self.kind, "spec": self.spec}


@dataclass
class PopulationConfig:
    n_agents: int = 16
    m_current: int = 8                 # seats held by the CURRENT net (the sample producers)
    p_history: int = 4                 # how many past checkpoints stay in the pool
    sims: int = 40                     # base search budget
    sims_budgets: tuple = (1.0, 0.5, 1.5)   # multipliers cycled over the seats
    noise_current: bool = True         # root Dirichlet noise on the current-net seats
    #: Fraction of the OPPONENT seats given to scripted anchors (never taken from
    #: `m_current`). 0 disables them. See the module docstring for why they exist.
    anchor_frac: float = 0.25
    anchor_specs: tuple = (
        # W4's transfer-spread reference build: plays every blind, one reroll a visit, buys
        # the first shop slot. Reaches ante ~3-4 and wins ~28% of Nemeses vs the vanilla
        # target, so it is a real bar rather than a punching bag.
        {"hand": "greedy", "rerolls_per_visit": 1, "buy_slot0": True},
        # A builder: no rerolls, opens the first booster and takes from it.
        {"hand": "greedy", "rerolls_per_visit": 0, "buy_slot0": True, "open_pack_slot": 0},
    )
    encoder: str = "mlb"
    device: str = "cpu"
    leaf_batch: int = 16               # BATCH_NOTES §7.1: the runner drives one agent at a time
    reuse: bool = True
    strategy: str = "gumbel"

    def __post_init__(self):
        if self.n_agents < 2:
            raise ValueError("n_agents must be >= 2 (an N x N matrix needs a pair)")
        if not 1 <= self.m_current <= self.n_agents:
            raise ValueError(f"m_current must be in [1, n_agents], got {self.m_current}")
        if self.p_history < 0:
            raise ValueError("p_history must be >= 0")


class CheckpointHistory:
    """The last `p` generation checkpoints, oldest first. Serialisable so `--resume`
    restores the exact opponent pool (a resumed run facing a different pool is a different
    experiment, and the value targets would not be comparable across the seam)."""

    def __init__(self, capacity: int = 4):
        self.capacity = int(capacity)
        self.entries: list[dict] = []        # {"path": str, "generation": int}

    def __len__(self) -> int:
        return len(self.entries)

    def add(self, path, generation: int) -> None:
        self.entries.append({"path": str(path), "generation": int(generation)})
        if self.capacity > 0:
            self.entries = self.entries[-self.capacity:]
        else:
            self.entries = []

    def existing(self) -> list[dict]:
        """Entries whose file is still on disk (`--keep-checkpoints` prunes old ones, and a
        pruned opponent must silently drop out rather than crash a 24 h run)."""
        return [e for e in self.entries if Path(e["path"]).is_file()]

    def state_dict(self) -> dict:
        return {"capacity": self.capacity, "entries": list(self.entries)}

    def load_state_dict(self, sd: dict) -> None:
        self.capacity = int(sd.get("capacity", self.capacity))
        self.entries = [dict(e) for e in sd.get("entries", [])]


def build_population(cfg: PopulationConfig, generation: int,
                     history: Optional[CheckpointHistory] = None,
                     base_seed: int = 0) -> list[PopulationMember]:
    """The `n_agents` seats for one generation. Deterministic in its arguments.

    Seats 0..m_current-1 are the current net (sample producers, root noise on, search
    budgets cycled through `sims_budgets`). The remaining seats are filled round-robin from
    the surviving history checkpoints, newest first; when there is no history yet
    (generation 0, or every past checkpoint pruned) they fall back to MORE current-net seats
    that are still opponents, not sample producers, and are given different budgets and
    seeds so the population is not N identical players. Documented fallback, not a silent
    one: `is_current=True, produces_samples=False` is not a thing — the fallback seats are
    `is_current=False, checkpoint=None`, which the instantiator reads as "the live net".
    """
    members: list[PopulationMember] = []
    budgets = cfg.sims_budgets or (1.0,)

    for i in range(cfg.m_current):
        mult = budgets[i % len(budgets)]
        members.append(PopulationMember(
            idx=i, name=f"cur{i}", is_current=True,
            sims=max(2, int(round(cfg.sims * mult))),
            seed=base_seed + 1_000 * generation + i,
            checkpoint=None, generation=generation,
            add_noise=cfg.noise_current,
        ))

    opponent_seats = list(range(cfg.m_current, cfg.n_agents))
    specs = tuple(cfg.anchor_specs or ())
    n_anchor = (min(len(opponent_seats), int(round(cfg.anchor_frac * cfg.n_agents)))
                if (cfg.anchor_frac > 0 and specs) else 0)
    anchor_seats = set(opponent_seats[-n_anchor:]) if n_anchor else set()

    pool = list(reversed(history.existing())) if history is not None else []
    if cfg.p_history > 0:
        pool = pool[:cfg.p_history]

    j = 0
    for i in opponent_seats:
        seed = base_seed + 1_000 * generation + i
        if i in anchor_seats:
            k = sorted(anchor_seats).index(i)
            spec = dict(specs[k % len(specs)])
            spec["name"] = f"anchor{i}"
            members.append(PopulationMember(
                idx=i, name=spec["name"], is_current=False, sims=0, seed=seed,
                checkpoint=None, generation=None, add_noise=False, kind="anchor",
                spec=tuple(sorted(spec.items()))))
            continue
        mult = budgets[(i + 1) % len(budgets)]
        sims = max(2, int(round(cfg.sims * mult)))
        if pool:
            entry = pool[j % len(pool)]
            j += 1
            members.append(PopulationMember(
                idx=i, name=f"gen{entry['generation']}#{i}", is_current=False,
                sims=sims, seed=seed, checkpoint=entry["path"],
                generation=entry["generation"], add_noise=False))
        else:
            members.append(PopulationMember(
                idx=i, name=f"cold{i}", is_current=False, sims=sims, seed=seed,
                checkpoint=None, generation=generation, add_noise=True))
    return members


class SkipCap:
    """`legal_filter` implementing `--max-skips-per-ante`.

    Reads the engine's own cumulative `game.skips` counter and remembers what it was when
    this ante started, so it needs no cooperation from the driver and survives tree reuse,
    resets and interleaving. Once the ante's allowance is spent, `skip_blind` is dropped
    from the candidate set; every other action is untouched.

    Why this exists (measured, TRAIN_NOTES sec.7.2): at ante <= 4 a cold net cannot clear a
    Big blind, so skipping is genuinely optimal against ANY population — scripted anchors
    included — and the policy converges on 97-99% skip before it ever learns to play a hand.
    The cap is a training-time constraint on the CANDIDATE SET, not an engine rule (real MLB
    lets you skip both blinds and `mp/engine` is frozen), and it is annealed away
    (`--skip-cap-anneal`) so the final policy is trained under the real rules.
    """

    def __init__(self, max_per_ante: int = 1):
        self.max_per_ante = int(max_per_ante)
        self.reset()

    def reset(self) -> None:
        self._ante = None
        self._base = 0

    def __call__(self, game, legal):
        ante = int(getattr(game, "ante", 1))
        skips = int(getattr(game, "skips", 0))
        if ante != self._ante:
            self._ante, self._base = ante, skips
        if skips - self._base < self.max_per_ante:
            return legal
        out = [a for a in legal if a.get("type") != "skip_blind"]
        return out or legal          # never hand back an empty candidate set


def instantiate(members: Sequence[PopulationMember], live_policy, *,
                device: str = "cpu", encoder: str = "mlb", leaf_batch: int = 16,
                reuse: bool = True, strategy: str = "gumbel", outcome=None,
                max_skips_per_ante: Optional[int] = None,
                policy_cache: Optional[dict] = None,
                heuristic_prior: float = 0.0, max_hand_candidates: int = 0,
                heuristic_tau: float = 0.5, heuristic_exact_top: int = 8,
                heuristic_discard_bias: float = 1.0,
                policy_for: Optional[Callable] = None) -> list:
    """Turn members into `mcts.MCTSPlayer`s, loading each distinct checkpoint at most once.

    W0 (2026-08-22): `heuristic_prior` / `max_hand_candidates` give every NET-driven
    seat (current and past-checkpoint alike) the same hand prior and candidate mask the
    vanilla warm-up trained under. Both sides of the population must run it: a past
    checkpoint searching without the prior is a different agent from the one whose
    weights were trained with it, and the rank target would be measuring the prior, not
    the net. Scripted anchors are unaffected (they never search).

    `live_policy` is the CURRENT net's `PolicyValueFn` — passed in rather than loaded from
    disk so the population always plays the net that is actually being trained (and so a
    generation costs one net in memory, not `m_current` copies of it).

    W1 (Phase 5, 2026-08-23): `policy_for(member) -> PolicyValueFn` overrides BOTH branches
    of that choice for the net-driven seats. It exists for the multi-process path, where a
    worker holds no net at all and every seat's policy is a `parallel.remote.RemotePolicy`
    pointed at the shared evaluator; `live_policy` is then unused and may be `None`.
    Scripted anchors never reach it. Default `None` = the behaviour above, unchanged.

    Import is function-local: `mp/agent/train` must stay importable without dragging the
    whole search stack in at module scope (`train` is imported by `mcts.load_policy`).
    """
    from mcts import MCTSConfig, MCTSPlayer, load_policy

    cache: dict = {} if policy_cache is None else policy_cache
    make_scripted = None
    players = []
    for m in members:
        if m.is_anchor:
            if make_scripted is None:
                make_scripted = tournament_module().players.make_scripted
            players.append(make_scripted(**dict(m.spec or ())))
            continue
        if policy_for is not None:
            policy = policy_for(m)
        elif m.checkpoint is None:
            policy = live_policy
        else:
            if m.checkpoint not in cache:
                cache[m.checkpoint] = load_policy(m.checkpoint, device=device,
                                                  batched=True, encoder=encoder)
            policy = cache[m.checkpoint]
        players.append(MCTSPlayer(
            legal_filter=(SkipCap(max_skips_per_ante)
                          if (max_skips_per_ante is not None and m.is_current) else None),
            policy=policy,
            config=MCTSConfig(num_simulations=m.sims, leaf_batch=leaf_batch,
                              heuristic_prior_weight=heuristic_prior,
                              heuristic_tau=heuristic_tau,
                              heuristic_exact_top=heuristic_exact_top,
                              heuristic_discard_bias=heuristic_discard_bias,
                              max_hand_candidates=max_hand_candidates),
            outcome=outcome,
            rng=np.random.default_rng(m.seed),
            strategy=strategy,
            add_noise=m.add_noise,
            reuse=reuse,
            leaf_batch=leaf_batch,
            no_action={"type": "advance"},     # the runner steps unconditionally
            name=m.name,
        ))
    return players


def population_summary(members: Sequence[PopulationMember]) -> dict:
    """One line for the generation log: how heterogeneous is this population, really."""
    ckpts = {m.checkpoint for m in members if m.checkpoint is not None}
    return {
        "n_agents": len(members),
        "n_current": sum(1 for m in members if m.is_current),
        "n_anchors": sum(1 for m in members if m.is_anchor),
        "n_checkpoint_opponents": sum(1 for m in members if m.checkpoint is not None),
        "distinct_checkpoints": len(ckpts),
        "sims_budgets": sorted({m.sims for m in members if not m.is_anchor}),
    }


__all__ = [
    "PopulationMember", "PopulationConfig", "CheckpointHistory",
    "build_population", "instantiate", "population_summary", "tournament_module",
    "SkipCap",
]
