"""
loop.py — the cold-start training loop as an object, so the script and the tests drive
exactly the same code path.

`scripts/train_cold.py` is a CLI + logging shell around `ColdTrainer`; the checkpoint
round-trip test (`tests/test_checkpoint.py`) trains 3 episodes, saves, rebuilds from the
checkpoint, trains 1 more, and compares against an uninterrupted 4-episode run. That
comparison is only meaningful if both go through the same `run_episode`, which is why
the loop lives here and not in the script.

Determinism contract
--------------------
ONE `numpy.random.Generator` drives everything the trainer chooses: the per-episode game
seed, the Gumbel noise inside MCTS, and the replay-batch indices. The engine itself is
deterministic given its seed (Phase 1: 126/126 exact), and Adam is deterministic. So on
CPU, a run is a pure function of `(TrainConfig, number of episodes)` and a resume from a
checkpoint that carries {model, optimizer, counters, rng, buffer} is **bit-exact** — that
is what the round-trip test asserts.

On CUDA the forward/backward kernels are not bit-reproducible across runs by default
(non-deterministic reduction order), so the resumed run is *statistically* the same but
not bit-identical; the test therefore runs on CPU and AGENT_NOTES says so.
"""
from __future__ import annotations

import time
import traceback
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import torch

from balatro_sim.game import BalatroGame
from mcts import MCTSConfig, MCTSPlayer, PolicyValueNet, get_encoder  # noqa: F401
from mcts.outcome import MLBOutcome, OutcomeFn, VanillaOutcome
from mcts.policy import make_policy

from .agent import SelfPlayAgent
from .sample import SampleBuilder
from .checkpoint import (
    global_rng_state, load_global_rng_state, numpy_rng_state, set_numpy_rng_state,
)
from .trainer import Trainer
from .trajectory import ReplayBuffer


@dataclass
class TrainConfig:
    seed: int = 0
    # Search
    sims: int = 30
    max_considered: int = 8
    # Optimisation
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.0
    # NOTE (measured): one Sample is ~97 KB, dominated by action_features (N, 56) with
    # N ~ 436 legal actions at SELECTING_HAND. 200_000 samples (the balatro-mcts default)
    # would be ~19 GB of RAM and an unshippable checkpoint, so the default is 20_000
    # (~1.9 GB). See AGENT_NOTES.md "Sample size" for the structural fix (subsample the
    # action set per sample) that would make a big buffer affordable.
    buffer_capacity: int = 20_000
    min_buffer: int = 128
    steps_per_episode: int = 1
    # Net
    hidden: int = 512
    n_res_blocks: int = 4
    policy_hidden: int = 128
    # Value targets are in [0, 1] for every OutcomeFn, so the head's range should be too.
    # "linear" is the pre-2026-08-22 unbounded head; a checkpoint that predates the field
    # is rebuilt as "linear" by `from_checkpoint`, keeping its trained semantics.
    value_activation: str = "sigmoid"    # sigmoid | clamp | linear
    encoder: str = "v7"          # "v7" (447) | "mlb" (453) | "set" (Phase 4 W1)
    device: str = "cpu"
    # Phase 4 W1 — action subsampling (brief §0.2). `subsample=False` reproduces the
    # Phase 3 full-action-set Sample; `encoder="set"` implies `Sample` v2 either way.
    subsample: bool = True
    k_unvisited: int = 8
    set_res_blocks: int = 2      # trunk depth of SetPolicyValueNet (encoder == "set")
    # Game
    ruleset: str = "vanilla"     # "vanilla" | "mlb"
    deck_key: str = "b_red"
    stake: int = 1
    lives: int = 4               # MLB only (MLBOutcome normalisation)
    max_decisions: int = 2000
    max_antes: Optional[int] = None    # MLB is endless: cap self-play episodes
    # Checkpointing
    checkpoint_buffer: bool = True
    buffer_checkpoint_cap: int = 5_000


@dataclass
class Counters:
    episodes: int = 0
    samples: int = 0
    train_steps: int = 0
    wins: int = 0
    errors: int = 0
    elapsed_sec: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


class ColdTrainer:
    """Owns the net, the self-play agent, the buffer, the optimizer and the RNG."""

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)

        self.encoder = get_encoder(cfg.encoder)
        if self.encoder.is_set:
            # The set net carries its own item encoders + attention block, so it uses its
            # OWN trunk depth (`set_res_blocks`, default 2) rather than the flat net's 4:
            # matching 4 would push it to ~2.9M params for no reason. `hidden` and
            # `policy_hidden` are shared so the CLI knobs mean the same thing on both.
            from mcts.model_set import SetPolicyValueNet
            self.net = SetPolicyValueNet(
                caps=self.encoder.caps, hidden=cfg.hidden,
                n_res_blocks=cfg.set_res_blocks, policy_hidden=cfg.policy_hidden,
                value_activation=cfg.value_activation,
            )
        else:
            self.net = PolicyValueNet(
                obs_dim=self.encoder.dim,
                hidden=cfg.hidden,
                n_res_blocks=cfg.n_res_blocks,
                policy_hidden=cfg.policy_hidden,
                value_activation=cfg.value_activation,
            )
        self.policy = make_policy(self.net, device=cfg.device, encoder=self.encoder,
                                  batched=False)
        self.mcts_cfg = MCTSConfig(
            num_simulations=cfg.sims,
            gumbel_max_considered=cfg.max_considered,
        )
        self.sample_builder = (
            SampleBuilder(self.encoder, k_unvisited=cfg.k_unvisited,
                          subsample=cfg.subsample, rng=self.rng)
            if (cfg.subsample or self.encoder.is_set) else None
        )
        self.agent = SelfPlayAgent(
            self.policy, self.mcts_cfg, rng=self.rng,
            max_decisions=cfg.max_decisions, encoder=self.encoder,
            outcome=self.make_outcome(), max_antes=cfg.max_antes,
            sample_builder=self.sample_builder,
        )
        self.buffer = ReplayBuffer(capacity=cfg.buffer_capacity)
        self.trainer = Trainer(self.net, lr=cfg.lr, device=cfg.device,
                               weight_decay=cfg.weight_decay)
        self.counters = Counters()

    # ── Pieces ───────────────────────────────────────────────────────────────

    def make_outcome(self) -> OutcomeFn:
        if self.cfg.ruleset == "mlb":
            return MLBOutcome(starting_lives=self.cfg.lives)
        return VanillaOutcome()

    def make_game(self, seed: int) -> BalatroGame:
        return BalatroGame(seed=seed, deck_key=self.cfg.deck_key,
                           stake=self.cfg.stake, ruleset=self.cfg.ruleset)

    # ── One episode + its training step(s) ───────────────────────────────────

    def run_episode(self) -> dict:
        """Self-play one episode, push it to the buffer, take `steps_per_episode`
        optimizer steps. Returns a JSONL-ready record. Never raises: a crashed episode
        is recorded as {"kind": "error"} and the loop continues (simulations still
        occasionally reach untested engine paths)."""
        cfg = self.cfg
        self.counters.episodes += 1
        ep = self.counters.episodes
        episode_seed = int(self.rng.integers(0, 2**31 - 1))
        t_ep = time.perf_counter()

        try:
            game = self.make_game(episode_seed)
            result = self.agent.play_episode(game)
        except Exception as e:                        # noqa: BLE001 - deliberate
            self.counters.errors += 1
            return {
                "kind": "error", "ep": ep, "seed": episode_seed,
                "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc(),
            }

        t_play = time.perf_counter() - t_ep
        self.buffer.extend(result.samples)
        self.counters.samples += len(result.samples)
        if result.won:
            self.counters.wins += 1

        metrics = None
        t_train = 0.0
        if len(self.buffer) >= cfg.min_buffer and result.samples:
            t_tr = time.perf_counter()
            for _ in range(cfg.steps_per_episode):
                batch = self.buffer.sample(cfg.batch_size, rng=self.rng)
                metrics = self.trainer.step(batch)
                self.counters.train_steps += 1
            t_train = time.perf_counter() - t_tr

        return {
            "kind": "episode",
            "ep": ep,
            "seed": episode_seed,
            "ante": result.final_ante,
            "won": result.won,
            "shaped_z": result.z,
            "stop": result.stop_reason,
            "len": len(result.samples),
            "searches": result.searches,
            "t_play": t_play,
            "t_train": t_train,
            "buf": len(self.buffer),
            "metrics": metrics,
        }

    # ── Checkpointing ────────────────────────────────────────────────────────

    def state_dict(self, include_buffer: Optional[bool] = None) -> dict:
        include_buffer = self.cfg.checkpoint_buffer if include_buffer is None else include_buffer
        return {
            "config": asdict(self.cfg),
            "net_desc": self.net.describe(),
            "encoder": self.cfg.encoder,
            "net_kind": "set" if self.encoder.is_set else "flat",
            "encoder_caps": (self.encoder.caps.as_dict() if self.encoder.is_set else None),
            "model": self.net.state_dict(),
            "trainer": self.trainer.state_dict(),
            "counters": self.counters.as_dict(),
            "rng": {
                "numpy": numpy_rng_state(self.rng),
                **global_rng_state(self.cfg.device),
            },
            "buffer": (self.buffer.state_dict(max_items=self.cfg.buffer_checkpoint_cap)
                       if include_buffer else None),
        }

    def load_state_dict(self, ckpt: dict, *, strict_config: bool = True) -> None:
        if strict_config:
            self._check_config(ckpt)
        self.net.load_state_dict(ckpt["model"])
        self.trainer.load_state_dict(ckpt["trainer"])
        self.counters = Counters(**ckpt["counters"])
        rng = ckpt.get("rng") or {}
        if "numpy" in rng:
            set_numpy_rng_state(self.rng, rng["numpy"])
        load_global_rng_state(rng)
        if ckpt.get("buffer"):
            self.buffer.load_state_dict(ckpt["buffer"])

    def _check_config(self, ckpt: dict) -> None:
        """A resume that quietly changes the observation, the net shape or the game is a
        different experiment. Warn-worthy fields are ignored; these are hard errors."""
        old = ckpt.get("config", {})
        for field_name in ("encoder", "ruleset", "deck_key", "stake", "hidden",
                           "n_res_blocks", "policy_hidden", "set_res_blocks",
                           "value_activation"):
            if field_name in old and getattr(self.cfg, field_name) != old[field_name]:
                raise ValueError(
                    f"checkpoint was written with {field_name}={old[field_name]!r}, "
                    f"resuming with {getattr(self.cfg, field_name)!r}. Pass the same value "
                    "or start a new run."
                )
        # Phase 4 W1: the observation's SHAPE also lives in the caps and the net kind.
        want_kind = "set" if self.encoder.is_set else "flat"
        got_kind = ckpt.get("net_kind", "flat")
        if got_kind != want_kind:
            raise ValueError(f"checkpoint holds a {got_kind!r} net, this run is {want_kind!r}")
        want_caps = self.encoder.caps.as_dict() if self.encoder.is_set else None
        got_caps = ckpt.get("encoder_caps")
        if got_kind == "set" and got_caps and got_caps != want_caps:
            raise ValueError(
                f"checkpoint was written with encoder caps {got_caps}, resuming with "
                f"{want_caps}. The padding width is part of the observation.")

    @classmethod
    def from_checkpoint(cls, ckpt: dict, overrides: Optional[dict] = None) -> "ColdTrainer":
        """Rebuild a trainer from a checkpoint. `overrides` may change run-shaping fields
        (device, minutes-equivalents, log settings) but not the ones `_check_config` pins."""
        cfg_fields = {f: v for f, v in ckpt["config"].items()
                      if f in TrainConfig.__dataclass_fields__}
        # A checkpoint written before the bounded value head carries neither the config
        # field nor the net-description key; it was trained unbounded, so resume it that
        # way instead of silently reinterpreting its value-head weights.
        if "value_activation" not in ckpt["config"]:
            cfg_fields["value_activation"] = ckpt.get("net_desc", {}).get(
                "value_activation", "linear")
        cfg_fields.update(overrides or {})
        trainer = cls(TrainConfig(**cfg_fields))
        trainer.load_state_dict(ckpt)
        return trainer


def rolling_summary(records: list[dict]) -> dict:
    """Mean stats over a window of episode records (for the CLI's status line)."""
    eps = [r for r in records if r.get("kind") == "episode"]
    if not eps:
        return {}
    metrics = [r["metrics"] for r in eps if r.get("metrics")]
    return {
        "ante": float(np.mean([r["ante"] for r in eps])),
        "len": float(np.mean([r["len"] for r in eps])),
        "z": float(np.mean([r["shaped_z"] for r in eps])),
        "win_pct": 100.0 * float(np.mean([bool(r["won"]) for r in eps])),
        "policy_loss": float(np.mean([m["policy_loss"] for m in metrics])) if metrics else float("nan"),
        "value_loss": float(np.mean([m["value_loss"] for m in metrics])) if metrics else float("nan"),
    }
