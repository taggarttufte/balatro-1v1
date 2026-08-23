"""
protocol.py — the picklable messages that cross the process boundary.

Everything here has to survive ``spawn`` (Windows has no ``fork``): plain dataclasses of
plain types, no closures, no open handles, no torch tensors.  The two exceptions are
handed to ``Process(target=..., args=...)`` rather than put on a queue, which is the only
way ``multiprocessing`` can transfer them: the command/result ``Queue``s and the reply
``Connection``.

Command / result vocabulary (main -> worker on ``cmd_q[w]``, worker -> main on ``res_q``):

    generation   build this generation's players for the agents I own
    fanout       build my games for one tournament (+ arm the trajectory loggers)
    drive        drive these agents to their next Nemesis
    apply        lose_life / game_over / cash_out / done, for the agents I own
    summarize    lives / ante / jokers / chips for the agents I own
    collect      the RecordedDecisions from the tournament just finished
    stats        counters (searches, leaves, waits) since the last call
    shutdown     drain and exit 0
    crash        TEST ONLY: die immediately, without draining
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

#: policy_id of the net currently being trained.  Past-self checkpoints get 1, 2, ...
LIVE_POLICY_ID = 0

OP_GENERATION = "generation"
OP_FANOUT = "fanout"
OP_DRIVE = "drive"
OP_APPLY = "apply"
OP_SUMMARIZE = "summarize"
OP_COLLECT = "collect"
OP_STATS = "stats"
OP_SHUTDOWN = "shutdown"
OP_CRASH = "crash"

#: Exit code a worker uses for the test-only ``crash`` op, so a real failure (an
#: exception, exit 1) is never mistaken for the deliberate one.
CRASH_EXIT_CODE = 70


@dataclass(frozen=True)
class WorkerSpec:
    """Everything a worker needs before it can take its first command."""
    worker_id: int
    roots: tuple                    # (engine_root, agent_root, mp_root) for sys.path
    arena: Any                      # channel.ArenaHandle, or None in "local" mode
    layout: Any                     # layout.LeafLayout (built once from a real leaf)
    encoder: str = "set"
    caps: Optional[dict] = None
    hand_type_features: bool = True
    mode: str = "remote"            # "remote" (shared evaluator) | "local" (own net)
    device: str = "cpu"             # local mode only: where the worker runs the net
    torch_threads: int = 1
    poll_seconds: float = 120.0     # how long a worker waits for a reply before giving up
    max_action_rows: int = 0        # local mode only; 0 = the policy's own default


@dataclass
class GenerationSpec:
    """The population, as a worker needs to rebuild its slice of it."""
    generation: int = 0
    members: tuple = ()             # PopulationMember, ONLY the ones this worker owns
    policy_ids: dict = field(default_factory=dict)     # agent idx -> policy_id
    checkpoints: dict = field(default_factory=dict)    # policy_id -> checkpoint path
    weights_path: Optional[str] = None                 # local mode: the live net
    encoder: str = "set"
    device: str = "cpu"
    leaf_batch: int = 1
    reuse: bool = True
    strategy: str = "gumbel"
    starting_lives: int = 4
    horizon_antes: int = 4
    max_skips_per_ante: Optional[int] = None
    heuristic: dict = field(default_factory=dict)
    # SampleBuilder: subsampling is RNG-driven, and in the serial path that RNG is the
    # trainer's shared generator.  A worker cannot share it, so each agent gets its own
    # stream seeded from (sample_seed, generation, agent idx) -- deterministic in the
    # worker COUNT, which the shared generator would not have been.  See PARALLEL_NOTES §5.
    sample: Optional[dict] = None
    sample_seed: int = 0
    record_current: bool = True
    max_samples_per_agent: int = 0


@dataclass
class TournamentSetup:
    """One tournament's fan-out, as a worker needs it."""
    seed_str: str = ""
    deck_key: str = "b_red"
    stake: int = 1
    lives: int = 4
    ruleset: str = "mlb"
    fanout: str = "clone"
    n_agents: int = 16
    life_rule: str = "paired"
    max_ante: int = 4
    #: Trajectory logging (mp/replay).  ``None`` = off.  ``path`` is per worker; the main
    #: process concatenates the parts at the end of the generation.
    traj: Optional[dict] = None


__all__ = [
    "WorkerSpec", "GenerationSpec", "TournamentSetup", "LIVE_POLICY_ID",
    "OP_GENERATION", "OP_FANOUT", "OP_DRIVE", "OP_APPLY", "OP_SUMMARIZE",
    "OP_COLLECT", "OP_STATS", "OP_SHUTDOWN", "OP_CRASH", "CRASH_EXIT_CODE",
]
