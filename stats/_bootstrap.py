"""
_bootstrap.py — sys.path plumbing + fork-guarded engine import shared by every mp/stats
module (Phase 5 rev 2).  Mirrors mp/replay/_bootstrap.py: puts mp/ on sys.path and imports
the FROZEN mp/engine fork through ``oracle.engine_parity.import_engine()`` (the same entry
point mp/tournament, mp/eval, mp/replay and mp/tests use), so a second call anywhere else in
the process is a no-op that confirms the same module, never a second copy.

Also puts mp/agent on sys.path so ``import mcts`` / ``import train`` resolve to
mp/agent/{mcts,train} (the W0 heuristic, the set encoder, the value net) without a dotted
``mp.agent...`` import.  Nothing here edits engine or rng code.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))            # mp/stats
MP_ROOT = os.path.dirname(HERE)                               # mp/
AGENT_ROOT = os.path.join(MP_ROOT, "agent")                   # mp/agent  (mcts/, train/)
ENGINE_ROOT = os.path.join(MP_ROOT, "engine")

for _p in (AGENT_ROOT, MP_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from oracle.engine_parity import import_engine  # noqa: E402

_game_mod = import_engine()                       # refuses to run against the wrong balatro_sim
BalatroGame = _game_mod.BalatroGame
State = _game_mod.State
GameState = _game_mod.GameState

from balatro_sim.mlb_match import MLBMatch  # noqa: E402
from balatro_sim import game_keys  # noqa: E402
from balatro_sim import constants  # noqa: E402
from balatro_sim.hand_eval import evaluate_hand  # noqa: E402

__all__ = [
    "MP_ROOT", "AGENT_ROOT", "ENGINE_ROOT", "BalatroGame", "State", "GameState", "MLBMatch",
    "game_keys", "constants", "evaluate_hand",
]
