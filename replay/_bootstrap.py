"""
_bootstrap.py — sys.path plumbing + fork-guarded engine import shared by every mp/replay
module.  Mirrors mp/tournament/bootstrap.py: puts mp/ on sys.path and imports the FROZEN
mp/engine fork through ``oracle.engine_parity.import_engine()`` (the same entry point
mp/tournament, mp/eval and mp/tests use), so a second call anywhere else in the process is a
no-op that confirms the same module, never a second copy.

mp/replay never imports mp/agent (no torch) or mp/tournament -- this file only reaches into
mp/engine (frozen, read-only) and mp/rng (frozen, read-only, via oracle.engine_parity's own
imports).  Nothing here edits engine or rng code.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))            # mp/replay
MP_ROOT = os.path.dirname(HERE)                               # mp/

if MP_ROOT not in sys.path:
    sys.path.insert(0, MP_ROOT)

from oracle.engine_parity import import_engine  # noqa: E402

_game_mod = import_engine()                       # refuses to run against the wrong balatro_sim
BalatroGame = _game_mod.BalatroGame
State = _game_mod.State
GameState = _game_mod.GameState

from balatro_sim.mlb_match import MLBMatch  # noqa: E402
from balatro_sim import game_keys  # noqa: E402
from balatro_sim import constants  # noqa: E402
from balatro_sim.hand_eval import evaluate_hand  # noqa: E402

MLB_STARTING_LIVES = constants.MLB_STARTING_LIVES
MLB_PVP_START_ROUND = constants.MLB_PVP_START_ROUND

__all__ = [
    "MP_ROOT", "BalatroGame", "State", "GameState", "MLBMatch",
    "game_keys", "constants", "evaluate_hand",
    "MLB_STARTING_LIVES", "MLB_PVP_START_ROUND",
]
