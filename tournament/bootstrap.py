"""
bootstrap.py — sys.path plumbing shared by every tournament module.

Puts the repo root and scripts on sys.path and imports the FROZEN engine fork through
``oracle.engine_parity.import_engine()`` — the same fork-guarded entry point
``scripts/mlb_match_demo.py`` and ``tests/test_mlb_match_gate.py`` use, so a second
``import_engine()`` call inside ``mlb_match_demo`` (which it does at its own module scope)
is a no-op that confirms the same module, never a second copy.

engine/** and rng/** are FROZEN for Phase 3: this file only arranges imports so the
rest of tournament can do ``from .bootstrap import BalatroGame, State, mlb_match_demo``
etc.  Nothing here edits engine or rng code.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))            # tournament
MP_ROOT = os.path.dirname(HERE)                               # repo root
ENGINE_ROOT = os.path.join(MP_ROOT, "engine")
SCRIPTS_ROOT = os.path.join(MP_ROOT, "scripts")

for _p in (MP_ROOT, SCRIPTS_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from oracle.engine_parity import import_engine  # noqa: E402

_game_mod = import_engine()                       # refuses to run against the wrong balatro_sim
BalatroGame = _game_mod.BalatroGame
State = _game_mod.State

from balatro_sim.mlb_match import MLBMatch  # noqa: E402  (not driven directly by the runner,
# but useful for tests that want to sanity-check a tournament Nemesis against the canonical
# two-player coordinator on n_agents=2)
from balatro_sim.constants import (  # noqa: E402
    MLB_STARTING_LIVES, MLB_PVP_START_ROUND, MLB_COMEBACK_PER_LIFE,
)

import mlb_match_demo  # noqa: E402  (ScriptedPlayer, make_policy, key_position, classify_key,
# diff_rng, OPENER, REROLLER — imported, never copied, per the brief)

__all__ = [
    "MP_ROOT", "ENGINE_ROOT", "SCRIPTS_ROOT",
    "BalatroGame", "State", "MLBMatch",
    "MLB_STARTING_LIVES", "MLB_PVP_START_ROUND", "MLB_COMEBACK_PER_LIFE",
    "mlb_match_demo",
]
