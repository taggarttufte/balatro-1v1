"""
_bootstrap.py — sys.path plumbing + fork-guarded engine import for the ghost package.
Mirrors replay/_bootstrap.py: repo root on sys.path, engine imported through
``oracle.engine_parity.import_engine()`` so it is always THE frozen fork under engine/,
never a stray balatro_sim.

ghost/export.py needs only the deck catalogue (display names) from the engine; ghost/make.py
additionally reaches into ev/ (via ev/h2h.py's own sys.path header) to build players.
Nothing here edits engine or rng code.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))            # ghost
MP_ROOT = os.path.dirname(HERE)                               # repo root
if MP_ROOT not in sys.path:
    sys.path.insert(0, MP_ROOT)

from oracle.engine_parity import import_engine  # noqa: E402

_game_mod = import_engine()                       # refuses to run against the wrong balatro_sim
BalatroGame = _game_mod.BalatroGame
State = _game_mod.State

from balatro_sim.mlb_match import MLBMatch  # noqa: E402
from balatro_sim import decks  # noqa: E402

__all__ = ["MP_ROOT", "BalatroGame", "State", "MLBMatch", "decks"]
