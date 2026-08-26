"""Test bootstrap for ev: run the shared fork-guarded bootstrap so bare
``import <module>`` from ev/tests/*.py resolves here and balatro_sim is the engine fork."""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent          # ev
_MP_ROOT = _HERE.parent

for _p in (str(_MP_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _bootstrap  # noqa: E402,F401  (fork guard; raises loudly on the wrong balatro_sim)
