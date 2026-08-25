"""Test bootstrap for mp/ev/active_poc — mirrors mp/ev/conftest.py.

These tests live under the package (not mp/ev/tests) on purpose: mp/ev's pytest.ini has
``testpaths = tests``, so the existing W5 gate collects exactly what it did before and this
POC cannot perturb it.  Run them with:

    python -m pytest mp/ev/active_poc/tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent          # mp/ev/active_poc/tests
_EV_ROOT = _HERE.parent.parent                   # mp/ev
_MP_ROOT = _EV_ROOT.parent                       # mp

for _p in (str(_MP_ROOT), str(_EV_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _bootstrap  # noqa: E402,F401
