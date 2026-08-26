"""
Test bootstrap for tournament.

Mirrors engine/conftest.py's fork-guard, using the same plain sys.path-insertion style
every in-repo module uses (never a dotted ``tournament...`` absolute import, which would
depend on ``mp`` resolving as a namespace package from whatever the caller's cwd happens to
be).  Puts the repo root and scripts on sys.path, then imports balatro_sim through
``oracle.engine_parity.import_engine()`` — the same fork-guarded entry point
``scripts/mlb_match_demo.py`` uses — and re-asserts loudly that the fork under engine
is the one that won, so a stray BRL top-level ``balatro_sim`` can never silently shadow it.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent          # tournament
_MP_ROOT = _HERE.parent                          # repo root
_ENGINE_ROOT = _MP_ROOT / "engine"
_SCRIPTS_ROOT = _MP_ROOT / "scripts"

for _p in (str(_MP_ROOT), str(_SCRIPTS_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from oracle.engine_parity import import_engine  # noqa: E402

import_engine()   # raises loudly if a different balatro_sim already won the module cache


def _assert_fork_is_the_one_imported() -> None:
    import balatro_sim  # noqa: WPS433 (runtime import is the point)

    pkg_file = Path(balatro_sim.__file__).resolve()
    expected = _ENGINE_ROOT / "balatro_sim" / "__init__.py"
    if pkg_file != expected:
        raise RuntimeError(
            "tournament tests imported the wrong balatro_sim:\n"
            f"  got:      {pkg_file}\n"
            f"  expected: {expected}\n"
            "Run with `python -m pytest tournament/tests` from the repo root."
        )


_assert_fork_is_the_one_imported()
