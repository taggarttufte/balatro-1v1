"""
Test bootstrap for mp/replay.

Mirrors mp/tournament/conftest.py's fork-guard: puts mp/ on sys.path, imports balatro_sim
through ``oracle.engine_parity.import_engine()`` (the same fork-guarded entry point every
other mp/* test suite uses), then re-asserts loudly that the fork under mp/engine is the one
that won, so a stray BRL top-level ``balatro_sim`` (from the repo root) can never silently
shadow it.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent          # mp/replay
_MP_ROOT = _HERE.parent                          # mp/
_ENGINE_ROOT = _MP_ROOT / "engine"

if str(_MP_ROOT) not in sys.path:
    sys.path.insert(0, str(_MP_ROOT))

from oracle.engine_parity import import_engine  # noqa: E402

import_engine()   # raises loudly if a different balatro_sim already won the module cache


def _assert_fork_is_the_one_imported() -> None:
    import balatro_sim  # noqa: WPS433 (runtime import is the point)

    pkg_file = Path(balatro_sim.__file__).resolve()
    expected = _ENGINE_ROOT / "balatro_sim" / "__init__.py"
    if pkg_file != expected:
        raise RuntimeError(
            "mp/replay tests imported the wrong balatro_sim:\n"
            f"  got:      {pkg_file}\n"
            f"  expected: {expected}\n"
            "Run with `python -m pytest mp/replay/tests` from the repo root."
        )


_assert_fork_is_the_one_imported()


def pytest_report_header(config):
    return [f"mp/replay: balatro_sim fork OK ({_ENGINE_ROOT / 'balatro_sim'})"]
