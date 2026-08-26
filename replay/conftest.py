"""
Test bootstrap for replay.

Mirrors tournament/conftest.py's fork-guard: puts the repo root on sys.path, imports balatro_sim
through ``oracle.engine_parity.import_engine()`` (the same fork-guarded entry point every
other in-repo test suite uses), then re-asserts loudly that the fork under engine is the one
that won, so a stray top-level ``balatro_sim`` from anywhere else can never silently
shadow it.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent          # replay
_MP_ROOT = _HERE.parent                          # repo root
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
            "replay tests imported the wrong balatro_sim:\n"
            f"  got:      {pkg_file}\n"
            f"  expected: {expected}\n"
            "Run with `python -m pytest replay/tests` from the repo root."
        )


_assert_fork_is_the_one_imported()


def pytest_report_header(config):
    return [f"replay: balatro_sim fork OK ({_ENGINE_ROOT / 'balatro_sim'})"]
