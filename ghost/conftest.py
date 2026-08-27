"""
Test bootstrap for ghost — mirrors replay/conftest.py's fork-guard: repo root on
sys.path, engine imported through ``oracle.engine_parity.import_engine()``, then a loud
re-assert that the frozen fork under engine/ is the balatro_sim that won.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent          # ghost
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
            "ghost tests imported the wrong balatro_sim:\n"
            f"  got:      {pkg_file}\n"
            f"  expected: {expected}\n"
            "Run with `python -m pytest ghost/tests` from the repo root."
        )


_assert_fork_is_the_one_imported()


def pytest_report_header(config):
    return [f"ghost: balatro_sim fork OK ({_ENGINE_ROOT / 'balatro_sim'})"]
