"""
Test bootstrap for eval (mirrors engine/conftest.py + tests/conftest.py).

Puts eval and the repo root on sys.path (bare ``import common`` / ``import eval_harness`` /
``import rho_decay`` from eval/tests/*.py, matching how tests/test_mlb_match_gate.py
bare-imports ``mlb_match_demo``), then imports ``common`` once so its own bootstrap runs
(repo root + ``scripts`` on sys.path, ``oracle.engine_parity.import_engine()``) and fails
loudly if a different ``balatro_sim`` has already won the process.
"""
from __future__ import annotations

import sys
from pathlib import Path

_EVAL_DIR = Path(__file__).resolve().parent      # eval
_MP_ROOT = _EVAL_DIR.parent                       # repo root
for _p in (str(_MP_ROOT), str(_EVAL_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import common as _common  # noqa: E402  (runs the fork-guarded bootstrap; raises if it fails)

_EXPECTED_ENGINE = _MP_ROOT / "engine" / "balatro_sim" / "__init__.py"


def _assert_fork_is_the_one_imported() -> None:
    import balatro_sim  # noqa: WPS433

    pkg_file = Path(balatro_sim.__file__).resolve()
    if pkg_file != _EXPECTED_ENGINE:
        raise RuntimeError(
            "eval tests imported the wrong balatro_sim:\n"
            f"  got:      {pkg_file}\n  expected: {_EXPECTED_ENGINE}\n"
            "Run with `python -m pytest eval/tests` from the repo root."
        )


_assert_fork_is_the_one_imported()


def pytest_report_header(config):
    return [f"eval: balatro_sim fork OK ({_EXPECTED_ENGINE})"]
