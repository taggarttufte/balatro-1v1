"""Shared pytest setup for tests.

* Puts the repo root on ``sys.path`` so tests import the subproject as top-level
  packages (``from rng.core import PseudoRandom``), matching how the engine
  code in this repo imports it.
* Exposes whether the LuaJIT oracle (``lupa`` + extracted Balatro Lua under
  ``_reference/balatro_src``) is available on this machine.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

MP_ROOT = Path(__file__).resolve().parents[1]
if str(MP_ROOT) not in sys.path:
    sys.path.insert(0, str(MP_ROOT))

BALATRO_SRC = MP_ROOT / "_reference" / "balatro_src"
MISC_FUNCTIONS_LUA = BALATRO_SRC / "functions" / "misc_functions.lua"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def lupa_available() -> bool:
    try:
        from lupa import luajit21  # noqa: F401
    except Exception:
        return False
    return True


def oracle_available() -> bool:
    """True when ground truth can be regenerated live (lupa + game Lua present)."""
    return lupa_available() and MISC_FUNCTIONS_LUA.is_file()


@pytest.fixture(scope="session")
def mp_root() -> Path:
    return MP_ROOT


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES_DIR


def pytest_report_header(config):
    return [
        "mp oracle: lupa=%s balatro_src=%s"
        % ("yes" if lupa_available() else "no", "yes" if MISC_FUNCTIONS_LUA.is_file() else "no")
    ]
