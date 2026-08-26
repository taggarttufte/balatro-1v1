"""Test bootstrap for ev/encode — mirrors ev/conftest.py so ``python -m pytest ev/encode``
collects standalone.

This package is deliberately invisible to the ev suite the other workstreams gate on.
Staying off ``ev/pytest.ini``'s ``testpaths`` is not enough: ``testpaths`` only applies when
pytest is given no arguments, and the repo's own documented command is
``python -m pytest ev`` — an explicit argument, which makes pytest recurse into every
``test_*.py`` under ``ev/``, this package included.  So ``pytest_ignore_collect`` below
skips the whole directory unless ``ev/encode/pytest.ini`` is the session's rootdir, i.e.
unless someone asked for this package by name.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent          # ev/encode
_EV = _HERE.parent                                # ev
_ROOT = _EV.parent                                # repo root

for _p in (str(_ROOT), str(_ROOT / "eval"), str(_ROOT / "agent"), str(_EV), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _bootstrap  # noqa: E402,F401  (fork guard; raises loudly on the wrong balatro_sim)


def pytest_ignore_collect(collection_path, config):
    """Collect this package only when it is the session's rootdir.

    `python -m pytest ev/encode`  -> rootdir is ev/encode  -> collected.
    `python -m pytest ev`          -> rootdir is ev         -> skipped entirely.
    `python -m pytest`  (repo root)-> rootdir is the repo   -> skipped entirely.
    """
    try:
        return Path(config.rootpath).resolve() != _HERE
    except Exception:
        return False
