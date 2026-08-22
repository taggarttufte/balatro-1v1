"""
_bootstrap.py — sys.path setup + fork guard for the scripts in this directory.

`python mp/agent/scripts/<x>.py` puts THIS directory on sys.path[0], so every script can
`import _bootstrap` as its first import and get:

    mp/agent   on sys.path -> `import mcts`, `import train`
    mp/engine  on sys.path -> `import balatro_sim`   (the FORK)

and a loud failure if something else on sys.path shadowed either. Same contract as
mp/agent/conftest.py, which is what the test suite uses.
"""
from __future__ import annotations

import sys
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parent.parent
ENGINE_ROOT = AGENT_ROOT.parent / "engine"
MP_ROOT = AGENT_ROOT.parent

for _root in (ENGINE_ROOT, AGENT_ROOT):       # agent ends up first
    _s = str(_root)
    if _s in sys.path:
        sys.path.remove(_s)
    sys.path.insert(0, _s)


def _assert_expected(module_name: str, expected: Path) -> None:
    mod = __import__(module_name)
    got = Path(mod.__file__).resolve()
    if got != expected:
        raise RuntimeError(
            f"imported the wrong {module_name}:\n  got:      {got}\n  expected: {expected}\n"
            "Run the script by path (`python mp/agent/scripts/<x>.py`) so this directory "
            "is sys.path[0]."
        )


def check() -> None:
    _assert_expected("balatro_sim", ENGINE_ROOT / "balatro_sim" / "__init__.py")
    _assert_expected("mcts", AGENT_ROOT / "mcts" / "__init__.py")


check()
