"""
Test bootstrap for the mp/agent fork.

Puts mp/agent and mp/engine at the front of sys.path so that

    import mcts, train        -> mp/agent/{mcts,train}
    import balatro_sim        -> mp/engine/balatro_sim   (the FORK, never the BRL
                                 package at the repo root that `python -m pytest`
                                 from the repo root would otherwise put on sys.path)

pytest.ini's `pythonpath = . ../engine` does the same thing; this file is the belt to
that suspenders and additionally fails loudly if the wrong package won. Same pattern as
mp/engine/conftest.py.
"""
import sys
from pathlib import Path

_AGENT_ROOT = Path(__file__).resolve().parent
_ENGINE_ROOT = _AGENT_ROOT.parent / "engine"

for _root in (_ENGINE_ROOT, _AGENT_ROOT):        # agent ends up first
    _s = str(_root)
    if _s in sys.path:
        sys.path.remove(_s)
    sys.path.insert(0, _s)


def _assert_expected(module_name: str, expected: Path) -> None:
    mod = __import__(module_name)
    got = Path(mod.__file__).resolve()
    if got != expected:
        raise RuntimeError(
            f"mp/agent tests imported the wrong {module_name}:\n"
            f"  got:      {got}\n"
            f"  expected: {expected}\n"
            "Something earlier on sys.path (or an already-imported module) is shadowing "
            "the fork. Run with `python -m pytest mp/agent/tests` from the repo root, or "
            "`pytest` from inside mp/agent."
        )


_assert_expected("balatro_sim", _ENGINE_ROOT / "balatro_sim" / "__init__.py")
_assert_expected("mcts", _AGENT_ROOT / "mcts" / "__init__.py")
_assert_expected("train", _AGENT_ROOT / "train" / "__init__.py")
