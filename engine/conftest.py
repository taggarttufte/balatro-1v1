"""
Test bootstrap for the engine fork.

Puts engine at the front of sys.path so `import balatro_sim` resolves to the
fork under engine/balatro_sim, not any other ``balatro_sim`` on sys.path (which
`python -m pytest` run from the repo root would otherwise put on sys.path via the
cwd entry). pytest.ini's `pythonpath = .` does the same thing; this file is the
belt to that suspenders and additionally fails loudly if the wrong package won.
"""
import sys
from pathlib import Path

_ENGINE_ROOT = Path(__file__).resolve().parent
_ENGINE_ROOT_STR = str(_ENGINE_ROOT)

if _ENGINE_ROOT_STR in sys.path:
    sys.path.remove(_ENGINE_ROOT_STR)
sys.path.insert(0, _ENGINE_ROOT_STR)


def _assert_fork_is_the_one_imported() -> None:
    """Refuse to run the suite against a balatro_sim that is not this fork."""
    import balatro_sim  # noqa: WPS433 (runtime import is the point)

    pkg_file = Path(balatro_sim.__file__).resolve()
    expected = _ENGINE_ROOT / "balatro_sim" / "__init__.py"
    if pkg_file != expected:
        raise RuntimeError(
            "engine tests imported the wrong balatro_sim:\n"
            f"  got:      {pkg_file}\n"
            f"  expected: {expected}\n"
            "Something earlier on sys.path (or an already-imported module) is "
            "shadowing the fork. Run with `python -m pytest engine/tests` "
            "from the repo root or `pytest` from inside engine."
        )


_assert_fork_is_the_one_imported()
