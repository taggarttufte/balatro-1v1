"""
_util.py — shared helpers used by log.py / replay.py / tags.py / export_viz.py:
per-step summaries, process-independent signature digests, and the synthetic-op dispatch
that lets a step list carry engine mutations that did not go through ``BalatroGame.step()``
(the tournament runner calls ``game.lose_life()`` directly -- see REPLAY_NOTES.md "Hook
contract" / "Tournament wiring").
"""
from __future__ import annotations

import hashlib
from typing import Any, Optional

# Synthetic action "type"s: recorded via the same log.step(game, action) call as a real
# action, but dispatched to a direct engine method instead of BalatroGame.step() on replay.
OP_LOSE_LIFE = "__lose_life__"
OP_SET_PVP_INFO = "__set_pvp_info__"   # solo external-target driver (P4-W2 selfplay.OP_SET_PVP_INFO)
SYNTHETIC_TYPES = (OP_LOSE_LIFE, OP_SET_PVP_INFO)


def summarize(game: Any, step: int) -> dict:
    """The per-step summary tuple from PHASE4_BRIEF §W3, as a dict: (step, state, ante,
    blind_kind, is_pvp, money, lives, chips_scored, hands_left, discards_left)."""
    blind = game.current_blind
    return {
        "step": step,
        "state": game.state.name,
        "ante": game.ante,
        "blind_kind": blind.kind,
        "is_pvp": bool(blind.is_pvp),
        "money": game.dollars,
        "lives": game.lives,
        "chips_scored": game.chips_scored,
        "hands_left": game.hands_left,
        "discards_left": game.discards_left,
    }


def sig_digest(game: Any) -> str:
    """Process-independent digest of ``game.state_signature()`` (which is itself already a
    hashable, Card-id-independent snapshot -- see game.py's own docstring).  Hashed rather
    than stored verbatim: the full signature repr is large (whole deck + shop + jokers) and
    a per-episode log stores one of these every ``sig_every`` steps, so hashing keeps the
    "few KB per episode" budget (PHASE4_BRIEF §W3) regardless of how deep a run goes."""
    return hashlib.blake2b(repr(game.state_signature()).encode(), digest_size=16).hexdigest()


def match_sig_digest(match: Any) -> str:
    """Same idea for ``MLBMatch.signature()`` (both games' state_signature() + match
    scalars + pvp_log already folded in by the engine)."""
    return hashlib.blake2b(repr(match.signature()).encode(), digest_size=16).hexdigest()


def apply_op(game: Any, action: dict) -> None:
    """Apply one recorded op to ``game`` during replay: a real action goes through
    ``BalatroGame.step()`` exactly as it did live; a synthetic op (currently only
    ``__lose_life__``, emitted when an orchestrator mutates the game directly -- the
    tournament runner's cross-agent life rule) is dispatched to the matching engine method
    instead, never to ``step()``."""
    atype = action.get("type")
    if atype == OP_LOSE_LIFE:
        game.lose_life()
    elif atype == OP_SET_PVP_INFO:
        game.set_pvp_info(action["score"], action["hands"])
    else:
        game.step(action)


class ReplayMismatch(Exception):
    """Raised by replay()/replay_match() at the first step whose recorded signature does not
    match what re-running the action list produced."""

    def __init__(self, step: int, action: Optional[dict], expected: str, got: str,
                 final: bool = False):
        self.step = step
        self.action = action
        self.expected = expected
        self.got = got
        self.final = final
        label = "final signature" if final else f"step {step}"
        super().__init__(
            f"replay divergence at {label} (action={action!r}): "
            f"expected sig {expected}, got {got}"
        )
