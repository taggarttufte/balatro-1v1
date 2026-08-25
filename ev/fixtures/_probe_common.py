"""
_probe_common.py -- shared construction helpers for the W-PROBE acceptance fixtures
(PHASE5_V2_BRIEF_2026-08.md section 7).

Every W-PROBE fixture is an ``MLBMatch`` halted at ante 1 / ``SELECTING_HAND`` for player 0,
built ENTIRELY through real engine/match APIs -- ``MLBMatch(seed=...)``, ``m.step(0,
{"type": "play_blind"})`` to enter the hand-decision state, then direct attribute writes on
the real ``BalatroGame`` (``.hand``, ``.deck``, ``.jokers``, card ``.seal`` / ``.enhancement``,
``.consumable_hand``, ``.current_blind.chips_target``) -- never a hand-forged legal-actions
dict. This mirrors ``mp/ev/tests/test_extraction.py``'s own ``_set_hand`` / ``_jokers``
helpers (W-EXTRACT's accepted pattern for constructed states: real ``JokerInstance``s, real
``Card`` objects pulled out of ``game.full_deck``, so the fixture survives any engine change
that keeps those same attributes meaningful) and ``fixtures/bloodstone_vs_invisible.py``'s use
of ``debug_add_joker`` -- it is simply the "no self-play needed" variant, because every
W-PROBE scenario is deliberately about a mid-blind HAND decision, not a shop/build state, so
there is no reason to spend a self-play run reaching one.

Nothing here is imported by any other workstream; this file and the six fixture modules that
use it are W-PROBE-owned (additive to ``fixtures/__init__.py``'s ``FIXTURES`` registry only).
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent            # mp/ev/fixtures
_EV = _HERE.parent                                  # mp/ev
for _p in (str(_EV),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _bootstrap  # noqa: E402,F401
from _bootstrap import MLBMatch, State  # noqa: E402
from balatro_sim.jokers.base import JokerInstance  # noqa: E402

__all__ = ["to_selecting_hand", "set_hand", "set_jokers"]

DEFAULT_LIVES = 4


def to_selecting_hand(seed: str, *, deck_key: str = "b_red", stake=1,
                      lives: int = DEFAULT_LIVES) -> MLBMatch:
    """A fresh ``MLBMatch`` on ``seed``, player 0 stepped through ``play_blind`` at the
    ante-1 Small blind -- lands at ``SELECTING_HAND`` with the engine's own default
    ``hands_left`` / ``discards_left`` / ``chips_target``, ready for the fixture to
    hand-edit.  Player 1 is left untouched (the advisor only reads ``match.games[0]``;
    ``player_view`` / ``opponent_public_block_lines`` still work off player 1's real,
    if untouched, state)."""
    m = MLBMatch(seed=seed, deck_key=deck_key, stake=stake, lives=lives)
    m.step(0, {"type": "play_blind"})
    if m.games[0].state != State.SELECTING_HAND:
        raise RuntimeError(f"to_selecting_hand({seed!r}): landed at "
                          f"{m.games[0].state} instead of SELECTING_HAND")
    return m


def set_hand(game, specs) -> None:
    """Put exactly the ``(rank, suit)`` cards in ``specs`` into ``game.hand``, pulled out of
    ``game.full_deck`` (so they are the SAME ``Card`` objects the engine's own scoring /
    seal / enhancement machinery reads -- ``full_deck``/``hand``/``deck`` share references,
    per ``game.py``'s own comment), the rest of the deck left as the draw pile, discard pile
    cleared, every dealt card flipped face up and un-debuffed.  Identical to
    ``test_extraction.py``'s ``_set_hand``."""
    pool = {}
    for c in game.full_deck:
        pool.setdefault((c.rank, c.suit), c)
    hand = [pool[s] for s in specs]
    ids = {id(c) for c in hand}
    game.deck = [c for c in game.full_deck if id(c) not in ids]
    game.hand = hand
    game.discard_pile = []
    for c in hand:
        c.face_down = False
        c.debuffed = False


def set_jokers(game, *keys: str) -> None:
    """Replace ``game.jokers`` with fresh ``JokerInstance``s for ``keys``, left to right --
    identical to ``test_extraction.py``'s ``_jokers``."""
    game.jokers = [JokerInstance(k) for k in keys]
