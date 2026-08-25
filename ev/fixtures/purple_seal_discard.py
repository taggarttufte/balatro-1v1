"""
purple_seal_discard.py -- W-PROBE fixture 1/6 (PHASE5_V2_BRIEF_2026-08.md section 7, Tagg's
first sandbag scenario): "blind nearly cleared, hands to spare, purple-seal cards in hand."

``build()``: player 0 has a made pair of Aces (14S/14H, indices 0/1) that clears the ante-1
Small blind (chips_target 50) on its own, plus two Purple-sealed junk cards (5D/6D, indices
5/6) that are NOT part of any structural play -- discarding them creates a Tarot per card
(``card.lua:2242-2268``, EXTRACT_NOTES.md section 1) at zero cost to the eventual clear
(the aces are untouched). ``hands_left``/``discards_left`` are the engine's own ante-1
defaults (4/4) -- genuinely "hands to spare."

``build_control()``: the SAME hand shape (same Ace pair, same junk-tail cards) with no Purple
seals -- "procs absent" per the brief's control recipe.  With nothing to extract,
``extract_on`` is False (the fast zero-cost path, EXTRACT_NOTES section 3) and the top-ranked
action is simply the clearing Ace-pair play.

Verified ordering (``mp/ev/tests/test_probe_fixtures.py``, fast budget):
  sandbag:  discard [5,6]  (extract $8.00, two Tarots) ranks ABOVE play [0,1] (clear now)
  control:  play [0,1] (clear now) ranks ABOVE any discard -- the seals carry no reason to
            leave the aces alone, but with nothing to bank, clearing wins outright.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent            # mp/ev/fixtures
_EV = _HERE.parent                                  # mp/ev
for _p in (str(_EV), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _bootstrap  # noqa: E402,F401
from _bootstrap import MLBMatch  # noqa: E402  (type hints only)
from _probe_common import to_selecting_hand, set_hand  # noqa: E402

__all__ = ["SEED", "SPEC", "CHIPS_TARGET", "build", "build_control"]

SEED = "PSEALDIS"
SPEC = [(14, "Spades"), (14, "Hearts"), (2, "Clubs"), (3, "Clubs"),
       (4, "Clubs"), (5, "Diamonds"), (6, "Diamonds"), (7, "Diamonds")]
CHIPS_TARGET = 50    # well under the made Ace pair's score -- "blind nearly cleared"


def _base(seed: str):
    m = to_selecting_hand(seed)
    g0 = m.games[0]
    set_hand(g0, SPEC)
    g0.consumable_hand = []            # both consumable slots free for the Tarots
    g0.current_blind.chips_target = CHIPS_TARGET
    g0.dollars = max(g0.dollars, 10)
    return m, g0


def build(seed: str = SEED) -> MLBMatch:
    m, g0 = _base(seed)
    g0.hand[5].seal = "Purple"
    g0.hand[6].seal = "Purple"
    return m


def build_control(seed: str = SEED) -> MLBMatch:
    m, _g0 = _base(seed)               # no seals -- nothing to extract
    return m
