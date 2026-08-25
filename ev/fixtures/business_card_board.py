"""
business_card_board.py -- W-PROBE fixture 3/6 (PHASE5_V2_BRIEF_2026-08.md section 7):
"Business Card + playable faces."

``build()``: player 0 has a made pair of Aces (14S/14H, indices 0/1, clears the ante-1 Small
blind at chips_target 50) plus a lone Jack (11C, index 2) and five low filler cards
(2D/3H/4C/5D/6S, indices 3-7) chosen so the Jack cannot join a straight/flush with them
(non-consecutive-enough ranks -- see the module-level ``SPEC`` comment) -- so playing the Jack
ALONE is a genuinely weak, non-clearing play. Business Card (``j_business``: 1-in-2 chance of
$2 per scored face, ``card.lua:3175-3184``) makes that weak play the top-ranked action anyway:
the $2 outweighs the delay, because the tail still clears almost certainly with the Ace pair
in hand and 3 hands left (EXTRACT_NOTES.md section 2/4).

``build_control()``: the SAME hand shape without Business Card -- "procs absent." Playing the
lone Jack no longer pays anything, so the immediate Ace-pair clear wins outright.

Verified ordering (``mp/ev/tests/test_probe_fixtures.py``, fast budget):
  sandbag:  play [2]  (lone Jack, extract $1.00 Business Card) ranks ABOVE play [0,1] (aces,
            clears now)
  control:  play [0,1] (aces, clears now) ranks ABOVE play [2].

Note: (14S,14H)+(2,3,4,5,6) also contains the Ace-low straight A-2-3-4-5 as a candidate
(``play [0,3,4,5,6]``) -- it clears too (tied with the Ace pair at the no-money EV) but never
includes the Jack, so it does not compete with the Business Card line either way; left in
deliberately as an honest "another clearing line exists, it just isn't the one being tested."
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
from _probe_common import to_selecting_hand, set_hand, set_jokers  # noqa: E402

__all__ = ["SEED", "SPEC", "CHIPS_TARGET", "build", "build_control"]

SEED = "BIZCARDX"
SPEC = [(14, "Spades"), (14, "Hearts"), (11, "Clubs"), (2, "Diamonds"), (3, "Hearts"),
       (4, "Clubs"), (5, "Diamonds"), (6, "Spades")]
CHIPS_TARGET = 50


def _base(seed: str):
    m = to_selecting_hand(seed)
    g0 = m.games[0]
    set_hand(g0, SPEC)
    g0.current_blind.chips_target = CHIPS_TARGET
    g0.dollars = max(g0.dollars, 10)
    return m, g0


def build(seed: str = SEED) -> MLBMatch:
    m, g0 = _base(seed)
    set_jokers(g0, "j_business")
    return m


def build_control(seed: str = SEED) -> MLBMatch:
    m, g0 = _base(seed)
    set_jokers(g0)                     # no Business Card -- nothing to extract
    return m
