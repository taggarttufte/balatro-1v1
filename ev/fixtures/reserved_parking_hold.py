"""
reserved_parking_hold.py -- W-PROBE fixture 4/6 (PHASE5_V2_BRIEF_2026-08.md section 7):
"Reserved Parking + faces worth holding."

``build()``: player 0 has a made pair of Jacks (11C/11D, indices 0/1) that alone clears the
ante-1 Small blind (chips_target 50), plus five filler cards (2S/3D/5C/7H/9S, indices 2-6...
wait -- see ``SPEC``, six filler cards indices 2-7) chosen with no two of the same rank and no
run of 5 consecutive ranks (so nothing among them, or with the Jacks, can accidentally form a
second clearing structural set) and owns Reserved Parking (``j_reserved_parking``: 1-in-2
chance of $1 per face card HELD while a hand scores, ``card.lua:3302-3319``). Playing a single
weak filler card instead of the Jacks does not clear this hand, but it keeps both Jacks
HELD -- banking $1 of Parking money and delaying the (still near-certain, 3 hands left) clear
by one hand, which the extraction layer ranks above playing the Jacks now and losing that
held-face money (EXTRACT_NOTES.md section 2's ``proc_hold`` / ``_play_extraction``).

``build_control()``: the SAME hand shape without Reserved Parking -- "procs absent." A weak
filler play no longer banks anything, so playing the Jacks to clear now wins outright.

Verified ordering (``mp/ev/tests/test_probe_fixtures.py``, fast budget):
  sandbag:  play [7]  (lone filler, holds both Jacks, extract $1.00 Parking) ranks ABOVE
            play [0,1] (Jacks, clears now) -- the Jacks line ranks 9th, all 8 lines above it
            extract Parking money.
  control:  play [0,1] (Jacks, clears now) ranks ABOVE play [7].
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

SEED = "PARKHOLD"
SPEC = [(11, "Clubs"), (11, "Diamonds"), (2, "Spades"), (3, "Diamonds"),
       (5, "Clubs"), (7, "Hearts"), (9, "Spades"), (10, "Diamonds")]
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
    set_jokers(g0, "j_reserved_parking")
    return m


def build_control(seed: str = SEED) -> MLBMatch:
    m, g0 = _base(seed)
    set_jokers(g0)                     # no Reserved Parking -- nothing to extract
    return m
