"""
faceless_discard.py -- W-PROBE fixture 2/6 (PHASE5_V2_BRIEF_2026-08.md section 7): "Faceless
Joker + >= 3 face cards."

``build()``: player 0 has a made pair of Aces (14S/14H, indices 0/1, clears the ante-1 Small
blind at chips_target 50) plus three face cards -- Jack/Queen/King of three different suits,
indices 2/3/4 -- that are NOT part of any structural play, and owns Faceless Joker
(``j_faceless``: $5 for 3+ discarded faces, ``card.lua:2858-2872``). Discarding the three
faces is free with respect to the eventual clear (the aces are untouched, and Faceless money
is unconditional -- EXTRACT_NOTES.md section 2's ``_discard_extraction``).

``build_control()``: the SAME hand shape without Faceless Joker -- "procs absent." Discarding
the faces then banks nothing, so the clearing Ace-pair play wins outright.

Verified ordering (``mp/ev/tests/test_probe_fixtures.py``, fast budget):
  sandbag:  discard [2,3,4]  (extract $5.00, Faceless) ranks ABOVE play [0,1] (clear now)
  control:  play [0,1] (clear now) ranks ABOVE any discard of the same three faces.
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

SEED = "FACELESS"
SPEC = [(14, "Spades"), (14, "Hearts"), (11, "Clubs"), (12, "Diamonds"), (13, "Hearts"),
       (2, "Clubs"), (3, "Diamonds"), (4, "Spades")]
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
    set_jokers(g0, "j_faceless")
    return m


def build_control(seed: str = SEED) -> MLBMatch:
    m, g0 = _base(seed)
    set_jokers(g0)                     # no Faceless -- nothing to extract
    return m
