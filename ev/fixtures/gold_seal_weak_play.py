"""
gold_seal_weak_play.py -- W-PROBE fixture 5/6 (PHASE5_V2_BRIEF_2026-08.md section 7):
"gold-seal cards playable in a deliberately weak hand."

``build()``: player 0 has a made pair of Aces (14S/14H, indices 0/1) that alone clears the
ante-1 Small blind (chips_target 40), a King and Queen (indices 2/3, unrelated to the aces),
and four low filler cards (2C/5D/7H/9S, indices 4-7) chosen with no run of 5 consecutive ranks
including or excluding the Ace -- an earlier draft used 2/3/4/5/6 filler, which accidentally
built an Ace-low straight (A-2-3-4-5) that muddied the comparison; this spec has no such
run. The 2 of Clubs (index 4) carries a Gold seal ($3 flat when it scores as part of a played hand, ``card.lua:1068-1073``).
Playing that single low card ALONE is a deliberately weak play (2 chips, does not clear this
hand) that still ranks top: the flat $3 outweighs delaying the clear by one hand, with 3
hands left and the Ace pair untouched for later (EXTRACT_NOTES.md section 2).

``build_control()``: the SAME hand shape with no Gold seal anywhere -- "procs absent." The
weak single-card play now banks nothing and the Ace-pair clear wins outright.

Verified ordering (``ev/tests/test_probe_fixtures.py``, fast budget):
  sandbag:  play [4]  (lone Gold-sealed 2, extract $3.00) ranks ABOVE play [0,1] (aces,
            clears now)
  control:  play [0,1] (aces, clears now) ranks ABOVE play [4] (not even in the top 8).
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent            # ev/fixtures
_EV = _HERE.parent                                  # ev
for _p in (str(_EV), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _bootstrap  # noqa: E402,F401
from _bootstrap import MLBMatch  # noqa: E402  (type hints only)
from _probe_common import to_selecting_hand, set_hand  # noqa: E402

__all__ = ["SEED", "SPEC", "GOLD_IDX", "CHIPS_TARGET", "build", "build_control"]

SEED = "GOLDWEAK"
SPEC = [(14, "Spades"), (14, "Hearts"), (13, "Clubs"), (12, "Diamonds"),
       (2, "Clubs"), (5, "Diamonds"), (7, "Hearts"), (9, "Spades")]
GOLD_IDX = 4         # the 2 of Clubs
CHIPS_TARGET = 40


def _base(seed: str):
    m = to_selecting_hand(seed)
    g0 = m.games[0]
    set_hand(g0, SPEC)
    g0.current_blind.chips_target = CHIPS_TARGET
    g0.dollars = max(g0.dollars, 10)
    return m, g0


def build(seed: str = SEED) -> MLBMatch:
    m, g0 = _base(seed)
    g0.hand[GOLD_IDX].seal = "Gold"
    return m


def build_control(seed: str = SEED) -> MLBMatch:
    m, _g0 = _base(seed)               # no gold seal -- nothing to extract
    return m
