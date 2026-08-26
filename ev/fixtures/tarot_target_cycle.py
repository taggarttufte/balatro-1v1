"""
tarot_target_cycle.py -- W-PROBE fixture 6/6 (PHASE5_V2_BRIEF_2026-08.md section 7): "held
targeted tarot whose best targets are not yet in hand."

``build()``: player 0 holds the Sun tarot (``c_sun``: converts up to 3 cards to Hearts,
``consumables.TAROT_SUIT``) with ZERO real Hearts in hand -- a flush wants
``flush_need - 3 = 2`` real Hearts already held (``hand.HandAnalysis._prep_tarot_wants``) and
none are present, so every candidate that draws fresh cards has a non-zero chance of supplying
them. The hand is a Clubs flush (2C-6C, indices 0-4) that clears the ante-1 Small blind
(chips_target 50) plus three Spades (7S/8S/9S, indices 5-7). Playing the flush draws 5 fresh
cards; ``_cycle_ev`` prices the hypergeometric chance those 5 include 2+ real Hearts as a
fraction of the Sun's value (EXTRACT_NOTES.md section 6) -- first-order, since which SPECIFIC
card the Sun then lands on is not modelled.

``build_control()``: the SAME hand and blind with NO consumable held at all --
``HandAnalysis._tarot_wants`` is empty, so ``_cycle_ev`` is identically 0 for every candidate
(the cleanest "procs absent" control: unlike "targets already satisfied," which still
produces a non-zero cycle value for any candidate that discards away the satisfying cards,
holding nothing means the tarot machinery never engages at all).

Verified ordering (``ev/tests/test_probe_fixtures.py``, fast budget): the top-ranked
clearing PLAY in the sandbag carries a strictly positive cycle bonus (``extract $0.93``) over
the identical action's EV in the control (no bonus) -- the ordering claim here is on the
EXTRACTION TERM, not a rank swap between two different actions (there is no non-cycling
candidate of equal chip value to swap against; see PROBE_NOTES.md for the honest reading of
what this fixture does and does not demonstrate).
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

__all__ = ["SEED", "SPEC", "CHIPS_TARGET", "build", "build_control"]

SEED = "TAROTCYC"
SPEC = [(2, "Clubs"), (3, "Clubs"), (4, "Clubs"), (5, "Clubs"),
       (6, "Clubs"), (7, "Spades"), (8, "Spades"), (9, "Spades")]
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
    g0.consumable_hand = ["c_sun"]     # Hearts target, none in hand
    return m


def build_control(seed: str = SEED) -> MLBMatch:
    m, g0 = _base(seed)
    g0.consumable_hand = []            # no tarot held -- nothing to cycle toward
    return m
