"""
tarot_target_cycle.py -- W-PROBE fixture 6/6 (PHASE5_V2_BRIEF_2026-08.md section 7), upgraded
by W-CYCLE (ev/CYCLE_NOTES.md) from "same action, more EV" to a GENUINE RANK SWAP: "held
targeted tarot whose best targets are not yet in hand", where the throwaway DIG outranks the
immediate clear.

``build()``: player 0 holds the Sun tarot (``c_sun``: converts up to 3 cards to Hearts,
``consumables.TAROT_SUIT``) and exactly ONE real Heart -- the Ace (index 5). A Sun flush
wants ``flush_need - 3 = 2`` real Hearts (``hand.HandAnalysis._prep_tarot_wants``), so the
tarot is one card short of being usable and the missing one has to be drawn. The hand also
holds a Clubs flush (3C 5C 7C 9C JC, indices 0-4) worth 276 chips, which clears the blind
outright at ``chips_target`` 250, and two junk Spades (2S index 6, 8S index 7). Deliberately
no straight and no pair: the Clubs flush is the only made hand, so "clear now" is
unambiguous. The 250 target (rather than a token 50) is what keeps the DEEPER dig lines --
the ones that break the flush to draw five -- honestly unattractive, so the line that wins is
the one that gives up nothing but the discard.

The three lines that matter, all at the fast budget:

* **clear now** -- ``play [0,1,2,3,4]``. Clears, so the round ENDS: ``_play_continues`` is 0
  and the five cards it draws are shuffled straight back into the deck without ever carrying
  a tarot. The clear banks no dig at all.
* **the dig** -- ``discard [6,7]``. Costs one discard, keeps the whole Clubs flush (so
  P(clear) is untouched -- the floor still clears next hand) AND keeps the Ace of Hearts, and
  draws 2 fresh cards to look for the second Heart. This is the line that wins.
  ``_dig_lines`` emits it, and so does the ordinary junk-out-2 line -- deliberately: this
  fixture isolates the VALUATION, not the generator, so the swap does not depend on the new
  candidate source existing (``_dig_lines``' own contribution is measured in
  ev/CYCLE_NOTES.md section 3, and the very next junk-out-k line, ``[5,6,7]``, DOES throw
  the Ace of Hearts away).
* **the same-``m`` comparison** -- ``discard [5,6]``, which throws the Ace of Hearts and
  keeps 8S instead. Same discard cost, same floor, same 2 cards drawn -- and roughly an
  eighth of the dig value, because the Sun then needs BOTH Hearts out of a 2-card draw
  instead of one. It is not itself a generated candidate here; it is scored directly through
  ``HandAnalysis.extraction_ev``, which accepts any action.

``build_control()``: the SAME hand and blind with NO consumable held at all --
``HandAnalysis._tarot_wants`` is empty, ``extract_on`` is False, ``_dig_lines`` returns
``[]`` and the dig is never even proposed, so the ordering reverses and the immediate clear
is the best action.
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

__all__ = ["SEED", "SPEC", "SPEC_LOW_TARGET", "CHIPS_TARGET", "CLEAR_NOW", "DIG_LINE",
           "SAME_M_CONTROL", "build", "build_control", "build_low_grade_target"]

SEED = "TAROTCYC"
#: 0-4 a Clubs flush that clears; 5 the lone Heart the Sun is waiting on; 6-7 junk.
#: Ranks chosen so no straight window is filled and no rank repeats.
SPEC = [(3, "Clubs"), (5, "Clubs"), (7, "Clubs"), (9, "Clubs"), (11, "Clubs"),
        (14, "Hearts"), (2, "Spades"), (8, "Spades")]
#: the same board with the Sun's one held target downgraded from the Ace of Hearts to the
#: FOUR of Hearts -- same suit, same count, same everything the old per-COUNT `_cycle_ev`
#: could see, and a materially smaller dig because the Ace also covers the top grade tiers
#: (`build_low_grade_target`; used to isolate the per-TARGET half of the model).
SPEC_LOW_TARGET = [(3, "Clubs"), (5, "Clubs"), (7, "Clubs"), (9, "Clubs"), (11, "Clubs"),
                   (4, "Hearts"), (2, "Spades"), (8, "Spades")]
CHIPS_TARGET = 250

CLEAR_NOW = ("play", (0, 1, 2, 3, 4))
DIG_LINE = ("discard", (6, 7))
SAME_M_CONTROL = ("discard", (5, 6))


def _base(seed: str, spec=None):
    m = to_selecting_hand(seed)
    g0 = m.games[0]
    set_hand(g0, spec or SPEC)
    g0.current_blind.chips_target = CHIPS_TARGET
    g0.dollars = max(g0.dollars, 10)
    return m, g0


def build(seed: str = SEED) -> MLBMatch:
    m, g0 = _base(seed)
    g0.consumable_hand = ["c_sun"]     # Hearts target, one of the two needed in hand
    return m


def build_control(seed: str = SEED) -> MLBMatch:
    m, g0 = _base(seed)
    g0.consumable_hand = []            # no tarot held -- nothing to dig toward
    return m


def build_low_grade_target(seed: str = SEED) -> MLBMatch:
    """``build()`` with the held Heart downgraded from the Ace to the Four (SPEC_LOW_TARGET).
    Not registered as a ``fixture:`` name -- it is the matched arm of the per-TARGET test,
    not a scenario Tagg would advise on."""
    m, g0 = _base(seed, SPEC_LOW_TARGET)
    g0.consumable_hand = ["c_sun"]
    return m
