"""fixtures/ — Tagg-facing state builders for the advisor (Phase 5 rev 2, W6).

``FIXTURES`` maps a name (used as ``fixture:<name>`` in the advisor CLI) to a zero/kwarg
``build() -> MLBMatch`` callable.
"""
from __future__ import annotations

from .bloodstone_vs_invisible import build as bloodstone_vs_invisible

# W-PROBE (Phase 5 rev 2, PHASE5_V2_BRIEF_2026-08.md section 7): Tagg's six sandbag
# acceptance scenarios, each with a matched "greedy is right" control -- additive
# registration only, nothing above this line was touched.
from .purple_seal_discard import build as purple_seal_discard
from .purple_seal_discard import build_control as purple_seal_discard_control
from .faceless_discard import build as faceless_discard
from .faceless_discard import build_control as faceless_discard_control
from .business_card_board import build as business_card_board
from .business_card_board import build_control as business_card_board_control
from .reserved_parking_hold import build as reserved_parking_hold
from .reserved_parking_hold import build_control as reserved_parking_hold_control
from .gold_seal_weak_play import build as gold_seal_weak_play
from .gold_seal_weak_play import build_control as gold_seal_weak_play_control
from .tarot_target_cycle import build as tarot_target_cycle
from .tarot_target_cycle import build_control as tarot_target_cycle_control

FIXTURES = {
    "bloodstone_vs_invisible": bloodstone_vs_invisible,
    "purple_seal_discard": purple_seal_discard,
    "purple_seal_discard_control": purple_seal_discard_control,
    "faceless_discard": faceless_discard,
    "faceless_discard_control": faceless_discard_control,
    "business_card_board": business_card_board,
    "business_card_board_control": business_card_board_control,
    "reserved_parking_hold": reserved_parking_hold,
    "reserved_parking_hold_control": reserved_parking_hold_control,
    "gold_seal_weak_play": gold_seal_weak_play,
    "gold_seal_weak_play_control": gold_seal_weak_play_control,
    "tarot_target_cycle": tarot_target_cycle,
    "tarot_target_cycle_control": tarot_target_cycle_control,
}

__all__ = [
    "FIXTURES", "bloodstone_vs_invisible",
    "purple_seal_discard", "purple_seal_discard_control",
    "faceless_discard", "faceless_discard_control",
    "business_card_board", "business_card_board_control",
    "reserved_parking_hold", "reserved_parking_hold_control",
    "gold_seal_weak_play", "gold_seal_weak_play_control",
    "tarot_target_cycle", "tarot_target_cycle_control",
]
