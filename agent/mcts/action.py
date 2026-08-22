"""
action.py — Action encoding for MCTS.

Actions in the engine are dicts (e.g. {"type": "play", "cards": [0, 2, 4]}). MCTS
needs hashable keys to index children. We use tuples — easy to debug, trivially
swappable for ints later if perf needs it.

Fork note (2026-08-21): the fork engine's `legal_actions()` emits one action type the
balatro-mcts original did not know about — `reroll_boss` (BLIND_SELECT, Directors Cut /
Retcon voucher; `game.py:1371`). It is argument-free so it falls through the tail of
both functions unchanged; ACTION_TYPES in action_features.py carries it explicitly so it
is not featurized as the all-zero unknown vector.
"""
from __future__ import annotations
from typing import Tuple

# Tuple-based action key. Concrete shape varies by action type.
ActionKey = Tuple


def action_key(action: dict) -> ActionKey:
    """Convert an action dict from legal_actions() into a hashable tuple."""
    atype = action["type"]

    if atype in ("play", "discard"):
        return (atype, tuple(sorted(action["cards"])))

    if atype == "use_consumable":
        return (
            atype,
            action["consumable_idx"],
            tuple(sorted(action.get("target_cards", []))),
        )

    if atype == "buy":
        return (atype, action["item_idx"])

    if atype == "sell_joker":
        return (atype, action["joker_idx"])

    if atype == "pick_booster":
        return (atype, tuple(sorted(action["indices"])))

    # play_blind, skip_blind, reroll_boss, reroll, leave_shop, skip_booster, advance
    return (atype,)


def action_from_key(key: ActionKey) -> dict:
    """
    Reverse of action_key. Used when MCTS picks an action and needs to
    pass it to game.step().
    """
    atype = key[0]
    if atype in ("play", "discard"):
        return {"type": atype, "cards": list(key[1])}
    if atype == "use_consumable":
        return {"type": atype, "consumable_idx": key[1], "target_cards": list(key[2])}
    if atype == "buy":
        return {"type": atype, "item_idx": key[1]}
    if atype == "sell_joker":
        return {"type": atype, "joker_idx": key[1]}
    if atype == "pick_booster":
        return {"type": atype, "indices": list(key[1])}
    return {"type": atype}
