"""
action_features.py — featurize an action dict into a fixed-size float vector for the
pointer-style policy head. Each legal action becomes one row; the row is concatenated
with the trunk embedding and scored by the policy MLP.

Re-targeted to the fork engine (2026-08-21)
-------------------------------------------
The balatro-mcts original was a 44-dim vector over 12 action types with 8 hand slots,
2 consumables, 7 shop slots and 5 joker slots — the V7 action space. The fork's
`legal_actions()` (mp/engine/balatro_sim/game.py:1342) can emit indices past every one
of those bounds, and an out-of-bounds index used to be silently dropped, which makes two
DIFFERENT actions share one all-but-identical feature row and therefore one prior:

  * `reroll_boss` — a 13th action type (Directors Cut / Retcon voucher). The original
    would have featurized it as the all-zero "unknown type" row.
  * hand indices — `hand_size` is 8 + Juggler/Turtle Bean/Ouija/Ectoplasm/voucher deltas
    (game.py:993-1026); plays and discards index into the LIVE hand, so 8 is not a bound.
  * consumables — `consumable_slots` grows with Crystal Ball and Negative editions.
  * jokers — `joker_slots` grows with Negative jokers / Antimatter; `sell_joker` indexes
    the live list.
  * shop — 4 shelf slots (Overstock/Overstock Plus) + 2 packs + 1 voucher = 7 today, 8
    with headroom.

Bounds here are therefore generous, and anything still past them sets the shared
`overflow` bit plus a normalized-index scalar, so two overflowing actions remain
distinguishable. Dim went 44 -> 56; the net is cold-start so nothing depended on 44.

Not addressed here (inherited, documented in AGENT_NOTES §5): the *observation* still
shows only the first 8 hand slots and 5 joker slots — that is env_v7's encoder, which is
frozen in Phase 3.
"""
from __future__ import annotations
import numpy as np

ACTION_TYPES = [
    "play_blind", "skip_blind", "reroll_boss",
    "play", "discard", "use_consumable",
    "buy", "sell_joker", "reroll", "leave_shop",
    "pick_booster", "skip_booster",
    "advance",
]
ACTION_TYPE_IDX = {t: i for i, t in enumerate(ACTION_TYPES)}

N_ACTION_TYPES   = len(ACTION_TYPES)              # 13
N_HAND_SLOTS     = 12                             # hand_size can exceed the obs's 8
N_CONSUMABLES    = 4                              # 2 base + Crystal Ball + headroom
N_SHOP_SLOTS     = 8                              # 4 shelf + 2 packs + 1 voucher + headroom
N_JOKER_SLOTS    = 8                              # 5 base + Negative jokers
N_BOOSTER_SLOTS  = 8

ACTION_FEATURE_DIM = (
    N_ACTION_TYPES        # 13
    + N_HAND_SLOTS        # 12
    + 1                   # cards_selected / targets count
    + N_CONSUMABLES       # 4
    + N_SHOP_SLOTS        # 8
    + N_JOKER_SLOTS       # 8
    + N_BOOSTER_SLOTS     # 8
    + 1                   # booster pick count
    + 1                   # overflow: (max index + 1) / 20, nonzero iff a slot bound was exceeded
)
assert ACTION_FEATURE_DIM == 56

_HAND_OFF      = N_ACTION_TYPES
_COUNT_OFF     = _HAND_OFF + N_HAND_SLOTS
_CONS_OFF      = _COUNT_OFF + 1
_SHOP_OFF      = _CONS_OFF + N_CONSUMABLES
_JOKER_OFF     = _SHOP_OFF + N_SHOP_SLOTS
_BOOSTER_OFF   = _JOKER_OFF + N_JOKER_SLOTS
_BOOSTER_N_OFF = _BOOSTER_OFF + N_BOOSTER_SLOTS
_OVERFLOW_OFF  = _BOOSTER_N_OFF + 1


def _set_slot(v: np.ndarray, base: int, i: int, n_slots: int) -> None:
    """One-hot slot i; when i is past the bound, record it in the overflow scalar
    ((i+1)/20, so it is nonzero iff something overflowed and still separates indices)."""
    if 0 <= i < n_slots:
        v[base + i] = 1.0
    else:
        v[_OVERFLOW_OFF] = max(float(v[_OVERFLOW_OFF]), (i + 1) / 20.0)


def featurize_action(action: dict) -> np.ndarray:
    v = np.zeros(ACTION_FEATURE_DIM, dtype=np.float32)
    t = action["type"]
    type_idx = ACTION_TYPE_IDX.get(t)
    if type_idx is None:
        v[_OVERFLOW_OFF] = 1.0   # unknown type: at least don't collide with a legal row
        return v
    v[type_idx] = 1.0

    if t in ("play", "discard"):
        cards = action.get("cards", [])
        for c in cards:
            _set_slot(v, _HAND_OFF, c, N_HAND_SLOTS)
        v[_COUNT_OFF] = len(cards) / 5.0

    elif t == "use_consumable":
        _set_slot(v, _CONS_OFF, action.get("consumable_idx", 0), N_CONSUMABLES)
        targets = action.get("target_cards", [])
        for c in targets:
            _set_slot(v, _HAND_OFF, c, N_HAND_SLOTS)
        v[_COUNT_OFF] = len(targets) / 5.0

    elif t == "buy":
        _set_slot(v, _SHOP_OFF, action.get("item_idx", 0), N_SHOP_SLOTS)

    elif t == "sell_joker":
        _set_slot(v, _JOKER_OFF, action.get("joker_idx", 0), N_JOKER_SLOTS)

    elif t == "pick_booster":
        idxs = action.get("indices", [])
        for c in idxs:
            _set_slot(v, _BOOSTER_OFF, c, N_BOOSTER_SLOTS)
        v[_BOOSTER_N_OFF] = len(idxs) / 5.0

    return v


def featurize_actions(actions: list[dict]) -> np.ndarray:
    """Stack featurize_action over a list -> (N, ACTION_FEATURE_DIM) array.

    W3 (2026-08-22): this is the hottest CPU call in the whole search — a leaf at
    SELECTING_HAND has ~436 legal actions and this runs once per leaf, so at 500 sims it
    is ~0.2 s per decision. The obvious implementation (`np.stack` over 436 separate
    56-float arrays) costs 0.44 ms per leaf on this box; collecting the one-hot
    coordinates in plain Python lists and writing them with ONE fancy-index assignment
    costs 0.14 ms. Byte-identical output — `tests/test_batched.py::
    test_featurize_actions_matches_per_action` asserts equality against the per-action
    function for every legal action of a real game in every state.
    """
    n = len(actions)
    if n == 0:
        return np.zeros((0, ACTION_FEATURE_DIM), dtype=np.float32)

    out = np.zeros((n, ACTION_FEATURE_DIM), dtype=np.float32)
    rows: list[int] = []          # coordinates of every 1.0 in the block
    cols: list[int] = []
    count_rows: list[int] = []    # the two scalar columns
    count_vals: list[float] = []
    boost_rows: list[int] = []
    boost_vals: list[float] = []
    overflow: dict[int, float] = {}

    def slot(i: int, base: int, idx: int, n_slots: int) -> None:
        if 0 <= idx < n_slots:
            rows.append(i)
            cols.append(base + idx)
        else:
            v = (idx + 1) / 20.0
            if v > overflow.get(i, 0.0):
                overflow[i] = v

    for i, action in enumerate(actions):
        t = action["type"]
        type_idx = ACTION_TYPE_IDX.get(t)
        if type_idx is None:
            overflow[i] = 1.0
            continue
        rows.append(i)
        cols.append(type_idx)

        if t == "play" or t == "discard":
            cards = action.get("cards", [])
            for c in cards:
                slot(i, _HAND_OFF, c, N_HAND_SLOTS)
            count_rows.append(i)
            count_vals.append(len(cards) / 5.0)
        elif t == "use_consumable":
            slot(i, _CONS_OFF, action.get("consumable_idx", 0), N_CONSUMABLES)
            targets = action.get("target_cards", [])
            for c in targets:
                slot(i, _HAND_OFF, c, N_HAND_SLOTS)
            count_rows.append(i)
            count_vals.append(len(targets) / 5.0)
        elif t == "buy":
            slot(i, _SHOP_OFF, action.get("item_idx", 0), N_SHOP_SLOTS)
        elif t == "sell_joker":
            slot(i, _JOKER_OFF, action.get("joker_idx", 0), N_JOKER_SLOTS)
        elif t == "pick_booster":
            idxs = action.get("indices", [])
            for c in idxs:
                slot(i, _BOOSTER_OFF, c, N_BOOSTER_SLOTS)
            boost_rows.append(i)
            boost_vals.append(len(idxs) / 5.0)

    out[rows, cols] = 1.0
    if count_rows:
        out[count_rows, _COUNT_OFF] = count_vals
    if boost_rows:
        out[boost_rows, _BOOSTER_N_OFF] = boost_vals
    for i, v in overflow.items():
        out[i, _OVERFLOW_OFF] = v
    return out
