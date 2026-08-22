"""
action_features_set.py — SET-BASED action features (Phase 4 W1).

The Phase 3 row (`action_features.featurize_actions`, 56 dims) spent 40 of its dims on slot
one-hots: 12 hand slots + 4 consumables + 8 shop + 8 joker + 8 booster. So "play the Ace of
Spades" was encoded as "play slot 3" and the network had to bind slot 3 to a card through
the trunk, separately for every slot — the same positional pathology the observation had.

Here an action carries **row-normalised masks over the observation's item slots**:

    act_sel  (N, caps.hand)     the CARDS the action selects  (play / discard /
                                use_consumable targets)
    act_tgt  (N, caps.target)   the ITEM(s) it targets, over the concatenated non-card
                                block [jokers | consumables | shelf | packs]
                                (buy / sell_joker / use_consumable slot / pick_booster)

and `model_set.SetPolicyValueNet` pools the item embeddings the observation already
computed:

    act_emb = type_emb(act_type)
            + act_sel @ hand_item_emb
            + act_tgt @ target_item_emb
            + num_proj(act_num)

so "buy Blueprint" is the Blueprint embedding wherever it sits on the shelf.

An index past a cap is dropped from the mask and sets the overflow scalar, so two
overflowing actions stay distinguishable (the Phase 3 featurizer's fix, kept).

Covers all 13 action types `game.legal_actions()` emits — the vocabulary is IMPORTED from
`action_features.ACTION_TYPES`, not re-listed, so the two cannot drift.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from balatro_sim.env_v7 import HAND_TYPES
from balatro_sim.game import State

from .action_features import ACTION_TYPES, ACTION_TYPE_IDX
from .encoder_set import CAT_DTYPE, DEFAULT_CAPS, ItemCaps

N_ACTION_TYPES_SET = len(ACTION_TYPES) + 1        # + 1 "unknown" slot at index 0
UNKNOWN_TYPE = 0

# type index used in `act_type` = ACTION_TYPE_IDX[t] + 1
_TYPE_IDX = {t: i + 1 for t, i in ACTION_TYPE_IDX.items()}

N_HAND_TYPES = len(HAND_TYPES)                    # 12
_HAND_TYPE_IDX = {ht: i for i, ht in enumerate(HAND_TYPES)}

# act_num layout (documented in SETENC_NOTES §2.2)
AN_N_SEL      = 0
AN_N_TGT      = 1
AN_COST       = 2
AN_AFFORDABLE = 3
AN_OVERFLOW   = 4
AN_HAND_TYPE  = 5                                  # .. 5 + 12
AN_HAND_LEVEL = AN_HAND_TYPE + N_HAND_TYPES        # 17
AN_SEL_CHIPS  = AN_HAND_LEVEL + 1                  # 18
AN_FREE       = AN_SEL_CHIPS + 1                   # 19
ACT_NUM_DIM   = AN_FREE + 1                        # 20


# ════════════════════════════════════════════════════════════════════════════════
# Per-state context (computed once per leaf, reused by every action row)
# ════════════════════════════════════════════════════════════════════════════════

class HandContext:
    """Precomputed per-card arrays for the fast hand-type evaluation (§ below)."""

    __slots__ = ("n", "rank_onehot", "suit_flush", "nonstone", "base_chips",
                 "present_j", "four_fingers", "shortcut", "smeared", "any_card")

    _SUITS = ("Spades", "Hearts", "Clubs", "Diamonds")
    _RED = frozenset({"Hearts", "Diamonds"})

    def __init__(self, hand, flags: dict):
        n = len(hand)
        self.n = n
        self.four_fingers = bool(flags.get("four_fingers", False))
        self.shortcut = bool(flags.get("shortcut", False))
        self.smeared = bool(flags.get("smeared", False))

        # rank one-hot over ranks 2..14 (13 columns), Stones excluded (get_id < 0)
        rank_onehot = np.zeros((n, 13), dtype=np.float32)
        suit_flush = np.zeros((n, 4), dtype=np.float32)
        nonstone = np.zeros(n, dtype=np.float32)
        base_chips = np.zeros(n, dtype=np.float32)
        # presence rows for the j = 1..14 straight walk (j == 1 stands for the Ace)
        present_j = np.zeros((n, 15), dtype=np.float32)

        for i, c in enumerate(hand):
            base_chips[i] = c.base_chips
            stone = c.enhancement == "Stone"
            if stone:
                continue
            nonstone[i] = 1.0
            if 2 <= c.rank <= 14:
                rank_onehot[i, c.rank - 2] = 1.0
                present_j[i, 14 if c.rank == 14 else c.rank] = 1.0
                if c.rank == 14:
                    present_j[i, 1] = 1.0
            wild = c.enhancement == "Wild" and not c.debuffed
            for si, suit in enumerate(self._SUITS):
                if wild:
                    suit_flush[i, si] = 1.0
                elif self.smeared and ((c.suit in self._RED) == (suit in self._RED)):
                    suit_flush[i, si] = 1.0
                elif c.suit == suit:
                    suit_flush[i, si] = 1.0

        self.rank_onehot = rank_onehot
        self.suit_flush = suit_flush
        self.nonstone = nonstone
        self.base_chips = base_chips
        self.present_j = present_j
        self.any_card = n > 0


def fast_hand_types(sel: np.ndarray, ctx: HandContext) -> np.ndarray:
    """Vectorised `hand_eval.evaluate_hand(...)[0]` for MANY card subsets at once.

    `sel` is a (N, n_hand) 0/1 matrix (one row per action, the RAW un-normalised
    selection). Returns (N,) int64 indices into `HAND_TYPES`, or -1 for an empty
    selection.

    Why not call `hand_eval.evaluate_hand` per action: it costs 4.8 us on this box, and a
    `SELECTING_HAND` leaf has ~218 plays + ~218 discards, so per-action evaluation is
    ~2.1 ms per leaf against the ~0.5 ms the entire per-leaf CPU path costs today. This
    routine does the same 218 + 218 in ~40 us.

    It is a SECOND implementation of engine logic, which is exactly the mistake
    AGENT_NOTES §2.1 documents for the copied encoder, so it carries the same mitigation:
    `tests/test_set_encoder.py::test_fast_hand_type_matches_hand_eval` asserts equality
    with `hand_eval.evaluate_hand` over every subset of size 1-5 of many real hands
    (Stone / Wild / debuffed cards included) across all four flag combinations.
    """
    n_actions = sel.shape[0]
    out = np.full(n_actions, -1, dtype=np.int64)
    if n_actions == 0 or ctx.n == 0:
        return out

    n_cards = sel.sum(axis=1)                                    # len(cards)
    live = n_cards > 0
    if not live.any():
        return out
    # An all-Stone selection falls through every branch below (Stones never group, never
    # flush and never straight) and lands on High Card — which is exactly what
    # evaluate_hand's `if not active` early return produces.

    rank_counts = sel @ ctx.rank_onehot                          # (N, 13)
    floor = 4.0 if ctx.four_fingers else 5.0

    same5 = (rank_counts == 5).any(axis=1)
    same4 = (rank_counts == 4).any(axis=1)
    n_same3 = (rank_counts == 3).sum(axis=1)
    n_same2 = (rank_counts == 2).sum(axis=1)
    same3 = n_same3 > 0
    same2 = n_same2 > 0

    size_ok = (n_cards <= 5) & (n_cards >= floor)
    flush = size_ok & ((sel @ ctx.suit_flush) >= floor).any(axis=1)
    straight = size_ok & _straight(sel, ctx, floor)

    # The decision tree of hand_eval.evaluate_hand, in the same order.
    ht = np.full(n_actions, _HAND_TYPE_IDX["High Card"], dtype=np.int64)
    order = [
        (same2, "Pair"),
        ((n_same2 == 2) | ((n_same3 == 1) & (n_same2 == 1)), "Two Pair"),
        (same3, "Three of a Kind"),
        (straight, "Straight"),
        (flush, "Flush"),
        (same3 & same2, "Full House"),
        (same4, "Four of a Kind"),
        (flush & straight, "Straight Flush"),
        (same5, "Five of a Kind"),
        (same3 & same2 & flush, "Flush House"),
        (same5 & flush, "Flush Five"),
    ]
    for cond, name in order:                       # later rows overwrite earlier ones,
        ht[cond] = _HAND_TYPE_IDX[name]            # which reproduces the if/elif order
    out[live] = ht[live]
    return out


def _straight(sel: np.ndarray, ctx: HandContext, floor: float) -> np.ndarray:
    """The Lua `get_straight` walk (misc_functions.lua:548-590), vectorised over rows."""
    present = (sel @ ctx.present_j) > 0                           # (N, 15), col 0 unused
    n = sel.shape[0]
    length = np.zeros(n, dtype=np.int64)
    found = np.zeros(n, dtype=bool)
    skipped = np.zeros(n, dtype=bool)
    can_skip = np.zeros(n, dtype=bool)
    for j in range(1, 15):
        p = present[:, j]
        if ctx.shortcut and j != 14:
            skip_ok = (~p) & (~skipped)
        else:
            skip_ok = can_skip
        length = np.where(p, length + 1, np.where(skip_ok, length, 0))
        skipped = skip_ok
        found |= length >= floor
    return found


# ════════════════════════════════════════════════════════════════════════════════
# The featurizer
# ════════════════════════════════════════════════════════════════════════════════

def featurize_actions_set(game, actions: Sequence[dict],
                          caps: ItemCaps = DEFAULT_CAPS,
                          hand_type_features: bool = True) -> dict:
    """`actions` (from `game.legal_actions()`) -> the `Acts` dict of SETENC_NOTES §0.3."""
    n = len(actions)
    hand_cap = caps.hand
    tgt_cap = caps.target
    act_type = np.zeros(n, dtype=CAT_DTYPE)
    sel = np.zeros((n, hand_cap), dtype=np.float32)
    tgt = np.zeros((n, tgt_cap), dtype=np.float32)
    num = np.zeros((n, ACT_NUM_DIM), dtype=np.float32)
    if n == 0:
        return {"act_type": act_type, "act_sel": sel, "act_tgt": tgt, "act_num": num}

    off_j, off_c = caps.off_jokers, caps.off_consumables
    off_s, off_p = caps.off_shelf, caps.off_packs
    n_hand_live = min(len(game.hand), hand_cap)

    dollars = game.dollars
    shop = game.current_shop
    jokers = game.jokers
    discount = game.shop_discount
    reroll_cost = max(0, game.reroll_cost - game.reroll_discount)
    free_reroll = game.free_rerolls_remaining > 0

    sel_rows: list[int] = []
    sel_cols: list[int] = []
    tgt_rows: list[int] = []
    tgt_cols: list[int] = []
    overflow = np.zeros(n, dtype=np.float32)
    is_card_action = np.zeros(n, dtype=bool)

    for i, a in enumerate(actions):
        t = a["type"]
        ti = _TYPE_IDX.get(t)
        if ti is None:
            act_type[i] = UNKNOWN_TYPE
            overflow[i] = 1.0
            continue
        act_type[i] = ti

        if t == "play" or t == "discard":
            for c in a.get("cards", ()):
                if 0 <= c < hand_cap:
                    sel_rows.append(i)
                    sel_cols.append(c)
                else:
                    overflow[i] = 1.0
            is_card_action[i] = True

        elif t == "use_consumable":
            ci = a.get("consumable_idx", 0)
            if 0 <= ci < caps.consumables:
                tgt_rows.append(i)
                tgt_cols.append(off_c + ci)
            else:
                overflow[i] = 1.0
            for c in a.get("target_cards", ()):
                if 0 <= c < hand_cap:
                    sel_rows.append(i)
                    sel_cols.append(c)
                else:
                    overflow[i] = 1.0

        elif t == "buy":
            ii = a.get("item_idx", 0)
            if 0 <= ii < caps.shelf:
                tgt_rows.append(i)
                tgt_cols.append(off_s + ii)
            else:
                overflow[i] = 1.0
            if 0 <= ii < len(shop):
                item = shop[ii]
                price = item.discounted_price(discount)
                num[i, AN_COST] = min(price / 20.0, 2.0)
                num[i, AN_AFFORDABLE] = float(dollars >= price)
                num[i, AN_FREE] = float(item.couponed or price == 0)

        elif t == "sell_joker":
            ji = a.get("joker_idx", 0)
            if 0 <= ji < caps.jokers:
                tgt_rows.append(i)
                tgt_cols.append(off_j + ji)
            else:
                overflow[i] = 1.0
            if 0 <= ji < len(jokers):
                sv = jokers[ji].state.get("sell_value", 1)
                num[i, AN_COST] = -min(float(sv) / 20.0, 2.0)
            num[i, AN_AFFORDABLE] = 1.0

        elif t == "pick_booster":
            for c in a.get("indices", ()):
                if 0 <= c < caps.packs:
                    tgt_rows.append(i)
                    tgt_cols.append(off_p + c)
                else:
                    overflow[i] = 1.0

        elif t == "reroll":
            num[i, AN_COST] = min(reroll_cost / 20.0, 2.0)
            num[i, AN_AFFORDABLE] = float(free_reroll or dollars >= reroll_cost)
            num[i, AN_FREE] = float(free_reroll)

        else:
            num[i, AN_AFFORDABLE] = 1.0

    if sel_rows:
        sel[sel_rows, sel_cols] = 1.0
    if tgt_rows:
        tgt[tgt_rows, tgt_cols] = 1.0

    n_sel = sel.sum(axis=1)
    n_tgt = tgt.sum(axis=1)
    num[:, AN_N_SEL] = n_sel / 5.0
    num[:, AN_N_TGT] = n_tgt / 5.0
    num[:, AN_OVERFLOW] = overflow

    # would-be hand type (play / discard only)
    if hand_type_features and n_hand_live > 0 and is_card_action.any():
        ctx = _hand_context(game)
        rows = np.flatnonzero(is_card_action)
        # A hand longer than the cap has its extra cards dropped from the mask (already
        # flagged as overflow); pad the block back out to ctx.n so the shapes line up.
        m = min(ctx.n, hand_cap)
        block = np.zeros((rows.size, ctx.n), dtype=np.float32)
        block[:, :m] = sel[rows, :m]
        types = fast_hand_types(block, ctx)
        good = types >= 0
        if good.any():
            r = rows[good]
            num[r, AN_HAND_TYPE + types[good]] = 1.0
            levels = np.fromiter(
                (min((game.planet_levels.get(HAND_TYPES[t], 1) - 1) / 10.0, 1.0)
                 for t in types[good]), dtype=np.float32, count=int(good.sum()))
            num[r, AN_HAND_LEVEL] = levels
        chips = block @ ctx.base_chips
        num[rows, AN_SEL_CHIPS] = np.log1p(chips) / np.log1p(300.0)

    # row-normalise the masks (mean pooling in the model)
    np.divide(sel, np.maximum(n_sel, 1.0)[:, None], out=sel)
    np.divide(tgt, np.maximum(n_tgt, 1.0)[:, None], out=tgt)

    return {"act_type": act_type, "act_sel": sel, "act_tgt": tgt, "act_num": num}


def _hand_context(game) -> HandContext:
    flags = game.hand_eval_flags() if hasattr(game, "hand_eval_flags") else {}
    return HandContext(game.hand, flags)


def concat_acts(blocks: Sequence[dict]) -> dict:
    """Concatenate per-state `Acts` dicts into one ragged block (for `score_actions_flat`)."""
    if not blocks:
        return {}
    if len(blocks) == 1:
        return blocks[0]
    return {k: np.concatenate([b[k] for b in blocks], axis=0) for k in blocks[0]}


def take_acts(acts: dict, idx) -> dict:
    """Row subset of an `Acts` dict (used by `Sample` v2 subsampling)."""
    return {k: np.ascontiguousarray(v[idx]) for k, v in acts.items()}


__all__ = [
    "featurize_actions_set", "fast_hand_types", "HandContext", "concat_acts", "take_acts",
    "ACT_NUM_DIM", "N_ACTION_TYPES_SET", "N_HAND_TYPES",
]
