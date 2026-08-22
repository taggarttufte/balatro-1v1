"""
encoder_set.py — the SET-BASED observation (Phase 4 W1).

`env_v7._encode_obs` (447 dims, reused by `encoder.V7Encoder`) writes 8 hand slots x 30 and
5 joker slots x 10 at FIXED OFFSETS. Both bounds are exceeded in real play — `hand_size`
grows with Juggler / Turtle Bean / Ouija / Ectoplasm / vouchers and `joker_slots` grows with
Negative jokers — so a 9th card in hand is a legal `play` target the value/policy net cannot
see (AGENT_NOTES §8, deferred by Phase 3). Raising the caps fixes the blindness and keeps
the pathology: slot k owns its own weights, so "Blueprint in slot 3" and "Blueprint in slot
4" are learned separately.

This module encodes the state as FIVE VARIABLE-LENGTH SETS plus one scalar vector:

    hand         playing cards in hand            cap 16
    jokers       owned jokers                     cap 12
    consumables  owned consumables                cap 6
    shelf        shop items (`game.current_shop`) cap 8
    packs        open booster choices             cap 8

Padding to a cap is TRANSPORT ONLY. Every array comes with a mask and the network
(`model_set.SetPolicyValueNet`) is invariant to the order of the rows and unaffected by the
padded ones — pinned by `tests/test_set_encoder.py`. Position is still available as an
explicit per-item FEATURE (jokers score left to right, so joker order is real information):
the net is invariant to row order, not blind to where a joker sits.

Caps (generous, and documented because the checkpoint records them):

    hand 16         base 8 + Juggler/Turtle Bean/Ouija/Ectoplasm/voucher deltas
    jokers 12       base 5 + Negative jokers (each adds a slot) + Antimatter
    consumables 6   base 2 + Crystal Ball + vouchers + Negative
    shelf 8         4 shelf slots (Overstock/Overstock Plus) + 2 packs + 1 voucher + headroom
    packs 8         mega packs offer 5; headroom

Anything past a cap is DROPPED from the observation (and flagged: `scalars` carries the
count of every set, so "there are 17 cards in hand and you can see 16" is visible). The
action featurizer flags the same overflow on its own rows.

Nothing here hardcodes a game key: `KEY_VOCAB` is built from `balatro_sim.game_keys` at
import, the same discipline `encoder.py` follows by calling env_v7 rather than copying it.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import numpy as np

from balatro_sim import game_keys as _gk
from balatro_sim.game import (
    ALL_REGULAR_BOSS_BLINDS, SHOWDOWN_BOSS_BLINDS, BalatroGame, State,
)
from balatro_sim.constants import (
    HAND_SIZE, MLB_STARTING_LIVES, STARTING_DISCARDS, STARTING_HANDS,
)
from balatro_sim.consumables import ALL_PLANETS, ALL_SPECTRALS, ALL_TAROTS, PLANET_HAND
from balatro_sim.env_v7 import EDITIONS, ENHANCEMENTS, HAND_TYPES, SEALS, SUIT_ORDER
from balatro_sim.shop import JOKER_CATALOGUE

Obs = dict


# ════════════════════════════════════════════════════════════════════════════════
# Caps
# ════════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ItemCaps:
    """Padding widths. Part of the checkpoint: a net trained with different caps has
    differently shaped action masks, so a resume across caps is refused."""
    hand: int = 16
    jokers: int = 12
    consumables: int = 6
    shelf: int = 8
    packs: int = 8

    @property
    def target(self) -> int:
        """Width of the concatenated NON-CARD item block the action `tgt` mask indexes:
        [jokers | consumables | shelf | packs]."""
        return self.jokers + self.consumables + self.shelf + self.packs

    @property
    def total(self) -> int:
        return self.hand + self.target

    # Offsets of each set inside the concatenated target block.
    @property
    def off_jokers(self) -> int: return 0
    @property
    def off_consumables(self) -> int: return self.jokers
    @property
    def off_shelf(self) -> int: return self.jokers + self.consumables
    @property
    def off_packs(self) -> int: return self.jokers + self.consumables + self.shelf

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "ItemCaps":
        return cls(**d) if d else cls()


DEFAULT_CAPS = ItemCaps()


# ════════════════════════════════════════════════════════════════════════════════
# Shared key vocabulary
# ════════════════════════════════════════════════════════════════════════════════

PAD_KEY = 0
UNK_KEY = 1

#: Categorical index arrays are int16, not int64. Every index is < 512 (the widest table
#: is KEY_VOCAB at 251), and int64 would quadruple the observation's footprint in the
#: replay buffer for nothing. `model_set` widens them to Long at the embedding lookup.
CAT_DTYPE = np.int16

KEY_VOCAB: list[str] = (
    ["<pad>", "<unk>"]
    + list(_gk.JOKER_KEYS)          # 150
    + list(_gk.TAROT_KEYS)          # 22
    + list(_gk.PLANET_KEYS)         # 12
    + list(_gk.SPECTRAL_KEYS)       # 18
    + list(_gk.VOUCHER_KEYS)        # 32
    + list(_gk.BOOSTER_TYPE_KEYS)   # 13
)
KEY_IDX: dict[str, int] = {k: i for i, k in enumerate(KEY_VOCAB)}
KEY_VOCAB_SIZE = len(KEY_VOCAB)


def key_index(key: Optional[str]) -> int:
    if not key:
        return PAD_KEY
    return KEY_IDX.get(key, UNK_KEY)


# ════════════════════════════════════════════════════════════════════════════════
# Card block (shared by hand cards, shelf playing cards and Standard-pack cards)
# ════════════════════════════════════════════════════════════════════════════════

CARD_CAT_DIM = 5        # rank, suit, enhancement, edition, seal   (index 0 = unknown/pad)
CARD_NUM_DIM = 9

N_RANK   = 14           # 0 unknown + ranks 2..14
N_SUIT   = 1 + len(SUIT_ORDER)
N_ENH    = 1 + len(ENHANCEMENTS)
N_EDITION = 1 + len(EDITIONS)
N_SEAL   = 1 + len(SEALS)
CARD_CARDINALITIES = (N_RANK, N_SUIT, N_ENH, N_EDITION, N_SEAL)

_SUIT_IDX = {s: i + 1 for i, s in enumerate(SUIT_ORDER)}
_ENH_IDX = {e: i + 1 for i, e in enumerate(ENHANCEMENTS)}
_EDITION_IDX = {e: i + 1 for i, e in enumerate(EDITIONS)}
_SEAL_IDX = {s: i + 1 for i, s in enumerate(SEALS)}

_RARITY_IDX = {"Common": 1, "Uncommon": 2, "Rare": 3, "Legendary": 4}
N_RARITY = 5

SHELF_KINDS = ["joker", "planet", "tarot", "spectral", "card", "voucher", "booster"]
_SHELF_KIND_IDX = {k: i + 1 for i, k in enumerate(SHELF_KINDS)}
N_SHELF_KIND = 1 + len(SHELF_KINDS)

PACK_SETS = ["Joker", "Tarot", "Planet", "Spectral", "Base", "Voucher", "Booster"]
_PACK_SET_IDX = {s: i + 1 for i, s in enumerate(PACK_SETS)}
N_PACK_SET = 1 + len(PACK_SETS)


def card_cat(card) -> tuple[int, int, int, int, int]:
    """The 5 categorical indices of one playing card. A FACE-DOWN card is hidden
    information (The House / Wheel / Mark / Fish): every index is 0 = unknown, exactly as
    env_v7 zeroes its features while keeping the "present" bit."""
    if card is None or getattr(card, "face_down", False):
        return (0, 0, 0, 0, 0)
    rank = card.rank
    rank_i = rank - 1 if 2 <= rank <= 14 else 0
    return (
        rank_i,
        _SUIT_IDX.get(card.suit, 0),
        _ENH_IDX.get(card.enhancement, 0),
        _EDITION_IDX.get(card.edition, 0),
        _SEAL_IDX.get(card.seal, 0),
    )


# ════════════════════════════════════════════════════════════════════════════════
# Per-set numeric widths
# ════════════════════════════════════════════════════════════════════════════════

JOKER_CAT_DIM = 2       # edition, rarity
JOKER_NUM_DIM = 16
CONS_NUM_DIM = 8
SHELF_CAT_DIM = 2       # kind, edition
SHELF_NUM_DIM = 12
PACK_CAT_DIM = 2        # set, edition
PACK_NUM_DIM = 8

# Numeric joker-state keys worth their own feature (grep over mp/engine/balatro_sim/jokers:
# these are the ones that actually scale). Everything else numeric is summed into the
# catch-all so an unlisted scaling joker is not flat.
_JOKER_STATE_KEYS = ("mult", "chips", "xmult", "mult_mult", "sell_value",
                     "rounds", "count", "streak", "pending_money")
_JOKER_FLAG_KEYS = ("eternal", "perishable", "rental")


# ════════════════════════════════════════════════════════════════════════════════
# Scalars
# ════════════════════════════════════════════════════════════════════════════════

BOSS_KEYS_OBS = list(ALL_REGULAR_BOSS_BLINDS) + list(SHOWDOWN_BOSS_BLINDS)   # 28
_BOSS_IDX = {k: i for i, k in enumerate(BOSS_KEYS_OBS)}

STATES_OBS = [State.BLIND_SELECT, State.SELECTING_HAND, State.ROUND_EVAL, State.SHOP,
              State.BOOSTER_OPEN, State.GAME_OVER, State.PVP_WAIT]
_STATE_IDX = {s: i for i, s in enumerate(STATES_OBS)}

BLIND_KINDS = ["Small", "Big", "Boss"]
_BLIND_KIND_IDX = {k: i for i, k in enumerate(BLIND_KINDS)}

VOUCHER_KEYS_OBS = list(_gk.VOUCHER_KEYS)      # 32 (env_v7 encoded only the first 27)
_VOUCHER_IDX = {k: i for i, k in enumerate(VOUCHER_KEYS_OBS)}
TAG_KEYS_OBS = list(_gk.TAG_KEYS)              # 24
_TAG_IDX = {k: i for i, k in enumerate(TAG_KEYS_OBS)}
DECK_KEYS_OBS = list(_gk.DECK_KEYS)            # 15
_DECK_IDX = {k: i for i, k in enumerate(DECK_KEYS_OBS)}

# (name, width). The encoder writes by generated offset, so this list IS the layout and
# `test_scalar_layout_matches_dim` cannot let the docs drift from the code.
SCALAR_LAYOUT: list[tuple[str, int]] = [
    ("state",             len(STATES_OBS)),      # 7  one-hot
    ("ante_blind",        8),                    # ante/8, log ante, blind_idx/2, progress,
                                                 # log target, is_boss, is_showdown, disabled
    ("blind_kind",        len(BLIND_KINDS)),     # 3
    ("boss",              len(BOSS_KEYS_OBS)),   # 28 one-hot
    ("economy",           10),
    ("capacity",          9),
    ("round",             8),
    ("planet_levels",     len(HAND_TYPES)),      # 12
    ("hand_types_played", len(HAND_TYPES)),      # 12
    ("vouchers",          len(VOUCHER_KEYS_OBS)),# 32 multi-hot
    ("tags",              len(TAG_KEYS_OBS)),    # 24 multi-hot
    ("tag_count",         1),
    ("deck_composition",  16),
    ("deck_id",           len(DECK_KEYS_OBS)),   # 15 one-hot
    ("stake",             1),
    ("mlb",               10),
]

_SCALAR_OFF: dict[str, int] = {}
_o = 0
for _name, _w in SCALAR_LAYOUT:
    _SCALAR_OFF[_name] = _o
    _o += _w
SCALAR_DIM = _o
del _o, _name, _w


def scalar_offsets() -> dict[str, int]:
    return dict(_SCALAR_OFF)


# ════════════════════════════════════════════════════════════════════════════════
# The encoder
# ════════════════════════════════════════════════════════════════════════════════

def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else (hi if x > hi else x)


class SetEncoder:
    """`--encoder set`. `enc(game)` returns the `Obs` dict documented in
    SETENC_NOTES.md §0.2. There is no `dim`: callers branch on `is_set`."""

    name = "set"
    is_set = True
    dim = None

    def __init__(self, caps: ItemCaps | dict | None = None):
        self.caps = caps if isinstance(caps, ItemCaps) else ItemCaps.from_dict(caps)

    # ── metadata ─────────────────────────────────────────────────────────────

    def describe(self) -> dict:
        return {"name": self.name, "caps": self.caps.as_dict(),
                "scalar_dim": SCALAR_DIM, "key_vocab": KEY_VOCAB_SIZE}

    @classmethod
    def from_description(cls, desc: dict) -> "SetEncoder":
        return cls(caps=ItemCaps.from_dict(desc.get("caps")))

    # ── the encode ───────────────────────────────────────────────────────────

    def __call__(self, game: BalatroGame) -> Obs:
        caps = self.caps
        obs: Obs = {}
        self._encode_hand(game, caps, obs)
        self._encode_jokers(game, caps, obs)
        self._encode_consumables(game, caps, obs)
        self._encode_shelf(game, caps, obs)
        self._encode_packs(game, caps, obs)
        obs["scalars"] = self._encode_scalars(game, caps)
        return obs

    # ── sets ─────────────────────────────────────────────────────────────────

    def _encode_hand(self, game, caps: ItemCaps, obs: Obs) -> None:
        n_cap = caps.hand
        cat = np.zeros((n_cap, CARD_CAT_DIM), dtype=CAT_DTYPE)
        num = np.zeros((n_cap, CARD_NUM_DIM), dtype=np.float32)
        mask = np.zeros(n_cap, dtype=np.float32)

        hand = game.hand
        visible = [c for c in hand if not getattr(c, "face_down", False)]
        suit_counts = Counter(c.suit for c in visible)
        rank_counts = Counter(c.rank for c in visible)
        hand_ranks = [c.rank for c in visible]

        for slot in range(min(len(hand), n_cap)):
            c = hand[slot]
            mask[slot] = 1.0
            cat[slot] = card_cat(c)
            row = num[slot]
            row[8] = slot / n_cap
            if getattr(c, "face_down", False):
                row[4] = 1.0                      # face_down; identity stays unknown
                continue
            row[0] = (c.rank - 2) / 12.0
            row[1] = c.base_chips / 11.0
            row[2] = min(getattr(c, "bonus_chips", 0) / 50.0, 2.0)
            row[3] = float(c.debuffed)
            row[5] = (suit_counts[c.suit] - 1) / 7.0
            row[6] = (rank_counts[c.rank] - 1) / 7.0
            row[7] = sum(1 for r in hand_ranks
                         if r != c.rank and abs(r - c.rank) <= 4) / 7.0

        obs["hand_cat"], obs["hand_num"], obs["hand_mask"] = cat, num, mask

    def _encode_jokers(self, game, caps: ItemCaps, obs: Obs) -> None:
        n_cap = caps.jokers
        key = np.zeros(n_cap, dtype=CAT_DTYPE)
        cat = np.zeros((n_cap, JOKER_CAT_DIM), dtype=CAT_DTYPE)
        num = np.zeros((n_cap, JOKER_NUM_DIM), dtype=np.float32)
        mask = np.zeros(n_cap, dtype=np.float32)

        jokers = game.jokers
        n = len(jokers)
        for slot in range(min(n, n_cap)):
            j = jokers[slot]
            mask[slot] = 1.0
            key[slot] = key_index(j.key)
            info = JOKER_CATALOGUE.get(j.key, {})
            cat[slot, 0] = _EDITION_IDX.get(j.edition, 0)
            cat[slot, 1] = _RARITY_IDX.get(info.get("rarity", "Common"), 0)

            st = j.state or {}
            row = num[slot]
            row[0] = slot / n_cap
            row[1] = (slot + 1) / max(n, 1)
            row[2] = math.log1p(max(_f(st.get("mult")), 0.0)) / 10.0
            row[3] = math.log1p(max(_f(st.get("chips")), 0.0)) / 10.0
            row[4] = min(_f(st.get("xmult")), 10.0) / 10.0
            row[5] = min(_f(st.get("mult_mult")), 5.0) / 5.0
            row[6] = math.log1p(max(_f(st.get("sell_value"), 2.0), 0.0)) / 5.0
            row[7] = float(bool(st.get("destroyed", False)))
            row[8] = math.log1p(max(_f(st.get("rounds")), 0.0)) / 5.0
            row[9] = math.log1p(max(_f(st.get("count")), 0.0)) / 5.0
            row[10] = math.log1p(max(_f(st.get("streak")), 0.0)) / 5.0
            row[11] = math.log1p(max(_f(st.get("pending_money")), 0.0)) / 5.0
            other = 0.0
            for k, v in st.items():
                if k in _JOKER_STATE_KEYS or k in _JOKER_FLAG_KEYS:
                    continue
                if isinstance(v, bool):
                    other += float(v)
                elif isinstance(v, (int, float)):
                    other += abs(float(v))
                elif isinstance(v, (list, set, dict)):
                    other += len(v)
            row[12] = math.log1p(other) / 10.0
            row[13] = float(bool(st.get("eternal", False)))
            row[14] = float(bool(st.get("perishable", False)))
            row[15] = float(bool(st.get("rental", False)))

        obs["joker_key"], obs["joker_cat"] = key, cat
        obs["joker_num"], obs["joker_mask"] = num, mask

    def _encode_consumables(self, game, caps: ItemCaps, obs: Obs) -> None:
        n_cap = caps.consumables
        key = np.zeros(n_cap, dtype=CAT_DTYPE)
        num = np.zeros((n_cap, CONS_NUM_DIM), dtype=np.float32)
        mask = np.zeros(n_cap, dtype=np.float32)

        held = game.consumable_hand
        for slot in range(min(len(held), n_cap)):
            k = held[slot]
            mask[slot] = 1.0
            key[slot] = key_index(k)
            row = num[slot]
            row[0] = slot / n_cap
            row[1] = float(k in ALL_PLANETS)
            row[2] = float(k in ALL_TAROTS)
            row[3] = float(k in ALL_SPECTRALS)
            row[4] = min(_gk.CONSUMABLE_COST.get(k, 3) / 10.0, 2.0)
            ht = PLANET_HAND.get(k)
            if ht in HAND_TYPES:
                row[5] = HAND_TYPES.index(ht) / 11.0
                row[6] = min((game.planet_levels.get(ht, 1) - 1) / 10.0, 1.0)
            row[7] = min(_tarot_targets(k) / 2.0, 1.0)

        obs["cons_key"], obs["cons_num"], obs["cons_mask"] = key, num, mask

    def _encode_shelf(self, game, caps: ItemCaps, obs: Obs) -> None:
        n_cap = caps.shelf
        key = np.zeros(n_cap, dtype=CAT_DTYPE)
        cat = np.zeros((n_cap, SHELF_CAT_DIM), dtype=CAT_DTYPE)
        card = np.zeros((n_cap, CARD_CAT_DIM), dtype=CAT_DTYPE)
        num = np.zeros((n_cap, SHELF_NUM_DIM), dtype=np.float32)
        mask = np.zeros(n_cap, dtype=np.float32)

        shelf = game.current_shop
        joker_room = len(game.jokers) < game.joker_slots
        cons_room = len(game.consumable_hand) < game.consumable_slots
        for slot in range(min(len(shelf), n_cap)):
            item = shelf[slot]
            mask[slot] = 1.0
            key[slot] = key_index(item.key) if item.kind != "card" else UNK_KEY
            cat[slot, 0] = _SHELF_KIND_IDX.get(item.kind, 0)
            cat[slot, 1] = _EDITION_IDX.get(item.edition, 0)
            if item.kind == "card":
                card[slot] = _shelf_card_cat(item)
            price = item.discounted_price(game.shop_discount)
            row = num[slot]
            row[0] = slot / n_cap
            row[1] = float(not item.sold)
            row[2] = min(item.price / 20.0, 2.0)
            row[3] = min(price / 20.0, 2.0)
            row[4] = float(game.dollars >= price and not item.sold)
            info = JOKER_CATALOGUE.get(item.key, {}) if item.kind == "joker" else {}
            row[5] = _RARITY_IDX.get(info.get("rarity", ""), 0) / 4.0
            row[6] = float(joker_room) if item.kind == "joker" else 0.0
            row[7] = float(cons_room) if item.kind in ("planet", "tarot", "spectral") else 0.0
            row[8] = float(item.eternal)
            row[9] = float(item.perishable)
            row[10] = float(item.rental)
            row[11] = float(item.couponed)

        obs["shelf_key"], obs["shelf_cat"], obs["shelf_card"] = key, cat, card
        obs["shelf_num"], obs["shelf_mask"] = num, mask

    def _encode_packs(self, game, caps: ItemCaps, obs: Obs) -> None:
        n_cap = caps.packs
        key = np.zeros(n_cap, dtype=CAT_DTYPE)
        cat = np.zeros((n_cap, PACK_CAT_DIM), dtype=CAT_DTYPE)
        card = np.zeros((n_cap, CARD_CAT_DIM), dtype=CAT_DTYPE)
        num = np.zeros((n_cap, PACK_NUM_DIM), dtype=np.float32)
        mask = np.zeros(n_cap, dtype=np.float32)

        choices = game.booster_choices
        for slot in range(min(len(choices), n_cap)):
            c = choices[slot]
            mask[slot] = 1.0
            c_set = getattr(c, "set", "") or ""
            is_card = c_set == "Base" or getattr(c, "card", None) is not None
            key[slot] = UNK_KEY if is_card else key_index(getattr(c, "key", None))
            cat[slot, 0] = _PACK_SET_IDX.get(c_set, 0)
            cat[slot, 1] = _EDITION_IDX.get(getattr(c, "edition", "None"), 0)
            if is_card:
                card[slot] = _pack_card_cat(c)
            row = num[slot]
            row[0] = slot / n_cap
            row[1] = float(c_set == "Joker")
            row[2] = float(c_set in ("Tarot", "Planet", "Spectral"))
            row[3] = float(is_card)
            row[4] = float(_can_grant(game, c))
            row[5] = float(getattr(c, "eternal", False))
            row[6] = float(getattr(c, "perishable", False))
            row[7] = float(getattr(c, "rental", False))

        obs["pack_key"], obs["pack_cat"], obs["pack_card"] = key, cat, card
        obs["pack_num"], obs["pack_mask"] = num, mask

    # ── scalars ──────────────────────────────────────────────────────────────

    def _encode_scalars(self, game, caps: ItemCaps) -> np.ndarray:
        v = np.zeros(SCALAR_DIM, dtype=np.float32)
        off = _SCALAR_OFF
        gs = game
        blind = gs.current_blind

        # state (7)
        v[off["state"] + _STATE_IDX.get(gs.state, len(STATES_OBS) - 2)] = 1.0

        # ante / blind (8)
        b = off["ante_blind"]
        v[b + 0] = min(gs.ante / 8.0, 4.0)
        v[b + 1] = math.log1p(max(gs.ante, 0)) / math.log1p(16)
        v[b + 2] = gs.blind_idx / 2.0
        target = max(getattr(blind, "chips_target", 0), 1)
        v[b + 3] = min(gs.chips_scored / target, 2.0)
        v[b + 4] = math.log1p(getattr(blind, "chips_target", 0)) / math.log1p(100000)
        v[b + 5] = float(getattr(blind, "is_boss", False))
        v[b + 6] = float(getattr(blind, "is_showdown", False))
        v[b + 7] = float(getattr(blind, "disabled", False))

        # blind kind (3)
        ki = _BLIND_KIND_IDX.get(getattr(blind, "kind", ""), None)
        if ki is not None:
            v[off["blind_kind"] + ki] = 1.0

        # boss one-hot (28)
        if getattr(blind, "is_boss", False):
            bi = _BOSS_IDX.get(getattr(blind, "boss_key", ""), None)
            if bi is not None:
                v[off["boss"] + bi] = 1.0

        # economy (10)
        e = off["economy"]
        reroll_cost = max(0, gs.reroll_cost - gs.reroll_discount)
        v[e + 0] = min(gs.dollars / 50.0, 2.0)
        v[e + 1] = math.log1p(max(gs.dollars, 0)) / math.log1p(200)
        v[e + 2] = min(reroll_cost, 10) / 10.0
        v[e + 3] = gs.free_rerolls_remaining / max(gs.free_rerolls_per_round + 1, 1)
        v[e + 4] = min(gs.shop_discount, 1.0)
        v[e + 5] = min(gs.reroll_discount / 5.0, 1.0)
        v[e + 6] = min(gs.interest_cap / 25.0, 2.0)
        v[e + 7] = min(getattr(gs, "money_per_hand", 1) / 5.0, 2.0)
        v[e + 8] = float(getattr(gs, "no_interest", False))
        v[e + 9] = float(getattr(gs, "plasma", False))

        # capacity (9)
        c = off["capacity"]
        v[c + 0] = len(gs.jokers) / max(gs.joker_slots, 1)
        v[c + 1] = gs.joker_slots / caps.jokers
        v[c + 2] = len(gs.consumable_hand) / max(gs.consumable_slots, 1)
        v[c + 3] = gs.consumable_slots / caps.consumables
        v[c + 4] = gs.hand_size / caps.hand
        v[c + 5] = len(gs.hand) / caps.hand
        v[c + 6] = len(gs.jokers) / caps.jokers          # overflow-visible raw counts
        v[c + 7] = len(gs.current_shop) / caps.shelf
        v[c + 8] = len(gs.booster_choices) / caps.packs

        # round (8)
        r = off["round"]
        v[r + 0] = gs.hands_left / max(gs.base_hands, 1)
        v[r + 1] = gs.discards_left / max(gs.base_discards, 1)
        v[r + 2] = gs.base_hands / max(STARTING_HANDS, 1)
        v[r + 3] = gs.base_discards / max(STARTING_DISCARDS, 1)
        v[r + 4] = min(getattr(gs, "_hands_played_round", 0) / 5.0, 2.0)
        v[r + 5] = min(getattr(gs, "_discards_used_round", 0) / 5.0, 2.0)
        v[r + 6] = min(gs.skips / 8.0, 2.0)
        v[r + 7] = min(getattr(gs, "booster_picks_remaining", 0) / 5.0, 1.0)

        # planet levels (12) + hand types played this round (12)
        p = off["planet_levels"]
        hp = off["hand_types_played"]
        played = getattr(gs, "played_hand_types_this_round", set()) or set()
        for i, ht in enumerate(HAND_TYPES):
            v[p + i] = min((gs.planet_levels.get(ht, 1) - 1) / 10.0, 1.0)
            v[hp + i] = float(ht in played)

        # vouchers (32)
        vo = off["vouchers"]
        for vkey in getattr(gs, "vouchers", ()) or ():
            i = _VOUCHER_IDX.get(vkey)
            if i is not None:
                v[vo + i] = 1.0

        # tags (24) + count
        tg = off["tags"]
        tag_keys = gs.tag_state.keys() if getattr(gs, "tag_state", None) is not None else []
        for tkey in tag_keys:
            i = _TAG_IDX.get(tkey)
            if i is not None:
                v[tg + i] = 1.0
        v[off["tag_count"]] = min(len(tag_keys) / 4.0, 2.0)

        # deck composition (16) — env_v7's two 8-blocks, over the draw pile
        v[off["deck_composition"]:off["deck_composition"] + 16] = _deck_composition(gs.deck)

        # deck id (15) + stake
        di = _DECK_IDX.get(getattr(gs, "deck_key", ""), None)
        if di is not None:
            v[off["deck_id"] + di] = 1.0
        v[off["stake"]] = getattr(gs, "stake", 1) / 8.0

        # MLB (10) — inert on a vanilla game (env_mp.MP_OBS_FEATURES, self-side only)
        m = off["mlb"]
        v[m + 0] = getattr(gs, "lives", 0) / max(MLB_STARTING_LIVES, 1)
        v[m + 1] = float(getattr(blind, "is_pvp", False))
        opp = getattr(gs, "pvp_opponent_score", 0)
        v[m + 2] = math.log1p(max(opp, 0)) / math.log1p(100000)
        mine = gs.chips_scored
        v[m + 3] = _clip((mine - opp) / max(mine + opp, 1), -1.0, 1.0)
        v[m + 4] = min(getattr(gs, "pvp_opponent_hands", 0) / max(gs.base_hands, 1), 2.0)
        v[m + 5] = float(getattr(blind, "is_pvp", False)
                         and getattr(gs, "pvp_opponent_hands", 0) == 0
                         and getattr(gs, "pvp_started", False))
        if gs.state is State.PVP_WAIT:
            v[m + 6] = 1.0
        elif getattr(gs, "pvp_ready", False):
            v[m + 6] = 0.5
        v[m + 7] = float(getattr(gs, "pvp_started", False))
        if not getattr(gs, "comeback_bonus_given", True):
            v[m + 8] = min(getattr(gs, "comeback_bonus", 0) / 16.0, 2.0)
        v[m + 9] = float(getattr(gs, "match_won", False))
        return v

    # ── batching ─────────────────────────────────────────────────────────────

    @staticmethod
    def batch(observations: Sequence[Obs]) -> Obs:
        """Stack B observations into (B, ...) arrays. Keys are taken from the first."""
        if not observations:
            return {}
        return {k: np.stack([o[k] for o in observations]) for k in observations[0]}

    @staticmethod
    def empty_like(obs: Obs) -> Obs:
        return {k: np.zeros_like(v) for k, v in obs.items()}


# ════════════════════════════════════════════════════════════════════════════════
# helpers
# ════════════════════════════════════════════════════════════════════════════════

def _f(x, default: float = 0.0) -> float:
    if isinstance(x, bool):
        return float(x)
    if isinstance(x, (int, float)):
        return float(x)
    return default


def _tarot_targets(key: str) -> int:
    """How many hand cards this consumable can target — the same 0 / 1 / 2 the engine's
    `_consumable_target_actions` (game.py:1455) enumerates. There is no per-key target
    table in the engine; it enumerates 0-2 for tarots and 0-1 for spectrals regardless of
    the card's real target count (an inherited `legal_actions` approximation, AGENT_NOTES
    §8), so mirroring that is the honest feature."""
    if key in ALL_PLANETS:
        return 0
    if key in ALL_TAROTS:
        return 2
    if key in ALL_SPECTRALS:
        return 1
    return 0


def _shelf_card_cat(item) -> tuple:
    """A shelf playing card (Magic Trick / Illusion) carries `front`/`enhancement`/`seal`
    rather than a `Card`; rebuild the 5-column block from them."""
    front = getattr(item, "front", None)
    rank_i, suit_i = _front_to_indices(front)
    return (rank_i, suit_i,
            _ENH_IDX.get(getattr(item, "enhancement", "None"), 0),
            _EDITION_IDX.get(getattr(item, "edition", "None"), 0),
            _SEAL_IDX.get(getattr(item, "seal", "None"), 0))


def _pack_card_cat(choice) -> tuple:
    card = getattr(choice, "card", None)
    if card is not None:
        return card_cat(card)
    rank_i, suit_i = _front_to_indices(getattr(choice, "front", None)
                                       or getattr(choice, "key", None))
    return (rank_i, suit_i,
            _ENH_IDX.get(getattr(choice, "enhancement", "None"), 0),
            _EDITION_IDX.get(getattr(choice, "edition", "None"), 0),
            _SEAL_IDX.get(getattr(choice, "seal", "None"), 0))


_FRONT_SUIT = {"S": "Spades", "H": "Hearts", "C": "Clubs", "D": "Diamonds"}
_FRONT_RANK = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
               "T": 10, "10": 10, "J": 11, "Q": 12, "K": 13, "A": 14}


def _front_to_indices(front: Optional[str]) -> tuple[int, int]:
    """`'H_A'` -> (rank index, suit index). Unknown -> (0, 0)."""
    if not front or "_" not in front:
        return (0, 0)
    s, _, r = front.partition("_")
    suit = _FRONT_SUIT.get(s.upper())
    rank = _FRONT_RANK.get(r.upper())
    if suit is None or rank is None:
        return (0, 0)
    return (rank - 1, _SUIT_IDX[suit])


def _can_grant(game, choice) -> bool:
    fn = getattr(game, "_can_grant_choice", None)
    if fn is None:
        return True
    try:
        return bool(fn(choice))
    except Exception:                                   # noqa: BLE001 - read-only probe
        return True


def _deck_composition(draw_pile) -> np.ndarray:
    """env_v7's two 8-feature draw-pile blocks, verbatim in meaning."""
    out = np.zeros(16, dtype=np.float32)
    n = max(1, len(draw_pile))
    suit_counts = [0, 0, 0, 0]
    face = ace = number = 0
    foil = holo = poly = gold_enh = wild = seal_gold = seal_red = seal_blue = 0
    for card in draw_pile:
        si = SUIT_ORDER.index(card.suit) if card.suit in SUIT_ORDER else 0
        suit_counts[si] += 1
        if 11 <= card.rank <= 13:
            face += 1
        elif card.rank == 14:
            ace += 1
        else:
            number += 1
        edn = getattr(card, "edition", "None")
        enh = getattr(card, "enhancement", "None")
        seal = getattr(card, "seal", "None")
        if edn == "Foil":
            foil += 1
        elif edn == "Holographic":
            holo += 1
        elif edn == "Polychrome":
            poly += 1
        if enh == "Gold":
            gold_enh += 1
        elif enh == "Wild":
            wild += 1
        if seal == "Gold":
            seal_gold += 1
        elif seal == "Red":
            seal_red += 1
        elif seal == "Blue":
            seal_blue += 1
    out[0:4] = [s / n for s in suit_counts]
    out[4] = face / n
    out[5] = ace / n
    out[6] = number / n
    out[7] = len(draw_pile) / 60.0
    out[8:16] = [foil / n, holo / n, poly / n, gold_enh / n, wild / n,
                 seal_gold / n, seal_red / n, seal_blue / n]
    return out


__all__ = [
    "SetEncoder", "ItemCaps", "DEFAULT_CAPS", "Obs",
    "KEY_VOCAB", "KEY_VOCAB_SIZE", "key_index", "card_cat",
    "CARD_CAT_DIM", "CARD_NUM_DIM", "CARD_CARDINALITIES", "CAT_DTYPE",
    "JOKER_CAT_DIM", "JOKER_NUM_DIM", "CONS_NUM_DIM",
    "SHELF_CAT_DIM", "SHELF_NUM_DIM", "PACK_CAT_DIM", "PACK_NUM_DIM",
    "N_RARITY", "N_SHELF_KIND", "N_PACK_SET", "N_EDITION",
    "SCALAR_DIM", "SCALAR_LAYOUT", "scalar_offsets",
]
