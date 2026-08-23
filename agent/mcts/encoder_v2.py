"""
encoder_v2.py — the STATE_SPEC v1 observation (Phase 5 W1, 2026-08-23).

`docs/STATE_SPEC_v1.md` is the frozen input of the value net V(state) ≈ P(win the MLB
match). This module implements it field-for-field on top of the Phase 4 set encoder
(`encoder_set.SetEncoder`, SETENC_NOTES §0.2), which is reused UNCHANGED for the five
existing item sets and the first 196 scalars. v1 adds:

    blind offers     a sixth item set, cap 3 (Small / Big / Boss-or-Nemesis of this ante):
                     blind kind, boss key, the TAG ON OFFER for Small/Big (what the skip
                     decision needs), chip target, reward $, is_pvp, skip-available, ...
    deck_counts      34  the REMAINING DRAW PILE's composition (ranks / suits / enhancements /
                         editions / seals) — counts, order-free by construction
    discard_counts   17  ranks / suits discarded or played this blind
    ruleset 3 + pvp_start_round 1
    opponent block   level-0 public opponent features (`OpponentView`): lives / skips / $ /
                     phase, the Nemesis in progress, the last 4 Nemeses, shop economics,
                     and a RESERVED 16-wide `opp_belief` block (zeros until level 1 exists)
    race 6, money_detail 6, reserved 32 (zeros)

Everything categorical indexes ONE pre-sized vocabulary `KEY_VOCAB_V2` (jokers, tarots,
planets, spectrals, vouchers, booster types + centers, tags, blinds, the 52 card fronts,
enhancements / editions / seals, `<unk>`, 32 spare ids). Its first `KEY_VOCAB_SIZE` entries
are `encoder_set.KEY_VOCAB` verbatim, so the reused item encoders need no re-indexing.

`layout_fingerprint()` hashes (spec version, scalar layout, item widths, caps, vocabulary);
`value_net.save_checkpoint` stores it and `load_checkpoint` refuses a mismatch, which is
what makes "adding a FIELD restarts training, adding a VALUE does not" enforceable.

Normalisation follows the Phase 4 encoder: clipped [0, 1]-ish linear or log scales, never
NaN / inf (`tests/test_encoder_v2.py` walks a full MLB match and asserts it). Caps are
transport only: position features and capacity ratios are normalised by the FIXED v1
default caps (`DEFAULT_CAPS_V2`), not by the live caps, so an encoder built with larger
caps produces the same numbers in the same rows plus more padding — and the net is
pad-invariant, so the value is unchanged.

Nothing here imports `mp/ev` or `mp/stats` (they import this).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from balatro_sim import game_keys as _gk
from balatro_sim.constants import (
    INTEREST_RATE, MLB_NEMESIS_KEY, MLB_STARTING_LIVES, STARTING_HANDS, blind_base_chips,
)
from balatro_sim.env_v7 import EDITIONS, ENHANCEMENTS, SEALS, SUIT_ORDER
from balatro_sim.game import BOSS_CHIP_MULT, SHOWDOWN_BOSS_BLINDS, BalatroGame, BlindInfo, State
from balatro_sim.shop import effective_price

from .encoder_set import (
    CAT_DTYPE, DEFAULT_CAPS, KEY_VOCAB, KEY_VOCAB_SIZE, PAD_KEY, UNK_KEY,
    CARD_CAT_DIM, CARD_NUM_DIM, JOKER_CAT_DIM, JOKER_NUM_DIM, CONS_NUM_DIM,
    SHELF_CAT_DIM, SHELF_NUM_DIM, PACK_CAT_DIM, PACK_NUM_DIM,
    ItemCaps, Obs, SCALAR_DIM, SCALAR_LAYOUT, SetEncoder, scalar_offsets,
)

STATE_SPEC_VERSION = 1

_LOG_SCORE = math.log1p(100000.0)


# ════════════════════════════════════════════════════════════════════════════════
# Caps
# ════════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ItemCapsV2(ItemCaps):
    """The Phase 4 caps + the blind-offer set. `target` / `total` (the action-mask widths
    of the policy net) deliberately exclude `blinds`: no v1 action targets a blind slot."""
    blinds: int = 3

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "ItemCapsV2":
        return cls(**d) if d else cls()


DEFAULT_CAPS_V2 = ItemCapsV2()


# ════════════════════════════════════════════════════════════════════════════════
# Vocabulary (pre-sized, never resized)
# ════════════════════════════════════════════════════════════════════════════════

N_SPARE_KEYS = 32

_CARD_RANK_CHARS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
_CARD_SUIT_CHARS = ["S", "H", "C", "D"]            # SUIT_ORDER: Spades, Hearts, Clubs, Diamonds
CARD_FRONT_KEYS: list[str] = [f"{s}_{r}" for s in _CARD_SUIT_CHARS for r in _CARD_RANK_CHARS]  # 52

BLIND_KEYS_V2: list[str] = [b["key"] for b in _gk.BLINDS] + [MLB_NEMESIS_KEY]   # 30 + Nemesis
TAG_KEYS_V2: list[str] = list(_gk.TAG_KEYS)                                      # 24

#: The v1 vocabulary. The first KEY_VOCAB_SIZE entries ARE encoder_set.KEY_VOCAB (asserted
#: below) so the reused joker / consumable / shelf / pack encoders index it unchanged.
KEY_VOCAB_V2: list[str] = (
    list(KEY_VOCAB)                                  # <pad>, <unk>, jokers, tarots, planets,
                                                     # spectrals, vouchers, booster TYPES
    + list(_gk.BOOSTER_CENTER_KEYS)                  # 32 booster centers (art variants;
                                                     #   reserved — the engine keys by type)
    + TAG_KEYS_V2                                    # 24
    + BLIND_KEYS_V2                                  # 31 (Small, Big, 28 bosses, Nemesis)
    + CARD_FRONT_KEYS                                # 52 (reserved: cards use the card block)
    + list(_gk.ENHANCEMENT_KEYS)                     # 8  (reserved)
    + list(_gk.EDITION_KEYS)                         # 5  (reserved)
    + list(_gk.SEAL_KEYS)                            # 4  (reserved)
    + [f"<spare_{i}>" for i in range(N_SPARE_KEYS)]  # 32 (mod content later, via SPARE_KEY_MAP)
)
assert KEY_VOCAB_V2[:KEY_VOCAB_SIZE] == list(KEY_VOCAB)
assert len(set(KEY_VOCAB_V2)) == len(KEY_VOCAB_V2), "duplicate key in KEY_VOCAB_V2"
KEY_IDX_V2: dict[str, int] = {k: i for i, k in enumerate(KEY_VOCAB_V2)}
KEY_VOCAB_SIZE_V2 = len(KEY_VOCAB_V2)
SPARE_KEY_BASE = KEY_VOCAB_SIZE_V2 - N_SPARE_KEYS

#: New content (a modded joker, say) is given a SPARE id here instead of regenerating the
#: vocabulary — that is the "adding a VALUE does not restart training" promise. Empty today.
SPARE_KEY_MAP: dict[str, int] = {}


def key_index_v2(key: Optional[str]) -> int:
    if not key:
        return PAD_KEY
    i = KEY_IDX_V2.get(key)
    if i is not None:
        return i
    i = SPARE_KEY_MAP.get(key)
    return UNK_KEY if i is None else i


# ════════════════════════════════════════════════════════════════════════════════
# Blind-offer set
# ════════════════════════════════════════════════════════════════════════════════

BLIND_CAT_DIM = 2       # kind (0 pad, 1 Small, 2 Big, 3 Boss), status (0 pad, 1 upcoming, 2 current, 3 done)
BLIND_NUM_DIM = 8
N_BLIND_KIND = 4
N_BLIND_STATUS = 4
_BLIND_KIND_IDX = {"Small": 1, "Big": 2, "Boss": 3}
BLIND_STATUS_UPCOMING, BLIND_STATUS_CURRENT, BLIND_STATUS_DONE = 1, 2, 3


# ════════════════════════════════════════════════════════════════════════════════
# Scalars
# ════════════════════════════════════════════════════════════════════════════════

_ENH_COUNTED = list(ENHANCEMENTS[1:])       # 8  Bonus..Lucky
_EDITION_COUNTED = list(EDITIONS)           # 5  None, Foil, Holographic, Polychrome, Negative
_SEAL_COUNTED = list(SEALS[1:])             # 4  Gold, Red, Blue, Purple
_SUIT_IDX0 = {s: i for i, s in enumerate(SUIT_ORDER)}
_ENH_IDX0 = {e: i for i, e in enumerate(_ENH_COUNTED)}
_EDITION_IDX0 = {e: i for i, e in enumerate(_EDITION_COUNTED)}
_SEAL_IDX0 = {s: i for i, s in enumerate(_SEAL_COUNTED)}

DECK_COUNTS_DIM = 13 + 4 + 8 + 5 + 4        # 34
DISCARD_COUNTS_DIM = 13 + 4                 # 17
RULESETS = ["vanilla", "mlb", "the_order"]
OPP_PHASES = ["selecting", "small", "big", "nemesis", "shop", "waiting"]
N_OPP_HISTORY = 4
OPP_HISTORY_FIELDS = 6                       # ante, their score, their hands used, my score, outcome, early end
OPP_BASIC_DIM = 6 + len(OPP_PHASES)          # known, lives, skips, $, log $, ante + phase one-hot = 12

#: Every block of the spec, in spec order. The first 16 blocks ARE encoder_set.SCALAR_LAYOUT.
SCALAR_LAYOUT_V2: list[tuple[str, int]] = list(SCALAR_LAYOUT) + [
    ("deck_counts",      DECK_COUNTS_DIM),               # 34
    ("discard_counts",   DISCARD_COUNTS_DIM),            # 17
    ("ruleset",          len(RULESETS)),                 # 3  one-hot
    ("pvp_start_round",  1),
    ("opp_basic",        OPP_BASIC_DIM),                 # 12
    ("opp_nemesis",      4),
    ("opp_history",      N_OPP_HISTORY * OPP_HISTORY_FIELDS),   # 24
    ("opp_belief",       16),                            # RESERVED (zeros)
    ("opp_econ",         4),
    ("race",             6),
    ("money_detail",     6),
    ("reserved",         32),                            # RESERVED (zeros)
]

_OFF_V2: dict[str, int] = {}
_o = 0
for _name, _w in SCALAR_LAYOUT_V2:
    _OFF_V2[_name] = _o
    _o += _w
SCALAR_DIM_V2 = _o
del _o, _name, _w
assert all(_OFF_V2[n] == o for n, o in scalar_offsets().items())   # shared prefix


def scalar_offsets_v2() -> dict[str, int]:
    return dict(_OFF_V2)


def scalar_layout_table() -> list[tuple[str, int, int]]:
    """[(name, width, offset)] — what VALUE_NOTES.md prints."""
    return [(n, w, _OFF_V2[n]) for n, w in SCALAR_LAYOUT_V2]


# ════════════════════════════════════════════════════════════════════════════════
# Fingerprint
# ════════════════════════════════════════════════════════════════════════════════

ITEM_WIDTHS_V2: dict[str, tuple[int, ...]] = {
    "hand":        (CARD_CAT_DIM, CARD_NUM_DIM),
    "jokers":      (1, JOKER_CAT_DIM, JOKER_NUM_DIM),
    "consumables": (1, CONS_NUM_DIM),
    "shelf":       (1, SHELF_CAT_DIM, CARD_CAT_DIM, SHELF_NUM_DIM),
    "packs":       (1, PACK_CAT_DIM, CARD_CAT_DIM, PACK_NUM_DIM),
    "blinds":      (1, 1, BLIND_CAT_DIM, BLIND_NUM_DIM),
}


def layout_fingerprint(caps: ItemCapsV2 | dict | None = None) -> str:
    """sha256 over (spec version, scalar layout, item widths, caps, vocabulary). Stored in
    every value checkpoint and asserted on load."""
    caps = caps if isinstance(caps, ItemCaps) else ItemCapsV2.from_dict(caps)
    payload = {
        "state_spec_version": STATE_SPEC_VERSION,
        "scalar_layout": SCALAR_LAYOUT_V2,
        "item_widths": {k: list(v) for k, v in ITEM_WIDTHS_V2.items()},
        "caps": caps.as_dict(),
        "key_vocab": KEY_VOCAB_V2,
        "cardinalities": {"blind_kind": N_BLIND_KIND, "blind_status": N_BLIND_STATUS},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


# ════════════════════════════════════════════════════════════════════════════════
# The opponent's PUBLIC information
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class NemesisLive:
    """The Nemesis in progress (both players inside it)."""
    their_score: int = 0
    their_hands_left: int = 0
    my_score: int = 0
    my_hands_left: int = 0


@dataclass
class NemesisRecord:
    """One resolved Nemesis, from my side."""
    ante: int
    their_score: int
    their_hands_used: int
    my_score: int
    outcome: int            # +1 I took a life, 0 tie, -1 I lost one
    early_end: bool         # the PvP ended with hands still unplayed on one side


@dataclass
class OpponentView:
    """Level-0 PUBLIC opponent information — exactly what `MLBMatch.player_view` (the mod's
    `enemyInfo` / lobby HUD) and the match's Nemesis log reveal. `known=False` (the default
    instance) is the vanilla / solo case and encodes as all zeros."""
    known: bool = False
    lives: int = 0
    skips: int = 0
    dollars: int = 0
    ante: int = 0
    blind_idx: int = 0
    state: str = ""                  # `State` name of the opponent's game
    pvp_ready: bool = False
    pvp_exhausted: bool = False
    chips_scored: int = 0
    hands_left: int = 0
    comeback_bonus: int = 0
    comeback_pending: int = 0
    current_nemesis: Optional[NemesisLive] = None
    last_nemeses: list = field(default_factory=list)   # NemesisRecord, most recent first, <= 4
    my_last_loss_ante: Optional[int] = None             # for `race`: antes since I last lost a life
    sells_per_ante: int = 0          # opp_econ — MP.GAME.enemy.sells_per_ante
    spent_in_shop: int = 0           #            MP.GAME.enemy.spent_in_shop
    sells_total: int = 0
    spent_total: int = 0

    @property
    def phase(self) -> Optional[str]:
        """One of OPP_PHASES, or None (unknown / game over)."""
        if not self.known:
            return None
        s = self.state
        if s == "PVP_WAIT" or self.pvp_ready:
            return "waiting"
        if s == "BLIND_SELECT":
            return "selecting"
        if s == "SELECTING_HAND":
            return ("small", "big", "nemesis")[min(max(self.blind_idx, 0), 2)]
        if s in ("SHOP", "BOOSTER_OPEN", "ROUND_EVAL"):
            return "shop"
        return None


NO_OPPONENT = OpponentView()


def opponent_view(match, player: int) -> OpponentView:
    """Build the opponent block for `player` from `match` using ONLY public information:
    `MLBMatch.player_view` of both players (lives / skips / $ / phase / live Nemesis score
    and hands / shop economics) and the match's Nemesis log. Never the opponent's hand,
    jokers, consumables, deck or shop — `tests/test_encoder_v2.py` mutates those and
    asserts this function cannot tell."""
    other = 1 - player
    them = match.player_view(other)
    me = match.player_view(player)
    live = None
    if getattr(match, "pvp_active", False):
        live = NemesisLive(their_score=them.chips_scored, their_hands_left=them.hands_left,
                           my_score=me.chips_scored, my_hands_left=me.hands_left)
    detail = getattr(match, "pvp_detail", None)
    if detail is None:           # a match object without the Phase 5 log: degrade gracefully
        detail = [(a, l, s0, s1, 0, 0, False) for (a, l, s0, s1) in match.pvp_log]
    records: list[NemesisRecord] = []
    my_last_loss: Optional[int] = None
    for ante, loser, s0, s1, h0, h1, early in detail:
        scores = (s0, s1)
        hands = (h0, h1)
        outcome = 0 if loser is None else (1 if loser == other else -1)
        if loser == player:
            my_last_loss = ante
        records.append(NemesisRecord(ante=ante, their_score=scores[other],
                                     their_hands_used=hands[other], my_score=scores[player],
                                     outcome=outcome, early_end=bool(early)))
    records.reverse()
    tracked = getattr(me, "last_life_loss_ante", None)
    if tracked is not None:                 # the match's own counter (any blind), else the log
        my_last_loss = tracked
    return OpponentView(
        known=True, lives=them.lives, skips=them.skips, dollars=them.dollars, ante=them.ante,
        blind_idx=them.blind_idx, state=them.state, pvp_ready=them.pvp_ready,
        pvp_exhausted=them.pvp_exhausted, chips_scored=them.chips_scored,
        hands_left=them.hands_left, comeback_bonus=them.comeback_bonus,
        comeback_pending=them.comeback_pending, current_nemesis=live,
        last_nemeses=records[:N_OPP_HISTORY], my_last_loss_ante=my_last_loss,
        sells_per_ante=getattr(them, "sells_per_ante", 0),
        spent_in_shop=getattr(them, "spent_in_shop", 0),
        sells_total=getattr(them, "sells_total", 0),
        spent_total=getattr(them, "spent_total", 0),
    )


# ════════════════════════════════════════════════════════════════════════════════
# The encoder
# ════════════════════════════════════════════════════════════════════════════════

def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _log_score(x) -> float:
    return math.log1p(max(float(x), 0.0)) / _LOG_SCORE


class SetEncoderV2(SetEncoder):
    """`enc(game, opp=None) -> Obs` per STATE_SPEC v1. Reuses `SetEncoder`'s five item
    encoders and its 196 scalars verbatim, then appends the v1 blocks."""

    name = "set_v2"
    is_set = True
    dim = None
    state_spec_version = STATE_SPEC_VERSION

    def __init__(self, caps: ItemCapsV2 | dict | None = None):
        if isinstance(caps, ItemCapsV2):
            self.caps = caps
        elif isinstance(caps, ItemCaps):
            self.caps = ItemCapsV2(**caps.as_dict())
        else:
            self.caps = ItemCapsV2.from_dict(caps)
        self.fingerprint = layout_fingerprint(self.caps)

    # ── metadata ─────────────────────────────────────────────────────────────

    def describe(self) -> dict:
        return {"name": self.name, "caps": self.caps.as_dict(),
                "scalar_dim": SCALAR_DIM_V2, "key_vocab": KEY_VOCAB_SIZE_V2,
                "state_spec_version": STATE_SPEC_VERSION, "fingerprint": self.fingerprint}

    @classmethod
    def from_description(cls, desc: dict) -> "SetEncoderV2":
        if desc.get("name", cls.name) != cls.name:
            raise ValueError(f"not a {cls.name} description: {desc.get('name')!r}")
        if desc.get("state_spec_version", STATE_SPEC_VERSION) != STATE_SPEC_VERSION:
            raise ValueError(f"STATE_SPEC_VERSION {desc.get('state_spec_version')} != "
                             f"{STATE_SPEC_VERSION}")
        enc = cls(caps=ItemCapsV2.from_dict(desc.get("caps")))
        want = desc.get("fingerprint")
        if want is not None and want != enc.fingerprint:
            raise ValueError("encoder layout fingerprint mismatch: the checkpoint was written "
                             f"against {want[:12]}…, this code is {enc.fingerprint[:12]}…")
        return enc

    # ── the encode ───────────────────────────────────────────────────────────

    def __call__(self, game: BalatroGame, opp: Optional[OpponentView] = None) -> Obs:
        caps = self.caps
        obs: Obs = {}
        self._encode_hand(game, caps, obs)
        self._encode_jokers(game, caps, obs)
        self._encode_consumables(game, caps, obs)
        self._encode_shelf(game, caps, obs)
        self._encode_packs(game, caps, obs)
        if caps != DEFAULT_CAPS_V2:
            _renormalise_positions(obs, caps)
        self._encode_blinds(game, caps, obs)
        obs["scalars"] = self._encode_scalars_v2(game, caps, opp or NO_OPPONENT)
        return obs

    # ── blind offers ─────────────────────────────────────────────────────────

    def _encode_blinds(self, game, caps: ItemCapsV2, obs: Obs) -> None:
        n_cap = caps.blinds
        key = np.zeros(n_cap, dtype=CAT_DTYPE)
        tag = np.zeros(n_cap, dtype=CAT_DTYPE)
        cat = np.zeros((n_cap, BLIND_CAT_DIM), dtype=CAT_DTYPE)
        num = np.zeros((n_cap, BLIND_NUM_DIM), dtype=np.float32)
        mask = np.zeros(n_cap, dtype=np.float32)

        cur = game.current_blind
        state = game.state
        # After a Boss the engine advances `ante` on entering the shop while `blind_idx`
        # still points at the beaten Boss: the offers on the table are then the NEW ante's
        # three blinds (its tags and boss are already drawn), none current, none done.
        # A tag pack opened at the Boss's blind-select screen is NOT that case.
        post_boss = bool(cur.is_boss) and (
            state is State.SHOP
            or (state is State.BOOSTER_OPEN
                and getattr(game, "_booster_return_state", None) is not State.BLIND_SELECT))
        cur_idx = -1 if post_boss else game.blind_idx
        at_select = state is State.BLIND_SELECT and not getattr(game, "pvp_ready", False)
        tags = getattr(game, "blind_tags", None) or {}
        ante = game.ante
        scaling = getattr(game, "blind_scaling", 1)
        ante_scaling = getattr(game, "ante_scaling", 1)
        mlb = getattr(game, "mlb", False)
        nemesis_ante = mlb and ante >= getattr(game, "pvp_start_round", 2)
        for slot, kind in enumerate(("Small", "Big", "Boss")):
            if slot >= n_cap:
                break
            mask[slot] = 1.0
            is_current = slot == cur_idx
            is_done = slot < cur_idx
            cat[slot, 0] = _BLIND_KIND_IDX[kind]
            cat[slot, 1] = (BLIND_STATUS_CURRENT if is_current
                            else BLIND_STATUS_DONE if is_done else BLIND_STATUS_UPCOMING)
            if kind == "Boss":
                is_pvp = bool(cur.is_pvp) if is_current else bool(nemesis_ante)
                boss_key = (cur.boss_key if is_current else
                            (MLB_NEMESIS_KEY if is_pvp else (getattr(game, "boss_blind", "") or "")))
                key[slot] = key_index_v2(boss_key)
            else:
                is_pvp = False
                tag[slot] = key_index_v2(tags.get(kind))
            if is_current:
                target = cur.chips_target
                showdown = bool(cur.is_showdown)
                disabled = bool(cur.disabled)
                reward = cur.money_reward
            else:
                base = int(blind_base_chips(ante, slot, scaling) * ante_scaling)
                showdown = False
                disabled = False
                if kind == "Boss":
                    if is_pvp:
                        target = 0
                    else:
                        target = int(base * BOSS_CHIP_MULT.get(boss_key, 1.0))
                        showdown = boss_key in SHOWDOWN_BOSS_BLINDS
                else:
                    target = base
                reward = BlindInfo("", kind, target, is_boss=(kind == "Boss"),
                                   is_showdown=showdown, is_pvp=is_pvp).money_reward
            if kind == "Small" and getattr(game, "no_small_blind_reward", False):
                reward = 0
            row = num[slot]
            row[0] = _log_score(target)
            row[1] = min(reward / 8.0, 2.0)
            row[2] = float(is_pvp)
            row[3] = float(at_select and is_current and kind != "Boss")   # skip available
            row[4] = float(is_current)
            row[5] = float(is_done)
            row[6] = float(showdown)
            row[7] = float(disabled)

        obs["blind_key"], obs["blind_tag"], obs["blind_cat"] = key, tag, cat
        obs["blind_num"], obs["blind_mask"] = num, mask

    # ── scalars ──────────────────────────────────────────────────────────────

    def _encode_scalars_v2(self, game, caps: ItemCapsV2, opp: OpponentView) -> np.ndarray:
        v = np.zeros(SCALAR_DIM_V2, dtype=np.float32)
        v[:SCALAR_DIM] = self._encode_scalars(game, DEFAULT_CAPS)   # fixed-cap normalisation
        off = _OFF_V2
        gs = game

        # deck_counts (34): the remaining DRAW PILE, counts (order-free by construction)
        _count_cards(gs.deck, v, off["deck_counts"], full=True)
        # discard_counts (17): played / discarded this blind
        _count_cards(gs.discard_pile, v, off["discard_counts"], full=False)

        # ruleset (3) + pvp_start_round (1)
        r = off["ruleset"]
        if getattr(gs, "queue_scope", "ante") == "run":
            v[r + 2] = 1.0
        elif getattr(gs, "mlb", False):
            v[r + 1] = 1.0
        else:
            v[r + 0] = 1.0
        if getattr(gs, "mlb", False):
            v[off["pvp_start_round"]] = min(getattr(gs, "pvp_start_round", 2) / 8.0, 2.0)

        # opponent block — all zeros unless an OpponentView is known
        if opp.known:
            b = off["opp_basic"]
            v[b + 0] = 1.0
            v[b + 1] = min(opp.lives / max(MLB_STARTING_LIVES, 1), 2.0)
            v[b + 2] = min(opp.skips / 8.0, 2.0)
            v[b + 3] = min(max(opp.dollars, 0) / 50.0, 2.0)
            v[b + 4] = math.log1p(max(opp.dollars, 0)) / math.log1p(200)
            v[b + 5] = min(opp.ante / 8.0, 4.0)
            ph = opp.phase
            if ph is not None:
                v[b + 6 + OPP_PHASES.index(ph)] = 1.0

            live = opp.current_nemesis
            if live is not None:
                n = off["opp_nemesis"]
                v[n + 0] = _log_score(live.their_score)
                v[n + 1] = min(live.their_hands_left / max(STARTING_HANDS, 1), 2.0)
                v[n + 2] = _log_score(live.my_score)
                v[n + 3] = min(live.my_hands_left / max(STARTING_HANDS, 1), 2.0)

            h = off["opp_history"]
            for i, rec in enumerate(opp.last_nemeses[:N_OPP_HISTORY]):
                o = h + i * OPP_HISTORY_FIELDS
                v[o + 0] = min(rec.ante / 8.0, 4.0)
                v[o + 1] = _log_score(rec.their_score)
                v[o + 2] = min(rec.their_hands_used / max(STARTING_HANDS, 1), 2.0)
                v[o + 3] = _log_score(rec.my_score)
                v[o + 4] = float(rec.outcome)
                v[o + 5] = float(rec.early_end)

            e = off["opp_econ"]
            v[e + 0] = min(opp.sells_per_ante / 5.0, 2.0)
            v[e + 1] = min(max(opp.spent_in_shop, 0) / 50.0, 2.0)
            v[e + 2] = math.log1p(max(opp.sells_total, 0)) / math.log1p(50)
            v[e + 3] = math.log1p(max(opp.spent_total, 0)) / math.log1p(500)
        # opp_belief (16): reserved, zeros

        # race (6)
        rc = off["race"]
        v[rc + 0] = min(getattr(gs, "lives", 0) / max(MLB_STARTING_LIVES, 1), 2.0)
        v[rc + 1] = min(opp.lives / max(MLB_STARTING_LIVES, 1), 2.0) if opp.known else 0.0
        v[rc + 2] = min(gs.ante / 8.0, 4.0)
        if not getattr(gs, "comeback_bonus_given", True):
            v[rc + 3] = min(4.0 * getattr(gs, "comeback_bonus", 0) / 16.0, 2.0)
        v[rc + 4] = float(getattr(gs, "life_lost_this_round", False))
        last = opp.my_last_loss_ante if opp.known else None
        since = gs.ante if last is None else max(gs.ante - last, 0)
        v[rc + 5] = min(since / 8.0, 2.0)

        # money_detail (6)
        m = off["money_detail"]
        dollars = gs.dollars
        cap = gs.interest_cap
        if getattr(gs, "no_interest", False):
            interest = 0
            to_next = 0
        else:
            interest = min(max(dollars, 0) // INTEREST_RATE, cap)
            to_next = 0 if interest >= cap else INTEREST_RATE - (max(dollars, 0) % INTEREST_RATE)
        v[m + 0] = min(interest / 10.0, 2.0)
        v[m + 1] = to_next / INTEREST_RATE
        v[m + 2] = min(max(0, gs.reroll_cost - gs.reroll_discount) / 10.0, 2.0)
        v[m + 3] = min(gs.free_rerolls_remaining / 3.0, 2.0)
        v[m + 4] = min(gs.shop_discount, 1.0)
        for item in gs.current_shop:
            if item.kind == "voucher" and not item.sold:
                v[m + 5] = min(effective_price(gs, item) / 20.0, 2.0)
                break
        # reserved (32): zeros
        return v

    # `batch(observations)` is inherited from SetEncoder.


# ════════════════════════════════════════════════════════════════════════════════
# helpers
# ════════════════════════════════════════════════════════════════════════════════

def _count_cards(cards, v: np.ndarray, base: int, full: bool) -> None:
    """Rank (13) + suit (4) counts, plus enhancements (8) / editions (5) / seals (4) when
    `full`. Counts, so the result cannot depend on the order of `cards`."""
    rank = np.zeros(13, dtype=np.float32)
    suit = np.zeros(4, dtype=np.float32)
    if full:
        enh = np.zeros(8, dtype=np.float32)
        edn = np.zeros(5, dtype=np.float32)
        seal = np.zeros(4, dtype=np.float32)
    for c in cards:
        r = c.rank
        if 2 <= r <= 14:
            rank[r - 2] += 1.0
        si = _SUIT_IDX0.get(c.suit)
        if si is not None:
            suit[si] += 1.0
        if full:
            i = _ENH_IDX0.get(c.enhancement)
            if i is not None:
                enh[i] += 1.0
            i = _EDITION_IDX0.get(c.edition)
            if i is not None:
                edn[i] += 1.0
            i = _SEAL_IDX0.get(c.seal)
            if i is not None:
                seal[i] += 1.0
    np.minimum(rank / 8.0, 1.0, out=rank)
    np.minimum(suit / 26.0, 1.0, out=suit)
    v[base:base + 13] = rank
    v[base + 13:base + 17] = suit
    if full:
        np.minimum(enh / 16.0, 1.0, out=enh)
        edn[0] = min(edn[0] / 52.0, 1.0)             # plain cards
        edn[1:] = np.minimum(edn[1:] / 16.0, 1.0)
        np.minimum(seal / 16.0, 1.0, out=seal)
        v[base + 17:base + 25] = enh
        v[base + 25:base + 30] = edn
        v[base + 30:base + 34] = seal


def _renormalise_positions(obs: Obs, caps: ItemCapsV2) -> None:
    """The Phase 4 item encoders write `slot / cap` position features. Caps are transport
    only, so v1 normalises positions by the FIXED default caps instead: an encoder with
    bigger caps then writes the same numbers into the same rows (plus more padding)."""
    d = DEFAULT_CAPS_V2
    for num_key, mask_key, cap, dcap, col in (
            ("hand_num", "hand_mask", caps.hand, d.hand, 8),
            ("joker_num", "joker_mask", caps.jokers, d.jokers, 0),
            ("cons_num", "cons_mask", caps.consumables, d.consumables, 0),
            ("shelf_num", "shelf_mask", caps.shelf, d.shelf, 0),
            ("pack_num", "pack_mask", caps.packs, d.packs, 0)):
        if cap != dcap:
            arr = obs[num_key]
            arr[:, col] = (np.arange(cap, dtype=np.float32) / dcap) * obs[mask_key]


def collate(obs_list: Sequence[Obs], device="cpu") -> dict:
    """Stack B observations into a batched dict of torch tensors on `device` (one packed
    host→device copy per dtype off the CPU — `policy_set._stack_obs`)."""
    from .policy_set import _stack_obs
    return _stack_obs(list(obs_list), device)


__all__ = [
    "STATE_SPEC_VERSION", "SetEncoderV2", "ItemCapsV2", "DEFAULT_CAPS_V2",
    "KEY_VOCAB_V2", "KEY_VOCAB_SIZE_V2", "KEY_IDX_V2", "key_index_v2", "SPARE_KEY_MAP",
    "N_SPARE_KEYS", "SPARE_KEY_BASE",
    "SCALAR_LAYOUT_V2", "SCALAR_DIM_V2", "scalar_offsets_v2", "scalar_layout_table",
    "BLIND_CAT_DIM", "BLIND_NUM_DIM", "N_BLIND_KIND", "N_BLIND_STATUS",
    "ITEM_WIDTHS_V2", "layout_fingerprint",
    "OpponentView", "NemesisLive", "NemesisRecord", "NO_OPPONENT", "opponent_view",
    "OPP_PHASES", "N_OPP_HISTORY", "OPP_HISTORY_FIELDS", "RULESETS",
    "collate", "Obs",
]
