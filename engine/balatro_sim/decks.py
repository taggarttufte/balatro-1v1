"""
decks.py — the 15 ``b_*`` decks (Phase 2 W3): catalogue + the engine-side hooks.

Ground truth: ``Back:apply_to_run`` (back.lua:173-288) for the run-start effects and
``Back:trigger_effect`` (back.lua:109-168) for the two per-event decks (Anaglyph's Double
Tag on ``eval`` after a boss, Plasma's ``final_scoring_step`` balance).  The deck table is
``game.lua:627-641`` (``pools.BACKS``), and ``Game:start_run`` applies the deck AFTER the
stake modifiers (game.lua:2050-2058).

Division of labour with the generation layer (``mp/rng/generate.py``, NOT owned here):

* ``generate.apply_deck`` / ``build_starting_deck`` already handle everything that changes
  which RNG calls happen: deck vouchers into ``used_vouchers`` (+ shop rates / shelf size),
  deck consumables into ``owned_consumables``/``used_jokers``, ``spectral_rate``, the
  no-faces / Checkered / Erratic starting decks.  ``BalatroGame._init_game_vars`` reads
  those back (``consumable_hand``, ``vouchers``).
* This module owns the ENGINE side: ``starting_params`` deltas (hands / discards / dollars /
  joker slots / hand size / consumable slots), ``G.GAME.modifiers`` (no_interest,
  money_per_hand, money_per_discard), ``ante_scaling`` (Plasma ×2 blind targets, read by
  ``BalatroGame._prepare_next_blind``), the Plasma balance flag (``scoring.score_hand``),
  Anaglyph's Double Tag (``BalatroGame._end_round``), the engine-side half of the deck
  vouchers (Crystal Ball's +1 consumable slot), and the creation (``sort_id``) order of
  the Checkered deck (see ``creation_order``).

Status per deck: see ``DECK_STATUS`` at the bottom and ``engine/DECKS_NOTES.md``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from . import game_keys as _gk

if TYPE_CHECKING:  # pragma: no cover
    from .game import BalatroGame

_BACKS = _gk.pools.BACKS          # game.lua:627-641, 15 entries in `order`
_BACK_BY_KEY = _gk.pools.BACK_BY_KEY


@dataclass(frozen=True)
class DeckSpec:
    """One ``b_*`` back.  Field names follow ``Back:apply_to_run``'s config keys."""
    key: str
    name: str
    order: int
    description: str
    # starting_params deltas (back.lua:179-181, 196-198, 213-215, 266-278)
    hands: int = 0
    discards: int = 0
    dollars: int = 0
    joker_slot: int = 0
    hand_size: int = 0
    consumable_slot: int = 0
    reroll_discount: int = 0
    ante_scaling: int = 1              # back.lua:270-272 (Plasma: 2)
    # G.GAME.modifiers (back.lua:279-287)
    no_interest: bool = False
    money_per_hand: Optional[int] = None      # `extra_hand_bonus`; vanilla pays $1 (state_events.lua:1166)
    money_per_discard: Optional[int] = None   # `extra_discard_bonus`; vanilla pays nothing (:1170)
    # generation-side (mirrored in generate.DECK_EFFECTS; listed for the catalogue)
    voucher: Optional[str] = None
    vouchers: tuple = ()
    consumables: tuple = ()
    spectral_rate: Optional[int] = None
    remove_faces: bool = False
    randomize_rank_suit: bool = False
    # by-name specials (back.lua:109-168, 239-256)
    checkered: bool = False            # Clubs->Spades, Diamonds->Hearts after creation
    anaglyph: bool = False             # Double Tag after every boss (`eval` context)
    plasma: bool = False               # chips/mult balanced at `final_scoring_step`

    @property
    def all_vouchers(self) -> tuple:
        return ((self.voucher,) if self.voucher else ()) + tuple(self.vouchers)


_DESCRIPTIONS = {
    "b_red": "+1 discard every round",
    "b_blue": "+1 hand every round",
    "b_yellow": "Start with extra $10",
    "b_green": "At end of round: $2 per remaining Hand, $1 per remaining Discard, earn no Interest",
    "b_black": "+1 Joker slot, -1 hand every round",
    "b_magic": "Start run with the Crystal Ball voucher and 2 copies of The Fool",
    "b_nebula": "Start run with the Telescope voucher, -1 consumable slot",
    "b_ghost": "Spectral cards may appear in the shop (rate 2), start with a Hex card",
    "b_abandoned": "Start run with no Face Cards in your deck",
    "b_checkered": "Start run with 26 Spades and 26 Hearts in deck",
    "b_zodiac": "Start run with Tarot Merchant, Planet Merchant and Overstock",
    "b_painted": "+2 hand size, -1 Joker slot",
    "b_anaglyph": "After defeating each Boss Blind, gain a Double Tag",
    "b_plasma": "Balance Chips and Mult when calculating score for played hand; x2 base Blind size",
    "b_erratic": "All Ranks and Suits in deck are randomized",
}


def _spec_from_pool(entry: dict) -> DeckSpec:
    cfg = dict(entry["config"] or {})
    key = entry["key"]
    kw = dict(
        key=key, name=entry["name"], order=entry["order"], description=_DESCRIPTIONS[key],
        hands=cfg.pop("hands", 0), discards=cfg.pop("discards", 0), dollars=cfg.pop("dollars", 0),
        joker_slot=cfg.pop("joker_slot", 0), hand_size=cfg.pop("hand_size", 0),
        consumable_slot=cfg.pop("consumable_slot", 0), reroll_discount=cfg.pop("reroll_discount", 0),
        ante_scaling=cfg.pop("ante_scaling", 1),
        no_interest=bool(cfg.pop("no_interest", False)),
        money_per_hand=cfg.pop("extra_hand_bonus", None),
        money_per_discard=cfg.pop("extra_discard_bonus", None),
        voucher=cfg.pop("voucher", None), vouchers=tuple(cfg.pop("vouchers", ())),
        consumables=tuple(cfg.pop("consumables", ())),
        spectral_rate=cfg.pop("spectral_rate", None),
        remove_faces=bool(cfg.pop("remove_faces", False)),
        randomize_rank_suit=bool(cfg.pop("randomize_rank_suit", False)),
        checkered=(key == "b_checkered"), anaglyph=(key == "b_anaglyph"), plasma=(key == "b_plasma"),
    )
    if cfg:   # a config key Back:apply_to_run reads that this port does not model
        raise ValueError(f"{key}: unmodelled deck config keys {sorted(cfg)}")
    return DeckSpec(**kw)


DECKS: dict[str, DeckSpec] = {e["key"]: _spec_from_pool(e) for e in _BACKS}
DECK_KEYS: list[str] = [e["key"] for e in _BACKS]           # in `order` (b_red .. b_erratic)
assert len(DECKS) == 15, len(DECKS)


def deck_spec(key: str) -> DeckSpec:
    try:
        return DECKS[key]
    except KeyError:
        raise KeyError(f"unknown deck {key!r}; valid: {DECK_KEYS}") from None


# ── Starting deck creation order (sort_id) ────────────────────────────────────────────

_SUIT_SWAP = {"C": "S", "D": "H"}   # back.lua:244-256


def creation_order(deck_key: str, deck_keys: list) -> list:
    """``G.playing_cards`` in creation (``sort_id``) order for a starting deck given in ANY
    order (``generate.start_run`` returns the post-'shuffle' order).

    game.lua:2330-2378 sorts the 52 protos by the ``suit..rank`` STRING of the *original*
    proto (``C2..C9,CA,CJ,CK,CQ,CT,D2,...,S...``), then creates the cards in that order.
    The Checkered swap (``change_suit``) runs AFTER creation and leaves ``sort_id`` alone,
    so for Checkered the creation order is ``S(ex-C)×13, H(ex-D)×13, H×13, S×13`` — NOT the
    order a sort on the post-swap keys would give.  Every ``pseudoshuffle`` sorts by
    ``sort_id`` first, so getting this wrong changes every dealt hand.  For the Erratic
    deck the protos are drawn first and then sorted the same way, so the plain sort is
    right (duplicates are interchangeable)."""
    if deck_key == "b_checkered":
        ranks = sorted({k[2:] for k in deck_keys})            # '2'..'9','A','J','K','Q','T'
        order = [f"{_SUIT_SWAP.get(s, s)}_{r}" for s in "CDHS" for r in ranks]
        if sorted(order) != sorted(deck_keys):
            raise ValueError("Checkered starting deck is not the expected 26 S + 26 H")
        return order
    return sorted(deck_keys, key=lambda k: k[0] + k[2:])


# ── Run-start hook (the engine side of Back:apply_to_run) ─────────────────────────────

def apply_deck_to_game(game: "BalatroGame") -> DeckSpec:
    """Apply the deck's ENGINE-side run-start effects.  Called from
    ``BalatroGame._init_game_vars`` right after ``generate.start_run`` (which already
    applied the generation side and whose results — deck consumables, vouchers — the
    caller has copied into ``consumable_hand`` / ``vouchers``).

    Order follows back.lua:175-288: voucher(s) → hands → (consumables) → dollars → faces →
    spectral_rate → discards → reroll_discount → joker_slot → hand_size → ante_scaling →
    consumable_slot → no_interest → money_per_hand/discard.  Nothing here touches RNG."""
    spec = deck_spec(game.deck_key)
    # Deck vouchers: generation side (rates, shelf size, used_vouchers) is done by
    # generate.apply_deck; the engine side of each is applied here.  Only Crystal Ball
    # has one (+1 consumable slot, card.lua:1922).  Telescope / Tarot & Planet Merchant /
    # Overstock are generation-only.
    for v in spec.all_vouchers:
        game.vouchers.add(v)
        if v == "v_crystal_ball":
            game.consumable_slots += 1
    game.base_hands += spec.hands
    game.dollars += spec.dollars
    game.base_discards += spec.discards
    game.reroll_discount += spec.reroll_discount
    game.joker_slots += spec.joker_slot
    # starting_params.hand_size: the engine rebuilds hand_size from HAND_SIZE + hand_size_mod
    # at every blind start, so the deck's delta goes into the permanent modifier.
    game.hand_size_mod += spec.hand_size
    game.ante_scaling = spec.ante_scaling
    game.consumable_slots += spec.consumable_slot
    game.no_interest = spec.no_interest
    game.money_per_hand = spec.money_per_hand if spec.money_per_hand is not None else 1
    game.money_per_discard = spec.money_per_discard if spec.money_per_discard is not None else 0
    game.plasma = spec.plasma
    game.anaglyph = spec.anaglyph
    # hands_left / discards_left were initialised from the bases before this hook; _start_blind
    # re-reads the bases, but keep the pre-blind view consistent for observers.
    game.hands_left = game.base_hands
    game.discards_left = game.base_discards
    return spec


def on_round_eval(game: "BalatroGame", last_blind_was_boss: bool) -> bool:
    """``Back:trigger_effect{context='eval'}`` (state_events.lua:1163, back.lua:111-119):
    Anaglyph adds a Double Tag after a Boss Blind.  The add runs inside an event, i.e.
    AFTER the synchronous Investment-Tag ``eval`` loop — so call this after
    ``tag_state.on_round_eval``.  Returns True when a tag was added."""
    if game.anaglyph and last_blind_was_boss:
        game.tag_state.acquire("tag_double", game._tag_ctx())
        return True
    return False


# ── Status table (mirrored in DECKS_NOTES.md) ─────────────────────────────────────────

DECK_STATUS = {
    # key: (status, note)
    "b_red": ("done", "verified: +1 discard (4/round) — the engine had 3 before W3"),
    "b_blue": ("done", "+1 hand"),
    "b_yellow": ("done", "+$10"),
    "b_green": ("done", "$2/hand, $1/discard, no interest"),
    "b_black": ("done", "+1 joker slot, -1 hand"),
    "b_magic": ("done", "Crystal Ball (+1 consumable slot engine-side) + 2 Fools held"),
    "b_nebula": ("done", "Telescope (generation-side) + -1 consumable slot"),
    "b_ghost": ("done", "spectral_rate 2 (generation-side) + Hex held"),
    "b_abandoned": ("done", "40-card no-face deck (generation-side)"),
    "b_checkered": ("done", "26 S + 26 H, creation order = pre-swap sort (creation_order)"),
    "b_zodiac": ("done", "3 vouchers (generation-side: rates + shelf 3)"),
    "b_painted": ("done", "+2 hand size, -1 joker slot"),
    "b_anaglyph": ("done", "Double Tag after each boss (on_round_eval)"),
    "b_plasma": ("done", "balance at final_scoring_step + ante_scaling 2"),
    "b_erratic": ("done", "52 'erratic' draws (generation-side); creation order = sorted"),
}
