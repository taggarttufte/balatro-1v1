"""
shop.py — Shop shelf, pricing, buy/sell logic and booster-pack contents.

Phase 1 W2: the engine no longer generates anything itself.  Every shelf
card, voucher, booster slot and pack card comes from ``rng/generate.py``
(oracle-verified against the game's Lua executing in LuaJIT) driven through
``game.run_state`` (``generate.RunState``):

  * ``generate_shop(game)``  -> ``gen.generate_shop(run_state)`` (fresh shop,
    ``shop_joker_max`` slots, voucher from ``current_round.voucher``, two
    ``get_pack`` slots, Voucher-Tag vouchers);
  * ``reroll_shop(game)``    -> ``gen.reroll_shop(run_state, shop)`` (releases
    the old shelf, refills from the SAME per-key streams -- the "queue");
  * booster contents         -> ``gen.open_pack(run_state, center_key)``
    (see ``BalatroGame._open_booster``).

``used_jokers`` semantics are the game's (GENERATION_SPEC §6): marked on card
CREATION (generate does it), released on ``Card:remove`` -- the engine calls
``run_state.acquire`` on purchase/pick, ``run_state.release_shop`` on leaving,
``run_state.release_pack`` when a pack closes, ``run_state.remove_owned`` on
sell/use.  ``BalatroGame._sync_run_state`` additionally rebuilds the ownership
view from the game's own collections before every generation call, so the
incremental bookkeeping can never drift from what the game actually holds.

Prices come from ``rng/pools.py`` (per-card costs); edition markups are
the game's ``extra_cost`` (foil +2, holo +3, polychrome +5, negative +5);
discounts follow ``Card:set_cost`` (``floor((cost+0.5)*(100-d)/100)``, min
1); couponed cards (Uncommon/Rare/edition/Coupon tags) cost 0.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .game import BalatroGame

from . import game_keys
from .game_keys import gen as _gen
from .consumables import (
    ALL_PLANETS, ALL_TAROTS, ALL_SPECTRALS, ALL_VOUCHERS,
    PLANET_NAME, TAROT_NAME, SPECTRAL_NAME, VOUCHER_NAME,
)
from .constants import ENHANCEMENT_FROM_KEY

# ════════════════════════════════════════════════════════════════════════════
# JOKER CATALOGUE — all joker keys with rarity and base price (from pools)
# ════════════════════════════════════════════════════════════════════════════

JOKER_CATALOGUE: dict[str, dict] = {
    j["key"]: {
        "key": j["key"],
        "name": j["name"],
        "rarity": game_keys.RARITY_NAME[j["rarity"]],
        "rarity_id": j["rarity"],
        "price": j["cost"],
        "order": j["order"],
        "blueprint_compat": j["blueprint_compat"],
        "eternal_compat": j["eternal_compat"],
        "perishable_compat": j["perishable_compat"],
    }
    for j in game_keys.JOKERS
}

# Module-level joker ban list (the MP ruleset bans Chicot, Matador, Mr. Bones,
# Luchador).  Applied as ``run_state.banned_keys`` when a game is created —
# the game's own ban mechanism (``G.GAME.banned_keys``: the pool slot is
# written UNAVAILABLE in place and resampled, GENERATION_SPEC §3/§4).
BANNED_JOKERS: set[str] = set()


def set_banned_jokers(banned: set[str]):
    """Set the global banned joker list used by shop generation (new games only)."""
    global BANNED_JOKERS
    BANNED_JOKERS = set(banned)


def clear_banned_jokers():
    """Clear the banned joker list (revert to base game)."""
    global BANNED_JOKERS
    BANNED_JOKERS = set()


# ── edition / enhancement name maps (generate -> engine display names) ──────
EDITION_FROM_GEN = {None: "None", "foil": "Foil", "holo": "Holographic",
                    "polychrome": "Polychrome", "negative": "Negative"}
EDITION_TO_GEN = {v: k for k, v in EDITION_FROM_GEN.items()}
EDITION_MARKUP = {"Foil": 2, "Holographic": 3, "Polychrome": 5, "Negative": 5}   # Card:set_cost extra_cost
_RANK_FROM_LETTER = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
                     "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}
_SUIT_FROM_LETTER = {"S": "Spades", "H": "Hearts", "C": "Clubs", "D": "Diamonds"}
_RANK_NAME = {2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10: "10",
              11: "Jack", 12: "Queen", 13: "King", 14: "Ace"}
PLAYING_CARD_COST = 1


def front_to_rank_suit(front: str) -> tuple[int, str]:
    """'H_A' -> (14, 'Hearts')."""
    s, r = front.split("_")
    return _RANK_FROM_LETTER[r], _SUIT_FROM_LETTER[s]


def enhancement_from_gen(key: Optional[str]) -> str:
    """'m_bonus' -> 'Bonus'; None / 'c_base' -> 'None'."""
    if not key or not key.startswith("m_"):
        return "None"
    return ENHANCEMENT_FROM_KEY.get(key, "None")


def card_from_gen(c) -> "Card":
    """Engine Card for a Base/Enhanced CardGen (Standard pack, Magic Trick shelf card)."""
    from .card import Card
    rank, suit = front_to_rank_suit(c.front)
    card = Card(rank=rank, suit=suit)
    card.enhancement = enhancement_from_gen(c.key)
    card.edition = EDITION_FROM_GEN.get(c.edition, "None")
    card.seal = c.seal or "None"
    return card


# ════════════════════════════════════════════════════════════════════════════
# SHOP ITEM
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class ShopItem:
    kind: str          # "joker" | "planet" | "tarot" | "spectral" | "card" | "voucher" | "booster"
    key: str           # game key (j_*/c_*/v_*, booster TYPE key p_<kind>_<size>, or 'H_A' for a playing card)
    name: str
    price: int
    edition: str = "None"        # engine edition name (Foil / Holographic / Polychrome / Negative)
    sold: bool = False
    # ── playing cards (Magic Trick / Illusion) ──
    enhancement: str = "None"
    seal: str = "None"
    front: Optional[str] = None  # 'H_A'
    # ── jokers at stake >= 4 ──
    eternal: bool = False
    perishable: bool = False
    rental: bool = False
    # ── tags ──
    couponed: bool = False       # cost 0 (Uncommon/Rare/edition/Coupon tags)
    from_tag: Optional[str] = None
    # ── boosters ──
    center: Optional[str] = None  # the pool center key ('p_arcana_normal_1'); `key` is the type key
    set: str = ""                 # game set: Joker / Tarot / Planet / Spectral / Base / Voucher / Booster

    def discounted_price(self, discount_frac: float) -> int:
        """``Card:set_cost`` (card.lua:370-383): floor((cost + 0.5) * (100 - d) / 100), min $1;
        couponed cards cost 0 while in the shop."""
        if self.couponed or self.price <= 0:
            return 0
        d = int(round(discount_frac * 100))
        return max(1, math.floor((self.price + 0.5) * (100 - d) / 100))


def shop_item_from_gen(c) -> ShopItem:
    """Engine ShopItem for a shelf CardGen from ``generate.create_card_for_shop``."""
    ed = EDITION_FROM_GEN.get(c.edition, "None")
    if c.set == "Joker":
        info = JOKER_CATALOGUE.get(c.key, {})
        price = info.get("price", 6) + EDITION_MARKUP.get(ed, 0)
        return ShopItem("joker", c.key, info.get("name", c.key), price, ed,
                        eternal=c.eternal, perishable=c.perishable, rental=c.rental,
                        couponed=c.couponed, from_tag=c.from_tag, set="Joker")
    if c.set in ("Tarot", "Planet", "Spectral"):
        kind = c.set.lower()
        names = {"tarot": TAROT_NAME, "planet": PLANET_NAME, "spectral": SPECTRAL_NAME}[kind]
        price = game_keys.CONSUMABLE_COST.get(c.key, 3) + EDITION_MARKUP.get(ed, 0)
        return ShopItem(kind, c.key, names.get(c.key, c.key), price, ed,
                        couponed=c.couponed, from_tag=c.from_tag, set=c.set)
    if c.set in ("Default", "Enhanced"):
        rank, suit = front_to_rank_suit(c.front)
        enh = enhancement_from_gen(c.key)
        name = (enh + " " if enh != "None" else "") + f"{_RANK_NAME[rank]} of {suit}"
        price = PLAYING_CARD_COST + EDITION_MARKUP.get(ed, 0)
        return ShopItem("card", c.front, name, price, ed, enhancement=enh, seal=c.seal or "None",
                        front=c.front, couponed=c.couponed, from_tag=c.from_tag, set="Base")
    raise ValueError(f"unexpected shop card set {c.set!r} for {c.key!r}")


def voucher_item(key: str, from_tag: bool = False) -> ShopItem:
    return ShopItem("voucher", key, VOUCHER_NAME.get(key, key), game_keys.VOUCHER_COST.get(key, 10),
                    from_tag=("tag_voucher" if from_tag else None), set="Voucher")


def booster_item(center_key: str) -> ShopItem:
    tkey = game_keys.booster_type_key(center_key)
    b = game_keys.BOOSTER_TYPES[tkey]
    return ShopItem("booster", tkey, b["name"], b["cost"], center=center_key, set="Booster")


# ════════════════════════════════════════════════════════════════════════════
# BOOSTER CHOICE — one card revealed by an opened pack
# ════════════════════════════════════════════════════════════════════════════

class BoosterChoice:
    """One card of an open booster pack (``game.booster_choices`` entry).

    ``key`` is the game key (``j_*`` / ``c_*``) or the playing-card front (``'H_A'``);
    ``set`` the game set; ``edition`` / ``enhancement`` / ``seal`` use the engine's display
    names; ``card`` is the engine ``Card`` for Standard-pack playing cards (``None``
    otherwise); ``gen`` keeps the ``CardGen`` for ``run_state`` bookkeeping.
    """
    __slots__ = ("key", "set", "edition", "enhancement", "seal", "front",
                 "eternal", "perishable", "rental", "card", "gen")

    def __init__(self, key: str, set: str = "", edition: str = "None", enhancement: str = "None",
                 seal: str = "None", front: Optional[str] = None, eternal: bool = False,
                 perishable: bool = False, rental: bool = False, card=None, gen=None):
        self.key = key
        self.set = set
        self.edition = edition
        self.enhancement = enhancement
        self.seal = seal
        self.front = front
        self.eternal = eternal
        self.perishable = perishable
        self.rental = rental
        self.card = card
        self.gen = gen

    @classmethod
    def from_gen(cls, c) -> "BoosterChoice":
        ed = EDITION_FROM_GEN.get(c.edition, "None")
        if c.set in ("Default", "Enhanced"):
            card = card_from_gen(c)
            return cls(c.front, "Base", ed, card.enhancement, card.seal, c.front, card=card, gen=c)
        return cls(c.key, c.set, ed, eternal=c.eternal, perishable=c.perishable, rental=c.rental, gen=c)

    @property
    def is_joker(self) -> bool:
        return self.set == "Joker"

    @property
    def is_consumable(self) -> bool:
        return self.set in ("Tarot", "Planet", "Spectral")

    @property
    def is_playing_card(self) -> bool:
        return self.set == "Base"

    @property
    def name(self) -> str:
        if self.is_joker:
            return game_keys.JOKER_NAME.get(self.key, self.key)
        if self.is_consumable:
            return {"Tarot": TAROT_NAME, "Planet": PLANET_NAME, "Spectral": SPECTRAL_NAME}[self.set].get(self.key, self.key)
        return repr(self.card) if self.card is not None else self.key

    def clone(self) -> "BoosterChoice":
        return BoosterChoice(self.key, self.set, self.edition, self.enhancement, self.seal, self.front,
                             self.eternal, self.perishable, self.rental,
                             card=(self.card.copy() if self.card is not None else None), gen=self.gen)

    def __eq__(self, other):
        if isinstance(other, BoosterChoice):
            return (self.key, self.set, self.edition, self.enhancement, self.seal) == \
                   (other.key, other.set, other.edition, other.enhancement, other.seal)
        return NotImplemented

    def __hash__(self):
        return hash((self.key, self.set, self.edition, self.enhancement, self.seal))

    def __repr__(self):
        bits = [self.key]
        if self.edition != "None":
            bits.append(self.edition)
        if self.enhancement != "None":
            bits.append(self.enhancement)
        if self.seal != "None":
            bits.append(self.seal + "-seal")
        return "BoosterChoice(" + " ".join(bits) + ")"


# ════════════════════════════════════════════════════════════════════════════
# SHOP GENERATION (delegated)
# ════════════════════════════════════════════════════════════════════════════

# Booster pack TYPES (15 = 5 kinds x normal/jumbo/mega), keyed `p_<kind>_<size>`.
# Tuple shape kept for existing callers: (display name, price, content kind,
# number of cards shown).  Picks per pack in BOOSTER_PICKS (mega = 2).
_BOOSTER_CONTENT = {"Arcana": "tarot", "Celestial": "planet", "Spectral": "spectral",
                    "Standard": "card", "Buffoon": "joker"}
BOOSTER_CATALOGUE: dict[str, tuple] = {
    k: (b["name"], b["cost"], _BOOSTER_CONTENT[b["kind"]], b["cards"])
    for k, b in game_keys.BOOSTER_TYPES.items()
}
BOOSTER_PICKS: dict[str, int] = {k: b["choose"] for k, b in game_keys.BOOSTER_TYPES.items()}

SHELF_KINDS = ("joker", "planet", "tarot", "spectral", "card")


def _shelf_items_from(shop) -> list[ShopItem]:
    return [shop_item_from_gen(c) for c in shop.cards]


def _fire_joker_hook(game: "BalatroGame", hook: str) -> None:
    """Call ``hook(inst, ctx)`` on every owned joker (one shared context) and drain what it
    left pending -- base.fire_hook (W3); created-card tokens resolve via game._materialize."""
    from .jokers.base import fire_hook
    fire_hook(game, hook)


def generate_shop(game: "BalatroGame") -> list[ShopItem]:
    """A FRESH shop for the blind just finished (game.lua:3072-3181 ``update_shop``).

    Order, as the Lua: tags ``shop_start`` (D6) -> ``shop_joker_max`` shelf slots
    (``create_card_for_shop``, incl. the Uncommon/Rare/edition tag hooks that
    ``generate.generate_shop`` applies itself) -> voucher card from
    ``current_round.voucher`` (drawn at run start / boss defeat, NOT here) -> two
    ``get_pack('shop_pack')`` slots (the very first pack of a run is a Buffoon pack
    that consumes nothing) -> Voucher-Tag vouchers -> tags ``shop_final_pass`` (Coupon).
    """
    rs = game.run_state
    game._sync_run_state()
    ctx = game._tag_ctx()
    game.tag_state.on_shop_start(ctx)                      # D6 -> reroll base $0 (no RNG)
    rs.tags = game.tag_state.keys()                         # option B: generate owns the shop-time tags
    shop = _gen.generate_shop(rs)
    game._absorb_tag_triggers()
    items: list[ShopItem] = _shelf_items_from(shop)
    if shop.voucher:
        items.append(voucher_item(shop.voucher))
    for vk in shop.tag_vouchers:
        items.append(voucher_item(vk, from_tag=True))
    for pk in shop.boosters:
        if pk:
            items.append(booster_item(pk))
    game._shop_gen = shop
    game.current_shop = items
    game.tag_state.on_shop_final_pass(ctx)                 # Coupon -> make_shop_free (no RNG)
    _fire_joker_hook(game, "on_shop_enter")
    return items


def _random_consumable_item(game: "BalatroGame") -> ShopItem:   # pragma: no cover - legacy name
    raise NotImplementedError("shop generation is delegated to rng.generate (W2)")


# ════════════════════════════════════════════════════════════════════════════
# BUY / SELL LOGIC
# ════════════════════════════════════════════════════════════════════════════

def emplace_joker(game: "BalatroGame", j, sell_value: Optional[int] = None):
    """Put a joker the player just obtained (purchase / pack pick / created card / tag) on
    the board: slot bookkeeping (a Negative joker takes no slot), ``run_state.acquire``
    (``used_jokers``, Showman), ``on_init`` (To Do List's first hand) and the Oops! resync
    -- base.init_joker / sync_probabilities (W3)."""
    from .jokers.base import init_joker, sync_probabilities
    if sell_value is not None:
        j.state["sell_value"] = sell_value
    elif "sell_value" not in j.state:
        j.state["sell_value"] = max(1, game_keys.JOKER_COST.get(j.key, 2) // 2)
    game.jokers.append(j)
    if j.edition == "Negative":
        game.joker_slots += 1
    game.run_state.acquire(j.key)
    init_joker(j, game._hook_ctx())
    sync_probabilities(game)
    return j


def effective_price(game: "BalatroGame", item: ShopItem) -> int:
    """Price after discounts, coupons and shop-side joker passives (Astronomer: Planets and
    Celestial packs cost $0 -- base.passive_modifiers)."""
    from .jokers.base import passive_modifiers
    if item.kind == "planet" or (item.kind == "booster" and "celestial" in item.key):
        if passive_modifiers(game.jokers)["free_planets"]:
            return 0
    return item.discounted_price(game.shop_discount)


def can_afford(game: "BalatroGame", price: int) -> bool:
    """Credit Card lets the player go to -$20 (G.GAME.bankrupt_at)."""
    from .jokers.base import passive_modifiers
    return game.dollars - price >= passive_modifiers(game.jokers)["bankrupt_at"]


def buy_item(game: "BalatroGame", item: ShopItem) -> bool:
    """
    Attempt to purchase a shop item. Returns True on success.
    Modifies game.dollars, game.jokers, game.consumable_hand, game.full_deck as appropriate;
    a booster purchase fills ``game.booster_choices`` (``BalatroGame.step`` then enters
    ``State.BOOSTER_OPEN``).
    """
    if item.sold:
        return False

    price = effective_price(game, item)
    if not can_afford(game, price):
        return False
    rs = game.run_state

    if item.kind == "joker":
        negative = item.edition == "Negative"
        if len(game.jokers) >= game.joker_slots and not negative:
            return False
        from .jokers.base import JokerInstance
        j = JokerInstance(item.key, item.edition)
        base = item.price if item.couponed else price
        for flag in ("eternal", "perishable", "rental"):
            if getattr(item, flag):
                j.state[flag] = True
        emplace_joker(game, j, sell_value=max(1, base // 2))
        game.dollars -= price
        item.sold = True
        return True

    if item.kind in ("planet", "tarot", "spectral"):
        if len(game.consumable_hand) >= game.consumable_slots and item.edition != "Negative":
            return False
        if item.edition == "Negative":
            game.add_negative_consumable(item.key)     # slot bump tracked per key (W5)
        else:
            game.consumable_hand.append(item.key)
            rs.acquire(item.key)
        game.dollars -= price
        item.sold = True
        return True

    if item.kind == "card":
        from .card import Card
        rank, suit = front_to_rank_suit(item.front or item.key)
        card = Card(rank=rank, suit=suit)
        card.enhancement = item.enhancement
        card.edition = item.edition
        card.seal = item.seal
        game.add_card(card)
        game.dollars -= price
        item.sold = True
        return True

    if item.kind == "voucher":
        from .consumables import apply_voucher
        if apply_voucher(game, item.key):
            game.dollars -= price
            item.sold = True
            if not item.from_tag:
                rs.current_round_voucher = None     # slot stays empty for the rest of the ante
            if item.key in rs.shop_voucher_keys:
                rs.shop_voucher_keys.remove(item.key)
            return True
        return False

    if item.kind == "booster":
        game.dollars -= price
        item.sold = True
        game._open_booster(item.center or item.key, free=False)
        return True

    return False


def sell_joker(game: "BalatroGame", joker_idx: int) -> int:
    """Sell joker at index. Returns dollars gained (0 if invalid)."""
    if joker_idx < 0 or joker_idx >= len(game.jokers):
        return 0
    from .jokers.base import joker_sell_value, remove_joker, sell_hooks
    j = game.jokers[joker_idx]
    sell_value = joker_sell_value(j)
    game.dollars += sell_value
    if j.edition == "Negative":
        game.joker_slots = max(1, game.joker_slots - 1)
    # Card:remove -> used_jokers released unless another copy is owned (+ Oops! resync)
    remove_joker(game, j)
    # on_sell (Diet Cola -> Double Tag, Invisible Joker, Luchador) then selling_card on the
    # rest (Campfire) -- base.sell_hooks (W3); tags the sold joker queued go to tags.py
    sell_hooks(game, j)
    for tag_key in j.state.pop("pending_tags", []):
        game.tag_state.acquire(tag_key, game._tag_ctx())      # add_tag (Diet Cola: card.lua:2361-2369)
    return sell_value


def reroll_shop(game: "BalatroGame") -> bool:
    """Pay for a reroll and refill the shelf from the same streams
    (button_callbacks.lua:2855-2911).  Returns True on success."""
    cost = max(0, game.reroll_cost - game.reroll_discount)
    if game.free_rerolls_remaining > 0:       # Chaos the Clown / env free rerolls
        game.free_rerolls_remaining -= 1
        cost = 0
    if not can_afford(game, cost):
        return False
    game.dollars -= cost
    game.reroll_cost += 1

    rs = game.run_state
    game._sync_run_state()
    shop = game._shop_gen
    if shop is None:                         # shelf not produced by generate (tests that hand-build a shop)
        shop = _gen.ShopContents()
    rs.tags = game.tag_state.keys()
    _gen.reroll_shop(rs, shop)
    game._absorb_tag_triggers()
    game._shop_gen = shop
    keep = [it for it in game.current_shop if it.kind not in SHELF_KINDS]
    game.current_shop = _shelf_items_from(shop) + keep
    _fire_joker_hook(game, "on_reroll")
    return True


# ════════════════════════════════════════════════════════════════════════════
# BOOSTER PACK OPENING (contents; the state machine lives in BalatroGame)
# ════════════════════════════════════════════════════════════════════════════

def booster_contents(game: "BalatroGame", center_key: str) -> list[BoosterChoice]:
    """``Card:open`` (card.lua:1723-1784) via ``generate.open_pack``: the cards are created in
    order and each marks ``used_jokers`` as it is created (dedupe = resample, and the shelf on
    display behind the pack is excluded)."""
    if center_key in game_keys.BOOSTER_TYPES:                  # type key -> first art variant
        center_key = game_keys.BOOSTER_TYPES[center_key]["centers"][0]
    game._sync_run_state()
    cards = _gen.open_pack(game.run_state, center_key)
    return [BoosterChoice.from_gen(c) for c in cards]


def _open_booster(game: "BalatroGame", booster_key: str):       # legacy entry point
    game._open_booster(booster_key)


def _get_effect(key: str):
    from .jokers.base import JOKER_REGISTRY
    return JOKER_REGISTRY.get(key)
