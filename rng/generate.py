"""Balatro 1.0.1o generation layer -- shops, packs, vouchers, bosses, tags, consumable
and joker-created cards -- ported from the game's Lua with exact key construction and
the in-place ``UNAVAILABLE``/resample semantics.

Contract: ``GENERATION_SPEC.md`` (same directory).  Every function here cites the Lua it
ports; the spec carries the reasoning, ambiguities and the file:line map.

Dependencies (owned by other agents, imported defensively so this module always imports):

* ``core.PseudoRandom``  (Agent A) -- ``pseudoseed`` / ``pseudorandom`` /
  ``pseudorandom_element`` / ``pseudoshuffle`` with per-key state + LuaJIT ``math.random``.
* ``pools``              (Agent B) -- ordered pools and prototype fields.

Nothing here touches game state beyond what the Lua generation code itself mutates:
``used_jokers`` (marked on card CREATION -- see ``RunState.mark_used``), ``bosses_used``,
``first_shop_buffoon``, ``current_round.used_packs``.  Purchase / removal bookkeeping is
exposed as explicit ``RunState`` methods for the engine to call.

Quick demo::

    python -m rng.generate EXAMPLE1
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

# ----------------------------------------------------------------------------------------
# Guarded imports of sibling modules
# ----------------------------------------------------------------------------------------

try:  # Agent A
    from .core import PseudoRandom
except Exception as _e:  # pragma: no cover - only when core.py is absent/broken
    PseudoRandom = None  # type: ignore[assignment]
    _CORE_IMPORT_ERROR: Optional[BaseException] = _e
else:
    _CORE_IMPORT_ERROR = None

try:  # Agent B
    from . import pools as P
except Exception as _e:  # pragma: no cover - only when pools.py is absent/broken
    P = None  # type: ignore[assignment]
    _POOLS_IMPORT_ERROR: Optional[BaseException] = _e
else:
    _POOLS_IMPORT_ERROR = None

UNAVAILABLE = "UNAVAILABLE"


def _require_deps() -> None:
    if PseudoRandom is None:
        raise ImportError("rng.core (Agent A) is not importable: %r" % (_CORE_IMPORT_ERROR,))
    if P is None:
        raise ImportError("rng.pools (Agent B) is not importable: %r" % (_POOLS_IMPORT_ERROR,))


# ----------------------------------------------------------------------------------------
# Key construction.  Every pseudoseed key string used by the generation layer is built
# here so the spec and the code cannot drift.  ``ante`` is formatted like Lua's
# ``..G.GAME.round_resets.ante`` (an integer -> no decimal point).
# ----------------------------------------------------------------------------------------

def _ante_str(ante: int) -> str:
    # Lua: number .. string -> "%.14g"; the ante is always integral.
    if isinstance(ante, float) and ante.is_integer():
        ante = int(ante)
    return str(ante)


def _lua_num_str(x: float) -> str:
    """Lua ``number .. string`` for a non-integral double: LuaJIT formats with ``%.14g``."""
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return "%.14g" % x


# The two Multiplayer-mod switches (NOTES_ORDER.md).  Both live on RunState.
KEY_SCOPE_ANTE = "ante"   # vanilla AND Major League Balatro: ante-suffixed keys
KEY_SCOPE_RUN = "run"     # "The Order" lobby option: one run-global queue per key
RULESET_VANILLA = "vanilla"
RULESET_MLB = "mlb"


def _scope(state) -> str:
    return getattr(state, "key_scope", KEY_SCOPE_ANTE)


def _is_order(state) -> bool:
    return _scope(state) == KEY_SCOPE_RUN


def _culled_vouchers(state) -> bool:
    """``MP.should_use_the_order() or MP.is_major_league_ruleset()`` -- the guard on the
    mod's ``SMODS.get_next_vouchers`` / ``get_next_voucher_key`` overrides."""
    return _is_order(state) or getattr(state, "ruleset", RULESET_VANILLA) == RULESET_MLB


class Keys:
    """Key builders (see GENERATION_SPEC.md section 2 for the table with citations).

    Every builder that takes an ``ante`` int is the VANILLA construction.  The Multiplayer
    mod's "The Order" (``RunState.key_scope == "run"``) changes a fixed set of sites; those
    are the ``state``-aware builders below (``gen_ante`` / ``ante_suffix`` / ``boss`` /
    ``new_round_shuffle(ante, state)`` / ``cashout_shuffle(ante, state)`` / ``halu_for`` /
    the ``order_*`` helpers).  NOTES_ORDER.md has the full vanilla -> Order -> MLB table.
    """

    # -- The Order: the ante the mod substitutes at the patched sites ----------------------
    @staticmethod
    def gen_ante(state) -> int:
        """``MP.ante_based()`` (= 0 under The Order, else the real ante) -- also what
        ``G.GAME.round_resets.ante`` reads as inside the mod's ``create_card`` wrapper."""
        return 0 if _is_order(state) else state.ante

    @staticmethod
    def ante_suffix(state) -> str:
        return _ante_str(Keys.gen_ante(state))

    @staticmethod
    def boss(state=None) -> str:
        """``'boss'`` (vanilla/MLB) or ``'boss'..ante`` under The Order (TheOrder.toml patch
        of get_new_boss).  The ante is the REAL ante (this site is outside create_card)."""
        if state is not None and _is_order(state):
            return "boss" + _ante_str(state.ante)
        return "boss"

    @staticmethod
    def order_round(state, ante: int) -> str:
        """``MP.order_round_based(true)`` under The Order:
        ``ante .. G.GAME.blind.config.blind.key .. G.GAME.blind_on_deck`` (each ``or ''``)."""
        return _ante_str(ante) + (getattr(state, "blind_key", None) or "") + (getattr(state, "blind_type", None) or "")

    @staticmethod
    def order_key_append(_type: str, area: str, key_append: Optional[str]) -> Optional[str]:
        """The mod's create_card wrapper rewrites ``key_append`` under The Order:
        Tarot/Planet/Spectral -> ``_type`` (``_type..'_pack'`` in G.pack_cards); Base/Enhanced
        keep theirs; everything else (Joker) -> nil."""
        if _type in ("Tarot", "Planet", "Spectral"):
            return _type + "_pack" if area == "pack" else _type
        if _type in ("Base", "Enhanced"):
            return key_append
        return None

    @staticmethod
    def order_sticker_poll(pool_key: str) -> str:     # pseudorandom("_etper".._pool_key)
        return "_etper" + pool_key

    @staticmethod
    def order_rental_poll(pool_key: str) -> str:      # pseudorandom("_rent".._pool_key)
        return "_rent" + pool_key

    @staticmethod
    def order_joker_draw(pool_key: str, sticker: bool) -> str:   # pseudoseed(_pool_key.._s_append)
        return pool_key + ("_sticker" if sticker else "")

    @staticmethod
    def order_joker_edition(pool_key: str, sticker: bool) -> str:  # poll_edition('edi'.._pool_key.._s_append)
        return "edi" + pool_key + ("_sticker" if sticker else "")

    ORDER_JUD_RARITY = "order_jud_rarity"   # Judgement with eternals enabled (mod wrapper)
    VOUCHER_ORDER = "Voucher0"              # MLB / The Order voucher stream (shop AND Voucher Tag)

    @staticmethod
    def voucher_order_fallback(it: int) -> str:       # "Voucher0"..it after 1000 redraws
        return "Voucher0" + str(it)

    # -- shop (UI_definitions.lua:742-800, common_events.lua:1963-2154)
    @staticmethod
    def cdt(ante: int) -> str:                      # shop slot type roll
        return "cdt" + _ante_str(ante)

    @staticmethod
    def rarity(ante: int, append: str) -> str:      # joker rarity roll
        return "rarity" + _ante_str(ante) + (append or "")

    @staticmethod
    def joker_pool(rarity: int, append: str, ante: int, legendary: bool = False) -> str:
        # 'Joker'..rarity..((not _legendary and _append) or '') .. (not _legendary and ante or '')
        if legendary:
            return "Joker4"
        return "Joker" + str(rarity) + (append or "") + _ante_str(ante)

    @staticmethod
    def center_pool(_type: str, append: str, ante: int) -> str:   # Tarot/Planet/Spectral/Voucher/Enhanced/Tag
        return _type + (append or "") + _ante_str(ante)

    @staticmethod
    def resample(pool_key: str, it: int) -> str:
        return pool_key + "_resample" + str(it)

    @staticmethod
    def edition(append: str, ante: int) -> str:     # create_card joker edition
        return "edi" + (append or "") + _ante_str(ante)

    @staticmethod
    def front(append: str, ante: int) -> str:       # playing-card front for Base/Enhanced
        return "front" + (append or "") + _ante_str(ante)

    @staticmethod
    def soul(_type: str, ante: int) -> str:
        return "soul_" + _type + _ante_str(ante)

    @staticmethod
    def sticker_poll(area: str, ante: int) -> str:  # eternal/perishable
        return ("packetper" if area == "pack" else "etperpoll") + _ante_str(ante)

    @staticmethod
    def rental_poll(area: str, ante: int) -> str:
        return ("packssjr" if area == "pack" else "ssjr") + _ante_str(ante)

    @staticmethod
    def pack(key: Optional[str], ante: int) -> str:  # get_pack
        return (key or "pack_generic") + _ante_str(ante)

    # -- run structure
    BOSS = "boss"
    VOUCHER_FROM_TAG = "Voucher_fromtag"
    SHUFFLE = "shuffle"
    ERRATIC = "erratic"
    ORBITAL = "orbital"

    @staticmethod
    def new_round_shuffle(ante: int, state=None) -> str:
        """``'nr'..ante`` (vanilla/MLB); under The Order ``'nr'..MP.order_round_based(true)``
        = ``'nr'..ante..blind_key..blind_type`` (state_events.lua:344 patch) -- pass ``state``
        with ``blind_key``/``blind_type`` set.  ``shuffle_deck`` upgrades the vanilla form
        automatically, so engines that build the key without ``state`` still get it right."""
        if state is not None and _is_order(state):
            return "nr" + Keys.order_round(state, ante)
        return "nr" + _ante_str(ante)

    @staticmethod
    def cashout_shuffle(ante: int, state=None) -> str:
        """``'cashout'..ante``; under The Order ``'cashout'..ante..blind_key..blind_type`` with
        the DEFEATED blind's key/type (button_callbacks.lua:2918 patch; ``blind_on_deck`` only
        moves to the next blind at the blind-select screen / reset_blinds, after this shuffle)."""
        if state is not None and _is_order(state):
            return "cashout" + Keys.order_round(state, ante)
        return "cashout" + _ante_str(ante)

    @staticmethod
    def idol(ante: int) -> str:
        return "idol" + _ante_str(ante)

    @staticmethod
    def mail(ante: int) -> str:
        return "mail" + _ante_str(ante)

    @staticmethod
    def ancient(ante: int) -> str:
        return "anc" + _ante_str(ante)

    @staticmethod
    def castle(ante: int) -> str:
        return "cas" + _ante_str(ante)

    # -- standard pack modifiers (card.lua:1759-1773)
    @staticmethod
    def stdset(ante: int) -> str:
        return "stdset" + _ante_str(ante)

    @staticmethod
    def standard_edition(ante: int) -> str:
        return "standard_edition" + _ante_str(ante)

    @staticmethod
    def stdseal(ante: int) -> str:
        return "stdseal" + _ante_str(ante)

    @staticmethod
    def stdsealtype(ante: int) -> str:
        return "stdsealtype" + _ante_str(ante)

    # -- misc single-stream keys
    OMEN_GLOBE = "omen_globe"
    ILLUSION = "illusion"

    @staticmethod
    def halu(ante: int) -> str:
        return "halu" + _ante_str(ante)

    @staticmethod
    def halu_for(state) -> str:
        """``'halu'..MP.ante_based()`` (card.lua patch): ``halu0`` under The Order."""
        return "halu" + Keys.ante_suffix(state)


# ----------------------------------------------------------------------------------------
# pairs(G.GAME.hands) iteration order
# ----------------------------------------------------------------------------------------
#
# To Do List (card.lua:311-323, 2975-2982) and the Orbital Tag (UI_definitions.lua:1509-1516)
# build their candidate list with ``for k, v in pairs(G.GAME.hands) do if v.visible then ... end``
# and then ``pseudorandom_element`` over that array, so the *hash-table iteration order* of the
# 12-key ``hands`` constructor (game.lua:2001-2014) is part of the RNG contract.
#
# ORACLE-DERIVED, NOT DERIVABLE IN PYTHON: this is LuaJIT string-hash order.  The game ships
# LuaJIT 2.0.5 (``lua51.dll`` next to Balatro.exe; ``jit.version`` reports "LuaJIT 2.0.5") whose
# string hash is fixed, so the order below is stable across processes (verified 5/5 runs by
# executing the verbatim ``hands = {...}`` constructor inside that DLL via ctypes -- see
# tests/test_generate_oracle.py::test_hands_pairs_order_matches_game_dll).  lupa's LuaJIT 2.1
# randomises its string-hash seed per VM (LUAJIT_SECURITY_STRHASH) and yields a different,
# rotating order, so it cannot be used to check this constant.
#
# Caveat: a *loaded* save rebuilds ``G.GAME.hands`` through STR_UNPACK with a different insertion
# order, which may change the layout; this constant is for a run that has not been reloaded.
HANDS_PAIRS_ORDER = (
    "Flush House", "Full House", "Flush", "Pair", "High Card", "Straight Flush",
    "Straight", "Two Pair", "Flush Five", "Five of a Kind", "Three of a Kind", "Four of a Kind",
)


def visible_hands_in_pairs_order(visible, state=None) -> list:
    """Candidate array for ``'to_do'`` / ``'orbital'``: ``pairs(G.GAME.hands)`` filtered to
    visible hands, in iteration order.  ``visible`` is the set of currently visible hand names
    (at run start: everything except Flush Five / Flush House / Five of a Kind).

    Under The Order the Multiplayer mod replaces the ``pairs`` walk with
    ``MP.sorted_hand_list`` -- the visible hands in ``G.GAME.hands[k].order`` order (=
    ``G.handlist`` / ``pools.HANDLIST``), which is what makes To Do List / Orbital identical
    on both clients (TheOrder.toml card.lua patches, ui/game/blind_choice.lua:37-43)."""
    if state is not None and _is_order(state):
        return [h for h in P.HANDLIST if h in visible]
    return [h for h in HANDS_PAIRS_ORDER if h in visible]


def to_do_hand(state: "RunState", visible, previous: Optional[str] = None, title_area: bool = False) -> str:
    """To Do List hand pick (card.lua:318-321): ``pseudorandom_element(_poker_hands,
    pseudoseed('to_do'))`` repeated until it differs from the previous hand (each retry is a
    fresh draw on the same key).  Under The Order the candidate list is ``MP.sorted_hand_list
    (nil)`` (the patch runs after ``to_do_poker_hand = nil``): order-sorted, same retry loop.
    (The end-of-round re-pick, card.lua:2975-2982, is a single draw over the visible hands
    MINUS the current one -- not modelled here; the engine does not implement it either.)"""
    key = "false_to_do" if title_area else "to_do"
    cands = visible_hands_in_pairs_order(visible, state)
    while True:
        hand, _ = state.rng.pseudorandom_element(cands, key)
        if hand != previous:
            return hand


def orbital_hand(state: "RunState", visible) -> str:
    """Orbital Tag hand for one blind type (UI_definitions.lua:1515), drawn once per ante per
    blind type when the blind-select UI is built."""
    hand, _ = state.rng.pseudorandom_element(visible_hands_in_pairs_order(visible, state), Keys.ORBITAL)
    return hand


# ----------------------------------------------------------------------------------------
# Result records
# ----------------------------------------------------------------------------------------

@dataclass
class CardGen:
    """One generated card (the subset of a Lua ``Card`` that generation decides)."""
    key: str                         # center key: 'j_joker', 'c_fool', 'c_base', 'm_bonus', 'v_blank'
    set: str                         # center set: 'Joker','Tarot','Planet','Spectral','Default','Enhanced','Voucher'
    type_requested: str              # the _type argument create_card was called with
    key_append: str = ""
    area: str = "shop"               # 'shop' | 'pack' | 'jokers' | 'consumables' | 'deck'
    front: Optional[str] = None      # 'H_A' etc. for Base/Enhanced playing cards
    edition: Optional[str] = None    # 'negative' | 'polychrome' | 'holo' | 'foil'
    seal: Optional[str] = None       # 'Red' | 'Blue' | 'Gold' | 'Purple'
    eternal: bool = False
    perishable: bool = False
    rental: bool = False
    rarity: Optional[int] = None     # 1..4 for jokers
    pool_key: Optional[str] = None   # e.g. 'Joker1sho1' (None when the key was forced)
    pool_index: Optional[int] = None # 0-based index into the culled pool that was finally drawn
    resamples: int = 0               # number of '_resample<it>' redraws that were needed
    forced: bool = False             # forced_key path (Base, Soul/Black Hole hit, Fool, deck consumables)
    couponed: bool = False           # set by Uncommon/Rare/edition tags (free)
    from_tag: Optional[str] = None   # tag key that created/modified this card

    @property
    def is_joker(self) -> bool:
        return self.set == "Joker"

    @property
    def is_consumable(self) -> bool:
        return self.set in ("Tarot", "Planet", "Spectral")

    @property
    def is_playing_card(self) -> bool:
        return self.set in ("Default", "Enhanced")

    def short(self) -> str:
        bits = [self.key]
        if self.front:
            bits.append(self.front)
        if self.edition:
            bits.append(self.edition)
        if self.seal:
            bits.append(self.seal + "-seal")
        for flag in ("eternal", "perishable", "rental", "couponed"):
            if getattr(self, flag):
                bits.append(flag)
        if self.resamples:
            bits.append("resampled x%d" % self.resamples)
        return " ".join(bits)


@dataclass
class ShopContents:
    cards: list = field(default_factory=list)            # G.shop_jokers.cards (CardGen), slot order
    voucher: Optional[str] = None                        # G.GAME.current_round.voucher as displayed
    tag_vouchers: list = field(default_factory=list)     # extra vouchers added by Voucher Tags
    boosters: list = field(default_factory=list)         # 2 entries: pack key, or None if 'USED'
    rerolls: int = 0

    def describe(self) -> str:
        lines = ["  slot %d: %s" % (i + 1, c.short()) for i, c in enumerate(self.cards)]
        lines.append("  voucher: %s%s" % (self.voucher, (" + " + ", ".join(self.tag_vouchers)) if self.tag_vouchers else ""))
        lines.append("  boosters: %s" % ", ".join(str(b) for b in self.boosters))
        return "\n".join(lines)


@dataclass
class RunStart:
    seed: str
    boss: str
    voucher: str
    tag_small: str
    tag_big: str
    deck: list                       # playing-card keys in G.deck.cards order AFTER 'shuffle' (top of deck = last)
    idol: Optional[str] = None       # 'idol'..ante pick (playing-card key, 'S_A')
    mail: Optional[str] = None       # 'mail'..ante pick (playing-card key; only the rank is meaningful)
    ancient_suit: Optional[str] = None
    castle: Optional[str] = None


# ----------------------------------------------------------------------------------------
# Run state
# ----------------------------------------------------------------------------------------

def _all_boss_keys() -> list:
    return [b["key"] for b in P.BOSS_BLINDS]


@dataclass
class RunState:
    """The slice of ``G.GAME`` (+ a few ``G.*`` globals) that generation reads or writes.

    Defaults are a fresh White-stake Red-deck run on a FULLY UNLOCKED, fully discovered
    profile (``locked_keys``/``undiscovered_keys`` empty) -- the MLB ranked assumption and
    what public seed analyzers model.  Use ``RunState.fresh_profile`` for a new save file.

    Multiplayer-mod switches (Phase 2 W2, NOTES_ORDER.md) -- the engine sets these at run
    start, BEFORE any draw:

    * ``key_scope``: ``"ante"`` (default; vanilla AND Major League Balatro) or ``"run"``
      (the lobby option "The Order").  ``"run"`` does three things: (1) the RNG seed becomes
      ``'*'..seed`` (the mod prefixes the seed before ``hashed_seed`` is computed, so EVERY
      stream changes, not just the display); (2) the mod's patched key sites use ante ``0``
      / run-global keys (``Keys.gen_ante``); (3) jokers, vouchers, deck shuffles and
      card/joker picks use the mod's own selection algorithms (see ``create_card``,
      ``next_voucher``, ``shuffle_deck``, ``order_pick``).  Assigning ``key_scope`` after
      construction re-seeds the RNG (allowed only while no key has been drawn).
    * ``ruleset``: ``"vanilla"`` (default) or ``"mlb"``.  ``"mlb"`` alone (The Order off, as
      the MLB ruleset forces) changes ONLY the voucher draws: shop voucher and Voucher Tag
      both come from the culled pool on the run-global ``'Voucher0'`` stream
      (``next_voucher``).  Bans are the engine's job (``banned_keys``).
    * ``blind_key`` / ``blind_type``: ``G.GAME.blind.config.blind.key`` (``'bl_small'``,
      ``'bl_big'``, ``'bl_hook'``, ``'bl_mp_nemesis'``...) and ``G.GAME.blind_on_deck``
      (``'Small'``/``'Big'``/``'Boss'``) of the blind being played / just defeated.  Only
      read under The Order (the ``nr``/``cashout`` shuffle keys); ignored otherwise.
    """
    seed: str
    ante: int = 1                                  # G.GAME.round_resets.ante
    round: int = 0                                 # G.GAME.round (not used by any generation key)
    rng: Any = None                                # PseudoRandom(seed); created in __post_init__
    key_scope: str = KEY_SCOPE_ANTE                # "ante" | "run"  (The Order)      -- see class doc
    ruleset: str = RULESET_VANILLA                 # "vanilla" | "mlb"                -- see class doc
    blind_key: Optional[str] = None                # G.GAME.blind.config.blind.key   (The Order shuffles)
    blind_type: Optional[str] = None               # G.GAME.blind_on_deck            (The Order shuffles)

    # eligibility state --------------------------------------------------------------------
    used_jokers: set = field(default_factory=set)          # G.GAME.used_jokers (marked on card CREATION)
    used_vouchers: set = field(default_factory=set)        # G.GAME.used_vouchers (redeemed / deck-granted)
    bosses_used: dict = field(default_factory=dict)        # G.GAME.bosses_used: key -> count (all bosses, 0)
    banned_keys: set = field(default_factory=set)          # G.GAME.banned_keys
    # TODO(Phase 2 / MLB): populate from the Multiplayer mod's ban list (bosses such as the
    # ones Attrition/Nemesis disallow, plus any banned jokers/tags). Out of Agent C's scope;
    # every generation path already honours this set (pool cull, get_pack weights, boss
    # eligibility, forced keys, soul gate) so filling it is the only change needed.
    locked_keys: set = field(default_factory=set)          # prototypes with unlocked == false
    undiscovered_keys: set = field(default_factory=set)    # for Tag.requires (discovered check)
    pool_flags: set = field(default_factory=set)           # G.GAME.pool_flags ('gros_michel_extinct')
    hands_played: dict = field(default_factory=dict)       # hand name -> G.GAME.hands[h].played
    deck_enhancements: set = field(default_factory=set)    # enhancement keys present on any playing card
    perscribed_bosses: dict = field(default_factory=dict)  # ante -> boss key (challenges)
    force_boss: Optional[str] = None                       # G.FORCE_BOSS
    force_tag: Optional[str] = None                        # G.FORCE_TAG
    win_ante: int = 8                                      # G.GAME.win_ante

    # ownership (mirrors G.jokers.cards / G.consumeables.cards / G.shop_vouchers.cards) -----
    owned_jokers: list = field(default_factory=list)       # joker keys in G.jokers.cards order
    owned_consumables: list = field(default_factory=list)  # consumable keys in G.consumeables.cards order
    showman: bool = False                                  # next(find_joker('Showman')) -- owned AND not debuffed
    shop_voucher_keys: list = field(default_factory=list)  # vouchers currently displayed in G.shop_vouchers

    # rates / multipliers ------------------------------------------------------------------
    probabilities_normal: float = 1.0     # G.GAME.probabilities.normal (x2 per Oops! All 6s)
    joker_rate: float = 20
    tarot_rate: float = 4
    planet_rate: float = 4
    playing_card_rate: float = 0
    spectral_rate: float = 0
    edition_rate: float = 1               # Hone -> 2, Glow Up -> 4
    shop_joker_max: int = 2               # G.GAME.shop.joker_max (+1 per Overstock tier)

    # stake / challenge modifiers ----------------------------------------------------------
    stake: int = 1
    enable_eternals_in_shop: bool = False     # stake >= 4
    enable_perishables_in_shop: bool = False  # stake >= 7
    enable_rentals_in_shop: bool = False      # stake >= 8
    all_eternal: bool = False                 # challenge modifier

    # per-round / per-shop bookkeeping -----------------------------------------------------
    first_shop_buffoon: bool = False                       # G.GAME.first_shop_buffoon
    current_round_voucher: Optional[str] = None            # G.GAME.current_round.voucher
    used_packs: list = field(default_factory=lambda: [None, None])  # G.GAME.current_round.used_packs[1..2]
    last_tarot_planet: Optional[str] = None                # for The Fool
    tags: list = field(default_factory=list)               # G.GAME.tags: list of tag keys, oldest first
    triggered_tags: set = field(default_factory=set)       # ids (indices) of tags already triggered this shop
    blind_tags: dict = field(default_factory=dict)         # {'Small': key, 'Big': key}
    boss_blind: Optional[str] = None                       # G.GAME.round_resets.blind_choices.Boss

    # -- construction --------------------------------------------------------------------

    def __post_init__(self) -> None:
        _require_deps()
        if self.rng is None:
            self.rng = PseudoRandom(self.effective_seed)
        if not self.bosses_used:
            self.bosses_used = {k: 0 for k in _all_boss_keys()}
        if not self.hands_played:
            self.hands_played = {h: 0 for h in P.HANDLIST}

    @property
    def effective_seed(self) -> str:
        """The string ``pseudoseed`` hashes with: the seed, or ``'*'..seed`` under The Order
        (TheOrder.toml game.lua patch, applied before ``hashed_seed = pseudohash(seed)``)."""
        if self.key_scope == KEY_SCOPE_RUN and not self.seed.startswith("*"):
            return "*" + self.seed
        return self.seed

    def __setattr__(self, name, value) -> None:
        object.__setattr__(self, name, value)
        if name == "key_scope" and self.__dict__.get("rng") is not None:
            self._reseed_for_scope()

    def _reseed_for_scope(self) -> None:
        want = self.effective_seed
        if self.rng.seed == want:
            return
        if list(self.rng.keys()):
            raise RuntimeError("RunState.key_scope changed after keys were drawn (%r -> %r); "
                               "set it at run start" % (self.rng.seed, want))
        object.__setattr__(self, "rng", PseudoRandom(want))

    @classmethod
    def for_stake(cls, seed: str, stake: int = 1, **kw) -> "RunState":
        """Apply the stake modifiers Game:start_run sets (game.lua:2047-2054)."""
        st = cls(seed, stake=stake, **kw)
        st.enable_eternals_in_shop = stake >= 4
        st.enable_perishables_in_shop = stake >= 7
        st.enable_rentals_in_shop = stake >= 8
        return st

    @classmethod
    def fresh_profile(cls, seed: str, **kw) -> "RunState":
        """A brand-new save file: default locks (P_LOCKED) and nothing discovered."""
        st = cls(seed, **kw)
        st.locked_keys = set(P.P_LOCKED_DEFAULT)
        st.undiscovered_keys = {c["key"] for c in P.JOKERS if not c.get("discovered")}
        st.undiscovered_keys |= {e["key"] for e in P.EDITIONS if not e.get("discovered")}
        return st

    def clone(self) -> "RunState":
        """Deep copy for tree search.  ``rng`` is cloned via PseudoRandom.clone()."""
        rng = self.rng
        self.rng = None
        try:
            c = copy.deepcopy(self)
        finally:
            self.rng = rng
        c.rng = rng.clone()
        return c

    # -- helpers that mirror Lua predicates ------------------------------------------------

    @property
    def ante_key(self) -> str:
        return _ante_str(self.ante)

    def is_unlocked(self, key: str) -> bool:
        """Lua: ``v.unlocked ~= false``.  Consumables have no unlocked field (always true)."""
        return key not in self.locked_keys

    def is_discovered(self, key: str) -> bool:
        return key not in self.undiscovered_keys

    def used_blocks(self, key: str) -> bool:
        """``G.GAME.used_jokers[key] and not next(find_joker('Showman'))``."""
        return key in self.used_jokers and not self.showman

    # -- the used_jokers lifecycle (card.lua:349-354 / 4741-4747) --------------------------

    def mark_used(self, key: str) -> None:
        """What ``Card:set_ability`` does for every new Card: ``used_jokers[key] = true``.
        ``create_card`` calls this itself; call it for cards created elsewhere (copy_card,
        deck-granted consumables, challenge starting jokers)."""
        self.used_jokers.add(key)

    def release(self, key: str) -> None:
        """What ``Card:remove`` does: clear ``used_jokers[key]`` unless a copy is still in
        G.jokers / G.consumeables (``find_joker(name, true)``)."""
        if key in self.owned_jokers or key in self.owned_consumables:
            return
        self.used_jokers.discard(key)

    def acquire(self, card: "CardGen | str") -> None:
        """Purchase / take-from-pack: the card now lives in an owned area, so a later
        ``release`` keeps it marked."""
        key = card.key if isinstance(card, CardGen) else card
        cset = card.set if isinstance(card, CardGen) else (P.JOKER_BY_KEY.get(key, {}).get("rarity") and "Joker")
        if cset == "Joker" or key.startswith("j_"):
            self.owned_jokers.append(key)
            if key == "j_ring_master":
                self.showman = True
        elif key.startswith("c_"):
            self.owned_consumables.append(key)
        self.used_jokers.add(key)

    def remove_owned(self, key: str) -> None:
        """Sell / use / destroy an owned card."""
        for lst in (self.owned_jokers, self.owned_consumables):
            if key in lst:
                lst.remove(key)
                break
        if key == "j_ring_master" and "j_ring_master" not in self.owned_jokers:
            self.showman = False
        self.release(key)

    def release_shop(self, shop: ShopContents) -> None:
        """Leaving the shop / rerolling: every unbought shop card is ``Card:remove``d."""
        for c in shop.cards:
            self.release(c.key)
        self.shop_voucher_keys = []

    def release_pack(self, cards: Iterable[CardGen]) -> None:
        for c in cards:
            self.release(c.key)

    # -- round structure -----------------------------------------------------------------

    def new_round(self) -> None:
        """state_events.lua:290-353 ``new_round``: ``used_packs = {}`` (so every round's shop
        rolls two fresh packs) and the deck is reshuffled with ``'nr'..ante`` (the caller
        shuffles; see ``shuffle_deck``)."""
        self.used_packs = [None, None]


# ----------------------------------------------------------------------------------------
# get_current_pool  (common_events.lua:1963-2053)
# ----------------------------------------------------------------------------------------

def _tarot_planet_pool() -> list:
    # P_CENTER_POOLS['Tarot_Planet'] is Tarots+Planets table.sort'ed by `order` -- but the two
    # sets share order values 1..12, so the Lua order is unspecified (non-stable sort).  Only
    # The Fool uses this pool and it always passes a forced key, so the order never matters.
    return sorted(P.TAROTS + P.PLANETS, key=lambda c: c["order"])


_CENTER_INDEX: Optional[dict] = None


def center(key: str) -> dict:
    """Prototype dict for a center key (jokers, consumables, vouchers, boosters, enhancements)."""
    global _CENTER_INDEX
    if _CENTER_INDEX is None:
        idx: dict = {}
        for lst in (P.JOKERS, P.TAROTS, P.PLANETS, P.SPECTRALS, P.VOUCHERS, P.BOOSTERS, P.ENHANCEMENTS, P.EDITIONS):
            for c in lst:
                idx[c["key"]] = c
        idx.setdefault("c_base", {"key": "c_base", "name": "Default Base", "set": "Default"})
        _CENTER_INDEX = idx
    return _CENTER_INDEX[key]


def center_set(key: str) -> str:
    if key.startswith("j_"):
        return "Joker"
    if key.startswith("v_"):
        return "Voucher"
    if key.startswith("m_"):
        return "Enhanced"
    if key.startswith("p_"):
        return "Booster"
    if key == "c_base":
        return "Default"
    return center(key)["set"]


def rarity_from_roll(r: float) -> int:
    """common_events.lua:1970: ``(r > 0.95 and 3) or (r > 0.7 and 2) or 1``."""
    return 3 if r > 0.95 else (2 if r > 0.7 else 1)


def get_current_pool(state: RunState, _type: str, _rarity: Optional[float] = None,
                     _legendary: bool = False, _append: Optional[str] = None,
                     ante: Optional[int] = None):
    """Port of ``get_current_pool``.  Returns ``(pool, pool_key, rarity)`` where ``pool`` is
    the culled key list with ``'UNAVAILABLE'`` written IN PLACE for ineligible entries
    (so indices never shift), ``pool_key`` is the finished pseudoseed key (ante suffix
    included) and ``rarity`` is 1..4 for jokers else None.

    Side effect: for jokers without an explicit ``_rarity`` this consumes one draw of
    ``'rarity'..ante..append`` -- even when ``_legendary`` is true (the Lua rolls first and
    overrides afterwards).

    ``ante``: the ``G.GAME.round_resets.ante`` the Lua reads -- ``state.ante`` by default;
    ``create_card`` passes ``Keys.gen_ante(state)`` because the Multiplayer mod's wrapper
    zeroes the ante for the duration of ``create_card`` under The Order.  Callers OUTSIDE
    create_card (tags, vouchers) keep the real ante (the mod does not patch them).
    """
    if ante is None:
        ante = state.ante
    append = _append or ""
    rarity_n: Optional[int] = None
    if _type == "Joker":
        if _legendary and _is_order(state):
            # Steamodded's get_current_pool (rarity.toml) resolves `_legendary` BEFORE the
            # roll, so The Soul does not step the rarity stream.  Unobservable in vanilla
            # (the 'rarity<a>sou' stream is the Soul's own) but under The Order the key is
            # the shared 'rarity0', so the SMODS behaviour is the one to reproduce.
            r = None
        else:
            r = _rarity if _rarity is not None else state.rng.pseudorandom(Keys.rarity(ante, append))
        rarity_n = 4 if _legendary else rarity_from_roll(r)
        starting = P.JOKERS_BY_RARITY[rarity_n]
        pool_key = Keys.joker_pool(rarity_n, append, ante, legendary=_legendary)
    else:
        if _type == "Tarot_Planet":
            starting = [c["key"] for c in _tarot_planet_pool()]
        else:
            starting = P.POOL_KEYS[_type]
        pool_key = Keys.center_pool(_type, append, ante)

    pool: list = []
    size = 0
    for key in starting:
        add = False
        if _type == "Enhanced":
            add = True
        elif _type == "Demo":
            add = True
        elif _type == "Tag":
            t = P.TAG_BY_KEY[key]
            req = t.get("requires")
            if (not req or state.is_discovered(req)) and (not t.get("min_ante") or t["min_ante"] <= state.ante):
                add = True
        else:
            c = center(key)
            cset = center_set(key)
            if not state.used_blocks(key) and (state.is_unlocked(key) or c.get("rarity") == 4):
                if cset == "Voucher":
                    if key not in state.used_vouchers:
                        include = True
                        for req in (c.get("requires") or []):
                            if req not in state.used_vouchers:
                                include = False
                        if key in state.shop_voucher_keys:
                            include = False
                        if include:
                            add = True
                elif cset == "Planet":
                    if not c.get("softlock") or state.hands_played.get(c["hand_type"], 0) > 0:
                        add = True
                elif c.get("enhancement_gate"):
                    add = c["enhancement_gate"] in state.deck_enhancements
                else:
                    add = True
                if c.get("name") in ("Black Hole", "The Soul"):
                    add = False
            npf = c.get("no_pool_flag") if cset == "Joker" else None
            ypf = c.get("yes_pool_flag") if cset == "Joker" else None
            if npf and npf in state.pool_flags:
                add = False
            if ypf and ypf not in state.pool_flags:
                add = False
        if add and key not in state.banned_keys:
            pool.append(key)
            size += 1
        else:
            pool.append(UNAVAILABLE)

    if size == 0:
        pool = [P.EMPTY_POOL_FALLBACK.get(_type, P.EMPTY_POOL_FALLBACK["*"])]
    return pool, pool_key, rarity_n


def draw_from_pool(state: RunState, pool: Sequence[str], pool_key: str, order_resample: bool = False):
    """The draw + resample loop shared by create_card / get_next_voucher_key /
    get_next_tag_key (common_events.lua:1904-1909, 1917-1922, 2116-2121)::

        center = pseudorandom_element(pool, pseudoseed(pool_key))
        it = 1
        while center == 'UNAVAILABLE':
            it = it + 1
            center = pseudorandom_element(pool, pseudoseed(pool_key..'_resample'..it))

    Returns ``(key, index0, resamples)``.  The resample keys are their own per-key streams:
    the main ``pool_key`` stream advances exactly once per call no matter how many redraws
    happen, which is what keeps other slots / later antes untouched.

    ``order_resample``: the Multiplayer mod's TheOrder.toml "Resample advances queue instead
    of rerolling" patch (common_events.lua create_card line 2118 -- and the identical
    get_next_voucher_key line 1908, superseded by the mod's own voucher path).  Under The
    Order each redraw is ANOTHER step of the ``pool_key`` stream itself; after 1000 redraws
    a ``'_resample'..it`` draw is made on top (both draws happen).  ``get_next_tag_key``'s
    loop (line 1921, ``_tag = ...``) does not match the pattern and stays vanilla.
    """
    key, idx = state.rng.pseudorandom_element(pool, pool_key)
    it = 1
    while key == UNAVAILABLE:
        it += 1
        if order_resample:
            key, idx = state.rng.pseudorandom_element(pool, pool_key)
            if it > 1000:  # the mod's fallback
                key, idx = state.rng.pseudorandom_element(pool, Keys.resample(pool_key, it))
        else:
            key, idx = state.rng.pseudorandom_element(pool, Keys.resample(pool_key, it))
    return key, idx, it - 1


# ----------------------------------------------------------------------------------------
# poll_edition (common_events.lua:2055-2080) and seals
# ----------------------------------------------------------------------------------------

def poll_edition(state: RunState, key: Optional[str], mod: Optional[float] = None,
                 no_neg: bool = False, guaranteed: bool = False) -> Optional[str]:
    """Exact port.  Note NEGATIVE is NOT scaled by ``edition_rate`` (Hone/Glow Up), only
    polychrome/holo/foil are; ``mod`` scales everything."""
    mod = 1 if mod is None else mod
    poll = state.rng.pseudorandom(key or "edition_generic")
    if guaranteed:
        if poll > 1 - 0.003 * 25 and not no_neg:
            return "negative"
        elif poll > 1 - 0.006 * 25:
            return "polychrome"
        elif poll > 1 - 0.02 * 25:
            return "holo"
        elif poll > 1 - 0.04 * 25:
            return "foil"
    else:
        er = state.edition_rate
        if poll > 1 - 0.003 * mod and not no_neg:
            return "negative"
        elif poll > 1 - 0.006 * er * mod:
            return "polychrome"
        elif poll > 1 - 0.02 * er * mod:
            return "holo"
        elif poll > 1 - 0.04 * er * mod:
            return "foil"
    return None


def seal_from_type_poll(x: float) -> str:
    """card.lua:1767-1770 / 2470-2473: ``>0.75 Red, >0.5 Blue, >0.25 Gold, else Purple``."""
    if x > 0.75:
        return "Red"
    if x > 0.5:
        return "Blue"
    if x > 0.25:
        return "Gold"
    return "Purple"


def poll_seal(state: RunState, type_key: str, gate_key: Optional[str] = None,
              gate_threshold: float = 0.8) -> Optional[str]:
    """Seal rolls are inline in the Lua, two shapes:

    * Standard pack (card.lua:1763-1772): ``pseudorandom(pseudoseed('stdseal'..ante)) > 1 - 0.02*10``
      gates a second draw ``pseudorandom(pseudoseed('stdsealtype'..ante))`` -> type.
      -> ``poll_seal(state, Keys.stdsealtype(ante), Keys.stdseal(ante))``
    * Certificate (card.lua:2469-2473): single draw ``'certsl'`` -> type, no gate.
      -> ``poll_seal(state, 'certsl')``
    """
    if gate_key is not None:
        if not (state.rng.pseudorandom(gate_key) > gate_threshold):
            return None
    return seal_from_type_poll(state.rng.pseudorandom(type_key))


# ----------------------------------------------------------------------------------------
# create_card (common_events.lua:2082-2154)
# ----------------------------------------------------------------------------------------

def create_card(state: RunState, _type: str, area: str = "shop", legendary: bool = False,
                rarity: Optional[float] = None, soulable: bool = False,
                forced_key: Optional[str] = None, key_append: Optional[str] = None) -> CardGen:
    """Port of ``create_card(_type, area, legendary, _rarity, skip_materialize, soulable,
    forced_key, key_append)``.  ``area`` is one of 'shop' (G.shop_jokers), 'pack'
    (G.pack_cards), 'jokers', 'consumables', 'deck'; only shop/pack matter (sticker rolls).

    Order of RNG consumption, exactly as the Lua:
      1. soul roll(s)            'soul_'..type..ante          (only if soulable and no forced key)
      2. rarity roll             'rarity'..ante..append       (Joker without explicit rarity)
      3. pool draw + resamples   pool_key / pool_key..'_resample'..it
      4. front                   'front'..append..ante        (Base/Enhanced)
      5. sticker poll            'etperpoll'|'packetper'..ante (Joker in shop/pack, ALWAYS rolled)
      6. rental poll             'ssjr'|'packssjr'..ante      (only if rentals enabled)
      7. edition                 'edi'..append..ante          (every Joker)
    ``used_jokers[key]`` is marked between 4 and 5 (Card:set_ability runs in the constructor).

    Under The Order (``state.key_scope == "run"``, NOTES_ORDER.md section 3) the Multiplayer
    mod wraps create_card: the ante reads as 0 for every key above, ``key_append`` becomes
    ``_type`` / ``_type..'_pack'`` for consumables and nil for jokers (``Keys.order_key_append``),
    resamples re-step the pool stream (``draw_from_pool(order_resample=True)``), and Jokers
    are picked by the mod's own loop (``_order_joker_pick``) whose stickers and
    ``'edi'..pool_key`` edition replace steps 5-7 (the patched create_card returns early).
    """
    order = _is_order(state)
    original_append = key_append or ""
    if order:
        key_append = Keys.order_key_append(_type, area, key_append)
    append = key_append or ""
    ante = Keys.gen_ante(state)           # 0 under The Order (the mod's wrapper), else state.ante
    rng = state.rng
    forced = forced_key
    rarity_n: Optional[int] = None
    order_ed = None
    order_stickers = (False, False, False)

    # 1. The Soul / Black Hole forced-key rolls (both independent; a Black Hole hit overrides)
    if not forced and soulable and ("c_soul" not in state.banned_keys):
        if _type in ("Tarot", "Spectral", "Tarot_Planet") and not state.used_blocks("c_soul"):
            if rng.pseudorandom(Keys.soul(_type, ante)) > 0.997:
                forced = "c_soul"
        if _type in ("Planet", "Spectral") and not state.used_blocks("c_black_hole"):
            if rng.pseudorandom(Keys.soul(_type, ante)) > 0.997:
                forced = "c_black_hole"

    if _type == "Base":
        forced = "c_base"

    pool_key = None
    idx = None
    resamples = 0
    was_forced = False
    if order and _type == "Joker" and not forced:
        # TheOrder.toml "stable shop": the mod picks the joker (and its stickers/edition)
        # BEFORE vanilla's forced_key test and hands it over as forced_key.
        if original_append == "jud" and state.enable_eternals_in_shop:
            rarity = rng.pseudorandom(Keys.ORDER_JUD_RARITY)   # the wrapper's separate Judgement rarity queue
        forced, pool_key, idx, resamples, rarity_n, order_stickers, order_ed = \
            _order_joker_pick(state, area, rarity, legendary, ante)
    if forced and forced not in state.banned_keys:
        key = forced
        was_forced = pool_key is None
        cset = center_set(key)
        if cset != "Default":
            _type = cset
        if cset == "Joker" and rarity_n is None:
            rarity_n = center(key)["rarity"]
    else:
        # 2-3. pool + draw
        pool, pool_key, rarity_n = get_current_pool(state, _type, rarity, legendary, append, ante=ante)
        key, idx, resamples = draw_from_pool(state, pool, pool_key, order_resample=order)

    cset = center_set(key)

    # 4. playing-card front
    front = None
    if _type in ("Base", "Enhanced"):
        front, _ = rng.pseudorandom_element(P.PLAYING_CARD_KEYS, Keys.front(append, ante))

    # Card:set_ability -> used_jokers[key] = true (card.lua:349-354)
    state.mark_used(key)

    card = CardGen(key=key, set=cset, type_requested=_type, key_append=append, area=area,
                   front=front, rarity=rarity_n, pool_key=pool_key, pool_index=idx,
                   resamples=resamples, forced=was_forced)

    # 5-7. joker-only modifiers
    if _type == "Joker":
        if state.all_eternal:
            card.eternal = True
        if order:
            # the mod's early return: its own stickers + 'edi'..pool_key edition, nothing else
            et, per, rent = order_stickers
            card.eternal = card.eternal or et
            card.perishable = per
            card.rental = rent
            card.edition = order_ed
            return card
        if area in ("shop", "pack"):
            poll = rng.pseudorandom(Keys.sticker_poll(area, ante))       # ALWAYS consumed
            if state.enable_eternals_in_shop and poll > 0.7:
                card.eternal = True
            elif state.enable_perishables_in_shop and (poll > 0.4) and (poll <= 0.7):
                card.perishable = True
            if state.enable_rentals_in_shop and rng.pseudorandom(Keys.rental_poll(area, ante)) > 0.7:
                card.rental = True
        card.edition = poll_edition(state, Keys.edition(append, ante))
    return card


def _order_joker_pick(state: RunState, area: str, rarity: Optional[float], legendary: bool, ante: int):
    """The Order's joker selection loop (TheOrder.toml, inserted before create_card's
    ``if forced_key and not G.GAME.banned_keys[forced_key]``):

        _pool, _pool_key = get_current_pool('Joker', _rarity, legendary, nil)   -- ante is 0 here
        forced_key, it = 'UNAVAILABLE', 0
        while forced_key == 'UNAVAILABLE' do
            (shop/pack only) eternal_perishable_poll = pseudorandom('_etper'.._pool_key)
                             rental: pseudorandom('_rent'.._pool_key) (rentals enabled only)
            _s_append = any sticker and '_sticker' or ''
            forced_key = pseudorandom_element(_pool, pseudoseed(_pool_key.._s_append))
            if it > 1000 then forced_key = pseudorandom_element(_pool, pseudoseed(_pool_key.._s_append..'_resample'..it)) end
            if forced_key ~= 'UNAVAILABLE' then _order_ed = poll_edition('edi'.._pool_key.._s_append)
            else pseudorandom(pseudoseed('edi'.._pool_key.._s_append)) end   -- same one-step advance
            it = it + 1
        end

    Stickered jokers therefore live on their OWN queue (``Joker1<ante0>_sticker``) and the
    edition queue is per rarity pool (``ediJoker10``), not per key_append.
    Returns ``(key, pool_key, index0, resamples, rarity, (eternal, perishable, rental), edition)``.
    """
    rng = state.rng
    pool, pool_key, rarity_n = get_current_pool(state, "Joker", rarity, legendary, None, ante=ante)
    it = 0
    key = UNAVAILABLE
    idx = None
    while key == UNAVAILABLE:
        et = per = rent = False
        if area in ("shop", "pack"):
            poll = rng.pseudorandom(Keys.order_sticker_poll(pool_key))
            if state.enable_eternals_in_shop and poll > 0.7:
                et = True
            elif state.enable_perishables_in_shop and (poll > 0.4) and (poll <= 0.7):
                per = True
            if state.enable_rentals_in_shop and rng.pseudorandom(Keys.order_rental_poll(pool_key)) > 0.7:
                rent = True
        sticker = et or per or rent
        key, idx = rng.pseudorandom_element(pool, Keys.order_joker_draw(pool_key, sticker))
        if it > 1000:  # the mod's fallback
            key, idx = rng.pseudorandom_element(pool, Keys.resample(Keys.order_joker_draw(pool_key, sticker), it))
        if key != UNAVAILABLE:
            ed = poll_edition(state, Keys.order_joker_edition(pool_key, sticker))
        else:
            rng.pseudorandom(Keys.order_joker_edition(pool_key, sticker))
        it += 1
    return key, pool_key, idx, it - 1, rarity_n, (et, per, rent), ed


# ----------------------------------------------------------------------------------------
# Shop (UI_definitions.lua:742-800 create_card_for_shop; game.lua:3072-3181 update_shop;
#       button_callbacks.lua:2855-2911 reroll_shop)
# ----------------------------------------------------------------------------------------

def shop_type_table(state: RunState):
    """The five (type, rate) rows create_card_for_shop walks.  Building the table itself
    consumes one ``'illusion'`` draw when Illusion is owned (the Enhanced/Base decision is
    made for every slot, whether or not a playing card is rolled)."""
    if "v_illusion" in state.used_vouchers and state.rng.pseudorandom(Keys.ILLUSION) > 0.6:
        pc_type = "Enhanced"
    else:
        pc_type = "Base"
    return [
        ("Joker", state.joker_rate),
        ("Tarot", state.tarot_rate),
        ("Planet", state.planet_rate),
        (pc_type, state.playing_card_rate),
        ("Spectral", state.spectral_rate),
    ]


def create_card_for_shop(state: RunState) -> Optional[CardGen]:
    """One G.shop_jokers slot (tags handled by the caller, see ``_fill_shop_slot``)."""
    rng = state.rng
    total_rate = state.joker_rate + state.tarot_rate + state.planet_rate + state.playing_card_rate + state.spectral_rate
    polled_rate = rng.pseudorandom(Keys.cdt(Keys.gen_ante(state))) * total_rate   # 'cdt0' under The Order
    check_rate = 0
    for _type, val in shop_type_table(state):
        if polled_rate > check_rate and polled_rate <= check_rate + val:
            card = create_card(state, _type, area="shop", key_append="sho")
            if _type in ("Base", "Enhanced") and "v_illusion" in state.used_vouchers \
                    and rng.pseudorandom(Keys.ILLUSION) > 0.8:
                ep = rng.pseudorandom(Keys.ILLUSION)
                if ep > 1 - 0.15:
                    card.edition = "polychrome"
                elif ep > 0.5:
                    card.edition = "holo"
                else:
                    card.edition = "foil"
            return card
        check_rate = check_rate + val
    return None  # only if polled_rate == 0 exactly (Lua would then emplace nil)


def _tag_store_joker_create(state: RunState) -> Optional[CardGen]:
    """tag.lua:344-374: the first untriggered Uncommon/Rare tag forces the slot."""
    for i, tkey in enumerate(state.tags):
        if i in state.triggered_tags:
            continue
        if tkey == "tag_rare":
            # "#G.P_JOKER_RARITY_POOLS[3] > distinct rares owned"
            owned_rares = {k for k in state.owned_jokers if P.JOKER_BY_KEY[k]["rarity"] == 3}
            state.triggered_tags.add(i)
            if len(P.JOKER_POOL_RARITY_3) > len(owned_rares):
                card = create_card(state, "Joker", area="shop", rarity=1, key_append="rta")
                card.couponed = True
                card.from_tag = tkey
                return card
            return None  # tag:nope() -- consumed, no card; next tag may still apply
        if tkey == "tag_uncommon":
            state.triggered_tags.add(i)
            card = create_card(state, "Joker", area="shop", rarity=0.9, key_append="uta")
            card.couponed = True
            card.from_tag = tkey
            return card
    return None


def _tag_store_joker_modify(state: RunState, card: CardGen) -> None:
    """tag.lua:395-446: first untriggered Foil/Holo/Polychrome/Negative tag edits an
    edition-less Joker (consumes the tag)."""
    if card is None or card.set != "Joker" or card.edition:
        return
    editions = {"tag_foil": "foil", "tag_holo": "holo", "tag_polychrome": "polychrome", "tag_negative": "negative"}
    for i, tkey in enumerate(state.tags):
        if i in state.triggered_tags or tkey not in editions:
            continue
        state.triggered_tags.add(i)
        card.edition = editions[tkey]
        card.couponed = True
        card.from_tag = tkey
        return


def _fill_shop_slot(state: RunState) -> Optional[CardGen]:
    """create_card_for_shop including the tag hooks, in the Lua order: store_joker_create
    tags first (Uncommon/Rare); otherwise the rate roll; then store_joker_modify."""
    forced = None
    # the Lua loops tags until one returns a card; a Rare tag that nope()s returns nil and the
    # loop continues to the next untriggered tag
    while True:
        before = len(state.triggered_tags)
        forced = _tag_store_joker_create(state)
        if forced is not None or len(state.triggered_tags) == before:
            break
    if forced is not None:
        _tag_store_joker_modify(state, forced)
        return forced
    card = create_card_for_shop(state)
    _tag_store_joker_modify(state, card)
    return card


def get_pack(state: RunState, key: Optional[str] = "shop_pack", kind: Optional[str] = None) -> str:
    """Port of ``get_pack(_key, _type)`` (common_events.lua:1944-1961).

    The very first pack ever requested in a run is a Buffoon pack (no RNG: the ``_1/_2`` art
    suffix comes from the UNSEEDED global ``math.random`` and is cosmetic -- we return ``_1``).
    Otherwise a weighted pick over ``P_CENTER_POOLS.Booster`` in pool order with key
    ``(_key or 'pack_generic')..ante``; banned packs are dropped from the weight total.
    """
    if not state.first_shop_buffoon and "p_buffoon_normal_1" not in state.banned_keys:
        state.first_shop_buffoon = True
        return "p_buffoon_normal_1"
    cume = 0
    for v in P.BOOSTERS:
        if (kind is None or kind == v["kind"]) and v["key"] not in state.banned_keys:
            cume = cume + (v["weight"] or 1)
    poll = state.rng.pseudorandom(Keys.pack(key, Keys.gen_ante(state))) * cume    # 'shop_pack0' under The Order
    it = 0
    for v in P.BOOSTERS:
        if v["key"] not in state.banned_keys:
            w = v["weight"] or 1
            if kind is None or kind == v["kind"]:
                it = it + w
            if it >= poll and it - w <= poll:
                return v["key"]
    raise AssertionError("get_pack: no booster matched poll=%r cume=%r" % (poll, cume))


def generate_shop(state: RunState) -> ShopContents:
    """A FRESH shop (G.shop did not exist): game.lua:3072-3181.

    Order: tags ``shop_start`` (D6, no RNG) -> fill ``shop_joker_max`` slots via
    create_card_for_shop -> voucher card from ``current_round.voucher`` (no draw here; it
    was drawn at run start / boss defeat) -> two boosters via ``get_pack('shop_pack')``
    unless ``used_packs[i]`` already set this round -> tags ``voucher_add`` (Voucher Tag:
    ``get_next_voucher_key(true)``) -> tags ``shop_final_pass`` (Coupon, no RNG).
    """
    shop = ShopContents()
    state.triggered_tags = set()
    state.shop_voucher_keys = []
    for _ in range(state.shop_joker_max - len(shop.cards)):
        card = _fill_shop_slot(state)
        if card is not None:
            shop.cards.append(card)
    if state.current_round_voucher:
        shop.voucher = state.current_round_voucher
        state.shop_voucher_keys.append(shop.voucher)
    for i in range(2):
        if state.used_packs[i] is None:
            state.used_packs[i] = get_pack(state, "shop_pack")
        shop.boosters.append(None if state.used_packs[i] == "USED" else state.used_packs[i])
    for i, tkey in enumerate(state.tags):
        if tkey == "tag_voucher" and i not in state.triggered_tags:
            state.triggered_tags.add(i)
            vk = next_voucher(state, from_tag=True)
            shop.tag_vouchers.append(vk)
            state.shop_voucher_keys.append(vk)
    return shop


def reroll_shop(state: RunState, shop: ShopContents) -> ShopContents:
    """button_callbacks.lua:2855-2911: every G.shop_jokers card is ``Card:remove``d (so its
    ``used_jokers`` mark clears unless owned), then ``shop_joker_max`` slots are refilled by
    the SAME create_card_for_shop path -- same keys ('cdt'..ante, 'rarity'..ante..'sho',
    'Joker<r>sho'..ante, 'edisho'..ante, ...), each advanced by one more step.  Vouchers
    and boosters are untouched.  Mutates and returns ``shop``."""
    for c in shop.cards:
        state.release(c.key)
    shop.cards = []
    for _ in range(state.shop_joker_max):
        card = _fill_shop_slot(state)
        if card is not None:
            shop.cards.append(card)
    shop.rerolls += 1
    return shop


def open_pack(state: RunState, pack_key: str, most_played_hand: Optional[str] = None) -> list:
    """Booster contents, card.lua:1723-1784 ``Card:open``.  Cards are created in order in
    one pass; each is marked in ``used_jokers`` as it is created, which is the ONLY
    duplicate suppression (it is a resample, not a redraw, and it also excludes cards
    currently displayed in the shop and owned cards).

    ``most_played_hand``: Telescope forces card 1 of a Celestial pack to the planet of the
    most-played visible hand (first max in G.handlist order, ties to the earlier hand);
    pass None to derive it from ``state.hands_played``.
    """
    b = next(x for x in P.BOOSTERS if x["key"] == pack_key)
    kind = b["kind"]
    size = b["extra"]
    ante = Keys.gen_ante(state)   # Standard-pack keys use MP.ante_based() / the mod's poll_seal wrapper under The Order
    rng = state.rng
    cards: list = []
    for i in range(1, size + 1):
        if kind == "Arcana":
            if "v_omen_globe" in state.used_vouchers and rng.pseudorandom(Keys.OMEN_GLOBE) > 0.8:
                card = create_card(state, "Spectral", area="pack", soulable=True, key_append="ar2")
            else:
                card = create_card(state, "Tarot", area="pack", soulable=True, key_append="ar1")
        elif kind == "Celestial":
            if "v_telescope" in state.used_vouchers and i == 1:
                hand = most_played_hand
                if hand is None:
                    tally = 0
                    for h in P.HANDLIST:
                        visible = P.POKER_HANDS[P.HANDLIST.index(h)]["visible"] or state.hands_played.get(h, 0) > 0
                        if visible and state.hands_played.get(h, 0) > tally:
                            hand, tally = h, state.hands_played[h]
                planet = None
                if hand is not None:
                    for pl in P.PLANETS:
                        if pl["hand_type"] == hand:
                            planet = pl["key"]
                card = create_card(state, "Planet", area="pack", soulable=True, forced_key=planet, key_append="pl1")
            else:
                card = create_card(state, "Planet", area="pack", soulable=True, key_append="pl1")
        elif kind == "Spectral":
            card = create_card(state, "Spectral", area="pack", soulable=True, key_append="spe")
        elif kind == "Standard":
            _type = "Enhanced" if rng.pseudorandom(Keys.stdset(ante)) > 0.6 else "Base"
            card = create_card(state, _type, area="pack", soulable=True, key_append="sta")
            card.edition = poll_edition(state, Keys.standard_edition(ante), 2, True)
            card.seal = poll_seal(state, Keys.stdsealtype(ante), Keys.stdseal(ante), 1 - 0.02 * 10)
        elif kind == "Buffoon":
            card = create_card(state, "Joker", area="pack", soulable=True, key_append="buf")
        else:  # pragma: no cover
            raise ValueError(pack_key)
        cards.append(card)
    return cards


# ----------------------------------------------------------------------------------------
# Vouchers, bosses, tags  (common_events.lua:1901-1925, 2338-2383)
# ----------------------------------------------------------------------------------------

def get_culled(pool: Sequence[str]) -> list:
    """The Multiplayer mod's ``get_culled(_pool)`` (compatibility/TheOrder.lua): collapse the
    voucher pool's (base, upgrade) pairs -- ``P_CENTER_POOLS.Voucher`` interleaves them --
    into one entry per pair: both if both available (only modded tier-3 vouchers), the
    available one, or ``'UNAVAILABLE'`` when neither is.  A trailing unpaired entry is
    kept as is."""
    culled: list = []
    n = len(pool)
    for i in range(0, n, 2):
        first = pool[i]
        second = pool[i + 1] if i + 1 < n else None
        if second is None:
            culled.append(first if first != UNAVAILABLE else UNAVAILABLE)
        elif first != UNAVAILABLE and second != UNAVAILABLE:
            culled.append(first)
            culled.append(second)
        elif first != UNAVAILABLE:
            culled.append(first)
        elif second != UNAVAILABLE:
            culled.append(second)
        else:
            culled.append(UNAVAILABLE)
    return culled


def _next_voucher_culled(state: RunState, pool: Sequence[str], spawn=None) -> str:
    """MLB / The Order voucher draw (TheOrder.lua ``SMODS.get_next_vouchers`` /
    ``get_next_voucher_key`` overrides, guarded by ``should_use_the_order() or
    is_major_league_ruleset()``)::

        local culled = get_culled(_pool)
        local center = pseudorandom_element(culled, pseudoseed("Voucher0"))
        local it = 1
        while center == "UNAVAILABLE" [or vouchers.spawn[center]] do
            it = it + 1
            center = pseudorandom_element(culled, pseudoseed("Voucher0"))
            if it > 1000 then center = pseudorandom_element(culled, pseudoseed("Voucher0"..it)) end
        end

    One run-global ``'Voucher0'`` stream for the shop voucher AND the Voucher Tag (no
    ``Voucher<ante>`` / ``Voucher_fromtag``); redraws re-step the same stream.  ``spawn`` is
    the shop-voucher path's already-spawned set (``vouchers.spawn``; empty with the vanilla
    one voucher per shop)."""
    culled = get_culled(pool)
    rng = state.rng
    key, _ = rng.pseudorandom_element(culled, Keys.VOUCHER_ORDER)
    it = 1
    while key == UNAVAILABLE or (spawn is not None and key in spawn):
        it += 1
        key, _ = rng.pseudorandom_element(culled, Keys.VOUCHER_ORDER)
        if it > 1000:  # the mod's fallback
            key, _ = rng.pseudorandom_element(culled, Keys.voucher_order_fallback(it))
    return key


def next_voucher(state: RunState, from_tag: bool = False, spawn=None) -> str:
    """``get_next_voucher_key(_from_tag)``: pool 'Voucher' culled by used_vouchers /
    requires / currently-displayed; key ``'Voucher'..ante`` (shop voucher, drawn at run start
    and after each boss) or ``'Voucher_fromtag'`` (Voucher Tag, NO ante suffix).

    Under ``ruleset == "mlb"`` or ``key_scope == "run"`` both draws go through the
    Multiplayer mod's culled-pool ``'Voucher0'`` path instead (``_next_voucher_culled``);
    ``spawn`` is only meaningful there (shop path; leave None for the Voucher Tag)."""
    pool, pool_key, _ = get_current_pool(state, "Voucher")
    if _culled_vouchers(state):
        return _next_voucher_culled(state, pool, None if from_tag else spawn)
    if from_tag:
        pool_key = Keys.VOUCHER_FROM_TAG
    key, _, _ = draw_from_pool(state, pool, pool_key)
    return key


def next_tag(state: RunState, append: Optional[str] = None) -> str:
    """``get_next_tag_key(append)``: key ``'Tag'..(append or '')..ante``; eligibility is
    ``requires`` discovered + ``min_ante``.  Drawn twice per ante (Small then Big)."""
    if state.force_tag:
        return state.force_tag
    pool, pool_key, _ = get_current_pool(state, "Tag", None, False, append)
    key, _, _ = draw_from_pool(state, pool, pool_key)
    return key


def eligible_bosses(state: RunState) -> list:
    """The candidate list ``get_new_boss`` ends up drawing from, in the order
    ``pseudorandom_element`` sorts it (alphabetical by key, because the table's values are
    numbers, not sort_id tables)."""
    ante = state.ante
    a1 = max(1, ante)
    elig: dict = {}
    for b in P.BOSS_BLINDS:
        k = b["key"]
        if not b["showdown"] and (b["boss_min"] <= a1 and (a1 % state.win_ante != 0 or ante < 2)):
            elig[k] = True
        elif b["showdown"] and ante % state.win_ante == 0 and ante >= 2:
            elig[k] = True
    for k in state.banned_keys:
        elig.pop(k, None)
    min_use = 100
    for k, v in state.bosses_used.items():
        if k in elig:
            elig[k] = v
            if v <= min_use:
                min_use = v
    for k in list(elig):
        if elig[k] is True:  # boss missing from bosses_used -> the Lua would error; treat as 0 uses
            elig[k] = 0
            min_use = min(min_use, 0)
        if elig[k] > min_use:
            del elig[k]
    return sorted(elig)


def next_boss(state: RunState) -> str:
    """``get_new_boss()``: key ``'boss'`` (no ante; ``'boss'..ante`` under The Order).  Eligible = non-showdown bosses with
    ``min <= max(1,ante)`` unless ``max(1,ante) % win_ante == 0`` (ante >= 2), else the
    five showdown bosses; minus ``banned_keys``; then only those with the MINIMUM
    ``bosses_used`` count (this is the "reset when exhausted": every eligible boss is seen
    once before any repeats).  Increments ``bosses_used[boss]``."""
    pb = state.perscribed_bosses.get(state.ante)
    if pb:
        del state.perscribed_bosses[state.ante]
        state.bosses_used[pb] = state.bosses_used.get(pb, 0) + 1
        return pb
    if state.force_boss:
        return state.force_boss
    cands = eligible_bosses(state)
    boss, _ = state.rng.pseudorandom_element(cands, Keys.boss(state))   # 'boss' / 'boss<ante>' (The Order)
    state.bosses_used[boss] = state.bosses_used.get(boss, 0) + 1
    return boss


# ----------------------------------------------------------------------------------------
# Run start / ante transitions
# ----------------------------------------------------------------------------------------

DECK_EFFECTS = {
    # Back:apply_to_run (back.lua:173-288) + Card:apply_to_run (card.lua:1880-1971); only the
    # parts that touch generation.
    "b_red": {}, "b_blue": {}, "b_yellow": {}, "b_green": {}, "b_black": {},
    "b_magic": {"vouchers": ["v_crystal_ball"], "consumables": ["c_fool", "c_fool"]},
    "b_nebula": {"vouchers": ["v_telescope"]},
    "b_ghost": {"spectral_rate": 2, "consumables": ["c_hex"]},
    "b_abandoned": {"no_faces": True},
    "b_checkered": {"checkered": True},
    "b_zodiac": {"vouchers": ["v_tarot_merchant", "v_planet_merchant", "v_overstock_norm"]},
    "b_painted": {}, "b_anaglyph": {}, "b_plasma": {},
    "b_erratic": {"erratic": True},
}


def apply_deck(state: RunState, deck_key: str) -> None:
    """Deck effects that change generation (no RNG is consumed here except that the
    deck-granted consumables mark ``used_jokers`` -- they are created with forced keys,
    ``create_card('Tarot', G.consumeables, nil, nil, nil, nil, key, 'deck')``, no draw)."""
    eff = DECK_EFFECTS[deck_key]
    for v in eff.get("vouchers", []):
        state.used_vouchers.add(v)
        rate = P.SHOP_RATE_BY_VOUCHER.get(v)
        if rate:
            setattr(state, rate[0], rate[1])
        if v in ("v_overstock_norm", "v_overstock_plus"):
            state.shop_joker_max += 1
    for c in eff.get("consumables", []):
        state.owned_consumables.append(c)
        state.mark_used(c)
    if "spectral_rate" in eff:
        state.spectral_rate = eff["spectral_rate"]


def build_starting_deck(state: RunState, erratic: bool = False, no_faces: bool = False,
                        checkered: bool = False) -> list:
    """game.lua:2330-2378: the 52 card protos, sorted by ``suit..rank`` string (so the
    initial ``sort_id`` order is C2..C9,CA,CJ,CK,CQ,CT,D2,...,S...), then Card objects are
    created in that order.  Erratic draws each proto from ``pseudorandom_element(G.P_CARDS,
    pseudoseed('erratic'))`` -- 52 draws, one per P_CARDS entry.  Returns playing-card
    keys in sort_id order."""
    protos = []
    for k in P.PLAYING_CARD_KEYS:
        if erratic:
            k, _ = state.rng.pseudorandom_element(P.PLAYING_CARD_KEYS, Keys.ERRATIC)
        r, s = k[2], k[0]
        if no_faces and r in ("K", "Q", "J"):
            continue
        protos.append((s, r))
    protos.sort(key=lambda p: p[0] + p[1])
    cards = [s + "_" + r for s, r in protos]
    if checkered:  # back.lua:244-256 Clubs->Spades, Diamonds->Hearts (after creation; sort_ids unchanged)
        cards = [("S" if c[0] == "C" else "H" if c[0] == "D" else c[0]) + c[1:] for c in cards]
    return cards


# ----------------------------------------------------------------------------------------
# The Order's playing-card / joker selection (compatibility/TheOrder.lua give_shufflevals,
# pseudoshuffle + pseudorandom_element overrides, reset_idol_card, reset_mail_rank)
# ----------------------------------------------------------------------------------------
#
# Under The Order the mod replaces ``pseudoshuffle`` for lists of playing cards and
# ``pseudorandom_element`` for lists of playing cards OR jokers with a value-ranking scheme:
#
#     true_seed = pseudorandom(seed)                     -- seed = pseudoseed(key) as usual
#     group the cards by suit..rank_id ('Hearts14'; 'Stone' for stone cards) or by joker key;
#     sort each group (cards: highest "stdval" first; jokers: oldest sort_id first);
#     for each group: mega = group_key .. true_seed (Lua %.14g), each member gets
#                     pseudorandom(mega) in turn, then G.GAME.pseudorandom[mega] = nil;
#     shuffle  -> list sorted by value DESCENDING;  element -> the maximum.
#
# Cards that are identical in every respect the scheme sees (same suit/rank/enhancement/
# seal/edition) tie inside their group; which of them gets which draw is decided by the exact
# behaviour of LuaJIT's unstable ``table.sort`` on the group's insertion (list) order, so
# that algorithm is ported verbatim (``lua_table_sort``).  The cards are indistinguishable for
# play anyway, but the port keeps the deck order bit-identical.

_RANK_ID = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}
_RANK_VALUE = {2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10: "10",
               11: "Jack", 12: "Queen", 13: "King", 14: "Ace"}
_SUIT_NAME = {"S": "Spades", "H": "Hearts", "C": "Clubs", "D": "Diamonds"}
_ENH_CENTER = {"None": "c_base", "": "c_base", None: "c_base", "Bonus": "m_bonus", "Mult": "m_mult",
               "Wild": "m_wild", "Glass": "m_glass", "Steel": "m_steel", "Stone": "m_stone",
               "Gold": "m_gold", "Lucky": "m_lucky"}
# TheOrder.lua `stdval` tables (give_stdval)
_STDVAL_CENTERS = {"c_base": 0, "m_stone": 106, "m_bonus": 107, "m_mult": 108, "m_wild": 109,
                   "m_gold": 110, "m_lucky": 111, "m_steel": 112, "m_glass": 113}
_STDVAL_SEALS = {"Gold": 122, "Blue": 131, "Purple": 140, "Red": 149}
_STDVAL_EDITIONS = {"foil": 157, "holo": 192, "polychrome": 227}
# SMODS.Rank / SMODS.Suit obj_buffer order (registration order in Steamodded game_object.lua):
# ranks 2..9, 10, Jack, Queen, King, Ace; suits Diamonds, Clubs, Hearts, Spades.
ORDER_RANK_INDEX = {v: i + 1 for i, v in enumerate(["2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King", "Ace"])}
ORDER_SUIT_INDEX = {"Diamonds": 1, "Clubs": 2, "Hearts": 3, "Spades": 4}
_RANK_NOMINAL = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10,
                 "Jack": 10, "Queen": 10, "King": 10, "Ace": 11}
_FACE_RANKS = {"Jack", "Queen", "King"}


def card_props(card) -> tuple:
    """``(suit_name, rank_id, center_key, seal, edition_type)`` of a playing card, from any of
    the shapes generation meets: a key string ``'H_7'`` (plain card), a dict with
    ``suit``/``rank``/``enhancement``|``center``/``seal``/``edition``, or an object with those
    attributes (the engine's ``Card``: suit name, rank 2-14, enhancement/edition/seal names,
    ``'None'`` for absent)."""
    if isinstance(card, str):
        s, r = card[0], card[2]
        return _SUIT_NAME[s], _RANK_ID[r], "c_base", None, None
    get = card.get if isinstance(card, dict) else (lambda k, d=None: getattr(card, k, d))
    suit = get("suit")
    suit = _SUIT_NAME.get(suit, suit)
    rank = get("rank")
    if isinstance(rank, str):
        rank = _RANK_ID.get(rank) or int(rank)
    center = get("center") or get("center_key")
    if center is None:
        center = _ENH_CENTER.get(get("enhancement"), "c_base")
    seal = get("seal")
    if seal in ("None", ""):
        seal = None
    edition = get("edition")
    if edition in ("None", ""):
        edition = None
    if edition is not None:
        edition = str(edition).lower()
        if edition == "holographic":
            edition = "holo"
    return suit, rank, center, seal, edition


def lua_table_sort(lst: list, lt) -> None:
    """In-place port of LuaJIT's ``table.sort`` (lib_table.c ``auxsort``, the Lua 5.1
    quicksort; identical in 2.0.5 and 2.1).  Needed because the mod sorts each shuffle group
    with ``a.mp_stdval > b.mp_stdval`` and the algorithm is NOT stable: where several cards
    tie (e.g. two plain Stone cards) the resulting order -- and hence which card gets which
    draw -- is whatever this exact sequence of compares and swaps produces."""
    a = [None] + list(lst)          # 1-based

    def swap(i, j):
        a[i], a[j] = a[j], a[i]

    def auxsort(l, u):
        while l < u:
            if lt(a[u], a[l]):
                swap(l, u)
            if u - l == 1:
                break
            i = (l + u) // 2
            if lt(a[i], a[l]):
                swap(i, l)
            elif lt(a[u], a[i]):
                swap(i, u)
            if u - l == 2:
                break
            P = a[i]
            swap(i, u - 1)
            i, j = l, u - 1
            while True:
                i += 1
                while lt(a[i], P):
                    if i > u:
                        raise ValueError("invalid order function for sorting")
                    i += 1
                j -= 1
                while lt(P, a[j]):
                    if j < l:
                        raise ValueError("invalid order function for sorting")
                    j -= 1
                if j < i:
                    break
                swap(i, j)
            swap(u - 1, i)
            if i - l < u - i:
                j, i, l = l, i - 1, i + 1
            else:
                j, i, u = i + 1, u, i - 1
            auxsort(j, i)

    auxsort(1, len(lst))
    lst[:] = a[1:]


def _order_group_key(props: tuple) -> str:
    suit, rank, center, _, _ = props
    return "Stone" if center == "m_stone" else suit + str(rank)


def _stdval(props: tuple) -> int:
    _, _, center, seal, edition = props
    return _STDVAL_CENTERS.get(center, 0) + _STDVAL_SEALS.get(seal or "", 0) + _STDVAL_EDITIONS.get(edition or "", 0)


def order_shufflevals(state: RunState, items: Sequence, key, jokers: bool = False) -> list:
    """``give_shufflevals(tbl, seed, joker)``: one value per item (list-aligned).  ``key`` is a
    key string (pseudoseed'ed here, as ``pseudoshuffle(list, pseudoseed(key))`` does) or an
    already computed seed float.  ``jokers``: items are joker keys in sort_id order; else
    playing cards (see ``card_props``)."""
    rng = state.rng
    seed = rng.pseudoseed(key) if isinstance(key, str) else key
    true_seed = rng.pseudorandom(seed)
    groups: dict = {}
    if jokers:
        for i, it in enumerate(items):
            groups.setdefault(it if isinstance(it, str) else getattr(it, "key", it), []).append(i)
    else:
        props = [card_props(c) for c in items]
        for i, pr in enumerate(props):
            groups.setdefault(_order_group_key(pr), []).append(i)
    vals = [0.0] * len(items)
    ts = _lua_num_str(true_seed)
    for gk, idxs in groups.items():          # groups are independent streams: order irrelevant
        if not jokers:
            lua_table_sort(idxs, lambda x, y: _stdval(props[x]) > _stdval(props[y]))   # highest value first (LuaJIT's sort, ties included)
        mega = gk + ts
        for i in idxs:
            vals[i] = rng.pseudorandom(mega)
        rng.drop_key(mega)
    return vals


def order_shuffle(state: RunState, cards: Sequence, key) -> list:
    """The Order's ``pseudoshuffle`` for playing cards: descending ``mp_shuffleval`` order
    (the game draws from the END of the deck list, i.e. the lowest values first)."""
    vals = order_shufflevals(state, cards, key, jokers=False)
    order = sorted(range(len(cards)), key=lambda i: (-vals[i], i))
    return [cards[i] for i in order]


def order_pick(state: RunState, items: Sequence, key, jokers: bool = False):
    """The Order's ``pseudorandom_element`` for a list of playing cards / jokers: the item
    with the highest ``mp_shuffleval``.  Returns ``(item, index0)``."""
    if not items:
        return None, None
    vals = order_shufflevals(state, items, key, jokers=jokers)
    i = max(range(len(items)), key=lambda j: (vals[j], -j))
    return items[i], i


def _order_shuffle_key(state: RunState, key) -> str:
    """Upgrade a vanilla ``'nr<ante>'`` / ``'cashout<ante>'`` key to the mod's round-based form
    when the caller built it without ``state`` (engines that predate the switch)."""
    if isinstance(key, str):
        if key == Keys.new_round_shuffle(state.ante):
            return Keys.new_round_shuffle(state.ante, state)
        if key == Keys.cashout_shuffle(state.ante):
            return Keys.cashout_shuffle(state.ante, state)
    return key


def shuffle_deck(state: RunState, cards: list, key: str = Keys.SHUFFLE) -> list:
    """``CardArea:shuffle(_seed)`` -> ``pseudoshuffle(self.cards, pseudoseed(_seed or 'shuffle'))``
    (cardarea.lua:572-575).  pseudoshuffle FIRST sorts the list by ``sort_id`` (misc_functions.lua:209)
    so the result depends only on the card set + creation order, not the current order.
    ``cards`` must be in sort_id order (creation order).  Cards are drawn from the END of
    the returned list (``CardArea:remove_card`` takes ``_cards[#_cards]`` for decks).

    Under The Order: the ``nr``/``cashout`` keys gain the blind suffix (``Keys.order_round``;
    requires ``state.blind_key``/``blind_type``) and the shuffle is ``order_shuffle``."""
    lst = list(cards)
    if _is_order(state):
        key = _order_shuffle_key(state, key)
        if key != Keys.SHUFFLE and (state.blind_key is None or state.blind_type is None) \
                and (key.startswith("nr") or key.startswith("cashout")):
            raise ValueError("The Order shuffle %r needs RunState.blind_key/blind_type set by the engine" % key)
        return order_shuffle(state, lst, key)
    state.rng.pseudoshuffle(lst, key)
    return lst


def _order_idol(state: RunState, cards: Sequence, ante: int):
    """The Order's ``reset_idol_card`` (TheOrder.lua:31-343): score every distinct
    (rank, suit) of the non-stone deck, sort, then a count-weighted walk with
    ``pseudorandom('idol'..ante)``.  Returns ``(rank_value, suit_name)`` or None."""
    W_GEN, GEN_FLOOR, TARGET = 0.05, 0.0, 5
    W_EDITION_A, W_EDITION_B, W_COUNT_A, W_MAIN, W_OFF, W_STR = 1.3, 0.7, 0.5, 2.0, 1.0, 1.0
    entries: dict = {}
    valid: list = []
    for c in cards:
        pr = card_props(c)
        if pr[2] == "m_stone":
            continue
        value, suit = _RANK_VALUE[pr[1]], pr[0]
        k = value + "_" + suit
        e = entries.get(k)
        if e is None:
            e = entries[k] = {"count": 0, "value": value, "suit": suit, "props": [], "wild": 0}
            valid.append(e)
        e["count"] += 1
        e["props"].append(pr)
        if pr[2] == "m_wild":
            e["wild"] += 1
    if not valid:
        return None
    rank_totals: dict = {}
    wild_by_rank: dict = {}
    for e in valid:
        rank_totals[e["value"]] = rank_totals.get(e["value"], 0) + e["count"]
        wild_by_rank[e["value"]] = wild_by_rank.get(e["value"], 0) + e["wild"]
    distinct_ranks = len(rank_totals)
    total_cards = sum(e["count"] for e in valid)
    raw_mean = total_cards / distinct_ranks
    face_pool = low_pool = 0
    face_present = low_present = 0
    for rk, total in rank_totals.items():
        if rk in _FACE_RANKS:
            face_pool += total
            face_present += 1
        elif 2 <= _RANK_NOMINAL[rk] <= 5:
            low_pool += total
            low_present += 1

    def round05(x):
        import math
        return math.floor(x * 20 + 0.5) / 20

    face_baseline = round05(raw_mean * face_present)
    low_baseline = round05(raw_mean * low_present)

    def face_score(rk):
        return max(GEN_FLOOR, W_GEN * 1.1 * max(0.0, face_pool - face_baseline)) if rk in _FACE_RANKS else 0.0

    def low_score(rk):
        return max(GEN_FLOOR, W_GEN * max(0.0, low_pool - low_baseline)) if 2 <= _RANK_NOMINAL[rk] <= 5 else 0.0

    ranks_in_order = sorted(ORDER_RANK_INDEX, key=ORDER_RANK_INDEX.get)

    def prev_rank(rk):
        i = ORDER_RANK_INDEX[rk]
        return ranks_in_order[-1] if i == 1 else ranks_in_order[i - 2]

    def seal_w(s):
        return {"Red": 1.2, "Purple": 0.15, "Gold": 0.30, "Blue": 0.05}.get(s or "", 0.0)

    def edition_w(ed):
        return {"polychrome": 1.05, "glass": 0.95, "holo": 0.50, "foil": 0.15}.get(ed or "", 0.0)

    def enh_w(center):
        return {"m_glass": 0.95, "m_lucky": 0.45, "m_steel": 0.15, "m_wild": 0.15, "m_bonus": 0.10,
                "m_mult": 0.10, "m_gold": 0.05}.get(center, 0.0)

    for e in valid:
        rk, suit, own = e["value"], e["suit"], e["count"]
        wild_elsewhere = wild_by_rank[rk] - e["wild"]
        effective = own + wild_elsewhere
        fs, ls = face_score(rk), low_score(rk)
        seal_score = 0.0
        edition_score = 0.0
        for pr in e["props"]:
            seal_score = seal_score + seal_w(pr[3])
            edition_score = edition_score + edition_w(pr[4]) + enh_w(pr[2])
        if effective >= TARGET:
            e["tier"] = 1
            e["score"] = (W_COUNT_A * effective) + fs + ls + ((seal_score + edition_score) * W_EDITION_A)
        else:
            e["tier"] = 0
            needed = TARGET - effective
            main_hit = W_MAIN * effective
            convertible = rank_totals[rk] - own - wild_elsewhere
            off_hit = W_OFF * min(3, max(0.0, convertible), needed)
            pr_rk = prev_rank(rk)
            nb = entries.get(pr_rk + "_" + suit)
            physical = nb["count"] if nb else 0
            nb_wild = nb["wild"] if nb else 0
            neighbor = physical + (wild_by_rank.get(pr_rk, 0) - nb_wild)
            strength = W_STR * min(2, neighbor, needed)
            e["score"] = main_hit + off_hit + strength + fs + ls + ((seal_score + edition_score) * W_EDITION_B)
    valid.sort(key=lambda e: (-e["tier"], -e["score"], -ORDER_RANK_INDEX[e["value"]], -ORDER_SUIT_INDEX[e["suit"]]))
    total_weight = sum(e["count"] for e in valid)
    if total_weight <= 0:
        return None
    raw = state.rng.pseudorandom(Keys.idol(ante))
    threshold = 0.0
    for e in valid:
        threshold = threshold + (e["count"] / total_weight)
        if raw < threshold:
            return e["value"], e["suit"]
    return None


def _order_mail(state: RunState, cards: Sequence, ante: int):
    """The Order's ``reset_mail_rank`` (TheOrder.lua:348-415): a count-weighted walk over the
    deck's ranks with ``pseudorandom('mail'..ante)``.  The mod means to sort the ranks by
    count descending then rank order, but the ``count`` field it sorts on is never
    incremented (the real counts live in its ``count_map``), so the sort is by rank order
    only -- reproduced as is.  Returns the rank value ('Ace', '10', ...) or None."""
    counts: dict = {}
    order: list = []
    for c in cards:
        pr = card_props(c)
        if pr[2] == "m_stone":
            continue
        v = _RANK_VALUE[pr[1]]
        if v not in counts:
            counts[v] = 0
            order.append(v)
        counts[v] += 1
    if not order:
        return None
    order.sort(key=lambda v: ORDER_RANK_INDEX[v])      # NOT by count: see the docstring
    total = sum(counts[v] for v in order)
    raw = state.rng.pseudorandom(Keys.mail(ante))
    threshold = 0.0
    for v in order:
        threshold = threshold + (counts[v] / total)
        if raw < threshold:
            return v
    return None


def reset_round_picks(state: RunState, cards: Sequence, previous_ancient: Optional[str] = None) -> dict:
    """``reset_idol_card`` / ``reset_mail_rank`` / ``reset_ancient_card`` / ``reset_castle_card``
    (common_events.lua:2271-2324) at run start and after every round, keys
    ``idol/mail/anc/cas..ante``.  ``cards`` = ``G.playing_cards`` in sort_id order (keys or
    engine Cards; stone cards are skipped for idol/mail/castle).

    Returns ``{"idol": (rank_value, suit), "mail": rank_value, "ancient": suit, "castle": (rank_value, suit),
    "idol_card": item, "mail_card": item, "castle_card": item}`` -- the ``*_card`` entries are
    the list items the draw landed on (None under The Order for idol/mail, which pick by
    rank/suit, and when the deck has no eligible card).

    Under The Order the mod replaces idol and mail outright (``_order_idol``/``_order_mail``,
    same keys) and ``reset_castle_card``'s ``pseudorandom_element`` over Card objects goes
    through the mod's override (``order_pick``); ancient draws suit strings -> vanilla."""
    ante = state.ante
    non_stone = [c for c in cards if card_props(c)[2] != "m_stone"]
    out: dict = {"idol": None, "mail": None, "ancient": None, "castle": None,
                 "idol_card": None, "mail_card": None, "castle_card": None}
    if _is_order(state):
        out["idol"] = _order_idol(state, non_stone, ante)
        out["mail"] = _order_mail(state, non_stone, ante)
    elif non_stone:
        c, _ = state.rng.pseudorandom_element(non_stone, Keys.idol(ante))
        pr = card_props(c)
        out["idol"], out["idol_card"] = (_RANK_VALUE[pr[1]], pr[0]), c
        c, _ = state.rng.pseudorandom_element(non_stone, Keys.mail(ante))
        out["mail"], out["mail_card"] = _RANK_VALUE[card_props(c)[1]], c
    suits = [s for s in ("Spades", "Hearts", "Clubs", "Diamonds") if s != previous_ancient]
    out["ancient"], _ = state.rng.pseudorandom_element(suits, Keys.ancient(ante))
    if non_stone:
        if _is_order(state):
            c, _ = order_pick(state, non_stone, Keys.castle(ante))
        else:
            c, _ = state.rng.pseudorandom_element(non_stone, Keys.castle(ante))
        pr = card_props(c)
        out["castle"], out["castle_card"] = (_RANK_VALUE[pr[1]], pr[0]), c
    return out


def start_run(state: RunState, deck_key: str = "b_red") -> RunStart:
    """Everything ``Game:start_run`` draws before the first blind select (game.lua:2018-2447):

        1. Back:apply_to_run (deck vouchers / consumables / rates)         -- no RNG
        2. pseudorandom.seed = seed; hashed_seed = pseudohash(seed)        -- (PseudoRandom ctor)
        3. blind_choices.Boss = get_new_boss()                             -- 'boss'
        4. current_round.voucher = get_next_voucher_key()                  -- 'Voucher1'
        5. blind_tags.Small = get_next_tag_key()                           -- 'Tag1'
        6. blind_tags.Big   = get_next_tag_key()                           -- 'Tag1'
        7. deck creation (Erratic: 52 x 'erratic')
        8. deck:shuffle()                                                  -- 'shuffle'
        9. reset_idol_card / reset_mail_rank / reset_ancient_card / reset_castle_card
           ('idol1' / 'mail1' / 'anc1' / 'cas1')
       10. (blind select UI) orbital_choices per blind type                -- 'orbital' x3
    Steps 9-10 only feed their own keys; they are included so those streams start at the
    right offset if the engine models those jokers/tags.
    """
    apply_deck(state, deck_key)
    eff = DECK_EFFECTS[deck_key]
    boss = next_boss(state)
    state.boss_blind = boss
    voucher = next_voucher(state)
    state.current_round_voucher = voucher
    tag_small = next_tag(state)
    tag_big = next_tag(state)
    state.blind_tags = {"Small": tag_small, "Big": tag_big}
    deck = build_starting_deck(state, erratic=eff.get("erratic", False), no_faces=eff.get("no_faces", False),
                               checkered=eff.get("checkered", False))
    shuffled = shuffle_deck(state, deck, Keys.SHUFFLE)
    # reset_idol_card etc. (common_events.lua:2271-2324): pseudorandom_element over G.playing_cards
    # (sorted by sort_id -> creation order), excluding Stone cards (none at start).
    picks = reset_round_picks(state, deck)
    idol = picks["idol"] and _card_key(*picks["idol"])
    castle = picks["castle"] and _card_key(*picks["castle"])
    # vanilla: the key of the card the 'mail' draw landed on; The Order picks a rank only
    # (suit placeholder 'S' -- consumers read the rank char, [2:]).
    mail = picks["mail_card"] if picks["mail_card"] is not None else (picks["mail"] and "S_" + _RANK_CHAR[picks["mail"]])
    return RunStart(seed=state.seed, boss=boss, voucher=voucher, tag_small=tag_small, tag_big=tag_big,
                    deck=shuffled, idol=idol, mail=mail, ancient_suit=picks["ancient"], castle=castle)


_RANK_CHAR = {v: k for k, v in _RANK_ID.items()}            # 10 -> 'T'
_RANK_CHAR = {_RANK_VALUE[i]: _RANK_CHAR[i] for i in _RANK_VALUE}   # 'Ace' -> 'A', '10' -> 'T'


def _card_key(rank_value: str, suit_name: str) -> str:
    """('Ace', 'Spades') -> 'S_A' (the P_CARDS key / ``build_starting_deck`` convention)."""
    return suit_name[0] + "_" + _RANK_CHAR[rank_value]


def defeat_boss(state: RunState) -> dict:
    """The ante transition, in Lua event order:

      end_round (state_events.lua:248-263): ``ease_ante(1)`` runs FIRST, then
      ``current_round.voucher = get_next_voucher_key()``  -> key 'Voucher'..(ante+1)
      ``reset_idol_card()`` ... (new ante keys)
      cash_out (button_callbacks.lua:2947-2953): ``blind_tags.Small/Big = get_next_tag_key()`` x2
      ``reset_blinds()`` -> ``get_new_boss()`` for the new ante.

    Returns the new ante's boss/voucher/tags.  Deck reshuffle keys per round are
    'cashout'..ante (on Cash Out) and 'nr'..ante (on blind select) -- see shuffle_deck.
    """
    state.ante += 1
    voucher = next_voucher(state)
    state.current_round_voucher = voucher
    tag_small = next_tag(state)
    tag_big = next_tag(state)
    state.blind_tags = {"Small": tag_small, "Big": tag_big}
    boss = next_boss(state)
    state.boss_blind = boss
    return {"ante": state.ante, "voucher": voucher, "tag_small": tag_small, "tag_big": tag_big, "boss": boss}


def reroll_boss(state: RunState) -> str:
    """Director's Cut / Retcon / Boss Tag (button_callbacks.lua:2800-2848): another
    ``get_new_boss()`` on the same 'boss' stream; the replaced boss keeps its +1 in
    ``bosses_used`` (the Lua never decrements)."""
    boss = next_boss(state)
    state.boss_blind = boss
    return boss


# ----------------------------------------------------------------------------------------
# Consumable- and joker-created cards (card.lua Card:use_consumeable / calculate_joker, tag.lua)
# ----------------------------------------------------------------------------------------

# name -> (create_card kwargs).  All create into owned areas (no sticker rolls) except where noted.
CREATE_SPECS = {
    # consumables (card.lua:1373-1460)
    "judgement":      dict(_type="Joker", area="jokers", key_append="jud"),                          # rarity roll 'rarity<ante>jud'
    "soul":           dict(_type="Joker", area="jokers", legendary=True, key_append="sou"),          # pool 'Joker4' (no ante); still rolls 'rarity<ante>sou'
    "wraith":         dict(_type="Joker", area="jokers", rarity=0.99, key_append="wra"),             # Rare
    "emperor":        dict(_type="Tarot", area="consumables", key_append="emp"),                     # x min(2, free slots)
    "high_priestess": dict(_type="Planet", area="consumables", key_append="pri"),                    # x min(2, free slots)
    # jokers (card.lua:2260, 2343, 2535, 2551, 2611, 3115, 3750, 3774, 3794)
    "8_ball":         dict(_type="Tarot", area="consumables", key_append="8ba"),
    "purple_seal":    dict(_type="Tarot", area="consumables", key_append="8ba"),                     # same key as 8 Ball!
    "hallucination":  dict(_type="Tarot", area="consumables", key_append="hal"),                     # after 'halu<ante>' < normal/2
    "riff_raff":      dict(_type="Joker", area="jokers", rarity=0, key_append="rif"),                # Common, x up to 2
    "cartomancer":    dict(_type="Tarot", area="consumables", key_append="car"),
    "sixth_sense":    dict(_type="Spectral", area="consumables", key_append="sixth"),
    "vagabond":       dict(_type="Tarot", area="consumables", key_append="vag"),
    "superposition":  dict(_type="Tarot", area="consumables", key_append="sup"),
    "seance":         dict(_type="Spectral", area="consumables", key_append="sea"),
    # tags (tag.lua:138, 356, 370)
    "top_up_tag":     dict(_type="Joker", area="jokers", rarity=0, key_append="top"),                # Common, x2 (slots permitting)
    "rare_tag":       dict(_type="Joker", area="shop", rarity=1, key_append="rta"),                  # Rare, into G.shop_jokers
    "uncommon_tag":   dict(_type="Joker", area="shop", rarity=0.9, key_append="uta"),                # Uncommon, into G.shop_jokers
}


def create_from_spec(state: RunState, name: str) -> CardGen:
    """``create_card`` with the exact arguments the named consumable / joker / tag uses."""
    return create_card(state, **CREATE_SPECS[name])


def fool(state: RunState) -> Optional[CardGen]:
    """The Fool: ``create_card('Tarot_Planet', G.consumeables, nil,nil,nil,nil, G.GAME.last_tarot_planet, 'fool')``
    -- forced key, no draw (card.lua:1377)."""
    if not state.last_tarot_planet:
        return None
    return create_card(state, "Tarot_Planet", area="consumables", forced_key=state.last_tarot_planet, key_append="fool")


def blue_seal(state: RunState, last_hand_played: str) -> CardGen:
    """Blue seal (card.lua:1040-1060): forced planet of ``last_hand_played``, append 'blusl'."""
    planet = None
    for pl in P.PLANETS:
        if pl["hand_type"] == last_hand_played:
            planet = pl["key"]
    return create_card(state, "Planet", area="consumables", forced_key=planet, key_append="blusl")


def prob_roll(state: RunState, key: str, odds: float) -> bool:
    """The joker probability idiom: ``pseudorandom(key) < G.GAME.probabilities.normal / odds``.
    Oops! All 6s doubles ``probabilities.normal`` (card.lua:606-610), so the threshold scales
    linearly and can exceed 1 (always true).  Each key is its own stream."""
    return state.rng.pseudorandom(key) < state.probabilities_normal / odds


# (name, key, odds) of every ``pseudorandom(key) < normal/odds`` site in 1.0.1o
PROBABILITY_ROLLS = [
    ("Lucky Card mult",          "lucky_mult",        5),      # card.lua:988
    ("Lucky Card money",         "lucky_money",       15),     # card.lua:1076
    ("Glass Card shatter",       "glass",             4),      # state_events.lua:961
    ("Wheel of Fortune",         "wheel_of_fortune",  4),      # card.lua:1470 (then 2 more draws on the same key)
    ("Hallucination",            "halu<ante>",        2),      # card.lua:2337
    ("Gros Michel",              "gros_michel",       6),      # card.lua:3020
    ("Cavendish",                "cavendish",         1000),   # card.lua:3020
    ("8 Ball",                   "8ball",             4),      # card.lua:3107
    ("Business Card",            "business",          2),      # card.lua:3177
    ("Bloodstone",               "bloodstone",        2),      # card.lua:3249
    ("Reserved Parking",         "parking",           2),      # card.lua:3304
    ("Space Joker",              "space",             4),      # card.lua:3420
    ("The Wheel (boss) flip",    "wheel",             7),      # blind.lua:608  (pseudorandom(pseudoseed('wheel')))
]


def aura(state: RunState) -> str:
    """Aura: ``poll_edition('aura', nil, true, true)`` -> polychrome > 0.85, holo > 0.5, else foil."""
    return poll_edition(state, "aura", None, True, True)  # type: ignore[return-value]


def pick_joker(state: RunState, jokers: Sequence[str], key: str) -> int:
    """``pseudorandom_element(<joker list in sort_id order>, pseudoseed(key))`` -> index0.
    Under The Order the mod's override ranks the jokers instead (``order_pick``; same
    one-step consumption of ``key``)."""
    lst = list(jokers)
    if _is_order(state):
        _, idx = order_pick(state, lst, key, jokers=True)
        return idx
    _, idx = state.rng.pseudorandom_element(lst, key)
    return idx


def wheel_of_fortune(state: RunState, eligible: Sequence[str]):
    """card.lua:1466-1480: ``pseudorandom('wheel_of_fortune') < normal/4`` then
    ``pseudorandom_element(eligible_strength_jokers, pseudoseed('wheel_of_fortune'))`` then
    ``poll_edition('wheel_of_fortune', nil, true, true)`` -- three draws on ONE key.
    Returns (index0, edition) or None (Nope!)."""
    if not prob_roll(state, "wheel_of_fortune", 4):
        return None
    idx = pick_joker(state, eligible, "wheel_of_fortune")
    return idx, poll_edition(state, "wheel_of_fortune", None, True, True)


def ectoplasm(state: RunState, eligible_editionless: Sequence[str]) -> int:
    return pick_joker(state, eligible_editionless, "ectoplasm")


def hex_(state: RunState, eligible_editionless: Sequence[str]) -> int:
    return pick_joker(state, eligible_editionless, "hex")


def ankh(state: RunState, owned_jokers: Sequence[str]) -> int:
    return pick_joker(state, owned_jokers, "ankh_choice")


def sigil(state: RunState) -> str:
    s, _ = state.rng.pseudorandom_element(["S", "H", "D", "C"], "sigil")
    return s


def ouija(state: RunState) -> str:
    r, _ = state.rng.pseudorandom_element(["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"], "ouija")
    return r


def spectral_create_cards(state: RunState, name: str, hand_size: int) -> dict:
    """Familiar / Grim / Incantation (card.lua:1292-1338): one ``'random_destroy'`` pick from
    the hand (index), then per created card: rank/suit draws on ``'<name>_create'`` (Familiar
    and Incantation draw rank THEN suit on the same key; Grim only suit), then the
    enhancement from the Enhanced pool minus Stone via ``'spe_card'``."""
    n = {"familiar": 3, "grim": 2, "incantation": 4}[name]
    _, destroy_idx = state.rng.pseudorandom_element(list(range(hand_size)), "random_destroy")
    created = []
    for _ in range(n):
        if name == "familiar":
            rank, _ = state.rng.pseudorandom_element(["J", "Q", "K"], "familiar_create")
            suit, _ = state.rng.pseudorandom_element(["S", "H", "D", "C"], "familiar_create")
        elif name == "grim":
            rank = "A"
            suit, _ = state.rng.pseudorandom_element(["S", "H", "D", "C"], "grim_create")
        else:
            rank, _ = state.rng.pseudorandom_element(["2", "3", "4", "5", "6", "7", "8", "9", "T"], "incantation_create")
            suit, _ = state.rng.pseudorandom_element(["S", "H", "D", "C"], "incantation_create")
        enh, _ = state.rng.pseudorandom_element(P.ENHANCEMENTS_NO_STONE, "spe_card")
        created.append((suit + "_" + rank, enh))
    return {"destroy_index": destroy_idx, "created": created}


def immolate(state: RunState, hand_in_sort_id_order: list) -> list:
    """card.lua:1340-1345: copy of the hand sorted by playing_card id, ``pseudoshuffle(...,
    pseudoseed('immolate'))`` (which re-sorts by sort_id first), destroy the first 5."""
    lst = list(hand_in_sort_id_order)
    if _is_order(state):   # the mod's pseudoshuffle override (playing cards): value-ranked
        return order_shuffle(state, lst, "immolate")[:5]
    state.rng.pseudoshuffle(lst, "immolate")
    return lst[:5]


def certificate(state: RunState):
    """card.lua:2463-2474: front from ``'cert_fr'`` over G.P_CARDS, seal from ``'certsl'``."""
    front, _ = state.rng.pseudorandom_element(P.PLAYING_CARD_KEYS, "cert_fr")
    return front, poll_seal(state, "certsl")


def marble_joker(state: RunState) -> str:
    front, _ = state.rng.pseudorandom_element(P.PLAYING_CARD_KEYS, "marb_fr")
    return front


def misprint(state: RunState) -> int:
    """``pseudorandom('misprint', 0, 23)`` -> integer via math.random(min, max)."""
    return state.rng.pseudorandom("misprint", 0, 23)


# ----------------------------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------------------------

def _demo(seed: str = "EXAMPLE1", antes: int = 2) -> None:
    st = RunState(seed)
    rs = start_run(st)
    print("seed %s  hashed_seed=%.13f" % (seed, st.rng.hashed_seed))
    print("run start: boss=%s voucher=%s tags=%s/%s" % (rs.boss, rs.voucher, rs.tag_small, rs.tag_big))
    print("  deck top 8 (drawn first):", rs.deck[-8:][::-1])
    print("  idol=%s mail=%s ancient=%s castle=%s" % (rs.idol, rs.mail, rs.ancient_suit, rs.castle))
    for a in range(1, antes + 1):
        if a > 1:
            info = defeat_boss(st)
            print("ante %d: boss=%s voucher=%s tags=%s/%s" % (a, info["boss"], info["voucher"], info["tag_small"], info["tag_big"]))
        for blind in ("Small", "Big", "Boss"):
            st.new_round()
            shop = generate_shop(st)
            print("ante %d after %s blind -- shop:" % (a, blind))
            print(shop.describe())
            if blind == "Small":
                reroll_shop(st, shop)
                print("  after reroll:")
                print(shop.describe())
                for pk in shop.boosters:
                    if pk:
                        cards = open_pack(st, pk)
                        print("  open %s -> %s" % (pk, "; ".join(c.short() for c in cards)))
                        st.release_pack(cards)
            st.release_shop(shop)


if __name__ == "__main__":  # pragma: no cover
    import sys
    _demo(sys.argv[1] if len(sys.argv) > 1 else "EXAMPLE1", int(sys.argv[2]) if len(sys.argv) > 2 else 2)
