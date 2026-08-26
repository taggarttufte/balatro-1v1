"""
tags.py -- Balatro 1.0.1o skip-blind Tags: the EFFECTS of all 24 tags, ported from
``_reference/balatro_src/tag.lua`` (``Tag:apply_to_run``) and the call sites that drive it.

Scope
-----
* The *draw* of a tag (``get_next_tag_key``, key ``'Tag'..ante``) is Agent C's
  ``rng/generate.py``.  The *orbital hand* draw (key ``'orbital'``) is
  ``generate.orbital_hand``.  Neither is imported here: this module is engine-agnostic and
  pure.  Everything it needs from the outside world comes through a :class:`TagContext`.
* :class:`TagState` mirrors ``G.GAME.tags`` (oldest first) plus the three pieces of
  ``G.GAME`` bookkeeping that only tags touch (``shop_d6ed``, ``shop_free``,
  ``round_resets.temp_handsize`` / ``temp_reroll_cost``).  The engine (W2) owns one per run
  and calls the ``on_*`` drivers at the points listed in ``TAGS_NOTES.md``.
* :func:`apply_tag` / :func:`tag_is_consumed_at` are the narrow primitives the brief asked
  for; the drivers on :class:`TagState` reproduce the Lua *loop* semantics (break-after-first
  for ``new_blind_choice`` / ``store_joker_modify``, fire-all for the rest, Double Tag
  copying, per-shop guards), which a single ``apply_tag`` call cannot.

Lua trigger names are kept verbatim as the :class:`Trigger` values so every branch can be
cross-read against ``tag.lua`` (line ranges in each docstring refer to that file unless a
different file is named).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------------------
# Triggers  (the ``_context.type`` strings of Tag:apply_to_run, tag.lua:115-468)
# ---------------------------------------------------------------------------------------

class Trigger(str, Enum):
    """One value per ``_context.type`` that ``Tag:apply_to_run`` switches on.

    The ``ON_*`` members are readability aliases (same value, same member) for the names the
    engine brief uses.
    """
    TAG_ADD = "tag_add"                      # add_tag(): another tag is being acquired   (UI_definitions.lua:1266-1268)
    IMMEDIATE = "immediate"                  # right after a skip / when blind select opens (button_callbacks.lua:2772, game.lua:3290)
    NEW_BLIND_CHOICE = "new_blind_choice"    # blind select opens / pack closed / boss rerolled (break after first)
    EVAL = "eval"                            # round-end cash-out rows                    (state_events.lua:1183-1190)
    ROUND_START_BONUS = "round_start_bonus"  # first draw of a round                      (game.lua:3214-3216)
    SHOP_START = "shop_start"                # shop UI built                              (game.lua:3093-3095)
    STORE_JOKER_CREATE = "store_joker_create"  # each shop card slot before the rate roll  (UI_definitions.lua:754-763)
    STORE_JOKER_MODIFY = "store_joker_modify"  # each created shop card (break after first)(UI_definitions.lua:758-760, 780-782)
    VOUCHER_ADD = "voucher_add"              # after boosters on a FRESH shop             (game.lua:3161-3163)
    SHOP_FINAL_PASS = "shop_final_pass"      # last step of a FRESH shop                  (game.lua:3164-3166)

    # aliases
    ON_ACQUIRE = "tag_add"
    ON_BLIND_SELECT = "new_blind_choice"
    ON_ROUND_END = "eval"
    ON_ROUND_START = "round_start_bonus"
    ON_SHOP_ENTER = "shop_start"


# ---------------------------------------------------------------------------------------
# Tag prototypes  (G.P_TAGS, game.lua; identical to rng/pools.py TAGS -- asserted by tests)
# ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class TagDef:
    key: str
    name: str
    order: int
    trigger: Trigger
    config: Dict[str, Any]


def _d(key: str, name: str, order: int, trigger: Trigger, **config: Any) -> TagDef:
    return TagDef(key, name, order, trigger, dict(config))


TAG_DEFS: Dict[str, TagDef] = {t.key: t for t in (
    _d("tag_uncommon",   "Uncommon Tag",    1,  Trigger.STORE_JOKER_CREATE),
    _d("tag_rare",       "Rare Tag",        2,  Trigger.STORE_JOKER_CREATE, odds=3),
    _d("tag_negative",   "Negative Tag",    3,  Trigger.STORE_JOKER_MODIFY, edition="negative", odds=5),
    _d("tag_foil",       "Foil Tag",        4,  Trigger.STORE_JOKER_MODIFY, edition="foil", odds=2),
    _d("tag_holo",       "Holographic Tag", 5,  Trigger.STORE_JOKER_MODIFY, edition="holo", odds=3),
    _d("tag_polychrome", "Polychrome Tag",  6,  Trigger.STORE_JOKER_MODIFY, edition="polychrome", odds=4),
    _d("tag_investment", "Investment Tag",  7,  Trigger.EVAL, dollars=25),
    _d("tag_voucher",    "Voucher Tag",     8,  Trigger.VOUCHER_ADD),
    _d("tag_boss",       "Boss Tag",        9,  Trigger.NEW_BLIND_CHOICE),
    _d("tag_standard",   "Standard Tag",    10, Trigger.NEW_BLIND_CHOICE),
    _d("tag_charm",      "Charm Tag",       11, Trigger.NEW_BLIND_CHOICE),
    _d("tag_meteor",     "Meteor Tag",      12, Trigger.NEW_BLIND_CHOICE),
    _d("tag_buffoon",    "Buffoon Tag",     13, Trigger.NEW_BLIND_CHOICE),
    _d("tag_handy",      "Handy Tag",       14, Trigger.IMMEDIATE, dollars_per_hand=1),
    _d("tag_garbage",    "Garbage Tag",     15, Trigger.IMMEDIATE, dollars_per_discard=1),
    _d("tag_ethereal",   "Ethereal Tag",    16, Trigger.NEW_BLIND_CHOICE),
    _d("tag_coupon",     "Coupon Tag",      17, Trigger.SHOP_FINAL_PASS),
    _d("tag_double",     "Double Tag",      18, Trigger.TAG_ADD),
    _d("tag_juggle",     "Juggle Tag",      19, Trigger.ROUND_START_BONUS, h_size=3),
    _d("tag_d_six",      "D6 Tag",          20, Trigger.SHOP_START),
    _d("tag_top_up",     "Top-up Tag",      21, Trigger.IMMEDIATE, spawn_jokers=2),
    _d("tag_skip",       "Skip Tag",        22, Trigger.IMMEDIATE, skip_bonus=5),
    _d("tag_orbital",    "Orbital Tag",     23, Trigger.IMMEDIATE, levels=3),
    _d("tag_economy",    "Economy Tag",     24, Trigger.IMMEDIATE, max=40),
)}

TAG_KEYS: List[str] = sorted(TAG_DEFS, key=lambda k: TAG_DEFS[k].order)

#: Booster pack a ``new_blind_choice`` pack tag opens (tag.lua:211, 226, 241, 256, 271).
#: Charm/Meteor pick ``_1``/``_2`` with the UNSEEDED ``math.random`` -- the two keys are the
#: same pack (cosmetic art variant), so ``_1`` is used deterministically.
TAG_PACKS: Dict[str, str] = {
    "tag_charm": "p_arcana_mega_1",
    "tag_meteor": "p_celestial_mega_1",
    "tag_ethereal": "p_spectral_normal_1",
    "tag_standard": "p_standard_mega_1",
    "tag_buffoon": "p_buffoon_mega_1",
}

#: Edition a ``store_joker_modify`` tag applies (tag.lua:398-442).
TAG_EDITIONS: Dict[str, str] = {
    "tag_foil": "foil",
    "tag_holo": "holo",
    "tag_polychrome": "polychrome",
    "tag_negative": "negative",
}


def tag_is_consumed_at(tag_key: str) -> Trigger:
    """The single trigger at which ``tag_key`` fires and is removed (``config.type``)."""
    return TAG_DEFS[tag_key].trigger


# ---------------------------------------------------------------------------------------
# Context: what a tag effect may read and the hooks it may call
# ---------------------------------------------------------------------------------------

class TagHookNotImplemented(NotImplementedError):
    """Raised when a tag effect needs a hook the engine's TagContext does not provide."""


_CTX_FIELDS = ("dollars", "skips", "hands_played", "unused_discards", "last_blind_was_boss")


class TagContext:
    """Everything a tag effect reads or does.  The engine subclasses this (override the
    hooks; expose the read-only fields as plain attributes or properties).

    Read-only fields (the exact ``G.GAME`` values the Lua reads):

    ``dollars``             ``G.GAME.dollars`` (may be negative with Credit Card) -- Economy.
    ``skips``               ``G.GAME.skips``.  Incremented BEFORE the tag is added on a skip
                            (button_callbacks.lua:2755), so it already counts the skip that
                            granted the tag -- Skip Tag.
    ``hands_played``        ``G.GAME.hands_played``: hands played this run (state_events.lua:523) -- Handy.
    ``unused_discards``     ``G.GAME.unused_discards``: discards left over, summed at the end of
                            every *played* round (state_events.lua:124) -- Garbage.
    ``last_blind_was_boss`` ``G.GAME.last_blind.boss``: set by ``Blind:set_blind`` (blind.lua:97-99),
                            i.e. whether the round just finished was a Boss -- Investment.

    Hooks (all default to raising :class:`TagHookNotImplemented`):  see each method.
    Hook names are deliberately engine-neutral; ``TAGS_NOTES.md`` maps them onto the engine.
    """
    dollars: int = 0
    skips: int = 0
    hands_played: int = 0
    unused_discards: int = 0
    last_blind_was_boss: bool = False

    def __init__(self, **fields: Any) -> None:
        for k, v in fields.items():
            if k not in _CTX_FIELDS:
                raise TypeError("TagContext has no field %r (fields: %s)" % (k, ", ".join(_CTX_FIELDS)))
            setattr(self, k, v)

    def _missing(self, hook: str) -> Any:
        raise TagHookNotImplemented("TagContext hook %r is required by a tag effect but not implemented" % hook)

    # -- money ---------------------------------------------------------------------------
    def add_dollars(self, amount: int, source: str) -> None:
        """``ease_dollars(amount)``.  ``source`` is the tag key (for logging/eval rows).
        Sources ``tag_skip``/``tag_garbage``/``tag_handy``/``tag_economy`` are paid at once;
        ``tag_investment`` is a round-eval row (paid at cash-out with the rest)."""
        self._missing("add_dollars")

    # -- jokers --------------------------------------------------------------------------
    def joker_slots_free(self) -> int:
        """``G.jokers.config.card_limit - #G.jokers.cards`` (re-read before EACH Top-up spawn:
        a Negative joker spawned first does not use a slot)."""
        return self._missing("joker_slots_free")

    def spawn_joker_to_slots(self, rarity: float, key_append: str) -> Any:
        """``create_card('Joker', G.jokers, nil, rarity, nil, nil, nil, key_append)`` then
        ``add_to_deck`` + emplace into the joker area.  Top-up passes ``rarity=0`` (Common)
        and ``key_append='top'``.  Returns the created joker (opaque to this module)."""
        return self._missing("spawn_joker_to_slots")

    def create_shop_joker(self, rarity: float, key_append: str) -> Any:
        """``create_card('Joker', G.shop_jokers, nil, rarity, nil, nil, nil, key_append)`` --
        the card is returned to the caller, NOT emplaced here (``store_joker_create`` hands it
        back to the shop-fill loop).  Uncommon: ``(0.9, 'uta')``; Rare: ``(1, 'rta')``."""
        return self._missing("create_shop_joker")

    def rare_joker_available(self) -> bool:
        """tag.lua:347-355: ``#G.P_JOKER_RARITY_POOLS[3] > (number of DISTINCT rare joker keys
        currently owned)``.  False -> the Rare Tag ``nope()``s (consumed, no card)."""
        return self._missing("rare_joker_available")

    def card_is_editionless_joker(self, card: Any) -> bool:
        """tag.lua:395: ``not card.edition and not card.temp_edition and card.ability.set == 'Joker'``."""
        return self._missing("card_is_editionless_joker")

    def set_card_edition(self, card: Any, edition: str) -> None:
        """``card:set_edition({<edition>=true}, true)``; edition in foil/holo/polychrome/negative."""
        self._missing("set_card_edition")

    def mark_card_couponed(self, card: Any) -> None:
        """``card.ability.couponed = true; card:set_cost()`` -> cost 0 while in shop_jokers /
        shop_booster (card.lua:383).  Sell value is unaffected (computed from the real cost)."""
        self._missing("mark_card_couponed")

    # -- hands ---------------------------------------------------------------------------
    def level_up_hand(self, hand: str, levels: int) -> None:
        """``level_up_hand(tag, hand, nil, levels)`` (Orbital: +3 levels to one hand)."""
        self._missing("level_up_hand")

    def choose_orbital_hand(self, blind_type: Optional[str]) -> str:
        """``G.GAME.orbital_choices[ante][blind_type]`` (UI_definitions.lua:1506-1516): drawn
        ONCE per ante per blind type with ``pseudoseed('orbital')`` from the visible hands in
        ``pairs(G.GAME.hands)`` order -- see ``rng.generate.orbital_hand``.  Only called when
        :meth:`TagState.acquire` is given no ``orbital_hand`` for an Orbital Tag."""
        return self._missing("choose_orbital_hand")

    def change_hand_size(self, delta: int) -> None:
        """``G.hand:change_size(delta)`` (Juggle +3 at round start, reverted at round end)."""
        self._missing("change_hand_size")

    # -- blind select --------------------------------------------------------------------
    def open_pack(self, pack_key: str) -> None:
        """Open a free booster (``card.cost = 0; card.from_tag = true; G.FUNCS.use_card``) from
        the blind-select screen.  The pack is an INTERRUPT: the engine must resolve it and then
        call :meth:`TagState.on_new_blind_choice` again (button_callbacks.lua:2617-2619)."""
        self._missing("open_pack")

    def reroll_boss(self) -> None:
        """``G.from_boss_tag = true; G.FUNCS.reroll_boss()`` (button_callbacks.lua:2800-2804):
        no $10 charge, but it DOES set ``round_resets.boss_rerolled`` (burns this ante's
        Director's Cut reroll).  Boss key: ``get_new_boss`` / ``rng.generate`` boss draw."""
        self._missing("reroll_boss")

    # -- shop ----------------------------------------------------------------------------
    def add_shop_voucher(self) -> None:
        """tag.lua:305-314: ``get_next_voucher_key(true)`` (key ``'Voucher_fromtag'``, see
        ``rng.generate.next_voucher(state, from_tag=True)``), ``shop_vouchers.card_limit += 1``,
        emplace the voucher card.  Not free."""
        self._missing("add_shop_voucher")

    def set_temp_reroll_cost(self, cost: int) -> None:
        """tag.lua:386-387: ``round_resets.temp_reroll_cost = cost; calculate_reroll_cost(true)``
        -> this shop's reroll cost = ``cost + reroll_cost_increase`` (i.e. $0, $1, $2, ...)."""
        self._missing("set_temp_reroll_cost")

    def clear_temp_reroll_cost(self) -> None:
        """state_events.lua:271: ``temp_reroll_cost = nil; calculate_reroll_cost(true)``."""
        self._missing("clear_temp_reroll_cost")

    def make_shop_free(self) -> None:
        """tag.lua:451-460: every card currently in ``shop_jokers`` and ``shop_booster`` gets
        ``couponed`` (cost 0).  Vouchers stay priced; rerolled cards are NOT free."""
        self._missing("make_shop_free")


# ---------------------------------------------------------------------------------------
# Live tags and results
# ---------------------------------------------------------------------------------------

@dataclass
class TagInstance:
    """One entry of ``G.GAME.tags`` (tag.lua:5-22).  ``orbital_hand`` is ``ability.orbital_hand``
    (only meaningful for ``tag_orbital``; copied across by Double Tag, tag.lua:324-328)."""
    key: str
    orbital_hand: Optional[str] = None
    triggered: bool = False
    uid: int = 0

    @property
    def defn(self) -> TagDef:
        return TAG_DEFS[self.key]

    @property
    def name(self) -> str:
        return self.defn.name

    @property
    def trigger(self) -> Trigger:
        return self.defn.trigger

    @property
    def config(self) -> Dict[str, Any]:
        return self.defn.config


@dataclass
class TagEvent:
    """Record of one tag firing (for logs / tests)."""
    key: str
    trigger: Trigger
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TagResult:
    consumed: bool
    value: Any = None
    event: Optional[TagEvent] = None


@dataclass
class BlindChoiceOutcome:
    """Result of :meth:`TagState.on_new_blind_choice` / :meth:`TagState.on_blind_select`.
    ``pending_pack`` is the booster key the engine must open (an interrupt) before calling
    ``on_new_blind_choice`` again; ``None`` when no pack tag fired."""
    events: List[TagEvent] = field(default_factory=list)
    pending_pack: Optional[str] = None


# ---------------------------------------------------------------------------------------
# Core: one tag x one trigger   (Tag:apply_to_run, tag.lua:115-468)
# ---------------------------------------------------------------------------------------

def _apply(tag: TagInstance, trigger: Trigger, ctx: TagContext, *,
           added: Optional[TagInstance] = None, card: Any = None,
           shop_flags: Optional[Dict[str, bool]] = None) -> TagResult:
    """Port of ``Tag:apply_to_run(_context)`` for a single tag.  Sets ``tag.triggered`` exactly
    where the Lua does.  ``added`` = ``_context.tag`` (TAG_ADD), ``card`` = ``_context.card``
    (STORE_JOKER_MODIFY), ``shop_flags`` = ``{'shop_d6ed':..,'shop_free':..}`` (G.GAME)."""
    trigger = Trigger(trigger)
    if tag.triggered or tag.trigger is not trigger:          # tag.lua:116
        return TagResult(False)
    key = tag.key
    cfg = tag.config
    flags = shop_flags if shop_flags is not None else {}

    def fired(value: Any = None, **detail: Any) -> TagResult:
        tag.triggered = True
        return TagResult(True, value, TagEvent(key, trigger, detail))

    # -- 'eval' (tag.lua:117-130) -------------------------------------------------------
    if trigger is Trigger.EVAL:
        if key == "tag_investment" and ctx.last_blind_was_boss:
            ctx.add_dollars(cfg["dollars"], key)
            return fired(cfg["dollars"], dollars=cfg["dollars"])
        return TagResult(False)

    # -- 'immediate' (tag.lua:131-205) --------------------------------------------------
    if trigger is Trigger.IMMEDIATE:
        if key == "tag_top_up":                                # tag.lua:134-148
            spawned = []
            for _ in range(cfg["spawn_jokers"]):
                if ctx.joker_slots_free() > 0:
                    spawned.append(ctx.spawn_joker_to_slots(0, "top"))
            return fired(spawned, spawned=len(spawned))
        if key == "tag_skip":                                  # tag.lua:149-157
            amount = (ctx.skips or 0) * cfg["skip_bonus"]
            ctx.add_dollars(amount, key)
            return fired(amount, dollars=amount)
        if key == "tag_garbage":                               # tag.lua:158-166
            amount = (ctx.unused_discards or 0) * cfg["dollars_per_discard"]
            ctx.add_dollars(amount, key)
            return fired(amount, dollars=amount)
        if key == "tag_handy":                                 # tag.lua:167-175
            amount = (ctx.hands_played or 0) * cfg["dollars_per_hand"]
            ctx.add_dollars(amount, key)
            return fired(amount, dollars=amount)
        if key == "tag_economy":                               # tag.lua:176-190
            amount = min(cfg["max"], max(0, ctx.dollars))
            ctx.add_dollars(amount, key)
            return fired(amount, dollars=amount)
        if key == "tag_orbital":                               # tag.lua:191-205
            if tag.orbital_hand is None:
                raise ValueError("Orbital Tag has no orbital_hand; pass one to TagState.acquire "
                                 "or implement TagContext.choose_orbital_hand")
            ctx.level_up_hand(tag.orbital_hand, cfg["levels"])
            return fired(tag.orbital_hand, hand=tag.orbital_hand, levels=cfg["levels"])
        return TagResult(False)

    # -- 'new_blind_choice' (tag.lua:206-301) -------------------------------------------
    if trigger is Trigger.NEW_BLIND_CHOICE:
        if key in TAG_PACKS:                                   # tag.lua:209-283
            pack = TAG_PACKS[key]
            ctx.open_pack(pack)
            return fired(pack, pack=pack)
        if key == "tag_boss":                                  # tag.lua:284-301
            ctx.reroll_boss()
            return fired("reroll_boss", reroll_boss=True)
        return TagResult(False)

    # -- 'voucher_add' (tag.lua:302-318) ------------------------------------------------
    if trigger is Trigger.VOUCHER_ADD:
        if key == "tag_voucher":
            ctx.add_shop_voucher()
            return fired("voucher")
        return TagResult(False)

    # -- 'tag_add' (tag.lua:319-333) ----------------------------------------------------
    if trigger is Trigger.TAG_ADD:
        if key == "tag_double" and added is not None and added.key != "tag_double":
            # G.orbital_hand carries the copied Orbital's hand into the new Tag (tag.lua:324-328)
            copy_ = TagInstance(added.key, orbital_hand=added.orbital_hand)
            return fired(copy_, copied=added.key)
        return TagResult(False)

    # -- 'round_start_bonus' (tag.lua:334-343) ------------------------------------------
    if trigger is Trigger.ROUND_START_BONUS:
        if key == "tag_juggle":
            ctx.change_hand_size(cfg["h_size"])
            return fired(cfg["h_size"], h_size=cfg["h_size"])
        return TagResult(False)

    # -- 'store_joker_create' (tag.lua:344-381) -----------------------------------------
    if trigger is Trigger.STORE_JOKER_CREATE:
        if key == "tag_rare":                                  # tag.lua:346-368
            if ctx.rare_joker_available():
                c = ctx.create_shop_joker(1, "rta")
                ctx.mark_card_couponed(c)
                return fired(c, card=True)
            return fired(None, card=False, noped=True)         # Tag:nope(): consumed, no card
        if key == "tag_uncommon":                              # tag.lua:369-381
            c = ctx.create_shop_joker(0.9, "uta")
            ctx.mark_card_couponed(c)
            return fired(c, card=True)
        return TagResult(False)

    # -- 'shop_start' (tag.lua:382-392) -------------------------------------------------
    if trigger is Trigger.SHOP_START:
        if key == "tag_d_six" and not flags.get("shop_d6ed"):
            flags["shop_d6ed"] = True
            ctx.set_temp_reroll_cost(0)
            return fired(0)
        return TagResult(False)

    # -- 'store_joker_modify' (tag.lua:393-446) -----------------------------------------
    if trigger is Trigger.STORE_JOKER_MODIFY:
        if key in TAG_EDITIONS and card is not None and ctx.card_is_editionless_joker(card):
            edition = TAG_EDITIONS[key]
            ctx.set_card_edition(card, edition)
            ctx.mark_card_couponed(card)
            return fired(edition, edition=edition)
        return TagResult(False)

    # -- 'shop_final_pass' (tag.lua:447-465) --------------------------------------------
    if trigger is Trigger.SHOP_FINAL_PASS:
        if key == "tag_coupon" and not flags.get("shop_free"):
            flags["shop_free"] = True
            ctx.make_shop_free()
            return fired("coupon")
        return TagResult(False)

    return TagResult(False)  # pragma: no cover


def apply_tag(tag: "str | TagInstance", trigger: Trigger, ctx: TagContext, **kw: Any) -> bool:
    """Narrow primitive: apply ``trigger`` to one tag (a key or a live :class:`TagInstance`).
    Returns True iff the tag fired (and is therefore consumed).  A bare key becomes a fresh,
    untriggered instance (``orbital_hand`` may be passed as a keyword).  Other keywords are
    forwarded to the core (``added=``, ``card=``, ``shop_flags=``).

    Prefer the :class:`TagState` drivers for anything involving more than one tag."""
    if isinstance(tag, str):
        tag = TagInstance(tag, orbital_hand=kw.pop("orbital_hand", None))
    return _apply(tag, trigger, ctx, **kw).consumed


# ---------------------------------------------------------------------------------------
# TagState: G.GAME.tags and the loops that drive it
# ---------------------------------------------------------------------------------------

@dataclass
class TagState:
    """``G.GAME.tags`` + tag-only bookkeeping.  One per run; clone with :meth:`clone`.

    Consumed tags are purged at the end of each driver call.  In the Lua, removal is an
    event queued by ``Tag:yep``/``nope`` and runs after the loop that fired it; because the
    ``triggered`` flag already blocks re-firing, purging after each loop is equivalent.
    """
    tags: List[TagInstance] = field(default_factory=list)
    shop_d6ed: bool = False        # G.GAME.shop_d6ed  (reset at cash_out, button_callbacks.lua:2933)
    shop_free: bool = False        # G.GAME.shop_free  (reset at cash_out, button_callbacks.lua:2932)
    temp_handsize: int = 0         # G.GAME.round_resets.temp_handsize (Juggle; nil == 0)
    temp_reroll_cost_set: bool = False  # G.GAME.round_resets.temp_reroll_cost ~= nil (D6)
    tag_tally: int = 0             # G.GAME.tag_tally (uids)

    # -- introspection -------------------------------------------------------------------
    def keys(self) -> List[str]:
        """Live tag keys, oldest first (what the HUD shows)."""
        return [t.key for t in self.tags]

    def __len__(self) -> int:
        return len(self.tags)

    def clone(self) -> "TagState":
        return copy.deepcopy(self)

    def _purge(self) -> None:
        self.tags = [t for t in self.tags if not t.triggered]

    def _shop_flags(self) -> Dict[str, bool]:
        return {"shop_d6ed": self.shop_d6ed, "shop_free": self.shop_free}

    def _store_flags(self, flags: Dict[str, bool]) -> None:
        self.shop_d6ed = bool(flags.get("shop_d6ed"))
        self.shop_free = bool(flags.get("shop_free"))

    def _new_instance(self, key: str, orbital_hand: Optional[str]) -> TagInstance:
        if key not in TAG_DEFS:
            raise KeyError("unknown tag key %r" % key)
        self.tag_tally += 1
        return TagInstance(key, orbital_hand=orbital_hand, uid=self.tag_tally)

    # -- acquisition (add_tag, UI_definitions.lua:1252-1272) -----------------------------
    def acquire(self, key: str, ctx: TagContext, *, blind_type: Optional[str] = None,
                orbital_hand: Optional[str] = None) -> List[TagEvent]:
        """``add_tag(Tag(key))``: run ``tag_add`` on every EXISTING tag (Double Tags fire, each
        queueing one copy), append the new tag, then add the copies (each copy goes through
        ``add_tag`` again, so a Double acquired *after* another Double does not re-trigger it).
        Final order for ``[D1, D2]`` + X is ``[X, X', X'']`` (D1/D2 consumed).

        For ``tag_orbital`` the hand is ``orbital_hand`` if given, else
        ``ctx.choose_orbital_hand(blind_type)`` (tag.lua:103-113).

        Callers: skip_blind (after ``skips += 1``), Anaglyph Deck boss win (back.lua:111-114),
        Diet Cola (card.lua:2361-2369) -- the latter two add ``tag_double``.
        Returns the events (one per Double Tag that copied)."""
        if key == "tag_orbital" and orbital_hand is None:
            orbital_hand = ctx.choose_orbital_hand(blind_type)
        new = self._new_instance(key, orbital_hand)
        events: List[TagEvent] = []
        copies: List[TagInstance] = []
        for t in list(self.tags):                                      # UI_definitions.lua:1266-1268
            res = _apply(t, Trigger.TAG_ADD, ctx, added=new)
            if res.consumed:
                events.append(res.event)
                copies.append(res.value)
        self.tags.append(new)                                          # UI_definitions.lua:1270
        self._purge()
        for c in copies:                                               # tag.lua:327 (queued add_tag)
            events.extend(self.acquire(c.key, ctx, orbital_hand=c.orbital_hand))
        return events

    # -- blind select ----------------------------------------------------------------------
    def on_immediate(self, ctx: TagContext) -> List[TagEvent]:
        """``for i=1,#tags: apply({type='immediate'})`` -- fire-all, in order."""
        events = []
        for t in list(self.tags):
            res = _apply(t, Trigger.IMMEDIATE, ctx)
            if res.consumed:
                events.append(res.event)
        self._purge()
        return events

    def on_new_blind_choice(self, ctx: TagContext) -> BlindChoiceOutcome:
        """``for i=1,#tags: if apply({type='new_blind_choice'}) then break end``.

        Exactly one tag fires per pass.  A Boss Tag's ``reroll_boss`` re-runs the pass itself
        (button_callbacks.lua:2847-2849), so this loops on; a pack tag returns with
        ``pending_pack`` set -- the engine opens that pack and, when it closes, calls this again
        (button_callbacks.lua:2617-2619)."""
        out = BlindChoiceOutcome()
        while True:
            hit: Optional[TagResult] = None
            for t in list(self.tags):
                res = _apply(t, Trigger.NEW_BLIND_CHOICE, ctx)
                if res.consumed:
                    hit = res
                    break
            self._purge()
            if hit is None:
                return out
            out.events.append(hit.event)
            if hit.value == "reroll_boss":
                continue
            out.pending_pack = hit.value
            return out

    def on_blind_select(self, ctx: TagContext) -> BlindChoiceOutcome:
        """What runs when the blind-select screen appears (game.lua:3290-3295) and right after
        ``add_tag`` in skip_blind (button_callbacks.lua:2772-2777): the ``immediate`` pass, then
        the ``new_blind_choice`` pass."""
        events = self.on_immediate(ctx)
        out = self.on_new_blind_choice(ctx)
        out.events = events + out.events
        return out

    def skip_blind(self, key: str, ctx: TagContext, *, blind_type: Optional[str] = None,
                   orbital_hand: Optional[str] = None) -> BlindChoiceOutcome:
        """Convenience = ``G.FUNCS.skip_blind`` tag part (button_callbacks.lua:2755-2777):
        ``acquire`` then ``on_blind_select``.  The engine must have incremented ``ctx.skips``
        first (Lua does ``G.GAME.skips = skips + 1`` before ``add_tag``)."""
        events = self.acquire(key, ctx, blind_type=blind_type, orbital_hand=orbital_hand)
        out = self.on_blind_select(ctx)
        out.events = events + out.events
        return out

    # -- round ---------------------------------------------------------------------------
    def on_round_start(self, ctx: TagContext) -> List[TagEvent]:
        """``update_draw_to_hand`` (game.lua:3214-3216): ``round_start_bonus`` fire-all.  Juggle
        adds to ``temp_handsize`` so :meth:`on_round_end_cleanup` can revert it."""
        events = []
        for t in list(self.tags):
            res = _apply(t, Trigger.ROUND_START_BONUS, ctx)
            if res.consumed:
                events.append(res.event)
                self.temp_handsize += int(res.value)
        self._purge()
        return events

    def on_round_eval(self, ctx: TagContext) -> int:
        """Round-eval rows (state_events.lua:1183-1190): ``eval`` fire-all.  Returns the total
        dollars added (each Investment also calls ``ctx.add_dollars(25, 'tag_investment')``).
        In the Lua the tag rows come after joker dollar bonuses and BEFORE interest, and
        interest is computed from ``G.GAME.dollars`` *before* any row is paid."""
        total = 0
        for t in list(self.tags):
            res = _apply(t, Trigger.EVAL, ctx)
            if res.consumed:
                total += int(res.value)
        self._purge()
        return total

    def on_round_end_cleanup(self, ctx: TagContext) -> None:
        """``end_round`` (state_events.lua:270-271): revert Juggle's hand size and D6's reroll
        base.  Call once per played round, after the round is won and before the shop."""
        if self.temp_handsize:
            ctx.change_hand_size(-self.temp_handsize)
            self.temp_handsize = 0
        if self.temp_reroll_cost_set:
            self.temp_reroll_cost_set = False
            ctx.clear_temp_reroll_cost()

    # -- shop ----------------------------------------------------------------------------
    def on_cash_out(self) -> None:
        """``G.FUNCS.cash_out`` (button_callbacks.lua:2932-2933): per-shop guards reset."""
        self.shop_d6ed = False
        self.shop_free = False

    def on_shop_start(self, ctx: TagContext) -> List[TagEvent]:
        """``shop_start`` fire-all (game.lua:3093-3095).  Only D6; ``shop_d6ed`` guards so a
        second D6 Tag waits for the next shop."""
        flags = self._shop_flags()
        events = []
        for t in list(self.tags):
            res = _apply(t, Trigger.SHOP_START, ctx, shop_flags=flags)
            if res.consumed:
                events.append(res.event)
                self.temp_reroll_cost_set = True
        self._store_flags(flags)
        self._purge()
        return events

    def store_joker_create(self, ctx: TagContext) -> Any:
        """``create_card_for_shop`` tag prologue (UI_definitions.lua:753-763): try every tag in
        order until one returns a card; that card then goes through ``store_joker_modify`` and
        is returned for this slot.  A Rare Tag with no rare left is consumed and the scan
        continues.  Returns ``None`` when no tag forced a card (caller does the rate roll)."""
        forced = None
        for t in list(self.tags):
            res = _apply(t, Trigger.STORE_JOKER_CREATE, ctx)
            if res.consumed and res.value is not None:
                forced = res.value
                break
        self._purge()
        if forced is not None:
            self.store_joker_modify(ctx, forced)
        return forced

    def store_joker_modify(self, ctx: TagContext, card: Any) -> bool:
        """``for kk, vv in ipairs(tags): if apply({type='store_joker_modify', card=card}) then
        break end`` -- at most ONE edition tag per card; the card must be an edition-less
        Joker.  Call for every card created for ``shop_jokers`` (fresh shop AND rerolls)."""
        applied = False
        for t in list(self.tags):
            res = _apply(t, Trigger.STORE_JOKER_MODIFY, ctx, card=card)
            if res.consumed:
                applied = True
                break
        self._purge()
        return applied

    def on_voucher_add(self, ctx: TagContext) -> List[TagEvent]:
        """``voucher_add`` fire-all (game.lua:3161-3163), FRESH shop only, after boosters."""
        events = []
        for t in list(self.tags):
            res = _apply(t, Trigger.VOUCHER_ADD, ctx)
            if res.consumed:
                events.append(res.event)
        self._purge()
        return events

    def on_shop_final_pass(self, ctx: TagContext) -> List[TagEvent]:
        """``shop_final_pass`` fire-all (game.lua:3164-3166), FRESH shop only, last.  Coupon;
        ``shop_free`` guards so a second Coupon waits for the next shop."""
        flags = self._shop_flags()
        events = []
        for t in list(self.tags):
            res = _apply(t, Trigger.SHOP_FINAL_PASS, ctx, shop_flags=flags)
            if res.consumed:
                events.append(res.event)
        self._store_flags(flags)
        self._purge()
        return events


__all__ = [
    "Trigger", "TagDef", "TAG_DEFS", "TAG_KEYS", "TAG_PACKS", "TAG_EDITIONS",
    "TagContext", "TagHookNotImplemented", "TagInstance", "TagEvent", "TagResult",
    "BlindChoiceOutcome", "TagState", "apply_tag", "tag_is_consumed_at",
]
