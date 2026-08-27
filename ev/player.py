"""
player.py — ``EVPlayer``: the analytic EV player (Phase 5 rev 2, W3).

The ``Player`` protocol used by the tournament runner, the eval harness, the replay tools
and W5's label rollouts: ``act(game) -> dict`` (always one of ``game.legal_actions()``, or
``no_action`` when there are none), ``reset()``, and ``explain(game)`` for W6's advisor.

Every state is covered:

* ``SELECTING_HAND`` — ``hand.rank_hand_actions`` (fast or full budget; hand.py).
* ``BLIND_SELECT`` — play, or skip Small/Big for a premium tag when the build clears the
  following blind with high probability (rule) / argmax V over sampled worlds (value_fn).
* ``ROUND_EVAL`` — ``advance``.
* ``SHOP`` / ``BOOSTER_OPEN`` — three tiers, in order of preference:
    1. ``stats`` (W4's ``decision_table(game) -> list[Row]``): the max ``net_ev`` row with
       positive EV, else leave / skip;
    2. ``value_fn``: argmax V over the stepped clones, chance actions (reroll, pack
       open, pack pick) averaged over ``n_worlds`` sampled worlds;
    3. built-in rules on the analytic *build proxy* (below).
* ``PVP_WAIT`` / ``GAME_OVER`` — ``no_action``.

The build proxy (the bootstrap valuation until V exists)::

    proxy(g) = P(clear next blind | deck, board) + lam_build * log1p(strength) + lam_money * money(g)

``P(clear next blind)`` = the blind model's tail from a fresh position (``hand.BlindModel``)
with the need divided by the board's exact/cheap ``ratio``; ``strength`` = the model's mean
fresh-hand score times the ratio; ``money`` = dollars + two rounds of the interest they
earn.  Jokers are bought when they raise the proxy net of their price; packs when they
are affordable after the ante's interest floor; vouchers likewise; the shop is rerolled
at most once per visit, never below the interest floor; a joker is sold only to make
room for a shelf joker that is worth more than the weakest owned one.

Determinism: with ``epsilon == 0`` every decision is a function of ``(seed, observable
state, this player's own history)`` — the world sampler is seeded from the first two
(``sampling.world_rng``) and no per-visit memory is kept (rerolls done are read off
``game.reroll_cost``).  ``epsilon > 0`` mixes in uniformly random legal actions (W5's
self-play diversity) from the same seeded stream.  The one piece of history that survives
between ``act`` calls is ``_ratio_cache``, ``hand.board_ratio``'s memo: it is per-INSTANCE
and cleared by ``reset()``, so a run cannot inherit another run's numbers even when a
worker pool reuses the process (W-FIX 2026-08-26; it used to be a module global, and
W-ENCODE-POC measured 8% of seeds changing with the worker partition because of it).

Side-effect freedom: ``act`` only reads the live game; every evaluation is on a clone.
"""
from __future__ import annotations

import math
import os
import random
import sys
from collections import Counter
from dataclasses import dataclass, replace
from typing import Callable, Optional

import _bootstrap  # noqa: F401
from _bootstrap import State
from balatro_sim.constants import (blind_base_chips, INTEREST_RATE, INTEREST_CAP,
                                   MLB_STARTING_LIVES)
from balatro_sim.consumables import PLANET_HAND
from balatro_sim.shop import effective_price

# W-SHOP: W4's decision statistics live in the sibling top-level package ``stats`` (P(hit)
# from the generator's OWN culled pools, the $ valuation tables, the interest arithmetic).
# ``ev/h2h.py`` already wires it onto sys.path the same way; doing it here means every
# entry point (gate, tournament, advisor, a bare ``import player``) gets it.  A missing or
# broken stats package degrades to the pre-W-SHOP rules rather than raising — the same
# contract ``_rank_with_stats`` already documents for a broken ``decision_table``.
_STATS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stats")
if _STATS_DIR not in sys.path:
    sys.path.insert(0, _STATS_DIR)
try:
    import hit as _hit                # stats/hit.py
    import economy as _seconomy       # stats/economy.py
    _SHOP_STATS_ERROR = None
except Exception as _e:               # pragma: no cover - stats package absent/broken
    _hit = None
    _seconomy = None
    _SHOP_STATS_ERROR = _e

import hand as _hand
from hand import HandConfig, DEFAULT_HAND_CONFIG, blind_model_for, board_ratio
from sampling import sample_world, world_rng

__all__ = ["EVPlayer", "PlayerConfig", "DEFAULT_PLAYER_CONFIG", "OLD_SHOP_CONFIG",
           "build_proxy", "PREMIUM_TAGS", "adapt_match_player", "protocol_hand_cfg",
           "PVP_PROTOCOL_HAND_CFG", "shop_arm_cfgs"]

#: Tags worth skipping a Small/Big blind for (rule tier), by the value of what they give.
PREMIUM_TAGS = frozenset({
    "tag_rare", "tag_uncommon", "tag_buffoon", "tag_foil", "tag_holo", "tag_polychrome",
    "tag_negative", "tag_top_up", "tag_charm", "tag_meteor", "tag_ethereal", "tag_orbital",
    "tag_investment", "tag_coupon",
})

#: Pack type keys in order of preference for the rule tier.
_PACK_PREF = ("buffoon", "celestial", "arcana", "standard", "spectral")

#: Vouchers the rule tier never buys.
_SKIP_VOUCHERS = frozenset({"v_blank", "v_antimatter_placeholder"})


@dataclass(frozen=True)
class PlayerConfig:
    lam_build: float = 0.30        # weight of log1p(strength) in the proxy
    lam_money: float = 0.010       # weight of one $ in the proxy (P(clear) units)
    interest_weight: float = 0.4   # extra worth of a $ that still earns interest (2 rounds)
    skip_min_p_clear: float = 0.85 # skip a blind for a premium tag only above this
    skip_tags: bool = True
    skip_min_ante: int = 2         # never skip at ante 1 (the first shops decide the run)
    pack_floor_ante_step: int = 5  # interest floor for packs/vouchers = min(25, step*ante)
    max_rerolls_per_visit: int = 1
    reroll_floor: int = 25         # never reroll below this many $ after the reroll
    buy_margin: float = 0.0        # proxy gain a purchase must exceed
    sell_margin: float = 0.05      # gain a shelf joker must have over the weakest owned one
    n_worlds: int = 4              # value_fn tier: worlds per chance action
    max_shop_candidates: int = 12  # value_fn tier: actions evaluated per shop state
    # ══════════════ W-SHOP (2026-08-26): the EV-driven shop economy, ev/SHOP_NOTES.md ══
    # Every flag below defaults to the NEW behaviour; `OLD_SHOP_CONFIG` (all three off) is
    # the pre-W-SHOP player bit-for-bit and is the h2h's comparison arm.
    reroll_ev: bool = True         # §2: uncapped, EV-driven rerolls (replaces max_rerolls_per_visit)
    pack_ev: bool = True           # §3: per-family pack EV (replaces the fixed _PACK_PREF order)
    fool_order: bool = True        # §4: The Fool sequencing among shop consumable uses
    # -- reroll stopping rule (§2) --
    reroll_worlds: int = 2         # determinized fresh shelves sampled per reroll decision
    reroll_max_depth: int = 8      # rolls looked ahead per shop visit (a hard loop bound)
    reroll_reserve: int = 0        # $ that must survive the whole roll plan (0 = spend to $0)
    reroll_guard: bool = True      # keep the OLD rule's two money guards under `reroll_ev`
                                   #   (roll only above `reroll_floor`, and only when nothing
                                   #   on the shelf is worth buying); the EV rule then drives
                                   #   the DEPTH, which is the cap the brief asks to remove
    interest_rounds: float = 2.0   # rounds of forfeited interest a spend is charged (see
                                   #   `interest_cost`: matches build_proxy's own money term)
    reroll_defer_delta: float = 0.75  # worth of the SAME queue depth one shop later (§2c)
    reroll_margin_dollars: float = 0.25  # surplus a roll plan must beat to fire (hysteresis)
    reroll_hurdle: float = 2.0     # multiplier on a roll's true cost in the stopping rule --
                                   #   the marginal $ is worth more to this player than
                                   #   `lam_money` says (SHOP_NOTES.md §2.3, swept)
    race_aggression: float = 0.8   # §2d: max uplift of the money->P(win) rate when behind
    reroll_rich_floor: float = 0.25  # floor on the "money above the interest cap is cheap"
                                     #   discount applied to a roll's cost (SHOP_NOTES §2.4)
    # -- pack EV (§3) --
    pack_low_money_floor: int = 0  # extra $ a pack must leave above the ante interest floor
    pack_take_floor: float = 0.012  # a consumable/joker pack that clears the money guard is
                                    #   never scored below this -- the EV decides WHICH pack,
                                    #   the money guard decides WHETHER.  0.012 is the middle
                                    #   of the OLD rule's own pack band (0.007 .. 0.015), kept
                                    #   so the pack-vs-reroll ordering is not silently changed
                                    #   by the new EV scale (SHOP_NOTES §3.1)
    tarot_fix_cap_dollars: float = 12.0   # cap on the measured tarot deck-fix $ value
    effect_cap_dollars: float = 20.0   # cap on any single MEASURED deck-effect $ value
    cycle_sell_margin: float = 1.0        # $ a cycle-sell must gain before it is worth a slot


DEFAULT_PLAYER_CONFIG = PlayerConfig()

#: The pre-W-SHOP shop tier, bit-for-bit (the h2h's "old" arm; see ``ev/SHOP_NOTES.md`` §0).
OLD_SHOP_CONFIG = replace(DEFAULT_PLAYER_CONFIG, reroll_ev=False, pack_ev=False,
                          fool_order=False)


def shop_arm_cfgs(arm: str, base_cfg: PlayerConfig = DEFAULT_PLAYER_CONFIG,
                  base_hand: HandConfig = DEFAULT_HAND_CONFIG):
    """``("old"|"new") -> (PlayerConfig, HandConfig)`` for the W-SHOP comparison arms.

    ``"old"`` turns every W-SHOP flag off in BOTH configs (the shop tier's three and the
    hand player's Fool sequencing), which is the pre-2026-08-26 player bit-for-bit; every
    other knob is inherited from ``base_*`` so a caller can vary an unrelated field."""
    if arm not in ("old", "new"):
        raise ValueError(f"arm must be 'old' or 'new', got {arm!r}")
    if arm == "new":
        return base_cfg, base_hand
    return (replace(base_cfg, reroll_ev=False, pack_ev=False, fool_order=False),
            replace(base_hand, fool_order=False))


# ═══════════════════════════════════════════════════════════════════ the build proxy

def _next_blind_target(game) -> float:
    """Chip target of the NEXT blind to be played from this (non-hand) state."""
    st = game.state
    ante, idx = game.ante, game.blind_idx
    if st in (State.SHOP, State.BOOSTER_OPEN, State.ROUND_EVAL):
        idx += 1
        if idx >= 3:
            idx = 0
            ante += 1
    base = blind_base_chips(ante, idx, getattr(game, "blind_scaling", 1)) * getattr(game, "ante_scaling", 1)
    if idx == 2:
        if game.mlb and ante >= getattr(game, "pvp_start_round", 2):
            return float(base)          # Nemesis: the opponent's score; use the boss-slot base
        from balatro_sim.game import BOSS_CHIP_MULT  # the boss multiplier table
        key = game.boss_blind if ante == game._boss_blind_ante else ""
        return float(base * BOSS_CHIP_MULT.get(key, 1.0))
    return float(base)


def _money_value(game, cfg: PlayerConfig) -> float:
    d = float(game.dollars)
    if getattr(game, "no_interest", False):
        return d
    interest = min(max(0.0, d) // INTEREST_RATE, float(getattr(game, "interest_cap", INTEREST_CAP)))
    return d + 2.0 * cfg.interest_weight * interest


def _auto_use_planets(clone) -> None:
    """Apply held planets on a clone (the rule tier always uses them right away, so a
    proxy measured on a clone should see the levels)."""
    changed = True
    while changed:
        changed = False
        for i, key in enumerate(list(clone.consumable_hand)):
            if key in PLANET_HAND:
                before = len(clone.consumable_hand)
                from balatro_sim.consumables import apply_planet
                if apply_planet(clone, key):
                    clone.consumable_hand.pop(i)
                    changed = True
                    break
                if len(clone.consumable_hand) == before:
                    continue


def build_proxy(game, cfg: PlayerConfig = DEFAULT_PLAYER_CONFIG,
                hcfg: HandConfig = DEFAULT_HAND_CONFIG, *, auto_planets: bool = True,
                ratio_cache: Optional[dict] = None) -> dict:
    """The analytic valuation of a non-hand state (see the module docstring).  Returns a
    dict with ``value`` and its parts.  Read-only (works on a clone when planets are held).

    ``ratio_cache``: the board-ratio memo to use.  ``EVPlayer`` passes its own instance
    dict so a run's shop/pack numbers depend on nothing but that run (hand.board_ratio has
    the full argument); omitted, ``hand._RATIO_CACHE`` is used, which is fine for a
    one-shot module-level call and NOT fine inside a reused worker."""
    g = game
    if auto_planets and any(k in PLANET_HAND for k in g.consumable_hand):
        g = game.clone()
        _auto_use_planets(g)
    model = blind_model_for(g, hcfg)
    ratio = board_ratio(g, 3, hcfg, cache=ratio_cache)
    target = _next_blind_target(g)
    hands = int(g.base_hands) + (3 if any(j.key == "j_burglar" for j in g.jokers) else 0)
    discards = int(g.base_discards)
    p_clear = model.p_clear(target / max(ratio, 1e-6), hands, discards)
    strength = model.mean0 * ratio
    money = _money_value(g, cfg)
    value = p_clear + cfg.lam_build * math.log1p(strength) + cfg.lam_money * money
    return {"value": value, "p_clear": p_clear, "strength": strength, "ratio": ratio,
            "money": money, "target": target}


# ══════════════════════════════ W-SHOP: the EV shop economy (ev/SHOP_NOTES.md) ═══════
#
# Everything below is in DOLLARS, W4's common unit (stats/STATS_NOTES.md §1).  The rules
# tier ranks in P(clear) units, and `PlayerConfig.lam_money` is this player's own $ -> P
# exchange rate (the SAME constant `build_proxy` prices a held dollar with), so a dollar
# figure enters a rules-tier row multiplied by `lam_money` and nothing else.  The one place
# the two scales must not be mixed is "is a fresh shelf better than this one", and there
# BOTH sides are priced with the cheap $ model (`hit.pool_dollar_value` and the tables
# below) — never one side with a dry run and the other with a pool proxy.

#: How many shop visits still draw from the CURRENT ante's `...sho<a>` streams, including
#: this one.  GENERATION_SPEC §8.3: "the ante-N shop queue is the sequence of values of the
#: '...sho<a>'-family streams" and a reroll advances that shared pointer.  `game.ante` in a
#: SHOP state is already the ante whose streams the shelf came from (`_end_round` bumps the
#: ante before the post-boss shop, `_advance_blind` only moves `blind_idx` on `leave_shop`),
#: so `blind_idx` alone says where on the queue this shop sits:
#:   2 -> the post-Boss shop, the FIRST of this ante's three;  0 -> after Small (second);
#:   1 -> after Big (third and LAST) — the shop right before the Boss / Nemesis.
_QUEUE_SHOPS_LEFT = {2: 3, 0: 2, 1: 1}


def shops_left_on_queue(game) -> int:
    """Shops still to come on THIS ante's shop queue, including the current one (1..3)."""
    if game.state not in (State.SHOP, State.BOOSTER_OPEN):
        return 0
    return _QUEUE_SHOPS_LEFT.get(int(getattr(game, "blind_idx", 1)), 1)


def interest_cost(game, cfg: PlayerConfig, spend: float) -> float:
    """$ of future income forfeited by spending ``spend`` now — the interest-THRESHOLD term.

    ``economy.interest_now/interest_after`` are the engine's own formula
    (``min(dollars // 5, interest_cap)``, game.py:2010), so the *threshold* structure is
    exact: this is 0 while the balance stays above the cap ($25 at cap 5) and above the
    breakpoint, and jumps by one round's dollar for every $5 tier the spend crosses.

    What this deliberately does NOT reuse is ``economy.interest_loss``'s horizon: that
    module discounts the shortfall geometrically over the shops remaining to ante 8
    (``decay=0.85``, STATS_NOTES §3), which prices a $4 pack at ante 1 at $4.55 of forfeited
    interest — more than the pack.  This player already has a documented answer to "what is
    a dollar that still earns interest worth", and it is ``build_proxy``'s
    ``money = $ + 2*interest_weight*interest`` (EXTRACT_NOTES §2 quotes the same 0.16/$
    marginal rate).  Using two different horizons inside one ranking would price a pack row
    against a joker row on different money; ``interest_rounds`` keeps them on the player's."""
    if getattr(game, "no_interest", False) or spend <= 0:
        return 0.0
    cap = float(getattr(game, "interest_cap", INTEREST_CAP))
    now = min(max(0.0, float(game.dollars)) // INTEREST_RATE, cap)
    after = min(max(0.0, float(game.dollars) - spend) // INTEREST_RATE, cap)
    return cfg.interest_rounds * cfg.interest_weight * max(0.0, now - after)


def _shop_price(game, base_cost: int) -> int:
    """``Card:set_cost`` (card.lua:370-383) for a card that is NOT on the shelf yet — the
    price a pool member would carry if a reroll put it there."""
    d = int(round(float(getattr(game, "shop_discount", 0.0)) * 100))
    return max(1, math.floor((float(base_cost) + 0.5) * (100 - d) / 100))


# ── the tarot table (SHOP_NOTES.md §3.2) ─────────────────────────────────────────────
#
# `hit.tarot_value` prices every tarot at a flat $4 and STATS_NOTES §1 flags that as a
# documented gap ("real Balatro tarot power varies a lot").  That flat number is exactly why
# the shop passed Arcana packs: a deck-fixing tarot and a dead one looked identical, so the
# only thing separating an Arcana pack from a Celestial pack was the fixed preference order.
# The table below is the same KIND of object as `hit._VOUCHER_STANDOUT` — a curated,
# documented prior, not a derivation — plus three genuinely state-dependent entries.
# Anchors: the enhancement numbers are `hit.StatsConfig.enhancement_card_value` (Steel 4.5,
# Gold 3.5, Glass/Lucky 3.0, Mult 2.5, Bonus/Wild 2.0, Stone 1.0) times the number of cards
# the engine's `apply_tarot` actually converts (2 for every enhancement tarot, 3 for the
# suit tarots, consumables.py:113-129).
_TAROT_ENH_CARDS = 2
#: tarot -> the enhancement it applies (``consumables.TAROT_ENHANCEMENT``, re-stated here so
#: this module does not import the engine's table at call time)
_TAROT_ENHANCEMENT_OF = {
    "c_magician": "Lucky", "c_empress": "Mult", "c_heirophant": "Bonus", "c_lovers": "Wild",
    "c_chariot": "Steel", "c_justice": "Glass", "c_devil": "Gold", "c_tower": "Stone",
}
#: the tarots whose effect the build proxy CANNOT see (nothing about the deck changes), so
#: they keep a curated $ prior — the same kind of object as ``hit._VOUCHER_STANDOUT``.
_TAROT_STATIC_DOLLARS = {
    "c_emperor":        6.0,    # 2 random Tarots (2 x the $4 flat base, minus slot pressure)
    "c_high_priestess": 6.0,    # 2 random Planets (2 x `planet_base_value` 6 x mean share)
    "c_judgement":     10.0,    # a random Joker — the pool's own mean hit value
    "c_wheel_of_fortune": 3.0,  # 1-in-4 for a joker edition (~$8 x 0.25, rounded up: it can
                                #   hit a Polychrome, which the flat edition table underprices)
}
#: suit tarots — value is MEASURED per deck (`EVPlayer._deck_effects`), this is the
#: floor used when the measurement is unavailable.
_TAROT_SUIT_KEYS = ("c_star", "c_moon", "c_sun", "c_world")
_TAROT_SUIT_FLOOR = 4.0


def _gk_consumable_kind(key: str) -> str:
    """``'c_sun' -> 'tarot'`` etc., via the engine's own ``CONSUMABLE_SET`` map."""
    from balatro_sim import game_keys as _gk
    return _gk.CONSUMABLE_SET.get(key, "").lower()


def tarot_dollars(game, key: str, cfg: PlayerConfig = DEFAULT_PLAYER_CONFIG,
                  effects: Optional[dict] = None) -> float:
    """$-equivalent of ONE tarot for THIS run (SHOP_NOTES.md §3.2).

    ``effects``: the per-deck MEASURED effect values (``EVPlayer._deck_effects``) —
    ``suit3`` (convert 3 cards to the modal suit), ``steel1`` (one card to Steel),
    ``destroy2``, ``rank2``, ``add_plain``.  Every tarot whose whole effect is a change to
    the DECK is priced from those, because a deck change is exactly what the build proxy
    can already evaluate; the shape WITHIN the enhancement family comes from
    ``hit.card_value``'s own enhancement table, so one measurement covers all eight.  What
    stays tabulated is what the proxy cannot see: cards created, money, and joker editions.
    A module-level caller with no measurements falls back to the table's own floors."""
    eff = effects or {}
    tbl = _hit.DEFAULT.enhancement_card_value if _hit is not None else {}
    steel = float(tbl.get("Steel", 4.5)) or 4.5

    def per_enh(enh: str, n_cards: float) -> float:
        base = eff.get("steel1")
        if base is None:
            return float(tbl.get(enh, 2.0)) * n_cards
        return base * n_cards * float(tbl.get(enh, 2.0)) / steel

    if key in _TAROT_SUIT_KEYS:
        return max(0.0, float(eff.get("suit3", _TAROT_SUIT_FLOOR)))
    enh = _TAROT_ENHANCEMENT_OF.get(key)
    if enh is not None:
        return max(0.0, per_enh(enh, _TAROT_ENH_CARDS))
    if key == "c_hanged_man":
        return max(0.0, float(eff.get("destroy2", 5.0)))
    if key == "c_strength":
        return max(0.0, float(eff.get("rank2", 3.0)))
    if key == "c_death":
        return max(0.0, float(eff.get("rank2", 3.0)))      # one card upgraded to the best one
    if key == "c_hermit":
        return float(min(int(getattr(game, "dollars", 0)), 20))       # doubles money, cap $20
    if key == "c_temperance":
        from balatro_sim.jokers.base import joker_sell_value
        return float(min(50, sum(joker_sell_value(j) for j in game.jokers)))
    if key == "c_fool":
        last = getattr(game.run_state, "last_tarot_planet", None)
        if not last or last == "c_fool":
            return 0.0                                                # unusable
        if _gk_consumable_kind(last) == "planet":
            return _hit.planet_value(game, last) if _hit is not None else 4.0
        return tarot_dollars(game, last, cfg, effects)
    if key == "c_judgement" and len(game.jokers) >= game.joker_slots:
        return 0.0                                                    # engine refuses the use
    return float(_TAROT_STATIC_DOLLARS.get(key, 4.0))


# ── Standard-pack playing cards (SHOP_NOTES.md §3.4) ────────────────────────────────
#
# `hit.pack_p_hit` returns (0, 0) for a Standard pack — a documented hole that makes the
# pack "correctly never recommended, but only because the floor is 0" (STATS_NOTES §5).
# The generator's own recipe (generate.open_pack:1283-1287) is three independent rolls:
#   Enhanced iff `stdset > 0.6`        -> P = 0.40, uniform over the 8 Enhanced centers
#   seal iff `stdseal > 1 - 0.02*10`    -> P = 0.20, then a type roll (4 seals)
#   edition `poll_edition(.., mult=2)`  -> ~8% for foil/holo/poly at doubled rates
# so the value distribution below is the recipe, priced with `hit.card_value`'s own tables
# (+ a seal term the card table has no entry for).  It is deliberately NOT a build-fit
# model: a plain card added to a 52-card deck is worth its $1 base and dilutes everything
# else, which is why a low Standard take rate is the honest outcome, not a bug.
_STD_P_ENHANCED = 0.40
_STD_P_SEAL = 0.20
_STD_P_EDITION = 0.08
_STD_SEAL_DOLLARS = 2.0     # Gold/Red/Blue/Purple, mean; `hit.card_value` prices no seal


def _standard_card_atoms(cfg: PlayerConfig, effects: Optional[dict] = None) -> list:
    """[(weight, $)] for ONE Standard-pack card.  Enhancement is the dominant term, so it
    carries the atom structure and the (small, independent) seal/edition terms enter as
    their means — the order statistics only need the spread that actually separates cards."""
    if _hit is None:
        return [(1.0, 1.0)]
    eff = effects or {}
    tbl = _hit.DEFAULT.enhancement_card_value
    steel_tbl = float(tbl.get("Steel", 4.5)) or 4.5
    steel = eff.get("steel1")
    base = eff.get("add_plain")
    if base is None:
        base = _hit.DEFAULT.card_base_value    # module-level fallback: the flat card table
    extra = _STD_P_SEAL * _STD_SEAL_DOLLARS + _STD_P_EDITION * 1.7
    enh = [(k, v) for k, v in tbl.items() if k != "None"]
    out = [(1.0 - _STD_P_ENHANCED, base + extra)]
    w = _STD_P_ENHANCED / max(1, len(enh))
    for k, v in enh:
        bump = (steel * v / steel_tbl) if steel is not None else v
        out.append((w, base + bump + extra))
    return out


# ── order statistics over a discrete value distribution ─────────────────────────────

def _levels_from_atoms(atoms: list, max_levels: int = 32) -> tuple:
    """``[(weight, value)] -> (levels, tails)``: strictly increasing POSITIVE value levels
    ``u_1 < ... < u_K`` and ``tails[k] = P(V >= u_k)`` for one draw.  Values <= 0 are
    dropped (never taking the item is always allowed), and the levels are bucketed to at
    most ``max_levels`` so the layer-cake sums below stay O(32) whatever the pool size."""
    pos = [(w, v) for w, v in atoms if v > 0.0 and w > 0.0]
    if not pos:
        return (), ()
    buckets: dict = {}
    if len({v for _, v in pos}) <= max_levels:
        for w, v in pos:                       # few enough distinct values: keep them exact
            buckets[v] = buckets.get(v, 0.0) + w
    else:
        hi = max(v for _, v in pos)
        step = hi / max_levels
        for w, v in pos:
            b = max(1, int(math.ceil(v / step))) if step > 0 else 1
            u = b * step
            buckets[u] = buckets.get(u, 0.0) + w
    levels = sorted(buckets)
    tails = []
    acc = 0.0
    for u in reversed(levels):
        acc += buckets[u]
        tails.append(acc)
    tails.reverse()
    return tuple(levels), tuple(min(1.0, t) for t in tails)


def _e_top_k(levels: tuple, tails: tuple, n: int, k: int) -> float:
    """``E[sum of the best k of n iid draws]`` (negative draws counted as 0 — a pack pick can
    always be declined).  ``= sum_levels (u_j - u_{j-1}) * sum_{i=1..k} P(Bin(n, S_j) >= i)``."""
    if k <= 0 or n <= 0:
        return 0.0
    k = min(k, n)
    total = 0.0
    prev = 0.0
    for u, s in zip(levels, tails):
        if u <= prev:
            continue
        # E[min(k, #draws >= u)] = sum_{i=1..k} P(Bin(n, s) >= i)
        q = 1.0 - s
        pmf = q ** n
        tail = 1.0 - pmf
        exp_cnt = tail
        for i in range(1, k):
            if q <= 0.0:
                pmf = 0.0
            else:
                pmf *= (n - i + 1) / i * (s / q)
            tail -= pmf
            exp_cnt += max(0.0, tail)
        total += (u - prev) * exp_cnt
        prev = u
    return total


# ═══════════════════════════════════════════════════════════════════════ the player

class EVPlayer:
    """See the module docstring.  ``budget`` = "fast" | "full" for hand decisions."""

    def __init__(self, value_fn: Optional[Callable] = None, *, stats=None, budget: str = "fast",
                 seed: int = 0, epsilon: float = 0.0, no_action: Optional[dict] = None,
                 cfg: PlayerConfig = DEFAULT_PLAYER_CONFIG, hand_cfg: HandConfig = DEFAULT_HAND_CONFIG,
                 n_worlds: Optional[int] = None, top_k: Optional[int] = None, name: str = "ev",
                 value_fn_leaf_only: bool = False):
        if budget not in ("fast", "full"):
            raise ValueError(f"budget must be 'fast' or 'full', got {budget!r}")
        self.value_fn = value_fn
        self.stats = stats
        self.budget = budget
        # W-LEAF (Phase 5 rev 2 V2 round): the brief's lever (c) is "V at the expectimax
        # LEAF only" -- but the plumbing below (built by W3/W5) argmaxes value_fn over EVERY
        # SHOP / BOOSTER_OPEN / BLIND_SELECT candidate too the moment value_fn is set, which
        # is a DIFFERENT, already-measured thing (PHASE5_V2_BRIEF section 0: "argmax-V as a
        # policy loses to the rules player 2/60"). value_fn_leaf_only=True keeps value_fn
        # wired into the full-budget hand rollout's leaf (hand.rank_hand_actions /
        # end_of_blind_value -- unaffected by this flag) while SHOP / BOOSTER_OPEN /
        # BLIND_SELECT fall through to the analytic rules tier, exactly as if value_fn were
        # None there.  Default False: every existing caller (W5 rollouts, W6 advisor,
        # tournament_v) is unchanged.
        self.value_fn_leaf_only = bool(value_fn_leaf_only)
        self.seed = int(seed)
        self.epsilon = float(epsilon)
        self.no_action = dict(no_action) if no_action is not None else {"type": "advance"}
        self.cfg = cfg
        self.hand_cfg = hand_cfg
        # the shop / pack proxies rebuild the blind model per candidate (a pick changes the
        # deck or the levels): a lighter simulation + coarser grid is plenty for a purchase
        # comparison (fix pass: the 96-sample full-grid proxies were 58% of a W5 rollout)
        self.proxy_cfg = replace(hand_cfg, model_samples=min(hand_cfg.model_samples, 48),
                                 model_atoms=min(hand_cfg.model_atoms, 4),
                                 grid_ratio=max(hand_cfg.grid_ratio, 1.20))
        self.n_worlds = n_worlds
        self.top_k = top_k
        self.name = name
        self._last_explain: list = []
        # epsilon gets its OWN sequential stream, advanced once per act() and re-seeded by
        # reset(): seeding it from the observable state wedged W5's rollouts (a no-op
        # epsilon pick left the state unchanged, so the same pick was redrawn forever).
        self._eps_rng = random.Random(f"ev-eps:{self.seed}")
        # anti-cycling guard: how often act() has seen an unchanged SHOP/BOOSTER signature
        self._sig_seen: Counter = Counter()
        # W-FIX (2026-08-26): board_ratio's memo, owned by the PLAYER rather than by the
        # hand module.  hand._board_sig deliberately omits planet levels and the exact deck
        # composition (EV_NOTES 8b item 1: a planet pick must not force a ratio recompute),
        # so a process-global dict let one run be served a number computed for another —
        # W-ENCODE-POC measured 2 of 24 seeds (8%) changing trajectory with the worker
        # partition of a reused multiprocessing pool.  One dict per player keeps the
        # within-run hit rate the fix pass was after (the shop evaluates a dozen candidates
        # per visit and the state a purchase produces hits the candidate's entry) and makes
        # a run's decisions a function of that run alone.
        self._ratio_cache: dict = {}
        # W-SHOP: two more per-PLAYER memos, for exactly the reason above (a module global
        # would make a run's shop decisions depend on what the worker played before it).
        # `_fix_cache`: the MEASURED deck-fix $ of a suit tarot, keyed by a deck signature.
        # `_atom_cache`: the fresh-shop-slot $ distribution, keyed by a shop signature.
        self._fix_cache: dict = {}
        self._atom_cache: dict = {}
        # The race read (§2d): bound by `adapt_match_player` from the live MLBMatch, and
        # left None by every solo harness — where the race term is then exactly neutral.
        self._race_p_win: Optional[float] = None
        self._race_cache: dict = {}

    # ── protocol ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        self._last_explain = []
        self._eps_rng = random.Random(f"ev-eps:{self.seed}")
        self._sig_seen = Counter()
        self._ratio_cache.clear()
        self._fix_cache.clear()
        self._atom_cache.clear()
        self._race_cache.clear()
        self._race_p_win = None

    # ── W-SHOP: the race read (§2d) ─────────────────────────────────────────

    def bind_race(self, match, player_idx: int) -> None:
        """Give the shop tier the race calculator's ``P(win)`` for this seat.

        ``ev/race.py``'s model needs BOTH sides' Nemesis-score curves and BOTH lives
        counts; a bare ``BalatroGame`` carries neither (it sees its own lives and, only
        during a Nemesis, the opponent's live score), so the number has to come from the
        match.  ``adapt_match_player`` calls this before every ``act``; anything that drives
        a lone game leaves ``_race_p_win`` None and the aggression term is 1.0 exactly.
        Cached on (ante, both lives, log length): the DP is milliseconds, a shop visit is
        a dozen decisions, and none of those inputs move inside one visit."""
        try:
            g = match.games[player_idx]
            if not getattr(g, "mlb", False):
                self._race_p_win = None
                return
            other = match.games[1 - player_idx]
            log = getattr(match, "pvp_log", ()) or ()
            key = (int(g.ante), int(g.lives), int(other.lives), len(log))
            if key not in self._race_cache:
                import race as _race
                a = _race.curve_from_history(log, player_idx, g.ante)
                b = _race.curve_from_history(log, 1 - player_idx, g.ante)
                self._race_cache[key] = float(
                    _race.p_win(a, b, g.lives, other.lives, g.ante))
            self._race_p_win = self._race_cache[key]
        except Exception:            # noqa: BLE001 — a race read must never break `act`
            self._race_p_win = None

    def _race_aggression(self, game) -> float:
        """``1 + race_aggression * pressure``: the factor by which being BEHIND raises the
        money -> P(win) exchange rate (Tagg: "$20 on a $6 reroll before the PvP is
        reasonable if you feel behind").  Exactly 1.0 in vanilla — `game.mlb` gates it.

        ``pressure`` is ``2*(0.5 - p_win)`` clipped to [0,1] when the race calculator is
        bound (so a 50/50 race is neutral and a lost-looking one is maximal), otherwise the
        observable fallback "how many of my lives are gone".  Either way it is topped up
        when the next blind is the Nemesis itself: that is the shop the ante's shop queue
        ends at, and the last chance to spend before the race is decided again."""
        cfg = self.cfg
        if not cfg.reroll_ev or not getattr(game, "mlb", False) or cfg.race_aggression <= 0:
            return 1.0
        if self._race_p_win is not None:
            pressure = min(1.0, max(0.0, 2.0 * (0.5 - self._race_p_win)))
        else:
            start = max(1, int(getattr(game, "starting_lives", 0)) or MLB_STARTING_LIVES)
            pressure = min(1.0, max(0.0, (start - int(game.lives)) / start))
        if self._next_is_nemesis(game):
            pressure = min(1.0, pressure + 0.25)
        return 1.0 + cfg.race_aggression * pressure

    @staticmethod
    def _next_is_nemesis(game) -> bool:
        if not getattr(game, "mlb", False):
            return False
        idx = int(getattr(game, "blind_idx", 0))
        ante = int(game.ante)
        if game.state in (State.SHOP, State.BOOSTER_OPEN, State.ROUND_EVAL):
            idx += 1
            if idx >= 3:
                idx, ante = 0, ante + 1
        return idx == 2 and ante >= int(getattr(game, "pvp_start_round", 2))

    #: an unchanged shop/booster signature seen this often falls back to the rules tier
    STUCK_AFTER = 3
    #: ... and seen this often forces leave_shop / skip_booster
    FORCE_LEAVE_AFTER = 6

    def act(self, game, extra_actions: Optional[list] = None) -> dict:
        """``extra_actions`` (W-PVP): MATCH-level actions the coordinator is offering that
        the game itself knows nothing about — today exactly ``{"type": "pvp_pass"}``, the
        Nemesis leader's wait (PVP_NOTES.md §4).  ``adapt_match_player`` fills it in from
        ``MLBMatch.legal_actions``; every existing caller passes nothing and is unaffected.

        The match has the last word on legality: the analysis generates the pass from the
        two scores it can see, and anything the match did not actually offer is dropped
        here rather than returned as an illegal action."""
        legal = game.legal_actions()
        allow_pass = bool(extra_actions) and any(
            a.get("type") == "pvp_pass" for a in extra_actions)
        if not legal:
            if allow_pass:
                # rare: SELECTING_HAND with an empty hand (deck-out).  The match offered a
                # wait and the game offers nothing, so waiting is the only thing to do.
                return {"type": "pvp_pass"}
            return dict(self.no_action)
        if self.epsilon > 0.0 and self._eps_rng.random() < self.epsilon:
            return dict(legal[self._eps_rng.randrange(len(legal))])
        if allow_pass and game.state == State.SELECTING_HAND:
            ranked = self._rank_hand(game, legal, explain=False, allow_pass=True)
            if ranked:
                return dict(ranked[0][0])
            return dict(legal[0])
        force_rules = False
        if game.state in (State.SHOP, State.BOOSTER_OPEN):
            # a repeated identical signature means the previous pick was a no-op (an
            # engine silent-no-op consumable, a V that prefers standing still): stop
            # repeating it (fix pass: W5 saw 40k-step shop loops under a value_fn)
            if len(self._sig_seen) > 512:
                self._sig_seen.clear()
            sig = game.state_signature()
            self._sig_seen[sig] += 1
            stuck = self._sig_seen[sig]
            force_rules = stuck >= self.STUCK_AFTER
            if stuck >= self.FORCE_LEAVE_AFTER:
                out = {"type": "leave_shop"} if game.state == State.SHOP else {"type": "skip_booster"}
                keys = {_hand._action_sort_key(a) for a in legal}
                return out if _hand._action_sort_key(out) in keys else dict(legal[0])
        ranked = self._rank(game, legal, explain=False, force_rules=force_rules)
        if ranked:
            return dict(ranked[0][0])
        return dict(legal[0])

    def explain(self, game, extra_actions: Optional[list] = None) -> list:
        """Ranked ``[(action, ev, reason)]`` for the state (W6's advisor prints this)."""
        legal = game.legal_actions()
        allow_pass = bool(extra_actions) and any(
            a.get("type") == "pvp_pass" for a in extra_actions)
        if not legal:
            return [(dict(self.no_action), 0.0, "no legal action (waiting / over)")]
        if allow_pass and game.state == State.SELECTING_HAND:
            return [(a, ev, r) for a, ev, r in
                    self._rank_hand(game, legal, explain=True, allow_pass=True)]
        return [(a, ev, r) for a, ev, r in self._rank(game, legal)]

    # ── dispatch ────────────────────────────────────────────────────────────

    def _rank(self, game, legal: list, explain: bool = True, force_rules: bool = False) -> list:
        s = game.state
        if s == State.SELECTING_HAND:
            return self._rank_hand(game, legal, explain=explain)
        if s == State.ROUND_EVAL:
            return [({"type": "advance"}, 0.0, "cash out")]
        if s == State.BLIND_SELECT:
            return self._rank_blind_select(game, legal)
        if s in (State.SHOP, State.BOOSTER_OPEN):
            if not force_rules:
                if self.stats is not None:
                    r = self._rank_with_stats(game, legal)
                    if r:
                        return r
                if self.value_fn is not None and not self.value_fn_leaf_only:
                    return self._rank_with_value(game, legal)
            if s == State.SHOP:
                return self._rank_shop_rules(game, legal)
            return self._rank_booster_rules(game, legal)
        return [(legal[0], 0.0, "fallback: first legal action")]

    # ── hands ───────────────────────────────────────────────────────────────

    def _rank_hand(self, game, legal: list, explain: bool = True,
                   allow_pass: bool = False) -> list:
        kw = dict(budget=self.budget, cfg=self.hand_cfg, legal=legal)
        if allow_pass and self.budget == "fast":
            kw["allow_pass"] = True
        if self.budget == "full":
            kw["value_fn"] = self.value_fn
            kw["rng"] = world_rng(self.seed, game, salt=0xF0)
            if self.n_worlds is not None:
                kw["n_worlds"] = self.n_worlds
            if self.top_k is not None:
                kw["top_k"] = self.top_k
        ranked = _hand.rank_hand_actions(game, **kw)
        if not explain:
            return [(a, ev, "") for a, ev in ranked]
        # W-EXTRACT: the money decomposition the advisor renders (the sandbag fixtures of
        # brief §7 read it).  One HandAnalysis for the whole ranking, not one per action.
        money: dict = {}
        try:
            an = _hand.HandAnalysis(game, self.hand_cfg, legal=legal)
            if an.extract_on:
                for a, _ in ranked:
                    if a.get("type") in ("play", "discard"):
                        d = an.extraction_ev(a)
                        if abs(d) >= 0.005:
                            money[_hand._action_sort_key(a)] = d
        except Exception:               # noqa: BLE001  (an explanation must never break act)
            money = {}
        out = []
        for a, ev in ranked:
            r = self._hand_reason(game, a, ev)
            d = money.get(_hand._action_sort_key(a))
            if d is not None:
                r += f" [extract ${d:+.2f}]"
            out.append((a, ev, r))
        return out

    @staticmethod
    def _hand_reason(game, a: dict, ev: float) -> str:
        t = a.get("type")
        if t == "play":
            cards = [game.hand[i] for i in a.get("cards", []) if i < len(game.hand)]
            try:
                ht, _ = _hand.evaluate_hand(cards, **game.hand_eval_flags())
            except Exception:       # noqa: BLE001
                ht = "?"
            return f"play {ht} {' '.join(map(str, cards))} (EV {ev:.3f})"
        if t == "discard":
            cards = [str(game.hand[i]) for i in a.get("cards", []) if i < len(game.hand)]
            keep = [str(c) for i, c in enumerate(game.hand) if i not in set(a.get("cards", []))]
            return f"discard {' '.join(cards)}, keep {' '.join(keep)} (EV {ev:.3f})"
        if t == "use_consumable":
            idx = a.get("consumable_idx", 0)
            key = game.consumable_hand[idx] if idx < len(game.consumable_hand) else "?"
            return f"use {key} on {a.get('target_cards', [])} (EV {ev:.3f})"
        if t == "pvp_pass":
            return (f"WAIT — leading {game.chips_scored} vs {game.pvp_opponent_score}, "
                    f"conserve {game.hands_left} hands (EV {ev:.3f})")
        return f"{t} (EV {ev:.3f})"

    # ── blind select ────────────────────────────────────────────────────────

    def _rank_blind_select(self, game, legal: list) -> list:
        keys = [a["type"] for a in legal]
        if self.value_fn is not None and not self.value_fn_leaf_only:
            return self._rank_with_value(game, legal)
        px = build_proxy(game, self.cfg, self.hand_cfg, ratio_cache=self._ratio_cache)
        out = [({"type": "play_blind"}, px["p_clear"],
                f"play {game.current_blind.kind} (P(clear) {px['p_clear']:.2f}, target {px['target']:.0f})")]
        if "skip_blind" in keys and self.cfg.skip_tags and game.ante >= self.cfg.skip_min_ante:
            tag = game.blind_tags.get(game.current_blind.kind, "")
            # the blind faced after the skip
            nxt_idx = game.blind_idx + 1
            nxt_target = blind_base_chips(game.ante, nxt_idx, getattr(game, "blind_scaling", 1)) * getattr(game, "ante_scaling", 1)
            if nxt_idx == 2:
                if game.mlb and game.ante >= getattr(game, "pvp_start_round", 2):
                    p_next = 0.0     # never skip into a Nemesis
                else:
                    from balatro_sim.game import BOSS_CHIP_MULT
                    nxt_target *= BOSS_CHIP_MULT.get(game.boss_blind or "", 1.0)
                    p_next = blind_model_for(game, self.hand_cfg).p_clear(
                        nxt_target / max(px["ratio"], 1e-6), int(game.base_hands), int(game.base_discards))
            else:
                p_next = blind_model_for(game, self.hand_cfg).p_clear(
                    nxt_target / max(px["ratio"], 1e-6), int(game.base_hands), int(game.base_discards))
            premium = tag in PREMIUM_TAGS
            ev = (p_next + 0.02) if (premium and p_next >= self.cfg.skip_min_p_clear) else (p_next - 1.0)
            out.append(({"type": "skip_blind"}, ev,
                        f"skip for {tag or 'no tag'} (P(clear next) {p_next:.2f}{', premium' if premium else ''})"))
        if "reroll_boss" in keys:
            out.append(({"type": "reroll_boss"}, -2.0, "reroll boss (rule: never)"))
        out.sort(key=lambda x: -x[1])
        return out

    # ── W4 stats tier ───────────────────────────────────────────────────────

    def _rank_with_stats(self, game, legal: list) -> list:
        try:
            rows = self.stats.decision_table(game)
        except Exception:           # noqa: BLE001 - W4 not ready: fall through to the rules
            return []
        legal_keys = {_hand._action_sort_key(a) for a in legal}
        out = []
        for row in rows or []:
            action = getattr(row, "action", None) if not isinstance(row, dict) else row.get("action")
            net = getattr(row, "net_ev", None) if not isinstance(row, dict) else row.get("net_ev")
            if not isinstance(action, dict) or net is None:
                continue
            if _hand._action_sort_key(action) not in legal_keys:
                continue
            label = getattr(row, "label", None) if not isinstance(row, dict) else row.get("label")
            out.append((dict(action), float(net), f"stats: {label or action.get('type')} net_ev {float(net):.3f}"))
        if not out:
            return []
        out.sort(key=lambda x: -x[1])
        leave = {"type": "leave_shop"} if game.state == State.SHOP else {"type": "skip_booster"}
        if out[0][1] <= 0.0:
            return [(leave, 0.0, "stats: nothing with positive net EV")] + out
        return out

    # ── V tier ──────────────────────────────────────────────────────────────

    def _rank_with_value(self, game, legal: list) -> list:
        cands = self._shop_candidates(game, legal)
        rng = world_rng(self.seed, game, salt=0x5A)
        nw = self.n_worlds if self.n_worlds is not None else self.cfg.n_worlds
        worlds = None
        out = []
        for a in cands:
            chance = a["type"] in ("reroll", "skip_blind", "play_blind") or (
                a["type"] == "buy" and a.get("item_idx", 0) < len(game.current_shop)
                and game.current_shop[a["item_idx"]].kind == "booster")
            if chance:
                if worlds is None:
                    worlds = [sample_world(game, rng) for _ in range(max(1, nw))]
                vals = []
                for w in worlds:
                    c = w.clone()
                    c.step(a)
                    vals.append(self._v(c))
                ev = sum(vals) / len(vals)
            else:
                c = game.clone()
                c.step(a)
                ev = self._v(c)
            out.append((a, ev, f"V after {a['type']} = {ev:.3f}"))
        out.sort(key=lambda x: (-x[1], _hand._action_sort_key(x[0])))
        return out

    def _v(self, clone) -> float:
        # a broken value_fn propagates (fix pass: silent degradation hid V bugs from W5)
        return float(self.value_fn(clone))

    def _shop_candidates(self, game, legal: list) -> list:
        """Legal actions worth evaluating (caps the enumeration)."""
        out = []
        for a in legal:
            if a["type"] == "use_consumable" and a.get("target_cards"):
                continue
            out.append(a)
        return out[: self.cfg.max_shop_candidates]

    # ── W-SHOP: $ valuation of what a slot / a pack / the shelf can hold ────

    def _shop_ev_on(self) -> bool:
        return _hit is not None and _seconomy is not None

    def _cycle_cost(self, game) -> float:
        """$ it costs to put ONE more joker on a FULL board — Tagg's cycle-sell, priced.

        With a free slot: 0.  With the board full, taking a joker means selling the weakest
        owned one, which returns its sell value and gives up its ongoing value, so the net
        toll is ``min_j (value(j) - sell(j))`` over the owned jokers — never below 0,
        because a joker whose sell price already exceeds its value should be cycled anyway.
        Both terms use the SAME cheap pool model the candidate is priced with, so the
        comparison is like-for-like (STATS_NOTES §1's "never price a known card with the
        pool proxy" applies to the decision table's precise rows, not to this A-vs-B toll)."""
        if len(game.jokers) < game.joker_slots or not game.jokers:
            return 0.0
        from balatro_sim.jokers.base import joker_sell_value
        owned = [j.key for j in game.jokers]
        ante = int(game.ante)
        best = None
        for j in game.jokers:
            if j.state.get("eternal"):
                continue                       # an Eternal joker cannot be sold at all
            toll = self._joker_dollars(j.key, owned, ante) - float(joker_sell_value(j))
            if best is None or toll < best:
                best = toll
        if best is None:
            return 1e9                          # every joker is Eternal: no cycle exists
        return max(0.0, best)

    #: the elementary deck changes `_deck_effects` can measure, and which valuations need
    #: which — measuring lazily matters because most shops need one or two of the five.
    _EFFECT_KEYS = ("suit3", "steel1", "destroy2", "rank2", "add_plain")
    _TAROT_EFFECTS = ("suit3", "steel1", "destroy2", "rank2")
    _CARD_EFFECTS = ("steel1", "add_plain")

    def _deck_effects(self, game, need: tuple = _EFFECT_KEYS) -> dict:
        """The MEASURED $ value of elementary deck changes, for THIS deck.

        This is the number a flat ``hit.tarot_value`` ($4 for every tarot, flagged in
        STATS_NOTES §1 as a documented gap) cannot have: an Arcana pack's whole case is that
        a suit tarot turns a scattered deck into a flush deck, and how much that is worth is
        a property of the deck in front of you.  So it is measured, not tabulated — the
        build proxy is evaluated on a clone whose deck has been changed, and the
        P(clear)+strength delta is divided by ``lam_money`` to land back in dollars:

        ``suit3``      3 off-suit cards converted to the deck's modal suit  (the suit tarots)
        ``steel1``     one plain card enhanced to Steel   (the anchor for all 8 enhancement
                       tarots and for a Standard pack's enhanced cards; the shape within the
                       family comes from ``hit.card_value``'s enhancement table)
        ``destroy2``   two worst cards removed             (The Hanged Man; deck thinning)
        ``rank2``      two cards' rank raised by one       (Strength, and Death's upgrade)
        ``add_plain``  one average plain card ADDED        (a Standard pack's base card —
                       routinely small, and the dilution the flat $1 card table cannot say)

        Memoised per (deck shape, levels, ante) AND per effect: ``need`` names only the
        effects the caller's valuation actually uses, so a shop with no Arcana pack and no
        shelf tarot never pays for ``suit3``."""
        cfg = self.cfg
        deck = game.full_deck
        suits: dict = {}
        n_enh = 0
        for c in deck:
            suits[c.suit] = suits.get(c.suit, 0) + 1
            if c.enhancement != "None":
                n_enh += 1
        # coarse on purpose: these numbers move with the deck's SHAPE, and a key that
        # changed on every individual enhancement would miss on every shop visit.
        key = (tuple(sorted(suits.items())), len(deck), n_enh, int(game.ante),
               tuple(sorted(game.planet_levels.items())), int(game.hand_size))
        out = self._fix_cache.get(key)
        if out is None:
            if len(self._fix_cache) > 64:
                self._fix_cache.clear()
            out = {}
            self._fix_cache[key] = out
        todo = [k for k in need if k not in out]
        if not todo:
            return out
        try:
            base = build_proxy(game, cfg, self.proxy_cfg, ratio_cache=self._ratio_cache)

            def measure(mutate) -> float:
                c = game.clone()
                if not mutate(c):
                    return 0.0
                px = build_proxy(c, cfg, self.proxy_cfg, ratio_cache=self._ratio_cache)
                d = (px["value"] - base["value"]) / max(1e-9, cfg.lam_money)
                return max(-cfg.effect_cap_dollars, min(cfg.effect_cap_dollars, d))

            top = max(suits, key=lambda x: (suits[x], x)) if suits else None

            def to_suit(c):
                if top is None:
                    return False
                n = 0
                for card in c.full_deck:
                    if n >= 3:
                        break
                    if card.suit != top and card.enhancement != "Stone":
                        card.suit = top
                        n += 1
                return n > 0

            def to_steel(c):
                for card in c.full_deck:
                    if card.enhancement == "None":
                        card.enhancement = "Steel"
                        return True
                return False

            def destroy2(c):
                victims = sorted((x for x in c.full_deck if x.enhancement == "None"
                                  and x.seal == "None" and x.edition == "None"),
                                 key=lambda x: x.rank)[:2]
                for card in victims:
                    c.remove_card(card)
                return bool(victims)

            def rank2(c):
                n = 0
                for card in c.full_deck:
                    if n >= 2:
                        break
                    if card.rank < 14 and card.enhancement != "Stone":
                        card.rank += 1
                        n += 1
                return n > 0

            def add_plain() -> float:
                ranks: dict = {}
                for card in deck:
                    ranks[card.rank] = ranks.get(card.rank, 0) + 1
                probes = sorted(ranks, key=lambda r: (ranks[r], r))
                probes = [probes[0], probes[-1]] if probes else []

                def add_of(rank):
                    def go(c):
                        from balatro_sim.card import Card
                        c.add_card(Card(rank=rank, suit=top or "Spades"))
                        return True
                    return go
                # a Standard pack's base card is a uniform draw, so the probe must not be
                # one rank: duplicating a rank the deck is deep in reads as a pair bonus,
                # duplicating a thin one reads as dilution.  Rarest and commonest, averaged.
                adds = [measure(add_of(r)) for r in probes]
                return (sum(adds) / len(adds)) if adds else 0.0

            makers = {"suit3": lambda: max(_TAROT_SUIT_FLOOR, measure(to_suit)),
                      "steel1": lambda: measure(to_steel),
                      "destroy2": lambda: measure(destroy2),
                      "rank2": lambda: measure(rank2),
                      "add_plain": add_plain}
            for k in todo:
                out[k] = makers[k]()
        except Exception:                # noqa: BLE001 — a valuation must never break `act`
            for k in todo:
                out.setdefault(k, _TAROT_SUIT_FLOOR if k == "suit3" else 0.0)
        return out

    def _joker_dollars(self, key: str, owned: list, ante: int) -> float:
        """The pool model's $ for an UNSEEN joker (``hit.pool_dollar_value``: a hand-tuned
        1..10 strength times a coherence multiplier).  Used where the comparison is
        pool-member vs pool-member or pool-member vs its own sticker price — a Buffoon
        pack's contents, and the cycle-sell toll.  Deliberately NOT used for the reroll:
        SHOP_NOTES §2.1 measures why (it cannot see the board interaction that makes a
        joker worth rolling for, and no monotone rescaling of it can)."""
        return _hit.pool_dollar_value(key, owned, ante)

    def _consumable_dollars(self, game, kind: str, key: str, effects: dict) -> float:
        if kind == "tarot":
            return tarot_dollars(game, key, self.cfg, effects)
        if kind == "planet":
            return _hit.planet_value(game, key)
        if kind == "spectral":
            return _hit.spectral_value(key)
        return 0.0

    def _fresh_shelf_values(self, game, base_value: float) -> list:
        """One number per sampled world: what the BEST purchase on a FRESH shelf would be
        worth, in the rules tier's own P(clear) units, floored at 0 (a shelf you buy
        nothing from is worth nothing, not something negative).

        This is the piece SHOP_NOTES §2.1 argues has to be Monte-Carlo rather than
        analytic.  The obvious analytic route — ``hit.shop_slot_distribution`` over the
        generator's culled pools, priced with ``hit.pool_dollar_value`` — was built and
        measured first, and its numbers do not describe this player: over 40 seeds and 297
        real shelf jokers the pool model's value quantiles run $4.8 / $7.2 / $10.0 / $12.9
        (10th / 50th / 90th / 99th) while the SAME jokers' value to the build proxy runs
        -$1.6 / $0 / $43 / $142.  Half of every shelf is worth nothing to this build and a
        tenth of it is worth an ante; no monotone rescaling of a $5-to-$13 model produces
        that, and the one I fitted (x2.4, matching the means) made the typical draw look
        like a bargain and rolled the player broke — measured, and the reason this code
        exists in this shape.

        So the shelf is valued the way every other row in this tier is valued: the build
        proxy, on real draws.  ``clone_determinized`` (W2) gives a clone whose keyed RNG is
        replaced wholesale, so ``reroll`` on it produces a genuine draw from the same culled
        pools with a decorrelated stream — an honest sample of "the next two queue slots",
        never the true seed's.  The world's money and reroll counter are restored after the
        roll because the roll's PRICE is accounted separately (``_reroll_costs``); what is
        wanted here is the shelf, not the shelf minus the fee.

        Cached per (shop position, reroll count, owned jokers, deck, money tier): a visit
        asks for this once per reroll depth, not once per decision."""
        cfg = self.cfg
        nw = max(1, int(cfg.reroll_worlds))
        paid = 0 if int(getattr(game, "free_rerolls_remaining", 0)) > 0 else             max(0, int(game.reroll_cost) - int(game.reroll_discount))
        restore = cfg.lam_money * (paid + interest_cost(game, cfg, paid))
        key = (int(game.ante), int(game.blind_idx), int(game.reroll_cost),
               int(game.dollars) // 10, tuple(j.key for j in game.jokers),
               len(game.full_deck), len(game.consumable_hand),
               tuple(sorted(game.planet_levels.items())), nw)
        got = self._atom_cache.get(key)
        if got is not None:
            return got
        # seeded from the OBSERVABLE STATE ONLY, not `self.seed`: with epsilon = 0 this
        # player must be a function of the state alone in the fast budget (two seats of one
        # h2h are the same trajectory mirrored -- `test_h2h.py::
        # test_seatings_are_mirrors_for_an_identical_matchup` pins it), and the shop tier
        # has no use for per-player world variety the way the full budget's rollouts do.
        rng = world_rng(0, game, salt=0x5B)
        vals: list = []
        for _ in range(nw):
            try:
                w = game.clone_determinized(rng.getrandbits(62))
                if not w.step({"type": "reroll"}):
                    pass
                # the roll's PRICE is accounted separately (`_reroll_costs`), but the
                # money really is gone when the new shelf is on display, so the world must
                # not be allowed to buy something the post-roll balance cannot afford.
                w.dollars = max(0, int(game.dollars) - int(paid))
                w.reroll_cost = game.reroll_cost
                w.free_rerolls_remaining = game.free_rerolls_remaining
                # ... and added back into the VALUE, because `base_value` was measured at
                # the pre-roll balance and `build_proxy` prices money: without this the
                # world's proxy is already down by the fee and `_reroll_plan` would charge
                # it a second time.  What is wanted from a world is the shelf alone.
                vals.append(self._best_purchase_value(w, base_value) + restore)
            except Exception:        # noqa: BLE001 — a sampled world must never break `act`
                vals.append(0.0)
        if len(self._atom_cache) > 64:
            self._atom_cache.clear()
        self._atom_cache[key] = vals
        return vals

    def _best_purchase_value(self, w, base_value: float) -> float:
        """Best proxy gain over the REROLLABLE items of ``w``'s shelf, floored at 0.

        Only ``shop.SHELF_KINDS`` are considered: a reroll leaves the voucher and the two
        booster slots untouched (shop.py:496), so they are not part of what a roll buys."""
        cfg = self.cfg
        best = 0.0
        for i, item in enumerate(w.current_shop):
            if item.sold or item.kind not in ("joker", "tarot", "planet", "spectral", "card"):
                continue
            price = effective_price(w, item)
            if w.dollars < price:
                continue
            c = w.clone()
            try:
                c.step({"type": "buy", "item_idx": i})
            except Exception:        # noqa: BLE001
                continue
            if c.dollars == w.dollars:
                continue             # the engine refused it (slots full)
            px = build_proxy(c, cfg, self.proxy_cfg, ratio_cache=self._ratio_cache)
            gain = px["value"] - base_value
            if item.kind in ("tarot", "spectral"):
                # the proxy cannot see a consumable sitting in hand (§3.5): price it
                gain += cfg.lam_money * self._consumable_dollars(
                    w, item.kind, item.key, self._deck_effects(w, self._TAROT_EFFECTS))
            if gain > best:
                best = gain
        return best

    def _reroll_costs(self, game, n_max: int) -> list:
        """True $ cost of the 1st..n-th reroll of THIS visit, in the order they would be
        paid.  Three terms, all of them Tagg's:

        (a) the sticker price, which ESCALATES by $1 per reroll inside a visit
            (``shop.reroll_shop``) and resets to $5 at the next shop
            (``game._end_blind_and_enter_shop``); free rerolls (Chaos the Clown) come first;
        (b) the MARGINAL interest forfeited by the cumulative spend
            (``economy.interest_loss``, geometrically discounted over the remaining shops).
            This is the interest-threshold term: above the cap ($25 at cap 5) it is exactly
            0, and every $5 breakpoint the plan crosses on the way down adds a real dollar
            of future income to the price of the roll that crossed it;
        (c) the SPREAD opportunity cost.  The reroll price resets each shop while the ante's
            queue is shared and persistent (GENERATION_SPEC §8.3), so while more shops on
            this queue remain, the $1-per-roll escalation is avoidable — the same queue
            depth is available at $5 one blind later.  Rolls above the base price are
            therefore charged ``reroll_defer_delta`` of their escalation premium again.
            At the LAST shop of the queue (``shops_left_on_queue == 1``, the shop before the
            Boss / Nemesis) the term is exactly 0: depth not bought now is lost."""
        cfg = self.cfg
        base = max(0, int(game.reroll_cost) - int(game.reroll_discount))
        free = int(getattr(game, "free_rerolls_remaining", 0))
        spread = shops_left_on_queue(game) > 1
        out = []
        cum = 0
        prev_loss = 0.0
        for i in range(n_max):
            price = 0 if i < free else max(0, base + (i - free))
            cum += price
            loss = interest_cost(game, cfg, cum)
            marginal = loss - prev_loss
            prev_loss = loss
            extra = (cfg.reroll_defer_delta * max(0, price - 5)) if spread else 0.0
            out.append((float(price), float(price) + marginal + extra, cum))
        return out

    def _reroll_plan(self, game, base_value: float, best_buy: float) -> dict:
        """The stopping rule (SHOP_NOTES.md §2), entirely in the rules tier's own units.

        Optimal-stopping backward induction over at most ``reroll_max_depth`` rolls, on the
        sampled fresh-shelf values ``X`` of ``_fresh_shelf_values``::

            V_K = E[max(0, X)]                              # last allowed roll: take or leave
            V_j = E[max( max(0, X), V_{j+1} - cost_{j+1} )]  # ... or pay for one more shelf
            plan = V_1 - cost_1

        "take the best thing this shelf shows, or pay for another shelf, whichever is worth
        more" — which is what makes the rule ITERATE correctly down the escalating price
        schedule without any per-visit memory: the player is re-asked after every reroll,
        `game.reroll_cost` has moved, and the same induction runs on the new schedule.

        ``plan`` is a proxy-value CHANGE, exactly like every buy row's gain, so the caller
        ranks the two against each other and the argmax settles "buy this, or roll for
        something better" — which is also how the forfeited shelf gets priced, since taking
        the buy row instead of the reroll row IS keeping it.  ``best_buy`` is carried only
        for the reason string."""
        cfg = self.cfg
        out = {"roll": False, "plan": 0.0, "v_now": max(0.0, best_buy), "depth": 0,
               "agg": 1.0, "cost1": 0.0, "shops_left": shops_left_on_queue(game),
               "worlds": ()}
        agg = self._race_aggression(game)
        out["agg"] = agg
        budget = int(game.dollars) - int(cfg.reroll_reserve)
        costs = self._reroll_costs(game, cfg.reroll_max_depth)
        k_max = 0
        for _price, _true, cum in costs:
            if cum > budget:
                break
            k_max += 1
        if k_max <= 0:
            return out
        xs = self._fresh_shelf_values(game, base_value)
        if not xs:
            return out
        out["worlds"] = tuple(round(x, 4) for x in xs)
        out["cost1"] = costs[0][1]
        # Proxy units per dollar.  The race factor divides the cost, which is the same
        # thing as multiplying every improvement by it and keeps the sampled values
        # untouched.  So does the money-slack factor: `build_proxy` prices a dollar at a
        # CONSTANT `lam_money` whatever the balance, and above the interest cap that is
        # simply wrong -- the marginal dollar earns no interest and, measured, this player
        # never spends it (it ends MLB matches on $32-53 of cash).  Discounting a roll's
        # cost toward `reroll_rich_floor` as the balance runs past the cap is what makes
        # the rule spend money that has no other use, which is the ONE thing the old
        # `reroll_floor = 25` rule got right and the first EV draft lost (SHOP_NOTES §2.4).
        cap_dollars = INTEREST_RATE * float(getattr(game, "interest_cap", INTEREST_CAP))
        slack = 1.0
        if cap_dollars > 0 and game.dollars > cap_dollars:
            slack = max(cfg.reroll_rich_floor, cap_dollars / float(game.dollars))
        lam = cfg.lam_money * cfg.reroll_hurdle * slack / max(1e-9, agg)
        v = sum(max(0.0, x) for x in xs) / len(xs)                       # V_K
        for j in range(k_max - 1, 0, -1):
            c = v - lam * costs[j][1]
            v = sum(max(max(0.0, x), c) for x in xs) / len(xs)
        plan = v - lam * costs[0][1]
        out["plan"] = plan
        out["depth"] = k_max
        out["roll"] = plan > cfg.lam_money * cfg.reroll_margin_dollars
        return out

    def _pack_ev(self, game, item) -> tuple:
        """``(net $ EV, reason)`` of BUYING an unopened booster (SHOP_NOTES.md §3).

        ``E[sum of the best `choose` of `size` iid pack cards]`` under the pack's own
        content pool, minus the price and its interest loss.  Pack picks are FREE once the
        pack is bought, so a pack card's value is its full $ value, not a surplus — that
        asymmetry versus a reroll (where every hit still has to be paid for) is most of why
        packs beat rerolls at low money, and it now falls out of the arithmetic instead of
        being asserted by a preference order."""
        from balatro_sim import game_keys as _gk
        b = _gk.BOOSTER_TYPES.get(item.key, {})
        kind = b.get("kind", "")
        size = max(1, int(b.get("cards", 1)))
        choose = max(1, int(b.get("choose", 1)))
        price = float(effective_price(game, item))
        cfg = self.cfg
        effects = self._deck_effects(
            game, self._CARD_EFFECTS if kind == "Standard"
            else (self._TAROT_EFFECTS if kind == "Arcana" else ()))
        atoms: list = []
        if kind == "Standard":
            atoms = _standard_card_atoms(cfg, effects)
        elif kind == "Buffoon":
            cycle = self._cycle_cost(game)
            owned = [j.key for j in game.jokers]
            ante = int(game.ante)
            try:
                slices = _hit.pack_slices(game, kind)
            except Exception:            # noqa: BLE001
                slices = []
            for s in slices:
                n = len(s.pool)
                if n <= 0 or s.p_component <= 0:
                    continue
                w = s.p_component / n
                for k in s.pool:
                    atoms.append((w, self._joker_dollars(k, owned, ante) - cycle))
        else:
            if len(game.consumable_hand) >= game.consumable_slots:
                atoms = [(1.0, 0.0)]
            else:
                gtype = {"Arcana": "tarot", "Celestial": "planet", "Spectral": "spectral"}.get(kind, "")
                try:
                    slices = _hit.pack_slices(game, kind)
                except Exception:        # noqa: BLE001
                    slices = []
                for s in slices:
                    n = len(s.pool)
                    if n <= 0 or s.p_component <= 0:
                        continue
                    w = s.p_component / n
                    for k in s.pool:
                        atoms.append((w, self._consumable_dollars(game, gtype, k, effects)))
                # a pack fills consumable SLOTS: only what fits can actually be taken
                choose = min(choose, max(0, game.consumable_slots - len(game.consumable_hand)))
        if not atoms or choose <= 0:
            return -1.0, f"{item.name}: nothing takeable"
        levels, tails = _levels_from_atoms(atoms)
        gross = _e_top_k(levels, tails, size, choose)
        il = interest_cost(game, cfg, price)
        agg = self._race_aggression(game)
        net = gross * agg - price - il
        return net, (f"{item.name} ${price:.0f}: E[best {choose} of {size}] ${gross:.2f}"
                     f"{f' x{agg:.2f}' if agg != 1.0 else ''} - int ${il:.2f} = ${net:+.2f}")

    def _fool_shift(self, game, key: str) -> float:
        """Sequencing shift for using consumable ``key`` while The Fool is held (§4).

        The Fool copies ``run_state.last_tarot_planet`` — whatever Tarot or Planet was used
        LAST (``consumables.apply_tarot``, and The Fool itself never overwrites it).  So the
        order of a batch of uses is worth something: use the cheap ones first and the best
        one last, and the Fool doubles the best one instead of the last one that happened to
        sort first.  Two shifts, both in the rules tier's P(clear) units:

        * a use that is NOT the last one available is pushed down in proportion to how much
          better than the rest of the batch it is (``fool_order_dollars`` per $), so the
          batch drains weakest-first;
        * The Fool itself is deferred while some held use would give it a better copy.

        Both are inert unless ``c_fool`` is actually in hand, and inert under
        ``fool_order=False``."""
        cfg = self.cfg
        hcfg = self.hand_cfg
        if not cfg.fool_order or not getattr(hcfg, "fool_order", True) or not self._shop_ev_on():
            return 0.0
        hand = list(game.consumable_hand)
        if "c_fool" not in hand:
            return 0.0
        fix = self._deck_effects(game, self._TAROT_EFFECTS)

        def val(k: str) -> float:
            kd = _gk_consumable_kind(k)
            return self._consumable_dollars(game, kd, k, fix) if kd else 0.0

        others = [k for k in hand if k != "c_fool"]
        if key == "c_fool":
            best_other = max((val(k) for k in others), default=0.0)
            last = getattr(game.run_state, "last_tarot_planet", None)
            have = val(last) if last and last != "c_fool" else 0.0
            return -hcfg.fool_defer_penalty if best_other > have else 0.0
        rest = list(others)
        if key in rest:
            rest.remove(key)
        if not rest:
            return 0.0                     # this IS the last use before the Fool: no shift
        mean_rest = sum(val(k) for k in rest) / len(rest)
        return -hcfg.fool_order_dollars * max(0.0, val(key) - mean_rest)

    # ── rule tier: shop ─────────────────────────────────────────────────────

    def _interest_floor(self, game) -> int:
        return min(25, self.cfg.pack_floor_ante_step * max(0, game.ante))

    def _rank_shop_rules(self, game, legal: list) -> list:
        cfg = self.cfg
        legal_keys = {_hand._action_sort_key(a): a for a in legal}
        out = []
        # 1. consumables usable here (planets first, then any untargeted use that succeeds)
        for a in legal:
            if a["type"] != "use_consumable" or a.get("target_cards"):
                continue
            key = game.consumable_hand[a["consumable_idx"]] if a["consumable_idx"] < len(game.consumable_hand) else ""
            c = game.clone()
            n0 = len(c.consumable_hand)
            c.step(a)
            if len(c.consumable_hand) < n0 and c.state == State.SHOP:
                bonus = 5.0 if key in PLANET_HAND else 4.0
                shift = self._fool_shift(game, key)
                out.append((a, bonus + shift,
                            f"use {key} now" + (f" [fool {shift:+.3f}]" if shift else "")))
        base = build_proxy(game, cfg, self.proxy_cfg, ratio_cache=self._ratio_cache)
        floor = self._interest_floor(game)
        p_clear = base["p_clear"]
        # 2. shelf purchases
        sells = []
        for a in legal:
            if a["type"] != "buy":
                continue
            item = game.current_shop[a["item_idx"]]
            price = effective_price(game, item)
            if item.kind == "joker":
                c = game.clone()
                c.step(a)
                if len(c.jokers) <= len(game.jokers) and item.edition != "Negative":
                    continue
                px = build_proxy(c, cfg, self.proxy_cfg, ratio_cache=self._ratio_cache)
                gain = px["value"] - base["value"]
                out.append((a, gain, f"buy {item.name} ${price}: P(clear) {base['p_clear']:.2f}->{px['p_clear']:.2f}, "
                                     f"strength {base['strength']:.0f}->{px['strength']:.0f} (gain {gain:+.3f})"))
            elif item.kind in ("planet", "tarot", "spectral"):
                c = game.clone()
                c.step(a)
                px = build_proxy(c, cfg, self.proxy_cfg, ratio_cache=self._ratio_cache)
                gain = px["value"] - base["value"]
                why = ""
                if item.kind != "planet":
                    if cfg.pack_ev and self._shop_ev_on():
                        # W-SHOP §3.5: the proxy sees a bought Tarot/Spectral only as money
                        # leaving (a use needs cards in hand, which no SHOP state has), so
                        # its own row was "worth its price only when cheap and early" — the
                        # rule that made the shop pass deck-fixing tarots.  Add the card's
                        # own $ value on top of what the proxy did see.
                        d = self._consumable_dollars(game, item.kind, item.key,
                                                     self._deck_effects(game, self._TAROT_EFFECTS))
                        gain = -1.0 if game.dollars - price < floor else gain + cfg.lam_money * d
                        why = f", ${d:.1f} unpriced-by-proxy"
                    else:
                        # an unvalued tarot/spectral: worth its price only when cheap and early
                        gain = (cfg.lam_money * (3 - price)) if game.dollars - price >= floor else -1.0
                out.append((a, gain, f"buy {item.name} ${price} (gain {gain:+.3f}{why})"))
            elif item.kind == "voucher":
                ok = (item.key not in _SKIP_VOUCHERS) and (game.dollars - price >= floor)
                gain = 0.02 if ok else -1.0
                out.append((a, gain, f"buy voucher {item.name} ${price} ({'affordable after floor' if ok else 'below floor'})"))
            elif item.kind == "booster":
                # W-SHOP §3: price the pack instead of ranking its family.  The low-money
                # guard stays exactly as it was (a pack must leave the ante's interest
                # floor intact, unless the build is failing and needs the help).
                ok = game.dollars - price >= floor + cfg.pack_low_money_floor or \
                    (p_clear < 0.6 and game.dollars >= price)
                if cfg.pack_ev and self._shop_ev_on():
                    if not ok:
                        gain, why = -1.0, f"open {item.name} ${price} (below the interest floor)"
                    else:
                        net, why = self._pack_ev(game, item)
                        gain = cfg.lam_money * net
                        if "standard" not in item.key:
                            # Tagg's prior, and the old rule's actual behaviour: a pack that
                            # clears the money guard is taken -- what the EV replaces is the
                            # fixed buffoon > celestial > arcana > standard > spectral ORDER,
                            # not the taking.  Measured: without this floor the h2h against
                            # the old shop is 43.3% and the new arm ends MLB matches on $42
                            # of unspent money (SHOP_NOTES §3.1).  Standard packs are exempt
                            # -- theirs is the one family the measurement, not the prior,
                            # is allowed to talk down.
                            gain = max(gain, cfg.pack_take_floor)
                        why = "open " + why
                else:
                    pref = next((i for i, k in enumerate(_PACK_PREF) if k in item.key), len(_PACK_PREF))
                    gain = (0.015 - 0.002 * pref) if ok else -1.0
                    why = f"open {item.name} ${price} ({'ok' if ok else 'below floor'})"
                out.append((a, gain, why))
            elif item.kind == "card":
                ok = game.dollars - price >= floor + 5
                gain = 0.005 if ok else -1.0
                out.append((a, gain, f"buy card {item.name} ${price}"))
        # 3. make room: sell the weakest joker when a shelf joker beats it clearly
        if len(game.jokers) >= game.joker_slots and game.jokers:
            shelf_gain = 0.0
            for a in legal:
                if a["type"] == "buy" and game.current_shop[a["item_idx"]].kind == "joker":
                    continue
            # gains of shelf jokers that could not be bought (slots full) -- evaluate on a clone with a slot freed
            worst = None
            for ji in range(len(game.jokers)):
                c = game.clone()
                c.step({"type": "sell_joker", "joker_idx": ji})
                px = build_proxy(c, cfg, self.proxy_cfg, ratio_cache=self._ratio_cache)
                loss = base["value"] - px["value"]
                if worst is None or loss < worst[1]:
                    worst = (ji, loss, c)
            if worst is not None:
                ji, loss, c_sold = worst
                best_shelf = None
                for i, item in enumerate(game.current_shop):
                    if item.kind != "joker" or item.sold:
                        continue
                    price = effective_price(game, item)
                    if c_sold.dollars < price:
                        continue
                    c2 = c_sold.clone()
                    c2.step({"type": "buy", "item_idx": i})
                    if len(c2.jokers) <= len(c_sold.jokers):
                        continue
                    px2 = build_proxy(c2, cfg, self.proxy_cfg, ratio_cache=self._ratio_cache)
                    g2 = px2["value"] - base["value"]
                    if best_shelf is None or g2 > best_shelf[1]:
                        best_shelf = (item, g2)
                # W-SHOP §3.3, Tagg's cycle-sell: an unopened Buffoon pack ALSO wants the
                # slot, and it is the shelf joker that makes the trade near-riskless (the
                # pack can disappoint and the sell is still paid for).  So the value of
                # freeing a slot is the better of the two uses; the pack itself is then
                # priced with `_cycle_cost == 0` on its own row and opens on the next step.
                pack_gain, pack_why = 0.0, ""
                if cfg.pack_ev and self._shop_ev_on():
                    for i, it in enumerate(game.current_shop):
                        if it.kind != "booster" or it.sold or "buffoon" not in it.key:
                            continue
                        if c_sold.dollars < effective_price(game, it):
                            continue
                        net, why = self._pack_ev(c_sold, it)
                        if cfg.lam_money * net > pack_gain:
                            pack_gain, pack_why = cfg.lam_money * net, why
                floor_gain = best_shelf[1] if best_shelf is not None else None
                gain = max(floor_gain if floor_gain is not None else -1.0, pack_gain)
                if gain > cfg.sell_margin:
                    tail = (f"buy {best_shelf[0].name}" if floor_gain is not None
                            and floor_gain >= pack_gain else f"open {pack_why}")
                    out.append(({"type": "sell_joker", "joker_idx": ji}, gain,
                                f"sell {game.jokers[ji].key} to {tail} (gain {gain:+.3f})"))
        # 4. reroll -- W-SHOP §2: EV-driven and uncapped, or the pre-2026-08-26 rule
        if _hand._action_sort_key({"type": "reroll"}) in legal_keys:
            done = max(0, int(game.reroll_cost) - 5)
            cost = max(0, game.reroll_cost - game.reroll_discount)
            free = game.free_rerolls_remaining > 0
            best_buy = max((ev for a, ev, _ in out if a["type"] in ("buy", "sell_joker")), default=-1.0)
            if cfg.reroll_ev and self._shop_ev_on():
                # The reroll row is scored on the SAME scale as every buy row: the expected
                # change in the build proxy from here.  So it is left to the argmax to
                # decide between "buy this" and "roll for something better", which is the
                # whole point -- an earlier draft gated the roll behind "nothing is worth
                # buying", and that gate is what made rolling look worthless: it only ever
                # asked the question after the player had already spent down to the money
                # that cannot afford whatever the new shelf shows.  Measured, on 24 seeds:
                # E[best purchase on a fresh shelf] is 0.016 proxy units asked AFTER the
                # buying (the gated order) and 0.093 asked BEFORE it (SHOP_NOTES §2.2).
                # The two guards the OLD rule used are kept under `reroll_guard`: roll only
                # when nothing on the shelf is worth buying, and only while the balance
                # stays at the interest cap.  What the EV rule replaces is the hard
                # `max_rerolls_per_visit = 1` cap -- the DEPTH, which is what the brief asks
                # for.  Measured (SHOP_NOTES §2.5): dropping the guards as well and letting
                # the plan compete with every buy row loses the h2h at every hurdle tried.
                guarded = (not cfg.reroll_guard) or (
                    best_buy <= cfg.buy_margin
                    and (free or game.dollars - cost >= cfg.reroll_floor))
                p = self._reroll_plan(game, base["value"], best_buy) if guarded else None
                if p is None:
                    out.append(({"type": "reroll"}, -1.0,
                                f"reroll ${cost} (guard: best buy {best_buy:+.4f}, "
                                f"${game.dollars} - ${cost} vs floor ${cfg.reroll_floor})"))
                    p = {"plan": 0.0, "roll": False, "depth": 0, "worlds": (), "agg": 1.0,
                         "shops_left": shops_left_on_queue(game)}
                    ev = None
                else:
                    ev = p["plan"] if p["roll"] else -1.0
                race = f", race x{p['agg']:.2f}" if p["agg"] != 1.0 else ""
                if ev is not None:
                    out.append(({"type": "reroll"}, ev,
                                f"reroll ${cost} (plan {p['plan']:+.4f}, best buy "
                                f"{best_buy:+.4f}, depth {p['depth']}, worlds "
                                f"{list(p['worlds'])}, {p['shops_left']} shop(s) on this "
                                f"queue{race})"))
            else:
                ok = (free or (done < cfg.max_rerolls_per_visit and game.dollars - cost >= cfg.reroll_floor)) \
                    and best_buy <= cfg.buy_margin
                out.append(({"type": "reroll"}, 0.001 if ok else -1.0,
                            f"reroll ${cost} ({'ok' if ok else 'no'}; {done} done)"))
        out.append(({"type": "leave_shop"}, cfg.buy_margin, f"leave (P(clear next) {p_clear:.2f}, ${game.dollars})"))
        out.sort(key=lambda x: (-x[1], _hand._action_sort_key(x[0])))
        return out

    # ── rule tier: booster ──────────────────────────────────────────────────

    def _rank_booster_rules(self, game, legal: list) -> list:
        cfg = self.cfg
        base = build_proxy(game, cfg, self.proxy_cfg, ratio_cache=self._ratio_cache)
        out = []
        for a in legal:
            if a["type"] != "pick_booster":
                continue
            c = game.clone()
            c.step(a)
            px = build_proxy(c, cfg, self.proxy_cfg, ratio_cache=self._ratio_cache)
            gain = px["value"] - base["value"]
            names = []
            blind_dollars = 0.0
            for i in a.get("indices", []):
                ch = game.booster_choices[i] if i < len(game.booster_choices) else None
                names.append(getattr(ch, "key", None) or (repr(ch.card) if getattr(ch, "card", None) else "?"))
                # W-SHOP §3.5: the build proxy is BLIND to a Tarot / Spectral in hand (it
                # only sees the deck a use would produce, and a use needs cards in hand,
                # which a SHOP/BOOSTER state does not have).  The old rule handled that by
                # flooring every such pick at 1e-4, which made every unvalued pick tie and
                # let `_action_sort_key` choose — i.e. the Arcana pick was arbitrary.  Price
                # them instead, on the same $ table the buy rows now use.
                if cfg.pack_ev and self._shop_ev_on() and ch is not None and \
                        getattr(ch, "set", "") in ("Tarot", "Spectral"):
                    blind_dollars += self._consumable_dollars(
                        game, ch.set.lower(), ch.key,
                        self._deck_effects(game, self._TAROT_EFFECTS))
            gain += cfg.lam_money * blind_dollars
            # a pick that the proxy STILL cannot see is free value: prefer taking to skipping
            gain = max(gain, 1e-4) if gain <= 0 else gain
            out.append((a, gain, f"take {', '.join(names)} (gain {gain:+.3f}"
                                 + (f", ${blind_dollars:.1f} unpriced-by-proxy" if blind_dollars else "") + ")"))
        out.append(({"type": "skip_booster"}, 0.0, "skip the pack"))
        out.sort(key=lambda x: (-x[1], _hand._action_sort_key(x[0])))
        return out


# ═══════════════════════════════════════════════════════ the PvP turn protocol (W-PVP)

#: `HandConfig` overrides that turn the level-1 Nemesis objective, the leader's PASS and
#: the decided-race extraction gate ON together.  See `ev/PVP_NOTES.md`.
PVP_PROTOCOL_HAND_CFG = dict(pvp_level1=True, pvp_pass=True, pvp_extract=True)


def protocol_hand_cfg(base: HandConfig = DEFAULT_HAND_CONFIG, *, level1: bool = True,
                      pvp_pass: bool = True, extract: bool = True) -> HandConfig:
    """``HandConfig`` for a player that plays under ``pvp_protocol="trailer_compelled"``.

    The three levers are separable on purpose — the attribution h2h in PVP_NOTES.md §8
    runs level-1 against level-0 with the protocol on for BOTH sides, which needs
    ``level1=False`` while ``pvp_pass``/``extract`` stay on."""
    return replace(base, pvp_level1=bool(level1), pvp_pass=bool(pvp_pass),
                   pvp_extract=bool(extract))


def adapt_match_player(player) -> Callable:
    """``player`` -> ``(match, p, acts) -> action``, the ``MLBMatch.play_out`` policy form.

    Same contract as ``eval/common.adapt_player`` with one addition: the match-level
    actions in ``acts`` that no ``BalatroGame`` knows about (``pvp_pass``) are handed to
    ``act`` as ``extra_actions``, so the player can choose to WAIT.  Use this instead of
    ``adapt_player`` whenever the match runs a non-canonical ``pvp_protocol``; with the
    canonical protocol ``acts`` never contains one and the two adapters are identical."""
    def pol(m, p, acts):
        # W-SHOP §2d: the shop's race-conditional aggression needs BOTH sides' lives and
        # Nemesis curves, which only the match has.  Binding it here (and nowhere else) is
        # what keeps a solo harness exactly neutral.
        bind = getattr(player, "bind_race", None)
        if bind is not None:
            bind(m, p)
        extra = [a for a in (acts or []) if a.get("type") == "pvp_pass"]
        return player.act(m.games[p], extra_actions=extra) if extra else player.act(m.games[p])
    return pol
