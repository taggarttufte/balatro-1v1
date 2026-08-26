"""
hit.py -- "is this a hit for my build, and how likely am I to see one" for the W4
decision-statistics module (Phase 5 rev 2, docs/PHASE5_BRIEF_2026-08.md).

Two valuation regimes, by design (STATS_NOTES.md has the full rationale):

1. **Known cards** (the shelf, an open pack, the voucher slot) -- there is nothing left to
   average over, so a joker gets the PRECISE treatment: a side-effect-free dry run
   (``_score_with_jokers``, the same pattern as ``card_selection.HypotheticalScorer``) scores
   a handful of sampled hands with and without the candidate, and the delta becomes a $
   uplift. Consumables / playing cards / vouchers get small documented flat tables (dry-
   running a Tarot's card-transform effect or a Voucher's run-long modifier is out of scope
   for a sub-50ms shop decision; see STATS_NOTES.md "known simplifications").

2. **Unknown cards** (what a reroll or an unopened pack MIGHT show) -- there are up to 150
   jokers / 22 tarots / 12 planets / 18 spectrals behind that curtain, an order of magnitude
   too many to dry-run every shop visit. ``pool_dollar_value`` is a FAST, cheap proxy (a
   handful of dict lookups) built from ``synergy.py``'s existing ``estimate_joker_strength`` /
   ``coherence_score`` -- exactly what the brief calls "synergy tags ... as a cheap prior".
   ``is_hit`` thresholds that proxy; P(hit) then comes from ``rng/generate.get_current_pool``
   -- the SAME function the real game/generator uses to cull a pool -- called with an explicit
   (non-rolled) rarity so it consumes ZERO RNG and is safe to call on the live ``run_state``
   without cloning (``get_current_pool``'s only RNG read is the rarity roll, which an explicit
   ``_rarity`` skips; see ``generate.py:714-804``). We still clone defensively (cheap; a few
   primitive fields + small sets) so a bug in that assumption can never leak into the game.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Optional

import _bootstrap  # noqa: F401
from balatro_sim import game_keys as _gk
from balatro_sim.hand_eval import evaluate_hand
from balatro_sim.scoring import score_hand
from balatro_sim.jokers.base import JokerInstance
from balatro_sim.synergy import estimate_joker_strength, coherence_score
from balatro_sim.consumables import TAROT_NAME, SPECTRAL_NAME, PLANET_HAND

_gen = _gk.gen           # rng.generate, the SAME module instance balatro_sim wires against
_pools = _gk.pools        # rng.pools

# ══════════════════════════════════════════════════════════════════ config

@dataclass(frozen=True)
class StatsConfig:
    # -- joker valuation --
    n_hand_samples: int = 6                 # dry-run samples for a SHELF joker's precise value
    dollars_per_score_ratio: float = 15.0    # $ anchor: value of moving 100% of the way to the
                                              # next blind's chip target in ONE hand (see hit_value)
    dollars_per_strength_point: float = 2.2  # $/point for the FAST pool-wide proxy (synergy.py
                                              # estimate_joker_strength is 1..10)
    pool_hit_threshold_dollars: float = 8.0  # a pool member is a "hit" iff pool_dollar_value >= this
    anti_synergy_floor: float = 0.3          # coherence floor so an anti-synergy joker isn't valued at 0
    # -- consumables / cards / vouchers (flat tables) --
    tarot_base_value: float = 4.0
    spectral_base_value: float = 5.0
    soul_black_hole_value: float = 30.0      # c_soul / c_black_hole: legendary-joker-tier, special-cased
    planet_base_value: float = 6.0
    voucher_cost_multiplier: float = 1.3     # default voucher value = cost * this
    card_base_value: float = 1.0
    enhancement_card_value: dict = field(default_factory=lambda: {
        "None": 0.0, "Bonus": 2.0, "Mult": 2.5, "Wild": 2.0, "Glass": 3.0,
        "Steel": 4.5, "Stone": 1.0, "Gold": 3.5, "Lucky": 3.0,
    })
    edition_card_bonus: dict = field(default_factory=lambda: {
        "None": 0.0, "Foil": 1.0, "Holographic": 2.0, "Polychrome": 3.5, "Negative": 3.5,
    })
    # -- sampling --
    rng_seed_salt: int = 0


DEFAULT = StatsConfig()

# Hand-curated voucher standouts (documented prior, not exhaustive -- STATS_NOTES.md).
_VOUCHER_STANDOUT: dict[str, float] = {
    "v_blank": 0.5,
    "v_overstock_norm": 10.0, "v_overstock_plus": 14.0,
    "v_clearance_sale": 8.0, "v_liquidation": 10.0,
    "v_reroll_surplus": 9.0, "v_reroll_glut": 11.0,
    "v_telescope": 16.0, "v_observatory": 18.0,
    "v_hieroglyph": 14.0, "v_petroglyph": 16.0,   # banned under MLB but harmless to price
    "v_seed_money": 12.0, "v_money_tree": 14.0,
    "v_crystal_ball": 15.0, "v_omen_globe": 16.0,
    "v_grabber": 12.0, "v_nacho_tong": 14.0,
    "v_wasteful": 6.0, "v_recyclomancy": 10.0,
    "v_tarot_merchant": 13.0, "v_tarot_tycoon": 20.0,
    "v_planet_merchant": 13.0, "v_planet_tycoon": 20.0,
    "v_directors_cut": 14.0, "v_retcon": 18.0,     # banned under MLB (constants.MLB_BANNED_VOUCHERS)
}


# ══════════════════════════════════════════════════════════════════ dry-run scorer (shelf jokers)

def _clone_joker(j: JokerInstance) -> JokerInstance:
    """Same shape as ``card_selection._clone_joker_for_dry_run`` (one level of container
    copying is enough for every joker state shape in the registry)."""
    new = j.clone()
    new.state = {k: (v.copy() if isinstance(v, (list, set, dict)) else v) for k, v in j.state.items()}
    return new


def _score_with_jokers(game, rng, rng_snapshot, cards: list, extra_joker: Optional[JokerInstance]) -> int:
    """Side-effect-free ``score_hand`` over ``cards`` (a synthetic sample, not ``game.hand``)
    with the game's owned jokers plus an optional candidate.  Mirrors
    ``card_selection.HypotheticalScorer.score`` with ``model_held=False`` (no held-in-hand /
    full-deck hooks -- a sampled hand has no "held" complement by construction) and an extra
    joker slot HypotheticalScorer does not offer (it can only score against ``gs.jokers``)."""
    rng.restore(rng_snapshot)
    flags = game.hand_eval_flags() if hasattr(game, "hand_eval_flags") else {}
    hand_type, scoring = evaluate_hand(cards, **flags)
    jokers = [_clone_joker(j) for j in game.jokers]
    if extra_joker is not None:
        jokers.append(_clone_joker(extra_joker))
    score, _ctx = score_hand(
        scoring_cards=scoring, all_cards=cards, hand_type=hand_type, jokers=jokers,
        planet_levels=dict(game.planet_levels), hands_left=game.hands_left,
        discards_left=game.discards_left, dollars=game.dollars, ante=game.ante,
        deck_remaining=len(game.deck), rng=rng, held_cards=None, full_deck=None,
        run_state=None, probabilities_normal=1.0, round_cards={},
        joker_slots=game.joker_slots, consumable_slots=game.consumable_slots,
        consumables=game.consumable_hand, hands_played=getattr(game, "_hands_played_round", 0),
        plasma=getattr(game, "plasma", False),
    )
    return score


def _seed_for(game, *parts) -> int:
    """Deterministic, process-independent seed (``crc32``, not ``hash()``, which is salted
    per-process for strings) -- so ``decision_table`` gives the same numbers if called twice
    on an equal state, and a multiprocess sweep is reproducible."""
    key = "|".join(str(p) for p in (game.seed_str, game.ante, game.blind_idx, len(game.full_deck), *parts))
    return zlib.crc32(key.encode())


def sample_hands(game, n_samples: int, n_cards: int = 5, seed_extra: str = "") -> list[list]:
    """``n_samples`` hands of ``n_cards`` drawn from ``game.full_deck``'s COMPOSITION (never
    its order -- a fresh ``random.Random``, never ``game.run_state.rng``) -- "a representative
    sample of hands drawn from full_deck" per the brief.  Copies so a scoring hook can never
    touch the live deck."""
    import random
    deck = game.full_deck
    if not deck:
        return []
    n = min(n_cards, len(deck))
    rng = random.Random(_seed_for(game, seed_extra, n_samples, n_cards))
    return [[c.copy() for c in rng.sample(deck, n)] for _ in range(n_samples)]


def sample_hand_scores(game, n_samples: int, n_cards: int = 5, seed_extra: str = "",
                       extra_joker: Optional[JokerInstance] = None) -> list[float]:
    """Score ``n_samples`` sampled hands against the owned jokers (+ ``extra_joker`` if
    given). Shared by ``joker_hit_value`` (owned+candidate vs owned) and ``urgency.py``
    (owned only, for a build-strength estimate)."""
    hands = sample_hands(game, n_samples, n_cards, seed_extra)
    if not hands:
        return []
    rng = game.run_state.rng.clone()
    snap = rng.snapshot()
    return [float(_score_with_jokers(game, rng, snap, cards, extra_joker)) for cards in hands]


def joker_hit_value(game, key: str, edition: str = "None", cfg: StatsConfig = DEFAULT) -> float:
    """Precise $-equivalent value of buying joker ``key`` RIGHT NOW: expected score uplift
    over ``cfg.n_hand_samples`` sampled hands, converted to $ via the SAME anchor
    ``urgency.py`` uses for "how much is one hand's worth of chips worth" (documented once,
    in one place: ``dollars_per_score_ratio`` $ per 100% of the next blind's chip target moved
    in a single hand) -- an ante-aware schedule BY CONSTRUCTION, since the chip target already
    scales with ante, without a second hand-typed $/point table. Scaling jokers are then
    scaled by ``synergy.SCALING_DECAY_BY_ANTE`` / ``IMMEDIATE_BONUS_BY_ANTE`` (the documented
    horizon assumption the brief asks for -- reused, not re-derived), and the joker's sell
    value is added (buying it is never worse than buying-then-selling)."""
    from balatro_sim.synergy import EARLY_GAME_JOKERS, IMMEDIATE_PAYOFF_JOKERS, \
        SCALING_DECAY_BY_ANTE, IMMEDIATE_BONUS_BY_ANTE

    candidate = JokerInstance(key, edition)
    without = sample_hand_scores(game, cfg.n_hand_samples, 5, seed_extra=f"base:{key}")
    with_j = sample_hand_scores(game, cfg.n_hand_samples, 5, seed_extra=f"base:{key}", extra_joker=candidate)
    if not without:
        uplift = 0.0
    else:
        uplift = (sum(with_j) - sum(without)) / len(without)

    target = _next_blind_target_for_value(game)
    raw = (uplift / max(1.0, target)) * cfg.dollars_per_score_ratio

    ante = max(1, min(8, game.ante))
    if key in EARLY_GAME_JOKERS:
        raw *= SCALING_DECAY_BY_ANTE[ante]
    elif key in IMMEDIATE_PAYOFF_JOKERS:
        raw *= IMMEDIATE_BONUS_BY_ANTE[ante]

    sell = max(1, _gk.JOKER_COST.get(key, 2) // 2)
    return max(0.0, raw) + float(sell)


def _next_blind_target_for_value(game) -> float:
    """The chip anchor ``joker_hit_value`` / ``pool_dollar_value`` normalise by -- the SAME
    "next blind" ``urgency.next_blind_info`` computes, imported lazily to avoid a cycle
    (``urgency.py`` imports this module for ``sample_hand_scores``)."""
    import urgency as _u
    ante, blind_idx = _u.next_blind_info(game)
    from balatro_sim.constants import blind_base_chips
    return float(blind_base_chips(ante, blind_idx, game.blind_scaling) * game.ante_scaling)


# ══════════════════════════════════════════════════════════════════ fast pool-wide proxy

def pool_dollar_value(key: str, owned_keys: list[str], ante: int, cfg: StatsConfig = DEFAULT) -> float:
    """Cheap $-equivalent for a joker NOT on the shelf (a pool member behind a reroll / pack):
    ``synergy.estimate_joker_strength`` (1..10, hand-tuned tier list) x a coherence multiplier
    (``synergy.coherence_score``, floored so an anti-synergy pick still has SOME value) x a
    documented $/point rate. O(1) dict lookups -- this is what makes summing it over a
    150-joker pool affordable inside a 50ms budget."""
    strength = estimate_joker_strength(key)
    coh = coherence_score(key, owned_keys, ante)
    mult = cfg.anti_synergy_floor + (1.0 - cfg.anti_synergy_floor) * coh
    return strength * mult * cfg.dollars_per_strength_point


def is_hit(key: str, owned_keys: list[str], ante: int, cfg: StatsConfig = DEFAULT) -> bool:
    return pool_dollar_value(key, owned_keys, ante, cfg) >= cfg.pool_hit_threshold_dollars


# ══════════════════════════════════════════════════════════════════ P(hit) — exact from the generator's pools

# rarity_from_roll (generate.py:709-711): r>0.95 Rare (5%), 0.7<r<=0.95 Uncommon (25%), else Common (70%).
_RARITY_PROBE = {1: (0.5, 0.70), 2: (0.85, 0.25), 3: (0.99, 0.05)}   # rarity_n -> (probe r, P(rarity))


@dataclass
class PoolSlice:
    """One (type[, rarity]) component of a shop slot / pack card's distribution."""
    label: str            # "Joker (Common)", "Tarot", ...
    p_component: float    # P this component is what gets drawn (shop_type_table weight x rarity P)
    pool: list             # culled pool (game keys; UNAVAILABLE entries already stripped)
    hits: list             # pool members classified as hits
    p_hit_given_component: float
    mean_hit_value: float


def _culled_pool(rs, _type: str, _rarity: Optional[float], ante: int) -> list[str]:
    pool, _pool_key, _rarity_n = _gen.get_current_pool(rs, _type, _rarity=_rarity, ante=ante)
    return [k for k in pool if k != _gen.UNAVAILABLE]


def _joker_slices(rs, owned_keys: list[str], ante: int, type_p: float, cfg: StatsConfig) -> list[PoolSlice]:
    slices = []
    for rarity_n, (probe, p_rarity) in _RARITY_PROBE.items():
        pool = _culled_pool(rs, "Joker", probe, ante)
        if not pool:
            continue
        hits = [k for k in pool if is_hit(k, owned_keys, ante, cfg)]
        mean_val = (sum(pool_dollar_value(k, owned_keys, ante, cfg) for k in hits) / len(hits)) if hits else 0.0
        slices.append(PoolSlice(
            label=f"Joker ({_gk.RARITY_NAME[rarity_n]})", p_component=type_p * p_rarity,
            pool=pool, hits=hits, p_hit_given_component=len(hits) / len(pool), mean_hit_value=mean_val,
        ))
    return slices


def _consumable_slice(rs, gtype: str, label: str, ante: int, type_p: float,
                      value_fn) -> Optional[PoolSlice]:
    pool = _culled_pool(rs, gtype, None, ante)
    if not pool:
        return None
    hits = [k for k in pool if value_fn(k) > 0]     # every eligible consumable in these small
                                                     # pools (<=22) "hits" iff it has positive
                                                     # documented value (planets: play_share > 0)
    mean_val = (sum(value_fn(k) for k in hits) / len(hits)) if hits else 0.0
    return PoolSlice(label=label, p_component=type_p, pool=pool, hits=hits,
                     p_hit_given_component=len(hits) / len(pool), mean_hit_value=mean_val)


def shop_slot_distribution(game, cfg: StatsConfig = DEFAULT, rs=None) -> list[PoolSlice]:
    """Every ``PoolSlice`` a FRESH shop slot can land on, in the shop's own rate table
    (``run_state.joker_rate`` / ``tarot_rate`` / ... -- ``generate.shop_type_table``, honours
    Illusion / Magic Trick / Ghost rate changes automatically since we read the live
    ``run_state``). Zero RNG consumed (every ``get_current_pool`` call below passes an
    explicit, non-rolled rarity). ``rs``: a caller-supplied ``run_state.clone()`` so one
    ``decision_table`` call can share a single clone across the reroll row and every pack
    row instead of cloning per row (``decide.py``); defaults to a fresh clone."""
    if rs is None:
        rs = game.run_state.clone()
    ante = rs.ante
    owned_keys = [j.key for j in game.jokers]
    rates = {"Joker": rs.joker_rate, "Tarot": rs.tarot_rate, "Planet": rs.planet_rate,
             "Spectral": rs.spectral_rate}   # playing-card rate omitted: no "hit" model for it
    total = sum(rates.values()) + rs.playing_card_rate
    if total <= 0:
        return []
    slices: list[PoolSlice] = []
    slices += _joker_slices(rs, owned_keys, ante, rates["Joker"] / total, cfg)
    tp = rates["Tarot"] / total
    if tp > 0:
        s = _consumable_slice(rs, "Tarot", "Tarot", ante, tp, lambda k: tarot_value(k, cfg))
        if s:
            slices.append(s)
    pp = rates["Planet"] / total
    if pp > 0:
        s = _consumable_slice(rs, "Planet", "Planet", ante, pp, lambda k: planet_value(game, k, cfg))
        if s:
            slices.append(s)
    sp = rates["Spectral"] / total
    if sp > 0:
        s = _consumable_slice(rs, "Spectral", "Spectral", ante, sp, lambda k: spectral_value(k, cfg))
        if s:
            slices.append(s)
    return slices


def p_hit_and_value(slices: list[PoolSlice]) -> tuple[float, float]:
    """``P(this draw is a hit)`` and ``E[value | hit]`` over one draw from the combined
    distribution -- shared by the reroll and pack-card computations below."""
    p_hit = sum(s.p_component * s.p_hit_given_component for s in slices)
    if p_hit <= 0:
        return 0.0, 0.0
    weighted_value = sum(s.p_component * s.p_hit_given_component * s.mean_hit_value for s in slices)
    return p_hit, weighted_value / p_hit


def reroll_p_hit(game, cfg: StatsConfig = DEFAULT, slices: Optional[list["PoolSlice"]] = None
                 ) -> tuple[float, float, dict]:
    """P(at least one of the ``shop_joker_max`` shelf slots is a hit after a reroll) and the
    expected value of the best hit found, plus a ``details`` dict for ``Row.details``.

    Independence approximation, documented: slots are treated as independent draws from the
    same distribution (``1 - (1 - p_slot)**n_slots``). The real generator dedupes across
    slots (a joker on slot 1 cannot also appear on slot 2 -- GENERATION_SPEC §6), which makes
    the TRUE P(>=1 hit) very slightly higher than this estimate when the pool is small and a
    hit is likely; ``STATS_NOTES.md`` validates the gap empirically (it is small: dedup only
    matters when the pool itself is comparable in size to ``shop_joker_max`` — 2, versus
    pools of 20-150)."""
    if slices is None:
        slices = shop_slot_distribution(game, cfg)
    p_slot, mean_hit_value = p_hit_and_value(slices)
    n_slots = game.run_state.shop_joker_max
    p_reroll = 1.0 - (1.0 - p_slot) ** n_slots
    details = {
        "p_slot": p_slot, "n_slots": n_slots, "mean_hit_value": mean_hit_value,
        "components": [(s.label, s.p_component, s.p_hit_given_component, len(s.pool), len(s.hits))
                       for s in slices],
    }
    return p_reroll, mean_hit_value, details


def pack_p_hit(game, pack_kind: str, size: int, cfg: StatsConfig = DEFAULT, rs=None
              ) -> tuple[float, float, dict]:
    """P(at least one of ``size`` revealed pack cards is a hit) and the expected value of the
    best hit, for one pack kind ("Arcana"/"Celestial"/"Spectral"/"Buffoon"/"Standard").
    Standard packs (playing cards) have no hit model here -- returns ``(0.0, 0.0, {})``
    (documented gap, STATS_NOTES.md). ``rs``: see ``shop_slot_distribution``."""
    if rs is None:
        rs = game.run_state.clone()
    ante = rs.ante
    owned_keys = [j.key for j in game.jokers]
    if pack_kind == "Buffoon":
        slices = _joker_slices(rs, owned_keys, ante, 1.0, cfg)
    elif pack_kind == "Arcana":
        slices = [s for s in [_consumable_slice(rs, "Tarot", "Tarot", ante, 1.0,
                              lambda k: tarot_value(k, cfg))] if s]
    elif pack_kind == "Celestial":
        slices = [s for s in [_consumable_slice(rs, "Planet", "Planet", ante, 1.0,
                              lambda k: planet_value(game, k, cfg))] if s]
    elif pack_kind == "Spectral":
        slices = [s for s in [_consumable_slice(rs, "Spectral", "Spectral", ante, 1.0,
                              lambda k: spectral_value(k, cfg))] if s]
    else:
        return 0.0, 0.0, {}
    p_card, mean_hit_value = p_hit_and_value(slices)
    p_pack = 1.0 - (1.0 - p_card) ** max(1, size)
    details = {"p_card": p_card, "size": size, "mean_hit_value": mean_hit_value,
              "components": [(s.label, s.p_component, s.p_hit_given_component, len(s.pool), len(s.hits))
                            for s in slices]}
    return p_pack, mean_hit_value, details


# ══════════════════════════════════════════════════════════════════ consumables / cards / vouchers (flat tables)

def tarot_value(key: str, cfg: StatsConfig = DEFAULT) -> float:
    """Flat table (brief: "simple documented tables"). The Soul is a Spectral card in this
    engine (``c_soul``), not drawable as a Tarot, so no special case is needed here."""
    return cfg.tarot_base_value if key in TAROT_NAME else 0.0


def spectral_value(key: str, cfg: StatsConfig = DEFAULT) -> float:
    if key in ("c_soul", "c_black_hole"):
        return cfg.soul_black_hole_value
    return cfg.spectral_base_value if key in SPECTRAL_NAME else 0.0


_PLANET_SMOOTHING_PSEUDOCOUNT = 2.0   # Laplace smoothing: 1-2 hands played this run must not
                                      # swing a hand type's play-share from 1/12 to ~1.0


def planet_value(game, key: str, cfg: StatsConfig = DEFAULT) -> float:
    """"level-up of a hand type weighted by how often the build plays it" (brief). Play share
    comes from ``game._hand_type_counts`` (the run's own tally), Laplace-smoothed against a
    uniform 1/12 prior so a handful of early-game hands (the common case in a shop) cannot
    spike one hand type's share to ~1.0 and its value to 12x the base rate -- with ZERO hands
    played every type gets exactly the uniform share (``share * 12 == 1``), and the value
    converges to the true empirical share as more hands accumulate. Diminishing by current
    level -- a hand already levelled 5x is a smaller marginal gain than a fresh level 1. Capped
    at 3x the base rate so even a heavily concentrated build's dominant hand type does not
    dwarf every other row in the table."""
    hand = PLANET_HAND.get(key)
    if hand is None:
        return 0.0
    counts = getattr(game, "_hand_type_counts", None) or {}
    n = len(PLANET_HAND)
    total = sum(counts.values())
    p = _PLANET_SMOOTHING_PSEUDOCOUNT
    share = (counts.get(hand, 0) + p) / (total + p * n)
    level = game.planet_levels.get(hand, 1)
    raw = cfg.planet_base_value * share * n / max(1, level)
    return min(raw, cfg.planet_base_value * 3.0 / max(1, level))


def voucher_value(key: str, cfg: StatsConfig = DEFAULT) -> float:
    if key in _VOUCHER_STANDOUT:
        return _VOUCHER_STANDOUT[key]
    return cfg.voucher_cost_multiplier * float(_gk.VOUCHER_COST.get(key, 10))


def card_value(item, cfg: StatsConfig = DEFAULT) -> float:
    """A shelf/pack playing card (Standard-pack contents, Magic Trick shelf cards): flat base
    + enhancement + edition table, no build-fit modelling (documented gap)."""
    enh = getattr(item, "enhancement", "None") or "None"
    ed = getattr(item, "edition", "None") or "None"
    return (cfg.card_base_value + cfg.enhancement_card_value.get(enh, 0.0)
           + cfg.edition_card_bonus.get(ed, 0.0))


__all__ = [
    "StatsConfig", "DEFAULT", "sample_hands", "sample_hand_scores", "joker_hit_value",
    "pool_dollar_value", "is_hit", "shop_slot_distribution", "p_hit_and_value",
    "reroll_p_hit", "pack_p_hit", "tarot_value", "spectral_value", "planet_value",
    "voucher_value", "card_value", "PoolSlice",
]
