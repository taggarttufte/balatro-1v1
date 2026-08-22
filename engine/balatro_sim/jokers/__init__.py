"""
jokers/ — Joker effect implementations.

JOKER_REGISTRY maps GAME joker key (mp/rng/pools.py) -> singleton effect object.
Every one of the 150 game keys has exactly one implementation; registering a
key twice raises at import (see base._JokerRegistry), and the check at the
bottom of this file raises if any registry key is not a game key or any game
key is missing. Phase 1 W1 removed the old "later import wins" duplicates —
see mp/engine/REKEY_NOTES.md for every decision.

Trigger points:
  pre_score(context)               — fires before card loop (retrigger setup, flags)
  on_score_card(card, context)     — fires once per scoring card
  on_hand_scored(context)          — fires after all cards scored
  on_discard(cards, context)       — fires when player discards
  on_round_end(context)            — fires at end of round (cash out)
  on_blind_selected(context)       — fires when blind is selected
  on_boss_beaten(context)          — fires when boss blind is beaten
  on_planet_used(planet, context)  — fires when planet card is used
  on_tarot_used(context)           — fires when tarot card is used
  on_sell(context)                 — fires when this joker is sold
  on_shop_enter(context)           — fires when shop is entered
  on_shop_leave(context)           — fires when shop is left (Perkeo)
  on_card_destroyed(card, context) — fires when a card is destroyed
  on_card_added(context)           — fires when a playing card is added to the deck (Hologram)
  on_blind_skipped(context)        — fires when a blind is skipped
  on_held_card(card, context)      — held-in-hand `individual` effect, per pass; returns True if it did anything
  on_reroll(context)               — shop reroll (Flash Card)
  on_booster_opened(context)       — a booster pack is opened (Hallucination)
  on_card_sold(context)            — any card sold (Campfire); on_sell fires on the sold joker itself
  on_boss_ability_triggered(context) — the boss rejected/punished a hand (Matador); Matador also reads ctx.boss_triggered
  on_init(context)                 — Card:set_ability on acquire (To Do List draw, Popcorn/Ramen/Ice Cream start values)

Phase 1 W3: rolls use the game's key strings on ctx.prng (base.prob_roll / rng_of);
created cards come from mp/rng/generate.py (base.create_consumable); the
non-scoring hooks are fired through base.fire_hook(game, name) — see
mp/engine/EFFECTS_NOTES.md.
"""
from .base import JOKER_REGISTRY, JokerInstance, register_joker

# Each module owns a disjoint set of keys (duplicates raise).
from . import economy     # noqa: F401  — money jokers
from . import scaling     # noqa: F401  — persistent state jokers
from . import hand_type   # noqa: F401  — hand-type conditional jokers
from . import misc        # noqa: F401  — retrigger, blueprint, special mechanics
from . import chips       # noqa: F401  — flat bonus jokers
from . import mult        # noqa: F401  — mult/xMult jokers (canonical, loads last)


def _check_registry_against_pools() -> None:
    from ..game_keys import JOKER_BY_KEY
    extra = sorted(set(JOKER_REGISTRY) - set(JOKER_BY_KEY))
    missing = sorted(set(JOKER_BY_KEY) - set(JOKER_REGISTRY))
    if extra or missing:
        raise RuntimeError(
            "JOKER_REGISTRY is out of sync with mp/rng/pools.py: "
            f"not game keys={extra}; unimplemented game keys={missing}"
        )


_check_registry_against_pools()
