"""
scoring.py — Chip x mult scoring engine.

Scoring order (mirrors functions/state_events.lua `evaluate_play`, 1.0.1o):
  1. `before` phase: jokers' pre_score (retrigger setup, hand-eval flags, Blueprint
     copies, and Space Joker's level-up — which the real game applies BEFORE the
     base chips/mult are read, so the current hand scores at the new level).
  2. Base chips + base mult from hand type at its (possibly just raised) level.
     running_mult starts at base mult.
  3. For each scoring card (in play order), for each pass (1 + Red seal + joker
     retriggers):
     a. Card base chips (+ Hiker's permanent bonus)
     b. Enhancement effects (Bonus +30, Mult +4, Glass x2, Lucky: `lucky_mult`
        roll < normal/5 -> +20 Mult, then `lucky_money` roll < normal/15 -> $20)
     c. Edition effects on card (Foil +50 chips, Holo +10 mult, Poly x1.5 mult)
     d. Seal effects that fire on score (Gold seal $3)
     e. FOLD
     f. Each joker fires on_score_card(card, ctx) — `individual` context —, FOLD
        (Lucky Cat reads ctx.lucky_trigger here, like `other_card.lucky_trigger`)
  4. Held-in-hand phase, hand order, per card per pass: Steel x1.5 (FOLD), then each
     joker's on_held_card (Baron, Shoot the Moon, Raised Fist, Reserved Parking —
     FOLD each). A held card that produced ANY effect on its first pass gets one
     extra pass per Red seal and one per Mime (state_events.lua:798-870).
  5. `joker_main`: for each joker left to right — its Foil/Holo edition (FOLD), its
     on_hand_scored (FOLD), its Polychrome (FOLD). Editions apply exactly ONCE per
     joker here, never on per-card passes (state_events.lua:876-947).
  6. `final_scoring_step` (state_events.lua:946-948): the deck's
     `Back:trigger_effect{context='final_scoring_step', chips, mult}` — only Plasma does
     anything (back.lua:121-128): `tot = chips + mult; chips = floor(tot/2);
     mult = floor(tot/2)`.  It sees the FULLY modified chips and mult (every card, held
     card, joker and joker-edition effect has already folded in) and runs BEFORE the
     Glass/destroy pass and the final product.  `plasma=True` enables it.
  7. Final score = floor(chips * mult)  (`math.floor(hand_chips*mult)`, :1031/:1046).

FOLD means `running_mult = (running_mult + ctx.mult) * ctx.mult_mult`, then reset
the pending pair. Folding after every individual contributor is what makes joker
ORDER matter.

Glass shatter is NOT rolled here: the real game rolls `glass` once per scoring
Glass card after the whole hand has scored (state_events.lua:951-963); the caller
(game._play_hand) does that over ctx.glass_scored.

RNG: every roll goes through the keyed PseudoRandom on ctx.prng (`rng=` argument;
game.run_state.rng for real play, a private clone in HypotheticalScorer). There is
no unseeded fallback — a roll without a PRNG raises jokers.base.MissingPRNG.
"""
import math

from .card import Card
from .constants import HAND_BASE, HAND_LEVEL_CHIPS, HAND_LEVEL_MULT
from .jokers.base import (
    JokerInstance, ScoreContext, JOKER_REGISTRY, is_prng, prob_roll, rng_of,
)


def _fire_joker_on_card(joker: JokerInstance, ctx: ScoreContext, card: Card):
    """One joker's `individual` effect for one scoring-card pass, then fold.
    No edition here — editions pay out once per joker in the joker_main phase."""
    joker.on_score_card(card, ctx)
    ctx.fold_mult()


def _score_single_card(card: Card, ctx: ScoreContext, jokers: list[JokerInstance]):
    """Score one card pass (used for base scoring + each retrigger)."""
    ctx.chips += card.base_chips + getattr(card, "bonus_chips", 0)
    ctx.lucky_trigger = False

    # Enhancement effects
    if card.enhancement == "Bonus":
        ctx.chips += 30
    elif card.enhancement == "Mult":
        ctx.mult += 4
    elif card.enhancement == "Glass":
        ctx.mult_mult *= 2.0
        # Shatter is rolled by the caller AFTER all scoring, once per scoring
        # Glass card (not per pass) — state_events.lua:957-963.
        if card not in ctx.glass_scored:
            ctx.glass_scored.append(card)
    elif card.enhancement == "Lucky":
        # card.lua:988 then :1076 — two independent keyed rolls, in this order,
        # on every pass (retriggers roll again).
        if prob_roll(ctx, "lucky_mult", 5):
            ctx.mult += 20
            ctx.lucky_trigger = True
        if prob_roll(ctx, "lucky_money", 15):
            ctx.pending_money += 20
            ctx.lucky_trigger = True
    # Steel is "X1.5 Mult while held in hand" and contributes nothing when
    # played — see the held-in-hand phase in score_hand().

    # Edition effects on card
    if card.edition == "Foil":
        ctx.chips += 50
    elif card.edition == "Holographic":
        ctx.mult += 10
    elif card.edition == "Polychrome":
        ctx.mult_mult *= 1.5

    # Gold Seal: "$3 when this card is played and scores" — retriggers pay again.
    if card.seal == "Gold":
        ctx.pending_money += 3

    # The card's own modifiers are one contributor — commit them before jokers.
    ctx.fold_mult()

    # Jokers: `individual` context, left to right, each folded separately.
    for joker in jokers:
        _fire_joker_on_card(joker, ctx, card)
    ctx.lucky_trigger = False


def _n_mime_reps(jokers: list[JokerInstance]) -> int:
    """Extra held-card passes granted by Mime (+1 each), including a Blueprint
    whose right neighbour is Mime and a Brainstorm whose leftmost joker is Mime
    (Mime has no `not context.blueprint` guard — card.lua:2879)."""
    n = 0
    for i, j in enumerate(jokers):
        if j.key == "j_mime":
            n += 1
        elif j.key == "j_blueprint" and i + 1 < len(jokers) and jokers[i + 1].key == "j_mime":
            n += 1
        elif j.key == "j_brainstorm" and jokers and jokers[0].key == "j_mime" and jokers[0] is not j:
            n += 1
    return n


def _held_phase(ctx: ScoreContext, jokers: list[JokerInstance]):
    """Held-in-hand effects, state_events.lua:798-870 (see module docstring, step 4)."""
    mime_reps = _n_mime_reps(jokers)
    for card in ctx.held_cards:
        reps = 1
        j = 0
        while j < reps:
            had_effect = False
            if not card.debuffed and card.enhancement == "Steel":
                ctx.mult_mult *= 1.5
                had_effect = True
            ctx.fold_mult()
            for joker in jokers:
                if joker.on_held_card(card, ctx):
                    had_effect = True
                ctx.fold_mult()
            if j == 0 and had_effect:
                if card.seal == "Red" and not card.debuffed:
                    reps += 1
                reps += mime_reps
            j += 1


def _joker_main_phase(ctx: ScoreContext, jokers: list[JokerInstance]):
    """`joker_main` with editions, state_events.lua:876-947: Foil/Holo before the
    joker's own effect, Polychrome after, each exactly once per joker."""
    for joker in jokers:
        if joker.edition == "Foil":
            ctx.chips += 50
            ctx.fold_mult()
        elif joker.edition == "Holographic":
            ctx.mult += 10
            ctx.fold_mult()
        joker.on_hand_scored(ctx)
        ctx.fold_mult()
        if joker.edition == "Polychrome":
            ctx.mult_mult *= 1.5
            ctx.fold_mult()
        # Negative grants a joker slot and has no scoring effect.


def score_hand(
    scoring_cards: list[Card],
    all_cards: list[Card],
    hand_type: str,
    jokers: list[JokerInstance],
    planet_levels: dict[str, int],
    hands_left: int,
    discards_left: int,
    dollars: int,
    ante: int,
    deck_remaining: int,
    rng=None,
    held_cards: list[Card] | None = None,
    full_deck: list[Card] | None = None,
    hand_type_counts: dict | None = None,
    *,
    run_state=None,
    probabilities_normal: float | None = None,
    round_cards: dict | None = None,
    joker_slots: int = 5,
    consumable_slots: int = 2,
    consumables: list | None = None,
    boss_triggered: bool = False,
    hands_played: int = 0,
    plasma: bool = False,
) -> tuple[int, ScoreContext]:
    """
    Compute the total score for a played hand.

    `rng` must be the run's keyed PseudoRandom (game.run_state.rng) or an object
    with the same three methods (pseudorandom / pseudorandom_element /
    pseudoshuffle). It may be None only for hands that make no roll at all — the
    first roll then raises MissingPRNG.

    Returns:
        (int score, ScoreContext) — score is chips*mult floored; ctx holds
        side-effects like pending_money, pending_consumables and prevent_loss.
    """
    if rng is not None and not is_prng(rng):
        raise TypeError(
            "score_hand(rng=...) needs a keyed PseudoRandom (game.run_state.rng); "
            f"got {type(rng).__name__}. The legacy random.Random stream is gone (W3).")
    if probabilities_normal is None:
        probabilities_normal = getattr(run_state, "probabilities_normal", 1.0) if run_state is not None else 1.0

    ctx = ScoreContext(
        chips=0.0,
        mult=0.0,
        mult_mult=1.0,
        hand_type=hand_type,
        scoring_cards=scoring_cards,
        all_cards=all_cards,
        jokers=jokers,
        hands_left=hands_left,
        discards_left=discards_left,
        dollars=dollars,
        ante=ante,
        deck_remaining=deck_remaining,
        planet_levels=planet_levels,
        prng=rng,
        run_state=run_state,
        probabilities_normal=float(probabilities_normal),
        round_cards=round_cards if round_cards is not None else {},
        joker_slots=joker_slots,
        consumable_slots=consumable_slots,
        consumables=list(consumables) if consumables else [],
        boss_triggered=boss_triggered,
        hands_played=hands_played,
        held_cards=list(held_cards) if held_cards else [],
        full_deck=list(full_deck) if full_deck else [],
        hand_type_counts=dict(hand_type_counts) if hand_type_counts else {},
    )

    # 1. `before` phase — flags, retrigger counts, Blueprint copies, Space Joker.
    for joker in jokers:
        effect = JOKER_REGISTRY.get(joker.key)
        if effect and hasattr(effect, "pre_score"):
            effect.pre_score(joker, ctx)

    # 2. Base chips/mult at the hand's current level (after Space Joker).
    base_chips, base_mult = HAND_BASE.get(hand_type, (5, 1))
    level = ctx.planet_levels.get(hand_type, 1)
    if level > 1:
        base_chips += HAND_LEVEL_CHIPS.get(hand_type, 0) * (level - 1)
        base_mult += HAND_LEVEL_MULT.get(hand_type, 0) * (level - 1)
    ctx.running_mult = float(base_mult)

    # 3. Score each card + retriggers
    for i, card in enumerate(scoring_cards):
        if card.debuffed:
            continue
        _score_single_card(card, ctx, jokers)          # base pass
        if card.seal == "Red":                         # Red seal: retrigger once
            _score_single_card(card, ctx, jokers)
        for _ in range(ctx.card_retriggers.get(i, 0)): # Hack, Sock and Buskin, Hanging Chad, Dusk, Seltzer
            _score_single_card(card, ctx, jokers)

    # 4. Held-in-hand phase
    _held_phase(ctx, jokers)

    # 5. joker_main with editions
    _joker_main_phase(ctx, jokers)

    # Fold anything a joker left pending without triggering a fold.
    ctx.fold_mult()

    total_chips = base_chips + ctx.chips
    mult = max(ctx.running_mult, 0)
    # 6. final_scoring_step — Plasma Deck balance (back.lua:121-128), after every joker
    #    effect and before the product.  Lua `math.floor` on the (possibly fractional)
    #    sum; both halves get the same floored value, so the score is floor(tot/2)^2
    #    exactly (the product of two equal integers needs no further floor).
    if plasma:
        tot = total_chips + mult
        total_chips = math.floor(tot / 2)
        mult = math.floor(tot / 2)
        ctx.chips = total_chips - base_chips      # keep ctx's view consistent for observers
        ctx.running_mult = float(mult)
    score = int(math.floor(total_chips * mult))
    return score, ctx
