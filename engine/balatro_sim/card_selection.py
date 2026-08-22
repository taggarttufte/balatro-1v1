"""
card_selection.py — Subset enumeration and intent-to-action translation for V7.

The V7 agent picks an intent (PLAY / DISCARD / USE_CONSUMABLE) and then selects
a card subset via learned card scores. This module handles:

1. Enumerating all valid 1-5 card subsets (cached by hand size)
2. Scoring subsets from per-card scores (play = sum of scores, discard = sum of 1-scores)
3. Validating subsets against boss blind restrictions
4. Converting (intent, subset) into game.step()-compatible action dicts
"""
from __future__ import annotations

import itertools
from functools import lru_cache
from typing import Optional

import numpy as np

from .card import Card
from .hand_eval import evaluate_hand
from .scoring import score_hand

# ════════════════════════════════════════════════════════════════════════════
# Intent constants
# ════════════════════════════════════════════════════════════════════════════

INTENT_PLAY = 0
INTENT_DISCARD = 1
# One intent per consumable slot. Before the 2026-07-30 fix there was a single
# USE_CONSUMABLE intent hardcoded to slot 0, so slot 1 was unreachable for the
# entire history of the project.
INTENT_USE_CONSUMABLE_0 = 2
INTENT_USE_CONSUMABLE_1 = 3
N_INTENTS = 4

# Kept so older code referring to the single intent still resolves to slot 0.
# New code should use CONSUMABLE_INTENTS / consumable_slot().
INTENT_USE_CONSUMABLE = INTENT_USE_CONSUMABLE_0

CONSUMABLE_INTENTS = (INTENT_USE_CONSUMABLE_0, INTENT_USE_CONSUMABLE_1)

INTENT_NAMES = {
    INTENT_PLAY: "PLAY",
    INTENT_DISCARD: "DISCARD",
    INTENT_USE_CONSUMABLE_0: "USE_CONSUMABLE[0]",
    INTENT_USE_CONSUMABLE_1: "USE_CONSUMABLE[1]",
}


# Intents whose card subset is meaningful and must therefore be sampled:
# PLAY/DISCARD pick cards to act on, and the consumable intents pick the card
# TARGET. Omitting the consumable intents here silently reproduces A1 — the agent
# would emit no subset and the target list would be empty again.
SUBSET_INTENTS = (INTENT_PLAY, INTENT_DISCARD) + CONSUMABLE_INTENTS


def consumable_slot(intent: int) -> Optional[int]:
    """Consumable slot an intent refers to, or None if it isn't a consumable intent."""
    if intent in CONSUMABLE_INTENTS:
        return intent - INTENT_USE_CONSUMABLE_0
    return None


# ════════════════════════════════════════════════════════════════════════════
# Subset enumeration (cached)
# ════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=16)
def enumerate_subsets(n_cards: int) -> list[tuple[int, ...]]:
    """
    All 1-5 card subsets of n_cards, as tuples of indices.
    Cached by n_cards (only 1-8 possible values).

    Returns subsets sorted by size then lexicographic order.

    Examples:
        enumerate_subsets(3) -> [(0,), (1,), (2,), (0,1), (0,2), (1,2), (0,1,2)]
        len(enumerate_subsets(8)) -> 218
    """
    if n_cards <= 0:
        return []
    subsets = []
    for k in range(1, min(6, n_cards + 1)):
        for combo in itertools.combinations(range(n_cards), k):
            subsets.append(combo)
    return subsets


def subset_count(n_cards: int) -> int:
    """Number of valid 1-5 card subsets for n_cards cards."""
    return len(enumerate_subsets(n_cards))


# ════════════════════════════════════════════════════════════════════════════
# Subset scoring
# ════════════════════════════════════════════════════════════════════════════

def compute_subset_logits(card_scores: np.ndarray, subsets: list[tuple[int, ...]],
                          intent: int, temperature: float = 1.0) -> np.ndarray:
    """
    Compute logits for a Categorical distribution over subsets.

    For PLAY intent: logit = sum of card_scores for cards in subset
    For DISCARD intent: logit = sum of (1 - card_scores) for cards in subset
                        (high score = keep, so low score = discard)

    Args:
        card_scores: shape (n_cards,), values in [0, 1] from sigmoid
        subsets: list of index tuples from enumerate_subsets()
        intent: INTENT_PLAY or INTENT_DISCARD
        temperature: softmax temperature (lower = sharper distribution)

    Returns:
        logits: shape (n_subsets,), ready for softmax / Categorical
    """
    n_subsets = len(subsets)
    if n_subsets == 0:
        return np.zeros(0, dtype=np.float32)

    logits = np.zeros(n_subsets, dtype=np.float32)

    if intent == INTENT_PLAY:
        for i, subset in enumerate(subsets):
            logits[i] = sum(card_scores[j] for j in subset)
    elif intent == INTENT_DISCARD:
        for i, subset in enumerate(subsets):
            logits[i] = sum(1.0 - card_scores[j] for j in subset)
    elif intent in CONSUMABLE_INTENTS:
        # The subset IS the consumable's card target, so score it the same way as
        # PLAY: a high card score means "this card is worth acting on". Returning
        # a uniform distribution here (the pre-2026-07-30 behaviour) made target
        # choice unlearnable — and it did not matter then, because the target was
        # discarded before reaching the game anyway.
        for i, subset in enumerate(subsets):
            logits[i] = sum(card_scores[j] for j in subset)
    else:
        logits[:] = 0.0

    if temperature != 1.0 and temperature > 0:
        logits /= temperature

    return logits


# ════════════════════════════════════════════════════════════════════════════
# Boss blind validation
# ════════════════════════════════════════════════════════════════════════════

def validate_play_subset(subset: tuple[int, ...], hand: list[Card],
                         boss_key: str, played_types: set[str],
                         flags: Optional[dict] = None) -> bool:
    """
    Check if a play subset is valid under boss blind restrictions.

    Args:
        subset: card indices to play
        hand: current hand cards
        boss_key: active boss blind key (e.g. "bl_psychic") or ""
        played_types: set of hand types already played this round
        flags: ``game.hand_eval_flags()`` (Four Fingers / Shortcut / Smeared) so the
            hand type matches what the play path will evaluate (W5)

    Returns:
        True if the subset can be played, False if boss would reject it.
    """
    if not subset or not hand:
        return False

    cards = [hand[i] for i in subset if i < len(hand)]
    if not cards:
        return False

    # bl_psychic: must play exactly 5 cards
    if boss_key == "bl_psychic" and len(cards) != 5:
        return False

    try:
        hand_type, _ = evaluate_hand(cards, **(flags or {}))
    except Exception:
        return False

    # bl_eye: can't repeat a hand type
    if boss_key == "bl_eye" and hand_type in played_types:
        return False

    # bl_mouth: must play same type as first play (if any plays made)
    if boss_key == "bl_mouth" and played_types and hand_type not in played_types:
        return False

    return True


def get_valid_play_mask(subsets: list[tuple[int, ...]], hand: list[Card],
                        boss_key: str, played_types: set[str],
                        flags: Optional[dict] = None) -> np.ndarray:
    """
    Return boolean mask over subsets indicating which are valid plays.
    For non-boss or non-restrictive bosses, all subsets are valid.
    """
    n = len(subsets)
    if n == 0:
        return np.zeros(0, dtype=bool)

    # Fast path: most bosses don't restrict subset choice
    if boss_key not in ("bl_psychic", "bl_eye", "bl_mouth"):
        return np.ones(n, dtype=bool)

    mask = np.zeros(n, dtype=bool)
    for i, subset in enumerate(subsets):
        mask[i] = validate_play_subset(subset, hand, boss_key, played_types, flags)

    # Ensure at least one valid option (fallback: best single card)
    if not mask.any():
        mask[0] = True

    return mask


# ════════════════════════════════════════════════════════════════════════════
# Action application
# ════════════════════════════════════════════════════════════════════════════

def apply_action(intent: int, subset: tuple[int, ...],
                 hand: list[Card], game_state) -> dict:
    """
    Convert (intent, subset) into a game.step()-compatible action dict.

    Args:
        intent: INTENT_PLAY, INTENT_DISCARD, or INTENT_USE_CONSUMABLE
        subset: card indices selected by the agent
        hand: current hand cards
        game_state: BalatroGame instance (for validation)

    Returns:
        Action dict for game.step(), e.g.:
        {"type": "play", "cards": [0, 2, 4]}
        {"type": "discard", "cards": [1, 3]}
        {"type": "use_consumable", "consumable_idx": 1, "target_cards": [0, 3]}
    """
    if intent == INTENT_PLAY:
        # Validate indices are in range
        valid_indices = [i for i in subset if i < len(hand)]
        if not valid_indices:
            valid_indices = [0]  # fallback
        return {"type": "play", "cards": valid_indices}

    elif intent == INTENT_DISCARD:
        valid_indices = [i for i in subset if i < len(hand)]
        if not valid_indices:
            valid_indices = [0]
        return {"type": "discard", "cards": valid_indices}

    elif intent in CONSUMABLE_INTENTS:
        # The selected subset is the consumable's card target. Until 2026-07-30
        # this returned `target_cards: []` and `consumable_idx: 0`, so every
        # card-targeting tarot and spectral was consumed for zero effect and
        # slot 1 was unreachable. apply_tarot/apply_spectral slice the target
        # list to whatever each card needs, so an over-long subset is harmless.
        return {
            "type": "use_consumable",
            "consumable_idx": consumable_slot(intent),
            "target_cards": [i for i in subset if i < len(hand)],
        }

    else:
        # Unknown intent — fallback to playing first card
        return {"type": "play", "cards": [0]}


# ════════════════════════════════════════════════════════════════════════════
# Side-effect-free hypothetical scoring
# ════════════════════════════════════════════════════════════════════════════

def _clone_joker_for_dry_run(joker):
    """
    Copy a JokerInstance for a dry-run scoring pass.

    JokerInstance.clone() does a shallow ``state.copy()``, which is fine for
    MCTS (the clone is the new truth) but not for a dry run: Card Sharp keeps a
    ``set`` under ``state["played_hands"]`` and mutates it in place from
    ``on_hand_scored``, and several jokers keep lists under
    ``state["pending_consumables"]``. One level of container copying covers every
    state shape the joker modules currently use (primitives, flat lists/sets/dicts).
    """
    new = joker.clone()
    new.state = {
        k: (v.copy() if isinstance(v, (list, set, dict)) else v)
        for k, v in joker.state.items()
    }
    return new


class HypotheticalScorer:
    """
    Score hypothetical subsets of ``gs.hand`` with ZERO effect on game state.

    ``score_hand()`` is not a pure function. Lucky cards, Bloodstone, Business
    Card, 8 Ball, Misprint, Space Joker and Reserved Parking roll on
    ``ctx.rng``; scaling jokers (Ride the Bus, Vampire, Obelisk, Loyalty Card,
    Card Sharp, ...) bump ``inst.state``; Space Joker writes the ``planet_levels``
    dict it was handed; Vampire and Midas Mask rewrite ``card.enhancement`` and
    Sixth Sense / DNA set ``card.debuffed``. Every env's "best possible hand"
    estimate used to call it on the LIVE jokers, the LIVE planet dict, the LIVE
    hand cards and either the live ``gs.rng`` (env_v7) or, with ``rng=None``, the
    unseeded process-global ``random`` module (env_sim, env_v5) — so the reward
    estimate mutated the world before the real play, advanced the seeded stream by
    a hand-dependent amount, and was not even deterministic call-to-call
    (MP_UPDATE_LIST_2026-08 §3).

    This class clones the run's keyed ``PseudoRandom`` (``gs.run_state.rng``)
    once per hand and, for every candidate, scores against fresh copies of the
    jokers, the involved cards and the planet-level dict with that private clone
    restored to the snapshot (W3: the per-key streams mean every candidate sees
    exactly the Lucky / Bloodstone / ... luck the real play would). Card-CREATING
    effects (8 Ball, Seance, ...) get ``run_state=None`` so the dry run never
    marks ``used_jokers``; their probability roll is still consumed on the clone.
    Consequences:

      * the live game is untouched — ``gs.run_state.rng``, joker state, planet
        levels, cards, dollars are bit-identical before and after;
      * the estimate is a pure function of (game state, candidate): calling it
        twice gives the same number, and candidate order does not matter because
        every candidate sees the same state and the same luck;
      * the process-global ``random`` module is never consulted.

    Usage::

        scorer = HypotheticalScorer(gs, model_held=True)
        for combo in enumerate_subsets(len(gs.hand)):
            cards = [gs.hand[i] for i in combo]
            ht, sc = evaluate_hand(cards)
            s = scorer.score(cards, ht, sc)

    ``cards`` / ``scoring_cards`` are the ORIGINAL hand cards (as returned by
    ``evaluate_hand`` on the live hand); copies are made internally. Pass
    ``model_held=True`` to also model held-in-hand and full-deck effects
    (Steel, Baron, Steel Joker, ...) the way the real play does — env_v7 wants
    that, env_sim/env_v5 historically did not and keep their old semantics.
    """

    __slots__ = ("_gs", "_rng", "_rng_state", "_model_held", "_full_deck", "_round_cards")

    def __init__(self, gs, *, model_held: bool = False):
        self._gs = gs
        # A private clone of the live keyed PseudoRandom: the estimate is
        # deterministic given the game state (same per-key positions) but never
        # advances gs.run_state.rng. `restore(snapshot)` rewinds it per candidate.
        self._rng = gs.run_state.rng.clone()
        self._rng_state = self._rng.snapshot()
        self._model_held = model_held
        # round cards (Idol / Ancient / ...) are read-only for scoring; copy so a
        # lazily-rolled value in a dry run cannot leak into the game
        from .round_cards import from_round_picks
        self._round_cards = from_round_picks(getattr(gs, "round_picks", None))
        # full_deck is read-only for every scoring hook today (Steel Joker, Stone
        # Joker, Driver's License only count), so one copy per scorer suffices.
        self._full_deck = [c.copy() for c in gs.full_deck] if model_held else None

    def score(self, cards: list[Card], hand_type: str,
              scoring_cards: list[Card]) -> int:
        """Return score_hand()'s integer score for one candidate, side-effect-free."""
        gs = self._gs

        # Fresh card copies so Vampire / Midas Mask / Sixth Sense cannot touch the
        # live hand. scoring_cards is a sub-list of `cards` (same objects) from
        # evaluate_hand; map it through the same copies so identity-based checks
        # inside the hooks (`card in ctx.scoring_cards`) still hold.
        copy_of = {id(c): c.copy() for c in cards}
        sim_cards = [copy_of[id(c)] for c in cards]
        sim_scoring = [copy_of.get(id(c)) or c.copy() for c in scoring_cards]

        held = None
        if self._model_held:
            played_ids = copy_of.keys()
            held = [c.copy() for c in gs.hand if id(c) not in played_ids]

        # Every candidate starts from the same per-key stream positions.
        self._rng.restore(self._rng_state)
        from .jokers.base import n_owned

        s, _ctx = score_hand(
            scoring_cards=sim_scoring,
            all_cards=sim_cards,
            hand_type=hand_type,
            jokers=[_clone_joker_for_dry_run(j) for j in gs.jokers],
            planet_levels=dict(gs.planet_levels),
            hands_left=gs.hands_left,
            discards_left=gs.discards_left,
            dollars=gs.dollars,
            ante=gs.ante,
            deck_remaining=len(gs.deck),
            rng=self._rng,
            held_cards=held,
            full_deck=self._full_deck,
            run_state=None,                       # dry run: no card creation / used_jokers
            probabilities_normal=float(2 ** n_owned(gs.jokers, "j_oops")),
            round_cards=dict(self._round_cards),
            joker_slots=gs.joker_slots,
            consumable_slots=gs.consumable_slots,
            consumables=gs.consumable_hand,
            hands_played=getattr(gs, "_hands_played_round", 0),
            plasma=getattr(gs, "plasma", False),  # W3: Plasma Deck balance ranks hands differently
        )
        return s
