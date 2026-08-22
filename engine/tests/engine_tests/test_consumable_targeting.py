"""
test_consumable_targeting.py — consumables must reach the cards the agent picked.

Regression tests for audit finding A1. Until 2026-07-30 both the action builder
and the env sent `target_cards: []` with a hardcoded `consumable_idx: 0`:

    elif intent == INTENT_USE_CONSUMABLE:
        return {"type": "use_consumable", "consumable_idx": 0, "target_cards": []}

`apply_tarot` builds its targets from `target_indices or []`, so it modified
nothing — and still returned True, so the consumable was consumed anyway. The
agent paid a card for nothing, every time. Consumable slot 1 was unreachable for
the entire history of the project.

That silently blocked four of the 2026-07-29 sim fixes: C7 (persistent deck),
C3 (Steel held-in-hand), H2 (Glass destruction) and C6 (Purple seal) all require
a tarot or spectral landing on a card first. Measured before the fix: 28% of
episodes used a tarot, 0% ended with a single modified card.
"""
import pytest

from balatro_sim.card import Card
from balatro_sim.card_selection import (
    CONSUMABLE_INTENTS, INTENT_DISCARD, INTENT_PLAY,
    INTENT_USE_CONSUMABLE_0, INTENT_USE_CONSUMABLE_1,
    apply_action, consumable_slot,
)
from balatro_sim.env_v7 import BalatroV7Env
from balatro_sim.game import BalatroGame, State


def _env(seed=5):
    e = BalatroV7Env(seed=seed)
    e.reset()
    if e.game.state == State.BLIND_SELECT:
        e.step_phase(0)                      # play_blind
    return e


def _modified(game):
    return sum(1 for c in game.full_deck
               if c.enhancement != "None" or c.seal != "None"
               or c.edition != "None")


# ── the action builder ────────────────────────────────────────────────────────

class TestApplyAction:
    def test_targets_are_forwarded(self):
        hand = [Card(rank=5, suit="Spades") for _ in range(8)]
        act = apply_action(INTENT_USE_CONSUMABLE_0, (0, 3), hand, None)
        assert act["target_cards"] == [0, 3]

    def test_slot_comes_from_the_intent(self):
        hand = [Card(rank=5, suit="Spades") for _ in range(8)]
        assert apply_action(INTENT_USE_CONSUMABLE_0, (0,), hand, None)["consumable_idx"] == 0
        assert apply_action(INTENT_USE_CONSUMABLE_1, (0,), hand, None)["consumable_idx"] == 1

    def test_out_of_range_indices_are_dropped(self):
        hand = [Card(rank=5, suit="Spades") for _ in range(3)]
        act = apply_action(INTENT_USE_CONSUMABLE_0, (0, 2, 7), hand, None)
        assert act["target_cards"] == [0, 2]

    def test_play_and_discard_are_unchanged(self):
        hand = [Card(rank=5, suit="Spades") for _ in range(8)]
        assert apply_action(INTENT_PLAY, (1, 2), hand, None) == {
            "type": "play", "cards": [1, 2]}
        assert apply_action(INTENT_DISCARD, (0,), hand, None) == {
            "type": "discard", "cards": [0]}

    def test_consumable_slot_helper(self):
        assert consumable_slot(INTENT_USE_CONSUMABLE_0) == 0
        assert consumable_slot(INTENT_USE_CONSUMABLE_1) == 1
        assert consumable_slot(INTENT_PLAY) is None
        assert consumable_slot(INTENT_DISCARD) is None


# ── end to end through the env ────────────────────────────────────────────────

class TestTarotsActuallyLand:
    def test_justice_enhances_the_targeted_card(self):
        e = _env()
        g = e.game
        g.consumable_hand = ["c_justice"]              # -> Glass
        target = g.hand[2]
        assert _modified(g) == 0

        e.step_hand(INTENT_USE_CONSUMABLE_0, (2,))
        assert target.enhancement == "Glass"
        assert _modified(g) == 1

    def test_consumable_is_actually_spent(self):
        e = _env()
        e.game.consumable_hand = ["c_justice"]
        e.step_hand(INTENT_USE_CONSUMABLE_0, (0,))
        assert e.game.consumable_hand == []

    def test_slot_1_is_reachable(self):
        """Slot 1 could never be used before this fix."""
        e = _env()
        g = e.game
        g.consumable_hand = ["c_justice", "c_devil"]   # Glass, Gold
        target = g.hand[1]

        e.step_hand(INTENT_USE_CONSUMABLE_1, (1,))
        assert target.enhancement == "Gold", "slot 1 consumable did not fire"
        assert g.consumable_hand == ["c_justice"], "wrong slot was consumed"

    def test_multi_target_tarot_hits_multiple_cards(self):
        e = _env()
        g = e.game
        g.consumable_hand = ["c_hanged_man"]           # destroys up to 2
        before = len(g.full_deck)
        e.step_hand(INTENT_USE_CONSUMABLE_0, (0, 1))
        assert len(g.full_deck) == before - 2

    def test_spectral_seal_lands(self):
        e = _env()
        g = e.game
        g.consumable_hand = ["c_medium"]               # Purple seal
        target = g.hand[0]
        e.step_hand(INTENT_USE_CONSUMABLE_0, (0,))
        assert target.seal == "Purple"

    def test_planet_still_works_without_targets(self):
        e = _env()
        g = e.game
        g.consumable_hand = ["c_mercury"]
        before = g.planet_levels["Pair"]
        e.step_hand(INTENT_USE_CONSUMABLE_0, (0,))
        assert g.planet_levels["Pair"] == before + 1

    def test_modification_persists_into_the_next_blind(self):
        """A1 + C7 together: the whole point of the deck-building path."""
        e = _env()
        g = e.game
        g.consumable_hand = ["c_justice"]
        e.step_hand(INTENT_USE_CONSUMABLE_0, (0,))
        assert _modified(g) == 1

        g.chips_scored = g.current_blind.chips_target
        g.state = State.ROUND_EVAL
        g._end_round()
        g.step({"type": "leave_shop"})
        g.step({"type": "play_blind"})
        dealt = g.deck + g.hand + g.discard_pile
        assert sum(1 for c in dealt if c.enhancement == "Glass") == 1


# ── masking ───────────────────────────────────────────────────────────────────

class TestIntentMask:
    def test_no_consumables_means_no_consumable_intents(self):
        e = _env()
        e.game.consumable_hand = []
        mask = e.get_intent_mask()
        assert not mask[INTENT_USE_CONSUMABLE_0]
        assert not mask[INTENT_USE_CONSUMABLE_1]

    def test_one_consumable_enables_only_slot_0(self):
        e = _env()
        e.game.consumable_hand = ["c_justice"]
        mask = e.get_intent_mask()
        assert mask[INTENT_USE_CONSUMABLE_0]
        assert not mask[INTENT_USE_CONSUMABLE_1]

    def test_two_consumables_enable_both_slots(self):
        e = _env()
        e.game.consumable_hand = ["c_justice", "c_devil"]
        mask = e.get_intent_mask()
        assert mask[INTENT_USE_CONSUMABLE_0]
        assert mask[INTENT_USE_CONSUMABLE_1]

    def test_masked_slot_is_a_noop_if_taken_anyway(self):
        e = _env()
        e.game.consumable_hand = ["c_justice"]
        before = list(e.game.consumable_hand)
        e.step_hand(INTENT_USE_CONSUMABLE_1, (0,))     # slot 1 is empty
        assert e.game.consumable_hand == before


# ── shop path ─────────────────────────────────────────────────────────────────

class TestShopPlanetUse:
    def test_planet_in_slot_1_is_reachable(self):
        """Shop action 16 only looked at slot 0, so a planet behind a tarot
        could never be used."""
        e = _env()
        g = e.game
        g.chips_scored = g.current_blind.chips_target
        g.state = State.ROUND_EVAL
        g._end_round()
        assert g.state == State.SHOP

        g.consumable_hand = ["c_justice", "c_mercury"]
        before = g.planet_levels["Pair"]
        e.step_phase(16)
        assert g.planet_levels["Pair"] == before + 1
        assert g.consumable_hand == ["c_justice"]

    def test_no_planet_is_a_noop(self):
        e = _env()
        g = e.game
        g.chips_scored = g.current_blind.chips_target
        g.state = State.ROUND_EVAL
        g._end_round()
        g.consumable_hand = ["c_justice"]
        e.step_phase(16)
        assert g.consumable_hand == ["c_justice"]
