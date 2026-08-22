"""
test_seals.py — the four seals must fire on the correct events.

Regression tests for audit findings C5 and C6. Before 2026-07-29:
  Gold Seal   paid $3 for being HELD at end of round (real: $3 when PLAYED and
              it scores) — conflated with the Gold enhancement.
  Purple Seal fired on PLAYED cards (real: when DISCARDED). _discard had no seal
              handling at all, so the mechanic was unreachable.
  Blue Seal   fired on scoring cards right after a play (real: at END OF ROUND,
              for cards HELD IN HAND, creating the Planet for the final played
              hand type of the round).

Reference: https://balatrowiki.org/w/Card_modifiers
"""
from balatro_sim.game_keys import core as _core
PseudoRandom = _core.PseudoRandom

import pytest

from balatro_sim.card import Card
from balatro_sim.consumables import PLANET_HAND
from balatro_sim.game import BalatroGame, State
from balatro_sim.scoring import score_hand


def _fresh(seed=5):
    g = BalatroGame(seed=seed)
    g.step({"type": "play_blind"})
    return g


def _score(cards, seal=None, hand_type="High Card", **kw):
    for c in cards:
        if seal:
            c.seal = seal
    return score_hand(
        scoring_cards=cards, all_cards=cards, hand_type=hand_type,
        jokers=[], planet_levels={hand_type: 1}, hands_left=3, discards_left=3,
        dollars=4, ante=1, deck_remaining=40, rng=PseudoRandom("TEST1"), **kw)


# ── Gold Seal — $3 when played and scores ────────────────────────────────────

class TestGoldSeal:
    def test_pays_three_when_scored(self):
        card = Card(rank=5, suit="Spades", seal="Gold")
        _score_result, ctx = _score([card])
        assert ctx.pending_money == 3

    def test_no_payout_without_seal(self):
        card = Card(rank=5, suit="Spades")
        _s, ctx = _score([card])
        assert ctx.pending_money == 0

    def test_retrigger_pays_again(self):
        """Each scoring pass pays $3, so a retriggered gold-sealed card pays $6.

        Cards have a single seal, so the retrigger is driven the way jokers do it
        (Sock and Buskin, Hanging Chad, Dusk) — via ctx.card_retriggers.
        """
        from balatro_sim.jokers.base import ScoreContext
        from balatro_sim.scoring import _score_single_card

        card = Card(rank=5, suit="Spades", seal="Gold")
        ctx = ScoreContext(prng=PseudoRandom("TEST1"))
        _score_single_card(card, ctx, [])
        assert ctx.pending_money == 3
        _score_single_card(card, ctx, [])          # the retrigger pass
        assert ctx.pending_money == 6

    def test_money_reaches_the_bank_on_play(self):
        g = _fresh()
        g.hand[0].seal = "Gold"
        before = g.dollars
        g.step({"type": "play", "cards": [0]})
        assert g.dollars == before + 3

    def test_holding_pays_nothing(self):
        """The inverted behaviour: holding a gold-sealed card must pay $0."""
        g = _fresh()
        g.hand[0].seal = "Gold"
        g.chips_scored = g.current_blind.chips_target
        g.state = State.ROUND_EVAL
        g.hands_left = 0
        g.dollars = 0
        g._end_round()
        assert g.dollars == g.current_blind.money_reward


# ── Purple Seal — creates a Tarot when discarded ──────────────────────────────

class TestPurpleSeal:
    def test_discard_creates_tarot(self):
        g = _fresh()
        g.consumable_hand.clear()
        g.hand[0].seal = "Purple"
        g.step({"type": "discard", "cards": [0]})
        assert len(g.consumable_hand) == 1

    def test_playing_creates_nothing(self):
        """Purple Seal must NOT fire on play — that was the bug."""
        g = _fresh()
        g.consumable_hand.clear()
        g.hand[0].seal = "Purple"
        g.step({"type": "play", "cards": [0]})
        assert g.consumable_hand == []

    def test_two_purple_discards_create_two_tarots(self):
        g = _fresh()
        g.consumable_hand.clear()
        g.consumable_slots = 4
        g.hand[0].seal = "Purple"
        g.hand[1].seal = "Purple"
        g.step({"type": "discard", "cards": [0, 1]})
        assert len(g.consumable_hand) == 2

    def test_respects_consumable_slot_limit(self):
        g = _fresh()
        g.consumable_hand.clear()
        g.consumable_slots = 1
        g.hand[0].seal = "Purple"
        g.hand[1].seal = "Purple"
        g.step({"type": "discard", "cards": [0, 1]})
        assert len(g.consumable_hand) == 1

    def test_no_discards_left_means_no_tarot(self):
        g = _fresh()
        g.consumable_hand.clear()
        g.discards_left = 0
        g.hand[0].seal = "Purple"
        g.step({"type": "discard", "cards": [0]})
        assert g.consumable_hand == []


# ── Blue Seal — Planet for final played hand, if held at end of round ─────────

class TestBlueSeal:
    def test_held_at_round_end_creates_planet_for_last_hand(self):
        g = _fresh()
        g.consumable_hand.clear()
        # Force a known pair into the hand rather than relying on the shuffle
        g.hand[0] = Card(rank=7, suit="Spades")
        g.hand[1] = Card(rank=7, suit="Hearts")
        g.step({"type": "play", "cards": [0, 1]})
        assert g.last_played_hand_type == "Pair"

        g.hand[0].seal = "Blue"
        g.chips_scored = g.current_blind.chips_target
        g.state = State.ROUND_EVAL
        g._end_round()
        assert PLANET_HAND[g.consumable_hand[0]] == "Pair"

    def test_planet_matches_hand_type(self):
        g = _fresh()
        g.consumable_hand.clear()
        g.last_played_hand_type = "Flush"
        g.hand[0].seal = "Blue"
        g.chips_scored = g.current_blind.chips_target
        g.state = State.ROUND_EVAL
        g._end_round()
        assert len(g.consumable_hand) == 1
        assert PLANET_HAND[g.consumable_hand[0]] == "Flush"

    def test_not_held_means_no_planet(self):
        """A blue-sealed card that was PLAYED (so not held) creates nothing."""
        g = _fresh()
        g.consumable_hand.clear()
        g.hand[0].seal = "Blue"
        g.step({"type": "play", "cards": [0]})
        played_planets = list(g.consumable_hand)
        g.chips_scored = g.current_blind.chips_target
        g.state = State.ROUND_EVAL
        g._end_round()
        # the played card is in the discard pile, not the hand
        assert g.consumable_hand == played_planets

    def test_no_hand_played_means_no_planet(self):
        g = _fresh()
        g.consumable_hand.clear()
        g.hand[0].seal = "Blue"
        g.last_played_hand_type = ""      # nothing played this round
        g.chips_scored = g.current_blind.chips_target
        g.state = State.ROUND_EVAL
        g._end_round()
        assert g.consumable_hand == []

    def test_stale_hand_type_does_not_leak_across_blinds(self):
        g = _fresh()
        g.step({"type": "play", "cards": [0]})
        assert g.last_played_hand_type != ""
        g.chips_scored = g.current_blind.chips_target
        g.state = State.ROUND_EVAL
        g._end_round()
        g.step({"type": "leave_shop"})
        g.step({"type": "play_blind"})
        assert g.last_played_hand_type == ""


# ── Red Seal — unchanged, guard against regression ────────────────────────────

def test_red_seal_still_retriggers_once():
    plain = Card(rank=10, suit="Spades")
    sealed = Card(rank=10, suit="Spades", seal="Red")
    base, _ = _score([plain])
    doubled, _ = _score([sealed])
    assert doubled > base
