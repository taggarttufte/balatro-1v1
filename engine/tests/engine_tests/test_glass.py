"""
test_glass.py — Glass cards shatter, and Glass Joker scales on that.

Regression tests for audit finding H2. Before 2026-07-29 Glass cards gave x2 Mult
with no destruction implemented anywhere, i.e. a free x2 with no downside, and
Glass Joker counted "Glass cards in ctx.all_cards" — which is the played
selection, not the deck — so it was a different joker entirely.

Real behaviour (https://balatrowiki.org/w/Glass_Cards):
  Glass Card  - X2 Mult when scored, 1 in 4 chance to be destroyed after all
                scoring finishes.
  Glass Joker - gains X0.75 Mult for every Glass Card that is destroyed.
"""
from balatro_sim.game_keys import core as _core
PseudoRandom = _core.PseudoRandom

import pytest

from balatro_sim.card import Card
from balatro_sim.game import BalatroGame, State
from balatro_sim.jokers.base import JokerInstance, ScoreContext
from balatro_sim.scoring import score_hand


def _fresh(seed=3):
    g = BalatroGame(seed=seed)
    g.step({"type": "play_blind"})
    return g


class TestGlassScoring:
    def test_glass_gives_x2_mult(self):
        plain = Card(rank=10, suit="Spades")
        glass = Card(rank=10, suit="Spades", enhancement="Glass")
        base, _ = score_hand(
            scoring_cards=[plain], all_cards=[plain], hand_type="High Card",
            jokers=[], planet_levels={"High Card": 1}, hands_left=3,
            discards_left=3, dollars=4, ante=1, deck_remaining=40,
            rng=PseudoRandom("TEST1"))
        doubled, _ = score_hand(
            scoring_cards=[glass], all_cards=[glass], hand_type="High Card",
            jokers=[], planet_levels={"High Card": 1}, hands_left=3,
            discards_left=3, dollars=4, ante=1, deck_remaining=40,
            rng=PseudoRandom("TEST1"))
        assert doubled == base * 2

    def test_scored_glass_is_recorded_for_destruction(self):
        glass = Card(rank=10, suit="Spades", enhancement="Glass")
        _s, ctx = score_hand(
            scoring_cards=[glass], all_cards=[glass], hand_type="High Card",
            jokers=[], planet_levels={"High Card": 1}, hands_left=3,
            discards_left=3, dollars=4, ante=1, deck_remaining=40,
            rng=PseudoRandom("TEST1"))
        assert ctx.glass_scored == [glass]

    def test_non_glass_not_recorded(self):
        plain = Card(rank=10, suit="Spades")
        _s, ctx = score_hand(
            scoring_cards=[plain], all_cards=[plain], hand_type="High Card",
            jokers=[], planet_levels={"High Card": 1}, hands_left=3,
            discards_left=3, dollars=4, ante=1, deck_remaining=40,
            rng=PseudoRandom("TEST1"))
        assert ctx.glass_scored == []

    def test_retrigger_does_not_double_count_the_same_card(self):
        """A retriggered glass card must still only roll destruction once."""
        glass = Card(rank=10, suit="Spades", enhancement="Glass", seal="Red")
        _s, ctx = score_hand(
            scoring_cards=[glass], all_cards=[glass], hand_type="High Card",
            jokers=[], planet_levels={"High Card": 1}, hands_left=3,
            discards_left=3, dollars=4, ante=1, deck_remaining=40,
            rng=PseudoRandom("TEST1"))
        assert ctx.glass_scored.count(glass) == 1


class TestGlassDestruction:
    def test_glass_cards_break_at_roughly_one_in_four(self):
        broken = kept = 0
        for seed in range(300):
            g = BalatroGame(seed=seed)
            g.step({"type": "play_blind"})
            # The injected card must be a member of the permanent deck, or its
            # removal cannot change len(full_deck) and the test reads 0%.
            glass = Card(rank=9, suit="Spades", enhancement="Glass")
            g.hand[0] = glass
            g.full_deck.append(glass)
            before = len(g.full_deck)
            g.step({"type": "play", "cards": [0]})
            if len(g.full_deck) == before - 1:
                broken += 1
            else:
                kept += 1
        rate = broken / (broken + kept)
        assert 0.18 < rate < 0.32, f"expected ~0.25 destruction rate, got {rate:.3f}"

    def test_destroyed_glass_leaves_the_permanent_deck(self):
        """Find a seed where the card breaks, then verify it stays gone."""
        for seed in range(200):
            g = BalatroGame(seed=seed)
            g.step({"type": "play_blind"})
            glass = Card(rank=9, suit="Spades", enhancement="Glass")
            g.hand[0] = glass
            g.full_deck.append(glass)
            g.step({"type": "play", "cards": [0]})
            if glass not in g.full_deck:
                assert glass not in g.deck
                assert glass not in g.hand
                assert glass not in g.discard_pile
                return
        pytest.fail("no seed in 200 destroyed the glass card")

    def test_surviving_glass_stays_in_the_deck(self):
        for seed in range(200):
            g = BalatroGame(seed=seed)
            g.step({"type": "play_blind"})
            glass = Card(rank=9, suit="Spades", enhancement="Glass")
            g.hand[0] = glass
            g.full_deck.append(glass)
            g.step({"type": "play", "cards": [0]})
            if glass in g.full_deck:
                assert glass in g.discard_pile
                return
        pytest.fail("no seed in 200 preserved the glass card")

    def test_destruction_is_seed_determined(self):
        def run(seed):
            g = BalatroGame(seed=seed)
            g.step({"type": "play_blind"})
            g.hand[0] = Card(rank=9, suit="Spades", enhancement="Glass")
            g.step({"type": "play", "cards": [0]})
            return len(g.full_deck)
        assert run(12) == run(12)


class TestGlassJoker:
    def test_scales_on_destruction(self):
        g = BalatroGame(seed=5)
        j = JokerInstance("j_glass")
        g.jokers.append(j)
        g.step({"type": "play_blind"})

        glass = Card(rank=9, suit="Spades", enhancement="Glass")
        g.full_deck.append(glass)
        g.destroy_card(glass)
        assert j.state["mult_mult"] == pytest.approx(1.75)

        glass2 = Card(rank=8, suit="Hearts", enhancement="Glass")
        g.full_deck.append(glass2)
        g.destroy_card(glass2)
        assert j.state["mult_mult"] == pytest.approx(2.5)

    def test_does_not_scale_on_non_glass_destruction(self):
        g = BalatroGame(seed=5)
        j = JokerInstance("j_glass")
        g.jokers.append(j)
        plain = Card(rank=9, suit="Spades")
        g.full_deck.append(plain)
        g.destroy_card(plain)
        assert j.state.get("mult_mult", 1.0) == 1.0

    def test_does_not_scale_on_glass_cards_merely_sitting_in_deck(self):
        """The old (wrong) behaviour: mult from deck contents alone."""
        g = BalatroGame(seed=5)
        j = JokerInstance("j_glass")
        g.jokers.append(j)
        for c in g.full_deck[:6]:
            c.enhancement = "Glass"
        assert j.state.get("mult_mult", 1.0) == 1.0

    def test_accumulated_mult_applies_to_score(self):
        card = Card(rank=10, suit="Spades")
        j = JokerInstance("j_glass")
        j.state["mult_mult"] = 2.5
        with_joker, _ = score_hand(
            scoring_cards=[card], all_cards=[card], hand_type="High Card",
            jokers=[j], planet_levels={"High Card": 1}, hands_left=3,
            discards_left=3, dollars=4, ante=1, deck_remaining=40,
            rng=PseudoRandom("TEST1"))
        without, _ = score_hand(
            scoring_cards=[card], all_cards=[card], hand_type="High Card",
            jokers=[], planet_levels={"High Card": 1}, hands_left=3,
            discards_left=3, dollars=4, ante=1, deck_remaining=40,
            rng=PseudoRandom("TEST1"))
        assert with_joker == int(without * 2.5)


class TestCainoStillWorks:
    def test_caino_scales_on_face_card_destruction(self):
        """destroy_card must notify every on_card_destroyed joker, not just Glass."""
        g = BalatroGame(seed=5)
        j = JokerInstance("j_caino")
        g.jokers.append(j)
        king = Card(rank=13, suit="Spades")
        g.full_deck.append(king)
        g.destroy_card(king)
        assert j.state["xmult"] == pytest.approx(1.1)
