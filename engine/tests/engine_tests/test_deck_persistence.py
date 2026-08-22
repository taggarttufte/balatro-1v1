"""
test_deck_persistence.py — permanent card modifications must survive blinds.

Regression tests for audit finding C7: _start_blind() called _init_deck(), which
rebuilt a fresh vanilla 52-card deck via make_standard_deck() every blind. Every
enhancement, seal and edition the player bought was silently destroyed at the
start of the next blind, which made deck-building strategies impossible and left
Steel Joker / Stone Joker / Driver's License permanently reading zero.
"""
import pytest

from balatro_sim.card import Card
from balatro_sim.consumables import apply_spectral, apply_tarot
from balatro_sim.game import BalatroGame, State


def _advance_one_blind(g):
    """Clear the current blind and land in the next BLIND_SELECT."""
    if g.state == State.BLIND_SELECT:
        g.step({"type": "play_blind"})
    g.chips_scored = g.current_blind.chips_target
    g.state = State.ROUND_EVAL
    g._end_round()                      # -> SHOP
    g.step({"type": "leave_shop"})      # -> next blind
    return g


def _dealt_cards(g):
    """Every card actually in play this blind (draw pile + hand + discard pile).

    Assertions must target THIS, not full_deck. The C7 bug rebuilt only the draw
    pile, so a test that checks full_deck membership passes even when the cards
    being dealt are fresh vanilla copies.
    """
    return g.deck + g.hand + g.discard_pile


class TestEnhancementPersistence:
    @pytest.mark.parametrize("attr,value", [
        ("enhancement", "Steel"),
        ("seal", "Purple"),
        ("edition", "Polychrome"),
        ("enhancement", "Gold"),
        ("seal", "Red"),
    ])
    def test_modification_survives_three_blinds(self, attr, value):
        """The exact test the audit asked for, generalised over modifier kinds.

        Checks the modified card is still being DEALT after three blinds, which
        is what the C7 bug actually broke.
        """
        g = BalatroGame(seed=7)
        setattr(g.full_deck[0], attr, value)
        target_id = g.full_deck[0].id

        for _ in range(3):
            _advance_one_blind(g)
        g.step({"type": "play_blind"})       # deal a fresh blind

        dealt = _dealt_cards(g)
        assert len(dealt) == 52, f"expected 52 cards in play, got {len(dealt)}"
        survivor = next((c for c in dealt if c.id == target_id), None)
        assert survivor is not None, "modified card was not dealt"
        assert getattr(survivor, attr) == value

    def test_modified_card_count_is_stable_in_play(self):
        """Four Steel cards must still be four Steel cards when dealt later."""
        g = BalatroGame(seed=11)
        for c in g.full_deck[:4]:
            c.enhancement = "Steel"

        for _ in range(3):
            _advance_one_blind(g)
        g.step({"type": "play_blind"})

        steel = sum(1 for c in _dealt_cards(g) if c.enhancement == "Steel")
        assert steel == 4, f"expected 4 Steel cards in play, got {steel}"

    def test_modifying_a_card_in_hand_reaches_the_permanent_deck(self):
        """Cards in hand are the same objects as in full_deck, not copies."""
        g = BalatroGame(seed=17)
        g.step({"type": "play_blind"})
        card = g.hand[0]
        card.enhancement = "Glass"
        assert any(c is card and c.enhancement == "Glass" for c in g.full_deck)

        _advance_one_blind(g)
        assert sum(1 for c in g.full_deck if c.enhancement == "Glass") == 1


class TestDeckSizeInvariants:
    def test_deck_size_stable_across_blinds(self):
        g = BalatroGame(seed=19)
        for _ in range(4):
            _advance_one_blind(g)
            assert len(g.full_deck) == 52

    def test_partitions_sum_to_full_deck(self):
        """draw pile + hand + discard pile must exactly partition the deck."""
        g = BalatroGame(seed=23)
        g.step({"type": "play_blind"})
        for _ in range(3):
            if g.state != State.SELECTING_HAND:
                break
            n = min(5, len(g.hand))
            if n == 0:
                break
            g.step({"type": "play", "cards": list(range(n))})
            partition = len(g.deck) + len(g.hand) + len(g.discard_pile)
            assert partition == len(g.full_deck), (
                f"{len(g.deck)}+{len(g.hand)}+{len(g.discard_pile)} "
                f"!= {len(g.full_deck)}")

    def test_played_cards_return_next_blind(self):
        """Cards played this blind are back in the draw pile once the next starts.

        Note the deck is reshuffled by _start_blind, i.e. on "play_blind" — not
        on leaving the shop — so the assertion has to come after that step.
        """
        g = BalatroGame(seed=29)
        g.step({"type": "play_blind"})
        played = list(g.hand[:5])
        g.step({"type": "play", "cards": [0, 1, 2, 3, 4]})
        assert all(c in g.discard_pile for c in played)

        _advance_one_blind(g)
        g.step({"type": "play_blind"})          # reshuffles full_deck
        assert all(c in g.deck or c in g.hand for c in played)
        assert g.discard_pile == []


class TestDestruction:
    def test_remove_card_is_permanent(self):
        g = BalatroGame(seed=31)
        victim = g.full_deck[0]
        assert g.remove_card(victim) is True
        assert len(g.full_deck) == 51

        _advance_one_blind(g)
        assert victim not in g.full_deck
        assert victim not in g.deck
        assert len(g.full_deck) == 51

    def test_hanged_man_destruction_is_permanent(self):
        """The Hanged Man removed from hand/draw pile only, so cards came back."""
        g = BalatroGame(seed=37)
        g.step({"type": "play_blind"})
        targets = list(g.hand[:2])
        apply_tarot(g, "c_hanged_man", [0, 1])   # tarots take hand INDICES
        assert len(g.full_deck) == 50

        _advance_one_blind(g)
        assert len(g.full_deck) == 50
        assert all(t not in g.full_deck for t in targets)

    def test_added_cards_join_the_permanent_deck(self):
        g = BalatroGame(seed=41)
        g.step({"type": "play_blind"})
        before = len(g.full_deck)
        new = Card(rank=14, suit="Spades", enhancement="Steel")
        g.add_card(new)
        assert len(g.full_deck) == before + 1

        _advance_one_blind(g)
        assert new in g.full_deck
        assert new in g.deck


class TestDeckDependentJokers:
    def test_steel_joker_sees_deck_steel_cards(self):
        """Steel Joker read a vanilla deck before C7, so it was always zero."""
        g = BalatroGame(seed=43)
        for c in g.full_deck[:4]:
            c.enhancement = "Steel"
        _advance_one_blind(g)
        g.step({"type": "play_blind"})
        assert sum(1 for c in _dealt_cards(g) if c.enhancement == "Steel") == 4

    def test_tarot_enhancement_accumulates_over_antes(self):
        """Repeated enhancement across blinds should compound, not reset.

        Before C7 this stayed at 1 forever: each blind wiped the previous
        enhancement, so no amount of tarot use could build a deck.
        """
        g = BalatroGame(seed=47)
        for _ in range(3):
            if g.state == State.BLIND_SELECT:
                g.step({"type": "play_blind"})
            # pick a card that is not already Glass, so each iteration adds one
            idx = next(i for i, c in enumerate(g.hand)
                       if c.enhancement != "Glass")
            apply_tarot(g, "c_justice", [idx])          # -> Glass
            g.chips_scored = g.current_blind.chips_target
            g.state = State.ROUND_EVAL
            g._end_round()
            g.step({"type": "leave_shop"})
        glass = sum(1 for c in g.full_deck if c.enhancement == "Glass")
        assert glass == 3, f"expected 3 accumulated Glass cards, got {glass}"
