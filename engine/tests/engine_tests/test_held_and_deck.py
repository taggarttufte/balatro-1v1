"""
test_held_and_deck.py — held-in-hand and full-deck effects read the right cards.

Regression tests for audit findings C2 and C3. ScoreContext had a single
`all_cards` field set to the PLAYED selection, but jokers read it as three
different things:

  "held in hand"  Baron, Blackboard, Raised Fist, Shoot the Moon,
                  Reserved Parking          -> now ctx.held_cards
  "in full deck"  Steel Joker, Stone Joker,
                  Driver's License          -> now ctx.full_deck
  "played cards"  Splash, Flower Pot        -> still ctx.all_cards (correct)

Steel cards were also applied in the played-card loop, so they paid out for being
played (which does nothing in the real game) and never for being held. There was
no held-in-hand phase at all.

Reference: https://balatrowiki.org/w/Card_modifiers,
           https://balatrowiki.org/w/Scoring
"""
from balatro_sim.game_keys import core as _core
PseudoRandom = _core.PseudoRandom

import pytest

from balatro_sim.card import Card
from balatro_sim.jokers.base import JokerInstance
from balatro_sim.scoring import score_hand


def _score(played, jokers=(), held=(), full_deck=(), hand_type="High Card"):
    return score_hand(
        scoring_cards=list(played), all_cards=list(played), hand_type=hand_type,
        jokers=list(jokers), planet_levels={hand_type: 1}, hands_left=3,
        discards_left=3, dollars=4, ante=1, deck_remaining=40,
        rng=PseudoRandom("TEST1"), held_cards=list(held), full_deck=list(full_deck))


TEN = lambda: Card(rank=10, suit="Spades")            # 10 chips
KING = lambda s="Spades": Card(rank=13, suit=s)
QUEEN = lambda s="Spades": Card(rank=12, suit=s)


# ── Steel enhancement: X1.5 Mult while HELD ───────────────────────────────────

class TestSteelEnhancement:
    def test_steel_held_gives_x1_5(self):
        base, _ = _score([TEN()])
        steel = Card(rank=2, suit="Hearts", enhancement="Steel")
        withheld, _ = _score([TEN()], held=[steel])
        assert withheld == int(base * 1.5)

    def test_steel_played_does_nothing(self):
        """Playing a Steel card must not multiply — that was the inversion."""
        plain = Card(rank=10, suit="Spades")
        steel = Card(rank=10, suit="Spades", enhancement="Steel")
        base, _ = _score([plain])
        played, _ = _score([steel])
        assert played == base

    def test_two_steel_held_compounds(self):
        base, _ = _score([TEN()])
        s1 = Card(rank=2, suit="Hearts", enhancement="Steel")
        s2 = Card(rank=3, suit="Hearts", enhancement="Steel")
        got, _ = _score([TEN()], held=[s1, s2])
        assert got == int(base * 1.5 * 1.5)

    def test_debuffed_steel_does_nothing(self):
        base, _ = _score([TEN()])
        steel = Card(rank=2, suit="Hearts", enhancement="Steel", debuffed=True)
        got, _ = _score([TEN()], held=[steel])
        assert got == base

    def test_red_sealed_steel_retriggers_in_hand(self):
        base, _ = _score([TEN()])
        steel = Card(rank=2, suit="Hearts", enhancement="Steel", seal="Red")
        got, _ = _score([TEN()], held=[steel])
        assert got == int(base * 1.5 * 1.5)


# ── Held-in-hand jokers ───────────────────────────────────────────────────────

class TestBaron:
    def test_counts_kings_held_not_played(self):
        base, _ = _score([TEN()], jokers=[])
        baron = JokerInstance("j_baron")
        held_two_kings, _ = _score([TEN()], jokers=[baron],
                                   held=[KING(), KING("Hearts")])
        assert held_two_kings == int(base * 1.5 * 1.5)

    def test_kings_in_the_played_hand_do_not_count(self):
        """Before C2, Baron counted Kings among the played cards."""
        baron = JokerInstance("j_baron")
        with_played_kings, _ = _score([KING(), KING("Hearts")], jokers=[baron],
                                      held=[], hand_type="Pair")
        without, _ = _score([KING(), KING("Hearts")], jokers=[],
                            held=[], hand_type="Pair")
        assert with_played_kings == without

    def test_no_kings_held_is_neutral(self):
        baron = JokerInstance("j_baron")
        got, _ = _score([TEN()], jokers=[baron], held=[QUEEN()])
        base, _ = _score([TEN()])
        assert got == base


class TestShootTheMoon:
    def test_counts_queens_held(self):
        j = JokerInstance("j_shoot_the_moon")
        base, _ = _score([TEN()])
        got, _ = _score([TEN()], jokers=[j], held=[QUEEN(), QUEEN("Hearts")])
        # High Card 5/1 + 10 chips = 15 chips, mult 1 + 26 = 27
        assert got > base
        assert got == 15 * (1 + 26)

    def test_queens_played_do_not_count(self):
        j = JokerInstance("j_shoot_the_moon")
        got, _ = _score([QUEEN()], jokers=[j], held=[])
        without, _ = _score([QUEEN()], jokers=[])
        assert got == without


class TestBlackboard:
    def test_x3_when_all_held_are_black(self):
        j = JokerInstance("j_blackboard")
        base, _ = _score([TEN()])
        got, _ = _score([TEN()], jokers=[j],
                        held=[Card(rank=4, suit="Spades"),
                              Card(rank=5, suit="Clubs")])
        assert got == base * 3

    def test_no_bonus_when_a_red_card_is_held(self):
        j = JokerInstance("j_blackboard")
        base, _ = _score([TEN()])
        got, _ = _score([TEN()], jokers=[j],
                        held=[Card(rank=4, suit="Spades"),
                              Card(rank=5, suit="Hearts")])
        assert got == base

    def test_red_cards_in_played_hand_are_irrelevant(self):
        """Only HELD cards matter; the played hand can be any suit."""
        j = JokerInstance("j_blackboard")
        got, _ = _score([Card(rank=10, suit="Hearts")], jokers=[j],
                        held=[Card(rank=4, suit="Spades")])
        base, _ = _score([Card(rank=10, suit="Hearts")])
        assert got == base * 3


class TestRaisedFist:
    def test_uses_lowest_held_rank(self):
        j = JokerInstance("j_raised_fist")
        got, _ = _score([TEN()], jokers=[j],
                        held=[Card(rank=4, suit="Spades"),
                              Card(rank=9, suit="Hearts")])
        # 15 chips, mult 1 + 2*4 = 9
        assert got == 15 * 9

    def test_ignores_played_cards(self):
        j = JokerInstance("j_raised_fist")
        got, _ = _score([Card(rank=2, suit="Spades")], jokers=[j],
                        held=[Card(rank=9, suit="Hearts")])
        # 5 + 2 = 7 chips, mult 1 + 18 = 19  (uses held 9, not played 2)
        assert got == 7 * 19


# ── Full-deck jokers ──────────────────────────────────────────────────────────

class TestSteelJoker:
    def test_scales_multiplicatively_with_deck_steel_count(self):
        j = JokerInstance("j_steel_joker")
        deck = [Card(rank=2, suit="Spades", enhancement="Steel") for _ in range(4)]
        base, _ = _score([TEN()])
        got, _ = _score([TEN()], jokers=[j], full_deck=deck)
        assert got == int(base * (1.0 + 0.2 * 4))

    def test_no_steel_in_deck_is_neutral(self):
        j = JokerInstance("j_steel_joker")
        base, _ = _score([TEN()])
        got, _ = _score([TEN()], jokers=[j],
                        full_deck=[Card(rank=2, suit="Spades")])
        assert got == base

    def test_played_steel_alone_does_not_feed_it(self):
        """It reads the deck, not the played selection."""
        j = JokerInstance("j_steel_joker")
        steel = Card(rank=10, suit="Spades", enhancement="Steel")
        got, _ = _score([steel], jokers=[j], full_deck=[])
        without, _ = _score([steel], jokers=[])
        assert got == without


class TestStoneJoker:
    def test_counts_stone_cards_in_deck(self):
        j = JokerInstance("j_stone")
        deck = [Card(rank=2, suit="Spades", enhancement="Stone") for _ in range(3)]
        got, _ = _score([TEN()], jokers=[j], full_deck=deck)
        # 15 chips + 75, mult 1
        assert got == (15 + 75)

    def test_empty_deck_is_neutral(self):
        j = JokerInstance("j_stone")
        base, _ = _score([TEN()])
        got, _ = _score([TEN()], jokers=[j], full_deck=[])
        assert got == base


class TestDriversLicense:
    def test_fires_at_sixteen_enhanced_cards_in_deck(self):
        j = JokerInstance("j_drivers_license")
        deck = [Card(rank=2, suit="Spades", enhancement="Bonus") for _ in range(16)]
        base, _ = _score([TEN()])
        got, _ = _score([TEN()], jokers=[j], full_deck=deck)
        assert got == base * 3

    def test_does_not_fire_at_fifteen(self):
        j = JokerInstance("j_drivers_license")
        deck = [Card(rank=2, suit="Spades", enhancement="Bonus") for _ in range(15)]
        base, _ = _score([TEN()])
        got, _ = _score([TEN()], jokers=[j], full_deck=deck)
        assert got == base

    def test_unenhanced_cards_are_not_counted(self):
        """enhancement defaults to the string "None", not a falsy value."""
        j = JokerInstance("j_drivers_license")
        deck = [Card(rank=2, suit="Spades") for _ in range(40)]
        base, _ = _score([TEN()])
        got, _ = _score([TEN()], jokers=[j], full_deck=deck)
        assert got == base, "plain cards must not count as enhanced"

    def test_is_reachable_at_all(self):
        """With a 5-card played selection it could never reach 16 — the old bug."""
        j = JokerInstance("j_drivers_license")
        deck = [Card(rank=2, suit="Spades", enhancement="Mult") for _ in range(20)]
        got, _ = _score([TEN()], jokers=[j], full_deck=deck)
        base, _ = _score([TEN()])
        assert got > base


# ── Played-selection jokers must be unaffected ────────────────────────────────

class TestPlayedSelectionJokersUnchanged:
    def test_flower_pot_reads_played_cards(self):
        j = JokerInstance("j_flower_pot")
        played = [Card(rank=2, suit="Spades"), Card(rank=3, suit="Hearts"),
                  Card(rank=4, suit="Clubs"), Card(rank=5, suit="Diamonds")]
        base, _ = _score(played)
        got, _ = _score(played, jokers=[j])
        assert got == base * 3

    def test_flower_pot_ignores_held_cards(self):
        j = JokerInstance("j_flower_pot")
        played = [Card(rank=2, suit="Spades")]
        held = [Card(rank=3, suit="Hearts"), Card(rank=4, suit="Clubs"),
                Card(rank=5, suit="Diamonds")]
        base, _ = _score(played)
        got, _ = _score(played, jokers=[j], held=held)
        assert got == base


# ── Integration through the real game ─────────────────────────────────────────

def test_held_steel_multiplies_in_a_real_game():
    from balatro_sim.game import BalatroGame
    g = BalatroGame(seed=101)
    g.step({"type": "play_blind"})
    # make everything except the played card a Steel card held in hand
    g.hand[0] = Card(rank=10, suit="Spades")
    for c in g.hand[1:]:
        c.enhancement = "Steel"
    n_held = len(g.hand) - 1
    g.step({"type": "play", "cards": [0]})
    expected = int(15 * (1.5 ** n_held))
    assert g.chips_scored == expected, f"{g.chips_scored} != {expected}"
