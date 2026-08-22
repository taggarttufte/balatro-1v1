"""
test_joker_order.py — joker position must change the score.

Regression tests for audit finding C4. scoring.py pooled every additive mult and
applied every xMult at the very end:

    total_mult = (base_mult + ctx.mult) * ctx.mult_mult

which is the best possible ordering by the distributive law. Joker position was
therefore arithmetically irrelevant, every logged score in this project was
inflated, synergy.auto_position_on_buy was pointless, and "joker positioning as
an explicit action" (listed in the V7 retrospective as never tried) was
unimplementable.

The real game applies jokers left to right against a running mult, so a +Mult
joker placed left of an xMult joker gets multiplied and one placed right does not.
Reference example from https://balatrowiki.org/w/Scoring :
    40 x ((4+4) x 2) = 640      +Mult left of xMult
    40 x ((4 x 2) + 4) = 480    reversed
"""
from balatro_sim.game_keys import core as _core
PseudoRandom = _core.PseudoRandom

import pytest

from balatro_sim.card import Card
from balatro_sim.jokers.base import JokerInstance, ScoreContext
from balatro_sim.scoring import score_hand


def _score(cards, jokers, hand_type="High Card", **kw):
    s, ctx = score_hand(
        scoring_cards=list(cards), all_cards=list(cards), hand_type=hand_type,
        jokers=list(jokers), planet_levels={hand_type: 1}, hands_left=3,
        discards_left=3, dollars=10, ante=1, deck_remaining=44,
        rng=PseudoRandom("TEST1"), **kw)
    return s


ACE = lambda: Card(rank=14, suit="Spades")     # 11 chips


class TestOrderMatters:
    def test_plus_mult_before_xmult_scores_higher(self):
        """The headline property. j_joker is +4 Mult, j_duo is x2 on a Pair."""
        pair = [Card(rank=7, suit="Spades"), Card(rank=7, suit="Hearts")]
        plus_then_x = _score(pair, [JokerInstance("j_joker"),
                                    JokerInstance("j_duo")], hand_type="Pair")
        x_then_plus = _score(pair, [JokerInstance("j_duo"),
                                    JokerInstance("j_joker")], hand_type="Pair")
        assert plus_then_x > x_then_plus, (
            f"order had no effect: {plus_then_x} vs {x_then_plus}")

    def test_exact_values_match_the_distributive_law(self):
        """Pair base 10/2, two 7s = 14 chips -> 24 chips.
        +4 then x2: (2+4)*2 = 12  -> 288
        x2 then +4: (2*2)+4 = 8   -> 192
        """
        pair = [Card(rank=7, suit="Spades"), Card(rank=7, suit="Hearts")]
        assert _score(pair, [JokerInstance("j_joker"),
                             JokerInstance("j_duo")], hand_type="Pair") == 288
        assert _score(pair, [JokerInstance("j_duo"),
                             JokerInstance("j_joker")], hand_type="Pair") == 192

    def test_three_jokers_order_sensitive(self):
        pair = [Card(rank=7, suit="Spades"), Card(rank=7, suit="Hearts")]
        a = _score(pair, [JokerInstance("j_joker"), JokerInstance("j_jolly"),
                          JokerInstance("j_duo")], hand_type="Pair")
        b = _score(pair, [JokerInstance("j_duo"), JokerInstance("j_joker"),
                          JokerInstance("j_jolly")], hand_type="Pair")
        assert a != b

    def test_two_additive_jokers_are_order_insensitive(self):
        """Sanity check: pure addition must stay commutative."""
        a = _score([ACE()], [JokerInstance("j_joker"), JokerInstance("j_jolly")])
        b = _score([ACE()], [JokerInstance("j_jolly"), JokerInstance("j_joker")])
        assert a == b

    def test_two_multiplicative_jokers_are_order_insensitive(self):
        pair = [Card(rank=7, suit="Spades"), Card(rank=7, suit="Hearts")]
        a = _score(pair, [JokerInstance("j_duo"), JokerInstance("j_glass")],
                   hand_type="Pair")
        b = _score(pair, [JokerInstance("j_glass"), JokerInstance("j_duo")],
                   hand_type="Pair")
        assert a == b


class TestCardModifiersApplyBeforeJokers:
    def test_card_xmult_precedes_joker_plus_mult(self):
        # Poly Ace: mult 1 -> x1.5 = 1.5, then j_joker +4 -> 5.5; chips 16 -> 88
        assert _score([Card(rank=14, suit="Spades", edition="Polychrome")],
                      [JokerInstance("j_joker")]) == 88

    def test_glass_precedes_joker_plus_mult(self):
        # Glass Ace: mult 1 -> x2 = 2, then +4 -> 6; chips 16 -> 96
        assert _score([Card(rank=14, suit="Spades", enhancement="Glass")],
                      [JokerInstance("j_joker")]) == 96


class TestJokerEditions:
    def test_polychrome_applies_after_its_own_joker(self):
        # +4 lands, then x1.5, then the plain joker's +4: (1+4)*1.5+4 = 11.5
        result = _score([ACE()], [JokerInstance("j_joker", edition="Polychrome"),
                                  JokerInstance("j_joker")])
        assert result == int(16 * 11.5)

    def test_polychrome_position_matters(self):
        poly_first = _score([ACE()], [JokerInstance("j_joker", edition="Polychrome"),
                                      JokerInstance("j_joker")])
        poly_last = _score([ACE()], [JokerInstance("j_joker"),
                                     JokerInstance("j_joker", edition="Polychrome")])
        # poly last: (1+4) = 5, then (5+4)*1.5 = 13.5 -> higher
        assert poly_last > poly_first

    def test_edition_does_not_fire_on_passes_where_joker_is_idle(self):
        """
        j_joker contributes in on_hand_scored only. Its Polychrome must apply once,
        not once per scoring card — the bug in the first draft of this refactor.
        """
        one_card = _score([ACE()], [JokerInstance("j_joker", edition="Polychrome")])
        assert one_card == int(16 * 5 * 1.5)   # (1+4)*1.5 = 7.5 -> 120

        five = [Card(rank=7, suit="Spades"), Card(rank=7, suit="Hearts"),
                Card(rank=8, suit="Spades"), Card(rank=8, suit="Hearts"),
                Card(rank=9, suit="Spades")]
        two_pair = _score(five, [JokerInstance("j_joker", edition="Polychrome")],
                          hand_type="Two Pair")
        # All five cards are passed as scoring_cards here, so chips are
        # 20 + (7+7+8+8+9) = 59, and mult is (2+4)*1.5 = 9  ->  531.
        # The edition fired exactly ONCE despite five scoring passes; per-pass
        # firing would have given 59 * (2+4) * 1.5**5 instead.
        assert two_pair == 531
        assert two_pair != int(59 * 6 * 1.5 ** 5)

    def test_foil_joker_adds_chips_once(self):
        plain = _score([ACE()], [JokerInstance("j_joker")])
        foil = _score([ACE()], [JokerInstance("j_joker", edition="Foil")])
        assert foil - plain == 50 * 5   # +50 chips, multiplied by mult 5

    def test_negative_edition_has_no_scoring_effect(self):
        plain = _score([ACE()], [JokerInstance("j_joker")])
        negative = _score([ACE()], [JokerInstance("j_joker", edition="Negative")])
        assert plain == negative


class TestFoldMechanics:
    def test_fold_is_the_documented_formula(self):
        ctx = ScoreContext(running_mult=2.0, mult=4.0, mult_mult=3.0)
        ctx.fold_mult()
        assert ctx.running_mult == (2.0 + 4.0) * 3.0
        assert ctx.mult == 0.0
        assert ctx.mult_mult == 1.0

    def test_fold_is_idempotent_when_nothing_pending(self):
        ctx = ScoreContext(running_mult=7.0)
        ctx.fold_mult()
        ctx.fold_mult()
        assert ctx.running_mult == 7.0

    def test_running_mult_starts_at_base_mult(self):
        """Pair base is 10 chips / 2 mult, so two 7s score (10+14) * 2."""
        pair = [Card(rank=7, suit="Spades"), Card(rank=7, suit="Hearts")]
        assert _score(pair, [], hand_type="Pair") == (10 + 14) * 2


class TestHeldSteelOrdering:
    def test_steel_multiplies_after_played_cards_score(self):
        """Steel is in-hand, so it applies after the played cards' +mult."""
        steel = Card(rank=2, suit="Hearts", enhancement="Steel")
        mult_card = Card(rank=10, suit="Spades", enhancement="Mult")  # +4 mult
        # chips 5+10 = 15; mult (1+4) = 5 then x1.5 = 7.5 -> 112
        assert _score([mult_card], [], held_cards=[steel]) == 112
