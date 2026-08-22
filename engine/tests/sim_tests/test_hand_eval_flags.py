"""
test_hand_eval_flags.py — Four Fingers / Shortcut / Smeared Joker / Pareidolia in
``hand_eval.evaluate_hand`` (port of misc_functions.lua:376-620), and the play path
computing those flags from the board BEFORE evaluation.

Phase 1 W5 (P1-sweep).
"""
import pytest

from balatro_sim.card import Card
from balatro_sim.hand_eval import evaluate_hand, _get_straight, _get_flush
from balatro_sim.game import BalatroGame, State
from balatro_sim.jokers.base import JokerInstance, hand_eval_flags
import balatro_sim.jokers  # noqa: F401

S, H, D, C = "Spades", "Hearts", "Diamonds", "Clubs"


def c(rank, suit=S, enh="None"):
    return Card(rank=rank, suit=suit, enhancement=enh)


def ranks(cards):
    return sorted(x.rank for x in cards)


# ─────────────────────────────────────────────────────────────────────────────
# Four Fingers
# ─────────────────────────────────────────────────────────────────────────────

class TestFourFingers:
    def test_four_card_flush(self):
        hand = [c(2, H), c(5, H), c(7, H), c(9, H)]
        assert evaluate_hand(hand)[0] == "High Card"
        ht, sc = evaluate_hand(hand, four_fingers=True)
        assert ht == "Flush" and len(sc) == 4

    def test_four_of_five_flush_scores_only_the_four(self):
        """get_flush returns the matching cards: the off-suit 5th card does not score."""
        hand = [c(2, H), c(5, H), c(7, H), c(9, H), c(13, S)]
        ht, sc = evaluate_hand(hand, four_fingers=True)
        assert ht == "Flush"
        assert [x.suit for x in sc] == [H, H, H, H]

    def test_four_card_straight(self):
        hand = [c(5, S), c(6, H), c(7, C), c(8, D)]
        assert evaluate_hand(hand)[0] == "High Card"
        ht, sc = evaluate_hand(hand, four_fingers=True)
        assert ht == "Straight" and ranks(sc) == [5, 6, 7, 8]

    def test_wheel_a234_with_four_fingers(self):
        """A-2-3-4 is a 4-card straight (j == 1 stands for the Ace)."""
        hand = [c(14, S), c(2, H), c(3, C), c(4, D)]
        ht, sc = evaluate_hand(hand, four_fingers=True)
        assert ht == "Straight" and ranks(sc) == [2, 3, 4, 14]

    def test_a234_plus_off_card_scores_four(self):
        hand = [c(14, S), c(2, H), c(3, C), c(4, D), c(13, S)]
        ht, sc = evaluate_hand(hand, four_fingers=True)
        assert ht == "Straight" and ranks(sc) == [2, 3, 4, 14]

    def test_jqka_with_four_fingers(self):
        hand = [c(11, S), c(12, H), c(13, C), c(14, D)]
        ht, sc = evaluate_hand(hand, four_fingers=True)
        assert ht == "Straight" and ranks(sc) == [11, 12, 13, 14]

    def test_no_wraparound_kA23(self):
        hand = [c(13, S), c(14, H), c(2, C), c(3, D)]
        assert evaluate_hand(hand, four_fingers=True)[0] == "High Card"

    def test_three_cards_never_enough(self):
        assert evaluate_hand([c(2, H), c(3, H), c(4, H)], four_fingers=True)[0] == "High Card"

    def test_straight_flush_union_of_different_subsets(self):
        """2H 3H 4H 5S KH: flush = the four Hearts, straight = 2-3-4-5 -> Straight Flush
        scoring all five (misc_functions.lua:426-437 union)."""
        hand = [c(2, H), c(3, H), c(4, H), c(5, S), c(13, H)]
        ht, sc = evaluate_hand(hand, four_fingers=True)
        assert ht == "Straight Flush" and len(sc) == 5

    def test_flush_five_with_four_suited(self):
        hand = [c(14, H), c(14, H), c(14, H), c(14, H), c(14, S)]
        assert evaluate_hand(hand)[0] == "Five of a Kind"
        ht, sc = evaluate_hand(hand, four_fingers=True)
        assert ht == "Flush Five" and len(sc) == 5

    def test_flush_house_with_four_suited(self):
        hand = [c(9, C), c(9, C), c(9, S), c(4, C), c(4, C)]
        assert evaluate_hand(hand)[0] == "Full House"
        assert evaluate_hand(hand, four_fingers=True)[0] == "Flush House"

    def test_stone_counts_toward_hand_size_but_not_flush(self):
        """#hand includes the Stone for the size floor; flush_count does not."""
        hand = [c(2, H), c(5, H), c(7, H), c(9, H), c(3, S, "Stone")]
        ht, sc = evaluate_hand(hand, four_fingers=True)
        assert ht == "Flush" and len(sc) == 5       # 4 Hearts + the Stone scores too
        hand = [c(2, H), c(5, H), c(7, H), c(3, S, "Stone"), c(4, S, "Stone")]
        assert evaluate_hand(hand, four_fingers=True)[0] == "High Card"

    def test_four_fingers_does_not_affect_pairs(self):
        hand = [c(5), c(5, H), c(7, C), c(9, D), c(11)]
        assert evaluate_hand(hand, four_fingers=True)[0] == "Pair"


# ─────────────────────────────────────────────────────────────────────────────
# Shortcut
# ─────────────────────────────────────────────────────────────────────────────

class TestShortcut:
    def test_one_gap(self):
        hand = [c(2, S), c(4, H), c(6, C), c(8, D), c(10, S)]
        assert evaluate_hand(hand)[0] == "High Card"
        ht, sc = evaluate_hand(hand, shortcut=True)
        assert ht == "Straight" and len(sc) == 5

    def test_two_consecutive_gaps_break(self):
        hand = [c(2, S), c(5, H), c(6, C), c(7, D), c(8, S)]
        assert evaluate_hand(hand, shortcut=True)[0] == "High Card"

    def test_gap_then_gap_later_ok(self):
        hand = [c(2, S), c(3, H), c(5, C), c(7, D), c(8, S)]
        assert evaluate_hand(hand, shortcut=True)[0] == "Straight"

    def test_tjqka_irrelevant(self):
        hand = [c(10, S), c(11, H), c(12, C), c(13, D), c(14, S)]
        assert evaluate_hand(hand)[0] == "Straight"
        assert evaluate_hand(hand, shortcut=True)[0] == "Straight"

    def test_no_skip_past_the_ace_high(self):
        """9-J-Q-K + A works (skip 10); Q-K-A + 2-3 does not (no wrap; j == 14 can't skip)."""
        assert evaluate_hand([c(9), c(11, H), c(12, C), c(13, D), c(14, S)], shortcut=True)[0] == "Straight"
        assert evaluate_hand([c(12), c(13, H), c(14, C), c(2, D), c(3, S)], shortcut=True)[0] == "High Card"

    def test_wheel_with_gap(self):
        """A-2-4-5-6: Ace low, skip the 3."""
        hand = [c(14, S), c(2, H), c(4, C), c(5, D), c(6, S)]
        assert evaluate_hand(hand, shortcut=True)[0] == "Straight"

    def test_shortcut_plus_four_fingers(self):
        hand = [c(2, S), c(3, H), c(5, C), c(6, D)]
        assert evaluate_hand(hand, four_fingers=True)[0] == "High Card"
        assert evaluate_hand(hand, shortcut=True)[0] == "High Card"
        assert evaluate_hand(hand, four_fingers=True, shortcut=True)[0] == "Straight"

    def test_straight_flush_with_shortcut(self):
        hand = [c(2, H), c(4, H), c(5, H), c(6, H), c(8, H)]
        assert evaluate_hand(hand)[0] == "Flush"
        assert evaluate_hand(hand, shortcut=True)[0] == "Straight Flush"

    def test_duplicate_rank_in_straight_scores_all(self):
        hand = [c(2, S), c(3, H), c(3, C), c(5, D), c(6, S)]
        ht, sc = evaluate_hand(hand, shortcut=True, four_fingers=True)
        assert ht == "Straight" and len(sc) == 5


# ─────────────────────────────────────────────────────────────────────────────
# Smeared Joker
# ─────────────────────────────────────────────────────────────────────────────

class TestSmeared:
    def test_hearts_and_diamonds_flush(self):
        hand = [c(2, H), c(5, D), c(7, H), c(9, D), c(11, H)]
        assert evaluate_hand(hand)[0] == "High Card"
        ht, sc = evaluate_hand(hand, smeared=True)
        assert ht == "Flush" and len(sc) == 5

    def test_spades_and_clubs_flush(self):
        hand = [c(2, S), c(5, C), c(7, S), c(9, C), c(11, C)]
        assert evaluate_hand(hand, smeared=True)[0] == "Flush"

    def test_red_and_black_mixed_is_not_a_flush(self):
        hand = [c(2, S), c(5, C), c(7, H), c(9, C), c(11, C)]
        assert evaluate_hand(hand, smeared=True)[0] == "High Card"

    def test_smeared_straight_flush(self):
        hand = [c(5, H), c(6, D), c(7, H), c(8, D), c(9, H)]
        assert evaluate_hand(hand, smeared=True)[0] == "Straight Flush"

    def test_smeared_plus_four_fingers(self):
        hand = [c(2, H), c(5, D), c(7, H), c(9, D), c(11, S)]
        assert evaluate_hand(hand, smeared=True)[0] == "High Card"
        ht, sc = evaluate_hand(hand, smeared=True, four_fingers=True)
        assert ht == "Flush" and len(sc) == 4 and all(x.suit in (H, D) for x in sc)

    def test_wild_completes_smeared_flush(self):
        hand = [c(2, H), c(5, D), c(7, H), c(9, D), c(11, S, "Wild")]
        assert evaluate_hand(hand, smeared=True)[0] == "Flush"


# ─────────────────────────────────────────────────────────────────────────────
# Wild / Stone / Pareidolia edge cases (the rewritten get_flush / get_highest)
# ─────────────────────────────────────────────────────────────────────────────

class TestEdges:
    def test_wild_fills_any_suit(self):
        hand = [c(2, H), c(5, H), c(7, H), c(9, H), c(11, S, "Wild")]
        ht, sc = evaluate_hand(hand)
        assert ht == "Flush" and len(sc) == 5

    def test_debuffed_wild_is_its_base_suit(self):
        w = c(11, S, "Wild"); w.debuffed = True
        hand = [c(2, H), c(5, H), c(7, H), c(9, H), w]
        assert evaluate_hand(hand)[0] == "High Card"

    def test_first_suit_in_lua_order_wins_for_wild_ties(self):
        """Two Hearts + two Clubs + a Wild: get_flush walks Spades, Hearts, Clubs, Diamonds
        -> neither reaches 5; with Four Fingers Hearts (3 incl. Wild) still < 4."""
        hand = [c(2, H), c(5, H), c(7, C), c(9, C), c(11, S, "Wild")]
        assert evaluate_hand(hand, four_fingers=True)[0] == "High Card"

    def test_all_stones_is_high_card_of_stones(self):
        hand = [c(2, H, "Stone"), c(5, H, "Stone")]
        ht, sc = evaluate_hand(hand)
        assert ht == "High Card" and len(sc) == 2

    def test_stone_never_pairs(self):
        hand = [c(5, H, "Stone"), c(5, S)]
        ht, sc = evaluate_hand(hand)
        assert ht == "High Card" and sc[0].rank == 5 and sc[0].enhancement == "None" and len(sc) == 2

    def test_high_card_picks_highest_rank(self):
        hand = [c(2, H), c(13, S), c(7, D)]
        ht, sc = evaluate_hand(hand)
        assert ht == "High Card" and sc[0].rank == 13

    def test_pareidolia_flag_is_accepted_and_neutral(self):
        hand = [c(2, H), c(5, D), c(7, H), c(9, D), c(11, H)]
        assert evaluate_hand(hand, pareidolia=True) == evaluate_hand(hand)

    def test_two_pair_scoring_cards(self):
        hand = [c(5), c(5, H), c(9), c(9, H), c(11)]
        ht, sc = evaluate_hand(hand)
        assert ht == "Two Pair" and ranks(sc) == [5, 5, 9, 9]
        assert [x.rank for x in sc[:2]] == [9, 9]     # highest pair first (get_X_same order)

    def test_helpers_size_floor(self):
        assert _get_straight([c(2), c(3), c(4), c(5), c(6), c(7)], False, False) == []   # > 5 cards
        assert _get_flush([c(2, H)] * 6, False, False) == []


# ─────────────────────────────────────────────────────────────────────────────
# Play path: flags come from the board BEFORE evaluate_hand
# ─────────────────────────────────────────────────────────────────────────────

def _game_with(jokers, hand_specs):
    g = BalatroGame(seed="7I4M53DL")
    for k in jokers:
        g.debug_add_joker(k)
    g.step({"type": "play_blind"})
    cards = [c(r, s) for r, s in hand_specs]
    g.full_deck = [x for x in g.full_deck if x not in g.hand] + cards
    g.hand = list(cards)
    return g, cards


class TestPlayPath:
    def test_hand_eval_flags_helper(self):
        flags = hand_eval_flags([JokerInstance("j_four_fingers"), JokerInstance("j_smeared")])
        assert flags == {"four_fingers": True, "shortcut": False, "smeared": True, "pareidolia": False}
        assert hand_eval_flags([]) == {"four_fingers": False, "shortcut": False, "smeared": False, "pareidolia": False}

    @pytest.mark.parametrize("joker, hand, expect", [
        ("j_four_fingers", [(2, H), (5, H), (7, H), (9, H)], "Flush"),
        ("j_shortcut", [(2, S), (4, H), (6, C), (8, D), (10, S)], "Straight"),
        ("j_smeared", [(2, H), (5, D), (7, H), (9, D), (11, H)], "Flush"),
    ])
    def test_flag_joker_changes_played_hand_type(self, joker, hand, expect):
        g, cards = _game_with([joker], hand)
        g.step({"type": "play", "cards": list(range(len(cards)))})
        assert g.last_played_hand_type == expect
        g0, cards0 = _game_with([], hand)
        g0.step({"type": "play", "cards": list(range(len(cards0)))})
        assert g0.last_played_hand_type == "High Card"

    def test_game_hand_eval_flags(self):
        g = BalatroGame(seed="ALEEB")
        g.debug_add_joker("j_shortcut")
        assert g.hand_eval_flags()["shortcut"] is True
        assert g.hand_eval_flags([])["shortcut"] is False

    def test_crimson_heart_disabled_flag_joker_is_not_found(self):
        """find_joker skips debuffed jokers: with only Four Fingers on the board, Crimson
        Heart always disables it, so a 4-card flush is a High Card."""
        g, cards = _game_with([], [(2, H), (5, H), (7, H), (9, H)])
        g.debug_add_joker("j_four_fingers")
        g.current_blind.is_boss = True
        g.current_blind.kind = "Boss"
        g.current_blind.boss_key = "bl_final_heart"
        g.step({"type": "play", "cards": [0, 1, 2, 3]})
        assert g.last_played_hand_type == "High Card"
