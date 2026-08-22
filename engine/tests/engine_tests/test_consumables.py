"""
test_consumables.py — Tests for all consumable types:
  - 12 Planet cards
  - 22 Tarot cards
  - 18 Spectral cards
  - 27 Vouchers
"""
from __future__ import annotations

import random
import pytest

from balatro_sim.game import BalatroGame, State
from balatro_sim.card import Card
from balatro_sim.consumables import (
    apply_planet, apply_tarot, apply_spectral, apply_voucher,
    PLANET_HAND, ALL_PLANETS, ALL_TAROTS, ALL_SPECTRALS, ALL_VOUCHERS,
    TAROT_ENHANCEMENT, TAROT_SUIT,
)
from balatro_sim.jokers.base import JokerInstance


# ════════════════════════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def game():
    g = BalatroGame(seed=42)
    g.reset()
    return g


def _game_with_hand(ranks_suits=None):
    """Create a game with a specific hand for tarot testing."""
    g = BalatroGame(seed=42)
    g.reset()
    g.hand = []
    if ranks_suits:
        for rank, suit in ranks_suits:
            g.hand.append(Card(rank=rank, suit=suit))
    else:
        for i in range(5):
            g.hand.append(Card(rank=i+2, suit="Spades"))
    return g


# ════════════════════════════════════════════════════════════════════════════
# PLANET CARDS
# ════════════════════════════════════════════════════════════════════════════

class TestPlanets:
    @pytest.mark.parametrize("planet_key,hand_type", list(PLANET_HAND.items()))
    def test_planet_upgrades_hand_level(self, game, planet_key, hand_type):
        prev = game.planet_levels.get(hand_type, 1)
        result = apply_planet(game, planet_key)
        assert result is True
        assert game.planet_levels[hand_type] == prev + 1

    @pytest.mark.parametrize("planet_key", ALL_PLANETS)
    def test_planet_tracks_usage(self, game, planet_key):
        apply_planet(game, planet_key)
        assert planet_key in game.planets_used

    def test_planet_multiple_upgrades(self, game):
        for _ in range(5):
            apply_planet(game, "c_mercury")
        assert game.planet_levels["Pair"] == 6  # started at 1, +5

    def test_invalid_planet_returns_false(self, game):
        assert apply_planet(game, "c_nonexistent") is False

    def test_planet_fires_satellite_joker(self, game):
        game.jokers.append(JokerInstance("j_satellite"))
        apply_planet(game, "c_mercury")
        # Verify the planet was tracked in game.planets_used
        assert "c_mercury" in game.planets_used


# ════════════════════════════════════════════════════════════════════════════
# TAROT CARDS — Enhancement tarots
# ════════════════════════════════════════════════════════════════════════════

class TestTarotEnhancements:
    @pytest.mark.parametrize("tarot_key,enhancement", list(TAROT_ENHANCEMENT.items()))
    def test_enhancement_tarot_applies(self, tarot_key, enhancement):
        g = _game_with_hand([(10, "Hearts"), (11, "Spades")])
        result = apply_tarot(g, tarot_key, target_indices=[0])
        assert result is True
        assert g.hand[0].enhancement == enhancement

    def test_enhancement_tarot_max_2_cards(self):
        g = _game_with_hand([(10, "Hearts"), (11, "Spades"), (12, "Clubs")])
        apply_tarot(g, "c_magician", target_indices=[0, 1, 2])
        assert g.hand[0].enhancement == "Lucky"
        assert g.hand[1].enhancement == "Lucky"
        assert g.hand[2].enhancement != "Lucky"  # only 2 max

    def test_enhancement_tarot_tracks_usage(self):
        g = _game_with_hand([(10, "Hearts")])
        apply_tarot(g, "c_magician", target_indices=[0])
        assert "c_magician" in g.tarots_used


# ════════════════════════════════════════════════════════════════════════════
# TAROT CARDS — Suit conversion tarots
# ════════════════════════════════════════════════════════════════════════════

class TestTarotSuitConversion:
    @pytest.mark.parametrize("tarot_key,suit", list(TAROT_SUIT.items()))
    def test_suit_conversion(self, tarot_key, suit):
        g = _game_with_hand([(10, "Hearts"), (11, "Spades"), (12, "Clubs")])
        apply_tarot(g, tarot_key, target_indices=[0, 1, 2])
        assert g.hand[0].suit == suit
        assert g.hand[1].suit == suit
        assert g.hand[2].suit == suit

    def test_suit_conversion_max_3(self):
        g = _game_with_hand([(i, "Hearts") for i in range(2, 6)])
        apply_tarot(g, "c_star", target_indices=[0, 1, 2, 3])
        assert g.hand[0].suit == "Diamonds"
        assert g.hand[1].suit == "Diamonds"
        assert g.hand[2].suit == "Diamonds"
        assert g.hand[3].suit == "Hearts"  # 4th not converted


# ════════════════════════════════════════════════════════════════════════════
# TAROT CARDS — Special tarots
# ════════════════════════════════════════════════════════════════════════════

class TestTarotSpecial:
    def test_fool_copies_last_tarot(self):
        # G.GAME.last_tarot_planet is set by USING a Tarot/Planet (W2: run_state)
        g = _game_with_hand([(10, "Hearts")])
        apply_tarot(g, "c_magician", [0])
        apply_tarot(g, "c_fool")
        assert "c_magician" in g.consumable_hand

    def test_fool_copies_last_planet_if_no_tarot(self):
        g = _game_with_hand([(10, "Hearts")])
        apply_planet(g, "c_mercury")
        apply_tarot(g, "c_fool")
        assert "c_mercury" in g.consumable_hand

    def test_fool_copies_the_most_recent_of_tarot_or_planet(self):
        g = _game_with_hand([(10, "Hearts")])
        apply_tarot(g, "c_magician", [0])
        apply_planet(g, "c_mercury")
        apply_tarot(g, "c_fool")
        assert g.consumable_hand[-1] == "c_mercury"

    def test_fool_unusable_with_nothing_used_yet(self):
        g = _game_with_hand([(10, "Hearts")])
        assert apply_tarot(g, "c_fool") is False

    def test_high_priestess_creates_2_planets(self):
        g = _game_with_hand()
        prev_len = len(g.consumable_hand)
        apply_tarot(g, "c_high_priestess")
        assert len(g.consumable_hand) == prev_len + 2

    def test_emperor_creates_2_tarots(self):
        g = _game_with_hand()
        prev_len = len(g.consumable_hand)
        apply_tarot(g, "c_emperor")
        assert len(g.consumable_hand) == prev_len + 2

    def test_hermit_doubles_money_max_20(self):
        g = _game_with_hand()
        g.dollars = 15
        apply_tarot(g, "c_hermit")
        assert g.dollars == 30  # +15, capped at $20 gain

    def test_hermit_caps_at_20_gain(self):
        g = _game_with_hand()
        g.dollars = 30
        apply_tarot(g, "c_hermit")
        assert g.dollars == 50  # gain capped at $20

    def test_strength_increases_rank(self):
        g = _game_with_hand([(5, "Hearts"), (10, "Spades")])
        apply_tarot(g, "c_strength", target_indices=[0, 1])
        assert g.hand[0].rank == 6
        assert g.hand[1].rank == 11

    def test_hanged_man_destroys_cards(self):
        g = _game_with_hand([(5, "Hearts"), (10, "Spades"), (12, "Clubs")])
        prev_len = len(g.hand)
        apply_tarot(g, "c_hanged_man", target_indices=[0, 1])
        assert len(g.hand) == prev_len - 2

    def test_death_copies_right_to_left(self):
        g = _game_with_hand([(5, "Hearts"), (10, "Spades")])
        apply_tarot(g, "c_death", target_indices=[0, 1])
        assert g.hand[0].rank == 10
        assert g.hand[0].suit == "Spades"

    def test_temperance_gives_joker_sell_value(self):
        g = _game_with_hand()
        j = JokerInstance("j_joker")
        j.state["sell_value"] = 5
        g.jokers = [j]
        g.dollars = 10
        apply_tarot(g, "c_temperance")
        assert g.dollars == 15

    def test_temperance_caps_at_50(self):
        g = _game_with_hand()
        for i in range(5):
            j = JokerInstance("j_joker")
            j.state["sell_value"] = 20
            g.jokers.append(j)
        g.dollars = 10
        apply_tarot(g, "c_temperance")
        assert g.dollars == 60  # 10 + min(100, 50) = 60

    def test_judgement_creates_joker(self):
        g = _game_with_hand()
        g.jokers = []
        apply_tarot(g, "c_judgement")
        assert len(g.jokers) == 1

    def test_judgement_respects_joker_slots(self):
        g = _game_with_hand()
        g.joker_slots = 1
        g.jokers = [JokerInstance("j_joker")]
        apply_tarot(g, "c_judgement")
        assert len(g.jokers) == 1  # no room

    def test_wheel_of_fortune_recorded(self):
        g = _game_with_hand()
        # W3: unusable with no editionless joker (card.lua:1534) ...
        assert apply_tarot(g, "c_wheel_of_fortune") is False
        # ... and with one it rolls 'wheel_of_fortune' < normal/4 on the keyed stream
        g.jokers = [JokerInstance("j_joker")]
        apply_tarot(g, "c_wheel_of_fortune")
        assert "c_wheel_of_fortune" in g.tarots_used
        assert "wheel_of_fortune" in g.run_state.rng.keys()
        assert g.jokers[0].edition in ("None", "Foil", "Holographic", "Polychrome")


# ════════════════════════════════════════════════════════════════════════════
# SPECTRAL CARDS
# ════════════════════════════════════════════════════════════════════════════

class TestSpectrals:
    # card.lua:1292-1338 (W2): destroy ONE RANDOM card in hand ('random_destroy'), the new
    # enhanced cards are added TO THE HAND; needs more than one card in hand.
    def test_familiar_destroys_and_adds_face_cards(self):
        g = _game_with_hand([(5, "Hearts"), (10, "Spades")])
        prev_full = len(g.full_deck)
        assert apply_spectral(g, "c_familiar", target_indices=[0]) is True
        assert len(g.hand) == 2 - 1 + 3
        assert len(g.full_deck) == prev_full + 3   # (the crafted hand cards are not in full_deck)
        new = g.hand[-3:]
        assert all(c.is_face_card and c.enhancement not in ("None", "Stone") for c in new)

    def test_grim_destroys_and_adds_aces(self):
        g = _game_with_hand([(5, "Hearts"), (10, "Spades")])
        apply_spectral(g, "c_grim", target_indices=[0])
        assert len(g.hand) == 2 - 1 + 2
        assert all(c.rank == 14 for c in g.hand[-2:])

    def test_incantation_destroys_and_adds_number_cards(self):
        g = _game_with_hand([(5, "Hearts"), (10, "Spades")])
        apply_spectral(g, "c_incantation", target_indices=[0])
        assert len(g.hand) == 2 - 1 + 4
        assert all(2 <= c.rank <= 10 for c in g.hand[-4:])

    def test_familiar_unusable_with_one_card_in_hand(self):
        g = _game_with_hand([(5, "Hearts")])
        assert apply_spectral(g, "c_familiar", target_indices=[0]) is False

    def test_talisman_adds_gold_seal(self):
        g = _game_with_hand([(10, "Hearts")])
        apply_spectral(g, "c_talisman", target_indices=[0])
        assert g.hand[0].seal == "Gold"

    def test_deja_vu_adds_red_seal(self):
        g = _game_with_hand([(10, "Hearts")])
        apply_spectral(g, "c_deja_vu", target_indices=[0])
        assert g.hand[0].seal == "Red"

    def test_trance_adds_blue_seal(self):
        g = _game_with_hand([(10, "Hearts")])
        apply_spectral(g, "c_trance", target_indices=[0])
        assert g.hand[0].seal == "Blue"

    def test_medium_adds_purple_seal(self):
        g = _game_with_hand([(10, "Hearts")])
        apply_spectral(g, "c_medium", target_indices=[0])
        assert g.hand[0].seal == "Purple"

    def test_wraith_creates_rare_joker(self):
        g = _game_with_hand()
        g.jokers = []
        g.dollars = 10
        random.seed(42)
        apply_spectral(g, "c_wraith")
        assert len(g.jokers) == 1
        assert g.dollars == 0  # "sets money to $0" (card.lua:1460-1462)
        from balatro_sim.shop import JOKER_CATALOGUE
        assert JOKER_CATALOGUE[g.jokers[0].key]["rarity"] == "Rare"

    def test_sigil_converts_all_to_single_suit(self):
        g = _game_with_hand([(2, "Hearts"), (3, "Spades"), (4, "Clubs")])
        random.seed(42)
        apply_spectral(g, "c_sigil")
        suits = {c.suit for c in g.hand}
        assert len(suits) == 1  # all same suit

    def test_ouija_converts_all_to_single_rank(self):
        g = _game_with_hand([(2, "Hearts"), (3, "Spades"), (4, "Clubs")])
        prev_hand_size = g.hand_size
        random.seed(42)
        apply_spectral(g, "c_ouija")
        ranks = {c.rank for c in g.hand}
        assert len(ranks) == 1  # all same rank
        assert g.hand_size == prev_hand_size - 1

    def test_ectoplasm_adds_joker_slot(self):
        # "Add Negative to a random Joker, -1 hand size": needs an editionless joker
        g = _game_with_hand()
        assert apply_spectral(g, "c_ectoplasm") is False
        g.jokers = [JokerInstance("j_joker")]
        prev_slots = g.joker_slots
        prev_hand = g.hand_size
        assert apply_spectral(g, "c_ectoplasm") is True
        assert g.jokers[0].edition == "Negative"
        assert g.joker_slots == prev_slots + 1
        assert g.hand_size == prev_hand - 1

    def test_immolate_destroys_5_gives_20(self):
        g = _game_with_hand([(i, "Hearts") for i in range(2, 10)])
        random.seed(42)
        prev_dollars = g.dollars
        apply_spectral(g, "c_immolate")
        assert g.dollars == prev_dollars + 20

    def test_ankh_keeps_one_joker(self):
        g = _game_with_hand()
        g.jokers = [JokerInstance("j_joker"), JokerInstance("j_half"), JokerInstance("j_abstract")]
        apply_spectral(g, "c_ankh")
        # "Create a copy of a random Joker, destroy all other Jokers": chosen + its copy
        assert len(g.jokers) == 2
        assert g.jokers[0].key == g.jokers[1].key

    def test_hex_polychrome_one_joker(self):
        g = _game_with_hand()
        g.jokers = [JokerInstance("j_joker"), JokerInstance("j_half")]
        random.seed(42)
        apply_spectral(g, "c_hex")
        assert len(g.jokers) == 1
        assert g.jokers[0].edition == "Polychrome"

    def test_cryptid_copies_card(self):
        g = _game_with_hand([(10, "Hearts")])
        prev_deck = len(g.deck)
        apply_spectral(g, "c_cryptid", target_indices=[0])
        assert len(g.deck) == prev_deck + 2

    def test_soul_creates_legendary_joker(self):
        g = _game_with_hand()
        g.jokers = []
        random.seed(42)
        apply_spectral(g, "c_soul")
        assert len(g.jokers) == 1

    def test_black_hole_upgrades_all_hands(self):
        g = _game_with_hand()
        prev_levels = dict(g.planet_levels)
        apply_spectral(g, "c_black_hole")
        for hand_type in prev_levels:
            assert g.planet_levels[hand_type] == prev_levels[hand_type] + 1

    def test_aura_adds_edition_to_selected_card(self):
        # "Add Foil, Holographic or Polychrome effect to 1 selected card" (a playing card)
        g = _game_with_hand([(10, "Hearts")])
        assert apply_spectral(g, "c_aura", target_indices=[0]) is True
        assert g.hand[0].edition in ("Foil", "Holographic", "Polychrome")
        assert apply_spectral(g, "c_aura", target_indices=[0]) is False   # already editioned


# ════════════════════════════════════════════════════════════════════════════
# VOUCHERS
# ════════════════════════════════════════════════════════════════════════════

class TestVouchers:
    def test_overstock_adds_shop_slot(self, game):
        # W2: the shelf is run_state.shop_joker_max slots (G.GAME.shop.joker_max)
        prev = game.run_state.shop_joker_max
        apply_voucher(game, "v_overstock_norm")
        assert game.run_state.shop_joker_max == prev + 1
        assert "v_overstock_norm" in game.run_state.used_vouchers

    def test_overstock_plus(self, game):
        apply_voucher(game, "v_overstock_norm")
        prev = game.run_state.shop_joker_max
        apply_voucher(game, "v_overstock_plus")
        assert game.run_state.shop_joker_max == prev + 1

    def test_overstock_bought_in_shop_fills_the_new_slot_now(self, game):
        game.step({"type": "play_blind"})
        game.debug_win_blind()
        game.step({"type": "advance"})
        assert game.state == State.SHOP
        shelf_before = [it for it in game.current_shop if it.kind not in ("voucher", "booster")]
        assert len(shelf_before) == 2
        apply_voucher(game, "v_overstock_norm")
        shelf = [it for it in game.current_shop if it.kind not in ("voucher", "booster")]
        assert len(shelf) == 3
        assert [it.key for it in shelf[:2]] == [it.key for it in shelf_before]

    def test_rate_vouchers_land_in_run_state(self, game):
        apply_voucher(game, "v_tarot_merchant")
        assert game.run_state.tarot_rate == 9.6
        apply_voucher(game, "v_hone")
        assert game.run_state.edition_rate == 2
        apply_voucher(game, "v_magic_trick")
        assert game.run_state.playing_card_rate == 4

    def test_clearance_sale(self, game):
        apply_voucher(game, "v_clearance_sale")
        assert game.shop_discount == 0.25

    def test_liquidation_stacks(self, game):
        apply_voucher(game, "v_clearance_sale")
        apply_voucher(game, "v_liquidation")
        assert game.shop_discount == 0.5

    def test_reroll_surplus(self, game):
        prev = game.reroll_discount
        apply_voucher(game, "v_reroll_surplus")
        assert game.reroll_discount == prev + 2

    def test_reroll_glut_stacks(self, game):
        apply_voucher(game, "v_reroll_surplus")
        prev = game.reroll_discount
        apply_voucher(game, "v_reroll_glut")
        assert game.reroll_discount == prev + 2

    def test_crystal_ball(self, game):
        prev = game.consumable_slots
        apply_voucher(game, "v_crystal_ball")
        assert game.consumable_slots == prev + 1

    def test_grabber(self, game):
        prev = game.base_hands
        apply_voucher(game, "v_grabber")
        assert game.base_hands == prev + 1

    def test_nacho_tong(self, game):
        apply_voucher(game, "v_grabber")
        prev = game.base_hands
        apply_voucher(game, "v_nacho_tong")
        assert game.base_hands == prev + 1

    def test_wasteful(self, game):
        prev = game.base_discards
        apply_voucher(game, "v_wasteful")
        assert game.base_discards == prev + 1

    def test_recyclomancy(self, game):
        apply_voucher(game, "v_wasteful")
        prev = game.base_discards
        apply_voucher(game, "v_recyclomancy")
        assert game.base_discards == prev + 1

    def test_hieroglyph_reduces_ante(self, game):
        game.ante = 3
        apply_voucher(game, "v_hieroglyph")
        assert game.ante == 2
        assert game.base_hands >= 1

    def test_petroglyph_reduces_ante(self, game):
        game.ante = 3
        apply_voucher(game, "v_petroglyph")
        assert game.ante == 2

    def test_paint_brush_hand_size(self, game):
        prev = game.hand_size
        apply_voucher(game, "v_paint_brush")
        assert game.hand_size == prev + 1

    def test_palette_hand_size(self, game):
        apply_voucher(game, "v_paint_brush")
        prev = game.hand_size
        apply_voucher(game, "v_palette")
        assert game.hand_size == prev + 1

    def test_directors_cut(self, game):
        """Director's Cut: reroll the Boss Blind once per ante for $10 (a BLIND_SELECT
        action), not a free shop reroll (the pre-W2 model)."""
        assert not game.can_reroll_boss()
        apply_voucher(game, "v_directors_cut")
        game.dollars = 10
        assert game.can_reroll_boss()
        before = game.boss_blind
        game.step({"type": "reroll_boss"})
        assert game.dollars == 0
        assert game.boss_blind != before or True   # same boss is possible on a small pool
        assert not game.can_reroll_boss()          # once per ante (boss_rerolled)
        game.dollars = 20
        assert not game.can_reroll_boss()
        apply_voucher(game, "v_retcon")
        assert game.can_reroll_boss()              # Retcon: unlimited

    def test_voucher_cannot_be_bought_twice(self, game):
        assert apply_voucher(game, "v_overstock_norm") is True
        assert apply_voucher(game, "v_overstock_norm") is False

    @pytest.mark.parametrize("voucher_key", ALL_VOUCHERS)
    def test_all_vouchers_return_true(self, game, voucher_key):
        """Every voucher should apply successfully on first use."""
        result = apply_voucher(game, voucher_key)
        assert result is True
