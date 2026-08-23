"""
test_boss_blinds_h4.py — showdown bosses, chip multipliers, previously-inert bosses.

Regression tests for audit finding H4. Before 2026-07-29:
  - _prepare_next_blind drew from one mixed pool for every ante, so ante 8's
    finisher was an ordinary boss and the sim's final blind was materially
    easier than the real game's at the point runs are decided.
  - bl_goad debuffed Clubs (that is The Club's effect; the real Goad targets
    Spades), and The Club did not exist.
  - bl_wall, bl_house, bl_mark, bl_pillar and bl_grim sat in the selection pool
    with no implementation, so a large share of boss blinds were a plain
    2x-target blind with no gimmick.
  - bl_wall's target was documented as "+100 chips" (real: 4x base) and
    bl_needle kept a full 2x target (real: 1x).
  - bl_ox, bl_final_leaf, bl_final_acorn and bl_final_bell had comments describing entirely
    different mechanics from the real ones.

Reference: https://balatrowiki.org/w/Blinds
"""
import pytest

from balatro_sim.card import Card
from balatro_sim.constants import BLIND_CHIPS
from balatro_sim.game import (
    BalatroGame, BlindInfo, State,
    REGULAR_BOSS_BLINDS, SHOWDOWN_BOSS_BLINDS, UNMODELLED_BOSS_BLINDS,
    ALL_REGULAR_BOSS_BLINDS,
    BOSS_CHIP_MULT,
)
from balatro_sim.jokers.base import JokerInstance


def _boss_game(boss_key, target=10**9, seed=5, jokers=()):
    g = BalatroGame(seed=seed)
    g.jokers = [JokerInstance(k) for k in jokers]
    g.current_blind = BlindInfo(
        "x", "Boss", target, is_boss=True, boss_key=boss_key,
        is_showdown=boss_key in SHOWDOWN_BOSS_BLINDS)
    g._start_blind()
    return g


def _pair_into_hand(g):
    g.hand[0] = Card(rank=7, suit="Spades")
    g.hand[1] = Card(rank=7, suit="Hearts")


class TestShowdownBosses:
    def test_ante_8_always_draws_a_showdown_boss(self):
        for seed in range(40):
            g = BalatroGame(seed=seed)
            g.ante, g.blind_idx = 8, 2
            g._prepare_next_blind()
            assert g.current_blind.boss_key in SHOWDOWN_BOSS_BLINDS
            assert g.current_blind.is_showdown

    def test_earlier_antes_never_draw_a_showdown_boss(self):
        for seed in range(15):
            for ante in range(1, 8):
                g = BalatroGame(seed=seed)
                g.ante, g.blind_idx = ante, 2
                g._prepare_next_blind()
                # W2: drawn by generate.next_boss from the game's pool (face-down
                # bosses included; their effect is still unmodelled -- W5)
                assert g.current_blind.boss_key in ALL_REGULAR_BOSS_BLINDS
                assert not g.current_blind.is_showdown

    def test_showdown_pays_eight_dollars(self):
        g = BalatroGame(seed=1)
        g.ante, g.blind_idx = 8, 2
        g._prepare_next_blind()
        assert g.current_blind.money_reward == 8

    def test_unmodelled_bosses_are_excluded_from_pools(self):
        """Face-down bosses need hidden-information support; excluded, not inert."""
        for key in UNMODELLED_BOSS_BLINDS:
            assert key not in REGULAR_BOSS_BLINDS
            assert key not in SHOWDOWN_BOSS_BLINDS

    def test_grim_is_gone(self):
        """bl_grim was in the pool but is not a real Balatro boss."""
        assert "bl_grim" not in REGULAR_BOSS_BLINDS + SHOWDOWN_BOSS_BLINDS


class TestChipMultipliers:
    def _target_for(self, ante, boss_key, tries=600):
        for seed in range(tries):
            g = BalatroGame(seed=seed)
            g.ante, g.blind_idx = ante, 2
            g._prepare_next_blind()
            if g.current_blind.boss_key == boss_key:
                return g.current_blind.chips_target
        pytest.fail(f"never drew {boss_key} in {tries} seeds")

    def test_wall_is_double_a_normal_boss(self):
        assert self._target_for(3, "bl_wall") == BLIND_CHIPS[3][2] * 2

    def test_needle_is_half_a_normal_boss(self):
        assert self._target_for(3, "bl_needle") == int(BLIND_CHIPS[3][2] * 0.5)

    def test_violet_vessel_is_triple(self):
        assert self._target_for(8, "bl_final_vessel") == BLIND_CHIPS[8][2] * 3

    def test_unscaled_boss_matches_the_table(self):
        assert self._target_for(3, "bl_hook") == BLIND_CHIPS[3][2]

    def test_multiplier_table_only_lists_real_bosses(self):
        for key in BOSS_CHIP_MULT:
            assert key in REGULAR_BOSS_BLINDS + SHOWDOWN_BOSS_BLINDS


class TestTheOx:
    def test_zeroes_money_on_most_used_hand_type(self):
        g = _boss_game("bl_ox")
        g._hand_type_counts = {"Pair": 5, "High Card": 1}
        g.dollars = 30
        _pair_into_hand(g)
        g._play_hand([0, 1])
        assert g.dollars == 0

    def test_spares_other_hand_types(self):
        g = _boss_game("bl_ox")
        g._hand_type_counts = {"Flush": 5}
        g.dollars = 30
        _pair_into_hand(g)
        g._play_hand([0, 1])
        assert g.dollars == 30

    def test_is_not_hardcoded_to_flush(self):
        """The old comment claimed 'playing Flush sets money to $0'."""
        g = _boss_game("bl_ox")
        g._hand_type_counts = {"Pair": 9}
        g.dollars = 25
        g.hand[0] = Card(rank=2, suit="Spades")
        g.hand[1] = Card(rank=5, suit="Hearts")
        g.hand[2] = Card(rank=9, suit="Clubs")
        g._play_hand([0])          # High Card, not the most-used type
        assert g.dollars == 25


class TestTheArm:
    def test_reduces_played_hand_level(self):
        g = _boss_game("bl_arm")
        g.planet_levels["Pair"] = 4
        _pair_into_hand(g)
        g._play_hand([0, 1])
        assert g.planet_levels["Pair"] == 3

    def test_will_not_go_below_level_one(self):
        g = _boss_game("bl_arm")
        g.planet_levels["Pair"] = 1
        _pair_into_hand(g)
        g._play_hand([0, 1])
        assert g.planet_levels["Pair"] == 1

    def test_reduction_is_permanent_across_blinds(self):
        g = _boss_game("bl_arm")
        g.planet_levels["Pair"] = 5
        _pair_into_hand(g)
        g._play_hand([0, 1])
        g.state = State.ROUND_EVAL
        g._end_round()
        assert g.planet_levels["Pair"] == 4


class TestTheCrimsonHeart:
    def test_disables_exactly_one_joker(self):
        """Two identical +4 Mult jokers, but only one should contribute."""
        g = _boss_game("bl_final_heart", jokers=["j_joker", "j_joker"])
        g.hand[0] = Card(rank=14, suit="Spades")
        g._play_hand([0])
        # High Card 5 + 11 = 16 chips; mult 1 + 4 = 5  ->  80
        assert g.chips_scored == 80

    def test_without_the_boss_both_jokers_fire(self):
        """Control for the test above. Uses bl_water (0 discards, no scoring
        effect) rather than bl_hook, which randomly removes played cards and so
        makes the surviving card's chip value non-deterministic."""
        g = _boss_game("bl_water", jokers=["j_joker", "j_joker"])
        g.hand[0] = Card(rank=14, suit="Spades")
        g._play_hand([0])
        assert g.chips_scored == 16 * 9   # mult 1 + 4 + 4

    def test_no_crash_with_zero_jokers(self):
        g = _boss_game("bl_final_heart")
        g.hand[0] = Card(rank=14, suit="Spades")
        g._play_hand([0])
        assert g.chips_scored == 16


class TestTheVerdantLeaf:
    def test_debuffs_everything_until_a_joker_is_sold(self):
        g = _boss_game("bl_final_leaf", jokers=["j_joker"])
        assert all(c.debuffed for c in g.full_deck)
        assert g._verdant_active

        g.state = State.SHOP
        g.step({"type": "sell_joker", "joker_idx": 0})
        assert not g._verdant_active
        assert all(not c.debuffed for c in g.full_deck)

    def test_selling_outside_the_boss_is_harmless(self):
        g = _boss_game("bl_hook", jokers=["j_joker"])
        g.state = State.SHOP
        g.step({"type": "sell_joker", "joker_idx": 0})
        assert not g._verdant_active


class TestTheAmberAcorn:
    def test_shuffles_jokers(self):
        """Order affects scoring after audit C4, so this is a real disruption."""
        for seed in range(60):
            g = BalatroGame(seed=seed)
            g.jokers = [JokerInstance(f"j_stub{i}") for i in range(5)]
            before = [j.key for j in g.jokers]
            g.current_blind = BlindInfo("x", "Boss", 100, is_boss=True,
                                        boss_key="bl_final_acorn", is_showdown=True)
            g._start_blind()
            if [j.key for j in g.jokers] != before:
                return
        pytest.fail("joker order never changed across 60 seeds")


class TestTheCeruleanBell:
    def test_forces_a_card_to_be_selected(self):
        g = _boss_game("bl_final_bell")
        forced_before = g._forced_card_id
        assert forced_before == -1
        g._play_hand([0])
        assert g._forced_card_id >= 0
        # the chosen card plus the forced one both left the hand
        assert len(g.discard_pile) == 2

    def test_forced_card_is_stable_within_a_blind(self):
        g = _boss_game("bl_final_bell")
        g._play_hand([0])
        first = g._forced_card_id
        g._play_hand([0])
        assert g._forced_card_id == first

    def test_forced_card_resets_between_blinds(self):
        g = _boss_game("bl_final_bell")
        g._play_hand([0])
        assert g._forced_card_id >= 0
        g._start_blind()
        assert g._forced_card_id == -1


class TestThePillar:
    def test_debuffs_cards_played_earlier_this_ante(self):
        g = BalatroGame(seed=5)
        g.blind_idx = 0
        g._prepare_next_blind()
        g._start_blind()
        played = list(g.hand[:2])
        g._play_hand([0, 1])
        assert all(c.id in g._played_this_ante for c in played)

        g.current_blind = BlindInfo("x", "Boss", 100, is_boss=True,
                                    boss_key="bl_pillar")
        g._start_blind()
        debuffed_ids = {c.id for c in g.full_deck if c.debuffed}
        assert debuffed_ids == g._played_this_ante

    def test_cards_played_at_the_boss_itself_are_not_tracked(self):
        g = _boss_game("bl_pillar")
        before = set(g._played_this_ante)
        g._play_hand([0, 1])
        assert g._played_this_ante == before

    def test_tracking_clears_between_antes(self):
        g = BalatroGame(seed=5)
        g._start_blind()
        g._play_hand([0, 1])
        assert g._played_this_ante
        g.blind_idx = 2
        g._end_shop()
        assert g._played_this_ante == set()


class TestEveryPooledBossIsPlayable:
    @pytest.mark.parametrize("boss_key",
                             REGULAR_BOSS_BLINDS + SHOWDOWN_BOSS_BLINDS)
    def test_boss_survives_a_discard_and_a_hand(self, boss_key):
        """Smoke test: no boss in a selection pool may crash, and all debuffs
        must be cleared when the blind ends."""
        g = _boss_game(boss_key, jokers=["j_joker"], seed=9)
        if g.discards_left > 0:
            g._discard([0])
        g._play_hand(list(range(min(5, len(g.hand)))))
        g._undo_boss_debuffs(boss_key)
        assert all(not c.debuffed for c in g.full_deck)


class TestTheHookPoolsUnplayedCardsOnly:
    """state_events.lua:478-488 moves every played card to G.play BEFORE
    Blind:press_play(), so The Hook's two 'hook' draws (blind.lua:470-484) are over the
    UNPLAYED cards only: the played hand always scores in full.  Phase 5 lead fix."""

    def test_played_cards_always_score(self):
        for seed in range(40):
            g = _boss_game("bl_hook", seed=seed)
            g.hand[0] = Card(rank=14, suit="Spades")
            g.hand[1] = Card(rank=14, suit="Hearts")
            g.hand[2] = Card(rank=14, suit="Clubs")
            g.hand[3] = Card(rank=14, suit="Diamonds")
            g.hand[4] = Card(rank=13, suit="Spades")
            played = [g.hand[i] for i in range(5)]
            g._play_hand([0, 1, 2, 3, 4])
            # Four of a Kind: 60 chips + 4 aces (11 each) = 104, mult 7 — the King kicker
            # does not score; a Hook that ate a played ace would drop this to a Trips/Pair.
            assert g.chips_scored == 104 * 7, (seed, g.chips_scored)
            assert all(c in g.discard_pile for c in played)

    def test_two_unplayed_cards_leave_the_hand(self):
        g = _boss_game("bl_hook")
        n = len(g.hand)
        assert n >= 4
        unplayed_before = set(id(c) for c in g.hand[1:])
        g._play_hand([0])
        # 1 played + 2 hooked leave; draw-to-full refills, so count what is gone instead
        gone = [c for c in g.discard_pile]
        hooked = [c for c in gone if id(c) in unplayed_before]
        assert len(hooked) == 2
