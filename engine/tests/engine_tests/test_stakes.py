"""
test_stakes.py — Phase 2 W3: the 8 stakes.  White is the vanilla run and must be a byte-
identical no-op; Red / Green / Blue / Purple have small non-sticker effects that are
implemented; Black / Orange / Gold need the sticker system (Eternal / Perishable / Rental
EFFECTS) which is a later phase — catalogued + generation flags only, effects xfail.

Ground truth: game.lua:253-260 (stake table), game.lua:2050-2057 (modifiers), blind.lua:84
(no_blind_reward zeroes blind.dollars), misc_functions.lua:919-954 (scaling tables).
"""
import pytest

from balatro_sim import stakes
from balatro_sim.constants import BLIND_CHIPS, blind_base_chips, get_blind_amount
from balatro_sim.game import BalatroGame, State
from balatro_sim.jokers.base import JokerInstance

SEED = "7I4M53DL"


def _win_current_blind(g):
    assert g.state == State.BLIND_SELECT
    g.step({"type": "play_blind"})
    g.debug_win_blind()
    g.step({})
    return g


def _script(g, n_steps=150):
    sigs = []
    for _ in range(n_steps):
        s = g.state
        if s == State.GAME_OVER:
            break
        if s == State.BLIND_SELECT:
            g.step({"type": "play_blind"})
        elif s == State.SELECTING_HAND:
            if g._hands_played_round == 0:      # score one real hand per blind, then clear it
                g.step({"type": "play", "cards": list(range(min(5, len(g.hand))))})
            else:
                g.debug_win_blind()
        elif s == State.ROUND_EVAL:
            g.step({})
        elif s == State.SHOP:
            g.step({"type": "leave_shop"})
        elif s == State.BOOSTER_OPEN:
            g.step({"type": "skip_booster"})
        sigs.append(g.state_signature())
    return sigs


class TestCatalogue:
    def test_eight_stakes(self):
        assert stakes.STAKE_KEYS == ["stake_white", "stake_red", "stake_green", "stake_black",
                                     "stake_blue", "stake_purple", "stake_orange", "stake_gold"]
        assert [stakes.STAKES[n].level for n in range(1, 9)] == list(range(1, 9))
        assert stakes.stake_spec("stake_gold") is stakes.STAKES[8]
        assert stakes.stake_spec(3).key == "stake_green"

    def test_modifier_table_is_cumulative(self):
        """game.lua:2050-2057 — every `stake >= N` line."""
        rows = {n: stakes.STAKES[n] for n in range(1, 9)}
        assert [rows[n].no_small_blind_reward for n in range(1, 9)] == [False] + [True] * 7
        assert [rows[n].scaling for n in range(1, 9)] == [1, 1, 2, 2, 2, 3, 3, 3]
        assert [rows[n].eternals_in_shop for n in range(1, 9)] == [False] * 3 + [True] * 5
        assert [rows[n].discards for n in range(1, 9)] == [0] * 4 + [-1] * 4
        assert [rows[n].perishables_in_shop for n in range(1, 9)] == [False] * 6 + [True] * 2
        assert [rows[n].rentals_in_shop for n in range(1, 9)] == [False] * 7 + [True]
        assert [rows[n].needs_stickers for n in range(1, 9)] == [False] * 3 + [True] * 5

    def test_unknown_stake_rejected(self):
        with pytest.raises(KeyError):
            BalatroGame(seed=SEED, stake=9)
        with pytest.raises(KeyError):
            BalatroGame(seed=SEED, stake="stake_pink")

    @pytest.mark.parametrize("n", range(1, 9))
    def test_every_stake_constructs(self, n):
        g = BalatroGame(seed=SEED, stake=n)
        assert g.stake == n and g.stake_key == stakes.STAKES[n].key
        assert g.run_state.stake == n
        _win_current_blind(g)
        assert g.state in (State.SHOP, State.BOOSTER_OPEN)


class TestWhiteIsVanilla:
    def test_default_stake_is_white(self):
        g = BalatroGame(seed=SEED)
        assert (g.stake, g.stake_key) == (1, "stake_white")
        assert not g.no_small_blind_reward and g.blind_scaling == 1
        assert not g.run_state.enable_eternals_in_shop
        assert not g.run_state.enable_perishables_in_shop
        assert not g.run_state.enable_rentals_in_shop

    def test_white_trajectory_identical_to_default(self):
        a = _script(BalatroGame(seed=SEED))
        b = _script(BalatroGame(seed=SEED, stake=1))
        c = _script(BalatroGame(seed=SEED, stake="stake_white"))
        assert a == b == c
        assert len(a) > 20

    def test_white_starting_params(self):
        g = BalatroGame(seed=SEED, stake=1)
        assert (g.base_hands, g.base_discards, g.dollars, g.current_blind.chips_target) == (4, 4, 4, 300)


class TestRedStake:
    def test_small_blind_pays_nothing(self):
        """blind.lua:84: no_blind_reward.Small -> blind.dollars = 0.  Win the Small Blind with
        all 4 hands unused and $4: White = 4 + $3 + $4 = 11; Red stake = 4 + $0 + $4 = 8."""
        white = _win_current_blind(BalatroGame(seed=SEED, stake=1))
        red = _win_current_blind(BalatroGame(seed=SEED, stake=2))
        assert white.dollars == 11
        assert red.dollars == 8

    def test_big_and_boss_still_pay(self):
        g = BalatroGame(seed=SEED, stake=2)
        _win_current_blind(g)
        while g.state == State.BOOSTER_OPEN:
            g.step({"type": "skip_booster"})
        g.step({"type": "leave_shop"})
        assert g.current_blind.kind == "Big"
        before = g.dollars
        _win_current_blind(g)
        assert g.dollars == before + 4 + 4 + min(before // 5, 5)   # $4 Big + 4 unused hands + interest

    def test_generation_identical_to_white(self):
        """Red stake changes no RNG call (no generation flag below stake 4)."""
        w = BalatroGame(seed=SEED, stake=1)
        r = BalatroGame(seed=SEED, stake=2)
        assert (w.boss_blind, w.blind_tags, w.run_state.current_round_voucher) == \
               (r.boss_blind, r.blind_tags, r.run_state.current_round_voucher)
        for g in (w, r):
            _win_current_blind(g)
        assert [(i.kind, i.key) for i in w.current_shop] == [(i.kind, i.key) for i in r.current_shop]


class TestScalingStakes:
    def test_scaling_tables(self):
        # misc_functions.lua:933-954
        assert [get_blind_amount(a, 2) for a in range(1, 9)] == [300, 900, 2600, 8000, 20000, 36000, 60000, 100000]
        assert [get_blind_amount(a, 3) for a in range(1, 9)] == [300, 1000, 3200, 9000, 25000, 60000, 110000, 200000]
        assert get_blind_amount(0, 2) == 100 and get_blind_amount(0, 3) == 100
        # endless (misc_functions.lua:928-930): floor(a * (b + (k*c)^d)^c), then drop everything
        # below the second significant digit.  ante 9: c = 1, d = 1.2 -> (0.75)^1.2 = 0.70813.. ; 1.6 + that = 2.30813.. ; then
        # floor(a * 2.30813) and drop everything below the second significant digit
        assert get_blind_amount(9, 1) == 110000      # floor(50000 * 2.30813) = 115406 -> 110000
        assert get_blind_amount(9, 2) == 230000      # floor(100000 * 2.30813) = 230813 -> 230000
        assert get_blind_amount(9, 3) == 460000      # floor(200000 * 2.30813) = 461626 -> 460000

    @pytest.mark.parametrize("stake,scaling", [(1, 1), (2, 1), (3, 2), (4, 2), (5, 2), (6, 3), (7, 3), (8, 3)])
    def test_engine_blind_scaling(self, stake, scaling):
        g = BalatroGame(seed=SEED, stake=stake)
        assert g.blind_scaling == scaling
        for ante in (1, 2, 5, 8, 10):
            for idx in (0, 1):
                g.ante = ante
                g.blind_idx = idx
                g._prepare_next_blind()
                assert g.current_blind.chips_target == int(get_blind_amount(ante, scaling) * (1.0, 1.5)[idx])

    def test_green_stake_ante_2_small_is_900(self):
        g = BalatroGame(seed=SEED, stake=3)
        g.ante = 2
        g.blind_idx = 0
        g._prepare_next_blind()
        assert g.current_blind.chips_target == 900
        assert blind_base_chips(2, 0, 2) == 900 and blind_base_chips(2, 0) == BLIND_CHIPS[2][0] == 800

    def test_purple_stake_ante_3_big_is_4800(self):
        g = BalatroGame(seed=SEED, stake=6)
        g.ante = 3
        g.blind_idx = 1
        g._prepare_next_blind()
        assert g.current_blind.chips_target == 4800

    def test_plasma_composes_with_scaling(self):
        g = BalatroGame(seed=SEED, stake=3, deck_key="b_plasma")
        g.ante = 2
        g.blind_idx = 0
        g._prepare_next_blind()
        assert g.current_blind.chips_target == 1800


class TestBlueStake:
    def test_one_fewer_discard(self):
        """starting_params.discards 3 - 1 (stake) + 1 (Red Deck) = 3."""
        g = BalatroGame(seed=SEED, stake=5)
        assert g.base_discards == 3
        g.step({"type": "play_blind"})
        assert g.discards_left == 3
        assert BalatroGame(seed=SEED, stake=5, deck_key="b_blue").base_discards == 2


class TestStickerStakes:
    """Black / Orange / Gold: the generation-side flags are set (generate rolls the
    sticker streams), the shelf carries the flags, but the engine has no sticker EFFECTS
    yet (later phase)."""

    @pytest.mark.parametrize("stake,flags", [
        (4, (True, False, False)), (7, (True, True, False)), (8, (True, True, True))])
    def test_generation_flags(self, stake, flags):
        rs = BalatroGame(seed=SEED, stake=stake).run_state
        assert (rs.enable_eternals_in_shop, rs.enable_perishables_in_shop, rs.enable_rentals_in_shop) == flags

    @pytest.mark.xfail(reason="sticker system (Eternal: cannot be sold) not implemented — later phase", strict=True)
    def test_eternal_joker_cannot_be_sold(self):
        from balatro_sim.shop import sell_joker
        g = BalatroGame(seed=SEED, stake=4)
        j = g.debug_add_joker("j_joker")
        j.state["eternal"] = True
        _win_current_blind(g)
        while g.state == State.BOOSTER_OPEN:
            g.step({"type": "skip_booster"})
        sell_joker(g, 0)
        assert len(g.jokers) == 1

    @pytest.mark.xfail(reason="sticker system (Rental: -$3 at end of round) not implemented — later phase", strict=True)
    def test_rental_costs_three_per_round(self):
        g = BalatroGame(seed=SEED, stake=8)
        j = g.debug_add_joker("j_joker")
        j.state["rental"] = True
        g.dollars = 20
        _win_current_blind(g)
        # $0 Small (stake >= 2) + 4 unused hands + $4 interest - $3 rental
        assert g.dollars == 20 + 0 + 4 + 4 - 3

    @pytest.mark.xfail(reason="sticker system (Perishable: debuffed after 5 rounds) not implemented — later phase", strict=True)
    def test_perishable_debuffs_after_five_rounds(self):
        g = BalatroGame(seed=SEED, stake=7)
        j = g.debug_add_joker("j_joker")
        j.state["perishable"] = True
        for _ in range(5):
            _win_current_blind(g)
            while g.state == State.BOOSTER_OPEN:
                g.step({"type": "skip_booster"})
            g.step({"type": "leave_shop"})
        assert j.state.get("debuffed") or getattr(j, "debuffed", False)


class TestClone:
    @pytest.mark.parametrize("stake", [2, 3, 5, 6])
    def test_clone_copies_stake_modifiers(self, stake):
        g = BalatroGame(seed=SEED, stake=stake)
        c = g.clone()
        assert (c.stake, c.stake_key, c.no_small_blind_reward, c.blind_scaling, c.base_discards) == \
               (g.stake, g.stake_key, g.no_small_blind_reward, g.blind_scaling, g.base_discards)
        assert c.run_state.stake == g.run_state.stake
