"""
test_decks.py — Phase 2 W3: the 15 decks (Red / Checkered / Plasma required; the rest
where the engine already had the machinery).  Ground truth is back.lua (Back:apply_to_run
173-288, Back:trigger_effect 109-168), game.lua:627-641 (deck table), blind.lua:107
(target = get_blind_amount * mult * ante_scaling) and state_events.lua:946-948 (the
final_scoring_step hook).  Expected numbers are hand-computed from the Lua, never from the
implementation.  See engine/DECKS_NOTES.md.
"""
import math
from collections import Counter

import pytest

from balatro_sim import decks
from balatro_sim.card import Card
from balatro_sim.card_selection import HypotheticalScorer
from balatro_sim.constants import BLIND_CHIPS, blind_base_chips, get_blind_amount
from balatro_sim.game import BalatroGame, State
from balatro_sim.game_keys import core as _core
from balatro_sim.hand_eval import evaluate_hand
from balatro_sim.jokers.base import JokerInstance
from balatro_sim.scoring import score_hand
from balatro_sim.shop import SHELF_KINDS

PseudoRandom = _core.PseudoRandom
SEED = "7I4M53DL"


# ── helpers ──────────────────────────────────────────────────────────────────────────

def _win_current_blind(g):
    """BLIND_SELECT -> play -> clear without scoring -> ROUND_EVAL step -> SHOP."""
    assert g.state == State.BLIND_SELECT
    g.step({"type": "play_blind"})
    g.debug_win_blind()
    assert g.state == State.ROUND_EVAL
    g.step({})
    return g


def _close_shop(g):
    while g.state == State.BOOSTER_OPEN:
        g.step({"type": "skip_booster"})
    assert g.state == State.SHOP
    g.step({"type": "leave_shop"})


def _script(g, n_steps=150):
    """A fixed policy: play the first 5 cards, never skip, never buy.  Returns the list of
    state_signature() after every step (the trajectory)."""
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


def _plain_score(cards, hand_type, jokers=(), plasma=False, seed="TEST1", held=None):
    s, ctx = score_hand(
        scoring_cards=list(cards), all_cards=list(cards), hand_type=hand_type,
        jokers=list(jokers), planet_levels={}, hands_left=3, discards_left=3, dollars=4,
        ante=1, deck_remaining=40, rng=PseudoRandom(seed), held_cards=held, plasma=plasma)
    return s, ctx


# ── catalogue ────────────────────────────────────────────────────────────────────────

class TestCatalogue:
    def test_fifteen_decks_in_game_order(self):
        assert decks.DECK_KEYS == [
            "b_red", "b_blue", "b_yellow", "b_green", "b_black", "b_magic", "b_nebula",
            "b_ghost", "b_abandoned", "b_checkered", "b_zodiac", "b_painted", "b_anaglyph",
            "b_plasma", "b_erratic"]
        assert [decks.DECKS[k].order for k in decks.DECK_KEYS] == list(range(1, 16))
        assert set(decks.DECK_STATUS) == set(decks.DECK_KEYS)

    def test_specs_match_game_lua_table(self):
        d = decks.DECKS
        assert d["b_red"].discards == 1
        assert d["b_blue"].hands == 1
        assert d["b_yellow"].dollars == 10
        assert (d["b_green"].money_per_hand, d["b_green"].money_per_discard, d["b_green"].no_interest) == (2, 1, True)
        assert (d["b_black"].hands, d["b_black"].joker_slot) == (-1, 1)
        assert (d["b_magic"].voucher, d["b_magic"].consumables) == ("v_crystal_ball", ("c_fool", "c_fool"))
        assert (d["b_nebula"].voucher, d["b_nebula"].consumable_slot) == ("v_telescope", -1)
        assert (d["b_ghost"].spectral_rate, d["b_ghost"].consumables) == (2, ("c_hex",))
        assert d["b_abandoned"].remove_faces
        assert d["b_checkered"].checkered
        assert d["b_zodiac"].vouchers == ("v_tarot_merchant", "v_planet_merchant", "v_overstock_norm")
        assert (d["b_painted"].hand_size, d["b_painted"].joker_slot) == (2, -1)
        assert d["b_anaglyph"].anaglyph
        assert (d["b_plasma"].ante_scaling, d["b_plasma"].plasma) == (2, True)
        assert d["b_erratic"].randomize_rank_suit

    @pytest.mark.parametrize("key", decks.DECK_KEYS)
    def test_every_deck_constructs_and_reaches_the_first_shop(self, key):
        g = BalatroGame(seed=SEED, deck_key=key)
        assert g.deck_key == key
        assert g.state == State.BLIND_SELECT
        _win_current_blind(g)
        assert g.state in (State.SHOP, State.BOOSTER_OPEN)

    def test_unknown_deck_rejected(self):
        with pytest.raises(KeyError):
            BalatroGame(seed=SEED, deck_key="b_nope")

    def test_default_is_red_white(self):
        g = BalatroGame(seed=SEED)
        assert (g.deck_key, g.stake, g.stake_key) == ("b_red", 1, "stake_white")


# ── Red (baseline) ───────────────────────────────────────────────────────────────────

class TestRed:
    def test_red_has_four_discards(self):
        """get_starting_params().discards = 3 (misc_functions.lua:1873) + Red's config
        {discards = 1} (game.lua:627, back.lua:213-215) = 4 per round.  The engine had 3."""
        g = BalatroGame(seed=SEED)
        assert g.base_discards == 4
        assert g.base_hands == 4
        assert g.dollars == 4
        g.step({"type": "play_blind"})
        assert (g.hands_left, g.discards_left, g.hand_size) == (4, 4, 8)

    def test_red_vanilla_targets(self):
        g = BalatroGame(seed=SEED)
        assert g.current_blind.chips_target == 300
        assert g.ante_scaling == 1 and g.blind_scaling == 1 and not g.plasma and not g.anaglyph
        assert g.money_per_hand == 1 and g.money_per_discard == 0 and not g.no_interest

    def test_explicit_red_white_equals_default(self):
        a = _script(BalatroGame(seed=SEED))
        b = _script(BalatroGame(seed=SEED, deck_key="b_red", stake=1))
        assert a == b and len(a) > 20

    def test_reset_reapplies_deck(self):
        g = BalatroGame(seed=SEED, deck_key="b_blue")
        g.reset()
        assert g.base_hands == 5


# ── Checkered ────────────────────────────────────────────────────────────────────────

class TestCheckered:
    RANK_ORDER = ["2", "3", "4", "5", "6", "7", "8", "9", "A", "J", "K", "Q", "T"]   # `suit..rank` string sort
    RANK_OF = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}

    def test_26_spades_26_hearts(self):
        g = BalatroGame(seed=SEED, deck_key="b_checkered")
        suits = Counter(c.suit for c in g.full_deck)
        assert suits == {"Spades": 26, "Hearts": 26}
        # two of every rank per suit
        assert Counter((c.suit, c.rank) for c in g.full_deck) == {
            (s, r): 2 for s in ("Spades", "Hearts") for r in range(2, 15)}

    def test_creation_order_is_pre_swap_sort_ids(self):
        """game.lua:2330-2378 creates C2..CT, D2..DT, H2..HT, S2..ST; back.lua:244-256 then
        swaps Clubs->Spades and Diamonds->Hearts IN PLACE (sort_ids unchanged)."""
        g = BalatroGame(seed=SEED, deck_key="b_checkered")
        ids = [c.id for c in g.full_deck]
        assert ids == sorted(ids)                       # full_deck is creation order
        ranks = [self.RANK_OF[r] for r in self.RANK_ORDER]
        expect = [("Spades", r) for r in ranks] + [("Hearts", r) for r in ranks] \
            + [("Hearts", r) for r in ranks] + [("Spades", r) for r in ranks]
        assert [(c.suit, c.rank) for c in g.full_deck] == expect

    def test_first_hand_is_red_first_hand_with_suits_swapped(self):
        """Since the swap keeps sort_ids, every shuffle ('shuffle', 'nr1', ...) lands the
        same sort_id in the same position as the Red deck — so the dealt hand is Red's hand
        with C->S, D->H.  (Independent check of decks.creation_order.)"""
        swap = {"Clubs": "Spades", "Diamonds": "Hearts", "Hearts": "Hearts", "Spades": "Spades"}
        red = BalatroGame(seed=SEED, deck_key="b_red")
        chk = BalatroGame(seed=SEED, deck_key="b_checkered")
        for g in (red, chk):
            g.step({"type": "play_blind"})
        assert [(c.rank, swap[c.suit]) for c in red.hand] == [(c.rank, c.suit) for c in chk.hand]
        assert [(c.rank, swap[c.suit]) for c in red.deck] == [(c.rank, c.suit) for c in chk.deck]

    def test_generation_identical_to_red(self):
        """Checkered changes no RNG call: boss, voucher, tags and the first shop are Red's."""
        red = BalatroGame(seed=SEED, deck_key="b_red")
        chk = BalatroGame(seed=SEED, deck_key="b_checkered")
        assert (red.boss_blind, red.blind_tags, red.run_state.current_round_voucher) == \
               (chk.boss_blind, chk.blind_tags, chk.run_state.current_round_voucher)
        for g in (red, chk):
            _win_current_blind(g)
        assert [(i.kind, i.key, i.edition) for i in red.current_shop] == \
               [(i.kind, i.key, i.edition) for i in chk.current_shop]

    def test_starting_params_are_vanilla(self):
        g = BalatroGame(seed=SEED, deck_key="b_checkered")
        assert (g.base_hands, g.base_discards, g.dollars, g.joker_slots, g.consumable_slots) == (4, 3, 4, 5, 2)

    def test_creation_order_helper_rejects_non_checkered_multiset(self):
        with pytest.raises(ValueError):
            decks.creation_order("b_checkered", ["S_2"] * 52)


# ── Plasma ───────────────────────────────────────────────────────────────────────────

class TestPlasmaBlinds:
    @pytest.mark.parametrize("ante", list(range(0, 13)))
    @pytest.mark.parametrize("idx", [0, 1, 2])
    def test_targets_doubled_at_every_ante(self, ante, idx):
        """blind.lua:107 `get_blind_amount(ante) * mult * starting_params.ante_scaling`."""
        g = BalatroGame(seed=SEED, deck_key="b_plasma")
        g.ante = ante
        g.blind_idx = idx
        g._prepare_next_blind()
        boss_mult = {"bl_wall": 2.0, "bl_needle": 0.5, "bl_final_vessel": 3.0}.get(g.current_blind.boss_key, 1.0)
        expect = int(blind_base_chips(ante, idx) * 2 * boss_mult)
        assert g.current_blind.chips_target == expect
        if ante <= 8:
            assert blind_base_chips(ante, idx) == BLIND_CHIPS[ante][idx]

    def test_ante_1_values(self):
        g = BalatroGame(seed=SEED, deck_key="b_plasma")
        assert g.current_blind.chips_target == 600
        g.blind_idx = 1; g._prepare_next_blind()
        assert g.current_blind.chips_target == 900
        g.blind_idx = 2; g._prepare_next_blind()
        assert g.current_blind.chips_target == int(1200 * {"bl_wall": 2.0, "bl_needle": 0.5}.get(g.current_blind.boss_key, 1.0))

    def test_endless_formula_doubles_too(self):
        assert get_blind_amount(9) == 110000          # floor(50000 * (1.6 + 0.75^1.2)) = 115406 -> 110000
        g = BalatroGame(seed=SEED, deck_key="b_plasma")
        g.ante = 9; g.blind_idx = 0; g._prepare_next_blind()
        assert g.current_blind.chips_target == 220000

    def test_generation_identical_to_red(self):
        red = BalatroGame(seed=SEED, deck_key="b_red")
        pl = BalatroGame(seed=SEED, deck_key="b_plasma")
        assert (red.boss_blind, red.blind_tags) == (pl.boss_blind, pl.blind_tags)
        for g in (red, pl):
            _win_current_blind(g)
        assert [(i.kind, i.key) for i in red.current_shop] == [(i.kind, i.key) for i in pl.current_shop]
        assert [(c.rank, c.suit) for c in red.hand] == [(c.rank, c.suit) for c in pl.hand]


class TestPlasmaScoring:
    """back.lua:121-128 at state_events.lua:946: tot = chips + mult; chips = floor(tot/2);
    mult = floor(tot/2); then score = floor(chips * mult).  Runs AFTER every joker."""

    def test_high_card_ace(self):
        # High Card lvl 1: 5 chips, 1 mult; Ace = 11 chips -> chips 16, mult 1
        # tot 17 -> floor(8.5) = 8 -> 8 * 8 = 64   (vanilla 16)
        ace = Card(rank=14, suit="Spades")
        assert _plain_score([ace], "High Card")[0] == 16
        s, ctx = _plain_score([ace], "High Card", plasma=True)
        assert s == 64
        assert ctx.running_mult == 8

    def test_pair_of_kings(self):
        # Pair lvl 1: 10 chips, 2 mult; K+K = 20 -> chips 30, mult 2; tot 32 -> 16 -> 256 (vanilla 60)
        ks = [Card(rank=13, suit="Hearts"), Card(rank=13, suit="Spades")]
        assert _plain_score(ks, "Pair")[0] == 60
        assert _plain_score(ks, "Pair", plasma=True)[0] == 256

    def test_fractional_total_floors_both_halves(self):
        # Pair of 5s, first 5 Polychrome: chips 10+5+5 = 20; mult 2 -> x1.5 after card 1 = 3.0
        # tot 23 -> floor(11.5) = 11 -> 121   (vanilla 20*3 = 60)
        cards = [Card(rank=5, suit="Clubs", edition="Polychrome"), Card(rank=5, suit="Diamonds")]
        assert _plain_score(cards, "Pair")[0] == 60
        assert _plain_score(cards, "Pair", plasma=True)[0] == 121

    def test_after_jokers(self):
        # Pair of 5s + Joker (+4 Mult, joker_main): chips 20, mult 6 -> tot 26 -> 13 -> 169 (vanilla 120)
        cards = [Card(rank=5, suit="Clubs"), Card(rank=5, suit="Diamonds")]
        j = [JokerInstance("j_joker")]
        assert _plain_score(cards, "Pair", jokers=j)[0] == 120
        assert _plain_score(cards, "Pair", jokers=j, plasma=True)[0] == 169

    def test_after_joker_polychrome_edition(self):
        # Pair of 5s + Polychrome Joker: mult (2 + 4) * 1.5 = 9; chips 20 -> tot 29 -> 14 -> 196 (vanilla 180)
        cards = [Card(rank=5, suit="Clubs"), Card(rank=5, suit="Diamonds")]
        j = [JokerInstance("j_joker", "Polychrome")]
        assert _plain_score(cards, "Pair", jokers=j)[0] == 180
        assert _plain_score(cards, "Pair", jokers=j, plasma=True)[0] == 196

    def test_held_steel_is_before_balance(self):
        # High Card Ace (16 chips, 1 mult) with a Steel card held: mult 1.5 -> tot 17.5 -> 8 -> 64
        ace = Card(rank=14, suit="Spades")
        steel = Card(rank=2, suit="Clubs", enhancement="Steel")
        assert _plain_score([ace], "High Card", held=[steel])[0] == 24        # floor(16 * 1.5)
        assert _plain_score([ace], "High Card", held=[steel], plasma=True)[0] == 64

    def test_balance_is_a_square(self):
        for cards, ht in [([Card(rank=r, suit="Spades") for r in (2, 3, 4, 5, 6)], "Straight Flush"),
                          ([Card(rank=10, suit="Hearts")] * 1, "High Card")]:
            s, ctx = _plain_score(cards, ht, plasma=True)
            root = int(math.isqrt(s))
            assert root * root == s

    def test_through_step(self):
        g = BalatroGame(seed=SEED, deck_key="b_plasma")
        g.step({"type": "play_blind"})
        g.hand[0] = Card(rank=14, suit="Spades")
        ht, scoring = evaluate_hand([g.hand[0]])
        assert ht == "High Card"
        g.step({"type": "play", "cards": [0]})
        assert g.chips_scored == 64

    def test_hypothetical_scorer_uses_plasma(self):
        g = BalatroGame(seed=SEED, deck_key="b_plasma")
        g.step({"type": "play_blind"})
        ace = Card(rank=14, suit="Spades")
        g.hand[0] = ace
        assert HypotheticalScorer(g, model_held=False).score([ace], "High Card", [ace]) == 64
        g2 = BalatroGame(seed=SEED, deck_key="b_red")
        g2.step({"type": "play_blind"})
        g2.hand[0] = ace
        assert HypotheticalScorer(g2, model_held=False).score([ace], "High Card", [ace]) == 16

    def test_clone_keeps_plasma(self):
        g = BalatroGame(seed=SEED, deck_key="b_plasma")
        c = g.clone()
        assert c.plasma and c.ante_scaling == 2 and c.deck_key == "b_plasma"


# ── the other decks through step() ───────────────────────────────────────────────────

class TestTrivialDecks:
    def test_blue_five_hands(self):
        g = BalatroGame(seed=SEED, deck_key="b_blue")
        g.step({"type": "play_blind"})
        assert (g.hands_left, g.discards_left) == (5, 3)

    def test_yellow_extra_ten_dollars(self):
        assert BalatroGame(seed=SEED, deck_key="b_yellow").dollars == 14

    def test_black_one_more_slot_one_fewer_hand(self):
        g = BalatroGame(seed=SEED, deck_key="b_black")
        assert g.joker_slots == 6
        g.step({"type": "play_blind"})
        assert g.hands_left == 3

    def test_green_payout_and_no_interest(self):
        """state_events.lua:1166-1173 (money_per_hand 2 / money_per_discard 1), :1191 (no
        interest).  Win the Small Blind untouched with $20: Green = 20 + $3 blind + 4 hands*$2
        + 3 discards*$1 = 34; Red = 20 + 3 + 4*$1 + interest min(20//5, 5)=4 = 31."""
        green = BalatroGame(seed=SEED, deck_key="b_green")
        red = BalatroGame(seed=SEED, deck_key="b_red")
        for g in (green, red):
            g.dollars = 20
            _win_current_blind(g)
        assert green.dollars == 34
        assert red.dollars == 31

    def test_green_no_interest_with_no_hands_left(self):
        g = BalatroGame(seed=SEED, deck_key="b_green")
        g.dollars = 25
        g.step({"type": "play_blind"})
        g.hands_left = 0
        g.discards_left = 0
        g.debug_win_blind()
        g.step({})
        assert g.dollars == 25 + 3

    def test_magic_crystal_ball_and_two_fools(self):
        g = BalatroGame(seed=SEED, deck_key="b_magic")
        assert g.consumable_hand == ["c_fool", "c_fool"]
        assert g.consumable_slots == 3
        assert "v_crystal_ball" in g.vouchers and "v_crystal_ball" in g.run_state.used_vouchers
        assert "c_fool" in g.run_state.used_jokers

    def test_nebula_telescope_and_one_slot(self):
        g = BalatroGame(seed=SEED, deck_key="b_nebula")
        assert g.consumable_slots == 1
        assert "v_telescope" in g.vouchers and "v_telescope" in g.run_state.used_vouchers

    def test_ghost_hex_and_spectral_rate(self):
        g = BalatroGame(seed=SEED, deck_key="b_ghost")
        assert g.consumable_hand == ["c_hex"]
        assert g.run_state.spectral_rate == 2

    def test_abandoned_no_faces(self):
        g = BalatroGame(seed=SEED, deck_key="b_abandoned")
        assert len(g.full_deck) == 40
        assert not any(c.rank in (11, 12, 13) for c in g.full_deck)
        assert Counter(c.suit for c in g.full_deck) == {s: 10 for s in ("Spades", "Hearts", "Clubs", "Diamonds")}
        ids = [c.id for c in g.full_deck]
        assert ids == sorted(ids)

    def test_zodiac_vouchers_and_three_shelf_slots(self):
        g = BalatroGame(seed=SEED, deck_key="b_zodiac")
        assert {"v_tarot_merchant", "v_planet_merchant", "v_overstock_norm"} <= g.vouchers
        assert g.run_state.shop_joker_max == 3
        _win_current_blind(g)
        shelf = [i for i in g.current_shop if i.kind in SHELF_KINDS]
        assert len(shelf) == 3

    def test_painted_hand_size_and_slots(self):
        g = BalatroGame(seed=SEED, deck_key="b_painted")
        assert g.joker_slots == 4
        g.step({"type": "play_blind"})
        assert g.hand_size == 10 and len(g.hand) == 10

    def test_anaglyph_double_tag_after_boss_only(self):
        g = BalatroGame(seed=SEED, deck_key="b_anaglyph")
        _win_current_blind(g)                     # Small
        assert "tag_double" not in g.tag_state.keys()
        _close_shop(g)
        _win_current_blind(g)                     # Big
        assert "tag_double" not in g.tag_state.keys()
        _close_shop(g)
        assert g.current_blind.kind == "Boss"
        _win_current_blind(g)                     # Boss -> eval -> Double Tag
        assert g.tag_state.keys().count("tag_double") == 1
        assert g.ante == 2

    def test_anaglyph_not_on_red(self):
        g = BalatroGame(seed=SEED, deck_key="b_red")
        for _ in range(3):                         # Small, Big, Boss of ante 1
            _win_current_blind(g)
            if g.ante == 1:
                _close_shop(g)
        assert g.ante == 2
        assert "tag_double" not in g.tag_state.keys()

    def test_erratic_is_randomised_and_deterministic(self):
        a = BalatroGame(seed=SEED, deck_key="b_erratic")
        b = BalatroGame(seed=SEED, deck_key="b_erratic")
        assert len(a.full_deck) == 52
        comp = Counter((c.suit, c.rank) for c in a.full_deck)
        assert comp == Counter((c.suit, c.rank) for c in b.full_deck)
        assert any(n != 1 for n in comp.values())          # not the standard deck
        ids = [c.id for c in a.full_deck]
        assert ids == sorted(ids)
        assert [(c.suit, c.rank) for c in a.full_deck] == sorted(
            [(c.suit, c.rank) for c in a.full_deck], key=lambda p: p[0][0] + {10: "T", 11: "J", 12: "Q", 13: "K", 14: "A"}.get(p[1], str(p[1])))


class TestGenerationUnchanged:
    """Checkered and Plasma change no generation call, so every shop / boss / tag the run
    produces must equal the Red deck's (the Red ground truth is 126/126 exact through
    ante 8; this is the oracle-by-proxy for the two decks the corpus does not cover).
    12 seeds through ante 4 were checked offline with 0 mismatches; 3 seeds here."""

    @staticmethod
    def _visits(g, max_ante=3):
        out = []
        for _ in range(400):
            s = g.state
            if s == State.GAME_OVER or g.ante > max_ante:
                break
            if s == State.BLIND_SELECT:
                g.step({"type": "play_blind"})
            elif s == State.SELECTING_HAND:
                if g._hands_played_round == 0:
                    g.step({"type": "play", "cards": list(range(min(5, len(g.hand))))})
                else:
                    g.debug_win_blind()
            elif s == State.ROUND_EVAL:
                g.step({})
            elif s == State.SHOP:
                out.append((g.ante, g.boss_blind, tuple(sorted(g.blind_tags.items())),
                            tuple((i.kind, i.key, i.edition) for i in g.current_shop)))
                g.step({"type": "leave_shop"})
            elif s == State.BOOSTER_OPEN:
                g.step({"type": "skip_booster"})
        return out

    @pytest.mark.parametrize("seed", ["7I4M53DL", "11111111", "1558AXDL"])
    @pytest.mark.parametrize("deck", ["b_checkered", "b_plasma"])
    def test_shops_bosses_tags_equal_red(self, seed, deck):
        red = self._visits(BalatroGame(seed=seed, deck_key="b_red"))
        other = self._visits(BalatroGame(seed=seed, deck_key=deck))
        assert len(red) >= 6
        assert other == red


class TestCloneAndSignature:
    @pytest.mark.parametrize("key", ["b_green", "b_plasma", "b_anaglyph", "b_painted"])
    def test_clone_copies_deck_modifiers(self, key):
        g = BalatroGame(seed=SEED, deck_key=key)
        c = g.clone()
        for attr in ("deck_key", "stake", "stake_key", "no_small_blind_reward", "blind_scaling",
                     "ante_scaling", "no_interest", "money_per_hand", "money_per_discard", "plasma", "anaglyph"):
            assert getattr(c, attr) == getattr(g, attr), attr
        assert c.state_signature() == g.state_signature()

    def test_different_decks_differ_in_signature(self):
        assert BalatroGame(seed=SEED, deck_key="b_red").state_signature() != \
               BalatroGame(seed=SEED, deck_key="b_blue").state_signature()
