"""
test_joker_catalogue.py — the shop catalogue and the effect registry must agree.

Regression tests for audit findings M1, M2 and M3, plus a guard so the whole
class of bug cannot recur silently.

Before 2026-07-29:
  M1  j_supernova was buyable with no implementation at all — a literal no-op
      purchase.
  M2  Ten jokers had live implementations but no catalogue entry, so they could
      never appear in a shop. j_juggler (+1 hand size) and j_drunkard (+1
      discard) even had passive handling wired up in game._start_blind that could
      never run. The audit originally spotted three of the ten; comparing the
      registry against the catalogue by implementation identity found the rest.
  M3  Four suit jokers had TWO implementations each — a per-hand version under
      the buyable *_mult keys and a per-card version under dead j_greedy /
      j_greedy_joker keys. The audit warned the dead copy might be the correct
      one, and it was: the real Greedy Joker gives +3 Mult per scoring Diamond,
      not +3 once per hand.
"""
import pytest

from balatro_sim.game_keys import core as _core
from balatro_sim.card import Card
from balatro_sim.game import BalatroGame
from balatro_sim.jokers.base import JOKER_REGISTRY, JokerInstance
from balatro_sim.scoring import score_hand
from balatro_sim.shop import JOKER_CATALOGUE
import balatro_sim.jokers  # noqa: F401  (populates the registry)
PseudoRandom = _core.PseudoRandom


def _score(cards, joker_keys=(), hand_type="High Card"):
    s, _ = score_hand(
        scoring_cards=list(cards), all_cards=list(cards), hand_type=hand_type,
        jokers=[JokerInstance(k) for k in joker_keys],
        planet_levels={hand_type: 1}, hands_left=3, discards_left=3, dollars=10,
        ante=1, deck_remaining=44, rng=PseudoRandom("TEST1"))
    return s


class TestCatalogueRegistryConsistency:
    def test_every_buyable_joker_has_an_implementation(self):
        """M1 guard. A buyable joker with no registry entry is a no-op purchase."""
        missing = sorted(set(JOKER_CATALOGUE) - set(JOKER_REGISTRY))
        assert missing == [], f"buyable but no effect: {missing}"

    def test_no_implementation_is_unreachable(self):
        """
        M2/M3 guard. Every distinct implementation must be reachable from some
        catalogue key. A registry entry whose implementation object no buyable key
        points at is dead code — and as M3 showed, the dead copy is sometimes the
        correct one.
        """
        buyable_impls = {id(JOKER_REGISTRY[k])
                         for k in JOKER_CATALOGUE if k in JOKER_REGISTRY}
        unreachable = sorted(
            k for k in set(JOKER_REGISTRY) - set(JOKER_CATALOGUE)
            if id(JOKER_REGISTRY[k]) not in buyable_impls)
        # Phase 1 W1 (2026-08-21) resolved the four known dead aliases
        # (j_lucky_joker removed — not a real joker; j_ring_master / j_space /
        # j_ticket are the GAME keys and now carry the single implementation).
        # The registry is keyed 1:1 against mp/rng/pools.py, so nothing may be
        # unreachable any more. See mp/engine/REKEY_NOTES.md.
        assert unreachable == [], (
            f"unexpected unreachable implementations: {unreachable}")

    def test_catalogue_entries_are_well_formed(self):
        for key, info in JOKER_CATALOGUE.items():
            assert info["key"] == key
            assert info["name"], key
            assert info["rarity"] in ("Common", "Uncommon", "Rare", "Legendary"), key
            assert info["price"] > 0, key


class TestNewlyBuyableJokers:
    @pytest.mark.parametrize("key", [
        "j_juggler", "j_drunkard", "j_stone", "j_ticket",
        "j_smiley", "j_fortune_teller", "j_hallucination", "j_mail",
        "j_golden", "j_idol", "j_to_the_moon", "j_bootstraps",
    ])
    def test_is_now_in_the_catalogue(self, key):
        assert key in JOKER_CATALOGUE
        assert key in JOKER_REGISTRY

    def test_juggler_passive_now_reachable(self):
        """game._start_blind grants +1 hand size, which could never fire before."""
        g = BalatroGame(seed=3)
        g.jokers.append(JokerInstance("j_juggler"))
        g.step({"type": "play_blind"})
        assert g.hand_size == 9      # base 8 + 1

    def test_drunkard_passive_now_reachable(self):
        g = BalatroGame(seed=3)
        g.jokers.append(JokerInstance("j_drunkard"))
        g.step({"type": "play_blind"})
        # Red Deck = starting_params.discards 3 + the deck's +1 (back.lua:213, W3 decks) = 4,
        # then Drunkard's +1.
        assert g.discards_left == 5


class TestSupernova:
    def test_adds_times_hand_type_played_this_run(self):
        g = BalatroGame(seed=3)
        g.jokers.append(JokerInstance("j_supernova"))
        g.step({"type": "play_blind"})
        g.hand[0] = Card(rank=7, suit="Spades")
        g.hand[1] = Card(rank=7, suit="Hearts")
        g._hand_type_counts = {"Pair": 4}    # already played 4 pairs this run
        g._play_hand([0, 1])
        # Pair 10 + 14 chips = 24; mult 2 + 5 (this is the 5th Pair) = 7 -> 168
        assert g.chips_scored == 168

    def test_contributes_nothing_on_a_fresh_run(self):
        cards = [Card(rank=7, suit="Spades")]
        assert _score(cards, ["j_supernova"]) == _score(cards)


class TestSuitJokersFirePerCard:
    def test_greedy_scales_with_diamond_count(self):
        """+3 per scoring Diamond, not +3 once. A 5-Diamond flush is +15."""
        flush = [Card(rank=r, suit="Diamonds") for r in (2, 4, 6, 8, 10)]
        base = _score(flush, hand_type="Flush")
        with_j = _score(flush, ["j_greedy_joker"], hand_type="Flush")
        # Flush base 35/4, chips 35 + 30 = 65; mult 4 vs 4 + 15 = 19
        assert base == 65 * 4
        assert with_j == 65 * 19

    def test_two_diamonds_give_six_mult(self):
        pair = [Card(rank=7, suit="Diamonds"), Card(rank=7, suit="Diamonds")]
        assert _score(pair, ["j_greedy_joker"], hand_type="Pair") == 24 * 8

    def test_off_suit_cards_contribute_nothing(self):
        pair = [Card(rank=7, suit="Spades"), Card(rank=7, suit="Hearts")]
        assert _score(pair, ["j_greedy_joker"], hand_type="Pair") == 24 * 2

    @pytest.mark.parametrize("key,suit", [
        ("j_greedy_joker", "Diamonds"), ("j_lusty_joker", "Hearts"),
        ("j_wrathful_joker", "Spades"), ("j_gluttenous_joker", "Clubs"),
    ])
    def test_each_suit_joker_matches_its_own_suit(self, key, suit):
        matching = [Card(rank=5, suit=suit), Card(rank=5, suit=suit)]
        assert _score(matching, [key], hand_type="Pair") == 20 * 8

    def test_debuffed_cards_do_not_count(self):
        pair = [Card(rank=7, suit="Diamonds", debuffed=True),
                Card(rank=7, suit="Diamonds")]
        # the debuffed card is skipped entirely by the scoring loop
        assert _score(pair, ["j_greedy_joker"], hand_type="Pair") == (10 + 7) * 5


class TestShopCanActuallyOfferTheNewJokers:
    def test_new_keys_appear_in_generated_shops(self):
        """Sanity check that catalogue additions are reachable through the shop."""
        # W2: shop cards come from generate.create_card over pools.JOKERS_BY_RARITY;
        # draw shop jokers on one RunState (Showman on, so nothing is blocked)
        from balatro_sim.game_keys import gen
        st = gen.RunState("1")
        st.showman = True
        st.deck_enhancements = {"m_stone", "m_steel", "m_lucky", "m_gold", "m_glass"}   # enhancement gates
        seen = set()
        for _ in range(6000):
            seen.add(gen.create_card(st, "Joker", area="shop", key_append="sho").key)
        for key in ("j_juggler", "j_drunkard", "j_stone", "j_golden"):
            assert key in seen, f"{key} never generated"
