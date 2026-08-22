"""
test_clone.py - Correctness checks for BalatroGame.clone().

Verifies the clone is fully independent: mutating the clone must not affect
the original, and vice versa. Critical for MCTS — a leaky clone produces
silent tree corruption.
"""
import random

from balatro_sim.game import BalatroGame, State
from balatro_sim.jokers.base import JokerInstance
from balatro_sim.shop import ShopItem


def _populate_rich(game: BalatroGame) -> BalatroGame:
    """Force the game into a representative mid-run state."""
    game.reset()
    game.step({"type": "play_blind"})
    game.ante = 4
    game.dollars = 28
    game.jokers = [
        JokerInstance("j_joker"),
        JokerInstance("j_green_joker"),
        JokerInstance("j_steel_joker"),
    ]
    game.jokers[1].state = {"mult": 12, "sell_value": 3}
    game.consumable_hand = ["c_strength", "c_pl_pluto"]
    game.current_shop = [
        ShopItem(kind="joker", key="j_blueprint", name="Blueprint", price=10),
        ShopItem(kind="planet", key="c_pl_mars", name="Mars", price=3),
    ]
    game.played_hand_types_this_round = {"Pair"}
    game.planet_levels["Flush"] = 4
    game.vouchers = {"v_overstock_norm"}
    return game


def test_clone_returns_independent_object():
    g = _populate_rich(BalatroGame(seed=11))
    c = g.clone()
    assert c is not g
    assert c.deck is not g.deck
    assert c.hand is not g.hand
    assert c.jokers is not g.jokers
    assert c.consumable_hand is not g.consumable_hand
    assert c.current_shop is not g.current_shop
    assert c.played_hand_types_this_round is not g.played_hand_types_this_round
    assert c.planet_levels is not g.planet_levels
    assert c.vouchers is not g.vouchers
    assert not hasattr(c, "rng") and not hasattr(g, "rng")     # W3: no legacy single stream
    assert c.run_state is not g.run_state
    assert c.run_state.rng is not g.run_state.rng


def test_clone_preserves_scalar_state():
    g = _populate_rich(BalatroGame(seed=11))
    c = g.clone()
    assert c.ante == g.ante
    assert c.dollars == g.dollars
    assert c.state == g.state
    assert c.chips_scored == g.chips_scored
    assert c.hands_left == g.hands_left
    assert c.discards_left == g.discards_left
    assert c.current_blind.kind == g.current_blind.kind
    assert c.current_blind.chips_target == g.current_blind.chips_target


def test_clone_preserves_collections_by_value():
    g = _populate_rich(BalatroGame(seed=11))
    c = g.clone()
    assert len(c.deck) == len(g.deck)
    assert len(c.hand) == len(g.hand)
    assert [card.rank for card in c.hand] == [card.rank for card in g.hand]
    assert [card.suit for card in c.hand] == [card.suit for card in g.hand]
    assert [j.key for j in c.jokers] == [j.key for j in g.jokers]
    assert c.consumable_hand == g.consumable_hand
    assert c.played_hand_types_this_round == g.played_hand_types_this_round
    assert c.planet_levels == g.planet_levels
    assert c.vouchers == g.vouchers


def test_mutating_clone_does_not_affect_original():
    g = _populate_rich(BalatroGame(seed=11))
    g_dollars_before = g.dollars
    g_hand_len_before = len(g.hand)
    g_jokers_before = list(g.jokers)
    g_joker1_mult = g.jokers[1].state["mult"]

    c = g.clone()
    c.dollars = 999
    c.hand.pop()
    c.jokers.append(JokerInstance("j_blueprint"))
    c.jokers[1].state["mult"] = 999
    c.consumable_hand.append("c_test")
    c.played_hand_types_this_round.add("Flush")
    c.planet_levels["Flush"] = 99

    assert g.dollars == g_dollars_before
    assert len(g.hand) == g_hand_len_before
    assert g.jokers == g_jokers_before
    assert g.jokers[1].state["mult"] == g_joker1_mult
    assert "c_test" not in g.consumable_hand
    assert "Flush" not in g.played_hand_types_this_round
    assert g.planet_levels["Flush"] != 99


def test_mutating_original_does_not_affect_clone():
    g = _populate_rich(BalatroGame(seed=11))
    c = g.clone()

    g.dollars = 0
    g.hand.clear()
    g.jokers.clear()
    g.jokers = [JokerInstance("j_test")]
    g.consumable_hand.clear()
    g.planet_levels["Flush"] = 0

    assert c.dollars == 28
    assert len(c.hand) > 0
    assert len(c.jokers) == 3
    assert c.consumable_hand == ["c_strength", "c_pl_pluto"]
    assert c.planet_levels["Flush"] == 4


def test_rng_state_is_independent():
    g = _populate_rich(BalatroGame(seed=11))
    c = g.clone()

    # Both should produce identical first draw on the same keyed stream
    assert g.run_state.rng.pseudorandom("lucky_mult") == c.run_state.rng.pseudorandom("lucky_mult")

    # But after one diverges, the other is unaffected
    g_next_after_one = g.run_state.rng.pseudorandom("lucky_mult")
    c.run_state.rng.pseudorandom("lucky_mult")  # advance c past the same point
    c_next_after_one = c.run_state.rng.pseudorandom("lucky_mult")
    # g advanced once, c advanced twice — so c_next_after_one != g_next_after_one
    assert g_next_after_one != c_next_after_one


def test_clone_then_step_does_not_corrupt_original():
    """Smoke test: stepping the clone through a full hand should not touch the original."""
    g = _populate_rich(BalatroGame(seed=11))
    orig_hand_ranks = [c.rank for c in g.hand]
    orig_dollars = g.dollars
    orig_chips = g.chips_scored

    c = g.clone()
    if c.state == State.SELECTING_HAND and len(c.hand) >= 1:
        c.step({"type": "play", "cards": [0]})

    assert [card.rank for card in g.hand] == orig_hand_ranks
    assert g.dollars == orig_dollars
    assert g.chips_scored == orig_chips


def test_clone_chain_is_stable():
    """Cloning a clone of a clone should still match the original by value."""
    g = _populate_rich(BalatroGame(seed=11))
    c1 = g.clone()
    c2 = c1.clone()
    c3 = c2.clone()
    assert c3.dollars == g.dollars
    assert c3.ante == g.ante
    assert [card.rank for card in c3.hand] == [card.rank for card in g.hand]
    assert [j.state for j in c3.jokers] == [j.state for j in g.jokers]


def test_card_mutation_isolated():
    """Mutating a card in the clone's hand must not affect original card."""
    g = _populate_rich(BalatroGame(seed=11))
    if not g.hand:
        return
    c = g.clone()
    c.hand[0].debuffed = True
    c.hand[0].enhancement = "Steel"
    assert g.hand[0].debuffed is False
    assert g.hand[0].enhancement != "Steel"
