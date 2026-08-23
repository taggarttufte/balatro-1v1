"""Unit tests for economy.py -- interest arithmetic and true cost."""
from __future__ import annotations

import _bootstrap  # noqa: F401
from _bootstrap import BalatroGame, State

import economy as econ


def _fresh_shop_game(seed="11111111", dollars=None):
    g = BalatroGame(seed=seed, deck_key="b_red", stake=1, ruleset="mlb")
    steps = 0
    while g.state != State.SHOP and steps < 4000:
        acts = g.legal_actions()
        if not acts:
            break
        a = next((x for x in acts if x["type"] == "play_blind"), acts[0])
        g.step(a)
        steps += 1
        if g.state == State.SELECTING_HAND:
            acts2 = g.legal_actions()
            play = [x for x in acts2 if x["type"] == "play"]
            if play:
                g.step(play[0])
                steps += 1
    assert g.state == State.SHOP, "fixture failed to reach a shop state"
    if dollars is not None:
        g.dollars = dollars
    return g


def test_interest_now_formula():
    g = _fresh_shop_game(dollars=23)
    g.interest_cap = 5
    assert econ.interest_now(g) == 4          # floor(23/5) = 4, under the cap
    g.dollars = 30
    assert econ.interest_now(g) == 5          # floor(30/5) = 6, capped at 5


def test_interest_now_zero_when_no_interest():
    g = _fresh_shop_game(dollars=50)
    g.no_interest = True
    assert econ.interest_now(g) == 0


def test_interest_after_never_negative_dollars():
    g = _fresh_shop_game(dollars=3)
    assert econ.interest_after(g, 10) == 0     # dollars - spend clips at 0, not negative


def test_dollars_to_next_tier():
    g = _fresh_shop_game(dollars=21)
    g.interest_cap = 10
    assert econ.dollars_to_next_tier(g) == 4   # 21 -> 25 is the next $5 tier
    g.dollars = 25
    assert econ.dollars_to_next_tier(g) == 5   # exactly on a tier, cap not yet hit: next is +5 away
    g.dollars = 26
    g.interest_cap = 1
    assert econ.dollars_to_next_tier(g) == 0   # already at/above the cap: no further tier matters


def test_shops_remaining_counts_to_ante_8():
    g = _fresh_shop_game()
    g.ante, g.blind_idx = 1, 0     # just cleared Small of ante 1
    assert econ.shops_remaining(g) == 1 + 2 + 7    # this shop's payout + 2 more this ante + 7 antes x3
    g.ante, g.blind_idx = 8, 1     # just cleared Big of ante 8
    assert econ.shops_remaining(g) == 1 + 1 + 0
    g.ante, g.blind_idx = 8, 2     # just cleared the ante-8 boss (ante already advanced in-engine
                                    # to 9 at that point; blind_idx is still 2 until leave_shop)
    assert econ.shops_remaining(g) == 1 + 0 + 0


def test_shops_remaining_horizon_override():
    g = _fresh_shop_game()
    assert econ.shops_remaining(g, horizon_rounds=3) == 3
    assert econ.shops_remaining(g, horizon_rounds=0) == 0


def test_interest_loss_zero_when_no_tier_crossed():
    g = _fresh_shop_game(dollars=100)
    g.interest_cap = 5
    # already at the cap; a small spend does not lower interest at all
    assert econ.interest_loss(g, 3) == 0.0


def test_interest_loss_positive_and_bounded_by_flat_case():
    g = _fresh_shop_game(dollars=23)
    g.interest_cap = 5
    g.ante, g.blind_idx = 1, 0
    loss = econ.interest_loss(g, 5)            # 23 -> 18: interest 4 -> 3, one tier lost
    flat = 1.0 * econ.shops_remaining(g)        # decay=1.0 upper bound
    assert 0.0 < loss <= flat
    assert loss == econ.interest_loss(g, 5, decay=1.0) or loss < econ.interest_loss(g, 5, decay=1.0)


def test_interest_loss_decay_matches_geometric_series():
    g = _fresh_shop_game(dollars=23)
    g.interest_cap = 5
    g.ante, g.blind_idx = 1, 0
    per_round = econ.interest_now(g) - econ.interest_after(g, 5)
    h = econ.shops_remaining(g)
    decay = 0.85
    expected = per_round * decay * (1 - decay ** h) / (1 - decay)
    assert abs(econ.interest_loss(g, 5, decay=decay) - expected) < 1e-9


def test_true_cost_is_cost_plus_interest_loss():
    g = _fresh_shop_game(dollars=23)
    cost = 6.0
    assert econ.true_cost(g, cost) == cost + econ.interest_loss(g, cost)


def test_reroll_cost_now_free_when_free_rerolls_remaining():
    g = _fresh_shop_game()
    g.reroll_cost = 7
    g.free_rerolls_remaining = 1
    assert econ.reroll_cost_now(g) == 0
    g.free_rerolls_remaining = 0
    assert econ.reroll_cost_now(g) == max(0, 7 - g.reroll_discount)


def test_economy_snapshot_matches_the_individual_functions():
    g = _fresh_shop_game(dollars=17)
    snap = econ.EconomySnapshot.build(g)
    assert snap.dollars == g.dollars
    assert snap.interest_now == econ.interest_now(g)
    assert snap.shops_remaining == econ.shops_remaining(g)
    assert snap.reroll_cost_now == econ.reroll_cost_now(g)
