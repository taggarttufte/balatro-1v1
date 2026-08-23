"""Unit tests for urgency.py."""
from __future__ import annotations

import _bootstrap  # noqa: F401
from _bootstrap import BalatroGame, State

import urgency as urg


def _fresh_shop_game(seed="11111111"):
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
    assert g.state == State.SHOP
    return g


def test_next_blind_info_small_to_big():
    g = _fresh_shop_game()
    g.ante, g.blind_idx = 3, 0
    assert urg.next_blind_info(g) == (3, 1)


def test_next_blind_info_big_to_boss():
    g = _fresh_shop_game()
    g.ante, g.blind_idx = 3, 1
    assert urg.next_blind_info(g) == (3, 2)


def test_next_blind_info_boss_to_next_antes_small():
    g = _fresh_shop_game()
    # ante already advanced by the engine at boss defeat (game.py:1966-1973); blind_idx
    # is still 2 until leave_shop.
    g.ante, g.blind_idx = 4, 2
    assert urg.next_blind_info(g) == (4, 0)


def test_is_next_blind_pvp_respects_pvp_start_round():
    g = _fresh_shop_game()
    g.mlb = True
    g.pvp_start_round = 2
    assert urg.is_next_blind_pvp(g, ante=1, blind_idx=2) is False   # ante 1 boss: not yet PvP
    assert urg.is_next_blind_pvp(g, ante=2, blind_idx=2) is True
    assert urg.is_next_blind_pvp(g, ante=2, blind_idx=1) is False   # Big, not Boss


def test_compute_urgency_in_unit_interval():
    g = _fresh_shop_game()
    for lives in (4, 2, 1):
        g.lives = lives
        r = urg.compute(g)
        assert 0.0 <= r.urgency <= 1.0
        assert 0.0 <= r.shortfall <= 1.0
        assert 0.0 <= r.life_pressure <= 1.0


def test_compute_urgency_rises_as_lives_fall():
    g = _fresh_shop_game()
    g.mlb = True
    g.lives = 4
    r_full = urg.compute(g)
    g.lives = 1
    r_low = urg.compute(g)
    assert r_low.urgency >= r_full.urgency
    assert r_low.life_pressure > r_full.life_pressure


def test_compute_urgency_side_effect_free():
    g = _fresh_shop_game()
    sig_before = g.state_signature()
    urg.compute(g)
    assert g.state_signature() == sig_before


def test_next_blind_chip_target_matches_blind_base_chips():
    from balatro_sim.constants import blind_base_chips
    g = _fresh_shop_game()
    g.ante, g.blind_idx = 2, 0
    expected = blind_base_chips(2, 1, g.blind_scaling) * g.ante_scaling
    assert urg.next_blind_chip_target(g) == expected
