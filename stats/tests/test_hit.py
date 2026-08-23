"""Unit tests for hit.py: pool-fraction P(hit), the fast synergy-based proxy, the precise
joker dry run and the flat consumable/card/voucher tables. Side-effect freedom on the dry
run (the one path that touches ``game.run_state.rng``, via a clone) is pinned here too;
``test_decide.py`` pins it again for the whole ``decision_table`` call."""
from __future__ import annotations

import _bootstrap  # noqa: F401
from _bootstrap import BalatroGame, State

import hit as hitmod
from balatro_sim import game_keys as _gk


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


# ══════════════════════════════════════════════════════════ pool_dollar_value / is_hit

def test_legendary_jokers_score_high():
    # Legendaries are S-tier in synergy.estimate_joker_strength; even with zero synergy
    # match (empty loadout -> coherence 0.5) they must clear the default hit threshold.
    assert hitmod.is_hit("j_perkeo", [], ante=4)
    assert hitmod.is_hit("j_triboulet", [], ante=4)


def test_weak_jokers_score_low():
    assert not hitmod.is_hit("j_credit_card", [], ante=4)
    assert not hitmod.is_hit("j_egg", [], ante=4)


def test_coherence_raises_value_for_matching_synergy():
    # j_droll is a Flush joker; owning other Flush jokers should raise its value over an
    # empty-loadout baseline (coherence_score's dominant-direction overlap).
    baseline = hitmod.pool_dollar_value("j_droll", [], ante=3)
    with_flush_build = hitmod.pool_dollar_value("j_droll", ["j_crafty", "j_smeared"], ante=3)
    assert with_flush_build >= baseline


# ══════════════════════════════════════════════════════════ P(hit) pool mechanics

def test_shop_slot_distribution_sums_to_at_most_one():
    g = _fresh_shop_game()
    slices = hitmod.shop_slot_distribution(g)
    total_p = sum(s.p_component for s in slices)
    assert 0.0 <= total_p <= 1.0 + 1e-9


def test_reroll_p_hit_is_a_probability():
    g = _fresh_shop_game()
    p, mean_val, details = hitmod.reroll_p_hit(g)
    assert 0.0 <= p <= 1.0
    assert mean_val >= 0.0
    assert details["n_slots"] == g.run_state.shop_joker_max


def test_reroll_p_hit_side_effect_free():
    g = _fresh_shop_game()
    sig_before = g.state_signature()
    hitmod.reroll_p_hit(g)
    assert g.state_signature() == sig_before


def test_reroll_p_hit_monotone_in_threshold():
    """A LOWER hit threshold (easier bar) can only raise or hold P(hit), never lower it."""
    g = _fresh_shop_game()
    loose = hitmod.StatsConfig(pool_hit_threshold_dollars=2.0)
    strict = hitmod.StatsConfig(pool_hit_threshold_dollars=20.0)
    p_loose, _, _ = hitmod.reroll_p_hit(g, cfg=loose)
    p_strict, _, _ = hitmod.reroll_p_hit(g, cfg=strict)
    assert p_loose >= p_strict


def test_pack_p_hit_scales_with_size():
    g = _fresh_shop_game()
    p_small, _, _ = hitmod.pack_p_hit(g, "Buffoon", size=2)
    p_big, _, _ = hitmod.pack_p_hit(g, "Buffoon", size=5)
    assert p_big >= p_small


def test_pack_p_hit_standard_pack_has_no_model():
    g = _fresh_shop_game()
    p, val, details = hitmod.pack_p_hit(g, "Standard", size=5)
    assert (p, val, details) == (0.0, 0.0, {})


# ══════════════════════════════════════════════════════════ precise joker dry run

def test_joker_hit_value_side_effect_free():
    g = _fresh_shop_game()
    sig_before = g.state_signature()
    v = hitmod.joker_hit_value(g, "j_joker")
    assert v >= 0.0
    assert g.state_signature() == sig_before


def test_joker_hit_value_deterministic():
    g = _fresh_shop_game()
    v1 = hitmod.joker_hit_value(g, "j_blueprint")
    v2 = hitmod.joker_hit_value(g, "j_blueprint")
    assert v1 == v2


def test_joker_hit_value_at_least_sell_value():
    """A joker's value can never be worth less than what you could immediately resell it
    for (raw uplift is floored at 0 before the sell value is added)."""
    g = _fresh_shop_game()
    v = hitmod.joker_hit_value(g, "j_credit_card")   # a weak, non-scoring joker
    sell = max(1, _gk.JOKER_COST.get("j_credit_card", 2) // 2)
    assert v >= sell


# ══════════════════════════════════════════════════════════ flat tables

def test_tarot_value_positive_for_real_tarots_zero_otherwise():
    assert hitmod.tarot_value("c_magician") > 0
    assert hitmod.tarot_value("not_a_real_key") == 0.0


def test_spectral_value_soul_and_black_hole_are_standouts():
    assert hitmod.spectral_value("c_soul") > hitmod.spectral_value("c_familiar")
    assert hitmod.spectral_value("c_black_hole") > hitmod.spectral_value("c_familiar")


def test_planet_value_bounded_and_documented_uniform_prior():
    g = _fresh_shop_game()
    for key in _gk.PLANET_KEYS:
        v = hitmod.planet_value(g, key)
        assert 0.0 <= v <= hitmod.DEFAULT.planet_base_value * 3.0 + 1e-9


def test_planet_value_diminishes_with_level():
    g = _fresh_shop_game()
    key = _gk.PLANET_KEYS[0]
    hand = _gk.PLANET_HAND[key]
    g.planet_levels[hand] = 1
    v1 = hitmod.planet_value(g, key)
    g.planet_levels[hand] = 5
    v5 = hitmod.planet_value(g, key)
    assert v5 < v1


def test_voucher_value_standouts_beat_default_formula():
    assert hitmod.voucher_value("v_blank") < hitmod.voucher_value("v_hieroglyph")


def test_card_value_enhancement_and_edition_stack():
    class FakeItem:
        enhancement = "Steel"
        edition = "Polychrome"
    v = hitmod.card_value(FakeItem())
    base = hitmod.DEFAULT.card_base_value
    assert v == base + hitmod.DEFAULT.enhancement_card_value["Steel"] + hitmod.DEFAULT.edition_card_bonus["Polychrome"]
