"""sampling.py — world sampling never touches the live game and is draw-order invariant."""
from __future__ import annotations

import random

import _bootstrap  # noqa: F401
from _bootstrap import BalatroGame, State

import sampling as S


def _in_blind(seed="11111111", ruleset="vanilla"):
    g = BalatroGame(seed=seed, ruleset=ruleset)
    g.step({"type": "play_blind"})
    assert g.state == State.SELECTING_HAND
    return g


def test_sample_world_is_side_effect_free():
    g = _in_blind()
    sig = g.state_signature()
    snap = g.run_state.rng.snapshot()
    w = S.sample_world(g, random.Random(1))
    assert g.state_signature() == sig
    assert g.run_state.rng.snapshot() == snap
    assert w is not g and w.deck is not g.deck


def test_sample_world_keeps_composition_and_reshuffles():
    g = _in_blind()
    w = S.sample_world(g, random.Random(7))
    assert S.deck_composition(w) == S.deck_composition(g)
    assert len(w.hand) == len(g.hand)
    orders = {tuple(S.canonical_card_key(c) for c in S.sample_world(g, random.Random(i)).deck) for i in range(6)}
    assert len(orders) > 1, "the draw pile must actually be reshuffled"


def test_sample_world_is_invariant_under_a_deck_permutation():
    g = _in_blind()
    w1 = S.sample_world(g, S.world_rng(0, g))
    random.Random(5).shuffle(g.deck)
    w2 = S.sample_world(g, S.world_rng(0, g))
    assert [S.canonical_card_key(c) for c in w1.deck] == [S.canonical_card_key(c) for c in w2.deck]


def test_world_rng_is_a_function_of_seed_and_observable_state():
    g = _in_blind()
    a = S.world_rng(3, g).random()
    random.Random(9).shuffle(g.deck)
    b = S.world_rng(3, g).random()
    c = S.world_rng(4, g).random()
    assert a == b and a != c


def test_sample_world_prefers_the_engine_determinize_api_when_present():
    g = _in_blind()
    calls = []

    def fake(seed):
        calls.append(seed)
        return g.clone()
    g.clone_determinized = fake          # feature-detected at call time
    w = S.sample_world(g, random.Random(2))
    assert calls and S.deck_composition(w) == S.deck_composition(g)
