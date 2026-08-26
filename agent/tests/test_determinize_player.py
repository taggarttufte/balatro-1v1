"""
test_determinize_player.py — Phase 5 W2: `mcts/determinize.py`'s non-clairvoyant MCTS
player, built on `BalatroGame.clone_determinized` without touching `search.py`/
`player.py`.

Covers:
  - the wrapper never mutates the game handed to `act()`
  - reproducibility given a fixed `determinize_seed`; divergence across different ones
  - both modes ("per_sim" PIMC, "per_search" cheap) run without raising on BLIND_SELECT,
    a mid-hand state, a SHOP with money, and an MLB Nemesis state
  - a short full playthrough (cold-start net, small sims) completes
  - the `real1/latest.pt` checkpoint loads through `make_determinized_player` and
    produces a legal action (skipped if the checkpoint file is not present — it lives
    under the gitignored `agent/runs/`)
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from balatro_sim.game import BalatroGame, State
from balatro_sim.jokers.base import JokerInstance
from mcts import MCTSConfig, NNPolicy, PolicyValueNet
from mcts.action import action_key
from mcts.determinize import (
    DeterminizedMCTSPlayer, make_determinized_player, make_determinizing_view, seed_stream,
)
from mcts.player import MCTSPlayer, make_player

_HERE = os.path.dirname(os.path.abspath(__file__))
_REAL1_CKPT = os.path.join(_HERE, os.pardir, "runs", "real1", "latest.pt")


def _mid_blind(seed: int = 42) -> BalatroGame:
    g = BalatroGame(seed=seed)
    g.step({"type": "play_blind"})
    g.dollars = 30
    g.jokers = [JokerInstance("j_joker"), JokerInstance("j_green_joker")]
    return g


def _shop(seed: int = 42) -> BalatroGame:
    g = BalatroGame(seed=seed)
    g.step({"type": "play_blind"})
    g.debug_win_blind()
    g.step({"type": "advance"})
    while g.state == State.BOOSTER_OPEN:
        g.step({"type": "skip_booster"})
    assert g.state == State.SHOP
    g.dollars = 40
    return g


def _blind_select(seed: int = 42) -> BalatroGame:
    return BalatroGame(seed=seed)


def _mlb_nemesis(seed: str = "7I4M53DL") -> BalatroGame:
    """A vanilla BLIND_SELECT for MLB ruleset; ante bumped past `pvp_start_round` so the
    Boss slot is a Nemesis blind (no MLBMatch needed — `pvp_solo=True` starts it alone)."""
    g = BalatroGame(seed=seed, ruleset="mlb")
    g.ante = g.pvp_start_round
    g.blind_idx = 2
    g._prepare_next_blind()
    assert g.current_blind.is_pvp
    return g


@pytest.fixture(scope="module")
def cold_player() -> MCTSPlayer:
    torch.manual_seed(0)
    policy = NNPolicy(PolicyValueNet(), device="cpu")
    return MCTSPlayer(policy=policy, config=MCTSConfig(num_simulations=10),
                      strategy="gumbel", reuse=True, name="cold")


def _fresh_inner(sims: int = 10) -> MCTSPlayer:
    torch.manual_seed(0)
    policy = NNPolicy(PolicyValueNet(), device="cpu")
    # `rng=` fixed: MCTSPlayer defaults to a fresh (non-reproducible) np.random.Generator,
    # which would make Gumbel sampling noise differ between two "identical" players and
    # confound the determinize_seed reproducibility check below with unrelated randomness.
    return MCTSPlayer(policy=policy, config=MCTSConfig(num_simulations=sims),
                      strategy="gumbel", reuse=True, name="cold",
                      rng=np.random.default_rng(0))


# ══════════════════════════════════════════════════════════════════════════ seed_stream
def test_seed_stream_deterministic_given_base():
    a = list(next(seed_stream(123)) for _ in range(20))
    b = list(next(seed_stream(123)) for _ in range(20))
    assert a == b


def test_seed_stream_differs_across_base_seeds():
    a = [next(seed_stream(1)) for _ in range(10)]
    b = [next(seed_stream(2)) for _ in range(10)]
    assert a != b


def test_seed_stream_none_is_fresh():
    a = [next(seed_stream(None)) for _ in range(10)]
    b = [next(seed_stream(None)) for _ in range(10)]
    assert a != b


def test_seed_stream_values_in_range():
    for v in [next(seed_stream(7)) for _ in range(200)]:
        assert 0 <= v < (1 << 40)


# ══════════════════════════════════════════════════════════════════════════ make_determinizing_view
def test_view_does_not_mutate_source_game():
    g = _shop()
    before = g.state_signature()
    view = make_determinizing_view(g, seed_stream(1))
    assert g.state_signature() == before
    assert view is not g


def test_view_clone_calls_are_determinized_and_do_not_recurse():
    g = _shop()
    view = make_determinizing_view(g, seed_stream(1))
    c1 = view.clone()
    c2 = view.clone()
    assert getattr(c1, "determinized", False) and getattr(c2, "determinized", False)
    # view itself keeps behaving like a normal game after being cloned from repeatedly
    assert view.state == g.state
    assert view.dollars == g.dollars


def test_view_clone_many_times_no_recursion_error():
    """Regression: the shadowed `.clone()` must not recurse into itself via
    `clone_determinized`'s own internal `self.clone()` call."""
    g = _mid_blind()
    view = make_determinizing_view(g, seed_stream(5))
    for _ in range(50):
        c = view.clone()
        assert c.state == view.state


# ══════════════════════════════════════════════════════════════════════════ DeterminizedMCTSPlayer
@pytest.mark.parametrize("mode", ["per_sim", "per_search"])
@pytest.mark.parametrize("state_fn", [_blind_select, _mid_blind, _shop, _mlb_nemesis])
def test_act_returns_legal_action_and_does_not_mutate(mode, state_fn):
    g = state_fn()
    before = g.state_signature()
    player = DeterminizedMCTSPlayer(inner=_fresh_inner(sims=8), determinize_seed=1, mode=mode)
    legal = g.legal_actions()
    a = player.act(g)
    assert g.state_signature() == before      # never mutated
    if not legal:
        assert a is None or isinstance(a, dict)
    else:
        assert a is not None
        assert action_key(a) in {action_key(x) for x in legal}


@pytest.mark.parametrize("mode", ["per_sim", "per_search"])
def test_reproducible_given_same_determinize_seed(mode):
    g = _shop()
    p1 = DeterminizedMCTSPlayer(inner=_fresh_inner(sims=8), determinize_seed=99, mode=mode)
    p2 = DeterminizedMCTSPlayer(inner=_fresh_inner(sims=8), determinize_seed=99, mode=mode)
    a1 = p1.act(g.clone())
    a2 = p2.act(g.clone())
    assert action_key(a1) == action_key(a2)


def test_no_action_state_delegates_without_determinizing():
    """A state with no legal actions must not crash the wrapper — it should fall
    through to the inner player's own `no_action` handling. `play_blind` on a
    match-coordinated (`pvp_solo=False`) Nemesis just readies the player and waits for
    `startBlind` (no MLBMatch here to send it) — one of MLB's two no-action states."""
    g = _mlb_nemesis()
    g.pvp_solo = False    # a solo Nemesis (no MLBMatch) would resolve immediately instead
    g.step({"type": "play_blind"})
    assert g.state == State.BLIND_SELECT and g.pvp_ready
    assert g.legal_actions() == []
    inner = make_player(checkpoint=None, sims=4, seed=0)
    player = DeterminizedMCTSPlayer(inner=inner, determinize_seed=1)
    a = player.act(g)
    assert a == inner.no_action


def test_reset_delegates_to_inner():
    inner = _fresh_inner(sims=8)
    player = DeterminizedMCTSPlayer(inner=inner, determinize_seed=1)
    g = _shop()
    player.act(g)
    assert inner.cache.armed or not inner.cache.cfg.enabled
    player.reset()
    assert not inner.cache.armed


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        DeterminizedMCTSPlayer(inner=_fresh_inner(sims=4), mode="bogus")


# ══════════════════════════════════════════════════════════════════════════ make_determinized_player
def test_make_determinized_player_cold_start():
    player = make_determinized_player(checkpoint=None, sims=6, device="cpu", seed=0,
                                      determinize_seed=3)
    assert isinstance(player, DeterminizedMCTSPlayer)
    g = _blind_select()
    a = player.act(g)
    assert a is not None


@pytest.mark.skipif(not os.path.exists(_REAL1_CKPT),
                    reason="agent/runs/real1/latest.pt not present on this checkout")
def test_make_determinized_player_loads_real1_checkpoint():
    """Sanity: the checkpoint loader real1.sh trained under (encoder=set, heuristic
    prior) loads through the determinized wrapper and returns a legal action on a
    fresh BLIND_SELECT."""
    player = make_determinized_player(
        checkpoint=_REAL1_CKPT, sims=4, device="cpu", seed=0, determinize_seed=1,
        encoder="set", heuristic_prior=0.4, heuristic_tau=0.35, max_hand_candidates=32)
    g = BalatroGame(seed="7I4M53DL", ruleset="vanilla")
    legal = g.legal_actions()
    a = player.act(g)
    assert action_key(a) in {action_key(x) for x in legal}


# ══════════════════════════════════════════════════════════════════════════ playthrough smoke
@pytest.mark.parametrize("mode", ["per_sim", "per_search"])
def test_short_playthrough_cold_start(mode):
    player = DeterminizedMCTSPlayer(inner=_fresh_inner(sims=6),
                                    determinize_seed=11, mode=mode)
    g = BalatroGame(seed="AAAAAAAA", ruleset="vanilla")
    steps = 0
    while g.state != State.GAME_OVER and steps < 30:
        a = player.act(g)
        if a is None:
            a = {"type": "advance"}
        g.step(a)
        steps += 1
    assert steps > 0   # completed without raising
