"""
test_gumbel.py — tests for Gumbel root selection (Danihelka et al. 2022).

Forked from balatro-mcts `balatro_sim/tests/test_gumbel.py`, re-targeted to the fork
engine (no `reset()`, 447-dim obs) and extended with the fork's no-legal-action cases.

Covers:
  - run_gumbel completes on SELECTING_HAND (large m) and BLIND_SELECT (small m)
  - chosen_action is always a legal action
  - total root visits ~= num_simulations
  - all actions in the initial top-m get at least one visit
  - high-logit actions are preferred when logits are imbalanced
  - small action space (legal < max_considered) is handled
  - Gumbel visits more distinct edges than vanilla PUCT does in the lock-in case
  - a terminal or no-action root returns (root, {}, None) instead of raising
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from balatro_sim.game import BalatroGame, State
from balatro_sim.jokers.base import JokerInstance
from mcts import MCTS, NNPolicy, PolicyValueNet
from mcts.action import action_key
from mcts.search import MCTSConfig


# ── Fixtures ───────────────────────────────────────────────────────────────

def _mid_blind(seed: int = 42) -> BalatroGame:
    g = BalatroGame(seed=seed)
    g.step({"type": "play_blind"})
    g.dollars = 30
    g.jokers = [
        JokerInstance("j_joker"),
        JokerInstance("j_green_joker"),
        JokerInstance("j_steel_joker"),
    ]
    return g


def _blind_select(seed: int = 42) -> BalatroGame:
    return BalatroGame(seed=seed)


@pytest.fixture(scope="module")
def policy() -> NNPolicy:
    torch.manual_seed(0)
    return NNPolicy(PolicyValueNet(), device="cpu")


# ── Smoke ──────────────────────────────────────────────────────────────────

def test_gumbel_runs_on_selecting_hand(policy):
    g = _mid_blind()
    legal = g.legal_actions()
    cfg = MCTSConfig(num_simulations=200, gumbel_max_considered=16)
    mcts = MCTS(policy, cfg, rng=np.random.default_rng(0))
    root, visits, chosen = mcts.run_gumbel(g)

    assert chosen in {action_key(a) for a in legal}
    assert sum(visits.values()) == cfg.num_simulations
    assert root.visit_count == cfg.num_simulations


def test_gumbel_runs_on_blind_select(policy):
    """Small action space (m_init <= 3) — should not crash, just allocate sims to the
    surviving candidates."""
    g = _blind_select()
    legal = g.legal_actions()
    assert len(legal) <= 3  # play_blind / skip_blind (/ reroll_boss with a voucher)

    cfg = MCTSConfig(num_simulations=50)
    mcts = MCTS(policy, cfg, rng=np.random.default_rng(0))
    root, visits, chosen = mcts.run_gumbel(g)

    assert chosen in {action_key(a) for a in legal}
    assert sum(visits.values()) == cfg.num_simulations


def test_gumbel_top_m_all_get_visited(policy):
    """Every action in the initial top-m candidate set must get >= 1 visit."""
    g = _mid_blind()
    cfg = MCTSConfig(num_simulations=200, gumbel_max_considered=16)
    mcts = MCTS(policy, cfg, rng=np.random.default_rng(0))
    _, visits, _ = mcts.run_gumbel(g)
    visited = sum(1 for v in visits.values() if v > 0)
    assert visited >= 16, f"only {visited} edges visited, expected >= 16"


# ── Logit-bias preference ──────────────────────────────────────────────────

class HandcraftedPolicy:
    """
    Stub policy that returns a sharp prior over a single 'preferred_type' action type
    (e.g. 'discard'), uniform within that type. Other types get epsilon mass. Value is
    fixed at 0.5. Used to verify that Gumbel respects logit ordering.
    """
    def __init__(self, preferred_type: str = "discard", boost: float = 100.0):
        self.preferred = preferred_type
        self.boost = boost

    def __call__(self, game):
        legal = game.legal_actions()
        if not legal:
            return {}, 0.0
        keys = [action_key(a) for a in legal]
        weights = np.array([
            self.boost if k[0] == self.preferred else 1.0 for k in keys
        ])
        probs = weights / weights.sum()
        return {k: float(p) for k, p in zip(keys, probs)}, 0.5

    def evaluate_many(self, games):
        return [self(g) for g in games]


def test_gumbel_respects_logit_ordering():
    """When priors heavily favor 'discard' actions, Gumbel should mostly pick a discard
    action, and discard visits should dominate play visits."""
    g = _mid_blind()
    pol = HandcraftedPolicy(preferred_type="discard", boost=100.0)

    chosen_types = []
    for seed in range(8):
        cfg = MCTSConfig(num_simulations=200, gumbel_max_considered=16)
        mcts = MCTS(pol, cfg, rng=np.random.default_rng(seed))
        _, visits, chosen = mcts.run_gumbel(g)
        chosen_types.append(chosen[0])
        discard_v = sum(v for k, v in visits.items() if k[0] == "discard")
        play_v = sum(v for k, v in visits.items() if k[0] == "play")
        assert discard_v > play_v, (
            f"seed {seed}: discard {discard_v} <= play {play_v} despite 100x prior boost"
        )

    assert chosen_types.count("discard") >= 6, chosen_types


# ── Edge cases ─────────────────────────────────────────────────────────────

def test_gumbel_handles_legal_smaller_than_max_considered(policy):
    """When legal_actions < max_considered, all legals are candidates."""
    g = _blind_select()
    n = len(g.legal_actions())
    cfg = MCTSConfig(num_simulations=20, gumbel_max_considered=16)
    mcts = MCTS(policy, cfg, rng=np.random.default_rng(0))
    _, visits, _ = mcts.run_gumbel(g)
    visited = sum(1 for v in visits.values() if v > 0)
    assert visited == n, f"expected all {n} legals visited, got {visited}"


def test_gumbel_explores_more_than_locked_puct(policy):
    """
    The motivation for Gumbel: vanilla PUCT (no Dirichlet noise) collapses on ~1 edge
    with random init. Gumbel should never lock in — it always visits at least m_init
    distinct edges.
    """
    g = _mid_blind()

    mcts_puct = MCTS(policy, MCTSConfig(num_simulations=200, dirichlet_eps=0.0),
                     rng=np.random.default_rng(0))
    _, v_puct = mcts_puct.run(g, add_noise=False)
    edges_puct = sum(1 for v in v_puct.values() if v > 0)

    mcts_g = MCTS(policy, MCTSConfig(num_simulations=200, gumbel_max_considered=16),
                  rng=np.random.default_rng(0))
    _, v_g, _ = mcts_g.run_gumbel(g)
    edges_g = sum(1 for v in v_g.values() if v > 0)

    assert edges_g > edges_puct, (
        f"Gumbel should explore more edges than locked PUCT "
        f"(puct={edges_puct}, gumbel={edges_g})"
    )
    assert edges_g >= 16


def test_gumbel_on_terminal_root_returns_none(policy):
    """A GAME_OVER root has no legal actions. The original raised RuntimeError; the
    fork returns an empty search so drivers can handle it."""
    g = _mid_blind()
    g.state = State.GAME_OVER
    mcts = MCTS(policy, MCTSConfig(num_simulations=10), rng=np.random.default_rng(0))
    root, visits, chosen = mcts.run_gumbel(g)
    assert visits == {} and chosen is None
    assert root.is_terminal and root.stop_reason == "game_over"


def test_puct_on_terminal_root_returns_empty(policy):
    g = _mid_blind()
    g.state = State.GAME_OVER
    mcts = MCTS(policy, MCTSConfig(num_simulations=10), rng=np.random.default_rng(0))
    root, visits = mcts.run(g)
    assert visits == {}
    assert root.is_terminal
