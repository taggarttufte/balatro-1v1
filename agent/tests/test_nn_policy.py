"""
test_nn_policy.py — smoke tests for the NN policy/value pipeline.

Forked from balatro-mcts `balatro_sim/tests/test_nn_policy.py` and re-targeted:
the encoder is env_v7's (447 dims, not the copied 434), the action feature vector is 56
dims, and `game.reset()` is not needed (the fork's constructor does the run-start draws).

Covers:
  - encode_obs: shape, dtype, finiteness, and equality with a real BalatroV7Env
  - featurize_actions: shape, finiteness, type one-hot is set, no key collisions
  - PolicyValueNet: forward pass produces (N,) logits + scalar value
  - NNPolicy: priors sum to 1.0, keys cover all legal actions
  - MCTS.run with NNPolicy: completes without crashing, visits == num_sims
  - Dirichlet noise + temperature sampling
  - the batching seam: evaluate_many == single-leaf, score_actions_flat == per-state
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from balatro_sim.game import BalatroGame
from balatro_sim.jokers.base import JokerInstance
from mcts import NNPolicy, PolicyValueNet, MCTS, UniformPolicy
from mcts.action import action_key
from mcts.action_features import (
    ACTION_FEATURE_DIM, ACTION_TYPE_IDX, featurize_action, featurize_actions,
)
from mcts.encoder import OBS_DIM, encode_obs, encode_obs_mlb, MLB_OBS_DIM
from mcts.search import MCTSConfig


# ── Fixtures ───────────────────────────────────────────────────────────────

def _mid_blind_state(seed: int = 42) -> BalatroGame:
    """SELECTING_HAND with 3 jokers — same shape as the demo / benchmark."""
    g = BalatroGame(seed=seed)
    g.step({"type": "play_blind"})
    g.dollars = 30
    g.jokers = [
        JokerInstance("j_joker"),
        JokerInstance("j_green_joker"),
        JokerInstance("j_steel_joker"),
    ]
    return g


def _blind_select_state(seed: int = 42) -> BalatroGame:
    """Fresh game in BLIND_SELECT (2-3 legal actions)."""
    return BalatroGame(seed=seed)


@pytest.fixture(scope="module")
def net() -> PolicyValueNet:
    """Single shared model — random init."""
    torch.manual_seed(0)
    return PolicyValueNet()


@pytest.fixture(scope="module")
def policy(net) -> NNPolicy:
    return NNPolicy(net, device="cpu")


# ── Encoder ────────────────────────────────────────────────────────────────

def test_encode_obs_shape_and_finite():
    g = _mid_blind_state()
    obs = encode_obs(g)
    assert obs.shape == (OBS_DIM,)
    assert OBS_DIM == 447
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()


def test_encode_obs_blind_select():
    g = _blind_select_state()
    obs = encode_obs(g)
    assert obs.shape == (OBS_DIM,)
    assert np.isfinite(obs).all()


def test_encode_obs_phase_onehot_changes():
    """Phase one-hot at idx 9..13 must shift between BLIND_SELECT and SELECTING_HAND."""
    o_blind = encode_obs(_blind_select_state())
    o_play = encode_obs(_mid_blind_state())
    assert o_blind[9] == 1.0 and o_play[9] == 0.0
    assert o_play[10] == 1.0 and o_blind[10] == 0.0


def test_encoder_matches_env_v7():
    """The whole point of the re-target: we reuse env_v7's encoder rather than copying
    it. If `_encode_obs` ever starts reading more of `self` than `.game`, this fails."""
    from balatro_sim.env_v7 import BalatroV7Env
    env = BalatroV7Env(seed=42)
    env.reset()
    assert np.array_equal(encode_obs(env.game), env._encode_obs())


def test_mlb_encoder_extends_v7():
    g = BalatroGame(seed=42, ruleset="mlb")
    obs = encode_obs_mlb(g)
    assert obs.shape == (MLB_OBS_DIM,)
    assert MLB_OBS_DIM == OBS_DIM + 6
    assert np.array_equal(obs[:OBS_DIM], encode_obs(g))
    assert np.isfinite(obs).all()
    assert obs[OBS_DIM] == pytest.approx(1.0)     # 4/4 lives at run start


def test_mlb_encoder_is_safe_on_vanilla():
    g = _mid_blind_state()
    obs = encode_obs_mlb(g)
    assert np.isfinite(obs).all()
    assert obs[OBS_DIM] == 0.0                    # vanilla has no lives


# ── Action featurization ───────────────────────────────────────────────────

def test_featurize_action_type_onehot():
    f = featurize_action({"type": "play", "cards": [0, 2, 4]})
    assert f.shape == (ACTION_FEATURE_DIM,)
    assert f[ACTION_TYPE_IDX["play"]] == 1.0
    base = len(ACTION_TYPE_IDX)
    assert f[base + 0] == 1.0 and f[base + 2] == 1.0 and f[base + 4] == 1.0
    assert f[base + 1] == 0.0 and f[base + 3] == 0.0


def test_featurize_actions_stack_shape():
    g = _mid_blind_state()
    legal = g.legal_actions()
    feats = featurize_actions(legal)
    assert feats.shape == (len(legal), ACTION_FEATURE_DIM)
    assert np.isfinite(feats).all()
    type_block = feats[:, : len(ACTION_TYPE_IDX)]
    assert np.allclose(type_block.sum(axis=1), 1.0)


def test_featurize_covers_reroll_boss():
    """The fork's legal_actions() emits `reroll_boss`; the original's 12-type vocabulary
    would have featurized it as the all-zero unknown row."""
    f = featurize_action({"type": "reroll_boss"})
    assert f[ACTION_TYPE_IDX["reroll_boss"]] == 1.0
    assert f.sum() == 1.0


def test_featurize_distinguishes_overflow_indices():
    """Indices past a slot bound must still be distinguishable from one another."""
    a = featurize_action({"type": "play", "cards": [0, 13]})
    b = featurize_action({"type": "play", "cards": [0, 15]})
    assert not np.array_equal(a, b)


def test_featurize_rows_are_unique_for_distinct_actions():
    """No two legal actions in a real state may share a feature row — identical rows get
    identical priors, which silently collapses the search's action space."""
    g = _mid_blind_state()
    legal = g.legal_actions()
    feats = featurize_actions(legal)
    rows = {f.tobytes() for f in feats}
    assert len(rows) == len(legal)


# ── Model ──────────────────────────────────────────────────────────────────

def test_model_forward_shapes(net):
    g = _mid_blind_state()
    obs = encode_obs(g)
    legal = g.legal_actions()
    feats = featurize_actions(legal)

    with torch.no_grad():
        logits, value = net(torch.from_numpy(obs), torch.from_numpy(feats))

    assert logits.shape == (len(legal),)
    assert value.dim() == 0
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).item()


def test_score_actions_flat_matches_per_state(net):
    """The ragged-batch policy-head path W3 will use must equal the per-state one."""
    games = [_mid_blind_state(seed=s) for s in (1, 2, 3)]
    obs = torch.from_numpy(np.stack([encode_obs(g) for g in games]))
    feat_list = [featurize_actions(g.legal_actions()) for g in games]
    counts = torch.tensor([f.shape[0] for f in feat_list])
    flat = torch.from_numpy(np.concatenate(feat_list, axis=0))

    with torch.no_grad():
        trunk = net.get_trunk(obs)
        got = net.score_actions_flat(trunk, flat, counts)
        want = torch.cat([net.score_actions(trunk[i], torch.from_numpy(f))
                          for i, f in enumerate(feat_list)])
    assert torch.allclose(got, want, atol=1e-5)


# ── NNPolicy ───────────────────────────────────────────────────────────────

def test_nn_policy_priors_sum_to_one(policy):
    g = _mid_blind_state()
    priors, value = policy(g)
    assert isinstance(value, float)
    assert np.isfinite(value)
    assert len(priors) > 0
    assert abs(sum(priors.values()) - 1.0) < 1e-5


def test_nn_policy_covers_all_legal_actions(policy):
    g = _mid_blind_state()
    legal = g.legal_actions()
    priors, _ = policy(g)
    assert set(priors.keys()) == {action_key(a) for a in legal}


def test_nn_policy_blind_select(policy):
    """Small action set (play_blind, skip_blind) — make sure nothing regresses."""
    g = _blind_select_state()
    priors, value = policy(g)
    assert len(priors) >= 1
    assert abs(sum(priors.values()) - 1.0) < 1e-5
    assert np.isfinite(value)


def test_nn_policy_rejects_encoder_mismatch(net):
    from mcts.encoder import MLBEncoder
    with pytest.raises(ValueError):
        NNPolicy(net, encoder=MLBEncoder())      # 453-dim encoder, 447-dim net


# ── The batching seam (W3's contract) ──────────────────────────────────────

def test_evaluate_many_matches_single_leaf(policy):
    games = [_mid_blind_state(seed=s) for s in (1, 2, 3)]
    many = policy.evaluate_many(games)
    assert len(many) == len(games)
    for g, (priors, value) in zip(games, many):
        one_priors, one_value = policy(g)
        assert priors.keys() == one_priors.keys()
        assert value == pytest.approx(one_value)
        for k in priors:
            assert priors[k] == pytest.approx(one_priors[k])


def test_evaluate_many_exists_on_every_policy(policy):
    from mcts.policy import PolicyValueFn
    for p in (policy, UniformPolicy()):
        assert isinstance(p, PolicyValueFn)
        assert callable(p.evaluate_many)


# ── End-to-end MCTS run ────────────────────────────────────────────────────

def test_mcts_with_nn_policy_runs(policy):
    """100 sims with NNPolicy — must finish, root.visit_count must equal num_sims."""
    g = _mid_blind_state()
    mcts = MCTS(policy, MCTSConfig(num_simulations=100, dirichlet_eps=0.0))
    root, visits = mcts.run(g, add_noise=False)
    assert root.visit_count == 100
    assert sum(visits.values()) == 100
    assert len(visits) == len(g.legal_actions())


# ── Root exploration: Dirichlet noise + temperature sampling ───────────────

def test_dirichlet_noise_perturbs_root_priors(policy):
    """With eps>0, root priors must differ from the bare policy output."""
    g = _mid_blind_state()
    bare_priors, _ = policy(g)
    bare = np.array([bare_priors[k] for k in sorted(bare_priors)])

    mcts = MCTS(
        policy,
        MCTSConfig(num_simulations=10, dirichlet_alpha=0.3, dirichlet_eps=0.25),
        rng=np.random.default_rng(0),
    )
    root, _ = mcts.run(g, add_noise=True)
    noisy = np.array([root.children[k].prior for k in sorted(bare_priors)])

    assert abs(noisy.sum() - 1.0) < 1e-4
    assert np.max(np.abs(noisy - bare)) > 1e-3
    assert noisy.std() > bare.std()


def test_dirichlet_disabled_when_eps_zero(policy):
    """eps=0 (or alpha=0) leaves priors unchanged."""
    g = _mid_blind_state()
    bare_priors, _ = policy(g)

    mcts = MCTS(
        policy,
        MCTSConfig(num_simulations=10, dirichlet_alpha=0.3, dirichlet_eps=0.0),
        rng=np.random.default_rng(0),
    )
    root, _ = mcts.run(g, add_noise=True)
    for k, child in root.children.items():
        assert abs(child.prior - bare_priors[k]) < 1e-7


def test_dirichlet_noise_increases_exploration(policy):
    """
    Without noise, PUCT collapses on a single positive-Q action and only ~2 edges get
    visits. With noise (alpha=0.03 to match the ~436-action space), averaged over
    multiple seeds the visited-edge count should exceed the no-noise baseline.

    Why averaged: a single Dirichlet draw is high-variance — sometimes it concentrates
    noise on a single action, which still locks in. The *expected* visit-edge count
    rises, but any single seed can fail.
    """
    g = _mid_blind_state()

    edges_off, edges_on = [], []
    for seed in range(5):
        mcts_off = MCTS(policy, MCTSConfig(num_simulations=200, dirichlet_eps=0.0),
                        rng=np.random.default_rng(seed))
        _, v_off = mcts_off.run(g, add_noise=False)
        edges_off.append(sum(1 for v in v_off.values() if v > 0))

        mcts_on = MCTS(policy,
                       MCTSConfig(num_simulations=200, dirichlet_alpha=0.03,
                                  dirichlet_eps=0.25),
                       rng=np.random.default_rng(seed))
        _, v_on = mcts_on.run(g, add_noise=True)
        edges_on.append(sum(1 for v in v_on.values() if v > 0))

    assert np.mean(edges_on) > np.mean(edges_off), (
        f"noise should increase mean exploration "
        f"(off mean={np.mean(edges_off):.1f}, on mean={np.mean(edges_on):.1f})"
    )


def test_sample_action_argmax_at_zero_temperature():
    visits = {("a",): 1, ("b",): 100, ("c",): 5}
    for _ in range(20):
        assert MCTS.sample_action(visits, temperature=0.0) == ("b",)


def test_sample_action_distribution_at_unit_temperature():
    """At T=1, sample frequencies should approach the visit-count distribution."""
    visits = {("a",): 10, ("b",): 80, ("c",): 10}
    rng = np.random.default_rng(123)
    counts = {k: 0 for k in visits}
    n = 5000
    for _ in range(n):
        counts[MCTS.sample_action(visits, temperature=1.0, rng=rng)] += 1
    for k, v in visits.items():
        target = v / sum(visits.values())
        empirical = counts[k] / n
        assert abs(empirical - target) < 0.03, f"{k}: target {target:.3f}, got {empirical:.3f}"


def test_sample_action_handles_zero_counts():
    """If no edge has been visited, fall back to a uniform pick."""
    visits = {("a",): 0, ("b",): 0, ("c",): 0}
    assert MCTS.sample_action(visits, temperature=1.0, rng=np.random.default_rng(0)) in visits


def test_sample_and_best_action_on_empty_visits():
    """A search on a state with no actions returns {}; both selectors must say None."""
    assert MCTS.best_action({}) is None
    assert MCTS.sample_action({}, temperature=1.0) is None
