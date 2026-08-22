"""
test_train.py — unit tests for the train/ package.

Forked from balatro-mcts `balatro_sim/tests/test_train.py` and re-targeted: 447-dim obs,
56-dim action features, no `reset()`, and the outcome/z assertions now go through the
OutcomeFn instead of the hardcoded ante-8 rule.

Covers:
  ReplayBuffer:    bounded capacity, add/extend/sample, checkpoint round-trip
  SelfPlayAgent:   plays a non-trivial trajectory; samples have the right shapes;
                   the episode result still unpacks as the old 3-tuple
  Outcome:         vanilla / MLB / external value functions
  Trainer:         step produces finite losses and changes the network's weights
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from balatro_sim.game import BalatroGame, State
from mcts import NNPolicy, PolicyValueNet
from mcts.action_features import ACTION_FEATURE_DIM
from mcts.encoder import OBS_DIM
from mcts.outcome import (
    ExternalOutcome, MLBOutcome, VanillaOutcome, default_outcome_for, margin_to_value,
)
from mcts.search import MCTSConfig
from train import ReplayBuffer, Sample, SelfPlayAgent, Trainer


# ── ReplayBuffer ────────────────────────────────────────────────────────────

def _fake_sample(n_actions: int = 5, z: float = 0.0,
                 rng: np.random.Generator | None = None) -> Sample:
    """
    Random non-zero obs/features. Zero obs would drive the trunk through all-zero biases
    to a literal zero output, giving zero gradient — useful for catching bugs but bad for
    the "weights move" assertion below.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    obs = rng.standard_normal(OBS_DIM).astype(np.float32) * 0.3
    feats = rng.standard_normal((n_actions, ACTION_FEATURE_DIM)).astype(np.float32) * 0.3
    return Sample(
        obs=obs,
        action_features=feats,
        target_policy=np.full(n_actions, 1.0 / n_actions, dtype=np.float32),
        z=z,
    )


def test_buffer_capacity_is_bounded():
    buf = ReplayBuffer(capacity=5)
    for _ in range(10):
        buf.add(_fake_sample())
    assert len(buf) == 5


def test_buffer_sample_returns_requested_size():
    buf = ReplayBuffer(capacity=100)
    buf.extend(_fake_sample() for _ in range(20))
    out = buf.sample(8, rng=np.random.default_rng(0))
    assert len(out) == 8
    assert all(isinstance(s, Sample) for s in out)


def test_buffer_empty_sample_returns_empty_list():
    assert ReplayBuffer(capacity=10).sample(4) == []


def test_buffer_state_dict_round_trip():
    buf = ReplayBuffer(capacity=100)
    buf.extend(_fake_sample(z=float(i) / 10) for i in range(10))
    clone = ReplayBuffer(capacity=1)
    clone.load_state_dict(buf.state_dict())
    assert len(clone) == len(buf)
    a = buf.sample(5, rng=np.random.default_rng(3))
    b = clone.sample(5, rng=np.random.default_rng(3))
    for x, y in zip(a, b):
        assert np.array_equal(x.obs, y.obs) and x.z == y.z


def test_buffer_state_dict_truncation_is_flagged():
    buf = ReplayBuffer(capacity=100)
    buf.extend(_fake_sample() for _ in range(10))
    sd = buf.state_dict(max_items=4)
    assert sd["truncated"] is True and len(sd["samples"]) == 4 and sd["n_total"] == 10


# ── Outcome functions ───────────────────────────────────────────────────────

def test_vanilla_outcome_win_and_progress():
    o = VanillaOutcome()
    g_low = BalatroGame(seed=1)
    g_low.state = State.GAME_OVER
    g_low.ante, g_low.blind_idx, g_low.chips_scored = 1, 0, 0

    g_high = BalatroGame(seed=1)
    g_high.step({"type": "play_blind"})
    g_high.state = State.GAME_OVER
    g_high.ante, g_high.blind_idx, g_high.chips_scored = 2, 0, 0

    assert o.is_terminal(g_low) and not o.is_win(g_low)
    assert o.value(g_high) > o.value(g_low)

    g_win = BalatroGame(seed=1)
    g_win.ante = 9
    g_win.state = State.GAME_OVER
    assert o.is_win(g_win) and o.value(g_win) == 1.0


def test_mlb_outcome_does_not_use_the_ante_8_rule():
    """Under MLB, ante > 8 is just a long run — the win is `match_won`."""
    g = BalatroGame(seed=1, ruleset="mlb")
    g.ante = 12
    g.state = State.GAME_OVER
    o = MLBOutcome()
    assert o.is_terminal(g)
    assert not o.is_win(g)              # endless: passing ante 8 is not a win
    assert VanillaOutcome().is_win(g)   # ...whereas the vanilla rule would say it is
    g.match_won = True
    assert o.is_win(g) and o.value(g) == 1.0


def test_mlb_outcome_is_monotone_in_lives_and_progress():
    o = MLBOutcome(starting_lives=4)
    g = BalatroGame(seed=1, ruleset="mlb")
    full = o.value(g)
    g.lives = 1
    assert o.value(g) < full
    g.ante = 4
    assert o.value(g) > o.value(BalatroGame(seed=1, ruleset="mlb")) - 1.0  # bounded
    assert 0.0 <= o.value(g) <= 1.0


def test_default_outcome_for_picks_by_ruleset():
    assert default_outcome_for(BalatroGame(seed=1)).name == "vanilla"
    assert default_outcome_for(BalatroGame(seed=1, ruleset="mlb")).name == "mlb"


def test_external_outcome_supplies_the_value():
    g = BalatroGame(seed=1, ruleset="mlb")
    o = ExternalOutcome(value_fn=lambda _g: 0.75)
    assert o.value(g) == 0.75
    assert o.is_win(g) is True
    m = ExternalOutcome.from_margin(lambda _g: 0.0)
    assert m.value(g) == pytest.approx(0.5)
    assert margin_to_value(3.0) > 0.9 and margin_to_value(-3.0) < 0.1


def test_external_outcome_clamps_to_unit_range():
    g = BalatroGame(seed=1)
    assert ExternalOutcome(value_fn=lambda _g: 9.0).value(g) == 1.0
    assert ExternalOutcome(value_fn=lambda _g: -9.0).value(g) == 0.0


# ── SelfPlayAgent ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def cheap_agent():
    """Tiny config: small sim budget so tests run in seconds."""
    torch.manual_seed(0)
    net = PolicyValueNet()
    policy = NNPolicy(net, device="cpu")
    cfg = MCTSConfig(num_simulations=10, gumbel_max_considered=4)
    return SelfPlayAgent(policy, cfg, rng=np.random.default_rng(0), max_decisions=200)


def test_agent_plays_an_episode(cheap_agent):
    result = cheap_agent.play_episode(BalatroGame(seed=1))
    assert len(result.samples) >= 1, "episode should produce at least one sample"
    assert isinstance(result.won, bool)
    assert result.final_ante >= 1
    assert result.stop_reason in ("game_over", "max_decisions", "max_antes")


def test_episode_result_unpacks_as_the_old_tuple(cheap_agent):
    samples, final_ante, won = cheap_agent.play_episode(BalatroGame(seed=1))
    assert isinstance(samples, list) and final_ante >= 1 and isinstance(won, bool)


def test_sample_shapes_and_policy_sums_to_one(cheap_agent):
    result = cheap_agent.play_episode(BalatroGame(seed=1))
    for s in result.samples:
        assert s.obs.shape == (OBS_DIM,)
        N = s.target_policy.shape[0]
        assert s.action_features.shape == (N, ACTION_FEATURE_DIM)
        assert N >= 1
        assert abs(s.target_policy.sum() - 1.0) < 1e-4


def test_shaped_z_is_consistent_within_a_trajectory(cheap_agent):
    """All samples in one episode share the same shaped z."""
    result = cheap_agent.play_episode(BalatroGame(seed=1))
    z0 = result.samples[0].z
    assert all(s.z == z0 for s in result.samples)
    assert z0 == result.z


def test_shaped_z_in_unit_range(cheap_agent):
    result = cheap_agent.play_episode(BalatroGame(seed=1))
    assert 0.0 <= result.samples[0].z <= 1.0


def test_shaped_z_shim_still_works():
    """`train.agent._shaped_z` is kept as a back-compat shim over VanillaOutcome."""
    from train.agent import _shaped_z
    g_low = BalatroGame(seed=1)
    g_low.state, g_low.ante, g_low.blind_idx, g_low.chips_scored = State.GAME_OVER, 1, 0, 0
    g_high = BalatroGame(seed=1)
    g_high.step({"type": "play_blind"})
    g_high.state, g_high.ante, g_high.blind_idx, g_high.chips_scored = State.GAME_OVER, 2, 0, 0
    assert _shaped_z(g_high, won=False) > _shaped_z(g_low, won=False)
    g_win = BalatroGame(seed=1)
    g_win.ante, g_win.state = 9, State.GAME_OVER
    assert _shaped_z(g_win, won=True) == 1.0


def test_agent_uses_the_external_outcome_for_z(cheap_agent):
    """W2/W4 supply the outcome; the training label must follow it, not the ante rule."""
    result = cheap_agent.play_episode(
        BalatroGame(seed=1), outcome=ExternalOutcome(value_fn=lambda _g: 0.42)
    )
    assert result.z == pytest.approx(0.42)
    assert all(s.z == pytest.approx(0.42) for s in result.samples)


# ── Trainer ─────────────────────────────────────────────────────────────────

def test_trainer_step_finite_and_moves_weights():
    torch.manual_seed(0)
    net = PolicyValueNet()
    trainer = Trainer(net, lr=1e-3, device="cpu")

    rng = np.random.default_rng(42)
    # Mix of wins and losses so the value gradient is non-degenerate.
    batch = [_fake_sample(n_actions=8, z=float(i % 2), rng=rng) for i in range(4)]
    sig_before = torch.cat([p.detach().flatten().clone() for p in net.parameters()])

    metrics = trainer.step(batch)
    sig_after = torch.cat([p.detach().flatten().clone() for p in net.parameters()])

    assert np.isfinite(metrics["policy_loss"])
    assert np.isfinite(metrics["value_loss"])
    assert metrics["n"] == 4
    assert (sig_after - sig_before).abs().mean().item() > 0


def test_trainer_step_on_empty_batch_is_safe():
    torch.manual_seed(0)
    assert Trainer(PolicyValueNet()).step([])["n"] == 0


def test_trainer_state_dict_round_trip():
    torch.manual_seed(0)
    net = PolicyValueNet()
    trainer = Trainer(net, lr=1e-3)
    rng = np.random.default_rng(1)
    trainer.step([_fake_sample(z=0.5, rng=rng) for _ in range(2)])
    sd = trainer.state_dict()
    assert sd["lr"] == 1e-3
    other = Trainer(net, lr=5e-4)
    other.load_state_dict(sd)
    assert other.lr == 1e-3
    assert other.optimizer.state_dict()["state"].keys() == \
        trainer.optimizer.state_dict()["state"].keys()
