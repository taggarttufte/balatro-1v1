"""
test_sample_v2.py — subsampled `Sample` v2, the replay buffer, the trainer dispatch and
the set-encoder checkpoint round trip (Phase 4 W1).

The properties the brief asks for:

  * `make_sample` keeps EVERY visited action and renormalises the target over the kept
    rows (exactly — the kept support is the full support);
  * the sample shrinks by the order of magnitude the decision was made for;
  * the buffer and the checkpoint carry v1 and v2 records side by side;
  * a set-encoder run's checkpoint round trip is BIT-EXACT on CPU, exactly as the flat
    one's is.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from balatro_sim.game import BalatroGame, State

from mcts.action import action_key
from mcts.encoder import get_encoder
from mcts.encoder_set import ItemCaps
from train import ColdTrainer, TrainConfig, ReplayBuffer, load_checkpoint, save_checkpoint
from train.sample import (
    DEFAULT_K_UNVISITED, Sample, SampleBuilder, make_sample_v2, renormalised_target,
    sample_nbytes, subsample_indices,
)
from train.trainer import Trainer
from train.trajectory import Sample as SampleV1

from _states import collect_states


@pytest.fixture(scope="module")
def states():
    return collect_states(60)


def _visits(legal_keys, rng, n_visited=5):
    """A plausible search result: a handful of actions with visit counts."""
    idx = rng.choice(len(legal_keys), size=min(n_visited, len(legal_keys)), replace=False)
    return {legal_keys[i]: int(rng.integers(1, 40)) for i in idx}


# ── subsampling ─────────────────────────────────────────────────────────────────

def test_subsample_keeps_every_visited_action():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(1, 500))
        counts = np.zeros(n)
        n_vis = int(rng.integers(0, min(n, 12) + 1))
        if n_vis:
            counts[rng.choice(n, size=n_vis, replace=False)] = rng.integers(1, 50, size=n_vis)
        keep = subsample_indices(counts, 8, rng)
        visited = set(np.flatnonzero(counts > 0).tolist())
        assert visited <= set(keep.tolist())
        assert list(keep) == sorted(set(keep.tolist())), "kept indices must be sorted+unique"
        n_extra = min(8, n - len(visited)) if (n_vis or n > 0) else 0
        assert len(keep) == len(visited) + max(n_extra, 1 if not visited else n_extra) \
            or len(keep) == len(visited) + n_extra


def test_subsample_never_returns_an_empty_set():
    rng = np.random.default_rng(1)
    keep = subsample_indices(np.zeros(50), 0, rng)      # nothing visited, k=0
    assert keep.size == 1


def test_k_unvisited_zero_keeps_only_visited():
    counts = np.zeros(30)
    counts[[3, 17]] = [5, 9]
    keep = subsample_indices(counts, 0, np.random.default_rng(2))
    assert list(keep) == [3, 17]


def test_target_renormalises_exactly():
    """Because every visited action survives, the kept-row target is the FULL target
    restricted to its support — not an approximation."""
    rng = np.random.default_rng(3)
    for _ in range(100):
        n = int(rng.integers(2, 300))
        counts = np.zeros(n)
        idx = rng.choice(n, size=min(6, n), replace=False)
        counts[idx] = rng.integers(1, 60, size=idx.size)
        full = counts / counts.sum()
        keep = subsample_indices(counts, 8, rng)
        target = renormalised_target(counts, keep)
        assert abs(target.sum() - 1.0) < 1e-6
        for pos, i in enumerate(keep):
            assert abs(target[pos] - full[i]) < 1e-6


def test_target_is_uniform_when_nothing_was_visited():
    target = renormalised_target(np.zeros(10), np.array([0, 4, 9]))
    assert np.allclose(target, 1 / 3)


# ── make_sample ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("encoder_name", ["v7", "mlb", "set"])
def test_make_sample_shapes(states, encoder_name):
    enc = get_encoder(encoder_name)
    rng = np.random.default_rng(4)
    for game in states[:20]:
        legal = game.legal_actions()
        if not legal:
            continue
        keys = [action_key(a) for a in legal]
        visits = _visits(keys, rng)
        s = make_sample_v2(game, legal, keys, visits, enc, z=0.5, rng=rng)
        k = s.target_policy.shape[0]
        assert s.version == 2
        assert k == min(len(legal), len(set(visits)) + DEFAULT_K_UNVISITED)
        assert abs(float(s.target_policy.sum()) - 1.0) < 1e-5
        assert s.meta["n_legal"] == len(legal)
        assert s.meta["k"] == k
        assert s.meta["encoder"] == encoder_name
        assert s.meta["state"] == game.state.name
        if encoder_name == "set":
            assert s.is_set and isinstance(s.actions, dict)
            for key, arr in s.actions.items():
                assert arr.shape[0] == k, key
        else:
            assert not s.is_set
            assert s.obs.shape == (enc.dim,)
            assert s.actions.shape == (k, 56)


def test_make_sample_rows_line_up_with_the_kept_actions(states):
    """The k-th row of `actions` must be the k-th kept legal action, in legal order."""
    from mcts.action_features import featurize_actions
    enc = get_encoder("v7")
    rng = np.random.default_rng(5)
    game = next(g for g in states if g.state is State.SELECTING_HAND)
    legal = game.legal_actions()
    keys = [action_key(a) for a in legal]
    visits = _visits(keys, rng, n_visited=6)
    builder = SampleBuilder(enc, k_unvisited=8, rng=np.random.default_rng(11))
    s = builder(game, legal, keys, visits, None, 0.25)

    counts = np.array([visits.get(k, 0) for k in keys], dtype=float)
    keep = subsample_indices(counts, 8, np.random.default_rng(11))
    assert np.allclose(s.actions, featurize_actions([legal[i] for i in keep]))
    assert np.allclose(s.target_policy, renormalised_target(counts, keep))


def test_no_subsample_keeps_everything(states):
    enc = get_encoder("v7")
    game = next(g for g in states if g.state is State.SELECTING_HAND)
    legal = game.legal_actions()
    keys = [action_key(a) for a in legal]
    s = SampleBuilder(enc, subsample=False)(game, legal, keys, {}, None, 0.0)
    assert s.actions.shape[0] == len(legal)


def test_builder_matches_w2s_sample_fn_signature(states):
    """`train/selfplay.py::SampleCollector` calls
    `sample_fn(game, legal, legal_keys, visits, encoder, z)` positionally."""
    import inspect
    from train.selfplay import SampleCollector
    src = inspect.getsource(SampleCollector.__call__)
    assert "self.sample_fn(game, decision.legal, decision.legal_keys," in src
    enc = get_encoder("set")
    game = states[0]
    legal = game.legal_actions()
    keys = [action_key(a) for a in legal]
    s = SampleBuilder(enc)(game, legal, keys, {}, enc, 0.0)
    assert isinstance(s, Sample)


def test_make_sample_refuses_a_state_with_no_actions():
    game = BalatroGame(seed=3, deck_key="b_red", stake=1, ruleset="mlb")
    game.state = State.GAME_OVER
    with pytest.raises(ValueError):
        SampleBuilder(get_encoder("set"))(game, [], [], {}, None, 0.0)


# ── size ────────────────────────────────────────────────────────────────────────

def test_sample_v2_is_far_smaller_than_v1(states):
    """The headline of the decision: a subsampled sample is an order of magnitude
    smaller. Measured over real states, both encoders."""
    from mcts.action_features import featurize_actions
    rng = np.random.default_rng(6)
    v7 = get_encoder("v7")
    st = get_encoder("set")
    v1_bytes, flat_bytes, set_bytes = [], [], []
    for game in states:
        legal = game.legal_actions()
        if not legal:
            continue
        keys = [action_key(a) for a in legal]
        visits = _visits(keys, rng)
        v1 = SampleV1(obs=v7(game), action_features=featurize_actions(legal),
                      target_policy=np.zeros(len(legal), dtype=np.float32))
        v1_bytes.append(sample_nbytes(v1))
        flat_bytes.append(sample_nbytes(
            make_sample_v2(game, legal, keys, visits, v7, rng=rng)))
        set_bytes.append(sample_nbytes(
            make_sample_v2(game, legal, keys, visits, st, rng=rng)))
    # Thresholds are deliberately below the measured ratios (benchmarks/bench_sample_size.py
    # reports the real numbers over 200 states) so the test pins the ORDER of magnitude and
    # does not become a tripwire for a one-feature change.
    assert np.mean(v1_bytes) / np.mean(flat_bytes) > 10.0
    assert np.mean(v1_bytes) / np.mean(set_bytes) > 6.0
    # at a big SELECTING_HAND leaf, which is what actually blew up the buffer
    biggest = int(np.argmax(v1_bytes))
    assert v1_bytes[biggest] / flat_bytes[biggest] > 18.0
    assert v1_bytes[biggest] / set_bytes[biggest] > 10.0


# ── buffer ──────────────────────────────────────────────────────────────────────

def test_buffer_round_trips_v1_and_v2(states):
    enc = get_encoder("set")
    rng = np.random.default_rng(7)
    buf = ReplayBuffer(capacity=100)
    buf.add(SampleV1(obs=np.arange(5, dtype=np.float32),
                     action_features=np.zeros((3, 56), dtype=np.float32),
                     target_policy=np.array([0.2, 0.3, 0.5], dtype=np.float32), z=0.4))
    for game in states[:5]:
        legal = game.legal_actions()
        if not legal:
            continue
        keys = [action_key(a) for a in legal]
        buf.add(make_sample_v2(game, legal, keys, _visits(keys, rng), enc, z=0.7, rng=rng))

    restored = ReplayBuffer(capacity=100)
    restored.load_state_dict(buf.state_dict())
    assert len(restored) == len(buf)
    for a, b in zip(list(buf._buf), list(restored._buf)):
        assert getattr(a, "version", 1) == getattr(b, "version", 1)
        assert np.allclose(a.target_policy, b.target_policy)
        assert a.z == b.z
        if getattr(a, "version", 1) >= 2:
            assert a.meta == b.meta
            for k in a.obs:
                assert np.array_equal(a.obs[k], b.obs[k])
            for k in a.actions:
                assert np.array_equal(a.actions[k], b.actions[k])
        else:
            assert np.array_equal(a.obs, b.obs)
            assert np.array_equal(a.action_features, b.action_features)


# ── trainer dispatch ────────────────────────────────────────────────────────────

def test_trainer_handles_v1_and_v2_flat_samples(states):
    from mcts.action_features import featurize_actions
    from mcts.model import PolicyValueNet
    torch.manual_seed(0)
    v7 = get_encoder("v7")
    net = PolicyValueNet(obs_dim=v7.dim)
    trainer = Trainer(net, lr=1e-3, device="cpu")
    rng = np.random.default_rng(8)
    batch = []
    for game in states[:8]:
        legal = game.legal_actions()
        if not legal:
            continue
        keys = [action_key(a) for a in legal]
        visits = _visits(keys, rng)
        batch.append(make_sample_v2(game, legal, keys, visits, v7, z=0.6, rng=rng))
        batch.append(SampleV1(
            obs=v7(game), action_features=featurize_actions(legal),
            target_policy=np.full(len(legal), 1 / len(legal), dtype=np.float32), z=0.6))
    metrics = trainer.step(batch)
    assert metrics["n"] == len(batch)
    assert np.isfinite(metrics["policy_loss"]) and np.isfinite(metrics["value_loss"])


def test_trainer_handles_set_samples(states):
    from mcts.model_set import SetPolicyValueNet
    torch.manual_seed(0)
    enc = get_encoder("set")
    net = SetPolicyValueNet(caps=enc.caps)
    trainer = Trainer(net, lr=1e-3, device="cpu")
    rng = np.random.default_rng(9)
    batch = []
    for game in states[:12]:
        legal = game.legal_actions()
        if not legal:
            continue
        keys = [action_key(a) for a in legal]
        batch.append(make_sample_v2(game, legal, keys, _visits(keys, rng), enc,
                                    z=0.3, rng=rng))
    before = torch.cat([p.detach().flatten().clone() for p in net.parameters()])
    metrics = trainer.step(batch)
    after = torch.cat([p.detach().flatten() for p in net.parameters()])
    assert np.isfinite(metrics["policy_loss"]) and np.isfinite(metrics["value_loss"])
    assert not torch.equal(before, after), "the set net did not move"


def test_set_policy_loss_is_the_masked_subsample_softmax(states):
    """Padded rows must contribute nothing — a hand-rolled per-sample loss must equal the
    padded batched one."""
    import torch.nn.functional as F
    from mcts.model_set import SetPolicyValueNet
    from mcts.policy_set import pad_acts, stack_obs
    torch.manual_seed(0)
    enc = get_encoder("set")
    net = SetPolicyValueNet(caps=enc.caps).eval()
    trainer = Trainer(net, lr=1e-3, device="cpu")
    rng = np.random.default_rng(10)
    batch = []
    for game in states[:6]:
        legal = game.legal_actions()
        if len(legal) < 2:
            continue
        keys = [action_key(a) for a in legal]
        batch.append(make_sample_v2(game, legal, keys, _visits(keys, rng), enc,
                                    z=0.42, rng=rng, k_unvisited=int(rng.integers(1, 9))))
    assert len({s.k for s in batch}) > 1, "need ragged k for this test to bite"

    with torch.no_grad():
        p_batched, v_batched = trainer._set_losses(batch)
        p_manual = torch.zeros(())
        v_manual = torch.zeros(())
        for s in batch:
            obs = stack_obs([s.obs], "cpu")
            acts, _ = pad_acts([s.actions], [s.k], "cpu")
            logits, value = net(obs, acts)
            lp = F.log_softmax(logits[0], dim=-1)
            p_manual = p_manual + -(torch.from_numpy(s.target_policy) * lp).sum()
            v_manual = v_manual + (value[0] - s.z) ** 2
    assert abs(float(p_batched) - float(p_manual)) < 1e-3
    assert abs(float(v_batched) - float(v_manual)) < 1e-4


# ── checkpoints ─────────────────────────────────────────────────────────────────

def set_cfg(**over) -> TrainConfig:
    base = dict(seed=7, sims=6, max_considered=4, batch_size=6, min_buffer=6,
                max_decisions=40, max_antes=1, device="cpu", buffer_capacity=2_000,
                encoder="set", ruleset="mlb")
    base.update(over)
    return TrainConfig(**base)


def _weights(trainer: ColdTrainer) -> torch.Tensor:
    return torch.cat([p.detach().flatten().cpu().clone() for p in trainer.net.parameters()])


def _moments(trainer: ColdTrainer) -> list[torch.Tensor]:
    out = []
    for st in trainer.trainer.optimizer.state_dict()["state"].values():
        for k in ("exp_avg", "exp_avg_sq"):
            if k in st:
                out.append(st[k].detach().flatten().cpu().clone())
    return out


def test_set_encoder_checkpoint_round_trip_is_bit_exact(tmp_path: Path):
    reference = ColdTrainer(set_cfg())
    for _ in range(4):
        reference.run_episode()

    interrupted = ColdTrainer(set_cfg())
    for _ in range(3):
        interrupted.run_episode()
    path = save_checkpoint(tmp_path / "set.pt", interrupted.state_dict())

    ckpt = load_checkpoint(path)
    assert ckpt["net_kind"] == "set"
    assert ckpt["encoder_caps"] == ItemCaps().as_dict()
    assert ckpt["net_desc"]["kind"] == "set"

    resumed = ColdTrainer.from_checkpoint(ckpt)
    assert resumed.counters.episodes == 3
    resumed.run_episode()

    assert torch.equal(_weights(reference), _weights(resumed)), "weights diverged"
    ref_m, res_m = _moments(reference), _moments(resumed)
    assert len(ref_m) == len(res_m) and ref_m
    for a, b in zip(ref_m, res_m):
        assert torch.equal(a, b), "Adam moments diverged"
    assert reference.counters.as_dict() == resumed.counters.as_dict()
    assert len(reference.buffer) == len(resumed.buffer)


def test_resume_refuses_a_different_net_kind(tmp_path: Path):
    trainer = ColdTrainer(set_cfg())
    trainer.run_episode()
    path = save_checkpoint(tmp_path / "set.pt", trainer.state_dict())
    ckpt = load_checkpoint(path)
    flat = ColdTrainer(set_cfg(encoder="mlb"))
    with pytest.raises(ValueError, match="encoder"):
        flat.load_state_dict(ckpt)


def test_resume_refuses_different_caps(tmp_path: Path):
    trainer = ColdTrainer(set_cfg())
    trainer.run_episode()
    ckpt = load_checkpoint(save_checkpoint(tmp_path / "set.pt", trainer.state_dict()))
    ckpt["encoder_caps"] = {"hand": 10, "jokers": 12, "consumables": 6,
                            "shelf": 8, "packs": 8}
    other = ColdTrainer(set_cfg())
    with pytest.raises(ValueError, match="caps"):
        other.load_state_dict(ckpt)


def test_set_checkpoints_are_much_smaller(tmp_path: Path):
    """The 20x sample shrink is what makes `latest.pt` shippable — Phase 3's was 137 MB."""
    trainer = ColdTrainer(set_cfg())
    for _ in range(3):
        trainer.run_episode()
    path = save_checkpoint(tmp_path / "set.pt", trainer.state_dict())
    mb = path.stat().st_size / 1e6
    assert mb < 40.0, f"latest.pt is {mb:.1f} MB"
    assert len(trainer.buffer) > 20


def test_load_policy_rebuilds_a_set_net(tmp_path: Path):
    from mcts.player import load_policy
    from mcts.policy_set import BatchedSetNNPolicy
    trainer = ColdTrainer(set_cfg())
    trainer.run_episode()
    path = save_checkpoint(tmp_path / "set.pt", trainer.state_dict())
    policy = load_policy(str(path), device="cpu", batched=True)
    assert isinstance(policy, BatchedSetNNPolicy)
    assert policy.encoder.caps == ItemCaps()
    game = BalatroGame(seed=42, deck_key="b_red", stake=1, ruleset="mlb")
    priors, value = policy(game)
    assert priors and np.isfinite(value)
    # same weights as the trainer's net
    for a, b in zip(trainer.net.state_dict().values(), policy.model.state_dict().values()):
        assert torch.equal(a.cpu(), b.cpu())


def test_a_phase_3_version_1_checkpoint_still_resumes(tmp_path: Path):
    """The Phase 3 runs under `agent/runs/` (including the overnight shakedown) must stay
    loadable: `load_checkpoint` accepts version 1, `net_kind` reads as "flat", and the
    resumed run collects v2 samples into the same buffer as the v1 ones it restored."""
    from train.checkpoint import CHECKPOINT_KIND
    flat = TrainConfig(seed=7, sims=6, max_considered=4, batch_size=6, min_buffer=6,
                       max_decisions=40, max_antes=1, device="cpu", encoder="mlb",
                       ruleset="mlb", subsample=False)
    trainer = ColdTrainer(flat)
    for _ in range(2):
        trainer.run_episode()
    assert all(getattr(s, "version", 1) == 1 for s in list(trainer.buffer._buf))

    # Downgrade the payload to exactly what Phase 3 wrote.
    sd = trainer.state_dict()
    for key in ("net_kind", "encoder_caps"):
        sd.pop(key)
    for key in ("subsample", "k_unvisited", "set_res_blocks"):
        sd["config"].pop(key, None)
    sd["version"] = 1
    sd["kind"] = CHECKPOINT_KIND
    path = save_checkpoint(tmp_path / "v1.pt", sd)

    ckpt = load_checkpoint(path)
    assert ckpt["version"] == 1 and "net_kind" not in ckpt
    resumed = ColdTrainer.from_checkpoint(ckpt)
    assert resumed.counters.episodes == 2
    assert resumed.cfg.subsample is True          # the new default applies
    n_before = len(resumed.buffer)
    resumed.run_episode()
    versions = {getattr(s, "version", 1) for s in list(resumed.buffer._buf)}
    assert versions == {1, 2}, versions
    assert len(resumed.buffer) > n_before
    # a mixed batch trains
    metrics = resumed.trainer.step(list(resumed.buffer._buf)[:8])
    assert np.isfinite(metrics["policy_loss"])
