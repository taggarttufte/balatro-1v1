"""
test_followups.py — the three lead-requested follow-ups before the first long run
(Phase 4 W1, 2026-08-22).

  1. the categorical embedding tables are merged behind offsets (fewer kernel launches),
     and the merge is EQUIVALENT to per-field tables — each field still owns its own rows;
  2. both value heads are bounded to the OutcomeFn's [0, 1] range, with a checkpoint that
     predates the change rebuilt with its original unbounded semantics;
  3. `train_cold.py --ruleset mlb` refuses, because that objective is degenerate.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from mcts.encoder import get_encoder
from mcts.encoder_set import CARD_CARDINALITIES
from mcts.model import PolicyValueNet, VALUE_ACTIVATIONS, apply_value_activation
from mcts.model_set import SetPolicyValueNet, _OffsetEmbedding
from train import ColdTrainer, TrainConfig, load_checkpoint, save_checkpoint

AGENT_ROOT = Path(__file__).resolve().parents[1]


# ── 1. merged embedding tables ──────────────────────────────────────────────────

def test_offset_embedding_gives_each_field_its_own_rows():
    """One table + offsets must be exactly as expressive as F separate tables: field i's
    value v addresses row offset[i] + v, and no two (field, value) pairs collide."""
    torch.manual_seed(0)
    cards = (14, 5, 10, 6, 6)
    emb = _OffsetEmbedding(cards, dim=4)
    assert emb.table.num_embeddings == sum(cards)
    assert emb.out_dim == 4 * len(cards)
    assert emb.offsets.tolist() == [0, 14, 19, 29, 35]

    seen = set()
    for f, card in enumerate(cards):
        for v in range(card):
            row = int(emb.offsets[f]) + v
            assert row not in seen, "two (field, value) pairs share a row"
            seen.add(row)
    assert len(seen) == sum(cards)


def test_offset_embedding_matches_separate_tables():
    """Gather-with-offsets == concat of per-field gathers, for the same weights."""
    torch.manual_seed(1)
    cards = tuple(CARD_CARDINALITIES)
    dim = 9
    merged = _OffsetEmbedding(cards, dim)
    separate = [torch.nn.Embedding(c, dim) for c in cards]
    with torch.no_grad():
        at = 0
        for c, e in zip(cards, separate):
            e.weight.copy_(merged.table.weight[at:at + c])
            at += c

    cat = torch.stack([torch.randint(0, c, (3, 7)) for c in cards], dim=-1).to(torch.int16)
    got = merged(cat)
    want = torch.cat([e(cat[..., i].long()) for i, e in enumerate(separate)], dim=-1)
    assert torch.allclose(got, want, atol=1e-6)


def test_set_net_has_three_categorical_tables_not_eleven():
    """The whole point of the merge: one card table, one aux table, one key table."""
    net = SetPolicyValueNet()
    tables = [name for name, m in net.named_modules() if isinstance(m, torch.nn.Embedding)]
    # card.emb.table, aux_table, key_embed, set_embed, act_type_embed
    assert sorted(tables) == ["act_type_embed", "aux_table", "card.emb.table",
                              "key_embed", "set_embed"], tables
    for gone in ("edition_embed", "rarity_embed", "shelf_kind_embed", "pack_set_embed"):
        assert not hasattr(net, gone), gone


def test_set_net_still_round_trips_and_is_finite():
    from balatro_sim.game import BalatroGame
    from mcts.action_features_set import featurize_actions_set
    torch.manual_seed(0)
    enc = get_encoder("set")
    net = SetPolicyValueNet(caps=enc.caps).eval()
    rebuilt = SetPolicyValueNet.from_description(net.describe())
    assert rebuilt.describe() == net.describe()
    rebuilt.load_state_dict(net.state_dict())
    assert net.n_params < 3_000_000

    game = BalatroGame(seed=42, deck_key="b_red", stake=1, ruleset="mlb")
    game.step({"type": "play_blind"})
    legal = game.legal_actions()
    obs = {k: torch.from_numpy(v).unsqueeze(0) for k, v in enc(game).items()}
    acts = {k: torch.from_numpy(v).unsqueeze(0)
            for k, v in featurize_actions_set(game, legal, enc.caps).items()}
    with torch.no_grad():
        logits, value = net(obs, acts)
    assert logits.shape == (1, len(legal))
    assert torch.isfinite(logits).all() and torch.isfinite(value).all()


# ── 2. bounded value heads ──────────────────────────────────────────────────────

def test_apply_value_activation():
    x = torch.tensor([-5.0, 0.0, 5.0])
    assert torch.allclose(apply_value_activation(x, "linear"), x)
    assert torch.allclose(apply_value_activation(x, "clamp"),
                          torch.tensor([0.0, 0.0, 1.0]))
    sig = apply_value_activation(x, "sigmoid")
    assert bool(((sig > 0) & (sig < 1)).all())
    assert set(VALUE_ACTIVATIONS) == {"sigmoid", "clamp", "linear"}


@pytest.mark.parametrize("kind", ["flat", "set"])
def test_value_head_is_in_the_outcome_range_at_init(kind):
    """Every OutcomeFn returns a value in [0, 1]; an unbounded head starts outside it."""
    torch.manual_seed(0)
    if kind == "flat":
        net = PolicyValueNet(obs_dim=453).eval()
        trunk = net.get_trunk(torch.randn(64, 453))
        v = net.value(trunk)
    else:
        enc = get_encoder("set")
        net = SetPolicyValueNet(caps=enc.caps).eval()
        from _states import collect_states
        obs_list = [enc(g) for g in collect_states(64)]
        obs = {k: torch.from_numpy(np.stack([o[k] for o in obs_list]))
               for k in obs_list[0]}
        v = net.value(net.encode_state(obs))
    assert net.value_activation == "sigmoid"
    assert float(v.min()) >= 0.0 and float(v.max()) <= 1.0


@pytest.mark.parametrize("kind", ["flat", "set"])
def test_linear_is_still_reachable_and_unbounded(kind):
    torch.manual_seed(0)
    net = (PolicyValueNet(obs_dim=453, value_activation="linear") if kind == "flat"
           else SetPolicyValueNet(value_activation="linear"))
    assert net.value_activation == "linear"
    assert net.describe()["value_activation"] == "linear"


@pytest.mark.parametrize("cls,kwargs", [(PolicyValueNet, {"obs_dim": 453}),
                                        (SetPolicyValueNet, {})])
def test_a_description_without_the_field_rebuilds_as_linear(cls, kwargs):
    """A checkpoint written before the bounded head was trained unbounded, so it must be
    rebuilt unbounded — not silently reinterpreted through a sigmoid."""
    net = cls(**kwargs)
    desc = net.describe()
    assert desc["value_activation"] == "sigmoid"
    desc.pop("value_activation")
    assert cls.from_description(desc).value_activation == "linear"


def test_unknown_value_activation_is_refused():
    with pytest.raises(ValueError, match="value_activation"):
        PolicyValueNet(obs_dim=453, value_activation="relu")
    with pytest.raises(ValueError, match="value_activation"):
        SetPolicyValueNet(value_activation="relu")


def test_value_activation_is_pinned_across_a_resume(tmp_path: Path):
    cfg = dict(seed=7, sims=6, max_considered=4, batch_size=6, min_buffer=6,
               max_decisions=40, device="cpu", buffer_capacity=1_000)
    trainer = ColdTrainer(TrainConfig(**cfg))
    trainer.run_episode()
    assert trainer.net.value_activation == "sigmoid"
    ckpt = load_checkpoint(save_checkpoint(tmp_path / "c.pt", trainer.state_dict()))
    assert ckpt["net_desc"]["value_activation"] == "sigmoid"

    other = ColdTrainer(TrainConfig(**cfg, value_activation="linear"))
    with pytest.raises(ValueError, match="value_activation"):
        other.load_state_dict(ckpt)


def test_a_pre_bounded_head_checkpoint_resumes_as_linear(tmp_path: Path):
    cfg = dict(seed=7, sims=6, max_considered=4, batch_size=6, min_buffer=6,
               max_decisions=40, device="cpu", buffer_capacity=1_000,
               value_activation="linear")
    trainer = ColdTrainer(TrainConfig(**cfg))
    trainer.run_episode()
    sd = trainer.state_dict()
    sd["config"].pop("value_activation")          # what a pre-follow-up run wrote
    sd["net_desc"].pop("value_activation")
    ckpt = load_checkpoint(save_checkpoint(tmp_path / "old.pt", sd))

    resumed = ColdTrainer.from_checkpoint(ckpt)
    assert resumed.cfg.value_activation == "linear"
    assert resumed.net.value_activation == "linear"
    resumed.run_episode()                          # and it still trains


def test_checkpoint_round_trip_stays_bit_exact_with_the_bounded_head(tmp_path: Path):
    """The headline Phase 3 property, re-asserted after the head changed."""
    def cfg():
        return TrainConfig(seed=7, sims=8, max_considered=4, batch_size=8, min_buffer=8,
                           max_decisions=80, device="cpu", buffer_capacity=5_000)

    reference = ColdTrainer(cfg())
    for _ in range(4):
        reference.run_episode()

    interrupted = ColdTrainer(cfg())
    for _ in range(3):
        interrupted.run_episode()
    ckpt = load_checkpoint(save_checkpoint(tmp_path / "c.pt", interrupted.state_dict()))
    resumed = ColdTrainer.from_checkpoint(ckpt)
    resumed.run_episode()

    a = torch.cat([p.detach().flatten() for p in reference.net.parameters()])
    b = torch.cat([p.detach().flatten() for p in resumed.net.parameters()])
    assert torch.equal(a, b)


# ── 3. train_cold refuses MLB ───────────────────────────────────────────────────

def _run_train_cold(*args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(AGENT_ROOT / "scripts" / "train_cold.py"), *args],
        capture_output=True, text=True, timeout=300)


def test_train_cold_refuses_mlb():
    r = _run_train_cold("--episodes", "1", "--ruleset", "mlb", "--device", "cpu")
    assert r.returncode != 0
    msg = r.stdout + r.stderr
    assert "degenerate" in msg
    assert "train_mlb.py" in msg and "--objective external" in msg


def test_train_cold_refuses_resuming_an_mlb_checkpoint(tmp_path: Path):
    """`--resume` takes the ruleset from the checkpoint, so the arg check alone is not
    enough — an old degenerate run must not be continuable either."""
    trainer = ColdTrainer(TrainConfig(seed=1, sims=4, max_considered=2, batch_size=4,
                                      min_buffer=4, max_decisions=20, max_antes=1,
                                      device="cpu", ruleset="mlb", buffer_capacity=100))
    trainer.run_episode()
    path = save_checkpoint(tmp_path / "mlb.pt", trainer.state_dict())
    r = _run_train_cold("--resume", str(path), "--episodes", "1", "--device", "cpu",
                        "--run-dir", str(tmp_path))
    assert r.returncode != 0
    assert "degenerate" in (r.stdout + r.stderr)


def test_the_mlb_objective_is_still_reachable_through_the_library():
    """The refusal is CLI-only: W2's MLBTrainer and the tests drive `ColdTrainer` with
    ruleset='mlb' behind a non-degenerate outcome, and must keep working."""
    trainer = ColdTrainer(TrainConfig(seed=1, sims=4, max_considered=2, batch_size=4,
                                      min_buffer=4, max_decisions=20, max_antes=1,
                                      device="cpu", ruleset="mlb", buffer_capacity=100))
    rec = trainer.run_episode()
    assert rec["kind"] == "episode"
