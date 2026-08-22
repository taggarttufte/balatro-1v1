"""
test_checkpoint.py — the checkpoint round-trip the Phase 3 brief asks for.

The headline test trains 3 episodes, saves, rebuilds a trainer from the checkpoint,
trains 1 more, and compares against an uninterrupted 4-episode run.

**The comparison is bit-exact** (`torch.equal` on every parameter and on Adam's moments),
not just "same sample count, finite losses". That is possible because on CPU the whole
loop is a deterministic function of the config: one seeded `numpy.random.Generator`
drives the episode seed, the Gumbel noise and the replay-batch indices; the engine is
deterministic given its seed; Adam is deterministic; and the checkpoint carries the
replay buffer, so the resumed run draws the same mini-batches.

On CUDA the kernels are not bit-reproducible by default, so these tests pin `device=cpu`
and AGENT_NOTES.md records the caveat.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from train import (
    CHECKPOINT_KIND, ColdTrainer, TrainConfig, latest_checkpoint,
    load_checkpoint, save_checkpoint,
)


def cfg(**over) -> TrainConfig:
    base = dict(seed=7, sims=8, max_considered=4, batch_size=8, min_buffer=8,
                max_decisions=80, device="cpu", buffer_capacity=5_000)
    base.update(over)
    return TrainConfig(**base)


def weights(trainer: ColdTrainer) -> torch.Tensor:
    return torch.cat([p.detach().flatten().cpu().clone() for p in trainer.net.parameters()])


def adam_moments(trainer: ColdTrainer) -> list[torch.Tensor]:
    out = []
    for st in trainer.trainer.optimizer.state_dict()["state"].values():
        for k in ("exp_avg", "exp_avg_sq"):
            if k in st:
                out.append(st[k].detach().flatten().cpu().clone())
    return out


# ── The round trip ──────────────────────────────────────────────────────────

def test_train_3_save_resume_1_equals_train_4(tmp_path: Path):
    reference = ColdTrainer(cfg())
    for _ in range(4):
        reference.run_episode()

    interrupted = ColdTrainer(cfg())
    for _ in range(3):
        interrupted.run_episode()
    ckpt_path = save_checkpoint(tmp_path / "ckpt.pt", interrupted.state_dict())

    resumed = ColdTrainer.from_checkpoint(load_checkpoint(ckpt_path))
    assert resumed.counters.episodes == 3
    resumed.run_episode()

    # Bit-exact: weights, optimizer moments, counters, buffer.
    assert torch.equal(weights(reference), weights(resumed)), "weights diverged after resume"
    ref_m, res_m = adam_moments(reference), adam_moments(resumed)
    assert len(ref_m) == len(res_m) and ref_m
    for a, b in zip(ref_m, res_m):
        assert torch.equal(a, b), "Adam moments diverged after resume"
    assert reference.counters.as_dict() == resumed.counters.as_dict()
    assert len(reference.buffer) == len(resumed.buffer)


def test_resume_continues_the_same_episode_seeds(tmp_path: Path):
    """The 4th episode of a resumed run must be the 4th episode of the original run —
    same game seed, same trajectory length."""
    reference = ColdTrainer(cfg())
    recs = [reference.run_episode() for _ in range(4)]

    interrupted = ColdTrainer(cfg())
    for _ in range(3):
        interrupted.run_episode()
    p = save_checkpoint(tmp_path / "c.pt", interrupted.state_dict())
    resumed = ColdTrainer.from_checkpoint(load_checkpoint(p))
    rec4 = resumed.run_episode()

    assert rec4["seed"] == recs[3]["seed"]
    assert rec4["len"] == recs[3]["len"]
    assert rec4["ante"] == recs[3]["ante"]


def test_checkpoint_without_buffer_still_loads(tmp_path: Path):
    """--no-checkpoint-buffer: smaller files, resume is no longer bit-exact but must
    still restore weights, optimizer and counters and keep training."""
    t = ColdTrainer(cfg(checkpoint_buffer=False))
    for _ in range(3):
        t.run_episode()
    p = save_checkpoint(tmp_path / "nb.pt", t.state_dict())
    ck = load_checkpoint(p)
    assert ck["buffer"] is None

    r = ColdTrainer.from_checkpoint(ck)
    assert torch.equal(weights(t), weights(r))
    assert r.counters.episodes == 3
    assert len(r.buffer) == 0
    rec = r.run_episode()
    assert rec["kind"] == "episode"


def test_checkpoint_files_are_small_enough_to_keep(tmp_path: Path):
    """A capped buffer keeps the checkpoint bounded; the cap is recorded as truncation
    so a resume can say it is no longer bit-exact."""
    t = ColdTrainer(cfg(buffer_checkpoint_cap=5))
    for _ in range(2):
        t.run_episode()
    sd = t.state_dict()
    assert sd["buffer"]["truncated"] is (len(t.buffer) > 5)
    assert len(sd["buffer"]["samples"]) <= 5


# ── Contents / metadata ─────────────────────────────────────────────────────

def test_checkpoint_carries_everything_the_brief_asks_for(tmp_path: Path):
    t = ColdTrainer(cfg())
    t.run_episode()
    sd = t.state_dict()
    for key in ("config", "model", "trainer", "counters", "rng", "buffer", "net_desc"):
        assert key in sd, key
    assert {"numpy", "torch", "python"} <= set(sd["rng"])
    assert sd["trainer"]["optimizer"]["param_groups"]
    p = save_checkpoint(tmp_path / "c.pt", sd)
    ck = load_checkpoint(p)
    assert ck["kind"] == CHECKPOINT_KIND and ck["version"] == 1 and ck["saved_at"]


def test_load_rejects_a_foreign_file(tmp_path: Path):
    p = tmp_path / "not_ours.pt"
    torch.save({"hello": "world"}, p)
    with pytest.raises(ValueError):
        load_checkpoint(p)


def test_resume_refuses_a_config_that_changes_the_experiment(tmp_path: Path):
    t = ColdTrainer(cfg())
    t.run_episode()
    p = save_checkpoint(tmp_path / "c.pt", t.state_dict())
    ck = load_checkpoint(p)
    with pytest.raises(ValueError, match="ruleset"):
        ColdTrainer.from_checkpoint(ck, overrides={"ruleset": "mlb"})
    with pytest.raises(ValueError, match="encoder"):
        ColdTrainer.from_checkpoint(ck, overrides={"encoder": "mlb"})
    # device is allowed to change
    ColdTrainer.from_checkpoint(ck, overrides={"device": "cpu"})


def test_save_is_atomic(tmp_path: Path):
    """A crash mid-write must not destroy the previous checkpoint: we write a temp file
    and os.replace it, so no .tmp file survives a successful save."""
    t = ColdTrainer(cfg())
    p = save_checkpoint(tmp_path / "c.pt", t.state_dict())
    assert p.is_file()
    assert not list(tmp_path.glob("*.tmp"))


def test_latest_checkpoint_finds_the_newest(tmp_path: Path):
    t = ColdTrainer(cfg())
    save_checkpoint(tmp_path / "ckpt_000001.pt", t.state_dict())
    t.run_episode()
    newest = save_checkpoint(tmp_path / "ckpt_000002.pt", t.state_dict())
    assert latest_checkpoint(tmp_path) == newest
    assert latest_checkpoint(tmp_path / "nothing_here") is None


# ── MLB config round-trips too ──────────────────────────────────────────────

def test_mlb_run_checkpoints_and_resumes(tmp_path: Path):
    c = cfg(ruleset="mlb", encoder="mlb", max_antes=2, max_decisions=120)
    reference = ColdTrainer(c)
    for _ in range(3):
        reference.run_episode()

    t = ColdTrainer(c)
    for _ in range(2):
        t.run_episode()
    p = save_checkpoint(tmp_path / "mlb.pt", t.state_dict())
    r = ColdTrainer.from_checkpoint(load_checkpoint(p))
    assert r.encoder.name == "mlb" and r.net.obs_dim == 453
    r.run_episode()
    assert torch.equal(weights(reference), weights(r))
