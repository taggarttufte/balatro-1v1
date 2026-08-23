"""test_train_v.py — the V trainer (W5): metrics, loop, PAUSE, checkpoints, bit-exact
resume.  Runs on the dummy model (CPU, seconds); the real SetValueNet gets one smoke test
that skips until W1's net is importable."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

import _bootstrap  # noqa: F401

import dataset as DS
import train_v as TV


# ── synthetic shards: lives difference -> P(win), learnable by the dummy net ──

def _rows(seed: str, n: int, rng: np.random.Generator) -> list:
    out = []
    for i in range(n):
        x = np.zeros(16, np.float32)
        d = rng.integers(-3, 4)
        x[3] = d / 4.0
        x[0] = rng.random()
        x[15] = 1.0
        y = float(np.clip(0.5 + 0.15 * d + rng.normal(0, 0.05), 0, 1))
        out.append(DS.LabelRow({"x": x}, y, {"seed": seed, "step": i, "player": i % 2,
                                            "kind": ["hand", "shop", "nemesis"][i % 3], "ante": 1 + i % 5,
                                            "ci": 0.2, "n_rollouts": 8, "trunc_frac": 0.0}))
    return out


@pytest.fixture(scope="module")
def shards(tmp_path_factory):
    d = tmp_path_factory.mktemp("shards")
    rng = np.random.default_rng(0)
    seeds = [f"S{i:07d}" for i in range(12)]
    for k in range(3):
        rows = []
        for s in seeds[k * 4:(k + 1) * 4]:
            rows += _rows(s, 60, rng)
        DS.save_shard(d / f"shard_{k:04d}.npz", rows)
    return d


def _cfg(shards, run_dir, **kw) -> TV.TrainVConfig:
    base = dict(shards=[str(shards)], run_dir=str(run_dir), model="dummy", batch_size=32, lr=3e-3,
                warmup_steps=5, max_steps=60, eval_every=20, checkpoint_every=20, device="cpu",
                holdout_frac=0.25, keep=10, seed=1)
    base.update(kw)
    return TV.TrainVConfig(**base)


# ── metrics ──

def test_auc_and_reliability_and_metrics():
    p = np.array([0.1, 0.4, 0.35, 0.8, 0.9, 0.6])
    y = np.array([0, 0, 1, 1, 1, 0], dtype=np.float32)
    auc = TV.auc_score(p, y)
    assert 0.0 <= auc <= 1.0
    assert TV.auc_score(np.array([0.9, 0.1]), np.array([1.0, 0.0])) == 1.0
    assert TV.auc_score(np.array([0.1, 0.9]), np.array([1.0, 0.0])) == 0.0
    assert np.isnan(TV.auc_score(p, np.ones_like(y)))
    rc = TV.reliability_curve(p, y, n_bins=10)
    assert len(rc["bins"]) == 10 and sum(b["n"] for b in rc["bins"]) == len(p)
    assert 0.0 <= rc["ece"] <= 1.0
    m = TV.metrics(p, y, ci=np.full(6, 0.196), const=0.5)
    assert m["const"]["bce"] == pytest.approx(float(np.log(2)))
    assert m["noise_floor_brier"] == pytest.approx(0.01)
    assert m["brier"] < m["const"]["brier"]
    json.dumps(m)


# ── data + loop ──

def test_dataset_split_and_batches(shards):
    ds = DS.LabelDataset.load(shards)
    assert len(ds) == 720 and len(ds.seeds()) == 12
    tr, ho = ds.split_by_seed(0.25)
    assert set(tr.seeds()).isdisjoint(ho.seeds())
    assert len(tr) + len(ho) == 720 and len(ho) > 0


def test_training_beats_constant_and_writes_run_artifacts(shards, tmp_path):
    run = tmp_path / "run"
    summ = TV.run(_cfg(shards, run, max_steps=150, eval_every=50, checkpoint_every=50), log=lambda *_: None)
    assert summ["stop_reason"] == "max_steps" and summ["done"]
    fe = summ["final_eval"]
    assert fe["bce"] < fe["const"]["bce"] - 0.02
    assert fe["brier"] < fe["const"]["brier"]
    assert fe["auc"] > 0.8
    assert (run / ".DONE").exists() and (run / "latest.pt").exists()
    assert len(list(run.glob("ckpt_*.pt"))) >= 1
    recs = [json.loads(l) for l in (run / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    kinds = [r["kind"] for r in recs]
    assert kinds[0] == "config" and kinds[-1] == "summary"
    assert "eval" in kinds and "checkpoint" in kinds
    ev = [r for r in recs if r["kind"] == "eval"][-1]
    assert len(ev["reliability"]["bins"]) == 10 and "by_kind" in ev


def test_pause_file_stops_and_resume_clears_it(shards, tmp_path):
    run = tmp_path / "run"
    cfg = _cfg(shards, run, max_steps=100, eval_every=1000, checkpoint_every=1000)
    pause = run / TV.PAUSE_FILE
    run.mkdir()
    hit = {"n": 0}

    def stop_check():
        hit["n"] += 1
        if hit["n"] == 8:
            pause.write_text("")
        return False

    summ = TV.run(cfg, log=lambda *_: None, stop_check=stop_check)
    assert summ["stop_reason"] == "PAUSE" and not summ["done"]
    assert 5 <= summ["step"] <= 10
    assert not (run / TV.DONE_FILE).exists() and pause.exists()
    summ2 = TV.run(resume=str(run / "latest.pt"), overrides={"max_steps": 100}, log=lambda *_: None)
    assert not pause.exists()
    assert summ2["stop_reason"] == "max_steps" and summ2["step"] == 100 and (run / TV.DONE_FILE).exists()


def test_resume_is_a_bit_exact_continuation(shards, tmp_path):
    run = tmp_path / "run"
    cfg = _cfg(shards, run, max_steps=60, eval_every=30, checkpoint_every=30)
    TV.run(cfg, log=lambda *_: None)
    netA, _, extraA = TV.VTrainer.read_checkpoint(run / "latest.pt")
    sdA = {k: v.clone() for k, v in netA.state_dict().items()}
    optA = extraA["trainer"]["optimizer"]
    assert extraA["trainer"]["step"] == 60
    # continue from the step-30 checkpoint to 60 with the same schedule
    TV.run(resume=str(run / "ckpt_0000030.pt"), overrides={"max_steps": 60}, log=lambda *_: None)
    netB, _, extraB = TV.VTrainer.read_checkpoint(run / "latest.pt")
    assert extraB["trainer"]["step"] == 60
    assert all(torch.equal(sdA[k], v) for k, v in netB.state_dict().items())
    optB = extraB["trainer"]["optimizer"]
    for sa, sb in zip(optA["state"].values(), optB["state"].values()):
        for a, b in zip(sa.values(), sb.values()):
            if isinstance(a, torch.Tensor):
                assert torch.equal(a, b)
    # the eval history is carried across
    assert [h["step"] for h in extraB["trainer"]["history"]] == [0, 30, 60]


def test_cli_parses_and_runs(shards, tmp_path):
    run = tmp_path / "cli"
    rc = TV.main(["--shards", str(shards), "--run-dir", str(run), "--model", "dummy", "--max-steps", "10",
                  "--eval-every", "5", "--checkpoint-every", "5", "--device", "cpu", "--batch-size", "16",
                  "--warmup-steps", "2", "--holdout-frac", "0.25"])
    assert rc == 0 and (run / ".DONE").exists()
    rc = TV.main(["--resume", str(run), "--max-steps", "14"])
    assert rc == 0
    _, _, extra = TV.VTrainer.read_checkpoint(run / "latest.pt")
    assert extra["trainer"]["step"] == 14
    assert extra["trainer"]["config"]["batch_size"] == 16          # untouched by the resume


def test_lr_schedule_shapes():
    cfg = TV.TrainVConfig(model="dummy", lr=1.0, warmup_steps=10, max_steps=110, min_lr_frac=0.1)
    tr = TV.VTrainer.__new__(TV.VTrainer)
    tr.cfg = cfg
    assert tr.lr_at(0) == pytest.approx(0.1)
    assert tr.lr_at(9) == pytest.approx(1.0)
    assert tr.lr_at(10) == pytest.approx(1.0)
    assert tr.lr_at(60) == pytest.approx(0.55)
    assert tr.lr_at(110) == pytest.approx(0.1)
    cfg.lr_schedule = "flat"
    assert tr.lr_at(500) == 1.0


# ── the real net (W1) ──

@pytest.mark.skipif(not __import__("labels").has_encoder_v2(), reason="W1 encoder_v2/value_net not landed")
def test_set_value_net_round_trip_and_one_step(tmp_path):
    pytest.importorskip("mcts.value_net")
    import labels as L
    enc = L.make_encoder("v2")
    snaps = L.sample_states("7I4M53DL", n_states=4, policy_factory=L.make_policy_factory("scripted", epsilon=0.1),
                            policy="scripted")
    rows = [DS.LabelRow(enc(s.match, p), 0.5 + 0.1 * p, {"seed": f"S{i}", "step": s.step, "player": p,
                                                         "kind": s.kind, "ante": s.ante, "ci": 0.3})
            for i, s in enumerate(snaps) for p in (0, 1)]
    DS.save_shard(tmp_path / "s.npz", rows)
    ds = DS.LabelDataset.load(tmp_path)
    tr_ds, ho_ds = ds.split_by_seed(holdout_seeds=["S0"])
    small = {"d_item": 32, "trunk_width": 64, "n_res_blocks": 1, "scalar_hidden": 32, "key_emb": 8}
    cfg = TV.TrainVConfig(shards=[str(tmp_path)], run_dir=str(tmp_path / "run"), model="set_value_net",
                          net_cfg=small, batch_size=4, max_steps=2, eval_every=1, checkpoint_every=1,
                          device="cpu", warmup_steps=1, keep=5)
    summ = TV.run(cfg, data=(tr_ds, ho_ds), log=lambda *_: None)
    assert summ["step"] == 2
    from mcts.value_net import load_checkpoint
    net, encoder, extra = load_checkpoint(tmp_path / "run" / "latest.pt")
    assert extra["trainer"]["step"] == 2 and "optimizer" in extra["trainer"]
    tr = TV.VTrainer.from_checkpoint(tmp_path / "run" / "latest.pt", data=(tr_ds, ho_ds), log=lambda *_: None)
    assert all(torch.equal(a, b) for a, b in zip(net.state_dict().values(), tr.net.state_dict().values()))
