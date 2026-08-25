"""test_train_v_pairs.py — lever (b): the pairwise ranking loss, W-PAIRS's frozen pair
shard schema, player_fingerprint filtering, and bit-exact resume extended to pair batches
(W-RANK, Phase 5 rev 2; PHASE5_V2_BRIEF §6).

W-PAIRS's producer is not landed yet (brief §6: "code against the frozen schema... do not
wait for W-PAIRS"), so every fixture here is hand-built against the field names frozen in
PHASE5_V2_BRIEF §5.3, not real rollout output — see RANK_NOTES §2 for the on-disk format
this module chose and why.  Runs on the dummy model (CPU, seconds); one smoke test exercises
the real SetValueNet and skips until W1's net is importable, matching test_train_v.py's
existing pattern.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

import _bootstrap  # noqa: F401

import dataset as DS
import train_v as TV

SEEDS = [f"S{i:07d}" for i in range(12)]   # same seed alphabet as test_train_v.py's `shards`


# ── fixture builders ─────────────────────────────────────────────────────────────

def _abs_rows(seed: str, n: int, rng: np.random.Generator, fingerprint=None) -> list:
    out = []
    for i in range(n):
        x = np.zeros(16, np.float32)
        d = rng.integers(-3, 4)
        x[3] = d / 4.0
        x[0] = rng.random()
        x[15] = 1.0
        y = float(np.clip(0.5 + 0.15 * d + rng.normal(0, 0.05), 0, 1))
        meta = {"seed": seed, "step": i, "player": i % 2, "kind": ["hand", "shop", "nemesis"][i % 3],
                "ante": 1 + i % 5, "ci": 0.2, "n_rollouts": 8, "trunc_frac": 0.0}
        if fingerprint is not None:
            meta["player_fingerprint"] = fingerprint
        out.append(DS.LabelRow({"x": x}, y, meta))
    return out


def _pair_row(seed: str, step: int, rng: np.random.Generator, *, delta_ci: float = 0.05,
             pair_source: str = "close_call", state_kind: str = "hand",
             fingerprint: str = "fp_new_1") -> TV.PairRow:
    """A pair whose x[3] (obs_a - obs_b) has the same sign as ``delta`` — the dummy net
    already learns y from x[3] (test_train_v.py's `_rows`), so it can learn to rank these."""
    sign = 1.0 if rng.random() < 0.5 else -1.0
    mag_x, mag_delta = rng.uniform(0.3, 1.0), rng.uniform(0.1, 0.4)
    xa, xb = np.zeros(16, np.float32), np.zeros(16, np.float32)
    xa[3], xb[3] = sign * mag_x / 2.0, -sign * mag_x / 2.0
    xa[15] = xb[15] = 1.0
    delta = sign * mag_delta
    fields = {"seed": seed, "step": step, "actor": 0, "state_kind": state_kind, "ante": 2,
             "player_fingerprint": fingerprint, "pair_source": pair_source, "n_worlds": 8}
    extra = {"action_a": "play:0", "action_b": "play:1", "outcomes_a": [1.0] * 8, "outcomes_b": [0.0] * 8,
            "meta": {"note": "synthesized fixture, not a real rollout"}}
    return TV.PairRow({"x": xa}, {"x": xb}, float(delta), float(delta_ci), fields, extra)


@pytest.fixture(scope="module")
def abs_shards(tmp_path_factory):
    d = tmp_path_factory.mktemp("abs_shards")
    rng = np.random.default_rng(0)
    for k in range(3):
        rows = []
        for s in SEEDS[k * 4:(k + 1) * 4]:
            rows += _abs_rows(s, 60, rng)
        DS.save_shard(d / f"shard_{k:04d}.npz", rows)
    return d


@pytest.fixture(scope="module")
def pair_shards(tmp_path_factory):
    """12 seeds x 3 sources x 2 kinds = 72 pairs (the same seed alphabet as ``abs_shards``,
    so the seed-hash split test below is meaningful)."""
    d = tmp_path_factory.mktemp("pair_shards")
    rng = np.random.default_rng(1)
    rows = []
    for i, s in enumerate(SEEDS):
        for j, src in enumerate(("close_call", "greedy_vs_extract", "random")):
            for k, kind in enumerate(("hand", "shop")):
                rows.append(_pair_row(s, i * 10 + j * 2 + k, rng, pair_source=src, state_kind=kind))
    TV.save_pair_shard(d / "pairs_0000.npz", rows[:36])
    TV.save_pair_shard(d / "pairs_0001.npz", rows[36:])
    return d


def _cfg(abs_shards, pair_shards, run_dir, **kw) -> TV.TrainVConfig:
    base = dict(shards=[str(abs_shards)], pair_shards=[str(pair_shards)], run_dir=str(run_dir),
                model="dummy", batch_size=32, pair_batch_size=16, lr=3e-3, warmup_steps=5,
                max_steps=60, eval_every=20, checkpoint_every=20, device="cpu",
                holdout_frac=0.25, keep=10, seed=1, lam_rank=1.0, tau=0.05)
    base.update(kw)
    return TV.TrainVConfig(**base)


# ── shard I/O ─────────────────────────────────────────────────────────────────────

def test_save_load_pair_shard_round_trip(tmp_path):
    rng = np.random.default_rng(2)
    rows = [_pair_row("SXXXXXXX", i, rng) for i in range(5)]
    path = TV.save_pair_shard(tmp_path / "p.npz", rows)
    shard = TV.load_pair_shard_npz(path)
    assert len(shard) == 5
    assert set(shard.obs_a.keys()) == {"x"}
    np.testing.assert_allclose(shard.obs_a["x"][0], rows[0].obs_a["x"])
    np.testing.assert_allclose(shard.obs_b["x"][0], rows[0].obs_b["x"])
    assert shard.delta[0] == pytest.approx(rows[0].delta)
    assert shard.delta_ci[0] == pytest.approx(rows[0].delta_ci)
    assert shard.columns["pair_source"][0] == rows[0].fields["pair_source"]
    assert shard.columns["state_kind"][0] == rows[0].fields["state_kind"]
    assert shard.extra[0]["action_a"] == "play:0"
    assert shard.extra[0]["outcomes_a"] == [1.0] * 8
    with pytest.raises(ValueError):
        TV.save_pair_shard(tmp_path / "empty.npz", [])


def test_save_pair_shard_rejects_mismatched_obs_keys(tmp_path):
    r = TV.PairRow({"x": np.zeros(4, np.float32)}, {"y": np.zeros(4, np.float32)}, 0.1, 0.05)
    with pytest.raises(ValueError):
        TV.save_pair_shard(tmp_path / "bad.npz", [r])


def test_pair_dataset_load_concatenates(pair_shards):
    pds = TV.PairDataset.load(pair_shards)
    assert len(pds) == 72
    assert len(pds.seeds()) == 12
    assert set(pds.columns["pair_source"].tolist()) == {"close_call", "greedy_vs_extract", "random"}


def test_pair_dataset_empty_is_a_safe_zero():
    pds = TV.PairDataset.empty()
    assert len(pds) == 0
    tr, ho = pds.split_by_seed(0.25)
    assert len(tr) == 0 and len(ho) == 0


def test_split_by_seed_matches_absolute_dataset(abs_shards, pair_shards):
    """The seed-hash rule (dataset.seed_in_holdout, salt='v-holdout') is the SAME function
    applied to both — a seed's absolute rows and its pairs must land on the same side."""
    ads = DS.LabelDataset.load(abs_shards)
    pds = TV.PairDataset.load(pair_shards)
    _, aho = ads.split_by_seed(0.25)
    _, pho = pds.split_by_seed(0.25)
    assert set(aho.seeds()) == set(pho.seeds())
    assert 0 < len(set(aho.seeds())) < 12


def test_load_pair_records_json_jsonl_and_list(tmp_path):
    rec = {"kind": "pair", "seed": "SJSON001", "step": 3, "actor": 0, "state_kind": "hand",
           "ante": 2, "player_fingerprint": "fp_json", "pair_source": "random",
           "action_a": "discard:0", "action_b": "play:0", "n_worlds": 8,
           "outcomes_a": [1, 1, 0, 1, 0, 1, 1, 0], "outcomes_b": [0, 0, 0, 1, 0, 0, 1, 0],
           "delta": 0.25, "delta_ci": 0.09,
           "obs_a": {"x": [0.1] * 16}, "obs_b": {"x": [0.0] * 16}, "meta": {"foo": "bar"}}
    jsonl = tmp_path / "p.jsonl"
    jsonl.write_text(json.dumps(rec) + "\n" + json.dumps({**rec, "step": 4}) + "\n", encoding="utf-8")
    rows = TV.load_pair_records_json(jsonl)
    assert len(rows) == 2
    assert rows[0].fields["seed"] == "SJSON001" and rows[0].fields["pair_source"] == "random"
    assert rows[0].delta == pytest.approx(0.25) and rows[0].delta_ci == pytest.approx(0.09)
    np.testing.assert_allclose(rows[0].obs_a["x"], np.full(16, 0.1, np.float32))
    assert rows[0].extra["meta"] == {"foo": "bar"}

    single = tmp_path / "p.json"
    single.write_text(json.dumps(rec), encoding="utf-8")
    assert len(TV.load_pair_records_json(single)) == 1

    lst = tmp_path / "list.json"
    lst.write_text(json.dumps([rec, rec]), encoding="utf-8")
    assert len(TV.load_pair_records_json(lst)) == 2

    # PairDataset.load dispatches .npz first, falls back to json/jsonl in a directory
    pds = TV.PairDataset.load(tmp_path / "p.jsonl")
    assert len(pds) == 2


# ── player_fingerprint filtering (RANK_NOTES §3) ────────────────────────────────────

def test_filter_by_fingerprint_absolute_any_and_new_only():
    rng = np.random.default_rng(3)
    old_rows = _abs_rows("SOLD0001", 10, rng)                       # no player_fingerprint
    new_rows = _abs_rows("SNEW0001", 10, rng, fingerprint="fp_v2")
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        DS.save_shard(Path(td) / "s.npz", old_rows + new_rows)
        ds = DS.LabelDataset.load(td)
    assert len(ds) == 20
    any_ds = TV.filter_by_fingerprint(ds, "any", None)
    assert len(any_ds) == 20
    new_only = TV.filter_by_fingerprint(ds, "new_only", "fp_v2")
    assert len(new_only) == 10
    assert all(m.get("player_fingerprint") == "fp_v2" for m in new_only.meta)
    with pytest.raises(ValueError):
        TV.filter_by_fingerprint(ds, "new_only", None)
    with pytest.raises(ValueError):
        TV.filter_by_fingerprint(ds, "bogus_mode", "fp_v2")


def test_filter_pairs_by_fingerprint(pair_shards):
    pds = TV.PairDataset.load(pair_shards)
    assert set(pds.columns["player_fingerprint"].tolist()) == {"fp_new_1"}
    kept = TV.filter_pairs_by_fingerprint(pds, ["fp_new_1"])
    assert len(kept) == len(pds)
    dropped = TV.filter_pairs_by_fingerprint(pds, ["fp_other"])
    assert len(dropped) == 0
    unfiltered = TV.filter_pairs_by_fingerprint(pds, None)
    assert len(unfiltered) == len(pds)


# ── training loop: pair loss + metrics + no-pairs equivalence ──────────────────────

def test_no_pairs_configured_is_unaffected(abs_shards, tmp_path):
    """The bit-exact-resume guarantee for the ORIGINAL (pre-lever-b) trainer must still
    hold: with no --pair-shards, train_step/eval never touch the pair machinery."""
    cfg = TV.TrainVConfig(shards=[str(abs_shards)], run_dir=str(tmp_path / "run"), model="dummy",
                          batch_size=32, lr=3e-3, warmup_steps=5, max_steps=5, eval_every=5,
                          checkpoint_every=5, device="cpu", holdout_frac=0.25, seed=1)
    tr = TV.VTrainer(cfg, log=lambda *_: None)
    assert len(tr.train_pairs) == 0 and len(tr.holdout_pairs) == 0
    rec = tr.train_step()
    assert "pair_loss" not in rec and "pair_epoch" not in rec
    m = tr.eval()
    assert "pairs" not in m


def test_train_step_and_eval_report_pair_metrics(abs_shards, pair_shards, tmp_path):
    run = tmp_path / "run"
    cfg = _cfg(abs_shards, pair_shards, run, max_steps=40, eval_every=40, checkpoint_every=40)
    summ = TV.run(cfg, log=lambda *_: None)
    recs = [json.loads(l) for l in (run / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    ev = [r for r in recs if r["kind"] == "eval"][-1]
    assert "pairs" in ev and ev["pairs"]["n"] > 0
    p = ev["pairs"]
    assert 0 <= p["pair_acc"] <= 1 or np.isnan(p["pair_acc"])
    assert p["n_resolved"] <= p["n"]
    assert set(p["by_pair_source"]) <= {"close_call", "greedy_vs_extract", "random"}
    assert set(p["by_state_kind"]) <= {"hand", "shop"}
    assert "ece_guardrail_breached" in ev
    cfgrec = [r for r in recs if r["kind"] == "config"][0]
    assert cfgrec["n_train_pairs"] + cfgrec["n_holdout_pairs"] == 72
    summary = [r for r in recs if r["kind"] == "summary"][0]
    assert "pairs" in summary


def test_ece_guardrail_flags_when_exceeded(abs_shards, pair_shards, tmp_path):
    run = tmp_path / "run"
    cfg = _cfg(abs_shards, pair_shards, run, max_steps=1, eval_every=1, checkpoint_every=1,
              ece_guardrail=-1.0)   # impossible to satisfy -> always flagged
    tr = TV.VTrainer(cfg, log=lambda *_: None)
    m = tr.eval()
    assert m["ece_guardrail_breached"] is True


# ── bit-exact resume, extended to pair batches (the pinned test) ───────────────────

def test_resume_is_bit_exact_with_pair_batches(abs_shards, pair_shards, tmp_path):
    run = tmp_path / "run"
    cfg = _cfg(abs_shards, pair_shards, run, max_steps=60, eval_every=30, checkpoint_every=30)
    TV.run(cfg, log=lambda *_: None)
    netA, _, extraA = TV.VTrainer.read_checkpoint(run / "latest.pt")
    sdA = {k: v.clone() for k, v in netA.state_dict().items()}
    optA = extraA["trainer"]["optimizer"]
    tsA = extraA["trainer"]
    assert tsA["step"] == 60
    assert tsA["pair_epoch"] > 0            # 36 pairs / pair_batch_size 16 -> several epochs by step 60
    # continue from the step-30 checkpoint to 60 with the same schedule
    TV.run(resume=str(run / "ckpt_0000030.pt"), overrides={"max_steps": 60}, log=lambda *_: None)
    netB, _, extraB = TV.VTrainer.read_checkpoint(run / "latest.pt")
    tsB = extraB["trainer"]
    assert tsB["step"] == 60
    assert tsB["pair_epoch"] == tsA["pair_epoch"] and tsB["pair_cursor"] == tsA["pair_cursor"]
    assert all(torch.equal(sdA[k], v) for k, v in netB.state_dict().items())
    optB = tsB["optimizer"]
    for sa, sb in zip(optA["state"].values(), optB["state"].values()):
        for a, b in zip(sa.values(), sb.values()):
            if isinstance(a, torch.Tensor):
                assert torch.equal(a, b)
    assert [h["step"] for h in tsB["history"]] == [0, 30, 60]


# ── smoke: gradients flow through both branches; overfit to pair-acc ~1.0 ──────────

def test_gradients_flow_through_both_branches_dummy(abs_shards, pair_shards, tmp_path):
    cfg = _cfg(abs_shards, pair_shards, tmp_path / "run", max_steps=1)
    tr = TV.VTrainer(cfg, log=lambda *_: None)
    tr.net.zero_grad(set_to_none=True)
    loss, info = tr._pair_loss()
    loss.backward()
    assert info["pair_batch_n"] > 0
    grads = [p.grad for p in tr.net.parameters()]
    assert all(g is not None for g in grads)
    assert all(float(g.abs().sum()) > 0 for g in grads)


def test_overfit_tiny_pair_set_to_pair_accuracy_near_one(tmp_path):
    """Sanity per PHASE5_V2_BRIEF §6.4: a tiny, cleanly-separable pair set should drive
    held (here: train-set, since holdout_frac=0) pair accuracy to ~1.0, proving the loss
    and both forward branches actually carry a learnable signal end to end."""
    rng = np.random.default_rng(7)
    abs_d = tmp_path / "abs"
    DS.save_shard(abs_d / "s.npz", _abs_rows("SABS0001", 40, rng) + _abs_rows("SABS0002", 40, rng))
    pair_d = tmp_path / "pairs"
    rows = []
    for i in range(16):
        xa, xb = np.zeros(16, np.float32), np.zeros(16, np.float32)
        xa[3], xb[3] = 0.5, -0.5           # fixed, cleanly separable feature gap
        xa[15] = xb[15] = 1.0
        fields = {"seed": f"SPAIR{i:04d}", "step": i, "actor": 0, "state_kind": "hand", "ante": 2,
                 "player_fingerprint": "fp_new_1", "pair_source": "close_call", "n_worlds": 8}
        extra = {"action_a": "play:0", "action_b": "play:1", "outcomes_a": [1.0] * 8,
                "outcomes_b": [0.0] * 8, "meta": {}}
        rows.append(TV.PairRow({"x": xa}, {"x": xb}, 0.3, 0.05, fields, extra))
    TV.save_pair_shard(pair_d / "p.npz", rows)

    cfg = TV.TrainVConfig(shards=[str(abs_d)], pair_shards=[str(pair_d)], run_dir=str(tmp_path / "run"),
                          model="dummy", batch_size=16, pair_batch_size=16, lr=1e-2, warmup_steps=5,
                          max_steps=300, eval_every=300, checkpoint_every=300, device="cpu",
                          holdout_frac=0.0, lam_rank=5.0, tau=0.05, seed=2)
    tr = TV.VTrainer(cfg, log=lambda *_: None)
    assert len(tr.train_pairs) == 16 and len(tr.holdout_pairs) == 0
    for _ in range(cfg.max_steps):
        tr.train_step()
    m = TV.evaluate_pairs(tr.net, tr.train_pairs, tr.device, tau=cfg.tau, weight_cap=cfg.pair_weight_cap)
    assert m["n_resolved"] == 16
    assert m["pair_acc"] >= 0.95


@pytest.mark.skipif(not __import__("labels").has_encoder_v2(), reason="W1 encoder_v2/value_net not landed")
def test_set_value_net_pair_loss_grad_flow(tmp_path):
    """Real net smoke (brief §6.4): a real-shaped pair batch through the actual SetValueNet
    -> every parameter (all six item heads, attention, trunk, value head) gets a nonzero
    gradient from the pairwise loss alone."""
    pytest.importorskip("mcts.value_net")
    import labels as L
    enc = L.make_encoder("v2")
    snaps = L.sample_states("7I4M53DL", n_states=4, policy_factory=L.make_policy_factory("scripted", epsilon=0.1),
                            policy="scripted")
    abs_rows = [DS.LabelRow(enc(s.match, p), 0.5 + 0.1 * p, {"seed": f"SABS{i}", "step": s.step, "player": p,
                                                             "kind": s.kind, "ante": s.ante, "ci": 0.3})
               for i, s in enumerate(snaps) for p in (0, 1)]
    DS.save_shard(tmp_path / "abs" / "s.npz", abs_rows)

    pair_rows = []
    for i in range(len(snaps) - 1):
        obs_a, obs_b = enc(snaps[i].match, 0), enc(snaps[i + 1].match, 0)
        fields = {"seed": f"SPAIR{i}", "step": i, "actor": 0, "state_kind": snaps[i].kind, "ante": snaps[i].ante,
                 "player_fingerprint": "fp_new_1", "pair_source": "close_call", "n_worlds": 8}
        extra = {"action_a": "a", "action_b": "b", "outcomes_a": [1.0] * 8, "outcomes_b": [0.0] * 8, "meta": {}}
        pair_rows.append(TV.PairRow(obs_a, obs_b, 0.2, 0.05, fields, extra))
    TV.save_pair_shard(tmp_path / "pairs" / "p.npz", pair_rows)

    small = {"d_item": 32, "trunk_width": 64, "n_res_blocks": 1, "scalar_hidden": 32, "key_emb": 8}
    cfg = TV.TrainVConfig(shards=[str(tmp_path / "abs")], pair_shards=[str(tmp_path / "pairs")],
                          run_dir=str(tmp_path / "run"), model="set_value_net", net_cfg=small,
                          batch_size=4, pair_batch_size=4, max_steps=1, eval_every=1, checkpoint_every=1,
                          device="cpu", warmup_steps=1, holdout_frac=0.0, keep=5)
    tr = TV.VTrainer(cfg, log=lambda *_: None)
    tr.net.zero_grad(set_to_none=True)
    loss, info = tr._pair_loss()
    loss.backward()
    assert info["pair_batch_n"] > 0

    # A per-item-type head (joker_mlp/cons_mlp/shelf_mlp/pack_mlp) legitimately gets ZERO
    # gradient if that item set is empty (mask all-zero) in EVERY row of the batch on BOTH
    # sides: the masked mean/max pooling returns a mask-independent constant for an empty
    # set, so the head's output never reaches the loss.  This is a property of the
    # encoder's masked pooling, not of the pairwise loss — only skip a head's params when
    # its mask is provably empty across the whole batch; require gradient everywhere else
    # (in particular: hand_mlp, the shared card/aux/key tables, attention, trunk, value_head
    # must ALWAYS get gradient, since a decision state always has a hand and blind offers).
    per_type_prefix = {"joker_mlp": "joker_mask", "cons_mlp": "cons_mask", "shelf_mlp": "shelf_mask",
                       "pack_mlp": "pack_mask", "blind_mlp": "blind_mask"}
    populated = set()
    for name, mask_key in per_type_prefix.items():
        if any(float(r.obs_a[mask_key].sum()) > 0 or float(r.obs_b[mask_key].sum()) > 0 for r in pair_rows):
            populated.add(name)
    allowed_empty = {p for p in per_type_prefix if p not in populated}

    named = list(tr.net.named_parameters())
    missing = [n for n, p in named if p.grad is None or float(p.grad.abs().sum()) == 0.0]
    unexpected = [n for n in missing if not any(n.startswith(pfx + ".") for pfx in allowed_empty)]
    assert not unexpected, f"no pair-loss gradient reached (unexpected): {unexpected}; " \
        f"(expected-empty item types this fixture happened to sample: {sorted(allowed_empty)})"


# ── real interop: W-PAIRS's actual pairs.py landed mid-build (2026-08-25) — verify this
# module reads its ACTUAL shard output, not just the schema text ─────────────────────

def test_reads_pairs_py_shards_directly(tmp_path):
    """``pairs.py`` (W-PAIRS) landed concurrently with this workstream.  Its
    ``save_pair_shard`` turned out to match this module's independently-chosen on-disk
    layout (obs_a__<key>/obs_b__<key> + typed PAIR_COLUMNS + a per-row JSON blob) almost
    exactly — the one difference was the blob column's name (``pair_json`` there,
    ``extra_json`` here), reconciled in ``load_pair_shard_npz`` (RANK_NOTES §2).  This test
    constructs a real ``pairs.PairRow`` + ``pairs.save_pair_shard`` (their actual functions,
    not a re-implementation) and confirms THIS module's ``PairDataset.load`` reads it."""
    import pairs as WP
    rec = {"kind": "pair", "seed": "SINTEROP", "step": 5, "actor": 1, "state_kind": "shop",
           "ante": 3, "player_fingerprint": WP.player_fingerprint(), "pair_source": "greedy_vs_extract",
           "action_a": {"type": "buy", "idx": 0}, "action_b": {"type": "skip"}, "n_worlds": 8,
           "outcomes_a": [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], "outcomes_b": [0.0] * 8,
           "delta": 0.625, "delta_ci": 0.31, "meta": {"note": "interop test, not a real rollout"}}
    obs_a = {"x": np.full(16, 0.7, np.float32)}
    obs_b = {"x": np.full(16, -0.7, np.float32)}
    row = WP.PairRow(obs_a=obs_a, obs_b=obs_b, rec=rec)
    path = WP.save_pair_shard(tmp_path / "wpairs_0000.npz", [row])

    # W-PAIRS's own reader agrees with itself (sanity on the fixture, not this module)
    wshard = WP.load_pair_shard(path)
    assert len(wshard) == 1 and wshard.records[0]["delta"] == pytest.approx(0.625)

    pds = TV.PairDataset.load(tmp_path)
    assert len(pds) == 1
    assert pds.delta[0] == pytest.approx(0.625) and pds.delta_ci[0] == pytest.approx(0.31)
    assert pds.columns["seed"][0] == "SINTEROP" and pds.columns["pair_source"][0] == "greedy_vs_extract"
    assert pds.columns["state_kind"][0] == "shop" and pds.columns["actor"][0] == 1
    np.testing.assert_allclose(pds.obs_a["x"][0], obs_a["x"])
    np.testing.assert_allclose(pds.obs_b["x"][0], obs_b["x"])
    assert pds.extra[0]["outcomes_a"] == rec["outcomes_a"]
    assert pds.extra[0]["action_b"] == {"type": "skip"}
    assert pds.extra[0]["meta"] == {"note": "interop test, not a real rollout"}


def test_cli_parses_pair_flags(abs_shards, pair_shards, tmp_path):
    run = tmp_path / "cli"
    rc = TV.main(["--shards", str(abs_shards), "--pair-shards", str(pair_shards), "--run-dir", str(run),
                  "--model", "dummy", "--max-steps", "10", "--eval-every", "5", "--checkpoint-every", "5",
                  "--device", "cpu", "--batch-size", "16", "--pair-batch-size", "8", "--warmup-steps", "2",
                  "--holdout-frac", "0.25", "--lam-rank", "0.5", "--tau", "0.1", "--pair-weight-cap", "3",
                  "--ece-guardrail", "0.08", "--absolute-fingerprint-mode", "any"])
    assert rc == 0 and (run / ".DONE").exists()
    _, _, extra = TV.VTrainer.read_checkpoint(run / "latest.pt")
    cfg = extra["trainer"]["config"]
    assert cfg["lam_rank"] == 0.5 and cfg["tau"] == 0.1 and cfg["pair_weight_cap"] == 3.0
    assert cfg["pair_batch_size"] == 8 and cfg["ece_guardrail"] == 0.08
    assert extra["trainer"]["n_train_pairs"] + extra["trainer"]["n_holdout_pairs"] == 72
    # resume without repeating --pair-shards keeps the pair config (like `shards`)
    rc = TV.main(["--resume", str(run), "--max-steps", "14"])
    assert rc == 0
