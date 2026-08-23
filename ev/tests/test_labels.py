"""test_labels.py — snapshots, rollouts, labels, shards (W5).

Stage 1 runs on the scripted fallback policy (W3's EVPlayer is feature-detected; the
EV-specific tests skip until it lands).  Everything here is seconds."""
from __future__ import annotations

import numpy as np
import pytest

import _bootstrap
from _bootstrap import MLBMatch

import dataset as DS
import labels as L
import race as R

SEED = "7I4M53DL"


@pytest.fixture(scope="module")
def scripted():
    return L.make_policy_factory("scripted", epsilon=0.1)


@pytest.fixture(scope="module")
def snaps(scripted):
    return L.sample_states(SEED, n_states=10, policy_factory=scripted, policy="scripted", epsilon=0.1)


# ── state kinds + snapshots ──

def test_state_kind_covers_the_enum():
    m = MLBMatch(seed=SEED)
    assert L.state_kind(m.games[0]) == "blind_select"
    assert set(L.STATE_KINDS) >= {"blind_select", "hand", "nemesis", "shop", "pack", "other"}


def test_sample_states_is_stratified_and_tagged(snaps):
    assert 1 <= len(snaps) <= 10
    kinds = {s.kind for s in snaps}
    assert "hand" in kinds and "blind_select" in kinds
    for s in snaps:
        assert s.seed == SEED and s.step == s.match.steps
        assert s.kind == L.state_kind(s.match.games[s.actor])
        assert s.actor == s.match.current_player()
        assert "winner" in s.selfplay and s.selfplay["policy"] == "scripted"
        assert s.ante == s.match.games[s.actor].ante
    assert [s.step for s in snaps] == sorted(s.step for s in snaps)


def test_sample_states_is_deterministic_and_reconstructible(scripted):
    a = L.sample_states(SEED, n_states=6, policy_factory=L.make_policy_factory("scripted", epsilon=0.1),
                        policy="scripted", epsilon=0.1)
    b = L.sample_states(SEED, n_states=6, policy_factory=L.make_policy_factory("scripted", epsilon=0.1),
                        policy="scripted", epsilon=0.1)
    assert [(s.step, s.actor, s.kind) for s in a] == [(s.step, s.actor, s.kind) for s in b]
    for s in a[:3]:
        m = L.reconstruct_snapshot(SEED, s.step, policy_factory=L.make_policy_factory("scripted", epsilon=0.1))
        assert m.signature() == s.match.signature()


def test_snapshot_clone_is_independent(snaps):
    s = snaps[1]
    sig = s.match.signature()
    L.rollout(s.match, seed=3, policy_factory=L.make_policy_factory("scripted", epsilon=0.05),
              determinize=False)
    assert s.match.signature() == sig


# ── rollouts + labels ──

def test_rollout_terminates_with_an_outcome_or_a_race(snaps):
    pf = L.make_policy_factory("scripted", epsilon=0.05)
    r = L.rollout(snaps[0].match, seed=1, policy_factory=pf, determinize=False)
    assert 0.0 <= r.p0_win <= 1.0
    assert r.decisions > 0 and r.seconds > 0
    if r.truncated:
        assert r.winner is None and min(r.antes) > L.DEFAULT_MAX_ANTE
    else:
        assert r.winner in (0, 1) and r.p0_win == float(r.winner == 0)
    # a tiny max_ante forces the race-calculator path
    r2 = L.rollout(snaps[0].match, seed=1, policy_factory=pf, determinize=False, max_ante=1)
    assert r2.truncated and 0.0 < r2.p0_win < 1.0


def test_label_both_perspectives_sum_to_one_exactly(snaps):
    pf = L.make_policy_factory("scripted", epsilon=0.05)
    r0, r1 = L.label_both(snaps[2].match, n_rollouts=6, seed=5, policy_factory=pf, determinize=False)
    assert r0.n == 6 and r1.n == 6
    assert r0.y + r1.y == pytest.approx(1.0)
    assert r0.ci == r1.ci and 0.0 < r0.ci <= 1.0
    assert len(r0.outcomes) == 6
    y, ci = L.label_state(snaps[2].match, 1, n_rollouts=6, seed=5, policy_factory=pf, determinize=False)
    assert y == pytest.approx(r1.y) and ci == pytest.approx(r1.ci)


def test_independent_rollout_sets_sum_to_about_one(snaps):
    """The real sanity test: label player 0 and player 1 with DIFFERENT rollout seeds and
    check the means sum to ~1 over several snapshots (a finished match has one winner and
    the race model is symmetric, so the only slack is Monte-Carlo noise)."""
    pf = L.make_policy_factory("scripted", epsilon=0.2)
    sums = []
    for i, s in enumerate(snaps[:5]):
        y0, _ = L.label_state(s.match, 0, n_rollouts=6, seed=100 + i, policy_factory=pf, determinize=False)
        y1, _ = L.label_state(s.match, 1, n_rollouts=6, seed=200 + i, policy_factory=pf, determinize=False)
        sums.append(y0 + y1)
    assert abs(float(np.mean(sums)) - 1.0) < 0.35


def test_wilson_and_ci():
    assert L.wilson_halfwidth(0, 8) == pytest.approx(L.wilson_halfwidth(8, 8))
    assert L.wilson_halfwidth(4, 8) > L.wilson_halfwidth(4, 32)
    assert L._ci_of([0.0, 1.0, 1.0, 0.0]) == pytest.approx(L.wilson_halfwidth(2, 4))
    soft = L._ci_of([0.3, 0.7, 0.5, 0.4])
    assert 0.0 < soft < 0.5


def test_race_truncation_uses_match_history():
    m = MLBMatch(seed=SEED)
    m.pvp_log = [(2, 1, 1500, 1200), (3, 0, 3000, 4000)]
    m.games[0].lives, m.games[1].lives = 3, 2
    for g in m.games:
        g.ante = 13
    p0 = L._race_p0(m, 12, R.DEFAULT)
    assert 0.0 < p0 < 1.0
    m.games[0].lives = 1
    assert L._race_p0(m, 12, R.DEFAULT) < p0


# ── the worker job + shards ──

def test_label_job_dummy_encoder_and_shard_round_trip(tmp_path):
    res = L.label_job({"seed": SEED, "n_states": 5, "n_rollouts": 3, "policy": "scripted",
                       "encoder": "dummy", "allow_clairvoyant": True})
    rows = L.rows_from_result(res)
    assert len(rows) == 2 * res["timing"]["n_snapshots"] and len(rows) >= 2
    t = res["timing"]
    assert t["n_rollouts"] == 3 * t["n_snapshots"] and t["ms_per_rollout"] > 0
    assert 0.0 <= t["policy_frac"] <= 1.0
    for r in rows:
        assert r.obs["x"].shape == (16,) and 0.0 <= r.y <= 1.0
        assert r.meta["seed"] == SEED and r.meta["player"] in (0, 1) and r.meta["kind"] in L.STATE_KINDS
    # both perspectives of a snapshot sum to one
    by_step = {}
    for r in rows:
        by_step.setdefault(r.meta["step"], []).append(r.y)
    assert all(abs(sum(v) - 1.0) < 1e-9 for v in by_step.values() if len(v) == 2)
    p = DS.save_shard(tmp_path / "s.npz", rows)
    sh = DS.load_shard(p)
    assert len(sh) == len(rows)
    assert np.array_equal(sh.obs["x"], np.stack([r.obs["x"] for r in rows]))
    assert sh.columns["seed"][0] == SEED and sh.columns["kind"].dtype.kind == "U"
    assert sh.meta[0]["outcomes"] == rows[0].meta["outcomes"]
    ds = DS.LabelDataset.from_shards([sh, sh])
    assert len(ds) == 2 * len(rows)
    summ = ds.summary()
    assert summ["n"] == len(ds) and set(summ["by_kind"]) <= set(L.STATE_KINDS)


def test_label_job_refuses_clairvoyant_rollouts_unless_asked(monkeypatch):
    monkeypatch.setattr(L, "has_determinize", lambda: False)
    with pytest.raises(RuntimeError, match="clone_determinized"):
        L.label_job({"seed": SEED, "n_states": 1, "n_rollouts": 1, "policy": "scripted", "encoder": "dummy"})


def test_split_by_seed_never_splits_a_seed(tmp_path):
    rows = []
    for seed in ("AAAAAAAA", "BBBBBBBB", "CCCCCCCC", "DDDDDDDD"):
        for i in range(5):
            rows.append(DS.LabelRow({"x": np.full(4, i, np.float32)}, 0.5, {"seed": seed, "step": i, "player": i % 2,
                                                                          "kind": "hand", "ante": 1}))
    ds = DS.LabelDataset.from_shards([DS.load_shard(DS.save_shard(tmp_path / "a.npz", rows))])
    tr, ho = ds.split_by_seed(holdout_seeds=["BBBBBBBB"])
    assert len(ho) == 5 and set(ho.columns["seed"]) == {"BBBBBBBB"}
    assert len(tr) == 15 and "BBBBBBBB" not in set(tr.columns["seed"])
    tr2, ho2 = ds.split_by_seed(holdout_frac=0.5)
    assert set(tr2.seeds()).isdisjoint(ho2.seeds()) and len(tr2) + len(ho2) == 20
    # the hash rule is stable
    assert DS.seed_in_holdout("AAAAAAAA", 0.5) == DS.seed_in_holdout("AAAAAAAA", 0.5)
    n_batches = sum(1 for _ in ds.batches(8, rng=np.random.default_rng(0)))
    assert n_batches == 3


# ── determinized rollouts (W2) ──

@pytest.mark.skipif(not L.has_determinize(), reason="W2 clone_determinized not landed")
def test_determinized_rollouts_differ_across_seeds(snaps):
    pf = L.make_policy_factory("scripted", epsilon=0.0)
    s = next(x for x in snaps if x.kind == "hand")
    outs = set()
    for k in range(4):
        m = s.match.clone_determinized(1000 + k)
        assert m.signature()[2:] == s.match.signature()[2:]        # match scalars preserved
        r = L.rollout(s.match, seed=1000 + k, policy_factory=pf)
        assert r.determinized
        outs.add((r.steps, r.antes, r.lives))
    assert len(outs) >= 2            # different worlds -> different futures


# ── the real rollout policy (W3) ──

@pytest.mark.skipif(not L.has_ev_player(), reason="W3 EVPlayer not landed")
def test_ev_policy_factory_drives_a_rollout(snaps):
    pf = L.make_policy_factory("ev", budget="fast", epsilon=0.02)
    r = L.rollout(snaps[0].match, seed=7, policy_factory=pf, max_ante=2)
    assert r.decisions > 0 and 0.0 <= r.p0_win <= 1.0


@pytest.mark.skipif(not L.has_encoder_v2(), reason="W1 encoder_v2 not landed")
def test_encoder_v2_rows_have_fixed_shapes(snaps):
    enc = L.make_encoder("v2")
    o0 = enc(snaps[0].match, 0)
    o1 = enc(snaps[-1].match, 1)
    assert set(o0) == set(o1) and all(o0[k].shape == o1[k].shape for k in o0)
    assert "scalars" in o0
