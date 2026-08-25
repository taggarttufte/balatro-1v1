"""test_active_poc.py — the POC's own invariants (seconds; no GPU, no label campaign).

The load-bearing ones are:
  * ``snapshot_rng_seed`` is process-stable (``labels.sample_states``'s default is NOT — it
    hashes a str, and PYTHONHASHSEED is unset), and the snapshot set it produces really is
    reproducible, because ``arm_job`` re-derives the states ``pool_job`` scored;
  * the base corpus and the candidate pool can never contain an evaluation-holdout seed;
  * stratification actually caps a kind, and each arm comes out at the requested size.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import _bootstrap  # noqa: F401
from dataset import LabelDataset, LabelRow, META_COLUMNS, seed_in_holdout

from active_poc import corpus as C
from active_poc import select as S
from active_poc.jobs import CORPUS_CONFIG, obs_fingerprint, snapshot_rng_seed

EV_ROOT = Path(__file__).resolve().parent.parent.parent


# ── reproducibility ───────────────────────────────────────────────────────────────

def test_snapshot_rng_seed_is_process_stable():
    """A fresh interpreter must produce the same value (plain ``hash`` does not)."""
    code = ("import sys; sys.path.insert(0, r'%s'); "
            "from active_poc.jobs import snapshot_rng_seed; "
            "print(snapshot_rng_seed('QZ7M4KP2', 0), hash(('QZ7M4KP2', 0)) & 0xFFFFFFFF)" % EV_ROOT)
    outs = []
    for _ in range(2):
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        outs.append(r.stdout.split())
    assert outs[0][0] == outs[1][0] == str(snapshot_rng_seed("QZ7M4KP2", 0))
    # the guard rail this exists for: the stdlib hash really does differ across processes
    assert outs[0][1] != outs[1][1], "PYTHONHASHSEED appears pinned; the gotcha note is stale"


def test_snapshot_rng_seed_varies_with_inputs():
    assert snapshot_rng_seed("AAAA1111") != snapshot_rng_seed("AAAA1112")
    assert snapshot_rng_seed("AAAA1111", 0) != snapshot_rng_seed("AAAA1111", 1)


def test_obs_fingerprint_detects_change():
    a = {"x": np.zeros(4, np.float32), "m": np.ones(2, np.float32)}
    b = {"m": np.ones(2, np.float32), "x": np.zeros(4, np.float32)}
    assert obs_fingerprint(a) == obs_fingerprint(b)          # key order irrelevant
    c = dict(a)
    c["x"] = a["x"].copy()
    c["x"][2] = 1.0
    assert obs_fingerprint(a) != obs_fingerprint(c)


def test_snapshots_are_reproducible_across_calls():
    """The contract ``arm_job`` relies on: same seed -> same snapshot steps and same encoding."""
    from active_poc.jobs import _factories, snapshots_for_seed
    sp, _ro, encode = _factories(CORPUS_CONFIG)
    a = snapshots_for_seed("QZ7M4KP2", CORPUS_CONFIG, sp)
    b = snapshots_for_seed("QZ7M4KP2", CORPUS_CONFIG, sp)
    assert [s.step for s in a] == [s.step for s in b]
    assert [s.kind for s in a] == [s.kind for s in b]
    assert [obs_fingerprint(encode(s.match, 0)) for s in a] == \
           [obs_fingerprint(encode(s.match, 0)) for s in b]


# ── corpus / holdout hygiene ──────────────────────────────────────────────────────

def _fake_ds(seeds, per_seed=2):
    rows = []
    for s in seeds:
        for i in range(per_seed):
            rows.append(LabelRow({"x": np.full(3, float(i), np.float32)}, 0.5,
                                 {"seed": s, "step": i, "player": i % 2, "kind": "hand",
                                  "ante": 1, "ci": 0.2, "n_rollouts": 8, "trunc_frac": 0.0}))
    obs = {"x": np.stack([r.obs["x"] for r in rows])}
    y = np.asarray([r.y for r in rows], np.float32)
    cols = {n: np.asarray([r.meta.get(n) for r in rows]) for n in META_COLUMNS}
    return LabelDataset(obs, y, cols, [r.meta for r in rows])


def test_canonical_seed_matches_the_engine():
    """Balatro's seed alphabet has no zero; the engine reports the canonical form, and the
    holdout hash rule is applied to THAT.  Regression for the leak found 2026-08-25."""
    from _bootstrap import MLBMatch
    for s in ("DH0HXASZ", "A2FZ01XM", "ABCDEFGH", "01234567"):
        assert C.canonical_seed(s) == MLBMatch(seed=s).seed_str
    assert C.canonical_seed("DH0HXASZ") == "DHOHXASZ"


def test_fresh_seeds_are_canonical_and_avoid_holdout():
    """The bug: a raw-string holdout test passes a seed whose CANONICAL form is in holdout."""
    got = C.fresh_seeds(400, rng_seed=20260825, exclude=[])
    assert all(C.canonical_seed(s) == s for s in got), "fresh seeds must already be canonical"
    assert not [s for s in got if seed_in_holdout(C.canonical_seed(s), C.STANDARD_HOLDOUT_FRAC)]
    assert "0" not in "".join(got)


def test_drop_holdout_seeds_removes_leaked_rows():
    seeds = [f"U{i:07d}" for i in range(400)]
    ds = _fake_ds(seeds)
    clean, dropped = C.drop_holdout_seeds(ds, C.STANDARD_HOLDOUT_FRAC)
    assert dropped, "fixture should contain some holdout seeds"
    assert not [s for s in clean.columns["seed"].tolist()
                if seed_in_holdout(s, C.STANDARD_HOLDOUT_FRAC)]
    assert len(clean) == len(ds) - 2 * len(dropped)


def test_fresh_seeds_avoid_holdout_and_exclusions():
    excl = ["AAAAAAAA", "BBBBBBBB"]
    got = C.fresh_seeds(200, rng_seed=1, exclude=excl)
    assert len(got) == len(set(got)) == 200
    assert not (set(got) & set(excl))
    assert not [s for s in got if seed_in_holdout(s, C.STANDARD_HOLDOUT_FRAC)]


def test_subsample_keeps_whole_seeds_and_hits_the_fraction():
    seeds = [f"S{i:07d}" for i in range(2000)]
    ds = _fake_ds(seeds, per_seed=4)
    sub = C.subsample_by_seed_hash(ds, 0.25)
    keep = set(sub.columns["seed"].tolist())
    assert 0.20 < len(keep) / len(seeds) < 0.30
    for s in keep:                                   # every row of a kept seed survives
        assert int((sub.columns["seed"] == s).sum()) == 4
    assert not (keep & (set(seeds) - keep))


def test_base_and_holdout_never_share_a_seed():
    seeds = [f"T{i:07d}" for i in range(1500)]
    ds = _fake_ds(seeds)
    train, hold = ds.split_by_seed(C.STANDARD_HOLDOUT_FRAC)
    base = C.subsample_by_seed_hash(train, 0.3)
    assert not (set(base.columns["seed"].tolist()) & set(hold.columns["seed"].tolist()))
    assert len(hold) > 0 and len(base) > 0


def test_concat_and_select_rows_by_state():
    a = _fake_ds(["AAA00001", "AAA00002"])
    b = _fake_ds(["BBB00001"])
    both = C.concat(a, b)
    assert len(both) == len(a) + len(b)
    assert both.obs["x"].shape[0] == len(both)
    # base rows come first (the pairing property stage_final relies on)
    assert both.columns["seed"].tolist()[: len(a)] == a.columns["seed"].tolist()
    picked = C.select_rows_by_state(both, [("AAA00002", 0), ("AAA00002", 1)])
    assert len(picked) == 2
    assert set(picked.columns["seed"].tolist()) == {"AAA00002"}


# ── acquisition rules ─────────────────────────────────────────────────────────────

def _fake_scores(n=600, kinds=("hand", "shop", "nemesis")):
    rng = np.random.default_rng(0)
    out = {}
    for i in range(n):
        k = kinds[i % len(kinds)]
        # make "hand" systematically the most disagreed-on, so an unstratified top-k
        # would be monopolised by it
        bump = 0.5 if k == "hand" else 0.0
        out[(f"S{i:06d}", i)] = {"kind": k, "ante": 1 + i % 4,
                                 "disagreement": float(rng.random() * 0.1 + bump),
                                 "err_proxy": float(rng.random()),
                                 "y_probe": 0.5, "ci_probe": 0.3, "fp": "x", "n_rows": 2}
    return out


def test_stratified_topk_caps_a_dominant_kind():
    scores = _fake_scores()
    n = 120
    plain = sorted(scores.items(), key=lambda kv: -kv[1]["disagreement"])[:n]
    assert all(r["kind"] == "hand" for _s, r in plain), "fixture should be monopolisable"
    picked = S.stratified_topk(scores, "disagreement", n, cap_mult=1.5)
    assert len(picked) == n
    counts = {}
    for s in picked:
        counts[scores[s]["kind"]] = counts.get(scores[s]["kind"], 0) + 1
    # pool share of each kind is 1/3, so the cap is 1.5 * n / 3 = n/2
    assert counts["hand"] <= int(np.ceil(1.5 * n / 3))
    assert len(counts) > 1


def test_stratified_topk_prefers_high_scores_within_a_kind():
    scores = _fake_scores()
    picked = set(S.stratified_topk(scores, "err_proxy", 90, cap_mult=1.5))
    chosen = [scores[s]["err_proxy"] for s in picked]
    rest = [r["err_proxy"] for s, r in scores.items() if s not in picked]
    assert np.mean(chosen) > np.mean(rest)


def test_uniform_sample_is_deterministic_and_sized():
    scores = _fake_scores()
    a = S.uniform_sample(scores, 50, rng_seed=7)
    b = S.uniform_sample(scores, 50, rng_seed=7)
    c = S.uniform_sample(scores, 50, rng_seed=8)
    assert a == b and len(a) == 50 and len(set(a)) == 50
    assert a != c


def test_state_scores_aggregates_both_perspectives():
    ds = _fake_ds(["Z0000001"], per_seed=2)          # step 0 player 0, step 1 player 1
    ds.columns["step"] = np.asarray([5, 5])          # one state, two perspectives
    ds.columns["player"] = np.asarray([0, 1])
    ds.y = np.asarray([0.25, 0.75], np.float32)
    ds.meta[0]["obs_fp"] = "abc"
    mean_v = np.asarray([0.5, 0.5])
    std_v = np.asarray([0.1, 0.3])
    sc = S.state_scores(ds, mean_v, std_v)
    assert list(sc) == [("Z0000001", 5)]
    r = sc[("Z0000001", 5)]
    assert r["n_rows"] == 2
    assert r["disagreement"] == pytest.approx(0.2)          # mean of 0.1 and 0.3
    assert r["err_proxy"] == pytest.approx(0.25)            # mean of |0.5-0.25|, |0.5-0.75|
    assert r["y_probe"] == pytest.approx(0.25)              # player-0 perspective
    assert r["fp"] == "abc"


def test_overlap_table():
    arms = {"a": [("s", 1), ("s", 2), ("s", 3)], "b": [("s", 3), ("s", 4)]}
    o = S.overlap_table(arms)["a|b"]
    assert o["intersection"] == 1 and o["union"] == 4
    assert o["jaccard"] == pytest.approx(0.25)
    assert o["frac_of_arm"] == pytest.approx(1 / 3)
