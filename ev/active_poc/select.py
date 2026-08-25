"""select.py — the three acquisition rules, scored per STATE (not per row).

A snapshot yields TWO rows (both perspectives) from ONE rollout set, so the label budget is
spent per state and the selection must be per state too.  Each state's score is the mean of
its two perspectives' scores:

* ``disagreement`` — sd of the 3 ensemble members' V on that row.  Pure epistemic signal:
  it needs no rollouts at all, only three forward passes.
* ``err_proxy``    — ``|mean_V - y_probe|`` where ``y_probe`` is the cheap 2-rollout label
  from ``pool_job``.  This is the "naive error proxy": with 2 Bernoulli-ish rollouts its own
  sd is ~0.35, so most of what it ranks on is LABEL NOISE, not model error.  Quantifying
  that is half the point of the POC.
* ``uniform``      — a seeded random sample of the pool.  The control.

Stratification (light).  Top-k on a raw score lets one state kind monopolise an arm — the
kinds differ systematically in label sd (nemesis 0.343 vs other 0.293 in the 51k corpus), so
an unstratified "most uncertain" arm is partly just "the kind with the noisiest labels", and
the comparison would measure kind mix rather than acquisition.  Each kind is therefore capped
at ``cap_mult`` x its share of the pool (default 1.5, i.e. a kind may be over-represented by
half again but not more); states are filled in global score order subject to the caps, and if
the caps leave the arm short it is topped up with the best remaining states regardless of
kind.  The uniform control is NOT stratified — its kind mix is the pool's natural mix, which
is the honest baseline.
"""
from __future__ import annotations

import math
import random
from typing import Optional

import numpy as np

__all__ = ["state_scores", "stratified_topk", "uniform_sample", "overlap_table", "ARMS"]

ARMS = ("disagreement", "err_proxy", "uniform")


def state_scores(ds, mean_v: np.ndarray, std_v: np.ndarray) -> dict:
    """Collapse row-level scores to per-state records.

    Returns ``{(seed, step): {...}}`` with ``disagreement``, ``err_proxy``, ``kind``,
    ``ante``, ``y_probe`` (player-0 perspective), ``ci_probe``, ``fp`` and ``n_rows``.
    """
    seeds = ds.columns["seed"].tolist()
    steps = ds.columns["step"].tolist()
    players = ds.columns["player"].tolist()
    kinds = ds.columns["kind"].tolist()
    antes = ds.columns["ante"].tolist()
    ci = ds.columns["ci"]
    out: dict = {}
    for i, (sd, st) in enumerate(zip(seeds, steps)):
        key = (str(sd), int(st))
        rec = out.setdefault(key, {"kind": str(kinds[i]), "ante": int(antes[i]),
                                   "dis_parts": [], "err_parts": [], "n_rows": 0,
                                   "y_probe": float("nan"), "ci_probe": float(ci[i]), "fp": None})
        rec["dis_parts"].append(float(std_v[i]))
        rec["err_parts"].append(abs(float(mean_v[i]) - float(ds.y[i])))
        rec["n_rows"] += 1
        if int(players[i]) == 0:
            rec["y_probe"] = float(ds.y[i])
            rec["fp"] = ds.meta[i].get("obs_fp")
    for rec in out.values():
        rec["disagreement"] = float(np.mean(rec["dis_parts"]))
        rec["err_proxy"] = float(np.mean(rec["err_parts"]))
        del rec["dis_parts"], rec["err_parts"]
    return out


def _kind_shares(scores: dict) -> dict:
    n = len(scores)
    counts: dict = {}
    for r in scores.values():
        counts[r["kind"]] = counts.get(r["kind"], 0) + 1
    return {k: c / n for k, c in counts.items()}


def stratified_topk(scores: dict, key: str, n: int, *, cap_mult: float = 1.5) -> list:
    """Top-``n`` states by ``key`` with each kind capped at ``cap_mult`` x its pool share."""
    shares = _kind_shares(scores)
    caps = {k: max(1, int(math.ceil(cap_mult * n * s))) for k, s in shares.items()}
    order = sorted(scores.items(), key=lambda kv: -kv[1][key])
    taken: list = []
    used: dict = {k: 0 for k in caps}
    for state, rec in order:
        if len(taken) >= n:
            break
        k = rec["kind"]
        if used[k] >= caps.get(k, 0):
            continue
        used[k] += 1
        taken.append(state)
    if len(taken) < n:                       # caps left the arm short: top up on raw score
        have = set(taken)
        for state, _rec in order:
            if len(taken) >= n:
                break
            if state not in have:
                taken.append(state)
    return taken


def uniform_sample(scores: dict, n: int, *, rng_seed: int = 0) -> list:
    states = sorted(scores)
    rng = random.Random(rng_seed)
    rng.shuffle(states)
    return states[:n]


def overlap_table(arms: dict) -> dict:
    """Pairwise |A ∩ B| and Jaccard for the selected state sets."""
    out: dict = {}
    names = list(arms)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            sa, sb = set(arms[a]), set(arms[b])
            inter = len(sa & sb)
            out[f"{a}|{b}"] = {"intersection": inter, "union": len(sa | sb),
                               "jaccard": inter / max(len(sa | sb), 1),
                               "frac_of_arm": inter / max(len(sa), 1)}
    return out


def arm_profile(scores: dict, states: list, key: Optional[str] = None) -> dict:
    """Score/kind/ante/label-noise profile of one arm's selection."""
    recs = [scores[s] for s in states]
    prof = {"n_states": len(recs),
            "by_kind": {}, "by_ante": {},
            "disagreement_mean": float(np.mean([r["disagreement"] for r in recs])),
            "err_proxy_mean": float(np.mean([r["err_proxy"] for r in recs])),
            "ci_probe_mean": float(np.mean([r["ci_probe"] for r in recs])),
            "y_probe_mean": float(np.mean([r["y_probe"] for r in recs])),
            "y_probe_sd": float(np.std([r["y_probe"] for r in recs]))}
    for r in recs:
        prof["by_kind"][r["kind"]] = prof["by_kind"].get(r["kind"], 0) + 1
        prof["by_ante"][str(r["ante"])] = prof["by_ante"].get(str(r["ante"]), 0) + 1
    if key:
        prof["selected_on"] = key
    return prof
