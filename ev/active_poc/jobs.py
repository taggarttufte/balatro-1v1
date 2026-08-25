"""jobs.py — the two pool jobs of the active-learning POC (W-ACTIVE).

Both are module-level, picklable functions for ``workers.run_pool`` (Windows spawn).  They
call ``labels.py`` as a library and reproduce the ``labels_full`` corpus config exactly, so
the candidate pool is drawn from the SAME state distribution and labelled by the SAME policy
as the base corpus.  That consistency is load-bearing: a difference between arms must come
from WHICH states were chosen, not from a different generator.

``pool_job``   — self-play one fresh seed -> the same stratified snapshots ``label_job``
                 would take -> encode both perspectives -> a CHEAP ``n_probe``-rollout label
                 (the error proxy's noisy reference).  No 8-rollout label.
``arm_job``    — re-derive the snapshots of one seed and label ONLY the selected steps at the
                 full ``n_rollouts`` (8), i.e. the standard label pipeline on chosen states.

Reproducibility gotcha (found 2026-08-25, and the reason both jobs pass ``rng_seed=``
explicitly):  ``labels.sample_states`` defaults its reservoir RNG to
``hash((seed, policy_seed)) & 0xFFFFFFFF``.  ``hash()`` of a *str* is salted per process
(PYTHONHASHSEED is unset here), so the snapshot SET a seed produces is not reproducible
across processes.  ``snapshot_rng_seed`` below replaces it with a sha1 of the same pair, so
``arm_job`` re-derives byte-identical snapshots to the ones ``pool_job`` scored.  ``arm_job``
verifies that per state against the observation ``pool_job`` recorded (``obs_fingerprint``)
and refuses to label a state whose reconstruction drifted.
"""
from __future__ import annotations

import hashlib
import time
from typing import Optional

import numpy as np

import _bootstrap  # noqa: F401  (fork guard + sys.path for mcts / engine)
import labels as L

__all__ = [
    "CORPUS_CONFIG", "PROBE_ROLLOUT_SEED", "snapshot_rng_seed", "obs_fingerprint",
    "pool_job", "arm_job", "snapshots_for_seed",
]

# The `labels_full` campaign config (mp/results/labels_full.json -> "config"), verbatim.
# Anything the POC changes is passed per job, never here.
CORPUS_CONFIG = {
    "n_states": 12, "n_rollouts": 8, "policy": "ev", "budget": "fast",
    "epsilon_selfplay": 0.1, "epsilon_rollout": 0.02, "encoder": "v2", "max_ante": 12,
    "deck_key": "b_red", "stake": 1, "lives": 4, "policy_seed": 0, "rollout_seed": 0,
    "shop_tier": "rules",
}

# The probe (cheap 2-rollout) labels use a DIFFERENT rollout-seed base than the arm labels.
# `label_both` draws rollout seeds `seed * 1_000_003 + i`, so sharing the base would make the
# probe's rollouts a literal SUBSET of the arm's 8 — the error proxy would then select states
# whose noise persists into the very label it is judged on.  Independent streams keep the
# proxy an honest (noisy) second opinion and let the noise-chasing effect show up as it
# really is.
PROBE_ROLLOUT_SEED = 918_273_645


def snapshot_rng_seed(seed: str, policy_seed: int = 0) -> int:
    """Process-stable replacement for ``sample_states``'s default ``hash((seed, policy_seed))``."""
    h = hashlib.sha1(f"w-active-snap:{seed}:{int(policy_seed)}".encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big")


def obs_fingerprint(obs: dict) -> str:
    """Order-independent sha1 over an encoded observation (used to pin reconstruction)."""
    h = hashlib.sha1()
    for k in sorted(obs):
        a = np.ascontiguousarray(obs[k])
        h.update(k.encode("utf-8"))
        h.update(str(a.dtype).encode("utf-8"))
        h.update(str(a.shape).encode("utf-8"))
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def _factories(cfg: dict) -> tuple:
    """(self-play factory, rollout factory, encoder) — exactly as ``labels.label_job`` builds them."""
    sp = L.make_policy_factory(cfg["policy"], budget=cfg["budget"],
                               epsilon=cfg["epsilon_selfplay"], shop_tier=cfg["shop_tier"])
    ro = L.make_policy_factory(cfg["policy"], budget=cfg["budget"],
                               epsilon=cfg["epsilon_rollout"], shop_tier=cfg["shop_tier"])
    return sp, ro, L.make_encoder(cfg["encoder"])


def snapshots_for_seed(seed: str, cfg: dict, sp_factory) -> list:
    """``labels.sample_states`` with the corpus config and a process-stable reservoir seed."""
    return L.sample_states(
        seed, n_states=cfg["n_states"], policy_factory=sp_factory, policy=cfg["policy"],
        budget=cfg["budget"], epsilon=cfg["epsilon_selfplay"], policy_seed=cfg["policy_seed"],
        rng_seed=snapshot_rng_seed(seed, cfg["policy_seed"]), deck_key=cfg["deck_key"],
        stake=cfg["stake"], lives=cfg["lives"], max_ante=cfg["max_ante"],
    )


def _meta(s, p: int, cfg: dict, r, *, extra: Optional[dict] = None) -> dict:
    """The row meta ``label_job`` writes, so shards from this POC are drop-in for dataset.py."""
    m = {"seed": s.seed, "step": s.step, "player": p, "actor": s.actor, "kind": s.kind,
         "ante": s.ante, "ci": r.ci, "n_rollouts": r.n, "trunc_frac": r.trunc_frac,
         "determinized": r.determinized,
         "lives": [s.match.games[q].lives for q in (0, 1)],
         "outcomes": [round(o, 4) for o in r.outcomes],
         "selfplay": {k: v for k, v in s.selfplay.items() if k != "decisions_seen"},
         "n_rollouts_cfg": r.n, "epsilon_rollout": cfg["epsilon_rollout"],
         "independent": False, "forced": r.forced, "shop_tier": cfg["shop_tier"],
         "budget": cfg["budget"]}
    if extra:
        m.update(extra)
    return m


# ── candidate pool ────────────────────────────────────────────────────────────────

def pool_job(payload: dict) -> dict:
    """One fresh seed -> ``n_states`` snapshots, encoded for both perspectives, each with a
    cheap ``n_probe``-rollout label.

    payload: ``seed`` (str), ``n_probe`` (int, default 2), plus any ``CORPUS_CONFIG`` override.
    Returns ``{"rows": [{"obs", "y_probe", "ci_probe", "meta"} ...], "timing": {...}}`` where
    ``meta`` carries ``(seed, step, player, kind, ante, obs_fp)`` — enough for ``arm_job`` to
    re-derive and verify the state.
    """
    cfg = dict(CORPUS_CONFIG, **{k: v for k, v in payload.items() if k in CORPUS_CONFIG})
    seed = str(payload["seed"])
    n_probe = int(payload.get("n_probe", 2))
    if not L.has_determinize():
        raise RuntimeError("clone_determinized (W2) missing: rollouts would be clairvoyant")
    sp, ro, encode = _factories(cfg)

    t0 = time.perf_counter()
    snaps = snapshots_for_seed(seed, cfg, sp)
    t_sp = time.perf_counter() - t0

    rows, t_lab, n_ro, dec = [], 0.0, 0, 0
    for i, s in enumerate(snaps):
        base = PROBE_ROLLOUT_SEED + s.step * 31 + i
        r0, r1 = L.label_both(s.match, n_rollouts=n_probe, seed=base, policy_factory=ro,
                              max_ante=cfg["max_ante"])
        t_lab += r0.seconds
        n_ro += r0.n
        dec += r0.decisions
        for p, r in ((0, r0), (1, r1)):
            obs = encode(s.match, p)
            rows.append({"obs": obs, "y_probe": r.y, "ci_probe": r.ci,
                         "meta": _meta(s, p, cfg, r, extra={"obs_fp": obs_fingerprint(obs),
                                                            "probe_rollouts": r.n})})
    return {"rows": rows, "selfplay": snaps[0].selfplay if snaps else {},
            "timing": {"selfplay_s": t_sp, "label_s": t_lab, "n_snapshots": len(snaps),
                       "n_rollouts": n_ro, "rollout_decisions": dec,
                       "total_s": time.perf_counter() - t0}}


# ── arm labelling ─────────────────────────────────────────────────────────────────

def arm_job(payload: dict) -> dict:
    """Label the SELECTED states of one seed at the full ``n_rollouts``.

    payload: ``seed``, ``steps`` (list of ``match.steps`` values chosen by the selector),
    ``fps`` (optional list of ``obs_fp`` per (step, player=0) to verify reconstruction),
    plus ``CORPUS_CONFIG`` overrides.  Returns the same row shape ``labels.label_job`` does
    (``{"obs", "y", "meta"}``) so ``labels.rows_from_result`` / ``dataset.save_shard`` accept
    it unchanged.

    The arms are labelled in ONE pass over the UNION of their selections: a state chosen by
    two arms is labelled once and shared, which is both cheaper and a tighter control (an
    overlapping state cannot differ between arms by label noise).
    """
    cfg = dict(CORPUS_CONFIG, **{k: v for k, v in payload.items() if k in CORPUS_CONFIG})
    seed = str(payload["seed"])
    want = {int(s) for s in payload["steps"]}
    fps = {int(k): v for k, v in (payload.get("fps") or {}).items()}
    n_rollouts = int(cfg["n_rollouts"])
    if not L.has_determinize():
        raise RuntimeError("clone_determinized (W2) missing: rollouts would be clairvoyant")
    sp, ro, encode = _factories(cfg)

    t0 = time.perf_counter()
    snaps = snapshots_for_seed(seed, cfg, sp)
    t_sp = time.perf_counter() - t0

    by_step = {s.step: (i, s) for i, s in enumerate(snaps)}
    missing = sorted(want - set(by_step))
    rows, t_lab, n_ro, dec, mismatched = [], 0.0, 0, 0, []
    for step in sorted(want & set(by_step)):
        i, s = by_step[step]
        obs0 = encode(s.match, 0)
        if step in fps and obs_fingerprint(obs0) != fps[step]:
            mismatched.append(step)          # reconstruction drifted: refuse to label it
            continue
        # the corpus rollout-seed formula (labels.label_job), so an arm label is exactly the
        # label the standard pipeline would have produced for this state.
        base = cfg["rollout_seed"] * 7919 + s.step * 31 + i
        r0, r1 = L.label_both(s.match, n_rollouts=n_rollouts, seed=base, policy_factory=ro,
                              max_ante=cfg["max_ante"])
        t_lab += r0.seconds
        n_ro += r0.n
        dec += r0.decisions
        for p, r in ((0, r0), (1, r1)):
            obs = obs0 if p == 0 else encode(s.match, p)
            rows.append({"obs": obs, "y": r.y, "meta": _meta(s, p, cfg, r)})
    return {"rows": rows, "missing_steps": missing, "mismatched_steps": mismatched,
            "timing": {"selfplay_s": t_sp, "label_s": t_lab, "n_states": len(rows) // 2,
                       "n_rollouts": n_ro, "rollout_decisions": dec,
                       "total_s": time.perf_counter() - t0}}
