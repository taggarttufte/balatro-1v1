"""stage_select.py — score the candidate pool with the ensemble and pick the three arms.

    python ev/active_poc/stage_select.py --arm-states 1200

Loads the pool shards, runs the 3 base-corpus ensemble members over every pool row (GPU,
sequential, one member resident at a time), collapses to per-state scores, and writes
``<out>/arms.json``: the selected ``(seed, step, obs_fp)`` per arm, the per-arm profiles,
the pairwise overlap and the score distributions.

Also computes the HOLDOUT disagreement, so the "high-disagreement stratum" the arms are
finally judged on is defined once, by the same ensemble, independent of any arm.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
EV_ROOT = HERE.parent
MP_ROOT = EV_ROOT.parent
for _p in (str(EV_ROOT), str(MP_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _bootstrap  # noqa: E402,F401
import numpy as np  # noqa: E402

from dataset import LabelDataset, seed_in_holdout  # noqa: E402
from active_poc import corpus as C  # noqa: E402
from active_poc import select as S  # noqa: E402
from active_poc import training as T  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="ev/runs/active_poc")
    ap.add_argument("--pool", default="ev/runs/active_poc/pool/shards")
    ap.add_argument("--arm-states", type=int, default=1200)
    ap.add_argument("--cap-mult", type=float, default=1.5, help="light stratification cap")
    ap.add_argument("--uniform-rng", type=int, default=4242)
    ap.add_argument("--hard-quantile", type=float, default=0.75,
                    help="holdout rows above this disagreement quantile = the hard stratum")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    manifest = json.loads((out / "base_manifest.json").read_text(encoding="utf-8"))
    ckpts = [m["checkpoint"] for m in manifest["ensemble"]]

    pool_raw = LabelDataset.load(args.pool)
    if not len(pool_raw):
        raise SystemExit(f"no pool rows under {args.pool}")
    # Hard guarantee: nothing an arm can select may be an evaluation-holdout seed.  The pool
    # was generated with a seed filter that tested the RAW string, but the engine canonicalises
    # '0' -> 'O' (corpus.canonical_seed), so a few generated seeds land in the holdout after
    # canonicalisation.  They are dropped here, before any selection can see them.
    pool, leaked = C.drop_holdout_seeds(pool_raw, C.STANDARD_HOLDOUT_FRAC)
    if leaked:
        print(f"  [holdout hygiene] dropped {len(pool_raw) - len(pool)} rows from "
              f"{len(leaked)} canonicalised-into-holdout seeds: {leaked[:4]}...")
    assert not [s for s in pool.seeds() if seed_in_holdout(s, C.STANDARD_HOLDOUT_FRAC)]
    print(f"=== pool: {len(pool)} rows / {len(pool) // 2} states / {len(pool.seeds())} seeds ===")
    print(f"  kinds {C.kind_counts(pool)}")

    print(f"=== scoring with {len(ckpts)} ensemble members ===", flush=True)
    P, mean_v, std_v = T.ensemble_scores(ckpts, pool)
    print(f"  V mean {mean_v.mean():.4f}  disagreement(sd) mean {std_v.mean():.4f} "
          f"p50 {np.median(std_v):.4f} p90 {np.quantile(std_v, 0.9):.4f} max {std_v.max():.4f}")

    scores = S.state_scores(pool, mean_v, std_v)
    print(f"  {len(scores)} states scored")
    dis = np.array([r["disagreement"] for r in scores.values()])
    err = np.array([r["err_proxy"] for r in scores.values()])
    corr = float(np.corrcoef(dis, err)[0, 1])
    print(f"  disagreement: mean {dis.mean():.4f} p90 {np.quantile(dis, 0.9):.4f}")
    print(f"  err_proxy   : mean {err.mean():.4f} p90 {np.quantile(err, 0.9):.4f}")
    print(f"  corr(disagreement, err_proxy) = {corr:.3f}")

    n = min(args.arm_states, len(scores))
    arms = {
        "disagreement": S.stratified_topk(scores, "disagreement", n, cap_mult=args.cap_mult),
        "err_proxy": S.stratified_topk(scores, "err_proxy", n, cap_mult=args.cap_mult),
        "uniform": S.uniform_sample(scores, n, rng_seed=args.uniform_rng),
    }
    profiles = {a: S.arm_profile(scores, st, a if a != "uniform" else None) for a, st in arms.items()}
    overlap = S.overlap_table(arms)
    for a in S.ARMS:
        p = profiles[a]
        print(f"  arm {a:13s} n={p['n_states']:5d} dis {p['disagreement_mean']:.4f} "
              f"err {p['err_proxy_mean']:.4f} ci_probe {p['ci_probe_mean']:.3f} kinds {p['by_kind']}")
    for k, v in overlap.items():
        print(f"  overlap {k}: {v['intersection']} states (jaccard {v['jaccard']:.3f})")

    # the hard stratum of the HOLDOUT (arm-independent, defined once)
    print("=== holdout disagreement stratum ===", flush=True)
    _base, holdout = C.build_base_and_holdout(manifest["corpus_shards"],
                                              base_frac=manifest["base_frac"])
    _Ph, _mh, std_h = T.ensemble_scores(ckpts, holdout)
    thr = float(np.quantile(std_h, args.hard_quantile))
    hard = std_h >= thr
    print(f"  threshold sd >= {thr:.4f} (q{args.hard_quantile}) -> {int(hard.sum())}/{len(holdout)} rows")

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "pool": {"shards": str(args.pool), "n_rows": len(pool), "n_states": len(scores),
                 "n_seeds": len(pool.seeds()), "by_kind": C.kind_counts(pool),
                 "rows_dropped_holdout_hygiene": len(pool_raw) - len(pool),
                 "seeds_dropped_holdout_hygiene": leaked,
                 "probe_rollouts": int(pool.columns["n_rollouts"][0]),
                 "ci_probe_mean": float(np.nanmean(pool.columns["ci"]))},
        "ensemble_checkpoints": ckpts,
        "score_stats": {"disagreement": {"mean": float(dis.mean()), "sd": float(dis.std()),
                                         "p50": float(np.median(dis)), "p90": float(np.quantile(dis, 0.9)),
                                         "max": float(dis.max())},
                        "err_proxy": {"mean": float(err.mean()), "sd": float(err.std()),
                                      "p50": float(np.median(err)), "p90": float(np.quantile(err, 0.9)),
                                      "max": float(err.max())},
                        "corr_dis_err": corr},
        "arm_states": n, "cap_mult": args.cap_mult, "uniform_rng": args.uniform_rng,
        "arms": {a: [[s, t] for (s, t) in st] for a, st in arms.items()},
        "fps": {f"{s}|{t}": scores[(s, t)]["fp"] for st in arms.values() for (s, t) in st},
        "profiles": profiles, "overlap": overlap,
        "hard_stratum": {"quantile": args.hard_quantile, "threshold": thr,
                         "n_rows": int(hard.sum()), "n_holdout": len(holdout)},
    }
    (out / "arms.json").write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    np.save(out / "holdout_hard_mask.npy", hard)
    union = sorted({tuple(s) for st in arms.values() for s in st})
    print(f"=== wrote {out / 'arms.json'}: union to label = {len(union)} states "
          f"({sum(len(v) for v in arms.values())} arm-slots, "
          f"{100 * (1 - len(union) / sum(len(v) for v in arms.values())):.1f}% saved by overlap) ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
