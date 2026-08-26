"""stage_final.py — retrain one V per arm (several paired seeds), evaluate, write the results.

    python ev/active_poc/stage_final.py --seeds 5 --arm-only-seeds 3

Each arm trains on ``base-12k + that arm's rows`` with the recipe and step count fixed by
``stage_base`` (``S*``), identical across arms; only the added rows differ.

Pairing.  ``base`` rows are always concatenated FIRST and every arm has the same row count,
so for a given training seed the epoch permutation (``default_rng([seed, epoch])`` over the
same length) is identical across arms and the initial weights are identical — the arms differ
only in which rows occupy the last ``K`` slots.  Arm-vs-arm differences are therefore reported
PAIRED by seed, which removes most of the ~0.006 BCE seed-to-seed noise that would otherwise
swamp an effect this size.

Secondary, more sensitive: an ``arm-only`` comparison that trains on the arm's rows ALONE
(no base).  Adding 2.4k rows to 12.3k is a ~20% data change whose total effect is small; the
arm-only view removes the dilution and shows the acquisition rules' relative teaching value
much more clearly, at the cost of being a different (smaller-data) regime.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
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

from dataset import LabelDataset  # noqa: E402
from active_poc import corpus as C  # noqa: E402
from active_poc import select as S  # noqa: E402
from active_poc import training as T  # noqa: E402

METRICS = ("bce", "brier", "auc", "ece")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="ev/runs/active_poc")
    ap.add_argument("--arm-shards", default="ev/runs/active_poc/arms/shards")
    ap.add_argument("--seeds", type=int, default=5, help="paired training seeds per arm")
    ap.add_argument("--seed0", type=int, default=201)
    ap.add_argument("--arm-only-seeds", type=int, default=3)
    ap.add_argument("--pool-shards", default="ev/runs/active_poc/pool/shards")
    ap.add_argument("--results", default="results/active_poc_2026-08-25")
    return ap


def _agg(vals: list) -> dict:
    return {"mean": float(st.fmean(vals)), "sd": float(st.stdev(vals)) if len(vals) > 1 else 0.0,
            "sem": float(st.stdev(vals) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0,
            "n": len(vals), "values": [float(v) for v in vals]}


def diagnostics(union: LabelDataset, pool_shards, ckpts, arms: dict) -> dict:
    """Does either acquisition signal actually track V's error, or just label noise?

    Now that the selected states carry an 8-rollout label ``y8`` drawn from a rollout stream
    INDEPENDENT of the 2-rollout probe, the proxy can be decomposed on real data:

    * ``err_proxy    = |mean_V − y_probe|``  — what the rule ranked on (2 rollouts, sd ~0.35)
    * ``realized_err = |mean_V − y8|``       — a much better estimate of V's actual error

    If the proxy were selecting genuine model error the two would correlate strongly across
    states; if it is mostly ranking the probe's own noise, the correlation collapses and the
    arm's realized error is barely above uniform's.  ``disagreement`` is scored the same way
    for comparison.
    """
    pool, _dropped = C.drop_holdout_seeds(LabelDataset.load(pool_shards))
    probe = {(str(s), int(t), int(p)): float(y) for s, t, p, y in zip(
        pool.columns["seed"].tolist(), pool.columns["step"].tolist(),
        pool.columns["player"].tolist(), pool.y.tolist())}
    _P, mean_v, std_v = T.ensemble_scores(ckpts, union)
    keys = list(zip(union.columns["seed"].tolist(), union.columns["step"].tolist(),
                    union.columns["player"].tolist()))
    have = np.asarray([(str(s), int(t), int(p)) in probe for s, t, p in keys], dtype=bool)
    yp = np.asarray([probe.get((str(s), int(t), int(p)), np.nan) for s, t, p in keys])
    proxy = np.abs(mean_v - yp)
    realized = np.abs(mean_v - union.y)
    m = have & np.isfinite(proxy)
    out = {"n_rows": int(m.sum()),
           "corr_err_proxy_vs_realized": float(np.corrcoef(proxy[m], realized[m])[0, 1]),
           "corr_disagreement_vs_realized": float(np.corrcoef(std_v[m], realized[m])[0, 1]),
           "corr_err_proxy_vs_disagreement": float(np.corrcoef(proxy[m], std_v[m])[0, 1]),
           "mean_realized_err_all": float(realized[m].mean()), "by_arm": {}}
    for a, states in arms.items():
        want = {(str(s), int(t)) for s, t in states}
        sel = np.asarray([(str(s), int(t)) in want for s, t, _p in keys], dtype=bool) & m
        if not sel.any():
            continue
        out["by_arm"][a] = {
            "n_rows": int(sel.sum()),
            "err_proxy_mean": float(proxy[sel].mean()),
            "realized_err_mean": float(realized[sel].mean()),
            "disagreement_mean": float(std_v[sel].mean()),
            "y8_sd": float(union.y[sel].std()),
        }
    return out


def _sweep_checkpoints(run_dir: Path) -> None:
    """Drop the 60 MB checkpoints once a run has been evaluated (29 runs x 2 would be ~3.5 GB;
    train.jsonl keeps the curve, and every run is reproducible from its seed)."""
    for p in Path(run_dir).glob("*.pt"):
        try:
            p.unlink()
        except OSError:
            pass


def _paired(arm_vals: list, ref_vals: list) -> dict:
    d = [a - b for a, b in zip(arm_vals, ref_vals)]
    sd = st.stdev(d) if len(d) > 1 else 0.0
    sem = sd / math.sqrt(len(d)) if len(d) > 1 else 0.0
    return {"mean_delta": float(st.fmean(d)), "sd": float(sd), "sem": float(sem),
            "t": float(st.fmean(d) / sem) if sem > 0 else float("nan"),
            "df": len(d) - 1, "deltas": [float(x) for x in d]}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    manifest = json.loads((out / "base_manifest.json").read_text(encoding="utf-8"))
    spec = json.loads((out / "arms.json").read_text(encoding="utf-8"))
    s_star = int(manifest["s_star"])

    base, holdout = C.build_base_and_holdout(manifest["corpus_shards"], base_frac=manifest["base_frac"])
    hard_mask = np.load(out / "holdout_hard_mask.npy")
    assert len(hard_mask) == len(holdout), (
        f"hard-stratum mask ({len(hard_mask)}) does not match the holdout ({len(holdout)}); "
        "stage_select and stage_final must rebuild the same split")
    union = LabelDataset.load(args.arm_shards)
    if not len(union):
        raise SystemExit(f"no arm rows under {args.arm_shards}")
    have = {(str(s), int(t)) for s, t in zip(union.columns["seed"].tolist(), union.columns["step"].tolist())}
    print(f"=== labelled union: {len(union)} rows / {len(have)} states "
          f"(requested {spec['arm_states'] * 3} arm-slots) ===")

    # arms, in their stored (rank / shuffle) order, trimmed to a COMMON size
    ordered = {a: [(str(s), int(t)) for s, t in spec["arms"][a] if (str(s), int(t)) in have]
               for a in S.ARMS}
    k = min(len(v) for v in ordered.values())
    arms = {a: v[:k] for a, v in ordered.items()}
    print(f"  available per arm: { {a: len(v) for a, v in ordered.items()} } -> common size {k} states")

    arm_ds = {a: C.select_rows_by_state(union, sts) for a, sts in arms.items()}
    # the pairing claim: every arm adds exactly the same NUMBER of rows, so for a given
    # training seed the epoch permutation and the initial weights are identical across arms
    for a, d in arm_ds.items():
        assert len(d) == 2 * k, f"arm {a}: {len(d)} rows for {k} states (expected {2 * k})"
    const = float(base.y.mean())
    label_noise = {}
    for a in S.ARMS:
        d = arm_ds[a]
        ci = d.columns["ci"]
        label_noise[a] = {"n_rows": len(d), "n_states": len(d) // 2,
                          "ci_mean": float(np.nanmean(ci)),
                          "ci_sd": float(np.nanstd(ci)),
                          "noise_floor_brier": float(((np.nan_to_num(ci) / 1.96) ** 2).mean()),
                          "y_mean": float(d.y.mean()), "y_sd": float(d.y.std()),
                          "by_kind": C.kind_counts(d),
                          "trunc_frac": float(d.columns["trunc_frac"].mean())}
        print(f"  arm {a:13s} {len(d):5d} rows  ci {label_noise[a]['ci_mean']:.4f}  "
              f"y_sd {label_noise[a]['y_sd']:.4f}  kinds {label_noise[a]['by_kind']}")

    print("\n=== diagnostics: does either signal track V's real error? ===", flush=True)
    diag = diagnostics(union, args.pool_shards, [m["checkpoint"] for m in manifest["ensemble"]], arms)
    print(f"  corr(err_proxy, |V-y8|)     = {diag['corr_err_proxy_vs_realized']:+.3f}")
    print(f"  corr(disagreement, |V-y8|)  = {diag['corr_disagreement_vs_realized']:+.3f}")
    for a, v in diag["by_arm"].items():
        print(f"  {a:13s} proxy {v['err_proxy_mean']:.4f} -> realized |V-y8| "
              f"{v['realized_err_mean']:.4f}  (disagreement {v['disagreement_mean']:.4f})")

    train_sets = {a: C.concat(base, arm_ds[a]) for a in S.ARMS}
    train_sets["base_only"] = base
    for a, d in train_sets.items():
        print(f"  train set {a:13s} {len(d):6d} rows")

    seeds = [args.seed0 + i for i in range(args.seeds)]
    runs: dict = {a: [] for a in train_sets}
    for a in ("base_only",) + S.ARMS:
        for sd in seeds:
            rd = out / "final" / f"{a}_s{sd}"
            print(f"\n=== {a} seed {sd} ({len(train_sets[a])} rows, {s_star} steps) ===", flush=True)
            summary = T.train_one(train_sets[a], holdout, rd, seed=sd, max_steps=s_star,
                                  log=lambda *x: None)
            net = T.load_net(summary["checkpoint"])
            m = T.evaluate_full(net, holdout, const=const, hard_mask=hard_mask)
            del net
            _sweep_checkpoints(rd)
            rec = {"seed": sd, "n_train": len(train_sets[a]),
                   **{k2: m[k2] for k2 in METRICS},
                   "acc": m["acc@0.5"], "p_sd": m["p_sd"],
                   "by_kind": m.get("by_kind", {}), "hard": m.get("hard_stratum", {}),
                   "noise_floor_brier": m.get("noise_floor_brier")}
            runs[a].append(rec)
            print(f"  bce {m['bce']:.4f}  brier {m['brier']:.4f}  auc {m['auc']:.4f}  "
                  f"ece {m['ece']:.4f}  hard-bce {m.get('hard_stratum', {}).get('bce', float('nan')):.4f}",
                  flush=True)

    # ── arm-only (secondary) ──
    arm_only: dict = {a: [] for a in S.ARMS}
    ao_reg = None
    if args.arm_only_seeds > 0:
        ao_seeds = [args.seed0 + 50 + i for i in range(args.arm_only_seeds)]
        # derive the small-data step count on the CONTROL arm, so it cannot favour a treatment
        print(f"\n=== arm-only regime probe on the uniform arm ({len(arm_ds['uniform'])} rows) ===",
              flush=True)
        ao_reg = T.derive_regime(arm_ds["uniform"], holdout, out / "probe_armonly",
                                 probe_steps=400, seed=0, log=lambda *x: None)
        ao_steps = int(ao_reg["s_star"])
        _sweep_checkpoints(out / "probe_armonly")
        print(f"  arm-only S* = {ao_steps} (bce {ao_reg['best_bce']:.4f})"
              f"{'  [AT EDGE]' if ao_reg['at_edge'] else ''}", flush=True)
        for a in S.ARMS:
            for sd in ao_seeds:
                rd = out / "final_armonly" / f"{a}_s{sd}"
                summary = T.train_one(arm_ds[a], holdout, rd, seed=sd, max_steps=ao_steps,
                                      log=lambda *x: None)
                net = T.load_net(summary["checkpoint"])
                m = T.evaluate_full(net, holdout, const=const, hard_mask=hard_mask)
                del net
                _sweep_checkpoints(rd)
                arm_only[a].append({"seed": sd, **{k2: m[k2] for k2 in METRICS}})
                print(f"  {a:13s} s{sd}: bce {m['bce']:.4f} brier {m['brier']:.4f} auc {m['auc']:.4f}",
                      flush=True)

    # ── aggregate ──
    agg = {a: {k2: _agg([r[k2] for r in runs[a]]) for k2 in METRICS} for a in runs}
    for a in runs:
        agg[a]["hard_bce"] = _agg([r["hard"].get("bce", float("nan")) for r in runs[a]])
        agg[a]["hard_brier"] = _agg([r["hard"].get("brier", float("nan")) for r in runs[a]])
        agg[a]["n_train"] = runs[a][0]["n_train"]
        kinds = sorted({k2 for r in runs[a] for k2 in r["by_kind"]})
        agg[a]["by_kind_bce"] = {k2: _agg([r["by_kind"][k2]["bce"] for r in runs[a] if k2 in r["by_kind"]])
                                 for k2 in kinds}
    paired = {}
    for a in S.ARMS:
        if a == "uniform":
            continue
        paired[f"{a}_vs_uniform"] = {
            k2: _paired([r[k2] for r in runs[a]], [r[k2] for r in runs["uniform"]]) for k2 in METRICS}
        paired[f"{a}_vs_uniform"]["hard_bce"] = _paired(
            [r["hard"].get("bce", float("nan")) for r in runs[a]],
            [r["hard"].get("bce", float("nan")) for r in runs["uniform"]])
    for a in S.ARMS:
        paired[f"{a}_vs_base_only"] = {
            k2: _paired([r[k2] for r in runs[a]], [r[k2] for r in runs["base_only"]]) for k2 in METRICS}
    agg_ao = {a: {k2: _agg([r[k2] for r in arm_only[a]]) for k2 in METRICS} for a in arm_only if arm_only[a]}
    paired_ao = {}
    for a in S.ARMS:
        if a != "uniform" and arm_only.get(a) and arm_only.get("uniform"):
            paired_ao[f"{a}_vs_uniform"] = {
                k2: _paired([r[k2] for r in arm_only[a]], [r[k2] for r in arm_only["uniform"]])
                for k2 in METRICS}

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "design": {"base_rows": len(base), "holdout_rows": len(holdout),
                   "arm_states": k, "arm_rows": len(arm_ds["uniform"]),
                   "s_star": s_star, "training_seeds": seeds,
                   "recipe": manifest["recipe"], "const_predictor": const},
        "base_manifest": {k2: manifest[k2] for k2 in ("base", "holdout", "regime", "s_star", "ensemble")},
        "selection": {k2: spec[k2] for k2 in ("pool", "score_stats", "arm_states", "cap_mult",
                                              "profiles", "overlap", "hard_stratum")},
        "label_noise": label_noise, "diagnostics": diag,
        "results": agg, "runs": runs, "paired": paired,
        "arm_only": {"aggregate": agg_ao, "paired": paired_ao, "runs": arm_only,
                     "regime": {k2: v for k2, v in (ao_reg or {}).items() if k2 != "summary"}},
    }
    res = Path(args.results)
    res.parent.mkdir(parents=True, exist_ok=True)
    res.with_suffix(".json").write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    print(f"\n=== wrote {res.with_suffix('.json')} ===")
    _print_table(agg, paired, label_noise, agg_ao)
    return 0


def _print_table(agg, paired, noise, agg_ao) -> None:
    print(f"\n{'arm':14s} {'rows':>6s} {'BCE':>16s} {'Brier':>16s} {'AUC':>15s} {'ECE':>15s} {'ci':>7s}")
    for a in ("base_only",) + S.ARMS:
        r = agg[a]
        n = noise.get(a, {}).get("ci_mean", float("nan"))
        print(f"{a:14s} {r['n_train']:6d} "
              f"{r['bce']['mean']:.4f}+-{r['bce']['sem']:.4f} "
              f"{r['brier']['mean']:.4f}+-{r['brier']['sem']:.4f} "
              f"{r['auc']['mean']:.4f}+-{r['auc']['sem']:.4f} "
              f"{r['ece']['mean']:.4f}+-{r['ece']['sem']:.4f} {n:7.4f}")
    print("\npaired deltas (negative BCE/Brier = better):")
    for k, v in paired.items():
        print(f"  {k:28s} dBCE {v['bce']['mean_delta']:+.4f}+-{v['bce']['sem']:.4f} "
              f"(t={v['bce']['t']:+.2f}, df={v['bce']['df']})  "
              f"dBrier {v['brier']['mean_delta']:+.5f}+-{v['brier']['sem']:.5f}  "
              f"dAUC {v['auc']['mean_delta']:+.4f}")
    if agg_ao:
        print("\narm-only (secondary):")
        for a, r in agg_ao.items():
            print(f"  {a:14s} bce {r['bce']['mean']:.4f}+-{r['bce']['sem']:.4f}  "
                  f"brier {r['brier']['mean']:.4f}  auc {r['auc']['mean']:.4f}")


if __name__ == "__main__":
    sys.exit(main())
