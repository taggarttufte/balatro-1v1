"""stage_base.py — base corpus, training regime, and the 3-member scoring ensemble.

    python mp/ev/active_poc/stage_base.py --corpus <labels_full/shards> --out mp/ev/runs/active_poc

1. Split the existing 51k corpus by the STANDARD seed-hash rule -> the full evaluation
   holdout (never trained on) and the training side.
2. Subsample the training side to ~12k rows by an independent seed hash = the base corpus.
3. Probe run on the base corpus to re-derive the early-stopping step ``S*`` for 12k rows
   (the known-good regime was step 1250 / epoch 7 at 45.9k rows).
4. Train 3 ``set_value_net`` members on the base corpus with identical config and different
   seeds (different init AND different batch order) — the disagreement scorer.

Writes ``<out>/base_manifest.json`` (the seed lists, the corpus summaries, ``S*``, the
probe curve and the ensemble checkpoints).  GPU runs are sequential and polite.
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

from active_poc import corpus as C  # noqa: E402
from active_poc import training as T  # noqa: E402

DEFAULT_CORPUS = str(Path(__file__).resolve().parents[1] / "runs" / "labels_full" / "shards")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", default=DEFAULT_CORPUS, help="labels_full shards (read-only)")
    ap.add_argument("--out", default="mp/ev/runs/active_poc")
    ap.add_argument("--base-frac", type=float, default=C.DEFAULT_BASE_FRAC)
    ap.add_argument("--probe-steps", type=int, default=800)
    ap.add_argument("--ensemble", type=int, default=3)
    ap.add_argument("--ensemble-seed0", type=int, default=101)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"=== base corpus from {args.corpus} ===", flush=True)
    base, holdout = C.build_base_and_holdout(args.corpus, base_frac=args.base_frac)
    print(f"  base    {len(base):6d} rows / {len(base.seeds()):5d} seeds  {C.kind_counts(base)}")
    print(f"  holdout {len(holdout):6d} rows / {len(holdout.seeds()):5d} seeds (STANDARD rule, never trained on)")
    overlap = set(base.seeds()) & set(holdout.seeds())
    assert not overlap, f"base/holdout seed overlap: {sorted(overlap)[:5]}"

    print(f"\n=== regime probe: {args.probe_steps} steps on the base corpus ===", flush=True)
    reg = T.derive_regime(base, holdout, out / "probe", probe_steps=args.probe_steps, seed=0)
    print(f"  S* = {reg['s_star']} (bce {reg['best_bce']:.4f}, brier {reg['best_brier']:.4f}, "
          f"auc {reg['best_auc']:.3f}){'  [AT EDGE - bracket too short]' if reg['at_edge'] else ''}")

    s_star = int(reg["s_star"])
    members = []
    for i in range(args.ensemble):
        seed = args.ensemble_seed0 + i
        rd = out / f"ens_{i}"
        print(f"\n=== ensemble member {i} (seed {seed}, {s_star} steps) ===", flush=True)
        summary = T.train_one(base, holdout, rd, seed=seed, max_steps=s_star)
        fe = summary.get("final_eval", {})
        members.append({"i": i, "seed": seed, "run_dir": str(rd),
                        "checkpoint": summary["checkpoint"], "step": summary["step"],
                        "bce": fe.get("bce"), "brier": fe.get("brier"), "auc": fe.get("auc")})
        print(f"  member {i}: bce {fe.get('bce'):.4f} brier {fe.get('brier'):.4f} auc {fe.get('auc'):.3f}")

    manifest = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "corpus_shards": str(args.corpus),
        "base_frac": args.base_frac, "base_salt": C.BASE_SALT,
        "holdout_frac": C.STANDARD_HOLDOUT_FRAC,
        "base": C.summarise(base, "base"), "holdout": C.summarise(holdout, "holdout"),
        "base_seeds": base.seeds(), "holdout_seeds": holdout.seeds(),
        "regime": {k: v for k, v in reg.items() if k != "summary"},
        "recipe": {k: v for k, v in T.RECIPE.items()},
        "s_star": s_star, "ensemble": members,
    }
    (out / "base_manifest.json").write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    print(f"\n=== wrote {out / 'base_manifest.json'} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
