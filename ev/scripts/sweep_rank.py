"""ev/scripts/sweep_rank.py -- overnight hyperparameter sweep for the lever-(b) ranking loss.

For each config: train V (train_v.py subprocess, GPU, sequential), pick the best checkpoint by
held-out pair accuracy (tiebreak Brier), then score it BEHAVIORALLY with a 30-seed tournament
against the rules player (tournament_v.py). One row per config into --out (.json + .md).

    python ev/scripts/sweep_rank.py --out results/sweep_rank_2026-08-26.json \
        [--configs a,b,...] [--tournament-workers 8] [--max-steps 3000]

Configs are named entries in CONFIGS below; default = all. A failed config logs and continues.
Baseline anchor: config `center` replicates v_v2 (lam 1.0, tau 0.05, cap 4, aux on) at the
sweep's step budget -- its row calibrates every comparison.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = sys.executable

AUX7 = "money_next_shop,lives_2antes,blind_cleared,xmult_by_ante4,extract_income,cards_modified,tarots_used"

BASE = [
    "--shards", "ev/runs/pairs_v2/abs_shards", "ev/runs/labels_full/shards",
    "--pair-shards", "ev/runs/pairs_v2/shards",
    "--model", "set_value_net", "--device", "cuda", "--batch-size", "256",
    "--lr", "3e-4", "--warmup-steps", "500", "--holdout-frac", "0.1",
    "--torch-threads", "6", "--absolute-fingerprint-mode", "any",
    "--eval-every", "250", "--checkpoint-every", "250", "--keep", "15",
]

CONFIGS = {
    # name: extra flags (aux on unless stated; lam/tau/cap default to trainer defaults 1.0/0.05/4)
    "center":        ["--lam-rank", "1.0", "--tau", "0.05", "--pair-weight-cap", "4", "--aux-heads", AUX7],
    "lam0.3":        ["--lam-rank", "0.3", "--tau", "0.05", "--pair-weight-cap", "4", "--aux-heads", AUX7],
    "lam3":          ["--lam-rank", "3.0", "--tau", "0.05", "--pair-weight-cap", "4", "--aux-heads", AUX7],
    "lam10":         ["--lam-rank", "10.0", "--tau", "0.05", "--pair-weight-cap", "4", "--aux-heads", AUX7],
    "tau0.02":       ["--lam-rank", "1.0", "--tau", "0.02", "--pair-weight-cap", "4", "--aux-heads", AUX7],
    "tau0.10":       ["--lam-rank", "1.0", "--tau", "0.10", "--pair-weight-cap", "4", "--aux-heads", AUX7],
    "cap2":          ["--lam-rank", "1.0", "--tau", "0.05", "--pair-weight-cap", "2", "--aux-heads", AUX7],
    "cap8":          ["--lam-rank", "1.0", "--tau", "0.05", "--pair-weight-cap", "8", "--aux-heads", AUX7],
    "noaux":         ["--lam-rank", "1.0", "--tau", "0.05", "--pair-weight-cap", "4"],
    "auxw0.3":       ["--lam-rank", "1.0", "--tau", "0.05", "--pair-weight-cap", "4", "--aux-heads", AUX7, "--aux-weight", "0.3"],
    "newonly":       ["--lam-rank", "1.0", "--tau", "0.05", "--pair-weight-cap", "4", "--aux-heads", AUX7, "--absolute-fingerprint-mode", "new_only"],
    "lam3_tau0.02":  ["--lam-rank", "3.0", "--tau", "0.02", "--pair-weight-cap", "4", "--aux-heads", AUX7],
    "pairbatch256":  ["--lam-rank", "1.0", "--tau", "0.05", "--pair-weight-cap", "4", "--aux-heads", AUX7, "--pair-batch-size", "256"],
}


def best_eval(run_dir: Path):
    """Best eval record by (pair_acc desc, brier asc); falls back to brier if no pair metrics."""
    evs = []
    with open(run_dir / "train.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("kind") == "eval":
                evs.append(r)
    def key(r):
        p = r.get("pairs") or {}
        pa = p.get("pair_acc")
        pa = -1.0 if pa is None or pa != pa else pa
        return (pa, -r["brier"])
    return max(evs, key=key)


def run(cmd, log_tail=25):
    t0 = time.time()
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    dt = time.time() - t0
    if r.returncode != 0:
        tail = "\n".join((r.stdout + "\n" + r.stderr).splitlines()[-log_tail:])
        raise RuntimeError(f"exit {r.returncode} after {dt:.0f}s:\n{tail}")
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--configs", default=None, help="comma subset of config names (default: all)")
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--tournament-workers", type=int, default=8)
    ap.add_argument("--tournament-seeds", default="default:30")
    args = ap.parse_args()

    names = [n.strip() for n in args.configs.split(",")] if args.configs else list(CONFIGS)
    rows, t_start = [], time.time()
    for name in names:
        row = {"config": name, "flags": CONFIGS[name]}
        run_dir = REPO / "ev" / "runs" / "sweep_rank" / name
        try:
            t_train = run([PY, "ev/train_v.py", *BASE, *CONFIGS[name],
                           "--max-steps", str(args.max_steps), "--run-dir", str(run_dir)])
            ev = best_eval(run_dir)
            step = ev["step"]
            ckpt = run_dir / f"ckpt_{step:07d}.pt"
            pairs = ev.get("pairs") or {}
            row.update({
                "train_s": round(t_train), "best_step": step,
                "brier": round(ev["brier"], 5), "bce": round(ev["bce"], 5),
                "auc": round(ev["auc"], 5),
                "ece": round((ev.get("reliability") or {}).get("ece", float("nan")), 5),
                "pair_acc": pairs.get("pair_acc"), "pair_n_resolved": pairs.get("n_resolved"),
            })
            tname = f"rank_{name}"
            run([PY, "ev/scripts/tournament_v.py", "--checkpoint", str(ckpt),
                 "--seeds", args.tournament_seeds, "--workers", str(args.tournament_workers),
                 "--threads", "1", "--name", tname])
            tj = json.load(open(REPO / "results" / f"tournament_v_{tname}.json", encoding="utf-8"))
            row.update({"v_wins": tj["v_wins"], "n_matches": tj["n_matches"],
                        "lives_margin": round(tj["mean_v_lives_margin"], 3),
                        "checkpoint": str(ckpt)})
        except Exception as e:  # keep sweeping; the row records the failure
            row["error"] = str(e)[:2000]
        rows.append(row)
        done = [r for r in rows if "v_wins" in r]
        print(f"[{time.time()-t_start:7.0f}s] {name}: " +
              (f"v_wins {row['v_wins']}/{row['n_matches']} pair_acc {row['pair_acc']}"
               if "v_wins" in row else f"FAILED: {row.get('error','')[:120]}"), flush=True)
        out = {"base_flags": BASE, "max_steps": args.max_steps, "rows": rows,
               "best_so_far": max(done, key=lambda r: (r["v_wins"], r.get("pair_acc") or 0))["config"] if done else None}
        Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")

    md = ["# Ranking-loss sweep — " + time.strftime("%Y-%m-%d"), "",
          f"{args.max_steps} steps/config, ckpt by (pair_acc, brier), tournament {args.tournament_seeds} vs rules.",
          "", "| config | v_wins | pair_acc | brier | auc | ece | best step |", "|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: -(r.get("v_wins", -1))):
        if "error" in r:
            md.append(f"| {r['config']} | FAILED | | | | | |")
        else:
            md.append(f"| {r['config']} | **{r['v_wins']}/{r['n_matches']}** | {r['pair_acc']:.3f} | "
                      f"{r['brier']:.4f} | {r['auc']:.4f} | {r['ece']:.4f} | {r['best_step']} |")
    Path(args.out).with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("sweep done ->", args.out)


if __name__ == "__main__":
    main()
