"""
tournament_v.py — paired-by-seed MLB matches: ``EVPlayer(value_fn=V)`` vs ``EVPlayer(value_fn=None)``
(Phase 5 rev 2, W5; gate 4's end-to-end check).

    python ev/scripts/tournament_v.py --checkpoint ev/runs/v_s2/latest.pt --seeds default:30 \
        --workers 4 --name v_s2

One pool job = one seed = TWO matches (V in seat 0, then V in seat 1) through
``eval.common.play_1v1`` (``MLBMatch.play_out``, canonical alternation), so the seat is
balanced within the pair.  Reports V's win count / rate with a Wilson 95% interval, the
per-seat split, mean lives margin, mean final ante, and the V-call count / errors.  Writes
``results/tournament_v_<name>.json``.  Nothing in eval or tournament is edited.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
EV_ROOT = HERE.parent
MP_ROOT = EV_ROOT.parent
for _p in (str(EV_ROOT), str(MP_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _bootstrap  # noqa: E402,F401
import workers as W  # noqa: E402

RESULTS_DIR = MP_ROOT / "results"


def tournament_job(payload: dict) -> dict:
    """Both seat orders on one seed.  Loads the checkpoint itself (workers share nothing)."""
    import torch
    torch.set_num_threads(int(payload.get("threads", 1)))
    import labels as L
    import match_player as MPL
    from eval.common import play_1v1
    net, enc = MPL.load_value(payload["checkpoint"], device=payload.get("device", "cpu"))
    seed = payload["seed"]
    budget = payload.get("budget", "fast")
    base_seed = int(payload.get("policy_seed", 0))
    out = {"seed": seed, "matches": []}
    for v_seat in (0, 1):
        mp = MPL.MatchAwareEVPlayer(net, enc, device=payload.get("device", "cpu"), budget=budget,
                                    seed=base_seed + 11 * v_seat, epsilon=0.0)
        pol_v = mp.policy()
        pol_0 = L.make_policy_factory("ev", budget=budget, epsilon=0.0)(base_seed + 7 + v_seat, 1 - v_seat)
        pols = [pol_v, pol_0] if v_seat == 0 else [pol_0, pol_v]
        # `play_1v1` bypasses labels' rollout guard, and W3's `_rank_with_value` has no
        # anti-cycling — a V-tier shop loop burned 40k steps / 637k V calls in the first
        # smoke.  The same `_Guard` (match-signature no-progress detector) wraps both
        # policies here: `after` runs at the START of the next call, when the signature
        # already reflects the previous step.
        guard = L._Guard()
        last = {"a": None}

        def _wrap(pol):
            def g(m, p, acts, _pol=pol):
                if last["a"] is not None:
                    guard.after(m, last["a"])
                a = guard.choose(m, p, acts, _pol(m, p, acts))
                last["a"] = a
                return a
            return g
        t = time.perf_counter()
        r = play_1v1(seed, _wrap(pols[0]), _wrap(pols[1]), max_steps=int(payload.get("max_steps", 40_000)))
        out["matches"].append({
            "v_seat": v_seat, "winner": r["winner"], "v_won": (r["winner"] == v_seat),
            "lives": r["lives"], "v_lives_margin": (r["lives"][v_seat] - r["lives"][1 - v_seat]),
            "final_ante": r["final_ante"], "steps": r["steps"], "done": r["done"],
            "n_nemeses": len(r["pvp_log"]), "seconds": time.perf_counter() - t,
            "v_calls": mp.n_calls, "v_errors": mp.n_errors, "guard_forced": guard.forced,
        })
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="paired V-vs-noV tournament (W5)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seeds", default="default:30", help="'default:N' (first N ground-truth seeds) or a comma list")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=1, help="torch threads per worker")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--budget", default="fast")
    ap.add_argument("--policy-seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--run-dir", default=None, help="state file for resume (default: next to the checkpoint)")
    args = ap.parse_args(argv)

    if args.seeds.startswith("default"):
        from eval.common import DEFAULT_SEEDS
        n = int(args.seeds.split(":", 1)[1]) if ":" in args.seeds else len(DEFAULT_SEEDS)
        seeds = DEFAULT_SEEDS[:n]
    else:
        seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    name = args.name or Path(args.checkpoint).resolve().parent.name
    run_dir = Path(args.run_dir) if args.run_dir else Path(args.checkpoint).resolve().parent / f"tournament_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "matches.jsonl"
    payload = {"checkpoint": str(Path(args.checkpoint).resolve()), "device": args.device, "budget": args.budget,
               "policy_seed": args.policy_seed, "threads": args.threads}
    rows: list = []
    if results_path.exists():
        rows = [json.loads(l) for l in results_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    def on_result(job_id, res):
        rows.append(res)
        with open(results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(res) + "\n")
        ms = res["matches"]
        print(f"  {res['seed']}: V seat0 {'W' if ms[0]['v_won'] else 'L'} (antes {ms[0]['final_ante']}, "
              f"{ms[0]['seconds']:.0f}s), V seat1 {'W' if ms[1]['v_won'] else 'L'} (antes {ms[1]['final_ante']}, "
              f"{ms[1]['seconds']:.0f}s), V calls {ms[0]['v_calls'] + ms[1]['v_calls']}", flush=True)

    print(f"=== tournament {name}: {len(seeds)} seeds x 2 seats, {args.workers} workers, V = {args.checkpoint} ===", flush=True)
    t0 = time.perf_counter()
    summ = W.run_pool(tournament_job, ((s, dict(payload, seed=s)) for s in seeds), n_workers=args.workers,
                      on_result=on_result, state_path=run_dir / "pool.state", pause_file=run_dir / "PAUSE",
                      threads_per_worker=args.threads)
    matches = [m for r in rows for m in r["matches"]]
    n = len(matches)
    k = sum(1 for m in matches if m["v_won"])
    lo, hi = wilson(k, n)
    by_seat = {s: {"n": sum(1 for m in matches if m["v_seat"] == s),
                   "wins": sum(1 for m in matches if m["v_seat"] == s and m["v_won"])} for s in (0, 1)}
    pair_wins = sum(1 for r in rows if all(m["v_won"] for m in r["matches"]))
    pair_losses = sum(1 for r in rows if not any(m["v_won"] for m in r["matches"]))
    out = {"name": name, "checkpoint": payload["checkpoint"], "timestamp": datetime.now().isoformat(timespec="seconds"),
           "n_seeds": len(rows), "n_matches": n, "v_wins": k, "v_win_rate": (k / n if n else float("nan")),
           "wilson95": [lo, hi], "by_seat": by_seat, "pairs_v_swept": pair_wins, "pairs_v_lost_both": pair_losses,
           "mean_v_lives_margin": statistics.fmean(m["v_lives_margin"] for m in matches) if n else float("nan"),
           "mean_final_ante": statistics.fmean(max(m["final_ante"]) for m in matches) if n else float("nan"),
           "v_calls": sum(m["v_calls"] for m in matches), "v_errors": sum(m["v_errors"] for m in matches),
           "mean_match_s": statistics.fmean(m["seconds"] for m in matches) if n else float("nan"),
           "elapsed_s": time.perf_counter() - t0, "pool": summ.as_dict(), "seeds": seeds}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p = RESULTS_DIR / f"tournament_v_{name}.json"
    p.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(f"=== V won {k}/{n} ({100 * out['v_win_rate']:.0f}%, Wilson 95% [{100 * lo:.0f}%, {100 * hi:.0f}%]); "
          f"seat0 {by_seat[0]['wins']}/{by_seat[0]['n']}, seat1 {by_seat[1]['wins']}/{by_seat[1]['n']}; "
          f"swept {pair_wins} / lost both {pair_losses} of {len(rows)} pairs; V errors {out['v_errors']}; "
          f"{out['elapsed_s'] / 60:.1f} min -> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
