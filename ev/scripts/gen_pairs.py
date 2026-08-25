"""
gen_pairs.py — the PAIR campaign driver (Phase 5 rev 2, W-PAIRS; lever (b)'s data).

    python mp/ev/scripts/gen_pairs.py --run-dir mp/ev/runs/pairs_s1 --seeds default+random:400 \
        --n-states 6 --n-worlds 8 --workers 8 --minutes 60 --name s1
    touch mp/ev/runs/pairs_s1/PAUSE          # stops submitting; in-flight jobs finish; flush
    python mp/ev/scripts/gen_pairs.py --run-dir mp/ev/runs/pairs_s1 --seeds default+random:400  # resumes

Same worker-pool conventions as ``gen_labels.py`` (which is NOT touched): one pool job =
one seed (``pairs.pair_job``), rows buffered in the main process and flushed every
``--flush-jobs`` jobs, a seed recorded in ``<run-dir>/done.ids`` only AFTER the shards
holding its rows are on disk, restart with the same ``--run-dir`` skips recorded seeds,
``PAUSE`` stops submission.

Two shard streams per run dir:
  ``shards/pair_NNNN.npz``      the frozen §5.3 pair records (``pairs.load_pair_shard``)
  ``abs_shards/shard_NNNN.npz`` the two absolute label rows each pair also yields
                                (plain ``dataset.LabelDataset.load`` reads this directory)

``--probe-jobs N``: the first N seeds run with ``--reps R`` disjoint world blocks per pair
— the DIRECT replication measurement of the variance-reduction factor (pairs.variance_report
``direct``), which costs R x the rollouts of a normal pair.

Outputs: ``<run-dir>/gen.jsonl`` (config / flush / summary), console one line per flush,
``mp/results/pairs_<name>.json`` (variance report, realised mix, throughput).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent          # mp/ev/scripts
EV_ROOT = HERE.parent                            # mp/ev
MP_ROOT = EV_ROOT.parent                         # mp
for _p in (str(EV_ROOT), str(MP_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _bootstrap  # noqa: E402,F401
import dataset as DS  # noqa: E402
import labels as L  # noqa: E402
import pairs as PR  # noqa: E402
import workers as W  # noqa: E402
from gen_labels import parse_seeds  # noqa: E402  (the shared seed-spec parser, read-only)

RESULTS_DIR = MP_ROOT / "results"
PAUSE_FILE = "PAUSE"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="pair campaign driver (W-PAIRS)")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--seeds", default="default+random:400")
    ap.add_argument("--seed-rng", type=int, default=0)
    ap.add_argument("--n-states", type=int, default=6, help="pairs attempted per self-play match")
    ap.add_argument("--n-worlds", type=int, default=PR.DEFAULT_N_WORLDS, help="shared worlds per pair")
    ap.add_argument("--reps", type=int, default=4, help="world blocks per pair on the probe jobs")
    ap.add_argument("--probe-jobs", type=int, default=0, help="first N seeds get --reps blocks")
    ap.add_argument("--close-gap", type=float, default=PR.DEFAULT_CLOSE_GAP)
    ap.add_argument("--mix", default=None, help='JSON, e.g. \'{"close_call":0.5,"greedy_vs_extract":0.4,"random":0.1}\'')
    ap.add_argument("--per-kind", default=None,
                    help='JSON per-kind snapshot caps forwarded to sample_states, e.g. '
                         '\'{"hand":5,"nemesis":4,"shop":2,"pack":2,"blind_select":1}\' '
                         '(pair_job default: even spread over pairable kinds)')
    ap.add_argument("--policy", default="auto", choices=["auto", "ev", "scripted"])
    ap.add_argument("--budget", default="fast")
    ap.add_argument("--shop-tier", default="rules", choices=["rules", "stats"])
    ap.add_argument("--epsilon-selfplay", type=float, default=0.1)
    ap.add_argument("--epsilon-rollout", type=float, default=0.02)
    ap.add_argument("--encoder", default="auto", choices=["auto", "v2", "dummy"])
    ap.add_argument("--max-ante", type=int, default=L.DEFAULT_MAX_ANTE)
    ap.add_argument("--deck", default="b_red")
    ap.add_argument("--stake", type=int, default=1)
    ap.add_argument("--lives", type=int, default=4)
    ap.add_argument("--policy-seed", type=int, default=0)
    ap.add_argument("--rollout-seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-jobs", type=int, default=None)
    ap.add_argument("--minutes", type=float, default=None)
    ap.add_argument("--flush-jobs", type=int, default=8)
    ap.add_argument("--shard-pairs", type=int, default=500, help="...or when the buffer holds N pairs")
    ap.add_argument("--coupling", default="step_then_determinize", choices=list(PR.COUPLINGS),
                    help="frozen order by default; determinize_then_step is the §4 diagnostic")
    ap.add_argument("--aux", action="store_true",
                    help="W-AUX: record auxiliary targets on BOTH branches from the rollouts "
                         "this job already runs (ev/AUX_NOTES.md)")
    ap.add_argument("--allow-clairvoyant", action="store_true", help="plumbing only (no W2)")
    ap.add_argument("--name", default=None)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = Path(args.run_dir)
    pair_dir = run_dir / "shards"
    abs_dir = run_dir / "abs_shards"
    pair_dir.mkdir(parents=True, exist_ok=True)
    abs_dir.mkdir(parents=True, exist_ok=True)
    done_path = run_dir / "done.ids"
    pause_path = run_dir / PAUSE_FILE
    log_path = run_dir / "gen.jsonl"
    name = args.name or run_dir.name
    seeds = parse_seeds(args.seeds, args.seed_rng)
    done = W.load_done_ids(done_path)
    if pause_path.exists():
        pause_path.unlink()

    mix = json.loads(args.mix) if args.mix else dict(PR.DEFAULT_MIX)
    base = {"n_states": args.n_states, "n_worlds": args.n_worlds, "close_gap": args.close_gap,
            "mix": mix, "policy": args.policy, "budget": args.budget, "shop_tier": args.shop_tier,
            "epsilon_selfplay": args.epsilon_selfplay, "epsilon_rollout": args.epsilon_rollout,
            "encoder": args.encoder, "max_ante": args.max_ante, "deck_key": args.deck,
            "stake": args.stake, "lives": args.lives, "policy_seed": args.policy_seed,
            "rollout_seed": args.rollout_seed, "allow_clairvoyant": args.allow_clairvoyant,
            "coupling": args.coupling, "aux": args.aux,
            "per_kind": json.loads(args.per_kind) if args.per_kind else None}

    def jobs():
        for i, s in enumerate(seeds):
            if s in done:
                continue
            payload = dict(base, seed=s)
            if i < args.probe_jobs:
                payload["reps"] = args.reps
            yield s, payload

    lf = open(log_path, "a", encoding="utf-8")

    def emit(rec: dict) -> None:
        lf.write(json.dumps(rec, default=str) + "\n")
        lf.flush()

    state = {"pairs": [], "rows": [], "ids": [],
             "n_pair_shards": len(sorted(pair_dir.glob("pair_*.npz"))),
             "n_abs_shards": len(sorted(abs_dir.glob("shard_*.npz"))),
             "n_pairs": 0, "n_rows": 0, "jobs": 0, "skipped": 0, "t0": time.perf_counter(),
             "timing": {"selfplay_s": 0.0, "pair_s": 0.0, "n_snapshots": 0, "n_pairs": 0,
                        "n_rollouts": 0, "rollout_decisions": 0, "rollout_policy_s": 0.0,
                        "total_s": 0.0}}

    def flush(why: str) -> None:
        if not state["ids"]:
            return
        pp = ap_ = None
        if state["pairs"]:
            pp = pair_dir / f"pair_{state['n_pair_shards']:04d}.npz"
            PR.save_pair_shard(pp, state["pairs"])
            state["n_pair_shards"] += 1
        if state["rows"]:
            ap_ = abs_dir / f"shard_{state['n_abs_shards']:04d}.npz"
            DS.save_shard(ap_, state["rows"])
            state["n_abs_shards"] += 1
        for jid in state["ids"]:
            W.mark_done(done_path, jid)
            done.add(jid)
        n_p, n_r = len(state["pairs"]), len(state["rows"])
        state["n_pairs"] += n_p
        state["n_rows"] += n_r
        el = time.perf_counter() - state["t0"]
        tm = state["timing"]
        rec = {"kind": "flush", "why": why, "pair_shard": str(pp), "abs_shard": str(ap_),
               "pairs": n_p, "rows": n_r, "jobs": len(state["ids"]),
               "pairs_total": state["n_pairs"], "jobs_total": state["jobs"], "elapsed_s": el,
               "pairs_per_min": 60 * state["n_pairs"] / el if el else 0.0,
               "s_per_pair": tm["pair_s"] / tm["n_pairs"] if tm["n_pairs"] else 0.0,
               "ms_per_rollout": 1000 * tm["pair_s"] / tm["n_rollouts"] if tm["n_rollouts"] else 0.0,
               "policy_frac": tm["rollout_policy_s"] / tm["pair_s"] if tm["pair_s"] else 0.0,
               "skipped": state["skipped"]}
        emit(rec)
        print(f"  [flush:{why}] +{n_p} pairs / {n_r} rows ({state['n_pairs']} pairs, "
              f"{state['jobs']} jobs, {rec['pairs_per_min']:.1f} pairs/min, "
              f"{rec['ms_per_rollout']:.0f} ms/rollout, policy {100 * rec['policy_frac']:.0f}%)",
              flush=True)
        state["pairs"], state["rows"], state["ids"] = [], [], []

    def on_result(job_id, result) -> None:
        state["pairs"].extend(PR.pairs_from_result(result))
        state["rows"].extend(PR.rows_from_result(result))
        state["ids"].append(job_id)
        state["jobs"] += 1
        state["skipped"] += int(result.get("skipped", 0))
        for k, v in result["timing"].items():
            if k in state["timing"]:
                state["timing"][k] += v
        if len(state["pairs"]) >= args.shard_pairs:
            flush("pairs")

    def on_checkpoint(summary) -> None:
        flush("jobs")

    emit({"kind": "config", "timestamp": datetime.now().isoformat(timespec="seconds"),
          "args": vars(args), "payload_base": base, "n_seeds": len(seeds), "n_done_before": len(done),
          "has_determinize": L.has_determinize(), "has_ev_player": L.has_ev_player(),
          "has_encoder_v2": L.has_encoder_v2(), "has_extraction": PR.has_extraction(),
          "extraction_entry_point": PR.extraction_entry_point(),
          "player_fingerprint": PR.player_fingerprint(
              policy=("ev" if args.policy == "auto" and L.has_ev_player() else args.policy),
              budget=args.budget, shop_tier=args.shop_tier, epsilon_rollout=args.epsilon_rollout)})
    print(f"=== pair campaign {name}: {len(seeds)} seeds ({len(done)} done), {args.workers} workers, "
          f"{args.n_states} states x {args.n_worlds} shared worlds, policy={args.policy} "
          f"(ev={L.has_ev_player()}), encoder={args.encoder}, determinize={L.has_determinize()}, "
          f"extraction={PR.has_extraction()} ===")
    print(f"  run dir: {run_dir}   pause: touch {pause_path}")
    summary = W.run_pool(PR.pair_job, jobs(), n_workers=args.workers, on_result=on_result,
                         pause_file=pause_path, checkpoint_every=args.flush_jobs,
                         on_checkpoint=on_checkpoint, state_path=None, max_jobs=args.max_jobs,
                         deadline_s=(args.minutes * 60 if args.minutes else None))
    flush("exit")

    pd = PR.PairDataset.load(pair_dir)
    ds = DS.LabelDataset.load(abs_dir)
    var = PR.variance_report(pd.records, n_worlds=args.n_worlds)
    mixr = PR.mix_report(pd.records)
    aux_cov = None
    if args.aux and len(pd):
        import aux_targets as AX  # noqa: WPS433
        aux_cov = {b: AX.coverage([(r.get("aux") or {}).get(b) for r in pd.records])
                   for b in ("a", "b")}
    el = time.perf_counter() - state["t0"]
    out = {"name": name, "run_dir": str(run_dir),
           "timestamp": datetime.now().isoformat(timespec="seconds"),
           "pool": summary.as_dict(), "n_pairs": len(pd), "n_abs_rows": len(ds),
           "n_seeds": len(pd.seeds()), "skipped_states": state["skipped"],
           "variance": var, "mix": mixr, "abs_dataset": ds.summary(), "aux_coverage": aux_cov,
           "throughput": {"pairs_per_min": 60 * len(pd) / el if el else 0.0,
                          "workers": args.workers,
                          "s_per_pair_in_worker": state["timing"]["pair_s"] / max(state["timing"]["n_pairs"], 1),
                          "ms_per_rollout": 1000 * state["timing"]["pair_s"] / max(state["timing"]["n_rollouts"], 1),
                          "elapsed_s": el},
           "timing_totals": state["timing"], "config": base, "seeds_spec": args.seeds,
           "has_extraction": PR.has_extraction()}
    emit({"kind": "summary", **out})
    lf.close()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    res_path = RESULTS_DIR / f"pairs_{name}.json"
    res_path.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    why = "paused" if summary.paused else ("interrupted" if summary.interrupted else
                                            ("exhausted" if summary.exhausted else "deadline/max_jobs"))
    print(f"=== stopped ({why}): {summary.done} jobs ({summary.failed} failed), {len(pd)} pairs / "
          f"{len(ds)} abs rows, {out['throughput']['pairs_per_min']:.1f} pairs/min; -> {res_path}")
    c = var["crn"]
    print(f"  VARIANCE REDUCTION (crn, n={var['n_pairs']}): {c['var_reduction_factor']:.2f}x "
          f"(mean rho {c['mean_rho']:.3f}); direct (n={var['direct']['n_pairs']}): "
          f"{var['direct']['var_reduction_factor']:.2f}x; resolved {100 * var['resolved_frac']:.1f}%")
    print(f"  realised mix: {mixr['pair_source']} / kinds {mixr['state_kind']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
