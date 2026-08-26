"""bench.py — cost probe for the POC's two jobs, and the reconstruction pin.

    python ev/active_poc/bench.py --seeds 2 --n-probe 2

Prints seconds per self-play, per rollout and per job, which is what sizes the pool and the
arms (the whole POC is rollout-bound).  Also asserts the reproducibility contract that the
design rests on: ``arm_job`` re-derives byte-identical snapshots to the ones ``pool_job``
scored (see ``jobs.snapshot_rng_seed``).
"""
from __future__ import annotations

import argparse
import random
import string
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EV_ROOT = HERE.parent
MP_ROOT = EV_ROOT.parent
for _p in (str(EV_ROOT), str(MP_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _bootstrap  # noqa: E402,F401
import numpy as np  # noqa: E402

from active_poc.jobs import arm_job, pool_job  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--n-probe", type=int, default=2)
    ap.add_argument("--arm-states", type=int, default=2, help="states to label at n_rollouts=8")
    args = ap.parse_args(argv)

    rng = random.Random(20260825)
    alphabet = string.ascii_uppercase + string.digits
    seeds = ["".join(rng.choice(alphabet) for _ in range(8)) for _ in range(args.seeds)]

    tot = {"pool_s": 0.0, "sp_s": 0.0, "probe_ro": 0, "probe_lab_s": 0.0,
           "arm_s": 0.0, "arm_ro": 0, "arm_lab_s": 0.0, "arm_sp_s": 0.0}
    for seed in seeds:
        t0 = time.perf_counter()
        res = pool_job({"seed": seed, "n_probe": args.n_probe})
        dt = time.perf_counter() - t0
        tm = res["timing"]
        tot["pool_s"] += dt
        tot["sp_s"] += tm["selfplay_s"]
        tot["probe_ro"] += tm["n_rollouts"]
        tot["probe_lab_s"] += tm["label_s"]
        obs = res["rows"][0]["obs"]
        nbytes = sum(np.asarray(v).nbytes for v in obs.values())
        print(f"[{seed}] pool {dt:6.1f}s  selfplay {tm['selfplay_s']:5.1f}s  "
              f"probe-label {tm['label_s']:6.1f}s  snaps {tm['n_snapshots']:2d}  "
              f"rollouts {tm['n_rollouts']:3d}  {tm['label_s'] / max(tm['n_rollouts'], 1):.2f} s/rollout  "
              f"{tm['rollout_decisions'] / max(tm['n_rollouts'], 1):.0f} dec/rollout  "
              f"obs {nbytes / 1024:.1f} KiB/row")

        steps = sorted({r["meta"]["step"] for r in res["rows"]})[: args.arm_states]
        fps = {str(r["meta"]["step"]): r["meta"]["obs_fp"]
               for r in res["rows"] if r["meta"]["player"] == 0 and r["meta"]["step"] in steps}
        t0 = time.perf_counter()
        ares = arm_job({"seed": seed, "steps": steps, "fps": fps})
        dt = time.perf_counter() - t0
        atm = ares["timing"]
        tot["arm_s"] += dt
        tot["arm_ro"] += atm["n_rollouts"]
        tot["arm_lab_s"] += atm["label_s"]
        tot["arm_sp_s"] += atm["selfplay_s"]
        assert not ares["mismatched_steps"], f"RECONSTRUCTION DRIFT on {seed}: {ares['mismatched_steps']}"
        assert not ares["missing_steps"], f"MISSING STEPS on {seed}: {ares['missing_steps']}"
        print(f"        arm  {dt:6.1f}s  selfplay {atm['selfplay_s']:5.1f}s  "
              f"label {atm['label_s']:6.1f}s  states {atm['n_states']}  rollouts {atm['n_rollouts']:3d}  "
              f"{atm['label_s'] / max(atm['n_rollouts'], 1):.2f} s/rollout  "
              f"y={[round(r['y'], 3) for r in ares['rows']]}  reconstruction OK")

    n = len(seeds)
    s_probe = tot["probe_lab_s"] / max(tot["probe_ro"], 1)
    s_arm = tot["arm_lab_s"] / max(tot["arm_ro"], 1)
    print(f"\n== per-seed means over {n} seeds ==")
    print(f"  self-play           {tot['sp_s'] / n:6.1f} s   (paid once per seed, in BOTH jobs)")
    print(f"  probe rollout       {s_probe:6.2f} s")
    print(f"  arm rollout         {s_arm:6.2f} s")
    print(f"  pool_job (12 states x {args.n_probe} rollouts) {tot['pool_s'] / n:6.1f} s")
    print(f"  arm_job  ({args.arm_states} states x 8 rollouts)  {tot['arm_s'] / n:6.1f} s")
    r = (s_probe + s_arm) / 2
    print(f"\n== sizing at 8 workers (s/rollout ~ {r:.2f}, self-play ~ {tot['sp_s'] / n:.1f} s) ==")
    for n_pool in (400, 800, 1200):
        mins = n_pool * (tot["sp_s"] / n + 12 * args.n_probe * s_probe) / 8 / 60
        print(f"  pool {n_pool:4d} seeds = {n_pool * 24:5d} rows: {mins:5.1f} min")
    for n_states in (1500, 3000, 4500):
        mins = n_states * 8 * s_arm / 8 / 60
        print(f"  arms {n_states:4d} states = {n_states * 2:5d} rows: {mins:5.1f} min (+ self-play per seed touched)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
