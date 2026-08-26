"""
cli.py — run an N-agent same-seed MLB tournament from the command line.

    python -m tournament.cli --seed 7I4M53DL --n 100 --life-rule none --max-ante 8
    python -m tournament.cli --seed 7I4M53DL --n 100 --life-rule paired --max-ante 40
    python -m tournament.cli --seed 7I4M53DL --n 32 --life-rule median --max-ante 8 \
        --out tournament/runs/demo

Prints a per-ante summary (n present, score mean/std/median, tie fraction — the degeneracy
metric, TOURNAMENT_NOTES.md §"heterogeneity"), how many died into that ante's Nemesis, and
the wall clock at the end.  ``--out`` (optional) also serializes the N x N matrices per ante
(``.npz``) + a JSONL summary (``matrix.py`` / TOURNAMENT_NOTES.md "file formats").
"""
from __future__ import annotations

import argparse
import sys
import time

from .players import default_population
from .runner import Tournament


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", default="7I4M53DL")
    ap.add_argument("--n", type=int, default=100, dest="n_agents")
    ap.add_argument("--deck", default="b_red", dest="deck_key")
    ap.add_argument("--stake", default="1")
    ap.add_argument("--life-rule", default="paired", choices=("paired", "median", "none"))
    ap.add_argument("--max-ante", type=int, default=8)
    ap.add_argument("--lives", type=int, default=None,
                     help="override starting lives (default: engine MLB default under "
                          "paired/median; a large sentinel under --life-rule none)")
    ap.add_argument("--fanout", default="auto", choices=("auto", "construct", "clone"))
    ap.add_argument("--base-seed", type=int, default=0,
                     help="seeds the default heterogeneous population (per-agent specs / RNG)")
    ap.add_argument("--out", default=None, help="run directory for .npz + JSONL artifacts")
    args = ap.parse_args(argv)

    stake = int(args.stake) if args.stake.isdigit() else args.stake
    players = default_population(args.n_agents, base_seed=args.base_seed)
    kwargs = dict(seed=args.seed, n_agents=args.n_agents, players=players, deck_key=args.deck_key,
                  stake=stake, life_rule=args.life_rule, max_ante=args.max_ante,
                  fanout=args.fanout, out_dir=args.out)
    if args.lives is not None:
        kwargs["lives"] = args.lives

    t = Tournament(**kwargs)
    t0 = time.perf_counter()
    res = t.run()
    wall = time.perf_counter() - t0

    print(f"Tournament seed={res.seed} n_agents={res.n_agents} life_rule={res.life_rule} "
          f"max_ante={res.max_ante} deck={res.deck_key} stake={res.stake} "
          f"fanout={res.fanout_method}")
    for row in res.summary_rows():
        q = row.get("quantiles") or {}
        mean = row.get("mean")
        if mean is None:
            print(f"  ante {row['ante']:>2d}  n_present=0 (nobody reached this Nemesis)")
            continue
        std = row.get("std")
        median = q.get("0.5", float("nan"))
        print(f"  ante {row['ante']:>2d}  n_present={row['n_present']:>4d}  "
              f"mean={mean:>10.1f}  std={std:>10.1f}  median={median:>10.1f}  "
              f"tie_frac={row['tie_fraction']:.4f}  deaths_this_ante={len(row['losers']):>3d}")
    print(f"alive at end: {len(res.alive_at_end)} / {res.n_agents}")
    if wall > 0:
        print(f"wall clock: {wall:.3f}s  ({res.steps_total} engine steps total, "
              f"{res.steps_total / wall:.0f} steps/s)")
    else:
        print("wall clock: 0s")
    if args.out:
        print(f"artifacts written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
