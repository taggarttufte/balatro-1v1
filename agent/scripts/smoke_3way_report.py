"""
smoke_3way_report.py — read the JSONL of `smoke_3way.sh`'s runs and print the comparison
table TRAIN_NOTES.md §7.3 wants: skip rate, blind-clear rate, value-target sd, and the
current net's mean rank against the scripted anchors, for each lever.

    python agent/scripts/smoke_3way_report.py [--runs agent/runs] [--tag 3way]
    python agent/scripts/smoke_3way_report.py --names p4w2_gate2 p4w2_smoke_a

The last-generation row is what the levers are judged on, but every generation is printed:
five generations is a small sample and a trend that reverses matters more than any single
number.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LABELS = {
    "a": "(a) warm start",
    "b": "(b) skip cap",
    "c": "(c) warm + cap",
}


def load(run_dir: Path) -> list[dict]:
    """Generation rows from a run directory, in order."""
    log = run_dir / f"{run_dir.name}.jsonl"
    if not log.is_file():
        return []
    rows = []
    for line in log.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("kind") == "generation":
            rows.append(rec)
    return rows


def fmt(x, spec=".3f", nan="  -  "):
    try:
        if x is None or x != x:
            return nan
        return format(x, spec)
    except (TypeError, ValueError):
        return nan


def report(name: str, label: str, rows: list[dict]) -> None:
    if not rows:
        print(f"{label:<16} {name:<22} (no generations logged)")
        return
    print(f"\n{label}  -  {name}  ({len(rows)} generations)")
    print("  gen |  skip |  clear |  z sd | rank cur / anchor | jokers | ante | cap")
    for r in rows:
        cap = r.get("skip_cap")
        print(f"  {r['generation']:>3} | "
              f"{fmt(r.get('skip_rate'), '5.1%')} | "
              f"{fmt(r.get('blind_clear_rate'), '6.1%')} | "
              f"{fmt(r.get('value_target_sd'))} | "
              f"      {fmt(r.get('rank_current'), '.2f')} / {fmt(r.get('rank_anchor'), '.2f')}      | "
              f"{fmt(r.get('mean_jokers'), '6.2f')} | "
              f"{fmt(r.get('mean_ante_reached'), '4.2f')} | "
              f"{'-' if cap is None else cap}")
    last = rows[-1]
    print(f"  final: skip {fmt(last.get('skip_rate'), '.1%')}  "
          f"clear {fmt(last.get('blind_clear_rate'), '.1%')}  "
          f"z sd {fmt(last.get('value_target_sd'))}  "
          f"rank {fmt(last.get('rank_current'), '.2f')} vs anchors "
          f"{fmt(last.get('rank_anchor'), '.2f')}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="agent/runs")
    ap.add_argument("--tag", default="3way")
    ap.add_argument("--names", nargs="*", default=None,
                    help="explicit run directory names instead of <tag>_a/b/c")
    args = ap.parse_args()

    root = Path(args.runs)
    if args.names:
        targets = [(n, n) for n in args.names]
    else:
        targets = [(f"{args.tag}_{k}", LABELS[k]) for k in ("a", "b", "c")]

    for name, label in targets:
        report(name, label, load(root / name))

    print("\nWhat to look for: the lever that keeps SKIP RATE from climbing while BLIND-CLEAR")
    print("RATE rises. A lever that only lowers the skip rate without teaching the net to")
    print("clear blinds has made the policy worse, not better - it will simply lose more")
    print("lives. If two levers are indistinguishable at this sample size, prefer (c): the")
    print("skip cap anneals itself off, so it costs nothing once the warm-up has done its job.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
