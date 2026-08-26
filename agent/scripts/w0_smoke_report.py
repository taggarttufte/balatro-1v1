"""
w0_smoke_report.py — read the W0 smoke arms and print the comparison table.

    python agent/scripts/w0_smoke_report.py [run-dir] [prefix]

One row per `<run-dir>/<prefix>*/` that holds a `*.jsonl`:

    episodes | ep/min | ante-1 clear rate | mean ante | mean blinds | mean len
    | value loss | policy loss | lambda

`clear` is "reached ante 2", i.e. cleared all three ante-1 blinds — the same definition
`ColdTrainer` logs as `cleared` and anneals `--heuristic-prior-anneal clear:<r>` on.
`value loss` is the number that says whether there is a learnable target at all: Stage A
of the first real run sat at 0.0008 because every episode ended at ante 1 with the same z.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def read(path: Path) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("kind") == "episode":
                out.append(rec)
    return out


def summarise(name: str, eps: list[dict], wall: float) -> dict:
    n = len(eps)
    if n == 0:
        return {"arm": name, "episodes": 0}
    metrics = [e["metrics"] for e in eps if e.get("metrics")]
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")   # noqa: E731
    return {
        "arm": name,
        "episodes": n,
        "ep_min": n / wall * 60.0 if wall > 0 else float("nan"),
        "clear": mean([1.0 if e.get("cleared") else 0.0 for e in eps]),
        "ante": mean([e["ante"] for e in eps]),
        "blinds": mean([e.get("blinds", 0) for e in eps]),
        "len": mean([e["len"] for e in eps]),
        "vloss": mean([m["value_loss"] for m in metrics]),
        "ploss": mean([m["policy_loss"] for m in metrics]),
        "lam": mean([e.get("h_lambda") or 0.0 for e in eps]),
    }


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "agent/runs")
    prefix = sys.argv[2] if len(sys.argv) > 2 else "w0_"
    rows = []
    for d in sorted(root.glob(prefix + "*")):
        if not d.is_dir():
            continue
        logs = list(d.glob("*.jsonl"))
        if not logs:
            continue
        eps = read(logs[0])
        wall = sum(e.get("t_play", 0.0) + e.get("t_train", 0.0) for e in eps)
        rows.append(summarise(d.name, eps, wall))

    hdr = (f"{'arm':<16}{'eps':>6}{'ep/min':>9}{'clear%':>9}{'ante':>7}"
           f"{'blinds':>8}{'len':>7}{'v-loss':>9}{'p-loss':>8}{'lambda':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if not r["episodes"]:
            print(f"{r['arm']:<16}{0:>6}  (no episodes)")
            continue
        print(f"{r['arm']:<16}{r['episodes']:>6}{r['ep_min']:>9.1f}"
              f"{r['clear'] * 100:>9.1f}{r['ante']:>7.2f}{r['blinds']:>8.2f}"
              f"{r['len']:>7.1f}{r['vloss']:>9.4f}{r['ploss']:>8.3f}{r['lam']:>8.2f}")
    print()
    print("clear% = reached ante 2 (all three ante-1 blinds cleared).")
    print("ep/min is measured over summed in-episode wall clock, so it is comparable")
    print("between arms even when they ran in parallel.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
