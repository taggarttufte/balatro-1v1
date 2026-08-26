"""
bench_decide.py -- gate-3 timing benchmark: mean/p95 of ``decide.decision_table`` across
>=300 real SHOP / BOOSTER_OPEN states from actual scripted-player runs (not synthetic
states). Single-process, sequential -- this is a light read-only benchmark (state
collection + timing only), safe to run interactively.

Usage: python stats/bench_decide.py [--n-states 300] [--seeds N]
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_MP_ROOT = _HERE.parent
for _p in (str(_MP_ROOT), str(_HERE), str(_MP_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _bootstrap  # noqa: E402
from _bootstrap import BalatroGame, State  # noqa: E402
import decide  # noqa: E402
import mlb_match_demo as D  # noqa: E402

DEFAULT_SEEDS = sorted(p.stem for p in (_MP_ROOT / "oracle" / "ground_truth").glob("*.json"))


class _SoloShim:
    def __init__(self, g):
        self.games = [g]


def _step(game, policy):
    acts = game.legal_actions()
    if not acts:
        return {"type": "advance"}
    return policy(_SoloShim(game), 0, acts)


def collect_states(seed: str, max_states: int, max_steps: int = 40_000):
    spec = D.ScriptedPlayer(name="bench", hand="greedy", rerolls_per_visit=1,
                            buy_slot0=True, open_pack_slot=0, buy_voucher=True)
    policy = D.make_policy(spec)
    game = BalatroGame(seed=seed, deck_key="b_red", stake=1, ruleset="mlb")
    out = []
    last = None
    steps = 0
    while game.state != State.GAME_OVER and len(out) < max_states and steps < max_steps:
        if game.state != last and game.state in (State.SHOP, State.BOOSTER_OPEN):
            out.append(game.clone())
        last = game.state
        game.step(_step(game, policy))
        steps += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-states", type=int, default=300)
    ap.add_argument("--per-seed", type=int, default=8)
    args = ap.parse_args()

    states = []
    for seed in DEFAULT_SEEDS:
        if len(states) >= args.n_states:
            break
        states.extend(collect_states(seed, args.per_seed))
    states = states[: args.n_states]

    times_ms = []
    for g in states:
        t0 = time.perf_counter()
        decide.decision_table(g)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    times_ms.sort()
    mean_ms = statistics.fmean(times_ms)
    p50 = times_ms[len(times_ms) // 2]
    p95 = times_ms[int(0.95 * (len(times_ms) - 1))]
    p99 = times_ms[int(0.99 * (len(times_ms) - 1))]
    print(f"n_states={len(states)}  mean={mean_ms:.3f}ms  p50={p50:.3f}ms  "
         f"p95={p95:.3f}ms  p99={p99:.3f}ms  max={times_ms[-1]:.3f}ms")


if __name__ == "__main__":
    main()
