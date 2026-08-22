"""
bench_search.py — single-tree search throughput on the fork engine.

This is the BASELINE W3 measures against. It reports, per (policy, strategy):

    sims/sec         end-to-end search throughput
    ms/sim           per-simulation cost
    NN / sim / other where the time actually goes, by instrumenting the two hot calls
                     (`MCTS._evaluate_leaf` and `BalatroGame.clone`/`step`)

    python mp/agent/benchmarks/bench_search.py
    python mp/agent/benchmarks/bench_search.py --device cuda --sims 500
    python mp/agent/benchmarks/bench_search.py --ruleset mlb --nemesis

The split is measured by monkey-patching wrappers around the two seams for the duration
of one timed run, so the "other" bucket is genuinely Python overhead (PUCT selection,
dict churn, action-key construction), not an estimate.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import _bootstrap  # noqa: E402,F401  (sys.path + fork guard)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from balatro_sim.game import BalatroGame  # noqa: E402
from mcts import MCTS, MCTSConfig, NNPolicy, PolicyValueNet, UniformPolicy  # noqa: E402


def _demo_state(seed: int, ruleset: str, nemesis: bool) -> BalatroGame:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from mcts_demo import make_demo_state
    return make_demo_state(seed=seed, ruleset=ruleset, nemesis=nemesis)


def timed_run(mcts: MCTS, game: BalatroGame, strategy: str, instrument: bool):
    """One timed search. With `instrument`, also returns the NN / sim time split."""
    nn_time = [0.0]
    sim_time = [0.0]

    if instrument:
        real_eval = mcts._evaluate_leaf
        real_clone = BalatroGame.clone
        real_step = BalatroGame.step

        def eval_leaf(g):
            t = time.perf_counter()
            try:
                return real_eval(g)
            finally:
                nn_time[0] += time.perf_counter() - t

        def clone(self):
            t = time.perf_counter()
            try:
                return real_clone(self)
            finally:
                sim_time[0] += time.perf_counter() - t

        def step(self, action):
            t = time.perf_counter()
            try:
                return real_step(self, action)
            finally:
                sim_time[0] += time.perf_counter() - t

        mcts._evaluate_leaf = eval_leaf
        BalatroGame.clone = clone
        BalatroGame.step = step

    try:
        t0 = time.perf_counter()
        if strategy == "gumbel":
            mcts.run_gumbel(game)
        else:
            mcts.run(game, add_noise=True)
        elapsed = time.perf_counter() - t0
    finally:
        if instrument:
            BalatroGame.clone = real_clone
            BalatroGame.step = real_step
            mcts._evaluate_leaf = real_eval

    return elapsed, nn_time[0], sim_time[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ruleset", choices=["vanilla", "mlb"], default="vanilla")
    ap.add_argument("--nemesis", action="store_true")
    ap.add_argument("--encoder", choices=["v7", "mlb"], default="v7")
    args = ap.parse_args()

    from mcts import get_encoder
    encoder = get_encoder(args.encoder)
    game = _demo_state(args.seed, args.ruleset, args.nemesis)
    n_legal = len(game.legal_actions())

    torch.manual_seed(0)
    net = PolicyValueNet(obs_dim=encoder.dim)
    policies = [
        ("uniform", UniformPolicy()),
        (f"nn-{args.device}", NNPolicy(net, device=args.device, encoder=encoder)),
    ]

    print(f"bench_search: sims={args.sims} repeat={args.repeat} device={args.device} "
          f"ruleset={args.ruleset} legal_actions={n_legal}")
    print(f"{'policy':<12} {'strategy':<8} {'sims/sec':>10} {'ms/sim':>8} "
          f"{'nn%':>6} {'sim%':>6} {'other%':>7}")
    for label, pol in policies:
        for strategy in ("puct", "gumbel"):
            # Throughput: best of `repeat` UNinstrumented runs (instrumentation costs
            # two perf_counter calls per clone/step and would bias the number).
            elapsed = min(
                timed_run(MCTS(pol, MCTSConfig(num_simulations=args.sims),
                               rng=np.random.default_rng(0)),
                          game, strategy, instrument=False)[0]
                for _ in range(args.repeat)
            )
            # Split: one separate instrumented run, normalised to ITS own total.
            total_i, nn_t, sim_t = timed_run(
                MCTS(pol, MCTSConfig(num_simulations=args.sims),
                     rng=np.random.default_rng(0)),
                game, strategy, instrument=True)
            other = max(0.0, total_i - nn_t - sim_t)
            denom = max(total_i, 1e-9)
            print(f"{label:<12} {strategy:<8} {args.sims/elapsed:>10.0f} "
                  f"{elapsed*1000/args.sims:>8.3f} "
                  f"{100*nn_t/denom:>5.1f}% {100*sim_t/denom:>5.1f}% "
                  f"{100*other/denom:>6.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
