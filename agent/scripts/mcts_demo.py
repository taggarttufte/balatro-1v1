"""
mcts_demo.py — run MCTS on a real Balatro state with either UniformPolicy or a
freshly-init NNPolicy, and report sims/sec.

Usage:
    python agent/scripts/mcts_demo.py                       # uniform policy (default)
    python agent/scripts/mcts_demo.py --policy nn           # neural net (random init)
    python agent/scripts/mcts_demo.py --policy both --strategy both
    python agent/scripts/mcts_demo.py --policy nn --device cuda
    python agent/scripts/mcts_demo.py --ruleset mlb --nemesis  # MLB Nemesis state

Goal: confirm the search loop is correct end-to-end on the fork engine and measure the
per-sim cost of NN inference vs uniform. `--repeat N` averages N timed runs (the first
run pays import/CUDA warm-up, so `--repeat 3` is the honest number).
"""
from __future__ import annotations
import argparse
import statistics
import sys
import time

import _bootstrap  # noqa: F401  (sys.path + fork guard)

import numpy as np
import torch

from balatro_sim.game import BalatroGame, State
from balatro_sim.jokers.base import JokerInstance
from mcts import MCTS, UniformPolicy, NNPolicy, PolicyValueNet
from mcts.search import MCTSConfig


def make_demo_state(seed: int = 42, ruleset: str = "vanilla",
                    nemesis: bool = False) -> BalatroGame:
    """A non-trivial mid-blind state with jokers, similar to the benchmark fixture.

    With `nemesis=True` (MLB only) the game is advanced to the ante-2 Boss slot, which
    under MLB is the Nemesis (`bl_mp_nemesis`), with an externally supplied opponent
    score — the state W2/W4 will actually ask an agent to play.
    """
    g = BalatroGame(seed=seed, ruleset=ruleset)
    if nemesis:
        _advance_to_nemesis(g)
        g.set_pvp_info(40_000, 2)
        return g
    g.step({"type": "play_blind"})
    g.dollars = 30
    g.jokers = [
        JokerInstance("j_joker"),
        JokerInstance("j_green_joker"),
        JokerInstance("j_steel_joker"),
    ]
    g.jokers[1].state = {"mult": 8, "sell_value": 3}
    return g


def _advance_to_nemesis(g: BalatroGame, max_steps: int = 40) -> None:
    """Walk the free-win path to the first PvP blind (ante 2 boss slot)."""
    for _ in range(max_steps):
        if g.current_blind.is_pvp and g.state is State.SELECTING_HAND:
            return
        if g.state is State.BLIND_SELECT:
            g.step({"type": "play_blind"})
        elif g.state is State.SELECTING_HAND:
            g.debug_win_blind()
        elif g.state is State.ROUND_EVAL:
            g.step({"type": "advance"})
        elif g.state is State.SHOP:
            g.step({"type": "leave_shop"})
        elif g.state is State.BOOSTER_OPEN:
            g.step({"type": "skip_booster"})
        else:
            break
    raise RuntimeError(f"could not reach a Nemesis blind (state={g.state})")


def fmt_action(k):
    """Human-readable action label."""
    if k is None:
        return "<none>"
    if k[0] in ("play", "discard"):
        return f"{k[0]}({list(k[1])})"
    if k[0] == "use_consumable":
        return f"use_c[{k[1]}]->{list(k[2])}"
    return str(k)


def run_demo(num_sims: int, policy, label: str, game: BalatroGame,
             strategy: str = "puct", add_noise: bool = True, repeat: int = 1):
    """
    strategy: "puct"   -> classic AlphaZero PUCT, with optional Dirichlet noise
              "gumbel" -> Gumbel-Top-k + Sequential Halving (add_noise ignored)
    """
    tag = "gumbel" if strategy == "gumbel" else ("puct+noise" if add_noise else "puct")
    print(f"[{label} {tag}] State: {game.state.name}, hand={len(game.hand)}, "
          f"legal={len(game.legal_actions())}")

    times = []
    root = visits = chosen = None
    for r in range(repeat):
        mcts = MCTS(policy, MCTSConfig(num_simulations=num_sims),
                    rng=np.random.default_rng(0))
        t0 = time.perf_counter()
        if strategy == "gumbel":
            root, visits, chosen = mcts.run_gumbel(game)
        else:
            root, visits = mcts.run(game, add_noise=add_noise)
            chosen = MCTS.best_action(visits)
        times.append(time.perf_counter() - t0)

    elapsed = min(times)
    mean = statistics.mean(times)
    print(f"  Ran {num_sims} sims in {elapsed*1000:.0f} ms "
          f"({elapsed*1000/num_sims:.2f} ms/sim, {num_sims/elapsed:.0f} sims/sec)"
          + (f"   [best of {repeat}; mean {num_sims/mean:.0f} sims/sec]" if repeat > 1 else ""))
    print(f"  chosen: {fmt_action(chosen)}")
    print(f"  root visits: {root.visit_count}, unique edges: {len(visits)}")
    if not visits:
        print(f"  (no legal actions here — stop_reason={root.stop_reason!r}, "
              f"value={root.terminal_value:.3f})")
        return num_sims / elapsed

    top = sorted(visits.items(), key=lambda kv: -kv[1])[:5]
    print("  Top-5 visit counts:")
    for k, n in top:
        child = root.children[k]
        marker = " *" if k == chosen else ""
        print(f"    {fmt_action(k):28s}  N={n:4d}  Q={child.mean_value:+.3f}  "
              f"P={child.prior:.4f}{marker}")

    total = sum(visits.values())
    max_share = max(visits.values()) / max(total, 1)
    unique_visited = sum(1 for v in visits.values() if v > 0)
    priors = [c.prior for c in root.children.values()]
    if priors:
        print(f"  max visit share: {max_share*100:.1f}%, edges visited: "
              f"{unique_visited}/{len(visits)}, "
              f"prior min/max: {min(priors):.5f}/{max(priors):.5f}")
    return num_sims / elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["uniform", "nn", "both"], default="uniform")
    ap.add_argument("--device", default="cpu", help="cpu or cuda for NN policy")
    ap.add_argument("--sims", type=int, nargs="+", default=[100, 500, 2000])
    ap.add_argument("--strategy", choices=["puct", "gumbel", "both"], default="puct",
                    help="root action selector")
    ap.add_argument("--noise", choices=["off", "on", "both"], default="on",
                    help="(puct only) Dirichlet root noise off / on / compare both")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ruleset", choices=["vanilla", "mlb"], default="vanilla")
    ap.add_argument("--nemesis", action="store_true",
                    help="(mlb) run at the ante-2 Nemesis with an external target")
    ap.add_argument("--encoder", choices=["v7", "mlb"], default="v7")
    ap.add_argument("--repeat", type=int, default=1, help="timed repeats; report best")
    args = ap.parse_args()

    if args.nemesis and args.ruleset != "mlb":
        ap.error("--nemesis requires --ruleset mlb")

    from mcts import get_encoder
    encoder = get_encoder(args.encoder)

    policies = []
    if args.policy in ("uniform", "both"):
        policies.append(("uniform", UniformPolicy()))
    if args.policy in ("nn", "both"):
        torch.manual_seed(0)
        net = PolicyValueNet(obs_dim=encoder.dim)
        policies.append((f"nn-{args.device}", NNPolicy(net, device=args.device,
                                                       encoder=encoder)))

    runs: list[tuple[str, bool]] = []  # (strategy, add_noise)
    if args.strategy in ("puct", "both"):
        if args.noise in ("off", "both"):
            runs.append(("puct", False))
        if args.noise in ("on", "both"):
            runs.append(("puct", True))
    if args.strategy in ("gumbel", "both"):
        runs.append(("gumbel", False))  # add_noise ignored for gumbel

    game = make_demo_state(seed=args.seed, ruleset=args.ruleset, nemesis=args.nemesis)
    print(f"ruleset={args.ruleset} encoder={args.encoder} "
          f"blind={'PVP' if game.current_blind.is_pvp else game.current_blind.kind} "
          f"target={game.current_blind.chips_target}")

    for n in args.sims:
        print("=" * 60)
        print(f"  num_simulations = {n}")
        print("=" * 60)
        for label, pol in policies:
            for strategy, add_noise in runs:
                run_demo(n, pol, label, game, strategy=strategy,
                         add_noise=add_noise, repeat=args.repeat)
                print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
