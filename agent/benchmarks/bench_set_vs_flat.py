"""
bench_set_vs_flat.py — search throughput, set encoder vs flat, CPU and CUDA.

This is the measurement that should decide `--device` for the first long training run.
Phase 4 W1 measured it BEFORE the embedding merge (SETENC_NOTES §6.2):

    flat mlb cpu   449 sims/s      set cpu    259 sims/s
    flat mlb cuda  327 sims/s      set cuda   118 sims/s   (69 before the transfer packing)

The merge (SETENC_NOTES §8.1) took the set net from ~25 embedding kernels per forward to 7,
which is exactly what that CUDA gap called for, but the re-run was deferred (machine in
use). Run this and paste the table into SETENC_NOTES §6.2.

    python agent/benchmarks/bench_set_vs_flat.py
    python agent/benchmarks/bench_set_vs_flat.py --sims 500 --repeats 3

Same state as every other Phase 3/4 throughput number: the ante-1 `SELECTING_HAND` demo
state with 436 legal actions, Gumbel selection, `leaf_batch=16`, cold-init nets, best of
`--repeats` after one warm-up.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))
import _bootstrap  # noqa: E402,F401  (sys.path + fork guard)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from mcts import MCTS, MCTSConfig, PolicyValueNet, get_encoder, make_policy  # noqa: E402
from mcts.model_set import SetPolicyValueNet  # noqa: E402
from mcts_demo import make_demo_state  # noqa: E402


def build(encoder_name: str, device: str, seed: int = 0):
    torch.manual_seed(seed)
    enc = get_encoder(encoder_name)
    net = (SetPolicyValueNet(caps=enc.caps) if enc.is_set
           else PolicyValueNet(obs_dim=enc.dim))
    return enc, net, make_policy(net, device=device, encoder=enc, batched=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--leaf-batch", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--encoders", default="mlb,set")
    ap.add_argument("--devices", default="cpu,cuda")
    ap.add_argument("--ruleset", default="mlb")
    ap.add_argument("--nemesis", action="store_true")
    args = ap.parse_args(argv)

    game = make_demo_state(seed=42, ruleset=args.ruleset, nemesis=args.nemesis)
    n_legal = len(game.legal_actions())
    devices = [d for d in args.devices.split(",")
               if d != "cuda" or torch.cuda.is_available()]
    print(f"{game.state.name}, {n_legal} legal actions, {args.sims} sims, "
          f"leaf_batch={args.leaf_batch}, best of {args.repeats}\n")
    print(f"{'encoder':<10}{'device':<8}{'params':>12}{'sims/s':>10}{'ms/sim':>10}")

    results = {}
    for encoder_name in args.encoders.split(","):
        for device in devices:
            enc, net, policy = build(encoder_name, device)
            cfg = MCTSConfig(num_simulations=args.sims, gumbel_max_considered=8,
                             leaf_batch=args.leaf_batch)
            mcts = MCTS(policy, cfg, rng=np.random.default_rng(0))
            mcts.run_gumbel(game)                      # warm up kernels / allocator
            best = 0.0
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                mcts.run_gumbel(game)
                best = max(best, args.sims / (time.perf_counter() - t0))
            n_params = sum(p.numel() for p in net.parameters())
            results[(encoder_name, device)] = best
            print(f"{encoder_name:<10}{device:<8}{n_params:>12,}{best:>10.0f}"
                  f"{1000 / best:>10.2f}")

    if ("set", "cpu") in results and ("set", "cuda") in results:
        cpu, cuda = results[("set", "cpu")], results[("set", "cuda")]
        faster = "CPU" if cpu > cuda else "CUDA"
        print(f"\nset encoder: {faster} is faster ({max(cpu, cuda) / min(cpu, cuda):.2f}x)"
              f" -> use --device {faster.lower()} for a self-play-bound run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
