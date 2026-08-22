"""
smoke_selfplay.py — end-to-end smoke test for the self-play training loop.

Runs:
  1. One self-play episode (Gumbel MCTS, NN policy at random init)
  2. Push trajectory into a replay buffer
  3. One training step on a mini-batch
  4. Save a checkpoint, reload it, and confirm the weights round-trip exactly

Goal: confirm all the pipes connect. No claim about quality of play or convergence —
just that nothing crashes, losses are finite, the network's weights move, and a
checkpoint can be written and read back.

    python mp/agent/scripts/smoke_selfplay.py
    python mp/agent/scripts/smoke_selfplay.py --device cuda
    python mp/agent/scripts/smoke_selfplay.py --ruleset mlb --max-antes 3
"""
from __future__ import annotations
import argparse
import sys
import tempfile
import time
from pathlib import Path

import _bootstrap  # noqa: F401  (sys.path + fork guard)

import numpy as np
import torch

from train import ColdTrainer, TrainConfig, load_checkpoint, save_checkpoint


def _weight_signature(net: torch.nn.Module) -> torch.Tensor:
    """Concatenate a flat tensor of all params for a fast before/after diff."""
    return torch.cat([p.detach().flatten().clone() for p in net.parameters()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sims", type=int, default=50,
                    help="MCTS simulations per decision (kept small for smoke)")
    ap.add_argument("--max-considered", type=int, default=8, help="Gumbel m_init")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--encoder", choices=["v7", "mlb"], default="v7")
    ap.add_argument("--ruleset", choices=["vanilla", "mlb"], default="vanilla")
    ap.add_argument("--max-antes", type=int, default=None)
    ap.add_argument("--max-decisions", type=int, default=2000)
    args = ap.parse_args()

    print(f"=== Smoke loop (seed={args.seed}, device={args.device}, "
          f"ruleset={args.ruleset}, encoder={args.encoder}) ===")

    cfg = TrainConfig(
        seed=args.seed, sims=args.sims, max_considered=args.max_considered,
        batch_size=args.batch_size, lr=args.lr, device=args.device,
        encoder=args.encoder, ruleset=args.ruleset, max_antes=args.max_antes,
        max_decisions=args.max_decisions,
        min_buffer=1,                 # train on the first episode
        buffer_capacity=10_000,
    )
    trainer = ColdTrainer(cfg)
    print(f"  net params: {sum(p.numel() for p in trainer.net.parameters()):,} "
          f"(obs_dim={trainer.encoder.dim}, encoder={trainer.encoder.name})")

    # ── Steps 1-3: one episode + buffer + a training step ───────────────────
    print("\n[1/4] Running one self-play episode + one training step...")
    sig_before = _weight_signature(trainer.net)
    t0 = time.perf_counter()
    rec = trainer.run_episode()
    ep_time = time.perf_counter() - t0
    sig_after = _weight_signature(trainer.net)

    if rec["kind"] == "error":
        print("ERROR: episode crashed")
        print(rec["trace"])
        return 1

    print(f"  episode finished in {ep_time:.1f}s ({rec['stop']})")
    print(f"  trajectory length: {rec['len']} decisions ({rec['searches']} searches)")
    print(f"  final ante: {rec['ante']}   shaped z: {rec['shaped_z']:.4f}   "
          f"won: {rec['won']}")
    print(f"  buffer size: {rec['buf']}")

    metrics = rec["metrics"]
    if metrics is None:
        print("ERROR: no training step ran (empty episode?)")
        return 1
    weight_delta = (sig_after - sig_before).abs().mean().item()
    print(f"  trained on batch of {metrics['n']} samples in {rec['t_train']*1000:.0f} ms")
    print(f"  policy_loss: {metrics['policy_loss']:.4f}")
    print(f"  value_loss:  {metrics['value_loss']:.4f}")
    print(f"  total_loss:  {metrics['total_loss']:.4f}")
    print(f"  mean abs(weight delta): {weight_delta:.2e}")

    # ── Step 4: checkpoint round-trip ───────────────────────────────────────
    print("\n[2/4] Checkpoint save + reload...")
    tmp = Path(tempfile.mkdtemp()) / "smoke.pt"
    save_checkpoint(tmp, trainer.state_dict())
    size_mb = tmp.stat().st_size / 1e6
    reloaded = ColdTrainer.from_checkpoint(load_checkpoint(tmp))
    sig_reloaded = _weight_signature(reloaded.net).to(sig_after.device)
    same = torch.equal(sig_after.cpu(), sig_reloaded.cpu())
    print(f"  wrote {tmp} ({size_mb:.1f} MB); weights identical after reload: {same}")
    print(f"  counters: {reloaded.counters}")

    # ── Sanity checks ───────────────────────────────────────────────────────
    failures = []
    if not np.isfinite(metrics["policy_loss"]):
        failures.append("policy_loss is not finite")
    if not np.isfinite(metrics["value_loss"]):
        failures.append("value_loss is not finite")
    if weight_delta == 0:
        failures.append("weights did not change after training step")
    if not same:
        failures.append("checkpoint reload changed the weights")
    if reloaded.counters.episodes != trainer.counters.episodes:
        failures.append("checkpoint did not restore the counters")

    print()
    if failures:
        print("FAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("OK: all pipes connected, losses finite, weights moved, checkpoint round-trips.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
