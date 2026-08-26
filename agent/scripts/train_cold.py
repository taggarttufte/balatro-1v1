"""
train_cold.py — cold-start AlphaZero-style training loop, with checkpointing.

For each iteration:
  1. Self-play one episode using current network + Gumbel MCTS.
  2. Push the trajectory into a replay buffer.
  3. Once the buffer has >= --min-buffer samples, run --steps-per-episode training
     steps per episode on random mini-batches.

Logging:
  - JSONL line per episode in <run-dir>/<run-name>.jsonl (full data for plotting)
  - Stdout summary every --log-every episodes (rolling stats over the recent window)

Checkpointing (new in the agent fork):
  - Every --checkpoint-every episodes, and unconditionally at exit (deadline reached,
    Ctrl+C, or a fatal error), the run writes <run-dir>/<run-name>/ckpt_<ep>.pt plus
    <run-dir>/<run-name>/latest.pt. A checkpoint carries model + optimizer + counters +
    RNG states + config + replay buffer, so
        python agent/scripts/train_cold.py --resume <path> --minutes 30
    continues the run bit-exactly on CPU (see AGENT_NOTES.md "Checkpointing").
  - --resume latest (or a directory) picks the newest checkpoint under it.
  - --no-checkpoint-buffer drops the replay buffer from the checkpoint: much smaller
    files, but a resume then trains on a different mini-batch stream.

Runtime is bounded by --minutes (or --episodes). Ctrl+C exits cleanly with a checkpoint.
Each episode runs inside a try/except so a single crash logs an error and the loop
continues (sims still occasionally hit untested engine paths).

This is the "cold-start" variant, and it is the ONLY variant: the V7 checkpoint that the
balatro-mcts README planned to warm-start the value head from was lost (confirmed
2026-06-10), and the observation is 447 dims now, not the 434 V7 trained on. Random init
means the value head is meaningless for many episodes; we expect mostly losses until it
picks up gradations from the shaped z labels.

    python agent/scripts/train_cold.py --minutes 2 --device cuda --checkpoint-every 5
    python agent/scripts/train_cold.py --resume agent/runs/<name>/latest.pt --minutes 2
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import _bootstrap  # noqa: F401  (sys.path + fork guard)

from train import (
    ColdTrainer, TrainConfig, latest_checkpoint, load_checkpoint, save_checkpoint,
    rolling_summary,
)

DEFAULT_RUN_DIR = Path(_bootstrap.AGENT_ROOT) / "runs"


def fmt_secs(s: float) -> str:
    s = int(max(0, s))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def print_status(ep: int, window: list[dict], elapsed: float, remaining: float,
                 buf_size: int, total_wins: int, ep_offset: int = 0):
    stats = rolling_summary(window)
    ep_per_min = ((ep - ep_offset) / max(elapsed, 1e-9)) * 60.0
    if not stats:
        print(f"[ep {ep:5d}] elapsed {fmt_secs(elapsed)} — no completed episodes yet",
              flush=True)
        return
    print(
        f"[ep {ep:5d}] "
        f"elapsed {fmt_secs(elapsed)} / remain {fmt_secs(remaining)} "
        f"({ep_per_min:.1f} ep/min) "
        f"buf {buf_size:6d} "
        f"recent: ante {stats['ante']:.2f} blinds {stats['blinds']:.2f} "
        f"clear% {stats['clear_pct']:.1f} len {stats['len']:.1f} z={stats['z']:.3f} "
        f"lam {stats['h_lambda']:.2f} "
        f"win% {stats['win_pct']:.2f} (total {total_wins}) "
        f"loss p={stats['policy_loss']:.3f} v={stats['value_loss']:.4f}",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--episodes", type=int, default=None,
                    help="stop after this many episodes (in addition to --minutes)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sims", type=int, default=30, help="MCTS sims per decision")
    ap.add_argument("--max-considered", type=int, default=8, help="Gumbel m_init")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--buffer-capacity", type=int, default=20_000,
                    help="replay capacity; a Sample is ~97 KB, so 20k is ~1.9 GB of RAM")
    ap.add_argument("--min-buffer", type=int, default=128,
                    help="don't start training until buffer has this many samples")
    ap.add_argument("--steps-per-episode", type=int, default=1)
    ap.add_argument("--log-every", type=int, default=10,
                    help="stdout summary every N episodes")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--encoder", choices=["v7", "mlb", "set"], default="v7",
                    help="v7 (447 flat) | mlb (453 flat) | set (Phase 4 set encoder)")
    ap.add_argument("--k-unvisited", type=int, default=8,
                    help="Sample v2: random zero-visit actions kept per sample "
                         "(every VISITED action is always kept)")
    ap.add_argument("--no-subsample", action="store_true",
                    help="keep every legal action in each Sample (the Phase 3 shape; "
                         "~20x bigger samples)")
    ap.add_argument("--set-res-blocks", type=int, default=2,
                    help="trunk depth of SetPolicyValueNet (--encoder set)")
    ap.add_argument("--value-activation", choices=["sigmoid", "clamp", "linear"],
                    default="sigmoid",
                    help="bound the value head to the OutcomeFn's [0, 1] range")
    ap.add_argument("--ruleset", choices=["vanilla", "mlb"], default="vanilla")
    ap.add_argument("--deck", default="b_red")
    ap.add_argument("--stake", type=int, default=1)
    ap.add_argument("--max-antes", type=int, default=None,
                    help="cap self-play episodes (MLB is endless)")
    ap.add_argument("--max-decisions", type=int, default=2000)
    ap.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR),
                    help="root directory for logs + checkpoints")
    ap.add_argument("--run-name", default=None,
                    help="subdirectory / log name (default: train_cold_<timestamp>)")
    # Checkpointing
    ap.add_argument("--checkpoint-every", type=int, default=25,
                    help="write a checkpoint every N episodes (0 = only at exit)")
    ap.add_argument("--keep-checkpoints", type=int, default=3,
                    help="how many numbered checkpoints to keep (latest.pt is always kept)")
    ap.add_argument("--no-checkpoint-buffer", action="store_true",
                    help="do not store the replay buffer in checkpoints (smaller files, "
                         "resume is no longer bit-exact)")
    ap.add_argument("--buffer-checkpoint-cap", type=int, default=5_000,
                    help="store at most this many (most recent) samples per checkpoint")
    # ── W0: the heuristic hand prior + candidate mask (mcts/heuristic.py) ──
    ap.add_argument("--heuristic-prior", type=float, default=0.0,
                    help="lambda in prior = (1-lambda)*net + lambda*heuristic over the "
                         "play/discard actions of SELECTING_HAND. 0 = off (the pre-W0 "
                         "search); 0.8 is the recommended vanilla warm-up")
    ap.add_argument("--heuristic-prior-floor", type=float, default=0.1,
                    help="lambda never anneals below this")
    ap.add_argument("--heuristic-prior-anneal", default="",
                    help="'' = constant | '<N>' or 'ep:<N>' = linear to the floor over N "
                         "episodes | 'clear:<r>' = decay as the rolling blind-clear rate "
                         "approaches r (same criterion as --skip-cap-anneal-clear-rate)")
    ap.add_argument("--heuristic-tau", type=float, default=0.5,
                    help="softmax temperature on log1p(score): 1.0 = prior ~ score, "
                         "0.5 = prior ~ score^2, ->0 = argmax")
    ap.add_argument("--heuristic-exact-top", type=int, default=8,
                    help="play subsets per leaf refined with the exact side-effect-free "
                         "score_hand dry run (0 = the cheap tier only). Skipped "
                         "automatically when the board is plain, where the two agree")
    ap.add_argument("--heuristic-discard-bias", type=float, default=1.0,
                    help="multiplier on every discard potential (>1 discards more)")
    ap.add_argument("--max-hand-candidates", type=int, default=32,
                    help="expand only the top-K play + top-K discard subsets per leaf "
                         "(0 = every legal action, the pre-W0 tree). Search-side only: "
                         "pruned actions stay legal in the engine")
    ap.add_argument("--resume", default=None,
                    help="path to a checkpoint, a run directory, or 'latest'")
    return ap


def config_from_args(args) -> TrainConfig:
    return TrainConfig(
        seed=args.seed, sims=args.sims, max_considered=args.max_considered,
        batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
        buffer_capacity=args.buffer_capacity, min_buffer=args.min_buffer,
        steps_per_episode=args.steps_per_episode, device=args.device,
        encoder=args.encoder, ruleset=args.ruleset, deck_key=args.deck,
        stake=args.stake, max_antes=args.max_antes, max_decisions=args.max_decisions,
        checkpoint_buffer=not args.no_checkpoint_buffer,
        buffer_checkpoint_cap=args.buffer_checkpoint_cap,
        subsample=not args.no_subsample, k_unvisited=args.k_unvisited,
        set_res_blocks=args.set_res_blocks, value_activation=args.value_activation,
        heuristic_prior=args.heuristic_prior,
        heuristic_prior_floor=args.heuristic_prior_floor,
        heuristic_prior_anneal=args.heuristic_prior_anneal,
        heuristic_tau=args.heuristic_tau,
        heuristic_exact_top=args.heuristic_exact_top,
        heuristic_discard_bias=args.heuristic_discard_bias,
        max_hand_candidates=args.max_hand_candidates,
    )


def resolve_resume(spec: str, run_dir: Path) -> Path:
    p = Path(spec)
    if spec == "latest":
        found = latest_checkpoint(run_dir, "**/latest.pt") or latest_checkpoint(run_dir, "**/*.pt")
        if found is None:
            raise SystemExit(f"--resume latest: no checkpoint under {run_dir}")
        return found
    if p.is_dir():
        found = latest_checkpoint(p, "latest.pt") or latest_checkpoint(p, "*.pt")
        if found is None:
            raise SystemExit(f"--resume {p}: no checkpoint in that directory")
        return found
    if not p.is_file():
        raise SystemExit(f"--resume {p}: no such file")
    return p


def prune_checkpoints(ckpt_dir: Path, keep: int) -> None:
    numbered = sorted(ckpt_dir.glob("ckpt_*.pt"), key=lambda q: q.stat().st_mtime)
    for old in numbered[:-keep] if keep > 0 else []:
        try:
            old.unlink()
        except OSError:
            pass


MLB_REFUSAL = (
    "train_cold.py --ruleset mlb is the degenerate free-Nemesis objective: with "
    "pvp_solo=True the engine resolves the Nemesis at hand exhaustion at no cost, so the "
    "agent learns to skip every blind and coast (CAMPAIGN_LOG 2026-08-22 07:35 -- 2,072 "
    "episodes, value-target sd collapsed to 0.07). Use: python agent/scripts/train_mlb.py "
    "--objective external   (or --objective tournament), which makes the Nemesis cost a "
    "life. train_cold.py is for --ruleset vanilla."
)


def main():
    args = build_parser().parse_args()
    run_root = Path(args.run_dir)

    # The overnight shakedown proved this objective is degenerate; refuse rather than let
    # someone spend a day of GPU on it. Nothing else changes: --ruleset vanilla is
    # untouched, and `ColdTrainer(ruleset="mlb")` stays available to W2's MLBTrainer and to
    # the tests, which drive it through a non-degenerate outcome.
    if getattr(args, "ruleset", "vanilla") == "mlb":
        raise SystemExit(MLB_REFUSAL)

    # ── Setup / resume ──────────────────────────────────────────────────────
    if args.resume:
        ckpt_path = resolve_resume(args.resume, run_root)
        ckpt = load_checkpoint(ckpt_path, map_location=args.device)
        overrides = {
            "device": args.device,
            "checkpoint_buffer": not args.no_checkpoint_buffer,
            "buffer_checkpoint_cap": args.buffer_checkpoint_cap,
        }
        # W0: the heuristic knobs are run-shaping (not pinned by `_check_config`), so a
        # resume may change them -- that is how you turn the prior off, or hand-anneal it.
        # The ANNEALED lambda itself comes from the checkpoint and is only overridden when
        # the flag was passed explicitly.
        for name in ("heuristic_prior", "heuristic_prior_floor", "heuristic_prior_anneal",
                     "heuristic_tau", "heuristic_exact_top", "heuristic_discard_bias",
                     "max_hand_candidates"):
            flag = "--" + name.replace("_", "-")
            if flag in sys.argv:
                overrides[name] = getattr(args, name)
        trainer = ColdTrainer.from_checkpoint(ckpt, overrides=overrides)
        run_name = args.run_name or ckpt_path.parent.name
        print(f"=== Resumed from {ckpt_path} "
              f"(saved {ckpt.get('saved_at')}, episode {trainer.counters.episodes}) ===")
        if not ckpt.get("buffer"):
            print("  ! checkpoint carried no replay buffer — this resume is NOT bit-exact")
        elif ckpt["buffer"].get("truncated"):
            print("  ! replay buffer was truncated at save time — resume is NOT bit-exact")
    else:
        cfg = config_from_args(args)
        trainer = ColdTrainer(cfg)
        run_name = args.run_name or f"train_cold_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ckpt = None

    cfg = trainer.cfg
    # `--resume` takes the ruleset from the checkpoint, so the CLI check above cannot see
    # it: catch an MLB run here too rather than let an old degenerate run be continued.
    if cfg.ruleset == "mlb":
        raise SystemExit(MLB_REFUSAL)

    run_dir = run_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / f"{run_name}.jsonl"
    log_file = open(log_path, "a")

    log_file.write(json.dumps({
        "kind": "config",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "config": {k: v for k, v in trainer.state_dict(include_buffer=False)["config"].items()},
        "n_params": sum(p.numel() for p in trainer.net.parameters()),
        "resumed_from": str(args.resume) if args.resume else None,
        "start_episode": trainer.counters.episodes,
    }) + "\n")
    log_file.flush()

    print(f"=== Cold-start training, {args.minutes:.1f} min ===")
    print(f"  run dir: {run_dir}")
    print(f"  log: {log_path}")
    print(f"  device: {cfg.device}, sims={cfg.sims}, m_init={cfg.max_considered}, "
          f"batch={cfg.batch_size}, lr={cfg.lr}, ruleset={cfg.ruleset}, "
          f"encoder={cfg.encoder}")
    print(f"  heuristic prior: lambda={trainer.heuristic_lambda:.2f} "
          f"(start {cfg.heuristic_prior}, floor {cfg.heuristic_prior_floor}, anneal "
          f"{cfg.heuristic_prior_anneal or 'none'}), tau={cfg.heuristic_tau}, "
          f"exact_top={cfg.heuristic_exact_top}, discard_bias="
          f"{cfg.heuristic_discard_bias}, max_hand_candidates="
          f"{cfg.max_hand_candidates or 'all'}")
    print(f"  net params: {sum(p.numel() for p in trainer.net.parameters()):,}")
    print(f"  checkpoint: every {args.checkpoint_every} ep -> {run_dir}/latest.pt "
          f"(buffer: {'yes' if cfg.checkpoint_buffer else 'NO'})")
    print()

    window: deque[dict] = deque(maxlen=50)
    start = time.time()
    deadline = start + args.minutes * 60
    ep_offset = trainer.counters.episodes
    base_elapsed = trainer.counters.elapsed_sec     # wall clock from previous runs
    n_checkpoints = 0
    interrupted = False

    def write_checkpoint(tag: str) -> Path:
        """Checkpoint the CURRENT state, with the cumulative wall clock folded in."""
        nonlocal n_checkpoints
        trainer.counters.elapsed_sec = base_elapsed + (time.time() - start)
        # Numbered checkpoints are weights-only (small, archival / for eval); latest.pt
        # carries the replay buffer, which is what makes a resume bit-exact. A Sample is
        # ~97 KB (action_features is (436, 56) at SELECTING_HAND), so a buffer-carrying
        # checkpoint is far bigger than a weights-only one: measured 29 MB vs 130 MB
        # after 141 episodes. Resume from latest.pt.
        target = run_dir / f"ckpt_{trainer.counters.episodes:06d}.pt"
        save_checkpoint(target, trainer.state_dict(include_buffer=False))
        latest = save_checkpoint(run_dir / "latest.pt", trainer.state_dict())
        prune_checkpoints(run_dir, args.keep_checkpoints)
        n_checkpoints += 1
        log_file.write(json.dumps({
            "kind": "checkpoint", "ep": trainer.counters.episodes,
            "path": str(target), "tag": tag,
            "bytes": target.stat().st_size,
            "latest_bytes": latest.stat().st_size,
        }) + "\n")
        log_file.flush()
        print(f"  [checkpoint] ep {trainer.counters.episodes} -> {target.name} "
              f"({target.stat().st_size/1e6:.1f} MB weights-only; latest.pt "
              f"{latest.stat().st_size/1e6:.1f} MB with buffer, {tag})", flush=True)
        return latest

    paused = False
    pause_path = run_dir / "PAUSE"
    try:
        while time.time() < deadline:
            if args.episodes is not None and (trainer.counters.episodes - ep_offset) >= args.episodes:
                break
            if pause_path.exists():          # same contract as train_mlb: touch <run-dir>/PAUSE
                paused = True
                print("")
                print("[PAUSE file found: " + str(pause_path) + "] checkpointing and exiting; "
                      "--resume removes it", flush=True)
                break
            rec = trainer.run_episode()
            ep = trainer.counters.episodes

            if rec["kind"] == "error":
                log_file.write(json.dumps(rec) + "\n")
                log_file.flush()
                if trainer.counters.errors <= 3:
                    print(f"  ! ep {ep} crashed ({rec['error']}); continuing")
            else:
                window.append(rec)
                log_file.write(json.dumps(rec) + "\n")
                if ep % 50 == 0:
                    log_file.flush()

            if args.checkpoint_every and ep % args.checkpoint_every == 0:
                write_checkpoint("periodic")

            if ep % args.log_every == 0 or (ep - ep_offset) == 1:
                elapsed = time.time() - start
                print_status(ep, list(window), elapsed,
                             max(0, deadline - time.time()),
                             len(trainer.buffer), trainer.counters.wins, ep_offset)

    except KeyboardInterrupt:
        interrupted = True
        print("\n[interrupted]")

    elapsed = time.time() - start
    final = write_checkpoint("interrupt" if interrupted else ("exit:PAUSE" if paused else "exit"))
    if paused:
        try:
            pause_path.unlink()
        except OSError:
            pass

    log_file.write(json.dumps({
        "kind": "summary",
        "episodes": trainer.counters.episodes,
        "episodes_this_run": trainer.counters.episodes - ep_offset,
        "wins": trainer.counters.wins,
        "errors": trainer.counters.errors,
        "train_steps": trainer.counters.train_steps,
        "elapsed_sec": trainer.counters.elapsed_sec,
        "buffer_final": len(trainer.buffer),
        "checkpoints": n_checkpoints,
        "final_checkpoint": str(final),
    }) + "\n")
    log_file.close()

    print()
    print(f"=== Done after {fmt_secs(elapsed)} "
          f"({trainer.counters.episodes - ep_offset} episodes this run, "
          f"{trainer.counters.episodes} total, {trainer.counters.wins} wins, "
          f"{trainer.counters.errors} errors, {n_checkpoints} checkpoints) ===")
    print(f"log:        {log_path}")
    print(f"checkpoint: {run_dir / 'latest.pt'}")
    print(f"resume:     python agent/scripts/train_cold.py --resume {run_dir / 'latest.pt'} "
          f"--minutes <N> --device {cfg.device}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
