"""
train_mlb.py — generation-based training against an objective that costs something.

    python mp/agent/scripts/train_mlb.py --minutes 30 --device cuda \
        --n-agents 16 --seeds-per-gen 2 --max-ante 4 --sims 40

Why this script exists (CAMPAIGN_LOG 2026-08-22 07:35): `train_cold --ruleset mlb` learns
fine and learns the wrong thing. Under solo MLB a Nemesis blind is FREE — `pvp_solo=True`
auto-resolves it with no life lost — so the optimal policy is "skip 15 of 16 blinds, coast
to ante 9", and the value target's standard deviation collapsed from 0.18 to 0.07. Nothing
about the pipeline was broken; the objective was.

Two objectives replace it (`--objective`):

  tournament  (default, the real one)  Each generation runs `--seeds-per-gen` tournaments
              of `--n-agents` agents on ONE seed each (`mp/tournament`), with the population
              built from the current net (several root-noise seeds and search budgets) plus
              the last `--p-history` checkpoints. At every Nemesis the N x N matrix ranks
              the population; a current-net agent's value target is its rank at the NEXT
              Nemesis blended with its final standing (`--value-blend`, default 0.7/0.3).
              Only current-net agents produce samples.

  external    Solo episodes against an external per-ante chip target (W4's
              `mp/eval/targets.py` when present, else the identical local formula). The
              driver charges the life the engine will not: score below target at the
              Nemesis -> `game.lose_life()`. Cheap sanity objective, not the real one.

Pause / resume (Tagg wants his machine back sometimes)
------------------------------------------------------
    touch <run-dir>/PAUSE

The loop checks between tournaments (and between episodes under `--objective external`),
finishes the one in flight, trains on what it collected, writes a checkpoint, prints the
exact resume command and exits 0. Ctrl+C and SIGTERM take the same path. Resume with

    python mp/agent/scripts/train_mlb.py --resume <run-dir>/latest.pt --minutes <N>

which restores the net, the optimizer moments, the replay buffer, the numpy/torch/python
RNG states, the generation counter AND the opponent-checkpoint history — a resumed
generation faces the same population it would have faced. Delete the PAUSE file first (the
script does it for you on `--resume`).

Logging: `<run-dir>/<run-name>.jsonl` — one `config` line, one `generation` line per
generation (every metric in `GenerationMetrics` plus the population summary), one
`checkpoint` line per checkpoint, one `summary` line at exit. Console prints one line per
generation and shouts if the value-target sd falls under 0.15.

See `mp/agent/TRAIN_NOTES.md` for the gate run, the metric definitions and the command for
the first real run.
"""
from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import _bootstrap  # noqa: F401  (sys.path + fork guard)

from train import (
    GenerationMetrics, MLBTrainConfig, MLBTrainer, TrainConfig,
    latest_checkpoint, load_checkpoint, save_checkpoint,
)

DEFAULT_RUN_DIR = Path(_bootstrap.AGENT_ROOT) / "runs"
PAUSE_FILE = "PAUSE"


# ══════════════════════════════════════════════════════════════════════ helpers

def fmt_secs(s: float) -> str:
    s = int(max(0, s))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def make_trajectory_logger(run_dir: Path, enabled: bool, sig_every: int = 50):
    """W3 hook (`mp/replay`, REPLAY_NOTES 2). Returns `(factory, note)` where `factory(seed)`
    is a fresh `TrajectoryLogger` - one per AGENT per tournament, because W3's logger holds
    exactly one open episode and a tournament is N independent games.

    `sig_every=50`, not W3's default 10: signature capture is 25-35% of wall clock at 10 and
    under 2% at 50 (W3's measurement), and the training loop cares about episodes per minute.
    Replay still verifies every 50th step plus the final state.

    If `mp/replay` is missing the run says so once and continues - logging is not the job.
    """
    if not enabled:
        return None, "disabled"
    try:
        from train.selfplay import tournament_module
        tournament_module()                                   # puts mp/ on sys.path
        from replay.log import TrajectoryLogger               # type: ignore  # noqa: WPS433
    except Exception as e:                                    # noqa: BLE001
        return None, f"unavailable ({type(e).__name__}: {e})"
    path = run_dir / "trajectories.jsonl"

    def factory(seed, _cls=TrajectoryLogger, _p=str(path), _s=sig_every):
        return _cls(_p, sig_every=_s)

    return factory, f"{path} (sig_every={sig_every})"


def init_weights(trainer, path: str, device: str) -> str:
    """Load ANOTHER run's weights into a fresh trainer: no optimizer moments, no replay
    buffer, no counters, no opponent history.

    This is the Stage-A -> Stage-B hand-off. Stage A warms the net up on VANILLA
    (`train_cold.py --ruleset vanilla`), where failing a blind ends the run outright, so the
    net has to learn hand play and shop economy before anything else. Stage B then trains
    the MLB tournament objective from those weights. It is deliberately not `--resume`: the
    two stages have different objectives, so carrying Adam's moments or a replay buffer
    across the seam would be optimising against a distribution that no longer exists.

    The encoder must match; `load_state_dict` would fail with a shape error anyway, so this
    is the friendly version of the same refusal.
    """
    ckpt = load_checkpoint(path, map_location=device)
    old_enc, new_enc = ckpt.get("encoder"), trainer.cfg.encoder
    if old_enc is not None and old_enc != new_enc:
        raise SystemExit(f"--init {path}: that checkpoint uses encoder {old_enc!r} and this "
                         f"run uses {new_enc!r}. Warm up with the SAME encoder.")
    trainer.net.load_state_dict(ckpt["model"])
    old = ckpt.get("config") or {}
    counters = ckpt.get("counters") or {}
    return (f"=== Initialised weights from {path} (saved {ckpt.get('saved_at')}, ruleset "
            f"{old.get('ruleset')}, {counters.get('episodes')} episodes) - fresh optimizer, "
            f"buffer, counters and population ===")


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


def prune_checkpoints(ckpt_dir: Path, keep: int, protect: set) -> None:
    """Keep the newest `keep` numbered checkpoints; NEVER delete one the population is
    still playing against (`protect`), or a generation would silently face a smaller
    opponent pool than its config says."""
    numbered = sorted(ckpt_dir.glob("ckpt_gen*.pt"), key=lambda q: q.stat().st_mtime)
    for old in (numbered[:-keep] if keep > 0 else numbered):
        if str(old.resolve()) in protect:
            continue
        try:
            old.unlink()
        except OSError:
            pass


# ══════════════════════════════════════════════════════════════════════ CLI

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Generation-based MLB training (tournament or external-target objective)")
    # run length
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--generations", type=int, default=None,
                    help="stop after this many generations (in addition to --minutes)")
    ap.add_argument("--seed", type=int, default=0)
    # objective
    ap.add_argument("--objective", choices=["tournament", "external"], default="tournament")
    ap.add_argument("--n-agents", type=int, default=16, help="N in the N x N tournament")
    ap.add_argument("--m-current", type=int, default=None,
                    help="seats held by the current net (default: n_agents // 2, min 1)")
    ap.add_argument("--p-history", type=int, default=4,
                    help="how many past checkpoints stay in the opponent pool")
    ap.add_argument("--seeds-per-gen", type=int, default=2,
                    help="tournaments (= distinct game seeds) per generation")
    ap.add_argument("--max-ante", type=int, default=4, help="tournament horizon")
    ap.add_argument("--life-rule", choices=["paired", "median", "none"], default="paired")
    ap.add_argument("--anchors", type=float, default=0.25,
                    help="fraction of the population given to SCRIPTED never-skipping "
                         "reference players. They produce no samples; they exist so a "
                         "skip-everything policy has somebody to lose the rank comparison "
                         "to. 0 disables them - see TRAIN_NOTES sec.7")
    ap.add_argument("--value-blend", type=float, default=0.7,
                    help="weight on the SHORT-HORIZON rank target; 1-x goes to the final "
                         "match standing")
    ap.add_argument("--episodes-per-gen", type=int, default=16,
                    help="--objective external: solo episodes per generation")
    ap.add_argument("--target-kind", default="own_big_blind",
                    choices=["own_big_blind", "vanilla_boss", "table"],
                    help="--objective external: which target from mp/eval/targets.py. "
                         "own_big_blind mirrors the agent's own Big blind (~50/50 by "
                         "construction); vanilla_boss is a FIXED bar a cold net cannot "
                         "reach, which collapses the value target -- see TRAIN_NOTES sec.9")
    ap.add_argument("--target-multiplier", type=float, default=1.0)
    ap.add_argument("--target-floor", type=float, default=1.0,
                    help="--objective external: floor the Nemesis target at this multiple "
                         "of the ante's vanilla BIG-blind amount. Without it a skipped Big "
                         "blind makes an own_big_blind target 0, i.e. free")
    ap.add_argument("--margin-scale", type=float, default=1.0,
                    help="--objective external: logistic scale on the natural-log score margin")
    ap.add_argument("--no-pvp-relay", action="store_true",
                    help="--objective external: do NOT show the agent its target via "
                         "set_pvp_info. Matches the tournament (agents play the Nemesis "
                         "blind) and makes solo trajectories fully replayable")
    # search
    ap.add_argument("--sims", type=int, default=40, help="MCTS sims per decision (base budget)")
    ap.add_argument("--sims-budgets", default="1.0,0.5,1.5",
                    help="multipliers on --sims, cycled over the population's seats")
    ap.add_argument("--max-considered", type=int, default=8, help="Gumbel m_init")
    ap.add_argument("--leaf-batch", type=int, default=1,
                    help="within-tree leaf batching. BATCH_NOTES sec.7.1 recommends 16, but "
                         "that was measured at 500 sims; at 40-60 sims L=1/4/16 measure "
                         "238/217/218 sims/s on CUDA here, so L=1 (the exact search) wins")
    ap.add_argument("--no-reuse", action="store_true", help="disable tree reuse")
    ap.add_argument("--no-root-noise", action="store_true",
                    help="disable root Dirichlet noise on the current-net seats")
    # optimisation
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--buffer-capacity", type=int, default=20_000)
    ap.add_argument("--min-buffer", type=int, default=128)
    ap.add_argument("--train-steps", type=int, default=0,
                    help="optimizer steps per generation (0 = auto: new_samples/batch)")
    ap.add_argument("--max-train-steps", type=int, default=2_000)
    # net / game
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--n-res-blocks", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--encoder", choices=["v7", "mlb", "set"], default="mlb",
                    help="'set' is W1's set-based encoder (SETENC_NOTES 0)")
    ap.add_argument("--no-subsample", action="store_true",
                    help="keep every legal action in a Sample (W1 Sample v1 shape); the "
                         "default subsamples to visited + --k-unvisited random rows")
    ap.add_argument("--k-unvisited", type=int, default=8,
                    help="zero-visit actions kept per sample when subsampling")
    ap.add_argument("--max-samples-per-agent", type=int, default=2000,
                    help="hard cap on samples one agent may contribute to one tournament")
    ap.add_argument("--deck", default="b_red")
    ap.add_argument("--stake", type=int, default=1)
    ap.add_argument("--lives", type=int, default=4)
    ap.add_argument("--max-decisions", type=int, default=2000)
    # run dir / checkpoints / logging
    ap.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--checkpoint-every", type=int, default=1,
                    help="write a checkpoint every N generations")
    ap.add_argument("--keep-checkpoints", type=int, default=8)
    ap.add_argument("--no-checkpoint-buffer", action="store_true")
    ap.add_argument("--buffer-checkpoint-cap", type=int, default=5_000)
    ap.add_argument("--save-matrices", action="store_true",
                    help="write each tournament's per-ante .npz matrices under the run dir")
    ap.add_argument("--log-trajectories", action="store_true",
                    help="log every episode through W3's mp/replay TrajectoryLogger")
    ap.add_argument("--sig-every", type=int, default=50,
                    help="trajectory signature checkpoint interval; 10 costs 25-35%% of "
                         "wall clock, 50 costs <2%%")
    ap.add_argument("--max-skips-per-ante", type=int, default=None,
                    help="training-time cap on skip_blind per ante for the "
                         "sample-producing seats. A cold net cannot clear a Big blind, "
                         "so skipping is genuinely optimal and the policy converges on "
                         "97-99 pct skip before it learns to play a hand (TRAIN_NOTES "
                         "sec.7.2). Not an engine rule: it masks the candidate set, and "
                         "it is annealed away")
    ap.add_argument("--skip-cap-anneal-clear-rate", type=float, default=0.5,
                    help="lift the skip cap once the rolling blind-clear rate exceeds this")
    ap.add_argument("--skip-cap-anneal-generations", type=int, default=0,
                    help="lift the skip cap after this many generations (0 = only the "
                         "clear-rate criterion)")
    ap.add_argument("--init", default=None,
                    help="start from another checkpoint WEIGHTS ONLY (fresh optimizer, "
                         "buffer, population, counters). The Stage-A to Stage-B hand-off: "
                         "warm up on vanilla with train_cold.py, then train the "
                         "tournament objective from those weights. Not --resume, which "
                         "continues the same run")
    ap.add_argument("--resume", default=None,
                    help="path to a checkpoint, a run directory, or 'latest'")
    return ap


def configs_from_args(args) -> tuple:
    m_current = args.m_current if args.m_current is not None else max(1, args.n_agents // 2)
    budgets = tuple(float(x) for x in str(args.sims_budgets).split(",") if x.strip())
    cfg = TrainConfig(
        seed=args.seed, sims=args.sims, max_considered=args.max_considered,
        batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
        buffer_capacity=args.buffer_capacity, min_buffer=args.min_buffer,
        hidden=args.hidden, n_res_blocks=args.n_res_blocks,
        encoder=args.encoder, device=args.device, ruleset="mlb",
        deck_key=args.deck, stake=args.stake, lives=args.lives,
        max_decisions=args.max_decisions, max_antes=args.max_ante,
        subsample=not args.no_subsample, k_unvisited=args.k_unvisited,
        checkpoint_buffer=not args.no_checkpoint_buffer,
        buffer_checkpoint_cap=args.buffer_checkpoint_cap,
    )
    mlb = MLBTrainConfig(
        objective=args.objective, n_agents=args.n_agents, m_current=m_current,
        p_history=args.p_history, seeds_per_generation=args.seeds_per_gen,
        max_ante=args.max_ante, life_rule=args.life_rule, value_blend=args.value_blend,
        anchor_frac=args.anchors,
        sims_budgets=budgets or (1.0,), leaf_batch=args.leaf_batch,
        reuse=not args.no_reuse, noise_current=not args.no_root_noise,
        episodes_per_generation=args.episodes_per_gen, target_kind=args.target_kind,
        target_multiplier=args.target_multiplier, target_floor=args.target_floor,
        margin_scale=args.margin_scale,
        pvp_relay=not args.no_pvp_relay,
        max_skips_per_ante=args.max_skips_per_ante,
        skip_cap_anneal_clear_rate=args.skip_cap_anneal_clear_rate,
        skip_cap_anneal_generations=args.skip_cap_anneal_generations,
        train_steps=args.train_steps, max_train_steps=args.max_train_steps,
        max_samples_per_agent=args.max_samples_per_agent,
    )
    return cfg, mlb


# ══════════════════════════════════════════════════════════════════════ main

def main() -> int:
    args = build_parser().parse_args()
    run_root = Path(args.run_dir)

    if args.resume:
        ckpt_path = resolve_resume(args.resume, run_root)
        ckpt = load_checkpoint(ckpt_path, map_location=args.device)
        trainer = MLBTrainer.from_checkpoint(
            ckpt,
            overrides={"device": args.device,
                       "checkpoint_buffer": not args.no_checkpoint_buffer,
                       "buffer_checkpoint_cap": args.buffer_checkpoint_cap},
        )
        run_name = args.run_name or ckpt_path.parent.name
        resumed_note = (f"=== Resumed from {ckpt_path} (saved {ckpt.get('saved_at')}, "
                        f"generation {trainer.generation}) ===")
        if not ckpt.get("buffer"):
            resumed_note += "\n  ! checkpoint carried no replay buffer - resume is NOT bit-exact"
        elif ckpt["buffer"].get("truncated"):
            resumed_note += "\n  ! replay buffer was truncated at save time - NOT bit-exact"
    else:
        cfg, mlb = configs_from_args(args)
        trainer = MLBTrainer(cfg, mlb)
        run_name = args.run_name or f"train_mlb_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        resumed_note = init_weights(trainer, args.init, args.device) if args.init else None

    cfg, mlb = trainer.cfg, trainer.mlb
    run_dir = run_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    pause_path = run_dir / PAUSE_FILE
    if args.resume and pause_path.exists():
        pause_path.unlink()                 # a resume clears the pause that stopped the run
    log_path = run_dir / f"{run_name}.jsonl"
    log_file = open(log_path, "a", encoding="utf-8")

    traj_factory, traj_note = make_trajectory_logger(run_dir, args.log_trajectories,
                                                     sig_every=args.sig_every)

    def emit(rec: dict) -> None:
        log_file.write(json.dumps(rec, default=str) + "\n")
        log_file.flush()

    _sd = trainer.state_dict(include_buffer=False)
    emit({"kind": "config", "timestamp": datetime.now().isoformat(timespec="seconds"),
          "args": vars(args), "config": _sd["config"], "mlb": _sd["mlb"]["config"],
          "target_source": trainer.target_source,
          "n_params": sum(p.numel() for p in trainer.net.parameters()),
          "resumed_from": str(args.resume) if args.resume else None,
          "start_generation": trainer.generation, "trajectories": traj_note})

    if resumed_note:
        print(resumed_note)
    print(f"=== MLB training ({mlb.objective}), {args.minutes:.1f} min ===")
    print(f"  run dir:  {run_dir}")
    print(f"  log:      {log_path}")
    print(f"  device:   {cfg.device}, encoder={cfg.encoder}, sims={cfg.sims}, "
          f"leaf_batch={mlb.leaf_batch}, reuse={mlb.reuse}")
    if mlb.objective == "tournament":
        n_anchor = min(mlb.n_agents - mlb.m_current, round(mlb.anchor_frac * mlb.n_agents))
        print(f"  objective: N={mlb.n_agents} ({mlb.m_current} current + {n_anchor} scripted "
              f"anchors + {mlb.n_agents - mlb.m_current - n_anchor} past selves from the "
              f"last {mlb.p_history} checkpoints), {mlb.seeds_per_generation} seeds/gen, "
              f"max_ante={mlb.max_ante}, life_rule={mlb.life_rule}, "
              f"value_blend={mlb.value_blend}")
    else:
        print(f"  objective: solo vs external target ({trainer.target_source}, "
              f"x{mlb.target_multiplier}), {mlb.episodes_per_generation} ep/gen, "
              f"max_ante={mlb.max_ante}")
    print(f"  net params: {sum(p.numel() for p in trainer.net.parameters()):,}")
    if mlb.max_skips_per_ante is not None:
        print(f"  skip cap: {mlb.max_skips_per_ante}/ante on the current-net seats, "
              f"lifted at clear rate > {mlb.skip_cap_anneal_clear_rate}"
              + (f" or generation {mlb.skip_cap_anneal_generations}"
                 if mlb.skip_cap_anneal_generations else ""))
    print(f"  trajectories: {traj_note}")
    print(f"  pause with: touch {pause_path}")
    print()

    # ── stop conditions ─────────────────────────────────────────────────────
    start = time.time()
    deadline = start + args.minutes * 60
    signalled = {"hit": False, "why": ""}

    def _on_signal(signum, _frame):
        if not signalled["hit"]:
            signalled["hit"] = True
            signalled["why"] = signal.Signals(signum).name
            print(f"\n[{signalled['why']}] finishing the unit in flight, then checkpointing...",
                  flush=True)

    for sig in (signal.SIGINT, getattr(signal, "SIGTERM", None), getattr(signal, "SIGBREAK", None)):
        if sig is not None:
            try:
                signal.signal(sig, _on_signal)
            except (ValueError, OSError):
                pass          # not the main thread / unsupported on this platform

    def stop_reason():
        if signalled["hit"]:
            return signalled["why"]
        if pause_path.exists():
            return "PAUSE"
        if time.time() >= deadline:
            return "deadline"
        return None

    def stop_check() -> bool:
        return stop_reason() is not None

    # ── the loop ────────────────────────────────────────────────────────────
    gen0 = trainer.generation
    n_checkpoints = 0
    history_note = ""
    keep = max(args.keep_checkpoints, mlb.p_history + 1)

    def write_checkpoint(tag: str) -> Path:
        nonlocal n_checkpoints
        trainer.counters.elapsed_sec = base_elapsed + (time.time() - start)
        numbered = run_dir / f"ckpt_gen{trainer.generation:04d}.pt"
        save_checkpoint(numbered, trainer.state_dict(include_buffer=False))
        latest = save_checkpoint(run_dir / "latest.pt", trainer.state_dict())
        # The numbered checkpoint joins the opponent pool BEFORE pruning, so a checkpoint
        # that is still an opponent can never be deleted underneath the population.
        trainer.history.add(numbered.resolve(), trainer.generation)
        prune_checkpoints(run_dir, keep,
                          protect={e["path"] for e in trainer.history.entries})
        n_checkpoints += 1
        emit({"kind": "checkpoint", "generation": trainer.generation, "tag": tag,
              "path": str(numbered), "bytes": numbered.stat().st_size,
              "latest_bytes": latest.stat().st_size,
              "history": [e["generation"] for e in trainer.history.entries]})
        print(f"  [checkpoint] gen {trainer.generation} -> {numbered.name} "
              f"({numbered.stat().st_size / 1e6:.1f} MB; latest.pt "
              f"{latest.stat().st_size / 1e6:.1f} MB with buffer, {tag})", flush=True)
        return latest

    base_elapsed = trainer.counters.elapsed_sec
    why = None
    try:
        while True:
            why = stop_reason()
            if why is not None:
                break
            if args.generations is not None and (trainer.generation - gen0) >= args.generations:
                why = "generations"
                break
            out_dir = (str(run_dir / "matrices" / f"gen{trainer.generation:04d}")
                       if args.save_matrices else None)
            m = trainer.run_generation(stop_check=stop_check,
                                       traj_logger_factory=traj_factory,
                                       out_dir=out_dir)
            rec = {"kind": "generation", **m.as_dict(),
                   **{k: v for k, v in getattr(trainer, "extra_metrics", {}).items()},
                   "buffer": len(trainer.buffer),
                   "elapsed_s": time.time() - start}
            emit(rec)
            print(m.console_line(), flush=True)
            if m.collapsed and m.n_samples:
                if m.objective == "tournament":
                    print(f"    !! value-target sd {m.value_target_sd:.3f} <= "
                          f"{GenerationMetrics.ALARM_SD}: the objective has gone degenerate "
                          f"(tie_fraction {m.tie_fraction:.3f}, skip_rate {m.skip_rate:.3f})",
                          flush=True)
                else:
                    print(f"    !  value-target sd {m.value_target_sd:.3f} (mean "
                          f"{m.value_target_mean:.3f}): an ABSOLUTE target clusters when "
                          f"every episode fails or every episode succeeds. Re-scale with "
                          f"--target-multiplier, or use the tournament objective.",
                          flush=True)
            if args.checkpoint_every and (trainer.generation % args.checkpoint_every == 0):
                write_checkpoint("periodic")
    except KeyboardInterrupt:               # a second Ctrl+C, or one that beat the handler
        why = "KeyboardInterrupt"

    why = why or stop_reason() or "deadline"
    final = write_checkpoint(f"exit:{why}")
    elapsed = time.time() - start
    emit({"kind": "summary", "stop_reason": why,
          "generations": trainer.generation, "generations_this_run": trainer.generation - gen0,
          "episodes": trainer.counters.episodes, "samples": trainer.counters.samples,
          "train_steps": trainer.counters.train_steps,
          "errors": trainer.counters.errors, "elapsed_sec": trainer.counters.elapsed_sec,
          "buffer_final": len(trainer.buffer), "checkpoints": n_checkpoints,
          "final_checkpoint": str(final)})
    log_file.close()

    resume_cmd = (f"python mp/agent/scripts/train_mlb.py --resume {run_dir / 'latest.pt'} "
                  f"--minutes <N> --device {cfg.device}")
    print()
    print(f"=== Stopped ({why}) after {fmt_secs(elapsed)} - "
          f"{trainer.generation - gen0} generations this run, {trainer.generation} total, "
          f"{trainer.counters.episodes} episodes, {n_checkpoints} checkpoints ===")
    print(f"log:        {log_path}")
    print(f"checkpoint: {run_dir / 'latest.pt'}")
    if why == "PAUSE":
        print(f"paused by:  {pause_path}  (delete it, or just --resume: that removes it)")
    print(f"resume:     {resume_cmd}")
    print(history_note, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
