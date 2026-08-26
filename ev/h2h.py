"""
h2h.py -- head-to-head evals, paired by seed, BOTH seatings per seed (Phase 5 rev 2, W6).

    python ev/h2h.py --a ev:fast --b ev:full --n-seeds 30 --procs 4 \
        --out-json results/h2h_ev_fast_vs_ev_full.json --out-md results/h2h_ev_fast_vs_ev_full.md

Player specs (``build_player``):

    ev:fast              EVPlayer(budget="fast")
    ev:full               EVPlayer(budget="full")
    ev:full+stats         EVPlayer(budget="full", stats=stats/decide.py)
    ev:full+Vleaf          EVPlayer(budget="full", value_fn=V, value_fn_leaf_only=True) via
                           MatchAwareEVPlayer -- V values the full-budget hand rollout's
                           LEAF only (SHOP/BOOSTER_OPEN/BLIND_SELECT keep the rules tier);
                           V = ev/runs/v_full_best/ckpt_0001000.pt (W-LEAF, Phase 5 rev 2
                           V2 round) unless ``--checkpoint`` overrides it; the K=3 x 8-world
                           leaf resolution (EV_NOTES §8.3) applies automatically (hand.py)
    real1:det              the non-clairvoyant `real1` MCTS baseline
                           (mcts.determinize.make_determinized_player on
                           agent/runs/real1/latest.pt, real1.sh's Stage B search flags)
    real1:clair             the SAME checkpoint, clairvoyant search (table-only baseline --
                           mirrors measure_clairvoyance.py's REAL1_FLAGS exactly)
    scripted:<fields>      scripts/mlb_match_demo.ScriptedPlayer via eval/common.py's
                           existing spec parser (e.g. "scripted:hand=greedy,reroll=1")

One seed -> TWO matches (A as player 0 / B as player 1, and the mirror), so seat bias
(the fixed alternation order `MLBMatch.current_player()` uses, first-mover effects in a
shared-seed shop) cancels in the aggregate. A single worker job covers BOTH seatings of one
seed (the checkpoint/tree for an MCTS player is built once per job, not per match) --
``--procs`` bounds concurrent worker processes via a spawn ``multiprocessing.Pool``, matching
``ev/gate_ev_player.py`` / ``agent/scripts/measure_clairvoyance.py``'s pattern.

Output: JSON with every trial record + a summary (win rate + bootstrap CI via
``eval/common.bootstrap_ci``, mean final ante both sides, mean lives margin, Nemesis win
rate, wall-clock per match), and a human-readable MD table. See ``ADVISOR_NOTES.md`` "H2H
JSON schema" for the exact field list -- ``test_h2h.py`` pins it.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
import zlib
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent            # ev
_MP = _HERE.parent
for _p in (str(_HERE), str(_MP), str(_MP / "eval"), str(_MP / "agent"), str(_MP / "stats")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _bootstrap  # noqa: E402,F401
from _bootstrap import MLBMatch  # noqa: E402

import common as C  # noqa: E402  (eval/common.py: DEFAULT_SEEDS, bootstrap_ci, make_player_policy)

__all__ = ["build_player", "run_h2h", "write_report", "REAL1_FLAGS", "REAL1_CKPT_DEFAULT"]

REAL1_CKPT_DEFAULT = str(_MP / "agent" / "runs" / "real1" / "latest.pt")
# W-LEAF's keeper V checkpoint (Phase 5 rev 2 V2 round, lever (c)): CPU by default -- the
# ops cap is on concurrent torch-LOADING processes, not device, and CUDA across an 8-worker
# spawn pool contends for one GPU for no benefit at this batch size (1 state/call).
VLEAF_CKPT_DEFAULT = str(_MP / "ev" / "runs" / "v_full_best" / "ckpt_0001000.pt")
# Exactly measure_clairvoyance.py's REAL1_FLAGS -- real1.sh's Stage B search hyperparameters,
# the only thing that must be shared between the clairvoyant and determinized arms.
REAL1_FLAGS = dict(
    encoder="set", strategy="gumbel", heuristic_prior=0.4, heuristic_tau=0.35,
    max_hand_candidates=32, heuristic_exact_top=8, heuristic_discard_bias=1.0,
)


def _stable_seed(text: str) -> int:
    """A reproducible (across processes / runs / Python versions) int from ``text`` --
    ``hash(str)`` is per-process salted in CPython, so it cannot seed a worker deterministically
    (matches the reasoning `agent/mcts/determinize.py` gives for using `hashlib`/`crc32`
    over `hash()`)."""
    return zlib.crc32(text.encode("utf-8"))


def _decide_module():
    import decide  # stats/decide.py -- stats is on sys.path (header block above)
    return decide


def build_player(spec: str, seed: int, *, sims: int = 40, checkpoint: Optional[str] = None):
    """``spec`` -> ``(policy_fn, player_obj_or_None)``.  ``policy_fn(match, p, acts) ->
    action`` (``MLBMatch.play_out``'s signature); ``player_obj`` is the underlying
    ``.act(game)``/``.reset()`` object when there is one (for `.reset()` between matches),
    else ``None`` (scripted policies are already stateless closures)."""
    kind, _, body = spec.partition(":")
    if kind == "ev":
        import player as P  # ev/player.py, W3
        tokens = {t for t in body.split("+") if t}
        budget = "full" if "full" in tokens else "fast"
        stats = _decide_module() if "stats" in tokens else None
        if "Vleaf" in tokens:
            # W-LEAF: lever (c), V at the expectimax leaf ONLY -- via the existing
            # MatchAwareEVPlayer wrapper (ev/match_player.py, W5) so the opponent view V
            # sees is bound from the live match, not a bare clone.  `.policy()` gives the
            # (match, p, acts) -> action form `_one_worker_job` needs (NOT `C.adapt_player`,
            # which has no match to bind the opponent view from); the wrapper itself is the
            # returned "player_obj" (`.reset()` clears its per-seat EVPlayers too).
            # `value_fn_leaf_only=True`: the brief's own section 0 already measured "argmax-V
            # as a policy loses to the rules player 2/60" -- a DIFFERENT, already-known-bad
            # thing from lever (c). Without this flag the SAME value_fn would also argmax
            # SHOP/BOOSTER_OPEN/BLIND_SELECT (EVPlayer's existing, broader value_fn contract),
            # which would silently re-measure that known failure instead of the leaf lever.
            import match_player as MPl
            net, encoder = MPl.load_value(checkpoint or VLEAF_CKPT_DEFAULT, device="cpu")
            obj = MPl.MatchAwareEVPlayer(net, encoder, device="cpu", budget=budget, seed=seed,
                                         epsilon=0.0, stats=stats, name=spec,
                                         value_fn_leaf_only=True)
            return obj.policy(), obj
        obj = P.EVPlayer(budget=budget, stats=stats, seed=seed, epsilon=0.0, name=spec)
        return C.adapt_player(obj), obj
    if kind == "real1":
        ckpt = checkpoint or REAL1_CKPT_DEFAULT
        if body == "det":
            from mcts.determinize import make_determinized_player
            obj = make_determinized_player(checkpoint=ckpt, sims=sims, seed=seed,
                                           determinize_seed=seed, reuse=True, **REAL1_FLAGS)
        elif body == "clair":
            from mcts.player import make_player
            obj = make_player(checkpoint=ckpt, sims=sims, seed=seed, reuse=True, **REAL1_FLAGS)
        else:
            raise ValueError(f"unknown real1 mode {body!r} in {spec!r} (want 'det' or 'clair')")
        return C.adapt_player(obj), obj
    if kind == "scripted":
        _, pol = C.make_player_policy(f"scripted:{body}")
        return pol, None
    raise ValueError(f"unknown player spec kind {kind!r} in {spec!r} "
                    f"(want 'ev:' / 'real1:' / 'scripted:')")


def _one_worker_job(job) -> list:
    """One seed, BOTH seatings.  Module-level for multiprocessing pickling (spawn re-imports
    this module fresh in the child, exactly `gate_ev_player.py` / `measure_clairvoyance.py`)."""
    (seed, spec_a, spec_b, seed_a, seed_b, sims, checkpoint, lives, max_steps,
     deck_key, stake) = job
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass

    pol_a, obj_a = build_player(spec_a, seed_a, sims=sims, checkpoint=checkpoint)
    pol_b, obj_b = build_player(spec_b, seed_b, sims=sims, checkpoint=checkpoint)

    trials = []
    for seating in (0, 1):
        if obj_a is not None:
            obj_a.reset()
        if obj_b is not None:
            obj_b.reset()
        if seating == 0:
            policies, a_idx, b_idx = [pol_a, pol_b], 0, 1
        else:
            policies, a_idx, b_idx = [pol_b, pol_a], 1, 0

        m = MLBMatch(seed=seed, deck_key=deck_key, stake=stake, lives=lives)
        t0 = time.perf_counter()
        m.play_out(policies, max_steps=max_steps)
        dt = time.perf_counter() - t0
        g = m.games
        nem_wins_p = [sum(1 for (_a, loser, _s0, _s1) in m.pvp_log if loser == 1),
                     sum(1 for (_a, loser, _s0, _s1) in m.pvp_log if loser == 0)]
        trials.append({
            "seed": str(seed), "seating": seating, "steps": m.steps, "done": bool(m.done),
            "seconds": dt,
            "a_win": (bool(m.winner == a_idx) if m.done else None),
            "lives_a": g[a_idx].lives, "lives_b": g[b_idx].lives,
            "lives_margin_a": g[a_idx].lives - g[b_idx].lives,
            "final_ante_a": g[a_idx].ante, "final_ante_b": g[b_idx].ante,
            "final_money_a": g[a_idx].dollars, "final_money_b": g[b_idx].dollars,
            "nem_wins_a": nem_wins_p[a_idx], "nem_wins_b": nem_wins_p[b_idx],
            "nem_total": len(m.pvp_log),
        })
    return trials


def run_h2h(spec_a: str, spec_b: str, seeds, *, procs: int = 1, sims: int = 40,
           checkpoint: Optional[str] = None, lives: int = 4, max_steps: int = 100_000,
           deck_key: str = "b_red", stake=1, seed_base: int = 0, progress: bool = False) -> dict:
    """Run the whole matchup: every seed x both seatings.  Returns the full result dict
    (``spec_a``/``spec_b``/config/``trials``/``summary`` -- see the module docstring's
    "H2H JSON schema" pointer)."""
    seeds = list(seeds)
    jobs = []
    for sd in seeds:
        seed_a = _stable_seed(f"{seed_base}:{sd}:A")
        seed_b = _stable_seed(f"{seed_base}:{sd}:B")
        jobs.append((sd, spec_a, spec_b, seed_a, seed_b, sims, checkpoint, lives, max_steps,
                    deck_key, stake))

    t0 = time.perf_counter()
    trials: list = []
    if procs and procs > 1:
        import multiprocessing as mpc
        with mpc.get_context("spawn").Pool(procs) as pool:
            for i, batch in enumerate(pool.imap_unordered(_one_worker_job, jobs, chunksize=1)):
                trials.extend(batch)
                if progress:
                    print(f"  {i + 1}/{len(jobs)} seeds done ({time.perf_counter() - t0:.0f}s)",
                         flush=True)
    else:
        for i, job in enumerate(jobs):
            trials.extend(_one_worker_job(job))
            if progress:
                print(f"  {i + 1}/{len(jobs)} seeds done ({time.perf_counter() - t0:.0f}s)",
                     flush=True)
    wall = time.perf_counter() - t0

    decided = [t for t in trials if t["a_win"] is not None]
    a_wins = sum(1 for t in decided if t["a_win"])
    b_wins = len(decided) - a_wins
    win_flags = [1.0 if t["a_win"] else 0.0 for t in decided]
    win_ci = C.bootstrap_ci(win_flags) if win_flags else {"point": float("nan"), "lo": float("nan"),
                                                          "hi": float("nan"), "n": 0}

    def _mean(xs):
        xs = list(xs)
        return (sum(xs) / len(xs)) if xs else float("nan")

    nem_wins_a_total = sum(t["nem_wins_a"] for t in trials)
    nem_total_total = sum(t["nem_total"] for t in trials)

    summary = {
        "n_trials": len(trials), "n_decided": len(decided),
        "a_wins": a_wins, "b_wins": b_wins, "undecided": len(trials) - len(decided),
        "win_rate_a": win_ci,
        "mean_final_ante_a": _mean(t["final_ante_a"] for t in trials),
        "mean_final_ante_b": _mean(t["final_ante_b"] for t in trials),
        "mean_lives_margin_a": _mean(t["lives_margin_a"] for t in trials),
        "nemesis_win_rate_a": (nem_wins_a_total / nem_total_total) if nem_total_total else float("nan"),
        "mean_seconds_per_match": _mean(t["seconds"] for t in trials),
    }
    return {
        "spec_a": spec_a, "spec_b": spec_b, "seeds": seeds, "n_seeds": len(seeds),
        "sims": sims, "checkpoint": checkpoint, "lives": lives, "max_steps": max_steps,
        "deck_key": deck_key, "stake": stake, "procs": procs, "seed_base": seed_base,
        "wall_clock_s": wall, "trials": trials, "summary": summary,
    }


def write_report(result: dict, path_json: str, path_md: str) -> None:
    Path(path_json).parent.mkdir(parents=True, exist_ok=True)
    Path(path_json).write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")

    s = result["summary"]
    lines = [
        f"# H2H: {result['spec_a']} vs {result['spec_b']}", "",
        f"- seeds: {result['n_seeds']} ({result['seeds']})", f"- trials: {s['n_trials']} "
        f"(both seatings per seed); decided {s['n_decided']}, undecided {s['undecided']}",
        f"- sims={result['sims']}  checkpoint={result['checkpoint']}  lives={result['lives']}  "
        f"max_steps={result['max_steps']}  deck={result['deck_key']}  stake={result['stake']}",
        f"- procs={result['procs']}  wall clock: {result['wall_clock_s']:.1f}s  "
        f"mean {s['mean_seconds_per_match']:.2f}s/match", "",
        "## Summary", "",
        f"- **A ({result['spec_a']}) wins**: {s['a_wins']} / {s['n_decided']}  "
        f"(win rate {s['win_rate_a']['point']:.3f}, 95% CI "
        f"[{s['win_rate_a']['lo']:.3f}, {s['win_rate_a']['hi']:.3f}])",
        f"- **B ({result['spec_b']}) wins**: {s['b_wins']} / {s['n_decided']}",
        f"- mean final ante: A {s['mean_final_ante_a']:.2f}  vs  B {s['mean_final_ante_b']:.2f}",
        f"- mean lives margin (A - B): {s['mean_lives_margin_a']:+.2f}",
        f"- Nemesis win rate (A's side of every resolved Nemesis): {s['nemesis_win_rate_a']:.3f}",
        "", "## Per-trial", "",
        "| seed | seating | a_win | lives A/B | ante A/B | nem A/B / total | steps | seconds |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for t in result["trials"]:
        lines.append(f"| {t['seed']} | {t['seating']} | {t['a_win']} | "
                    f"{t['lives_a']}/{t['lives_b']} | {t['final_ante_a']}/{t['final_ante_b']} | "
                    f"{t['nem_wins_a']}/{t['nem_wins_b']}/{t['nem_total']} | {t['steps']} | "
                    f"{t['seconds']:.2f} |")
    Path(path_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


# ═══════════════════════════════════════════════════════════════ CLI

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="player spec A (ev:fast / real1:det / scripted:...)")
    ap.add_argument("--b", required=True, help="player spec B")
    ap.add_argument("--seeds", default=None, help="comma-separated seed list (overrides --n-seeds)")
    ap.add_argument("--n-seeds", type=int, default=2, help="take the first N of DEFAULT_SEEDS")
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--sims", type=int, default=40, help="MCTS sims for real1:* specs")
    ap.add_argument("--checkpoint", default=None, help="override real1's checkpoint path")
    ap.add_argument("--lives", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=100_000)
    ap.add_argument("--deck-key", default="b_red")
    ap.add_argument("--stake", default=1)
    ap.add_argument("--seed-base", type=int, default=0, help="salts the per-player RNG seeds")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.seeds:
        seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = C.DEFAULT_SEEDS[args.seed_offset: args.seed_offset + args.n_seeds]
    stake = int(args.stake) if str(args.stake).lstrip("-").isdigit() else args.stake

    result = run_h2h(args.a, args.b, seeds, procs=args.procs, sims=args.sims,
                     checkpoint=args.checkpoint, lives=args.lives, max_steps=args.max_steps,
                     deck_key=args.deck_key, stake=stake, seed_base=args.seed_base,
                     progress=not args.quiet)

    date = _dt.date.today().isoformat()
    slug = f"{args.a}_vs_{args.b}".replace(":", "-").replace("+", "-").replace(",", "-")
    default_stem = C.RESULTS_DIR / f"h2h_{slug}_{date}"
    out_json = args.out_json or str(default_stem.with_suffix(".json"))
    out_md = args.out_md or str(default_stem.with_suffix(".md"))
    write_report(result, out_json, out_md)

    s = result["summary"]
    print(f"\nA={args.a}  B={args.b}  n_trials={s['n_trials']}  wall={result['wall_clock_s']:.1f}s")
    print(f"A wins {s['a_wins']}/{s['n_decided']}  (win rate {s['win_rate_a']['point']:.3f}  "
         f"CI [{s['win_rate_a']['lo']:.3f}, {s['win_rate_a']['hi']:.3f}])")
    print(f"mean final ante A/B: {s['mean_final_ante_a']:.2f} / {s['mean_final_ante_b']:.2f}   "
         f"lives margin A-B: {s['mean_lives_margin_a']:+.2f}   "
         f"Nemesis win rate A: {s['nemesis_win_rate_a']:.3f}")
    print(f"wrote {out_json} and {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
