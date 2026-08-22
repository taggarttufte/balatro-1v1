"""
mp/eval/rho_decay.py -- measure rho(h): the correlation of paired-arm outcomes h antes after
a single divergent decision, and the variance-reduction factor that correlation implies for a
paired (common-random-numbers) experiment design.

This is the experiment MP_TRAINING_DESIGN_2026-08.md §1 asks for and guesses numbers for
("~0.9 at h=1 falling to ~0.3-0.5 at h=4-8") -- see mp/eval/EVAL_NOTES.md for the measured
numbers and how they compare.

Design (Phase 3 brief §W4): two `BalatroGame(seed, ruleset="mlb")` "arms" on the SAME seed,
driven by the IDENTICAL scripted policy (`BASE_SPEC`) except for ONE decision at ante 1 (the
"perturbation", pluggable -- see `PERTURBATIONS`), then both play forward with the identical
policy.  Outcomes are read at horizons h in {1, 2, 4, 8} antes after the perturbation: the
Nemesis at ante `1 + h` is always played to hand-exhaustion (`play_arm_to_horizons` in
common.py; the Nemesis never ends early under MLB regardless of target -- see that function's
docstring) against an EXTERNAL target (`external_vanilla_big_blind_target`: the vanilla
Big-Blind chip requirement for that ante, a function of ante/deck/stake only, coupled to
NEITHER arm) so the two arms' scores are not coupled through a shared live opponent.

    python -m mp.eval.rho_decay --perturbation buy_slot0 --horizons 1,2,4,8 \\
        --out mp/results/rho_decay_buy_slot0.json
    python -m mp.eval.rho_decay --all --n-extra-seeds 24 --out-dir mp/results
    python -m mp.eval.rho_decay --list-perturbations
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Callable, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import common as C  # noqa: E402

BalatroGame = C.BalatroGame
State = C.State
ScriptedPlayer = C.ScriptedPlayer
make_policy = C.make_policy
shelf_indices = C.shelf_indices

DEFAULT_HORIZONS = [1, 2, 4, 8]
# The "forward" policy after the perturbation MUST keep making economic decisions -- if it
# never buys anything, jokers/vouchers never differ between arms and log-score is driven
# ENTIRELY by the per-ante `shuffle` stream, which every shop/pack/reroll decision is
# provably independent of (Phase 1 stream-independence invariant): a from-empty greedy-only
# base policy gave rho == 1.000 EXACTLY at every horizon for every perturbation (verified
# empirically) because nothing ever touched the joker/money state that could feed back into
# scoring. `buy_slot0=True` makes the base policy keep buying every shop it can afford,
# so the one-off perturbation's money difference propagates into different affordability,
# different joker portfolios, and genuinely different downstream scores -- see EVAL_NOTES.md.
BASE_SPEC = ScriptedPlayer(name="rho_base", hand="greedy", buy_slot0=True)
RHO_ARM_LIVES = 999    # generous: a regular-blind fail or Nemesis loss must never truncate a
                       # run before every requested horizon is reached (common.py note)

PERTURBATIONS = {
    "buy_slot0": "ante-1 first shop: arm A buys shelf slot 0 if affordable, arm B does not (default)",
    "reroll_once": "ante-1 first shop: arm B rerolls once (if affordable), arm A does not",
    "skip_small": "ante-1: arm B skips the Small blind entirely (no shop after a skip), arm A plays it normally",
}

SEED_ALPHABET = "123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"   # any 8-char string is a valid seed
                                                          # (normalize_seed maps '0'->'O'); this
                                                          # alphabet just avoids relying on that.


def make_extra_seeds(n: int, rng_seed: int = 20260821) -> list:
    """`n` extra synthetic seeds, deterministic given `rng_seed` (so a rerun reproduces the
    same seed list without committing a hardcoded one)."""
    rng = random.Random(rng_seed)
    return ["".join(rng.choice(SEED_ALPHABET) for _ in range(8)) for _ in range(n)]


def _legal(acts: list, cand: dict) -> bool:
    return any(a == cand for a in acts)


def _wrap_perturbation(base_policy: Callable, perturbation: str, arm: str) -> Callable:
    """Wrap `base_policy` (a `(shim, p, acts) -> action` policy) so that the FIRST time the
    relevant decision point is reached, `arm`'s perturbation is applied instead of the base
    decision; every other call (before and after) falls through to `base_policy` unchanged.
    This is what makes "diverge at ONE decision, then play forward identically" true instead
    of a persistently-different policy."""
    done = {"flag": perturbation == "none"}   # "none": never overrides -- both arms identical
                                               # (rho(h) == 1 sanity check / test fixture)

    def pol(m, p: int, acts: list) -> dict:
        g = m.games[p]
        if not done["flag"]:
            if perturbation == "buy_slot0" and g.state == State.SHOP:
                done["flag"] = True
                if arm == "A":
                    return base_policy(m, p, acts)   # BASE_SPEC.buy_slot0=True already buys it
                return {"type": "leave_shop"}          # arm B: deliberately skip THIS ONE buy
            if perturbation == "reroll_once" and g.state == State.SHOP:
                done["flag"] = True
                if arm == "B":
                    cand = {"type": "reroll"}
                    if _legal(acts, cand):
                        return cand
                return base_policy(m, p, acts)
            if (perturbation == "skip_small" and g.state == State.BLIND_SELECT
                    and g.ante == 1 and g.blind_idx == 0):
                done["flag"] = True
                if arm == "B":
                    return {"type": "skip_blind"}
                return base_policy(m, p, acts)
        return base_policy(m, p, acts)

    return pol


def make_perturbed_game(seed: str, perturbation: str, arm: str, deck_key: str = "b_red",
                        stake=1, lives: int = RHO_ARM_LIVES, base_spec: ScriptedPlayer = BASE_SPEC):
    """Build ONE arm: construct on `seed`, apply `perturbation` for `arm` ('A' / 'B') at its
    decision point, run forward (still through the SAME wrapped policy, which is a no-op past
    that one decision) to ante 1's Big-Blind BLIND_SELECT screen -- the common resume point
    both arms reach regardless of which perturbation fired -- then hand back the game plus the
    UNWRAPPED base policy for the caller to continue with (`play_arm_to_horizons`)."""
    if perturbation not in PERTURBATIONS and perturbation != "none":
        raise ValueError(f"unknown perturbation {perturbation!r} (have: {sorted(PERTURBATIONS)}, 'none')")
    base_policy = make_policy(base_spec)
    game = BalatroGame(seed=seed, deck_key=deck_key, stake=stake, ruleset="mlb")
    game.lives = lives
    wrapped = _wrap_perturbation(base_policy, perturbation, arm)
    C.run_until(game, wrapped, C.at_big_blind_select, max_steps=2000)
    return game, base_policy


def measure_rho(perturbation: str, seeds: Sequence[str], horizons: Sequence[int] = DEFAULT_HORIZONS,
                deck_key: str = "b_red", stake=1, target_fn: Optional[Callable] = None,
                n_boot: int = 2000, ci_seed: int = 0, lives: int = RHO_ARM_LIVES) -> dict:
    """Run both arms on every seed, then compute rho(h) (Pearson + Spearman, bootstrap CI),
    the paired-vs-unpaired variance-reduction factor, and the sample sizes it implies, for
    every horizon and for each of the three outcome variables (log-score, money, lives lost)."""
    if target_fn is None:
        target_fn = C.external_vanilla_big_blind_target
    horizons = list(horizons)
    per_seed = []
    t0 = time.time()
    for seed in seeds:
        gA, polA = make_perturbed_game(seed, perturbation, "A", deck_key, stake, lives)
        outA = C.play_arm_to_horizons(gA, polA, target_fn, horizons)
        gB, polB = make_perturbed_game(seed, perturbation, "B", deck_key, stake, lives)
        outB = C.play_arm_to_horizons(gB, polB, target_fn, horizons)
        per_seed.append({"seed": seed, "A": outA, "B": outB})
    wall_s = time.time() - t0

    per_horizon = {}
    for h in horizons:
        rows = [r for r in per_seed if r["A"].get(h) is not None and r["B"].get(h) is not None]
        n_reached = len(rows)
        metrics = {}
        for var, key in (("log_score", "log_score"), ("money", "dollars"), ("lives_lost", "cum_lives_lost")):
            xs = [r["A"][h][key] for r in rows]
            ys = [r["B"][h][key] for r in rows]
            pear = C.bootstrap_corr_ci(xs, ys, "pearson", n_boot=n_boot, seed=ci_seed)
            spear = C.bootstrap_corr_ci(xs, ys, "spearman", n_boot=n_boot, seed=ci_seed)
            diffs = [x - y for x, y in zip(xs, ys)]
            var_paired = _pvariance(diffs)
            var_unpaired = C.unpaired_control_variance(xs, ys, n_perm=500, seed=ci_seed)
            vrf = (var_unpaired / var_paired) if (var_paired and var_paired > 0) else float("nan")
            metrics[var] = {
                "n": n_reached, "pearson": pear, "spearman": spear,
                "var_paired_diff": var_paired, "var_unpaired_diff": var_unpaired,
                "variance_reduction_factor": vrf,
                "sample_size": {
                    "d=0.2": C.sample_size_per_arm(0.2, rho=pear["point"] if pear["point"] == pear["point"] else 0.0),
                    "d=0.5": C.sample_size_per_arm(0.5, rho=pear["point"] if pear["point"] == pear["point"] else 0.0),
                },
            }
        per_horizon[str(h)] = {"n_reached": n_reached, "metrics": metrics}

    return {
        "perturbation": perturbation,
        "description": PERTURBATIONS.get(perturbation, "no perturbation (test fixture): both arms identical"),
        "horizons": horizons, "n_seeds": len(seeds), "deck": deck_key, "stake": stake,
        "target_fn": getattr(target_fn, "__name__", repr(target_fn)),
        "wall_clock_s": wall_s,
        "per_horizon": per_horizon,
        "per_seed": per_seed,
    }


def _pvariance(xs: Sequence[float]) -> float:
    import statistics
    return statistics.pvariance(xs) if len(xs) > 1 else float("nan")


# ============================================================================ CLI

def _parse_seeds_arg(args) -> list:
    if args.seeds:
        seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = list(C.DEFAULT_SEEDS)
    if args.n_extra_seeds:
        seeds = seeds + make_extra_seeds(args.n_extra_seeds, rng_seed=args.extra_seed_rng)
    return seeds


def _summary_line(perturbation: str, result: dict) -> str:
    parts = [f"{perturbation}: N={result['n_seeds']} ({result['wall_clock_s']:.1f}s)"]
    for h in result["horizons"]:
        m = result["per_horizon"][str(h)]["metrics"]["log_score"]
        pear = m["pearson"]
        vrf = m["variance_reduction_factor"]
        parts.append(f"h={h}: rho={pear['point']:.3f} [{pear['lo']:.3f},{pear['hi']:.3f}] "
                     f"n={pear['n']} VRF={vrf:.2f}x")
    return "\n  ".join(parts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perturbation", choices=sorted(PERTURBATIONS), default="buy_slot0")
    ap.add_argument("--all", action="store_true", help="run every registered perturbation")
    ap.add_argument("--list-perturbations", action="store_true")
    ap.add_argument("--horizons", default="1,2,4,8")
    ap.add_argument("--seeds", default=None, help="comma-separated; default: all 126 ground-truth seeds")
    ap.add_argument("--n-extra-seeds", type=int, default=0, help="append this many synthetic seeds (cheap)")
    ap.add_argument("--extra-seed-rng", type=int, default=20260821)
    ap.add_argument("--deck", default="b_red")
    ap.add_argument("--stake", default="1")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=None, help="single-perturbation mode: JSON path")
    ap.add_argument("--out-dir", default=str(C.RESULTS_DIR), help="--all mode: directory for rho_decay_<p>.json")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if args.list_perturbations:
        for name, desc in PERTURBATIONS.items():
            print(f"{name}: {desc}")
        return 0

    stake = int(args.stake) if args.stake.isdigit() else args.stake
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    seeds = _parse_seeds_arg(args)

    names = sorted(PERTURBATIONS) if args.all else [args.perturbation]
    for name in names:
        result = measure_rho(name, seeds, horizons, deck_key=args.deck, stake=stake, n_boot=args.n_boot)
        if not args.quiet:
            print(_summary_line(name, result))
        if args.all:
            out_path = os.path.join(args.out_dir, f"rho_decay_{name}.json")
        else:
            out_path = args.out or str(C.RESULTS_DIR / f"rho_decay_{name}.json")
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1, default=str)
        if not args.quiet:
            print(f"  -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
