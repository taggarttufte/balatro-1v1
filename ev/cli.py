"""
cli.py -- the advisor CLI (Phase 5 rev 2, W6).

    python mp/ev/cli.py advise fixture:bloodstone_vs_invisible --player 0
    python mp/ev/cli.py advise fixture:bloodstone_vs_invisible --player 1 --rollouts 32
    python mp/ev/cli.py advise replay:mp/replay/some_match.jsonl:40 --player 0
    python mp/ev/cli.py advise seed:11111111:120 --player 1 --checkpoint mp/agent/runs/v1/latest.pt

See ``mp/ev/advisor.py`` for what each section means and ``mp/ev/ADVISOR_NOTES.md`` for the
full writeup (how each of the three P(win) numbers is computed, the fixture's construction).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent            # mp/ev
_MP = _HERE.parent
for _p in (str(_HERE), str(_MP), str(_MP / "eval"), str(_MP / "agent"), str(_MP / "stats")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import advisor  # noqa: E402


def _cmd_advise(args) -> int:
    t0 = time.perf_counter()
    match, default_player = advisor.load_state_source(
        args.state_source, policy_seed=args.policy_seed, budget=args.drive_budget)
    load_s = time.perf_counter() - t0
    player = args.player if args.player is not None else default_player

    text = advisor.advise(
        match, player, n_rollouts=args.rollouts, rollout_seed=args.rollout_seed,
        rollout_budget=args.rollout_budget, budget=args.budget, checkpoint=args.checkpoint,
        top_n=args.top_n)

    print(f"[state source {args.state_source!r} loaded in {load_s:.2f}s]")
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"\n[also wrote {args.out}]")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="ev-cli", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    adv = sub.add_parser("advise", help="print the snapshot advisor report for one player")
    adv.add_argument("state_source",
                     help="fixture:<name> | replay:<path>:<step> | seed:<seed>:<step>")
    adv.add_argument("--player", type=int, default=None, choices=(0, 1),
                     help="which player to advise (default: the state source's default, 0)")
    adv.add_argument("--checkpoint", default=None, help="V checkpoint path (optional)")
    adv.add_argument("--rollouts", type=int, default=32, help="n_rollouts for the label estimator")
    adv.add_argument("--rollout-seed", type=int, default=0)
    adv.add_argument("--rollout-budget", default="fast", choices=("fast", "full"),
                     help="EVPlayer budget used INSIDE each rollout (default fast, matches "
                          "the label generator)")
    adv.add_argument("--budget", default="full", choices=("fast", "full"),
                     help="EVPlayer budget for the printed ranked-action table (default full)")
    adv.add_argument("--top-n", type=int, default=8, help="how many ranked actions to print")
    adv.add_argument("--policy-seed", type=int, default=0,
                     help="policy seed for seed:/fixture: self-play driving")
    adv.add_argument("--drive-budget", default="fast", choices=("fast", "full"),
                     help="EVPlayer budget used to DRIVE a seed: state source to its step")
    adv.add_argument("--out", default=None, help="also write the report to this file")
    adv.set_defaults(func=_cmd_advise)

    return ap


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
