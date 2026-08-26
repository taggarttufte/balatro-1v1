"""
bench_sample_size.py — bytes per `Sample`, before and after (Phase 4 W1).

Measures the array payload of one training sample over 200 real states, in four shapes:

    v1  flat encoder, ALL legal actions          the Phase 3 record
    v2  flat encoder, subsampled                 subsampling alone
    v2  set encoder,  subsampled                 what a Phase 4 run will actually store
    v2  set encoder,  ALL legal actions          the set encoding's own cost

and the per-leaf CPU cost of the two featurizers, since that is the other thing the set
encoding changes.

    python agent/benchmarks/bench_sample_size.py
    python agent/benchmarks/bench_sample_size.py --checkpoint agent/runs/overnight_2026-08-22/ckpt_002072.pt

With `--checkpoint`, the 200 states are walked under that checkpoint's own priors
(argmax, no search) instead of a uniform random policy — the same seeds either way.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))
import _bootstrap  # noqa: E402,F401  (sys.path + fork guard)

sys.path.insert(0, str(_ROOT / "tests"))

import numpy as np  # noqa: E402

from _states import collect_states, state_histogram  # noqa: E402
from mcts.action import action_key  # noqa: E402
from mcts.action_features import featurize_actions  # noqa: E402
from mcts.action_features_set import featurize_actions_set  # noqa: E402
from mcts.encoder import get_encoder  # noqa: E402
from train.sample import SampleBuilder, sample_nbytes  # noqa: E402
from train.trajectory import Sample as SampleV1  # noqa: E402


def checkpoint_policy(path: str, device: str = "cpu"):
    """`game -> action`: the checkpoint's prior argmax, no search."""
    from mcts.player import load_policy
    policy = load_policy(path, device=device, batched=True)

    def act(game):
        priors, _ = policy(game)
        if not priors:
            return game.legal_actions()[0]
        best = max(priors, key=priors.get)
        by_key = {action_key(a): a for a in game.legal_actions()}
        return by_key[best]
    return act


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-states", type=int, default=200)
    ap.add_argument("--k-unvisited", type=int, default=8)
    ap.add_argument("--visited", type=int, default=8,
                    help="simulated number of visited actions per state")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    policy = checkpoint_policy(args.checkpoint, args.device) if args.checkpoint else None
    t0 = time.perf_counter()
    states = collect_states(args.n_states, policy=policy)
    print(f"{len(states)} states in {time.perf_counter() - t0:.1f}s "
          f"({'checkpoint priors' if policy else 'uniform random'})")
    print("  " + ", ".join(f"{k}={v}" for k, v in sorted(state_histogram(states).items())))

    v7 = get_encoder("v7")
    st = get_encoder("set")
    rng = np.random.default_rng(0)
    builders = {
        "v2 flat  subsampled": SampleBuilder(v7, k_unvisited=args.k_unvisited, rng=rng),
        "v2 set   subsampled": SampleBuilder(st, k_unvisited=args.k_unvisited, rng=rng),
        "v2 set   all actions": SampleBuilder(st, subsample=False),
    }
    sizes: dict[str, list[int]] = defaultdict(list)
    by_state: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    n_legal = []

    for game in states:
        legal = game.legal_actions()
        if not legal:
            continue
        keys = [action_key(a) for a in legal]
        n_legal.append(len(legal))
        idx = rng.choice(len(keys), size=min(args.visited, len(keys)), replace=False)
        visits = {keys[i]: int(rng.integers(1, 40)) for i in idx}

        v1 = SampleV1(obs=v7(game), action_features=featurize_actions(legal),
                      target_policy=np.zeros(len(legal), dtype=np.float32))
        sizes["v1 flat  all actions"].append(sample_nbytes(v1))
        by_state[game.state.name]["v1 flat  all actions"].append(sample_nbytes(v1))
        for name, builder in builders.items():
            n = sample_nbytes(builder(game, legal, keys, visits, None, 0.0))
            sizes[name].append(n)
            by_state[game.state.name][name].append(n)

    base = float(np.mean(sizes["v1 flat  all actions"]))
    print(f"\nmean legal actions {np.mean(n_legal):.0f} (max {max(n_legal)})")
    print(f"\n{'shape':<24}{'mean B':>10}{'median B':>10}{'max B':>10}{'vs v1':>9}")
    for name in ("v1 flat  all actions", "v2 flat  subsampled",
                 "v2 set   subsampled", "v2 set   all actions"):
        vals = sizes[name]
        print(f"{name:<24}{np.mean(vals):>10.0f}{np.median(vals):>10.0f}"
              f"{np.max(vals):>10.0f}{base / np.mean(vals):>8.1f}x")

    print(f"\nper game state (mean bytes):")
    print(f"{'state':<18}{'v1':>10}{'v2 flat':>10}{'v2 set':>10}{'v1/v2set':>10}")
    for state in sorted(by_state):
        row = by_state[state]
        a = np.mean(row["v1 flat  all actions"])
        b = np.mean(row["v2 flat  subsampled"])
        c = np.mean(row["v2 set   subsampled"])
        print(f"{state:<18}{a:>10.0f}{b:>10.0f}{c:>10.0f}{a / c:>9.1f}x")

    # what that means for a buffer / a checkpoint
    print("\nbuffer footprint at 200,000 samples:")
    for name in ("v1 flat  all actions", "v2 flat  subsampled", "v2 set   subsampled"):
        print(f"  {name:<24}{np.mean(sizes[name]) * 200_000 / 1e9:>8.2f} GB")

    # featurizer cost per leaf
    big = max(states, key=lambda g: len(g.legal_actions()))
    legal = big.legal_actions()
    for label, fn in (("featurize_actions (56-dim)", lambda: featurize_actions(legal)),
                      ("featurize_actions_set", lambda: featurize_actions_set(big, legal, st.caps)),
                      ("encode_obs v7", lambda: v7(big)),
                      ("SetEncoder", lambda: st(big))):
        t = time.perf_counter()
        for _ in range(50):
            fn()
        print(f"\n{label:<28}{(time.perf_counter() - t) / 50 * 1000:.3f} ms "
              f"({len(legal)} actions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
