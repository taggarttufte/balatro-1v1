"""
bench_batched.py — what batching and tree reuse are actually worth (Phase 3, W3).

Three tables, all on the same box, same state, same per-decision simulation budget:

  A. single-tree SERIAL — W1's baseline re-run (AGENT_NOTES §4.4), CPU and CUDA, so the
     comparison is against numbers produced by this run and not by a remembered one.
  B. BATCHED — K trees in lockstep for K = 1 / 8 / 32 / 100 on CUDA and on CPU, plus the
     within-tree virtual-loss variant (K = 1, leaf_batch = L) that a single agent can
     use. sims/s total and per tree, and where the time goes: NN / sim (clone + step) /
     Python overhead, measured with `time.perf_counter` around the real calls.
  C. REUSE — how much of the previous decision's tree survives, and what that buys:
     wall clock at a fixed evidence level ("subtract"), or evidence at a fixed wall clock
     ("add").

    python mp/agent/benchmarks/bench_batched.py                       # the full table
    python mp/agent/benchmarks/bench_batched.py --sims 200 --k 1 8    # a quick pass
    python mp/agent/benchmarks/bench_batched.py --only reuse

The throughput columns come from an UNinstrumented run; the split comes from a second,
instrumented one, normalised to its own total — the same protocol bench_search.py uses,
for the same reason (two perf_counter calls per clone/step is not free).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import _bootstrap  # noqa: E402,F401  (sys.path + fork guard)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from balatro_sim.game import BalatroGame, State  # noqa: E402
from mcts import (  # noqa: E402
    MCTS, MCTSConfig, MCTSPlayer, NNPolicy, PolicyValueNet, get_encoder,
    BatchedNNPolicy, BatchedSearch, ReuseConfig,
)


# ── Timing helpers ──────────────────────────────────────────────────────────

class SimTimer:
    """Times every `BalatroGame.clone` / `.step` for the duration of a `with` block."""

    def __init__(self):
        self.seconds = 0.0

    def __enter__(self):
        self._clone, self._step = BalatroGame.clone, BalatroGame.step
        outer = self

        def clone(self):
            t = time.perf_counter()
            try:
                return outer._clone(self)
            finally:
                outer.seconds += time.perf_counter() - t

        def step(self, action):
            t = time.perf_counter()
            try:
                return outer._step(self, action)
            finally:
                outer.seconds += time.perf_counter() - t

        BalatroGame.clone, BalatroGame.step = clone, step
        return self

    def __exit__(self, *exc):
        BalatroGame.clone, BalatroGame.step = self._clone, self._step
        return False


def _demo_state(seed: int, ruleset: str, nemesis: bool) -> BalatroGame:
    from mcts_demo import make_demo_state
    return make_demo_state(seed=seed, ruleset=ruleset, nemesis=nemesis)


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _row(label, device, k, leaf, sims_total, wall, nn_s, sim_s, mean_batch, n_trees):
    other = max(0.0, wall - nn_s - sim_s)
    denom = max(wall, 1e-9)
    print(f"{label:<22} {device:<5} {k:>4} {leaf:>4} {sims_total/wall:>10.0f} "
          f"{sims_total/wall/n_trees:>10.0f} {1000*wall/max(1,n_trees):>9.1f} "
          f"{100*nn_s/denom:>5.1f}% {100*sim_s/denom:>5.1f}% {100*other/denom:>6.1f}% "
          f"{mean_batch:>7.1f}")
    return {
        "config": label, "device": device, "k": k, "leaf_batch": leaf,
        "sims_per_s": sims_total / wall, "sims_per_s_per_tree": sims_total / wall / n_trees,
        "ms_per_decision": 1000 * wall / max(1, n_trees),
        "nn_pct": 100 * nn_s / denom, "sim_pct": 100 * sim_s / denom,
        "other_pct": 100 * other / denom, "mean_batch": mean_batch,
    }


HEADER = (f"{'config':<22} {'dev':<5} {'K':>4} {'L':>4} {'sims/s':>10} {'/tree':>10} "
          f"{'ms/dec':>9} {'nn%':>6} {'sim%':>6} {'other%':>7} {'batch':>7}")


# ── A. single-tree serial baseline ──────────────────────────────────────────

def bench_serial(args, game, encoder) -> list[dict]:
    print("\nA. single tree, serial (one forward pass per leaf) — W1's baseline, re-run")
    print(HEADER)
    out = []
    for device in args.devices:
        torch.manual_seed(0)
        policy = NNPolicy(PolicyValueNet(obs_dim=encoder.dim), device=device,
                          encoder=encoder)
        for strategy in args.strategies:
            cfg = MCTSConfig(num_simulations=args.sims)
            # warm-up (CUDA context, lazy kernels)
            MCTS(policy, MCTSConfig(num_simulations=4), rng=np.random.default_rng(0)).run(game)
            best = None
            for _ in range(args.repeat):
                mcts = MCTS(policy, cfg, rng=np.random.default_rng(0))
                t0 = time.perf_counter()
                _run(mcts, game, strategy)
                _sync(device)
                best = min(best or 1e9, time.perf_counter() - t0)
            # split: a separate instrumented run
            mcts = MCTS(policy, cfg, rng=np.random.default_rng(0))
            nn = [0.0]
            real_eval = mcts._evaluate_leaf

            def timed_eval(g, _nn=nn, _real=real_eval):
                t = time.perf_counter()
                try:
                    return _real(g)
                finally:
                    _nn[0] += time.perf_counter() - t

            mcts._evaluate_leaf = timed_eval
            with SimTimer() as sim:
                t0 = time.perf_counter()
                _run(mcts, game, strategy)
                _sync(device)
                total_i = time.perf_counter() - t0
            out.append(_row(f"serial {strategy}", device, 1, 1, args.sims, best,
                            nn[0] * best / total_i, sim.seconds * best / total_i, 1.0, 1))
    return out


def _run(mcts: MCTS, game: BalatroGame, strategy: str):
    if strategy == "gumbel":
        return mcts.run_gumbel(game)
    return mcts.run(game, add_noise=True)


# ── B. batched ──────────────────────────────────────────────────────────────

def bench_batched(args, game, encoder) -> list[dict]:
    print("\nB. batched leaf evaluation — K trees in lockstep (+ leaf_batch L within a tree)")
    print(HEADER)
    out = []
    for device in args.devices:
        torch.manual_seed(0)
        policy = BatchedNNPolicy(PolicyValueNet(obs_dim=encoder.dim), device=device,
                                 encoder=encoder)
        for strategy in args.strategies:
            for k in args.k:
                for leaf in args.leaf_batch:
                    if k > 1 and leaf > 1 and not args.cross_product:
                        continue     # K x L is measured only for K = 1 by default
                    out.append(_bench_one(args, policy, game, device, strategy, k, leaf))
    return out


def _bench_one(args, policy, game, device, strategy, k, leaf) -> dict:
    cfg = MCTSConfig(num_simulations=args.sims, leaf_batch=leaf)
    seeds = list(range(k))

    def once():
        search = BatchedSearch(policy, cfg, strategy=strategy)
        games = [game.clone() for _ in range(k)]
        t0 = time.perf_counter()
        search.run_many(games, seeds=seeds, add_noise=True)
        _sync(device)
        return time.perf_counter() - t0, search

    # warm-up
    BatchedSearch(policy, MCTSConfig(num_simulations=4, leaf_batch=leaf),
                  strategy=strategy).run_many([game.clone()], seeds=[0])

    best, best_search = None, None
    for _ in range(args.repeat):
        wall, search = once()
        if best is None or wall < best:
            best, best_search = wall, search

    # split: one instrumented run (clone/step timed; NN comes from BatchStats)
    search = BatchedSearch(policy, cfg, strategy=strategy)
    games = [game.clone() for _ in range(k)]
    with SimTimer() as sim:
        t0 = time.perf_counter()
        search.run_many(games, seeds=seeds, add_noise=True)
        _sync(device)
        total_i = time.perf_counter() - t0
    scale = best / total_i
    return _row(f"batched {strategy}", device, k, leaf, args.sims * k, best,
                search.stats.nn_seconds * scale, sim.seconds * scale,
                best_search.stats.mean_batch, k)


# ── C. tree reuse ───────────────────────────────────────────────────────────

def _episode(player: MCTSPlayer, game: BalatroGame, decisions: int, relay=None):
    """Drive one game with `player` for at most `decisions` decisions. `relay` is the
    driver mutation MLB really performs between decisions (`set_pvp_info`)."""
    t0 = time.perf_counter()
    n = 0
    for _ in range(decisions):
        if game.state is State.GAME_OVER:
            break
        if relay is not None:
            relay(game)
        action = player.act(game)
        if action is None:
            break
        game.step(action)
        n += 1
    return time.perf_counter() - t0, n


def _pvp_relay(game: BalatroGame) -> None:
    """What a tournament/`MLBMatch` driver does between decisions at a Nemesis: push the
    opponent's live score in. It changes the game state, so it invalidates the tree."""
    if getattr(game.current_blind, "is_pvp", False):
        game.set_pvp_info(int(game.chips_scored) + 1_000, 2)


def bench_reuse(args, encoder) -> list[dict]:
    print("\nC. tree reuse — retention and what it buys "
          f"({args.reuse_decisions} decisions, {args.reuse_sims} sims/decision)")
    print(f"{'scenario':<18} {'mode':<9} {'hit%':>6} {'retain%':>8} {'ret.N':>7} "
          f"{'dec/s':>7} {'eff.sims':>9} {'evid/s':>8}")
    device = "cuda" if "cuda" in args.devices else args.devices[0]
    torch.manual_seed(0)
    policy = BatchedNNPolicy(PolicyValueNet(obs_dim=encoder.dim), device=device,
                             encoder=encoder)

    scenarios = {
        "vanilla SP": (lambda: BalatroGame(seed=args.seed), None),
        "MLB solo": (lambda: BalatroGame(seed="7I4M53DL", ruleset="mlb"), None),
        "MLB Nemesis": (lambda: _demo_state(args.seed, "mlb", True), None),
        "Nemesis+relay": (lambda: _demo_state(args.seed, "mlb", True), _pvp_relay),
    }
    out = []
    for name, (make, relay) in scenarios.items():
        for mode in ("off", "subtract", "add"):
            reuse = (False if mode == "off"
                     else ReuseConfig(budget_mode=mode, min_new_sims=args.min_new_sims))
            player = MCTSPlayer(policy=policy,
                                config=MCTSConfig(num_simulations=args.reuse_sims),
                                strategy="gumbel", reuse=reuse,
                                rng=np.random.default_rng(0))
            wall, n = _episode(player, make(), args.reuse_decisions, relay)
            st = player.reuse_stats
            evidence = st.new_sims + st.retained_visits
            row = {
                "scenario": name, "mode": mode, "decisions": n,
                "hit_rate": st.hit_rate, "node_fraction": st.node_fraction,
                "retained_visits_per_decision": st.retained_visits / max(1, st.decisions),
                "decisions_per_s": n / wall, "effective_sims": st.effective_sims,
                "evidence_per_s": evidence / wall,
            }
            print(f"{name:<18} {mode:<9} {100*st.hit_rate:>5.1f}% "
                  f"{100*st.node_fraction:>7.1f}% "
                  f"{st.retained_visits/max(1, st.decisions):>7.1f} "
                  f"{n/wall:>7.2f} {st.effective_sims:>9.1f} {evidence/wall:>8.0f}")
            out.append(row)
    return out


# ── main ────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=500)
    ap.add_argument("--k", type=int, nargs="+", default=[1, 8, 32, 100])
    ap.add_argument("--leaf-batch", type=int, nargs="+", default=[1, 16])
    ap.add_argument("--devices", nargs="+", default=None,
                    help="default: cuda cpu when CUDA is available, else cpu")
    ap.add_argument("--strategies", nargs="+", default=["gumbel"],
                    choices=["gumbel", "puct"])
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ruleset", choices=["vanilla", "mlb"], default="vanilla")
    ap.add_argument("--nemesis", action="store_true")
    ap.add_argument("--encoder", choices=["v7", "mlb"], default="v7")
    ap.add_argument("--cross-product", action="store_true",
                    help="also measure K > 1 with L > 1")
    ap.add_argument("--reuse-decisions", type=int, default=30)
    ap.add_argument("--reuse-sims", type=int, default=100)
    ap.add_argument("--min-new-sims", type=int, default=0)
    ap.add_argument("--only", choices=["serial", "batched", "reuse"], nargs="+",
                    default=["serial", "batched", "reuse"])
    ap.add_argument("--json", type=str, default=None, help="write the rows to this path")
    args = ap.parse_args()

    if args.devices is None:
        args.devices = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
    encoder = get_encoder(args.encoder)
    game = _demo_state(args.seed, args.ruleset, args.nemesis)

    print(f"bench_batched: torch {torch.__version__} "
          f"device(s) {'/'.join(args.devices)} "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")
    print(f"state: ruleset={args.ruleset} nemesis={args.nemesis} "
          f"legal_actions={len(game.legal_actions())} sims/decision={args.sims} "
          f"repeat={args.repeat}")

    rows: list[dict] = []
    if "serial" in args.only:
        rows += bench_serial(args, game, encoder)
    if "batched" in args.only:
        rows += bench_batched(args, game, encoder)
    if "reuse" in args.only:
        rows += bench_reuse(args, encoder)

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
