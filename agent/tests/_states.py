"""
_states.py — a fixture of ~200 real game states for the Phase 4 W1 tests.

Provenance: the seeds are the first 40 `kind:"episode"` seeds of the overnight shakedown
run (`runs/overnight_2026-08-22/overnight_2026-08-22.jsonl`, read-only). They are read
from that file when it is present and fall back to the copy below otherwise, because
`agent/runs/` is gitignored and the tests must run on a clean checkout.

Each game is walked to `max_decisions` by a driver policy and a state is snapshotted every
`stride` decisions. Two drivers:

  * `collect_states()`            — a seeded uniform-random legal action. Fast (no torch,
                                    no checkpoint), deterministic, and it reaches all six
                                    game states, which is what the tests need.
  * `collect_states(policy=fn)`   — any `game -> action` callable, used by
                                    `benchmarks/bench_sample_size.py` to walk the same
                                    seeds under `ckpt_002072.pt`'s own learned policy.

The trajectories are NOT bit-identical to the logged ones: the run's Gumbel noise stream
is not recoverable from the JSONL. So these are "states from the overnight run's seeds",
not "the logged states" — see SETENC_NOTES §6.1.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from typing import Callable, Optional

import numpy as np

from balatro_sim.game import BalatroGame, State

_HERE = os.path.dirname(os.path.abspath(__file__))
_JSONL = os.path.join(_HERE, os.pardir, "runs", "overnight_2026-08-22",
                      "overnight_2026-08-22.jsonl")

#: Copied from the JSONL above on 2026-08-22 so the fixture survives a clean checkout.
OVERNIGHT_SEEDS = [
    1826701614, 1391214169, 1500766875, 1428383765, 1131031821, 84902227, 1523534768,
    1258450545, 1551178827, 1023827567, 1692230127, 1697906450, 1829060657, 264017423,
    2138054996, 2022015807, 845488399, 24366848, 866227862, 2005530613, 1907878242,
    1172093234, 1229124482, 1539371868, 1860037075, 1842037052, 1897652649, 1121983106,
    1375153445, 2121404236, 2053528816, 1886008386, 487136579, 118955683, 215293565,
    1292901445, 326856215, 1751257442, 23995739, 1718453068,
]

STATE_FIXTURE_SPEC = {
    "source": "runs/overnight_2026-08-22 episode seeds (first 40)",
    "ruleset": "mlb",
    "deck": "b_red",
    "stake": 1,
    "driver": "seeded uniform legal action (tests) / ckpt_002072.pt priors (benchmark)",
}


def overnight_seeds(n: int = 40) -> list[int]:
    """The run's own episode seeds, from the JSONL when it is there."""
    try:
        seeds: list[int] = []
        with open(_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("kind") == "episode":
                    seeds.append(int(rec["seed"]))
                if len(seeds) >= n:
                    break
        if seeds:
            return seeds
    except (OSError, ValueError, KeyError):
        pass
    return OVERNIGHT_SEEDS[:n]


def collect_states(n_states: int = 200, seeds: Optional[list[int]] = None,
                   policy: Optional[Callable[[BalatroGame], dict]] = None,
                   stride: int = 7, max_decisions: int = 400, rng_seed: int = 0,
                   ruleset: str = "mlb", deck_key: str = "b_red", stake: int = 1,
                   max_antes: Optional[int] = 8) -> list[BalatroGame]:
    """`n_states` live `BalatroGame`s, snapshotted (cloned) along real trajectories."""
    rng = np.random.default_rng(rng_seed)
    seeds = seeds if seeds is not None else overnight_seeds()
    out: list[BalatroGame] = []
    for seed in seeds:
        game = BalatroGame(seed=int(seed), deck_key=deck_key, stake=stake, ruleset=ruleset)
        for step in range(max_decisions):
            if game.state is State.GAME_OVER:
                break
            if max_antes is not None and game.ante > max_antes:
                break
            legal = game.legal_actions()
            if not legal:
                break
            if step % stride == 0:
                out.append(game.clone())
                if len(out) >= n_states:
                    return out
            if policy is not None:
                action = policy(game)
            else:
                action = legal[int(rng.integers(0, len(legal)))]
            game.step(action)
    return out


def state_histogram(games) -> dict:
    return dict(Counter(g.state.name for g in games))


def walk_states(game: BalatroGame, max_steps: int = 120, rng_seed: int = 0):
    """Yield `(game, legal_actions)` under an always-win policy so the walk reaches SHOP /
    BOOSTER_OPEN / ROUND_EVAL and (under MLB) the Nemesis.

    Same shape as `tests/test_action_space.py::_walk_states` — the Phase 3 test that
    established seed `7I4M53DL` reaches `reroll_boss` (Directors Cut / Retcon), the one
    action type a random walk essentially never finds.
    """
    rng = np.random.default_rng(rng_seed)
    for _ in range(max_steps):
        legal = game.legal_actions()
        yield game, legal
        if game.state is State.GAME_OVER or not legal:
            return
        if game.state is State.SELECTING_HAND:
            game.debug_win_blind()
            continue
        if game.state is State.SHOP:
            buys = [a for a in legal if a["type"] == "buy"]
            act = (buys[int(rng.integers(len(buys)))] if buys and rng.random() < 0.6
                   else {"type": "leave_shop"})
            game.step(act)
            continue
        game.step(legal[0])         # deterministic tail — this is what reaches reroll_boss
