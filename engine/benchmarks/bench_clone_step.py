"""
bench_clone_step.py - Baseline performance benchmarks for the Balatro sim.

Measures:
  1. Average wallclock per env step across full random games
  2. Average wallclock per deepcopy of a mid-game state
  3. Average wallclock per game.clone() if a clone() method exists, else skip

The clone:step ratio is the key number for MCTS feasibility. If clone is much
slower than step, every tree expansion pays a clone tax that PPO never paid.
"""
from __future__ import annotations
import copy
import random
import sys
import time
from pathlib import Path

# Make the sim importable when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from balatro_sim.game import BalatroGame, State


def random_legal_action(game: BalatroGame) -> dict:
    """Pick a syntactically legal action for the current state."""
    s = game.state
    if s == State.BLIND_SELECT:
        return {"type": "play_blind"}
    if s == State.SELECTING_HAND:
        n = len(game.hand)
        if n == 0:
            return {"type": "play", "cards": [0]}
        # Sometimes discard, mostly play
        if game.discards_left > 0 and random.random() < 0.25:
            k = random.randint(1, min(5, n))
            return {"type": "discard", "cards": random.sample(range(n), k)}
        k = random.randint(1, min(5, n))
        return {"type": "play", "cards": random.sample(range(n), k)}
    if s == State.ROUND_EVAL:
        return {"type": "advance"}
    if s == State.SHOP:
        return {"type": "leave_shop"}
    if s == State.BOOSTER_OPEN:
        return {"type": "skip_booster"}
    return {"type": "noop"}


def time_random_games(n_games: int = 50, max_steps: int = 2000) -> tuple[float, int]:
    """Play n random games, return (avg_step_seconds, total_steps)."""
    total_steps = 0
    t0 = time.perf_counter()
    for i in range(n_games):
        game = BalatroGame(seed=i)
        game.reset()
        for _ in range(max_steps):
            if game.state == State.GAME_OVER:
                break
            action = random_legal_action(game)
            game.step(action)
            total_steps += 1
    elapsed = time.perf_counter() - t0
    return elapsed / max(1, total_steps), total_steps


def make_midgame_state(seed: int = 7) -> BalatroGame:
    """
    Construct a rich mid-game state for cloning benchmarks.

    Random agents die before the shop, so we synthesize a representative state:
    Ante 4 mid-blind with full joker slots, shop populated, consumables held.
    """
    from balatro_sim.jokers.base import JokerInstance
    from balatro_sim.shop import ShopItem

    game = BalatroGame(seed=seed)
    game.reset()
    # Step into SELECTING_HAND so deck/hand are populated
    game.step({"type": "play_blind"})

    # Synthesize a richer state representative of late-game MCTS clones
    game.ante = 4
    game.dollars = 28
    game.jokers = [
        JokerInstance("j_joker"),
        JokerInstance("j_green_joker"),
        JokerInstance("j_space"),
        JokerInstance("j_steel_joker"),
        JokerInstance("j_ride_the_bus"),
    ]
    # Populate joker state dicts (scaling jokers carry counters)
    game.jokers[1].state = {"mult": 14, "sell_value": 3}
    game.jokers[2].state = {"sell_value": 4}
    game.jokers[4].state = {"mult": 8, "sell_value": 3}

    game.consumable_hand = ["c_strength", "c_pl_pluto"]
    game.current_shop = [
        ShopItem(kind="joker", key="j_blueprint", name="Blueprint", price=10),
        ShopItem(kind="joker", key="j_brainstorm", name="Brainstorm", price=10),
        ShopItem(kind="planet", key="c_pl_mars", name="Mars", price=3),
        ShopItem(kind="tarot", key="c_t_chariot", name="The Chariot", price=3),
        ShopItem(kind="booster", key="b_celestial", name="Celestial Pack", price=4),
    ]
    game.played_hand_types_this_round = {"Pair", "Two Pair"}
    game.planet_levels["Flush"] = 4
    game.planet_levels["Straight"] = 3
    game.planet_levels["Full House"] = 2

    return game


def time_deepcopy(game: BalatroGame, n: int = 2000) -> float:
    """Average seconds per deepcopy."""
    t0 = time.perf_counter()
    for _ in range(n):
        _ = copy.deepcopy(game)
    return (time.perf_counter() - t0) / n


def time_clone(game: BalatroGame, n: int = 2000) -> float | None:
    """Average seconds per game.clone(), or None if not implemented."""
    if not hasattr(game, "clone"):
        return None
    t0 = time.perf_counter()
    for _ in range(n):
        _ = game.clone()
    return (time.perf_counter() - t0) / n


def main():
    print("=" * 60)
    print("Balatro sim baseline benchmark")
    print("=" * 60)

    print("\n[1/3] Timing env steps over random games...")
    step_avg, total_steps = time_random_games(n_games=30, max_steps=1500)
    print(f"  total steps:       {total_steps}")
    print(f"  avg step time:     {step_avg*1e6:.1f} us")
    print(f"  steps/sec:         {1/step_avg:,.0f}")

    print("\n[2/3] Building mid-game state...")
    midgame = make_midgame_state(seed=7)
    print(f"  state:             {midgame.state.name}")
    print(f"  ante/blind:        {midgame.ante}/{midgame.current_blind.kind}")
    print(f"  jokers held:       {len(midgame.jokers)}")
    print(f"  deck remaining:    {len(midgame.deck)}")

    print("\n[3/3] Timing state cloning...")
    deepcopy_avg = time_deepcopy(midgame, n=1000)
    print(f"  deepcopy avg:      {deepcopy_avg*1e6:.1f} us  ({1/deepcopy_avg:,.0f} clones/sec)")

    clone_avg = time_clone(midgame, n=2000)
    if clone_avg is None:
        print(f"  game.clone():      not implemented yet")
    else:
        print(f"  game.clone() avg:  {clone_avg*1e6:.1f} us  ({1/clone_avg:,.0f} clones/sec)")
        print(f"  speedup vs deepcopy: {deepcopy_avg/clone_avg:.1f}x")

    print("\n" + "=" * 60)
    print("Interpretation:")
    print("=" * 60)
    ratio = deepcopy_avg / step_avg
    print(f"  deepcopy/step ratio:  {ratio:.1f}x")
    print(f"  At 100 sims/decision, MCTS deepcopy tax = {100*deepcopy_avg*1e3:.1f} ms/decision")
    if clone_avg:
        print(f"  At 100 sims/decision, MCTS clone tax    = {100*clone_avg*1e3:.1f} ms/decision")


if __name__ == "__main__":
    main()
