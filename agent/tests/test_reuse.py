"""
test_reuse.py — keeping the chosen child's subtree between decisions (Phase 3, W3).

What is pinned here:

  1. The assumption the whole scheme rests on: on THIS engine `clone().step(a)` is a
     deterministic function of (state, action) — Phase 1 moved every draw onto the keyed
     `PseudoRandom` whose position table is part of the cloned state. If that ever stops
     being true, `test_engine_is_deterministic_under_clone_step` fails first and loudest.
  2. Reuse == resumption: a search of N simulations run as (M retained + N-M new) gives
     exactly the tree one N-simulation search gives.
  3. A valid subtree is kept, with its visits, and the search resumes into it.
  4. An invalid one is dropped — a driver that applies a different action (a shop
     `reroll`), or that mutates the game between decisions the way MLB's `set_pvp_info`
     does, must not be able to hand the search a tree from another state.
  5. Gumbel on a reused root: fresh sampled noise, sigma scaled by THIS decision's visits
     only, retained Q kept.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from balatro_sim.game import BalatroGame, State
from balatro_sim.jokers.base import JokerInstance
from mcts import (
    MCTS, MCTSConfig, MCTSPlayer, NNPolicy, PolicyValueNet, UniformPolicy,
    ReuseConfig, TreeCache, count_nodes,
)
from mcts.action import action_from_key, action_key
from mcts.node import Node


# ── Fixtures ────────────────────────────────────────────────────────────────

def _mid_blind(seed: int = 42) -> BalatroGame:
    g = BalatroGame(seed=seed)
    g.step({"type": "play_blind"})
    g.dollars = 30
    g.jokers = [
        JokerInstance("j_joker"),
        JokerInstance("j_green_joker"),
        JokerInstance("j_steel_joker"),
    ]
    return g


def _shop(seed: int = 42) -> BalatroGame:
    """A SHOP state with money — `reroll` and several `buy`s are legal."""
    g = BalatroGame(seed=seed)
    g.step({"type": "play_blind"})
    g.debug_win_blind()
    g.step({"type": "advance"})
    assert g.state is State.SHOP
    g.dollars = 40
    return g


def _nemesis(seed: str = "7I4M53DL") -> BalatroGame:
    g = BalatroGame(seed=seed, ruleset="mlb")
    for _ in range(80):
        if g.current_blind.is_pvp and g.state is State.SELECTING_HAND:
            return g
        if g.state is State.BLIND_SELECT:
            g.step({"type": "play_blind"})
        elif g.state is State.SELECTING_HAND:
            g.debug_win_blind()
        elif g.state is State.ROUND_EVAL:
            g.step({"type": "advance"})
        elif g.state is State.SHOP:
            g.step({"type": "leave_shop"})
        elif g.state is State.BOOSTER_OPEN:
            g.step({"type": "skip_booster"})
        else:
            raise AssertionError(f"unexpected state: {g.state}")
    raise AssertionError("never reached a Nemesis")


@pytest.fixture(scope="module")
def policy() -> NNPolicy:
    torch.manual_seed(0)
    return NNPolicy(PolicyValueNet(), device="cpu")


# ── 1. The assumption ───────────────────────────────────────────────────────

@pytest.mark.parametrize("state_fn,kind", [(_mid_blind, "play"), (_shop, "reroll")])
def test_engine_is_deterministic_under_clone_step(state_fn, kind):
    """Stepping the same action from the same state always lands in the same state —
    including a `play` (which draws replacement cards) and a shop `reroll` (which
    regenerates the whole shop). This is why a retained subtree is sound at all, and it
    is why the brief's expected "a random effect will generally NOT match" does not
    happen on the post-Phase-1 engine.
    """
    game = state_fn()
    action = next(a for a in game.legal_actions() if a["type"] == kind)
    signatures = set()
    for _ in range(8):
        clone = game.clone()
        clone.step(dict(action))
        signatures.add(clone.state_signature())
    assert len(signatures) == 1

    # ...and the real game agrees with the clones.
    live = game.clone()
    live.step(dict(action))
    assert live.state_signature() in signatures


# ── 2. Reuse == resumption ──────────────────────────────────────────────────

def test_retained_plus_new_equals_one_search(policy):
    """A budget of N spent as (M retained + N-M new) gives the same tree as N in one go.

    PUCT with no root noise, so nothing in the search consumes the rng and "resume" is
    the only difference between the two runs. Gumbel deliberately does NOT have this
    property: it redraws its root noise every decision (see
    `test_gumbel_redraws_its_noise_on_a_reused_root`).
    """
    cfg = MCTSConfig(num_simulations=100)
    game = _mid_blind(42)

    whole = MCTS(policy, cfg, rng=np.random.default_rng(0))
    _, want = whole.run(game, add_noise=False)

    part = MCTS(policy, cfg, rng=np.random.default_rng(0))
    root, _ = part.run(game, add_noise=False, sims=40)
    root, got = part.run(game, add_noise=False, root=root, sims=60)

    assert got == want
    assert sum(got.values()) == 100


def test_resuming_into_a_child_keeps_its_statistics(policy):
    """The retained root IS the chosen child object: its N, W and its children survive."""
    cfg = MCTSConfig(num_simulations=80)
    game = _mid_blind(42)
    mcts = MCTS(policy, cfg, rng=np.random.default_rng(0))
    root, visits = mcts.run(game, add_noise=False)

    chosen = MCTS.best_action(visits)
    child = root.children[chosen]
    retained_visits = child.visit_count
    retained_nodes = count_nodes(child)
    assert retained_visits > 1

    nxt = game.clone()
    nxt.step(action_from_key(chosen))
    new_root, new_visits = mcts.run(nxt, add_noise=False, root=child, sims=20)

    assert new_root is child
    assert new_root.visit_count == retained_visits + 20
    assert count_nodes(new_root) >= retained_nodes
    assert sum(new_visits.values()) == new_root.visit_count - 1  # root's own expansion visit


# ── 3. The cache: keep when valid ───────────────────────────────────────────

def test_cache_keeps_the_subtree_when_the_signature_matches(policy):
    cache = TreeCache()
    cfg = MCTSConfig(num_simulations=60)
    game = _mid_blind(42)
    mcts = MCTS(policy, cfg, rng=np.random.default_rng(0))
    root, visits = mcts.run(game, add_noise=False)
    chosen = MCTS.best_action(visits)
    cache.store(game, root, chosen)
    assert cache.armed

    game.step(action_from_key(chosen))
    kept = cache.take(game)
    assert kept is root.children[chosen]
    assert cache.stats.hits == 1 and cache.stats.misses == 0


def test_player_reuses_across_a_real_episode(policy):
    """The end-to-end shape: act -> step -> act. In vanilla single player every
    decision's tree survives (the engine is deterministic and nothing else touches the
    game), so the only non-hit is the very first decision."""
    player = MCTSPlayer(policy=policy, config=MCTSConfig(num_simulations=24),
                        strategy="gumbel", reuse=True, rng=np.random.default_rng(0))
    game = BalatroGame(seed=42)
    for _ in range(20):
        if game.state is State.GAME_OVER:
            break
        action = player.act(game)
        if action is None:
            break
        game.step(action)

    stats = player.reuse_stats
    assert stats.decisions >= 8
    assert stats.misses == 0
    assert stats.empty == 1                      # only the first decision
    assert stats.hits == stats.decisions - 1
    assert stats.retained_visits > 0
    assert 0.0 < stats.node_fraction <= 1.0
    assert stats.new_sims < stats.budget_sims    # reuse bought simulations back


def test_forced_singleton_states_carry_the_tree(policy):
    """`ROUND_EVAL` -> `advance` sits between every blind and every shop. The shortcut
    skips the search but must still walk the retained tree down, or the tree dies at
    every cash-out."""
    player = MCTSPlayer(policy=policy, config=MCTSConfig(num_simulations=60),
                        strategy="puct", reuse=True, rng=np.random.default_rng(0))
    game = _mid_blind(42)
    game.chips_scored = game.current_blind.chips_target - 1   # any hand clears the blind

    action = player.act(game)                      # a real search; arms the cache
    game.step(action)
    assert game.state is State.ROUND_EVAL and len(game.legal_actions()) == 1
    assert player.cache.armed

    forced = player.act(game)                      # the shortcut: no search
    assert forced == {"type": "advance"}
    assert player.shortcuts == 1
    assert player.reuse_stats.hits == 1            # the tree was walked down, not dropped
    assert player.cache.armed                      # ...and re-armed for the shop
    game.step(forced)
    assert player.cache.take(game) is not None
    assert player.reuse_stats.misses == 0


# ── 4. The cache: discard when invalid ──────────────────────────────────────

def test_cache_discards_when_the_driver_applies_a_different_action(policy):
    """The search picked something; the driver rerolled the shop instead. Different
    state, different signature, no reuse."""
    cache = TreeCache()
    game = _shop(42)
    mcts = MCTS(policy, MCTSConfig(num_simulations=40), rng=np.random.default_rng(0))
    root, visits = mcts.run(game, add_noise=False)
    chosen = MCTS.best_action(visits)
    cache.store(game, root, chosen)

    reroll = next(a for a in game.legal_actions() if a["type"] == "reroll")
    assert action_key(reroll) != chosen
    game.step(reroll)

    assert cache.take(game) is None
    assert cache.stats.misses == 1 and cache.stats.hits == 0
    assert not cache.armed


def test_cache_discards_when_the_driver_mutates_the_game(policy):
    """MLB: the opponent's live score arrives via `set_pvp_info` between decisions. The
    game the search planned in no longer exists, so the tree goes."""
    cache = TreeCache()
    game = _nemesis()
    game.set_pvp_info(10_000, 3)
    mcts = MCTS(policy, MCTSConfig(num_simulations=24), rng=np.random.default_rng(0))
    root, visits = mcts.run(game, add_noise=False)
    chosen = MCTS.best_action(visits)
    cache.store(game, root, chosen)

    game.step(action_from_key(chosen))
    game.set_pvp_info(40_000, 1)                   # the relay the search never saw
    assert cache.take(game) is None
    assert cache.stats.misses == 1


def test_cache_declines_to_store_a_terminal_or_unvisited_child(policy):
    cache = TreeCache()
    game = _mid_blind(42)
    mcts = MCTS(policy, MCTSConfig(num_simulations=20), rng=np.random.default_rng(0))
    root, visits = mcts.run(game, add_noise=False)

    unvisited = next(k for k, c in root.children.items() if not c.is_expanded)
    cache.store(game, root, unvisited)
    assert not cache.armed

    cache.store(game, root, None)
    assert not cache.armed

    terminal = Node(is_expanded=True, is_terminal=True)
    fake = Node()
    fake.children[("advance",)] = terminal
    cache.store(game, fake, ("advance",))
    assert not cache.armed


def test_disabled_cache_never_reuses(policy):
    cache = TreeCache(ReuseConfig(enabled=False))
    game = _mid_blind(42)
    mcts = MCTS(policy, MCTSConfig(num_simulations=20), rng=np.random.default_rng(0))
    root, visits = mcts.run(game, add_noise=False)
    cache.store(game, root, MCTS.best_action(visits))
    assert not cache.armed
    assert cache.take(game) is None


def test_reset_drops_the_tree(policy):
    player = MCTSPlayer(policy=policy, config=MCTSConfig(num_simulations=12),
                        strategy="puct", reuse=True, rng=np.random.default_rng(0))
    game = _mid_blind(42)
    key = player.act_key(game)
    assert player.cache.armed
    player.reset()
    assert not player.cache.armed
    game.step(action_from_key(key))
    player.act_key(game)
    assert player.reuse_stats.hits == 0


# ── 5. Budgeting ────────────────────────────────────────────────────────────

def test_budget_modes():
    root = Node(visit_count=30)
    subtract = TreeCache(ReuseConfig(budget_mode="subtract"))
    add = TreeCache(ReuseConfig(budget_mode="add"))
    assert subtract.budget(root, 100) == 70
    assert add.budget(root, 100) == 100
    assert subtract.budget(None, 100) == 100
    floored = TreeCache(ReuseConfig(budget_mode="subtract", min_new_sims=25))
    assert floored.budget(Node(visit_count=95), 100) == 25


def test_effective_sims_counts_retained_work(policy):
    player = MCTSPlayer(policy=policy, config=MCTSConfig(num_simulations=32),
                        strategy="puct", reuse=True, rng=np.random.default_rng(0))
    game = BalatroGame(seed=42)
    for _ in range(12):
        action = player.act(game)
        if action is None or game.state is State.GAME_OVER:
            break
        game.step(action)
    stats = player.reuse_stats
    # Under "subtract" a decision never exceeds its budget in total work...
    assert stats.effective_sims <= 32 + 1e-9
    # ...and the tree really did carry evidence forward.
    assert stats.retained_visits > 0


# ── 6. Gumbel on a reused root ──────────────────────────────────────────────

def test_gumbel_sigma_uses_only_this_decisions_visits(policy):
    """sigma(q) = (c_visit + N_max) * q. With a retained root N_max must be the visits
    added THIS decision, otherwise a tree carrying 400 visits multiplies every q by ~450
    and the sampled Gumbel noise stops mattering."""
    cfg = MCTSConfig(num_simulations=8)
    mcts = MCTS(policy, cfg, rng=np.random.default_rng(0))
    root = Node()
    for i, key in enumerate([("a",), ("b",)]):
        child = root.add_child(key, prior=0.5)
        child.visit_count = 100 + i
        child.value_sum = 50.0
    keys = list(root.children)
    noise = np.zeros(2)
    logits = np.log(np.array([0.5, 0.5]))

    without = mcts._gumbel_scores(root, keys, [0, 1], noise, logits)
    baseline = {k: root.children[k].visit_count for k in keys}
    with_baseline = mcts._gumbel_scores(root, keys, [0, 1], noise, logits, baseline)

    assert without[0] > with_baseline[0]        # inflated by the retained visits
    # N_max = 0 this decision -> sigma = (c_visit + 0) * c_scale * q, q = 50/100 = 0.5
    assert with_baseline[0] == pytest.approx(cfg.gumbel_c_visit * 0.5 + logits[0], abs=1e-9)
    # ...against N_max = 101 without the baseline.
    assert without[0] == pytest.approx((cfg.gumbel_c_visit + 101) * 0.5 + logits[0], abs=1e-9)


def test_gumbel_redraws_its_noise_on_a_reused_root(policy):
    """Documented behaviour, not a bug: every decision samples a fresh Gumbel top-k, so a
    reused root does not lock the previous decision's candidate set in. The retained
    visits and Q estimates are still used."""
    cfg = MCTSConfig(num_simulations=40, gumbel_max_considered=8)
    game = _mid_blind(42)
    mcts = MCTS(policy, cfg, rng=np.random.default_rng(0))
    root, visits, chosen = mcts.run_gumbel(game)
    child = root.children[chosen]
    before = child.visit_count

    nxt = game.clone()
    nxt.step(action_from_key(chosen))
    new_root, new_visits, new_chosen = mcts.run_gumbel(nxt, root=child, sims=20)

    assert new_root is child
    assert new_root.visit_count == before + 20
    assert new_chosen in new_visits
    assert sum(new_visits.values()) >= 20


def test_reused_root_is_re_noised_but_not_compounded(policy):
    """Noise goes into the previous root's OWN children only, so the grandchildren that
    become the next root's children carry bare priors — re-noising them is the same
    one-shot mix a fresh search does."""
    cfg = MCTSConfig(num_simulations=40, dirichlet_eps=0.9, dirichlet_alpha=0.5)
    game = _mid_blind(42)
    mcts = MCTS(policy, cfg, rng=np.random.default_rng(0))
    root, visits = mcts.run(game, add_noise=True)
    chosen = MCTS.best_action(visits)
    child = root.children[chosen]
    grandchild_priors = np.array([c.prior for c in child.children.values()])
    assert abs(grandchild_priors.sum() - 1.0) < 1e-4     # untouched by the root noise

    nxt = game.clone()
    nxt.step(action_from_key(chosen))
    mcts.run(nxt, add_noise=True, root=child, sims=10)
    after = np.array([c.prior for c in child.children.values()])
    assert not np.allclose(grandchild_priors, after)      # noise WAS applied, once
    assert abs(after.sum() - 1.0) < 1e-4
