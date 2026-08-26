"""
reuse.py — keep the chosen child's subtree as the next decision's root (Phase 3, W3).

The rule
--------
After a decision, the subtree under the chosen action is exactly the search's beliefs
about the state the game is *about to be in*. Keeping it means the next search starts
with N simulations of evidence instead of zero. It is only sound if the state the subtree
was built from IS the state the game actually reached, and the engine gives us the exact
test for that: `BalatroGame.state_signature()` (`engine/balatro_sim/game.py:894` —
"two games with equal signatures produce identical futures"; it covers the keyed RNG's
full position table).

So: on the next call, hash the live game and compare. Equal -> reuse. Different -> drop
the tree and search fresh. There is no partial credit and no attempt to salvage.

What actually invalidates a tree (measured, not assumed)
-------------------------------------------------------
The brief expected chance-node invalidation: "a `play` that triggers a random effect or a
shop action that draws will generally NOT match". **On this engine it always matches.**
Phase 1 deleted `game.rng` and moved every draw onto the keyed `PseudoRandom`, whose
position table is part of the cloned state, so `clone().step(a)` is a deterministic
function of (state, action): stepping the same action from the same state a hundred times
gives one signature, every time (verified in `tests/test_reuse.py::
test_engine_is_deterministic_under_clone_step`, which is the assumption this whole module
rests on — if it ever fails, reuse must be reconsidered, not just re-tuned).

What DOES invalidate a tree is everything the *driver* does between decisions, which the
search never saw:

  * MLB match plumbing — `set_pvp_info()` (the opponent's live score arrives),
    `end_pvp()`, `lose_life()`, the comeback bonus. The tournament runner and `MLBMatch`
    all mutate the game between `act()` calls.
  * more than one engine step between decisions (the runner's `_cash_out`, a `PVP_WAIT`
    resolution, a booster auto-close).
  * a driver that applies an action other than the one the player returned.
  * a search whose chosen edge was never simulated (no subtree to keep) or that ended on
    a terminal / stuck child.

That makes the signature check load-bearing in MP and near-free in vanilla SP, which is
what the retention numbers in BATCH_NOTES.md §4 show.

Cost
----
One `clone()` + one `step()` + one `state_signature()` when storing (~150 us), one
`state_signature()` when taking (~42 us) — about 0.2 ms per decision against a 500-sim
search costing ~1 s. Recording the signature inside the search instead (on every root-edge
traversal) would cost 42 us x sims, i.e. 10-20% of a batched search, for no extra
correctness — the determinism above makes the single post-hoc step exactly equivalent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from balatro_sim.game import BalatroGame
from .action import ActionKey, action_from_key
from .node import Node


@dataclass
class ReuseConfig:
    """How a retained subtree is used.

    budget_mode:
        "subtract" (default) — a decision costs `num_simulations` in TOTAL, retained
            visits included, so reuse buys wall clock at a fixed evidence level. This is
            the mode the equivalence test pins ("budget counted as retained + new").
        "add" — always run the full `num_simulations` of new work, so reuse buys evidence
            at a fixed wall clock. `effective_sims` in the benchmark is this view.
    min_new_sims:
        floor under "subtract" so a deep retained tree cannot reduce a decision to zero
        new simulations (Gumbel in particular needs a few sims to rank its candidates).
    renoise:
        re-apply root Dirichlet noise to the reused root. Default True: the reused root's
        children are the grandchildren of the previous root and carry the *bare* policy
        priors installed at their parent's expansion — noise was only ever mixed into the
        previous root's own children — so re-noising cannot compound, and skipping it
        would leave every decision after the first with no root exploration at all.
        (PUCT only; Gumbel does its exploration by sampling.)
    """
    enabled: bool = True
    budget_mode: str = "subtract"
    min_new_sims: int = 0
    renoise: bool = True


@dataclass
class ReuseStats:
    decisions: int = 0          # calls that consulted the cache
    hits: int = 0               # signature matched, subtree kept
    misses: int = 0             # a subtree was offered but the signature differed
    empty: int = 0              # nothing was cached (first decision, or cleared)
    retained_visits: int = 0    # sum over hits of the retained root's visit count
    retained_nodes: int = 0     # sum over hits of the retained subtree's node count
    stored_nodes: int = 0       # sum over stores of the whole tree's node count
    new_sims: int = 0           # simulations actually run
    budget_sims: int = 0        # simulations that would have been run without reuse
    searches: int = 0           # decisions that actually ran a search (not shortcuts)
    search_retained: int = 0    # retained visits carried into those searches

    @property
    def hit_rate(self) -> float:
        n = self.hits + self.misses + self.empty
        return self.hits / n if n else 0.0

    @property
    def node_fraction(self) -> float:
        """Average fraction of the previous decision's tree that survived into the next."""
        return self.retained_nodes / self.stored_nodes if self.stored_nodes else 0.0

    @property
    def effective_sims(self) -> float:
        """Simulations informing a SEARCHED decision, retained ones included. Shortcut
        decisions (a single legal action) are excluded — they run no simulations, so
        counting them would dilute the number rather than describe it."""
        return ((self.new_sims + self.search_retained) / self.searches
                if self.searches else 0.0)

    def as_dict(self) -> dict:
        return {
            "decisions": self.decisions, "hits": self.hits, "misses": self.misses,
            "empty": self.empty, "hit_rate": self.hit_rate,
            "retained_visits": self.retained_visits,
            "node_fraction": self.node_fraction,
            "new_sims": self.new_sims, "budget_sims": self.budget_sims,
            "searches": self.searches, "search_retained": self.search_retained,
            "effective_sims": self.effective_sims,
        }


def count_nodes(node: Optional[Node]) -> int:
    """Nodes in a subtree, root included. Iterative — a Balatro tree can be deep."""
    if node is None:
        return 0
    total = 0
    stack = [node]
    while stack:
        n = stack.pop()
        total += 1
        stack.extend(n.children.values())
    return total


class TreeCache:
    """One agent's retained subtree between two consecutive decisions.

    Usage (this is exactly what `MCTSPlayer` does):

        root = cache.take(game)                     # None -> search fresh
        root, visits, chosen = mcts.run_gumbel(game, root=root, sims=budget)
        cache.store(game, root, chosen)             # arms the next decision

    `store` takes the game as it was BEFORE the action (the state the search ran from);
    it clones it, applies the chosen action and records the resulting signature, which is
    the state the kept subtree describes.
    """

    def __init__(self, config: ReuseConfig | None = None,
                 stats: ReuseStats | None = None):
        self.cfg = config or ReuseConfig()
        self.stats = stats or ReuseStats()
        self._root: Optional[Node] = None
        self._signature = None

    # ── take / store ────────────────────────────────────────────────────────

    def take(self, game: BalatroGame) -> Optional[Node]:
        """The retained root iff it belongs to `game`'s exact state, else None."""
        self.stats.decisions += 1
        if not self.cfg.enabled or self._root is None:
            self.stats.empty += 1
            return None
        if game.state_signature() != self._signature:
            self.stats.misses += 1
            self.clear()
            return None
        root = self._root
        self.stats.hits += 1
        self.stats.retained_visits += root.visit_count
        self.stats.retained_nodes += count_nodes(root)
        self._root = None            # consumed; `store` re-arms it
        self._signature = None
        return root

    def store(self, root_game: BalatroGame, root: Optional[Node],
              chosen: Optional[ActionKey]) -> None:
        """Keep `root.children[chosen]` for the next decision, with the signature of the
        state that applying `chosen` to `root_game` produces."""
        self.clear()
        if not self.cfg.enabled or root is None or chosen is None:
            return
        child = root.children.get(chosen)
        if child is None or not child.is_expanded or child.is_terminal:
            # Nothing worth keeping: an unvisited edge, or one that ends the run.
            return
        self.stats.stored_nodes += count_nodes(root)
        nxt = root_game.clone()
        nxt.step(action_from_key(chosen))
        self._root = child
        self._signature = nxt.state_signature()

    def clear(self) -> None:
        self._root = None
        self._signature = None

    # ── Budgeting ───────────────────────────────────────────────────────────

    def budget(self, root: Optional[Node], num_simulations: int) -> int:
        """How many NEW simulations this decision should run."""
        retained = 0 if root is None else root.visit_count
        self.stats.searches += 1
        self.stats.search_retained += retained
        if root is None or self.cfg.budget_mode == "add":
            sims = num_simulations
        else:
            sims = max(self.cfg.min_new_sims, num_simulations - retained)
        self.stats.new_sims += sims
        self.stats.budget_sims += num_simulations
        return sims

    # ── Introspection ───────────────────────────────────────────────────────

    @property
    def armed(self) -> bool:
        return self._root is not None

    @property
    def signature(self):
        return self._signature


__all__ = ["ReuseConfig", "ReuseStats", "TreeCache", "count_nodes"]
