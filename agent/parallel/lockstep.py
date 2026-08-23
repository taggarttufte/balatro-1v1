"""
lockstep.py — one ``decide_many`` for a whole slice of the population.

``mp/tournament/parallel.py::drive_many`` asks a callable for "an action for each of these
agents, right now".  The default answer (``serial_decide``) asks each player in turn, which
is what ``Tournament.run`` has always done and is why no two MCTS trees ever wanted a leaf
at the same moment (BATCH_NOTES §7.3).  :class:`LockstepDecider` is the answer that batches:
every agent's search descends to a leaf, all the leaves go to the net in one call, every
tree backs up, repeat — ``mcts.BatchedSearch``'s contract exactly, and at ``leaf_batch=1``
each tree's search is bit-identical to running it alone (BATCH_NOTES §3, pinned by
``mp/agent/tests/test_batched.py``).

Why this is not ``BatchedMCTSPlayerGroup``
-----------------------------------------
That class builds its own players and is missing the two things a *training* decision needs:
``MCTSPlayer._constrain`` (the ``--max-skips-per-ante`` candidate filter, which changes the
recorded policy target) and ``record_hook`` (which IS the training sample).  This decider
drives the players the population factory already built — heterogeneous budgets, past-self
checkpoints, scripted anchors and all — and reproduces ``MCTSPlayer.act_key`` line for line,
including the order of ``_constrain`` -> ``cache.store`` -> ``_record``.

Heterogeneous policies
----------------------
A population holds the live net plus up to ``p_history`` past selves, i.e. several distinct
``PolicyValueFn`` objects.  ``BatchedSearch`` batches one policy at a time, so requests are
grouped by policy identity and each group is run in turn.  Grouping loses nothing that
matters: the live net holds every sample-producing seat, so it is the group that fills a
batch, and cross-WORKER batching (the evaluator's job) is where the rest comes from.
Agent-to-worker assignment keeps same-policy seats together for the same reason
(``pool.partition_agents``).
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

from mcts.action import action_from_key, action_key
from mcts.batched import BatchedSearch, SearchRequest
from mcts.player import MCTSPlayer
from mcts.search import MCTS


class LockstepDecider:
    """``decide_many([(idx, game, player), ...]) -> [action, ...]``.

    Stateless between calls apart from the per-policy ``BatchedSearch`` objects it caches
    (they exist only to accumulate ``BatchStats``; ``run_requests`` reads nothing else off
    them).
    """

    def __init__(self, strategy: str = "gumbel"):
        self.strategy = strategy
        self._searches: dict = {}
        self.decisions = 0
        self.searched = 0
        self.shortcut = 0

    # ── stats ────────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        rounds = sum(s.stats.rounds for s in self._searches.values())
        leaves = sum(s.stats.leaves for s in self._searches.values())
        return {"decisions": self.decisions, "searched": self.searched,
                "shortcut": self.shortcut, "search_rounds": rounds,
                "search_leaves": leaves,
                "mean_trees_per_round": (leaves / rounds) if rounds else 0.0}

    def _search_for(self, policy) -> BatchedSearch:
        s = self._searches.get(id(policy))
        if s is None:
            s = BatchedSearch(policy, strategy=self.strategy)
            self._searches[id(policy)] = s
        return s

    # ── the decision ─────────────────────────────────────────────────────────

    def __call__(self, items: Sequence[tuple]) -> list:
        out: list = [None] * len(items)
        pending: list = []                # (slot, game, player, legal, root, sims, request)
        groups: dict = {}                 # id(policy) -> (policy, [slot indices])

        for slot, (_idx, game, player) in enumerate(items):
            self.decisions += 1
            if not isinstance(player, MCTSPlayer):
                out[slot] = player.act(game)
                continue
            legal = game.legal_actions()
            if not legal:
                player.cache.clear()
                out[slot] = player.no_action
                continue
            if player.shortcut_singletons and len(legal) == 1:
                self.shortcut += 1
                player.shortcuts += 1
                key = player._shortcut(game, action_key(legal[0]))
                if player.record_hook is not None and player.record_singletons:
                    player._record(game, legal, [key], {}, key, shortcut=True, sims=0)
                out[slot] = action_from_key(key)
                continue

            self.searched += 1
            player.searches += 1
            root = player.cache.take(game)
            sims = player.cache.budget(root, player.config.num_simulations)
            request = SearchRequest(
                game=game, mcts=player.mcts, strategy=player.strategy, root=root,
                sims=sims, add_noise=player._noise(root),
                leaf_batch=player.config.leaf_batch,
            )
            pending.append((slot, game, player, legal, sims, request))
            entry = groups.setdefault(id(player.policy), (player.policy, []))
            entry[1].append(len(pending) - 1)

        for policy, slots in groups.values():
            search = self._search_for(policy)
            results = search.run_requests([pending[s][5] for s in slots])
            for s, res in zip(slots, results):
                slot, game, player, legal, sims, _req = pending[s]
                out[slot] = self._finish(game, player, legal, sims, res)
        return out

    # ── MCTSPlayer.act_key, from `root, visits, chosen` onwards ──────────────

    @staticmethod
    def _finish(game, player: MCTSPlayer, legal, sims: int, res):
        root, visits, chosen = res.root, res.visit_counts, res.chosen
        if chosen is None and visits:
            chosen = MCTS.sample_action(visits, temperature=player.temperature,
                                        rng=player.rng)
        visits = visits or {}
        if player.legal_filter is not None:
            legal, visits, chosen = player._constrain(game, legal, visits, chosen)
        player.cache.store(game, root, chosen)
        if player.record_hook is not None and chosen is not None:
            player._record(game, legal, [action_key(a) for a in legal], visits, chosen,
                           shortcut=False, sims=sims)
        return player.no_action if chosen is None else action_from_key(chosen)


__all__ = ["LockstepDecider"]
