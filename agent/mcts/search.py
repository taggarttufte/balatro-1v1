"""
search.py — AlphaZero-style MCTS over the fork Balatro engine.

One simulation:
  1. SELECT — descend from root using PUCT until reaching an unexpanded node
     or a stopping state. Each step clones, applies the chosen action.
  2. EXPAND — at the leaf, query the policy/value fn. Create child nodes
     for each legal action with their priors.
  3. BACKUP — propagate the value up the path, incrementing N and adding to W.

Root exploration: optional Dirichlet noise mixed into root priors after the initial
expansion (standard AlphaZero). Without it, PUCT collapses on the first action that
backs up positive Q. Controlled by MCTSConfig.dirichlet_alpha / dirichlet_eps.

Stochasticity: the sim is stochastic (deck draws, shop gen). We don't use explicit
chance nodes — each simulation clones and steps, so different visits see different RNG
outcomes. This is "sample-based MCTS via determinization": higher variance than proper
chance nodes but doesn't require a sim refactor. (W3: this is also exactly why tree
reuse must check `state_signature()` before keeping a subtree.)

Note on action selection at root: the visit-count distribution IS the improved policy.
For training, sample with temperature; for eval, take argmax.

Fork changes vs balatro-mcts @ ee75d11
--------------------------------------
1. **The outcome signal is a parameter** (`mcts/outcome.py`). `_is_win` used to be
   `state == GAME_OVER and ante > 8`, which is meaningless under MLB (endless, win is
   `match_won`, and the real signal at a Nemesis is an external score margin). Pass
   `outcome=`; when omitted it is chosen per root game by `default_outcome_for`.
2. **States with no legal actions are handled.** MLB has two of them — `PVP_WAIT` and
   "readied at the Nemesis, waiting for `startBlind`" — where the player cannot act and
   only the match can move the game on. The original would have called
   `_select_child` on a childless node and crashed on `action_from_key(None)`. Such a
   node now stops descent with `stop_reason="stuck"` and is valued by the outcome's
   pending estimate; `run`/`run_gumbel` on such a ROOT return empty visit counts and
   `chosen=None` instead of raising.
3. `_evaluate_leaf` is factored out as the single leaf-evaluation seam for W3.

W3 changes (2026-08-22) — batched leaf evaluation
-------------------------------------------------
4. **Every search is also available as a generator** (`run_iter` / `run_gumbel_iter` /
   `_simulate_iter` / `_expand_iter`). A generator *yields a list of leaf games* and is
   `send()`-ed back the list of `(priors, value)` evaluations. `run` / `run_gumbel` are
   now thin wrappers that drive their generator with `self._evaluate_leaf`, one leaf at a
   time, in exactly the original order — so single-tree behaviour is unchanged (the 85
   W1 tests pin it). `mcts/batched.py` drives K such generators in lockstep and evaluates
   all their leaves in ONE forward pass.
5. **A search may start from an EXISTING root** (`root=` on `run` / `run_gumbel`) and with
   an explicit simulation budget (`sims=`). That is tree reuse (`mcts/reuse.py`): the
   retained subtree keeps its N / W / P and the search resumes into it.
6. **Optional within-tree leaf batching** (`MCTSConfig.leaf_batch > 1`) runs L simulations
   in flight with **virtual loss** (a descending simulation increments N immediately and
   contributes 0.0 to W until its real value arrives). L = 1 (the default) applies no
   virtual loss and is bit-identical to the original serial search.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from balatro_sim.game import BalatroGame, State
from .action import ActionKey, action_from_key
from .node import Node
from .outcome import OutcomeFn, default_outcome_for
from .policy import PolicyValueFn


@dataclass
class MCTSConfig:
    num_simulations: int = 100
    c_puct: float = 1.5            # exploration constant in PUCT
    discount: float = 1.0          # full Balatro game; no need to discount
    # Root Dirichlet noise (standard AlphaZero). Set either to 0 to disable.
    # AlphaZero rule of thumb: alpha ~= 10 / avg_legal_actions.
    # SELECTING_HAND in Balatro has ~200-450 actions, so alpha ~= 0.03.
    # Larger alpha (e.g. chess's 0.3 default) is too uniform here — noise
    # spreads thin over hundreds of children and fails to break PUCT lock-in.
    dirichlet_alpha: float = 0.03
    dirichlet_eps: float = 0.25
    # Gumbel root selection (Danihelka et al. 2022, ICLR).
    # Used by run_gumbel(). PUCT-based run() ignores these.
    gumbel_max_considered: int = 16   # m_init = min(num_legals, this)
    gumbel_c_visit: float = 50.0      # paper default
    gumbel_c_scale: float = 1.0       # assumes q in [0, 1]
    # W3: within-tree leaf batching. L simulations descend before any of them is
    # evaluated, held apart by virtual loss, then all L leaves go to the net in one
    # call. L = 1 disables virtual loss entirely and is the original serial search.
    # Only useful when a single tree is the whole workload (the tournament calls
    # Player.act one agent at a time); cross-tree batching is strictly better when
    # several trees are available. See BATCH_NOTES.md §3.
    leaf_batch: int = 1


class MCTS:
    def __init__(
        self,
        policy_value_fn: PolicyValueFn,
        config: MCTSConfig | None = None,
        rng: np.random.Generator | None = None,
        outcome: OutcomeFn | None = None,
    ):
        self.policy_value_fn = policy_value_fn
        self.cfg = config or MCTSConfig()
        self.rng = rng if rng is not None else np.random.default_rng()
        self.outcome = outcome        # None -> resolved per root game

    # ── Outcome resolution ───────────────────────────────────────────────────

    def _outcome_for(self, game: BalatroGame) -> OutcomeFn:
        if self.outcome is None:
            self.outcome = default_outcome_for(game)
        return self.outcome

    # ── Leaf evaluation (THE seam W3 replaces / batches) ─────────────────────

    def _evaluate_leaf(self, game: BalatroGame):
        """Every policy/value query in the search funnels through here.

        Returns (priors: {ActionKey: float}, value: float). W3's batched implementation
        collects several leaves and calls `policy_value_fn.evaluate_many([...])`; this
        method is the only place a single-leaf call is made, so a leaf-parallel variant
        replaces exactly one call site.

        W3: this stays the SERIAL seam (bench_search.py's instrumentation patches it).
        The batched path never comes through here — it drives `run_iter` /
        `run_gumbel_iter` and calls `policy_value_fn.evaluate_many` on the whole batch.
        """
        return self.policy_value_fn(game)

    def _drive(self, gen):
        """Run a search generator to completion, satisfying each leaf request serially.

        The generator yields `list[BalatroGame]` and is sent `list[(priors, value)]`.
        This driver answers one leaf at a time through `_evaluate_leaf`, which is what
        makes `run` / `run_gumbel` identical to the pre-W3 implementations.
        """
        try:
            request = next(gen)
            while True:
                request = gen.send([self._evaluate_leaf(g) for g in request])
        except StopIteration as stop:
            return stop.value

    # ── Public API ───────────────────────────────────────────────────────────

    def run(
        self,
        root_game: BalatroGame,
        add_noise: bool = True,
        root: Node | None = None,
        sims: int | None = None,
    ) -> tuple[Node, dict[ActionKey, int]]:
        """
        Run num_simulations from a clone of root_game. Returns the root Node
        and a dict {action_key: visit_count} as the search-improved policy.

        add_noise: when True (default), mix Dirichlet noise into the root
        priors per MCTSConfig. Pass False at eval/inference time to use the
        bare policy.

        root: an EXISTING root node to resume into (tree reuse, `mcts/reuse.py`). It
        must already be expanded and must correspond to `root_game`'s state — the
        caller proves that with `state_signature()`. Its N / W / P are kept.
        sims: override the simulation budget (reuse counts retained visits against it).

        A root with no legal actions (terminal, or MLB PVP_WAIT / readied) returns
        ({}, {}) — an empty visit-count dict — rather than raising.
        """
        return self._drive(self.run_iter(root_game, add_noise=add_noise, root=root,
                                         sims=sims))

    def run_iter(
        self,
        root_game: BalatroGame,
        add_noise: bool = True,
        root: Node | None = None,
        sims: int | None = None,
        leaf_batch: int | None = None,
    ):
        """Generator form of `run` — see the module docstring's W3 note.

        Yields `list[BalatroGame]` (the leaves to evaluate), is sent the matching
        `list[(priors, value)]`, and returns `(root, visit_counts)`.
        """
        self._outcome_for(root_game)
        leaf_batch = self.cfg.leaf_batch if leaf_batch is None else leaf_batch
        if root is None:
            root = Node()
            # Expand the root once so it has children to descend into
            yield from self._expand_iter(root, root_game)

        if not root.children:
            return root, {}

        if add_noise:
            self._add_dirichlet_noise(root)

        n_sims = self.cfg.num_simulations if sims is None else max(0, sims)
        yield from self._run_sims_iter(root, root_game, [None] * n_sims, leaf_batch)

        visit_counts = {k: c.visit_count for k, c in root.children.items()}
        return root, visit_counts

    # ── Root exploration ─────────────────────────────────────────────────────

    def _add_dirichlet_noise(self, root: Node):
        """
        Mix Dir(alpha) noise into root children's priors:
            P(a) <- (1 - eps) * P(a) + eps * eta(a),  eta ~ Dir(alpha)

        No-op if either alpha or eps is 0, or if the root has no children.
        """
        alpha = self.cfg.dirichlet_alpha
        eps = self.cfg.dirichlet_eps
        if alpha <= 0 or eps <= 0 or not root.children:
            return
        n = len(root.children)
        noise = self.rng.dirichlet([alpha] * n)
        for child, eta in zip(root.children.values(), noise):
            child.prior = (1.0 - eps) * child.prior + eps * float(eta)

    # ── Gumbel root action selection ─────────────────────────────────────────

    def run_gumbel(
        self,
        root_game: BalatroGame,
        root: Node | None = None,
        sims: int | None = None,
    ) -> tuple[Node, dict[ActionKey, int], Optional[ActionKey]]:
        """
        Gumbel-Top-k + Sequential Halving root selection
        (Danihelka et al. 2022, "Policy Improvement by Planning with Gumbel").

        Returns (root, visit_counts, chosen_action). Unlike run(), the chosen
        action is NOT argmax of visits — it is the survivor of sequential
        halving, ranked by (Gumbel noise + logits + sigma(empirical Q)).

        The interior of the tree still uses standard PUCT — only root action
        selection changes. Dirichlet noise is not used (Gumbel sampling
        provides exploration).

        On a root with no legal actions (terminal, or MLB PVP_WAIT / readied at the
        Nemesis) returns (root, {}, None). Callers must handle `chosen is None`;
        `MCTSPlayer.act` and `SelfPlayAgent.play_episode` do.

        `root=` / `sims=`: tree reuse — see `run`. On a REUSED root the sampled Gumbel
        top-k is drawn fresh from this decision's own noise, the prior visit counts are
        excluded from sigma's N_max scale (only sims spent THIS decision count), and the
        retained Q estimates are used as-is. See BATCH_NOTES.md §4.3.
        """
        return self._drive(self.run_gumbel_iter(root_game, root=root, sims=sims))

    def run_gumbel_iter(
        self,
        root_game: BalatroGame,
        root: Node | None = None,
        sims: int | None = None,
        leaf_batch: int | None = None,
    ):
        """Generator form of `run_gumbel`; returns `(root, visit_counts, chosen)`."""
        self._outcome_for(root_game)
        leaf_batch = self.cfg.leaf_batch if leaf_batch is None else leaf_batch
        if root is None:
            root = Node()
            yield from self._expand_iter(root, root_game)
        if root.is_terminal or not root.children:
            return root, {}, None

        # Visits already in the tree when this decision started (all zero for a fresh
        # root). Sequential halving's sigma scale must see only THIS decision's visits,
        # otherwise a reused root with N_max=400 multiplies q_hat by ~450 and the
        # sampled Gumbel noise stops mattering at all.
        baseline = {k: c.visit_count for k, c in root.children.items()}

        legal_keys = list(root.children.keys())
        n_legal = len(legal_keys)
        m_init = min(n_legal, self.cfg.gumbel_max_considered)

        # Logits from priors. softmax(log(p)) == p, so log(prior) is a valid
        # set of logits up to an irrelevant additive constant.
        priors = np.array([
            max(root.children[k].prior, 1e-12) for k in legal_keys
        ])
        logits = np.log(priors)

        # Standard Gumbel(0,1): -log(-log(U)), U ~ Uniform(0,1)
        u = self.rng.random(n_legal)
        u = np.maximum(u, 1e-12)
        gumbel_noise = -np.log(-np.log(u))

        # Initial top-m by g + logits (== sample m without replacement from softmax)
        init_scores = gumbel_noise + logits
        surv = np.argsort(-init_scores)[:m_init].tolist()

        budget = self.cfg.num_simulations if sims is None else max(0, sims)
        sims_remaining = budget
        n_phases = max(1, math.ceil(math.log2(max(m_init, 2))))
        sims_per_phase = max(1, budget // n_phases)

        while len(surv) > 1 and sims_remaining > 0:
            per_action = max(1, sims_per_phase // len(surv))
            # The phase's plan, in exactly the order the serial loop issued it.
            plan: list[ActionKey] = []
            for ai in surv:
                if len(plan) >= sims_remaining:
                    break
                for _ in range(per_action):
                    if len(plan) >= sims_remaining:
                        break
                    plan.append(legal_keys[ai])
            yield from self._run_sims_iter(root, root_game, plan, leaf_batch)
            sims_remaining -= len(plan)

            # Rank survivors by g + logits + sigma(q_hat); keep top half
            scores = self._gumbel_scores(root, legal_keys, surv, gumbel_noise, logits,
                                         baseline)
            m_next = max(1, len(surv) // 2)
            order = np.argsort(-scores)[:m_next]
            surv = [surv[i] for i in order]

        # Spend any remaining sims on the survivor(s)
        if sims_remaining > 0 and surv:
            plan = [legal_keys[surv[i % len(surv)]] for i in range(sims_remaining)]
            yield from self._run_sims_iter(root, root_game, plan, leaf_batch)
            sims_remaining = 0

        # Final pick (only matters if loop exited with multiple survivors)
        if len(surv) > 1:
            scores = self._gumbel_scores(root, legal_keys, surv, gumbel_noise, logits,
                                         baseline)
            chosen_idx = surv[int(np.argmax(scores))]
        else:
            chosen_idx = surv[0]

        chosen_action = legal_keys[chosen_idx]
        visit_counts = {k: c.visit_count for k, c in root.children.items()}
        return root, visit_counts, chosen_action

    def _gumbel_scores(
        self,
        root: Node,
        legal_keys: list[ActionKey],
        surv_idx: list[int],
        gumbel_noise: np.ndarray,
        logits: np.ndarray,
        baseline: dict[ActionKey, int] | None = None,
    ) -> np.ndarray:
        """
        Score survivors as g_a + l_a + sigma(q_hat_a).

        sigma(q) = (c_visit + N_max) * c_scale * q, per Danihelka et al.
        Q values from value head are in [0, 1] (loss=0, win=1) so c_scale=1.
        Unvisited surviving actions have q_hat = 0; they rely on g + l only.

        `baseline` (tree reuse): visits each child already had before this decision.
        N_max is taken over the visits added THIS decision, so a retained subtree does
        not silently inflate sigma; q_hat still uses all visits, retained ones included.
        """
        if baseline:
            N_max = max(c.visit_count - baseline.get(k, 0)
                        for k, c in root.children.items())
        else:
            N_max = max(c.visit_count for c in root.children.values())
        c_visit = self.cfg.gumbel_c_visit
        c_scale = self.cfg.gumbel_c_scale
        out = np.empty(len(surv_idx), dtype=np.float64)
        for i, ai in enumerate(surv_idx):
            child = root.children[legal_keys[ai]]
            q_hat = child.mean_value
            sigma = (c_visit + N_max) * c_scale * q_hat
            out[i] = float(gumbel_noise[ai]) + float(logits[ai]) + sigma
        return out

    # ── One simulation ───────────────────────────────────────────────────────

    def _simulate(
        self,
        root: Node,
        game: BalatroGame,
        force_first: ActionKey | None = None,
    ):
        """Serial one-simulation entry point (see `_simulate_iter`)."""
        return self._drive(self._simulate_iter(root, game, force_first))

    def _run_sims_iter(
        self,
        root: Node,
        root_game: BalatroGame,
        plan: list,
        leaf_batch: int = 1,
    ):
        """Issue the simulations in `plan` (one entry per sim: its `force_first`, or
        None for plain PUCT descent), keeping up to `leaf_batch` of them in flight.

        With leaf_batch == 1 this is exactly the original loop: clone, descend, evaluate,
        back up, repeat — same order, same RNG consumption, same tree.

        With leaf_batch > 1, L simulations descend before any of them is evaluated. They
        are kept apart by VIRTUAL LOSS (see `_simulate_iter`), and their L leaves are
        yielded as one request so the batched driver can evaluate them in a single
        forward pass. This changes the search (the later descents in a batch see stale
        statistics) — it is an approximation, and it is only worth it when there is no
        second tree to batch against.
        """
        it = iter(plan)
        leaf_batch = max(1, int(leaf_batch))     # 0 would silently run no simulations
        virtual_loss = leaf_batch > 1
        pending: list = []      # [(generator, request)]
        while True:
            while len(pending) < leaf_batch:
                try:
                    force_first = next(it)
                except StopIteration:
                    break
                gen = self._simulate_iter(root, root_game.clone(), force_first,
                                          virtual_loss=virtual_loss)
                try:
                    request = next(gen)
                except StopIteration:
                    continue     # a leaf that needed no evaluation (terminal / stuck)
                pending.append((gen, request))
            if not pending:
                return
            replies = yield [g for _, request in pending for g in request]
            resumed = []
            at = 0
            for gen, request in pending:
                reply = replies[at:at + len(request)]
                at += len(request)
                try:
                    resumed.append((gen, gen.send(reply)))
                except StopIteration:
                    pass
            pending = resumed

    def _simulate_iter(
        self,
        root: Node,
        game: BalatroGame,
        force_first: ActionKey | None = None,
        virtual_loss: bool = False,
    ):
        """
        Walk down the tree, expand a leaf, back up the value.

        force_first: when set, the first step from root uses this action
        instead of PUCT. After the first step, descent continues with PUCT.
        Used by Gumbel root selection to drive sims toward chosen candidates.

        virtual_loss: increment N on every node of the path AS IT IS DESCENDED, and
        contribute nothing to W until the real value arrives. A concurrently descending
        simulation therefore sees this path as visited-with-value-0 and is pushed
        elsewhere. Final N and W are identical to the serial case — the visit is just
        counted early instead of at backup.
        """
        outcome = self._outcome_for(game)
        path = [root]
        node = root
        first_step = True
        if virtual_loss:
            root.visit_count += 1

        # SELECT — descend until unexpanded, terminal, or stuck
        while node.is_expanded and not node.is_terminal:
            if not node.children:
                # Expanded with zero children: nothing to descend into. Only reachable
                # if a policy returned empty priors for a state that HAS legal actions.
                self._mark_stop(node, game, "no_actions")
                break
            if first_step and force_first is not None:
                action_k = force_first
                child = node.children[action_k]
            else:
                action_k, child = self._select_child(node)
            first_step = False

            game.step(action_from_key(action_k))
            path.append(child)
            node = child
            if virtual_loss:
                child.visit_count += 1

            # If the new state stops descent, mark it and stop
            if outcome.is_terminal(game):
                self._mark_stop(node, game, "game_over")
                break
            if outcome.is_stuck(game):
                self._mark_stop(node, game, "stuck")
                break

        # EXPAND if we landed on an unexpanded non-terminal leaf
        if not node.is_terminal and not node.is_expanded:
            value = yield from self._expand_iter(node, game)
        else:
            value = node.terminal_value

        # BACKUP (with virtual loss the visit was already counted on the way down)
        for n in path:
            if not virtual_loss:
                n.visit_count += 1
            n.value_sum += value

    def _mark_stop(self, node: Node, game: BalatroGame, reason: str) -> float:
        """Mark a node as a descent stop and value it with the outcome function."""
        node.is_terminal = True
        node.stop_reason = reason
        node.terminal_value = self.outcome.value(game)
        return node.terminal_value

    # ── Expansion ────────────────────────────────────────────────────────────

    def _expand(self, node: Node, game: BalatroGame) -> float:
        """Query policy/value, create children with priors. Returns the value."""
        return self._drive(self._expand_iter(node, game))

    def _expand_iter(self, node: Node, game: BalatroGame):
        """Generator form of `_expand`: yields `[game]`, is sent `[(priors, value)]`."""
        outcome = self._outcome_for(game)
        if outcome.is_terminal(game):
            node.is_expanded = True
            return self._mark_stop(node, game, "game_over")
        if outcome.is_stuck(game):
            node.is_expanded = True
            return self._mark_stop(node, game, "stuck")

        priors, value = (yield [game])[0]
        return self._apply_expansion(node, game, priors, value)

    def _apply_expansion(self, node: Node, game: BalatroGame,
                         priors: dict, value: float) -> float:
        """Install an evaluation on a leaf. Idempotent: under virtual loss two in-flight
        simulations can reach the same unexpanded node and both be evaluated (identical
        states, identical results); the second one must NOT re-create the children,
        which would throw away the statistics the first one's children accumulated."""
        if node.is_expanded:
            return node.terminal_value if node.is_terminal else value
        if not priors:
            # Defensive: a state we did not classify as stopping, but with nothing to do.
            node.is_expanded = True
            return self._mark_stop(node, game, "no_actions")
        for k, p in priors.items():
            node.add_child(k, prior=p)
        node.is_expanded = True
        return value

    # ── PUCT selection ───────────────────────────────────────────────────────

    def _select_child(self, node: Node) -> tuple[ActionKey, Node]:
        """
        PUCT: argmax_a [Q(a) + c_puct * P(a) * sqrt(N(parent)) / (1 + N(a))]
        """
        N_parent = max(1, node.visit_count)
        sqrt_N = math.sqrt(N_parent)

        best_score = -float("inf")
        best_key = None
        best_child = None
        for k, child in node.children.items():
            q = child.mean_value
            u = self.cfg.c_puct * child.prior * sqrt_N / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_key = k
                best_child = child

        return best_key, best_child

    # ── Convenience: choose an action from a finished search ─────────────────

    @staticmethod
    def best_action(visit_counts: dict[ActionKey, int]) -> Optional[ActionKey]:
        """Argmax visit count — used at eval time. None on an empty search."""
        if not visit_counts:
            return None
        return max(visit_counts.items(), key=lambda kv: kv[1])[0]

    @staticmethod
    def sample_action(
        visit_counts: dict[ActionKey, int],
        temperature: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> Optional[ActionKey]:
        """
        Sample an action proportional to N^(1/T).

        T=1   -> sample proportional to visit counts (used in early self-play)
        T->0  -> argmax (eval / late self-play)
        T>1   -> flatter than visit counts (more exploratory)

        At T<=0, falls through to best_action (deterministic argmax).
        """
        if temperature <= 0 or not visit_counts:
            return MCTS.best_action(visit_counts)
        rng = rng if rng is not None else np.random.default_rng()
        items = list(visit_counts.items())
        counts = np.asarray([c for _, c in items], dtype=np.float64)
        if counts.sum() == 0:
            # No simulations got assigned — sample uniformly from legal actions
            idx = int(rng.integers(0, len(items)))
            return items[idx][0]
        if temperature == 1.0:
            probs = counts / counts.sum()
        else:
            logits = np.log(np.maximum(counts, 1e-12)) / temperature
            logits -= logits.max()
            probs = np.exp(logits)
            probs /= probs.sum()
        idx = int(rng.choice(len(items), p=probs))
        return items[idx][0]


def _is_win(game: BalatroGame) -> bool:
    """Back-compat shim for the balatro-mcts call sites. Prefer an OutcomeFn:
    this is the VANILLA rule and is wrong under MLB (endless; win == match_won)."""
    return game.state is State.GAME_OVER and game.ante > 8
