"""
batched.py — batched leaf evaluation for MCTS (Phase 3, W3).

Two pieces:

    BatchedNNPolicy   a `PolicyValueFn` whose `evaluate_many` evaluates B leaves in ONE
                      forward pass: (B, obs_dim) through the trunk, the ragged
                      (sum(N_i), A) action block through `score_actions_flat`, one
                      segmented softmax. Single-leaf `__call__` is inherited from
                      `NNPolicy` unchanged, so "batched == single-leaf" is testable.

    BatchedSearch     a scheduler that drives K independent searches in LOCKSTEP: every
                      tree descends to a leaf, all K leaves go to `evaluate_many` in one
                      call, every tree backs up, repeat. Trees that finish early simply
                      leave the pool; the batch shrinks.

Why this and not "make CUDA faster"
-----------------------------------
W1 measured single-leaf CUDA at 328-362 sims/s against CPU's 451-475 (AGENT_NOTES §4.4):
a 2.4M-param net evaluating one (447,) observation and one (436, 56) action block is far
too small to amortise a kernel launch plus two host<->device transfers. The fix is not a
faster kernel, it is a bigger one — which means finding several leaves to evaluate at
once. There are two independent sources of them:

  1. **Across trees** (the mandatory piece). The tournament runs N agents on one seed and
     self-play can run K games at once; their leaves are independent, so K trees in
     lockstep give a batch of K with no approximation whatsoever — each tree's search is
     bit-identical to running it alone (`tests/test_batched.py`).
  2. **Within one tree** (`MCTSConfig.leaf_batch`, optional). L simulations descend
     before any is evaluated, held apart by virtual loss. This DOES change the search
     (later descents in a batch see stale statistics), so it is opt-in and only worth it
     when a single tree is the entire workload — which is exactly the tournament's
     `Player.act(game)` shape today (`mp/tournament/runner.py::_drive_to_next_nemesis`
     drives one agent at a time). See BATCH_NOTES.md §3.

The two compose: K trees x L in flight = a batch of K*L.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch

from balatro_sim.game import BalatroGame
from .action import ActionKey
from .node import Node
from .outcome import OutcomeFn
from .policy import Evaluation, NNPolicy, PolicyValueFn
from .search import MCTS, MCTSConfig

# A (B, max_actions) padded logit block is materialised for the segmented softmax, and
# the flat action features are (sum(N_i), 56) float32. 4M action rows is ~900 MB of
# features, so chunk well below that; 250k rows is ~56 MB and already far past the point
# where the GPU is saturated.
DEFAULT_MAX_ACTION_ROWS = 250_000


class BatchedNNPolicy(NNPolicy):
    """`NNPolicy` + a real `evaluate_many`.

    Everything on the CPU side is `NNPolicy.encode_leaf` verbatim (the same numpy
    observation and the same action features); only the torch half differs. A leaf with
    no legal actions (MLB `PVP_WAIT`, readied at a Nemesis) returns `({}, 0.0)` and never
    reaches the net — it is dropped from the batch, not padded into it.

    `max_action_rows` caps how many action rows go into one forward pass; larger batches
    are split into several passes and stitched back together, so a caller can hand this
    100 trees x 436 actions without thinking about memory.
    """

    def __init__(self, model, device: str | torch.device = "cpu", encoder=None,
                 max_action_rows: int = DEFAULT_MAX_ACTION_ROWS):
        super().__init__(model, device=device, encoder=encoder)
        self.max_action_rows = max(1, int(max_action_rows))
        # Instrumentation (the benchmark reads these; nothing in the search does).
        self.calls = 0            # evaluate_many calls
        self.forwards = 0         # actual forward passes (>= calls when chunking)
        self.leaves = 0           # leaves evaluated by the net
        self.batch_sizes: list[int] = []

    # ── PolicyValueFn ────────────────────────────────────────────────────────

    def evaluate_many(self, games: Sequence[BalatroGame]) -> list[Evaluation]:
        out: list[Evaluation] = [({}, 0.0)] * len(games)
        encoded = [self.encode_leaf(g) for g in games]
        live = [i for i, e in enumerate(encoded) if e is not None]
        self.calls += 1
        if not live:
            self.batch_sizes.append(0)
            return out
        self.batch_sizes.append(len(live))
        self.leaves += len(live)

        chunk: list[int] = []
        rows = 0
        for i in live:
            n = encoded[i][2].shape[0]
            if chunk and rows + n > self.max_action_rows:
                self._forward_chunk(chunk, encoded, out)
                chunk, rows = [], 0
            chunk.append(i)
            rows += n
        if chunk:
            self._forward_chunk(chunk, encoded, out)
        return out

    # ── One forward pass over a ragged batch ────────────────────────────────

    def _forward_chunk(self, chunk: list[int], encoded: list, out: list) -> None:
        self.forwards += 1
        feat_list = [encoded[i][2] for i in chunk]
        counts = np.fromiter((f.shape[0] for f in feat_list), dtype=np.int64,
                             count=len(feat_list))
        obs = np.stack([encoded[i][1] for i in chunk])
        flat = feat_list[0] if len(feat_list) == 1 else np.concatenate(feat_list, axis=0)

        obs_t = torch.from_numpy(obs).to(self.device)
        feats_t = torch.from_numpy(flat).to(self.device)
        counts_t = torch.from_numpy(counts).to(self.device)

        with torch.no_grad():
            trunk = self.model.get_trunk(obs_t)                       # (B, H)
            values = self.model.value(trunk)                          # (B,)
            logits = self.model.score_actions_flat(trunk, feats_t, counts_t)
            probs = _segment_softmax(logits, counts_t)

        probs_np = probs.cpu().numpy()
        values_np = values.cpu().numpy()
        at = 0
        for j, i in enumerate(chunk):
            n = int(counts[j])
            legal = encoded[i][0]
            out[i] = (self.priors_from_logits(legal, probs_np[at:at + n]),
                      float(values_np[j]))
            at += n


def _segment_softmax(logits: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Softmax each variable-length segment of a flat logit vector, in one kernel chain.

    Scatters the segments into a (B, max_count) block padded with -inf, softmaxes the
    block (padding exponentiates to exactly 0 and contributes nothing to the row sum),
    and gathers the live entries back out. B small softmax calls would be correct too but
    cost B kernel launches, which is precisely the overhead this module exists to avoid.
    """
    if counts.numel() == 1:
        return torch.softmax(logits, dim=-1)
    max_n = int(counts.max())
    mask = torch.arange(max_n, device=logits.device)[None, :] < counts[:, None]
    padded = torch.full((counts.numel(), max_n), float("-inf"),
                        device=logits.device, dtype=logits.dtype)
    padded[mask] = logits
    return torch.softmax(padded, dim=-1)[mask]


# ── The K-tree scheduler ────────────────────────────────────────────────────────


@dataclass
class SearchRequest:
    """One tree's search. `mcts` carries that tree's config / rng / outcome, so a
    heterogeneous population (different budgets, different checkpoints) batches together
    as happily as K clones of one agent."""
    game: BalatroGame
    mcts: MCTS
    strategy: str = "gumbel"           # "gumbel" | "puct"
    root: Optional[Node] = None        # tree reuse: resume into this root
    sims: Optional[int] = None         # budget override (reuse subtracts retained visits)
    add_noise: bool = True             # PUCT only; Gumbel does its own exploration
    leaf_batch: Optional[int] = None   # within-tree leaf batching; None -> mcts.cfg


@dataclass
class SearchResult:
    root: Node
    visit_counts: dict[ActionKey, int]
    chosen: Optional[ActionKey] = None   # Gumbel's pick; None for PUCT (caller decides)


@dataclass
class BatchStats:
    """Where a batched run spent itself. All wall clock, `time.perf_counter`."""
    rounds: int = 0                 # lockstep rounds (== forward-pass batches)
    leaves: int = 0                 # leaves evaluated
    trees: int = 0
    sims: int = 0
    nn_seconds: float = 0.0         # inside evaluate_many
    total_seconds: float = 0.0
    batch_sizes: list[int] = field(default_factory=list)

    @property
    def mean_batch(self) -> float:
        return float(np.mean(self.batch_sizes)) if self.batch_sizes else 0.0

    @property
    def other_seconds(self) -> float:
        """Everything that is not the net: clone/step, PUCT, dict churn, backup."""
        return max(0.0, self.total_seconds - self.nn_seconds)


class BatchedSearch:
    """Runs K searches in lockstep so their leaves share one forward pass.

    Contract: with `leaf_batch == 1` each tree's search is **bit-identical** to running
    that tree alone through `MCTS.run` / `MCTS.run_gumbel` with the same rng — batching
    only changes *when* an evaluation happens, never *what* it is (given a deterministic
    policy; see BATCH_NOTES.md §2 on float reproducibility). Pinned by
    `tests/test_batched.py::test_k_trees_match_k_single_tree_searches`.

    Early finish: a tree whose search returns (budget spent, no legal actions at the
    root, terminal root) drops out of the pool at the round in which it finishes. The
    remaining trees keep batching at the smaller size; nothing waits.
    """

    def __init__(self, policy: PolicyValueFn, config: MCTSConfig | None = None,
                 outcome: OutcomeFn | None = None, strategy: str = "gumbel"):
        self.policy = policy
        self.config = config or MCTSConfig()
        self.outcome = outcome
        self.strategy = strategy
        self.stats = BatchStats()

    # ── Convenience: K games, one config ────────────────────────────────────

    def make_requests(self, games: Sequence[BalatroGame],
                      seeds: Sequence[int] | None = None,
                      strategy: str | None = None,
                      roots: Sequence[Optional[Node]] | None = None,
                      sims: Sequence[Optional[int]] | None = None,
                      add_noise: bool = True) -> list[SearchRequest]:
        strategy = strategy or self.strategy
        out = []
        for i, g in enumerate(games):
            seed = i if seeds is None else seeds[i]
            mcts = MCTS(self.policy, self.config, rng=np.random.default_rng(seed),
                        outcome=self.outcome)
            out.append(SearchRequest(
                game=g, mcts=mcts, strategy=strategy, add_noise=add_noise,
                root=None if roots is None else roots[i],
                sims=None if sims is None else sims[i],
            ))
        return out

    def run_many(self, games: Sequence[BalatroGame], **kwargs) -> list[SearchResult]:
        return self.run_requests(self.make_requests(games, **kwargs))

    # ── The lockstep loop ───────────────────────────────────────────────────

    def run_requests(self, requests: Sequence[SearchRequest]) -> list[SearchResult]:
        t0 = time.perf_counter()
        results: list[Optional[SearchResult]] = [None] * len(requests)
        active: list[tuple[int, object, list]] = []

        for i, req in enumerate(requests):
            gen = self._make_generator(req)
            self._advance(i, gen, None, results, active, first=True)

        while active:
            leaves: list[BalatroGame] = []
            for _, _, request in active:
                leaves.extend(request)
            t_nn = time.perf_counter()
            evaluations = self.policy.evaluate_many(leaves)
            self.stats.nn_seconds += time.perf_counter() - t_nn
            self.stats.rounds += 1
            self.stats.leaves += len(leaves)
            self.stats.batch_sizes.append(len(leaves))

            pending, active = active, []
            at = 0
            for i, gen, request in pending:
                reply = evaluations[at:at + len(request)]
                at += len(request)
                self._advance(i, gen, reply, results, active, first=False)

        self.stats.total_seconds += time.perf_counter() - t0
        self.stats.trees += len(requests)
        self.stats.sims += sum(
            (r.mcts.cfg.num_simulations if r.sims is None else r.sims) for r in requests
        )
        return [r if r is not None else SearchResult(Node(), {}, None) for r in results]

    # ── Generator plumbing ──────────────────────────────────────────────────

    def _make_generator(self, req: SearchRequest):
        if req.strategy == "gumbel":
            return req.mcts.run_gumbel_iter(req.game, root=req.root, sims=req.sims,
                                            leaf_batch=req.leaf_batch)
        return req.mcts.run_iter(req.game, add_noise=req.add_noise, root=req.root,
                                 sims=req.sims, leaf_batch=req.leaf_batch)

    @staticmethod
    def _advance(i: int, gen, reply, results: list, active: list, first: bool) -> None:
        """Push a tree to its next leaf request, or retire it with its result."""
        try:
            request = next(gen) if first else gen.send(reply)
        except StopIteration as stop:
            value = stop.value
            if value is None:                      # defensive: generator with no return
                results[i] = SearchResult(Node(), {}, None)
            elif len(value) == 3:
                root, visits, chosen = value
                results[i] = SearchResult(root, visits, chosen)
            else:
                root, visits = value
                results[i] = SearchResult(root, visits, None)
            return
        active.append((i, gen, request))


__all__ = [
    "BatchedNNPolicy", "BatchedSearch", "SearchRequest", "SearchResult", "BatchStats",
    "DEFAULT_MAX_ACTION_ROWS",
]
