"""
remote.py — the worker's ``PolicyValueFn``: a leaf evaluator with no net behind it.

``RemotePolicy`` satisfies the same protocol ``mcts/policy.py`` publishes
(``__call__(game) -> (priors, value)``, ``evaluate_many(games) -> [...]``), so ``MCTS``,
``BatchedSearch``, ``MCTSPlayer`` and ``TreeCache`` are as blind to it as they are to the
flat / set split — not one line of the search knows the net is in another process.

What it actually does per ``evaluate_many``:

1. ``LeafEncoder.encode_leaf`` on every game (the ~13%-of-a-search per-leaf numpy work
   BATCH_NOTES §6 measured, which is exactly the work that SHOULD be in the worker);
2. pack the live leaves into this worker's shared-memory request arena;
3. one queue message with the offsets, then block on the reply pipe;
4. read the priors straight out of the reply arena and build the ``{ActionKey: p}`` dicts.

A game with no legal actions returns ``({}, 0.0)`` and never leaves the process, matching
``BatchedNNPolicy`` exactly.  A batch too big for the arena is split and submitted in
pieces; nothing above notices.
"""
from __future__ import annotations

from typing import Sequence

from mcts.policy import PolicyValueBase

from .channel import reply_offsets


class RemotePolicy(PolicyValueBase):
    """One per (worker, net).  ``policy_id`` selects the net on the evaluator side: 0 is
    the live net being trained, 1..P are the past-self checkpoints in the population."""

    def __init__(self, policy_id: int, channel, leaf_encoder, layout, name: str = ""):
        self.policy_id = int(policy_id)
        self.channel = channel
        self.leaf = leaf_encoder
        self.layout = layout
        self.name = name or f"remote{policy_id}"
        # Same instrumentation names BatchedNNPolicy exposes, so `selfplay._leaf_count`
        # and the benchmarks read a RemotePolicy the way they read a local one.
        self.calls = 0
        self.forwards = 0
        self.leaves = 0
        self.batch_sizes: list = []

    # ── PolicyValueFn ────────────────────────────────────────────────────────

    def __call__(self, game):
        return self.evaluate_many([game])[0]

    def evaluate_many(self, games: Sequence) -> list:
        out: list = [({}, 0.0)] * len(games)
        encoded = [self.leaf.encode_leaf(g) for g in games]
        live = [i for i, e in enumerate(encoded) if e is not None]
        self.calls += 1
        self.batch_sizes.append(len(live))
        if not live:
            return out
        self.leaves += len(live)

        arena = self.channel.arena
        chunk: list = []
        metas: list = []
        offset = 0
        reply_floats = 0
        for i in live:
            n = len(encoded[i][0])
            need = self.layout.record_bytes(n)
            rep_need = 1 + n
            if chunk and (offset + need > arena.request_bytes
                          or reply_floats + rep_need > arena.reply_floats):
                self._exchange(chunk, metas, encoded, out)
                chunk, metas, offset, reply_floats = [], [], 0, 0
            if need > arena.request_bytes or rep_need > arena.reply_floats:
                raise ValueError(
                    f"a single leaf with {n} actions needs {need} request bytes / "
                    f"{rep_need} reply floats, more than the arena holds "
                    f"({arena.request_bytes} / {arena.reply_floats}). Raise "
                    "--worker-arena-mb.")
            self.layout.pack(arena.request, offset, encoded[i][1], encoded[i][2], n)
            chunk.append(i)
            metas.append((n, offset))
            offset += need
            reply_floats += rep_need
        if chunk:
            self._exchange(chunk, metas, encoded, out)
        return out

    # ── one round trip ───────────────────────────────────────────────────────

    def _exchange(self, chunk: list, metas: list, encoded: list, out: list) -> None:
        self.forwards += 1
        self.channel.submit_and_wait(self.policy_id, metas)
        rep = self.channel.arena.reply
        offsets = reply_offsets(metas)
        for j, i in enumerate(chunk):
            n = metas[j][0]
            probs = rep[offsets[j]:offsets[j] + n]
            out[i] = (self.leaf.priors_from_logits(encoded[i][0], probs),
                      float(rep[j]))


__all__ = ["RemotePolicy"]
