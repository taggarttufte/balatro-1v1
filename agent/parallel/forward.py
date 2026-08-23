"""
forward.py — the torch half of a leaf evaluation, given leaves that are already encoded.

``BatchedNNPolicy.evaluate_many`` and ``BatchedSetNNPolicy.evaluate_many`` each do two
things: encode B games (pure numpy, `mcts` calls it ``encode_leaf``) and then run one
forward pass over the batch.  In a multi-process run those two halves happen in different
processes — the workers encode, the evaluator forwards — so this module is the second half
on its own.

The bodies below are ``batched.BatchedNNPolicy._forward_chunk`` and
``policy_set.BatchedSetNNPolicy._forward_chunk`` transposed to take encoded leaves instead
of games, with the same chunking rules (flat: cap on TOTAL action rows, because its action
block is ragged and concatenated; set: cap on PADDED rows, because its block is padded to
``(B, max_n, ...)`` and the cost is ``len(chunk) * max_n``).  Nothing about the maths
changes, and ``tests/test_parallel.py::test_forward_matches_batched_policy`` pins the two
paths **exactly equal** on real leaves, for both encoders, on CPU and (when present) CUDA.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from mcts.batched import DEFAULT_MAX_ACTION_ROWS, _segment_softmax
from mcts.policy_set import DEFAULT_MAX_ACTION_ROWS as DEFAULT_MAX_PADDED_ROWS
from mcts.policy_set import _pad_acts, _stack_obs


def default_max_rows(is_set: bool) -> int:
    return DEFAULT_MAX_PADDED_ROWS if is_set else DEFAULT_MAX_ACTION_ROWS


def forward_leaves(model, obs_list: Sequence, acts_list: Sequence, *, is_set: bool,
                   device, max_action_rows: int = 0) -> tuple:
    """One or more forward passes over ``len(obs_list)`` encoded leaves.

    Returns ``(probs, values)``: ``probs[i]`` is a ``(n_i,)`` float32 array of action
    priors for leaf ``i`` and ``values`` is a ``(B,)`` float32 array.  Never called with an
    empty batch (a leaf with no legal actions never reaches the net — the worker answers
    ``({}, 0.0)`` for it, exactly as ``BatchedNNPolicy`` drops it from the batch).
    """
    if not obs_list:                                 # pragma: no cover - defensive
        return [], np.zeros(0, dtype=np.float32)
    cap = int(max_action_rows) or default_max_rows(is_set)
    fn = _forward_chunk_set if is_set else _forward_chunk_flat

    probs: list = [None] * len(obs_list)
    values = np.zeros(len(obs_list), dtype=np.float32)
    chunk: list = []
    rows = 0
    max_n = 0
    for i, acts in enumerate(acts_list):
        n = _n_actions(acts, is_set)
        if is_set:
            new_max = max(max_n, n)
            if chunk and (len(chunk) + 1) * new_max > cap:
                fn(model, obs_list, acts_list, chunk, device, probs, values)
                chunk, max_n = [], n
            else:
                max_n = new_max
        else:
            if chunk and rows + n > cap:
                fn(model, obs_list, acts_list, chunk, device, probs, values)
                chunk, rows = [], 0
            rows += n
        chunk.append(i)
    if chunk:
        fn(model, obs_list, acts_list, chunk, device, probs, values)
    return probs, values


def _n_actions(acts, is_set: bool) -> int:
    return int(acts["act_type"].shape[0]) if is_set else int(acts.shape[0])


# ── flat encoder (v7 / mlb) ─────────────────────────────────────────────────────

def _forward_chunk_flat(model, obs_list, acts_list, chunk, device, probs, values) -> None:
    feat_list = [acts_list[i] for i in chunk]
    counts = np.fromiter((f.shape[0] for f in feat_list), dtype=np.int64,
                         count=len(feat_list))
    obs = np.stack([obs_list[i] for i in chunk])
    flat = feat_list[0] if len(feat_list) == 1 else np.concatenate(feat_list, axis=0)

    obs_t = torch.from_numpy(np.ascontiguousarray(obs)).to(device)
    feats_t = torch.from_numpy(np.ascontiguousarray(flat)).to(device)
    counts_t = torch.from_numpy(counts).to(device)

    with torch.no_grad():
        trunk = model.get_trunk(obs_t)
        vals = model.value(trunk)
        logits = model.score_actions_flat(trunk, feats_t, counts_t)
        p = _segment_softmax(logits, counts_t)

    p_np = p.cpu().numpy()
    v_np = vals.cpu().numpy()
    at = 0
    for j, i in enumerate(chunk):
        n = int(counts[j])
        probs[i] = p_np[at:at + n]
        values[i] = float(v_np[j])
        at += n


# ── set encoder ─────────────────────────────────────────────────────────────────

def _forward_chunk_set(model, obs_list, acts_list, chunk, device, probs, values) -> None:
    obs_t = _stack_obs([obs_list[i] for i in chunk], device)
    counts = [int(acts_list[i]["act_type"].shape[0]) for i in chunk]
    acts_t, mask = _pad_acts([acts_list[i] for i in chunk], counts, device)

    with torch.no_grad():
        state = model.encode_state(obs_t)
        vals = model.value(state)
        logits = model.action_logits(state, acts_t)
        logits = logits.masked_fill(~mask, float("-inf"))
        p = torch.softmax(logits, dim=-1)

    p_np = p.cpu().numpy()
    v_np = vals.cpu().numpy()
    for j, i in enumerate(chunk):
        probs[i] = p_np[j, :counts[j]]
        values[i] = float(v_np[j])


__all__ = ["forward_leaves", "default_max_rows"]
