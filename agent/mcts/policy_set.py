"""
policy_set.py — `PolicyValueFn` implementations for the set encoder.

The `PolicyValueFn` contract is UNCHANGED (SETENC_NOTES §0.4):

    fn(game) -> ({ActionKey: prior}, value)
    fn.evaluate_many(games) -> [ ... ]        order preserved, ({}, 0.0) for a no-action game

`SetNNPolicy` is the reference (serial) implementation and `BatchedSetNNPolicy` overrides
only `evaluate_many` — the same split `policy.NNPolicy` / `batched.BatchedNNPolicy` use, and
for the same reason: the per-leaf CPU work (`encode_leaf`) is shared verbatim, so
"batched == single-leaf" is a meaningful test rather than a comparison of two codebases.

`search.py`, `batched.py`, `reuse.py` and `player.py::MCTSPlayer` are entirely
encoder-blind and were not touched: they only ever call the `PolicyValueFn`.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from balatro_sim.game import BalatroGame
from .action import ActionKey, action_key
from .action_features_set import featurize_actions_set
from .encoder_set import SetEncoder
from .model_set import SetPolicyValueNet
from .policy import Evaluation, PolicyValueBase

# Padded action rows per forward pass. A (B, maxN, 70) float block plus the (B, maxN, D)
# pooled blocks is ~1.2 KB per padded row, so 250k rows is ~300 MB — chunk below that.
DEFAULT_MAX_ACTION_ROWS = 120_000


class SetNNPolicy(PolicyValueBase):
    """Serial reference implementation for the set encoder."""

    def __init__(self, model: SetPolicyValueNet, device: str | torch.device = "cpu",
                 encoder: SetEncoder | None = None, hand_type_features: bool = True):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.encoder = encoder or SetEncoder(model.caps)
        self.hand_type_features = hand_type_features
        if self.encoder.caps != model.caps:
            raise ValueError(
                f"encoder caps {self.encoder.caps} != net caps {model.caps}; a checkpoint "
                "records the caps precisely so this cannot happen silently")

    # ── the CPU half of a leaf evaluation ────────────────────────────────────

    def encode_leaf(self, game: BalatroGame):
        """(legal_actions, obs dict, acts dict) or None when there is nothing to evaluate.
        Pure numpy — no torch, safe off the main thread."""
        legal = game.legal_actions()
        if not legal:
            return None
        return (legal,
                self.encoder(game),
                featurize_actions_set(game, legal, self.encoder.caps,
                                      hand_type_features=self.hand_type_features))

    @staticmethod
    def priors_from_logits(legal: list[dict], probs: np.ndarray) -> dict[ActionKey, float]:
        return {action_key(a): float(probs[i]) for i, a in enumerate(legal)}

    # ── PolicyValueFn ────────────────────────────────────────────────────────

    def __call__(self, game: BalatroGame) -> Evaluation:
        encoded = self.encode_leaf(game)
        if encoded is None:
            return {}, 0.0
        legal, obs, acts = encoded
        obs_t = _stack_obs([obs], self.device)
        acts_t, _ = _pad_acts([acts], [acts["act_type"].shape[0]], self.device)
        with torch.no_grad():
            logits, value = self.model(obs_t, acts_t)
            probs = torch.softmax(logits[0], dim=-1)
        return (self.priors_from_logits(legal, probs.detach().cpu().numpy()),
                float(value.item()))


class BatchedSetNNPolicy(SetNNPolicy):
    """One forward pass for B leaves. The action blocks are padded to (B, max_actions, ...)
    — which is what `batched._segment_softmax` already materialised for the flat net — and
    the padded rows are masked out of the softmax."""

    def __init__(self, model: SetPolicyValueNet, device: str | torch.device = "cpu",
                 encoder: SetEncoder | None = None, hand_type_features: bool = True,
                 max_action_rows: int = DEFAULT_MAX_ACTION_ROWS):
        super().__init__(model, device=device, encoder=encoder,
                         hand_type_features=hand_type_features)
        self.max_action_rows = max(1, int(max_action_rows))
        self.calls = 0
        self.forwards = 0
        self.leaves = 0
        self.batch_sizes: list[int] = []

    def evaluate_many(self, games: Sequence[BalatroGame]) -> list[Evaluation]:
        out: list[Evaluation] = [({}, 0.0)] * len(games)
        encoded = [self.encode_leaf(g) for g in games]
        live = [i for i, e in enumerate(encoded) if e is not None]
        self.calls += 1
        self.batch_sizes.append(len(live))
        if not live:
            return out
        self.leaves += len(live)

        # Chunk on PADDED rows: the cost of a chunk is len(chunk) * max_n, not the sum.
        chunk: list[int] = []
        max_n = 0
        for i in live:
            n = encoded[i][2]["act_type"].shape[0]
            new_max = max(max_n, n)
            if chunk and (len(chunk) + 1) * new_max > self.max_action_rows:
                self._forward_chunk(chunk, encoded, out)
                chunk, max_n = [], 0
                new_max = n
            chunk.append(i)
            max_n = new_max
        if chunk:
            self._forward_chunk(chunk, encoded, out)
        return out

    # ── one padded forward pass ──────────────────────────────────────────────

    def _forward_chunk(self, chunk: list[int], encoded: list, out: list) -> None:
        self.forwards += 1
        obs_t = _stack_obs([encoded[i][1] for i in chunk], self.device)
        counts = [encoded[i][2]["act_type"].shape[0] for i in chunk]
        acts_t, mask = _pad_acts([encoded[i][2] for i in chunk], counts, self.device)

        with torch.no_grad():
            state = self.model.encode_state(obs_t)
            values = self.model.value(state)
            logits = self.model.action_logits(state, acts_t)
            logits = logits.masked_fill(~mask, float("-inf"))
            probs = torch.softmax(logits, dim=-1)

        probs_np = probs.cpu().numpy()
        values_np = values.cpu().numpy()
        for j, i in enumerate(chunk):
            n = counts[j]
            out[i] = (self.priors_from_logits(encoded[i][0], probs_np[j, :n]),
                      float(values_np[j]))


# ── helpers (also used by the trainer) ──────────────────────────────────────────
#
# THE TRANSFER IS THE COST ON CUDA. A batched observation is 21 arrays and an action block
# is 4 more, so the obvious `torch.from_numpy(x).to(device)` per key is 25 small host->device
# copies per forward pass. Measured on this box (RTX 3080 Ti): 20 such copies cost 5.35 ms,
# while ONE concatenated copy of the same bytes costs 0.027 ms — 200x, and 5.35 ms was more
# than the forward pass itself (5.8 ms) for a 16-leaf batch. So for a non-CPU device the
# arrays are packed key-major into one float32 and one int16 buffer, transferred once, and
# split into contiguous views on the device.
#
# On CPU `torch.from_numpy` is a free view, so packing would only add a copy: the CPU path
# stays per-key. `tests/test_set_encoder.py::test_packed_and_unpacked_transfers_agree`
# pins the two paths equal.


def _numpy_dtype_groups(proto: dict) -> tuple[list, list]:
    floats = sorted(k for k, v in proto.items() if v.dtype == np.float32)
    ints = sorted(k for k, v in proto.items() if v.dtype != np.float32)
    return floats, ints


def _pack_group(arrays_by_key: dict, keys: list, dtype, device):
    """One contiguous host buffer -> one transfer -> per-key contiguous views."""
    if not keys:
        return {}
    blocks = [arrays_by_key[k] for k in keys]
    sizes = [b.size for b in blocks]
    buf = np.empty(sum(sizes), dtype=dtype)
    at = 0
    for b, n in zip(blocks, sizes):
        buf[at:at + n] = b.reshape(-1)
        at += n
    flat = torch.from_numpy(buf).to(device)
    out = {}
    at = 0
    for k, b, n in zip(keys, blocks, sizes):
        out[k] = flat[at:at + n].view(b.shape)
        at += n
    return out


def _to_device(arrays_by_key: dict, device) -> dict:
    if torch.device(device).type == "cpu":
        return {k: torch.from_numpy(v) for k, v in arrays_by_key.items()}
    floats, ints = _numpy_dtype_groups(arrays_by_key)
    out = _pack_group(arrays_by_key, floats, np.float32, device)
    if ints:
        # every categorical array is int16 (encoder_set.CAT_DTYPE)
        out.update(_pack_group(arrays_by_key, ints, arrays_by_key[ints[0]].dtype, device))
    return out


def _stack_obs(observations: Sequence[dict], device) -> dict:
    if len(observations) == 1:
        stacked = {k: v[None] for k, v in observations[0].items()}
    else:
        stacked = {k: np.stack([o[k] for o in observations]) for k in observations[0]}
    return _to_device(stacked, device)


def _pad_acts(blocks: Sequence[dict], counts: Sequence[int], device):
    """Pad per-state `Acts` dicts to (B, max_n, ...) + a (B, max_n) bool live-mask."""
    B = len(blocks)
    max_n = max(counts) if counts else 0
    padded = {}
    for k, proto in blocks[0].items():
        shape = (B, max_n) + proto.shape[1:]
        buf = np.zeros(shape, dtype=proto.dtype)
        for j, b in enumerate(blocks):
            n = counts[j]
            if n:
                buf[j, :n] = b[k]
        padded[k] = buf
    mask = np.zeros((B, max_n), dtype=bool)
    for j, n in enumerate(counts):
        mask[j, :n] = True
    return _to_device(padded, device), torch.from_numpy(mask).to(device)


stack_obs = _stack_obs
pad_acts = _pad_acts

__all__ = ["SetNNPolicy", "BatchedSetNNPolicy", "stack_obs", "pad_acts"]
