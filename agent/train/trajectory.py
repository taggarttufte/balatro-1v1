"""
trajectory.py — Sample dataclass + bounded replay buffer.

A Sample is one decision point: the encoded state, the featurized legal actions, the
search-improved policy target, and the eventual outcome z. The buffer is a bounded
deque sampled uniformly at random for training.

Fork note (2026-08-21): the buffer is now serialisable (`state_dict` / `load_state_dict`)
so `--resume` restores the exact training distribution. Without it a resumed run trains
on a different mini-batch stream than the uninterrupted run and the round-trip test can
only compare sample counts, not weights.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


@dataclass
class Sample:
    """One training tuple from self-play.

    obs              (D,)         encoded game state at decision time (447, or 453 with
                                  the MLB encoder)
    action_features  (N, A)       featurized legal actions, in fixed order (A = 56)
    target_policy    (N,)         search-improved policy, sums to 1
    z                scalar       outcome of the episode in [0, 1], from the OutcomeFn
                                  (filled in once the episode ends)
    """
    obs: np.ndarray
    action_features: np.ndarray
    target_policy: np.ndarray
    z: float = 0.0


class ReplayBuffer:
    """Bounded FIFO buffer of Samples. sample(k) draws k items with replacement."""

    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self._buf: deque[Sample] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._buf)

    def add(self, sample: Sample) -> None:
        self._buf.append(sample)

    def extend(self, samples: Iterable[Sample]) -> None:
        for s in samples:
            self._buf.append(s)

    def sample(self, batch_size: int, rng: np.random.Generator | None = None) -> list[Sample]:
        if len(self._buf) == 0:
            return []
        rng = rng if rng is not None else np.random.default_rng()
        idx = rng.integers(0, len(self._buf), size=batch_size)
        items = list(self._buf)
        return [items[i] for i in idx]

    # ── Checkpointing ────────────────────────────────────────────────────────

    def state_dict(self, max_items: Optional[int] = None) -> dict:
        """Serialisable snapshot. `max_items` keeps only the most recent N samples
        (bounding checkpoint size); `truncated` records whether anything was dropped,
        because a truncated buffer makes a resume non-bit-exact."""
        items = list(self._buf)
        truncated = max_items is not None and len(items) > max_items
        if truncated:
            items = items[-max_items:]
        return {
            "capacity": self.capacity,
            "truncated": truncated,
            "n_total": len(self._buf),
            "samples": [
                (s.obs, s.action_features, s.target_policy, float(s.z)) for s in items
            ],
        }

    def load_state_dict(self, sd: dict) -> None:
        self.capacity = sd.get("capacity", self.capacity)
        self._buf = deque(maxlen=self.capacity)
        for obs, feats, target, z in sd["samples"]:
            self._buf.append(Sample(obs=obs, action_features=feats,
                                    target_policy=target, z=float(z)))
