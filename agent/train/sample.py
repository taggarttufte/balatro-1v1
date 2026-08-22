"""
sample.py — `Sample` v2: subsampled action sets, encoder-agnostic (Phase 4 W1).

The Phase 3 `Sample` (`train/trajectory.py`) carried `action_features` for EVERY legal
action: ~436 x 56 float32 = 97 KB at a `SELECTING_HAND` leaf, against 1.8 KB for the
observation. 141 episodes made a 130 MB checkpoint and the balatro-mcts default
`buffer_capacity=200_000` would have been ~19 GB of RAM (AGENT_NOTES §3).

Phase 4 brief §0.2 (Tagg's decision): keep every action the search VISITED plus a few
random zero-visit ones and renormalise the policy target. Because every visited action is
kept, the renormalisation is EXACT — the kept-row distribution is the full distribution
restricted to its own support — and what is lost is only the explicit "these 400 actions
are worth 0" signal, which is now sampled (`k_unvisited` negatives per state, drawn afresh
each time the state is seen) instead of exhaustive. That is the standard sampled-softmax
treatment for a large action space; SETENC_NOTES §4.3 records the bias it introduces.

Encoder-agnostic on purpose: subsampling is orthogonal to the observation encoding, so the
same `Sample` v2 carries a flat (D,) observation with a (k, 56) action block OR the set
encoder's dict-of-arrays with a dict-of-(k, ...) action block. `version` and the shape of
`obs` say which.

Interop with W2 (`train/selfplay.py`)
-------------------------------------
`SampleBuilder` implements W2's `SampleCollector(sample_fn=...)` protocol verbatim —
`sample_fn(game, legal, legal_keys, visits, encoder, z) -> Sample` — so switching a
training run onto subsampled / set-encoded samples is one constructor argument at the
collector, with no change to the collector itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from mcts.action_features import featurize_actions
from mcts.action_features_set import featurize_actions_set

SAMPLE_VERSION = 2
DEFAULT_K_UNVISITED = 8


@dataclass
class Sample:
    """One decision point, with the action set subsampled.

    obs            (D,) float32                 flat encoder
                   dict[str, ndarray]           set encoder (SETENC_NOTES §0.2)
    actions        (k, 56) float32              flat encoder
                   dict[str, ndarray], k rows   set encoder (SETENC_NOTES §0.3)
    target_policy  (k,) float32, sums to 1      search visits renormalised over the k rows
    z              float                        outcome / value target in [0, 1]
    meta           dict                         seed, ante, state, n_legal, k, encoder,
                                                visited  (+ anything the caller adds)
    """
    obs: Any
    actions: Any
    target_policy: np.ndarray
    z: float = 0.0
    meta: dict = field(default_factory=dict)
    version: int = SAMPLE_VERSION

    # ── convenience ──────────────────────────────────────────────────────────

    @property
    def is_set(self) -> bool:
        return isinstance(self.obs, dict)

    @property
    def k(self) -> int:
        return int(self.target_policy.shape[0])

    def nbytes(self) -> int:
        return sample_nbytes(self)


# ════════════════════════════════════════════════════════════════════════════════
# Subsampling
# ════════════════════════════════════════════════════════════════════════════════

def subsample_indices(counts: np.ndarray, k_unvisited: int,
                      rng: Optional[np.random.Generator]) -> np.ndarray:
    """Indices to keep: every visited action + `k_unvisited` uniformly-drawn zero-visit
    ones, returned in ASCENDING order (stable, independent of the draw order).

    `k_unvisited <= 0` keeps only the visited actions — except when nothing was visited,
    where at least one row must survive or the sample carries no action at all."""
    visited = np.flatnonzero(counts > 0)
    unvisited = np.flatnonzero(counts <= 0)
    n_extra = max(0, min(int(k_unvisited), unvisited.size))
    if n_extra == 0 and visited.size == 0:
        n_extra = min(1, unvisited.size)
    if n_extra == 0:
        return np.sort(visited)
    rng = rng if rng is not None else np.random.default_rng()
    extra = rng.choice(unvisited, size=n_extra, replace=False)
    return np.sort(np.concatenate([visited, extra]))


def renormalised_target(counts: np.ndarray, keep: np.ndarray) -> np.ndarray:
    kept = counts[keep].astype(np.float64)
    total = kept.sum()
    if total > 0:
        return (kept / total).astype(np.float32)
    return np.full(keep.size, 1.0 / max(keep.size, 1), dtype=np.float32)


# ════════════════════════════════════════════════════════════════════════════════
# Building
# ════════════════════════════════════════════════════════════════════════════════

class SampleBuilder:
    """Turns `(game, legal, legal_keys, visits, z)` into a `Sample` v2.

    Callable with W2's `sample_fn` signature, so
    `SampleCollector(idx, encoder, sample_fn=SampleBuilder(encoder, rng=rng))` is the
    whole wiring.
    """

    def __init__(self, encoder, k_unvisited: int = DEFAULT_K_UNVISITED,
                 subsample: bool = True, rng: Optional[np.random.Generator] = None,
                 hand_type_features: bool = True, keep_meta: bool = True):
        self.encoder = encoder
        self.k_unvisited = int(k_unvisited)
        self.subsample = bool(subsample)
        self.rng = rng
        self.hand_type_features = bool(hand_type_features)
        self.keep_meta = bool(keep_meta)

    @property
    def is_set(self) -> bool:
        return bool(getattr(self.encoder, "is_set", False))

    def __call__(self, game, legal: Sequence[dict], legal_keys: Sequence,
                 visits: Optional[dict] = None, encoder=None, z: float = 0.0,
                 meta: Optional[dict] = None) -> Sample:
        return self.make_sample(game, legal, legal_keys, visits, encoder, z, meta)

    def make_sample(self, game, legal: Sequence[dict], legal_keys: Sequence,
                    visits: Optional[dict] = None, encoder=None, z: float = 0.0,
                    meta: Optional[dict] = None) -> Sample:
        enc = encoder if encoder is not None else self.encoder
        n = len(legal)
        if n == 0:
            raise ValueError("make_sample called on a state with no legal actions")

        counts = np.zeros(n, dtype=np.float64)
        if visits:
            for i, key in enumerate(legal_keys):
                counts[i] = visits.get(key, 0)

        if self.subsample:
            keep = subsample_indices(counts, self.k_unvisited, self.rng)
        else:
            keep = np.arange(n)
        target = renormalised_target(counts, keep)

        kept_actions = [legal[i] for i in keep]
        obs = enc(game)
        if getattr(enc, "is_set", False):
            actions = featurize_actions_set(
                game, kept_actions, enc.caps,
                hand_type_features=self.hand_type_features)
        else:
            actions = featurize_actions(kept_actions)

        m: dict = {}
        if self.keep_meta:
            m = {
                "seed": getattr(game, "seed_str", None),
                "ante": int(getattr(game, "ante", 0)),
                "state": game.state.name,
                "n_legal": n,
                "k": int(keep.size),
                "encoder": getattr(enc, "name", "?"),
                "visited": int((counts > 0).sum()),
            }
        if meta:
            m.update(meta)
        return Sample(obs=obs, actions=actions, target_policy=target,
                      z=float(z), meta=m)


def make_sample_v2(game, legal: Sequence[dict], legal_keys: Sequence,
                   visits: Optional[dict] = None, encoder=None, z: float = 0.0,
                   *, k_unvisited: int = DEFAULT_K_UNVISITED,
                   rng: Optional[np.random.Generator] = None,
                   subsample: bool = True, meta: Optional[dict] = None) -> Sample:
    """One-shot form of `SampleBuilder`. Same argument order as W2's `sample_fn`."""
    return SampleBuilder(encoder, k_unvisited=k_unvisited, subsample=subsample,
                         rng=rng).make_sample(game, legal, legal_keys, visits,
                                              encoder, z, meta)


# ════════════════════════════════════════════════════════════════════════════════
# Sizing / serialisation
# ════════════════════════════════════════════════════════════════════════════════

def _nbytes(x) -> int:
    if isinstance(x, np.ndarray):
        return int(x.nbytes)
    if isinstance(x, dict):
        return sum(_nbytes(v) for v in x.values())
    return 0


def sample_nbytes(s) -> int:
    """Array payload of a Sample (v1 or v2), in bytes. What the replay buffer costs."""
    if getattr(s, "version", 1) >= 2:
        return _nbytes(s.obs) + _nbytes(s.actions) + _nbytes(s.target_policy)
    return (_nbytes(s.obs) + _nbytes(s.action_features) + _nbytes(s.target_policy))


def to_state(s: Sample) -> tuple:
    """Serialisable tuple form (what the buffer's `state_dict` stores)."""
    return (s.obs, s.actions, s.target_policy, float(s.z), dict(s.meta), int(s.version))


def from_state(t: tuple) -> Sample:
    obs, actions, target, z, meta, version = t
    return Sample(obs=obs, actions=actions, target_policy=target, z=float(z),
                  meta=dict(meta or {}), version=int(version))


__all__ = [
    "Sample", "SampleBuilder", "make_sample_v2", "sample_nbytes",
    "subsample_indices", "renormalised_target", "to_state", "from_state",
    "SAMPLE_VERSION", "DEFAULT_K_UNVISITED",
]
