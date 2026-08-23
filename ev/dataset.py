"""
dataset.py — label shards for V (Phase 5 rev 2, W5).

A **row** is ``(obs, y, meta)``:

* ``obs``  — the observation dict from W1's ``SetEncoderV2(game, opponent_view(match, p))``:
  ``{key: np.ndarray}`` with a FIXED shape per key (caps are transport, see SETENC_NOTES);
  this module never looks inside it, so a different encoder (or the test dummy) works too.
* ``y``    — the soft label ``P(player wins the MLB match | state)`` in [0, 1].
* ``meta`` — JSON-able dict; the fields below are also stored as typed columns so a split
  or a filter never has to parse JSON: ``seed`` (str), ``step`` (int), ``player`` (int),
  ``kind`` (str, ``labels.STATE_KINDS``), ``ante`` (int), ``ci`` (float, half-width),
  ``n_rollouts`` (int), ``trunc_frac`` (float, fraction of rollouts closed by the race
  calculator instead of a real outcome).

A **shard** is one ``.npz`` (compressed): ``obs__<key>`` stacked along axis 0, ``y``,
``seed`` / ``step`` / ``player`` / ``kind`` / ``ante`` / ``ci`` / ``n_rollouts`` /
``trunc_frac`` columns, and ``meta_json`` (one JSON string per row).  A shard is written
atomically (temp + ``os.replace``).  Rows within a shard come from whatever jobs finished
between two pool checkpoints, so a shard mixes seeds — the split is by seed, never by shard
or row.

``LabelDataset`` loads many shards into memory (50k rows x a few KB is a few hundred MB at
most), offers ``split_by_seed`` (held-out seeds chosen by a hash of the seed string, so the
split is stable across runs and across shard boundaries; explicit ``holdout_seeds`` also
accepted) and ``batches`` (index-select per key → numpy; the trainer's collate turns it into
tensors).
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

import numpy as np

__all__ = [
    "LabelRow", "Shard", "save_shard", "load_shard", "LabelDataset", "seed_in_holdout",
    "META_COLUMNS", "list_shards",
]

SHARD_VERSION = 1
META_COLUMNS = ("seed", "step", "player", "kind", "ante", "ci", "n_rollouts", "trunc_frac")
_COLUMN_DTYPES = {
    "seed": "U16", "step": np.int32, "player": np.int8, "kind": "U16", "ante": np.int16,
    "ci": np.float32, "n_rollouts": np.int16, "trunc_frac": np.float32,
}
_COLUMN_DEFAULTS = {"seed": "", "step": -1, "player": -1, "kind": "", "ante": -1,
                    "ci": float("nan"), "n_rollouts": 0, "trunc_frac": 0.0}


@dataclass
class LabelRow:
    obs: dict
    y: float
    meta: dict = field(default_factory=dict)


@dataclass
class Shard:
    obs: dict                 # {key: (N, ...) array}
    y: np.ndarray             # (N,) float32
    columns: dict             # {name: (N,) array} for META_COLUMNS
    meta: list                # N dicts
    path: Optional[str] = None

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def rows(self) -> Iterator[LabelRow]:
        for i in range(len(self)):
            yield LabelRow({k: v[i] for k, v in self.obs.items()}, float(self.y[i]), self.meta[i])


# ── save / load ───────────────────────────────────────────────────────────────────

def _columns_from_meta(metas: Sequence[dict]) -> dict:
    cols = {}
    for name in META_COLUMNS:
        dt = _COLUMN_DTYPES[name]
        vals = [m.get(name, _COLUMN_DEFAULTS[name]) for m in metas]
        cols[name] = np.asarray(vals, dtype=dt)
    return cols


def save_shard(path, rows: Sequence[LabelRow]) -> Path:
    """Write ``rows`` as one compressed ``.npz``; returns the path.  Empty ``rows`` raises."""
    rows = list(rows)
    if not rows:
        raise ValueError("save_shard: no rows")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].obs.keys())
    payload = {"version": np.asarray(SHARD_VERSION), "obs_keys": np.asarray(keys)}
    for k in keys:
        payload[f"obs__{k}"] = np.stack([np.asarray(r.obs[k]) for r in rows])
    payload["y"] = np.asarray([r.y for r in rows], dtype=np.float32)
    metas = [dict(r.meta) for r in rows]
    payload.update(_columns_from_meta(metas))
    payload["meta_json"] = np.asarray([json.dumps(m, sort_keys=True, default=str) for m in metas])
    tmp = path.with_suffix(path.suffix + ".tmp.npz")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **payload)
    os.replace(tmp, path)
    return path


def load_shard(path) -> Shard:
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        keys = [str(k) for k in z["obs_keys"]]
        obs = {k: z[f"obs__{k}"] for k in keys}
        y = z["y"].astype(np.float32)
        columns = {name: z[name] for name in META_COLUMNS if name in z.files}
        meta = [json.loads(s) for s in z["meta_json"]]
    return Shard(obs=obs, y=y, columns=columns, meta=meta, path=str(path))


def list_shards(spec) -> list:
    """``spec``: a directory (all ``*.npz`` inside, sorted), a glob, a single file, or a
    list/tuple of those."""
    if isinstance(spec, (list, tuple)):
        out = []
        for s in spec:
            out.extend(list_shards(s))
        return out
    p = Path(spec)
    if p.is_dir():
        return sorted(q for q in p.glob("*.npz") if not q.name.endswith(".tmp.npz"))
    if p.exists():
        return [p]
    import glob
    return sorted(Path(q) for q in glob.glob(str(spec)) if not q.endswith(".tmp.npz"))


# ── the held-out split ────────────────────────────────────────────────────────────

def seed_in_holdout(seed: str, holdout_frac: float, salt: str = "v-holdout") -> bool:
    """Stable, shard-independent membership test: ``sha1(salt + seed)`` → [0, 1) < frac."""
    if holdout_frac <= 0:
        return False
    if holdout_frac >= 1:
        return True
    h = hashlib.sha1(f"{salt}:{seed}".encode("utf-8")).digest()
    u = int.from_bytes(h[:8], "big") / float(1 << 64)
    return u < holdout_frac


# ── the in-memory dataset ─────────────────────────────────────────────────────────

class LabelDataset:
    """Many shards, concatenated.  ``obs[k]`` is (N, ...); ``y`` is (N,); ``columns`` are the
    typed meta columns; ``meta`` the dicts."""

    def __init__(self, obs: dict, y: np.ndarray, columns: dict, meta: list, sources=()):
        self.obs = obs
        self.y = y
        self.columns = columns
        self.meta = meta
        self.sources = list(sources)

    @classmethod
    def from_shards(cls, shards: Iterable) -> "LabelDataset":
        loaded = [s if isinstance(s, Shard) else load_shard(s) for s in shards]
        loaded = [s for s in loaded if len(s)]
        if not loaded:
            return cls({}, np.zeros((0,), np.float32), {n: np.zeros((0,), _COLUMN_DTYPES[n]) for n in META_COLUMNS}, [])
        keys = list(loaded[0].obs.keys())
        for s in loaded[1:]:
            if list(s.obs.keys()) != keys:
                raise ValueError(f"shard {s.path} has obs keys {list(s.obs.keys())} != {keys}")
        obs = {k: np.concatenate([s.obs[k] for s in loaded]) for k in keys}
        y = np.concatenate([s.y for s in loaded]).astype(np.float32)
        columns = {n: np.concatenate([s.columns[n] for s in loaded]) for n in META_COLUMNS
                   if all(n in s.columns for s in loaded)}
        meta = [m for s in loaded for m in s.meta]
        return cls(obs, y, columns, meta, sources=[s.path for s in loaded])

    @classmethod
    def load(cls, spec) -> "LabelDataset":
        return cls.from_shards(list_shards(spec))

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def subset(self, idx) -> "LabelDataset":
        idx = np.asarray(idx)
        return LabelDataset({k: v[idx] for k, v in self.obs.items()}, self.y[idx],
                            {n: c[idx] for n, c in self.columns.items()},
                            [self.meta[i] for i in idx.tolist()], sources=self.sources)

    def seeds(self) -> list:
        return sorted(set(self.columns["seed"].tolist())) if "seed" in self.columns else []

    def split_by_seed(self, holdout_frac: float = 0.1, holdout_seeds: Optional[Iterable[str]] = None,
                      salt: str = "v-holdout") -> tuple:
        """(train, holdout): every row of a seed lands on the same side.  Explicit
        ``holdout_seeds`` win over the hash rule."""
        seeds = self.columns["seed"]
        if holdout_seeds is not None:
            hs = set(str(s) for s in holdout_seeds)
            mask = np.asarray([str(s) in hs for s in seeds.tolist()], dtype=bool)
        else:
            cache = {}
            mask = np.asarray([cache.setdefault(s, seed_in_holdout(s, holdout_frac, salt))
                               for s in seeds.tolist()], dtype=bool)
        return self.subset(np.nonzero(~mask)[0]), self.subset(np.nonzero(mask)[0])

    def batches(self, batch_size: int, rng: Optional[np.random.Generator] = None,
                shuffle: bool = True, drop_last: bool = False) -> Iterator[tuple]:
        """Yield ``(obs_batch: {key: (B, ...)}, y: (B,), idx: (B,))`` for one epoch."""
        n = len(self)
        order = np.arange(n)
        if shuffle:
            (rng or np.random.default_rng()).shuffle(order)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            if drop_last and len(idx) < batch_size:
                break
            yield {k: v[idx] for k, v in self.obs.items()}, self.y[idx], idx

    def summary(self) -> dict:
        """Label statistics by state kind and by ante (mean / sd / n / mean CI half-width)."""
        out = {"n": len(self), "n_seeds": len(self.seeds()),
               "y_mean": float(self.y.mean()) if len(self) else float("nan"),
               "y_sd": float(self.y.std()) if len(self) else float("nan"),
               "by_kind": {}, "by_ante": {}}
        if not len(self):
            return out
        ci = self.columns.get("ci")
        trunc = self.columns.get("trunc_frac")
        for name, col in (("by_kind", self.columns.get("kind")), ("by_ante", self.columns.get("ante"))):
            if col is None:
                continue
            for v in sorted(set(col.tolist())):
                m = col == v
                d = {"n": int(m.sum()), "y_mean": float(self.y[m].mean()), "y_sd": float(self.y[m].std())}
                if ci is not None:
                    d["ci_mean"] = float(np.nanmean(ci[m]))
                if trunc is not None:
                    d["trunc_frac"] = float(trunc[m].mean())
                out[name][str(v)] = d
        if ci is not None:
            out["ci_mean"] = float(np.nanmean(ci))
        if trunc is not None:
            out["trunc_frac"] = float(trunc.mean())
        return out
