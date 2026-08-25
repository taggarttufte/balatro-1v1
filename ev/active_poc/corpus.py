"""corpus.py — base corpus, holdout, and the training recipe shared by every arm.

The evaluation holdout is the STANDARD seed-hash holdout (``dataset.seed_in_holdout(seed,
0.1)``) of the existing 51k ``labels_full`` corpus — 5,160 rows / 215 seeds.  It is built
once here and is never trained on by any arm:

* the base corpus is subsampled from the NON-holdout side only;
* candidate-pool seeds are filtered through the same rule before they are ever labelled
  (``fresh_seeds``), so no arm can pick up a holdout seed.

The base corpus is a seed-hash subsample (whole seeds, so no seed straddles base/holdout) of
the non-holdout side down to ~12k rows, which makes a ~2.4k arm addition a visible ~20%.
"""
from __future__ import annotations

import random
import string
from typing import Iterable, Optional

import numpy as np

from dataset import LabelDataset, META_COLUMNS, seed_in_holdout

__all__ = [
    "STANDARD_HOLDOUT_FRAC", "BASE_SALT", "DEFAULT_BASE_FRAC", "load_corpus",
    "subsample_by_seed_hash", "build_base_and_holdout", "concat", "fresh_seeds",
    "select_rows_by_state", "kind_counts", "summarise", "canonical_seed", "drop_holdout_seeds",
]

# 45,864 non-holdout rows x 0.262 ~ 12k, so a ~2.4k arm addition is ~20% of the base.
DEFAULT_BASE_FRAC = 0.262

STANDARD_HOLDOUT_FRAC = 0.1
BASE_SALT = "w-active-base"          # a salt of its own: independent of the holdout rule


def load_corpus(shards, holdout_frac: float = STANDARD_HOLDOUT_FRAC) -> tuple:
    """``(train_side, holdout)`` of the existing corpus under the STANDARD hash rule."""
    full = LabelDataset.load(shards)
    if not len(full):
        raise FileNotFoundError(f"no label rows under {shards}")
    return full.split_by_seed(holdout_frac)


def subsample_by_seed_hash(ds: LabelDataset, frac: float, salt: str = BASE_SALT) -> LabelDataset:
    """Keep whole seeds whose ``sha1(salt:seed)`` falls in ``[0, frac)``."""
    seeds = ds.columns["seed"].tolist()
    cache: dict = {}
    mask = np.asarray([cache.setdefault(s, seed_in_holdout(s, frac, salt)) for s in seeds], dtype=bool)
    return ds.subset(np.nonzero(mask)[0])


def build_base_and_holdout(shards, *, base_frac: float = DEFAULT_BASE_FRAC,
                           holdout_frac: float = STANDARD_HOLDOUT_FRAC) -> tuple:
    """``(base, holdout)`` — deterministic, so every stage rebuilds the identical split from
    the same shards without carrying an index around.  The holdout is the FULL standard
    holdout (never subsampled); only the training side is thinned to the base corpus."""
    train_side, holdout = load_corpus(shards, holdout_frac)
    base = subsample_by_seed_hash(train_side, base_frac)
    return base, holdout


def concat(*parts: LabelDataset) -> LabelDataset:
    """Concatenate datasets that share obs keys (arms = base + the arm's new rows)."""
    parts = [p for p in parts if len(p)]
    if not parts:
        raise ValueError("concat: nothing to concatenate")
    keys = list(parts[0].obs.keys())
    for p in parts[1:]:
        if list(p.obs.keys()) != keys:
            raise ValueError(f"obs keys differ: {list(p.obs.keys())} != {keys}")
    obs = {k: np.concatenate([p.obs[k] for p in parts]) for k in keys}
    y = np.concatenate([p.y for p in parts]).astype(np.float32)
    cols = {n: np.concatenate([p.columns[n] for p in parts]) for n in META_COLUMNS
            if all(n in p.columns for p in parts)}
    meta = [m for p in parts for m in p.meta]
    srcs = [s for p in parts for s in p.sources]
    return LabelDataset(obs, y, cols, meta, sources=srcs)


def canonical_seed(s: str) -> str:
    """The seed string the ENGINE will report (``game.seed_str``), which is what lands in a
    shard's ``seed`` column and therefore what the holdout hash rule is applied to.

    Balatro's seed alphabet has no zero: ``normalize_seed`` maps ``'0' -> 'O'``.  A holdout
    test on the RAW string is therefore wrong for any seed containing a zero — the bug this
    function exists to prevent (see NOTES.md §Gotchas)."""
    from balatro_sim import game_keys as _gk
    return _gk.normalize_seed(str(s))


def drop_holdout_seeds(ds: LabelDataset, holdout_frac: float = STANDARD_HOLDOUT_FRAC) -> tuple:
    """``(clean, dropped_seeds)`` — remove any row whose seed is in the evaluation holdout."""
    seeds = ds.columns["seed"].tolist()
    cache: dict = {}
    bad = np.asarray([cache.setdefault(s, seed_in_holdout(s, holdout_frac)) for s in seeds], dtype=bool)
    return ds.subset(np.nonzero(~bad)[0]), sorted({s for s, b in zip(seeds, bad.tolist()) if b})


def fresh_seeds(n: int, *, rng_seed: int, exclude: Iterable[str] = (),
                holdout_frac: float = STANDARD_HOLDOUT_FRAC) -> list:
    """``n`` new 8-char seeds that are neither in ``exclude`` (the existing corpus) nor in the
    standard evaluation holdout — the guarantee that no arm ever trains on holdout data.

    The holdout test is applied to the CANONICAL seed (``canonical_seed``), because that is
    the string the engine records and the trainer splits on."""
    rng = random.Random(rng_seed)
    alphabet = string.ascii_uppercase + string.digits
    seen = set(str(s) for s in exclude) | {canonical_seed(s) for s in exclude}
    out: list = []
    while len(out) < n:
        s = canonical_seed("".join(rng.choice(alphabet) for _ in range(8)))
        if s in seen or seed_in_holdout(s, holdout_frac):
            continue
        seen.add(s)
        out.append(s)
    return out


def select_rows_by_state(ds: LabelDataset, states: Iterable[tuple]) -> LabelDataset:
    """Rows whose ``(seed, step)`` is in ``states`` (both perspectives of each state)."""
    want = {(str(a), int(b)) for a, b in states}
    seeds = ds.columns["seed"].tolist()
    steps = ds.columns["step"].tolist()
    mask = np.asarray([(str(s), int(t)) in want for s, t in zip(seeds, steps)], dtype=bool)
    return ds.subset(np.nonzero(mask)[0])


def kind_counts(ds: LabelDataset) -> dict:
    if "kind" not in ds.columns:
        return {}
    ks = ds.columns["kind"].tolist()
    return {k: int(ks.count(k)) for k in sorted(set(ks))}


def summarise(ds: LabelDataset, name: str) -> dict:
    """Compact description of a dataset for the results JSON."""
    ci = ds.columns.get("ci")
    return {"name": name, "n_rows": len(ds), "n_seeds": len(ds.seeds()),
            "n_states": len(ds) // 2,
            "y_mean": float(ds.y.mean()) if len(ds) else float("nan"),
            "y_sd": float(ds.y.std()) if len(ds) else float("nan"),
            "ci_mean": float(np.nanmean(ci)) if ci is not None and len(ci) else float("nan"),
            "by_kind": kind_counts(ds)}
