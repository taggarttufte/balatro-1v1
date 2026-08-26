"""
train_v.py — regression of V = SetValueNet on label shards (Phase 5 rev 2, W5),
extended with lever (b) — a within-state pairwise ranking loss on W-PAIRS's pair shards
(Phase 5 rev 2, W-RANK; see ``ev/RANK_NOTES.md``).

    python ev/train_v.py --shards ev/runs/labels_a/shards --run-dir ev/runs/v1 \
        --max-steps 20000 --device cuda
    touch ev/runs/v1/PAUSE            # pause between steps; resume with --resume latest
    python ev/train_v.py --resume ev/runs/v1/latest.pt --max-steps 40000

    # + lever (b): add pair shards, tune lam_rank/tau
    python ev/train_v.py --shards ... --pair-shards ev/runs/pairs_s1/shards \
        --lam-rank 1.0 --tau 0.05 --run-dir ev/runs/v2 --max-steps 20000

Loss: BCE on logits vs the SOFT label y (P(win) from rollouts) PLUS, when ``--pair-shards``
is given, ``lam_rank`` times a confidence-weighted pairwise logistic term on
``(V(obs_a) - V(obs_b)) / tau`` vs the paired outcome (RANK_NOTES §1) — ONE forward pass
over ``cat([obs_a, obs_b])`` through the shared encoder, so gradients from both branches hit
the same weights.  AdamW, cosine (default) or flat LR with warmup, batch 256.

Held-out-by-SEED evaluation every ``--eval-every`` steps (same ``dataset.seed_in_holdout``
hash rule applied to BOTH absolute rows and pairs, by their respective ``seed`` field): BCE,
Brier, AUC (labels binarised at 0.5), and a 10-bin reliability curve (+ ECE, with a
configurable guardrail), each against the constant predictor (the training-set mean) and
against the label-noise floor (the Brier a perfect V would score against noisy rollout
labels, ``mean((ci/1.96)^2)``); when pairs are present, held-out PAIR ACCURACY on resolved
pairs (``|delta| > delta_ci``), broken out by ``pair_source`` and ``state_kind``.

Checkpoints: ``latest.pt`` + ``ckpt_<step>.pt`` (pruned to ``--keep``), W1's
``value_net.save_checkpoint`` format with the trainer state in ``extra["trainer"]``
(optimizer moments, step / epoch / batch cursor for BOTH the absolute and pair streams,
numpy + torch + python RNG, config, eval history) — a resume is a CONTINUATION: each
epoch's permutation is a function of (seed, epoch) and the cursor says where in it we were,
so the next batch (absolute AND pair) is the one the interrupted run would have drawn.
``test_train_v.py`` / ``test_train_v_pairs.py`` pin the round trip bit-exact.

Run dir: ``train.jsonl`` (config / eval / checkpoint / summary records), console one line
per eval, ``PAUSE`` file honoured between steps (and Ctrl+C), ``.DONE`` sentinel written on
natural completion (max steps / epochs reached), never on a pause.

Model kinds: ``set_value_net`` (W1, the real thing; obs from ``encoder="v2"`` shards) and
``dummy`` (a 16-scalar MLP for the plumbing tests; ``labels.make_encoder("dummy")`` shards).

Auxiliary heads (Phase 5 rev 2, W-AUX; see ``ev/AUX_NOTES.md``): ``--aux-heads all`` attaches
small heads (linear, or one hidden layer) to the SHARED TRUNK and adds ``sum_i w_i * aux_i``
to the loss — BCE for the binary targets, MSE on the spec's transform (log1p) for money /
score / counts.  Targets come from ``meta["aux"]`` on absolute rows and ``rec["aux"]["a"|"b"]``
on pairs (``ev/aux_targets.py`` records them during the rollouts the label / pair workers
already run); a row without them is MASKED, so the old 51k corpus trains with aux muted.
Per-head held-out metrics are reported every eval.  **With no ``--aux-heads`` the trainer is
bit-identical to the pre-W-AUX one** — no heads are constructed, no extra forward pass runs,
and play-time inference / loading old checkpoints is unaffected either way.

Shard filtering by ``player_fingerprint``: absolute shards do not carry a typed column for
it (dataset.py's ``META_COLUMNS`` is frozen — not this workstream's file to change), so
filtering reads it out of each row's free-form ``meta`` dict, defaulting to "no fingerprint
recorded" for the old 51k corpus.  ``--absolute-fingerprint-mode {any,new_only}`` +
``--new-fingerprint <str>`` control it (RANK_NOTES §3); pairs (which the frozen schema says
always carry the field) filter via ``--pair-fingerprint-allow``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import _bootstrap  # noqa: F401
import aux_targets as AX
from dataset import LabelDataset, list_shards, seed_in_holdout

__all__ = [
    "TrainVConfig", "VTrainer", "evaluate", "reliability_curve", "auc_score", "build_model",
    "DummyValueNet", "PAUSE_FILE", "DONE_FILE", "main",
    # lever (b) — pairs
    "PairRow", "PairShard", "PairDataset", "save_pair_shard", "load_pair_shard_npz",
    "load_pair_records_json", "list_pair_shards", "evaluate_pairs",
    "filter_by_fingerprint", "filter_pairs_by_fingerprint",
    "PAIR_COLUMNS", "PAIR_SHARD_VERSION",
    # W-AUX — auxiliary heads
    "resolve_aux_specs", "aux_arrays", "aux_arrays_from_metas", "aux_arrays_from_pairs",
    "AuxData", "evaluate_aux",
]

PAUSE_FILE = "PAUSE"
DONE_FILE = ".DONE"
TRAINER_STATE_VERSION = 1


# ── config ────────────────────────────────────────────────────────────────────────

@dataclass
class TrainVConfig:
    shards: list = field(default_factory=list)
    run_dir: str = "ev/runs/v_default"
    model: str = "set_value_net"        # | "dummy"
    net_cfg: dict = field(default_factory=dict)   # ValueNetConfig overrides
    holdout_frac: float = 0.1
    holdout_seeds: Optional[list] = None
    batch_size: int = 256
    lr: float = 3e-4
    min_lr_frac: float = 0.05
    weight_decay: float = 1e-4
    lr_schedule: str = "cosine"         # | "flat"
    warmup_steps: int = 200
    max_steps: int = 10_000
    max_epochs: Optional[int] = None
    minutes: Optional[float] = None
    clip_grad: float = 1.0
    eval_every: int = 500
    eval_batch_size: int = 1024
    checkpoint_every: int = 1000
    keep: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0
    label_clip: float = 0.0             # clip y into [c, 1-c] (0 = off)
    torch_threads: int = 4              # CPU threads (Tagg shares the box)

    # ── lever (b): pairwise ranking loss (RANK_NOTES §1) ──
    pair_shards: list = field(default_factory=list)   # dirs/globs/files, W-PAIRS's frozen schema
    lam_rank: float = 1.0               # loss = bce + lam_rank * pair_loss
    tau: float = 0.05                   # (V(a)-V(b))/tau before the pairwise logistic
    pair_weight_cap: float = 4.0        # cap on |delta|/delta_ci as the per-pair loss weight
    pair_batch_size: int = 128
    ece_guardrail: float = 0.05         # log a WARNING if holdout ECE exceeds this

    # ── player_fingerprint filtering (RANK_NOTES §3) ──
    absolute_fingerprint_mode: str = "any"   # "any" | "new_only"
    new_fingerprint: Optional[str] = None    # required if absolute_fingerprint_mode="new_only"
    pair_fingerprint_allow: Optional[list] = None   # None = allow every fingerprint found

    # ── W-AUX: auxiliary prediction heads (AUX_NOTES §3) ──
    aux_heads: list = field(default_factory=list)  # [] = OFF (bit-identical to pre-W-AUX);
                                                   # ["all"] = every aux_targets.AUX_SPECS head
    aux_weight: float = 0.1             # default per-head loss weight (brief §6b.2)
    aux_weights: dict = field(default_factory=dict)   # per-head overrides {name: w}
    aux_hidden: int = 0                 # 0 = linear head; > 0 = ONE hidden layer that wide
    aux_on_pairs: bool = True           # also apply the aux loss to both pair branches
    aux_standardize: bool = True        # z-score the REGRESSION targets on the train split
                                        # (AUX_NOTES §3.3: without it a log1p target's sd is
                                        #  ~0.08, so at w=0.1 its gradient is ~30x below the
                                        #  BCE term's and the head effectively never trains)

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainVConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


# ── models ────────────────────────────────────────────────────────────────────────

class DummyValueNet(nn.Module):
    """16 scalars -> logit.  The plumbing stand-in (tests run in seconds on CPU).

    W-AUX: ``aux_heads`` mirrors ``SetValueNet``'s contract (heads off the shared trunk,
    built LAST so that with none configured the module order and every init RNG draw are
    exactly what they were before)."""
    KIND = "dummy"

    def __init__(self, d_in: int = 16, hidden: int = 64, aux_heads: Optional[dict] = None,
                 aux_hidden: int = 0):
        super().__init__()
        self.cfg = {"d_in": d_in, "hidden": hidden}
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden),
                                 nn.ReLU(), nn.Linear(hidden, 1))
        if aux_heads:
            self.cfg = {"d_in": d_in, "hidden": hidden, "aux_heads": dict(aux_heads),
                        "aux_hidden": int(aux_hidden)}
        self.aux_heads = nn.ModuleDict()
        for name, dim in sorted((aux_heads or {}).items()):
            self.aux_heads[name] = (
                nn.Sequential(nn.Linear(hidden, aux_hidden), nn.ReLU(), nn.Linear(aux_hidden, int(dim)))
                if aux_hidden and aux_hidden > 0 else nn.Linear(hidden, int(dim)))

    def forward(self, batch: dict) -> torch.Tensor:
        return self.net(batch["x"]).squeeze(-1)

    def encode(self, batch: dict) -> torch.Tensor:
        """The trunk: everything but the final value Linear."""
        return self.net[:-1](batch["x"])

    def aux_head_names(self) -> list:
        return list(self.aux_heads.keys())

    def forward_with_aux(self, batch: dict) -> tuple:
        trunk = self.encode(batch)
        return self.net[-1](trunk).squeeze(-1), {n: h(trunk) for n, h in self.aux_heads.items()}

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(kind: str, net_cfg: Optional[dict] = None, device="cpu"):
    """``(net, encoder_or_None)``: the real net needs W1's encoder for its checkpoint.

    ``net_cfg`` may carry W-AUX's ``aux_heads`` / ``aux_hidden`` (both model kinds accept
    them; absent or empty = no heads at all, no parameters, no behaviour change)."""
    if kind == "dummy":
        return DummyValueNet(**(net_cfg or {})).to(device), None
    if kind == "set_value_net":
        from mcts.encoder_v2 import SetEncoderV2
        from mcts.value_net import SetValueNet, ValueNetConfig
        net = SetValueNet(ValueNetConfig.from_dict(dict(net_cfg or {}))).to(device)
        return net, SetEncoderV2()
    raise ValueError(f"unknown model kind {kind!r}")


def to_batch(obs: dict, device) -> dict:
    """Stacked numpy obs (B, ...) per key -> tensors on ``device`` (the same tensors W1's
    ``collate`` builds from a per-row list)."""
    dev = torch.device(device)
    return {k: torch.from_numpy(np.ascontiguousarray(v)).to(dev, non_blocking=(dev.type == "cuda"))
            for k, v in obs.items()}


# ── pairs (lever b): W-PAIRS's frozen shard schema (PHASE5_V2_BRIEF §5.3) ──────────
#
# One pair record: {"kind":"pair", "seed","step","actor","state_kind","ante",
# "player_fingerprint", "pair_source", "action_a","action_b", "n_worlds",
# "outcomes_a":[...], "outcomes_b":[...], "delta","delta_ci", "obs_a","obs_b", "meta":{...}}.
#
# The brief freezes the FIELD NAMES, not a file format, and says obs_a/obs_b use "the same
# storage as labels" — i.e. dataset.py's per-key stacked-array convention.  W-RANK does not
# own dataset.py this round (ground rule §2: don't touch files owned by another
# workstream / stop at the interface), and W-PAIRS is building the producer concurrently, so
# rather than risk a concurrent edit collision on dataset.py this module carries its own
# small mirror of dataset.py's Shard/LabelDataset machinery, specialised to two obs blocks
# per row (see RANK_NOTES §2 for the full rationale + the exact on-disk layout chosen).
#
# Two loaders are provided so this trainer works against W-PAIRS's ACTUAL output whichever
# way it lands:
#   * ``.npz`` pair shards (this module's own convention: ``obs_a__<key>`` / ``obs_b__<key>``
#     stacked arrays, typed columns for the scalar fields, everything else — action_a/
#     action_b/outcomes_a/outcomes_b/meta/kind — folded into one ``extra_json`` blob per row,
#     mirroring dataset.py's ``meta_json``).  This is the primary path and what W-RANK's own
#     synthesized test fixtures use.
#   * literal JSON / JSONL records matching the schema text field-for-field (obs_a/obs_b as
#     nested lists, cast to float32 on load — safe for both the numeric and the categorical
#     encoder-v2 fields, since ``value_net._ix`` widens anything non-long to long before an
#     embedding lookup).  A compatibility path, not the expected common case.

PAIR_SHARD_VERSION = 1
# NOTE: "delta"/"delta_ci" are NOT typed columns here (same precedent as dataset.py keeping
# `y` outside `META_COLUMNS`) — they get their own dedicated (N,) arrays (``PairRow.delta``/
# ``.delta_ci``, ``PairDataset.delta``/``.delta_ci``) since every consumer needs them as
# plain floats, not looked up by name through ``.fields``/``.columns``.
PAIR_COLUMNS = ("seed", "step", "actor", "state_kind", "ante", "player_fingerprint",
               "pair_source", "n_worlds")
_PAIR_COLUMN_DTYPES = {
    "seed": "U32", "step": np.int32, "actor": np.int8, "state_kind": "U16", "ante": np.int16,
    "player_fingerprint": "U64", "pair_source": "U32", "n_worlds": np.int16,
}
_PAIR_COLUMN_DEFAULTS = {"seed": "", "step": -1, "actor": -1, "state_kind": "", "ante": -1,
                         "player_fingerprint": "", "pair_source": "", "n_worlds": 0}
_PAIR_ORDER_SALT = 991_301   # decorrelates the pair epoch permutation from the absolute one


@dataclass
class PairRow:
    obs_a: dict
    obs_b: dict
    delta: float
    delta_ci: float
    fields: dict = field(default_factory=dict)   # PAIR_COLUMNS values for this row
    extra: dict = field(default_factory=dict)    # action_a/action_b/outcomes_a/outcomes_b/meta


@dataclass
class PairShard:
    obs_a: dict
    obs_b: dict
    delta: np.ndarray
    delta_ci: np.ndarray
    columns: dict
    extra: list
    path: Optional[str] = None

    def __len__(self) -> int:
        return int(self.delta.shape[0])


def _pair_columns_from_fields(fields_list) -> dict:
    cols = {}
    for name in PAIR_COLUMNS:
        dt = _PAIR_COLUMN_DTYPES[name]
        vals = [f.get(name, _PAIR_COLUMN_DEFAULTS[name]) for f in fields_list]
        cols[name] = np.asarray(vals, dtype=dt)
    return cols


def save_pair_shard(path, rows) -> Path:
    """Write ``rows`` (``PairRow``s) as one compressed ``.npz``.  Empty ``rows`` raises.
    The blob column is named ``pair_json`` (verified against W-PAIRS's actual
    ``pairs.save_pair_shard`` once it landed — RANK_NOTES §2 note) and holds the FULL frozen
    §5.3 record per row (kind/seed/.../delta/delta_ci/meta), not just the ``extra`` fields —
    same content ``pairs.pair_record`` produces."""
    rows = list(rows)
    if not rows:
        raise ValueError("save_pair_shard: no rows")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].obs_a.keys())
    if list(rows[0].obs_b.keys()) != keys:
        raise ValueError("save_pair_shard: obs_a/obs_b key sets differ")
    payload = {"version": np.asarray(PAIR_SHARD_VERSION), "shard_kind": np.asarray("pair"),
              "obs_keys": np.asarray(keys)}
    for k in keys:
        payload[f"obs_a__{k}"] = np.stack([np.asarray(r.obs_a[k]) for r in rows])
        payload[f"obs_b__{k}"] = np.stack([np.asarray(r.obs_b[k]) for r in rows])
    payload["delta"] = np.asarray([r.delta for r in rows], dtype=np.float32)
    payload["delta_ci"] = np.asarray([r.delta_ci for r in rows], dtype=np.float32)
    payload.update(_pair_columns_from_fields([dict(r.fields) for r in rows]))
    recs = [{"kind": "pair", **dict(r.fields), "delta": r.delta, "delta_ci": r.delta_ci, **dict(r.extra)}
           for r in rows]
    payload["pair_json"] = np.asarray([json.dumps(rc, sort_keys=True, default=str) for rc in recs])
    tmp = path.with_suffix(path.suffix + ".tmp.npz")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **payload)
    os.replace(tmp, path)
    return path


def load_pair_shard_npz(path) -> PairShard:
    """Reads either this module's own shards OR W-PAIRS's actual ``pairs.save_pair_shard``
    output directly — both use the same ``obs_a__<key>``/``obs_b__<key>``/typed-columns/
    JSON-blob layout; the only per-shard-writer difference is the blob column's name, which
    this checks for both (``pair_json`` — W-PAIRS's name — first, ``extra_json`` as this
    module's original name, for shards written before this reconciliation)."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        keys = [str(k) for k in z["obs_keys"]]
        obs_a = {k: z[f"obs_a__{k}"] for k in keys}
        obs_b = {k: z[f"obs_b__{k}"] for k in keys}
        delta = z["delta"].astype(np.float32)
        delta_ci = z["delta_ci"].astype(np.float32)
        columns = {name: z[name] for name in PAIR_COLUMNS if name in z.files}
        blob_key = "pair_json" if "pair_json" in z.files else ("extra_json" if "extra_json" in z.files else None)
        extra = [json.loads(s) for s in z[blob_key]] if blob_key else [{} for _ in range(len(delta))]
    return PairShard(obs_a=obs_a, obs_b=obs_b, delta=delta, delta_ci=delta_ci, columns=columns,
                     extra=extra, path=str(path))


def _pair_row_from_json_record(rec: dict) -> PairRow:
    def _arr(d: dict) -> dict:
        return {k: np.asarray(v, dtype=np.float32) for k, v in d.items()}
    obs_a, obs_b = _arr(rec["obs_a"]), _arr(rec["obs_b"])
    fields = {name: rec.get(name, _PAIR_COLUMN_DEFAULTS[name]) for name in PAIR_COLUMNS}
    extra = {"kind": rec.get("kind", "pair"), "action_a": rec.get("action_a"),
            "action_b": rec.get("action_b"), "outcomes_a": rec.get("outcomes_a"),
            "outcomes_b": rec.get("outcomes_b"), "meta": rec.get("meta", {})}
    return PairRow(obs_a, obs_b, float(rec["delta"]), float(rec["delta_ci"]), fields, extra)


def load_pair_records_json(path) -> list:
    """A ``.json`` (list-of-records or one record) or ``.jsonl`` file -> ``[PairRow, ...]``,
    parsed against the literal frozen field names (the compatibility path)."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        recs = [json.loads(l) for l in text.splitlines() if l.strip()]
    else:
        obj = json.loads(text)
        recs = obj if isinstance(obj, list) else [obj]
    return [_pair_row_from_json_record(r) for r in recs]


def list_pair_shards(spec) -> list:
    """Directory (``*.npz`` preferred, else ``*.jsonl``/``*.json``), a glob, a file, or a
    list/tuple of those."""
    if isinstance(spec, (list, tuple)):
        out = []
        for s in spec:
            out.extend(list_pair_shards(s))
        return out
    p = Path(spec)
    if p.is_dir():
        npz = sorted(q for q in p.glob("*.npz") if not q.name.endswith(".tmp.npz"))
        if npz:
            return npz
        return sorted(list(p.glob("*.jsonl")) + list(p.glob("*.json")))
    if p.exists():
        return [p]
    import glob as _glob
    return sorted(Path(q) for q in _glob.glob(str(spec)) if not q.endswith(".tmp.npz"))


class PairDataset:
    """Many pair shards, concatenated.  ``obs_a[k]``/``obs_b[k]`` are (N, ...); ``delta`` /
    ``delta_ci`` are (N,); ``columns`` are the typed ``PAIR_COLUMNS``; ``extra`` the
    per-row dicts (action_a/action_b/outcomes_a/outcomes_b/meta)."""

    def __init__(self, obs_a: dict, obs_b: dict, delta: np.ndarray, delta_ci: np.ndarray,
                columns: dict, extra: list, sources=()):
        self.obs_a, self.obs_b = obs_a, obs_b
        self.delta = np.asarray(delta, dtype=np.float32)
        self.delta_ci = np.asarray(delta_ci, dtype=np.float32)
        self.columns = columns
        self.extra = extra
        self.sources = list(sources)

    @classmethod
    def empty(cls) -> "PairDataset":
        return cls({}, {}, np.zeros((0,), np.float32), np.zeros((0,), np.float32),
                   {n: np.zeros((0,), _PAIR_COLUMN_DTYPES[n]) for n in PAIR_COLUMNS}, [])

    @classmethod
    def from_pair_shards(cls, paths) -> "PairDataset":
        loaded = []
        for p in paths:
            p = Path(p)
            if p.suffix == ".npz":
                s = load_pair_shard_npz(p)
                if len(s):
                    loaded.append(s)
            else:
                rows = load_pair_records_json(p)
                if rows:
                    keys = list(rows[0].obs_a.keys())
                    obs_a = {k: np.stack([r.obs_a[k] for r in rows]) for k in keys}
                    obs_b = {k: np.stack([r.obs_b[k] for r in rows]) for k in keys}
                    delta = np.asarray([r.delta for r in rows], dtype=np.float32)
                    delta_ci = np.asarray([r.delta_ci for r in rows], dtype=np.float32)
                    columns = _pair_columns_from_fields([r.fields for r in rows])
                    extra = [r.extra for r in rows]
                    loaded.append(PairShard(obs_a, obs_b, delta, delta_ci, columns, extra, str(p)))
        if not loaded:
            return cls.empty()
        keys = list(loaded[0].obs_a.keys())
        for s in loaded[1:]:
            if list(s.obs_a.keys()) != keys:
                raise ValueError(f"pair shard {s.path} has obs keys {list(s.obs_a.keys())} != {keys}")
        obs_a = {k: np.concatenate([s.obs_a[k] for s in loaded]) for k in keys}
        obs_b = {k: np.concatenate([s.obs_b[k] for s in loaded]) for k in keys}
        delta = np.concatenate([s.delta for s in loaded]).astype(np.float32)
        delta_ci = np.concatenate([s.delta_ci for s in loaded]).astype(np.float32)
        columns = {n: np.concatenate([s.columns[n] for s in loaded]) for n in PAIR_COLUMNS
                  if all(n in s.columns for s in loaded)}
        extra = [e for s in loaded for e in s.extra]
        return cls(obs_a, obs_b, delta, delta_ci, columns, extra, sources=[s.path for s in loaded])

    @classmethod
    def load(cls, spec) -> "PairDataset":
        return cls.from_pair_shards(list_pair_shards(spec))

    def __len__(self) -> int:
        return int(self.delta.shape[0])

    def subset(self, idx) -> "PairDataset":
        idx = np.asarray(idx)
        return PairDataset({k: v[idx] for k, v in self.obs_a.items()},
                           {k: v[idx] for k, v in self.obs_b.items()},
                           self.delta[idx], self.delta_ci[idx],
                           {n: c[idx] for n, c in self.columns.items()},
                           [self.extra[i] for i in idx.tolist()], sources=self.sources)

    def seeds(self) -> list:
        return sorted(set(self.columns["seed"].tolist())) if "seed" in self.columns else []

    def split_by_seed(self, holdout_frac: float = 0.1, holdout_seeds: Optional[Iterable] = None,
                      salt: str = "v-holdout") -> tuple:
        """Same hash rule as ``dataset.LabelDataset.split_by_seed`` (same ``salt``), applied
        to the pair's own ``seed`` field, so a seed's absolute rows AND its pairs land on the
        same side of the split."""
        if len(self) == 0:
            return self, self
        seeds = self.columns["seed"]
        if holdout_seeds is not None:
            hs = set(str(s) for s in holdout_seeds)
            mask = np.asarray([str(s) in hs for s in seeds.tolist()], dtype=bool)
        else:
            cache = {}
            mask = np.asarray([cache.setdefault(s, seed_in_holdout(s, holdout_frac, salt))
                               for s in seeds.tolist()], dtype=bool)
        return self.subset(np.nonzero(~mask)[0]), self.subset(np.nonzero(mask)[0])


# ── player_fingerprint filtering (RANK_NOTES §3) ────────────────────────────────────

def filter_by_fingerprint(ds: LabelDataset, mode: str, new_fingerprint: Optional[str]) -> LabelDataset:
    """Absolute rows: dataset.py's typed columns don't carry ``player_fingerprint`` (not
    this workstream's file), so this reads it out of each row's free-form ``meta`` dict.
    ``mode="any"`` (default): no filtering — the old 51k corpus (no field at all) mixes with
    any new-fingerprint rows, exactly as PHASE5_V2_BRIEF §2 allows for absolute-BCE
    pretraining.  ``mode="new_only"``: keep only rows whose meta ``player_fingerprint``
    equals ``new_fingerprint`` (old rows, which lack the field, are dropped)."""
    if mode == "any" or len(ds) == 0:
        return ds
    if mode != "new_only":
        raise ValueError(f"unknown absolute_fingerprint_mode {mode!r}")
    if not new_fingerprint:
        raise ValueError("absolute_fingerprint_mode='new_only' needs --new-fingerprint")
    keep = np.asarray([m.get("player_fingerprint") == new_fingerprint for m in ds.meta])
    return ds.subset(np.nonzero(keep)[0])


def filter_pairs_by_fingerprint(pds: PairDataset, allow: Optional[list]) -> PairDataset:
    """``allow=None`` (default): keep every fingerprint found.  Otherwise keep only pairs
    whose typed ``player_fingerprint`` column is in ``allow``."""
    if allow is None or len(pds) == 0:
        return pds
    allow_set = set(str(a) for a in allow)
    fp = pds.columns.get("player_fingerprint")
    if fp is None:
        return pds
    keep = np.asarray([str(v) in allow_set for v in fp.tolist()])
    return pds.subset(np.nonzero(keep)[0])


# ── W-AUX: auxiliary targets on the shards (AUX_NOTES §3) ─────────────────────────
#
# The aux dict is ADDITIVE and lives inside data the loaders already parse:
#   * absolute rows -> ``meta["aux"]``      (dataset.py's ``meta_json``, unchanged loader)
#   * pairs         -> ``rec["aux"]["a"|"b"]``  (the frozen record's blob, unchanged layout)
# Nothing about either on-disk format changes, so W-PAIRS's shards and the old 51k corpus
# load exactly as before — a row with no ``aux`` key is simply MASKED (its heads contribute
# nothing to the loss and nothing to the metrics), which is what "old shards train with aux
# muted" means in practice.


@dataclass
class AuxData:
    """Aligned aux targets for one dataset: ``values`` (N, D) already TRANSFORMED by each
    spec, ``mask`` (N, S) presence flags, ``slices`` the per-spec column range."""
    values: np.ndarray
    mask: np.ndarray
    specs: list

    def __len__(self) -> int:
        return int(self.values.shape[0])

    @property
    def slices(self) -> list:
        out, at = [], 0
        for s in self.specs:
            out.append(slice(at, at + s.dim))
            at += s.dim
        return out

    @property
    def any_present(self) -> bool:
        return bool(self.mask.size and self.mask.any())

    def coverage(self) -> dict:
        n = max(int(self.mask.shape[0]), 1)
        return {s.name: float(self.mask[:, j].sum()) / n for j, s in enumerate(self.specs)}

    def subset(self, idx) -> "AuxData":
        idx = np.asarray(idx)
        return AuxData(self.values[idx], self.mask[idx], self.specs)


def resolve_aux_specs(names: Optional[Iterable]) -> list:
    """``[]``/``None`` -> no aux at all.  ``["all"]`` -> every ``AUX_SPECS`` head.  Otherwise
    the named heads, in ``AUX_SPECS`` order (so the column layout never depends on the order
    the flag was typed in)."""
    names = list(names or [])
    if not names:
        return []
    if len(names) == 1 and str(names[0]).lower() in ("all", "*"):
        return list(AX.AUX_SPECS)
    want = {str(n).strip() for n in names if str(n).strip()}
    unknown = want - set(AX.AUX_NAMES)
    if unknown:
        raise ValueError(f"unknown aux head(s) {sorted(unknown)}; known: {list(AX.AUX_NAMES)}")
    return [s for s in AX.AUX_SPECS if s.name in want]


def aux_arrays(aux_dicts: Sequence[Optional[dict]], specs: Sequence) -> AuxData:
    """``[{name: raw value or None}, ...]`` -> ``AuxData`` (transformed values + mask)."""
    specs = list(specs)
    n, S = len(aux_dicts), len(specs)
    D = sum(s.dim for s in specs)
    values = np.zeros((n, D), dtype=np.float32)
    mask = np.zeros((n, S), dtype=bool)
    if not specs:
        return AuxData(values, mask, specs)
    at = 0
    cols = []
    for s in specs:
        cols.append((at, at + s.dim))
        at += s.dim
    for i, d in enumerate(aux_dicts):
        if not d:
            continue
        for j, s in enumerate(specs):
            raw = d.get(s.name)
            if raw is None:
                continue
            # validate the RAW value first: `aggregate` never writes a non-finite one, but a
            # hand-edited / third-party shard might, and a silently-transformed NaN would
            # poison a whole head's gradient.
            try:
                raws = [float(raw)] if s.dim == 1 else [float(x) for x in raw]
            except (TypeError, ValueError):
                continue
            if len(raws) != s.dim or not all(math.isfinite(x) for x in raws):
                continue
            vv = [s.transform(x) for x in raws]
            if not all(math.isfinite(float(x)) for x in vv):
                continue
            lo, hi = cols[j]
            values[i, lo:hi] = vv
            mask[i, j] = True
    return AuxData(values, mask, specs)


def aux_norm_from(aux_list: Sequence[AuxData], specs: Sequence) -> dict:
    """``{column: (mean, sd)}`` for the REGRESSION columns, pooled over the given (train-side
    only) blocks.  Binary heads are left alone — their targets are already 0..1 BCE targets.

    Why standardise at all (AUX_NOTES §3.3): the raw log1p transforms land money at sd ~0.085
    and the count heads at sd ~0.02-0.05, so an unstandardised MSE term is ~30x weaker than
    the main BCE and the head never moves at the brief's default weight of 0.1.  R^2 is
    scale-invariant, so the reported per-head metrics mean the same thing either way."""
    specs = list(specs)
    out: dict = {}
    at = 0
    for j, s in enumerate(specs):
        lo, hi = at, at + s.dim
        at = hi
        if s.is_binary:
            continue
        for c in range(lo, hi):
            xs = [a.values[a.mask[:, j], c] for a in aux_list if len(a) and a.mask[:, j].any()]
            if not xs:
                continue
            v = np.concatenate(xs)
            out[c] = (float(v.mean()), max(float(v.std()), 1e-3))
    return out


def apply_aux_norm(aux: AuxData, norm: dict) -> AuxData:
    """Z-score the columns ``norm`` names, in place on a copy.  Rows whose head is masked are
    untouched (they are zeros that never enter a loss or a metric)."""
    if not norm or not len(aux):
        return aux
    values = aux.values.copy()
    at = 0
    for j, s in enumerate(aux.specs):
        lo, hi = at, at + s.dim
        at = hi
        for c in range(lo, hi):
            if c not in norm:
                continue
            mu, sd = norm[c]
            m = aux.mask[:, j]
            values[m, c] = (values[m, c] - mu) / sd
    return AuxData(values, aux.mask, aux.specs)


def aux_arrays_from_metas(metas: Sequence[dict], specs: Sequence) -> AuxData:
    return aux_arrays([(m or {}).get("aux") for m in metas], specs)


def aux_arrays_from_pairs(extras: Sequence[dict], specs: Sequence, branch: str) -> AuxData:
    """``branch`` is ``"a"`` or ``"b"``; a record with no ``aux`` key masks out."""
    return aux_arrays([((e or {}).get("aux") or {}).get(branch) for e in extras], specs)


def _masked_head_loss(pred: torch.Tensor, target: torch.Tensor, present: torch.Tensor,
                      binary: bool) -> torch.Tensor:
    """Mean loss over the PRESENT rows only, computed without a host sync (no ``.any()``
    branch): rows whose target is missing are multiplied by 0 and excluded from the
    denominator, and an all-missing head contributes exactly 0."""
    per = (F.binary_cross_entropy_with_logits(pred, target, reduction="none") if binary
           else (pred - target) ** 2)
    w = present.to(per.dtype).unsqueeze(-1)
    denom = (w.sum() * per.shape[-1]).clamp_min(1e-6)
    return (per * w).sum() / denom


def _r2(pred: np.ndarray, y: np.ndarray) -> float:
    if len(y) < 2:
        return float("nan")
    sst = float(((y - y.mean()) ** 2).sum())
    sse = float(((y - pred) ** 2).sum())
    return 1.0 - sse / sst if sst > 0 else float("nan")


def _aux_head_metrics(spec, pred: np.ndarray, target: np.ndarray) -> dict:
    """Per-head held-out metrics.  Binary heads: Brier / BCE / accuracy vs the base rate.
    Regression heads: R^2 (the brief's money-head sanity check) / RMSE, on the TRANSFORMED
    scale the head actually regresses."""
    n = int(target.shape[0])
    out = {"n": n, "kind": spec.kind, "dim": spec.dim}
    if n == 0:
        return out
    if spec.is_binary:
        p = 1.0 / (1.0 + np.exp(-np.clip(pred, -30, 30)))
        out["brier"] = float(((p - target) ** 2).mean())
        out["bce"] = _bce(p.ravel(), target.ravel())
        out["base_rate"] = float(target.mean())
        base = np.full_like(target, out["base_rate"])
        out["brier_base"] = float(((base - target) ** 2).mean())
        out["acc@0.5"] = float(((p >= 0.5) == (target >= 0.5)).mean())
        out["p_mean"] = float(p.mean())
    else:
        out["rmse"] = float(np.sqrt(((pred - target) ** 2).mean()))
        out["r2"] = _r2(pred.ravel(), target.ravel())
        if spec.dim > 1:
            out["r2_per_dim"] = [_r2(pred[:, k], target[:, k]) for k in range(spec.dim)]
        out["y_mean"] = float(target.mean())
        out["y_sd"] = float(target.std())
        out["p_mean"] = float(pred.mean())
    return out


@torch.no_grad()
def evaluate_aux(net: nn.Module, obs: dict, aux: AuxData, device, *, batch_size: int = 1024,
                 source: str = "absolute") -> dict:
    """Per-head held-out metrics over an obs block (brief §6b.4: "all existing metrics +
    per-head held-out metrics each eval")."""
    n = len(aux)
    if n == 0 or not aux.specs or not aux.any_present:
        return {"n": 0, "source": source}
    was_training = net.training
    net.eval()
    preds = {s.name: [] for s in aux.specs}
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        batch = to_batch({k: v[start:stop] for k, v in obs.items()}, device)
        _logits, ap = net.forward_with_aux(batch)
        for s in aux.specs:
            preds[s.name].append(ap[s.name].float().cpu().numpy())
    if was_training:
        net.train()
    out = {"n": n, "source": source, "heads": {}, "coverage": aux.coverage()}
    for j, (s, sl) in enumerate(zip(aux.specs, aux.slices)):
        m = aux.mask[:, j]
        p = np.concatenate(preds[s.name])[m]
        t = aux.values[m, sl]
        out["heads"][s.name] = _aux_head_metrics(s, p, t)
    return out


# ── metrics ───────────────────────────────────────────────────────────────────────

def _bce(p: np.ndarray, y: np.ndarray, eps: float = 1e-7) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def auc_score(p: np.ndarray, y_bin: np.ndarray) -> float:
    """Rank AUC (Mann-Whitney), ties averaged.  NaN if one class is missing."""
    pos = y_bin.astype(bool)
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=np.float64)
    sp = p[order]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def reliability_curve(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> dict:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    bins = []
    ece = 0.0
    n = len(p)
    for b in range(n_bins):
        m = idx == b
        c = int(m.sum())
        if c:
            mp_, my_ = float(p[m].mean()), float(y[m].mean())
            ece += c / n * abs(mp_ - my_)
        else:
            mp_, my_ = float("nan"), float("nan")
        bins.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": c, "p_mean": mp_, "y_mean": my_})
    return {"bins": bins, "ece": float(ece)}


def metrics(p: np.ndarray, y: np.ndarray, ci: Optional[np.ndarray] = None, const: Optional[float] = None) -> dict:
    y_bin = y >= 0.5
    out = {
        "n": int(len(y)),
        "bce": _bce(p, y),
        "brier": float(((p - y) ** 2).mean()),
        "auc": auc_score(p, y_bin.astype(np.float32)),
        "reliability": reliability_curve(p, y),
        "p_mean": float(p.mean()), "y_mean": float(y.mean()), "p_sd": float(p.std()),
        "acc@0.5": float(((p >= 0.5) == y_bin).mean()),
    }
    if const is not None:
        pc = np.full_like(y, float(const))
        out["const"] = {"value": float(const), "bce": _bce(pc, y), "brier": float(((pc - y) ** 2).mean()),
                        "acc@0.5": float(((pc >= 0.5) == y_bin).mean())}
    if ci is not None and len(ci):
        sd = np.nan_to_num(ci, nan=0.0) / 1.96
        out["noise_floor_brier"] = float((sd ** 2).mean())
    return out


@torch.no_grad()
def predict(net: nn.Module, ds: LabelDataset, device, batch_size: int = 1024) -> np.ndarray:
    was_training = net.training
    net.eval()
    out = []
    for obs, _y, _idx in ds.batches(batch_size, shuffle=False):
        out.append(net(to_batch(obs, device)).sigmoid().float().cpu().numpy())
    if was_training:
        net.train()
    return np.concatenate(out) if out else np.zeros((0,), np.float32)


def evaluate(net: nn.Module, ds: LabelDataset, device, *, const: Optional[float] = None,
             batch_size: int = 1024) -> dict:
    if len(ds) == 0:
        return {"n": 0}
    p = predict(net, ds, device, batch_size)
    ci = ds.columns.get("ci")
    m = metrics(p, ds.y, ci=ci, const=const)
    if "kind" in ds.columns:
        m["by_kind"] = {}
        kinds = ds.columns["kind"]
        for k in sorted(set(kinds.tolist())):
            sel = kinds == k
            if sel.sum() >= 20:
                m["by_kind"][str(k)] = {"n": int(sel.sum()), "bce": _bce(p[sel], ds.y[sel]),
                                        "brier": float(((p[sel] - ds.y[sel]) ** 2).mean())}
    return m


def _pair_breakdown(col: Optional[np.ndarray], resolved: np.ndarray, correct: np.ndarray,
                    min_n: int = 5) -> dict:
    """Per-value pair-accuracy breakdown (used for both ``pair_source`` and ``state_kind``).
    ``min_n`` is deliberately low next to ``evaluate``'s ``by_kind`` threshold of 20: proof-
    scale pair campaigns (hundreds-low-thousands, PHASE5_V2_BRIEF §5.5) are much smaller than
    the 50k absolute corpus — revisit once the idle-box campaign lands real volumes."""
    if col is None:
        return {}
    out = {}
    for v in sorted(set(col.tolist())):
        sel = col == v
        n = int(sel.sum())
        if n < min_n:
            continue
        r = resolved & sel
        nr = int(r.sum())
        out[str(v)] = {"n": n, "n_resolved": nr, "pair_acc": float(correct[r].mean()) if nr else float("nan")}
    return out


@torch.no_grad()
def evaluate_pairs(net: nn.Module, pds: PairDataset, device, *, tau: float, weight_cap: float,
                   batch_size: int = 1024) -> dict:
    """Held-out pair metrics (RANK_NOTES §1/§2): ``pair_acc`` = fraction of RESOLVED pairs
    (``|delta| > delta_ci`` — 0 excluded from the ~95% CI) where ``sign(V(a)-V(b))`` matches
    ``sign(delta)``; tau-independent by construction (sign of a probability difference is a
    monotonic function of tau).  ``pair_loss`` reports the same confidence-weighted pairwise
    logistic used in training, for visibility (not used to pick ``best``)."""
    if len(pds) == 0:
        return {"n": 0}
    was_training = net.training
    net.eval()
    preds = []
    for start in range(0, len(pds), batch_size):
        stop = min(start + batch_size, len(pds))
        obs_a = {k: v[start:stop] for k, v in pds.obs_a.items()}
        obs_b = {k: v[start:stop] for k, v in pds.obs_b.items()}
        ba, bb = to_batch(obs_a, device), to_batch(obs_b, device)
        cat = {k: torch.cat([ba[k], bb[k]], dim=0) for k in ba}
        logits = net(cat)
        m = stop - start
        va = logits[:m].sigmoid().float().cpu().numpy()
        vb = logits[m:].sigmoid().float().cpu().numpy()
        preds.append(va - vb)
    if was_training:
        net.train()
    pred_delta = np.concatenate(preds) if preds else np.zeros((0,), np.float32)
    delta, ci = pds.delta.astype(np.float64), pds.delta_ci.astype(np.float64)
    resolved = np.abs(delta) > np.nan_to_num(ci, nan=np.inf)
    n_resolved = int(resolved.sum())
    correct = np.sign(pred_delta) == np.sign(delta)
    pair_acc = float(correct[resolved].mean()) if n_resolved else float("nan")
    weight = np.clip(np.abs(delta) / np.clip(ci, 1e-3, None), 0, weight_cap)
    score = np.clip(pred_delta / max(tau, 1e-6), -30, 30)
    p = 1.0 / (1.0 + np.exp(-score))
    target = 0.5 * (np.sign(delta) + 1.0)
    eps = 1e-7
    p_c = np.clip(p, eps, 1 - eps)
    bce = -(target * np.log(p_c) + (1 - target) * np.log(1 - p_c))
    wsum = float(weight.sum())
    pair_loss = float((bce * weight).sum() / wsum) if wsum > 0 else float("nan")
    return {
        "n": int(len(pds)), "n_resolved": n_resolved, "pair_acc": pair_acc, "pair_loss": pair_loss,
        "weight_mean": float(weight.mean()),
        "by_pair_source": _pair_breakdown(pds.columns.get("pair_source"), resolved, correct),
        "by_state_kind": _pair_breakdown(pds.columns.get("state_kind"), resolved, correct),
    }


# ── the trainer ───────────────────────────────────────────────────────────────────

class VTrainer:
    def __init__(self, cfg: TrainVConfig, *, data: Optional[tuple] = None,
                pair_data: Optional[tuple] = None, log=print):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.log = log
        if cfg.torch_threads and cfg.torch_threads > 0:
            torch.set_num_threads(int(cfg.torch_threads))
        self.step = 0
        self.epoch = 0
        self.cursor = 0                    # batches consumed in the current epoch
        self.pair_epoch = 0                # lever (b): same bookkeeping, its own stream
        self.pair_cursor = 0
        self.samples_seen = 0
        self.elapsed_s = 0.0
        self.history: list = []
        self.best: Optional[dict] = None
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        # W-AUX: heads are part of the net's config, so they must be resolved BEFORE the
        # model is built.  With `aux_heads = []` (the default) `net_cfg` is passed through
        # untouched and nothing about construction, parameter order or init RNG changes.
        self.aux_specs = resolve_aux_specs(cfg.aux_heads)
        net_cfg = dict(cfg.net_cfg or {})
        if self.aux_specs:
            net_cfg["aux_heads"] = {s.name: s.dim for s in self.aux_specs}
            net_cfg["aux_hidden"] = int(cfg.aux_hidden)
        self.net, self.encoder = build_model(cfg.model, net_cfg, self.device)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.rng = np.random.default_rng(cfg.seed)
        if data is not None:
            self.train_ds, self.holdout_ds = data
        else:
            self.load_data()
        if pair_data is not None:
            self.train_pairs, self.holdout_pairs = pair_data
        else:
            self.load_pair_data()
        self.const = float(self.train_ds.y.mean()) if len(self.train_ds) else 0.5
        self._order = None
        self._order_epoch = -1
        self._pair_order = None
        self._pair_order_epoch = -1
        self._last_idx = None
        self._last_pair_idx = None
        self._last_pair_aux = None
        self.load_aux_data()

    # ── data ──
    def load_data(self) -> None:
        paths = list_shards(self.cfg.shards)
        if not paths:
            raise FileNotFoundError(f"no shards under {self.cfg.shards}")
        full = LabelDataset.load(paths)
        if self.cfg.label_clip > 0:
            full.y = np.clip(full.y, self.cfg.label_clip, 1 - self.cfg.label_clip).astype(np.float32)
        full = filter_by_fingerprint(full, self.cfg.absolute_fingerprint_mode, self.cfg.new_fingerprint)
        self.train_ds, self.holdout_ds = full.split_by_seed(self.cfg.holdout_frac, self.cfg.holdout_seeds)
        self.log(f"  data: {len(full)} rows from {len(paths)} shards, {len(full.seeds())} seeds -> "
                 f"train {len(self.train_ds)} ({len(self.train_ds.seeds())} seeds) / "
                 f"holdout {len(self.holdout_ds)} ({len(self.holdout_ds.seeds())} seeds)")

    def load_pair_data(self) -> None:
        """Lever (b) pair shards.  No ``--pair-shards`` -> both splits empty and every pair
        code path below is a no-op — training with no pairs configured is bit-identical to
        the pre-lever-(b) trainer (RANK_NOTES §4)."""
        if not self.cfg.pair_shards:
            self.train_pairs, self.holdout_pairs = PairDataset.empty(), PairDataset.empty()
            return
        paths = list_pair_shards(self.cfg.pair_shards)
        if not paths:
            raise FileNotFoundError(f"no pair shards under {self.cfg.pair_shards}")
        full = PairDataset.load(paths)
        full = filter_pairs_by_fingerprint(full, self.cfg.pair_fingerprint_allow)
        self.train_pairs, self.holdout_pairs = full.split_by_seed(self.cfg.holdout_frac, self.cfg.holdout_seeds)
        self.log(f"  pairs: {len(full)} from {len(paths)} shards, {len(full.seeds())} seeds -> "
                 f"train {len(self.train_pairs)} / holdout {len(self.holdout_pairs)}")

    def load_aux_data(self) -> None:
        """Build the aligned aux target/mask arrays for whatever data is loaded (W-AUX).

        Runs AFTER the splits, straight off the already-parsed ``meta`` dicts / pair records
        — no second pass over the shards.  With no heads configured every array is empty and
        every aux code path below is a no-op."""
        self.train_aux = self.holdout_aux = None
        self.train_pair_aux = self.holdout_pair_aux = None
        self.aux_weight_vec = []
        self.aux_norm = {}
        if not self.aux_specs:
            self.aux_on = False
            return
        self.aux_weight_vec = [float(self.cfg.aux_weights.get(s.name, self.cfg.aux_weight))
                               for s in self.aux_specs]
        self.train_aux = aux_arrays_from_metas(self.train_ds.meta, self.aux_specs)
        self.holdout_aux = aux_arrays_from_metas(self.holdout_ds.meta, self.aux_specs)
        if len(self.train_pairs) or len(self.holdout_pairs):
            self.train_pair_aux = (aux_arrays_from_pairs(self.train_pairs.extra, self.aux_specs, "a"),
                                   aux_arrays_from_pairs(self.train_pairs.extra, self.aux_specs, "b"))
            self.holdout_pair_aux = (aux_arrays_from_pairs(self.holdout_pairs.extra, self.aux_specs, "a"),
                                     aux_arrays_from_pairs(self.holdout_pairs.extra, self.aux_specs, "b"))
        if self.cfg.aux_standardize:
            # from the TRAIN side only (same discipline as `const`), then applied everywhere
            train_blocks = [self.train_aux] + list(self.train_pair_aux or ())
            self.aux_norm = aux_norm_from(train_blocks, self.aux_specs)
            self.train_aux = apply_aux_norm(self.train_aux, self.aux_norm)
            self.holdout_aux = apply_aux_norm(self.holdout_aux, self.aux_norm)
            if self.train_pair_aux is not None:
                self.train_pair_aux = tuple(apply_aux_norm(a, self.aux_norm) for a in self.train_pair_aux)
                self.holdout_pair_aux = tuple(apply_aux_norm(a, self.aux_norm)
                                              for a in self.holdout_pair_aux)
        self.aux_on = True
        cov = self.train_aux.coverage() if self.train_aux is not None else {}
        self.log(f"  aux: {len(self.aux_specs)} heads {[s.name for s in self.aux_specs]}, "
                 f"w={self.cfg.aux_weight} hidden={self.cfg.aux_hidden}; "
                 f"train coverage " + ", ".join(f"{k} {v:.2f}" for k, v in cov.items()))

    def restore_aux_norm(self, saved: Optional[dict]) -> None:
        """Re-apply a CHECKPOINT's standardisation instead of the one just recomputed, so a
        resume fits the heads to exactly the scale they were trained on even if the data
        slice moved.  A no-op when aux is off, when the checkpoint predates it, or when the
        two agree (the normal resume, which stays bit-exact)."""
        if not self.aux_on or not saved:
            return
        norm = {int(k): (float(v[0]), float(v[1])) for k, v in saved.items()}
        if norm == self.aux_norm:
            return
        # undo the freshly-computed scaling, then apply the saved one
        self.load_aux_data_raw()
        self.aux_norm = norm
        self.train_aux = apply_aux_norm(self.train_aux, norm)
        self.holdout_aux = apply_aux_norm(self.holdout_aux, norm)
        if self.train_pair_aux is not None:
            self.train_pair_aux = tuple(apply_aux_norm(a, norm) for a in self.train_pair_aux)
            self.holdout_pair_aux = tuple(apply_aux_norm(a, norm) for a in self.holdout_pair_aux)

    def load_aux_data_raw(self) -> None:
        """The un-standardised arrays (used by ``restore_aux_norm``)."""
        self.train_aux = aux_arrays_from_metas(self.train_ds.meta, self.aux_specs)
        self.holdout_aux = aux_arrays_from_metas(self.holdout_ds.meta, self.aux_specs)
        if len(self.train_pairs) or len(self.holdout_pairs):
            self.train_pair_aux = (aux_arrays_from_pairs(self.train_pairs.extra, self.aux_specs, "a"),
                                   aux_arrays_from_pairs(self.train_pairs.extra, self.aux_specs, "b"))
            self.holdout_pair_aux = (aux_arrays_from_pairs(self.holdout_pairs.extra, self.aux_specs, "a"),
                                     aux_arrays_from_pairs(self.holdout_pairs.extra, self.aux_specs, "b"))

    @property
    def aux_pairs_on(self) -> bool:
        return bool(self.aux_on and self.cfg.aux_on_pairs and self.train_pair_aux is not None
                    and (self.train_pair_aux[0].any_present or self.train_pair_aux[1].any_present))

    # ── schedule ──
    def lr_at(self, step: int) -> float:
        cfg = self.cfg
        if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
            return cfg.lr * (step + 1) / cfg.warmup_steps
        if cfg.lr_schedule == "flat":
            return cfg.lr
        total = max(cfg.max_steps - cfg.warmup_steps, 1)
        t = min(max(step - cfg.warmup_steps, 0) / total, 1.0)
        return cfg.lr * (cfg.min_lr_frac + (1 - cfg.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * t)))

    # ── batches (resumable permutation) ──
    def _epoch_order(self, epoch: int) -> np.ndarray:
        if self._order_epoch != epoch:
            g = np.random.default_rng([self.cfg.seed, epoch])
            self._order = g.permutation(len(self.train_ds))
            self._order_epoch = epoch
        return self._order

    @property
    def batches_per_epoch(self) -> int:
        return max(1, math.ceil(len(self.train_ds) / self.cfg.batch_size))

    def next_batch(self) -> tuple:
        if self.cursor >= self.batches_per_epoch:
            self.epoch += 1
            self.cursor = 0
        order = self._epoch_order(self.epoch)
        bs = self.cfg.batch_size
        idx = order[self.cursor * bs:(self.cursor + 1) * bs]
        self.cursor += 1
        self._last_idx = idx          # W-AUX: aux targets are looked up by row index
        obs = {k: v[idx] for k, v in self.train_ds.obs.items()}
        return obs, self.train_ds.y[idx]

    # ── pair batches (resumable permutation, own stream — RANK_NOTES §4) ──
    def _pair_epoch_order(self, epoch: int) -> np.ndarray:
        if self._pair_order_epoch != epoch:
            # salted differently from _epoch_order's [seed, epoch] so the two permutation
            # streams never coincide even when both epoch counters line up.
            g = np.random.default_rng([self.cfg.seed, epoch, _PAIR_ORDER_SALT])
            self._pair_order = g.permutation(len(self.train_pairs))
            self._pair_order_epoch = epoch
        return self._pair_order

    @property
    def pair_batches_per_epoch(self) -> int:
        if len(self.train_pairs) == 0:
            return 0
        return max(1, math.ceil(len(self.train_pairs) / self.cfg.pair_batch_size))

    def next_pair_batch(self) -> tuple:
        if self.pair_cursor >= self.pair_batches_per_epoch:
            self.pair_epoch += 1
            self.pair_cursor = 0
        order = self._pair_epoch_order(self.pair_epoch)
        bs = self.cfg.pair_batch_size
        idx = order[self.pair_cursor * bs:(self.pair_cursor + 1) * bs]
        self.pair_cursor += 1
        self._last_pair_idx = idx     # W-AUX: both branches' aux targets, by row index
        obs_a = {k: v[idx] for k, v in self.train_pairs.obs_a.items()}
        obs_b = {k: v[idx] for k, v in self.train_pairs.obs_b.items()}
        return obs_a, obs_b, self.train_pairs.delta[idx], self.train_pairs.delta_ci[idx]

    def _pair_loss(self) -> tuple:
        """One forward pass over ``cat([obs_a, obs_b])`` through the SAME net (the shared
        encoder both branches must route gradient through) -> the confidence-weighted
        pairwise logistic (RANK_NOTES §1).  Unresolved pairs (``|delta|`` small next to
        ``delta_ci``) get weight ~0; ``pair_weight_cap`` bounds how much a handful of
        extremely one-sided pairs can dominate the batch."""
        obs_a, obs_b, delta, delta_ci = self.next_pair_batch()
        ba, bb = to_batch(obs_a, self.device), to_batch(obs_b, self.device)
        cat = {k: torch.cat([ba[k], bb[k]], dim=0) for k in ba}
        n = int(delta.shape[0])
        self._last_pair_aux = None
        if self.aux_pairs_on:
            # ONE trunk pass already covers both branches, so the aux heads ride along for
            # free — brief §6b.1's "both branches of a pair".
            logits, aux_pred = self.net.forward_with_aux(cat)
            idx = self._last_pair_idx
            aux_a, aux_b = self.train_pair_aux
            la, _ = self._aux_loss({k: v[:n] for k, v in aux_pred.items()}, idx, aux_a)
            lb, _ = self._aux_loss({k: v[n:] for k, v in aux_pred.items()}, idx, aux_b)
            # MEAN, not sum: the two branches are the two halves of ONE pair batch, and no
            # other term in the loss counts a batch twice (`_pair_loss` itself is one
            # weighted mean over the batch).  Summing would silently double the pair stream's
            # aux weight relative to the absolute stream's.
            self._last_pair_aux = 0.5 * (la + lb)
        else:
            logits = self.net(cat)
        va, vb = logits[:n].sigmoid(), logits[n:].sigmoid()
        delta_t = torch.from_numpy(np.asarray(delta, dtype=np.float32)).to(self.device)
        ci_t = torch.from_numpy(np.asarray(delta_ci, dtype=np.float32)).to(self.device)
        score = (va - vb) / max(self.cfg.tau, 1e-6)
        target = 0.5 * (torch.sign(delta_t) + 1.0)
        weight = torch.clamp(delta_t.abs() / ci_t.clamp_min(1e-3), max=self.cfg.pair_weight_cap)
        per = F.binary_cross_entropy_with_logits(score, target, reduction="none")
        wsum = weight.sum()
        loss = (per * weight).sum() / wsum if float(wsum.item()) > 0.0 else per.new_zeros(())
        info = {"pair_loss": float(loss.item()), "pair_epoch": self.pair_epoch,
               "pair_cursor": self.pair_cursor, "pair_batch_n": n, "pair_weight_sum": float(wsum.item())}
        return loss, info

    # ── auxiliary heads (W-AUX; AUX_NOTES §3) ──
    def _aux_loss(self, pred: dict, idx, aux: "AuxData") -> tuple:
        """``sum_i w_i * aux_i`` over the configured heads for the rows in ``idx``.

        BCE-with-logits for the binary heads (the stored target is the mean over the shared
        worlds, i.e. a SOFT 0..1 target — the same convention the main BCE already uses for
        ``y``), MSE for the regression heads on the spec's transformed scale.  A head with
        no present target in this batch contributes exactly 0 (``_masked_head_loss``), which
        is how a batch of old, aux-less rows trains with aux muted."""
        total = None
        info = {}
        vals = torch.from_numpy(aux.values[idx]).to(self.device)
        mask = torch.from_numpy(aux.mask[idx]).to(self.device)
        for j, (s, sl) in enumerate(zip(self.aux_specs, aux.slices)):
            head = pred.get(s.name)
            if head is None:
                continue
            li = _masked_head_loss(head, vals[:, sl], mask[:, j], s.is_binary)
            w = self.aux_weight_vec[j]
            total = (w * li) if total is None else total + w * li
            info[f"aux_{s.name}"] = float(li.item())
        if total is None:
            total = torch.zeros((), device=self.device)
        info["aux_loss"] = float(total.item())
        return total, info

    # ── one step ──
    def train_step(self) -> dict:
        obs, y = self.next_batch()
        lr = self.lr_at(self.step)
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        self.net.train()
        batch = to_batch(obs, self.device)
        target = torch.from_numpy(np.asarray(y, dtype=np.float32)).to(self.device)
        aux_info = {}
        if self.aux_on:
            logits, aux_pred = self.net.forward_with_aux(batch)
        else:
            logits = self.net(batch)
        bce_loss = F.binary_cross_entropy_with_logits(logits, target)
        loss = bce_loss
        if self.aux_on:
            aux_loss, aux_info = self._aux_loss(aux_pred, self._last_idx, self.train_aux)
            loss = loss + aux_loss
        pair_info = {}
        if len(self.train_pairs):
            pair_loss, pair_info = self._pair_loss()
            loss = loss + self.cfg.lam_rank * pair_loss
            if self._last_pair_aux is not None:
                loss = loss + self._last_pair_aux
                pair_info["pair_aux_loss"] = float(self._last_pair_aux.item())
                self._last_pair_aux = None
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.clip_grad)) \
            if self.cfg.clip_grad and self.cfg.clip_grad > 0 else float("nan")
        self.optimizer.step()
        self.step += 1
        self.samples_seen += int(len(y))
        rec = {"step": self.step, "loss": float(loss.item()), "bce_loss": float(bce_loss.item()),
              "lr": lr, "grad_norm": gn, "epoch": self.epoch, "cursor": self.cursor}
        rec.update(aux_info)
        rec.update(pair_info)
        return rec

    def eval(self) -> dict:
        m = evaluate(self.net, self.holdout_ds, self.device, const=self.const,
                     batch_size=self.cfg.eval_batch_size)
        m["step"] = self.step
        m["epoch"] = self.epoch
        m["samples_seen"] = self.samples_seen
        if len(self.holdout_pairs):
            m["pairs"] = evaluate_pairs(self.net, self.holdout_pairs, self.device, tau=self.cfg.tau,
                                        weight_cap=self.cfg.pair_weight_cap,
                                        batch_size=self.cfg.eval_batch_size)
        if self.aux_on:
            # per-head held-out metrics every eval (brief §6b.4).  Primary source: the
            # held-out ABSOLUTE rows.  When those carry no aux (a pairs-only corpus), fall
            # back to the held-out pairs' two branches so the heads are never unmeasured.
            am = evaluate_aux(self.net, self.holdout_ds.obs, self.holdout_aux, self.device,
                              batch_size=self.cfg.eval_batch_size, source="absolute")
            if not am.get("n") and self.holdout_pair_aux is not None:
                am = evaluate_aux(self.net, self.holdout_pairs.obs_a, self.holdout_pair_aux[0],
                                  self.device, batch_size=self.cfg.eval_batch_size,
                                  source="pair_branch_a")
            m["aux"] = am
        if m.get("n") and "reliability" in m:
            ece = m["reliability"]["ece"]
            m["ece_guardrail_breached"] = bool(ece > self.cfg.ece_guardrail)
            if m["ece_guardrail_breached"]:
                self.log(f"  [ECE GUARDRAIL] holdout ECE {ece:.4f} > guardrail {self.cfg.ece_guardrail:.4f}")
        self.history.append({k: v for k, v in m.items()
                             if k not in ("reliability", "by_kind", "pairs", "aux")})
        if m.get("n") and (self.best is None or m["bce"] < self.best["bce"]):
            self.best = {"step": self.step, "bce": m["bce"], "brier": m["brier"], "auc": m["auc"]}
        return m

    # ── checkpoints ──
    def state(self) -> dict:
        return {
            "version": TRAINER_STATE_VERSION,
            "step": self.step, "epoch": self.epoch, "cursor": self.cursor,
            "pair_epoch": self.pair_epoch, "pair_cursor": self.pair_cursor,
            "samples_seen": self.samples_seen, "elapsed_s": self.elapsed_s,
            "optimizer": self.optimizer.state_dict(),
            "rng": {"numpy": self.rng.bit_generator.state, "torch": torch.get_rng_state(),
                    "python": random.getstate(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
            "config": self.cfg.as_dict(), "history": self.history, "best": self.best,
            "const": self.const, "n_train": len(self.train_ds), "n_holdout": len(self.holdout_ds),
            "holdout_seeds": self.holdout_ds.seeds(),
            "n_train_pairs": len(self.train_pairs), "n_holdout_pairs": len(self.holdout_pairs),
            "holdout_pair_seeds": self.holdout_pairs.seeds(),
            # W-AUX: the heads live in the net's own state_dict / cfg, so nothing extra has
            # to be persisted for a bit-exact resume — these are provenance only.
            "aux_heads": [s.name for s in self.aux_specs],
            "aux_version": AX.AUX_VERSION,
            "aux_coverage": (self.train_aux.coverage() if self.train_aux is not None else {}),
            # the per-column (mean, sd) the regression targets were z-scored by: derived from
            # the train split (like `const`), persisted so a resume cannot silently re-scale
            # what the heads were fit to.
            "aux_norm": {str(k): list(v) for k, v in (self.aux_norm or {}).items()},
        }

    def save(self, path) -> Path:
        path = Path(path)
        extra = {"trainer": self.state()}
        if self.cfg.model == "set_value_net":
            from mcts.value_net import save_checkpoint
            return save_checkpoint(path, self.net, self.encoder, extra=extra)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"kind": "mp/ev dummy value", "cfg": self.net.cfg,
                   "state_dict": {k: v.detach().cpu() for k, v in self.net.state_dict().items()},
                   "extra": extra}
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)
        return path

    @staticmethod
    def read_checkpoint(path, device="cpu") -> tuple:
        """``(net, encoder_or_None, extra)`` for either model kind."""
        raw = torch.load(Path(path), map_location="cpu", weights_only=False)
        if isinstance(raw, dict) and raw.get("kind") == "mp/ev dummy value":
            net = DummyValueNet(**raw["cfg"])
            net.load_state_dict(raw["state_dict"], strict=True)
            return net.to(device), None, raw["extra"]
        from mcts.value_net import load_checkpoint
        return load_checkpoint(path, device=device)

    @classmethod
    def from_checkpoint(cls, path, *, overrides: Optional[dict] = None, data: Optional[tuple] = None,
                        pair_data: Optional[tuple] = None, log=print) -> "VTrainer":
        net, encoder, extra = cls.read_checkpoint(path)
        ts = extra["trainer"]
        cfg = TrainVConfig.from_dict({**ts["config"], **(overrides or {})})
        tr = cls(cfg, data=data, pair_data=pair_data, log=log)
        _load_net_state(tr.net, net.state_dict(), log)
        tr.net.to(tr.device)
        if encoder is not None:
            tr.encoder = encoder
        _load_optimizer_state(tr.optimizer, ts["optimizer"], log)
        tr.step, tr.epoch, tr.cursor = ts["step"], ts["epoch"], ts["cursor"]
        tr.pair_epoch, tr.pair_cursor = ts.get("pair_epoch", 0), ts.get("pair_cursor", 0)
        tr.samples_seen, tr.elapsed_s = ts["samples_seen"], ts["elapsed_s"]
        tr.history, tr.best = list(ts.get("history", [])), ts.get("best")
        tr.const = float(ts.get("const", tr.const))
        tr.restore_aux_norm(ts.get("aux_norm"))
        r = ts["rng"]
        tr.rng.bit_generator.state = r["numpy"]
        torch.set_rng_state(r["torch"].to(dtype=torch.uint8, device="cpu"))
        random.setstate(_as_tuple(r["python"]))
        if r.get("cuda") is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all([s.to(dtype=torch.uint8, device="cpu") for s in r["cuda"]])
            except (RuntimeError, ValueError):
                pass
        tr.net.train()
        return tr


def _as_tuple(x):
    if isinstance(x, (list, tuple)):
        return tuple(_as_tuple(i) for i in x)
    return x


# ── W-AUX: adding heads to a checkpoint that has none (AUX_NOTES §4) ──────────────
#
# The ONLY case these two helpers do anything other than a plain strict load is a deliberate
# `--resume <old ckpt> --aux-heads all`: the saved weights have no `aux_heads.*` tensors and
# the saved optimizer has fewer parameters than the rebuilt net.  Heads are registered LAST,
# so the first k parameters line up one-for-one and the new tail simply starts fresh (fresh
# init + fresh Adam moments) — exactly the brief's "fresh-init heads when absent".  A normal
# resume (checkpoint and config agree) takes the strict path and stays bit-exact.

def _load_net_state(net: nn.Module, sd: dict, log=print) -> None:
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if not missing and not unexpected:
        return
    bad = [k for k in missing if not k.startswith("aux_heads.")]
    if bad or unexpected:
        raise RuntimeError(f"checkpoint does not match the model: missing {bad}, "
                           f"unexpected {list(unexpected)}")
    log(f"  [aux] {len(missing)} aux-head tensor(s) not in the checkpoint -> fresh init")


def _load_optimizer_state(opt, sd: dict, log=print) -> None:
    cur = opt.state_dict()["param_groups"]
    saved = sd
    if len(cur) == len(sd.get("param_groups", [])) and any(
            len(g_old["params"]) != len(g_new["params"]) for g_old, g_new in zip(sd["param_groups"], cur)):
        import copy
        saved = copy.deepcopy(sd)
        n_old = sum(len(g["params"]) for g in saved["param_groups"])
        for g_old, g_new in zip(saved["param_groups"], cur):
            if len(g_old["params"]) > len(g_new["params"]):
                raise RuntimeError("optimizer state has MORE params than the model "
                                   "(the checkpoint's aux heads are not configured here)")
            g_old["params"] = list(g_new["params"])
        n_new = sum(len(g["params"]) for g in saved["param_groups"])
        log(f"  [aux] optimizer state covers {n_old}/{n_new} params -> the new tail starts fresh")
    opt.load_state_dict(saved)


# ── the run loop ──────────────────────────────────────────────────────────────────

def prune(run_dir: Path, keep: int) -> None:
    files = sorted(run_dir.glob("ckpt_*.pt"), key=lambda p: p.stat().st_mtime)
    for p in files[:-keep] if keep > 0 else []:
        try:
            p.unlink()
        except OSError:
            pass


def fmt_eval(m: dict) -> str:
    c = m.get("const", {})
    s = (f"step {m['step']:>6}  ep {m['epoch']}  holdout n={m['n']}  "
         f"bce {m['bce']:.4f} (const {c.get('bce', float('nan')):.4f})  "
         f"brier {m['brier']:.4f} (const {c.get('brier', float('nan')):.4f}, floor "
         f"{m.get('noise_floor_brier', float('nan')):.4f})  auc {m['auc']:.3f}  "
         f"ece {m['reliability']['ece']:.3f}  p_sd {m['p_sd']:.3f}")
    p = m.get("pairs")
    if p and p.get("n"):
        s += f"  pair_acc {p['pair_acc']:.3f} (resolved {p['n_resolved']}/{p['n']})"
    a = m.get("aux")
    if a and a.get("n"):
        bits = []
        for name, h in a["heads"].items():
            if not h.get("n"):
                continue
            bits.append(f"{name} " + (f"br {h['brier']:.3f}" if h["kind"] == "binary"
                                      else f"r2 {h['r2']:+.2f}"))
        if bits:
            s += "  aux[" + " | ".join(bits) + "]"
    return s


def run(cfg: Optional[TrainVConfig] = None, *, resume: Optional[str] = None,
        overrides: Optional[dict] = None, data: Optional[tuple] = None,
        pair_data: Optional[tuple] = None, log=print, stop_check=None) -> dict:
    """Train (``cfg``) or resume (``resume`` = checkpoint path; ``overrides`` = the config
    keys to change, typically ``max_steps`` / ``minutes`` / ``device``)."""
    if resume:
        ckpt_path = Path(resume)
        if ckpt_path.is_dir():
            ckpt_path = ckpt_path / "latest.pt"
        tr = VTrainer.from_checkpoint(ckpt_path, overrides=overrides or {}, data=data,
                                      pair_data=pair_data, log=log)
        cfg = tr.cfg
        run_dir = Path(cfg.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        pause_path = run_dir / PAUSE_FILE
        done_path = run_dir / DONE_FILE
        if pause_path.exists():
            pause_path.unlink()
        if done_path.exists():
            done_path.unlink()
        log(f"=== resumed {ckpt_path} at step {tr.step} (epoch {tr.epoch}, cursor {tr.cursor}) ===")
    else:
        if cfg is None:
            raise ValueError("run() needs a config or a checkpoint to resume")
        run_dir = Path(cfg.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        pause_path = run_dir / PAUSE_FILE
        done_path = run_dir / DONE_FILE
        tr = VTrainer(cfg, data=data, pair_data=pair_data, log=log)
    log_path = run_dir / "train.jsonl"
    lf = open(log_path, "a", encoding="utf-8")

    def emit(rec: dict) -> None:
        lf.write(json.dumps(rec, default=str) + "\n")
        lf.flush()

    n_params = sum(p.numel() for p in tr.net.parameters())
    emit({"kind": "config", "timestamp": datetime.now().isoformat(timespec="seconds"),
          "config": cfg.as_dict(), "n_params": n_params, "resumed_from": resume,
          "n_train": len(tr.train_ds), "n_holdout": len(tr.holdout_ds),
          "holdout_seeds": tr.holdout_ds.seeds(), "const": tr.const,
          "n_train_pairs": len(tr.train_pairs), "n_holdout_pairs": len(tr.holdout_pairs),
          "holdout_pair_seeds": tr.holdout_pairs.seeds(),
          "aux_heads": [s.name for s in tr.aux_specs],
          "aux_coverage": (tr.train_aux.coverage() if tr.train_aux is not None else {})})
    log(f"=== V training: {cfg.model} ({n_params:,} params) on {cfg.device}, batch {cfg.batch_size}, "
        f"lr {cfg.lr} {cfg.lr_schedule}, max_steps {cfg.max_steps}"
        + (f", pairs lam_rank={cfg.lam_rank} tau={cfg.tau}" if len(tr.train_pairs) else "")
        + (f", aux {len(tr.aux_specs)} heads w={cfg.aux_weight}" if tr.aux_on else "") + " ===")
    log(f"  run dir: {run_dir}   pause: touch {pause_path}")

    signalled = {"hit": False, "why": ""}

    def _on_signal(signum, _frame):
        if not signalled["hit"]:
            signalled["hit"] = True
            signalled["why"] = signal.Signals(signum).name
            log(f"\n[{signalled['why']}] finishing the step in flight, then checkpointing...")

    if stop_check is None:
        for s in (signal.SIGINT, getattr(signal, "SIGTERM", None), getattr(signal, "SIGBREAK", None)):
            if s is not None:
                try:
                    signal.signal(s, _on_signal)
                except (ValueError, OSError):
                    pass

    t_start = time.time()
    base_elapsed = tr.elapsed_s
    deadline = t_start + cfg.minutes * 60 if cfg.minutes else None

    def stop_reason() -> Optional[str]:
        if signalled["hit"]:
            return signalled["why"]
        if stop_check is not None and stop_check():
            return "stop_check"
        if pause_path.exists():
            return "PAUSE"
        if tr.step >= cfg.max_steps:
            return "max_steps"
        if cfg.max_epochs is not None and tr.epoch >= cfg.max_epochs:
            return "max_epochs"
        if deadline is not None and time.time() >= deadline:
            return "deadline"
        return None

    def checkpoint(tag: str) -> Path:
        tr.elapsed_s = base_elapsed + (time.time() - t_start)
        numbered = run_dir / f"ckpt_{tr.step:07d}.pt"
        tr.save(numbered)
        latest = tr.save(run_dir / "latest.pt")
        prune(run_dir, cfg.keep)
        emit({"kind": "checkpoint", "step": tr.step, "tag": tag, "path": str(numbered),
              "bytes": numbered.stat().st_size})
        log(f"  [checkpoint] step {tr.step} -> {numbered.name} ({numbered.stat().st_size / 1e6:.1f} MB, {tag})")
        return latest

    why = None
    last_eval = None
    recent, recent_pair = [], []
    try:
        if tr.step == 0 and len(tr.holdout_ds):
            m = tr.eval()
            emit({"kind": "eval", **m})
            log("  " + fmt_eval(m))
            last_eval = m
        while True:
            why = stop_reason()
            if why is not None:
                break
            rec = tr.train_step()
            recent.append(rec["loss"])
            if "pair_loss" in rec:
                recent_pair.append(rec["pair_loss"])
            if cfg.eval_every and tr.step % cfg.eval_every == 0:
                m = tr.eval()
                m["train_loss"] = float(np.mean(recent)) if recent else float("nan")
                if recent_pair:
                    m["train_pair_loss"] = float(np.mean(recent_pair))
                recent, recent_pair = [], []
                m["elapsed_s"] = base_elapsed + (time.time() - t_start)
                emit({"kind": "eval", **m})
                log("  " + fmt_eval(m) + f"  train {m['train_loss']:.4f}  lr {rec['lr']:.2e}")
                last_eval = m
            if cfg.checkpoint_every and tr.step % cfg.checkpoint_every == 0:
                checkpoint("periodic")
    except KeyboardInterrupt:
        why = "KeyboardInterrupt"
    why = why or stop_reason() or "unknown"
    if last_eval is None or last_eval.get("step") != tr.step:
        if len(tr.holdout_ds):
            m = tr.eval()
            m["train_loss"] = float(np.mean(recent)) if recent else float("nan")
            if recent_pair:
                m["train_pair_loss"] = float(np.mean(recent_pair))
            emit({"kind": "eval", **m})
            log("  " + fmt_eval(m))
            last_eval = m
    final = checkpoint(f"exit:{why}")
    natural = why in ("max_steps", "max_epochs")
    if natural:
        done_path.write_text(f"{why} at step {tr.step} {datetime.now().isoformat(timespec='seconds')}\n")
    summary = {"kind": "summary", "stop_reason": why, "step": tr.step, "epoch": tr.epoch,
               "samples_seen": tr.samples_seen, "elapsed_s": tr.elapsed_s, "best": tr.best,
               "final_eval": {k: v for k, v in (last_eval or {}).items()
                             if k not in ("reliability", "by_kind", "pairs", "aux")},
               "reliability": (last_eval or {}).get("reliability"), "by_kind": (last_eval or {}).get("by_kind"),
               "pairs": (last_eval or {}).get("pairs"), "aux": (last_eval or {}).get("aux"),
               "final_checkpoint": str(final), "done": natural}
    emit(summary)
    lf.close()
    log(f"=== stopped ({why}) at step {tr.step}; latest: {final}"
        + (f"; paused by {pause_path} (delete it or --resume)" if why == "PAUSE" else "")
        + (f"; {done_path.name} written" if natural else ""))
    log(f"resume: python ev/train_v.py --resume {final} --max-steps <N>")
    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────────

_RESUME_KEYS = ("max_steps", "max_epochs", "minutes", "device", "eval_every", "checkpoint_every",
                "keep", "shards", "run_dir", "lr", "lr_schedule", "holdout_frac",
                # lever (b): safe to change on a --resume (like holdout_frac, these change
                # WHAT data is trained on, so they are not covered by the bit-exact guarantee
                # — that guarantee is for a plain --resume with no overrides)
                "pair_shards", "lam_rank", "tau", "pair_weight_cap", "ece_guardrail",
                "absolute_fingerprint_mode", "new_fingerprint", "pair_fingerprint_allow",
                # W-AUX: loss weights are free to change on a resume; `aux_heads` /
                # `aux_hidden` change the GRAPH (that is the "bolt heads onto the keeper"
                # path) and, like holdout_frac, are not covered by the bit-exact guarantee
                "aux_heads", "aux_weight", "aux_weights", "aux_hidden", "aux_on_pairs",
                "aux_standardize")


def build_parser() -> argparse.ArgumentParser:
    """Defaults are None so a ``--resume`` only overrides what was given explicitly."""
    ap = argparse.ArgumentParser(description=__doc__.split(chr(10) * 2)[0])
    ap.add_argument("--shards", nargs="*", default=None, help="shard dirs / globs / files")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--resume", default=None, help="checkpoint path or run dir (uses latest.pt)")
    ap.add_argument("--model", choices=["set_value_net", "dummy"], default=None)
    ap.add_argument("--net-cfg", default=None, help="JSON ValueNetConfig overrides")
    ap.add_argument("--holdout-frac", type=float, default=None)
    ap.add_argument("--holdout-seeds", default=None, help="comma-separated seeds (overrides the hash rule)")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--lr-schedule", choices=["cosine", "flat"], default=None)
    ap.add_argument("--warmup-steps", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--minutes", type=float, default=None)
    ap.add_argument("--clip-grad", type=float, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--checkpoint-every", type=int, default=None)
    ap.add_argument("--keep", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--label-clip", type=float, default=None)
    ap.add_argument("--torch-threads", type=int, default=None)
    # lever (b): pairwise ranking loss
    ap.add_argument("--pair-shards", nargs="*", default=None, help="pair shard dirs / globs / files")
    ap.add_argument("--lam-rank", type=float, default=None)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--pair-weight-cap", type=float, default=None)
    ap.add_argument("--pair-batch-size", type=int, default=None)
    ap.add_argument("--ece-guardrail", type=float, default=None)
    # player_fingerprint filtering
    ap.add_argument("--absolute-fingerprint-mode", choices=["any", "new_only"], default=None)
    ap.add_argument("--new-fingerprint", default=None)
    ap.add_argument("--pair-fingerprint-allow", default=None,
                    help="comma-separated player_fingerprint values (default: allow all found)")
    # W-AUX: auxiliary prediction heads
    ap.add_argument("--aux-heads", default=None,
                    help="'all' or a comma-separated subset of " + ",".join(AX.AUX_NAMES)
                         + " (default: none -> the trainer is bit-identical to pre-W-AUX)")
    ap.add_argument("--aux-weight", type=float, default=None, help="per-head loss weight (default 0.1)")
    ap.add_argument("--aux-weights", default=None, help='JSON per-head overrides, e.g. \'{"blind_cleared":0.2}\'')
    ap.add_argument("--aux-hidden", type=int, default=None, help="0 = linear head; N = one hidden layer")
    ap.add_argument("--aux-on-pairs", type=int, choices=[0, 1], default=None,
                    help="also apply the aux loss to both pair branches (default 1)")
    ap.add_argument("--aux-standardize", type=int, choices=[0, 1], default=None,
                    help="z-score the regression targets on the train split (default 1)")
    return ap


def _given(args) -> dict:
    d = {}
    for k, v in vars(args).items():
        if v is None or k == "resume":
            continue
        if k in ("net_cfg", "aux_weights"):
            v = json.loads(v)
        if k in ("holdout_seeds", "pair_fingerprint_allow", "aux_heads"):
            v = [s for s in v.split(",") if s]
        if k in ("aux_on_pairs", "aux_standardize"):
            v = bool(v)
        d[k] = v
    return d


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    given = _given(args)
    if args.resume:
        p = Path(args.resume)
        if p.is_dir():
            p = p / "latest.pt"
        run(resume=str(p), overrides={k: v for k, v in given.items() if k in _RESUME_KEYS})
        return 0
    if not args.shards or not args.run_dir:
        raise SystemExit("--shards and --run-dir are required (or --resume)")
    run(TrainVConfig.from_dict(given))
    return 0


if __name__ == "__main__":
    sys.exit(main())
