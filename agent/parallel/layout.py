"""
layout.py — the byte layout of one MCTS leaf in a shared-memory arena.

Why shared memory and not a ``multiprocessing.Queue``
-----------------------------------------------------
Measured on the real run's configuration (set encoder, `ItemCaps(16,12,6,8,8)`, a
SELECTING_HAND leaf with 436 legal actions):

    observation   21 arrays,   3 660 bytes
    action block   4 arrays, 122 952 bytes   (act_num 436x20, act_sel 436x16,
                                              act_tgt 436x34, act_type 436)
    ------------------------------------------------------------------
    per leaf                 126 612 bytes   -> reply is 436*4 + 4 = 1 748 bytes

A single process runs ~184 sims/s with W0's heuristic prior, so 16 workers is ~2 900
leaves/s = **~370 MB/s of request traffic** against ~5 MB/s of replies.  Pushing that
through a ``Queue`` means pickling and un-pickling 25 numpy arrays per leaf and copying
every byte through a pipe, twice; a shared-memory arena means the worker writes the arrays
straight into the buffer the evaluator will read (``np.stack`` on the evaluator side is
then the only copy, and it is the copy ``BatchedSetNNPolicy`` already makes today).  The
queue carries only the 4-tuple of offsets per leaf.

Layout of one leaf record, at an 8-byte-aligned offset
-----------------------------------------------------
    [ float32 block ]  obs float arrays (fixed size), then n_actions x act_float_width
    [ int16   block ]  obs int arrays   (fixed size), then n_actions x act_int_width

Key order inside each block is ``sorted()``, so the two sides cannot disagree; every shape
except the action count is fixed by the encoder's caps, so a leaf is described completely
by ``(offset, n_actions)``.  The flat (`v7` / `mlb`) encoder produces a bare ``(D,)`` array
and a bare ``(N, 56)`` array rather than dicts; they are handled as one-key dicts under
``FLAT_OBS_KEY`` / ``FLAT_ACT_KEY`` so there is exactly one layout implementation.

`tests/test_parallel.py` pins ``pack`` -> ``unpack`` round-trips bit-exact for both
encoders and pins the layout against a real ``encode_leaf``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FLAT_OBS_KEY = "_obs"
FLAT_ACT_KEY = "_acts"

#: Every record starts on a multiple of this so the float32 / int16 views inside it are
#: naturally aligned (numpy tolerates unaligned buffers but pays for them).
ALIGN = 8


def _align(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


def as_obs_dict(obs) -> dict:
    """A leaf observation as a dict, whichever encoder produced it."""
    return obs if isinstance(obs, dict) else {FLAT_OBS_KEY: obs}


def as_acts_dict(acts) -> dict:
    return acts if isinstance(acts, dict) else {FLAT_ACT_KEY: acts}


@dataclass(frozen=True)
class _Field:
    key: str
    shape: tuple            # per-leaf shape (action arrays keep their leading axis out)
    count: int              # elements per leaf (action arrays: elements per ACTION)


class LeafLayout:
    """Fixed byte layout for the ``(obs, acts)`` pair a given encoder produces.

    Built from ONE prototype leaf (``LeafLayout.from_prototype(obs, acts)``): the shapes of
    everything except the action count are properties of the encoder, so a single real
    ``encode_leaf`` call at pool start-up describes every leaf the run will ever produce.
    """

    def __init__(self, obs_fields: tuple, act_fields: tuple, flat: bool,
                 int_dtype=np.int16):
        """``obs_fields`` / ``act_fields`` are ``(float_fields, int_fields)`` pairs; use
        :meth:`from_prototype` rather than calling this directly."""
        self.flat = bool(flat)
        self.int_dtype = np.dtype(int_dtype)
        self.obs_f: list = list(obs_fields[0])
        self.obs_i: list = list(obs_fields[1])
        self.act_f: list = list(act_fields[0])
        self.act_i: list = list(act_fields[1])
        self.obs_f_count = sum(f.count for f in self.obs_f)
        self.obs_i_count = sum(f.count for f in self.obs_i)
        self.act_f_width = sum(f.count for f in self.act_f)
        self.act_i_width = sum(f.count for f in self.act_i)

    # ── construction ─────────────────────────────────────────────────────────

    @classmethod
    def from_prototype(cls, obs, acts) -> "LeafLayout":
        flat = not isinstance(obs, dict)
        obs_d = as_obs_dict(obs)
        acts_d = as_acts_dict(acts)

        obs_f, obs_i, int_dtype = [], [], None
        for key in sorted(obs_d):
            arr = np.asarray(obs_d[key])
            field = _Field(key, tuple(arr.shape), int(arr.size))
            if arr.dtype == np.float32:
                obs_f.append(field)
            else:
                int_dtype = cls._pin_int_dtype(int_dtype, arr.dtype, key)
                obs_i.append(field)

        act_f, act_i = [], []
        for key in sorted(acts_d):
            arr = np.asarray(acts_d[key])
            if arr.ndim < 1:                        # pragma: no cover - defensive
                raise ValueError(f"action array {key!r} has no action axis")
            tail = tuple(arr.shape[1:])
            per = int(np.prod(tail)) if tail else 1
            field = _Field(key, tail, per)
            if arr.dtype == np.float32:
                act_f.append(field)
            else:
                int_dtype = cls._pin_int_dtype(int_dtype, arr.dtype, key)
                act_i.append(field)

        return cls((obs_f, obs_i), (act_f, act_i), flat,
                   int_dtype if int_dtype is not None else np.dtype(np.int16))

    @staticmethod
    def _pin_int_dtype(current, dtype, key):
        if current is not None and np.dtype(dtype) != current:
            raise ValueError(
                f"leaf array {key!r} has dtype {dtype}, but this layout already uses "
                f"{current} for its integer block. The arena assumes ONE integer dtype "
                "(encoder_set.CAT_DTYPE); a second one needs a third block here.")
        return np.dtype(dtype)

    # ── sizes ────────────────────────────────────────────────────────────────

    def record_bytes(self, n_actions: int) -> int:
        """Bytes one leaf with ``n_actions`` legal actions occupies, including padding."""
        f = (self.obs_f_count + n_actions * self.act_f_width) * 4
        i = (self.obs_i_count + n_actions * self.act_i_width) * self.int_dtype.itemsize
        return _align(_align(f) + i)

    def reply_floats(self, n_actions: int) -> int:
        """Floats one leaf's reply occupies: the action priors.  Values are stored
        separately (one float per leaf, at the head of the reply arena)."""
        return n_actions

    def describe(self) -> dict:
        return {"flat": self.flat, "int_dtype": str(self.int_dtype),
                "obs_float_keys": [f.key for f in self.obs_f],
                "obs_int_keys": [f.key for f in self.obs_i],
                "act_float_keys": [f.key for f in self.act_f],
                "act_int_keys": [f.key for f in self.act_i],
                "obs_floats": self.obs_f_count, "obs_ints": self.obs_i_count,
                "act_float_width": self.act_f_width, "act_int_width": self.act_i_width}

    # ── pack / unpack ────────────────────────────────────────────────────────

    def pack(self, buf: np.ndarray, offset: int, obs, acts, n_actions: int) -> int:
        """Write one leaf into ``buf`` (a uint8 view of the arena) at ``offset``.

        Returns the number of bytes written (already 8-aligned), so the caller bumps
        ``offset`` by the return value.
        """
        obs_d = as_obs_dict(obs)
        acts_d = as_acts_dict(acts)
        n_f = self.obs_f_count + n_actions * self.act_f_width
        n_i = self.obs_i_count + n_actions * self.act_i_width
        f_bytes = _align(n_f * 4)

        fview = buf[offset:offset + n_f * 4].view(np.float32)
        at = 0
        for field in self.obs_f:
            fview[at:at + field.count] = obs_d[field.key].reshape(-1)
            at += field.count
        for field in self.act_f:
            w = field.count * n_actions
            fview[at:at + w] = acts_d[field.key].reshape(-1)
            at += w

        if n_i:
            isz = self.int_dtype.itemsize
            iview = buf[offset + f_bytes:offset + f_bytes + n_i * isz].view(self.int_dtype)
            at = 0
            for field in self.obs_i:
                iview[at:at + field.count] = obs_d[field.key].reshape(-1)
                at += field.count
            for field in self.act_i:
                w = field.count * n_actions
                iview[at:at + w] = acts_d[field.key].reshape(-1)
                at += w
        return self.record_bytes(n_actions)

    def unpack(self, buf: np.ndarray, offset: int, n_actions: int) -> tuple:
        """``(obs, acts)`` as numpy VIEWS into the arena — no copy.

        The views stay valid only until the worker reuses that arena, which it cannot do
        before its reply arrives, so the evaluator may read them freely for the duration
        of one forward pass.  ``np.stack`` / the padding buffers copy out of them.
        """
        n_f = self.obs_f_count + n_actions * self.act_f_width
        n_i = self.obs_i_count + n_actions * self.act_i_width
        f_bytes = _align(n_f * 4)
        fview = buf[offset:offset + n_f * 4].view(np.float32)

        obs: dict = {}
        acts: dict = {}
        at = 0
        for field in self.obs_f:
            obs[field.key] = fview[at:at + field.count].reshape(field.shape)
            at += field.count
        for field in self.act_f:
            w = field.count * n_actions
            acts[field.key] = fview[at:at + w].reshape((n_actions,) + field.shape)
            at += w

        if n_i:
            isz = self.int_dtype.itemsize
            iview = buf[offset + f_bytes:offset + f_bytes + n_i * isz].view(self.int_dtype)
            at = 0
            for field in self.obs_i:
                obs[field.key] = iview[at:at + field.count].reshape(field.shape)
                at += field.count
            for field in self.act_i:
                w = field.count * n_actions
                acts[field.key] = iview[at:at + w].reshape((n_actions,) + field.shape)
                at += w

        if self.flat:
            return obs[FLAT_OBS_KEY], acts[FLAT_ACT_KEY]
        return obs, acts


__all__ = ["LeafLayout", "FLAT_OBS_KEY", "FLAT_ACT_KEY", "as_obs_dict", "as_acts_dict",
           "ALIGN"]
