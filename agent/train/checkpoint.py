"""
checkpoint.py — save / restore a cold-start training run.

What a checkpoint holds (everything needed for `--resume` to be a *continuation*, not a
new run that happens to start from some weights):

    model         PolicyValueNet.state_dict() + its constructor description
    trainer       Adam's state_dict (the moments — dropping them silently changes the run)
    counters      episodes / samples / train steps / wins / errors / wall clock
    rng           numpy Generator state (episode seeds, Gumbel noise, batch sampling),
                  torch CPU + CUDA generator states, and Python's `random`
    config        the TrainConfig, so a resume cannot silently change sims / lr / ruleset
    buffer        the replay buffer (optional but ON by default): resuming with an empty
                  buffer trains on a different mini-batch stream, so the round-trip could
                  never be bit-exact

Format: a single `torch.save` pickle. torch >= 2.6 defaults `torch.load` to
`weights_only=True`, which cannot load numpy arrays / plain dicts, so `load_checkpoint`
passes `weights_only=False` explicitly. Only load checkpoints you produced.

Writes are atomic (temp file + `os.replace`) so a Ctrl+C during the write cannot leave a
truncated checkpoint where the good one used to be.
"""
from __future__ import annotations

import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

CHECKPOINT_VERSION = 1
CHECKPOINT_KIND = "mp/agent train_cold"


# ── Global RNG states ───────────────────────────────────────────────────────────

def global_rng_state(device: torch.device | str = "cpu") -> dict:
    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def load_global_rng_state(state: dict) -> None:
    if "torch" in state:
        torch.set_rng_state(_as_byte_tensor(state["torch"]))
    if "python" in state and state["python"] is not None:
        random.setstate(_as_tuple(state["python"]))
    if "cuda" in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([_as_byte_tensor(s) for s in state["cuda"]])
        except (RuntimeError, ValueError):
            # A checkpoint written on a box with a different GPU count. The training loop
            # does not draw from the CUDA generator (no dropout, no CUDA sampling), so a
            # mismatch is recoverable — say so rather than dying on resume.
            pass


def _as_byte_tensor(x) -> torch.Tensor:
    t = x if isinstance(x, torch.Tensor) else torch.tensor(x)
    return t.to(dtype=torch.uint8, device="cpu")


def _as_tuple(x):
    """random.setstate needs tuples; pickling round-trips them but be defensive."""
    if isinstance(x, tuple):
        return tuple(_as_tuple(i) if isinstance(i, list) else i for i in x)
    if isinstance(x, list):
        return tuple(_as_tuple(i) if isinstance(i, list) else i for i in x)
    return x


# ── Save / load ─────────────────────────────────────────────────────────────────

def save_checkpoint(path: str | Path, payload: dict) -> Path:
    """Atomically write `payload` (see `ColdTrainer.state_dict`) to `path`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload.setdefault("version", CHECKPOINT_VERSION)
    payload.setdefault("kind", CHECKPOINT_KIND)
    payload.setdefault("saved_at", datetime.now().isoformat(timespec="seconds"))
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    """Load a checkpoint written by `save_checkpoint`. `weights_only=False` is required
    (the payload carries numpy arrays and plain Python objects, not just tensors)."""
    path = Path(path)
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(ckpt, dict) or ckpt.get("kind") != CHECKPOINT_KIND:
        raise ValueError(f"{path} is not an {CHECKPOINT_KIND} checkpoint")
    version = ckpt.get("version")
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            f"{path} is checkpoint version {version}, this code writes {CHECKPOINT_VERSION}"
        )
    return ckpt


def latest_checkpoint(run_dir: str | Path, pattern: str = "*.pt") -> Optional[Path]:
    """Newest checkpoint in a run directory, or None."""
    files = sorted(Path(run_dir).glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def numpy_rng_state(rng: np.random.Generator) -> dict:
    return rng.bit_generator.state


def set_numpy_rng_state(rng: np.random.Generator, state: dict) -> None:
    rng.bit_generator.state = state
