"""
train_v.py — regression of V = SetValueNet on label shards (Phase 5 rev 2, W5).

    python mp/ev/train_v.py --shards mp/ev/runs/labels_a/shards --run-dir mp/ev/runs/v1 \
        --max-steps 20000 --device cuda
    touch mp/ev/runs/v1/PAUSE            # pause between steps; resume with --resume latest
    python mp/ev/train_v.py --resume mp/ev/runs/v1/latest.pt --max-steps 40000

Loss: BCE on logits vs the SOFT label y (P(win) from rollouts), AdamW, cosine (default) or
flat LR with warmup, batch 256.  Held-out-by-SEED evaluation every ``--eval-every`` steps:
BCE, Brier, AUC (labels binarised at 0.5), and a 10-bin reliability curve (+ ECE), each
against the constant predictor (the training-set mean) and against the label-noise floor
(the Brier a perfect V would score against noisy rollout labels, ``mean((ci/1.96)^2)``).

Checkpoints: ``latest.pt`` + ``ckpt_<step>.pt`` (pruned to ``--keep``), W1's
``value_net.save_checkpoint`` format with the trainer state in ``extra["trainer"]``
(optimizer moments, step / epoch / batch cursor, numpy + torch + python RNG, config, eval
history) — a resume is a CONTINUATION: the epoch permutation is a function of (seed, epoch)
and the cursor says where in it we were, so the next batch is the one the interrupted run
would have drawn.  ``test_train_v.py`` pins the round trip bit-exact.

Run dir: ``train.jsonl`` (config / eval / checkpoint / summary records), console one line
per eval, ``PAUSE`` file honoured between steps (and Ctrl+C), ``.DONE`` sentinel written on
natural completion (max steps / epochs reached), never on a pause.

Model kinds: ``set_value_net`` (W1, the real thing; obs from ``encoder="v2"`` shards) and
``dummy`` (a 16-scalar MLP for the plumbing tests; ``labels.make_encoder("dummy")`` shards).
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
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import _bootstrap  # noqa: F401
from dataset import LabelDataset, list_shards

__all__ = [
    "TrainVConfig", "VTrainer", "evaluate", "reliability_curve", "auc_score", "build_model",
    "DummyValueNet", "PAUSE_FILE", "DONE_FILE", "main",
]

PAUSE_FILE = "PAUSE"
DONE_FILE = ".DONE"
TRAINER_STATE_VERSION = 1


# ── config ────────────────────────────────────────────────────────────────────────

@dataclass
class TrainVConfig:
    shards: list = field(default_factory=list)
    run_dir: str = "mp/ev/runs/v_default"
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

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainVConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


# ── models ────────────────────────────────────────────────────────────────────────

class DummyValueNet(nn.Module):
    """16 scalars -> logit.  The plumbing stand-in (tests run in seconds on CPU)."""
    KIND = "dummy"

    def __init__(self, d_in: int = 16, hidden: int = 64):
        super().__init__()
        self.cfg = {"d_in": d_in, "hidden": hidden}
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden),
                                 nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, batch: dict) -> torch.Tensor:
        return self.net(batch["x"]).squeeze(-1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(kind: str, net_cfg: Optional[dict] = None, device="cpu"):
    """``(net, encoder_or_None)``: the real net needs W1's encoder for its checkpoint."""
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


# ── the trainer ───────────────────────────────────────────────────────────────────

class VTrainer:
    def __init__(self, cfg: TrainVConfig, *, data: Optional[tuple] = None, log=print):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.log = log
        if cfg.torch_threads and cfg.torch_threads > 0:
            torch.set_num_threads(int(cfg.torch_threads))
        self.step = 0
        self.epoch = 0
        self.cursor = 0                    # batches consumed in the current epoch
        self.samples_seen = 0
        self.elapsed_s = 0.0
        self.history: list = []
        self.best: Optional[dict] = None
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        self.net, self.encoder = build_model(cfg.model, cfg.net_cfg, self.device)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.rng = np.random.default_rng(cfg.seed)
        if data is not None:
            self.train_ds, self.holdout_ds = data
        else:
            self.load_data()
        self.const = float(self.train_ds.y.mean()) if len(self.train_ds) else 0.5
        self._order = None
        self._order_epoch = -1

    # ── data ──
    def load_data(self) -> None:
        paths = list_shards(self.cfg.shards)
        if not paths:
            raise FileNotFoundError(f"no shards under {self.cfg.shards}")
        full = LabelDataset.load(paths)
        if self.cfg.label_clip > 0:
            full.y = np.clip(full.y, self.cfg.label_clip, 1 - self.cfg.label_clip).astype(np.float32)
        self.train_ds, self.holdout_ds = full.split_by_seed(self.cfg.holdout_frac, self.cfg.holdout_seeds)
        self.log(f"  data: {len(full)} rows from {len(paths)} shards, {len(full.seeds())} seeds -> "
                 f"train {len(self.train_ds)} ({len(self.train_ds.seeds())} seeds) / "
                 f"holdout {len(self.holdout_ds)} ({len(self.holdout_ds.seeds())} seeds)")

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
        obs = {k: v[idx] for k, v in self.train_ds.obs.items()}
        return obs, self.train_ds.y[idx]

    # ── one step ──
    def train_step(self) -> dict:
        obs, y = self.next_batch()
        lr = self.lr_at(self.step)
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        self.net.train()
        batch = to_batch(obs, self.device)
        target = torch.from_numpy(np.asarray(y, dtype=np.float32)).to(self.device)
        logits = self.net(batch)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.clip_grad)) \
            if self.cfg.clip_grad and self.cfg.clip_grad > 0 else float("nan")
        self.optimizer.step()
        self.step += 1
        self.samples_seen += int(len(y))
        return {"step": self.step, "loss": float(loss.item()), "lr": lr, "grad_norm": gn,
                "epoch": self.epoch, "cursor": self.cursor}

    def eval(self) -> dict:
        m = evaluate(self.net, self.holdout_ds, self.device, const=self.const,
                     batch_size=self.cfg.eval_batch_size)
        m["step"] = self.step
        m["epoch"] = self.epoch
        m["samples_seen"] = self.samples_seen
        self.history.append({k: v for k, v in m.items() if k not in ("reliability", "by_kind")})
        if m.get("n") and (self.best is None or m["bce"] < self.best["bce"]):
            self.best = {"step": self.step, "bce": m["bce"], "brier": m["brier"], "auc": m["auc"]}
        return m

    # ── checkpoints ──
    def state(self) -> dict:
        return {
            "version": TRAINER_STATE_VERSION,
            "step": self.step, "epoch": self.epoch, "cursor": self.cursor,
            "samples_seen": self.samples_seen, "elapsed_s": self.elapsed_s,
            "optimizer": self.optimizer.state_dict(),
            "rng": {"numpy": self.rng.bit_generator.state, "torch": torch.get_rng_state(),
                    "python": random.getstate(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
            "config": self.cfg.as_dict(), "history": self.history, "best": self.best,
            "const": self.const, "n_train": len(self.train_ds), "n_holdout": len(self.holdout_ds),
            "holdout_seeds": self.holdout_ds.seeds(),
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
                        log=print) -> "VTrainer":
        net, encoder, extra = cls.read_checkpoint(path)
        ts = extra["trainer"]
        cfg = TrainVConfig.from_dict({**ts["config"], **(overrides or {})})
        tr = cls(cfg, data=data, log=log)
        tr.net.load_state_dict(net.state_dict(), strict=True)
        tr.net.to(tr.device)
        if encoder is not None:
            tr.encoder = encoder
        tr.optimizer.load_state_dict(ts["optimizer"])
        tr.step, tr.epoch, tr.cursor = ts["step"], ts["epoch"], ts["cursor"]
        tr.samples_seen, tr.elapsed_s = ts["samples_seen"], ts["elapsed_s"]
        tr.history, tr.best = list(ts.get("history", [])), ts.get("best")
        tr.const = float(ts.get("const", tr.const))
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
    return (f"step {m['step']:>6}  ep {m['epoch']}  holdout n={m['n']}  "
            f"bce {m['bce']:.4f} (const {c.get('bce', float('nan')):.4f})  "
            f"brier {m['brier']:.4f} (const {c.get('brier', float('nan')):.4f}, floor "
            f"{m.get('noise_floor_brier', float('nan')):.4f})  auc {m['auc']:.3f}  "
            f"ece {m['reliability']['ece']:.3f}  p_sd {m['p_sd']:.3f}")


def run(cfg: Optional[TrainVConfig] = None, *, resume: Optional[str] = None,
        overrides: Optional[dict] = None, data: Optional[tuple] = None,
        log=print, stop_check=None) -> dict:
    """Train (``cfg``) or resume (``resume`` = checkpoint path; ``overrides`` = the config
    keys to change, typically ``max_steps`` / ``minutes`` / ``device``)."""
    if resume:
        ckpt_path = Path(resume)
        if ckpt_path.is_dir():
            ckpt_path = ckpt_path / "latest.pt"
        tr = VTrainer.from_checkpoint(ckpt_path, overrides=overrides or {}, data=data, log=log)
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
        tr = VTrainer(cfg, data=data, log=log)
    log_path = run_dir / "train.jsonl"
    lf = open(log_path, "a", encoding="utf-8")

    def emit(rec: dict) -> None:
        lf.write(json.dumps(rec, default=str) + "\n")
        lf.flush()

    n_params = sum(p.numel() for p in tr.net.parameters())
    emit({"kind": "config", "timestamp": datetime.now().isoformat(timespec="seconds"),
          "config": cfg.as_dict(), "n_params": n_params, "resumed_from": resume,
          "n_train": len(tr.train_ds), "n_holdout": len(tr.holdout_ds),
          "holdout_seeds": tr.holdout_ds.seeds(), "const": tr.const})
    log(f"=== V training: {cfg.model} ({n_params:,} params) on {cfg.device}, batch {cfg.batch_size}, "
        f"lr {cfg.lr} {cfg.lr_schedule}, max_steps {cfg.max_steps} ===")
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
    recent = []
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
            if cfg.eval_every and tr.step % cfg.eval_every == 0:
                m = tr.eval()
                m["train_loss"] = float(np.mean(recent)) if recent else float("nan")
                recent = []
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
            emit({"kind": "eval", **m})
            log("  " + fmt_eval(m))
            last_eval = m
    final = checkpoint(f"exit:{why}")
    natural = why in ("max_steps", "max_epochs")
    if natural:
        done_path.write_text(f"{why} at step {tr.step} {datetime.now().isoformat(timespec='seconds')}\n")
    summary = {"kind": "summary", "stop_reason": why, "step": tr.step, "epoch": tr.epoch,
               "samples_seen": tr.samples_seen, "elapsed_s": tr.elapsed_s, "best": tr.best,
               "final_eval": {k: v for k, v in (last_eval or {}).items() if k not in ("reliability", "by_kind")},
               "reliability": (last_eval or {}).get("reliability"), "by_kind": (last_eval or {}).get("by_kind"),
               "final_checkpoint": str(final), "done": natural}
    emit(summary)
    lf.close()
    log(f"=== stopped ({why}) at step {tr.step}; latest: {final}"
        + (f"; paused by {pause_path} (delete it or --resume)" if why == "PAUSE" else "")
        + (f"; {done_path.name} written" if natural else ""))
    log(f"resume: python mp/ev/train_v.py --resume {final} --max-steps <N>")
    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────────

_RESUME_KEYS = ("max_steps", "max_epochs", "minutes", "device", "eval_every", "checkpoint_every",
                "keep", "shards", "run_dir", "lr", "lr_schedule", "holdout_frac")


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
    return ap


def _given(args) -> dict:
    d = {}
    for k, v in vars(args).items():
        if v is None or k == "resume":
            continue
        if k == "net_cfg":
            v = json.loads(v)
        if k == "holdout_seeds":
            v = v.split(",")
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
