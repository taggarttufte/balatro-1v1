"""training.py — the one training recipe every arm shares, and the evaluation surface.

``train_v.py`` is used strictly as a library: ``VTrainer`` accepts ``data=(train, holdout)``,
so the POC never has to materialise its arm corpora as shards or touch the trainer's own
data path.  Every run in the POC uses ``RECIPE`` verbatim — only ``seed``, ``run_dir`` and
the training set differ, which is what makes the three-arm comparison a comparison of DATA.

Regime: the known-good full-corpus run (``ev/runs/v_full_best``) bottomed out on held-out
BCE at step 1250 / epoch 7 of a cosine-to-2000 schedule at 45.9k rows.  At 12k rows the
step count that corresponds to is re-derived here by a probe run (``derive_regime``) rather
than assumed, and the resulting ``S*`` is then FIXED for every ensemble and arm run so that
no arm gets a private early-stopping decision.
"""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import train_v as TV
from dataset import LabelDataset

__all__ = ["RECIPE", "train_one", "derive_regime", "load_net", "predict_probs",
           "evaluate_full", "ensemble_scores"]

# Identical for every run in the POC.  ``shards`` is unused (data is passed in memory).
RECIPE = dict(
    model="set_value_net", net_cfg={}, batch_size=256, lr=3e-4, lr_schedule="cosine",
    min_lr_frac=0.05, weight_decay=1e-4, warmup_steps=100, clip_grad=1.0,
    eval_every=25, eval_batch_size=1024, checkpoint_every=10 ** 9, keep=1,
    device="cuda" if torch.cuda.is_available() else "cpu", torch_threads=4, label_clip=0.0,
)


def train_one(train_ds: LabelDataset, holdout_ds: LabelDataset, run_dir, *, seed: int,
              max_steps: int, log=print) -> dict:
    """One V run on ``train_ds``, evaluated against ``holdout_ds``.  Returns ``run()``'s
    summary plus the path of the final checkpoint (which is exactly at ``max_steps``)."""
    cfg = TV.TrainVConfig.from_dict(dict(RECIPE, shards=[], run_dir=str(run_dir), seed=int(seed),
                                         max_steps=int(max_steps),
                                         warmup_steps=min(RECIPE["warmup_steps"], max(int(max_steps) // 5, 1))))
    summary = TV.run(cfg, data=(train_ds, holdout_ds), log=log)
    summary["checkpoint"] = str(Path(run_dir) / "latest.pt")
    summary["n_train"] = len(train_ds)
    summary["seed"] = int(seed)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def derive_regime(train_ds: LabelDataset, holdout_ds: LabelDataset, run_dir, *,
                  probe_steps: int = 800, seed: int = 0, log=print) -> dict:
    """Probe run on the base corpus -> the step at which held-out BCE bottoms out.

    ``S*`` is rounded to the eval grid and clamped away from the two ends (a best at step 0
    or at the very last step means the probe bracket was wrong, and the caller should know)."""
    summary = train_one(train_ds, holdout_ds, run_dir, seed=seed, max_steps=probe_steps, log=log)
    hist = [h for h in _history(run_dir) if h.get("n")]
    curve = [(int(h["step"]), float(h["bce"]), float(h["brier"]), float(h["auc"])) for h in hist]
    best = min((c for c in curve if c[0] > 0), key=lambda c: c[1])
    return {"probe_steps": probe_steps, "s_star": best[0], "best_bce": best[1],
            "best_brier": best[2], "best_auc": best[3],
            "at_edge": best[0] >= probe_steps, "curve": curve, "summary": summary}


def _history(run_dir) -> list:
    """Eval records of the LAST run in this dir.  ``train_v.run`` appends to ``train.jsonl``,
    so a re-run into the same dir would otherwise mix an old curve into the new one."""
    import json
    p = Path(run_dir) / "train.jsonl"
    if not p.exists():
        return []
    recs = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            recs.append(json.loads(line))
        except ValueError:
            continue
    starts = [i for i, r in enumerate(recs) if r.get("kind") == "config"]
    if starts:
        recs = recs[starts[-1]:]
    return [r for r in recs if r.get("kind") == "eval"]


def load_net(ckpt_path, device=None):
    device = device or RECIPE["device"]
    net, _enc, _extra = TV.VTrainer.read_checkpoint(ckpt_path, device=device)
    net.to(device).eval()
    return net


@torch.no_grad()
def predict_probs(net, ds: LabelDataset, device=None, batch_size: int = 1024) -> np.ndarray:
    return TV.predict(net, ds, torch.device(device or RECIPE["device"]), batch_size)


def ensemble_scores(ckpts, ds: LabelDataset, device=None, batch_size: int = 1024) -> tuple:
    """``(P, mean, std)`` where ``P`` is (n_members, n_rows) of sigmoid outputs."""
    preds = []
    for c in ckpts:
        net = load_net(c, device)
        preds.append(predict_probs(net, ds, device, batch_size))
        del net
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    P = np.stack(preds)
    return P, P.mean(axis=0), P.std(axis=0)


# ── evaluation ────────────────────────────────────────────────────────────────────

def evaluate_full(net, holdout: LabelDataset, *, const: float, device=None,
                  hard_mask: Optional[np.ndarray] = None) -> dict:
    """Overall + per-kind metrics on the holdout, plus the high-disagreement stratum.

    ``hard_mask`` selects the stratum (computed once from the ensemble's disagreement ON THE
    HOLDOUT, so it is the same set of rows for every arm)."""
    p = predict_probs(net, holdout, device)
    m = TV.metrics(p, holdout.y, ci=holdout.columns.get("ci"), const=const)
    m["ece"] = m["reliability"]["ece"]
    if "kind" in holdout.columns:
        kinds = holdout.columns["kind"]
        m["by_kind"] = {}
        for k in sorted(set(kinds.tolist())):
            sel = kinds == k
            if sel.sum() >= 20:
                m["by_kind"][str(k)] = {
                    "n": int(sel.sum()), "bce": TV._bce(p[sel], holdout.y[sel]),
                    "brier": float(((p[sel] - holdout.y[sel]) ** 2).mean()),
                    "auc": TV.auc_score(p[sel], (holdout.y[sel] >= 0.5).astype(np.float32)),
                }
    if hard_mask is not None and hard_mask.any():
        sub = holdout.subset(np.nonzero(hard_mask)[0])
        ps = p[hard_mask]
        m["hard_stratum"] = {
            "n": int(hard_mask.sum()), "bce": TV._bce(ps, sub.y),
            "brier": float(((ps - sub.y) ** 2).mean()),
            "auc": TV.auc_score(ps, (sub.y >= 0.5).astype(np.float32)),
            "ece": TV.reliability_curve(ps, sub.y)["ece"],
        }
    return m
