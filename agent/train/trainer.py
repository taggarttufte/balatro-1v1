"""
trainer.py — single-step trainer for the AlphaZero-style policy/value loss.

Loss components per Sample:
  policy_loss = - sum_a target_policy[a] * log_softmax(logits)[a]
  value_loss  = (value_pred - z)^2

Total = mean over batch of (policy_loss + value_loss).

Action sets vary per sample (legal actions depend on game state), so we can't trivially
stack everything into one batched forward pass. The trainer loops over samples and
accumulates loss — slow but straightforward. (W3 owns the batched path: with
`PolicyValueNet.score_actions_flat` a ragged batch is one trunk call plus one policy-head
call; the same trick applies here once self-play throughput stops being the bottleneck.)

Fork note (2026-08-21): `state_dict` / `load_state_dict` added so `--resume` restores the
Adam moments, not just the weights — a resume that drops the optimizer state is a
silently different training run.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

import numpy as np

from mcts.model import PolicyValueNet
from .trajectory import Sample


class Trainer:
    def __init__(
        self,
        net: PolicyValueNet,
        lr: float = 1e-3,
        device: str | torch.device = "cpu",
        weight_decay: float = 0.0,
    ):
        self.device = torch.device(device)
        self.net = net.to(self.device)
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=lr, weight_decay=weight_decay
        )

    def step(self, batch: list[Sample]) -> dict[str, float]:
        """One optimizer step on a batch. Returns mean loss components."""
        if not batch:
            return {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0, "n": 0}

        self.net.train()
        self.optimizer.zero_grad()

        policy_loss_sum = torch.zeros((), device=self.device)
        value_loss_sum = torch.zeros((), device=self.device)

        # Phase 4 W1: a batch may mix v1 records, v2 records with a flat observation, and
        # v2 records with the set encoder's dict observation. The flat paths keep the
        # Phase 3 per-sample loop verbatim; the set records go through ONE padded forward
        # pass for the whole sub-batch (they are ~16 rows each, so padding is free).
        flat = [s for s in batch if not _is_set_sample(s)]
        set_samples = [s for s in batch if _is_set_sample(s)]

        for s in flat:
            obs = torch.from_numpy(s.obs).to(self.device)
            feats = torch.from_numpy(_actions_of(s)).to(self.device)
            target = torch.from_numpy(s.target_policy).to(self.device)
            z = torch.tensor(s.z, dtype=torch.float32, device=self.device)

            logits, value = self.net(obs, feats)               # (N,), scalar
            log_probs = F.log_softmax(logits, dim=-1)          # (N,)
            policy_loss_sum = policy_loss_sum + -(target * log_probs).sum()
            value_loss_sum = value_loss_sum + (value - z).pow(2)

        if set_samples:
            p, v = self._set_losses(set_samples)
            policy_loss_sum = policy_loss_sum + p
            value_loss_sum = value_loss_sum + v

        n = len(batch)
        policy_loss = policy_loss_sum / n
        value_loss = value_loss_sum / n
        total = policy_loss + value_loss

        total.backward()
        self.optimizer.step()

        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "total_loss": float(total.item()),
            "n": n,
        }

    # ── The set-encoder sub-batch ────────────────────────────────────────────

    def _set_losses(self, samples: list) -> tuple[torch.Tensor, torch.Tensor]:
        """Summed (policy, value) loss over set-encoded v2 samples, in ONE forward pass.

        The softmax is over the SUBSAMPLED rows only — that is the intended objective
        change (SETENC_NOTES §4.3), not an approximation of a full-set softmax that this
        code could have taken and didn't.
        """
        from mcts.policy_set import pad_acts, stack_obs

        obs = stack_obs([s.obs for s in samples], self.device)
        counts = [int(s.target_policy.shape[0]) for s in samples]
        acts, mask = pad_acts([s.actions for s in samples], counts, self.device)

        max_k = max(counts)
        tgt = np.zeros((len(samples), max_k), dtype=np.float32)
        for i, s in enumerate(samples):
            tgt[i, :counts[i]] = s.target_policy
        target = torch.from_numpy(tgt).to(self.device)
        z = torch.tensor([s.z for s in samples], dtype=torch.float32, device=self.device)

        logits, values = self.net(obs, acts)
        logits = logits.masked_fill(~mask, float("-inf"))
        log_probs = F.log_softmax(logits, dim=-1)
        # -inf * 0 is nan, so zero the padded columns after the log_softmax.
        log_probs = torch.where(mask, log_probs, torch.zeros_like(log_probs))
        policy = -(target * log_probs).sum()
        value = (values - z).pow(2).sum()
        return policy, value

    # ── Checkpointing ────────────────────────────────────────────────────────

    def state_dict(self) -> dict:
        return {
            "optimizer": self.optimizer.state_dict(),
            "lr": self.lr,
            "weight_decay": self.weight_decay,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.optimizer.load_state_dict(sd["optimizer"])
        self.lr = sd.get("lr", self.lr)
        self.weight_decay = sd.get("weight_decay", self.weight_decay)


# ── sample-shape helpers (v1 / v2 dispatch) ─────────────────────────────────────

def _actions_of(s):
    """The action feature block, whichever `Sample` version this is."""
    return s.actions if getattr(s, "version", 1) >= 2 else s.action_features


def _is_set_sample(s) -> bool:
    return getattr(s, "version", 1) >= 2 and isinstance(s.obs, dict)
