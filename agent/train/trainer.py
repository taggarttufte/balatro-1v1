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

        for s in batch:
            obs = torch.from_numpy(s.obs).to(self.device)
            feats = torch.from_numpy(s.action_features).to(self.device)
            target = torch.from_numpy(s.target_policy).to(self.device)
            z = torch.tensor(s.z, dtype=torch.float32, device=self.device)

            logits, value = self.net(obs, feats)               # (N,), scalar
            log_probs = F.log_softmax(logits, dim=-1)          # (N,)
            policy_loss_sum = policy_loss_sum + -(target * log_probs).sum()
            value_loss_sum = value_loss_sum + (value - z).pow(2)

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
