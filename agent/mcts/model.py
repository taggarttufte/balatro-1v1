"""
model.py — PolicyValueNet for AlphaZero-style search over Balatro.

Architecture (unchanged from balatro-mcts @ ee75d11):
  obs (447) -> embed(H) -> N x ResBlock(H) -> trunk(H)
  trunk -> value(1)
  trunk + action_features(56) -> policy_mlp -> logit per action

Fork note (2026-08-21): the trunk and value head are still V7-SHAPED (hidden=512,
n_res_blocks=4) because that shape was tuned on this game and there is no reason to
change it — but every "so V7 weights can be warm-started" affordance is gone. The V7
Run-4 checkpoint was lost (confirmed 2026-06-10) and the observation is 447 dims now,
not the 434 V7 was trained on, so no V7 state_dict could load even if one turned up.
Cold start is the only start. `PolicyValueNet.describe()` records obs/action dims in the
checkpoint so a mismatched net fails at load instead of silently mis-encoding.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

from .encoder import OBS_DIM
from .action_features import ACTION_FEATURE_DIM


class ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + torch.relu(self.fc2(torch.relu(self.fc1(x)))))


class PolicyValueNet(nn.Module):
    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        action_feat_dim: int = ACTION_FEATURE_DIM,
        hidden: int = 512,
        n_res_blocks: int = 4,
        policy_hidden: int = 128,
    ):
        super().__init__()
        H = hidden
        self.obs_dim = obs_dim
        self.action_feat_dim = action_feat_dim
        self.hidden = hidden
        self.n_res_blocks = n_res_blocks
        self.policy_hidden = policy_hidden

        self.embed = nn.Sequential(nn.Linear(obs_dim, H), nn.ReLU())
        self.res_blocks = nn.Sequential(*[ResidualBlock(H) for _ in range(n_res_blocks)])
        self.value_head = nn.Linear(H, 1)

        # Pointer-style policy head: scores each action given trunk + action features
        self.policy_head = nn.Sequential(
            nn.Linear(H + action_feat_dim, policy_hidden),
            nn.ReLU(),
            nn.Linear(policy_hidden, 1),
        )

        # Init
        nn.init.orthogonal_(self.embed[0].weight, gain=np.sqrt(2))
        nn.init.constant_(self.embed[0].bias, 0)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.constant_(self.value_head.bias, 0)
        nn.init.orthogonal_(self.policy_head[0].weight, gain=np.sqrt(2))
        nn.init.constant_(self.policy_head[0].bias, 0)
        nn.init.orthogonal_(self.policy_head[2].weight, gain=0.01)  # near-uniform at init
        nn.init.constant_(self.policy_head[2].bias, 0)

    # ── Introspection (checkpoint metadata) ──────────────────────────────────

    def describe(self) -> dict:
        """The constructor arguments, so a checkpoint can rebuild an identical net."""
        return {
            "obs_dim": self.obs_dim,
            "action_feat_dim": self.action_feat_dim,
            "hidden": self.hidden,
            "n_res_blocks": self.n_res_blocks,
            "policy_hidden": self.policy_hidden,
        }

    @classmethod
    def from_description(cls, desc: dict) -> "PolicyValueNet":
        return cls(**desc)

    # ── Forward ──────────────────────────────────────────────────────────────

    def get_trunk(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, obs_dim) or (obs_dim,) -> trunk (B, H) or (H,)"""
        return self.res_blocks(self.embed(obs))

    def value(self, trunk: torch.Tensor) -> torch.Tensor:
        """trunk: (B, H) -> (B,) scalar"""
        return self.value_head(trunk).squeeze(-1)

    def score_actions(self, trunk: torch.Tensor, action_feats: torch.Tensor) -> torch.Tensor:
        """
        Score each action given a single trunk vector.

        trunk:        (H,)             - state embedding
        action_feats: (N, action_dim)  - featurized legal actions
        returns:      (N,)             - logits, one per action
        """
        N = action_feats.shape[0]
        trunk_rep = trunk.unsqueeze(0).expand(N, -1)        # (N, H)
        x = torch.cat([trunk_rep, action_feats], dim=-1)    # (N, H+action_dim)
        return self.policy_head(x).squeeze(-1)              # (N,)

    def score_actions_flat(self, trunk: torch.Tensor, action_feats: torch.Tensor,
                           counts: torch.Tensor) -> torch.Tensor:
        """
        Batched variant for W3: score a RAGGED batch in one policy-head call.

        trunk:        (B, H)            - one trunk row per state
        action_feats: (sum(counts), A)  - every state's actions concatenated, in order
        counts:       (B,)              - number of actions per state
        returns:      (sum(counts),)    - logits, aligned with action_feats

        Equivalent to concatenating `score_actions(trunk[i], feats_i)` over i, but a
        single `policy_head` call instead of B of them. Pinned by
        tests/test_nn_policy.py::test_score_actions_flat_matches_per_state.
        """
        trunk_rep = torch.repeat_interleave(trunk, counts, dim=0)   # (sum(counts), H)
        x = torch.cat([trunk_rep, action_feats], dim=-1)
        return self.policy_head(x).squeeze(-1)

    def forward(self, obs: torch.Tensor, action_feats: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Single-state forward pass.

        obs:          (obs_dim,)
        action_feats: (N, action_dim)
        returns:      (logits (N,), value scalar)
        """
        trunk = self.get_trunk(obs)
        v = self.value_head(trunk).squeeze(-1)
        logits = self.score_actions(trunk, action_feats)
        return logits, v
