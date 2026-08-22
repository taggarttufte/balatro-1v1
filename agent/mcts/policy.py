"""
policy.py — policy/value functions used to expand MCTS leaf nodes.

    UniformPolicy  flat prior, value 0. Baseline for verifying the search loop.
    NNPolicy       PolicyValueNet wrapper. Encodes obs, featurizes legal actions,
                   runs the model once per leaf. The REFERENCE implementation.

The interface (this is what W3 implements against)
--------------------------------------------------
    fn(game) -> (priors, value)
        priors: {ActionKey: probability}, summing to 1 over the legal actions
                (empty dict when the game has no legal actions — MLB PVP_WAIT etc.)
        value:  scalar; the value head's output for this state

    fn.evaluate_many(games) -> [(priors, value), ...]
        SAME semantics, one entry per game, order preserved. `PolicyValueBase` supplies
        a correct-but-serial default (a Python loop over `__call__`), so every existing
        implementation already satisfies the protocol; a batched implementation
        overrides ONLY this method and gets to do one forward pass for all the leaves.

W3's job is `mcts/batched.py`: a `PolicyValueFn` whose `evaluate_many` stacks the obs
into (B, obs_dim), concatenates the ragged action features into (sum(N_i), A) and calls
`PolicyValueNet.score_actions_flat` once. `NNPolicy.encode_leaf()` below is factored out
precisely so the batched implementation can reuse the per-leaf CPU work verbatim and
only replace the torch part — and so a test can assert batched == single-leaf.

Do NOT batch inside this file. W1 owns the interface; W3 owns the implementation.
"""
from __future__ import annotations
from typing import Protocol, Sequence, runtime_checkable

import numpy as np
import torch

from balatro_sim.game import BalatroGame
from .action import ActionKey, action_key
from .action_features import featurize_actions
from .encoder import ObsEncoder, V7Encoder, encode_obs
from .model import PolicyValueNet

Evaluation = tuple[dict[ActionKey, float], float]


@runtime_checkable
class PolicyValueFn(Protocol):
    """A leaf evaluator. Single-leaf `__call__` is required; `evaluate_many` is the
    batching seam and has a serial default in `PolicyValueBase`."""

    def __call__(self, game: BalatroGame) -> Evaluation: ...

    def evaluate_many(self, games: Sequence[BalatroGame]) -> list[Evaluation]: ...


class PolicyValueBase:
    """Mixin providing the serial `evaluate_many`. Subclass and implement `__call__`."""

    def __call__(self, game: BalatroGame) -> Evaluation:   # pragma: no cover - abstract
        raise NotImplementedError

    def evaluate_many(self, games: Sequence[BalatroGame]) -> list[Evaluation]:
        return [self(g) for g in games]


class UniformPolicy(PolicyValueBase):
    """Uniform prior, constant value 0. Validates the search loop end-to-end."""

    def __call__(self, game: BalatroGame) -> Evaluation:
        legal = game.legal_actions()
        if not legal:
            return {}, 0.0
        p = 1.0 / len(legal)
        return {action_key(a): p for a in legal}, 0.0


class NNPolicy(PolicyValueBase):
    """
    Wraps a PolicyValueNet. Per call:
      1. Encode obs (447 floats by default; 453 with the MLB encoder).
      2. Enumerate legal actions, featurize each (56 floats per action).
      3. Single forward pass: logits over actions + scalar value.
      4. Softmax logits, return (priors_dict, value).

    Inference is wrapped in torch.no_grad(); the model is set to eval mode.
    A game with no legal actions (MLB `PVP_WAIT`, readied at a Nemesis) returns
    ({}, 0.0) WITHOUT touching the net — the search treats that node as a stop.
    """

    def __init__(self, model: PolicyValueNet, device: str | torch.device = "cpu",
                 encoder: ObsEncoder | None = None):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.encoder: ObsEncoder = encoder or V7Encoder()
        if self.encoder.dim != model.obs_dim:
            raise ValueError(
                f"encoder {self.encoder.name!r} produces {self.encoder.dim} dims but the "
                f"net expects {model.obs_dim}"
            )

    # ── The CPU half of a leaf evaluation (reused by W3's batched impl) ──────

    def encode_leaf(self, game: BalatroGame):
        """(legal_actions, obs (D,), action_feats (N, A)) or None when there is nothing
        to evaluate. No torch here — pure numpy, safe to call from a collector thread."""
        legal = game.legal_actions()
        if not legal:
            return None
        return legal, self.encoder(game), featurize_actions(legal)

    @staticmethod
    def priors_from_logits(legal: list[dict], probs: np.ndarray) -> dict[ActionKey, float]:
        return {action_key(a): float(probs[i]) for i, a in enumerate(legal)}

    # ── PolicyValueFn ────────────────────────────────────────────────────────

    def __call__(self, game: BalatroGame) -> Evaluation:
        encoded = self.encode_leaf(game)
        if encoded is None:
            return {}, 0.0
        legal, obs, feats = encoded

        obs_t = torch.from_numpy(obs).to(self.device)
        feats_t = torch.from_numpy(feats).to(self.device)

        with torch.no_grad():
            logits, value = self.model(obs_t, feats_t)
            probs = torch.softmax(logits, dim=-1)

        return self.priors_from_logits(legal, probs.detach().cpu().numpy()), float(value.item())


# ── Encoder-aware factory (Phase 4 W1) ──────────────────────────────────────────

def make_policy(net, device: str | torch.device = "cpu", encoder: ObsEncoder | None = None,
                batched: bool = True, **kwargs):
    """The one place that turns `(net, encoder)` into a `PolicyValueFn`.

    Flat encoder (`v7` / `mlb`) -> `BatchedNNPolicy` / `NNPolicy` (unchanged).
    Set encoder (`--encoder set`) -> `policy_set.BatchedSetNNPolicy` / `SetNNPolicy`.

    Both satisfy the SAME `PolicyValueFn` protocol, so `MCTS`, `BatchedSearch`,
    `MCTSPlayer` and `TreeCache` are encoder-blind and were not touched. Imports are
    function-local so `policy.py` keeps no import edge onto the set modules (and so a
    caller that only wants the 447-dim path never loads torch modules it does not use).
    """
    if encoder is not None and getattr(encoder, "is_set", False):
        from .policy_set import BatchedSetNNPolicy, SetNNPolicy
        cls = BatchedSetNNPolicy if batched else SetNNPolicy
        return cls(net, device=device, encoder=encoder, **kwargs)
    from .batched import BatchedNNPolicy
    cls = BatchedNNPolicy if batched else NNPolicy
    return cls(net, device=device, encoder=encoder, **kwargs)


__all__ = [
    "Evaluation", "PolicyValueFn", "PolicyValueBase", "UniformPolicy", "NNPolicy",
    "make_policy", "encode_obs",
]
