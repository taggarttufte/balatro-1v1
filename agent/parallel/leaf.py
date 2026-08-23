"""
leaf.py — the CPU half of a leaf evaluation, without a net.

``NNPolicy.encode_leaf`` and ``SetNNPolicy.encode_leaf`` are the two implementations the
search already uses, but both live on a class that owns a ``torch`` model.  A worker
process holds **no net** (that is the entire point of a shared evaluator), so it needs the
same three lines without the model: legal actions, the observation, the action features.

``LeafEncoder`` is exactly those three lines.  It is not a re-implementation with room to
drift: ``tests/test_parallel.py::test_leaf_encoder_matches_policy_encode_leaf`` asserts,
for BOTH encoders, that ``LeafEncoder.encode_leaf(game)`` is bit-identical to the
corresponding ``*NNPolicy.encode_leaf(game)`` on real MLB states.  If W1 or W3 changes what
a leaf looks like, that test fails here rather than a run silently sending the evaluator
the wrong bytes.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from mcts.action import ActionKey, action_key
from mcts.action_features import featurize_actions
from mcts.action_features_set import featurize_actions_set
from mcts.encoder import get_encoder


class LeafEncoder:
    """`encode_leaf(game) -> (legal, obs, acts) | None`, for either encoder.

    ``caps`` is only meaningful for ``encoder="set"`` and comes from the checkpoint
    (``ckpt["encoder_caps"]``), so a worker pads exactly the way the net was trained —
    the same rule ``mcts.load_policy`` follows.
    """

    def __init__(self, encoder: str = "set", caps: Optional[dict] = None,
                 hand_type_features: bool = True):
        kwargs = {"caps": caps} if (caps and encoder == "set") else {}
        self.encoder = get_encoder(encoder, **kwargs)
        self.name = encoder
        self.is_set = bool(getattr(self.encoder, "is_set", False))
        self.hand_type_features = bool(hand_type_features)

    # ── the three lines ──────────────────────────────────────────────────────

    def encode_leaf(self, game):
        legal = game.legal_actions()
        if not legal:
            return None
        if self.is_set:
            return (legal, self.encoder(game),
                    featurize_actions_set(game, legal, self.encoder.caps,
                                          hand_type_features=self.hand_type_features))
        return legal, self.encoder(game), featurize_actions(legal)

    # ── and the one on the way back ──────────────────────────────────────────

    @staticmethod
    def priors_from_logits(legal, probs: np.ndarray) -> dict:
        """Identical to ``NNPolicy.priors_from_logits`` / ``SetNNPolicy``'s (they are the
        same static method twice)."""
        return {action_key(a): float(probs[i]) for i, a in enumerate(legal)}

    def prototype(self, game):
        """One real ``(obs, acts)`` pair, for ``LeafLayout.from_prototype``."""
        encoded = self.encode_leaf(game)
        if encoded is None:                          # pragma: no cover - caller picks a game
            raise ValueError("prototype game has no legal actions")
        return encoded[1], encoded[2]


__all__ = ["LeafEncoder", "ActionKey"]
