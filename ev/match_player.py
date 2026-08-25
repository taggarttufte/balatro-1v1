"""
match_player.py — ``EVPlayer(value_fn=V)`` for MATCH play (Phase 5 rev 2, W5).

W3's ``EVPlayer.value_fn`` takes only a ``BalatroGame`` (the cloned / stepped candidate
states it evaluates), but V's input is ``SetEncoderV2(game, opponent_view(match, player))``
— the opponent block comes from the live match.  ``MatchAwareEVPlayer`` binds a mutable
opponent view into the ``value_fn`` closure and refreshes it from the live match right
before every ``act``: every clone V is asked about during that decision shares the same
opponent view (the opponent does not move while we think — exactly the information a
player has at decision time).

    mp = MatchAwareEVPlayer(net, encoder, device="cuda", budget="fast", seed=0)
    policy = mp.policy()                       # (match, player, acts) -> action, for play_out / play_1v1
    # or, per game:  mp.bind(match, player); mp.act(match.games[player])

``policy()`` keeps one ``EVPlayer`` per seat (seed + player) so a wrapper can serve both
sides of a match or a whole tournament.  ``stats_ok`` counts ``value_fn`` exceptions —
W3's ``_v`` swallows them (player.py:360-363) and returns 0.0, which would silently turn V
off; the wrapper re-raises on the first one unless ``swallow=True``.

``load_value(path, device)`` → ``(net, encoder)`` from W1's checkpoint (the trainer's
``latest.pt`` is in that format).
"""
from __future__ import annotations

from typing import Callable, Optional

import _bootstrap  # noqa: F401

from mcts.encoder_v2 import opponent_view, SetEncoderV2, NO_OPPONENT
from mcts.value_net import load_checkpoint, make_values_many

__all__ = ["MatchAwareEVPlayer", "load_value", "make_match_policy"]


def load_value(path, device: str = "cpu"):
    net, encoder, _extra = load_checkpoint(path, device=device)
    return net, encoder


class MatchAwareEVPlayer:
    """See the module docstring."""

    def __init__(self, net, encoder: SetEncoderV2, *, device: str = "cpu", budget: str = "fast",
                 seed: int = 0, epsilon: float = 0.0, stats=None, n_worlds: Optional[int] = None,
                 top_k: Optional[int] = None, swallow: bool = False, name: str = "ev+V",
                 value_fn_leaf_only: bool = False):
        import player as P                      # W3
        self.values_many = make_values_many(net, encoder, device)
        self.net, self.encoder, self.device = net, encoder, device
        self._opp = NO_OPPONENT
        self.n_calls = 0
        self.n_errors = 0
        self.swallow = swallow
        self.match = None
        self.player: Optional[int] = None
        self.name = name
        # W-LEAF passthrough (Phase 5 rev 2 V2 round): isolate "V at the expectimax leaf
        # only" (EVPlayer.value_fn_leaf_only) from "argmax-V as the SHOP/BOOSTER_OPEN/
        # BLIND_SELECT policy too" -- see player.py's EVPlayer.__init__ docstring. Default
        # False: identical to every pre-existing caller of this wrapper.
        self._kw = dict(budget=budget, seed=int(seed), epsilon=float(epsilon), stats=stats,
                        n_worlds=n_worlds, top_k=top_k, value_fn_leaf_only=bool(value_fn_leaf_only))
        self._P = P
        self._seats: dict = {}
        self.ev = self._make(int(seed))

    def _make(self, seed: int):
        kw = dict(self._kw)
        kw["seed"] = seed
        kw["name"] = self.name
        return self._P.EVPlayer(self.value_fn, **{k: v for k, v in kw.items() if v is not None or k == "stats"})

    # ── the closure ──
    def value_fn(self, game) -> float:
        self.n_calls += 1
        try:
            return float(self.values_many([(game, self._opp)])[0])
        except Exception:
            self.n_errors += 1
            if not self.swallow:
                raise
            return 0.0

    def values(self, games) -> list:
        """Batched convenience (same opponent view) — for the advisor / evals."""
        return [float(x) for x in self.values_many([(g, self._opp) for g in games])]

    # ── binding ──
    def bind(self, match, player: int) -> None:
        self.match, self.player = match, int(player)
        self.refresh()

    def refresh(self) -> None:
        if self.match is not None and self.player is not None:
            self._opp = opponent_view(self.match, self.player)

    # ── Player protocol ──
    def act(self, game) -> dict:
        self.refresh()
        return self.ev.act(game)

    def reset(self) -> None:
        self.ev.reset()
        for e in self._seats.values():
            e.reset()

    def explain(self, game) -> list:
        self.refresh()
        return self.ev.explain(game)

    # ── match policy ──
    def policy(self) -> Callable:
        """``(match, player, acts) -> action`` with the opponent view refreshed from the
        match passed in on every call; one EVPlayer per player index."""
        def pol(match, p, acts):
            self.match, self.player = match, int(p)
            self._opp = opponent_view(match, int(p))
            ev = self._seats.get(p)
            if ev is None:
                ev = self._seats[p] = self._make(self._kw["seed"] * 2 + int(p))
            return ev.act(match.games[p])
        pol.reset = self.reset                      # type: ignore[attr-defined]
        pol.player = self                            # type: ignore[attr-defined]
        return pol


def make_match_policy(net, encoder, **kw) -> Callable:
    return MatchAwareEVPlayer(net, encoder, **kw).policy()
