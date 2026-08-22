"""
players.py — the tournament's Player protocol + adapters.

``Player.act(game) -> action`` is the whole contract; ``reset()`` is optional and is called by
``Tournament.run()`` before every run so a Player object can be constructed once and reused
across repeated ``Tournament`` runs without carrying state forward (the determinism tests rely
on this).  ``game`` is always a single ``BalatroGame`` (``ruleset="mlb"``) — the tournament
runner drives N of these independently; a Player never sees another agent's game.

Adapters here wrap ``mp/scripts/mlb_match_demo.py``'s ``ScriptedPlayer`` / ``make_policy``
(imported, never copied, per the brief) and add a uniformly-random-legal player.  Both are
dataclasses/objects so a heterogeneous population is just "many differently-parameterized
instances" (design doc §6: the population must be heterogeneous or the N x N matrix
degenerates).  ``MCTSPlayer`` is a clearly-marked placeholder — W1/W3 plug in there; this
module never imports ``mp.agent``.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable

from .bootstrap import mlb_match_demo as _demo

# Re-exported (not copied): the demo's ScriptedPlayer spec dataclass + its two named presets.
ScriptedPlayer = _demo.ScriptedPlayer
OPENER = _demo.OPENER
REROLLER = _demo.REROLLER

__all__ = [
    "Player", "ScriptedPlayer", "OPENER", "REROLLER",
    "ScriptedPlayerAdapter", "RandomLegalPlayer", "MCTSPlayer",
    "default_population", "identical_population", "make_scripted",
]


@runtime_checkable
class Player(Protocol):
    """The only contract the tournament runner needs from an agent."""

    def act(self, game: Any) -> dict:
        """Return one action from ``game.legal_actions()`` (or anything BalatroGame.step()
        accepts as a no-op if ``game.legal_actions()`` is empty — the runner never calls
        ``act`` in that state, but adapters are defensive anyway)."""
        ...

    def reset(self) -> None:
        """Optional: called by Tournament.run() before every run.  Adapters that hold no
        cross-run state may leave this a no-op."""
        ...


class _SoloShim:
    """Fakes ``MLBMatch``'s ``.games`` indexing so ``mlb_match_demo.make_policy``'s
    ``policy(match, player, legal_actions)`` signature runs unmodified against ONE game
    (player index is always 0 — there is no second game to reference)."""

    __slots__ = ("games",)

    def __init__(self, game):
        self.games = [game]


class ScriptedPlayerAdapter:
    """Adapts an ``mlb_match_demo.ScriptedPlayer`` spec to the tournament ``Player``
    protocol.  Heterogeneity is just different ``ScriptedPlayer`` field values (hand style,
    rerolls per visit, buy/pack behaviour) — see ``default_population`` below."""

    def __init__(self, spec: ScriptedPlayer):
        self.spec = spec
        self._policy = _demo.make_policy(spec)

    def act(self, game) -> dict:
        acts = game.legal_actions()
        if not acts:
            return {"type": "advance"}
        return self._policy(_SoloShim(game), 0, acts)

    def reset(self) -> None:
        pass   # stateless: per-visit bookkeeping lives on the game object (`_w4_visit`), not here

    def __repr__(self) -> str:
        return f"ScriptedPlayerAdapter({self.spec.name!r})"


@dataclass
class RandomLegalPlayer:
    """Uniformly random among legal actions.  Heterogeneity hook = the per-agent ``seed``:
    two instances with different seeds diverge from their very first shop, even against the
    same seed and starting deck."""

    seed: int = 0
    name: str = "random"

    def __post_init__(self):
        self._rng = random.Random(self.seed)

    def act(self, game) -> dict:
        acts = game.legal_actions()
        if not acts:
            return {"type": "advance"}
        return self._rng.choice(acts)

    def reset(self) -> None:
        self._rng = random.Random(self.seed)

    def __repr__(self) -> str:
        return f"RandomLegalPlayer(seed={self.seed})"


class MCTSPlayer:
    """Placeholder.  W1 (agent fork, ``mp/agent/mcts``) and W3 (batched inference / tree
    reuse) are concurrent workstreams building the search agent; this module deliberately
    does NOT import ``mp.agent`` (owned by them, still in flight).  Wire this up once that
    package hands off: ``act(game)`` should run search from ``game`` (a live ``BalatroGame``,
    ``ruleset='mlb'``, possibly mid-Nemesis with ``pvp_opponent_score`` unset — see
    TOURNAMENT_NOTES.md "how the MCTS player will plug in") and return one of
    ``game.legal_actions()``."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("W1/W3 plug in here")

    def act(self, game) -> dict:
        raise NotImplementedError("W1/W3 plug in here")

    def reset(self) -> None:
        raise NotImplementedError("W1/W3 plug in here")


def make_scripted(**kwargs) -> ScriptedPlayerAdapter:
    """Convenience: ``make_scripted(name=..., hand="greedy", rerolls_per_visit=2, ...)``."""
    return ScriptedPlayerAdapter(ScriptedPlayer(**kwargs))


def default_population(n: int, base_seed: int = 0) -> list:
    """A heterogeneous population of ``n`` players for the tournament smoke tests / CLI
    default: a rotating mix of scripted specs (varying hand style, reroll propensity, buy /
    pack behaviour — the design doc's heterogeneity requirement) plus random-legal players
    each with a distinct seed.  Deterministic in ``n`` and ``base_seed``."""
    specs = [
        dict(hand="greedy", rerolls_per_visit=0, buy_slot0=False, open_pack_slot=0),
        dict(hand="greedy", rerolls_per_visit=1, buy_slot0=True, open_pack_slot=None),
        dict(hand="greedy", rerolls_per_visit=2, buy_slot0=True, open_pack_slot=1),
        dict(hand="greedy", rerolls_per_visit=0, buy_slot0=True, open_pack_slot=None, buy_voucher=True),
        dict(hand="greedy_until", weak_from_ante=5, rerolls_per_visit=1, buy_slot0=False, open_pack_slot=0),
    ]
    out = []
    for i in range(n):
        if i % 3 == 2:
            out.append(RandomLegalPlayer(seed=base_seed + i, name=f"random{i}"))
        else:
            spec = dict(specs[i % len(specs)])
            spec["name"] = f"scripted{i}"
            out.append(make_scripted(**spec))
    return out


def identical_population(n: int, spec: Optional[ScriptedPlayer] = None) -> list:
    """``n`` copies of the SAME policy (used by the degeneracy test: this must collapse the
    N x N matrix to a high tie fraction)."""
    spec = spec or OPENER
    return [ScriptedPlayerAdapter(spec) for _ in range(n)]
