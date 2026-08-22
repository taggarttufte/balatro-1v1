"""
log.py — TrajectoryLogger (one BalatroGame) + MatchLogger (one MLBMatch): the whole logging
hook a caller wires in.  See REPLAY_NOTES.md "Hook contract" for the exact call sites in
mp/agent/train/loop.py::run_episode, the future W2 tournament-driven loop, and
mp/tournament/runner.py::Tournament / mp/engine/balatro_sim/mlb_match.py::MLBMatch.

Contract (3 calls total, ``step`` reused once per actual state transition):

    log = TrajectoryLogger(path)
    log.begin(game, meta={...})
    ...
    game.step(action)          # <- caller's existing line, unchanged
    log.step(game, action)     # <- 1 line added right after every game.step() call
    ...
    log.end(game, outcome={...})

Everything is buffered in memory between begin() and end() and serialized as ONE JSONL line
at end() (append-only, UTF-8, `path` opened in "a" mode) -- safe to call step() thousands of
times per episode.  ``sig_every`` (default 10) controls how often a ``state_signature()``
digest is captured; index 0 ("start", before any action) and the final state are always
captured regardless.
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

from ._util import match_sig_digest, sig_digest, summarize

FORMAT_VERSION = 1


def _append_jsonl(path: str, line: dict) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(line, sort_keys=True) + "\n")


class TrajectoryLogger:
    """Logs one ``BalatroGame`` episode (vanilla, or MLB solo with ``pvp_solo=True``) per
    ``begin()``/``end()`` pair, as one JSONL line."""

    def __init__(self, path: str, sig_every: int = 10):
        self.path = str(path)
        self.sig_every = max(1, int(sig_every))
        self._active = False
        self._reset()

    def _reset(self) -> None:
        self._seed = None
        self._deck_key = None
        self._stake = None
        self._ruleset = None
        self._lives_start = None
        self._meta: dict = {}
        self._actions: list = []
        self._steps: list = []
        self._sigs: dict = {}

    # -- hook ---------------------------------------------------------------------

    def begin(self, game: Any, meta: Optional[dict] = None) -> None:
        if self._active:
            raise RuntimeError(
                "TrajectoryLogger.begin() called while an episode is already open "
                "(missing a matching end()?)"
            )
        self._active = True
        self._seed = game.seed_str
        self._deck_key = game.deck_key
        self._stake = game.stake
        self._ruleset = game.ruleset
        self._lives_start = game.lives
        self._meta = dict(meta) if meta else {}
        self._actions = []
        self._steps = []
        self._sigs = {"start": sig_digest(game)}

    def step(self, game: Any, action: dict) -> None:
        if not self._active:
            raise RuntimeError("TrajectoryLogger.step() called before begin()")
        idx = len(self._actions)
        self._actions.append(dict(action))
        self._steps.append(summarize(game, idx))
        if idx % self.sig_every == 0:
            self._sigs[str(idx)] = sig_digest(game)

    def end(self, game: Any, outcome: Optional[dict] = None) -> dict:
        if not self._active:
            raise RuntimeError("TrajectoryLogger.end() called before begin()")
        n = len(self._actions)
        self._sigs["final"] = sig_digest(game) if n > 0 else self._sigs["start"]
        final_state = {
            "state": game.state.name,
            "ante": game.ante,
            "lives": game.lives,
            "money": game.dollars,
            "joker_count": len(game.jokers),
            "jokers": [j.key for j in game.jokers],
            "consumables": list(game.consumable_hand),
        }
        line = {
            "v": FORMAT_VERSION,
            "kind": "episode",
            "seed": self._seed,
            "deck_key": self._deck_key,
            "stake": self._stake,
            "ruleset": self._ruleset,
            "lives_start": self._lives_start,
            "meta": self._meta,
            "actions": self._actions,
            "steps": self._steps,
            "sig_every": self.sig_every,
            "signatures": self._sigs,
            "outcome": dict(outcome) if outcome else {},
            "final_state": final_state,
            "tags": [],
        }
        _append_jsonl(self.path, line)
        self._active = False
        self._reset()
        return line


class MatchLogger:
    """Logs one ``MLBMatch`` (two ``BalatroGame``s, ``ruleset='mlb'``) per ``begin()``/
    ``end()`` pair, as one JSONL line.  ``step()`` records ops in the EXACT interleaved order
    ``match.step(player, action)`` was called in -- that order is part of what makes the
    match deterministic (MLB_NOTES.md §3 / mlb_match.py's own docstring), so replay must
    reproduce it, not just each player's own action list independently."""

    def __init__(self, path: str, sig_every: int = 10):
        self.path = str(path)
        self.sig_every = max(1, int(sig_every))
        self._active = False
        self._reset()

    def _reset(self) -> None:
        self._seed = None
        self._deck_key = None
        self._stake = None
        self._lives_start = None
        self._pvp_start_round = None
        self._meta: dict = {}
        self._ops: list = []
        self._steps: list = []
        self._sigs: dict = {}

    def begin(self, match: Any, meta: Optional[dict] = None) -> None:
        if self._active:
            raise RuntimeError(
                "MatchLogger.begin() called while a match is already open "
                "(missing a matching end()?)"
            )
        self._active = True
        g0 = match.games[0]
        self._seed = match.seed_str
        self._deck_key = g0.deck_key
        self._stake = g0.stake
        self._lives_start = match.starting_lives
        self._pvp_start_round = match.pvp_start_round
        self._meta = dict(meta) if meta else {}
        self._ops = []
        self._steps = []
        self._sigs = {"start": match_sig_digest(match)}

    def step(self, match: Any, player: int, action: dict) -> None:
        if not self._active:
            raise RuntimeError("MatchLogger.step() called before begin()")
        idx = len(self._ops)
        self._ops.append({"player": int(player), "action": dict(action)})
        self._steps.append({
            "step": idx,
            "player": int(player),
            "players": [summarize(match.games[0], idx), summarize(match.games[1], idx)],
        })
        if idx % self.sig_every == 0:
            self._sigs[str(idx)] = match_sig_digest(match)

    def end(self, match: Any, outcome: Optional[dict] = None) -> dict:
        if not self._active:
            raise RuntimeError("MatchLogger.end() called before begin()")
        n = len(self._ops)
        self._sigs["final"] = match_sig_digest(match) if n > 0 else self._sigs["start"]
        line = {
            "v": FORMAT_VERSION,
            "kind": "match",
            "seed": self._seed,
            "deck_key": self._deck_key,
            "stake": self._stake,
            "lives_start": self._lives_start,
            "pvp_start_round": self._pvp_start_round,
            "meta": self._meta,
            "ops": self._ops,
            "steps": self._steps,
            "sig_every": self.sig_every,
            "signatures": self._sigs,
            "pvp_log": [list(t) for t in match.pvp_log],
            "outcome": dict(outcome) if outcome else {},
            "final_state": {
                "winner": match.winner,
                "players": [
                    {
                        "lives": g.lives, "ante": g.ante, "money": g.dollars,
                        "joker_count": len(g.jokers),
                        "jokers": [j.key for j in g.jokers],
                    }
                    for g in match.games
                ],
            },
            "tags": {"0": [], "1": []},
        }
        _append_jsonl(self.path, line)
        self._active = False
        self._reset()
        return line
