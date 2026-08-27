"""
Test helpers: a random-legal MLBMatch driver through the MatchLogger hook contract.
Hand-rolled (mirrors replay/tests/_helpers.py) so ghost's tests never import another
package's test tree.
"""
from __future__ import annotations

import random

from .._bootstrap import MLBMatch
from replay.log import MatchLogger


class RandomLegalPlayer:
    """Uniformly random among legal actions; deterministic in ``seed``."""

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)

    def act(self, game) -> dict:
        acts = game.legal_actions()
        if not acts:
            return {"type": "advance"}
        return self._rng.choice(acts)


def run_logged_match(path, seed, deck_key="b_red", stake=1, lives=4,
                     max_steps=2000, sig_every=50, player_seeds=(0, 1), meta=None) -> dict:
    match = MLBMatch(seed=seed, deck_key=deck_key, stake=stake, lives=lives)
    players = [RandomLegalPlayer(seed=s) for s in player_seeds]
    mlog = MatchLogger(path, sig_every=sig_every)
    mlog.begin(match, meta=meta or {})
    while not match.done and match.steps < max_steps:
        p = match.current_player()
        if p is None:
            break
        action = players[p].act(match.games[p])
        match.step(p, action)
        mlog.step(match, p, action)
    return mlog.end(match, outcome={"winner": match.winner, "steps": match.steps})
