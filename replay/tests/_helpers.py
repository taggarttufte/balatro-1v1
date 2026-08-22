"""
Shared test helpers: a self-contained, engine-only random-legal player (mirrors
mp/tournament/players.py::RandomLegalPlayer, hand-rolled here so mp/replay's tests never
import the frozen mp/tournament package) and small drivers that exercise the EXACT hook
contract documented in REPLAY_NOTES.md -- these ARE the "≤3 lines" the contract promises,
proven by being the thing the tests actually run.
"""
from __future__ import annotations

import random

from .._bootstrap import BalatroGame, MLBMatch, State
from ..log import MatchLogger, TrajectoryLogger

# 20 of the 126 oracle ground-truth seeds (mp/oracle/ground_truth/*.json stems) -- any
# BalatroGame-constructible seed string works; these are just real, already-verified seeds.
SEEDS = [
    "11111111", "1558AXDL", "15H9Z3IY", "1KV4W6YS", "1MD1YZ9T",
    "28V7DD4H", "29DAQVG1", "29Y3L4S9", "29ZSW8MY", "2BRGI767",
    "2CP4KSXZ", "2GHBLJD9", "2H9N3ISZ", "2K9H9HN", "34JCNMPA",
    "3SZ71111", "41Y71M6E", "46Y8UZEG", "4H2L46CE", "4K8A9QER",
]


class RandomLegalPlayer:
    """Uniformly random among legal actions; deterministic in ``seed``."""

    def __init__(self, seed: int = 0):
        self.seed = seed
        self._rng = random.Random(seed)

    def act(self, game) -> dict:
        acts = game.legal_actions()
        if not acts:
            return {"type": "advance"}
        return self._rng.choice(acts)


def run_logged_episode(path, seed, ruleset="vanilla", deck_key="b_red", stake=1,
                        max_steps=600, sig_every=10, player_seed=0, meta=None) -> dict:
    """Drives one BalatroGame with a RandomLegalPlayer through TrajectoryLogger using
    EXACTLY the 3-call hook contract (begin / step-per-transition / end)."""
    game = BalatroGame(seed=seed, deck_key=deck_key, stake=stake, ruleset=ruleset)
    player = RandomLegalPlayer(seed=player_seed)
    log = TrajectoryLogger(path, sig_every=sig_every)

    log.begin(game, meta=meta or {})
    n = 0
    while game.state != State.GAME_OVER and n < max_steps:
        action = player.act(game)
        game.step(action)
        log.step(game, action)
        n += 1
    won = game.match_won if game.mlb else (game.ante > 8 and game.state == State.GAME_OVER)
    return log.end(game, outcome={
        "won": bool(won), "final_ante": game.ante, "steps": n,
        "stop_reason": "game_over" if game.state == State.GAME_OVER else "max_steps",
    })


def run_logged_match(path, seed, deck_key="b_red", stake=1, lives=4,
                      max_steps=2000, sig_every=10, player_seeds=(0, 1), meta=None) -> dict:
    """Drives one MLBMatch with two RandomLegalPlayers through MatchLogger using EXACTLY the
    3-call hook contract, alternating via ``current_player()`` as the tournament / MCTSPlayer
    integration will."""
    match = MLBMatch(seed=seed, deck_key=deck_key, stake=stake, lives=lives)
    players = [RandomLegalPlayer(seed=s) for s in player_seeds]
    mlog = MatchLogger(path, sig_every=sig_every)

    mlog.begin(match, meta=meta or {})
    n = 0
    while not match.done and n < max_steps:
        p = match.current_player()
        if p is None:
            break
        action = players[p].act(match.games[p])
        match.step(p, action)
        mlog.step(match, p, action)
        n += 1
    return mlog.end(match, outcome={"winner": match.winner, "steps": n})
