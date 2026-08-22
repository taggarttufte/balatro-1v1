"""
outcome.py — the terminal / outcome signal, as a PARAMETER of the search and of self-play.

The balatro-mcts original hardcoded two assumptions in `search.py` and `train/agent.py`:

    _is_win(game)  ==  game.state == GAME_OVER and game.ante > 8
    _shaped_z      ==  (blinds_completed + chip_ratio) / 24

Both are false under Major League Balatro (`mp/engine/MLB_NOTES.md`): MLB is **endless**
(there is no ante-8 win; `game.ante > 8` just means the run is going well), the run ends
when `lives` hits 0 or the opponent's does (`game.match_won`), and the thing a Nemesis
blind actually produces is a *score margin against an external opponent*, which the
engine cannot see at all — W2's tournament runner and W4's eval harness supply it.

So the signal is an object with these questions:

    is_terminal(game)   descent stops and the value is final
    is_stuck(game)      no legal actions but NOT final (MLB `PVP_WAIT`, or readied at a
                        Nemesis waiting for `startBlind`) — value with the pending estimate
    is_win(game)        the unshaped win flag (logging / eval, never the training label)
    value(game)         a scalar in [0, 1] for backup and for the training label z

Implementations
    VanillaOutcome   SP: win iff GAME_OVER past ante 8; shaped by fraction of 24 blinds
    MLBOutcome       MLB: win iff `game.match_won`; shaped by lives left + blind progress
    ExternalOutcome  the caller supplies value(game) — e.g. W2's N x N log-score margin
                     at the Nemesis, or W4's paired margin vs a reference player. This is
                     the hook the Phase 3 brief asks for.

`default_outcome_for(game)` picks Vanilla/MLB off `game.mlb`, so nothing has to be
threaded through by callers that do not care.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

from balatro_sim.game import BalatroGame, State

# 8 antes x 3 blinds. Progress denominator for the vanilla shaped label.
TOTAL_BLINDS = 24


def is_stuck_state(game: BalatroGame) -> bool:
    """True when the game has no legal actions but is not over — always an MLB
    coordination state (`mp/engine/MLB_NOTES.md` rules 1.2e / 1.3f):

      * `State.PVP_WAIT` — hands exhausted at the Nemesis, waiting for the opponent
      * BLIND_SELECT with `pvp_ready` — readied, waiting for the server's `startBlind`

    Checked by state rather than by `legal_actions()` because `legal_actions()` is the
    combinatorial enumeration and costs orders of magnitude more than this.
    """
    if game.state is State.PVP_WAIT:
        return True
    return game.state is State.BLIND_SELECT and bool(getattr(game, "pvp_ready", False))


def blinds_completed(game: BalatroGame) -> int:
    return (game.ante - 1) * 3 + game.blind_idx


class OutcomeFn(Protocol):
    name: str

    def is_terminal(self, game: BalatroGame) -> bool: ...
    def is_stuck(self, game: BalatroGame) -> bool: ...
    def is_win(self, game: BalatroGame) -> bool: ...
    def value(self, game: BalatroGame) -> float: ...


@dataclass
class VanillaOutcome:
    """Single-player vanilla Balatro: a run is a win iff GAME_OVER was reached by
    clearing ante 8."""
    name: str = "vanilla"
    win_value: float = 1.0
    loss_value: float = 0.0
    shaped: bool = True

    def is_terminal(self, game: BalatroGame) -> bool:
        return game.state is State.GAME_OVER

    def is_stuck(self, game: BalatroGame) -> bool:
        return is_stuck_state(game)

    def is_win(self, game: BalatroGame) -> bool:
        return game.state is State.GAME_OVER and game.ante > 8

    def value(self, game: BalatroGame) -> float:
        if self.is_win(game):
            return self.win_value
        if not self.shaped:
            return self.loss_value
        # Continuous progress in [0, 1]: fraction of the run's 24 blinds cleared, plus
        # partial chips inside whatever blind ended it. Dense gradations keep the value
        # head from collapsing to "0 everywhere" before the first win is ever seen.
        target = max(1, game.current_blind.chips_target)
        chip_ratio = min(1.0, game.chips_scored / target)
        z = (blinds_completed(game) + chip_ratio) / TOTAL_BLINDS
        return float(min(1.0, max(0.0, z)))


@dataclass
class MLBOutcome:
    """Major League Balatro. No ante-8 win: the run ends when this player runs out of
    lives (`lives <= 0` -> GAME_OVER) or when the opponent does (`match_won`). Endless
    antes, so blind progress is normalised against `horizon_antes` and saturates rather
    than exceeding 1.

        value = 1.0                                              if match_won
              = w * lives_left/starting + (1-w) * blind_progress otherwise

    Monotone in both terms; a 0-lives loss still credits how far the run got, which is
    the whole reason the vanilla label was shaped in the first place.
    """
    name: str = "mlb"
    starting_lives: int = 4
    horizon_antes: int = 8
    win_value: float = 1.0
    lives_weight: float = 0.5

    def is_terminal(self, game: BalatroGame) -> bool:
        return game.state is State.GAME_OVER

    def is_stuck(self, game: BalatroGame) -> bool:
        return is_stuck_state(game)

    def is_win(self, game: BalatroGame) -> bool:
        return bool(getattr(game, "match_won", False))

    def value(self, game: BalatroGame) -> float:
        if self.is_win(game):
            return self.win_value
        lives = max(0, getattr(game, "lives", 0))
        life_frac = min(1.0, lives / max(1, self.starting_lives))
        progress = min(1.0, blinds_completed(game) / max(1, 3 * self.horizon_antes))
        w = self.lives_weight
        return float(min(1.0, max(0.0, w * life_frac + (1.0 - w) * progress)))


@dataclass
class ExternalOutcome:
    """Outcome supplied from outside the single game — the W2 / W4 hook.

    `value_fn(game) -> float in [0, 1]` is whatever the driver knows and the game does
    not: the N x N outcome at a Nemesis, a log-score margin against a fixed target, a
    match verdict from `MLBMatch`. `terminal_fn` / `stuck_fn` / `win_fn` default to the
    `base` outcome's (MLB unless told otherwise), so a driver only has to supply value.

    Convenience: `ExternalOutcome.from_margin(fn, scale=...)` wraps a signed margin
    (e.g. log10(my_score) - log10(their_score)) through a logistic into [0, 1].
    """
    value_fn: Callable[[BalatroGame], float]
    base: OutcomeFn = field(default_factory=lambda: MLBOutcome())
    terminal_fn: Optional[Callable[[BalatroGame], bool]] = None
    stuck_fn: Optional[Callable[[BalatroGame], bool]] = None
    win_fn: Optional[Callable[[BalatroGame], bool]] = None
    name: str = "external"

    def is_terminal(self, game: BalatroGame) -> bool:
        fn = self.terminal_fn or self.base.is_terminal
        return bool(fn(game))

    def is_stuck(self, game: BalatroGame) -> bool:
        fn = self.stuck_fn or self.base.is_stuck
        return bool(fn(game))

    def is_win(self, game: BalatroGame) -> bool:
        if self.win_fn is not None:
            return bool(self.win_fn(game))
        return self.value(game) > 0.5

    def value(self, game: BalatroGame) -> float:
        return float(min(1.0, max(0.0, self.value_fn(game))))

    @staticmethod
    def from_margin(margin_fn: Callable[[BalatroGame], float], scale: float = 1.0,
                    **kwargs) -> "ExternalOutcome":
        def _v(game: BalatroGame) -> float:
            return margin_to_value(margin_fn(game), scale=scale)
        return ExternalOutcome(value_fn=_v, **kwargs)


def margin_to_value(margin: float, scale: float = 1.0) -> float:
    """Signed margin -> [0, 1] through a logistic. margin 0 -> 0.5 (a tie, which under
    the MLB server rule costs nobody a life)."""
    return 1.0 / (1.0 + math.exp(-margin / max(scale, 1e-9)))


def default_outcome_for(game: BalatroGame) -> OutcomeFn:
    """MLBOutcome for a `ruleset="mlb"` game, VanillaOutcome otherwise."""
    if getattr(game, "mlb", False):
        return MLBOutcome(starting_lives=max(1, getattr(game, "lives", 4) or 4))
    return VanillaOutcome()
