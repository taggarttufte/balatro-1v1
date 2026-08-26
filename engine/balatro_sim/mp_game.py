"""
mp_game.py — RETIRED (Phase 2 W1, 2026-08-21).  Use ``balatro_sim.mlb_match.MLBMatch``.

The V8-era ``MultiplayerBalatro`` coordinator encoded rules that the Multiplayer mod's
source (v0.5.2) and its server contradict: it called the regular-blind life loss a
"house rule" (it is the mod's default, ``death_on_round_loss = true``), resolved a PvP
tie as "no lives lost, both keep playing" without the server's exhaustion/early-end
logic, paid comeback money to the PvP loser immediately instead of at the next Cash Out
(``4 x cumulative lives lost``), and needed a "revive" hack because the underlying games
could not lose a blind and go on.  All of that is now native to ``BalatroGame`` with
``ruleset="mlb"`` plus ``MLBMatch`` (see ``engine/MLB_NOTES.md``).

Kept as an import shim so old call sites fail loudly with a pointer rather than silently
running the wrong rules.
"""
from __future__ import annotations

from .mlb_match import MLBMatch, MLBMatchState, DEFAULT_LIVES, COMEBACK_MONEY_PER_LIFE  # noqa: F401

__all__ = ["MLBMatch", "MLBMatchState", "DEFAULT_LIVES", "COMEBACK_MONEY_PER_LIFE", "MultiplayerBalatro"]


def MultiplayerBalatro(*args, **kwargs):
    raise ImportError(
        "balatro_sim.mp_game.MultiplayerBalatro was retired in Phase 2 (wrong rules); "
        "use balatro_sim.mlb_match.MLBMatch (see engine/MLB_NOTES.md)."
    )
