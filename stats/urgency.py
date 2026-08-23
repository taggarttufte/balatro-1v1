"""
urgency.py -- "how badly does this build need to improve before the next blind(s)" for the
W4 decision-statistics module (Phase 5 rev 2, mp/docs/PHASE5_BRIEF_2026-08.md).

``urgency(game) in [0, 1]`` feeds ``decide.py``: a needed improvement now is worth more than
the same improvement bought comfortably ahead, and (under MLB) worth much more with few
lives left or a Nemesis blind on deck (losing ANY blind costs a life under MLB -- see
``play_sp_mlb``'s note in mp/eval/common.py -- so life pressure is not Nemesis-only).

Method (documented, simple by design): sample a handful of hands from the deck's
COMPOSITION (never its shuffle order -- ``hit.sample_hand_scores``, itself a thin wrapper
around the same side-effect-free dry-run pattern as ``card_selection.HypotheticalScorer``),
average their score, multiply by the hands available this round, and compare that projected
total to the next blind's chip target. This is a W0-heuristic-shaped proxy (same spirit as
``mcts/heuristic.py``'s ``HandHeuristic``, which cannot be reused directly here: it scores
``game.hand``, and in ``SHOP`` / ``BOOSTER_OPEN`` that list still holds the PREVIOUS blind's
cards -- game.py:1431-1435 -- so it is meaningless as a forward-looking estimate).
"""
from __future__ import annotations

from dataclasses import dataclass

import _bootstrap  # noqa: F401
from balatro_sim.constants import blind_base_chips, MLB_STARTING_LIVES

import hit as hitmod


def next_blind_info(game) -> tuple[int, int]:
    """``(ante, blind_idx)`` of the blind about to be selected, from a ``SHOP`` state.

    ``game.blind_idx`` / ``game.ante`` while in ``SHOP`` still describe the blind just
    finished for Small/Big (``_advance_blind`` only increments ``blind_idx`` on
    ``leave_shop``, game.py:1995-2010); the ONE exception is a just-defeated Boss, where
    ``_end_round`` already bumped ``self.ante`` (game.py:1966-1973) before entering the shop,
    while ``blind_idx`` is still 2 until ``leave_shop``. So: after a Boss shop the next blind
    is (``game.ante`` [already new], Small); otherwise it is (``game.ante``, ``blind_idx+1``).
    """
    if game.blind_idx == 2:
        return game.ante, 0
    return game.ante, game.blind_idx + 1


def is_next_blind_pvp(game, ante: int, blind_idx: int) -> bool:
    return bool(getattr(game, "mlb", False) and blind_idx == 2
               and ante >= getattr(game, "pvp_start_round", 2))


def next_blind_chip_target(game) -> float:
    """The next blind's chip target, or (under MLB, a Nemesis) a documented proxy for it: the
    real Nemesis target is the opponent's live score, unknowable from the shop, so the
    vanilla boss-blind chip formula at that ante stands in -- the same "external, calibration-
    free" idea ``mp/eval/common.py::external_vanilla_big_blind_target`` uses."""
    ante, blind_idx = next_blind_info(game)
    return float(blind_base_chips(ante, blind_idx, game.blind_scaling) * game.ante_scaling)


@dataclass(frozen=True)
class UrgencyResult:
    urgency: float
    shortfall: float          # 1 - projected_total/target, clipped to [0, 1]
    life_pressure: float
    is_nemesis_next: float    # 1.0 / 0.0 (kept float so it composes with the blend below)
    projected_total: float
    target: float
    mean_hand_score: float


def compute(game, n_hand_samples: int = 5) -> UrgencyResult:
    scores = hitmod.sample_hand_scores(game, n_hand_samples, n_cards=5, seed_extra="urgency")
    mean_score = (sum(scores) / len(scores)) if scores else 0.0
    hands = max(1, getattr(game, "base_hands", 4))
    projected_total = mean_score * hands
    target = next_blind_chip_target(game)
    shortfall = max(0.0, min(1.0, 1.0 - projected_total / max(1.0, target)))

    ante, blind_idx = next_blind_info(game)
    nemesis = 1.0 if is_next_blind_pvp(game, ante, blind_idx) else 0.0
    if getattr(game, "mlb", False):
        life_pressure = max(0.0, min(1.0, (MLB_STARTING_LIVES - game.lives) / MLB_STARTING_LIVES))
    else:
        life_pressure = 0.0

    urgency = max(0.0, min(1.0, 0.7 * shortfall + 0.2 * life_pressure + 0.1 * nemesis))
    return UrgencyResult(urgency=urgency, shortfall=shortfall, life_pressure=life_pressure,
                        is_nemesis_next=nemesis, projected_total=projected_total,
                        target=target, mean_hand_score=mean_score)


__all__ = ["next_blind_info", "is_next_blind_pvp", "next_blind_chip_target", "UrgencyResult", "compute"]
