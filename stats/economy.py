"""
economy.py -- interest arithmetic and true-cost-of-spending for the W4 decision-statistics
module (Phase 5 rev 2, mp/docs/PHASE5_BRIEF_2026-08.md).

The game's interest rule (``constants.INTEREST_RATE`` / ``game.interest_cap``, ported from
``evaluate_round``, game.py:1911): every shop payout, you earn ``$1`` per ``$5`` held
(floor), capped at ``game.interest_cap`` (5 by default; raised by Seed Money / Money Tree
vouchers). This module turns that one rule into a "true cost" for a shop action: the sticker
price PLUS the interest you forfeit by holding less money for the rest of the run.

Everything here is a pure function of ``(game, ...)`` -- no mutation, no RNG, no clone
needed (we only ever READ ``game.dollars`` / ``game.interest_cap`` / ``game.no_interest`` /
``game.ante`` / ``game.blind_idx``).
"""
from __future__ import annotations

from dataclasses import dataclass

import _bootstrap  # noqa: F401  (sys.path + fork-guard; see mp/stats/_bootstrap.py)
from balatro_sim.constants import INTEREST_RATE


def interest_now(game) -> int:
    """``$1`` per ``$5`` held, capped -- the exact formula ``_end_round`` uses
    (game.py:1911), evaluated on the CURRENT balance."""
    if getattr(game, "no_interest", False):
        return 0
    return min(max(0, game.dollars) // INTEREST_RATE, game.interest_cap)


def interest_after(game, spend: float) -> int:
    """Interest on ``max(0, dollars - spend)`` -- ``spend`` may be non-integer (a Row's
    ``cost`` is always a game-legal integer price, but callers may probe fractional deltas)."""
    if getattr(game, "no_interest", False):
        return 0
    remaining = max(0, game.dollars - spend)
    return min(int(remaining) // INTEREST_RATE, game.interest_cap)


def dollars_to_next_tier(game) -> int:
    """How many more dollars would raise this round's interest by ``$1`` (0 once the cap is
    hit). Informational (``Row.details``), not used by ``true_cost``."""
    if getattr(game, "no_interest", False) or interest_now(game) >= game.interest_cap:
        return 0
    next_tier_dollars = (game.dollars // INTEREST_RATE + 1) * INTEREST_RATE
    return next_tier_dollars - game.dollars


def shops_remaining(game, horizon_rounds: int | None = None) -> int:
    """Remaining shop visits this run: one after every blind (Small / Big / Boss-or-Nemesis),
    from THIS shop (inclusive of the current one, since the interest lost by spending now is
    forfeited starting the very next payout) through the ante-8 boss/Nemesis.

    ``game.blind_idx`` / ``game.ante`` while ``game.state == SHOP`` still refer to the blind
    just finished (``_advance_blind`` only increments them on ``leave_shop`` -- game.py:1995-
    2010), so the blind about to be selected is ``blind_idx + 1`` and the shops remaining
    through ante 8 are the ones after it and every later blind.

    ``horizon_rounds``, if given, overrides this with a flat count (mp/ev/player.py and the
    advisor may want a shorter look-ahead than "to ante 8").  MLB is endless past ante 8;
    "to ante 8" is a documented truncation, not a claim the run stops there.
    """
    if horizon_rounds is not None:
        return max(0, int(horizon_rounds))
    remaining_this_ante = max(0, 2 - game.blind_idx)          # blinds AFTER the one just played
    remaining_full_antes = max(0, 8 - game.ante)
    return 1 + remaining_this_ante + remaining_full_antes     # +1: this shop's own next payout


#: Per-round survival probability of the current dollar shortfall: by the NEXT shop, blind
#: income (min $3 reward + $1/unused hand + interest) usually re-crosses the $5 tier a small
#: spend knocked you out of, so the deficit that actually persists shrinks round over round
#: rather than staying flat for the whole horizon. Geometric decay is the simplest one-
#: parameter way to encode "future rounds count for less" -- literally "horizon-discounted"
#: per the brief -- without projecting an actual money curve (V's job, not this module's).
DEFAULT_INTEREST_DECAY = 0.85


def interest_loss(game, cost: float, horizon_rounds: int | None = None,
                  decay: float = DEFAULT_INTEREST_DECAY) -> float:
    """$ of future interest forfeited by spending ``cost`` now, geometrically discounted over
    the horizon: ``per_round * sum_{t=1..H} decay**t`` where ``per_round = interest_now -
    interest_after(cost)`` (Balatro's own interest does not compound -- game.py:1911 pays it
    once off the current balance every shop -- so this discount is purely a persistence
    assumption on the shortfall, not compounding). ``decay=1.0`` recovers the flat
    ``per_round * shops_remaining`` (no persistence decay)."""
    per_round = max(0, interest_now(game) - interest_after(game, cost))
    h = shops_remaining(game, horizon_rounds)
    if per_round <= 0 or h <= 0:
        return 0.0
    if decay >= 1.0:
        geo = float(h)
    else:
        geo = decay * (1.0 - decay ** h) / (1.0 - decay)
    return float(per_round * geo)


def true_cost(game, cost: float, horizon_rounds: int | None = None) -> float:
    """``cost + interest_loss(cost)``. Slot-opportunity cost is NOT modelled (brief: "+ slot
    opportunity cost if you model it" -- left as a documented gap, see STATS_NOTES.md)."""
    return float(cost) + interest_loss(game, cost, horizon_rounds)


def reroll_cost_now(game) -> int:
    """The dollar cost of a reroll RIGHT NOW, honouring free rerolls and the discount --
    exactly what ``shop.reroll_shop`` charges (shop.py:478-481)."""
    if getattr(game, "free_rerolls_remaining", 0) > 0:
        return 0
    return max(0, game.reroll_cost - game.reroll_discount)


@dataclass(frozen=True)
class EconomySnapshot:
    """A cached economy read for one ``decision_table`` call -- computed once, reused by
    every Row so ``interest_now`` etc. are not recomputed per action."""
    dollars: int
    interest_now: int
    interest_cap: int
    dollars_to_next_tier: int
    shops_remaining: int
    reroll_cost_now: int

    @classmethod
    def build(cls, game, horizon_rounds: int | None = None) -> "EconomySnapshot":
        return cls(
            dollars=game.dollars,
            interest_now=interest_now(game),
            interest_cap=game.interest_cap,
            dollars_to_next_tier=dollars_to_next_tier(game),
            shops_remaining=shops_remaining(game, horizon_rounds),
            reroll_cost_now=reroll_cost_now(game),
        )


__all__ = [
    "interest_now", "interest_after", "dollars_to_next_tier", "shops_remaining",
    "interest_loss", "true_cost", "reroll_cost_now", "EconomySnapshot",
]
