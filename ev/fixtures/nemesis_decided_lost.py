"""
nemesis_decided_lost.py -- W-PVP fixture (2026-08-26): the extraction pivot at a Nemesis.

The extraction layer used to be OFF at every Nemesis, unconditionally ("no unused-hand
money at a PvP blind, and every hand is played anyway").  That is right while the RACE IS
LIVE and wrong once it is over: a Nemesis always reaches Cash Out whether it is won or lost
(MLB_NOTES.md 1.4b), the discard money row is NOT patched out at a PvP blind (1.4a), and a
player who cannot reach the opponent's score is holding hands and discards that are worth
nothing to the race and something real in dollars.  ``PVP_NOTES.md`` section 5.

``build()`` -- **decided-LOST.**  Player 0 is inside the real ante-2 Nemesis with a plain
ante-2 board (no jokers) and four hands; the opponent's revealed live score is 10**7, which
no line on this board reaches.  Two Purple-sealed junk cards (5D/6D, indices 5/6) sit
outside every structural play: discarding them creates a Tarot each (card.lua:2242-2268)
at zero cost to a race that is already lost.  Player 0 has 3 lives, so the loss is a life,
not ``loseGame`` -- the dollars survive to a Cash Out.

``build_control()`` -- **the LIVE race.**  Byte-identical board and hand; the only change is
the opponent's revealed score, set to something this board can chase.  The gate must stay
shut: with the race live the ranking has to be bit-identical to the ranking of a player
with ``pvp_extract=False``.

Both states are reached through the real engine and match APIs -- a real MLB match walked to
the real ante-2 Nemesis blind (``debug_win_blind`` for the regular blinds, the match's own
``readyBlind`` -> ``startBlind``), then the same direct attribute writes ``_probe_common``
documents.  The opponent's score is set on ``games[1].chips_scored`` and RELAYED by
``MLBMatch.sync()`` through the real ``set_pvp_info`` path, so player 0 learns it exactly
the way ``enemyInfo`` delivers it in the mod.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent            # ev/fixtures
_EV = _HERE.parent                                  # ev
for _p in (str(_EV), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _bootstrap  # noqa: E402,F401
from _bootstrap import MLBMatch, State  # noqa: E402
from _probe_common import set_hand  # noqa: E402

__all__ = ["SEED", "SPEC", "OPP_SCORE_LOST", "OPP_SCORE_LIVE", "PVP_ANTE",
           "build", "build_control", "to_nemesis_hand"]

SEED = "NEMLOST1"
PVP_ANTE = 2
#: Unreachable on a plain ante-2 board: the race is over before the decision starts.
OPP_SCORE_LOST = 10 ** 7
#: Reachable: an ante-2 Small-blind-sized number, so the race is genuinely live.
OPP_SCORE_LIVE = 300
SPEC = [(14, "Spades"), (14, "Hearts"), (2, "Clubs"), (3, "Clubs"),
        (4, "Clubs"), (5, "Diamonds"), (6, "Diamonds"), (7, "Diamonds")]


def _advance(m: MLBMatch, p: int) -> None:
    g = m.games[p]
    if g.state == State.ROUND_EVAL:
        m.step(p, {"type": "advance"})
    elif g.state == State.BOOSTER_OPEN:
        m.step(p, {"type": "skip_booster"})
    elif g.state == State.SHOP:
        m.step(p, {"type": "leave_shop"})
    else:
        raise RuntimeError(f"cannot auto-advance from {g.state}")


def to_nemesis_hand(seed: str = SEED, *, ante: int = PVP_ANTE, lives: int = 3,
                    pvp_protocol: str = "trailer_compelled") -> MLBMatch:
    """A real ``MLBMatch`` with BOTH players inside the ante-``ante`` Nemesis blind,
    ``SELECTING_HAND``, nothing scored.  Regular blinds are cleared with the engine's own
    ``debug_win_blind`` harness helper (touches no RNG stream)."""
    m = MLBMatch(seed=seed, lives=lives, pvp_protocol=pvp_protocol)
    for p in (0, 1):
        g = m.games[p]
        while not (g.ante == ante and g.current_blind.is_pvp
                   and g.state == State.BLIND_SELECT):
            if g.state == State.BLIND_SELECT:
                m.step(p, {"type": "play_blind"})
                g.debug_win_blind()
                m.sync()
                _advance(m, p)                       # cash out -> shop / pack
                while g.state == State.BOOSTER_OPEN:
                    _advance(m, p)
                _advance(m, p)                       # leave the shop
                while g.state == State.BOOSTER_OPEN:
                    _advance(m, p)
            else:
                _advance(m, p)
            if g.ante > ante:
                raise RuntimeError("overshot the Nemesis ante")
    for p in (0, 1):
        m.step(p, {"type": "play_blind"})             # readyBlind x2 -> startBlind
    if not (m.pvp_active and all(g.state == State.SELECTING_HAND for g in m.games)):
        raise RuntimeError("the Nemesis did not start for both players")
    return m


def _base(seed: str, opp_score: int, *, seals: bool) -> MLBMatch:
    m = to_nemesis_hand(seed)
    g0 = m.games[0]
    set_hand(g0, SPEC)
    g0.consumable_hand = []                # both consumable slots free for the Tarots
    g0.jokers = []                         # a plain board: nothing inflates the ratio
    g0.dollars = max(g0.dollars, 10)
    if seals:
        g0.hand[5].seal = "Purple"
        g0.hand[6].seal = "Purple"
    m.games[1].chips_scored = int(opp_score)
    m.sync()                               # enemyInfo relay: the real set_pvp_info path
    assert g0.pvp_opponent_score == int(opp_score)
    return m


def build(seed: str = SEED) -> MLBMatch:
    """Decided-LOST: the opponent is out of reach and the Purple seals are free money."""
    return _base(seed, OPP_SCORE_LOST, seals=True)


def build_control(seed: str = SEED) -> MLBMatch:
    """The LIVE race: same board, a reachable opponent score.  The gate stays shut."""
    return _base(seed, OPP_SCORE_LIVE, seals=True)
