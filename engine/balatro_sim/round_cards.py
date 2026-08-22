"""
round_cards.py — G.GAME.current_round.{idol_card, mail_card, ancient_card, castle_card}
as the joker hooks read them.

The real game rolls these four values at run start (game.lua:2385-2389) and again
after EVERY blind (state_events.lua:273-276), whether or not the matching joker is
owned (keys ``idol<ante>`` / ``mail<ante>`` / ``anc<ante>`` / ``cas<ante>``,
common_events.lua:2271-2324). W2 owns the rolling (``generate.start_run`` and
``BalatroGame._round_end_resets`` -> ``game.round_picks``, stored in the
generation layer's front-key format: idol/castle ``'S_A'``, mail ``'A'``, ancient
a suit name). This module converts that into what The Idol / Mail-In Rebate /
Ancient Joker / Castle compare against, and provides the same draw for a context
built without a game (unit tests) so the stream and key are still the real ones.

``ScoreContext.round_cards`` = {"idol": (rank, suit), "mail": rank, "ancient": suit,
"castle": suit}.
"""
from __future__ import annotations

from . import game_keys as _gk

SUITS_LUA_ORDER = ["Spades", "Hearts", "Clubs", "Diamonds"]   # common_events.lua:2306-2312
NAMES = ("idol", "mail", "ancient", "castle")

_SUIT_FROM_LETTER = {"S": "Spades", "H": "Hearts", "C": "Clubs", "D": "Diamonds"}
_RANK_FROM_CHAR = {"T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}


def _rank(ch) -> int:
    if isinstance(ch, int):
        return ch
    return _RANK_FROM_CHAR.get(ch) or int(ch)


def _front(front: str):
    s, r = front.split("_")
    return _rank(r), _SUIT_FROM_LETTER[s]


def from_round_picks(picks: dict) -> dict:
    """``game.round_picks`` (W2, generation-layer keys) -> hook-context dict.
    Missing/None entries fall back to the Lua defaults (Ace of Spades / Ace / Spades)."""
    out = {}
    if not picks:
        return out
    idol = picks.get("idol")
    if idol is not None:
        out["idol"] = _front(idol) if isinstance(idol, str) else tuple(idol)
    mail = picks.get("mail")
    if mail is not None:
        out["mail"] = _rank(mail)
    if picks.get("ancient"):
        out["ancient"] = picks["ancient"]
    castle = picks.get("castle")
    if castle is not None:
        out["castle"] = _front(castle)[1] if isinstance(castle, str) else castle
    return out


def _playing_cards(cards):
    """``G.playing_cards`` minus Stone cards, in ``sort_id`` (creation) order."""
    return sorted((c for c in cards if c.enhancement != "Stone"), key=lambda c: c.id)


def roll_round_cards(prng, ante: int, full_deck, into: dict, names=NAMES) -> dict:
    """The four ``reset_*`` draws for ``ante`` (game order idol, mail, anc, cas) into
    ``into``. Used by ``round_card`` for game-less contexts; the game itself rolls
    through ``BalatroGame._round_end_resets`` (same keys, same pools)."""
    K = _gk.gen.Keys
    cards = _playing_cards(full_deck)
    if "idol" in names:      # reset_idol_card (2271-2287); fallback Ace of Spades
        if cards:
            c, _ = prng.pseudorandom_element(cards, K.idol(ante))
            into["idol"] = (c.rank, c.suit)
        else:
            into["idol"] = (14, "Spades")
    if "mail" in names:      # reset_mail_rank (2289-2303); fallback Ace
        if cards:
            c, _ = prng.pseudorandom_element(cards, K.mail(ante))
            into["mail"] = c.rank
        else:
            into["mail"] = 14
    if "ancient" in names:   # reset_ancient_card (2305-2315): a suit that is not the current one
        prev = into.get("ancient")
        pool = [s for s in SUITS_LUA_ORDER if s != prev]
        s, _ = prng.pseudorandom_element(pool, K.ancient(ante))
        into["ancient"] = s
    if "castle" in names:    # reset_castle_card (2317-2324): suit of a random playing card
        if cards:
            c, _ = prng.pseudorandom_element(cards, K.castle(ante))
            into["castle"] = c.suit
        else:
            into["castle"] = "Spades"
    return into


def round_card(ctx, name: str):
    """Joker-side accessor. Rolls the ONE missing value lazily on the context's
    PRNG (same key the game would have used for this ante) so a context built
    without a game still draws from the right stream."""
    rc = ctx.round_cards
    if name not in rc:
        from .jokers.base import rng_of
        roll_round_cards(rng_of(ctx), ctx.ante, ctx.full_deck, rc, names=(name,))
    return rc[name]
