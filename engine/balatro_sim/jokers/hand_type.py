"""
hand_type.py — Jokers that modify or respond to poker hand types.
"""
from .base import JOKER_REGISTRY, ScoreContext
from ..round_cards import round_card

# The hand-type mult jokers (j_jolly .. j_droll, j_duo .. j_tribe) live in
# mult.py; the hand-type chip jokers (j_sly .. j_crafty) live in chips.py.
# Each game key is registered in exactly one module (see jokers/__init__.py).

# ── j_mail: earn $5 per discarded card of the round's Mail-In rank ──────────
# card.lua:2825-2835 reads G.GAME.current_round.mail_card.id — a GAME-level rank
# rolled on 'mail<ante>' over the deck at run start and after every blind
# (round_cards.py). Paid immediately, per matching non-debuffed card.
class _MailInRebate:
    def on_discard(self, inst, cards, ctx):
        rank = round_card(ctx, "mail")
        for card in cards:
            if not card.debuffed and card.rank == rank and card.enhancement != "Stone":
                ctx.pending_money += 5
JOKER_REGISTRY["j_mail"] = _MailInRebate()

# Most hand type jokers are now implemented across the other modules.
# This file can remain as a reference/organizational layer.

# Add a few more unique hand mechanics:

# These are mostly implemented or stubbed. The key remaining work is:
# 1. Retrigger system
# 2. Consumable creation/upgrade
# 3. Shop system
# 4. Boss blind effects
# 5. Deck modification
# 6. Hand eval modification (Shortcut, FourFingers, Pareidolia, Splash, etc.)

# For now, this file serves as an organizational reference.
