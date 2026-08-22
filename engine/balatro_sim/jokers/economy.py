"""
economy.py — Money-generating jokers.

`dollars` effects that the real game pays immediately (ease_dollars inside the
scoring loop) go through ctx.pending_money, which game._play_hand / _discard
apply right after the hook; end-of-round payouts use inst.state['pending_money'],
drained by base.drain_joker_state.
"""
from .base import JOKER_REGISTRY, ScoreContext, prob_roll

# ── j_golden: earn $4 at end of round ────────────────────────────────────────
class _Golden:
    def on_round_end(self, inst, ctx):
        inst.state["pending_money"] = inst.state.get("pending_money", 0) + 4
JOKER_REGISTRY["j_golden"] = _Golden()

# ── j_to_the_moon: +$1 interest per $5 held (extra interest) ─────────────────
class _ToTheMoon:
    def on_hand_scored(self, inst, ctx):
        inst.state["dollars"] = ctx.dollars  # track for round_end
    def on_round_end(self, inst, ctx):
        dollars = inst.state.get("dollars", ctx.dollars)
        extra = dollars // 5
        inst.state["pending_money"] = inst.state.get("pending_money", 0) + extra
JOKER_REGISTRY["j_to_the_moon"] = _ToTheMoon()

# ── j_business: face cards have a 1 in 2 chance of giving $2 ─────────────────
# card.lua:3177 — key 'business', odds 2; one roll per scored face card per pass
# (`individual` context, so retriggers roll again). is_face honours Pareidolia.
class _BusinessCard:
    def on_score_card(self, inst, card, ctx):
        if ctx.is_face_card(card) and not card.debuffed and prob_roll(ctx, "business", 2):
            ctx.pending_money += 2
JOKER_REGISTRY["j_business"] = _BusinessCard()

# ── j_ticket: played Gold card gives $4 (card.lua:3150) ──────────────────────
class _GoldenTicket:
    def on_score_card(self, inst, card, ctx):
        if card.enhancement == "Gold" and not card.debuffed:
            ctx.pending_money += 4
JOKER_REGISTRY["j_ticket"] = _GoldenTicket()

# ── j_rocket: earn $1 per round, +$2 when boss blind beaten ──────────────────
class _Rocket:
    def on_round_end(self, inst, ctx):
        bonus = inst.state.get("bonus", 1)
        inst.state["pending_money"] = inst.state.get("pending_money", 0) + bonus
    def on_boss_beaten(self, inst, ctx):
        inst.state["bonus"] = inst.state.get("bonus", 1) + 2
JOKER_REGISTRY["j_rocket"] = _Rocket()

# ── j_red_card: +3 Mult permanently when any Booster Pack is skipped ─────────
class _RedCard:
    def on_booster_skipped(self, inst, ctx):
        inst.state["mult"] = inst.state.get("mult", 0) + 3
    def on_hand_scored(self, inst, ctx):
        ctx.mult += inst.state.get("mult", 0)
JOKER_REGISTRY["j_red_card"] = _RedCard()

# ── j_odd_todd: +31 chips per played card with odd rank (A,9,7,5,3) ──────────
class _OddTodd:
    ODD_RANKS = {14, 9, 7, 5, 3}  # A=14, 9, 7, 5, 3
    def on_score_card(self, inst, card, ctx):
        if card.rank in self.ODD_RANKS and not card.debuffed:
            ctx.chips += 31
JOKER_REGISTRY["j_odd_todd"] = _OddTodd()

# ── j_scholar: +20 chips and +4 mult if Ace is scored ────────────────────────
class _Scholar:
    def on_score_card(self, inst, card, ctx):
        if card.rank == 14 and not card.debuffed:
            ctx.chips += 20
            ctx.mult += 4
JOKER_REGISTRY["j_scholar"] = _Scholar()

# ── j_even_steven: +4 mult if scored card is even rank (10, 8, 6, 4, 2) ──────
# card.lua: `get_id() <= 10 and get_id() % 2 == 0` — an Ace (id 14) is NOT even.
class _EvenSteven:
    def on_score_card(self, inst, card, ctx):
        if card.rank <= 10 and card.rank % 2 == 0 and not card.debuffed:
            ctx.mult += 4
JOKER_REGISTRY["j_even_steven"] = _EvenSteven()
