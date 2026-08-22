"""
chips.py — Flat chip bonus jokers (plus the jokers that historically lived here).

Every probability roll uses the game's key string through prob_roll(ctx, key, odds)
— see mp/rng/keys.py and EFFECTS_NOTES.md for the table.
"""
from .base import (
    JOKER_REGISTRY, ScoreContext, prob_roll, rng_of, has_consumable_room, create_consumable,
)

# ── j_steel_joker: X1 Mult, gains X0.2 per Steel card in your FULL DECK ──────
class _SteelJoker:
    def on_hand_scored(self, inst, ctx):
        steel_count = sum(1 for c in ctx.full_deck if c.enhancement == "Steel")
        if steel_count:
            ctx.mult_mult *= (1.0 + 0.2 * steel_count)
JOKER_REGISTRY["j_steel_joker"] = _SteelJoker()

# ── j_gros_michel: +15 mult, 1 in 6 chance to be destroyed at end of round ──
# card.lua:3020 — key 'gros_michel', odds 6. Destruction is applied by
# base.drain_joker_state (removes the joker, sets pool_flags 'gros_michel_extinct'
# so Cavendish becomes available in the joker pool).
class _GrosMichel:
    def on_hand_scored(self, inst, ctx):
        ctx.mult += 15
    def on_round_end(self, inst, ctx):
        if prob_roll(ctx, "gros_michel", 6):
            inst.state["destroyed"] = True
JOKER_REGISTRY["j_gros_michel"] = _GrosMichel()

# ── j_cavendish: x3 Mult, 1 in 1000 chance of being destroyed at end of round
# card.lua:3020 — key 'cavendish', odds 1000.
class _Cavendish:
    def on_hand_scored(self, inst, ctx):
        ctx.mult_mult *= 3
    def on_round_end(self, inst, ctx):
        if prob_roll(ctx, "cavendish", 1000):
            inst.state["destroyed"] = True
JOKER_REGISTRY["j_cavendish"] = _Cavendish()

# ── j_madness: X1 Mult; when a Small or Big blind is selected gains X0.5 and
# destroys a random other joker (card.lua:2503-2520, key 'madness'; the pick is
# made in game._start_blind over the other non-eternal jokers in board order).
# Before W3 the X started at 0 (+0.5 -> x0.5 on first use, HALVING the mult).
class _Madness:
    def on_blind_selected(self, inst, ctx):
        if ctx.blind_kind == "Boss":
            return
        inst.state["xmult"] = inst.state.get("xmult", 1.0) + 0.5
        inst.state["destroy_random"] = True  # game._start_blind rolls 'madness'
    def on_hand_scored(self, inst, ctx):
        xm = inst.state.get("xmult", 1.0)
        if xm > 1.0:
            ctx.mult_mult *= xm
JOKER_REGISTRY["j_madness"] = _Madness()

# ── j_square: gains +4 chips if the played hand has exactly 4 cards ──────────
class _SquareJoker:
    def on_hand_scored(self, inst, ctx):
        if len(ctx.scoring_cards) == 4:
            inst.state["chips"] = inst.state.get("chips", 0) + 4
        ctx.chips += inst.state.get("chips", 0)
JOKER_REGISTRY["j_square"] = _SquareJoker()

# ── j_vampire: remove enhancement from scored card, gain x0.1 xMult per card ─
class _Vampire:
    def on_score_card(self, inst, card, ctx):
        if card.enhancement and card.enhancement not in ("None", "Base") and not card.debuffed:
            inst.state["xmult"] = inst.state.get("xmult", 1.0) + 0.1
            card.enhancement = "None"
    def on_hand_scored(self, inst, ctx):
        xm = inst.state.get("xmult", 1.0)
        if xm > 1.0:
            ctx.mult_mult *= xm
JOKER_REGISTRY["j_vampire"] = _Vampire()

# ── j_hologram: x0.25 Mult per playing card added to deck ────────────────────
# card.lua:2452 (`playing_card_added`, +0.25 per card). Fired by game.add_card
# through base.fire_hook(game, "on_card_added").
class _Hologram:
    def on_card_added(self, inst, ctx):
        inst.state["xmult"] = inst.state.get("xmult", 1.0) + 0.25
    def on_hand_scored(self, inst, ctx):
        xm = inst.state.get("xmult", 1.0)
        if xm > 1.0:
            ctx.mult_mult *= xm
JOKER_REGISTRY["j_hologram"] = _Hologram()

# ── j_vagabond: create a Tarot if the hand is played with $4 or less ─────────
# card.lua:3743 — room check first, then create_card('Tarot', ..., 'vag').
class _Vagabond:
    def on_hand_scored(self, inst, ctx):
        if ctx.dollars <= 4 and has_consumable_room(ctx):
            create_consumable(ctx, "vagabond")
JOKER_REGISTRY["j_vagabond"] = _Vagabond()

# ── j_hit_the_road: x0.5 xMult per Jack discarded this round ────────────────
class _HitTheRoad:
    def on_discard(self, inst, cards, ctx):
        for card in cards:
            if card.rank == 11:  # Jack
                inst.state["xmult"] = inst.state.get("xmult", 1.0) + 0.5
    def on_hand_scored(self, inst, ctx):
        xm = inst.state.get("xmult", 1.0)
        if xm > 1.0:
            ctx.mult_mult *= xm
    def on_round_end(self, inst, ctx):
        inst.state["xmult"] = 1.0  # reset each round
JOKER_REGISTRY["j_hit_the_road"] = _HitTheRoad()

# ── j_drivers_license: x3 mult if you have >= 16 enhanced cards ─────────────
class _DriversLicense:
    def on_hand_scored(self, inst, ctx):
        enhanced = sum(1 for c in ctx.full_deck if c.enhancement and c.enhancement not in ("None", "Base"))
        if enhanced >= 16:
            ctx.mult_mult *= 3
JOKER_REGISTRY["j_drivers_license"] = _DriversLicense()

# ── j_caino: x1 Mult, gains x0.1 when a face card is destroyed ──────────────
class _Caino:
    def on_card_destroyed(self, inst, card, ctx):
        if card.is_face_card:
            inst.state["xmult"] = inst.state.get("xmult", 1.0) + 0.1
    def on_hand_scored(self, inst, ctx):
        ctx.mult_mult *= inst.state.get("xmult", 1.0)
JOKER_REGISTRY["j_caino"] = _Caino()

# ── j_yorick: x1 Mult, gains x1 per 23 cards discarded ──────────────────────
class _Yorick:
    def on_discard(self, inst, cards, ctx):
        inst.state["discarded"] = inst.state.get("discarded", 0) + len(cards)
    def on_hand_scored(self, inst, ctx):
        sets = inst.state.get("discarded", 0) // 23
        ctx.mult_mult *= (1.0 + sets)
JOKER_REGISTRY["j_yorick"] = _Yorick()
