"""
mult.py — Additive mult jokers and xMult jokers.
"""
from .base import JOKER_REGISTRY, ScoreContext, rng_of, prob_roll, is_suit
from ..round_cards import round_card
from ..constants import RANK_CHIPS

# ── Helpers ──────────────────────────────────────────────────────────────────

def _has_hand(hand_type, *targets):
    return hand_type in targets

def _suit_in_scoring(cards, suit):
    return any(c.suit == suit for c in cards if not c.debuffed)

def _all_suits(cards):
    suits = {c.suit for c in cards if not c.debuffed and c.enhancement != "Stone"}
    return suits

# ── j_joker: +4 Mult ─────────────────────────────────────────────────────────
class _Joker:
    def on_hand_scored(self, inst, ctx):
        ctx.mult += 4
JOKER_REGISTRY["j_joker"] = _Joker()

# ── Greedy/Lusty/Wrathful/Gluttonous: +3 Mult per scoring card of the suit ────
# Real text: "Played cards with Diamond suit give +3 Mult when scored" — it fires
# PER SCORING CARD, so a five-Diamond flush is +15 Mult, not +3.
#
# This fired once per hand (on_hand_scored, gated on the suit merely being
# present) until 2026-07-29. A duplicate, per-card implementation existed in
# scaling.py under the dead keys j_greedy/j_greedy_joker — the audit flagged the
# risk that a dead duplicate might be the correct one, and here it was.
class _SuitMult:
    def __init__(self, suit): self.suit = suit
    def on_score_card(self, inst, card, ctx):
        if card.suit == self.suit and not card.debuffed:
            ctx.mult += 3
JOKER_REGISTRY["j_greedy_joker"]     = _SuitMult("Diamonds")
JOKER_REGISTRY["j_lusty_joker"]      = _SuitMult("Hearts")
JOKER_REGISTRY["j_wrathful_joker"]   = _SuitMult("Spades")
JOKER_REGISTRY["j_gluttenous_joker"] = _SuitMult("Clubs")

# ── j_jolly: +8 Mult if Pair ─────────────────────────────────────────────────
class _Jolly:
    def on_hand_scored(self, inst, ctx):
        if _has_hand(ctx.hand_type, "Pair", "Two Pair", "Full House", "Four of a Kind", "Five of a Kind", "Flush House"):
            ctx.mult += 8
JOKER_REGISTRY["j_jolly"] = _Jolly()

# ── j_zany: +12 Mult if Three of a Kind ──────────────────────────────────────
class _Zany:
    def on_hand_scored(self, inst, ctx):
        if _has_hand(ctx.hand_type, "Three of a Kind", "Full House", "Five of a Kind", "Flush House"):
            ctx.mult += 12
JOKER_REGISTRY["j_zany"] = _Zany()

# ── j_mad: +10 Mult if Two Pair ───────────────────────────────────────────────
class _Mad:
    def on_hand_scored(self, inst, ctx):
        if _has_hand(ctx.hand_type, "Two Pair", "Full House"):
            ctx.mult += 10
JOKER_REGISTRY["j_mad"] = _Mad()

# ── j_crazy: +12 Mult if Straight ────────────────────────────────────────────
class _Crazy:
    def on_hand_scored(self, inst, ctx):
        if _has_hand(ctx.hand_type, "Straight", "Straight Flush"):
            ctx.mult += 12
JOKER_REGISTRY["j_crazy"] = _Crazy()

# ── j_droll: +10 Mult if Flush ───────────────────────────────────────────────
class _Droll:
    def on_hand_scored(self, inst, ctx):
        if _has_hand(ctx.hand_type, "Flush", "Straight Flush", "Flush House", "Flush Five"):
            ctx.mult += 10
JOKER_REGISTRY["j_droll"] = _Droll()

# ── j_half: +20 Mult if ≤3 cards played ─────────────────────────────────────
class _Half:
    def on_hand_scored(self, inst, ctx):
        if len(ctx.scoring_cards) <= 3:
            ctx.mult += 20
JOKER_REGISTRY["j_half"] = _Half()

# ── j_abstract: +3 Mult per Joker owned ─────────────────────────────────────
class _Abstract:
    def on_hand_scored(self, inst, ctx):
        ctx.mult += 3 * ctx.n_jokers
JOKER_REGISTRY["j_abstract"] = _Abstract()

# ── j_banner: +30 Chips per remaining Discard ────────────────────────────────
class _Banner:
    def on_hand_scored(self, inst, ctx):
        ctx.chips += 30 * ctx.discards_left
JOKER_REGISTRY["j_banner"] = _Banner()

# ── j_mystic_summit: +15 Mult when 0 discards remaining ──────────────────────
class _MysticSummit:
    def on_hand_scored(self, inst, ctx):
        if ctx.discards_left == 0:
            ctx.mult += 15
JOKER_REGISTRY["j_mystic_summit"] = _MysticSummit()

# ── j_misprint: +0 to +23 Mult (random) ─────────────────────────────────────
# card.lua:3701 — `pseudorandom('misprint', 0, 23)` (math.random(min, max) form).
class _Misprint:
    def on_hand_scored(self, inst, ctx):
        ctx.mult += rng_of(ctx).pseudorandom("misprint", 0, 23)
JOKER_REGISTRY["j_misprint"] = _Misprint()

# ── j_raised_fist: adds double the CHIP value of the lowest held card to Mult ─
# card.lua:3320-3343 (held `individual` context): the "raised" card is the LAST
# card in hand order with the lowest rank id, Stone excluded; the effect is
# 2 * base.nominal (Ace = 11, King = 10, ... — NOT the rank number) and only
# fires for that card, so Mime / Red seal retrigger it. Debuffed: message only.
class _RaisedFist:
    @staticmethod
    def _raised(ctx):
        temp_id, raised = 15, None
        for c in ctx.held_cards:
            if c.enhancement != "Stone" and temp_id >= c.rank:
                temp_id, raised = c.rank, c
        return raised
    def on_held_card(self, inst, card, ctx):
        if card is not self._raised(ctx):
            return False
        if not card.debuffed:
            ctx.mult += 2 * RANK_CHIPS.get(card.rank, 0)
        return True
JOKER_REGISTRY["j_raised_fist"] = _RaisedFist()

# ── j_fibonacci: +8 Mult if scoring card is Ace, 2, 3, 5, or 8 ──────────────
class _Fibonacci:
    FIB = {14, 2, 3, 5, 8}
    def on_score_card(self, inst, card, ctx):
        if card.rank in self.FIB and not card.debuffed:
            ctx.mult += 8
JOKER_REGISTRY["j_fibonacci"] = _Fibonacci()

# ── j_smiley: +5 Mult per scored face card ───────────────────────────────────
class _Smiley:
    def on_score_card(self, inst, card, ctx):
        if ctx.is_face_card(card) and not card.debuffed:
            ctx.mult += 5
JOKER_REGISTRY["j_smiley"] = _Smiley()

# ── j_shoot_the_moon: +13 Mult per Queen in hand ────────────────────────────
# card.lua:3272 — held `individual` context: +13 per Queen held, per pass.
class _ShootTheMoon:
    def on_held_card(self, inst, card, ctx):
        if card.rank != 12:
            return False
        if not card.debuffed:
            ctx.mult += 13
        return True
JOKER_REGISTRY["j_shoot_the_moon"] = _ShootTheMoon()

# ── j_walkie_talkie: +10 chips +4 mult if card is 10 or 4 ───────────────────
class _WalkieTalkie:
    def on_score_card(self, inst, card, ctx):
        if card.rank in (4, 10) and not card.debuffed:
            ctx.chips += 10
            ctx.mult += 4
JOKER_REGISTRY["j_walkie_talkie"] = _WalkieTalkie()

# ── j_bootstraps: +2 Mult per $5 held ───────────────────────────────────────
class _Bootstraps:
    def on_hand_scored(self, inst, ctx):
        ctx.mult += 2 * (ctx.dollars // 5)
JOKER_REGISTRY["j_bootstraps"] = _Bootstraps()

# ── j_photograph: x2 Mult when the FIRST face card of the scoring hand scores ─
# card.lua:3093-3101 — per hand (not per round), Pareidolia-aware, and every
# retrigger pass of that card gets the x2 again.
class _Photograph:
    def on_score_card(self, inst, card, ctx):
        first_face = next((c for c in ctx.scoring_cards if ctx.is_face_card(c)), None)
        if card is first_face and not card.debuffed:
            ctx.mult_mult *= 2
JOKER_REGISTRY["j_photograph"] = _Photograph()

# ── j_scary_face: +30 chips if scoring card is face card ────────────────────
class _ScaryFace:
    def on_score_card(self, inst, card, ctx):
        if ctx.is_face_card(card) and not card.debuffed:
            ctx.chips += 30
JOKER_REGISTRY["j_scary_face"] = _ScaryFace()

# ── j_pareidolia: treat all cards as face cards ──────────────────────────────
# (affects j_smiley, j_photograph, j_scary_face, etc. — requires engine support)
# TODO: implement flag on ScoreContext

# ── j_acrobat: x3 Mult on last hand of round ────────────────────────────────
class _Acrobat:
    def on_hand_scored(self, inst, ctx):
        if ctx.hands_left == 0:
            ctx.mult_mult *= 3
JOKER_REGISTRY["j_acrobat"] = _Acrobat()

# ── xMult jokers (The Duo, Trio, etc.) ───────────────────────────────────────
class _TheDuo:
    def on_hand_scored(self, inst, ctx):
        if _has_hand(ctx.hand_type, "Pair", "Two Pair", "Full House", "Four of a Kind", "Five of a Kind", "Flush House"):
            ctx.mult_mult *= 2
JOKER_REGISTRY["j_duo"] = _TheDuo()

class _TheTrio:
    def on_hand_scored(self, inst, ctx):
        if _has_hand(ctx.hand_type, "Three of a Kind", "Full House", "Five of a Kind", "Flush House"):
            ctx.mult_mult *= 3
JOKER_REGISTRY["j_trio"] = _TheTrio()

class _TheFamily:
    def on_hand_scored(self, inst, ctx):
        if _has_hand(ctx.hand_type, "Four of a Kind", "Five of a Kind"):
            ctx.mult_mult *= 4
JOKER_REGISTRY["j_family"] = _TheFamily()

class _TheOrder:
    def on_hand_scored(self, inst, ctx):
        if _has_hand(ctx.hand_type, "Straight", "Straight Flush"):
            ctx.mult_mult *= 3
JOKER_REGISTRY["j_order"] = _TheOrder()

class _TheTribe:
    def on_hand_scored(self, inst, ctx):
        if _has_hand(ctx.hand_type, "Flush", "Straight Flush", "Flush House", "Flush Five"):
            ctx.mult_mult *= 2
JOKER_REGISTRY["j_tribe"] = _TheTribe()

# ── j_stencil: x1 Mult per empty Joker slot; Stencil itself counts as empty ──
# card.lua:4203-4208 x_mult = (card_limit - #jokers) + #Stencils; applied only
# when at least one slot is genuinely empty (:3966-3973). Reads game.joker_slots.
class _Stencil:
    def on_hand_scored(self, inst, ctx):
        empty = ctx.joker_slots - ctx.n_jokers
        if empty > 0:
            xm = empty + sum(1 for j in ctx.jokers if j.key == "j_stencil")
            ctx.mult_mult *= xm
JOKER_REGISTRY["j_stencil"] = _Stencil()

# ── j_flower_pot: x3 Mult if the SCORING hand has all four suits ─────────────
# card.lua:3807-3840: non-Wild cards claim their suit first (Smeared-aware),
# then Wild cards fill whichever suit is still missing.
class _FlowerPot:
    SUITS = ("Hearts", "Diamonds", "Spades", "Clubs")
    def on_hand_scored(self, inst, ctx):
        have = {s: 0 for s in self.SUITS}
        cards = [c for c in ctx.scoring_cards if not c.debuffed]
        for c in cards:
            if c.enhancement == "Wild":
                continue
            for s in self.SUITS:
                if have[s] == 0 and is_suit(ctx, c, s):
                    have[s] = 1
                    break
        for c in cards:
            if c.enhancement == "Wild":
                for s in self.SUITS:
                    if have[s] == 0:
                        have[s] = 1
                        break
        if all(have.values()):
            ctx.mult_mult *= 3
JOKER_REGISTRY["j_flower_pot"] = _FlowerPot()

# ── j_triboulet: x2 Mult per King or Queen scored ───────────────────────────
class _Triboulet:
    def on_score_card(self, inst, card, ctx):
        if card.rank in (12, 13) and not card.debuffed:
            ctx.mult_mult *= 2
JOKER_REGISTRY["j_triboulet"] = _Triboulet()

# ── j_seeing_double: x2 Mult if Club + another suit in scoring hand ──────────
class _SeeingDouble:
    def on_hand_scored(self, inst, ctx):
        suits = _all_suits(ctx.scoring_cards)
        if "Clubs" in suits and len(suits) >= 2:
            ctx.mult_mult *= 2
JOKER_REGISTRY["j_seeing_double"] = _SeeingDouble()

# ── j_loyalty_card: x4 Mult every 6th hand played ───────────────────────────
class _LoyaltyCard:
    def on_hand_scored(self, inst, ctx):
        inst.state.setdefault("count", 0)
        inst.state["count"] += 1
        if inst.state["count"] % 6 == 0:
            ctx.mult_mult *= 4
JOKER_REGISTRY["j_loyalty_card"] = _LoyaltyCard()

# ── j_bloodstone: 1/2 chance x1.5 Mult per Heart scored ─────────────────────
# card.lua:3249 — key 'bloodstone', odds 2; one roll per scored Heart per pass
# (Wild / Smeared-aware is_suit).
class _Bloodstone:
    def on_score_card(self, inst, card, ctx):
        if not card.debuffed and is_suit(ctx, card, "Hearts") and prob_roll(ctx, "bloodstone", 2):
            ctx.mult_mult *= 1.5
JOKER_REGISTRY["j_bloodstone"] = _Bloodstone()

# ── j_ancient: x1.5 Mult per scored card of the round's Ancient suit ─────────
# card.lua:3255 reads G.GAME.current_round.ancient_card.suit — a GAME-level
# value rolled on 'anc<ante>' at run start and after every blind
# (round_cards.py), never per-joker state.
class _Ancient:
    def on_score_card(self, inst, card, ctx):
        if not card.debuffed and is_suit(ctx, card, round_card(ctx, "ancient")):
            ctx.mult_mult *= 1.5
JOKER_REGISTRY["j_ancient"] = _Ancient()

# ── j_throwback: x0.25 Mult per Blind skipped, starts at x1 ─────────────────
class _Throwback:
    def on_hand_scored(self, inst, ctx):
        bonus = inst.state.get("bonus", 1.0)
        ctx.mult_mult *= bonus
    def on_blind_skipped(self, inst, ctx):
        inst.state["bonus"] = inst.state.get("bonus", 1.0) + 0.25
JOKER_REGISTRY["j_throwback"] = _Throwback()

# ── j_idol: x2 Mult per scored copy of the round's Idol card ─────────────────
# card.lua:3127-3135 (`individual`: get_id() == idol_card.id and is_suit) —
# G.GAME.current_round.idol_card is rolled on 'idol<ante>' over the deck at run
# start and after every blind (round_cards.py).
class _TheIdol:
    def on_score_card(self, inst, card, ctx):
        rank, suit = round_card(ctx, "idol")
        if not card.debuffed and card.rank == rank and is_suit(ctx, card, suit):
            ctx.mult_mult *= 2
JOKER_REGISTRY["j_idol"] = _TheIdol()

# ── j_dusk: retrigger all cards on last hand of round ────────────────────────
# TODO: retrigger requires engine support (mark last hand in ctx)

# ── j_mime: retrigger held cards ─────────────────────────────────────────────
# TODO: retrigger requires engine support

# ── j_sock_and_buskin: retrigger face cards ───────────────────────────────────
# TODO: retrigger requires engine support

# ── j_hanging_chad: retrigger first scored card twice ────────────────────────
# TODO: retrigger requires engine support

# ── j_selzer: retrigger all cards, vanishes after 10 hands ──────────────────
# TODO: retrigger + vanish requires engine support

# ── j_hack: retrigger 2s, 3s, 4s, 5s ────────────────────────────────────────
# TODO: retrigger requires engine support

# ── j_blueprint: copy joker to right ─────────────────────────────────────────
# TODO: complex chain - requires engine support

# ── j_brainstorm: copy leftmost joker ────────────────────────────────────────
# TODO: complex chain

# ── j_burnt: upgrade most played hand ───────────────────────────────────
# TODO: requires tracking most played hand + planet upgrade

# ── j_mr_bones: prevent loss if ≥25% of blind ────────────────────────────────
# TODO: requires engine hook on loss condition (not a scoring joker)

# ── j_four_fingers: Flushes/Straights with 4 cards ───────────────────────────
# TODO: requires hand eval modification flag

# ── j_smeared: Hearts/Diamonds same suit, Spades/Clubs same suit ───────
# TODO: requires hand eval modification flag

# ── j_oops: double all listed probabilities ────────────────────────
# TODO: requires probability system

# ── j_8_ball: 1/4 chance of Tarot if 4+ 8s played ───────────────────────────
# TODO: requires consumable creation system

# ── j_cartomancer: create Tarot card each round ───────────────────────────────
# TODO: requires consumable creation system

# ── j_astronomer: free Planet cards ──────────────────────────────────────────
# TODO: requires shop system integration

# ── j_invisible: after 2 rounds, duplicate a random joker ──────────────
# TODO: complex - requires joker creation system

# ── j_perkeo: creates Negative Tarot copy of last consumed card ──────────────
# TODO: requires consumable tracking + creation

# ── j_chicot: disables boss blind effect ─────────────────────────────────────
# TODO: requires boss blind system integration

# ── j_matador: earns $8 if boss blind ability triggered ──────────────────────
# TODO: requires boss event hook

# ── j_ring_master: reroll boss blind ─────────────────────────────────────────
# TODO: requires shop system integration

# ── j_credit_card: go up to $20 in debt ──────────────────────────────────────
# TODO: requires economic system integration (not a scoring joker)

# ── j_merry_andy: +3 discards, -1 hand size ──────────────────────────────────
# TODO: game state modification (not scoring)

# ── j_troubadour: +2 hand size, -1 hand count ────────────────────────────────
# TODO: game state modification

# ── j_stuntman: +250 chips, -2 hand size ─────────────────────────────────────
class _Stuntman:
    def on_hand_scored(self, inst, ctx):
        ctx.chips += 250
JOKER_REGISTRY["j_stuntman"] = _Stuntman()

# ── j_space: 1 in 4 chance to level up the played hand ───────────────────────
# card.lua:3420 — `before` context, key 'space', odds 4: the level-up lands
# BEFORE the base chips/mult are read, so this very hand scores at the new level.
class _SpaceJoker:
    def pre_score(self, inst, ctx):
        if prob_roll(ctx, "space", 4):
            ctx.planet_levels[ctx.hand_type] = ctx.planet_levels.get(ctx.hand_type, 1) + 1
JOKER_REGISTRY["j_space"] = _SpaceJoker()

# ── j_drivers_license: +3 Mult per enhanced card if ≥16 enhanced in deck ─────
# TODO: needs full deck access for enhanced card count
