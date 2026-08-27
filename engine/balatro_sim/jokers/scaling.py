"""
scaling.py — Jokers that gain permanent stat increases over time.
These build state across multiple hands/rounds.
"""
from .base import JOKER_REGISTRY, ScoreContext, rng_of, is_suit
from ..round_cards import round_card

# The four suit jokers are registered in mult.py under their catalogue keys
# (j_greedy_joker / j_lusty_joker / j_wrathful_joker / j_gluttenous_joker) via
# _SuitMult. The duplicate _Greedy/_Lusty/_Wrathful/_Gluttonous classes that
# used to be registered here under j_greedy / j_greedy_joker etc. were dead —
# no catalogue entry pointed at them, so they never ran. Removed 2026-07-29
# (audit M3) so nobody edits the copy that has no effect.

# ── j_sly: +50 chips if hand contains a Pair ────────────────────────────────
_PAIR_TYPES = {"Pair", "Two Pair", "Full House", "Four of a Kind", "Five of a Kind", "Flush House", "Flush Five"}
class _Sly:
    def on_hand_scored(self, inst, ctx):
        if ctx.hand_type in _PAIR_TYPES:
            ctx.chips += 50
JOKER_REGISTRY["j_sly"] = _Sly()

# ── j_wily: +100 chips if hand contains Three of a Kind ─────────────────────
class _Wily:
    def on_hand_scored(self, inst, ctx):
        if "Three" in ctx.hand_type:
            ctx.chips += 100
JOKER_REGISTRY["j_wily"] = _Wily()

# ── j_clever: +80 chips if hand contains Two Pair ───────────────────────────
class _Clever:
    def on_hand_scored(self, inst, ctx):
        if "Two Pair" in ctx.hand_type:
            ctx.chips += 80
JOKER_REGISTRY["j_clever"] = _Clever()

# ── j_devious: +100 chips if hand contains a Straight ───────────────────────
class _Devious:
    def on_hand_scored(self, inst, ctx):
        if "Straight" in ctx.hand_type:  # includes Straight Flush
            ctx.chips += 100
JOKER_REGISTRY["j_devious"] = _Devious()

# ── j_crafty: +80 chips if hand contains a Flush ────────────────────────────
class _Crafty:
    def on_hand_scored(self, inst, ctx):
        if "Flush" in ctx.hand_type:  # includes Straight Flush, Flush House, Flush Five
            ctx.chips += 80
JOKER_REGISTRY["j_crafty"] = _Crafty()

# ── j_green_joker: +1 mult per hand played, -1 mult per discard ────────────
# card.lua:3563 (`context.before`, `and not context.blueprint`) raises the counter;
# :4010-4015 (joker_main) pays it.  Two different passes, so the increment lands
# BEFORE any joker — including a Blueprint copying this one — reads the value, and a
# copy pays the RAISED figure without raising it a second time.
class _GreenJoker:
    def pre_score(self, inst, ctx):
        if ctx.blueprint:
            return
        inst.state["mult"] = inst.state.get("mult", 0) + 1
    def on_hand_scored(self, inst, ctx):
        ctx.mult += inst.state.get("mult", 0)
    def on_discard(self, inst, cards, ctx):
        inst.state["mult"] = max(0, inst.state.get("mult", 0) - 1)
JOKER_REGISTRY["j_green_joker"] = _GreenJoker()

# ── j_supernova: +Mult equal to times this poker hand was played this run ────
# Real text: "Adds the number of times poker hand has been played this run to
# Mult". Was buyable with NO implementation at all (audit M1) — the only such
# joker in the catalogue. ctx.hand_type_counts already includes the current hand,
# matching the real game.
class _Supernova:
    def on_hand_scored(self, inst, ctx):
        ctx.mult += ctx.hand_type_counts.get(ctx.hand_type, 0)
JOKER_REGISTRY["j_supernova"] = _Supernova()

# ── j_superposition: create a Tarot if hand contains Ace and Straight ───────
# TODO: consumable creation

# ── j_todo_list: earn $4 if hand is {specific hand type}, changes ──────────
# TODO: requires tracking target hand type

# ── j_bull: +2 chips per $1 held ────────────────────────────────────────────
class _Bull:
    def on_hand_scored(self, inst, ctx):
        ctx.chips += 2 * ctx.dollars
JOKER_REGISTRY["j_bull"] = _Bull()

# ── j_diet_cola: sell to create free Double Tag ─────────────────────────────
# TODO: tag system

# ── j_trading: first discard each round costs $3 but creates Foil/Holo/Poly ──
# TODO: discard modification

# ── j_popcorn: +20 mult, -4 mult per round played ───────────────────────────
class _Popcorn:
    def on_init(self, inst, ctx):
        inst.state["mult"] = 20
    def on_hand_scored(self, inst, ctx):
        ctx.mult += inst.state.get("mult", 20)
    def on_round_end(self, inst, ctx):
        inst.state["mult"] = max(0, inst.state.get("mult", 20) - 4)
JOKER_REGISTRY["j_popcorn"] = _Popcorn()

# ── j_ramen: x2 mult, loses x0.01 mult per card discarded ───────────────────
class _Ramen:
    def on_init(self, inst, ctx):
        inst.state["mult"] = 2.0
    def on_hand_scored(self, inst, ctx):
        ctx.mult_mult *= inst.state.get("mult", 2.0)
    def on_discard(self, inst, cards, ctx):
        inst.state["mult"] = max(1.0, inst.state.get("mult", 2.0) - 0.01 * len(cards))
JOKER_REGISTRY["j_ramen"] = _Ramen()

# ── j_selzer: retrigger all cards for next 3 hands ─────────────────────────
# TODO: retrigger system

# ── j_castle: gains +3 chips per discarded card of the round's Castle suit ───
# card.lua:2814-2823 reads G.GAME.current_round.castle_card.suit — a GAME-level
# value rolled on 'cas<ante>' over the deck at run start and after every blind
# (round_cards.py). The chips are permanent (the old per-round reset was wrong).
class _Castle:
    def on_discard(self, inst, cards, ctx):
        suit = round_card(ctx, "castle")
        for card in cards:
            if not card.debuffed and is_suit(ctx, card, suit):
                inst.state["chips"] = inst.state.get("chips", 0) + 3
    def on_hand_scored(self, inst, ctx):
        ctx.chips += inst.state.get("chips", 0)
JOKER_REGISTRY["j_castle"] = _Castle()

# ── j_campfire: x0.25 Mult per card sold, resets when Boss Blind is defeated ─
# card.lua:2396 (`selling_card`: ANY card sold, joker or consumable) / :2889.
# Fired by shop.sell_joker through base.sell_hooks -> fire_hook("on_card_sold").
class _Campfire:
    def on_hand_scored(self, inst, ctx):
        xm = 1.0 + inst.state.get("sold", 0) * 0.25
        if xm > 1.0:
            ctx.mult_mult *= xm
    def on_card_sold(self, inst, ctx):
        inst.state["sold"] = inst.state.get("sold", 0) + 1
    def on_boss_beaten(self, inst, ctx):
        inst.state["sold"] = 0  # reset on boss defeat
JOKER_REGISTRY["j_campfire"] = _Campfire()

# ── j_mr_bones: prevents death once, then destroys itself ───────────────────
# TODO: death prevention system

# ── j_sock_and_buskin: retrigger all face cards ─────────────────────────────
# TODO: retrigger system

# ── j_troubadour: +2 hand size, -1 hand per round ───────────────────────────
# TODO: hand size/hand count modification

# ── j_certificate: +1 dollar per round, each copy of held scored card +1 more
# TODO: held card tracking

# ── j_smeared: Hearts and Diamonds count as same suit ─────────────────
# TODO: suit evaluation modification

# ── j_hanging_chad: retrigger first played card 2 times ─────────────────────
# TODO: retrigger system

# ── j_rough_gem: +$1 per Diamond card scored ────────────────────────────────
class _RoughGem:
    def on_score_card(self, inst, card, ctx):
        if card.suit == "Diamonds" and not card.debuffed:
            ctx.pending_money += 1
JOKER_REGISTRY["j_rough_gem"] = _RoughGem()

# ── j_arrowhead: +50 chips per Spade card scored ────────────────────────────
class _Arrowhead:
    def on_score_card(self, inst, card, ctx):
        if card.suit == "Spades" and not card.debuffed:
            ctx.chips += 50
JOKER_REGISTRY["j_arrowhead"] = _Arrowhead()

# ── j_onyx_agate: scored cards with Club suit give +7 mult ──────────────────
class _OnyxAgate:
    def on_score_card(self, inst, card, ctx):
        if card.suit == "Clubs" and not card.debuffed:
            ctx.mult += 7
JOKER_REGISTRY["j_onyx_agate"] = _OnyxAgate()

# ── j_glass: gains x0.75 Mult for every Glass card DESTROYED ───────────
# Real text: "This Joker gains X0.75 Mult for every Glass Card that is destroyed."
# It scales on destruction, not on Glass cards sitting in the deck. Before the
# 2026-07-29 audit this counted `ctx.all_cards` — which is the played selection,
# not the deck — so it was a different joker entirely, and Glass cards were never
# destroyed at all.
class _GlassJoker:
    def on_card_destroyed(self, inst, card, ctx):
        if card.enhancement == "Glass":
            inst.state["mult_mult"] = inst.state.get("mult_mult", 1.0) + 0.75

    def on_hand_scored(self, inst, ctx):
        gained = inst.state.get("mult_mult", 1.0)
        if gained > 1.0:
            ctx.mult_mult *= gained
JOKER_REGISTRY["j_glass"] = _GlassJoker()

# ── j_ring_master: for each Joker, reroll shop 1 time ───────────────────────────
# TODO: shop system

# ── j_blueprint: copy right-most Joker ──────────────────────────────────────
# TODO: joker copying

# ── j_obelisk: x0.2 Mult per consecutive hand that isn't your most played type
# card.lua:3543 (`context.before`, `and not context.blueprint`) walks the streak;
# the x_mult is paid in the generic joker_main x_mult branch.  Split across the two
# passes for the same reason as Green Joker.
class _Obelisk:
    def pre_score(self, inst, ctx):
        if ctx.blueprint:
            return
        counts = inst.state.setdefault("counts", {})
        counts[ctx.hand_type] = counts.get(ctx.hand_type, 0) + 1
        most_played = max(counts, key=counts.get)
        if ctx.hand_type != most_played:
            inst.state["streak"] = inst.state.get("streak", 0) + 1
        else:
            inst.state["streak"] = 0
    def on_hand_scored(self, inst, ctx):
        xm = 1.0 + inst.state.get("streak", 0) * 0.2
        ctx.mult_mult *= xm
JOKER_REGISTRY["j_obelisk"] = _Obelisk()

# ── j_luchador: sell to disable current Boss Blind ──────────────────────────
# TODO: boss blind system

# ── j_turtle_bean: +5 hand size, reduce by 1 per round ──────────────────────
# TODO: hand size modification

# ── j_erosion: +4 mult per card below 52 in deck ────────────────────────────
class _Erosion:
    def on_hand_scored(self, inst, ctx):
        missing = max(0, 52 - ctx.deck_remaining)
        ctx.mult += 4 * missing
JOKER_REGISTRY["j_erosion"] = _Erosion()

# ── j_hallucination: 1 in 4 chance create Tarot when any Booster opened ─────
# TODO: booster system

# ── j_fortune_teller: +1 mult per Tarot used this run ───────────────────────
class _FortuneTeller:
    def on_tarot_used(self, inst, ctx):
        inst.state["mult"] = inst.state.get("mult", 0) + 1
    def on_hand_scored(self, inst, ctx):
        ctx.mult += inst.state.get("mult", 0)
JOKER_REGISTRY["j_fortune_teller"] = _FortuneTeller()

# ── j_juggler: +1 hand size ──────────────────────────────────────────────────
# TODO: hand size modification

# ── j_drunkard: +1 discard ──────────────────────────────────────────────────
# TODO: discard modification

# ── j_stone: full deck gives +25 chips per Stone card ───────────────────────
# (same as j_stone above)

# ── j_lucky_cat: x0.25 xMult per successful Lucky trigger (permanent) ───────
# card.lua:3076 (`individual`: other_card.lucky_trigger `and not context.blueprint`) —
# scoring.py sets ctx.lucky_trigger on each pass where either Lucky roll hit.
class _LuckyCat:
    def on_score_card(self, inst, card, ctx):
        if ctx.lucky_trigger and not ctx.blueprint:
            inst.state["xmult"] = inst.state.get("xmult", 1.0) + 0.25
    def on_hand_scored(self, inst, ctx):
        xm = inst.state.get("xmult", 1.0)
        if xm > 1.0:
            ctx.mult_mult *= xm
JOKER_REGISTRY["j_lucky_cat"] = _LuckyCat()

# ── j_baseball: each Uncommon Joker gives x1.5 Mult ─────────────────────────
class _Baseball:
    def on_hand_scored(self, inst, ctx):
        from ..shop import JOKER_CATALOGUE
        for j in ctx.jokers:
            meta = JOKER_CATALOGUE.get(j.key, {})
            if meta.get("rarity") == "Uncommon":
                ctx.mult_mult *= 1.5
JOKER_REGISTRY["j_baseball"] = _Baseball()

# ── j_trousers: +2 mult if played hand contains Two Pair ──────────────
# card.lua:3412 (`context.before`, `and not context.blueprint`) gains; :3986 (joker_main)
# pays.  Same before/main split as Green Joker.
class _SpareTrousers:
    def pre_score(self, inst, ctx):
        if ctx.blueprint:
            return
        if "Two Pair" in ctx.hand_type:
            inst.state["mult"] = inst.state.get("mult", 0) + 2
    def on_hand_scored(self, inst, ctx):
        ctx.mult += inst.state.get("mult", 0)
JOKER_REGISTRY["j_trousers"] = _SpareTrousers()

# ── j_burglar: +3 hands, -3 discards when blind is selected ─────────────────
# Real Balatro: game-state modifier, not a scoring joker. Gives extra hands
# but removes all discards. Applied via on_blind_selected hook in game.py.
class _Burglar:
    def on_blind_selected(self, inst, ctx):
        # game.py reads this state and applies +3 hands, sets discards to 0
        inst.state["extra_hands"] = 3
        inst.state["zero_discards"] = True
JOKER_REGISTRY["j_burglar"] = _Burglar()

# ── j_blackboard: x3 mult if all cards in hand are Spades or Clubs ──────────
class _Blackboard:
    def on_hand_scored(self, inst, ctx):
        all_black = all(c.suit in ["Spades", "Clubs"] for c in ctx.held_cards if not c.debuffed)
        if all_black:
            ctx.mult_mult *= 3
JOKER_REGISTRY["j_blackboard"] = _Blackboard()

# ── j_runner: +15 chips per Straight made this run ──────────────────────────
# card.lua:3435 (`context.before`, `and not context.blueprint`) gains; :3908 (joker_main)
# pays.  Same before/main split as Green Joker.
class _Runner:
    def pre_score(self, inst, ctx):
        if ctx.blueprint:
            return
        if "Straight" in ctx.hand_type:
            inst.state["chips"] = inst.state.get("chips", 0) + 15
    def on_hand_scored(self, inst, ctx):
        ctx.chips += inst.state.get("chips", 0)
JOKER_REGISTRY["j_runner"] = _Runner()

# ── j_ice_cream: +100 chips, -5 chips per hand played, MELTS at zero ─────────
# joker_main pays `extra.chips` (card.lua:3915-3920); the decrement is a separate
# `context.after` branch (:3571, `and not context.blueprint`) that runs after every
# joker_main contribution (state_events.lua:1070).
#
# The melt is the branch the engine used to be missing: `if extra.chips - chip_mod <= 0
# then ... G.jokers:remove_card(self); self:remove()` (:3572-3592) DESTROYS the joker on
# the hand that would take it to zero — it does not clamp.  `extra.chips` is left at its
# last positive value; the card simply stops existing, and the joker slot is freed.
# 100 chips at 5/hand therefore means 20 scoring hands (100, 95, ..., 5 = 1050 chips
# total) and then a free slot, not a permanent 0-chip squatter.
class _IceCream:
    def on_init(self, inst, ctx):
        inst.state["chips"] = 100
    def on_hand_scored(self, inst, ctx):
        ctx.chips += inst.state.get("chips", 100)
    def on_hand_after(self, inst, ctx):
        if ctx.blueprint:
            return
        chips = inst.state.get("chips", 100)
        if chips - 5 <= 0:
            # base.drain_joker_state pops 'destroyed' and calls remove_joker (the slot
            # is freed because joker_slots is untouched).  Same mechanism as Seltzer.
            inst.state["destroyed"] = True
        else:
            inst.state["chips"] = chips - 5
JOKER_REGISTRY["j_ice_cream"] = _IceCream()

# ── j_dna: if first hand has only 1 card, permanent copy added to deck ──────
# TODO: deck modification

# ── j_splash: every played card counts in scoring ───────────────────────────
# TODO: hand eval modification

# ── j_blue_joker: +2 chips per remaining card in deck ───────────────────────
class _BlueJoker:
    def on_hand_scored(self, inst, ctx):
        ctx.chips += 2 * ctx.deck_remaining
JOKER_REGISTRY["j_blue_joker"] = _BlueJoker()

# ── j_sixth_sense: if first hand is single 6, destroy it and create Spectral ─
# TODO: consumable creation

# ── j_constellation: x0.1 mult per Planet card used ─────────────────────────
class _Constellation:
    def on_planet_used(self, inst, planet_name):
        inst.state["mult"] = inst.state.get("mult", 1.0) + 0.1
    def on_hand_scored(self, inst, ctx):
        ctx.mult_mult *= inst.state.get("mult", 1.0)
JOKER_REGISTRY["j_constellation"] = _Constellation()

# ── j_hiker: every played card permanently gains +5 chips ───────────────────
# card.lua:3067 (perma_bonus += 5, per scoring pass). Card.bonus_chips is
# scored in scoring._score_single_card and survives Card.copy().
class _Hiker:
    def on_score_card(self, inst, card, ctx):
        if not card.debuffed:
            card.bonus_chips = getattr(card, "bonus_chips", 0) + 5
JOKER_REGISTRY["j_hiker"] = _Hiker()

# ── j_faceless: earn $5 if 3+ face cards discarded at once ──────────────────
# TODO: batch discard tracking

# ── j_ride_the_bus: +1 mult per consecutive hand without face card, resets ──
# card.lua:3525 (`context.before`, `and not context.blueprint`) scans
# `context.scoring_hand` and resets or gains; :3992-3997 (joker_main) pays.  Same
# before/main split as Green Joker.
class _RideTheBus:
    def pre_score(self, inst, ctx):
        if ctx.blueprint:
            return
        has_face = any(ctx.is_face_card(c) for c in ctx.scoring_cards if not c.debuffed)
        if has_face:
            inst.state["mult"] = 0
        else:
            inst.state["mult"] = inst.state.get("mult", 0) + 1
    def on_hand_scored(self, inst, ctx):
        ctx.mult += inst.state.get("mult", 0)
JOKER_REGISTRY["j_ride_the_bus"] = _RideTheBus()

# ── j_egg: sell to gain $3 of sell value ────────────────────────────────────
# TODO: sell system

# ── j_baron: x1.5 Mult per King held in hand ────────────────────────────────
# card.lua:3286 — held `individual` context, per pass (Mime / Red seal retrigger).
class _Baron:
    def on_held_card(self, inst, card, ctx):
        if card.rank != 13:
            return False
        if not card.debuffed:
            ctx.mult_mult *= 1.5
        return True
JOKER_REGISTRY["j_baron"] = _Baron()

# ── j_oops_all_6s: all cards are considered 6s, doubles probabilities ───────
# TODO: card eval modification
