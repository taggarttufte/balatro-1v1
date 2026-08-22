"""
misc.py — Remaining jokers: retrigger mechanics, hand eval flags,
          blueprint/brainstorm, economy specials, card/consumable creators.

Keyed RNG (W3): every roll names the game's key (mp/rng/keys.py); every created
card goes through mp/rng/generate.py via base.create_consumable / the
generate.* creators so the pool, dedupe (used_jokers) and edition streams are
the real ones. Sentinel strings ("tarot", "common_joker", ...) are gone.
"""
from .base import (
    JOKER_REGISTRY, ScoreContext, JokerInstance, rng_of, prob_roll, is_suit,
    has_consumable_room, create_consumable, joker_sell_value, sort_id_order,
)
from ..round_cards import round_card
from .. import game_keys as _gk

# ════════════════════════════════════════════════════════════════════════════
# RETRIGGER JOKERS
# These set ctx.card_retriggers[i] in on_score_card (fires during card loop).
# ════════════════════════════════════════════════════════════════════════════

# ── j_hack: retrigger 2s, 3s, 4s, 5s ────────────────────────────────────────
class _Hack:
    def on_score_card(self, inst, card, ctx):
        if card.rank in (2, 3, 4, 5) and not card.debuffed:
            i = ctx.scoring_cards.index(card)
            ctx.card_retriggers[i] = ctx.card_retriggers.get(i, 0) + 1
JOKER_REGISTRY["j_hack"] = _Hack()

# ── j_sock_and_buskin: retrigger all face cards ───────────────────────────────
class _SockAndBuskin:
    def on_score_card(self, inst, card, ctx):
        if ctx.is_face_card(card) and not card.debuffed:
            i = ctx.scoring_cards.index(card)
            ctx.card_retriggers[i] = ctx.card_retriggers.get(i, 0) + 1
JOKER_REGISTRY["j_sock_and_buskin"] = _SockAndBuskin()

# ── j_hanging_chad: retrigger the FIRST scoring card 2 extra times ───────────
# card.lua:3352 `context.other_card == context.scoring_hand[1]` — every hand,
# no per-round state (the old `fired` flag made it fire once per ROUND).
class _HangingChad:
    def on_score_card(self, inst, card, ctx):
        if ctx.scoring_cards and card is ctx.scoring_cards[0]:
            i = 0
            ctx.card_retriggers[i] = ctx.card_retriggers.get(i, 0) + 2
JOKER_REGISTRY["j_hanging_chad"] = _HangingChad()

# ── j_dusk: retrigger all cards on last hand of round ────────────────────────
class _Dusk:
    def pre_score(self, inst, ctx):
        if ctx.hands_left == 0:
            for i in range(len(ctx.scoring_cards)):
                ctx.card_retriggers[i] = ctx.card_retriggers.get(i, 0) + 1
JOKER_REGISTRY["j_dusk"] = _Dusk()

# ── j_selzer: retrigger all cards for 10 hands, then self-destructs ─────────
class _Seltzer:
    def pre_score(self, inst, ctx):
        remaining = inst.state.get("hands", 10)
        if remaining > 0:
            for i in range(len(ctx.scoring_cards)):
                ctx.card_retriggers[i] = ctx.card_retriggers.get(i, 0) + 1
    def on_hand_scored(self, inst, ctx):
        inst.state["hands"] = inst.state.get("hands", 10) - 1
        if inst.state["hands"] <= 0:
            inst.state["destroyed"] = True
JOKER_REGISTRY["j_selzer"] = _Seltzer()

# ── j_mime: retrigger all held-in-hand card abilities ────────────────────────
# card.lua:2879 (`repetition` context, cardarea == G.hand): +1 pass for every
# held card that produced an effect (Steel, Baron's King, Shoot the Moon's
# Queen, Raised Fist's lowest card, Reserved Parking's face). The mechanic lives
# in scoring._held_phase; the effect object is a marker.
class _Mime:
    pass
JOKER_REGISTRY["j_mime"] = _Mime()

# ════════════════════════════════════════════════════════════════════════════
# HAND EVAL FLAG JOKERS
# Set ctx flags that hand_eval.py and scoring respect.
# ════════════════════════════════════════════════════════════════════════════

# ── j_pareidolia: all cards count as face cards ───────────────────────────────
class _Pareidolia:
    def pre_score(self, inst, ctx):
        ctx.all_face_cards = True
JOKER_REGISTRY["j_pareidolia"] = _Pareidolia()

# ── j_four_fingers: Flush/Straight valid with 4 cards ────────────────────────
class _FourFingers:
    def pre_score(self, inst, ctx):
        ctx.four_finger_mode = True
JOKER_REGISTRY["j_four_fingers"] = _FourFingers()

# ── j_smeared: Hearts=Diamonds, Spades=Clubs for suit checks ─────────────────
class _SmearedJoker:
    def pre_score(self, inst, ctx):
        ctx.smear_suits = True
JOKER_REGISTRY["j_smeared"] = _SmearedJoker()

# ── j_splash: all played cards count in scoring ───────────────────────────────
class _Splash:
    def pre_score(self, inst, ctx):
        ctx.all_scoring_mode = True
        # Extend scoring_cards to include all played cards
        for card in ctx.all_cards:
            if card not in ctx.scoring_cards and not card.debuffed:
                ctx.scoring_cards.append(card)
JOKER_REGISTRY["j_splash"] = _Splash()

# ── j_shortcut: Straights can skip one rank (gaps of 1 allowed) ──────────────
class _Shortcut:
    def pre_score(self, inst, ctx):
        ctx.shortcut_mode = True  # honoured in hand_eval when flag present
JOKER_REGISTRY["j_shortcut"] = _Shortcut()

# ════════════════════════════════════════════════════════════════════════════
# BLUEPRINT / BRAINSTORM — copy adjacent joker effects
# Recursion guard prevents infinite loop when Blueprint copies Brainstorm
# which copies Blueprint (or vice versa).
# ════════════════════════════════════════════════════════════════════════════

_copy_depth = 0
_MAX_COPY_DEPTH = 3


def _guarded_call(method_name, target, ctx, card=None):
    """Call a joker effect method with recursion depth guard."""
    global _copy_depth
    if _copy_depth >= _MAX_COPY_DEPTH:
        return None
    effect = JOKER_REGISTRY.get(target.key)
    if effect and hasattr(effect, method_name):
        _copy_depth += 1
        try:
            fn = getattr(effect, method_name)
            if card is not None:
                return fn(target, card, ctx)
            return fn(target, ctx)
        finally:
            _copy_depth -= 1
    return None


class _Blueprint:
    """Copies the effect of the joker immediately to the right."""
    def _get_copy_target(self, inst, ctx):
        idx = ctx.jokers.index(inst)
        if idx + 1 < len(ctx.jokers):
            return ctx.jokers[idx + 1]
        return None

    def pre_score(self, inst, ctx):
        target = self._get_copy_target(inst, ctx)
        if target: _guarded_call("pre_score", target, ctx)

    def on_score_card(self, inst, card, ctx):
        target = self._get_copy_target(inst, ctx)
        if target: _guarded_call("on_score_card", target, ctx, card)

    def on_held_card(self, inst, card, ctx):
        target = self._get_copy_target(inst, ctx)
        return bool(_guarded_call("on_held_card", target, ctx, card)) if target else False

    def on_hand_scored(self, inst, ctx):
        target = self._get_copy_target(inst, ctx)
        if target: _guarded_call("on_hand_scored", target, ctx)

JOKER_REGISTRY["j_blueprint"] = _Blueprint()


class _Brainstorm:
    """Copies the effect of the leftmost joker."""
    def _get_copy_target(self, inst, ctx):
        if ctx.jokers and ctx.jokers[0] is not inst:
            return ctx.jokers[0]
        return None

    def pre_score(self, inst, ctx):
        target = self._get_copy_target(inst, ctx)
        if target: _guarded_call("pre_score", target, ctx)

    def on_score_card(self, inst, card, ctx):
        target = self._get_copy_target(inst, ctx)
        if target: _guarded_call("on_score_card", target, ctx, card)

    def on_held_card(self, inst, card, ctx):
        target = self._get_copy_target(inst, ctx)
        return bool(_guarded_call("on_held_card", target, ctx, card)) if target else False

    def on_hand_scored(self, inst, ctx):
        target = self._get_copy_target(inst, ctx)
        if target: _guarded_call("on_hand_scored", target, ctx)

JOKER_REGISTRY["j_brainstorm"] = _Brainstorm()

# ════════════════════════════════════════════════════════════════════════════
# SURVIVABILITY / GAME-STATE JOKERS
# ════════════════════════════════════════════════════════════════════════════

# ── j_mr_bones: prevent death if score >= 25% of required chips ──────────────
class _MrBones:
    def on_hand_scored(self, inst, ctx):
        inst.state["active"] = True  # game.py reads this
        ctx.prevent_loss = True      # always set; game.py validates threshold
JOKER_REGISTRY["j_mr_bones"] = _MrBones()

# ── j_satellite: +$1 per unique Planet card used this run ────────────────────
class _Satellite:
    def on_round_end(self, inst, ctx):
        n = len(inst.state.get("planets_used", set()))
        inst.state["pending_money"] = inst.state.get("pending_money", 0) + n
    def on_planet_used(self, inst, planet_name):
        if "planets_used" not in inst.state:
            inst.state["planets_used"] = set()
        inst.state["planets_used"].add(planet_name)
JOKER_REGISTRY["j_satellite"] = _Satellite()

# ── j_cloud_9: +$1 per 9 in FULL DECK at end of round ────────────────────────
class _Cloud9:
    def on_round_end(self, inst, ctx):
        nines = inst.state.get("deck_nines")
        if nines is None:
            nines = sum(1 for c in ctx.full_deck if c.rank == 9)
        inst.state["pending_money"] = inst.state.get("pending_money", 0) + nines
JOKER_REGISTRY["j_cloud_9"] = _Cloud9()

# ── j_wee (wee joker): permanently gains +8 chips each time a 2 is scored ────
class _Wee:
    def on_score_card(self, inst, card, ctx):
        if card.rank == 2 and not card.debuffed:
            inst.state["chips"] = inst.state.get("chips", 0) + 8
    def on_hand_scored(self, inst, ctx):
        ctx.chips += inst.state.get("chips", 0)
JOKER_REGISTRY["j_wee"] = _Wee()

# ── j_stone: +25 chips per Stone card in full deck ───────────────────────────
class _StoneJoker:
    def on_hand_scored(self, inst, ctx):
        stones = sum(1 for c in ctx.full_deck if c.enhancement == "Stone")
        ctx.chips += 25 * stones
JOKER_REGISTRY["j_stone"] = _StoneJoker()

# ════════════════════════════════════════════════════════════════════════════
# PROBABILISTIC / CONSUMABLE-CREATING JOKERS
# Created cards are drawn through run_state (generate.create_card with the
# joker's key_append) and queued on ctx.pending_consumables as REAL keys.
# ════════════════════════════════════════════════════════════════════════════

# ── j_8_ball: 1 in 4 chance of a Tarot per scored 8 ─────────────────────────
# card.lua:3106-3115 — room check, then `8ball` < normal/4 per scored 8 per
# pass, then create_card('Tarot', ..., '8ba') (stream shared with Purple Seal).
# Stone cards have a random negative id and can never be an 8.
class _EightBall:
    def on_score_card(self, inst, card, ctx):
        if card.rank != 8 or card.debuffed or card.enhancement == "Stone":
            return
        if has_consumable_room(ctx) and prob_roll(ctx, "8ball", 4):
            create_consumable(ctx, "8_ball")
JOKER_REGISTRY["j_8_ball"] = _EightBall()

# ── j_seance: Straight Flush -> random Spectral (card.lua:3787, 'sea') ───────
class _Seance:
    def on_hand_scored(self, inst, ctx):
        if ctx.hand_type == "Straight Flush" and has_consumable_room(ctx):
            create_consumable(ctx, "seance")
JOKER_REGISTRY["j_seance"] = _Seance()

# ── j_riff_raff: when a blind is selected, create 2 Common jokers ────────────
# card.lua:2529-2543 — per joker: room check (jokers + buffer < limit) then
# create_card('Joker', G.jokers, nil, 0, ..., 'rif') -> pool 'Joker1rif<ante>'.
class _RiffRaff:
    def on_blind_selected(self, inst, ctx):
        rs = ctx.run_state
        if rs is None:
            return
        for _ in range(2):
            if len(ctx.jokers) + len(ctx.pending_jokers) >= ctx.joker_slots:
                break
            card = _gk.gen.create_from_spec(rs, "riff_raff")
            ctx.pending_jokers.append(_joker_from_gen(card))
JOKER_REGISTRY["j_riff_raff"] = _RiffRaff()


def _joker_from_gen(card) -> JokerInstance:
    from .base import GEN_EDITION_TO_ENGINE
    return JokerInstance(card.key, GEN_EDITION_TO_ENGINE.get(card.edition, "None"))


# ── j_superposition: Tarot if the played hand is a Straight containing an Ace ─
# card.lua:3762-3785 ('sup').
class _Superposition:
    def on_hand_scored(self, inst, ctx):
        has_ace = any(c.rank == 14 for c in ctx.scoring_cards if not c.debuffed)
        if "Straight" in ctx.hand_type and has_ace and has_consumable_room(ctx):
            create_consumable(ctx, "superposition")
JOKER_REGISTRY["j_superposition"] = _Superposition()

# ── j_sixth_sense: first hand of the round is a single 6 -> destroy it, Spectral
# card.lua:2604-2621 (`destroying_card` context): the 6 is destroyed regardless;
# the Spectral ('sixth') only if there is room. No "used" flag — every round.
class _SixthSense:
    def on_hand_scored(self, inst, ctx):
        if ctx.hands_played != 0 or len(ctx.all_cards) != 1:
            return
        six = ctx.all_cards[0]
        if six.rank != 6 or six.enhancement == "Stone":
            return
        ctx.pending_destroy.append(six)
        if has_consumable_room(ctx):
            create_consumable(ctx, "sixth_sense")
JOKER_REGISTRY["j_sixth_sense"] = _SixthSense()

# ── j_hallucination: 1 in 2 chance of a Tarot whenever a Booster pack is opened
# card.lua:2337 — key 'halu<ante>', odds 2, room check first; create 'hal'.
# Fired by shop._open_booster through base.fire_hook(game, "on_booster_opened").
class _Hallucination:
    def on_booster_opened(self, inst, ctx):
        if has_consumable_room(ctx) and prob_roll(ctx, _gk.gen.Keys.halu(ctx.ante), 2):
            create_consumable(ctx, "hallucination")
JOKER_REGISTRY["j_hallucination"] = _Hallucination()

# ── j_cartomancer: create a Tarot when a blind is selected (card.lua:2545, 'car')
class _Cartomancer:
    def on_blind_selected(self, inst, ctx):
        if has_consumable_room(ctx):
            create_consumable(ctx, "cartomancer")
JOKER_REGISTRY["j_cartomancer"] = _Cartomancer()

# ── j_astronomer: Planet cards and Celestial packs cost $0 ───────────────────
# Passive (card.lua:617 only re-prices the shop). Read through
# base.passive_modifiers(game.jokers)["free_planets"] in shop.py.
class _Astronomer:
    pass
JOKER_REGISTRY["j_astronomer"] = _Astronomer()

# ── j_burnt: upgrade most-played hand at start of round ──────────────────────
class _BurntJoker:
    def on_blind_selected(self, inst, ctx):
        most_played = inst.state.get("most_played")
        if most_played:
            inst.state["planet_upgrade"] = most_played  # game.py applies this
    def on_hand_scored(self, inst, ctx):
        counts = inst.state.setdefault("counts", {})
        counts[ctx.hand_type] = counts.get(ctx.hand_type, 0) + 1
        inst.state["most_played"] = max(counts, key=counts.get)
JOKER_REGISTRY["j_burnt"] = _BurntJoker()

# ── j_invisible: after 2 rounds, SELL to duplicate a random other joker ──────
# card.lua:2371-2395 — `pseudorandom_element(other jokers, 'invisible')` (sort_id
# order, like every element draw over G.jokers.cards), room check
# `#G.jokers.cards <= card_limit` with the Invisible still on the board (= the copy
# may take the slot it vacates); `copy_card(..., strip_edition = negative)`: a
# Negative original is copied WITHOUT its edition, other editions are kept;
# copied Invisible Jokers restart at 0 rounds.  The copy goes through
# `base.add_joker` (drain) -> `run_state.acquire` (Card:set_ability marks used_jokers).
class _InvisibleJoker:
    def on_round_end(self, inst, ctx):
        inst.state["rounds"] = inst.state.get("rounds", 0) + 1
    def on_sell(self, inst, ctx):
        if inst.state.get("rounds", 0) < 2:
            return
        others = [j for j in ctx.jokers if j is not inst]
        if not others or len(others) >= ctx.joker_slots:
            return
        board = sorted(others, key=lambda j: getattr(j, "sort_id", 0))
        chosen, _ = rng_of(ctx).pseudorandom_element(board, "invisible")
        copy = JokerInstance(chosen.key, "None" if chosen.edition == "Negative" else chosen.edition)
        copy.state = {k: (v.copy() if isinstance(v, (list, set, dict)) else v)
                      for k, v in chosen.state.items()}
        copy.state.pop("_init", None)
        if copy.key == "j_invisible":
            copy.state["rounds"] = 0
        ctx.pending_jokers.append(copy)
JOKER_REGISTRY["j_invisible"] = _InvisibleJoker()

# ── j_perkeo: on leaving the shop, create a Negative copy of a random held consumable
# card.lua:2417-2424 — `copy_card(pseudorandom_element(G.consumeables.cards,
# 'perkeo'))` + `set_edition({negative = true})`: the copy takes NO slot
# (card_limit + 1 while it exists).  The engine holds consumables as bare keys, so
# the copy is queued as ``"negative:<key>"`` and ``game._materialize`` emplaces it
# with the slot bump + `negative_consumables` bookkeeping (W5; needs ``consumables``
# in slot order = G.consumeables.cards order, the element draw sorts by sort_id
# which is acquisition order for cards created in sequence).
class _Perkeo:
    def on_shop_leave(self, inst, ctx):
        if not ctx.consumables:
            return
        key, _ = rng_of(ctx).pseudorandom_element(list(ctx.consumables), "perkeo")
        ctx.pending_consumables.append("negative:" + key)
JOKER_REGISTRY["j_perkeo"] = _Perkeo()

# ════════════════════════════════════════════════════════════════════════════
# BOSS BLIND EFFECTS
# ════════════════════════════════════════════════════════════════════════════

# ── j_chicot: disables the Boss Blind's effect ───────────────────────────────
# Passive: card.lua:596 (add_to_deck) / :2492 (setting_blind) call blind:disable().
# game._start_blind checks ownership and neutralises the boss.
class _Chicot:
    pass
JOKER_REGISTRY["j_chicot"] = _Chicot()

# ── j_matador: earn $8 if the Boss Blind's ability triggers this hand ────────
# card.lua:2736 (`debuffed_hand`) and :3719 (`joker_main`) both read
# G.GAME.blind.triggered — game._play_hand sets ctx.boss_triggered for Hook,
# Tooth, Flint, Crimson Heart, Arm, Ox, and fires on_boss_ability_triggered
# for hands the boss rejects (Eye, Mouth, Psychic).
class _Matador:
    def on_hand_scored(self, inst, ctx):
        if ctx.boss_triggered:
            ctx.pending_money += 8
    def on_boss_ability_triggered(self, inst, ctx):
        ctx.pending_money += 8
JOKER_REGISTRY["j_matador"] = _Matador()

# ── j_luchador: sell this to disable the current Boss Blind ──────────────────
# Needs sell-anytime + the boss-disable plumbing (harness xfail); the flag is
# left on the sold instance for W5.
class _Luchador:
    def on_sell(self, inst, ctx):
        inst.state["boss_disabled"] = True
JOKER_REGISTRY["j_luchador"] = _Luchador()

# ════════════════════════════════════════════════════════════════════════════
# DECK MODIFICATION JOKERS
# ════════════════════════════════════════════════════════════════════════════

_SUIT_FROM_LETTER = {"S": "Spades", "H": "Hearts", "C": "Clubs", "D": "Diamonds"}
_RANK_FROM_CHAR = {"T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}


def _card_from_front(front: str, **kw):
    """'S_A' / 'H_T' / 'C_2' (G.P_CARDS key) -> engine Card."""
    from ..card import Card
    suit_l, rank_c = front.split("_")
    rank = _RANK_FROM_CHAR.get(rank_c) or int(rank_c)
    return Card(rank=rank, suit=_SUIT_FROM_LETTER[suit_l], **kw)


# ── j_marble: add a Stone card to the deck when a Blind is selected ──────────
# card.lua:2583 — front from 'marb_fr' over G.P_CARDS (key-string order).
class _Marble:
    def on_blind_selected(self, inst, ctx):
        rs = ctx.run_state
        if rs is None:
            return
        front = _gk.gen.marble_joker(rs)
        ctx.pending_cards.append((_card_from_front(front, enhancement="Stone"), "deck"))
JOKER_REGISTRY["j_marble"] = _Marble()

# ── j_dna: if the first hand of the round is a single card, add a copy to hand ─
# card.lua:2483 (`before`, hands_played == 0, #full_hand == 1) — copy_card into
# G.hand (also a `playing_card_added` event for Hologram).
class _DNA:
    def pre_score(self, inst, ctx):
        if ctx.hands_played != 0 or len(ctx.all_cards) != 1:
            return
        orig = ctx.all_cards[0]
        from ..card import Card
        copy = Card(rank=orig.rank, suit=orig.suit, enhancement=orig.enhancement,
                    edition=orig.edition, seal=orig.seal)
        ctx.pending_cards.append((copy, "hand"))
JOKER_REGISTRY["j_dna"] = _DNA()

# ── j_oops: doubles all listed probabilities ─────────────────────────────────
# Passive: G.GAME.probabilities.normal = 2 ** (owned Oops), kept in sync by
# base.sync_probabilities (run_state.probabilities_normal) and read by every
# prob_roll through ctx.probabilities_normal.
class _OopsAllSixes:
    pass
JOKER_REGISTRY["j_oops"] = _OopsAllSixes()

# ── j_trading: if the first discard of the round is a single card, destroy it, +$3
# card.lua:2802 — `discards_used <= 0 and #full_hand == 1`; no RNG.
class _TradingCard:
    def on_discard(self, inst, cards, ctx):
        if inst.state.get("used"):
            return
        inst.state["used"] = True          # first discard of the round (reset on round end)
        if len(cards) == 1:
            ctx.pending_money += 3
            ctx.pending_destroy.append(cards[0])
    def on_round_end(self, inst, ctx):
        inst.state["used"] = False
JOKER_REGISTRY["j_trading"] = _TradingCard()

# ════════════════════════════════════════════════════════════════════════════
# ECONOMY / GAME STATE JOKERS
# ════════════════════════════════════════════════════════════════════════════

# ── j_merry_andy: +3 discards, -1 hand size (passive, constant while owned) ──
# Applied in game.py _start_blind via base.passive_modifiers
class _MerryAndy:
    pass
JOKER_REGISTRY["j_merry_andy"] = _MerryAndy()

# ── j_troubadour: +2 hand size, -1 hand per round (passive, constant) ────────
class _Troubadour:
    pass
JOKER_REGISTRY["j_troubadour"] = _Troubadour()

# ── j_credit_card: can go up to -$20 in debt ─────────────────────────────────
# Passive (card.lua:593 G.GAME.bankrupt_at -= 20). shop.buy_item reads
# base.passive_modifiers(game.jokers)["bankrupt_at"].
class _CreditCard:
    pass
JOKER_REGISTRY["j_credit_card"] = _CreditCard()

# ── j_turtle_bean: +5 hand size, -1 per round; eaten at 0 ────────────────────
# card.lua:605 (h_size on add) / :2903-2930 (end of round: h_size - 1 <= 0 -> eaten).
# game._start_blind applies state['h_size'] through base.passive_modifiers.
class _TurtleBean:
    def on_round_end(self, inst, ctx):
        h = inst.state.get("h_size", 5)
        if h - 1 <= 0:
            inst.state["destroyed"] = True
        else:
            inst.state["h_size"] = h - 1
JOKER_REGISTRY["j_turtle_bean"] = _TurtleBean()

# ── j_juggler: +1 hand size (passive, constant while owned) ──────────────────
class _Juggler:
    pass
JOKER_REGISTRY["j_juggler"] = _Juggler()

# ── j_drunkard: +1 discard per round (passive, constant while owned) ─────────
class _Drunkard:
    pass
JOKER_REGISTRY["j_drunkard"] = _Drunkard()

# ── j_chaos: 1 free reroll per shop ──────────────────────────────────────────
# Passive (card.lua:601 current_round.free_rerolls += 1). shop.py reads
# base.passive_modifiers(game.jokers)["free_rerolls"].
class _Chaos:
    pass
JOKER_REGISTRY["j_chaos"] = _Chaos()

# ── j_gift: +$1 of sell value to every Joker and Consumable at end of round ──
# card.lua:2993 — every owned joker's extra_value += 1 (consumables too; the
# engine's consumables are bare keys and carry no sell value).
class _GiftCard:
    def on_round_end(self, inst, ctx):
        for j in ctx.jokers:
            j.state["sell_value"] = joker_sell_value(j) + 1
JOKER_REGISTRY["j_gift"] = _GiftCard()

# ── j_egg: gains $3 of sell value each round ─────────────────────────────────
class _Egg:
    def on_round_end(self, inst, ctx):
        inst.state["sell_value"] = joker_sell_value(inst) + 3
JOKER_REGISTRY["j_egg"] = _Egg()

# ── j_delayed_grat: earn $2 per remaining discard if none used by round end ──
class _DelayedGrat:
    def on_round_end(self, inst, ctx):
        if not inst.state.get("discarded"):
            remaining = inst.state.get("discards_left", ctx.discards_left)
            inst.state["pending_money"] = inst.state.get("pending_money", 0) + 2 * remaining
        inst.state["discarded"] = False
    def on_discard(self, inst, cards, ctx):
        inst.state["discarded"] = True
    def on_hand_scored(self, inst, ctx):
        inst.state["discards_left"] = ctx.discards_left
JOKER_REGISTRY["j_delayed_grat"] = _DelayedGrat()

# ── j_faceless: earn $5 if 3+ face cards discarded at once ────────────────────
class _Faceless:
    def on_discard(self, inst, cards, ctx):
        face_count = sum(1 for c in cards if ctx.is_face_card(c))
        if face_count >= 3:
            inst.state["pending_money"] = inst.state.get("pending_money", 0) + 5
JOKER_REGISTRY["j_faceless"] = _Faceless()

# ── j_todo_list: earn $4 if the played hand matches the listed hand ──────────
# card.lua:311-323 (creation: draw 'to_do' over the visible hands in
# pairs(G.GAME.hands) order, redrawing while equal to the old hand) and
# :2975-2982 (end of round: draw over the visible hands MINUS the current one,
# one draw). The candidate order is LuaJIT hash order — generate.HANDS_PAIRS_ORDER.
_BASE_VISIBLE_HANDS = frozenset(h for h in _gk.gen.HANDS_PAIRS_ORDER
                                if h not in ("Flush Five", "Flush House", "Five of a Kind"))


def visible_hands(ctx):
    """G.GAME.hands[k].visible: the 9 starting hands plus any secret hand played."""
    played = {h for h, n in ctx.hand_type_counts.items() if n}
    return _BASE_VISIBLE_HANDS | played


class _ToDoList:
    def on_init(self, inst, ctx):
        inst.state["target"] = _gk.gen.to_do_hand(_PrngState(ctx), visible_hands(ctx),
                                                 previous=inst.state.get("target"))
    def _ensure(self, inst, ctx):
        if "target" not in inst.state:
            self.on_init(inst, ctx)
    def on_hand_scored(self, inst, ctx):
        self._ensure(inst, ctx)
        if ctx.hand_type == inst.state["target"]:
            ctx.pending_money += 4
    def on_round_end(self, inst, ctx):
        self._ensure(inst, ctx)
        cands = [h for h in _gk.gen.visible_hands_in_pairs_order(visible_hands(ctx))
                 if h != inst.state["target"]]
        hand, _ = rng_of(ctx).pseudorandom_element(cands, "to_do")
        inst.state["target"] = hand
JOKER_REGISTRY["j_todo_list"] = _ToDoList()


class _PrngState:
    """Adapter so generate.* helpers that take a RunState can run on a context's PRNG."""
    __slots__ = ("rng",)
    def __init__(self, ctx):
        self.rng = rng_of(ctx)


# ── j_ring_master (Showman): Joker/Tarot/Planet/Spectral cards may appear
# multiple times — generation-side (run_state.showman, set by run_state.acquire).
class _Showman:
    pass
JOKER_REGISTRY["j_ring_master"] = _Showman()

# ── j_diet_cola: sell to create a free Double Tag ────────────────────────────
# Tags are W2/W6 (tags.py). The sold instance carries state['pending_tags'] =
# ['tag_double'] for shop.sell_joker to hand to the tag system.
class _DietCola:
    def on_sell(self, inst, ctx):
        inst.state["pending_tags"] = inst.state.get("pending_tags", []) + ["tag_double"]
JOKER_REGISTRY["j_diet_cola"] = _DietCola()

# ── j_flash: +2 Mult permanently per shop reroll used ────────────────────────
# card.lua:2403 (`reroll_shop`). Fired by shop.reroll_shop through
# base.fire_hook(game, "on_reroll").
class _Flash:
    def on_reroll(self, inst, ctx):
        inst.state["mult"] = inst.state.get("mult", 0) + 2
    def on_hand_scored(self, inst, ctx):
        ctx.mult += inst.state.get("mult", 0)
JOKER_REGISTRY["j_flash"] = _Flash()

# ── j_ceremonial: destroy Joker to right on blind select, gain 2x its sell value as Mult
class _Ceremonial:
    def on_blind_selected(self, inst, ctx):
        inst.state["destroy_right"] = True   # game._start_blind applies after all hooks
    def on_hand_scored(self, inst, ctx):
        ctx.mult += inst.state.get("mult", 0)
JOKER_REGISTRY["j_ceremonial"] = _Ceremonial()

# ── j_midas_mask: all played face cards become Gold during scoring ───────────
class _MidasMask:
    def on_score_card(self, inst, card, ctx):
        if ctx.is_face_card(card) and not card.debuffed:
            card.enhancement = "Gold"
JOKER_REGISTRY["j_midas_mask"] = _MidasMask()

# ── j_certificate: when a blind is selected, add a random card with a random seal to hand
# card.lua:2463-2474 (`first_hand_drawn`): front 'cert_fr' over G.P_CARDS, seal 'certsl'.
class _Certificate:
    def on_first_hand_drawn(self, inst, ctx):     # after the draw (W5), not at blind select
        rs = ctx.run_state
        if rs is None:
            return
        front, seal = _gk.gen.certificate(rs)
        ctx.pending_cards.append((_card_from_front(front, seal=seal or "None"), "hand"))
JOKER_REGISTRY["j_certificate"] = _Certificate()

# ── j_swashbuckler: Mult equals the total sell value of all OTHER owned Jokers ─
# card.lua:4240-4247 (`G.jokers.cards[i] ~= self`).
class _Swashbuckler:
    def on_hand_scored(self, inst, ctx):
        total = sum(joker_sell_value(j) for j in ctx.jokers if j is not inst)
        if total > 0:
            ctx.mult += total
JOKER_REGISTRY["j_swashbuckler"] = _Swashbuckler()

class _CardSharp:
    def on_hand_scored(self, inst, ctx):
        if ctx.hand_type in inst.state.get("played_hands", set()):
            ctx.mult_mult *= 3
        played = inst.state.setdefault("played_hands", set())
        played.add(ctx.hand_type)
    def on_round_end(self, inst, ctx):
        inst.state["played_hands"] = set()
JOKER_REGISTRY["j_card_sharp"] = _CardSharp()

# ── j_reserved_parking: 1 in 2 chance of $1 per face card HELD in hand ───────
# card.lua:3304 — key 'parking', odds 2; held-card `individual` context, so it
# rolls once per face card per held pass (Mime / Red seal re-roll). The roll is
# consumed BEFORE the debuff check, exactly like the Lua.
class _ReservedParking:
    def on_held_card(self, inst, card, ctx):
        if ctx.is_face_card(card) and prob_roll(ctx, "parking", 2):
            if not card.debuffed:
                ctx.pending_money += 1
            return True
        return False
JOKER_REGISTRY["j_reserved_parking"] = _ReservedParking()
