"""
consumables.py — Planet, Tarot, and Spectral card definitions and apply logic.

Usage:
    from balatro_sim.consumables import apply_planet, apply_tarot, apply_spectral
    apply_planet(game, "c_mercury")            # Pair +1 level
    apply_tarot(game, "c_hermit")              # Double money
    apply_tarot(game, "c_star", target_indices=[0, 1, 2])  # 3 cards → Diamonds
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .game import BalatroGame

from . import game_keys
from .game_keys import gen as _gen           # mp.rng.generate — created cards + target picks (W2)
from .constants import INTEREST_CAP, ENHANCEMENT_FROM_KEY

_SUIT_FULL = {"S": "Spades", "H": "Hearts", "C": "Clubs", "D": "Diamonds"}
_RANK_FULL = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "T": 10,
              "J": 11, "Q": 12, "K": 13, "A": 14}
_EDITION_FROM_GEN = {None: "None", "foil": "Foil", "holo": "Holographic",
                     "polychrome": "Polychrome", "negative": "Negative"}


def _free_consumable_slots(game: "BalatroGame") -> int:
    """Free slots while the consumable being used still occupies its slot in
    ``consumable_hand`` (the engine pops it after ``apply_*`` succeeds), i.e. what the Lua's
    ``card_limit - #G.consumeables.cards`` sees once the used card has left the area."""
    return game.consumable_slots - (len(game.consumable_hand) - 1)


def _hand_sorted(game: "BalatroGame") -> list:
    """``G.hand.cards`` as ``pseudorandom_element`` indexes them: sorted by sort_id (creation order)."""
    return sorted(game.hand, key=lambda c: c.id)

# ════════════════════════════════════════════════════════════════════════════
# PLANET CARDS — each upgrades one hand type by 1 level
# ════════════════════════════════════════════════════════════════════════════

# Keys, names and hand types come from mp/rng/pools.py (game keys `c_*`; the
# sim used a `pl_` prefix before the Phase 1 re-key).
PLANET_HAND = dict(game_keys.PLANET_HAND)
PLANET_NAME = dict(game_keys.PLANET_NAME)
ALL_PLANETS = list(game_keys.PLANET_KEYS)   # game pool order


def apply_planet(game: "BalatroGame", planet_key: str) -> bool:
    """Upgrade the associated hand type by 1 level. Returns True on success."""
    hand = PLANET_HAND.get(planet_key)
    if not hand:
        return False
    game.planet_levels[hand] = game.planet_levels.get(hand, 1) + 1
    # Fire satellite jokers
    for j in game.jokers:
        effect = _get_effect(j.key)
        if effect and hasattr(effect, "on_planet_used"):
            effect.on_planet_used(j, planet_key)
    # Track for Fortune Teller / Constellation, and G.GAME.last_tarot_planet (The Fool)
    game.planets_used.append(planet_key)
    game.run_state.last_tarot_planet = planet_key
    return True


# ════════════════════════════════════════════════════════════════════════════
# TAROT CARDS — 22 cards + The Fool
# ════════════════════════════════════════════════════════════════════════════

# NOTE the game's own key for The Hierophant is the misspelt `c_heirophant`
# (game.lua); the sim had it "fixed" to c_hierophant, which is not a game key.
TAROT_NAME = dict(game_keys.TAROT_NAME)
ALL_TAROTS = list(game_keys.TAROT_KEYS)     # game pool order

# Enhancement each tarot applies to cards (for Magician, Empress, etc.)
TAROT_ENHANCEMENT = {
    "c_magician":    "Lucky",
    "c_empress":     "Mult",
    "c_heirophant":  "Bonus",   # game spelling
    "c_lovers":      "Wild",
    "c_chariot":     "Steel",
    "c_justice":     "Glass",
    "c_devil":       "Gold",
    "c_tower":       "Stone",
}

# Suit each tarot converts cards to
TAROT_SUIT = {
    "c_star":  "Diamonds",
    "c_moon":  "Clubs",
    "c_sun":   "Hearts",
    "c_world": "Spades",
}


def apply_tarot(
    game: "BalatroGame",
    tarot_key: str,
    target_indices: list[int] | None = None,
) -> bool:
    """
    Apply a Tarot card effect.

    target_indices: indices into game.hand for card-targeting tarots.
    Returns True on success.
    """
    targets = [game.hand[i] for i in (target_indices or []) if i < len(game.hand)]

    # G.GAME.last_tarot_planet (misc_functions.lua: set for every Tarot/Planet but The Fool)
    if tarot_key != "c_fool":
        game.run_state.last_tarot_planet = tarot_key

    # Enhancement tarots (1-2 cards)
    if tarot_key in TAROT_ENHANCEMENT:
        enh = TAROT_ENHANCEMENT[tarot_key]
        for card in targets[:2]:
            card.enhancement = enh
        game.tarots_used.append(tarot_key)
        _fire_tarot_hooks(game, tarot_key)
        return True

    # Suit conversion tarots (up to 3 cards)
    if tarot_key in TAROT_SUIT:
        suit = TAROT_SUIT[tarot_key]
        for card in targets[:3]:
            card.suit = suit
        game.tarots_used.append(tarot_key)
        _fire_tarot_hooks(game, tarot_key)
        return True

    # Special tarots
    if tarot_key == "c_fool":
        # Create a copy of the last Tarot or Planet used (G.GAME.last_tarot_planet): forced
        # key, no pool draw (card.lua:1377). Unusable when nothing was used yet or that card
        # was The Fool itself.
        last = game.run_state.last_tarot_planet
        if not last or last == "c_fool":
            return False
        game._sync_run_state()
        cg = _gen.fool(game.run_state)
        if cg is None:
            return False
        game.consumable_hand.append(cg.key)
        game.run_state.acquire(cg.key)
        game.tarots_used.append(tarot_key)
        _fire_tarot_hooks(game, tarot_key)
        return True

    if tarot_key in ("c_high_priestess", "c_emperor"):
        # min(2, free slots) x create_card('Planet'/'Tarot', G.consumeables, ..., 'pri'/'emp')
        # (card.lua:1401-1413); each created card marks used_jokers, so two different cards.
        spec = "high_priestess" if tarot_key == "c_high_priestess" else "emperor"
        n = min(2, _free_consumable_slots(game))
        game._sync_run_state()
        for _ in range(n):
            cg = _gen.create_from_spec(game.run_state, spec)
            game.consumable_hand.append(cg.key)
            game.run_state.acquire(cg.key)
        game.tarots_used.append(tarot_key)
        _fire_tarot_hooks(game, tarot_key)
        return True

    if tarot_key == "c_hermit":
        # Double money, max $20 gain
        gain = min(game.dollars, 20)
        game.dollars += gain
        game.tarots_used.append(tarot_key)
        _fire_tarot_hooks(game, tarot_key)
        return True

    if tarot_key == "c_wheel_of_fortune":
        # card.lua:1466-1486 (W3): usable only with >= 1 editionless joker
        # (card.lua:1534); then THREE draws on the one 'wheel_of_fortune' stream —
        # `pseudorandom('wheel_of_fortune') < probabilities.normal / 4` (Oops!
        # doubles it), `pseudorandom_element(editionless jokers, ...)` in sort_id
        # order, `poll_edition('wheel_of_fortune', nil, true, true)` (polychrome >
        # 0.85, holo > 0.5, else foil; never negative). A "Nope!" consumes only
        # the first draw. generate.wheel_of_fortune is the oracle-verified port.
        eligible = sorted((j for j in game.jokers if j.edition == "None"),
                          key=lambda j: getattr(j, "sort_id", 0))
        if not eligible:
            return False
        from .jokers.base import sync_probabilities
        sync_probabilities(game)
        hit = _gen.wheel_of_fortune(game.run_state, [j.key for j in eligible])
        if hit is not None:
            idx, edition = hit
            eligible[idx].edition = _EDITION_FROM_GEN.get(edition, "Foil")
        game.tarots_used.append(tarot_key)
        _fire_tarot_hooks(game, tarot_key)
        return True

    if tarot_key == "c_strength":
        # Increase rank of up to 2 cards by 1 (wraps A back to 2)
        for card in targets[:2]:
            card.rank = (card.rank % 14) + 1 if card.rank < 14 else 2
        game.tarots_used.append(tarot_key)
        _fire_tarot_hooks(game, tarot_key)
        return True

    if tarot_key == "c_hanged_man":
        # Destroy up to 2 selected cards
        for card in targets[:2]:
            game.remove_card(card)
        game.tarots_used.append(tarot_key)
        _fire_tarot_hooks(game, tarot_key)
        return True

    if tarot_key == "c_death":
        # Convert left card to copy of right card (both selected)
        if len(targets) >= 2:
            left, right = targets[0], targets[1]
            left.rank = right.rank
            left.suit = right.suit
            left.enhancement = right.enhancement
        game.tarots_used.append(tarot_key)
        _fire_tarot_hooks(game, tarot_key)
        return True

    if tarot_key == "c_temperance":
        # Give $ equal to total joker sell value (max $50)
        sell_total = sum(j.state.get("sell_value", 2) for j in game.jokers)
        game.dollars += min(sell_total, 50)
        game.tarots_used.append(tarot_key)
        _fire_tarot_hooks(game, tarot_key)
        return True

    if tarot_key == "c_judgement":
        # create_card('Joker', G.jokers, false, nil, ..., 'jud') (card.lua:1418): rolls
        # 'rarity<ante>jud', draws 'Joker<r>jud<ante>', edition 'edijud<ante>'.
        # Unusable without a free joker slot (card.lua:1557-1562).
        if len(game.jokers) >= game.joker_slots:
            return False
        game.grant_created("judgement")
        game.tarots_used.append(tarot_key)
        _fire_tarot_hooks(game, tarot_key)
        return True

    return False


def _fire_tarot_hooks(game: "BalatroGame", tarot_key: str):
    """Notify jokers that a Tarot was used (e.g. Fortune Teller)."""
    for j in game.jokers:
        effect = _get_effect(j.key)
        if effect and hasattr(effect, "on_tarot_used"):
            effect.on_tarot_used(j, None)


# ════════════════════════════════════════════════════════════════════════════
# SPECTRAL CARDS — 18 powerful deck-modifying cards
# ════════════════════════════════════════════════════════════════════════════

# Game keys are `c_*` (the sim used `s_*` before the re-key).
SPECTRAL_NAME = dict(game_keys.SPECTRAL_NAME)
ALL_SPECTRALS = list(game_keys.SPECTRAL_KEYS)   # game pool order


def apply_spectral(
    game: "BalatroGame",
    spectral_key: str,
    target_indices: list[int] | None = None,
) -> bool:
    """Apply a Spectral card effect. Returns True on success."""
    targets = [game.hand[i] for i in (target_indices or []) if i < len(game.hand)]

    if spectral_key in ("c_familiar", "c_grim", "c_incantation"):
        # card.lua:1292-1338: destroy ONE RANDOM card in hand ('random_destroy' over the hand
        # in sort_id order -- not a selected card), then create 3 / 2 / 4 cards: rank+suit on
        # '<name>_create', enhancement from the Enhanced pool minus Stone on 'spe_card'.
        # The new cards go to the hand.  Needs more than one card in hand (card.lua:1570-1577).
        if len(game.hand) <= 1:
            return False
        from .card import Card
        name = spectral_key[2:]
        hand = _hand_sorted(game)
        res = _gen.spectral_create_cards(game.run_state, name, len(hand))
        _remove_card(game, hand[res["destroy_index"]])
        for front, enh in res["created"]:
            s, r = front.split("_")
            c = Card(rank=_RANK_FULL[r], suit=_SUIT_FULL[s])
            c.enhancement = ENHANCEMENT_FROM_KEY.get(enh, "None")
            game.add_card(c, to_draw_pile=False)
            game.hand.append(c)
        return True

    if spectral_key == "c_talisman":
        # Add Gold seal to 1 selected card
        for card in targets[:1]:
            card.seal = "Gold"
        return True

    if spectral_key == "c_aura":
        # "Add Foil, Holographic or Polychrome to 1 selected card (in hand)":
        # poll_edition('aura', nil, true, true) (card.lua:1211) -- a PLAYING card, editionless.
        if not targets or targets[0].edition != "None":
            return False
        targets[0].edition = _EDITION_FROM_GEN.get(_gen.aura(game.run_state), "Foil")
        return True

    if spectral_key == "c_wraith":
        # create_card('Joker', G.jokers, nil, 0.99, ..., 'wra') -> Rare (card.lua:1457);
        # then money is set to $0.  Unusable without a free joker slot.
        if len(game.jokers) >= game.joker_slots:
            return False
        game.grant_created("wraith")
        game.dollars = 0
        return True

    if spectral_key == "c_sigil":
        # pseudorandom_element({'S','H','D','C'}, 'sigil') (card.lua:1233); all cards in hand
        if len(game.hand) <= 1:
            return False
        suit = _SUIT_FULL[_gen.sigil(game.run_state)]
        for card in game.hand:
            if card.enhancement != "Stone":
                card.suit = suit
        return True

    if spectral_key == "c_ouija":
        # pseudorandom_element({'2'..'A'}, 'ouija') (card.lua:1247); all cards in hand; -1 hand size
        if len(game.hand) <= 1:
            return False
        rank = _RANK_FULL[_gen.ouija(game.run_state)]
        for card in game.hand:
            if card.enhancement != "Stone":
                card.rank = rank
        # G.hand:change_size(-1) is PERMANENT (and stacks): base for every later blind
        # drops too (W5: was reset by _start_blind each blind)
        game.hand_size_mod -= 1
        game.hand_size = max(1, game.hand_size - 1)
        return True

    if spectral_key == "c_ectoplasm":
        # "Add Negative to a random Joker, -1 hand size": pseudorandom_element over the
        # editionless jokers on 'ectoplasm' (card.lua:1473-1486). Negative = +1 joker slot.
        eligible = [j for j in game.jokers if j.edition == "None"]
        if not eligible:
            return False
        idx = _gen.ectoplasm(game.run_state, [j.key for j in eligible])
        eligible[idx].edition = "Negative"
        game.joker_slots += 1
        game.hand_size_mod -= 1          # permanent, stacks (W5)
        game.hand_size = max(1, game.hand_size - 1)
        return True

    if spectral_key == "c_immolate":
        # card.lua:1340-1345: hand copy sorted by sort_id, pseudoshuffle('immolate'), destroy
        # the first 5; +$20
        if len(game.hand) <= 1:
            return False
        for card in _gen.immolate(game.run_state, _hand_sorted(game)):
            _remove_card(game, card)
        game.dollars += 20
        return True

    if spectral_key == "c_ankh":
        # pseudorandom_element(G.jokers.cards, 'ankh_choice') (card.lua:1434): copy THAT joker
        # (Negative is not copied) and destroy every OTHER joker -- the chosen one stays.
        if not game.jokers or game.joker_slots <= 1:
            return False
        from .jokers.base import JokerInstance
        idx = _gen.ankh(game.run_state, [j.key for j in game.jokers])
        keep = game.jokers[idx]
        copy = JokerInstance(keep.key, "None" if keep.edition == "Negative" else keep.edition)
        copy.state = {k: (v.copy() if isinstance(v, (list, set, dict)) else v) for k, v in keep.state.items()}
        for j in game.jokers:
            if j is not keep:
                game.run_state.remove_owned(j.key)
        game.jokers = [keep, copy]
        game.run_state.acquire(copy.key)
        return True

    if spectral_key == "c_deja_vu":
        # Add Red seal to 1 selected card
        for card in targets[:1]:
            card.seal = "Red"
        return True

    if spectral_key == "c_hex":
        # pseudorandom_element(editionless jokers, 'hex') -> Polychrome; destroy all others
        eligible = [j for j in game.jokers if j.edition == "None"]
        if not eligible:
            return False
        lucky = eligible[_gen.hex_(game.run_state, [j.key for j in eligible])]
        lucky.edition = "Polychrome"
        for j in game.jokers:
            if j is not lucky:
                game.run_state.remove_owned(j.key)
        game.jokers = [lucky]
        return True

    if spectral_key == "c_trance":
        # Add Blue seal to 1 selected card
        for card in targets[:1]:
            card.seal = "Blue"
        return True

    if spectral_key == "c_medium":
        # Add Purple seal to 1 selected card
        for card in targets[:1]:
            card.seal = "Purple"
        return True

    if spectral_key == "c_cryptid":
        # Create 2 copies of 1 selected card
        if targets:
            from .card import Card
            orig = targets[0]
            for _ in range(2):
                c = Card(rank=orig.rank, suit=orig.suit)
                c.enhancement = orig.enhancement
                c.edition = orig.edition
                c.seal = orig.seal
                game.add_card(c)
        return True

    if spectral_key == "c_soul":
        # create_card('Joker', G.jokers, true, ..., 'sou'): consumes 'rarity<ante>sou' (discarded),
        # draws the bare 'Joker4' legendary stream, edition 'edisou<ante>'. Needs a free slot.
        if len(game.jokers) >= game.joker_slots:
            return False
        game.grant_created("soul")
        return True

    if spectral_key == "c_black_hole":
        # Upgrade every hand type by 1 level
        for hand in list(game.planet_levels.keys()):
            game.planet_levels[hand] = game.planet_levels.get(hand, 1) + 1
        return True

    return False


# ════════════════════════════════════════════════════════════════════════════
# VOUCHERS — passive upgrades purchased in shop
# ════════════════════════════════════════════════════════════════════════════

# Game keys from pools (32; base/upgrade interleaved in game order). Effects:
#   v_overstock_norm / v_overstock_plus   +1 card slot in shop (each)
#   v_clearance_sale / v_liquidation      -25% / -50% shop prices
#   v_hone / v_glow_up                    2x / 4x edition rates (W2: generation)
#   v_reroll_surplus / v_reroll_glut      reroll costs $2 less (each)
#   v_crystal_ball                        +1 consumable slot
#   v_omen_globe                          spectrals in Arcana packs (W2: generation)
#   v_telescope / v_observatory           planet for most-played hand / x1.5 (W2/W5)
#   v_grabber / v_nacho_tong              +1 hand per round (each)
#   v_wasteful / v_recyclomancy           +1 discard per round (each)
#   v_tarot_merchant / v_tarot_tycoon     tarot shop rate x2 / x4 (W2: generation)
#   v_planet_merchant / v_planet_tycoon   planet shop rate x2 / x4 (W2: generation)
#   v_seed_money / v_money_tree           interest cap $10 / $20 per round
#   v_blank / v_antimatter                nothing / +1 joker slot
#   v_magic_trick / v_illusion            playing cards in shop / with editions (W2)
#   v_hieroglyph / v_petroglyph           -1 ante, -1 hand / -1 ante, -1 discard
#   v_directors_cut / v_retcon            reroll boss once per ante ($10) / unlimited
#   v_paint_brush / v_palette             +1 hand size (each)
VOUCHER_NAME = dict(game_keys.VOUCHER_NAME)
VOUCHER_REQUIRES = dict(game_keys.VOUCHER_REQUIRES)   # upgrade -> [base]; W2 gates on this
ALL_VOUCHERS = list(game_keys.VOUCHER_KEYS)           # game pool order


def apply_voucher(game: "BalatroGame", voucher_key: str) -> bool:
    """Apply a voucher's permanent effect. Returns True on success."""
    if voucher_key in game.vouchers:
        return False  # already owned

    game.vouchers.add(voucher_key)

    # ── Generation-relevant effects live on run_state (Card:apply_to_run, card.lua:1880-1971;
    # GENERATION_SPEC §1): redeemed set, shop rates, edition rate, shelf size, ante.
    rs = game.run_state
    rs.used_vouchers.add(voucher_key)
    rate = game_keys.pools.SHOP_RATE_BY_VOUCHER.get(voucher_key)
    if rate:
        setattr(rs, rate[0], rate[1])

    if voucher_key == "v_overstock_norm":
        rs.shop_joker_max += 1       # change_shop_size(1): the new slot fills from the same streams
        _fill_new_shop_slot(game)
    elif voucher_key == "v_overstock_plus":
        rs.shop_joker_max += 1
        _fill_new_shop_slot(game)
    elif voucher_key == "v_clearance_sale":
        game.shop_discount = min(game.shop_discount + 0.25, 0.5)
    elif voucher_key == "v_liquidation":
        game.shop_discount = min(game.shop_discount + 0.25, 0.5)
    elif voucher_key == "v_reroll_surplus":
        game.reroll_discount += 2
    elif voucher_key == "v_reroll_glut":
        game.reroll_discount += 2
    elif voucher_key == "v_crystal_ball":
        game.consumable_slots += 1
    elif voucher_key == "v_grabber":
        game.base_hands += 1
        game.hands_left = min(game.hands_left + 1, game.base_hands)
    elif voucher_key == "v_nacho_tong":
        game.base_hands += 1
        game.hands_left = min(game.hands_left + 1, game.base_hands)
    elif voucher_key == "v_wasteful":
        game.base_discards += 1
    elif voucher_key == "v_recyclomancy":
        game.base_discards += 1
    elif voucher_key == "v_hieroglyph":
        # ease_ante(-1), unclamped: at ante 1 the run really goes to ante 0 (blind base 100,
        # generation keys '...0' — constants.blind_base_chips / get_blind_amount).  W5.
        _ease_ante(game, -1)
        game.base_hands = max(1, game.base_hands - 1)
    elif voucher_key == "v_petroglyph":
        _ease_ante(game, -1)
        game.base_discards = max(0, game.base_discards - 1)
    elif voucher_key == "v_paint_brush":
        game.hand_size += 1
    elif voucher_key == "v_palette":
        game.hand_size += 1
    elif voucher_key == "v_directors_cut":
        # "Reroll Boss Blind once per ante, $10 each": the {"type": "reroll_boss"} action at
        # blind select (BalatroGame.can_reroll_boss / _reroll_boss -> generate.reroll_boss).
        # Was modelled as a free SHOP reroll before W2 (REKEY_NOTES §6.11).
        pass
    # ── Added in the Phase 1 re-key (were missing from the sim entirely) ─────
    elif voucher_key == "v_seed_money":
        # Raise the interest cap to $10/round (i.e. interest on up to $50 held).
        game.interest_cap = max(getattr(game, "interest_cap", INTEREST_CAP), 10)
    elif voucher_key == "v_money_tree":
        # Raise the interest cap to $20/round (interest on up to $100 held).
        game.interest_cap = max(getattr(game, "interest_cap", INTEREST_CAP), 20)
    elif voucher_key == "v_blank":
        pass  # "Does nothing?" — genuinely no effect (it is also the empty-pool fallback)
    elif voucher_key == "v_antimatter":
        game.joker_slots += 1
    elif voucher_key == "v_retcon":
        # "Reroll Boss Blind at any time, $10 each": unlimited reroll_boss actions
        # (BalatroGame.can_reroll_boss).
        pass

    return True


def _ease_ante(game: "BalatroGame", mod: int) -> None:
    """``ease_ante(mod)`` (common_events.lua:191-203): only ``round_resets.ante`` moves.
    The ante's boss (``round_resets.blind_choices.Boss``) and tags were drawn at the
    last Cash Out and are NOT redrawn — so the engine's boss stays pinned to the new
    ante number instead of being re-rolled by ``_prepare_next_blind`` (W5)."""
    game.ante += mod
    game.run_state.ante = game.ante
    if getattr(game, "_boss_blind_ante", None) is not None:
        game._boss_blind_ante = game.ante


def _fill_new_shop_slot(game: "BalatroGame"):
    """Overstock bought mid-shop: ``change_shop_size(1)`` (common_events.lua:1097-1118) fills
    the new slot immediately with one more ``create_card_for_shop`` on the same streams."""
    shop = getattr(game, "_shop_gen", None)
    if shop is None or getattr(game, "state", None) is None:
        return
    from .shop import shop_item_from_gen, SHELF_KINDS
    game._sync_run_state()
    game.run_state.tags = game.tag_state.keys()
    card = _gen._fill_shop_slot(game.run_state)
    game._absorb_tag_triggers()
    if card is None:
        return
    shop.cards.append(card)
    item = shop_item_from_gen(card)
    # keep shelf items first, then vouchers / boosters
    shelf_end = 0
    for i, it in enumerate(game.current_shop):
        if it.kind in SHELF_KINDS:
            shelf_end = i + 1
    game.current_shop.insert(shelf_end, item)


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

def _remove_card(game: "BalatroGame", card):
    """Permanently destroy a card. Delegates to game.remove_card so the card
    leaves full_deck too — otherwise a 'destroyed' card reappears next blind."""
    game.remove_card(card)


def _get_effect(key: str):
    from .jokers.base import JOKER_REGISTRY
    return JOKER_REGISTRY.get(key)
