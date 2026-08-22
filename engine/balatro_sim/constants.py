"""
constants.py — Static game tables ported from Balatro source.
All values verified against game source (game.lua, back.lua).
"""

# Hand base chips and mult (level 1)
# Source: G.GAME.hands in game.lua
HAND_BASE = {
    "High Card":       (5,  1),
    "Pair":            (10, 2),
    "Two Pair":        (20, 2),
    "Three of a Kind": (30, 3),
    "Straight":        (30, 4),
    "Flush":           (35, 4),
    "Full House":      (40, 4),
    "Four of a Kind":  (60, 7),
    "Straight Flush":  (100, 8),
    "Five of a Kind":  (120, 12),
    "Flush House":     (140, 14),
    "Flush Five":      (160, 16),
}

# Chips added per hand level (additive)
HAND_LEVEL_CHIPS = {
    "High Card":       10,
    "Pair":            15,
    "Two Pair":        20,
    "Three of a Kind": 20,
    "Straight":        30,
    "Flush":           15,
    "Full House":      25,
    "Four of a Kind":  30,
    "Straight Flush":  40,
    "Five of a Kind":  35,
    "Flush House":     40,
    "Flush Five":      50,
}

# Mult added per hand level (additive)
HAND_LEVEL_MULT = {
    "High Card":       1,
    "Pair":            1,
    "Two Pair":        1,
    "Three of a Kind": 2,
    "Straight":        3,
    "Flush":           2,
    "Full House":      2,
    "Four of a Kind":  3,
    "Straight Flush":  4,
    "Five of a Kind":  3,
    "Flush House":     4,
    "Flush Five":      3,   # was 6 before the 2026-07-29 audit; real value is +3
}

# Blind chip targets by ante (ante 0-8, small/big/boss)
# Source: get_blind_amount(ante) (functions/misc_functions.lua:919-931, white-stake
# scaling) x blind mult (Small 1, Big 1.5, Boss 2 — blind.lua:107).  Ante 0 is real:
# Hieroglyph / Petroglyph at ante 1 run ease_ante(-1) and `ante < 1` returns 100.
# Format: {ante: (small, big, boss)} — boss same as big (varies by boss type)
BLIND_CHIPS = {
    0: (100,   150,   200),
    1: (300,   450,   600),
    2: (800,   1200,  1600),
    3: (2000,  3000,  4000),
    4: (5000,  7500,  10000),
    5: (11000, 16500, 22000),
    6: (20000, 30000, 40000),
    7: (35000, 52500, 70000),
    8: (50000, 75000, 100000),
}
BLIND_MULT = (1.0, 1.5, 2.0)   # Small / Big / Boss


# `G.GAME.modifiers.scaling` ante tables (misc_functions.lua:919-954): 1 = White/Red stake,
# 2 = Green stake and up, 3 = Purple stake and up.  Same endless formula past ante 8.  (W3)
BLIND_AMOUNTS_BY_SCALING = {
    1: (300, 800, 2000, 5000, 11000, 20000, 35000, 50000),
    2: (300, 900, 2600, 8000, 20000, 36000, 60000, 100000),
    3: (300, 1000, 3200, 9000, 25000, 60000, 110000, 200000),
}


def get_blind_amount(ante: int, scaling: int = 1) -> int:
    """``get_blind_amount(ante)`` (misc_functions.lua:919-954): 100 below ante 1, the
    ``scaling`` table through ante 8, then the endless formula.  ``scaling`` is
    ``G.GAME.modifiers.scaling`` (1 default, 2 at Green stake+, 3 at Purple stake+)."""
    import math
    amounts = BLIND_AMOUNTS_BY_SCALING[scaling if scaling in BLIND_AMOUNTS_BY_SCALING else 1]
    if ante < 1:
        return 100
    if ante <= 8:
        return amounts[ante - 1]
    k = 0.75
    a, b, c, d = amounts[7], 1.6, ante - 8, 1 + 0.2 * (ante - 8)
    amount = math.floor(a * (b + (k * c) ** d) ** c)
    amount = amount - amount % (10 ** math.floor(math.log10(amount) - 1))
    return int(amount)


def blind_base_chips(ante: int, blind_idx: int, scaling: int = 1) -> int:
    """Chip target of blind ``blind_idx`` (0 Small / 1 Big / 2 Boss) at ``ante`` before
    any boss-specific multiplier and before the deck's ``ante_scaling`` (Plasma ×2, applied
    by the caller — blind.lua:107 ``get_blind_amount(ante) * mult * ante_scaling``)."""
    if scaling == 1 and ante in BLIND_CHIPS:
        return BLIND_CHIPS[ante][blind_idx]
    return int(get_blind_amount(ante, scaling) * BLIND_MULT[blind_idx])

# Interest: $1 per $5 held, capped at $5/round (i.e., max $25 in bank earns interest)
INTEREST_RATE   = 5    # $1 per $5 held
INTEREST_CAP    = 5    # max $5 interest per round

# Base money awarded for defeating a blind ("$$$" on the blind token).
# Source: balatrowiki.org/w/Blinds. Was entirely missing before 2026-07-29 —
# the sim paid only $1/unused-hand + interest, i.e. ~40% of real income.
BLIND_REWARD = {
    "Small": 3,
    "Big":   4,
    "Boss":  5,
}
SHOWDOWN_BLIND_REWARD = 8   # Showdown (finisher) bosses at ante 8, 16, 24, ...

# Starting hand/discard counts
STARTING_HANDS    = 4
STARTING_DISCARDS = 3
HAND_SIZE         = 8

# Card rank chip values (for scoring)
RANK_CHIPS = {
    2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,
    9: 9, 10: 10, 11: 10, 12: 10, 13: 10, 14: 11,  # J=11, Q=12, K=13, A=14
}

# Suits
SUITS = ["Spades", "Hearts", "Clubs", "Diamonds"]
SUIT_ID = {s: i for i, s in enumerate(SUITS)}

# Ranks (2-14, where 11=J, 12=Q, 13=K, 14=A)
RANKS = list(range(2, 15))
RANK_NAMES = {2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',
              9:'9',10:'10',11:'J',12:'Q',13:'K',14:'A'}

# Enhancements
ENHANCEMENTS = ["None", "Bonus", "Mult", "Wild", "Glass", "Steel", "Stone", "Gold", "Lucky"]

# Editions
EDITIONS = ["None", "Foil", "Holographic", "Polychrome", "Negative"]

# Seals
SEALS = ["None", "Gold", "Red", "Blue", "Purple"]

# Shop reroll cost (increases by $1 each reroll, resets each round)
REROLL_BASE_COST = 5

# Starting money
STARTING_MONEY = 4

# Round end payout: $1 per hand remaining + interest
HAND_PAYOUT = 1

# ════════════════════════════════════════════════════════════════════════════
# Game-key catalogues (Phase 1 W1). Everything below is DERIVED from
# mp/rng/pools.py via game_keys.py — do not hand-edit. Tags / decks / stakes
# are catalogue-only here (no behaviour yet): W6 owns tag effects, W2 wires
# decks/stakes into run setup.
# ════════════════════════════════════════════════════════════════════════════
from .game_keys import (  # noqa: E402
    TAGS, TAG_KEYS, TAG_NAME, TAG_BY_KEY,
    DECKS, DECK_KEYS, DECK_NAME, DECK_BY_KEY,
    STAKES, STAKE_KEYS, STAKE_NAME, STAKE_BY_LEVEL,
    ENHANCEMENT_KEYS, EDITION_KEYS, SEAL_KEYS,
    GAME_VERSION,
)

# Engine display name -> game key for the three card modifier families.
# The engine keeps its short names ("Wild", "Foil", "Red") as the runtime
# representation on Card; these maps are for anything that must talk to the
# generation layer (mp/rng/generate.py) in game keys.
ENHANCEMENT_KEY = {
    "None": None, "Bonus": "m_bonus", "Mult": "m_mult", "Wild": "m_wild",
    "Glass": "m_glass", "Steel": "m_steel", "Stone": "m_stone", "Gold": "m_gold",
    "Lucky": "m_lucky",
}
EDITION_KEY = {
    "None": "e_base", "Foil": "e_foil", "Holographic": "e_holo",
    "Polychrome": "e_polychrome", "Negative": "e_negative",
}
ENHANCEMENT_FROM_KEY = {v: k for k, v in ENHANCEMENT_KEY.items() if v}
EDITION_FROM_KEY = {v: k for k, v in EDITION_KEY.items()}

# ════════════════════════════════════════════════════════════════════════════
# Major League Balatro (Phase 2 W1) — the Multiplayer mod's Attrition gamemode as
# forced by the `majorleague` ruleset.  Ported from $MOD/gamemodes/attrition.lua
# (bans) and $MOD/core.lua:169-203 (lobby defaults); see mp/engine/MLB_NOTES.md.
# ════════════════════════════════════════════════════════════════════════════
MLB_NEMESIS_KEY = "bl_mp_nemesis"      # SMODS.Blind key, $MOD/objects/blinds/nemesis.lua
MLB_NEMESIS_REWARD = 5                 # nemesis.lua: dollars = 5 (paid win OR lose, game.toml:149-152)
MLB_STARTING_LIVES = 4                 # core.lua:175 starting_lives
MLB_PVP_START_ROUND = 2                # core.lua:176 pvp_start_round (Boss slot = Nemesis from ante 2)
MLB_COMEBACK_PER_LIFE = 4              # game.toml:29-30: 4 * cumulative lives lost, once per loss
# attrition.lua:13-33 — applied through MP.ApplyBans (rulesets/_rulesets.lua:198-229):
# every key lands in G.GAME.banned_keys, which the vanilla generation honours in place
# (pool slots become 'UNAVAILABLE' and are resampled; bosses drop out of the eligible set).
MLB_BANNED_JOKERS = ("j_mr_bones", "j_luchador", "j_matador", "j_chicot")
MLB_BANNED_VOUCHERS = ("v_hieroglyph", "v_petroglyph", "v_directors_cut", "v_retcon")
MLB_BANNED_TAGS = ("tag_boss",)
MLB_BANNED_BLINDS = ("bl_wall", "bl_final_vessel")
MLB_BANNED_KEYS = frozenset(MLB_BANNED_JOKERS + MLB_BANNED_VOUCHERS + MLB_BANNED_TAGS + MLB_BANNED_BLINDS)
