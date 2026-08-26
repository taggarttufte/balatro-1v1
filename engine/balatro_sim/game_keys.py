"""
game_keys.py — the engine's view of `rng/pools.py`, the single source of truth
for every game key / name / rarity / cost / eligibility flag.

Phase 1 W1 (re-key): the engine speaks the game's own keys (`j_*`, `c_*`, `v_*`,
`bl_*`, `p_*`, `tag_*`, `b_*`, `stake_*`). Nothing in this module is typed by
hand; every table is derived from `pools` at import time so the engine cannot
drift from the oracle-derived catalogue again (before the re-key only 7/150
jokers agreed with the game on key+rarity+cost).

`pools` lives outside the engine package (rng/pools.py). It is imported as
`rng.pools` when the repo root is on sys.path (the normal case), otherwise the
repo root is appended and the import retried, otherwise it is loaded straight
from its file path — it is pure data with no imports, so every route yields
identical tables.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]       # repo root (engine/balatro_sim/..)
_POOLS_FILE = _REPO_ROOT / "rng" / "pools.py"


def _load_pools():
    try:
        from rng import pools as _p          # repo root on sys.path
        return _p
    except ImportError:
        pass
    if str(_REPO_ROOT) not in sys.path:
        sys.path.append(str(_REPO_ROOT))
        try:
            from rng import pools as _p
            return _p
        except ImportError:
            pass
    spec = importlib.util.spec_from_file_location("mp_rng_pools_standalone", _POOLS_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pools = _load_pools()

def _load_gen():
    """rng.generate (RunState, PseudoRandom, create_card, ...) with the same fallbacks as pools."""
    try:
        from rng import generate as _g
        return _g
    except ImportError:
        pass
    if str(_REPO_ROOT) not in sys.path:
        sys.path.append(str(_REPO_ROOT))
    from rng import generate as _g
    return _g


gen = _load_gen()

def _load_core():
    try:
        from rng import core as _c
        return _c
    except ImportError:
        if str(_REPO_ROOT) not in sys.path:
            sys.path.append(str(_REPO_ROOT))
        from rng import core as _c
        return _c


core = _load_core()
normalize_seed = core.normalize_seed

_SEED_CORPUS = "123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"   # the game's seed corpus: 1-9A-Z, no 0, 8 chars


def seed_from_int(n: int) -> str:
    """Deterministic int -> 8-char game seed (base-35 over the game's corpus). Lets the old
    `BalatroGame(seed=int)` call sites keep working while producing a real, normalizable seed."""
    n = abs(int(n))
    out = []
    for _ in range(8):
        n, r = divmod(n, len(_SEED_CORPUS))
        out.append(_SEED_CORPUS[r])
    return "".join(reversed(out))

GAME_VERSION = pools.GAME_VERSION

# ── Jokers ───────────────────────────────────────────────────────────────────
RARITY_NAME = {1: "Common", 2: "Uncommon", 3: "Rare", 4: "Legendary"}
RARITY_ID = {v: k for k, v in RARITY_NAME.items()}

JOKERS = pools.JOKERS                                  # 150, in `order`
JOKER_BY_KEY = pools.JOKER_BY_KEY
JOKER_KEYS = [j["key"] for j in JOKERS]
JOKER_KEYS_BY_RARITY = {
    RARITY_NAME[r]: list(keys) for r, keys in pools.JOKERS_BY_RARITY.items()   # already key lists
}
JOKER_NAME = {j["key"]: j["name"] for j in JOKERS}
JOKER_RARITY = {j["key"]: RARITY_NAME[j["rarity"]] for j in JOKERS}
JOKER_COST = {j["key"]: j["cost"] for j in JOKERS}
JOKER_BY_NAME = {j["name"]: j["key"] for j in JOKERS}

# ── Consumables ──────────────────────────────────────────────────────────────
TAROTS = pools.TAROTS                                  # 22
PLANETS = pools.PLANETS                                # 12
SPECTRALS = pools.SPECTRALS                            # 18
TAROT_KEYS = [c["key"] for c in TAROTS]
PLANET_KEYS = [c["key"] for c in PLANETS]
SPECTRAL_KEYS = [c["key"] for c in SPECTRALS]
TAROT_NAME = {c["key"]: c["name"] for c in TAROTS}
PLANET_NAME = {c["key"]: c["name"] for c in PLANETS}
SPECTRAL_NAME = {c["key"]: c["name"] for c in SPECTRALS}
PLANET_HAND = {c["key"]: c["hand_type"] for c in PLANETS}
HAND_PLANET = {v: k for k, v in PLANET_HAND.items()}
CONSUMABLE_COST = {c["key"]: c["cost"] for c in TAROTS + PLANETS + SPECTRALS}
CONSUMABLE_SET = {c["key"]: c["set"] for c in TAROTS + PLANETS + SPECTRALS}

# ── Vouchers ─────────────────────────────────────────────────────────────────
VOUCHERS = pools.VOUCHERS                              # 32, base/upgrade interleaved
VOUCHER_KEYS = [v["key"] for v in VOUCHERS]
VOUCHER_NAME = {v["key"]: v["name"] for v in VOUCHERS}
VOUCHER_COST = {v["key"]: v["cost"] for v in VOUCHERS}
VOUCHER_REQUIRES = pools.VOUCHER_REQUIRES              # key -> [prerequisite keys]
VOUCHER_UPGRADE_OF = pools.VOUCHER_UPGRADE_OF          # base key -> upgrade key

# ── Boosters ─────────────────────────────────────────────────────────────────
# pools has 32 centers = 13 (kind, size) types × cosmetic art variants
# (`_1.._4`). The engine keys packs by TYPE, `p_<kind>_<size>`; the variant
# suffix is cosmetic (unseeded math.random in the game) and is stripped.
def booster_type_key(center_key: str) -> str:
    """'p_arcana_normal_3' -> 'p_arcana_normal'. Idempotent on type keys."""
    parts = center_key.split("_")
    if parts[-1].isdigit():
        parts = parts[:-1]
    return "_".join(parts)


BOOSTER_TYPES: dict[str, dict] = {}
for _b in pools.BOOSTERS:
    _t = booster_type_key(_b["key"])
    if _t not in BOOSTER_TYPES:
        BOOSTER_TYPES[_t] = {
            "key": _t, "name": _b["name"], "kind": _b["kind"], "size": _b["size"],
            "cost": _b["cost"], "cards": _b["extra"], "choose": _b["choose"],
            "weight": 0.0, "centers": [],
        }
    BOOSTER_TYPES[_t]["weight"] += _b["weight"]
    BOOSTER_TYPES[_t]["centers"].append(_b["key"])
BOOSTER_TYPE_KEYS = list(BOOSTER_TYPES)                # 13
BOOSTER_CENTER_KEYS = [b["key"] for b in pools.BOOSTERS]  # 32

# ── Blinds ───────────────────────────────────────────────────────────────────
BLINDS = pools.BLINDS
BLIND_BY_KEY = pools.BLIND_BY_KEY
BOSS_BLINDS = pools.BOSS_BLINDS                        # 28, by order
BOSS_KEYS = [b["key"] for b in BOSS_BLINDS]
BOSS_KEYS_REGULAR = [b["key"] for b in pools.BOSS_BLINDS_REGULAR]    # 23
BOSS_KEYS_SHOWDOWN = [b["key"] for b in pools.BOSS_BLINDS_SHOWDOWN]  # 5
BOSS_KEYS_ALPHA = pools.BOSS_KEYS_ALPHA                # the game's draw order
BOSS_NAME = {b["key"]: b["name"] for b in BOSS_BLINDS}
BOSS_MIN_ANTE = {b["key"]: b["boss"]["min"] for b in BOSS_BLINDS}
BOSS_MAX_ANTE = {b["key"]: b["boss"]["max"] for b in BOSS_BLINDS}
BOSS_CHIP_MULT_RAW = {b["key"]: b["mult"] for b in BOSS_BLINDS}     # vs small blind base
BOSS_DOLLARS = {b["key"]: b["dollars"] for b in BOSS_BLINDS}

# ── Tags / decks / stakes (catalogue only; behaviour is W6 / W2) ─────────────
TAGS = pools.TAGS                                      # 24
TAG_KEYS = [t["key"] for t in TAGS]
TAG_NAME = {t["key"]: t["name"] for t in TAGS}
TAG_BY_KEY = pools.TAG_BY_KEY
DECKS = pools.BACKS                                    # 15
DECK_KEYS = [d["key"] for d in DECKS]
DECK_NAME = {d["key"]: d["name"] for d in DECKS}
DECK_BY_KEY = pools.BACK_BY_KEY
STAKES = pools.STAKES                                  # 8
STAKE_KEYS = [s["key"] for s in STAKES]
STAKE_NAME = {s["key"]: s["name"] for s in STAKES}
STAKE_BY_LEVEL = pools.STAKE_BY_LEVEL

# ── Enhancements / editions / seals (game keys alongside the engine's names) ─
ENHANCEMENT_KEYS = [e["key"] for e in pools.ENHANCEMENTS]
EDITION_KEYS = [e["key"] for e in pools.EDITIONS]
SEAL_KEYS = [s["key"] for s in pools.SEALS]


def is_joker(key: str) -> bool:     return key in JOKER_BY_KEY
def is_tarot(key: str) -> bool:     return key in TAROT_NAME
def is_planet(key: str) -> bool:    return key in PLANET_NAME
def is_spectral(key: str) -> bool:  return key in SPECTRAL_NAME
def is_voucher(key: str) -> bool:   return key in VOUCHER_NAME
def is_booster(key: str) -> bool:   return key in BOOSTER_TYPES
def is_boss(key: str) -> bool:      return key in BOSS_NAME


__all__ = [n for n in dir() if n.isupper() or n.startswith("is_") or n == "booster_type_key" or n == "pools"]
