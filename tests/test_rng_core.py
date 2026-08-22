"""Bit-exact parity tests for ``rng.core`` / ``rng.luajit_random`` against the
real thing: Balatro's own ``pseudohash``/``pseudoseed``/``pseudorandom``/
``pseudorandom_element``/``pseudoshuffle`` Lua executing inside LuaJIT 2.1
(via ``lupa``).

Ground truth is regenerated live whenever ``lupa`` and the extracted game Lua
(``mp/_reference/balatro_src``) are present; otherwise the cached copy at
``mp/tests/fixtures/rng_ground_truth.json`` is used.  When both are present
the live data is also diffed against the cache, so a LuaJIT/game-source change
cannot go unnoticed.

Every recorded double is stored as its 16-hex-digit IEEE bit pattern, so
"match" means bit-for-bit (including NaN sign/payload), not "close".

Regenerate the cache with::

    python mp/tests/test_rng_core.py --regen

Force the cache-only path (to verify it) with ``MP_RNG_NO_ORACLE=1``.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import struct
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
MP_ROOT = HERE.parent
if str(MP_ROOT) not in sys.path:
    sys.path.insert(0, str(MP_ROOT))

from rng.core import (  # noqa: E402
    PseudoRandom,
    lcg_step,
    normalize_seed,
    pseudohash,
    pseudoseed_predict,
)
from rng.luajit_random import FIXED_SEED_STATE, LuaJITRandom  # noqa: E402

MISC_FUNCTIONS_LUA = MP_ROOT / "_reference" / "balatro_src" / "functions" / "misc_functions.lua"
FIXTURE_PATH = HERE / "fixtures" / "rng_ground_truth.json"

# --------------------------------------------------------------------------------------
# bit-pattern helpers
# --------------------------------------------------------------------------------------

_pack_d = struct.Struct("<d").pack
_unpack_d = struct.Struct("<d").unpack


def bits(x: float) -> str:
    """IEEE-754 bit pattern of a double as 16 hex digits (big-endian order)."""
    return _pack_d(float(x))[::-1].hex()


def unbits(h: str) -> float:
    return _unpack_d(bytes.fromhex(h)[::-1])[0]


# --------------------------------------------------------------------------------------
# the oracle: Balatro's Lua inside LuaJIT
# --------------------------------------------------------------------------------------

# Extracted verbatim from the game file at runtime; never stored in the repo.
_LUA_FUNCTIONS = ("SWAP", "pseudoshuffle", "pseudorandom_element", "random_string",
                  "pseudohash", "pseudoseed", "pseudorandom")

_LUA_PRELUDE = r"""
-- The oracle runs with the JIT OFF.  The interpreter + C library are the
-- reference semantics; and LuaJIT traces assume differently-typed FFI
-- pointers never alias, which silently breaks double<->uint64 type punning
-- once a helper gets hot (stale loads -> duplicated/incorrect bit patterns).
jit.off()
G = {GAME = {pseudorandom = {}}, SETTINGS = {}}
local ffi = require('ffi')
local dbuf = ffi.new('double[1]')
local HEX = {}
for i = 0, 255 do HEX[i] = string.format('%02x', i) end

-- IEEE bit pattern of a double as 16 hex digits (big-endian digit order),
-- via raw byte copies only -- no type-punned loads.
function BITS(x)
  dbuf[0] = x
  local raw = ffi.string(dbuf, 8)  -- little-endian bytes
  local out = {}
  for i = 8, 1, -1 do out[#out+1] = HEX[string.byte(raw, i)] end
  return table.concat(out)
end
-- LuaJIT's very first math.random() in this runtime (lj_prng_seed_fixed state);
-- captured before anything below seeds the generator.
UNSEEDED_FIRST_BITS = BITS(math.random())
function UNBITS(h)
  local bytes = {}
  for i = 8, 1, -1 do bytes[#bytes+1] = string.char(tonumber(string.sub(h, 2*i-1, 2*i), 16)) end
  ffi.copy(dbuf, table.concat(bytes), 8)
  return dbuf[0]
end
-- Harness self-check: consecutive draws must never round-trip to the same pattern.
function SELF_CHECK(n)
  local dup = 0
  for i = 1, n do
    math.randomseed(i * 0.001)
    local a, b = BITS(math.random()), BITS(math.random())
    if a == b then dup = dup + 1 end
    if BITS(UNBITS(a)) ~= a or BITS(UNBITS(b)) ~= b then dup = dup + 1 end
  end
  return dup
end

-- Mirrors Game:start_run (game.lua:2164-2168) for a fresh, unseeded-by-save run.
function START_RUN(seed)
  G.GAME.pseudorandom = {}
  G.GAME.pseudorandom.seed = seed
  for k, v in pairs(G.GAME.pseudorandom) do if v == 0 then G.GAME.pseudorandom[k] = pseudohash(k..G.GAME.pseudorandom.seed) end end
  G.GAME.pseudorandom.hashed_seed = pseudohash(G.GAME.pseudorandom.seed)
  return BITS(G.GAME.pseudorandom.hashed_seed)
end
function STATE_BITS(key) return BITS(G.GAME.pseudorandom[key]) end
function SET_STATE(key, h) G.GAME.pseudorandom[key] = UNBITS(h) end

-- One keyed draw as the game does it, recording every intermediate.
-- Returns: state-after-LCG, pseudoseed value, then a sequence of math.random
-- outputs from that single seeding: (), (1,6), (3), (-5,5), (1,1000000), ()
function DRAW(key)
  local ps = pseudoseed(key)
  local st = G.GAME.pseudorandom[key]
  math.randomseed(ps)
  local r1 = math.random()
  local r2 = math.random(1, 6)
  local r3 = math.random(3)
  local r4 = math.random(-5, 5)
  local r5 = math.random(1, 1000000)
  local r6 = math.random()
  return BITS(st), BITS(ps), BITS(r1), r2, r3, r4, r5, BITS(r6)
end

-- pseudorandom() exactly as the game calls it (string key), float and ranged forms.
function PR_FLOAT(key) return BITS(pseudorandom(key)) end
function PR_INT(key, m, n) return pseudorandom(key, m, n) end

-- Raw LuaJIT generator: seed from a bit pattern, then a fixed op sequence.
function LJ_SEQ(h)
  math.randomseed(UNBITS(h))
  local out = {}
  out[#out+1] = BITS(math.random())
  out[#out+1] = BITS(math.random())
  out[#out+1] = math.random(1, 6)
  out[#out+1] = math.random(52)
  out[#out+1] = math.random(-5, 5)
  out[#out+1] = math.random(1, 1000000)
  out[#out+1] = math.random(0, 1)
  out[#out+1] = math.random(1, 2147483648)
  out[#out+1] = math.random(1)
  out[#out+1] = math.random(0)
  out[#out+1] = math.random(2, 2)
  out[#out+1] = BITS(math.random())
  return out
end
function LJ_UNSEEDED_FIRST() return UNSEEDED_FIRST_BITS end

function PH_BITS(s) return BITS(pseudohash(s)) end
function LCG_BITS(h)
  local x = UNBITS(h)
  return BITS(math.abs(tonumber(string.format("%.13f", (2.134453429141 + x*1.72431234) % 1))))
end
function FMT13(h) return string.format("%.13f", UNBITS(h)) end
function TONUM_BITS(s) return BITS(tonumber(s)) end
function PREDICT_BITS(key, pseed) return BITS(pseudoseed(key, pseed)) end

-- pseudorandom_element on an array of strings (Lua sorts keys 1..n numerically).
function ELEM_ARRAY(key, csv)
  local t = {}
  for item in string.gmatch(csv, '[^,]+') do t[#t+1] = item end
  local v, k = pseudorandom_element(t, pseudoseed(key))
  return v, k
end
-- pseudorandom_element on a string-keyed table (sorted by key string).
function ELEM_MAP(key, csv)
  local t = {}
  for item in string.gmatch(csv, '[^,]+') do t[item] = {name = item} end
  local v, k = pseudorandom_element(t, pseudoseed(key))
  return k
end
-- pseudorandom_element on an array of card-like tables with sort_id, given in
-- scrambled order (Lua sorts by sort_id before indexing).
function ELEM_SORTID(key, csv)
  local t = {}
  for item in string.gmatch(csv, '[^,]+') do t[#t+1] = {sort_id = tonumber(item)} end
  local v, k = pseudorandom_element(t, pseudoseed(key))
  return v.sort_id
end
-- pseudoshuffle on an array of plain tables (no sort_id) tagged 1..n; returns
-- the resulting order.  (The Lua indexes list[1].sort_id, so elements must be
-- tables -- the game only ever shuffles Card lists.)
function SHUFFLE(key, n)
  local t = {}
  for i = 1, n do t[i] = {v = i} end
  pseudoshuffle(t, pseudoseed(key))
  local out = {}
  for i, c in ipairs(t) do out[i] = tostring(c.v) end
  return table.concat(out, ',')
end
-- pseudoshuffle on card-like tables with sort_id in scrambled order (Lua sorts first).
function SHUFFLE_SORTID(key, csv)
  local t = {}
  for item in string.gmatch(csv, '[^,]+') do t[#t+1] = {sort_id = tonumber(item)} end
  pseudoshuffle(t, pseudoseed(key))
  local out = {}
  for i, c in ipairs(t) do out[i] = tostring(c.sort_id) end
  return table.concat(out, ',')
end
function SEED_KEY_BITS() return BITS(pseudoseed('seed')) end
"""


def _extract_lua_functions(src: str, names=_LUA_FUNCTIONS) -> str:
    """Pull top-level ``function NAME(...) ... end`` blocks out of the game file."""
    lines = src.splitlines()
    chunks = []
    for name in names:
        start = next((i for i, l in enumerate(lines) if re.match(r"^function %s\(" % re.escape(name), l)), None)
        if start is None:
            raise RuntimeError("function %s not found in %s" % (name, MISC_FUNCTIONS_LUA))
        end = next(i for i in range(start + 1, len(lines)) if re.match(r"^end\s*$", lines[i]))
        chunks.append("\n".join(lines[start:end + 1]))
    return "\n\n".join(chunks)


class LuaOracle:
    """Balatro's RNG Lua, running in LuaJIT 2.1 via lupa."""

    def __init__(self):
        from lupa import luajit21 as lj

        self.L = lj.LuaRuntime()
        src = MISC_FUNCTIONS_LUA.read_text(encoding="utf-8", errors="replace")
        self.L.execute(_extract_lua_functions(src))
        self.L.execute(_LUA_PRELUDE)
        g = self.L.globals()
        self.version = self.L.execute("return jit.version")
        for name in ("SELF_CHECK", "START_RUN", "STATE_BITS", "SET_STATE", "DRAW", "PR_FLOAT", "PR_INT", "LJ_SEQ",
                     "LJ_UNSEEDED_FIRST", "PH_BITS", "LCG_BITS", "FMT13", "TONUM_BITS", "PREDICT_BITS",
                     "ELEM_ARRAY", "ELEM_MAP", "ELEM_SORTID", "SHUFFLE", "SHUFFLE_SORTID", "SEED_KEY_BITS"):
            setattr(self, name, getattr(g, name))
        assert int(self.SELF_CHECK(3000)) == 0, "oracle BITS/UNBITS helper is not round-trip safe"
        # round-trip spot checks incl. -0.0, NaN sign, denormals (inside Lua: lupa
        # would convert integral doubles such as -0.0 to Python ints on the way out)
        bad = self.L.execute("""
            local bad = {}
            for _, h in ipairs({'3ff0000000000000', '8000000000000000', '0000000000000001', '7ff8000000000000',
                                'fff8000000000000', '7ff0000000000000', 'fff0000000000000', '3fe992ead03ac1e0'}) do
              if BITS(UNBITS(h)) ~= h then bad[#bad+1] = h end
            end
            return table.concat(bad, ',')
        """)
        assert bad == "", "oracle round-trip failed for %s" % bad


_oracle_cache = {}


def get_oracle():
    """Session-cached oracle, or None if unavailable."""
    if "o" not in _oracle_cache:
        try:
            if os.environ.get("MP_RNG_NO_ORACLE"):  # force the cached-fixture path
                raise RuntimeError("MP_RNG_NO_ORACLE set")
            if not MISC_FUNCTIONS_LUA.is_file():
                raise FileNotFoundError(MISC_FUNCTIONS_LUA)
            _oracle_cache["o"] = LuaOracle()
        except Exception as exc:  # lupa missing, Lua missing, ...
            _oracle_cache["o"] = None
            _oracle_cache["err"] = repr(exc)
    return _oracle_cache["o"]


# --------------------------------------------------------------------------------------
# ground-truth corpus
# --------------------------------------------------------------------------------------

# Real game keys (misc_functions/common_events/card.lua call sites) incl. the
# digit-suffixed and "_resample" forms, plus edge cases.
KEYS = [
    "Joker1", "Joker2", "Joker3", "Joker4", "rarity1", "rarity3", "Tarot1", "Tarot5_resample2",
    "Planet1", "Spectral2", "front", "front1", "frontsho1", "shuffle", "nr1", "nr2", "boss",
    "Voucher1", "Voucher1_resample1", "Tag1", "Tag2_resample3", "edition_generic", "edisho1",
    "cdt1", "stdset1", "stdseal1", "soul_Tarot1", "lucky_mult", "lucky_money", "wheel_of_fortune",
    "glass", "misprint", "bloodstone", "gros_michel", "space", "8ball", "halu1", "erratic",
    "anc1", "idol1", "to_do", "stdpc1", "cert_fr", "marb_fr", "Planet1_resample10",
    "", "a", "Z", "joker1", "JOKER1", "Joker10", "Joker1 ", " Joker1",
    "x" * 64, "Tarot" * 20 + "_resample99",
]

# Seeds: game-format (8 chars from [1-9A-Z]), short, special, verbatim-lowercase.
_SPECIAL_SEEDS = ["TUTORIAL", "AAAAAAAA", "7LB2WVPK", "ALEEB", "A", "7", "", "O0O0O0O0", "abcdefgh",
                  "ZZZZZZZZ", "11111111", "1", "IHAVENOS", "OOOOOOOO", "seed", "Joker1"]


def make_seeds(n_random: int = 36) -> list[str]:
    rnd = random.Random(20260820)
    corpus = "123456789ABCDEFGHIJKLMNPQRSTUVWXYZ"  # what random_string() can produce
    out = list(_SPECIAL_SEEDS)
    while len(out) < len(_SPECIAL_SEEDS) + n_random:
        s = "".join(rnd.choice(corpus) for _ in range(8))
        if s not in out:
            out.append(s)
    return out


def make_lj_seeds() -> list[float]:
    rnd = random.Random(1)
    special = [0.0, -0.0, 1.0, -1.0, 0.5, 0.123456789, float("nan"), -float("nan"), float("inf"),
               float("-inf"), 5e-324, -5e-324, 1e-300, 1e300, 1.7976931348623157e308,
               -1.7976931348623157e308, -math.e / math.pi, 2.0 ** 53, 2.0 ** 63, 2.0 ** 64, 12345678.9,
               0.7991842333777264, 0.9999999999999999, 1e-13, 3.0, 100.0, -2.7182818284590452354]
    # the (x + hashed)/2 values the game actually produces are in [0, 1)
    out = special + [rnd.random() for _ in range(120)] + [rnd.uniform(-1e9, 1e9) for _ in range(60)]
    out += [rnd.random() * 10.0 ** rnd.randint(-300, 300) for _ in range(60)]
    return out


def make_lcg_inputs() -> list[float]:
    rnd = random.Random(2)
    out = [0.0, 1.0, 0.5, 0.25, 1e-14, 4e-14, 5e-14, 6e-14, 1.5e-13, 0.99999999999995, 0.99999999999996,
           0.9999999999999999, 0.123456789012345, 0.5000000000000499, 0.5000000000000501]
    out += [k / 16384 for k in range(1, 64, 2)]  # exact ties at the 14th decimal
    out += [k / 8192 for k in range(1, 32, 2)]
    out += [rnd.random() for _ in range(3000)]
    return out


def make_fmt_inputs(n: int) -> list[float]:
    rnd = random.Random(3)
    out = [k / 16384 for k in range(0, 16384, 7)] + [k / 2 ** 20 for k in range(0, 2 ** 20, 4099)]
    out += [float("%.14f" % rnd.random()) for _ in range(2000)]  # near 14-digit boundaries
    out += [float("%.13f" % rnd.random()) + 5e-14 for _ in range(2000)]
    out += [rnd.random() for _ in range(n)]
    return out


ELEM_ARRAY_ITEMS = ["S", "H", "D", "C", "X", "Y", "Z"]
ELEM_MAP_ITEMS = ["%s_%s" % (s, r) for s in "HCDS" for r in "23456789TJQKA"] + ["H_2_extra", "A", "zz", "Zz"]
SORTID_SCRAMBLED = [37, 2, 99, 14, 5, 61, 8, 23, 70, 1, 45, 12, 33, 90, 3, 58, 27, 19, 80, 6]
SHUFFLE_SIZES = [0, 1, 2, 3, 4, 5, 8, 13, 52, 100]
N_DRAWS = 3


def generate_ground_truth(o: LuaOracle, n_seeds_random: int = 36) -> dict:
    seeds = make_seeds(n_seeds_random)
    gt = {
        "meta": {"luajit": o.version, "balatro": "1.0.1o", "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "n_seeds": len(seeds), "n_keys": len(KEYS), "n_draws": N_DRAWS},
        "unseeded_first": o.LJ_UNSEEDED_FIRST(),
        "luajit_random": [],
        "pseudohash": [],
        "lcg": [],
        "fmt13": [],
        "runs": [],
    }
    # Raw generator.
    for d in make_lj_seeds():
        tbl = o.LJ_SEQ(bits(d))
        seq = [tbl[i] for i in range(1, len(tbl) + 1)]
        gt["luajit_random"].append({"seed": bits(d), "seq": seq})
    # pseudohash on assorted inputs (bytes for the non-ASCII cases).
    ph_inputs = (KEYS + seeds + [k + s for k in KEYS[:12] for s in seeds[:12]]
                 + ["\x00", "\x00abc", "abc\x00", "\x7f", "\t\n", " " * 10])
    for s in ph_inputs:
        gt["pseudohash"].append({"s": s, "v": o.PH_BITS(s.encode("latin-1"))})
    for raw in (b"\xff", b"\x80\x81", b"ab\xe9", bytes(range(1, 256))):
        gt["pseudohash"].append({"bytes": raw.hex(), "v": o.PH_BITS(raw)})
    # LCG step (format %.13f + tonumber + abs).
    for x in make_lcg_inputs():
        gt["lcg"].append({"x": bits(x), "v": o.LCG_BITS(bits(x))})
    for x in make_fmt_inputs(2000):
        s = o.FMT13(bits(x))
        gt["fmt13"].append({"x": bits(x), "s": s, "back": o.TONUM_BITS(s)})
    # Full runs.
    for seed in seeds:
        run = {"seed": seed, "hashed_seed": o.START_RUN(seed), "keys": {}, "predict": {},
               "elements": [], "shuffles": [], "pr": []}
        for key in KEYS:
            draws = []
            for _ in range(N_DRAWS):
                st, ps, r1, r2, r3, r4, r5, r6 = o.DRAW(key)
                draws.append({"state": st, "ps": ps, "r": r1, "r_1_6": int(r2), "r_3": int(r3),
                              "r_m5_5": int(r4), "r_1_1e6": int(r5), "r2": r6})
            run["keys"][key] = {"hash": o.PH_BITS((key + seed).encode("latin-1")), "draws": draws}
        for key in ("Joker4", "Joker1", ""):
            run["predict"][key] = o.PREDICT_BITS(key, seed)
        for key in ("sigil", "ouija", "familiar_create", "front", "erratic", "boss", "cert_fr"):
            v, k = o.ELEM_ARRAY(key, ",".join(ELEM_ARRAY_ITEMS))
            run["elements"].append([key, {"array": [v, int(k)],
                                          "map": o.ELEM_MAP(key, ",".join(ELEM_MAP_ITEMS)),
                                          "sortid": int(o.ELEM_SORTID(key, ",".join(map(str, SORTID_SCRAMBLED))))}])
        for n in SHUFFLE_SIZES:
            res = o.SHUFFLE("nr1", n)
            run["shuffles"].append(["nr1:%d" % n, [int(x) for x in res.split(",")] if res else []])
        res = o.SHUFFLE("immolate", 5)
        run["shuffles"].append(["immolate:5", [int(x) for x in res.split(",")]])
        run["shuffles"].append(["sortid", [int(x) for x in o.SHUFFLE_SORTID("shuffle", ",".join(map(str, SORTID_SCRAMBLED))).split(",")]])
        # pseudorandom() via the game's own entry point, interleaved keys.
        pr = []
        for key in ("lucky_mult", "glass", "Joker1", "lucky_mult", "misprint", "8ball"):
            pr.append([key, o.PR_FLOAT(key), int(o.PR_INT(key, 1, 100)), int(o.PR_INT(key, 0, 23))])
        run["pr"] = pr
        run["seed_key"] = o.SEED_KEY_BITS()  # pseudoseed('seed') == raw math.random()
        gt["runs"].append(run)
    return gt


def load_ground_truth() -> tuple[dict, str]:
    """(ground truth, source) where source is 'live' or 'cache'."""
    o = get_oracle()
    if o is not None:
        return generate_ground_truth(o), "live"
    if FIXTURE_PATH.is_file():
        return json.loads(FIXTURE_PATH.read_text(encoding="utf-8")), "cache"
    pytest.skip("no oracle (lupa + balatro_src) and no cached fixture at %s" % FIXTURE_PATH)


_GT = {}


@pytest.fixture(scope="module")
def gt():
    if "gt" not in _GT:
        _GT["gt"], _GT["src"] = load_ground_truth()
    return _GT["gt"]


# --------------------------------------------------------------------------------------
# tests: LuaJIT generator
# --------------------------------------------------------------------------------------


def test_fixed_seed_state_matches_luajit_constants():
    """seed(0.0) must reproduce lj_prng_seed_fixed()'s precomputed constants."""
    r = LuaJITRandom()
    r.seed(0.0)
    assert r.get_state() == FIXED_SEED_STATE
    r2 = LuaJITRandom()
    r2.seed_reference(0.0)
    assert r2.get_state() == FIXED_SEED_STATE


def test_table_seed_equals_reference_seed():
    rnd = random.Random(7)
    cases = [rnd.random() for _ in range(2000)] + [rnd.uniform(-1e12, 1e12) for _ in range(500)]
    cases += [0.0, -0.0, float("nan"), float("inf"), -float("inf"), 5e-324, -math.e / math.pi]
    a, b = LuaJITRandom(), LuaJITRandom()
    for d in cases:
        a.seed(d)
        b.seed_reference(d)
        assert a.get_state() == b.get_state(), bits(d)


def test_unseeded_first_draw(gt):
    assert bits(LuaJITRandom().random()) == gt["unseeded_first"]


def test_luajit_random_sequences(gt):
    r = LuaJITRandom()
    fails = []
    for case in gt["luajit_random"]:
        r.seed(unbits(case["seed"]))
        got = [bits(r.random()), bits(r.random()), r.random_int(1, 6), r.random_index(52), r.random_int(-5, 5),
               r.random_int(1, 1000000), r.random_int(0, 1), r.random_int(1, 2147483648), r.random_index(1),
               r.random_index(0), r.random_int(2, 2), bits(r.random())]
        exp = [v if isinstance(v, str) else int(v) for v in case["seq"]]
        if got != exp:
            fails.append((case["seed"], got, exp))
    assert not fails, fails[:5]


# --------------------------------------------------------------------------------------
# tests: pseudohash / LCG step / string formatting parity
# --------------------------------------------------------------------------------------


def test_pseudohash(gt):
    fails = []
    for case in gt["pseudohash"]:
        s = bytes.fromhex(case["bytes"]) if "bytes" in case else case["s"]
        got = bits(pseudohash(s))
        if got != case["v"]:
            fails.append((s, got, case["v"]))
    assert not fails, fails[:5]


def test_pseudohash_edge_semantics():
    assert pseudohash("") == 1.0
    assert pseudohash("a") == pseudohash(b"a")
    # toxic path: NaN propagates and carries LuaJIT's sign bit
    assert bits(pseudohash("\x00")) != "fff8000000000000"  # byte 0 alone is benign: pi*1 % 1
    x = lcg_step(float("nan"))
    assert x != x


def test_lcg_step(gt):
    fails = []
    for case in gt["lcg"]:
        got = bits(lcg_step(unbits(case["x"])))
        if got != case["v"]:
            fails.append((case["x"], got, case["v"]))
    assert not fails, fails[:5]


def test_format_13f_and_tonumber_parity(gt):
    fails = []
    for case in gt["fmt13"]:
        x = unbits(case["x"])
        s = "%.13f" % x
        if s != case["s"] or bits(float(case["s"])) != case["back"]:
            fails.append((case["x"], s, case["s"], bits(float(case["s"])), case["back"]))
    assert not fails, fails[:5]


@pytest.mark.skipif(get_oracle() is None, reason="needs live LuaJIT oracle")
def test_format_13f_bulk_live():
    """Brute-force '%.13f' parity on a larger sample than the cache holds."""
    o = get_oracle()
    rnd = random.Random(11)
    xs = make_fmt_inputs(60000) + [rnd.random() * 3 for _ in range(20000)]
    fails = 0
    for x in xs:
        if o.FMT13(bits(x)) != "%.13f" % x:
            fails += 1
    assert fails == 0


# --------------------------------------------------------------------------------------
# tests: full keyed chain
# --------------------------------------------------------------------------------------


def _check_run(run: dict) -> list:
    fails = []
    seed = run["seed"]
    p = PseudoRandom(seed)
    if bits(p.hashed_seed) != run["hashed_seed"]:
        fails.append(("hashed_seed", seed, bits(p.hashed_seed), run["hashed_seed"]))
    rng = p.rng
    for key, rec in run["keys"].items():
        if bits(pseudohash(key + seed)) != rec["hash"]:
            fails.append(("hash", seed, key))
        for d_i, d in enumerate(rec["draws"]):
            ps = p.pseudoseed(key)
            st = p.get_key_state(key)
            rng.seed(ps)
            got = {"state": bits(st), "ps": bits(ps), "r": bits(rng.random()), "r_1_6": rng.random_int(1, 6),
                   "r_3": rng.random_index(3), "r_m5_5": rng.random_int(-5, 5),
                   "r_1_1e6": rng.random_int(1, 1000000), "r2": bits(rng.random())}
            if got != d:
                fails.append(("draw", seed, key, d_i, got, d))
    for key, exp in run["predict"].items():
        if bits(pseudoseed_predict(key, seed)) != exp:
            fails.append(("predict", seed, key))
    for key, exp in run["elements"]:
        v, i = p.pseudorandom_element(ELEM_ARRAY_ITEMS, key)
        if [v, i + 1] != exp["array"]:
            fails.append(("elem_array", seed, key, [v, i + 1], exp["array"]))
        table = {k: {"name": k} for k in ELEM_MAP_ITEMS}
        _, k = p.pseudorandom_element(table, key)
        if k != exp["map"]:
            fails.append(("elem_map", seed, key, k, exp["map"]))
        cards = [{"sort_id": s} for s in sorted(SORTID_SCRAMBLED)]
        c, _ = p.pseudorandom_element(cards, key)
        if c["sort_id"] != exp["sortid"]:
            fails.append(("elem_sortid", seed, key, c["sort_id"], exp["sortid"]))
    for name, exp in run["shuffles"]:
        if name == "sortid":
            cards = [{"sort_id": s} for s in sorted(SORTID_SCRAMBLED)]
            p.pseudoshuffle(cards, "shuffle")
            got = [c["sort_id"] for c in cards]
        else:
            key, n = name.split(":")
            lst = list(range(1, int(n) + 1))
            p.pseudoshuffle(lst, key)
            got = lst
        if got != exp:
            fails.append(("shuffle", seed, name, got, exp))
    for key, r_f, r_i, r_j in run["pr"]:
        got = [key, bits(p.pseudorandom(key)), p.pseudorandom(key, 1, 100), p.pseudorandom(key, 0, 23)]
        if got != [key, r_f, r_i, r_j]:
            fails.append(("pseudorandom", seed, got, [key, r_f, r_i, r_j]))
    if bits(p.pseudoseed("seed")) != run["seed_key"]:
        fails.append(("seed_key", seed))
    return fails


def test_full_chain_all_seeds(gt):
    all_fails = []
    for run in gt["runs"]:
        all_fails.extend(_check_run(run))
    n_values = sum(len(r["keys"]) * N_DRAWS * 8 for r in gt["runs"])
    assert not all_fails, "%d mismatches (of ~%d values); first: %r" % (len(all_fails), n_values, all_fails[:3])


def test_mapping_element_with_sort_id_values():
    """Dict whose values carry sort_id must be ordered by sort_id, not by key."""
    p = PseudoRandom("AAAAAAAA")
    table = {i + 1: {"sort_id": s} for i, s in enumerate(SORTID_SCRAMBLED)}
    q = p.clone()
    v, k = p.pseudorandom_element(table, "hook")
    cards = [{"sort_id": s} for s in sorted(SORTID_SCRAMBLED)]
    v2, _ = q.pseudorandom_element(cards, "hook")
    assert v["sort_id"] == v2["sort_id"]
    assert table[k] is v


def test_nan_state_is_deterministic_and_matches_oracle():
    """A 'toxic' key (NaN state) must still produce the same draws as the game."""
    p = PseudoRandom("ABCDEFGH")
    p.set_key_state("toxic", float("nan"))
    ps = p.pseudoseed("toxic")
    assert ps != ps
    r = p.pseudorandom("toxic")
    r2 = p.pseudorandom("toxic", 1, 10)
    assert 0.0 <= r < 1.0 and 1 <= r2 <= 10
    o = get_oracle()
    if o is None:
        pytest.skip("live oracle needed for the NaN cross-check")
    o.START_RUN("ABCDEFGH")
    o.SET_STATE("toxic", bits(float("nan")))
    p2 = PseudoRandom("ABCDEFGH")
    p2.set_key_state("toxic", float("nan"))
    for _ in range(3):
        st, ps_b, r1, r2_, r3, r4, r5, r6 = o.DRAW("toxic")
        ps2 = p2.pseudoseed("toxic")
        p2.rng.seed(ps2)
        got = [bits(p2.get_key_state("toxic")), bits(ps2), bits(p2.rng.random()), p2.rng.random_int(1, 6),
               p2.rng.random_index(3), p2.rng.random_int(-5, 5), p2.rng.random_int(1, 1000000), bits(p2.rng.random())]
        assert got == [st, ps_b, r1, int(r2_), int(r3), int(r4), int(r5), r6]


def test_snapshot_restore_clone():
    p = PseudoRandom("7LB2WVPK")
    for k in ("Joker1", "Tarot1", "front"):
        p.pseudorandom(k)
    snap = p.snapshot()
    c = p.clone()
    seq_a = [p.pseudorandom("Joker1"), p.pseudorandom("Tarot1", 1, 22), p.pseudoseed("seed")]
    p.restore(snap)
    seq_b = [p.pseudorandom("Joker1"), p.pseudorandom("Tarot1", 1, 22), p.pseudoseed("seed")]
    seq_c = [c.pseudorandom("Joker1"), c.pseudorandom("Tarot1", 1, 22), c.pseudoseed("seed")]
    assert seq_a == seq_b == seq_c
    # snapshot is a copy, not a view
    p.pseudorandom("Joker1")
    assert snap["state"]["Joker1"] != p.get_key_state("Joker1")


def test_pseudorandom_accepts_precomputed_seed_float():
    p = PseudoRandom("AAAAAAAA")
    q = PseudoRandom("AAAAAAAA")
    a = p.pseudorandom("Joker1", 1, 100)
    b = q.pseudorandom(q.pseudoseed("Joker1"), 1, 100)
    assert a == b


def test_normalize_seed():
    assert normalize_seed("abcd1234") == "ABCD1234"
    assert normalize_seed("0o0o") == "OOOO"
    assert normalize_seed("a-b c!d_e") == "ABCDE"
    assert normalize_seed("ABCDEFGHIJ") == "ABCDEFGH"
    assert normalize_seed("") == ""


def test_cache_matches_live_if_both_available():
    """Drift detector: live LuaJIT/game output must equal the committed fixture."""
    if get_oracle() is None or not FIXTURE_PATH.is_file():
        pytest.skip("needs both live oracle and cached fixture")
    if "gt" not in _GT:
        _GT["gt"], _GT["src"] = load_ground_truth()
    assert _GT["src"] == "live"
    live = _GT["gt"]
    cached = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    for section in ("unseeded_first", "luajit_random", "pseudohash", "lcg", "fmt13", "runs"):
        assert live[section] == cached[section], "fixture drift in section %r" % section


def test_throughput_smoke():
    """Not a benchmark assertion -- just makes sure the hot path is not pathological."""
    p = PseudoRandom("7LB2WVPK")
    n = 20000
    t = time.perf_counter()
    for i in range(n):
        p.pseudorandom("Joker1")
    dt = time.perf_counter() - t
    assert n / dt > 20000, "pseudorandom() too slow: %.0f/s" % (n / dt)


# --------------------------------------------------------------------------------------
# fixture regeneration
# --------------------------------------------------------------------------------------


def regenerate_fixture(path: Path = FIXTURE_PATH) -> dict:
    o = get_oracle()
    if o is None:
        raise SystemExit("oracle unavailable: %s" % _oracle_cache.get("err"))
    data = generate_ground_truth(o)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=None, separators=(",", ":")), encoding="utf-8")
    return data


if __name__ == "__main__":
    if "--regen" in sys.argv:
        d = regenerate_fixture()
        print("wrote %s (%d runs, %d keys, luajit=%s)" % (FIXTURE_PATH, len(d["runs"]), len(KEYS), d["meta"]["luajit"]))
    else:
        print(__doc__)
