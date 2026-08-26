"""Balatro's keyed pseudorandom core, ported from ``functions/misc_functions.lua``
(1.0.1o) and validated bit-for-bit against the game's own Lua executing in
LuaJIT (see ``tests/test_rng_core.py``).

The game's chain for one keyed draw is::

    pseudoseed(key):
        state[key] ??= pseudohash(key .. seed)                      # first use only
        state[key]   = |tonumber(format("%.13f", (2.134453429141 + state[key]*1.72431234) % 1))|
        return (state[key] + hashed_seed) / 2                       # hashed_seed = pseudohash(seed)

    pseudorandom(key[, m, n]):
        math.randomseed(pseudoseed(key))                            # LuaJIT Tausworthe, see luajit_random.py
        return math.random() | math.random(m, n)

    pseudorandom_element(t, pseudoseed(key)):   t[sorted_keys[math.random(#t)]]
    pseudoshuffle(list, pseudoseed(key)):       for i = #list..2: swap(list[i], list[math.random(i)])

Everything here is pure Python with no dependencies.  All float operations are
performed in exactly the order LuaJIT performs them so results are bit-identical,
including the NaN semantics of "toxic" hashes (see ``pseudohash``).
"""

from __future__ import annotations

import struct
from typing import Any, Mapping, MutableSequence

from .luajit_random import LuaJITRandom

__all__ = [
    "PI",
    "PseudoRandom",
    "SEED_CORPUS",
    "lcg_step",
    "normalize_seed",
    "pseudohash",
    "pseudoseed_predict",
]

PI = 3.141592653589793  # LuaJIT math.pi (same double as Python's math.pi)

_HASH_K = 1.1239285023
_LCG_A = 2.134453429141
_LCG_B = 1.72431234

# LuaJIT's NaN as produced by `inf % 1` / `inf * 0` on x86-64 (sign bit set,
# 0xFFF8000000000000).  Python's `inf % 1.0` yields the positive quiet NaN, so
# pseudohash canonicalises to this pattern to stay bit-identical with the game.
_LUA_NAN = struct.unpack("<d", bytes.fromhex("000000000000f8ff"))[0]

# Characters the in-game seed text box accepts (button_callbacks.lua:989).
# Note: no '0' -- the key handler remaps '0' to 'o' (line 970), which all_caps
# then uppercases to 'O'.  Generated seeds (random_string) additionally never
# contain 'O'.
SEED_CORPUS = "123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SEED_MAX_LEN = 8


def pseudohash(s: str | bytes) -> float:
    """``pseudohash(str)`` from misc_functions.lua::

        num = 1
        for i = #str, 1, -1 do
            num = ((1.1239285023/num) * string.byte(str, i) * math.pi + math.pi*i) % 1
        end

    Iterates from the LAST byte to the first.  Returns a double in [0, 1) --
    except:

    * empty string -> exactly 1.0 (loop body never runs);
    * if ``num`` ever becomes exactly 0.0 at a step that is not the last one
      (``%1`` of an integer-valued double, which happens when the intermediate
      exceeds 2^52, i.e. when the previous ``num`` was < ~1e-13), the next step
      divides by zero and the result is NaN forever after ("toxic" input).
      Lua's NaN here has the sign bit set; we reproduce that bit pattern.
      Probability per byte is ~2e-12, so real keys essentially never hit it,
      but the downstream code (pseudoseed/LuaJITRandom) handles NaN exactly
      like the game does anyway.
    """
    if isinstance(s, str):
        data = s.encode("latin-1")  # string.byte semantics: one byte per char
    else:
        data = bytes(s)
    num = 1.0
    i = len(data)
    for b in reversed(data):
        if num == 0.0:
            # Lua: 1.1239285023/0 = inf -> inf*b*pi + pi*i = inf (or inf*0 = nan)
            # -> inf % 1 = inf - floor(inf) = nan; NaN then propagates through
            # every remaining step.  LuaJIT's NaN here carries the sign bit.
            return _LUA_NAN
        num = ((_HASH_K / num) * b * PI + PI * i) % 1.0
        i -= 1
    if num != num:  # overflow path (tiny denormal num -> inf % 1): match LuaJIT's NaN
        return _LUA_NAN
    return num


def lcg_step(x: float) -> float:
    """One per-key state update from ``pseudoseed``::

        math.abs(tonumber(string.format("%.13f", (2.134453429141 + x*1.72431234) % 1)))

    Python's ``'%.13f'`` and LuaJIT's ``string.format`` are both exact
    (round-half-even on the true binary value) and ``float()``/``tonumber`` are
    both correctly rounded, so this is bit-identical (verified in tests).
    """
    return abs(float("%.13f" % ((_LCG_A + x * _LCG_B) % 1.0)))


def pseudoseed_predict(key: str, predict_seed: str) -> float:
    """``pseudoseed(key, predict_seed)`` -- the stateless prediction branch used
    by ``get_first_legendary`` when generating a starting seed.  Does not touch
    (or need) per-key state."""
    x = lcg_step(pseudohash(key + predict_seed))
    return (x + pseudohash(predict_seed)) / 2


def normalize_seed(text: str) -> str:
    """Apply the game's seed text-box rules to arbitrary text: uppercase,
    '0' -> 'O', drop characters outside the corpus, keep the first 8.

    This mirrors ``G.FUNCS.text_input_key`` (all_caps=true, max_length=8) as
    driven by typing or the Paste button.  ``PseudoRandom`` itself does NOT
    normalise -- it takes the seed verbatim, exactly like
    ``G.GAME.pseudorandom.seed`` -- so apply this to user input first.
    """
    out = []
    for ch in text:
        if ch == "0":
            ch = "o"
        ch = ch.upper()
        if ch in SEED_CORPUS:
            out.append(ch)
            if len(out) == SEED_MAX_LEN:
                break
    return "".join(out)


class PseudoRandom:
    """Per-run keyed RNG: ``G.GAME.pseudorandom`` + LuaJIT's global ``math.random``.

    ``seed`` is used verbatim (like ``G.GAME.pseudorandom.seed``); run
    ``normalize_seed`` on user input first.  On construction this performs what
    ``Game:start_run`` does at game.lua:2164-2168: ``hashed_seed = pseudohash(seed)``
    with an empty per-key state table.

    Thread-safety: none (one instance per simulated run; ``snapshot``/``restore``
    or ``clone`` for tree search).
    """

    __slots__ = ("seed", "hashed_seed", "_state", "_rng")

    def __init__(self, seed: str, *, state: Mapping[str, float] | None = None,
                 rng_state=None):
        self.seed = seed
        self.hashed_seed = pseudohash(seed)
        self._state: dict[str, float] = dict(state) if state else {}
        self._rng = LuaJITRandom()
        if rng_state is not None:
            self._rng.set_state(rng_state)

    # -- the chain ----------------------------------------------------------------

    def pseudoseed(self, key: str) -> float:
        """``pseudoseed(key)``: advance ``state[key]`` and return the seed value
        to feed ``math.randomseed``.  ``key == 'seed'`` is special-cased by the
        game to return a raw ``math.random()`` from the current global state."""
        if key == "seed":
            return self._rng.random()
        st = self._state
        x = st.get(key)
        if x is None:
            x = pseudohash(key + self.seed)
        x = abs(float("%.13f" % ((_LCG_A + x * _LCG_B) % 1.0)))
        st[key] = x
        return (x + self.hashed_seed) / 2

    def pseudorandom(self, key, m=None, n=None):
        """``pseudorandom(seed, min, max)``.

        ``key`` may be a key string (the normal case, goes through
        ``pseudoseed``) or an already-computed seed float (the game passes
        ``pseudoseed(...)`` results around too).  With both ``m`` and ``n``
        returns an int in [m, n]; otherwise a float in [0, 1).  As in Lua, a
        lone ``m`` without ``n`` is ignored.
        """
        rng = self._rng
        rng.seed(self.pseudoseed(key) if isinstance(key, str) else key)
        if m is not None and n is not None:
            return rng.random_int(m, n)
        return rng.random()

    def pseudorandom_element(self, seq, key):
        """``pseudorandom_element(_t, pseudoseed(key))``.

        For a sequence (list/tuple): the caller must supply it in Lua's
        post-sort order (arrays keep their order; runtime card lists are sorted
        by ``sort_id``; string-keyed prototype tables by key -- see the
        ``Mapping`` branch).  Returns ``(element, index0)``.

        For a Mapping (e.g. ``G.P_CARDS``-style dicts): keys are sorted the way
        the Lua does -- by ``sort_id`` of the values if the values carry one
        (attribute or item), else by key -- and ``(value, key)`` is returned.

        ``key`` may be a key string or a precomputed seed float.
        """
        rng = self._rng
        rng.seed(self.pseudoseed(key) if isinstance(key, str) else key)
        if isinstance(seq, Mapping):
            keys = _lua_sorted_keys(seq)
            k = keys[rng.random_index(len(keys)) - 1]
            return seq[k], k
        i = rng.random_index(len(seq)) - 1
        return seq[i], i

    def pseudoshuffle(self, lst: MutableSequence, key) -> None:
        """``pseudoshuffle(list, pseudoseed(key))`` -- in place.

        The Lua first sorts ``list`` by ``sort_id`` when ``list[1].sort_id``
        exists (Card objects); pass the list already in that order.  The
        shuffle itself is ``for i = #list, 2, -1: swap(list[i], list[random(i)])``.
        ``key`` may be a key string or a precomputed seed float.
        """
        rng = self._rng
        rng.seed(self.pseudoseed(key) if isinstance(key, str) else key)
        random = rng.random
        floor_int = int
        for i in range(len(lst), 1, -1):
            j = floor_int(random() * i)  # math.random(i) - 1  (product >= 0 so trunc == floor)
            lst[i - 1], lst[j] = lst[j], lst[i - 1]

    # -- direct access to the global LuaJIT generator ---------------------------------

    @property
    def rng(self) -> LuaJITRandom:
        """The LuaJIT ``math.random`` state (shared/global in the game)."""
        return self._rng

    def randomseed(self, d: float) -> None:
        self._rng.seed(d)

    def random(self) -> float:
        return self._rng.random()

    def random_int(self, m, n) -> int:
        return self._rng.random_int(m, n)

    def random_index(self, n) -> int:
        return self._rng.random_index(n)

    # -- state management ------------------------------------------------------------

    def get_key_state(self, key: str) -> float | None:
        """Current ``G.GAME.pseudorandom[key]`` (None if the key is unused)."""
        return self._state.get(key)

    def set_key_state(self, key: str, value: float) -> None:
        self._state[key] = value

    def drop_key(self, key: str) -> None:
        """``G.GAME.pseudorandom[key] = nil`` -- forget a key's stream so its next use
        re-hashes from scratch (the Multiplayer mod's Order shuffle does this)."""
        self._state.pop(key, None)

    def keys(self):
        return self._state.keys()

    def snapshot(self) -> dict[str, Any]:
        """Copy of all state needed to reproduce future draws exactly."""
        return {
            "seed": self.seed,
            "state": dict(self._state),
            "rng": self._rng.get_state(),
        }

    def restore(self, snap: Mapping[str, Any]) -> None:
        if snap["seed"] != self.seed:
            self.seed = snap["seed"]
            self.hashed_seed = pseudohash(self.seed)
        self._state = dict(snap["state"])
        self._rng.set_state(snap["rng"])

    def clone(self) -> "PseudoRandom":
        c = PseudoRandom.__new__(PseudoRandom)
        c.seed = self.seed
        c.hashed_seed = self.hashed_seed
        c._state = dict(self._state)
        c._rng = self._rng.copy()
        return c

    def __repr__(self) -> str:
        return "PseudoRandom(seed=%r, keys=%d)" % (self.seed, len(self._state))


def _sort_id_of(v):
    if isinstance(v, Mapping):
        return v.get("sort_id")
    return getattr(v, "sort_id", None)


def _lua_sorted_keys(table: Mapping) -> list:
    """Key order used by ``pseudorandom_element`` for a Lua table: by the
    values' ``sort_id`` when the first value has one, else by key
    (LuaJIT compares strings bytewise, which matches Python for ASCII keys)."""
    keys = list(table.keys())
    if keys and _sort_id_of(table[keys[0]]) is not None:
        keys.sort(key=lambda k: _sort_id_of(table[k]))
    else:
        keys.sort()
    return keys
