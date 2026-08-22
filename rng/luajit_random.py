"""Bit-exact pure-Python port of LuaJIT 2.1's ``math.randomseed`` / ``math.random``.

Balatro seeds LuaJIT's global PRNG before every game-logic draw
(``math.randomseed(pseudoseed(key))`` then ``math.random(...)``), so the
generator that sits behind ``pseudorandom`` is *this* one, not the LCG in
``pseudoseed``.  Reproducing the game therefore requires a port of LuaJIT's
generator, which is:

* a Tausworthe / combined-LFSR generator with four 64-bit components
  (L'Ecuyer 1991, table 3, 1st entry; period 2^223), stepped in lock-step and
  XORed together;
* a ``math.randomseed(d)`` that maps the double ``d`` through
  ``d <- d*pi + e`` four times, takes the raw IEEE bit pattern of each
  intermediate as the component state (bumping it if it is below a small
  per-component minimum), then discards 10 outputs;
* ``math.random()`` that builds a double in [1,2) from 52 output bits and
  subtracts 1.0;
* ``math.random(n)``  = ``floor(r*n) + 1``   and
  ``math.random(m,n)`` = ``floor(r*(n-m+1)) + m``   (no range validation).

Algorithm reference: LuaJIT 2.1 ``src/lib_math.c`` (``random_seed``,
``math_random``) and ``src/lj_prng.c`` (``TW223_GEN``/``TW223_STEP``,
``lj_prng_u64d``).  LuaJIT is MIT licensed (Copyright (C) 2005-2026 Mike Pall);
this file is an independent re-implementation of the algorithm, not a copy.

The same generator has been in LuaJIT since 2.0, so it is version-independent
as far as LOVE/Balatro builds are concerned.  Validation against LuaJIT itself
(via ``lupa``) lives in ``mp/tests/test_rng_core.py``.

Performance notes
-----------------
Everything is inlined into ``seed``/``random``/``random_int`` because this is
the innermost loop of the whole engine: every ``pseudorandom`` call costs one
``seed`` (4 bit-casts + 10 discarded LFSR steps) plus one output step.  The
LFSR update is linear over GF(2), so the 10 discard steps are folded into a
single table-driven linear map per component (``_DISCARD_TABLES``: eight
256-entry tables per component, one per input byte).  ``_step`` remains the
straightforward transcription and is also used to build/verify the tables.
"""

from __future__ import annotations

import math
import struct

__all__ = ["LuaJITRandom", "FIXED_SEED_STATE"]

_M64 = 0xFFFFFFFFFFFFFFFF
_M52 = 0x000FFFFFFFFFFFFF
_TWO_POW_MINUS_52 = 2.0 ** -52  # exact

_pack_d = struct.Struct("<d").pack
_unpack_q = struct.Struct("<Q").unpack

# d <- d * 3.14159265358979323846 + 2.7182818284590452354  (C double literals
# round to exactly math.pi and math.e).
_SEED_MUL = math.pi
_SEED_ADD = math.e

# Per-component (k, q, s) from TW223_STEP, and the derived constants used by
# the inlined step:  shift_r = k - s,  mask_k = ~0 << (64 - k).
# gen0: k=63 q=31 s=18    gen1: k=58 q=19 s=28
# gen2: k=55 q=24 s= 7    gen3: k=47 q=21 s= 8
_MASK0 = (_M64 << 1) & _M64   # 0xFFFFFFFFFFFFFFFE
_MASK1 = (_M64 << 6) & _M64   # 0xFFFFFFFFFFFFFFC0
_MASK2 = (_M64 << 9) & _M64   # 0xFFFFFFFFFFFFFE00
_MASK3 = (_M64 << 17) & _M64  # 0xFFFFFFFFFFFE0000

# Minimum component values enforced by random_seed: 1 << (64 - k).
_SEED_MIN = (1 << 1, 1 << 6, 1 << 9, 1 << 17)

# lj_prng_seed_fixed(): "the precomputed result of random_seed(rs, 0.0)".
# LuaJIT's initial (unseeded) math.random state.
FIXED_SEED_STATE = (
    0xA0D277570A345B8C,
    0x764A296C5D4AA64F,
    0x51220704070ADEAA,
    0x2A2717B5A7B7B927,
)


def _gen_steps(z: int, idx: int, n: int) -> int:
    """``n`` TW223_GEN updates of a single 64-bit component (reference form)."""
    q, shift_r, mask_k, s = _GEN_PARAMS[idx]
    for _ in range(n):
        z = ((((z << q) & _M64) ^ z) >> shift_r) ^ (((z & mask_k) << s) & _M64)
    return z


_GEN_PARAMS = (
    (31, 63 - 18, _MASK0, 18),
    (19, 58 - 28, _MASK1, 28),
    (24, 55 - 7, _MASK2, 7),
    (21, 47 - 8, _MASK3, 8),
)


def _build_discard_tables(n_steps: int = 10):
    """Tables T[idx][byte_pos][byte_value] such that

        gen_steps(z, idx, n_steps) == XOR over byte_pos of T[idx][byte_pos][(z >> 8*byte_pos) & 0xFF]

    Valid because each component update is GF(2)-linear in its 64 state bits.
    """
    tables = []
    for idx in range(4):
        per_byte = []
        for bpos in range(8):
            tbl = [0] * 256
            # Image of each basis bit within this byte.
            basis = [_gen_steps(1 << (8 * bpos + b), idx, n_steps) for b in range(8)]
            for v in range(1, 256):
                lowbit = v & -v
                b = lowbit.bit_length() - 1
                tbl[v] = tbl[v ^ lowbit] ^ basis[b]
            per_byte.append(tuple(tbl))
        tables.append(tuple(per_byte))
    return tuple(tables)


_DISCARD_TABLES = _build_discard_tables(10)


class LuaJITRandom:
    """LuaJIT 2.1 ``math.random`` state machine.

    ``LuaJITRandom()`` starts in LuaJIT's fixed unseeded state (equivalent to
    ``seed(0.0)``); ``LuaJITRandom(d)`` is ``math.randomseed(d)``.
    """

    __slots__ = ("_s0", "_s1", "_s2", "_s3")

    def __init__(self, seed: float | None = None):
        if seed is None:
            self._s0, self._s1, self._s2, self._s3 = FIXED_SEED_STATE
        else:
            self.seed(seed)

    # -- seeding ------------------------------------------------------------

    def seed(self, d: float) -> None:
        """``math.randomseed(d)`` -- bit-exact, including NaN/inf/denormal ``d``."""
        d = float(d)
        T = _DISCARD_TABLES

        d = d * _SEED_MUL + _SEED_ADD
        u = _unpack_q(_pack_d(d))[0]
        if u < 2:
            u += 2
        t = T[0]
        self._s0 = (t[0][u & 0xFF] ^ t[1][(u >> 8) & 0xFF] ^ t[2][(u >> 16) & 0xFF]
                    ^ t[3][(u >> 24) & 0xFF] ^ t[4][(u >> 32) & 0xFF] ^ t[5][(u >> 40) & 0xFF]
                    ^ t[6][(u >> 48) & 0xFF] ^ t[7][u >> 56])

        d = d * _SEED_MUL + _SEED_ADD
        u = _unpack_q(_pack_d(d))[0]
        if u < 64:
            u += 64
        t = T[1]
        self._s1 = (t[0][u & 0xFF] ^ t[1][(u >> 8) & 0xFF] ^ t[2][(u >> 16) & 0xFF]
                    ^ t[3][(u >> 24) & 0xFF] ^ t[4][(u >> 32) & 0xFF] ^ t[5][(u >> 40) & 0xFF]
                    ^ t[6][(u >> 48) & 0xFF] ^ t[7][u >> 56])

        d = d * _SEED_MUL + _SEED_ADD
        u = _unpack_q(_pack_d(d))[0]
        if u < 512:
            u += 512
        t = T[2]
        self._s2 = (t[0][u & 0xFF] ^ t[1][(u >> 8) & 0xFF] ^ t[2][(u >> 16) & 0xFF]
                    ^ t[3][(u >> 24) & 0xFF] ^ t[4][(u >> 32) & 0xFF] ^ t[5][(u >> 40) & 0xFF]
                    ^ t[6][(u >> 48) & 0xFF] ^ t[7][u >> 56])

        d = d * _SEED_MUL + _SEED_ADD
        u = _unpack_q(_pack_d(d))[0]
        if u < 131072:
            u += 131072
        t = T[3]
        self._s3 = (t[0][u & 0xFF] ^ t[1][(u >> 8) & 0xFF] ^ t[2][(u >> 16) & 0xFF]
                    ^ t[3][(u >> 24) & 0xFF] ^ t[4][(u >> 32) & 0xFF] ^ t[5][(u >> 40) & 0xFF]
                    ^ t[6][(u >> 48) & 0xFF] ^ t[7][u >> 56])

    def seed_reference(self, d: float) -> None:
        """``math.randomseed(d)`` as a literal transcription (10 explicit discard
        steps).  Slower; exists to cross-check the table-driven ``seed``."""
        d = float(d)
        state = []
        for m in _SEED_MIN:
            d = d * _SEED_MUL + _SEED_ADD
            u = _unpack_q(_pack_d(d))[0]
            if u < m:
                u += m
            state.append(u)
        self._s0, self._s1, self._s2, self._s3 = state
        for _ in range(10):
            self._step()

    # -- core step ------------------------------------------------------------

    def _step(self) -> int:
        """One TW223_STEP: advance all four components, return their XOR (u64)."""
        z = self._s0
        z = ((((z << 31) & _M64) ^ z) >> 45) ^ (((z & _MASK0) << 18) & _M64)
        self._s0 = z
        r = z
        z = self._s1
        z = ((((z << 19) & _M64) ^ z) >> 30) ^ (((z & _MASK1) << 28) & _M64)
        self._s1 = z
        r ^= z
        z = self._s2
        z = ((((z << 24) & _M64) ^ z) >> 48) ^ (((z & _MASK2) << 7) & _M64)
        self._s2 = z
        r ^= z
        z = self._s3
        z = ((((z << 21) & _M64) ^ z) >> 39) ^ (((z & _MASK3) << 8) & _M64)
        self._s3 = z
        r ^= z
        return r

    step_u64 = _step  # lj_prng_u64

    # -- outputs --------------------------------------------------------------

    def random(self) -> float:
        """``math.random()`` -> double in [0, 1)."""
        # lj_prng_u64d builds the double 1.m (m = low 52 bits of r) and
        # math_random subtracts 1.0; that difference is exactly m * 2^-52.
        z = self._s0
        z = ((((z << 31) & _M64) ^ z) >> 45) ^ (((z & _MASK0) << 18) & _M64)
        self._s0 = z
        r = z
        z = self._s1
        z = ((((z << 19) & _M64) ^ z) >> 30) ^ (((z & _MASK1) << 28) & _M64)
        self._s1 = z
        r ^= z
        z = self._s2
        z = ((((z << 24) & _M64) ^ z) >> 48) ^ (((z & _MASK2) << 7) & _M64)
        self._s2 = z
        r ^= z
        z = self._s3
        z = ((((z << 21) & _M64) ^ z) >> 39) ^ (((z & _MASK3) << 8) & _M64)
        self._s3 = z
        r ^= z
        return (r & _M52) * _TWO_POW_MINUS_52

    def random_index(self, n) -> int:
        """``math.random(n)`` -> integer in [1, n]  (``floor(r*n) + 1.0``)."""
        d = self.random() * float(n)
        return int(math.floor(d) + 1.0)

    def random_int(self, m, n) -> int:
        """``math.random(m, n)`` -> integer in [m, n]  (``floor(r*(n-m+1)) + m``).

        Like LuaJIT, does not validate ``m <= n``.
        """
        r1 = float(m)
        d = self.random() * (float(n) - r1 + 1.0)
        return int(math.floor(d) + r1)

    def lua_random(self, *args):
        """Dispatch exactly like Lua's ``math.random`` with 0, 1 or 2 args."""
        if not args:
            return self.random()
        if len(args) == 1:
            return self.random_index(args[0])
        if len(args) == 2:
            return self.random_int(args[0], args[1])
        raise TypeError("math.random takes at most 2 arguments")

    # -- state ----------------------------------------------------------------

    def get_state(self) -> tuple[int, int, int, int]:
        return (self._s0, self._s1, self._s2, self._s3)

    def set_state(self, state) -> None:
        self._s0, self._s1, self._s2, self._s3 = (int(x) & _M64 for x in state)

    def copy(self) -> "LuaJITRandom":
        c = LuaJITRandom.__new__(LuaJITRandom)
        c._s0, c._s1, c._s2, c._s3 = self._s0, self._s1, self._s2, self._s3
        return c

    def __repr__(self) -> str:
        return "LuaJITRandom(state=(%#018x, %#018x, %#018x, %#018x))" % self.get_state()
