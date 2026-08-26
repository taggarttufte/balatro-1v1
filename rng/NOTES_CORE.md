# RNG core — port notes (Agent A, Phase 0)

**Status 2026-08-21: bit-exact.** `rng/core.py` + `rng/luajit_random.py` reproduce
Balatro 1.0.1o's `pseudohash` / `pseudoseed` / `pseudorandom` / `pseudorandom_element` /
`pseudoshuffle` **and** LuaJIT 2.1's `math.randomseed` / `math.random`, validated against the
game's own Lua executing inside LuaJIT (`lupa`). Every compared value is a 64-bit IEEE
pattern — no tolerances anywhere.

Test run: `python -m pytest tests/test_rng_core.py` → **17 passed** (live oracle);
`MP_RNG_NO_ORACLE=1` (cached fixture only) → 14 passed, 3 skipped (the live-only tests).

---

## 1. What is ported, and from where

| Python | Game source | Notes |
|---|---|---|
| `pseudohash(s)` | `misc_functions.lua:279` | iterates last byte → first; `((1.1239285023/num)*byte*pi + pi*i) % 1` |
| `lcg_step(x)` / inside `PseudoRandom.pseudoseed` | `misc_functions.lua:298` | `abs(tonumber(format("%.13f", (2.134453429141 + x*1.72431234) % 1)))` |
| `PseudoRandom.pseudoseed(key)` | `:298` | `state[key] ??= pseudohash(key..seed)`; LCG step; `(state + hashed_seed)/2` |
| `pseudoseed_predict(key, seed)` | `:298` (`predict_seed` branch) | stateless; used by `get_first_legendary` during seed generation |
| `PseudoRandom.pseudorandom(key[, m, n])` | `:315` | `math.randomseed(pseudoseed(key))` then `math.random()` or `math.random(m,n)` |
| `PseudoRandom.pseudorandom_element(seq, key)` | `:253` | `keys[math.random(#keys)]` after Lua's sort (see §4) |
| `PseudoRandom.pseudoshuffle(list, key)` | `:206` | `for i=#list,2,-1: swap(list[i], list[math.random(i)])` |
| `PseudoRandom(seed)` constructor | `game.lua:2164-2168` | `hashed_seed = pseudohash(seed)`; empty per-key table |
| `LuaJITRandom.seed(d)` | LuaJIT `lib_math.c:random_seed` | `d <- d*pi + e` ×4, raw bit pattern → 4×u64 state, bump if `< 1<<(64-k)`, 10 discards |
| `LuaJITRandom.random()` | `lib_math.c:math_random` + `lj_prng.c:lj_prng_u64d` | TW223 combined LFSR (k=63,58,55,47), `(r & 2^52-1) | 0x3ff0…` − 1.0 |
| `LuaJITRandom.random_index(n)` | same | `floor(r*n) + 1` |
| `LuaJITRandom.random_int(m, n)` | same | `floor(r*(n-m+1)) + m` — **no** range check (LuaJIT has none) |
| `LuaJITRandom()` default state | `lj_prng.h:lj_prng_seed_fixed` | = `seed(0.0)`; our `seed(0.0)` reproduces LuaJIT's four hard-coded constants exactly |

Nothing from the Lua or LuaJIT sources is vendored; the test harness extracts the five
Balatro functions from `_reference/` at test time (gitignored) and falls back to the
cached fixture when that directory or `lupa` is absent.

### Self-check that the LuaJIT port is the real algorithm
LuaJIT ships `lj_prng_seed_fixed()` = "the precomputed result of `random_seed(rs, 0.0)`"
(four u64 constants). `LuaJITRandom().seed(0.0)` produces those same four words, which
exercises the whole seeding path (π/e recurrence, bit-cast, min-bump, 10 discards) against
something hard-coded in LuaJIT itself, independent of lupa.

---

## 2. What the oracle compared (all bit-exact, all passing)

Corpus: 52 seeds (36 random game-format + `TUTORIAL`, `7LB2WVPK`, `AAAAAAAA`, short/empty/
lowercase/`O`-containing ones) × 55 keys (real game keys incl. `Tarot5_resample2`,
`Voucher1_resample1`, `Planet1_resample10`, `Joker10`, `""`, 64-char, 110-char, leading/
trailing space, case variants) × 3 successive draws, recording per draw: post-LCG state,
`pseudoseed` return, then from one `math.randomseed`: `random()`, `random(1,6)`,
`random(3)`, `random(-5,5)`, `random(1,1e6)`, `random()`.

| Section | Values | Result |
|---|---|---|
| keyed chain (above) | 68,640 + 2,860 hashes | all equal |
| `pseudorandom()` via the game's own entry point, interleaved keys | 52 × 18 | equal |
| `pseudorandom_element`: array / string-keyed table / `sort_id` tables | 52 × 7 × 3 | equal |
| `pseudoshuffle`: sizes 0,1,2,3,4,5,8,13,52,100 + `sort_id` list | 52 × 12 lists | equal |
| `pseudoseed(key, predict_seed)` | 52 × 3 | equal |
| `pseudoseed('seed')` (raw `math.random()` from current global state) | 52 | equal |
| raw `math.randomseed(d)` + 12-op sequence, 267 seeds incl. `±0, ±nan, ±inf, 5e-324, 1e300, 2^53, 2^64, -e/π` | 3,204 | equal |
| `pseudohash` on keys/seeds/concats, NUL bytes, bytes ≥ 0x80, all 255 byte values | ~1,000 | equal |
| LCG step (`%.13f` + `tonumber` + `abs`), incl. exact ties `k/16384`, `k/8192`, `0.99999999999995` | 3,060 | equal |
| `string.format("%.13f", x)` string + `tonumber` back | 5,000 cached + 80,000 live | equal |
| NaN ("toxic") per-key state, 3 draws | live only | equal |

Cached fixture: `tests/fixtures/rng_ground_truth.json` (2.3 MB, LuaJIT 2.1.1774896198 x64
Windows). `test_cache_matches_live_if_both_available` diffs live vs cache every run, so a
lupa/LuaJIT upgrade or game-source change cannot drift silently.
Regenerate: `python tests/test_rng_core.py --regen`.

---

## 3. Discrepancies found (none in the port; two in the *oracle* that matter for everyone)

1. **LuaJIT JIT traces break FFI type-punning.** My first harness read a double's bits via
   `ffi.cast('uint64_t*', double_buf)[0]`. Once hot, LuaJIT's trace optimizer assumed the
   `double` store and `uint64_t` load don't alias and forwarded a *stale* load → two
   consecutive `math.random()` values appeared identical and NaN leaked between fields. It
   looked exactly like a re-seeding bug; it was not (integer fields in the same call were
   right). **Any agent using lupa as an oracle: run `jit.off()` in the runtime** (the
   interpreter is the reference semantics anyway) and never pun through FFI pointers. The
   harness now does byte copies (`ffi.string`/`ffi.copy`) and asserts a 3,000-draw self-check
   at construction.
2. **lupa converts integral doubles to Python `int`** (`-0.0` → `0`). Pass doubles between
   Python and Lua as hex bit patterns (`BITS`/`UNBITS` in the test file), never as numbers.

Two platform caveats that the oracle *cannot* see (both are about the real game binary, not
lupa) — Agent D's end-to-end seed checks are what validate them:

* `random_seed` computes `d*pi + e`. If a LuaJIT build contracted that into an FMA the
  sequences would differ completely. MSVC x64 (LÖVE's Windows build, lupa's wheel) does not
  contract; LuaJIT's own ARM64 backend *does* fuse mul+add in JIT code, so a Balatro build
  on Apple Silicon could in principle diverge from Windows. Out of scope; noting it.
* 32-bit x87 builds would evaluate in 80-bit; Balatro ships 64-bit.

---

## 4. Semantics Phase 1 agents must get right when threading keys

* **Seed string is used verbatim.** `G.GAME.pseudorandom.seed` is whatever the text box
  produced. Normalisation rule (`button_callbacks.lua:970-1040`, `:1863`): `max_length = 8`,
  `all_caps = true`, corpus `123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ` — **there is no `0`**; a
  typed `0` is remapped to `o` → `O`; anything outside the corpus is silently dropped (also
  when pasting). Generated seeds (`random_string`) use `1-9`, `A-N`, `P-Z` (never `O`).
  `normalize_seed(text)` implements the text-box rule; `PseudoRandom(seed)` does **not**
  apply it (so oracle tests can use odd seeds) — call it on user input first. Seeds shorter
  than 8 are legal (`"A"`, even `""`).
* **Every draw re-seeds LuaJIT's global generator.** Cross-key order therefore does not
  matter for `pseudorandom`/`pseudorandom_element`/`pseudoshuffle`; only the *per-key* call
  count matters. The one exception is `pseudoseed('seed')`, which returns a raw
  `math.random()` from the current global state — unreproducible in the real game (UI/sound
  calls interleave) and unused by game logic as far as grep shows.
* **Keys are `key .. seed` at first use**, so `"Joker1"` vs `"Joker1 "` vs `"joker1"` are
  different streams. The state dict is keyed by the exact string; build keys exactly as the
  Lua concatenates them (`'Tarot'..ante`, `_pool_key..'_resample'..it`, `'front'..key_append..ante`).
* `pseudorandom_element` indexes a **sorted** key list (`misc_functions.lua:253-271`):
  values carrying `sort_id` (Card objects) → sorted by `sort_id`; otherwise by key
  (arrays: 1..n numeric = array order; string-keyed tables like `G.P_CARDS` or
  `eligible_bosses` → **bytewise string order of the keys**). The Python takes a sequence
  already in that order (returns `(elem, index0)`), or a `Mapping` which it sorts the Lua
  way (returns `(value, key)`). Prototype tables (`P_CENTERS`, `P_CARDS`) have **no**
  `sort_id`; only runtime `Card`s do (`card.lua:24-25`, a global counter).
* `pseudoshuffle` sorts the list by `sort_id` first when `list[1].sort_id` exists, *then*
  shuffles; pass the list in `sort_id` order. The game only shuffles Card lists.
* `math.random(m, n)` has **no validation** in LuaJIT; `math.random(0)` returns 1,
  `random(5, 2)` returns junk rather than erroring. The port mirrors that.
* Stateless siblings: `pseudoseed(key, predict_seed)` (seed generation) is
  `pseudoseed_predict`; `random_string` (seed generation UI) is not ported — it is driven by
  cursor position/time and is not reproducible anyway.
* **Unseeded `math.random` in game logic** (found by grep; all cosmetic or intentionally
  non-reproducible, do not model them with keyed RNG):
  `tag.lua:211/226` mega Arcana/Celestial pack art variant `_1`/`_2` (same contents);
  `common_events.lua:1947` first-shop Buffoon pack art variant; `card.lua:959`
  `Card:get_id()` for Stone cards returns `-random(100,1e6)` so Stone cards never pair
  (treat as "unique id").
* `game.lua:2167` resets any per-key state that is exactly `0` to `pseudohash(k..seed)` at
  run start/load — only reachable via save files; irrelevant for fresh runs.

### NaN / "toxic" inputs — real, rare, and reproduced exactly
`pseudohash` multiplies by `1/num`. If an intermediate lands near an integer, `num` becomes
tiny, the next intermediate becomes huge with coarse ulp, and can be an exact integer →
`num == 0` → `1.1239285023/0 = inf` → `inf % 1 = NaN` → NaN for the rest of the hash. The
key's state is then NaN forever: `%.13f` prints `nan`, `tonumber` gives NaN, `abs` clears the
sign, `(NaN + hashed)/2` is NaN, and `math.randomseed(NaN)` seeds all four LFSR words with
the NaN bit pattern — a perfectly deterministic but fixed stream. The port reproduces every
step bit-for-bit, including LuaJIT's NaN sign (`0xfff8…` from `inf % 1`, `0x7ff8…` after
`math.abs`); Python's `inf % 1.0` gives the opposite sign and `x/0.0` raises, both handled.
Empirical rate: **0 in 4.6 M random (key, seed) pairs**, but the corpus hit one real case:
`pseudohash("erratic7LB2WVPK")` is NaN in the actual game (trace: `x = 370.0000000000267`
→ `num = 2.7e-11` → `x = 13084155696724.0` exactly → `0` → NaN). Harmless for normal play
(`erratic` only fires on the Erratic deck) but it shows the path is reachable. Also
`pseudohash("") == 1.0` (loop never runs) and a NUL byte is benign.

---

## 5. Performance (Python 3.13.5, RTX-3080Ti desktop, single core, pure Python)

| Operation | calls/s | µs |
|---|---|---|
| `LuaJITRandom.seed(d)` (table-driven 10 discards) | 461 k | 2.2 |
| `LuaJITRandom.seed_reference(d)` (literal 10 steps) | 130 k | 7.7 |
| `LuaJITRandom.random()` | 1.32 M | 0.76 |
| `LuaJITRandom.random_int(1, 100)` | 1.18 M | 0.85 |
| `pseudohash` (14-byte string) | 826 k | 1.2 |
| `lcg_step` (`%.13f` round trip) | 3.5 M | 0.28 |
| `PseudoRandom.pseudoseed(key)` (warm key) | 2.5 M | 0.40 |
| **`PseudoRandom.pseudorandom(key)`** | **287 k** | **3.5** |
| `PseudoRandom.pseudorandom(key, 1, 100)` | 269 k | 3.7 |
| `PseudoRandom.pseudorandom_element(list52, key)` | 235 k | 4.3 |
| `PseudoRandom.pseudoshuffle(list52, key)` | 21.7 k | 46 |
| `PseudoRandom.clone()` (3 / 200 keys) | 3.6 M / 1.3 M | 0.28 / 0.77 |

The LFSR update is GF(2)-linear, so the 10 discard steps in `math.randomseed` are folded
into one table lookup per input byte per component (8×256 entries × 4 components, built at
import in ~20 ms, verified against the literal form in `test_table_seed_equals_reference_seed`).
A full run does on the order of 10³–10⁴ keyed draws, so RNG is ~10–40 ms per run; tree
search should clone via `PseudoRandom.clone()` / `snapshot()`/`restore()` (dict copy + 4 ints).

---

## 6. API summary

```python
from rng import PseudoRandom, LuaJITRandom, pseudohash, normalize_seed, pseudoseed_predict

p = PseudoRandom(normalize_seed("7lb2wvpk"))      # -> seed "7LB2WVPK"
p.pseudorandom("Joker1")                         # float in [0,1)
p.pseudorandom("Joker1", 1, 100)                 # int in [1,100]
elem, i = p.pseudorandom_element(pool_list, "Joker1")     # list in Lua-sorted order
value, key = p.pseudorandom_element(proto_dict, "front")  # dict sorted the Lua way
p.pseudoshuffle(cards, "nr1")                    # in place; pass cards in sort_id order
p.pseudoseed("Tarot1")                           # raw seed value (feeds p.rng.seed(...))
p.rng                                            # the LuaJITRandom (global math.random)
snap = p.snapshot(); p.restore(snap); q = p.clone()
p.get_key_state("Joker1"); p.set_key_state("Joker1", x)   # G.GAME.pseudorandom[key]
```

`pseudorandom` / `pseudorandom_element` / `pseudoshuffle` also accept an already-computed
seed float in place of the key string, mirroring how the Lua passes `pseudoseed(...)` around.

---

## 7. Open questions / hand-offs

* **Real-binary confirmation** (Agent D): the oracle proves parity with LuaJIT's *algorithm*;
  ante-1 shop/voucher/boss matches on known seeds are what prove LÖVE's bundled LuaJIT
  behaves the same (FMA/x87 caveats in §3). The public seed analyzers (Immolate etc.) implement this same
  TW223 + π/e seeding and match the game, so I expect no surprise, but it is unverified here.
* Ties in `table.sort` (Lua's sort is unstable) can only matter if two pool entries share a
  `sort_id`/key, which the game's data does not do — Agent B should keep that invariant.
* `mp/` has no `__init__.py`; tests put the repo root on `sys.path` (`tests/conftest.py`) and the
  package uses relative imports, so `from rng.core import …` works from anywhere under `mp/`.
