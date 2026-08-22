# NOTES_ORDER — The Order switch + MLB voucher path in the generation layer

**Agent W2, Phase 2, 2026-08-21.**  Files: `rng/generate.py` (+ `RunState.key_scope` /
`ruleset` / `blind_key` / `blind_type`, The Order algorithms, MLB voucher path),
`rng/core.py` (one additive method, `PseudoRandom.drop_key`), `tests/test_the_order.py` (new,
151 tests), `engine/balatro_sim/game.py` (one line: `rs.key_scope = queue_scope`, W1 had
already wired `rs.ruleset`), this note.  Ground truth: the installed BalatroMultiplayer mod
(`$MOD` = `%APPDATA%/Balatro/Mods/Multiplayer`, v0.5.2) + Steamodded `smods-1.0.0-beta-1620a`
next to it; nothing from either is copied into the repo — the test reads the mod's toml and
Lua from `$MOD` at test time (`BALATRO_MP_MOD_DIR` to relocate) and skips without it.

## 0. Gates (run from repo root, after all edits)

| gate | result |
|---|---|
| `python -m pytest mp/tests -q` | **544 passed / 2 xfailed / 0 failed** (393 + 151 new in `test_the_order.py`) |
| `python -m pytest mp/engine/tests -q` | **1609 passed / 10 skipped / 3 xfailed / 0 failed** (final run; an earlier run during W1's concurrent `env_mp.py` rewrite showed one import error + one state-machine failure in W1's files, both gone by hand-off) |
| `python -m mp.oracle.engine_parity --antes 1-8 --rerolls 5 --quiet` | **126/126 exact through ante 8** (vanilla path byte-identical) |
| `python -m mp.oracle.parity_check --antes 1-8 --variant faithful` | **126/126 exact through ante 8** |
| `python -m pytest mp/tests/test_generate_oracle.py -q` | 129 passed (the Phase-0 vanilla Lua oracle, untouched) |

No cached fixture was regenerated.

## 1. What the mod actually does (corrections to the brief)

Read `$MOD/lovely/TheOrder.toml` (all 14 patches) and `$MOD/compatibility/TheOrder.lua`
(696 lines) in full, plus the Steamodded files the toml targets.  Three things the brief
(§1.7) did not say:

1. **The `*` seed prefix is NOT display-only.**  The game.lua patch runs *before*
   `hashed_seed = pseudohash(seed)` and `pseudoseed(key)` hashes `key..G.GAME.pseudorandom.seed`,
   so under The Order every stream of the run is the vanilla stream of the seed `'*'..seed`.
   `RunState.effective_seed` / the `key_scope` setter implement this; `RunState.seed` stays the
   lobby seed.
2. **The ante → 0 substitution is mostly one wrapper, not a list of sites.**  The mod wraps
   `create_card`: for the duration of the call `G.GAME.round_resets.ante = 0` (so *every*
   `..ante` inside it — soul, rarity, pool, front, stickers, edition, Enhanced pool — reads 0)
   and `key_append` is rewritten: Tarot/Planet/Spectral → `_type` (`_type..'_pack'` in
   `G.pack_cards`), Base/Enhanced keep theirs, everything else (jokers) → nil.  Outside
   `create_card` only `cdt`, the pack key, `halu`, the Standard-pack keys and `SMODS.poll_seal`
   get `MP.ante_based()` = 0.  Tags, `idol/mail/anc/cas`, `boss` keep the REAL ante.
3. **Algorithms change, not just keys.**  Jokers are picked by the mod's own loop (own
   sticker polls, own `'_sticker'` queue, edition keyed by pool instead of by `key_append`);
   resamples re-step the pool stream; vouchers come from a paired-culled pool; playing-card
   shuffles and every `pseudorandom_element` over Card lists (castle, hook, random_destroy,
   joker picks…) use a value-ranking scheme (`give_shufflevals`); `reset_idol_card` /
   `reset_mail_rank` are replaced outright; To Do / Orbital use the hand `order` list.

MLB itself (`rulesets/majorleague.lua`) forces `the_order = false`; the only generation
change under MLB is the voucher path (§3, row "Voucher"), because the mod guards its voucher
overrides with `should_use_the_order() or is_major_league_ruleset()`.  Bans are
`G.GAME.banned_keys` (UNAVAILABLE in place; `MP.should_exclude_from_pool` only removes the
mod's own `*_mp_*` keys, so vanilla pool indices are unchanged) — W1's `banned_keys`.

## 2. Engine contract (W1): which fields to set

```python
rs.key_scope = "ante" | "run"      # "run" = The Order; set BEFORE any draw (re-seeds '*'..seed)
rs.ruleset   = "vanilla" | "mlb"   # "mlb" = culled 'Voucher0' vouchers (shop + Voucher Tag)
```
Both are plain `RunState` fields (documented at the top of the class); W1's
`_init_game_vars` already sets them (`rs.ruleset = self.ruleset`,
`rs.key_scope = self.queue_scope`) right after construction, before `start_run`.  Verified:
`BalatroGame("EXAMPLE1", ruleset="mlb")` draws its ante-1 voucher from `Voucher0`
(`v_crystal_ball`, vs vanilla `v_wasteful` from `Voucher1`).

**The Order only** (not needed for MLB; listed so the switch is complete):

* maintain `rs.blind_key` (`G.GAME.blind.config.blind.key`: `'bl_small'`, `'bl_big'`,
  `'bl_hook'`, `'bl_mp_nemesis'`…) and `rs.blind_type` (`G.GAME.blind_on_deck`:
  `'Small'/'Big'/'Boss'`) at blind start (before the `nr` shuffle) and leave them on the
  *defeated* blind for the `cashout` shuffle.  `shuffle_deck` upgrades the engine's existing
  `Keys.new_round_shuffle(ante)` / `cashout_shuffle(ante)` keys automatically and raises if
  the two fields are unset under `"run"`.
* round picks: call `generate.reset_round_picks(rs, playing_cards_in_sort_id_order,
  previous_ancient)` instead of the inline `pseudorandom_element` calls in
  `game._round_end_resets` / `round_cards.py` (same result as today under `"ante"`; under
  `"run"` it runs the mod's idol/mail/castle algorithms).
* Hallucination: `Keys.halu_for(rs)` instead of `Keys.halu(ante)` (`jokers/misc.py:319`).
* To Do / Orbital already take `state` (`to_do_hand`, `orbital_hand`): nothing to do.
* joker picks already route through `generate.hex_/ankh/ectoplasm/wheel_of_fortune`
  (now Order-aware via `pick_joker`); `immolate` likewise.  Other Card-list picks the engine
  does itself (Hook, Cerulean Bell, Crimson Heart, random_destroy, Invisible, Madness,
  Perkeo) would need `generate.order_pick(rs, items, key, jokers=...)` under The Order.

## 3. Key-site table (vanilla → The Order → MLB)

`<a>` = real ante.  "MLB" = `ruleset="mlb"`, The Order off.  Sources: `T` = TheOrder.toml
patch, `W` = the mod's `create_card` wrapper (ante := 0, key_append rewrite), `L` = other
TheOrder.lua code.

| site (vanilla key) | The Order | MLB | src |
|---|---|---|---|
| seed hashed by `pseudoseed` | `'*'..seed` (every stream) | seed | T game.lua |
| `boss` | `boss<a>` | `boss` | T |
| `Voucher<a>` (shop), `Voucher_fromtag` (tag) | `Voucher0` for both; culled pairs; redraw = same stream; `Voucher0<it>` after 1000 | **same as The Order** | L |
| `Tag<a>` (+`_resample<it>`) | unchanged | unchanged | — |
| `cdt<a>` | `cdt0` | `cdt<a>` | T |
| `shop_pack<a>` | `shop_pack0` | `shop_pack<a>` | T (SMODS get_pack line = vanilla line) |
| `rarity<a><app>` (sho/buf/jud/…) | `rarity0` (shared by every joker source; `jud` with eternals enabled: `order_jud_rarity` instead); The Soul rolls nothing | `rarity<a><app>` | W+T, SMODS rarity.toml |
| `Joker<r><app><a>` | `Joker<r>0` / `Joker<r>0_sticker` (when a sticker was rolled); `Joker4` legendary | vanilla | T |
| `Joker..._resample<it>` | pool stream re-stepped; `_resample<it>` only after 1000 | vanilla | T |
| `edi<app><a>` (joker edition, every joker source) | `ediJoker<r>0` / `ediJoker<r>0_sticker` (one step per loop iteration whether or not the draw was UNAVAILABLE); `ediJoker4` for The Soul | vanilla | T |
| `etperpoll<a>` / `packetper<a>` | `_etperJoker<r>0` (shop+pack, always), per loop iteration | vanilla | T |
| `ssjr<a>` / `packssjr<a>` | `_rentJoker<r>0` (rentals enabled only) | vanilla | T |
| `Tarot<app><a>` etc. (shop `sho`, created `emp/8ba/hal/car/vag/sup/sea/sixth/pri`) | `TarotTarot0`, `PlanetPlanet0`, `SpectralSpectral0` | vanilla | W |
| `Tarotar1<a>`, `Spectralar2<a>`, `Planetpl1<a>`, `Spectralspe<a>` (packs) | `TarotTarot_pack0`, `SpectralSpectral_pack0`, `PlanetPlanet_pack0` | vanilla | W |
| consumable `_resample<it>` | pool stream re-stepped (fallback after 1000) | vanilla | T |
| `soul_<Type><a>` | `soul_<Type>0` | vanilla | W |
| `front<app><a>`, `Enhanced<app><a>` (`sho`/`sta`) | `frontsho0`, `frontsta0`, `Enhancedsho0`, `Enhancedsta0` | vanilla | W |
| pack-card / shop-card editions for consumables | none (unchanged: vanilla never polls them either) | — | — |
| `stdset<a>`, `standard_edition<a>`, `stdseal<a>`, `stdsealtype<a>` | `…0` | vanilla | L (Standard recipe + `poll_seal` wrapper) |
| `halu<a>` | `halu0` | vanilla | T (SMODS line) |
| `shuffle` (run start) | key unchanged, **algorithm = value ranking** | vanilla | L |
| `nr<a>` | `nr<a><blind_key><blind_type>` + value ranking | vanilla | T+L |
| `cashout<a>` | `cashout<a><defeated blind_key><type>` + value ranking | vanilla | T+L |
| `immolate` (pseudoshuffle over hand) | value ranking | vanilla | L |
| `idol<a>` | same key, mod's scored weighted walk | vanilla | L |
| `mail<a>` | same key, count-weighted walk in rank order (see §4.2) | vanilla | L |
| `anc<a>` | unchanged | unchanged | — |
| `cas<a>` | same key, value-ranked pick | vanilla | L (pseudorandom_element override) |
| `hex`, `ankh_choice`, `ectoplasm`, `wheel_of_fortune` (pick), `hook`, `cerulean_bell`, `crimson_heart`, `random_destroy`, `invisible`, `madness` | same keys, value-ranked pick (joker / playing-card lists) | vanilla | L |
| `to_do`, `orbital` | same keys, candidates in hand `order` (= `HANDLIST`) instead of `pairs` order | vanilla | T + `MP.sorted_hand_list` |
| `illusion`, `omen_globe`, `erratic`, `aura`, `sigil`, `ouija`, `*_create`, `spe_card`, `cert_fr`, `certsl`, `marb_fr`, `perkeo`, `misprint`, all `prob_roll` keys except `halu` | unchanged | unchanged | — |

Under The Order the patched `create_card` returns right after setting the mod's stickers /
edition, so vanilla's `all_eternal` is applied but the Steamodded sticker-apply loop is not
(vanilla stickers define `should_apply = false`, so nothing is lost at any stake).

## 4. What was verified, and how

`tests/test_the_order.py` builds a second Lua oracle: the vanilla reference files
(`misc_functions.lua`, `common_events.lua`, the `create_card_for_shop` and `Card:open`
slices, `init_item_prototypes` / `init_game_object`) with the mod's toml pattern patches
applied to the text in Python (lovely's trimmed-line match + re-indent; every expected
pattern's match count is asserted: boss 1, seed 1, pack 1, nr 1, cashout 1, cdt 1, resample
2, joker block 1, sticker/edition block 1), then **the mod's `TheOrder.lua` executed verbatim**
on top.  The `nr`/`cashout`/seed lines that live in files the oracle does not load are
repeated verbatim in the harness and patched the same way.  Three modes are driven through
one script — `order` (`key_scope="run"`), `mlb` (`ruleset="mlb"`), `vanilla` (mod loaded, both
off; must equal plain vanilla, guarding the stubs) — over 22 seeds × antes 1-8, 3 shops per
ante with 2 rerolls, every pack opened, a Voucher Tag draw per ante, `nr`+`cashout` shuffles
per blind with `bl_small/bl_big/<boss>`, idol/mail/ancient/castle after every round,
`used_jokers` and `bosses_used` per ante; plus creation paths (Judgement ×2, Soul ×2, Wraith,
Riff-raff ×2, Top-up, Emperor ×2, High Priestess, Sixth Sense, 8 Ball, Purple Seal,
Hallucination, Rare/Uncommon Tag, Voucher Tag ×2, Aura ×3, `halu` key), joker picks with
duplicate keys (hex/ankh/ectoplasm), a 60-card deck with random enhancements/seals/editions/
stone cards/duplicates through `shuffle`, `nr`, `immolate`, `cas/hook/cerulean_bell/
random_destroy` picks and the round resets, Orbital/To Do under The Order; scenarios: buy,
Gold stake (the `_etper`/`_rent`/`_sticker` queues and `order_jud_rarity`), Showman + buy,
Attrition bans, MLB + Attrition bans.  A perturbation of the edition key / voucher key / mail
algorithm produces mismatches, so the comparison is live.

Executed as REAL Lua: everything in §3 marked T, W, L (including `get_culled`, the voucher
loops, `give_shufflevals`/`give_stdval`, both overrides, the idol/mail replacements,
`MP.ante_based`/`order_round_based`/`sorted_hand_list`, the Standard-pack recipe).

Ported by reading / stubbed (our own Lua in `_LUA_MOD_STUBS`, each documented there):
`MP.should_use_the_order` / `is_major_league_ruleset` as flags; `SMODS.poll_seal({mod=10})`
as the vanilla seal roll (weights 4×W with a 2 % base: gate `1-0.02·mod`, thresholds
0.75/0.5/0.25 in pool order Red/Blue/Gold/Purple — algebraically identical);
`SMODS.create_card(t)` as argument-table → `create_card` + `set_edition`/`set_seal`;
`SMODS.get_next_vouchers` (non-Order path) = `get_next_voucher_key` with a spawn set;
`SMODS.size_of_pool`; `SMODS.Rank`/`Suit` registration order (2…9,10,J,Q,K,A; Diamonds,
Clubs, Hearts, Spades) and `nominal`/`face`; stickers' `should_apply = false`; Steamodded's
legendary-before-roll rarity line.  Not executed: Steamodded's `poll_edition` /
`poll_rarity` (read; stream-equivalent for vanilla content: same thresholds 0.96/0.98/0.994/
0.997 with `edition_rate` on the non-negative ones, 0.7/0.95 rarity — differing only at
exact boundary values), the `halu` and `to_do` toml patches (card.lua sites not loaded;
`sorted_hand_list` itself is executed), and the real `G.deck.cards` order at shuffle time
(§4.1).

### 4.1 Identity among identical cards

The mod's shuffle ranks cards by value; cards of identical kind (suit, rank, enhancement,
seal, edition — Negative counts as nothing) tie, and which physical copy gets which draw is
decided by LuaJIT's **unstable** `table.sort` on the group's insertion order, i.e. on the
order of the list the shuffle was given (`G.deck.cards` in the game — which depends on how
cards were returned to the deck — vs creation order in the engine).  `lua_table_sort` ports
the algorithm verbatim, so for a given list order the result is bit-identical (the test
checks the first shuffle by identity and every later one by kind).  Since tied cards are
indistinguishable for play this cannot change an outcome — except for per-card state the
mod's `stdval` ignores (Hiker `bonus_chips`, debuffs, face-down), which is a genuine, tiny
gap: such copies may swap identity between the engine and the game.

### 4.2 Mod bugs reproduced as is

* `reset_mail_rank` sorts its rank list on a `count` field it never increments (the counts
  live in a separate map), so the intended "count descending" sort is a no-op and the walk is
  in rank order (2…Ace) with count weights.  Ported faithfully (`_order_mail`).
* `reset_idol_card` comparator's last line (`rank_index a < b`) is unreachable; the sort is
  tier ↓, score ↓, rank ↓, suit ↓ — a total order, so no tie handling needed.
* The `'_resample'..it` fallback after 1000 redraws (create_card, the Joker loop, vouchers)
  is reproduced even though it cannot trigger with vanilla pool sizes.

### 4.3 Ambiguities and what I chose

* Steamodded vs vanilla for the Order-off paths: Phase 0/1 verified against the vanilla Lua,
  the live MLB client runs Steamodded.  I read every Steamodded generation site the mod
  touches (`get_current_pool` patches, `poll_edition`, `poll_rarity`, `poll_seal`, `get_pack`,
  `get_next_vouchers`, the vanilla Booster recipes, `create_card_for_shop` → `SMODS.create_card`)
  and found them stream-equivalent for vanilla content, with one exception that is
  observable only under The Order: Steamodded resolves `_legendary` before the rarity roll,
  so The Soul does not step `rarity0` (vanilla steps `rarity<a>sou`, its own dead stream).
  `get_current_pool` follows Steamodded under `"run"` and vanilla otherwise.
* `SMODS.poll_rarity` uses `poll < cumulative` where vanilla uses `> 0.95 / > 0.7`; identical
  except at the exact boundary doubles.  Kept vanilla.
* `MP.sorted_hand_list(current)` for To Do: the creation-time patch runs after
  `to_do_poker_hand = nil`, so the list is NOT pre-filtered and vanilla's retry loop applies;
  the end-of-round re-pick (not modelled by the engine) would be pre-filtered.
* `Keys.new_round_shuffle/cashout_shuffle(ante, state)` keep the int-ante signature the
  engine uses and add an optional `state`; `shuffle_deck` upgrades the bare form itself.

## 5. Found, not fixed (other owners)

1. **Brief §1.7 "Seed gets a `*` prefix for display only"** is wrong (see §1.1) — lead.
2. **Steamodded may change the `pairs(G.GAME.hands)` order** used by Orbital / To Do under
   MLB (The Order off): Steamodded rebuilds `G.GAME.hands` through its PokerHand objects, so
   the hash layout (and `generate.HANDS_PAIRS_ORDER`, dumped from the vanilla constructor in
   the game's `lua51.dll`) may not hold on the modded client.  Checkable by running the
   Steamodded-patched constructor in the same DLL.  Affects two tags/jokers only.
3. **Engine `_round_end_resets` / `round_cards.py`** inline the vanilla idol/mail/castle draws;
   under The Order they must use `generate.reset_round_picks` (W1/W3; MLB unaffected).
4. **`jokers/misc.py` Hallucination** uses `Keys.halu(ante)`; `Keys.halu_for(state)` for The Order.
5. **MLB boss draw**: `reset_blinds` still calls `get_new_boss()` (consumes `boss`, increments
   `bosses_used`) and only then `$MOD/ui/game/round.lua:52-64` overwrites the Boss slot with
   `bl_mp_nemesis` for ante ≥ `pvp_start_round` — W1's engine must keep drawing the boss every
   ante so the stream and `bosses_used` stay aligned with the game.
6. **`objects/jokers/standard/bloodstone.lua`** re-keys Bloodstone as `bloodstone<order_round>`
   under The Order (a "standard-layer" rework; whether it loads under MLB depends on
   `MP.UTILS.get_standard_rulesets` — not checked).  Effects-side, W3/W5.
7. (resolved before hand-off) W1's transient `test_env_mp.py` / `test_sweep.py::TestEnvRevival` failures.
8. `DELEGATE_NOTES.md §3`'s site list (Phase 1's guess) is superseded by §3 here; the
   `idol/mail/ancient/castle` keys it lists as dropping the suffix do NOT.
