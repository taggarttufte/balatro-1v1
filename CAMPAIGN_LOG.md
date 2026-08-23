# MP Campaign Log

Subproject: **Balatro Multiplayer (MLB ruleset) statistical player.** Lives in `mp/`, separate from the BRL project. Not referenced from the top-level README/PROJECT_MAP by design.

## Locked parameters (2026-08-20)

- **Goal:** an engine with real-Balatro RNG parity + MLB rules + 3 decks, with infra to run N-agent same-seed tournaments and produce the N×N comparison matrix. Plan: `docs/MP_CAMPAIGN_PLAN_2026-08.md`.
- **Budget:** local compute only. No paid API spend without asking.
- **Latitude:** agents may create/modify anything under `mp/`. **Nothing outside `mp/` is touched** — BRL code, branches, `results/`, README all stay as-is. No commits, no branch changes, no merge of `fix/sim-fidelity-2026-07` (that is a BRL decision for Tagg).
- **Reference source:** Balatro 1.0.1o Lua extracted from the local Steam install to `mp/_reference/balatro_src/` — **gitignored, never committed, never copied into deliverables.** Port algorithms; do not vendor Lua.
- **Oracle strategy (upgraded):** `lupa` provides LuaJIT 2.1 in-process. Ground truth for the RNG core = Balatro's actual `pseudohash`/`pseudoseed`/`pseudorandom` executed in LuaJIT. Public seed analyzers (Immolate / TheSoul / Blueprint / balatrohq) are the *end-to-end* cross-check for shop/pack/voucher/boss output.
- **Gate:** Phase 0 must reproduce ante 1–3 shops/packs/vouchers/bosses on ≥10 known seeds exactly before Phase 1 keyed-RNG threading begins.

## Key facts established

- `pseudorandom(key)` chain (misc_functions.lua:279-319): `pseudohash(key..seed)` → per-key LCG step `|fmt("%.13f", (2.134453429141 + x*1.72431234) % 1)|` → average with `hashed_seed` → `math.randomseed(v)` → LuaJIT `math.random()`. **So LuaJIT's Tausworthe `math.random` must be ported too, not just the LCG.**
- `pseudohash`: `num = ((1.1239285023/num) * byte(str,i) * pi + pi*i) % 1`, iterating i from #str down to 1.
- `pseudorandom_element` (:253) and `pseudoshuffle` (:206) are the pool-draw and shuffle primitives.

## Phase 0 — Oracle spike

**Started 2026-08-20.** Fleet of 5, disjoint file ownership:

| Agent | Owns | Deliverable |
|---|---|---|
| A — RNG core | `mp/rng/core.py`, `mp/rng/luajit_random.py`, `mp/tests/test_rng_core.py`, `mp/rng/NOTES_CORE.md` | Python port validated against LuaJIT-executed Lua to 1e-13 |
| B — Pools + keys | `mp/rng/pools.py`, `mp/rng/keys.py`, `mp/rng/NOTES_POOLS.md` | Exact game-order pools; every pseudoseed key string with construction rule |
| C — Generation spec + port | `mp/rng/GENERATION_SPEC.md`, `mp/rng/generate.py`, `mp/rng/NOTES_GEN.md` | Algorithm for every generation event incl. resample; skeleton that reproduces an ante-1 shop |
| D — Ground truth | `mp/oracle/ground_truth/*.json`, `mp/oracle/SOURCES.md`, `mp/oracle/parity_check.py`, `mp/oracle/blueprint_runner/` | ≥10 seeds with ante 1–3 outcomes; second independent oracle if Blueprint runs locally |
| E — Engine fork | `mp/engine/balatro_sim/`, `mp/engine/tests/`, `mp/engine/FORK_NOTES.md` | `balatro_sim` from `fix/sim-fidelity-2026-07` + `clone()`/`legal_actions()` from balatro-mcts; tests green in new location |

Log entries follow.

---

### 2026-08-20 — Agent E (engine fork) — DONE ✅

- Forked `balatro_sim` @ balatro-rl `4411dbf` (`fix/sim-fidelity-2026-07`) → `mp/engine/balatro_sim/`; tests → `mp/engine/tests/{engine_tests,sim_tests}/`.
- Ported from balatro-mcts `ee75d11`: `clone()` (`game.py:291-389`), `legal_actions()` (`:662-763`), `_consumable_target_actions()` (`:765-812`), `JokerInstance.clone()` (`base.py:153-159`) + 3 clone test files + `bench_clone_step.py`. `game.py`/`base.py` now byte-identical to the mcts fork; kept balatro-rl's post-A1 `card_selection.py` (`N_INTENTS=4`) and post-audit `env_v7.py` (`OBS_DIM=443`).
- Standalone: `mp/engine/pytest.ini` + `conftest.py` that **raises** if `balatro_sim` resolves to the BRL package.
- **Tests: 847 passed / 13 skipped / 0 failed** (baseline 825/3; −10 `TestActionMasking` skipped for missing `train_sim`, +32 clone tests).
- **Bench:** clone ~15.9k/s (63 µs), step ~36k/s (28 µs), 6.2–6.8× faster than deepcopy. On target.
- `FORK_NOTES.md` written with provenance + inherited fidelity issues (fork-relative line numbers: `game.py` shifted +105/+257).

**🔴 NEW ENGINE BUG (not in update list §7) — logged as §7 item 0, NOT fixed (Phase 1):**
`State.BOOSTER_OPEN` is never entered anywhere in the engine. Via raw `step()`, buying a pack debits money and fills `booster_choices` but state stays `SHOP`; `pick_booster` is silently ignored and `legal_actions()` never offers it. **Booster packs are a pure money sink for the raw engine and for `env_v7`** (only `env_v5` works around it with its own substate). Verified: buy `p_celestial_mega` → $100→$92, state still `SHOP`, 5 pending choices, nothing granted. Consequence for BRL history: V7 could never have received pack contents through the engine path → another confound on every V7/V8 number. Consequence for MP: Wraith/Hermit/Fool reasoning (decision-economics doc §4) is unrepresentable until fixed.

- Side note: 3,022 clone-steps over 93 states, `legal_actions()`↔`step()` agreed with zero exceptions (`BOOSTER_OPEN` unreachable per above).

### 2026-08-21 ~00:10 — Agents A/B/C/D hit session limit mid-work; RESUMED ~03:15

State on disk at interruption: `rng/core.py` 302 L, `rng/luajit_random.py` 287 L, `tests/test_rng_core.py` (fixtures empty, NOTES_CORE missing); `rng/pools.py` 715 L / 126 KB, `rng/keys.py` 363 L (NOTES_POOLS missing); `rng/generate.py` 1,386 L (spec + notes missing); `oracle/blueprint_runner/{run_blueprint.ts, run_thesoul.js, check_fixtures.ts, vendor/}` (ground_truth/ EMPTY, schema/SOURCES/parity_check missing).

**RNG core signal before resume:** `pytest mp/tests/test_rng_core.py` → 3 passed, `test_luajit_random_sequences` failed. **First `math.random()` after seeding matches LuaJIT bit-for-bit.** Expected sequence shows identical consecutive doubles → the Lua oracle harness re-seeds between draws. Harness bug, not port bug. Told A to fix the harness, not the port.

All four resumed in place (context intact) with precise pick-up instructions.

### 2026-08-21 — Agent B (pools + keys) — DONE ✅

- `rng/pools.py` (715 L), `rng/keys.py` (363 L, 77 key records), `rng/NOTES_POOLS.md` (387 L incl. 150-row joker map).
- **Method: executed the game's real `Game:init_item_prototypes` (game.lua:216-843) in LuaJIT via lupa with stubbed globals and dumped `P_CENTER_POOLS`/`P_JOKER_RARITY_POOLS` AFTER the game's own `table.sort`.** Nothing hand-typed. This is the right pattern — reuse it for anything else pool-shaped.
- **Counts all correct:** jokers 150 (61/64/20/5), tarots 22, planets 12, spectrals 18, vouchers 32, tags 24, bosses 28 (23+5), decks 15, stakes 8, boosters 32 centers = 13 types, enhancements 8, editions 5, seals 4, P_CARDS 52.
- Sort rule `a.order < b.order`; `order` verified unique in every pool that is ever drawn from.

**Surprises that bind generation (all in NOTES §3):**
1. **Boss draw is ALPHABETICAL by key, not by `order`** — `get_new_boss` draws from a hash table and `pseudorandom_element` sorts hash tables by key string → `pools.BOSS_KEYS_ALPHA`.
2. Vouchers interleave base/upgrade; Voucher Tag key = `'Voucher_fromtag'`, **no ante suffix**.
3. Booster pool = 32 centers, weighted walk (total 22.42); **first pack of a run forced Buffoon without consuming `'shop_pack'`**.
4. `P_CARDS` draws (`front`/`erratic`/`cert_fr`/`marb_fr`) are key-string order (`C_2..C_9,C_A,C_J,C_K,C_Q,C_T,D_…`); Card-array draws sort by `sort_id`.
5. Rarity roll consumed even for The Soul; legendary key is bare `'Joker4'`; Spectral packs roll `soul_Spectral<ante>` twice per card; `'8ba'` shared by 8 Ball + Purple Seal.
6. `'to_do'`/`'orbital'` pools come from `pairs(G.GAME.hands)` — **hash order, not reproducible from source. Oracle-only.**

**Lead action taken:** `functions/UI_definitions.lua` (holds `create_card_for_shop`, the `'cdt'..ante` slot roll, `'sho'` append, Illusion edition) was not in the original extraction. **All 33 .lua files now extracted** to `_reference/balatro_src/`. Agent C notified.

**Sim catalogue vs game (read-only survey — Phase 1 must reconcile):**
- Jokers: **only 7/150 agree on key+rarity+cost.** 21 renamed keys; **79 wrong rarities**; 107 wrong costs (sim prices by tier); 11 keys registered twice (last wins); 6 duplicate keys. 0 missing by name.
- Consumables: `c_heirophant`→`c_hierophant`; planets use `pl_` not `c_`; spectrals use `s_` not `c_`.
- Vouchers: `v_overstock_norm`→`v_overstock`; **missing `v_seed_money`, `v_money_tree`, `v_blank`, `v_antimatter`, `v_retcon`**; no `requires` chain.
- Boosters: missing `p_standard_mega`, `p_buffoon_mega`; no weights.
- Bosses: 5 showdowns renamed; **`bl_fish` missing**; no `boss.min` gates or min-use rotation.
- Tags / decks / stakes: **no catalogue in the sim at all.**

→ Implication: Phase 1 should **re-key the engine to game keys** (`j_*`, `c_*`, `v_*`, `bl_*`, `tag_*`) rather than maintain a translation layer. The rename count is too high for a shim to be safe.

### 2026-08-21 — Agent A (RNG core) — DONE ✅ BIT-EXACT

- **17/17 tests pass against live LuaJIT; 14 pass + 3 skip with `MP_RNG_NO_ORACLE=1`** (cached fixture `tests/fixtures/rng_ground_truth.json`, 2.3 MB; regen: `python mp/tests/test_rng_core.py --regen`).
- `rng/luajit_random.py`: TW223 combined-LFSR + π/e seeding from LuaJIT 2.1 `lib_math.c`/`lj_prng.c`; the 10 discard steps folded into GF(2) byte-lookup tables (3.5× faster; literal form kept as `seed_reference` and cross-checked). `seed(0.0)` reproduces LuaJIT's hard-coded `lj_prng_seed_fixed` constants — independent anchor.
- `rng/core.py`: `pseudohash`, `lcg_step`, `pseudoseed_predict`, `normalize_seed`, `PseudoRandom(seed)` with `pseudoseed/pseudorandom/pseudorandom_element/pseudoshuffle`, `snapshot/restore/clone`.
- **Compared (all 64-bit bit patterns, no tolerance):** 52 seeds × 55 keys × 3 draws × 8 intermediates = 68,640 chain values; `pseudorandom_element` on arrays / string-keyed tables / `sort_id` lists; `pseudoshuffle` at 11 sizes; raw `math.randomseed` on 267 seeds incl. ±NaN/±inf/denormals/2^64; `pseudohash` on ~1,000 inputs incl. all 255 bytes; `%.13f` LCG step on 3,060 inputs incl. exact ties; 85k `string.format`+`tonumber` round trips. **Zero mismatches.**

**Correction to my earlier diagnosis:** the duplicated consecutive doubles were a harness bug, but NOT re-seeding. A `BITS` helper type-punned `double[1]` through `uint64_t*`; once hot, LuaJIT's trace optimizer (strict-aliasing) forwarded a stale load — integer fields stayed correct, only hex-formatted doubles went stale. Fixed with byte-copy helpers + `jit.off()` + a 3,000-draw self-check at oracle construction.
**→ RULE FOR ALL LUPA ORACLES: `jit.off()`, never pun through FFI pointers, and pass doubles as hex bit patterns (lupa converts integral doubles to Python `int`, `-0.0` → `0`).**

**Real port bug found + fixed:** `pseudohash("erratic7LB2WVPK")` is **NaN in the real game** (7LB2WVPK is a famous real seed): an intermediate hits exactly `0.0`, Lua divides to `inf`, `inf % 1 = NaN`. Python raised `ZeroDivisionError`. Now reproduces Lua's NaN bit-for-bit incl. LuaJIT's sign bit and the downstream NaN-state chain. Rate otherwise: 0 in 4.6M random (key, seed) pairs.

**Perf (Py 3.13, 1 core):** `pseudorandom(key)` 287k/s (3.5 µs); `pseudorandom_element(52)` 235k/s; `pseudoshuffle(52)` 21.7k/s; `random()` 1.3M/s; `clone()` 1.3–3.6M/s. Not a bottleneck vs the engine's 36k steps/s.

**Phase 1 must know (NOTES_CORE.md §4):** seed box is `max_length=8`, all-caps, corpus `1-9A-Z` with **no `0`** (typed `0`→`O`, invalid chars dropped) — call `normalize_seed()` on user input; every draw reseeds the global generator so only per-key call counts matter; `pseudorandom_element` indexes a *sorted* key list (`sort_id` for runtime Cards, bytewise key order for string-keyed prototype tables, array order for arrays); `pseudoshuffle` sorts by `sort_id` first; LuaJIT `math.random(m,n)` has no range validation (mirrored); unseeded `math.random` in game logic is cosmetics-only (pack art variant, Stone `get_id()`).

**Open:** FMA/x87 — parity is proven vs LuaJIT's algorithm on MSVC x64; LÖVE's bundled binary assumed identical. Agent D's end-to-end seed checks confirm.

### 2026-08-21 — Agent C (generation spec + port) — DONE ✅, harness promotion in progress

- `rng/generate.py` (1,386 L) coded against the real `core.PseudoRandom` + `pools.py`. `generate_shop(RunState(seed="EXAMPLE1", ante=1))` → `j_egg`, `j_wrathful_joker`, `p_buffoon_normal_1`, `p_standard_mega_1`. `python -m mp.rng.generate EXAMPLE1 2` walks 2 antes.
- `rng/GENERATION_SPEC.md` — 18 sections, every `file:line` citation spot-checked. `rng/NOTES_GEN.md` — assumptions, untested areas, required ground-truth format (§5).
- Implemented: `RunState` (clone + acquire/release lifecycle), `get_current_pool`, `draw_from_pool`, `create_card`, `poll_edition`, `poll_seal`, `create_card_for_shop`, `generate_shop`, `reroll_shop`, `get_pack`, `open_pack`, `next_voucher`, `next_boss`, `next_tag`, `start_run`, `defeat_boss`, `reroll_boss`, `build_starting_deck`, `shuffle_deck`, consumable/joker creators (fool, blue_seal, aura, wheel_of_fortune, ectoplasm, hex, ankh, sigil, ouija, spectral_create_cards, immolate, certificate, marble_joker, misprint), `prob_roll` + `PROBABILITY_ROLLS`.

**🎯 GENERATION-LAYER ORACLE WORKS.** C loaded the real Lua `get_current_pool`/`create_card`/`get_pack`/`get_next_voucher_key`/`get_next_tag_key`/`get_new_boss`/`poll_edition`/`create_card_for_shop`/`Card:open` loop/`pseudoshuffle` verbatim into LuaJIT and diffed the port: **0 mismatches** over 30 seeds × 7 scenarios × 3 antes (fresh shops, 2 rerolls each, all pack types, purchases, Showman, bans, Gold-stake stickers, fresh profile) + 22 seeds of consumable/tag/Erratic/reset paths. Only diff: cosmetic `p_buffoon_normal_1/_2` art suffix (unseeded `math.random`). Harness being promoted to `mp/tests/test_generate_oracle.py` (was in session scratchpad).

**Fidelity corrections to my brief:**
1. **`used_jokers` is set on card CREATION (`Card:set_ability`), not purchase.** This is the dedupe mechanism for shelves AND packs — resample, not redraw — and pack cards also exclude the shelf behind them. The campaign brief assumed purchase; wrong.
2. Next-ante voucher is drawn at **boss death** (after `ease_ante`); tags + boss at **Cash Out**. First pack forced Buffoon w/o consuming `shop_pack1` (confirms B).
3. The Soul consumes `rarity<a>sou` then bare `Joker4`; `etperpoll` consumed at every stake; negative edition ignores Hone/Glow Up.

**Delegated back to C:** dump LuaJIT `pairs(G.GAME.hands)` order for `to_do`/`orbital` as an oracle-derived constant + assertion test. MLB `banned_keys` left as a `RunState` hook for Phase 2.

**Not yet covered by either harness:** cross-state call *sequencing* over a full run and the `Card` lifecycle stub — that's what Agent D's end-to-end seed ground truth confirms.

### 2026-08-21 — Agent D (ground truth) — DONE ✅ · **PHASE 0 GATE: PASSED** (one live check outstanding)

- **Two local generators ran headlessly:** Blueprint (TS port of Immolate/TheSoul, commit `62898ed0`, via vite-node — reproduces its own Immolate fixtures 5961/5961) and TheSoul's `immolate.wasm` (Emscripten build of C++ Immolate, commit `780c1c21`, bare Node). **They agree on every field for all 126 seeds** (~2,970 fields/seed). balatrohq's server-rendered ALEEB page agrees on all of ante 1.
- **126 ground-truth files**, antes 1–8, Red/White/unlocked: 50-deep shop queue per ante, all shop visits with both packs opened, soul spawns, legendary stream, buy-every-voucher branch. Game-internal keys throughout. `parity_check.py --validate-only` → 126/126 valid.
- 3 gaming-article seed claims (PTSBMSMQ, 8Q47WV6K, PQNVFI72/Triboulet) do NOT reproduce on either generator — articles are the weak source.

**KEY FINDING — every published analyzer omits one game rule:** `Card:set_ability` marks EVERY created card in `used_jokers` (cleared on `Card:remove`), so the real game never repeats a card across the two slots of one shop and never puts a displayed shelf card into a pack opened there; each resample also shifts the shared `_resample{n}` streams. D added this to a driver on Blueprint's `Game` class → stored as `variants.game_faithful_used_jokers` (953 fields differ corpus-wide: 222 same-shop dups, 319 pack/shelf collisions, 412 downstream shifts). **Agent C found the identical rule independently by reading `Card:set_ability` and validated it by executing the real Lua.** Two independent derivations converge.

**Parity (independently re-run by lead):**
- vs **faithful** variant: **126/126 seeds EXACT through ante 8, every field.**
- vs analyzer-as-published: 16/126 through ante 3 — and every one of the 953 mismatches is a flagged `analyzer_gap: used_jokers` field.

**Gate verdict:** the requirement was ≥10 seeds exact through ante 3. We have 126 through ante 8 against an oracle that two independent derivations + execution of the real Lua all agree on. **PASSED.** The one thing not yet confirmed against the *running game* is the faithful rule itself — but the Phase 1 work is identical either way (the rule is a one-flag difference in `generate.py`), so Phase 1 proceeds.

**🧑 LIVE CHECK FOR TAGG (2 min, closes the last loop):**
Seed **`7I4M53DL`**, Red Deck, White Stake. Beat the Small Blind, enter the first shop. The shelf should show **Banner** and **The Hierophant**; the packs should be a Buffoon and an **Arcana**. Open the Arcana pack and look at its **3rd card**:
- **The Lovers** → faithful rule confirmed (Hierophant on the shelf blocked it from the pack). ✅ Everything stands.
- **The Hierophant** → the published analyzers are right and the faithful rule is wrong → flip one flag in `generate.py` and regenerate; nothing else changes.
(Also expected at ante 1 on that seed: boss The Hook, voucher Wasteful, tags Speed/Economy, Buffoon pack = Acrobat + Wily Joker. If *those* don't match, the game version differs from 1.0.1o and we need to look harder.)

### 2026-08-21 — Agent C follow-up — harness promoted, **shipped-runtime parity confirmed**

- `mp/tests/test_generate_oracle.py` promoted (jit.off, no FFI punning, only str/int/bool cross the boundary, Lua sliced from `_reference` at test time). **`pytest mp/tests` → 146 passed / 1.7 s.** Perturbing a key makes it fail — it has teeth.
- **The game ships LuaJIT 2.0.5** (`lua51.dll` beside `Balatro.exe`), not 2.1. C loaded that DLL via ctypes and compared seeded `math.random` sequences, the keyed chain, `pseudoshuffle`, `pseudorandom_element`, and the unseeded first draw against Agent A's core: **all match to `%.17g`.** → A's FMA/x87 caveat is CLOSED; the core is bit-exact for the shipped runtime.
- Only 2.0.5/2.1 difference: string-hash `pairs()` order (2.1 randomizes per VM; 2.0.5 fixed). `pairs(G.GAME.hands)` order for To Do List / Orbital taken from the real DLL (stable 5/5 processes): `Flush House, Full House, Flush, Pair, High Card, Straight Flush, Straight, Two Pair, Flush Five, Five of a Kind, Three of a Kind, Four of a Kind` → `generate.HANDS_PAIRS_ORDER`, asserted against the DLL (skips w/ reason if Balatro isn't at the default Steam path; `BALATRO_DIR` overrides).
- Still open (Phase 1 W8): end-to-end *game-state plumbing* around generation (run start, boss-defeat→voucher, Cash Out→tags/boss, shelf release on leave). MLB `banned_keys` hooked for Phase 2.

## Phase 1 — Engine correctness (REVISED architecture)

**Decision:** do NOT re-implement generation inside the engine. `mp/rng/generate.py` is oracle-verified; the engine **delegates** to it for everything that appears (shop shelves, packs, vouchers, bosses, tags, deck shuffle, created cards) and only threads keyed RNG through *effect* rolls (lucky, glass, bloodstone, 8ball, …) that live in scoring/jokers. This gives queues, resample/in-place blocking, dedupe, Showman, The Order switch, and tags for free. Prerequisite: the engine must speak game keys (only 7/150 jokers currently agree).

**Wave 1 (launched 2026-08-21):**

| WS | Agent | Owns | Task |
|---|---|---|---|
| W1 re-key (GATE) | P1-rekey | `engine/balatro_sim/{shop.py catalogue, consumables.py names, constants.py, jokers/*} registry keys`, `engine/tests/*` | Re-key everything to game keys with `pools.py` as the single source of truth for key/name/rarity/cost; remove sim-side rarity/price tables; fix 21 renames, 79 rarities, 107 costs, 6 dups, 11 double-regs, `c_heirophant`, `pl_`/`s_` prefixes, 5 showdown boss names; add missing keys as stubs (`v_seed_money`, `v_money_tree`, `v_blank`, `v_antimatter`, `v_retcon`, `bl_fish`, `p_*_mega`) with TODO effects. Tests green. |
| W6 tags | P1-tags | NEW `engine/balatro_sim/tags.py`, `engine/tests/sim_tests/test_tags.py` | Self-contained tag module: all 24 tags with effects, as pure functions over a narrow `TagContext` interface. No `game.py` edits yet — W2 wires it. |
| W7 repro | P1-repro | `engine/balatro_sim/env_v7.py`, `env_sim.py`, `env_v5.py`, `card_selection.py` | `_best_hand_score` must not touch live RNG (clone or throwaway rng); thread rng in `env_sim`/`env_v5`; kill every reachable `rng_of` global fallback path. |
| W8 harness | P1-harness | NEW `mp/tests/test_engine_invariants.py`, `mp/tests/test_engine_reachability.py`, `mp/oracle/engine_parity.py` | Stream-independence invariants (Lucky hit must not move next shop; owning a joker must not shift other slots); reachability probe (every joker/consumable measurably changes state); end-to-end ENGINE-vs-ground-truth harness with a scripted policy. Written against the target interfaces; expected RED until wave 2. |

**Wave 2 (after W1):** W2 delegate generation (`shop.py`/`game.py` → `generate.RunState`, booster state machine fix, tag wiring), W3 effect-roll keys (`scoring.py`, `jokers/*`, `consumables.py` effects, `game.py` glass/purple → `pseudorandom(key)`; delete `game.rng`).
**Wave 3:** W5 bug sweep (§7 list minus whatever W2/W3 already fixed), then run W8 harness green.

### 2026-08-21 ~05:00 — Wave 1 cut off by session limit (resets 3:40pm CT) — CHECKPOINT

All four wave-1 agents (W1 rekey, W6 tags, W7 repro, W8 harness) were terminated while still in their reading phase. **Zero files written**; `mp/engine/` is pristine, `pytest mp/engine/tests` → 847 passed / 13 skipped; `pytest mp/tests` → 146 passed. Agent contexts are preserved; resume each via its ID with "pick up where you left off" (W1 was mid-way through resolving 45 jokers that have >1 implementation; W7 was about to write the baseline mutation probe).

**Resume order next session:** W1 first (it gates W2/W3), W6/W7/W8 concurrently. Then wave 2 (W2 delegate generation + booster state + tag wiring; W3 effect-roll keys), then W5 bug sweep, then W8 green.

**Still needs Tagg:** the `7I4M53DL` live check (see Phase 0 gate entry above).

### 2026-08-21 — W6 P1-tags — DONE ✅

- `engine/balatro_sim/tags.py`: pure module, all 24 tags from `tag.lua:115-468` (ranges cited per branch). `Trigger` enum uses Lua `_context.type` strings verbatim. `TagContext` = 5 read-only fields + 17 hooks (unimplemented hooks raise `TagHookNotImplemented`). `TagState` mirrors `G.GAME.tags` + `shop_d6ed`/`shop_free`/`temp_handsize`/`temp_reroll_cost`, with drivers reproducing the Lua loop semantics.
- `engine/tests/sim_tests/test_tags.py`: **124 passed.** Per-tag effects, Economy boundaries + event-dependent ordering (`common_events.lua:68-108`), Top-up slot re-check, Rare `nope()`, one-edition-per-card, one-pack-per-pass, Boss Tag chaining (burns Director's Cut), 2× Investment, 2× Juggle, Double ×1/×2 and Double+{Handy,Boss,Charm,Investment,Orbital}, per-shop D6/Coupon guards, full shop cycle, clone independence, `TAG_DEFS == pools.TAGS`.
- `engine/TAGS_NOTES.md`: W2 wiring table (engine moment → Lua site → call), pack-interrupt protocol, hook table, not-modelled list.

**⚠ W2 must read:** `rng/generate.py` ALREADY runs `store_joker_create`/`store_joker_modify`/`voucher_add` internally against `RunState.tags`. W2 must pick ONE path (TAGS_NOTES options A/B) — running both double-applies shop tags.

W2 drivers: `TagState()` at run start; `skip_blind(...)`; `on_blind_select`; `on_new_blind_choice` (loop until `pending_pack is None`); `acquire('tag_double')` for Anaglyph/Diet Cola; `on_round_start/eval/end_cleanup`; `on_cash_out`; `on_shop_start`; `store_joker_create/modify` per slot; `on_voucher_add`; `on_shop_final_pass`.

### 2026-08-21 — W7 P1-repro — DONE ✅ (finding is broader than the brief)

**Root cause broader than "RNG draws":** `score_hand()` in the V7 dry-run also (a) bumped scaling jokers' `inst.state` per candidate, (b) **Space Joker wrote the live `gs.planet_levels`** — seed 0's V7 estimate went 3928 → 14556 between two back-to-back calls (hands levelled for free), (c) Vampire/Midas Mask/Sixth Sense rewrote card fields, (d) Card Sharp mutated a `set` nested in `inst.state` that shallow `state.copy()` doesn't isolate.
→ **V7's largest reward term (`+2.0 × played/best`) had a systematically inflated denominator for the whole of V7/V8.** Another confound on every historical number.

**Fix:** `card_selection.HypotheticalScorer` used by all three envs — snapshots `gs.rng.getstate()` once per hand; per candidate resets a private `random.Random` to it and scores against fresh card copies, one-level-deep joker-state copies, and a planet-dict copy; `rng=` always explicit. (Chosen over `clone()`-per-candidate: env_sim/v5 score all 218 subsets → ~8 µs vs ~14 ms each.) `env_v7._best_hand_score` → `model_held=True`; `env_sim`/`env_v5` `_update_play_combos` → `model_held=False`. `env_mp` audited, inherits via `step_hand` (its `_revive_boss_if_needed` `generate_shop` call is flagged for W2 — regenerates outside `_end_round`). OBS_DIM unchanged (443 / MP 447).

**Numeric estimate changed, necessarily:** seeds 0–4 first-call `_best_hand_score`: 3928→3928, 1260→390, 2650→2862, 1102→480, 3496→1748. Cost 0.71 → 0.95 ms/play. Documented in `engine/REPRO_NOTES.md`.

**Tests:** `engine/tests/sim_tests/test_env_rng_isolation.py` — **40 tests**: snapshot no-side-effect + determinism per env × 3 seeds; V7 reference-play fingerprint incl. `rng.getstate()`; scorer unit tests; same-seed trajectory with global `random` deliberately poisoned; AST static guards (no `score_hand(` without `rng=`, no module-level `import random` in envs). **Negative check:** pre-fix BRL env files swapped in → 29/40 fail exactly where predicted.

**Counts (W7's final run, with other agents' work included): 1279 passed / 13 skipped / 0 failed.**

**⚠ Engine wart for W2/W3 (owner of `jokers/base.py`):** `JokerInstance.clone()` — and hence `BalatroGame.clone()` — **shares nested containers** (Card Sharp `played_hands`, Satellite `planets_used`, `pending_consumables`) between clone and original. **MCTS siblings cross-contaminate.** 3-line one-level-deep copy fixes it (see `_clone_joker_for_dry_run`). This was present in balatro-mcts too → the 2026-05 cold runs had it.

Cross-agent note: P1-rekey touched `env_v7.py`/`env_v5.py` for key renames (kept) and flipped line endings to LF (W7 normalized back to CRLF). 29 transient failures mid-session were P1-rekey's in-flight `consumables.py`/`shop.py` edits; resolved by the final run.

### 2026-08-21 — W1 P1-rekey — DONE ✅ **(WAVE-2 GATE OPEN)**

- **1279 passed / 13 skipped / 0 failed** (`pytest mp/engine/tests`). New `test_game_keys.py` alone: 263.
- **New `balatro_sim/game_keys.py`** derives EVERY catalogue from `mp.rng.pools` at import. Nothing hand-typed remains. `JOKER_REGISTRY` is now a dict subclass that **raises on duplicate registration**; `jokers/__init__.py` raises at import if registry ≠ pools. Registry: 197 registrations / 161 keys → exactly 150.
- Jokers: 30 renames incl. the game's own typos (`j_gluttenous_joker`, `j_selzer`). name/rarity/cost/order match pools 150/150.
- Consumables: `pl_*`/`s_*` → `c_*`. **Correction: `c_heirophant` IS the game key (the game's typo)** — my brief had it backwards; pools + ground truth are authoritative. Phantom `c_wheel` in env_v5 fixed.
- Vouchers: 32 from pools, `VOUCHER_REQUIRES` exported. Implemented `v_seed_money`/`v_money_tree` (new cloned `game.interest_cap`), `v_blank`, `v_antimatter` (+1 slot). Stub: `v_retcon`.
- Bosses: showdowns → `bl_final_*` (game keys); `BOSS_MIN_ANTE`/`BOSS_MAX_ANTE` exposed. Stub: `bl_fish` → `UNMODELLED_BOSS_BLINDS`.
- Boosters: `p_<kind>_<size>` from 32 centers → **15 types, not 13** (sim was missing 2 megas); `BOOSTER_PICKS` + weights for W2.
- `constants.TAGS/DECKS/STAKES` + `ENHANCEMENT_KEY`/`EDITION_KEY` maps.

**Duplicate resolutions: 36 keys had two implementations (not 11).** Full table `engine/REKEY_NOTES.md` §2.3. Behaviour-changing picks where the previously-WINNING copy was wrong: **8 Ball** (+20 chips → creates Tarot), **Four Fingers** (+10 chips → `four_finger_mode`), **Satellite** (empty TODO → tracks planets). `j_duo..j_tribe` → XMult impls; `j_ticket` = Golden Ticket; `j_space` = working level-up; `j_ring_master` = Showman stub (the "reroll boss" copy was Director's Cut mislabeled). `j_lucky_joker` removed. 246 orphan pointer comments stripped. **Unmapped keys: none.**

**For W5 (found, not fixed — REKEY_NOTES §6):** Photograph resets per round not per hand; Ancient suit never changes; Flower Pot reads all played cards; Swashbuckler counts itself; Stencil hard-codes 5 slots (matters now Antimatter is buyable); Flash Card / Hallucination / Matador depend on hooks with NO callers; Stuntman −2 hand size unapplied; Director's Cut modelled as shop reroll; `random_joker_key` ignores pool flags (W2 replaces it).

### 2026-08-21 — W8 P1-harness — DONE ✅

- `oracle/engine_parity.py`: ENGINE-vs-ground-truth. `EngineDriver` walks `BalatroGame.step()` with a scripted policy, diffs shelves/packs/voucher/boss/tags per ante vs `ground_truth/<SEED>.json` (faithful default); replays the policy's own consumption into the expectation; `--reference generate` replays the engine's action history through `generate.RunState`; `--buy-vouchers`; `--probe` lists which target hooks exist. **Today: 0/126 exact through ante 1** (expected — generation not delegated yet).
- `tests/test_engine_invariants.py` (14): effect rolls don't move generation; ownership blocks only its slot + Showman lifts it; reroll k = queue[2k:2k+2]; divergent players replayed through RunState; determinism + clone isolation. **3 pass / 1 xfail / 10 fail — all "single stream / generation not delegated".**
- `tests/test_engine_reachability.py` (235 probes over all 150 jokers, 52 consumables, 32 vouchers; 1.0 s): **185 pass / 4 xfail / 46 fail — all engine gaps, none harness bugs.** Verified the probe fails on no-op scenarios.
- `tests/HARNESS_NOTES.md`: per-test assertions, expected-red reasons, hook list, run commands.

**Reachability findings (46) — this is W5's worklist:** 10 sentinel producers (incl. **`j_marble` → `"stone_card"`, not in the §7 list**); 4 hand-eval flags set AFTER `evaluate_hand` (`j_four_fingers`, `j_shortcut`, `j_smeared`, `j_pareidolia`); 9 hooks never invoked (`j_flash`, `j_hallucination`, `j_hologram`, `j_lucky_cat`, `j_matador`, `j_perkeo`, `j_astronomer`, `j_chaos`, `j_credit_card`); 5 flags nothing reads (`j_chicot`, `j_gift`, `j_oops`, `j_turtle_bean`, `j_hiker`); `j_ancient`/`j_castle` dead `on_init`; `j_mime` wrong mechanic; `j_campfire`, `j_diet_cola`, `j_invisible`, `j_red_card`; 12 vouchers no-op or need `run_state`. xfail (no measurable change definable): `j_luchador`, `j_ring_master` (needs `run_state.showman`), `v_blank` (by design), `v_retcon`.

**Engine hooks W2/W3 must provide (from `--probe`, 9/12 missing):** `BalatroGame(seed=str)` via `normalize_seed`; `game.run_state` = synced `generate.RunState` (acquire/release/used_vouchers/showman/rates/blind_tags/boss_blind); `run_state.rng = PseudoRandom` and **`game.rng` deleted**; `boss_blind` known at ante start; `blind_tags {'Small','Big'}`; `game.tags` (skip → tag, not +$5); booster purchase enters `BOOSTER_OPEN`, `skip_booster` releases, `pick_booster` acquires; `booster_choices` entries with `.key/.edition/.enhancement/.seal/.front`; shelf = `shop_joker_max` slots; `clone()` deep-copies `run_state` + clones `PseudoRandom`; voucher effects land in `run_state` (Overstock → `shop_joker_max`, Hone/Glow Up → `edition_rate`, Merchants/Tycoons/Magic Trick/Illusion → rates, Hieroglyph/Petroglyph → `ante`). Optional: `debug_win_blind()`, `debug_add_joker(key)`, `state_signature()`.

---

## ⏸ STOPPING POINT — 2026-08-21 (end of wave 1) — NEXT SESSION STARTS HERE

**Verified by lead, independently:** `pytest mp/engine/tests` → **1279 passed / 13 skipped / 0 failed**. `pytest mp/tests` → **334 passed / 5 xfailed / 56 failed** (all 56 expected-red: they test the wave-2 target state). `git status` → only `?? mp/` (nothing committed, nothing outside `mp/` touched, fix branch unmerged). 78 Python files, ~97 MB incl. ground truth (vendor/ and _reference/ gitignored).

**Done:** Phase 0 (oracle, 126/126 seeds exact through ante 8) and Phase 1 wave 1 (W1 re-key ✅, W6 tags ✅, W7 repro ✅, W8 harness ✅).

**Next session — wave 2 (two agents, then W5):**
- **W2 delegate generation:** `shop.py`/`game.py` → `generate.RunState` for shelves/rerolls/packs/vouchers/bosses/tags/shuffle/created cards; fix `BOOSTER_OPEN` never entered; wire `tags.py` per `TAGS_NOTES.md` (pick ONE shop-tag path — generate.py already applies store_joker tags); honour the hook list above; fix `JokerInstance.clone()` nested-container sharing. Success = `engine_parity.py` goes from 0/126 toward 126/126 and `test_engine_invariants` goes green.
- **W3 effect-roll keys:** `scoring.py`, `jokers/*`, `consumables.py` effects, `game.py` glass/purple → `run_state.rng.pseudorandom(<key from keys.py>)`; delete `game.rng`. Success = "effect rolls don't move generation" invariants green.
- **Then W5 bug sweep** from the combined list (REKEY_NOTES §6 + reachability's 46 + update-list §7), driving `test_engine_reachability` to green.
- **Still needs Tagg:** the `7I4M53DL` live check (Phase 0 gate entry).

## Phase 1 wave 2 — launched 2026-08-21 (finish Phase 1, then stop)

**Lead seam (done, 1279 green):** `BalatroGame.__init__(seed: int|str)` now builds `self.seed_str` (via `normalize_seed` / `seed_from_int`) and `self.run_state = generate.RunState(seed)` whose `.rng` is the real `PseudoRandom`; `clone()` deep-copies it. `self.rng` (legacy single stream) still exists — W3 deletes it. `game_keys.py` exposes `gen`, `core`, `normalize_seed`, `seed_from_int`.

| WS | Agent | Owns (functions, not just files) |
|---|---|---|
| W2 delegate | P1-delegate | `shop.py` (all of it), `game.py`: `_init_game_vars`, `_prepare_next_blind`/boss selection, `_end_round`, `_skip_blind`, `_end_blind_and_enter_shop`, booster state machine (`BOOSTER_OPEN`/`_pick_booster`/`skip_booster`), `_init_deck`/shuffle, run-start draws, tag wiring, `legal_actions` for new states; `consumables.py`: only the *created-card* paths (Judgement/Wraith/Soul/Emperor/High Priestess/Familiar/Grim/Incantation/Aura/Sigil/Ouija/Ankh/Hex/Immolate/Ectoplasm/Cryptid/DNA/Certificate → `generate.create_card`/creators); `jokers/base.py`: ONLY `JokerInstance.clone()` nested-container fix; `constants.py` as needed; `engine/tests/*` for the above |
| W3 effects | P1-effects | `scoring.py`, `jokers/*` except `base.py:clone()` (but INCLUDING `base.py:rng_of` → `run_state.rng`), `consumables.py`: only *probability rolls* (Wheel of Fortune etc.), `game.py`: ONLY glass shatter + Purple Seal + any other `self.rng` effect roll; then delete `self.rng` once nothing references it; `engine/tests/*` for the above |

### 2026-08-21 — W3 P1-effects — DONE ✅ (`game.rng` DELETED)

- **Gates (agent-run):** `pytest mp/engine/tests` **1321 / 10 skip / 0 fail** (+34 `test_effect_keys.py`); `test_engine_invariants` **14/14** incl. all three "effect rolls don't move generation"; `test_engine_reachability` **226 pass / 6 fail / 3 xfail** (was 185/46/4); `engine_parity --probe` → `keyed_rng` ok, 11/12 hooks (only `state_signature` missing).
- `jokers/base.py`: `ScoreContext.prng` + `run_state` + `probabilities_normal` etc. **`rng_of` now raises `MissingPRNG` — global-random fallback deleted.** `prob_roll(ctx, key, odds)` = `pseudorandom(key) < normal/odds`. Helpers `fire_hook`, `add_joker/remove_joker`, `init_joker`, `sync_probabilities` (Oops = 2**n), `create_consumable` → `generate.create_from_spec`. P1-delegate adopted these in `shop.py`/`game.py`.
- `scoring.py` rewritten to the Lua order: Space Joker in `before`; held phase per card per pass; **joker editions once per joker in `joker_main`** (Foil Greedy = +50, not +250); Lucky = `lucky_mult` then `lucky_money` per pass; Glass rolled in `_play_hand`. **Correction to my brief: Lua rolls `glass` once per scoring Glass card, NOT per retrigger** — `keys.py` was right; test pins it.
- `jokers/*`: every roll keyed (`8ball`, `business`, `bloodstone`, `parking`, `space`, `misprint`, `gros_michel`, `cavendish`, `halu<ante>`, `invisible`, `perkeo`, `to_do`, `madness`…). **All 10 sentinel producers now create real cards** (`Tarot8ba1`, `Joker1rif1`, `marb_fr`, `cert_fr`/`certsl`…). Fixed: Photograph, Ancient, Castle, Idol, Mail, Flower Pot, Swashbuckler, Stencil, Stuntman, Turtle Bean, Hiker, Gift, Lucky Cat, Chicot, Matador, Trading, Sixth Sense, DNA, Hanging Chad, Raised Fist, Even Steven, Madness.
- `game.py` (targeted): `game.rng` removed from `__init__`/`clone`; `_hook_ctx` carries keyed PRNG; boss rolls keyed (`cerulean_bell`, `hook`×2, `crimson_heart`, `aajk`×3, `madness`); Purple Seal → shared `Tarot8ba` stream; `add_card` fires Hologram; `BlindInfo.disabled` (Chicot). `consumables.py`: Wheel of Fortune = `generate.wheel_of_fortune`. `card_selection.py`: `HypotheticalScorer` clones `run_state.rng` + `restore()`s per candidate, `run_state=None` so dry runs never mark `used_jokers`; 40/40 isolation tests pass. New `round_cards.py`. **No `import random` anywhere in game logic (asserted).**
- **Remaining red → W5:** `j_four_fingers`/`j_shortcut`/`j_smeared` (`hand_eval.py` has no flag support); `j_space` (harness seeds all roll ≥0.25 first draw — probe needs `repeats=2`, not a bug); `j_flower_pot` (probe plays High Card, joker reads scoring hand — probe issue); `j_chicot` (boss debuff at play time). `EFFECTS_NOTES.md` has the full key table with file:line.

### 2026-08-21 — W2 P1-delegate — DONE ✅ · **ENGINE PARITY 126/126 THROUGH ANTE 8**

**Gates (independently re-run by lead):** `pytest mp/engine/tests` **1358 / 10 skip / 0 fail**; `test_engine_invariants` **14/14**; **`engine_parity --antes 1-8 --rerolls 5` → 126/126 seeds EXACT through ante 8, zero fallbacks** (was 0/126 through ante 1); `test_engine_reachability` 226 / 6 fail / 3 xfail; `--probe` 11/12 (only optional `state_signature` missing). No `random.Random` in game logic (only `env_sim.py:663` rollout helper, non-game).

- `shop.py` rewritten: shelves/rerolls/packs/vouchers from `generate` via `game.run_state`; `ShopItem` carries editions/stickers/coupons/playing-card fields; `BoosterChoice` objects; `used_jokers` = owned ∪ shelf ∪ open pack, rebuilt by `_sync_run_state` before every generation call + incremental acquire/release.
- `game.py`: run start via `gen.start_run` (deck in `sort_id` order — `7I4M53DL` first deal matches ground-truth deck order); `gen.defeat_boss` at boss death (**ante increments at boss defeat, as the game does**); tags + boss + `orbital` at blind select; `nr<ante>`/`cashout<ante>` shuffles; **`BOOSTER_OPEN` bug FIXED** (buy → open → pick acquires / skip releases, mega = 2 picks, tag packs interrupt blind select); **skip now grants the blind's tag and does NOT open a shop** (real game; the sim had invented skip→shop); tags wired option B (generate owns shop-time tags, `tags.py` drives the rest via `_GameTagContext`, all 17 hooks); Director's Cut/Retcon `reroll_boss` action; `debug_win_blind`/`debug_add_joker`; **`JokerInstance.clone()` one-level-deep fix + regression test**; `run_state`/`tag_state` cloned without deepcopy (77 µs/clone).
- `consumables.py`: all created-card paths through generate; effects corrected to card text (Wraith → $0, Familiar destroys a random hand card, Ankh keeps chosen+copy, Aura targets a playing card); vouchers routed into `run_state` (rates, Overstock w/ immediate slot fill, ante).
- **The Order:** `game.queue_scope` flag exists but is a **no-op** — `generate.Keys` needs a suffix hook; one-change spec in `DELEGATE_NOTES.md` §3. → Phase 2.
- 37 new tests `test_delegate.py`.
- **Harness fix (out of ownership, justified):** `engine_parity.py` recorded next ante's boss/tags one blind too early (before the boss was fought — impossible for any faithful engine). Now recorded at boss death. This was the only thing between 0/126 and 126/126.
- Other policies: `--buy-shelf --reference generate` 117/126 — every mismatch is the engine being MORE faithful than generate-only replay (Riff-raff/Hallucination creations, Turtle Bean/Gros Michel removals). `--buy-vouchers`: all 43 chain mismatches are Hieroglyph/Petroglyph ante shifts the analyzers ignore (verified 43/43).
- **For W5:** `bl_fish` now drawable but `_draw_to_full`'s wrong model is reachable; Invisible Joker/Perkeo copies; Ouija/Ectoplasm permanent hand size; Hieroglyph at ante 1 clamps (game goes to ante 0); `env_mp._revive_boss_if_needed` regenerates outside the round transition; `env_v5._step_pack_open` needs `BoosterChoice` isinstance; `JokerInstance.clone()` should copy `sort_id`.

**WAVE 2 COMPLETE.** Same-seed multiplayer is now architecturally possible in the engine.

### 2026-08-21 — W5 P1-sweep — DONE ✅ · **PHASE 1 COMPLETE (reachability 0 failures)**

**Gates (agent-run, lead should re-run):** `pytest mp/engine/tests` **1441 / 10 skip / 0 fail** (+79 new); `test_engine_invariants` **14/14**; **`engine_parity --antes 1-8 --rerolls 5` → 126/126 exact through ante 8** (never dropped); `test_engine_reachability` **233 / 0 fail / 2 xfail** (`j_luchador`, `v_blank` only); `pytest mp/tests` **393 / 0 / 2**; `--probe` **12/12**.

- **Hand-eval flags:** `hand_eval.py` rewritten as a port of `evaluate_poker_hand`/`get_flush`/`get_straight`/`get_X_same` with `four_fingers`/`shortcut`/`smeared` (Four Fingers off-card does not score, SF = union of the two 4-subsets, Flush Five with 4 suited; Shortcut = the Lua j=1..14 walk, Ace low+high, one skip, no wrap). Flags computed from the active board BEFORE evaluation (Crimson Heart roll moved ahead); envs' dry-run subset evaluators use the same flags.
- **Boss debuffs are a play-time predicate** (`_boss_debuffs_card` = `Blind:debuff_card`): Wild cards debuffed by every suit boss, Smeared extends the suit, Pareidolia + The Plant = everything; re-evaluated on draw / play / consumable use. Chicot verified end to end.
- **Face-down bosses modelled** (`Card.face_down`, `Blind:stay_flipped`): House / Wheel (`wheel` key) / Mark / Fish; obs hides the card (present bit only); `UNMODELLED_BOSS_BLINDS = []`; env_v7 boss one-hot → all 28 (**OBS_DIM 443→447, MP 447→451** — the 4 were drawable since W2 but encoded as "no boss").
- **Blind-start order is now the game's:** setting_blind hooks → `nr<ante>` shuffle → Juggle → draw → `first_hand_drawn` (Marble's Stone is in the first shuffle; Certificate's card joins a full hand).
- Invisible Joker (sort_id order, Negative stripped per `copy_card`, acquires), Perkeo **Negative copy takes no slot** (`negative_consumables` multiset; slot released on use — also fixes the leaked slot bump for Negative shelf/pack consumables), Ouija/Ectoplasm permanent `hand_size_mod`, **ante 0** (`BLIND_CHIPS[0]`, `get_blind_amount`; clamp gone) + **`ease_ante` no longer re-rolls the boss** (it consumed an extra `boss` draw), `JokerInstance.clone` copies `sort_id`, `state_signature()`, env_mp revival through `_end_round` (queue no longer consumed twice; revived shelf == ground truth), env_v5 `BoosterChoice` picks acquire/release.
- **§7 list: every item disposed** (table in `engine/SWEEP_NOTES.md` §2) — nothing open. Remaining honest gaps (§4): Crimson Heart per-play roll, Luchador sell-mid-blind, Gift Card/Negative on bare-key consumables, Baseball interleaving, The Order switch (Phase 2), MLB lost-PvP payout unknown.

---

## ✅ PHASE 1 COMPLETE — 2026-08-21 — STOPPING POINT (Tagg's request: finish Phase 1, then stop)

**Final gates, independently re-run by lead from a clean state:**

| Gate | Result |
|---|---|
| `pytest mp/engine/tests` | **1441 passed / 10 skipped / 0 failed** (started the day at 847) |
| `pytest mp/tests` (RNG oracle + generation oracle + invariants + reachability) | **393 passed / 2 xfailed / 0 failed** (xfail: `j_luchador`, `v_blank` — no measurable change definable) |
| `engine_parity --antes 1-8 --rerolls 5` | **126/126 seeds EXACT through ante 8** |
| `engine_parity --probe` | **12/12 hooks** |
| `git status` | only `?? mp/` — nothing committed, nothing outside `mp/` touched, fix branch unmerged |
| `random.Random` in game logic | none (only `env_sim.py:663` rollout helper, non-game) |

**W5 P1-sweep (final step):** `hand_eval.py` ported from `evaluate_poker_hand` with Four Fingers/Shortcut/Smeared/Pareidolia flags computed from the active board BEFORE evaluation (43 unit tests); boss debuffs as a play-time predicate (`Blind:debuff_card` port, Chicot end-to-end); face-down bosses (House/Wheel/Mark/Fish) modelled via `Card.face_down` + `Blind:stay_flipped` → `UNMODELLED_BOSS_BLINDS = []` → **OBS_DIM 443→447 (MP 451)** (those 4 bosses were drawable since W2 but encoded as "no boss"); Invisible Joker + Perkeo copies; Ouija/Ectoplasm permanent hand-size; ante 0 (`BLIND_CHIPS[0]`, clamp removed) **+ found: `ease_ante` was re-rolling the boss** (extra `boss` draw, now pinned); `clone()` copies `sort_id`; `state_signature()`; env_mp revival through `_end_round` (revived shelf == ground truth); Negative-consumable slot leak fixed; blind-start order corrected to the game's (hooks → `nr` shuffle → Juggle → draw). Every §7 item disposed in `SWEEP_NOTES.md`. +79 tests.

**What Phase 1 delivered, in one paragraph:** the engine speaks the game's keys (150/150 jokers 1:1 with a registry that raises on duplicates); ALL generation — shelves, rerolls, packs, vouchers, bosses, tags, shuffles, created cards — is delegated to the oracle-verified `generate.py` through `game.run_state`; every effect roll draws from its real per-key stream; the legacy single `random.Random` is gone; `BOOSTER_OPEN` works; skip grants a tag and doesn't open a shop; `clone()` no longer shares nested state; the V7 reward's dry-run is side-effect-free; and the engine reproduces 126 real seeds exactly through ante 8. **Same-seed multiplayer is now architecturally real in the engine.**

**Honest remaining gaps (SWEEP_NOTES §4 + DELEGATE_NOTES §3):**
- **The Order switch is a no-op** — `game.queue_scope` exists; `generate.Keys` needs a suffix hook (one-change spec in DELEGATE_NOTES §3). → Phase 2.
- Crimson Heart rolls per play not per draw; Luchador needs sell-mid-blind; Gift Card/Negative tracking per key not per card; Baseball interleaving.
- MLB lost-PvP payout unknown (revived loser currently paid like a winner) → Phase 2 + Tagg's practice-lobby check.
- Decks other than Red, all stakes, stickers: not started (Phase 2).
- `7I4M53DL` live check still outstanding (Phase 0 gate entry).

**NEXT SESSION = PHASE 2** (`docs/MP_CAMPAIGN_PLAN_2026-08.md`): MLB rules (nemesis blind w/ opponent-score target, 4 lives on any blind lost, comeback money, early-win rule, two-player coordinator in ante lockstep, MLB `banned_keys` into `RunState`, The Order hook in `generate.Keys`), then Red/Checkered/Plasma at White stake, then Phase 3 infra (checkpoints, eval harness, N-agent same-seed runner + N×N matrix). All Phase 1 agent IDs are spent; spawn fresh.

---

## Phase 2 — MLB rules + decks — KICKED OFF 2026-08-21 (evening)

**Lead brief: `docs/PHASE2_BRIEF_2026-08.md`.** Key pre-launch findings from reading the installed
BalatroMultiplayer mod (v0.5.2, `AppData/Roaming/Balatro/Mods/Multiplayer/`, Lua + lovely patches):

- **The "regular-blind life rule" is settled from source:** `death_on_round_loss = true` is the lobby
  default and MLB forces lobby options → lose a life on ANY blind lost. Tagg's practice-lobby check is
  no longer blocking.
- **Lost-PvP payout (SWEEP §4 item 7) is settled:** Nemesis `dollars = 5`, paid win OR lose
  (`lovely/game.toml` patches `or MP.is_pvp_boss()`); **no unused-hand money at a PvP blind**; a failed
  regular blind ALSO pays its reward (`blind.chips = -1` trick) and the run proceeds.
- **Comeback money (MLB):** `$4 × cumulative lives lost`, paid once at the next Cash Out after each life loss.
- **MLB voucher generation is NOT vanilla:** `compatibility/TheOrder.lua:483,510` route vouchers through
  `pseudoseed("Voucher0")` + culled pool whenever `is_major_league_ruleset()`, even with The Order off.
- **Endless:** `win_ante = 999` — match ends only at 0 lives.
- Attrition bans: jokers Mr Bones/Luchador/Matador/Chicot; vouchers Hieroglyph/Petroglyph/Director's Cut/
  Retcon; `tag_boss`; `bl_wall`, `bl_final_vessel`.
- Tags were already fully wired in Phase 1 (`_GameTagContext`, 66 tests) → plan item "TAGS" collapses to
  banning `tag_boss`.

Fleet (disjoint ownership; `game.py` shared via targeted Edit only):

| Agent | Workstream | Deliverable |
|---|---|---|
| W1 | MLB match rules + lockstep coordinator | `mlb_match.py`, env_mp rework, nemesis blind in game.py, `MLB_NOTES.md` |
| W2 | The Order hook + MLB voucher path (oracle-verified in LuaJIT) | `generate.py` key_scope, `test_the_order.py`, `NOTES_ORDER.md` |
| W3 | Decks Red/Checkered/Plasma (+others if trivial) + stake catalogue | `decks.py`, `stakes.py`, Plasma scoring, `DECKS_NOTES.md` |
| W4 | Exit gate (after W1+W2) | `scripts/mlb_match_demo.py`, `test_mlb_match_gate.py` |

Log entries follow.

### 2026-08-21 — W3 decks + stakes — DONE ✅

**Gates (agent-run, with W1/W2 editing concurrently):** `pytest mp/engine/tests` **1584 / 10 skip / 3 xfail / 0 fail**
(1441 at kickoff, +106 `test_decks.py`, +37+3xfail `test_stakes.py`, 1 assertion updated); `pytest mp/tests`
**393 / 2 xfail / 0**; `engine_parity --antes 1-8 --rerolls 5` **126/126 EXACT** (Red/White untouched).
Notes: `engine/DECKS_NOTES.md`.

- **New `decks.py`** (15-deck catalogue from `pools.BACKS` + engine-side `Back:apply_to_run` hook, Anaglyph
  `eval` hook, Checkered `creation_order`) and **`stakes.py`** (8-stake catalogue + cumulative modifier table,
  engine-side `game.lua:2050-2057`).  `BalatroGame(seed, deck_key=..., stake=1..8|'stake_*')`; `stake` flows
  into `RunState.for_stake` + `blind_scaling` / `no_small_blind_reward` / discards.
- **All 15 decks done, none xfail, no `generate.py` change needed.**  Red / Checkered / Plasma fully tested;
  Blue / Yellow / Green / Black / Magic / Nebula / Ghost / Abandoned / Zodiac / Painted / Anaglyph / Erratic
  each observable through `step()`.
- **Plasma:** `score_hand(plasma=True)` = `final_scoring_step` (state_events.lua:946, back.lua:121-128): after
  every joker, `tot = chips + mult; chips = mult = floor(tot/2)`; score `floor(chips*mult)` = a perfect square.
  Blind targets ×2 via `ante_scaling` in `_prepare_next_blind` (composes with boss mults + stake scaling),
  tested at every ante 0-12.  Dry-run scorer (`HypotheticalScorer`) passes the flag too.
- **Found + fixed (Red baseline):** the engine gave the Red Deck **3 discards; the game gives 4**
  (`starting_params.discards 3 + config {discards=1}`).  Every Red/White run now starts with 4 discards.
- **Found + fixed (Checkered):** the creation (`sort_id`) order was rebuilt by sorting the post-swap keys
  → wrong `sort_id` for 39/52 cards → every shuffle would deal the wrong hand.  Now
  `S(ex-C)×13, H(ex-D)×13, H×13, S×13`; independent check: Checkered's first hand == Red's with C→S, D→H.
- **Stakes:** White verified byte-identical (`state_signature()` trajectory, 150 steps, default ≡ `stake=1` ≡
  `'stake_white'`); Red (Small pays $0), Green/Purple (scaling tables 2/3 in `constants.get_blind_amount`),
  Blue (−1 discard) implemented + tested; Black/Orange/Gold = generation flags only, sticker EFFECTS xfail.
- **Oracle-by-proxy for Checkered/Plasma:** neither changes a generation call → shops/bosses/tags equal Red's
  on the same seed: 12 corpus seeds through ante 4, 0 mismatches (3 seeds in the suite).
- **Tagg spot-check (DECKS_NOTES §7):** `7I4M53DL` Plasma → blinds 600/900/1200, play 7♠ alone = **36**,
  5♣ alone = **25**, 2♥2♣ = **64**; Checkered first hand **A♠ K♥ J♠ 7♠ 5♠ 3♥ 2♠ 2♥**; Red shows 4 discards.
- **For W1:** `_end_round` hand row is now `hands_left * money_per_hand` + a separate `money_per_discard` row
  (Green Deck) — wrap the MLB no-unused-hand-money guard around the hands row; check the mod for the discard row.
- **Found, not fixed:** The Flint still halves the final score (game halves BASE chips/mult at `modify_hand`,
  blind.lua:512-515) — matters more under Plasma; Checkered ante-1 idol/castle pre-swap suit (unobservable).

### 2026-08-21 — W2 The Order + MLB vouchers — DONE ✅

Gates at hand-off (repo root): `pytest mp/tests` **544 / 2 xfail / 0** (393 + 151 new
`tests/test_the_order.py`); `pytest mp/engine/tests` **1609 / 10 skip / 3 xfail / 0**;
`engine_parity --antes 1-8 --rerolls 5` **126/126 EXACT**; `parity_check --antes 1-8 --variant faithful`
**126/126** (vanilla path byte-identical; no fixture regenerated).  Notes: `rng/NOTES_ORDER.md`.

- **Fields for the engine:** `RunState.key_scope` (`"ante"` default = vanilla/MLB, `"run"` = The Order) and
  `RunState.ruleset` (`"vanilla"` | `"mlb"`), set before any draw; W1's `_init_game_vars` already wires both
  (`BalatroGame(seed, ruleset="mlb")` → ante-1 voucher from `Voucher0`).  The Order additionally needs
  `rs.blind_key` / `rs.blind_type` (nr/cashout keys) — documented, not wired (MLB forces The Order off).
- **MLB voucher path** (`generate.next_voucher` → `_next_voucher_culled`, `get_culled`): shop voucher AND
  Voucher Tag from the paired-culled pool on the run-global `Voucher0` stream, redraw = same stream,
  `Voucher0<it>` fallback after 1000 (TheOrder.lua:481-525).  Everything else under MLB is vanilla
  (`test_mlb_differs_from_vanilla_only_in_vouchers`).
- **The Order, as the mod really does it (brief §1.7 was incomplete):** (1) the `*` seed prefix is applied
  BEFORE `hashed_seed` → every stream = vanilla stream of `'*'..seed` (not display-only); (2) `create_card`
  is wrapped with ante := 0 and `key_append` rewritten (Tarot/Planet/Spectral → `TarotTarot0` /
  `TarotTarot_pack0`; jokers → no append: `rarity0`, `Joker<r>0`, `ediJoker<r>0`); (3) jokers come from the
  mod's own loop (`_etper`/`_rent` polls, `Joker<r>0_sticker` queue), resamples re-step the pool stream
  (`_resample` only after 1000), `boss<a>`, `cdt0`, `shop_pack0`, `halu0`, `stdset0/standard_edition0/
  stdseal0/stdsealtype0`; (4) `nr`/`cashout` keys become `nr<a><blind_key><Small|Big|Boss>` and ALL
  playing-card shuffles / Card-list picks use the mod's value-ranking (`give_shufflevals`, ported incl.
  LuaJIT's unstable `table.sort`); idol/mail replaced (mail's count sort is a no-op bug — ported as is);
  To Do / Orbital use the hand `order` list.  Tags, `anc`, and every other key unchanged.
- **Verified:** a second Lua oracle = vanilla reference text + the mod's toml patches applied in Python +
  the mod's `TheOrder.lua` executed VERBATIM (read from `%APPDATA%` at test time, nothing copied), 22 seeds ×
  antes 1-8 × 3 modes (order / mlb / vanilla-with-mod-loaded), shops, rerolls, packs, bosses, tags, vouchers,
  Voucher Tag, shuffles, round picks, creation paths, joker picks, Gold stake, Showman, bans; 0 mismatches;
  perturbation check live.  Stubbed (documented): SMODS poll_seal/create_card/get_next_vouchers/size_of_pool,
  Rank/Suit order, the `should_use_the_order`/`is_major_league_ruleset` flags.
- **Found, not fixed:** Steamodded may change the `pairs(G.GAME.hands)` order (Orbital/To Do under MLB);
  `reset_blinds` still draws `boss` before the Nemesis overwrite (W1 must keep drawing it); engine
  `_round_end_resets`/`round_cards.py`/`misc.py` halu need the new `reset_round_picks` / `halu_for` only
  under The Order; identity among identical-kind cards after an Order shuffle follows the deck-list order.

### 2026-08-21 — W1 MLB match — DONE ✅

**Gates (final, same tree as W2 + W3):** `pytest mp/engine/tests` **1609 passed / 10 skipped / 3 xfailed / 0 failed**;
`pytest mp/tests` **544 passed / 2 xfailed / 0 failed**; `engine_parity --antes 1-8 --rerolls 5` **126/126 EXACT**
(vanilla byte-identical: `ruleset="vanilla"` is the default and every MLB field is inert). Notes: `engine/MLB_NOTES.md`.

**Landed.** `BalatroGame(ruleset="mlb")` = the mod client: Attrition bans → `run_state.banned_keys` before the first
draw (in-place resample invariant holds; bosses leave the eligible set as in vanilla `get_new_boss`), `run_state.ruleset`
→ W2's `Voucher0` path, Nemesis (`bl_mp_nemesis`, `BlindInfo.is_pvp`, $5, mult 1, no effect, target = opponent's live
score) at the Boss slot from ante 2 while the `'boss'` stream is still drawn behind it (mod's `reset_blinds` runs
vanilla first), `lives` / `comeback_bonus` / `comeback_bonus_given` / once-per-round life blocker, **failed regular
blind proceeds + pays its reward + costs a life** (0 lives → GAME_OVER without Cash Out), **no unused-hand money at a PvP
blind** (Green Deck discard row still paid — only the hands row is patched), comeback `4 × cumulative lives lost` paid
once right after interest, endless (no ante-8 win), new `State.PVP_WAIT`, deck-out edge (`handle_deck_out`), and the
`lose_life` / `set_pvp_info` / `end_pvp` "network message" entry points. `MLBMatch` (`mlb_match.py`) = the server: two
games on ONE seed, ante lockstep (play = readyBlind at the Nemesis, both start on startBlind, a ready player cannot act;
everything else independent), enemyInfo relay, **server PvP rule confirmed from the GitHub server repo**
(`actionHandlers.ts playHandAction`): ends when a player is out of hands AND strictly behind, or both are out; strictly
lower score loses one life; **exact tie loses nobody** (the ghost replay's `>=` is NOT the live rule); winner's remaining
hands forfeited; 0 lives → match over. `current_player()` = deterministic alternation (outcome is interleaving-invariant,
pinned by test). `clone()` composes with `BalatroGame.clone()` (126 µs/match). `env_mp.py` rewritten on `MLBMatch`:
V7 obs + 10 MP features (both lives, opp score/hands/lead/exhausted during PvP, waiting, comeback pending, opp location)
→ **MP OBS_DIM 451 → 457**; MP event rewards separated in `info["p{1,2}_mp_reward"]`; no more process-global ban
pollution at import. `mp_game.py` retired to an ImportError shim; V8-era `test_mp_game.py` / `test_mp_integration.py`
deleted; `test_mlb_match.py` (59) + `test_env_mp.py` (30) added.

**Assumptions:** tie = server rule (no life); comeback paid at the same Cash Out as the loss (the playerInfo reply races
evaluate_round — Tagg's practice-lobby glance still useful); first-to-0 loses when both fail the same blind (step order);
exhausted hand stays in hand until Cash Out (same money); Rocket/Campfire/Investment/Anaglyph fire on a LOST Nemesis
(vanilla end_round runs unconditionally in MP); ante ≥ 2 shadow boss may differ from vanilla because `bl_wall` leaves
the candidate list (real-game behaviour; stream position pinned equal).

**Found & fixed out of scope (flagged):** `clone()` kept the ORIGINAL card ids in `_played_this_ante` / `_forced_card_id`
(Card.copy mints fresh ids) → clones silently lost The Pillar / Cerulean Bell and `state_signature()` differed; now
remapped. `sim_tests/test_clone_deck_identity.py` asserted the raw ids (edited to compare cards); `sim_tests/test_sweep.py
::TestEnvRevival` used the removed `_revive_boss_if_needed` (ported to the native MLB lost-boss path).
**Found, not fixed:** vanilla wedges (no legal actions, no end_round) when hand AND deck are empty mid-blind.
**Gaps:** timers / pvp_timer / speedrun, disconnects, `hide_score_until_played` branch, ghost `>=` tie as an option.

**For W4:** `MLBMatch(seed).play_out([pol0, pol1])` or the `current_player()/legal_actions()/step()` loop; `m.pvp_log`,
`m.signature()`, `game.lives`, `game.comeback_bonus`; queue-alignment diff = compare `run_state.rng.snapshot()["state"]`
key positions between the two games (MLB_NOTES §7).

### 2026-08-21 — W4 Phase 2 exit gate — PASSED ✅ (0 engine bugs; 2 design findings)

**Gates (final run, repo root):**

| gate | result |
|---|---|
| `python -m pytest mp/engine/tests -q` | **1609 passed / 10 skipped / 3 xfailed / 0 failed** (unchanged) |
| `python -m pytest mp/tests -q` | **1073 passed / 2 xfailed / 0 failed** (544 + 507 `test_mlb_match_gate.py` + 22 new invariants) |
| `python -m mp.oracle.engine_parity --antes 1-8 --rerolls 5 --quiet` | **126/126 EXACT through ante 8** |
| `python -m mp.oracle.parity_check --antes 1-8 --variant faithful` | **126/126 EXACT through ante 8** |
| `python mp/scripts/mlb_match_demo.py --seed 7I4M53DL` | full trace; P2 (reroller) wins at ante 2, 117 steps |
| `python -m pytest mp/tests/test_mlb_match_gate.py -q -rx` | **507 passed / 0 xfail / 0 failed** (~27 s) |

Notes: `tests/GATE_NOTES.md` (how to run, what each item proves, the key-classification table, the voucher-stream finding).
Deliverables: `scripts/mlb_match_demo.py` (CLI `--seed --deck --stake --lives --max-antes --quiet --alignment --json`; two
deliberately different scripted players; importable `MatchRecorder` / `ScriptedPlayer` / `key_position` / `classify_key` /
`diff_rng`), `tests/test_mlb_match_gate.py`, `tests/test_engine_invariants.py` §6. Nothing under `engine/` or `rng/` touched.

**Exit-gate items, 10 ground-truth seeds × Red/Checkered/Plasma (White), every match driven through `MLBMatch.step()`:**
1. **Lives — PASS.** Every lost blind (regular + PvP) = exactly one life of exactly that player at that step; won blinds and
   ties cost nobody; comeback `4 × cumulative` lands at that player's next Cash Out exactly once per loss (`[4, 8, 12]`);
   match ends at 0 lives (final step = loser's 4th loss, winner `match_won`), never at ante 8 — endless scenario plays
   antes 9-12 (Small 110k / 560k / 7.2M / 300M, ×2 Plasma) and ends at 12; early end fires (loser exhausts strictly
   behind at the step of its last hand, winner's remaining hands forfeited) — its observable is interleaving-dependent as
   W1 §3.2 says, both outcomes asserted.
2. **Money — PASS.** Nemesis: $5 win or lose, 0 unused-hand money even when the early-ended winner has hands left, interest
   `min(pre//5, 5)`; failed regular blind pays 3/4/5 + interest + comeback, 0 hand money. Cash Out delta exact in the
   joker-free scenarios.
3. **Queue alignment — PASS.** Per-key QUEUE POSITIONS (state value → exact call count via the LCG chain) diffed at entry and
   exit of every shop-visit ordinal; every differing key classified by name: SHARED (boss/Tag/shuffle/nr/cashout/idol/
   mail/anc/cas/orbital/shop_pack → must be equal, asserted present), VOUCHER, OWN_SHOP (`cdt<a>` == slots drawn,
   asserted absolutely for both players at every visit; others monotone in slots), OWN_PACK (0 without a pack opened),
   OWN_ANY (`soul_*`), OWN_RESAMPLE, PER_PLAYER (named effect keys), UNKNOWN → fail. 30 matches: OWN_SHOP 21 313 / OWN_PACK
   17 585 / OWN_RESAMPLE 8 427 / OWN_ANY 3 496 / PER_PLAYER 345 / SHARED+VOUCHER+UNKNOWN **0**; teeth test perturbs
   boss/cdt/Voucher0. First shelf of every ante, packs, voucher, shadow boss, tags identical for both players modulo
   in-place own-collection slots (21/342).
4. **Vanilla unchanged — PASS.** engine_parity 126/126 in-process; all 126 seeds vanilla-vs-MLB solo (8 antes, packs, 5
   rerolls): differs only in voucher, banned-item slots (+ their resample-offset second-order slots), `tag_boss` slots, the
   ante ≥ 2 Boss slot (Nemesis; shadow boss from the `bl_wall`-less list, `boss` position equal, ante-1 boss equal).
5. **Clone — PASS.** Mid-Nemesis `clone()` + fresh policies: `signature()` equal after every step to the end, same winner /
   `pvp_log`; clone isolated.

**Findings (design, not bugs):**
- **Shared voucher stream.** Buying never steps `Voucher0`: both players stay at the same position and are offered the same
  voucher PAIR each ante (buyer sees the upgrade where it owns the base). The stream **diverges once a player owns both tiers
  of a pair** (pair → UNAVAILABLE → that player redraws on the same stream): `ALEEB` ante 5 (Clearance Sale + Liquidation),
  `7I4M53DL` ante 11; from then on the players see different vouchers for the rest of the run. Banned pairs redraw
  symmetrically. Faithful to the mod (W2 oracle); Phase 3 must not assume "same seed ⇒ same voucher" after a completed pair.
- **Second-order resample offset.** A ban/ownership redraw advances `<pool>_resample<it>`, so a later unrelated redraw in the
  same ante/area/pool (locked joker, hidden-hand planet) yields a different card (8 shop slots / 126×8 antes; e.g.
  `1MD1YZ9T` ante 7 slot 10 `j_seance → j_sixth_sense`). Game-faithful; the comparison rule allows exactly this.
- Not exercised: skips/tags under MLB (Voucher Tag steps `Voucher0` per player), sells, consumables, deck-out at a Nemesis,
  both failing at 1 life, stakes ≥ 2 in money assertions. `env_v7` reward `R_BLIND_BASE*(9-ante)` goes negative past ante 8.

**Biggest remaining risk for Phase 3:** the PvP decision frame is interleaving-dependent in everything but the verdict
(cut point of the winner's remaining hands, hand/deck contents at Cash Out, discards left) and the canonical alternation is
an engine convention, not the real game's real-time order — an env that steps both players per tick, a replay of a real
match, or self-play that learns to exploit "P1 moves first" will disagree with the live mod at exactly the moment the
policy is being evaluated. Pin the interleaving contract (and Tagg's `7I4M53DL` live check) before training on it.

---

## ✅ PHASE 2 COMPLETE — 2026-08-21 — MLB rules + The Order + decks/stakes

**Final gates, independently re-run by lead from a clean state (`-p no:cacheprovider`):**

| Gate | Result |
|---|---|
| `pytest mp/engine/tests` | **1609 passed / 10 skipped / 3 xfailed / 0 failed** (1441 at kickoff) |
| `pytest mp/tests` | **1073 passed / 2 xfailed / 0 failed** (393 at kickoff; +151 The Order oracle, +507 MLB gate, +22 invariants) |
| `engine_parity --antes 1-8 --rerolls 5` | **126/126 EXACT through ante 8** |
| `parity_check --antes 1-8 --variant faithful` | **126/126 EXACT through ante 8** |
| `scripts/mlb_match_demo.py --seed 7I4M53DL` | full match, P2 wins ante 2, 117 steps, 7 life losses logged with causes |
| `git status` | only `?? mp/` — nothing committed, nothing outside `mp/` touched |

**Delivered:** `mlb_match.py` (two-player ante-lockstep coordinator, clone 126 µs), nemesis blind + lives +
comeback + PvP money + endless in `game.py` under `ruleset="mlb"` (vanilla byte-identical), `env_mp.py`
rewritten (MP OBS_DIM 451→457), `mp_game.py` retired; `RunState.key_scope`/`ruleset` + The Order key-site
hook + MLB `Voucher0` voucher path, verified by executing the mod's patched Lua in LuaJIT (22 seeds × 8 antes × 3
modes, 0 mismatches); all 15 decks + 8-stake catalogue (`decks.py`, `stakes.py`), Plasma scoring + ×2 blinds,
Red/Green/Blue/Purple stakes implemented, Black/Orange/Gold sticker effects xfail; exit-gate harness
(`test_mlb_match_gate.py`, `GATE_NOTES.md`) — queue alignment proven by per-key RNG position classification over
30 matches: 0 unexplained diffs.

**Rules settled from the mod source / server repo (were "unknown" after Phase 1):** life on ANY lost blind;
Nemesis $5 win or lose; no unused-hand money at PvP (Green discard row still paid); failed regular blind pays +
proceeds; comeback $4 × cumulative lives lost, once per loss at next Cash Out; endless; **exact PvP tie = nobody
loses a life** (server-confirmed; ghost replay's `>=` is not live); MLB vouchers on The Order's `Voucher0` path.

**Real bugs found and fixed along the way:** Red Deck had 3 discards (game: 4) — every Phase 0/1 run and all
V7/V8 history had one discard too few; Checkered creation order gave wrong `sort_id` for 39/52 cards;
`clone()` kept original card ids in `_played_this_ante`/`_forced_card_id` (clones silently lost The Pillar /
Cerulean Bell).

**Design findings (mod-faithful, not bugs):** `Voucher0` stream diverges between players once one owns both tiers
of a voucher pair (ALEEB ante 5); ban/ownership redraws advance `<pool>_resample<it>` so later unrelated redraws
in the same ante/pool differ.

**Honest gaps:** Black/Orange/Gold stickers (Eternal/Perishable/Rental) not implemented; The Flint halves final
score not base chips/mult (pre-existing; worse under Plasma); engine-side Card-list picks under The Order ONLY
(Hook/Cerulean/Crimson/Invisible/Madness/Hallucination) still use vanilla keys — irrelevant to MLB; vanilla
wedge when hand AND deck empty mid-blind; `env_v7` `(9 - ante)` reward term goes negative past ante 8;
PvP interleaving is a canonical alternation, not the live real-time order (verdict is interleaving-invariant,
cut point is not) — **pin this contract before training on it.** Unexercised by the gate: skips/tags under MLB,
sells, consumables, deck-out at a Nemesis, both players failing at 1 life.

**Still Tagg's:** Plasma/Checkered/Red 5-minute spot-check script in `engine/DECKS_NOTES.md` §7 (seed
`7I4M53DL`: Plasma blinds 600/900/1200, 7♠ alone → 36; Checkered first hand A♠ K♥ J♠ 7♠ 5♠ 3♥ 2♠ 2♥; Red shows 4
discards); `7I4M53DL` Lovers/Hierophant check from Phase 0.

**NEXT = PHASE 3 infra** (`docs/MP_CAMPAIGN_PLAN_2026-08.md`): checkpoint saving, eval harness, N-agent
same-seed runner + full N×N matrix at each Nemesis, batched inference, ρ-decay harness. Phase 2 agent IDs are spent.

---

### 2026-08-21 — 🧑 Tagg LIVE CHECKS — ALL CONFIRMED ✅ (closes the last Phase 0 loop + Phase 2 deck spot-checks)

Real game, seed `7I4M53DL`, Red/White:
- Ante-1 shop exactly as predicted: shelf Banner + The Hierophant, packs Buffoon + Arcana; boss The Hook,
  voucher Wasteful, tags Speed/Economy, Buffoon = Acrobat + Wily Joker. Game version matches 1.0.1o extraction.
- **Arcana pack 3rd card = The Lovers → the "faithful" `used_jokers` rule is CONFIRMED against the running
  game.** Every published analyzer (Immolate / TheSoul / Blueprint as shipped) is wrong on this; our corpus
  variant stands. No flag flip.
- **Bonus check Tagg ran:** buy The Hierophant, sell it, then open the Arcana → **The Hierophant appears as the
  3rd card.** That independently confirms the second clause of the rule (`Card:remove` clears the `used_jokers`
  mark), not just the blocking half.
- Phase 2 deck spot-checks (DECKS_NOTES §7) also confirmed correct in-game: Plasma blinds 600/900/1200 and
  7♠ → 36; Checkered first hand; Red Deck shows 4 discards.

**Nothing in the engine needs to change. No human-verification items remain open for Phases 0-2.**

---

## Phase 3 — Infrastructure — KICKED OFF 2026-08-21 (late evening)

**Lead brief: `docs/PHASE3_BRIEF_2026-08.md`.** Decisions: MCTS agent layer FORKED from `balatro-mcts` (`ee75d11`,
read-only) into `mp/agent/`; N-agent interleaving contract = play Nemesis to exhaustion, server tie rule; lives
pluggable (`paired` default / `median` / `none`); `mp/engine/**` + `mp/rng/**` FROZEN (mtime snapshot in
`docs/phase3_frozen_snapshot.txt`); model split W1/W3 strong, W2/W4 Sonnet.

| Agent | Workstream | Model | Deliverable |
|---|---|---|---|
| W1 | Agent-layer fork + sync + checkpointing | strong | `mp/agent/**`, `AGENT_NOTES.md` |
| W2 | N-agent same-seed runner + N×N matrix | sonnet | `mp/tournament/**`, `TOURNAMENT_NOTES.md` |
| W3 | Batched inference + tree reuse (after W1) | strong | `mp/agent/mcts/batched.py`, `BATCH_NOTES.md` |
| W4 | Eval harness + ρ-decay harness + first ρ(h) numbers | sonnet | `mp/eval/**`, `mp/results/`, `EVAL_NOTES.md` |

Log entries follow.

---

### 2026-08-21 — P3-W2 tournament runner + N×N matrix — DONE ✅

`mp/tournament/` (new package): `bootstrap.py`, `players.py`, `runner.py`, `matrix.py`,
`cli.py`, `conftest.py`, `tests/` (31 tests), `TOURNAMENT_NOTES.md`. No engine/rng edits.

**Gates:** `pytest mp/tournament/tests -q` 31/0 (~49s); `pytest mp/engine/tests -q`
1609/10/3/0 (unchanged); `pytest mp/tests -q` 1073/2/0 (unchanged);
`python -m mp.tournament.cli --seed 7I4M53DL --n 100 --life-rule none --max-ante 8` runs,
prints per-ante summary + wall clock (22.7s, 24483 steps).

**Architecture:** each agent gets its OWN `BalatroGame(ruleset="mlb")`; with the engine's
default `pvp_solo=True` a Nemesis auto-resolves solo (no life lost) the instant hands run
out — so N agents drive completely independently with NO `MLBMatch` object at all; the only
sync point is after every alive agent finishes an ante's Nemesis, where the N×N matrix is
built from final scores and `life_rule` (`paired`/`median`/`none`) decides who loses a life.
Fan-out: `clone()` beat N constructions ~3.9× (0.0073s vs 0.0284s for n=100, identical
`state_signature()`s) and is the default.

**100-agent numbers (seed `7I4M53DL`, heterogeneous scripted+random population):**
`--life-rule none --max-ante 8` → all 100 present every ante, 22.7s; `--life-rule paired
--max-ante 40` → last-agent-standing (100→90→41→0) in 2 antes, 6.4s. Degeneracy metric
(n=12): 12 identical policies → tie_fraction = 1.0000 at every ante (exact degeneracy, as
predicted by the design doc); the heterogeneous default population → 0.06-0.11.

**Found, not fixed (documented as "needs engine change" in TOURNAMENT_NOTES.md §5):**
`game.py:1551,1579,1591` (`bl_hook`/`bl_eye`/`bl_mouth` boss-ability "hand rejected"
branches) hard-set `GAME_OVER` on hand exhaustion without checking `self.mlb`, unlike every
other exhaustion path in the file — bypasses `_mlb_fail_round`'s life/comeback bookkeeping.
Only reachable at ante 1's vanilla Boss (~1/100 agents in the smoke). Worked around in the
runner (`_repair_mlb_gameover_bug`, detects the only situation that can never legitimately
occur under MLB: `GAME_OVER` with `lives > 0`), never edits engine files.

**MCTS player:** `players.MCTSPlayer` is a clearly-marked `NotImplementedError` placeholder;
`mp.agent` was never imported. See TOURNAMENT_NOTES.md §8 for the exact plug-in contract.

### 2026-08-21 — P3-W1 agent fork + checkpoints — DONE ✅

**Gates (repo root, `-p no:cacheprovider`):** `pytest mp/agent/tests` **85 passed / 0 failed**;
`pytest mp/engine/tests` **1609 / 10 skip / 3 xfail / 0** (unchanged); `pytest mp/tests` **1073 / 2 xfail / 0**
(unchanged); all 41 files in `docs/phase3_frozen_snapshot.txt` byte-size identical (mtimes within the
snapshot's 1 s rounding) — `mp/engine/**` and `mp/rng/**` untouched. Notes: `agent/AGENT_NOTES.md`.

**Landed.** `mp/agent/` = `balatro-mcts` @ `ee75d11` (read-only source, still clean) re-targeted onto the
frozen fork engine: `mcts/` (+`outcome.py`, +`player.py`), `train/` (+`loop.py`, +`checkpoint.py`),
`scripts/`, `benchmarks/bench_search.py`, `tests/`, `pytest.ini` + `conftest.py` (fork guard: `balatro_sim`
must be `mp/engine`'s, `mcts`/`train` must be `mp/agent`'s, else `RuntimeError`). `agent/runs/` appended to
`mp/.gitignore`.

**Re-target.** *Obs*: the copied 434-dim encoder is gone — `mcts/encoder.py` now CALLS
`env_v7.BalatroV7Env._encode_obs` (it reads only `self.game`, so it is bound to a one-slot shim), pinned by a
test that builds a real `BalatroV7Env` and asserts byte equality → **447**. Opt-in `MLBEncoder` adds 6 MLB
features (lives, is_pvp, opp hands, comeback pending, pvp/waiting, log-ante — V7's `ante/8` saturates exactly
where endless MLB gets interesting) → 453; the checkpoint records which encoder a run used and refuses to
resume into the other. *Action features*: 12→**13** types (the fork emits `reroll_boss`, which the original
would have featurized as the all-zero unknown row) and every V7-era slot bound widened + an overflow scalar,
because `hand_size`/`consumable_slots`/`joker_slots` all grow past them in real play and out-of-range indices
used to be **silently dropped** — two different actions could share one feature row and one prior; 44→**56**.
A walk over 3 seeds × both rulesets asserts every emitted type is in the vocabulary, key→dict→`step()`
round-trips on a clone, and keys AND feature rows are unique per legal-action list. *Keys*: nothing in the
agent hardcodes one; the stale 15-key `BOSS_TYPES` died with the copied encoder. *Warm start*: removed, not
left inert (V7 checkpoint lost 2026-06-10 and 447 ≠ 434); trunk stays V7-shaped, `describe()` puts the dims in
the checkpoint.

**Outcome is a parameter** (`mcts/outcome.py`) — the brief's `_is_win`/`_shaped_z` item. `VanillaOutcome`
(ante-8 win, /24 progress) | `MLBOutcome` (**win = `match_won`, endless**, `0.5·lives + 0.5·progress`) |
`ExternalOutcome` (W2's N×N margin, W4's paired margin, `MLBMatch`'s verdict; `.from_margin` logistic, margin
0 → 0.5 = the server's no-life tie). `default_outcome_for(game)` switches on `game.mlb`.

**MLB awareness.** `SelfPlayAgent(pvp_target_fn=…)` relays `set_pvp_info` every decision at a PvP blind
(`startBlind` resets the target). Both no-action states are handled: `run`/`run_gumbel` on a `PVP_WAIT` or
readied root return `{}`/`chosen=None` instead of the original's `RuntimeError`, an interior stuck node stops
descent with `stop_reason="stuck"` and takes the outcome's pending value, `MCTSPlayer.act()` → `None`, and
`play_episode` stops with `stop_reason="stuck"` so `resume_episode()` continues the SAME trajectory after the
driver's `end_pvp()` — one z per episode. `MCTSPlayer` is the `act(game) -> action|None` shape W2's
`players.py` expects (W2/W4 still do not import `mp/agent`).

**Checkpointing — round trip is BIT-EXACT on CPU.** `torch.save` of model + Adam + counters + RNG (numpy
Generator, torch CPU/CUDA, python `random`) + config + replay buffer; atomic temp+`os.replace`;
`weights_only=False` (torch 2.6 flipped the default) with a kind/version check; `--resume` refuses a config
that changes the experiment. `train 3 → save → reload → train 1` equals `train 4` under `torch.equal` on every
parameter AND every Adam moment, same counters/buffer, and the resumed 4th episode has the same game seed /
length / ante — because ONE seeded numpy Generator drives episode seed + Gumbel noise + batch indices, the
engine is deterministic, and the buffer is carried. Same test passes for `ruleset="mlb"` + 453-dim encoder.
Not bit-exact (and flagged at runtime): CUDA, `--no-checkpoint-buffer`, a truncated buffer.

**Numbers (RTX 3080 Ti, py 3.13.5, torch 2.6.0+cu124; demo state = 436 legal actions).** `mcts_demo` @500
sims: uniform puct/gumbel **1010 / 1275**, nn-cpu **468 / 495** sims/s. `bench_search` split (`_evaluate_leaf`
/ clone+step / python): nn-cpu puct 451 s/s at 39/47/14%. **nn-CUDA is SLOWER than CPU** (362 / 328 s/s) —
per-leaf launch + transfer on a 2.4 M-param net; that is W3's whole case, and their baseline is the CPU
number. At the MLB Nemesis nn-cpu drops to ~310 s/s with **78%** in leaf eval (a PvP round never ends on
`chips ≥ target`, so descent runs deeper). Running the untouched source repo on this same box gives nn-cpu
475 s/s, so the fork costs ~nothing; the brief's historical "~745 with NN" does not reproduce here even with
the original code. `train_cold --minutes 2 --device cuda`: **153 episodes, 0 errors, 7 checkpoints**,
75.9 ep/min; `--resume latest.pt --minutes 1` picked up at **episode 154 → 232**, buffer 1472 → 2207, losses
continuing across the boundary.

**Found, not fixed.** (a) **A `Sample` is ~97 KB** — `action_features` is (436, 56) float32 vs 1.8 KB of obs.
The inherited `buffer_capacity=200_000` would be ~19 GB of RAM and an unshippable checkpoint. Mitigated
(capacity default → 20 000, `--buffer-checkpoint-cap` 5 000, numbered checkpoints weights-only 28.9 MB while
only `latest.pt` carries the buffer); the real fix is to subsample the action set per training sample (~20×),
which is a training-design decision, not W1's. (b) **needs engine change: `mp/engine/balatro_sim/env_v7.py`
:702 — the obs encodes only 8 hand slots / 5 joker slots**, but `hand_size` and `joker_slots` both grow past
those in real play, so a 9th card in hand is invisible to the net while being a legal `play` target (the
*action* features handle it). Widening changes `OBS_DIM`, so do it before a long training run, not after.
(c) `env_v7._finish_step`'s `R_BLIND_BASE*(9-ante)` still goes negative past ante 8 (already known); the agent
uses `OutcomeFn`, not that reward.

**For W3 (`agent/AGENT_NOTES.md` §4).** Interface = `PolicyValueFn` with `evaluate_many(games)`;
`PolicyValueBase` gives every existing policy a serial default, `NNPolicy` is the reference and exposes
`encode_leaf()` (pure numpy) + `priors_from_logits()` to reuse, and `PolicyValueNet.score_actions_flat(trunk,
feats, counts)` already scores a RAGGED batch in one policy-head call (pinned equal to the per-state path).
Leaf evaluation happens in exactly one place: `MCTS._evaluate_leaf`, called only from `_expand`. Tree reuse:
compare `game.state_signature()` after the action; discount is 1.0 and values are in [0, 1], so nothing
rescales. Baseline table in §4.4.

---

### 2026-08-21 — P3-W4 eval + ρ-decay — DONE ✅

**Agent W4.** `mp/eval/` (new): `common.py` (fork-guarded bootstrap; `Player` protocol + `scripted:`/`checkpoint:`
spec parsing; `SoloShim` drives one `BalatroGame` through `mlb_match_demo`'s policy signature; `play_sp_vanilla`
/ `play_sp_mlb` / `play_1v1` drivers; `play_arm_to_horizons` paired-arm driver; target functions; pure-python
bootstrap/Pearson/Spearman stats; `sample_size_per_arm`), `eval_harness.py` (CLI, 3 modes, `--compare` paired
diff+CI), `rho_decay.py` (CLI, 3 registered perturbations + `measure_rho`), `conftest.py`, `tests/` (49 tests).
`mp/results/` (new): `demo_1v1.json`, `rho_decay_{buy_slot0,reroll_once,skip_small}.json`. Nothing under
`mp/engine/` or `mp/rng/` touched.

**Gates:** `pytest mp/eval/tests -q` **49 passed**; `pytest mp/engine/tests -q` **1609/10 skip/3 xfail** and
`pytest mp/tests -q` **1073/2 xfail** both unchanged from Phase 2 hand-off; `eval_harness --mode 1v1` (126
seeds, 21.7s) and `rho_decay --all --n-extra-seeds 24` (150 seeds × 3 perturbations, n_boot=2000, ~100s each)
both ran clean and wrote the JSON above.

**Default SP-MLB target:** the agent's own Big-Blind chip score that same ante (k=1 "mirror Nemesis") —
calibration-free, no second checkpoint needed, ~50/50 by construction. `rho_decay` instead uses a target
decoupled from BOTH arms (`external_vanilla_big_blind_target` = the vanilla Big-Blind chip requirement for
that ante/deck/stake) so a life-loss correlation isn't an artifact of a shared target.

**ρ(h) — measured for real, N=150 seeds (126 ground-truth + 24 synthetic), n_boot=2000, log-score, Pearson
[95% CI], VRF = paired/unpaired variance-reduction factor:**

| perturbation | h=1 | h=2 | h=4 | h=8 |
|---|---|---|---|---|
| buy_slot0 (buy 1 item) | 0.876 [0.797,0.933] VRF=8.00× | 0.877 [0.803,0.931] VRF=8.16× | 0.897 [0.851,0.932] VRF=9.61× | 0.870 [0.804,0.923] VRF=7.69× |
| skip_small (skip a blind) | 0.805 [0.717,0.881] VRF=5.10× | 0.774 [0.675,0.859] VRF=4.46× | 0.772 [0.674,0.844] VRF=4.37× | 0.772 [0.674,0.852] VRF=4.40× |
| reroll_once (redraw shelf) | 0.606 [0.445,0.739] VRF=2.53× | 0.669 [0.523,0.788] VRF=3.02× | 0.715 [0.608,0.798] VRF=3.50× | 0.728 [0.615,0.813] VRF=3.69× |

Ordering (buy_slot0 > skip_small > reroll_once) holds at every horizon — the monotonicity test, confirmed on
the full run. **Versus the design doc's guess (~0.9 at h=1 falling to ~0.3-0.5 at h=4-8, VRF settling to
~1.4-2×): ρ stays HIGH and roughly FLAT through h=8 (two of three perturbations drift slightly UP), and VRF
stays in the 2.5-10× range at every horizon measured** — the guessed decay did not materialize for these
scripted players. Root cause (confirmed directly via a `"none"` no-perturbation control giving ρ=1.000 exactly
at every horizon/seed): the `shuffle` RNG stream (cards dealt) is provably independent of shop/pack/voucher
streams (Phase 1 invariant), so `greedy_hand`'s base score is identical between arms unless a joker they
disagree on owning changes the scoring — a smaller share of total score variance than card luck, for these
early antes and this policy. Caveat: this likely makes ρ measured here an UPPER bound / VRF a LOWER bound —
a stronger/trained policy that actually targets synergistic jokers would plausibly decorrelate faster.
Full writeup, money/lives_lost tables, sample-size table: `mp/eval/EVAL_NOTES.md`.

**Needs engine change (found, not fixed):** `game.py:1546-1552,1571-1580,1582-1592` — the `bl_hook`/`bl_eye`/
`bl_mouth` boss-ability-rejection branches inside `_play_hand` set `GAME_OVER` unconditionally on hand
exhaustion, bypassing the mlb-aware `_mlb_fail_round` a few lines below; under `ruleset="mlb"` a failed blind
against one of these three bosses ends the run instead of costing one life and proceeding. Found via
`rho_decay`'s synthetic seed `BP49PU2Y` (1/150 seeds affected); harnesses already tolerate it (excluded from
ρ(h), not miscounted) and `play_sp_mlb` now flags it (`ended_early_engine_gap`). `mp/engine` frozen — worked
around, not patched.

Next: W1's checkpoint loader plugs into `mp.eval.common.Player` / `parse_player_spec("checkpoint:...")`
(interface documented in code + EVAL_NOTES.md §9) to eval trained agents through the same harness.


### 2026-08-22 — P3-W3 batched inference + tree reuse — DONE ✅

Phase 3 exit-gate item 3. All in `mp/agent/` (W1's package, handed off): `mcts/batched.py` +
`mcts/reuse.py` new, `search.py` / `player.py` / `action_features.py` extended,
`benchmarks/bench_batched.py`, `tests/test_batched.py` (29) + `tests/test_reuse.py` (17),
`mp/agent/BATCH_NOTES.md`. Gates: `pytest mp/agent/tests -q` **131 passed** (85 W1 + 46 W3),
`pytest mp/engine/tests -q` **1609 / 10 skip / 3 xfail / 0** and `pytest mp/tests -q`
**1073 / 2 xfail / 0** both unchanged; **41/41 frozen files** byte-identical vs
`docs/phase3_frozen_snapshot.txt`.

**How batching was made possible without changing the search:** every search is now also a
generator (`run_iter` / `run_gumbel_iter`) that *yields a list of leaf games* and is
`send()`-ed the evaluations; `MCTS.run` / `run_gumbel` drive it one leaf at a time and are
**byte-identical to the pre-W3 implementation** — pinned by a test that keeps a verbatim
copy of W1's loops and compares the whole tree (visits, value sums, priors, stop reasons)
plus the Gumbel pick, PUCT and Gumbel, two budgets. `BatchedSearch` drives K such
generators in lockstep: K trees batched == K independent searches, exactly (`UniformPolicy`)
and to TV ≤ 0.01 with the net. Trees finishing early (spent budget, terminal root, MLB
`PVP_WAIT`) drop out; the batch shrinks. Optional within-tree leaf batching
(`MCTSConfig.leaf_batch=L`) uses virtual loss.

**Benchmarks (RTX 3080 Ti, 500 sims/decision, gumbel, 436 legal actions; this box has
±20% run-to-run noise, so component measurements back every claim):**

| config | vanilla blind | MLB Nemesis |
|---|---|---|
| serial cpu / cuda (re-run) | 563 / 531 sims/s | 404 / 427 |
| batched cuda K=8 / 32 / 100 | 682 / **761** / 729 | 453 / 443 / — |
| batched cpu K=8 / 32 / 100 | 688 / 667 / 654 | 415 / 366 / — |
| K=1 leaf_batch=16, cuda | **1034** | 569 |
| leaf-eval share of the search | 55% → **25%** | 78% → 69% |

**The headline finding is that the NN was never the wall.** The forward pass amortises
26× (1.25 ms/leaf at B=1 → 0.048 ms at B=64) and cross-tree batching still only buys
1.4× end-to-end, because what it uncovers is (a) ~0.73 ms/sim of un-batchable per-leaf
Python — `featurize_actions` 0.32 ms + `priors_from_logits` 0.12 ms + 436 `Node`
allocations 0.20 ms, all linear in the 436-action legal set — and (b) ~0.8 ms/sim of
engine `clone()`+`step()`, which is 50-66% of a batched search. At the MLB Nemesis, where
W1 measured 78% of time in leaf eval and batching should pay best, it buys ~4%: nearly
every leaf there carries ~400 actions (measured mean 397), so the leaf bucket is CPU-side
featurization, not the GPU call. **Next lever is not more batching — it is expanding
fewer than all 436 actions per leaf** (top-k by prior + a sample of the rest), which is
the same fix W1 already recommended for the 97 KB `Sample` problem and would cut leaf
cost, tree memory (26 304 nodes / 15 MB per vanilla tree; 119 494 at a Nemesis) and
sample size together. That is a training-design decision, not plumbing — flagged, not
made. `featurize_actions` was rewritten as one fancy-indexed block (byte-identical
output, 0.46 → 0.32 ms/leaf, ~6% end-to-end, verified by interleaved A/B).

**Tree reuse: the brief's premise does not hold on this engine, and that is a Phase 1
dividend.** The brief expected chance-node invalidation ("a `play` that triggers a random
effect ... will generally NOT match"). Phase 1 deleted `game.rng` and moved every draw
onto the keyed `PseudoRandom` whose position table is part of the cloned state, so
`clone().step(a)` is a *deterministic function of (state, action)* — verified over `play`
and shop `reroll`, 8 repeats each, one signature every time. So the `state_signature()`
guard fires only on **driver** mutation, which is exactly the MP case: measured retention
**90.9% vanilla SP / 96.7% MLB solo / 96.7% MLB Nemesis / 80.0% with an opponent-score
relay** (`set_pvp_info` between decisions — the discard case working). Reuse buys
**1.3-1.7× decisions/s at fixed evidence** (`budget_mode="subtract"`) or **1.36-1.53×
effective sims/decision at fixed wall clock** (`"add"`) — a bigger end-to-end win than
cross-tree batching. Documented decisions: root Dirichlet noise IS re-applied to a reused
root (it cannot compound — noise only ever entered the previous root's own children, and
the new root's children carry bare priors); Gumbel redraws its sampled top-k every
decision and now scales sigma's `N_max` by **this decision's** visits only (a reused root
carrying 400 visits would otherwise multiply every q̂ by ~450 and turn sequential halving
into greedy-by-Q) while q̂ still uses all visits; forced single-action states
(`ROUND_EVAL` → `advance`) walk the tree down instead of dropping it.

**Plug-in for the tournament (W3 did not edit `mp/tournament/**`):** `mcts.make_player()`
+ `mcts.load_policy()` turn a checkpoint path into a `Player` with `act(game) -> dict`
(never `None` — `no_action={"type":"advance"}`, since `_drive_to_next_nemesis` steps
unconditionally) and `reset()`. The exact 12-line replacement for the `MCTSPlayer`
placeholder is in `BATCH_NOTES.md` §7.2. **Recommended settings: `leaf_batch=16`,
`reuse=True`** — because `runner.py` drives one agent to its Nemesis before starting the
next, so K is 1 no matter what the player does. Cross-tree batching needs the drive loop
to become lockstep (`BatchedMCTSPlayerGroup.act_many`, sketch in §7.3); worth ~1.4×, and
on the evidence above the 436-actions-per-leaf fix is worth more.

**Needs engine change (frozen, worked around):** `game.py` `clone()` 69 µs + `step()`
~0.08 ms are the throughput wall once the NN is batched (50-66% of a batched search);
`game.py:894` `state_signature()` at 42 µs is fine for reuse (2 calls/decision) but too
expensive for a per-edge signature or a transposition table.

---

## ✅ PHASE 3 COMPLETE — 2026-08-22 — Infrastructure

**Final gates, re-run by lead from a clean state after lifting the engine freeze for one fix:**

| Gate | Result |
|---|---|
| `pytest mp/engine/tests` | **1614 / 10 skip / 3 xfail / 0** (+5 regression tests for the fix below) |
| `pytest mp/tests` | **1073 / 2 xfail / 0** |
| `pytest mp/agent/tests` | **131 / 0** |
| `pytest mp/tournament/tests` | **31 / 0** |
| `pytest mp/eval/tests` | **49 / 0** |
| `engine_parity` + `parity_check` `--antes 1-8` | **126/126 both** |
| `git status` | only `?? mp/` |

**Lead fix at close (engine freeze lifted):** `game.py` bl_hook/bl_eye/bl_mouth rejection branches now route
exhaustion through `_mlb_fail_round()` under MLB (found independently by W2 and W4; `TestBossRejectionRespectsMLB`).
The W2 workaround in `tournament/runner.py::_repair_mlb_gameover_bug` is now dead code — remove in Phase 4.

**Delivered:** `mp/agent/` (MCTS layer forked from balatro-mcts `ee75d11`, re-targeted to the fork engine, obs =
env_v7's 447-dim, action features 56-dim, **bit-exact checkpoint round-trip**, outcome as a parameter, batched
cross-tree leaf eval + tree reuse + `make_player()`); `mp/tournament/` (N-agent same-seed runner, 100 agents × 8
antes in ~25 s, N×N outcome + log-margin matrices, population rank, per-ante distributions, 3 life rules,
degeneracy metric); `mp/eval/` (3-mode eval harness with paired-by-seed CIs, ρ-decay harness).

**Measurements that change the plan:**
- **ρ(h) is FLAT, not decaying** (N=150 seeds): buy_slot0 0.88→0.87, skip_small 0.81→0.77, reroll_once
  0.61→0.73 from h=1 to h=8; paired-seed VRF 2.5–10× at every horizon. Mechanism: `shuffle` stream is
  independent of shop streams, so card luck (dominant for greedy scripted play) is shared. Upper bound on ρ for a
  trained policy. The design doc's "~0.3–0.5 at h=4–8" guess is retired.
- **The NN was never the MCTS wall.** Batching cuts forward pass 26×/leaf (1.25 ms → 0.048 ms) but end-to-end only
  1.4× (531 → 761 sims/s, K=32 CUDA); 59% of time is `clone()`+`step()`, 13% per-leaf Python linear in the
  436-action set. **Next lever = expand fewer actions per leaf (top-k + sample)** — same fix as the 97 KB `Sample`.
- **Tree reuse never invalidates on chance**: `clone().step(a)` is deterministic (Phase 1 dividend); 90–97% hit.
- Serial CUDA is slower than CPU (per-leaf launch overhead) — use `leaf_batch=16, reuse=True` for K=1 players.

**Found, deferred to Phase 4 (design calls):** obs encodes only 8 hand / 5 joker slots (`env_v7.py:702`) — a 9th
card is invisible but playable; changes OBS_DIM → decide caps BEFORE a real training run. `Sample` ~97 KB (full
436×56 action features) → subsample actions. `train/loop.py` runs episodes serially (K-tree win is free via
`BatchedSearch.run_requests`). Tree memory 15 MB / 500-sim tree (119k nodes at a Nemesis) → `slots=True` on Node.
`env_v7` `(9-ante)` reward negative past ante 8. `MCTSPlayer` plug-in diff in `agent/BATCH_NOTES.md` §7.2.

**NEXT = PHASE 4** (`docs/PHASE3_BRIEF_2026-08.md` roadmap + this entry): plug MCTS player into the tournament,
obs-cap + action-subsampling decisions (Tagg), three-deck transfer spread, ρ with learned players, first
heterogeneous N×N from real checkpoints. **Overnight 2026-08-22: disposable cold MLB `train_cold` shakedown run**
(see next entry).

### 2026-08-22 ~01:40 — Overnight run launched + Phase 4 design decisions (Tagg)

- **Overnight shakedown running:** `mp/agent/runs/overnight_2026-08-22/` — `train_cold --minutes 360 --device cuda
  --ruleset mlb --encoder mlb --sims 30 --checkpoint-every 50`, started 01:27 CDT, self-terminates ~07:30, console log
  + `.DONE` sentinel beside the run dir. Purpose: does the value head move under the MLB outcome signal? Checkpoints
  are DISPOSABLE (see below).
- **DECISION (Tagg): set-based observation encoding**, not raised slot caps. Hand / jokers / consumables / shelf as
  masked variable-length sets (shared item embedding + pool/attention; joker position feature kept since order
  scores). Action features become set-based too (pool over selected cards). Drops the V7-shaped trunk, `OBS_DIM`,
  the 56-dim action row and the current `Sample` format. Must land BEFORE the first real training run; first test =
  paired-by-seed vs the flat encoder on the same cold run. Strong-model workstream in Phase 4.
- **DECISION (Tagg): per-sample action subsampling** as described (keep visited actions + a few random zero-visit,
  renormalize) — mechanical, near-lossless. Novelty-style selection is NOT for the replay buffer; it targets the
  exploring-starts sampler (which states to inject) to serve the strategy-diversity criterion.
- **DECISION (Tagg): log every episode's trajectory** — `(seed, deck, stake, ruleset, action list)`, a few KB, exact
  replay via the deterministic engine — and TAG interesting ones (win, high ante, novelty of build, comeback) for
  viewing. Viewing tiers for Phase 4: JSON/text replay; resurrect `viz/` via an exporter; investigate writing the MP
  mod's ghost-replay log format (`$MOD/lib/replay_log.lua`, `ghost_replay.lua`, `log_parser.lua`) so Tagg could play
  the agent's ghost in the real game.

### 2026-08-22 07:35 — OVERNIGHT SHAKEDOWN RESULT (lead read-out)

Run `mp/agent/runs/overnight_2026-08-22/` (`train_cold --ruleset mlb --encoder mlb --sims 30 --max-antes 8
--max-decisions 1500`, cuda, 6 h): **2,072 episodes, 0 errors, 42 checkpoints, 5.8 ep/min steady-state.**

| episodes | ante | len | z | z sd | policy loss | value loss |
|---|---|---|---|---|---|---|
| 1–50 | 6.70 | 210 | 0.409 | 0.180 | 3.41 | 2.21 |
| 51–200 | 8.93 | 168 | 0.763 | 0.122 | 3.36 | 0.023 |
| 501–1000 | 8.90 | 223 | 0.828 | 0.079 | 3.11 | 0.004 |
| 1501–2072 | 8.87 | 254 | 0.832 | 0.078 | 3.12 | 0.004 |

Stops: 1,985 `max_antes` / 47 `max_decisions` / 40 `game_over`.

**What happened (verified by replaying `ckpt_002072.pt` on 3 seeds):** the agent **skips 15/16 regular blinds**
(free under MLB: tag, no life) and coasts through the solo Nemesis, which `pvp_solo=True` auto-resolves with no
life lost. Net: one life lost at the ante-1 Boss, ante 9 reached with 0–4 jokers, z ≈ 0.83 every time. The engine
charges lives correctly (verified: fail-every-blind → 4→3→2→1→0). **The solo-MLB objective is degenerate, not the
engine** — and the pipeline found the exploit in ~100 episodes, which is the positive signal: MCTS + net + outcome
+ checkpoints all learn end to end.

**Consequences for Phase 4/5:**
1. **Never train on solo MLB with a free Nemesis.** The value head collapses (z sd 0.07) for a new reason: the
   policy reliably attains the objective's max, not "0 wins". Any useful signal must make the Nemesis cost
   something: (a) the tournament (real N×N outcomes — the Phase 5 design), or (b) interim solo training with an
   external Nemesis target (W4's `external_vanilla_big_blind_target` or a fixed per-ante table) so skipping both
   blinds and building nothing loses a life at the Nemesis.
2. The "skip straight to PvP" behaviour the campaign plan attributes to strong MLB players at high antes is
   something the agent will rediscover on its own as soon as skipping is ever free — encode nothing for it.
3. Checkpoints are disposable regardless (set-based encoder decision).

---

## Phase 4 — Make training real — KICKED OFF 2026-08-22 (morning)

**Lead brief: `docs/PHASE4_BRIEF_2026-08.md`.** W1 set-based encoder + subsampled Sample (strong); W2
tournament-driven training loop + MCTS plug-in + interim external-target objective (strong); W3 trajectory
logging/replay/tags/viz export (sonnet); W4 transfer-spread harness + `targets.py` + cleanups (sonnet). Engine/rng
frozen. Exit = first real training run launched by lead. Log entries follow.

### 2026-08-22 — P4-W3 trajectory log + replay — DONE ✅

**Agent W3.** `mp/replay/**` (new package: `log.py`, `replay.py`, `tags.py`, `export_viz.py`,
`cli.py`, `_bootstrap.py`, `_util.py`, `conftest.py`, `tests/` — 82 tests). Engine-only (no
`mp/agent` torch import, no `mp/tournament` import); `mp/engine/**`/`mp/rng/**`/`mp/agent/**`/
`mp/tournament/**`/`mp/eval/**` read-only. Full spec, hook contract, tag definitions, viz-export
coverage and the ghost-replay investigation are in `mp/replay/REPLAY_NOTES.md`.

- **`TrajectoryLogger`/`MatchLogger`**: `begin()`/`step()`/`end()`, exactly 3 call sites per
  loop. JSONL, one line/episode: `seed, deck_key, stake, ruleset, lives_start, actions[],
  steps[] (10-field summary), signatures{} (sig_every=10 default + start/final, blake2b-16 of
  state_signature()), outcome{}, final_state{} (read off the live game — jokers/lives/ante/
  money — so tags need no caller cooperation), tags`. Match lines carry a SINGLE interleaved
  `ops` list (player, action) in the exact order `match.step()` was called, since `sync()`'s
  side effects depend on that interleaving.
- **`replay()`/`replay_match()`**: re-run the action/op list through the same engine entry
  point, assert every logged signature, raise `ReplayMismatch` at the first divergent step.
  20-seed x {vanilla, MLB solo} + MLBMatch round trip all clean; a deliberately corrupted
  `skip_blind`→`play_blind` swap is caught at the first checkpoint at/after the corrupted step.
- **`narrate()`**: readable per-step story (blind/hand-type/chips/shop/lives/Nemesis scores),
  both episode and match lines; **`tags.py`**: win/reached_ante_{k}/skip_heavy/no_build/
  comeback/lives_lost_{n}/archetype_novel (corpus-wide, `tag_file()` only)/interest_score;
  **`export_viz.py`**: best-effort `viz/trajectory.json` shape (confirmed against the shipped
  file+`main.js`), documents exactly what doesn't map (`value_estimate`/`top_probs`/per-step
  `reward` — no agent-inference data in an engine-only log).
- **Measured**: bytes/episode 9.3-36.8 KB (mean 16.1 KB, MLB, ≤800 steps); logging bookkeeping
  overhead < 2% excluding signature capture; `state_signature()` capture itself is NOT cheap
  (comparable to a step()'s own cost) — `sig_every=10` default costs ~25-35% wall clock,
  `sig_every=100` ~5%; documented as a real tradeoff (finer divergence localization vs
  throughput), not a bug, with the tuning knob already exposed.
- **Ghost-replay investigation** (`$MOD/lib/replay_log.lua`/`ghost_replay.lua`/`log_parser.lua`,
  read-only, never copied): a ghost replay is NOT a full-run recreation — the live PvP
  resolver only ever reads, per Nemesis ante, a `hands: [{score, hands_left, side}]` sequence
  (an `MP.INSANE_INT`-encoded score string, which is just a plain decimal string for any chip
  value in our range). Everything else in the replay JSON is display-only. **Feasibility:
  HIGH for the score-ticker mechanism** (mechanically derivable from fields `log.step()`
  already records), **MEDIUM overall** (a small converter is the only remaining work; not
  built here per the brief — investigation only). Full field list + a worked JSON skeleton in
  REPLAY_NOTES.md §6.
- Gates: `mp/replay/tests` 82/82; `mp/engine/tests` 1614/10/3/0 unchanged; `mp/tests`
  1073/2/0 unchanged. No engine bug found; no engine change requested.

### 2026-08-22 — P4-W4 transfer spread + targets — DONE ✅

**Agent W4.** New: `mp/eval/targets.py` (per-ante external Nemesis targets, engine-only deps — no torch, no
`mp.eval`'s heavy `mlb_match_demo`/oracle/rng chain, so `mp/agent` can import it directly), `mp/eval/
transfer_spread.py` (the Red/Checkered/Plasma decision-gate harness), `mp/eval/tests/test_targets.py` (57) +
`test_transfer_spread.py` (19). `mp/results/transfer_spread_{greedy,greedy_reroll1_buy1,weak}.{json,md}` — 3
real scripted-player runs. `mp/engine/**`, `mp/rng/**`, `mp/agent/**`, `mp/tournament/**`, `mp/replay/**` only
read, never edited. Gates: `pytest mp/eval/tests` **125/0** (49 Phase-3 + 76 new); `pytest mp/engine/tests`
**1614/10/3/0** and `pytest mp/tests` **1073/2/0**, both unchanged.

`targets.py`: `vanilla_boss_target(ante, deck_key, stake)`, `vanilla_boss_target_fn()` (the one to register as
the Nemesis `chips_target` via `set_pvp_info` — the interim fix for the 07:35 overnight finding: a solo agent
that skips both blinds and builds nothing now loses a life at the Nemesis), `scaled_own_big_blind(k)`,
`table_target(path, quantile)` (reads a tournament run's `summary.jsonl`), `get_target(name, **kw)` registry.
W2 integration point: `mp.agent.mcts.outcome.ExternalOutcome.from_margin` is already built for exactly this hook
("the W2 / W4 hook", its own docstring).

`transfer_spread.py`: SP-MLB-solo (mode a, vs `vanilla_boss_target`) + tournament population rank (mode b,
`Tournament(n=32, life_rule="none")`), paired by seed across Red/Checkered/Plasma at White, cross-cell spread
with a bootstrap CI (resamples the paired seed set, recomputes each cell's mean + the range/variance per
replicate). Real numbers, 3 scripted specs (White stake, max_ante 8, 150 solo seeds / 16 tournament seeds):

| player | b_red win rate | b_checkered win rate | b_plasma win rate | rank_frac range [CI] |
|---|---|---|---|---|
| `hand=greedy` (no economy) | 0.000 [.000,.000] | 0.023 [.007,.043] | 0.300 [.257,.340] | 0.100 [.070,.132] |
| `hand=greedy,reroll=1,buy=1` | 0.280 [.220,.341] | 0.289 [.244,.334] | 0.430 [.391,.468] | 0.043 [.013,.113] |
| `hand=weak` (floor) | n/a — 0 Nemeses reached, any deck | | | 0.003 [.001,.007] |

**Reading:** for these fixed scripted baselines, **Red is the hardest cell against `vanilla_boss_target`, not
Plasma** — the reverse of the assessment's naive layer-1 prior. Mechanism: Plasma's `ante_scaling=2` makes the
external target 2x harder in absolute chips, but its `final_scoring_step` chips/mult BALANCE formula inflates a
jokerless/low-mult hand by MORE than 2x (a lone Ace scores 64 under Plasma vs 16 vanilla per `DECKS_NOTES.md`
S2) — exactly what a no/light-economy scripted policy plays — so the fixed target under-estimates true Plasma
difficulty for these policies. Checkered (also "LOW transfer" in the prior) doesn't collapse either — it's
consistently a little easier than Red (26S+26H makes an unforced flush more likely). The weakest policy
(`hand=weak`) hits a genuine floor (exhausts all 4 lives on regular blinds before any Nemesis, every seed, every
deck) and shows near-zero cross-deck spread, as expected. Methodological finding: tournament-mode `rank_frac`
spread (0.043-0.100) is much smaller than solo-mode win-rate spread (0.155-0.295) for the same two live
policies, because tournament mode normalizes away the target-formula/scoring-formula interaction that drives
the solo result — every population member faces the same Plasma-inflated scoring. **Caveat stated plainly in
EVAL_NOTES.md S13**: this is a naive-scripted-policy measurement, not the trained-policy measurement the
assessment's prior was actually about — a trained agent optimizing Plasma's real incentive structure could
still show the predicted collapse. Full tables + per-cell furthest-ante/lives-lost + CIs: `EVAL_NOTES.md` S13.

`env_v7` reward-reachability audit (notes only): `env_v7._finish_step`'s `R_BLIND_BASE*(9-ante)` reward
(`env_v7.py:120,455,479`) is unreachable from every path `mp/agent` drives — the sole `BalatroV7Env(` in that
tree (`agent/tests/test_nn_policy.py:96`) only calls `.reset()`, and `mcts/encoder.py:44,59` steals only the
unbound `_encode_obs` method via a `.game`-only shim, never constructing a real instance; `mp/agent`'s outcome
signal is exclusively `mcts.outcome.OutcomeFn`. It IS live, for real, through a DIFFERENT path:
`mp/engine/balatro_sim/env_mp.py`'s `MultiplayerBalatroEnv`/`_PlayerEnvProxy` (`:80-81,145-147`, pre-Phase-3,
built on `MLBMatch` directly) — already flagged in `MLB_NOTES.md` S5 and `agent/AGENT_NOTES.md` S8 — but that
module is exercised only by engine-layer tests within `mp/`, never by any `mp/agent` training path. No code
change needed for Phase 4.

Sanity/design details, `table_target` reader format, tournament-mode seed-count trade, and the "random" player
substitution: `EVAL_NOTES.md` S11-16.

### 2026-08-22 — P4-W1 set encoder + Sample v2 — DONE ✅

**Delivered** (`mp/agent/`, notes in `agent/SETENC_NOTES.md`): `mcts/encoder_set.py`,
`mcts/action_features_set.py`, `mcts/model_set.py`, `mcts/policy_set.py`,
`train/sample.py`, `--encoder set` wired through `policy.py` / `player.py` / `loop.py` /
`checkpoint.py` / `train_cold.py`, `scripts/eval_checkpoint.py`,
`benchmarks/bench_sample_size.py`, **53 new tests**. `mp/engine/**`, `mp/rng/**`,
`mp/tournament/**`, `mp/eval/**` untouched; `search.py` / `batched.py` / `model.py` /
`action_features.py` untouched, so the flat serial path is byte-identical.

**Gates:** `mp/agent/tests` **253**, `mp/engine/tests` **1614 / 10 skip / 3 xfail / 0**,
`mp/tests` **1073 / 2 xfail / 0**, `mp/tournament/tests + mp/eval/tests` **181** — all green.

**The observation is now five masked sets + one scalar vector** (hand 16, jokers 12,
consumables 6, shelf 8, packs 8; `scalars` 196) with a shared 251-key game-key embedding.
Caps are transport, not model structure: the net is permutation-invariant to within 1.4e-6
(value) / 1.9e-8 (logits) over all 200 fixture states, and garbage written into padded rows
changes nothing. Every `env_v7._encode_obs` feature is present (three consumable-slot
duplicates deduplicated, nothing dropped) plus: hand slots 9-16 and joker slots 6-12 — the
blindness AGENT_NOTES §8 flagged — joker key embedding instead of an alphabetical scalar,
stickers, pack contents, tags, deck/stake identity, and the full `env_mp` MLB block.
**Actions are pointer-style over the same item slots** (row-normalised `act_sel` over hand,
`act_tgt` over [jokers|consumables|shelf|packs]) so "buy Blueprint" is the Blueprint
embedding wherever it sits. `SetPolicyValueNet` = shared item encoders → one masked
4-head attention block over the 50-slot union → per-set mean+max pooling → 512-wide trunk,
**1 793 268 params** vs the flat net's 2 411 266.

**`Sample` v2 (`train/sample.py`)** keeps every visited action + `k_unvisited=8` random
zero-visit ones, renormalised EXACTLY (kept support = full support). Encoder-agnostic, and
callable with W2's `SampleCollector(sample_fn=...)` signature verbatim — `train_mlb
--encoder set --objective external` needed no W2 edit. Buffer/trainer/checkpoint carry v1
and v2 side by side; a Phase 3 version-1 checkpoint still resumes (test).

**Sizes (measured, `bench_sample_size.py` + a real 12-episode buffer):** mean bytes/sample
**45 810 → 6 497 (7.1×)** over 200 states; **94 628 → 8 236 (11.5×)** at a 436-action
`SELECTING_HAND` leaf; 28 597 → 6 563 on a real self-play buffer. **Not the ~20× the brief
projected** — subsampling alone gives 11.3× / 17×, and the set observation is a fixed
~5.2 KB dict that makes small-action states bigger. Still: a 200k buffer is **1.30 GB
instead of 9.16 GB**, weights-only checkpoints **21.6 MB vs 28.9 MB**, and `latest.pt` goes
from Phase 3's 137.8 MB to ~24 MB at the same buffer cap. Checkpoint round-trip **bit-exact
on CPU** for the set encoder (`CHECKPOINT_VERSION` 2, reads 1 and 2; records `net_kind` +
`encoder_caps` and refuses a resume across either).

**Paired 10-min comparison** (`train_mlb --objective external --device cuda --sims 40`,
`--encoder v7` vs `--encoder set`, seed 0; then 60 seeds SP-MLB solo each, paired through
the frozen `eval_harness --compare`): **every CI straddles zero** — furthest ante +0.05
[−0.22, +0.32], lives lost +0.10 [−0.17, +0.38], final money +0.92 [−1.08, +3.15].
No claim, and none is warranted: 10 minutes buys **one generation** (v7: 16 episodes /
271 train steps; set: 11 / 137), both nets die around ante 2, and the box was shared with
W2's gate run. Reports in `mp/results/w1_cmp_{v7,set,set_vs_v7}.json`.

**Throughput finding — the set net is launch-bound.** 436-action leaf, 200 sims,
`leaf_batch=16`: flat **449** CPU / **327** CUDA sims/s; set **259** CPU / **118** CUDA. At a
fixed 16-leaf batch the two are within 8%, so this is per-CALL kernel overhead (~60 launches
vs ~15), not arithmetic. Half of it I DID fix: batching an observation naively meant 25 small
host→device copies per forward — measured **20 such copies = 5.35 ms against 0.027 ms for one
packed copy**, more than the forward pass itself — so `policy_set` now packs key-major into
one float32 + one int16 buffer per transfer (pad+xfer 5.66 → 1.62 ms, CUDA search 69 → 118
sims/s). The other half needs the 5 card-field embeddings merged into one offset-indexed
table; that changes the param count, so it must land BEFORE a long run. **And both nets are
faster on CPU than on CUDA at this scale** — `--device cuda` is not automatically the right
choice for a self-play-bound run.

**Also found, not fixed** (details in `SETENC_NOTES.md` §7.3): `train_cold --ruleset mlb`
still trains against the degenerate free-Nemesis objective (the non-degenerate one lives only
in W2's `train_mlb`) — it should be removed or made to refuse; `mp/eval/common.py::
parse_player_spec` still raises `NotImplementedError` for `checkpoint:` although
`mcts.player.load_policy` / `make_player` now satisfy exactly the interface its docstring
asks for (three lines in a file frozen for W1 — `scripts/eval_checkpoint.py` is the
workaround); the set net's value head is unbounded like the flat net's (outputs ~2.1 at init
against a [0,1] target) — a sigmoid is an obvious cheap win before the first real run.
**Needs engine change: none.**

**Follow-up 2026-08-22 (lead-requested, before the first long run) — DONE ✅** — `SETENC_NOTES.md` §8,
`agent/tests/test_followups.py` (**18 tests**), `mp/agent/tests` **270 passed / 1 failed (not W1's, below)**.
(1) **Embedding tables merged**: eleven `nn.Embedding`s → **three** offset-indexed tables (card block; an
"aux" table for edition/rarity/shelf-kind/pack-set; the game-key table gathered ONCE for all four sets and
split) — **~25 embedding kernels per forward → 7**; params 1,793,268 → **1,793,536**. The merge is
equivalent, not approximate: every (field, value) pair keeps its own row and a test pins the merged gather
equal to per-field gathers on shared weights. Old set checkpoints won't load (table shapes changed) — which
is why this landed now. **sims/s re-measurement PENDING — machine in use**; one command re-runs all four
cells and prints the `--device` recommendation: `python mp/agent/benchmarks/bench_set_vs_flat.py` (new).
Pre-merge baseline to beat: flat 449 CPU / 327 CUDA, set 259 CPU / 118 CUDA.
(2) **Both value heads bounded**: `value_activation` (default `sigmoid`, also `clamp`/`linear`) on
`PolicyValueNet` and `SetPolicyValueNet` + `TrainConfig`, pinned by `_check_config`. Targets are in [0,1]
for every OutcomeFn and W2's population rank; both heads now sit at exactly 0.5 at init instead of ~1-2.
**`CHECKPOINT_VERSION` stays 2** — no state_dict shape change. Pre-follow-up checkpoints rebuild as
`linear` (`describe`/`from_description`/`from_checkpoint` all default that way) so their weights keep the
semantics they were trained under rather than being reinterpreted. **Bit-exact CPU round-trip still green**
for both nets.
(3) **`train_cold.py --ruleset mlb` now refuses** with a message naming `train_mlb.py --objective external`
(the overnight degeneracy, 07:35 entry). Two guards — on the parsed args and on `trainer.cfg.ruleset` after
a `--resume`, since resume takes the ruleset from the checkpoint. **CLI-only**: `ColdTrainer(ruleset="mlb")`
is untouched and still drives W2's `MLBTrainer` and the MLB tests. `--ruleset vanilla` unchanged.
**Not W1's, flagged not fixed:** `agent/tests/test_train_mlb.py::test_logged_tournament_trajectories_replay_exactly`
fails in `mp/replay/replay.py:87` — `ReplayMismatch at step 40 (action={'indices': [2], 'type': 'pick_booster'})`.
**Reproduced with the value-head change reverted**, so the follow-ups did not cause it. `replay.apply_op` is a
straight `game.step(action)` compared against a `state_signature()` digest, so the recorded action list is not
sufficient to reproduce the live state across a `pick_booster` (something mutating the game outside the logged
ops, or logged indices being pre-pick while the engine shrinks `booster_choices` between picks). W3 owns
`mp/replay/**`, W2 owns the logging hook.

### 2026-08-22 — P4-W2 tournament training loop — DONE ✅

Phase 4 exit-gate item 2 (+ the command for item 6). New: `mp/agent/train/{selfplay.py,
population.py}`, `mp/agent/scripts/{train_mlb.py, smoke_3way.sh, smoke_3way_report.py}`,
`mp/agent/TRAIN_NOTES.md`, `mp/agent/tests/test_train_mlb.py` (69). Edited:
`mp/agent/mcts/player.py` (additive `record_hook` / `Decision` / `legal_filter` /
`batch_leaf_eval`), `mp/tournament/players.py` (BATCH_NOTES §7.2 applied) and
`runner.py` (Phase 3 workaround deleted, no-progress guard, W3 replay hooks) + 4 new
tournament test files (25). **`mp/engine/**`, `mp/rng/**`, `mp/eval/**` untouched.**
Gates: `mp/agent/tests` + `mp/tournament/tests` **328**, `mp/engine/tests`
**1614 / 10 skip / 3 xfail / 0**, `mp/tests` + `mp/eval/tests` **1198 / 2 xfail**,
`mp/replay/tests` **82**.

**The objective is fixed and it is measurably alive.** A generation runs `s` tournaments of
N agents on one seed; at every Nemesis the N×N matrix gives each current-net agent a value
target of *population rank at the next Nemesis* (0.7) blended with *final standing* (0.3),
and the policy target is the search's visit distribution. Two 30-minute CUDA gate runs, 0
errors, `PAUSE` + `--resume` clean in both: **value-target sd 0.20-0.29 every generation**
against the overnight run's collapsed 0.07 and an alarm at 0.15; tie fraction 0.009-0.087.
Samples are collected through an additive `record_hook` on `MCTSPlayer` (free when unset,
pinned by a test that a hooked player plays the identical game), so the tournament runner
still owns the play loop. An eliminated agent scores 0.0, not NaN — otherwise losing your
last life is free, which is the degeneracy this exists to remove.

**What the gate runs actually proved, including the part that is not good news.** The
*target* is right: with scripted anchors in the population the current net's mean rank sits
at 0.27-0.55 against the anchors' 0.87-0.91, so the value head is told every ante that this
policy is losing. The *policy* still converges on skipping 90-99% of Small/Big blinds — and
at `--max-ante 4` it is correct to: a cold net cannot clear an ante-3 Big blind, so playing
one costs a life while skipping costs a tag's worth of tempo. Everything else improves while
it skips (mean jokers 1.2→3.6, mean ante 3.5→4.5, runs losing all 4 lives 21/32→10/32,
distinct joker sets 12→28). It is learning "skip and build" *before* learning to play a hand,
which is a curriculum problem, not an objective problem. Ten minutes of vanilla warm-up
(`--init`, weights only) already holds the skip rate at 65-77% flat where the cold baseline
goes to 99%; a `--max-skips-per-ante 1` mask holds it at 32-44% but costs mean ante
(2.7 vs 4.5). `TRAIN_NOTES.md` §7-8 has the tables and the two-stage first-real-run recipe;
`smoke_3way.sh` runs the unfinished (b)/(c) comparison in ~40 min.

**Needs engine change (frozen, worked around).** `game.py:1433` + `:1854`: the SHOP branch
of `legal_actions()` offers card-targeting `use_consumable` actions against `self.hand`,
which in the shop still holds the *previous blind's* cards, and `_use_consumable` silently
no-ops when the application fails — a legal action whose result is bit-identical to the state
it was taken from, i.e. an infinite loop. One MCTS agent burned all 20 000 steps of
`max_steps_per_drive` in a single shop; under another noise seed the same thing ate 55 s of a
60 s generation and produced 14 338 samples. Scripted/random players never hit it, which is
why Phase 3's 100-agent smoke did not. Worked around with a no-progress guard in
`_drive_to_next_nemesis` (signature before/after each step, force progress after 8 no-ops,
`noop_budget=0` disables) — invisible to any agent that ever changes the state, pinned by a
test.

**Also fixed / found:** `mcts.make_player` passed `leaf_batch` only into `MCTSConfig` and not
into the `MCTSPlayer` field that overrides it, so **every** tournament player built by
BATCH_NOTES §7.2's factory was silently running at L=1; and `MCTS._drive` answers leaf
requests one at a time by design, so `leaf_batch>1` never batched a forward pass at all.
Both addressed — and at 40 sims/decision it turns out not to matter (`leaf_batch` 1/4/16 =
238/217/218 sims/s on CUDA, interleaved), so the loop uses `--leaf-batch 1`, the exact search.
W3's replay caught a bug of ours the same way: the guard's diagnostic counter was stored on
the game object, and `state_signature()` sweeps up every scalar attribute, so every
trajectory it touched stopped replaying. Counter moved to `TournamentResult.forced_progress`;
a test now asserts the driver adds no attribute to the game.

**Wired:** W1's `SampleBuilder` as the sample seam (subsampled `Sample` v2, ~6 KB vs ~97 KB;
`--encoder set` runs end to end, measured **2.7× slower** than the flat encoder — 136-157
vs 354-469 sims/s at ante 8); W3's `TrajectoryLogger` through three new `Tournament` hooks
(`on_fanout` / `on_step` / `on_agent_done`) including the synthetic `__lose_life__` op, with
logged tournament trajectories replaying bit-exactly; W4's `targets.py` via `get_target` for
the interim `--objective external`. That interim objective needed a floor on the mirror
target (`--target-floor`): a self-referential target is 0 whenever the agent skips its Big
blind, which makes the Nemesis free again.

**NEXT (lead):** run `smoke_3way.sh`, then launch the two-stage first real run from
`TRAIN_NOTES.md` §8 — `train_cold --ruleset vanilla` for 4-6 h, then `train_mlb --init` for
24-48 h — with the skip-rate / blind-clear-rate watchdog in §8's table.

### 2026-08-22 — Lead close-out edits (machine in use by Tagg — code only, no compute)

- **Engine fix (freeze lifted for one change):** `game.py` SHOP branch now enumerates `use_consumable` with NO
  card targets (real game has no hand in the shop; `self.hand` still held the previous blind's cards and the
  targeted uses silently no-op'd → a legal action that changes nothing; P4-W2 saw an MCTS agent loop on it for
  20k steps). `TestShopConsumableActionsHaveNoCardTargets`. Residual (Phase 5 engine item): target-requiring
  tarots used with `[]` in SELECTING_HAND still return True and consume with no effect (enhancement/suit) or
  no-op (others); W2's driver-level no-progress guard covers the wedge.
- `replay/_util.py`: `__set_pvp_info__` synthetic op (W2's diff) so solo external-target trajectories verify.
- `eval/common.py::parse_player_spec`: `checkpoint:<path>[,sims=N,device=..]` → `mp/agent` `make_player` (lazy
  import). Verified live: the overnight checkpoint's first act on ALEEB is `skip_blind`.
- Deleted disposable run dirs (`p4w2_*`, `w1_cmp_*`, `cuda_smoke`, ~1.1 GB); kept `overnight_2026-08-22/`.
- Stray repo-root `C/overhead_*.jsonl` (W3 benchmark, bad Windows path) deleted; nothing outside `mp/` remains.

**Pending the machine:** `bench_set_vs_flat.py`, `smoke_3way.sh` (~40 min), full gate re-run, Phase 4 commit,
first real run launch (TRAIN_NOTES §8 two-stage command).

---

## ✅ PHASE 4 COMPLETE — 2026-08-22 — committed `ff6884e` on `mp/campaign`; FIRST REAL RUN LAUNCHED

**Gates (lead, clean):** engine **1616/10/3/0**, mp/tests **1073/2**, agent **271**, tournament **57**, eval **125**,
replay **82**; engine_parity + parity_check **126/126**. Lead additions at close: SHOP consumable-target fix +
tests, `__set_pvp_info__` replay op, `checkpoint:` player spec, PAUSE-file support in `train_cold` (Stage A) —
verified: PAUSE at ep 6 → checkpoint → `--resume` → ep 110.

**Bench (`bench_set_vs_flat.py`, 200 sims, leaf_batch 16):** flat CPU 430 / CUDA 369; **set CPU 302 / CUDA 168**
sims/s. → real run is `--encoder set --device cpu` (Tagg's architecture decision; 1.4× slower than flat).

**3-way smoke (10-min stages, set/CPU):** (a) warm-up only → skip 82%, clear 0%; (b) cap only → skip 46%, clear
4%; **(c) warm-up + cap → skip 48%, clear up to 25%, most jokers (2.6), best rank vs anchors.** Recipe = (c).

**FIRST REAL RUN `mp/agent/runs/real1.sh`** (launched ~17:00 CDT 2026-08-22, console `runs/real1.console.log`,
sentinel `runs/real1.DONE`): Stage A `real1_stageA/` vanilla warm-up 300 min (45 ep/min at start, ~13k episodes)
→ Stage B `real1/` MLB tournament 2880 min (N=16, m=8, anchors 0.25, p-history 4, skip cap 1 annealed at
clear-rate 0.5, value blend 0.7, sims 40, trajectories logged). **PAUSE:** `touch <run dir>/PAUSE`.
Watch per generation: value-target sd (> 0.15), **blind_clear_rate rising**, skip rate, joker sets / mean jokers,
rank vs anchors. Abort rule (TRAIN_NOTES §8): skip > 0.8 and rising after 10 generations → stop and rethink.
Stage A gate: if mean ante is still 1.0 after Stage A, do NOT start B — raise `--sims`.

### 2026-08-22 19:30 — FIRST REAL RUN PAUSED AT STAGE A (lead) — cold MCTS cannot clear ante 1

Stage A after 1 h 05 m: **4,350 episodes, 46 cleared ante 1 (1%), mean ante 1.00, value loss 0.0008** — the
May-2026 cold-MCTS failure reproduced (constant value target). Paused via PAUSE file (chain stopped, Stage B not
started; `real1_stageA/latest.pt` kept). Diagnostic: **200 sims → 0/53 cleared** ⇒ not a budget problem. The
scripted greedy player clears ante 1 on only 15/40 vanilla seeds; ante 1 needs a built flush/straight/two-pair
line, which a uniform prior over ~436 subsets never finds.

**Action:** P5-W0 (Opus) building a heuristic hand-quality prior (dry-run score → softmax, mixed with the net
prior, annealed) + top-K candidate pruning — "encode as prior, not constraint". Gate: 10-min vanilla smoke must
clear ante 1 at a real rate before Stage A restarts.

**Also raised by Tagg — multi-process self-play:** the CPU does engine clone/step (~60%) + per-leaf Python (~13%)
single-threaded; the GPU only wins on batched leaves (K≈32). Real lever = N worker processes on the 16-core
7950X feeding one GPU evaluator (AlphaZero shape), est. 8–10×. Phase 5 infra item #1.


### 2026-08-22 — W0 heuristic hand prior + candidate pruning — DONE (unblocks Stage A)

The urgent Phase-4/5 unblocker. Stage A of the first real run cleared ante 1 in **46 of
4 350 episodes (1.0%)** with **value loss 0.0008** — a constant target, nothing learning —
and a 200-sim diagnostic cleared **0/53**, so it was never the budget. New:
`mp/agent/mcts/heuristic.py`, `mp/agent/PRIOR_NOTES.md`,
`mp/agent/tests/test_heuristic_prior.py` (**38**), `scripts/w0_smoke_report.py`,
`runs/w0_smoke.sh` + `w0_smoke2.sh`. Edited: `mcts/{search,player,__init__}.py`,
`train/{loop,population,selfplay}.py`, `scripts/{train_cold,train_mlb}.py`.
**`mp/engine/**`, `mp/rng/**`, `mp/tournament/**`, `mp/eval/**`, `mp/replay/**`
untouched; no engine change needed.**
Gates: `mp/agent/tests` **309** (271 + 38), `mp/tournament` + `mp/eval` + `mp/replay`
**264 (57/125/82), unchanged**.

**The prior, in one line each.** `play(S)` = the engine's own dry run — `evaluate_hand`
for the type, `(base_chips + scoring-card chips) * base_mult` at the run's planet level,
refined for the top 8 by the side-effect-free `HypotheticalScorer` (skipped when the
board is plain, where the two are provably equal and a test says so).
`discard(D)` = `max(floor, draw)`: `floor` is the best play still available from the kept
cards (exact max-over-submasks DP), `draw` is `value(T) * P(Binom(|D|, p_T) >= m_T)` over
Flush / Straight / n-of-a-kind targets read off the actual remaining deck. Both are the
same units, so ONE softmax over play + discard is meaningful. Prior
= `(1-λ)·net + λ·h` where `h` redistributes only the mass the net already gave to hand
actions — **λ never moves mass between playing a card and using a Tarot**, and a
non-`SELECTING_HAND` state comes back untouched (the same dict object). Softmax is on
`log1p(score)/τ`, so τ is scale-free. λ anneals (`ep:<N>` or `clear:<r>`) to a 0.1 floor
and **is carried in the checkpoint** with its clear-rate EMA.

**Measured (10-min vanilla arms, `--encoder set --sims 40`, CPU, 9 arms over 2 rounds —
PRIOR_NOTES §4 has the full table):**

| arm | clear% (reached ante 2) | mean ante | mean len | value loss |
|---|---|---|---|---|
| cold baseline (no prior, no mask) | **0.0%** (0/499) | 1.00 | 9.3 | 0.0023 |
| **mask only** (λ 0, K 32) | **0.4%** | 1.00 | 8.9 | 0.0072 |
| λ 0.8, K 32, τ 0.5 | 14.1% | 1.16 | 17.3 | 0.0266 |
| **λ 0.8, K 32, τ 0.35** | **15.7%** | **1.18** | 19.1 | **0.0251** |
| λ 1.0, K 32, τ 0.5 | 9.8% | 1.10 | 16.5 | 0.0180 |
| λ 0.8, K 32, **τ 1.0** | 5.9% | 1.06 | 12.7 | 0.0139 |
| λ 0.8, K 32, **80 sims** | 6.9% | 1.07 | 17.9 | 0.0196 |
| λ 0.8, K 32, **discard bias 1.5** | 9.1% | 1.09 | 16.6 | 0.0191 |
| λ 0.9, **K 16**, τ 0.35 | 14.6% | 1.15 | 20.8 | 0.0229 |

**The brief's 20% bar was not hit** (14-16%, se ~3 points at n≈130); what was hit is what
the bar stood for — **the value target is no longer constant** (0.0251 vs 0.0008/0.0023),
so Stage A has something to learn from.

**Three findings worth carrying forward.** (1) **The mask alone does nothing** — pruning
to the 64 best-by-heuristic actions while keeping the net's near-uniform prior over them
scores 0.4%, i.e. the baseline. The prior is the mechanism; K is only a tree-size lever.
(2) **τ matters more than λ**: 5.9% / 14.1% / 15.7% at τ = 1.0 / 0.5 / 0.35, and below
0.35 is untried. λ = 1.0 is *worse* than 0.8 — keep a fifth of the net's spread.
(3) **More sims still does not help** (80 sims: 6.9%), which confirms the lead's
diagnostic from the other direction.

**Throughput, stated straight:** the heuristic costs **~1.9 ms per SELECTING_HAND leaf**
and the search runs **351 -> 184 sims/s** (set encoder, CPU, 40 sims). The mask pays back
part of it (168 -> 184 sims/s) but **pruning does NOT make the search faster overall** —
it is applied at `_apply_expansion`, after the policy has already featurised all 436
actions, so it saves the tree and not `featurize_actions_set`. Making it save the
featurisation means pushing the allowed set into `NNPolicy.encode_leaf`, a W1-owned
`PolicyValueFn` contract change; that is the biggest remaining per-leaf lever and it is
NOT done. Episodes/min falls further (44 -> 13) because a good episode is twice as long,
which is the point.

**Stage A (relaunch):** add
`--heuristic-prior 0.8 --heuristic-tau 0.35 --max-hand-candidates 32
--heuristic-prior-anneal clear:0.6 --heuristic-prior-floor 0.1` to TRAIN_NOTES §8's Stage
A command. **Stage B:** add
`--heuristic-prior 0.4 --heuristic-tau 0.35 --max-hand-candidates 32
--heuristic-prior-anneal clear:0.5 --heuristic-prior-floor 0.1` — 0.4 not 0.8 because a
large shared λ makes every tournament seat play the same hands and pushes `tie_fraction`
up. `population.instantiate` gives the prior to **every net seat, current and past-self
alike** (a past checkpoint searching without it is a different agent from the one whose
weights were trained with it), anchors are unaffected, and `MLBTrainer.anneal_heuristic()`
runs off the SAME `clear_rate_ema` as the skip cap so both crutches come off together
(`h_lambda` is logged next to `skip_cap`). **New watchdog line: `v=` (value loss) must
stay above ~0.01** — a collapse back toward 0.001 is the old failure returning.

**Found, not fixed** (PRIOR_NOTES §6): the featurisation lever above; the prior has no
notion of *finishing* a blind (the obvious next term if a real Stage A plateaus under
~40%); the draw potential ignores Shortcut/Smeared and uses a binomial rather than a
hypergeometric tail (both understate a draw); `evaluate_hand` over 218 subsets is 70% of
the heuristic's cost and a vectorised rewrite was deliberately NOT done (a second hand
evaluator that can drift from the engine's is the bug class this project spent two phases
removing). Also: the `blinds` column in the smoke report counts the blind INDEX, which a
SKIP advances — judge arms on `clear%`, not on it.

### 2026-08-22 ~20:00 — FIRST REAL RUN RELAUNCHED with the W0 heuristic prior (lead)

W0 delivered (`mp/agent/mcts/heuristic.py`, `PRIOR_NOTES.md`, 309 agent tests): dry-run hand-quality prior mixed
into the MCTS prior on play/discard (λ, τ), top-K candidate mask, anneal on the blind-clear EMA. 10-min smoke:
cold **0.0%** ante-1 clears → **λ0.8/τ0.35/K32: 15.7%**, value loss 0.0023 → 0.025 (target alive). Mask alone
does nothing (0.4%); τ matters more than λ; more sims still doesn't help. Throughput 351 → 184 sims/s.

Relaunched `real1.sh` at ~20:00 CDT with Stage A `--heuristic-prior 0.8 --heuristic-tau 0.35
--max-hand-candidates 32 --heuristic-prior-anneal clear:0.6 --heuristic-prior-floor 0.1` and Stage B
`--heuristic-prior 0.4 … clear:0.5`. First 50 episodes: clear% ~20%, mean ante 1.2, v loss 0.025, 16 ep/min.
Quirk (left as is): the anneal is λ = λ0·(1 − clear_EMA/r) with the EMA seeded from episode 1 (100% clear) → λ
started at the 0.1 floor and climbs back as the EMA settles (0.29 by ep 50). Expected steady state λ≈0.5 at a
20% clear rate. Old attempt's console kept as `runs/real1.console.attempt1.log`.
**Stage A gate for the morning:** clear% rising past ~20% and mean ante climbing; `v=` must stay > 0.01.

---

## Phase 5 — KICKED OFF 2026-08-23 ~03:30 (while `real1` Stage B runs)

**Tagg's two questions set the first two workstreams:**
1. **Parallelism (P5-W1, Opus):** the run is single-process, one core, GPU idle (~230 sims/s). Build N worker
   processes + one batched evaluator (AlphaZero shape), lockstep tournament, same checkpoint format so `real1`
   resumes on all cores. Est. 8–10× on the 7950X. Swap procedure: PAUSE → `--resume … --workers N`.
2. **🔴 CLAIRVOYANCE (P5-W2, Opus) — the search cheats.** `search.py`'s docstring ("each simulation sees different
   RNG outcomes") was true only on the old engine's global `random.Random`. On the Phase-1 engine the keyed RNG +
   draw-pile order are cloned, so every simulation sees THE TRUE FUTURE: exact draws after a discard, exact
   reroll, pack contents, every probability roll. **All Stage A/B numbers so far are perfect-information.** Fix =
   determinization at the root of every simulation (keep observed state; reshuffle the draw pile; fresh seed
   for every future stream). W2 adds `BalatroGame.determinize()` + measures the clairvoyant-vs-determinized
   gap on the latest Stage B checkpoint; the lead wires it into search/batched after W1 hands off.

### 2026-08-23 — P5-W1 multi-process self-play + shared batched evaluator — DONE ✅

**Agent W1.** Phase 5 infra item #1, built while `runs/real1` (Stage B) was live — its run dir untouched, and
`mp/engine/**`, `mp/rng/**`, `mp/eval/**`, `mp/replay/**` read only. New: `mp/tournament/parallel.py`,
`mp/agent/parallel/` (10 modules), `mp/agent/train/parallel.py`, `mp/agent/benchmarks/bench_parallel.py`,
`mp/agent/PARALLEL_NOTES.md`, `mp/agent/tests/test_parallel.py` (**28**),
`mp/tournament/tests/test_parallel_runner.py` (**17**). Edited additively: `train/population.py`
(`instantiate(..., policy_for=)`), `scripts/train_mlb.py` (`--workers`, `--evaluator-device`, …).
Gates: `mp/agent/tests` **337** (309 + 28), `mp/tournament/tests` **74** (57 + 17), `mp/eval` + `mp/replay`
**207 unchanged**.

**Architecture (AlphaZero shape).** N worker processes each own a SUBSET of the tournament's seats — games,
MCTS trees and caches, per-agent RNGs, W0's heuristic prior, the skip cap, the sample collectors, the trajectory
loggers — and **no net**. A worker drives its seats in lockstep (`drive_many` + `LockstepDecider`), so their
leaves batch; `RemotePolicy` ships them to ONE evaluator, a daemon thread in the main process, which groups by
net (live + past selves), runs one forward per group and replies. Every cross-agent decision (N×N matrix, life
rule, elimination, the ante barrier) stays in the main process in `Tournament.run`'s order — `Tournament` itself
is **not edited**; `ParallelTournament` subclasses it and talks to a `TournamentDriver`. `--workers 0` is the old
path, byte for byte.

**Transport = shared memory, and here is why.** A 436-action set-encoder leaf is **126 612 bytes** (obs 21
arrays / 3 660 B + action block 4 arrays / 122 952 B); the reply is 1 748 B. 16 workers at ~184 sims/s is
**~370 MB/s of requests**. So: one shared-memory arena per worker for the payload (the worker writes the arrays
where the evaluator will read them; the evaluator's only copy is the `np.stack`/pad `BatchedSetNNPolicy` already
does), and a `Queue` carrying just `(n_actions, offset)` per leaf plus a `Pipe` carrying the round id. Batching
policy: the evaluator blocks for the FIRST submission then drains — it never waits for a named worker, so a slow
tree can't stall the batch, and the queue refills during the forward pass, which self-balances.
Measured: `--evaluator-max-wait-ms 2` raises the mean batch 1.64 → 2.44 and cuts forward time 14.6 → 9.8 s but
LOSES on wall clock (531 → 482 sims/s) — on CPU, leave it at 0; it is a CUDA lever.

**Checkpoints: identical format, both directions, verified live.** `ParallelMLBTrainer` subclasses `MLBTrainer`
and overrides exactly one method (`_play_tournament`), so `state_dict`/`load_state_dict`/`from_checkpoint` are
the inherited ones and the worker count is NOT in the payload. A parallel checkpoint resumed single-process
(gen 1 → 2 ✅) and a single-process checkpoint resumed into the parallel trainer (test ✅). PAUSE mid-generation
with workers: stopped after the tournament in flight, trained, drained the pool, checkpointed, exit 0 ✅.
Worker crash: its agents are marked `crashed` (handled exactly like a death), the matrix is built from the
survivors, the run continues; `respawn_dead()` brings the process back next generation.

**Determinism.** Structural determinism is guaranteed and tested (seed string resolved once in main; per-agent
`default_rng(member.seed)`; samples re-ordered by `(seed, agent, decision)`); numerical agreement is contracted
at BATCH_NOTES §3's ~1e-7 and **measured exact** — 1 vs 4 workers with a real net gave identical matrices,
lives, value targets and sample counts. `ParallelTournament` + `LocalDriver` == `Tournament` byte for byte for
all three life rules, both fan-outs, odd N, the degenerate identical population, and the trajectory hooks.
**One deliberate difference:** `SampleBuilder`'s subsampling RNG is per-agent (`cfg.seed`, generation, idx)
instead of the trainer's shared generator — it has to be, and it makes the subsampling independent of the worker
count. So a parallel generation is a *continuation*, not a bit-exact replay, of the serial one.

**Throughput — smoke only so far** (N=8, sims 40, max_ante 4, 1 seed, set/CPU, **with `real1` still holding a
core**): serial **229 sims/s** → 1 worker **268 (1.17×)** → **4 workers 531 sims/s (2.32×)**. Even one worker
beats serial, because the serial runner drives one agent at a time and never has two leaves to batch (mean batch
3.24 at 1 worker). 4 workers is sub-linear: the ante is a barrier, 26% of worker time was spent waiting on the
evaluator, and with 2 seats per worker there is little to batch inside a worker (mean batch falls to 1.64).
CUDA was sanity-checked only (runs; 9.9 s of a 17 s wall inside the forward at batch 1.3 — launch overhead,
as BATCH_NOTES §6 predicted).

**FOR THE LEAD — one command, machine free:**
```
python mp/agent/benchmarks/bench_parallel.py            # workers {1,4,8,12,16} x {cpu,cuda} + serial
python mp/agent/benchmarks/bench_parallel.py --include-local   # + the "each worker holds its own net" arm
```
**FOR THE LEAD — the swap for `real1`:** `touch mp/agent/runs/real1/PAUSE`, wait for `=== Stopped (PAUSE)`, then
```
python mp/agent/scripts/train_mlb.py --resume mp/agent/runs/real1/latest.pt \
    --minutes 2880 --device cpu --workers 12 --evaluator-device cpu --run-dir mp/agent/runs
```
(add `--log-trajectories --sig-every 50` to keep trajectory logging; pick `--workers` / `--evaluator-device`
from the benchmark — 12/cpu is the conservative pre-benchmark choice on a 16C/32T box.)

**Found, not fixed** (PARALLEL_NOTES §9): the evaluator is ONE thread and may become the bottleneck at 16
workers (diagnose with `eval_forward_s` / `worker_wait_s_mean` in the generation log; fixes are its own process
or two threads split by `policy_id`); `p_history=4` distinct checkpoints fragment the batch into four
one-leaf forwards per round; **the 436-actions-per-leaf cost is now also the transport cost** — pushing W0's
candidate mask into `encode_leaf` would cut per-leaf Python, arena traffic and the padded forward at once, and
is still the biggest single lever (PRIOR_NOTES §6, BATCH_NOTES §6); a crashed worker's agents are lost for the
generation, not just the tournament; `partition_agents` balances by `sims`, which is a proxy for cost;
`--objective external` is deliberately not parallelised.
3. **Decision statistics (P5-W3, Opus) — Tagg's ask:** `mp/stats/` — for packs/rerolls/vouchers at any state: hit
   valuation by dry-run gain on a clone, P(≥1 hit) exact (hypergeometric) + Monte Carlo through the real generator
   on determinized clones, true cost incl. interest loss ($1/$5, cap $25 held), urgency from the chip gap to the
   next target, net EV table; CLI + a 126-seed sweep for antes 1–3. = the curated-shortlist / layer-1 instrument;
   later fed to the MCTS prior as features.

### 2026-08-23 ~04:10 — PHASE 5 PAUSED at Tagg's request (strategic question pending)

All three Phase 5 agents stopped. W2 (determinization) and W3 (decision stats) had not written anything. W1
(parallel self-play) left PARTIAL, uncommitted, default-off work on disk: `mp/agent/parallel/`,
`mp/agent/train/parallel.py`, `mp/tournament/parallel.py`, `tests/test_parallel.py`, `benchmarks/bench_parallel.py`,
`PARALLEL_NOTES.md`, plus additive `--workers`-style flags in `train_mlb.py`/`selfplay.py`. Tree is green
(agent 309, tournament 74 with `test_parallel.py` excluded). **Do not trust the parallel path until finished.**
The `real1` Stage B run continues untouched (gen 19 at 04:00).

**Tagg's open question for the morning:** *if Balatro is fundamentally an EV calculation, MCTS may not be the
right search method.* Decide before resuming Phase 5 (parallelism, determinization, decision stats all
presuppose a search agent). Keep in view: the search is currently clairvoyant (W2 finding), so no number from
`real1` is evidence either way about MCTS's fitness.
