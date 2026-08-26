# Oracle sources, provenance and caveats

Built 2026-08-21 (Agent D, Phase 0).  Target game: **Balatro 1.0.1o** (local Steam install, Lua
extracted to `_reference/balatro_src/`, gitignored).  Machine: Windows 11, Node v24.13.0,
npm 11.6.2, Python 3.13.5.  No C/C++ toolchain (no gcc / cl / cmake), so the OpenCL and C++ code
bases were used as reference only.

## What ran locally

| # | Source | Role | Status |
|---|---|---|---|
| 1 | **Blueprint** (miaklwalker/Blueprint @ `62898ed0`, 2026-07-24) -- TypeScript port of TheSoul/Immolate | **primary generator** | ran headlessly via `vite-node`; 126 seeds x 8 antes x 50-deep queues + all packs |
| 2 | **TheSoul** (SpectralPack/TheSoul @ `780c1c21`, 2025-04-12) -- `immolate.wasm`, Emscripten build of MathIsFun0's C++ Immolate | **independent cross-check** | ran in bare Node; agrees with Blueprint on every field for all 126 seeds |
| 3 | **balatrohq.com/tools/seed-analyzer/** | third-party web analyzer | client-side JS; only its example seed **ALEEB** is server-rendered -- transcribed and checked: agrees on ante 1 (boss, voucher, tags, 8 queue items, 4 packs incl. standard-card enhancements/editions, Soul->Canio/Triboulet) and on the "ante 4 = Blank voucher" claim.  Prose claims for OG4YQPSI / 2K9H9HN / 7LB2WVPK also verified (below). |
| 4 | **Immolate** (MathIsFun0/Immolate @ `26f41efc`) -- OpenCL | reference only | not runnable here (no OpenCL toolchain) |
| 5 | **balatro-seed-finder** (izanagi1995 @ `b3a112f8`) -- C++ CPU port | reference only | needs a C++ compiler; none available |
| 6 | **Agent C's port** `rng/generate.py` (independent port from the game Lua) | consumer / corroboration | via `parity_check.py` adapter: **126/126 seeds exact through ante 8** against the game-faithful variant |

All clones live in `oracle/blueprint_runner/vendor/` (gitignored via `blueprint_runner/.gitignore`);
`setup.ps1` re-creates them at the pinned commits.  Raw analyzer dumps are in
`blueprint_runner/_raw/` (gitignored, 73 MB, regenerable).

## Exact commands

```powershell
powershell -ExecutionPolicy Bypass -File oracle/blueprint_runner/setup.ps1     # clone + npm ci
cd oracle/blueprint_runner/vendor/Blueprint
$env:BLUEPRINT_COMMIT = (git rev-parse HEAD)
npx vite-node ../../check_fixtures.ts                                              # 5961/5961 fields match Blueprint's Immolate fixtures
npx vite-node ../../run_blueprint.ts          -- --seed-file ../../seeds.txt --antes 8 --cards 50 --buy-vouchers --out ../../_raw
npx vite-node ../../run_blueprint_faithful.ts -- --seed-file ../../seeds.txt --antes 8 --cards 50 --out ../../_raw
cd ../..
node run_thesoul.js --seed-file seeds.txt --antes 8 --cards 50 --out _raw
cd ..
python build_ground_truth.py                   # -> ground_truth/<SEED>.json (+ cross-checks, variants)
cd ../..
python -m oracle.parity_check --validate-only
python -m oracle.parity_check --antes 1-3 [--variant faithful] [--seeds A,B]
```

Settings used everywhere: Red Deck, White Stake, fully unlocked profile, analyzer version flag
`10106` (Immolate's "1.0.1f" switch -- the newest pool set those tools have; no generation-relevant
change is known between 1.0.1f and 1.0.1o and the community runs these analyzers against 1.0.1o).

## Seeds (126; `blueprint_runner/seeds.txt`)

* 26 known seeds: the 9 Blueprint fixture seeds (2K9H9HN, 3SZ71111, 7LB2WVPK, 7ODNKXP, 9ZXMM1M,
  SF9SZOB1, U8RJYV6M, V3PUR5L4, VNOMH111), balatrohq examples (ALEEB, OG4YQPSI), seeds named in
  gaming articles (HSC1L2DX, 8Q47WV6K, 4T1SZKLF, XD55DZ57, PTSBMSMQ, LHZYIWR6, 29ZSW8MY, PQNVFI72,
  9Q9HQXZG), three from Blueprint's legendary list (HPR8Q7K, 5YVHAEP, 8QBRTPD), IMMOLATE, and two
  edge cases (`A`, `11111111`).
* 100 random seeds: `random.Random(20260821)`, 8 chars from the game's own seed alphabet
  `123456789ABCDEFGHIJKLMNPQRSTUVWXYZ`.

Every file covers antes 1-8, 50 shop-queue items per ante, all 2/3 shop visits with both packs
opened, soul spawns, legendary stream, and the buy-every-voucher branch.

## Cross-check results

* **Blueprint vs its Immolate fixtures**: `check_fixtures.ts` re-runs the 9 fixtures shipped in
  Blueprint's `___tests___/seedJson` (labelled `immolateResults`, mostly Ghost Deck, antes 1-8):
  5961/5961 compared fields match (boss, voucher, tags, 50-deep queues with editions/stickers, all
  packs and contents).  Blueprint's own vitest suite shows 219 "failures" that are all a fixture
  representation change (`base` array vs string) -- not engine differences.
* **Blueprint vs TheSoul WASM**: 126/126 seeds agree on every compared field (~2,970 per seed:
  boss/voucher/2 tags x 8, 400 queue items with type/edition/rarity/stickers, 46 packs with kind and
  every card incl. enhancement/edition/seal).  These are two separate code bases (TS port vs
  C++-to-WASM), so this rules out porting slips in either, not shared design assumptions.
* **balatrohq (ALEEB)**: agree, see table above.
* **Community prose claims** (low reliability; recorded for honesty):
  * OG4YQPSI -- balatrohq: "ante 1 Hieroglyph + Soul->Perkeo; ante 2 Baron + Reroll Surplus": **confirmed** (Baron is queue item 2 of ante 2).
  * 2K9H9HN -- balatrohq: "ante 2 Souls -> Perkeo and Triboulet, Crystal Ball, Vampire": **confirmed**.
  * 7LB2WVPK -- balatrohq: "ante 3 Seed Money, Ride the Bus, The Tribe": **confirmed** (queue items 6 and 8).
  * HSC1L2DX -- gamerant "DNA in first shop": DNA is queue item 3, i.e. slot 1 of the *second* shop (after the Big Blind), **off by one shop**.
  * PQNVFI72 -- gamerant "Photograph or Ancient Joker ante 1; Triboulet ante 3": Ancient Joker is item 2 of ante 1, Photograph item 13; **no Soul in ante 3** on Red/White -- not reproduced.
  * PTSBMSMQ ("Mime + Ankh ante 1"), 8Q47WV6K ("two legendaries via Souls"): **not reproduced** on Red/White (possibly other deck/version or mistyped seeds).  Both generators agree with each other on these seeds; the articles are the weak source.

## Modelling caveats (read before trusting a mismatch)

1. **`used_jokers` suppression is missing from every published analyzer.**  `card.lua`
   `Card:set_ability` sets `G.GAME.used_jokers[key] = true` for *every* card created (shop cards
   and pack cards included; cleared in `Card:remove` unless one is owned), and
   `get_current_pool` marks used keys `UNAVAILABLE`.  So the real game (a) never shows the same
   card in both slots of one shop and (b) never puts a currently-displayed shop card into a pack
   opened from that shop; each such resample also advances the shared `<pool>_resample{n}` streams.
   Immolate / TheSoul / Blueprint / balatrohq do not model (a) or (b) (they only lock within a pack).
   Evidence it is real: the Lua is unconditional; my own driver on Blueprint's `Game` class with
   these two rules added (`run_blueprint_faithful.ts`) agrees **exactly** with Agent C's independent
   port from the Lua on all 126 seeds x 8 antes, while the published output differs from both at
   precisely the 953 flagged fields (222 same-shop duplicates, 319 pack/shop collisions, 412
   downstream resample-stream shifts).  Primary `antes` stays the published-analyzer output (it is
   what two independent tools and balatrohq reproduce); the faithful sequence is available via
   `variants.game_faithful_used_jokers.overrides` and `parity_check --variant faithful`.  What is
   still unverified against the running game: the faithful variant itself (both derivations read the
   same Lua).  A live-game spot check of one flagged shop (e.g. seed 1558AXDL, ante 1, post-Big-Blind shop,
   11th reroll = queue pair 13: analyzers say Clever Joker + Clever Joker, faithful says Clever + Crafty)
   would settle it.
2. Packs are modelled as opened at their visit, in order, before any reroll.  Opening a pack after a
   reroll changes which cards are "displayed" and therefore (under rule 1) possibly its contents.
3. Pack art variants (`p_arcana_normal_1..4`, ...) are not resolved; keys drop the suffix.
4. `deck_order_unverified` (per-blind shuffle) comes from Blueprint only; TheSoul's WASM has no
   shuffle API.  Excluded from parity by default.
5. Stickers only roll at Black+ stakes; all data here is White stake (stickers all false).  Deck/stake
   variants were not generated (Red/White is the Phase 0 gate); `run_blueprint.ts --deck/--stake`
   supports them.
6. `legendary_stream` is the raw `Joker4` stream without purchase-locking; `first_soul_joker_by_ante`
   gives the `edisou{ante}` edition for the *first* Soul used in that ante.
7. `voucher_chain_if_bought` applies each ante's voucher purchase after that ante's queue (Blueprint's
   buy semantics), so rate-changing vouchers affect the *next* ante's queue only.
8. The extracted reference is missing `functions/UI_definitions.lua` (it holds
   `create_card_for_shop`, the `cdt{ante}` type roll and the `'sho'` key append) -- not needed by the
   oracle, but Agent C should know the canonical source is inside the exe's zip.

## Files

| path | purpose |
|---|---|
| `ground_truth/<SEED>.json` | 126 files, schema in `schema.md` |
| `schema.md` | field-by-field schema, assumptions, variants |
| `keymap.py` | analyzer name -> game key tables (generated from the game's center tables) |
| `build_ground_truth.py` | raw dumps -> schema files, cross-checks, faithful-variant overrides |
| `parity_check.py` | CLI: `--validate-only`, `--list`, port comparison (generic contract or RunState adapter), `--variant` |
| `blueprint_runner/run_blueprint.ts` | headless Blueprint `analyzeSeed` driver |
| `blueprint_runner/run_blueprint_faithful.ts` | Blueprint `Game` class driver with `used_jokers` rules |
| `blueprint_runner/run_thesoul.js` | TheSoul WASM driver (fresh-run locks, fully unlocked) |
| `blueprint_runner/check_fixtures.ts` | Blueprint vs its Immolate fixtures |
| `blueprint_runner/seeds.txt`, `setup.ps1`, `.gitignore` | seed list, reproducible vendor setup, ignore rules |
