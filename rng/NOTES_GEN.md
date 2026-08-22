# NOTES_GEN -- Agent C (generation layer)

Companion to `GENERATION_SPEC.md` (the contract) and `generate.py` (the port). This file is the
honest ledger: what was assumed, what is stubbed, what is untested, and what Agent D's ground
truth has to contain to close each gap.

## 1. Status at hand-off

* `generate.py` imports cleanly against the real `core.py` (Agent A) and `pools.py` (Agent B).
  No local stubs remain; both imports are still wrapped in `try/except` so the module imports
  even if a sibling breaks (you get a clear `ImportError` at first use instead).
* `python -m mp.rng.generate EXAMPLE1 2` runs end-to-end: run start (boss, voucher, tags,
  shuffled deck, idol/mail/ancient/castle), 3 shops per ante with a reroll, both packs opened,
  ante transition.
* `keys.py` (Agent B) arrived after my `Keys` class was written; the two agree on every
  construction (`rarity_key`, `pool_key`, `resample_key`, `KEY_APPENDS`). I kept `Keys` local so
  the spec and the code cannot drift; Phase 1 may collapse them.

### Oracle result (the important part)

`mp/tests/test_generate_oracle.py` (promoted from my scratchpad harness; follows Agent A's
conventions: `jit.off()`, no FFI punning, nothing but strings/ints/bools across the boundary, Lua
sliced from `_reference` at test time with boundary-line assertions, clean skip without lupa)
loads the *real* Lua generation functions verbatim into LuaJIT 2.1 and drives them and
`generate.py` through identical scripts:

| Scope | Seeds | Result |
|---|---|---|
| run start + antes 1-3, 3 shops/ante, 2 rerolls/shop, both packs opened each shop; variants: buy a card, Showman from ante 2, banned keys (`bl_hook`, `bl_wall`, `j_joker`, `c_fool`, `p_buffoon_normal_1`), Gold-stake stickers, fresh-profile locks + discoveries | 15 x 7 scenarios in the test (30 x 7 in the original sweep) | **0 mismatches** (keys, fronts, editions, seals, eternal/perishable/rental, vouchers, tags, bosses, `used_jokers`, `bosses_used`, deck shuffle) |
| Judgement, Soul x2, Wraith, Riff-raff x2, Top-up, Emperor x2, High Priestess, Sixth Sense, 8 Ball + Purple Seal (shared `8ba`), Rare/Uncommon Tag cards, Voucher Tag x2 (shelf exclusion), Aura x3, Erratic deck, idol/mail/anc/cas | 20 | **0 mismatches** |
| `pairs(G.GAME.hands)` order (To Do List / Orbital) inside the game's own `lua51.dll` | 3 fresh VMs | identical, equals `HANDS_PAIRS_ORDER` |

Suite: `python -m pytest mp/tests -q` -> **146 passed** (17 Agent A + 129 here) in ~2 s. A
deliberate perturbation of one key (`edi...`) produces mismatches, so the comparison is live.

The only difference ever seen was the `p_buffoon_normal_1` vs `_2` art suffix of the forced
first pack (unseeded `math.random(1,2)`; cosmetic) -- normalised in the harness.

What the harness proves: `get_current_pool`, the resample loop, `create_card` (incl. soul rolls,
front, stickers, edition), `poll_edition`, `get_pack`, `get_next_voucher_key`,
`get_next_tag_key`, `get_new_boss`, `create_card_for_shop`, the `Card:open` pack loop,
`pseudoshuffle`, and the reset_* picks are ported exactly, and that my `used_jokers` model
matches the Lua's `set_ability`/`remove` semantics under purchase/reroll/pack/Showman.

What it does **not** prove (because I stubbed it, from reading): the *sequence* of calls across
game states -- run start order, boss-defeat order (`ease_ante` before the voucher draw), Cash Out
tags/boss, `used_packs` reset per round, shop shelf release on leave -- and the `Card` stub
itself. These are exactly what end-to-end ground truth must confirm.

## 2. Findings that change the brief or the engine

1. **`used_jokers` is marked on card *creation*, not purchase** (card.lua:349-354) and released
   on `Card:remove` only when no owned copy remains (card.lua:4741-4747). The brief assumed
   purchase. Consequences the engine must reproduce: no duplicate within a shop shelf, no
   duplicate within a pack, pack cards exclude the shop shelf behind them, rerolls release the
   old shelf first. `generate.create_card` marks automatically; the engine calls
   `acquire`/`release_shop`/`release_pack`/`remove_owned`.
2. **Showman** only disables the `used_jokers` test (3 sites) and only when owned in `G.jokers`
   and not debuffed. It does not touch `used_vouchers`, locks, gates or bans.
3. **The first pack of a run is a Buffoon pack without consuming `shop_pack1`**; the second
   slot of the first shop is the first `shop_pack1` draw.
4. **Voucher for ante N+1 is drawn when the ante-N boss dies**, after `ease_ante(1)`, key
   `Voucher<N+1>`. Tags and the boss for N+1 are drawn at Cash Out. Redeeming empties the slot
   for the rest of the ante (no redraw).
5. **`rarity<a>sou` is consumed by The Soul** even though the result is discarded; the legendary
   pool key is bare `Joker4` (no append, no ante) -- so a Soul in ante 3 and a Soul in ante 5
   draw from the same stream.
6. **Spectral pack cards roll `soul_Spectral<a>` twice** (Soul gate, then Black Hole gate; Black
   Hole wins on a double hit). Shop consumables never roll either (`soulable` is pack-only).
7. **`etperpoll<a>`/`packetper<a>` are consumed at every stake** for shop/pack jokers; only the
   rental roll is gated on the modifier. Irrelevant for parity at White stake (own key), but
   any engine that later models Black+ stakes must not add a roll that wasn't there.
8. **Negative edition is not scaled by `edition_rate`** (Hone/Glow Up); poly/holo/foil are.
9. **Boss draw is alphabetical** over the min-usage-filtered eligible set; `boss.max` is dead.
10. `create_card_for_shop` lives in `functions/UI_definitions.lua:742-800` (now extracted into
    `_reference`; I originally read it from the exe zip -- byte-identical at those lines).
11. **The game runs LuaJIT 2.0.5, not 2.1** (`lua51.dll` next to `Balatro.exe`; `jit.version`).
    Checked Agent A's core against that DLL via ctypes (verbatim misc_functions.lua primitives):
    seeded `math.random` sequences, the keyed chain, `pseudoshuffle`, `pseudorandom_element`
    and even the unseeded first draw (0.79420629243124097) all match to `%.17g`. The only
    observable 2.0.5-vs-2.1 difference is string-hash / `pairs` order, which 2.1 randomises
    per VM and 2.0.5 fixes -- hence `HANDS_PAIRS_ORDER` had to come from the DLL, not lupa.
12. `pairs(G.GAME.hands)` order is now a constant (`generate.HANDS_PAIRS_ORDER`) with helpers
    `visible_hands_in_pairs_order`, `to_do_hand`, `orbital_hand`.

## 3. Assumptions baked into `generate.py`

| Assumption | Where | Risk |
|---|---|---|
| Profile is fully unlocked and fully discovered by default (`locked_keys`/`undiscovered_keys` empty) | `RunState` defaults | MLB ranked assumption; public seed analyzers do the same. `RunState.fresh_profile()` reproduces a new save (validated against the Lua fresh-profile run). Any *specific* player profile needs its own lock/discovery sets. |
| `ante` is an int; keys use `str(int)` | `_ante_str` | Lua `..number` uses `%.14g`; integral antes print identically. Ante 0 (Hieroglyph at ante 1) gives `...0` keys on both sides. |
| `hands_played` is the only input to Planet softlock / Telescope | `get_current_pool`, `open_pack` | Lua also needs `hands[h].visible` for Telescope; I use `visible or played > 0`. |
| `deck_enhancements` (set of enhancement keys on any playing card) drives `enhancement_gate` | `get_current_pool` | Must be maintained by the engine (Tarot conversions, Standard packs, Spectral creates, Marble, Certificate). |
| `showman` is a bool the engine maintains (owned + not debuffed) | `RunState.showman` | `acquire('j_ring_master')` sets it; debuff by a boss (e.g. The Plant? no -- Verdant Leaf debuffs all jokers) must clear it for the duration. |
| Tag hooks are modelled as an ordered list of tag keys + `triggered_tags` index set per shop | `_fill_shop_slot`, `generate_shop` | Matches tag.lua's `triggered` flag for a single shop; multi-shop persistence of untriggered tags is handled by the engine re-seeding `state.tags`. |
| Forced-first-Buffoon suffix `_1` | `get_pack` | Cosmetic; compare kind+size. |
| Deck effects table covers the 15 base decks' generation-relevant effects only | `DECK_EFFECTS` | Painted/Anaglyph/Plasma/Black/Blue/Yellow/Green have none. Challenge decks not modelled. |
| `P_CENTER_POOLS.Tarot_Planet` order = stable sort by `order` | `_tarot_planet_pool` | Unspecified in Lua; never drawn from. |

## 4. Not modelled / untested

* ~~`to_do` / `orbital` candidate order~~ -- resolved (finding 12). Remaining caveat: a
  reloaded save rebuilds `G.GAME.hands` through STR_UNPACK (different insertion order), so the
  constant is for never-reloaded runs; `to_do_hand`/`orbital_hand` themselves are not
  oracle-checked beyond the order constant.
* Boss-time and play-time rolls (`hook`, `cerulean_bell`, `crimson_heart`, `wheel`, `aajk`,
  `glass`, `lucky_*`, ...): helpers/inventory only (`prob_roll`, `PROBABILITY_ROLLS`); no
  validation beyond reading.
* Copy-type effects (Perkeo `perkeo`, Invisible Joker `invisible`, Ankh `ankh_choice`, Madness
  `madness`, Wheel/Ectoplasm/Hex target picks): helpers provided, not oracle-checked (they need
  real Card lists; semantics are plain `pseudorandom_element` over the area in sort_id order).
* Overstock bought mid-shop (immediate extra `create_card_for_shop`), Hieroglyph ante-0 keys,
  challenge modifiers (`all_eternal`, `flipped_cards`, `perscribed_bosses`, `banned_*`),
  saved-game reload: implemented from reading, untested.
* `G.FORCE_BOSS` / `G.FORCE_TAG`: passthrough fields, untested.
* The end-to-end *sequencing* listed in section 1.

## 5. What Agent D's ground truth must contain (per function)

Minimum per seed (fully unlocked profile, Red deck, White stake), ante 1-3:

| To validate | Ground truth needed |
|---|---|
| `start_run` / `next_boss` / `next_voucher` / `next_tag` | ante-1 boss, ante-1 voucher, Small/Big tags; then per ante: boss, voucher, tags **as shown at the blind-select screen before any skip** |
| `generate_shop` + `create_card_for_shop` + `create_card` | for each of the 3 shops per ante: the `joker_max` shelf cards **in slot order** with edition; whether the player bought anything (and what) before the next shop -- purchases change later pools via `used_jokers` |
| `reroll_shop` | at least one shop with N rerolls: each reshelf in order; nothing else bought in between (or record it) |
| `get_pack` | both pack slots per shop (kind + size suffice; art index optional) |
| `open_pack` | contents in creation order for at least one pack of each kind (Arcana, Celestial, Spectral, Standard, Buffoon); Standard cards need rank/suit, enhancement, edition, seal; Buffoon cards need edition |
| `used_jokers` lifecycle | a run where a shelf card is bought and the next shop / next pack is recorded (the bought key must be absent; its slot is resampled); and a run where a card is left on the shelf, the shop is rerolled, and the same card reappears later |
| Showman | a run owning Showman where a duplicate of an owned joker appears |
| Soul / Black Hole | any pack where they spawned (needs many seeds: 0.3% per card) |
| eligibility filters | a seed with Gros Michel destroyed (Cavendish appears), a Planet X/Ceres/Eris appearance after the hand was played, a voucher tier-2 appearance after tier-1 redeemed |
| shuffle | the first 8 cards dealt at the ante-1 Small blind for a Red deck (validates `shuffle` + `nr1` ordering and the sort_id convention) |

Format suggestion: one JSON per seed with `events: [{ante, blind, kind: "shop"|"reroll"|"pack"|"buy"|"skip"|"boss"|"voucher"|"tags", ...}]` in chronological order, so a replayer can feed `acquire`/`release_*` at the right moments. Without the purchase/leave events interleaved, shop 2 of an ante cannot be validated.

Public seed analyzers (Immolate / TheSoul / Blueprint) model exactly the fully-unlocked profile
and this `used_jokers` semantics (they call it the "shop queue" with "blockers"); their output
for the same seeds is the cheapest second oracle for everything in this table except Showman.

## 6. Interface notes for Phase 1

* All functions take a `RunState`; nothing else is global. `RunState.clone()` deep-copies
  collections and clones the `PseudoRandom` for tree search.
* `create_card(...)` returns a `CardGen`; the engine turns it into its own card object and is
  responsible for `acquire`/`release_*`.
* `generate_shop` expects `state.tags` (tag keys, oldest first) and resets `triggered_tags`.
  Uncommon/Rare tags fill slots; edition tags modify; Voucher tags append to `tag_vouchers`.
* `defeat_boss` does voucher -> tags -> boss in Lua order (different keys, so order is only
  cosmetic, but keep it).
* Keys: build them only via `Keys` (or `keys.py`); never concatenate ad hoc.
* If `pools.py`'s data shape changes, the touch points are `center()`, `center_set()`,
  `get_current_pool` (fields `requires`, `softlock`, `hand_type`, `enhancement_gate`,
  `no_pool_flag`, `yes_pool_flag`, `rarity`), `get_pack` (`kind`, `weight`), `open_pack`
  (`kind`, `extra`), `eligible_bosses` (`showdown`, `boss_min`), `next_tag` (`requires`,
  `min_ante`).

## 7. Coordinator decisions (closed)

1. Harness promoted to `mp/tests/test_generate_oracle.py` (done).
2. `pairs(G.GAME.hands)` order dumped from the game's own DLL and recorded as
   `generate.HANDS_PAIRS_ORDER` with a DLL-backed assertion test (done).
3. MLB `banned_keys`: out of scope; `RunState.banned_keys` is the hook (TODO comment in
   `generate.py`), every generation path already honours it. Phase 2 fills it, along with any
   MLB changes to `win_ante`, `shop.joker_max` or rates.

## 8. Still open

* Agent D's end-to-end ground truth for the cross-state sequencing (section 1 / spec 18).
* Whether the `Card` lifecycle stub misses any `Card:remove` path that matters (e.g. cards
  destroyed by bosses or jokers also release `used_jokers`; modelled by `remove_owned`).
