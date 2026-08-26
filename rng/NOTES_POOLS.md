# NOTES_POOLS — Balatro 1.0.1o item pools, ordering, and keys

Agent B deliverable notes for `rng/pools.py` and `rng/keys.py`. Source: the game Lua
extracted from the local `Balatro.exe` (1.0.1o-FULL). Line numbers refer to the files under
`_reference/balatro_src/` unless marked NOT EXTRACTED.

## 0. Reference extraction gap (action for the lead)

`_reference/balatro_src/` holds 15 of the 32 Lua files in the exe. Missing and relevant:

| file | why it matters |
|---|---|
| `functions/UI_definitions.lua` (350 KB) | defines **`create_card_for_shop`** (the `'cdt'..ante` slot-type roll, the `'sho'` append, the Illusion edition logic) and the **Orbital Tag** `'orbital'` draw. `create_card_for_shop` is *called* at game.lua:3112, button_callbacks.lua:2884, common_events.lua:1114 but defined nowhere in the extracted set. |
| `engine/event.lua` | event-queue ordering (decides that `ease_ante(1)` runs before the post-boss voucher draw) |
| `card_character.lua`, `engine/*.lua` | cosmetic / no keyed RNG |

The exe is a LÖVE fused binary; Python's `zipfile` opens it directly
(`zipfile.ZipFile(r"C:\Program Files (x86)\Steam\steamapps\common\Balatro\Balatro.exe")`).
I read the missing files from a scratchpad extraction and did **not** add them to
`_reference/` (outside my ownership). `keys.py` cites `functions/UI_definitions.lua:766,772,786,787,1515`
from that copy. Recommend extracting at least `functions/UI_definitions.lua` into `_reference/balatro_src/functions/`.

## 1. How pools are built and ordered (game.lua:216-843)

1. `P_CENTERS`, `P_BLINDS`, `P_TAGS`, `P_SEALS`, `P_STAKES`, `P_CARDS` are **hash tables** literal
   in the source; each entry carries a hand-assigned integer `order` (no formula: it is a field
   in the literal, so "how order is assigned" = "typed by the developer").
2. `P_CENTER_POOLS[set]` are filled by `for k, v in pairs(self.P_CENTERS)` (game.lua:814-823):
   `Joker` gets every `set == 'Joker'`; every other set goes to `P_CENTER_POOLS[v.set]` unless
   `v.wip`, `v.skip_pool` or `v.omit` (only `b_challenge` has `omit`); Tarot+Planet also go to
   `Tarot_Planet`; anything with `consumeable` also goes to `Consumeables`; jokers with `rarity`
   go to `P_JOKER_RARITY_POOLS[rarity]`.
3. **The sort rule** (game.lua:826-842): `table.sort(pool, function(a, b) return a.order < b.order end)`
   for Joker, Tarot, Planet, Tarot_Planet, Spectral, Voucher, Booster, Consumeables, Enhanced,
   Stake, Tag, Seal and each rarity pool. Exceptions:
   * `Back`: `(a.order - (a.unlocked and 100 or 0)) < (b.order - (b.unlocked and 100 or 0))`, i.e. unlocked decks first. Irrelevant in practice: both "fresh profile" (only b_red unlocked) and "all unlocked" collapse to plain `order`.
   * `Demo`: `order + 1000*(set=='Joker')`; empty pool in release.
   * **`Edition` and `Default` are never sorted** (hash order). Neither is drawn from.
4. Lua's `table.sort` is **not stable**, and `pairs()` order over a hash table is
   implementation-defined, so a pool is only reproducible when its `order` values are unique.
   Verified by executing the real function in LuaJIT 2.1 (lupa) and checking: **unique for every
   pool that is ever drawn from** (Joker, Joker1-4, Tarot, Planet, Spectral, Voucher, Booster,
   Enhanced, Tag, Seal, Stake, Back). `Tarot_Planet` (Tarot and Planet both start at order 1)
   and `Consumeables` (Tarot/Planet/Spectral all overlap 1-18) have duplicate orders and therefore
   a launch-dependent sort, but `Tarot_Planet` is only reached through The Fool with a *forced key*
   (card.lua:1377) and `Consumeables` is never used for generation, so nothing depends on them.
   `pools.py` deliberately omits both.
5. Method: `init_item_prototypes` (lines 217-842) was loaded as a function body under LuaJIT
   2.1.1774896198 (`lupa.luajit21`) with stubs for `localize`, `HEX`, `love.filesystem`,
   `G.SETTINGS.profile`, `STR_UNPACK`, `get_compressed`, `convert_save_to_meta`, `self:save_progress`,
   `_RELEASE_MODE`; the resulting arrays were dumped to JSON and emitted as Python literals by
   `scratchpad/gen_pools.py`. No ordering in `pools.py` was typed by hand. (Scripts live in the
   session scratchpad, not in `mp/`; the procedure above is enough to regenerate.)

### Counts (all as expected)

| pool | n | notes |
|---|---|---|
| Joker | 150 | orders 1..150 contiguous; rarity 1/2/3/4 = 61/64/20/5 |
| Tarot | 22 | 1..22 |
| Planet | 12 | 1..12 |
| Spectral | 18 | 1..18; c_soul (17) and c_black_hole (18) always UNAVAILABLE |
| Voucher | 32 | 1..32, base/plus **interleaved** |
| Booster | 32 centers | 13 distinct pack types; see surprise 4 |
| Enhanced | 8 | orders 2..9 (c_base is set 'Default', not in pool) |
| Edition | 5 | unsorted, never drawn |
| Seal | 4 | Gold 1, Red 2, Blue 3, Purple 4; never drawn (threshold pick) |
| Tag | 24 | 1..24 |
| Back | 15 | + b_challenge (omit) |
| Stake | 8 | |
| Blinds | 30 | small, big + 28 bosses = 23 regular + 5 showdown |
| P_CARDS | 52 | never mutated; `G.P_CARDS.empty` is nil |
| P_LOCKED (fresh) | 75 | 45 jokers (order 106-150) + 16 vouchers + 14 backs |

## 2. Draw-time semantics that affect pool *contents* (common_events.lua:1963-2053)

* `get_current_pool` walks the sorted pool with `ipairs` and writes either the key or the string
  `'UNAVAILABLE'` at the **same index**. The pool length never changes; ineligible slots are
  resolved by `pseudorandom_element(pool, pseudoseed(key..'_resample'..it))`, it = 2, 3, ...
  (each retry a fresh stream). If every slot is UNAVAILABLE the pool is replaced by a single
  fallback (`pools.EMPTY_POOL_FALLBACK`).
* Eligibility (non-Tag/non-Enhanced): not in `G.GAME.used_jokers` unless a Showman is held;
  `v.unlocked ~= false or v.rarity == 4`; vouchers need all `requires` in `used_vouchers`, not
  already redeemed, not already on display in `G.shop_vouchers`; softlock planets need the hand
  played; `enhancement_gate` needs a matching card in the full deck; The Soul / Black Hole by
  name always false; `no_pool_flag` / `yes_pool_flag` (`gros_michel_extinct`); `banned_keys`.
* Tag eligibility: `requires` center must be **discovered** on the profile (`tag_rare` <- `j_blueprint`,
  `tag_negative` <- `e_negative`, foil/holo/polychrome <- their editions) and `min_ante <= ante`
  (`None` or 2). A fresh profile therefore has 6 tags UNAVAILABLE at first; a full profile has all 24 eligible from ante 2.
* `used_jokers[k]` is set by **name** in `Card:set_ability` (card.lua:349-354) for *every* card
  created outside the overlay menu, so a joker sitting unsold in the shop already blocks itself
  in packs opened during that shop, and is cleared in `Card:remove` only when no card of that name is left.
* Enhanced pool: never culled. `create_card('Base')` forces `c_base`, no pool draw.

## 3. Surprises

1. **Boss order is alphabetical, not `order`.** `get_new_boss` builds `eligible_bosses` as a hash
   table keyed by blind key; `pseudorandom_element` sorts hash tables by key string (values are
   numbers, so no `sort_id` path). Effective pool = eligible keys in byte-wise order, e.g. at ante 1:
   `bl_club, bl_goad, bl_head, bl_hook, bl_manacle, bl_pillar, bl_psychic, bl_window` (8 with
   `boss.min == 1`). Among eligible bosses only those with the minimum `bosses_used` count survive.
   `boss.max` (always 10) is never read. Showdown bosses are eligible iff `ante % win_ante == 0`
   and `ante >= 2`; regular bosses iff `min <= max(1, ante)` and not a showdown ante.
2. **Vouchers interleave**: order 1 overstock_norm, 2 overstock_plus, 3 clearance_sale,
   4 liquidation ... so (1-based) odd indices are base vouchers and even indices their upgrades.
   Key names: `v_overstock_norm` (not `v_overstock`), `v_directors_cut`, `v_reroll_surplus`.
3. **Voucher Tag** uses pool key `'Voucher_fromtag'`; it *replaces* the whole key, so no ante suffix.
4. **Booster pool = 32 centers** (art variants `_1.._4`), sorted by order with **Spectral last**
   (29-32) although it is declared before Standard in the Lua. Weighted walk, not element draw:
   total weight 22.42 (Arcana 6.5, Celestial 6.5, Standard 6.5, Buffoon 1.95, Spectral 0.97).
   The very first pack slot of a run is forced to `p_buffoon_normal_<math.random(1,2)>` *without*
   touching the `'shop_pack'` stream (`first_shop_buffoon`, common_events.lua:1945-1948).
5. **`P_CARDS` draws are key-string ordered**: `C_2..C_9, C_A, C_J, C_K, C_Q, C_T, D_..., H_..., S_...`
   (`pools.PLAYING_CARD_KEYS`). Affects `'front..'`, `'erratic'`, `'cert_fr'`, `'marb_fr'`.
   The starting deck is built from the same keys sorted by `suit..rank` string (game.lua:2367)
   before the first `'shuffle'`, so the pre-shuffle deck order is deterministic even though the
   build loop is `pairs(P_CARDS)`.
6. **Card-array pools sort by `sort_id`** (global creation counter, card.lua:24), not by area
   position: hook, cerulean_bell, crimson_heart, random_destroy, ankh_choice, invisible, perkeo,
   madness, idol/mail/cas, wheel_of_fortune/ectoplasm/hex. `pseudoshuffle` also re-sorts by
   `sort_id` first, which overrides Immolate's explicit `playing_card` pre-sort.
7. **Hash-ordered pools that cannot be derived from source**: `'to_do'` (To Do List) and
   `'orbital'` (Orbital Tag) build their candidate list with `pairs(G.GAME.hands)`. Modern LuaJIT
   2.1 randomises the string-hash seed per process, so these two may differ between launches of
   the real game; treat them as oracle-only.
8. **Rarity roll is consumed even for The Soul**: `local rarity = _rarity or pseudorandom('rarity'..ante..append)`
   runs before `_legendary` forces 4, so `'rarity<ante>sou'` advances once per Soul use. Callers
   with a fixed `_rarity` (Wraith 0.99, Riff-raff 0, Top-up 0, Rare Tag 1, Uncommon Tag 0.9) skip it.
9. **Legendary pool key has no ante and no append**: always `'Joker4'`. The gold-stake seed
   filter `get_first_legendary(seed)` uses the stateless `pseudoseed('Joker4', seed)` and equals
   the in-run first legendary.
10. **Spectral packs roll `'soul_Spectral<ante>'` twice per card** (soul check, then black-hole
    check on the same key); a black-hole hit overrides.
11. **`'8ba'` is shared** by 8 Ball and the Purple Seal; `'wheel_of_fortune'` is one stream used
    three times per Wheel (success, target joker, edition).
12. **Rarity vs. wiki**: all 150 rarities in `pools.py` are the game's; ones people get
    wrong: Stuntman and Driver's License are Rare at cost 7; Blueprint/Brainstorm cost 10; Credit Card cost 1;
    Joker cost 2; Oops! All 6s Uncommon cost 4; Golden Joker Common; Half Joker Common.
13. Key spellings: `j_gluttenous_joker` (sic), `c_heirophant` (sic), `j_selzer`, `j_todo_list`,
    `j_ring_master` (Showman), `j_ticket` (Golden Ticket), `j_trousers`, `j_mail`, `j_gift`,
    `j_trading`, `j_square`, `j_stone`, `j_glass`, `j_smeared`, `j_invisible`, `j_burnt`, `j_idol`,
    `j_business`, `j_space`, `j_wee`, `j_delayed_grat`, `j_8_ball`.
14. Eternal/perishable poll `'etperpoll'..ante` is rolled for **every** shop/pack joker at any
    stake (the stream advances even when stickers are disabled); rental only when enabled.
15. Deck configs: Checkered (`config = {}`) and Anaglyph (`config = {}`) are special-cased by
    *name* in back.lua:239 / back.lua:111; Plasma by name in back.lua:121,125. Magic/Ghost starting
    consumables use a forced key with append `'deck'`; no RNG consumed.
16. Stakes: `P_STAKES` has no modifier fields; the ladder lives in `Game:start_run`
    game.lua:2050-2059 and is cumulative (`>=`). Ported into `pools.STAKES[i]["cumulative"]`.

## 4. Mapping: game key -> `balatro_sim` catalogue (read-only audit)

Sim sources: `balatro_sim/shop.py` (`JOKER_CATALOGUE`, `BOOSTER_CATALOGUE`),
`balatro_sim/consumables.py` (`TAROT_NAME`, `PLANET_NAME`, `SPECTRAL_NAME`, `VOUCHER_NAME`),
`balatro_sim/constants.py`, `balatro_sim/game.py` (boss lists). Matching is by exact key, else by
normalised name. "sim rarity/cost" is the value that survives the `_reg` overwrites (dict, last wins).

### 4.1 Jokers

7/150 entries agree on key, rarity and cost; 21 renamed keys; 79 wrong rarities; 107 wrong costs; 0 missing by name; 6 sim keys map to no game joker (`j_the_duo`, `j_the_family`, `j_the_order`, `j_the_tribe`, `j_the_trio`, `j_wee_joker`).

The sim prices by rarity tier ($6/$7/$8/$20) instead of per-card cost, and most of the 2026-07-29
"audit M2" additions were registered as Common, so rarity is wrong for 79 of 150. Eleven keys are
registered twice (`j_abstract`, `j_half`, `j_odd_todd`, `j_stencil`, `j_flash`, `j_drivers_license`
and the five legendaries); the later registration wins. Six sim keys are duplicates of real entries
under a second spelling (`j_the_duo/trio/family/order/tribe` vs `j_duo/...`, `j_wee_joker` vs `j_wee`):
the `j_the_*` copies sit in the **Rare** block with the right rarity while the `j_duo`-style keys are
wrongly Uncommon, so the catalogue double-counts those five and mis-tiers one copy of each.

| order | game key | name | game rarity | game cost | sim key | mismatch |
|---|---|---|---|---|---|---|
| 1 | `j_joker` | Joker | Common | 2 | `j_joker` | cost sim=6 |
| 2 | `j_greedy_joker` | Greedy Joker | Common | 5 | `j_greedy_mult` | renamed; cost sim=6 |
| 3 | `j_lusty_joker` | Lusty Joker | Common | 5 | `j_lusty_mult` | renamed; cost sim=6 |
| 4 | `j_wrathful_joker` | Wrathful Joker | Common | 5 | `j_wrathful_mult` | renamed; cost sim=6 |
| 5 | `j_gluttenous_joker` | Gluttonous Joker | Common | 5 | `j_gluttonous_mult` | renamed; cost sim=6 |
| 6 | `j_jolly` | Jolly Joker | Common | 3 | `j_jolly` | cost sim=6 |
| 7 | `j_zany` | Zany Joker | Common | 4 | `j_zany` | cost sim=6 |
| 8 | `j_mad` | Mad Joker | Common | 4 | `j_mad` | cost sim=6 |
| 9 | `j_crazy` | Crazy Joker | Common | 4 | `j_crazy` | cost sim=6 |
| 10 | `j_droll` | Droll Joker | Common | 4 | `j_droll` | cost sim=6 |
| 11 | `j_sly` | Sly Joker | Common | 3 | `j_sly` | cost sim=6 |
| 12 | `j_wily` | Wily Joker | Common | 4 | `j_wily` | cost sim=6 |
| 13 | `j_clever` | Clever Joker | Common | 4 | `j_clever` | cost sim=6 |
| 14 | `j_devious` | Devious Joker | Common | 4 | `j_devious` | cost sim=6 |
| 15 | `j_crafty` | Crafty Joker | Common | 4 | `j_crafty` | cost sim=6 |
| 16 | `j_half` | Half Joker | Common | 5 | `j_half` | rarity sim=Uncommon; cost sim=7; registered x2 (last wins) |
| 17 | `j_stencil` | Joker Stencil | Uncommon | 8 | `j_stencil` | rarity sim=Rare; registered x2 (last wins) |
| 18 | `j_four_fingers` | Four Fingers | Uncommon | 7 | `j_four_fingers` | rarity sim=Common; cost sim=6 |
| 19 | `j_mime` | Mime | Uncommon | 5 | `j_mime` | rarity sim=Common; cost sim=6 |
| 20 | `j_credit_card` | Credit Card | Common | 1 | `j_credit_card` | cost sim=6 |
| 21 | `j_ceremonial` | Ceremonial Dagger | Uncommon | 6 | `j_ceremonial` | rarity sim=Common |
| 22 | `j_banner` | Banner | Common | 5 | `j_banner` | cost sim=6 |
| 23 | `j_mystic_summit` | Mystic Summit | Common | 5 | `j_mystic_summit` | cost sim=6 |
| 24 | `j_marble` | Marble Joker | Uncommon | 6 | `j_marble` | rarity sim=Common |
| 25 | `j_loyalty_card` | Loyalty Card | Uncommon | 5 | `j_loyalty_card` | rarity sim=Common; cost sim=6 |
| 26 | `j_8_ball` | 8 Ball | Common | 5 | `j_8_ball` | cost sim=6 |
| 27 | `j_misprint` | Misprint | Common | 4 | `j_misprint` | cost sim=6 |
| 28 | `j_dusk` | Dusk | Uncommon | 5 | `j_dusk` | rarity sim=Common; cost sim=6 |
| 29 | `j_raised_fist` | Raised Fist | Common | 5 | `j_raised_fist` | cost sim=6 |
| 30 | `j_chaos` | Chaos the Clown | Common | 4 | `j_chaos` | cost sim=6 |
| 31 | `j_fibonacci` | Fibonacci | Uncommon | 8 | `j_fibonacci` | rarity sim=Common; cost sim=6 |
| 32 | `j_steel_joker` | Steel Joker | Uncommon | 7 | `j_steel_joker` | rarity sim=Common; cost sim=6 |
| 33 | `j_scary_face` | Scary Face | Common | 4 | `j_scary_face` | cost sim=6 |
| 34 | `j_abstract` | Abstract Joker | Common | 4 | `j_abstract` | rarity sim=Uncommon; cost sim=7; registered x2 (last wins) |
| 35 | `j_delayed_grat` | Delayed Gratification | Common | 4 | `j_delayed_grat` | cost sim=6 |
| 36 | `j_hack` | Hack | Uncommon | 6 | `j_hack` | rarity sim=Common |
| 37 | `j_pareidolia` | Pareidolia | Uncommon | 5 | `j_pareidolia` | rarity sim=Common; cost sim=6 |
| 38 | `j_gros_michel` | Gros Michel | Common | 5 | `j_gros_michel` | cost sim=6 |
| 39 | `j_even_steven` | Even Steven | Common | 4 | `j_even_steven` | cost sim=6 |
| 40 | `j_odd_todd` | Odd Todd | Common | 4 | `j_odd_todd` | rarity sim=Uncommon; cost sim=7; registered x2 (last wins) |
| 41 | `j_scholar` | Scholar | Common | 4 | `j_scholar` | cost sim=6 |
| 42 | `j_business` | Business Card | Common | 4 | `j_business_card` | renamed; cost sim=6 |
| 43 | `j_supernova` | Supernova | Common | 5 | `j_supernova` | cost sim=6 |
| 44 | `j_ride_the_bus` | Ride the Bus | Common | 6 | `j_ride_the_bus` | ok |
| 45 | `j_space` | Space Joker | Uncommon | 5 | `j_space_joker` | renamed; rarity sim=Common; cost sim=6 |
| 46 | `j_egg` | Egg | Common | 4 | `j_egg` | cost sim=6 |
| 47 | `j_burglar` | Burglar | Uncommon | 6 | `j_burglar` | rarity sim=Common |
| 48 | `j_blackboard` | Blackboard | Uncommon | 6 | `j_blackboard` | rarity sim=Common |
| 49 | `j_runner` | Runner | Common | 5 | `j_runner` | cost sim=6 |
| 50 | `j_ice_cream` | Ice Cream | Common | 5 | `j_ice_cream` | cost sim=6 |
| 51 | `j_dna` | DNA | Rare | 8 | `j_dna` | rarity sim=Common; cost sim=6 |
| 52 | `j_splash` | Splash | Common | 3 | `j_splash` | cost sim=6 |
| 53 | `j_blue_joker` | Blue Joker | Common | 5 | `j_blue_joker` | cost sim=6 |
| 54 | `j_sixth_sense` | Sixth Sense | Uncommon | 6 | `j_sixth_sense` | rarity sim=Common |
| 55 | `j_constellation` | Constellation | Uncommon | 6 | `j_constellation` | rarity sim=Common |
| 56 | `j_hiker` | Hiker | Uncommon | 5 | `j_hiker` | rarity sim=Common; cost sim=6 |
| 57 | `j_faceless` | Faceless Joker | Common | 4 | `j_faceless` | cost sim=6 |
| 58 | `j_green_joker` | Green Joker | Common | 4 | `j_green_joker` | cost sim=6 |
| 59 | `j_superposition` | Superposition | Common | 4 | `j_superposition` | cost sim=6 |
| 60 | `j_todo_list` | To Do List | Common | 4 | `j_to_do_list` | renamed; cost sim=6 |
| 61 | `j_cavendish` | Cavendish | Common | 4 | `j_cavendish` | cost sim=6 |
| 62 | `j_card_sharp` | Card Sharp | Uncommon | 6 | `j_card_sharp` | rarity sim=Common |
| 63 | `j_red_card` | Red Card | Common | 5 | `j_red_card` | cost sim=6 |
| 64 | `j_madness` | Madness | Uncommon | 7 | `j_madness` | rarity sim=Common; cost sim=6 |
| 65 | `j_square` | Square Joker | Common | 4 | `j_square_joker` | renamed; cost sim=6 |
| 66 | `j_seance` | Seance | Uncommon | 6 | `j_seance` | rarity sim=Common |
| 67 | `j_riff_raff` | Riff-raff | Common | 6 | `j_riff_raff` | ok |
| 68 | `j_vampire` | Vampire | Uncommon | 7 | `j_vampire` | rarity sim=Common; cost sim=6 |
| 69 | `j_shortcut` | Shortcut | Uncommon | 7 | `j_shortcut` | rarity sim=Common; cost sim=6 |
| 70 | `j_hologram` | Hologram | Uncommon | 7 | `j_hologram` | rarity sim=Common; cost sim=6 |
| 71 | `j_vagabond` | Vagabond | Rare | 8 | `j_vagabond` | rarity sim=Common; cost sim=6 |
| 72 | `j_baron` | Baron | Rare | 8 | `j_baron` | rarity sim=Common; cost sim=6 |
| 73 | `j_cloud_9` | Cloud 9 | Uncommon | 7 | `j_cloud_9` | rarity sim=Common; cost sim=6 |
| 74 | `j_rocket` | Rocket | Uncommon | 6 | `j_rocket` | rarity sim=Common |
| 75 | `j_obelisk` | Obelisk | Rare | 8 | `j_obelisk` | rarity sim=Common; cost sim=6 |
| 76 | `j_midas_mask` | Midas Mask | Uncommon | 7 | `j_midas_mask` | rarity sim=Common; cost sim=6 |
| 77 | `j_luchador` | Luchador | Uncommon | 5 | `j_luchador` | rarity sim=Common; cost sim=6 |
| 78 | `j_photograph` | Photograph | Common | 5 | `j_photograph` | cost sim=6 |
| 79 | `j_gift` | Gift Card | Uncommon | 6 | `j_gift_card` | renamed; rarity sim=Common |
| 80 | `j_turtle_bean` | Turtle Bean | Uncommon | 6 | `j_turtle_bean` | rarity sim=Common |
| 81 | `j_erosion` | Erosion | Uncommon | 6 | `j_erosion` | rarity sim=Common |
| 82 | `j_reserved_parking` | Reserved Parking | Common | 6 | `j_reserved_parking` | ok |
| 83 | `j_mail` | Mail-In Rebate | Common | 4 | `j_mail_in_rebate` | renamed; cost sim=6 |
| 84 | `j_to_the_moon` | To the Moon | Uncommon | 5 | `j_to_the_moon` | cost sim=7 |
| 85 | `j_hallucination` | Hallucination | Common | 4 | `j_hallucination` | cost sim=6 |
| 86 | `j_fortune_teller` | Fortune Teller | Common | 6 | `j_fortune_teller` | ok |
| 87 | `j_juggler` | Juggler | Common | 4 | `j_juggler` | cost sim=6 |
| 88 | `j_drunkard` | Drunkard | Common | 4 | `j_drunkard` | cost sim=6 |
| 89 | `j_stone` | Stone Joker | Uncommon | 6 | `j_stone_joker` | renamed; rarity sim=Common |
| 90 | `j_golden` | Golden Joker | Common | 6 | `j_golden` | rarity sim=Uncommon; cost sim=7 |
| 91 | `j_lucky_cat` | Lucky Cat | Uncommon | 6 | `j_lucky_cat` | rarity sim=Common |
| 92 | `j_baseball` | Baseball Card | Rare | 8 | `j_baseball` | rarity sim=Common; cost sim=6 |
| 93 | `j_bull` | Bull | Uncommon | 6 | `j_bull` | rarity sim=Common |
| 94 | `j_diet_cola` | Diet Cola | Uncommon | 6 | `j_diet_cola` | rarity sim=Common |
| 95 | `j_trading` | Trading Card | Uncommon | 6 | `j_trading_card` | renamed; rarity sim=Common |
| 96 | `j_flash` | Flash Card | Uncommon | 5 | `j_flash` | rarity sim=Rare; cost sim=8; registered x2 (last wins) |
| 97 | `j_popcorn` | Popcorn | Common | 5 | `j_popcorn` | cost sim=6 |
| 98 | `j_trousers` | Spare Trousers | Uncommon | 6 | `j_spare_trousers` | renamed; cost sim=7 |
| 99 | `j_ancient` | Ancient Joker | Rare | 8 | `j_ancient` | rarity sim=Uncommon; cost sim=7 |
| 100 | `j_ramen` | Ramen | Uncommon | 6 | `j_ramen` | rarity sim=Common |
| 101 | `j_walkie_talkie` | Walkie Talkie | Common | 4 | `j_walkie_talkie` | cost sim=6 |
| 102 | `j_selzer` | Seltzer | Uncommon | 6 | `j_seltzer` | renamed; rarity sim=Common |
| 103 | `j_castle` | Castle | Uncommon | 6 | `j_castle` | rarity sim=Common |
| 104 | `j_smiley` | Smiley Face | Common | 4 | `j_smiley` | cost sim=6 |
| 105 | `j_campfire` | Campfire | Rare | 9 | `j_campfire` | rarity sim=Uncommon; cost sim=7 |
| 106 | `j_ticket` | Golden Ticket | Common | 5 | `j_golden_ticket` | renamed; cost sim=6 |
| 107 | `j_mr_bones` | Mr. Bones | Uncommon | 5 | `j_mr_bones` | rarity sim=Common; cost sim=6 |
| 108 | `j_acrobat` | Acrobat | Uncommon | 6 | `j_acrobat` | rarity sim=Common |
| 109 | `j_sock_and_buskin` | Sock and Buskin | Uncommon | 6 | `j_sock_and_buskin` | rarity sim=Common |
| 110 | `j_swashbuckler` | Swashbuckler | Common | 4 | `j_swashbuckler` | cost sim=6 |
| 111 | `j_troubadour` | Troubadour | Uncommon | 6 | `j_troubadour` | rarity sim=Common |
| 112 | `j_certificate` | Certificate | Uncommon | 6 | `j_certificate` | rarity sim=Common |
| 113 | `j_smeared` | Smeared Joker | Uncommon | 7 | `j_smeared_joker` | renamed; rarity sim=Common; cost sim=6 |
| 114 | `j_throwback` | Throwback | Uncommon | 6 | `j_throwback` | rarity sim=Common |
| 115 | `j_hanging_chad` | Hanging Chad | Common | 4 | `j_hanging_chad` | cost sim=6 |
| 116 | `j_rough_gem` | Rough Gem | Uncommon | 7 | `j_rough_gem` | rarity sim=Common; cost sim=6 |
| 117 | `j_bloodstone` | Bloodstone | Uncommon | 7 | `j_bloodstone` | rarity sim=Common; cost sim=6 |
| 118 | `j_arrowhead` | Arrowhead | Uncommon | 7 | `j_arrowhead` | rarity sim=Common; cost sim=6 |
| 119 | `j_onyx_agate` | Onyx Agate | Uncommon | 7 | `j_onyx_agate` | rarity sim=Common; cost sim=6 |
| 120 | `j_glass` | Glass Joker | Uncommon | 6 | `j_glass_joker` | renamed; rarity sim=Common |
| 121 | `j_ring_master` | Showman | Uncommon | 5 | `j_showman` | renamed; rarity sim=Common; cost sim=6 |
| 122 | `j_flower_pot` | Flower Pot | Uncommon | 6 | `j_flower_pot` | rarity sim=Common |
| 123 | `j_blueprint` | Blueprint | Rare | 10 | `j_blueprint` | cost sim=8 |
| 124 | `j_wee` | Wee Joker | Rare | 8 | `j_wee` | ok |
| 125 | `j_merry_andy` | Merry Andy | Uncommon | 7 | `j_merry_andy` | rarity sim=Common; cost sim=6 |
| 126 | `j_oops` | Oops! All 6s | Uncommon | 4 | `j_oops` | rarity sim=Common; cost sim=6 |
| 127 | `j_idol` | The Idol | Uncommon | 6 | `j_the_idol` | renamed; cost sim=7 |
| 128 | `j_seeing_double` | Seeing Double | Uncommon | 6 | `j_seeing_double` | cost sim=7 |
| 129 | `j_matador` | Matador | Uncommon | 7 | `j_matador` | ok |
| 130 | `j_hit_the_road` | Hit the Road | Rare | 8 | `j_hit_the_road` | rarity sim=Uncommon; cost sim=7 |
| 131 | `j_duo` | The Duo | Rare | 8 | `j_duo` | rarity sim=Uncommon; cost sim=7 |
| 132 | `j_trio` | The Trio | Rare | 8 | `j_trio` | rarity sim=Uncommon; cost sim=7 |
| 133 | `j_family` | The Family | Rare | 8 | `j_family` | rarity sim=Uncommon; cost sim=7 |
| 134 | `j_order` | The Order | Rare | 8 | `j_order` | rarity sim=Uncommon; cost sim=7 |
| 135 | `j_tribe` | The Tribe | Rare | 8 | `j_tribe` | rarity sim=Uncommon; cost sim=7 |
| 136 | `j_stuntman` | Stuntman | Rare | 7 | `j_stuntman` | rarity sim=Common; cost sim=6 |
| 137 | `j_invisible` | Invisible Joker | Rare | 8 | `j_invisible_joker` | renamed; rarity sim=Common; cost sim=6 |
| 138 | `j_brainstorm` | Brainstorm | Rare | 10 | `j_brainstorm` | rarity sim=Common; cost sim=6 |
| 139 | `j_satellite` | Satellite | Uncommon | 6 | `j_satellite` | rarity sim=Common |
| 140 | `j_shoot_the_moon` | Shoot the Moon | Common | 5 | `j_shoot_the_moon` | cost sim=6 |
| 141 | `j_drivers_license` | Driver's License | Rare | 7 | `j_drivers_license` | cost sim=8; registered x2 (last wins) |
| 142 | `j_cartomancer` | Cartomancer | Uncommon | 6 | `j_cartomancer` | rarity sim=Common |
| 143 | `j_astronomer` | Astronomer | Uncommon | 8 | `j_astronomer` | rarity sim=Common; cost sim=6 |
| 144 | `j_burnt` | Burnt Joker | Rare | 8 | `j_burnt_joker` | renamed; rarity sim=Common; cost sim=6 |
| 145 | `j_bootstraps` | Bootstraps | Uncommon | 7 | `j_bootstraps` | ok |
| 146 | `j_caino` | Caino | Legendary | 20 | `j_caino` | registered x2 (last wins) |
| 147 | `j_triboulet` | Triboulet | Legendary | 20 | `j_triboulet` | registered x2 (last wins) |
| 148 | `j_yorick` | Yorick | Legendary | 20 | `j_yorick` | registered x2 (last wins) |
| 149 | `j_chicot` | Chicot | Legendary | 20 | `j_chicot` | registered x2 (last wins) |
| 150 | `j_perkeo` | Perkeo | Legendary | 20 | `j_perkeo` | registered x2 (last wins) |

### 4.2 Consumables

| game key | sim key | note |
|---|---|---|
| `c_heirophant` | `c_hierophant` | renamed (game spelling is the typo); other 21 tarots match exactly |
| `c_<planet>` x12 | `pl_<planet>` | sim uses a `pl_` prefix for all planets; hand_type mapping agrees for all 12 |
| `c_<spectral>` x18 | `s_<spectral>` | sim uses an `s_` prefix for all spectrals, incl. `s_soul`, `s_black_hole` |

### 4.3 Vouchers (game 32, sim 27)

| game key | sim key | note |
|---|---|---|
| `v_overstock_norm` | `v_overstock` | renamed |
| `v_seed_money` | - | **missing** |
| `v_money_tree` | - | **missing** |
| `v_blank` | - | **missing** (also the empty-pool fallback voucher) |
| `v_antimatter` | - | **missing** |
| `v_retcon` | - | **missing** |
| other 26 | same key | ok; sim has no `requires` chain data (see `pools.VOUCHER_REQUIRES`) |

### 4.4 Boosters (game 13 types / 32 centers, sim 13 keys)

Sim keys are per type (`p_arcana`, `p_arcana_jumbo`, ...) with no weights; sizes/costs/picks agree
where present. **Missing**: `p_standard_mega` (8, 5 cards, choose 2) and `p_buffoon_mega` (8, 4 cards,
choose 2). Sim `generate_shop` picks packs with `rng.choice` (uniform over 13) instead of the weighted
walk over 32 centers (Mega packs 0.25/0.15/0.07, Buffoon 0.6, Spectral 0.3).

### 4.5 Boss blinds (game 28, sim 27)

| game key | sim key | note |
|---|---|---|
| `bl_final_acorn` | `bl_amber` | renamed |
| `bl_final_bell` | `bl_cerulean` | renamed |
| `bl_final_heart` | `bl_crimson` | renamed |
| `bl_final_leaf` | `bl_verdant` | renamed |
| `bl_final_vessel` | `bl_violet` | renamed |
| `bl_fish` | - | **missing** (mentioned in a comment as unmodelled but in no list) |
| `bl_house`, `bl_wheel`, `bl_mark` | same | present but in `UNMODELLED_BOSS_BLINDS`, excluded from selection |
| other 19 regular | same key | ok; sim has no `boss.min` ante gates and no min-use rotation |

### 4.6 Tags, decks, stakes, enhancements, editions, seals

* **Tags**: the sim has no tag catalogue at all (`synergy.py` "tags" are strategy labels). All 24 missing.
* **Decks / stakes**: no catalogue; `env_*.py` hard-code Red-deck defaults. All 15 / 8 missing.
* **Enhancements**: `constants.ENHANCEMENTS` uses short names (`"Wild"` vs game `m_wild` / "Wild Card")
  plus a `"None"` slot; order matches game order 2..9. No keys.
* **Editions**: `constants.EDITIONS` = None/Foil/Holographic/Polychrome/Negative vs game
  `e_base/e_foil/e_holo/e_polychrome/e_negative`; `_roll_edition` in shop.py is a plain `rng` roll,
  not the threshold chain.
* **Seals**: `constants.SEALS` = None/Gold/Red/Blue/Purple; game pool order is Gold, Red, Blue, Purple
  but the game picks by threshold (Red > 0.75 > Blue > 0.5 > Gold > 0.25 > Purple).
* **Blind rewards**: sim 3/4/5/8 agree with `P_BLINDS.dollars`.

## 5. Files

* `rng/pools.py` (generated data). Top-level names: `JOKERS`, `JOKERS_BY_RARITY`,
  `JOKER_POOL_RARITY_1..4`, `TAROTS`, `PLANETS`, `SPECTRALS`, `VOUCHERS`, `VOUCHER_REQUIRES`,
  `BOOSTERS`, `BOOSTER_TOTAL_WEIGHT`, `ENHANCEMENTS`, `EDITIONS` (+ poll thresholds), `SEALS`,
  `BLINDS`, `BOSS_BLINDS[_REGULAR|_SHOWDOWN]`, `BOSS_KEYS_ALPHA`, `TAGS`, `BACKS`, `BACK_CHALLENGE`,
  `STAKES`, `PLAYING_CARDS`, `PLAYING_CARD_KEYS`, `POKER_HANDS`, `HANDLIST`, `POOL_KEYS`,
  `EMPTY_POOL_FALLBACK`, `JOKER_RARITY_THRESHOLDS`, `SHOP_RATES_DEFAULT`, `SOUL_THRESHOLD`,
  `STICKER_THRESHOLDS`, `P_LOCKED_DEFAULT`, `pool(name)`.
* `rng/keys.py`: 77 key records (`KEYS`, `KEY_BY_NAME`), `KEY_APPENDS` (26), `FORCED_RARITY`,
  helpers `rarity_key`, `pool_key`, `resample_key`, `rarity_from_poll`, plus `RUN_START_SEQUENCE`,
  `ROUND_END_SEQUENCE`, `UNSEEDED_GAMEPLAY`.
