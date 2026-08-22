# Balatro 1.0.1o Generation Spec

**Owner:** Agent C. **Implements:** `mp/rng/generate.py`. **Depends on:** `core.py` (Agent A, RNG
primitives), `pools.py` (Agent B, ordered pools). **Status:** every algorithm below is ported and
cross-checked against the *real* Lua functions executing in LuaJIT 2.1 (see section 18).

All citations are `file:line` into `mp/_reference/balatro_src/` (1.0.1o). Port the algorithms;
never copy the Lua into deliverables.

---

## 0. The one rule that explains everything

`pseudoseed(key)` (misc_functions.lua:298-313) keeps **one LCG state per key string** in
`G.GAME.pseudorandom[key]`, seeded on first use from `pseudohash(key..seed)`. Every generation
event is a `pseudoseed`/`pseudorandom` call on some key; the key strings embed the ante and a
short "append" naming the event. Consequences:

1. **Only the order of calls on the *same* key matters.** Calls on different keys never
   interact. Whether the shop is generated before or after the voucher is irrelevant; whether
   slot 1 is generated before slot 2 (both `cdt1`) is essential.
2. **There is no "queue" data structure.** The n-th joker the `'Joker1sho1'` stream will ever
   produce is fixed by the seed; a reroll simply asks that stream for its next value. The
   "pointer" is `G.GAME.pseudorandom['Joker1sho1']`.
3. **Ineligible entries are replaced in place.** `get_current_pool` writes the literal string
   `'UNAVAILABLE'` into the slot of every ineligible item (common_events.lua:2034), so pool
   indices never shift. A draw that lands on `'UNAVAILABLE'` is redrawn from a *side* stream
   `pool_key..'_resample'..it` (section 4); the main stream advanced exactly once. Two players
   with different collections therefore see identical streams except at blocked slots.
4. **Global `math.random` is shared.** `pseudorandom` reseeds LuaJIT's global generator every
   call (misc_functions.lua:315-319). A handful of sites call `math.random` *unseeded*
   (section 17); their result depends on UI-level consumption and is unreproducible -- all of
   them are cosmetic.

### Primitive semantics that matter here

| Primitive | Lua | Behaviour |
|---|---|---|
| `pseudoseed(key)` | misc_functions.lua:298-313 | advance `state[key]`; return `(state[key] + hashed_seed)/2`. `key=='seed'` returns raw `math.random()`. |
| `pseudorandom(key[,m,n])` | :315-319 | string key -> `pseudoseed`; `math.randomseed(x)`; `math.random()` or `math.random(m,n)`. **`pseudorandom(pseudoseed(k))` is identical to `pseudorandom(k)`.** |
| `pseudorandom_element(t, seed)` | :253-268 | seed global RNG; collect `(k,v)` pairs; **sort by `v.sort_id` if the first value is a table with `sort_id`, else by key** (`a.k < b.k`); return `t[keys[math.random(#keys)]]`. Arrays -> index order; `G.P_CARDS` (string keys) -> key-string order `C_2..C_9,C_A,C_J,C_K,C_Q,C_T,D_2,...`; `eligible_bosses` (string keys, number values) -> alphabetical; Card lists -> `sort_id`. |
| `pseudoshuffle(list, seed)` | :206-217 | seed; **if `list[1].sort_id` then `table.sort` by sort_id first**; then `for i=#list,2,-1: j=math.random(i); swap(i,j)`. |

---

## 1. Glossary of state read by generation

| `G.GAME.*` field | Init (game.lua:1862-2017) | Mutated by |
|---|---|---|
| `pseudorandom.seed`, `.hashed_seed` | start_run :2164-2168 | -- |
| `round_resets.ante` | 1 | `ease_ante` (common_events.lua:191-222), Hieroglyph/Petroglyph (`-1`, card.lua:1952-1968) |
| `used_jokers` | `{}` | **`Card:set_ability` on every card creation** (card.lua:349-354); cleared in `Card:remove` (card.lua:4741-4747) -- section 6 |
| `used_vouchers` | `{}` | `Card:redeem` (card.lua:1822), deck vouchers (back.lua:175-178, 238-243), challenge vouchers |
| `bosses_used` | every boss key -> 0 | `get_new_boss` (+1) |
| `banned_keys` | `{}` | challenges (game.lua:2127-2145); the MP mod |
| `pool_flags` | `{}` | Gros Michel extinction (card.lua:3037) |
| `probabilities.normal` | 1 | Oops! All 6s x2 (card.lua:608-612) |
| `joker_rate/tarot_rate/planet_rate/playing_card_rate/spectral_rate` | 20/4/4/0/0 | vouchers (card.lua:1890-1909): Tarot Merchant 9.6, Tycoon 32; Planet same; Magic Trick/Illusion `playing_card_rate=4`; Ghost deck `spectral_rate=2`; challenge `no_shop_jokers` -> `joker_rate=0` |
| `edition_rate` | 1 | Hone 2, Glow Up 4 (card.lua:1900-1903) |
| `shop.joker_max` | 2 | Overstock/Overstock Plus +1 each (`change_shop_size`, common_events.lua:1097-1118) |
| `first_shop_buffoon` | nil | `get_pack` first call |
| `current_round.voucher` | start_run / boss defeat | `Card:redeem` sets nil |
| `current_round.used_packs` | `{}` per round (state_events.lua:301) | `update_shop`, `open_booster` -> `'USED'` |
| `modifiers.enable_eternals_in_shop / perishables / rentals` | stake >= 4 / 7 / 8 (game.lua:2050-2054) | -- |
| `G.jokers.cards`, `G.consumeables.cards` | -- | `find_joker` (misc_functions.lua:903-917) searches ONLY these two areas |

Prototype fields per item (all in `pools.py`): `order` (pool position), `rarity`, `unlocked`
(profile), `discovered` (profile), `requires` (vouchers: list of voucher keys; tags: one center
key), `min_ante` (tags), `softlock`+`hand_type` (planets), `enhancement_gate`,
`no_pool_flag`/`yes_pool_flag` (jokers), `weight`/`kind`/`extra`/`choose` (boosters),
`boss.min`/`boss.showdown` (blinds; `boss.max` exists but is **never read**).

---

## 2. Key inventory

`<a>` = `G.GAME.round_resets.ante` formatted as a Lua number (integer, no decimal point).
`<app>` = `key_append` (section 2.1). `r` = rarity digit 1-4.

| Key | Primitive | Site | Event |
|---|---|---|---|
| `cdt<a>` | pseudorandom | UI_definitions.lua:766 | shop slot type roll |
| `illusion` | pseudorandom | UI_definitions.lua:772, 786, 787 | Illusion: Enhanced-vs-Base (every slot when owned), edition gate, edition pick |
| `rarity<a><app>` | pseudorandom | common_events.lua:1969 | joker rarity (skipped when `_rarity` passed) |
| `Joker<r><app><a>` | pseudorandom_element | :1971, :2052, :2117 | joker pool draw |
| `Joker4` | pseudorandom_element | :1971, :2052 | legendary pool (no append, **no ante**) |
| `Tarot<app><a>`, `Planet<app><a>`, `Spectral<app><a>`, `Enhanced<app><a>` | pseudorandom_element | :1972, :2052, :2117 | consumable / enhancement pool draws |
| `Voucher<a>` | pseudorandom_element | :1903, :2052 | shop voucher |
| `Voucher_fromtag` | pseudorandom_element | :1903 | Voucher Tag (**no ante**) |
| `Tag<app><a>` | pseudorandom_element | :1916 | blind skip tags (`append` is never passed in 1.0.1o -> `Tag<a>`) |
| `<pool_key>_resample<it>` | pseudorandom_element | :1908, :1921, :2120 | redraw after `'UNAVAILABLE'`, `it` = 2,3,... |
| `edi<app><a>` | pseudorandom | :2149 via poll_edition | joker edition (every Joker create_card) |
| `front<app><a>` | pseudorandom_element over `G.P_CARDS` | :2124 | playing-card front (Base/Enhanced) |
| `soul_<Type><a>` | pseudorandom | :2091, :2097 | The Soul / Black Hole forced-key rolls (`<Type>` = Tarot, Planet, Spectral, Tarot_Planet) |
| `etperpoll<a>` / `packetper<a>` | pseudorandom | :2138 | eternal/perishable poll (shop / pack) -- **always rolled** for shop+pack jokers |
| `ssjr<a>` / `packssjr<a>` | pseudorandom | :2144 | rental poll (only if rentals enabled) |
| `shop_pack<a>` | pseudorandom | :1953 (`get_pack('shop_pack')`, game.lua:3148) | shop booster slots |
| `pack_generic<a>` | pseudorandom | :1953 | `get_pack(nil)` -- not called in 1.0.1o |
| `stdset<a>` | pseudorandom | card.lua:1759 | standard pack Enhanced-vs-Base |
| `standard_edition<a>` | pseudorandom (poll_edition mod 2, no_neg) | card.lua:1761 | standard pack edition |
| `stdseal<a>`, `stdsealtype<a>` | pseudorandom | card.lua:1764, 1766 | standard pack seal gate / type |
| `omen_globe` | pseudorandom | card.lua:1731 | Arcana pack: Spectral instead of Tarot |
| `boss` | pseudorandom_element (alphabetical) | common_events.lua:2379 | boss blind |
| `shuffle` | pseudoshuffle | game.lua:2383 via cardarea.lua:573 | run-start deck shuffle |
| `nr<a>` | pseudoshuffle | state_events.lua:344 | deck shuffle at every blind start |
| `cashout<a>` | pseudoshuffle | button_callbacks.lua:2918 | deck shuffle on Cash Out |
| `erratic` | pseudorandom_element over `G.P_CARDS` | game.lua:2342 | Erratic deck, 52 draws |
| `idol<a>`, `mail<a>`, `anc<a>`, `cas<a>` | pseudorandom_element | common_events.lua:2281, 2298, 2309, 2321 | Idol / Mail-In Rebate / Ancient Joker / Castle targets (run start + every round end) |
| `orbital` | pseudorandom_element over `pairs(G.GAME.hands)` (= `HANDS_PAIRS_ORDER`, visible only) | UI_definitions.lua:1515 | Orbital Tag hand, once per blind type per ante |
| `to_do`, `false_to_do` | pseudorandom_element over `pairs(G.GAME.hands)` (= `generate.HANDS_PAIRS_ORDER`, visible only) | card.lua:320, 2980 | To Do List hand (on creation and each round; redrawn while equal to the previous hand) |
| `aura`, `wheel_of_fortune`, `ectoplasm`, `hex`, `ankh_choice`, `sigil`, `ouija`, `random_destroy`, `familiar_create`, `grim_create`, `incantation_create`, `spe_card`, `immolate` | various | card.lua:1211-1480 | consumables, section 14 |
| `cert_fr`, `certsl`, `marb_fr`, `misprint`, `invisible`, `perkeo`, `madness`, `halu<a>`, `lucky_mult`, `lucky_money`, `glass`, `gros_michel`, `cavendish`, `8ball`, `business`, `bloodstone`, `parking`, `space`, `wheel`, `hook`, `cerulean_bell`, `crimson_heart`, `aajk`, `flipped_card`, `edition_deck` | various | card.lua / blind.lua / cardarea.lua / back.lua | joker & blind effects, section 15 |

### 2.1 `key_append` values and their callers

| append | call | `_rarity` | pool key |
|---|---|---|---|
| `sho` | shop slot, UI_definitions.lua:776 | rolled | `Joker<r>sho<a>`, `Tarotsho<a>`, `Planetsho<a>`, `Spectralsho<a>`, `Enhancedsho<a>`, front `frontsho<a>` |
| `buf` | Buffoon pack, card.lua:1774 | rolled | `Joker<r>buf<a>` |
| `ar1` / `ar2` | Arcana pack Tarot / Omen-Globe Spectral, card.lua:1734 / 1732 | -- | `Tarotar1<a>` / `Spectralar2<a>` |
| `pl1` | Celestial pack, card.lua:1752-1754 | -- | `Planetpl1<a>` |
| `spe` | Spectral pack, card.lua:1757 | -- | `Spectralspe<a>` |
| `sta` | Standard pack, card.lua:1759 | -- | `Enhancedsta<a>`, front `frontsta<a>` |
| `jud` | Judgement, card.lua:1418 | rolled | `Joker<r>jud<a>` |
| `sou` | The Soul, card.lua:1418 (`legendary=true`) | rolled then overridden to 4 | `Joker4` |
| `wra` | Wraith, card.lua:1457 | 0.99 -> Rare | `Joker3wra<a>` |
| `emp` / `pri` | Emperor / High Priestess, card.lua:1406 | -- | `Tarotemp<a>` / `Planetpri<a>` |
| `rif` | Riff-raff, card.lua:2535 | 0 -> Common | `Joker1rif<a>` |
| `top` | Top-up Tag, tag.lua:138 | 0 -> Common | `Joker1top<a>` |
| `rta` / `uta` | Rare / Uncommon Tag, tag.lua:356 / 370 | 1 -> Rare / 0.9 -> Uncommon | `Joker3rta<a>` / `Joker2uta<a>` |
| `8ba` | 8 Ball (card.lua:3115) **and** Purple Seal (card.lua:2260) | -- | `Tarot8ba<a>` (shared stream) |
| `hal`, `car`, `vag`, `sup` | Hallucination, Cartomancer, Vagabond, Superposition | -- | `Tarot<app><a>` |
| `sixth`, `sea` | Sixth Sense, Seance | -- | `Spectral<app><a>` |
| `fool`, `blusl`, `deck` | The Fool (card.lua:1377), Blue Seal (card.lua:1054), deck consumables (back.lua:189) | forced key | **no pool draw**; only `edi`/`front` could apply and never do (not Joker/Base) |

Edition keys follow the append: `edisho<a>`, `edibuf<a>`, `edijud<a>`, `edisou<a>`, `ediwra<a>`,
`edirif<a>`, `editop<a>`, `edirta<a>`, `ediuta<a>`.

---

## 3. `get_current_pool(_type, _rarity, _legendary, _append)` -- common_events.lua:1963-2053

```
pool = EMPTY(G.ARGS.TEMP_POOL); pool_size = 0
if _type == 'Joker':
    r = _rarity or pseudorandom('rarity'..ante..(_append or ''))          -- :1969  (ROLLED even if _legendary)
    rarity = _legendary and 4 or (r > 0.95 and 3) or (r > 0.7 and 2) or 1  -- :1970
    starting = P_JOKER_RARITY_POOLS[rarity]                                -- sorted by `order`
    pool_key = 'Joker'..rarity..((not _legendary and _append) or '')       -- :1971
else:
    starting = P_CENTER_POOLS[_type]; pool_key = _type..(_append or '')    -- :1972
for v in ipairs(starting):                                                  -- :1976-2036, pool order preserved
    add = nil
    if _type == 'Enhanced': add = true
    elif _type == 'Demo':   add = v.pos and v.config
    elif _type == 'Tag':    add = (not v.requires or P_CENTERS[v.requires].discovered)
                                  and (not v.min_ante or v.min_ante <= ante)        -- :1983-1986
    elif not (used_jokers[v.key] and not next(find_joker("Showman")))               -- :1987  (section 6)
         and (v.unlocked ~= false or v.rarity == 4):                                -- :1988  legendaries ignore locks
        if v.set == 'Voucher':
            add = not used_vouchers[v.key]
                  and every vv in v.requires is in used_vouchers                    -- :1992-1997
                  and v.key not displayed in G.shop_vouchers.cards                  -- :1999-2003
        elif v.set == 'Planet':
            add = not v.config.softlock or G.GAME.hands[v.config.hand_type].played > 0   -- :2009 (Planet X / Ceres / Eris)
        elif v.enhancement_gate:
            add = some card in G.playing_cards has center.key == v.enhancement_gate -- :2012-2018
                  (Steel Joker m_steel, Stone Joker m_stone, Lucky Cat m_lucky, Golden Ticket m_gold, Glass Joker m_glass)
        else: add = true
        if v.name == 'Black Hole' or v.name == 'The Soul': add = false              -- :2022-2024  (they still occupy Spectral slots 17, 18)
    if v.no_pool_flag  and pool_flags[v.no_pool_flag]:      add = nil               -- :2027  Gros Michel after extinction
    if v.yes_pool_flag and not pool_flags[v.yes_pool_flag]: add = nil               -- :2028  Cavendish before extinction
    if add and not banned_keys[v.key]: pool[#pool+1] = v.key; pool_size += 1        -- :2030-2032
    else:                              pool[#pool+1] = 'UNAVAILABLE'                -- :2034  IN PLACE
if pool_size == 0:                                                                  -- :2039-2050
    pool = { Tarot/Tarot_Planet: 'c_strength', Planet: 'c_pluto', Spectral: 'c_incantation',
             Joker/Demo/other: 'j_joker', Voucher: 'v_blank', Tag: 'tag_handy' }
return pool, pool_key..(not _legendary and ante or '')                              -- :2052
```

Notes:
* The `'rarity'` roll is consumed for The Soul too (its value is discarded).
* The `unlocked` check applies to vouchers as well (tier-2 vouchers are locked on a fresh
  profile). Consumables have no `unlocked` field -> always pass.
* `Tarot_Planet` (Fool only) is `P_CENTER_POOLS.Tarot_Planet`, sorted by `order` with
  Tarot/Planet order collisions -> unspecified order; never drawn from (Fool forces its key).

---

## 4. Draw + resample loop -- common_events.lua:1904-1909 / 1917-1922 / 2116-2121

```
center = pseudorandom_element(pool, pseudoseed(pool_key))
it = 1
while center == 'UNAVAILABLE':
    it = it + 1
    center = pseudorandom_element(pool, pseudoseed(pool_key..'_resample'..it))
```

* The **first** redraw uses `..._resample2` (the counter starts at 1 and is incremented before
  use), then `_resample3`, ... Each resample key is its own per-key stream that persists for the
  run: the second time this pool_key ever needs a resample, `_resample2`'s state advances to
  its 2nd value (it does not restart).
* The main `pool_key` stream advanced exactly once, so every later draw on it is unaffected by
  whether a resample happened. This is the in-place/side-stream property Phase 1 must preserve
  (do **not** filter-then-draw, do **not** delete-and-shift).
* `pool` is an array of strings -> `pseudorandom_element` sorts by numeric key -> draw is
  `pool[math.random(#pool)]` with `#pool` = full pool length including UNAVAILABLE slots.

---

## 5. `create_card(_type, area, legendary, _rarity, skip_materialize, soulable, forced_key, key_append)` -- common_events.lua:2082-2154

Order of RNG consumption:

1. **Soul / Black Hole** (:2087-2101), only if `not forced_key and soulable and not banned_keys.c_soul`:
   * `_type in {Tarot, Spectral, Tarot_Planet}` and `not (used_jokers.c_soul and not Showman)`:
     `pseudorandom('soul_'.._type..ante) > 0.997` -> `forced_key = 'c_soul'`.
   * `_type in {Planet, Spectral}` and `not (used_jokers.c_black_hole and not Showman)`:
     `pseudorandom('soul_'.._type..ante) > 0.997` -> `forced_key = 'c_black_hole'`.
   * Both `if`s run: a Spectral pack card rolls `soul_Spectral<a>` **twice**, and a Black Hole hit
     overrides a Soul hit. Only pack cards pass `soulable=true` (card.lua:1732-1774); shop
     consumables and all consumable-created cards cannot be The Soul / Black Hole.
2. `_type == 'Base'` -> `forced_key = 'c_base'` (:2103).
3. If `forced_key and not banned_keys[forced_key]` (:2109-2111): `center = P_CENTERS[forced_key]`
   and `_type = center.set` unless the set is `'Default'`. A banned forced key silently falls
   through to a normal pool draw.
   Else: `get_current_pool(_type, _rarity, legendary, key_append)` then the section-4 loop (:2113-2121).
4. **Front** (:2124): if `_type` is `Base` or `Enhanced`:
   `pseudorandom_element(G.P_CARDS, pseudoseed('front'..(key_append or '')..ante))` -- `G.P_CARDS`
   has exactly the 52 keys, drawn in key-string order (`pools.PLAYING_CARD_KEYS`).
5. `Card(...)` constructor -> `Card:set_ability` -> **`used_jokers[k] = true`** for every
   `P_CENTERS` entry whose `name` equals the center's name (card.lua:349-354; `not G.OVERLAY_MENU`).
   Also consumes 3 unseeded `math.random()` (card.lua:46-50, `discard_pos`; cosmetic).
6. If `_type == 'Joker'` (:2133-2152):
   * `modifiers.all_eternal` -> eternal.
   * If `area == G.shop_jokers or area == G.pack_cards`:
     `p = pseudorandom((pack and 'packetper' or 'etperpoll')..ante)` -- **consumed at every
     stake**; eternal if `enable_eternals_in_shop and p > 0.7`; perishable if
     `enable_perishables_in_shop and 0.4 < p <= 0.7`. Then only if `enable_rentals_in_shop`:
     `pseudorandom((pack and 'packssjr' or 'ssjr')..ante) > 0.7` -> rental.
   * `edition = poll_edition('edi'..(key_append or '')..ante)` (:2149) -- for **every** Joker
     create_card, including Judgement/Soul/Wraith/Riff-raff/tags (area G.jokers gets no sticker
     roll but does get the edition roll).

`_rarity` is passed as a *number in [0,1]* that replaces the roll: Wraith 0.99 (Rare), Rare Tag 1
(Rare), Uncommon Tag 0.9 (Uncommon), Riff-raff/Top-up 0 (Common).

---

## 6. `used_jokers` lifecycle and Showman

**Finding (contradicts the brief):** `used_jokers` is **not** set on purchase. It is set in
`Card:set_ability` (card.lua:349-354), i.e. the moment *any* Card object with that center is
constructed -- shop cards, pack cards, owned cards, even the shop's voucher/booster display
cards. It is cleared in `Card:remove` (card.lua:4741-4747) **only if**
`find_joker(name, true)` finds no copy in `G.jokers.cards`/`G.consumeables.cards`.

So at any moment `used_jokers` = {centers of all cards that currently exist anywhere}:
owned jokers + owned consumables + cards on display in the shop + cards of an open pack (+
harmless `c_base`, `p_*`, `v_*` marks). Observable consequences, all confirmed against the Lua:

* The two (or more) shop slots of one visit can never show the same joker/consumable: slot 2's
  pool has slot 1's key `UNAVAILABLE` (-> resample).
* A pack can never contain a duplicate; its cards also exclude whatever is on the shop shelf
  behind it (the shop UI persists while a pack is open: game.lua:3077, `shop_exists`) and owned
  cards. Dedupe is therefore **a resample, not a redraw**, and there is no local "seen" table.
* Reroll (button_callbacks.lua:2874-2878) `Card:remove`s the old shelf first, so a rerolled
  shop **can** re-offer a joker that was on the previous shelf (but not an owned one).
* Leaving the shop (`toggle_shop` -> `G.shop:remove()` -> `CardArea:remove` -> `Card:remove`,
  button_callbacks.lua:2499-2503, cardarea.lua:657-668) releases everything unbought; closing a
  pack (`end_consumeable`/`skip_booster`) releases the unchosen cards; using a consumable
  (removed after use) releases it; selling a joker releases it.
* Two Emperors in a row produce two *different* tarots because the first exists when the second
  is drawn.

**Showman (`j_ring_master`)**: `next(find_joker("Showman"))` -- owned in `G.jokers` **and not
debuffed** (misc_functions.lua:907). It disables the `used_jokers` test in exactly three places:
the generic cull branch (common_events.lua:1987, i.e. jokers, consumables *and* vouchers) and the
two soul/black-hole gates (:2090, :2096). It does **not** bypass `used_vouchers`, `requires`,
`banned_keys`, locks, softlocks or gates.

`generate.py` mirrors this: `create_card` calls `state.mark_used(key)`; the engine calls
`state.acquire(card)` on purchase/take, `state.release_shop(shop)` / `state.release_pack(cards)`
on leaving, `state.remove_owned(key)` on sell/use.

---

## 7. Editions and seals

### 7.1 `poll_edition(_key, _mod, _no_neg, _guaranteed)` -- common_events.lua:2055-2080

```
_mod = _mod or 1
poll = pseudorandom(pseudoseed(_key or 'edition_generic'))
if _guaranteed:
    poll > 1 - 0.003*25 (0.925) and not _no_neg -> negative
    poll > 1 - 0.006*25 (0.85)                  -> polychrome
    poll > 1 - 0.02*25  (0.5)                   -> holo
    poll > 1 - 0.04*25  (0.0)                   -> foil
else:
    poll > 1 - 0.003*_mod                and not _no_neg -> negative      -- NOT scaled by edition_rate
    poll > 1 - 0.006*edition_rate*_mod                   -> polychrome
    poll > 1 - 0.02 *edition_rate*_mod                   -> holo
    poll > 1 - 0.04 *edition_rate*_mod                   -> foil
nil otherwise
```

| Caller | key | mod | no_neg | guaranteed | resulting odds (rate 1 / Hone 2 / Glow Up 4) |
|---|---|---|---|---|---|
| create_card Joker | `edi<app><a>` | 1 | no | no | neg 0.3% always; poly 0.6/1.2/2.4%; holo 2/4/8%; foil 4/8/16% (cumulative bands, checked top-down) |
| Standard pack card | `standard_edition<a>` | 2 | yes | no | poly 1.2/2.4/4.8%; holo 4/8/16%; foil 8/16/32% |
| Aura | `aura` | -- | yes | yes | poly 15%, holo 35%, foil 50% |
| Wheel of Fortune | `wheel_of_fortune` | -- | yes | yes | same as Aura |

Illusion playing cards (UI_definitions.lua:786-793): if `pseudorandom('illusion') > 0.8` then
`e = pseudorandom('illusion')`: `> 0.85` polychrome, `> 0.5` holo, else foil (always some edition).

Edition **tags** (tag.lua:395-446) do not roll: the first untriggered Foil/Holo/Polychrome/Negative
tag sets its edition on the first edition-less Joker the shop creates (`store_joker_modify`, run
per created card, `break` on first applied).

### 7.2 Seals

There is no `poll_seal` function in 1.0.1o; two inline shapes:

* Standard pack (card.lua:1763-1772): `pseudorandom(pseudoseed('stdseal'..ante)) > 1 - 0.02*10`
  (20%) gates `t = pseudorandom(pseudoseed('stdsealtype'..ante))`: `> 0.75` Red, `> 0.5` Blue,
  `> 0.25` Gold, else Purple.
* Certificate (card.lua:2469-2473): `t = pseudorandom(pseudoseed('certsl'))`, same thresholds, no gate.

Talisman/Deja Vu/Trance/Medium set a fixed seal (card.lua:1186), no RNG.

---

## 8. The shop

### 8.1 Fresh shop -- game.lua:3072-3181 `Game:update_shop` (when `G.shop` does not exist)

1. tags `shop_start` (D6 -> reroll cost; no RNG).
2. `for i = 1, G.GAME.shop.joker_max - #G.shop_jokers.cards: G.shop_jokers:emplace(create_card_for_shop(G.shop_jokers))` (:3111-3113). The shelf is empty on a fresh shop, so exactly `joker_max` slots, left to right.
3. Voucher card from `G.GAME.current_round.voucher` if non-nil (:3124-3131). **No draw here** -- the voucher was drawn at run start or at boss defeat (section 10) and is the same for all three shops of the ante; it is `nil` after redemption until the next ante.
4. Boosters (:3145-3160): `for i = 1, 2: if not used_packs[i] then used_packs[i] = get_pack('shop_pack').key`; a slot `== 'USED'` shows nothing. `used_packs` is reset to `{}` at every `new_round` (state_events.lua:301), so each blind's shop has two fresh packs, and re-entering the same shop after opening a pack does not regenerate.
5. tags `voucher_add` (Voucher Tag -> `get_next_voucher_key(true)`, extra voucher card, :3163-3165, tag.lua:300-316), then `shop_final_pass` (Coupon: prices to 0; no RNG).

### 8.2 One slot -- UI_definitions.lua:742-800 `create_card_for_shop(area)`

```
for each tag: forced = tag:apply_to_run{type='store_joker_create', area}   -- Uncommon/Rare tags (8.4)
    if forced: run store_joker_modify tags on it; return forced
total = joker_rate + tarot_rate + planet_rate + playing_card_rate + spectral_rate
polled = pseudorandom(pseudoseed('cdt'..ante)) * total                       -- :766
rows = { {Joker, joker_rate}, {Tarot, tarot_rate}, {Planet, planet_rate},
         {(used_vouchers.v_illusion and pseudorandom(pseudoseed('illusion')) > 0.6) and 'Enhanced' or 'Base', playing_card_rate},
         {Spectral, spectral_rate} }                                           -- :768-774 (the 'illusion' draw happens HERE, every slot, if Illusion owned)
check = 0
for row in rows:                                                              -- :775
    if polled > check and polled <= check + row.val:
        card = create_card(row.type, area, nil, nil, nil, nil, nil, 'sho')    -- :776
        (Base/Enhanced + Illusion: edition per 7.1)                           -- :786-793
        return card
    check += row.val
```

Default rates 20/4/4/0/0 -> Joker 71.4%, Tarot 14.3%, Planet 14.3%. A `polled` of exactly 0 matches
nothing (returns nil) -- unreachable in practice.

### 8.3 Reroll -- button_callbacks.lua:2855-2911

1. every `G.shop_jokers` card: `remove_card` + `Card:remove()` (releases `used_jokers`).
2. `for i = 1, joker_max - #cards: emplace(create_card_for_shop(G.shop_jokers))` -- the **same
   function, same keys** (`cdt<a>`, `rarity<a>sho`, `Joker<r>sho<a>`, `Tarotsho<a>`, ...,
   `etperpoll<a>`, `edisho<a>`), each stream simply advanced by one more step per slot.
3. Vouchers and boosters untouched.

Hence "the ante-N shop queue" is the sequence of values of the `'...sho<a>'`-family streams, and
the per-player "pointer" is the per-key state; two players on one seed who reroll different
numbers of times are at different offsets of the same streams. Overstock mid-shop
(`change_shop_size(1)`, common_events.lua:1097-1118) fills the new slot immediately with one more
`create_card_for_shop` call on the same streams.

### 8.4 Tag hooks that generate (tag.lua)

| Tag | hook | effect |
|---|---|---|
| Uncommon | `store_joker_create` (:369-374) | `create_card('Joker', shop, nil, 0.9, nil,nil,nil,'uta')` fills the slot; couponed |
| Rare | `store_joker_create` (:345-368) | only if `#P_JOKER_RARITY_POOLS[3] > distinct rares owned`: `create_card('Joker', shop, nil, 1, ..., 'rta')`; else `nope()` (tag consumed, loop continues to the next tag) |
| Foil/Holo/Polychrome/Negative | `store_joker_modify` (:395-446) | first edition-less Joker created in the shop gets the edition; couponed |
| Voucher | `voucher_add` (:300-316) | `get_next_voucher_key(true)` (`Voucher_fromtag`); the shelf voucher is excluded because it is already in `G.shop_vouchers.cards` |
| Top-up | `immediate` on skip (:133-146) | up to 2x `create_card('Joker', G.jokers, nil, 0, ..., 'top')` (slots permitting) |
| Charm / Meteor | `new_blind_choice` (:211-243) | opens `p_arcana_mega_` / `p_celestial_mega_` .. `math.random(1,2)` (unseeded; the two are identical) |
| Ethereal / Standard / Buffoon | `new_blind_choice` (:244-289) | opens `p_spectral_normal_1` / `p_standard_mega_1` / `p_buffoon_mega_1` |
| Boss | `new_blind_choice` (:290-305) | `G.FUNCS.reroll_boss` -> `get_new_boss()` (section 11) |
| Orbital | hand chosen at blind-select UI build (UI_definitions.lua:1506-1516) | `pseudorandom_element(visible hands in pairs() order, pseudoseed('orbital'))` once per ante per blind type |

A tag is single-use: `self.triggered = true` on first application; the Lua loops
`G.GAME.tags` in order (oldest first).

---

## 9. Booster packs

### 9.1 `get_pack(_key, _type)` -- common_events.lua:1944-1961

```
if not first_shop_buffoon and not banned_keys.p_buffoon_normal_1:
    first_shop_buffoon = true
    return P_CENTERS['p_buffoon_normal_'..math.random(1,2)]       -- :1945-1948  NO pseudoseed; unseeded art pick
cume = sum(weight) over Booster pool where (not _type or kind == _type) and not banned
poll = pseudorandom(pseudoseed((_key or 'pack_generic')..ante)) * cume   -- :1953
it = 0
for v in ipairs(Booster pool):                                             -- pool order = `order`: Arcana 1-8, Celestial 9-16, Standard 17-24, Buffoon 25-28, Spectral 29-32
    if not banned[v.key]:
        if not _type or kind == _type: it += weight
        if it >= poll and it - weight <= poll: return v                    -- :1957
```

Weights: normal/jumbo 1 (x4 normal, x2 jumbo per kind for Arcana/Celestial/Standard), mega 0.25
(x2); Buffoon 0.6/0.6/0.6/0.15; Spectral 0.3/0.3/0.3/0.07. Total 22.42 -> Arcana 29.0%,
Celestial 29.0%, Standard 29.0%, Buffoon 8.7%, Spectral 4.3%. The first pack of a run is always a
Buffoon pack and **does not consume `shop_pack1`**; slot 2 of the first shop is the first
`shop_pack1` draw. Accumulate `cume`/`it` in pool order for bit-identical floats.

### 9.2 Contents -- card.lua:1723-1784 `Card:open`

`_size = config.extra` (3/5/5 for Arcana/Celestial/Standard normal/jumbo/mega; 2/4/4 Buffoon and
Spectral), `choose` 1 (2 for mega). Cards are created sequentially in one event; each marks
`used_jokers` as it is created (section 6 = the dedupe).

| Kind | per card `i` |
|---|---|
| Arcana | if `used_vouchers.v_omen_globe and pseudorandom('omen_globe') > 0.8`: `create_card('Spectral', pack, nil,nil, true, true, nil, 'ar2')` else `create_card('Tarot', ..., 'ar1')` (:1731-1735) |
| Celestial | if `used_vouchers.v_telescope and i == 1`: forced planet of the most-played visible hand (first strict max over `G.handlist` order, globals.lua:487-500; `nil` if nothing played -> normal draw) with append `pl1` (:1737-1752); else `create_card('Planet', ..., 'pl1')` (:1754) |
| Spectral | `create_card('Spectral', ..., 'spe')` (:1757) |
| Standard | `create_card(pseudorandom(pseudoseed('stdset'..ante)) > 0.6 and 'Enhanced' or 'Base', pack, nil,nil,nil, true, nil, 'sta')` (:1759); then `poll_edition('standard_edition'..ante, 2, true)` (:1761); then seal per 7.2 (:1763-1772). Enhanced pool = `m_bonus, m_mult, m_wild, m_glass, m_steel, m_stone, m_gold, m_lucky` (key `Enhancedsta<a>`), front `frontsta<a>` |
| Buffoon | `create_card('Joker', pack, nil,nil, true, true, nil, 'buf')` (:1774) -> `rarity<a>buf`, `Joker<r>buf<a>`, `packetper<a>`, `edibuf<a>` |

All pack cards pass `soulable = true` -> section 5 step 1 (`soul_Tarot<a>`, `soul_Planet<a>`,
`soul_Spectral<a>` x2). Opening a pack also fires `calculate_joker{open_booster}` (Hallucination:
`pseudorandom('halu'..ante) < normal/2` -> `create_card('Tarot', G.consumeables, ..., 'hal')`).

---

## 10. Vouchers -- common_events.lua:1901-1912 `get_next_voucher_key(_from_tag)`

```
pool, pool_key = get_current_pool('Voucher')      -- pool_key = 'Voucher'..ante
if _from_tag: pool_key = 'Voucher_fromtag'        -- no ante
draw + resample (section 4)
```

Eligibility (section 3): not redeemed, all `requires` redeemed (tier-2 needs its tier-1),
unlocked, not on the current shelf, not banned. Fallback `v_blank`.

When drawn: run start (game.lua:2178) and at **boss defeat** (state_events.lua:263) -- which runs
*after* `ease_ante(1)` (:248) in the same event queue, so the key is `'Voucher'..(new ante)`: the
voucher for ante N+1's shops is drawn the moment the ante-N boss dies. Redeeming sets
`current_round.voucher = nil` (card.lua:1819, 1850); the slot stays empty until the next ante.
Voucher Tag draws are a separate stream (`Voucher_fromtag`) and exclude the shelf voucher.

---

## 11. Boss blinds -- common_events.lua:2338-2383 `get_new_boss()`

```
if perscribed_bosses[ante]: return it (challenges)                          -- :2339-2347
if G.FORCE_BOSS: return it
eligible = {}
for k, v in pairs(P_BLINDS) with v.boss:                                    -- :2351-2359
    a = max(1, ante)
    non-showdown: v.boss.min <= a and (a % win_ante ~= 0 or ante < 2)
    showdown:     ante % win_ante == 0 and ante >= 2
remove banned_keys                                                           -- :2360-2362
min_use = min(bosses_used[k] for k in eligible); keep only k with bosses_used[k] == min_use   -- :2364-2377
boss = pseudorandom_element(eligible, pseudoseed('boss'))                    -- :2379  values are numbers -> ALPHABETICAL key order (pools.BOSS_KEYS_ALPHA filtered)
bosses_used[boss] += 1                                                       -- :2380
```

* `boss.max` is never consulted. `boss.min`: Hook/Club/Manacle/Psychic/Goad/Head/Window/Pillar 1;
  Mouth/Fish/Wall/House/Mark/Wheel/Arm/Water/Needle/Flint 2; Tooth/Eye 3; Plant 4; Serpent 5; Ox 6.
* "Reset when exhausted" is the min-usage filter: every eligible boss appears once before any
  repeats; a freshly eligible boss (e.g. Ox at ante 6, count 0) is guaranteed next if all others
  have been used.
* Called at: run start (game.lua:2177), `reset_blinds()` after a boss is defeated (button_callbacks.lua:2954 ->
  common_events.lua:2326-2336, with the new ante), and `reroll_boss` (button_callbacks.lua:2800-2848:
  Director's Cut / Retcon / Boss Tag) -- the replaced boss keeps its +1.
* `banned_keys` is where the MP mod's boss bans go; a banned boss is removed before the min-usage
  filter.

---

## 12. Tags -- common_events.lua:1914-1925 `get_next_tag_key(append)`

```
if G.FORCE_TAG: return it
pool, pool_key = get_current_pool('Tag', nil, nil, append)   -- 'Tag'..(append or '')..ante  (append never passed -> 'Tag'..ante)
draw + resample
```

Eligibility: `requires` center discovered (Rare: `j_blueprint`; Negative/Foil/Holo/Polychrome:
`e_negative`/`e_foil`/`e_holo`/`e_polychrome`) and `min_ante <= ante` (ante >= 2: Negative,
Standard, Meteor, Buffoon, Handy, Garbage, Ethereal, Top-up, Orbital). Pool order = `order` 1..24.
Fallback `tag_handy`. Drawn twice per ante, Small then Big: run start (game.lua:2179-2180) and at
Cash Out after a boss (button_callbacks.lua:2949-2953, new ante). Skipping a blind adds the tag
(button_callbacks.lua:2740-2782) and fires `immediate` then `new_blind_choice` hooks (8.4).

---

## 13. Deck creation and shuffles

* **Creation** (game.lua:2326-2378): one proto per `pairs(P_CARDS)` entry; Erratic replaces the
  key with `pseudorandom_element(G.P_CARDS, pseudoseed('erratic'))` per entry (52 independent
  draws; iteration order irrelevant); Abandoned drops K/Q/J; protos **sorted by the string
  `suit..rank..enh..edition..seal`**, then `card_from_control` creates Cards in that order ->
  `sort_id`s (global counter, card.lua:20-21) follow `C2..C9,CA,CJ,CK,CQ,CT,D2,...,S...`.
  Checkered then rewrites Clubs->Spades, Diamonds->Hearts in place (back.lua:244-256).
* **Shuffle** `CardArea:shuffle(_seed)` = `pseudoshuffle(cards, pseudoseed(_seed or 'shuffle'))`
  (cardarea.lua:572-575). `pseudoshuffle` re-sorts by `sort_id` first, so a shuffle depends only
  on {cards in the deck, their creation order, key state}. Keys: `'shuffle'` at run start
  (game.lua:2383), `'cashout'..ante` on Cash Out (button_callbacks.lua:2918), `'nr'..ante` at
  every blind start (state_events.lua:344). Cards are dealt from the **end** of `G.deck.cards`
  (cardarea.lua:76-77).
* **Round-start picks** (common_events.lua:2271-2324): `idol<a>`, `mail<a>`, `cas<a>` =
  `pseudorandom_element(non-Stone G.playing_cards, ...)` (sort_id order); `anc<a>` over the
  three suits other than the current (list order Spades, Hearts, Clubs, Diamonds). Run at run
  start (game.lua:2384-2389) and every round end (state_events.lua:273-276, after `ease_ante`).
* `edition_deck` (back.lua:222) is only used by challenge decks with a starting edition.

---

## 14. Consumable-created cards (card.lua `Card:use_consumeable`, :1091-1522)

| Consumable | RNG, in order |
|---|---|
| Judgement | `create_card('Joker', G.jokers, false, nil, ..., 'jud')` (:1418) |
| The Soul | `create_card('Joker', G.jokers, true, nil, ..., 'sou')`: rolls `rarity<a>sou` (discarded), draws `Joker4` (5 legendaries by order: Canio, Triboulet, Yorick, Chicot, Perkeo), edition `edisou<a>` |
| Wraith | `create_card('Joker', G.jokers, nil, 0.99, ..., 'wra')` -> Rare (:1457) |
| Emperor / High Priestess | `min(2, free slots)` x `create_card('Tarot'/'Planet', G.consumeables, ..., 'emp'/'pri')`, one event each (:1403-1413) |
| The Fool | forced `last_tarot_planet` (misc_functions.lua:1219 sets it), append `fool`, **no draw** (:1377) |
| Aura | `poll_edition('aura', nil, true, true)` (:1211) |
| Wheel of Fortune | `pseudorandom('wheel_of_fortune') < normal/4`; then `pseudorandom_element(eligible_strength_jokers, pseudoseed('wheel_of_fortune'))`; then `poll_edition('wheel_of_fortune', nil, true, true)` -- three draws on one key (:1470-1480) |
| Ectoplasm / Hex | `pseudorandom_element(eligible_editionless_jokers, pseudoseed('ectoplasm'/'hex'))` -> negative / polychrome (:1473-1486); no probability gate |
| Ankh | `pseudorandom_element(G.jokers.cards, pseudoseed('ankh_choice'))` (:1434) |
| Sigil / Ouija | `pseudorandom_element({'S','H','D','C'}, 'sigil')` / `({'2',..,'9','T','J','Q','K','A'}, 'ouija')` (:1233, 1247) |
| Familiar / Grim / Incantation | destroy: `pseudorandom_element(G.hand.cards, 'random_destroy')` (:1293); per created card (3/2/4): Familiar rank `{'J','Q','K'}` then suit on `familiar_create`; Grim suit on `grim_create` (rank A); Incantation rank `{'2'..'9','T'}` then suit on `incantation_create`; then enhancement `pseudorandom_element(Enhanced pool minus m_stone, 'spe_card')` (:1318-1336) |
| Immolate | hand copy sorted by playing_card id, `pseudoshuffle(copy, pseudoseed('immolate'))` (re-sorts by sort_id), destroy first 5 (:1340-1345) |
| Cryptid / Death / DNA / Strength / Hanged Man / Talisman family / planets | no RNG |

Joker-created: 8 Ball `Tarot8ba<a>` (after `8ball` < normal/4), Purple Seal `Tarot8ba<a>` (same
stream), Hallucination `Tarothal<a>`, Cartomancer `Tarotcar<a>`, Vagabond `Tarotvag<a>`,
Superposition `Tarotsup<a>`, Sixth Sense `Spectralsixth<a>`, Seance `Spectralsea<a>`, Riff-raff
`Joker1rif<a>` x2, Blue Seal forced planet `blusl`, Certificate `cert_fr`+`certsl`, Marble
`marb_fr`, Perkeo `perkeo` (copy), Invisible Joker `invisible` (copy), Madness `madness`
(destroy), To Do List `to_do`.

---

## 15. Probability rolls

Idiom: `pseudorandom(key) < G.GAME.probabilities.normal / odds` (one `math.random()` in [0,1)).
Oops! All 6s multiplies every entry of `G.GAME.probabilities` by 2 on `add_to_deck`
(card.lua:608-612) -- i.e. `normal` is 2^k for k Oops; the threshold can exceed 1.

| Effect | key | odds | cite |
|---|---|---|---|
| Lucky Card +Mult | `lucky_mult` | 5 | card.lua:988 |
| Lucky Card $ | `lucky_money` | 15 | card.lua:1076 |
| Glass shatter | `glass` | 4 | state_events.lua:961 |
| Wheel of Fortune | `wheel_of_fortune` | 4 | card.lua:1470 |
| Hallucination | `halu<a>` | 2 | card.lua:2337 |
| Gros Michel | `gros_michel` | 6 | card.lua:3020 |
| Cavendish | `cavendish` | 1000 | card.lua:3020 |
| 8 Ball | `8ball` | 4 | card.lua:3107 |
| Business Card | `business` | 2 | card.lua:3177 |
| Bloodstone | `bloodstone` | 2 | card.lua:3249 |
| Reserved Parking | `parking` | 2 | card.lua:3304 |
| Space Joker | `space` | 4 | card.lua:3420 |
| The Wheel (boss) flip | `wheel` | 7 | blind.lua:608 (`pseudorandom(pseudoseed('wheel'))`) |
| Misprint | `misprint` | `math.random(0, 23)` | card.lua:3701 |
| flipped_cards (challenge) | `flipped_card` | `1/modifiers.flipped_cards` | cardarea.lua:602 |

---

## 16. Run start and ante transitions

### 16.1 Seed

`G.GAME.pseudorandom.seed = args.seed or "TUTORIAL" (before tutorial completion) or
generate_starting_seed()` (game.lua:2164). `generate_starting_seed` (misc_functions.lua:219-244)
= `random_string(8, cursor-derived float)`: 8 chars, each `30%` digit `'1'..'9'` else letter
`'A'..'N'` or `'P'..'Z'` (no `'0'`, no `'O'`), uppercased; Gold stake re-rolls until
`get_first_legendary(seed)` (= `pseudorandom_element(P_JOKER_RARITY_POOLS[4], pseudoseed('Joker4', seed))`,
the stateless branch) is a legendary without a gold-stake win sticker. Typed seeds go through
`core.normalize_seed`. `hashed_seed = pseudohash(seed)` (game.lua:2168).

### 16.2 Draw order at run start (game.lua:2018-2447)

1. `Back:apply_to_run` (back.lua:173-288): deck vouchers -> `used_vouchers` + rate effects; Magic
   2x Fool / Ghost Hex created with forced keys into `G.consumeables` (marks `used_jokers`);
   Ghost `spectral_rate = 2`; Zodiac -> `tarot_rate = planet_rate = 9.6`, `joker_max = 3`.
   No RNG.
2. `blind_choices.Boss = get_new_boss()` -- `boss` (:2177)
3. `current_round.voucher = get_next_voucher_key()` -- `Voucher1` (:2178)
4. `blind_tags.Small = get_next_tag_key()`; `blind_tags.Big = get_next_tag_key()` -- `Tag1` x2 (:2179-2180)
5. deck protos (Erratic: 52 x `erratic`), Cards created in sorted order (:2326-2378)
6. `deck:shuffle()` -- `shuffle` (:2383)
7. `reset_idol_card`, `reset_mail_rank`, `reset_ancient_card`, `reset_castle_card` -- `idol1`, `mail1`, `anc1`, `cas1` (:2385-2389)
8. blind-select UI: `orbital` once per blind type (UI_definitions.lua:1509-1516)

### 16.3 After a boss is defeated

1. `end_round` (state_events.lua:87-288): `ease_ante(1)` (:248, queued first) ... then
   `current_round.voucher = get_next_voucher_key()` with the **new** ante (:263), then
   `reset_idol_card()` etc. (:273-276).
2. Cash Out (button_callbacks.lua:2912-2957): `deck:shuffle('cashout'..ante)`;
   `blind_tags.Small/Big = get_next_tag_key()` x2 (:2951-2952); `reset_blinds()` ->
   `blind_choices.Boss = get_new_boss()` (:2954 -> common_events.lua:2326-2336).
3. Every blind start: `new_round` resets `used_packs`, shuffles `nr<a>`.

`generate.defeat_boss(state)` performs 1-2 (voucher, tags, boss) in this order.

---

## 17. Ambiguities and what to test

| Issue | Status |
|---|---|
| `p_buffoon_normal_1` vs `_2` for the forced first pack; `p_arcana_mega_1/2`, `p_celestial_mega_1/2` from Charm/Meteor tags | unseeded `math.random(1,2)` after arbitrary UI consumption (`Card:init` eats 3 per card, `juice_up`, sound pitch). **Cosmetic**: identical contents. `generate.py` returns `_1`; parity tools must compare kind+size. |
| `to_do` / `orbital` candidate order | **Resolved.** `pairs(G.GAME.hands)` is LuaJIT string-hash order. The game ships **LuaJIT 2.0.5** (`lua51.dll`, fixed string hash), so the order is deterministic: `generate.HANDS_PAIRS_ORDER` = Flush House, Full House, Flush, Pair, High Card, Straight Flush, Straight, Two Pair, Flush Five, Five of a Kind, Three of a Kind, Four of a Kind (filtered to visible hands). Dumped by executing the verbatim `hands = {...}` constructor (game.lua:2001-2014) inside the game's own DLL; stable across processes; asserted by `test_hands_pairs_order_matches_game_dll`. lupa's LuaJIT 2.1 randomises its string-hash seed per VM and gives a different (rotating) order, so it is *not* an oracle for this. Caveat: a reloaded save rebuilds the table via STR_UNPACK in a different insertion order. |
| `Tarot_Planet` pool order | `table.sort` with colliding `order` values; unspecified but never drawn from. |
| Telescope "most played hand" | requires `G.GAME.hands[h].visible` (set when a hidden hand is first played); `generate.open_pack` uses `visible or played > 0`. Test: Flush Five played once vs Pair played once -> Flush Five wins (earlier in `G.handlist`). |
| Hieroglyph/Petroglyph at ante 1 | ante becomes 0: keys like `Joker1sho0`, `boss` eligibility uses `max(1, ante)`. Untested against the game. |
| Overstock bought mid-shop | fills the new slot immediately from the same streams; Python caller must invoke `_fill_shop_slot` once. Untested against the game. |
| Negative-edition consumables | `used_jokers` is keyed by center, so a Negative copy and a normal copy of the same consumable cannot both be drawn while one exists. Follows from the code; untested. |
| Saved-game reload | `G.GAME.pseudorandom` is persisted; save/load is transparent. |

---

## 18. Validation performed (`mp/tests/test_generate_oracle.py`)

The **real** Lua functions `get_current_pool`, `create_card`, `poll_edition`, `get_pack`,
`get_next_voucher_key`, `get_next_tag_key`, `get_new_boss`, `reset_idol_card/mail/ancient/castle`,
`pseudoshuffle`, `create_card_for_shop` (UI_definitions.lua:742-800) and the `Card:open` pack
loop (card.lua:1726-1781) were loaded verbatim into LuaJIT 2.1 (lupa) together with
`Game:init_item_prototypes`, `Game:init_game_object`, and a minimal `Card` stub reproducing
`set_ability`'s `used_jokers` mark and `Card:remove`'s release. Driving both sides through the
same script (run start; antes 1-3; 3 shops per ante with 2 rerolls each; both packs opened each
shop; variants: buying a card, Showman, banned keys incl. a boss and the first-shop Buffoon,
Gold stake stickers, fresh profile locks/discoveries; plus Judgement, Soul x2, Wraith,
Riff-raff x2, Top-up, Emperor x2, High Priestess, Sixth Sense, 8 Ball + Purple Seal, Rare/Uncommon
Tag cards, Voucher Tag x2, Aura x3, Erratic deck, idol/mail/anc/cas) over 30+22 seeds:
**0 mismatches** in keys, fronts, editions, seals, stickers, vouchers, tags, bosses,
`used_jokers` and `bosses_used`, after normalising the cosmetic Buffoon art suffix.

The harness is the repo test `mp/tests/test_generate_oracle.py` (15 seeds x 7 scenarios + 20
creation-path seeds, ~2 s; skips without lupa / `_reference`).

**Runtime check against the game's own binary.** Balatro's `lua51.dll` reports `LuaJIT 2.0.5`
(not 2.1). Loaded via ctypes with the verbatim `pseudohash`/`pseudoseed`/`pseudorandom`/
`pseudorandom_element`/`pseudoshuffle`, it agrees with `core.py` to the last digit (`%.17g`) on
seeded `math.random()`/`math.random(m,n)` sequences, the full keyed chain, `pseudoshuffle` and
`pseudorandom_element`, **and** on the unseeded first `math.random()` (2.0.5 lazily seeds with
0.0, which is exactly what 2.1's `lj_prng_seed_fixed` precomputes). So the 2.1-validated core is
bit-exact for the shipped runtime; the only 2.0.5/2.1 behavioural difference found is string-hash
order (section 17).

What this does **not** validate: the *sequencing* of calls across game states (sections 8.1,
16.2, 16.3) and the `Card` lifecycle stub -- those are from reading `game.lua` /
`state_events.lua` / `button_callbacks.lua` and need Agent D's end-to-end ground truth.
