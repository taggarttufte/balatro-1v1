# TAGS_NOTES.md -- skip-blind Tags: module contract for W2

Owner: P1-tags (W6). Files: `balatro_sim/tags.py`, `tests/sim_tests/test_tags.py`, this note.
Source of truth: `_reference/balatro_src/tag.lua` (`Tag:apply_to_run`, lines 115-468) plus
the call sites listed under "When to call what". Nothing in `game.py`/`shop.py` was touched;
`_skip_blind` still pays its flat +$5 until W2 wires this in.

## 1. What the module is

`tags.py` is pure and engine-agnostic. It knows the 24 tag prototypes, the trigger each one
consumes at, and the exact effect logic; it knows nothing about `BalatroGame`. The engine
talks to it through two objects:

* **`TagState`** -- mirrors `G.GAME.tags` (list, oldest first) plus the tag-only bookkeeping
  `shop_d6ed`, `shop_free`, `round_resets.temp_handsize`, `round_resets.temp_reroll_cost`.
  One per run, stored on the game; `clone()` deep-copies it (put it in `BalatroGame.clone()`).
* **`TagContext`** -- the only thing an effect reads or calls. W2 subclasses it: five
  read-only fields and 17 hooks (section 3). A hook that is not overridden raises
  `TagHookNotImplemented` naming the hook, so a wiring gap fails loudly rather than silently
  mis-paying.

Low-level primitives also exist -- `apply_tag(tag_or_key, trigger, ctx) -> bool` and
`tag_is_consumed_at(key) -> Trigger` -- but the **`TagState` drivers are the contract**: only
they reproduce the Lua loop semantics (break-after-first vs fire-all, Double Tag copying,
per-shop guards, Boss Tag re-running the blind-select pass).

The tag *draw* is not here. `rng/generate.py` (Agent C) owns `get_next_tag_key`
(`'Tag'..ante`), `orbital_hand` (`'orbital'`), `next_voucher(state, from_tag=True)`
(`'Voucher_fromtag'`), and `create_card(...)` with the tag key-appends `'top'`/`'uta'`/`'rta'`.
Those feed the hooks.

## 2. When to call what (the wiring checklist)

Lua trigger names are the `Trigger` enum values, so each line can be checked against the source.

| Engine moment | Lua site | Call |
|---|---|---|
| New run | game.lua:2423 | `game.tags = TagState()` |
| **Skip blind** (Small/Big) | button_callbacks.lua:2755-2777 | `game.skips += 1` **first**; joker `skip_blind` hooks; then `out = tags.skip_blind(tag_key, ctx, blind_type='Small'|'Big', orbital_hand=<generate.orbital_hand pick for that blind>)`. `skip_blind` = `acquire` + `on_blind_select`. Handle `out.pending_pack` (below). |
| Blind-select screen shown (after every shop) | game.lua:3290-3295 | `out = tags.on_blind_select(ctx)` (immediate pass, then new_blind_choice pass) |
| A tag-opened pack has been fully resolved while still on blind select | button_callbacks.lua:2617-2619 | `out = tags.on_new_blind_choice(ctx)` -- repeat until `out.pending_pack is None` |
| Anaglyph Deck after a boss win / Diet Cola sold | back.lua:111-114, card.lua:2361-2369 | `tags.acquire('tag_double', ctx)` |
| Round start, first draw of the round | game.lua:3214-3216 | `tags.on_round_start(ctx)` (Juggle) -- once per played round, before dealing |
| Round won: bookkeeping | state_events.lua:124 | `game.unused_discards += discards_left` (engine-owned; Garbage reads it) |
| Round won: end_round cleanup | state_events.lua:270-271 | `tags.on_round_end_cleanup(ctx)` (reverts Juggle hand size, clears D6 base) |
| Round eval rows | state_events.lua:1183-1190 | `tags.on_round_eval(ctx)` with `ctx.last_blind_was_boss` set. Returns total $; each Investment also calls `ctx.add_dollars(25,'tag_investment')`. In Lua the tag rows sit after joker $ bonuses and before interest, and interest is computed from `G.GAME.dollars` *before* any row is paid -- keep that ordering. |
| Cash out (entering shop) | button_callbacks.lua:2932-2933 | `tags.on_cash_out()` (resets the two per-shop guards) |
| Shop build | game.lua:3093-3095 | `tags.on_shop_start(ctx)` (D6) |
| Each shop card slot, fresh shop **and** every reroll | UI_definitions.lua:753-763 | `card = tags.store_joker_create(ctx)`; if `None`, do the normal rate roll, then `tags.store_joker_modify(ctx, card)` on the rolled card. (A forced card is already modified inside `store_joker_create`.) |
| Fresh shop only, after boosters | game.lua:3161-3163 | `tags.on_voucher_add(ctx)` (Voucher) |
| Fresh shop only, last | game.lua:3164-3166 | `tags.on_shop_final_pass(ctx)` (Coupon) |

`pending_pack` protocol: a `new_blind_choice` pack tag is an interrupt. `on_blind_select` /
`on_new_blind_choice` / `skip_blind` fire at most one pack tag and return its key in
`BlindChoiceOutcome.pending_pack`. Open that pack free (`cost 0`, `from_tag`), let the agent
pick, and when it closes call `on_new_blind_choice(ctx)` again; stop when `pending_pack` is
`None`. Boss Tags do **not** interrupt: `reroll_boss` re-runs the pass internally
(button_callbacks.lua:2847-2849), so a Boss Tag followed by a Charm Tag fires both in one call.

Skip-blind tag *preview*: the real game shows the tag before you commit. `get_next_tag_key`
is drawn per blind when the blind-select UI is built; W2 should draw both Small/Big tags at
blind-select time (as `generate.RunState.blind_tags` already anticipates) and pass the chosen
one to `skip_blind`.

### Overlap with `rng/generate.py` -- pick ONE path for the shop tags

`generate.generate_shop` / `_fill_shop_slot` already implement `store_joker_create`,
`store_joker_modify`, `voucher_add` against `RunState.tags` + `RunState.triggered_tags`
(by index), because those tags change *which RNG calls happen*. `tags.py` implements the same
four passes behind hooks. W2 must not run both. Two consistent options:

* **A (recommended):** drive from `TagState`; have the hooks call `generate.create_card(state,
  'Joker', area='shop', rarity=0.9|1, key_append='uta'|'rta')` and `generate.next_voucher(state,
  from_tag=True)`, and call `generate.create_card_for_shop` yourself for the rate roll instead
  of `generate_shop`. Then `RunState.tags` stays empty and generate's internal tag code is inert.
* **B:** keep `generate.generate_shop` and before each shop set `state.tags =
  game.tags.keys()`; afterwards mark `game.tags.tags[i].triggered = True` for each `i` in
  `state.triggered_tags` and purge; `CardGen.couponed`/`.edition`/`.from_tag` carry the
  results. D6 and Coupon are not in generate, so still call `on_shop_start` /
  `on_shop_final_pass`.

Either way the RNG call order is generate's job; `tags.py` never touches RNG.

## 3. `TagContext` -- fields and hooks W2 must provide

Read-only fields (set as attributes or expose as properties on the subclass):

| field | Lua | note |
|---|---|---|
| `dollars` | `G.GAME.dollars` | may be negative (Credit Card) |
| `skips` | `G.GAME.skips` | already incremented for the current skip |
| `hands_played` | `G.GAME.hands_played` | run total (state_events.lua:523) |
| `unused_discards` | `G.GAME.unused_discards` | sum of `discards_left` at the end of each played round |
| `last_blind_was_boss` | `G.GAME.last_blind.boss` | blind just finished was a Boss |

Hooks (all 17; `TagHookNotImplemented` if missing):

| hook | used by | must do |
|---|---|---|
| `add_dollars(amount, source)` | Skip, Garbage, Handy, Economy, Investment | `ease_dollars(amount)`; `source` is the tag key |
| `joker_slots_free() -> int` | Top-up | `card_limit - #jokers`, re-read before each spawn |
| `spawn_joker_to_slots(rarity, key_append)` | Top-up (`0, 'top'`) | create Joker (`generate.create_card(..., rarity=0, key_append='top')`), add to joker area |
| `create_shop_joker(rarity, key_append) -> card` | Uncommon (`0.9,'uta'`), Rare (`1,'rta'`) | create for `shop_jokers`, return it (caller emplaces) |
| `rare_joker_available() -> bool` | Rare | `len(rare pool) > number of DISTINCT rare joker keys owned` |
| `card_is_editionless_joker(card) -> bool` | Foil/Holo/Poly/Negative | `set == 'Joker' and no edition` |
| `set_card_edition(card, edition)` | Foil/Holo/Poly/Negative | edition in `foil|holo|polychrome|negative` |
| `mark_card_couponed(card)` | Uncommon, Rare, edition tags | cost 0 while in shop; sell value from real cost |
| `level_up_hand(hand, levels)` | Orbital (`+3`) | `level_up_hand` |
| `choose_orbital_hand(blind_type) -> str` | Orbital, only if `acquire` got no `orbital_hand` | `generate.orbital_hand(state, visible)` memoised per ante per blind type |
| `change_hand_size(delta)` | Juggle (`+3`, then `-n` at cleanup) | `G.hand:change_size` |
| `open_pack(pack_key)` | Charm/Meteor/Ethereal/Standard/Buffoon | open free pack; interrupt (see protocol) |
| `reroll_boss()` | Boss | new boss via generate's boss draw; **no $10**, but set `boss_rerolled` (burns Director's Cut for this ante) |
| `add_shop_voucher()` | Voucher | `generate.next_voucher(state, from_tag=True)`, `+1` voucher slot, emplace; priced normally |
| `set_temp_reroll_cost(0)` | D6 | this shop's reroll = `0 + reroll_cost_increase` ($0, $1, $2, ...) |
| `clear_temp_reroll_cost()` | cleanup after D6 | back to normal base |
| `make_shop_free()` | Coupon | every card currently in `shop_jokers` + `shop_booster` -> couponed; vouchers stay priced; rerolled cards are not free |

Pack keys: `tag_charm -> p_arcana_mega_1`, `tag_meteor -> p_celestial_mega_1`,
`tag_ethereal -> p_spectral_normal_1`, `tag_standard -> p_standard_mega_1`,
`tag_buffoon -> p_buffoon_mega_1` (`TAG_PACKS`). Charm/Meteor roll `_1`/`_2` with the
*unseeded* `math.random` in Lua -- same pack, cosmetic art only -- so `_1` is used.

## 4. Semantics decisions (each verified against the Lua)

* **Skip Tag counts the skip that granted it**: `G.GAME.skips` is incremented before
  `add_tag` (button_callbacks.lua:2755), so a Skip Tag from the run's first skip pays $5, from
  the third skip $15.
* **Economy** pays `min(40, max(0, dollars))` -- doubles money up to +$40, $0 at negative money.
  Its dollars read is a queued event, as are the other tags' `ease_dollars` calls
  (common_events.lua:68-108), so an Economy *later* in the list than a Handy/Garbage/Skip
  fired in the same pass sees their payouts; an Economy *earlier* does not. The synchronous
  model reproduces this exactly because each hook updates `ctx.dollars` before the next tag
  runs. Tests cover both orders.
* **Garbage** uses `G.GAME.unused_discards`, which only accumulates at the end of *played*
  rounds (skipped blinds add nothing).
* **Investment** fires at the round eval after a Boss (`last_blind.boss`), persists through
  Small/Big evals; two Investments both pay at the same Boss ($50).
* **Top-up** re-checks free joker slots before each of its two spawns; a Negative joker
  spawned first leaves room for the second. Consumed even with zero free slots.
* **Orbital** keeps its hand on the instance (`ability.orbital_hand`); a Double Tag copy
  inherits the same hand (tag.lua:324-328), so Double+Orbital = +6 levels to one hand.
* **Double Tag**: fires on the next non-Double tag acquired; each untriggered Double present
  adds one copy; copies are appended *after* the original and themselves pass through
  `add_tag` (no further Doubles fire because the originals are already consumed). `[D, D]` +
  X gives `[X, X, X]`. A Double acquired while holding a Double is not copied. Copies are in
  place before the `immediate` pass, so Double+Handy pays twice at the skip, Double+Boss
  rerolls twice, Double+Charm opens two packs (one per pass).
* **Juggle** stacks (+3 each, both at the same round start), reverted together at round end.
* **D6 / Coupon** are guarded per shop (`shop_d6ed` / `shop_free`, reset at cash-out): a
  second copy waits for the next shop. Coupon only frees the initial stock; rerolled cards and
  vouchers are priced.
* **Uncommon/Rare** force slot 1 of the shop (first untriggered create-tag wins; one per
  slot). A Rare Tag with no unowned rare left `nope()`s (consumed, no card) and the scan
  continues to the next tag in the same slot. The forced card still goes through the
  edition-tag pass.
* **Edition tags**: at most one per card; only edition-less Jokers; consumed only when
  applied (a Tarot in the slot leaves the tag alone). Applies to reroll cards too.
* **Boss Tag** sets `round_resets.boss_rerolled` like a paid reroll, so it consumes the
  Director's Cut reroll for that ante (Retcon unaffected).
* **`new_blind_choice` passes fire one tag at a time**; after a Boss Tag the pass restarts
  from the head of the list (Lua reroll_boss re-invokes the loop).
* Consumed tags are purged synchronously at the end of each driver call. In Lua removal is a
  delayed event, but `triggered` already blocks re-firing, so the observable state is the same.

## 5. Not modelled here (needs engine support or is out of scope)

* **Tag draw / eligibility** (`get_next_tag_key`: `requires` must be discovered, `min_ante`,
  `tag_handy` fallback, `G.FORCE_TAG`) -- `rng/generate.py`.
* **MP ruleset bans** (e.g. Boss Tag banned under Attrition): the mod removes banned tags
  from the draw pool; that belongs in `RunState.banned_keys`. `tags.py` will happily apply a
  Boss Tag if handed one.
* **Pack contents and pack interrupts** (`G.FUNCS.use_card` on a `from_tag` booster, the
  agent's picks, `pack_cards` RNG) -- engine + generate.
* **Boss reroll draw** (`get_new_boss` eligibility, `bosses_used` bookkeeping) -- hook.
* **`last_blind`** must be set by the engine when a blind is set (`Blind:set_blind`).
* **Hand-level text / UI, tag_tally HUD ordering, save/load** -- none.
* **`Tag:nope()` for Rare** is modelled as "consumed, no card"; the visual is irrelevant.
* **Anaglyph / Diet Cola** acquisition points are one-liners for W2 (`acquire('tag_double')`),
  listed in the table, not wired here.
* **Orbital hand visibility set** (`G.GAME.hands[h].visible`) for `choose_orbital_hand` --
  engine tracks which hands have been played/seen; generate needs it as `visible`.

## 6. Tests

`python -m pytest engine/tests/sim_tests/test_tags.py -q` -> **124 passed** (0.13 s).
Coverage: every tag at least once (24 parametrised `consumed_at` + 24 `wrong_trigger_never_consumes`
+ per-tag effect tests), all six Economy boundaries, Top-up slot rules, Rare nope + scan
continuation, edition-tag ordering, one-pack-per-pass, Boss-chains-into-next, two Boss Tags,
two Investments, Juggle x2, Double x1/x2, Double+{Handy, Boss, Charm, Investment, Orbital},
per-shop guards for D6/Coupon, full shop cycle with all shop tags, clone independence, and
`TAG_DEFS` == `rng/pools.py` `TAGS` (keys, names, order, config, type).
