# DELEGATE_NOTES — Phase 1 W2: the engine delegates all generation to `rng/generate.py`

**Agent P1-delegate, 2026-08-21.** Files: `balatro_sim/shop.py` (rewritten), `balatro_sim/game.py`
(targeted edits, list in §8), `balatro_sim/consumables.py` (created-card paths + voucher routing),
`balatro_sim/jokers/base.py` (`JokerInstance.clone()` only), `tests/sim_tests/test_delegate.py` (37
tests), this note.  Out-of-ownership edits, each flagged: §9.

## 0. Gate numbers (run 2026-08-21, after P1-effects' W3 landed in the same tree)

| gate | result |
|---|---|
| `python -m pytest engine/tests -q` | **1358 passed / 10 skipped / 0 failed** at the final run (1321 before P1-effects' later test additions; was 1279/13; +37 `test_delegate.py`, +6 new consumable/voucher/Fool tests; the 3 runtime skips in `test_rewards_v5` ("Could not reach shop") now reach the shop) |
| `python -m pytest tests/test_engine_invariants.py -q` | **14 passed / 0 failed / 0 xfail** (was 3 pass / 1 xfail / 10 fail) — incl. the three "effect rolls don't move generation" tests, which needed W3's `game.rng` deletion |
| `python -m oracle.engine_parity --antes 1-3` | **126/126 exact through ante 3**; `--antes 1-8 --rerolls 5` → **126/126 exact through ante 8**, no harness fallbacks (was 0/126 through ante 1) |
| `python -m pytest tests/test_engine_reachability.py -q` | **226 passed / 6 failed / 3 xfailed** at the final run (240/6/3 an hour earlier; the probe file is being edited concurrently by P1-effects) (was 186 / 46 / 3). Remaining 6 are not W2: `j_four_fingers`, `j_shortcut`, `j_smeared`, `j_flower_pot` (hand-eval flags set after `evaluate_hand` — W5), `j_chicot` (boss-disable flag nothing reads — W5), `j_space` (probability roll — W3/W5) |
| `engine_parity --probe` | 11/12 hooks ok; only the optional `state_signature` is missing (harness fallback works) |

Other parity policies (not gates, reported for completeness):

* `--antes 1-8 --reference generate --reroll-every-shop --buy-shelf`: 117/126 through ante 3
  (126/126 through ante 1).  Every mismatch traced is the **engine being more faithful than the
  generate-only replay**: a bought Riff-raff creates two Commons at blind select (`Joker1rif<a>`),
  Hallucination creates Tarots on pack opens, Turtle Bean self-destructs and is released, Gros
  Michel's extinction lets Cavendish into the pool — joker *effects* that `replay_visits` (pure
  generation) cannot model.  Seed `1558AXDL` is the worked example (Riff-raff bought at visit 0).
* `--antes 1-8 --buy-vouchers`: all 43 seeds with a `voucher[bought-chain]` mismatch have
  **Hieroglyph or Petroglyph earlier in the chain** (checked programmatically, 43/43, 0
  unexplained): the engine applies the real `ease_ante(-1)`, so the next voucher is drawn from
  `Voucher<a-1>`'s stream again, while the analyzers' `voucher_chain_if_bought` ignores the ante
  change.  The `shop_queue[bought-chain]` rows are expected: the JSON queue is the no-voucher
  queue, and Merchant/Tycoon/Overstock/Magic Trick legitimately change it.

## 1. Architecture: which engine moment calls what

Everything below goes through `game.run_state` (`generate.RunState`, rebuilt per `reset()`,
cloned in `clone()` with a flat-container copier instead of deepcopy).

| Engine moment | Function (game.py unless noted) | generate / tags call | Keys consumed |
|---|---|---|---|
| New run | `_init_game_vars` | `gen.start_run(rs, deck_key)` → `boss_blind`, `blind_tags`, `current_round_voucher`, `full_deck` built in creation order from `RunStart.deck`, `round_picks`; `banned_keys = shop.BANNED_JOKERS` | `boss`, `Voucher1`, `Tag1`×2, `shuffle`, `idol1/mail1/anc1/cas1` |
| Blind-select screen shown (run start, after every shop, after a skip) | `_enter_blind_select` | `gen.orbital_hand` once per ante per blind type (Small, Big, Boss), then `tag_state.on_blind_select(ctx)` (immediate + new_blind_choice passes); a pack tag opens a free pack (`_open_booster(..., return_state=BLIND_SELECT)`) | `orbital`×3 per ante |
| Blind start | `_start_blind` → `_init_deck` | `rs.new_round()` (used_packs reset), `gen.shuffle_deck(rs, full_deck sorted by Card.id, 'nr<ante>')` (dealt from the END), then `tag_state.on_round_start` (Juggle) before the draw | `nr<ante>` |
| Skip blind | `_skip_blind` | `skips += 1`; `fire_hook('on_blind_skipped')`; blind advances (`_prepare_next_blind`); `tag_state.skip_blind(blind_tags[kind], ctx, orbital_hand=…)`; pending pack → BOOSTER_OPEN. **No shop after a skip** (real game). | — |
| Round won | `_end_round` | `unused_discards += discards_left`; payout (interest from pre-row dollars) + `tag_state.on_round_eval` (Investment); `fire_hook('on_boss_beaten'/'on_round_end')`; Blue Seal → `gen.blue_seal`; `tag_state.on_round_end_cleanup`; **if boss: `ante += 1` then `gen.defeat_boss(rs)`** → new `boss_blind`, `blind_tags`, `current_round_voucher`; `_round_end_resets()`; Cash Out shuffle; `tag_state.on_cash_out()`; shop | `Voucher<a+1>`, `Tag<a+1>`×2, `boss`, `idol/mail/anc/cas<a>`, `cashout<a>` |
| Shop build (fresh) | `shop.generate_shop(game)` | `_sync_run_state()`; `tag_state.on_shop_start` (D6); `rs.tags = tag_state.keys()`; `gen.generate_shop(rs)` (shelf = `shop_joker_max` slots incl. Uncommon/Rare/edition tags, voucher from `current_round_voucher`, 2 × `get_pack`, Voucher-Tag vouchers); `_absorb_tag_triggers()`; `tag_state.on_shop_final_pass` (Coupon); `fire_hook('on_shop_enter')` | `cdt<a>`, `rarity<a>sho`, `Joker<r>sho<a>`, `Tarotsho<a>`…, `etperpoll<a>`, `edisho<a>`, `shop_pack<a>`, `Voucher_fromtag`, `illusion` |
| Reroll | `shop.reroll_shop(game)` | cost (D6 base, Chaos free rerolls, Credit Card floor); `gen.reroll_shop(rs, game._shop_gen)` (old shelf released, same streams advanced); `fire_hook('on_reroll')` | same shelf keys |
| Buy | `shop.buy_item` | joker/consumable/card/voucher/booster; `emplace_joker` (acquire + `on_init` + Oops! sync); voucher → `apply_voucher` (syncs `used_vouchers`, rates, `shop_joker_max` (+ immediate slot fill), ante); shelf voucher → `current_round_voucher = None`; booster → `_open_booster` and `step()` enters `BOOSTER_OPEN` | Overstock mid-shop: one more `create_card_for_shop` |
| Pack open | `_open_booster` → `shop.booster_contents` | `_sync_run_state()`; `gen.open_pack(rs, center_key)` → `BoosterChoice` entries (`.key/.set/.edition/.enhancement/.seal/.front/.card/.gen`); `fire_hook('on_booster_opened')` | `soul_*`, `Tarotar1<a>`, `Planetpl1<a>`, `Spectralspe<a>`, `stdset/standard_edition/stdseal/stdsealtype<a>`, `rarity<a>buf`, `Joker<r>buf<a>`, `packetper<a>`, `edibuf<a>`, `omen_globe` |
| Pick / skip | `_pick_booster` / `_skip_booster` → `_close_booster` | grant (`emplace_joker` / consumable slot / `add_card`) + `rs.acquire`; unpicked → `rs.release_pack`; back to `SHOP`, or to `BLIND_SELECT` where `tag_state.on_new_blind_choice` re-runs | — |
| Sell | `shop.sell_joker` | `remove_joker` (`rs.remove_owned`), `sell_hooks` (W3), `state['pending_tags']` → `tag_state.acquire` (Diet Cola → Double Tag) | — |
| Leave shop | `_end_shop` | `fire_hook('on_shop_leave')`; `rs.release_shop(_shop_gen)`; `current_shop = []`; `_advance_blind` → `_enter_blind_select` | — |
| Boss reroll | `_reroll_boss` (action `{"type": "reroll_boss"}`; Boss Tag via `TagContext.reroll_boss`) | $10 unless from tag; `boss_rerolled`; `gen.reroll_boss(rs)`; `new_blind_choice` pass re-run | `boss` |
| Created cards | `grant_created(spec)`, `_materialize(token)`, consumables.py | `gen.create_from_spec` / `gen.create_card` / `gen.fool` / `gen.spectral_create_cards` / `gen.aura` / `gen.sigil` / `gen.ouija` / `gen.ectoplasm` / `gen.hex_` / `gen.ankh` / `gen.immolate` / `gen.marble_joker` / `gen.certificate` / `gen.blue_seal` | per GENERATION_SPEC §14 |

`_sync_run_state()` runs before every generation call and rebuilds the ownership view from the
game's own collections: `owned_jokers`, `owned_consumables`, `showman`, **`used_jokers` = owned ∪
unsold shelf ∪ open-pack cards** (exactly `Card:set_ability`/`Card:remove` semantics, §6 of the
spec), `shop_voucher_keys`, `used_vouchers`, `deck_enhancements` (enhancement gates), `hands_played`
(Planet softlocks, Telescope).  The incremental `acquire`/`release_*`/`remove_owned` calls are
still made at the natural moments so observers see a live RunState; the resync makes hand-edited
state (tests, envs, harness `add_owned_joker`) safe.

**Ante semantics changed to the game's:** `game.ante` increments at boss defeat (`ease_ante`
before the post-boss shop), not when the post-boss shop is left.  `run_state.ante == game.ante`
always.  `_advance_blind` only resets `blind_idx`.

**Deck order:** `full_deck` is created in the game's `sort_id` order (`C2..C9,CA,CJ,CK,CQ,CT,D2,…`);
`Card.id` is a global creation counter, so `sorted(full_deck, key=id)` is the `G.playing_cards`
order every `pseudoshuffle`/`pseudorandom_element` indexes.  The first 8 cards dealt on
`7I4M53DL` match the ground truth's `deck_order_unverified.small` (Blueprint's model) — so the
unverified deck order now has a second independent source.

## 2. Tag path decision: **option B** (TAGS_NOTES §2)

`generate.generate_shop` / `reroll_shop` own the shop-time tags (`store_joker_create`,
`store_joker_modify`, `voucher_add`) because they change which RNG calls happen and are
oracle-verified.  Protocol: before each fresh shop / reroll `rs.tags = tag_state.keys()`; after
the call `_absorb_tag_triggers()` marks `tag_state.tags[i].triggered` for `i in rs.triggered_tags`,
purges, and resets `rs.tags`/`rs.triggered_tags` so indices stay aligned.  `tags.py` drives
everything else through `_GameTagContext` (game.py, bottom): immediate tags at skip/blind select,
`new_blind_choice` (pack interrupts, Boss Tag), `eval` (Investment), `round_start_bonus`
(Juggle), `shop_start` (D6 → `reroll_cost = 0`), `shop_final_pass` (Coupon →
`ShopItem.couponed`), Double Tag copying, Top-up (`spawn_joker_to_slots` → `Joker1top<a>`).
The unused hooks (`create_shop_joker`, `add_shop_voucher`, …) are implemented anyway so option A
is a switch away.  `game.tags` is a property alias of `game.tag_state` (harness hook 6).

Verified: D6 + Coupon (`test_d6_and_coupon`), Uncommon Tag forcing slot 1 couponed
(`test_uncommon_tag_forces_shop_slot_via_generate`), Double+Investment, Charm pack interrupt from
blind select, Boss Tag reroll, Skip/Economy payouts.

## 3. The Order switch — status: flag only (no-op)

`game.queue_scope = "ante" | "run"` exists and is cloned; `"ante"` (default, MLB/vanilla) needs
nothing.  `"run"` is **not implemented**: `generate.Keys` builds every ante-suffixed key via the
module-level `_ante_str(ante)` with no hook, and `generate.py` is Agent C's file.  What generate.py
needs (one change): a per-state suffix, e.g. `RunState.key_scope: str = "ante"` and
`Keys.ante_suffix(state) -> str` returning `""` when `key_scope == "run"`, used by `cdt`, `rarity`,
`joker_pool` (non-legendary), `center_pool` (Tarot/Planet/Spectral/Enhanced/Voucher/Tag),
`edition`, `front`, `soul`, `sticker_poll`, `rental_poll`, `pack`, `stdset`, `standard_edition`,
`stdseal`, `stdsealtype`, `halu`, `new_round_shuffle`, `cashout_shuffle`, `idol/mail/ancient/castle`
(resample keys follow automatically).  Then `_init_game_vars` sets `rs.key_scope = self.queue_scope`.
Whether The Order also drops the suffix from `nr`/`cashout`/`idol…` must be checked against the
MLB mod's source (not in `_reference`).

## 4. Booster state machine (§7 item 0 fixed)

`step({"type":"buy"})` on a booster → `buy_item` debits and fills `booster_choices` → `step`
enters `State.BOOSTER_OPEN`.  `legal_actions()` offers `skip_booster` plus `pick_booster` index
combinations over the **grantable** cards (free slot, or Negative).  `pick_booster` grants up to
`booster_picks_remaining` (mega = 2), closes when the picks are used up / nothing grantable
remains / nothing was granted in the call (so legacy envs that pick blindly never wedge).
`skip_booster` fires `on_booster_skipped` (Red Card).  Closing releases unpicked cards
(`release_pack`) and returns to `_booster_return_state` (`SHOP`, or `BLIND_SELECT` for tag packs,
re-running the `new_blind_choice` pass).  `buy_item` itself does not change state, so `env_v5`'s
direct calls keep its own substate working.

`BoosterChoice` (shop.py) is a plain object, not a str: the harness's `item_from_engine` treats
`str` as a bare key and would lose editions/seals.  **`env_v5._step_pack_open` still type-checks
`str`/tuple and silently grants nothing for `BoosterChoice`** (pre-existing env; its tests only
check substates) — one `isinstance` change for its owner, noted in §9.

## 5. Created cards (consumables.py) — what changed vs. the card text

Judgement (`jud`), The Soul (`Joker4` after the discarded `rarity<a>sou`), Wraith (Rare, **sets
money to $0**, was −$3), Emperor / High Priestess (`min(2, free slots)`, two different cards),
The Fool (`run_state.last_tarot_planet`, set by every used Tarot/Planet; unusable before any use
or after a Fool; was "last tarot, else last planet"), Familiar / Grim / Incantation (**destroy one
RANDOM card in hand** on `random_destroy`, created cards go **to the hand**, need >1 card in hand;
was "destroy the selected card, add to deck"), Aura (**a selected playing card**, editionless;
was a joker), Sigil / Ouija (keyed picks), Ectoplasm (**Negative on a random editionless joker,
−1 hand size**; was +1 slot and a nonsense mult penalty), Hex (keyed pick over editionless
jokers), Ankh (keyed pick; **chosen joker + its copy survive**; was copy only), Immolate
(`pseudoshuffle('immolate')` over the hand in sort_id order), Blue Seal (`gen.blue_seal`).
Joker/tag-created cards: `grant_created(spec)` for every `generate.CREATE_SPECS` name;
`_materialize(token)` resolves `"create:<spec>"`, `"common_joker"` (Riff-raff → `Joker1rif<a>`),
`"stone_card"` (Marble → `marb_fr`), `"random_enhanced_card"` (Certificate → `cert_fr`+`certsl`),
`"copy_card:r:s"` (DNA), `"double_tag"`, and real `c_*` keys; unknown legacy sentinels pass
through unchanged (W3 has since rewritten the producers to real keys via `create_consumable`).
Not done: Invisible Joker's copy (`invisible` over jokers by `sort_id`; engine jokers have no
creation ids — W5), Perkeo (`negative_tarot` sentinel — W5), Ouija/Ectoplasm permanent hand-size
(`_start_blind` resets `hand_size` each blind; needs a persistent modifier field — W5).

## 6. Vouchers routed into `run_state` (consumables.apply_voucher)

`used_vouchers` on every redemption; `pools.SHOP_RATE_BY_VOUCHER` (Tarot/Planet Merchant+Tycoon,
Magic Trick, Illusion → `playing_card_rate`, Hone/Glow Up → `edition_rate`); Overstock (+1
`shop_joker_max` and an immediate `_fill_shop_slot` when bought in a shop); Hieroglyph/Petroglyph
(`ante −1` on both views, **clamped at 1** — the real game reaches ante 0 with `…0` keys, the
engine's blind table starts at 1; Petroglyph also −1 discard, was missing); Director's Cut / Retcon
→ the `reroll_boss` action (Director's Cut was a free *shop* reroll — REKEY §6.11).  Omen Globe /
Telescope / Observatory need only `used_vouchers` (generate reads them).  `game.shop_joker_slots` /
`game.shop_card_slots` are gone: the shelf is `run_state.shop_joker_max` slots shared by jokers,
consumables and playing cards (the real shop has no card row).

## 7. Other engine changes in my areas

* `ShopItem` gained `enhancement/seal/front` (Magic Trick cards, kind `"card"`), stickers,
  `couponed/from_tag`, `center` (booster pool key; `key` stays the type key for the envs), `set`.
  `discounted_price` is now `Card:set_cost`'s `floor((cost+0.5)*(100-d)/100)`, min 1, couponed 0.
* Pack/shelf jokers carry editions (+$2/3/5/5 markup), stickers land in `JokerInstance.state`.
* Negative jokers/consumables take no slot (`joker_slots`/`consumable_slots` +1 on acquire, −1 on
  joker sell).
* `reroll_shop` honours D6 (`reroll_cost = 0` base), Chaos (`free_rerolls_remaining` +1 per Chaos
  per shop via `passive_modifiers`), Credit Card (`bankrupt_at` −$20), Astronomer (Planets and
  Celestial packs $0).
* `round_picks` (`idol`/`mail`/`ancient`/`castle`) drawn at run start and after every round with
  the game's keys — consumed by W3's `round_cards.py` for The Idol / Mail-In Rebate / Ancient /
  Castle.
* Harness helpers: `debug_win_blind()`, `debug_add_joker(key, edition)`; `can_reroll_boss()`.
* `JokerInstance.clone()` copies one container level (regression:
  `TestJokerInstanceClone`); `BalatroGame.clone()` copies `run_state`/`tag_state` with flat
  copiers (77 µs/clone vs 108 µs with deepcopy; pre-W2 63 µs).

## 8. game.py functions touched (targeted edits only)

`__init__` (added `deck_key` kwarg), `_init_game_vars`, `clone` (new fields + fast copiers),
`_prepare_next_blind`, `_start_blind` (`new_round` + Juggle hook + `_materialize_pending`; W3
later replaced the joker loop with `fire_hook`), `_init_deck`, `step` (BOOSTER_OPEN entry,
`skip_booster`, `reroll_boss`), `legal_actions` (BLIND_SELECT/SHOP/BOOSTER_OPEN), `_end_round`,
`_end_shop`, `_advance_blind` (new), `_skip_blind`, `_end_blind_and_enter_shop`, booster machine
(`_open_booster`, `_can_grant_choice`, `_grant_choice`, `_pick_booster`, `_skip_booster`,
`_close_booster`), new helpers (`_playing_cards_sorted`, `_sync_run_state`,
`_absorb_tag_triggers`, `_tag_ctx`, `_visible_hands`, `_enter_blind_select`,
`_handle_blind_choice`, `_round_end_resets`, `grant_created`, `_materialize`,
`_materialize_pending`, `debug_*`, `can_reroll_boss`, `_reroll_boss`, `tags` property), module
level `_GameTagContext`, `_fast_clone_run_state`, `_fast_clone_tag_state`.  Not touched:
`_play_hand`, `_discard`, `_use_consumable`, `_hook_ctx`, `_draw_to_full`, `_obs` (W3/W5).

## 9. Edits outside my ownership (each minimal, each flagged)

1. **`oracle/engine_parity.py` `EngineDriver.win_blind/leave_shop`** — the harness recorded
   "ante N+1 start" when leaving the *after-Big* shop of ante N, i.e. before the ante-N boss is
   fought; the real game (and the engine) draw the next ante's boss/tags/voucher at Cash Out
   *after* the boss dies.  Now recorded when a Boss was just cleared; the old point is kept as a
   fallback for engines without the hooks.  This was the only thing between 0/126 and 126/126.
2. **`tests/test_engine_reachability.py` `skip_then_score`** — asserted `SHOP` after a skip;
   now `BLIND_SELECT` (after closing a tag pack if any).  Needed for `j_throwback`.
3. **`balatro_sim/env_v5.py` `_step_play`** — two lines: advance `ROUND_EVAL` after a cleared
   blind.  env_v5 never handled `ROUND_EVAL`; it only ever reached shops through the sim-invented
   skip→shop, so with faithful skipping it could never enter a shop.  env_v5's
   `_step_pack_open` still needs the `BoosterChoice` `isinstance` change (left to its owner).
4. Tests in `engine/tests` outside my nominal areas that encoded sim-invented behaviour:
   `test_game_transitions.TestSkipBlind`, `test_play_env_v5.TestBlindSelect` (skip→shop/+$5),
   `test_shop_v5._advance_to_shop`/`_game_in_shop`/`test_agent_alternates_play_shop`,
   `test_rewards_v5._advance_to_shop`/`test_quality_set_on_skip_blind`, `test_boss_blinds(_h4)`
   (face-down bosses are now drawable), `test_consumables` (Overstock → `shop_joker_max`,
   Director's Cut, Fool, Familiar/Grim/Incantation, Wraith, Ectoplasm, Ankh, Aura),
   `test_determinism`/`test_joker_catalogue`/`test_env_mp` (`random_joker_key` is gone; bans flow
   through `run_state.banned_keys`), `test_game_keys` (Fool needs a prior use),
   `test_env_mp.test_different_actions_different_scores` (on the real `nr1` deal the 1-card and
   5-card plays both scored High Card Ace; p2 now plays the pair).

## 10. What I could not delegate / known gaps (W5 worklist from this side)

* **`bl_fish` is now drawable** (generate owns the pool; resampling it would break parity) and
  `_draw_to_full`'s wrong hand-size model for it is reachable — `_draw_to_full` is not mine.
  `bl_house/wheel/mark` are drawn and do nothing (effect unmodelled, as specified).
* `env_mp._revive_boss_if_needed` calls `generate_shop(game)` after a GAME_OVER without the
  round/ante transition (W7 flagged it); it now at least generates through `run_state`.
* Invisible Joker copy, Perkeo negative copy, Ouija/Ectoplasm permanent hand size (above).
* Hieroglyph at ante 1 clamps to ante 1 on both views (game: ante 0).
* `state_signature()` (optional harness hook) not provided; the harness fallback is used.
* The two remaining Showman subtleties: `run_state.showman` is "owned", not "owned and not
  debuffed" (Verdant Leaf / Crimson Heart disable) — effect-side.
* `engine_parity._apply_voucher_rates` expects `SHOP_RATE_BY_VOUCHER` values to be dicts; they
  are `(attr, value)` tuples, so rates are not applied in `--reference generate --buy-vouchers`
  replays (harness-side, not edited).
