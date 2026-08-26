# REKEY_NOTES — Phase 1 W1: the engine speaks game keys

**Agent P1-rekey, 2026-08-21.** `rng/pools.py` is now the single source of truth for every
key / name / rarity / cost / eligibility flag in `engine/balatro_sim/`. Nothing is hand-typed
any more: `balatro_sim/game_keys.py` loads `pools` at import time and every catalogue in
`shop.py`, `consumables.py`, `constants.py` and `game.py` is derived from it.

**Tests: 1279 passed / 13 skipped / 0 failed** (`python -m pytest engine/tests -q`, from the
repo root). The baseline at the start of W1 was 847 / 13; the delta is W1's new
`tests/sim_tests/test_game_keys.py` (263 tests) plus the W6 (`test_tags.py`) and W7
(`test_env_rng_isolation.py`) suites that landed concurrently in the same tree.

---

## 1. New module: `balatro_sim/game_keys.py`

Imports `rng.pools` (namespace package, repo root on `sys.path`); falls back to appending the
repo root, then to loading `pools.py` straight from its file path (it is pure data). Exposes:

| name | from pools | notes |
|---|---|---|
| `JOKERS`, `JOKER_BY_KEY`, `JOKER_KEYS`, `JOKER_KEYS_BY_RARITY`, `JOKER_NAME/RARITY/COST`, `JOKER_BY_NAME` | `JOKERS`, `JOKERS_BY_RARITY` | `RARITY_NAME = {1: Common .. 4: Legendary}` |
| `TAROT_*`, `PLANET_*`, `SPECTRAL_*`, `PLANET_HAND`, `HAND_PLANET`, `CONSUMABLE_COST/SET` | `TAROTS`, `PLANETS`, `SPECTRALS` | |
| `VOUCHER_KEYS/NAME/COST`, `VOUCHER_REQUIRES`, `VOUCHER_UPGRADE_OF` | `VOUCHERS` | `requires` carried for W2 |
| `BOOSTER_TYPES` (15), `BOOSTER_TYPE_KEYS`, `BOOSTER_CENTER_KEYS` (32), `booster_type_key()` | `BOOSTERS` | art variants `_1.._4` collapsed; per-type `weight` = sum of its centers |
| `BOSS_KEYS[_REGULAR|_SHOWDOWN|_ALPHA]`, `BOSS_NAME`, `BOSS_MIN_ANTE`, `BOSS_MAX_ANTE`, `BOSS_CHIP_MULT_RAW`, `BOSS_DOLLARS` | `BLINDS` | |
| `TAGS/TAG_KEYS/TAG_NAME/TAG_BY_KEY`, `DECKS/DECK_*`, `STAKES/STAKE_*` | `TAGS`, `BACKS`, `STAKES` | catalogue only |
| `ENHANCEMENT_KEYS`, `EDITION_KEYS`, `SEAL_KEYS` | | |

`constants.py` re-exports the tag/deck/stake tables (`constants.TAGS`, `constants.DECKS`,
`constants.STAKES`, …) and adds `ENHANCEMENT_KEY` / `EDITION_KEY` (engine display name ↔ game
`m_*` / `e_*` key) for the generation layer. The engine's runtime representation of
enhancements/editions/seals on `Card` is unchanged (short names).

## 2. Jokers

### 2.1 Renamed keys (21 from the survey + 9 more found in the registry)

| old sim key | game key | | old sim key | game key |
|---|---|---|---|---|
| `j_greedy_mult` | `j_greedy_joker` | | `j_trading_card` | `j_trading` |
| `j_lusty_mult` | `j_lusty_joker` | | `j_spare_trousers` | `j_trousers` |
| `j_wrathful_mult` | `j_wrathful_joker` | | `j_seltzer` | `j_selzer` (game typo) |
| `j_gluttonous_mult` | `j_gluttenous_joker` (game typo) | | `j_golden_ticket` | `j_ticket` |
| `j_business_card` | `j_business` | | `j_smeared_joker` | `j_smeared` |
| `j_space_joker` | `j_space` | | `j_glass_joker` | `j_glass` |
| `j_to_do_list` | `j_todo_list` | | `j_showman` | `j_ring_master` |
| `j_square_joker` | `j_square` | | `j_the_idol` | `j_idol` |
| `j_gift_card` | `j_gift` | | `j_invisible_joker` | `j_invisible` |
| `j_mail_in_rebate` | `j_mail` | | `j_burnt_joker` | `j_burnt` |
| `j_stone_joker` | `j_stone` | | `j_the_duo/trio/family/order/tribe` | `j_duo/trio/family/order/tribe` |
| `j_wee_joker` | `j_wee` | | `j_oops_all_sixes` | `j_oops` |

Applied with word-boundary regex across `balatro_sim/**`, `tests/**` and `benchmarks/`.
`synergy.py` / `quality.py` (strategy-label tables) were re-keyed and de-duplicated where two
spellings collapsed onto one key (`j_duo`… ×2, `j_wee` ×2, `j_lucky_joker`→dropped).

### 2.2 Rarity / cost

`JOKER_CATALOGUE` (shop.py) is now a comprehension over `pools.JOKERS` in game `order`; the
`_reg()` tier tables ($6/$7/$8/$20, 79 wrong rarities, 107 wrong costs, 11 keys registered twice,
6 duplicate spellings) are gone. Entry shape: `{key, name, rarity, rarity_id, price, order,
blueprint_compat, eternal_compat, perishable_compat}`. `RARITY_WEIGHTS` / `random_joker_key` are
untouched (W2 replaces them with `generate.py`).

### 2.3 Duplicate implementations — every decision

`JOKER_REGISTRY` is now a dict subclass that **raises on a second registration**, and
`jokers/__init__.py` raises at import if the registry ≠ the 150 pools keys. Before W1, 36 keys
were registered in two modules with "last import wins" (order economy → scaling → hand_type →
misc → chips → mult), and for several the winner was wrong. Each pair was read against the real
card text; "kept" = the single implementation now registered under the game key.

| game key | kept (module) | deleted (module) | reasoning |
|---|---|---|---|
| `j_8_ball` | misc: 1/4 chance per scored 8 → Tarot | chips: +20 chips (**was winning**) | Real: creates a Tarot card. |
| `j_four_fingers` | misc: sets `ctx.four_finger_mode` | chips: +10 chips (**was winning**) | Real: 4-card flushes/straights; no chips. |
| `j_satellite` | misc: tracks `on_planet_used`, +$1/unique planet at round end | mult: empty TODO (**was winning**) | Only the misc copy works. |
| `j_drivers_license` | chips: counts Stone as enhanced | misc: excludes Stone | `m_stone` is an enhancement in the game; ≥16 enhanced → X3. |
| `j_duo/trio/family/order/tribe` | mult `_The*`: X2/X3/X4/X3/X2 if hand contains type | chips: +2/+4/+8/+3/+2 Mult | Real cards are XMult. The `j_the_*` spellings were the Rare copies; chips' `j_duo` copies were wrong and mis-tiered Uncommon. |
| `j_space` | mult `_SpaceJoker`: 1/4 level up played hand | scaling `_Space`: empty TODO | |
| `j_ticket` | economy `_GoldenTicket`: played Gold card → $4 | scaling `_Ticket`: "+3 chips per $10 held" | Scaling copy was not the real card at all. |
| `j_ring_master` (Showman) | misc `_Showman`: shop-enter state flag (stub; W2 handles duplicates in generation) | misc `_RingMaster`: "reroll boss blind" | `_RingMaster` described Director's Cut, not Showman. |
| `j_oops` | misc `_OopsAllSixes` (flag for probability system) | the `j_oops = j_oops_all_sixes` alias line | single impl, game key. |
| `j_wee` | misc: permanently +8 chips per scored 2 | scaling (TODO), mult `_WeeJoker` (non-permanent +8) | Real effect is permanent scaling. |
| `j_stone` | misc: +25 chips per Stone card in full deck | scaling: TODO stub | |
| `j_flash` | misc: +2 Mult per reroll (`on_reroll`) | scaling: TODO stub | (see §6: hook wiring unverified) |
| `j_reserved_parking` | misc: held face cards, 1/2 → $1 | economy: *played* face cards | Real: held in hand. |
| `j_swashbuckler` | misc: Mult = sum of joker sell values | scaling: +1 Mult per joker | Real: sell value (see §6: should exclude itself). |
| `j_midas_mask` | misc: uses `ctx.is_face_card` (Pareidolia-aware) | scaling: `card.is_face_card` | |
| `j_banner` | mult: +30 chips/discard | chips: +40 | Real: 30. |
| `j_mystic_summit` | mult: +15 **Mult** at 0 discards | chips: +15 chips | |
| `j_fibonacci` | mult: +8 **Mult** per A/2/3/5/8 | chips: +8 chips | |
| `j_smiley` | mult: +5 **Mult** per face | chips: +5 chips | |
| `j_misprint` | mult: +0..23 **Mult** | chips: +0..23 chips | |
| `j_raised_fist` | mult: 2× lowest held rank → Mult | chips: +50 if all same rank | |
| `j_loyalty_card` | mult: X4 every 6th hand | chips: +4·n Mult ramp | |
| `j_droll` | mult: +10 Mult if Flush | chips: +30 chips "retrigger odd" | chips copy was a different card entirely. |
| `j_acrobat` | mult: X3 on last hand | chips: +18 chips | |
| `j_stencil` | mult: X(empty slots) | chips: +50 chips/empty slot | |
| `j_bootstraps` | mult: +2 Mult per $5, uncapped | chips: capped at $50 | Real card is uncapped. |
| `j_jolly/zany/mad/crazy` | mult: `_has_hand` containment (Full House counts as Pair etc.) | scaling: substring checks | |
| `j_photograph` | mult: X2 first face card, reset per round | scaling: +2 Mult per face | Neither is exactly right — see §6. |
| `j_ancient` | mult: X1.5 per scoring card of stored suit | scaling: face cards only + `on_init` referencing undefined `ctx` | See §6 for the missing suit change. |
| `j_flower_pot` | mult: 4 suits among all played (Stone excluded) | scaling: scoring cards, Stone not excluded | See §6. |
| `j_throwback` | mult: X0.25 per skip | scaling: X2 per skip | |
| `j_seeing_double` | mult (Stone excluded) | chips | |
| `j_bloodstone`, `j_scary_face`, `j_shoot_the_moon`, `j_walkie_talkie`, `j_triboulet`, `j_stuntman` | mult | identical duplicates | no behaviour change. |

### 2.4 Dead aliases (the survey's five)

* `j_lucky_joker` — **removed**. Not a real joker (the "+20 Mult per Lucky trigger" is the Lucky
  *card*'s own effect). `synergy.py` entry dropped.
* `j_oops_all_sixes` → `j_oops`; `j_space_joker` → `j_space`; `j_golden_ticket` → `j_ticket`;
  `j_showman` → `j_ring_master`. The game keys now carry the single implementation; the former
  "real-key stubs" under those names were deleted (table above).
* `j_lucky_cat` was already correct.

246 orphan placeholder comments of the form `# ── j_x: already in foo.py` (many now pointing at
deleted copies) were stripped from `jokers/*.py`; no code lines were touched by that pass.

## 3. Consumables

* Planets `pl_*` → `c_*`, spectrals `s_*` → `c_*`; `PLANET_HAND`, `*_NAME`, `ALL_*` derived from
  pools in game pool order.
* **`c_heirophant` is the game's key** (typo in game.lua). The brief said the opposite; pools and
  NOTES_POOLS §4.2 are authoritative, so the sim's `c_hierophant` → `c_heirophant`.
* `env_v5.py` had a phantom `c_wheel` in its tarot-target tables → `c_wheel_of_fortune`.
* `apply_spectral` branches re-keyed; every tarot / planet / spectral key is asserted to dispatch
  (`test_game_keys.py`).

## 4. Vouchers

`v_overstock` → `v_overstock_norm`; `VOUCHER_NAME`/`ALL_VOUCHERS` from pools (32); new
`VOUCHER_REQUIRES` exported from `consumables.py` for W2. Added to `apply_voucher`:

| key | effect implemented | status |
|---|---|---|
| `v_seed_money` | `game.interest_cap = 10` (interest on up to $50) | done; `interest_cap` is a new game field, cloned, used at cash-out |
| `v_money_tree` | `game.interest_cap = 20` (up to $100) | done |
| `v_blank` | nothing | done (by definition) |
| `v_antimatter` | `game.joker_slots += 1` | done |
| `v_retcon` | recorded as owned only | **STUB/TODO(W2/W5)** — needs boss-reroll plumbing |

Shop-rate / edition-rate / pack-content vouchers (`v_hone`, `v_glow_up`, `v_omen_globe`,
`v_tarot_*`, `v_planet_*`, `v_magic_trick`, `v_illusion`, `v_telescope`, `v_observatory`) were
already "owned-only" in the sim and remain so; W2's delegation to `generate.py` gives the shop-side
ones for free.

## 5. Bosses / boosters / tags

* Showdowns renamed `bl_amber/cerulean/crimson/verdant/violet` →
  `bl_final_acorn/bell/heart/leaf/vessel` everywhere (game.py, env_sim, tests). `BOSS_CHIP_MULT`
  keyed on `bl_final_vessel`.
* `REGULAR_BOSS_BLINDS` = pools' 23 regular keys minus `UNMODELLED_BOSS_BLINDS`;
  `ALL_REGULAR_BOSS_BLINDS` (23) added; `SHOWDOWN_BOSS_BLINDS` from pools.
  `BOSS_MIN_ANTE` / `BOSS_MAX_ANTE` exposed for W2 (selection is still a flat uniform draw here).
* **`bl_fish` added → `UNMODELLED_BOSS_BLINDS`** (face-down draw needs hidden info; TODO(W5)).
  Note `game._draw_to_full` already contains a `bl_fish` branch that models it as a hand-size
  penalty — that is not the real effect; it is unreachable while the key is unmodelled.
* Boosters: `BOOSTER_CATALOGUE` keyed `p_<kind>_<size>` (`p_arcana` → `p_arcana_normal`, …),
  derived from the 32 centers; **15 types, not 13** (5 kinds × 3 sizes — the survey's "13" was the
  sim's count with the two megas missing). `p_standard_mega` / `p_buffoon_mega` added.
  `BOOSTER_PICKS` (choose 2 for megas) replaces the `"mega" in key` heuristic. Per-type weights
  in `game_keys.BOOSTER_TYPES[...]["weight"]` for W2.
* Tags / decks / stakes: catalogue only via `constants.TAGS/DECKS/STAKES` (+ `*_KEYS`, `*_NAME`,
  `*_BY_KEY`). No behaviour.

## 6. Found but NOT fixed (out of W1 scope) — for W5

1. **Photograph** (`mult._Photograph`): resets on `on_round_end`; real card fires on the *first
   face card of every hand* (`context.scoring_hand` first `is_face()`). Should reset in
   `pre_score`.
2. **Ancient Joker**: suit never changes at end of round (real: new suit each round); suit is
   lazily chosen on first score instead of at creation.
3. **Flower Pot**: reads `ctx.all_cards` (all played); real reads the scoring hand.
4. **Swashbuckler**: sums sell value of *all* jokers incl. itself; real excludes itself.
5. **Joker Stencil**: `MAX_SLOTS = 5` hard-coded; should read `game.joker_slots` (Antimatter /
   Ectoplasm change it — and W1 just made Antimatter purchasable).
6. **Flash Card**: depends on an `on_reroll` hook; `shop.reroll_shop` never calls it (grep shows
   no caller). Same class of bug as the pre-2026-07-29 passives.
7. **Hallucination** / **Matador**: depend on `on_booster_opened` / `on_boss_ability_triggered`
   hooks with no callers.
8. **Stuntman**: −2 hand size not applied (`game._start_blind` handles juggler/drunkard/
   troubadour/merry_andy by key but not stuntman).
9. **Showman** (`j_ring_master`) is a state-flag stub; real behaviour belongs to generation (W2).
10. **`bl_fish`** `_draw_to_full` model is wrong (see §5).
11. **Director's Cut** is modelled as a free *shop* reroll; real effect is one boss reroll per ante
    for $10. Retcon (unlimited) stubbed on the same missing plumbing.
12. `random_joker_key` still uses `RARITY_WEIGHTS` 70/20/8/2 over the catalogue and ignores
    `enhancement_gate` / `gros_michel_extinct` flags — W2 replaces it with `generate.py`.

## 7. Files touched outside W1's nominal ownership (one-line key fixes — owners please rebase)

| file (owner) | change |
|---|---|
| `balatro_sim/env_v5.py` (W7) | `c_hierophant` → `c_heirophant`; phantom `c_wheel` → `c_wheel_of_fortune` |
| `balatro_sim/env_v7.py` (W7) | `j_invisible_joker` → `j_invisible` |
| `tests/sim_tests/test_env_rng_isolation.py` (W7) | `j_business_card` → `j_business`, `j_space_joker` → `j_space` (the old keys had silently become no-ops, which broke 2 of its env_v5 trajectory tests) |
| `benchmarks/bench_clone_step.py` | `j_space_joker` → `j_space`, `pl_*` → `c_*` |

`env_sim.py`, `card_selection.py`, `mp_game.py`, `env_mp.py` needed no changes (their `bl_*` /
`j_*` literals were already game keys).

## 8. Where the guards live

`tests/sim_tests/test_game_keys.py` — registry ⇔ pools 1:1 (runtime + source-site count),
catalogue name/rarity/cost/order = pools for all 150, 22/12/18 consumables dispatch, 32 vouchers
accepted, 28 bosses partitioned (19 modelled + 4 unmodelled + 5 showdown), 15 booster types with
cost/cards/picks/weights = pools, tags/decks/stakes via constants, and a grep that fails if any
legacy key literal (`"j_greedy_mult"`, `"pl_*"`, `"s_*"`, `"bl_amber"`, `"p_arcana"`, …) reappears
under `balatro_sim/`. `jokers/__init__.py` additionally raises at import if the registry drifts.
