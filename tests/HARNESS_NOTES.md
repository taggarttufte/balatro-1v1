# W8 harness notes — engine invariants, reachability, engine-vs-ground-truth

**Owner:** P1-harness (W8). **Files:** `tests/test_engine_invariants.py`,
`tests/test_engine_reachability.py`, `oracle/engine_parity.py`, this file.
**Status 2026-08-21:** written against the *target* Phase-1 architecture; expected RED today.
Goes green as W2 (delegate generation) / W3 (effect-roll keys) / W5 (bug sweep) land.

## How to run

```
python -m pytest tests -q                                  # everything under tests (incl. Phase 0 oracles)
python -m pytest tests/test_engine_invariants.py -q
python -m pytest tests/test_engine_reachability.py -q -rx  # -rx lists the xfail reasons (= findings)
python -m oracle.engine_parity --probe                     # which target hooks the engine exposes
python -m oracle.engine_parity --seeds 7I4M53DL,ALEEB --antes 1-3          # engine vs JSON corpus
python -m oracle.engine_parity --antes 1-8 --rerolls 5 --quiet             # all 126 seeds
python -m oracle.engine_parity --seeds ALEEB --reference generate --reroll-every-shop --buy-shelf
python -m oracle.engine_parity --seeds ALEEB --buy-vouchers                # voucher_chain_if_bought branch
```

All three files import the **fork** (`engine/balatro_sim`) through `engine_parity.import_engine()`,
which puts `engine` first on `sys.path` and refuses to run if the repo-root BRL `balatro_sim` won
(same guard as `engine/conftest.py`). Run from the repo root; `tests/conftest.py` puts `mp/`
on the path so `rng.*` / `oracle.*` import as top-level packages.

## Counts at hand-off (2026-08-21, after W1 re-key + W6 tags landed, before W2/W3)

| file | pass | xfail | fail | runtime |
|---|---|---|---|---|
| `test_engine_invariants.py` | 3 | 1 | 10 | 0.3 s |
| `test_engine_reachability.py` | 185 | 4 | 46 | 1.0 s |
| Phase 0 oracles (`test_rng_core`, `test_generate_oracle`) | 146 | 0 | 0 | 1.7 s |
| **`python -m pytest tests`** | **334** | **5** | **56** | 2.7 s |
| `engine_parity.py --antes 1-3` | 0/126 seeds exact (first mismatch: ante-1 voucher, then every shelf slot); `boss`/`tags` "not produced" | | | ~6 s |

Every failure was checked to be an engine gap, not a harness bug (the probe was also checked to
*fail* on no-op scenarios: a `j_joker` at blind-select / round-end, a Jolly Joker on a High Card).

## Engine hooks the harness needs (ONE list for W2 / W3 / W5)

`python -m oracle.engine_parity --probe` prints this list with ok/missing per hook. Names are
the ones the harness reads; anything else is read through the public `step()` API.

| # | hook | who | used by | status today |
|---|---|---|---|---|
| 1 | `BalatroGame(seed="7I4M53DL")` — Balatro seed string, normalised with `core.normalize_seed` (today `random.Random(str)` silently accepts it, so the probe says ok, but the streams are wrong) | W2 | all | accepted, not keyed |
| 2 | `game.run_state` — `generate.RunState` kept in sync: `ante`, `used_jokers`/`owned_*` via `acquire`/`release_shop`/`release_pack`/`remove_owned`, `used_vouchers`, `showman`, rates (`joker_rate`, `edition_rate`, `shop_joker_max`, …), `hands_played`, `deck_enhancements`, `pool_flags`, `blind_tags`, `boss_blind`. Duck-typed (W2 may import it as `rng.generate`). | W2 | invariants 2/4/5, reachability (Showman flag, voucher rates), parity | missing |
| 3 | `game.run_state.rng` — `core.PseudoRandom`; **`game.rng` deleted** (probe flags `keyed_rng` only when `game.rng` is gone) | W3 | invariants 1 | missing |
| 4 | `game.boss_blind` *or* `run_state.boss_blind` — this ante's boss, known at ante start (run start / Cash Out), not drawn in `_prepare_next_blind` | W2 | parity (`boss` field), invariants 1 | missing (harness falls back to `current_blind.boss_key` at the Boss blind) |
| 5 | `game.blind_tags` *or* `run_state.blind_tags` — `{'Small': tag_key, 'Big': tag_key}` drawn at run start and Cash Out | W2 + W6 | parity (`tags`), invariants 1/4 | missing ("not produced") |
| 6 | `game.tags` — owned tag keys (skip rewards); `_skip_blind` must add the blind's tag, not `+$5` | W2 + W6 | reachability signature, Diet Cola / Double Tag | missing |
| 7 | Buying a booster enters `State.BOOSTER_OPEN` with `game.booster_choices` populated; `skip_booster` releases the cards (`run_state.release_pack`); `pick_booster` acquires | W2 (§7 item 0) | parity (packs), invariants 1/2, Red Card / Hallucination probes | **missing — engine bug**: state stays `SHOP`, choices stale |
| 8 | `booster_choices` entries expose `.key` (+ `.edition`, `.enhancement`, `.seal`, `.front` for playing cards). Today: bare key strings / `("card", Card)` tuples — the adapter `engine_parity.item_from_engine` already accepts both, plus `generate.CardGen`, engine `Card`, dicts, `ShopItem` | W2 | parity | ok (adapter) |
| 9 | `current_shop` items carry game keys (`j_*`, `c_*`, `v_*`, `p_*_<size>[_n]`) and game edition names (`foil`/`holo`/`polychrome`/`negative` or `e_*`; the adapter also maps `Foil`/`Holographic`/…). Shelf = `shop_joker_max` slots (2), not 2 jokers + 2 cards | W1 ✅ / W2 | parity, invariants | keys ok since W1; slot count still 4 |
| 10 | `game.debug_win_blind()` — clear the current blind without scoring (harness helper; keeps effect streams untouched). Fallback today: harness sets `chips_scored = target`, `state = ROUND_EVAL` | W2 (nice-to-have) | parity, invariants, reachability | missing (fallback works) |
| 11 | `game.debug_add_joker(key, edition=None)` — own a joker AND do the `run_state.acquire` bookkeeping. Fallback: harness appends `JokerInstance` and calls `run_state.acquire` itself when `run_state` exists | W2 (nice-to-have) | invariants 2, reachability | missing (fallback works) |
| 12 | `game.state_signature()` — canonical, Card-id-independent snapshot incl. `run_state` + PseudoRandom state. Fallback: `test_engine_invariants.state_signature` (skips `Card.id`-keyed fields `_played_this_ante`, `_forced_card_id`) | W2 (nice-to-have) | determinism tests | missing (fallback works) |
| 13 | `clone()` deep-copies `run_state` and clones `PseudoRandom` (`RunState.clone()` exists) | W2 | `test_clone_then_diverge_leaves_original_untouched` | xfail today |
| 14 | Voucher effects that are generation-relevant must land in `run_state`: Overstock → `shop_joker_max`, Hone/Glow Up → `edition_rate`, Tarot/Planet Merchant/Tycoon → rates (`pools.SHOP_RATE_BY_VOUCHER`), Magic Trick/Illusion → `playing_card_rate`, Omen Globe/Telescope/Observatory flags, Hieroglyph/Petroglyph → `ante` | W2 | reachability (12 voucher probes) | missing |

Engine-side **policy** assumptions the parity harness makes (document if W2 changes them):
post-boss shop belongs to the *next* ante (`ease_ante` before the shop; the harness counts visits:
2 in ante 1, then 3 per ante — independent of when `game.ante` increments); the forced first
Buffoon pack consumes no `shop_pack1`; packs are opened before rerolls; art suffix `_1/_2` is ignored.

## What each test asserts, and why it is red today

### `test_engine_invariants.py`

| test | asserts | today |
|---|---|---|
| `test_effect_rolls_do_not_move_generation[3 seeds]` | A and B own the same 3 conditional jokers (Bloodstone, 8 Ball, Business Card); A plays 5 Hearts with 3 Lucky + 2 Glass (≈15 effect rolls), B plays a quiet hand. Next shelf, voucher, pack kinds, both packs' contents, 3 rerolls, next boss, tags identical — *modulo* slots where B shows something A now owns (A's 8-Ball tarot): the only legitimate in-place difference | FAIL: every slot shifted (single `game.rng`) |
| `test_ownership_blocks_only_that_slot` | seed + slot + key X chosen from `generate.py`'s first shelf. B (no jokers) shows X at that slot (engine = generate.py anchor); A (owns X) shows ≠ X there and identical everywhere else (other slots, voucher, packs, 4 rerolls, modulo X) | FAIL at the anchor: engine shelf ≠ generate.py |
| `test_showman_lifts_the_block` | C owns X + Showman → shelf and 3 rerolls exactly equal B's; `run_state.showman` True | FAIL (anchor) |
| `test_reroll_is_queue_advance[3 seeds]` | after k rerolls the shelf == ground-truth (faithful) `shop_queue[2k:2k+2]` == `generate.reroll_shop` k times, keys+editions, k = 0..4 | FAIL |
| `test_two_players_divergent_play_stay_queue_aligned[2 seeds]` | A: 1 reroll + buy slot 0 at every visit; B: 2 rerolls, buys nothing; 2 antes each. Replay each history through `generate.RunState` (`engine_parity.replay_visits`) → must equal what the engine showed (shelves, packs, voucher, boss, tags). Differences between A and B are thereby fully explained by their own rerolls + blocked slots | FAIL (64/72 unexplained fields each; boss/tags not produced) |
| `test_same_seed_same_actions_identical_state[2 seeds]` | 40-step scripted walk twice → identical signatures at every step | PASS |
| `test_different_seeds_diverge` | guard | PASS |
| `test_clone_then_diverge_leaves_original_untouched` | clone == original; stepping the clone (reroll/buy/leave/play) leaves the original's signature unchanged and the original continues like a fresh twin; clone has its own `run_state`/PseudoRandom | XFAIL (no `run_state`); the first two assertions pass |

### `test_engine_reachability.py` (150 jokers + 52 consumables + 32 vouchers, game keys)

Mechanism: each item has a `Scenario` (hand, held cards, neighbours, consumables used first,
discards first, forced boss, shop actions, repeats, seed retries for probabilistic effects). The
scenario runs WITH and WITHOUT the item on a real `BalatroGame` via `step()`; the two state
signatures (scalar game attrs, hand levels, consumables, deck composition, other jokers' keys/editions/
sell values, hand contents, shop, tags, `run_state` rates) must differ — or a named joker-state key
must change before/after. Two extra teeth: any non-game key in a consumable slot fails (sentinel
tokens), and a used consumable must leave the slot.

**Findings today (46 failures, all engine gaps):**

* **Sentinel producers (10):** `j_8_ball` (`"tarot"`), `j_cartomancer`, `j_certificate` (`"random_enhanced_card"`),
  `j_dna` (`"copy_card:14:Spades"`), `j_marble` (`"stone_card"` — not in the §7 list), `j_riff_raff` (`"common_joker"`),
  `j_seance`, `j_sixth_sense` (`"spectral"`), `j_superposition`, `j_vagabond`. These should create cards through
  `run_state` (`generate.CREATE_SPECS`: `8_ball`/`Tarot8ba<a>`, `riff_raff`/`Joker1rif<a>`, `marble`/`marb_fr`, …).
* **Hand-eval flags set too late (4):** `j_four_fingers`, `j_shortcut`, `j_smeared`, `j_pareidolia` set `ctx` flags in
  `pre_score`, but `evaluate_hand` runs in `_play_hand` *before* `score_hand` → no effect on hand type / face test.
* **Hooks never invoked (9):** `j_flash` (`on_reroll`), `j_hallucination` (`on_booster_opened`), `j_hologram`
  (`on_card_added`), `j_lucky_cat` (`on_lucky_trigger`), `j_matador` (`on_boss_ability_triggered`), `j_perkeo`
  (`on_shop_leave`), `j_astronomer` / `j_chaos` / `j_credit_card` (`on_shop_enter` + flags nothing reads).
* **Flags nothing reads (5):** `j_chicot` (`boss_disabled`), `j_gift` (`pending_shop_buff`), `j_oops` (`double_prob`),
  `j_turtle_bean` (`bonus` never applied to hand size), `j_hiker` (`card.bonus_chips` never scored).
* **Dead `on_init` / wrong mechanic (4):** `j_ancient`, `j_castle` (suit never chosen — `on_init` would `NameError`),
  `j_mime` (retriggers *played* Steel/Gold instead of held cards), `j_campfire` (needs a game-level "card sold" event).
* **Sell-time effects (3):** `j_diet_cola` (needs tags), `j_invisible` (sentinel `"duplicate_joker"`), `j_red_card`
  (needs `BOOSTER_OPEN` + `on_booster_skipped`).
* **Vouchers that are no-ops or need `run_state` (12):** `v_hone`, `v_glow_up` (edition rate), `v_tarot_merchant`,
  `v_tarot_tycoon`, `v_planet_merchant`, `v_planet_tycoon`, `v_magic_trick`, `v_illusion`, `v_omen_globe`,
  `v_telescope`, `v_observatory`, and `v_petroglyph` (only `v_hieroglyph` changes `base_hands`; ante clamp at 1 hides it — probe starts at ante 3).
* Fidelity notes that are NOT failures (probe adapted): joker `pending_money` is only paid out at
  round end / discard (`j_business`, `j_ticket`, `j_reserved_parking`, `j_to_the_moon` probe with
  `post="round_end"`); `j_square` counts *scoring* cards, not played cards; Hanged Man uses
  `remove_card`, so `j_caino` / `j_glass` only see Glass-shatter destruction (`destroy_card`).

**xfail list (= "no measurable change definable" finding):**

| item | reason |
|---|---|
| `j_luchador` | selling a joker during a Boss blind is not an engine action (sell only in `SHOP`); needs sell-anytime + a boss-disable hook |
| `j_ring_master` | needs `run_state.showman` (hook 2) — the only observable of Showman short of a shelf statistic |
| `v_blank` | does nothing by design |
| `v_retcon` | reroll-the-boss at blind select: no `reroll_boss` action in the engine (W2/W6; `generate.reroll_boss` exists) |

Consumables: all 52 pass today under game keys (W1). Note the engine's `c_fool` copies
`tarots_used[-1]` — the probe uses Hermit first; `c_soul`/`c_black_hole` pass (legendary via
`random_joker_key`, all hands levelled) but are not stream-exact until W2.

### `oracle/engine_parity.py`

Drives `BalatroGame.step()` with the scripted policy (win every blind without scoring; open both
packs in display order before rerolling; `--rerolls N` at the last visit of each ante; buy nothing)
and diffs `shelf`/`packs`/`voucher`/`boss`/`tags` per ante against `ground_truth/<SEED>.json`
(`--variant faithful` default) using `parity_check`'s loaders, variant overlay, `item_sig` and
`print_table`. The policy's consumption is replayed into the expectation: k rerolls → queue depth
`2·(visits+k)`; `--reference generate` replays the engine's *own* action history (rerolls, slot
purchases, pack opens, voucher buys) through `generate.RunState` so any policy (`--reroll-every-shop`,
`--buy-shelf`) is comparable; `--buy-vouchers` checks the JSON's `voucher_chain_if_bought` branch.

Today: `0/126 exact through ante 1`; first mismatch on every seed is `ante1.voucher`, then every
shelf slot; `boss`/`tags` "not produced" (hooks 4/5); fallbacks used: `booster_state` (engine never
enters `BOOSTER_OPEN`), `debug_win_blind`. Exit 1. With W2 landed the expected result is
`126/126 exact through ante 8` (the generation layer already is — `parity_check.py --variant faithful`).

## Known limits of the harness itself

* Rerolling before the last visit of an ante (`--reroll-every-shop`) changes which shelf is displayed
  while a pack is opened, so under the faithful `used_jokers` rule pack contents can legitimately
  differ from the JSON corpus — use `--reference generate` for that policy.
* `replay_visits` applies voucher rate effects itself (`_apply_voucher_rates`); once `run_state`
  exists the engine's own bookkeeping is the thing under test, so keep that helper minimal.
* Same-shop duplicate edge (1/150): with Showman, B (no Showman) resamples a slot-1==slot-2
  duplicate and C (Showman) does not; `test_showman_lifts_the_block` would then differ by one slot.
* Reachability signatures deliberately ignore `reroll_cost`/`free_rerolls_remaining` (Chaos is probed
  through a $0 reroll instead) and the probe joker's own state (use `probe_state_key`).
