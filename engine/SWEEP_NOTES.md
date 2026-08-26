# SWEEP_NOTES — Phase 1 W5 bug sweep (P1-sweep), 2026-08-21

The last step of Phase 1: clear the reachability probe, close the W2/W3 "for W5" lists,
dispose of every `docs/MP_UPDATE_LIST_2026-08.md` §7 item, protect each fix with a test.
Nothing outside `engine/` was edited except two probe fixes in
`tests/test_engine_reachability.py` (allowed by the brief).  `rng/*` and
`oracle/ground_truth/*` untouched.  Nothing committed.

## 0. Gate numbers (final clean run, lead should re-run)

| gate | before W5 | after W5 |
|---|---|---|
| `python -m pytest engine/tests -q` | 1358 passed / 10 skipped / 0 failed | **1441 passed / 10 skipped / 0 failed** (+43 `test_hand_eval_flags.py`, +36 `test_sweep.py`, +4 from existing parametrised tests) |
| `python -m pytest tests/test_engine_invariants.py -q` | 14/14 | **14/14** |
| `python -m oracle.engine_parity --antes 1-8 --rerolls 5` | 126/126 exact through ante 8 | **126/126 exact through ante 8** (re-run after every non-trivial change; never dropped) |
| `python -m pytest tests/test_engine_reachability.py -q` | 226 passed / 6 failed / 3 xfailed | **233 passed / 0 failed / 2 xfailed** (`j_luchador`, `v_blank` — both "no measurable change definable") |
| `python -m pytest tests -q` | 386 / 6 / 3 | **393 passed / 0 failed / 2 xfailed** |
| `engine_parity --probe` | 11/12 (`state_signature` missing) | **12/12** |

Non-gate policies (for the record): `--reference generate --reroll-every-shop --buy-shelf`
117/126 through ante 3, 106/126 through ante 8 (unchanged from W2 — every mismatch is a
joker effect the generate-only replay cannot model); `--buy-vouchers` is inherently
non-exact (it compares against the no-buy JSON) — see §B8 for what changed there.

## 1. Disposition — worklist A/B/C

| # | item | disposition | where |
|---|---|---|---|
| A1 | `j_four_fingers` / `j_shortcut` / `j_smeared` (+`j_pareidolia`) flags set AFTER `evaluate_hand` | **FIXED.** `hand_eval.py` rewritten as a port of `evaluate_poker_hand` + `get_X_same` / `get_flush` / `get_straight` / `get_highest` (misc_functions.lua:376-620) with `evaluate_hand(cards, *, four_fingers, shortcut, smeared, pareidolia)`. Four Fingers: 4-card floor, the off card does NOT score, Straight Flush = union of the flush subset and the straight subset (2H 3H 4H 5S KH scores all 5), Flush Five / Flush House with 4 suited. Shortcut: the Lua j=1..14 walk (Ace low+high, ONE skip between present ranks, never at j=14, no wrap). Smeared: `is_suit(..., flush_calc)` pairing. Wild = any suit unless debuffed. `pareidolia` accepted, no-op (face-ness never affects the hand TYPE in the Lua either). `base.hand_eval_flags(jokers)` + `BalatroGame.hand_eval_flags()` compute the flags from the ACTIVE board (Crimson Heart's disabled joker is not `find_joker`'d) and `_play_hand` passes them BEFORE scoring; the Crimson Heart roll moved ahead of evaluation. The three envs' dry-run subset evaluators (`env_v7._best_hand_score`, `env_sim` / `env_v5 _update_play_combos`) and `card_selection.validate_play_subset` / `get_valid_play_mask` take the same flags so the reward estimate evaluates what the play path will. | `hand_eval.py`, `jokers/base.py`, `game.py::_play_hand`, `card_selection.py`, `env_v7.py`, `env_sim.py`, `env_v5.py`; tests `sim_tests/test_hand_eval_flags.py` (43: A-2-3-4 wheel w/ Four Fingers, T-J-Q-K-A shortcut irrelevance, wheel+gap, no wrap-around, Stone in the size floor, Wild ties, Flush Five/House with 4 suited, play-path + Crimson Heart) |
| A2 | `j_flower_pot` probe | **PROBE FIXED** (joker correct per card.lua:3807-3840 — reads the scoring hand with the Wild fill rule). The probe played a High Card (only the Jack scores); now plays the 4-suit STRAIGHT. | `tests/test_engine_reachability.py` |
| A3 | `j_space` probe | **PROBE FIXED** — `repeats=2` (first `space` draw ≥ 0.25 on all 7 harness seeds; ALEEB fires on the 2nd, 0.208). Verified the level-up lands in `planet_levels` (the probe signature includes it) and on THIS hand (`pre_score`). Not an engine bug. | same |
| A4 | `j_chicot` / boss debuffs at play time | **FIXED.** `game._boss_debuffs_card(card)` = `Blind:debuff_card` (blind.lua:624-648) as a predicate: suit bosses via `is_suit(suit, bypass_debuff)` (**Wild cards are debuffed by every suit boss; Smeared extends the suit; Stone never**), The Plant via `is_face(true)` (**Pareidolia ⇒ every card debuffed**), The Pillar via `played_this_ante`, Verdant Leaf until a Joker is sold, nothing on non-boss / disabled blinds. Applied over `full_deck` at blind start, on every draw, **re-evaluated over the hand at play time** and after a consumable use (the game re-runs it on `set_base`/`set_ability`). `BlindInfo.disabled` is consulted by the predicate, by `_stay_flipped`, and every other boss effect keys off `boss_key`, which Chicot blanks. | `game.py`; `test_sweep.py::TestPlayTimeDebuffs` (7) |
| B5 | `bl_fish` wrong `_draw_to_full` model; `bl_house` / `bl_wheel` / `bl_mark` no-ops | **FIXED — all four modelled.** `Card.face_down` (copied by `Card.copy`). `game._stay_flipped(card, after_play)` = `Blind:stay_flipped` (blind.lua:605-622): Wheel `pseudorandom('wheel') < normal/7` per drawn card, House = initial deal (`hands_played == 0 and discards_used == 0`, new `_discards_used_round`), Mark = `is_face(true)` (Pareidolia counts), Fish = the redraw after a PLAYED hand (`prepped`), nothing when disabled (Chicot). Revealed when played / discarded (`G.play:emplace` flips), cleared at every blind start. **Obs layer:** `env_v7._encode_obs` (and env_sim) set only the "present" bit for a face-down card and exclude it from the other cards' suit/rank/connectivity stats; env_mp inherits. `UNMODELLED_BOSS_BLINDS` is now `[]` (kept for callers); env_v7's boss one-hot covers all **28** bosses (the four were drawable since W2 but encoded as all-zeros = "no boss") → `OBS_DIM` 443 → **447**, MP 447 → **451** (tests assert the composition, not the literal; `N_BOSS_TYPES == 28` pinned). The old Fish hand-size penalty is gone. | `card.py`, `game.py`, `env_v7.py`, `env_sim.py`; `test_sweep.py::TestFaceDownBosses` (9) |
| B6 | Invisible Joker / Perkeo copies | **FIXED.** Invisible: element draw over the other jokers in **sort_id order** (was board order); `copy_card(..., strip_edition = negative)` → a Negative original is copied WITHOUT its edition (other editions kept); room check `#G.jokers.cards <= card_limit` with the Invisible still on the board (the copy may take the slot it vacates — test); the copy goes through `base.add_joker` → `run_state.acquire` so `used_jokers` is faithful. Perkeo: queues `"negative:<key>"` → `game.add_negative_consumable` — **no slot needed**, `consumable_slots += 1`, tracked in the new `game.negative_consumables` key-multiset so the slot drops again when that card is used (`_consumable_removed`); the same path now serves Negative consumables from shelves and packs (before, their slot bump leaked forever). `base.add_joker` honours Negative (no slot; +1). | `jokers/misc.py`, `game.py`, `shop.py`, `jokers/base.py`; `test_sweep.py::TestCopies` (6) |
| B7 | Ouija / Ectoplasm permanent hand size | **FIXED.** `game.hand_size_mod` (permanent `G.hand:change_size` delta, cloned); `_start_blind` starts from `HAND_SIZE + hand_size_mod`; Ouija / Ectoplasm decrement it AND the live `hand_size`; stacks. | `game.py`, `consumables.py`; `test_sweep.py::TestPermanentHandSize` (3) |
| B8 | Hieroglyph at ante 1 clamps | **FIXED.** `constants.BLIND_CHIPS[0] = (100, 150, 200)` + `get_blind_amount(ante)` (the Lua formula incl. `ante < 1 → 100` and the endless branch) + `blind_base_chips(ante, idx)`; `_prepare_next_blind` uses it; the clamp is gone (`consumables._ease_ante`). **Also found:** `ease_ante` does NOT redraw the boss (`round_resets.blind_choices.Boss` was drawn at Cash Out) but the engine's `_prepare_next_blind` re-rolled `'boss'` whenever `_boss_blind_ante != ante` — so every Hieroglyph/Petroglyph consumed an extra `boss` draw and changed the boss. `_ease_ante` pins `_boss_blind_ante`. Under `--buy-vouchers` the boss-mismatch rows vs the analyzers' chain drop 228 → 99 (the rest are the expected ante-label shift); voucher-chain rows 228 (unchanged — the analyzers ignore the ante change, DELEGATE_NOTES §0). | `constants.py`, `game.py`, `consumables.py`; `test_sweep.py::TestAnteZero` (3) |
| B9 | `JokerInstance.clone()` copies `sort_id` | **FIXED** (+ `BalatroGame.clone` inherits). | `jokers/base.py`; `TestCopies::test_clone_copies_sort_id` |
| B10a | `env_v5._step_pack_open` `BoosterChoice` | **FIXED.** A `BoosterChoice` pick leaves the pack list and goes through `game._grant_choice` (acquire, Negative bookkeeping, playing cards to the deck); a targeted Tarot keeps the PACK_TARGET substate (used straight from the pack); no-slot consumables are applied immediately as before; `_exit_pack_substate` `release_pack`s the unpicked cards (the env had taken the choices off the game, so `used_jokers` leaked). | `env_v5.py`; `TestEnvRevival::test_env_v5_pack_pick_acquires_and_releases` |
| B10b | `env_mp._revive_boss_if_needed` regenerates outside the round transition | **FIXED.** Revival = `state = ROUND_EVAL; step(advance)` → the normal `_end_round` (ante transition, `Voucher<a>` / `Tag<a>` / `boss` / `cashout<a>`, shop). The old path built a shop on a GAME_OVER game and the real `_end_round` then built another — the queue was consumed twice. Test: a player who fails the PvP blind gets exactly the ground truth's ante-2 shelf / voucher / boss. **Policy note for MLB:** the revived loser now receives the boss blind payout + interest like a winner (the engine has no "lost round" Cash Out); if MLB pays differently, adjust in `_end_round` via a flag. | `env_mp.py`; `TestEnvRevival::test_revived_player_next_shelf_matches_ground_truth` |
| B11 | `state_signature()` | **ADDED** on `BalatroGame`: hashable tuple of every scalar, the blind, hand levels, jokers (key/edition/state), consumables + Negative multiset, vouchers, deck composition + partitions (Card-id-free), Pillar/Bell cards by identity, shop, open pack, tags, round picks, `run_state` view, and a blake2b digest of the keyed PseudoRandom's full snapshot (process-independent). `--probe` 12/12; the harness uses it. | `game.py`; `test_sweep.py::TestStateSignature` (2) |
| C | §7 list | see §2 | |
| C | `j_marble` sentinel (W8) | **VERIFIED W3's fix**: real Stone card via `marb_fr`, and (W5) it now enters the deck BEFORE the `nr<ante>` shuffle — see §3. | `TestBlindStartOrder::test_marble_stone_is_in_the_first_shuffle` |
| D | stale `v_retcon` xfail ("no reroll_boss action") | **PROBE FIXED** (W2 added the action); `test_sweep.py::TestRetcon` pins unlimited $10 rerolls that change the boss. | |

## 2. Disposition — `docs/MP_UPDATE_LIST_2026-08.md` §7 (every item)

| §7 item | disposition | by |
|---|---|---|
| 0. Booster state machine never enters `BOOSTER_OPEN` | fixed | W2 (`test_delegate`) |
| 8 Ball wrong effect (import order) | fixed — creates a Tarot via `8ball` + `Tarot8ba<a>` | W1 (pick) + W3 (keys) |
| Gros Michel / Cavendish destruction never happens | fixed — `drain_joker_state`, `gros_michel_extinct` pool flag | W3 |
| The Idol initial target never randomized | fixed — `idol<a>` at run start + every round (`round_picks`) | W2 + W3 |
| Lucky Cat `on_lucky_trigger` has no callers | fixed — per-pass `ctx.lucky_trigger` | W3 |
| Nine jokers produce sentinel strings (Vagabond, Superposition, Cartomancer, Seance, Sixth Sense, Riff-Raff, Certificate, DNA, Perkeo) | fixed — all create real keys through `generate`; Perkeo's Negative copy W5; `_materialize` no longer parks unknown tokens in a slot (W5) | W3 + W5 |
| Six hooks defined, never invoked (`on_booster_opened`, `on_shop_enter`, `on_shop_leave`, `on_lucky_trigger`, `on_card_added`, `on_boss_ability_triggered`, `on_reroll`) | fixed — all have callers (grep-verified: `game.py` / `shop.py` / `scoring.py`); `on_first_hand_drawn` added (W5) | W2 + W3 |
| Six dead `on_init` (Castle, Ancient, To Do List `NameError`) | fixed — Castle/Ancient read the game-level round cards; To Do List draws `to_do` on init; remaining `on_init`s (Popcorn, Ramen, Ice Cream) are live | W3 |
| Boss blinds have no exhaustion pool | fixed — `generate.next_boss` (min-usage filter, `boss` stream), parity-verified | W2 |
| Vouchers have no ante gating | fixed — `generate` draws from `Voucher<a>` with `VOUCHER_REQUIRES` | W1 + W2 |
| Hone / Glow Up purchasable no-ops | fixed — `run_state.edition_rate` (`SHOP_RATE_BY_VOUCHER`) | W2 |
| Booster packs allow duplicates | fixed — `generate.open_pack` resample semantics | W2 |
| Standard packs produce vanilla cards | fixed — `stdset` / `standard_edition` / `stdseal` / `stdsealtype` | W2 |
| Editions only roll for shop jokers | fixed — pack jokers, Judgement / Wraith / Soul / Riff-raff / Invisible-copy / tag jokers carry `poll_edition` results (`grant_created`, `_joker_from_gen`, `BoosterChoice.from_gen`); Negative created jokers take no slot (W5 `add_joker`) | W2 + W3 + W5 |
| Glass retrigger rolls once per card not per trigger | **not a bug** — the Lua rolls once per scoring Glass card after all retriggers (state_events.lua:951-963); pinned by `test_glass_rolls_once_per_scoring_card_not_per_trigger` | W3 |
| Wheel of Fortune targets editioned jokers | fixed — editionless only, unusable otherwise (`generate.wheel_of_fortune`) | W3 |
| Rarity distribution wrong (123/18/10) | fixed — `pools.py` is the single source (61/64/20) | W1 |
| Six jokers double-listed at two rarities | fixed — registry raises on duplicates, 150/150 | W1 |
| `j_marble` `"stone_card"` sentinel (W8 addendum) | fixed — `marb_fr` real Stone card; in the first shuffle (W5) | W3 + W5 |

Nothing in §7 is still open.

### REKEY_NOTES §6 (W1 "found, not fixed") — all closed

Photograph (W3), Ancient suit (W3), Flower Pot (W3 + probe W5), Swashbuckler (W3), Stencil
(W3), Flash Card `on_reroll` (W3), Hallucination / Matador hooks (W3), Stuntman (W3), Showman
→ `run_state.showman` (W2), `bl_fish` (W5), Director's Cut → `reroll_boss` (W2),
`random_joker_key` (W2, deleted).

### EFFECTS_NOTES "deferred to W5" — status

Hand-eval flags (done), boss debuffs at play time (done), Perkeo's Negative copy (done),
The Wheel / flipped cards (done). Still open — see §4: Crimson Heart per-play vs per-draw,
Gift Card on consumables, Luchador sell-during-blind, Baseball interleaving.

## 3. Other changes made on the way (each tested)

* **Blind-start order** now follows `new_round` / `DRAW_TO_HAND` (state_events.lua:290-353,
  game.lua:3207-3222): resets → Chicot disable → boss start effects + debuffs →
  `setting_blind` joker hooks (Riff-raff, **Marble's Stone INTO THE DECK**, Cartomancer,
  Madness, Burglar, Dagger) → **`nr<ante>` shuffle** → Juggle → draw → `first_hand_drawn`
  (Certificate's sealed card joins a full hand: 9 cards). Before W5 the engine dealt first
  and fired the hooks afterwards, so Marble's card sat at the bottom of an already-shuffled
  deck and Certificate's card displaced a drawn card. `_Certificate` moved from
  `on_blind_selected` to the new `on_first_hand_drawn` hook (test hook lists updated).
* `_prepare_next_blind` chips via `blind_base_chips` (supports ante ≤ 0 and > 8).
* `add_joker` (pending-joker path: Riff-raff, Invisible) — Negative takes no slot.
* `env_v7.BOSS_TYPES` = all 28 game bosses.

## 4. Remaining known gaps (honest)

1. **Crimson Heart** rolls `crimson_heart` once per PLAY; the game rolls on every
   draw-to-hand (`drawn_to_hand`), so the draw count differs (stream is isolated — no
   parity impact; which joker is disabled on a given hand can differ from the game).
2. **Luchador**: selling a joker during a Boss blind is not an engine action (sell only in
   `SHOP`) — `j_luchador` stays xfail. Needs a sell-anytime action + `Blind:disable` mid-blind
   (the `disabled` flag and the un-debuff / un-flip logic exist: `_refresh_card_debuffs`).
3. **Gift Card** cannot add sell value to consumables; **Negative consumables** are tracked
   per key (a multiset), not per card — if two copies of one key are held, one Negative, the
   first use releases the Negative slot regardless of which card "was" used. Consumables are
   bare keys everywhere (`consumable_hand: list[str]`); a `ConsumableInstance` would close both.
4. **Baseball Card** (joker-on-joker) applies in one step rather than interleaved per joker.
5. **The Order switch** (`queue_scope="run"`) is still a no-op — needs the `generate.Keys`
   suffix hook (DELEGATE_NOTES §3; `generate.py` is not engine-owned). Phase 2.
6. **Showman** flag is "owned", not "owned and not debuffed" (Verdant Leaf / Crimson Heart).
7. **MLB revival payout** (B10b): the PvP loser is paid like a winner; the MP ruleset's
   exact Cash Out for a lost PvP blind is not in `_reference`.
8. **Face-down cards and the reward**: the obs hides them, but `env_v7._best_hand_score`
   (a reward term, not an observation) still sees their identity. Acceptable for a reward;
   flag if it is ever used as a policy input.
9. **Voucher reachability probes** all pass through `run_state.used_vouchers` in the
   signature (harness-side; W8's design) — they prove the voucher is *redeemed*, not that
   each effect is observable. The effects themselves are covered by `test_delegate` /
   `test_consumables` / `test_sweep`.
10. `--buy-vouchers` voucher-chain rows (228 over 50 seeds) are the analyzers' known
    omission of `ease_ante` (DELEGATE_NOTES §0) and now include the ante-1 Hieroglyph seeds
    (the clamp that hid them is gone).

## 5. Files touched

`balatro_sim/`: `hand_eval.py` (rewritten), `card.py`, `constants.py`, `game.py`,
`consumables.py`, `shop.py`, `card_selection.py`, `env_v7.py`, `env_sim.py`, `env_v5.py`,
`env_mp.py`, `jokers/base.py`, `jokers/misc.py`.
`tests/`: NEW `sim_tests/test_hand_eval_flags.py` (43), NEW `sim_tests/test_sweep.py` (36);
edited `engine_tests/test_env_v7.py` (N_BOSS_TYPES 28), `engine_tests/test_jokers.py` and
`sim_tests/test_game_keys.py` (hook list + `UNMODELLED == []`).
`tests/test_engine_reachability.py`: `j_space` (`repeats=2`), `j_flower_pot` (STRAIGHT),
`v_retcon` (xfail lifted).
