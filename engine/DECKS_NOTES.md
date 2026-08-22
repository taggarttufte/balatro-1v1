# DECKS_NOTES — Phase 2 W3: decks + stakes (2026-08-21)

**Agent W3.** Files: `balatro_sim/decks.py` (new), `balatro_sim/stakes.py` (new),
`balatro_sim/scoring.py` (Plasma `final_scoring_step`), `balatro_sim/constants.py`
(`get_blind_amount(ante, scaling)` / `blind_base_chips(ante, idx, scaling)` + the 3 stake tables),
`balatro_sim/game.py` (targeted edits, list in §6), `balatro_sim/card_selection.py` (one line:
`plasma=` through the dry-run scorer), `tests/engine_tests/test_decks.py` (106 tests),
`tests/engine_tests/test_stakes.py` (37 + 3 xfail), `tests/engine_tests/test_joker_catalogue.py`
(one assertion: Red deck discards), this note.  Lua sources are line-referenced below; nothing
was copied.

## 0. Gates (run 2026-08-21, W1/W2 editing concurrently)

| gate | result |
|---|---|
| `python -m pytest mp/engine/tests -q` | **1584 passed / 10 skipped / 3 xfailed / 0 failed** (1441 at kickoff; +143 new, 1 assertion updated) |
| `python -m pytest mp/tests -q` | **393 passed / 2 xfailed / 0 failed** (unchanged) |
| `python -m mp.oracle.engine_parity --antes 1-8 --rerolls 5 --quiet` | **126/126 EXACT through ante 8** (Red / White unchanged) |
| Checkered + Plasma vs Red, 12 ground-truth seeds, every shop / boss / tag through ante 4 | **0 mismatches** (script in §4; 3-seed version is `TestGenerationUnchanged`) |

## 1. Per-deck status

`BalatroGame(seed, deck_key="b_red", stake=1)`.  "gen" = already in `generate.DECK_EFFECTS` /
`build_starting_deck` (oracle-verified, W2's file, untouched); "engine" = this workstream.

| deck | status | effect (Lua) | where |
|---|---|---|---|
| `b_red` | **done, verified** | `config {discards = 1}` game.lua:627 → `starting_params.discards = 3 + 1` back.lua:213-215 → **4 discards** | decks.py (`base_discards`) — **the engine had 3; fixed** |
| `b_blue` | done | `{hands = 1}` back.lua:179-181 → 5 hands | decks.py |
| `b_yellow` | done | `{dollars = 10}` back.lua:196-198 → start $14 | decks.py |
| `b_green` | done | `{extra_hand_bonus=2, extra_discard_bonus=1, no_interest}` back.lua:279-287 → `modifiers.money_per_hand/money_per_discard/no_interest`, read at state_events.lua:1166-1173 (hand/discard rows) and :1191 (interest) | decks.py + `game._end_round` |
| `b_black` | done | `{hands=-1, joker_slot=1}` back.lua:179, :263-265 | decks.py |
| `b_magic` | done | `{voucher='v_crystal_ball', consumables={'c_fool','c_fool'}}` back.lua:175-194: voucher → `used_vouchers` + `Card.apply_to_run` (Crystal Ball = +1 consumable slot, card.lua:1922); 2 Fools created with forced keys (`create_card(..., v, 'deck')`, no draw) | gen (used_vouchers, owned_consumables, used_jokers) + decks.py (engine side of Crystal Ball) |
| `b_nebula` | done | `{voucher='v_telescope', consumable_slot=-1}` back.lua:175-179, :273-275 | gen (Telescope read by `generate`) + decks.py (slot) |
| `b_ghost` | done | `{spectral_rate=2, consumables={'c_hex'}}` back.lua:183-194, :206-208 | gen |
| `b_abandoned` | done | `{remove_faces}` back.lua:200-202 → game.lua:2355 skips K/Q/J → 40 cards | gen |
| `b_checkered` | **done, no oracle** | by name, back.lua:239-256: after creation every Clubs → `change_suit('Spades')`, Diamonds → Hearts; `sort_id`s unchanged | gen (suits) + decks.py `creation_order` (**the engine's `sorted(deck)` creation order was wrong for this deck — see §3**) |
| `b_zodiac` | done | `{vouchers={tarot_merchant, planet_merchant, overstock_norm}}` back.lua:232-238 → rates + `shop_joker_max = 3` | gen |
| `b_painted` | done | `{hand_size=2, joker_slot=-1}` back.lua:263-269 | decks.py (`hand_size_mod += 2`, `joker_slots -= 1`) |
| `b_anaglyph` | done | by name, `Back:trigger_effect{context='eval'}` back.lua:111-119: `add_tag(Tag('tag_double'))` when `G.GAME.last_blind.boss`; called from `evaluate_round` state_events.lua:1163 | decks.py `on_round_eval` ← `game._end_round` |
| `b_plasma` | **done, no oracle** | `{ante_scaling=2}` back.lua:270-272 → blind.lua:107 `chips = get_blind_amount(ante) * mult * ante_scaling`; balance by name at `final_scoring_step` back.lua:121-128 ← state_events.lua:946-948 | `game._prepare_next_blind` (×2) + `scoring.score_hand(plasma=True)` |
| `b_erratic` | done | `{randomize_rank_suit}` back.lua:259-261 → game.lua:2342 52 × `pseudorandom_element(G.P_CARDS, 'erratic')` | gen |

Nothing is xfail on the deck side.  Nothing needs a `generate.py` change.

## 2. Plasma arithmetic (state_events.lua:946-948, back.lua:121-128)

After **every** scoring contributor has folded in — scoring cards and retriggers, held-in-hand
(Steel, Baron, …), `joker_main` with joker editions, joker-on-joker — and **before** the
Glass/destroy pass and the product:

```
tot   = hand_chips + mult          -- both may be non-integers (x1.5 editions, Steel)
chips = math.floor(tot / 2)
mult  = math.floor(tot / 2)
score = math.floor(chips * mult)   -- = floor(tot/2)^2, a perfect square
```

`mod_mult`/`mod_chips` are identity functions (misc_functions.lua:684-693).  In the engine this
is `scoring.score_hand(..., plasma=True)`, called with `plasma=self.plasma` from
`game._play_hand` and with `plasma=getattr(gs, "plasma", False)` from
`card_selection.HypotheticalScorer` (so env_v5/v7/sim rank candidate hands under the balance).
Hand-computed cases in `TestPlasmaScoring`: lone Ace 16/1 → 8·8 = **64** (vanilla 16); pair of
Kings 30/2 → **256** (60); pair of 5s with one Polychrome 20/3 → tot 23 → 11·11 = **121** (60);
pair of 5s + Joker 20/6 → 13·13 = **169** (120); + Polychrome Joker 20/9 → 14·14 = **196**
(180); lone Ace with a Steel card held 16/1.5 → tot 17.5 → **64** (24).

Blind targets: `int(blind_base_chips(ante, idx, blind_scaling) * ante_scaling)` then the boss
multiplier (`BOSS_CHIP_MULT`: Wall ×2, Needle ×½, Vessel ×3) — all factors commute, as in
`get_blind_amount * self.mult * ante_scaling`.  Ante 1: 600 / 900 / 1200; ante 9 Small
220 000.  Tested at every ante 0-12 × 3 blinds.

## 3. Checkered creation order (found + fixed)

`game.lua:2330-2378` sorts the 52 protos by the `suit..rank` string of the ORIGINAL proto and
creates the cards in that order (`C2..C9,CA,CJ,CK,CQ,CT,D2,…,S…`); `back.lua:244-256` then swaps
suits in place, `sort_id`s untouched.  `_init_game_vars` rebuilt the creation order with
`sorted(start.deck, key=suit+rank)` — on the post-swap keys that gives `H×26, S×26`, i.e. the
wrong `sort_id` for 39 of the 52 cards, and every `pseudoshuffle` (which sorts by `sort_id`
first) would then deal a different hand.  `decks.creation_order` returns
`S(ex-C)×13, H(ex-D)×13, H×13, S×13` for Checkered and the plain sort for every other deck
(Erratic's protos are sorted the same way after the draws, so the plain sort is right there).
Independent check: with identical `sort_id`s, Checkered's dealt hand must be Red's hand with
C→S, D→H on the same seed — `test_first_hand_is_red_first_hand_with_suits_swapped`.

## 4. Oracle status

* Red / White: `engine_parity` 126/126 through ante 8 (unchanged by this work).
* Checkered / Plasma: no ground-truth corpus (it is Red-only, and `engine_parity` has no
  engine-side deck flag).  Proxy: neither deck changes a generation call, so their shops /
  bosses / tags / vouchers must equal Red's on the same seed under the same actions.  Checked
  for 12 corpus seeds through ante 4 with a no-buy policy (one real hand per blind, then
  `debug_win_blind`): **0 mismatches**; a 3-seed/ante-3 version is in `test_decks.py`.  What
  the proxy cannot check is exactly what Tagg's spot-check covers (§7).
* Not attempted: the Blueprint/TheSoul runners with a deck flag (brief's 20-minute cap; the
  proxy above is stronger for these two decks anyway).

## 5. Stakes

`stakes.py`: 8-entry catalogue (`STAKES[level]`, `STAKE_BY_KEY`), cumulative modifier table
cross-checked against `pools.STAKES[*].cumulative`, `apply_stake_to_game` (engine side).
`BalatroGame(stake=1..8 | 'stake_white'..'stake_gold')` flows into
`RunState.for_stake(seed, stake)` (generation side: `enable_eternals/perishables/rentals_in_shop`).

| level | key | modifier (game.lua:2050-2057) | status |
|---|---|---|---|
| 1 | white | none | **verified no-op**: `BalatroGame(seed)` ≡ `stake=1` ≡ `stake='stake_white'`, identical `state_signature()` trajectory over 150 steps; `run_state` flags all False |
| 2 | red | `no_blind_reward.Small` → blind.lua:84 `blind.dollars = 0` | done (`game._end_round`; Big/Boss still pay; generation identical to White) |
| 3 | green | `scaling = 2` → table 2 (300, 900, 2600, 8000, 20000, 36000, 60000, 100000) | done (`constants.BLIND_AMOUNTS_BY_SCALING`, endless formula shared) |
| 4 | black | `enable_eternals_in_shop` | generation flag only; Eternal effect (cannot sell) **xfail** |
| 5 | blue | `starting_params.discards -= 1` (applied BEFORE the deck's +1: Red deck at Blue stake = 3) | done |
| 6 | purple | `scaling = 3` → table 3 (300, 1000, 3200, 9000, 25000, 60000, 110000, 200000) | done |
| 7 | orange | `enable_perishables_in_shop` | flag only; Perishable effect **xfail** |
| 8 | gold | `enable_rentals_in_shop` | flag only; Rental −$3/round **xfail** |

Sticker flags already ride on `ShopItem`/`BoosterChoice` into `JokerInstance.state`
(`eternal`/`perishable`/`rental`) — the sticker EFFECTS are the later phase.

## 6. game.py edits (targeted, all re-read before editing)

`__init__` (`stake` kwarg; `deck_key` validated through `decks.deck_spec`; `run_state` via
`for_stake`), `_init_game_vars` (`for_stake`; 9 modifier defaults; `creation_order`;
`apply_stake_to_game` + `apply_deck_to_game` after `start_run`), `clone` (10 new scalars),
`_prepare_next_blind` (`blind_scaling`, `ante_scaling`), `_play_hand` (`plasma=` kwarg),
`_end_round` (Small-blind $0 at stake ≥ 2; `money_per_hand`/`money_per_discard`/`no_interest`;
`decks.on_round_eval` after the Investment `eval` pass), two imports.  `state_signature()`
picks up the new scalars automatically.

**For W1 (same `_end_round` block):** the unused-hand row is now
`self.hands_left * (self.money_per_hand or HAND_PAYOUT)` and there is a separate discard row —
the MLB "no unused-hand money at a PvP blind" guard (`game.toml:96-101`) should wrap the hands
row; check whether the mod's patch also covers the `money_per_discard` row (Green Deck at a PvP
blind) before deciding what the discard row does.

## 7. What Tagg should spot-check in the real game (≈5 min)

Seed **`7I4M53DL`**, White stake, fully-unlocked profile, no mods.  The engine's predictions:

**Plasma Deck** (boss The Hook, tags Skip / Economy, shop voucher Wasteful — same as Red):
1. Blind select: Small **600**, Big **900**, Boss **1200** (Red shows 300 / 450 / 600).
2. Play the Small Blind.  First hand (same cards as Red): A♣ K♦ J♠ 7♠ 5♣ 3♦ 2♥ 2♣.
3. Play the **7♠ alone** → after "Balanced" both counters read **6** and the hand scores **36**.
   (Vanilla would be 12.)  Then the **5♣ alone** → **5 × 5 = 25**.  Then the pair **2♥ 2♣** →
   14 chips / 2 mult → **8 × 8 = 64**.
4. Optional: after the boss, the ante-2 Small Blind should read **1600**.

**Checkered Deck** (same seed): blind select must show the same boss / tags / voucher as Red
(The Hook, Skip, Economy, Wasteful), and the first hand must be Red's first hand with
Clubs→Spades, Diamonds→Hearts: **A♠ K♥ J♠ 7♠ 5♠ 3♥ 2♠ 2♥** (Red: A♣ K♦ J♠ 7♠ 5♣ 3♦ 2♥ 2♣).
If the ranks match but some suit pairing is off, `decks.creation_order` is wrong.

**Red Deck**: confirm the blind-select screen shows **4 discards** (3 + the deck's +1) — this is
the engine change that touches every Red/White run.

## 8. Found, not fixed

1. **The Flint is still "halve the final score"** (`game._play_hand`: `score // 2`).  The game
   halves the hand's BASE chips and mult at evaluation (`Blind:modify_hand`, blind.lua:512-515:
   `mult = max(floor(mult*0.5+0.5), 1)`, `chips = max(floor(chips*0.5+0.5), 0)`) before any
   card or joker scores.  Pre-existing approximation; under Plasma the two differ more (the
   balance is applied to the already-halved totals in the game).  Owner: whoever next touches
   the boss effects.
2. **Checkered ante-1 `idol`/`castle` picks:** `generate.start_run` draws them from the post-swap
   deck, but in the game `reset_idol_card()` runs synchronously in `start_run` while the
   Checkered `change_suit` is a queued event — the game's ante-1 pick can name a Club/Diamond.
   Unobservable (no Idol/Castle can be owned before the first shop; picks are redrawn at every
   round end), documented only.
3. `apply_voucher` is bypassed for deck vouchers (they are pre-registered in `game.vouchers` by
   `_init_game_vars`); `decks.apply_deck_to_game` applies the one engine-side effect among them
   (Crystal Ball +1 slot).  If a future deck/challenge grants a voucher with another engine-side
   effect (Grabber, Antimatter, …) it must be added there.
4. Anaglyph fires on any `_end_round` with `last_blind.boss` — under W1's MLB "failed blind
   proceeds" path that includes a LOST Nemesis blind; that matches the mod (it runs vanilla
   `evaluate_round` with `blind.chips = -1`), so no change intended, but W1 should be aware.
5. Stake ≥ 4 sticker EFFECTS (Eternal / Perishable / Rental) — xfail, later phase.
