# FIX_NOTES — engine correctness round (W-FIX, 2026-08-26)

Four defects, three in `engine/balatro_sim/` and one in `ev/`. All four came out of
**W-ENCODE-POC** (`ev/encode/POC_NOTES.md`), and two of them were found by *measurement*
rather than by re-reading the Lua — which is the reason this round exists.

Nothing here touches generation. `python -m oracle.engine_parity --antes 1-8 --rerolls 5`
is **126/126 exact** before and after.

Reproduce:

```
python -m pytest engine/tests -q            # 1715 passed, 10 skipped, 3 xfailed
python -m pytest tests -q                   # 1073 passed, 2 xfailed
python -m pytest ev -q                      # 341 passed
python -m pytest ev/encode -q               # 63 passed
python -m oracle.engine_parity --antes 1-8 --rerolls 5      # 126/126
python ev/encode/run_poc.py --workers 4 --traj-seeds 72 --worlds 40
python ev/gate_ev_player.py --procs 12
```

New tests: `engine/tests/engine_tests/test_joker_state_fidelity.py` (37), plus 5 in
`ev/tests/test_player.py` and 2 in `ev/tests/test_hand.py` (+1 engine hook-list line in
each of `test_jokers.py` / `test_game_keys.py`). Three `ev/encode` gap-pins were flipped
from "the engine is wrong, here is what it does" to "the engine is right"; the fourth
(Cloud 9 / Stone) is still a gap and still pinned as one.

---

## 1. Blueprint / Brainstorm re-ran the copied joker's self-mutation

**Lua.** A copier re-dispatches the target's own effect with
`context.blueprint = (context.blueprint and (context.blueprint + 1)) or 1` — a depth
counter, not a bool (`card.lua:2310-2312` Blueprint, `:2324-2326` Brainstorm). **26 joker
branches** then read `and not context.blueprint`, so a copy reproduces the *score
contribution* and never the *state change*. In `card.lua`, by grep:

> Caino, Campfire, Ceremonial Dagger, Chicot, Constellation, DNA, Flash Card, Fortune
> Teller, Glass Joker, Green Joker, Ice Cream, Invisible Joker, Madness, Midas Mask,
> Obelisk, Popcorn, Ramen, Red Card, Ride the Bus, Seltzer, Throwback, To Do List,
> Trading Card, Turtle Bean, Vampire, Yorick

plus four whose name test sits on an earlier line: Castle (`:2816`), Hit the Road
(`:2837`), Lucky Cat (`:3076`), Wee Joker (`:3084`), Spare Trousers (`:3412`), Square
Joker (`:3427`), Runner (`:3435`), and the `elseif not context.blueprint` arm at `:2888`
(Campfire's boss reset, Rocket's boss upgrade).

**Engine, before.** `_Blueprint` / `_Brainstorm` called the target's hook on the target's
own instance with no flag (`jokers/misc.py::_guarded_call`), and the engine folded "apply
the chips" and "decay" into that one hook. So the copy mutated the target a second time:
**Ice Cream melted twice as fast; Green Joker and Ride the Bus scaled twice as fast.**

Found by measurement, not by reading. The Lua was read correctly and the POC entry was
written correctly; 1 seed in 40 of the Ice Cream trajectory run came back at 10 chips
instead of 40, and tracing seed `7YTVQERM` showed the decay rate doubling from 5/hand to
10/hand on the exact hand a Brainstorm entered the board.

**Fix.**

* `ScoreContext.blueprint: int = 0`, set and restored (in a `finally`) by
  `_guarded_call` — the one place a copied effect is dispatched.
* Every joker whose Lua branch carries the guard and whose engine hook mutates its own
  state now reads it. Fourteen jokers: Green Joker, Ride the Bus, Obelisk, Ice Cream,
  Seltzer, Spare Trousers, Runner, Square Joker, Vampire, Wee Joker, Lucky Cat, Card
  Sharp, Burnt Joker, Loyalty Card — plus Midas Mask, which mutates the *cards* rather
  than itself (idempotent today; guarded because the Lua's list is the contract).
* **The engine's phase structure was already the Lua's, and is now used.** `card.lua`
  raises these counters in `context.before` (state_events.lua:630) and pays them in
  `joker_main` (:876); the engine's `pre_score` and `on_hand_scored` are exactly those two
  passes. The gain moved into `pre_score` for the six `context.before` jokers (Green
  Joker :3563, Ride the Bus :3525, Obelisk :3543, Spare Trousers :3412, Runner :3435,
  Square Joker :3427) and for Loyalty Card. Without a copier this changes nothing; with
  one it is the difference between right and off-by-one:

  | | mult rows on a High Card Ace, Green Joker at 6, Blueprint to its left | score |
  |---|---|---|
  | Lua, and this engine now | 1 + 7 + 7 | **240** |
  | before this fix | 1 + 7 + 8 | 256 |
  | a guard applied in place, no phase split | 1 + 6 + 7 | 224 |

* A new `after` pass (`scoring.py::_after_phase`, state_events.lua:1070) holds the two
  self-consuming jokers. `card.lua` has exactly one `elseif context.after then` block
  (`:3570`) and it contains Ice Cream's melt and Seltzer's countdown, both guarded — so
  the pass is fired directly on the owned jokers and needs no copy dispatch.

**Deliberate non-fixes, verified against the Lua:**

* **Hiker** (`card.lua:3067`) has *no* guard — a copy really does add another +5 per
  scoring card, and the engine must keep doing it. Pinned by
  `test_hiker_is_deliberately_not_guarded`.
* The retrigger jokers (Mime, Sock and Buskin, Hanging Chad, Dusk) are `context.repetition`
  and unguarded; `scoring._n_mime_reps` already models Blueprint-copied Mime.
* Jokers whose state change lives in a hook the engine's copier never dispatches
  (`on_discard`, `on_round_end`, `on_card_added`, `on_planet_used`, `on_card_sold`,
  `on_reroll`, `on_sell`, …) could not double-fire and were left alone: Castle, Hit the
  Road, Ramen, Campfire, Constellation, Fortune Teller, Glass Joker, Hologram, Popcorn,
  Flash Card, Red Card, Throwback, Yorick, Trading Card, Turtle Bean, To Do List, Invisible
  Joker, Ceremonial Dagger, DNA, Madness, Chicot, Caino.
* `_MrBones`, `_ToTheMoon`, `_DelayedGrat` write their own state from a copied hook but the
  write is *idempotent* (a flag, or a cache of a `ctx` value), so a copy cannot change the
  outcome. Left unguarded and recorded here rather than guarded silently.

`test_every_hook_a_copy_dispatches_is_guarded_where_the_lua_guards_it` is the structural
check that catches the next joker somebody adds without the guard: it calls every
dispatched hook of every guarded joker with `ctx.blueprint` set and asserts the state is
untouched. Verified to bite (removing Wee Joker's guard fails it).

## 2. Satellite read a per-instance planet set instead of run-global usage

**Lua** (`card.lua:1667-1673`) counts distinct `G.GAME.consumeable_usage` entries with
`set == 'Planet'`. That table is run-global, written by `set_consumeable_usage`
(`functions/misc_functions.lua:1184-1195`) from `Card:use_consumeable` (`card.lua:1093`)
for every consumable the run has ever used — so a Satellite bought at ante 4 after five
distinct planets pays **$5 on its very first round end**.

**Engine, before.** The set lived on the joker instance and was seeded only by an
`on_planet_used` sweep, so Satellite paid **$0** until *new* planets were used. Satellite
is a $6 rarity-2 pure-economy joker that is essentially always acquired mid-run, which is
exactly the case the engine got wrong — 100% of its value, not a fraction.

**Fix.** `game.planets_used` already recorded every planet use, and
`consumables.apply_planet` is the single entry point every path funnels through
(`game._use_consumable`, the `env_v5` booster paths, `ev/player`'s clone). It is now
carried on `ScoreContext.planets_used` (filled by `game._hook_ctx`, by reference) and
`_Satellite.on_round_end` counts its distinct entries. `on_planet_used` is gone from
Satellite.

Members and non-members match the Lua: **Black Hole** is a Spectral (`set ~= 'Planet'`) and
**Space Joker / Burnt Joker / the Orbital tag** call `level_up_hand` rather than
`use_consumeable`, so none of them pays Satellite —
`test_satellite_ignores_level_ups_that_are_not_planet_uses`. `game.planets_used` is copied
by `BalatroGame.clone` (game.py:598) and is already in `state_signature` (:1009), so
clone / determinize carry it; pinned by two tests.

## 3. Ice Cream never melted

**Lua** (`card.lua:3571-3592`): `if extra.chips - chip_mod <= 0 then …
G.jokers:remove_card(self); self:remove()`. Ice Cream is **destroyed** on the hand that
would take it to zero — `extra.chips` is left at its last positive value and the card stops
existing. 100 chips at 5/hand is therefore **20 scoring hands** (100, 95, …, 5 = 1050 chips
in total) and then a **freed joker slot**.

**Engine, before.** `max(0, chips - 5)` — a dead 0-chip joker squatting in a slot forever.
Scoring impact nil; *policy* impact not, because a permanently-occupied slot changes every
later buy decision.

**Fix.** The decay moved into the new `after` pass and the melt sets `state['destroyed']`,
which `base.drain_joker_state` already turns into `remove_joker` (the same mechanism
Seltzer uses). `joker_slots` is untouched, so the slot is genuinely freed and refillable —
`test_the_slot_ice_cream_frees_can_be_filled`.

## 4. `ev/hand.py::_RATIO_CACHE` leaked between runs in a shared process

Not an engine defect — a player-layer one, and the second finding that came out of
measurement rather than reading (POC_NOTES §3.5).

`board_ratio`'s memo was a process-global dict keyed by `_board_sig`, and that key
**deliberately** omits planet levels and the exact deck composition (EV_NOTES §8b item 1:
"a planet pick must not force a ratio recompute — it was 40% of a pack decision"). But
`board_ratio` samples real hands from the real deck at the run's real planet levels, so two
states that differ only in an omitted field share an entry and whichever was computed first
wins. Inside one run that is a documented approximation and is deterministic given the
seed. **Across runs sharing a worker it was a determinism leak**: the POC measured 2 of 24
seeds (8%) changing trajectory with the worker partition, and every per-seed row of a
pooled gate was partition-dependent by the same mechanism.

**Fix: the scope, not the key.** `board_ratio(..., cache=...)` memoises into a
caller-supplied dict; `EVPlayer` owns `self._ratio_cache`, clears it in `reset()`, and
passes it at all eight `build_proxy` call sites. `hand._RATIO_CACHE` survives as the
fallback for module-level callers (tests, `estimate_clear_probability`) and no player
writes it — `test_nothing_a_player_computes_reaches_the_module_cache`.

Widening the key was the alternative and would have paid back the 40% pack cost. It was
also unnecessary, because the cross-run sharing was buying almost nothing. Measured over
12 seeds of `ev:fast`, 3516 memo lookups:

| memo scope | hit rate | shop+pack mean | wall |
|---|---|---|---|
| per player (this fix) | 2678 / 3516 = **76.2%** | 5.16 ms | 5.7 s |
| shared across runs (before) | 2679 / 3516 = 76.2% | 5.61 ms | 6.2 s |
| no cache at all | 0% | 13.66 ms | 11.1 s |

**The leak was worth one cache hit in 3516 lookups.** All of the cache's value is
within-run reuse — shop candidates inside one visit — which the per-player dict keeps.

Per-decision cost, 12 seeds single process (EV_NOTES §8b reports shop 3.9 ms / pack
6.1 ms):

| | before | after | |
|---|---|---|---|
| shop | 4.41 ms | 4.56 / 4.58 ms | +3.4% |
| pack | 6.63 ms | 6.76 / 6.87 ms | +2.6% |
| hand | 3.33 ms | 3.38 / 3.43 ms | +2.1% |

`EV_NOTES.md` §8b item 1 is amended in place.

---

## What moved, and what did not

**Engine parity: unchanged.** 126/126 exact through ante 8, every field. Generation is
untouched by all four fixes.

**126-seed EV gate** (`python ev/gate_ev_player.py --procs 12`):

| | fast before → after | full before → after | greedy |
|---|---|---|---|
| ante-1 clear | 95.2% → **95.2%** | 95.2% → **96.0%** | 31.7% (unchanged) |
| ante-2 / 3 / 4 clear | 81.7 / 77.0 / 60.3% → same | 86.5 / 79.4 / 67.5% → 87.3 / 80.2 / 68.3% | unchanged |
| won | 3.2% → 3.2% | 3.2% → 3.2% | 0% |
| mean final ante | 4.78 → **4.78** | 4.95 → **4.98** | 1.32 |
| mean blinds cleared | 9.746 → 9.738 | 10.230 → 10.310 | 2.159 |
| $ at ante 3 | 20.52 → 20.41 | 20.28 → 20.29 | — |
| **per-seed rows changed** | **1 / 126** | **6 / 126** | **0 / 126** |
| draw-order invariance | 743/743 | 741/741 | — |

Every movement is inside the bootstrap CI, and `greedy` is bit-identical (it buys nothing,
so none of the four fixes can reach it) — which is a useful negative control on the blast
radius. Ante-1 stays at 95.2%.

**Per-seed reproducibility is now partition-independent, without
`verify.reset_player_caches()`:**

* POC trajectory measurement, 24 seeds × 3 jokers, `reset_player_caches` monkeypatched to a
  no-op: **workers 1 == 4 == 6**, every per-seed row identical.
* `gate_ev_player.run_seed` through a real reused spawn pool, 12 seeds: **procs 1 vs 6,
  12/12 identical** on (ante, blind, $, blinds cleared, won, steps).

**POC harness** (`python ev/encode/run_poc.py --workers 4 --traj-seeds 72 --worlds 40`),
real entries **8/10 → 9/10 accepted**, controls still 2/2 correctly rejected:

| item | mode | before | after |
|---|---|---|---|
| Cloud 9 | round_end | 3.000 / 3.000 ACCEPT | 3.000 / 3.000 ACCEPT |
| Cloud 9 | rollout | 21.500 / 21.475 ± 0.111 ACCEPT | 21.500 / 21.475 ± 0.111 ACCEPT |
| Rocket | round_end | 3.000 / 3.000 ACCEPT | 3.000 / 3.000 ACCEPT |
| **Satellite** | round_end | 1.500 / **0.750 ± 0.807 REJECT** (inexact) | 1.500 / **1.500 ± 0.000 ACCEPT** (exact) |
| Ride the Bus | trajectory | 1.461 / 1.652 ± 0.766 ACCEPT | 1.552 / 1.696 ± 0.754 ACCEPT |
| Green Joker | trajectory | 1.542 / 4.271 ± 0.399 REJECT | 1.542 / 4.271 ± 0.399 REJECT |
| **Ice Cream** | trajectory | 40.000 / **39.516 ± 0.948** ACCEPT | 40.000 / **40.000 ± 0.000** ACCEPT |
| The Hermit | use | 13.167 / 13.167 ACCEPT | 13.167 / 13.167 ACCEPT |
| Seed Money | round_end | 2.167 / 2.167 ACCEPT | 2.167 / 2.167 ACCEPT |
| Seed Money | rollout | INFO (no claim) | INFO (no claim) |
| `j_cloud_9__x3` (control) | round_end | 9.000 / 3.000 REJECT | 9.000 / 3.000 REJECT |
| `j_joker__doublecount` (control) | round_end | 4.000 / 0.000 REJECT | 4.000 / 0.000 REJECT |

Raw data: `ev/encode/poc_results_after_wfix.json` (the pre-fix `ev/encode/poc_results.json`
is left untouched, because POC_NOTES cites it as the evidence behind its own table).
Gate artefacts: `results/ev_player_gate_2026-08-26.{md,json}` hold the **post-fix** run.

Three rows are the fixes:

* **Satellite** flips REJECT → ACCEPT and is now *exact on both scenario families*. The
  entry was always right; the reject was the engine's. This is the harness's headline claim
  ("verification bites, and it localises the bug") closing out.
* **Ice Cream** goes from `39.516 ± 0.948` to `40.000 ± 0.000`. That CI was one seed's
  Brainstorm double-decay: with the copy guard the trajectory is the deterministic
  `100 − 5·hands` for every seed. **This is the "scaling trajectories with Blueprint
  present halve" check** — the halving is visible as the disappearance of the outlier.
* **Ride the Bus** moves because its face-rate calibration is fitted on real `ev:fast`
  trajectories, and those trajectories changed.
* **Green Joker** is unmoved, correctly: its rejection is a *modelling* bias (the closed
  form ignores the counter's floor at 0), not an engine defect, and no engine fix should
  have touched it.

**`pvp_canonical_transcripts.json` re-captured.** That fixture pinned "adding the PvP
protocol did not change the default path", byte for byte, over four seeds. Three of the
four moved — the fixes change what `ev:fast` plays, e.g. on `1558AXDL` player 1's Ice Cream
melts at step 366 and frees a slot, which changes every later shop decision. `1KV4W6YS` is
byte-identical. Each changed row keeps its pre-fix values under
`superseded_2026-08-26_wfix`, and the test's docstring records why, so the W-PVP claim
stays auditable and the pin keeps doing its job from here on.

## Corpora and fingerprints

**Existing label / pair corpora were generated under the pre-fix dynamics and are not
invalidated by this round.** V learned a value function for the simulator as it was; the
corpora remain internally consistent. Future generation runs inherit the fixed dynamics,
so a corpus mixing pre- and post-fix shards would be mixing two slightly different
simulators — antes reached move by ~0.03 for `ev:full` and not at all for `ev:fast`, and
1/126 (`fast`) to 6/126 (`full`) seeds take a different path.

**No fingerprint changes, and none of them would have.** The only fingerprint in the tree
is `agent/mcts/encoder_v2.layout_fingerprint()`, a sha256 over
`(STATE_SPEC_VERSION, SCALAR_LAYOUT_V2, ITEM_WIDTHS_V2, caps, KEY_VOCAB_V2, cardinalities)`
— it hashes the *observation layout*, not engine behaviour, and no file under `agent/` was
touched, so it is unchanged at
`5167cdc1785a4d7bf79d53c66b5fff447a745384287dee7310a4a20f36b19a3b`. `dataset.SHARD_VERSION`
/ `pairs.SHARD_VERSION` are likewise format versions, not behaviour hashes, and the shard
metadata records the *policy* (`ev:fast`, budget, epsilon) but nothing about the engine.

**There is therefore no automatic guard against mixing pre- and post-fix corpora.** If the
lead wants one, an engine-behaviour stamp in the shard metadata is the missing piece; this
round did not add one because inventing a versioning scheme is a design decision, not a
fix.

## Still open (not in this round's scope)

* **Cloud 9 counts a Stone-enhanced nine** (POC_NOTES §3.4, minor). `nine_tally` counts
  `v:get_id() == 9` and `Card:get_id` returns a random *negative* id for a Stone card
  (`card.lua:957-962`), so a 9 turned to Stone stops paying; the engine counts `c.rank == 9`
  regardless. Still pinned as a gap by
  `test_ENGINE_GAP_cloud_9_counts_a_stone_enhanced_nine`.
* **Vampire strips enhancements one card at a time** (`on_score_card`) where the Lua does
  the whole `context.scoring_hand` in the `before` pass (`card.lua:3465`). In the Lua a
  stripped Glass card therefore scores as a base card; in the engine it scores its x2 and
  is stripped afterwards. Pre-existing, unrelated to the copy guard, not measured here.
* **Square Joker** reads `len(ctx.scoring_cards)` where the Lua reads
  `#context.full_hand == 4` (`card.lua:3427`) — the played set, not the scoring set.
  Pre-existing.
* **Burnt Joker** is implemented as "upgrade the most-played hand at start of round"; the
  Lua (`card.lua:2749`) levels up the *first discarded* poker hand each round. Its `counts`
  bookkeeping is now guarded, but the joker is a different joker. Pre-existing.
