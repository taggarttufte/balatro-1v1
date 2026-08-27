# POC_NOTES — W-ENCODE-POC: does "LLM writes the encode layer, harness verifies it" work?

**2026-08-26.** Files: `ev/encode/{__init__,registry,verify,run_poc,conftest}.py`,
`ev/encode/pytest.ini`, `ev/encode/tests/{test_registry,test_verify,test_engine_fidelity}.py`
(63 tests), raw data `ev/encode/poc_results.json`. Nothing outside `ev/encode/` was written.
`ev/player.py`, `ev/hand.py`, `ev/train_v.py`, `ev/runs/`, `engine/**` and `_reference/**`
are read-only here. **Not committed.**

Reproduce:

```
python ev/encode/run_poc.py --workers 4 --traj-seeds 72 --worlds 40 --json ev/encode/poc_results.json
python -m pytest ev/encode -q
```

28 s and 3 s respectively, 4 worker processes, no GPU.

## 0. Headline

| | |
|---|---|
| real entries accepted | **8 / 10 measurements** (6 of 8 items clean on every mode) |
| negative controls | **2 / 2 correctly REJECTED**, for two different reasons |
| engine-fidelity divergences found | **5** (2 material, 1 material-under-Blueprint, 2 minor) |
| of those, found by MEASUREMENT rather than by reading the Lua | **2** (Blueprint double-scaling; the `_RATIO_CACHE` determinism leak) |
| harness bugs that masqueraded as entry bugs during the build | **1** (§6) |
| tests | **63 passed** (`python -m pytest ev/encode`) |
| wall clock, whole POC | 28 s (measurements) + 3 s (tests) |

> **Superseded in part, later the same day.**  W-FIX fixed three of the five divergences
> (§3.1 Satellite, §3.2 Blueprint double-scaling, §3.3 the Ice Cream melt) plus the §3.5
> `_RATIO_CACHE` leak, and flipped their pins in `tests/test_engine_fidelity.py` and
> `tests/test_verify.py` from "the engine is wrong, here is what it does" to "the engine is
> right".  Re-running `run_poc.py` on the fixed tree gives **9/10 real entries accepted**
> (Satellite REJECT → ACCEPT and exact; Ice Cream `39.516 ± 0.948` → `40.000 ± 0.000`),
> controls still 2/2 rejected.  §3.4 (Cloud 9 / Stone) is still open.  This document is
> left as the dated record of what was measured BEFORE those fixes — every number below is
> the pre-fix one — and `engine/FIX_NOTES.md` carries the after table and the before/after
> on every gate.

**Verdict: the loop works and verification bites.** Two of the eight hand-written entries
were wrong or unverifiable, and neither would have been caught by re-reading the Lua — one
is an engine defect, one is a modelling bias that only a rollout can size. Both controls
were rejected by the same code path that accepted the true entries. §8 has the honest
caveats and §9 the recommendation.

## 1. The per-item table

`predicted` and `measured` are means over the mode's scenarios/seeds; `±` is the 95%
half-width on the mean **residual** (`predicted_i − measured_i`), not on the measurement —
see §4 for why. Units: `$` dollars, `M` mult, `C` chips.

| item | tier | mode | n | predicted | measured ± CI | verdict | Lua |
|---|---|---|---|---|---|---|---|
| **Cloud 9** | det | round_end_paired | 4 | 3.000 $ | 3.000 ± 0.000 | **ACCEPT** (exact) | card.lua:1661-1663, :4191-4196; game.lua:444 |
| **Cloud 9** | det | rollout_paired | 40 | 21.500 $ | 21.475 ± 0.111 | **ACCEPT** | as above |
| **Rocket** | det | round_end_paired | 4 | 3.000 $ | 3.000 ± 0.000 | **ACCEPT** (exact) | card.lua:1664-1666, :2896-2902; state_events.lua:101, :1174 |
| **Satellite** | det | round_end_paired | 8 | 1.500 $ | 0.750 ± 0.807 | **REJECT** (inexact) | card.lua:1667-1673; game.lua:515 |
| **Ride the Bus** | policy | scaling_trajectory | 23 | 1.461 M | 1.652 ± 0.766 | **ACCEPT** | card.lua:3525-3540, :3992-3997, :964-970 |
| **Green Joker** | policy | scaling_trajectory | 59 | 1.542 M | 4.271 ± 0.399 | **REJECT** (band+ci) | card.lua:3563-3569, :2846-2856, :4010-4015 |
| **Ice Cream** | policy | scaling_trajectory | 62 | 40.000 C | 39.516 ± 0.948 | **ACCEPT** | card.lua:3570-3599, :3915-3920; game.lua:420 |
| **The Hermit** | det | use_paired | 6 | 13.167 $ | 13.167 ± 0.000 | **ACCEPT** (exact) | card.lua:1385-1391, :1530; game.lua:542 |
| **Seed Money** | det | round_end_paired | 6 | 2.167 $ | 2.167 ± 0.000 | **ACCEPT** (exact) | card.lua:1933; game.lua:602, :1909-1910; state_events.lua:1191-1202 |
| **Seed Money** | det | rollout_paired | 40 | — | Δ$ = −0.35 ± 1.41 | **INFO** (no claim) | as above |

Per-scenario detail for the deterministic entries (every one of these landed on the nose):

| item | scenarios | predicted → measured |
|---|---|---|
| Cloud 9 | nines = 0, 1, 4, 7 | 0→0, 1→1, 4→4, 7→7 |
| Rocket | (boss, bonus) = (0,1) (1,1) (0,3) (1,3) | 1→1, **3→3**, 3→3, **5→5** |
| The Hermit | balance $0, 5, 14, 20, 33, 60 | 0→0, 5→5, 14→14, 20→20, **20→20**, **20→20** |
| Seed Money | balance $0, 12, 27, 40, 55, 80 | 0→0, 0→0, **0→0**, 3→3, 5→5, **5→5** |
| Satellite | 3 planet counts × {used before purchase, used after} | before: 0,1,3,2 → **0,0,0,0**; after: 0,1,3,2 → 0,1,3,2 |

The bolded rows are the ones where a naive entry would have been wrong: Rocket's boss round
pays the *upgraded* $3 (not $1), The Hermit caps the **gain** at $20 (so $60 → $80, not
$40), and Seed Money's marginal value is **zero** below a $30 balance and saturates at $50.

### The two rejections

**Satellite — the entry is right and the ENGINE is wrong.** The two scenario families split
perfectly: every "planets used *after* the purchase" scenario is exact, every "planets used
*before* the purchase" scenario measures $0 against a predicted $1–$3. That is not a
modelling error, it is engine gap §3.1. The reject is correct behaviour — the harness
should refuse an entry it cannot confirm — but the *fix* is in the engine, and the harness's
own measured value (`$0.75 ± 0.81`, the mean over both families) is recorded as the
empirical fallback so nothing downstream has to guess in the meantime.

**Green Joker — the entry is biased and the harness sized the bias.** The closed form
`max(0, m₀ + hands − discards)` is exact only if the walk never touches its floor at 0. It
does, constantly: `ev:fast` discards **0.994 times per hand played**, so the counter spends
much of its life near zero and every visit to the floor is a permanent gain relative to the
unfloored walk. Predicted 1.54, measured **4.271 ± 0.399** mult after 12 hands — a 2.8×
underestimate, far outside both gates. The assumption was declared in the entry
(`registry.py`, `_predict_green_joker`), which is why this is a clean result rather than a
surprise: the entry said what it was taking on faith, and the harness priced it.

## 2. The negative controls

Both are scored by the **same** `verify.measure_round_end` call on the **same** scenarios as
their honest twins (`test_the_controls_and_the_truth_go_through_the_identical_measurement`);
if they had failed via a special code path they would have proved nothing.

| control | predicted | measured | verdict | which gate caught it |
|---|---|---|---|---|
| `j_cloud_9__x3` — 3× the true payout | 9.000 $ | 3.000 ± 6.198 | **REJECT** | **band** + inexact (**not** ci) |
| `j_joker__doublecount` — a +4 Mult joker priced as $4/round | 4.000 $ | 0.000 ± 0.000 | **REJECT** | unreachable + band + ci + inexact |

Two things are worth dwelling on.

**The CI gate did not catch the 3× error.** The residual is 6.00 and its 95% half-width is
6.198, so a residual-CI test *accepts* it. A pure scale error inflates the residual spread
as fast as it inflates the residual, so across a scenario family that spans a wide range the
CI is nearly blind to it. The 2× magnitude band is what bites, and it is not a redundant
belt-and-braces check — it is the only gate that sees this failure. Pinned in
`test_the_band_gate_rejects_a_scale_error_the_ci_would_swallow`.

**The double-count control was rejected for the sharper reason.** `j_joker` has no
end-of-round hook *at all*, so the reachability probe reports zero firings and the verdict
reads `unreachable+band+...` — the harness does not merely say "wrong number", it says "you
priced a row that does not exist". This is exactly the marginal rule doing its job: because
every measurement is `arm_with − arm_without` with **everything else active**, an effect the
dry-run scorer already prices (`HypotheticalScorer`, EV_NOTES §1) contributes exactly $0
marginal dollars, and any entry that re-prices one collapses to a measured zero. Stated in
code at `verify.MARGINAL_RULE` and demonstrated at
`test_an_already_priced_scoring_effect_measures_zero_dollars`.

## 3. Engine fidelity — the eight items

Five items faithful, four item-level divergences, plus one player-layer defect (§3.5). Per
the brief, **nothing was fixed**; each divergence is pinned by a test that asserts the
*current* behaviour and names what it should be, so a later fix has to update a test rather
than pass silently.

| item | verdict |
|---|---|
| Cloud 9 | MATCH on the payout, the whole-deck tally and the interest ordering — **minor gap §3.4** on Stone cards |
| Rocket | **MATCH**, including the ordering claim (`on_boss_beaten` fires before `on_round_end`, game.py ≈:2021-2032, mirroring state_events.lua:101 before :1174) |
| Satellite | **GAP §3.1 (material)** |
| Ride the Bus | MATCH (reads the scoring hand, not the played set; `is_face` guards) — but **§3.2** under Blueprint |
| Green Joker | MATCH (−1 per discard *action*, floored, incremented in `before`) — but **§3.2** under Blueprint |
| Ice Cream | MATCH on chips-then-decay ordering — **GAP §3.2** under Blueprint, **GAP §3.3** on the melt |
| The Hermit | **MATCH** (gain capped at $20, `max(0, …)` guard on a negative balance) |
| Seed Money | **MATCH** (engine stores the cap in interest-dollars 5→10, Lua in balance-dollars 25→50 — a units difference, equivalent everywhere) |

### 3.1 Satellite forgets every planet used before it was bought — MATERIAL

Lua reads the **global** `G.GAME.consumeable_usage` and counts distinct `set == 'Planet'`
entries (`card.lua:1667-1673`), so a Satellite bought at ante 4 after five distinct planets
pays $5 on its very first round end. The engine keeps the set on the **joker instance**
(`jokers/misc.py:208-217`), populated only by `consumables.apply_planet`'s `on_planet_used`
sweep (`consumables.py:55-59`), and never seeds it from `game.planets_used`. It therefore
pays **$0** until new planets are used.

Materiality is high: Satellite is a $6 rarity-2 joker that is essentially always acquired
mid-run — precisely the case the engine gets wrong — and it is a *pure* economy joker, so
this is 100% of its value, not a fraction. Test:
`test_ENGINE_GAP_satellite_forgets_planets_used_before_it_was_bought`.

### 3.2 Blueprint / Brainstorm double-scale every self-mutating joker — MATERIAL, and the one the harness found

Every self-mutating scaling joker guards its state change with `and not context.blueprint`
in the Lua: Ride the Bus (`card.lua:3525`), Green Joker (`:3563`), Ice Cream (`:3571`),
Obelisk (`:3542`). A copy reproduces the **score contribution** but not the **state
change**. The engine's `_Blueprint` / `_Brainstorm` call the target's `on_hand_scored` on the
target's own instance with no blueprint flag (`jokers/misc.py:145-195`, `_guarded_call`), and
the engine folds "apply the chips" and "decay" into that single hook
(`jokers/scaling.py:305-312`, `:51-58`, `:349-358`). So the copy mutates the target a second
time: **Ice Cream melts twice as fast; Green Joker and Ride the Bus scale twice as fast.**

This is the finding the POC exists to demonstrate, because **it was found by measurement,
not by reading**. The Lua was read correctly, the entry was written correctly, and 1 seed in
40 of the Ice Cream trajectory run came back at 10 chips instead of 40. Tracing that seed
(`7YTVQERM`) showed the decay rate doubling from 5/hand to 10/hand on the exact hand a
Brainstorm entered the board:

```
hand=6  chips 75->70   jokers=[ice_cream, raised_fist, swashbuckler, joker, abstract]
hand=7  chips 70->60   jokers=[ice_cream, raised_fist, swashbuckler, abstract, brainstorm]
hand=8  chips 60->50   ...
```

No amount of re-reading `card.lua` finds this: the bug is not in the item, it is in the
interaction, and only running the item inside the real engine surfaces it. Test:
`test_ENGINE_GAP_blueprint_double_scales_a_self_mutating_joker`. The blast radius is larger
than the POC set — every `not context.blueprint`-guarded joker in `card.lua` is a candidate,
which is a scan the fleet should do.

### 3.3 Ice Cream never melts — LOW for scoring, non-zero for policy

`if extra.chips - chip_mod <= 0 then … G.jokers:remove_card(self)` (`card.lua:3571-3592`):
Ice Cream is **destroyed** on the hand that would take it to zero and its joker slot is
freed. The engine floors it at 0 (`max(0, chips - 5)`) and leaves a dead card on the board
forever. Scoring impact is nil (a 0-chip joker adds 0 either way); *policy* impact is not,
because a permanently-occupied joker slot changes every later buy decision — and the encode
layer's buy value for Ice Cream should price "20 hands of decay, then a freed slot", not
"20 hands, then a blocked one". Test: `test_ENGINE_GAP_ice_cream_never_melts`.

### 3.4 Cloud 9 counts a Stone-enhanced nine — MINOR

`nine_tally` counts `v:get_id() == 9` and `Card:get_id` returns a random **negative** id for
a Stone card (`card.lua:957-962`), so a 9 turned to Stone stops paying. The engine counts
`c.rank == 9` regardless of enhancement (`game.py:2026-2027`). Needs a Stone conversion to
land on a 9 inside a Cloud 9 build; recorded because the `j_cloud_9` entry states the
assumption explicitly, and an untested assumption is not an assumption. Test:
`test_ENGINE_GAP_cloud_9_counts_a_stone_enhanced_nine`.

### 3.5 `ev:fast` is not reproducible across runs that share a process — MATERIAL for any harness

Not an engine gap; a **player-layer** one, and the second finding that came out of
measurement rather than reading. Symptom: two identical `--workers 4` runs of the same
trajectory measurement disagreed (Ride the Bus mean mult 1.52 vs 1.74 over the same 72
seeds), while `--workers 1` was perfectly reproducible with itself and disagreed with
`--workers 4`. Per-seed trajectories run in isolation were bit-identical, and
`PYTHONHASHSEED` made no difference — so the state was leaking *between seeds inside a
worker*.

Cause: `hand._RATIO_CACHE` is a process-global dict keyed by `_board_sig`
(`ev/hand.py:341-358`), and that key **deliberately omits planet levels and the exact deck
composition** — a documented speed trade ("a planet pick must not force a ratio recompute —
it was 40% of a pack decision"). But `board_ratio` samples real hands from the real deck at
the run's real planet levels, so two states that differ only in an omitted field share a
cache entry and whichever was computed first wins. With a 256-entry FIFO eviction on top,
what a seed sees depends on everything the process computed before it.

Isolated and quantified: sharing `_RATIO_CACHE` across runs changed **2 of 24 seeds (8%)**;
sharing `_MODEL_CACHE` changed **0 of 24**. `EV_NOTES` §5's determinism claim covers
side-effect freedom and *within-process* draw-order invariance, and is silent on this case.

Reach: it is not confined to this harness. `gate_ev_player.py` runs the 126-seed gate with
`pool.imap_unordered` over a reused worker pool, so its per-seed rows are
partition-dependent by the same mechanism — the aggregate is fine at 126 seeds, a single
seed's row is not. Any labels/pairs worker that plays multiple runs per process is in the
same position.

Not fixed here (`ev/hand.py` is read-only to this workstream). The harness defends itself:
`verify.reset_player_caches()` clears both caches at every run boundary, which restores
`workers=1 == workers=4 == workers=6` exactly. Pinned mechanistically (not statistically) by
`test_the_board_ratio_cache_key_ignores_planet_levels` — two boards with an identical
`_board_sig` and different planet levels, where the second is provably served the first's
number — plus `test_a_trajectory_is_independent_of_what_ran_before_it`.

## 4. What the harness measures, and why it is built this way

**Everything is marginal.** Two arms are built from one state, differing only by the item,
with every other joker, voucher, seal and blind reward live in both; the measurement is
`arm_with − arm_without`. This is stated at `verify.MARGINAL_RULE` and is the single
design decision that makes double-counting fail loudly instead of silently inflating the
player's shop scores.

**Four gates, and they catch different lies** (`verify.Measurement`):

| gate | catches |
|---|---|
| `reachable` | "the number matched because nothing fired" — the project's A1 lesson |
| `within_ci` | ordinary noise around a mean, on the **residual**, not the measurement |
| `within_band` | scale errors (the 2× band) and, via its zero-aware case, every double count |
| `exact` | deterministic entries must land on **every** scenario, not on average |

The residual (rather than the measurement) is what carries the CI because the scenario
families deliberately span a wide range — a CI on `measured` would be a CI on "how spread
out are my test cases", which is not a statistic about the entry at all.

**Reachability is instrumented, not assumed.** `verify.reach_probe` installs a proxy around
the item's engine effect for the duration of a block and counts hook invocations and pending
money written. `JOKER_REGISTRY` is a `_JokerRegistry` that refuses double registration
(`jokers/base.py:16-28`) — a deliberate guard from the Phase-1 re-key — so the probe goes
around it via `dict.__setitem__` explicitly, restores in a `finally`, and is pinned to be
both transparent to the engine and non-weakening of the guard
(`test_the_probe_is_transparent_to_the_engine`,
`test_the_probe_does_not_weaken_the_double_registration_guard`).

**Policy rates are measured, not guessed.** The whole reason the scaling tier is hard is
that its value depends on what the policy does. The trajectory mode reports what `ev:fast`
actually did over 72 real seeds:

| rate | value |
|---|---|
| discards per hand played | **0.994** (Green Joker), 0.978 (Ride the Bus), 0.891 (Ice Cream) |
| P(the scoring hand contains a face) | **0.406** |

Those two numbers are a Lua-invisible deliverable — they are the constants the closed forms
need, and no amount of source reading produces them.

**The rate is not fitted on the data it is scored against.** Ride the Bus is the only entry
that needs a fitted rate, so the first half of the seeds is a calibration split used *only*
to estimate the face rate (23 seeds), and the entry is scored on the held-out remainder (23
seeds). Green Joker needs no fit at all — each seed's predictor is fed that seed's own
realized hand and discard counts, so its residual is pure mechanism error. Ice Cream needs
no rate.

## 5. The shop blind spot, measured

An unplanned but load-bearing result. The trajectory mode installs the joker at ante 1 and
plays real `ev:fast`; the joker can leave the board, and it does — **the current shop rules
sell it**, which is the exact blind spot this package was built to close (EV_NOTES §8 item
4: "scaling jokers … show no immediate strength"):

| joker | seeds run | **sold before 12 hands** | survival |
|---|---|---|---|
| Ride the Bus | 72 | **18 (25.0%)** | 75.0% |
| Green Joker | 72 | **10 (13.9%)** | 86.1% |
| Ice Cream | 72 | **5 (6.9%)** | 93.1% |

Every loss was a sale (`lost_reasons: {'sold': …}`), not a destruction. Those seeds are
**reported, not averaged in** — including them would measure the shop's opinion of the item
rather than the item — but the rate itself is the size of the prize: on a quarter of seeds
the player throws Ride the Bus away before it has scaled at all.

The other half of the prize is the **realized** value, which is smaller than the sticker
number:

* **Ride the Bus** reaches only **1.65 mult after 12 hands** (sd 1.87 over 23 seeds, so the
  mean is ±0.77), because a 0.406 face rate resets it roughly every 2.5 hands. It is a much
  weaker buy than "+1 mult per hand" reads.
* **Cloud 9** paid a realized gross of **$21.48 over ~5.4 round-ends** (`gross_per_payout =
  3.995` against a 4-nine deck — the predictor is right to four significant figures), and
  its CRN with-vs-without final-money delta was **+$6.35 ± 3.80** with an ante-progress delta
  of **+0.10 ± 0.29**. The gross and the delta are different numbers on purpose: the extra
  dollars get *spent*, and the arms diverge the moment one can afford something the other
  cannot. Only the gross is what the entry claims.
* **Seed Money's** realized buy value over antes 1–5 is **−$0.35 ± 1.41, i.e. statistically
  zero** — `ev:fast` rarely holds the $30+ balance the voucher needs. The per-round formula
  is exact and the buy value is nothing at this depth. (Amusingly this means the shop rules'
  flat +0.02 for a voucher is roughly *right* for Seed Money before ante 5, for entirely
  the wrong reason. It will be badly wrong deeper, and badly wrong for other vouchers.)

## 6. A harness bug that looked exactly like an entry bug

Worth recording because it is the failure mode a fleet will hit at scale. The first version
of the paired-rollout mode scored Cloud 9's per-round predictor against a **blind counter**
built from state transitions. It over-counted by ~0.9 rounds per run, and Cloud 9 — an entry
that is exact in every other mode — came back `pred=25.0 meas=22.2 REJECT (ci)`. The entry
was fine; the harness's bookkeeping was not.

Fixed by scoring against the **hook-firing count** from the reachability probe instead of a
derived tally: the number of payout opportunities is now measured by the same instrument
that measures the payout. After the fix: `pred=21.5 meas=21.475 ± 0.111 ACCEPT`.

The lesson for L1: **a rejection is not evidence about the entry until the harness's own
denominators are instrumented rather than derived.** A fleet producing hundreds of entries
will generate false rejects at a rate set by the harness's weakest bookkeeping, and a false
reject is worse than no check — it burns an agent-hour and erodes trust in the gate.

## 7. Empirical fallbacks recorded

Per the brief, a rejected or ambiguous entry hands back the harness's own measured number
(`verify.empirical_fallback`, in `poc_results.json` under `empirical_fallbacks`):

| key | measured | ci95 | n | supersedes the closed form? |
|---|---|---|---|---|
| `j_satellite` | $0.75 / round | 0.807 | 8 | yes — and it is a fallback for the *engine*, not the item |
| `j_green_joker` | 4.271 mult @ 12 hands | 0.399 | 59 | yes |
| `v_seed_money` (buy value) | −$0.35 @ ante 5 | 1.413 | 40 | n/a — the entry makes no buy-value claim; INFO row |
| `j_cloud_9` (buy value) | +$6.35 @ ante 4 | 3.795 | 40 | n/a — same |

The Green Joker row is the interesting one: a measured `4.271 ± 0.399` mult after 12 hands
under `ev:fast` is directly usable by shop logic *today*, and is strictly better than either
the wrong closed form or the current implicit zero.

## 8. What this POC does NOT show

Said plainly, because the temptation is to over-read a 8/10.

1. **No item was integrated into the player.** Nothing here has been shown to improve a
   single decision, let alone a win rate. The POC answers "can we generate and verify", not
   "does verified knowledge help". Gate (b)-style A/B evidence is entirely absent.
2. **8 items is not a sample.** They were chosen to span the tiers, and three of them
   (Cloud 9, Hermit, Seed Money) are close to the easiest closed forms in the game. A fleet
   faces Obelisk, Vampire, Hologram, Campfire, the Spectral cards and the boss blinds, and
   the pass rate on *those* is unknown.
3. **The entries and the harness were written by the same agent in the same session.** That
   is the strongest confound here. The scenarios test what I already knew the predictors
   did; a fleet worker's blind spot and its verifier's blind spot would be correlated in the
   same way unless they are deliberately separated (§9).
4. **`ev:fast` is one policy.** Every policy-conditional number — the 0.99 discards/hand,
   the 0.394 face rate, the sell rates, the Seed Money buy value — is a property of *this*
   player at *this* depth. A stronger shop policy changes all of them, which is circular in
   a way L1 has to confront: the encode layer is calibrated against the player it is meant
   to improve.
5. **Depth is shallow.** Trajectories run 12 hands; rollouts stop at ante 4–5. Rocket's
   unbounded growth, Ice Cream's melt at 20 hands and Seed Money's real value all live
   deeper than anything measured here.
6. **The stochastic tier is untested.** No POC item is 1-in-k, so the CI band has never had
   to do real work; the deterministic entries all have zero-variance measurements. The tier
   exists in `registry.py` with a documented consumer and no evidence behind it.
7. **Single-player vanilla only.** No Nemesis, no MLB ruleset, no PvP blind. The PvP money
   rows differ (PVP_NOTES §1.4a-b) and none of that was exercised.
8. **The `ev/encode` suite is fast because it is shallow.** 63 tests in 3 s means almost
   nothing is running a real game to depth. That is deliberate for a POC and is not a
   property L1 gets to keep.

## 9. What the full L1 harness must add

Ordered by how much they would change the outcome.

1. **Separate the author from the verifier.** The confound in §8.3 is the one that most
   threatens the whole design. L1 should have the entry-writing agent emit `predict` **plus
   the scenarios it believes are discriminating**, and a *different* agent (or a scripted
   adversary) generate the scenarios that actually score it — including out-of-range,
   boundary and interaction cases the author did not think of. A cheap version: auto-generate
   scenarios by sweeping every summary field the predictor reads, and reject any entry whose
   predictor ignores a field it declares.
2. **Interaction sweeps, not solo measurements.** §3.2 is the whole argument. Every entry
   should be measured at least once with Blueprint/Brainstorm adjacent, once with a
   retrigger joker (Mime, Sock and Buskin, Hanging Chad, Seltzer), once debuffed by a boss,
   and once with Oops! All 6s present. Four of the five divergences found here are
   interaction, provenance or process-state bugs rather than arithmetic bugs, and a solo
   single-run harness finds none of them by design.
3. **A cold-cache contract for the player.** §3.5. Either `_board_sig` gets the fields it
   is missing (and the shop pays the recompute), or `ev/player.py` grows a documented
   `reset()`-style cache boundary that every multi-run worker calls. Until one of those
   exists, no per-seed number produced by a reused worker pool — the gate's included — is
   reproducible, and a verification fleet cannot tell an entry regression from a scheduling
   artefact.
4. **Instrumented denominators everywhere.** §6. Any quantity a predictor is multiplied by
   (rounds, hands, triggers) must come from the probe, never from a derived counter.
5. **A real stochastic path.** Wilson/normal bands sized for a target resolution, a declared
   `n` per entry driven by the variance the item actually has, and the Oops! All 6s
   probability doubling folded in (`base.prob_roll`, EXTRACT_NOTES §2) — the existing
   proc-EV layer already has the semantics and should be reused rather than re-derived.
6. **Depth and policy sweeps.** Trajectories to 40+ hands (past Ice Cream's melt and into
   Rocket's growth), rollouts to ante 8, and at least two policies (`ev:fast`, `ev:full`) so
   the policy-conditional constants come with a sensitivity, not a point estimate. Report
   the constant *and* its policy derivative.
7. **A regression corpus, not a one-shot table.** Every accepted entry becomes a pinned test
   with its measured value and CI, so an engine change that breaks an entry fails loudly.
   The four `test_ENGINE_GAP_*` tests (and the two §3.5 cache tests) are the pattern: pin
   the current behaviour, name the divergence, force a later fix to update a test.
8. **Currency conversion, and its own verification.** Everything here stops at dollars,
   mult and chips. Turning "4.15 mult after 12 hands" into a shop decision needs an exchange
   rate, and that rate is V's job (brief §1, the ENCODE/LEARN split). L1 must not smuggle a
   hand-tuned conversion into the encode layer — but it does need a documented interface for
   handing the raw units over, and a check that the conversion is applied exactly once.
9. **Double-count auditing against the existing layers.** The marginal rule catches a double
   count *if a scenario exercises it*. L1 should additionally cross-reference every new entry
   against what `ev/hand.py`'s `ProcBoard` already prices (the nine procs of EXTRACT_NOTES
   §2) and what the dry-run scorer covers, and refuse an entry whose key already appears
   there without an explicit override.
10. **Cost accounting.** This POC cost ~28 s of compute for 8 items on 4 cores. That is not
    the fleet's cost — the fleet's cost is agent time reading Lua and iterating on rejects,
    and L1 should measure reject-and-retry loops per item before anyone budgets a fleet.

## 10. Files

```
ev/encode/
  __init__.py              what this package is and is not
  registry.py              10 entries (8 real + 2 negative controls), the Entry contract
  verify.py                4 measurement modes, the 4 accept gates, the reachability probe
  run_poc.py               the scenario builders + the driver
  conftest.py              ev/conftest.py's bootstrap + a `pytest_ignore_collect` that
                           hides this package from any session whose rootdir is not
                           ev/encode (see below)
  pytest.ini               rootdir for `python -m pytest ev/encode`
  POC_NOTES.md             this file
  poc_results.json         raw data behind every number above
  tests/
    test_registry.py       the entry contract + every closed form vs hand arithmetic
                           (incl. Ride the Bus against a brute-force enumeration)
    test_verify.py         each accept gate in isolation, the probe, the marginal rule,
                           both controls end to end, and the §3.5 cache leak
    test_engine_fidelity.py  the 8 items vs the Lua, and the 4 item divergences pinned
```

`python -m pytest ev/encode` → **63 passed in ~3 s**.

**Staying out of the ev suite took more than not editing `testpaths`.** `testpaths` only
applies when pytest is given no arguments, and the repo's documented command is
`python -m pytest ev` — an explicit path, which makes pytest recurse into every `test_*.py`
underneath it. Adding the files alone silently grew that suite from 333 to 396 collected.
`ev/encode/conftest.py`'s `pytest_ignore_collect` skips the directory unless
`ev/encode/pytest.ini` is the session rootdir, so:

| command | collects |
|---|---|
| `python -m pytest ev` | **333** — unchanged, this package invisible |
| `python -m pytest ev/tests` | 317 — unchanged |
| `python -m pytest ev/encode` | **63** — this package only |

`ev/conftest.py` and `ev/pytest.ini` were not touched.
