# STATS_NOTES — W4 decision-statistics module (Phase 5 rev 2)

Owner: W4. Implements `docs/PHASE5_BRIEF_2026-08.md` row W4 / interface §2 / gate 3.
Files: `decide.py` (the table), `hit.py` (hit valuation + P(hit)), `economy.py` (interest /
true cost), `urgency.py`, `sweep.py` (126-seed CLI), `bench_decide.py` (gate-3 timing
benchmark), `tests/test_*.py`.

## 0. TL;DR for W3 / W6

```python
import decide
rows = decide.decision_table(game)          # list[Row], sorted by net_ev desc, <=50ms
best = rows[0]                               # argmax net_ev, ties broken by legal_actions() order
action = best.action                         # exactly one of game.legal_actions()
print(decide.explain(rows))                  # pretty table for a CLI/advisor
```

`decision_table` returns `[]` outside `State.SHOP` / `State.BOOSTER_OPEN`. Everything is
read-only: the only mutation anywhere in the call graph is a *clone* (`game.run_state.clone()`
for pool lookups, `game.run_state.rng.clone()` for dry runs) — `game.state_signature()` is
bit-identical before/after (`tests/test_decide.py::test_decision_table_side_effect_free`,
`tests/test_hit.py`, `tests/test_urgency.py` pin this per-module too).

## 1. Common unit — the $-equivalent, precisely

Every `hit_value` / `cost` / `interest_loss` / `true_cost` / `net_ev` is in dollars.
Two conversions feed everything else (brief §2's ask: "value = expected score uplift
converted via a documented $/score-multiplier schedule by ante, plus its sell value"):

1. **Known joker → $ (precise, `hit.joker_hit_value`).** A side-effect-free dry run
   (`hit._score_with_jokers`, the same pattern as `card_selection.HypotheticalScorer` but
   able to inject a *hypothetical extra* joker, which `HypotheticalScorer` cannot) scores
   `cfg.n_hand_samples` (default 6) hands sampled from `game.full_deck`'s COMPOSITION (never
   its shuffle order — never `game.run_state.rng` directly; a private clone) with and without
   the candidate, same sampled hands and same "luck" stream position both times (so the delta
   isolates the joker's contribution, not sampling noise). The mean score delta is divided by
   the **next blind's chip target** (`urgency.next_blind_chip_target`) and multiplied by
   `dollars_per_score_ratio` ($15 default): *"how many dollars is moving 100% of the way to
   clearing the next blind, in one hand, worth."* Dividing by the ante-scaled chip target IS
   the "$/score-multiplier schedule by ante" the brief asks for — no separate per-ante table
   is needed because the target already grows with ante the same way score needs to.
   Scaling jokers are then scaled by `synergy.SCALING_DECAY_BY_ANTE` /
   `IMMEDIATE_BONUS_BY_ANTE` (**reused, not re-derived** — the documented horizon assumption
   the brief asks for), and the joker's game sell value (`JOKER_COST[key]//2`, floored at 1)
   is added: buying is never valued below buying-then-immediately-selling.

2. **Unknown joker (behind a reroll / pack) → $ (fast, `hit.pool_dollar_value`).** Dry-running
   all ~150 jokers every shop visit is out of budget, so a pool member gets a cheap proxy:
   `synergy.estimate_joker_strength(key)` (1..10, the existing hand-tuned S-tier/weak/etc.
   table) × a coherence multiplier (`synergy.coherence_score`, floored at `anti_synergy_floor`
   = 0.3 so an off-direction pick still has *some* value, not zero) × `dollars_per_strength_point`
   ($2.2 default). This is **only** used to (a) classify "hit" by a $ threshold
   (`pool_hit_threshold_dollars` = $8) and (b) average over the hits that survive — it never
   prices a KNOWN card; a shelf/pack item always gets the precise dry run (or the flat table
   below).

Consumables / cards / vouchers use small flat tables, per the brief ("simple documented
tables"):

| Kind | Formula | Notes |
|---|---|---|
| Tarot | flat `$4` | every tarot is priced identically — real Balatro tarot power varies a lot; explicitly a documented gap, not modelled (dry-running a Tarot's card-transform effect is out of scope for a sub-50ms decision) |
| Spectral | flat `$5`, except `c_soul`/`c_black_hole` = `$30` | those two are legendary-joker-tier outcomes |
| Planet | `$6 × play_share × 12 / level`, Laplace-smoothed, capped at 3× base | play_share comes from `game._hand_type_counts` (the run's own tally), smoothed against a uniform 1/12 prior with pseudocount 2 so 1-2 early hands cannot spike one type's share to ~1.0 (a real bug caught during development — see §5) |
| Voucher | hand-curated table (24 of 32) else `cost × 1.3` | a full per-voucher effect model is out of scope; the curated list covers the well-known standouts (Telescope, Observatory, the Tycoon/Merchant tiers, ...) |
| Playing card | `$1` base + enhancement table (Steel `$4.5`, Gold `$3.5`, ...) + edition table | no build-fit modelling |

## 2. P(hit) — exact from the generator's pools (`hit.py`)

`rng/generate.get_current_pool(state, _type, _rarity, ante)` is the SAME function the
game's shop/pack generation calls — importing it via `balatro_sim.game_keys.gen` (the module
instance the engine is already wired against, never a second copy) means "the generator's own
API" is used, not a reimplementation. Its only RNG read is the joker rarity roll, which an
explicit `_rarity` (0.5 / 0.85 / 1.0, probing Common/Uncommon/Rare — `rarity_from_roll`:
`r>0.95` Rare 5%, `0.7<r<=0.95` Uncommon 25%, else Common 70%, **fixed odds, independent of
pool size**) skips entirely — so `shop_slot_distribution` / `pack_p_hit` consume **zero RNG**
and are safe to call directly (we still clone `run_state` defensively — cheap, a few small
sets/dicts — so a bug in that invariant can never leak into the game).

**One shop slot's distribution** = `shop_type_table` weights (`run_state.joker_rate` /
`tarot_rate` / `planet_rate` / `spectral_rate` — read live, so Illusion / Magic Trick / Ghost
rate changes are honoured automatically) × (for Joker) the fixed rarity odds × (fraction of
that culled pool classified a "hit"). **A reroll** = `1 - (1 - p_slot)^shop_joker_max`
(independence approximation across slots — documented below). **A pack** =
`1 - (1 - p_card)^size` over the pack's own content type/kind (Buffoon→Joker, Arcana→Tarot,
Celestial→Planet, Spectral→Spectral; Standard packs have no hit model, see §5).

Consumable "hit": every eligible Tarot/Planet/Spectral in the culled pool with positive
documented value counts as a hit (in practice: every eligible member, since the flat tables
are never zero for an eligible key) — the *differentiator* between consumable pool draws is
`hit_value`, not `p_hit`. Validated empirically below: Arcana/Celestial/Spectral packs come
out at `p_hit = 1.0` both analytically and empirically, exactly as this design predicts.

### 2.1 Independence approximation (documented, not fixed)

Slots/cards are treated as independent draws from the same distribution. The real generator
dedupes across slots and pack cards (a joker on slot 1 cannot also appear on slot 2 —
GENERATION_SPEC §6), which makes the TRUE P(≥1 hit) very slightly higher than this estimate.
The gap is small when the pool (20-150 members) is large relative to the number of draws
(2 shop slots, 2-5 pack cards) — confirmed by the empirical validation below (max observed
gap 0.017, well inside sampling noise).

## 3. Economy (`economy.py`)

- `interest_now` / `interest_after`: `min(dollars // 5, interest_cap)`, exactly
  `_end_round`'s formula (game.py:1911), reading `game.interest_cap` (raised by Seed
  Money / Money Tree) not the static constant.
- `shops_remaining`: from a `SHOP` state, `game.blind_idx`/`game.ante` describe the blind
  JUST FINISHED for Small/Big; the ONE exception is right after a Boss, where `_end_round`
  has already bumped `game.ante` before entering the shop while `blind_idx` is still 2 until
  `leave_shop`. `shops_remaining` (and `urgency.next_blind_info`, which needed the same fact)
  both encode this explicitly, with a unit test each (`test_economy.py::
  test_shops_remaining_counts_to_ante_8`, `test_urgency.py::test_next_blind_info_*`).
- `interest_loss`: **geometrically discounted**, not flat. A flat `per_round × shops_remaining`
  (tried first) overstated the loss badly — at ante 1 with `shops_remaining=10`, a $5 reroll
  showed `interest_loss=$10`, more than doubling its true cost, because it assumes the
  dollar shortfall persists unchanged for all 10 future shops even though blind income
  routinely re-crosses the lost $5 tier within a round or two. `decay=0.85` (documented
  constant) gives `per_round × Σ_{t=1..H} decay^t`, matching the brief's own phrase
  "horizon-discounted" and roughly halving the naive estimate at typical horizons.
- No slot-opportunity cost (brief: "+ slot opportunity cost if you model it") — not modelled;
  a joker/consumable-slot purchase's opportunity cost of a FULL slot is implicitly present in
  `legal_actions()` itself (a full-slots buy is simply not legal), so the gap is narrower than
  it sounds, but a genuine "is this slot better spent on something else" comparison is not done.

## 4. Urgency (`urgency.py`)

`urgency = clip01(0.7 × shortfall + 0.2 × life_pressure + 0.1 × nemesis_bonus)`.

- `shortfall = clip01(1 − projected_total / next_blind_target)`, where `projected_total =
  mean(sample_hand_scores) × base_hands` — sampled 5-card hands from `full_deck`'s
  composition (a W0-heuristic-shaped proxy; `mcts/heuristic.py::HandHeuristic` could not be
  reused directly, because it scores `game.hand`, and in `SHOP`/`BOOSTER_OPEN` that list
  still holds the PREVIOUS blind's cards — game.py:1431-1435 says so explicitly).
- `life_pressure = clip01((4 − lives) / 4)` under MLB (0 otherwise) — losing ANY blind costs a
  life under MLB (not Nemesis-only; `eval/common.py::play_sp_mlb`'s own comment says so),
  so life pressure is not gated on the next blind being a Nemesis.
- `nemesis_bonus = 1.0` iff the next blind is a Nemesis (small top-up, since the Nemesis
  target itself is already folded into the horizon via `next_blind_chip_target`'s proxy —
  see below).
- **MLB Nemesis target proxy**: the real target is the opponent's live score, unknowable from
  the shop. `next_blind_chip_target` falls back to the vanilla boss-blind chip formula at that
  ante — the same "external, calibration-free" idea `eval/common.py::
  external_vanilla_big_blind_target` uses. Documented approximation, not a claim of accuracy.
- **Known gap**: `next_blind_info`'s stale-blind_idx correction assumes `BOOSTER_OPEN` was
  entered from `SHOP`. A tag-triggered pack opened at `BLIND_SELECT` (Charm/Meteor/Ethereal/
  Standard/Buffoon tags — `_booster_return_state` can be `BLIND_SELECT`) gets a
  one-blind-off urgency estimate in that rare path. Low priority: it only perturbs `urgency`
  (a soft multiplier on `hit_value`), not `p_hit` or the base `hit_value`.

## 5. Known simplifications / deviations from the interface

- **`kind` enum extended**: the brief lists `"buy_joker" | "buy_consumable" | "buy_card" |
  "buy_voucher" | "buy_pack" | "reroll" | "pick" | "skip_pack" | "leave" | "sell"`.
  `SHOP`'s `legal_actions()` also offers a genuinely useful, cost-free `use_consumable` action
  (using an already-owned Planet to level up a hand type — the ONLY consumable action
  offered in `SHOP`, per game.py:1431-1435's own comment about stale hand targets) with no
  home in that enum, so `decide.py` emits `kind="use_consumable"` for it. Flagged for W3/W6:
  if `EVPlayer` filters on the exact 10-kind set it will silently never use an owned Planet
  from the shop table — worth a one-line allowlist addition.
- **Standard (playing-card) packs**: `pack_p_hit` returns `(0.0, 0.0, {})` — no P(hit)/value
  model for random enhanced playing cards (confirmed in the sweep: Standard packs are always
  the worst `buy_pack` row by a wide margin, `net_ev ≈ −cost − interest_loss`, i.e. correctly
  never recommended, but only because the floor is 0, not because a real evaluation was done).
- **Telescope's forced first Celestial card** (guarantees the most-played hand's planet) is
  not special-cased; `pack_p_hit("Celestial", ...)` treats every card as an ordinary Planet
  draw. Rare in practice (needs the Telescope voucher) and the aggregate error is small since
  every planet already counts as a "hit" in this model.
- **Independence approximation** across shelf slots / pack cards (§2.1).
- **Sell row's `ongoing_value`** reuses `joker_hit_value` (the "buy" appraisal) as a proxy for
  "value of continuing to own it" — reasonable but not a distinct model of *removing* a joker
  from an established board (e.g. losing synergy with jokers built around it is not priced).

## 6. Validation

### 6.1 P(hit) analytic vs. empirical (gate 3 requirement)

Method: `game.clone_determinized(seed)` (W2's API, confirmed present via `hasattr` per the
brief's instruction — no RNG-state hacking) gives each trial an independent fresh
`run_state.rng` with identical generation-layer bookkeeping (`used_jokers`, owned lists,
`shop_joker_max`, rates, banned keys — everything the analytic computation itself reads).
`balatro_sim.shop.reroll_shop` (the engine-level wrapper) is money-gated and silently no-ops
if the clone can't afford it — caught during development (see below) — so trials top up
`clone.dollars` first. Fresh determinized clones + a well-funded reroll/pack-open removes
every remaining state dependency; the only thing varying between trials is the RNG stream.

**Reroll P(hit)**, 500 trials/state, 6 seeds × 3 shop visits (18 states, ante 1, 0-2 owned
jokers):

| seed | ante | jokers owned | analytic P(hit) | empirical P(hit) | 95% CI halfwidth | diff |
|---|---|---|---|---|---|---|
| 11111111 | 1 | 0 | 0.721 | 0.720 | 0.039 | 0.001 |
| 11111111 | 1 | 1 | 0.724 | 0.722 | 0.039 | 0.002 |
| 11111111 | 1 | 1 | 0.721 | 0.722 | 0.039 | 0.001 |
| 1558AXDL | 1 | 0 | 0.723 | 0.720 | 0.039 | 0.003 |
| 1558AXDL | 1 | 1 | 0.726 | 0.724 | 0.039 | 0.002 |
| 1558AXDL | 1 | 1 | 0.723 | 0.724 | 0.039 | 0.001 |
| 15H9Z3IY | 1 | 0 | 0.715 | 0.720 | 0.039 | 0.005 |
| 15H9Z3IY | 1 | 1 | 0.742 | 0.748 | 0.038 | 0.006 |
| 15H9Z3IY | 1 | 1 | 0.742 | 0.748 | 0.038 | 0.006 |
| 1KV4W6YS | 1 | 0 | 0.718 | 0.720 | 0.039 | 0.002 |
| 1KV4W6YS | 1 | 1 | 0.774 | 0.758 | 0.037 | 0.017 |
| 1KV4W6YS | 1 | 1 | 0.773 | 0.758 | 0.037 | 0.015 |
| 1MD1YZ9T | 1 | 0 | 0.704 | 0.720 | 0.039 | 0.016 |
| 1MD1YZ9T | 1 | 1 | 0.707 | 0.724 | 0.039 | 0.017 |
| 1MD1YZ9T | 1 | 1 | 0.720 | 0.724 | 0.039 | 0.004 |
| 1RG3WA3F | 1 | 0 | 0.720 | 0.720 | 0.039 | 0.000 |
| 1RG3WA3F | 1 | 1 | 0.723 | 0.722 | 0.039 | 0.001 |
| 1RG3WA3F | 1 | 2 | 0.717 | 0.722 | 0.039 | 0.005 |

**18/18 within their 95% CI, max diff 0.017.** (An earlier run without the dollar top-up
produced 4/18 catastrophic failures — 100%/0% empirical rates with zero variance, because the
gated `reroll_shop` was silently refusing every trial on an unaffordable clone, leaving the
ORIGINAL undisturbed shelf in place every time. Caught precisely because zero-variance
empirical rates across 500 independent RNG streams is itself diagnostic of a broken harness,
not a model problem — worth remembering for whoever validates a probability model against a
game state next.)

**Pack P(hit)**, 500 trials/state, 3 seeds × 4 pack kinds (12 states):

| pack kind | size | analytic P(hit) | empirical P(hit) | within CI |
|---|---|---|---|---|
| Buffoon | 2 | 0.442–0.457 | 0.476–0.494 | yes (3/3) |
| Arcana | 3 | 1.000 | 1.000 | yes (3/3, exact) |
| Celestial | 5 | 1.000 | 1.000 | yes (3/3, exact) |
| Spectral | 2 | 1.000 | 1.000 | yes (3/3, exact) |

**12/12 within CI.** The Arcana/Celestial/Spectral packs' exact 1.0/1.0 match is the
"every eligible consumable counts as a hit" design (§2) working as predicted, not a
coincidence — Buffoon (Joker) packs are the only ones with a non-trivial threshold and they
show real sampling variation matching the analytic number within a few points.

Reproduce: `python -m pytest stats/tests/test_phit_validation.py -q` (small N=150/state,
CI-fast); the 500-trial numbers above came from ad hoc scripts (not committed — see §8) run
single-process per the lead's 2026-08-23 resource note.

### 6.2 Gate-3 timing benchmark

`python stats/bench_decide.py --n-states 320` — 320 REAL `SHOP`/`BOOSTER_OPEN` states from
scripted-player runs across all 126 ground-truth seeds (8 states/seed until the quota fills):

```
n_states=320  mean=1.293ms  p50=1.226ms  p95=2.496ms  p99=3.019ms  max=3.288ms
```

Mean is **~39x under** the 50ms gate. (In-suite `tests/test_decide.py::
test_decision_table_timing_sanity` re-checks this on a smaller N so CI catches a regression
without needing the full 320-state / 126-seed drive.)

### 6.3 Test suite

`python -m pytest stats/tests -q` → **50 passed in 0.88s** (well under the 60s gate).

## 7. Sweep headline (SMOKE SCALE ONLY — see §9 for the real command)

Per the lead's 2026-08-23 resource note, the box was in interactive use, so only a
**6-seed, 4-process smoke test** was run (`results/stats_sweep_2026-08-23_smoke.{json,md}`)
to validate `sweep.py` itself, not to produce the real headline numbers. 6 seeds × up to 3
antes-worth of visits reached (the scripted player is slow to clear later antes) = 38 visits,
0 errors, 0.7s wall with 4 workers.

| ante | visits | reroll P(hit) | reroll net_ev | best net_ev | % leave best | int.loss/true_cost | urgency mean | voucher net_ev |
|---|---|---|---|---|---|---|---|---|
| 1 | 12 | 0.72 | -1.97 | 2.58 | 25% | 40% | 0.52 | -3.28 |
| 2 | 16 | 0.64 | -1.39 | 5.67 | 0% | 43% | 0.69 | -1.09 |
| 3 | 9 | 0.63 | 0.60 | 5.83 | 11% | 33% | 0.79 | -1.16 |
| 4 | 1 | 0.64 | 2.97 | 6.49 | 0% | 0% | 0.83 | 6.49 |

**Interpretation (smoke-scale, indicative only — do not cite as the real headline):**

1. The best row's net EV rises with ante (2.6 → 6.5): later shops have more money and pricier,
   higher-value jokers/packs available, so there is more surplus to capture.
2. Rerolling is usually net-negative on its own row (-1.97 to +0.60): a lone reroll rarely
   beats its true cost, but it is never the *best* row here either — buying/using something
   already on the shelf usually wins, consistent with reroll being a situational tool, not a
   default action.
3. `% best-is-leave` falls from 25% (ante 1, money-constrained) to 0% (ante 2-4): once dollars
   accumulate, the shop consistently has *something* worth the spend.
4. Interest-loss is 33-43% of the best row's true cost at every ante shown — a substantial
   share, confirming the brief's instinct that ignoring interest would materially mis-price
   shop actions (an un-scaled "sticker price only" comparison would rank several rows
   differently).
5. Urgency climbs steadily with ante (0.52 → 0.83): blind targets are scaling up faster than
   this scripted player's build, exactly the "the game gets harder" signal `urgency` exists
   to surface.
6. Voucher `net_ev` is negative at ante 1 (-3.28, the sticker price outweighs the flat/curated
   table for whatever voucher was on offer) but strongly positive once one is a genuine
   standout (+6.49 at ante 4) — the curated table (§1) is doing real, not degenerate, work.
7. Standard (playing-card) packs are the worst `buy_pack` row at every ante (see §5's
   documented gap) — expected, not a bug.
8. These 4 numbered antes and 38 visits are far too few to trust as "what the shop is worth" —
   they exist ONLY to prove `sweep.py` end-to-end (aggregation, JSON/MD writing,
   multiprocessing) works. The real interpretation belongs in the 126-seed run (§9).

## 8. Ad hoc validation scripts (not committed)

The 500-trial P(hit) validation numbers in §6.1 came from two throwaway scripts written to
the session scratchpad (`validate_phit.py`, `validate_pack_phit.py`) — not part of this
package (they are one-off checks, not regression tests; `tests/test_phit_validation.py` is
the permanent, CI-fast regression test covering the same logic at N=150). Available on
request if the lead wants to re-run at a larger N.

## 9. The real 126-seed sweep — exact command for the lead

```
python stats/sweep.py --out results/stats_sweep_2026-08-23.json --processes 16
```

(`--processes` up to ~28 is reasonable on this 32-core box when idle; the smoke test above
used 4 deliberately, per the 2026-08-23 resource note.) Defaults: all 126 ground-truth seeds,
`hand=greedy,reroll=1,buy=0` scripted player (brief's example spec), `--max-ante 8`. Expect
low-single-digit minutes wall time at `--processes 16` given the smoke test's 0.7s/6-seed rate
(a very rough linear extrapolation — later antes take longer per seed than the smoke run's
shallow 4-ante depth, so budget more than that naive scaling suggests).

Once run, replace §7 above with the real 126-seed table.

## 10. Open issues for W3 / W6

1. `use_consumable` kind (§5) — add to any kind-based filtering.
2. Standard-pack / Telescope-forced-planet gaps (§5) are real but low-impact; flag if V
   training data shows the shop consistently mis-valuing either.
3. `urgency`'s BOOSTER_OPEN-from-BLIND_SELECT off-by-one-blind gap (§4) — only matters for tag
   packs, a small fraction of pack opens.
4. §9's real sweep is still pending — the headline table in §7 is smoke-scale only.
