# EV_NOTES — W3: the analytic hand player and `EVPlayer` (Phase 5 rev 2, 2026-08-23)

Files: `ev/hand.py` (analytic hand EV, two budgets), `ev/player.py` (`EVPlayer`, every
state), `ev/sampling.py` (draw-world sampling), `ev/gate_ev_player.py` (gate 2, one
command), tests `ev/tests/test_hand.py` / `test_player.py` / `test_sampling.py` (47 tests),
results `results/ev_player_gate_2026-08-23.{md,json}`.  Nothing outside `ev/` and
`results/` was touched; `engine/**`, `eval/**`, `agent/**` are read only.

## 0. Headline (12-seed subset, 4 processes — the lead runs the 126; §8b fix-pass
numbers supersede the timing columns: fast hand 3.2 ms mean / 9.0 p95, shop 5 ms under load)

| | ante-1 clear | mean final ante | blinds cleared | $ at ante 3 | hands unused / cleared blind | hand ms mean / p95 |
|---|---|---|---|---|---|---|
| **EVPlayer fast** | **100%** (12/12) | 5.33 | 11.0 | 23.9 | 2.17 | **4.4 / 12.3** |
| EVPlayer full | 100% (12/12) | 5.25 | 10.8 | 24.4 | 2.30 | 73.6 / 180.7 |
| scripted greedy | 50% (6/12) | 1.50 | 2.3 | — | 1.36 | 2.0 / 2.7 |

Three more 12-seed slices during development (seeds 24–35, 36–47, 60–71, 96–107): fast
ante-1 clear 100%, 92%, 92%, 92%; greedy 25–50%.  Draw-order invariance 72/72 states per
budget; one `MLBMatch` EV-vs-EV finishes (ante 7, 4 Nemeses, 7 s).  Full gate:

    python ev/gate_ev_player.py --procs 16        # 126 seeds x 3 players, ~4 min on 16 cores

## 1. The fast budget — what is exact, what is approximated

A `SELECTING_HAND` decision is `argmax_a EV(a)` over a *structural* candidate set, where
`EV(a)` = value of the position the action leaves, horizon = end of the blind.

**Objective.** Regular blind: `P(clear) + beta·E[hands unused] + gamma·E[discards unused]`,
`beta = 0.012` (a hand unused is $1; in P(clear) units that is the exchange rate the player
accepts between risk and banking), `gamma = 0.002` (a tie-break: keep resources).  Nemesis
blind: `0.5·P(score ≥ opp) + 0.5·P(score > opp)` — a tie loses nobody a life (MLBMatch
`_resolve_pvp`), so it sits halfway.  The objective is what `value_fn=None` means; with V the
full budget values the end-of-blind state with V instead.

**Candidates** (`HandAnalysis._scoring_sets`, `_discard_lines`).  Plays: every n-of-a-kind,
two pair, full house (per rank pair), flush (top-5 and bottom-5 of each 5+ suit), straight
flush, straight (one card per rank), every visible single; each non-5-card set also with the
least valuable other cards dumped alongside (1 junk and up to 5 cards) — the "cycle cards
through a play" line; plus a pure 5-card dump and a 1-card dump.  Discards: keep a 3+ suit,
keep a straight window with ≤ 2 missing ranks, keep each rank group (and all groups), keep the
best made play's cards, junk out the k worst cards for k = 1..5.  Junk order = a keep-value
(rank count, suit count, straight adjacency, chips, enhancements).  Everything is filtered
through `game.legal_actions()` (Psychic's 5-card rule, `discards_left == 0`).  Typical
state: ~15–25 plays + ~10–14 discards vs 436 legal actions.  Face-down cards (House /
Wheel / Mark / Fish) are never read: they are junk with unknown value.

**Play scores.**  `cheap(S) = (base_chips(T) + Σ chips of scoring cards)·base_mult(T)` at the
run's planet level (W0's formula, `evaluate_hand` decides T and the scoring cards).  The top
`exact_top = 8` by cheap, plus the best candidate of every hand type, are re-scored through
the engine's `HypotheticalScorer(model_held=True)` — jokers, editions, enhancements, held
and full-deck effects, on a private RNG clone.  Skipped entirely on a plain board (no
jokers, no Plasma, plain cards: provably equal, W0's test).  `ratio` = median exact/cheap
over the top refined plays is the **board multiplier** used to price everything that cannot
be dry-run (draw targets, the tail).  Boss post-processing: Flint halves, Eye / Mouth make a
forbidden type worth 0.  The Hook (since engine `8d3f0d8`, prompted by W3's finding)
discards 2 random UNPLAYED cards after a play, so the play scores in full; the
perturbation of the kept cards is not modelled (second order; the freshness mixture
absorbs most of it).

**Draw targets** (`targets(keep, m)`).  For a kept set `K` and `m` fresh cards from the draw
pile's real composition `D` (counts by rank and suit, wilds in every suit, stones in
neither): Flush per held suit, Straight per window with ≤ 2 missing ranks (exact
inclusion–exclusion for "≥ 1 of each missing rank"), Pair / Trips / Quads per held rank,
Two Pair (pair + any single's rank; two singles; a brand-new drawn pair), Full House (trips
+ single, pair + pair, pair + single).  `p_T` is the **hypergeometric** tail (or the
multi-group inclusion–exclusion), `v_T` = cheap score with the drawn cards at the deck's
mean chips (rank chips when the rank is known) × `ratio`.  These are the only probabilities
the decision needs and they are exact for the single-draw event they describe; what is
approximate is treating the draw as "the best single target completes or it does not".

**The position value** `V(K, m, h, d, need)` (`_value_for_need`):

```
floor     = best made play inside K                         (exact units)
spec      = G(need - floor, h-1, d)                           play the floor next, then the tail
          ∨ max_T [ p_T·G(need - v_T, h-1, d) + (1-p_T)·miss ]  chase T; miss = floor path, or
                                                             one re-try of T if a discard is left
gen       = max_{k ≤ min(2,d)} Σ_i q^k_i · G(need - max(floor, ratio·s^k_i), h-1, d-k)
V         = spec + w·max(0, gen - spec),   w = m / hand_size
```

`gen` is the next hand modelled as a *generic fresh round* of this deck (below), lower-bounded
by the floor; the freshness weight `w` is how much of the next hand is new cards (a 5-card
play leaves 3 known cards and 5 unknown: the generic model deserves 62% of the say; a 1-card
discard 12%).  This mixture was the single most important modelling decision: with the pure
"floor + target" model a made straight was held instead of played (the 3 kept cards looked
worthless), with the generic model at the root every discard collapsed to one number.  At a
Nemesis the same function is evaluated at the need mixture of §3.

**The tail `G(need, h, d)`** (`BlindModel`): P(clear `need` chips with `h` hands and `d`
discards from a fresh position) + `beta`·E[unused hands·1clear] + `gamma`·E[unused
discards·1clear], a dynamic programme over *rounds*: a round is a hand preceded by k ∈
{0,1,2} discards, `G(need,h,d) = max_k Σ_i q^k_i · G(need - s^k_i, h-1, d-k)`, `G(≤0,h,d) =
1 + beta·h + gamma·d`, `G(>0, 0, ·) = 0`, on a geometric need grid (ratio 1.1, 1..2·10⁷,
linear interpolation in log-need) for `h ≤ base_hands (+3 with Burglar)`, `d ≤ base_discards
(+3)`.  The per-round score distributions `Q^0, Q^1, Q^2` come from `_FreshHandSim`: 256
fresh hands of the FULL deck's composition simulated in numpy (deal `hand_size`, score the
best made type with the cheap formula; chase once — keep the longest suit if 3+, else the
paired ranks, else the best three cards, redraw; chase twice), compressed to 6 atoms with
thin top bins (20/20/20/20/13/7%) so the big-hit tail (a flush, quads) survives.  Cheap
units; the decision divides `need` by `ratio` at lookup.  Cached per (full-deck composition,
hand size, planet levels, Four Fingers, resource caps) — NOT per joker, so a shop full of
candidate purchases reuses it; rebuilt when a pack or a tarot changes the deck (≈ 6 ms: 3 ms
simulation + 2.5 ms DP).  Monotone in need, h and d (tested).

**What is exact / what is not.**  Exact: the current hand's play scores (engine dry run),
every single-draw completion probability over the real pile, the Hook expectation, the
tail DP given its atoms.  Approximations, in decreasing order of how much they matter: (1)
future hands are "generic fresh rounds" of this deck (not the actual kept cards two hands
ahead); (2) one target per draw, independence between the target and the generic part;
(3) joker effects enter draw targets and the tail only through the scalar `ratio`
(type-specific jokers — "+mult if the hand has a pair" — are right for the plays in hand,
blurred for the future); (4) the re-try after a miss redraws at the same `p`; (5) Hook /
Serpent / Cerulean Bell / Pillar effects on the KEPT cards are ignored (Serpent: the whole
position is generic).  The full budget removes (1)(2)(5) by simulation.

## 2. The full budget

`rank_hand_actions(budget="full")`: the fast ranking's top `K = 5`; `n_worlds = 3` draw
worlds from `sampling.sample_world` (deck sorted by canonical card key, then shuffled by a
`random.Random` seeded from `(seed, observable state)` — so the same worlds for every
candidate (common random numbers) and the same decision under any permutation of the live
pile); each candidate is stepped on each world and the blind is played out by the fast
policy in *lite* mode (≤ 24 plays, ≤ 6 lines, no dry runs: the root's `ratio` is reused)
until the blind ends; the end state is valued by `end_of_blind_value`: `value_fn(world)`
if given, else the §1 objective on the outcome (cleared → `1 + beta·hands_left + gamma·
discards_left`; failed → 0; Nemesis → the §3 win/tie mixture on the final score).  The
rest of the ranking keeps its fast EV shifted below the rolled-out head.  Budget: 51 ms
mean sequential / 74 ms mean with 4 processes on this box (p95 181 ms: a first decision of a
blind with 8 rollout decisions).  With `value_fn`: ROUND_EVAL is stepped past (`advance`)
so V sees the SHOP (or the next BLIND_SELECT after a skip-free flow) — a real decision
state; GAME_OVER is valued 0 without calling V; PVP_WAIT is handed to V as is.  Caveat: on a
plain `clone()` the shop V sees after `advance` is generated by the true seed (clairvoyant
contents); W2's `clone_determinized` removes that (feature-detected in `sample_world`).
Top-1 agreement fast vs full ≈ 50% on ante-1–3 states; outcomes on the subset are equal
within noise (full is 3-sample Monte-Carlo; its value is the V plug, not a better proxy).

## 3. The Nemesis objective

Opponent final score atoms (`opponent_final_atoms`): their live score `O` plus, per hand they
still have, a symmetric per-hand score `mean1·ratio` (this deck's one-discard round, OUR
board ratio — level-0 opponent modelling: same deck, same build), with spread `sd =
sqrt(opp_h·var1)·ratio`, as three atoms (0.3 @ −0.97 sd, 0.4 @ mean, 0.3 @ +0.97 sd); out
of hands → the point mass `O`.  Need to tie = `a − scored`, need to win = `a − scored + 1`;
the hand value is `Σ_a w_a·[0.5·V(need_tie) + 0.5·V(need_win)]` with `beta = gamma = 0` (no
unused-hand money at a PvP blind, and every hand is played anyway — the engine never ends a
Nemesis on chips).  Not modelled: the opponent reacting to my score, the early-end cut
(exhausted and strictly behind → immediate loss) as a reason to sequence big hands first.

> **2026-08-26, W-PVP — the first gap is now optionally closed.**  `cfg.pvp_level1` adds the
> opponent's REVEALED live score as a fourth atom (weight `pvp_live_weight`, only when I am
> strictly behind, because only then may they sit on their hands), which is the level-1
> objective for `MLBMatch(pvp_protocol="trailer_compelled")`.  **Default OFF: with
> `DEFAULT_HAND_CONFIG` everything in this section is bit-for-bit unchanged.**  The
> early-end cut is still only reached indirectly.  See `PVP_NOTES.md` §3.

## 4. `EVPlayer` — every state

`act(game)` always returns an element of `legal_actions()` (or `no_action = {"type":
"advance"}` when there are none: PVP_WAIT, GAME_OVER, readied Nemesis).  `reset()` clears
nothing that matters (no cross-decision memory).  `explain(game)` = ranked `(action, ev,
one-line reason)`.  `epsilon > 0`: uniformly random legal action with probability ε from a
stream seeded by `(seed, observable state)`; ε = 0 is a deterministic function of that pair
(two players with the same seed on the same state act identically; the per-visit reroll
count is read off `game.reroll_cost`, never stored).  Per-decision cost ≈ 4.4 ms hand /
22 ms shop (fast).

- **SELECTING_HAND**: §1/§2, plus `use_consumable`: planets and every untargeted use are
  tried on a clone (a use the engine silently no-ops — the shop-state bug the tournament
  guard exists for — is detected by the consumable count and dropped); targeted tarots are
  tried on ≤ 3 targets drawn from the best play's scoring cards; a use is ranked by the
  resulting position's best EV (+1e-6 untargeted: frees the slot; −1e-6 targeted: must
  strictly help).
- **BLIND_SELECT**: `play_blind`, except skip Small/Big (never into a Nemesis, never at
  ante 1) when the offered tag is in `PREMIUM_TAGS` (rare / uncommon / buffoon / edition /
  top-up / charm / meteor / ethereal / orbital / investment / coupon) and `P(clear the
  following blind) ≥ 0.85` from the tail with the board ratio.  A/B on two 12-seed slices:
  skip-from-ante-2 ≥ never-skip ≥ skip-from-ante-1.  With `value_fn`: argmax V over stepped
  clones.
- **ROUND_EVAL**: `advance`.
- **SHOP / BOOSTER_OPEN**: three tiers.  (1) `stats` (W4): `stats.decision_table(game)` rows
  with `.action` (dict) and `.net_ev` (objects or dicts; `.label` optional) → the max
  positive `net_ev` row that is legal, else leave / skip; a raising `decision_table` falls
  through to the rules.  (2) `value_fn`: argmax V over the stepped clones of ≤ 12 candidate
  actions; reroll / pack open / blind choices are averaged over `n_worlds` sampled worlds.
  (3) rules on the **build proxy** `P(clear next blind) + 0.30·log1p(strength) + 0.010·
  money`, `strength = mean fresh-hand score × ratio`, `money = $ + 0.8·interest` (two
  rounds of what the balance earns): a joker is bought when the proxy rises net of its
  price (so `lam_money` IS the interest rule: below $25 each $5 costs 0.01 + 0.008); packs,
  vouchers and playing cards when affordable after the interest floor `min(25, 5·ante)`
  (packs also when `P(clear next) < 0.6`: the build needs help), in the order buffoon >
  celestial > arcana > standard > spectral; planets are used immediately (here and in any
  proxy evaluation), other untargeted consumables when the engine consumes them; reroll at
  most once per visit, never below $25 after the cost, only when nothing is worth buying;
  a joker is sold only when the slots are full and a shelf joker beats the weakest owned one
  by > 0.05.  Pack picks: the proxy after each pick (planets auto-used), anything the proxy
  cannot value (a tarot, a spectral) is still taken over skipping.

## 5. Side-effect freedom, determinism, invariance

All public functions work on clones or the `HypotheticalScorer`; the tests pin
`state_signature()` and `run_state.rng.snapshot()` before/after `rank_hand_actions` (both
budgets), `hand_ev`, `sample_world`, and `EVPlayer.act` at every state of two full vanilla
runs and at Nemesis states of an `MLBMatch`.  `HandAnalysis` reads `game.deck` only through
`DeckComp` (counts); the full budget canonicalises the pile before sampling — the gate
permutes `game.deck` at 72 states per budget (6 per seed; 126 seeds → 756) and got the
identical decision every time.

**What this paragraph did NOT cover until 2026-08-26**: side-effect freedom and *in-process*
draw-order invariance say nothing about state carried *between runs* by a module-level
cache.  `board_ratio`'s memo was process-global, so a run's decisions depended on what the
worker had played before it — 8% of seeds, see §8b item 1.  The memo is now per-`EVPlayer`
and cleared by `reset()`, and the property is pinned end-to-end by
`ev/tests/test_player.py::test_a_run_is_unchanged_by_what_the_process_played_before_it`
(on a seed pair verified to diverge under the old scope).  **A run is a function of
`(seed, budget, cfg)` alone, in a cold process or a reused worker.**

## 6. Benchmarks (this box, Tagg active; AFTER the fix pass)

Sequential (4 seeds to ante 6):

| decision | mean | p50 | p95 | max |
|---|---|---|---|---|
| fast hand | 2.9 ms | 2.0 | 7.7 | 19 |
| shop (rules) | 3.9 ms | 1.8 | 15.9 | 20 |
| pack (rules) | 6.1 ms | 1.0 | 34.6 | 38 |
| blind model build (hand cfg / proxy cfg) | ~6 / ~1.5 ms | | | |

12-seed gate under 4 parallel processes: fast hand 3.2 mean / 9.0 p95; full hand 58 mean /
146 p95 (K=5, 3 worlds); shop 5–6 ms.  An `MLBMatch` EV-vs-EV now finishes in ~2 s (was
7 s).  The p95s are first-visit cache misses: a pick that changes the deck or the levels
rebuilds the (lighter) proxy blind model; a new joker set recomputes the board ratio.

## 7. The curve of attempts (12 seeds, hand player only, greedy's no-buy shop → then the player)

| step | ante-1 clear | note |
|---|---|---|
| W0-style floor + binomial targets, tail = 10 sampled hands | 58% | 1-card discards everywhere (generic tail at the root flattened every discard to one number) |
| generic tail only after the next play | 67% | plays sane; tail truncated at the sampled max (quads never exist) |
| numpy fresh-hand simulator, 256 hands, thin top atoms | 67% (money +60%) | all remaining losses at the ante-1 Boss, 600 chips, no jokers |
| + EVPlayer shop rules (joker Δproxy) | 67% | two Big-blind losses: a made straight held for a flush draw; 1-card "high card" plays tied with the straight |
| + freshness mixture in the position value | 92–100% | generalised to three other slices (92%) |
| + Hook expectation, PvP ratio fix, skip from ante 2 | 92–100% | remaining failures are Hook bosses with no scoring joker |
| fix pass (engine Hook fixed upstream; caches; lighter proxies) | 83–100% (slices 0-11 / 36-47 / 60-71: 100 / 83 / 100) | the two 36-47 failures are the scalar-ratio flaw (§8.2), not Hook |

## 8. Open issues / the next lever

1. ~~The Hook~~ RESOLVED: the lead confirmed against state_events.lua:478-488 and fixed
   the engine (`8d3f0d8`); the pick-pair expectation was removed from `_score_plays` in the
   fix pass.  The remaining ante-1 failures on the dev slices are now issue 2 below (a
   face-mult board — Photograph + Smiley — inflates the scalar ratio, so the tail thinks
   EVERY fresh hand scores 6×; seeds 8QBRTPD / 9Q9HQXZG).
2. The tail is per deck, not per build: type-specific jokers are a scalar.  Next lever = a
   per-TYPE ratio (exact/cheap of the best play of each type from the dry runs, applied to
   the simulator's per-type scores before compression) — a 30-line change in `_FreshHandSim`
   / `_score_plays`.
3. `full` is 3-sample Monte-Carlo; with V it should use more worlds and fewer candidates
   (K=3, 8 worlds fits 100 ms).  Its end-of-blind proxy ignores what the next blind needs
   (V fixes that by construction).
4. Shop proxy blind spots: scaling jokers (Ride the Bus, Obelisk…) show no immediate
   strength; tarots / spectrals are unvalued ("take over skip"); vouchers are a flat +0.02.
   W4's stats tier and V are the designed replacements; the rules only need to bootstrap.
5. PvP: the opponent is level-0 (symmetric).  The early-end cut is not exploited.
6. Nemesis-blind hand decisions evaluate 6 needs per candidate (3 atoms × tie/win): ~2× a
   regular decision.

## 8b. Fix pass (lead-directed, 2026-08-23, after all workstreams landed)

Four defects W5/W6 surfaced, fixed at the source:

1. **Shop/pack proxy hotspot** (58% of a W5 rollout): `board_ratio` is now cached by a
   board signature (jokers + scaling state, Plasma, hand size, deck-modifier counts —
   deliberately NOT planet levels or the exact composition: the exact/cheap ratio is
   level-invariant to first order), so the state a purchase produces hits the candidate's
   entry from the previous `act()`; the shop/pack proxies use a lighter `proxy_cfg`
   (48-sample simulator, 4 atoms, 1.20 grid) and a 3-hand ratio.  Shop 22 → 3.9 ms, pack
   27 → 6.1 ms, MLB match 7 → 2 s.

   **Amended 2026-08-26 (W-FIX): the cache is per-PLAYER, not process-global.**  The key
   above is unchanged and the speed argument still holds; what was wrong was the *scope*.
   Because the key omits planet levels and the exact deck composition while `board_ratio`
   samples real hands from the real deck at the run's real levels, two states that differ
   only in an omitted field share an entry and whichever was computed first wins.  Inside
   one run that is the intended approximation and is deterministic given the seed.  Across
   runs sharing a worker process it was a determinism leak: W-ENCODE-POC measured **2 of 24
   seeds (8%)** changing trajectory with the worker partition (POC_NOTES §3.5), which made
   every per-seed row of any pooled harness — `gate_ev_player.py` included — depend on what
   the worker had played before it.

   `board_ratio(game, n_hands, cfg, *, cache=...)` now memoises into a caller-supplied
   dict.  `EVPlayer` owns `self._ratio_cache`, clears it in `reset()`, and passes it at
   every `build_proxy` call site; `hand._RATIO_CACHE` remains only as the fallback for
   module-level callers and no player writes it.  Widening the key was the alternative and
   would have paid back the 40% pack cost — and would have bought nothing, because the
   cross-run sharing was worth **one cache hit in 3516 lookups**: over 12 `ev:fast` seeds
   the hit rate is 76.2% per-player vs 76.2% shared (2678 vs 2679), against 0% and a 2.6×
   slowdown with no cache at all.  All the value is within-run reuse, which the per-player
   dict keeps.  Cost of the change, 12 seeds single process: shop 4.41 → 4.56 ms, pack
   6.63 → 6.76 ms, hand 3.33 → 3.38 ms (+2-3%).  `workers=1 == 4 == 6` now holds without
   `ev/encode/verify.py::reset_player_caches()`.  Full write-up and the other three fixes
   of that round: `engine/FIX_NOTES.md`.
2. **ε-wedge**: ε draws now come from a private sequential `random.Random("ev-eps:<seed>")`
   advanced once per `act()` and re-seeded by `reset()` — an ε-pick that no-ops (Wheel of
   Fortune whiff) no longer freezes the stream on an unchanged observable state (W5 saw
   39,952 identical shop steps).  With ε > 0 the player is deterministic given `(seed, call
   history)`, not `(seed, state)`; ε = 0 unchanged.
3. **`value_fn` exceptions propagate** from `EVPlayer._v` and `hand.end_of_blind_value`
   (they silently degraded to the analytic proxy, hiding V bugs).  A broken W4 `stats`
   object still falls through to the rules — that one is the documented contract.
4. **Anti-cycling guard**: `act()` counts identical `state_signature()`s per SHOP /
   BOOSTER visit (`Counter`, reset by `reset()`); the 3rd sight of an unchanged signature
   forces the rules tier (which verifies consumable uses on a clone and cannot no-op), the
   6th forces leave_shop / skip_booster.  Breaks W5's 40k-step V-preferred-no-op loop in
   ≤ ~8 steps; a normal visit never reaches 3.

Regression tests: `test_player.py` (fix-pass block: ε non-repetition + replay-on-reset, V
exception from `act`, the Wheel-whiff V loop broken ≤ 12 steps, guard-inertness on a normal
visit), `test_hand.py` (`test_full_budget_value_fn_exception_propagates`,
`test_board_ratio_is_cached_by_board_signature`, `test_hook_leaves_play_scores_intact`).
Suite: 122 passed (mine + W5/W6's untouched).

## 9. Wiring for the lead

- `value_fn: Callable[[BalatroGame], float]` → `EVPlayer(value_fn=V)`: used by the full
  budget at end-of-blind states and by the shop / pack / blind-select tiers (argmax V over
  clones, chance actions averaged over `n_worlds`).  `budget="full"` to make hand
  decisions use it too.
- `stats` → `EVPlayer(stats=obj)` with `obj.decision_table(game) -> list[Row]`, `Row.action`
  / `Row.net_ev` (`.label` optional).  Takes precedence over `value_fn` in SHOP / BOOSTER.
- `clone_determinized(seed)` (W2) is picked up automatically by `sampling.sample_world`.
- W5 rollouts: `EVPlayer(budget="fast", epsilon=ε, seed=worker_seed)`; ~4.4 ms per hand
  decision, ~22 ms per shop decision, a vanilla run to ante 5 ≈ 1.2 s, an MLB match ≈ 7 s.
- W6 advisor: `EVPlayer.explain(game)`; `hand.rank_hand_actions(game, budget=...)` and
  `hand.hand_ev(game, action, ...)` for a single action.
