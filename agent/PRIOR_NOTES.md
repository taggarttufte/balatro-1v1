# PRIOR_NOTES — W0: the heuristic hand prior + candidate pruning

**Agent W0, 2026-08-22.** Deliverable: a `SELECTING_HAND` prior that makes the vanilla
warm-up produce a *learnable value target*, and the search-side candidate mask that goes
with it.

New: `agent/mcts/heuristic.py`, `agent/tests/test_heuristic_prior.py` (38 tests),
`agent/scripts/w0_smoke_report.py`, `agent/runs/w0_smoke.sh` + `w0_smoke2.sh`, this file.
Edited: `mcts/search.py` (`MCTSConfig` gains five inert-by-default fields; `MCTS` gains
`heuristic_lambda` + `_shape_priors`, called from the one existing expansion seam),
`mcts/player.py` (`MCTSPlayer.heuristic_prior` / `set_heuristic_prior`; `make_player`
kwargs), `mcts/__init__.py` (exports), `train/loop.py` (`TrainConfig` knobs, the anneal,
the clear-rate EMA, the checkpoint payload, two new log fields), `train/population.py`
(`instantiate` passes the prior to every net seat), `train/selfplay.py` (`MLBTrainer`
anneals it per generation), `scripts/train_cold.py` + `scripts/train_mlb.py` (flags).
**`engine/**`, `rng/**`, `tournament/**`, `eval/**`, `replay/**`
untouched.**

---

## 0. Headline

Stage A of the first real run played **4 350 episodes and cleared ante 1 forty-six times
(1.0%)**, mean ante 1.00, **value loss 0.0008** — a constant target, so nothing learned.
A 200-sim diagnostic cleared **0 of 53**, which ruled out the simulation budget. The
cause is the prior: a cold net is uniform over ~436 legal actions at `SELECTING_HAND`,
~218 `play` subsets and ~218 `discard` subsets, and the lines that clear a 300-chip blind
are a vanishing fraction of that.

Ten minutes of the same command with the prior on:

| | Stage A as launched (4 350 ep) | matched 10-min cold baseline | **W0 arm F** |
|---|---|---|---|
| ante-1 clear rate | 1.0% | **0.0%** (0 / 499) | **15.7%** |
| mean final ante | 1.00 | 1.00 | **1.18** |
| mean episode length (decisions) | 9.7 | 9.3 | **19.1** |
| **value loss** | **0.0008** | 0.0023 | **0.0251** (11x) |

Arm F is `--heuristic-prior 0.8 --heuristic-tau 0.35 --max-hand-candidates 32`. **The
brief's 20% bar was not reached** — nine arms over two 10-minute rounds all land at
14-16% at best, and §4 says what was tried. What WAS reached is the thing the bar was a
proxy for: the value target is no longer constant, so Stage A can learn.

## 1. The formulas, in five lines

Let `flags = game.hand_eval_flags()`, and `base(T) = HAND_BASE[T]` bumped to the run's
planet level. Everything is computed on the LIVE game and touches nothing
(`state_signature()` is bit-identical before and after; three tests pin it).

1. **`cheap(S) = (base_chips(T) + sum of chips of the scoring cards) * base_mult(T)`**,
   where `(T, scoring) = evaluate_hand(S, **flags)` — the engine's own evaluator — and a
   card's chips are `base_chips + bonus_chips` (0 if debuffed).
2. **`play(S) = HypotheticalScorer(game, model_held=True).score(S, T, scoring)`** for the
   top `--heuristic-exact-top` (8) subsets by `cheap`, else `cheap(S)`. That is the real
   `score_hand` with jokers, editions, enhancements, held-in-hand and full-deck effects,
   on a private RNG clone. It is skipped entirely when the board is *plain* (no jokers,
   no Plasma, every card without enhancement/edition/seal), where the two are provably
   equal — a test asserts the equality rather than trusting the comment.
3. **`discard(D) = discard_bias * max(floor(D), draw(D))`** with `K` = the kept cards and
   `d = |D|`: `floor(D) = max{ cheap(S) : S ⊆ K }` (exact max-over-submasks DP over the
   already-scored play candidates) and
   `draw(D) = max_T value(T) * P(X >= m_T)`, `X ~ Binom(d, p_T)`, over draw targets
   T ∈ {Flush per held suit, Straight per 5-rank window, Pair/Three/Four per held rank};
   `m_T` = cards still needed, `p_T` = useful cards left in `game.deck` / deck remaining,
   `value(T) = (base_chips(T) + n_scoring(T) * avg_deck_chips) * base_mult(T)`.
4. **The prior.** With H = play ∪ discard, O = everything else, `W_H = Σ_{a∈H} net(a)`:
   `h(a) = W_H * softmax_H(log1p(score(a)) / tau)` on H, `h(a) = net(a)` on O, and
   `prior = (1 - lambda) * net + lambda * h`. So **lambda only ever moves mass between
   hand actions** — the net keeps its say on play-vs-Tarot, and every non-`SELECTING_HAND`
   state is returned untouched (the same dict object, in fact).
5. **The mask.** `--max-hand-candidates K` expands only the top-K `play` and top-K
   `discard` by the same score, plus all of O, renormalised. Search-side only: pruned
   actions stay legal in the engine, still appear in `legal_actions()`, and still appear
   with visit count 0 in the `Sample`.

`tau` is on `log1p(score)` and is therefore scale-free: `tau = 1` is `prior ∝ score`,
`tau = 0.5` is `prior ∝ score²`, `tau → 0` is argmax. Raw scores span 10 (a lone Two) to
10⁴⁺ (a levelled Flush behind jokers), which no additive-temperature softmax survives.

**Why play and discard are comparable.** `floor(D)` is a `cheap(S)`; `value(T)` is the
same formula with the unknown drawn cards at their mean; `play(S)` is a `cheap(S)`
possibly refined upward. One softmax over both is therefore meaningful: *discard wins
exactly when the position it leaves is worth more than the best hand on the table*. On
the smoke's first fixture hand (`QH 9S 6D 5C 2C AS 4C 8C` — four clubs, nothing made) the
top action is `discard (0,1,2,5)`, the four non-clubs, at 172.5 against the best play's
16.0. That is the decision the 1%-clearing run could not find.

## 2. Flags

Both `train_cold.py` and `train_mlb.py`:

| flag | default | what it does |
|---|---|---|
| `--heuristic-prior` | `0.0` | starting lambda. `0` = the pre-W0 search exactly |
| `--heuristic-prior-floor` | `0.1` | lambda never anneals below this |
| `--heuristic-prior-anneal` | `""` | `""` constant / `"<N>"` or `"ep:<N>"` linear to the floor over N episodes / `"clear:<r>"` decay as the rolling blind-clear rate approaches r |
| `--heuristic-tau` | `0.5` | softmax temperature on `log1p(score)` |
| `--heuristic-exact-top` | `8` | dry-run refinements per leaf (auto-skipped on a plain board) |
| `--heuristic-discard-bias` | `1.0` | multiplier on every discard potential |
| `--max-hand-candidates` | **`32`** | top-K play + top-K discard expanded per leaf; `0` = the pre-W0 tree |

`--max-hand-candidates` defaults to **32 at the CLI** (the brief's number) but to **0 in
`MCTSConfig` / `TrainConfig` / `make_player`**, so every library caller and every existing
test gets the old behaviour and only a training command opts in.

**The anneal is checkpointed.** `ColdTrainer.state_dict()["heuristic"]` carries
`{lambda, clear_rate_ema}`; `from_checkpoint` restores both and pushes lambda back onto
the search. A Phase 4 checkpoint that predates W0 reads as `lambda = 0` rather than
crashing (tested). A `--resume` may CHANGE any heuristic flag — they are run-shaping, not
experiment-defining, so `_check_config` does not pin them — but only if the flag is
actually on the command line; otherwise the checkpoint's annealed value wins.

## 3. Where it plugs in

One seam: `MCTS._apply_expansion` calls `MCTS._shape_priors(game, priors)` before the
children are created. Every expansion goes through it — the serial `run`/`run_gumbel`,
the virtual-loss `leaf_batch > 1` path, and `BatchedSearch`'s cross-tree lockstep — so
there is exactly one place to read and no path that silently misses the prior.

`MCTSConfig` carries the SHAPE (`heuristic_tau`, `heuristic_exact_top`,
`heuristic_discard_bias`, `max_hand_candidates`, and a default
`heuristic_prior_weight`); the WEIGHT that anneals lives on the `MCTS` object as
`heuristic_lambda`, and `MCTSPlayer.set_heuristic_prior(lam)` moves it. That split exists
because a tournament generation builds N players off ONE shared `MCTSConfig` — mutating
the config to anneal would move every seat's lambda at once, including the frozen past
selves.

**Stage B (the tournament) runs the prior on both sides.** `population.instantiate` gives
every net seat — current and past-checkpoint alike — the same lambda and the same mask.
It has to: a past checkpoint searching without the prior is a different agent from the
one whose weights were trained with it, and the population-rank target would be measuring
the prior instead of the net. Scripted anchors are unaffected (they never search).
`MLBTrainer.anneal_heuristic()` runs once per generation off the SAME `clear_rate_ema`
that drives `--skip-cap-anneal-clear-rate`, so the two training-time crutches come off
together, and `h_lambda` is logged next to `skip_cap` in the generation line.

`eval`'s `checkpoint:<path>,k=v` spec needs no change: it forwards unknown `k=v` pairs
straight to `make_player`, so
`checkpoint:runs/real1_stageA/latest.pt,sims=40,heuristic_prior=0.1,max_hand_candidates=32`
already works (the string `"0.1"` is coerced by `MCTS.heuristic_weight`).

## 4. The 10-minute gate smoke

`bash agent/runs/w0_smoke.sh 10` (arms A-E) and `w0_smoke2.sh 10` (F-I) — all
`--ruleset vanilla --encoder set --device cpu`, run in parallel with 4 torch threads each
on the 32-core box. `python agent/scripts/w0_smoke_report.py` prints the table.

| arm | what | eps | ep/min | **clear%** | mean ante | mean len | **v-loss** |
|---|---|---|---|---|---|---|---|
| A `w0_A_cold` | baseline: no prior, no mask | 499 | 49.9 | **0.0%** | 1.00 | 9.3 | 0.0023 |
| B `w0_B_maskonly` | λ 0, K 32 | 256 | 25.6 | 0.4% | 1.00 | 8.9 | 0.0072 |
| C `w0_C_prior` | λ 0.8, K 32, τ 0.5 | 135 | 13.5 | 14.1% | 1.16 | 17.3 | 0.0266 |
| D `w0_D_full` | λ 1.0, K 32, τ 0.5 | 133 | 13.3 | 9.8% | 1.10 | 16.5 | 0.0180 |
| E `w0_E_tau1` | λ 0.8, K 32, **τ 1.0** | 169 | 16.8 | 5.9% | 1.06 | 12.7 | 0.0139 |
| **F `w0_F_tau035`** | **λ 0.8, K 32, τ 0.35** | 127 | 12.7 | **15.7%** | **1.18** | 19.1 | 0.0251 |
| G `w0_G_sims80` | λ 0.8, K 32, τ 0.5, **80 sims** | 72 | 7.2 | 6.9% | 1.07 | 17.9 | 0.0196 |
| H `w0_H_disc15` | λ 0.8, K 32, τ 0.5, **discard bias 1.5** | 143 | 14.3 | 9.1% | 1.09 | 16.6 | 0.0191 |
| I `w0_I_tight` | λ 0.9, **K 16**, τ 0.35 | 123 | 12.3 | 14.6% | 1.15 | 20.8 | 0.0229 |

`clear%` = the episode reached ante 2, i.e. cleared all three ante-1 blinds. At n ≈ 130
its standard error is ~3 points, so **F, C and I are one result and everything else is
below them.** `ep/min` is summed in-episode wall clock, so it is comparable between arms
even though they ran in parallel; it is not the throughput a solo run gets.

**Read it in this order.**

* **Arm A is the thing being replaced**, and it reproduces the real run's pathology
  exactly: 0 clears in 499 episodes, mean ante 1.00, value loss 0.0023 and falling.
* **Arm F is the recommendation.** 15.7% of episodes clear ante 1, mean ante moves off
  the floor, episodes are twice as long, and the value loss is **11x the baseline** —
  which is the actual deliverable. A higher value loss here is not "worse": it is
  *non-degenerate*, the target finally varies between episodes.
* **Arm B (mask, no prior) is the control that matters, and it is the strongest single
  finding.** The mask ALONE does **nothing** — 0.4% against the baseline's 0.0%. Pruning
  to the 64 best-by-heuristic actions while leaving the net's near-uniform prior over them
  does not find the flush. **The prior is the mechanism; the mask is only a tree-size
  lever.** Anyone tempted to ship the cheap half should not.
* **Arm D (λ = 1.0) is worse than 0.8**, twice measured. Dropping the net entirely drops
  every source of spread the root had except Gumbel's own sampling; the heuristic is
  greedy about hand value and blind about everything else. Keep 20% of the net.
* **τ matters more than λ.** τ 1.0 (`prior ∝ score`) is 5.9%, τ 0.5 is 14.1%, τ 0.35 is
  15.7%. At τ = 1 the best action gets a few percent of the mass against 218 alternatives
  and Gumbel's top-16 rarely samples it. Below 0.35 was not tried and is the obvious next
  sweep point.
* **More sims does not help** (arm G, 80 sims: 6.9%). Consistent with the lead's 200-sim
  diagnostic: this was never a budget problem, and doubling the budget just halves the
  episodes the net learns from.
* **A discard bias above 1 hurts** (arm H, 9.1%). The `max(floor, draw)` formula already
  discards when it should; pushing it further trades hands for draws that do not arrive.

One metric to distrust: the report's `blinds` column is `blinds_completed(game)`, the
blind INDEX reached — **a skipped blind advances it too**. That is why the cold arm shows
1.52 "blinds" with zero clears (it skips Small and Big and dies at the Boss) while arm B
shows 0.89. Use `clear%`, not `blinds`, to judge an arm.

**Throughput, stated honestly.** The heuristic costs **~1.9 ms per `SELECTING_HAND`
leaf** (218 `evaluate_hand` ≈ 1.35 ms, the discard machinery ≈ 0.5 ms; the exact tier
adds ~0.4 ms when the board is not plain). Measured on one ante-1 decision at 40 sims,
set encoder, CPU: **351 sims/s baseline → 184 sims/s with lambda 0.8 + K 32**, i.e. 1.9x
slower *per simulation*. The mask does pay for part of itself (168 sims/s with the prior
and no mask → 184 with it, +9%; K 16 gives 189) but **it does not make the search faster overall**, and
the brief's hope that it would is not what the numbers say. The reason is structural: the
mask is applied at `_apply_expansion`, i.e. AFTER the policy has already featurised all
436 actions, so it saves `add_child` / `priors_from_logits` / the tree, not
`featurize_actions_set`. Making it save the featurisation means pushing the allowed set
into `NNPolicy.encode_leaf`, which changes the `PolicyValueFn` contract — see §6.

Episodes per minute drop further than sims/s does (44 → 13 in the smoke) and that is a
GOOD sign, not a regression: an arm-C episode is 17.4 decisions instead of 8.9 because
the agent survives further. Samples per minute are roughly flat.

## 5. What to set

### Stage A — vanilla warm-up (replaces TRAIN_NOTES §8's Stage A)

```
python agent/scripts/train_cold.py \
    --minutes 300 --device cpu \
    --ruleset vanilla --encoder set \
    --sims 40 --max-decisions 1500 \
    --heuristic-prior 0.8 --heuristic-tau 0.35 \
    --max-hand-candidates 32 \
    --heuristic-prior-anneal clear:0.6 --heuristic-prior-floor 0.1 \
    --batch-size 32 --lr 1e-3 --buffer-capacity 20000 \
    --checkpoint-every 200 --keep-checkpoints 6 \
    --run-dir agent/runs --run-name real1_stageA
```

`clear:0.6` holds lambda at 0.8 until the net starts clearing ante 1 on its own and walks
it to the 0.1 floor as the rolling clear rate approaches 60%. The floor is deliberate:
hand-value knowledge is not something the net should have to *un*learn, and 0.1 is small
enough that a trained policy overrides it freely.

**Watch:** `clear%` and `lam` in the status line — both are new. `clear%` starts around
15 and should be climbing within the first hour. **`v=` (value loss) is the alarm**: it
must stay above ~0.01. If it collapses back toward 0.001 the target has gone constant
again and the run is the old failure with extra steps — stop and say so rather than
spending five hours on it. `blinds` is logged too but counts the blind INDEX, which a
SKIP advances, so it is not a clear-rate proxy (see §4).
**Move on when** mean ante is climbing off 1.0 AND `clear%` is over ~40 — at which point
`clear:0.6` has already walked lambda most of the way to the floor. **Abort rule
unchanged:** if mean ante is still 1.00 after the full Stage A, do not start Stage B.

Throughput note: the ~13 ep/min in the table was measured with five processes sharing the
box at 4 torch threads each; a solo run gets more. Budget Stage A on decisions rather than
episodes — a *good* episode is twice as long as a bad one, which is the point.

### Stage B — the tournament run

Add to TRAIN_NOTES §8's Stage B command:

```
    --heuristic-prior 0.4 --heuristic-tau 0.35 --max-hand-candidates 32 \
    --heuristic-prior-anneal clear:0.5 --heuristic-prior-floor 0.1 \
```

**0.4, not 0.8**: Stage B starts from a net that already knows how to play a hand, and the
whole point of the tournament objective is to rank policies against each other — a large
shared lambda makes every seat play the same hands and pushes `tie_fraction` up, which is
the degeneracy detector W2 built. `clear:0.5` is the same threshold as
`--skip-cap-anneal-clear-rate 0.5`, so the skip cap and the hand prior are removed by the
same event and the final policy is trained under the real rules with lambda at the floor.
Watch `h_lambda` next to `skip_cap` in the generation line; if `tie_fraction` climbs over
0.2, lower `--heuristic-prior` before touching anything else.

## 6. Found, not fixed

* **The mask does not save the featurisation.** As above: pruning happens after the net
  has featurised every legal action, so the ~50 ms/decision that
  `featurize_actions_set` costs at a `SELECTING_HAND` leaf is still paid in full. The fix
  is a `PolicyValueFn`-level action filter (`NNPolicy.encode_leaf` /
  `SetNNPolicy.encode_leaf` / `BatchedNNPolicy` taking an allowed-action list), plus a
  cache so the heuristic is computed once per leaf rather than once for the filter and
  once for the prior. That is a W1-owned contract change and would roughly halve the
  per-leaf cost; it is the single biggest remaining throughput lever and it is NOT done.
* **`evaluate_hand` on all 218 subsets is 70% of the heuristic's cost.** A vectorised
  bitmask re-implementation would be ~10x faster, and was deliberately NOT written: a
  second hand evaluator that can drift from the engine's is exactly the class of bug this
  project has spent two phases removing. If it is ever wanted, it needs a test that pins
  it against `evaluate_hand` over every subset of a few hundred sampled hands.
* **The draw potential ignores Shortcut and Smeared.** `four_fingers` is honoured (the
  flush/straight completion count drops to 4); Shortcut's gap-tolerant straights and
  Smeared's suit pairing are not modelled, so the potential UNDERSTATES a hand behind
  those jokers. `evaluate_hand` sees all three flags, so the *play* side is exact either
  way. Both jokers are rare enough that this was not worth the complexity at ante 1.
* **The completion probability is binomial, not hypergeometric** — sampling with
  replacement from the remaining deck. At `d <= 5` draws from a ~40-card deck the error is
  a few percent and always in the direction of understating a draw. Stated in the module
  docstring; a hypergeometric would be one more table.
* **No blind-target awareness.** The prior does not know that a play which would *finish*
  the blind is worth more than a bigger one that would not, nor that `hands_left == 1`
  changes everything. Both are things the search and the value head are supposed to
  learn, and adding a "clears the blind" bonus would be the first thing to try if a longer
  Stage A plateaus below ~50% clear.
* **The 20% bar was not reached and one lever is untried.** The prior does not know
  which play would *finish* the blind. Ante 1 is three blinds (300 / 450 / 600) and the
  arms die around the second, so a bonus of the form "multiply the score of any play that
  takes `chips_scored` past `chips_target`" is the first thing to add if a real Stage A
  plateaus below ~40% clear. It was deliberately left out of v1: it is an ad-hoc term
  bolted onto an otherwise principled score, and the search plus the value head are
  supposed to learn exactly that. Sweep `tau` below 0.35 first — it is the knob that
  moved the number most (5.9% -> 14.1% -> 15.7% at 1.0 / 0.5 / 0.35).
* **The smoke arms were 10 minutes and ~130 episodes each**, so 14-16% has a ~3-point
  standard error and the F/C/I ordering is not established. What IS established is the
  gap to the 0.0%/499-episode baseline.
* **Ties are everywhere and the tie-break is arbitrary.** Every play subset whose scoring
  cards are the same single Ace scores identically, so the top-K keeps an arbitrary member
  of a large tied group (sorted by `(-score, indices)`). Harmless for the prior; worth
  knowing if someone reads the surviving candidate list and expects the "natural" subset.
* **`numpy` has no popcount before 2.0 and the obvious DP is wrong.** The vectorised
  `pc[1:] = pc[idx >> 1] + (idx & 1)` reads from the all-zero array instead of
  accumulating, so it silently computes `i & 1`. It cost an hour and a wrong discard
  ranking; `_popcount_table` now does `n` explicit passes and says so.

## 7. Gates

```
python -m pytest agent/tests -q                                    # 309 (271 + 38)
python -m pytest tournament/tests eval/tests replay/tests -q # 264 (57/125/82)
```

`engine/**`, `rng/**`, `tournament/**`, `eval/**`, `replay/**` were not
edited. No engine change is needed.
