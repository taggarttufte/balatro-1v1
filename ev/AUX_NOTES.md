# AUX_NOTES — W-AUX: auxiliary prediction heads on rollout intermediates

Phase 5 rev 2 (V-v2), brief `docs/PHASE5_V2_BRIEF_2026-08.md` §6b. Built 2026-08-25, after
W-PAIRS and W-RANK landed. **Nothing was committed.**

Files owned and changed:

| file | what |
|---|---|
| `mp/ev/aux_targets.py` | **new** — the spec table, the rollout observer, aggregation/masking |
| `mp/ev/labels.py` | additive: `rollout(observer_factory=)`, `RolloutResult.aux`, `label_both` aggregation, `label_job` payload `aux` |
| `mp/ev/pairs.py` | additive: `roll_pair(observer_factory=)`, `PairRollout.aux_a/aux_b`, `pair_record(aux=)`, `pair_job` payload `aux` |
| `mp/agent/mcts/value_net.py` | additive: `ValueNetConfig.aux_heads/aux_hidden`, `SetValueNet.aux_heads`, `forward_with_aux`, `aux_head_names` |
| `mp/ev/train_v.py` | additive: aux config, target extraction + masking + standardisation, multi-task loss, per-head metrics, tolerant net/optimizer load |
| `mp/ev/scripts/gen_labels.py`, `gen_pairs.py` | one `--aux` flag + aux coverage in the results JSON |
| `mp/ev/tests/test_aux.py` | **new**, 31 tests, ~8 s |
| `mp/ev/AUX_NOTES.md` | this file |

Outputs written (all new files):

| path | what |
|---|---|
| `mp/ev/runs/aux_g1/` | 576 aux-recorded labels, 6 shards |
| `mp/ev/runs/aux_pairs_g1/` | 238 aux-recorded pairs + 476 absolute rows |
| `mp/results/labels_aux_g1.json` | label-campaign summary + per-head coverage |
| `mp/results/pairs_aux_pairs_g1.json` | pair-campaign summary + per-branch coverage |
| `mp/results/aux_ablation.json` | the §6.2 gate: 9 runs, 3 selection rules, per-head metrics, and the two supporting 12-run ablations |

Not touched: `hand.py` (W-EXTRACT), `fixtures/` (W-PROBE), `dataset.py`, `player.py`,
`workers.py`, `match_player.py`, `race.py`, `mp/engine/`, and everything else in
`mp/results/`.

---

## 0. Headline

* **8 heads, all computable.** Nothing was dropped as uncomputable; two were re-scoped and
  one was narrowed to a binary — §2 lists every deviation. Measured coverage on the fresh
  campaign is **82–100 %** per head.
* **The instrumentation is free**: **0.37 ms per rollout** against ~1,050 ms of rollout, i.e.
  **0.04 %**. Matched-trajectory A/B (18/18 identical decision counts and outcomes) puts the
  end-to-end overhead inside measurement noise (−0.6 %).
* **The no-aux path is bit-identical to the pre-W-AUX trainer** — verified by swapping the
  pre-W-AUX `train_v.py` + `value_net.py` back in and diffing: 6/6 and 52/52 weight tensors,
  18/18 and 156/156 Adam moment tensors, identical eval histories, on both the dummy net and
  the real `SetValueNet` (§6.3).
* **The keeper checkpoint still loads and plays** (`runs/v_full_best/ckpt_0001000.pt`,
  4,996,789 params, no aux heads) and a checkpoint saved WITH heads still loads for play.
* **The ablation is a null at proof scale, with the per-head numbers as the clear positive.**
  Held-out Brier **0.0876 (aux) vs 0.0877 (no aux)** — a tie inside a ±0.007 seed spread; BCE
  and ECE favour aux (0.6935/0.086 vs 0.7028/0.105), AUC and pair accuracy favour no-aux
  (0.666/0.552 vs 0.685/0.598 on **29 resolved pairs**, CIs overlapping almost completely).
  **The money head reaches R² +0.54** and four more heads carry real held-out signal (§6.2).
  The honest read: the plumbing is demonstrated and free; whether aux helps V cannot be
  answered on 3,654 rows and has to be re-asked on the full campaign (§8.1).

---

## 1. What is recorded, and where

An **observer** is threaded into the rollout loop the workers already run
(`labels.rollout(..., observer_factory=...)`); `pairs.roll_pair` passes the same factory to
BOTH branches. One fresh `aux_targets.AuxRecorder` per rollout; it reads public attributes
of the two `BalatroGame`s and of `MLBMatch.pvp_detail` after each step and never steps,
clones or scores anything. `observer_factory=None` (the default everywhere) is a single
`is not None` test per step — that is what keeps the no-aux path bit-identical.

Both players are recorded (a label job needs both perspectives; a pair keeps the actor's).
Per-rollout results are averaged over the shared worlds by `aux_targets.aggregate` — the
brief's "mean over shared worlds". A field no world produced is `None` (strict JSON `null`,
never NaN, per PAIRS_NOTES §3.3) and the trainer masks it.

**Schema — additive, nothing frozen was renamed** (brief §6b.3):

```
absolute row   meta["aux"]        = {name: value | null}          + meta["aux_version"]
pair record    rec["aux"]         = {"a": {...}, "b": {...}, "version": 1}
```

`rec["aux"]` rides inside the frozen record's JSON blob (`pair_json`), so **no on-disk
layout changed at all**: `pairs.load_pair_shard`, `train_v.PairDataset` and
`dataset.LabelDataset` all read the new shards with zero changes, and the old 51k corpus /
`pairs_s1` shards load exactly as before and simply have no `aux` key. The absolute rows a
pair also yields carry that branch's aux, so the pair-derived label rows are aux-bearing
too (verified: `test_pair_job_records_both_branches`).

---

## 2. The eight targets (`aux_targets.AUX_SPECS`)

All are recorded from the trajectory the worker already walks; none costs an extra
simulation. Values are stored RAW and interpretable; the trainer applies each spec's
transform at load, so a shard can always be audited in game units.

| head | dim | loss | raw value | transform |
|---|---|---|---|---|
| `money_next_shop` | 1 | MSE | dollars at the first transition INTO a shop after the state | `log1p(x)/log1p(200)` |
| `lives_2antes` | 2 | MSE | `[own lives, opp lives]` when the player's ante first hits `ante0+2` (terminal lives if the match ends first) | `x/4` |
| `pvp_margin_next` | 1 | MSE | `log10((1+my score)/(1+their score))` at the next resolved Nemesis, clipped ±6 | `x/6` |
| `blind_cleared` | 1 | BCE | 1 if the blind in progress (or the next one actually played) resolves without this player losing a life | identity |
| `xmult_by_ante4` | 1 | BCE | 1 if an xMult joker is owned at any point through the end of ante 4 | identity |
| `extract_income` | 1 | MSE | dollars gained during steps taken FROM `SELECTING_HAND`, over the remainder of the current ante | `log1p(x)/log1p(50)` |
| `cards_modified` | 1 | MSE | playing cards whose `(enhancement, edition, seal)` changed + cards added, over the remainder of the current ante | `log1p(x)/log1p(20)` |
| `tarots_used` | 1 | MSE | Tarot consumables used by this player over the remainder of the current ante | `log1p(x)/log1p(10)` |

Binary heads take the **mean over worlds** as a SOFT 0..1 BCE target — the same convention
the main loss already uses for `y`.

### Targets dropped, narrowed or re-scoped (brief §6b.1: "document any target you drop")

**None was dropped as uncomputable.** Four were operationalised in a way the brief's
one-line description leaves open, and each choice is a real limitation:

1. **`extract_income` is "in-blind money", not a proc ledger.** The engine attributes no
   money to a source (`ctx.add_dollars(amount, source)` ignores `source`; gold-seal money
   lives in `scoring.py`), and building one would mean editing `game.py` — W-EXTRACT's
   territory and a concurrently-edited file. The computable, well-defined stand-in is
   **dollars gained during steps taken from `SELECTING_HAND`**: outside the end-of-round
   payout (a `ROUND_EVAL` step) and the shop, the only dollars that move during hand play
   ARE procs — gold seal, Lucky, Business Card, Reserved Parking, Faceless. **What it
   misses:** the gold ENHANCEMENT's "$3 held at end of round" and Golden-Joker-style
   end-of-round rows, which land inside `_end_round` bundled with the blind reward and
   interest and are not separable without engine instrumentation. Only positive per-step
   deltas are summed, so boss penalties that take money during hand play (`bl_tooth`
   −$1/card, `bl_ox` money→0) do not make the target negative.
2. **"next ante" means the REMAINDER OF THE CURRENT ante** for the three window targets
   (`extract_income`, `cards_modified`, `tarots_used`): the window closes when the player's
   `ante` moves past its value at the state. That is the horizon the sandbag decision
   actually controls, and it is well defined at every state kind (a post-boss shop already
   belongs to the new ante, so there the window is the whole new ante).
3. **`xmult_by_ante4` is BINARY, not ordinal** (the brief allows either). The engine has no
   declarative xMult table — an xMult joker is one whose scoring hook multiplies
   `ScoreContext.mult_mult` — and the 35 that do split into fixed-multiplier and
   state-scaling kinds whose "tier" is not comparable without reading each joker's live
   scaling state. `XMULT_JOKERS` is a FROZEN 35-key list, and
   `test_xmult_joker_set_matches_the_engine` re-derives it by source introspection over
   `JOKER_REGISTRY` and fails on any engine drift. The head is **masked when the state is
   already past ante 4** (17 % of the fresh batch), since there the target is a fact about
   the present, not a prediction.
4. **`money_next_shop` skips the shop you are standing in.** "Next shop entry" is a
   TRANSITION into `State.SHOP`, so at a shop state the target is the *following* ante's
   shop, not the trivially-visible current balance. Masked (2–5 %) when no shop is reached
   before the rollout ends. Debt (negative dollars, Credit Card) maps to 0 through
   `log1p(max(0, x))` — a known floor, not observed in the fresh batch beyond a −0.62 mean.

`blind_cleared` is masked (~2 %) only when the rollout ends mid-blind by step cap /
ante-12 truncation rather than by the blind resolving; `pvp_margin_next` (4–5 %) when no
Nemesis resolves at all.

---

## 3. The trainer side

### 3.1 Heads

`ValueNetConfig.aux_heads: dict = {}` (default EMPTY) and `aux_hidden: int = 0`. When empty,
`SetValueNet.__init__` constructs **nothing** — no modules, no parameters, no `state_dict`
entries, no init RNG draws — and the heads are registered LAST, after `value_head`, so the
parameter order and every init draw above them are exactly what they were. `forward` /
`p_win` / `make_value_fn` / `make_values_many` never touch the heads: **play-time inference
is byte-for-byte the pre-W-AUX path whether or not a checkpoint carries them.**

`forward_with_aux(batch) -> (logits, {name: (B, dim)})` shares ONE trunk pass, which is the
whole point — the aux gradient shapes the representation V reads. Heads are
`Linear(W, dim)` or `Linear(W, h) -> ReLU -> Linear(h, dim)` (`aux_hidden`), orthogonal
init with `value_head`'s conventions. `DummyValueNet` mirrors the same contract so the
plumbing tests run on CPU in seconds.

Cost: all 8 heads linear off `trunk_width=712` = **6,417 params** on a 4,996,789-param net
(+0.13 %).

### 3.2 Loss

```
loss = BCE(absolute)  +  lam_rank * pair_logistic  +  sum_i w_i * aux_i(absolute)
                                                   [+ sum_i w_i * aux_i(pair branch a)
                                                    + sum_i w_i * aux_i(pair branch b)]
```

`w_i` defaults to `aux_weight = 0.1` with `aux_weights` per-head overrides (brief §6b.2).
`--aux-on-pairs` (default on) adds the aux term to both pair branches — free, since
`_pair_loss` already does one forward over `cat([obs_a, obs_b])`; measured branch a−b
differences are real (money nonzero on 65 % of pairs, `blind_cleared` on 38 %), so this is
not a duplicate of the absolute term. **Note the consequence:** with pairs configured the
aux objective appears in the batch three times (absolute + both branches), each at `w_i`;
`--aux-on-pairs 0` reduces it to once.

`_masked_head_loss` computes the mean over PRESENT rows only, without a `.any()` host sync
(missing rows are multiplied by 0 and excluded from the denominator), so **a batch of old,
aux-less rows contributes exactly 0** — that is what "old shards train with aux muted"
means operationally, and `test_old_shards_train_with_aux_muted` pins `aux_loss == 0.0` and
`loss == bce_loss` on an aux-free corpus.

### 3.3 Target standardisation — a deviation, and why it was necessary

**`aux_standardize` defaults to True: the REGRESSION targets are z-scored on the train
split** (binary heads are left alone). This is not in the brief and was added after the
first ablation showed why.

The log1p transforms compress hard: `money_next_shop` has raw sd $11.3 but transformed sd
**0.085**, and the count heads land at 0.02–0.05. An MSE term at that scale produces
gradients ~30× smaller than the BCE term, so at the brief's `w_i = 0.1` the heads barely
move. First ablation (unstandardised, 3,000 steps, `w = 0.1`): **money head R² = −0.12 /
−0.18 / −0.27** on the mixed corpus (−1.38 / −1.43 / −1.55 on the small one), still crawling
up from −9 at step 100 — under-trained, not broken. Standardising puts
every regression head on a comparable gradient scale and makes `w_i = 0.1` mean the same
thing across heads. R² is scale-invariant, so the reported per-head numbers mean the same
thing either way; RMSE is reported in sd units.

The `(mean, sd)` per column is computed from the TRAIN side only — the same discipline
`const` (the train-set label mean) already follows — persisted in `state()["aux_norm"]`,
and **restored from the checkpoint on a resume** (`restore_aux_norm`) so a resume cannot
silently refit the heads to a different scale. `--aux-standardize 0` restores the literal
brief behaviour.

### 3.4 Metrics

`m["aux"]` every eval, on the held-out ABSOLUTE rows (falling back to held-out pair branch a
when the absolute holdout carries no aux, so a pairs-only corpus still measures its heads):
per head, `n` (present rows only) plus

* binary: `brier`, `brier_base` (the base-rate predictor), `bce`, `acc@0.5`, `base_rate`
* regression: `r2`, `rmse`, `y_mean`/`y_sd`, `p_mean`

plus `coverage` (the masking diagnostic). Every pre-existing metric is computed and reported
exactly as before — `evaluate` and `evaluate_pairs` are untouched, and the ECE guardrail
still fires unconditionally. The console line gains a compact `aux[...]` segment.

### 3.5 CLI

```
--aux-heads all|<comma list>   --aux-weight 0.1   --aux-weights '{"blind_cleared":0.2}'
--aux-hidden 0                 --aux-on-pairs 0|1 --aux-standardize 0|1
```

All are in `_RESUME_KEYS`. `--aux-weight`/`--aux-weights`/`--aux-on-pairs` are pure loss
knobs; `--aux-heads`/`--aux-hidden` change the GRAPH and take the "bolt heads onto an
existing checkpoint" path below — like `holdout_frac`, they are a deliberate override and
are not covered by the bit-exact guarantee.

Producers: `gen_labels.py --aux` and `gen_pairs.py --aux` (both off by default; both write
per-head coverage into their `mp/results/*.json`).

---

## 4. Preserved: resume, PAUSE, fingerprints, old checkpoints

* **Bit-exact resume, extended to aux state** (`test_resume_is_bit_exact_with_aux_state`):
  W-RANK's pinned test structure with `--aux-heads all --aux-hidden 8` configured — resume
  a 60-step run from its step-30 checkpoint and every weight tensor (aux heads included),
  every Adam moment, the optimizer's param-group size, `pair_epoch`/`pair_cursor` and the
  eval history match a straight 60-step run. Nothing extra had to be persisted for this:
  the heads live in the net's own `state_dict` and its `cfg`, so `value_net.save_checkpoint`
  / `load_checkpoint` carry them for free.
* **Loading an old checkpoint is unaffected.** Its `cfg` has no `aux_heads` key,
  `ValueNetConfig.from_dict` defaults to `{}`, no heads are built, `load_state_dict(strict=
  True)` passes. Pinned on the real keeper (`test_keeper_checkpoint_still_loads_and_plays`:
  4,996,789 params, `aux_head_names() == []`, `make_value_fn` returns a valid P(win)).
* **A checkpoint saved WITH heads still loads for play**
  (`test_checkpoint_saved_with_heads_still_loads_for_play`): every tensor round-trips
  bit-equal, `forward` equals `forward_with_aux`'s first return, and the play-time call path
  never evaluates a head.
* **Bolting heads onto a headless checkpoint** (`--resume <ckpt> --aux-heads all`) is the
  one case that is not a strict load. `_load_net_state` loads `strict=False` and REFUSES
  anything missing that is not an `aux_heads.*` tensor (and any unexpected key at all);
  `_load_optimizer_state` extends the saved param-group index list, so the first k
  parameters keep their Adam moments and the fresh head tail starts clean — the brief's
  "fresh-init heads when absent". `test_heads_can_be_added_to_a_checkpoint_that_has_none`
  pins that the trunk comes across bit-equal and that training continues.
* **PAUSE / `.DONE`** unchanged and re-pinned with aux configured
  (`test_pause_and_done_still_work_with_aux`).
* **Fingerprint filtering** unchanged; `test_fingerprint_filtering_still_works_with_aux`
  checks the aux arrays stay row-aligned after `filter_by_fingerprint` drops rows.

---

## 5. Measured: cost of the instrumentation

| | value |
|---|---|
| `AuxRecorder.start()` (2 full-deck signatures) | 0.062 ms |
| `AuxRecorder.after()` | **1.3 µs / call** |
| `AuxRecorder.finish()` | 0.073 ms |
| per rollout at 182 decisions | **0.37 ms** |
| against ~1,050 ms/rollout in-worker | **0.04 %** |

Matched-trajectory end-to-end A/B (global `Card._counter` pinned so both arms replay the
identical trajectory, order alternated, warm-up pass discarded): **18/18 pairs matched on
decision count and outcome**, mean 0.7334 s without vs 0.7288 s with — i.e. **inside noise
(−0.6 %)**, consistent with the 0.04 % microbenchmark.

Two traps worth recording for whoever measures rollouts next:

1. **`BalatroGame._playing_cards_sorted()` sorts by `Card.id`, and `Card._counter` is
   process-global**, so two rollouts of the "same" state on the same seed diverge if a
   different number of cards was created earlier in the process. Back-to-back they agree;
   after an intervening rollout they do not. This is pre-existing (it is a cousin of
   PAIRS_NOTES §7.3's `PYTHONHASHSEED` caveat) and it invalidates any naive A/B unless the
   counter is pinned.
2. The EV player warms per-process caches, so a first-run/second-run comparison flatters the
   second arm by ~35 %. Alternate the order and discard a warm-up pass.

`test_recorder_does_not_change_the_rollout` pins the side-effect-freedom claim directly.

---

## 6. The gate

### 6.1 The data (brief §6b.5: "the proof shards + a small fresh aux-recorded batch")

Two fresh aux-recorded campaigns, 8 workers each, box otherwise quiet, ~6 and ~8 minutes:

```bash
python mp/ev/scripts/gen_labels.py --run-dir mp/ev/runs/aux_g1 --seeds random:60 --seed-rng 77 \
    --n-states 12 --n-rollouts 8 --workers 8 --policy ev --budget fast --shop-tier rules \
    --encoder v2 --flush-jobs 4 --aux --max-jobs 24 --minutes 13 --name aux_g1
python mp/ev/scripts/gen_pairs.py --run-dir mp/ev/runs/aux_pairs_g1 --seeds random:60 --seed-rng 78 \
    --n-states 6 --n-worlds 8 --workers 8 --probe-jobs 0 --flush-jobs 4 --aux \
    --max-jobs 44 --minutes 13 --name aux_pairs_g1
```

| | labels | pairs |
|---|---|---|
| rows | **576 labels** (24 seeds × 12 states × 2) | **238 pairs + 476 absolute rows** (44 seeds) |
| throughput, 8 workers | 91.5 labels/min, 1,138 ms/rollout | 29.3 pairs/min, 939 ms/rollout |
| failures | 0 | 0 |

The pair campaign's variance-reduction factor came out **1.75× (mean ρ +0.464)**, against
PAIRS_NOTES' 1.78× / +0.468 on 1,301 pairs — an independent check that the aux
instrumentation did not perturb the pair machinery.

**Per-head coverage on the fresh batch** (`aux_coverage` in `mp/results/labels_aux_g1.json`
and `pairs_aux_pairs_g1.json`):

| head | labels | pairs (a / b) |
|---|---|---|
| `lives_2antes` | 1.00 | 1.00 / 1.00 |
| `extract_income` | 1.00 | 1.00 / 1.00 |
| `cards_modified` | 1.00 | 1.00 / 1.00 |
| `tarots_used` | 1.00 | 1.00 / 1.00 |
| `blind_cleared` | 1.00 | 0.983 / 0.983 |
| `money_next_shop` | 0.979 | 0.945 / 0.945 |
| `pvp_margin_next` | 0.958 | 0.954 / 0.945 |
| `xmult_by_ante4` | 0.854 | 0.815 / 0.815 |

**What the targets look like** (576 labels, raw units):

| head | mean | sd | min | max |
|---|---|---|---|---|
| `money_next_shop` | $23.74 | 11.32 | −0.62 | 58.0 |
| `lives_2antes` | [2.09, 2.09] | [1.21, 1.21] | | |
| `pvp_margin_next` | 0.000 | 0.201 | −1.13 | 1.13 |
| `blind_cleared` | 0.711 | 0.359 | 0 | 1 |
| `xmult_by_ante4` | 0.385 | 0.374 | 0 | 1 |
| `extract_income` | $0.695 | 2.491 | 0 | 22.5 |
| `cards_modified` | 0.298 | 0.519 | 0 | 3.25 |
| `tarots_used` | 0.362 | 0.598 | 0 | 3.75 |

Branch a−b differences on the 238 pairs (the within-state signal `--aux-on-pairs` adds):
`money_next_shop` nonzero on 65 % of pairs (sd 4.87), `pvp_margin_next` 88 %,
`lives_2antes` 58 %, `blind_cleared` 38 %, `cards_modified` 35 %, `tarots_used` 36 %,
`xmult_by_ante4` 29 %, `extract_income` 18 %.

### 6.2 The ablation — `mp/results/aux_ablation.json`

**Corpus (the brief's literal ask):** absolute = `aux_g1/shards` (576) + `aux_pairs_g1/abs_shards`
(476) + `pairs_s1/abs_shards` (2,602) = **3,654 rows**; pairs = `aux_pairs_g1/shards` (238) +
`pairs_s1/shards` (1,301) = **1,539 pairs**. Held out by seed at 0.25 → **166 absolute rows /
439 pairs, of which 29 are RESOLVED**. Aux coverage over the whole corpus is 29 % of absolute
rows and 15 % of pairs (only the fresh batch carries it); the rest train with aux muted.

**Config, identical in every arm:** real `SetValueNet` (5,003,206 params with 8 linear heads /
4,996,789 without), batch 256, lr 3e-4 cosine, warmup 200, 3,000 steps, `--eval-every 100`,
`lam_rank 1.0`, `tau 0.05`, CUDA, seeds 0/1/2. Aux arms: all 8 heads, `aux_hidden 0`,
`aux_on_pairs 1`, `aux_standardize 1`, `aux_weight ∈ {0.10, 0.03}`.

Three selection rules are reported because they do not agree, and picking one after the fact
would be the wrong kind of reporting. `±` is the spread over the 3 seeds.

**best-BCE eval** (the trainer's own `best` criterion):

| arm | Brier | BCE | AUC | ECE | pair acc (of 29 resolved) | step |
|---|---|---|---|---|---|---|
| noaux | 0.0877 ± 0.0071 | 0.7028 ± 0.0236 | **0.685** ± 0.017 | 0.105 ± 0.020 | **0.598** | 500 |
| aux w=0.10 | **0.0876** ± 0.0016 | **0.6935** ± 0.0056 | 0.666 ± 0.039 | **0.086** ± 0.024 | 0.552 | 300 |
| aux w=0.03 | 0.0911 ± 0.0071 | 0.7095 ± 0.0152 | 0.675 ± 0.028 | 0.113 ± 0.002 | 0.517 | 367 |

**best-Brier eval:**

| arm | Brier | BCE | AUC | ECE | pair acc | step |
|---|---|---|---|---|---|---|
| noaux | **0.0859** ± 0.0052 | 0.7130 ± 0.0318 | **0.695** ± 0.008 | 0.101 ± 0.014 | **0.598** | 867 |
| aux w=0.10 | 0.0868 ± 0.0031 | **0.6979** ± 0.0130 | 0.669 ± 0.045 | **0.086** ± 0.023 | 0.540 | 533 |
| aux w=0.03 | 0.0888 ± 0.0063 | 0.7182 ± 0.0219 | 0.692 ± 0.019 | 0.117 ± 0.008 | 0.575 | 800 |

**final step 3,000** (every arm is deep in overfit by here — 3,654 rows, 14 batches/epoch):

| arm | Brier | BCE | AUC | ECE | pair acc |
|---|---|---|---|---|---|
| noaux | **0.0902** ± 0.0063 | **0.7862** ± 0.0292 | 0.694 ± 0.012 | **0.114** ± 0.015 | 0.575 |
| aux w=0.10 | 0.0972 ± 0.0074 | 0.8049 ± 0.0317 | **0.696** ± 0.019 | 0.141 ± 0.012 | 0.517 |
| aux w=0.03 | 0.0951 ± 0.0077 | 0.8051 ± 0.0299 | **0.696** ± 0.018 | 0.134 ± 0.014 | 0.563 |

**Pair accuracy, per seed, with Wilson 95 % CIs** — this is the number the gate asks about and
it is the one with no power at all:

| arm | s0 | s1 | s2 |
|---|---|---|---|
| noaux | 18/29 = 0.621 [0.44, 0.77] | 18/29 = 0.621 [0.44, 0.77] | 16/29 = 0.552 [0.38, 0.72] |
| aux w=0.10 | 17/29 = 0.586 [0.41, 0.74] | 14/29 = 0.483 [0.31, 0.66] | 17/29 = 0.586 [0.41, 0.74] |
| aux w=0.03 | 18/29 = 0.621 [0.44, 0.77] | 15/29 = 0.517 [0.34, 0.69] | 12/29 = 0.414 [0.26, 0.59] |

**Per-head held-out metrics, best over the run** (mean of 3 seeds; regression heads report R²
on the standardised target, so 0 is "no better than predicting the holdout mean"; RMSE is in
sd units):

| head | n | w = 0.10 | w = 0.03 | baseline |
|---|---|---|---|---|
| `money_next_shop` | 158 | **R² +0.541** (rmse 0.659) | **R² +0.464** (rmse 0.712) | 0 |
| `lives_2antes` | 166 | **R² +0.549** (rmse 0.670) | R² +0.512 | 0 |
| `tarots_used` | 166 | **R² +0.451** | R² +0.399 | 0 |
| `cards_modified` | 166 | **R² +0.427** | R² +0.407 | 0 |
| `extract_income` | 166 | R² +0.075 | R² +0.050 | 0 |
| `pvp_margin_next` | 154 | **R² −0.271** | R² −0.404 | 0 |
| `blind_cleared` | 166 | Brier **0.1308** | Brier **0.1251** | 0.1462 (base rate 0.656) |
| `xmult_by_ante4` | 130 | Brier 0.1036 | Brier 0.0981 | 0.0949 (base rate 0.321) |

### Gate verdict, line by line (brief §6b.5)

| ask | result |
|---|---|
| held-out **Brier** with aux ≥ without | **tie at best-BCE** (0.0876 vs 0.0877); noaux ahead by 0.0009 at best-Brier and 0.0070 at step 3,000. Every gap is inside the ±0.005–0.007 seed spread. **Not a pass, not a fail — no separation.** |
| held-out **pair accuracy** with aux ≥ without | **NOT met**: 0.552 vs 0.598 at w=0.10 (16.0 vs 17.3 of 29 resolved pairs). Wilson CIs overlap almost completely; 29 resolved pairs cannot resolve a 0.05 difference. |
| **money head R² clearly > 0** | **MET: R² = +0.541** (per seed +0.568 / +0.523 / +0.531 at w=0.10). Three more heads clear +0.4. |
| **no ECE degradation** past W-RANK's guardrail | **MET, and improved**: 0.086 vs 0.105 (aux is better calibrated at both selection rules). Both arms exceed W-RANK's 0.05 default at this data scale — that is a 166-row-holdout artefact, not an aux effect (keeper V measured 0.021 on 51k labels). |
| no-aux path bit-identical to the pre-W-AUX trainer | **MET** — §6.3. |

**Honest reading: the ablation is a null.** BCE and ECE favour aux, AUC and pair accuracy
favour no-aux, Brier ties, and no gap exceeds the seed-to-seed spread. That is the expected
outcome of asking a 3,654-row / 29-resolved-pair holdout to detect a representation-shaping
effect, and it is why §8.1 says the real read is R2 on the full campaign. What the numbers DO
establish is that the heads work: five of eight carry real held-out signal, the money head at
R² +0.54, and the machinery costs 0.04 % of a rollout and nothing at play time.

Two secondary observations worth carrying forward:

* **Aux consistently improves calibration and BCE while costing AUC.** ECE 0.086 vs 0.105 and
  BCE 0.6935 vs 0.7028, against AUC 0.666 vs 0.685, in every selection rule and in both
  supporting ablations. A shared-trunk regulariser that trades ranking sharpness for
  calibration is a plausible mechanism; on 166 holdout rows it is equally plausibly noise.
* **The aux arms peak earlier** (best-BCE at step 300 vs 500, best-Brier at 533 vs 867). With
  8 heads at w=0.1 the aux block is roughly the same magnitude as the BCE term at init, so it
  is a substantial fraction of the objective — see §8.2.

**Supporting runs** (same corpus, both 12 runs, in `aux_ablation.json` under
`supporting_runs`):

| variant | money head R² at step 3,000 | B_mixed Brier (noaux → aux) |
|---|---|---|
| unstandardised, branch-sum, w=0.1 | **−0.18 / −0.27 / −0.12** | 0.0877 → 0.0867 |
| standardised, branch-sum, w=0.1 | **+0.51 / +0.53 / +0.44** | 0.0877 → 0.0882 |
| standardised, branch-mean, w=0.1 (the gate) | **+0.54** (best over run) | 0.0877 → 0.0876 |

The first row is the measurement that forced §3.3: with the raw log1p targets the money head
never gets off the ground. `A_aux_only` (the fresh 1,052-row batch alone, 886 train rows) is
also in the JSON and is **entirely uninformative** — the aux arm's best eval is step 0 on two
of three seeds and only 3 holdout pairs resolve. It is reported for completeness, not as
evidence.

### 6.3 The no-aux path is bit-identical to the pre-W-AUX trainer

Method: snapshot the working-tree `train_v.py` / `labels.py` / `pairs.py` /
`value_net.py` **before** any W-AUX edit; build two fixed shard sets once; swap the pre-W-AUX
files back into the tree, run a fixed config, dump weights + optimizer state; restore the
W-AUX files, run the identical config, diff.

| | dummy net, 120 steps | real `SetValueNet` (small cfg), 20 steps |
|---|---|---|
| weight tensors bit-equal | **6 / 6** | **52 / 52** |
| Adam moment tensors bit-equal | **18 / 18** | **156 / 156** |
| eval history (step, BCE, Brier) | identical | identical |
| `n_params` | — | 79,517 = 79,517 |

An incidental second confirmation from the ablation itself: on `A_aux_only` seed 2 both arms
selected the step-0 eval and reported **byte-identical** BCE/Brier/AUC/ECE — the aux heads
do not perturb the base net's initialisation, because they are registered last.

### 6.4 Test counts

| suite | result |
|---|---|
| `mp/ev/tests/test_aux.py` (new) | **31 passed**, ~8 s |
| `python -m pytest mp/ev` | **285 passed / 0 failed**, ~75 s (254 before this file) |
| `python -m pytest mp/agent/tests` | **396 passed** (`value_net.py` is the only `mp/agent` file touched; `test_value_net.py` + `test_encoder_v2.py` = 35 of them) |

No unrelated failures were seen in the final runs. Earlier in the build the tree was green
too — W-EXTRACT's `hand.py`/engine churn and W-PROBE's fixtures were already settled by the
time this workstream started.

What `test_aux.py` pins, by layer: the spec table and its transforms; the xMult joker list
re-derived from `JOKER_REGISTRY` by source introspection; aggregation means / masking /
strict-JSON output / coverage; **the recorder against an independent second pass over a raw
trace of the same rollout** (8 player-perspectives × every field but `xmult`); the recorder's
side-effect-freedom; the xMult flag on a hand-built board and its past-ante-4 masking;
`label_job` / `pair_job` with and without `aux=True`; both branches' aux matching the
absolute rows a pair yields; the shard round trip through BOTH `pairs.load_pair_shard` and
`train_v.PairDataset`; the net's zero-parameter default, the head parameter arithmetic, the
keeper checkpoint, a heads-carrying checkpoint loading for play; spec resolution; array
transform/masking incl. malformed values; standardisation (train-only stats, binaries
untouched, carried across a resume); the no-aux path leaving the trainer untouched;
per-head metrics + learning on a synthetic deterministic target; old shards training with
aux muted; aux gradients reaching the shared trunk but not the value head; the pair-branch
MEAN; **bit-exact resume with aux state**; bolting heads onto a headless checkpoint; PAUSE /
`.DONE`; fingerprint filtering staying row-aligned; and the CLI.

---

## 7. Deviations, one line each

1. **`aux_standardize=True` by default** (not in the brief) — a log1p target's sd is ~0.085,
   so at `w_i = 0.1` the regression heads' gradients are ~30× below the BCE term's and they
   effectively never train; measured directly (§3.3, §6.2). `--aux-standardize 0` restores
   the literal behaviour.
2. **`extract_income` = in-blind money, not a proc ledger** — the engine attributes no source
   to a dollar and adding one means editing `game.py`, which is W-EXTRACT's file (§2.1).
3. **"next ante" = the remainder of the CURRENT ante** for the three window targets — the
   horizon the sandbag decision actually controls, and well defined at every state kind (§2.2).
4. **`xmult_by_ante4` is binary and its joker set is a frozen list guarded by an
   introspection test** — the engine has no declarative xMult table (§2.3).
5. **`aux` rides inside the existing JSON blobs, not new npz arrays** — keeps W-PAIRS's and
   W-RANK's on-disk layouts byte-identical and every existing loader working unchanged (§1).
6. **The aux term appears three times per step when pairs are configured** (absolute +
   both branches), each at `w_i` — `--aux-on-pairs 0` reduces it to once (§3.2).
7. **`_pair_loss` still returns `(loss, info)`** — W-RANK's tests call it directly, so the
   pair-branch aux loss is stashed on `self._last_pair_aux` for `train_step` instead of being
   added to the return tuple.
8. **`gen_labels.py` / `gen_pairs.py` each gained one `--aux` flag** — W5's/W-PAIRS's drivers,
   but the brief puts "aux-target recording in the label/pair rollout loop" in this
   workstream's ownership and a producer needs a switch. Additive; default off.

---

## 8. Open issues / what the lead should decide

1. **The gate is a null at proof scale and cannot be otherwise.** 2,740 training rows and 22
   resolved held-out pairs cannot resolve a Brier difference of 0.001 or a pair-accuracy
   difference of 0.1. The real read is R2 in the idle-box runbook: retrain V-v2 on the full
   pair campaign WITH `--aux` recorded from the start, and compare with/without there.
   **This means the R1 full pair campaign should be run with `--aux`** — it is 0.04 % of the
   cost and it is the only way to get an aux corpus big enough to answer the question.
2. **`aux_weight` is unfit, and 8 heads at 0.1 each is a bigger term than it looks.** The
   brief specifies `sum_i w_i * aux_i` with `w_i ≈ 0.1`; with 8 standardised heads that sum
   is ≈ 0.8 at init against a BCE of ≈ 0.69, i.e. the aux block is roughly half the objective.
   The measured symptom is that the aux arms peak earlier (best-BCE at step 300 vs 500). The
   0.03 arm was swept here and is not obviously better (Brier 0.0911 vs 0.0876), so the fix is
   probably not "smaller", but this needs a real sweep on the full corpus alongside
   `lam_rank`/`tau` (RANK_NOTES §6's open item). If a single knob is wanted, consider making
   `aux_weight` the TOTAL and dividing by the head count — a one-line change here, but a
   deviation from the brief's literal formula, so it is left for the lead.
3. **`extract_income` needs an engine hook to be exact.** If W-EXTRACT (or a later round)
   ever threads `source` through `ctx.add_dollars` and the `scoring.py` money rows into a
   per-round ledger, this head becomes the real sandbag signal instead of a good proxy, and
   the gold-enhancement / end-of-round rows stop being invisible.
4. **Two heads are worth reconsidering, and they are not the ones I expected.**
   `cards_modified` (+0.43) and `tarots_used` (+0.45) predict fine despite being sparse; the
   two that do not are **`pvp_margin_next` (R² −0.27 — worse than predicting the mean)** and
   **`extract_income` (+0.075, barely above chance)**, and `xmult_by_ante4` is the only binary
   head that does not beat its base rate (0.104 vs 0.095). `pvp_margin_next` is plausibly
   irreducible — the next Nemesis's score margin depends on the opponent's hidden build, which
   the encoder cannot see by construction — so it may belong out of the default list.
   `extract_income` is sparse (mean $0.70, 82 % zeros) rather than unpredictable; it should
   improve on proc-biased state selection, which is the same lever PAIRS_NOTES §7.4 asks for.
5. **`money_next_shop` at a shop state predicts the NEXT ante's shop**, which is much harder
   than the mid-blind case. Splitting it by state kind (or masking it at shop states) is a
   cheap next-round refinement.

---

## 9. Note for W-ACTIVE (brief §6b.6) — NOT implemented

`mp/results/active_poc_*.md` did not exist when this workstream closed (checked at the start
and at the end of the build), so there is no POC verdict to react to. The hook is recorded
here for whoever picks it up.

**Per-head error is a ready-made, already-computed acquisition signal, and it is strictly
richer than V's own error.** An active-labelling score currently has to guess where V is
wrong from V alone — a scalar, whose residual against a ±0.24-CI rollout label is mostly
label noise. With aux heads the trainer already produces, per candidate state, a VECTOR of
held-out residuals whose noise floors are much lower: `money_next_shop` is near-deterministic
given the state (R² +0.54 on 158 rows and rising with data), `blind_cleared` beats its base
rate, and `lives_2antes` is the two-step version of the very quantity V integrates. Three
concrete uses, cheapest first: (a) **surprise-weighting** — score a candidate by the
magnitude of its predicted-vs-realised aux residual on the heads with the lowest noise floor,
which flags states whose dynamics the trunk has NOT learned even when V's own value looks
confident; (b) **disagreement without an ensemble** — the aux heads and the value head read
the SAME trunk, so a state where the money head is confidently right while V is uncertain is
representationally understood but strategically ambiguous (the pair lever's territory),
whereas one where the aux heads are all wrong is genuinely out of distribution (the label
lever's territory), which splits the acquisition budget between the two levers on a signal
that costs nothing to compute; (c) **cheap proxy labels** — every aux target is recorded
during the rollouts a candidate would need anyway, so an acquisition function can be
validated offline against a held-out aux target long before it is validated against win rate.
The prerequisite is only that the campaign feeding W-ACTIVE was generated with `--aux`
(§8.1), which is why that recommendation matters beyond this workstream.
