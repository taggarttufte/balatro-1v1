# RANK_NOTES — W-RANK: lever (b)'s loss (Phase 5 rev 2 V-v2, 2026-08-24/25)

Owns: `ev/train_v.py` additions (the pairwise ranking loss, pair shard I/O,
`player_fingerprint` filtering, extended bit-exact resume), `ev/tests/test_train_v_pairs.py`
(17 tests), this file.  Nothing else touched.  `ev/dataset.py`, `ev/labels.py`,
`ev/pairs.py`, `ev/hand.py`, `ev/player.py` are read-only from here — see §2 for why
this module carries its own small mirror of `dataset.py`'s shard conventions instead of
extending it.

Coded against the FROZEN schema in `docs/PHASE5_V2_BRIEF_2026-08.md` §5.3 without waiting for
W-PAIRS (brief instruction).  W-PAIRS's actual `ev/pairs.py` landed mid-build (2026-08-25);
§2 records the reconciliation once it did.

---

## 0. What is where

| file | what |
|---|---|
| `ev/train_v.py` | unchanged W5 trainer + this round's additions (below) |
| `ev/tests/test_train_v_pairs.py` | 17 tests: shard I/O, splits, fingerprint filtering, loss/metrics, bit-exact resume w/ pairs, gradient-flow smoke, the tiny-set overfit, real-`pairs.py` interop, CLI |
| `ev/tests/test_train_v.py` | unchanged (8 tests) — the pre-existing pinned resume test is left as-is; a NEW test extends it to pairs (§4) rather than editing the original, so "no pairs configured" stays independently pinned |

Additions to `train_v.py`, by section:

* `TrainVConfig`: `pair_shards`, `lam_rank`, `tau`, `pair_weight_cap`, `pair_batch_size`,
  `ece_guardrail`, `absolute_fingerprint_mode`, `new_fingerprint`, `pair_fingerprint_allow`.
* Pair shard I/O: `PairRow`, `PairShard`, `save_pair_shard`, `load_pair_shard_npz`,
  `load_pair_records_json`, `list_pair_shards`, `PairDataset` (§2).
* Fingerprint filtering: `filter_by_fingerprint` (absolute rows), `filter_pairs_by_fingerprint`
  (pairs) (§3).
* Metrics: `evaluate_pairs`, `_pair_breakdown` (§1).
* `VTrainer`: `load_pair_data`, `_pair_epoch_order`/`pair_batches_per_epoch`/`next_pair_batch`
  (own resumable stream, §4), `_pair_loss`, `train_step`/`eval`/`state`/`from_checkpoint`
  extended, `run()` threads a `pair_data` override through like `data`.
* CLI: `--pair-shards --lam-rank --tau --pair-weight-cap --pair-batch-size --ece-guardrail
  --absolute-fingerprint-mode --new-fingerprint --pair-fingerprint-allow`.

---

## 1. The loss and the metric — defaults and why

**Loss** (`VTrainer._pair_loss`): `loss = bce_loss + lam_rank * pair_loss`, where `pair_loss`
is a confidence-weighted pairwise logistic on the SAME net's output, computed with ONE forward
pass over `cat([obs_a_batch, obs_b_batch])` (so both branches route gradient through the
identical shared-encoder weights — no separate heads, no `.detach()` anywhere on either side;
verified directly, §4/§5):

```
va, vb   = net(cat).sigmoid()[:n], net(cat).sigmoid()[n:]         # V(obs_a), V(obs_b) — both P(win)
score    = (va - vb) / tau
target   = 0.5 * (sign(delta) + 1)                                 # 1 if a beat b, 0 if b beat a, 0.5 tie
weight   = clip(|delta| / max(delta_ci, 1e-3), 0, pair_weight_cap)
pair_loss = sum(weight * BCEWithLogits(score, target)) / sum(weight)     # 0 if all weights are 0
```

`score` is literally the brief's `(V(obs_a) - V(obs_b)) / tau` — a PROBABILITY difference (not
a logit difference), matching how V is used everywhere else in this codebase (a calibrated
`P(win)`, brief §0).  Feeding that into the standard `binary_cross_entropy_with_logits` formula
is a temperature-scaled logistic on a bounded quantity, which is well-behaved and keeps `score`
numerically tame (`|score| ≤ 1/tau`) without extra clipping in the training path (`evaluate_pairs`,
which runs outside autograd, does clip to ±30 for `exp` safety).

**Defaults chosen and why** (all config, all overridable per-run; the idle-box campaign should
retune once real pair volumes and realized deltas are in hand — these are principled starting
points, not fit to real pair data, which does not exist yet):

| knob | default | rationale |
|---|---|---|
| `lam_rank` | **1.0** | Both loss terms are bounded cross-entropy-scale quantities (≈ log 2 ≈ 0.69 at "no signal"); equal weighting is the neutral starting point until real pair data shows whether rank-loss needs up/down-weighting relative to the 51k-row BCE pretraining signal. |
| `tau` | **0.05** | Brief §0's own diagnosis: "per-action EV gaps (≪ 0.05) sit below the label noise" — that IS the scale lever (b) exists to resolve. `tau` on that scale keeps the logistic sensitive to a 0.02–0.05 probability gap (`sigmoid(0.05/0.05)=0.73`, `sigmoid(0.02/0.05)=0.60`) instead of saturating on it. |
| `pair_weight_cap` | **4.0** | `\|delta\|/delta_ci` is a resolution statistic (`delta_ci` is a ~95% half-width, ≈ 1.96σ of the paired difference); a ratio ≈ 1 is borderline-significant, ≈ 4 is "≈4× the 95% half-width from zero" — very confidently resolved. Capping there stops a handful of unanimous pairs (e.g. 8/8 vs 0/8 worlds) from dominating a minibatch's gradient relative to the many more marginal, more numerous ones. |
| `pair_batch_size` | **128** | Pairs are scarce next to absolute rows (brief §5.5's proof campaign target is 1–2k pairs vs the existing 50k-row absolute corpus); small enough to give several gradient steps per pair-epoch even at proof scale, large enough for a stable weighted-mean estimate. |
| `ece_guardrail` | **0.05** | The keeper V (absolute-only) measured ECE 0.021 (brief §0, "well-calibrated"); ~2.5× that flags a real calibration regression from adding the rank loss while tolerating ordinary small-holdout noise. Soft: logs a `[ECE GUARDRAIL]` line and sets `ece_guardrail_breached` in every eval record — it does not stop training, since the brief's ask is "must not silently degrade," not "must never move." |
| weight-ratio floor `1e-3` | (not configurable) | `delta_ci` is a probability-scale CI half-width; any real pair (≥ 8 shared-world outcomes) has a Wilson floor far above `1e-3`, so this only guards the pathological exact-zero-CI case without ever touching a real weight. |

**Metric** (`evaluate_pairs`): held-out **pair accuracy** = fraction of RESOLVED pairs where
`sign(V(obs_a) - V(obs_b))` matches `sign(delta)`.  "Resolved" = `|delta| > delta_ci` (zero
excluded from the ~95% CI) — a threshold chosen independently of the training-time weight (that
one caps the top end; this one floors the bottom end at exactly the 95% mark, so the reported
accuracy is never diluted by pairs whose own rollout couldn't tell `a` from `b` either).
Accuracy is **tau-independent by construction** — `sign` of a probability difference doesn't
depend on how it's temperature-scaled — so it is safe to compare across `tau` sweeps.
Broken out by `pair_source` and `state_kind` (`_pair_breakdown`, `min_n=5` — deliberately far
below `evaluate`'s absolute-row threshold of 20, since proof-scale pair campaigns are hundreds–
low-thousands rows total, not 50k; revisit once the idle-box campaign lands real volumes).
`pair_loss` (same weighted formula as training) is also reported, for visibility, but does not
feed `VTrainer.best` (kept as BCE-only, unchanged, so an existing "best checkpoint" convention
doesn't silently start meaning something else).

All existing absolute metrics (BCE/Brier/AUC/ECE/reliability, `by_kind`) are computed and
reported EXACTLY as before — `evaluate()` itself is untouched; pairs are additive (`m["pairs"]`
is only present when `--pair-shards` is given, and `eval()` sets `m["ece_guardrail_breached"]`
unconditionally so calibration is always visible, with or without pairs).

---

## 2. Pair shard format — the mirror, and the W-PAIRS reconciliation

The brief freezes FIELD NAMES (§5.3), not a file format, and says `obs_a`/`obs_b` use "the
same storage as labels" — `dataset.py`'s per-key stacked-array convention.  W-RANK doesn't own
`dataset.py` this round (ground rule §2: don't touch another workstream's files, stop at the
interface) and W-PAIRS was building the producer concurrently, so rather than risk a concurrent
edit collision this module carries its OWN small mirror of `dataset.py`'s `Shard`/`LabelDataset`
machinery, specialised to two obs blocks per row: `obs_a__<key>`/`obs_b__<key>` stacked arrays,
typed columns for the 8 scalar fields (`PAIR_COLUMNS`), dedicated `delta`/`delta_ci` float32
arrays (same reasoning `dataset.py` keeps `y` OUTSIDE `META_COLUMNS` — every consumer needs
them as plain floats, not a `.columns` lookup by name), and everything else (`action_a`,
`action_b`, `outcomes_a`, `outcomes_b`, `meta`) in one JSON blob column per row, mirroring
`dataset.py`'s `meta_json`.  Two loaders: `.npz` (primary) and literal `.json`/`.jsonl` records
matching the schema text field-for-field (a compatibility path — `obs_a`/`obs_b` cast to
float32 on load, which is safe for the encoder-v2 categorical fields too, since
`value_net._ix` widens anything non-long to `long` before an embedding lookup).

**W-PAIRS's actual `pairs.py` landed mid-build (2026-08-25).**  Its `save_pair_shard` /
`load_pair_shard` / `PairDataset` / `PAIR_COLUMNS` turned out to match this module's
independently-chosen layout almost exactly: `obs_a__<key>`/`obs_b__<key>`, the same 8 typed
scalar columns (order differs, immaterial), `delta`/`delta_ci` as top-level float32 arrays
under those exact names.  The one real difference: their per-row JSON blob column is named
`pair_json` (holding the FULL frozen record, delta/delta_ci included), this module's was
`extra_json` (holding only the non-typed fields).  Reconciled: `save_pair_shard` now writes
`pair_json` with the full record (matching `pairs.pair_record`'s shape exactly);
`load_pair_shard_npz` checks `pair_json` first, `extra_json` second (for any shard written by
an earlier version of this module).  `test_reads_pairs_py_shards_directly` builds a REAL
`pairs.PairRow` + calls the REAL `pairs.save_pair_shard`, then loads it with THIS module's
`PairDataset.load` — passing end to end, so `--pair-shards` pointed at W-PAIRS's real
`ev/runs/pairs_s1/shards` (or the idle-box campaign's output) should load unchanged.
Not re-verified against `pairs.pair_job`'s actual output end-to-end (that needs a real rollout
campaign, out of scope for a trainer smoke test) — only against its shard-writing path, which
is the actual interface boundary between the two workstreams.

`PairDataset.split_by_seed` calls `dataset.seed_in_holdout` directly (same `salt="v-holdout"`,
same hash) — a seed's absolute rows and its pairs are therefore guaranteed to land on the same
side of the holdout split (`test_split_by_seed_matches_absolute_dataset`).

---

## 3. `player_fingerprint` filtering

`dataset.py`'s typed `META_COLUMNS` doesn't carry `player_fingerprint` (not this workstream's
file to add a column to), so absolute-row filtering reads it out of each row's free-form
`meta` dict — the old 51k corpus (`ev/runs/labels_full*`) simply lacks the key entirely,
which is exactly how brief §2 says to identify it ("usable for absolute-BCE pretraining, never
silently mixed [with a DIFFERENT new policy] — the trainer must be able to filter").

* `--absolute-fingerprint-mode any` (**default**): no filtering. The old fingerprint-less 51k
  rows mix with any new-fingerprint absolute rows a pair campaign also emits (brief §2
  explicitly allows this for absolute-BCE pretraining) — this is "train on old-fingerprint
  absolute rows + new pairs" from the brief's ask, and is the default because it never
  silently drops 50k rows of already-paid-for signal.
* `--absolute-fingerprint-mode new_only --new-fingerprint <fp>`: keep only absolute rows whose
  `meta.player_fingerprint == fp` (old rows, lacking the field, are dropped) — "new-only" from
  the brief's ask, for a run that wants a policy-consistent absolute+pair corpus end to end.
* `--pair-fingerprint-allow <fp1,fp2,...>` (default: allow every fingerprint found): pairs
  always carry the field per the frozen schema, so this is a plain allow-list — useful once
  more than one pair campaign (different fast-player versions) exists and a run wants to
  pin to one.

Both filters run only when loading from `--shards`/`--pair-shards` (inside `load_data`/
`load_pair_data`), never when a caller injects pre-split `data=`/`pair_data=` tuples directly
(same precedent as the existing `--label-clip`, which is also `load_data`-only) — so tests that
build datasets by hand aren't surprised by filtering they didn't ask for.

Changing any of these three flags, or `pair_shards`, `lam_rank`, `tau`, `pair_weight_cap`, or
`ece_guardrail`, IS allowed on a `--resume` (added to `_RESUME_KEYS`, same precedent as the
pre-existing `holdout_frac`) — but like `holdout_frac`, changing what's IN the training set on
a resume is a deliberate override, not covered by the bit-exact guarantee (that guarantee is
for a plain `--resume` with nothing overridden — §4).

---

## 4. Preserved: bit-exact resume, PAUSE, no-pairs equivalence

**No pairs configured is bit-identical to the pre-lever-(b) trainer.**  `load_pair_data`
leaves `train_pairs`/`holdout_pairs` as `PairDataset.empty()` when `--pair-shards` isn't given;
`train_step` only builds/adds the pairwise term `if len(self.train_pairs)`, so with no pairs
there is no extra RNG draw, no extra forward/backward pass, and `loss` is exactly `bce_loss` —
the SAME computation the original trainer did (`test_no_pairs_configured_is_unaffected` pins
`"pair_loss" not in rec` and `"pairs" not in m`).  The pre-existing pinned resume test in
`test_train_v.py` (dummy-only, no pairs) is untouched.

**Bit-exact resume extended to pair batches**
(`test_resume_is_bit_exact_with_pair_batches`, mirrors the existing pinned test's structure
exactly but with `--pair-shards` configured): pairs get their OWN resumable
epoch-permutation/cursor stream, exactly parallel to the absolute-row one —
`pair_epoch`/`pair_cursor` fields, `_pair_epoch_order(epoch)` seeded
`np.random.default_rng([seed, epoch, _PAIR_ORDER_SALT])` (`_PAIR_ORDER_SALT = 991_301`,
an arbitrary constant whose only job is to keep this permutation stream from ever coinciding
with the absolute-row one's `[seed, epoch]`, even when both epoch counters line up).  Both are
persisted in `VTrainer.state()` and restored in `from_checkpoint` (`.get(..., 0)` for
backward compatibility with checkpoints saved before this round — an old checkpoint simply
resumes with `pair_epoch=pair_cursor=0`, which is correct since it never had pairs to begin
with).  The test resumes a 60-step run from its step-30 checkpoint and asserts weights, ALL
Adam moments, `pair_epoch`, AND `pair_cursor` are identical to a straight 60-step run — the
same standard the original test holds the absolute-row path to.

**PAUSE / `.DONE`**: untouched (`run()`'s stop/checkpoint logic doesn't know pairs exist; the
pair-batch cursor rides along in every checkpoint exactly like the absolute one already did).

---

## 5. Smoke (brief §6.4)

* **Dummy net, tiny overfit** (`test_overfit_tiny_pair_set_to_pair_accuracy_near_one`): 16
  hand-built pairs with a fixed, cleanly-separable feature gap (`x[3] = +0.5` / `-0.5`,
  `delta=0.3`, `delta_ci=0.05` — everything resolved), `holdout_frac=0` (all pairs in train,
  since the ask is "overfit a tiny pair set," not a held-out generalization claim — kept
  distinct from the held-out metric definition in §1), `lam_rank=5.0`, 300 steps at `lr=1e-2`.
  **Result: pair_acc 1.00 (16/16 resolved)** — full command output below.
* **Gradient flow through both branches, dummy** (`test_gradients_flow_through_both_branches_dummy`):
  one `_pair_loss()` + `.backward()` in isolation (no BCE term) → every parameter of the dummy
  net has a nonzero-norm gradient.
* **Gradient flow through both branches, real net** (`test_set_value_net_pair_loss_grad_flow`,
  skips if `mcts.value_net`/`encoder_v2` aren't importable — they are, on this box): a
  real-shaped pair batch (actual `encoder_v2` obs from `labels.sample_states`) through the
  actual `SetValueNet`, `_pair_loss()` alone, `.backward()`.  Every parameter gets a nonzero
  gradient EXCEPT the per-item-type heads (`joker_mlp`/`cons_mlp`/`shelf_mlp`/`pack_mlp`/
  `blind_mlp`) whose item set is empty in every row of the batch on both branches — that's a
  property of the encoder's masked mean/max pooling (an empty set pools to a mask-independent
  constant, so that head's output never reaches the loss), not of the pairwise loss; the test
  computes which sets are actually populated in the sampled fixture and only requires gradient
  for the rest (in this run: `joker_mlp` was the one legitimately-empty head — the scripted
  fallback policy used for the fixture never buys jokers in 4 early snapshots; `blind_mlp` and
  every core shared component — `card`/`aux_table`/`key_embed`/`set_embed`/`scalar_proj`/
  `trunk_in`/`res_blocks`/attention/`value_head` — all get gradient, since a decision state
  always has a hand and blind offers).
* **Both nets briefly on synthesized fixtures**: covered by
  `test_train_step_and_eval_report_pair_metrics` (dummy, 40 steps, checks the full eval-record
  shape including `pairs.pair_acc`/`by_pair_source`/`by_state_kind`) and
  `test_set_value_net_pair_loss_grad_flow` (real net, 1 step's worth of forward/backward,
  CPU, small `net_cfg` — matches `test_train_v.py`'s existing real-net smoke pattern in scope
  and cost).

Real-net GPU load: none — all real-net tests here run CPU with a tiny `net_cfg` (32/64/1/32/8),
matching the existing `test_set_value_net_round_trip_and_one_step` pattern; no full 5M-param
GPU smoke was run for this round (the box is shared, ops cap ≤ 8 workers / short bursts — a
real GPU pair-loss smoke belongs in the idle-box run alongside the full retrain, brief §8).

---

## 6. Test count, deviations, open items

**Tests**: 17 new (`test_train_v_pairs.py`) + 8 unchanged (`test_train_v.py`) = 25 in this
workstream's scope, all green.  Full `ev` suite: green as of this writing modulo unrelated,
in-flight churn from concurrent workstreams editing shared files this round (W-EXTRACT's
`hand.py`/engine changes, W-PAIRS's `pairs.py`/`test_pairs.py`) — verified each failure seen
during this build traced into `ev/hand.py`/`ev/tests/test_extraction.py` (not this
workstream's files) via traceback inspection before moving on; re-run `python -m pytest ev`
at merge time for a final clean read.

**Deviations from the brief, with rationale**:

1. **Physical pair-shard format is this module's own choice, not literally specified** — the
   brief freezes field NAMES and says "same storage as labels," not a file extension. Chose
   `.npz` (dataset.py's convention) over raw JSON/JSONL because it turned out to be exactly
   what W-PAIRS also converged on independently (§2) — a JSON-compat reader is kept as a
   defensive fallback but is not the expected common path.
2. **`evaluate_pairs`'s `pair_loss` does not feed `VTrainer.best`** — `best` stays BCE-only
   (unchanged selection criterion) rather than silently redefining what "best checkpoint"
   means; `pair_acc`/`pair_loss` are fully reported every eval regardless.
3. **ECE guardrail is a soft warning + a logged boolean, not a hard stop** — the brief says
   "calibration must not silently degrade," which a visible flag satisfies; a hard stop would
   block a run of a config someone is deliberately sweeping past its calibration cost.
4. Did not verify against `pairs.pair_job`'s full producer output end-to-end (a real rollout
   campaign) — only against its shard-WRITING interface (`pairs.PairRow`/`save_pair_shard`),
   which is the actual boundary this workstream needs; running `pair_job` for real is W-PAIRS's
   proof-campaign scope (brief §5.5), not this one's.

**Open for the lead / idle-box round (brief §8)**: retune `lam_rank`/`tau`/`pair_weight_cap`
against REAL pair deltas once the full campaign lands (defaults here are principled but
un-fit); confirm `--pair-shards ev/runs/pairs_s1/shards` loads with zero changes needed
(expected green per §2, not yet run against real campaign output since none existed during
this build); decide whether `ece_guardrail`'s default (0.05) needs adjusting once real V-v2
holdout ECE is measured with pairs in the loss.
