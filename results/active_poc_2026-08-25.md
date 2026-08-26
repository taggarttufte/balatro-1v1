# Active label selection vs uniform sampling — a measurement POC

*W-ACTIVE, 2026-08-25T02:07:38.  Phase 5 rev 2 (EV player), branch `mp/campaign`.*

**Question.** Per label spent, does actively choosing WHICH states to label buy more V quality than sampling states uniformly?

## Verdict

* **disagreement vs uniform:** ΔBCE -0.0012 ± 0.0010 (paired over 6 training seeds, t = -1.29); ΔBrier -0.00052; ΔAUC +0.0041.
* **error-proxy vs uniform:** ΔBCE -0.0021 ± 0.0012 (t = -1.79); ΔBrier -0.00071; ΔAUC -0.0023.

**A qualified no at this scale — with one recommendation.** Both active rules beat uniform
nominally (1.2–1.4x BCE improvement per label), and **neither difference is significant**
over 6 paired training seeds: |t| would have to reach 2.57 for p < 0.05 at df = 5, and the
observed values are 1.29 and 1.79. What the experiment *can* resolve is about ±0.0024 BCE
(2 SEM); the arm-vs-uniform gaps are 0.0012–0.0021, i.e. right at the edge. All three arms
beat the no-addition baseline decisively (t = 3.0–7.4), so the labels are clearly worth
buying — the open question is only whether *choosing* them beats *not* choosing them.

Three findings are solid enough to act on, because they do not depend on the marginal
BCE gaps:

1. **Once acquisition cost is counted, only disagreement is even a candidate.** Scoring by
   ensemble disagreement is free (three forward passes per state). The error proxy needs a
   2-rollout label on *every state in the pool*, so its 1.38x per-label gain costs **2.47x
   the rollouts** — a net loss of roughly 44% on a compute basis (§2, "value per ROLLOUT").
2. **The two rules improve different things.** Disagreement produces the best *ranking* of
   all four arms (AUC 0.7614, +0.0041 vs uniform, +0.0094 vs base). The error proxy produces
   by far the best *calibration* (ECE 0.0178 vs 0.0303) while slightly *hurting* ranking
   (ΔAUC −0.0023). Averaging them into a single "is active learning better" verdict hides
   this.
3. **Disagreement re-weights toward the under-sampled late game on its own.** 31% of its
   states are ante ≥ 5, against 19% for uniform and 16% for the error proxy — and ante ≥ 5 is
   only 17.6% of the existing 51k corpus, which is exactly where V has the least data. It
   found the corpus's thin tail without being told to.

**Recommendation.** Carry disagreement selection into the pair campaign as a cheap option
(§7), on the strength of free acquisition, the AUC gain, and that mechanism — but do not
budget against a 1.2x label-efficiency assumption, because this POC did not establish it.
Drop the error proxy: it is more expensive than the labels it saves and it demonstrably
ranks on label noise (§3).

## 1. Design

| | |
|---|---|
| base corpus | 12,264 rows (6,132 states), a seed-hash subsample of the existing 51k `labels_full` corpus |
| evaluation holdout | 5,160 rows — the STANDARD `seed_in_holdout(seed, 0.1)` holdout, never trained on by any arm |
| candidate pool | 7,020 fresh self-play states (585 seeds), same self-play config and same label policy as the corpus |
| arm size | 1,197 states = 2,394 rows (+20% on the base) |
| labels | `n_rollouts=8`, standard `labels.label_both` pipeline, unchanged |
| training | identical recipe for every arm, 325 steps (re-derived for this corpus size), 6 paired seeds per arm |

Both perspectives of a snapshot come from one rollout set, so the budget is spent per STATE and selection is per state (each state's score is the mean of its two perspectives).

### Acquisition rules and what they picked

| arm | rule | mean disagreement | mean error-proxy | kinds |
|---|---|---|---|---|
| disagreement | sd of 3-member ensemble V | 0.0819 | 0.3307 | blind_select 171, hand 186, nemesis 225, other 136, pack 288, shop 194 |
| error-proxy | \|mean V − 2-rollout label\| | 0.0461 | 0.5933 | blind_select 209, hand 205, nemesis 194, other 188, pack 199, shop 205 |
| uniform (control) | seeded random | 0.0435 | 0.2952 | blind_select 210, hand 178, nemesis 197, other 202, pack 213, shop 200 |

Where in the run each rule looks (states by ante):

| arm | ante 1 | ante 2 | ante 3 | ante 4 | ante 5 | ante 6 | ante 7 | ante 8 | ante ≥5 share |
|---|---|---|---|---|---|---|---|---|---|
| disagreement | 40 | 180 | 277 | 326 | 244 | 107 | 21 | 5 | **31%** |
| error-proxy | 207 | 355 | 246 | 200 | 139 | 44 | 6 | 3 | **16%** |
| uniform (control) | 193 | 266 | 248 | 264 | 169 | 49 | 10 | 1 | **19%** |

Pool-wide: disagreement mean 0.0426 (p90 0.0738), error-proxy mean 0.2947 (p90 0.5382).  **corr(disagreement, error-proxy) = 0.147** — the two rules are nearly orthogonal.

Overlap between the selected sets:

| pair | shared states | Jaccard |
|---|---|---|
| disagreement ∩ err_proxy | 240 | 0.111 |
| disagreement ∩ uniform | 220 | 0.101 |
| err_proxy ∩ uniform | 209 | 0.095 |

## 2. Three-arm results (full holdout, mean ± SEM over training seeds)

| arm | train rows | BCE ↓ | Brier ↓ | AUC ↑ | ECE ↓ | hard-stratum BCE ↓ |
|---|---|---|---|---|---|---|
| base only (no addition) | 12,264 | 0.6300 ± 0.0009 | 0.07022 ± 0.00030 | 0.7520 ± 0.0008 | 0.0396 ± 0.0022 | 0.6565 ± 0.0022 |
| disagreement | 14,658 | 0.6232 ± 0.0011 | 0.06782 ± 0.00042 | 0.7614 ± 0.0005 | 0.0306 ± 0.0027 | 0.6512 ± 0.0016 |
| error-proxy | 14,658 | 0.6223 ± 0.0008 | 0.06763 ± 0.00032 | 0.7549 ± 0.0010 | 0.0178 ± 0.0019 | 0.6502 ± 0.0018 |
| uniform (control) | 14,658 | 0.6244 ± 0.0013 | 0.06833 ± 0.00046 | 0.7573 ± 0.0006 | 0.0303 ± 0.0026 | 0.6516 ± 0.0018 |

Constant predictor: BCE 0.6931 / Brier ~0.1004.  Label-noise Brier floor ≈ 0.0154.

### Paired differences (same training seed, same base rows in the same order)

| comparison | ΔBCE | ΔBrier | ΔAUC | ΔECE | Δhard-BCE | t (BCE) |
|---|---|---|---|---|---|---|
| disagreement vs uniform | -0.0012 ± 0.0010 | -0.00052 | +0.0041 | +0.0003 | -0.0004 | -1.29 |
| err proxy vs uniform | -0.0021 ± 0.0012 | -0.00071 | -0.0023 | -0.0125 | -0.0014 | -1.79 |
| disagreement vs base only | -0.0068 ± 0.0013 | -0.00240 | +0.0094 | -0.0090 | - | -5.33 |
| err proxy vs base only | -0.0077 ± 0.0010 | -0.00259 | +0.0030 | -0.0218 | - | -7.37 |
| uniform vs base only | -0.0056 ± 0.0018 | -0.00188 | +0.0053 | -0.0093 | - | -3.02 |

Negative = better for BCE / Brier / ECE, positive = better for AUC.

### Value per label

What 2,394 extra rows bought over the 12,264-row base, and how each acquisition rule compares to spending the same budget uniformly:

| arm | ΔBCE vs base-only | per 1,000 labels | vs uniform |
|---|---|---|---|
| disagreement | -0.0068 ± 0.0013 | -0.00284 | 1.22x |
| error-proxy | -0.0077 ± 0.0010 | -0.00321 | 1.38x |
| uniform (control) | -0.0056 ± 0.0018 | -0.00233 | 1.00x (reference) |

The multiplier is a ratio of two small, noisy differences — read its sign and rough magnitude, not its decimals.

### Value per ROLLOUT — what the acquisition itself costs

Ranking the pool is not free, and the two rules differ enormously in what it costs. Scoring by ensemble disagreement needs three forward passes per state (milliseconds). Scoring by the error proxy needs a real 2-rollout label on **every state in the pool**, whether or not that state is ever selected — a quarter of a full label each.

| arm | acquisition rollouts | labelling rollouts | total | vs uniform |
|---|---|---|---|---|
| uniform (control) | 0 | 9,576 | 9,576 | 1.00x |
| disagreement | 0 | 9,576 | 9,576 | 1.00x |
| error-proxy | 14,040 | 9,576 | 23,616 | 2.47x |

At the measured ~1.28 s/rollout on 8 workers that is 26 min for either free-to-score arm and 63 min for the error-proxy arm. The error-proxy arm therefore has to beat uniform by 2.5x on quality merely to break even on compute — and the probe cost scales with the POOL, so it gets worse the more selective you try to be.

## 3. Per-arm label noise

| arm | rows | mean CI half-width | Brier noise floor | label sd | truncated |
|---|---|---|---|---|---|
| disagreement | 2,394 | 0.2339 | 0.01483 | 0.3319 | 0.000 |
| error-proxy | 2,394 | 0.2514 | 0.01686 | 0.2731 | 0.000 |
| uniform (control) | 2,394 | 0.2373 | 0.01526 | 0.3204 | 0.000 |

### Does either signal track V's real error, or just label noise?

The selected states carry an 8-rollout label `y8` from a rollout stream independent of the 2-rollout probe, so the proxy can be decomposed on real data: `err_proxy = |mean V − y_probe|` is what the rule ranked on, `|mean V − y8|` is a far better estimate of V's actual error.

* **corr(err_proxy, |V − y8|) = 0.328**
* **corr(disagreement, |V − y8|) = 0.149**
* corr(err_proxy, disagreement) = -0.080

| arm | ranked-on err_proxy | realized \|V − y8\| | ensemble disagreement | label sd (y8) |
|---|---|---|---|---|
| disagreement | 0.3312 | 0.2548 | 0.0819 | 0.3319 |
| error-proxy | 0.5935 | 0.2719 | 0.0462 | 0.2731 |
| uniform (control) | 0.2953 | 0.2062 | 0.0435 | 0.3204 |

Mean realized \|V − y8\| over all labelled states: 0.2327.

**The error proxy chases noise, and the size of it is measurable three ways.**

1. **Most of what it ranked on evaporated.** It selected states on a mean `|V − y_probe|` of
   0.594; the realized `|V − y8|` on those same states is 0.272. Roughly 54% of the ranked
   magnitude was the probe's own noise, not V's error.
2. **Its labels are noisier.** Mean CI half-width 0.2514 vs uniform's 0.2373 (+6%); Brier
   noise floor 0.01686 vs 0.01526 (+10%).
3. **Its labels are less informative.** Label sd 0.2731 vs uniform's 0.3204 (−15%). A
   2-rollout estimate is maximally noisy exactly where the true P(win) is near 0.5, so
   "V disagrees with the probe" preferentially selects near-0.5 states — where the label
   carries the least information, being close to the prior.

**But it is not *only* noise, and the honest version of that is more interesting.** Its
realized error (0.272) really is above uniform's (0.206), and `corr(err_proxy, |V − y8|) =
0.328` exceeds `corr(disagreement, |V − y8|) = 0.149`. Two caveats keep that from being a win
for the proxy: (a) `err_proxy` and `|V − y8|` share the `V` term, which inflates their
correlation, whereas disagreement contains no `y` at all — the two correlations are not
like-for-like; (b) `|V − y8|` still contains `y8`'s own noise, which is largest for precisely
the near-0.5 states the proxy prefers, so part of its realized-error advantage is aleatoric
inflation rather than model error.

The coherent reading is that **the error proxy buys calibration by feeding V a diet of
near-0.5 targets** — a form of confidence regularisation — rather than by finding states V
gets wrong for learnable reasons. That is exactly the signature in §2: best ECE and best BCE,
worst AUC of the three arms. The two acquisition signals are essentially orthogonal
(`corr = −0.08`), which is consistent with them measuring different things: epistemic
uncertainty about the state versus disagreement with a noisy sample of its label.

§5 seals it. Trained on its own rows alone, the error-proxy set yields a V with **AUC 0.503 —
no ranking ability whatsoever** — against uniform's 0.618. A selected set is a biased set;
it works only as a *supplement* to a representative corpus, never as a replacement.

## 4. Per state-kind held-out BCE

| arm | blind_select | hand | nemesis | other | pack | shop |
|---|---|---|---|---|---|---|
| base only (no addition) | 0.6273 | 0.6233 | 0.6371 | 0.6584 | 0.6241 | 0.6102 |
| disagreement | 0.6229 | 0.6160 | 0.6293 | 0.6534 | 0.6160 | 0.6019 |
| error-proxy | 0.6224 | 0.6171 | 0.6268 | 0.6495 | 0.6147 | 0.6038 |
| uniform (control) | 0.6218 | 0.6179 | 0.6325 | 0.6504 | 0.6194 | 0.6050 |

The hard stratum is the top 25% of holdout rows by the same ensemble's disagreement (1,290 of 5,160 rows, sd ≥ 0.0585); it is fixed once, from the base ensemble, so it is identical for every arm.

**Where a rule helps tracks exactly where it over-sampled.** Disagreement's per-kind result
lines up with its per-kind allocation almost monotonically — it beats uniform on the four
kinds it took *more* of (pack +75 states, 0.6160 vs 0.6194; nemesis +28, 0.6293 vs 0.6325;
hand +8, 0.6160 vs 0.6179; shop −6, 0.6019 vs 0.6050) and loses on the two it took *fewer*
of (blind_select −39, 0.6229 vs 0.6218; other −66, 0.6534 vs 0.6504). So the gain is a
*re-allocation* effect, not a "these are magically better labels" effect — which is also the
argument for keeping the stratification cap: unconstrained, the rule would starve the kinds
it finds boring, and the losses would eventually outrun the wins.

The hard-stratum column tells the same story more weakly: every arm improves on the base
there, but the arm-vs-uniform differences (−0.0004 ± 0.0015 and −0.0014 ± 0.0014) are well
inside noise. Selecting high-disagreement *training* states did not measurably help on
high-disagreement *holdout* states.

## 5. Secondary: trained on the arm's rows ALONE (no base corpus)

Removes the dilution of adding 2,394 rows to 12,264; 25 steps, 3 seeds per arm.

*Caveat first:* the small-data regime bottoms out at the very first eval point (25 steps of
2,394 rows) and all three arms land at AUC 0.50–0.62 against 0.75 for the full runs, so these
are weak, undertrained models. Read this table as "is a selected subset representative enough
to train on by itself", not as a precise efficiency measurement.

| arm | BCE ↓ | Brier ↓ | AUC ↑ |
|---|---|---|---|
| disagreement | 0.6898 ± 0.0011 | 0.09874 | 0.5665 |
| error-proxy | 0.6960 ± 0.0018 | 0.10184 | 0.5031 |
| uniform (control) | 0.6832 ± 0.0018 | 0.09546 | 0.6178 |

* disagreement vs uniform: ΔBCE +0.0067 ± 0.0015 (t = 4.33), ΔAUC -0.0512
* err proxy vs uniform: ΔBCE +0.0129 ± 0.0031 (t = 4.21), ΔAUC -0.1147

## 6. What this POC cannot conclude

1. **It says nothing about playing strength.** Every number here is held-out calibration of
   V. No tournament was run. A −0.001 to −0.002 BCE difference has not been shown to move a
   win rate, and on the evidence of the Phase-5 smoke tournament the mapping from V quality
   to match results is not gentle. Only `tournament_v.py` / `h2h.py` can answer that.
2. **Neither arm-vs-uniform difference is statistically significant.** t = −1.29 and −1.79
   at df = 5. This is "not detectable at this scale", not "no effect", and equally not
   "a 1.2x effect exists". Six more seeds would roughly halve the SEM; that is the cheapest
   available strengthening (~15 s of GPU each) and it was not done here for budget reasons.
3. **One round, one corpus size, one arm size.** Real active learning is iterative
   (score → label → retrain → rescore) and its gains are usually reported as compounding over
   rounds and growing with the pool-to-arm ratio. This is a *single* round from a fixed 12k
   model at a 17% selection rate — close to the least favourable setting for the active arms.
   A negative result here does not transfer to a 5-round campaign.
4. **The effect is small by construction.** Adding 2,394 rows to 12,264 is a ~20% data
   increase worth about −0.0056 BCE in total (measured, uniform arm). Any arm-vs-arm gap is a
   fraction of that. The design deliberately traded effect size for a realistic base-corpus
   scenario; a smaller base would have produced larger, cleaner differences that generalised
   less.
5. **The disagreement signal is weak by design.** Three members trained on the *same* data
   differ only through initialisation and batch order. That is the cheapest possible epistemic
   estimate; bootstrapped or differently-regularised members would give a stronger one, and
   might change the ranking.
6. **"Realized error" `|V − y8|` still contains label noise.** `y8` has a CI half-width of
   ~0.24, so a state whose true P(win) is 0.5 shows a large `|V − y8|` even against a perfect
   V. Cleanly separating epistemic from aleatoric error needs *repeated* 8-rollout labels on
   the same state, which was outside the budget. §3's decomposition is therefore directional,
   not exact.
7. **Everything is conditional on the frozen label policy and state distribution.** The pool
   was generated with the same `EVPlayer(budget="fast", shop_tier="rules")` self-play and the
   same label definition as the 51k corpus — deliberately, so the arms differ only in
   selection. If the pair campaign changes the rollout policy, the ε, or the state
   distribution, the ranking of acquisition rules can change with it.
8. **The pool is not the state distribution the trained agent will visit.** States were drawn
   from ε = 0.1 self-play of the *current* player. Active selection interacts with
   distribution shift, and nothing here probes that.

## 7. Integration sketch — the minimal hook for the pair campaign

Only the disagreement rule is worth wiring in (§Verdict). The hook is small because
`sample_states` already over-produces candidates internally.

**Where.** Not in `sample_states` — in the *job*. `labels.label_job` currently samples
`n_states` snapshots and labels all of them; the change is to sample more and label the best.

```python
# labels.label_job — new optional payload keys, default off (oversample=1 == today's behaviour)
#   "oversample": 4              # sample 4x n_states, label the best n_states
#   "ensemble":   [ckpt, ...]    # 2-3 V checkpoints; absent -> uniform, as today
#   "cap_mult":   1.5            # per-kind cap, as in active_poc/select.py
#   "uniform_frac": 0.5          # keep half the arm uniform (see the guard rail below)

snaps = sample_states(seed, n_states=n_states * oversample, ..., rng_seed=<explicit>)
if oversample > 1 and ensemble:
    snaps = select_by_disagreement(snaps, ensemble, n_states,
                                   cap_mult=cap_mult, uniform_frac=uniform_frac)
# ... unchanged from here: label_both(s.match, n_rollouts=8, ...) per surviving snapshot
```

`select_by_disagreement` is ~30 lines and already exists in this package in all but name:
encode both perspectives (`labels.make_encoder("v2")`), run the K nets, score each state as
the mean of its two perspectives' sd, then `active_poc.select.stratified_topk`.

**Cost.** Essentially free, which is the whole point.
* Extra self-play: **none.** One match already yields every candidate; a higher `n_states`
  only keeps more `match.clone()`s (0.14 ms each) in the reservoir. Memory, not time.
* Scoring: K forward passes per candidate perspective. At the measured 4.2 ms single-state V
  latency on CPU (2 threads, `EV_NOTES` §5.4), 4x oversample x 12 states x 2 perspectives x
  3 nets ≈ 288 forwards ≈ **1.2 s against a ~77 s job — under 2% overhead.**
* Compare with the error proxy, which would have added ~150 s per job.

**Where the ensemble comes from.** No extra training: take the last 2–3 `ckpt_*.pt` of the
running V training as a snapshot ensemble. Their disagreement is a weaker signal than
independently-trained members but it costs nothing and is already on disk.

**Ops.** Each worker holds K x 5M-param nets (~60 MB each): at 16 workers and K = 3 that is
~2.9 GB of RAM. Load them **once** in a module-level cache from `workers.worker_init`, never
per job, and pin `torch.set_num_threads(1)` (already done there).

**Two guard rails, both earned by the measurements above.**
1. **Keep a uniform fraction** (start at 50/50). §5 showed that a purely selected corpus is
   not representative enough to train on — the arm-only V's AUC collapsed. Active selection
   is a supplement to a uniform spine, not a replacement for it.
2. **Keep the per-kind cap.** §4 showed the gain is a re-allocation effect: the kinds a rule
   over-samples improve and the kinds it starves get worse. Uncapped, the losses will
   eventually outrun the wins.

**How to know it worked, at campaign scale.** Run the pair campaign's first N seeds both ways
(the `oversample=1` default is the control) and compare held-out BCE **and AUC** — AUC is
where the disagreement rule showed its clearest edge here, and it is the metric a
`tournament_v` result should track. At campaign scale (≥ 50k labels rather than 2.4k) the
effect this POC could not resolve should be within reach.


## 8. Cost, throughput and reproduction

Measured on the shared box (RTX 3080 Ti, 8 label workers, GPU runs strictly sequential):

| stage | work | throughput | wall |
|---|---|---|---|
| candidate pool | 600 seeds x 12 states x 2 probe rollouts | 352 rows/min | 41 min |
| base + ensemble | 1 probe + 3 members, 5M params | 12–16 s per 325-step run | 3 min |
| selection | score 14,040 pool rows with 3 members | seconds | 2 min |
| arm labelling | 2,967 union states x 8 rollouts | **102 labels/min**, 1.13 s/rollout | 58 min |
| final | 24 arm runs + 9 arm-only + 1 probe | ~1 min per run incl. startup | 33 min |

The arm-labelling rate of ~102 labels/min at 8 workers is below the ~144/min the 51k campaign
implies (288/min at 16 workers). The gap is per-rollout cost — 1.13 s here against ~0.81 s
there — not pool overhead: the box was shared throughout, and selected states sit later in a
match than the corpus average, so their rollouts are not shorter.

Failures: 0 job failures across 1,176 pool + arm jobs. 7 of 2,967 union states were not
labelled — 6 whose step was absent from the re-derived snapshot set and **1 genuine
reconstruction drift** caught by the `obs_fingerprint` guard (`WTNWY1BG` step 19), 0.03% of
the union. All arms were trimmed to a common 1,197 states.

Reproduction: `ev/active_poc/NOTES.md` §0 has the five commands in order. Stages 1 and 4
are resumable. Run dirs are under `ev/runs/active_poc/` (gitignored).

**Files created** (all new; nothing existing was modified, nothing committed):

* `ev/active_poc/` — `jobs.py`, `corpus.py`, `select.py`, `training.py`, `bench.py`,
  `gen_pool.py`, `stage_base.py`, `stage_select.py`, `gen_arms.py`, `stage_final.py`,
  `report.py`, `NOTES.md`, `tests/test_active_poc.py` (16 tests, ~6 s, all green)
* `results/active_poc_2026-08-25.md` (this file) and `.json` (every number above)

> Sections "Verdict", §3's interpretation, §4's note, §5's caveat, §6 and §7 are written by
> hand; the tables are generated by `report.py` from the JSON. Re-running `report.py`
> regenerates the tables and **drops the prose** — edit, or diff, accordingly.
