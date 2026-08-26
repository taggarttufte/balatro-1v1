# active_poc — NOTES (W-ACTIVE, 2026-08-25)

**Question.** Per label spent, does actively selecting WHICH states to label (PLR /
active-learning style) buy more V quality than uniform sampling?

**Status.** A measurement POC, not an integration. Everything in this package is NEW.
`labels.py`, `dataset.py`, `train_v.py`, `workers.py`, `player.py` and the engine are used
strictly as libraries and are **not modified** — the only files this workstream touched are
the ones listed below. Nothing is committed.

Results: `results/active_poc_2026-08-25.md` / `.json`.

**Answer, in one line.** A qualified no at this scale: both active rules beat uniform
nominally (1.2–1.4x BCE per label) but neither significantly over 6 paired seeds (t = −1.29,
−1.79, df = 5); once acquisition cost is counted only *disagreement* is viable (free to
score, and the only arm that improves AUC — 0.7614 vs uniform's 0.7573), while the *error
proxy* costs 2.47x the rollouts and demonstrably ranks on label noise (it ranked on a mean
|V − y_probe| of 0.594 whose realized |V − y8| is 0.272, and its labels are 6% wider-CI and
15% lower-variance than uniform's). Disagreement's one striking behaviour: unprompted, it put
31% of its budget on ante ≥ 5 states, against uniform's 19% and the corpus's 17.6% — it found
the thin tail by itself.

> The results markdown mixes generated tables with hand-written prose (Verdict, §3
> interpretation, §4 note, §5 caveat, §6, §7, §8). Re-running `report.py` regenerates the
> tables and **drops the prose**; diff before overwriting.

---

## 0. What is where

| file | what |
|---|---|
| `jobs.py` | the two pool jobs: `pool_job` (candidate states + cheap probe label), `arm_job` (label selected states at `n_rollouts=8`); `CORPUS_CONFIG`, `snapshot_rng_seed`, `obs_fingerprint` |
| `corpus.py` | base/holdout construction, seed-hash subsampling, `fresh_seeds`, `canonical_seed`, `drop_holdout_seeds`, `concat` |
| `select.py` | the three acquisition rules, per-STATE scoring, light stratification, overlap table |
| `training.py` | the one training recipe (`RECIPE`), `train_one`, `derive_regime`, ensemble scoring, `evaluate_full` |
| `bench.py` | cost probe (s/rollout, s/self-play) + the reconstruction pin |
| `gen_pool.py` | stage 1 CLI: build the candidate pool (8 workers, resumable) |
| `stage_base.py` | stage 2 CLI: base corpus → regime probe → 3-member ensemble (GPU, sequential) |
| `stage_select.py` | stage 3 CLI: score the pool, pick the three arms, define the hard stratum |
| `gen_arms.py` | stage 4 CLI: label the UNION of the arms (8 workers, resumable) |
| `stage_final.py` | stage 5 CLI: retrain per arm over paired seeds, evaluate, emit the results JSON |
| `report.py` | results JSON → the markdown write-up |
| `tests/test_active_poc.py` | 16 tests, ~6 s (`python -m pytest ev/active_poc/tests -q` from `ev`) |

Run dirs live under `ev/runs/active_poc/` (gitignored by `.gitignore`'s `runs/`+`*.pt`).

### The pipeline, end to end

```bash
python ev/active_poc/gen_pool.py    --seeds 600 --workers 8 --n-probe 2 --minutes 52
python ev/active_poc/stage_base.py                        # GPU, ~3 min
python ev/active_poc/stage_select.py --arm-states 1200    # GPU, ~2 min
python ev/active_poc/gen_arms.py    --workers 8 --minutes 70
python ev/active_poc/stage_final.py --seeds 6 --arm-only-seeds 3
python ev/active_poc/report.py
```

Stages 1 and 4 are resumable (re-run the same command; `touch <run-dir>/PAUSE` to stop) and
carry wall-clock deadlines, because they are the only expensive parts.

---

## 1. Design decisions (and why)

**Base corpus = a seed-hash subsample of `labels_full`, not a fresh corpus.** The existing
51k corpus is the only large body of labels that exists; regenerating 12k labels would have
cost ~1.5 h for nothing. Subsampling is by a *seed* hash (salt `w-active-base`, frac 0.262),
so whole seeds move together and no seed straddles base/holdout. Result: 12,264 rows / 511
seeds, which makes a ~2.4k arm addition a visible ~20%.

**The evaluation holdout is the standard one, untouched.** `dataset.seed_in_holdout(seed,
0.1)` on the 51k corpus = 5,160 rows / 215 seeds. No arm trains on it, and the candidate
pool's seeds are filtered through the same rule *before* any labelling (see §3 for the bug
this caught).

**Selection is per STATE, not per row.** A snapshot yields two rows (both perspectives) from
ONE rollout set — `label_both` derives `y1 = 1 − y0` exactly. So the label budget is spent
per state and both rows come free; a per-row selector would "buy" a row whose partner it did
not pay for. Each state's score is the mean of its two perspectives' scores.

**Candidate pool uses the corpus config verbatim.** `jobs.CORPUS_CONFIG` is
`results/labels_full.json`'s `config` copied field for field (EV player, `budget=fast`,
`shop_tier=rules`, `epsilon_selfplay=0.1`, `epsilon_rollout=0.02`, `n_states=12`, `max_ante=12`,
Red deck, stake 1, 4 lives). This consistency is load-bearing: the arms must differ only in
WHICH states were chosen, never in how the states were generated or how they were labelled.
The pool seeds are fresh (not in the corpus's `default+random:2000`) and non-holdout.

**The error proxy's probe rollouts use an INDEPENDENT seed stream.** `label_both` draws
rollout seeds `seed * 1_000_003 + i`, so if the probe shared the arm's seed base its 2
rollouts would be a literal subset of the arm's 8. The error-proxy arm would then be
selecting states whose label noise persists into the very label it is judged on. `jobs.
PROBE_ROLLOUT_SEED` gives the probe its own base, which keeps the proxy an honest (noisy)
second opinion and lets the noise-chasing effect appear at its true size rather than an
inflated one.

**Light stratification over `state_kind`.** Unstratified top-k lets one kind monopolise an
arm, and the kinds differ systematically in label sd (nemesis 0.343 vs other 0.293 in the
51k corpus), so "most uncertain" would partly mean "the kind with the noisiest labels" and
the comparison would measure kind mix rather than acquisition. Each kind is capped at
`cap_mult = 1.5 ×` its share of the pool; states fill in global score order subject to the
caps, then the arm is topped up on raw score if the caps left it short. The uniform control
is deliberately NOT stratified — its kind mix is the pool's natural mix, which is the honest
baseline.

**One labelling pass over the UNION of the three arms.** A state chosen by two arms is
labelled once and shared. Cheaper, and a tighter control: an overlapping state cannot differ
between arms through label noise. Overlap is logged (`arms.json → overlap`).

**Arms are compared PAIRED across training seeds.** This was not optional. The three base
ensemble members, identical but for their seed, spread over 0.0055 BCE — comparable to the
*entire* expected effect of adding 2.4k rows to 12.3k (the 12.3k → 45.9k learning curve
implies ~0.004 BCE for a 20% data increase). A single run per arm could not have separated
signal from seed noise. Because `base` rows are always concatenated FIRST and every arm has
the same row count, a given seed produces identical initial weights and an identical epoch
permutation across arms — the arms differ only in which rows sit in the last K slots — so the
paired difference cancels most of that noise. Deviation from the brief's "retrain one V per
arm": 6 seeds per arm instead of 1. It is cheap (~15 s of GPU each) and it is the difference
between a measurement and a coin flip.

**Fixed step count, no per-arm early stopping.** `stage_base` re-derives `S*` for the 12k
corpus by a probe run (`S* = 325`, epoch ~7 — consistent with `v_full_best`'s best at epoch 7
on 45.9k rows) and every ensemble and arm run then trains for exactly `S*` steps with the
same cosine schedule. No arm gets a private early-stopping decision, so the reported holdout
numbers carry no per-arm selection bias.

**A secondary "arm-only" comparison.** Adding 2.4k rows to 12.3k is a ~20% data change whose
total effect is small; training on the arm's rows ALONE removes the dilution and shows the
acquisition rules' relative teaching value much more clearly, at the cost of being a
different (smaller-data) regime. Its step count is derived on the *uniform* arm, so it cannot
favour a treatment.

---

## 2. Deviations from the brief

| deviation | rationale |
|---|---|
| brief read as `docs/PHASE5_BRIEF_2026-08.md` | `PHASE5_V2_BRIEF_2026-08.md` does not exist on `mp/campaign` |
| worktree branch reset to `mp/campaign` | the worktree was created from `main` (59588ba), which predates `mp/`; the tree was clean, and only this throwaway worktree branch moved |
| arm = ~1,200 states = ~2,400 **rows** | the brief says both "~3k states each" and "a ~3k addition [to] ~12k rows"; rows is the reading consistent with the base-corpus framing, and states are what the budget is actually spent on |
| 6 paired training seeds per arm, not 1 | seed noise (0.0055 BCE) is the same size as the whole effect; see §1 |
| candidate pool 600 seeds (~7.2k states), not ~20k | the pool's cost is entirely the error proxy's 2 rollouts per state; ~20k states would have been ~2 h on its own. 1,200 of ~7.2k is a 17% selection rate, which is contrast enough |
| base corpus read from the MAIN checkout's `runs/labels_full/shards` | run dirs are gitignored, so they do not exist in this worktree; the shards are static, read-only data (12 MB), and copying them would have bought nothing |
| tests live in `ev/active_poc/tests/`, not `ev/tests/` | `ev/pytest.ini` has `testpaths = tests`, so the existing W5 gate collects exactly what it did before and this POC cannot perturb another workstream's suite |
| `arm_job` given the RAW seed string, rows keyed by the canonical one | see §3 |

---

## 3. Gotchas (the ones that cost real time)

**`sample_states` is not reproducible across processes.** `labels.sample_states` defaults its
reservoir RNG to `hash((seed, policy_seed)) & 0xFFFFFFFF`. `hash()` of a *str* is salted per
process (PYTHONHASHSEED is unset here), so the same seed yields a DIFFERENT set of 12
snapshots in a different process. This does not affect the 51k corpus (it never re-derives a
snapshot set), but the POC's whole design does: `pool_job` scores states that `arm_job` must
later re-derive. `jobs.snapshot_rng_seed` replaces the default with a sha1 of the same pair
and is passed explicitly by both jobs. `arm_job` additionally verifies each state against the
`obs_fingerprint` recorded by `pool_job` and refuses to label one that drifted — belt and
braces, and it is what caught the next item.

**The engine canonicalises seed strings: `'0' → 'O'`.** Balatro's seed alphabet has no zero,
so `rng.core.normalize_seed` maps it (`'DH0HXASZ'` → `'DHOHXASZ'`), and `game.seed_str` — the
canonical form — is what lands in a shard's `seed` column and therefore what the holdout hash
rule is applied to. Two consequences, both real bugs, both caught by assertions rather than by
being noticed:

1. *Holdout leak.* `fresh_seeds` originally tested `seed_in_holdout` on the RAW generated
   string. ~20% of random 8-char seeds contain a `0`, and ~10% of those canonicalise into the
   evaluation holdout — so the pool contained 5 holdout seeds (120 rows). `corpus.fresh_seeds`
   now canonicalises before testing, and `stage_select` additionally drops any holdout seed
   from the pool before selection can see it (`drop_holdout_seeds`), which is what was done
   for the already-generated pool rather than spending 25 min regenerating it.
2. *Reconstruction mismatch.* `pool_job` was driven with the raw strings, so its reservoir was
   seeded from the raw string while the shard recorded the canonical one. Feeding `arm_job`
   the canonical seed from `arms.json` would have re-derived a different snapshot set, and
   every state of every seed containing a `0` (24 of 135 seeds in the smoke run) would have
   failed the fingerprint check and been dropped. `gen_arms.py` rebuilds the canonical → raw
   map from the pool's `done.ids` (whose job ids *are* the raw strings) and passes the raw
   seed in the payload while keeping the canonical seed as the row identity.

  For a clean future run this is moot: `fresh_seeds` now returns already-canonical seeds, so
  raw == canonical and the map is the identity.

**`nohup … &` inside a backgrounded tool call double-detaches.** The completion notification
fires when the launching shell exits, not when the work finishes; poll the run dir instead.

**Checkpoints are 60 MB each.** 29 training runs × (numbered + `latest`) would be ~3.5 GB, so
`stage_final` deletes a run's `.pt` files once it has been evaluated. `train.jsonl` keeps the
curve and every run is reproducible from its seed.

---

## 4. Measured cost (this box, 8 workers, shared)

| thing | measured |
|---|---|
| self-play one seed (12 snapshots) | ~1.5 s |
| one rollout | 1.0–1.6 s (contention-dependent; 1.37–1.42 s during the pool) |
| `pool_job` (12 states × 2 probe rollouts) | ~33 s |
| candidate pool, 8 workers | ~320 rows/min |
| arm labelling, `n_rollouts=8` | ~8 s of worker time per state |
| one V training run (325 steps, 5M params, batch 256) | 12–16 s GPU + ~40 s load/startup |
| ensemble scoring of the pool (3 members) | seconds |

Ops caps honoured throughout: ≤ 8 label workers, GPU runs strictly sequential, GPU memory
checked before each stage (2.9 GB of 12.3 GB in use by the desktop; no stage added more than
~1.5 GB).
