# TRAINV_NOTES — Phase 5 rev 2, W5: labels, race calculator, V trainer, worker pool

Stage 1 built 2026-08-23 (before W3's `player.py` landed; W1's `encoder_v2`/`value_net` and W2's
`clone_determinized` landed mid-build and are used directly).  Stage 2 (gate 4 numbers) is
appended below when the lead gives the go.

---

## 0. What is where

| file | what |
|---|---|
| `mp/ev/race.py` | the lives-race calculator: `p_win`, `curve_from_history`, `Curve`, `race_table` |
| `mp/ev/labels.py` | snapshots (`sample_states`), rollouts (`rollout`), labels (`label_state`, `label_both`), the pool job (`label_job`) |
| `mp/ev/dataset.py` | label shards (`.npz`), `LabelDataset`, split-by-seed, batches, summary |
| `mp/ev/workers.py` | resumable spawn pool over pure functions (`run_pool`), PAUSE file, `--bench` |
| `mp/ev/train_v.py` | V trainer (`VTrainer`, `run`, CLI), metrics, checkpoints, PAUSE/resume, `.DONE` |
| `mp/ev/scripts/gen_labels.py` | the label campaign driver (seeds → jobs → shards → `mp/results/labels_<name>.json`) |
| `mp/ev/tests/test_{race,labels,workers,train_v}.py` | 44 tests, ~21 s |

Run dirs live under `mp/ev/runs/` (gitignored by `mp/.gitignore`'s `runs/` + `*.pt`).  Shards go
under the run dir too (`<run>/shards/*.npz`), so nothing large is ever staged.

---

## 1. The label (frozen target)

`y = P(player wins the MLB match | full match state)`, estimated by `n_rollouts` (8) determinized
rollouts of the analytic policy on BOTH sides:

1. `match.clone_determinized(seed_i)` (W2) — both games re-seeded with the SAME fresh seed
   (same-seed MP), draw orders reshuffled, future RNG streams fresh.  No true seed to peek at.
2. fresh policies per side: `EVPlayer(budget="fast", epsilon=0.02, seed=…)` (W3; scripted greedy
   fallback while W3 is missing — `policy="scripted"`, never for real labels).
3. play until `match.done` → outcome 1/0, or `min(ante) > 12` → `race.p_win(curve_0, curve_1,
   lives_0, lives_1, ante=13)` with curves fitted from the rollout's own `pvp_log`.
4. mean over rollouts; CI = Wilson 95% half-width for binary outcomes, normal approximation
   when race values are mixed in.

Both perspectives of a snapshot are two rows from ONE rollout set (`y1 = 1 − y0` exactly:
one winner per finished match; the race model is symmetric to 1e-9).  The symmetry SANITY
check therefore uses independent rollout sets (`independent_perspectives` jobs /
`--symmetry-jobs`): smoke run, 18 pairs, scripted policy: mean(y0 + y1) = 1.014 (sd 0.23).

`label_job` refuses to run without W2's `clone_determinized` (clairvoyant rollouts) unless
`allow_clairvoyant=True` (plumbing tests only).

### Where the states come from — decision

In-process snapshots, not log replay.  `sample_states(seed)` self-plays one match with the
same policy (`epsilon_selfplay=0.1` for diversity) and keeps `match.clone()` (0.14 ms) at
reservoir-sampled decision steps, stratified over `STATE_KINDS = (blind_select, hand, nemesis,
shop, pack, other)` (`other` = ROUND_EVAL / cash-out etc. — single-action states where the EV
player still evaluates V as a leaf).  Every row's meta carries `(seed, step, actor, kind, ante,
selfplay config)`; `reconstruct_snapshot(seed, step, …)` replays to the exact state
(`test_labels.py` pins `signature()` equality).  The replay logger is not needed.

---

## 2. The race calculator — model and assumptions

`p_win(my_curve, their_curve, my_lives, their_lives, ante, *, cfg, blinds_done=0)`:

* Nemesis score at ante `a` is log-normal: `log10 S ~ N(mu(a), sigma(a))`, independent across
  antes and players.  `Curve` = per-ante `(mu, sigma)` table + a linear `slope` (log10/ante)
  beyond it (both directions).
* Ante = Small → Big → Boss slot.  Small/Big are regular blinds for both players independently:
  `P(fail) = P(S < target)` with the engine's `blind_base_chips` table (stake scaling,
  Plasma `ante_scaling`), clipped to `[0.01, 0.995]`.  A failed regular blind costs ONE life
  and the run proceeds (MLB).  The ante-1 Boss (before `pvp_start_round = 2`) is a regular
  blind.  From ante 2 the Boss slot is the Nemesis: `P(I lose) = Φ((mu_t − mu_m) / sqrt(σ_m² +
  σ_t²))`; optional tie mass `p_tie` (default 0) costs nobody.
* ≤ 1 life per blind (engine `lose_life` round blocker).  Both at 0 in the same blind = 0.5.
* Not modelled: comeback money, skips/tags, boss kind, the opponent affecting my score.  The
  fitted curve is meant to absorb them.
* Solved by DP over `(ante, phase, my_lives, their_lives)` for `horizon_antes = 24`, closed with
  the exact negative-binomial race `Σ_{k<my} C(their−1+k, k) q^their p^k` at the last ante's
  Nemesis probability.  Past ante ~12 endless targets grow super-exponentially while a linear
  curve does not, so regular blinds fail w.p. → 1 for both and the chain ends on its own.
* `curve_from_history(pvp_log, player, ante)`: last 4 Nemeses (= the opponent block's history),
  zeros (deck-outs) dropped when other points exist, OLS in (ante, log10 score) with slope
  clipped to `[0.05, 1.5]`, sigma = residual sd shrunk to 0.35 with 2 pseudo-obs, floor 0.15;
  1 point → default slope 0.35 through it; 0 points → `prior_curve` (log10 Boss target + 0.15).

Sanity numbers (prior curve at ante 3): equal curves 4v4 **0.500**; 4v3 **0.662**, 3v4 0.338;
4v1 **0.945**; me +0.3 log10 (2× score) 4v4 **0.937**; me +0.3 at 2v4 0.647; both collapsing at
ante 13: 4v4 0.5, 4v3 0.75, 4v2 0.998.  Closed form 4v3 at p=½ = 21/32.  ~0.3 ms per call.

---

## 3. The worker pool

`run_pool(fn, jobs, n_workers, on_result, pause_file, checkpoint_every, on_checkpoint,
state_path, max_jobs, deadline_s)` — spawn context (Windows), `worker_init` imports
`_bootstrap` and pins BLAS/torch to 1 thread, callback-driven reaping (no polling cap), a
bounded in-flight window (2× workers) so a PAUSE is honoured within ~one job per worker,
per-job exceptions isolated (logged, counted, not marked done → retried next run), `n_workers=0`
runs inline.  Completed ids persist one-per-line; `gen_labels.py` manages that file itself so an
id is written only AFTER the shard with its rows is on disk.

Dummy-job benchmark (6 ms engine jobs): 1 worker 7.0k jobs/min, 4 workers 25k jobs/min, pool
overhead ~3.5 ms/job (incl. spawn start-up amortised over 6 s).  Real `label_job` with the
scripted fallback + encoder v2, 4 workers: **~500 labels/min**, 101 ms/rollout, 52 decisions
per rollout (scripted players die at ante 2–3), policy = 76 % of rollout time.  The EV-player
numbers are Stage 2.

**Phase-5 rev-1 `mp/agent/parallel/**` + `train/parallel.py`: kept, untouched.**  Label
rollouts carry no net (the rollout policy is analytic), so its shared-memory leaf transport is
irrelevant here; deleting it is a separate decision for the lead (its 28 + 17 tests still pass
in their own suites).

---

## 4. The trainer

`python mp/ev/train_v.py --shards <dir|glob…> --run-dir mp/ev/runs/<name> [--max-steps N]
[--minutes M] [--device cuda]`; `--resume <run-dir|ckpt> [--max-steps N …]` (only explicitly
given flags override the checkpoint's config).

* BCE-with-logits on soft labels, AdamW (lr 3e-4, wd 1e-4), warmup 200 + cosine to 5 % (or
  flat), batch 256, grad-clip 1.0, optional `--label-clip`.
* Held-out by SEED (`seed_in_holdout`: sha1 hash rule, stable across shards/runs; or
  `--holdout-seeds`).  Every `--eval-every` steps: BCE, Brier, AUC (labels binarised at 0.5),
  ECE + 10-bin reliability curve, accuracy, all vs the constant predictor (train mean), plus
  the label-noise Brier floor `mean((ci/1.96)²)`; per-kind BCE/Brier.
* Checkpoints `latest.pt` + `ckpt_<step>.pt` (pruned to `--keep`) in W1's
  `value_net.save_checkpoint` format with `extra["trainer"]` = optimizer state, step/epoch/
  batch cursor, numpy/torch/python/cuda RNG, config, eval history, holdout seeds.  The epoch
  permutation is `default_rng([seed, epoch])` so a resume draws the batch the interrupted run
  would have: `test_train_v.py` pins weights AND Adam moments bit-exact (resume from step 30 →
  60 equals a straight run to 60).
* PAUSE file between steps (and Ctrl+C / SIGTERM): checkpoint, exit; `--resume` deletes it.
  `.DONE` only on `max_steps` / `max_epochs`.  `train.jsonl`: config / eval / checkpoint /
  summary records; console one line per eval.
* Model kinds: `set_value_net` (W1, 4,996,789 params) and `dummy` (16-scalar MLP) for the
  plumbing tests.  Measured: one training step of the real net at batch 256 on the 3080 Ti =
  **32 ms** (~1.9k steps/min) → 50k labels × 20 epochs ≈ 4k steps ≈ 2 min; training is not the
  bottleneck, labelling is.

---

## 5. Stage 2 (gate 4) — to be filled in after the lead's go

- [ ] EV-player rollout cost (ms/rollout, decisions/rollout, policy vs engine share)
- [ ] ≥ 50k labels campaign: `labels_<name>.json` (mean/sd by kind & ante, CI widths, symmetry)
- [ ] V training: held-out BCE/Brier/AUC vs constant, reliability curve, PAUSE/resume verified
- [ ] end-to-end tournament `EVPlayer(value_fn=V)` vs `EVPlayer(value_fn=None)`, ≥ 30 seeds paired
- [ ] the exact launch command for the long campaign
