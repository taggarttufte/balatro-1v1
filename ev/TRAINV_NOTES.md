# TRAINV_NOTES — Phase 5 rev 2, W5: labels, race calculator, V trainer, worker pool

Stage 1 built 2026-08-23 (before W3's `player.py` landed; W1's `encoder_v2`/`value_net` and W2's
`clone_determinized` landed mid-build and are used directly).  Stage 2 (gate 4 numbers) is
appended below when the lead gives the go.

---

## 0. What is where

| file | what |
|---|---|
| `ev/race.py` | the lives-race calculator: `p_win`, `curve_from_history`, `Curve`, `race_table` |
| `ev/labels.py` | snapshots (`sample_states`), rollouts (`rollout`), labels (`label_state`, `label_both`), the pool job (`label_job`) |
| `ev/dataset.py` | label shards (`.npz`), `LabelDataset`, split-by-seed, batches, summary |
| `ev/workers.py` | resumable spawn pool over pure functions (`run_pool`), PAUSE file, `--bench` |
| `ev/train_v.py` | V trainer (`VTrainer`, `run`, CLI), metrics, checkpoints, PAUSE/resume, `.DONE` |
| `ev/scripts/gen_labels.py` | the label campaign driver (seeds → jobs → shards → `results/labels_<name>.json`) |
| `ev/tests/test_{race,labels,workers,train_v}.py` | 44 tests, ~21 s |

Run dirs live under `ev/runs/` (gitignored by `.gitignore`'s `runs/` + `*.pt`).  Shards go
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

**Phase-5 rev-1 `agent/parallel/**` + `train/parallel.py`: kept, untouched.**  Label
rollouts carry no net (the rollout policy is analytic), so its shared-memory leaf transport is
irrelevant here; deleting it is a separate decision for the lead (its 28 + 17 tests still pass
in their own suites).

---

## 4. The trainer

`python ev/train_v.py --shards <dir|glob…> --run-dir ev/runs/<name> [--max-steps N]
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

## 5. Stage 2 — reduced scale (2026-08-23, box shared with Tagg: 4 workers / 4 threads / 60-min caps)

The ≥ 50k campaign and the 30-seed tournament are DEFERRED to an idle box; §6 has the exact
commands.  Everything below ran at reduced scale — the V numbers are a PIPELINE PROOF on
tiny data (1,152 labels), not a result.

### 5.1 Rollout policy (frozen for the label definition) + cost

`EVPlayer(budget="fast", epsilon=0.02, seed=<per-rollout>)`, **W3 built-in shop rules**
(`shop_tier="rules"`).  W4's `stats=` tier measured 4x cheaper but much weaker (self-play
mean final ante 3.9 vs 6.1 on 10 seeds, both sides same tier) — selectable via
`shop_tier="stats"` but NOT the label definition.  Self-play ε = 0.1 for snapshot diversity.

Measured in the real campaign (4 workers): **2.51 s/rollout**, 180 decisions/rollout,
**policy = 94 %** of rollout time (engine step+clone ≈ 1 ms/decision; the fast player's
matches end at ante 5–6 on lives, so no ante-12 truncations occurred).  Hotspot (cProfile,
one rollout = 5.7 s): W3's shop/pack rule tier — `player.py:380 _rank_shop_rules` /
`:487 _rank_booster_rules` call `player.py:140 build_proxy` per candidate → `hand.py:210
board_ratio` → `hand.py:613 BlindModel.__init__` → `hand.py:843 _score_plays` (1,837 model
builds = 58 % of the rollout; shop 9 ms/dec, pack 27 ms/dec, hand 3 ms/dec).  A per-shop-visit
model cache in W3 would roughly double label throughput.

**Two W3 issues worked around on W5's side** (W3's files untouched):
1. ε-exploration wedge: `player.py:196-199` seeds its ε-RNG from `sampling.world_rng`
   (`sampling.py:45-53`), whose key omits shop/consumable contents — the same "random" pick
   repeats while the state key is unchanged, so a legal-but-no-op pick (`use_consumable`
   Wheel of Fortune with no editionless joker: `consumables.py:171-184` returns False →
   `game.py:1927` no-op) loops forever (observed: 39,952 consecutive shop steps).  W5 builds
   `EVPlayer(epsilon=0)` and applies ε from its own sequential `random.Random(seed)`
   (`labels._with_epsilon`).
2. Belt and braces: `labels._Guard` forces a progress action after 3 consecutive
   signature-unchanged steps (`forced` counted in every label's meta; 0 across the whole
   campaign after fix 1).  Also `player.py:360-363 _v` swallows value_fn exceptions →
   `MatchAwareEVPlayer` counts and re-raises them instead (`n_errors`).

Clairvoyance guard note (W2 gotcha): the guard never reads `game.determinized`/`det_seed`
(which do not survive a later plain `.clone()`) — `rollout()` calls `clone_determinized`
itself and flags from that call, plus an assert on the fresh clones.

### 5.2 The small label set (`results/labels_s2_small.json`)

`gen_labels.py`, seeds `default+random:400`, 4 workers, 58-min cap → **1,152 labels** (48
seeds x 12 snapshots x 2 perspectives, n_rollouts 8) at **20.1 labels/min** incl. 12.5 %
symmetry overhead (steady ≈ 22.6/min at 4 workers).  The driver process was killed right at
the cap before its exit summary — the crash-safe design held (ids recorded only after shard
writes; 6 shards + 48 done.ids intact, ~4 in-flight jobs lost); summary regenerated with
`--max-jobs 0`.

Labels: mean 0.498, sd 0.317; per-kind means all 0.495–0.501 (blind_select/hand/nemesis/
shop/pack/other n = 172–204 each); sd RISES with ante (0.21 @1 → 0.39 @5 — later states are
more decided); mean CI half-width 0.238 (8 rollouts); truncation fraction 0.000.
Sum-to-one with INDEPENDENT rollout sets: **mean(y0+y1) = 0.965 ± 0.025** (72 pairs).
Held-out at the 0.1 hash rule: 5 seeds / 120 rows.

### 5.3 V training (pipeline proof — 1k rows, ~600 epochs, overfits by design)

`train_v.py`, CUDA, batch 256, lr 3e-4 cosine: **~60 ms/step** after a one-off ~7-min
first-250-steps warmup (first CUDA run only; resumes did not repeat it).  Held-out (120
rows): best **BCE 0.674 vs constant 0.693**, **Brier 0.083 vs 0.103** (noise floor 0.015),
**AUC 0.80**, acc@0.5 0.63 vs 0.44, ECE 0.11–0.12, reliability roughly monotone across 9
occupied bins.  Verified on this run: PAUSE file → clean stop mid-run at step 4532
(`stop_reason PAUSE`, checkpoint written, no `.DONE`); resume 1000→3000 continued the
schedule; 5M checkpoint round trip **bit-exact** (weights + Adam moments + layout
fingerprint) — and pinned by `test_train_v.py` for the continuation.

### 5.4 End-to-end match play

`ev/match_player.py::MatchAwareEVPlayer(net, encoder, ...)` — binds a mutable
`opponent_view(match, player)` into the `value_fn(game)` closure and refreshes it from the
live match before every `act` (all clones evaluated during one decision share that view).
`.policy()` gives the `(match, p, acts)` form for `play_out`/`play_1v1`; single-state V
latency 4.2 ms CPU (2 threads) / 13.7 ms CUDA (launch-bound).
`ev/scripts/tournament_v.py`: paired by seed, both seat orders per seed, pool-parallel,
resumable, Wilson 95% CI → `results/tournament_v_<name>.json`.  Both policies are wrapped
in the same `_Guard` — the first smoke hit a V-tier shop loop (W3's `_rank_with_value` has no
anti-cycling): seed 1KV4W6YS burned 40k steps / 637k V calls / 35 min before the step cap;
with the guard the same seed finishes in 5 s with ONE intervention (`guard_forced` is in
every match record).

### 5.5 Smoke tournament (6 seeds x 2 seats, `results/tournament_v_v_s2_smoke.json`)

`EVPlayer(value_fn=V @ step 4532)` vs `EVPlayer(value_fn=None)`: **V won 0/12** (Wilson 95%
[0%, 24%]), 0 V errors, ~600–1,000 V calls per match.  NO CLAIM intended or possible: a
1,152-label overfit V REPLACES the hand-tuned shop/blind rule tier wholesale (that is how
W3's tiering works), so losing to the rules is the expected outcome of the pipeline proof.
The full-scale question is (iii) in §6.

## 6. The deferred full-scale commands (lead launches on an idle box)

```bash
# (i) ≥ 50k labels, 16 workers, ~9 h (measured 22.6 labels/min at 4 workers -> ~90/min at 16;
#     2,126 seeds x 24 labels = ~51k; resumable: same command; pause: touch <run-dir>/PAUSE)
python ev/scripts/gen_labels.py --run-dir ev/runs/labels_full     --seeds default+random:2000 --workers 16 --policy ev --budget fast --shop-tier rules     --encoder v2 --n-states 12 --n-rollouts 8 --flush-jobs 32 --symmetry-jobs 24 --name full

# (ii) V on the full set (~50k rows, ~100 epochs; ~25 min GPU after the one-off warmup)
python ev/train_v.py --shards ev/runs/labels_full/shards --run-dir ev/runs/v_full     --model set_value_net --max-steps 20000 --batch-size 256 --lr 3e-4 --warmup-steps 500     --eval-every 500 --checkpoint-every 2000 --device cuda --holdout-frac 0.1 --torch-threads 8

# (iii) the 30-seed paired tournament (60 matches; ~10 min at 16 workers, ~35 min at 4)
python ev/scripts/tournament_v.py --checkpoint ev/runs/v_full/latest.pt     --seeds default:30 --workers 16 --threads 1 --name v_full
```
