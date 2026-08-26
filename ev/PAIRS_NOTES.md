# PAIRS_NOTES — W-PAIRS: lever (b)'s data (Phase 5 rev 2, 2026-08-25)

Files: `ev/pairs.py` (new), `ev/scripts/gen_pairs.py` (new driver), tests
`ev/tests/test_pairs.py` (30 tests, ~13 s), campaign outputs `ev/runs/pairs_s1/`,
results `results/pairs_s1.json` + `results/pairs_s1_diag_coupling.json`.
**Nothing owned by another workstream was touched** — `labels.py`, `dataset.py`,
`workers.py`, `hand.py`, `player.py`, `train_v.py`, `gen_labels.py` are all read-only from
here (the driver *imports* `gen_labels.parse_seeds`, the shared seed-spec parser).

---

## 0. Headline — **the lever is 1.78×, not ≥ 2×, on a uniform state mix. Say it loudly.**

| | value |
|---|---|
| **variance-reduction factor, CRN estimator, n = 1,301 pairs** | **1.78×** (mean ρ +0.468) |
| **variance-reduction factor, direct replication audit, n = 54** | **2.00×** |
| … restricted to `hand` + `nemesis` states, n = 560 | **2.54×** (ρ +0.629); direct 2.62× |
| … `shop` / `pack` / `blind_select` | 1.42× / 1.80× / 1.23× |
| campaign | 1,301 pairs + 2,602 absolute rows, 232 seeds, 29 shards |
| throughput | **22.2 pairs/min at 8 workers** (shared box) → **~44 pairs/min at 16** |

The brief's rule is "if it's < 2×, stop and say so loudly": **on the uniform state mix the
label campaign samples, it is 1.78× and the lever does not clear the bar.** The reason is
not the horizon and not the coupling order (both measured, §4): it is that **the correlation
lives almost entirely at hand decisions.** Restricted to `hand`/`nemesis` states — which is
where the measured argmax-V failure lives (per-action hand EV gaps ≪ 0.05, brief §0) — the
lever *does* clear the bar at **2.54×**, and on `close_call` pairs specifically at 2.34×.
So the honest reading is: **the lever works where the policy failure is and does not work at
shop/pack/blind-select decisions**, and the decision the lead has to make is whether the
full campaign weights sampling toward hand states (§7.1).

For scale: at `n_worlds = 8` the paired estimator's standard error on Δ is **0.153** vs
**0.204** for two independent absolute labels at the same 16 rollouts — a real but modest
gain against per-action gaps of ≪ 0.05. Only **5.7 %** of pairs are resolved
(`|delta| > delta_ci`) at 8 worlds; W-RANK's `|delta|/delta_ci` weighting is doing the heavy
lifting, exactly as designed.

---

## 1. What a pair is

Exactly TRAINV_NOTES §1's label semantics, applied twice on ONE shared world list:

1. `labels.sample_states(seed)` (unchanged machinery) → a decision snapshot `s` with an
   actor `p`, stratified over `STATE_KINDS`.
2. `choose_pair` picks two legal actions `a`, `b` (§2). Branch **a is always the rules
   player's own choice**, so `delta > 0` reads as "the rules player was right".
3. `m_a = s.match.clone(); m_a.step(p, a)` and likewise `m_b` — both are REAL states. The
   immediate transition consumes the true stream exactly as it would in the match, so
   `obs_a` / `obs_b` are honest encoder-v2 observations of states V has to value.
4. One world-seed list `w_1..w_n` (n = `n_worlds`, default 8). For each `w_i`,
   `labels.rollout(m_x, seed=w_i, …)` — which `clone_determinized(w_i)`-es BOTH games with
   the same fresh seed and plays `EVPlayer(budget="fast", ε=0.02)` on both sides to
   `match.done` (1/0) or `ante > 12` (`race.p_win`). The same `w_i` also seeds both
   branches' rollout policies, so the ε-streams are common random numbers too.
5. Outcomes are stored **from the actor's perspective** (`p0_win` if actor 0 else `1−p0_win`)
   — V's target for the actor's own observation. `delta = mean(a) − mean(b)`; `delta_ci` =
   1.96·sd(per-world differences)/√n, i.e. **the CI of the paired difference**, not of either
   branch.

Every pair also emits the two absolute rows `(obs_a, mean(outcomes_a))` and
`(obs_b, mean(outcomes_b))`: pairs do double duty as labels (brief §5.3).

### The one ordering decision, and its diagnostic

The brief freezes **step-then-determinize** (§5.1: "post-action match clones, roll BOTH …
on the SAME `clone_determinized` seed list"). The alternative, `determinize_then_step`
(determinize once per world, then step both branches on clones of that ONE world), couples
strictly more — the immediate draw and the post-action pile are shared too — but then a
branch state is a *sampled* world, one per world, so there is no single `obs_a` to store and
V would be trained on a resampled hand it will never see. It is implemented anyway
(`roll_pair(coupling=…)`, `COUPLINGS`) and run as a **diagnostic** in §4: it separates
"the lever is weak" from "the coupling is weak".

---

## 2. Pair selection (`pair_source`)

`close_call` — the rules player's top-2 when the fast-EV gap < `close_gap` (0.03).
`greedy_vs_extract` — the rules choice vs the best extraction line (W-EXTRACT).
`random` — the rules choice vs a uniformly drawn other legal action.

Target mix 50 / 40 / 10 (`DEFAULT_MIX`), drawn per snapshot by `mix_sequence`.

**Fallback (documented deviation from a literal reading of the brief).** A requested source
that a state cannot supply — a `close_call` whose top-2 gap is ≥ 0.03, a
`greedy_vs_extract` on a board with nothing to extract — **cascades to the other
informative source before degrading to `random`**; a requested `random` never cascades (it
is the deliberate 10 % of uninformed pairs, not a fallback bucket). Both the requested and
the realised source are stored (`meta.requested_source`), and `mix_report` prints the full
requested→realised cross-tab. Rationale: without the cascade every failed close call became
a random pair, and the first burst realised 47/53 close_call/random instead of anything near
the target.

**Three hard facts about the realised mix**, all measured, none fixable from this side:

* `other` (ROUND_EVAL / cash-out) states have exactly **one** legal action and cannot host a
  pair at all. The sampler is told not to spend snapshots on them (`PAIRABLE_KINDS`), and
  `pair_job` skips any snapshot with < 2 legal actions (counted in `skipped`).
* **Extraction only exists at `SELECTING_HAND` and only on a board with procs.**
  `hand.HandAnalysis.extract_on` is `bool(cfg.extract) and not self.pvp and model is not
  None and (procs.any or has_card_proc or _tarot_wants)` — so it is off at every Nemesis, at
  every shop / pack / blind-select state, and on a vanilla b_red board with no seal / Lucky /
  Business-Card / tarot-target. `greedy_vs_extract` therefore cannot reach 40 % on
  uniformly sampled states; §6 reports what it does reach.
* The EV scale differs by state kind. `close_gap = 0.03` is measured in the rules player's
  own units: `P(clear) + 0.012·E[hands unused]` at a hand/nemesis state (range ~0…1.05) but
  a *proxy gain* at a shop/pack state (range ~±0.02) and a near-binary 1.0 vs −1.0 at
  blind_select. So a shop pair is almost always a "close call" and a blind_select pair
  almost never is. The threshold is applied literally as the brief specifies; the per-kind
  close-call rate is in the results JSON.

### W-EXTRACT feature detection (the hook)

`extraction_entry_point()` probes, in order: a line **generator**
(`hand.extraction_lines` / `extraction_candidates` / `extraction_actions`, then the same
names on `EVPlayer`), then the per-action **EV term** (`hand.extraction_ev`). A generator
may return actions, `(action, ev)` pairs or `(action, ev, reason)` triples and may or may
not take `legal=`; all forms are accepted, and anything that raises is treated as "not
landed" rather than killing the campaign. Two notes from the build:

* the term's argument order is **`(game, action)`** in `hand.py` while the brief writes
  `extraction_ev(action, state)`; `_try_call` tries both (and moves on for *any* exception,
  not just `TypeError` — passing a dict where a game is expected raises `AttributeError`);
* the module-level `hand.extraction_ev` rebuilds a whole `HandAnalysis` per call (1–3 ms ×
  up to 762 legal actions = **seconds** per decision). When the same term exists as a
  `HandAnalysis` **method**, the hook builds ONE analysis and scans with it (0.1 ms for the
  whole scan) — `_ev_term_lines`. The legal scan is capped at `max_scan = 800`.

W-EXTRACT landed `hand.extraction_ev` at 00:11 and `hand.extraction_lines(game, legal=None,
*, cfg) -> [(action, ev, reason)]` at 00:20 on 2026-08-25, explicitly documented as "the
interface for W-PAIRS's `greedy_vs_extract` pair source". The hook picks the generator up
automatically; the EV-term path stays as the fallback and is still tested (with the
generator names hidden).

One semantic caveat: the extraction lines are also inside `HandAnalysis.evaluate()`, so the
rules player's own top-1 can BE the best extraction line. The hook is called with
`avoid = the greedy choice`, so in that case the pair is greedy vs the *second* best
extraction line — still a real extract-vs-extract-less contrast, never an action against
itself, but not literally "greedy vs the best line".

---

## 3. The frozen shard schema (brief §5.3) and how it is stored

One JSON record per pair, exactly the frozen field set:

```
{"kind":"pair","seed","step","actor","state_kind","ante","player_fingerprint","pair_source",
 "action_a","action_b","n_worlds","outcomes_a","outcomes_b","delta","delta_ci","meta"}
```

`obs_a` / `obs_b` are **not** inside that JSON — they are stacked npz arrays, *exactly the
way the existing label shards store `obs`* (`dataset.save_shard` puts `obs__<key>` outside
`meta_json` for the same reason). A pair shard is one compressed `.npz`, written atomically
(temp + `os.replace`) like `dataset.save_shard`:

| npz key | content |
|---|---|
| `version`, `shard_kind`, `obs_keys` | `1`, `"pair"`, the encoder's key list |
| `obs_a__<key>`, `obs_b__<key>` | `(N, …)` stacked, one row per pair |
| `PAIR_COLUMNS` | typed columns: `seed step actor state_kind ante pair_source player_fingerprint n_worlds delta delta_ci` |
| `pair_json` | one frozen record per row (STRICT json — see below) |

**W-RANK interop (checked 2026-08-25).** W-RANK built its own mirror of this layout
concurrently and then reconciled to it (`RANK_NOTES` §"W-PAIRS's actual `pairs.py` landed
mid-build"): same `obs_a__<key>` / `obs_b__<key>`, same typed scalar columns, `delta` /
`delta_ci` as top-level float32 arrays, and its blob column renamed to `pair_json` holding
the full frozen record. Its `test_reads_pairs_py_shards_directly` calls the REAL
`pairs.save_pair_shard`, and its `PairDataset.load` was pointed at this campaign's real
`ev/runs/pairs_s1/shards` as a final end-to-end check (§6).

**Schema-collision notes for W-RANK** (frozen names kept, collisions documented, nothing
renamed):

1. The label shards' column is `kind` (the *state* kind) and their `meta["kind"]` likewise;
   the frozen pair record uses `"kind": "pair"` for the RECORD type and `"state_kind"` for
   the state. Both are kept verbatim. A pair shard is therefore *not* loadable by
   `dataset.load_shard` — use `pairs.load_pair_shard` / `pairs.PairDataset`. The two live in
   different directories for exactly this reason (§5).
2. The label shards' `player` column is 0/1 (two perspectives of one snapshot). A pair's two
   absolute rows are BOTH the actor's perspective, so their `player` column equals `actor`
   for both, and `meta` carries `branch` ("a"/"b"), `from_pair`, `pair_source`, `action` and
   `post_action: true`. Their `step` is the DECISION step (pre-action) so the seed/step
   grouping still matches the pair record; the state they encode is the post-action one.
3. **No NaN/Inf ever reaches a shard.** `ev_b` / `ev_gap` are genuinely unknown when branch
   b is outside the ranked head; they are stored as `null`, not `NaN`, because a non-finite
   float round-trips unequal under `json` and is rejected outright by strict parsers.
   `save_pair_shard` writes with `allow_nan=False` so a leak fails loudly.
4. `meta` carries everything needed to reconstruct and to audit: the `selfplay` config
   (`labels.reconstruct_snapshot(seed, step, …)` replays the snapshot), the `rollout` config
   (policy / budget / shop_tier / ε / max_ante / encoder / deck / stake / lives / coupling),
   `world_seeds`, the per-pair variance decomposition (`var_a var_b var_d cov_ab rho`), the
   replicate means (`reps`, `rep_means_a/b`), the selection detail (`requested_source`,
   `ev_a`, `ev_b`, `ev_gap`, `n_ranked`, `n_legal`, `close_gap`), `y_a/y_b`, `ci_a/ci_b`,
   `trunc_frac`, `determinized`, `forced`, `lives`.

### `player_fingerprint`

New field, no prior convention to collide with. Format
`"<policy>-<budget>-<shop_tier>:<12 hex>"`, e.g. `ev-fast-rules:f4439c51…`. The digest is
sha1 over **the fast player's source** (`hand.py`, `player.py`, `sampling.py`) plus the
knobs that select its behaviour (`policy`, `budget`, `shop_tier`, `epsilon_rollout`,
`has_extraction()`). Brief §2's requirement is "any change to the fast player changes the
label/rollout policy" — hashing the source is the only way to get that automatically, and it
means **W-EXTRACT landing flips the fingerprint by construction**. The old 51k corpus
(`ev/runs/labels_full*`) carries no such field at all, so "old policy" is identifiable by
its absence. Cached per process (`functools.lru_cache`) at first use, so a worker's
fingerprint always describes the code that worker actually imported.

---

## 4. THE MEASUREMENT THAT JUSTIFIES THE LEVER

`pairs.variance_report(records, n_worlds=n)` reports the empirical variance of the paired
estimator against two independent absolute labels **at equal rollout budget** (2n rollouts
either way), by two independent routes.

**(a) `crn` — every pair, free.** Per pair with per-world outcomes `a_i`, `b_i` on shared
worlds, `s²_d = var(a_i − b_i)` and `s²_a + s²_b` are unbiased for `n·var` of the paired and
the unpaired difference-of-means respectively (the unpaired estimator is exactly "label a on
one world list, label b on an independent one" — the same 2n rollouts). Aggregated as a
ratio of sums (a variance-weighted mean, so a pair whose 8 worlds all agree contributes
0/0 = nothing rather than a NaN). Equivalently `VRF = 1/(1−ρ)` at equal σ.

**(b) `direct` — the replication audit.** On `--probe-jobs` seeds the estimator is
REPLICATED on `reps` disjoint world blocks. `var(Δ_r)` across blocks measures the paired
estimator's variance directly; `var(A_r) + var(B_r)` measures the unpaired one directly
(blocks are independent, so an `A` block and a `B` block from different replicates ARE an
independent pair of absolute labels). No modelling assumption at all. It costs `reps ×` a
normal pair, which is why it runs on a subset.

A tempting third estimator is wrong and was rejected: taking two replicates and comparing
`(A₁−B₁, A₂−B₂)` with `(A₁−B₂, A₂−B₁)` looks like a matched paired/unpaired comparison, but
the two "unpaired" contrasts share world blocks and are correlated by exactly `−2·cov`, so
their spread is biased UP by the very covariance the pairing exploits — it would flatter the
lever. `direct` uses the per-block means instead, which are genuinely independent.

### Results — `results/pairs_s1.json`, n = 1,301 pairs (`ev-fast-rules:07e21933f382`)

| | CRN (n = 1,301) | direct (n = 54) |
|---|---|---|
| var(Δ̂) paired, per world | 0.2340 | 0.02358 (at n = 8) |
| var(Δ̂) unpaired, per world | 0.4174 | 0.04707 |
| **variance-reduction factor** | **1.784** | **1.996** |
| mean ρ | +0.468 (907 pairs with both variances > 0) | — |
| se(Δ̂) at `n_worlds = 8` | **0.153** paired vs **0.204** unpaired | |

The two routes agree (1.78 vs 2.00 with n = 54 for the direct one), which is the point of
having both: the cheap CRN decomposition is not flattering the lever.

**By `pair_source`** — close_call 621 (47.7 %) **2.34×**, random 677 (52.0 %) 1.46×,
greedy_vs_extract 3 (0.2 %) n/a. Informative pairs are worth ~1.6× as much reduction as
uninformed ones, which is the expected direction: two actions the rules player rates equally
lead to more similar positions.

**By `state_kind`** — the finding:

| kind | n | VRF | mean ρ | resolved |
|---|---|---|---|---|
| nemesis | 271 | **2.62** | +0.63 | 3.3 % |
| hand | 289 | **2.48** | +0.63 | 3.8 % |
| pack | 272 | 1.80 | +0.24 | 5.1 % |
| shop | 272 | 1.42 | +0.41 | 9.6 % |
| blind_select | 197 | 1.23 | +0.18 | 7.1 % |
| **hand + nemesis only** | **560** | **2.54** (direct 2.62) | +0.629 | 3.6 % |

**By ante** (VRF): a1 1.45, a2 1.90, a3 1.93, a4 2.00, a5 1.79, a6 1.89 — **flat**. The
lever is not decaying with horizon; it is a per-state-kind effect.

### The two control measurements

1. **Coupling order** (`results/pairs_s1_diag_coupling.json`, 290 pairs on the same seed
   list, `coupling=determinize_then_step`): **VRF 1.86×, ρ +0.504**; hand+nemesis 2.48×.
   Against the frozen order's 1.78× / 2.54×, that is inside the noise. **The frozen
   step-then-determinize order is NOT what is limiting the lever** — sharing the immediate
   draw and the post-action pile buys ~4 %, so there is nothing to win by giving up honest
   `obs_a`/`obs_b`. (Fingerprint differs, `…:5667f3cc6ca8` — W-EXTRACT edited `hand.py`
   again between the two runs, and the field caught it.)
2. **Pre-extraction player** (`runs/pairs_s1/pre_extract/`, 206 pairs,
   `ev-fast-rules:2caf82bf67cb`): VRF 1.63× / direct 1.93×, hand+nemesis 2.74×. Same
   picture with the old fast player, so the extraction layer neither created nor destroyed
   the effect.

### Why hand pairs correlate and shop pairs do not (mechanism, from the engine)

`clone_determinized(w)` gives both branches the same fresh seed, so their `PseudoRandom`
key streams (shop contents, bosses, tags, pack contents, the round-end `'nr'` reshuffle)
are identical *as long as both branches consume those keys the same way*. A **hand**
decision leaves both branches with the same board, same jokers, same money, and the round
still ends at the same cash-out — where the `'nr'` reshuffle **re-synchronises the deck**.
A **shop / pack / blind_select** decision changes what the branch OWNS, so from that point
the branches draw different keys in different orders and the shared streams desynchronise
permanently; only the deck composition stays common. The remaining `ρ ≈ 0.2–0.4` there is
the shared *pre-branch* run state, not shared future luck.

A second, unavoidable factor: the label is **binary** (match win) at a full-match horizon.
The lead's own pre-round measurement (`results/rho_decay_*.json`) got ρ = 0.77–0.90 on
*continuous* targets (log score, money, lives lost) at 1–8 blind horizons. Binarising a
bivariate-normal pair at the median maps ρ → (2/π)·arcsin ρ, i.e. 0.87 → 0.67 — which is
almost exactly the ρ = 0.63 measured here at hand states. **The hand-state number is
therefore about as good as a binary match-win label can be**; the shop-state shortfall is
the desynchronisation above, not binarisation.

---

## 5. The campaign and its conventions

`gen_pairs.py` mirrors `gen_labels.py` exactly (that file is untouched): one pool job = one
seed (`pairs.pair_job`), `workers.run_pool` with a PAUSE file, rows buffered in the main
process and flushed every `--flush-jobs` jobs, a seed appended to `<run-dir>/done.ids`
**only after** the shards holding its rows are on disk, restart with the same `--run-dir`
skips recorded seeds, `--minutes` deadline, `--max-jobs` cap.

Two shard streams per run dir, deliberately separated so `dataset.list_shards` never sees a
pair shard:

```
<run-dir>/shards/pair_NNNN.npz        pairs.load_pair_shard / pairs.PairDataset
<run-dir>/abs_shards/shard_NNNN.npz   dataset.LabelDataset.load  (unchanged loader)
<run-dir>/done.ids, PAUSE, gen.jsonl
```

```bash
# the proof campaign (resumable: re-run the same command; pause: touch <run-dir>/PAUSE)
python ev/scripts/gen_pairs.py --run-dir ev/runs/pairs_s1 --seeds default+random:600 \
    --n-states 6 --n-worlds 8 --workers 8 --probe-jobs 10 --reps 4 --flush-jobs 8 \
    --minutes 70 --name s1

# the §4 coupling diagnostic (same knobs, the non-frozen coupling; NOT training data)
python ev/scripts/gen_pairs.py --run-dir ev/runs/pairs_s1/diag_coupling \
    --seeds default+random:600 --n-states 6 --n-worlds 8 --workers 8 --probe-jobs 0 \
    --minutes 15 --coupling determinize_then_step --name s1_diag_coupling
```

**Training data is `ev/runs/pairs_s1/shards` (pairs) + `.../abs_shards` (absolute rows),
and nothing else under that tree.** `diag_coupling/` and `pre_extract/` are measurements,
not training data, and they carry DIFFERENT `player_fingerprint`s (`…5667f3cc6ca8` and
`…2caf82bf67cb` vs the campaign's `…07e21933f382`). `dataset.list_shards` on a directory
does not recurse, so pointing the trainer at the two paths above is already safe — but a
recursive glob (`pairs_s1/**/*.npz`) would silently mix three policies. W-RANK's
`--pair-fingerprint-allow` is the belt to that braces.

`ev/runs/pairs_s1/pre_extract/` holds a first 12-minute burst run BEFORE W-EXTRACT landed
(a different `player_fingerprint`, `close_call`/`random` only). Kept deliberately: it is the
clean A/B on the same measurement with the old fast player, and its shards are readable by
the same loaders. It is NOT part of the campaign's headline numbers.

---

## 6. Measured

### The proof campaign (`ev/runs/pairs_s1/`, `results/pairs_s1.json`)

**1,301 pairs + 2,602 absolute rows from 232 seeds**, 29 pair shards + 29 label shards,
`n_worlds = 8`, `n_states = 6`, 8 workers, 0 failed jobs, 91 snapshots skipped (< 2 legal
actions). One fingerprint throughout: `ev-fast-rules:07e21933f382`. Ran 58 min of a 70-min
deadline before the driver was stopped; the crash-safe design held exactly as designed —
every flushed shard and its `done.ids` entries were intact, and the summary was regenerated
without re-rolling anything (`--workers 0 --max-jobs 0`). Resuming just needs the same
command; 232 of 726 seeds are recorded done.

Absolute rows: `y_mean 0.464`, `y_sd 0.319`, `mean ci 0.232`, `trunc_frac 0.000` — in line
with the 51k label corpus (0.498 / 0.317 / 0.238 / 0.000), so the post-action rows are not
a distributionally odd label set.

### Throughput (for the runbook projection)

| | measured |
|---|---|
| steady rate, 8 workers | **22.2 pairs/min** (355 rollouts/min) |
| per rollout, in-worker | 1,156 ms (87 % of it inside the policy) |
| per pair, in-worker | ~18.5 s (16 rollouts) |
| ramp | 4.0 → 13.3 → 20.4 → 22.2 pairs/min (the first 10 jobs are the 4×-cost `--reps 4` probe) |

**Caveat on the number: the box was NOT quiet.** A second workstream's 8-worker pool ran
throughout (measured 16 busy python processes of 32 cores), which is why ms/rollout sat at
1,156–1,380 here against **1,046** in the last (less contended) diagnostic run and ~1,100
single-threaded.

**16-worker projection: ~44 pairs/min** (linear from the contended 8-worker rate — the
conservative read). On a genuinely idle box the per-rollout time should fall back toward
1.0 s, giving ~60 pairs/min. So the runbook's R1 full campaign costs roughly **3.8 h for
10k pairs** (2.8 h at the optimistic rate), producing 20k absolute rows alongside.

### Realised mix (`mix_report`, and the requested→realised cross-tab)

| requested | → close_call | → greedy_vs_extract | → random |
|---|---|---|---|
| close_call (646) | 367 | 0 | 279 |
| greedy_vs_extract (434) | 254 | **3** | 177 |
| random (221) | — | — | 221 |

Realised: **close_call 621 (47.7 %) / greedy_vs_extract 3 (0.2 %) / random 677 (52.0 %)**
against the 50/40/10 target. State kinds: hand 22.2 %, pack 20.9 %, shop 20.9 %, nemesis
20.8 %, blind_select 15.1 %, `other` 0 % (structurally impossible).

* `close_call` lands its 50 % target almost exactly.
* **`greedy_vs_extract` is effectively empty: 3 of 434 requests.** This is not a plumbing
  failure — the hook is verified live against W-EXTRACT's real `hand.extraction_lines`
  (tests + `test_extraction_feature_detection_matches_reality`) and it *did* fire 3 times.
  It is §2's third hard fact: `extract_on` requires a non-PvP `SELECTING_HAND` state on a
  board that actually carries a proc (seal / Lucky / Business Card / gold / a tarot wanting
  a target), and a uniformly-sampled `b_red` stake-1 run at antes 1–6 almost never has one.
  Getting the 40 % bucket needs proc-biased state selection — W-EXTRACT's own 12-seed
  extraction dev slice is the obvious source — not a change here (§7.4).
* `random` overshoots to 52 % because 279 close-call requests and 177 extraction requests
  had nowhere else to go. Those are still valid pairs (just less informative: VRF 1.46 vs
  2.34), and they are labelled honestly.

### W-RANK interop, verified on real data

`train_v.PairDataset.load("ev/runs/pairs_s1/pre_extract/shards")` → 206 pairs, all
`obs_a`/`obs_b` keys, all 8 of its typed columns, `delta` as float32. W-RANK's trainer reads
this campaign's shards unchanged.

### Tests

`ev/tests/test_pairs.py`: **31 tests, ~14 s**, all on the scripted policy / synthetic
records except two EV-player integration tests. Coverage: action identity, the fingerprint
(stability + config sensitivity + source sensitivity), the mix sequence with and without
extraction, close-call/cascade/random selection, the extraction hook in all three shapes
(generator, `HandAnalysis` method, module function in either argument order) plus a raising
hook, both couplings, the CRN plumbing (identical actions ⇒ identical worlds, actor
perspective, reproducibility, snapshot immutability, disjoint replicate blocks), the frozen
schema key set, strict-JSON round trip, shard round trip, reconstruction from `meta`, the
absolute rows loading through the unchanged `LabelDataset`, and `variance_report` /
`mix_report` against hand arithmetic. Full suite `python -m pytest ev` green.

---

## 7. Open issues / what the lead should decide

1. **THE DECISION: the lever's strength is a function of STATE KIND, not of ante** (§4).
   1.78× uniform, 2.54× at hand+nemesis, 1.23–1.80× at blind_select/pack/shop, flat across
   antes. Mechanism in §4. Three options, in the order I would rank them:
   * **(a) weight the full campaign toward hand/nemesis states** (e.g. `per_kind` 3/3/1/1/1
     instead of 2/2/2/2/2 — a `--per-kind` flag would be a ~10-line addition to
     `gen_pairs.py`). Clears the 2× bar and concentrates pairs exactly where the measured
     argmax-V failure is (per-action hand EV gaps ≪ 0.05, brief §0). Cost: fewer pairs to
     teach V shop ordering, which is the *other* half of what a policy needs.
   * **(b) accept 1.78× uniform.** Still a real halving of the rollouts needed for a given
     resolution; W-RANK's confidence weighting already down-weights what stays unresolved.
     Honest, but it does not meet the bar the brief set.
   * **(c) raise `n_worlds` for shop/pack/blind pairs only.** Variance falls as 1/n
     regardless of ρ, so 16 worlds at a shop state buys the same resolution the CRN does not.
     Linear cost, no schema change, and `n_worlds` is already per-record.
   What I would NOT do is drop lever (b): at hand states it delivers, and hand states are
   where the policy is losing.
2. **The frozen coupling order is NOT the problem — measured and closed.**
   `determinize_then_step` on 290 pairs of the same seeds gives 1.86× vs the frozen order's
   1.78× (hand+nemesis 2.48× vs 2.54×): inside the noise. There is nothing to buy by giving
   up honest `obs_a`/`obs_b`, so the frozen order stays and the "store obs from the real
   state, roll from the determinized one" middle option is not worth building. The knob is
   left in place (`roll_pair(coupling=…)`, `--coupling`) so the measurement is repeatable.
3. **`sample_states`' default reservoir seed is `hash((seed, policy_seed))`** — Python's
   `str` hash is `PYTHONHASHSEED`-salted, so the SET of snapshots a seed yields differs
   between processes. Harmless for a campaign (every row still carries its own reconstruction
   tag and `reconstruct_snapshot` replays exactly by step) but it makes any "same seeds →
   same snapshots" claim false across runs, and it made the first version of this module's
   tests flaky. Not fixed here (`labels.py` is W5's file); the tests pin `rng_seed=` instead.
4. **`greedy_vs_extract` is structurally capped** by `HandAnalysis.extract_on` (§2): no
   Nemesis, no shop/pack/blind-select, and no board without a proc. Reaching the brief's
   40 % needs proc-biased seed/state selection — e.g. sampling from W-EXTRACT's own 12-seed
   extraction dev slice — not a change to this module.
5. **Resolution at `n_worlds = 8` is low** (§6's `resolved_frac`): most pairs' `|delta|` does
   not exceed their own paired CI. That is expected and is exactly why W-RANK weights by
   `|delta| / delta_ci` — unresolved pairs contribute ~0. If the lead wants a higher resolved
   fraction, `n_worlds` is the knob and its cost is linear.
