# PHASE 5 rev 2 — V-v2 build brief: levers (b)+(c) + the extraction layer

Date: 2026-08-24 · Branch: `mp/campaign` · Lead commits; workstreams do NOT commit.
Predecessors to read first: `docs/PHASE5_BRIEF_2026-08.md`, `docs/STATE_SPEC_v1.md`,
`ev/EV_NOTES.md`, `ev/TRAINV_NOTES.md`, `ev/ADVISOR_NOTES.md`, `CAMPAIGN_LOG.md` (last two
entries: build-complete + first-results).

## 0. Why — the measured failure this round attacks

From the 2026-08-24 results (all in `results/`, commits `20e1ad0`/`8c99909`):

* V (5M, keeper `ev/runs/v_full_best/ckpt_0001000.pt`) is a well-calibrated match-win
  predictor: held-out Brier 0.060 / AUC 0.784 / ECE 0.021 on 51,024 labels.
* **argmax-V as a policy loses to the rules player 2/60.** Per-action EV gaps (≪ 0.05) sit
  below the label noise (mean CI ±0.24 at `n_rollouts=8`). Absolute labels cannot resolve
  within-state action ordering at any affordable rollout count (CI ∝ 1/√n → n≈500 for ±0.03).
* stats-tier-as-policy loses 8/60 → stays a diagnostic layer.

Tagg picked levers **(b) within-state ranking loss** (same-world action pairs; measured
ρ = 0.77–0.90 flat ⇒ paired differencing cancels shared luck, var(Δ) = 2σ²(1−ρ)) and
**(c) V at the expectimax leaf only** (V values end-of-blind states where differences are
large; exact search handles the fine-grained part).

## 1. The extraction/sandbag requirement (Tagg, 2026-08-24 — first-class, not a nice-to-have)

Tagg's design note, paraphrased: on NON-PvP blinds the right line is often to *sandbag* —
use all hands/discards to extract value instead of clearing in one: draw+discard purple
seals (tarot per discarded card), play gold-seal / lucky / Business-Card-relevant cards for
money procs, hold faces for Reserved Parking, hold gold cards to round end, cycle the deck
so held tarots land on the cards you actually care about. Greedy never finds this: clearing
in 1 hand pays ~$3 more in unused-hand money, but that is sometimes worth less than the
deck-fixing + econ-joker income the extra hands buy.

**As built this is unrepresentable in three places** (verified in `ev/EV_NOTES.md` §1–2):

1. The fast objective `P(clear) + 0.012·E[hands unused] + 0.002·E[discards unused]`
   *pays to bank hands* and has no term for any proc/extraction value.
2. The candidate generator cannot emit the key lines: junk order keep-value counts
   "enhancements" as keep-worthy, so "discard the purple seals" is never generated;
   deliberately weak proc-plays likewise.
3. The full budget re-ranks only fast's top K=5, so a sandbag line pruned by the greedy
   fast objective never reaches V's leaf.

**Split by the campaign's encode-vs-learn criterion** (MP_TRAINING_DESIGN):
* ENCODE (closed-form, this round): per-action money-proc EV — gold seal +$3 when scored;
  lucky 1/15·$20 (money part only; the +mult part is already in the dry-run scorer);
  Business Card ½·$2 per scored face; Reserved Parking ½·$1 per held face per hand played;
  gold enhancement $3 if held at end of round; Faceless $5 per discard containing ≥3 faces;
  purple seal → +1 Tarot on discard (value = config constant `tarot_value_dollars`,
  default 4, until V takes over). Currency: the objective's existing rate, $1 = 0.012.
* LEARN (V at the leaf): what a fixed deck / extra tarot / seal is worth antes later.
* PAIR (lever b's best data): "clear now" vs "extract one more hand" from one state on the
  same determinized worlds.

## 2. Common ground rules

* Repo: `C:\Users\Taggart\projects\balatro-rl`, all work under `mp/`, branch `mp/campaign`.
* Tests: `python -m pytest ev` (pytest.ini there), engine suites must stay green if you
  touch `engine`. Run what you touch.
* **Do not commit. Do not touch files owned by another workstream** (ownership below);
  additive needs elsewhere → document in your NOTES file and stop at the interface.
* Each workstream writes/extends its NOTES file (named per section) in the same style as
  `ev/EV_NOTES.md`: what/where, decisions, measurements, open issues.
* **Ops caps (hard):** ≤ 10 concurrent torch-loading processes on this box (47 GB, OOM
  measured); every h2h uses `--max-steps 4000`; the box is SHARED with Tagg today —
  dev gates at ≤ 8 worker processes, short bursts; anything > ~15 min of full-box load goes
  into the idle-box runbook (§8), not run now.
* Baselines you may cite: EVPlayer 126-seed gate 95.2/96.0% ante-1; ev:full beats
  real1:det 57/58; argmax-V 2/60; stats 8/60.
* Label-policy versioning: any change to the fast player changes the label/rollout policy.
  New shards MUST carry `player_fingerprint` (see §5). The 51k corpus
  (`ev/runs/labels_full*`) is OLD-policy: usable for absolute-BCE pretraining, never
  silently mixed — the trainer must be able to filter by fingerprint.

## 3. W-EXTRACT (strong) — the extraction layer in the analytic player

Owns: `ev/hand.py`, `ev/player.py` (candidates/objective/keep-value paths),
`engine/` fixes for proc fidelity, `ev/EXTRACT_NOTES.md`, tests.

1. **Engine fidelity first.** For each §1 proc: verify the engine implements it and fires
   on its real per-key RNG stream, against the Lua reference in `_reference/balatro_src/`
   (READ-ONLY, gitignored, never vendor/quote at length). Fix + test what's wrong or
   missing (the Phase-1 pattern: cite the Lua file:line in the test docstring).
   `engine_parity`/`parity_check` 126/126 must hold after any engine change.
2. **Proc-EV term** `extraction_ev(action, state)` in dollars, exact per the engine's
   actual mechanics (face counts from the real cards, Parking per hand *remaining*, etc.),
   entering the hand objective at $1 = 0.012. Interest awareness: below $25, use the
   existing `money = $ + 0.8·interest` convention rather than a new model.
3. **Keep/junk value overhaul:** discard-value vs play-value vs hold-value per card
   (purple seal = discard-valuable; gold seal / lucky / face-with-Business-Card =
   play-valuable; gold enhancement / face-with-Parking = hold-valuable). Junk ordering and
   `_discard_lines` use the right one for the line being generated.
4. **Extraction candidate lines, safety-gated:** generate "discard the seal/proc cards",
   "weakest clearing play", and "stall play with proc cards" lines ONLY when the tail DP
   says the blind is safe with what remains (gate on P(clear | remaining hands/discards
   after the line) ≥ 0.90, config). Never at a Nemesis. Cost budget: fast-budget mean
   hand decision must stay ≤ 5 ms.
5. **Tarot targeting:** when a held targeted tarot's best targets are not in hand, cycling
   lines that draw toward them count the expected improvement (via the existing targets/
   hypergeometric machinery) as extraction value. Keep it first-order; document what you
   don't model.
6. **Gates:** (a) 126-seed official gate `python ev/gate_ev_player.py --procs 8` — ante-1
   ≥ 95% both budgets (no regression); (b) a 12-seed extraction dev slice (pick seeds whose
   ante-1/2 decks+jokers contain procs; document them): mean end-of-ante-2 money and
   tarots-used strictly up vs pre-change player, blinds lost not up; (c) unit tests for each
   proc EV against hand-computed values; (d) h2h new-fast vs old-fast, 30 seeds paired,
   `--max-steps 4000`, ≤ 8 procs — must be ≥ 50% (extraction should not lose matches).

## 4. W-LEAF (sonnet, runs in a git worktree — lead merges) — lever (c)

Owns (in its worktree): `ev/hand.py` value_fn path + `ev/player.py` full-budget
config, h2h driver invocations, `ev/LEAF_NOTES.md`.

1. Wire keeper V (`ev/runs/v_full_best/ckpt_0001000.pt`) into the full budget via the
   existing `MatchAwareEVPlayer`/`value_fn` plumbing (fix-pass semantics: exceptions
   propagate; ROUND_EVAL advanced so V sees the shop; GAME_OVER=0).
2. Implement EV_NOTES §8.3: with `value_fn` set, K=3 candidates × 8 worlds (flag-driven,
   default unchanged without V). Verify per-decision cost ≤ 100 ms budget.
3. **Numbers (the deliverable):** paired 30-seed h2hs at the ops caps —
   (i) ev:full+Vleaf vs ev:full (the lever-c read), (ii) ev:full+Vleaf vs real1:det
   (no regression vs 57/58). Report per-match lives margin and ante reached, not just W/L.
4. Diagnose: on 20 sampled leaf evaluations, log V's value vs the analytic proxy — where do
   they disagree most (state kind, ante)? Two paragraphs in NOTES.

Expectation is honest: current V may be a null here (proxy may already capture what V
knows). A clean null with the diagnosis is a full success for this workstream — the wiring
is what (b)'s retrained V plugs into.

## 5. W-PAIRS (strong) — lever (b) data

Owns: `ev/pairs.py` (new), additions to the worker-pool entry points, small proof
campaign outputs under `ev/runs/pairs_s1/`, `ev/PAIRS_NOTES.md`, tests.
Coordinates by SCHEMA ONLY with W-RANK (schema frozen below — implement it exactly; if it
collides with an existing convention, document, don't rename).

1. **Pair job:** from a sampled decision state s (reuse `sample_states` machinery), take
   actions a, b → post-action match clones; roll BOTH to match end on the SAME
   `clone_determinized` seed list (n_worlds per pair, default 8), fast policy both sides,
   race calc at truncation — exactly the label semantics of TRAINV_NOTES §1, but paired.
2. **Pair selection** (the `pair_source` field): `close_call` = rules player's top-2 when
   fast-EV gap < 0.03; `greedy_vs_extract` = rules choice vs the best extraction line
   (needs W-EXTRACT's generator — feature-detect: if not landed yet, emit close_call/random
   only and leave the hook + test ready); `random` = rules choice vs uniform legal (10%).
   Target mix ~50/40/10, all state kinds; log realized mix.
3. **Frozen shard schema** (one JSON record per pair, alongside the two absolute rows the
   branches also yield — pairs do double duty as labels):
   `{"kind":"pair", "seed","step","actor","state_kind","ante", "player_fingerprint",
   "pair_source", "action_a","action_b", "n_worlds",
   "outcomes_a":[...], "outcomes_b":[...],  # per shared world, win=1/0 or race float
   "delta","delta_ci",                      # mean paired diff + CI OF THE PAIRED diff
   "obs_a","obs_b",                         # encoder-v2 inputs, same storage as labels
   "meta":{reconstruction + selfplay config}}`
4. **The measurement that justifies the lever:** on ≥ 200 pairs, report the empirical
   variance of the paired estimator vs two independent absolute labels at equal rollout
   budget — the realized variance-reduction factor. If it's < 2×, stop and say so loudly.
5. Proof campaign: ~1–2k pairs at ≤ 8 workers (box shared), resumable, PAUSE-file, same
   worker-pool conventions as labels. Throughput number for the runbook projection.

## 6. W-RANK (sonnet) — lever (b) loss

Owns: `ev/train_v.py` additions, `ev/RANK_NOTES.md`, tests. Codes against §5's frozen
schema (synthesize fixture shards for tests; do not wait for W-PAIRS).

1. Loss = BCE(absolute rows) + `lam_rank` · pairwise logistic on
   (V(obs_a) − V(obs_b))/τ vs the paired outcome, confidence-weighted (weight by how
   resolved the pair is, e.g. |delta|/delta_ci capped; unresolved pairs contribute ~0).
   `lam_rank`, τ, weighting scheme = config; document defaults and why.
2. Metrics per eval step: held-out PAIR ACCURACY (resolved pairs only, by seed-held-out
   rule), per `pair_source` and `state_kind`; existing BCE/Brier/AUC/ECE must keep being
   reported — the calibration must not degrade (ECE guardrail in NOTES).
3. Preserve: bit-exact resume (extend the pinned test to cover pair batches), PAUSE,
   shard filtering by `player_fingerprint` (train on old 51k absolute + new pairs, or
   new-only — a flag).
4. Smoke: train dummy + real net briefly on synthesized fixtures; overfit a tiny pair set
   to pair-acc ~1.0 (sanity that gradients flow through both branches).

## 7. W-PROBE (sonnet, launched after W-EXTRACT lands) — acceptance fixtures

Owns: fixture states + advisor integration + `ev/PROBE_NOTES.md`.
Tagg's scenarios as named advisor fixtures (like `fixture:bloodstone_vs_invisible`):
purple-seal discard, Faceless + 3 faces, Business-Card board, Reserved Parking hold,
gold-seal weak-play, tarot-targeting cycle — each with a matched control where greedy is
right (low P(clear) or no procs). `python ev/cli.py advise fixture:<name>` must render
the extraction lines with their EV decomposition. Regression tests pin the qualitative
ordering (extract > clear-now in the sandbag fixtures, reversed in controls).

## 8. After landing — the idle-box runbook (lead)

R1 full pair campaign (16 workers, overnight, resumable) with W-EXTRACT's player →
R2 retrain V (BCE+rank) → R3 evals: 126-seed gate, tournament_v vs rules (the 2/60
baseline), h2h ev:full+Vleaf(new V) vs ev:full and vs real1:det, W-PROBE fixtures.
The numbers Tagg reads next: pair accuracy, the variance-reduction factor, the fixture
table, and whether V-at-leaf moved the h2h.

## 6b. W-AUX (strong, wave 2 — launches AFTER W-PAIRS and W-RANK land; must land BEFORE the
full campaign) — auxiliary prediction heads on rollout intermediates

Approved by Tagg 2026-08-25. Rationale: every label costs 8 full-match rollouts and currently
yields one bit; the rollouts contain dense proximal quantities the workers throw away.
Predicting them as auxiliary heads densifies the signal and shapes the trunk representation
(UNREAL-style). It is also the mechanized post-mortem ("lost because no xmult"), and per-head
errors become a diagnostic for WHERE V is wrong.

Owns: aux-target recording in the label/pair rollout loop (extending W-PAIRS' landed code),
head additions to SetValueNet, multi-task loss in train_v.py (extending W-RANK's landed code),
`ev/AUX_NOTES.md`, tests.

1. **Targets** (mean over the shared worlds; recorded for BOTH branches of a pair; all must be
   computable from the rollout trajectory/match state the workers already see — light
   instrumentation only, no extra simulation): money at next shop entry; own+opp lives at end
   of next 2 antes; log-score margin at next PvP blind; current blind cleared (binary); best
   xmult tier acquired by ante 4 (binary/ordinal); extraction income over the next ante
   (dollars from procs — ties to the sandbag layer); cards modified + tarots used over the
   next ante. ~6–8 heads, config-listed; document any target you drop as uncomputable.
2. **Heads**: linear (or 1 hidden layer max) off the shared trunk; BCE for binaries,
   regression on log1p for money/score. Loss = main + Σ w_i·aux_i, defaults ~0.1, config.
   Aux heads exist only in the trainer graph — play-time inference and checkpoint LOADING of
   old checkpoints must be unaffected (fresh-init heads when absent; keeper ckpt loads clean).
3. **Schema**: additive `aux` dict on label and pair records; missing-field masking so old
   shards train with aux terms muted. Coordinate ONLY through the landed code + NOTES of
   W-PAIRS/W-RANK — do not redesign their schema, extend it.
4. **Preserve**: bit-exact resume (extend the pinned test to aux state), PAUSE, fingerprint
   filtering, all existing metrics + per-head held-out metrics each eval.
5. **Gate**: on equal label data (the proof shards + a small fresh aux-recorded batch),
   held-out Brier/pair-accuracy with aux ≥ without (report the ablation); per-head sanity
   (money head R² clearly > 0 — it is nearly deterministic given state); no ECE degradation.
6. If W-ACTIVE's POC verdict is positive by the time you land, note (do not implement) how
   per-head error could sharpen the acquisition score — next-round material.
