# Phase 3 Brief — Infrastructure (2026-08-21)

Lead-authored kickoff brief. Phase 2 closed the same evening with every gate green and every human
check confirmed in the live game (`CAMPAIGN_LOG.md` "PHASE 2 COMPLETE" + "Tagg LIVE CHECKS"). Phase 3
is *infrastructure*: nothing here has an oracle, so the verification is "it runs end-to-end, the numbers
are reproducible, and the tests pin the contract". Expect more lead review and less unattended chug.

State at kickoff (nothing committed): `pytest mp/engine/tests` **1609 / 10 skip / 3 xfail / 0 fail**;
`pytest mp/tests` **1073 / 2 xfail / 0**; `engine_parity --antes 1-8` and `parity_check --antes 1-8 --variant
faithful` **126/126**. These must stay green at every hand-off. The engine is `mp/engine/balatro_sim`
(`BalatroGame(seed, deck_key=, stake=, ruleset="vanilla"|"mlb")`, `clone()`, `legal_actions()`,
`state_signature()`; MLB single-player hooks `lose_life` / `set_pvp_info` / `end_pvp`, `State.PVP_WAIT`,
`BlindInfo.is_pvp` — see `engine/MLB_NOTES.md`). Two-player lockstep = `engine/balatro_sim/mlb_match.py`.
Exit-gate driver = `scripts/mlb_match_demo.py` (`ScriptedPlayer`, `MatchRecorder`, RNG key classification).

## 0. Decisions made by the lead

1. **The MCTS agent layer is FORKED into `mp/agent/`.** Source: `C:/Users/Taggart/projects/recovered/balatro-mcts/`
   (`mcts/`, `train/`, `scripts/train_cold.py`, `scripts/smoke_selfplay.py`, `scripts/mcts_demo.py`, its tests;
   ~1.9k lines, git `ee75d11`). That repo is **read-only for this campaign** — do not edit it, do not commit to it.
   Its encoder/action-features are written against the OLD engine (pre-rekey catalogue keys, 434-dim obs); they
   must be re-targeted to the fork (game keys `j_*`/`c_*`/`v_*`, `OBS_DIM` 447 SP / 457 MP, `hand_eval` flags).
   The V7 checkpoint is LOST (memory: confirmed 2026-06-10) — **no warm-start; cold start only.**
2. **Interleaving contract for the N-agent instrument** (this was W4's top Phase-3 risk): in the tournament runner
   every agent plays its Nemesis blind **to exhaustion** (all hands) and the N×N outcomes are computed from final
   scores with the server rule (strictly lower loses, tie = nobody). Rationale: MLB pays no unused-hand money at a
   PvP blind, so playing on costs nothing except per-hand joker state (Ice Cream, Popcorn, Seltzer, Turtle Bean…)
   and deck-out — document those as the known gap. `MLBMatch` keeps the canonical alternation for true 1v1.
3. **Lives in the N-agent run are pluggable** (`life_rule`): `"paired"` (default — each agent has a fixed assigned
   opponent for the life decrement, faithful 1v1 lives; matrix still extracted from ALL scores), `"median"`
   (below population median loses a life — the design doc's option (a), explicitly NOT faithful, useful for
   selection), `"none"` (pure measurement, nobody dies, run to a fixed ante). The N×N matrix is the product;
   lives only decide who keeps playing.
4. **Model split for usage economy:** W1 and W3 (design + judgment, no oracle) on the strong model; W2 and W4
   (clear specs, verifiable by running) on Sonnet. Lead stays on the strong model.

## 1. Workstreams (disjoint ownership)

| # | Workstream | Model | Owns | Depends on |
|---|---|---|---|---|
| **W1** | Agent-layer fork + sync + **checkpointing** | strong | `mp/agent/**` (new package: `mcts/`, `train/`, `scripts/`, `tests/`), `mp/agent/AGENT_NOTES.md` | — |
| **W2** | **N-agent same-seed runner + N×N matrix** | sonnet | `mp/tournament/**` (new: `runner.py`, `matrix.py`, `players.py`, `cli.py`, `tests/`), `mp/tournament/TOURNAMENT_NOTES.md` | engine only |
| **W3** | **Batched NN inference + tree reuse** | strong | `mp/agent/mcts/batched.py`, `mp/agent/mcts/reuse.py` (or in-place edits to `search.py`/`policy.py` once W1 hands off), `mp/agent/benchmarks/`, `mp/agent/BATCH_NOTES.md` | **W1** |
| **W4** | **Eval harness + ρ-decay harness** (+ first scripted-player ρ numbers) | sonnet | `mp/eval/**` (new: `eval_harness.py`, `rho_decay.py`, `tests/`), `mp/eval/EVAL_NOTES.md`, `mp/results/` (new; JSON/CSV outputs only) | engine + `scripts/mlb_match_demo.py`'s players |

Shared files: **none by design.** `mp/engine/**` and `mp/rng/**` are FROZEN in Phase 3 — if you need an engine
change, write it up in your notes as "needs engine change: file:line, why" and work around it. The lead decides.

### W1 — Agent-layer fork + sync + checkpointing (strong model)

- Fork `balatro-mcts` `mcts/`, `train/`, the three scripts and their tests into `mp/agent/` with provenance
  (`AGENT_NOTES.md` §0: source commit, file map, what changed and why). Standalone `pytest.ini`/`conftest.py`
  following `mp/engine/conftest.py`'s pattern (must import the FORK engine, raise if the BRL package wins).
- Re-target to the fork engine: encoder → reuse `env_v7`'s 447-dim encoding (don't maintain a second copy;
  if the MCTS obs must differ, subclass/extend, and say why); action features → game keys; `legal_actions()`
  is already the fork's. `PolicyValueNet` trunk is V7-shaped — keep the architecture but drop every "loads V7
  weights" affordance (the checkpoint is lost) or leave it inert and documented.
- **Checkpointing**: `torch.save` of model + optimizer + replay/episode counters + RNG states + config every K
  episodes and at exit/KeyboardInterrupt; `--resume <path>`; a test that trains 3 episodes, saves, reloads,
  trains 1 more, and compares to an uninterrupted 4-episode run (bit-exact if seeded, else assert identical
  sample counts + finite losses; document which).
- **Pluggable policy/value interface** that W3 can batch: define `PolicyValueFn` so that a batched
  implementation can evaluate many leaves in one call (e.g. `evaluate_many(games) -> list[(priors, value)]`),
  with the current single-leaf `NNPolicy` as the reference impl. Don't build the batching — W3 does.
- MLB awareness: the agent must be able to play a `ruleset="mlb"` game including the Nemesis blind where the
  target is external (`set_pvp_info`) and `PVP_WAIT` is a no-action state. `_is_win` / `_shaped_z` must not
  assume ante-8 win under MLB — make the terminal/outcome signal a parameter (SP win vs MLB match outcome vs
  externally-supplied margin), because W2/W4 will supply it.
- Re-measure: `mcts_demo.py` sims/sec on the fork (the historical figure was ~1000-1600 sims/s CPU,
  ~745 with NN); `smoke_selfplay.py` end-to-end on CUDA for 2 minutes with checkpoints landing.
- Gates: all forked tests green on the fork engine (66 historically; the 8 `TestActionMasking` failures were a
  missing `train_sim` module — resolve or skip with reason); checkpoint round-trip test; a 2-minute cold run
  that writes ≥1 checkpoint and resumes.

### W2 — N-agent same-seed runner + N×N matrix (Sonnet)

- `mp/tournament/runner.py`: `Tournament(seed, n_agents, players, deck_key, stake, life_rule, max_ante)` —
  N independent `BalatroGame(seed, ruleset="mlb")` instances stepped by their `Player` in **ante lockstep**:
  everyone plays Small/Big (own shops, rerolls, skips), then at the Nemesis every agent plays to exhaustion
  (decision 0.2), scores are collected, the matrix is extracted, lives applied per `life_rule` (0.3), dead
  agents drop out, survivors continue. Must run **100 agents on one seed end-to-end** (to ante 8 under
  `life_rule="none"`, and to last-agent-standing under `"paired"`) in reasonable time with scripted players —
  report wall clock. Use `clone()` of a single post-start game to fan out N copies if that's faster than N
  constructions (measure; both must give identical `state_signature()`).
- `mp/tournament/matrix.py`: from N final scores at one Nemesis → N×N outcome matrix (+1/0/−1 with tie=0),
  N×N log-score margin matrix, population rank per agent, per-ante score distribution (quantiles) — this IS
  "layer 1" from the assessment. Serialize per ante to `.npz` + a JSONL summary. 4,950 comparisons from 100
  agents; it's a sort, not a sim.
- `mp/tournament/players.py`: the `Player` protocol (`act(game) -> action`) + adapters for
  `scripts/mlb_match_demo.ScriptedPlayer` and a random-legal player; leave a clearly-marked slot for an MCTS
  player (W1/W3 will plug in — do not import `mp/agent`).
- Heterogeneity hook: the population must be heterogeneous or the matrix degenerates (design doc §6). Support
  per-agent player objects with different parameters; log a degeneracy metric (fraction of exact ties per
  Nemesis) so a collapsed population is visible.
- Tests: determinism (same seed + players → identical matrices), matrix properties (antisymmetry, tie
  diagonal, rank consistency with scores), the three life rules, 100-agent smoke, queue alignment at the
  first shop of every ante for a sample of agents (reuse `mlb_match_demo.diff_rng`/`classify_key` — import
  them, don't copy).
- `TOURNAMENT_NOTES.md`: the interleaving contract and its known gap (decision 0.2), how lives map, file
  formats, wall-clock numbers, what's next for plugging the MCTS player in.

### W3 — Batched NN inference + tree reuse (strong model; starts when W1 hands off)

- Read W1's `AGENT_NOTES.md` and the interface it left. Implement (a) **batched leaf evaluation** across many
  concurrent trees (the tournament / self-play has N agents deciding at once — batch their leaves into one
  forward pass; virtual loss or leaf-parallel collection within a single tree is optional, justify if done),
  (b) **tree reuse** between consecutive decisions (keep the chosen child's subtree; invalidate correctly on
  chance nodes — the engine is stochastic via keyed RNG, so document exactly when a subtree is still valid:
  same `state_signature()` after the action ⇒ reuse, else discard).
- Benchmark before/after on the same hardware: sims/sec single-tree CPU, sims/sec for 32 / 100 concurrent
  trees on CUDA, fraction of time in NN vs sim vs Python overhead. Write the numbers down.
- Tests: batched == single-leaf results (same priors/values to float tolerance for the same leaves),
  reuse == fresh-search visit distributions when the subtree is valid, correct discard when not.
- `BATCH_NOTES.md`: design, numbers, the remaining bottleneck.

### W4 — Eval harness + ρ-decay harness (Sonnet)

- `mp/eval/eval_harness.py`: evaluate a `Player` (scripted now; checkpoint later via a `--player` loader that
  W1 will document) over a fixed seed list (default: the 126 ground-truth seeds) in three modes — SP vanilla
  (win rate / furthest blind / final ante), SP-MLB solo (furthest ante, lives lost, money curve), **1v1 MLB via
  `MLBMatch` vs a reference player** (win rate, mean lives margin, per-Nemesis log-score margin) — with
  bootstrap CIs, a JSON report in `mp/results/`, and a `--compare a.json b.json` that reports the paired
  difference with CI (paired by seed — that's the whole point of common random numbers).
- `mp/eval/rho_decay.py`: the design doc §1 experiment. Paired arms on one seed, diverging at a chosen decision
  (e.g. ante-1 first shop: arm A buys shelf slot 0, arm B doesn't; make the perturbation pluggable), then both
  arms play the same scripted policy forward. Outcome variables at horizons h = 1, 2, 4, 8 antes: log-score at
  the Nemesis h antes later (play-to-exhaustion, like W2), money, lives. Compute ρ(h) = corr(outcome_A,
  outcome_B) across seeds, AND the implied variance-reduction factor vs unpaired seeds (run unpaired controls).
  Run it for real with scripted players over the 126 seeds (and more if cheap) and report the numbers — this is
  the first measurement the campaign has been waiting for, even if scripted players make it preliminary.
- Tests: harness determinism, CI sanity (a player vs itself → difference CI contains 0), ρ(h) = 1 when arms are
  identical, ρ → lower with a larger perturbation.
- `EVAL_NOTES.md`: how to run, the ρ(h) table with N and CIs, what changed vs the design doc's guesses
  (it guessed ~0.9 at h=1 → ~0.3-0.5 at h=4-8).

## 2. Exit gate (from the campaign plan)

1. 100 agents run one seed end-to-end in the tournament runner and the N×N matrix is produced per Nemesis.
2. `train_cold` (forked) saves checkpoints and reloads; the round-trip test passes.
3. Batched inference + tree reuse benchmarked with numbers; correctness tests pass.
4. Eval harness produces a paired-comparison report; ρ(h) measured at h = 1/2/4/8 with scripted players.
5. All Phase 0-2 gates still green; `mp/engine/**` and `mp/rng/**` unchanged (`git diff --stat` on those paths
   is empty relative to the Phase 2 close — the lead will check by mtime since nothing is committed).

## 3. Things only Tagg can do

Nothing blocking in Phase 3. Optional: decide whether to commit `mp/` at the Phase 3 close (three phases of
uncommitted work is a real loss risk — the lead will raise it).
