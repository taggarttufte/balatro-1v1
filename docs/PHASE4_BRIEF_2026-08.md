# Phase 4 Brief — Make training real (2026-08-22)

Lead-authored. Phase 3 closed 2026-08-22 with every gate green; the overnight shakedown (`CAMPAIGN_LOG.md`
07:35 entry) proved the pipeline learns end to end AND that **solo MLB with a free Nemesis is a degenerate
objective** (the agent learned to skip 15/16 blinds and coast). Phase 4 turns the infrastructure into a training
run whose checkpoints are worth keeping, then launches the first real run.

State at kickoff: branch `mp/campaign` @ `2801ddb` (+ uncommitted 07:35 log entry). Gates: `pytest engine/tests`
**1614/10/3/0**, `tests` **1073/2/0**, `agent/tests` **131**, `tournament/tests` **31**, `eval/tests`
**49**; `engine_parity` + `parity_check --antes 1-8` **126/126**. All must stay green. `engine/**` and `rng/**`
are FROZEN again (mtime snapshot: `docs/phase3_frozen_snapshot.txt` + the one lead fix to `game.py`).

## 0. Decisions already made (Tagg, 2026-08-22)

1. **Set-based observation encoding.** Hand / jokers / consumables / shop shelf / packs as masked variable-length
   sets with a shared per-item embedding + pooling/attention; joker position feature kept (order scores). No
   `OBS_DIM`. Action features set-based too (pool over the selected cards + action-type embedding). Replaces the
   V7-shaped trunk and the 56-dim action row.
2. **Per-sample action subsampling**: keep every action the search visited (visit count > 0) + a few random
   zero-visit actions, renormalise the policy target. `Sample` shrinks ~20×.
3. **Log every trajectory** `(seed, deck, stake, ruleset, action list, per-step summaries)`; exact replay via the
   deterministic engine; tag interesting ones (win, high ante, novelty of build, comeback, skip-heavy line).
4. **Never train on solo MLB with a free Nemesis.** The Nemesis must cost something: tournament outcomes
   (primary) or an external per-ante target (interim).
5. **Novelty targets exploring-starts, not the replay buffer.** (Not in this phase — design hook only.)
6. Lead commits on `mp/campaign` at phase close; agents do not commit.

## 1. Workstreams

| # | Workstream | Model | Owns | Depends on |
|---|---|---|---|---|
| **W1** | Set-based encoder + set-based action features + subsampled `Sample` | strong | `agent/mcts/{encoder_set.py,model_set.py,action_features_set.py}`, `agent/train/{trajectory.py,sample.py}`, tests, `agent/SETENC_NOTES.md`; may edit `agent/mcts/{policy.py,search.py,batched.py}` ONLY behind a flag/interface, serial path byte-identical | — |
| **W2** | Tournament-driven training loop (the real objective) + MCTS plug-in | strong | `agent/train/{selfplay.py,population.py}`, `agent/scripts/train_mlb.py`, `tournament/players.py::MCTSPlayer` (apply BATCH_NOTES §7.2), `tournament/runner.py` (`act_many` lockstep per §7.3 is OPTIONAL; remove `_repair_mlb_gameover_bug` — dead since the lead fix), tests, `agent/TRAIN_NOTES.md` | W1's `Sample`/encoder interface — coordinate through an agreed `PolicyValueFn` + `Sample` contract (read each other's notes; W1 publishes the contract FIRST in `SETENC_NOTES.md` §0 within its first hour) |
| **W3** | Trajectory logging + replay + tagging + viewer export | sonnet | `replay/**` (new: `log.py`, `replay.py`, `tags.py`, `export_viz.py`, `cli.py`, tests), `replay/REPLAY_NOTES.md`; a `TrajectoryLogger` hook that W2's loop and the tournament call (define it; W2 wires it) | engine only |
| **W4** | Transfer-spread harness + interim external-target objective + cleanups | sonnet | `eval/transfer_spread.py`, `eval/targets.py` (per-ante external Nemesis targets; expose to `agent` via a tiny importable module), `eval/tests`, `eval/EVAL_NOTES.md` §new; `tournament/tests` additions for the cleanup | engine only |

### W1 — Set-based encoder (strong)

- Items: playing card (rank, suit, enhancement, edition, seal, debuffed, face-down, in-hand/selected/in-deck
  counts), joker (key embedding over the 150 game keys, edition, position, sell value, per-joker scalar state
  normalised, stickers), consumable (key embedding), shelf item (kind + key + cost + edition), pack (kind), voucher
  (key), blind (kind/boss key/target log-scale/is_pvp/chips scored/hands/discards), scalars (money, ante, lives,
  comeback state, hand-size, slots, skips, tags held). Each set → shared embedding → masked mean+max pool (or a
  small attention block — pick one, justify, keep params ≲ 3M). Trunk concatenates pooled sets + scalars.
- Action features: action type embedding + pooled embedding of the selected cards (for play/discard) + target
  item embedding (buy/sell/use/pick) + scalars (count, would-be hand type from `hand_eval`, cost). Policy head =
  pointer-style score per candidate action, same as today.
- `Sample` v2: obs as a dict of padded arrays + masks; `action_feats` only for the SUBSAMPLED candidate set
  (visited ∪ k random unvisited, k=8 default), `policy_target` renormalised over that set, `value_target`, and
  metadata (seed, ante, state). Bit-exact checkpoint round-trip must still hold.
- Test: on 200 logged states from the overnight run (`agent/runs/overnight_2026-08-22/` — read-only), the
  set encoder produces finite, mask-invariant outputs (permuting items within a set changes nothing; padding
  changes nothing); a 10-minute cold run on the flat encoder vs the set encoder, paired by seed, via
  `eval/eval_harness.py --compare` — report it, no claim needed.

### W2 — Tournament-driven training (strong)

- `train_mlb.py`: generations. Each generation: population = current net (×m root-noise seeds / sims budgets) +
  last p checkpoints (heterogeneity); for each of s seeds run `Tournament(n=N, life_rule="paired", max_ante=A)`
  with `MCTSPlayer`s; at every Nemesis the N×N matrix yields, per current-net agent, **value target = population
  rank ∈ [0,1]** (dense) and **auxiliary pairwise outcome vs its assigned opponent** (faithful, zero-sum);
  samples from current-net agents (obs, subsampled visit distribution, targets) go to the buffer; train; checkpoint
  → population. Use `TrajectoryLogger` (W3) for every episode. Bootstrap: short-horizon target (rank at the NEXT
  Nemesis) blended with match outcome — make the blend a config and document the default.
- Must log per generation: value-target sd (the collapse detector — overnight was 0.07), skip rate, distinct joker
  sets / archetype counts (the strategy-diversity criterion), tie fraction (degeneracy), sims/s, ep/min.
- Throughput: runner drives one agent at a time → `leaf_batch=16, reuse=True` (BATCH_NOTES §7.1). If you do §7.3
  lockstep `act_many`, measure it; don't block on it.
- Interim objective for solo runs: `--objective external` using W4's `targets.py` so `train_cold`-style solo runs
  are non-degenerate (skipping + not building must lose a life at the Nemesis).
- Gate: a 30-minute `train_mlb` run on CUDA with N=16, s=2, A=4: value-target sd > 0.15 throughout, checkpoints
  land and resume, trajectories logged, no errors. Then **hand the lead the exact command for the first real run**
  (Red/White, sized for ~24–48 h of GPU) with expected ep/min and generation time.

### W3 — Trajectories (sonnet)

- `TrajectoryLogger` hook: `begin(game_meta) / step(action, summary) / end(outcome)`; writes compact JSONL (one
  line per episode: seed, deck, stake, ruleset, lives, actions[], per-step `(state, ante, blind, money, lives,
  chips)` summaries, outcome, tags). Few KB per episode.
- `replay.py`: re-run an episode from its line through the engine and assert `state_signature()` at every step
  matches the logged one (store a signature every k steps for this). `cli.py`: `show <file> <idx>` prints a
  readable line-by-line narrative; `filter --tag`.
- `tags.py`: win / reached ante ≥ k / lives-lost pattern / skip-heavy / archetype (joker-set novelty vs a running
  counter) / comeback — pure functions over the line.
- `export_viz.py`: write the `trajectory.json` shape the V7-era `viz/` (repo root, read-only) expects, so the old
  visualiser renders a new-engine line — best effort; document what doesn't map.
- Investigate (≤30 min, notes only): the MP mod's ghost-replay format (`$MOD/lib/replay_log.lua`,
  `ghost_replay.lua`, `log_parser.lua`) — what a recorded opponent log contains and whether our trajectory can be
  written into it. Report feasibility; don't build it.

### W4 — Transfer spread + targets + cleanups (sonnet)

- `targets.py`: per-ante external Nemesis targets: `vanilla_boss(ante, deck, stake)`, `k × own Big-blind score`,
  and a table derived from `results/` tournament score distributions (median at each ante). Importable from
  `agent` without circularity (tiny module, engine-only deps).
- `transfer_spread.py`: given a player spec (scripted now, `checkpoint:` later), evaluate over the 126 seeds on
  Red / Checkered / Plasma at White via the eval harness in SP-MLB-solo mode with an external target, AND via
  the tournament (N=32 scripted+checkpoint mix) — output per-cell distributions + the cross-cell spread
  (variance of rank / win-rate across cells). Run it for real with scripted players and write the numbers.
- Cleanups: delete `tournament/runner.py::_repair_mlb_gameover_bug` and its test (lead fixed the engine; add a
  test that a Hook-rejected exhaustion under MLB costs a life through the runner); check `env_v7`'s `(9-ante)`
  reward is not reachable from any MLB path (it's reward-only; document).

## 2. Exit gate

1. Set encoder trains; paired comparison vs flat encoder reported. `Sample` v2 ~20× smaller; checkpoint round-trip
   bit-exact.
2. `train_mlb` 30-min gate passes with value-target sd > 0.15 and diversity metrics logged.
3. Every episode logged; replay reproduces signatures; tags + viewer export work.
4. Transfer-spread numbers for scripted players on three decks.
5. All prior gates green; engine/rng unchanged.
6. **Lead launches the first real training run** from W2's command, with a watchdog, and commits Phase 4.

## 3. For Tagg

Nothing blocking. The first real run will take 1–2 days; the read-out you want is the diversity metrics
(does it ever build non-generic?) and the value-target sd staying alive.
