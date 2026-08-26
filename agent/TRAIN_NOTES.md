# TRAIN_NOTES — Phase 4 W2: tournament-driven training + the MCTS plug-in

**Agent W2, 2026-08-22.** Deliverable: the training loop whose objective is not degenerate.

New: `agent/train/{selfplay.py, population.py}`, `agent/scripts/{train_mlb.py,
smoke_3way.sh, smoke_3way_report.py}`, `agent/tests/test_train_mlb.py` (69 tests), this
file. Edited: `agent/mcts/player.py` (an additive `record_hook` + `legal_filter` +
`batch_leaf_eval`, and a real bug in `make_player`), `tournament/players.py` (the
`MCTSPlayer` factory, BATCH_NOTES §7.2 applied), `tournament/runner.py` (dead Phase 3
workaround removed; a no-progress guard; `on_fanout` / `on_step` / `on_agent_done` for W3's
replay logging) plus four new tournament test files (25 tests).
**`engine/**`, `rng/**` and `eval/**` untouched.**

---

## 0. Headline

The overnight shakedown (CAMPAIGN_LOG 07:35) showed the pipeline learning an objective worth
nothing: under solo MLB the Nemesis is free, so the optimal policy is *skip 15 of 16 blinds
and coast*, and the value target's standard deviation collapsed to **0.07**. This workstream
replaces the objective, and then measures what the replacement actually rewards.

| | overnight `train_cold --ruleset mlb` | `train_mlb` (this workstream) |
|---|---|---|
| value-target sd | **0.07**, and falling | **0.20 – 0.29**, every generation of every gate run |
| what a Nemesis costs | nothing | your rank against 15 other runs of your own seed |
| distinct joker sets / generation | ~1 (nobody builds) | 12 – 29 of 32 runs, rising with training |
| mean jokers held at the end | 0 – 4, static | 1.2 → 3.4 over 11 generations |
| runs losing all 4 lives | — | 21/32 → 10/32 over 8 generations |

The mechanism is the tournament: N agents on ONE seed, so "I skipped everything" is measured
against fifteen other runs of the same shops, the same shuffles, the same bosses. Rank is a
dense target in [0, 1] whose spread is a property of the *population*, not of the policy —
which is why it cannot collapse the way `MLBOutcome` did, and why `tie_fraction` is logged
next to it as the thing that would have to go wrong first (it stays under 0.09).

**The honest caveat, measured twice and not talked around:** at `--max-ante 4` a cold net
converges on skipping 90-99% of Small/Big blinds anyway, and it is *right* to. §7.2 is the
whole story, and §8's recipe is built around it.

## 1. The objective, precisely

For a decision made by a current-net agent `i` while `game.ante == a`:

```
rank_next = normalised population rank of i at the FIRST Nemesis ante >= a
            1.0 = highest score in the population, 0.0 = lowest, ties averaged
            0.0 if i is absent from that matrix (it was eliminated before reaching it)
            the final standing if there is no later Nemesis
outcome   = normalised final standing of i over the whole match, ranking agents by
            (Nemesis rounds survived, final lives, last Nemesis score)
z         = value_blend * rank_next + (1 - value_blend) * outcome
policy    = the search's root visit distribution at that decision
```

**`--value-blend` defaults to 0.7.** The short-horizon term is the one with real variance and
a short credit-assignment path — a per-ante comparison against N−1 runs of the same seed —
while the match term is what stops the agent trading the whole run for one good ante. 0.7/0.3
keeps the target dominated by the signal that moves per decision while still ordering "won
the ante, died at the next one" below "won the ante, survived". Both terms are dense ranks,
so with N = 16 a uniform rank alone has sd 0.30 and the blend cannot fall much below that
unless the population has collapsed.

The single most load-bearing line is `assign_value_targets`'s treatment of an **absent**
agent: NaN would be the mathematically obvious choice and it is the wrong one. Absent means
*eliminated before the next Nemesis*, i.e. the worst outcome there is, and it is scored 0.0.
If it were dropped or imputed to the mean, losing your last life would be free — the exact
shape of the degeneracy this workstream exists to remove. `test_an_eliminated_agent_gets_the_
worst_possible_short_horizon_target` pins it.

**Only current-net agents produce samples.** Past selves are opponents; learning from their
decisions would be imitation of a worse net. They still count in the N×N matrix, which is
what makes the rank mean anything.

### Population (`train/population.py`)

Three axes of heterogeneity, in descending order of how much signal each carries:

1. **Past selves** — the last `--p-history` checkpoints (a genuinely different policy).
2. **Search budget** — `--sims-budgets 1.0,0.5,1.5` cycled over the seats; a 60-sim agent and
   a 20-sim agent are different players.
3. **Root Dirichlet noise + per-agent rng seed** — the only axis available at generation 0.

…and a fourth thing that is not diversity but a **reference**:

4. **Scripted anchors** (`--anchors`, default 0.25 of the population). All three axes above
   are the current net or a recent version of it, so **a habit the whole lineage shares is
   invisible to a rank target**. That is not hypothetical — it is what the first gate run
   found. See §7.1.

`build_population(cfg, generation, history, base_seed)` is a pure function, which is what
makes `--resume` face the identical population (`test_resume_faces_the_same_population_it_
would_have`). A pruned checkpoint drops out of the pool silently rather than crashing a 24 h
run (`CheckpointHistory.existing()`), and `prune_checkpoints` refuses to delete a checkpoint
that is still an opponent.

### Interim solo objective — `--objective external`

`play_solo_external_episode` charges the life the engine will not. With `pvp_solo=True` a
Nemesis auto-resolves at hand exhaustion with **no life lost** (`game.py::end_pvp`), so the
driver compares `chips_scored` to an external per-ante target and calls the same public
`game.lose_life()` hook the tournament runner uses. Targets come from **W4's
`eval/targets.py`** through its shared `target_fn(game, big_blind=None) -> int` signature;
`selfplay.vanilla_boss_target` is a local fallback with the same formula from the same engine
constants, kept so the module works if `eval` is ever absent. `load_target_fn` returns the
source string and the run log records it — a run trained against a different target table is
a different experiment.

Four details that are easy to get wrong, three of them found by getting them wrong:

* **The default target is `own_big_blind`, not `vanilla_boss`.** A fixed vanilla-Boss bar is
  unreachable for a cold net, every margin saturates the logistic, and the value target
  collapses for the second time (measured: sd 0.06, mean 0.47, every episode failing). W4's
  mirror target is ~50/50 by construction.
* **…and a self-referential target needs a floor.** `own_big_blind` reads the agent's own
  Big-blind score that ante; ours *skips* the Big blind, so the score is 0, the target is 0
  and the Nemesis is free again — the overnight degeneracy reached by a different road
  (measured: 77.5% skip, every Nemesis cleared, z mean 0.84). `--target-floor` (default 1.0 ×
  the ante's vanilla Big-blind amount, `selfplay.big_blind_floor`) closes it without touching
  W4's function: playing the Big blind already clears the floor, so it only bites on a skip
  or a fail.
* **The target must be re-applied via `set_pvp_info` on every `SELECTING_HAND` decision**,
  because `_start_blind` (game.py:1005) zeroes a PvP blind's `chips_target` at `play_blind`.
  `--no-pvp-relay` turns the relay off, which matches the tournament's own contract (every
  agent plays its Nemesis blind) and makes solo trajectories fully replayable — see §9.
* **The search's `OutcomeFn` (`external_outcome_for`) derives the target from the game's
  ante**, never from `current_blind.chips_target`, so a subtree that crosses a blind start
  does not lose it; and it values a leaf sitting at a resolved-but-not-cashed-out Nemesis *as
  if* the pending life had already been taken. Without that the search cannot see the cost of
  coasting even though the driver charges it.

Value targets mirror the tournament's shape: a logistic of the log-margin against the next
Nemesis's target, blended with the final `MLBOutcome` value.

**This objective removes the "always wins" degeneracy — the mean moved from 0.84 to ~0.45 and
lives are genuinely lost — but it cannot hold a high value-target sd, and that is structural
rather than a bug.** An absolute target clusters when every episode fails or every episode
succeeds; measured 0.06-0.13 across both target kinds and multipliers 0.25 / 0.5 / 1.0. The
collapse alarm is objective-aware and says "target mis-scaled for this agent" here. **The
tournament is the real objective**, and this is exactly why.

## 2. How samples are collected without owning the play loop

`tournament/runner.py` drives the games; the players are handed to it. So collection rides
on an **additive** `record_hook` on `mcts.MCTSPlayer` (`None` by default, and free when
unset — no dict is built, no extra `legal_actions()` call is made). Every decision hands back
a `mcts.Decision` (live game, legal actions, their keys, root visit counts, chosen key) and
`train.selfplay.SampleCollector` turns it into a `Sample` with a placeholder value. Values
are filled in afterwards from the tournament's own `AnteMatrix` objects — no second source of
truth. `test_record_hook_is_free_when_unset` pins that a player with a hook plays the
identical game to one without.

Two consequences worth writing down:

* the hook fires **before** the step, which is why trajectory logging does *not* ride on it
  (see §4);
* with `reuse=True` a reused root's visit counts include simulations spent on it in earlier
  decisions. That is more evidence, not stale evidence — `clone().step(a)` is deterministic on
  this engine (Phase 1 dividend) — so the policy target is over accumulated visits,
  deliberately.

### The W1 seam — wired

`make_sample(game, legal, legal_keys, visits, encoder, z)` is the only sample constructor and
`SampleCollector.sample_fn` is the only place it is chosen. W1 published the contract in
`SETENC_NOTES.md` §0 and built `train.sample.SampleBuilder` with exactly this signature, so
`MLBTrainer` passes `ColdTrainer.sample_builder` straight through: `--encoder set` and the
default `--subsample` both produce subsampled `Sample` v2 records, and `--no-subsample
--encoder mlb` falls back to the local v1 constructor unchanged. Measured on the gate run,
v2 samples are ~7 KB against v1's ~97 KB, which is what makes a 20 000-sample buffer a 140 MB
proposition instead of a 1.9 GB one.

---

## 3. `MCTSPlayer` in the tournament, and what was removed

`tournament/players.py::MCTSPlayer` is now BATCH_NOTES §7.2's factory verbatim: a
function, not a class, so `tournament` still imports with no torch installed, and
`agent` goes on `sys.path` the way `bootstrap.py` already does it for `engine`.
Defaults are §7.1's recommendation — `leaf_batch=16, reuse=True` — because the runner drives
one agent at a time, so K is 1 no matter what the player does.

**`_repair_mlb_gameover_bug` is gone.** The lead fixed the engine at the Phase 3 close
(`TestBossRejectionRespectsMLB`), so the workaround was dead code that could only ever mask a
future regression. `tests/test_boss_rejection_life.py` is the runner-level replacement: a
Hook/Eye/Mouth-rejected exhaustion costs one life and the drive continues to the next
Nemesis; on the last life it ends the run; and `GAME_OVER with lives > 0` — the exact
signature the repair detected — never occurs.

**§7.3 lockstep `act_many` was deliberately NOT done**, and the reason is in BATCH_NOTES's own
table B2: `leaf_batch=16` on a single tree measures **1034 sims/s** on this box, while
cross-tree batching at K=32 with `leaf_batch=1` measures **761**. Lockstep would make the
runner's drive loop, its exhaustion bookkeeping and its life rule all move inside a new
loop — a real rewrite of the module W3's replay recipe and four test files now depend on —
to reach a configuration that is *slower* than the one the plug-in already uses. If the
profile ever changes (it would take fixing the 436-actions-per-leaf cost first), the sketch is
still in §7.3.

---

## 4. Trajectory logging (W3) — wired, and where it had to go

W3's `TrajectoryLogger` is one `BalatroGame` per `begin()/end()` pair; a tournament is N
independent games, and the runner is the only place that sees every mutation. So three hooks
were added to `Tournament`, and the logging happens there rather than in the record hook:

```
on_fanout(games, seed_str)              once, the instant the N games exist
on_step(agent_idx, game, action)        AFTER every step this module performs
on_agent_done(agent_idx, game, reason)  exactly once per agent
```

`on_step` covers the agent's own actions, the no-progress guard's forced action, and
`_cash_out`'s advance, and it emits the synthetic `{"type": "__lose_life__"}`
(REPLAY_NOTES §2.3) after the cross-agent life rule — the one life lost without a `step()`.
`on_agent_done` fires for an eliminated agent **before** `Tournament.run` force-sets
`State.GAME_OVER`, because that assignment is an out-of-band mutation no replay reproduces
and the logger has to take its final signature first. The solo driver logs the same way
(after the step, `__lose_life__` for the external life charge) and deliberately does **not**
set `GAME_OVER` when the external rule empties the lives — `final_lives == 0` plus
`stop_reason="out_of_lives"` says it, and the episode stays replayable.

`--sig-every 50` (not W3's default 10): signature capture costs 25–35% of wall clock at 10
and under 2% at 50. Every agent is logged, not just the sample producers; `meta.is_current`
distinguishes them. `test_logged_tournament_trajectories_replay_exactly` and
`tournament/tests/test_trajectory_hook.py::test_replays_a_logged_tournament` replay real logs
through W3's verifier — which is also the strongest available check that the hooks see every
mutation.

---

## 5. Metrics, and the alarm

One JSONL line per generation plus one console line. `GenerationMetrics.collapsed` is true
when `value_target_sd <= 0.15` and the console shouts `** VALUE-TARGET COLLAPSE **`.

| metric | definition | why |
|---|---|---|
| `value_target_sd` | sd of `z` over this generation's samples | the collapse detector; overnight was 0.07, alarm below 0.15 |
| `value_target_mean` | mean `z` | ~0.5 if the current net is mid-population; below that means it is losing |
| `skip_rate` | (skip_blind chosen) / (BLIND_SELECT decisions where the state offered a skip) | the overnight pathology, measured directly. The denominator comes off the STATE, not off the candidate set, so `--max-skips-per-ante` shrinks the numerator and not the denominator |
| `blind_clear_rate`, `blinds_played` | of the regular blinds actually played, the fraction cleared (read at every non-PvP `ROUND_EVAL`: `chips_scored >= chips_target`) | the other half of the skip story. High skip + low clear = a net that cannot play blinds yet. High skip + high clear = a strategy |
| `rank_current` / `rank_anchor` / `rank_history` | mean normalised rank per seat group over every Nemesis | the skip-vs-build referendum. If skipping were winning, `rank_current` would be above `rank_anchor` |
| `distinct_joker_sets`, `joker_top5`, `mean_jokers` | over every agent's final board | the strategy-diversity criterion: does it ever build non-generic? |
| `tie_fraction` | mean over the generation's `AnteMatrix.tie_fraction` | population degeneracy; if this goes to 1 the rank target is meaningless |
| `lives_lost` | histogram of `starting_lives − final_lives` | does the objective actually cost anything |
| `max/mean_ante_reached` | over every agent | progress |
| `sims_per_s` | mean seat budget × searches / wall | nominal search throughput |
| `leaf_evals_per_s` | net evaluations / wall, `NaN` when the policy exposes no counter | the honest one; differs from the above by design, because tree reuse spends fewer sims for the same evidence |
| `ep_per_min` | agent-runs per minute (`seeds × N`) | run sizing |
| `skip_cap`, `clear_rate_ema` | the cap in force this generation and the EMA driving its anneal | so the log says when the constraint came off |
| `train_steps`, `policy_loss`, `value_loss` | this generation's optimisation | |

**On `sims_per_s` and BATCH_NOTES.** W3 measured 1 034 sims/s single-tree at
`leaf_batch=16`; this loop reports 320-550. They are not the same measurement. W3's is 500
simulations on one vanilla ante-1 blind; this is 40 simulations per decision averaged over a
whole MLB run, including Nemesis states with ~400 legal actions and shop states. At 40 sims
the per-decision fixed costs — expanding ~436 root children, allocating the tree — amortise
over 40 simulations instead of 500. Measured directly, interleaved, on this box at 40 sims:
`leaf_batch` 1 / 4 / 16 gives **238 / 217 / 218** sims/s on CUDA and **249 / 251 / 247** on
CPU, i.e. the leaf batch is not the lever at this operating point, and L=1 is also the exact
reference search. Hence `--leaf-batch 1` (§8). Two real bugs turned up while checking this,
both fixed: `make_player` passed `leaf_batch` only into `MCTSConfig` and not into the
`MCTSPlayer` field that overrides it, so **every tournament player was silently running at
L=1** regardless; and `MCTS._drive` answers leaf requests one at a time by design, so
`leaf_batch>1` never batched a forward pass at all until `MCTSPlayer.batch_leaf_eval` routed
it through `BatchedSearch`.

## 6. Pause / resume

```
touch <run-dir>/PAUSE
```

Checked between tournaments (and between episodes under `--objective external`) — one
tournament is the atomic unit of play, so the loop finishes the one in flight, trains on what
it collected, checkpoints, prints the exact resume command and exits 0. `SIGINT` / `SIGTERM`
take the identical path. `--resume` restores net, optimizer moments, replay buffer,
numpy/torch/python RNG states, generation counter and the opponent-checkpoint history, and
deletes the PAUSE file for you.

A `train_mlb` checkpoint **is** a `train_cold` checkpoint (`CHECKPOINT_KIND` unchanged) with
an extra `"mlb"` block, so `mcts.load_policy` reads it directly — which is how the
population's past selves are loaded, and why `MLBTrainer` is a thin shell over `ColdTrainer`
rather than a parallel implementation.

---

## 7. The gate runs

Two 30-minute runs on CUDA (RTX 3080 Ti), both `--n-agents 16 --seeds-per-gen 2
--max-ante 4 --sims 40 --device cuda`, `mlb` encoder, 2.41 M-parameter net, subsampled
`Sample` v2. Both include a `PAUSE` mid-flight and a `--resume`. **0 errors in either.**

### 7.1 `p4w2_gate` — no anchors (`agent/runs/p4w2_gate/`)

10 generations, 288 agent-runs, 32.6 min (25.9 min → `PAUSE` → resumed for 6.7 min),
11 checkpoints.

| gen | wall s | eps | ep/min | samples | **z sd** | z mean | skip | joker sets | mean jokers | tie | mean ante | sims/s | v loss |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 195 | 32 | 9.8 | 1050 | **0.294** | 0.488 | 75% | 15 | 0.81 | 0.004 | 3.66 | 392 | 3.4787 |
| 1 | 181 | 32 | 10.6 | 1071 | **0.286** | 0.454 | 52% | 14 | 0.91 | 0.000 | 2.88 | 386 | 0.0918 |
| 2 | 207 | 32 | 9.3 | 1122 | **0.290** | 0.335 | 49% | 16 | 0.78 | 0.000 | 3.12 | 374 | 0.1003 |
| 3 | 204 | 32 | 9.4 | 1147 | **0.283** | 0.534 | 88% | 13 | 1.16 | 0.002 | 3.91 | 371 | 0.0925 |
| 4 | 205 | 32 | 9.4 | 1012 | **0.245** | 0.523 | 93% | 22 | 1.69 | 0.002 | 3.94 | 350 | 0.0783 |
| 5 | 225 | 32 | 8.5 | 1205 | **0.232** | 0.618 | 97% | 19 | 1.62 | 0.002 | 4.50 | 374 | 0.0901 |
| 6 | 208 | 32 | 9.2 | 1081 | **0.251** | 0.534 | 95% | 29 | 3.69 | 0.001 | 4.56 | 365 | 0.0690 |
| 7 | 125 | 16 | 7.7 | 627 | **0.249** | 0.537 | 98% | 15 | 3.88 | 0.003 | 4.75 | 331 | 0.0659 |
| — | | | | | | | | | | | | | *`PAUSE` → checkpoint → exit 0 at 25m52s; resumed at generation 8* |
| 8 | 262 | 32 | 7.3 | 1226 | **0.276** | 0.499 | 88% | 28 | 2.88 | 0.001 | 4.66 | 319 | 0.0709 |
| 9 | 140 | 16 | 6.9 | 472 | **0.239** | 0.278 | 70% | 13 | 2.69 | 0.000 | 3.88 | 230 | 0.0714 |

Generations 7 and 9 played one seed instead of two: `PAUSE` and the deadline are both
checked *between tournaments*, so the generation in flight closed out on what it had. That
is the pause path working, not a truncation bug.

**Gate criteria: value-target sd > 0.15 throughout — yes, minimum 0.232 against an alarm of
0.15 and an overnight baseline of 0.07. Checkpoints land (11) and resume. Metrics logged.
0 errors.** Trajectory logging was re-wired after this run (§4) and re-verified separately;
`p4w2_gate/trajectories.jsonl` is from the superseded wiring and does not replay.

### 7.2 `p4w2_gate2` — with scripted anchors, and what it proved

Same settings plus `--anchors 0.25 --log-trajectories --leaf-batch 1`. 11 generations,
352 agent-runs, 27.5 min to `PAUSE`, 12 checkpoints, 0 errors.

| gen | wall s | ep/min | samples | **z sd** | z mean | skip | rank cur / anchor | joker sets | mean jokers | tie | mean ante |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 154 | 12.5 | 1006 | **0.254** | 0.296 | 65% | 0.34 / 0.80 | 21 | 1.94 | 0.054 | 3.69 |
| 1 | 186 | 10.3 | 1124 | **0.218** | 0.265 | 59% | 0.27 / 0.88 | 12 | 1.16 | 0.087 | 3.47 |
| 2 | 184 | 10.4 | 1178 | **0.215** | 0.349 | 71% | 0.34 / 0.89 | 13 | 1.34 | 0.055 | 3.81 |
| 3 | 148 | 13.0 | 1183 | **0.215** | 0.397 | 84% | 0.36 / 0.91 | 18 | 1.94 | 0.022 | 3.91 |
| 4 | 116 | 16.6 | 1191 | **0.199** | 0.416 | 93% | 0.37 / 0.90 | 23 | 2.69 | 0.018 | 4.19 |
| 5 | 131 | 14.7 | 1105 | **0.202** | 0.493 | 98% | 0.46 / 0.88 | 23 | 1.97 | 0.014 | 4.19 |
| 6 | 163 | 11.8 | 1161 | **0.238** | 0.474 | 97% | 0.45 / 0.87 | 24 | 2.44 | 0.017 | 4.44 |
| 7 | 133 | 14.5 | 975 | **0.213** | 0.459 | 99% | 0.42 / 0.90 | 27 | 3.38 | 0.011 | 4.50 |
| 8 | 139 | 13.8 | 1252 | **0.203** | 0.400 | 91% | 0.35 / 0.91 | 28 | 3.56 | 0.014 | 4.34 |
| 9 | 136 | 14.1 | 1095 | **0.282** | 0.555 | 90% | 0.55 / 0.52 | 26 | 2.56 | 0.009 | 4.00 |
| 10 | 156 | 12.3 | 1242 | **0.214** | 0.430 | 94% | 0.39 / 0.90 | 24 | 2.25 | 0.027 | 4.47 |

The anchors did exactly what they were added to do to the *target*: the current net's mean
rank sits at 0.27-0.55 while the scripted anchors sit at 0.87-0.91, so the value head is now
being told, correctly and every ante, that this policy is losing. What they did **not** do is
change the policy: the skip rate still climbs to 99%.

**That is not a bug in the objective, and it is worth being precise about why.** At
`--max-ante 4` a cold net cannot clear an ante-3 Big blind (3 000 chips with 0-2 jokers).
Playing it costs a life; skipping it costs a tag's worth of tempo. Skipping is therefore
*locally optimal against any population*, anchors included — the anchors reach ante 3-4 and
then bleed lives at regular blinds themselves, so a skipper beats them by attrition inside a
4-ante horizon. And the other metrics say the net is genuinely improving while it skips:
mean jokers 1.16 → 3.56, mean ante reached 3.47 → 4.50, runs losing all four lives 21/32 →
10/32, distinct joker sets 12 → 28. It is learning "skip and build", which is a real MLB
strategy — it is simply learning it *before* learning to play a hand, which is the wrong
order for a curriculum.

So the fix is not a better target. **The net has to learn to clear blinds first**, and §8's
recipe is built on the two levers that address that directly: a vanilla warm-up where failing
a blind ends the run, and a training-time cap on skipping that is annealed away once the net
can clear blinds on its own.

### 7.3 Which lever: three 10-minute CUDA smokes

Stage A was `train_cold.py --minutes 10 --ruleset vanilla --encoder mlb --sims 40`: 644
episodes at 64.6 ep/min, 0 wins, mean ante 1.00, mean episode length 9.7 decisions. In
vanilla a failed blind is `GAME_OVER`, so a cold net's whole curriculum is the ante-1 Small
blind — narrow, and exactly the thing it has to learn. Ten minutes is a smoke, not a warm-up;
the real Stage A is hours.

Then three Stage-B variants, 10 minutes each, all `--anchors 0.25 --max-ante 4 --sims 40
--n-agents 16 --seeds-per-gen 2`:

| lever | skip rate (gen 0 → 4) | blind-clear rate | z sd | rank cur / anchor (final) | mean ante (final) |
|---|---|---|---|---|---|
| *baseline*: cold, anchors only (`p4w2_gate2`) | 65% → **99%** | not yet logged | 0.20-0.28 | 0.39 / 0.90 | 4.47 |
| **(a) warm start** (`p4w2_smoke_a`) | 77% → **70%**, flat | 0 - 9.5% | 0.22-0.27 | 0.47 / 0.96 | 3.44 |
| **(b) skip cap 1/ante** (`p4w2_smoke_b`) | 44% → **37%** | 0 - 11% | 0.22-0.25 | 0.34 / 0.77 | 2.72 |
| **(c) both** | not run — machine handed back | | | | |

**(a):** a *ten-minute* warm-up was already enough to stop the slide. The cold baseline goes
65% → 99% skip in five generations; the warm-started net sits at 65-77% and drifts *down*.
That is the strongest single result here, and it came from a Stage A that the run's own
metrics say barely learned anything (mean ante 1.00, 0 wins) — which is the argument for
giving Stage A hours rather than minutes.

**(b):** the cap does exactly what it says — 32-44% skip against the baseline's 99% — and it
costs something: mean ante reached drops to 2.72 against the warm start's 3.44 and the
baseline's 4.47. That is the failure mode the report script warns about in its own footer: a
lever that lowers the skip rate *without* teaching the net to clear blinds has only made it
play blinds it loses. The blind-clear rate stayed at 0-11% in every arm, so at ten minutes
none of them had learned to play a blind at all.

**(c) is unmeasured** (`smoke_3way.sh` runs it in ~40 min). §8 takes both levers anyway, for
a reason the numbers support rather than contradict: they are independent, the cap anneals
itself off the moment the blind-clear rate passes 0.5, and the thing that makes the cap
harmful — a net that cannot clear blinds — is precisely what Stage A exists to fix. Run the
script before committing 24 hours of GPU; if (b) alone still costs antes with a real Stage A
behind it, drop `--max-skips-per-ante` and keep the warm-up.

One number to carry forward regardless: **`blind_clear_rate` is the metric that matters
here**, not `skip_rate`. Skipping is a symptom.

## 8. First real run

**Two stages.** Stage A teaches the net to play a blind; Stage B trains the tournament
objective. §7.2 is why: the objective is sound and the metrics prove it works, but a net that
cannot clear an ante-3 Big blind will correctly learn to skip everything, and it will learn
that before it learns to play a hand. Stage A removes the option — in vanilla, a failed blind
is `GAME_OVER`.

### Stage A — vanilla warm-up (4-6 h)

```
python agent/scripts/train_cold.py \
    --minutes 300 --device cuda \
    --ruleset vanilla --encoder mlb \
    --sims 40 --max-decisions 1500 \
    --batch-size 32 --lr 1e-3 --buffer-capacity 20000 \
    --checkpoint-every 200 --keep-checkpoints 6 \
    --run-dir agent/runs --run-name real1_stageA
```

Measured: **64.6 episodes/min** on CUDA (10-minute run, 644 episodes, 0 errors), so 300
minutes is ~19 000 episodes. Episodes are short at first (mean length 9.7 decisions, mean
ante 1.00) because a cold net dies at the ante-1 Small blind; they lengthen as it learns,
which is the signal to watch.

**Move on when** the furthest-ante distribution is climbing off 1.0 and `z` is rising. If
after 6 h the mean ante is still 1.0, do not start Stage B — raise `--sims` (the search, not
the net, is what clears a first blind) and say so.

### Stage B — the tournament run (24-48 h)

```
python agent/scripts/train_mlb.py \
    --minutes 2880 --device cuda \
    --init agent/runs/real1_stageA/latest.pt \
    --encoder mlb \
    --objective tournament \
    --n-agents 16 --m-current 8 --anchors 0.25 --p-history 4 \
    --seeds-per-gen 2 --max-ante 8 --life-rule paired \
    --sims 40 --sims-budgets 1.0,0.5,1.5 --leaf-batch 1 \
    --value-blend 0.7 \
    --max-skips-per-ante 1 --skip-cap-anneal-clear-rate 0.5 \
    --batch-size 32 --lr 1e-3 --train-steps 128 \
    --buffer-capacity 20000 --buffer-checkpoint-cap 20000 \
    --checkpoint-every 1 --keep-checkpoints 12 \
    --log-trajectories --sig-every 50 \
    --deck b_red --stake 1 \
    --run-dir agent/runs --run-name real1
```

Red deck, White stake, ante-8 horizon. Pause it whenever the machine is wanted:
`touch agent/runs/real1/PAUSE`; resume with
`python agent/scripts/train_mlb.py --resume agent/runs/real1/latest.pt --minutes <N> --device cuda`.

**Expected throughput** (measured at `--max-ante 8`, N=16, `mlb` encoder, CUDA):

| | early generations | note |
|---|---|---|
| one tournament (16 agents to ante 8) | **120-150 s** | grows as the net survives longer; budget 1.5-2x by the end |
| one generation (2 seeds + training) | **~270 s** | `--train-steps 128` at batch 32 is ~4 s of it |
| generations / hour | **~13** early, ~7-9 late | |
| checkpoints / hour | **~13** (one per generation) | 29 MB each; `--keep-checkpoints 12` bounds the directory at ~350 MB + `latest.pt` (~150 MB with a 20 000-sample buffer) |
| agent-runs / min | **~7** | 32 per generation |
| in 24 h | ~250 generations, ~8 000 agent-runs, ~250 k samples | the buffer (20 000) turns over every ~18 generations |
| in 48 h | ~450 generations | |

**With `--encoder set` instead**, measured on the same box at `--max-ante 8`: **136-157
sims/s against the flat encoder's 354-469**, i.e. one tournament takes **295-417 s** and a
generation ~700-800 s — about **2.7x slower**, so ~110 generations in 24 h instead of ~250.
Tagg's decision §0.1 says the set encoder lands before the first real run, and the loop runs
on it unchanged (`test_the_loop_runs_on_w1s_set_encoder_and_produces_v2_samples`); this is
purely the throughput cost of that decision, stated so it is a choice and not a surprise.
Swap `--encoder mlb` → `--encoder set` in **both** stages if you take it — `--init` refuses a
mismatch.

### The watchdog: what to check, and when to stop

Read the console line, or `jq` the JSONL. Every threshold below is measured, not guessed.

| after | check | healthy | act if |
|---|---|---|---|
| every generation | `z sd` | > 0.15 (gates ran 0.20-0.29) | ≤ 0.15 for 3 straight generations → the objective has degenerated; check `tie_fraction` first |
| every generation | `tie_fraction` | < 0.1 (gates ran 0.009-0.087) | > 0.3 → the population has collapsed; raise `--anchors` or `--p-history` |
| **10 generations** | **`skip_rate`** | **falling, or < 0.8** | **> 0.8 and rising → stop.** This is the failure §7.2 documents. First move: confirm the skip cap is still on (`skip_cap` in the log) and lower it to 0; second: go back and give Stage A more hours |
| 10 generations | `blind_clear_rate` | rising off 0 | flat at ~0 after 20 generations → Stage A was too short; the cap will never anneal and you are training a skipper with fewer skips |
| 20 generations | `rank_current` vs `rank_anchor` | gap closing | gap widening → the net is losing ground to scripted play; the run is not learning |
| 20 generations | `mean_jokers`, `distinct_joker_sets` | rising (gate: 1.2 → 3.6, 12 → 28) | flat → nothing is being built |
| any | `errors` in the summary line | 0 | > 0 → read the traceback in the JSONL |

The skip cap lifts itself the moment `clear_rate_ema` passes 0.5, and the log records
`skip_cap: null` from that generation on, so the final policy is trained under the real MLB
rules. If it has not lifted by the end of the run, say so in the read-out — the number that
matters then is the blind-clear rate, not the skip rate.

### Not yet measured

`smoke_3way.sh` + `smoke_3way_report.py` run the (a) warm-start / (b) skip-cap / (c) both
comparison end to end in ~40 minutes; the machine was handed back before (b) and (c) could
run. What §7.3 has is (a) against the cold baseline, which is why the recipe above takes
BOTH levers: they are independent, (c) costs nothing extra, and the cap anneals itself away.
Run the script and, if (b) or (c) is clearly better on skip rate AND blind-clear rate,
adjust `--max-skips-per-ante` here before launching 24 hours of GPU.

## 9. Found, not fixed

### Needs engine change (frozen — worked around, not patched)

**`engine/balatro_sim/game.py:1433` + `:1854` — a legal action that changes nothing, in
the SHOP.** `legal_actions()`'s SHOP branch enumerates card-targeting `use_consumable`
actions against `self.hand`, which in the SHOP still holds the **previous blind's** cards
(nothing clears it at `_end_round`), and `_use_consumable` silently returns without consuming
the consumable when the application fails (`success = False`). The product is a legal action
whose result is bit-identical to the state it was taken from — an infinite loop for any agent
that likes it. Found immediately once the population became MCTS players rather than scripted
ones: one agent burned the entire 20 000-step `max_steps_per_drive` budget in a single shop,
and under a different noise seed the same pathology consumed 55 s of a 60 s generation and
produced **14 338 training samples** from one shop. Scripted and random-legal players never
hit it, which is why Phase 3's 100-agent smoke did not see it.

The real fix is in the engine and is two lines: don't offer card-target consumable actions
outside `SELECTING_HAND`, and make a failed `_use_consumable` either reject the action or
consume the card. Until then, `runner._drive_to_next_nemesis` carries a **no-progress
guard**: `state_signature()` before and after each step (42 µs, ~0.1% of a search-driven
decision), and after 8 consecutive steps that changed nothing the driver plays
`_force_progress_action` itself and counts it on `game._w2_forced_progress`. It is invisible
to any agent that ever changes the state — `tests/test_noop_guard.py::
test_guarded_and_unguarded_runs_agree_when_nothing_wedges` pins that, and the whole existing
tournament suite is unchanged — and `noop_budget=0` disables it. The solo external-target
driver carries the same guard (an episode was observed spending all 400 of its decisions in
one ante-1 shop), and `--max-samples-per-agent` (default 2 000) is the second line of defence
on the buffer.

**`engine/balatro_sim/game.py:894` `state_signature()` at 42 µs.** Fine at the driver
level (once per step, dwarfed by the search) but it is the reason the guard is a *driver*
feature and not a per-edge check inside the search. Already flagged by W3 in BATCH_NOTES §8.

**`replay/_util.py::apply_op` has no `__set_pvp_info__` (needs a W3 change, 3 lines).**
The solo external-target driver calls `game.set_pvp_info(target, hands)` so the agent can see
what it has to beat; that mutates `pvp_opponent_score`, `pvp_opponent_hands` and the blind's
`chips_target`, all of which `state_signature()` covers, and replay has no way to reproduce
it. The op is emitted (`selfplay.OP_SET_PVP_INFO`) so the log is complete the day `apply_op`
grows `elif atype == OP_SET_PVP_INFO: game.set_pvp_info(action["score"], action["hands"])`;
until then a `--objective external --log-trajectories` line stops verifying at its first
Nemesis. `--no-pvp-relay` avoids it entirely and matches the tournament's own contract (every
agent plays its Nemesis blind, TOURNAMENT_NOTES §2). **Tournament trajectories are unaffected
and verify exactly** — the runner never relays.

### Found, not fixed (this workstream, deliberate)

* **`MCTSPlayer.__post_init__` lets its `leaf_batch` FIELD overwrite `config.leaf_batch`.**
  So `MCTSPlayer(config=MCTSConfig(leaf_batch=16))` silently runs at L=1. This is not
  hypothetical — `mcts.make_player` did exactly that, which means *every* tournament player
  produced by BATCH_NOTES §7.2's factory was running at L=1 no matter what it was asked for.
  `make_player` now passes both and `test_the_leaf_batch_field_overrides_the_config_and_that_
  is_a_trap` pins the behaviour, but the constructor is still the trap it was; the real fix
  is `leaf_batch: Optional[int] = None` meaning "take the config's", and that is W1's file.
* **`MCTSConfig.leaf_batch > 1` never batched a forward pass.** `MCTS._drive` answers leaf
  requests one at a time on purpose (it is what keeps `run`/`run_gumbel` byte-identical to
  the pre-W3 implementation), so `leaf_batch=L` bought the virtual-loss tree shape and none
  of the amortisation. `MCTSPlayer.batch_leaf_eval` now routes such a search through
  `BatchedSearch`, which does batch — and at 40 sims it measures at **parity** (0.96-0.98x,
  interleaved, 4 repeats), because a leaf's cost here is CPU-side featurisation of ~400
  actions, not the GPU call. Kept on because it makes `leaf_evals_per_s` a real number
  instead of 0, and because it should pay at 500 sims. Not the throughput lever.
* **The interim `--objective external` cannot hold a high value-target sd, and that is
  structural.** An ABSOLUTE target clusters when every episode fails (weak agent) or every
  episode succeeds (target too low): measured 0.06-0.13 across `vanilla_boss`,
  `own_big_blind`, and multipliers 0.25 / 0.5 / 1.0. The alarm is objective-aware and says
  "target mis-scaled for this agent" rather than "collapse" there. The tournament's rank
  target has no such failure mode, which is the whole argument for it being the real one.
* **A self-referential target needs a floor.** W4's `own_big_blind` is ~50/50 by construction
  *for an agent that plays its Big blind*; ours skips it, so `big_blind[ante]` stayed 0, the
  target was 0, and the Nemesis was free again — the overnight degeneracy reached by a
  different road (measured: 77.5% skip, every Nemesis cleared, z mean 0.84). `--target-floor`
  (default 1.0 x the ante's vanilla Big-blind amount) fixes it without touching W4's
  function. Worth knowing before anyone else builds a self-scaling target.
* **`sims_per_s` is nominal.** With tree reuse a decision may spend fewer simulations than
  its budget for the same evidence, so `leaf_evals_per_s` is the honest number and both are
  logged. See §5 for why neither compares to BATCH_NOTES's 500-sim benchmarks.
* **The population shares one live policy object** for every current-net seat. That is
  correct (they *are* the same net) and it means `_leaf_count` has to de-duplicate policies
  when measuring throughput. Per-seat weight perturbation would need per-seat nets.
* **Opponent checkpoints load onto the trainer's device.** `--p-history 4` is 4 extra nets
  resident (~29 MB each at the default size). Harmless now; a `--history-device cpu` if the
  net grows.
* **The final-standing key is `(rounds survived, lives, log1p(last score))`, ranked as a
  tuple.** Deliberate — a composite scalar would need magic constants — but "survived one
  more ante" dominates any score difference. Intended; one function to change.
* **`--objective external` uses one player object across a generation's episodes**, so its
  `searches` counter is cumulative and its rng stream is shared. Deliberate (the rng is the
  trainer's, which is what makes `--resume` exact), but per-episode search stats are not
  separable there.
* **`m_current` seats all play with root noise on.** Exploration is the point, but the
  samples come from a noised policy while the checkpoint that gets evaluated is not. Same
  trade AlphaZero makes; noted because the eval harness will see a slightly stronger player
  than the buffer describes.
* **Anything a driver attaches to the game object changes its signature.** The no-progress
  guard originally counted its firings as `game._w2_forced_progress`, and
  `state_signature()` (game.py:923) sweeps up *every* int/float/str/bool attribute of the
  game — so a diagnostic counter silently changed the run's signature and every trajectory
  it appeared in stopped replaying. Symptom: `ReplayMismatch` at whatever step the guard
  first fired, which looked like a booster/RNG bug because the guard fires in shops and the
  next signature checkpoint often landed on a `pick_booster`. The counter now lives on
  `TournamentResult.forced_progress`, and
  `test_noop_guard.py::test_the_guard_leaves_no_trace_on_the_game` asserts the driver adds
  no attribute the engine did not. Worth knowing before anyone else instruments a game:
  `mlb_match_demo`'s `_w4_visit` is safe only because it is a dict.
* **`--buffer-checkpoint-cap` defaults to 5 000**, so a run whose buffer exceeds that prints
  "resume is NOT bit-exact" at every resume. §8's command raises it to the full buffer
  capacity; a v2 sample is ~6 KB, so 20 000 of them is ~120 MB of `latest.pt`.



---

## Running it on all the cores (Phase 5 W1, 2026-08-23)

`train_mlb.py --workers N --evaluator-device {cpu,cuda,local}` plays the tournament in N
worker processes feeding ONE shared batched evaluator. `--workers 0` (the default) is this
document's single-process path, unchanged. The **checkpoint format is identical and the
worker count is not in it**, so a run can be paused and resumed across the seam in either
direction:

    touch <run dir>/PAUSE
    python agent/scripts/train_mlb.py --resume <run dir>/latest.pt --minutes <N>         --device cpu --workers 12 --evaluator-device cpu

Everything else -- encoder, sims, budgets, skip cap and its anneal, W0's lambda and its
clear-rate EMA, the opponent history, the buffer, the optimizer moments, the generation
counter -- comes back out of the checkpoint; only pass the flags you want to change.
`--log-trajectories` still works (each worker writes a part file, merged per generation).

The architecture, the transport with its measurements, the determinism contract, the
throughput benchmark and what is found-not-fixed are all in **`PARALLEL_NOTES.md`**.
