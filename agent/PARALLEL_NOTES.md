# PARALLEL_NOTES — Phase 5 W1: multi-process self-play with a shared batched evaluator

Built 2026-08-23 while `runs/real1` (Stage B, single process, one core) was live. Nothing
here touched that run, and nothing here touches `mp/engine/**`, `mp/rng/**`, `mp/eval/**`
or `mp/replay/**`.

---

## 0. Headline

| | |
|---|---|
| **What** | N worker processes own the tournament's agents; ONE evaluator batches their MCTS leaves through the net. `Tournament` is untouched; `--workers 0` is the old path, byte for byte. |
| **Transport** | shared-memory arena per worker (127 KB per leaf, ~370 MB/s at 16 workers) + a `Queue` carrying only offsets. |
| **Checkpoints** | identical format, both directions. `--resume runs/real1/latest.pt --workers 12` continues the live run; a checkpoint the parallel trainer writes resumes single-process. Verified live, both ways. |
| **Determinism** | results do not depend on the worker count — measured **exact** (matrices, lives, value targets, sample count) for 1 vs 4 workers with a real net. |
| **Measured (≤4-worker smoke, N=8, sims 40, max_ante 4, CPU, *while `real1` held a core*)** | serial **229 sims/s** → 1 worker **268** (1.17×) → **4 workers 531 sims/s (2.32×)** |
| **The real number** | not measured yet — one command, §7, for when the box is free. |
| **Gates** | `mp/agent/tests` **337** (309 + 28), `mp/tournament/tests` **74** (57 + 17), `mp/replay/tests` + `mp/eval/tests` unchanged. |

---

## 1. What was built

| file | what |
|---|---|
| `mp/tournament/parallel.py` | `AgentDrive` / `drive_many` (the lockstep form of `_drive_to_next_nemesis`), the `TournamentDriver` protocol, `LocalDriver`, `ParallelTournament` |
| `mp/agent/parallel/layout.py` | byte layout of one leaf in a shared-memory arena (+ the measurements that chose it) |
| `mp/agent/parallel/channel.py` | arenas, the leaf queue, the reply pipes, the batching policy |
| `mp/agent/parallel/leaf.py` | `LeafEncoder`: `encode_leaf` with **no net** |
| `mp/agent/parallel/forward.py` | `forward_leaves`: the torch half, given already-encoded leaves |
| `mp/agent/parallel/remote.py` | `RemotePolicy`: a `PolicyValueFn` whose net is in another process |
| `mp/agent/parallel/evaluator.py` | `BatchEvaluator`: collect, batch per net, reply; `sync_weights` |
| `mp/agent/parallel/lockstep.py` | `LockstepDecider`: one `decide_many` for a whole slice of the population |
| `mp/agent/parallel/worker.py` | the worker process |
| `mp/agent/parallel/pool.py` | `WorkerPool`, `MPDriver`, `partition_agents` |
| `mp/agent/parallel/protocol.py` | the picklable messages |
| `mp/agent/train/parallel.py` | `ParallelMLBTrainer` (subclass of `MLBTrainer`, one method overridden) |
| `mp/agent/benchmarks/bench_parallel.py` | the throughput sweep — ONE command |
| `mp/agent/tests/test_parallel.py` (28) · `mp/tournament/tests/test_parallel_runner.py` (17) | |

Edited (additively, defaults unchanged): `train/population.py` (`instantiate(..., policy_for=)`),
`scripts/train_mlb.py` (`--workers`, `--evaluator-device`, `--evaluator-max-wait-ms`,
`--worker-arena-mb`, `--worker-threads`, and a `finally:` that drains the pool before the
final checkpoint).

---

## 2. Architecture, in five lines

1. **Main** holds the trainer (net, optimizer, buffer, RNG, checkpoints) and
   `ParallelTournament`, which keeps every cross-agent decision — the N×N matrix, the life
   rule, elimination, the ante barrier — in one process, in the order `Tournament.run`
   already does them.
2. **Each worker** owns a subset of the seats: their `BalatroGame`s, MCTS trees and tree
   caches, per-agent RNGs, W0's heuristic prior, the skip-cap filter, the sample collectors
   and the trajectory loggers. It holds **no net**.
3. **A worker drives its seats in lockstep** (`drive_many` + `LockstepDecider`): every one
   of its agents that needs an action descends to a leaf, all those leaves go to the net in
   one call, every tree backs up. At `leaf_batch=1` each tree's search is bit-identical to
   running it alone (BATCH_NOTES §3).
4. **The evaluator** is a daemon thread in the main process: it drains the leaf queue,
   groups the leaves by which net they belong to (live net + up to `p_history` past
   selves), runs one forward pass per group, writes the priors back and signals.
5. **Transport** is a shared-memory arena per worker for the payload and a `Queue`/`Pipe`
   pair for the 16 bytes of offsets and the round id.

```
main process                                    worker w (x N)
────────────────────────────────────────        ────────────────────────────────────
MLBTrainer  ── weights (sync_weights) ─┐        LocalDriver over ITS agents
ParallelTournament                     │        LockstepDecider -> BatchedSearch
  drive / apply / summarize ───────────┼──cmd──► RemotePolicy.evaluate_many
                                       │            encode_leaf (numpy, no torch)
BatchEvaluator (thread) ◄── leaf_q ────┴───────────  pack -> REQ[w] (shared memory)
  read REQ[*] as numpy views                         leaf_q.put(offsets)
  one forward per policy_id                          block on conn[w]
  write REP[w] ── conn[w] ──────────────────────►  read REP[w] -> priors, value
```

### Why the evaluator is a thread in the main process, not a process

* **The weights are already there.** A mirror module + one `load_state_dict` per generation
  (2.4M params, ~10 MB, ~8 ms) instead of shipping the net over a pipe. The brief's
  "broadcast to the evaluator after each training step" is `BatchEvaluator.sync_weights`,
  called once per generation after `_train` and before any worker plays
  (`test_sync_weights_is_what_the_workers_actually_see`).
* **Main is idle while a generation plays.** Play and train are strictly sequential in
  `MLBTrainer.run_generation`, so the core the evaluator thread uses is spare by
  construction. Torch releases the GIL for the forward pass and the transfers; the main
  thread is blocked in `Queue.get`.
* Moving it to its own process later is small: everything it touches (`EvaluatorChannel`,
  the arenas, the queue) already crosses process boundaries. Only the weight broadcast
  would have to be added.

---

## 3. Transport, with the numbers

Measured on the real run's configuration (set encoder, `ItemCaps(16,12,6,8,8)`, a
`SELECTING_HAND` leaf with 436 legal actions):

| | arrays | bytes |
|---|---|---|
| observation | 21 | 3 660 |
| action block (`act_num` 436×20, `act_sel` 436×16, `act_tgt` 436×34, `act_type` 436) | 4 | 122 952 |
| **per leaf, request** | **25** | **126 612** |
| per leaf, reply (436 priors + 1 value) | 2 | 1 748 |

A single process runs ~184 sims/s with W0's prior, so 16 workers is ~2 900 leaves/s =
**~370 MB/s of requests against ~5 MB/s of replies**. Through a `Queue` that is pickling
and unpickling 25 numpy arrays per leaf and copying every byte through a pipe, twice.
Through shared memory the worker writes the arrays straight into the buffer the evaluator
reads, and the only copy left on the evaluator side is the `np.stack` / padding
`BatchedSetNNPolicy` already does today. **So: shared memory for the payload, a `Queue` for
the offsets** (16 bytes per leaf — pickling that is free, and it gives a ready-made
multi-producer / single-consumer channel with a blocking `get(timeout=…)`).

Layout (`layout.py`): one 8-aligned record per leaf, `[float32 block][int16 block]`, keys
in `sorted()` order, every shape but the action count fixed by the encoder's caps — so a
leaf is described completely by `(offset, n_actions)`. The flat encoder's bare arrays are
handled as one-key dicts, so there is one implementation, not two.

**Windows / spawn specifics.** Everything crossing the boundary is a plain dataclass of
plain types; the queues and the `Connection`s are passed as `Process` arguments (the only
way `multiprocessing` can transfer them). The parent holds a handle on every arena for the
whole run, so a worker dying cannot take the memory with it. `torch.set_num_threads(1)` in
every worker before its first forward — 16 workers × 16 BLAS threads is the classic way to
make a 16-core box slower than one core.

### The batching policy

The evaluator **blocks only until the first submission of a round arrives, then drains
whatever else is already queued** and forwards that. No worker is ever waited for by name,
so a worker stuck in a long `game.step()` cannot hold anybody up; and because a forward
pass takes time, the queue refills while it runs, which makes the batch size self-balancing
under load. `--evaluator-max-wait-ms` adds a fixed extra drain budget after the first
arrival, trading latency for batch size.

**Measured (4 workers, N=8, sims 40, CPU):**

| max_wait | mean batch | evaluator forward | wall | sims/s |
|---|---|---|---|---|
| 0 ms (drain) | 1.64 | 14.6 s | 27.6 s | **531** |
| 2 ms | 2.44 | 9.8 s | 30.0 s | 482 |

On CPU the extra latency costs more than the bigger batch saves — **leave it at 0**. It is
a CUDA lever (a 3080 Ti at batch 1.3 is all launch overhead; BATCH_NOTES §6 measured the
forward pass amortising 26× between B=1 and B=64), which is why the sweep exists.

---

## 4. The determinism contract

**Structural (guaranteed, tested).** Given the same seed, the same population and the same
per-agent RNG seeds, *which agent does what, in what order* does not depend on the worker
count or on scheduling:

* the tournament seeds come from the trainer's generator in the main process;
* the seed **string** is resolved once in the main process and handed to every worker, so N
  workers building 16 games between them build the same 16 games one process would
  (`construct` and `clone` fan-out are pinned equal by `tournament/tests/test_fanout.py`);
* each agent's search RNG is `default_rng(member.seed)` — a function of the population, not
  of the partition;
* every cross-agent decision stays in `ParallelTournament`, in `Tournament.run`'s order;
* collected samples are re-ordered by `(seed, agent index, decision index)` before they
  reach the replay buffer.

**Numerical (contract: ~1e-7; measured: exact).** Batching may only change *when* a leaf is
evaluated, never *what* — the tolerance BATCH_NOTES §3 already accepts. Two tests:
`test_forward_batch_matches_single_leaf` (a leaf in a batch of 4 == the same leaf alone,
both encoders) and `test_one_worker_and_two_workers_play_the_same_tournament`, which plays a
real 4-agent tournament with a real net under 1 and 2 workers and asserts the matrices,
final lives, value targets and sample count are **identical**. A 1-vs-4-worker run of the
same shape was also checked by hand during the build: identical.

That equality is measured, not proved. If a future net or BLAS makes a batched row differ
from a single row at 1e-7, an argmax somewhere can flip and the two runs will diverge into
different (equally valid) games. The contract is the tolerance, not the bit-equality.

**`ParallelTournament` + `LocalDriver` == `Tournament`, byte for byte.** Serial equivalence
is pinned at the tournament level for all three life rules, both fan-out methods, odd
populations and the identical-population degenerate case, including the trajectory hooks
(`test_parallel_runner.py`), and at the agent level with a real MCTS population
(`test_lockstep_decider_is_byte_identical_to_the_serial_tournament`).

---

## 5. The one deliberate difference from the serial path

W1's `SampleBuilder` subsamples the action set with an RNG, and in the serial path that RNG
is **the trainer's single shared generator**. A worker cannot draw from it, so each agent
gets its own stream seeded from `(cfg.seed, generation, agent idx)` — which makes the
subsampling independent of the worker COUNT, something the shared generator never was.

Stated plainly: **a parallel generation is not bit-identical to the serial generation it
replaces.** It is the same experiment with a different (and more reproducible) noise
stream. Everything that defines the experiment — the population, the seeds, the search, the
value targets, the checkpoint — is unchanged. This is the only such difference, and it is
why §7's swap is described as a continuation rather than a seamless resume.

---

## 6. Failure, PAUSE, and the things that can go wrong

* **A worker dies** → its agents report `status="crashed"`, which `ParallelTournament`
  handles exactly like a death: they leave the population, the matrix is built from whoever
  is left, the generation still trains and checkpoints. `TournamentResult.crashed` and the
  generation log's `crashed_agents` / `dead_workers` / `dead_worker_reasons` say who and
  why. The worker is **not** restarted mid-generation (its games and trees are gone);
  `WorkerPool.respawn_dead()` brings it back at the start of the next one.
  (`test_a_dead_worker_costs_its_agents_not_the_run`, `test_respawn_…`.)
* **Every worker dies on the same command** → that is a setup bug, not a flake, and
  `pool.call` raises with the first traceback rather than quietly playing a tournament with
  nobody in it.
* **The evaluator thread dies** → the next `pool.call` raises; workers time out on their
  reply (`--worker-poll-seconds`, default 120 s) and exit cleanly rather than hanging.
* **PAUSE / SIGTERM** → unchanged contract: one tournament is the atomic unit, the
  generation trains on what it collected, the pool is drained, then the checkpoint is
  written and the resume command printed. Verified live: PAUSE 23 s into a 4-seed
  generation stopped after 2 tournaments, checkpointed, exited 0.
* **Trajectory logs**: each worker writes `trajectories.w<id>.jsonl` (N processes appending
  12 KB JSON lines to one handle is a corrupted line waiting to happen) and the generation
  end folds the parts into `trajectories.jsonl` and deletes them. `trajectories_merged` in
  the generation log is the count. Verified: 16 episodes, 16 lines, no parts left.

---

## 7. Throughput

### The benchmark — ONE command, run it when the box is free

```
python mp/agent/benchmarks/bench_parallel.py
```

That sweeps **workers ∈ {1, 4, 8, 12, 16} × evaluator ∈ {cpu, cuda}** plus the serial
baseline, one generation each, on the real run's Stage B configuration (N=16, 8 current +
4 anchors + 4 past selves, set encoder, sims 40, `sims_budgets 1.0,0.5,1.5`, leaf_batch 1,
skip cap 1, W0's prior at λ=0.4/τ=0.35/K=32), 1 seed per generation, `max_ante 8`. It
reports wall clock, sims/s total and per worker, leaf evaluations/s, the evaluator's mean
batch size and forward seconds, and mean worker wait — and writes
`benchmarks/bench_parallel_<date>.json`.

Useful variants:

```
# add the control arm: no shared evaluator, every worker runs the net on its own core
python mp/agent/benchmarks/bench_parallel.py --include-local

# is CUDA worth it with a bigger batch?
python mp/agent/benchmarks/bench_parallel.py --workers 8,16 --devices cuda --max-wait-ms 2
```

It does not touch `runs/real1/`; the net is cold unless `--init <a copy of a checkpoint>`
is given, and throughput is dominated by search cost rather than by what the weights say.

### What has been measured so far (with `real1` still holding a core)

**≤4 workers, N=8, sims 40, `max_ante 4`, 1 seed, set encoder, CPU evaluator:**

| arm | wall | sims/s | per worker | mean batch | speed-up |
|---|---|---|---|---|---|
| serial (`--workers 0`) | 55.7 s | 229 | 229 | — | 1.00× |
| 1 worker | 49.3 s | 268 | 268 | 3.24 | **1.17×** |
| 4 workers | 27.6 s | **531** | 133 | 1.64 | **2.32×** |

Three things to read off it.

1. **Even one worker beats serial (1.17×).** The worker drives its 8 agents in lockstep, so
   its leaves batch (mean 3.24) where the serial runner — which drives one agent to its
   Nemesis before starting the next — never has two leaves to batch at all.
2. **4 workers is 2.32×, not 4×.** The ante is a barrier, `worker_wait` was 7.2 s of 27.6 s
   (26% of worker time waiting on the evaluator), and this ran against a live training
   process on the same box. With 8 agents over 4 workers each worker owns 2 seats, so
   there is very little to batch inside a worker either — which is why the mean batch
   *falls* from 3.24 to 1.64 as workers go up.
3. **That last point is the thing to watch at 16 workers.** With N=16 agents and 16
   workers each worker owns one seat, so all the batching has to come from the evaluator
   drain, and the evaluator is one thread. The `--include-local` arm is there precisely to
   answer "is the shared evaluator earning its transport on this box", and I would not be
   surprised if it wins on CPU. **The benchmark decides this, not this document.**

CUDA was sanity-checked only (2 workers, N=6: it runs, 248 sims/s, mean batch 1.31, and
9.9 s of the 17 s wall was inside the forward pass — a 3080 Ti at batch 1.3 is all launch
overhead, exactly as BATCH_NOTES §6 predicted). Whether CUDA wins at 12-16 workers is the
open question the sweep answers.

---

## 8. The swap procedure for the live `real1` run

The checkpoint format is unchanged and `--workers` is **not** recorded in it, so the swap is
a PAUSE and a resume with an extra flag.

```bash
# 1. stop the single-process run at the end of the tournament in flight
touch mp/agent/runs/real1/PAUSE
#    wait for "=== Stopped (PAUSE) ..." in mp/agent/runs/real1.console.log

# 2. resume it on N cores.  Everything else comes back out of the checkpoint;
#    --workers / --evaluator-device are the only additions.
python mp/agent/scripts/train_mlb.py \
    --resume mp/agent/runs/real1/latest.pt \
    --minutes 2880 --device cpu \
    --workers 12 --evaluator-device cpu \
    --run-dir mp/agent/runs
```

Notes on that command.

* `--resume` deletes the PAUSE file itself.
* Pick `--workers` from the benchmark. Until it has been run, **12** is the conservative
  choice on a 16C/32T box: 12 workers + the evaluator thread + the trainer, leaving cores
  for the machine. `--evaluator-device` should be whatever the sweep says (`cpu` unless
  CUDA wins at that worker count; `local` if the control arm wins, in which case the
  workers hold the nets and there is no shared evaluator at all).
  **A prediction worth checking against the sweep:** with N=16 seats, `--workers 8` gives
  each worker two seats and `--workers 16` gives it one, and §7's smoke says the batch
  mostly comes from *inside* a worker — so 8 or 12 may well beat 16 despite the extra
  cores. More than 16 is pointless: there are only 16 seats.
* Everything the run was doing carries over from the checkpoint: encoder, sims, budgets,
  the skip cap and its anneal, W0's λ and its clear-rate EMA, the opponent history, the
  buffer, the optimizer moments, the generation counter. Only pass flags you want to
  *change*.
* Trajectory logging: add `--log-trajectories --sig-every 50` again if you want it (it is a
  run-shaping flag, not checkpoint state). Workers write parts; they are merged per
  generation.
* To go back: PAUSE, resume without `--workers`. Verified in both directions.
* **It is a continuation, not a bit-exact resume** — see §5.

---

## 9. Found, not fixed

* **The evaluator is one thread, and at 16 workers it may be the bottleneck.** It does
  numpy stacking (GIL-bound) around a forward pass (not). The measured `eval_forward_s`
  and `worker_wait_s_mean` in each generation's log are the diagnostic. Fixes, in order of
  effort: raise `--evaluator-max-wait-ms` (CUDA only — it *lost* on CPU, §3), move the
  evaluator to its own process (small; everything it touches already crosses a boundary),
  or run two evaluator threads partitioned by `policy_id`.
* **Past-self seats fragment the batch.** With `p_history=4` the four opponent seats
  usually hold four *different* checkpoints, so they are four forward passes of one leaf
  each per round, next to one pass of eight for the live net. Batching by `policy_id` is
  correct and unavoidable; what would help is `p_history` seats sharing fewer distinct
  checkpoints (a population design decision, not a plumbing one).
* **The 436-actions-per-leaf cost is still the real cost**, and it is now *also* the
  transport cost (127 KB per leaf is 122 KB of action features). W0 measured that the
  candidate mask is applied at `_apply_expansion`, *after* `featurize_actions_set` has
  built all 436 rows; pushing the allowed set into `encode_leaf` would cut the per-leaf
  Python, the arena traffic and the padded forward all at once. It is a `PolicyValueFn`
  contract change and it is NOT done — it remains the biggest single lever, exactly as
  PRIOR_NOTES §6 and BATCH_NOTES §6 both say.
* **A crashed worker's agents are lost for the whole generation, not just the tournament.**
  `respawn_dead` runs at the start of the next generation. Restarting mid-generation would
  need the games re-created from the seed and re-driven, which is a different experiment.
* **`--objective external` is not parallelised.** It is one solo game per episode; the pool
  would only add overhead. `ParallelMLBTrainer` falls through to the inherited path.
* **`partition_agents` balances by `sims`, which is a proxy.** A seat that dies at ante 2
  costs nothing after that, and the ante barrier means the slowest worker sets the pace.
  Work stealing would fix it; it needs the games to be movable, which they are not.
* **`max_wait_ms` and `poll_seconds` are not adaptive.** Both are flags with measured
  defaults, not controllers.
* **No test asserts 1-vs-4 workers on the FULL N=16 population** — the pool tests use 1-2
  workers and 3-4 agents to keep the suite under a minute on Windows spawn. The 1-vs-4
  check at N=8 was done by hand during the build and was exact.

---

## 10. Gates

| gate | result |
|---|---|
| `python -m pytest mp/agent/tests -q` | **337 passed** (309 at start + 28) |
| `python -m pytest mp/tournament/tests -q` | **74 passed** (57 at start + 17) |
| `python -m pytest mp/replay/tests mp/eval/tests -q` | unchanged (82 + 125) |
| `train_mlb.py --workers 2` end to end | 1 generation, checkpoint, trajectories merged ✅ |
| PAUSE mid-generation with workers | drained after the tournament in flight, checkpointed, exit 0 ✅ |
| parallel checkpoint → single-process `--resume` | generation 1 → 2, continued ✅ |
| single-process checkpoint → parallel `--resume` | pinned by `test_a_single_process_checkpoint_resumes_into_the_parallel_trainer` ✅ |
