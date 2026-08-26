# BATCH_NOTES — Phase 3 W3: batched NN inference + tree reuse

**Agent W3, 2026-08-22.** Deliverable: batched leaf evaluation across concurrent trees,
tree reuse between decisions, benchmarks, tests, and the plug-in the tournament needs.
Everything is under `agent/`. `engine/**` and `rng/**` are untouched — verified
against `docs/phase3_frozen_snapshot.txt`: **41/41 frozen files identical** in size, mtime
within the snapshot's 1-second rounding.

---

## 0. Headline

500 simulations per decision, RTX 3080 Ti box, Gumbel selection, vanilla ante-1 blind with
436 legal actions (the state W1 benchmarked). All numbers re-measured today; see §5 for the
full tables and the +/-20% noise band this box has.

| | serial (this box, re-run) | after W3 |
|---|---|---|
| throughput, one tree | cpu **563** / cuda **531** sims/s | cuda **1034** (K=1, `leaf_batch=16`) |
| throughput, many trees | cpu 563 / cuda 531 (one agent at a time) | cuda **761** at K=32, *bit-identical* to K independent searches |
| leaf evaluation's share of the search | 41% cpu / 55% cuda | **20-28%** |
| forward-pass cost per leaf | 1.25 ms (B=1) | **0.048 ms** (B=64) — a 26x amortisation |
| at the MLB Nemesis | cuda 427 sims/s, **78%** in leaf eval | cuda 443 at K=32 (**69%**) / 569 at L=16 |
| decisions informed by the previous search | none | **91-97%** (80% with an MLB opponent-score relay) |
| decisions per second with reuse, fixed evidence | 3.9-8.9 | **6.2-11.6** (1.3-1.7x) |

Two things changed the shape of the problem versus the brief's expectations, and both are
measured, not assumed:

1. **The GPU was never the bottleneck; the per-leaf CPU work is.** Batching does exactly
   what it was supposed to — the forward pass drops from 1.25 ms per leaf at B=1 to
   0.048 ms at B=64, a 26x amortisation — but a leaf also costs ~0.5 ms of numpy/Python
   *before* the net sees it (featurizing 436 actions, building 436 priors, allocating 436
   child nodes) and the search costs ~0.8 ms per simulation in `clone()` + `step()`. Those
   do not batch. §6 has the full per-leaf decomposition.
2. **Tree reuse is nearly always valid on this engine, not "generally invalid".** The
   brief expected chance-node invalidation ("a `play` that triggers a random effect ...
   will generally NOT match"). Phase 1 deleted `game.rng` and moved every draw onto the
   keyed `PseudoRandom` whose position table is part of the cloned state, so
   `clone().step(a)` is a *deterministic function of (state, action)*. The signature check
   still earns its place — it is what catches the MLB driver mutating the game between
   decisions — but in vanilla single player it passes on every decision after the first.

---

## 1. What was built

| file | what |
|---|---|
| `mcts/batched.py` | **new** — `BatchedNNPolicy` (one forward pass for B leaves), `BatchedSearch` (K trees in lockstep), `SearchRequest` / `SearchResult` / `BatchStats` |
| `mcts/reuse.py` | **new** — `TreeCache` (signature-guarded subtree retention), `ReuseConfig`, `ReuseStats`, `count_nodes` |
| `mcts/search.py` | every search is now also a **generator** (`run_iter` / `run_gumbel_iter` / `_simulate_iter` / `_expand_iter`) that yields leaves and is sent evaluations; `run` / `run_gumbel` drive it serially and are **byte-identical** to the pre-W3 implementation; `root=` / `sims=` for reuse; `MCTSConfig.leaf_batch` for virtual-loss leaf batching |
| `mcts/player.py` | `MCTSPlayer` gained `reuse`, `leaf_batch`, `no_action`, `reset()`; **new** `BatchedMCTSPlayerGroup.act_many`, `load_policy`, `make_player` |
| `mcts/action_features.py` | `featurize_actions` rewritten as one fancy-indexed block instead of `np.stack` over N arrays — byte-identical output, 0.46 ms -> 0.32 ms per leaf (§6) |
| `benchmarks/bench_batched.py` | **new** — the tables in §5 |
| `tests/test_batched.py` | **new** — 29 tests |
| `tests/test_reuse.py` | **new** — 17 tests |

---

## 2. Design: one leaf-evaluation seam, three ways to fill a batch

W1 left exactly one place where a leaf is evaluated (`MCTS._evaluate_leaf`). The problem
with batching against that seam is that a search is a *loop*, and a loop cannot hand out
several leaves at once. So the search was turned inside out: `run_iter` and
`run_gumbel_iter` are generators that `yield` a list of leaf games and are `send()`-ed the
matching list of `(priors, value)`.

```python
def _drive(self, gen):                      # the serial driver, in search.py
    try:
        request = next(gen)
        while True:
            request = gen.send([self._evaluate_leaf(g) for g in request])
    except StopIteration as stop:
        return stop.value

def run(self, root_game, add_noise=True, root=None, sims=None):
    return self._drive(self.run_iter(root_game, add_noise, root, sims))
```

That is the whole trick. Consequences:

* **`MCTS.run` / `run_gumbel` are unchanged.** Same order of clones, steps, RNG draws and
  backups. `tests/test_batched.py::test_serial_search_matches_the_pre_w3_implementation`
  keeps a verbatim copy of W1's loops in the test file and compares the **entire tree**
  (visit counts, value sums, priors, stop reasons) plus the Gumbel pick, for PUCT and
  Gumbel at two budgets. Identical, not close.
* `_evaluate_leaf` survives as the serial seam, so `bench_search.py`'s instrumentation
  still works.
* A batched driver is then trivial: advance every tree's generator to its next request,
  concatenate, call `evaluate_many` once, hand each tree its slice back.

Three independent ways to fill a batch, and they compose:

| source | approximation? | who can use it |
|---|---|---|
| **K trees in lockstep** (`BatchedSearch`) | none — each tree is bit-identical to running alone | the tournament (N agents), self-play (K games), the eval harness |
| **L leaves in flight within one tree** (`MCTSConfig.leaf_batch`, virtual loss) | yes — later descents in a batch see stale statistics | a single agent deciding alone |
| ragged actions inside one leaf (`score_actions_flat`) | none | already there, W1 built it |

### 2.1 Why within-tree batching was built even though the brief made it optional

The brief said to do it "only if K=1 matters for the tournament player". It does:
`tournament/runner.py::_drive_to_next_nemesis` drives **one agent all the way to its
Nemesis before touching the next one**, so as the runner is written today there is never
more than one tree wanting a leaf. Cross-tree batching needs a runner change (§7); the
within-tree path needs nothing but `leaf_batch=16` on the player, and it is the fastest
single number in the table.

### 2.2 Virtual loss

A simulation that has descended but not yet been evaluated increments `N` on every node of
its path immediately and contributes `0.0` to `W` until its real value arrives. Concurrent
descents therefore see the path as "visited, worth nothing" and go elsewhere. At backup the
value is added and no second visit is counted, so **final N and W are exactly what the
serial search would have produced for the same set of leaves** — the visit is counted early,
not twice (`test_virtual_loss_keeps_the_visit_bookkeeping_exact`).

Two in-flight simulations can still land on the same unexpanded node (identical states,
identical evaluations); `_apply_expansion` is idempotent so the second one backs up the
value without re-creating children and discarding the first one's statistics.

### 2.3 Early finish

Trees leave the pool the round they finish: a search that spends its budget, a root with no
legal actions (MLB `PVP_WAIT`, readied at a Nemesis), a terminal root, or a shorter
per-agent budget. The batch simply gets smaller — nothing waits for anything.
`test_trees_finishing_early_do_not_stall_the_batch` mixes budgets 3 / none / 30 / 1 with a
`PVP_WAIT` root and asserts every tree returns and the batch sizes decrease monotonically.

---

## 3. `BatchedNNPolicy`

`evaluate_many(games) -> [(priors, value), ...]`, order preserved. The CPU half is
`NNPolicy.encode_leaf` **verbatim** (that is what makes "batched == single-leaf" a
meaningful test); only the torch half differs:

1. `(B, obs_dim)` through the trunk, one `value_head` call.
2. The ragged action features concatenated into `(sum(N_i), 56)` and scored by
   `PolicyValueNet.score_actions_flat` — one policy-head call for every action of every
   state (W1 built this and pinned it equal to the per-state path).
3. **Segmented softmax** in one kernel chain: scatter the flat logits into a
   `(B, max_N)` block padded with `-inf`, `torch.softmax` the block, gather the live
   entries. `torch.split` + B softmax calls would be correct but costs B kernel launches,
   which is the overhead this module exists to avoid.
4. Leaves with **no legal actions** are dropped from the batch entirely and returned as
   `({}, 0.0)` — they never reach the net (`test_batched_policy_skips_no_action_games`).
5. Batches larger than `max_action_rows` (default 250 000 action rows) are split into
   several forward passes and stitched back together
   (`test_batched_policy_chunking_changes_nothing`).

### Is batched == single-leaf *exactly*?

**No, and it cannot be** — the brief's "on CPU exact" does not hold. A B-state trunk is a
`(B, 447) @ (447, 512)` matmul; a single leaf is `(447,) @ (447, 512)`. BLAS blocks the
reduction differently, so the results differ in the last ulps. Measured on this box (CPU,
436-action leaves): **max prior difference 2.3e-10, max value difference 8.3e-7**. The
tests assert `< 1e-6` on priors and `< 1e-4` on values.

That difference is far too small to change a PUCT argmax except at an exact tie, and in
practice it changes nothing: a K-tree batched search and K single-tree searches produced
**identical visit counts and identical Gumbel picks** in every configuration tried (K = 3
and 8, 100 and 300 simulations, PUCT and Gumbel). The test suite still asserts exact
equality only for `UniformPolicy` (where it is guaranteed) and a total-variation bound of
0.01 for the net, so a future BLAS or driver change cannot make the suite flaky.

---

## 4. Tree reuse

### 4.1 The rule

Keep the chosen child's subtree as the next root **iff `game.state_signature()` on the
live game equals the signature of the state that subtree was built from**. No partial
credit, no salvage. The signature is the engine's own "two games with equal signatures
produce identical futures" hook (`engine/balatro_sim/game.py:894`), and it includes a
hash of the keyed RNG's entire position table.

The expected signature is computed once per decision, post hoc: clone the game the search
ran from, apply the chosen action, take the signature (~150 us). That is exactly
equivalent to recording it inside the search — because of the determinism in §4.2 — and it
costs 0.2 ms per decision instead of 42 us x sims (10-20% of a batched search).

### 4.2 What actually invalidates a tree (this is not what the brief expected)

`clone().step(a)` is a **deterministic function of (state, action)** on the post-Phase-1
engine: stepping the same `play` (which draws replacement cards) or the same shop `reroll`
(which regenerates the shop) from the same state a hundred times gives one signature every
time. `tests/test_reuse.py::test_engine_is_deterministic_under_clone_step` pins it, and it
is the assumption the whole module rests on — if it ever fails, that test fails first.

So the invalidation cases are all **driver** behaviour, not chance:

* MLB match plumbing between decisions — `set_pvp_info()` (the opponent's live score
  arriving), `end_pvp()`, `lose_life()`, the comeback bonus. `MLBMatch` and the tournament
  runner all do this.
* more than one engine step between decisions (the runner's `_cash_out`, a `PVP_WAIT`
  resolution).
* a driver that applies an action other than the one the player returned.
* a chosen edge that was never simulated, or one that ends the run.

Measured: **91% hit rate in vanilla SP, 97% in MLB solo, 80% at a Nemesis with a live
opponent-score relay** (§5 table D). The relay case is exactly the discard the check
exists for.

### 4.3 The three decisions the brief asked me to document

**Dirichlet noise on a reused root: re-applied, once.** Noise is only ever mixed into the
*root's own children*. The node that becomes the next root is one of those children, and
*its* children were created at its own expansion carrying bare policy priors — untouched
by the previous decision's noise. So re-noising cannot compound, and not doing it would
leave every decision after the first with no root exploration at all.
`test_reused_root_is_re_noised_but_not_compounded` pins both halves.

**Gumbel on a reused root.** Three separate questions:

* *The sampled top-k is redrawn every decision.* A reused root does not lock in the
  previous decision's candidate set. This is deliberate: the Gumbel-Top-k trick is an
  unbiased sample of the *current* prior, and the priors are what they are regardless of
  how many visits the node carries.
* *sigma's scale uses only this decision's visits.* `sigma(q) = (c_visit + N_max) * q`,
  and `N_max` is now computed over `visit_count - baseline`, where `baseline` is each
  child's visit count at the moment the search started. Without that, a root carrying 400
  retained visits multiplies every `q` by ~450 and the sampled Gumbel noise stops
  mattering entirely — sequential halving would degenerate into greedy-by-Q.
  `test_gumbel_sigma_uses_only_this_decisions_visits` checks the exact arithmetic.
* *`q_hat` uses all visits, retained ones included.* That is the entire point of reuse:
  the retained subtree's value estimates are better than nothing, and they are what the
  new decision inherits.

**Budget.** `ReuseConfig.budget_mode`:

* `"subtract"` (default) — a decision costs `num_simulations` in total, retained visits
  included. Reuse buys **wall clock at a fixed evidence level**. This is the mode the
  equivalence test pins.
* `"add"` — always run the full budget of new simulations. Reuse buys **evidence at a
  fixed wall clock**; `effective_sims` in table D is that view.
* `min_new_sims` floors "subtract" so a deep retained tree cannot reduce a decision to
  zero new simulations.

### 4.4 "Same visit distribution as a fresh search"

The exact statement that is true, and is tested
(`test_retained_plus_new_equals_one_search`): **reuse is resumption**. A budget of N spent
as (M retained + N-M new) gives *exactly* the tree that one N-simulation search gives,
provided the search is PUCT with the RNG untouched between the halves.

The stronger-sounding claim — "a search resumed from a subtree built under the *parent*
equals a fresh search from the child state" — is false in general and would be wrong to
test for: the retained subtree was built by simulations that descended through the parent's
PUCT and were valued from the parent's perspective, so it is not the prefix of any fresh
search. Reuse is a *warm start*, not a *replay*. Gumbel deliberately breaks resumption
equivalence too, by redrawing its root noise (§4.3).

### 4.5 Forced single-action states carry the tree

`ROUND_EVAL` -> `advance` sits between every blind and every shop, and `MCTSPlayer`
shortcuts single-action states without searching. The shortcut still walks the retained
tree down one level, otherwise the tree would die at every cash-out
(`test_forced_singleton_states_carry_the_tree`).

---

## 5. Benchmarks

All on the RTX 3080 Ti box, Python 3.13.5, torch 2.6.0+cu124, the same demo state as
W1's baseline (ante-1 Small blind, 3 jokers, **436 legal actions**), **500 simulations per
decision**, Gumbel root selection. Throughput is best-of-N uninstrumented runs; the
NN/sim/other split comes from a separate instrumented run normalised to its own total
(bench_search.py's protocol, for the same reason).

> **Read the noise before the numbers.** Repeated measurements of the *same* configuration
> on this box vary by up to **+/-20%** (CPU frequency scaling; an interleaved A/B of a
> change worth 6% needed 6 alternations to see it). Every claim below that matters is
> therefore backed by a component measurement (§6) as well as an end-to-end one.
> Rows within one table were measured in one process, in the order shown.

### A. Single tree, serial — the baseline, re-run (`--repeat 2`)

| state | policy | W1 (2026-08-21) | W3 re-run | in NN | in clone+step | other |
|---|---|---|---|---|---|---|
| vanilla blind | nn-cpu gumbel | 475 | **563** | 41.0% | 49.3% | 9.7% |
| vanilla blind | nn-cuda gumbel | 328 | **531** | 54.7% | 38.2% | 7.1% |
| MLB Nemesis | nn-cpu gumbel | 308-313 | **404** | 73.4% | 10.0% | 16.6% |
| MLB Nemesis | nn-cuda gumbel | — | **427** | 77.5% | 8.8% | 13.7% |

W1's headline "single-leaf CUDA is SLOWER than CPU" **no longer reproduces**: 531 vs 563 in
a vanilla blind is inside the noise, and at a Nemesis CUDA is now ahead. Two things moved:
this benchmark does an explicit warm-up search before timing (a cold CUDA context costs
~0.2 s of the first search), and `featurize_actions` got 30% faster (§6). The rest of
W1's diagnosis stands, and is the important half: **the leaf-evaluation bucket is 41-78% of
a search**, and at the MLB Nemesis — the state the tournament actually cares about — it is
**78%**, because a PvP round never ends on `chips >= target` so every descent stays in the
436-action `SELECTING_HAND` space (measured: **mean 397 legal actions per leaf** over a
300-simulation Nemesis search, median 436).

### B. Batched, vanilla blind — K trees in lockstep (`--repeat 2`)

| device | K | L | sims/s total | sims/s per tree | ms per decision | NN | clone+step | other | mean batch |
|---|---|---|---|---|---|---|---|---|---|
| cuda | 1 | 1 | 445 | 445 | 1123 | 58.5% | 34.7% | 6.7% | 1.0 |
| cuda | 8 | 1 | 682 | 85 | 733 | 37.4% | 52.7% | 9.8% | 8.0 |
| cuda | **32** | 1 | **761** | 24 | 657 | 28.0% | 58.7% | 13.3% | 32.0 |
| cuda | 100 | 1 | 729 | 7 | 686 | 25.3% | 59.6% | 15.1% | 99.9 |
| cpu | 1 | 1 | 532 | 532 | 940 | 39.7% | 49.9% | 10.3% | 1.0 |
| cpu | 8 | 1 | 688 | 86 | 726 | 25.4% | 61.9% | 12.7% | 8.0 |
| cpu | 32 | 1 | 667 | 21 | 750 | 19.7% | 66.5% | 13.8% | 32.0 |
| cpu | 100 | 1 | 654 | 7 | 764 | 23.1% | 62.1% | 14.9% | 99.9 |

**Batching does exactly what it was meant to do to the NN bucket — 58.5% -> 25.3% of the
search on CUDA — and total throughput still only moves 1.4x** (531 -> 761 sims/s), because
what it uncovers underneath is `clone()` + `step()`, which goes from 35% to 60% of the wall
and does not batch. K saturates at 32; K=100 is slightly *worse* (bigger batches, same
per-leaf CPU cost, more Python bookkeeping). CPU peaks at K=8 and then declines.

### B2. Batched, vanilla blind — L leaves in flight within ONE tree (`--repeat 3`)

| device | K | L | sims/s total | ms per decision | NN | clone+step | other | mean batch |
|---|---|---|---|---|---|---|---|---|
| cuda | 1 | 1 | 387 | 1291 | 61.1% | 32.6% | 6.3% | 1.0 |
| cuda | 1 | 4 | 680 | 736 | 49.0% | 43.3% | 7.7% | 4.0 |
| cuda | 1 | **16** | **1034** | 484 | 47.5% | 43.7% | 8.8% | 14.7 |
| cuda | 8 | 4 | 866 | 577 | 26.3% | 59.1% | 14.7% | 31.8 |
| cuda | 8 | 16 | 1006 | 497 | 37.3% | 50.6% | 12.1% | 117.9 |
| cpu | 1 | 16 | 868 | 576 | 40.1% | 48.5% | 11.4% | 14.7 |
| cpu | 8 | 16 | 846 | 591 | 39.5% | 49.7% | 10.8% | 117.9 |

**L=16 on ONE tree is the fastest configuration measured** — and roughly a third of that is
not batching at all. Instrumented directly (500 simulations, CUDA, same seed):

| L | engine `step()` calls | steps/sim | forward passes | NN seconds | wall |
|---|---|---|---|---|---|
| 1 | 4 684 | 9.37 | 500 | 0.83 | 1.31 s |
| 4 | 3 703 | 7.41 | 126 | 0.32 | 0.71 s |
| 16 | 2 287 | **4.57** | **34** | 0.22 | **0.53 s** |

Virtual loss pushes concurrent descents apart, so the tree grows **wider and shallower**:
half the engine steps per simulation at L=16. Of the 0.78 s saved, ~0.61 s is fewer/larger
forward passes and ~0.24 s is the shallower descent. The second half is a *change in the
search*, not free throughput — a shallower search is not the same search — which is why
L>1 is opt-in and K (which is exact) is the recommended lever whenever more than one tree
is available.

### C. Batched, MLB Nemesis (`--repeat 2`, `--ruleset mlb --nemesis --encoder mlb`)

| device | K | L | sims/s total | ms per decision | NN | clone+step | other |
|---|---|---|---|---|---|---|---|
| cuda | 1 | 1 (serial) | 427 | 1171 | 77.5% | 8.8% | 13.7% |
| cuda | 8 | 1 | 453 | 1104 | 70.5% | 11.5% | 18.0% |
| cuda | 32 | 1 | 443 | 1128 | 69.0% | 12.1% | 19.0% |
| cuda | 1 | 16 | **569** | 878 | 76.0% | 8.0% | 16.0% |
| cpu | 8 | 1 | 415 | 1206 | 60.7% | 13.5% | 25.8% |
| cpu | 32 | 1 | 366 | 1368 | 62.3% | 12.1% | 25.6% |

**This is the uncomfortable result, and the most important one.** At the Nemesis — where W1
measured 78% of the search inside leaf evaluation, and where batching should therefore pay
best — cross-tree batching buys almost nothing (427 -> 443 sims/s) and the NN bucket barely
moves (77.5% -> 69.0%). The reason is §6: at a Nemesis nearly every leaf carries ~400
legal actions, so the leaf-evaluation bucket is dominated by **per-leaf CPU work that
batching cannot touch** (featurizing 400 actions, building 400 priors), not by the GPU call
it can. The GPU part of a leaf really does amortise 26x; it just is not the expensive part
any more.

### D. Tree reuse (`--reuse-decisions 30 --reuse-sims 100`, CUDA)

`hit%` = decisions whose retained tree was valid; `retain%` = mean fraction of the previous
decision's tree (by node count) that survived; `ret.N` = retained visits per decision;
`eff.sims` = simulations informing a searched decision, retained ones included;
`evid/s` = (new + retained) simulations per second of wall clock.

| scenario | mode | hit% | retain% | ret.N | decisions/s | eff.sims | evid/s |
|---|---|---|---|---|---|---|---|
| vanilla SP | off | 0.0% | 0.0% | 0.0 | 8.88 | 100.0 | 807 |
| vanilla SP | subtract | **90.9%** | 23.5% | 28.5 | **11.58** | 100.0 | 1104 |
| vanilla SP | add | 90.9% | 21.1% | 39.5 | 10.12 | **136.1** | 1320 |
| MLB solo | off | 0.0% | 0.0% | 0.0 | 3.92 | 100.0 | 353 |
| MLB solo | subtract | **96.7%** | 37.4% | 36.7 | **6.71** | 100.0 | 621 |
| MLB solo | add | 96.7% | 35.0% | 53.4 | 4.00 | **152.7** | 574 |
| MLB Nemesis | off | 0.0% | 0.0% | 0.0 | 5.24 | 100.0 | 506 |
| MLB Nemesis | subtract | 96.7% | 25.3% | 27.0 | 7.15 | 100.0 | 695 |
| MLB Nemesis | add | 96.7% | 27.1% | 39.9 | 4.52 | 140.1 | 617 |
| Nemesis + relay | off | 0.0% | 0.0% | 0.0 | 4.56 | 100.0 | 441 |
| Nemesis + relay | subtract | **80.0%** | 19.1% | 22.0 | 6.24 | 100.0 | 603 |
| Nemesis + relay | add | 80.0% | 22.1% | 34.2 | 5.28 | 135.4 | 691 |

* **Retention is 91-97% of decisions in every scenario without a driver mutation**, and the
  only misses there are the first decision of an episode (nothing cached yet).
* **The relay scenario is the discard case working**: a driver calling `set_pvp_info()`
  between decisions (exactly what `MLBMatch` and a relaying tournament runner do at a
  Nemesis) drops the hit rate to 80%. The tree is dropped, not silently reused from a state
  that no longer exists.
* **What it buys**: at a fixed evidence level (`subtract`), 1.3-1.7x more decisions per
  second — a bigger end-to-end win than cross-tree batching. At a fixed wall clock (`add`),
  1.36-1.53x more simulations informing each decision.
* Retained *nodes* are only ~20-37% of the previous tree even though the retained *root* is
  the chosen child: Gumbel spreads its budget over up to 16 root children and only one of
  those subtrees is kept.


---

## 6. The remaining bottleneck

Per-leaf costs, measured directly on the 436-action demo state (best of 30-40 repeats):

| what | where | per leaf | batches? |
|---|---|---|---|
| `game.legal_actions()` | engine (frozen) | 0.048 ms | no |
| `encode_obs` (447 / 453 dims) | `mcts/encoder.py` | 0.042 ms | no |
| **`featurize_actions` (436 x 56)** | `mcts/action_features.py` | **0.32 ms** (was 0.46) | no |
| `priors_from_logits` (436 `action_key`s + dict) | `mcts/policy.py` | 0.12 ms | no |
| `Node.add_child` x 436 | `mcts/node.py` | 0.20 ms | no |
| **forward pass, B=1** | torch / CUDA | **1.25 ms** | — |
| forward pass, B=8 | torch / CUDA | 0.52 ms/leaf | yes |
| **forward pass, B=64** | torch / CUDA | **0.048 ms/leaf** | yes |
| `clone()` | engine (frozen) | 0.069 ms per **sim** | no |
| `step()` x 9.4 per sim (vanilla) | engine (frozen) | ~0.73 ms per **sim** | no |

Read down that column: **the GPU call amortises 26x with batching and then stops being the
problem.** What is left per simulation is ~0.73 ms of un-batchable Python/numpy around the
leaf plus ~0.8 ms of engine `clone()`/`step()` — a floor of ~1.5 ms/sim, i.e. **~650 sims/s
per process**, which is exactly where every batched configuration in table B lands.

Ranked list of what to attack next, with the evidence:

1. **436 actions per leaf is the real cost.** Three separate per-leaf costs
   (`featurize_actions` 0.32 ms, `priors_from_logits` 0.12 ms, 436 `Node` allocations
   0.20 ms) are all linear in the size of the legal-action set, and at a Nemesis *every*
   leaf is that big (mean 397). The fix is to stop expanding all of them: keep the top-k
   actions by prior plus a random sample of the rest — exactly the subsampling W1 already
   recommended for the 97 KB `Sample` problem (AGENT_NOTES §3). One change fixes the
   training-sample size, the tree memory (a 500-simulation vanilla tree is 26 304 nodes and
   ~15 MB; a Nemesis tree is 119 494 nodes) and the leaf cost together. **This is the single
   highest-value change left in the agent layer, and it is a training-design decision, not
   a plumbing one — it belongs to whoever owns the objective.**
2. **`clone()` + `step()`, ~0.8 ms/sim, is 50-66% of a batched vanilla search.** The engine
   is frozen in Phase 3, so nothing here touches it; a `clone()` is 69 us and a `step()`
   that plays a hand does full joker scoring. Nothing in the agent layer can improve it.
3. **Process-level parallelism, not more batching.** K saturates at 32 and the remaining
   cost is single-threaded Python; N worker processes each running a batched scheduler
   would scale nearly linearly where a bigger K does not. That is the natural next
   infrastructure step and it is outside W3's scope.


---

## 7. What the tournament should do

### 7.1 What K to use

| situation | setting | why |
|---|---|---|
| tournament, runner unchanged | `leaf_batch=16`, `reuse=True` | the runner drives one agent at a time, so K is 1 whatever the player does; L=16 is the fastest single-tree configuration (table B2) and reuse adds 1.3-1.7x on top |
| tournament, runner switched to `act_many` | `K = min(n_agents, 32)`, `leaf_batch=1`, `reuse=True` | exact (bit-identical to independent searches), saturates at 32 (table B) |
| self-play / training | `K = 8-32` independent games, `leaf_batch=1` | same, and the games are genuinely independent |
| a single eval game | `leaf_batch=16` | there is nothing else to batch with |

Do **not** use K=100: it is slower than K=32 on both devices and costs ~1.5 GB of tree
memory (15 MB per 500-simulation vanilla tree; a Nemesis tree is ~5x that).

### 7.2 The `tournament/players.py` diff (for the lead — W3 did not edit that file)

Replace the `MCTSPlayer` placeholder with a factory. `agent` goes on `sys.path` the same
way `tournament/bootstrap.py` already does it for `engine`; the import is
function-local so `tournament` still imports without torch installed.

```diff
--- a/tournament/players.py
+++ b/tournament/players.py
@@
-class MCTSPlayer:
-    """Placeholder.  W1 (agent fork, ``agent/mcts``) and W3 (batched inference / tree
-    reuse) are concurrent workstreams building the search agent; ..."""
-
-    def __init__(self, *args, **kwargs):
-        raise NotImplementedError("W1/W3 plug in here")
-
-    def act(self, game) -> dict:
-        raise NotImplementedError("W1/W3 plug in here")
-
-    def reset(self) -> None:
-        raise NotImplementedError("W1/W3 plug in here")
+def MCTSPlayer(checkpoint=None, sims=100, device="cpu", seed=0, strategy="gumbel",
+               reuse=True, leaf_batch=16, **kwargs):
+    """The agent-layer MCTS player (`agent/mcts/player.py`).  `checkpoint=None` gives
+    cold-start weights.  Returns a `Player`: `act(game) -> dict` (never None -- it returns
+    ``{"type": "advance"}`` on a no-action state, like the other adapters here) + `reset()`."""
+    import sys
+    from pathlib import Path
+    agent_root = str(Path(__file__).resolve().parents[1] / "agent")
+    if agent_root not in sys.path:
+        sys.path.insert(0, agent_root)
+    from mcts import make_player                    # noqa: E402  (lazy: needs torch)
+    return make_player(checkpoint=checkpoint, sims=sims, device=device, seed=seed,
+                       strategy=strategy, reuse=reuse, leaf_batch=leaf_batch, **kwargs)
```

Nothing else in `runner.py` / `matrix.py` needs to change: `act(game) -> dict` plus
`reset()` is the entire contract, `reset()` already clears the retained tree, and a
heterogeneous population is `MCTSPlayer(sims=...)` with different budgets / checkpoints
mixed into `default_population` exactly like `ScriptedPlayer` specs are today.
`tests/test_batched.py::test_make_player_is_a_tournament_shaped_player` runs the resulting
object against a live MLB game for six decisions.

Two contract details the runner relies on and this satisfies:

* `_drive_to_next_nemesis` steps unconditionally (`game.step(player.act(game))`), so `act`
  must never return `None` — `make_player` sets `no_action={"type": "advance"}` for exactly
  that. (`MCTSPlayer` used directly still returns `None`, which is what `MLBMatch` and the
  self-play agent need to detect a stuck state.)
* The runner is deliberately "blind" at a Nemesis (`pvp_opponent_score` stays 0,
  TOURNAMENT_NOTES §8 / decision 0.2). The search reads whatever the game says; it
  does not require the relay. If a future runner *does* relay, tree reuse handles it
  correctly by discarding (table D's relay row).

### 7.3 The optional runner change that unlocks cross-tree batching

`Tournament.run` currently drives agent i to its Nemesis before starting agent i+1
(`runner.py::_drive_to_next_nemesis`), so no two trees ever want a leaf at the same moment.
To batch across agents the per-agent loop has to become a lockstep one:

```python
group = BatchedMCTSPlayerGroup(n_agents, policy, config, reuse=True)   # from agent
while any_alive:
    live = [g if needs_action(g) else None for g in games]
    for g, a in zip(games, group.act_many(live)):     # ONE forward pass per round
        if a is not None:
            g.step(a)
```

That is a real change to W2's runner (the drive loop, the "exhausted at a Nemesis"
bookkeeping and the engine-gap repair all move inside the lockstep loop), it only pays when
every agent is an MCTS player, and per table B it is worth ~1.4x. **Recommendation: do the
`players.py` diff now, and only do the lockstep runner if the tournament becomes
MCTS-vs-MCTS and the profile still says the NN matters** — on the evidence in §6,
fixing the 436-actions-per-leaf cost is worth more than fixing the batch.


---

## 8. Found, not fixed

**Needs engine change (frozen in Phase 3, worked around, none blocking):**

* `engine/balatro_sim/game.py` — `clone()` (69 us) plus `step()` (~0.08 ms averaged over
  a descent, far more for a `play` that scores a full hand) are **50-66% of a batched
  search** (table B): the throughput wall once the NN is batched. Nothing in `agent` can
  move it. If Phase 4 wants more simulations per second, this is where they are.
* `engine/balatro_sim/game.py:894` — `state_signature()` costs 42 us and blake2b-hashes
  the whole RNG state. Tree reuse calls it twice per decision (0.2 ms, negligible), but
  anything wanting a *cheap* state identity — a signature per root edge inside the search,
  or a transposition table — would need a cheaper incremental version. Not needed for what
  W3 built.

**Found, not fixed (agent layer, deliberate):**

* **436 `Node` objects per expansion.** A 500-simulation vanilla tree is 26 304 nodes /
  ~15 MB; a Nemesis tree is 119 494 nodes. K=100 concurrent trees is ~1.5 GB in a vanilla
  blind. `Node` is a plain dataclass; `@dataclass(slots=True)` would cut that ~2.5x and
  action subsampling (§6, item 1) would cut it ~20x. Both change files the training design
  owns, so neither was done here.
* **`MCTSConfig.leaf_batch > 1` changes the search**, and about a third of its speedup is a
  shallower tree rather than fewer forward passes (table B2). It is off by default, and
  every test that asserts equality with the serial search runs at L=1.
* **`BatchedNNPolicy` is not bit-identical to `NNPolicy`** (§3, ~1e-10 on priors).
  No test asserts exact equality against the net; the exact-equality tests use
  `UniformPolicy`.
* **The padded softmax block is `(B, max_N)`.** One 436-action leaf batched with 99
  two-action leaves pads to (100, 436). Harmless at these sizes; worth knowing if leaf
  sizes ever become extremely skewed.
* **No transposition table.** Two different action orders reaching the same state build two
  subtrees. `state_signature()` makes it detectable and its 42 us makes it expensive; not
  attempted.
* **`train/loop.py` still runs episodes one at a time.** The self-play trainer would get the
  K-tree win for free by stepping K games in lockstep through `BatchedSearch`, but that is a
  training-loop change (checkpoint counters, buffer accounting) and W1 owns its shape.
  `BatchedSearch.run_requests` is the API it would call.
* **The reuse hit-rate denominator includes shortcut decisions.** A single-action state
  consults the cache (to walk the tree down) but runs no search, so it counts as a decision
  in `hit_rate` / `node_fraction` but not in `effective_sims`. Deliberate: those are two
  different questions.


---

## 9. Gates and how to run

| gate | result |
|---|---|
| `python -m pytest agent/tests -q` | **131 passed / 0 failed** (85 from W1, +46 from W3) |
| `python -m pytest engine/tests -q` | **1609 passed / 10 skipped / 3 xfailed / 0 failed** — unchanged |
| `python -m pytest tests -q` | **1073 passed / 2 xfailed / 0 failed** — unchanged |
| frozen-file check vs `docs/phase3_frozen_snapshot.txt` | **41/41** identical |
| `python agent/benchmarks/bench_batched.py` | the tables in §5 |

W3 test breakdown: `test_batched.py` 29 (policy equality, chunking, no-action leaves, K-tree
equality for uniform and NN policies, heterogeneous budgets, early finish, virtual-loss
bookkeeping, the player group, the pre-W3 reference comparison, `featurize_actions`
equality, the tournament factory), `test_reuse.py` 17 (engine determinism, resumption
equivalence, keep/discard, budget modes, Gumbel-on-a-reused-root, noise).


```bash
python -m pytest agent/tests -q                       # 131 passed (85 W1 + 46 W3)
python agent/benchmarks/bench_batched.py              # the tables in §5
python agent/benchmarks/bench_batched.py --only reuse --reuse-decisions 30
python agent/benchmarks/bench_batched.py --ruleset mlb --nemesis --encoder mlb
python agent/benchmarks/bench_search.py --sims 500 --device cuda   # W1's, still valid
```

```python
# K agents deciding together (no approximation)
from mcts import BatchedMCTSPlayerGroup, MCTSConfig, load_policy
group = BatchedMCTSPlayerGroup(n_agents=32, policy=load_policy(ckpt, device="cuda"),
                               config=MCTSConfig(num_simulations=500), reuse=True)
actions = group.act_many(games)          # one forward pass per lockstep round

# one agent, batching its own leaves (approximate search)
from mcts import make_player
player = make_player(ckpt, sims=500, device="cuda", leaf_batch=16, reuse=True)
```
