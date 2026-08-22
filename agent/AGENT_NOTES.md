# AGENT_NOTES — Phase 3 W1: agent-layer fork + sync + checkpointing

**Agent W1, 2026-08-21.** Deliverable: `mp/agent/**` — the MCTS agent layer forked from
`balatro-mcts` and re-targeted onto the frozen `mp/engine` fork, plus checkpointing, a
batchable policy/value interface for W3, and MLB awareness. Nothing under `mp/engine/**`
or `mp/rng/**` was touched (verified against `docs/phase3_frozen_snapshot.txt`: all 41
frozen files byte-size identical, mtimes within the snapshot's 1-second rounding).

---

## 0. Provenance

| Source | Commit | What was taken |
|---|---|---|
| `C:/Users/Taggart/projects/recovered/balatro-mcts` | `ee75d11` (master) | `mcts/` (7 modules), `train/` (4), `scripts/` (3), the three MCTS test files |
| `mp/engine` (this repo, FROZEN) | Phase 2 close | `balatro_sim` engine, `env_v7._encode_obs`, `legal_actions()`, MLB hooks |

The source repo is **read-only for this campaign**: nothing was edited or committed
there. It was executed once, read-only with `PYTHONDONTWRITEBYTECODE=1`, to get an
apples-to-apples sims/sec baseline on this box (§6); `git status` there is still clean.

### File map

| `mp/agent/…` | Source | State |
|---|---|---|
| `mcts/action.py` | `mcts/action.py` | +`reroll_boss` documented (falls through the arg-free tail unchanged) |
| `mcts/node.py` | `mcts/node.py` | +`stop_reason`; `is_terminal` now means "descent stops" (game over OR no legal actions) |
| `mcts/encoder.py` | `mcts/encoder.py` | **rewritten** — reuses `env_v7._encode_obs` instead of copying it; 434 → **447**; +`MLBEncoder` (453) |
| `mcts/action_features.py` | `mcts/action_features.py` | **re-targeted** — 12 → 13 action types, slot bounds widened, overflow feature; 44 → **56** dims |
| `mcts/model.py` | `mcts/model.py` | same architecture; V7-warm-start affordances removed; +`describe()`, +`score_actions_flat()` (for W3) |
| `mcts/policy.py` | `mcts/policy.py` | +`PolicyValueFn` protocol with `evaluate_many`, +`PolicyValueBase`, +`NNPolicy.encode_leaf` |
| `mcts/search.py` | `mcts/search.py` | outcome is a parameter; no-action states handled; `_evaluate_leaf` seam |
| `mcts/outcome.py` | — | **new** — `VanillaOutcome` / `MLBOutcome` / `ExternalOutcome` |
| `mcts/player.py` | — | **new** — `MCTSPlayer.act(game) -> action or None`, the W2/W4 plug-in shape |
| `train/trajectory.py` | `train/trajectory.py` | +`state_dict`/`load_state_dict` on the buffer |
| `train/trainer.py` | `train/trainer.py` | +optimizer `state_dict`/`load_state_dict` |
| `train/agent.py` | `train/agent.py` | outcome parameter, `EpisodeResult`, stuck handling, `resume_episode`, `pvp_target_fn` |
| `train/loop.py` | — | **new** — `ColdTrainer` / `TrainConfig`; the script and the tests share it |
| `train/checkpoint.py` | — | **new** — atomic save/load, RNG state capture |
| `scripts/mcts_demo.py` | same | +`--ruleset/--nemesis/--repeat`, MLB demo state, handles `chosen=None` |
| `scripts/smoke_selfplay.py` | same | rebuilt on `ColdTrainer`; +checkpoint round-trip step |
| `scripts/train_cold.py` | same | +checkpointing, `--resume`, `--episodes`, MLB flags, run directories |
| `scripts/_bootstrap.py` | — | **new** — sys.path + fork guard for scripts |
| `benchmarks/bench_search.py` | — | **new** — sims/sec + NN/sim/other split; W3's baseline |
| `tests/test_gumbel.py` | `balatro_sim/tests/test_gumbel.py` | re-targeted + no-action cases |
| `tests/test_nn_policy.py` | `balatro_sim/tests/test_nn_policy.py` | re-targeted + encoder-equality + batching-seam tests |
| `tests/test_train.py` | `balatro_sim/tests/test_train.py` | re-targeted + outcome tests |
| `tests/test_action_space.py` | — | **new** — action vocabulary vs the fork engine, fork guard |
| `tests/test_checkpoint.py` | — | **new** — the round trip |
| `tests/test_mlb_agent.py` | — | **new** — MLB smoke, `PVP_WAIT`, endless-outcome |

**Not forked:** `balatro_sim/tests/test_clone*.py` (3 files, 32 tests) — they are engine
tests and already live in `mp/engine/tests/sim_tests/`, adapted to the fork (Phase 0
FORK_NOTES §4). `balatro_sim/tests/test_edge_cases.py`'s `TestActionMasking` (the
historical "8 failures") is **already resolved** in the fork: it needs the repo-root
`train_sim` module, and Phase 0 skipped it with a reason. It is an engine test, not an
agent test — nothing to do here. `test_hand_eval.py` / `test_scoring.py` likewise.

Historical count reconciliation: the source repo's 66 = 32 clone tests (now in
`mp/engine/tests`) + 34 MCTS/train tests. The re-targeted MCTS/train tests are 56 here
(the originals plus the fork-specific cases), and the suite totals **85**.

---

## 1. Layout and how to run

```
mp/agent/
├── pytest.ini      pythonpath = . ../engine ; testpaths = tests   (rootdir = mp/agent)
├── conftest.py     puts mp/agent + mp/engine first on sys.path; RAISES if the wrong
│                   balatro_sim / mcts / train wins
├── mcts/  train/  scripts/  benchmarks/  tests/
└── runs/           training runs + checkpoints (gitignored: `agent/runs/` in mp/.gitignore)
```

```bash
# from the repo root
python -m pytest mp/agent/tests -q                       # 85 passed
python mp/agent/scripts/mcts_demo.py --policy both --strategy both --sims 100 500 2000
python mp/agent/scripts/smoke_selfplay.py --device cuda
python mp/agent/scripts/train_cold.py --minutes 2 --device cuda --checkpoint-every 25
python mp/agent/scripts/train_cold.py --resume mp/agent/runs/<name>/latest.pt --minutes 2
python mp/agent/benchmarks/bench_search.py --sims 500 --device cuda
```

`mcts` and `train` are top-level packages reached by putting `mp/agent` on `sys.path`,
exactly as `balatro_sim` is reached through `mp/engine` — same pattern as
`mp/engine/pytest.ini` + `conftest.py`, for the same reason (`pytest mp/agent/tests` from
the repo root would otherwise put the repo root's cwd entry first and import the BRL
`balatro_sim`). The guard is asserted twice: at conftest import and again in
`tests/test_action_space.py::test_imports_resolve_to_the_fork`. Scripts get the same
treatment from `scripts/_bootstrap.py`, which works because `python <path>/x.py` puts the
script's directory on `sys.path[0]`.

---

## 2. The re-target: what changed and why

### 2.1 Observation — 434 (copied) → 447 (reused)

The original `mcts/encoder.py` was a **copy** of V7's `_encode_obs` pinned at 434 dims so
"V7's value head + trunk weights can be loaded as a warm-start with no remapping". Both
halves of that are dead: the V7 checkpoint is lost (memory, confirmed 2026-06-10), and
the fork's encoder is 447 dims (434 → 443 at the 2026-07-29 audit → 447 when W5 widened
the boss one-hot to all 23 regular + 5 showdown bosses). Copying is also exactly how the
two repos drifted apart in the first place.

So `mcts/encoder.py` now calls `balatro_sim.env_v7.BalatroV7Env._encode_obs` directly.
That method reads exactly one attribute of `self` (`self.game`), so it is bound to a
one-slot shim and used as a pure function. `tests/test_nn_policy.py::
test_encoder_matches_env_v7` constructs a real `BalatroV7Env` and asserts byte equality
with `encode_obs(env.game)` — if a future env_v7 edit makes `_encode_obs` touch more of
`self`, that test fails instead of the agent silently mis-encoding.

**MLB extension, opt-in, `MLB_OBS_DIM = 453`.** The brief allows extending "if the MCTS
obs must differ, and justify". Under MLB the V7 block is blind to three decisive things,
so `encode_obs_mlb` appends six features: `lives/4`, `is_pvp`, `opponent hands left`,
`comeback pending`, `pvp started/waiting`, and a log-scaled ante (MLB is endless, so V7's
`ante/8` saturates at 1.0 exactly when the interesting part begins). Default is still the
447-dim `v7` encoder; the checkpoint records which one a run used and refuses to resume
into the other.

### 2.2 Action features — 44 → 56, 12 → 13 types

The fork's `legal_actions()` (`mp/engine/balatro_sim/game.py:1342`) emits `reroll_boss`
(Directors Cut / Retcon voucher), which the original's 12-type vocabulary would have
featurized as the all-zero "unknown" row. Worse, the original's slot bounds were the V7
action space (8 hand slots, 2 consumables, 7 shop, 5 jokers) and out-of-range indices
were **silently dropped**, so two different actions could share one feature row and
therefore one prior. Every one of those bounds is exceeded by real play: `hand_size` grows
(Juggler / Turtle Bean / Ouija / Ectoplasm / vouchers, `game.py:993-1026`),
`consumable_slots` grows (Crystal Ball, Negative), `joker_slots` grows (Negative jokers).

Now: 13 types, 12 hand slots, 4 consumables, 8 shop, 8 joker, 8 booster, plus one
**overflow scalar** `(i+1)/20` for anything still past a bound, so overflowing actions stay
distinguishable. `tests/test_action_space.py` walks real games on 3 seeds × both rulesets
and asserts (a) every emitted action type is in the vocabulary, (b) `action_key` →
`action_from_key` → `step()` round-trips on a clone for every action, (c) keys are unique
per legal-action list, (d) **feature rows are unique** per legal-action list.

The walk reaches all 13 types including `reroll_boss`, and all six game states.

### 2.3 Game keys

The engine now speaks `j_*` / `c_*` / `v_*` / `bl_*` (Phase 1 `REKEY_NOTES.md`). Nothing
in the agent hardcodes a key: the encoder reads `shop.JOKER_CATALOGUE` (150 `j_*` entries)
and the consumable/voucher tables through env_v7, and actions are indices, not keys. The
one place a stale key list existed — the original encoder's hardcoded 15-key
`BOSS_TYPES` — is gone with the copied encoder. `test_engine_speaks_game_keys` pins the
catalogue shape the encoder depends on.

### 2.4 Warm start

Removed, not left inert-but-tempting: `model.py` no longer claims V7 compatibility and
there is no load-V7 path anywhere. The trunk is still V7-*shaped* (hidden 512, 4 residual
blocks) because that shape was tuned on this game, and `PolicyValueNet.describe()` /
`from_description()` put the dims in the checkpoint so a mismatched net fails at load.

### 2.5 Outcome signal — now a parameter (`mcts/outcome.py`)

The original hardcoded `_is_win = (GAME_OVER and ante > 8)` in `search.py` and
`z = (blinds + chips)/24` in `train/agent.py`. Under MLB both are wrong: the run is
**endless** (passing ante 8 is not a win — `MLB_NOTES.md` §2), the win is `game.match_won`,
and what a Nemesis actually produces is a score margin against an opponent the game cannot
see. Three implementations behind one protocol:

| | `is_terminal` | `is_win` | `value` (the z label / backup) |
|---|---|---|---|
| `VanillaOutcome` | GAME_OVER | GAME_OVER and ante > 8 | `(blinds_completed + chip_ratio)/24`, win → 1.0 |
| `MLBOutcome` | GAME_OVER | `game.match_won` | `0.5·lives/starting + 0.5·blind_progress`, win → 1.0 |
| `ExternalOutcome` | caller's (default MLB's) | caller's | **caller's** — W2's N×N margin, W4's paired margin, `MLBMatch`'s verdict |

`ExternalOutcome.from_margin(fn, scale)` wraps a signed margin (e.g.
`log10(mine) - log10(theirs)`) through a logistic into [0, 1]; margin 0 → 0.5, which
matches the server's "exact tie costs nobody a life" rule. `default_outcome_for(game)`
picks Vanilla/MLB off `game.mlb`, so callers that do not care pass nothing.
`search._is_win` and `train.agent._shaped_z` survive as documented back-compat shims.

---

## 3. Checkpointing

`train/checkpoint.py` + `ColdTrainer.state_dict()`. One `torch.save` pickle holding:

* **model** — `state_dict()` + the constructor description (`obs_dim`, `hidden`, …)
* **trainer** — Adam's `state_dict()` (the moments; dropping them is a silently different run)
* **counters** — episodes / samples / train steps / wins / errors / cumulative wall clock
* **rng** — the numpy `Generator` state, torch CPU + CUDA generator states, Python `random`
* **config** — the full `TrainConfig`
* **buffer** — the replay buffer (optional, on by default, capped)

Writes are atomic (temp file + `os.replace`), so Ctrl+C during a write cannot destroy the
previous checkpoint. `torch >= 2.6` defaults `torch.load(weights_only=True)`, which cannot
load numpy arrays, so `load_checkpoint` passes `weights_only=False` explicitly and refuses
files that are not ours (`kind` / `version` check). `--resume` also refuses a config that
changes the experiment (`encoder`, `ruleset`, `deck`, `stake`, net shape); `device` may
change.

### Round-trip result: **BIT-EXACT** (on CPU)

`tests/test_checkpoint.py::test_train_3_save_resume_1_equals_train_4` trains 3 episodes,
saves, rebuilds a `ColdTrainer` from the checkpoint, trains 1 more, and compares to an
uninterrupted 4-episode run with `torch.equal` on **every parameter** and on **every Adam
moment**, plus identical counters and buffer length. It passes. A second test asserts the
resumed run's 4th episode has the same game seed, trajectory length and final ante as the
uninterrupted run's 4th. A third does the same end-to-end under `ruleset="mlb"` with the
453-dim MLB encoder.

Bit-exactness is possible because **one** seeded `numpy.random.Generator` drives every
choice the trainer makes (episode seed → Gumbel noise → replay-batch indices), the engine
is deterministic given its seed (Phase 1: 126/126 exact), Adam is deterministic, and the
checkpoint carries the buffer so the mini-batch stream is reproduced. Caveats, both
tested and reported rather than papered over:

* **CUDA is not bit-exact.** cuBLAS/cuDNN reduction order is not reproducible by default,
  so a CUDA resume is statistically the same run, not bit-identical. The round-trip tests
  pin `device="cpu"`.
* **`--no-checkpoint-buffer` is not bit-exact** and neither is a *truncated* buffer
  (`--buffer-checkpoint-cap`); both cases are flagged in the checkpoint (`truncated`) and
  `train_cold.py --resume` prints a warning when it sees them.

### Sample size — the thing to fix before a long run

A `Sample` is **~97 KB**, almost entirely `action_features`: at `SELECTING_HAND` there are
~436 legal actions × 56 features × 4 bytes = 97 KB, against 1.8 KB for the observation.
Consequences, measured: 141 episodes (1 374 samples) → a **130 MB** checkpoint; the
balatro-mcts default `buffer_capacity=200_000` would be **~19 GB of RAM** and an
unshippable checkpoint.

Mitigations landed: default `buffer_capacity` 200 000 → 20 000 (~1.9 GB), default
`--buffer-checkpoint-cap` 5 000, and **numbered checkpoints are weights-only** (28.9 MB)
while only `latest.pt` carries the buffer (137.8 MB after 153 episodes). Resume from
`latest.pt`.

**Recommended structural fix (not W1's to make):** subsample the action set per training
sample — keep the visited actions plus a random subset of the rest and renormalise the
policy target, which is the standard treatment for huge action spaces. That is ~20×
smaller samples and would make a large replay buffer affordable. It changes the training
objective slightly, so it belongs to whoever owns the training design.

---

## 4. For W3 — the interface, the seam, the hot loop, the baseline

### 4.1 The interface you implement

```python
# mcts/policy.py
class PolicyValueFn(Protocol):
    def __call__(self, game) -> tuple[dict[ActionKey, float], float]: ...
    def evaluate_many(self, games: Sequence[BalatroGame]) -> list[tuple[dict, float]]: ...
```

* `PolicyValueBase` supplies a correct serial `evaluate_many` (a loop over `__call__`), so
  every existing policy already satisfies the protocol — including the stubs in the tests.
* `NNPolicy` is the reference implementation. Its per-leaf CPU work is factored out as
  `NNPolicy.encode_leaf(game) -> (legal, obs, action_feats) | None` (pure numpy, no torch)
  and `NNPolicy.priors_from_logits(legal, probs)`. **Reuse both**: your batched
  implementation should differ from the reference only in the torch part, which is what
  makes "batched == single-leaf" a meaningful test.
* `PolicyValueNet.score_actions_flat(trunk, action_feats, counts)` already exists for you:
  it scores a RAGGED batch (B states, `sum(counts)` actions) in ONE policy-head call via
  `repeat_interleave`. `tests/test_nn_policy.py::test_score_actions_flat_matches_per_state`
  pins it equal to the per-state path.
* A game with no legal actions must return `({}, 0.0)` **without touching the net** — MLB
  `PVP_WAIT` and readied-at-the-Nemesis states show up in real batches.

### 4.2 Where leaf evaluation happens

Exactly one call site:

```python
# mcts/search.py
def _evaluate_leaf(self, game):        # <- the seam
    return self.policy_value_fn(game)
```

called only from `MCTS._expand`. Everything else in the search is bookkeeping. A
leaf-parallel or virtual-loss variant replaces this one method (collect N leaves, one
`evaluate_many`, distribute); a multi-tree batcher can leave `search.py` untouched
entirely and batch across the N trees that the tournament/self-play steps concurrently —
that is the higher-value direction, see 4.4.

### 4.3 The hot loop

`MCTS._simulate` → `while node.is_expanded and not node.is_terminal:` → `_select_child`
(PUCT over `node.children`, a dict with up to ~436 entries) → `game.step(action_from_key(k))`
→ stop checks (`outcome.is_terminal` / `outcome.is_stuck`, both O(1) state comparisons —
deliberately NOT `legal_actions()`, which is the combinatorial enumeration) → `_expand`.
Per simulation: one `root_game.clone()` (~63 µs, `mp/engine/FORK_NOTES.md` §6), a few
`step()`s (~27 µs each), one leaf evaluation.

**Tree reuse note (your (b)):** the engine is stochastic through keyed RNG, so a subtree is
only valid if the state after the action is the state the subtree was built from. The
engine gives you `game.state_signature()` for exactly this. `MCTSConfig.discount` is 1.0
and values are in [0, 1] throughout, so nothing else needs rescaling on reuse.

### 4.4 Measured baseline (RTX 3080 Ti box, Python 3.13.5, torch 2.6.0+cu124)

`bench_search.py`, demo state = ante-1 Small blind, 3 jokers, **436 legal actions**.
sims/sec is the best of 3 runs; the %-split comes from a separate instrumented run.

| policy | strategy | sims/sec | ms/sim | in `_evaluate_leaf` | in `clone`+`step` | other (Python) |
|---|---|---|---|---|---|---|
| uniform | puct | 1 327 | 0.75 | 51.7% | 19.1% | 29.2% |
| uniform | gumbel | 1 437 | 0.70 | 26.2% | 37.1% | 36.7% |
| nn-cpu | puct | 451 | 2.22 | 39.4% | 46.7% | 13.9% |
| nn-cpu | gumbel | 475 | 2.11 | 48.8% | 41.8% | 9.5% |
| nn-**cuda** | puct | 362 | 2.77 | 53.1% | 36.6% | 10.4% |
| nn-**cuda** | gumbel | 328 | 3.05 | 58.7% | 34.6% | 6.7% |

At the **MLB Nemesis** (`--ruleset mlb --nemesis --encoder mlb`), nn-cpu drops to
**308-313 sims/sec** with **78%** of the time in `_evaluate_leaf`: a PvP round never ends
on `chips >= target`, so descent runs deeper and every extra ply is another forward pass.
Batching matters more there than in a vanilla blind.

**The headline for you: single-leaf CUDA is SLOWER than CPU** (328-362 vs 451-475). The net
is 2.4 M params and each leaf is one (447,) obs + one (436, 56) action block — far too
small to amortise a kernel launch plus two host↔device transfers. That is the entire case
for batched inference, and it means your win is measured against the **CPU** number, not
the CUDA one.

`mcts_demo.py` (same state, best of 2, PUCT with Dirichlet noise / Gumbel):

| sims | uniform puct | uniform gumbel | nn-cpu puct | nn-cpu gumbel |
|---|---|---|---|---|
| 100 | 1 379 | 2 140 | 430 | 539 |
| 500 | 1 010 | 1 275 | 468 | 495 |
| 2 000 | 1 187 | 1 359 | 459 | 496 |

**Is the fork slower than the original?** No. Running the untouched `balatro-mcts`
`scripts/mcts_demo.py` on its own (old) engine on this same box, read-only: uniform puct
1 215, uniform gumbel 1 824, nn-cpu puct **475**, nn-cpu gumbel **554** sims/sec at 500
sims — i.e. within a few percent of the fork despite the fork's wider obs (447 vs 434) and
action features (56 vs 44). The brief's historical "~745 sims/s with NN" does not
reproduce on this hardware **even with the original code**, so treat 745 as a
different-machine number, not a regression.

---

## 5. MLB awareness

* **The agent can play `ruleset="mlb"`.** `tests/test_mlb_agent.py` runs a self-play
  episode on an MLB game to GAME_OVER or an ante cap, and separately advances to the
  ante-2 Nemesis and plays it.
* **External target.** `SelfPlayAgent(pvp_target_fn=...)` is called on every decision at a
  PvP blind and forwards to `game.set_pvp_info(score, hands_left)` — the `enemyInfo` relay.
  It re-applies each decision because `startBlind` resets the target to 0. W2/W4 own the
  value; the agent only relays it.
* **No-action states are normal.** MLB has two: `State.PVP_WAIT` (hands exhausted at the
  Nemesis, waiting for the opponent) and BLIND_SELECT with `pvp_ready` (readied, waiting
  for `startBlind`). `mcts.outcome.is_stuck_state()` detects both by state (cheap);
  `MCTS.run` / `run_gumbel` on such a root return empty visit counts and `chosen=None`
  instead of raising (the original raised `RuntimeError`), a stuck node inside the tree
  stops descent with `stop_reason="stuck"` and is valued by the outcome's estimate, and
  `MCTSPlayer.act()` returns `None`. `SelfPlayAgent.play_episode` stops with
  `stop_reason="stuck"` and `resume_episode(game, prior)` continues the SAME trajectory
  after the driver calls `end_pvp()` — so one episode still gets one z label.
* **No ante-8 assumption.** `MLBOutcome.is_win` is `game.match_won`;
  `test_mlb_game_over_is_not_valued_as_a_win` pins that an MLB run dying at ante 12 gets a
  value < 1.0 where the vanilla rule would have handed the value head a 1.0.
* **`MCTSPlayer`** (`mcts/player.py`) is the `act(game) -> action | None` shape W2's
  `players.py` protocol expects. It skips the search on single-action states
  (`ROUND_EVAL`'s dummy `advance`), which is free correctness *and* throughput.

---

## 6. Gate runs (repo root, `python` 3.13.5, `-p no:cacheprovider`)

| gate | result |
|---|---|
| `python -m pytest mp/agent/tests -q` | **85 passed / 0 failed** (26 s) |
| `python -m pytest mp/engine/tests -q` | **1609 passed / 10 skipped / 3 xfailed / 0 failed** — unchanged |
| `python -m pytest mp/tests -q` | **1073 passed / 2 xfailed / 0 failed** — unchanged |
| frozen-file check vs `docs/phase3_frozen_snapshot.txt` | 41/41 identical size, mtime within the snapshot's 1 s rounding |
| `mcts_demo.py` sims/sec | §4.4 |
| `smoke_selfplay.py --device cuda` | OK — 11-decision episode, finite losses, weights moved, checkpoint round-tripped |
| `train_cold.py --minutes 2 --device cuda --checkpoint-every 25` | **153 episodes, 0 errors, 7 checkpoints**, 75.9 ep/min, buffer 1 472 |
| `train_cold.py --resume …/latest.pt --minutes 1 --device cuda` | **resumed at episode 154, ran to 232**, 0 errors, 4 checkpoints, buffer 1 472 → 2 207 |

Agent-suite breakdown: `test_nn_policy` 27, `test_train` 21, `test_checkpoint` 10,
`test_action_space` 10, `test_mlb_agent` 9, `test_gumbel` 8.

---

## 7. The 2-minute CUDA run

```
python mp/agent/scripts/train_cold.py --minutes 2 --device cuda --sims 30 \
       --checkpoint-every 25 --log-every 25 --run-name cuda_smoke
```

153 episodes in 2m00s (75.9 ep/min), 0 errors, 7 checkpoints. Every episode ends in ante 1
with ~10 decisions and shaped z ≈ 0.08 — a random-init policy at 30 sims/decision does not
clear the first Small blind, which is the expected cold-start picture, not a bug. The
value loss falls 0.018 → 0.0006 (the value head learns the constant z fast); the policy
loss sits at ~4.4 ≈ `log(436/5)`, i.e. it is still near-uniform over the action set.

Checkpoints: numbered `ckpt_NNNNNN.pt` **28.9 MB** (weights + Adam, no buffer), `latest.pt`
**137.8 MB** (with 1 472 buffered samples). `--keep-checkpoints 3` prunes the numbered ones.

Then, immediately after:

```
python mp/agent/scripts/train_cold.py --resume mp/agent/runs/cuda_smoke/latest.pt \
       --minutes 1 --device cuda
```

picked up at **episode 154** (counters, weights, optimizer, RNG and the 1 472-sample
buffer all restored), ran 79 more episodes to **232 total** in 1m00s with 0 errors and 4
more checkpoints, buffer 1 472 → 2 207, policy loss continuing down 4.45 → 4.29 across the
boundary rather than restarting. The run directory ends with 3 numbered weights-only
checkpoints + `latest.pt` (195.5 MB) + the JSONL log, whose `kind: "config"` line records
`resumed_from` and `start_episode`. Both runs are in `runs/cuda_smoke/` (gitignored).

---

## 8. Found, not fixed / gaps / needs-an-engine-change

**Needs engine change (none blocking, all worked around):**

* `mp/engine/balatro_sim/env_v7.py:702` — the observation encodes only the first
  **8 hand slots** and **5 joker slots**, but the fork's `hand_size` and `joker_slots` both
  grow past those bounds in real play. A 9th card in hand is invisible to the value/policy
  net while being a perfectly legal `play` target. The *action* features handle it (§2.2);
  the observation cannot without changing `OBS_DIM`, and env_v7 is frozen in Phase 3.
  Worth widening in Phase 4 (it also changes the checkpoint format, so do it before a long
  training run, not after).
* `mp/engine/balatro_sim/env_v7.py` `_finish_step` pays `R_BLIND_BASE * (9 - ante)`, which
  goes negative past ante 8 under endless MLB (already logged in `MLB_NOTES.md` §5). The
  agent does not use env_v7's reward — it uses `OutcomeFn` — so this only matters if
  someone drives the agent through `BalatroV7Env`/`env_mp`.

**Gaps I did not close:**

* No batched inference, no tree reuse — W3's deliverable, deliberately untouched beyond
  the interface and `score_actions_flat`.
* `legal_actions()` semantics are inherited (booster picks reduced, tarot targets
  enumerated for sizes 0-2 regardless of the tarot's real target count, `ROUND_EVAL`
  represented by a dummy `advance`). The agent skips the search on those single-action
  states but does not fix the enumeration.
* Self-play under MLB is currently **solo** (`pvp_solo=True`: the Nemesis resolves at hand
  exhaustion). Real two-player self-play needs `MLBMatch` driving two agents — that is the
  W2/W4 wiring, and the `MCTSPlayer` + `resume_episode` pair is the half of it I own.
* The value target for a Nemesis is still an episode-level z. The natural MLB target is
  the per-Nemesis margin, which needs W2's matrix; `ExternalOutcome` is where it plugs in.
* `Sample` size (§3) — flagged, mitigated, not structurally fixed.
* CUDA determinism (§3) — not pursued (`torch.use_deterministic_algorithms` would cost
  throughput for a property only the test needs, and the test runs on CPU).
