# engine — Reproducibility Notes (W7 / P1-repro, 2026-08-21)

Scope: the env layer's *hypothetical* scoring — the "best possible hand" reward
estimate in `env_v7` and the combo ranking in `env_sim` / `env_v5` — must have
**zero** effect on game state and must be deterministic. Source of the bug
inventory: `docs/MP_UPDATE_LIST_2026-08.md` §3.

Files owned and changed: `balatro_sim/env_v7.py`, `env_sim.py`, `env_v5.py`,
`card_selection.py`, new `tests/sim_tests/test_env_rng_isolation.py`, this file.
`env_mp.py` audited, unchanged. Nothing committed to git.

---

## 1. What was wrong (measured, not inferred)

A probe built a `SELECTING_HAND` state with Lucky cards, Hearts/faces/an 8 and
the board `[Bloodstone, Business Card, Misprint, Space Joker, 8 Ball]`, snapshotted
`gs.rng.getstate()`, every `JokerInstance.state`, `planet_levels`, dollars, hand
and the process-global `random` state, then called the estimate twice.

| Env | Call site | `gs.rng` advanced | joker `state` mutated | `planet_levels` mutated | global `random` advanced | Estimate call 1 vs call 2 (seed 0) |
|---|---|---|---|---|---|---|
| `env_v7` | `_best_hand_score` (`rng=gs.rng`) | **yes** | **yes** | **yes** | no | 3928 → **14556** |
| `env_sim` | `_update_play_combos` (no `rng=`) | no | **yes** | **yes** | **yes** | combo ranking changed |
| `env_v5` | `_update_play_combos` (no `rng=`) | no | **yes** | **yes** | **yes** | combo ranking changed |

So the brief's description was complete but understated. Beyond the RNG draws,
`score_hand()` is not pure in four more ways, all of which leaked out of the
"dry run":

1. **Scaling jokers bump `inst.state` in `on_score_card` / `on_hand_scored`** —
   Ride the Bus, Loyalty Card, Vampire (`xmult`), Obelisk, Card Sharp, Ice Cream,
   Popcorn, Hit the Road, Wee, ... Each hypothetical candidate applied one more
   "hand played" to the live joker.
2. **Space Joker writes `ctx.planet_levels`** (`jokers/mult.py:383-384`), and the
   envs passed the live `gs.planet_levels` dict. The estimate levelled hands for
   free, 1-in-4 per candidate; that is why seed 0's second estimate quadrupled.
3. **Card mutation**: Vampire sets `card.enhancement = "None"` on scoring cards
   (`chips.py:224`), Midas Mask sets `"Gold"` (`misc.py:540`, `scaling.py:392`),
   Sixth Sense / DNA set `card.debuffed` (`misc.py:286`, `misc.py:401`). The envs
   passed the live hand cards.
4. **Nested containers in `inst.state`**: Card Sharp keeps a `set` under
   `state["played_hands"]` and mutates it in place in `on_hand_scored`. A shallow
   `state.copy()` (what `JokerInstance.clone()` does) does NOT isolate this — see
   §5 for the knock-on to `BalatroGame.clone()`.

The effect on training signal: the V7 card-quality reward
(`R_CARD_QUALITY * played / best`) compared the real play against a "best" that
was computed on a *different* world than the one the real play then happened in,
and the real play itself ran on a stream that had been advanced by 0–40 draws
depending on which subsets the estimate happened to score. The `env_sim`/`env_v5`
combo ranking — i.e. the *meaning of every action index 0–19* — depended on the
unseeded global `random`, so two same-seed envs diverged after the first hand.

## 2. The fix

### `card_selection.HypotheticalScorer` (new)

One helper, used by all three envs. Per hand it snapshots `gs.rng.getstate()`
once and creates a private `random.Random`. Per candidate it:

* resets the private RNG to the snapshot (`setstate`) — every candidate sees
  the same stream position, so the estimate is a pure function of
  (game state, candidate) and candidate order does not matter;
* scores against **fresh copies** of the candidate's cards (mapped so
  `scoring_cards` stays a sub-list of `all_cards`), the held cards (when
  `model_held=True`), a one-level-deep copy of every joker's `state`
  (`_clone_joker_for_dry_run`: primitives, flat lists/sets/dicts), and a copy of
  the `planet_levels` dict;
* `full_deck` is copied once per scorer (every current hook only *counts* it);
* passes `rng=<private rng>` explicitly — no call in the env layer can fall
  through to the global `random` any more.

Chosen over `gs.clone()`-per-candidate because `env_sim`/`env_v5` score all 218
subsets per refresh (218 × 63 µs ≈ 14 ms), whereas cloning only the mutable
surface costs ~8 µs per candidate. Chosen over "derive an integer seed from the
state" because `setstate` of the snapshot is the same determinism with no
hashing step and keeps the estimate's luck aligned with the stream the real play
will actually draw from.

### `env_v7._best_hand_score`
Phase 1 (pure `evaluate_hand` over all subsets) unchanged. Phase 2 now calls
`HypotheticalScorer(gs, model_held=True).score(cards, ht, sc)` for the ≤ 8
candidates. Same candidate selection, same held/full-deck modelling, same
`max()`. The unused `from .scoring import score_hand` import was dropped.

### `env_sim._update_play_combos`, `env_v5._update_play_combos`
Same scorer with `model_held=False`, preserving their historical ranking
semantics (they never modelled held-card or full-deck effects; adding that would
change action semantics, which is out of scope). The `score_hand` import is
replaced by the scorer import. `env_v5`'s unused top-level `import random` was
removed so the static guard test has nothing to argue about.

### Not changed
`OBS_DIM = 443`, every action mapping, every reward constant, the candidate
selection rules, and the formula `R_CARD_QUALITY * min(played/best, 1)`.

## 3. Did the numeric estimate change? **Yes** — and it had to.

Same rigged state, first call only (the only call that ever mattered in
training, since it runs once per play):

| seed | pre-fix `_best_hand_score` | post-fix |
|---|---|---|
| 0 | 3928 | 3928 |
| 1 | 1260 | 390 |
| 2 | 2650 | 2862 |
| 3 | 1102 | 480 |
| 4 | 3496 | 1748 |

Pre-fix, candidate *k* was scored on jokers that had already been "played" *k−1*
times by the earlier candidates (Ride the Bus / Vampire / Card Sharp bonuses
accumulated), on a planet table that Space Joker had possibly levelled, and on a
stream position the earlier candidates had advanced. That biased the estimate
upward for later candidates — which are the higher-priority hand types, because
`candidates_to_score` is walked best-priority-first — so "best" was
systematically inflated and the card-quality reward systematically deflated. The
post-fix numbers are what `score_hand()` returns for each candidate from the
actual current state with the actual current stream position. Seed 0 is
unchanged because its best candidate happened to be the first one scored.

For `env_sim`/`env_v5` the top-20 combo *ordering* also changed (e.g. seed 0's
second-ranked combo moved from `[0,1,3,4,5]` to `[0,1,3]`), for the same reason:
the old ranking rewarded whichever subsets were enumerated later. Any checkpoint
trained on those envs' action indices is already incompatible for other reasons
(see FORK_NOTES §7), so nothing relies on the old order.

The stochastic rolls inside the estimate are now **isolated but correlated with
the real play**: the first hypothetical's Lucky/Bloodstone rolls are exactly the
rolls the real play will get if the agent plays that same subset in that order.
That is a feature for a "how good was your choice" reward. If a future reward
wants the *expected* best score instead, average the scorer over several seeds
derived from the snapshot — the scorer is the right place to do that.

## 4. Cost

Post-fix vs pre-fix on the rigged 5-joker board (Python 3.13, this box):

| Path | pre-fix | post-fix |
|---|---|---|
| `env_v7._best_hand_score` (218 evals + ≤ 8 scores) | 0.71 ms | 0.95 ms |
| `env_sim._update_play_combos` (218 scores) | 2.39 ms | 4.07 ms |

The V7 path, the one that matters for training throughput, pays +0.24 ms per
play. `env_sim`/`env_v5` pay ~8 µs per candidate for the per-candidate copies;
they are legacy envs and nobody is training on them.

## 5. Audit findings outside the fix

* **`env_mp.py`** — no hypothetical scoring of its own; both proxies call
  `env_v7.step_hand`, so they inherit the fix. Its only direct RNG touch is
  `_revive_boss_if_needed` → `shop.generate_shop(game)`, a *real* state change
  (a real shop for a revived player) drawn from that player's own seeded
  `game.rng`, so it is reproducible. Two caveats for the shop-queue workstream:
  it regenerates a shop outside `_end_round`, so when the per-ante shop queue
  lands this call must consume from the same queue/pointer or the revived
  player gets an extra draw; and `_PlayerEnvProxy.__init__` constructs a
  throwaway `BalatroV7Env(seed)` (and therefore a throwaway `BalatroGame`) before
  swapping in the coordinator's game — harmless (separate `Random` instance,
  never stepped), just wasteful. The module docstring still says 434/438 dims;
  the code derives from `env_v7.OBS_DIM` so it is actually 443/447.
* **`card_selection.py`** — did no RNG before this change (subset enumeration,
  logits and boss validation are pure; sampling happens in the training script).
  The only RNG it now contains is the scorer's private `random.Random`, seeded
  explicitly from the game stream snapshot.
* **`JokerInstance.clone()` / `BalatroGame.clone()` share nested state**
  (`jokers/base.py`, not in my remit). `state.copy()` is shallow, so a clone and
  its original share Card Sharp's `played_hands` set, Satellite's `planets_used`
  set, and every `pending_consumables` list. Found the hard way: a test that
  compared "play on `game.clone()`" against "play via `step_hand`" got ×3 from
  Card Sharp on the second play because the first play had added the hand type
  to the *shared* set. MCTS expansion from a state holding any of those jokers
  will cross-contaminate siblings. Recommend `JokerInstance.clone()` adopt the
  one-level-deep copy used by `_clone_joker_for_dry_run`; it is a 3-line change
  in `jokers/base.py` for whoever owns it.
* **`hands_left` in the estimate** — `_best_hand_score` passes `gs.hands_left`
  while the real play passes `hands_left - 1` (`game.py` `_play_hand`). This
  affects "last hand" jokers (Dusk, Acrobat) in the estimate only. Pre-existing,
  not a reproducibility issue, left alone because it changes reward values.
* **Concurrent edits seen in my files** — P1-rekey's key-literal sweep touched
  `env_v7.py` (`j_invisible_joker` → `j_invisible`) and `env_v5.py`
  (`c_hierophant` → `c_heirophant`, `c_wheel` → `c_wheel_of_fortune`). Left in
  place; they are what makes the new `test_game_keys.py` pass.

## 6. Tests — `tests/sim_tests/test_env_rng_isolation.py` (40 tests)

* `test_estimate_has_no_side_effects[seed × env]` — rigged board with every
  stochastic + mutating joker; snapshot of `rng.getstate()`, deep-copied joker
  states, planet levels, dollars, consumables, hand/full-deck card fields,
  `_hand_type_counts` and the global `random` state must be identical after the
  estimate.
* `test_estimate_is_deterministic[seed × env]` — three calls, same result.
* `test_v7_reward_path_does_not_advance_rng_before_real_play[seed]` — two
  identical envs; one plays via the raw game, the other via `step_hand`
  (estimate first); full game fingerprint including `rng.getstate()` must match.
* Scorer unit tests: nested-set isolation (Card Sharp), `planet_levels`
  isolation (Space Joker, 218 candidates), card isolation (Vampire, 218
  candidates), candidate-order independence, and "seeded from the live stream"
  (same state ⇒ same score; one extra live draw ⇒ some candidate differs).
* `test_same_seed_same_actions_same_trajectory[seed × env]` — two envs, same
  seed, identical scripted action sequence (script RNG private to the test, the
  global `random` deliberately poisoned differently before each run); every
  step's `(obs bytes, reward, full game fingerprint)` must match.
* Static guards (AST, not regex, so docstring mentions don't count): no
  `score_hand(` call in `env_*.py` / `card_selection.py` without `rng=`; no
  direct `score_hand` call in `env_*.py` at all; no module-level `import random`
  in `env_*.py`.

**Negative check**: with the three BRL (pre-fix) env files swapped in, 29 of the
40 fail — every side-effect, determinism, reference-play and static test, plus
the `env_sim`/`env_v5` trajectory tests. The `env_v7` trajectory test passes
pre-fix as expected (its bug was mutation, not same-seed divergence). Fixed files
restored and verified byte-identical afterwards.

## 7. Suite counts

`python -m pytest engine/tests -q` from the repo root, 2026-08-21:

| When | Result |
|---|---|
| Before any of my edits (baseline) | 847 passed, 13 skipped |
| Mid-session, my fix in, other agents mid-edit | 29 failed (all in `test_consumables`, `test_shop_v5`, `test_play_env_v5`, `test_joker_catalogue`, `test_consumable_targeting`, `test_game_transitions` — `apply_planet` not levelling, seal names, catalogue keys; `consumables.py`/`shop.py`/`game.py` had been modified within the previous two minutes by P1-rekey). Re-running the same failing set with my env files swapped for the pre-fix copies produced the identical failure set, so none were mine. |
| End of session (other agents' tests now included) | **1279 passed, 13 skipped, 0 failed** (18.5 s); `test_env_rng_isolation.py` alone: 40 passed. |
