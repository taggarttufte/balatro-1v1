# engine — Fork Notes

**Created 2026-08-20 (Phase 0, Agent E).** Self-contained copy of the Balatro simulator
for the multiplayer (MLB) line. Phase 1 threads keyed RNG through *this* copy; the
top-level `balatro_sim/` (BRL) is untouched.

---

## 1. Provenance

| Source | Commit | Branch | What was taken |
|---|---|---|---|
| `C:\Users\Taggart\projects\balatro-rl` | `4411dbf` (2026-07-31) | `fix/sim-fidelity-2026-07` | `balatro_sim/` (engine + `jokers/`), `tests/`, `balatro_sim/tests/` |
| `C:\Users\Taggart\projects\recovered\balatro-mcts` | `ee75d11` (master; `game.py` last touched by `63ef7ca`, 2026-07-30) | `master` | `clone()`, `legal_actions()`, `_consumable_target_actions()`, `JokerInstance.clone()`, three clone tests, `benchmarks/bench_clone_step.py` |

The balatro-rl worktree was clean under `balatro_sim/` and `tests/` at copy time, so the
copy equals the committed tree at `4411dbf`. Nothing outside `engine/` was modified;
no git state was changed.

**Pre-copy diff of the two engine trees** (`diff -rq`, excluding `__pycache__`/tests):
exactly four files differed, as the brief predicted —

| File | Lines differing | Decision |
|---|---|---|
| `game.py` | 257 (all additions) | take the additions from balatro-mcts |
| `jokers/base.py` | 8 (all additions) | take the additions from balatro-mcts |
| `card_selection.py` | 55 | **keep balatro-rl** (post-A1, `N_INTENTS = 4`; mcts is pre-A1 with 3) |
| `env_v7.py` | 85 | **keep balatro-rl** (post-audit `OBS_DIM = 443`; mcts is 434) |

Everything else (`card.py`, `constants.py`, `consumables.py`, `env_mp.py`, `env_sim.py`,
`env_v5.py`, `hand_eval.py`, `mp_game.py`, `quality.py`, `scoring.py`, `shop.py`,
`synergy.py`, all other `jokers/*.py`) is byte-identical between the two repos, so
the fork is byte-identical to *both* for those files.

### Excluded from the copy
`__pycache__/`, `*.pyc`, `balatro_rl/` (live-game IPC), `train_*.py`, `scripts/`, `viz/`,
`mod/`, `mod_v2/`, `legacy/`, `results/`, checkpoints and logs. No data files live under
`balatro_sim/` or `tests/`, so nothing large was in scope.

---

## 2. Layout

```
engine/
├── __init__.py            package marker + orientation docstring
├── pytest.ini             pythonpath = . ; testpaths = tests   (makes engine the rootdir)
├── conftest.py            puts engine first on sys.path; RAISES if balatro_sim resolves elsewhere
├── FORK_NOTES.md          this file
├── balatro_sim/           the engine fork (25 modules incl. jokers/)
├── benchmarks/
│   └── bench_clone_step.py
└── tests/
    ├── engine_tests/      == balatro-rl tests/            (22 files, byte-identical)
    └── sim_tests/         == balatro-rl balatro_sim/tests/ (+ 3 clone tests from balatro-mcts,
                              + one skip marker — see §4)
```

The two source test trees were kept as separate packages (`engine_tests`, `sim_tests`)
so basenames cannot collide and module names cannot clash with the repo-root `tests`
package. `engine/tests/` deliberately has **no** `__init__.py`.

### Running
```
# from the repo root (the intended invocation)
python -m pytest engine/tests -q

# from inside engine (testpaths)
cd engine && python -m pytest -q

# the benchmark
python engine/benchmarks/bench_clone_step.py
```
`pytest-timeout` is not installed in this environment, so the suite was run without
`--timeout`; the full suite takes ~7–13 s.

### Why both `pytest.ini` and `conftest.py`
pytest only loads `conftest.py` files at or below `confcutdir`, which defaults to the
rootdir — and with no ini file the rootdir would be the *common ancestor of the
arguments* (`engine/tests`, or a single test file's directory). `engine/pytest.ini`
pins the rootdir to `engine` for any invocation targeting a path under it, so the
conftest always loads and `pythonpath = .` puts the fork first on `sys.path`. The conftest
then re-asserts the path ordering and **fails collection with a `RuntimeError`** if
`balatro_sim.__file__` is not the fork's — verified: pre-importing the BRL package and
then invoking pytest on the fork's tests exits with code 4 and the "imported the wrong
balatro_sim" message, instead of silently testing the wrong engine.

Sibling invocations (`pytest tests`, `pytest mp`) do not see `engine/pytest.ini`
because ini discovery walks *upward* from the arguments only.

---

## 3. What was ported from balatro-mcts

Applied as patches (`diff -u` between the two repos' files, `patch --binary` because the
whole tree is CRLF) so the port is exact. Post-patch, `engine/balatro_sim/game.py` and
`engine/balatro_sim/jokers/base.py` are **byte-identical to the balatro-mcts files**.

| Symbol | Fork location | Notes |
|---|---|---|
| `BalatroGame._bare_ctx = _hook_ctx` | `game.py:287` | alias kept for MCTS-side callers |
| `BalatroGame.clone()` | `game.py:291-389` | `__new__` + field-by-field copy; RNG via `getstate`/`setstate` |
| `BalatroGame.legal_actions()` | `game.py:662-763` | combinatorial play/discard subsets, shop/booster/consumable actions |
| `BalatroGame._consumable_target_actions()` | `game.py:765-812` | per-consumable target enumeration |
| `JokerInstance.clone()` | `jokers/base.py:153-159` | copies `key`, `edition`, `state.copy()` |

**Deck-aliasing invariant preserved.** `full_deck` is the permanent collection and
`deck`/`hand`/`discard_pile` hold references into it. `clone()` copies `full_deck` once
and rebuilds the three partitions as aliases keyed by the *original* card `id`
(`Card.copy()` mints a fresh id, so the map is built from the originals). Pinned by
`tests/sim_tests/test_clone_deck_identity.py` (18 tests).

**Coverage check of `clone()` against the balatro-rl `__init__`:** every instance
attribute assigned anywhere in `game.py` (41, including annotated assignments) is copied
by `clone()`, and `clone()` copies nothing that `game.py` does not assign. Attributes
written onto the game from other modules (`consumables.py`, `shop.py`, `env_mp.py`,
`env_v5.py`) are all members of that same set.

**Ad-hoc agreement check, `legal_actions()` vs `step()`:** walking 12 seeds with a
play-largest-subset policy and stepping every enumerated action on a clone — 93 states
(`BLIND_SELECT` ×20, `SHOP` ×25, `SELECTING_HAND` ×48), 3,022 clone-steps, **zero
exceptions**. `BOOSTER_OPEN` could not be reached — see §7, item 0.

---

## 4. Test adjustments

Exactly one source test was touched:

- `tests/sim_tests/test_edge_cases.py` — `class TestActionMasking` (10 tests, not 8 as the
  brief estimated) now carries `@pytest.mark.skip(reason=...)`. Its helper does
  `from train_sim import get_action_mask`; `train_sim.py` is the top-level BRL training
  script (imports torch) and is not part of the fork. Kept rather than deleted, per the
  brief. If an engine-local action mask lands, remove the decorator. The edit was done
  byte-preserving (CRLF intact); `diff` against the source shows only the 5 decorator
  lines.

Added (not adjustments): `tests/sim_tests/test_clone.py` (9 tests),
`test_clone_deck_identity.py` (18), `test_clone_stochastic.py` (5) — copied unchanged from
balatro-mcts. The other mcts test files (`test_gumbel.py`, `test_nn_policy.py`,
`test_train.py`) import `mcts`/`train` and are not engine tests; not brought over.

---

## 5. Test counts

| Where | Result | Notes |
|---|---|---|
| Source, in place (`balatro-rl`, `python -m pytest tests balatro_sim/tests`) | **825 passed, 3 skipped** (9.35 s) | run with `-p no:cacheprovider` and `PYTHONDONTWRITEBYTECODE=1` so nothing outside `engine` was written |
| Fork (`python -m pytest engine/tests -q -x` from repo root) | **847 passed, 13 skipped, 0 failed** (13.35 s) | |
| Fork, from inside `engine` (`python -m pytest -q`) | 847 passed, 13 skipped | same |

Reconciliation: 825 − 10 (`TestActionMasking`, now skipped) + 32 (three clone files) = 847.
Skips: 3 pre-existing runtime skips in `engine_tests/test_rewards_v5.py:62` ("Could not
reach shop" — identical in the source run) + 10 `TestActionMasking` = 13.

---

## 6. Benchmark (`benchmarks/bench_clone_step.py`, 3 runs, Python 3.13.5, RTX-3080-Ti box)

| Metric | Run 1 | Run 2 | Run 3 | Previously measured (brief) |
|---|---|---|---|---|
| `game.clone()` | 62.9 µs → **15,910 /s** | 62.6 µs → 15,965 /s | 63.2 µs → 15,821 /s | ~16k /s |
| `step()` (random games) | 27.5 µs → **36,413 /s** | 28.1 µs → 35,528 /s | 27.2 µs → 36,749 /s | ~34k /s |
| `copy.deepcopy` | 428 µs → 2,335 /s | 415 µs → 2,413 /s | 391 µs → 2,556 /s | — |
| clone speedup vs deepcopy | 6.8× | 6.6× | 6.2× | — |

Mid-game state used for cloning: `SELECTING_HAND`, ante 4, 5 jokers, 44 cards in deck.
Caveat inherited from the benchmark itself: the step timing comes from only ~180–190
steps (random agents lose in the first blind), so it is noisier than the clone number.

---

## 7. Known engine-level fidelity issues this fork INHERITS (not fixed here — Phase 1)

Authoritative inventory: **`docs/MP_UPDATE_LIST_2026-08.md` §1–§7.** Line numbers
there refer to the source `game.py`; the port shifted `game.py` lines (+105 after line
285, +257 after line 556). All other modules are unshifted. Fork-relative pointers:

**0. Found while building this fork (not in the update list):** `State.BOOSTER_OPEN` is
never entered. `shop.buy_item` → `_open_booster` (`shop.py:385-410`) fills
`booster_choices`/`booster_picks_remaining` but leaves `game.state == SHOP`; nothing in
the engine assigns `State.BOOSTER_OPEN`. Consequences through the raw `step()` API:
a booster purchase debits money and grants nothing (`pick_booster` is only handled in
the `BOOSTER_OPEN` branch, `game.py:653-657`, so it is silently ignored in `SHOP`), the
stale choices stay on the game object, and `legal_actions()`'s `BOOSTER_OPEN` branch
(`game.py:751-761`) is dead code. `env_v5` sidesteps it with its own pack substate
(`env_v5.py:503-511`); `env_v7`'s only handling is in `_auto_advance`
(`env_v7.py:844-848`) on a state that never occurs. Verified by buying `p_celestial_mega`
via `step()`: dollars 100→92, state still `SHOP`, 5 pending choices, consumables unchanged.

**1. Single RNG stream — P0.** `game.py:184` `self.rng = random.Random(seed)` is the
entire RNG architecture; no keys, no per-key state, no resample counter. Every draw
shifts every later draw, so same-seed multiplayer is architecturally impossible.
Variable-cost draws (`.choice`/`.shuffle`/`.randint`) and conditional draws make it
worse. ~40 call sites to key (update list §1.2); in the fork `game.py` the sites are
`:406` (boss pick), `:531` (joker shuffle), `:555` (deck shuffle), `:839`, `:946` (glass
1/4), `:993` (Purple Seal tarot); `shop.py`, `consumables.py`, `jokers/*` unchanged.
`clone()` copies the stream by `getstate`/`setstate`, which is correct for the current
single-stream design and will need to copy the per-key dict once §1 lands.

**2. No shop queue — P0.** `shop.py:229 generate_shop` draws i.i.d.; `shop.py:367
reroll_shop` regenerates from scratch. No per-ante queue, no pointer, no The-Order
switch.

**3. Reproducibility breaks — P0.** `env_v7.py:565` — `_best_hand_score`
(`env_v7.py:513-575`, called at `:261`) scores up to 8 hypothetical hands with the
*live* `gs.rng`, advancing the stream before the real play. `env_sim.py:619` /
`env_v5.py:847` call `score_hand` without `rng=` and fall through to the process-global
`random`. Latent `None` contexts at `consumables.py:250`, `shop.py:363`; module-`random`
defaults at `shop.py:179`, `shop.py:288`.

**4. MLB ruleset absent — P1.** No nemesis blind, lives, comeback money, PvP boss slot,
early-win rule, or lockstep coordinator. `mp_game.py` / `env_mp.py` are pre-fix
scaffolding.

**5. Decks / stakes / stickers — P1.** Only a Red-equivalent standard deck exists; no
stakes, no Eternal/Perishable/Rental.

**6. Tags — P0 for strategy fidelity.** `_skip_blind` (`game.py:1092-1103`) pays a flat
`+$5`; no tag pool/object/stream. `legal_actions()` offers `skip_blind` on non-boss
blinds, so a searcher sees the flat reward, not the real tag economy.

**7. Correctness bugs (update list §7) — P2.** Wrong effects: 8 Ball (`chips.py:80-85`
wins over the correct `misc.py:251-255` by `jokers/__init__.py` import order), Gros
Michel/Cavendish never destroyed, The Idol never randomised initially, Lucky Cat never
scales (`on_lucky_trigger` has no callers). Nine jokers emit sentinel strings that
`_use_consumable` (`game.py:1001-1015`) cannot resolve (Vagabond, Superposition,
Cartomancer, Seance, Sixth Sense, Riff-Raff, Certificate, DNA, Perkeo). Six hooks never
invoked; three `on_init` bodies would `NameError`. Boss blinds have no exhaustion pool
(`game.py:399-407`); vouchers have no ante gating (`shop.py:281-284`); Hone/Glow Up are
no-ops; booster packs allow duplicates and standard packs are vanilla; editions only
roll for shop jokers; Glass retrigger rolls once per card; Wheel of Fortune targets
editioned jokers; rarity weights 123/18/10 vs real 61/64/20; six jokers double-listed.

None of the above is changed in this fork. The 847-test suite passes with every one of
them present, exactly as the 828-test source suite does.

---

## 8. Not done / handed to Phase 1

- No keyed RNG, no `pseudoseed`, no shop queue — by design (Phase 0 gate must pass first).
- `legal_actions()` semantics are inherited as-is from balatro-mcts (e.g. `ROUND_EVAL`
  is represented by a dummy `{"type": "advance"}`; psychic boss restricts plays to
  exactly 5 cards; tarot targets are enumerated for sizes 0–2 regardless of the tarot's
  real target count). Not re-validated beyond the agreement check in §3.
- `clone()` will need updating once per-key RNG state exists (see §7 item 1).
- `TestActionMasking` stays skipped until the fork has its own action mask.
