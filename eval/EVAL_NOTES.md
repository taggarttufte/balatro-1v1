# EVAL_NOTES — Phase 3 W4: eval harness + ρ-decay harness (2026-08-21/22)

**Agent W4.** Files: `eval/common.py` (shared bootstrap / drivers / stats), `eval/eval_harness.py`,
`eval/rho_decay.py`, `eval/conftest.py`, `eval/tests/` (49 tests), `results/*.json` (outputs),
this note. `engine/**` and `rng/**` were only read, never edited. `mlb_match_demo.ScriptedPlayer` /
`make_policy` / `greedy_hand` / `weakest_play` / `shelf_indices` are imported from `scripts/`, not copied.

## 0. Gates (final run, repo root, `python` = 3.13)

| gate | result |
|---|---|
| `python -m pytest eval/tests -q` | **49 passed** (~30s) |
| `python -m pytest engine/tests -q` | **1609 passed / 10 skipped / 3 xfailed** — unchanged |
| `python -m pytest tests -q` | **1073 passed / 2 xfailed** — unchanged |
| `python -m eval.eval_harness --mode 1v1 --player ... --reference ... --out results/demo_1v1.json` | OK, 126 seeds, 21.7s |
| `python -m eval.rho_decay --all --n-extra-seeds 24 --horizons 1,2,4,8 --n-boot 2000 --out-dir results` | OK, 150 seeds × 3 perturbations, ~100s each |

## 1. How to run

```
# eval harness (repo root)
python -m eval.eval_harness --mode sp_vanilla --player "scripted:hand=greedy,buy=1,pack=0" \
    --out results/my_report.json
python -m eval.eval_harness --mode sp_mlb --player "scripted:hand=greedy,reroll=1,buy=1" \
    --max-antes 8 --target-k 1.0 --out results/my_sp_mlb.json
python -m eval.eval_harness --mode 1v1 --player "scripted:hand=greedy,reroll=1,buy=1" \
    --reference "scripted:hand=greedy,buy=1" --out results/my_1v1.json
python -m eval.eval_harness --compare results/a.json results/b.json --out results/cmp.json

# ρ-decay (repo root)
python -m eval.rho_decay --perturbation buy_slot0 --horizons 1,2,4,8 --out results/rho_decay_buy_slot0.json
python -m eval.rho_decay --all --n-extra-seeds 24 --out-dir results   # all 3 perturbations, 150 seeds
python -m eval.rho_decay --list-perturbations

# tests
python -m pytest eval/tests -q
```

`--player` / `--reference` take `scripted:<field>=<value>,...` (aliases: `reroll`→`rerolls_per_visit`,
`buy`→`buy_slot0`, `pack`→`open_pack_slot`, `voucher`→`buy_voucher`, `weak_from`→`weak_from_ante`; other
`ScriptedPlayer` fields by their own name: `hand`, `debug_win_regular`, `rich`, `pick_from_pack`). `--player
checkpoint:<path>` raises `NotImplementedError` documenting the interface (§5).

## 2. Report schema (eval_harness.py)

JSON: `mode`, `player`, `reference` (1v1 only), `deck`/`stake`/`lives`/`max_antes`, `target_fn` (sp_mlb only),
`seeds`, `n_seeds`, `wall_clock_s`, `per_seed` (one record per seed — see driver return shapes in `common.py`:
`play_sp_vanilla` / `play_sp_mlb` / `play_1v1`), `summary` (every numeric per-seed field, `bootstrap_ci`'d over
seeds, plus `win_rate`). `--compare a.json b.json` pairs `per_seed` records by seed value (drops any seed
missing from either side, reported in `dropped_seeds`) and reports `paired_bootstrap_ci(A_field, B_field)` for
every numeric field both reports share — this is what makes common random numbers pay off: two evaluations of
the SAME player are byte-for-byte identical per seed, so the diff CI is `[0, 0]` exactly (verified by test, not
assumed).

## 3. The default SP-MLB target, and why

SP-MLB solo mode needs an opponent, but there is no second live game. Default (`common.own_big_blind_target`,
`k=1.0`): **the target at ante *a*'s Nemesis is the agent's own chip score on ante *a*'s Big Blind**, captured
live as the agent plays it (`play_sp_mlb` fills in `big_blind[ante]` the moment that Big Blind cashes out,
strictly before that ante's Nemesis is reached). Rationale:

- **Calibration-free.** No historical corpus, no hand-tuned difficulty curve, no second checkpoint needed —
  the number comes from the agent's own play, which is measurable in any harness state.
- **Genuinely ~50/50 by construction**, matching the actual design intent of an MLB Nemesis (design doc §5:
  "every nemesis blind is ~50/50 by construction") — a fixed player that plays consistently will score
  similarly hand-to-hand, so the "mirror" target neither trivially wins nor loses.
  A player that improves ante-to-ante (buys jokers, builds an economy) faces a target that improves with it.
- **Reversible.** `--target-k` scales it (k>1 = a harder mirror, k<1 = easier); a fixed per-ante table is also
  pluggable (`target_fn(game, big_blind) -> int`, `common.py`) once a real reference distribution exists.

`rho_decay.py` uses a DIFFERENT target (`external_vanilla_big_blind_target`) on purpose — see §4.

## 4. ρ-decay design

Paired arms A/B on ONE seed (`BalatroGame(seed, ruleset="mlb")` × 2), driven by the IDENTICAL "forward" policy
(`rho_decay.BASE_SPEC`: greedy hand, `buy_slot0=True` — buys shelf slot 0 whenever affordable, every shop, for
the rest of the game) **except for ONE decision at ante 1** (the perturbation), then both play forward
identically (`rho_decay._wrap_perturbation`: a stateful wrapper that overrides exactly the first matching
decision, then falls through to the shared base policy for everything before AND after — this is what makes it
"diverge at one decision," not a persistently different policy).

Registered perturbations (`rho_decay.PERTURBATIONS`):

| name | description |
|---|---|
| `buy_slot0` (default) | ante-1 first shop: arm A buys shelf slot 0 if affordable (matches `BASE_SPEC`'s own default), arm B deliberately does not |
| `reroll_once` | ante-1 first shop: arm B rerolls once (redraws BOTH shelf slots) before proceeding, arm A does not |
| `skip_small` | ante-1: arm B skips the Small blind entirely (no shop visit, no cash-out payout for it), arm A plays it normally |

Outcomes at horizon *h* ∈ {1,2,4,8} = ante `1+h`'s Nemesis, played to hand-exhaustion (`common.play_arm_to_horizons`
— the Nemesis never ends early under MLB regardless of target, by construction: `game.py`'s `is_pvp` branch in
`_play_hand` never checks `chips_scored >= target`, only `hands_left <= 0`). Three outcome variables per horizon:
`log_score` (`log1p(own chips_scored)`), `money` (dollars at the following Cash Out), `lives_lost` (cumulative,
tracked via the OBSERVED `game.lives` delta, not just this driver's own life-loss decisions — see §6 finding 2).
Target function: `external_vanilla_big_blind_target` — the VANILLA Big-Blind chip requirement for that
ante/deck/stake, a pure function of (ante, deck, stake) that is coupled to **neither** arm, so a life-loss
correlation between arms isn't an artifact of sharing a target. Both arms use `lives=999` so a life loss never
truncates a run before every horizon is reached (§6 finding 1 explains why this matters).

ρ(h) = Pearson **and** Spearman correlation of `(outcome_A, outcome_B)` over seeds, each with a 2000-resample
percentile bootstrap CI (`common.bootstrap_corr_ci`). The **paired-vs-unpaired variance-reduction factor**
is `Var(A−B unpaired) / Var(A−B paired)`; the unpaired variance is estimated by randomly RE-PAIRING the same
seeds' already-collected outcomes (`common.unpaired_control_variance`: 500 random re-pairings, i.e. literally
"arm B on a different seed," reusing runs already paid for — statistically equivalent to fresh unpaired runs
since seeds are exchangeable, at zero extra simulation cost). This is cross-checked against the closed form
`1/(1−ρ)` (equal-variance approximation) in `common.sample_size_per_arm`, used for §5's table — the two methods
agree to within ~5% throughout (see raw JSON `variance_reduction_factor` per horizon vs. the analytic table).

## 5. ρ(h) — measured (N=150: 126 ground-truth seeds + 24 synthetic, `deck=b_red`, `stake=1`, n_boot=2000)

All values below are the **Pearson point estimate [95% bootstrap CI]** for `log_score`, `n` = seeds that
reached that horizon (some runs hit the found engine gap, §6 finding 3), and VRF = the measured
paired-vs-unpaired variance-reduction factor (permutation method, §4).

| perturbation | h=1 | h=2 | h=4 | h=8 |
|---|---|---|---|---|
| **buy_slot0** (buy 1 item) | 0.876 [0.797,0.933] n=150 VRF=8.00× | 0.877 [0.803,0.931] n=150 VRF=8.16× | 0.897 [0.851,0.932] n=150 VRF=9.61× | 0.870 [0.804,0.923] n=150 VRF=7.69× |
| **skip_small** (skip a blind) | 0.805 [0.717,0.881] n=150 VRF=5.10× | 0.774 [0.675,0.859] n=150 VRF=4.46× | 0.772 [0.674,0.844] n=150 VRF=4.37× | 0.772 [0.674,0.852] n=150 VRF=4.40× |
| **reroll_once** (redraw shelf) | 0.606 [0.445,0.739] n=149 VRF=2.53× | 0.669 [0.523,0.788] n=149 VRF=3.02× | 0.715 [0.608,0.798] n=149 VRF=3.50× | 0.728 [0.615,0.813] n=149 VRF=3.69× |

Spearman tracks Pearson within ~0.02–0.05 at every cell (both in the raw JSON; not separately tabulated here).
`money` and `lives_lost` correlations are in `results/rho_decay_<perturbation>.json` per horizon
(`per_horizon.<h>.metrics.{money,lives_lost}`); `money` starts lower than `log_score` at h=1 (the perturbation's
direct effect) and converges toward `log_score`'s level by h=8 as continued shared buying re-absorbs the
initial gap.

**Ordering (consistent at every horizon): buy_slot0 (least decorrelating) > skip_small > reroll_once (most
decorrelating).** This is the `test_rho_lower_for_a_larger_perturbation` monotonicity the brief asks for,
confirmed on the full 150-seed run, not just the 24-seed test subset.

### Sample sizes this implies (d = standardized effect size, `common.sample_size_per_arm`, α=0.05, power=0.8)

| perturbation | h | ρ | n/arm, unpaired (d=0.5) | n/arm, paired (d=0.5) |
|---|---|---|---|---|
| buy_slot0 | 1 | 0.876 | 63 | 8 |
| buy_slot0 | 8 | 0.870 | 63 | 9 |
| skip_small | 1 | 0.805 | 63 | 13 |
| skip_small | 8 | 0.772 | 63 | 15 |
| reroll_once | 1 | 0.606 | 63 | 25 |
| reroll_once | 8 | 0.728 | 63 | 18 |

(d=0.2, a small effect: n_unpaired=393 across the board; n_paired scales by the same `1−ρ` factor, e.g. 47-98.)

## 6. What changed vs. the design doc's guesses

`MP_TRAINING_DESIGN_2026-08.md` §1 guessed ρ **falls from ~0.9 at the decision point to ~0.3–0.5 at h=4–8**,
worth "~1.4–2×" variance reduction at long horizons (vs. 3–10× guessed for short ones). The measurement
disagrees on both counts, for these three perturbations:

1. **ρ does not decay toward 0.3–0.5 — it stays high (0.6–0.9) and roughly FLAT across h=1..8**, in two cases
   (`buy_slot0`, `reroll_once`) even drifting slightly *upward* from h=1 to h=8 rather than down. Only
   `skip_small` shows a (small, ~0.03) dip and then plateaus.
2. **The variance-reduction factor stays in the 3–10× range at h=8**, not the guessed 1.4–2×. `buy_slot0`
   is even ABOVE the guessed short-horizon ceiling (8× at h=8 vs. "3–10× only for short horizons" guessed).

**Why:** the design doc's own reasoning ("still drawing from the same underlying queue... same blind sizes and
boss sequence") is directionally right, but understates how strong the shared-queue effect is for THESE
scripted players, because of a stronger, more specific mechanism: **the `shuffle` RNG stream (which cards are
dealt) is provably independent of the shop/pack/voucher streams** (Phase 1 stream-independence invariant,
`GATE_NOTES.md` / `MLB_NOTES.md`). `greedy_hand`'s score on a given ante's cards is therefore identical
between arms UNLESS a joker/consumable the arms disagree on owning actually changes how that hand scores.
Since jokers are only a *multiplicative layer on top of* a base hand value that both arms compute identically
(same cards, same greedy algorithm), and this base value dominates the variance across seeds (different seeds
deal very different hands; a modest joker collection difference is a smaller share of total score variance for
these early antes), ρ stays high far longer than the design doc guessed. (Confirmed directly: with a
non-buying base policy, where jokers can NEVER differ between arms, ρ(h) = 1.000 exactly at every horizon and
every seed tested — see `test_rho_is_one_with_no_perturbation` and the `"none"` perturbation option in
`rho_decay.py`; the real perturbations only decorrelate anything BECAUSE `BASE_SPEC.buy_slot0=True` lets money
differences turn into different joker portfolios over time.)

**Caveat this puts on the whole measurement:** `greedy_hand` + "buy whatever's affordable in slot 0" is a weak,
economically indifferent policy (it does not target synergistic jokers, so most of what it buys barely changes
its own scoring). A trained or stronger scripted policy that actually builds a scaling engine around specific
jokers would plausibly show FASTER divergence (a missed/gained key joker early can compound multiplicatively
by ante 8-9) — i.e., this measurement is likely an **upper bound on ρ** / **lower bound on variance-reduction
need**, not a floor. This is exactly the caveat the design doc predicted matters (§1: "small differences lead
to different joker lineups... which decouples ρ") — it is real, just weaker for these specific players than
guessed, and probably understated for a policy that actually cares which joker it buys.

## 7. Needs engine change (found, not fixed)

**`engine/balatro_sim/game.py:1546-1552, 1571-1580, 1582-1592`** (`bl_hook`, `bl_eye`, `bl_mouth` boss-ability
rejection branches inside `_play_hand`). Each sets `self.state = State.GAME_OVER` **unconditionally** on hand
exhaustion (`if self.hands_left <= 0: self.state = State.GAME_OVER`), several lines BEFORE the main scoring
path's mlb-aware branch at line ~1712-1717 (`elif self.hands_left <= 0: if self.mlb: self._mlb_fail_round()
else: self.state = State.GAME_OVER`). Under `ruleset="mlb"`, if the current blind's boss is one of these three
AND a hand is exhausted via THAT boss's rejection (repeated hand type for `bl_eye`, off-type for `bl_mouth`, or
an empty selection after `bl_hook` removes cards), the run ends immediately regardless of `self.lives` — MLB's
"a failed blind proceeds and costs (at most) one life" rule (`MLB_NOTES.md` 1.3d) is bypassed entirely for
these three bosses. Found via `rho_decay`'s synthetic seed `BP49PU2Y` (ante-1 boss = `bl_hook`): both
`play_sp_mlb` and `play_arm_to_horizons` correctly tolerate this (their loops already only check
`game.state != State.GAME_OVER`, and `measure_rho` already drops any (seed, arm) pair that never reaches a
horizon — confirmed as `n=149` vs. `n=150` for `reroll_once` in §5, the one perturbation where this particular
seed's random reroll landed the run on one of these three bosses before ante 2). `play_sp_mlb`'s return now
also carries `ended_early_engine_gap: bool` (`game.state == GAME_OVER and game.lives > 0` at return time) so a
report can distinguish this from a genuine 0-lives loss. **Not fixed** — `engine` is frozen for Phase 3;
this is a work-around, not a patch. Affected 1/150 seeds in the real run (0.7%); harmless to the measurement
because it is excluded, not silently miscounted.

### 7b. `targets.py`'s "engine-only deps" test was a false pass (found 2026-08-26, repo split)

`test_targets_module_avoids_heavy_mp_eval_imports_when_imported_alone` asserted that importing
`targets.py` alone never pulls in `['mlb_match_demo', 'oracle.parity_check', 'rng.generate',
'rng.pools', 'torch']`. The last four names were checked, but the first three of those never
*could* match: while this project lived under `mp/`, the rng modules were imported as
`mp.rng.generate` / `mp.rng.pools`, so the literal strings `rng.generate` / `rng.pools` were
absent from `sys.modules` no matter what. Promoting the packages to the repo root renamed them
and the assertion failed for the first time.

**The underlying dependency is unchanged and was always there:** `engine/balatro_sim/game_keys.py`
derives every key/name/rarity/cost table from `pools` at import time, so `rng.pools` (and, via
`_load_gen`, `rng.generate`) is pulled in by *any* import of the engine fork — `targets.py`
cannot avoid it without avoiding `balatro_sim`. Both are pure-Python data modules: no torch, no
numpy. The claim that actually matters — no `mlb_match_demo`, no `oracle.parity_check`, no
`torch` — **still holds and is verified** (re-checked from the new root: heavy hits `[]`).

**Not fixed:** the test now names only the genuinely-heavy modules and asserts the transitive
`rng.pools` import *explicitly*, so the real dependency stays visible rather than being deleted
from the list. Whether `targets.py` should be made engine-table-free is a design question for
the owner, not a mechanical repo-split change.

## 8. Caveats

- **Scripted players, not trained ones.** Every number in §5 is for `greedy_hand` + fixed shopping rules
  (`BASE_SPEC` / `mlb_match_demo.ScriptedPlayer`), not a checkpointed agent — see §6's caveat on what a
  stronger/trained policy would likely change.
- **SP-MLB solo mode has no live opponent** — its target is a proxy (§3), not a measurement of actual match
  win probability. Use `--mode 1v1` for a faithful head-to-head number.
- **rho_decay uses `lives=999`** so life loss never truncates a run before every horizon is reached; this
  means `cum_lives_lost` in the raw per-seed JSON can exceed what a real `lives=4` MLB run would ever show
  (a real match would have ended). Treat it as "how many Nemeses would have been lost," not a literal life
  count under real rules.
- **`external_vanilla_big_blind_target`** (rho_decay's target) is a reasonable, cheap, decoupled proxy for
  Nemesis difficulty, not a calibrated one — it does not scale with deck/stake difficulty the way a live
  opponent's actual score would.
- The engine gap in §7 means a small fraction of very long (many-ante) synthetic-seed runs can end earlier
  than expected for reasons unrelated to lives; always filter on `ended_early_engine_gap` /
  missing-horizon (`None`) rather than assuming every run reaches `max_antes` or every horizon.
- Bootstrap CIs use `random.Random(seed)` reseeded per call (`ci_seed`, default 0) — reproducible across runs,
  not a substitute for more resamples if you need a tighter interval (`--n-boot`).

## 9. What W1's checkpoint player must provide

`eval.common.Player`: an object with `.act(game: balatro_sim.game.BalatroGame) -> dict` (one of
`game.legal_actions()`). Wire it into `eval.common.parse_player_spec` (currently `"checkpoint:<path>"` raises
`NotImplementedError` with this exact interface in the message) so `--player checkpoint:<path>` resolves to a
real player; `adapt_player(player)` already converts any such object into the `(match_or_shim, player_idx,
legal_actions) -> action` signature every driver in `common.py` and `MLBMatch.play_out` expect — no other
change needed in `eval_harness.py` / `rho_decay.py` once that one function is filled in. MLB awareness (a
checkpoint player must handle `ruleset="mlb"` including `PVP_WAIT`) is W1's concern per the Phase 3 brief;
nothing in this harness assumes the player is a `ScriptedPlayer` beyond that one dispatch point.

## 10. File map

- `eval/common.py` — bootstrap (fork-guarded import of `engine`), `Player` protocol + spec parsing,
  `SoloShim` (drives a lone `BalatroGame` through `mlb_match_demo`'s policy signature), `play_sp_vanilla` /
  `play_sp_mlb` / `play_1v1` drivers, `play_arm_to_horizons` (paired-arm horizon driver for rho_decay), target
  functions, pure-python bootstrap/correlation statistics, `sample_size_per_arm`.
- `eval/eval_harness.py` — CLI + `evaluate()` / `compare()`.
- `eval/rho_decay.py` — CLI + `PERTURBATIONS`, `_wrap_perturbation`, `make_perturbed_game`, `measure_rho`.
- `eval/conftest.py` — fork-guard bootstrap for `eval/tests`.
- `eval/tests/test_common.py`, `test_eval_harness.py`, `test_rho_decay.py` — 49 tests.
- `results/demo_1v1.json`, `rho_decay_buy_slot0.json`, `rho_decay_reroll_once.json`,
  `rho_decay_skip_small.json` — the real runs behind §5.

---

# Phase 4 — W4: transfer-spread harness + external Nemesis targets (2026-08-22)

**Agent W4.** New: `eval/targets.py`, `eval/transfer_spread.py`, `eval/tests/test_targets.py` (57
tests), `eval/tests/test_transfer_spread.py` (19 tests). `results/transfer_spread_{greedy,
greedy_reroll1_buy1,weak}.{json,md}` (the real runs behind S15). `engine/**`, `rng/**`, `agent/**`,
`tournament/**`, `replay/**` only read, never edited (frozen for this agent per the brief). The `W2 owns
tournament/runner.py cleanup` item from the Phase 4 brief is NOT this agent's — confirmed untouched.

## 11. Gates (final run, repo root, `python` = 3.13)

| gate | result |
|---|---|
| `python -m pytest eval/tests -q` | **125 passed** (49 Phase-3 + 76 new: 57 targets + 19 transfer_spread) |
| `python -m pytest engine/tests -q` | **1614 passed / 10 skipped / 3 xfailed** — unchanged |
| `python -m pytest tests -q` | **1073 passed / 2 xfailed** — unchanged |
| `python -m eval.transfer_spread --player ... --mode both --out results/transfer_spread_<name>.json` | ran for real, 3 player specs, ~5-7 min each (below) |

## 12. `targets.py` — API (for W2's `--objective external` / `agent`)

Engine-only deps (`balatro_sim.constants.blind_base_chips`, `.decks.deck_spec`, `.stakes.stake_spec` — no
`mlb_match_demo`, no `oracle.parity_check`, no `rng.generate`/`rng.pools`, no torch; verified by a
subprocess-isolated test, `test_targets_module_avoids_heavy_mp_eval_imports_when_imported_alone`, that imports
`targets.py` alone in a fresh interpreter and asserts none of those modules ever entered `sys.modules`). Its own
fork-guard against the repo-root BRL `balatro_sim` is reimplemented locally (not a call to
`oracle.engine_parity.import_engine()`) for exactly that reason.

Every target function shares ONE call signature — `target_fn(game, big_blind=None) -> int` — so it drops into
`eval/common.py`'s `play_sp_mlb(target_fn=...)` / `play_arm_to_horizons(target_fn=...)` and W2's training-loop
Nemesis hook (or `mcts.outcome.ExternalOutcome.from_margin`, which is already the exact hook this was built for
— see S14) unmodified:

- **`vanilla_boss_target(ante, deck_key="b_red", stake=1) -> int`** — the chip requirement a vanilla (non-Nemesis)
  Boss blind would have had, i.e. `int(blind_base_chips(ante, 2, blind_scaling) * ante_scaling)`
  (`game.py:642`'s own composition), BEFORE any boss-specific `BOSS_CHIP_MULT` (which needs a seed's actual boss
  draw this function never has — documented gap, `bl_needle` 0.5x is the only live case since `bl_wall`/
  `bl_final_vessel` are `MLB_BANNED_BLINDS`). Plasma's `ante_scaling=2` and a stake's `blind_scaling` both compose
  automatically; `ante > 8` falls through to `blind_base_chips`'s own endless formula. Pinned against the direct
  formula for ante 1-12 x 3 decks x White, against a LIVE vanilla boss's `chips_target` on 30 ground-truth seeds,
  and `vanilla_boss_target(a, "b_plasma", 1) == 2 * vanilla_boss_target(a, "b_red", 1)` for every ante 1-12.
- **`vanilla_boss_target_fn(deck_key=None, stake=None) -> Callable`** — adapter to the shared `(game, big_blind)`
  signature; reads `ante`/`deck_key`/`stake` off the live `game` unless overridden. **This is the function to
  register as the Nemesis's `chips_target`** (module docstring): `game.set_pvp_info(vanilla_boss_target_fn()
  (game, big_blind), 0)` so a solo agent that skips both regular blinds and builds nothing still risks a life at
  the Nemesis (the 07:35 overnight finding this whole module exists to fix).
- **`scaled_own_big_blind(k=1.0) -> Callable`** — W4-Phase-3's "mirror Nemesis" (`eval/common.py::
  own_big_blind_target`), DUPLICATED here (not imported — that would pull `common.py`'s heavy chain into
  `agent`) with zero extra deps; numerically identical to `common.own_big_blind_target` for the same input
  (pinned by test).
- **`table_target(path, quantile=0.5, fallback="nearest_below") -> Callable`** — reads a tournament run's
  per-ante score DISTRIBUTION off `summary.jsonl` (`tournament.matrix.write_run`'s format — `tournament/
  runs/*/` or any directory holding one, e.g. under `results/`); median by default, any of
  `score_distribution`'s quantiles (0.0/0.1/0.25/0.5/0.75/0.9/1.0) configurable. An ante missing from the table
  (a run that never reached it) falls back to the nearest tabulated ante <= the requested one (a conservative
  UNDER-estimate, since targets only grow with ante) unless `fallback="error"`. Exercised end-to-end against a
  real `Tournament(..., out_dir=...)` run in tests.
- **`get_target(name, **kw) -> Callable`** — tiny registry: `"vanilla_boss"` / `"own_big_blind"` / `"table"`.

## 13. `transfer_spread.py` — design + real numbers

Evaluates ONE player spec, paired by seed, across Red / Checkered / Plasma at White stake, in two modes:

**(a) SP-MLB-solo** (`play_sp_mlb` against `targets.vanilla_boss_target_fn()` — deliberately NOT
`own_big_blind_target`, since a target coupled to the agent's own play can't detect "skipped everything, built
nothing" per decision 4). Reports furthest ante, lives lost, win rate (fraction of Nemesis rounds with
`score >= target`) and per-Nemesis margin quantiles, pooled across seeds AND antes within a cell. Run over all
126 ground-truth seeds + 24 synthetic ones (`rho_decay.make_extra_seeds`) — cheap, ~1-2s/deck.

**(b) Tournament** (`Tournament(n=32, life_rule="none", max_ante=8)`): the evaluated player at population index
0 + `tournament.players.default_population(31, base_seed=0)` (an IDENTICAL background population across all
three decks — only the deck changes between cells), population RANK (`matrix.population_rank`) read off at
every ante 2-8 (`life_rule="none"`'s lives sentinel guarantees presence), reported as `rank_frac =
(rank-1)/(n_agents-1)` in [0,1], 0=best. Much more expensive (~5-7s/seed/deck on this machine), so it defaults
to a 16-seed subset of the same seed list — a deliberate compute trade, not an oversight (full-126 tournament
runs across 3 decks would be ~35 min; feasible if a future run wants it, `--tournament-n-seeds`).

**Cross-cell spread**: for each of `win_rate` (mode a) and `rank_frac` (mode b), the three cells' point means
plus a bootstrap CI on the SPREAD (range and population variance) — the CI resamples the shared PAIRED seed set
(same resampled indices applied to every cell each replicate) and recomputes each cell's mean + the resulting
range/variance per replicate, so it answers "does the measured spread reflect a real deck effect or seed
noise," not just "what is the spread." Sanity-pinned by test: a player evaluated against ITSELF with all three
"decks" wired to the same deck key gives EXACTLY zero spread at every bootstrap replicate (identical arrays
resampled with identical indices are trivially equal) — `range_ci == {lo: 0.0, hi: 0.0}` exactly, not just "CI
contains 0."

**Paired-by-seed caveat** (documented in the module docstring, repeated here since it matters for reading the
numbers below): the RNG stream KEYS are identical across decks on one seed (Phase 1 invariant), but Checkered's
post-creation suit swap changes which cards the SAME 'shuffle' draws deal, and Plasma's `ante_scaling=2` +
`plasma=True` scoring balance changes the target and the hand's worth respectively — neither touches the shop/
pack/voucher/boss/tag streams.

### Real runs (repo root, White stake, max_ante 8, n_agents 32, 16 tournament seeds, n_boot 2000)

| player | deck | furthest ante | lives lost | solo win rate | tournament rank_frac (0=best) |
|---|---|---|---|---|---|
| `scripted:hand=greedy` | b_red | 2.41 [2.33,2.49] | 4.00 | **0.000** [0.000,0.000] n=132 | 0.665 [0.624,0.707] |
| (never reroll, never buy) | b_checkered | 3.00 [2.98,3.02] | 4.00 | 0.023 [0.007,0.043] n=150 | 0.601 [0.564,0.641] |
| | b_plasma | 3.58 [3.51,3.65] | 4.00 | **0.300** [0.257,0.340] n=150 | 0.566 [0.522,0.612] |
| `scripted:hand=greedy,reroll=1,buy=1` | b_red | 3.06 [2.93,3.19] | 4.00 | 0.280 [0.220,0.341] n=146 | 0.333 [0.258,0.422] |
| (reroll-once + buy-slot-0) | b_checkered | 3.77 [3.63,3.92] | 4.00 | 0.289 [0.244,0.334] n=150 | 0.295 [0.249,0.344] |
| | b_plasma | 4.17 [4.04,4.30] | 3.99 | **0.430** [0.391,0.468] n=150 | 0.338 [0.285,0.397] |
| `scripted:hand=weak` | b_red | 2.00 [2.00,2.00] | 4.00 | n/a (n=0, floor) | 0.990 [0.982,0.997] |
| (first legal 1-card play, no economy) | b_checkered | 2.00 [2.00,2.00] | 4.00 | n/a (n=0, floor) | 0.992 [0.984,0.999] |
| | b_plasma | 2.00 [2.00,2.00] | 4.00 | n/a (n=0, floor) | 0.989 [0.981,0.997] |

Cross-cell spread (bootstrap CI, paired seeds):

| player | metric | range [95% CI] | variance [95% CI] | n_paired_seeds |
|---|---|---|---|---|
| greedy | win_rate | 0.295 [0.250,0.341] | 0.0178 [0.0126,0.0239] | 132 |
| greedy | rank_frac | 0.100 [0.070,0.132] | 0.0017 [0.0009,0.0030] | 16 |
| greedy,reroll=1,buy=1 | win_rate | 0.155 [0.108,0.222] | 0.0048 [0.0023,0.0093] | 146 |
| greedy,reroll=1,buy=1 | rank_frac | 0.043 [0.013,0.113] | 0.0004 [0.0000,0.0022] | 16 |
| weak | win_rate | n/a — 0 seeds ever reached a Nemesis in any deck | | 0 |
| weak | rank_frac | 0.003 [0.001,0.007] | 0.0000 [0.0000,0.0000] | 16 |

**Reading.** For these three FIXED scripted policies against `vanilla_boss_target`, **Red is the HARDEST
cell, not Plasma** — the reverse of the assessment's naive layer-1 prior ("Plasma... lowest transfer... expect
this to be the hole"). `greedy` (no economy at all) never wins a single Nemesis on Red (132 attempts, 0 wins)
but wins 30% of the time on Plasma with the IDENTICAL policy and seeds; `greedy,reroll=1,buy=1` shows the same
direction, smaller gap (28.0% Red -> 43.0% Plasma). The mechanism is legible and NOT "Plasma transfers well" in
the strategic sense the assessment meant: Plasma's `ante_scaling=2` makes the external target itself 2x harder
in absolute chips, but its `final_scoring_step` chips/mult BALANCE formula (`engine/DECKS_NOTES.md` S2: a lone
Ace scores 64 under Plasma vs. 16 vanilla) inflates a jokerless, low-mult hand — exactly what a
no-economy/light-economy scripted policy plays — by MORE than 2x, so the fixed target under-estimates true
Plasma difficulty for these specific policies. Checkered (the assessment's other "LOW transfer / composition
change" cell) does NOT collapse either — it is consistently a little EASIER than Red for both non-floor
policies (furthest ante 2.41->3.00 and 3.06->3.77; win rate 0.0%->2.3% and 28.0%->28.9%), plausibly because 26
Spades + 26 Hearts makes an unforced 2-suit flush noticeably more likely per hand than a normal 4-suit deck, a
benefit a pure hand-scoring policy (no joker investment needed) collects directly. The weakest policy
(`scripted:hand=weak`) hits a FLOOR: it exhausts all 4 lives on regular blinds before ever reaching a single
Nemesis, on every seed, on every deck (furthest ante exactly 2.00, zero variance) — deck identity cannot matter
to a policy that never survives long enough to face one, and this shows up as a genuinely near-zero cross-cell
spread in the one metric that still has signal (tournament rank_frac range 0.003, an order of magnitude
smaller than the two live policies' 0.043-0.100). **Methodological finding, independent of the "which deck is
hardest" question**: the two modes disagree on the SIZE of the spread (solo win-rate range 0.155-0.295 vs.
tournament rank_frac range 0.043-0.100 for the same two live policies) because tournament mode normalizes away
the external-target/scoring-formula interaction that drives the solo-mode result — every population member
faces the same Plasma-inflated scoring, so RELATIVE rank moves far less than ABSOLUTE win-rate-vs-fixed-target
does. **Caveat, stated plainly**: this is a naive/floor-scripted-policy measurement, not the trained-policy
measurement the assessment's prior was actually about — a trained agent that specifically builds around
Plasma's real (different) chip/mult incentive structure, or around Checkered's suit constraints, could show a
completely different pattern, including the collapse the assessment predicted. What this DOES establish: (1)
the harness and the external target work end-to-end and produce a real, reproducible, mechanistically-explained
signal, not noise (the bootstrap CIs on win_rate/furthest_ante are tight, and the direction is consistent
across two different non-floor policies); (2) anyone reading a Plasma-favorable transfer number from an early
`--objective external` training run should attribute it to the balance-formula/target-formula interaction
documented here BEFORE concluding Plasma "transfers fine" for a trained policy — the target used, not the deck,
may be doing the work.

## 14. `env_v7` reward-reachability audit (notes only, no code change)

**Verdict: `env_v7._finish_step`'s `R_BLIND_BASE * (9 - ante)` reward (`env_v7.py:120` `R_BLIND_BASE = 1.0`,
`:455` `def _finish_step`, `:479` `blind_reward = R_BLIND_BASE * (9 - self._prev_ante)`) is unreachable from
every path `agent` (the Phase 3/4 MCTS training stack) drives — confirmed by exhaustive grep for
`BalatroV7Env(` / `env_v7` / `.step_hand(` / `.step_phase(` across `agent/**`, `tournament/**`,
`eval/**`, `replay/**`: the only construction of a real `BalatroV7Env` anywhere in that tree is
`agent/tests/test_nn_policy.py:96`'s `test_encoder_matches_env_v7`, which calls `env.reset()` ONLY (never
`.step()`/`.step_hand()`/`.step_phase()`) to compare `_encode_obs`'s bytes against `mcts/encoder.py`'s
reimplementation. `mcts/encoder.py:44` imports `BalatroV7Env` purely to steal its UNBOUND `_encode_obs` method
(`:59` `_v7_encode = BalatroV7Env._encode_obs`), called through a `_GameOnly` shim (`:51`) that exposes only
`.game` — no `BalatroV7Env` instance is ever constructed on that path, so `_finish_step` cannot run. `agent`'s
actual outcome signal is exclusively `mcts.outcome.OutcomeFn` (`VanillaOutcome` / `MLBOutcome` / `ExternalOutcome`
— `agent/mcts/outcome.py`), which never reads any env's `.reward`.

**If any MLB path DOES consume it, precisely**: yes, one does, but it is NOT an `agent` training path —
`engine/balatro_sim/env_mp.py`'s `MultiplayerBalatroEnv` / `_PlayerEnvProxy` (a pre-Phase-3, engine-layer
self-play environment built directly on `MLBMatch`, predating the MCTS rewrite). `_PlayerEnvProxy.__init__`
(`env_mp.py:80-81`) builds `self._v7 = BalatroV7Env.__new__(BalatroV7Env)` bound to a REAL live game, and its
`step_hand`/`step_phase` (`:145-147`) delegate straight to `self._v7.step_hand`/`step_phase` — which DO run
`_finish_step` and DO add `R_BLIND_BASE*(9-ante)` per cleared blind, for real, going negative past ante 8 in an
endless MLB match exactly as `engine/MLB_NOTES.md` S5 already flags ("V7 heuristic reward, env_mp inherits it;
harmless for now") and `agent/AGENT_NOTES.md` S8 independently notes. Within the `mp/` subproject, `env_mp.py`
is exercised ONLY by engine-layer tests (`engine/tests/engine_tests/test_env_mp.py`, already counted in the
1614/10/3/0 gate) — grep confirms zero references to `env_mp` / `MultiplayerBalatroEnv` anywhere under
`agent/**` outside doc comments (`AGENT_NOTES.md`, `SETENC_NOTES.md`, a naming-convention comment in
`encoder_set.py:588`). The repo-ROOT `train_v8.py` (a different, legacy BRL-project script, outside `mp/`
entirely, unrelated to this campaign by design — `CAMPAIGN_LOG.md`'s locked parameters) also constructs
`MultiplayerBalatroEnv` for real; that is out of scope for this audit and for Phase 4's training stack. **Net:
no code change needed for Phase 4 — nothing `train_mlb.py` / `agent/train/**` will drive touches this
reward path.** If a future workstream ever wires training through `env_mp.py` instead of `agent`'s MCTS
stack, this reward becomes live again and would need fixing (or replacing with an `OutcomeFn`) first.

## 15. Caveats / needs-engine-change (this agent)

- No engine change needed or requested this phase — `engine`/`rng` were read-only. `targets.py`'s one
  documented gap (`bl_needle`'s 0.5x boss multiplier not reproducible without a seed) is a design limitation of
  a seed-free function, not an engine bug.
- `transfer_spread.py`'s tournament mode defaults to 16 seeds (not the full 126+) purely for wall-clock reasons
  (~5-7s/seed/deck at n=32 agents, max_ante=8, on this machine) — widen with `--tournament-n-seeds` if a future
  run wants tighter CIs on `rank_frac`.
- "random" (uniformly random over legal actions) is not directly expressible as a `scripted:` spec through
  `eval/common.py`'s parser — only `tournament.players.RandomLegalPlayer` implements it, and it is a `Player`
  (`.act(game)->dict`), not the `(match,p,acts)->action` policy function `play_sp_mlb` needs. Substituted
  `scripted:hand=weak` as the "worst baseline" analog for solo mode; tournament mode's background population
  (`default_population`) already includes `RandomLegalPlayer` instances (1/3 of it) in every cell regardless.
- `MCTSPlayer` (`tournament/players.py`) is no longer the Phase-3 placeholder as of this run — a concurrent
  workstream wired it to a real factory over `agent/mcts` mid-Phase-4. `transfer_spread.py`'s
  `_build_tournament_player` already passes a `"checkpoint:<path>"` spec through to it (`"checkpoint:"` with no
  path -> `checkpoint=None` -> cold-start weights, useful as an untrained-net transfer-spread baseline); not
  exercised on a REAL trained checkpoint this phase (none existed yet at hand-off time).
- Integration point for W2: `agent/mcts/outcome.py`'s `ExternalOutcome.from_margin(margin_fn, scale=1.0)` is
  already built for exactly this hook (its own docstring: "the W2 / W4 hook") — `margin_fn` can be built from
  any `targets.get_target(...)` call plus `game.chips_scored`.

## 16. File map (this agent, Phase 4)

- `eval/targets.py` — `vanilla_boss_target`, `vanilla_boss_target_fn`, `scaled_own_big_blind`,
  `table_target`, `get_target`/`TARGETS` registry. Engine-only deps, own fork-guard.
- `eval/transfer_spread.py` — `solo_cell`, `tournament_cell`, `_cross_cell_bootstrap`, `evaluate_player`,
  `to_markdown`, CLI (`python -m eval.transfer_spread`).
- `eval/tests/test_targets.py` (57), `eval/tests/test_transfer_spread.py` (19).
- `results/transfer_spread_greedy.{json,md}`, `transfer_spread_greedy_reroll1_buy1.{json,md}`,
  `transfer_spread_weak.{json,md}` — the real runs behind S13.
