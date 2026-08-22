# EVAL_NOTES — Phase 3 W4: eval harness + ρ-decay harness (2026-08-21/22)

**Agent W4.** Files: `mp/eval/common.py` (shared bootstrap / drivers / stats), `mp/eval/eval_harness.py`,
`mp/eval/rho_decay.py`, `mp/eval/conftest.py`, `mp/eval/tests/` (49 tests), `mp/results/*.json` (outputs),
this note. `mp/engine/**` and `mp/rng/**` were only read, never edited. `mlb_match_demo.ScriptedPlayer` /
`make_policy` / `greedy_hand` / `weakest_play` / `shelf_indices` are imported from `mp/scripts/`, not copied.

## 0. Gates (final run, repo root, `python` = 3.13)

| gate | result |
|---|---|
| `python -m pytest mp/eval/tests -q` | **49 passed** (~30s) |
| `python -m pytest mp/engine/tests -q` | **1609 passed / 10 skipped / 3 xfailed** — unchanged |
| `python -m pytest mp/tests -q` | **1073 passed / 2 xfailed** — unchanged |
| `python -m mp.eval.eval_harness --mode 1v1 --player ... --reference ... --out mp/results/demo_1v1.json` | OK, 126 seeds, 21.7s |
| `python -m mp.eval.rho_decay --all --n-extra-seeds 24 --horizons 1,2,4,8 --n-boot 2000 --out-dir mp/results` | OK, 150 seeds × 3 perturbations, ~100s each |

## 1. How to run

```
# eval harness (repo root)
python -m mp.eval.eval_harness --mode sp_vanilla --player "scripted:hand=greedy,buy=1,pack=0" \
    --out mp/results/my_report.json
python -m mp.eval.eval_harness --mode sp_mlb --player "scripted:hand=greedy,reroll=1,buy=1" \
    --max-antes 8 --target-k 1.0 --out mp/results/my_sp_mlb.json
python -m mp.eval.eval_harness --mode 1v1 --player "scripted:hand=greedy,reroll=1,buy=1" \
    --reference "scripted:hand=greedy,buy=1" --out mp/results/my_1v1.json
python -m mp.eval.eval_harness --compare mp/results/a.json mp/results/b.json --out mp/results/cmp.json

# ρ-decay (repo root)
python -m mp.eval.rho_decay --perturbation buy_slot0 --horizons 1,2,4,8 --out mp/results/rho_decay_buy_slot0.json
python -m mp.eval.rho_decay --all --n-extra-seeds 24 --out-dir mp/results   # all 3 perturbations, 150 seeds
python -m mp.eval.rho_decay --list-perturbations

# tests
python -m pytest mp/eval/tests -q
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
`money` and `lives_lost` correlations are in `mp/results/rho_decay_<perturbation>.json` per horizon
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

**`mp/engine/balatro_sim/game.py:1546-1552, 1571-1580, 1582-1592`** (`bl_hook`, `bl_eye`, `bl_mouth` boss-ability
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
report can distinguish this from a genuine 0-lives loss. **Not fixed** — `mp/engine` is frozen for Phase 3;
this is a work-around, not a patch. Affected 1/150 seeds in the real run (0.7%); harmless to the measurement
because it is excluded, not silently miscounted.

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

`mp.eval.common.Player`: an object with `.act(game: balatro_sim.game.BalatroGame) -> dict` (one of
`game.legal_actions()`). Wire it into `mp.eval.common.parse_player_spec` (currently `"checkpoint:<path>"` raises
`NotImplementedError` with this exact interface in the message) so `--player checkpoint:<path>` resolves to a
real player; `adapt_player(player)` already converts any such object into the `(match_or_shim, player_idx,
legal_actions) -> action` signature every driver in `common.py` and `MLBMatch.play_out` expect — no other
change needed in `eval_harness.py` / `rho_decay.py` once that one function is filled in. MLB awareness (a
checkpoint player must handle `ruleset="mlb"` including `PVP_WAIT`) is W1's concern per the Phase 3 brief;
nothing in this harness assumes the player is a `ScriptedPlayer` beyond that one dispatch point.

## 10. File map

- `mp/eval/common.py` — bootstrap (fork-guarded import of `mp/engine`), `Player` protocol + spec parsing,
  `SoloShim` (drives a lone `BalatroGame` through `mlb_match_demo`'s policy signature), `play_sp_vanilla` /
  `play_sp_mlb` / `play_1v1` drivers, `play_arm_to_horizons` (paired-arm horizon driver for rho_decay), target
  functions, pure-python bootstrap/correlation statistics, `sample_size_per_arm`.
- `mp/eval/eval_harness.py` — CLI + `evaluate()` / `compare()`.
- `mp/eval/rho_decay.py` — CLI + `PERTURBATIONS`, `_wrap_perturbation`, `make_perturbed_game`, `measure_rho`.
- `mp/eval/conftest.py` — fork-guard bootstrap for `mp/eval/tests`.
- `mp/eval/tests/test_common.py`, `test_eval_harness.py`, `test_rho_decay.py` — 49 tests.
- `mp/results/demo_1v1.json`, `rho_decay_buy_slot0.json`, `rho_decay_reroll_once.json`,
  `rho_decay_skip_small.json` — the real runs behind §5.
