# ADVISOR_NOTES -- W6: the snapshot advisor CLI + head-to-head evals (Phase 5 rev 2, 2026-08-23)

Owner: W6. Implements `mp/docs/PHASE5_BRIEF_2026-08.md` row W6 / interface section 2 / gate 5.
Files: `advisor.py` (the report), `cli.py` (`advise` subcommand), `h2h.py` (head-to-head
driver + CLI), `fixtures/bloodstone_vs_invisible.py` (Tagg's acceptance-test state),
`tests/test_advisor.py` / `tests/test_h2h.py` (20 tests), results
`mp/results/advisor_bloodstone_vs_invisible_2026-08-23.md`,
`mp/results/h2h_smoke_ev_fast_vs_ev_full.{json,md}`,
`mp/results/h2h_smoke_ev_fast_vs_real1_det.{json,md}`. Nothing outside `mp/ev/advisor.py`,
`mp/ev/cli.py`, `mp/ev/h2h.py`, `mp/ev/fixtures/**`, `mp/ev/tests/test_{advisor,h2h}.py`,
`mp/ev/ADVISOR_NOTES.md` and `mp/results/{advisor_*,h2h_*}` was touched; every other file
under `mp/ev/` (W3/W5), `mp/stats/**` (W4), `mp/agent/**` (W1/W2), `mp/engine/**`, `mp/rng/**`,
`mp/eval/**` is read-only -- including `mp/ev/match_player.py`, a W5 file that landed mid-
session and turned out to be exactly the opponent-aware-V hook this workstream needed
(section 1 / section 6 explain the wiring; it is imported, never edited).

## 0. TL;DR

```
python mp/ev/cli.py advise fixture:bloodstone_vs_invisible --player 0 --rollouts 32
python mp/ev/cli.py advise replay:<path/to/match.jsonl>:<step> --player 1 --checkpoint <ckpt>
python mp/ev/cli.py advise seed:11111111:120 --player 0

python mp/ev/h2h.py --a ev:fast --b ev:full --n-seeds 2 --procs 2 \
    --out-json mp/results/h2h_smoke.json --out-md mp/results/h2h_smoke.md
```

## 1. The advisor report -- what each section is and how it's computed

`advisor.advise(match, player, *, n_rollouts, rollout_seed, rollout_budget, budget,
checkpoint, top_n) -> str`. Four sections, in order:

1. **The situation** (`situation_lines` + `opponent_public_block_lines`): ante/blind/state,
   lives and $ both sides, my jokers, and the opponent's jokers -- labelled
   `[HIDDEN IN A REAL MATCH]` because every state source here has full simulator access (this
   is an offline analysis tool, not a live client); then the actual `opponent_view(match,
   player)` block (W1's public-info-only reader) so Tagg can see what a REAL deployment would
   actually have to work with, side by side with what we're printing anyway.

2. **Three P(win) estimators** (`prob_block`):
   - **rollout**: `labels.label_state(match, player, n_rollouts=..., policy_factory=
     labels.make_policy_factory(budget=rollout_budget, epsilon=0.02))` -- W5's determinized
     Monte-Carlo label estimator, unmodified. Default `rollout_budget="fast"` (matches how
     labels are actually generated for training; `--rollout-budget full` is available but
     32 rollouts x full-budget hand decisions would take many minutes -- not the default).
   - **race**: `race.curve_from_history(match.pvp_log, player, ante)` for both sides, then
     `race.p_win(my_curve, their_curve, my_lives, their_lives, ante, blinds_done=...)`, plus
     `race.race_table(...)` for the next 4 antes. `ante` = the FOCAL player's own current
     `game.ante`; `blinds_done` = that player's own `blind_idx` clipped to
     `cfg.regular_blinds`. **Approximation, documented**: `race.p_win`'s DP assumes both
     players share one ante clock (`blinds_done` regular blinds behind BOTH players); in an
     MLB match Small/Big are played independently so the two players' `ante`/`blind_idx` can
     differ by a blind or two outside the Nemesis. Using the focal player's own clock is the
     natural "what does MY next Nemesis look like" reading and is exact once the Nemesis
     synchronises both players anyway -- it just doesn't model the opponent being a blind
     ahead/behind mid-ante.
   - **V**: only when `--checkpoint` is given. `advise()` builds ONE
     `match_player.MatchAwareEVPlayer(net, encoder, budget=budget)` (W5's wrapper -- see
     below) bound to `(match, player)`, and both the standalone "V" number here AND the
     ranked-action table (point 3) are computed through it, so they share one checkpoint load
     and one consistently-bound opponent view. Prints `n/a` and is silently excluded from the
     disagreement check when no checkpoint is given -- **there is no trained V checkpoint
     yet** (Phase 5 W5 hasn't run the label campaign / trainer at scale), so every number in
     this doc's acceptance run is `n/a` for V; the wiring itself IS covered end to end by
     `test_advisor.py::test_advisor_with_checkpoint_is_opponent_aware_and_v_guided` against a
     tiny (~26k param, untrained) throwaway checkpoint -- see section 6.
   - Any two of the (up to three) numbers differing by more than 0.15 prints a
     `** DISAGREEMENT **` line naming which pair and by how much.

3. **The ranked action table** (`action_table_lines` + `stats_table_lines`):
   - No checkpoint: `EVPlayer(value_fn=None, stats=None, budget=budget).explain(game)` -- the
     rules tier.
   - A checkpoint given: the SAME `match_player.MatchAwareEVPlayer` built for point 2's "V"
     number is passed straight in as the `explainer` (`action_table_lines(game,
     explainer=mp_player, ...)`) and its own `.explain(game)` is used instead of building a
     second `EVPlayer`. **This is the opponent-visibility fix**: `EVPlayer`'s `value_fn`
     contract is `Callable[[BalatroGame], float]` (single argument -- every clone it
     evaluates internally only ever gets a bare game, never a match), which on its own would
     mean V could never see the true opponent while ranking shop/pack/blind candidates.
     `match_player.py` (W5, landed mid-session -- **not authored by this workstream**, but
     exactly the hook this workstream needed and flagged as missing during development) closes
     that gap: `MatchAwareEVPlayer` binds a MUTABLE opponent view into a bound method
     (`self.value_fn(self, game)`, itself single-argument, satisfying `EVPlayer`'s contract)
     that closes over `self._opp`, and `.bind(match, player)` / `.refresh()` keep `self._opp`
     current -- so every clone V is asked about during one decision shares the true,
     currently-bound opponent view. The tag `(V-guided, opponent-aware)` in the printed header
     marks when this path is active (see `test_advisor_with_checkpoint_is_opponent_aware_and_v_guided`).
   - `stats.decide.explain(decide.decision_table(game))` (W4) -- always called; prints its own
     "(no shop/pack rows...)" message outside SHOP/BOOSTER_OPEN, so no state-type branching is
     needed in `advisor.py` itself. This is a SEPARATE table from the ranked-action one on
     purpose (see section 4's finding 4 below for why printing both, side by side, actually
     surfaced a real inconsistency between the rules tier and the stats tier).

4. **Opponent read** (`opponent_read_lines`): the most recent resolved Nemesis (their score /
   hands used / who won it) and this-ante/lifetime shop econ (`sells_per_ante`,
   `spent_in_shop`, totals) from the same `opponent_view` block. Explicitly labelled
   level-0 -- no belief model over the shared shop menu, no signalling inference. This is
   everything the brief's item 4 asks for; nothing more is claimed.

## 2. The fixture (`fixtures/bloodstone_vs_invisible.py`)

`build(seed="TAGGADVR", policy_seed=0)`:

1. Drives one `MLBMatch(deck_key="b_red", stake=1)` (Red deck / White stake, per the brief)
   forward with `EVPlayer(budget="fast", epsilon=0)` self-play on BOTH sides (deterministic:
   `epsilon=0` means no RNG stream is consumed by the players at all) until player 0's OWN
   game reaches `ante in (4, 5)` at a `BLIND_SELECT` or `SHOP` state. In practice this lands
   at ante 4 SHOP (right after the ante-4 Nemesis) within under a second of wall clock.
2. **Player 0 -> Bloodstone + a Hearts-leaning deck.** `debug_add_joker("j_bloodstone")`
   (`j_bloodstone`: `mp/engine/balatro_sim/jokers/mult.py::_Bloodstone`, 1/2 chance of x1.5
   Mult per scored Heart). The "Hearts-leaning deck" is `lean_suit(g0, "Hearts", 0.5)`: a
   **fixture-only shortcut**, NOT a real tarot effect -- it directly retints plain
   (non-Stone) `full_deck` cards to `Hearts` until >= 50% of the 52(+)-card collection is
   Hearts. `game.py`'s own comment documents that `deck`/`hand`/`discard_pile` hold
   REFERENCES into `full_deck`, so this is visible everywhere immediately, exactly as if the
   player had bought that many Hearts cards -- just without spending money or consuming an
   actual Justice/Death/etc. tarot to get there. This is called out explicitly in the
   fixture's docstring and here so nobody mistakes the resulting deck composition for
   something the analytic hand player or V would ever see from real in-game economy.
3. **Player 1 -> Invisible Joker + Blueprint, positioned to copy it.**
   `debug_add_joker("j_blueprint")` then `debug_add_joker("j_invisible")` -- `debug_add_joker`
   appends, so Blueprint lands immediately to Invisible's LEFT (whatever the absolute index
   is; both self-play players had already bought ~5 jokers by ante 4, so the actual indices
   in the smoke run were 5 and 6, not 0 and 1). `jokers/misc.py::_Blueprint.
   _get_copy_target` copies `ctx.jokers[idx + 1]` -- exactly Invisible.

   **Read the engine before trusting this "combo" (this is the honest finding the brief asked
   for, not swept under the rug)**: `_Blueprint` forwards exactly four hooks to its target --
   `pre_score`, `on_score_card`, `on_held_card`, `on_hand_scored`
   (`mp/engine/balatro_sim/jokers/misc.py::_Blueprint`). `_InvisibleJoker` implements exactly
   TWO hooks -- `on_round_end` (its own rounds-survived counter) and `on_sell` (the
   duplicate-a-random-other-joker effect) -- and NEITHER is in Blueprint's forwarded set.
   **Blueprint positioned on Invisible is therefore a no-op at scoring time in this engine**:
   it does not accelerate Invisible's 2-round counter, does not trigger a duplicate on sell,
   and contributes nothing to a hand's chips/mult. This matches real Balatro too (Invisible's
   ability was never a scoring-time trigger) -- it is a correct, if unglamorous, reading of
   the two jokers' actual code, and it is exactly the kind of "this joker slot is doing
   nothing" read a strong human player -- or a good advisor -- should be able to say out
   loud. Section "Reading" of the results doc calls out that neither printed table has any
   lever to FIX this (no joker-reorder action exists in `legal_actions()` at all), which is a
   genuine action-space gap, not something this workstream's tables can paper over.
4. Plausible $ and lives: player 0 gets 3 lives (down from the self-play state, since the
   brief specifically wants a live life-race question), $ >= 12; player 1 gets 4 lives, $ >=
   8. Both floors are `max(existing, floor)` so self-play's own economy is respected when it
   already clears the floor.

**Determinism**: `build()` called twice with the same arguments produces byte-identical
`match.signature()` (RNG digests included) -- `test_fixture_builds_deterministically` pins
this. `state_signature()` does not include `JokerInstance.sort_id` (a process-global creation
counter), so this holds even though `sort_id` itself is not reproducible across separate
Python processes -- only `(key, edition, state)` per joker is part of the signature, and those
are set purely from the deterministic self-play + the fixed `debug_add_joker` calls.

## 3. Timings

**Advisor, 32 rollouts, sequential** (`labels.label_state` has no internal parallelism -- the
brief's own instruction was to time this honestly rather than fake parallelism):

| perspective | wall clock | s/rollout |
|---|---|---|
| player 0 | 77.3 s | 2.42 |
| player 1 | 80.0 s | 2.50 |

Both single-process. The ranked-action table (33-43 ms) and the race calculator (<1 ms) are
negligible next to the rollouts; almost the entire `advise()` wall clock IS the rollout label.
`--rollout-budget full` would multiply this by roughly (74 ms / 4.4 ms) per hand decision
inside each rollout (see `EV_NOTES.md` section 6) -- untested here, not the default.

**H2H smokes** (both required by the brief; `--procs 2`, 2 seeds x 2 seatings = 4 matches):

| matchup | wall clock | mean s/match |
|---|---|---|
| `ev:fast` vs `ev:full` | 43.7 s | 17.2 s |
| `ev:fast` vs `real1:det --sims 16` | 26.6 s | 9.7 s |

(A pure `ev:fast` vs `ev:fast` sanity check, not one of the two required smokes, ran inline
in ~17 s for 2 matches / 1 seed -- ~7-9 s/match, matching `EV_NOTES.md`'s own "an MLB match ~=
7 s" figure and confirming the seating-swap machinery is correct: with an IDENTICAL policy on
both seats the two seatings are exact mirror images of the same underlying trajectory --
`test_seatings_are_mirrors_for_an_identical_matchup` pins this.)

## 4. Deferred: the exact commands for the real head-to-head runs

Per the resource constraint, these are NOT run this session -- they are one-line commands
ready for the lead to launch once the box is free, with wall-clock projected from the smokes
above (procs capped at 4 to match the constraint that was in force while this workstream
built them; raise `--procs` once that constraint is lifted).

**(i) `ev:full+stats` vs `real1:det`, 30 seeds** -- the eventual "does the analytic+stats
player beat the trained-but-clairvoyant-search baseline played honestly" question:

```
python mp/ev/h2h.py --a ev:full+stats --b real1:det --sims 40 --n-seeds 30 --procs 4 \
    --out-json mp/results/h2h_ev_full_stats_vs_real1_det_30seeds.json \
    --out-md   mp/results/h2h_ev_full_stats_vs_real1_det_30seeds.md
```

Projection: the smoke used `ev:fast` (not `ev:full+stats`) and `--sims 16` (not real1.sh's own
40) to stay fast. Scaling from the smoke's mean 9.7 s/match: `--sims 16 -> 40` is roughly
2.5x per-decision MCTS cost for `real1:det` (its search covers every one of its own
decisions, hand and shop alike, so cost scales close to linearly in `sims`); swapping `ev:fast`
for `ev:full+stats` on the other side adds roughly another 2x, from the `ev:fast` vs `ev:full`
smoke's own ~2x jump over a same-budget baseline. Stacked (~5x, both multipliers are rough and
compound their uncertainty): ~19 s/match x 2 seatings x 30 seeds / 4 procs ~= **10-15
minutes**. Recommend a 1-2 seed check at the real settings before committing to the full 30.

**(ii) `ev:fast` vs `ev:full`, 30 seeds** -- directly scaled from the smoke that already ran
at these exact settings (mean 17.2 s/match, worst observed single seed-job ~41 s of match
time for its 2 seatings):

```
python mp/ev/h2h.py --a ev:fast --b ev:full --n-seeds 30 --procs 4 \
    --out-json mp/results/h2h_ev_fast_vs_ev_full_30seeds.json \
    --out-md   mp/results/h2h_ev_fast_vs_ev_full_30seeds.md
```

Projection: 30 seeds / 4 procs = 8 batches x ~40 s worst-case per-seed-job ~= **~5-6
minutes**.

**(iii) `ev:full+stats` vs `ev:full`, 30 seeds** -- both sides pay the full-budget hand-decision
cost (the dominant cost in (ii)'s smoke), so expect somewhat more than (ii) but well under 2x
(the OTHER side wasn't free in (ii) either -- `ev:full` was already the slow side there):

```
python mp/ev/h2h.py --a ev:full+stats --b ev:full --n-seeds 30 --procs 4 \
    --out-json mp/results/h2h_ev_full_stats_vs_ev_full_30seeds.json \
    --out-md   mp/results/h2h_ev_full_stats_vs_ev_full_30seeds.md
```

Projection: ~1.5-2x of (ii)'s per-match cost ~= **~8-12 minutes**.

All three projections stack at least one rough multiplier on top of a 2-seed smoke and should
be treated as planning numbers, not promises -- exactly the same honesty standard
`DETERMINIZE_NOTES.md` section 6 already set for the clairvoyance measurement's own
projection.

## 5. `h2h.py` design notes

- **Paired by seed, both seatings**: one seed produces two matches (A as player 0 / B as
  player 1, and the mirror) so the fixed `MLBMatch.current_player()` alternation order and any
  first-mover shop-seed effects cancel in the aggregate, per the brief.
- **One worker job = one seed, both seatings** (not one job per match): an MCTS player spec
  (`real1:det` / `real1:clair`) is built ONCE per job and reused (with `.reset()` between
  matches) for both seatings of that seed -- halves checkpoint loads relative to a
  one-job-per-match design. `EVPlayer`/scripted specs are cheap to rebuild either way.
- **Player specs** (`build_player`): `ev:fast` / `ev:full` / `ev:full+stats` (token-parsed on
  `+`, so `ev:fast+stats` also works even though the brief only names `full+stats`);
  `real1:det` (`mcts.determinize.make_determinized_player`, the non-clairvoyant baseline) /
  `real1:clair` (`mcts.player.make_player`, clairvoyant, table-only per the brief);
  `scripted:<fields>` (reuses `mp/eval/common.py::make_player_policy` verbatim -- no
  reimplementation). Both `real1:*` specs use exactly `measure_clairvoyance.py`'s
  `REAL1_FLAGS` (`encoder=set, strategy=gumbel, heuristic_prior=0.4, heuristic_tau=0.35,
  max_hand_candidates=32, heuristic_exact_top=8, heuristic_discard_bias=1.0`) so a `real1:det`
  vs `real1:clair` comparison isolates exactly "sees the future or not", nothing else.
- **Seeds for player construction** (`_stable_seed`): `zlib.crc32` of a text key, not
  `hash()` -- `hash(str)` is per-process salted in CPython (the same reasoning
  `mcts/determinize.py`'s own `seed_stream` docstring gives for avoiding it), so it cannot
  reproducibly seed a worker across separate processes/runs.
- **JSON schema** (`test_h2h.py` pins the field sets verbatim): top level `spec_a, spec_b,
  seeds, n_seeds, sims, checkpoint, lives, max_steps, deck_key, stake, procs, seed_base,
  wall_clock_s, trials, summary`. Each `trials[i]`: `seed, seating, steps, done, seconds,
  a_win (bool|None), lives_a, lives_b, lives_margin_a, final_ante_a, final_ante_b,
  final_money_a, final_money_b, nem_wins_a, nem_wins_b, nem_total`. `summary`: `n_trials,
  n_decided, a_wins, b_wins, undecided, win_rate_a ({point,lo,hi,n} via
  mp/eval/common.bootstrap_ci), mean_final_ante_a, mean_final_ante_b, mean_lives_margin_a,
  nemesis_win_rate_a, mean_seconds_per_match`.
- **Undecided matches**: if `play_out` hits `--max-steps` before either side reaches 0 lives,
  `done=False`, `a_win=None`, and the trial is excluded from `win_rate_a` (`n_decided` /
  `undecided` make this visible) but still counted in the ante/lives-margin/seconds means.
  Never observed in any run this session (`--max-steps` defaults to 100,000; real matches
  finish in a few hundred steps), but the schema handles it explicitly rather than crashing.

## 6. What's untested / known gaps for the lead

1. **No real V checkpoint exists yet** -- `load_value_fn` (still available as a simpler,
   opponent-agnostic-by-default utility; no longer used internally by `advise()`) and the
   `match_player.MatchAwareEVPlayer`-based V wiring are exercised end to end by
   `test_advisor_with_checkpoint_is_opponent_aware_and_v_guided` against a tiny (~26k param,
   untrained) throwaway checkpoint built in the test itself -- the WIRING is real-net-tested,
   the NUMBERS are not (an untrained net's output is meaningless by construction). Once W5
   trains a real one, run `python mp/ev/cli.py advise fixture:bloodstone_vs_invisible
   --player 0 --checkpoint <ckpt>` and confirm the V line + V-guided ranked table print
   sensibly before trusting the NUMBERS.
2. **The opponent-visibility gap this section originally flagged is fixed**: an earlier draft
   of `advisor.py` wrapped `value_net.make_value_fn`'s two-argument `fn(game, opp)` down to a
   bare one-argument lambda for `EVPlayer`, which meant V never saw the true opponent while
   ranking shop/pack/blind candidates. `mp/ev/match_player.py` (W5, landed mid-session) turned
   out to be exactly the hook needed -- `MatchAwareEVPlayer` binds a mutable opponent view into
   a bound `value_fn` closure and refreshes it from the live match before every decision.
   `advisor.py` now builds one `MatchAwareEVPlayer` per `--checkpoint` call and threads it
   through both the "V" number and the ranked-action table (section 1, points 2-3). No action
   needed from the lead here; noted so the history is visible.
3. **`race.p_win`'s single shared ante clock** (section 1, point 2's "race" bullet) is a
   documented approximation already noted in `race.py` itself; the advisor inherits it
   unchanged.
4. **`h2h.py`'s `real1:*` specs need `mp/agent/runs/real1/latest.pt`** to exist (it does, 230
   MB, from the 106-gen Phase 4 run) -- both smokes in this doc actually loaded and drove it.
5. `mp/ev/tests` was at 96 passing before this workstream's additions (92 in the brief's
   snapshot + 4 more from a concurrent W5 change mid-session); this workstream added 20
   (14 `test_advisor.py` + 6 `test_h2h.py`) for **116 passing, 0 failing**, run whole
   (`python -m pytest mp/ev/tests -q`). No flakiness observed; W5's files were not
   touched.

## 7. State-source spec reference

- `fixture:<name>` -- `fixtures.FIXTURES[<name>]()`. Only `bloodstone_vs_invisible` is
  registered; add more builders + register them in `fixtures/__init__.py`'s `FIXTURES` dict.
- `replay:<path>:<step>` -- an `mp/replay` `MatchLogger` JSONL log (first `kind=="match"`
  line), replayed through its first `<step>` ops via a fresh `MLBMatch` (this module's own
  driver -- `mp/replay` has no "replay to a given step" entry point and this workstream does
  not edit `mp/replay`; only its PUBLIC `replay.load_line` is used). `<step>` beyond the
  logged op count clamps to the end. Parsed with `rest.rpartition(":")` (not `partition`) so a
  Windows path's own drive-letter colon (`C:\...`) is not mistaken for the step separator.
- `seed:<seed>:<step>` -- a fresh `MLBMatch(seed=...)` re-driven `<step>` decisions by two
  fresh `EVPlayer(budget=..., epsilon=0)` (deterministic given `seed` + `--policy-seed` +
  `--drive-budget`).

`mp/replay` is a real Python package (its modules use relative imports, e.g. `from ._util
import ...`), so it must be imported as `replay.replay` / `replay.log` with `mp/` (not
`mp/replay/`) on `sys.path` -- importing `mp/replay/replay.py` directly as a bare top-level
module raises `ImportError: attempted relative import with no known parent package`. This
tripped both `advisor._replay_to_step` and `test_advisor.py`'s own log-writing helper during
development; both are fixed to import via the package.

## 8. Encoding note

Every printed line in `advisor.py`/`fixtures/bloodstone_vs_invisible.py` is plain ASCII (em
dashes and box-drawing separators were swapped for `--`/`=` after a smoke test showed
`print()` mangling them under this Windows box's console codepage) -- the CLI is meant to be
run directly in Tagg's PowerShell/cmd, not only through a UTF-8-forced pipe.
