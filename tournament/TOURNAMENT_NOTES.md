# TOURNAMENT_NOTES — Phase 3 W2: N-agent same-seed tournament runner + N x N matrix

**Agent W2, 2026-08-21.** Deliverables: `tournament/{bootstrap,players,runner,matrix,cli}.py`,
`tournament/conftest.py`, `tournament/tests/` (31 tests), this note. `engine/**` and
`rng/**` are untouched (read-only fork import via `bootstrap.py`); one engine gap was found
and worked around here, not fixed there (§5).

## 0. Gates (final run, repo root, `python` = 3.13)

| gate | result |
|---|---|
| `python -m pytest tournament/tests -q` | **31 passed / 0 failed**, ~49 s |
| `python -m pytest engine/tests -q` | **1609 passed / 10 skipped / 3 xfailed / 0 failed** (unchanged) |
| `python -m pytest tests -q` | **1073 passed / 2 xfailed / 0 failed** (unchanged) |
| `python -m tournament.cli --seed 7I4M53DL --n 100 --life-rule none --max-ante 8` | runs, prints per-ante summary + wall clock (§4) |

## 1. Architecture

Each of the N agents gets its OWN `BalatroGame(seed, deck_key=, stake=, ruleset="mlb")` —
independent shops, rerolls, skips, jokers. `runner.py`'s central fact (`game.py`'s own
`lose_life` / `end_pvp` / `pvp_solo` hooks, MLB_NOTES.md §2 items 1.2e/1.3f): a game with the
default `pvp_solo=True` (never attached to an `MLBMatch`) plays its Nemesis blind entirely on
its own — `play_blind` starts it immediately, and the instant the agent is out of hands the
game auto-calls `end_pvp()` and lands in `ROUND_EVAL` with `chips_scored` intact and **no
life lost yet**. So N agents can be driven completely independently, one at a time, through
an entire ante (Small, Big, and the Nemesis's own exhaustion) with **no coordination object at
all** — `MLBMatch` is never instantiated by the runner. The only synchronisation point is
*after* every alive agent has finished this ante's Nemesis: build the N x N matrix from their
final scores, apply the `life_rule`, drop anyone who hits 0 lives, cash out the survivors,
repeat. This is "ante lockstep": one `Tournament.run()` while-loop iteration per Nemesis ante,
not per individual action — see `runner._drive_to_next_nemesis` / `Tournament.run`.

## 2. The interleaving contract (decision 0.2) and its known gap

**Contract.** Every agent plays its Nemesis blind *blind* — it never sees another agent's
live score or hands-left (the runner never calls `set_pvp_info`) — and plays *to exhaustion*
(every hand). Outcomes are then computed from final scores with the server rule: strictly
lower score loses a life, an exact tie costs nobody (`matrix.outcome_matrix`, same rule
`MLBMatch._resolve_pvp` uses). This is exactly what `pvp_solo=True` already gives for free:
nothing in the tournament runner has to fake "waiting for an opponent."

**Known gap** (same one MLB_NOTES.md §1.4a and the brief's decision 0.2 flag): MLB pays no
unused-hand money at a PvP blind, so playing every hand out costs nothing in dollars, but it
is NOT free in game state. Two things differ from a real early-ended 1v1 PvP:

1. **Per-hand joker state.** Jokers that trigger "when this hand is played/discarded"
   regardless of the blind's outcome (Ice Cream −20 chips/hand, Popcorn −4 mult/hand, Turtle
   Bean −1 hand size/round, Seltzer's `-1` charge/hand played, Ramen/Hiker-style per-hand
   decay, Diet Cola style discard-based sodium, etc.) keep firing for every hand an agent
   plays after the point a real match would have ended the round early for the loser (whose
   hands are forfeited the instant the server decides the verdict). A real 1v1 loser who was
   crushed after 1 hand never fires jokers on hands 2-4; our same agent, played blind to
   exhaustion, does.
2. **Deck-out.** `_mlb_check_deck_out()` (game.py) can end a Nemesis (or take a life on the
   0-hands-played branch) purely from running out of cards mid-blind — a state a real
   early-ended PvP round may never reach for the "winner" side, whose hands are cut short.

Both are exactly the brief's flagged gap, not new findings; nothing here mitigates them
(scripted/random players rarely reach either in 8 antes — the smoke tests never hit a
deck-out). `MLBMatch` (unused by this runner) keeps the canonical 1v1 alternation and does
not have this gap; W4's eval harness driving real 1v1s via `MLBMatch` is the reference for
anything that needs the faithful early-end.

## 3. How lives map (`life_rule`)

All three rules only decide **who loses a life from the Nemesis outcome**; a regular
(non-Nemesis) blind loss (Small, Big, or ante 1's real vanilla Boss) still costs a life
through the engine's own faithful `_mlb_fail_round` / `_mlb_check_deck_out`, independent of
`life_rule` — that machinery is untouched and always runs.

- **`"none"`** (pure measurement): the Nemesis-triggered life step never fires
  (`Tournament._decide_losers` returns `set()` unconditionally). To make "nobody dies, run to
  a fixed ante" actually true, every game is constructed with `lives =
  NONE_RULE_LIVES_SENTINEL = 1_000_000_000` regardless of the `lives=` argument — a regular
  blind fail still decrements it (visible in `final_lives`, e.g. `999999985`), it just can
  never reach 0 within any realistic horizon. Verified in
  `tests/test_life_rules.py::test_integration_none_rule_...` and the 100-agent smoke.
- **`"median"`**: every agent whose score is **strictly below** the population median among
  agents present this Nemesis loses one life (`statistics.median`); an agent sitting exactly
  at the median survives, so an all-tied population costs nobody (matches "a tie costs
  nobody" — the design doc's option (a), explicitly **not** faithful to real MLB, useful for
  selection/value-target purposes per the training design doc §6).
- **`"paired"`** (default, faithful 1v1 lives): a **fixed** assignment computed once at
  construction (`runner._pairing`): consecutive pairs `(0,1), (2,3), (4,5), ...` in agent
  index order. The N x N matrix is still built from **every** pair's scores (all `C(N,2)`
  comparisons, `matrix.AnteMatrix`) — only the life decrement is restricted to the fixed
  pairing. Strictly-lower score in a pairing loses a life; a tie costs nobody; if either
  member of a pairing already died in an earlier round, that pairing is simply skipped this
  round (a "bye" — no life change from it) rather than reassigned.
  - **Odd `n_agents`**: the last agent (index `n_agents - 1`) has no fixed partner and
    instead gets a **rotating** opponent — one different agent per Nemesis round, cycling
    `0, 1, 2, ..., n_agents - 2, 0, 1, ...` in index order (`round_idx = ante -
    MLB_PVP_START_ROUND`). **Documented quirk, not hidden:** the rotating partner's own fixed
    pairing still runs that same round, so that agent can lose a life from either comparison
    in the same round (verified not to double-count via `set` union in
    `Tournament._decide_losers`; `tests/test_life_rules.py::
    test_paired_rule_odd_agent_out_rotates_deterministically` isolates the rotation from the
    fixed pairs to pin the exact cycle).

## 4. Fan-out: construct vs clone

`runner.construct_games` (N fresh `BalatroGame(...)` constructions, each doing the full
run-start draw sequence) vs `runner.clone_games` (one construction + `BalatroGame.clone()`
N-1 times). Benchmarked with `runner.benchmark_fanout(seed="7I4M53DL", n=100, repeats=3)`
(best-of-3, this machine):

| method | wall clock (n=100) |
|---|---|
| construct | 0.0284 s |
| clone | **0.0073 s** (≈3.9x faster) |

Both produce bit-identical initial `state_signature()`s across all N games
(`benchmark_fanout`'s `signatures_equal`, also `tests/test_fanout.py`). `clone` won and is
`FANOUT_DEFAULT`; both remain selectable (`Tournament(..., fanout="construct"|"clone"|"auto")`)
and are exercised by the same determinism tests either way
(`tests/test_determinism.py::test_determinism_scripted_population_construct_fanout`).
Fan-out cost is negligible either way next to the actual play-through (milliseconds vs tens of
seconds) — this only matters if a future workload calls `Tournament` many times per second.

## 5. Found, not fixed: an MLB gap in three boss-ability branches (needs engine change)

**`engine/balatro_sim/game.py:1551`, `:1579`, `:1591`** (the `bl_hook`, `bl_eye`, `bl_mouth`
boss-ability "hand rejected" branches inside `_play_hand`): each hard-sets
`self.state = State.GAME_OVER` the instant a rejected hand drives `hands_left` to 0, with
**no check of `self.mlb`** — unlike every other hand-exhaustion path in the same file
(`_play_hand`'s own plain `hands_left <= 0` branch at line ~1713, and
`_mlb_check_deck_out`), which route an MLB game through `_mlb_fail_round()` instead (a life
is lost only if one was actually played, `lives <= 0` is checked before the game is
permanently over, comeback bookkeeping applies, and the run proceeds to Cash Out either way).
Only reachable at **ante 1's real vanilla Boss blind** — the Nemesis's own `boss_key` is
always `MLB_NEMESIS_KEY`, never `bl_hook`/`bl_eye`/`bl_mouth` — and only when that boss is
actually drawn AND actually rejects a hand into exhaustion. Observed in ~1/100 agents in the
100-agent smoke population before the workaround below.

**Workaround** (`runner._repair_mlb_gameover_bug`, called after every `game.step()` inside
`_drive_to_next_nemesis`): detects the ONE situation that can never legitimately occur under
`ruleset="mlb"` — `State.GAME_OVER` with `lives > 0` (every other GAME_OVER path in the file
is lives-gated) — and reproduces exactly what `_mlb_fail_round()` would have done, through
the same public `lose_life()` hook the rest of this module already uses. Never edits engine
files. `engine` is frozen for Phase 3; this belongs on the lead's list for a real fix
(replace the three raw `self.state = State.GAME_OVER` assignments with a call through
`_mlb_fail_round`-equivalent logic, gated on `self.mlb`, matching the pattern already used
everywhere else in the file).

## 6. File formats

`Tournament(..., out_dir=...)` (or the CLI's `--out`) writes, under the run directory:

- `ante_<ante:04d>.npz` — one per Nemesis actually played. Arrays: `ante` (scalar), `scores`
  `(n_agents,)` float (`NaN` = agent not present that ante — already eliminated), `outcome`
  `(n_agents, n_agents)` float in `{-1, 0, +1}`, `log_margin` `(n_agents, n_agents)` float
  (`log1p(score_i) - log1p(score_j)`), `rank` `(n_agents,)` float (1 = best; ties share the
  average rank; `NaN` = absent), `losers` — 1-D int64 array of agent indices who lost a life
  at this Nemesis under the active `life_rule`.
- `summary.jsonl` — one JSON object per line (ante order): `ante`, `n_agents`, `n_present`,
  `mean`, `std`, `min`, `max`, `quantiles` (`{"0.0", "0.1", "0.25", "0.5", "0.75", "0.9",
  "1.0"}`), `tie_fraction`, `losers`.
- `meta.json` — run-level: `seed` (normalised seed string), `n_agents`, `life_rule`,
  `max_ante`, `deck_key`, `stake`, `fanout_method`, `wall_clock_s`, `steps_total`,
  `final_lives` (per agent, 0 if dead), `alive_at_end` (surviving agent indices),
  `last_score` (`{agent_idx: [ante, score]}` at each agent's most recent Nemesis).

`matrix.tie_fraction`: the fraction of OFF-DIAGONAL pairs, among agents **present** this
ante, whose outcome is an exact tie — the degeneracy metric (design doc §6). `NaN` when
fewer than 2 agents are present.

## 7. Wall-clock numbers

100 agents (`players.default_population`: a rotating mix of 5 heterogeneous scripted specs
+ 1/3 `RandomLegalPlayer`, distinct seeds), seed `7I4M53DL`, deck `b_red`, stake 1, `clone`
fan-out, this machine:

| run | antes played | wall clock | engine steps | steps/s |
|---|---|---|---|---|
| `--life-rule none --max-ante 8` | 7 (2..8), all 100 present every ante | **22.7 s** | 24 483 | 1 079 |
| `--life-rule paired --max-ante 40` | 2 (last-agent-standing reached: 100 -> 90 -> 41 -> 0) | **6.4 s** | 7 480 | 1 161 |

`paired` with only the default MLB starting lives (4) plus scripted/random players that
regularly fail regular blinds reaches **0 survivors** within 2 Nemeses — a legitimate, if
anticlimactic, "last-agent-standing" outcome (the loop correctly stops the instant `alive` is
empty; see `Tournament.run`'s `while alive and nemesis_ante <= self.max_ante`).
`tests/test_life_rules.py`'s integration test uses `debug_win_regular=True` + generous lives
to isolate the paired mechanism from ordinary attrition when a controlled demonstration is
needed instead.

Per-ante score distribution, seed `7I4M53DL`, `--life-rule none --max-ante 8`, `n=100`,
`base_seed=0` (population as above; "layer 1" from the self-play assessment):

| ante | n present | mean | std | median | tie fraction |
|---|---|---|---|---|---|
| 2 | 100 | 1 019.6 | 717.5 | 1 020.0 | 0.1214 |
| 3 | 100 | 1 796.1 | 2 261.5 | 624.0 | 0.1208 |
| 4 | 100 | 4 210.5 | 5 375.2 | 852.0 | 0.1212 |
| 5 | 100 | 4 522.3 | 6 243.7 | 804.0 | 0.0842 |
| 6 | 100 | 4 080.4 | 6 233.7 | 844.0 | 0.0844 |
| 7 | 100 | 6 192.3 | 8 537.7 | 772.0 | 0.0871 |
| 8 | 100 | 5 498.9 | 7 125.6 | 2 539.0 | 0.0840 |

Degeneracy metric, `n=12`, same seed, `--life-rule none --max-ante 5`
(`identical_population` = 12 copies of one deterministic scripted spec vs
`default_population` = the heterogeneous mix):

| ante | identical tie fraction | heterogeneous tie fraction |
|---|---|---|
| 2 | 1.0000 | 0.1061 |
| 3 | 1.0000 | 0.1061 |
| 4 | 1.0000 | 0.1061 |
| 5 | 1.0000 | 0.0606 |

An identical deterministic population is **exactly** degenerate (tie fraction = 1.0 at every
ante — N clones of one seed with no divergent decisions produce bit-identical trajectories,
hence bit-identical scores, hence every one of the `C(12,2) = 66` pairwise comparisons ties).
A heterogeneous population sits an order of magnitude lower.

## 8. How the MCTS player will plug in (W1 / W3)

`players.py` never imports `agent` (owned by W1/W3, concurrent). Once that package exists:

```python
class MCTSPlayer:
    def __init__(self, model, search_config, ...):
        ...
    def act(self, game) -> dict:
        # game is a live BalatroGame, ruleset="mlb". It may be MID-Nemesis
        # (game.current_blind.is_pvp) with pvp_opponent_score / pvp_opponent_hands
        # UNSET (0) -- the tournament runner is deliberately "blind" (decision 0.2), so
        # do not condition search on those fields; they will read 0 throughout a Nemesis
        # played through this runner. game.legal_actions() is the action space; return
        # one action from it (or an equivalent it accepts).
    def reset(self) -> None:
        ...   # optional: clear any per-episode tree/cache state between Tournament.run() calls
```

Then `players.default_population` / `identical_population` / a new heterogeneity helper can
mix `MCTSPlayer` instances with different checkpoints, search budgets or temperatures — the
design doc's "sample opponents from a pool of historical checkpoints, not N copies of current
weights" — exactly the way `ScriptedPlayer` specs vary today. Nothing else in `runner.py` /
`matrix.py` needs to change: the `Player` protocol (`act(game) -> dict`, optional `reset()`)
is the entire contract, and it is already satisfied by anything that can look at a
`BalatroGame` and return a legal action.

## 9. File map

- `tournament/bootstrap.py` — sys.path + fork-guarded `import_engine()` (mirrors
  `oracle.engine_parity.import_engine`), re-exports `BalatroGame`, `State`,
  `MLB_STARTING_LIVES`, `MLB_PVP_START_ROUND`, `MLB_COMEBACK_PER_LIFE`, `mlb_match_demo`.
- `tournament/conftest.py` — pytest fork-guard, mirrors `engine/conftest.py`.
- `tournament/players.py` — `Player` protocol, `ScriptedPlayerAdapter` /
  `RandomLegalPlayer` / `MCTSPlayer` placeholder, `default_population` /
  `identical_population` heterogeneity helpers.
- `tournament/runner.py` — `Tournament`, `TournamentResult`, fan-out functions +
  benchmark, the per-agent drive loop, the engine-gap workaround, `_pairing`.
- `tournament/matrix.py` — `outcome_matrix`, `log_margin_matrix`, `population_rank`,
  `score_distribution`, `tie_fraction`, `AnteMatrix`, `write_run`.
- `tournament/cli.py` — `python -m tournament.cli ...`.
- `tournament/tests/` — `test_fanout.py`, `test_determinism.py`,
  `test_matrix_properties.py`, `test_life_rules.py`, `test_heterogeneity.py`,
  `test_smoke_100.py`, `test_queue_alignment.py` (31 tests total).
