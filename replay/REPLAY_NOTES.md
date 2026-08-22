# REPLAY_NOTES — Phase 4 W3: trajectory logging + replay + tagging + viewer export

**Agent W3, 2026-08-22.** Package: `mp/replay/**` (new). Engine-only (never imports `mp/agent`
torch code or `mp/tournament`). `mp/engine/**`, `mp/rng/**`, `mp/agent/**`, `mp/tournament/**`,
`mp/eval/**` were read, never edited. No engine bug was found (see "Needs-engine-change" at the
end — empty, one design tradeoff noted instead).

## 0. File map

```
mp/replay/
  __init__.py         package docstring / marker
  _bootstrap.py        sys.path + fork-guarded engine import (mirrors mp/tournament/bootstrap.py)
  conftest.py           test bootstrap (mirrors mp/tournament/conftest.py)
  _util.py              summarize(), sig_digest(), match_sig_digest(), apply_op(), ReplayMismatch
  log.py                TrajectoryLogger, MatchLogger
  replay.py             replay(), replay_match(), replay_line(), narrate(), verify_file()
  tags.py               tag_episode(), tag_match(), interest_score(), tag_file()
  export_viz.py         export_viz(), export_viz_match(), export_viz_to_file()
  cli.py                python -m mp.replay.cli {show,verify,filter,stats,tag,export-viz}
  tests/
    _helpers.py          RandomLegalPlayer + run_logged_episode/run_logged_match (the hook
                          contract's own proof: these ARE the ≤3-line integration)
    test_log_replay.py    round trip (20 seeds x vanilla/MLB + MLBMatch), corruption detection,
                          overhead, bytes/episode
    test_tags.py          all tag predicates + tag_file()
    test_narrate.py        narrate() non-empty, mentions every ante
    test_export_viz.py     viz export shape + JSON validity
    test_cli.py            every subcommand, in-process
```

## 1. Format spec (v1, versioned via top-level `"v": 1`)

One JSONL line per episode/match. `kind` distinguishes the two shapes.

### 1.1 `kind: "episode"` (one `BalatroGame`, vanilla or MLB solo)

```jsonc
{
  "v": 1, "kind": "episode",
  "seed": "7I4M53DL", "deck_key": "b_red", "stake": 1, "ruleset": "mlb", "lives_start": 4,
  "meta": {},                          // whatever begin() was given, verbatim
  "actions": [ {"type": "skip_blind"}, {"type": "play", "cards": [0,2,4]}, ... ],
  "steps":   [ {"step":0,"state":"BLIND_SELECT","ante":1,"blind_kind":"Small","is_pvp":false,
                "money":4,"lives":4,"chips_scored":0,"hands_left":4,"discards_left":3}, ... ],
  "sig_every": 10,
  "signatures": {"start": "<32-hex>", "0": "<32-hex>", "10": "<32-hex>", ..., "final": "<32-hex>"},
  "outcome": {},                        // whatever end() was given, verbatim (see §2 "win" note)
  "final_state": {"state":"GAME_OVER","ante":8,"lives":2,"money":37,"joker_count":3,
                  "jokers":["j_hologram","j_green_joker","j_blue_joker"],"consumables":[]},
  "tags": []                            // filled in place by tags.tag_file()
}
```

`actions[i]` is the EXACT dict passed to `step()`. `steps[i]` is the 10-field summary from
PHASE4_BRIEF §W3, captured AFTER `actions[i]` is applied (same instant). `signatures` keys are
step indices as strings (JSON object keys are always strings) plus `"start"` (before any
action) and `"final"`; a signature is a blake2b-16 digest of `repr(game.state_signature())`,
not the raw signature (which is a full-run snapshot — deck, shop, jokers, rng state — and would
dominate the file size at any reasonable `sig_every`). `final_state` is read straight off the
live game object at `end()` time (not something the caller has to compute) so `tags.py`'s
`no_build`/`archetype_novel`/`lives_lost_n` work without the caller doing anything extra.

**Synthetic ops.** An action's `"type"` can be a real `BalatroGame.step()` action, or the one
synthetic marker `"__lose_life__"` — recorded via the SAME `log.step(game, action)` call, for
an engine mutation an orchestrator applies directly (not through `step()`). Today the only
caller of this is the tournament's cross-agent life rule (see §2 "Tournament wiring"). On
replay, `_util.apply_op()` dispatches `__lose_life__` to `game.lose_life()` instead of
`game.step()`; every other type goes through `step()` unchanged.

### 1.2 `kind: "match"` (one `MLBMatch`, two `BalatroGame`s)

```jsonc
{
  "v": 1, "kind": "match",
  "seed": "7I4M53DL", "deck_key": "b_red", "stake": 1, "lives_start": 4, "pvp_start_round": 2,
  "meta": {},
  "ops":   [ {"player": 0, "action": {"type": "play_blind"}}, {"player": 1, "action": {...}}, ... ],
  "steps": [ {"step":0,"player":0,"players":[<p0 10-field summary>,<p1 10-field summary>]}, ... ],
  "sig_every": 10,
  "signatures": {"start": "...", "0": "...", ..., "final": "..."},
  "pvp_log": [[2, 1, 300, 450], ...],    // (ante, loser|null, score0, score1) -- straight from MLBMatch.pvp_log
  "outcome": {"winner": 0},
  "final_state": {"winner": 0, "players": [{"lives":0,"ante":6,"money":0,"joker_count":0,"jokers":[]},
                                            {"lives":2,"ante":8,"money":37,"joker_count":3,"jokers":[...]}]},
  "tags": {"0": [], "1": []}
}
```

`ops` is a SINGLE interleaved list, in the exact order `match.step(player, action)` was called
— not two independent per-player action lists. This matters: `MLBMatch.sync()` runs after every
`step()` and its side effects (`readyBlind`→`startBlind`, `enemyInfo` relay, the PvP end check)
depend on that interleaving, so replay reconstructs the match by calling `match.step(p, a)` in
the SAME order, never by replaying each side separately. `signatures` are digests of
`MLBMatch.signature()` (both games' `state_signature()` + match scalars + `pvp_log`, already
folded together by the engine).

### 1.3 Signature checkpoint policy

`sig_every` (default 10): a digest is captured at step index 0, `sig_every`, `2*sig_every`, ...,
PLUS always at `"start"` (before any action — pins down whether a divergence is in
construction/seed reproduction vs the first action) and always at `"final"`. A zero-action
episode still gets a `"final"` digest (copied from `"start"`).

## 2. Hook contract

Exactly 3 call sites; `step()`/`step()` is reused once per actual state transition — this is
the whole integration:

```python
log = TrajectoryLogger(path)              # or MatchLogger(path) for an MLBMatch
log.begin(game, meta={...})
...
game.step(action)          # <- caller's existing line, unchanged
log.step(game, action)     # <- 1 line added right after EVERY game.step() call
...
log.end(game, outcome={"won": ..., ...})  # outcome dict is passed straight through
```

For `MatchLogger`: `mlog.begin(match, meta=...)` → `match.step(p, action); mlog.step(match, p,
action)` (in the SAME order the driver calls `match.step`) → `mlog.end(match, outcome={"winner":
match.winner, ...})`.

**"win" tag needs `outcome["won"]`** (episode) / `outcome["winner"]` (match) — `tags.py` never
guesses this from engine state (MLB is endless; "won" only means something the caller's own
stop condition defines), so callers who want the `win` tag to fire MUST pass it.

**`log.step()` must be called after LITERALLY every `game.step()`/`match.step()` call in the
episode** — the agent's own decisions, a cash-out `advance`, any other internal `step()` call —
with no omissions. Replay re-runs `actions`/`ops` verbatim through the same engine entry point;
skipping an internal step breaks index alignment between the logged line and what actually
happened, and any recorded signature after that point will (correctly) fail to verify.

### 2.1 `mp/agent/train/loop.py::ColdTrainer.run_episode` (read-only; W2 owns the real wiring)

`run_episode` (loop.py:140) calls `self.agent.play_episode(game)`, which loops internally in
`agent/train/agent.py::SelfPlayAgent.play_episode` (`legal = game.legal_actions()` at
agent.py:134, `game.step(action_from_key(chosen))` at agent.py:168). The 3-line hook goes around
that inner loop: `log.begin(game, meta={"ep": ep, "seed": episode_seed})` before it starts,
`log.step(game, action_from_key(chosen))` right after agent.py:168's `game.step(...)`, and
`log.end(game, outcome={"won": result.won, "final_ante": result.final_ante, "stop_reason":
result.stop_reason})` after `play_episode` returns. `SelfPlayAgent` is W1/W2's package (torch);
`mp/replay` is never imported from inside it — the CALLER (`run_episode`, or whatever W2's
tournament-driven loop replaces it with) constructs the logger and passes it in, or wraps the
call the way `tests/_helpers.py::run_logged_episode` does.

### 2.2 `mp/engine/balatro_sim/mlb_match.py::MLBMatch` (frozen; read-only)

`MLBMatch.step(player, action)` (mlb_match.py, `def step`) is the only place a match advances.
Wrap it: `match.step(p, action); mlog.step(match, p, action)`, called from whatever drives the
match (a training loop, `play_out`'s `policies[p](...)` driver, or a test). `match.signature()`
already exists and is exactly what `match_sig_digest()` hashes.

### 2.3 `mp/tournament/runner.py::Tournament` (frozen; read-only) — the one with a wrinkle

The tournament drives N INDEPENDENT `BalatroGame`s (not an `MLBMatch`); life loss is decided
externally by comparing all N agents' scores, then applied with a DIRECT `games[i].lose_life()`
call — not through `step()`. Concretely (line references as read 2026-08-22):

* `_drive_to_next_nemesis(game, player, max_steps)` (runner.py, `def _drive_to_next_nemesis`):
  the per-agent loop is `a = player.act(game); game.step(a); _repair_mlb_gameover_bug(game)`.
  Add one line: `log.step(game, a)` right after `game.step(a)`. (§7's cleanup item removes
  `_repair_mlb_gameover_bug` as dead code — once gone, nothing else here calls `step()` outside
  the shown line.)
* `_cash_out(game)` (runner.py, `def _cash_out`): `game.step({"type": "advance"})` — this is a
  REAL action, log it the same way: `log.step(game, {"type": "advance"})`.
* `Tournament.run()`'s life rule (runner.py, inside `for i in losers: games[i].lose_life()`):
  this is the ONE place a life is lost WITHOUT a `step()` call. Emit the synthetic op right
  after it: `log.step(games[i], {"type": "__lose_life__"})`. Skipping this line is the one way
  a tournament-driven trajectory log would silently go incomplete — `game.lives` would still be
  right (it's a live object field), but the ACTION LIST would be missing the event, and a later
  signature checkpoint would (correctly) fail to verify on replay, because `game.lives` is part
  of `state_signature()`'s scanned scalars.
* One `TrajectoryLogger` per agent, `begin()`ed once before the tournament's main `while alive`
  loop and `end()`ed when that agent dies or the tournament ends — a "trajectory" here spans
  the WHOLE multi-ante tournament run for that agent, not one ante.

W2 owns the actual edit (`mp/tournament/**` is frozen for W3); this is the wiring recipe for
whoever lands it.

## 3. CLI

```
python -m mp.replay.cli show <file> <idx>                    # narrate one line
python -m mp.replay.cli verify <file>                          # replay every line, report mismatches
python -m mp.replay.cli filter <file> --tag X [--min-interest F]
python -m mp.replay.cli stats <file>                            # tag counts, ante histogram, skip rate
python -m mp.replay.cli tag <file>                              # tag_file(): retag in place
python -m mp.replay.cli export-viz <file> <idx> <out.json> [--player 0|1]
```

`verify` exits 1 if any line fails to replay clean (a mismatch prints the line index and the
`ReplayMismatch`). `filter`/`stats` treat a match line's tags as `p{0,1}:<tag>` so `--tag win`
also matches `p0:win`/`p1:win`. Run from the repo root; `-m mp.replay.cli` works because `mp/`
has no `__init__.py` (an implicit namespace package under the repo-root cwd Python puts on
`sys.path` for `-m`) while `mp/replay/` itself IS a regular package (`__init__.py` present) —
same pattern `mp/oracle/engine_parity.py`'s own `python -m mp.oracle.engine_parity` usage line
relies on.

## 4. Tag definitions (`tags.py`)

| tag | condition |
|---|---|
| `win` | `outcome["won"]` truthy (episode) / `outcome["winner"] == player` (match) — caller-supplied, see §2 |
| `reached_ante_{k}` | `k` in `{1,2,3,4,5,6,7,8,10,12,16,20,24,32} ∩ [1, max_ante]`, PLUS always the exact max ante reached even if it isn't a milestone |
| `skip_heavy` | of the `play_blind`/`skip_blind` decisions, the skip fraction ≥ 0.5 |
| `no_build` | ≤ 1 joker at `final_state` |
| `comeback` | lives hit 1 at some step AND a LATER step reached ante ≥ (that step's ante) + 2 |
| `lives_lost_{n}` | `n = lives_start - final lives`, `n > 0`; never emitted when `lives_start == 0` (vanilla) |
| `archetype_novel` | final joker-set signature's frequency RANK (over the whole file, both players' builds for match lines) is outside the top 20 (`ARCHETYPE_TOP_N`) — ONLY set by `tag_file()` (a per-line `tag_episode()` call has no corpus to be novel against) |
| `interest_score` | not a tag string, a float alongside `tags`: `min(max_ante,24)/24 + 1.0·win + 0.5·comeback + 0.5·archetype_novel + 0.3·min(joker_count,5)/5 − 0.3·skip_heavy − 0.2·no_build`, clamped ≥ 0. A heuristic for sorting "interesting" episodes, not a scientific measure — documented as such in the module docstring. |

`tag_file(path)` retags every line in place (atomic replace via a temp file + `os.replace`):
recomputes every pure tag AND rebuilds the corpus-wide archetype-frequency table across the
whole file in one pass, so `archetype_novel` only ever needs a fresh full-file scan, never a
persisted sidecar counter.

## 5. Viz export coverage (`export_viz.py`)

Confirmed against the shipped `viz/trajectory.json` + `viz/main.js` (repo root, read-only,
never modified): `{"seed", "outcome": {"ante","reward","steps","dollars","won"}, "episode_id",
"trajectory": [...]}`, one `trajectory[i]` per action, holding the state BEFORE that action is
applied (confirmed by inspecting the shipped file, not assumed).

**Maps directly:** step/phase/ante/blind_idx/money/chips_scored/hands_left/discards_left/
deck_size/hand_size, `blind{name,kind,target,is_boss,boss_key}` (boss name looked up via
`game_keys.BOSS_NAME`), `hand_cards[]`/`jokers[]` (names via `game_keys.JOKER_NAME`)/`shop[]`
(names via the matching `game_keys.*_NAME` table)/`consumables[]`/`planet_levels`, `outcome`.

**Best-effort:** `action` — re-encoded into the SAME two shapes the viewer switches on
(`"hand"`: `intent`+`subset` for play/discard; `"phase"`: `name`+an `action` int, where only
`buy` gets a real int, `item_idx+2`, matching `main.js`'s `chosenIdx = act - 2` shop-highlight
logic — everything else gets `action: null` and relies on the viewer's existing `action.name`
fallback display path).

**NOT derivable:** `value_estimate`, per-step `reward`, `top_probs` — these come from the
agent's MCTS/policy, which an engine-only trajectory log never has. Always written as
`0.0`/`0.0`/`[]` (the viewer's own `??`/`|| []` fallbacks render fine with these; the value
panel and probability bars are just not meaningful for a replayed line).

**Unrendered phases:** `ROUND_EVAL`/`BOOSTER_OPEN`/`PVP_WAIT` have no branch in `main.js` (it
only special-cases hand/shop/blind_select/game_over) — they still export with a best-guess
phase string and the viewer shows the generic step info with an empty stage area, not broken.
MLB-only fields (lives, is_pvp, comeback) have no V7 UI at all; use `cli.py show`/`narrate()`
for MLB-aware text narration instead. `export_viz_match(line, player)` projects one player's
side of a match line into the same single-player shape (the viewer has no two-board mode).

## 6. Ghost replay — feasibility (investigation, NOT built; mod code read, never copied)

Read (read-only): `$MOD/lib/replay_log.lua` (228 lines), `$MOD/lib/ghost_replay.lua` (415
lines), `$MOD/lib/log_parser.lua` (445 lines), plus a 45-line peek at `$MOD/lib/insane_int.lua`
to resolve the one blocking unknown (the score string format — see below). `$MOD/replays/` is
empty (`.gitkeep` only, no recorded games yet on this machine).

**What a ghost replay actually IS.** It is NOT a recording of the opponent's full run (deck,
shop, jokers) that gets re-simulated — it is a data table that feeds the SAME live-score-ticker
UI real MP already has. During ghost playback, `MP.GHOST.get_enemy_hands(ante)` pulls
`replay.ante_snapshots[ante].hands` filtered to `side == "enemy"`, and the only two fields
`ghost_replay.lua` actually consumes to resolve a Nemesis blind
(`resolve_pvp_hands_exhausted`/`resolve_pvp_mid_hand`/`advance_hand`) are, per hand: `score`
(an `MP.INSANE_INT`-encoded string) and `hands_left`. Everything else in the replay JSON
(`player_jokers`, `nemesis_jokers`, `shop_spending`, `cards_bought/sold/used`, `player_stats`)
is DISPLAY-ONLY metadata read by the replay-picker UI (`build_label`, etc.), never consulted
during actual PvP resolution.

**Score string format** (`insane_int.lua::from_string`/`to_string`): `"<coefficient>e<exponent>"`
with optional leading `"e"` characters for an extra magnitude tier (`e_count`, for numbers
Talisman-scale absurd); a PLAIN decimal string with no `"e"` parses as `exponent=0, e_count=0`
— i.e. `tostring(143357)` round-trips correctly. Every chip score our engine produces in the
ante range we care about (1-8, and MLB's endless antes well past that before floats blow up
— see the overhead section's aside) fits as a plain decimal string; no special encoding is
needed unless someone wants to replay a genuinely absurd endless run.

**Minimal field list to write a ghost-playable replay from one of our trajectory lines**
(the mod's JSON loader, `ghost_replay.lua::load_json_replay`, accepts any `.json` under
`$MOD/replays/` with an `ante_snapshots` table — string or numeric ante keys both work, it
normalizes with `tonumber(k) or k`):

```jsonc
{
  "gamemode": "gamemode_mp_attrition",       // display only
  "ruleset": "ruleset_mp_blitz",             // display only, must be a MP.Rulesets key or is_ruleset_supported() rejects it
  "seed": "7I4M53DL", "deck": "Red Deck", "stake": 1,
  "final_ante": 8, "winner": "player" | "nemesis",   // FROM THE LIVE PLAYER'S SEAT: our agent is "nemesis" here
  "player_name": "Tagg", "nemesis_name": "agent v42", "timestamp": 0,
  "ante_snapshots": {
    "2": {
      "player_score": "0", "enemy_score": "450", "player_lives": 3, "enemy_lives": 4,
      "result": "loss",                       // display only
      "hands": [ {"score": "120", "hands_left": 3, "side": "enemy"},
                 {"score": "310", "hands_left": 2, "side": "enemy"},
                 {"score": "450", "hands_left": 0, "side": "enemy"} ]
    },
    "3": { ... }
  }
}
```

Only `hands[].{score,hands_left,side="enemy"}` per Nemesis ante is load-bearing for actual
gameplay; the rest keeps the replay picker UI happy and is trivially derivable from what we
already log (`final_state`, `outcome`, `seed`/`deck_key`/`stake`).

**Building it from our TrajectoryLogger line, mechanically:** filter `steps` to entries where
`is_pvp` is true, group by `ante`, and for each play-type step in a group emit one
`{"score": str(chips_scored), "hands_left": hands_left, "side": "enemy"}` — this is exactly
the same per-step summary field set `log.step()` already records, no new instrumentation
needed. `player_score`/`enemy_score`/`player_lives`/`enemy_lives`/`result` per ante come from
comparing the two sides' final chips at that ante (for a solo trajectory there is no real
"player" side — those fields would need to be synthesized, e.g. zeroed/omitted, since a solo
MLB run's Nemesis is `pvp_solo=True` and never sees an opponent at all; an `MLBMatch` line
already has BOTH sides for free).

**Feasibility verdict: HIGH for the score-ticker mechanism, MEDIUM overall.** The live PvP
resolution only needs the per-hand score/hands_left sequence, which we already log verbatim.
The remaining work (not done here, per the brief — investigation only) is a small converter
(`kind:"episode"`/`kind:"match"` line → the JSON shape above) plus verifying `MP.Rulesets`
actually has an entry matching our `ruleset="mlb"` runs (`is_ruleset_supported()` rejects
unknown ruleset keys) and that the mod's `replays/` folder + `ghost_replay_picker.lua` UI
accept a hand-authored `.json` the same way it accepts a real recorded one (their loader path
is identical — `MP.GHOST.load_folder_replays()` treats `.json` files as first-class, not just
`.log` files, so this looks straightforward but was not actually tried in-game). What ghost
mode CANNOT give Tagg: a full recreation of the agent's actual deck/build/shop decisions —
only its Nemesis-blind scoring performance, ante by ante. Full-run playback would need the mod
to consume actual card/joker/shop state per hand, which it does not do today.

## 7. Measured

**Bytes/episode** (`TrajectoryLogger`, MLB, random-legal player, up to 800 steps, 15 seeds):
9,352–36,766 bytes, mean 16,106. **Bytes/match** (`MatchLogger`, up to 3,000 combined ops, 8
seeds): 37,527–46,190 bytes, mean 42,471. Scales with episode length (the `actions`/`steps`
arrays dominate; `signatures` values are fixed-size 32-hex digests regardless of how deep the
run goes) — "a few KB" for short/typical random-legal episodes, tens of KB for long ones.

**Logging overhead.** `log.step()`'s OWN bookkeeping (copying the action dict, building the
10-field summary, list appends) is negligible: < 2% of a bare `game.step()`, measured
(`test_logger_bookkeeping_overhead_excluding_signatures`). The other component,
`state_signature()` capture every `sig_every` steps, is NOT cheap — it is a full canonical
snapshot of the whole run (game.py's own docstring: "covers every scalar, the blind, hand
levels, jokers ..., the deck composition and the per-blind partitions, the shop / open pack,
tags, and a hash of the keyed PseudoRandom's full state"), roughly comparable to (sometimes
several times) a `step()`'s own cost, and it GROWS as the deck/joker/shop grow. Measured on 10
seeds of random-legal MLB play: `sig_every=10` (the brief's default) costs ~25-35% wall clock;
`sig_every=50` ~8%; `sig_every=100` ~5%; `sig_every=200` ~4%. **Recommendation:** keep the
`sig_every=10` default for short/debug/eval logging where fine-grained divergence
localization matters; pass a larger `sig_every` (50-100) for a long training loop that logs
every episode and cares about raw throughput. This is a real, load-bearing design tradeoff
(not a bug) — the alternative (a cheaper approximate signature) would weaken the "assert every
logged signature matches" replay guarantee the whole package is built around, so it isn't
taken here.

**Aside — endless-MLB float overflow (found, NOT fixed, not this package's bug):** driving an
MLB game with an artificially large `lives` override past roughly ante 40-100 raises
`OverflowError` inside `constants.get_blind_amount` (`float ** float` blows up for the
endless-ante scaling formula at extreme ante). This only triggers with an unrealistic
"lives=100000, let every blind fail" stress pattern (used transiently while measuring the
overhead numbers above, not something a real agent or the tournament would do at these speeds)
— noted here for whoever eventually pushes a real MLB run past ante ~40, not filed as a Phase 4
blocker.

## 8. Gates (repo root, `python` 3.13, 2026-08-22)

| gate | result |
|---|---|
| `python -m pytest mp/replay/tests -q` | **82 passed** |
| `python -m pytest mp/engine/tests -q` | **1614 passed / 10 skipped / 3 xfailed** (unchanged) |
| `python -m pytest mp/tests -q` | **1073 passed / 2 xfailed** (unchanged) |
| `python -m mp.replay.cli verify <a log this package produced>` | clean (manually exercised on episode + match logs over `7I4M53DL`/`ALEEB`, see the demo run in the session transcript) |

## 9. Needs-engine-change

None. The one real tradeoff found (signature-capture cost, §7) is a property of
`state_signature()` being a deliberately complete snapshot, not a bug — no change requested.
