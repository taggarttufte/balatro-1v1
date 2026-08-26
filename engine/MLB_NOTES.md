# MLB_NOTES — Phase 2 W1: Major League Balatro rules + two-player lockstep coordinator

**Agent W1, 2026-08-21.**  Deliverables: `balatro_sim/mlb_match.py` (new), `balatro_sim/env_mp.py`
(rewritten), `balatro_sim/mp_game.py` (retired → import shim), targeted edits to
`balatro_sim/game.py` and `balatro_sim/constants.py`, tests `tests/engine_tests/test_mlb_match.py`
(59) + `tests/engine_tests/test_env_mp.py` (30, rewritten), this note.  Sources: the installed
BalatroMultiplayer mod v0.5.2 (`$MOD` = `AppData/Roaming/Balatro/Mods/Multiplayer/`) and its
server, **confirmed from GitHub** (`Balatro-Multiplayer/BalatroMultiplayerAPI-Server`,
`src/actionHandlers.ts`, `src/Client.ts`, `src/GameMode.ts`, fetched 2026-08-21 via `gh api`).
No mod/server code is copied anywhere; everything is a port.

## 0. Gates (final run, after W2 + W3 landed in the same tree)

| gate | result |
|---|---|
| `python -m pytest engine/tests -q` | **1609 passed / 10 skipped / 3 xfailed / 0 failed** (floor after W3 was 1584; −62 retired V8-era MP tests, +89 new, the rest W2/W3 additions landing in the same tree) |
| `python -m pytest tests -q` | **544 passed / 2 xfailed / 0 failed** (W2's `test_the_order.py` was red mid-session while they edited `generate.py`; green at hand-off) |
| `python -m oracle.engine_parity --antes 1-8 --rerolls 5 --quiet` | **126/126 EXACT through ante 8** (vanilla byte-identical) |

## 1. Architecture

The real system is client (Lua mod) + server (TypeScript relay that owns lives and the PvP
verdict).  The engine mirrors that split:

* **`BalatroGame(ruleset="mlb")` = the client.**  `ruleset` is a constructor kwarg (default
  `"vanilla"`, which is byte-identical to before: every MLB field is inert/zero, parity 126/126).
  Under `"mlb"`: Attrition bans in `run_state.banned_keys`, `run_state.ruleset = "mlb"` (W2's
  `Voucher0` path), the Nemesis at the Boss slot from ante `pvp_start_round` (2), `lives`,
  `comeback_bonus` / `comeback_bonus_given`, the once-per-round life blocker, failed blinds
  proceed + cost a life, Cash-Out money rules, endless, a new `State.PVP_WAIT`, and three
  "network message" entry points: `lose_life()` (= `playerInfo{lives-1}`), `set_pvp_info(score,
  hands)` (= `enemyInfo`), `end_pvp()` (= `endPvP`).  `pvp_solo` (default True) makes a
  stand-alone MLB game start the Nemesis immediately and end it at exhaustion, so harnesses
  that construct one game still run; `MLBMatch` sets it False.
* **`MLBMatch` = the server.**  Two games on ONE seed (`different_seeds = false`), one deck, one
  stake.  `step(player, action)` → `game.step` → `sync()`.  `sync()` (idempotent; also safe
  after a game was stepped directly — `env_mp` does that) does: `readyBlind`×2 → `startBlind`
  (both `_start_blind()` in index order); `enemyInfo` relay (each Nemesis target = the other's
  `chips_scored`); the `playHand` end-of-PvP rule; `winGame`/`loseGame` at 0 lives.
* **`env_mp.MultiplayerBalatroEnv`** = V7 obs/masks/rewards per player (a V7 env proxied onto
  the match's games) + a 10-feature MP block + MP event rewards.  `OBS_DIM` **451 → 457**
  (`MP_OBS_FEATURES` 4 → 10); the only assertions of it were in `test_env_mp.py` (rewritten).
  Nothing outside `mp/` imports this fork.

## 2. Rules, each with its source

| # | rule | engine | source |
|---|---|---|---|
| 1.1a | ruleset = Attrition gamemode, vanilla content, forced lobby options; `the_order=false`, `starting_lives=4`, `pvp_start_round=2`, `different_seeds=false`, `death_on_round_loss=true`, `gold_on_life_loss=true`, `no_gold_on_round_loss=false`, `hide_score_until_played=false` | `constants.MLB_*`, `MLBMatch.__init__` | `$MOD/rulesets/majorleague.lua:1-28`, `core.lua:169-203`, `play_button_callbacks.lua:115`, server `GameMode.ts` (`attrition: startingLives 4, boss: "bl_pvp"`) |
| 1.1b | bans: jokers Mr Bones/Luchador/Matador/Chicot, vouchers Hieroglyph/Petroglyph/Director's Cut/Retcon, `tag_boss`, `bl_wall`, `bl_final_vessel`, all as `G.GAME.banned_keys[k]=true` | `_init_game_vars`: `run_state.banned_keys \|= MLB_BANNED_KEYS` **before** the run-start draws | `attrition.lua:13-33`; `rulesets/_rulesets.lua:198-229` (`MP.ApplyBans`); applied from `game.lua:2170` (`G:save_settings()` patch, `lovely/game.toml:159-173`) which precedes `get_new_boss()` at `game.lua:2177` |
| 1.1c | ban semantics: pool slot `UNAVAILABLE` + side-stream resample for jokers/vouchers/tags (Phase-1 invariant holds: an unbanned vanilla draw is identical); bosses leave the **eligible set** (so from ante 2 the boss draw can legitimately differ from vanilla because `bl_wall` no longer occupies a list index; at ante 8 `bl_final_vessel` leaves the showdown list) | `generate.get_current_pool`, `generate.eligible_bosses` (already honoured `banned_keys`) | vanilla `common_events.lua:2030, 2359-2361` |
| 1.2a | Boss slot = `bl_mp_nemesis` for ante ≥ `pvp_start_round`; ante 1 boss vanilla | `_prepare_next_blind` (`BlindInfo.is_pvp`, `boss_key = MLB_NEMESIS_KEY`) | `attrition.lua:3-11`, `ui/game/round.lua:54-66` |
| 1.2b | the vanilla `reset_blinds` runs FIRST → the `'boss'` stream is drawn exactly as in single player and `bosses_used` keeps counting; only the displayed choice is replaced | `self.boss_blind` keeps the "shadow" draw; test pins equal `'boss'` stream position | `round.lua:54-57` (`reset_blinds_ref()` then override) |
| 1.2c | Nemesis: `dollars=5`, `mult=1`, no effect, no debuff, `Blind:disable` is a no-op (Chicot) | `BlindInfo.money_reward`, `_apply_boss_start` falls through, `_start_blind` Chicot guard | `objects/blinds/nemesis.lua:19-30`, `ui/game/blind_hud.lua:190-195` |
| 1.2d | target = opponent's live score, visible (no `???` mask under MLB); their hands-left shown | `set_pvp_info` → `current_blind.chips_target`, `pvp_opponent_hands`; reset to 0 at `startBlind` | `blind_hud.lua:168, 197-224`, `action_handlers.lua:332-338, 349-459`; server `actionHandlers.ts:221-300` |
| 1.2e | at the Nemesis the round never ends on `chips ≥ target`; out of hands → wait | `_play_hand` PvP branch → `State.PVP_WAIT` (no legal actions) | `ui/game/game_state.lua:163-236` (replaces `Game:update_hand_played`, vanilla `game.lua:3187-3206`) |
| 1.3a | **PvP end rule (server, confirmed):** evaluated on every `playHand`: ends when `(A out of hands and A.score < B.score)` or `(B out of hands and B.score < A.score)` or `(both out of hands)`; strictly lower score loses ONE life; **exact tie → nobody loses** (both get `endPvP{lost:false}`); the early-ended player's remaining hands are forfeited | `MLBMatch._resolve_pvp` | server `actionHandlers.ts:303-345`; client `action_end_pvp` `action_handlers.lua:472-508` → `game_state.lua:229-233, 313-318` |
| 1.3b | a life is lost at most once per round (`roundLivesBlocker`, reset by `newRound` which the client sends at every `select_blind`) | `lose_life()` + `life_lost_this_round`, reset in `_start_blind` | server `Client.ts:113-134`, `actionHandlers.ts:505-507`; client `ui/game/functions.lua:58-80` |
| 1.3c | 0 lives → `loseGame` (GAME_OVER at once, no Cash Out) for the loser, `winGame` for the other; the match is over | `_mlb_fail_round`, `MLBMatch._check_game_over` (`match_won`, both GAME_OVER) | server `actionHandlers.ts:321-332, 413-455`; client `action_handlers.lua:525-549` |
| 1.3d | regular blind lost → `failRound` → `death_on_round_loss` → a life, unless `hands_used == 0` | `_mlb_fail_round(hands_used)` | client `game_state.lua:245-261`, `action_handlers.lua:1143-1149`; server `actionHandlers.ts:413-418` |
| 1.3e | deck-out: (a) ≥1 hand played, hand+deck empty → `hands_left=0`, regular blind ends (won/lost on chips), Nemesis reports `playHand(chips,0)` and waits; (b) 0 hands played + discards used + empty → round ends "defeated", `fail_round(1)` (a life), Nemesis `playHand(0,0)` | `_mlb_check_deck_out` (after every play / discard) | `game_state.lua:287-311` (`update_selecting_hand`), `game_state.lua:446-459` (`MP.handle_deck_out`, called from `end_round` via `lovely/end_round.toml:52-61`) |
| 1.3f | readiness: "Play" at the Nemesis = `readyBlind`; the blind starts for both on `startBlind` when both are ready; a ready player cannot act | `step(play_blind)` → `pvp_ready`; `MLBMatch.sync` step 1 | `ui/game/functions.lua:14-30`, `action_handlers.lua:319-347`; server `actionHandlers.ts:173-215` |
| 1.4a | **no unused-hand money at a PvP blind** (only the `hands_left > 0` row is patched; the Green Deck `money_per_discard` row is NOT and is still paid) | `_end_round`: `earnings = 0` for `is_pvp`, discard row untouched | `lovely/game.toml:93-100` patching `state_events.lua:1165`; `state_events.lua:1170-1174` unpatched |
| 1.4b | the blind reward is paid at a PvP blind won OR lost ($5) | `_end_round` pays `blind.money_reward` whenever the round reaches ROUND_EVAL | `lovely/game.toml:146-154` patching `state_events.lua:1139` |
| 1.4c | a failed regular blind proceeds to the next blind as if defeated (reward paid, Boss eases the ante) | `_mlb_fail_round` → ROUND_EVAL → vanilla `_end_round` | `game_state.lua:249-250` (`blind.chips = -1`), `lovely/end_round.toml:25-48` (`game_over=false`, `game_won=nil`) |
| 1.4d | interest vanilla (on the pre-row balance); skip vanilla | untouched | — |
| 1.5 | **comeback money:** every life loss (any `playerInfo` decrease) → `comeback_bonus += 1`, `comeback_bonus_given = false`; next Cash Out, right after the interest row and outside the `dollars ≥ 5` guard: `+4 × comeback_bonus`, then `given = true`; initial `given = true, bonus = 0` | `lose_life()`, `_end_round` after interest | `action_handlers.lua:510-523`, `lovely/game.toml:15-49`, `core.lua:215-216` |
| 1.6 | MLB vouchers from the run-global culled `'Voucher0'` stream (shop + Voucher Tag) even with The Order off | `_init_game_vars`: `run_state.ruleset = "mlb"` before `start_run` (W2's generate path; bans honoured there too — 300 seeds, 0 banned vouchers) | `compatibility/TheOrder.lua:481-525` |
| 1.7 | The Order: `run_state.key_scope = game.queue_scope` is set at run start (W2 owns the wiring; MLB forces it off) | `_init_game_vars` | `majorleague.lua:24` |
| — | endless: no ante-8 win, blind scaling continues (`get_blind_amount` formula) | `_advance_blind` no GAME_OVER at ante > 8 under MLB; `_obs().won = match_won` | `game_state.lua:264-276` (`win_ante = 999` around `end_round`), `end_round.toml:30-43` |
| 1.8 | observation: both lives, opponent live score + hands during PvP, comeback pending, waiting flag, opponent location (blind ordinal delta) | `env_mp._mp_features` | brief §1.8; `enemyInfo` carries `score, handsLeft, skips, lives` |

## 3. Assumptions and decisions (read these, W4)

1. **Tie rule = the server's, not the ghost replay's.**  `ghost_replay.lua:140-252` uses `>=`
   (the player reaching the score first wins a tie); the live server (`actionHandlers.ts:320`)
   only takes a life when the scores differ.  Implemented the server rule.  Brief §1.3's "verify
   vs server" is done.
   **2026-08-26 (W-PVP) re-check:** that server citation is a REMOTE one (fetched over the
   network in Phase 2) and **nothing in the local install corroborates it** — there is no
   score comparison in the live-MP client at all, and `ghost_replay.lua:142-168`, the only
   local reimplementation, uses `>=` with no "nobody loses" branch.  The server rule stays
   implemented (`agents.md:58` phrases the loss condition with a strict `<`, which agrees);
   `_resolve_pvp`'s docstring names the line to change if it is ever re-verified.
   `ev/PVP_NOTES.md` §1.4.
2. **Canonical turn order.**  Both players act in real time in the real game.  The outcome of a
   Nemesis (who loses / tie) is invariant to the interleaving: scores only grow, and the rule
   ends the round only when the behind player can no longer improve — pinned by
   `test_outcome_is_independent_of_interleaving`.  What the interleaving DOES decide is where
   the winner's remaining hands are cut (hand/deck contents at Cash Out, unused discards, joker
   round counters, glass rolls…).  `MLBMatch.current_player()` therefore defines a
   deterministic order: strict alternation, action by action, skipping a player who cannot act
   (ready-waiting, `PVP_WAIT`, done).  `step()` is permissive — any player who can act may be
   stepped — so an env that moves both players per step is fine; the canonical order is for
   search/replay.
   **2026-08-26 (W-PVP):** this is now the `pvp_protocol="canonical"` branch and is still the
   default and still byte-identical (pinned step-by-step by
   `engine_tests/test_pvp_protocol.py` against transcripts captured before the change).
   `pvp_protocol="trailer_compelled"` adds one match-level action — the strictly-leading
   player inside a live Nemesis may `{"type": "pvp_pass"}`, i.e. wait — which is the thing
   strict alternation could not express.  It is a MODELLING CHOICE that cannot be
   oracle-verified; the mod-source facts that make it defensible (no clock at a PvP blind
   under MLB, no concede action) and every edge-case decision are in `ev/PVP_NOTES.md` §1-§2.
   End conditions are untouched.
3. **"First to 0 loses."**  When both are at 1 life and both fail the same blind the server
   kills whoever's `failRound` arrives first.  In the engine the first game stepped into its
   fail loses (`test_first_to_zero_loses_even_if_both_are_failing`); with alternation that's
   deterministic but ordering-dependent (env_mp: player 1 acts first in a step).
4. **Comeback timing.**  The `playerInfo` reply to a `failRound` races the Cash Out's
   `evaluate_round`; on a LAN-speed relay it lands first, so the payout is modelled at the SAME
   Cash Out (the brief's "next Cash Out").  The ghost replay does not bump the counter for a
   regular-blind fail; the live path (`action_player_info`) does — live path implemented.
   Still worth Tagg's practice-lobby glance.
5. **Exhausted hand stays in hand until Cash Out.**  The mod evaluates held-in-hand effects
   (Gold card $3, Blue Seal) and discards the hand the moment a player runs out of hands
   (`game_state.lua:203-206`); the engine leaves the hand in place and evaluates it in
   `_end_round`.  Same money / same Blue Seal key, different instant — no opponent action can
   touch a waiting player's hand, so it is unobservable.
6. **`on_boss_beaten` / Investment / Anaglyph fire on a LOST Nemesis too** — the vanilla
   `end_round` / `evaluate_round` run unconditionally in MP (`game_over=false`), and `Rocket`,
   `Campfire` (`end_of_round and G.GAME.blind.boss`), Investment (`last_blind.boss`) and
   `Back:trigger_effect{eval}` all key off "the blind was a boss", which the Nemesis is
   (`boss = {min=1,max=10}`).
7. **The Nemesis' own `G.GAME.blind.chips`** (= `get_blind_amount(ante) × 1`, the Small Blind
   amount) is not exposed; `chips_target` is the opponent's score, as the brief asks.  No vanilla
   joker reads the blind's chip count.
8. **Deck-out at a Nemesis** (rule 1.3e-b): the client sends `playHand(0,0)` then
   `fail_round(1)`; on the server the blocker makes that one life, whichever message lands
   first.  Engine: `lose_life()` then `PVP_WAIT`; the later PvP verdict is blocked.
9. **The ante-2+ shadow boss can differ from vanilla** (rule 1.1c).  This is the real game's
   behaviour (the ban shrinks the candidate list), not a parity bug; the `'boss'` stream
   position is identical and pinned.
10. **Lives live on the game** (`game.lives`), the match only coordinates.  `lives` kwarg on
    `MLBMatch` / `MultiplayerBalatroEnv` (default 4) sets both.

## 4. Not modelled (gaps)

* Timers (`timer_base_seconds`, `pvp_timer`, `failTimer`, speedrun grants) — no clock in the
  engine.  `endPvP{pvpTimerLost}` → `mp_pvp_loss` joker context is therefore unreachable.
* MP-only content (Asteroids, Pizza, Phantom, Magnet, Conjoined…) — `multiplayer_content = false`
  under MLB, nothing to do.
* Disconnect / reconnect / stopGame; `furthest_blind` (Survival only); end-game joker/deck
  exchange; replay log.
* The `hide_score_until_played` branch (off under MLB) is not implemented.
* `skips` are in `PlayerView` but not in the MP obs block (`opp_progress` is); add if a policy
  wants it.
* Ghost-replay semantics (`>=` tie) are not offered as an option.

## 5. Found, not fixed / fixed out of scope

* **Fixed (clone machinery, needed for clone fidelity):** `BalatroGame.clone()` copied
  `_played_this_ante` (The Pillar) and `_forced_card_id` (Cerulean Bell) as the ORIGINAL card
  ids, but `Card.copy()` mints fresh ids — a clone silently lost both effects and its
  `state_signature()` differed from the original's.  Now remapped through `by_id`
  (`game.py` `clone()`, 4 lines).  `tests/sim_tests/test_clone_deck_identity.py::
  test_audit_added_state_survives_clone` compared raw id sets (i.e. asserted the bug) and now
  compares the cards named; `tests/sim_tests/test_sweep.py::TestEnvRevival` used the removed
  `env_mp._revive_boss_if_needed` and now exercises the native MLB lost-boss path (same
  ground-truth shelf/boss assertion; voucher dropped because MLB vouchers are `Voucher0`).
* **Not fixed (vanilla):** a vanilla game whose hand AND draw pile are empty in
  `SELECTING_HAND` just sits there with no legal actions (the real game calls `end_round()`
  from `update_selecting_hand`, `game.lua:3056-3065`).  Reachable only with a ≤ 8-card deck
  after discards; the MLB path handles it (`_mlb_check_deck_out`), vanilla deliberately left
  byte-identical.
* `env_v7._finish_step` pays `R_BLIND_BASE * (9 - ante)` per cleared blind — negative past
  ante 8 in an endless MLB match (V7 heuristic reward, env_mp inherits it; harmless for now).

## 6. Running the tests

```
cd C:/Users/Taggart/projects/balatro-rl
python -m pytest engine/tests/engine_tests/test_mlb_match.py engine/tests/engine_tests/test_env_mp.py -q   # 89, ~1.5 s
python -m pytest engine/tests -q                                  # full engine suite
python -m pytest tests -q
python -m oracle.engine_parity --antes 1-8 --rerolls 5 --quiet
```
Retired: `tests/engine_tests/test_mp_game.py`, `tests/engine_tests/test_mp_integration.py` (V8
rules).  `mp_game.MultiplayerBalatro` raises `ImportError` with a pointer.

## 7. For W4 — driving a full match

```python
from balatro_sim.mlb_match import MLBMatch
from balatro_sim.game import State

m = MLBMatch(seed="7I4M53DL", deck_key="b_red", stake=1, lives=4)   # pvp_start_round=2
while not m.done:
    p = m.current_player()                 # canonical order; None only when done
    acts = m.legal_actions(p)              # == m.games[p].legal_actions() minus match gating
    m.step(p, policy[p](m, p, acts))       # any action dict BalatroGame.step accepts
print(m.winner, m.pvp_log)                 # pvp_log: (ante, loser|None, score0, score1) per Nemesis
# or: m.play_out([policy0, policy1], max_steps=...) -> MLBMatchState
```

* `m.games[p]` is the player's `BalatroGame`; everything you know from Phase 1 works on it
  (`debug_win_blind`, `state_signature`, `run_state`, `current_shop`, `clone`).  After stepping
  a game directly call `m.sync()`.
* `m.state()` → `MLBMatchState` (per-player `PlayerView`: ante, blind, state, lives, skips,
  dollars, score, hands, ready/exhausted flags, comeback pending).  `m.signature()` is the
  hashable snapshot for determinism/clone tests (`test_random_match_is_deterministic`).
* States you will see per game: the Phase-1 six plus `PVP_WAIT`.  A player readied at the
  Nemesis stays in `BLIND_SELECT` with `pvp_ready=True` and `legal_actions() == []`.
* Exit-gate item 1: lives decrement on lost regular blinds (`game.lives`), comeback lands at
  the next Cash Out (`4 × game.comeback_bonus`, check `dollars` across `advance`), the match
  ends at 0 lives (`m.done`, both games `GAME_OVER`, winner's `match_won`), the early-end fires
  when one player is out of hands and strictly behind (`m.pvp_log` + the winner's `hands_left > 0`
  at `ROUND_EVAL`).
* Exit-gate item 2 (queue alignment): with no buys/rerolls the two players' shelves, vouchers
  and shadow bosses are identical at every shop (`test_shop_queues_stay_aligned_without_purchases`);
  every difference after that must be a reroll (`run_state.rng` key `Joker1sho<ante>` etc.
  advanced further) or an in-place resample caused by that player's own `used_jokers`/
  `used_vouchers` — compare `run_state.rng.snapshot()["state"]` key positions between the two
  games, not the shelves by eye.
* Exit-gate item 3: `ruleset="vanilla"` is the parity path (126/126 above).  `ruleset="mlb"`
  vs vanilla on one seed differs in: vouchers (`Voucher0`, W2), bans, the ante ≥ 2 Boss slot, and
  — because of the `bl_wall` ban — possibly the shadow boss key from ante 2 (the `'boss'`
  stream position is equal; see §3.9).  `test_boss_stream_still_drawn_behind_the_nemesis` is
  the template for that assertion.
* env route: `MultiplayerBalatroEnv(seed, lives, deck_key, stake)`; `step(a1, a2)` applies
  both (player 1 first), returns `info["pvp_log"]`, `info["p{1,2}_mp_reward"]` (the MP event
  component of the reward, separated from V7's heuristics), `info["winner"]` (1/2).

## 8. Files touched

`balatro_sim/game.py` (targeted edits: constructor `ruleset`, `_init_game_vars` MLB block + W2
flags, `State.PVP_WAIT`, `BlindInfo.is_pvp`/`money_reward`, `clone()` (+ id remap),
`_prepare_next_blind`, `can_reroll_boss`, `_start_blind`, `_play_hand` tail, new
`lose_life`/`_mlb_fail_round`/`_mlb_check_deck_out`/`set_pvp_info`/`end_pvp`, `_discard` tail,
`_end_round` money, `_advance_blind`, `step` play_blind gating, `legal_actions`, `_obs`),
`balatro_sim/constants.py` (appended `MLB_*`), `balatro_sim/mlb_match.py` (new),
`balatro_sim/env_mp.py` (rewritten), `balatro_sim/mp_game.py` (shim), tests as in §6 plus the
two sim_tests edits in §5.
