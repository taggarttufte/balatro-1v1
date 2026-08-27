# G2 DESIGN — the live mirror: the agent plays the seed against you, in the real game

Session 2026-08-27 (same session as G1). Brief: `docs/GHOST_MOD_BRIEF_2026-08.md` §G2.
Status of this doc: protocol + semantics FROZEN before build; the two "recon" sections are
filled from the engine/mod recon agents before any code is written against them.

## 0. The shape

```
 Balatro (real game)                          Python sidecar (this repo)
 ┌──────────────────────────┐                 ┌──────────────────────────────┐
 │ Multiplayer mod (theirs)  │   outbox.jsonl │ ghost/live.py                 │
 │  practice + ghost scaffold│  ────────────► │  mirror: solo-MLB BalatroGame │
 │ GhostRace mod (OURS)      │                │  + EVPlayer(spec) on the SAME │
 │  wraps MP.GHOST.* in live │  ◄──────────── │  seed; human's scores injected│
 │  mode; owns resolution    │   inbox.jsonl  │  as the external opponent     │
 └──────────────────────────┘                 └──────────────────────────────┘
```

- **Coupling is blind-level only** (the brief's architecture decision): per-Nemesis final
  scores + life outcomes cross the boundary. The agent's within-blind play in v1 is
  canonical/level-0 — it does not read the human's live mid-blind score, so its whole
  Nemesis round can be computed the moment it reaches that ante (minutes before the human
  does). Latency is therefore invisible: every reveal the mod needs is already buffered.
- **Zero new UI.** The sidecar writes a launcher ghost `.json` (G1 format + a `_live` key)
  into `$MOD/replays/`; Tagg loads it through the existing Match Replays picker. Our mod's
  wrapped `MP.GHOST.load` sees `_live` and switches that run to live mode.
- **We own resolution in live mode** (server rule: strict comparison, exact tie takes
  nobody), so G1's exhaustion index-lag bug cannot occur, and reveals can animate freely.

## 1. Files & session bootstrap

1. `python -m ghost.live --spec ev:fast [--seed X]` →
   session dir `ghost/runs/live/<seed>_<ts>/` containing `outbox.jsonl` (mod→sidecar,
   created empty), `inbox.jsonl` (sidecar→mod), `session.log`.
2. Sidecar writes `$MOD/replays/live_<seed>.json`: the G1 display fields (picker preview,
   `nemesis_name = "<spec> LIVE"`, `ante_snapshots: {}`) plus
   `"_live": {"protocol": 1, "outbox": "<abs>", "inbox": "<abs>"}`.
3. Sidecar starts tailing `outbox.jsonl` and pre-plays the agent to its first Nemesis
   gate; it can already emit `agent_nemesis` for ante 2 before the human even loads.
4. Tagg: Practice → Match Replays → the `LIVE` entry → Play Match. Wrapped `load` arms
   live mode; mod appends `session_start` and begins polling `inbox.jsonl` (~0.3 s timer).
5. Stale-session guard: `session_start.ts` must be newer than the sidecar's launch; a
   second `session_start` (rerun of the same launcher) resets the mirror to a fresh run —
   the mirror is deterministic in (seed, spec), so replaying the outbox rebuilds any state.

Both files are append-only JSONL; both sides may re-read from byte 0 to recover after a
crash (all messages are idempotent under ante keys).

## 2. Protocol v1 (field `e` = event; every message carries `ts`)

mod → sidecar (`outbox.jsonl`):

| e | fields | when |
|---|---|---|
| `session_start` | seed, deck, stake | wrapped load() arms live mode |
| `nemesis_start` | ante, lives | the PvP blind begins (wrapped init_playback, pvp only) |
| `pvp_hand` | ante, score, hands_left | after each human hand at the Nemesis |
| `pvp_result` | ante, human_score, agent_score, loser ("human"/"agent"/null), human_lives | our resolution decided (informational; sidecar cross-checks) |
| `round_fail` | ante, lives | human lost a REGULAR blind (a life) |
| `match_end` | winner ("human"/"agent"/"abandoned") | run over |

sidecar → mod (`inbox.jsonl`):

| e | fields | when |
|---|---|---|
| `hello` | protocol, spec, agent_name | sidecar up |
| `agent_nemesis` | ante, hands: [{score, hands_left}…], final | the agent's full round for that ante, as soon as computed; mod buffers by ante |
| `agent_state` | ante, lives, money | after each agent blind resolution / cash-out |
| `agent_dead` | ante | agent hit 0 lives OUTSIDE a Nemesis (failed regular blinds) → human wins |
| `error` | msg | sidecar fault; mod shows the score as "?" and the race degrades to solo |

## 3. Race semantics per Nemesis ante A (v1, mod side — OUR resolution)

State: `hands[]` (from `agent_nemesis[A]`), `revealed` (idx), human score/hands from the
game. Display: enemy score = `hands[revealed].score`, enemy hands = its `hands_left`.

- **Reveal pacing:** one agent hand revealed every ~2.5 s from blind start — the agent
  "plays at its own pace" beside you, independent of your hands. (Tagg's value-squeeze
  rule: pacing is cosmetic; all data is local, nothing is weaponised.)
- **Mid-hand check** (human plays, still has hands): if all agent hands are revealed AND
  human score strictly > agent final → the agent is exhausted and behind → **cut**: agent
  loses the life, round ends. (The mirror of MLB's early-end cut.)
- **Human exhausted:** reveal all remaining agent hands instantly, then compare finals —
  strictly greater side wins the round, loser loses a life, **exact tie: nobody** (server
  rule; deliberately diverges from the mod's ghost `>=`-favours-the-reacher).
- Lives applied by our wrapper (`MP.GAME.lives` / `MP.GAME.enemy.lives` + the mod's
  ease-lives UI), `pvp_result` emitted. Sidecar resolves its mirror game with the same
  two finals and the same rule — divergence is a bug: logged loudly, mod's result wins.
- `agent_nemesis[A]` not yet buffered when the Nemesis starts (only possible in the first
  seconds of a session): enemy score shows 0 with `info_received=false`; the human can
  play on; resolution blocks until data arrives (poll timer keeps trying) — v1 accepts
  this; it cannot happen after ante 2 because the agent runs a full ante ahead.

## 4. The mirror (sidecar side)

The agent = ONE solo-MLB `BalatroGame` on the session seed + `EVPlayer(spec)` (canonical
`DEFAULT_HAND_CONFIG`), stepped by the sidecar's loop:

- Plays regular blinds/shops/packs at full speed until it reaches its Nemesis for ante A
  → plays its round against the injected human-so-far score → emits `agent_nemesis[A]`
  → **pauses pre-resolution** until the human's final for A arrives (`pvp_result` or
  derived from `pvp_hand` exhaustion) → injects the final → resolves (life/comeback per
  the ENGINE's own MLB accounting) → plays on to the next Nemesis gate.
- So the agent's mirror is always exactly one unresolved Nemesis behind the human's
  progress, and one full ante ahead in data availability.
- Human's regular-blind life losses (`round_fail`) update the opponent-lives view of the
  mirror (mechanism per engine recon §6).
- Sidecar detects agent death at a regular blind → `agent_dead` → match over.

**Engine mechanism (FILLED from the engine recon, 2026-08-27; implemented in
`ghost/mirror.py`):**

- Construction: `BalatroGame(seed, deck_key, stake, ruleset="mlb")` then
  **`game.pvp_solo = False`** — the sidecar IS the server: the Nemesis parks in
  `State.PVP_WAIT` (`legal_actions() == []`) until the driver resolves it
  (`game.py:1800-1809`; template `agent/tests/test_mlb_agent.py:131-181`). With
  `pvp_solo=True` (default) the round auto-resolves at no cost the instant hands run
  out — never waits for an opponent.
- `step({"type":"play_blind"})` at a Nemesis only sets `pvp_ready`
  (`legal_actions() == []` while set, `game.py:1444-1446`); the driver calls
  `game._start_blind()` itself (what `MLBMatch.sync` does, `mlb_match.py:373-374`).
  `_start_blind` ZEROES `pvp_opponent_score/hands` + `chips_target` (`game.py:1083-1087`)
  — so `set_pvp_info` is re-applied before EVERY decision (`agent/TRAIN_NOTES.md:122-125`).
- `set_pvp_info(score, hands)` sets exactly `pvp_opponent_score`, `pvp_opponent_hands`,
  and (during the Nemesis) `chips_target` (`game.py:1889-1895`). It is LOAD-BEARING for
  EVPlayer's objective: `ev/hand.py:857-858` reads both fields, and `hands=0` collapses
  the opponent projection to a point mass — pass the real hands-left, 0 only when truly
  out. A Nemesis never resolves on `chips_target` (`game.py:1810-1822` is non-PvP only).
- Resolution is DRIVER-side (no contested resolution exists for a single game): strict
  `<` loses; exact tie takes nobody (`mlb_match.py:427-431`, the server rule — remote
  citation, `MLB_NOTES.md` §3.1). `lose_life()` (one per round max; `comeback_bonus +=1`,
  `game.py:1826-1838`) then `end_pvp()` (routes to `ROUND_EVAL`/`GAME_OVER`,
  `game.py:1897-1906`); the `advance` step's `_end_round` then pays $5 Nemesis reward
  (win or lose) + interest + `$4 × comeback_bonus` automatically (`game.py:2018-2023`,
  `constants.py:199`) — order matters: life BEFORE advance.
- No `pvp_log` on a solo game — the driver keeps its own (like `eval/common.py:307-308`).
  `game.chips_scored` is the cumulative round score (`game.py:1763`).
- `eval/*.py external_vanilla_big_blind_target` is a SYNTHETIC per-ante bar for training
  — wrong tool for a real opponent; kept out.
- The human's LIVES have no engine field; they reach the agent through
  `EVPlayer.bind_race` fed a match-shaped shim (`.games[1-p].lives` + `.pvp_log`,
  `ev/player.py:589-615`) — without it the shop tier's race aggression is neutral.
- Gotchas honored in `mirror.py`: `PVP_WAIT`/`pvp_ready` are silent-no-op states (a
  naive drive loop spins) — gate on a local `is_stuck_state`; a Nemesis deck-out can
  take the round's life early (`game.py:1870-1875`), so `lose_life()`'s return value is
  not assumed; a no-progress budget guards the frozen engine's silent-no-op corners.

## 5. The GhostRace Lua mod

New mod, own folder (`ghost/mod/GhostRace/` in-repo, installed by `ghost/install_mod.py`
to `%APPDATA%/Balatro/Mods/GhostRace/`). Depends on Multiplayer (manifest per mod recon).
NEVER copies MP code: it saves references to and wraps `MP.GHOST.load`, `init_playback`,
`resolve_pvp_mid_hand`, `resolve_pvp_hands_exhausted` (+ the read helpers the recon says
the call sites use), delegating to the originals whenever live mode is off — replay mode
(G1) keeps working unchanged. File IO via nativefs (absolute paths), JSON via the same
`require("json")` the MP mod uses, polling via the event-manager timer pattern the mod
recon confirms.

**Wrap contract (FILLED from the mod recon, 2026-08-27; implemented in
`ghost/mod/GhostRace/main.lua`; every citation is `$MOD/…` = the installed Multiplayer
mod, read-only):**

- **Load order is priority, not dependencies**: smods loads mods by ascending numeric
  `priority` only (`smods…/loader.lua:734-745`); Multiplayer declares `10000000`, so
  GhostRace declares `10000001` and `author` must be an ARRAY (`loader.lua:137`).
  All `MP.GHOST.*` functions are plain, singly-assigned table fields resolved at call
  time — nothing holds local references, so late wraps take everywhere (whole-mod grep).
- **File IO**: smods binds nativefs as global `NFS`; `NFS.append/write` funnel to a raw
  `_wfopen` with NO sandbox (`nativefs.lua:280-288,464`) — absolute paths work both
  directions. Poll with `NFS.getInfo(path).size` change + `NFS.read`.
  `require("json")` = rxi/json via smods (`libs.toml:6-9`), encode + decode.
- **Polling** = the mod's own pattern: chain `Game.update`, wall-clock accumulate via
  `love.timer.getTime()` (`$MOD/ui/game/timer.lua:381-414`) — immune to pause, event
  blocking, and the `MP.suppress_next_event` hazard on `E_MANAGER` (`round.lua:83-90`).
- **Resolver contracts** (callers: `$MOD/ui/game/game_state.lua:160-234`):
  `resolve_pvp_hands_exhausted(chips) -> "won"|"game_over"|"continue"` — never calls
  `win_game()` itself (the caller does, on `"won"`); sets `MP.GAME.end_pvp = true` on
  `"continue"`; own-life loss = comeback bookkeeping under `gold_on_life_loss`, then
  `MP.GAME.lives -= 1` + `MP.UI.ease_lives(-1)` + `no_gold_on_round_loss` zeroing.
  `resolve_pvp_mid_hand(chips) -> bool` — calls `win_game()` ITSELF on the terminal
  win; the caller ignores the boolean and reads `MP.GAME.end_pvp`. `MP.GAME.won` must
  be set BEFORE `win_game()`. Enemy life loss is a silent decrement (no ease call).
- **Display**: the HUD derives `score_text` from `MP.GAME.enemy.score` every frame
  (`blind_hud.lua:200-224`) — mutate the score table, never replace it; our paced
  reveal calls the ORIGINAL `advance_hand` (eases + juice) on a 2.5 s wall clock and
  neutralizes `_start_advance_sequence` (it would race the sidecar).
- **Hook points**: Nemesis-ante init at `$MOD/ui/game/round.lua:67-69`
  (`init_playback(G.GAME.round_resets.ante)`); human hand at
  `game_state.lua:188-214`; regular-blind fail via `resolve_round_fail`
  (`game_state.lua:252`); human cumulative score = `G.GAME.chips` (wrap in `to_big` —
  Talisman is installed); `MP.GHOST.clear()` fires from menu buttons → wrap it to emit
  `match_end: abandoned` and disarm.
- **Live-marker replay must keep sane**: `seed`/`deck`/`stake` (run start,
  `practice_mode.lua:32-34`), `ruleset` (∈ `MP.Rulesets`), `gamemode`, `nemesis_name`
  (blind name), and an `ante_snapshots` table (loader requirement). Our launcher also
  carries the agent's first round in `_live.bootstrap` (zero-IPC data) and as a G1-v2
  single-entry snapshot (graceful degrade to a static ghost without GhostRace).

## 6. Testing without the game

`ghost/tests/test_live.py` drives the REAL sidecar against a FAKE mod: a Python driver
that appends mod-shaped events to `outbox.jsonl` (a scripted human: enters Nemesis,
plays hands, fails a blind, dies / wins) and asserts the sidecar's `inbox.jsonl` stream:
agent_nemesis per ante with sane monotone hands, lives accounting matching the engine,
tie → nobody, cut semantics, crash-recovery (kill + restart sidecar mid-session replays
the outbox to the same state — determinism in (seed, spec)). The in-game pass is Tagg's,
as in G1.

## 7. Build status (2026-08-27, same session)

| gate | result |
|---|---|
| `ghost/ipc.py` transport | 8 tests (partial-line, corrupt-line, crash re-read) |
| `ghost/mirror.py` (solo-MLB mirror, external opponent) | 5 tests: round shape, all three resolution branches, comeback money in-cash-out, agent death, determinism in (seed, spec) |
| `ghost/live.py` sidecar | 6 tests: opening publish, full match flow, human/agent death, abandon, **crash-recovery replay to identical state**, launcher bootstrap |
| `ghost/mod/GhostRace/main.lua` | **executed in real LuaJIT (lupa) over a stubbed game env** — 7 tests incl. reveal pacing, the cut, tie-nobody, comeback bookkeeping, agent-death win, replay-mode passthrough, and a **full loop against the REAL sidecar over real IPC files** |
| whole `ghost/` suite | `python -m pytest ghost/tests` green |
| in-game | **VALIDATED 2026-08-27** — after two live-found fixes (Talisman comma scores; recovery double-read, `eb6527c`), the first clean full match ran start-to-finish on seed `AOG8R942`: 5 Nemesis rounds, closed-loop lives + comeback, paced reveals, clean end at 0 lives + launcher cleanup. Record: `ghost/MATCHES.md` |

## 8. Explicitly deferred past v1 (Tagg-confirmed queue, 2026-08-27 close-out)

- **Turn-based within-blind play** (Tagg's #1 future want): real interleaving instead of
  the score chase — the mod streams his live mid-blind score into the agent's decisions
  (`pvp_level1` atoms exist), the agent answers hand-for-hand, and the trailer-compelled
  protocol (PASS at a lead) becomes playable against a human. Requires holding the
  mirror's Nemesis open per-hand instead of pre-computing the round.
- **In-game post-match build reveal** (Tagg's #2): show the agent's joker lineup + deck
  on the game-over screen. Data already flows (`agent_state.jokers`; the chronicle has
  the deck); the work is mod-side UI — find what the MP game-over screen reads for
  enemy jokers in lobby mode and feed it. Until then: `python -m ghost.report`.
- Capture mode / the human metrics column (G3); the "why did you buy that" explain
  tool; the difficulty ladder in the launcher.
