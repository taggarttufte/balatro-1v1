# Ghost Mod — design brief for the interactive play-against-the-agent product

Date: 2026-08-27 · For: a fresh session building the Lua mod + sidecar. Decisions below were
made with Tagg across 2026-08-26 sessions (CAMPAIGN_LOG entries of that date are the record).

## The product

Tagg plays real Balatro against the agent, in the actual game, with an MP-mod-like layout:
opponent score visible, MLB life rules, Nemesis race framing. Three uses: (1) the demo — "race
my own agent in the real game"; (2) the measurement ritual — periodic matches tracking when
the agent surpasses him; (3) the capture side — his real runs replayed in-engine (seed parity
makes this exact) to mine his-choice-vs-agent-choice counterfactual pairs.

## The architecture decision (made, with reasoning — do not relitigate casually)

**Ship a MIRROR, not a recording.** The agent is closed-loop: comeback money ($4 × cumulative
lives lost) and the lives differential change its shop economy and race calculus, so a
prerecorded ghost is *wrong*, not just static, the moment the human's PvP outcomes diverge.
Because `EVPlayer` at ε=0 is a deterministic function of (seed, observable state), no recording
is needed: a **local sidecar process** advances the agent's mirror game on the same seed and
couples to the real game ONLY through (per-blind outcomes, PvP scores). Engine speed makes
this trivially real-time: fast budget ≈ 0.1–0.3 s per ante vs minutes of human play; even
full+Vleaf ≈ 163 ms/decision. CPU-only, no GPU contention with the game.

**MVP first: the static ghost** (G1 below) — zero coupling, uses the mod's own ghost format,
correct enough for solo-style racing; the mirror is v2.

## Phases

- **G1 — static ghost race (MVP):** export an agent trajectory for a seed in the
  BalatroMultiplayer mod's ghost-replay format; Tagg races it in the real game. Needs: the
  format (see `lib/ghost_replay.lua` in the installed mod — W-PVP already read this file),
  a writer from our `replay/` trajectory logs, and a validation pass (play a seed, confirm
  the ghost's shops/scores match the real game — seed parity should make this exact).
- **G2 — live mirror:** the Lua mod streams (blind outcomes, PvP scores, current blind index)
  to a local sidecar (file-tail or localhost socket; see IPC notes below); the sidecar runs
  `MLBMatch`-half on the same seed, advances the agent when the human's results arrive, and
  streams the agent's score progression back for display. Rate-limit the agent's reveals to
  the turn protocol / human-ish cadence — never weaponize speed (Tagg's value-squeeze rule).
- **G3 — product polish:** difficulty ladder from checkpoints (MEASURED by paired h2h, not
  assumed by recency — the ladder doubles as a regression suite); match history; the
  "playthroughs-to-beat-the-ghost" counter (= a human clairvoyance curve, the same condition
  as the 63.3→33.3% agent measurement; replay-with-knowledge runs are EVAL-ONLY, never
  training data — Tagg's own call); capture/logging mode for the pair-mining loop.

## What already exists (read these first in the new session)

- `ev/PVP_NOTES.md` — W-PVP's mod-source reconnaissance WITH file:line citations: the mod's
  action handler tables (`networking/action_handlers.lua`), no PvP timer in MLB
  (`ui/game/timer.lua:445-450`), `lib/ghost_replay.lua` (the ghost engine — note its `>=`
  tie handling contradicts the server rule; decide which the mod should emulate), score
  streaming (`hide_score_until_played` off for MLB, `core.lua:201`).
- `replay/` + its NOTES — trajectory logging with exact replay; Phase 4 judged writing the
  mod's ghost format feasible. This is G1's foundation.
- `engine/balatro_sim/mlb_match.py` — the mirror's core; `pvp_protocol="trailer_compelled"`
  is the standard world as of 2026-08-26 (defensible from mod source: waiting is untimed).
- The installed mod: `%APPDATA%/Balatro/Mods/Multiplayer/` (Lua + lovely patches, v0.5.2).
  READ-ONLY reference; the ghost mod is a NEW mod, never a fork that vendors their code.
- Old-repo Lua-mod lessons (archived `balatro-rl`, `mod/` + `legacy/`): V1–V3 drove the live
  game over file/socket IPC — known pitfalls: IPC latency was the training bottleneck (fine
  here — human-paced), and long sessions degraded RAM in the modded game (watch for leaks;
  restart between matches is acceptable for v1).
- Player strength context for the ladder: honest `real1` (weak rung), `ev:fast`/`ev:full`
  (95%+ ante-1, current bar), V-v3 checkpoints. Current honest estimate: Tagg wins 70–85%
  of matches vs today's agent — the ladder has headroom on both ends.

## IPC sketch (G2)

Simplest robust: the mod appends JSON lines to a file in the mod's config dir (Lua side:
`love.filesystem` append; no sockets needed inside the sandboxed Lua), the sidecar tails it
and writes its own JSON-lines file the mod polls each frame for score updates. One message
each way per blind + per PvP hand. Localhost TCP is the upgrade if file-tail latency annoys;
Balatro's Lua CAN do sockets (the MP mod does) but copying their networking layer is scope.

## Open decisions for Tagg at kickoff

1. G1 seed source: fresh random seed per race, or the 126-corpus seeds (analyzable but
   familiar)?
2. Which rung of the ladder ships as default opponent?
3. Ghost UI: piggyback the MP mod's Nemesis layout as visual reference (rebuild, don't copy)
   — how much fidelity for v1?
4. Logging mod for capture (G3): same mod with a record flag, or a separate minimal mod?

## Ground rules carried over

Never vendor/copy the MP mod's or the game's Lua; the mod is unofficial and personal-use;
everything stays in this repo (a `mod/` top-level dir is fine — it's a NEW product surface,
not engine code); the sidecar reuses the engine as a library, no engine changes for the mod's
convenience without the usual gates.
