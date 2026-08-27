# GHOST_NOTES — G1: the static ghost race (play the agent in the real game)

**Session 2026-08-27.** Package: `ghost/**` (new). Design brief:
`docs/GHOST_MOD_BRIEF_2026-08.md` (G1 of G1→G2→G3). Read-only references: the installed
BalatroMultiplayer mod at `%APPDATA%/Balatro/Mods/Multiplayer/` (cited as `$MOD/…`, nothing
copied or vendored), `replay/REPLAY_NOTES.md` §6 (the Phase-4 feasibility investigation this
package turns into product). `engine/**`, `rng/**`, `replay/**`, `ev/**` were read, never
edited.

## 0. Headline

| gate | result |
|---|---|
| `python -m pytest ghost/tests -q` | **10 passed** |
| `python -m pytest replay/tests -q` | unchanged (nothing in `replay/` edited) |
| `python -m ghost.make --spec ev:fast` (seed `Q3YSA2CC`) | full pipeline ~3 s: 6 Nemesis rounds logged, ghost installed, `replay.cli verify` clean on the produced log |
| in-game load + race | **NOT yet verified — Tagg's validation pass, §5** |

**No Lua is shipped in G1.** The mod's own Practice-mode ghost race
(`$MOD/lib/ghost_replay.lua` + `ui/main_menu/play_button/ghost_replay_picker.lua`) is the
entire runtime: it lists any `.json` we drop into `$MOD/replays/`, and "Play Match" starts
the human's run **on the replay's seed and deck** (`$MOD/lib/practice_mode.lua:32-34`), so
the same-seed race — the thing seed parity was built for — is native mod behaviour.

## 1. File map

```
ghost/
  __init__.py         package docstring
  _bootstrap.py        repo root on sys.path + fork-guarded engine import (mirrors replay/)
  export.py            MatchLogger line -> mod ghost JSON; python -m ghost.export
  make.py              one command: self-play a seed -> log -> ghost -> install; -m ghost.make
  parity_card.py       the human-checkable seed-parity card; -m ghost.parity_card <seed>
  conftest.py           test bootstrap (mirrors replay/conftest.py)
  runs/                 (gitignored) logged matches + ghost copies + parity cards
  tests/
    _helpers.py          random-legal MLBMatch driver through the MatchLogger hook
    test_export.py        schema, resolution pins, seat mirroring, rejections
```

## 2. What the mod actually consumes (load-bearing vs display)

Per Nemesis ante, the ONLY fields `$MOD/lib/ghost_replay.lua` reads during play are
`ante_snapshots[ante].hands[] = {score, hands_left, side}` filtered to `side == "enemy"`:
the human's live chips race each entry's `score` in order (`resolve_pvp_mid_hand`,
`resolve_pvp_hands_exhausted`, `advance_hand`). `score` is an insane-int string; plain
decimals parse as themselves (`$MOD/lib/insane_int.lua`), which covers every score our
engine produces at sane antes. Everything else — per-ante `player_score`/`enemy_score`/
lives/`result`, jokers, names, `final_ante`, `winner`, `timestamp` — feeds the picker UI
only. `ruleset` must be a `MP.Rulesets` key (`is_ruleset_supported`): Major League is
`"ruleset_mp_majorleague"` (verbatim at `$MOD/core.lua:114`), which forces
`gamemode_mp_attrition` (`$MOD/rulesets/majorleague.lua:16`). The deck is looked up **by
display name** (`MP.UTILS.get_deck_key_from_name`, `$MOD/lib/card_utils.lua:168-172`), so
`export.py` writes `decks.DECKS[key].name` ("Red Deck"), pinned by test.

Both seats are written (`side "enemy"` = the agent, `side "player"` = its sim opponent),
so the picker's perspective-flip button works.

## 3. The one extraction subtlety (`export.py::_pvp_hand_entries`)

`replay/log.py` captures the per-step summary AFTER `match.step()`, and the step that
RESOLVES a Nemesis round tears the blind down inside that same call — the resolving play's
post-step summary can show reset chips and a non-PvP state. So PvP plays are detected from
the PRE-step summary (previous step's snapshot of that seat), and each round's final entry
takes its score from `pvp_log` (the engine's own resolution record). The converter
hard-errors if the last enemy entry disagrees with `pvp_log`
(`test_final_hand_matches_resolution` pins the mechanism).

## 4. Known artifacts — what a static ghost gets wrong (all inherited by design)

* **Open loop.** The recorded agent's comeback money ($4 × cumulative lives lost) and
  economy were shaped by its SIM opponent's results, not the human's. The moment the
  human's PvP outcomes diverge from the sim opponent's, the recording is *wrong*, not just
  static — this is the brief's argument for the G2 live mirror, restated here so nobody
  reads a G1 race as a calibrated measurement.
* **Early-cut truncation.** A recorded round ends when the SIM race resolved. A leader
  whose sim opponent busted early banks its remaining hands and records a LOWER final
  score than it could reach against a stronger human; symmetrically a cut trailer records
  fewer hands. The ghost is soft exactly where its sim opponent was.
* **Tie rule divergence.** The mod's ghost resolution uses `>=` in the HUMAN's favour and
  always takes a life on failure (`$MOD/lib/ghost_replay.lua:142-168`); the engine keeps
  the server rule (exact tie: nobody loses — `MLB_NOTES.md` §3.1, remote citation). An
  exact-tie ante therefore plays out slightly differently in-game than the recording's
  `result`/lives fields display. Cosmetic for G1; decide which rule the G2 mirror emulates.
* **Recording horizon.** `ante_snapshots` exist only for antes the sim match reached. If
  the human outlives the recording, later Nemesis antes have no data
  (`has_hand_data() == false`); what the mod does there is UNVERIFIED — observed behaviour
  goes here after the first long race.
* **Seat choice.** Default ghost = the sim-match WINNER's seat. Canonical self-play has a
  known seat bias (seat 1 ≈ 70% on the 24-seed run, `results/` 2026-08-26), so "winner"
  is a real selection, not a coin flip. `--agent-seat 0|1` overrides.
* **Protocol.** Matches are played under the CANONICAL protocol — the world every gate and
  h2h number was measured in. `trailer_compelled` generation would need the protocol
  player (`adapt_match_player` + `protocol_hand_cfg`) and would make the early-cut
  artifact *worse* (leaders wait more); revisit at G2 where the mirror makes it moot.

## 5. The validation pass (Tagg, in game — G1 is not "done" until this ran once)

1. `python -m ghost.make --spec ev:fast` (or race the already-installed
   `ghost_Q3YSA2CC_ev-fast.json`). Keep the printed **parity card**.
2. Balatro → Multiplayer → **Practice** → **Ghost Replays** → the entry named `ev:fast`
   → Play Match. (The list also shows entries parsed from lovely logs; ours is the blue
   file-sourced one.)
3. Check the parity card's run-start lines (boss, voucher, tags, opening hand) and the
   first shop before rerolling. Any mismatch: stop, save the card + a screenshot.
4. Race at least through ante 2-3: confirm the ghost's score ticks up hand-by-hand at the
   Nemesis with the card's numbers, lives are taken per MLB rules, and the match ends
   sanely at 0 lives (either side).
5. Report back: picker loaded it? scores matched? anything the mod did with a
   no-data ante (§4)? → results land in this file's §0 table.

## 6. Usage

```
python -m ghost.make                         # fresh random seed, ev:fast, install + card
python -m ghost.make --seed 7I4M53DL --spec ev:full
python -m ghost.make --no-install            # dry run into ghost/runs/ only
python -m ghost.export <log.jsonl> <idx> --install   # convert an existing logged match
python -m ghost.parity_card <seed>
```

`--spec` speaks `ev/h2h.py::build_player`'s language (`ev:fast`, `ev:full`, `ev:full+stats`,
`ev:full+Vleaf`, `real1:det`, `scripted:…`) — the G3 difficulty ladder is a spec string
plus an h2h measurement, no new machinery.

## 7. Next phases (pointers, not promises)

G2 live mirror (sidecar + minimal Lua for IPC — brief §"IPC sketch"), G3 ladder/history/
capture. The brief's four kickoff decisions for Tagg are still open; G1 committed to the
least-binding defaults (fresh random seeds, `ev:fast`, the mod's own UI, no capture mod).

## 8. Needs-engine-change

None. Nothing outside `ghost/` was edited.
