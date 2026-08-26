# Phase 2 Brief — MLB rules + decks (2026-08-21)

Lead-authored kickoff brief. Supersedes the "MLB lost-PvP payout unknown" gap: the
BalatroMultiplayer mod (v0.5.2) is installed at
`C:/Users/Taggart/AppData/Roaming/Balatro/Mods/Multiplayer/` (referred to below as `$MOD`)
and its Lua + lovely patches are the ground truth for every MLB rule. **Read the mod source;
do not trust this brief over it.** Same rule as `_reference/`: port algorithms, never vendor
or copy the mod's code into deliverables, never commit it.

Phase 1 state at kickoff (all green, nothing committed): `pytest engine/tests` 1441/10 skip/0;
`pytest tests` 393/2 xfail/0; `python -m oracle.engine_parity --antes 1-8 --rerolls 5 --quiet`
126/126 exact. **Every workstream re-runs these before hand-off; they must stay green.**

---

## 1. MLB rules, as read from the mod source

### 1.1 Ruleset = Attrition gamemode, vanilla content

`$MOD/rulesets/majorleague.lua`: `multiplayer_content = false` (no MP jokers/decks/etc.),
`forced_gamemode = "gamemode_mp_attrition"`, `forced_lobby_options = true`, and
`force_lobby_options` pins `timer_base_seconds = 180`, `timer_forgiveness = 1`,
**`the_order = false`**, `preview_disabled = true`. Every other lobby option keeps the default
from `$MOD/core.lua:169-203` (`MP.reset_lobby_config`):

| option | default | meaning |
|---|---|---|
| `gold_on_life_loss` | true | comeback money (§1.5) |
| `no_gold_on_round_loss` | false | blind reward is still paid on a lost blind (§1.4) |
| `death_on_round_loss` | **true** | **lose a life on any non-PvP blind failed** — this is the rule Tagg was going to check in a practice lobby; it is the default and MLB forces options, so it's settled from source |
| `different_seeds` | false | both players on one seed |
| `starting_lives` | 4 | |
| `pvp_start_round` | 2 | boss slot is the Nemesis blind from ante 2 on |
| `different_decks` | false | both on the lobby deck |
| `stake` | 1 | White |
| `hide_score_until_played` | false for MLB (`play_button_callbacks.lua:115`: on only for `standard == true` rulesets; majorleague doesn't set it) | **opponent's live score is visible during the PvP blind** |

`$MOD/gamemodes/attrition.lua`:
- `get_blinds_by_ante`: ante ≥ `pvp_start_round` → Boss slot = `bl_mp_nemesis`. Ante 1 boss is a normal vanilla boss.
- **Bans:** jokers `j_mr_bones, j_luchador, j_matador, j_chicot`; vouchers `v_hieroglyph, v_petroglyph, v_directors_cut, v_retcon`; tags `tag_boss`; blinds `bl_wall, bl_final_vessel`. No consumable/enhancement bans. Applied through `MP.ApplyBans()` (`$MOD/rulesets/_rulesets.lua:198`) → check exactly how it populates `G.GAME.banned_keys` so the resample-queue in-place replacement is preserved (Phase 1 invariant).

### 1.2 Nemesis blind (`$MOD/objects/blinds/nemesis.lua`)

`SMODS.Blind{ key="nemesis", dollars=5, mult=1, boss={min=1,max=10}, in_pool=false }`. No
debuff, no boss effect. Its chip target is the **opponent's live score**; the client sends
`playHand{score, handsLeft}` after every hand (`$MOD/lovely/game.toml:105-109`, `ui/game/game_state.lua:185`)
and receives `enemyInfo` (`action_handlers.lua:349-459`) with the opponent's score/hands.

### 1.3 PvP resolution (server-side; client mirror in `$MOD/lib/ghost_replay.lua:140-252`)

The live outcome comes from the server (`endPvP{lost}`, `playerInfo{lives}`, `winGame`/`loseGame`).
The mod's own offline emulation of that logic is the ghost-replay code — treat it as the spec unless
the server source says otherwise (server repo: `github.com/Balatro-Multiplayer/balatro-multiplayer-server`
or similar — **W1 should try to WebFetch it and confirm the tie rule**):

- Each player plays until `hands_left == 0` (`game_state.lua:190-213`: when out of hands, wait for enemy).
- `resolve_pvp_hands_exhausted` (player out of hands): if `my_chips >= opponent_final` AND the opponent is also
  exhausted → opponent loses a life; else → I lose a life.
- `resolve_pvp_mid_hand` (player still has hands): if `my_chips >= opponent_final` and opponent is exhausted →
  **early end, opponent loses a life**, my remaining hands are not played.
- Note `>=` — in ghost mode a tie favours the player who reaches it. Verify vs server.
- Deck-out edge (`game_state.lua:444-459`): 0 hands played + discards used + deck empty → counts as a fail
  (`fail_round(1)`), and at PvP sends `play_hand(0, 0)`.
- Life loss → 0 lives → `loseGame` for that player; the other gets `winGame`. `G.GAME.win_ante = 999`
  (`game_state.lua:266`): **there is no ante-8 win; the match runs until a player hits 0 lives** (endless
  blind scaling — `constants.get_blind_amount` already handles ante > 8).

### 1.4 Money at blind end (`$MOD/lovely/game.toml`)

- **Unused-hand money is NOT paid at a PvP blind** (patch at `game.toml:96-101`: `... and (not MP.is_pvp_boss())`).
- **The blind reward ($5 for Nemesis) IS paid at a PvP blind whether you won or lost** (`game.toml:124-131`:
  `if chips - blind.chips >= 0 or MP.is_pvp_boss()`). This closes SWEEP_NOTES §4 item 7.
- **A failed non-PvP blind also pays its reward**: `game_state.lua:255` sets `G.GAME.blind.chips = -1`
  ("Prevent player from losing"), so the vanilla `chips - blind.chips >= 0` test passes; the run proceeds
  to the next blind as if it were defeated, and `fail_round(hands_played)` costs a life (`death_on_round_loss`).
  `fail_round` returns without sending if `hands_used == 0` (`action_handlers.lua:1143-1149`).
- Interest: vanilla. Skip: vanilla (tag), no MP change.

### 1.5 Comeback money (`$MOD/lovely/game.toml:19-49`, `action_handlers.lua:510-523`)

On every life loss (live path: any `playerInfo` lives decrease while `lives ~= 0`), `comeback_bonus += 1`
(cumulative lives lost) and `comeback_bonus_given = false`. At the **next Cash Out**, after interest, pay
`4 * comeback_bonus` once (MLB branch: `MP.is_major_league_ruleset()`), then set `given = true`. So:
1st life lost → +$4 at the next cash-out; 2nd → +$8; 3rd → +$12. (Ghost path `ghost_replay.lua:240-252`
doesn't bump `comeback_bonus` for a regular-blind fail — that's the offline emulation diverging from the
live path; **follow the live path** and flag it for Tagg's practice-lobby check.)

### 1.6 Voucher generation under MLB ≠ vanilla (`$MOD/compatibility/TheOrder.lua:481-525`)

`SMODS.get_next_vouchers` and `get_next_voucher_key` are overridden when
`MP.should_use_the_order() or MP.is_major_league_ruleset()` — i.e. **also in MLB with The Order off**
(the integration is on by default: `config.lua` `integrations.TheOrder = true`). They draw from
`get_culled(pool)` (paired base/upgrade handling, `"UNAVAILABLE"` sentinel, see `TheOrder.lua:~440-478`)
with `pseudoseed("Voucher0")` — **no ante suffix** — re-drawing on `UNAVAILABLE`/already-spawned with a
1000-iteration fallback to `"Voucher0"..it`. This applies to shop vouchers AND the Voucher Tag
(`_from_tag`). So under MLB the voucher stream is run-global even though shops/packs stay ante-keyed.

### 1.7 The Order (when `the_order = true`; NOT MLB, but the switch must work) — `$MOD/lovely/TheOrder.toml` + `compatibility/TheOrder.lua`

Ante suffix → `MP.ante_based()` = `0` on: `halu`, `pack_generic`/pack key, shop polled rate, `cdt`, rarity,
joker/center pools, editions, fronts, soul, stickers, rentals, standard-pack keys… (read the whole toml —
every `..G.GAME.round_resets.ante` site that's patched). Shuffles: `nr`/`cashout` → `MP.order_round_based(true)`
= `ante .. blind_key .. blind_on_deck` under The Order. Boss: `'boss'..ante` under The Order (vanilla `'boss'`).
~~Seed gets a `*` prefix for display only.~~ **CORRECTION (W2, verified in LuaJIT):** the `*` prefix is applied
BEFORE `hashed_seed`, so under The Order every stream is keyed on `'*'..seed` — a different universe, not a
display change. Full authoritative key-site table: `rng/NOTES_ORDER.md` §3 (supersedes DELEGATE_NOTES §3).
Vouchers: §1.6 path. Keys NOT listed stay vanilla.

### 1.8 Opponent information available to a bot (for the PvP decision frame)

Same seed (§1.1) ⇒ you know the opponent's option set. During a PvP blind you see their live score and hands
left (`enemyInfo`). Between blinds you see their location, lives, skips. Nothing else is revealed in MLB
(no MP jokers). Lives/score/hands of both players must be in the MP observation.

---

## 2. Workstreams (fleet, disjoint ownership; `game.py` is SHARED — targeted `Edit` only, never whole-file `Write`)

| # | Workstream | Owns (create/modify) | Shared (Edit-only, re-read before each edit) |
|---|---|---|---|
| **W1** | MLB match rules + two-player lockstep coordinator | `engine/balatro_sim/mlb_match.py` (new), `engine/balatro_sim/mp_game.py` (retire or rewrite — V8-era, wrong rules), `engine/balatro_sim/env_mp.py`, `engine/tests/engine_tests/test_mlb_*.py`, `engine/MLB_NOTES.md` | `game.py` (nemesis blind type, PvP round-eval money, failed-blind-proceeds + life signal, comeback payout at cash-out, endless/no win_ante, Attrition bans → `run_state.banned_keys`), `constants.py` |
| **W2** | The Order hook + MLB voucher path in the generation layer, oracle-verified | `rng/generate.py`, `rng/keys.py`, `tests/test_the_order.py` (new), `rng/NOTES_ORDER.md`; may extend `tests/test_generate_oracle.py`'s `LuaGenOracle` | `game.py` one-liner: `_init_game_vars` sets `rs.key_scope = self.queue_scope` + a `ruleset`/`mlb` flag → `RunState` |
| **W3** | Decks (Red/Checkered/Plasma required, the rest where trivial) + stake catalogue (White verified) | `engine/balatro_sim/decks.py` (new), `engine/balatro_sim/stakes.py` (new), `scoring.py` (Plasma balance), `engine/tests/engine_tests/test_decks.py`, `test_stakes.py`, `engine/DECKS_NOTES.md` | `game.py` (deck/stake hooks at run start, Plasma blind ×2, Anaglyph Double Tag on boss defeat), `constants.py` |
| **W4** | Phase 2 exit gate (runs AFTER W1+W2) | `scripts/mlb_match_demo.py`, `tests/test_mlb_match_gate.py`, additions to `tests/test_engine_invariants.py` | — |

Sources for W3: `_reference/balatro_src/back.lua` (`Back:apply_to_run` 173-288, `Back:trigger_effect`),
`game.lua:2018-2060` (stake modifiers), `functions/state_events.lua` (`G.FUNCS.evaluate_play` Plasma
balance: `if G.GAME.modifiers.plasma ... ` — port the exact rounding), `blind.lua` (Plasma blind ×2 is
`G.GAME.starting_params.ante_scaling`/`modifiers` — read it). `rng/generate.py:1104-1118` already has
the generation-side deck effects (`DECK_EFFECTS`, `build_starting_deck` checkered/erratic/no_faces).

### Exit gate (from the campaign plan)

Two scripted players on one seed play a full MLB match end-to-end through the ENGINE:
1. lives decrement on every lost blind (regular and PvP), comeback money lands at the next cash-out with the
   right amount, the match ends at 0 lives (not ante 8), early-end PvP rule fires;
2. both players' shop queues stay aligned: every shelf/pack/voucher difference between the two players is
   explained by (a) that player's own rerolls or (b) a blocked-slot in-place resample due to their own
   collection — assert this with a diff, not by eye;
3. `engine_parity --antes 1-8` is still 126/126 with `ruleset="vanilla"`, and a `ruleset="mlb"` run differs from
   vanilla ONLY in vouchers (§1.6), bans (§1.1) and the ante ≥ 2 boss slot;
4. all prior gates green.

### Things only Tagg can do (reduced from the plan's three)

1. ~~Verify the regular-blind life rule~~ — settled from source (`death_on_round_loss = true`, forced). A live
   check is still a nice confirmation of §1.5's comeback-on-regular-fail question, but not blocking.
2. Spot-check Plasma and Checkered vs the real game (no oracle for deck semantics).
3. `7I4M53DL` live check from Phase 0 (still outstanding).
