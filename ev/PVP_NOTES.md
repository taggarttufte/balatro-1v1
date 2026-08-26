# PVP_NOTES — W-PVP: the PvP turn protocol, the level-1 Nemesis objective, the extraction pivot

**Agent W-PVP, 2026-08-26.**  Spec: `CAMPAIGN_LOG.md` "2026-08-26 — DESIGN INPUT (Tagg): PvP
TURN PROTOCOL — compulsion on the trailer".  Read `EV_NOTES.md` §3 (the level-0 Nemesis
objective and its two documented gaps) and `EXTRACT_NOTES.md` §4 (the tail-DP safety gate)
first; this document only describes what changed.

Files: `engine/balatro_sim/mlb_match.py` (the protocol, flag-gated), `ev/hand.py` (the
level-1 atoms, `race_value` / `pvp_decided` / `pass_candidate`, the amended gate),
`ev/player.py` (`act(extra_actions=…)`, `adapt_match_player`, `protocol_hand_cfg`),
`ev/fixtures/nemesis_decided_lost.py` (+ registry), tests
`engine/tests/engine_tests/test_pvp_protocol.py` (27) and `ev/tests/test_pvp_protocol.py`
(26), fixture `engine/tests/engine_tests/pvp_canonical_transcripts.json`, driver
`ev/scripts/pvp_protocol_h2h.py`.  `ev/train_v.py`, `ev/pairs.py`, `ev/scripts/gen_pairs.py`
and `ev/scripts/sweep_rank.py` are untouched (in use tonight), and so are `ev/h2h.py` and
`ev/gate_ev_player.py`.

---

## 0. Headline

| gate | result |
|---|---|
| `pytest engine/tests` | **1678 passed / 10 skipped / 3 xfailed / 0 failed** (1651 before + 27 new) |
| `pytest tests` | **1073 passed / 2 xfailed / 0 failed** (unchanged) |
| `pytest ev` | **333 passed / 0 failed** (307 before + 26 new) |
| `python -m oracle.engine_parity --antes 1-8 --rerolls 5` | **126/126 EXACT through ante 8** |
| canonical-mode byte-equivalence, 4 seeds × `ev:fast` self-play | **step-by-step signature chain identical**, incl. the signature tuple's length |

Everything new is **off by default**: `MLBMatch(pvp_protocol="canonical")` is the default and
is provably unchanged; every EV-side knob is a `HandConfig` field defaulting to `False`.

---

## 1. Mod-source check — what the real game does, and what it cannot tell us

Re-read from the installed mod at `%APPDATA%/Balatro/Mods/Multiplayer/` (read-only, nothing
copied or vendored).  Citations are `file:line` inside that tree, written `$MOD/…`.

### 1.1 Stalling at a PvP blind: **nothing happens.  There is no clock and no AFK handling.**

The classic timer — the one Major League uses — returns early inside a PvP blind:

```lua
-- $MOD/ui/game/timer.lua:445-450
    else
        -- Old timer: tick when opponent timering, not in pvp
        if is_pvp_boss then return end
```

`majorleague.lua:1-28` (and `minorleague.lua:1-24`) declare **no `layers`**, so no
`pvp_timer` layer is loaded; `pvp_timer` exists only in `ranked.lua:3` and as an
`experimental` default.  The classic timer is in any case an opponent-pressed button that
only applies *outside* the blind (`$MOD/ui/game/timer.lua:16`, `:8` — it refuses when the
opponent is already `loc_ready`).  A repo-wide grep for `afk|idle|timeout` finds only socket
keep-alive (`$MOD/networking/socket.lua:130-131`) and the *disconnect* grace timer
(`$MOD/networking/action_handlers.lua:153-163`, server-driven).  **No forced action exists.**

Where a `pvp_timer` IS loaded, expiry is a client-reported loss
(`$MOD/ui/game/timer.lua:473-486` → `failPvPTimer`, `$MOD/networking/action_handlers.lua:1361-1365`),
the clock only runs for the side the server names via `pvpTimerOrder` (`:429-431`), and it
stops at `hands_left <= 0` (`timer.lua:432`).  None of that is reachable under MLB — which is
why `MLB_NOTES.md` §4 lists `mp_pvp_loss` as unreachable, and it still is.

**This is the fact the protocol rests on**: waiting is a legal, unpunished, untimed line in
the real Major League game.  The only thing that makes a player act is needing to score.

### 1.2 Concede / forfeit: **definitively absent.**

The complete client→server action set is `$MOD/networking/action_handlers.lua:1076-1413`
and the complete inbound handler table is `:1452-1498`.  Neither contains a concede,
forfeit, surrender, resign or give-up message; a grep over every `.lua`, every `lovely/*.toml`
and all 17 localization files returns zero hits, so there is not even a button label.

The three voluntary exits are `leaveLobby` (`:1108-1116`), `stopGame` (`:1137-1141`) and the
vanilla Restart — which MP removes outright (`$MOD/lovely/pause.toml:44-52`,
`$MOD/overrides/disable_restart.lua:1-4`).  `stopGame`'s inbound handler records
`MP.RLOG.end_run({ result = "stop" })` and, unlike `action_lose_game` (`:543`), does **not**
call `MP.STATS.record_match(false)` — aborting is a no-contest, not a scored loss.

So "wait" is the only form of "not playing" the protocol gives a player.  Good for us: the
pass action needs no mod-side counterpart, because it is the *absence* of a message.

### 1.3 Round hard-end / the early-end cut: **real, and server-decided.**

The client never decides a PvP round.  On running out of hands it reports and idles:

```lua
-- $MOD/ui/game/game_state.lua:185-208
                if not ghost then
                    MP.ACTIONS.play_hand(G.GAME.chips, G.GAME.current_round.hands_left)
                end
                if G.GAME.current_round.hands_left < 1 then
                    ...
                        attention_text({ ... text = localize("k_wait_enemy"),
```

(`k_wait_enemy = "Waiting for enemy to finish..."`, `$MOD/localization/en-us.lua:1258`; no
action is legal there, `$MOD/ui/game/functions.lua:39-46`.)  The local fail-round check is
explicitly skipped at a PvP boss (`game_state.lua:249`).

That the cut is **real in the protocol** is visible from the client having two receive paths
for an `endPvP` that arrives *while hands remain*: `$MOD/networking/action_handlers.lua:472-480`
(the `pvpTimerLost` branch reads `hands_left > 0`) and `$MOD/ui/game/game_state.lua:313-318`,
which force-jumps `G.STATE` to `NEW_ROUND` in the middle of `update_selecting_hand`.  The
client-side reimplementation used for ghost replays does the cut explicitly
(`$MOD/lib/ghost_replay.lua:174-192`, `resolve_pvp_mid_hand`), and `$MOD/agents.md:58`
states the rule as *"Failing a PvP blind (chips < opponent's score when hands run out) costs
1 life."*  Deck-out forces the same exhausted report (`game_state.lua:298-307`).

**Confirmed** — this is what `MLBMatch._resolve_pvp` already implements, and W-PVP did not
change it.

### 1.4 The exact tie: **CONFIRMED ONLY REMOTELY.  Worth flagging.**

`MLB_NOTES.md` §1.3a records the tie rule as "server, confirmed" citing
`BalatroMultiplayerAPI-Server/src/actionHandlers.ts:303-345` — a repository fetched over the
network in Phase 2 (`docs/PHASE2_BRIEF_2026-08.md:54,61`).  **Nothing on this machine can
corroborate it.**  There is no score comparison in the live-MP client at all: every
`MP.INSANE_INT.greater_than` call site is bookkeeping or UI
(`action_handlers.lua:378`, `:1203`; `blind_hud.lua:219`; `speedlatro_timer.lua:83`), and
lives arrive pre-computed via `playerInfo` (`action_handlers.lua:510-522`).

The one client-side reimplementation **contradicts it**:

```lua
-- $MOD/lib/ghost_replay.lua:142-168
function MP.GHOST.resolve_pvp_hands_exhausted(chips)
    local beat_current = to_big(chips) >= MP.GHOST.current_target_big()
    local all_exhausted = MP.GHOST.playback_exhausted()
    if beat_current and all_exhausted then
        MP.GAME.enemy.lives = MP.GAME.enemy.lives - 1
```

`>=`, and no "nobody loses" branch — the `else` at `:152-165` always takes a life from the
player.  `MLB_NOTES.md` §3.1 already knew about this divergence and chose the server rule;
the new information is only that the ghost path is the *only* local evidence and it points
the other way.  The engine keeps the server rule (it is the live-play rule and the one
`agents.md:58`'s strict `<` phrasing agrees with), and `_resolve_pvp`'s docstring now names
the exact line to change if it is ever re-verified.  **Ledger status: remote citation, not
locally reproducible.**

### 1.5 Score visibility: live, and unmasked under MLB.

`playHand{score, handsLeft}` goes out on every hand
(`$MOD/networking/action_handlers.lua:1192-1211`; call sites `game_state.lua:185`, `:304`,
`:454`, `functions.lua:74`, `lovely/game.toml:107-109`) and comes back as `enemyInfo`
(`action_handlers.lua:349-355, 373-378, 424-425`), broadcast to both clients roughly every
3 s (`$MOD/agents.md:71`).  `hide_score_until_played` would mask it before your first hand
(`$MOD/ui/game/blind_hud.lua:203-216`, enforced server-side as `noScore`) but it defaults
`false` (`core.lua:201`) and is auto-enabled only for `standard`-layer rulesets
(`play_button_callbacks.lua:115`) — i.e. **off for Major League**.

This is what makes a level-1 objective meaningful at all: the opponent's score is genuinely
on the HUD, every hand, unmasked.

### 1.6 Lives / comeback / cost of a lost PvP round — no change

4 lives, `pvp_start_round = 2` (`core.lua:174-178`); one life per lost PvP round, pushed by
the server (`action_handlers.lua:510-522`); Major League comeback `4 × comeback_bonus` at the
next Cash Out (`lovely/game.toml:22-47`); the Nemesis pays $5 won **or lost**
(`objects/blinds/nemesis.lua:19-24` + `lovely/game.toml:149-153`); **no unused-hand money at
a PvP blind** (`lovely/game.toml:96-98`); 0 lives → `loseGame`, GAME_OVER with no Cash Out
(`action_handlers.lua:538-549`).  All of this matches `MLB_NOTES.md` §1.4–1.5 and is what
§5's money reasoning is built on.

---

## 2. The protocol as implemented (`engine/balatro_sim/mlb_match.py`)

`MLBMatch(pvp_protocol=...)`, `"canonical"` (default) | `"trailer_compelled"`.  An unknown
value raises.

> **The protocol is a MODELLING CHOICE and CANNOT be oracle-verified.**  The real game is a
> real-time race; there is no "turn" anywhere in the mod or the server to check it against.
> §1.1 and §1.2 make it a defensible discretisation — waiting is legal, untimed and
> unpunished, and it is the only non-action a player has — but nothing more than that.  This
> sentence is repeated in the module docstring, which is where a reader of the code lands.

**The one rule:** inside a live Nemesis, the player who is **strictly ahead** may play the
match-level action `{"type": "pvp_pass"}`.  Nobody else may.  Everything below is a
consequence of that plus one anti-wedge clause.

| # | edge case | decision | why |
|---|---|---|---|
| 2.1 | the start of every Nemesis | both scores are 0 ⇒ equal ⇒ **neither may pass** ⇒ both are compelled | this IS the brief's "both play hand 1 simultaneously"; it is not special-cased, it falls out of the strict comparison |
| 2.2 | an exact mid-blind tie | equal ⇒ neither may pass ⇒ simultaneous again | the brief's rule, same mechanism |
| 2.3 | the trailer | never offered a pass — **compelled** | "compulsion on the trailer"; a trailer who waits simply loses |
| 2.4 | the leader in `PVP_WAIT` | no pass | out of hands: there is nothing left to conserve, so waiting is not a choice they own |
| 2.5 | a player readied but not started | no pass (`_in_pvp` requires both games inside the blind) | the blind has not begun |
| 2.6 | two passes in a row | **the second is not offered** (`_pass_streak`) | anti-wedge: it can only arise when the compelled player's action changed nothing (an illegal or silently-no-op action from a policy), and without it the pair hands the turn back and forth forever.  In normal play the compelled player always moves in between, so it never bites |
| 2.7 | what resets the streak | any step that changes the stepping player's cheap progress tuple (`chips_scored, hands_left, discards_left, len(hand), len(consumable_hand), state`), and `startBlind` | cheap; it is a "did anything happen" fingerprint, never compared across matches |
| 2.8 | a pass that was NOT offered | **ignored**, turn still passes, not counted, streak not reset | `MLBMatch.step` is documented-permissive (it ignores illegal actions exactly as `BalatroGame.step` does); raising would break that contract.  `test_a_pass_only_policy_still_terminates` pins that even a policy which always tries to wait finishes the match |
| 2.9 | the effect of a pass | **nothing.**  It never reaches `BalatroGame.step`: same hand, same `hands_left`, same `state_signature()`, same RNG position.  It increments `steps`, flips `_turn`, and records `pvp_passes` / `pvp_pass_detail` | a pass is the absence of an action; anything else would be inventing a game rule |
| 2.10 | turn order | unchanged strict alternation (`current_player()`) | the leader is offered the turn and chooses "answer or wait", which is what makes PASS a real decision instead of an absence of one |
| 2.11 | end conditions | **unchanged** (`_resolve_pvp` untouched): out of hands and strictly behind ends the round at once, both out of hands compares, exact tie takes nobody | the protocol decides who may act, never who wins.  §1.3 confirms the cut; §1.4 flags the tie's provenance |
| 2.12 | `signature()` | the protocol tail (`pvp_protocol, _pass_streak, pvp_passes`) is **appended only when non-canonical** | so a canonical match's signature tuple is byte-identical — same length, same contents — which is what the transcript pin checks |
| 2.13 | `clone` / `clone_determinized` | carry the protocol, the counters and the streak; independent as before | MCTS / world-sampling machinery must not lose or share them |
| 2.14 | `env_mp` | untouched, therefore canonical | nothing in the V7 env expresses a wait |

**Not modelled** (deliberate): the wall-clock speed play.  It is meaningless in a turn
abstraction, and for human-facing play the agent is rate-limited to the protocol, so speed
is never weaponised — Tagg's own call, recorded in the design entry.

---

## 3. The level-1 Nemesis objective (`cfg.pvp_level1`, default OFF)

`EV_NOTES.md` §3's level-0 opponent model is: their final score = live score + one symmetric
per-hand score for each hand they still hold, as three atoms.  Its two documented gaps were
"the opponent reacting to my score" and "the early-end cut".  Level-1 closes the first.

**The change is one atom.**  `opponent_final_atoms(..., level1=True, live_weight=w)` adds the
opponent's **revealed live score** as an atom of weight `w` and renormalises the projection
to `1 - w`.  It is added *only when I am strictly behind*:

| my score vs theirs | live atom | why |
|---|---|---|
| ahead | none (w = 0) | under the protocol THEY are the compelled one — they must answer, so the projection is the whole story |
| equal | none | both compelled, same reasoning |
| **behind** | weight `pvp_live_weight` (0.5) at their revealed score | they are the leader and are allowed to wait.  If I never overtake, their final score is the number on the HUD *right now* — under the protocol that is a real and common outcome, not a rare one |
| they are out of hands | point mass at the live score, either way | already exact in level-0 |

That is precisely the brief's "compute needs against the opponent's REVEALED live score for
the tie/win mixture, the atoms model remains for projecting their remaining hands".  The
tie/win mixture, the `_value_for_need` machinery and `beta = gamma = 0` are all unchanged;
only the atom list they are evaluated over grew by one entry.

**What it does to play.**  For the trailer it lowers the bar from "beat everything they could
possibly still make" to a mixture that is half "beat what they have made", which (a) makes
an overtake look worth attempting instead of hopeless, and (b) — because reaching the live
score is a *nearer* target — pulls scoring earlier in the blind, which is the direction the
early-end cut wants.  It is also strictly *conservative* for the decided-lost gate of §5:
a higher race value makes "lost" harder to declare, never easier (pinned by
`test_level_one_raises_the_trailers_estimate_of_the_race`).

**Still not modelled (level-2, explicitly out of scope):** minimal overtake (pass them by
exactly 1 chip and keep the rest in reserve), statement-hand signalling (overshoot on purpose
to make them burn hands), and any inference about *why* they played what they played.  Also
unmodelled: the early-end cut as an explicit sequencing term — level-1 gets at it only
indirectly, through the nearer target.

---

## 4. The leader's PASS (`cfg.pvp_pass`, default OFF)

`HandAnalysis.pass_candidate()` → `({"type": "pvp_pass"}, race_value() + pvp_pass_tiebreak)`,
appended by `rank_hand_actions(..., allow_pass=True)`.

`race_value()` = `position_value(full_mask, m=0, h, d)` under the PvP need mixture: hold every
card, draw nothing, spend nothing, and let the hands I still have race the atoms.  It is
deliberately the **same analytic object** every play candidate is measured with, so "wait vs
play this hand" is apples to apples: passing keeps the cards and the hand, playing buys a
known score `s` for one hand and `m` fresh cards.  Because it holds the whole hand, its floor
is the best made play in hand — so PASS is worth roughly "play my best hand, later", which is
the honest reading of waiting one turn.

**Where PASS's value actually comes from**, in order:

1. **Hands are options.**  In the atoms where the opponent overtakes, `V(need, h)` beats
   `V(need − s, h − 1)` whenever `s` is a weak hand — so PASS wins exactly when the board has
   nothing worth spending a hand on, and loses when it does.  That is the economics of a war
   of attrition, and it is already in the tail DP; nothing new was added for it.
2. **Held-card money** (once §5's gate is open): a play that dumps a Gold-enhanced card
   forfeits its end-of-round $3 and carries a negative `_gold_delta`; PASS carries no money
   term at all.  So conserving and harvesting are compared **in dollars**, not by fiat.
3. **The tie-break**, `pvp_pass_tiebreak = 1e-6`: when the two are numerically equal — the
   decided-won case, both 1.0 — PASS wins, because it spends no hand, breaks no Glass card
   and forfeits nothing.  It is the same order of magnitude as the objective's existing
   `1e-6` play tie-break and cannot outrank a real difference.

**Deliberately NOT priced.**  (a) **Glass survival**: a Glass card that scores has a 1-in-4
chance of shattering, and what a permanently thinner deck costs two antes later is V's job,
not a dollars constant — inventing one here would be the least defensible number in the
layer.  PASS conserves Glass as a *side effect* and the h2h measures it as a behaviour delta
(§8).  (b) **The information gain** from watching the answer before committing: real, and it
needs a game tree we deliberately do not build at level 1.

**Plumbing safety.**  `pvp_pass` is a MATCH action; `BalatroGame.step` has never heard of it.
It is therefore emitted **only** through `rank_hand_actions(allow_pass=True)`, which is
reached only from `EVPlayer._rank_hand(allow_pass=True)`, which is reached only when the
MATCH offered one via `act(extra_actions=…)`.  Every path that steps a bare game — the full
budget's rollouts, `play_out_blind`, the blind model, `hand_ev` — calls the ranking without
the flag and cannot see it (`test_pass_never_leaks_into_a_path_that_steps_a_bare_game`).
`allow_pass` is ignored for `budget="full"` on purpose: the full budget values a candidate by
*stepping* it, and a pass would silently be scored as "the same position, played on" — i.e.
as the best play — making the comparison meaningless.  **The protocol player is a fast-budget
player.**

The match also has the last word on legality: the analysis derives leadership from the two
scores it can see (the same two `MLBMatch.pass_offered` compares), and `EVPlayer.act` returns
a pass only if the match actually offered one — so the anti-wedge streak of §2.6 can never
produce an illegal action.

---

## 5. The extraction pivot: money at a DECIDED Nemesis (`cfg.pvp_extract`, default OFF)

**Before:** `extraction_safe` was `False` at every Nemesis, unconditionally — "there is no
unused-hand money at a PvP blind and every hand is played anyway" (`EXTRACT_NOTES.md` §4).

**Why that was half right.**  The premise holds while the race is live.  It stops holding the
moment the race is over, and three facts from §1.6 say so: a Nemesis **always** reaches Cash
Out, won or lost (`lovely/game.toml:149-153`); only the *unused-hand* money row is patched
out, the **discard** row is not (`lovely/game.toml:96-98`); and per-action procs — Gold seal,
Lucky, Business Card, Purple seal → Tarot, Mail-In Rebate, Faceless, Trading Card — pay at a
PvP blind exactly as anywhere else.  A player who cannot reach the opponent's score is
holding hands and discards that are worth **nothing** to the race and something real in
dollars.

**The amended gate.**  At a Nemesis, `extraction_safe` no longer asks the tail DP for a clear
probability (a Nemesis has no chip target, so the question is malformed).  It asks
`pvp_decided()`:

```
race = race_value()                       # §4: P(I do not lose), position as it stands
decided = "lost"  if race <= pvp_decided_lost_max (0.02)   and game.lives > 1
        = "won"   if race >= pvp_decided_won_min (0.995)   and I am strictly ahead
        = ""      (LIVE) otherwise
```

* **decided-lost** — the brief's "tail DP says no remaining line reaches the opponent's
  reachable score".  `race_value` is exactly that tail-DP quantity.  The life is going; the
  money is not, so the remaining hands and discards are spent harvesting procs instead of
  chasing a score that cannot arrive.
* **decided-won** — the leader-passing case.  Nothing that happens now can cost the life, so
  the same layer opens and PASS (which spends nothing) is priced against the harvest.
* **`lives > 1` on the lost side** — a decided-lost Nemesis on the LAST life is `loseGame`:
  GAME_OVER at once, **no Cash Out** (`MLB_NOTES.md` §1.3c).  The dollars would never be
  banked, so the gate stays shut.  (Pinned: `test_the_last_life_suppresses_the_harvest`.)
* **Conditioning probability = 1.0.**  The regular-blind path multiplies the gold-hold and
  cycle terms by `P(clear after the line)` because a failed blind never reaches the payout.
  At a Nemesis the payout always arrives, so the factor is 1.

**Conservative by construction.**  Both thresholds are far from 0.5; and `race_value` uses
the level-1 atoms when they are on, which makes the trailer look *better*, so "lost" is
harder to declare than level-0 would make it.  Everything is `HandConfig`, so a sweep is a
one-line change.

**What is gated and what is not** — same split as `EXTRACT_NOTES.md` §4.  Gated: the money
term in `evaluate()`, `_extraction_discard_lines`, and `extraction_lines()` (the W-PAIRS
interface, which now reports `decided-lost` / `decided-won` in its reason string).  Not
gated: the proc-aware junk *orderings*, which only decide which of two structurally
equivalent cards goes.

### 5.1 The fixture (`ev/fixtures/nemesis_decided_lost.py`)

A real `MLBMatch` walked to the real ante-2 Nemesis (regular blinds cleared with
`debug_win_blind`, the Nemesis started through the match's own `readyBlind` → `startBlind`),
then the `_probe_common` attribute-write pattern: a plain no-joker board, two Purple-sealed
junk cards at indices 5/6, both consumable slots free, 3 lives.  The opponent's score is set
on `games[1].chips_scored` and **relayed by `MLBMatch.sync()` through the real
`set_pvp_info`** — player 0 learns it exactly the way `enemyInfo` delivers it.

| | `build()` — decided-LOST (opp 10,000,000) | `build_control()` — LIVE race (opp 300) |
|---|---|---|
| `extract_on` | True | True (*there is something to extract — the gate is what stops it*) |
| `race_value()` | **0.0000** | **0.6205** |
| `pvp_decided()` | `"lost"` | `""` |
| top action, gate open | **`discard [5, 6]` @ EV 0.096** (two Tarots, $8.00) | `play [3,4,5,6,7]` @ EV 0.6105 |
| top action, `pvp_extract=False` | `play [0]` @ EV −0.0 (the futile chase) | `play [3,4,5,6,7]` @ EV 0.6105 |
| ranking vs `pvp_extract=False` | differs (harvest > chase by **+0.096**) | **bit-identical** |
| `extraction_lines()` | `discard [5,6]` — "extract $8.00 (Nemesis decided-lost)" | `[]` |

The harvest line is worth 0.096 in P(clear) units — $8 of Tarot at the objective's own
`beta_hand · (1 + interest_bonus)` exchange rate — against a chase worth 0.  That is the
whole pivot in one number.

---

## 6. Configuration

| field | default | meaning |
|---|---|---|
| `MLBMatch(pvp_protocol=)` | `"canonical"` | `"trailer_compelled"` enables the leader's wait |
| `HandConfig.pvp_level1` | `False` | react to the opponent's revealed live score (§3) |
| `HandConfig.pvp_live_weight` | `0.5` | weight of the "they never play again" atom when trailing |
| `HandConfig.pvp_pass` | `False` | generate the leader's PASS candidate (§4) |
| `HandConfig.pvp_pass_tiebreak` | `1e-6` | PASS wins an exact tie |
| `HandConfig.pvp_extract` | `False` | allow the money layer at a DECIDED Nemesis (§5) |
| `HandConfig.pvp_decided_lost_max` | `0.02` | race value at or below this ⇒ decided-LOST |
| `HandConfig.pvp_decided_won_min` | `0.995` | … at or above, and ahead ⇒ decided-WON |

`player.protocol_hand_cfg(level1=…, pvp_pass=…, extract=…)` turns them on together (all three
default `True`); `player.adapt_match_player(obj)` is the `(match, p, acts) -> action` policy
that threads `pvp_pass` through to `act`.  Use it instead of `eval/common.adapt_player`
whenever the match runs a non-canonical protocol — with the canonical protocol `acts` never
contains one and the two adapters are identical.

---

## 7. What is still not modelled

* **Level-2**: minimal overtake, statement-hand signalling, any inference about the
  opponent's intent.  Explicitly out of scope (Tagg's ladder).
* **The early-end cut as a sequencing term** — "play the big hands first because being
  exhausted and behind ends it now".  Level-1 reaches it only indirectly (a nearer target).
* **Glass survival, joker retriggers, Reserved Parking beyond the current hand** — the
  `EXTRACT_NOTES.md` §5 omissions all still stand at a Nemesis.
* **The opponent's own protocol behaviour** — the atoms assume they use every hand they hold
  if they answer at all.  A level-2 opponent model would give them a pass too.
* **Wall-clock / speed** — deliberately omitted (§2).
* **The full budget** does not value PASS (§4) and its leaf keeps the level-0 mixture unless
  `cfg.pvp_level1` is set.

---

## 8. Measurement

Driver: `ev/scripts/pvp_protocol_h2h.py` (both seatings per seed, spawn pool,
`eval/common.bootstrap_ci`; `--procs` capped at 6).  `ev/h2h.py` could not express either arm
— it has no `HandConfig` hook (`EXTRACT_NOTES.md` §9.1's open request, still open) and no
`pvp_protocol` argument — and it is not this workstream's file to change.

```
python ev/scripts/pvp_protocol_h2h.py protocol --n-seeds 30 --procs 5 --max-steps 4000
python ev/scripts/pvp_protocol_h2h.py level    --n-seeds 30 --procs 5 --max-steps 4000
```

Results: `results/pvp_protocol_on_vs_off_2026-08-26.{json,md}` and
`results/pvp_level1_vs_level0_2026-08-26.{json,md}`.  Numbers and reading: §9.

---

## 9. Results

### 9.1 How often the protocol and the gate actually fire

Instrumented self-play, 8 seeds, protocol ON both sides, level-1 + PASS + `pvp_extract`:

| | count | share of Nemesis hand-decisions |
|---|---|---|
| Nemesis hand decisions | 643 | — |
| race **LIVE** | 604 | **93.9%** |
| decided-**lost** | 28 | **4.4%** |
| decided-**won** | 11 | **1.7%** |
| leader passes taken | 37 | — |

The gate is **not inert, and it is rare** — which is what a conservative gate is supposed to
look like.  94% of the time the player is racing and the extraction layer is shut, exactly as
before this workstream.

### 9.2 The protocol: ON vs OFF, self-play both arms

`results/pvp_protocol_on_vs_off_2026-08-26.{json,md}` — 30 seeds × 2 seatings = 60 matches
per arm, `ev:fast` both sides, ε = 0, lives 4, `--max-steps 4000`, 5 workers.

| metric | **protocol ON** | protocol OFF | Δ |
|---|---|---|---|
| matches (decided / undecided) | 60 / 0 | 60 / 0 | — |
| A win rate | 0.500 [0.383, 0.633] | 0.500 [0.367, 0.617] | — (self-play: exactly 0.5 by construction) |
| **P(seat 0 wins)** — seat-bias check | **0.567** | **0.300** | +0.267 |
| Nemeses resolved | 260 | 274 | −14 |
| **leader passes / match** | **5.53** | **0.00** | +5.53 |
| **early-end cuts / Nemesis** | **0.315** | **0.139** | **×2.3** |
| early-end cuts / match | 1.37 | 0.63 | +0.74 |
| **hands played / Nemesis** | **3.89** | **4.07** | **−0.18** |
| discards taken at Nemeses / match | 17.28 | 18.35 | −1.07 |
| Glass cards alive at the end | 0.15 | 0.17 | −0.02 |
| $ banked inside a LOST Nemesis | 1.07 | 1.18 | **−0.11** |
| mean final ante | 5.97 | 6.10 | −0.13 |
| mean final $ | 29.72 | 33.20 | −3.48 |
| mean steps / match | 379 | 389 | −10 |

**Reading, honestly.**

* **The win rate cannot say anything here and is not meant to.**  Both arms are self-play with
  ε = 0, so the two players are the same deterministic function of the same seed: seating 1 is
  the mirror of seating 0 and A wins exactly 30/60 by construction.  Its purpose is the
  symmetry sanity check, and it passes.
* **The seat-bias number is the interesting one.**  Under canonical alternation, self-play on
  this seed set is *not* symmetric: seat 1 wins 70% of matches.  Under the protocol it moves
  to 43% — closer to fair.  That is the right direction (a leader who can wait no longer
  hands the second mover a free look), but 30 seeds cannot establish it: the CIs are 25
  points wide and this is a between-arm comparison of two point estimates.  **Suggestive, not
  established.**  Worth a 126-seed rerun if the lead wants it.
* **The protocol is being exercised and it does what it was designed to do.**  5.5 leader
  passes per match, hands per Nemesis down 0.18 (conservation), and — the headline behaviour
  delta — the **early-end cut rate more than doubles, 0.139 → 0.315 per Nemesis**.  That is
  the mechanism working end to end: the leader waits, the trailer burns hands, the trailer
  runs out behind, the round is cut with the leader's hands still in hand.
* **Glass is degenerate at this sample.**  0.15 vs 0.17 Glass cards alive per run — the red
  deck starts with none and they arrive rarely.  The metric is reported because the brief
  asked for it; it carries no signal at n = 60 and should not be quoted.
* **The extraction pivot does NOT show up as money at the match level, and slightly the
  other way: −$0.11 per match banked inside a lost Nemesis.**  The explanation is in the same
  table: the protocol cuts 2.3× more Nemeses early, and a trailer who is cut off plays fewer
  hands, so it fires fewer procs.  The decided-lost harvest (4.4% of decisions, §9.1) does not
  make that back.  **The pivot is right where it applies — §5.1's fixture is unambiguous — and
  is close to irrelevant in aggregate self-play.**  It should be judged on the fixture, not on
  this row.
* **Final ante 5.97 vs 6.10 and final $ 29.72 vs 33.20** follow from the same thing: more
  decisive Nemeses ⇒ lives change hands sooner ⇒ shorter matches ⇒ less accumulated money.
  Not a regression in play quality; in self-play nobody loses.

### 9.3 Attribution: level-1 vs level-0, protocol ON for both sides

`results/pvp_level1_vs_level0_2026-08-26.{json,md}` — same harness; A = level-1 objective,
B = level-0 (the symmetric atoms); PASS and the decided-race gate on for **both**, so the
objective is the only difference.

| metric | A = level-1 | B = level-0 |
|---|---|---|
| A win rate | **0.500** [0.383, 0.633] (30 / 60) | 0.500 |
| A win rate by seating (0 / 1) | 0.533 / 0.467 | — |
| mean lives margin (A − B) | **−0.017** | — |
| Nemeses resolved | 262 | |
| leader passes / match (both sides) | 5.78 | |
| early-end cuts / Nemesis | 0.332 | |
| hands played / Nemesis | 3.89 | 3.88 |
| discards at Nemeses / match | 17.55 | 17.33 |
| $ banked in a LOST Nemesis | 1.22 | 0.93 |
| mean final ante | 5.98 | 5.98 |
| mean final $ | 30.52 | 29.82 |

**Does reacting to the revealed score win matches?  At this sample size: no measurable
effect.**  30 / 60 with a 95% CI of [0.383, 0.633] and a lives margin of −0.017 is a wash in
every direction.  Every behaviour column is inside noise as well; the largest is $1.22 vs
$0.93 banked in a lost Nemesis (+31% relative, +$0.29 absolute, on 60 matches).

This is the honest answer to the brief's question, and it is not a surprising one: level-1
changes the *needs* the mixture is evaluated at, but both players are drawing from the same
deck on the same seed and the level-0 atoms are already centred on the truth.  What level-1
buys is a better-shaped objective for the situations the protocol newly creates (a leader who
may sit on a lead), and there are ~5.8 passes per match in which to exercise it — too thin a
slice to move a 4-life match outcome.  A 25-point-wide CI cannot rule out a real effect of a
few points either.  **Do not quote this as "level-1 does not help"; quote it as "60 matches
cannot separate them".**

### 9.4 Ops

Both runs used 5 worker processes and `--max-steps 4000` (the driver refuses `--procs > 6`).
`ev/runs/pairs_v3_hires/` and `ev/runs/sweep_rank/` were never read or written.  No commits.
