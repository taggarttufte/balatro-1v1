# GATE_NOTES — Phase 2 exit gate (W4, 2026-08-21)

**Agent W4.**  Files: `scripts/mlb_match_demo.py` (match driver + trace + RNG-key tools),
`tests/test_mlb_match_gate.py` (507 tests), three additions at the end of
`tests/test_engine_invariants.py` (22 tests), this note.  Nothing under `engine/` or `rng/`
was touched; no engine bug was fixed (none found — see §6).

## 0. Gates (final run, repo root, `python` = 3.13)

| gate | result |
|---|---|
| `python -m pytest engine/tests -q` | **1609 passed / 10 skipped / 3 xfailed / 0 failed** (unchanged from W1/W2/W3 hand-off) |
| `python -m pytest tests -q` | **1073 passed / 2 xfailed / 0 failed** (544 + 507 gate + 22 invariants) |
| `python -m oracle.engine_parity --antes 1-8 --rerolls 5 --quiet` | **126/126 EXACT through ante 8** |
| `python -m oracle.parity_check --antes 1-8 --variant faithful` | **126/126 EXACT through ante 8** |
| `python scripts/mlb_match_demo.py --seed 7I4M53DL` | full match trace, P2 wins at ante 2 (117 steps) |
| `python -m pytest tests/test_mlb_match_gate.py -q -rx` | **507 passed / 0 xfail / 0 failed**, ~27 s |

## 1. How to run

```
cd C:/Users/Taggart/projects/balatro-rl
python scripts/mlb_match_demo.py --seed 7I4M53DL                       # readable trace (per ante: blinds,
                                                                           #   cash-outs, lives, comeback, shops,
                                                                           #   Nemesis verdicts) + summary
python scripts/mlb_match_demo.py --seed ALEEB --deck b_plasma --stake 2 --max-antes 4 --quiet
python scripts/mlb_match_demo.py --seed 7I4M53DL --alignment            # + per-visit RNG key diff P1 vs P2
python scripts/mlb_match_demo.py --seed 7I4M53DL --json trace.json      # dump every recorded event
python -m pytest tests/test_mlb_match_gate.py -q -rx                    # the gate (≈27 s)
python -m pytest tests/test_engine_invariants.py -q                     # incl. the Phase-2 invariants
```

Players in the demo: **P1 "opener"** (greedy best-hand play; never rerolls; opens booster slot 0 at
every shop and picks the first grantable card) vs **P2 "reroller"** (greedy; rerolls once per shop
when affordable; buys shelf slot 0 when affordable; never opens packs).  Neither skips, sells or
uses consumables.  `--deck` any `b_*`, `--stake` 1-8.

The module is importable (`sys.path` + `import mlb_match_demo`): `ScriptedPlayer` (a policy
description: hand style `greedy` / `weak` / `greedy_until`, rerolls per visit, buy slot 0, open
pack slot, buy voucher, `debug_win_regular`, `rich`), `make_policy(spec)`,
`MatchRecorder(seed, [specA, specB], deck_key, stake, lives, max_antes).run()` which drives
`MLBMatch` under the canonical turn order and records `blinds`, `cashouts`, `lives`, `pvp`,
`visits[p]` (shelf at entry + after each reroll, voucher, packs, opened packs + picks, purchases,
`rng_entry` / `rng_exit` = `run_state.rng.snapshot()["state"]`, shadow boss, tags, owned keys),
and the RNG tools `key_position`, `classify_key`, `diff_rng`.

`key_position(seed, key, value)`: the per-key state of `core.PseudoRandom` depends ONLY on how
many times `pseudoseed(key)` was called (`x = lcg_step(x)` from `pseudohash(key..seed)`), so a
state value maps to an exact **queue position** (0 = never drawn).  The diff is therefore a
diff of integer positions, not of floats.

## 2. Scenarios and what each gate item proves

All scenarios run on 10 ground-truth seeds × 3 decks (Red / Checkered / Plasma, White stake) =
30 matches each, all driven through `MLBMatch.step()` in the canonical `current_player()` order.
`debug_win_regular` uses the harness hook `debug_win_blind()` (chips := target, no stream touched)
so a match can last; every Nemesis is always played for real.

| scenario | players | exercises |
|---|---|---|
| `natural` | greedy vs one-card player, no shopping | regular-blind losses (both), failed-blind Cash Outs, match ends at ante 2 |
| `pvp` | both debug-win regular blinds; P0 one-card at every Nemesis | PvP losses, early end, $5 at a lost/won Nemesis, comeback at the same Cash Out |
| `endless` | both debug-win; identical greedy Nemesis play through ante 8 (exact ties), P0 one-card from ante 9 | ties cost nobody, antes 9-12 played, match ends at ante 12 at 0 lives, blind targets follow the endless formula |
| `shoppers` | the demo's opener vs reroller, regular blinds debug-won | queue alignment, first-shelf identity, clone |
| `voucher_match` (5 seeds, 12 lives, ≤ 16 antes) | voucher buyer (unlimited money) vs abstainer | the shared `Voucher0` stream (§4) |

**Item 1 — lives** (`TestLives`, 150 tests).  Over all four scenarios: every life event is a
decrement of exactly 1; at most one per (player, ante, blind); a lost regular blind with ≥ 1 hand
played has exactly one life event of THAT player at that step and none of the other; a won blind
has none; a decided Nemesis has exactly one life event, of the strictly-lower scorer; a tie has
none; `#life events == #lost regular blinds + #decided Nemeses`.  Comeback: at every Cash Out,
`comeback_expected = 4 × cumulative lives lost` iff a life was lost since that player's previous
Cash Out (else 0), and the sequence paid is exactly `[4, 8, 12, …]`, one payout per loss that is
followed by a Cash Out (the 4th loss is GAME_OVER, no Cash Out).  The match ends when a player hits
0 lives (last life event = final step, loser GAME_OVER, winner `match_won`), never at ante 8: the
endless scenario plays antes 9-12 (Small 110 000 / 560 000 / 7 200 000 / 300 000 000, ×2 under
Plasma) and ends at ante 12.  Early end: in `pvp`, P0 exhausts its 4 hands strictly behind; the
verdict lands at the step of P0's last hand; whenever P1 still has a hand the verdict is flagged
early-end and P1's blind record shows fewer than 4 hands used (forfeited); ≥ 1 early end per match.
Whether P1 still has a hand when P0 exhausts depends on who moved first in the alternation and on
P1's discards (MLB_NOTES §3.2) — both outcomes occur and both are asserted.

**Item 2 — money** (`TestMoney`, 60 tests).  In the joker-free scenarios the Cash Out delta is
exactly `reward + unused-hand money + interest + comeback`.  Nemesis: reward 5 win or lose,
unused-hand money 0 even when the early-ended winner has hands left (asserted to occur), interest
`min(pre // 5, 5)` as vanilla.  Failed regular blind: reward 3/4/5 by blind, 0 hand money, comeback
> 0, interest normal.

**Item 3 — queue alignment** (`TestAlignment`, 126 tests) — §3 below.

**Item 4 — vanilla unchanged** (`TestVanillaUnchanged`, 128 tests): `engine_parity --antes 1-8
--rerolls 5` invoked in-process → 126/126; and for all 126 ground-truth seeds the same scripted
policy (8 antes, packs opened, 5 rerolls/ante) on `ruleset="vanilla"` vs `ruleset="mlb"` differs
ONLY in (a) the voucher (every ante may differ: `Voucher0` path), (b) slots whose vanilla item is a
banned key (in-place replacement) plus the second-order offset they cause (§5), (c) tags where
vanilla drew `tag_boss`, (d) the ante ≥ 2 Boss slot (Nemesis, `is_pvp`) and the shadow boss from
ante 2 (`bl_wall` leaves the candidate list; `boss` stream position asserted equal; ante-1 boss
asserted equal); no banned key anywhere on the MLB side; `shuffle` position equal.  Pack kinds,
shop queue length and every other slot are identical.

**Item 5 — clone** (`TestClone`, 13 tests): the shoppers' match is played until the first Nemesis
is live for both, `clone()`d, and both copies are driven by fresh copies of the same policies:
`signature()` equal after every step until the end (≥ 2 Nemeses), same winner and `pvp_log`;
stepping a clone leaves the original's signature unchanged.

**Cross-cutting** (`test_engine_invariants.py` §6): no `random.Random` / `import random` /
`np.random` in any game-logic module after Phase 2 (envs excluded; `secrets` for the seedless
constructor); the default ruleset is byte-identical to `ruleset="vanilla"` and distinct from
`"mlb"`; over 20 seeds × 4 antes × 3 rerolls with every pack opened an MLB game never shows a
banned joker / voucher / tag / blind on a shelf, in a pack, as voucher, tag or (shadow) boss draw,
and the Boss slot from ante 2 is the Nemesis.

## 3. Queue-alignment check: key classification

At every shop-visit ordinal *i* (both players visit the same shops in the same order — no skips —
so ordinal *i* is the same (ante, after-blind) for both) the two `rng.snapshot()["state"]`
dicts are diffed at **entry** (shelf as shown, before any reroll) and at **exit** (after
`leave_shop`).  Every key whose position differs is classified by name; a class that is not
explicitly allowed (SHARED, VOUCHER, UNKNOWN) is a violation, and the allowed classes carry a
quantitative consistency rule.  `classify_key` in `mlb_match_demo.py`; rules in
`alignment_violations` in the test.

| class | keys (`<a>` = ante) | rule at equal visit ordinal |
|---|---|---|
| **SHARED** | `boss`, `Tag<a>` (+ `Tag<a>_resample<it>`: the `tag_boss` ban, symmetric), `shuffle`, `erratic`, `nr<a>`, `cashout<a>`, `idol<a>`, `mail<a>`, `anc<a>`, `cas<a>`, `orbital`, `shop_pack<a>`, `Voucher_fromtag` | positions MUST be equal (run structure: stepped once per run start / blind / round / ante by both). Asserted present on both sides too. `shop_pack<a>` absolute = 2 × shops visited in ante a (−1 in ante 1: the forced first Buffoon consumes nothing). |
| **VOUCHER** | `Voucher0`, `Voucher0<it>` | equal in the shoppers scenario (no voucher bought); see §4 for when it legitimately diverges |
| **OWN_SHOP** | `cdt<a>`, `rarity<a>sho`, `Joker[1-3]sho<a>`, `edisho<a>`, `Tarotsho<a>`, `Planetsho<a>`, `Spectralsho<a>`, `frontsho<a>`, `Enhancedsho<a>`, `etperpoll<a>`, `ssjr<a>` | `cdt<a>` position == shelf slots drawn by that player in ante a (Σ shelf sizes at entry + per reroll) — asserted for BOTH players at EVERY visit, absolutely. Other shop keys: equal slot counts ⇒ equal positions; otherwise the player with more slots is ahead. |
| **OWN_PACK** | `Joker[1-3]buf<a>`, `rarity<a>buf`, `edibuf<a>`, `packetper<a>`, `packssjr<a>`, `Tarotar1<a>`, `Spectralar2<a>`, `Planetpl1<a>`, `Spectralspe<a>`, `frontsta<a>`, `Enhancedsta<a>`, `stdset<a>`, `standard_edition<a>`, `stdseal<a>`, `stdsealtype<a>` | a player who opened no pack in ante a has position 0 |
| **OWN_ANY** | `soul_Tarot<a>`, `soul_Planet<a>`, `soul_Spectral<a>` (no key_append: 2 rolls per consumable created by shop slot, pack card OR joker effect) | the player ahead drew more shelf slots, opened a pack, or owns something |
| **OWN_RESAMPLE** | `<pool>_resample<it>` where `<pool>` is OWN_SHOP / OWN_PACK / OWN_ANY / PER_PLAYER | in-place redraw on a draw that hit an UNAVAILABLE entry (own collection, item currently displayed, a ban, a hidden-hand planet, a locked joker); which draws a player makes is their own consumption |
| **PER_PLAYER** | effect rolls `lucky_mult`, `lucky_money`, `glass`, `bloodstone`, `8ball`, `business`, `gros_michel`, `cavendish`, `parking`, `space`, `misprint`, `halu<a>`, `wheel`, `madness`, `aajk`, `cerulean_bell`, `hook`, `crimson_heart`, `invisible`, `perkeo`, `to_do`, `random_destroy`, `marb_fr`, `cert_fr`, `certsl`, `spe_card`, `illusion`, `omen_globe`, `hex`, `ankh_choice`, `ectoplasm`, `immolate`, `sigil`, `ouija`, `aura`, `Joker4`; created-card keys with append `8ba hal car vag sup sea sixth pri emp jud sou wra rif top rta uta fool blusl` (+ their `rarity<a><app>`) | depend on that player's own play; allowed by name |
| **UNKNOWN** | anything else | violation (the test names the key) |

Observed over the 30 shoppers matches (exit snapshots): OWN_SHOP 21 313, OWN_PACK 17 585,
OWN_RESAMPLE 8 427 (4 441 at exit), OWN_ANY 3 496, PER_PLAYER 345 (`gros_michel`, `to_do`,
`lucky_*`, `bloodstone`, `8ball`), SHARED / VOUCHER / UNKNOWN **0**.  A teeth test perturbs one
`boss`, one `cdt<a>` and `Voucher0` value and checks each is reported.

First shelf of every ante (entry of the post-boss shop / ante-1 after-Small), packs, voucher,
shadow boss and tags are asserted **identical** for both players, slot by slot, except a slot
where the other player's item is in this player's collection (in-place resample; 21 of 342 slots
over the 30 matches) — with a second-order allowance (same pool after such a slot on the same
shelf: the `_resample` side stream is one step ahead) that never triggered in these matches.

## 4. The shared voucher stream (finding, by design of the mod)

Under MLB both players draw the ante's voucher from the run-global culled `Voucher0` stream
(brief §1.6, W2's port of `TheOrder.lua:481-525`).  Measured with a voucher buyer vs an
abstainer (5 seeds, up to 16 antes):

1. **Buying does NOT step the stream.**  `Voucher0` positions stay equal for both players
   (1, 2, 3, … = one draw per ante) whatever the buyer buys.  Both players are offered the same
   voucher PAIR every ante; where the buyer owns the base, the culled pool shows it the upgrade
   and the abstainer the base (e.g. `7I4M53DL` ante 5: Palette vs Paint Brush; ante 7: Overstock
   Plus vs Overstock).  A Voucher Tag draw (not exercised: no skips) would step it for that player
   only (`_from_tag` uses the same key under MLB).
2. **The stream diverges once a player owns BOTH tiers of a pair.**  That pair collapses to
   `UNAVAILABLE` for them and the draw loops on the same stream, so the buyer steps `Voucher0`
   extra times; from then on the two players see different vouchers for the rest of the run
   (`7I4M53DL`: diverged at ante 11, positions 14 vs 12; `ALEEB`: ante 5 after Clearance Sale +
   Liquidation, 7 vs 6; `11111111` / `1558AXDL`: ante 15; `15H9Z3IY`: never within 13 antes).
   Banned pairs (Hieroglyph/Petroglyph, Director's Cut/Retcon) are `UNAVAILABLE` for both and
   cause symmetric redraws (both players stepped 10 → 12 at `7I4M53DL` ante 11).
   The test asserts: positions equal and same pair offered until the buyer completes a pair;
   afterwards the buyer is strictly ahead; divergence never happens without a completed pair.

This is faithful to the mod (W2's oracle executed the mod's Lua verbatim).  Consequence for
Phase 3: "same seed ⇒ same shop" holds for shelves/packs/tags/boss always, and for vouchers only
until someone completes a voucher pair.

## 5. Second-order effect of bans / ownership on the `_resample` side streams (finding)

A draw that hits an `UNAVAILABLE` pool entry redraws on `<pool>_resample<it>`, a side stream
shared by every redraw of that pool in that ante and area.  A ban (or an owned card) that
forces a redraw therefore advances the side stream, and a LATER unrelated redraw in the same
ante / area / pool (a locked joker, a hidden-hand planet, a displayed card) yields a different
replacement than it would without the ban.  Seen 8 times in the shop queue over 126 seeds × 8
antes of the MLB-vs-vanilla comparison (all uncommon jokers after a Mr Bones / Luchador /
Matador replacement in the same ante, e.g. `1MD1YZ9T` ante 7 slot 10: `j_seance` → `j_sixth_sense`,
`Joker2sho7_resample2` at position 1 vs 2).  The comparison allows exactly this: a non-banned
differing slot must be from the same pool as an earlier banned slot in the same ante and area.
Not an engine bug — it is what `create_card`'s resample loop does in the game.

## 6. Findings

* **No engine bug found.**  Every gate assertion passed on the first run that did not contain a
  harness mistake; the only engine-side surprises were §4 and §5, both faithful.
* Harness mistakes fixed along the way (documented so nobody re-discovers them): the recorder
  initially missed transitions caused by policy side effects (`debug_win_blind` inside the
  policy) and recorded the winner's Nemesis blind as "lost with 0 hands" when `winGame` cut it;
  `_ante_of_key` did not parse `rarity<a>sho`; the early-end assertion was interleaving-blind.
* The early-end rule's observable (who still has hands when the other exhausts) is
  interleaving-dependent (W1 §3.2); the engine's canonical alternation makes it deterministic,
  and an env that steps both players per tick would see a different cut point but the same
  verdict.  Phase 3's obs / reward must not assume "winner always keeps ≥ 1 hand".
* `env_v7._finish_step`'s `R_BLIND_BASE * (9 - ante)` (W1 §5) is negative past ante 8; the
  endless scenario here reaches ante 12 — relevant once Phase 3 trains on MLB.
* Not exercised by this gate (no scenario): skips / tags under MLB (Voucher Tag stepping
  `Voucher0` per player, Uncommon/Rare tags stepping `rarity<a>uta/rta`), selling, consumable
  use, deck-out at a Nemesis, both players failing the same blind at 1 life, stakes ≥ 2 in the
  money assertions (demo runs them; tests use White), Overstock changing `cdt` slot counts
  (the absolute `cdt` check uses recorded shelf sizes, so it would still hold).
