# EXTRACT_NOTES — W-EXTRACT: the sandbag / money-extraction layer (Phase 5 rev 2, 2026-08-24/25)

Files: `ev/hand.py` (the layer: `ProcBoard`, the per-card proc arrays, `extraction_ev`,
`extraction_lines`, the safety gate, the candidate lines), `ev/player.py` (the advisor's
money decomposition only), `engine/balatro_sim/{game.py, jokers/base.py, jokers/misc.py}`
(four proc-fidelity fixes), tests `ev/tests/test_extraction.py` (35), driver
`ev/scripts/extract_dev_slice.py` (new, mine).  Nothing else was touched — `ev/h2h.py`,
`gate_ev_player.py`, `pairs.py`, `train_v.py`, `advisor.py` are read-only here (see §9 for
the two interface requests).

Read `EV_NOTES.md` §1 first: this document only describes what changed.

## 0. Headline

| gate | before | after |
|---|---|---|
| 126-seed `gate_ev_player.py --procs 8`, **fast** ante-1 clear | **95.2%** [91.3, 98.4] | **95.2%** [91.3, 98.4] |
| … **full** ante-1 clear | **96.0%** [92.1, 99.2] | **96.0%** [92.1, 99.2] |
| … fast mean final ante | 4.69 | **4.76** |
| … full mean final ante | 4.91 | **4.98** |
| … fast hand ms mean / p95 | 4.02 / 12.35 (greedy probe 2.32) | 5.45 / 15.17 (greedy probe 2.65) |
| 12-seed extraction dev slice, $ at ante 3 | 18.75 | **20.33** (+1.58) |
| … tarots used | 1.08 | **1.33** (+0.25) |
| … blinds cleared / runs died | 7.00 / 0.167 | **7.00 / 0.167** (not up) |
| same slice run to ante 6, $ | 23.58 | **36.08** (+12.50) |
| … blinds cleared | 8.83 | **9.83** (+1.00) |
| paired h2h new-fast vs old-fast, 30 seeds × 2 seatings | — | **50.0%** [36.7, 61.7] |
| … same at 126 seeds × 2 seatings (252 matches) | — | **50.4%** [44.4, 56.7] |
| engine `engine_parity --antes 1-8 --rerolls 5` | 126/126 | **126/126** |

Full 126-seed gate detail (point estimates; n = 126, so 1 seed = 0.79 pp):

| | fast before | fast after | full before | full after |
|---|---|---|---|---|
| ante-1 clear | 0.9524 | **0.9524** | 0.9603 | **0.9603** |
| ante-2 clear | 0.8333 | 0.8175 | 0.8810 | 0.8730 |
| ante-3 clear | 0.7698 | 0.7619 | 0.7937 | 0.8016 |
| ante-4 clear | 0.6032 | 0.5952 | 0.6825 | 0.6825 |
| matches won | 0.0159 | **0.0317** | 0.0159 | **0.0317** |
| mean final ante | 4.690 | **4.762** | 4.913 | **4.984** |
| mean blinds cleared | 9.563 | **9.714** | 10.198 | **10.310** |
| $ at ante 3 | 19.75 | **20.43** | 19.91 | **20.29** |
| hands unused / cleared blind | 2.163 | 2.127 | 2.160 | 2.133 |
| draw-order invariance | 743/743 | 743/743 | 740/740 | 741/741 |

Ante-1 is bit-identical (the gate requirement).  The ante-2/3/4 wobbles are 1–2 seeds each
and go both ways; every aggregate that measures depth or money moved up, and unused hands
per cleared blind moved *down* — the player banks slightly fewer hands, which is exactly
what sandbagging is supposed to look like.

Timing: the two gate runs are not load-comparable — the *unchanged* scripted-greedy player's
hand time moved 2.32 → 2.65 ms in the same runs (a 1.14× box-load factor), which puts the
fast budget at ≈ **4.8 ms** normalised.  The clean measurement is a warm A/B on **identical
states** (`rank_hand_actions` with `extract=True` vs `False`, 260 decisions over 8 seeds to
ante 4): 2.446 vs 2.467 ms mean — **−0.9%, i.e. no measurable per-decision overhead**.  In
the dev slice (single process per run, both arms in the same batch) the ante-6 numbers are
4.932 vs 4.916 ms mean.  What the gate's +0.7 ms really is: a *trajectory* difference — the
extraction player owns more tarots, and every held consumable costs a `use_consumable`
evaluation per hand decision.

## 1. Engine fidelity — the audit and the four fixes

Every §1-brief proc was checked line-by-line against `_reference/balatro_src/`.  Five were
already faithful; four defects were found and fixed.  The Lua citations below are repeated in
the test docstrings (`ev/tests/test_extraction.py`).

| proc | Lua | engine | verdict |
|---|---|---|---|
| Gold **seal** $3 when played and scores | `card.lua:1068-1073` (`get_p_dollars`), read only for `cardarea == G.play` (`common_events.lua:608-611`), paid `state_events.lua:722-726` | `scoring.py:99-101` | MATCH (per pass, retriggers pay again, debuffed skipped) |
| **Lucky** money 1-in-15 → $20 | `card.lua:1076` | `scoring.py:85-87` | MATCH (key `lucky_money`, denom 15, `probabilities.normal` numerator, both rolls every pass) |
| **Business Card** face 1-in-2 → $2 | `card.lua:3175-3184` | `jokers/economy.py:30-34` | MATCH (a debuffed scoring card never reaches the `individual` context at all — `state_events.lua:655`) |
| **Golden Ticket** $4 per played Gold card | `card.lua:3150-3158` | `jokers/economy.py:36-41` | MATCH |
| **Purple seal** → Tarot on discard | `card.lua:2242-2268`, before the joker `discard` hooks (`state_events.lua:400-409`) | `game.py:_discard` | MATCH in substance (free-slot gate, debuff skip, `'8ba'` stream, ordering). One benign nuance: Python does all seals then one joker sweep, Lua alternates per card — only observable if a joker's discard hook competed for a consumable slot, which no vanilla joker does. |
| **Reserved Parking** 1-in-2 → $1 per held face | `card.lua:3302-3319` + `is_face` `card.lua:964-970` | `jokers/misc.py:_ReservedParking` | **BUG → FIXED** |
| **Faceless** $5 for 3+ discarded faces | `card.lua:2858-2872` | `jokers/misc.py:_Faceless` | **BUG ×2 → FIXED** |
| Gold **enhancement** $3 held at round end | `card.lua:1033-1039` (`h_dollars`), loop `state_events.lua:171-233`, interest `:1191-1202` | `game.py:_end_round` | **BUG ×3 → FIXED** |

### The fixes

1. **`ScoreContext.is_face_card` had no debuff guard** (`jokers/base.py:213`) — the root
   cause of the next two.  Lua's `Card:is_face()` returns nil for a debuffed card
   (`card.lua:965`), and because Lua's `and` short-circuits, that means a following
   `pseudorandom(...)` in the same chain is *never rolled*.  Added the guard where it
   belongs; the raw rank test stays available as `Card.is_face_card` for the `is_face(true)`
   call sites (The Plant / The Mark in `game.py`, which already use it correctly).
2. **Reserved Parking rolled `parking` on debuffed held face cards** (`jokers/misc.py`)
   — an **RNG-stream desync**: the real game takes no draw there, so every later keyed roll
   in the run was shifted, and a winning roll also returned `True` (= "had effect"), handing
   the card a Red-seal/Mime repetition it never earns.  Fixed by (1); the misleading comment
   ("the roll is consumed BEFORE the debuff check, exactly like the Lua") is corrected — the
   `if context.other_card.debuff` arm at `card.lua:3305-3310` is dead code in vanilla.
3. **Faceless counted debuffed faces and never saw Pareidolia.**  (1) fixes the debuff half.
   The Pareidolia half was `game._hook_ctx()` never setting `all_face_cards` — only
   `_Pareidolia.pre_score` did, and that runs *inside* `score_hand`, so **every non-scoring
   hook** (`on_discard`, `on_round_end`, …) saw Pareidolia as absent.  `_hook_ctx` now sets it.
4. **Gold enhancement at end of round** (`game.py:_end_round`): paid on debuffed cards
   (Lua `card.lua:1034` returns `{}`), ignored the Red-seal and Mime repetitions
   (`state_events.lua:191-207`, `ease_dollars` on every rep at `:221-223`), and — the one
   that changes the economy — was credited **after** the interest row was computed.  In Lua
   the held-card effects run inside `end_round()` and `G.FUNCS.evaluate_round` reads
   `G.GAME.dollars` *after* them (`state_events.lua:1191`), so Gold money counts toward the
   interest base.  $8 + two Gold cards is now interest on $14 ($2), was interest on $8 ($1).
   The block moved above `interest = …` and gained the debuff check and the rep multiplier.

Gates after the engine change: `engine_parity --antes 1-8 --rerolls 5` **126/126 exact**
(unchanged), `pytest engine/tests` **1651 passed / 10 skipped / 3 xfailed**,
`pytest tests` **1073 passed / 2 xfailed**.  The parity harness never scores a hand
(`debug_win_blind`), so it could not have caught any of these — which is exactly why the
Lua diff was done by hand.

### Found but deliberately NOT fixed (out of this workstream's scope — see §9)

* `jokers/chips.py:_Caino` uses the raw `card.is_face_card`, so it is blind to Pareidolia
  (Lua `card.lua:2626/2676` uses `v:is_face()`).  A mult joker, not a money proc.
* `game.py` fires `on_card_added` with `drain=False` — the only such call site; any
  `pending_money` / `pending_cards` written by that hook is silently dropped.  Nothing writes
  them today (`j_hologram` only), so it is a latent trap, not a live bug.
* `j_burnt` is implemented as a blind-select "most played hand" upgrade instead of Lua's
  `pre_discard` "level up the hand type of the first discard of the round"
  (`card.lua:2748-2755`).  Unrelated to money.

## 2. The proc-EV term — `extraction_ev(action)` in dollars

`ProcBoard(game)` reads the board once per `HandAnalysis`: which of Business Card / Reserved
Parking / Faceless / Golden Ticket / Trading Card / Rough Gem / Mail-In Rebate / Delayed
Gratification / To Do List is owned, the round's Mail rank, the free consumable slots, and
`probabilities.normal = 2 ** (owned Oops! All 6s)` (computed locally — calling the engine's
`sync_probabilities` would mutate `run_state`).  A `1-in-k` proc contributes
`min(1, normal/k)` times its payout: Oops! All 6s doubles Business Card to certainty and
Lucky money to 2/15, exactly as `base.prob_roll` does.

Per card, three numbers (`_prep_procs`):

| | what fires | value |
|---|---|---|
| `proc_play[j]` | the card is a **scoring** card of a played hand | Gold seal $3 · Lucky `20/15` · Business Card `½·$2` if face · Golden Ticket $4 if Gold enh · Rough Gem $1 if Diamond — all ×2 with a Red seal |
| `proc_hold[j]` | the card **stays in hand** | Gold enhancement $3 (×2 with a Red seal) · Reserved Parking `½·$1` per hand still to be played |
| `proc_discard[j]` | the card is **discarded** | Purple seal → `tarot_value_dollars` (default 4) · Mail-In Rebate $5 on the round's rank |

Debuffed and face-down cards contribute nothing to any of the three (the engine skips
debuffed scoring cards outright, `scoring.py:257`; a face-down card's rank is never read).

`extraction_ev(action)` composes them:

* **play** — sum `proc_play` over the **scoring mask** (not the played set: a Gold-sealed
  kicker in a high-card play pays nothing), `+$4` if the hand type is To Do List's, `+
  ½·$1 ×` the faces **still held** while the hand scores (the played cards have already left
  `G.hand`), and the gold-hold delta below.
* **discard** — `proc_discard` over the discarded set with the Purple-seal count **capped at
  the free consumable slots** (`game.py:_discard` re-checks per card, exactly Lua's
  `consumeable_buffer`), `+$5` for Faceless at 3+ faces, `+$3` for a single-card Trading Card
  discard, `−$2 × discards_left` for Delayed Gratification (the first discard of the round
  forfeits the whole row), and the gold-hold delta.
* **gold-hold delta** — the end-of-round $3 is the only *conditional* term: a Gold card that
  leaves the hand forfeits it, the `m` fresh cards may bring one in
  (`3·(m·gold_rate − gold_leaving)`), and the whole thing is multiplied by
  `P(clear | after the action)` because a failed blind never reaches the payout.

**Currency.** The term enters the hand objective at the objective's own rate — `beta_hand`
= 0.012 P(clear) units per dollar, the same rate that already pays for an unused hand —
scaled by `1 + interest_bonus` (0.16) while the balance is still below the interest cap.
That 0.16 is not a new model: it is the marginal rate of `player.build_proxy`'s
`money = $ + 0.8·interest` convention (`PlayerConfig.interest_weight` 0.4 over two rounds,
`INTEREST_RATE` 5 ⇒ `2·0.4/5`).

## 3. Keep-value overhaul: discard-value vs play-value vs hold-value

The old `keep_value` added a flat `+0.5` for *any* enhancement, seal or edition.  That single
line is why "discard the purple seals" was unrepresentable: a Purple seal (worth nothing when
played, a Tarot when discarded) looked keep-worthy.  Now:

* the `+0.5` covers only **scoring-relevant** modifiers (Bonus / Mult / Wild / Glass / Steel /
  Lucky enhancements, any edition, Red or Blue seal);
* Gold seal, Gold enhancement and Purple seal get their value from the dollar terms instead;
* two orderings replace the one `junk_order`, at `junk_dollars` = 0.8 keep-units per dollar:

```
play_keep[j]    = keep_value[j] + w·( proc_play + proc_hold + proc_discard )   # burning it as
                                                       # a non-scoring play filler wastes ALL
discard_keep[j] = keep_value[j] + w·( proc_play + proc_hold − proc_discard )   # discarding it
                                                       # CASHES proc_discard, forfeits the rest
```

`play_junk_order` (ascending `play_keep`) feeds `_dump_variants` and the pure-dump play line;
`discard_junk_order` (ascending `discard_keep`) feeds `_discard_lines`' junk-out-k lines, its
>5-card trimming, and which card of a rank a straight-window keep retains.  The exchange rate
`w` only orders junk — the objective prices the money itself, so `w` never decides a decision.

Consequences, all pinned by tests: a Purple seal is the **first** card discarded and one of
the **last** burned as play filler; a Gold-enhanced card is never the first thrown away in
either ordering; a face card becomes hold-valuable exactly when Reserved Parking is owned.

**Plain-board fast path.** When no per-card proc can fire (no proc joker, no Gold/Purple seal
and no Gold/Lucky enhancement in hand) `_prep_procs` returns immediately with shared zero
arrays, `play_keep is discard_keep is keep_value`, one sort instead of two, and
`extract_on = False`.  That is the ante-1 common case and is what keeps the fast budget flat
(§0, and `test_plain_board_takes_the_zero_cost_fast_path`).

## 4. Candidate lines and the tail-DP safety gate

`extraction_safe(h, d, need)` = `BlindModel.p_clear(need/ratio, h, d) >= extract_min_clear`
(0.90), and always False at a Nemesis (`is_pvp`: there is no unused-hand money at a PvP blind
and every hand is played anyway — EV_NOTES §3) or when there is nothing to extract.

Gated on it:

* **`_extraction_discard_lines`** — dump the proc-discardable cards (Purple seals capped at
  the free slots, Mail-In ranks), each also in a junk-filled 5-card variant; dump the three
  cheapest faces for Faceless (+ a filled variant); dump one card for Trading Card.  Appended
  **after** the `max_discard_lines` cap, because they are a different *kind* of line, not a
  worse chase line (3 in `lite`, `extract_lines` = 6 otherwise).
* **the money term in `evaluate()`** — added to a candidate only when the position it leaves
  is safe.  Below the gate the player is bit-identical to the pre-change objective.

Not gated (deliberate, documented deviation): the **junk orderings** of §3 stay proc-aware at
any P(clear).  They only decide which of two structurally equivalent cards goes; refusing to
prefer the Purple seal there would cost money for no safety.

The gate makes sandbagging **self-regulating** rather than a mode switch: banking a $4 Tarot
is worth 0.056 in P(clear) units, so it beats clearing now only when the risk it adds is
smaller than that — and it is compared against the objective's existing `beta·E[hands unused]`
(clearing in one hand banks $1 per unused hand), which is exactly the trade Tagg described.

**Play-side extraction lines** are generated *ungated* as structural alternates (`_reps`): for
every n-of-a-kind, flush and straight, the representative choice that maximises PLAY value in
addition to the existing highest-chip choice.  Two same-rank Kings score identically but only
one carries the Gold seal, and the chip-only representative never picked it.  These are ~2–6
extra candidates and only exist when `has_play_proc`; the *money* they carry is still gated.
The "weakest clearing play" and "stall play with proc cards" lines the brief asks for need no
new generator — every structural set and every single card is already a candidate, and the
gated money term is what promotes them.

## 5. What is deliberately NOT modelled

* **Reserved Parking beyond the current hand.** The held-face count is exact for the hand
  being played; later hands refill toward the same deck composition whatever this action does,
  so their Parking expectation is a constant across candidates and cancels in the argmax.
  What that drops is second order: the cards this action permanently removes slightly change
  the deck for the rest of the round, and a discard branch has one more potential Parking
  firing than a play branch (≈ $0.35, already inside the objective's per-hand `beta`).
* **Joker retriggers other than the Red seal** (Sock and Buskin doubles Business Card, Dusk,
  Hack, Seltzer, Hanging Chad, Mime on the held phase).  A Red seal is a property of the card
  and is priced; joker retriggers would need the retrigger table per candidate.
* **Trading Card destroys the discarded card.**  Priced at $3; the deck-thinning (usually a
  small gain) is not priced.
* **The card a tarot should land on.** §6.
* **Money → future strength.** The layer stops at dollars; what a fixed deck or an extra Tarot
  is worth two antes later is V's job (brief §1, the ENCODE/LEARN split).
* **Matador ($8 when the boss triggers)** and every end-of-round joker row (Golden, Rocket,
  Cloud 9, To the Moon, Egg): not per-action, so they cannot change a hand decision.

## 6. Tarot targeting (first-order)

A held targeted Tarot is only worth its `tarot_value_dollars` if there is something in hand to
put it on.  `_prep_tarot_wants` builds `(want_mask, need, pile_count)` per held Tarot and
`_cycle_ev(keep_mask, m)` prices a line that draws `m` fresh cards by the hypergeometric
probability that the draw supplies the missing targets, times
`tarot_value_dollars · tarot_cycle_fraction` (0.5).  It reuses the same `_hyper_tail` the draw
targets use, and is cached per `(keep_mask, m)`.

Modelled: the four **suit** tarots (Star / Moon / Sun / World convert up to 3 cards, so a
flush wants `flush_need − 3` real cards of that suit already in hand — restrictive exactly
when the hand is thin in that suit) and the eight **enhancement** tarots (want a plain card;
overwriting an existing enhancement wastes it).

Not modelled: Strength / Hanged Man / Death target selection; *which* card the tarot lands on
(the player's existing `_consumable_candidates` still draws targets from the best play's
scoring cards); the interaction between cycling for a tarot and cycling for a draw target
(they are added, not jointly optimised); Tarots that would arrive *later* in the blind.

## 7. Configuration (`HandConfig`, all new fields)

| field | default | meaning |
|---|---|---|
| `extract` | `True` | price the per-action money procs at all (`False` = the pre-change player, bit-identical) |
| `tarot_value_dollars` | `4.0` | what one created Tarot is worth, until V takes over |
| `extract_min_clear` | `0.90` | the tail-DP safety gate |
| `extract_lines` | `6` | extra extraction candidate lines per decision (3 in `lite`) |
| `junk_dollars` | `0.8` | keep-units per dollar, for the junk ORDERINGS only |
| `tarot_cycle_fraction` | `0.5` | fraction of a Tarot's value a good target unlocks |
| `interest_bonus` | `0.16` | marginal worth of a $ that still earns interest (§2) |

## 8. Measurements — how to reproduce

```
python ev/gate_ev_player.py --procs 8                                  # gate (a)
python ev/scripts/extract_dev_slice.py scan  --seeds 126 --procs 8     # pick the slice
python ev/scripts/extract_dev_slice.py slice --procs 8 --seeds <slice> # gate (b)
python ev/scripts/extract_dev_slice.py slice --procs 8 --to-ante 6 --seeds <slice>
python ev/scripts/extract_dev_slice.py h2h   --n-seeds 30 --procs 8 --max-steps 4000
python -m pytest ev                                                    # gate (c)
python -m oracle.engine_parity --antes 1-8 --rerolls 5 --quiet
```

**The dev slice (12 seeds).**  Chosen by `scan`, which plays every one of the 126 ground-truth
seeds to the start of ante 3 with the OLD player and scores what the board could extract from
(`3·proc jokers + 2·purple seals + 2·gold seals + gold enh + lucky`).  The slice — the highest
scorers, spread over the proc families:

```
GVYT2DGJ (To Do List + purple seal)   V3PUR5L4 (gold seal + purple seal)
29DAQVG1 (purple seal)                3SZ71111 (purple seal)
7UNQV1C9 (Business Card)              USQF4ZAV (Business Card)
8SIYIK9C (Reserved Parking)           A        (Faceless)
9ZXMM1M  (Mail-In Rebate)             SLRSKCG1 (Rough Gem)
SQVZX29L (Rough Gem)                  967889YL (Delayed Gratification — the negative control:
                                                the layer must discard LESS here)
```

Vanilla single-player runs have no lives, so "blinds lost" is `died` (the run ended before the
stop ante) plus `blinds_cleared`, not a life counter — the first version of this metric was
vacuously 0 in both arms and is fixed.

| metric (12 seeds, to ante 3) | extract ON | extract OFF | Δ |
|---|---|---|---|
| $ at the start of ante 3 | **20.333** | 18.750 | **+1.583** |
| tarots used | **1.333** | 1.083 | **+0.250** |
| planets used | 3.333 | 3.250 | +0.083 |
| blinds cleared | 7.000 | 7.000 | 0.000 |
| runs died | 0.167 | 0.167 | 0.000 |
| hand ms mean / p95 | 4.573 / 10.226 | 4.341 / 9.769 | +0.232 / +0.456 |

| same slice, to ante 6 | ON | OFF | Δ |
|---|---|---|---|
| $ at the stop | **36.083** | 23.583 | **+12.500** |
| final $ | **33.917** | 23.083 | **+10.833** |
| tarots / planets used | 3.583 / 6.167 | 2.667 / 4.583 | +0.917 / +1.583 |
| blinds cleared | **9.833** | 8.833 | **+1.000** |
| runs died | 0.917 | 1.000 | −0.083 |
| mean final ante | **4.917** | 4.333 | +0.583 |
| hand ms mean | 4.932 | 4.916 | +0.016 |

Gate (b) as specified — money and tarots strictly up, blinds lost not up — **passes**, and the
effect compounds sharply with depth (the shop turns dollars into build, which is the whole
reason the layer exists).

**h2h (gate d).**  Same player both sides, only `HandConfig.extract` differs; both seatings per
seed so the seat-order bias cancels; `--max-steps 4000`, 8 procs.

| seeds | matches | new-fast wins | CI | lives margin | mean $ A / B |
|---|---|---|---|---|---|
| 30 | 60 (0 undecided) | **50.0%** | [36.7, 61.7] | +0.000 | 33.5 / 31.8 |
| 60 | 120 | 48.3% | [40.0, 57.5] | −0.067 | 32.7 / 30.3 |
| 126 | 252 | **50.4%** | [44.4, 56.7] | +0.016 | 32.0 / 30.5 |

Read honestly: this is a **wash**, and a wash is the expected outcome of a near-mirror design
— on the 30-seed gate, 24 of 30 seeds split one win per seating and only 6 diverged.  The
requirement ("extraction should not lose matches") is met at every sample size; the 126-seed
run is the number to quote because the 30-seed CI is 25 points wide.  Money is consistently
+$1.5–2 for the extraction side even where the match outcome does not move: at antes 5–6 the
extra dollars have not yet been converted into enough build to flip a Nemesis.

## 9. Open issues / interface requests

1. **`h2h.py` cannot express "the fast player with a different `HandConfig`."**  `build_player`
   parses `ev:fast+full+stats` tokens only, so gate (d) is run by
   `ev/scripts/extract_dev_slice.py h2h` instead (same design as `h2h.py`: both seatings,
   spawn pool, `common.bootstrap_ci`).  A one-line addition — an `ev:fast+noextract` token, or
   a generic `hand_cfg` override — would let the lead run it through the standard driver and
   write the standard JSON/MD.  Not made here: `h2h.py` is not this workstream's file.
2. **`gate_ev_player.py` has no config knob either**, so the "before" column of §0 is a gate
   run taken *before* the change rather than an A/B in one process; hence the greedy-column
   load normalisation.  A `--hand-cfg extract=0` style flag would make the timing claim exact.
3. **The gate's timing columns are not load-controlled.**  The scripted-greedy row is a usable
   probe (it is unaffected by every EV-player change) — worth reporting as a normaliser in the
   gate's own summary.
4. **`tarot_value_dollars = 4` is a placeholder.**  It is the single most influential constant
   in the layer (it sets how hard the player chases Purple seals).  Brief §1 says V takes this
   over; until then a sweep on the dev slice would be cheap and is not done here.
5. **Reserved Parking's future-hand term** and joker retriggers (§5) are the two first-order
   omissions most likely to matter on a proc-heavy build.
6. **`extraction_lines` is the generator W-PAIRS feature-detects** — `(game, legal=None) ->
   [(action, ev, reason)]`, safety-gated, best first, `[]` when there is nothing to extract.
   `hand.extraction_ev(game, action)` is also exported but rebuilds a whole `HandAnalysis` per
   call; the per-decision path is `HandAnalysis.extraction_ev(action)` on a shared analysis
   (W-PAIRS's own test pins that).
7. **Not verified by me:** whether the gold-enhancement interest-ordering fix changes any
   existing stats-tier / shop-rule calibration.  It makes Gold cards slightly better than the
   engine previously modelled them.
