# CYCLE_NOTES — W-CYCLE: per-target tarot digging in the hand layer (Phase 5 rev 2, 2026-08-27)

Files: `ev/hand.py` (the cycle / dig paths only, +333/−64), `ev/player.py` (one new method,
`_held_tarot_values`, one new cache dict, and three lines in `_rank_hand`; +38/−1),
`ev/fixtures/tarot_target_cycle.py` (the acceptance fixture, rebuilt from the weaker claim),
`ev/tests/test_probe_fixtures.py` (its pins), `ev/tests/test_cycle.py` (new, 12 tests), this
file, plus a dated "superseded" pointer in `EXTRACT_NOTES.md` §6 and a dated "acted on"
pointer in `PROBE_NOTES.md` §3.3 (additive, following W-PVP's precedent in EXTRACT_NOTES §4
— nothing in either document was rewritten).  `ev/pairs.py`, `ev/labels.py`, `ev/scripts/`,
`ev/encode/` and the whole engine are untouched — **no engine change was needed and none was
made**, so `engine_parity` is not in play.  Nothing is committed.

Read `EXTRACT_NOTES.md` §6 first (the per-COUNT `_cycle_ev` this replaces) and
`PROBE_NOTES.md` §3.3 (the limitation it reports).  This document describes what changed.

The owner's design note, which is the whole brief: *"late-game, using hands as THROWAWAYS to
dig deeper into the deck so held tarots land on specific target cards is often worth far more
than the \$1/hand payout."*

## 0. Headline

The clean before/after is a **paired 126-seed A/B in one process**, `tarot_per_target` OFF vs
ON — and the OFF arm is verified bit-identical to the pre-change player by replaying the
five moved seeds in a `git worktree` at `47dae5e` (§6).  The gate run below is the ON arm of
exactly that pair (0/126 rows differ from it).

| paired 126-seed solo A/B (`cycle_h2h.py solo`) | OFF (= pre-change) | ON | Δ |
|---|---|---|---|
| ante-1 / 2 / 3 / 4 clear | 94.4 / 81.7 / 74.6 / 58.7% | **94.4 / 81.7 / 74.6 / 58.7%** | **0 / 0 / 0 / 0** |
| mean final ante | 4.627 | **4.627** | **+0.000** |
| mean blinds cleared | 9.508 | 9.492 | **−0.016** |
| **tarots used** | 3.976 | 3.984 | **+0.008** |
| planets used | 4.421 | 4.429 | +0.008 |
| final \$ | 26.325 | **26.643** | **+0.317** |
| runs died | 0.976 | 0.976 | 0.000 |
| seeds whose trajectory differs **at all** | — | — | **3 / 126** |

| gate / suite | |
|---|---|
| 126-seed `gate_ev_player.py --procs 12` (`results/ev_player_gate_2026-08-27.{md,json}`) — **fast** ante-1 clear | **94.4%** [90.5, 98.4] — **the bar, unchanged** |
| … fast ante-2 / 3 / 4 clear | 81.7 / 74.6 / 58.7% |
| … fast mean final ante / blinds cleared / \$ at ante 3 / won | 4.64 / 9.49 / 20.76 / 2.4% |
| … **full** ante-1 / 2 / 3 / 4 clear | 96.0 / 86.5 / 78.6 / 69.0% (**up** at 2/3/4 vs the 08-26 file's 85.7 / 77.0 / 67.5) |
| … full mean final ante | 4.92 |
| … fast hand ms mean / p95 (greedy load probe) | **4.13 / 12.36** (2.16 / 2.75) — under the 5 ms budget |
| … draw-order invariance | fast 737/737, full 742/742 |
| paired h2h ON vs OFF, 30 seeds × 2 seatings | **53.3%** [40.0, 66.7], lives margin +0.117 |
| … same at **126 seeds × 2 seatings (252 matches, 0 undecided)** | **50.8%** [44.8, 57.1], lives margin +0.028, \$ 34.1 / 33.8, tarots 5.76 / 5.72 |
| `python -m pytest ev -q` | **394 passed** (374 baseline + 8 fixture pins + 12 new `test_cycle.py`) |
| engine / parity | **untouched — no engine change was needed or made** |

| the acceptance fixture (`fixture:tarot_target_cycle`, fast budget) | sandbag | control |
|---|---|---|
| **dig** `discard [6,7]` | **1.051253 (#1)** | 1.042000 |
| **clear now** `play [0,1,2,3,4]` | 1.044001 | **1.044001 (#1)** |
| Δ (dig − clear) | **+0.007252** | **−0.002001** |

Was: 1.05700 vs 1.04400 on the *same* action (PROBE_NOTES §3.3's weaker claim).  Now: a
genuine rank swap between two **different** actions, reversed in the control.

**Read honestly.**  Every ante-N clear rate is *identical* between the paired arms — the
change is inert on 123 of 126 seeds and moves three (`EGE5ZY77`, `I68YASXJ` down one blind
each, `QY3TQZJ9` up one and +\$20).  The direction of travel is money up, blinds cleared a
hair down: the sandbag signature, at a much smaller amplitude than W-EXTRACT's.  The h2h is
a wash leaning positive at both sample sizes.  §6 has the arithmetic.

## 1. What "per-target" means, and what was wrong before

`EXTRACT_NOTES.md` §6's `_cycle_ev(keep_mask, m)` priced a held targeted tarot as

```
Σ_tarots  P(the m fresh cards supply the missing targets) · tarot_value_dollars · 0.5
```

with "missing" = `need − (count of wanted cards the line keeps)`.  Two things follow, and
`PROBE_NOTES.md` §3.3 found both by trying and failing to build a rank swap against it:

* the value depends on the **count** of targets, never on **which** ones — a lone Ace of the
  wanted suit and a lone Four of it are the same card to that formula;
* every line that draws the same `m` gets the same number, so in a hand where the whole
  clearing tier draws 5 the bonus attaches uniformly and cannot flip a rank.

### The model now

Per held targeted tarot, `_prep_tarot_wants` builds a `_TarotWant`: how many targets the
effect needs (`k`), the qualifying **hand** cards bucketed by grade, the qualifying **draw
pile** cards bucketed by the same grades (one pass over the real pile, only for the families
actually held), and the tarot's dollar value (§2).  A target's **grade** is the rank family
the engine's own chip table already separates —

| tier | ranks | grade | why |
|---|---|---|---|
| 0 | Ace | 1.00 | 11 chips, the most any card carries |
| 1 | 10 / J / Q / K | 0.85 | the ten-chip band |
| 2 | 7 / 8 / 9 | 0.70 | mid |
| 3 | 2 – 6 | 0.55 | low |

— because a Steel card is worth more on an Ace than on a 2, and a Sun flush is worth the
chips of the cards that complete it.  Four tiers is a deliberate cap: it is what keeps the
expectation below at four `_hyper_tail` calls per tarot per `(keep_mask, m)`.

`_target_quality(want, keep, m)` is then the **layer-cake expectation of the k-th best
target grade** available after the line:

```
E[X_(k)] = Σ_l ( g_l − g_(l+1) ) · P( #targets of grade ≥ g_l  ≥  k )
#targets  = (the kept ones, known exactly)  +  Hypergeometric(N, pile_l, m)
```

`P(·)` is the same `_hyper_tail` the draw targets use, over the same real pile composition
(`EV_NOTES.md` §1), so this is **exact given the grades** — no new probability model, and
`ev/tests/test_cycle.py::test_target_quality_is_the_layer_cake_expectation_of_the_kth_best_grade`
writes the four numbers out by hand.

Reading the "k-th best" choice honestly: the tarot's effect is a *threshold* — the Sun's
flush needs all `flush_need − 3` real cards of the suit, so a set is only as good as its
weakest member, and `X_(k)` is 0 exactly when fewer than `k` targets exist.  That is what
keeps the old all-or-nothing behaviour (a one-card draw can never supply two Hearts) while
adding the quality gradient on top.

`_cycle_ev(keep, m, p_cont)` is then, per tarot:

```
0                                              if the KEPT cards already supply k targets
value · tarot_cycle_fraction · E[X_(k)]        otherwise
```
all multiplied by `p_cont` (§3).  The first clause is the conservative half: when the tarot
can already land, digging would only swap one target for a marginally better one, and that
second-order gain is deliberately not priced.  It means a board that is *already* good for
the tarot carries no cycle term at all — the same constant for every candidate, which cancels
in the argmax — so this layer only ever speaks when the tarot is actually short of targets.

What is modelled: the four **suit** tarots (`k = flush_need − 3`, targets = held/pile cards of
that suit, Wilds counted) and the eight **enhancement** tarots (`k = 1`, targets = plain,
un-debuffed, non-Stone cards).  Not modelled, unchanged from EXTRACT_NOTES §6: Strength /
The Hanged Man / Death target selection (their effect is a rank or deck change the hand
layer has no per-card grade for), and a tarot that would *arrive* later in the blind.

## 2. The valuation: W-SHOP's measured dollars, not the flat \$4

The brief offered a choice between W-SHOP's measured deck-effect valuation and
`EXTRACT_NOTES.md` §7's flat `tarot_value_dollars = 4`.  **The measured one is taken**, and it
is taken by *reusing* W-SHOP's own code rather than re-deriving it: `EVPlayer.
_held_tarot_values` calls `_deck_effects(game, _TAROT_EFFECTS)` — the same memoised
build-proxy probes `SHOP_NOTES.md` §3.2 documents — and feeds them to the same
`player.tarot_dollars` table the shelf is priced with, then hands the result down as
`rank_hand_actions(..., tarot_values={key: dollars})`.

Why measured: it is the only version of this layer in which the brief's crossover is
arithmetic.  A flat \$4 is a constant, so "the dig is worth more than the \$1/hand payout"
would be true or false by fiat for the whole run.  The measured number is a property of the
deck in front of the player — SHOP_NOTES §3.2's own reading, The Star \$15.14 versus The
Chariot \$3.14 on the same board — so a suit tarot held over a scattered deck digs hard and
the same tarot over an already-converted deck digs barely at all, with no mode switch and no
ante threshold anywhere in the code.

**The fallback is explicit and documented**: every caller that is not `EVPlayer` — the
rollout policy inside `budget="full"`, `play_out_blind`, the blind model, the acceptance
fixtures, a bare `hand.rank_hand_actions` — gets `cfg.tarot_value_dollars`.  Two reasons, both
deliberate:

1. `_deck_effects` is an `EVPlayer` method whose memo is *per player instance*, on purpose
   (`SHOP_NOTES.md`; the same reason `board_ratio` takes a cache argument).  A module-level
   singleton would share a coarse cache key — `(suit histogram, deck size, #enhanced, ante,
   planet levels, hand size)` — across runs whose jokers and money differ, and the value
   would then depend on which run warmed the cache.  That is a reproducibility hazard, not a
   speed one, and it is not worth taking for a rollout leaf.
2. It keeps the acceptance fixture honest: the rank swap of §4 is produced on the **flat \$4
   fallback**, so the fixture demonstrates the per-target *mechanism*, not a bigger constant.
   The measured value only widens the margin (the advisor path, which goes through
   `EVPlayer`, prices the same fixture's Sun well above \$4).

`test_cycle.py::test_the_dig_scales_linearly_with_the_measured_tarot_value` pins the
linearity, so the two paths differ by a scalar and nothing else.

## 3. Throwaway-dig lines, and why a clear cannot bank a dig

The economics the brief asks for come from one new fact, `_play_continues(s)`:

> A play that reaches the chip target — or that spends the last hand — **ends the round**,
> and `game._end_round` puts the whole hand straight back into the deck.  The fresh cards
> such a play draws are never in hand at another decision, so they can never carry a tarot,
> and the dig is worth exactly **zero** on it.

This is the same conditioning `_gold_delta` already applies in the other direction (the Gold
enhancement's \$3 needs the round to *end*), and it is the entire reason a throwaway can
outrank an immediate clear: the clear cannot bank the dig, only the line that keeps the blind
alive can.  It is a hard 0/1 on the candidate's own exact score, which is a first-order read
— a play a hair under the target could be pushed over by a Lucky or a Glass roll.  A discard
never ends a round, so its `p_cont` is 1; at a Nemesis there is no chip target, so only
running out of hands ends it.

The candidate lines:

* **`_dig_lines()`** (discard side) — throw the cards no held tarot wants, keep the ones it
  does, draw as deep as the discard allows.  Three shapes: dig as deep as possible; the
  variant that *also* preserves the best made play, so the dig costs the discard and nothing
  else; and one shallower step.  Capped at `cfg.dig_lines` (4; 2 in `lite`).
  These exist because the ordinary junk-out-k lines are blind to tarot targets: `discard_keep`
  never sees them, so the very card the tarot is waiting for — a lone Ace of the wanted suit,
  which pairs with nothing and sits in no straight window — sits near the top of the junk
  ordering and gets thrown as soon as `k` reaches it.

  **Measured, because "cannot express" would be an overstatement**: over 48 real states with
  a targeted tarot held (the 60 first gate seeds to ante 6), `_dig_lines` emitted 45 lines, of
  which **27 (60%), in 19 of the 48 states (40%), are card sets the ordinary generator does
  not produce at all**.  The top-ranked action was a dig line in 3 of the 48 states, and in 1
  of those it was a dig-**only** line — unreachable without this generator.  So the generator
  is load-bearing but second-order next to the valuation; the acceptance fixture is
  deliberately built so the swap does *not* depend on it (§4).
* **the throwaway dig play** — the same dump the play side already generates
  (`junk_order[:5]`, `junk_order[0]`), but never spending the cards a tarot is waiting for.

**Gating.**  `_dig_lines` is gated exactly like `_extraction_discard_lines`
(`EXTRACT_NOTES.md` §4): the tail DP must still clear the blind with probability
`extract_min_clear` (0.90) after the discard is spent, never while a Nemesis race is live,
and the decided-race rules of `PVP_NOTES.md` §5 apply at a Nemesis.  The play-side variants
are generated *ungated*, following the precedent EXTRACT_NOTES §4 set for `_reps`' play-side
extraction alternates — they are structural alternates, and the money they carry is still
gated by the same `pc >= extract_min_clear` test in `evaluate()`.

**The crossover is arithmetic.**  Nothing in this layer knows what ante it is.  A dig wins
when `rate · value · 0.5 · E[X_(k)]` exceeds what the line gives up, and the two sides move
independently: the cost of a throwaway is the flat `beta_hand`/`gamma_discard` bookkeeping
plus whatever P(clear) the tail DP says it costs (which shrinks toward zero as the build gets
strong enough to clear from anywhere), while the improvement is the measured deck value of
the tarot (which grows as the deck gets worth fixing).  That is the crossover the design note
describes, and it falls out of the arithmetic rather than being switched on.

## 4. The acceptance fixture — the genuine rank swap

`ev/fixtures/tarot_target_cycle.py` was rebuilt.  The board: the Sun held, **one** real Heart
(the Ace, index 5) when the Sun's flush wants two, a Clubs flush 3C 5C 7C 9C JC (indices 0-4)
that clears the blind outright, two junk Spades (2S, 8S), `chips_target` 250.  No straight
window is filled and no rank repeats, so the Clubs flush is the only made hand and "clear
now" is unambiguous.

| line | sandbag EV | control EV |
|---|---|---|
| **dig** `discard [6,7]` (keep the flush AND the Ace of Hearts, draw 2) | **1.051253** | 1.042000 |
| **clear now** `play [0,1,2,3,4]` (276 chips ≥ 250) | 1.044001 | **1.044001** |
| Δ (dig − clear) | **+0.007252** | **−0.002001** |

The sandbag's #1 ranked action is the dig; the control's is the clear.  **The swap and its
reversal are both real**, and both arms of the fixture are the same 8 cards and the same
blind — the only difference is whether `c_sun` is in `consumable_hand`.

`discard [6,7]` is emitted both by `_dig_lines` and by the ordinary junk-out-2 line, on
purpose: this fixture isolates the **valuation**, so the swap does not depend on the new
candidate source existing.  (The next junk-out-k line, `[5,6,7]`, is where the ordinary
ordering starts throwing the Ace of Hearts — the junk order here is `2S, 8S, AH, …`.)

Where the margin comes from, decomposed at the flat \$4:

* the dig's extraction term is **\$0.6647** (= `E[X_(2)]` 0.33235 × \$4 × 0.5), worth
  0.009253 in P(clear) units at this state's dollar rate (0.012 × 1.16);
* the clear's extraction term is **\$0.00** — it ends the round;
* the dig's position value is 1.042000, the clear's 1.044001: the dig gives up 0.002001 of
  bookkeeping (it spends a discard instead of banking a hand) and P(clear) is untouched
  because the whole Clubs flush is kept and still clears next hand.

Two further pins in `test_probe_fixtures.py`, because the swap alone does not separate the
new model from the old one:

* **count** — `discard [6,7]` (keeps the Ace of Hearts) is worth \$0.6647 against
  `discard [5,6]`'s \$0.0853, same size, same floor, same draw.  ~8×.  *This half is not
  new*: the per-count form already subtracted the wanted cards a line keeps.  It is reported
  because it carries most of the fixture's margin, and pretending otherwise would misattribute
  the result.
* **which** — the genuinely new half, isolated by `build_low_grade_target()`: the identical
  board with the held Heart downgraded from the Ace to the Four.  Same suit, same count, same
  `m`, same pile depth; the dig is strictly smaller with the Four, and the per-COUNT form
  scores the two boards **identically** (`pytest.approx`).  That is exactly the "depends only
  on m drawn, not WHICH cards" limitation PROBE_NOTES §3.3 reported.

`test_tarot_per_target_off_restores_the_old_ordering` keeps the old claim alive as a
regression pin: with `tarot_per_target=False` the dig is not generated, the clearing play
banks the cycle bonus again, and the fixture reverts to PROBE_NOTES §3.3's "same action,
more EV" reading.

## 5. Cost

`_prep_tarot_wants` pays one pass over the draw pile *only* when a targeted tarot is held;
`_target_quality` is one `_hyper_tail` call per grade tier — four in total, globally
LRU-cached — and the whole per-tarot sum is memoised per `(keep_mask, m)` in the same
`_cycle_cache` the old form used.  The per-decision cost of the model itself is therefore in
the noise; the only cost worth naming is `EVPlayer._held_tarot_values`, i.e. the four
`_TAROT_EFFECTS` probes (five `build_proxy` calls counting the base), memoised per
`(deck shape, ante, planet levels, hand size)` on the player instance.

**The A/B, interleaved on identical real states** (`cycle_ab.py`: 96 states where a targeted
tarot is actually held and 300 where none is, collected from the 126 gate seeds to ante 6;
every state timed by every arm before the next state, two warm passes discarded, so the
module-level model / ratio caches are equally warm for both arms):

| path | tarot HELD, ON | tarot HELD, OFF | Δ | none held, ON | none held, OFF | Δ |
|---|---|---|---|---|---|---|
| `HandAnalysis(...).evaluate()` — the decision core | 1.861 ms | 1.776 ms | **+0.085** | 1.919 | 1.895 | +0.024 |
| `rank_hand_actions` — adds the `use_consumable` evaluation | 20.373 | 20.208 | +0.165 | 3.642 | 3.596 | +0.046 |
| `EVPlayer._rank_hand` — adds `_held_tarot_values` | 9.286 | 9.287 | **−0.001** | 2.139 | 2.137 | +0.002 |

and at the run level the 126-seed gate reads **4.13 ms mean / 12.36 p95** against the
08-26 file's 4.12 / 12.05, with the unchanged scripted-greedy load probe at 2.16 vs 2.21 —
i.e. **flat, and inside the 5 ms fast budget**.

**The brief's cost gate — "mean hand decision ≤ 5 ms with a held targeted tarot" — is NOT
met, and it was not met before this change either.**  A hand decision with a targeted tarot
in hand costs **9.3 ms** through the player path (20.4 ms through a cold module-level
`rank_hand_actions`) in **both** arms.  The cause is measured and is not this layer:

```
use_consumable candidates per such decision   mean 4.42, max 6
the _consumable_ev block                      mean 17.5 ms   (~4 ms per candidate)
_MODEL_CACHE occupancy during the measurement 64 / 64        (at its cap, so it thrashes)
```

`_consumable_ev` clones the game, steps the use, and builds a **whole new `HandAnalysis`** —
and because using a tarot changes the deck, that means a fresh `BlindModel` (~6 ms,
`EV_NOTES.md` §1) whose key is evicted from the 64-entry LRU almost immediately.
EXTRACT_NOTES §0 already identified this as the reason its own gate timing moved ("the
extraction player owns more tarots, and every held consumable costs a `use_consumable`
evaluation per hand decision"); this workstream measures it and hands it on as §9.1.  What
W-CYCLE itself adds is **+0.085 ms** to a tarot-held decision and nothing measurable to the
player path — `_deck_effects` is memoised per deck shape, so the measured-value plumbing is
free after the first decision of each deck shape.

## 6. Honest reading of the gate

**The "before" column has to be built, not read off the shelf.**  Diffing the new gate
against the committed `results/ev_player_gate_shop_2026-08-26.json` shows 5 of 126 seeds
moved — but two of them (`LMGJPMKP`, `RXI42HZ2`) also differ from **the pre-change player
run today**.  Checked out at `47dae5e` in a throwaway `git worktree` and replayed, the
baseline commit produces `LMGJPMKP` ante 3 / 5 blinds / \$15 and `RXI42HZ2` ante 6 / 9
blinds / \$52 — which is exactly what `tarot_per_target=False` produces in the current tree,
and *not* what that results file records (ante 8 / 19 blinds, and ante 6 / 11 blinds).

Two conclusions, both worth having:

1. **`tarot_per_target=False` is bit-identical to the pre-change player** — verified against
   the baseline commit itself on all five moved seeds, not just argued from the code.  The
   flag is therefore a sound A/B arm and a sound rollback.
2. **`results/ev_player_gate_shop_2026-08-26.json` is 2 rows stale** relative to the commit
   it ships with (it was presumably generated mid-workstream).  Anyone diffing a gate against
   it will see a phantom −0.8 pp at ante-3 and ante-4 that belongs to those two rows.  Flagged
   for the lead in §9; **not** corrected here, because that file is W-SHOP's record.

Against the properly paired baseline the gate reads: **every ante-N clear rate identical**,
mean final ante identical to three decimals, three seeds' trajectories moved (two lose a
blind, one gains a blind and \$20), mean blinds cleared −0.016, mean final \$ +0.317, tarots
used +0.008, runs died unchanged.  The h2h agrees: 50.8% [44.8, 57.1] over 252 matches.

**So: this is a wash on outcomes and a small positive on money, with the mechanism it was
built for demonstrably firing.**  That is the honest headline.  Three reasons the effect is
small, all of them real and none of them fixable inside this workstream's scope:

* **The player rarely holds a targeted tarot at a hand decision.**  Over the 126 gate seeds
  to ante 6, 96 of ~2,500 sampled SELECTING_HAND states had one — under 4%.  A layer that
  can only speak at 4% of decisions cannot move a 126-seed gate much.
* **When it does hold one, the targets are usually already there.**  `k = 1` for every
  enhancement tarot and `k = flush_need − 3 = 2` for the suit tarots, and the dig is
  suppressed outright when the kept cards already supply `k` (§1).  Most hands keep a plain
  card.
* **The dig competes against a well-tuned objective, not against nothing.**  Its whole
  budget is `rate · value · 0.5 · E[X_(k)]`; at the flat \$4 fallback the ceiling is 0.028
  P(clear) units, and the measured value only lifts it where the deck is genuinely worth
  fixing.  It wins narrow decisions, which is what it should do.

The one number that is unambiguously up is **money (+\$0.32 mean final, +\$0.3 in the h2h)**
with **blinds cleared essentially flat**, which is the same shape EXTRACT_NOTES §0 reported
for the extraction layer and compounds through the shop rather than in the gate.

## 7. Configuration (`HandConfig`, new fields)

| field | default | meaning |
|---|---|---|
| `tarot_per_target` | `True` | the model of §1–§3.  `False` = EXTRACT_NOTES §6's per-COUNT form bit-for-bit: flat `tarot_value_dollars` (measured values ignored), no `_play_continues` conditioning, no dig lines.  This is the gate / h2h A/B arm. |
| `dig_lines` | `4` | throwaway-dig discard lines per decision (2 in `lite`) |

Unchanged and still load-bearing: `tarot_cycle_fraction` (0.5 — the fraction of a tarot's
value that *accelerating its landing* is worth, since the tarot does not expire and a later
fresh hand would supply targets too), `tarot_value_dollars` (4.0 — now only the fallback),
`extract_min_clear` (0.90 — the dig's safety gate), `extract` (False disables this layer
along with the rest of the extraction layer).

## 8. What is deliberately NOT modelled

* **The tarot does not expire.**  The dig is priced as the gain in what the tarot can land on
  *now*; that a future fresh hand of the same deck would also supply targets is not
  subtracted.  `tarot_cycle_fraction = 0.5` is the blunt instrument standing in for it, and it
  is the single most influential constant left in this layer.  A sharper version would price
  the dig against the tarot's expected landing quality over the *rest of the run*, which is
  V's job (brief §1, the ENCODE/LEARN split).
* **Which card the tarot then actually lands on.**  `_consumable_candidates` still draws
  targets from the best play's scoring cards (EXTRACT_NOTES §5).  This layer grades *what a
  line makes available*; it does not choose.  Closing that loop is the natural next step and
  is the one place the two halves could disagree — the dig can now go looking for an Ace the
  target chooser would not pick.
* **Strength / The Hanged Man / Death.**  No per-card grade exists for "+1 rank" or "destroy"
  in the hand layer; they stay out, exactly as in EXTRACT_NOTES §6.
* **Multiple tarots interacting.**  Two held tarots are summed, not jointly optimised, and
  they may want the same card.  Same treatment cycling and draw targets already get
  (EXTRACT_NOTES §6, "added, not jointly optimised").
* **The full-budget rollout leaf** uses the flat fallback (§2), so a `budget="full"` decision
  scores its head with measured values and its rollouts with \$4.  Documented asymmetry, not a
  bug; the rollouts are `lite` for the same class of reason.
* **`p_cont` is a hard threshold** on the candidate's exact score (§3).

## 9. Open issues / interface requests

1. **A hand decision with a held tarot costs 9–20 ms, and it is `_consumable_ev`, not this
   layer** (§5).  4.42 `use_consumable` candidates per such decision × a full `HandAnalysis`
   each × a `BlindModel` rebuild whenever the use changes the deck, against a `_MODEL_CACHE`
   sitting at its 64-entry cap.  Three cheap levers, none of them mine to pull: cap the
   targeted-use candidates harder than `_consumable_candidates`' current 3-per-consumable;
   reuse the root `BlindModel` for a use that does not change the deck composition (Planets
   already do not, and the code already has `_blind_key` to test it); or raise
   `_MODEL_CACHE_MAX`.  **This is the single biggest remaining hand-budget item.**
2. **`results/ev_player_gate_shop_2026-08-26.json` is 2 rows stale** relative to `47dae5e`
   (§6): `LMGJPMKP` and `RXI42HZ2`.  A lead decision whether to regenerate it; until then
   every gate diff against it carries a phantom −0.8 pp at antes 3–4.
3. **`gate_ev_player.py` still has no config knob** (W-EXTRACT open issue 2, W-SHOP open
   issue 4, unchanged).  It is why the paired before/after in §0 had to be produced by a
   private driver instead of one command.  A `--hand-cfg tarot_per_target=0` style flag
   would collapse §6's whole argument into a two-line A/B — and it is now the third
   workstream in a row to ask for it.
4. **`EVPlayer._deck_effects` writes into a memo whose key does not include the jokers or the
   blind.**  Calling it from a mid-blind state — which W-CYCLE is the first thing to do —
   would therefore hand the shop a valuation measured off a mid-blind board.  Worked around
   here by swapping in a second dict (`_hand_fix_cache`) for the duration of the call, which
   leaves `_deck_effects` itself untouched; it moved one seed's planet use, so the leak was
   real.  The tidier fix is a `cache=` parameter on `_deck_effects`, which is W-SHOP's call.
5. **`tarot_cycle_fraction = 0.5` is now the most influential constant in this layer** (§8):
   it is standing in for "the tarot does not expire, so digging only *accelerates* it".  A
   sweep on a tarot-heavy seed slice would be cheap and is not done here.
6. **The dig and the target chooser can disagree.**  `_dig_lines` goes looking for a specific
   card; `_consumable_candidates` then picks targets from the best play's scoring cards and
   may not use it.  Closing that loop — letting the tarot land on the card the dig was for —
   is the natural next piece of work and is where the rest of this effect probably lives.
7. **Not verified by me:** whether the extra dig candidates change any W-PAIRS /
   W-RANK pair-source statistics.  `extraction_lines` now returns dig lines too (they carry a
   non-zero gated money term), so `greedy_vs_extract` will see a new *kind* of line.  The
   interface shape is unchanged (`[(action, ev, reason)]`, gated, best first).

## 10. Measurements — how to reproduce

```
python ev/gate_ev_player.py --procs 12                       # gate (a)
python -m pytest ev -q                                       # gate (c) -- 394
python -m pytest ev/tests/test_cycle.py ev/tests/test_probe_fixtures.py -q   # 69
python ev/cli.py advise fixture:tarot_target_cycle         --player 0
python ev/cli.py advise fixture:tarot_target_cycle_control --player 0
```

The advisor renders the dig at #1 on the sandbag with its dollar decomposition —
`1. discard 2S 8S ... [extract $+2.16]` — and nothing at all on the control.  The \$2.16
against the fixture's own \$0.66 is §2's two paths in one line: the advisor goes through
`EVPlayer`, so it prices this deck's Sun at the MEASURED \$13.00 instead of the flat \$4.

The paired A/B, h2h and timing drivers are `cycle_h2h.py` / `cycle_ab.py`, written for this
workstream and kept **outside the tree** (`ev/scripts/` is not this workstream's to touch —
they are in the session scratchpad and are reproducible from §0's numbers alone).  Same
design as `ev/scripts/extract_dev_slice.py h2h`: same player both sides, only
`HandConfig.tarot_per_target` differs, both seatings per seed, ε = 0, spawn pool,
`common.bootstrap_ci`.

```
python cycle_h2h.py solo --seeds 126 --to-ante 8 --procs 12   # the paired before/after of §0
python cycle_h2h.py h2h  --n-seeds 126 --procs 8 --max-steps 4000
python cycle_ab.py                                            # the interleaved timing A/B of §5
```

The baseline-identity check of §6 (worth repeating for any future flag of this kind):

```
git worktree add "$TEMP/base" 47dae5e
# replay the moved seeds there with a default EVPlayer, compare to tarot_per_target=False
git worktree remove "$TEMP/base" --force
```
