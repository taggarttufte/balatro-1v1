# PROBE_NOTES -- W-PROBE: acceptance fixtures for the extraction/sandbag layer
(Phase 5 rev 2, 2026-08-25)

Owns: `ev/fixtures/{purple_seal_discard,faceless_discard,business_card_board,
reserved_parking_hold,gold_seal_weak_play,tarot_target_cycle,_probe_common}.py`, the additive
registration in `ev/fixtures/__init__.py` (12 new entries, `bloodstone_vs_invisible` and
its own registration untouched), `ev/tests/test_probe_fixtures.py` (49 tests), this file.
`ev/hand.py`, `ev/player.py`, `ev/pairs.py` are read-only here -- everything below is
a report on W-EXTRACT's layer (`EXTRACT_NOTES.md`), not a change to it.

```
python ev/cli.py advise fixture:purple_seal_discard         --player 0
python ev/cli.py advise fixture:purple_seal_discard_control --player 0
... (same for faceless_discard, business_card_board, reserved_parking_hold,
     gold_seal_weak_play, tarot_target_cycle -- each with a _control twin)
python -m pytest ev/tests/test_probe_fixtures.py -q     # 49 passed
python -m pytest ev -q                                  # 254 passed (whole-package, this
                                                             # workstream's scope is green)
```

## 0. What's here and how it's built

Every fixture is a real `MLBMatch`, built with **zero self-play**: `MLBMatch(seed=...)` ->
`m.step(0, {"type": "play_blind"})` lands player 0 at ante-1 `SELECTING_HAND` with the
engine's own default `hands_left=4` / `discards_left=4` / `chips_target=300` (the ante-1
Small blind), then the fixture hand-edits `game.hand` (via `_probe_common.set_hand`, which
pulls the named `(rank, suit)` cards out of `game.full_deck` -- the same objects `deck` /
`hand` share references to, so seals/enhancements/jokers set afterward are read by the real
scoring/discard/round-end code paths, not a parallel model of them), `game.jokers`
(`_probe_common.set_jokers`, real `JokerInstance`s), individual `Card.seal` /
`Card.enhancement`, `game.consumable_hand`, and `game.current_blind.chips_target`. This is
exactly `ev/tests/test_extraction.py`'s own `_set_hand` / `_jokers` idiom (W-EXTRACT's
accepted pattern for constructed states), just wrapped in an `MLBMatch` because the advisor's
`fixture:<name>` state source needs one. No self-play means no RNG stream is consumed before
the hand-edits, so every fixture is trivially deterministic (`m1.signature() ==
m2.signature()` pinned for all 12, `test_fixture_and_control_are_registered_and_deterministic`).

Each of the six scenarios is a pair: `build()` (the sandbag) and `build_control()` (the
matched control), registered as `fixture:<name>` / `fixture:<name>_control`. All six controls
use the brief's "procs absent" recipe (same hand shape, joker/seal/tarot removed) rather than
"low P(clear)" -- section 3 explains why that was the more robust choice, and a *separate*,
explicit low-P(clear) test covers the brief's other ask (safety-gate suppression) directly.

## 1. The fixture table

All EVs are the fast-budget objective (`hand.rank_hand_actions`, no `value_fn`), read directly
off the advisor's own ranking path (`EVPlayer.explain` -> `player._rank_hand`, i.e. what
`cli.py advise` actually prints). "extraction EV" / "clear EV" name the two competing
candidates; Delta is extraction-minus-clear.

| fixture | extraction line | clear-now line | extraction EV | clear EV | Delta | $ extracted | proc(s) firing |
|---|---|---|---|---|---|---|---|
| `purple_seal_discard` | discard [5,6] (2 Purple seals) | play [0,1] (AA) | 1.15336 | 1.04400 | **+0.1094** | $8.00 (2 Tarots) | Purple seal x2 |
| `faceless_discard` | discard [2,3,4] (J,Q,K) | play [0,1] (AA) | 1.11160 | 1.04400 | **+0.0676** | $5.00 | Faceless (3+ discarded faces) |
| `business_card_board` | play [2] (lone Jack) | play [0,1] (AA) | 1.04592 | 1.04400 | **+0.0019** | $1.00 | Business Card (1 scored face) |
| `reserved_parking_hold` | play [7] (lone filler, holds 2 Jacks) | play [0,1] (JJ) | 1.04592 | 1.04400 | **+0.0019** | $1.00 | Reserved Parking (2 held faces) |
| `gold_seal_weak_play` | play [4] (lone Gold 2) | play [0,1] (AA) | 1.07376 | 1.04400 | **+0.0298** | $3.00 | Gold seal |
| `tarot_target_cycle` | play [0..4] *with* Sun held | *same action* without Sun | 1.05700 | 1.04400 | **+0.0130** | $0.93 (cycle EV) | Sun tarot cycling (no real Hearts in hand) |

Every sandbag row's extraction line is the literal #1 ranked action (`test_
sandbag_extraction_line_beats_clear_now`'s second assertion); every control's `extraction_lines()`
is `[]` and `extract_on` is `False` (`test_control_ordering_reverses`); the advisor CLI prints
`[extract $+...]` on line "1." for every sandbag and prints no `[extract $` anywhere for any
control (`test_advisor_renders_the_extraction_line_with_its_money_decomposition`,
`test_advisor_control_shows_no_extraction_line`). All 12 fixtures land at ante-1
`SELECTING_HAND`, non-Nemesis, deterministic, side-effect-free to rank (`state_signature()` /
RNG snapshot pinned before and after).

## 2. The dedicated safety-gate tests

Two tests independent of the "matched control" table above, per the brief's separate ask:

* `test_safety_gate_suppresses_extraction_when_p_clear_is_low` -- re-uses
  `purple_seal_discard`'s own board (real Purple seals still in hand) but overwrites
  `chips_target` to `10**7` (unreachable). `extraction_safe` flips `True -> False` and
  `extraction_lines()` goes from 7 lines to `[]`, on the SAME board that extracted $8.00 at
  its actual target.
* `test_safety_gate_is_off_at_a_nemesis_even_with_procs_present` -- the same board with
  `current_blind.is_pvp = True`: `extract_on` is unconditionally `False` regardless of
  P(clear) (EXTRACT_NOTES.md section 4 -- no unused-hand money at a Nemesis).

## 3. Honest findings -- where the layer's behaviour needed care to demonstrate, or surprised

These are reports on W-EXTRACT's layer, observed while building fixtures, not tuning of it.

### 3.1 "Disjoint procs attach for free" -- most naive constructions don't produce a rank
swap, only an EV bump on the SAME winning action

The first draft of every PLAY-type fixture (`business_card_board`, `reserved_parking_hold`,
`gold_seal_weak_play`) put the proc on cards that were **already** part of, or unrelated to
but never competing with, the objectively-best clearing play. Concretely:

* `business_card_board` v1: Business Card on the only pair the hand had (a King pair) --
  playing the Kings was simultaneously "clear now" AND "the money play." The control showed
  the identical action ranked #1 both times, just with a smaller EV (no reversal to pin).
* `reserved_parking_hold` v1/v2: the held faces were disjoint from every candidate clearing
  play in both directions -- Parking's `proc_hold` only counts what a play EXCLUDES
  (EXTRACT_NOTES.md section 2), so the best chip-max clearing play kept the faces held "for
  free" regardless of Parking. v2's filler cards also accidentally contained a second pair
  (9S/9H) that independently cleared the blind while still holding the target Jacks -- same
  "no sacrifice needed" trap from an unplanned direction.
* `gold_seal_weak_play` v1: filler ranks 2,3,4,5,6 next to an Ace (rank 14, Ace-low eligible)
  built an unintended A-2-3-4-5 straight that happened to include the Gold-sealed card, which
  muddied (though did not break) the intended "deliberately weak single card" reading.

None of these are bugs -- EXTRACT_NOTES.md section 5 already documents that Reserved Parking's
future-hand term is a constant that "cancels in the argmax," and the same logic extends to
*this* hand whenever the held cards are disjoint from the play. The practical reading: **when
a money proc happens to sit on cards that are already part of (or irrelevant to) the fastest
clear, extraction is free money with no tradeoff at all** -- which is the best case for a real
player, but means a naive/random board will usually NOT show a rank reversal, only a same-
action EV bump. Every fixture in the table above was deliberately re-built so the proc sits on
a genuinely WEAKER alternative to the best available clear (a lone face card, a lone
Gold-sealed junk card, a pair that itself doesn't reach the chip target), which is what
produces an actual swap. Anyone extending this fixture set should budget time for this --
it is the single biggest reason six "just add a joker and a card" fixtures took multiple
iterations each.

### 3.2 The safety gate's bite point differs by an order of magnitude between discard-type
and play-type extraction

Swept `chips_target` on two representative fixtures to find where `extraction_safe` actually
flips (holding everything else fixed):

* **`purple_seal_discard` (discard-type, pure junk)**: the gate does not engage until the
  target is essentially unreachable (`10**7` in the test above; a target of `10**5` was also
  checked and still gates off). This is because the two discarded cards were never part of
  the winning play -- discarding them costs nothing in P(clear) as long as ANY clear is still
  possible, so `_p_clear_after(h, d-1, need)` tracks `_p_clear_after(h, d, need)` almost
  exactly. The gate is a backstop against a catastrophically-misjudged board, not a
  fine-grained regulator, for this proc family.
* **`gold_seal_weak_play` (play-type, costs a hand's worth of progress)**: swept
  `chips_target` from 40 (the fixture's actual value) upward in steps; the weak/delayed line
  keeps winning through 420, and flips to "clear-now wins, `extraction_lines()` empties" at
  450 (`extraction_safe` crosses below 0.90 for the weak line's post-action position
  somewhere in [420, 450], i.e. roughly **10x** the fixture's own target). Here the gate is
  genuinely load-bearing: playing the weak card *does* cost the delayed hand's worth of
  P(clear), so the DP notices well before the blind becomes literally unwinnable.

Read together: the brief's phrase "self-regulating... beats clearing now only when the risk
it adds is smaller than that" (EXTRACT_NOTES.md section 4) is accurate, but the size of "that"
varies enormously by extraction TYPE -- pure-junk discards are near-costless almost everywhere
short of a lost blind; resource-spending plays are gated across a real, findable range. This
is useful context for W-RANK's `greedy_vs_extract` pair source (brief section 5.2): pairs built
from discard-type extraction lines will show resolved deltas across almost the whole safe
range of `chips_target`, while play-type pairs sample a genuine decision boundary and are more
likely to land near the pair-CI "unresolved" cutoff close to that boundary.

### 3.3 `tarot_target_cycle` does not produce a same-need rank swap, only a same-action
EV bump -- and this is a real property of `_cycle_ev`, not a fixture-construction failure

> **2026-08-27, W-CYCLE -- this finding was acted on and the fixture is now a rank swap.**
> The diagnosis below is confirmed and was the brief for a follow-up workstream: `_cycle_ev`
> is now per-TARGET (graded per card, expectation over the real pile) and a clearing play
> banks no cycle at all, so `fixture:tarot_target_cycle` was rebuilt and pins a genuine
> swap -- a throwaway dig at 1.051253 over clear-now at 1.044001, reversed in the control.
> `test_tarot_dig_value_depends_on_WHICH_target_card_the_line_keeps` pins the exact
> "depends only on m, not WHICH cards" limitation described below as fixed. See
> `CYCLE_NOTES.md` §1/§4; the reading below is of the model that has been replaced.

Unlike the other five, every attempt to build a "cycling beats a non-cycling clear" rank swap
for `tarot_target_cycle` failed for a structural reason: `_cycle_ev(keep_mask, m)`
(EXTRACT_NOTES.md section 6) depends on `m` (how many fresh cards the action draws) and the
tarot's own `want_mask`/`need`, but **not** on which specific cards are kept beyond that --
every 5-card play in the fixture's hand draws `m=5` fresh cards and gets the identical $0.93
cycle bonus, so there is no lower-chip "clear without cycling" candidate to out-rank; the
bonus attaches near-uniformly to the whole clearing tier. `test_tarot_target_cycle_
sandbag_beats_the_same_line_without_the_tarot` pins the claim that DOES hold (same action,
strictly more EV with the Sun held and no Hearts in hand than with no tarot at all -- 1.05700
vs 1.04400) rather than a rank-swap claim. This is consistent with EXTRACT_NOTES.md section 6's
own documented scope ("which SPECIFIC card the tarot lands on... not modelled") -- the layer
prices "did this line draw fresh cards" but not "did this line draw fresh cards while ALSO
making a different, cheaper structural choice than the best clear," because in this fixture's
hand those two things never diverge. A fixture where they diverge (e.g. a smaller, weaker
kept set that still clears but draws MORE cards) was not found within this workstream's time
budget; flagging as a genuine open question rather than asserting the layer is deficient here
-- it may simply be that "cycle toward a tarot AND sandbag chip surplus" is a real second-order
interaction EXTRACT_NOTES.md never claimed to capture (section 6's "added, not jointly
optimised" between cycling and draw targets).

### 3.4 Minor: two proc families (Business Card, Gold seal single-card play) produce EV
deltas 15-50x smaller than the two discard families in this table

Not a discrepancy, just worth flagging for anyone tuning `tarot_value_dollars` or reading pair
deltas later: a play-type proc fires once per scoring card in ONE hand ($1-3), while a
discard-type proc (Purple seal, Faceless) can fire on 2-3 cards at once without spending a
hand at all ($5-8) -- EXTRACT_NOTES.md section 0's own headline numbers show the same skew
(discard-type procs move the 12-seed dev-slice money more than play-type ones). The two
small-delta rows in the table above (`business_card_board`, `reserved_parking_hold`, both
~0.0019) are close enough to typical label noise (brief section 0: "CI +/-0.24 at
n_rollouts=8") that a real rollout-based pair built from either would likely land in
`pairs.py`'s "unresolved" bucket at low `n_worlds` -- worth a note for W-PAIRS/W-RANK, not an
error in this workstream.

## 4. Test counts / file inventory

* New: `ev/fixtures/_probe_common.py`, `ev/fixtures/{purple_seal_discard,
  faceless_discard, business_card_board, reserved_parking_hold, gold_seal_weak_play,
  tarot_target_cycle}.py`, `ev/tests/test_probe_fixtures.py` (49 tests), this file.
* Edited (additive only, 12 new dict entries + imports, `bloodstone_vs_invisible`'s own line
  untouched): `ev/fixtures/__init__.py`.
* `python -m pytest ev/tests/test_probe_fixtures.py -q` -- 49 passed.
* `python -m pytest ev -q` -- 254 passed (whole package, run after this workstream's
  changes; no failures attributable to this workstream or observed from any other).
