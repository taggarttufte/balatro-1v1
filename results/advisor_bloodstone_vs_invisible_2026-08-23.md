# Advisor acceptance test: Bloodstone vs Invisible Joker(+Blueprint) (2026-08-23)

Phase 5 rev 2 gate 5 (`mp/docs/PHASE5_BRIEF_2026-08.md` §3): "the snapshot advisor prints two
P(win) +/- CI for Tagg's state." Fixture: `mp/ev/fixtures/bloodstone_vs_invisible.py`.
Command: `python mp/ev/cli.py advise fixture:bloodstone_vs_invisible --player {0,1} --rollouts 32`.
No V checkpoint exists yet (Phase 5 W5 hasn't trained one) -- both runs below are `--checkpoint`-free,
so the report's "V" line correctly prints `n/a`; this doc still stands as the acceptance
check per PHASE5_BRIEF_2026-08.md's own note ("the advisor must work without one").

## Timing (32 rollouts, sequential -- `labels.label_state` has no internal parallelism)

| perspective | wall clock | s/rollout |
|---|---|---|
| player 0 | 77.3 s | 2.42 |
| player 1 | 80.0 s | 2.50 |

Both single-process, well inside the "<=4 processes, no job > ~30 min" constraint. Ranked
action table cost is negligible next to the rollouts (33-43 ms). Both perspectives together:
~2.6 minutes wall clock for the whole acceptance run.

## Verbatim output -- player 0 (Bloodstone side)

```
==============================================================================
EV ADVISOR -- player 0  (seed TAGGADVR)
==============================================================================
Ante 4  blind 2 (Boss <NEMESIS>)  state=SHOP
Lives:  me 3  vs  opp 4      $:  me 43  vs  opp 25
My jokers  (6/5): Clever Joker, Stuntman, Reserved Parking, Mystic Summit, Devious Joker, Bloodstone
Opponent jokers [HIDDEN IN A REAL MATCH -- shown here only because this state source has full simulator access]: Clever Joker, Stuntman, Reserved Parking, Mystic Summit, Devious Joker, Blueprint, Invisible Joker

Opponent PUBLIC block (opponent_view -- what a live match actually reveals):
  lives=4  $=25  ante=4  blind_idx=2  state=SHOP  phase=shop
  chips_scored=11870  hands_left=0  comeback_bonus=0  comeback_pending=0
  econ: sells_per_ante=0  spent_in_shop=0  sells_total=0  spent_total=31
  last Nemeses (most recent first):
    ante 3: they scored 11870 in 4 hands, I scored 8956 -> I lost a life
    ante 2: they scored 15122 in 4 hands, I scored 13798 -> I lost a life

P(win) -- three estimators:
  rollout  (n=32, budget=fast, determinized, symmetric analytic opponent): 0.906 +/- 0.105   [77.3s]
  race     (curve_from_history: my n_obs=2, their n_obs=2, ante=4): 0.215
  V        (no --checkpoint given): n/a
  ** DISAGREEMENT: rollout vs race differ by 0.691 **
  race table (next 4 antes):
    ante  4  my mu/sigma=4.12/0.37  their mu/sigma=4.20/0.36  P(I lose Nemesis)=0.56  p_win_from_here=0.210
    ante  5  my mu/sigma=4.17/0.37  their mu/sigma=4.25/0.36  P(I lose Nemesis)=0.56  p_win_from_here=0.207
    ante  6  my mu/sigma=4.22/0.37  their mu/sigma=4.30/0.36  P(I lose Nemesis)=0.56  p_win_from_here=0.203
    ante  7  my mu/sigma=4.27/0.37  their mu/sigma=4.35/0.36  P(I lose Nemesis)=0.56  p_win_from_here=0.203

Ranked actions (SHOP, budget=full, 6 candidates, 32.7 ms):
  1. buy Reroll Surplus ($10)                   ev=+0.0200  buy voucher Reroll Surplus $10 (affordable after floor)
  2. buy Arcana Pack ($4)                       ev=+0.0110  open Arcana Pack $4 (ok)
  3. buy Standard Pack ($4)                     ev=+0.0090  open Standard Pack $4 (ok)
  4. buy The Star ($3)                          ev=+0.0000  buy The Star $3 (gain +0.000)
  5. leave shop                                 ev=+0.0000  leave (P(clear next) 1.00, $43)
  6. reroll ($5)                                ev=-1.0000  reroll $5 (no; 0 done)

Decision-stats table (P(hit), true cost incl. interest, urgency, net EV):
kind            label                              p_hit  hit_val  cost  int_loss  true_cost  urgency  net_ev
--------------  ---------------------------------  -----  -------  ----  --------  ---------  -------  ------
reroll          reroll ($5, P(hit)=78%)            0.78   7.8      5     0.0       5.0        0.05     +1.4
buy_consumable  buy The Star ($3)                  1.00   4.0      3     0.0       3.0        0.05     +1.2
buy_pack        buy Arcana Pack ($4, P(hit)=100%)  1.00   4.0      4     0.0       4.0        0.05     +0.2
leave           leave shop                         1.00   0.0      0     0.0       0.0        0.05     +0.0
sell            sell Clever Joker (+$2)            1.00   -2.0     -2    0.0       -2.0       0.05     -0.1
sell            sell Devious Joker (+$2)           1.00   -2.0     -2    0.0       -2.0       0.05     -0.1
sell            sell Reserved Parking (+$3)        1.00   -3.0     -3    0.0       -3.0       0.05     -0.2
buy_voucher     buy voucher Reroll Surplus ($10)   1.00   9.0      10    0.0       10.0       0.05     -0.5
sell            sell Bloodstone (+$3)              1.00   -6.6     -3    0.0       -3.0       0.05     -4.0
buy_pack        buy Standard Pack ($4, P(hit)=0%)  0.00   0.0      4     0.0       4.0        0.05     -4.0
sell            sell Mystic Summit (+$2)           1.00   -15.4    -2    0.0       -2.0       0.05     -14.1
sell            sell Stuntman (+$3)                1.00   -16.9    -3    0.0       -3.0       0.05     -14.7

Opponent read (level-0 -- public features only, no belief model over the shared menu / signalling yet):
  most recent Nemesis (ante 3): they scored 11870 in 4 hands, they won it
  shop spend this ante: $0, sold 0 joker(s) this ante  (lifetime: $31 spent, 0 sold)
  no inference beyond these public numbers is attempted.
==============================================================================
```

## Verbatim output -- player 1 (Invisible + Blueprint side)

```
==============================================================================
EV ADVISOR -- player 1  (seed TAGGADVR)
==============================================================================
Ante 4  blind 2 (Boss <NEMESIS>)  state=SHOP
Lives:  me 4  vs  opp 3      $:  me 25  vs  opp 43
My jokers  (7/5): Clever Joker, Stuntman, Reserved Parking, Mystic Summit, Devious Joker, Blueprint, Invisible Joker
Opponent jokers [HIDDEN IN A REAL MATCH -- shown here only because this state source has full simulator access]: Clever Joker, Stuntman, Reserved Parking, Mystic Summit, Devious Joker, Bloodstone

Opponent PUBLIC block (opponent_view -- what a live match actually reveals):
  lives=3  $=43  ante=4  blind_idx=2  state=SHOP  phase=shop
  chips_scored=8956  hands_left=0  comeback_bonus=2  comeback_pending=0
  econ: sells_per_ante=0  spent_in_shop=0  sells_total=0  spent_total=27
  last Nemeses (most recent first):
    ante 3: they scored 8956 in 4 hands, I scored 11870 -> I took a life
    ante 2: they scored 13798 in 4 hands, I scored 15122 -> I took a life

P(win) -- three estimators:
  rollout  (n=32, budget=fast, determinized, symmetric analytic opponent): 0.094 +/- 0.105   [80.0s]
  race     (curve_from_history: my n_obs=2, their n_obs=2, ante=4): 0.785
  V        (no --checkpoint given): n/a
  ** DISAGREEMENT: rollout vs race differ by 0.691 **
  race table (next 4 antes):
    ante  4  my mu/sigma=4.20/0.36  their mu/sigma=4.12/0.37  P(I lose Nemesis)=0.44  p_win_from_here=0.790
    ante  5  my mu/sigma=4.25/0.36  their mu/sigma=4.17/0.37  P(I lose Nemesis)=0.44  p_win_from_here=0.793
    ante  6  my mu/sigma=4.30/0.36  their mu/sigma=4.22/0.37  P(I lose Nemesis)=0.44  p_win_from_here=0.797
    ante  7  my mu/sigma=4.35/0.36  their mu/sigma=4.27/0.37  P(I lose Nemesis)=0.44  p_win_from_here=0.797

Ranked actions (SHOP, budget=full, 6 candidates, 42.6 ms):
  1. buy Arcana Pack ($4)                       ev=+0.0110  open Arcana Pack $4 (ok)
  2. buy Standard Pack ($4)                     ev=+0.0090  open Standard Pack $4 (ok)
  3. leave shop                                 ev=+0.0000  leave (P(clear next) 1.00, $25)
  4. buy Jupiter ($3)                           ev=-0.4521  buy Jupiter $3 (gain -0.452)
  5. buy Reroll Surplus ($10)                   ev=-1.0000  buy voucher Reroll Surplus $10 (below floor)
  6. reroll ($5)                                ev=-1.0000  reroll $5 (no; 0 done)

Decision-stats table (P(hit), true cost incl. interest, urgency, net EV):
kind            label                              p_hit  hit_val  cost  int_loss  true_cost  urgency  net_ev
--------------  ---------------------------------  -----  -------  ----  --------  ---------  -------  ------
leave           leave shop                         1.00   0.0      0     0.0       0.0        0.00     +0.0
sell            sell Clever Joker (+$2)            1.00   -2.0     -2    0.0       -2.0       0.00     +0.0
sell            sell Reserved Parking (+$3)        1.00   -3.0     -3    0.0       -3.0       0.00     +0.0
sell            sell Devious Joker (+$2)           1.00   -2.0     -2    0.0       -2.0       0.00     +0.0
sell            sell Blueprint (+$5)               1.00   -5.0     -5    0.0       -5.0       0.00     +0.0
sell            sell Invisible Joker (+$4)         1.00   -4.0     -4    0.0       -4.0       0.00     +0.0
reroll          reroll ($5, P(hit)=84%)            0.84   7.8      5     3.2       8.2        0.00     -1.6
buy_pack        buy Arcana Pack ($4, P(hit)=100%)  1.00   4.0      4     3.2       7.2        0.00     -3.2
buy_consumable  buy Jupiter ($3)                   1.00   1.6      3     3.2       6.2        0.00     -4.5
buy_pack        buy Standard Pack ($4, P(hit)=0%)  0.00   0.0      4     3.2       7.2        0.00     -7.2
buy_voucher     buy voucher Reroll Surplus ($10)   1.00   9.0      10    6.3       16.3       0.00     -7.3
sell            sell Stuntman (+$3)                1.00   -16.3    -3    0.0       -3.0       0.00     -13.3
sell            sell Mystic Summit (+$2)           1.00   -15.4    -2    0.0       -2.0       0.00     -13.4

Opponent read (level-0 -- public features only, no belief model over the shared menu / signalling yet):
  most recent Nemesis (ante 3): they scored 8956 in 4 hands, I won it
  shop spend this ante: $0, sold 0 joker(s) this ante  (lifetime: $27 spent, 0 sold)
  no inference beyond these public numbers is attempted.
==============================================================================
```

## Reading (candid -- does this look like what a strong player would say?)

1. Both perspectives sum correctly (`rollout_p0 + rollout_p1 = 1.000`, `race_p0 + race_p1 =
   1.000` exactly) -- the symmetry the design promises holds numerically, not just in theory.
2. The DISAGREEMENT flag fires legitimately, not from noise: 32 rollouts give a tight CI
   (+/-0.105) around 0.906/0.094, and it stays that lopsided at n=32 after being noisier (and
   less extreme) at n=2/n=4 during development -- this is a real, reproducible gap, not
   sampling jitter.
3. **Why they disagree is understandable and worth explaining to Tagg**: `race` fits its
   curves from `match.pvp_log`, which is the history from BEFORE this fixture's joker edits
   (player 0 lost both recorded Nemeses 2-for-2) -- it has no way to know Bloodstone /
   the Hearts retint just landed. `rollout` plays the ACTUAL current board forward, so it
   captures the new strength immediately. This is exactly the situation §2's "curve is
   fitted to what the build actually scored" caveat in `race.py` warns about -- a real,
   documented blind spot, not a bug in either estimator.
4. The clearest "this looks dumb" finding is internal, not between the three P(win) numbers:
   for player 0, `EVPlayer.explain`'s own ranked-action table puts **reroll dead last**
   (`ev=-1.0000`, "no; 0 done") while the `decide.py` stats table puts the SAME reroll
   **first** (`net_ev=+1.4`, the best row in the shop). The rule tier's reroll gate ("reroll
   only when nothing else looks buyable") is categorical and coarser than `decide.py`'s
   actual dollar-denominated math -- exactly what `EV_NOTES.md` section 8 item 4 already
   flags as the rules tier's known weak spot. `ev:full+stats` (wired in `h2h.py`) is the
   fix already available; the bootstrap rules only need to hold until V lands.
5. Both ranked-action tables are otherwise sane: nothing looks urgently buyable (`ev`s
   cluster near 0), the model correctly refuses to sell Bloodstone (`-4.0` net) or the newly
   added Blueprint/Invisible pair (`+0.0`, i.e. no incentive either way) -- no obviously
   wrong call, just a shop this well-stocked a board shouldn't be excited about.
6. **The Blueprint-on-Invisible "combo" is a genuine dead joker slot** (see
   `bloodstone_vs_invisible.py`'s docstring: Blueprint forwards none of Invisible's two real
   hooks), and neither table has any lever to fix it -- this engine has no "reorder jokers"
   action at all, so even a perfect advisor could only ever recommend selling one of the two,
   never repositioning Blueprint next to something it can actually copy. Worth flagging as an
   action-space gap, not an advisor bug.
7. `V` correctly prints `n/a` and is silently skipped from the disagreement check -- the
   3-estimator design degrades to 2 cleanly, as the brief asks for.
8. The opponent PUBLIC block and the "hidden in a real match" joker line read exactly as
   intended: a real deployment would see the PUBLIC block's numbers and nothing else about
   the opponent's board -- this fixture prints both only because it has full simulator access.
9. Net verdict: the advisor's arithmetic is trustworthy and its own uncertainty-flagging
   caught a real disagreement worth a human's attention -- but its SHOP judgment is currently
   bottlenecked by the crude bootstrap rule tier, which is expected (V isn't trained yet) and
   already has a documented, wired replacement (`stats`) sitting one flag away.
10. This is a reasonable "junior EV player" read for a state this early in V's absence: right
    order of magnitude, correctly humble where it disagrees with itself, visibly not yet a
    strong player in the shop specifically.
