# Clairvoyance measurement — C:\Users\Taggart\projects\balatro-rl\mp\agent\runs\real1\latest.pt

Seeds: 30 (11111111, 1558AXDL, 15H9Z3IY, 1KV4W6YS, 1MD1YZ9T, 28V7DD4H, 29DAQVG1, 29Y3L4S9...) · sims=40 · determinize_mode=per_sim · ruleset=vanilla · max wall 1159s

## (a) Outcome table — clairvoyant vs determinized, paired by seed

| field | clairvoyant | determinized | diff (clair - det) | 95% CI |
|---|---|---|---|---|
| mean final ante | 2.600 | 1.833 | +0.767 | [+0.000, +1.533] |
| mean blinds cleared | 6.800 | 4.500 | +2.300 | [+0.000, +4.600] |
| ante-1 clear rate | 0.633 | 0.333 | +0.300 | [+0.100, +0.500] |
| ante-2 clear rate | 0.400 | 0.200 | +0.200 | [+0.000, +0.400] |
| ante-3 clear rate | 0.333 | 0.133 | +0.200 | [+0.000, +0.400] |
| mean final $ | 4.633 | 4.133 | +0.500 | [-2.167, +3.367] |
| mean s/decision | 0.188 | 0.278 | -0.090 | [-0.104, -0.075] |

## (b) Disagreement table — determinized vs clairvoyant's own trajectory

1264 real decision points probed (>1 legal action). Overall agreement: 0.357 [0.331, 0.384]

| action type | n | agreement rate | 95% CI |
|---|---|---|---|
| play | 391 | 0.105 | [0.077, 0.138] |
| buy | 188 | 0.367 | [0.298, 0.431] |
| discard | 150 | 0.007 | [0.000, 0.020] |
| leave_shop | 116 | 0.776 | [0.698, 0.845] |
| skip_blind | 89 | 0.820 | [0.730, 0.899] |
| pick_booster | 74 | 0.500 | [0.392, 0.608] |
| play_blind | 69 | 0.623 | [0.507, 0.725] |
| sell | 56 | 0.429 | [0.304, 0.571] |
| reroll | 55 | 0.364 | [0.236, 0.491] |
| use_consumable | 45 | 0.578 | [0.444, 0.711] |
| other | 31 | 0.871 | [0.742, 0.968] |

Mean wall-clock per probed decision: clairvoyant 174.2 ms, determinized 268.0 ms.

## Interpretation (lead, 2026-08-24)

1. **The Phase-4 numbers were mostly clairvoyance.** Denied the true future, the same net + search
   drops from 63% to 33% ante-1 clears and from 2.60 to 1.83 mean final ante — i.e. honest `real1`
   is roughly at the level of the scripted greedy player (31.7% ante-1 on the 126-seed gate), after
   106 generations of training.
2. **Where the cheating lived: hand play.** Determinized search agrees with the clairvoyant
   trajectory on only 10.5% of `play` and **0.7% of `discard`** decisions — the discards were almost
   pure future-reading (discard exactly what the known draw replaces). Strategic actions agree far
   more (skip_blind 82%, leave_shop 78%): the net's learned *policy shape* was mostly economic, the
   *hand skill* was the oracle's.
3. **Context.** The analytic EVPlayer (no net, no search) clears ante 1 at 95.2% [91.3, 98.4] and
   reaches mean ante 4.69 on the same-family seeds — 2.6x the honest baseline's final ante at ~1/40th
   the per-decision cost (4.6 ms vs 268 ms). This is the quantitative case for the Phase-5 pivot:
   Balatro hand play is an EV calculation over known chance, and MCTS was a poor, cheating estimator
   of it. `real1/latest.pt` remains the baseline, but every number citing it must be determinized.
