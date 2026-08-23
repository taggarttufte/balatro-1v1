# Clairvoyance measurement — C:\Users\Taggart\projects\balatro-rl\mp\agent\runs\real1\latest.pt

Seeds: 2 (11111111, 1558AXDL) · sims=8 · determinize_mode=per_sim · ruleset=vanilla · max wall 157s

## (a) Outcome table — clairvoyant vs determinized, paired by seed

| field | clairvoyant | determinized | diff (clair - det) | 95% CI |
|---|---|---|---|---|
| mean final ante | 1.500 | 1.000 | +0.500 | [+0.000, +1.000] |
| mean blinds cleared | 3.500 | 2.000 | +1.500 | [+0.000, +3.000] |
| ante-1 clear rate | 0.500 | 0.000 | +0.500 | [+0.000, +1.000] |
| ante-2 clear rate | 0.000 | 0.000 | +0.000 | [+0.000, +0.000] |
| ante-3 clear rate | 0.000 | 0.000 | +0.000 | [+0.000, +0.000] |
| mean final $ | 4.500 | 8.000 | -3.500 | [-7.000, +0.000] |
| mean s/decision | 0.045 | 0.070 | -0.025 | [-0.027, -0.023] |

## (b) Disagreement table — determinized vs clairvoyant's own trajectory

38 real decision points probed (>1 legal action). Overall agreement: 0.342 [0.211, 0.500]

| action type | n | agreement rate | 95% CI |
|---|---|---|---|
| play | 12 | 0.000 | [0.000, 0.000] |
| discard | 11 | 0.000 | [0.000, 0.000] |
| skip_blind | 6 | 1.000 | [1.000, 1.000] |
| buy | 4 | 0.750 | [0.250, 1.000] |
| other | 2 | 1.000 | [1.000, 1.000] |
| reroll | 1 | 1.000 | [1.000, 1.000] |
| leave_shop | 1 | 1.000 | [1.000, 1.000] |
| sell | 1 | 0.000 | [0.000, 0.000] |

Mean wall-clock per probed decision: clairvoyant 43.0 ms, determinized 61.6 ms.

## Interpretation

(fill in by hand after a full run: point at the outcome diff and the agreement-by-type breakdown above)
