# transfer spread -- `scripted:hand=greedy`

mode=both  stake=1  max_antes=8  target=vanilla_boss  n_solo_seeds=150  n_tournament_seeds=16  n_agents=32  wall_clock=290.9s

## mode (a): SP-MLB-solo vs external target

| deck | furthest ante | lives lost | win rate | margin p10 | p50 | p90 |
|---|---|---|---|---|---|---|
| b_red | 2.41 [2.33,2.49] | 4.00 [4.00,4.00] | 0.000 [0.000,0.000] | -1216 | -976 | -656 |
| b_checkered | 3.00 [2.98,3.02] | 4.00 [4.00,4.00] | 0.023 [0.007,0.043] | -2848 | -552 | -376 |
| b_plasma | 3.58 [3.51,3.65] | 4.00 [4.00,4.00] | 0.300 [0.257,0.340] | -5192 | -1506 | 1279 |

## mode (b): tournament population rank

| deck | mean rank_frac (0=best, 1=worst) |
|---|---|
| b_red | 0.665 [0.624,0.707] |
| b_checkered | 0.601 [0.564,0.641] |
| b_plasma | 0.566 [0.522,0.612] |

## cross-cell spread (bootstrap over paired seeds)

| metric | b_red | b_checkered | b_plasma | range [95% CI] | variance [95% CI] | n_paired_seeds |
|---|---|---|---|---|---|---|
| win_rate | 0.000 | 0.027 | 0.295 | 0.295 [0.250,0.341] | 0.0178 [0.0126,0.0239] | 132 |
| rank_frac | 0.665 | 0.601 | 0.566 | 0.100 [0.070,0.132] | 0.0017 [0.0009,0.0030] | 16 |
