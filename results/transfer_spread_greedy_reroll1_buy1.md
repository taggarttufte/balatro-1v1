# transfer spread -- `scripted:hand=greedy,reroll=1,buy=1`

mode=both  stake=1  max_antes=8  target=vanilla_boss  n_solo_seeds=150  n_tournament_seeds=16  n_agents=32  wall_clock=408.4s

## mode (a): SP-MLB-solo vs external target

| deck | furthest ante | lives lost | win rate | margin p10 | p50 | p90 |
|---|---|---|---|---|---|---|
| b_red | 3.06 [2.93,3.19] | 4.00 [4.00,4.00] | 0.280 [0.220,0.341] | -1590 | -564 | 1690 |
| b_checkered | 3.77 [3.63,3.92] | 4.00 [4.00,4.00] | 0.289 [0.244,0.334] | -2880 | -432 | 2630 |
| b_plasma | 4.17 [4.04,4.30] | 3.99 [3.98,4.00] | 0.430 [0.391,0.468] | -6186 | -326 | 6630 |

## mode (b): tournament population rank

| deck | mean rank_frac (0=best, 1=worst) |
|---|---|
| b_red | 0.333 [0.258,0.422] |
| b_checkered | 0.295 [0.249,0.344] |
| b_plasma | 0.338 [0.285,0.397] |

## cross-cell spread (bootstrap over paired seeds)

| metric | b_red | b_checkered | b_plasma | range [95% CI] | variance [95% CI] | n_paired_seeds |
|---|---|---|---|---|---|---|
| win_rate | 0.280 | 0.297 | 0.435 | 0.155 [0.108,0.222] | 0.0048 [0.0023,0.0093] | 146 |
| rank_frac | 0.333 | 0.295 | 0.338 | 0.043 [0.013,0.113] | 0.0004 [0.0000,0.0022] | 16 |
