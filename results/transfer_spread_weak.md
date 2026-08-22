# transfer spread -- `scripted:hand=weak`

mode=both  stake=1  max_antes=8  target=vanilla_boss  n_solo_seeds=150  n_tournament_seeds=16  n_agents=32  wall_clock=389.7s

## mode (a): SP-MLB-solo vs external target

| deck | furthest ante | lives lost | win rate | margin p10 | p50 | p90 |
|---|---|---|---|---|---|---|
| b_red | 2.00 [2.00,2.00] | 4.00 [4.00,4.00] | nan [nan,nan] | None | None | None |
| b_checkered | 2.00 [2.00,2.00] | 4.00 [4.00,4.00] | nan [nan,nan] | None | None | None |
| b_plasma | 2.00 [2.00,2.00] | 4.00 [4.00,4.00] | nan [nan,nan] | None | None | None |

## mode (b): tournament population rank

| deck | mean rank_frac (0=best, 1=worst) |
|---|---|
| b_red | 0.990 [0.982,0.997] |
| b_checkered | 0.992 [0.984,0.999] |
| b_plasma | 0.989 [0.981,0.997] |

## cross-cell spread (bootstrap over paired seeds)

| metric | b_red | b_checkered | b_plasma | range [95% CI] | variance [95% CI] | n_paired_seeds |
|---|---|---|---|---|---|---|
| win_rate | n/a | n/a | n/a | n/a | n/a | 0 |
| rank_frac | 0.990 | 0.992 | 0.989 | 0.003 [0.001,0.007] | 0.0000 [0.0000,0.0000] | 16 |
