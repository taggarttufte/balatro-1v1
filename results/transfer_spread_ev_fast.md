# transfer spread -- `ev:fast`

mode=solo  stake=1  max_antes=8  target=vanilla_boss  n_solo_seeds=150  n_tournament_seeds=0  n_agents=32  wall_clock=395.2s

## mode (a): SP-MLB-solo vs external target

| deck | furthest ante | lives lost | win rate | margin p10 | p50 | p90 |
|---|---|---|---|---|---|---|
| b_red | 6.97 [6.79,7.17] | 3.60 [3.43,3.75] | 0.713 [0.690,0.735] | -15792 | 2552 | 17765 |
| b_checkered | 7.33 [7.13,7.53] | 3.48 [3.29,3.65] | 0.726 [0.705,0.748] | -15247 | 3980 | 21661 |
| b_plasma | 7.07 [6.87,7.28] | 3.49 [3.29,3.66] | 0.714 [0.688,0.740] | -33170 | 6431 | 45557 |

## cross-cell spread (bootstrap over paired seeds)

| metric | b_red | b_checkered | b_plasma | range [95% CI] | variance [95% CI] | n_paired_seeds |
|---|---|---|---|---|---|---|
| win_rate | 0.713 | 0.726 | 0.714 | 0.013 [0.004,0.045] | 0.0000 [0.0000,0.0004] | 150 |
