# Stats sweep

- seeds: 6  processes: 4  player: `hand=greedy,reroll=1,buy=0`  max_ante: 8  wall: 0.7s
- visits: 38  errors: 0  decide_ms mean/p95: 1.666 / 2.383  urgency mean: 0.664

| ante | visits | reroll P(hit) | reroll net_ev | best net_ev | % leave best | int.loss/true_cost | urgency mean | voucher net_ev | decide ms |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 12 | 0.72 | -1.97 | 2.58 | 25% | 40% | 0.52 | -3.28 | 1.645 |
| 2 | 16 | 0.64 | -1.39 | 5.67 | 0% | 43% | 0.69 | -1.09 | 1.632 |
| 3 | 9 | 0.63 | 0.60 | 5.83 | 11% | 33% | 0.79 | -1.16 | 1.734 |
| 4 | 1 | 0.64 | 2.97 | 6.49 | 0% | 0% | 0.83 | 6.49 | 1.856 |

## Pack net EV by kind, by ante

- ante 1: Arcana=-2.3, Buffoon=-1.1, Celestial=-0.8, Spectral=-1.0, Standard=-10.4
- ante 2: Arcana=0.1, Buffoon=-2.3, Celestial=3.7, Standard=-10.8
- ante 3: Arcana=0.1, Celestial=4.0, Spectral=3.0, Standard=-10.1
- ante 4: Arcana=1.3, Standard=-4.0
