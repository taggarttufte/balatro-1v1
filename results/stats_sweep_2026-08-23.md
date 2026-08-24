# Stats sweep

- seeds: 126  processes: 16  player: `hand=greedy,reroll=1,buy=0`  max_ante: 8  wall: 1.9s
- visits: 800  errors: 0  decide_ms mean/p95: 2.097 / 3.280  urgency mean: 0.660

| ante | visits | reroll P(hit) | reroll net_ev | best net_ev | % leave best | int.loss/true_cost | urgency mean | voucher net_ev | decide ms |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 252 | 0.72 | -2.08 | 3.83 | 22% | 33% | 0.49 | -2.42 | 2.204 |
| 2 | 362 | 0.64 | -1.07 | 6.55 | 11% | 38% | 0.71 | 1.08 | 2.095 |
| 3 | 176 | 0.63 | 1.26 | 9.66 | 4% | 17% | 0.79 | 4.64 | 1.968 |
| 4 | 10 | 0.64 | 2.97 | 10.55 | 0% | 7% | 0.83 | 9.34 | 1.722 |

## Pack net EV by kind, by ante

- ante 1: Arcana=-2.4, Buffoon=-0.6, Celestial=0.3, Spectral=-1.6, Standard=-8.7
- ante 2: Arcana=-1.0, Buffoon=-1.8, Celestial=3.0, Spectral=0.5, Standard=-8.4
- ante 3: Arcana=0.7, Buffoon=1.2, Celestial=5.6, Spectral=2.1, Standard=-6.0
- ante 4: Arcana=2.3, Buffoon=1.2, Celestial=8.4, Spectral=3.1, Standard=-4.7
