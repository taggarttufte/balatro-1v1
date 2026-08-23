# H2H: ev:fast vs ev:full

- seeds: 2 (['11111111', '1558AXDL'])
- trials: 4 (both seatings per seed); decided 4, undecided 0
- sims=40  checkpoint=None  lives=4  max_steps=100000  deck=b_red  stake=1
- procs=2  wall clock: 43.7s  mean 17.22s/match

## Summary

- **A (ev:fast) wins**: 2 / 4  (win rate 0.500, 95% CI [0.000, 1.000])
- **B (ev:full) wins**: 2 / 4
- mean final ante: A 6.50  vs  B 6.50
- mean lives margin (A - B): -1.00
- Nemesis win rate (A's side of every resolved Nemesis): 0.400

## Per-trial

| seed | seating | a_win | lives A/B | ante A/B | nem A/B / total | steps | seconds |
|---|---|---|---|---|---|---|---|
| 1558AXDL | 0 | False | 0/3 | 6/6 | 1/4/5 | 371 | 13.97 |
| 1558AXDL | 1 | False | 0/3 | 6/6 | 1/4/5 | 369 | 13.71 |
| 11111111 | 0 | True | 1/0 | 7/7 | 3/2/5 | 517 | 20.69 |
| 11111111 | 1 | True | 1/0 | 7/7 | 3/2/5 | 517 | 20.50 |
