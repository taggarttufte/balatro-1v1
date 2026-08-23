# H2H: ev:fast vs real1:det

- seeds: 2 (['11111111', '1558AXDL'])
- trials: 4 (both seatings per seed); decided 4, undecided 0
- sims=16  checkpoint=None  lives=4  max_steps=100000  deck=b_red  stake=1
- procs=2  wall clock: 26.6s  mean 9.71s/match

## Summary

- **A (ev:fast) wins**: 4 / 4  (win rate 1.000, 95% CI [1.000, 1.000])
- **B (real1:det) wins**: 0 / 4
- mean final ante: A 4.50  vs  B 4.50
- mean lives margin (A - B): +4.00
- Nemesis win rate (A's side of every resolved Nemesis): 1.000

## Per-trial

| seed | seating | a_win | lives A/B | ante A/B | nem A/B / total | steps | seconds |
|---|---|---|---|---|---|---|---|
| 11111111 | 0 | True | 4/0 | 4/4 | 3/0/3 | 188 | 7.92 |
| 11111111 | 1 | True | 4/0 | 5/5 | 4/0/4 | 235 | 9.72 |
| 1558AXDL | 0 | True | 4/0 | 5/5 | 4/0/4 | 221 | 11.91 |
| 1558AXDL | 1 | True | 4/0 | 4/4 | 3/0/3 | 211 | 9.28 |
