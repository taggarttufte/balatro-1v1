# H2H: ev:fast vs ev:full

- seeds: 30 (['11111111', '1558AXDL', '15H9Z3IY', '1KV4W6YS', '1MD1YZ9T', '28V7DD4H', '29DAQVG1', '29Y3L4S9', '29ZSW8MY', '2BRGI767', '2CP4KSXZ', '2GHBLJD9', '2H9N3ISZ', '2K9H9HN', '34JCNMPA', '3SZ71111', '41Y71M6E', '46Y8UZEG', '4H2L46CE', '4K8A9QER', '4R219TNX', '4T1SZKLF', '4UEGRRRA', '51F2NVWK', '56QZEVDV', '5AWWF1M1', '5UIUKHCI', '5YVHAEP', '6H6WNQM2', '7I4M53DL'])
- trials: 60 (both seatings per seed); decided 60, undecided 0
- sims=40  checkpoint=None  lives=4  max_steps=100000  deck=b_red  stake=1
- procs=4  wall clock: 210.4s  mean 13.31s/match

## Summary

- **A (ev:fast) wins**: 26 / 60  (win rate 0.433, 95% CI [0.317, 0.567])
- **B (ev:full) wins**: 34 / 60
- mean final ante: A 6.00  vs  B 6.00
- mean lives margin (A - B): -0.40
- Nemesis win rate (A's side of every resolved Nemesis): 0.486

## Per-trial

| seed | seating | a_win | lives A/B | ante A/B | nem A/B / total | steps | seconds |
|---|---|---|---|---|---|---|---|
| 1KV4W6YS | 0 | False | 0/3 | 6/6 | 1/4/5 | 457 | 13.17 |
| 1KV4W6YS | 1 | False | 0/3 | 6/6 | 1/4/5 | 457 | 12.54 |
| 1558AXDL | 0 | True | 1/0 | 6/6 | 2/2/5 | 397 | 15.60 |
| 1558AXDL | 1 | True | 1/0 | 6/6 | 2/2/5 | 397 | 14.34 |
| 15H9Z3IY | 0 | False | 0/2 | 7/7 | 2/3/5 | 429 | 16.76 |
| 15H9Z3IY | 1 | False | 0/2 | 7/7 | 2/3/5 | 429 | 16.07 |
| 11111111 | 0 | True | 1/0 | 6/6 | 3/2/5 | 448 | 21.74 |
| 11111111 | 1 | True | 1/0 | 6/6 | 3/2/5 | 448 | 19.35 |
| 1MD1YZ9T | 0 | True | 1/0 | 6/6 | 2/2/4 | 356 | 12.70 |
| 1MD1YZ9T | 1 | True | 1/0 | 6/6 | 2/2/4 | 356 | 11.39 |
| 29DAQVG1 | 0 | True | 2/0 | 5/5 | 3/1/4 | 271 | 12.27 |
| 29DAQVG1 | 1 | True | 2/0 | 5/5 | 3/1/4 | 271 | 11.91 |
| 28V7DD4H | 0 | True | 2/0 | 6/6 | 3/2/5 | 402 | 17.75 |
| 28V7DD4H | 1 | True | 2/0 | 6/6 | 3/2/5 | 402 | 16.63 |
| 29Y3L4S9 | 0 | False | 0/1 | 6/6 | 2/3/5 | 450 | 16.15 |
| 29Y3L4S9 | 1 | False | 0/1 | 6/6 | 2/3/5 | 450 | 15.44 |
| 2BRGI767 | 0 | False | 0/2 | 7/7 | 2/3/5 | 519 | 11.28 |
| 2BRGI767 | 1 | False | 0/2 | 7/7 | 2/3/5 | 519 | 11.37 |
| 29ZSW8MY | 0 | True | 1/0 | 7/7 | 3/3/6 | 469 | 16.21 |
| 29ZSW8MY | 1 | True | 1/0 | 7/7 | 3/3/6 | 469 | 16.10 |
| 2K9H9HN | 0 | False | 0/3 | 5/5 | 1/3/4 | 295 | 7.72 |
| 2K9H9HN | 1 | False | 0/3 | 5/5 | 1/3/4 | 295 | 7.48 |
| 2GHBLJD9 | 0 | False | 0/2 | 6/6 | 1/3/4 | 366 | 13.16 |
| 2GHBLJD9 | 1 | False | 0/2 | 6/6 | 1/3/4 | 367 | 12.66 |
| 34JCNMPA | 0 | True | 3/0 | 5/5 | 3/1/4 | 262 | 8.64 |
| 34JCNMPA | 1 | True | 3/0 | 5/5 | 3/1/4 | 262 | 8.09 |
| 2CP4KSXZ | 0 | True | 1/0 | 8/8 | 4/3/7 | 549 | 25.34 |
| 2CP4KSXZ | 1 | True | 1/0 | 8/8 | 4/3/7 | 549 | 24.92 |
| 2H9N3ISZ | 0 | True | 1/0 | 6/6 | 4/1/5 | 417 | 20.42 |
| 2H9N3ISZ | 1 | True | 1/0 | 6/6 | 4/1/5 | 417 | 19.55 |
| 3SZ71111 | 0 | False | 0/1 | 6/6 | 2/2/4 | 350 | 15.53 |
| 3SZ71111 | 1 | False | 0/1 | 6/6 | 2/2/4 | 350 | 14.80 |
| 41Y71M6E | 0 | True | 3/0 | 5/5 | 3/1/4 | 277 | 8.47 |
| 41Y71M6E | 1 | True | 3/0 | 5/5 | 3/1/4 | 277 | 8.39 |
| 46Y8UZEG | 0 | False | 0/2 | 5/5 | 2/1/3 | 285 | 10.59 |
| 46Y8UZEG | 1 | False | 0/2 | 5/5 | 2/1/3 | 285 | 10.12 |
| 4H2L46CE | 0 | True | 1/0 | 6/6 | 4/1/5 | 345 | 12.42 |
| 4H2L46CE | 1 | True | 1/0 | 6/6 | 4/1/5 | 345 | 11.95 |
| 4T1SZKLF | 0 | False | 0/4 | 5/5 | 0/3/3 | 268 | 7.42 |
| 4T1SZKLF | 1 | False | 0/4 | 5/5 | 0/3/3 | 268 | 6.93 |
| 4K8A9QER | 0 | False | 0/3 | 6/6 | 1/4/5 | 351 | 12.81 |
| 4K8A9QER | 1 | False | 0/3 | 6/6 | 1/4/5 | 351 | 12.55 |
| 4UEGRRRA | 0 | False | 0/1 | 5/5 | 1/2/3 | 298 | 7.71 |
| 4UEGRRRA | 1 | False | 0/1 | 5/5 | 1/2/3 | 298 | 7.22 |
| 4R219TNX | 0 | False | 0/1 | 6/6 | 2/3/5 | 368 | 14.17 |
| 4R219TNX | 1 | False | 0/1 | 6/6 | 2/3/5 | 368 | 14.05 |
| 51F2NVWK | 0 | True | 2/0 | 5/5 | 3/1/4 | 235 | 8.54 |
| 51F2NVWK | 1 | True | 2/0 | 5/5 | 3/1/4 | 235 | 7.77 |
| 56QZEVDV | 0 | False | 0/3 | 5/5 | 1/2/3 | 286 | 7.40 |
| 56QZEVDV | 1 | False | 0/3 | 5/5 | 1/2/3 | 286 | 6.95 |
| 5AWWF1M1 | 0 | True | 3/0 | 6/6 | 3/1/4 | 333 | 11.04 |
| 5AWWF1M1 | 1 | True | 3/0 | 6/6 | 3/1/4 | 333 | 10.10 |
| 6H6WNQM2 | 0 | False | 0/3 | 6/6 | 1/3/4 | 318 | 10.40 |
| 6H6WNQM2 | 1 | False | 0/3 | 6/6 | 1/3/4 | 319 | 9.95 |
| 5UIUKHCI | 0 | False | 0/1 | 8/8 | 3/4/7 | 535 | 20.94 |
| 5UIUKHCI | 1 | False | 0/1 | 8/8 | 3/4/7 | 535 | 20.33 |
| 5YVHAEP | 0 | False | 0/1 | 8/8 | 3/4/7 | 547 | 19.29 |
| 5YVHAEP | 1 | False | 0/1 | 8/8 | 3/4/7 | 547 | 18.45 |
| 7I4M53DL | 0 | False | 0/1 | 6/6 | 2/2/4 | 327 | 12.88 |
| 7I4M53DL | 1 | False | 0/1 | 6/6 | 2/2/4 | 334 | 12.84 |
