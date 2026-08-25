# H2H: ev:full+Vleaf vs ev:full

- seeds: 30 (['11111111', '1558AXDL', '15H9Z3IY', '1KV4W6YS', '1MD1YZ9T', '28V7DD4H', '29DAQVG1', '29Y3L4S9', '29ZSW8MY', '2BRGI767', '2CP4KSXZ', '2GHBLJD9', '2H9N3ISZ', '2K9H9HN', '34JCNMPA', '3SZ71111', '41Y71M6E', '46Y8UZEG', '4H2L46CE', '4K8A9QER', '4R219TNX', '4T1SZKLF', '4UEGRRRA', '51F2NVWK', '56QZEVDV', '5AWWF1M1', '5UIUKHCI', '5YVHAEP', '6H6WNQM2', '7I4M53DL'])
- trials: 60 (both seatings per seed); decided 60, undecided 0
- sims=40  checkpoint=None  lives=4  max_steps=4000  deck=b_red  stake=1
- procs=8  wall clock: 337.6s  mean 38.10s/match

## Summary

- **A (ev:full+Vleaf) wins**: 24 / 60  (win rate 0.400, 95% CI [0.283, 0.517])
- **B (ev:full) wins**: 36 / 60
- mean final ante: A 5.73  vs  B 5.73
- mean lives margin (A - B): -0.52
- Nemesis win rate (A's side of every resolved Nemesis): 0.421

## Per-trial

| seed | seating | a_win | lives A/B | ante A/B | nem A/B / total | steps | seconds |
|---|---|---|---|---|---|---|---|
| 29DAQVG1 | 0 | True | 4/0 | 5/5 | 3/0/3 | 246 | 23.97 |
| 29DAQVG1 | 1 | True | 4/0 | 5/5 | 3/0/3 | 248 | 26.75 |
| 1558AXDL | 0 | True | 3/0 | 5/5 | 2/1/3 | 263 | 24.19 |
| 1558AXDL | 1 | True | 1/0 | 6/6 | 2/3/5 | 395 | 39.03 |
| 1MD1YZ9T | 0 | False | 0/3 | 6/6 | 1/3/4 | 314 | 34.25 |
| 1MD1YZ9T | 1 | False | 0/3 | 6/6 | 1/3/4 | 311 | 33.19 |
| 29Y3L4S9 | 0 | True | 1/0 | 6/6 | 1/3/4 | 376 | 40.91 |
| 29Y3L4S9 | 1 | False | 0/3 | 4/4 | 0/3/3 | 253 | 26.72 |
| 1KV4W6YS | 0 | False | 0/4 | 5/5 | 0/4/4 | 368 | 30.02 |
| 1KV4W6YS | 1 | False | 0/2 | 7/7 | 2/3/5 | 539 | 38.92 |
| 28V7DD4H | 0 | False | 0/1 | 7/7 | 2/3/5 | 455 | 52.55 |
| 28V7DD4H | 1 | True | 3/0 | 5/5 | 3/1/4 | 327 | 36.57 |
| 15H9Z3IY | 0 | True | 1/0 | 7/7 | 4/2/6 | 473 | 47.92 |
| 15H9Z3IY | 1 | False | 0/3 | 6/6 | 1/4/5 | 420 | 46.13 |
| 11111111 | 0 | True | 1/0 | 7/7 | 2/3/5 | 475 | 59.72 |
| 11111111 | 1 | False | 0/2 | 6/6 | 1/4/5 | 421 | 54.65 |
| 2BRGI767 | 0 | True | 4/0 | 5/5 | 4/0/4 | 347 | 24.52 |
| 2BRGI767 | 1 | True | 3/0 | 6/6 | 4/1/5 | 421 | 29.81 |
| 2CP4KSXZ | 0 | False | 0/3 | 6/6 | 1/4/5 | 389 | 41.47 |
| 2CP4KSXZ | 1 | False | 0/3 | 6/6 | 1/4/5 | 389 | 41.43 |
| 29ZSW8MY | 0 | True | 2/0 | 6/6 | 3/2/5 | 414 | 45.75 |
| 29ZSW8MY | 1 | False | 0/2 | 7/7 | 2/4/6 | 482 | 56.86 |
| 2GHBLJD9 | 0 | False | 0/2 | 6/6 | 1/3/4 | 384 | 43.29 |
| 2GHBLJD9 | 1 | False | 0/1 | 6/6 | 1/3/4 | 379 | 43.57 |
| 2K9H9HN | 0 | False | 0/3 | 5/5 | 1/2/3 | 267 | 25.96 |
| 2K9H9HN | 1 | False | 0/1 | 7/7 | 2/3/5 | 447 | 39.54 |
| 2H9N3ISZ | 0 | False | 0/2 | 5/5 | 0/4/4 | 329 | 42.77 |
| 2H9N3ISZ | 1 | True | 1/0 | 6/6 | 1/3/4 | 391 | 47.37 |
| 34JCNMPA | 0 | True | 3/0 | 5/5 | 3/0/3 | 285 | 33.35 |
| 34JCNMPA | 1 | False | 0/2 | 6/6 | 1/4/5 | 366 | 48.68 |
| 41Y71M6E | 0 | True | 2/0 | 5/5 | 3/1/4 | 281 | 28.55 |
| 41Y71M6E | 1 | False | 0/3 | 5/5 | 0/4/4 | 306 | 36.92 |
| 3SZ71111 | 0 | True | 4/0 | 5/5 | 4/0/4 | 297 | 38.61 |
| 3SZ71111 | 1 | True | 4/0 | 5/5 | 4/0/4 | 290 | 37.17 |
| 46Y8UZEG | 0 | False | 0/4 | 5/5 | 0/3/3 | 292 | 36.94 |
| 46Y8UZEG | 1 | False | 0/4 | 5/5 | 0/3/3 | 294 | 34.42 |
| 4K8A9QER | 0 | False | 0/4 | 5/5 | 0/4/4 | 275 | 33.63 |
| 4K8A9QER | 1 | False | 0/4 | 5/5 | 0/4/4 | 278 | 36.23 |
| 4R219TNX | 0 | True | 2/0 | 5/5 | 2/2/4 | 302 | 33.41 |
| 4R219TNX | 1 | True | 2/0 | 6/6 | 3/1/4 | 329 | 42.49 |
| 4T1SZKLF | 0 | False | 0/3 | 6/6 | 1/3/4 | 385 | 44.46 |
| 4T1SZKLF | 1 | False | 0/3 | 4/4 | 1/2/3 | 256 | 31.15 |
| 51F2NVWK | 0 | False | 0/1 | 6/6 | 2/2/4 | 264 | 31.83 |
| 51F2NVWK | 1 | False | 0/3 | 5/5 | 0/3/3 | 213 | 23.67 |
| 4H2L46CE | 0 | False | 0/3 | 6/6 | 1/4/5 | 338 | 46.90 |
| 4H2L46CE | 1 | False | 0/2 | 6/6 | 2/3/5 | 356 | 42.29 |
| 4UEGRRRA | 0 | True | 3/0 | 5/5 | 3/1/4 | 337 | 33.73 |
| 4UEGRRRA | 1 | True | 4/0 | 5/5 | 3/0/3 | 294 | 33.59 |
| 56QZEVDV | 0 | True | 2/0 | 5/5 | 2/1/3 | 270 | 26.16 |
| 56QZEVDV | 1 | True | 2/0 | 6/6 | 3/1/4 | 288 | 29.56 |
| 5AWWF1M1 | 0 | True | 2/0 | 6/6 | 3/1/4 | 343 | 34.81 |
| 5AWWF1M1 | 1 | False | 0/4 | 5/5 | 0/3/3 | 287 | 24.25 |
| 7I4M53DL | 0 | False | 0/2 | 5/5 | 1/2/3 | 253 | 26.97 |
| 7I4M53DL | 1 | True | 1/0 | 6/6 | 2/2/4 | 325 | 34.03 |
| 6H6WNQM2 | 0 | False | 0/3 | 6/6 | 1/3/4 | 357 | 38.79 |
| 6H6WNQM2 | 1 | False | 0/1 | 7/7 | 2/3/5 | 448 | 42.59 |
| 5UIUKHCI | 0 | False | 0/1 | 8/8 | 3/4/7 | 539 | 57.56 |
| 5UIUKHCI | 1 | False | 0/3 | 6/6 | 1/3/4 | 371 | 41.83 |
| 5YVHAEP | 0 | False | 0/1 | 8/8 | 3/3/6 | 547 | 54.91 |
| 5YVHAEP | 1 | False | 0/1 | 7/7 | 3/3/6 | 518 | 48.51 |
