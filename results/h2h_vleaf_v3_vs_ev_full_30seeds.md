# H2H: ev:full+Vleaf vs ev:full

- seeds: 30 (['11111111', '1558AXDL', '15H9Z3IY', '1KV4W6YS', '1MD1YZ9T', '28V7DD4H', '29DAQVG1', '29Y3L4S9', '29ZSW8MY', '2BRGI767', '2CP4KSXZ', '2GHBLJD9', '2H9N3ISZ', '2K9H9HN', '34JCNMPA', '3SZ71111', '41Y71M6E', '46Y8UZEG', '4H2L46CE', '4K8A9QER', '4R219TNX', '4T1SZKLF', '4UEGRRRA', '51F2NVWK', '56QZEVDV', '5AWWF1M1', '5UIUKHCI', '5YVHAEP', '6H6WNQM2', '7I4M53DL'])
- trials: 60 (both seatings per seed); decided 60, undecided 0
- sims=40  checkpoint=ev\runs\v_v3\s1\ckpt_0002000.pt  lives=4  max_steps=4000  deck=b_red  stake=1
- procs=8  wall clock: 321.0s  mean 37.86s/match

## Summary

- **A (ev:full+Vleaf) wins**: 29 / 60  (win rate 0.483, 95% CI [0.350, 0.600])
- **B (ev:full) wins**: 31 / 60
- mean final ante: A 5.88  vs  B 5.88
- mean lives margin (A - B): -0.15
- Nemesis win rate (A's side of every resolved Nemesis): 0.506

## Per-trial

| seed | seating | a_win | lives A/B | ante A/B | nem A/B / total | steps | seconds |
|---|---|---|---|---|---|---|---|
| 1MD1YZ9T | 0 | False | 0/4 | 5/5 | 0/3/3 | 237 | 24.17 |
| 1MD1YZ9T | 1 | False | 0/4 | 5/5 | 0/3/3 | 277 | 28.15 |
| 29Y3L4S9 | 0 | True | 1/0 | 6/6 | 2/2/4 | 353 | 34.58 |
| 29Y3L4S9 | 1 | False | 0/3 | 4/4 | 0/3/3 | 246 | 24.09 |
| 11111111 | 0 | False | 0/3 | 5/5 | 1/2/3 | 321 | 30.93 |
| 11111111 | 1 | False | 0/3 | 5/5 | 1/2/3 | 318 | 30.82 |
| 1558AXDL | 0 | False | 0/3 | 6/6 | 1/3/4 | 331 | 30.84 |
| 1558AXDL | 1 | True | 3/0 | 6/6 | 4/1/5 | 407 | 38.78 |
| 1KV4W6YS | 0 | False | 0/3 | 6/6 | 1/4/5 | 457 | 36.61 |
| 1KV4W6YS | 1 | False | 0/1 | 6/6 | 3/2/5 | 460 | 41.64 |
| 29DAQVG1 | 0 | True | 1/0 | 6/6 | 2/2/4 | 331 | 36.00 |
| 29DAQVG1 | 1 | False | 0/1 | 6/6 | 2/3/5 | 350 | 44.51 |
| 28V7DD4H | 0 | True | 2/0 | 6/6 | 3/2/5 | 395 | 44.06 |
| 28V7DD4H | 1 | True | 2/0 | 5/5 | 4/0/4 | 318 | 37.81 |
| 15H9Z3IY | 0 | False | 0/3 | 6/6 | 1/4/5 | 385 | 39.33 |
| 15H9Z3IY | 1 | False | 0/2 | 7/7 | 2/3/5 | 458 | 51.27 |
| 2BRGI767 | 0 | True | 1/0 | 7/7 | 3/2/5 | 475 | 33.59 |
| 2BRGI767 | 1 | True | 4/0 | 5/5 | 4/0/4 | 350 | 24.03 |
| 29ZSW8MY | 0 | True | 2/0 | 6/6 | 3/2/5 | 390 | 41.80 |
| 29ZSW8MY | 1 | True | 2/0 | 6/6 | 3/2/5 | 396 | 44.30 |
| 2GHBLJD9 | 0 | True | 2/0 | 6/6 | 4/1/5 | 390 | 46.06 |
| 2GHBLJD9 | 1 | True | 4/0 | 5/5 | 4/0/4 | 309 | 32.52 |
| 34JCNMPA | 0 | False | 0/1 | 6/6 | 2/2/4 | 329 | 39.44 |
| 34JCNMPA | 1 | True | 2/0 | 6/6 | 3/2/5 | 337 | 34.22 |
| 3SZ71111 | 0 | False | 0/4 | 4/4 | 0/3/3 | 237 | 32.12 |
| 3SZ71111 | 1 | True | 1/0 | 6/6 | 2/2/4 | 356 | 46.31 |
| 2K9H9HN | 0 | False | 0/1 | 7/7 | 3/2/5 | 398 | 41.22 |
| 2K9H9HN | 1 | False | 0/1 | 7/7 | 3/2/5 | 462 | 49.97 |
| 2CP4KSXZ | 0 | True | 2/0 | 7/7 | 4/2/6 | 486 | 49.96 |
| 2CP4KSXZ | 1 | False | 0/1 | 8/8 | 3/4/7 | 590 | 65.33 |
| 41Y71M6E | 0 | False | 0/3 | 6/6 | 1/3/4 | 316 | 29.14 |
| 41Y71M6E | 1 | True | 1/0 | 6/6 | 3/1/4 | 337 | 35.41 |
| 2H9N3ISZ | 0 | False | 0/1 | 7/7 | 3/2/5 | 468 | 60.71 |
| 2H9N3ISZ | 1 | True | 1/0 | 7/7 | 3/2/5 | 443 | 57.22 |
| 46Y8UZEG | 0 | False | 0/3 | 6/6 | 1/3/4 | 367 | 42.80 |
| 46Y8UZEG | 1 | True | 1/0 | 6/6 | 2/2/4 | 399 | 39.80 |
| 4H2L46CE | 0 | False | 0/1 | 7/7 | 2/3/5 | 361 | 42.97 |
| 4H2L46CE | 1 | False | 0/3 | 6/6 | 1/4/5 | 319 | 38.98 |
| 51F2NVWK | 0 | False | 0/1 | 6/6 | 2/2/4 | 260 | 27.85 |
| 51F2NVWK | 1 | False | 0/2 | 5/5 | 1/2/3 | 227 | 21.92 |
| 4UEGRRRA | 0 | True | 3/0 | 5/5 | 2/1/3 | 274 | 30.55 |
| 4UEGRRRA | 1 | True | 3/0 | 5/5 | 2/1/3 | 272 | 25.20 |
| 4R219TNX | 0 | False | 0/3 | 5/5 | 0/3/3 | 269 | 28.67 |
| 4R219TNX | 1 | False | 0/1 | 6/6 | 1/3/4 | 330 | 37.76 |
| 4T1SZKLF | 0 | True | 2/0 | 6/6 | 3/1/4 | 366 | 38.21 |
| 4T1SZKLF | 1 | False | 0/2 | 6/6 | 2/2/4 | 327 | 34.18 |
| 56QZEVDV | 0 | True | 4/0 | 5/5 | 4/0/4 | 271 | 27.47 |
| 56QZEVDV | 1 | True | 2/0 | 5/5 | 2/1/3 | 280 | 23.27 |
| 4K8A9QER | 0 | True | 2/0 | 7/7 | 4/2/6 | 414 | 53.06 |
| 4K8A9QER | 1 | True | 1/0 | 8/8 | 3/3/6 | 456 | 57.46 |
| 7I4M53DL | 0 | True | 4/0 | 5/5 | 3/0/3 | 222 | 24.91 |
| 7I4M53DL | 1 | True | 4/0 | 5/5 | 3/0/3 | 220 | 23.05 |
| 5AWWF1M1 | 0 | False | 0/3 | 5/5 | 1/3/4 | 300 | 32.22 |
| 5AWWF1M1 | 1 | False | 0/2 | 5/5 | 1/3/4 | 338 | 33.23 |
| 5YVHAEP | 0 | False | 0/2 | 6/6 | 2/3/5 | 410 | 43.91 |
| 5YVHAEP | 1 | False | 0/3 | 6/6 | 1/3/4 | 399 | 34.51 |
| 6H6WNQM2 | 0 | True | 2/0 | 6/6 | 3/2/5 | 383 | 41.23 |
| 6H6WNQM2 | 1 | True | 1/0 | 7/7 | 2/3/5 | 430 | 43.50 |
| 5UIUKHCI | 0 | True | 2/0 | 6/6 | 3/1/4 | 389 | 47.27 |
| 5UIUKHCI | 1 | False | 0/3 | 6/6 | 1/3/4 | 371 | 40.58 |
