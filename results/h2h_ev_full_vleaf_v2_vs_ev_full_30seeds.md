# H2H: ev:full+Vleaf vs ev:full

- seeds: 30 (['11111111', '1558AXDL', '15H9Z3IY', '1KV4W6YS', '1MD1YZ9T', '28V7DD4H', '29DAQVG1', '29Y3L4S9', '29ZSW8MY', '2BRGI767', '2CP4KSXZ', '2GHBLJD9', '2H9N3ISZ', '2K9H9HN', '34JCNMPA', '3SZ71111', '41Y71M6E', '46Y8UZEG', '4H2L46CE', '4K8A9QER', '4R219TNX', '4T1SZKLF', '4UEGRRRA', '51F2NVWK', '56QZEVDV', '5AWWF1M1', '5UIUKHCI', '5YVHAEP', '6H6WNQM2', '7I4M53DL'])
- trials: 60 (both seatings per seed); decided 60, undecided 0
- sims=40  checkpoint=mp/ev/runs/v_v2/ckpt_0002000.pt  lives=4  max_steps=4000  deck=b_red  stake=1
- procs=8  wall clock: 263.5s  mean 30.39s/match

## Summary

- **A (ev:full+Vleaf) wins**: 24 / 60  (win rate 0.400, 95% CI [0.283, 0.533])
- **B (ev:full) wins**: 36 / 60
- mean final ante: A 5.85  vs  B 5.85
- mean lives margin (A - B): -0.48
- Nemesis win rate (A's side of every resolved Nemesis): 0.455

## Per-trial

| seed | seating | a_win | lives A/B | ante A/B | nem A/B / total | steps | seconds |
|---|---|---|---|---|---|---|---|
| 1KV4W6YS | 0 | False | 0/2 | 6/6 | 2/3/5 | 472 | 20.91 |
| 1KV4W6YS | 1 | False | 0/4 | 5/5 | 0/4/4 | 368 | 18.62 |
| 1558AXDL | 0 | True | 3/0 | 6/6 | 3/1/4 | 359 | 20.38 |
| 1558AXDL | 1 | False | 0/2 | 6/6 | 1/3/4 | 350 | 19.97 |
| 29Y3L4S9 | 0 | False | 0/1 | 6/6 | 2/3/5 | 467 | 29.35 |
| 29Y3L4S9 | 1 | False | 0/3 | 4/4 | 0/3/3 | 248 | 15.26 |
| 1MD1YZ9T | 0 | False | 0/2 | 6/6 | 1/3/4 | 363 | 24.01 |
| 1MD1YZ9T | 1 | True | 1/0 | 6/6 | 2/2/4 | 366 | 25.98 |
| 29DAQVG1 | 0 | False | 0/2 | 7/7 | 2/3/5 | 393 | 30.25 |
| 29DAQVG1 | 1 | True | 3/0 | 6/6 | 4/1/5 | 319 | 23.08 |
| 15H9Z3IY | 0 | False | 0/3 | 6/6 | 1/3/4 | 376 | 25.68 |
| 15H9Z3IY | 1 | True | 2/0 | 6/6 | 4/1/5 | 394 | 28.07 |
| 11111111 | 0 | True | 1/0 | 6/6 | 3/2/5 | 436 | 31.95 |
| 11111111 | 1 | True | 2/0 | 6/6 | 3/2/5 | 443 | 32.01 |
| 28V7DD4H | 0 | False | 0/2 | 6/6 | 2/3/5 | 413 | 31.55 |
| 28V7DD4H | 1 | True | 1/0 | 7/7 | 4/2/6 | 514 | 36.85 |
| 2BRGI767 | 0 | False | 0/1 | 7/7 | 3/2/5 | 470 | 18.63 |
| 2BRGI767 | 1 | True | 2/0 | 6/6 | 3/1/4 | 395 | 17.13 |
| 29ZSW8MY | 0 | True | 3/0 | 5/5 | 3/1/4 | 338 | 22.70 |
| 29ZSW8MY | 1 | False | 0/2 | 5/5 | 1/3/4 | 358 | 24.92 |
| 2K9H9HN | 0 | False | 0/3 | 4/4 | 0/3/3 | 224 | 13.28 |
| 2K9H9HN | 1 | False | 0/1 | 7/7 | 2/3/5 | 415 | 21.20 |
| 2GHBLJD9 | 0 | False | 0/1 | 6/6 | 2/3/5 | 390 | 27.29 |
| 2GHBLJD9 | 1 | False | 0/1 | 6/6 | 2/2/4 | 359 | 24.13 |
| 2CP4KSXZ | 0 | True | 2/0 | 7/7 | 4/2/6 | 468 | 30.86 |
| 2CP4KSXZ | 1 | False | 0/2 | 7/7 | 2/4/6 | 473 | 38.47 |
| 41Y71M6E | 0 | False | 0/3 | 6/6 | 1/3/4 | 320 | 18.02 |
| 41Y71M6E | 1 | False | 0/1 | 6/6 | 3/1/4 | 342 | 33.23 |
| 2H9N3ISZ | 0 | True | 2/0 | 6/6 | 4/1/5 | 414 | 30.41 |
| 2H9N3ISZ | 1 | False | 0/1 | 7/7 | 2/3/5 | 456 | 46.30 |
| 3SZ71111 | 0 | True | 3/0 | 6/6 | 3/1/4 | 351 | 28.45 |
| 3SZ71111 | 1 | False | 0/4 | 5/5 | 0/3/3 | 293 | 34.17 |
| 34JCNMPA | 0 | True | 4/0 | 5/5 | 3/0/3 | 284 | 23.49 |
| 34JCNMPA | 1 | False | 0/1 | 7/7 | 2/4/6 | 468 | 52.68 |
| 4H2L46CE | 0 | False | 0/3 | 5/5 | 0/4/4 | 262 | 22.03 |
| 4H2L46CE | 1 | False | 0/1 | 7/7 | 2/4/6 | 370 | 39.47 |
| 46Y8UZEG | 0 | True | 1/0 | 6/6 | 3/2/5 | 381 | 36.01 |
| 46Y8UZEG | 1 | False | 0/2 | 5/5 | 1/2/3 | 303 | 32.25 |
| 4R219TNX | 0 | True | 3/0 | 5/5 | 2/1/3 | 254 | 25.73 |
| 4R219TNX | 1 | True | 2/0 | 5/5 | 1/2/3 | 297 | 28.96 |
| 4K8A9QER | 0 | False | 0/4 | 5/5 | 0/4/4 | 279 | 31.07 |
| 4K8A9QER | 1 | True | 3/0 | 6/6 | 4/1/5 | 338 | 38.07 |
| 51F2NVWK | 0 | True | 1/0 | 6/6 | 2/2/4 | 264 | 26.29 |
| 51F2NVWK | 1 | False | 0/2 | 5/5 | 1/2/3 | 229 | 22.30 |
| 56QZEVDV | 0 | True | 3/0 | 5/5 | 4/0/4 | 273 | 27.54 |
| 56QZEVDV | 1 | True | 1/0 | 6/6 | 3/1/4 | 322 | 29.13 |
| 4T1SZKLF | 0 | False | 0/2 | 6/6 | 2/3/5 | 378 | 38.26 |
| 4T1SZKLF | 1 | False | 0/4 | 5/5 | 0/4/4 | 318 | 31.45 |
| 5AWWF1M1 | 0 | False | 0/3 | 4/4 | 1/2/3 | 232 | 20.50 |
| 5AWWF1M1 | 1 | False | 0/3 | 5/5 | 0/4/4 | 313 | 30.65 |
| 4UEGRRRA | 0 | True | 3/0 | 6/6 | 3/1/4 | 337 | 37.54 |
| 4UEGRRRA | 1 | True | 1/0 | 6/6 | 2/2/4 | 354 | 35.42 |
| 7I4M53DL | 0 | True | 1/0 | 5/5 | 3/1/4 | 297 | 37.76 |
| 7I4M53DL | 1 | False | 0/1 | 6/6 | 2/2/4 | 351 | 36.30 |
| 5UIUKHCI | 0 | False | 0/3 | 6/6 | 1/4/5 | 398 | 53.14 |
| 5UIUKHCI | 1 | False | 0/2 | 6/6 | 1/4/5 | 394 | 46.97 |
| 6H6WNQM2 | 0 | True | 1/0 | 7/7 | 2/3/5 | 429 | 42.69 |
| 6H6WNQM2 | 1 | False | 0/2 | 7/7 | 2/3/5 | 402 | 42.30 |
| 5YVHAEP | 0 | False | 0/3 | 5/5 | 1/3/4 | 337 | 38.72 |
| 5YVHAEP | 1 | False | 0/1 | 8/8 | 3/4/7 | 548 | 53.27 |
