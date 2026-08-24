# H2H: ev:full+stats vs ev:full

- seeds: 30 (['11111111', '1558AXDL', '15H9Z3IY', '1KV4W6YS', '1MD1YZ9T', '28V7DD4H', '29DAQVG1', '29Y3L4S9', '29ZSW8MY', '2BRGI767', '2CP4KSXZ', '2GHBLJD9', '2H9N3ISZ', '2K9H9HN', '34JCNMPA', '3SZ71111', '41Y71M6E', '46Y8UZEG', '4H2L46CE', '4K8A9QER', '4R219TNX', '4T1SZKLF', '4UEGRRRA', '51F2NVWK', '56QZEVDV', '5AWWF1M1', '5UIUKHCI', '5YVHAEP', '6H6WNQM2', '7I4M53DL'])
- trials: 60 (both seatings per seed); decided 60, undecided 0
- sims=40  checkpoint=None  lives=4  max_steps=100000  deck=b_red  stake=1
- procs=4  wall clock: 208.3s  mean 13.42s/match

## Summary

- **A (ev:full+stats) wins**: 8 / 60  (win rate 0.133, 95% CI [0.050, 0.217])
- **B (ev:full) wins**: 52 / 60
- mean final ante: A 4.20  vs  B 4.20
- mean lives margin (A - B): -2.87
- Nemesis win rate (A's side of every resolved Nemesis): 0.231

## Per-trial

| seed | seating | a_win | lives A/B | ante A/B | nem A/B / total | steps | seconds |
|---|---|---|---|---|---|---|---|
| 15H9Z3IY | 0 | False | 0/4 | 2/2 | 0/1/1 | 133 | 6.92 |
| 15H9Z3IY | 1 | False | 0/4 | 2/2 | 0/1/1 | 133 | 6.92 |
| 1KV4W6YS | 0 | False | 0/4 | 4/4 | 0/3/3 | 277 | 11.93 |
| 1KV4W6YS | 1 | False | 0/4 | 4/4 | 0/3/3 | 277 | 12.17 |
| 1558AXDL | 0 | False | 0/3 | 4/4 | 0/3/3 | 244 | 12.35 |
| 1558AXDL | 1 | False | 0/3 | 4/4 | 0/3/3 | 244 | 12.45 |
| 1MD1YZ9T | 0 | False | 0/4 | 3/3 | 0/1/1 | 165 | 6.96 |
| 1MD1YZ9T | 1 | False | 0/4 | 3/3 | 0/1/1 | 165 | 6.46 |
| 29DAQVG1 | 0 | True | 4/0 | 5/5 | 3/0/3 | 249 | 13.31 |
| 29DAQVG1 | 1 | True | 4/0 | 5/5 | 3/0/3 | 249 | 13.39 |
| 11111111 | 0 | True | 1/0 | 6/6 | 3/2/5 | 422 | 27.58 |
| 11111111 | 1 | True | 1/0 | 6/6 | 3/2/5 | 422 | 28.25 |
| 28V7DD4H | 0 | False | 0/3 | 5/5 | 0/4/4 | 311 | 20.63 |
| 28V7DD4H | 1 | False | 0/3 | 5/5 | 0/4/4 | 311 | 19.38 |
| 2BRGI767 | 0 | False | 0/4 | 3/3 | 0/1/1 | 150 | 5.45 |
| 2BRGI767 | 1 | False | 0/4 | 3/3 | 0/1/1 | 150 | 5.17 |
| 29Y3L4S9 | 0 | False | 0/1 | 6/6 | 2/2/4 | 410 | 23.08 |
| 29Y3L4S9 | 1 | False | 0/1 | 6/6 | 2/2/4 | 410 | 22.60 |
| 29ZSW8MY | 0 | False | 0/4 | 3/3 | 0/2/2 | 179 | 11.27 |
| 29ZSW8MY | 1 | False | 0/4 | 3/3 | 0/2/2 | 186 | 12.00 |
| 2H9N3ISZ | 0 | False | 0/4 | 4/4 | 0/2/2 | 218 | 9.11 |
| 2H9N3ISZ | 1 | False | 0/4 | 4/4 | 0/2/2 | 218 | 8.80 |
| 2K9H9HN | 0 | False | 0/4 | 5/5 | 0/3/3 | 279 | 14.76 |
| 2K9H9HN | 1 | False | 0/4 | 5/5 | 0/3/3 | 279 | 14.29 |
| 2GHBLJD9 | 0 | False | 0/2 | 6/6 | 1/3/4 | 395 | 22.18 |
| 2GHBLJD9 | 1 | False | 0/2 | 6/6 | 1/3/4 | 395 | 22.12 |
| 34JCNMPA | 0 | False | 0/4 | 3/3 | 0/2/2 | 175 | 12.58 |
| 34JCNMPA | 1 | False | 0/4 | 3/3 | 0/2/2 | 175 | 12.34 |
| 41Y71M6E | 0 | False | 0/4 | 3/3 | 0/2/2 | 194 | 8.29 |
| 41Y71M6E | 1 | False | 0/4 | 3/3 | 0/2/2 | 194 | 8.19 |
| 46Y8UZEG | 0 | False | 0/4 | 3/3 | 0/2/2 | 176 | 8.71 |
| 46Y8UZEG | 1 | False | 0/4 | 3/3 | 0/2/2 | 176 | 8.32 |
| 2CP4KSXZ | 0 | True | 1/0 | 8/8 | 4/3/7 | 566 | 35.89 |
| 2CP4KSXZ | 1 | True | 1/0 | 8/8 | 4/3/7 | 566 | 35.42 |
| 4K8A9QER | 0 | False | 0/4 | 3/3 | 0/1/1 | 152 | 6.85 |
| 4K8A9QER | 1 | False | 0/4 | 3/3 | 0/1/1 | 152 | 6.51 |
| 3SZ71111 | 0 | False | 0/2 | 5/5 | 2/2/4 | 319 | 21.69 |
| 3SZ71111 | 1 | False | 0/2 | 5/5 | 2/2/4 | 319 | 21.18 |
| 4H2L46CE | 0 | False | 0/4 | 4/4 | 0/2/2 | 209 | 12.25 |
| 4H2L46CE | 1 | False | 0/4 | 4/4 | 0/2/2 | 209 | 11.80 |
| 4UEGRRRA | 0 | False | 0/4 | 4/4 | 0/2/2 | 223 | 9.82 |
| 4UEGRRRA | 1 | False | 0/4 | 4/4 | 0/2/2 | 223 | 9.70 |
| 4T1SZKLF | 0 | False | 0/4 | 4/4 | 0/2/2 | 219 | 10.11 |
| 4T1SZKLF | 1 | False | 0/4 | 4/4 | 0/2/2 | 219 | 10.09 |
| 51F2NVWK | 0 | False | 0/3 | 5/5 | 0/3/3 | 207 | 11.27 |
| 51F2NVWK | 1 | False | 0/3 | 5/5 | 0/3/3 | 207 | 10.98 |
| 56QZEVDV | 0 | False | 0/4 | 4/4 | 0/2/2 | 202 | 9.51 |
| 56QZEVDV | 1 | False | 0/4 | 4/4 | 0/2/2 | 202 | 9.39 |
| 5AWWF1M1 | 0 | False | 0/4 | 4/4 | 0/2/2 | 232 | 10.35 |
| 5AWWF1M1 | 1 | False | 0/4 | 4/4 | 0/2/2 | 232 | 10.57 |
| 4R219TNX | 0 | True | 2/0 | 7/7 | 3/2/5 | 389 | 26.41 |
| 4R219TNX | 1 | True | 2/0 | 7/7 | 3/2/5 | 389 | 26.14 |
| 7I4M53DL | 0 | False | 0/4 | 3/3 | 0/1/1 | 131 | 4.92 |
| 7I4M53DL | 1 | False | 0/4 | 3/3 | 0/1/1 | 131 | 4.82 |
| 5YVHAEP | 0 | False | 0/4 | 3/3 | 0/1/1 | 174 | 7.31 |
| 5YVHAEP | 1 | False | 0/4 | 3/3 | 0/1/1 | 174 | 7.20 |
| 5UIUKHCI | 0 | False | 0/4 | 4/4 | 0/2/2 | 233 | 14.53 |
| 5UIUKHCI | 1 | False | 0/4 | 4/4 | 0/2/2 | 233 | 14.09 |
| 6H6WNQM2 | 0 | False | 0/4 | 3/3 | 0/2/2 | 193 | 9.31 |
| 6H6WNQM2 | 1 | False | 0/4 | 3/3 | 0/2/2 | 193 | 9.31 |
