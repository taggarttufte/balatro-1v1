# EV player gate 2 — 2026-08-26

`python ev/gate_ev_player.py --procs 12`

Seeds: 126 of 126 ground-truth seeds (offset 0); vanilla ruleset, Red deck, White stake; 12 processes; wall 60 s.

## Outcomes (bootstrap 95% CI)

| metric | fast | full | greedy |
|---|---|---|---|
| ante1_clear | 95.2% [91.3, 98.4] | 96.0% [92.1, 99.2] | 31.7% [23.8, 39.7] |
| ante2_clear | 81.7% [74.6, 88.1] | 87.3% [81.0, 92.9] | 0.0% [0.0, 0.0] |
| ante3_clear | 77.0% [69.0, 84.1] | 80.2% [73.0, 86.5] | 0.0% [0.0, 0.0] |
| ante4_clear | 60.3% [51.6, 69.0] | 68.3% [59.5, 76.2] | 0.0% [0.0, 0.0] |
| won | 3.2% [0.8, 6.3] | 3.2% [0.8, 6.3] | 0.0% [0.0, 0.0] |
| mean_final_ante | 4.78 [4.43, 5.13] | 4.98 [4.66, 5.30] | 1.32 [1.24, 1.40] |
| mean_blinds_cleared | 9.74 [8.98, 10.53] | 10.31 [9.60, 11.02] | 2.16 [2.01, 2.31] |
| money_at_ante3 | 20.41 [18.94, 21.96] | 20.29 [18.89, 21.85] | nan [nan, nan] |
| hands unused / cleared blind | 2.13 | 2.13 | 1.25 |
| $ at ante 3: n reaching | 103 | 110 | 0 |

## Paired-by-seed deltas vs greedy (mean difference, bootstrap 95% CI)

- **fast − greedy** (n=126): ante-1 clear 63.5% [54.8, 72.2]; final ante 3.46 [3.10, 3.82]; blinds cleared 7.58 [6.78, 8.41]; final $ 5.36 [2.90, 7.94]
- **full − greedy** (n=126): ante-1 clear 64.3% [56.3, 72.2]; final ante 3.67 [3.33, 3.99]; blinds cleared 8.15 [7.44, 8.90]; final $ 6.44 [4.11, 9.04]

## Wall-clock per decision (ms)

| player | hand mean | hand p50 | hand p95 | hand max | n | shop mean | shop p95 | n |
|---|---|---|---|---|---|---|---|---|
| fast | 3.90 | 2.50 | 12.49 | 70.5 | 5433 | 6.1 | 21.5 | 5728 |
| full | 77.97 | 62.60 | 192.76 | 826.3 | 5504 | 8.0 | 26.6 | 6063 |
| greedy | 2.44 | 2.56 | 2.95 | 3.9 | 1750 | 0.1 | 0.1 | 272 |

Budgets: fast ≤ 5 ms mean, full ≤ 100 ms mean per SELECTING_HAND decision.

## Draw-order invariance

- fast: 743 sampled states, `game.deck` permuted → identical decision in 743 (0 mismatches)
- full: 741 sampled states, `game.deck` permuted → identical decision in 741 (0 mismatches)

## MLB match (EVPlayer fast vs EVPlayer fast, full MLBMatch)

- seed 11111111: done=True winner=1 final antes [8, 8] lives [0, 1] steps 681 in 3.0 s; Nemeses (ante, loser, s0, s1): [(2, None, 7224, 7224), (3, None, 16344, 16344), (4, 0, 13560, 15424), (5, 0, 14001, 19776), (6, 1, 54400, 41664), (7, 1, 67512, 54084), (8, 0, 5340, 11455)]

## Per-seed (furthest ante/blind, $)

| seed | fast | full | greedy |
|---|---|---|---|
| 11111111 | 6 Boss $26 | 6 Boss $26 | 2 Boss $21 |
| 1558AXDL | 6 Boss $27 | 8 Boss $28 | 2 Boss $22 |
| 15H9Z3IY | 4 Boss $24 | 2 Boss $11 | 1 Big $8 |
| 1KV4W6YS | 5 Boss $26 | 6 Boss $28 | 2 Boss $25 |
| 1MD1YZ9T | 2 Boss $6 | 3 Boss $17 | 2 Boss $33 |
| 28V7DD4H | 7 Boss $25 | 7 Boss $32 | 1 Big $9 |
| 29DAQVG1 | 5 Boss $29 | 5 Boss $27 | 1 Big $9 |
| 29Y3L4S9 | 3 Boss $17 | 6 Boss $3 | 2 Boss $25 |
| 29ZSW8MY | 2 Boss $13 | 2 Boss $14 | 2 Boss $26 |
| 2BRGI767 | 5 Boss $26 | 6 Boss $29 | 1 Boss $15 |
| 2CP4KSXZ | 9 Boss $20 | 7 Boss $26 | 1 Boss $15 |
| 2GHBLJD9 | 5 Boss $13 | 4 Boss $24 | 1 Boss $15 |
| 2H9N3ISZ | 2 Boss $12 | 5 Boss $47 | 1 Boss $15 |
| 2K9H9HN | 4 Boss $20 | 6 Boss $0 | 1 Boss $15 |
| 34JCNMPA | 5 Boss $25 | 7 Boss $25 | 2 Boss $24 |
| 3SZ71111 | 5 Boss $77 | 4 Boss $22 | 2 Boss $24 |
| 41Y71M6E | 7 Boss $25 | 7 Boss $13 | 1 Boss $15 |
| 46Y8UZEG | 2 Boss $6 | 2 Boss $4 | 2 Boss $21 |
| 4H2L46CE | 1 Boss $6 | 5 Boss $28 | 1 Big $9 |
| 4K8A9QER | 4 Boss $19 | 5 Boss $27 | 2 Boss $24 |
| 4R219TNX | 3 Boss $15 | 3 Boss $15 | 1 Boss $15 |
| 4T1SZKLF | 5 Boss $28 | 1 Boss $8 | 1 Boss $14 |
| 4UEGRRRA | 4 Boss $8 | 6 Boss $24 | 1 Big $10 |
| 51F2NVWK | 4 Boss $17 | 5 Boss $26 | 1 Boss $15 |
| 56QZEVDV | 4 Boss $25 | 5 Boss $25 | 1 Boss $15 |
| 5AWWF1M1 | 6 Boss $26 | 6 Boss $20 | 1 Boss $16 |
| 5UIUKHCI | 2 Boss $11 | 2 Boss $11 | 1 Boss $15 |
| 5YVHAEP | 4 Boss $23 | 4 Boss $22 | 1 Boss $14 |
| 6H6WNQM2 | 6 Boss $26 | 5 Boss $8 | 2 Boss $21 |
| 7I4M53DL | 6 Boss $28 | 5 Boss $25 | 1 Big $9 |
| 7LB2WVPK | 2 Boss $7 | 2 Boss $10 | 1 Boss $16 |
| 7ODNKXP | 2 Boss $3 | 5 Boss $54 | 2 Boss $21 |
| 7PEE6NA5 | 5 Boss $27 | 6 Boss $9 | 1 Big $9 |
| 7UNQV1C9 | 6 Boss $26 | 6 Boss $32 | 2 Boss $21 |
| 7YTVQERM | 5 Boss $25 | 6 Boss $23 | 1 Small $4 |
| 8I5QXNGE | 5 Boss $29 | 5 Boss $26 | 1 Boss $15 |
| 8IZ7HHE4 | 5 Boss $26 | 6 Boss $31 | 1 Boss $15 |
| 8Q47WV6K | 7 Boss $26 | 7 Boss $3 | 1 Boss $15 |
| 8QBRTPD | 1 Boss $7 | 1 Boss $7 | 1 Boss $14 |
| 8SIYIK9C | 5 Boss $46 | 6 Boss $46 | 1 Big $9 |
| 8UU564MF | 2 Boss $5 | 5 Boss $15 | 1 Boss $14 |
| 8Y7ABZ7C | 4 Boss $14 | 4 Boss $29 | 2 Boss $21 |
| 93W4UHC4 | 4 Boss $24 | 5 Boss $49 | 1 Boss $14 |
| 967889YL | 5 Boss $25 | 7 Boss $31 | 2 Boss $21 |
| 9Q9HQXZG | 1 Boss $7 | 6 Boss $27 | 2 Boss $31 |
| 9QV2ZT9X | 6 Boss $43 | 6 Boss $40 | 1 Boss $16 |
| 9VIGPRT6 | 2 Boss $13 | 3 Boss $16 | 2 Boss $31 |
| 9ZXMM1M | 2 Boss $11 | 2 Boss $11 | 2 Boss $24 |
| A | 6 Boss $17 | 5 Boss $32 | 1 Boss $14 |
| A1UAS1G5 | 6 Boss $26 | 5 Boss $27 | 1 Boss $14 |
| AE1K7DG1 | 5 Boss $20 | 4 Boss $23 | 2 Boss $20 |
| ALEEB | 7 Boss $30 | 7 Boss $25 | 2 Boss $26 |
| AMPYZUIU | 7 Boss $17 | 7 Boss $20 | 1 Boss $16 |
| APH9LX2Y | 8 Boss $29 | 7 Boss $27 | 2 Boss $33 |
| BA1D48RK | 5 Boss $17 | 5 Boss $25 | 1 Big $8 |
| BQ6A2D42 | 2 Boss $4 | 8 Boss $17 | 1 Big $9 |
| C19P8BQ4 | 2 Boss $10 | 2 Boss $12 | 2 Boss $21 |
| C1ZLPKAA | 4 Boss $20 | 4 Boss $22 | 1 Boss $15 |
| C93K3YWJ | 1 Boss $8 | 1 Boss $9 | 1 Boss $16 |
| CCBWC6CR | 8 Boss $28 | 5 Boss $28 | 1 Big $9 |
| CHPB293X | 3 Boss $15 | 3 Boss $15 | 2 Boss $22 |
| CIGHYTAU | 6 Boss $30 | 5 Boss $27 | 1 Big $10 |
| CMEEXU8I | 5 Boss $25 | 5 Boss $25 | 2 Boss $23 |
| D4TDD2B6 | 4 Boss $22 | 5 Boss $27 | 1 Big $9 |
| D7Y5419A | 2 Boss $7 | 2 Boss $10 | 1 Boss $15 |
| ECG8NVDZ | 4 Boss $13 | 4 Boss $13 | 2 Boss $27 |
| EGE5ZY77 | 6 Boss $25 | 6 Boss $24 | 1 Boss $16 |
| EVB8882H | 7 Boss $11 | 7 Boss $15 | 1 Big $7 |
| FSXYRB3J | 4 Boss $18 | 5 Boss $25 | 1 Big $10 |
| GVQL737K | 8 Boss $20 | 9 Boss $19 | 1 Boss $13 |
| GVYT2DGJ | 4 Boss $7 | 5 Boss $30 | 1 Boss $17 |
| H1EWXLNE | 9 Boss $43 | 7 Boss $26 | 1 Boss $15 |
| HPR8Q7K | 5 Boss $25 | 5 Boss $25 | 1 Boss $14 |
| HS8LL7TC | 6 Boss $56 | 1 Boss $9 | 1 Boss $14 |
| HSC1L2DX | 2 Boss $15 | 2 Boss $13 | 1 Big $10 |
| I68YASXJ | 6 Boss $26 | 6 Boss $26 | 1 Boss $14 |
| ICUESE5D | 5 Boss $19 | 4 Boss $25 | 1 Boss $15 |
| IMMOLATE | 5 Boss $25 | 5 Boss $27 | 1 Boss $12 |
| J3GHQHJM | 7 Boss $25 | 3 Boss $16 | 1 Big $7 |
| JQB4BN6F | 5 Boss $25 | 5 Boss $28 | 1 Boss $14 |
| K84ZCS3Q | 7 Boss $17 | 7 Boss $25 | 1 Boss $15 |
| K8PU119J | 4 Boss $20 | 5 Boss $1 | 1 Boss $17 |
| K94TS4B8 | 5 Boss $13 | 2 Boss $12 | 1 Boss $13 |
| KPGY6GCH | 6 Boss $21 | 6 Boss $28 | 1 Big $10 |
| LHZYIWR6 | 5 Boss $27 | 5 Boss $25 | 1 Boss $16 |
| LMGJPMKP | 3 Boss $15 | 3 Boss $14 | 1 Boss $16 |
| M2LAET6A | 8 Boss $36 | 9 Boss $20 | 1 Boss $15 |
| M4LV5E89 | 1 Boss $8 | 3 Boss $14 | 2 Boss $24 |
| M6XD1INZ | 2 Boss $8 | 9 Boss $35 | 1 Boss $15 |
| MIM66QT9 | 4 Boss $8 | 4 Boss $20 | 2 Boss $21 |
| MM1H85E3 | 6 Boss $38 | 7 Boss $27 | 1 Boss $14 |
| MRBF7L15 | 4 Boss $77 | 4 Boss $79 | 1 Boss $17 |
| MSQZGPMX | 4 Boss $10 | 4 Boss $10 | 1 Big $10 |
| OG4YQPSI | 5 Boss $26 | 5 Boss $26 | 2 Boss $32 |
| PI1J5YAG | 3 Boss $17 | 3 Boss $15 | 1 Boss $14 |
| PJ5TB88W | 5 Boss $26 | 5 Boss $26 | 1 Boss $12 |
| PQNVFI72 | 1 Boss $2 | 5 Boss $8 | 1 Boss $13 |
| PT6IF52R | 5 Boss $29 | 2 Boss $1 | 1 Boss $18 |
| PTSBMSMQ | 5 Boss $26 | 5 Boss $20 | 2 Boss $27 |
| QY3TQZJ9 | 5 Boss $28 | 5 Boss $27 | 2 Boss $35 |
| R8DFV7RD | 9 Boss $18 | 8 Boss $25 | 2 Boss $30 |
| R8W2CT59 | 4 Boss $21 | 5 Boss $25 | 1 Big $9 |
| RUXS91YF | 8 Boss $29 | 4 Boss $21 | 1 Big $7 |
| RXI42HZ2 | 8 Boss $25 | 7 Boss $25 | 2 Boss $22 |
| S1QJX1CS | 5 Boss $23 | 5 Boss $24 | 1 Big $8 |
| SF9SZOB1 | 6 Boss $25 | 6 Boss $27 | 1 Boss $13 |
| SLRSKCG1 | 5 Boss $37 | 1 Small $4 | 1 Big $7 |
| SMC7XNT8 | 8 Boss $0 | 5 Boss $24 | 1 Boss $17 |
| SQVZX29L | 3 Boss $16 | 3 Boss $17 | 2 Boss $21 |
| T5A1T1DZ | 7 Boss $25 | 7 Boss $29 | 1 Boss $15 |
| U516KUJP | 9 Boss $17 | 7 Boss $20 | 2 Boss $20 |
| U8RJYV6M | 2 Boss $3 | 4 Boss $20 | 2 Boss $26 |
| UCAMJYYK | 7 Boss $29 | 7 Boss $29 | 1 Boss $17 |
| USQF4ZAV | 6 Boss $40 | 5 Boss $27 | 1 Small $4 |
| V3PUR5L4 | 7 Boss $51 | 7 Boss $96 | 1 Boss $13 |
| VNOMH111 | 4 Boss $21 | 5 Boss $40 | 2 Boss $28 |
| WBC1DGJQ | 7 Boss $30 | 9 Boss $22 | 2 Boss $24 |
| WIF67A3S | 6 Boss $28 | 6 Boss $27 | 2 Boss $24 |
| WLAVAQ75 | 4 Boss $20 | 5 Boss $7 | 1 Big $9 |
| WVPJKPRD | 6 Boss $26 | 5 Boss $26 | 1 Big $9 |
| X7FG4FH8 | 5 Boss $26 | 4 Boss $22 | 1 Boss $14 |
| XD55DZ57 | 2 Boss $12 | 4 Boss $20 | 2 Boss $23 |
| XDW9VBHQ | 5 Boss $26 | 6 Boss $37 | 1 Boss $14 |
| YB54A1EH | 5 Boss $26 | 8 Boss $39 | 1 Boss $15 |
| YD2HMRBR | 5 Boss $30 | 5 Boss $30 | 1 Boss $14 |
| YPB7TZWJ | 5 Boss $23 | 5 Boss $24 | 2 Boss $28 |
