# EV player gate 2 — 2026-08-26

`python ev/gate_ev_player.py --procs 12 --out results/ev_player_gate_shop_2026-08-26`

Seeds: 126 of 126 ground-truth seeds (offset 0); vanilla ruleset, Red deck, White stake; 12 processes; wall 53 s.

## Outcomes (bootstrap 95% CI)

| metric | fast | full | greedy |
|---|---|---|---|
| ante1_clear | 94.4% [90.5, 98.4] | 96.0% [92.1, 99.2] | 31.7% [23.8, 39.7] |
| ante2_clear | 81.7% [75.4, 88.1] | 85.7% [79.4, 91.3] | 0.0% [0.0, 0.0] |
| ante3_clear | 75.4% [67.5, 82.5] | 77.0% [69.8, 84.1] | 0.0% [0.0, 0.0] |
| ante4_clear | 59.5% [50.8, 68.3] | 67.5% [58.7, 75.4] | 0.0% [0.0, 0.0] |
| won | 2.4% [0.0, 5.6] | 2.4% [0.0, 4.8] | 0.0% [0.0, 0.0] |
| mean_final_ante | 4.68 [4.34, 5.02] | 4.87 [4.56, 5.18] | 1.32 [1.24, 1.40] |
| mean_blinds_cleared | 9.63 [8.84, 10.40] | 9.88 [9.19, 10.60] | 2.16 [2.01, 2.31] |
| money_at_ante3 | 20.70 [19.05, 22.57] | 20.87 [19.32, 22.44] | nan [nan, nan] |
| hands unused / cleared blind | 2.13 | 2.18 | 1.25 |
| $ at ante 3: n reaching | 103 | 108 | 0 |

## Paired-by-seed deltas vs greedy (mean difference, bootstrap 95% CI)

- **fast − greedy** (n=126): ante-1 clear 62.7% [54.0, 71.4]; final ante 3.37 [3.01, 3.71]; blinds cleared 7.48 [6.67, 8.29]; final $ 9.28 [6.71, 11.96]
- **full − greedy** (n=126): ante-1 clear 64.3% [55.6, 73.0]; final ante 3.56 [3.24, 3.89]; blinds cleared 7.72 [6.99, 8.47]; final $ 13.46 [10.26, 16.90]

## Wall-clock per decision (ms)

| player | hand mean | hand p50 | hand p95 | hand max | n | shop mean | shop p95 | n |
|---|---|---|---|---|---|---|---|---|
| fast | 4.12 | 2.84 | 12.05 | 77.3 | 5461 | 8.8 | 29.9 | 6041 |
| full | 70.87 | 57.02 | 171.62 | 536.2 | 5286 | 9.8 | 32.9 | 6204 |
| greedy | 2.21 | 2.49 | 2.91 | 4.0 | 1750 | 0.1 | 0.1 | 272 |

Budgets: fast ≤ 5 ms mean, full ≤ 100 ms mean per SELECTING_HAND decision.

## Draw-order invariance

- fast: 737 sampled states, `game.deck` permuted → identical decision in 737 (0 mismatches)
- full: 741 sampled states, `game.deck` permuted → identical decision in 741 (0 mismatches)

## MLB match (EVPlayer fast vs EVPlayer fast, full MLBMatch)

- seed 11111111: done=True winner=0 final antes [6, 6] lives [3, 0] steps 419 in 2.2 s; Nemeses (ante, loser, s0, s1): [(2, None, 7224, 7224), (3, 1, 22218, 19536), (4, 0, 9204, 16458), (5, 1, 26341, 19932), (6, 1, 37789, 18118)]

## Per-seed (furthest ante/blind, $)

| seed | fast | full | greedy |
|---|---|---|---|
| 11111111 | 5 Boss $44 | 5 Boss $52 | 2 Boss $21 |
| 1558AXDL | 6 Boss $29 | 6 Boss $49 | 2 Boss $22 |
| 15H9Z3IY | 4 Boss $33 | 2 Boss $11 | 1 Big $8 |
| 1KV4W6YS | 5 Boss $35 | 6 Boss $68 | 2 Boss $25 |
| 1MD1YZ9T | 2 Boss $6 | 5 Boss $37 | 2 Boss $33 |
| 28V7DD4H | 5 Boss $45 | 7 Boss $39 | 1 Big $9 |
| 29DAQVG1 | 7 Boss $53 | 7 Boss $65 | 1 Big $9 |
| 29Y3L4S9 | 5 Boss $29 | 3 Boss $15 | 2 Boss $25 |
| 29ZSW8MY | 2 Boss $13 | 2 Boss $14 | 2 Boss $26 |
| 2BRGI767 | 2 Boss $12 | 1 Boss $8 | 1 Boss $15 |
| 2CP4KSXZ | 8 Boss $0 | 8 Boss $0 | 1 Boss $15 |
| 2GHBLJD9 | 5 Boss $34 | 6 Boss $35 | 1 Boss $15 |
| 2H9N3ISZ | 2 Boss $14 | 5 Boss $41 | 1 Boss $15 |
| 2K9H9HN | 6 Boss $0 | 6 Boss $26 | 1 Boss $15 |
| 34JCNMPA | 9 Boss $49 | 9 Boss $49 | 2 Boss $24 |
| 3SZ71111 | 6 Boss $46 | 6 Boss $42 | 2 Boss $24 |
| 41Y71M6E | 4 Boss $37 | 1 Boss $6 | 1 Boss $15 |
| 46Y8UZEG | 2 Boss $10 | 3 Boss $30 | 2 Boss $21 |
| 4H2L46CE | 1 Boss $6 | 5 Boss $30 | 1 Big $9 |
| 4K8A9QER | 4 Boss $19 | 6 Boss $27 | 2 Boss $24 |
| 4R219TNX | 3 Boss $18 | 3 Boss $18 | 1 Boss $15 |
| 4T1SZKLF | 5 Boss $40 | 5 Boss $45 | 1 Boss $14 |
| 4UEGRRRA | 6 Boss $42 | 5 Boss $30 | 1 Big $10 |
| 51F2NVWK | 5 Boss $27 | 5 Boss $26 | 1 Boss $15 |
| 56QZEVDV | 5 Boss $31 | 2 Boss $12 | 1 Boss $15 |
| 5AWWF1M1 | 8 Boss $44 | 6 Boss $26 | 1 Boss $16 |
| 5UIUKHCI | 5 Boss $26 | 5 Boss $25 | 1 Boss $15 |
| 5YVHAEP | 6 Boss $49 | 5 Boss $59 | 1 Boss $14 |
| 6H6WNQM2 | 6 Boss $54 | 6 Boss $96 | 2 Boss $21 |
| 7I4M53DL | 7 Boss $29 | 5 Boss $25 | 1 Big $9 |
| 7LB2WVPK | 4 Boss $20 | 4 Boss $21 | 1 Boss $16 |
| 7ODNKXP | 5 Boss $26 | 6 Boss $32 | 2 Boss $21 |
| 7PEE6NA5 | 6 Boss $31 | 5 Boss $38 | 1 Big $9 |
| 7UNQV1C9 | 5 Boss $23 | 6 Boss $31 | 2 Boss $21 |
| 7YTVQERM | 6 Boss $24 | 6 Boss $33 | 1 Small $4 |
| 8I5QXNGE | 3 Boss $19 | 6 Boss $40 | 1 Boss $15 |
| 8IZ7HHE4 | 5 Boss $25 | 3 Boss $14 | 1 Boss $15 |
| 8Q47WV6K | 7 Boss $25 | 7 Boss $32 | 1 Boss $15 |
| 8QBRTPD | 1 Boss $7 | 5 Boss $31 | 1 Boss $14 |
| 8SIYIK9C | 3 Boss $2 | 7 Boss $54 | 1 Big $9 |
| 8UU564MF | 2 Boss $5 | 2 Boss $5 | 1 Boss $14 |
| 8Y7ABZ7C | 6 Boss $27 | 6 Boss $23 | 2 Boss $21 |
| 93W4UHC4 | 5 Boss $25 | 5 Boss $28 | 1 Boss $14 |
| 967889YL | 5 Boss $29 | 5 Boss $33 | 2 Boss $21 |
| 9Q9HQXZG | 1 Boss $7 | 6 Boss $56 | 2 Boss $31 |
| 9QV2ZT9X | 5 Boss $28 | 7 Boss $52 | 1 Boss $16 |
| 9VIGPRT6 | 2 Boss $9 | 2 Boss $8 | 2 Boss $31 |
| 9ZXMM1M | 5 Boss $61 | 5 Boss $61 | 2 Boss $24 |
| A | 2 Boss $11 | 2 Boss $11 | 1 Boss $14 |
| A1UAS1G5 | 5 Boss $27 | 5 Boss $28 | 1 Boss $14 |
| AE1K7DG1 | 4 Boss $37 | 6 Boss $42 | 2 Boss $20 |
| ALEEB | 7 Boss $80 | 7 Boss $112 | 2 Boss $26 |
| AMPYZUIU | 7 Boss $34 | 6 Boss $33 | 1 Boss $16 |
| APH9LX2Y | 5 Boss $27 | 6 Boss $26 | 2 Boss $33 |
| BA1D48RK | 5 Boss $25 | 5 Boss $25 | 1 Big $8 |
| BQ6A2D42 | 1 Boss $5 | 7 Boss $18 | 1 Big $9 |
| C19P8BQ4 | 4 Boss $25 | 3 Boss $21 | 2 Boss $21 |
| C1ZLPKAA | 4 Boss $21 | 4 Boss $20 | 1 Boss $15 |
| C93K3YWJ | 1 Boss $6 | 2 Boss $7 | 1 Boss $16 |
| CCBWC6CR | 6 Boss $38 | 6 Boss $38 | 1 Big $9 |
| CHPB293X | 3 Boss $15 | 3 Boss $15 | 2 Boss $22 |
| CIGHYTAU | 5 Boss $28 | 5 Boss $28 | 1 Big $10 |
| CMEEXU8I | 5 Boss $38 | 4 Boss $25 | 2 Boss $23 |
| D4TDD2B6 | 5 Boss $24 | 5 Boss $29 | 1 Big $9 |
| D7Y5419A | 2 Boss $12 | 2 Boss $13 | 1 Boss $15 |
| ECG8NVDZ | 4 Boss $20 | 2 Boss $14 | 2 Boss $27 |
| EGE5ZY77 | 7 Boss $28 | 7 Boss $30 | 1 Boss $16 |
| EVB8882H | 8 Boss $47 | 7 Boss $33 | 1 Big $7 |
| FSXYRB3J | 4 Boss $29 | 4 Boss $19 | 1 Big $10 |
| GVQL737K | 8 Boss $26 | 6 Boss $29 | 1 Boss $13 |
| GVYT2DGJ | 4 Boss $11 | 5 Boss $11 | 1 Boss $17 |
| H1EWXLNE | 9 Boss $28 | 5 Boss $42 | 1 Boss $15 |
| HPR8Q7K | 2 Boss $20 | 6 Boss $25 | 1 Boss $14 |
| HS8LL7TC | 4 Boss $24 | 4 Boss $26 | 1 Boss $14 |
| HSC1L2DX | 2 Boss $0 | 1 Boss $9 | 1 Big $10 |
| I68YASXJ | 7 Boss $29 | 6 Boss $35 | 1 Boss $14 |
| ICUESE5D | 4 Boss $18 | 5 Boss $25 | 1 Boss $15 |
| IMMOLATE | 5 Boss $32 | 5 Boss $30 | 1 Boss $12 |
| J3GHQHJM | 5 Boss $32 | 6 Boss $28 | 1 Big $7 |
| JQB4BN6F | 7 Boss $25 | 6 Boss $15 | 1 Boss $14 |
| K84ZCS3Q | 7 Boss $21 | 7 Boss $20 | 1 Boss $15 |
| K8PU119J | 4 Boss $29 | 5 Boss $37 | 1 Boss $17 |
| K94TS4B8 | 5 Boss $25 | 3 Boss $17 | 1 Boss $13 |
| KPGY6GCH | 6 Boss $41 | 6 Boss $34 | 1 Big $10 |
| LHZYIWR6 | 2 Boss $8 | 5 Boss $26 | 1 Boss $16 |
| LMGJPMKP | 8 Boss $20 | 3 Boss $10 | 1 Boss $16 |
| M2LAET6A | 2 Boss $9 | 9 Boss $33 | 1 Boss $15 |
| M4LV5E89 | 3 Boss $6 | 2 Boss $6 | 2 Boss $24 |
| M6XD1INZ | 6 Boss $53 | 6 Boss $42 | 1 Boss $15 |
| MIM66QT9 | 4 Boss $11 | 4 Boss $23 | 2 Boss $21 |
| MM1H85E3 | 6 Boss $42 | 2 Boss $10 | 1 Boss $14 |
| MRBF7L15 | 5 Boss $42 | 4 Boss $90 | 1 Boss $17 |
| MSQZGPMX | 4 Boss $22 | 4 Boss $34 | 1 Big $10 |
| OG4YQPSI | 5 Boss $44 | 5 Boss $44 | 2 Boss $32 |
| PI1J5YAG | 3 Boss $17 | 3 Boss $15 | 1 Boss $14 |
| PJ5TB88W | 6 Boss $15 | 5 Boss $31 | 1 Boss $12 |
| PQNVFI72 | 1 Boss $6 | 6 Boss $32 | 1 Boss $13 |
| PT6IF52R | 5 Boss $34 | 2 Boss $1 | 1 Boss $18 |
| PTSBMSMQ | 6 Boss $55 | 5 Boss $78 | 2 Boss $27 |
| QY3TQZJ9 | 5 Boss $27 | 5 Boss $54 | 2 Boss $35 |
| R8DFV7RD | 9 Boss $38 | 6 Boss $0 | 2 Boss $30 |
| R8W2CT59 | 4 Boss $29 | 6 Boss $36 | 1 Big $9 |
| RUXS91YF | 7 Boss $18 | 8 Boss $26 | 1 Big $7 |
| RXI42HZ2 | 6 Boss $35 | 7 Boss $28 | 2 Boss $22 |
| S1QJX1CS | 2 Boss $12 | 5 Boss $35 | 1 Big $8 |
| SF9SZOB1 | 7 Boss $28 | 8 Boss $69 | 1 Boss $13 |
| SLRSKCG1 | 5 Boss $34 | 1 Small $4 | 1 Big $7 |
| SMC7XNT8 | 2 Boss $13 | 5 Boss $52 | 1 Boss $17 |
| SQVZX29L | 3 Boss $23 | 3 Boss $25 | 2 Boss $21 |
| T5A1T1DZ | 6 Boss $22 | 5 Boss $15 | 1 Boss $15 |
| U516KUJP | 6 Boss $29 | 7 Boss $24 | 2 Boss $20 |
| U8RJYV6M | 3 Boss $3 | 2 Boss $10 | 2 Boss $26 |
| UCAMJYYK | 7 Boss $53 | 9 Boss $45 | 1 Boss $17 |
| USQF4ZAV | 2 Boss $8 | 5 Boss $27 | 1 Small $4 |
| V3PUR5L4 | 4 Boss $22 | 3 Boss $14 | 1 Boss $13 |
| VNOMH111 | 5 Boss $56 | 5 Boss $49 | 2 Boss $28 |
| WBC1DGJQ | 1 Boss $10 | 1 Boss $9 | 2 Boss $24 |
| WIF67A3S | 5 Boss $27 | 6 Boss $28 | 2 Boss $24 |
| WLAVAQ75 | 4 Boss $2 | 4 Boss $4 | 1 Big $9 |
| WVPJKPRD | 6 Boss $25 | 7 Boss $51 | 1 Big $9 |
| X7FG4FH8 | 6 Boss $25 | 4 Boss $26 | 1 Boss $14 |
| XD55DZ57 | 4 Boss $7 | 4 Boss $6 | 2 Boss $23 |
| XDW9VBHQ | 6 Boss $46 | 6 Boss $38 | 1 Boss $14 |
| YB54A1EH | 5 Boss $30 | 5 Boss $26 | 1 Boss $15 |
| YD2HMRBR | 4 Boss $26 | 4 Boss $20 | 1 Boss $14 |
| YPB7TZWJ | 5 Boss $27 | 5 Boss $27 | 2 Boss $28 |
