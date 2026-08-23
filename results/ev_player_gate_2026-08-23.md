# EV player gate 2 — 2026-08-23

`python mp/ev/gate_ev_player.py --seeds 12 --procs 4`

Seeds: 12 of 126 ground-truth seeds (offset 0); vanilla ruleset, Red deck, White stake; 4 processes; wall 30 s.

## Outcomes (bootstrap 95% CI)

| metric | fast | full | greedy |
|---|---|---|---|
| ante1_clear | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] | 50.0% [25.0, 75.0] |
| ante2_clear | 83.3% [58.3, 100.0] | 83.3% [58.3, 100.0] | 0.0% [0.0, 0.0] |
| ante3_clear | 83.3% [58.3, 100.0] | 83.3% [58.3, 100.0] | 0.0% [0.0, 0.0] |
| ante4_clear | 75.0% [50.0, 100.0] | 75.0% [50.0, 100.0] | 0.0% [0.0, 0.0] |
| won | 8.3% [0.0, 25.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| mean_final_ante | 5.33 [4.25, 6.42] | 5.25 [4.25, 6.17] | 1.50 [1.25, 1.75] |
| mean_blinds_cleared | 11.00 [8.75, 13.08] | 10.75 [8.58, 12.83] | 2.33 [1.75, 2.83] |
| money_at_ante3 | 23.90 [19.30, 28.60] | 24.40 [20.90, 28.60] | nan [nan, nan] |
| hands unused / cleared blind | 2.17 | 2.30 | 1.36 |
| $ at ante 3: n reaching | 10 | 10 | 0 |

## Paired-by-seed deltas vs greedy (mean difference, bootstrap 95% CI)

- **fast − greedy** (n=12): ante-1 clear 50.0% [25.0, 75.0]; final ante 3.83 [2.75, 5.00]; blinds cleared 8.67 [6.50, 10.75]; final $ 5.42 [0.92, 9.92]
- **full − greedy** (n=12): ante-1 clear 50.0% [25.0, 75.0]; final ante 3.75 [2.75, 4.75]; blinds cleared 8.42 [6.25, 10.42]; final $ 3.25 [-1.00, 7.92]

## Wall-clock per decision (ms)

| player | hand mean | hand p50 | hand p95 | hand max | n | shop mean | shop p95 | n |
|---|---|---|---|---|---|---|---|---|
| fast | 4.43 | 3.01 | 12.25 | 50.1 | 564 | 22.1 | 46.8 | 665 |
| full | 73.58 | 58.08 | 180.65 | 415.8 | 531 | 22.5 | 47.4 | 653 |
| greedy | 2.02 | 2.03 | 2.72 | 2.9 | 168 | 0.1 | 0.1 | 28 |

Budgets: fast ≤ 5 ms mean, full ≤ 100 ms mean per SELECTING_HAND decision.

## Draw-order invariance

- fast: 72 sampled states, `game.deck` permuted → identical decision in 72 (0 mismatches)
- full: 72 sampled states, `game.deck` permuted → identical decision in 72 (0 mismatches)

## MLB match (EVPlayer fast vs EVPlayer fast, full MLBMatch)

- seed 11111111: done=True winner=1 final antes [7, 7] lives [0, 1] steps 504 in 6.7 s; Nemeses (ante, loser, s0, s1): [(2, 0, 6288, 6672), (3, 1, 17967, 13088), (4, 0, 26495, 30054), (5, 1, 24356, 22279), (6, 0, 22495, 25956)]

## Per-seed (furthest ante/blind, $)

| seed | fast | full | greedy |
|---|---|---|---|
| 11111111 | 6 Boss $25 | 5 Boss $20 | 2 Boss $21 |
| 1558AXDL | 6 Boss $26 | 6 Boss $27 | 2 Boss $22 |
| 15H9Z3IY | 2 Boss $12 | 2 Boss $11 | 1 Big $8 |
| 1KV4W6YS | 6 Boss $27 | 6 Boss $27 | 2 Boss $25 |
| 1MD1YZ9T | 6 Boss $38 | 6 Boss $26 | 2 Boss $33 |
| 28V7DD4H | 7 Boss $27 | 7 Boss $25 | 1 Big $9 |
| 29DAQVG1 | 5 Boss $27 | 5 Boss $25 | 1 Big $9 |
| 29Y3L4S9 | 6 Boss $27 | 6 Boss $24 | 2 Boss $24 |
| 29ZSW8MY | 2 Boss $13 | 2 Boss $14 | 2 Boss $26 |
| 2BRGI767 | 5 Boss $26 | 6 Boss $25 | 1 Boss $15 |
| 2CP4KSXZ | 9 Boss $15 | 8 Boss $15 | 1 Boss $15 |
| 2GHBLJD9 | 4 Boss $24 | 4 Boss $22 | 1 Boss $15 |
