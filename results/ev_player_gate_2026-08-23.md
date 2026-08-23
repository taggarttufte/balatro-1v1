# EV player gate 2 — 2026-08-23

`python mp/ev/gate_ev_player.py --seeds 12 --procs 4`

Seeds: 12 of 126 ground-truth seeds (offset 0); vanilla ruleset, Red deck, White stake; 4 processes; wall 16 s.

## Outcomes (bootstrap 95% CI)

| metric | fast | full | greedy |
|---|---|---|---|
| ante1_clear | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] | 50.0% [25.0, 75.0] |
| ante2_clear | 83.3% [58.3, 100.0] | 83.3% [58.3, 100.0] | 0.0% [0.0, 0.0] |
| ante3_clear | 66.7% [41.7, 91.7] | 75.0% [50.0, 100.0] | 0.0% [0.0, 0.0] |
| ante4_clear | 66.7% [41.7, 91.7] | 66.7% [41.7, 91.7] | 0.0% [0.0, 0.0] |
| won | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| mean_final_ante | 4.67 [3.75, 5.67] | 4.83 [3.92, 5.83] | 1.50 [1.25, 1.75] |
| mean_blinds_cleared | 9.58 [7.83, 11.42] | 10.17 [8.33, 11.83] | 2.33 [1.75, 2.83] |
| money_at_ante3 | 22.20 [19.10, 26.20] | 25.50 [21.50, 30.10] | nan [nan, nan] |
| hands unused / cleared blind | 2.13 | 2.20 | 1.39 |
| $ at ante 3: n reaching | 10 | 10 | 0 |

## Paired-by-seed deltas vs greedy (mean difference, bootstrap 95% CI)

- **fast − greedy** (n=12): ante-1 clear 50.0% [25.0, 75.0]; final ante 3.17 [2.08, 4.33]; blinds cleared 7.25 [5.17, 9.25]; final $ 2.67 [-4.92, 10.42]
- **full − greedy** (n=12): ante-1 clear 50.0% [25.0, 75.0]; final ante 3.33 [2.33, 4.42]; blinds cleared 7.83 [5.92, 9.67]; final $ 4.00 [-1.75, 9.92]

## Wall-clock per decision (ms)

| player | hand mean | hand p50 | hand p95 | hand max | n | shop mean | shop p95 | n |
|---|---|---|---|---|---|---|---|---|
| fast | 3.18 | 2.16 | 8.97 | 32.9 | 516 | 5.1 | 17.7 | 564 |
| full | 57.98 | 47.89 | 146.27 | 385.8 | 542 | 6.0 | 22.1 | 596 |
| greedy | 2.10 | 2.44 | 2.65 | 3.2 | 167 | 0.1 | 0.1 | 28 |

Budgets: fast ≤ 5 ms mean, full ≤ 100 ms mean per SELECTING_HAND decision.

## Draw-order invariance

- fast: 72 sampled states, `game.deck` permuted → identical decision in 72 (0 mismatches)
- full: 72 sampled states, `game.deck` permuted → identical decision in 72 (0 mismatches)

## MLB match (EVPlayer fast vs EVPlayer fast, full MLBMatch)

- seed 11111111: done=True winner=0 final antes [7, 7] lives [1, 0] steps 610 in 1.9 s; Nemeses (ante, loser, s0, s1): [(2, None, 7224, 7224), (3, 0, 12927, 18243), (4, 0, 7618, 11462), (5, 1, 23722, 20450), (6, 1, 47874, 33608), (7, 1, 53364, 50040)]

## Per-seed (furthest ante/blind, $)

| seed | fast | full | greedy |
|---|---|---|---|
| 11111111 | 6 Boss $26 | 6 Boss $26 | 2 Boss $21 |
| 1558AXDL | 6 Boss $28 | 6 Boss $24 | 2 Boss $22 |
| 15H9Z3IY | 3 Boss $18 | 2 Boss $11 | 1 Big $8 |
| 1KV4W6YS | 5 Boss $27 | 4 Boss $20 | 2 Boss $25 |
| 1MD1YZ9T | 2 Boss $6 | 3 Boss $17 | 2 Boss $33 |
| 28V7DD4H | 7 Boss $25 | 7 Boss $27 | 1 Big $9 |
| 29DAQVG1 | 5 Boss $32 | 5 Boss $28 | 1 Big $9 |
| 29Y3L4S9 | 3 Boss $17 | 5 Boss $26 | 2 Boss $25 |
| 29ZSW8MY | 2 Boss $13 | 2 Boss $14 | 2 Boss $26 |
| 2BRGI767 | 5 Boss $28 | 5 Boss $25 | 1 Boss $15 |
| 2CP4KSXZ | 7 Boss $28 | 7 Boss $26 | 1 Boss $15 |
| 2GHBLJD9 | 5 Boss $7 | 6 Boss $27 | 1 Boss $15 |
