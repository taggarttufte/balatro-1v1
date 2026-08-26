# PvP turn protocol: ON vs OFF (paired seeds, self-play both arms)

_2026-08-26T00:29:28_

## protocol ON (both sides)

- A = `on` · B = `on` · pvp_protocol = `trailer_compelled`
- seeds 30 x 2 seatings · lives 4 · max_steps 4000 · procs 5 · wall 26s

| metric | value |
|---|---|
| matches (both seatings) | 60 |
| decided / undecided | 60 / 0 |
| A win rate | 0.500  CI [0.383, 0.633] |
| A win rate by seating (0 / 1) | 0.567 / 0.433 |
| Nemeses resolved (total) | 260 |
| leader passes / match | 5.53 |
| early-end cuts / match | 1.37 |
| early-end rate / Nemesis | 0.315 |
| hands played per Nemesis (A / B) | 3.89 / 3.89 |
| discards taken at Nemeses / match (A / B) | 17.28 / 17.28 |
| Glass cards alive at the end (A / B) | 0.15 / 0.15 |
| $ banked in a LOST Nemesis (A / B) | 1.07 / 1.07 |
| mean final ante (A / B) | 5.97 / 5.97 |
| mean final $ (A / B) | 29.72 / 29.72 |
| mean lives margin (A - B) | +0.000 |
| mean steps / match | 379 |
| seconds / match | 1.78 |

## protocol OFF (both sides, the pre-W-PVP player)

- A = `off` · B = `off` · pvp_protocol = `canonical`
- seeds 30 x 2 seatings · lives 4 · max_steps 4000 · procs 5 · wall 34s

| metric | value |
|---|---|
| matches (both seatings) | 60 |
| decided / undecided | 60 / 0 |
| A win rate | 0.500  CI [0.367, 0.617] |
| A win rate by seating (0 / 1) | 0.300 / 0.700 |
| Nemeses resolved (total) | 274 |
| leader passes / match | 0.00 |
| early-end cuts / match | 0.63 |
| early-end rate / Nemesis | 0.139 |
| hands played per Nemesis (A / B) | 4.07 / 4.07 |
| discards taken at Nemeses / match (A / B) | 18.35 / 18.35 |
| Glass cards alive at the end (A / B) | 0.17 / 0.17 |
| $ banked in a LOST Nemesis (A / B) | 1.18 / 1.18 |
| mean final ante (A / B) | 6.10 / 6.10 |
| mean final $ (A / B) | 33.20 / 33.20 |
| mean lives margin (A - B) | +0.000 |
| mean steps / match | 389 |
| seconds / match | 2.47 |

