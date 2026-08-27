# MATCHES — the live-match ledger (the brief's "measurement ritual")

One row per completed G2 live match. This is the periodic "has it surpassed me" record:
when the agent column starts winning, that's the headline. Session logs live in
`ghost/runs/live/<seed>_<ts>/session.log` (gitignored; the ledger is the durable record).

| date | seed | spec | winner | lives (Tagg-agent) | Nemesis rounds (agent vs Tagg) | notes |
|---|---|---|---|---|---|---|
| 2026-08-27 | `V8XYRDID` | ev:fast | — VOID | — | A2: 2,930 vs 1,073 (agent) | first-ever live session; crashed on Talisman comma scores, then the recovery double-read corrupted the mirror (both fixed, `eb6527c`) — not a valid result |
| 2026-08-27 | `AOG8R942` | ev:fast | **Tagg** | **3 – 0** | A2: **18,409** vs 11,823 (agent) · A3: 12,410 vs **20,303** · A4: 16,650 vs **6,594,647** · A5: 31,674 vs **269,020** · A6: 65,126 vs **61,931,518** | first CLEAN full match — every mechanism worked (closed-loop lives, comeback money, reveals, launcher cleanup). Agent took ante 2 on burst hand-play (the Canio-seed 15.5k opener), then Tagg's build went exponential (20k → 62M over four antes) while the agent's crawled 12k → 65k. The shop/build-scaling gap is the whole story — consistent with the standing estimate (hand play at/past human, build layer far behind) and with the V-v2 sandbagging/scaling thread |

Reading so far: the agent is competitive exactly as long as raw hand-play decides the
race (antes 2-3) and falls off a cliff once multiplicative builds come online. The
ladder question for next matches: does `ev:full` (or a V-guided shop tier) close any of
the ante-4+ gap, and how many antes does Tagg need to be safe?
