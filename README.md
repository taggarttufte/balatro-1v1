# mp/ — Balatro Multiplayer (MLB) Statistical Player

**Internal subproject. Not linked from the top-level README by design.**

A rebuild of the Balatro engine with real-game RNG parity, the Major League multiplayer ruleset, and infrastructure for N-agent same-seed tournaments. Separate from the BRL (single-player PPO) line; shares the repo, nothing else.

## Layout

```
mp/
├── docs/          assessment, update list, decision economics, training design, campaign plan
├── rng/           keyed pseudorandom core (pseudohash + LCG + LuaJIT math.random port), pools, keys, generation
├── oracle/        ground-truth seed data + parity harness
├── engine/        balatro_sim fork (from fix/sim-fidelity-2026-07 + clone()/legal_actions() from balatro-mcts)
├── tests/
├── scripts/
├── _reference/    extracted game Lua — GITIGNORED, never commit
├── CAMPAIGN_LOG.md
└── README.md
```

## Start here

1. `docs/MP_CAMPAIGN_PLAN_2026-08.md` — phases, gates, what needs a human.
2. `docs/MP_UPDATE_LIST_2026-08.md` — every engine change, with file:line.
3. `CAMPAIGN_LOG.md` — what's been done, by whom, with what result.

## Ground rules

- Everything stays under `mp/`. Do not touch BRL code, branches, or `results/`.
- `_reference/` holds Balatro 1.0.1o Lua extracted from the local install. Port algorithms from it; never copy it into deliverables or commit it.
- Oracle first. No keyed-RNG threading into the engine until Phase 0 parity passes.
