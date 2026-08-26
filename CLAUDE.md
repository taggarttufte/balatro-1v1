# balatro-1v1 — router for AI readers

This repository holds **one project**: a from-the-RNG-up rebuild of Balatro, its Major League
multiplayer ruleset, and an analytic EV player for it. It has an archived predecessor, which
matters only for one story — see § 2.

## 1. Start here

**Start at [`README.md`](README.md).** The narrative — dated, agent-by-agent, with every gate
and every number including the negative ones — is [`CAMPAIGN_LOG.md`](CAMPAIGN_LOG.md).

This project (2026-08 →) is a from-the-RNG-up rebuild: a bit-exact port of Balatro's
random-number chain verified against the game's own LuaJIT, a 126-seed ground-truth corpus
cross-checked against two independent public seed analyzers, an engine that reproduces those
seeds exactly through ante 8, the Major League multiplayer ruleset read out of the mod's
source, and — after a measurement showed the trained MCTS agent had been reading the true
future — an **analytic expectimax EV player** that beats it, plus an in-progress
`V(state) = P(win)` value function.

Branch: `main`. `README.md` carries a **claims ledger** (every headline claim → evidence file
→ reproduction command) and a **self-audits that changed conclusions** section.

## 2. The predecessor is a separate, archived repo (2025 – 2026-04)

The PPO line (BRL) — eight architectures, peak 2.35% solo win rate, concluded April 2026 with
the verdict "shaped-reward PPO has hit a structural search ceiling" — lives at
**[github.com/taggarttufte/balatro-rl](https://github.com/taggarttufte/balatro-rl)** and is
kept intact as an artifact. This project began as a subdirectory of it and was split out with
its history; nothing in this tree depends on that repo at runtime.

**That conclusion was revised by this project's own July-2026 audit**
([`docs/SIM_AUDIT_2026-07-29.md`](docs/SIM_AUDIT_2026-07-29.md), a verbatim copy of the
predecessor's `results/SIM_AUDIT_2026-07-29.md`): the simulator awarded no money for beating a
blind, reset the deck every blind, and ignored joker ordering, so every logged score was
inflated and the "exploration failure" the retrospective diagnosed was in fact the reward
function's global optimum. Do not compare any number here against 2.35%, and do not treat that
repo's `results/PROJECT_RETROSPECTIVE.md` as current. The old story is kept intact on purpose —
the audit that overturned it is part of the record.

## Headline results (all reproducible — see the ledger)

| Result | Number | Evidence |
|---|---|---|
| Engine matches the real game's RNG | **126/126 seeds exact through ante 8**, every field | `python -m oracle.engine_parity --antes 1-8 --rerolls 5` |
| The trained MCTS agent was clairvoyant | ante-1 clear **63.3% → 33.3%** once the future is withheld; 0.7% discard agreement | `results/clairvoyance_2026-08-23.md` |
| Analytic EV player (no net, no search), 126 seeds | ante-1 clear **95.2%** vs scripted greedy **31.7%** | `results/ev_player_gate_2026-08-23.md` |
| …versus the 106-generation MCTS, played honestly | **57 / 58** decided matches (98.3%, CI [94.8, 100]) | `results/h2h_ev_full_vs_real1_det_30seeds.md` |
| *Negative:* the learned V as a full policy | **2 / 60** matches vs the hand-written rules — per-action EV gaps sit below label noise | `results/tournament_v_v_full.json` |
| …after the counterfactual-pairs round (2026-08-26) | **12 / 60**, while a same-data control without the ranking loss stays at 2 / 60 — the gain is the lever, attributed | `results/tournament_v_v_v2.json` |

## Repository layout

```
README.md      ← start here; claims ledger + self-audits
CAMPAIGN_LOG.md  the dated narrative
├─ rng/          bit-exact pseudorandom core + generation layer
├─ oracle/       126-seed ground truth + parity harnesses
├─ engine/       the simulator (game keys, delegated generation, MLB rules, 15 decks)
├─ agent/        MCTS + nets (the `real1` line), set encoder, SetValueNet (5M params)
├─ ev/           the analytic EV player, labels, V trainer, pairs, advisor CLI
├─ stats/ eval/ tournament/ replay/ tests/ results/ docs/
```

## Notes on what is and is not in the tree

- **`_reference/` is absent by design** — Balatro's Lua, extracted from the local Steam
  install. Copyright LocalThunk/Playstack; gitignored, never committed. Algorithms are ported
  and cited by `file:line`; the Lua is read from the local install at test time. Same for
  `oracle/blueprint_runner/vendor/` (third-party seed-analyzer clones, re-cloned by
  `setup.ps1` at the commits pinned in `oracle/SOURCES.md`) and the BalatroMultiplayer mod
  Lua (read from `%APPDATA%`).
- **Gitignored and regenerable:** `*/runs/`, `*.pt` checkpoints, `checkpoints_*/`, `logs_*/`.
  A few ledger rows depend on a checkpoint that is not in the repo; they are marked 🔒.
- **Low-signal for a reader:** `__pycache__/`, `results/*.json` (read the paired `.md`), and
  `oracle/ground_truth/*.json` (generated fixtures). This applies to *low-signal* only — the
  negative results in `README.md` are load-bearing and are meant to be read.
