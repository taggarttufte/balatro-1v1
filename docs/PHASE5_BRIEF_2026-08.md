# Phase 5 Brief — The EV player (2026-08-23)

## 0. The pivot, in one paragraph

Phase 4's run (`mp/agent/runs/real1/`, 106 generations, stopped 2026-08-23 12:40) proved the pipeline learns and
plateaued at mean ante ~5.5 — **with a search that could see the future** (the keyed RNG and draw order are cloned;
every MCTS simulation saw the true shop/draws/packs). Tagg's read, which the lead agrees with: Balatro is
fundamentally an EV calculation over known chance (draw pile composition, pack pools, interest arithmetic) plus a
long-horizon valuation (build strength, the lives race, the opponent). MCTS was a poor, cheating estimator of that.
**New design (TD-Gammon-shaped):** the only learned object is **V(state) ≈ P(win the MLB match)**; decisions are
argmax over actions of EV under V, where EV is computed **analytically where chance is known** (hand play by
expectimax over the real draw distribution to the end of the blind; shop/packs/rerolls/vouchers by the
decision-statistics module) and V supplies the leaf valuation. Labels come from **determinized Monte-Carlo
rollouts** of the analytic policy (honest by construction — rollouts sample worlds, there is no true seed to peek
at). No policy net in v1, so the action space and every calculator can evolve during training; only V's input
(`docs/STATE_SPEC_v1.md`, Tagg reviews) and target are frozen.

Tagg's decisions (2026-08-23): V target = P(win MLB match), MLB-only training; symmetric opponent for labels
first; Red/White first with conditioning fields present; **V ≈ 5M params**; acceptance test = the snapshot advisor
on a Bloodstone-vs-Invisible-Joker(+Blueprint) state; tempo-vs-late-game is per-action, never a "mode"; the
opponent is modelled in levels (0: features; 1: belief over the shared menu; 2: signalling).

State at kickoff: branch `mp/campaign` @ `b1ac0ef` (+ W1-parallel partial files uncommitted, default-off). Gates:
engine 1616, mp/tests 1073, agent 309, tournament 74, eval 125, replay 82, parity 126/126. `mp/engine/**` and
`mp/rng/**` frozen EXCEPT the determinize API (W2).

## 1. Workstreams (disjoint ownership)

| # | Workstream | Model | Owns |
|---|---|---|---|
| **W1** | State spec v1 → `SetValueNet` 5M + encoder v2 (opponent block, blind offers, deck/discard counts, reserved slots) | strong | `mp/agent/mcts/encoder_v2.py`, `mp/agent/mcts/value_net.py`, tests, `VALUE_NOTES.md`; may extend `encoder_set.py` behind a version flag |
| **W2** | Determinization (`BalatroGame.determinize` / `clone_determinized`) + clairvoyance measurement on `real1/latest.pt` | sonnet | `mp/engine/balatro_sim/game.py` (this API only), `mp/engine/tests/engine_tests/test_determinize.py`, `mp/agent/mcts/determinize.py`, `DETERMINIZE_NOTES.md` |
| **W3** | **Analytic hand player**: exact EV of play/discard over the real draw distribution, depth = end of blind, W0 scorer at the leaf; plus "plan-free" build reasoning via V hook | strong | `mp/ev/hand.py`, `mp/ev/player.py` (the rollout policy: hand + shop via W4's stats, falls back to W0 heuristics until W4 lands), tests, `EV_NOTES.md` |
| **W4** | Decision statistics (packs / rerolls / vouchers: hit valuation, P(hit), true cost incl. interest loss, urgency, net EV) + 126-seed sweep | sonnet | `mp/stats/**`, `STATS_NOTES.md`, `mp/results/stats_sweep_*.json` |
| **W5** | Label generator + V trainer + multi-process rollouts: sample states from logged play → determinized rollouts of the analytic policy (symmetric opponent) → P(win) labels (race calculator at the ante-12 truncation) → regression of V; 16-core worker pool (CPU), GPU for training; checkpoints/PAUSE/resume; eval hooks | strong | `mp/ev/labels.py`, `mp/ev/race.py`, `mp/ev/train_v.py`, `mp/ev/workers.py`, tests, `TRAINV_NOTES.md`; may reuse W1-parallel's partial files (`mp/agent/parallel/`, `train/parallel.py`) or delete them |
| **W6** | Race calculator API surface + **snapshot advisor CLI** + head-to-head evals (analytic player vs `real1` net, both non-clairvoyant; advisor on the Bloodstone/Invisible state) | sonnet (after W3/W4/W5 land) | `mp/ev/advisor.py`, `mp/ev/cli.py`, `mp/results/advisor_*.md`, tests |

Shared files: none. `mp/agent/mcts/search.py`/`batched.py` are retired from the main path (kept; `MCTSPlayer` stays
as a baseline/optional deep search).

## 2. Interfaces (agreed up front)

- `mp/ev/player.py::EVPlayer(value_fn: Optional[Callable[[BalatroGame], float]], stats=..., determinize_seed=...)`
  with `act(game) -> action` and `reset()` — the `Player` protocol used by tournament/eval/replay. `value_fn=None`
  → leaf = W0 scorer proxy (bootstrap); later `value_fn = V`.
- `mp/ev/hand.py::hand_ev(game, action, depth, n_draw_samples) -> float` (exact where enumerable, sampled
  otherwise; side-effect-free; **never reads the draw-pile ORDER**, only composition — assert it).
- `mp/stats/decide.py::decision_table(game) -> list[Row]` (W4) consumed by `EVPlayer` in shop/pack states.
- `BalatroGame.clone_determinized(seed) -> BalatroGame` (W2) used by W3's draw sampling and W5's rollouts.
- `mp/ev/race.py::p_win(my_curve, their_curve, my_lives, their_lives, ante) -> float` (W5) used by labels + advisor.
- Labels: `mp/ev/labels.py::label_state(game, n_rollouts, ...) -> (p_win, ci)`; states sampled from
  `mp/replay` logs of EVPlayer self-play (diverse, on-policy).
- V: `mp/agent/mcts/value_net.py::SetValueNet` + `encoder_v2` (W1); `value_fn = lambda g: V(encode(g))`.

## 3. Gates

1. W2: determinize invariants; engine_parity 126/126 unchanged; clairvoyance table (clairvoyant vs determinized
   on `real1/latest.pt`, 30 seeds, disagreement by action type).
2. W3: analytic player clears ante 1 on ≥ 80% of the 126 seeds (scripted greedy: 37%); mean final ante reported;
   side-effect-free; ≤ 100 ms per hand decision.
3. W4: sweep tables; decision table ≤ 50 ms.
4. W5: 16-worker label throughput (labels/min); V trains on ≥ 50k labels with held-out calibration (reliability
   curve) and beats a constant predictor; PAUSE/resume; one end-to-end "EVPlayer with V" tournament.
5. W6: head-to-head analytic-vs-`real1` paired by seed; the advisor prints two P(win)±CI for Tagg's state.
6. Lead: all prior gates green; commit; launch V training.

## 4. For Tagg
Review `docs/STATE_SPEC_v1.md` (the irreversible part) — additions welcome, deletions too. Nothing else blocks.
