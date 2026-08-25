# mp/ — Balatro Multiplayer (MLB): bit-exact engine, oracle, and an analytic EV player

**This is the active research program in this repository (2026-08 →).** The top-level
`README.md` describes the 2025 – 2026-04 PPO line (BRL), which a July-2026 self-audit found
confounded; that story is kept intact because the audit is part of the record.

Start here, then read [`CAMPAIGN_LOG.md`](CAMPAIGN_LOG.md) — a dated, agent-by-agent log of
every phase, every gate, and every number, including the ones that went the wrong way.

---

## The arc

**A self-audit reopened a concluded project.** The BRL line was declared finished in April
2026 at a 2.35% solo win rate, with the stated conclusion that shaped-reward PPO had hit a
structural search ceiling. A mechanics audit against the game's wiki in July 2026
([`results/SIM_AUDIT_2026-07-29.md`](../results/SIM_AUDIT_2026-07-29.md)) found the simulator
wrong in ways that change what optimal play *is* — no money for beating a blind, the deck
reset to vanilla every blind so deck-building was impossible, joker order not affecting
scoring so every logged score was inflated, and the "exploration failure" the retrospective
blamed turning out to be the reward function's global optimum. The conclusion was not
disproved; it was shown to rest on a measurement that could not support it. Everything after
this point exists because the audit did not stop at "the numbers are wrong."

**Ground truth first: a bit-exact RNG oracle.** Rather than fix the simulator by reading the
wiki again, the rebuild started from the game's own randomness. Balatro's `pseudohash` /
`pseudoseed` / `pseudorandom` chain terminates in LuaJIT's `math.random`, so LuaJIT's
Tausworthe generator had to be ported too, not just the LCG. The Python port is compared
against the real Lua executed in-process (`lupa`) on 64-bit bit patterns with no tolerance:
68,640 chain values across 52 seeds × 55 keys, plus shuffles, element draws, and ~1,000
`pseudohash` inputs — **zero mismatches**. The shipped runtime turned out to be LuaJIT 2.0.5
(`lua51.dll` beside `Balatro.exe`), which was loaded via ctypes and matched too, closing the
FMA/x87 caveat. A real port bug fell out of this: `pseudohash("erratic7LB2WVPK")` is NaN in
the actual game (an intermediate hits exactly 0.0 → `inf % 1`), and the port now reproduces
that NaN bit-for-bit including its sign.

**Ground truth, end to end: 126 seeds, two independent analyzers, and a rule they all get
wrong.** Two public seed analyzers were run locally and headlessly — Blueprint (TypeScript
port of Immolate/TheSoul, `62898ed0`) and TheSoul's `immolate.wasm` (`780c1c21`) — and agree
on every field for all 126 corpus seeds. Against them, one difference appeared and did not go
away: the real game's `Card:set_ability` marks **every created card** in `used_jokers`
(cleared on `Card:remove`), so a shop never repeats a card across its two slots and never puts
a displayed shelf card into a pack opened at that shop, and each resample also shifts the
shared `_resample{n}` streams. 953 fields differ corpus-wide. Two derivations converged on
this independently (one by reading `Card:set_ability`, one by patching a driver onto
Blueprint's `Game` class), and it was then **confirmed in the live game**: on seed `7I4M53DL`
the Arcana pack's 3rd card is The Lovers because The Hierophant is on the shelf — and buying
then selling The Hierophant makes it come back as that 3rd card. Against the corrected
corpus: **126/126 seeds exact through ante 8, every field.** Against the analyzers as
published: 16/126 through ante 3.

**Engine correctness.** The simulator was then re-keyed to the game's own keys (only 7 of 150
jokers had previously agreed on key + rarity + cost) and made to *delegate* all generation —
shelves, rerolls, packs, vouchers, bosses, tags, shuffles, created cards — to the
oracle-verified generator, with every effect roll drawing from its real per-key stream and the
legacy single `random.Random` deleted. Engine-vs-ground-truth parity went from **0/126 through
ante 1 to 126/126 exact through ante 8**. The engine test suite went 847 → 1,441 in the
process, and a reachability probe over all 150 jokers / 52 consumables / 32 vouchers was driven
from 46 failures to 0.

**MLB ruleset.** The Major League multiplayer rules were read out of the installed mod (v0.5.2)
and its server repository rather than guessed: a life lost on *any* blind lost, Nemesis pays $5
win or lose, no unused-hand money at a PvP blind, a failed regular blind still pays and the run
proceeds, comeback money `$4 × cumulative lives lost`, endless (`win_ante = 999`), and an
**exact PvP tie costs nobody a life** (server-confirmed; the ghost replay's `>=` is not the live
rule). MLB's voucher stream is not vanilla — it routes through The Order's `Voucher0` path — which
was verified by executing the mod's own patched Lua in LuaJIT across 22 seeds × 8 antes × 3 modes,
0 mismatches. An exit gate proved queue alignment between two same-seed players by classifying
every differing RNG key position over 30 matches: 0 unexplained differences.

**MCTS training — and the clairvoyance discovery.** With infrastructure in place (N-agent
same-seed tournaments, checkpointing, batched leaf evaluation, tree reuse), a tournament-driven
MCTS + value-net run reached 106 generations (`real1`). Then the search's own docstring was
checked against the Phase-1 engine and found false: "each simulation sees different RNG
outcomes" had been true only on the old global `random.Random`. On the keyed engine the RNG
state and the draw-pile order are part of the cloned state, so **every simulation saw the true
future** — exact draws after a discard, exact reroll results, pack contents, every probability
roll. Determinization was added and the gap measured on the same checkpoint: ante-1 clear
**63.3% → 33.3%**, mean final ante **2.60 → 1.83**. Determinized search agrees with the
clairvoyant agent's own trajectory on **10.5%** of `play` decisions and **0.7%** of `discard`
decisions, but 82.0% of `skip_blind` and 77.6% of `leave_shop`. The learned policy *shape* was
economic; the hand *skill* belonged to the oracle. Honest `real1` sits at roughly scripted-greedy
level after 106 generations.

**The analytic EV player.** That measurement prompted the question — if Balatro hand play is an
EV calculation over a known chance distribution, is tree search the right instrument at all? —
and the answer, so far, is no. An analytic player that computes hand EV by expectimax over the
real draw distribution (structural candidate set, engine dry-run scoring, no network, no search)
clears ante 1 on **95.2%** of the 126 corpus seeds at its fast budget and **96.0%** at its full
budget, against scripted greedy's **31.7%**, and **beats determinized `real1` 57 of 58 decided
head-to-head matches (98.3%, 95% CI [94.8, 100])** — at 4.6 ms per hand decision against the
determinized search's 268 ms. The fast budget is statistically indistinguishable from the full
budget (76.5 ms) head-to-head: 26/60, CI [31.7, 56.7], which contains 50%.

**Current program: V(state) = P(win) plus counterfactual pairs.** The only learned object now is
a value function `V(state) = P(win the MLB match)`, 4,996,789 parameters over a frozen state
spec, trained on determinized Monte-Carlo rollout labels of the analytic policy. The first
corpus is 51,024 labels (2,126 jobs, 0 failures, 2.95 h at 287.9 labels/min on 16 workers);
best held-out Brier 0.060 / AUC 0.784 / ECE 0.021. **It is not yet good enough to steer
decisions**: argmax-V as a full policy loses to the hand-written rules 2 wins in 60 matches,
because per-action EV gaps (≪ 0.05) sit below label noise (mean CI ±0.24 at 8 rollouts). The
current round attacks exactly that: a within-state pairwise ranking loss trained on
counterfactual action pairs rolled out on *shared* determinized worlds (variance reduction
**2.54×** on hand/nemesis states, 1.78× on the uniform state mix), V restricted to the
expectimax leaf, and auxiliary heads on rollout intermediates. A larger paired-label campaign is
running.

---

## Status (2026-08-25)

| | |
|---|---|
| Strongest player in the repo | the **analytic EV player** (`mp/ev/`) — no network, no search |
| Baseline | `real1` MCTS, 106 generations — **must be cited determinized**; clairvoyant numbers are not evidence |
| Learned V | 5M params, trained on 51k labels; **not yet a policy** (argmax-V 2/60 vs rules) |
| V-v2 round | **complete 2026-08-26**: 42,468 counterfactual pairs (realized VRF 2.08×); argmax-V vs rules **2/60 → 12/60**, with a same-data no-pairs control staying at 2/60 — the gain is the ranking loss, not the data. V at the expectimax leaf remains a null (24/60) |
| Full-run wins | the 126-seed gate's `won` metric: **1.6%** [0.0, 4.0], rising to **3.2%** with the extraction layer. (The top-level README's reference points: skilled human ≈ 70%, random < 0.01%) |

Nothing in this directory claims a strong Balatro player. The strongest claim here is about
**measurement**: the engine is bit-exact against the real game's RNG on 126 seeds through ante 8,
and every player is evaluated without access to the future.

---

## Claims ledger

Every headline claim, the file that holds its evidence, and the command that regenerates it.
Numbers are quoted exactly as committed. Rows marked 🔒 need a checkpoint that is **gitignored**
(`mp/agent/runs/real1/latest.pt`, `mp/ev/runs/v_full_best/ckpt_0001000.pt`) — the command is
correct but reproducing the number from a clean clone means retraining first.

| Claim | Evidence | Repro |
|---|---|---|
| RNG core is bit-exact vs LuaJIT: 68,640 chain values, 0 mismatches; `seed(0.0)` reproduces LuaJIT's fixed constants | [`rng/NOTES_CORE.md`](rng/NOTES_CORE.md) | `python -m pytest mp/tests/test_rng_core.py` (17 tests; needs `lupa` + the game install. `MP_RNG_NO_ORACLE=1` → 14 pass / 3 skip off the cached 2.3 MB fixture) |
| Generation layer matches the real Lua: 0 mismatches over 30 seeds × 7 scenarios × 3 antes | [`rng/NOTES_GEN.md`](rng/NOTES_GEN.md), [`rng/GENERATION_SPEC.md`](rng/GENERATION_SPEC.md) | `python -m pytest mp/tests/test_generate_oracle.py` |
| Corpus is exact vs two independent analyzers: 126 seeds, antes 1–8, ~2,970 fields/seed | [`oracle/SOURCES.md`](oracle/SOURCES.md), [`oracle/ground_truth/`](oracle/ground_truth/) | `python -m mp.oracle.parity_check --antes 1-8 --variant faithful` → **126/126** |
| Every published analyzer omits the `used_jokers` rule: 953 fields differ corpus-wide (222 same-shop dups, 319 pack/shelf collisions, 412 downstream shifts); analyzer-as-published scores 16/126 through ante 3 | [`CAMPAIGN_LOG.md`](CAMPAIGN_LOG.md) § Phase 0 Agent D | same command, `--variant` toggles the rule |
| The rule is confirmed against the **running game** (seed `7I4M53DL`: Arcana 3rd card = The Lovers; buy+sell Hierophant → it returns as the 3rd card) | [`CAMPAIGN_LOG.md`](CAMPAIGN_LOG.md) § 2026-08-21 Tagg live checks | manual, 2 minutes in-game — the procedure is in that entry |
| The engine reproduces 126 real seeds exactly through ante 8 (from 0/126 before delegation) | [`engine/DELEGATE_NOTES.md`](engine/DELEGATE_NOTES.md), [`engine/SWEEP_NOTES.md`](engine/SWEEP_NOTES.md) | `python -m mp.oracle.engine_parity --antes 1-8 --rerolls 5 --quiet` → **126/126** |
| Engine suite **1,651** tests green (847 at Phase-1 kickoff); rng + generation oracle + invariants + reachability suite **1,073** green | [`CAMPAIGN_LOG.md`](CAMPAIGN_LOG.md) § 2026-08-25 landings | `python -m pytest mp/engine/tests -q` · `python -m pytest mp/tests -q` |
| Same-seed two-player queue alignment: 0 unexplained RNG-position differences over 30 matches; SHARED + VOUCHER + UNKNOWN keys all 0 | [`tests/GATE_NOTES.md`](tests/GATE_NOTES.md) | `python -m pytest mp/tests/test_mlb_match_gate.py -q` (507 tests, ~27 s) |
| ρ(h) is **flat, not decaying** (design doc guessed 0.3–0.5 by h=4–8): buy_slot0 0.876 → 0.870, skip_small 0.805 → 0.772, reroll_once 0.606 → 0.728 from h=1 to h=8; paired-seed VRF 2.5–10× at every horizon | [`results/rho_decay_*.json`](results/), [`eval/EVAL_NOTES.md`](eval/EVAL_NOTES.md) | `python mp/eval/rho_decay.py --all --n-extra-seeds 24` (150 seeds × 3 perturbations, n_boot 2000) |
| The NN was never the MCTS wall: batching cuts the forward pass 26× (1.25 → 0.048 ms/leaf) but only 1.4× end-to-end (531 → 761 sims/s, K=32 CUDA); ~59% of time is `clone()` + `step()` | [`agent/BATCH_NOTES.md`](agent/BATCH_NOTES.md) | `python mp/agent/benchmarks/bench_batched.py` |
| Cold MCTS cannot clear ante 1: 4,350 episodes → 46 clears (1.0%), value loss 0.0008; raising to 200 sims cleared 0/53 | [`CAMPAIGN_LOG.md`](CAMPAIGN_LOG.md) § 2026-08-22 19:30, [`agent/PRIOR_NOTES.md`](agent/PRIOR_NOTES.md) § 0 | the baseline arm of the row below: `python mp/agent/scripts/train_cold.py --minutes 10 --device cpu --ruleset vanilla --encoder set --sims 40 --heuristic-prior 0 --max-hand-candidates 0 --run-dir mp/agent/runs --run-name w0_cold` |
| A heuristic hand prior fixes the constant-target failure: ante-1 clear 0.0% → **15.7%** (λ 0.8, τ 0.35, K 32), value loss 0.0023 → 0.0251; the **top-K mask alone does nothing** (0.4%); more sims still does not help (80 sims → 6.9%) | [`agent/PRIOR_NOTES.md`](agent/PRIOR_NOTES.md) § 4 (full 9-arm table), § 5 (the settings) | one arm: `python mp/agent/scripts/train_cold.py --minutes 10 --device cpu --ruleset vanilla --encoder set --sims 40 --heuristic-prior 0.8 --heuristic-tau 0.35 --max-hand-candidates 32 --run-dir mp/agent/runs --run-name w0_armF` (baseline = same command with `--heuristic-prior 0 --max-hand-candidates 0`), then `python mp/agent/scripts/w0_smoke_report.py mp/agent/runs w0_` |
| 🔒 **Clairvoyance:** the same 106-generation checkpoint drops from 63.3% → 33.3% ante-1 clear and 2.60 → 1.83 mean final ante when denied the future; agreement with its own clairvoyant trajectory is 10.5% on `play`, **0.7% on `discard`**, 82.0% on `skip_blind`, 77.6% on `leave_shop` | [`results/clairvoyance_2026-08-23.md`](results/clairvoyance_2026-08-23.md) | `python mp/agent/scripts/measure_clairvoyance.py --checkpoint mp/agent/runs/real1/latest.pt --n-seeds 30 --sims 40 --processes 10 --determinize-mode per_sim --determinize-seed-base 0 --max-steps 20000 --n-boot 2000 --out-json mp/results/clairvoyance_2026-08-23.json --out-md mp/results/clairvoyance_2026-08-23.md` (the committed run used `--processes 30`; 10 is the safe ceiling for torch-loading pools on a 47 GB box) |
| **Analytic EV player, 126 seeds:** ante-1 clear fast **95.2%** [91.3, 98.4] / full **96.0%** [92.1, 99.2] / scripted greedy **31.7%** [23.8, 39.7]; mean final ante 4.69 / 4.91 / 1.32; hand decision 4.60 ms / 76.52 ms / 2.45 ms | [`results/ev_player_gate_2026-08-23.md`](results/ev_player_gate_2026-08-23.md), [`ev/EV_NOTES.md`](ev/EV_NOTES.md) | `python mp/ev/gate_ev_player.py --procs 16` (~4 min on 16 cores) |
| Draw-order invariance (the player never reads the deck): 743/743 and 740/740 sampled states give an identical decision under a permuted `game.deck` | same gate file, § Draw-order invariance | same command |
| 🔒 **EV player beats determinized `real1` 57/58 decided matches (98.3%, [94.8, 100])**, lives margin +3.27, Nemesis win rate 92.3% (60 trials, both seatings, 2 undecided at the 4,000-step cap) | [`results/h2h_ev_full_vs_real1_det_30seeds.md`](results/h2h_ev_full_vs_real1_det_30seeds.md) | `python mp/ev/h2h.py --a ev:full --b real1:det --sims 40 --n-seeds 30 --procs 8 --max-steps 4000 --out-json mp/results/h2h_ev_full_vs_real1_det_30seeds.json --out-md mp/results/h2h_ev_full_vs_real1_det_30seeds.md` |
| Fast ≈ full head-to-head — 26/60 (43.3%, CI [31.7, 56.7], contains 50%) — at 4.60 ms vs 76.52 ms per hand decision | [`results/h2h_ev_fast_vs_ev_full_30seeds.md`](results/h2h_ev_fast_vs_ev_full_30seeds.md) | `python mp/ev/h2h.py --a ev:fast --b ev:full --n-seeds 30 --procs 4 --out-json … --out-md …` |
| **Negative:** the decision-statistics tier used as a shop *policy* loses to the hand-written rules — 8/60 (13.3%, [5.0, 21.7]), lives margin −2.87. Kept as the advisor's diagnostic layer only | [`results/h2h_ev_full_stats_vs_ev_full_30seeds.md`](results/h2h_ev_full_stats_vs_ev_full_30seeds.md), [`stats/STATS_NOTES.md`](stats/STATS_NOTES.md) | `python mp/ev/h2h.py --a ev:full+stats --b ev:full --n-seeds 30 --procs 4 --out-json … --out-md …` |
| Decision-statistics sweep, 126 seeds: analytic P(hit) inside the determinized-empirical CI 30/30; Standard packs are net-negative EV at every ante measured, antes 1–4 (−8.7, −8.4, −6.0, −4.7) | [`results/stats_sweep_2026-08-23.md`](results/stats_sweep_2026-08-23.md) | `python mp/stats/sweep.py --out mp/results/stats_sweep_2026-08-23.json --processes 16` |
| V label corpus: **51,024 labels / 2,126 jobs / 0 failures in 2.95 h** at 287.9 labels/min on 16 workers; independent-perspective sum-to-one 0.965 ± 0.025; truncation rate 0.000; label sd rises 0.21 → 0.39 from ante 1 to 5 | [`results/labels_full.json`](results/labels_full.json), [`ev/TRAINV_NOTES.md`](ev/TRAINV_NOTES.md) § 6 | `python mp/ev/scripts/gen_labels.py --run-dir mp/ev/runs/labels_full --seeds default+random:2000 --workers 16 --policy ev --budget fast --shop-tier rules --encoder v2 --n-states 12 --n-rollouts 8 --flush-jobs 32 --symmetry-jobs 24 --name full` |
| V (4,996,789 params) held-out **Brier 0.060 / AUC 0.784 / ECE 0.021**; overfits the 51k corpus by ~epoch 7 | [`agent/VALUE_NOTES.md`](agent/VALUE_NOTES.md), [`ev/TRAINV_NOTES.md`](ev/TRAINV_NOTES.md) | `python mp/ev/train_v.py --shards mp/ev/runs/labels_full/shards --run-dir mp/ev/runs/v_full --model set_value_net --max-steps 20000 --batch-size 256 --lr 3e-4 --warmup-steps 500 --eval-every 500 --checkpoint-every 2000 --device cuda --holdout-frac 0.1 --torch-threads 8` |
| 🔒 **Negative:** argmax-V as a full policy loses to the rules — **2 wins / 60 matches** (3.3%, Wilson [0.9, 11.4]), mean lives margin −3.15, 35,544 V calls, 0 errors. Per-action EV gaps (≪ 0.05) sit below label noise (mean CI ±0.24 at 8 rollouts) | [`results/tournament_v_v_full.json`](results/tournament_v_v_full.json) | `python mp/ev/scripts/tournament_v.py --checkpoint mp/ev/runs/v_full_best/ckpt_0001000.pt --seeds default:30 --workers 16 --threads 1 --name v_full` |
| 🔒 **Negative:** V at the expectimax leaf is a clean null against the same player without it — 24/60 (40.0%, [28.3, 51.7]); still 52/58 (89.7%, [81.0, 96.6]) vs determinized `real1`. Leaf cost 163.3 ms vs the 100 ms target (24 unbatched forward passes per decision) | [`results/h2h_ev_full_vleaf_vs_ev_full_30seeds.md`](results/h2h_ev_full_vleaf_vs_ev_full_30seeds.md), [`ev/LEAF_NOTES.md`](ev/LEAF_NOTES.md) | `python mp/ev/h2h.py --a "ev:full+Vleaf" --b "ev:full" --n-seeds 30 --procs 8 --max-steps 4000 --out-json … --out-md …` |
| Counterfactual pairs on shared determinized worlds reduce variance **2.54×** on hand/nemesis states (n=560, ρ +0.629) but only **1.78×** on the uniform state mix (n=1,301) — **below the 2× bar the brief set** — and 1.42× / 1.80× / 1.23× on shop / pack / blind_select. Only 5.7% of pairs resolve at 8 worlds | [`ev/PAIRS_NOTES.md`](ev/PAIRS_NOTES.md), [`results/pairs_s1.json`](results/pairs_s1.json) | `python mp/ev/scripts/gen_pairs.py --run-dir mp/ev/runs/pairs_s1 --seeds default+random:600 --n-states 6 --n-worlds 8 --workers 8 --probe-jobs 10 --reps 4 --flush-jobs 8 --minutes 70 --name s1` |
| Extraction / sandbag layer: ante-1 gate **bit-identical** (95.2 / 96.0), matches won 1.6% → 3.2%, dev slice +$12.50 at ante 6, per-decision cost **−0.9%** (no measurable overhead); head-to-head vs itself-without-the-layer is a wash, 50.4% [44.4, 56.7] over 252 matches. 4 engine fidelity bugs fixed against the Lua along the way | [`ev/EXTRACT_NOTES.md`](ev/EXTRACT_NOTES.md) § 0, § 8 | `python mp/ev/gate_ev_player.py --procs 8` · `python mp/ev/scripts/extract_dev_slice.py slice --procs 8 --to-ante 6 --seeds <slice>` |
| **Negative:** active label selection is a qualified no — disagreement ΔBCE −0.0012 ± 0.0010 (t = −1.29), error-proxy −0.0021 ± 0.0012 (t = −1.79) against a significance bar of 2.57 in absolute t. The error proxy costs 2.47× the rollouts and **ranks on label noise** (its rows alone: AUC 0.503). Disagreement is free and reallocates 31% of its budget to ante ≥ 5 (corpus 17.6%) unprompted | [`results/active_poc_2026-08-25.md`](results/active_poc_2026-08-25.md), [`ev/active_poc/NOTES.md`](ev/active_poc/NOTES.md) | staged CLIs in `mp/ev/active_poc/` (`gen_pool.py` → `stage_base.py` → `stage_select.py` → `stage_final.py`) |
| Auxiliary heads on rollout intermediates: instrumentation costs **0.37 ms per ~1,050 ms rollout (0.04%)** and the no-aux path is bit-identical to the pre-aux trainer; the ablation itself is a **null at proof scale** (Brier 0.0876 aux vs 0.0877 no-aux on 3,654 rows), with the money head reaching R² +0.54 | [`ev/AUX_NOTES.md`](ev/AUX_NOTES.md), [`results/aux_ablation.json`](results/aux_ablation.json) | `python mp/ev/scripts/gen_labels.py --aux ...` then `python mp/ev/train_v.py --aux ...` (§ AUX_NOTES 6.2) |
| 🔒 **The ranking lever works, attributed by control (2026-08-26):** 42,468-pair campaign (all 3,126 seeds, VRF **2.08×** realized, CRN and direct audits agree); argmax-V vs rules **12/60** for V trained with pairs + aux vs **2/60** for a same-data control without them (old V also 2/60) — data freshness contributed nothing; held-out resolved-pair accuracy 0.620 / 0.588 / 0.574. V at the expectimax leaf stays a null with the new V: 24/60, matching the old V exactly | [`results/pairs_pairs_v2.json`](results/pairs_pairs_v2.json), [`results/tournament_v_v_v2.json`](results/tournament_v_v_v2.json), [`results/tournament_v_v_v2_ctrl.json`](results/tournament_v_v_v2_ctrl.json), [`results/h2h_ev_full_vleaf_v2_vs_ev_full_30seeds.md`](results/h2h_ev_full_vleaf_v2_vs_ev_full_30seeds.md), [`CAMPAIGN_LOG.md`](CAMPAIGN_LOG.md) § 2026-08-26 | campaign: `python mp/ev/scripts/gen_pairs.py --run-dir mp/ev/runs/pairs_v2 --seeds default+random:3000 --seed-rng 42 --n-states 14 --n-worlds 8 --workers 16 --aux --per-kind '{"hand":5,"nemesis":4,"shop":2,"pack":2,"blind_select":1}' --mix '{"close_call":0.55,"greedy_vs_extract":0.30,"random":0.15}'` · then `train_v.py` both arms · then `tournament_v.py` per checkpoint |

---

## Self-audits that changed conclusions

Each of these overturned something this project had already written down.

1. **The sim audit overturned the project's own retrospective** (2026-07-29). A wiki
   cross-examination found no blind-clear money, per-blind deck resets, and order-independent
   joker scoring — so the "PPO search ceiling" conclusion rested on an invalid measurement.
   → [`results/SIM_AUDIT_2026-07-29.md`](../results/SIM_AUDIT_2026-07-29.md), and its own
   "where reading-only auditing was unreliable" table.
2. **A1 — every card-modifying tarot was a no-op.** Consumables were dispatched with no card
   target. Found not by reading but by a reachability probe asking whether each fix's
   precondition is ever met (`scripts/probe_fix_reachability.py`).
   → [`results/SIM_AUDIT_2026-07-29.md`](../results/SIM_AUDIT_2026-07-29.md) § A1.
3. **The clairvoyance measurement.** The MCTS docstring's claim that simulations see different
   RNG outcomes was true on the old engine and false on the new one; measuring the gap showed
   the trained agent's hand skill was the oracle's (0.7% discard agreement when the future is
   withheld). → [`results/clairvoyance_2026-08-23.md`](results/clairvoyance_2026-08-23.md).
4. **The Red Deck had 3 discards; the real game gives 4.** Every run in the project's entire
   history — including all V7/V8 numbers — had one discard too few.
   → [`engine/DECKS_NOTES.md`](engine/DECKS_NOTES.md).
5. **The `used_jokers` rule that every published seed analyzer omits.** Found twice
   independently, checked against two analyzers, and then confirmed in the live game — in both
   directions (blocking *and* the `Card:remove` clear).
   → [`CAMPAIGN_LOG.md`](CAMPAIGN_LOG.md) § Phase 0 Agent D and § 2026-08-21 live checks.
6. **`value_fn` argmaxed V at *every* decision, not just the leaf.** Caught while wiring the
   "V at the expectimax leaf" experiment — the pre-existing plumbing would have confounded the
   evaluation of that very lever. Fixed as `value_fn_leaf_only`.
   → [`ev/LEAF_NOTES.md`](ev/LEAF_NOTES.md) § 1.2.
7. **The error-proxy label selector was quantified as noise-chasing**, not merely
   non-significant: it picks near-0.5 maximal-aleatoric states, 54% of its "error" evaporates
   on re-measurement, and its own rows carry zero ranking information (AUC 0.503). It was
   dropped rather than shipped as a plausible-sounding heuristic.
   → [`results/active_poc_2026-08-25.md`](results/active_poc_2026-08-25.md) § 3.

Two more of the same kind, kept short: the V7 reward's dry-run scorer had **side effects** —
Space Joker wrote the live planet levels, so the reward's largest term had a systematically
inflated denominator for all of V7/V8 ([`engine/REPRO_NOTES.md`](engine/REPRO_NOTES.md)); and
`State.BOOSTER_OPEN` was **never entered anywhere in the engine**, making booster packs a pure
money sink for every historical run ([`engine/FORK_NOTES.md`](engine/FORK_NOTES.md)).

---

## Reader's map

**Where the narrative lives.** [`CAMPAIGN_LOG.md`](CAMPAIGN_LOG.md) — chronological, dated,
1,700+ lines, from 2026-08-20 to today. Every workstream logs its gates, its numbers, and a
"found, not fixed" list. If you read one file, read that one; if you read two, read this one
and that one.

**Where the results live.** [`results/`](results/) — one `.md` (human-readable tables) plus one
`.json` (full per-seed records) per measurement. Every `.md` names the exact command and seed
set that produced it.

**Where the design decisions live.** [`docs/`](docs/) — phase briefs (`PHASE2..PHASE5_*`), the
frozen [`STATE_SPEC_v1.md`](docs/STATE_SPEC_v1.md), the decision-economics and self-play
assessments, and [`FIELD_INSPIRATION_2026-08.md`](docs/FIELD_INSPIRATION_2026-08.md) (adjacent-field
techniques surveyed, with what was adopted and what was parked).

**Where the technical claims live.** Each package carries a `*_NOTES.md` written by the
workstream that built it, with an explicit "found, not fixed" section:

```
rng/     NOTES_CORE  NOTES_POOLS  NOTES_GEN  NOTES_ORDER  GENERATION_SPEC
engine/  FORK  REKEY  DELEGATE  EFFECTS  TAGS  SWEEP  DECKS  MLB  REPRO
agent/   AGENT  BATCH  PRIOR  SETENC  TRAIN  PARALLEL  DETERMINIZE  VALUE
ev/      EV  EXTRACT  TRAINV  PAIRS  RANK  AUX  LEAF  PROBE  ADVISOR
stats/   STATS        eval/ EVAL      replay/ REPLAY     tournament/ TOURNAMENT
tests/   GATE  HARNESS
```

### Low-signal areas — skip these, they are outputs, not work

These exist for good reasons but contain nothing a reader needs. A tool with a limited context
budget should spend it elsewhere.

| Path | Why it is low-signal |
|---|---|
| `mp/*/runs/`, `mp/*/checkpoints/`, `*.pt` | training / campaign outputs. Gitignored; large; regenerable |
| `**/__pycache__/` | bytecode |
| `mp/results/*.json` | the full machine-readable records behind the `.md` files. Read the `.md`; open the `.json` only to check a specific per-seed row |
| `mp/oracle/ground_truth/*.json` | the 126-seed oracle corpus (17 MB of generated fixtures). It is *evidence*, but it is data — the claim it supports is in `CAMPAIGN_LOG.md` and `oracle/SOURCES.md` |
| `mp/tests/fixtures/rng_ground_truth.json` | 2.3 MB cached LuaJIT oracle output, for running RNG tests without the game installed |
| Top-level `results/V1..V9*`, `checkpoints_*/`, `logs_*/` | the **pre-audit** BRL project. Superseded: `results/SIM_AUDIT_2026-07-29.md` explains why those numbers are not comparable to anything here |

**This rule applies to low-signal only.** The negative results above are load-bearing and are
meant to be read: argmax-V losing 2/60, the stats tier losing 8/60, V-at-leaf being a null,
active selection being a qualified no, the pairs lever missing its 2× bar, and honest `real1`
sitting at scripted-greedy level are all reported here deliberately.

### Deliberately absent — not missing, excluded by design

If a file-completeness check flags these, the check is wrong.

| Path | Why it is absent |
|---|---|
| `mp/_reference/balatro_src/` | Balatro 1.0.1o Lua **extracted from the local Steam install**. Copyright LocalThunk/Playstack. Gitignored (`mp/.gitignore`), never committed, never copied into deliverables. Algorithms are ported and cited by `file:line`; the Lua itself is read at test time from the local install and never vendored. Tests that need it skip with a reason when it is absent (`BALATRO_DIR` overrides the default Steam path) |
| `mp/oracle/blueprint_runner/vendor/` | clones of the third-party seed analyzers (Blueprint / TheSoul / Immolate). Gitignored (`mp/oracle/blueprint_runner/.gitignore`); `setup.ps1` re-clones them at the pinned commits recorded in `oracle/SOURCES.md` |
| The BalatroMultiplayer mod Lua | read from `%APPDATA%/Balatro/Mods/Multiplayer/` at test time to verify The Order and the MLB rules. Third-party; never copied into the repo |
| `mp/agent/runs/real1/`, `mp/ev/runs/` | checkpoints and campaign shards; gitignored. Rows marked 🔒 in the ledger depend on them |

---

## Layout

```
mp/
├── CAMPAIGN_LOG.md   the narrative — start here after this file
├── docs/             phase briefs, frozen state spec, design assessments, field survey
├── rng/              keyed pseudorandom core (pseudohash + LCG + LuaJIT math.random port),
│                     pools, key strings, and the generation layer (shops/packs/vouchers/...)
├── oracle/           126-seed ground truth + parity harnesses (analyzer-side and engine-side)
├── engine/           balatro_sim fork: game keys, delegated generation, keyed effect rolls,
│                     MLB ruleset, 15 decks, 8 stakes, determinization
├── agent/            MCTS + neural nets (the `real1` line), set encoder, SetValueNet (5M)
├── ev/               the analytic EV player, labels, V trainer, pairs, extraction, advisor CLI
├── stats/            decision statistics for packs / rerolls / vouchers (advisor layer)
├── eval/             eval harness, ρ-decay harness, transfer spread, target functions
├── tournament/       N-agent same-seed runner + N×N matrices
├── replay/           trajectory logging, exact replay, tagging, viz export
├── tests/            RNG + generation oracles, engine invariants, reachability, MLB gate
├── scripts/          mlb_match_demo.py
├── results/          measured results (.md + .json)
└── _reference/       extracted game Lua — GITIGNORED, never commit (see above)
```

## Ground rules for anyone working here

- Everything stays under `mp/`. BRL code, the top-level `results/`, and the top-level `README.md` are the
  historical record and are not rewritten (the surgical 2026-08 bridge at the top of that
  README is the exception).
- **Oracle first.** No number is reported before the thing producing it is checked against the
  real game — LuaJIT for RNG, the analyzer corpus for generation, the installed mod for MLB
  rules, and a hand-played run in the retail game for the two live checks.
- **No clairvoyant numbers.** Any evaluation of a search agent must determinize. Clairvoyant
  `real1` figures are in the log for the record and are not evidence about playing strength.
- `_reference/` is ported from, never copied or committed.
- Every workstream writes a `*_NOTES.md` with a "found, not fixed" section, and every gate is
  re-run by the lead before it is believed.
