# MP Rebuild — Campaign Plan

**2026-08-20.** Execution plan for the work in `MP_UPDATE_LIST_2026-08.md`. Format follows the validated autonomous-campaign pattern: locked goal, phase chain, gates, `CAMPAIGN_LOG`.

---

## 0. Correction to the timeline

My 6–9 week estimate was **sequential human working days reported as if they were calendar time.** That's the wrong unit for this work. Most of it is embarrassingly parallel:

- ~40 RNG call sites — independent
- ~20 correctness bugs — independent
- 9 sentinel-token jokers — independent
- 15 decks — independent
- Test generation — independent by construction

A fleet does those concurrently. **A few days of wall clock with overnight sessions is the right expectation.**

What I'd still hold: **the constraint isn't total work, it's the critical path.** A handful of things are genuinely ordered, and one of them determines whether the whole plan is a 3-day job or a 3-month one.

### The critical path

```
RNG core (pseudohash + LCG + key construction + exact pool orderings)
   │
   ├──► SEED-PARITY ORACLE ◄── the keystone
   │         │
   │         └──► automated verification for everything downstream
   │
   ├──► key all ~40 call sites          ─┐
   ├──► ante queues + pointers           ├─ all parallel, all oracle-verified
   ├──► resample/blocker queues          │
   └──► ~20 correctness bugs            ─┘
              │
              ▼
        MLB rules + decks  (partial oracle coverage — needs spot checks)
              │
              ▼
        infra: checkpoints, eval harness, N-agent runner
```

**Everything hinges on the oracle.** With byte-exact seed parity against Immolate/TheSoul, verification is a diff — mass-parallelizable, no human judgment, no "does this look right." Without it, every one of the ~20 bug fixes needs hand-written assertions and manual reasoning, and the base rate says you'd find 20 more.

So the honest version of my earlier estimate: **2–3 months was the no-oracle number. With the oracle it's days.** Your instinct is right, and the oracle is the thing that makes it right.

---

## 1. Phase chain

### Phase 0 — Oracle spike (GATE — do this alone, first)

**One session. Do not start Phase 1 until this passes.**

Build only:
- `pseudohash(str)` — Balatro's string hash
- LCG step: `x ← |round((2.134453429141 + x·1.72431234) mod 1, 13dp)|` — the `%.13f` rounding is load-bearing
- Key construction: type + source + ante + resample counter
- **Exact pool orderings** transcribed from Immolate `lib/items.cl` — joker pools by rarity, tarots, planets, spectrals, vouchers, bosses, tags. Draw is `pool[floor(r · len(pool))]`, so **order is the whole ballgame**. Your existing catalogue has all 150 jokers but almost certainly not in the game's internal order.

**Exit gate:** given 10 known real Balatro seeds, reproduce ante-1–3 shop contents, pack contents, vouchers and boss blinds **exactly**, cross-checked against Immolate or a public seed analyzer.

**If the gate fails, stop and replan.** Known hazards: `pseudohash` can produce NaN on "toxic seeds" (a real documented edge case), float precision is unforgiving, and the OpenCL→Python port needs care. This is the single highest-risk item in the campaign and it's ~500 lines. De-risk it standalone rather than discovering it mid-fleet.

---

### Phase 1 — Engine correctness (fleet, oracle-verified)

Parallel workstreams, each verified against the Phase 0 oracle:

| # | Workstream | Scope |
|---|---|---|
| 1 | Merge `fix/sim-fidelity-2026-07`, clean `%TEMP%\brl-prefix` worktree | prerequisite, do first |
| 2 | Thread keys through `game.py` + `scoring.py` call sites | ~12 sites |
| 3 | Thread keys through `shop.py` | ~12 sites |
| 4 | Thread keys through `consumables.py` | ~18 sites |
| 5 | Thread keys through `jokers/*` via `rng_of` | ~20 sites |
| 6 | Ante queue: generation, per-player pointer, +2 per reroll, The Order switch | |
| 7 | Resample/blocker queues, rarity-keyed, **in-place replacement** | see warning below |
| 8 | Duplicate suppression + make Showman real | currently a no-op |
| 9 | The 9 sentinel-token jokers (Vagabond, Superposition, Cartomancer, Seance, Sixth Sense, Riff-Raff, Certificate, DNA, Perkeo) | biggest single chunk |
| 10 | Remaining §7 bugs: 8 Ball, Gros Michel/Cavendish destruction, Lucky Cat hook, Idol init, 6 dead hooks, 3 `on_init` NameErrors, boss exhaustion pool, voucher ante gating, Hone/Glow Up, pack dedupe, standard-pack modifiers, rarity table, 6 double-listed jokers | mechanical |
| 11 | §3 reproducibility: clone state in `_best_hand_score`, thread rng in `env_sim`/`env_v5` | small, high value |

> ⚠️ **Workstream 7 is the one to get right.** Do **not** filter-the-pool-then-draw, and do **not** delete-and-shift. Blocking must substitute the slot *in place* from a rarity-keyed side stream, leaving every other slot in this ante and all later antes untouched. Get this wrong and two players with different collections desync — which is the entire point of the rebuild.

**Exit gate:**
- Oracle parity holds on 100+ seeds through ante 8
- **Stream-independence assertions pass**: triggering a Lucky hit does not change the next shop; owning a joker does not shift any other slot
- **Reachability probe passes**: every joker and consumable measurably changes state (generalizes the probe that caught A1)

---

### Phase 2 — MLB rules + decks (fleet, partial oracle)

| # | Workstream |
|---|---|
| 1 | Nemesis blind; target = opponent live score; early-win rule |
| 2 | 4 lives, decremented on **any** blind lost; first to 0 loses; comeback money |
| 3 | Two-player coordinator, ante lockstep, boss→PvP from ante 2 |
| 4 | Red deck (baseline) |
| 5 | Checkered deck (composition change) |
| 6 | Plasma deck (chips/mult balance, ×2 blind size — different scoring regime) |
| 7 | White stake |
| 8 | **TAGS — promoted into the MVP, see below** |

> **Why tags moved up.** Confirmed strategic evidence: in Black deck / Gold stake runs
> at high antes, strong players **skip small and big blinds straight through to the PvP
> blind** rather than risk failing them. Skipping costs no life; failing costs one. So
> skip is the primary *life-preservation* tool in the endgame, and the compensation for
> skipping is the tag. That means at high antes a large share of your economy comes from
> tags rather than blind rewards.
>
> The sim pays a flat `+$5` on skip (`game.py:835-846`) and has no tag system at all, so
> **the entire high-ante endgame is currently unrepresentable.** This is not a deferred
> nice-to-have; without it Phase 4's transfer measurement will mis-score exactly the
> cells where policy shape differs most. Note Attrition bans the Boss Tag (no boss blind
> to skip), leaving ~23 tags.
>
> It also sharpens a shortlist decision: because Balatro shows you which tag you'd get
> *before* you skip, "skip for this specific tag vs. play the blind" is a clean, fully
> observable, well-defined choice — a top-tier candidate for the curated Monte Carlo list.

Oracle covers shop/pack generation here but **not** deck semantics or MLB rules. Those need spot checks against the real game.

**Exit gate:** two agents on one seed play a full MLB match, lives resolve correctly, and their shop queues stay aligned except where each player's own rerolls and blocked slots explain the difference.

---

### Phase 3 — Infrastructure (fleet)

| # | Workstream |
|---|---|
| 1 | Checkpoint saving in `train_cold.py` (currently saves nothing — blocks the opponent pool) |
| 2 | Eval harness |
| 3 | N-agent same-seed runner |
| 4 | **Full N×N comparison matrix** extraction at each nemesis blind (4,950 comparisons from 100 rollouts, not 50) |
| 5 | Batched NN inference + tree reuse (the ~745 sims/sec bottleneck) |
| 6 | ρ-decay measurement harness (horizons 1/2/4/8 antes) |

**Exit gate:** 100 agents run one seed end-to-end, the N×N matrix is produced, checkpoints save and reload.

---

### Phase 4 — First measurement, not training

Run the ρ-decay experiment (half a day) and the three-deck transfer spread (Red / Checkered / Plasma at White stake). **That's the decision gate from the assessment** — if a Red-trained policy collapses on Plasma, you've learned the shape of the problem before building the other 12 decks and 7 stakes.

---

## 2. What genuinely needs you

Three things, all small:

1. **Verify the regular-blind life rule empirically.** Two clients, practice lobby, deliberately fail an ante-2 small blind, watch the counter. ~10 minutes. Everything in Phase 2 depends on it and it's server-side, so no agent can read it out of source.
2. **Phase 0 gate decision.** If seed parity doesn't land, the plan changes materially — that's a judgment call, not an agent call.
3. **Spot-check Plasma and Checkered** against the real game once implemented. No oracle covers deck semantics.

Everything else is agent work with `CAMPAIGN_LOG` entries per phase.

---

## 3. Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| **Oracle can't reach exact parity** (float precision, NaN seeds, pool-order errors) | Campaign reverts to the 2–3 month hand-verified path | Phase 0 as a standalone gate. Fail fast on 10 seeds before committing a fleet. |
| Pool orderings transcribed wrong | Silent, subtle parity failures | Diff every pool against Immolate `items.cl` as a test, not by eye |
| In-place replacement implemented as filter-then-draw | Players desync; MP premise broken again, exactly like H3 | Explicit invariant test in Phase 1 gate |
| More unknown bugs (base rate is high — 8 found in July, ~20 in August) | Schedule slip | The oracle *is* the mitigation; it finds these automatically |
| 100 identical agents → degenerate matrix | Wasted compute, V8 Run 1 repeat | Heterogeneous population: historical checkpoints + MCTS root Dirichlet (α=0.03). Never temperature asymmetry. |
| Deck/stake semantics have no oracle | Wrong rules ship silently | Manual spot checks; keep Phase 2 to 3 decks until validated |

---

## 4. Realistic wall clock

| Phase | Sessions | Notes |
|---|---|---|
| 0 — Oracle spike | 1 | **Gate. Highest risk.** |
| 1 — Engine correctness | 1–2 | Largest fleet; fully oracle-verified |
| 2 — MLB + 3 decks | 1 | Partial oracle |
| 3 — Infra | 1 | Independent of 1–2, could run concurrently |
| 4 — First measurement | 1 | ρ decay + transfer spread |

**≈4–6 sessions, 2–4 days wall clock**, most of it unattended. That matches your read, and Phase 3 can run in parallel with Phases 1–2 since it touches different files.

The number that stays true from my earlier estimate is the *total work*, not the *elapsed time* — and the oracle is what converts one into the other, because it removes the human from the verification loop. That's why it's Phase 0 and why it's a hard gate.
