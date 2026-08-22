# Training Design: Horizon, Timeline, Encoding, Starts, and the N-Agent Tournament

**2026-08-20.** Fourth companion doc. See also `MP_SELFPLAY_ASSESSMENT_2026-08.md`, `MP_UPDATE_LIST_2026-08.md`, `MP_DECISION_ECONOMICS_2026-08.md`.

---

## 1. You're right about paired seeds — and it points at a specific fix

The objection: *after a few antes, small differences lead to different joker lineups, different money, different strategy.* Correct. The variance reduction I quoted (3–10×) assumed a correlation ρ between arms that stays high, and **ρ decays with horizon** as the two trajectories decouple. This is a known limitation of common random numbers: shared randomness only helps while the trajectories stay near each other.

But the decay is not to zero, and the reason matters. Even after the arms diverge in *what they buy*, they're still drawing from **the same underlying queue**, facing the same blind sizes and the same boss sequence. What decouples is which slice of that shared structure each arm consumes. So ρ falls from ~0.9 at the decision point to maybe ~0.3–0.5 several antes out — still worth ~1.4–2×, not the 3–10× I quoted for short horizons.

**Don't take those numbers on faith — measure ρ.** It's the cheapest useful experiment in this whole plan: run paired arms, compute the correlation of outcomes at horizons of 1, 2, 4, 8 antes. That gives you the real variance-reduction factor *and* tells you what horizon to measure at. Half a day, and everything downstream depends on it.

### The real consequence: measure short-horizon outcomes and bootstrap

If ρ decays with horizon, then **don't use match outcome as your measurement target.** Use the log-score margin at the *next* nemesis blind — one to two antes out, where ρ is still high — and then account for the carried-forward state separately.

That second half is exactly what a **value function** is for. The choice is the classic bias–variance tradeoff:

- **Monte Carlo to termination**: unbiased, variance grows with horizon. This is what your objection is describing.
- **Bootstrapped / TD**: measure a short horizon, then add `V(s')` for the state you end up in. Variance stays bounded; you pay a bias term that shrinks as `V` gets better.

So: **your objection is the argument for a learned value function rather than pure Monte Carlo rollouts.** Which is convenient, because it's also the argument for MCTS (§5) — search bootstraps on `V` by construction.

Practical form for a decision study:

```
value(decision) = E[ log-score margin at next PvP blind ]        # measured, high ρ, low variance
                + γ · E[ V(state carried forward) − V(baseline) ] # bootstrapped
```

The first term is what your Monte Carlo shortlist measures. The second is what the network is for. Neither alone is sufficient, and that's the honest answer to the objection.

---

## 2. Timeline to "ready for training"

Minimum viable = correct RNG and queues, MLB rules, three decks at White stake, and the infrastructure to run and evaluate. Estimates are **focused working days**, not calendar days.

| Work | Days |
|---|---|
| Merge `fix/sim-fidelity-2026-07`, clean stale `%TEMP%\brl-prefix` worktree | 0.5 |
| §3 reproducibility: clone state in `_best_hand_score`, thread rng in old envs | 1 |
| §1 RNG core: pseudohash + LCG (+ xoshiro for shuffle), validated vs. known seeds | 2–3 |
| §1 thread keys through ~40 call sites (`game`, `shop`, `consumables`, `jokers/*`) | 4–6 |
| §2 ante queue, per-player pointer, +2 per reroll | 2–3 |
| §2 resample/blocker queues, rarity-keyed, in-place replacement | 2–3 |
| §2 duplicate suppression + Showman (currently a no-op) | 1 |
| §7 the ~20 correctness bugs (the 9 sentinel-token jokers are most of it) | 5 |
| §4 MLB rules: nemesis blind, lives, coordinator, early-win rule | 3–4 |
| §5 three decks (Red / Checkered / Plasma), White stake only | 2–3 |
| §8 checkpoint saving, eval harness, N-agent grid runner | 5 |
| Smoke + validation harness (see below) | 4–5 |
| **Subtotal** | **32–40** |
| Discovery multiplier (+30–50%) | **42–60** |

**Why the multiplier is not padding.** The July audit fixed 8 critical bugs. This audit found ~20 more that all 828 tests pass with. The base rate of unknown-unknowns in this codebase is demonstrably high, and you will find more while touching every RNG call site.

**So: ~6–9 focused weeks. Realistically 2–3 months part-time**, which is what it will be given MATS (Sep 4 / 6), ERA (Sep 13) and PhD applications. Full 15 decks + 8 stakes + stickers + tags adds **3–4 more focused weeks** on top.

### Smoke testing is the part that changes character

Your 828 tests pass with every bug in §7 present, so the current test *style* doesn't catch this class. You need three new kinds:

1. **Seed-parity tests against an oracle.** This is the big one, and it's the payoff for choosing byte-exact RNG. Once the sim reproduces real Balatro seeds, you can auto-generate thousands of cases from Immolate/TheSoul as ground truth and diff shops, packs, vouchers and bosses. Testing goes from "write assertions by hand" to "diff against an oracle." **This alone would have caught most of §7 automatically**, and it's why the 2–3 extra days for option (b) in the update list is the best-value work on the list.
2. **Stream-independence tests.** Assert that triggering a Lucky hit does *not* change the next shop. Assert that owning a joker does *not* shift any other slot (in-place replacement). These are the multiplayer-critical invariants and they're one-liners once keys exist.
3. **Reachability probes.** Generalize the technique that found A1 — a contradiction between "28% of episodes used a tarot" and "0% ended with a modified card." For every joker and consumable, assert it measurably changes state. That finds dead code that unit tests bless.

---

## 3. Encode vs. let the model find it

The right criterion isn't "is this play good" — it's **what kind of limitation is stopping the agent from finding it.**

**Encode when the decision is exploration-limited:**
- **Rare** — the agent won't see it enough times for gradient to accumulate (Wraith, Showman, Legendary drops)
- **Discrete and nameable** — a specific interaction with a clear trigger
- **Anti-greedy** — every partial execution scores *worse* than the alternative, so the gradient points away (Death→Fool)
- **Independently computable** — you can Monte Carlo it in isolation

**Let it learn when the decision is capacity- or weighting-limited:**
- **High-frequency** — happens every hand or shop, so there's plenty of gradient
- **Graded** — "how much is this joker worth given my board" has no crisp rule
- **High-dimensional in context** — depends on the whole board in ways you can't enumerate
- **Compositional** — five jokers interacting is not a list of pairwise rules

### The non-arbitrary test

You don't have to guess which bucket something is in. **Log the frequency of each named play during training.** Anything sitting at ≈0% after training is exploration-limited → encode it. Anything the agent does often but badly is weighting-limited → let it learn, and check the value head instead. That's measurable, and it's the same diagnostic that caught A1.

### On "encode as many as possible and add more as we go"

The instinct is right but the *mechanism* matters, because hard-coding has three failure modes: it caps performance at your own knowledge, individually-good heuristics can conflict, and encoded rules become load-bearing and hard to remove.

**Encode as a prior, not a constraint.** Three concrete ways, all of which preserve the agent's ability to overrule you:

1. **Policy prior in the search.** Bias MCTS's prior toward the encoded move; search overrides it when the value disagrees. This is the AlphaGo recipe and it's the right one — nothing is capped, and a *wrong* heuristic costs search efficiency, not correctness.
2. **Macro-actions as options.** Add "hold Death for Fool" as an *available* action, not a mandatory one. Expands the action space rather than pruning it.
3. **Potential-based reward shaping.** If you must shape rewards, use the potential-based form (Ng, Harada & Russell 1999): `F(s,s') = γΦ(s') − Φ(s)`. It provably leaves the optimal policy unchanged. Every ad-hoc shaping term in V1–V8 lacked this guarantee, which is one reason six retunes kept moving the behaviour without moving the outcome.

With prior-not-constraint, your "add more as we find them" loop is safe: a bad addition wastes some search, it doesn't corrupt the objective.

---

## 4. Training from chosen starting points — how it actually transfers

The technique has names: **exploring starts** / **starting-state distributions** (Sutton & Barto), and **reverse or backward curriculum** when you progressively move the start earlier.

### The mechanism, and why "transfer" isn't really a separate step

The thing to internalize: **you train one network on *states*, not on trajectories.** The value/policy network is a function `V(s)`, `π(s)`. If you feed it states drawn from "ante 4, Wraith on offer," it learns the right outputs for that region of state space. Later, when you run a full game from ante 1, the agent *reaches* those states and the network already knows them.

There is no separate transfer step. It's the same function being queried at the same inputs. That's the whole trick, and it's why this works at all.

### The caveat that matters more than the technique

**The starting-state distribution must cover the distribution the agent actually encounters.** If you train only from hand-picked Wraith states, the network is calibrated on a region real runs may rarely reach, and you get distribution shift — confident, wrong values in the states that actually matter. Standard fixes, all of which you should use:

- **Mix.** Something like 70% normal ante-1 starts, 30% injected states. Never 100% injected.
- **Generate the injected states from the current policy's own rollouts**, not hand-built ones, so they're on-distribution by construction.
- **Anneal the injection rate down** as training proceeds.

### You already have the machinery

`clone()` in the `balatro-mcts` fork (`game.py:291-389`) is exactly snapshot/restore, it's test-pinned including the subtle deck-aliasing invariant, and it runs at 16.6k clones/sec. The workflow is: run the current policy, snapshot every state where Wraith is offered, then restart from those snapshots with both branches forced. That's a few dozen lines on top of what exists.

---

## 5. PPO → MCTS, and the thing that makes MP and MCTS fit together

Agreed on dropping PPO, and the evidence is already in the repo: V7 Run 7 scaled the network 5.5× to 13.6M params and **reproduced the same plateau shape**. That's the signature of an exploration bottleneck, not a capacity one — and PPO's exploration mechanism (perturb action probabilities, keep what scores better) is precisely what fails on the anti-greedy multi-step plans in §3. Search doesn't have that failure mode: it evaluates sequences directly rather than needing to stumble onto them.

### But MCTS had its own blocker — and multiplayer removes it

The 2026-05-10 cold runs: 2,248 episodes, **0 wins**, and the value loss collapsed to ≈0.002 because the shaped target `z ≈ 0.09` was near-constant. Of course it was — **in a game you never win, the value target has no variance, so the value head learns nothing.** And AlphaZero-style search is only as good as its value function. That's why the cold start went nowhere.

Now: in self-play multiplayer, every nemesis blind is **~50/50 by construction**. The value target has real variance from iteration one, at every ante, forever.

> **The value-head collapse that killed the MCTS cold start is a single-player pathology, and multiplayer removes it.**

This is the strongest technical reason to combine the two threads. Neither MP-without-search nor search-without-MP fixes the other's blocker; together they do. Multiplayer supplies the learning signal that AlphaZero needs, and search supplies the exploration that multiplayer's dense signal can't provide on its own.

Practical notes: batched NN inference and tree reuse are both still on the deferred list, and the measured ~745 sims/sec was bottlenecked by un-batched per-leaf forward passes — those are the obvious wins before any long run. And `train_cold.py` still saves no checkpoints, which blocks the opponent pool.

---

## 6. The N-agent tournament — good idea, and it's better than you framed it

Running 100 agents on one seed instead of 1v1 is right, for a reason beyond parallelism: **it amortizes the seed.** One seed currently yields one match. With 100 agents it yields far more.

### Extract the full matrix, not 50 pairings

Your option (b) — randomly pair into 50 1v1s — is faithful and correct. But you can do much better for free. Since all 100 agents played the same seed, **you know every agent's score at every blind**, so you already know who would have beaten whom in *every* pairing:

> 100 rollouts → **4,950 pairwise comparisons** per blind, not 50.

That's ~100× the comparison density for the same compute. Random pairing throws away 99% of the information you already paid for. Build the full N×N comparison matrix at each nemesis blind — it's a sort, not a simulation.

That matrix is also **layer 1 directly**: the score distribution at each ante, per deck/stake cell. So the N-agent run isn't just a training mechanism, it's the measurement instrument from the assessment. One mechanism, both jobs.

### On option (a) — bottom half loses a life

This changes the game, and it's worth being explicit about how. Ranking against a *population median* is not the same as beating a *specific opponent*:

- Against a specific opponent, when you're behind you want **more** variance.
- Against a population median, you want to be **reliably** above median.

So option (a) would train a systematically different risk posture than real MLB. It's not faithful.

But it's not useless either — it's tournament/population selection, and it's smoother and lower-variance than pairwise. So use both, for different jobs:

> **Population rank → value target** (dense, low variance, stable).
> **Pairwise matrix → policy objective** (faithful, strictly zero-sum).

### The failure mode to design against

100 copies of one network on one seed will play near-identically — that's V8 Run 1's 74% draws, and the N-agent setup makes the degeneracy *more* visible, not less. A comparison matrix of 100 identical agents carries almost no information.

So the population must be **heterogeneous by construction**:
- Sample opponents from a pool of **historical checkpoints**, not 100 copies of current weights.
- MCTS root Dirichlet noise (already implemented, α=0.03, calibrated for ~436-action states) gives per-agent divergence from a single network.
- Vary search budget or temperature across the population.

Do **not** reach for V8 Run 2's temperature-asymmetry hack — it breaks symmetry and makes agents non-comparable, which is fatal once you're building an N×N matrix.

### Compute

100 concurrent envs is well within what V7 already ran (16–20 workers × N envs). The cost is ~100× env time per seed, but you get ~4,950× the comparisons, so information per unit compute goes up substantially. The binding constraint will be NN inference, not the sim — which is the same batching problem already on the deferred list.

---

## 7. Summary of what changed in this doc

- Paired-seed variance reduction decays with horizon; **measure ρ empirically** (half a day) and measure at short horizons, bootstrapping the rest with `V`.
- **~6–9 focused weeks / 2–3 months part-time** to be training-ready on three decks; +3–4 weeks for the full grid.
- Encode what's **exploration-limited**, learn what's **weighting-limited**, decide by logging play frequency; encode as **prior, not constraint**.
- Starting-state training needs no separate transfer step — it's one network over states — but the start distribution must cover what the agent actually reaches.
- **MP removes the value-head collapse that killed the MCTS cold start.** That's the argument for doing both together.
- N-agent same-seed: take the **full N×N matrix** (4,950 comparisons, not 50), use population rank for the value target and pairwise for the policy, and make the population heterogeneous or it degenerates.
