# Decision Economics: Sample Complexity, Learnability, and the PvP Decision Frame

**2026-08-20.** Third companion doc, with `MP_SELFPLAY_ASSESSMENT_2026-08.md` and `MP_UPDATE_LIST_2026-08.md`.

---

## 1. The "blocker queue" — you're right, it's the resample queue

Your description is accurate, and the mechanism has a name. When Balatro tries to place an item that can't appear — you already own it, or it's locked on the profile — it does **not** delete the entry and pull the rest of the queue forward. It:

1. Marks that entry unavailable,
2. Redraws **that one slot** from a **separate resample queue**,
3. And if the resample also fails, creates **additional resample queues** until one succeeds.

Two refinements to your model:

- **Each joker rarity has its own resample queue.** It isn't one blocker stream, it's a family indexed by rarity and by resample depth.
- **The critical property is in-place replacement.** The blocked slot is substituted *in place*; every other item in that ante — and every later ante — stays exactly where it was. Blocking does not shift the main queue at all.

And yes: **Showman** removes the blocking, for Jokers, Planet cards *and* Tarots. With Showman you consume the main-queue entry that would otherwise have been skipped. So Showman doesn't just "allow duplicates" — it changes **which stream you draw that slot from**.

### Why this matters more in multiplayer than single-player

In-place replacement is exactly what makes same-seed multiplayer coherent. If I own Blueprint and you don't, and Blueprint comes up in the shared queue, I get a resample-stream joker in that slot and you get Blueprint — but **every other slot, for both of us, stays identical.** Divergent collections don't desync the queue.

That's a design constraint on the rebuild, and it's easy to get wrong:

> ⚠️ When implementing §2 of the update list, do **not** filter the pool and then draw, and do **not** delete-and-shift. Both break in-place replacement and will desync two players who own different jokers. Draw from the full pool, detect the block, resample on a rarity-keyed side stream.

### What the sim does today

Nothing. There is **no duplicate suppression anywhere** — `random_joker_key` is a weighted draw with replacement, and the buffoon-pack path has no dedupe either. So the sim will hand you two Blueprints without Showman, which is impossible in the real game. And because there's no blocking to disable, **Showman is currently a no-op**. (Note `j_ring_master` — Showman's key — is one of the five registry keys the earlier survey flagged as an unreachable dead alias.)

So "tarot blocking" isn't modelled badly, it's **absent**. Same for the whole blocker mechanic. That's a prerequisite for any of the reasoning in §4 below, not a refinement of it.

---

## 2. How many runs to know which lines are better?

This one has a real answer. Work in log-score space (Balatro scores are multiplicative, so log is the natural scale).

### The base calculation

For a two-arm comparison on a binary outcome (win the match), at 80% power and α = 0.05:

> n per arm ≈ 2σ²(z<sub>α/2</sub> + z<sub>β</sub>)² / δ² = 3.92 / δ²  (for p ≈ 0.5)

| True edge δ | Runs per arm, naive |
|---|---|
| 10 pp | 392 |
| 5 pp | 1,568 |
| 2 pp | 9,800 |
| 1 pp | 39,200 |

### Two things cut that hard

**Paired seeds.** Evaluate both arms on the *same* seed from the same prior state. Variance becomes that of the difference: scales by (1 − ρ). For a decision at ante 4 where both arms share the entire history, ρ should be 0.7–0.9 → **3–10× fewer runs**. This is free in MP because the shared-seed structure is already there.

**Continuous outcome.** Use log-score margin rather than binary win/loss. Losing 40k-to-45k and 40k-to-4M carry very different information. Rule of thumb: another **2–5×**.

Combined, a realistic figure:

> **A single well-defined binary decision, measured with paired seeds and a continuous outcome, resolves a 2 pp edge in roughly 1,000–5,000 runs.**

At the observed ~985 sps and ~300 steps per run, that's **≈12,000 runs/hour** — so one decision is about **ten minutes**. Individually, these questions are cheap.

### Where it stops being cheap

The problem is not one decision, it's the cross product. "Should I take Wraith" isn't a question, it's a family:

```
15 decks × 8 stakes × 8 antes × ~5 money buckets × 7 life differentials
  ≈ 33,600 contexts   ×  2,000 runs  =  67,000,000 runs   ≈ 5,600 hours
```

That's not happening. So:

| Scope | Runs | Wall clock | Verdict |
|---|---|---|---|
| One decision, one context | 2,000 | ~10 min | trivial |
| One decision, ~10 contexts | 20,000 | ~2 h | easy |
| **~50 curated decisions, ~500 configs total** | **1,000,000** | **~83 h** | **feasible — this is the target** |
| 120-cell grid to ±2% each | 300,000 | ~25 h | feasible |
| Full context cross product | 67,000,000 | ~5,600 h | no |

**The conclusion: the statistical layer has to be a curated shortlist, not a policy.** Roughly 50 named, well-defined, locally-decidable questions is affordable in a few days of compute. Everything outside that list needs a learned value function or search — you cannot enumerate your way to a policy.

Two constraints that bite:

- **The policy must be frozen while you measure.** If it's still training, your estimates chase a moving target. Layer 1 is measured against a fixed reference policy, then re-measured after the policy changes — not continuously.
- **Estimates are conditional on the reference policy.** "Wraith is worth it" may be true for a greedy scorer and false for an economy build. Report the conditioning, or measure against 2–3 archetypes.

---

## 3. Could an agent like ours absorb these nuances? — No, and that's the useful finding

Look at what your examples have in common:

- **Save Death for The Fool** — hold a consumable across rounds, anticipate a card you don't have yet, sequence two uses in order.
- **Tarot ordering and blocking** — sequencing plus modelling the queue.
- **Bank money before a Spectral pack** (Wraith zeroes you) — pre-position for a random outcome that may not occur.
- **Don't open Arcana at $0** (Hermit doubles money, max $20, does nothing at ≤$0) — know the pack's content distribution and your own state.

Every one is **delayed, conditional, rare, and looks worse than greedy at every intermediate step.**

That last clause is the killer. PPO discovers behaviour by perturbing action probabilities and keeping what scores better. For "hold Death two rounds → acquire Fool → use in order," the joint probability of stumbling onto the full sequence is a product of small probabilities, and **every partial execution scores worse than the greedy alternative** — you held a consumable slot and got nothing. The gradient points *away* from the plan until the whole thing completes by luck. This is the textbook hard-exploration case.

Your own README already reached this: *"shaped-reward PPO cannot discover the multi-step coordinated strategies Balatro requires."* And V7's largest reward term — `+2.0 × (played_score / best_possible_score)` — **explicitly pays for greedy play every hand**, which `V9_BRAINSTORM_2026-07.md` already flags. That term is actively hostile to every example on your list.

So don't ask the network to learn them. In increasing order of effort:

1. **Compute them directly.** For a curated shortlist (§2), Monte Carlo the EV. No learning. This is the right tool for exactly your examples — they're rare, well-defined, and local in state.
2. **Search.** MCTS evaluates sequences directly instead of needing to stumble on them; it will find 2–3 step plans given a decent value function. The `balatro-mcts` fork exists for this.
3. **Imitation prior.** Your BC-from-Balatro-University idea encodes "save Death for Fool" for free as a human heuristic, then search refines it. This is the AlphaGo recipe and it's the right shape here.
4. **Macro-actions.** Hand-code the combos as primitive options. A cheat, but it works.

**Your instinct to go statistical rather than deeper-RL is correct, and these examples are the evidence for it.**

Caveat worth repeating: tarot ordering and blocking are **not representable** until the queue and duplicate-suppression exist (§1). Those decisions can't be measured, let alone learned, in the current engine.

---

## 4. The PvP decision frame — this is the strongest part of the idea

### The structural advantage: you know their option set

In most imperfect-information games you must infer the opponent's hidden state from scratch. In same-seed Balatro you don't — **you saw the same queue.** If Wraith was in your Spectral pack, it was almost certainly in theirs (modulo reroll offset and blocking).

That converts opponent modelling from *"guess their state"* into *"reason about which branch they took from a menu I can see."* That's dramatically more tractable, and it's genuinely unusual as a game structure. It's the thing that makes your game-theoretic framing actually computable rather than aspirational.

### The Wraith decision as a 2×2

|  | **They take** | **They don't** |
|---|---|---|
| **I take** | both −$M, both +Rare | I: Rare, broke. They: money |
| **I don't** | I: money. They: Rare, broke | nothing happens |

The **diagonal is near-symmetric** — costs and gains largely cancel, and P(win) barely moves. So the decision is determined **entirely by the off-diagonal cells.** You are not valuing Wraith in absolute terms; you're valuing *"Rare joker minus $M"* against the opponent's counterfactual. That's a real simplification, and it generalizes to every shared-option decision.

### Reframing "is it worth $40 or $100?"

The question as posed doesn't have an answer, because dollars aren't the currency. The right question is:

> **What is the money → log-score exchange rate at this ante, and does the Rare joker's expected log-score contribution exceed the money's?**

Convert both sides to expected log-score at the next nemesis blind, compare *margins*, and let the symmetric cells cancel. That exchange rate is precisely what layer 1 produces.

Three things fall straight out of the framing:

**The exchange rate is strongly ante-dependent.** Early, money compounds through interest and has time to convert into jokers that then scale — Wraith is expensive at ante 2. Late, money has no time left to convert and the Rare joker's immediate multiplier dominates — Wraith is cheap at ante 7. Same card, opposite verdict.

**The cost of going broke is capped, and this is the concrete heuristic.** Interest pays $1 per $5 held, capped at $5/round — i.e. capped at $25 held (Seed Money → $50, Money Tree → $100). So the *interest* cost of dropping to $0 is identical whether you're at $25 or at $80. The marginal dollar above the cap is worth far less. **Wraith at $80 is barely worse than Wraith at $25** — which inverts the intuition that a bigger bankroll makes Wraith more painful. Above the cap, it mostly doesn't.

**The currency is P(win the match), not score — so it's life-dependent.** A life at 1-remaining is worth much more than at 4-remaining. The same Wraith decision flips on the life differential: behind on lives you want the variance the Rare joker provides; ahead, you want the reliability money buys. That's the risk-posture inversion from the assessment, now with a number attached.

### And "will I lose a life off that?"

That's directly computable once layer 1 exists: **P(their score > my score at the next nemesis blind | they took Wraith, I didn't)**. It's a comparison of two score distributions, which is exactly what the model outputs.

That's the whole argument for building layer 1 in one line: **it's what turns every question on your list from a vibe into arithmetic.**

---

## 5. What this implies for sequencing

Nothing in §4 is measurable until §1 and §2 of the update list land — the queue, duplicate suppression, and keyed RNG. Before that:

- tarot blocking has no referent,
- Showman is a no-op,
- reroll depth has the wrong statistics,
- and two players on one seed desync on the first divergent hand.

After that, the curated-shortlist approach in §2 is affordable in days, and it targets exactly the decisions PPO provably won't find on its own. That's a coherent plan and it plays to the actual strength of the idea — **the statistical layer isn't a fallback from RL, it's the right tool for this specific class of decision.**

Suggested seed list for the ~50 curated decisions, in rough order of expected leverage:

1. Wraith: take / skip, × ante × money bucket × life differential
2. Reroll depth per ante (optional stopping), × money × ante
3. Spectral pack: open / skip at money level M
4. Arcana pack at $0 (Hermit dead) — open / skip
5. Death→Fool sequencing: hold vs. use immediately
6. Punt-or-push at a nemesis blind, × life differential
7. Intra-blind commitment: play another hand vs. stop, × opponent score × hands left
8. Buy vs. save at each ante, × money bucket
9. Skip vs. play small/big, × tag × ante
10. Showman: value as a function of collection size and ante
