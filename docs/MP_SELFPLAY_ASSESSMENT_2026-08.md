 # Balatro Multiplayer as a Self-Play Problem

**Assessment, 2026-08-20. Rev 2** — supersedes rev 1, which got the lives rule wrong.

Target ruleset: **Major League (MLB)**, The Order *disabled*.

Sources: balatromp.com docs (Major-and-Minor-league, Standard Ranked, Ranked Rules, FAQ); majorleaguebalatro.com/about; BalatroMultiplayer `dev` source (`agents.md`, `gamemodes/attrition.lua`, `networking/action_handlers.lua`); this repo's `SIM_AUDIT_2026-07-29.md`, `V8_RUN_LOG.md`, `balatro_sim/shop.py`.

---

## 0. Corrections to rev 1

**Lives — I had this wrong.** Rev 1 argued that antes must run in lockstep, therefore failing a regular blind can't be fatal, therefore small/big blinds demote to a pure economy minigame. That was a deduction from a structural argument plus an ambiguous FAQ sentence, and it's false. MLB's own rules state it plainly: *"players get four lives; you lose a life for each Blind lost, including PvP, with the first player to zero lives losing."* Lives are a **single pooled currency spent on both regular-blind failures and PvP losses.**

Two things follow:

1. **Rev 1 §2.2 is retracted.** MP does not delete the survival constraint; it buys you a 4-mistake buffer against it. That makes MP *harder* than rev 1 claimed, not easier.
2. **V8's "HOUSE RULE" was correct fidelity, and my critique (c) was wrong.** Run 3 charging a life for regular-blind failure was right. So Run 3's pathology — 91% ante-1 death rate, agent learning "failing blinds is OK, I have 4 lives" — was *not* the agent correctly discovering a feature of the mode. It was a **reward mispricing**: a life was worth −1.5 against shaped rewards large enough to make burning one profitable. In the real game a life is 1/4 of your entire match. The fix was to price lives correctly, not to add or remove rules.

The rest of rev 1's V8 critique stands, and (a) is the one that matters most — see §3.

**Also taken:** using MP as a side-channel to train SP is a flawed frame, and rev 1 leaned on it ("useful to the single-player line anyway"). Dropped. MP is the target here.

---

## 1. Why MLB is the right ruleset to target

Standard Ranked is the wrong build target and MLB is the right one, for reasons that are mostly about implementation cost:

| | Standard Ranked | **Major League** |
|---|---|---|
| Card pool | 10 new MP jokers, 4 nerfs, Asteroid, Ouija rework, Justice removed | **Vanilla. No joker changes.** |
| Bans | 4 jokers, 4 vouchers, 2 blinds | Boss blinds → PvP; Voucher/Bloodstone/Idol MP fixes |
| Shop queue | The Order on | **The Order off — ante-based queue** |
| Timer | ruleset-dependent | 180 s |
| Lives | 4 | 4, lost on **any** blind lost |

MLB is *vanilla Balatro plus a nemesis blind plus lives*. Your sim already has the vanilla card pool at high fidelity — 150/150 jokers, 27 vouchers, post-audit sequential-fold scoring. **Targeting MLB means you implement three things instead of thirty.** No MP jokers, no rebalance suite, no chasing mod-version churn on a card pool that changes every patch.

The one thing MLB *adds* that Standard Ranked doesn't is the deck/stake draft (§4), which is the interesting part anyway.

---

## 2. The Order, and a real gap in the sim

**What The Order is.** In vanilla Balatro each ante generates **one long ordered queue up front**. Every shop visit takes the next two entries off it; a reroll advances the pointer by two. Items you rolled past are gone for that ante. At the next ante, a fresh queue.

"The Order" (a mod by MathisFun_, bundled as an MP option) replaces that with a **single standardized queue for the whole run**, covering jokers, buffoon packs and Judgement, with separate queues for tarot and spectral packs. Its stated purpose is fairness: if your opponent rerolls once and you don't, you see next shop what they just saw, so nobody gets "robbed" of a card by an off-by-one.

- **Minor League:** The Order **enabled** — one global queue, cross-ante coupling. Rolling deep now costs you position later.
- **Major League:** The Order **disabled** — vanilla ante-based queues. Each ante is an independent draw. This is what you described: *"if you roll far in shops it's different from ante to ante instead of the same."*

MLB is the better choice for a statistical player, and not only for the reason you gave. With ante-based queues, "how deep do I roll this ante" is a **self-contained optional-stopping problem** with no credit assignment leaking into future antes. Under The Order it becomes a cross-ante resource allocation problem — strictly harder to learn and harder to solve exactly.

**The gap.** Your sim has no queue at all. `shop.py:367 reroll_shop()` pays the cost and calls `generate_shop(game)`, which at `shop.py:229` does fresh i.i.d. draws for every slot — 2 jokers, 2 consumables, 1 voucher, 2 boosters, each independently sampled. There is no pre-generated ordered list and no pointer.

For a *single* player this is nearly harmless: walking an i.i.d.-generated queue you can't see ahead in is predictively equivalent to redrawing. **For two players it is not.** In real MP both players hold the same queue from the shared seed and consume it at their own pace, so their shops are strongly correlated and offset only by rerolls. In your sim the two players' shops are statistically independent.

That has three consequences worth knowing before building:

1. **The Order on/off cannot be represented at all** — both settings collapse to the same behaviour. The exact mechanic you want to build strategy around is the one the sim can't express.
2. **Duplicate-seed pairing (§5.3) doesn't work as intended.** Its variance reduction comes from both players facing the same shop draw; with independent shops most of the benefit evaporates.
3. It explains `MULTIPLAYER_RULESET.md`'s claim that the real mod gives *"independent shops (different randomization per player)."* That isn't a description of the mod — it's a description of `generate_shop`, written up as if it were the rule.

Fixing it is contained: build an ordered queue at ante start, give each player a pointer, advance by two per reroll. That also makes The Order a one-line switch (queue scope = ante vs. run), so you could measure the two rulesets against each other — which is a result in itself.

---

## 3. The V8 verdict still doesn't hold

`V8_RUN_LOG.md` concluded *"self-play training is WORSE than solo training for this problem."* It ran entirely pre-`fix/sim-fidelity-2026-07`. Cross-referencing the audit against what V8 needed:

| Audit bug | Consequence for V8 specifically |
|---|---|
| **H3** — shop RNG bypassed `game.rng` at ~45 sites; shop contents not seed-determined at all | V8's stated rationale, *"same-seed environments force direct policy comparison,"* **was never true.** The premise was inoperative. (And even post-H3, §2 means shops still aren't coupled.) |
| **C7** — deck reset to a vanilla 52 every blind | Deckbuilding categorically impossible ⇒ two policies had **almost no mechanism to diverge**. Run 1's 74% draws and "near-identical outcomes" are the predicted symptom. |
| **C1** — no money for beating a blind (~56% of real income) | Broke the economy, which drives every shop decision. |
| **A1** — consumables consumed for zero effect, slot 1 unreachable | Removed the tarot/spectral axis entirely. |
| **C4** — joker order pooled | Sim granted optimal ordering for free. |

A self-play experiment whose coupling mechanism was inoperative, on an engine where builds cannot diverge, reporting that the two players converge to identical play, is not evidence about self-play. Add the design issues that stand from rev 1: rewards weren't zero-sum (+10/−5 plus **+5 to both** for mutual survival — a collusion incentive, and Run 1 duly drew 74%); binary PvP outcomes discarded margin information; no opponent pool; Run 4 cold-started and was compared to a mature V7 run.

**It hasn't been tested.** That's the claim — not that it will work.

---

## 4. The actual project: a statistical min-max player

What you're describing isn't a policy network. It's a **two-layer statistical player**, and the layers are cleanly separable:

### Layer 1 — per-cell win probability, from mass self-play

MP ranked requires a fully unlocked profile, so all **15 decks × 8 stakes = 120 cells** are legal. Layer 1 estimates, for each cell, the distribution of *score at ante k* under your policy — and from two such distributions, P(win) at each PvP blind.

This is where thousands of games of self-play go, and note what it produces: not a policy, but a **calibrated statistical model**. That's a much more robust artifact than a network, it's directly inspectable, and it's what makes layer 2 possible.

### Layer 2 — the draft is a solvable matrix game

The MLB format:

- **Regular season, Bo3.** Game 1's deck and stake are set by wheel spin. **The loser** picks the deck or the stake for games 2 and 3 — or defers and makes the opponent choose first. **Once a deck or stake is played, neither may be chosen again.**
- **Playoffs, Bo5.** Higher seed picks deck and stake for games 1, 3, 5; lower seed for 2 and 4.

This is a finite, perfect-information game over a 120-cell payoff matrix with a no-reuse constraint and an explicit defer option. Given layer 1's per-cell win probabilities, **layer 2 is an LP** — solvable exactly, cheaply, offline. No learning required.

And it gives you the min-max framing literally rather than metaphorically:

> Against an opponent who drafts adversarially, your rating is set by your **worst** cell, not your average.

A bot at a uniform 55% across all 120 cells beats a bot averaging 60% with a 20% hole, because the draft rules exist precisely to let the trailing player steer toward that hole. The no-reuse rule makes specialization not merely risky but *mechanically forbidden* — you will play 3 different decks and 3 different stakes in a Bo3, and 5 of each in a Bo5.

**So the objective is distributionally robust, not average-case.** Train to minimize worst-cell performance. That's a different loss than anything V1–V8 used, and it's the correct one for this format.

### Why this answers your transfer question directly

"How much translates deck to deck" stops being a vague worry and becomes **the measured output of layer 1**: the variance of win rate across the 120-cell grid *is* your exploitability in the draft. That's a crisp, quantitative result, and it's publishable-shaped in a way "2.35% win rate" never was.

My prior on how transfer will actually break down:

| Deck class | Decks | Expected transfer |
|---|---|---|
| **Resource shifts** | Red (+1 discard), Blue (+1 hand), Black (+1 joker, −1 hand), Painted (+2 hand size, −1 joker) | **High.** Quantitative changes to a shared strategy space, and hands/discards/slots are already observation features. |
| **Economy shifts** | Yellow (+$10), Green (flat cash, no interest) | **High-ish.** Retunes shop tempo, not the strategy space. |
| **Starting inventory** | Magic (Crystal Ball + 2 Fool), Nebula (Telescope, −1 consumable), Ghost (Hex, spectrals in shop), Zodiac (3 merchant vouchers) | **High**, provided the policy conditions on inventory — it does. Just different initial states. |
| **Composition changes** | Abandoned (no face cards), Checkered (spades/hearts only), Erratic (random ranks and suits) | **Low.** These change hand-frequency statistics at the root. Checkered makes flushes near-free; Abandoned kills every face-card synergy; Erratic randomizes the whole basis. |
| **Different game** | **Plasma** (balances chips and mult, ×2 blind size) | **Lowest.** A distinct scoring regime, not a modifier. Expect this to be the hole. |
| **Blocked** | Anaglyph (Double Tag after each boss) | **Unimplementable today** — needs tags, which the sim doesn't have. |

Stakes are cumulative and mostly cheap to model, with one cluster that isn't: Red (less reward money), Green (faster blind scaling), Black (**Eternal**, 30% of jokers unsellable), Blue (−1 discard), Purple (faster scaling again), Orange (**Perishable**, jokers expire after 5 rounds), Gold (**Rental**, jokers cost money per round). Red/Green/Blue/Purple are scalar retunes. Black/Orange/Gold need a **joker sticker system** the sim has no concept of — and Orange and Gold in particular invert the "buy and hold" logic the agent has always relied on.

---

## 5. What multiplayer actually changes about play

With the lives correction, the honest list is shorter than rev 1's.

**5.1 The boss target is endogenous.** SP is a threshold game — chips above the blind are wasted, so optimal play satisfices. At a nemesis blind the target is the opponent's live score, so excess becomes margin. Risk posture inverts with the life differential: behind → increase variance; ahead → decrease it. Against a stronger opponent, MP bosses are *harder* than SP's fixed curve; against a weaker one, easier. Under self-play they're exactly calibrated, which is the auto-curriculum property.

**5.2 Lives are a unified currency.** Four lives spent across both regular-blind failures and PvP losses. That makes "punt this blind and bank the money" a real, priceable option that SP doesn't have — but at 1/4 of your match each. This is the thing V8 mispriced.

**5.3 Denser, self-normalizing outcome signal.** Every nemesis blind is ~50/50 by construction under self-play, versus a ~2% terminal label in SP. More importantly the opponent's score is drawn from your *own current ability distribution* at every ante, so a margin reward — `tanh(k·(log my − log opp))` — stays centered near zero at roughly constant scale throughout training, instead of saturating at one end of a super-exponential curve. That is the property that ate six V7 reward retunes.

Rev 1 quoted ~5.8 PvP blinds per match from a race-to-4. **That was computed assuming lives are lost only at PvP blinds and is now an upper bound** — regular-blind failures drain the same pool, so expect fewer.

**5.4 Boss blinds are gone from ante 2 on.** In Attrition (which MLB uses) the boss slot *is* the nemesis blind. Boss debuff adaptation collapses to an ante-1 concern — welcome, given H4 found half the boss blinds inert or wrong.

**5.5 The intra-blind commitment game.** Both players act simultaneously, both see running scores, and the round ends the instant one finishes while behind. Play first and post 50k and I know exactly what to beat and can spend the minimum, banking the rest. The mod ships an anti-stall rule — *"if both stall, the lower-scoring player must play first"* — and sends `pvpTimerOrder`/`firstPlayer` to arbitrate, which tells you the standoff is real. State is small: (my score, their score, my hands left, their hands left, my per-hand score distribution, timer order). Tractable for CFR or DP on a discretized grid, and `balatro-mcts`'s `clone()`/`legal_actions()` at 16.6k clones/sec is already the right substrate. **Highest edge per unit effort in the whole build**, and the one place "near-optimal" is provable — humans are worst here, especially under a 180-second clock.

---

## 6. What has to be built

Ordered by whether it blocks the thing above it.

**Blocking:**

1. **Merge `fix/sim-fidelity-2026-07`** (14 commits, pushed, clean tree; also clean up the stale `%TEMP%\brl-prefix` worktree). Re-baseline. Nothing is interpretable until this lands.
2. **Shop queue** (§2). Ordered queue per ante, per-player pointer, +2 per reroll. Makes The Order a switch. Without it you cannot model the mechanic you want to reason about, and paired-seed pairing doesn't pay off.
3. **Decks (15) and stakes (8).** Currently *zero* — only a standard Red-equivalent. This is the whole point of the project and it's the largest single piece of work. Needs a **joker sticker system** (Eternal/Perishable/Rental) that doesn't exist.
4. **Tags.** Skip currently pays a flat +$5. Tags are load-bearing in MP tempo, and Anaglyph is unimplementable without them.
5. **Checkpoint saving + eval harness.** `train_cold.py` doesn't save weights. An opponent pool needs checkpoints; a 120-cell grid needs an eval harness.

**MLB rules proper (small, because MLB is vanilla):** nemesis blind, 4 lives lost on any blind lost, comeback money, boss→PvP from ante 2, the Voucher/Bloodstone/Idol fixes. The 180 s timer is irrelevant to a bot except as an advantage.

**Fidelity bugs found in this survey, not in `SIM_AUDIT_2026-07-29.md`:**

- **Rarity distribution is badly wrong** — sim Common 123 / Uncommon 18 / Rare 10 vs. real 61 / 64 / 20. `RARITY_WEIGHTS` drives the shop roll, so ~80% of real Uncommons are sold at Common odds *and Common price*. This distorts the shop economy directly, and the shop economy is what layer 1 is measuring.
- **Six jokers double-listed** under two keys at different rarities (Wee Joker; The Duo/Trio/Family/Order/Tribe). Over-represented in the pool. `test_no_implementation_is_unreachable` passes because both copies are reachable.

Both are cheap and both bias layer 1's estimates.

---

## 7. Against real players

**Where a bot is structurally strong:** the intra-blind commitment game under a 180 s clock (§5.5) — exact arithmetic, no timer panic, correct optional stopping; the reroll-depth decision, which under MLB's ante-based queue is a clean stopping problem; and the draft (§4), which is an LP a human is solving by intuition. It also doesn't tilt across a Bo5.

**Where it's structurally weak:** shop and build decisions over a long horizon, which is exactly where strong humans are strong and where V1–V8 topped out at ~2%.

**Calibration matters here.** MLB is 24 invited creators — that's the ceiling of the playerbase, not the median. "Can it beat a human" and "can it beat an MLB player" are very different questions, and only the first is plausibly in reach. Two things do genuinely lower the bar versus SP, though: you never need to beat ante 8, only to out-score one opponent for a handful of blinds; and 4 lives absorb the early mistakes that end SP runs outright.

**Norms:** ranked matchmaking explicitly bans third-party mods, so a bot on the ranked ladder against unwitting opponents is against the rules. Bot-vs-bot, bot-vs-you, and bot-vs-consenting-opponents give the same evidence.

**Validation caveat:** `LUA_REPLAY_MOD_PLAN.md` already established that Python's Mersenne Twister doesn't match LÖVE2D's xoshiro, so the same seed yields a different deck. Cross-checking the sim against the real game has to be distributional (score-vs-ante curves per deck/stake), never trace-level.

---

## 8. Verdict

The reframing is better than rev 1's. The project isn't "self-play to break the SP ceiling" — that was V8's flawed frame and it's rightly abandoned. It's: **build a calibrated statistical model of score distributions across the deck × stake grid, then solve the draft and the commitment subgame exactly on top of it.** The learned part is a distribution estimator, the strategic part is an LP and a DP, and the headline result is a robustness surface over 120 cells rather than a single win rate.

That's a genuinely better-posed problem than anything V1–V8 attempted, and the "how much transfers deck to deck" question stops being a worry and becomes the deliverable.

The cost is honest and it's mostly in §6: decks, stakes, stickers, tags and the shop queue are all real work, and none of it exists today. That's a bigger lift than rev 1 implied, and it lands in the same weeks as MATS (Sep 4 / Sep 6) and ERA (Sep 13).

If you want a cheap decisive first cut: **merge the branch, add the shop queue, implement 3 decks spanning the transfer taxonomy (Red, Checkered, Plasma) at White stake only, and measure the spread.** If a policy trained on Red transfers to Checkered and Plasma at all, layer 1 is viable and the grid is worth building. If it collapses — which I'd bet on for Plasma — you've learned the shape of the real problem for a few days' work instead of a few months'.
