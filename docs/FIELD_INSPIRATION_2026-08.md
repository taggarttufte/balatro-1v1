# Field inspiration — techniques for "shallow but heterogeneous" games

Date: 2026-08-25. Origin: survey conversation with Tagg after the Phase-5 first results.
Status column is as of this writing; live status lives in `CAMPAIGN_LOG.md`.

**The framing this list answers:** Balatro's difficulty is inverted relative to chess/Go —
*shallow but heterogeneous* (150 composable rule-modifications, trivial arithmetic each,
combinatorially composed) with a *deceptive* reward landscape (every path from a local
optimum crosses reward-downhill territory: greedy lock-in, skip-and-coast, the sandbag
valley). Humans are fast because the game is self-describing (card text = the model), they
learn from causal post-mortems rather than outcome statistics, and they plan in build space.
The Phase-5 architecture is the mechanization of that recipe: encode what humans read,
search what humans simulate, learn only what humans learn (long-horizon judgment = V).
Nobody publishes ML on Balatro (top search hit for "Balatro reinforcement learning" is this
repo); the literature home of the problem class is the reading-games / compositional-rules
line (RTFM, Baba Is AI / BALROG, Orak).

## 1. QD archive over build descriptors — QUEUED (after wave 2)

* **Lineage:** novelty search & deceptive landscapes (Lehman & Stanley 2011), MAP-Elites
  (Mouret & Clune 2015), Go-Explore "first return, then explore" (Ecoffet et al. 2019/2021).
* **Key move:** QD runs in a chosen LOW-DIM behavior-descriptor space, not state space —
  the answer to "novelty search can't scale here". Descriptors = build coordinates
  (dominant hand type, xmult/+mult ratio, econ-joker count, deck-thinning, skip rate),
  binned to a few thousand cells. Mutation operator = perturbing the rules player's config
  (our policy is parametric — unusually good fit). Archive elites + `clone()` (0.14 ms) =
  the exploring-starts distribution the training design already calls for.
* **Addresses:** strategy diversity (the V1–V8 killer), label/pair state coverage,
  archetype coverage for the transfer-spread measurement.
* **Estimate:** 1 build session + overnight CPU archive fill; the iteration risk is
  descriptor design, not code.

## 2. Active label selection — POC RUNNING (W-ACTIVE, 2026-08-25)

* **Lineage:** Prioritized Level Replay (Jiang et al. 2021), Unsupervised Environment
  Design / PAIRED (Dennis et al. 2020), classic active learning via ensemble disagreement.
* **Connection worth remembering:** the project's own "your rating is your worst cell"
  draft objective IS minimax-regret UED.
* **Key move:** label where V is uncertain (ensemble disagreement), not uniformly; the
  known failure mode is chasing aleatoric label noise via raw error — the POC measures
  disagreement vs |error| vs uniform arms explicitly.
* **Addresses:** the label-efficiency wall (5M params saturate 51k rows by epoch 7;
  ±0.24 CI per label at n_rollouts=8).

## 3. Auxiliary prediction heads — APPROVED, spec'd as brief §6b (W-AUX, wave 2)

* **Lineage:** UNREAL auxiliary tasks (Jaderberg et al. 2016) and the self-predictive
  representation line after it.
* **Key move:** the rollouts behind every label already contain dense proximal quantities
  (money at next shop, lives per ante, PvP score margin, xmult-by-ante-4, extraction
  income, cards fixed) that are currently discarded; predicting them as cheap heads
  densifies a 1-bit-per-8-rollouts signal and shapes the trunk. Also the mechanized human
  post-mortem ("lost because no xmult") and a per-head diagnostic for WHERE V is wrong.
* **Addresses:** sparse signal; highest payoff-confidence item of the survey.

## 4. LLM-writes-the-encode-layer — ★ BACK POCKET (Tagg, 2026-08-25: "might return to this")

* **Lineage:** RTFM (Zhong et al. 2020, agents reading rule text and composing novel rule
  combinations), Read and Reap the Rewards (Wu et al. 2023, auxiliary rewards from Atari
  manuals), Voyager (Wang et al. 2023, LLM writes a verifiable composable skill-code
  library), Baba Is AI / BALROG (2024) as the benchmark home of compositional rule reading.
* **Key move:** LLM reads each item's card text + Lua → emits the analytic EV function →
  **auto-verified empirically against the oracle-grade engine** (simulate states where the
  item fires, compare predicted vs realized delta with CI tolerance bands). The bit-exact
  engine makes generated code *checkable*, which is the novel part.
* **Addresses:** the hand-encoding bottleneck (W-EXTRACT covers ~7 procs by hand; ~120
  items have non-scoring semantics the dry-run scorer can't see) and the shop proxy's
  documented blind spots (EV_NOTES §8.4: scaling jokers, tarots/spectrals, vouchers).
* **Estimate (inside Claude Code, no paid API):** ~3 build sessions + one 5–8-agent fleet
  day; 2–3 days wall, mostly unattended; ~2–4M tokens; CPU-only. Phases: (1) verification
  harness — the real engineering (reachability-instrumented state targeting; per-tier
  checks: deterministic / stochastic / policy-conditional scaling jokers); (2) batch fleet,
  ~15–20 items per agent iterating to green; (3) registry integration + 126-seed gate +
  paired h2h. **#1 risk: double-counting effects the dry-run scorer already captures — the
  harness must check MARGINAL prediction vs realized delta so double-counts fail loudly.**
* Start only after the V-v2 round settles (W-EXTRACT defines the registry pattern).
* **Writeup potential:** "oracle-verified LLM-generated analytic priors for compositional
  games" — generalizes beyond Balatro; strongest publication-shaped idea in this list.

## 5. Parked, with explicit un-park triggers

* **Successor features / factored value transfer** (Barreto et al. 2017/2020): principled
  machinery for deck→deck transfer on the 120-cell grid; textbook SF assumes near-linear
  reward decomposition, so the realistic version is deck/stake conditioning + per-mechanic
  contribution heads (overlaps §3). *Trigger: transfer-spread measurement phase.*
* **Belief-state search** (DeepStack 2017; ReBeL, Brown et al. 2020; Player of Games,
  Schmid et al. 2021): fixes PIMC determinization pathologies (strategy fusion,
  non-locality) via values over public belief states; MLB's shared menu gives structured
  beliefs. Not binding while match outcomes are dominated by solo play quality.
  *Trigger: h2h margins between strong players compress.*
* **PSRO / league training** (Lanctot et al. 2017; AlphaStar league, Vinyals et al. 2019):
  exploiter populations against strategy collapse; cheap version (heterogeneous anchors,
  N×N) half-built. *Trigger: V-in-the-loop tournament training resumes.*

## Lineage notes for what we already do (named, for future writeups)

* Lever (b) same-world action pairs = **vine rollouts** (TRPO, Schulman et al. 2015) +
  **counterfactual credit assignment** (Mesnard et al. 2021) + common random numbers.
* Potential-based shaping guarantee (Ng & Russell 1999) — already in MP_TRAINING_DESIGN.
* EVPlayer-beats-learned-MCTS (57/58) rhymes with the **NetHack Challenge** (NeurIPS 2021),
  where symbolic bots beat deep RL for years — encoded knowledge beats sampled knowledge
  in heterogeneous domains.
* Deception terminology: Lehman & Stanley; exploring starts / reverse curriculum already
  cited in MP_TRAINING_DESIGN.
