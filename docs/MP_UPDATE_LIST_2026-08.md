# What Needs Updating to Run MLB Multiplayer

**2026-08-20.** Companion to `MP_SELFPLAY_ASSESSMENT_2026-08.md`. Audited at `fix/sim-fidelity-2026-07`.

---

## 0. Answer to the question: no, nothing is synced

**There are no queues, and there is exactly one RNG stream.**

- `game.py:184` — `self.rng = random.Random(seed)`. That is the *entire* RNG architecture.
- Grepping both repos for `pseudoseed|pseudorandom|rng_key|named_rng|per_key` returns **zero hits**. There is no key concept, no per-key state dict, no ante keying.
- No shop queue anywhere: `shop.py:229 generate_shop` does fresh i.i.d. draws; `shop.py:367 reroll_shop` throws the shop away and calls it again. `game.current_shop` is a plain list with no cursor.

So the specific things you asked about — Lucky hits, Glass breaks, the Judgement/buffoon joker queue — are not merely unsynced. They all draw from the **same** Mersenne Twister in temporal order, which means **every draw shifts every future draw.**

Concretely: playing a flush of Hearts with Bloodstone consumes ~5 more draws than a flush of Spades, and therefore changes the next shop, the next boss blind, and every subsequent deck shuffle. In real Balatro those come off `"bloodstone"`, `"Joker"..ante`, `"boss"` and the shuffle stream — four independent sequences that cannot perturb each other.

This is not a rate bug, and the July audit did not touch it. **H3 made the sim deterministic; it did not make it decorrelated.** Determinism is what single-player needed. Decorrelation is what multiplayer needs, and it's a different property.

It also explains the V8 "independent shops" note more completely than rev 1 of the assessment did: two players on one seed who play *any* differently — one plays Hearts, one plays Spades — desync every downstream draw immediately. Same-seed multiplayer is architecturally impossible in the current engine.

**And it's the live failure mode in the real mod too.** The MP mod's recent changelog includes *"fixed seal desync caused by Certificate (SMODS bug)"*, and MLB ships explicit *"Voucher, Bloodstone, and Idol multiplayer fixes."* Those are exactly this class of bug — a card consuming RNG from the wrong stream and shifting another player's draws. The real game has 40+ keys and still gets this wrong occasionally; the sim has one key.

---

## 1. RNG architecture — P0, blocks everything

### 1.1 Implement keyed pseudorandom

Replace the single `game.rng` with `game.pseudoseed(key) -> float`, backed by a per-key state dict. Real Balatro:

```
state[key]  initialised as  pseudohash(key .. run_seed)
advance:    x <- |round( (2.134453429141 + x * 1.72431234) mod 1, 13dp )|
```

Every draw site passes a key. Two decisions:

**(a) Structural parity only** — give each key its own `random.Random`, seeded by `hash(key, seed)`. Cheap, gets you independence and therefore working same-seed multiplayer. Does *not* reproduce real Balatro seeds.

**(b) Byte-exact parity** — port the actual `pseudohash` + LCG. Same amount of call-site work; the only extra cost is the ~50-line RNG core.

**Do (b).** It costs almost nothing more, and it reopens a door you closed: `project-ideas/LUA_REPLAY_MOD_PLAN.md` was abandoned because "Python MT doesn't match Balatro's LÖVE2D xoshiro." But Balatro's *item generation* RNG isn't xoshiro — it's this pseudohash/LCG. Port it and real Balatro seeds produce identical shops, packs, vouchers and bosses, which makes trace-level validation against the real game possible again. (Deck shuffle order does use LÖVE's xoshiro seeded from a pseudoseed; xoshiro128** is another ~20 lines if you want that too.)

Reference implementations: **Immolate** (`MathIsFun0/Immolate`, OpenCL — `lib/functions.cl` has the `R_*` key enum; note this is the same author as The Order) and **TheSoul** (`SpectralPack/TheSoul`). **Blueprint** (`miaklwalker/Blueprint`, TypeScript) is the readable one for the queue structure. There is no Python port on PyPI — you'd be writing the first one.

⚠️ The key-name mapping in §1.2 below is my reconstruction and several names are approximations. **Take the exact strings from Immolate, not from this table.**

### 1.2 Assign a key to every call site

40+ sites. Full inventory:

| Site | Key | Note |
|---|---|---|
| `scoring.py:89` / `:91` | `lucky_mult` / `lucky_money` | rates correct (1/5, 1/15) |
| `game.py:689` | `glass` | rate correct (1/4) |
| `game.py:301` | `boss` | needs exhaustion pool — see §7 |
| `game.py:450` | shuffle stream | must be decoupled from all effect keys |
| `game.py:381` / `:426` / `:567` / `:582` / `:636` | `madness` / Amber / Cerulean / Hook / Crimson | boss-internal picks |
| `game.py:736` | `Tarot`..ante | Purple Seal |
| `shop.py:188` | **`rarity`..ante + `Joker`..ante** | sim collapses two real keys into one `choices()` call |
| `shop.py:258` / `:269` | `shop_pack`..ante / `cdt`..ante | |
| `shop.py:271/274/277` | `Planet`/`Tarot`/`Spectral`..ante | |
| `shop.py:284` | `Voucher`..ante | |
| `shop.py:289` | `edi`..ante | |
| `shop.py:396/398/400` | `ar1` / `pl1` / `spe` (+ante) | pack contents |
| `shop.py:402` | `buf`..ante + `Joker`..ante | buffoon pack |
| `shop.py:406` | `stdset`..ante + `standard_edition` + `standard_has_enh` + `standard_seal` | sim draws **none** of the last three |
| `consumables.py:167/175` | `pl1` / `ar1` | High Priestess / Emperor |
| `consumables.py:190-192` | `wheel_of_fortune` | rate correct (1/4) |
| `consumables.py:237` | `jud` | Judgement |
| `consumables.py:298/311/324` | `familiar_create` / `grim_create` / `incantation_create` | |
| `consumables.py:338/348/354/362/378/387/403/438` | `aura`/`wraith`/`sigil`/`ouija`/`immolate`/`ankh_choice`/`hex`/`soul` | |
| `chips.py:175` / `:184` | `gros_michel` / `cavendish` | rates correct |
| `chips.py:83` | `8ball` | **wrong effect** — §7 |
| `economy.py:27` | `business` | correct (1/2) |
| `misc.py:400/501/573` | `trading_card` / `to_do` / `parking` | |
| `mult.py:106/263/272/298/383` | `misprint`/`bloodstone`/`anc`/`idol`/`space` | rates correct |
| `scaling.py:252` | `cas` | Castle |
| — | `tag` | **no counterpart — no tag system at all** |
| — | `certificate`/`dna`/`seal`/`hallucination`/`orbital` | **no counterpart — §7** |

### 1.3 Resample semantics

Real Balatro appends a **resample counter** to the key when a draw must be rejected (duplicate, banned, already owned) rather than redrawing from the same position. Needed for real-seed parity and for correct duplicate suppression in packs.

### 1.4 Kill the state-dependent draw counts

Even with correct keys, two things must be fixed or the streams still drift:

- **Python's `_randbelow` consumes a variable number of MT words.** Only `.random()` is fixed-cost. `.choice()`, `.randint()`, `.shuffle()`, `.sample()` all use rejection sampling on `getrandbits(n.bit_length())`. So `shop.py:284 rng.choice(available)` changes its word consumption as you buy vouchers (pool shrinks from 25). If you go with option (a) above, use only `.random()` and derive indices yourself. Option (b) sidesteps this entirely since the LCG is fixed-cost.
- **Conditional draws.** `generate_shop` draws `2·joker_slots + 2·card_slots + (1 if any voucher unowned) + 2`. Buying Overstock (`consumables.py:495/497`) permanently adds 2 draws to every future shop. Wheel of Fortune with zero jokers draws nothing (`consumables.py:188`). Judgement with full slots draws nothing (`consumables.py:232`). Each of these is a phase shift under one stream; under per-key streams they're harmless, so **1.1 fixes most of this for free**.

---

## 2. Shop queue — P0

Real model: each ante pre-generates **one long ordered queue**; each shop visit takes the next two entries; a reroll advances the pointer by two. Separate queues for tarot packs and spectral packs. The joker queue also feeds buffoon packs and Judgement.

To build:

1. Generate the ante's queue at ante start, from `Joker`..ante etc.
2. Per-player pointer; +2 per reroll; +2 per shop visit.
3. Separate queues per category (joker / tarot / spectral / planet / voucher / boss / tag).
4. **The Order as a switch:** queue scope = `ante` (MLB) or `run` (Minor League). One flag, and it makes the two rulesets measurable against each other.

Note this is also what makes duplicate-seed pairing (assessment §4.3) actually pay off — both players walk the same queue, offset only by their own rerolls.

---

## 3. Reproducibility breaks — P0, and one of these taints V7's history

**`env_v7.py:565` — the reward function mutates the world.** `_best_hand_score` (`env_v7.py:513-575`, called at `env_v7.py:261` immediately before the real `play`) runs `score_hand(..., rng=gs.rng, ...)` over **up to 8 hypothetical card subsets using the live game stream**. Every Lucky / Bloodstone / Business / 8-Ball / Misprint / Space roll inside those hypotheticals permanently advances the RNG before the actual hand is played. Draw count depends on hand composition and joker set, so it's unbounded and state-dependent.

This means the V7 **card-quality reward term** — `+2.0 × (played_score / best_possible_score)`, the term the run log calls the most impactful — was silently advancing the game RNG every single hand, for the whole of V7 and V8. Fix: clone the state (or thread a throwaway `Random`) inside `_best_hand_score`.

**`env_sim.py:619` and `env_v5.py:847`** call `score_hand()` without `rng=`, so `ctx.rng is None` and `jokers/base.py:25 rng_of()` falls through to the **unseeded process-global `random`**. These older envs avoid corrupting `game.rng` only by accident.

**Latent:** `consumables.py:250` and `shop.py:363` pass `None` as ctx to `on_tarot_used` / `on_sell`. No current implementation draws, so nothing escapes today — but the next one that does will silently leave seed control. Same for the `random`-module defaults at `shop.py:179` and `shop.py:288`.

---

## 4. MLB ruleset — P1, and genuinely small

MLB is vanilla Balatro plus:

- Nemesis blind object; chip target = opponent's live score
- **4 lives, decremented on *any* blind lost** (regular and PvP alike); first to 0 loses
- Comeback money on life loss
- Boss slot → PvP from ante 2 (Attrition)
- Early-win rule: round ends the moment the opponent finishes and you're ahead
- Two-player coordinator stepping both games in ante lockstep
- 180 s timer — irrelevant to a bot except as an edge over humans
- Voucher / Bloodstone / Idol MP fixes (these are §1 desync fixes, so §1 subsumes them)

No MP jokers, no rebalance suite. `mp_game.py` (251 LOC) and `env_mp.py` (418 LOC) already exist as scaffolding but were written against the pre-fix engine.

---

## 5. Decks, stakes, stickers — P1, the largest single piece

Currently **zero** of this exists — only a standard Red-equivalent (`make_standard_deck`).

**Decks (15).** Cheap: Red (+1 discard), Blue (+1 hand), Yellow (+$10), Green (flat round-end cash, no interest), Black (+1 joker, −1 hand), Painted (+2 hand size, −1 joker), Magic (Crystal Ball + 2× The Fool), Nebula (Telescope, −1 consumable), Ghost (Hex, spectrals in shop), Zodiac (Tarot Merchant + Planet Merchant + Overstock). Medium: Abandoned (40 cards, no face cards), Checkered (clubs→spades, diamonds→hearts), Erratic (all ranks/suits randomized). Hard: **Plasma** (balances chips and mult, ×2 base blind size — a different scoring regime, and the likely worst-transfer cell). Blocked: **Anaglyph** (Double Tag after each boss — needs §6).

**Stakes (8, cumulative).** Scalar retunes: Red (less reward money), Green (faster blind scaling), Blue (−1 discard), Purple (faster scaling again). Need a **joker sticker system** that doesn't exist: Black (**Eternal**, 30% unsellable), Orange (**Perishable**, expires after 5 rounds), Gold (**Rental**, costs money per round). Orange and Gold invert the buy-and-hold logic every version of the agent has relied on.

---

## 6. Tags — P0 for strategy fidelity (promoted from P1)

**Promoted after confirming how the endgame actually plays.** In Black deck / Gold stake
runs at high antes, strong players skip small and big blinds straight to the PvP blind
rather than risk failing them — skipping costs no life, failing costs one. Skip is the
endgame's life-preservation tool, and the tag is its compensation, so at high antes much
of your economy routes through tags rather than blind rewards. With a flat `+$5` and no
tag system, the sim cannot represent that strategy at all.

`_skip_blind` (`game.py:835-846`) pays a flat `+$5` with the comment *"approximate."* There is no tag pool, no tag object, no `tag` stream. Needed for skip strategy generally, for the Double Tag that Anaglyph requires, and for `misc.py:513`'s orphaned `"double_tag"` token.

---

## 7. Correctness bugs found in this audit — P2, mostly cheap

None of these are in `SIM_AUDIT_2026-07-29.md`, and the 828-test suite passes with all of them present.

**Wrong effect:**
- **8 Ball** — two implementations; import order in `jokers/__init__.py:29-37` makes `chips.py:80-85` (1/4 → +20 chips) win over the **correct** `misc.py:251-255` (creates a Tarot). Live version is wrong; correct version is dead code.
- **Gros Michel / Cavendish** — rates correct but **destruction never happens.** `inst.state["destroyed"]=True` is only ever read into an observation vector (`env_v7.py:738`); nothing removes the joker. Gros Michel is a permanent +15 Mult with no downside, and Cavendish is never created.
- **The Idol** — initial target never randomized; a fresh Idol is always **Ace of Spades** until the first round ends (`mult.py:289-300`).
- **Lucky Cat** — `on_lucky_trigger` is implemented (`scaling.py:444`, `economy.py:60`) but has **zero callers**, so Lucky Cat never scales and Golden Ticket's lucky path is dead.

**Nine jokers produce nothing.** `pending_consumables` producers emit sentinel strings — `"tarot"`, `"spectral"`, `"common_joker"`, `"random_enhanced_card"`, `"copy_card:{rank}:{suit}"`, `"duplicate_joker"`, `"negative_tarot"`, `"double_tag"` — which `_use_consumable` (`game.py:744-758`) can't match against `PLANET_HAND`/`ALL_TAROTS`/`ALL_SPECTRALS`. They occupy consumable slots as permanently unusable items. Affects **Vagabond, Superposition, Cartomancer, Seance, Sixth Sense, Riff-Raff, Certificate, DNA, Perkeo**.

**Six hooks defined, never invoked:** `on_booster_opened`, `on_shop_enter`, `on_shop_leave`, `on_lucky_trigger`, `on_card_added`, `on_boss_ability_triggered`, `on_reroll` (whose own comment claims it's called from `reroll_shop` — it isn't).

**Six dead `on_init` methods, three of which would raise `NameError`** on an undefined `ctx`: `scaling.py:242` (Castle), `scaling.py:485` (Ancient), `misc.py:496` (To Do List). Consequence: Castle starts with no suit, To Do List always starts on High Card.

**Pool / gating errors:**
- **Boss blinds have no exhaustion pool** (`game.py:294-302`) — real Balatro won't repeat a boss until all are seen; here they're i.i.d., so repeats are common.
- **Vouchers have no ante gating** (`shop.py:281-284`) — `v_overstock_plus` can appear before `v_overstock`.
- **Hone and Glow Up are purchasable no-ops** — in `VOUCHER_NAME` but `apply_voucher` has no branch for them, so they never affect edition rates.
- **Booster packs allow duplicates** (i.i.d. with replacement); real Balatro suppresses them.
- **Standard packs produce vanilla cards** — no enhancement, edition or seal, skipping three real RNG keys.
- **Editions only roll for shop jokers** — not buffoon-pack jokers, Judgement/Wraith/Soul jokers, or cards from Familiar/Grim/Incantation.
- **Glass retrigger** rolls once per card, not per trigger (`scoring.py:82` dedupe).
- **Wheel of Fortune** can target already-editioned jokers; real WoF only targets edition-less ones.

**Carried over from the earlier survey:**
- **Rarity distribution wrong** — sim Common 123 / Uncommon 18 / Rare 10 vs. real 61 / 64 / 20. `RARITY_WEIGHTS` drives the shop roll, so ~80% of real Uncommons sell at Common odds *and price*.
- **Six jokers double-listed** under two keys at different rarities: Wee Joker, The Duo, The Trio, The Family, The Order, The Tribe.

---

## 8. Infrastructure — P2

- `clone()` / `legal_actions()` exist **only** in the `balatro-mcts` fork, not in `balatro-rl`. (RNG cloning there is correct — `game.py:316-318` does a proper `getstate`/`setstate`.)
- `train_cold.py` saves **no checkpoints**; an opponent pool needs them.
- No eval harness.
- 120-cell (15 decks × 8 stakes) grid runner + per-cell win-probability table.
- Draft LP solver on top of that grid.

---

## 9. Suggested order

1. **Merge `fix/sim-fidelity-2026-07`.** Clean up the stale `%TEMP%\brl-prefix` worktree.
2. **§3 reproducibility breaks.** Small, and until `_best_hand_score` stops advancing the stream, nothing is measurable.
3. **§1 keyed RNG, option (b).** The big one. Blocks multiplayer entirely, and byte-exact parity is nearly free once you're editing every call site anyway.
4. **§2 shop queue** + The Order switch. Natural to do in the same pass as §1 since both touch `shop.py` generation.
5. **§7 correctness bugs.** Cheap, and they bias every statistic layer 1 would measure. Add regression tests — the current 828 don't catch any of these.
6. **§4 MLB rules.** Small once §1–2 land.
7. **§5 three decks only** — Red, Checkered, Plasma, White stake — and measure the transfer spread. This is the decision gate from the assessment. Full grid only if the spread is survivable.
8. **§6 tags**, **§5 remaining decks/stakes/stickers**, **§8 infra.**

Steps 1–5 are the honest prerequisite. That's a substantial refactor of `shop.py`, `consumables.py`, `game.py` and every module under `jokers/` — and it lands in the same weeks as MATS (Sep 4 / Sep 6) and ERA (Sep 13).

**One genuine upside:** step 3 with option (b) gets real-seed parity, which means you could validate the sim against the actual game seed-for-seed — and against the MP mod directly. That's a stronger correctness guarantee than 828 unit tests, and it would have caught most of §7 automatically.
