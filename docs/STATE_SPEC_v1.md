# STATE_SPEC v1 — what V sees (DRAFT for Tagg's review, 2026-08-23)

**Why this document exists.** In the Phase 5 design the only learned object is a state-value net
V(state) ≈ P(win the MLB match from here). Everything V's input contains is frozen the moment training
starts — adding a FIELD later means restarting from zero; adding a VALUE to a pre-sized vocabulary does
not. So this spec deliberately includes fields we will not use on day 1 and reserves slack. Review the
field list; anything missing that a strong player looks at must be added now.

Base: the Phase 4 set encoder (`agent/SETENC_NOTES.md` §0.2) — five masked item sets + scalars.
**v1 = that encoder + the additions marked ➕ below.** Caps are transport only (the net is set-invariant).

## Item sets (padded + mask)

| set | cap | per-item categorical | per-item numeric | notes |
|---|---|---|---|---|
| hand | 16 | rank, suit, enhancement, edition, seal | debuffed, face-down, selected, chips, sort position, … (9) | as today |
| ➕ draw pile | — | — | see `deck_counts` in scalars | order is HIDDEN; only composition is observed |
| ➕ discard pile | — | — | see `discard_counts` in scalars | composition only |
| jokers | 12 | key (vocab), edition, rarity | position, scaling state, stickers, sell value, … (16) | as today |
| consumables | 6 | key (vocab) | (8) | as today |
| shelf | 8 | key, kind, edition (+card block) | cost, affordability, … (12) | as today |
| pack choices | 8 | key, set, edition (+card block) | (8) | as today |
| ➕ blind offers | 3 | small/big/boss: blind kind, boss key, **tag key on offer for Small/Big** | chip target (log), reward $, is_pvp, skip-available | the skip decision needs the offered tag — not present today |

## Scalars (today 196 → v1 target ≈ 330; exact widths fixed by `SCALAR_LAYOUT`)

Existing blocks (keep): `state` 7 · `ante_blind` 8 · `blind_kind` 3 · `boss` 28 · `economy` 10 · `capacity` 9 ·
`round` 8 · `planet_levels` 12 · `hand_types_played` 12 · `vouchers` 32 · `tags` 24 + `tag_count` 1 ·
`deck_composition` 16 · `deck_id` 15 · `stake` 1 · `mlb` 10.

Additions:
- ➕ `deck_counts` 13 ranks + 4 suits + 8 enhancements + 5 editions + 4 seals (34) — the REMAINING DRAW PILE's
  composition (what a player sees by viewing the deck minus what's in hand/discard). Today's
  `deck_composition` 16 is the full deck; keep both.
- ➕ `discard_counts` 13 + 4 (17) — discarded this round.
- ➕ `ruleset` 3 one-hot (vanilla / mlb / the_order) + `pvp_start_round` 1.
- ➕ `opponent` block (level-0 opponent modelling, MLB):
  - lives, skips, location/phase one-hot (selecting / playing small / big / nemesis / shop / waiting) (≈10)
  - current Nemesis (if in progress): their live score (log), hands left, my score (log), my hands left (4)
  - last 4 Nemeses: for each — ante, their final score (log), hands they used, my score (log), outcome
    (+1/0/−1), early-end flag (6 × 4 = 24)
  - ➕ **reserved `opp_belief` 16** — zeros until the level-1 belief model exists (P(late-game build), expected
    score curve quantiles for antes +1..+3, …). Pre-reserved so it is not a restart.
- ➕ `race` 6 — my lives, their lives, ante, comeback bonus pending, lives lost this round, antes since last
  life lost (cheap, derived; V could learn these but giving them is free).
- ➕ `money_detail` 6 — interest this round at current $, $ to next interest threshold, reroll cost now,
  free rerolls, shop discount, voucher-on-offer cost.
- ➕ `reserved` 32 — zeros. For anything we think of during training.

## Vocabularies (pre-sized, never resized)
`KEY_VOCAB` = all 150 jokers + 22 tarots + 12 planets + 18 spectrals + 32 vouchers + 24 tags + 28 blinds +
32 boosters + 52 cards + enhancements/editions/seals + `unknown` + **32 spare ids** (for mod content later).

## Action features (no policy net in v1 — the agent is argmax EV under V — but kept for a future prior)
Set-based as today (`act_type`, selected-card pool, target item, numerics). Not frozen: V doesn't read them.

## Net (V only)
`SetValueNet`: per-type item encoders → one masked 4-head attention block over the item union (≈ 70 slots)
→ per-set mean+max pool → concat scalars → trunk. **Target ≈ 5M params** (Tagg's call, 2026-08-23): widen
the trunk (e.g. 1024 × 3 res blocks) and the item embeddings (96–128) to land there. Sigmoid head.
Optional auxiliary heads (later, no restart): next-Nemesis log-score, expected lives at ante+2.

## Target
`V(state) = P(win the MLB match | state)`, labelled by determinized Monte-Carlo rollouts with the analytic
rollout policy (symmetric opponent from public info), truncated at ante 12 with the race calculator.
Vanilla is an EVAL mode only.

## Game-theory note (Tagg's question, 2026-08-23)
v1 V is a **self-play value**: P(win | observable state) labelled by rollouts in which the opponent plays the
same policy from THEIR public information. The opponent's choices enter V only through the observed
opponent block (scores per hand, lives, skips, spend). It is NOT an equilibrium value: a style never present in
the label rollouts (e.g. a 3-life sacrifice for an Idol build) is invisible to V. Refinements, none of which
restart V because the input already reserves the slots: (a) opponent-STYLE diversity in label rollouts (label
generation change only); (b) level-1 belief over the shared menu → `opp_belief` 16; (c) level-2 signalling /
regret minimisation over information sets (CFR-family) as a later research item.

## Open for Tagg
1. Anything a strong MLB player looks at that is not in the list?
2. Opponent block: enough history (4 Nemeses)? Anything else public in the mod UI worth adding (their
   sells-per-ante and spent-in-shop ARE broadcast by the mod — `MP.GAME.enemy.sells_per_ante`,
   `spent_in_shop` — should V see them? I'd say yes: ➕ `opp_econ` 4).
3. Reserved sizes (belief 16, spare 32, vocab spare 32) — generous enough?
