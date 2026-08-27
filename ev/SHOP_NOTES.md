# SHOP_NOTES — W-SHOP: the EV shop economy (Phase 5 rev 2, 2026-08-26)

Owner: W-SHOP.  Spec: Tagg's design (CAMPAIGN_LOG 2026-08-26 "SHOP-ECONOMY DIAGNOSIS").
Files: `ev/player.py` (the shop tier), `ev/hand.py` (one hook, `_fool_ordered`),
`stats/hit.py` (one additive accessor, `pack_slices`), `ev/tests/test_shop.py` (33 tests),
`ev/scripts/shop_profile.py` (the profiler + the paired h2h), this note.
Nothing in `engine/**`, `rng/**`, `oracle/**`, `eval/**`, `agent/**` was touched.

Read `EV_NOTES.md` §4 first (the SHOP/BOOSTER tiers and the build proxy); this document
describes only what changed, and — because two of the three components did not do what the
design predicted — what was measured on the way.

## 0. Headline

Two arms, the SAME player, `player.shop_arm_cfgs("old"|"new")`:

| | old (pre-W-SHOP, bit-for-bit) | new (default) |
|---|---|---|
| `PlayerConfig.reroll_ev` / `pack_ev` / `fool_order` | False / False / False | **True / True / True** |
| `HandConfig.fool_order` | False | **True** |

| gate | before | after |
|---|---|---|
| 126-seed `gate_ev_player.py --procs 12`, **fast** ante-1 clear | 94.4% | **94.4%** |
| … fast mean final ante | 4.762 | 4.627 |
| … fast matches won (the gate's MLB row) | — | 2.4% |
| … **full** ante-1 clear | 96.0% | **96.0%** |
| paired h2h new-shop vs old-shop, 30 seeds × 2 seatings | — | **50.0%** [38.3, 63.3] |
| … 126 seeds × 2 seatings (252 matches) | — | **54.4%** [48.0, 61.1] |
| Arcana pack take rate | 40.1% | **62.5%** |
| Celestial pack take rate | 45.9% | 50.7% |
| Standard pack take rate | 32.9% | **22.3%** |
| rerolls in one shop visit, maximum observed | **1** (a hard rule) | **7** |
| `python -m pytest ev -q` | 341 | **374** |

**The take-rate ordering is now the human one.**  Tagg's prior is that celestial is the
weakest of the big three and arcana is a huge deck-fixing play; the agent had that inverted
(celestial 45.9% > arcana 40.1%).  After: arcana 62.5% > celestial 50.7%, and the inversion
is gone because the shop now prices what a tarot does to *this* deck instead of ranking pack
families in a fixed order.

**One half of the reroll design measures NEGATIVE and is reported as such** (§2.5).  Tagg's
item 1 has two parts: remove the ≤1 cap, and let the roll compete with buying.  The first
part ships and helps; the second loses the h2h at every hurdle tried (28.3% – 43.3% over 60
matches, against 48.3% with the reroll off entirely), and the shape of the loss is
identifiable — the build proxy prices a joker by its effect on the NEXT blind, which is not
why a human rolls.  So `reroll_guard` (default True) keeps the old rule's two money guards
and the EV rule replaces the **cap**; `reroll_guard=False` is the competing version, kept
and documented as the thing to re-measure once V replaces the proxy.

## 1. What the old shop tier was, precisely

`EV_NOTES.md` §4 tier (3), unchanged since W3:

* packs, vouchers and playing cards: bought when affordable after the ante's interest floor
  `min(25, 5·ante)`, scored `0.015 − 0.002·pref` in the fixed order
  buffoon > celestial > arcana > standard > spectral;
* tarots and spectrals: `lam_money · (3 − price)` — "worth its price only when cheap";
* reroll: at most ONE per visit, never below $25 after the cost, and only when nothing on
  the shelf is worth buying;
* consumable uses: a flat +5.0 (planet) / +4.0 (other) bonus, in `consumable_idx` order.

Three of those four are *rules*, not valuations, and that is what the diagnosis found: the
agent bought what the myopic proxy could price (planets = immediate levels) and passed what
it could not (tarots = horizon), and could not express deep rolling at all.

## 2. Rerolls (`PlayerConfig.reroll_ev`)

### 2.1 The valuation is Monte-Carlo, and the analytic version is why

The obvious build is analytic: `hit.shop_slot_distribution` gives the exact distribution of
one fresh shop slot over the generator's OWN culled pools (`generate.get_current_pool` with
an explicit rarity, zero RNG consumed — STATS_NOTES §2), price every pool member with
`hit.pool_dollar_value`, and take order statistics.  It was built first, and it does not
describe this player.  Over 40 seeds and **297 real shelf jokers**, valuing the same joker
both ways (`hit.pool_dollar_value` versus the build proxy's own gain, converted at
`lam_money`):

| quantile | 10th | 50th | 90th | 99th |
|---|---|---|---|---|
| pool model $ | 4.84 | 7.15 | 10.01 | 12.87 |
| build proxy $ | −1.60 | 0.00 | 43.07 | 142.01 |

Half of every shelf is worth **nothing** to this build and a tenth of it is worth an ante.
`pool_dollar_value` is `strength(1..10) × coherence × $2.2` — it was calibrated to CLASSIFY
pool members against an $8 "hit" threshold, and it has no room for the tail that makes
rolling correct.  No monotone rescaling fixes it: the one fitted here (×2.4, matching
`E[max(0, surplus)]` per shelf joker, $2.33 pool vs $12.23 proxy) makes the *typical* draw
look like a bargain and rolled the player broke — 80-seed profile, mean final ante 4.55 →
4.21, interest income 22.0 → 7.6, joker take rate 28.7% → 20.3%.  The same measurement on
shelf CONSUMABLES is $1.86 pool vs $2.52 proxy, i.e. the gap is joker-specific, which is
exactly what a board-multiplier model versus a deck model should look like.

So the fresh shelf is valued the way every other row in this tier is valued — with the build
proxy, on real draws.  `_fresh_shelf_values` takes `cfg.reroll_worlds` (2) determinized
clones (`clone_determinized`, W2: the keyed RNG is replaced wholesale, so `reroll` on the
clone is a genuine draw from the same culled pools on a decorrelated stream, never the true
seed's), rerolls each, and scores the best purchase its shelf offers.  Only
`shop.SHELF_KINDS` count: a reroll leaves the voucher and both booster slots untouched
(shop.py:496).

Two accounting details that were both bugs before they were features:

* the world's balance is reduced by the roll's price, so a world cannot "buy" something the
  post-roll balance cannot afford — and the same amount is **added back into the value**,
  because `base_value` was measured at the pre-roll balance and `build_proxy` prices money.
  Without the add-back the fee is charged twice;
* the worlds are seeded from the OBSERVABLE STATE only, never `self.seed`.  With ε = 0 the
  fast player must be a function of the state alone, or the two seatings of one h2h stop
  being mirror images of one trajectory — `test_h2h.py::
  test_seatings_are_mirrors_for_an_identical_matchup` catches it, and did (394 vs 393 steps).

### 2.2 The cost of a roll — three terms

`_reroll_costs(game, n)` returns the true cost of the 1st..n-th roll of this visit.

**(a) The escalating price.**  `shop.reroll_shop` adds $1 per reroll inside a visit and
`game._end_blind_and_enter_shop` resets it to $5 at the next shop; free rerolls (Chaos the
Clown) are consumed first.

**(b) The interest threshold** — Tagg: "interest thresholds are probably the most
important".  `interest_cost(game, cfg, spend)` is `interest_rounds · interest_weight ·
(interest_now − interest_after(spend))`, where the two interest terms are the engine's own
formula (`min(dollars // 5, interest_cap)`, game.py:2010).  It is exactly 0 above the cap
and jumps by one round's dollar for every $5 breakpoint the plan crosses on the way down,
and the *marginal* interest of the k-th roll is charged to that roll.

This deliberately does NOT reuse `economy.interest_loss`.  That module discounts the
shortfall geometrically over the shops remaining to ante 8 (`decay = 0.85`, STATS_NOTES §3),
which prices a $4 Arcana pack at ante 1 at **$4.55** of forfeited interest — more than the
pack.  Under it the shop bought almost nothing.  This player already has a documented answer
to "what is a dollar that still earns interest worth" and it is `build_proxy`'s
`money = $ + 2·interest_weight·interest` (EXTRACT_NOTES §2 quotes the same 0.16/$ marginal
rate); `interest_rounds = 2` reproduces it exactly, so a pack row and a joker row are
compared on the same money.  `test_interest_cost_matches_build_proxys_own_money_term` pins
the identity against `build_proxy` itself.

**(c) The spread across the ante's shops.**  GENERATION_SPEC §8.3: "the ante-N shop queue is
the sequence of values of the `'...sho<a>'`-family streams", and a reroll simply advances
that shared pointer.  Three shops draw from one ante's queue — the post-Boss shop (where
`_end_round` has already bumped `game.ante` while `blind_idx` is still 2), then the ones
after Small and Big — so `shops_left_on_queue` reads the position straight off `blind_idx`
(2 → 3 shops, 0 → 2, 1 → 1).  The price resets at each of them while the queue does not, so
while more shops on this queue remain the $1-per-roll escalation is **avoidable**: the same
queue depth is available at $5 one blind later.  Rolls above the base price are therefore
charged `reroll_defer_delta` (0.75) of their escalation premium a second time.  At the last
shop of the queue — `shops_left_on_queue == 1`, which is the shop immediately before the
Boss and, under MLB, before the Nemesis — the term is exactly 0: depth not bought now is
lost.

### 2.3 The stopping rule

Optimal-stopping backward induction over at most `reroll_max_depth` (8) rolls on the sampled
fresh-shelf values `X`:

```
V_K = E[max(0, X)]                                 # last allowed roll: take it or leave it
V_j = E[max( max(0, X), V_{j+1} − cost_{j+1} )]     # ... or pay for another shelf
plan = V_1 − cost_1
```

"take the best thing this shelf shows, or pay for another shelf, whichever is worth more".
The rule ITERATES itself: the player is re-asked after every reroll, `game.reroll_cost` has
moved, and the same induction runs on the new schedule — no per-visit memory (which is what
keeps `EV_NOTES.md` §5's "a run is a function of `(seed, budget, cfg)`" true; `reset()`
clears the two caches this workstream adds and
`test_a_new_arm_run_is_a_function_of_the_seed_alone` pins it).

`reroll_hurdle` (2.0) multiplies the cost side.  It is a correction, not a taste knob: the
plan takes a `max` over 2 sampled worlds of a `max` over 2 shelf slots of a NOISY proxy
estimate, and then reuses the same 2 worlds inside the induction's own `max` — three
winner's-curse layers, all biased up, against a realisation that gets the true value.  Swept
(30-seed h2h, guarded rule): 0.5 → 45.0%, 1.0 → 45.0%, **2.0 → 50.0%**.

### 2.4 Money above the interest cap

`build_proxy` prices a dollar at a constant `lam_money` whatever the balance, and above the
interest cap that is simply wrong: the marginal dollar earns nothing and, measured, this
player never spends it (it ends MLB matches on $32–53 of cash).  A roll's cost is therefore
discounted toward `reroll_rich_floor` (0.25) as the balance runs past `5 · interest_cap`.
This is the one thing the old `reroll_floor = 25` rule got right — "roll when you are at the
cap and there is nothing to buy" — and the first EV draft lost it.

### 2.5 The race term, and the negative result

**(d) Race-conditional aggression.**  `_race_aggression` returns `1 + race_aggression ·
pressure`, which divides the roll's cost — the money → P(win) exchange rate rising when you
are behind, Tagg's "$20 on a $6 reroll before the PvP is reasonable if you feel behind".  It
is **exactly 1.0 in vanilla** (`game.mlb` gates it).  Under MLB, `pressure` is
`clip01(2·(0.5 − p_win))` when `ev/race.py`'s calculator is bound — a 50/50 race is neutral,
a lost-looking one maximal — and `clip01(lives lost / starting lives)` otherwise, topped up
by 0.25 when the next blind is the Nemesis itself.  A bare `BalatroGame` cannot see the
opponent's lives or curve, so `EVPlayer.bind_race(match, player_idx)` takes them from the
live `MLBMatch` and `adapt_match_player` calls it before every `act`; anything that drives a
lone game leaves it None and the term is neutral.  The `p_win` DP is cached on
`(ante, both lives, log length)`.

**The negative result.**  The design says the reroll should compete with the buy rows — roll
for something better instead of buying what is here.  Scored that way (the reroll row's
`plan` is a proxy-value change, exactly like a buy row's gain, so the argmax settles it), it
loses:

| 30-seed h2h, new vs old shop | win rate | lives margin | mean final $ (new / old) |
|---|---|---|---|
| reroll competes with buys, `reroll_hurdle` 1.0 | 28.3% [16.7, 40.0] | −1.250 | 3.0 / 30.2 |
| … 2.0 | 43.3% [31.7, 56.7] | −0.283 | 26.4 / 30.6 |
| … 4.0 | 30.0% [18.3, 41.7] | −0.683 | 42.4 / 28.4 |
| reroll off entirely (`reroll_ev=0`, packs only) | 48.3% [36.7, 60.0] | +0.117 | 40.3 / 31.9 |
| **guards kept, EV drives the DEPTH (shipped)** | **50.0% [38.3, 63.3]** | **+0.250** | 31.9 / 31.2 |

The cause is identifiable, and it is not the stopping rule.  Ask the sampled-shelf question
at the two moments a visit contains:

* at the START of a visit, `E[best purchase on a fresh shelf]` is **0.093** proxy units
  (measured over 196 real shelves) — a roll costs 0.058, so rolling looks good;
* AFTER the visit's buying, the same quantity is **0.016**.

The player is richest before it buys and poorest after, and a fresh shelf is only worth
anything if you can afford what is on it.  Rolling first forfeits the current shelf's own
0.093 to draw another 0.093 for $5.80, which is plainly bad; rolling last draws 0.016 for
$5.80, which is also bad.  There is no moment in a visit at which this player's own
valuation wants to roll — and the valuation is the myopic build proxy, which prices a joker
by its effect on the NEXT blind while a human rolls for what a joker is worth over a run.
Tagg's prior and the proxy disagree, and the shop cannot act on the prior without a better
V.  **This is the single clearest statement I can make of what V is for.**

So `reroll_guard` (default True) keeps the old rule's two money guards — roll only when
nothing on the shelf is worth buying, and only while the balance stays above `reroll_floor`
— and the EV rule replaces the `max_rerolls_per_visit = 1` **cap**, which is the part of
Tagg's item 1 the measurement supports.  `reroll_guard=False` is the competing version, kept
because it is the thing to re-measure the day V replaces the proxy.

Guarded, and at 126 seeds rather than 30, the reroll component is worth roughly its own
weight: rerolls + Fool alone score 50.4% [44.4, 56.7] against the old shop, and adding them
to the pack changes moves the pair from 48.4% to **54.4%** (§6b).  That is the honest
summary — the cap removal is not what wins, but it is not what loses either, and it is the
component whose ceiling moves when the valuation improves.

Effect on depth, 126 seeds:

| | old | new |
|---|---|---|
| visits with 0 / 1 / 2 / 3 / 4 / 5 / 6 / 7 rolls | 1093 / 130 / 0 / 0 / 0 / 0 / 0 / 0 | 1117 / 47 / 14 / 9 / 3 / 3 / 1 / 1 |
| max rolls in one visit | **1** | **7** |
| rolls per visit | 0.106 | 0.119 |
| rolls per ROLLING visit | 1.00 | **1.82** |
| % of visits with a roll | 10.6% | 6.5% |
| $ spent on rolls per run | 3.49 | 5.54 |

Fewer visits roll and the ones that do roll go deeper and spend more — the cap is what
moved, and the money guard is what kept the frequency down.

## 3. Packs (`PlayerConfig.pack_ev`)

### 3.1 What is EV and what is prior

`_pack_ev` is `E[sum of the best `choose` of `size` iid pack cards] − price −
interest_cost`, by the layer cake `Σ_levels (u_j − u_{j−1}) · Σ_{i≤k} P(Bin(n, S_j) ≥ i)`
over the pack's own content pool (`hit.pack_slices`, split out of `pack_p_hit` for this —
§7).  Pack picks are FREE once the pack is bought, so a pack card's value is its full value,
not a surplus; that asymmetry versus a reroll (where every hit still has to be paid for) is
most of why packs beat rolls at low money, and it now falls out of the arithmetic instead of
being asserted by a preference order.  Consumable packs are capped at the free consumable
slots; a Buffoon pack's jokers are charged the cycle toll of §3.3.

The **low-money guard is unchanged** — a pack must leave the ante's interest floor
`min(25, 5·ante)` intact, unless `P(clear next) < 0.6` and the build needs the help.  And a
consumable/joker pack that clears that guard is never scored below `pack_take_floor`
(0.012), the middle of the OLD rule's own pack band (0.007..0.015).  That is deliberate and
it is prior, not measurement: Tagg's "packs are almost-always-take unless broke" is also
what the old rule did, so **the money guard decides WHETHER and the EV decides WHICH** —
replacing the fixed family order, which is what the design asks for.  Measured, without the
floor the new arm ends MLB matches on $42 of unspent money and the packs-only h2h is 43.3%;
with it, 48.3%.  Standard packs are the one family exempt from the floor — theirs is the
family the measurement is allowed to talk down (§3.4).

### 3.2 Arcana: what a tarot does to THIS deck, measured

`hit.tarot_value` prices every tarot at a flat $4 and STATS_NOTES §1 flags it as a gap
("real Balatro tarot power varies a lot").  That flat number IS the inversion: a deck-fixing
tarot and a dead one looked identical, so nothing separated an Arcana pack from a Celestial
pack except the fixed order.

`EVPlayer._deck_effects` measures five elementary deck changes with the build proxy on a
clone, dividing the P(clear)+strength delta by `lam_money` to land back in dollars:

| effect | what is mutated | who uses it |
|---|---|---|
| `suit3` | 3 off-suit cards → the deck's modal suit | The Star / Moon / Sun / World |
| `steel1` | one plain card → Steel | all 8 enhancement tarots, and a Standard pack's enhanced cards |
| `destroy2` | the two worst cards removed | The Hanged Man |
| `rank2` | two cards' rank +1 | Strength, and Death's upgrade |
| `add_plain` | one average plain card ADDED | a Standard pack's base card |

The shape *within* the enhancement family comes from `hit.card_value`'s own enhancement
table (Steel 4.5, Gold 3.5, Glass/Lucky 3.0, Mult 2.5, Bonus/Wild 2.0, Stone 1.0), so one
measurement covers all eight.  `add_plain` is averaged over two probes — the deck's rarest
and commonest rank — because duplicating a rank the deck is already deep in reads as a pair
bonus and duplicating a thin one reads as dilution.  Memoised per (deck shape, levels, ante)
AND per effect, so a shop with no Arcana pack and no shelf tarot never pays for `suit3`.

A representative reading (seed 11111111, ante 1, the fresh Red deck): `suit3` **$15.14**,
`steel1` $1.57, `destroy2` $9.56, `rank2` $5.72, `add_plain` $3.59.  That single line is the
whole diagnosis: suit conversion is the best thing a $3 card can do to this deck, and one
Steel card is worth almost nothing — so The Chariot is $3.14 and The Star is $15.14, where
the flat table said $4 and $4.

What stays tabulated is what the proxy cannot see: the creators (The Emperor $6, The High
Priestess $6, Judgement $10 — 0 when the joker slots are full, because the engine refuses
the use), The Wheel of Fortune ($3 ≈ ¼ of a joker edition), and the two money tarots, which
are exact: The Hermit is `min(dollars, 20)` and Temperance is `min(50, Σ joker sell values)`.
The Fool is priced as whatever `run_state.last_tarot_planet` holds, and 0 when that is empty
or itself — which is what the engine does (`consumables.apply_tarot`:136-137).

The shelf Tarot/Spectral buy row gets the same treatment: the proxy sees a bought consumable
only as money leaving (a use needs cards in hand, which no SHOP state has), so its own value
is added on top of what the proxy did see.  Shelf tarot take rate 27.8% → 36.7%.

### 3.3 Buffoon: the cycle-sell, priced

Tagg's trick: with a full board and a good joker on the shelf, opening a Buffoon pack is
near-riskless, because the floor is "sell the worst of (pack joker, weak owned joker) and
take the shelf joker" — something you were going to do anyway.

Two halves, and the engine forces the order.  In `BOOSTER_OPEN` the only legal actions are
`pick_booster` and `skip_booster` (game.py:1519-1528) and `_can_grant_choice` refuses a
joker with no free slot, so the sell must happen in the SHOP, before the pack is opened.
So:

* `_cycle_cost` prices putting one more joker on a FULL board: `min_j (value(j) − sell(j))`
  over the owned jokers, floored at 0 (a joker whose sell price already exceeds its value
  should be cycled anyway), and +∞ when every joker is Eternal.  Both sides use the same
  cheap pool model, so it is a like-for-like A-vs-B toll, not a priced card.  It is charged
  to every joker a reroll or a Buffoon pack could show.
* the SHOP's sell row now sees the Buffoon pack: the value of freeing a slot is the better
  of "buy the best shelf joker" (which is the floor that makes the trade near-riskless, and
  is what the old rule already did) and "open the pack with the slot free" — the pack's own
  row is then priced with `_cycle_cost == 0` and opens on the next step.

### 3.4 Standard: measured, and it stays low

`hit.pack_p_hit` returns `(0, 0)` for a Standard pack — "correctly never recommended, but
only because the floor is 0, not because a real evaluation was done" (STATS_NOTES §5).  The
model here is the generator's own recipe (`generate.open_pack`:1283-1287): Enhanced iff
`stdset > 0.6` (P = 0.40, uniform over the 8 Enhanced centres), a seal iff
`stdseal > 1 − 0.02·10` (P = 0.20), an edition from `poll_edition(.., mult=2)` (≈ 8%), priced
with `add_plain` + `steel1`-scaled enhancements + the seal/edition terms.  It is deliberately
NOT a build-fit model.

Take rate 32.9% → **22.3%**, which is the honest number and not a fudge in either direction:
the family is exempt from `pack_take_floor` precisely so the measurement can talk it down,
and it did — but only to 22%, not to 0, because `add_plain` on a real deck is not negative.
An early draft with a flat `card_value` put it at 4.2% and cost the run its main supply of
enhanced cards (deck enhancements 1.26 → 0.48 per run); the measured version keeps them
(1.25 → 1.22).

### 3.5 The pick side

The old booster-pick rule floored every pick the proxy could not value at `1e-4`, which made
every unvalued pick TIE and let `_action_sort_key` choose — i.e. the Arcana pick was
arbitrary.  Tarot and Spectral picks now carry their own $ value on top of the proxy's
number.  Planets need no such term (the proxy auto-uses them); playing cards need none (the
proxy sees the deck change).

## 4. The Fool sequencing (`fool_order`)

The Fool creates a copy of `run_state.last_tarot_planet` — the Tarot or Planet used LAST,
and The Fool itself never overwrites it (`consumables.apply_tarot`:110).  So when a batch of
consumables is going to be used anyway, the ORDER is worth something: drain the batch
weakest-first so the copy is of the best card, and hold The Fool back while a better copy is
still on the way.

Two hooks, because the two states rank uses differently:

* `hand._fool_ordered` (SELECTING_HAND): the consumable candidates' EVs are **permuted
  inside the consumable group**, best-first order receiving the ascending EVs.  The group's
  position relative to every play and discard — and therefore whether a consumable is used
  at all this decision — is bit-identical to the unordered ranking; only which one goes
  first changes.  The Fool is dropped below the group while some held use would give it a
  better copy.
* `EVPlayer._fool_shift` (SHOP): the shop ranks uses by a flat +5.0/+4.0 bonus, so there are
  no EVs to permute; the shift is `−fool_order_dollars` per $ that a use is better than the
  rest of the batch, plus `−fool_defer_penalty` on The Fool itself.

Both are inert unless `c_fool` is actually in `consumable_hand` (1 tarot in 22), and inert
under `fool_order=False`.  First-order, and the documented gap is that the EVs being
permuted are position values rather than standalone card values.

## 5. Before / after profile — `ev/scripts/shop_profile.py`

    python ev/scripts/shop_profile.py profile --seeds 126 --arms old,new --procs 12 \
        --to-ante 8 --out results/shop_profile_2026-08-26.json
    python ev/scripts/shop_profile.py h2h --n-seeds 126 --procs 8 --max-steps 4000

A *visit* is one SHOP entry through its `leave_shop`, including any BOOSTER_OPEN excursion.
An *offer* is one shelf/voucher/booster ITEM instance the visit showed: the `shop_joker_max`
slots at entry plus `shop_joker_max` more per reroll, plus the voucher and the two booster
slots (a reroll touches neither — shop.py:496).  `take_rate = takes / offers`.

> **On the campaign-log baseline.**  The lead's 80-seed sweep reports arcana 20.5 / celestial
> 23.2 / buffoon 26.8 / standard 16.5 / spectral 17.5 and vouchers 13.3, and rerolls in 6.4%
> of visits.  This script's numbers are ~2.2–2.5× higher across the board because the
> denominator differs (item instances here; that sweep appears to count an offer per shop
> DECISION, of which a visit has several).  The **ordering is identical** — buffoon >
> celestial > arcana > spectral > standard — which is what makes the two measurements the
> same measurement, and the ordering is what the gate is about.

**126 ground-truth seeds, to ante 8, both arms in one batch (so the timings are
load-comparable):**

| take rate (takes / offers) | old | new |
|---|---|---|
| joker | 29.2% (573/1963) | 28.9% (554/1918) |
| tarot | 27.8% (103/370) | **36.7%** (134/365) |
| planet | 55.4% (230/415) | 50.1% (217/433) |
| playing card | 4.2% (1/24) | 48.1% (13/27) |
| voucher | 22.0% (236/1072) | 21.0% (218/1036) |
| pack: arcana | 40.1% (256/639) | **62.5%** (404/646) |
| pack: celestial | 45.9% (318/693) | 50.7% (343/677) |
| pack: buffoon | 46.7% (156/334) | 40.2% (130/323) |
| pack: standard | 32.9% (217/660) | **22.3%** (138/620) |
| pack: spectral | 35.0% (42/120) | 45.2% (56/124) |

| per-run mean | old | new |
|---|---|---|
| ante-1 clear | 0.944 | **0.944** |
| ante-2 clear | 0.817 | **0.817** |
| $ at shop entry | 20.79 | 22.48 |
| $ at leave_shop | 13.55 | 15.53 |
| interest income | 22.68 | **23.20** |
| $ on rerolls | 3.49 | 5.54 |
| $ on buys | 79.44 | 80.97 |
| tarots used | 2.87 | **3.98** |
| planets used | 4.73 | 4.42 |
| deck enhancements | 1.25 | 1.22 |
| deck seals | 0.61 | 0.34 |
| jokers owned | 4.26 | 4.17 |
| blinds cleared | 9.74 | 9.51 |
| mean final ante | 4.762 | 4.627 |
| final $ | 22.57 | 26.33 |
| shop ms mean / p95 | 8.85 / 24.75 | 12.76 / 37.57 |
| hand ms mean | 6.21 | 6.52 |

Read honestly:

1. **The take-rate ordering is fixed** and arcana rose 22 points, which is the gate's own
   "arcana take up … this must RISE substantially".
2. **Reroll depth is uncapped** (§2.5's table) but the *frequency* fell, because the money
   guard the measurement kept is stricter in practice than the old $25 floor once the player
   is richer.  "Depth up substantially" is true of the conditional depth (1.00 → 1.82 rolls
   per rolling visit, maximum 1 → 7) and false of the rate.
3. **$ at entry did NOT drop** — it rose 20.79 → 22.48, and interest income rose with it
   (22.68 → 23.20), so the brief's "spending is the point / interest should not collapse"
   pair came out the wrong way round on the first half.  The player converts more of its
   money into deck (tarots used +38%) and less into shelf packs, and ends runs on more cash.
4. **Mean final ante is down 0.135 and blinds cleared down 0.23**, both inside the noise of
   126 unpaired runs but both pointing the same way; the h2h (§6) is the paired test.
5. Deck seals fell (0.61 → 0.34) with the Standard take rate — the honest cost of §3.4.
6. Shop decision time is up 44% (8.85 → 12.76 ms): two determinized world evaluations per
   reroll decision plus up to five proxy builds per deck change.  Both are memoised per
   player and per deck shape; the p95 is a cache miss.  This is over the ~5 ms the old tier
   cost and is the clearest place to buy performance back (§8).

## 6. Gates

**(a) 126-seed EV gate** — `python ev/gate_ev_player.py --procs 12`,
`results/ev_player_gate_shop_2026-08-26.{md,json}`:

| | fast | full | greedy |
|---|---|---|---|
| ante-1 clear | 94.4% [90.5, 98.4] | 96.0% [92.1, 99.2] | 31.7% [23.8, 39.7] |
| ante-2 / 3 / 4 clear | 81.7 / 75.4 / 59.5% | 85.7 / 77.0 / 67.5% | 0 / 0 / 0% |
| matches won | 2.4% | 2.4% | 0.0% |
| mean final ante | 4.68 [4.34, 5.02] | 4.87 [4.56, 5.18] | 1.32 |
| mean blinds cleared | 9.63 | 9.88 | 2.16 |
| $ at ante 3 | 20.70 | 20.87 | — |
| hand ms mean / p95 | 4.12 / 12.05 | 70.87 / 171.62 | 2.21 / 2.91 |
| shop ms mean | 8.8 | 9.8 | 0.1 |
| draw-order invariance | 737/737 | 741/741 | — |

**The gate's ante-1 criterion is ≥ 95% and this reads 94.4%, and the drop is NOT this
workstream's.**  The paired 126-seed run of §5 puts the OLD arm at 94.4% too, on the same
seeds and the same driver — the pre-W-SHOP player on today's engine already clears 119/126.
The 95.2% in `EXTRACT_NOTES` §0 predates W-FIX, whose own note records "EV gate: fast
bit-stable (1/126 rows)"; that one row is this seed.  Per seed, 4 of 126 ante-1 outcomes
move under W-SHOP (`BQ6A2D42`, `D7Y5419A`, `M4LV5E89`, `WBC1DGJQ`) — two each way, netting
zero.  18 of 126 ante-2 outcomes move, also netting zero (81.7% both arms).  **A re-baseline
of the gate's ante-1 line is a lead decision, not mine; nothing here should be read as
W-SHOP clearing or failing that criterion.**

**(b) Paired h2h, new-shop vs old-shop**, `ev/scripts/shop_profile.py h2h` — same player
both sides, only `shop_arm_cfgs` differs, both seatings per seed, ε = 0, `--max-steps 4000`,
8 procs:

| seeds | matches | new-shop wins | CI | lives margin | mean $ new / old |
|---|---|---|---|---|---|
| 30 | 60 (0 undecided) | **50.0%** | [38.3, 63.3] | +0.250 | 31.9 / 31.2 |
| **126** | **252 (0 undecided)** | **54.4%** | **[48.0, 61.1]** | **+0.194** | 33.0 / 30.4 |

The 126-seed number is the one to quote — the 30-seed CI is 25 points wide.  It is the
thesis test and it passes: the new shop wins 137 of 252 decided matches and ends 0.19 lives
ahead, on the same seeds, both seatings, with only `shop_arm_cfgs` differing.

Attribution, same 252 matches per row:

| arm | win rate | lives margin | mean $ new / old |
|---|---|---|---|
| packs + Fool only (`reroll_ev=0`) | 48.4% [42.5, 54.8] | −0.119 | 38.3 / 30.5 |
| rerolls + Fool only (`pack_ev=0`) | 50.4% [44.4, 56.7] | +0.004 | 29.8 / 32.8 |
| **all three (shipped)** | **54.4%** [48.0, 61.1] | **+0.194** | 33.0 / 30.4 |

Neither component wins on its own — each is a wash inside its own CI — and together they
beat both.  The mechanism is visible in the money column: packs alone convert money into
build but leave $38 unspent, rerolls alone spend down to $29.8 but buy less build, and the
pair lands at $33 with the highest joker count and the only positive lives margin.  Two
notes of caution: the three rows share seeds so the differences are correlated, and 54.4%'s
lower bound sits at 48.0 — this is "the new shop is at least not worse, and probably a
little better", not a demonstrated 4-point edge.

**(c) Suites.**

| suite | result |
|---|---|
| `python -m pytest ev -q` | **374 passed** (341 before + 33 new in `test_shop.py`) |
| `python -m pytest engine/tests -q` | **1715 passed / 10 skipped / 3 xfailed** |
| `python -m pytest tests -q` | **1073 passed / 2 xfailed** |
| `python -m pytest stats/tests -q` | **50 passed** |
| `python -m oracle.engine_parity --antes 1-8 --rerolls 5` | **126/126 exact through ante 8** |

**No engine code was changed**, and the parity run above confirms it.  One engine-side
FIXTURE did have to be re-captured, and it is worth being explicit about why:
`engine/tests/engine_tests/pvp_canonical_transcripts.json` pins the sha1 chain of four
`MLBMatch` runs played by `ev:fast` on both sides.  It pins the PLAYER's canonical path, so
any change to what the player buys moves it by construction — all four seeds moved here,
and `test_canonical_transcripts_are_unchanged` failed until they were re-captured.  This
follows W-FIX's own precedent on the same file (which re-captured three of the four seeds
in August): each changed row keeps its pre-W-SHOP values under
`superseded_2026-08-26_wshop` and the fixture's `note` says what moved them, so the original
W-PVP claim ("adding the protocol changed nothing") stays auditable.  **Flagged for the
lead: this is the one file outside my stated ownership that this workstream edits, and it is
a re-capture, not a loosened assertion.**

## 7. What was touched outside `ev/`

`stats/hit.py` gains ONE public function, `pack_slices(game, pack_kind, cfg, rs)`, split out
of `pack_p_hit`'s body so a caller that needs the pool MEMBERS — to compute order statistics
over its own valuation rather than a threshold hit rate — does not have to reach into the
private `_joker_slices` / `_consumable_slice`.  `pack_p_hit` now calls it and is otherwise
unchanged; `test_stats_pack_slices_is_additive_and_agrees_with_pack_p_hit` pins the
agreement, and `stats/tests` is green.

`engine/tests/engine_tests/pvp_canonical_transcripts.json` was re-captured — see §6(c) for
the reasoning and the audit trail.  No engine source file was touched.

`ev/player.py` also puts `stats/` on `sys.path` (exactly as `ev/h2h.py` already did) so
every entry point — the gate, the tournament, the advisor, a bare `import player` — gets the
same decision machinery.  A missing or broken `stats` package degrades to the pre-W-SHOP
rules rather than raising, the same contract `_rank_with_stats` already documents.

## 8. Open issues / the next lever

1. **The reroll wants V, not a better stopping rule** (§2.5).  Everything in §2.2–2.4 is
   arithmetic on top of one number — what a fresh shelf is worth — and that number comes
   from a proxy that values a joker over one blind.  `reroll_guard=False` is the version to
   re-measure the day V is good enough to replace `build_proxy` in `_best_purchase_value`;
   the plumbing needs no change, only the valuation.
2. **Vouchers are still a flat +0.02** when affordable after the interest floor.  Out of this
   workstream's scope, and now the last un-priced row in the tier — `hit.voucher_value`'s
   curated table (24 of 32 vouchers) is sitting right there.
3. **Shop decision time is 12.8 ms mean** (was 8.9 in the same batch).  The two levers are
   `reroll_worlds` (2) and the five `_deck_effects` probes; the cheapest real win is to skip
   `_fresh_shelf_values` entirely at shops where the money guard already refuses a roll.
4. **`gate_ev_player.py` still has no config knob** (W-EXTRACT's open issue 2, unchanged), so
   the gate's before/after cannot be an A/B in one process; §5's paired profile is what
   stands in.  A `--shop-arm old|new` flag would make §6(a)'s "the drop is not ours" claim a
   one-command check instead of a two-measurement argument.
5. **The cycle-sell has no BOOSTER_OPEN half.**  The real game lets you sell a joker with a
   pack open; this engine's `legal_actions` does not (game.py:1519), so the trick is only
   available if the shop-side sell fires first.  An engine change was NOT made (the brief
   says stop and document); it is a legality gap, not a fidelity bug in anything scored.
6. **`_deck_effects` measures the deck, not the play.**  Strength/Death/The Hanged Man are
   priced by a representative mutation, not by target selection; the hand player's own
   `_consumable_candidates` still chooses targets from the best play's scoring cards.
