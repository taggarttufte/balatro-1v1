# W3 — effect-roll keys (P1-effects), 2026-08-21

**Scope:** `scoring.py`, `jokers/*` (except `base.py:JokerInstance.clone()`), the probability
rolls in `consumables.py` (Wheel of Fortune), the effect rolls in `game.py` (glass, Purple
Seal, Madness, Amber Acorn, Cerulean Bell, Hook, Crimson Heart), `card_selection.py`'s
scorer RNG, and the deletion of `game.rng`. New: `round_cards.py`,
`tests/sim_tests/test_effect_keys.py`, this file.

**Result.** Every stochastic effect now draws `game.run_state.rng.pseudorandom(<key>)` /
`pseudorandom_element` / `pseudoshuffle` with the key string from `mp/rng/keys.py`, with the
same draw count and order per trigger as the Lua call site. `game.rng` (the legacy
`random.Random` single stream) is gone; `import random` does not exist in any game-logic
module under `balatro_sim/` (`test_effect_keys.py::test_no_random_module_in_engine`). A hook
that needs a roll without a PRNG raises `jokers.base.MissingPRNG` — there is no unseeded
fallback.

## Gate numbers (run by me, end of session)

| suite | result |
|---|---|
| `python -m pytest mp/engine/tests -q` | **1321 passed / 10 skipped / 0 failed** (34 of them new, `test_effect_keys.py`) |
| `python -m pytest mp/tests/test_engine_invariants.py -q` | **14 passed** (incl. `test_effect_rolls_do_not_move_generation` ×3 — needs W2's delegated generation, which landed concurrently) |
| `python -m pytest mp/tests/test_engine_reachability.py -q` | **226 passed / 6 failed / 3 xfailed** (was 185 / 46 / 4; `python -m pytest mp/tests -q` overall 386 / 6 / 3) |
| `python -m mp.oracle.engine_parity --probe` | 11/12 ok; `keyed_rng` ok (`game.rng` gone); only `state_signature` missing (W2 nice-to-have) |

Reachability cleared by W3 (all previously red): `j_oops`, `j_turtle_bean`, `j_hiker`,
`j_gift`, `j_ancient`, `j_castle`, `j_mime`, `j_campfire`, `j_hologram`, `j_lucky_cat`,
`j_flash`, `j_hallucination`, `j_matador`, `j_perkeo`, `j_astronomer`, `j_chaos`,
`j_credit_card`, `j_invisible`, `j_idol`, `j_madness`, `j_8_ball`, `j_cartomancer`,
`j_certificate`, `j_dna`, `j_marble`, `j_riff_raff`, `j_seance`, `j_sixth_sense`,
`j_superposition`, `j_vagabond`, `j_red_card` (+ W2's booster state), `j_diet_cola` (+ W2's tags).

Still red (none are effect-roll bugs — see "deferred"): `j_four_fingers`, `j_shortcut`,
`j_smeared` (hand-eval flags, `hand_eval.py` has no flag support — W5),
`j_space` (bad luck: the first `space` draw is ≥ 0.25 on all 7 harness seeds — 0.911,
0.861, 0.884, 0.374, 0.951, 0.577, 0.359 — so a single-hand probe cannot see a 1/4; use
`repeats=2`), `j_flower_pot` (the probe plays a High Card with four suits; the real joker
reads the SCORING hand, card.lua:3807-3840, so only the Jack scores — probe scenario is
wrong per source), `j_chicot` (the probe crafts the hand AFTER `play_blind`, bypassing
`_apply_boss_start`'s deck debuff, so the baseline is not debuffed either; Chicot itself
works — `current_blind.disabled=True`, boss_key blanked, target un-multiplied).

## Key table as implemented (effect → key → draws per trigger → site)

| effect | key (keys.py) | draws / trigger | threshold | engine site |
|---|---|---|---|---|
| Lucky Card +20 Mult | `lucky_mult` | 1 per scoring pass (retriggers roll again) | `< normal/5` | `scoring.py:74` |
| Lucky Card $20 | `lucky_money` | 1 per scoring pass, after `lucky_mult` | `< normal/15` | `scoring.py:77` |
| Glass shatter | `glass` | **1 per scoring Glass card, after the whole hand** (not per retrigger — state_events.lua:957-963; the brief's "per trigger" was wrong, `keys.py` is right) | `< normal/4` | `game.py:1366` |
| Gros Michel extinct | `gros_michel` | 1 at round end | `< normal/6` | `jokers/chips.py:27` |
| Cavendish destroyed | `cavendish` | 1 at round end | `< normal/1000` | `jokers/chips.py:37` |
| 8 Ball | `8ball` then `Tarot8ba<ante>` | per scored 8 per pass, only if a consumable slot is free (incl. buffer) | `< normal/4` | `jokers/misc.py:258` |
| Purple Seal | `Tarot8ba<ante>` (same stream as 8 Ball; no probability) | per discarded Purple card with a free slot, before the discard hooks | — | `game.py:1403` (`grant_created("purple_seal")`) |
| Business Card | `business` | per scored face per pass (Pareidolia-aware) | `< normal/2` | `jokers/economy.py:32` |
| Bloodstone | `bloodstone` | per scored Heart per pass (Wild / Smeared via `is_suit`) | `< normal/2` | `jokers/mult.py:297` |
| Reserved Parking | `parking` | per held face per held pass (Mime/Red seal re-roll); roll consumed before the debuff check | `< normal/2` | `jokers/misc.py:684` |
| Space Joker | `space` | 1, in the `before` phase — the level-up applies to THIS hand | `< normal/4` | `jokers/mult.py:414` (`pre_score`) |
| Misprint | `misprint` | 1, `pseudorandom('misprint', 0, 23)` integer form | — | `jokers/mult.py:109` |
| Hallucination | `halu<ante>` then `Tarothal<ante>` | 1 per booster opened, if a slot is free | `< normal/2` | `jokers/misc.py:319` |
| Wheel of Fortune | `wheel_of_fortune` ×3 (roll, `pseudorandom_element` over editionless jokers in sort_id order, `poll_edition` guaranteed/no-neg) | 3 on a hit, 1 on a Nope; unusable with no editionless joker | `< normal/4` | `consumables.py:185` (`generate.wheel_of_fortune`) |
| Madness | `madness` | 1 `pseudorandom_element` over the other jokers (board order) on Small/Big select | — | `game.py:841` |
| Amber Acorn | `aajk` | 3 `pseudoshuffle`s of the sort_id-ordered board (blind.lua:195-201) | — | `game.py:890` |
| Cerulean Bell | `cerulean_bell` | 1 element draw over the hand (sort_id order) per blind | — | `game.py:1204` |
| The Hook | `hook` | 2 element draws over a sort_id-ordered copy of the hand, first pick removed | — | `game.py:1229` |
| Crimson Heart | `crimson_heart` | 1 element draw over the jokers (sort_id order) per play (the game rolls per draw-to-hand — count differs, stream is isolated) | — | `game.py:1299` |
| Invisible Joker | `invisible` | 1 element draw over the other jokers on sell (rounds ≥ 2) | — | `jokers/misc.py:362` |
| Perkeo | `perkeo` | 1 element draw over held consumables on shop leave | — | `jokers/misc.py:381` |
| To Do List | `to_do` | creation: draw until ≠ previous; round end: 1 draw over visible hands minus current — pool order `generate.HANDS_PAIRS_ORDER` | — | `jokers/misc.py:584,597` |
| Riff-raff | `rarity`-less `Joker1rif<ante>` (+`_resample<n>`), `edirif<ante>` | up to 2, slot-gated | — | `jokers/misc.py:280` |
| Marble Joker | `marb_fr` | 1 over `P_CARDS` key order, Stone card into the deck | — | `jokers/misc.py:440` |
| Certificate | `cert_fr` then `certsl` | 1 + 1, card with seal into HAND | — | `jokers/misc.py:655` |
| Vagabond / Superposition / Seance / Sixth Sense / Cartomancer | `Tarotvag<a>` / `Tarotsup<a>` / `Spectralsea<a>` / `Spectralsixth<a>` / `Tarotcar<a>` (via `generate.create_from_spec`) | 1 when the condition holds and a slot is free | — | `jokers/chips.py:94`, `misc.py:266,296,311,327` |
| Idol / Mail / Ancient / Castle | `idol<a>` / `mail<a>` / `anc<a>` / `cas<a>` | game-level, run start + after every blind (W2 rolls: `generate.start_run`, `game._round_end_resets` → `game.round_picks`; `round_cards.py` converts for the hooks and rolls lazily for game-less contexts) | — | `round_cards.py:73-90` |
| Oops! All 6s | — | `run_state.probabilities_normal = 2 ** n_owned` (`base.sync_probabilities`, recomputed on every hook context / `_play_hand` / WoF) | — | `jokers/base.py` |

`probabilities_normal` flows: `run_state.probabilities_normal` → `ScoreContext.probabilities_normal`
→ `base.prob_roll(ctx, key, odds)` = `pseudorandom(key) < normal / odds`.

## Architecture

* `ScoreContext` (`jokers/base.py`) carries `prng` (the run's `PseudoRandom`, or a private
  clone in a dry run), `run_state` (None in a dry run → card-creating effects roll but do not
  create), `probabilities_normal`, `round_cards`, `joker_slots`, `consumable_slots`,
  `consumables`, `blind_kind`, `hands_played`, `boss_triggered`, `lucky_trigger`, and the
  pending queues `pending_jokers` / `pending_cards` / `pending_destroy` (besides
  `pending_money` / `pending_consumables`).
* `score_hand(..., rng=<PseudoRandom>, run_state=, probabilities_normal=, round_cards=,
  joker_slots=, consumable_slots=, consumables=, boss_triggered=, hands_played=)`. `rng` is
  type-checked (must expose `pseudorandom` / `pseudorandom_element` / `pseudoshuffle`).
* `game._hook_ctx()` builds the full context for non-scoring hooks; `base.fire_hook(game,
  name, *args)` fires a hook on every joker with ONE shared context and drains;
  `base.drain_joker_state(game, ctx)` is the single routine that turns hook output into game
  state (money, created consumables → `game._materialize`, created jokers, created cards,
  destroyed cards, joker self-destruction incl. `gros_michel_extinct`); `base.sell_hooks(game,
  j)` = the sold joker's `on_sell` + `selling_card` on the rest (Campfire); `base.add_joker`
  / `remove_joker` do the `run_state` bookkeeping + `on_init` + Oops sync;
  `base.passive_modifiers(jokers)` exposes hand size / `bankrupt_at` / free rerolls /
  Astronomer for `_start_blind` and `shop.py`. **W2 already calls `sell_hooks`,
  `remove_joker`, `passive_modifiers`, `init_joker` from `shop.py` / `debug_add_joker`.**
* `HypotheticalScorer` (`card_selection.py`) clones `gs.run_state.rng` once per hand and
  `restore()`s the snapshot per candidate; `run_state=None` so a dry run never touches
  `used_jokers`. env_sim `_update_play_combos` 3.1 ms (218 subsets), env_v7
  `_best_hand_score` 0.86 ms. All 40 `test_env_rng_isolation.py` tests pass (3 adapted to
  `run_state.rng`).
* Scoring order changes (all cited in `scoring.py`'s docstring): Space Joker in the `before`
  phase; held-in-hand phase is per card per pass with `on_held_card` (Baron, Shoot the Moon,
  Raised Fist, Reserved Parking) and Red-seal / Mime repetitions gated on "had an effect";
  joker editions apply exactly once per joker in `joker_main` (Foil/Holo before, Polychrome
  after) — never on per-card passes (a Foil Greedy Joker on five Diamonds is +50, not +250).

## Bugs fixed (beyond the keys)

Photograph (per hand, first face of the scoring hand, retriggers); Ancient / Castle / Idol /
Mail-In Rebate read the game-level round cards (Castle chips no longer reset per round;
Mail pays $5 per matching rank, not $3 once); Flower Pot reads scoring cards with the Wild
fill rule; Swashbuckler excludes itself and uses real sell values; Stencil = (slots − jokers)
+ #Stencils, applied only with an empty slot, reads `game.joker_slots`; Stuntman −2 hand
size and Turtle Bean +h_size (−1/round, eaten at 0) applied in `_start_blind`; Gros
Michel/Cavendish actually destroyed (+ `gros_michel_extinct` pool flag); Madness X starts at 1
(was 0 → x0.5 halved the mult) and skips the Boss; Hanging Chad per hand (was per round);
Raised Fist uses nominal chips (Ace = 11, not 14); Even Steven excludes the Ace; Hiker +5
(was +4 and never scored) via `Card.bonus_chips` (copied by `Card.copy`); Lucky Cat wired
to the per-pass lucky trigger; Trading Card destroys the single discarded card, no RNG;
Sixth Sense / DNA are first-hand-of-round effects, not once-per-run; Sixth Sense destroys
the 6; Chicot disables the boss; Matador pays on `blind.triggered` (Hook, Tooth, Flint,
Crimson Heart, Arm, Ox) and on rejected hands (Eye, Mouth, Psychic); Gift Card +$1 sell
value on every joker; Egg uses real sell value; Business / Golden Ticket / Rough Gem pay
immediately; Wheel of Fortune unusable without an editionless joker; all ten sentinel
producers replaced by real keys drawn through `generate` (`"tarot"`, `"spectral"`,
`"common_joker"`, `"stone_card"`, `"random_enhanced_card"`, `"copy_card:*"`,
`"duplicate_joker"`, `"negative_tarot"` are gone; `"double_tag"` → `state['pending_tags']
= ['tag_double']` which W2's `sell_joker` hands to `tags.py`).

## Edits outside my nominal list (targeted, all in `game.py` unless noted)

* `BlindInfo.disabled` field; `_start_blind` passive block + hook block (now `fire_hook`) +
  Chicot + Madness; `_apply_boss_start` Amber Acorn; `add_card` fires `on_card_added`
  (Hologram); `_play_hand` boss rolls / `score_hand` kwargs / `drain_joker_state` / glass /
  `boss_triggered` / `_hands_played_round`; `_discard` Purple Seal + `fire_hook("on_discard")`;
  `__init__` seed fallback `secrets.randbelow` (no `random`); `_hook_ctx`; `clone()` rng lines +
  `_hands_played_round`.
* `card.py`: `Card.bonus_chips` field (+ `copy()`).
* `consumables.py`: removed the now-unused `import random`.
* `jokers/base.py`: `JokerInstance.sort_id` (creation counter, for `pseudorandom_element` /
  `pseudoshuffle` over the board). **W2: `JokerInstance.clone()` should copy `sort_id`**
  (readers use `getattr(j, "sort_id", 0)` so a clone without it still works, just with a flat
  order for Amber Acorn / Crimson Heart / Invisible / Ankh).
* Tests adapted: `test_determinism.py` (rng_of now raises), `test_clone*.py`,
  `test_env_rng_isolation.py` (3 sites), `test_jokers.py` (creation tests assert real keys
  + keys drawn), `test_seals.py`, `test_glass.py`, `test_held_and_deck.py`,
  `test_joker_order.py`, `test_joker_catalogue.py`, `test_edge_cases.py` (Castle),
  `test_consumables.py` (WoF), `test_game_keys.py` (passive list, WoF needs a joker).

## Call sites / follow-ups for W2 (P1-delegate)

W2 picked the helpers up concurrently: `_end_round`, `_start_blind`, `_end_shop`,
`_open_booster`, `_skip_blind`, `_pick_booster` and `shop._fire_joker_hook` all go through
`base.fire_hook` (so `state['destroyed']`, `pending_jokers` / `pending_cards` /
`pending_destroy` are drained everywhere — verified: Gros Michel with 3 Oops is removed at
round end and `gros_michel_extinct` is set), `sell_joker` uses `sell_hooks` / `remove_joker`,
`buy_item` / `debug_add_joker` use `init_joker` + `sync_probabilities`, and `can_afford` /
planet pricing / free rerolls read `passive_modifiers`. Remaining:

1. `JokerInstance.clone()`: copy `sort_id` (readers fall back to `getattr(j, "sort_id", 0)`,
   so a clone without it still works, just with a flat board order for Amber Acorn /
   Crimson Heart / Invisible / Ankh draws).
2. Keep `round_picks` in the front-key format — `round_cards.from_round_picks` converts it
   for the hooks; `_round_end_resets` already matches `keys.py` ROUND_END_SEQUENCE.
3. `game.state_signature()` (the one probe hook still missing) — nice-to-have.

## Deferred to W5 (with reasons)

* `j_four_fingers` / `j_shortcut` / `j_smeared` / `j_pareidolia`-for-hand-type: `hand_eval.py`
  has no flag support; `_play_hand` calls `evaluate_hand(selected)` before the jokers can set
  anything. Needs `evaluate_hand(cards, four_fingers=, shortcut=, smeared=)` and the flags
  computed from owned jokers before evaluation.
* Boss debuffs are applied to `full_deck` once at blind start; the real game re-evaluates
  `debuff_card` on every draw/play, so hand-edited cards (the reachability probe, env
  injection) bypass Goad/Club/Window/Head/Plant. Chicot's probe fails for that reason only.
* Crimson Heart rolls per play, the game per draw-to-hand; Perkeo's copy needs a slot
  (negative consumables are not modelled — consumables are bare keys); Gift Card cannot add
  sell value to consumables (same reason); Luchador's boss-disable on sell (no sell-during-
  blind action) and Diet Cola's tag depend on W2 plumbing; The Wheel / flipped cards are
  hidden-information bosses (`wheel` key unused).
* Joker-on-joker effects (Baseball) apply in one step instead of per-joker interleaving.
* The brief asked for Glass "one roll per trigger"; the Lua rolls once per scoring Glass
  card after all retriggers (state_events.lua:951-963). Implemented per source;
  `test_glass_rolls_once_per_scoring_card_not_per_trigger` pins it.
