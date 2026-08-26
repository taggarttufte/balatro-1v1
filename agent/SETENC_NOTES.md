# SETENC_NOTES — Phase 4 W1: set-based encoder + set action features + `Sample` v2

**Agent W1, 2026-08-22.** Deliverable: `agent/mcts/{encoder_set.py, model_set.py,
action_features_set.py}`, `agent/train/sample.py`, the `--encoder set` wiring, tests, and
this file. `engine/**`, `rng/**`, `tournament/**`, `eval/**` are read-only for me.

---

## 0. Contract (published first — W2 codes against this)

*Published 2026-08-22 ~08:30. Changes after publication are appended as dated notes at the
end of this section; the field list above the line is stable.*

### 0.1 `--encoder set` selection

`mcts.encoder.get_encoder(name)` gains `"set"` alongside `"v7"` (447) and `"mlb"` (453).

```python
from mcts.encoder import get_encoder
enc = get_encoder("set")          # -> mcts.encoder_set.SetEncoder
enc.name          # "set"
enc.dim           # None  <-- there is no OBS_DIM for a set encoder. Do not read `.dim`
                  #     without checking `enc.is_set` first.
enc.is_set        # True on SetEncoder, False on V7Encoder / MLBEncoder
enc.caps          # ItemCaps(hand=16, jokers=12, consumables=6, shelf=8, packs=8)
enc.describe()    # dict -> goes into the checkpoint; `SetEncoder.from_description` rebuilds
enc(game)         # -> Obs, a dict[str, np.ndarray] (§0.2)
```

Everything that used to branch on `encoder.dim` now branches on `encoder.is_set`.

### 0.2 Observation — `Obs = dict[str, np.ndarray]`

Five **item sets** (padded to fixed caps + a mask) and one **scalar** vector. The caps are
transport only: the net is set-invariant, padded rows are masked out and cannot influence
any output (`tests/test_set_encoder.py::test_garbage_in_the_padded_rows_changes_nothing`).

| key | shape | dtype | meaning |
|---|---|---|---|
| `hand_cat` | (16, 5) | int16 | per card: rank, suit, enhancement, edition, seal (index 0 = unknown/pad) |
| `hand_num` | (16, 9) | float32 | per card numerics (§1.2) |
| `hand_mask` | (16,) | float32 | 1.0 for a real card |
| `joker_key` | (12,) | int16 | index into the shared `KEY_VOCAB` |
| `joker_cat` | (12, 2) | int16 | edition, rarity |
| `joker_num` | (12, 16) | float32 | position, scaling state, stickers (§1.3) |
| `joker_mask` | (12,) | float32 | |
| `cons_key` | (6,) | int16 | `KEY_VOCAB` index |
| `cons_num` | (6, 8) | float32 | |
| `cons_mask` | (6,) | float32 | |
| `shelf_key` | (8,) | int16 | `KEY_VOCAB` index |
| `shelf_cat` | (8, 2) | int16 | kind, edition |
| `shelf_card` | (8, 5) | int16 | the card block, for `kind="card"` shelf items (else all 0) |
| `shelf_num` | (8, 12) | float32 | |
| `shelf_mask` | (8,) | float32 | |
| `pack_key` | (8,) | int16 | `KEY_VOCAB` index |
| `pack_cat` | (8, 2) | int16 | set, edition |
| `pack_card` | (8, 5) | int16 | the card block, for Standard-pack playing cards |
| `pack_num` | (8, 8) | float32 | |
| `pack_mask` | (8,) | float32 | |
| `scalars` | (`SCALAR_DIM`,) | float32 | everything that is not an item (§1.6); `SCALAR_DIM = 196` |

`SetEncoder.batch(list_of_obs) -> dict[str, np.ndarray]` stacks B observations into
`(B, …)` arrays. On the torch side `policy_set.stack_obs(list_of_obs, device)` and
`policy_set.pad_acts(list_of_acts, counts, device)` produce the batched-and-padded tensors
that `SetPolicyValueNet.encode_state(obs) -> StateEmbedding(trunk, hand, target)` and
`SetPolicyValueNet.action_logits(state, acts) -> (B, N)` take; `net(obs, acts)` is the
one-shot `(logits, values)` form. `SetPolicyValueNet.value(state_or_trunk) -> (B,)`.

### 0.3 Action features — `Acts = dict[str, np.ndarray]`

`action_features_set.featurize_actions_set(game, actions, caps) -> Acts`, one row per action:

| key | shape | dtype | meaning |
|---|---|---|---|
| `act_type` | (N,) | int16 | index into `ACTION_TYPES` (13) + 1 unknown slot |
| `act_sel` | (N, 16) | float32 | **row-normalised** mask over HAND slots — the cards the action selects (play / discard / consumable targets) |
| `act_tgt` | (N, 34) | float32 | **row-normalised** mask over the concatenated non-card item slots `[jokers 12 | consumables 6 | shelf 8 | packs 8]` — the item(s) the action targets (buy / sell / use / pick) |
| `act_num` | (N, 20) | float32 | scalars incl. count, cost, affordability, would-be hand type one-hot, hand level (§2.2) |

An index past a cap is dropped from the mask and raises `act_num[4] = 1.0` (overflow), so
two overflowing actions are still distinguishable by their other features. Rows are unique
per legal-action list (pinned by a test on real games, as for the 56-dim featurizer).

### 0.4 `PolicyValueFn` — UNCHANGED

```python
fn(game) -> (dict[ActionKey, float], float)
fn.evaluate_many(games) -> [(priors, value), ...]        # order preserved, {} / 0.0 for no-action
```

Nothing about this interface changes. `mcts.policy.make_policy(net, device, encoder,
batched=True)` returns `BatchedNNPolicy` for a flat encoder and `BatchedSetNNPolicy` for a
set encoder; `mcts.player.load_policy(checkpoint, device, batched, encoder)` picks the right
one off the checkpoint's recorded encoder with no caller change. `MCTSPlayer`,
`BatchedMCTSPlayerGroup`, `MCTS`, `BatchedSearch`, `TreeCache` are untouched and encoder-blind.

### 0.5 `Sample` v2 — `agent/train/sample.py`

```python
@dataclass
class Sample:                 # train.sample.Sample — "v2"
    obs:           np.ndarray | dict[str, np.ndarray]   # flat (D,) OR the Obs dict
    actions:       np.ndarray | dict[str, np.ndarray]   # flat (k, 56) OR the Acts dict, k rows
    target_policy: np.ndarray                           # (k,) float32, sums to 1
    z:             float                                # value target in [0, 1]
    meta:          dict                                 # see below
    version:       int = 2
```

`meta` always carries `seed`, `ante`, `state` (the `State` name), `n_legal` (the FULL legal
count before subsampling), `k` (rows kept), `encoder` (`"v7"`/`"mlb"`/`"set"`),
`visited` (how many of the k rows had visit_count > 0). W2 may add keys; nothing reads
unknown keys.

The old 4-field `Sample` (`obs`, `action_features`, `target_policy`, `z`) still lives in
`train/trajectory.py` as `SampleV1` (aliased `Sample` there, unchanged) and the v1 training
path is byte-identical. `ReplayBuffer` holds either and serialises both.

### 0.6 Building a sample — `SampleBuilder`

```python
from train.sample import SampleBuilder, make_sample_v2

builder = SampleBuilder(encoder,          # ObsEncoder | SetEncoder
                        k_unvisited=8,    # default; 0 keeps only the visited actions
                        subsample=True,   # False -> every legal action (the v1 shape)
                        rng=np.random.default_rng(...))

s = builder(game, legal, legal_keys, visits, encoder, z)     # -> Sample v2
```

The call signature is **W2's `SampleCollector(sample_fn=...)` protocol verbatim**
(`train/selfplay.py:196` calls `sample_fn(game, decision.legal, decision.legal_keys,
decision.visits, self.encoder, 0.0)`), so wiring is one constructor argument:
`SampleCollector(idx, encoder, sample_fn=SampleBuilder(encoder, rng=rng))`.
`ColdTrainer.sample_builder` already holds one, which is what
`MLBTrainer` reads (`train/selfplay.py:870`).

* `visits`: `dict[ActionKey, int]` exactly as `MCTS.run_gumbel` returns; `{}` on a forced
  (single-action) state.
* Selection: **every** action with `visits[k] > 0` is kept, plus `min(k_unvisited, remaining)`
  actions drawn uniformly without replacement from the zero-visit remainder, via `rng`.
  Kept rows are in `legal` order (ascending index), not draw order.
* `target_policy` is `counts / counts.sum()` over the kept rows. Since every visited action
  is kept, `counts.sum()` is the FULL visit total and the renormalisation is EXACT, not
  approximate: the kept-row distribution equals the full distribution restricted to its
  support. If nothing was visited, the target is uniform over the kept rows.
* Rows are built ONCE per sample — only the kept actions are featurized, so the cost is
  O(k), not O(n_legal).
* `make_sample_v2(game, legal, legal_keys, visits, encoder, z, *, k_unvisited=8, rng=None,
  subsample=True, meta=None)` is the one-shot form.

> **Amendment 2026-08-22 (published ~09:40, before any consumer existed).** The pre-publication
> draft of this section proposed `make_sample(game, visits, value, encoder=..., ...)`. W2 had
> already landed `train/selfplay.py::make_sample` with the argument order above and a pluggable
> `SampleCollector.sample_fn`, so W1 conformed to W2's seam rather than the other way round:
> the builder is callable with W2's exact signature, and the module-level one-shot helper is
> named `make_sample_v2` so the two names do not collide in `train/__init__.py`.

### 0.7 Training

`train.trainer.Trainer.step(batch)` accepts a mixed list of v1 and v2 samples and dispatches
per sample; the loss is unchanged
(`-Σ target·log_softmax(logits)` + `(v - z)²`, mean over the batch). For a v2 sample the
softmax is over the **kept rows only** — which is the intended objective change and the one
line of this workstream that is not mechanical (§4.3).

### 0.8 Checkpoints

`ColdTrainer.state_dict()` gains `"encoder_caps"` (the `ItemCaps` as a dict, `None` for the
flat encoders) and `"net_kind"` (`"flat"` | `"set"`). `--resume` refuses a checkpoint whose
`encoder`, `encoder_caps` or `net_kind` differs. `SetPolicyValueNet.describe()` /
`from_description()` mirror `PolicyValueNet`'s (and carry `"kind": "set"`), so
`load_checkpoint` → net is one call and `mcts.player.load_policy(path)` needs no argument to
pick the right net class. Round-trip is bit-exact on CPU for the set encoder too.

`CHECKPOINT_VERSION` is **2**; `load_checkpoint` accepts `{1, 2}` so the Phase 3 runs under
`agent/runs/` (including the overnight shakedown) still load — a version-1 file is
necessarily a flat-encoder checkpoint and the two new keys read as absent.

`TrainConfig` gains `subsample` (default True), `k_unvisited` (8), `set_res_blocks` (2,
the set net's trunk depth — the flat net's `n_res_blocks=4` would push the set net to ~2.9M
params for no reason) and `value_activation` (`"sigmoid"`). All four are part of
`_check_config`'s resume guard.

> **Amendment 2026-08-22 (follow-up 2).** Both nets' `describe()` now carries
> `value_activation`, and both `from_description` implementations default it to `"linear"`
> when the key is absent, so a checkpoint written before the bounded head is rebuilt with
> the semantics it was TRAINED under rather than silently reinterpreted through a sigmoid.
> `ColdTrainer.from_checkpoint` backfills the config field the same way.
> **`CHECKPOINT_VERSION` stays 2** — the state_dict shape is unchanged.

---

## 1. The observation

### 1.1 Why sets

`env_v7._encode_obs` writes 8 hand slots × 30 and 5 joker slots × 10 at fixed offsets. Both
bounds are exceeded in real play (`hand_size` grows with Juggler / Turtle Bean / Ouija /
Ectoplasm / vouchers; `joker_slots` grows with Negative jokers), so a 9th card in hand is a
legal `play` target that the value/policy net cannot see — logged as a needs-engine-change
in AGENT_NOTES §8 and deferred by Phase 3. Raising the caps would have fixed the blindness
and kept the pathology: slot *k* has its own weights, so "Blueprint in slot 3" and
"Blueprint in slot 4" are learned separately from scratch.

A shared per-item encoder learns one representation of a card / a joker / a shelf item and
reuses it in every position, and the caps become transport padding rather than model
structure. Position is kept as an explicit *feature* on each item (jokers score
left-to-right, so joker order is real information) — the network is invariant to the ORDER
OF THE ROWS, not blind to where a joker sits.

### 1.2 Card items (`hand_*`)

`hand_cat` columns: `rank` (0 = unknown, 1..13 = ranks 2..14), `suit` (0 = unknown,
1..4 = Spades/Hearts/Clubs/Diamonds), `enhancement` (0 = unknown, 1..9 = `ENHANCEMENTS`),
`edition` (0 = unknown, 1..5 = `EDITIONS`), `seal` (0 = unknown, 1..5 = `SEALS`).
A face-down card (The House / Wheel / Mark / Fish) is hidden information: its mask is 1 and
every categorical is 0 (unknown), exactly as `env_v7` zeroes it.

`hand_num` columns: `(rank-2)/12`, `base_chips/11`, `bonus_chips/50`, `debuffed`,
`face_down`, `suit_match_count/7`, `rank_match_count/7`, `straight_connectivity/7`,
`slot/cap`. The three "V7 hand-context" features are computed exactly as `env_v7` does
(over the non-face-down cards only).

### 1.3 Joker items (`joker_*`)

`joker_key` indexes `KEY_VOCAB` (§1.7) — a real 150-way embedding, not `env_v7`'s
`JOKER_IDX[key] / N_JOKERS` scalar, which asked the net to read joker identity off a single
float in alphabetical order. `joker_cat`: edition, rarity.

`joker_num` (16): `slot/cap`, `(slot+1)/n_jokers` (relative position — what actually decides
Blueprint/Brainstorm targets and scoring order), `log1p(mult)/10`, `log1p(chips)/10`,
`min(xmult,10)/10`, `min(mult_mult,5)/5`, `log1p(sell_value)/5`, `destroyed`,
`log1p(rounds)/5`, `log1p(count)/5`, `log1p(streak)/5`, `log1p(pending_money)/5`,
`log1p(sum of every other numeric state value)/10` (catch-all so a scaling joker whose state
key is not in the list is still not flat), `eternal`, `perishable`, `rental`.

### 1.4 Consumables / shelf / packs

`cons_num` (8): `slot/cap`, is_planet, is_tarot, is_spectral, `cost/10`,
`planet hand-type index/11`, `planet level of that type/10`, `n_card_targets/2`.

`shelf_num` (12): `slot/cap`, `not sold`, `price/20`, `discounted_price/20`, affordable,
`rarity/4`, joker-slot-free, consumable-slot-free, eternal, perishable, rental, couponed.
`shelf_cat`: kind (`joker|planet|tarot|spectral|card|voucher|booster`), edition. A shelf
playing card (Magic Trick / Illusion) additionally fills `shelf_card` with the SAME 5-column
card block the hand uses, and the model runs it through the SAME card encoder.

`pack_num` (8): `slot/cap`, is_joker, is_consumable, is_playing_card, grantable
(`game._can_grant_choice`), eternal, perishable, rental. `pack_cat`: set, edition;
`pack_card` as above for Standard packs.

### 1.5 What became scalars instead of a set, and why

* **Draw pile** — up to 52+ cards, all unordered, and what matters is the *composition*.
  Kept as `env_v7`'s 16 aggregate features (suit fractions, face/ace/number fractions, size,
  edition/enhancement/seal fractions). A 52-item set would triple the per-leaf encode cost
  for information the aggregates already carry.
* **Vouchers owned** — a 32-wide multi-hot over the full `VOUCHER_KEYS` (`env_v7` used the
  first 27; the fork has 32).
* **Tags held** — a 24-wide multi-hot over `TAG_KEYS` + a count. Tag order matters only for
  the two break-after-first triggers, which is far less than the cost of a sixth set.
* **Blind** — a single item, so a one-hot block is a set of size one: kind (3), boss (28),
  is_boss / is_showdown / is_pvp / disabled, `log1p(target)/log1p(1e5)`, progress,
  `money_reward/8`.

### 1.6 `scalars` (196)

Layout is defined once in `encoder_set.SCALAR_LAYOUT` (a list of `(name, width)`); the
encoder writes by generated offset and `test_scalar_layout_matches_dim` asserts the widths
sum to `SCALAR_DIM`, so the documentation cannot drift from the code. Blocks:

`state` 7 · `ante_blind` 8 · `blind_kind` 3 · `boss` 28 · `economy` 10 · `capacity` 9 ·
`round` 8 · `planet_levels` 12 · `hand_types_played` 12 · `vouchers` 32 · `tags` 24 +
`tag_count` 1 · `deck_composition` 16 · `deck_id` 15 · `stake` 1 · `mlb` 10 = **196**.

### 1.7 `KEY_VOCAB`

One shared vocabulary over every game key an item can carry:
`["<pad>", "<unk>"] + JOKER_KEYS(150) + TAROT_KEYS(22) + PLANET_KEYS(12) + SPECTRAL_KEYS(18)
+ VOUCHER_KEYS(32) + BOOSTER_TYPE_KEYS(15)` = **251** entries, built from
`balatro_sim.game_keys` at import (nothing hardcoded — same discipline as the Phase 3
encoder, so a pools change moves the vocabulary automatically; the test asserts the total
against the live tables rather than a literal). Playing-card items do not consume a
vocabulary slot: they use the 5-column card block instead.

### 1.8 Coverage of `env_v7._encode_obs` — every feature accounted for

| `env_v7` block | where it went |
|---|---|
| game scalars (14) | `scalars` — `ante/8`, `blind_idx/2`, `is_boss`, progress, log target, hands, discards, dollars, joker fill, state one-hot (widened 5 → 7: `BOOSTER_OPEN` and `PVP_WAIT` were both folded into "game over" by V7) |
| hand cards 8 × 30 | `hand_*`, cap 16, every one of the 30 features present (26 V6 + the 4 V7 context features) |
| joker slots 5 × 10 | `joker_*`, cap 12; the identity scalar `JOKER_IDX/N` is **replaced** by a 150-way embedding (strict upgrade), everything else present + 8 more |
| shop items 7 × 6 | `shelf_*`, cap 8, all 6 present + 6 more |
| planet levels (12) | `scalars.planet_levels` |
| consumable hand 2 × 8 | `cons_*`, cap 6. Three of V7's 8 were **state duplicates** (`dollars/50`, `hands_left/base`, `is SELECTING_HAND` — copied into every consumable slot); they live once in `scalars`. |
| shop context: reroll (2) | `scalars.economy` |
| shop context: vouchers (27) | `scalars.vouchers`, widened to all 32 |
| shop context: boss one-hot (28) | `scalars.boss` |
| shop context: deck comp (8 + 8) | `scalars.deck_composition` |
| MLB block (`encoder.py`, 6) | `scalars.mlb`, widened to 10 (adds opponent score log, lead, opponent-exhausted, opponent progress — the `env_mp.MP_OBS_FEATURES` list) |

**Dropped on purpose: nothing.** **Deduplicated (3):** the three consumable-slot copies
above. Every other `env_v7` feature is present, and the set encoder additionally carries what
V7 could not see: a real joker-key embedding instead of an alphabetical scalar, hand slots
9-16 and joker slots 6-12 (the AGENT_NOTES §8 blindness), joker stickers and per-joker
`xmult` / `rounds` / `count` / `streak` / `pending_money`, shelf `discounted_price` /
stickers / `couponed` / rarity, the open pack's contents at all (V7 had no pack features),
tags held, the deck identity, the stake, `hand_size` and the three slot capacities,
`skips`, `unused_discards`, the hand types already played this round, and the four extra MLB
features from `env_mp`.

---

## 2. Action features

### 2.1 Pointer over items, not one-hots over slot indices

The 56-dim row spent 12 + 4 + 8 + 8 + 8 = 40 of its dims on slot one-hots, so "play the
Ace of Spades" was encoded as "play slot 3" and the net had to bind slot 3 to the card
through the trunk. Here an action carries a **row-normalised mask over the item slots** and
the model pools the item embeddings the observation already computed:

```
act_emb = type_emb(act_type)
        ⊕ act_sel @ hand_item_emb          # mean of the selected cards' embeddings
        ⊕ act_tgt @ target_item_emb        # the bought/sold/used/picked item
        ⊕ num_proj(act_num)
```

so "play the Ace of Spades" and "play the Ace of Spades from a different slot" are the same
vector, and "buy Blueprint" is the Blueprint embedding regardless of shelf position.

### 2.2 `act_num` (20)

`0` `n_sel/5` · `1` `n_tgt/5` · `2` `cost/20`, signed (a `sell_joker` carries
`-sell_value/20`) · `3` affordable · `4` overflow · `5..16` would-be hand-type one-hot (12) ·
`17` planet level of that hand type / 10 · `18` `log1p(base chips of the selected cards) /
log1p(300)` · `19` free (couponed item / free reroll).

### 2.3 All 13 action types

`play_blind`, `skip_blind`, `reroll_boss`, `play`, `discard`, `use_consumable`, `buy`,
`sell_joker`, `reroll`, `leave_shop`, `pick_booster`, `skip_booster`, `advance` — the same
vocabulary as `action_features.ACTION_TYPES`, imported from it rather than re-listed.
`tests/test_set_encoder.py::test_every_action_type_embeds` walks real games on both rulesets
until all 13 have been seen and asserts each produces a finite, non-degenerate row.

### 2.4 The would-be hand type, and why it is not `hand_eval.evaluate_hand`

A `SELECTING_HAND` leaf has ~436 legal actions, ~218 plays and ~218 discards.
`hand_eval.evaluate_hand` costs **4.8 µs** on this box, so calling it per action is **~2.1 ms
per leaf** against the ~0.5 ms the whole per-leaf CPU path costs today — a 4-5× throughput
regression for one feature.

`action_features_set.fast_hand_types` does the same 436 subsets in **0.119 ms** by
vectorising the engine's own decision tree: rank counts and per-suit flush counts are two
`(N, n_hand) @ (n_hand, ·)` matmuls off the selection mask, and `get_straight`'s j = 1..14
walk (`misc_functions.lua:548-590`) runs as 14 vectorised steps over the batch. With it on,
`featurize_actions_set` costs 0.453 ms per 436-action leaf against 0.238 ms with it off, and
0.299 ms for the 56-dim featurizer.

It is a SECOND implementation of engine logic, which is exactly the mistake AGENT_NOTES §2.1
documents for the copied encoder, so it carries the same mitigation:
`test_fast_hand_type_matches_hand_eval` asserts equality with `hand_eval.evaluate_hand` over
**every** subset of size 1-5 of 60 random hands (Stone / Wild / debuffed cards included)
across all four flag combinations (Four Fingers × Shortcut × Smeared) — 209 088 comparisons,
0 mismatches — and `test_fast_hand_type_matches_hand_eval_on_real_hands` repeats it on the
200 fixture hands with their live flags. If the engine's hand evaluation ever changes, those
fail rather than the features silently drifting.

Set `hand_type_features=False` on the featurizer to skip it entirely (the block is then
zeros); the paired comparison below ran with it ON.

---

## 3. The model — `SetPolicyValueNet`

```
per-item encoders (shared weights within a type)
   card block   (rank,suit,enh,ed,seal embeddings ⊕ 9 numerics) -> D=64      [hand, shelf cards, pack cards]
   joker block  (key emb 48 ⊕ ed ⊕ rarity ⊕ 16 numerics)        -> D=64
   consumable   (key emb ⊕ 8 numerics)                          -> D=64
   shelf        (key emb ⊕ kind ⊕ ed ⊕ card block ⊕ 12 numerics)-> D=64
   pack         (key emb ⊕ set ⊕ ed ⊕ card block ⊕ 8 numerics)  -> D=64
        |
   ⊕ per-set type embedding, concatenated into ONE 50-slot item sequence
        |
   1 masked multi-head self-attention block (4 heads, D=64) + FFN, pre-norm
        |
   per-set masked mean ⊕ max pooling  ->  5 sets × 2 × 64 = 640
        |
   ⊕ scalar_proj(scalars 193 -> 256)
        |
   trunk: Linear(896 -> 512) -> 2 × ResidualBlock(512)          -> (B, 512)
        |
   value head Linear(512 -> 1) -> sigmoid      [`value_activation`, follow-up 2]
   policy head Linear(512 + 128 -> 128) -> ReLU -> Linear(128 -> 1)   [pointer, per action]
```

**Why one attention block rather than pooling alone.** Mean+max pooling is invariant and
cheap, but Balatro's state is almost entirely *interaction*: a Flush-suit card is worth
something only next to a suit joker, Blueprint is worth whatever is to its right, and a shelf
joker's value is a function of the board it would join. Pooling forces every such
interaction through the trunk after the item identities have already been averaged away. One
masked attention layer over the union of all 50 item slots lets a joker attend to the cards
in hand and to its neighbours before pooling, costs 50×50 attention per state (a rounding
error next to the 512-wide trunk) and adds 33k parameters. It remains exactly
permutation-equivariant, so per-set masked pooling on top is still permutation-invariant,
which is what the tests assert.

**Parameters: 1 793 536** (`SetPolicyValueNet().n_params`, defaults `d_item=64, n_heads=4,
key_emb=48, card_emb=9, aux_emb=7, hidden=512, n_res_blocks=2`), against `PolicyValueNet`'s
**2 411 266** — inside the ≲3M budget while carrying a 251-key embedding table the flat net
did not have.

**Everything categorical lives in three tables** (follow-up 1, §8): the card block
(rank/suit/enhancement/edition/seal behind offsets), one "aux" table
(edition/rarity/shelf-kind/pack-set behind offsets), and the game-key table — and every game
key in a state is gathered ONCE and split, rather than four times. `_OffsetEmbedding` is
exactly as expressive as per-field tables (each field owns its own rows;
`test_offset_embedding_matches_separate_tables` pins the two equal for shared weights) and
turns ~25 embedding kernels into 7.

**Numerical note.** Permutation invariance is exact in exact arithmetic but not in float32:
permuting rows changes the summation order inside the attention weighting and the pooling.
Measured worst case over all 200 fixture states, one random within-set permutation each:
**1.4e-6** on the value, **1.9e-8** on the logits. The tests assert `< 1e-4` — the same
reasoning BATCH_NOTES §3 gives for "batched == single-leaf".

**Transfer packing.** A batched observation is 21 arrays and an action block is 4 more, so
the naive `torch.from_numpy(x).to(device)` per key is 25 small host→device copies per forward
pass. Measured on this box: **20 such copies cost 5.35 ms; one concatenated copy of the same
bytes costs 0.027 ms** — and 5.35 ms was more than the forward pass itself (5.8 ms at B=16).
`policy_set._to_device` therefore packs key-major into one float32 and one int16 buffer for a
non-CPU device, transfers once, and splits into contiguous views: pad+transfer for a 16-leaf
batch went **5.66 ms → 1.62 ms**, and search throughput on CUDA **69 → 118 sims/s**. The CPU
path stays per-key (`from_numpy` is a free view there); `test_packed_and_unpacked_transfers_agree`
and `test_cuda_matches_cpu` pin the two equal.

---

## 4. `Sample` v2 and subsampling

### 4.1 Size

`benchmarks/bench_sample_size.py`, 200 states from the overnight run's own episode seeds
(§6.1), MLB ruleset, `k_unvisited=8` with 8 simulated visited actions:

| shape | mean B | median B | max B | vs v1 |
|---|---|---|---|---|
| v1, flat, ALL legal actions (Phase 3) | **45 810** | 11 250 | 483 552 | 1.0× |
| v2, flat, subsampled | 4 050 | 5 436 | 5 436 | **11.3×** |
| v2, **set**, subsampled | **6 497** | 8 236 | 8 236 | **7.1×** |
| v2, set, no subsampling | 58 881 | 15 529 | 607 978 | 0.8× |

Per game state, which is where the real shape shows:

| state | v1 | v2 flat | v2 set | v1 / v2-set |
|---|---|---|---|---|
| `SELECTING_HAND` (436 actions) | 94 628 | 5 436 | 8 236 | **11.5×** |
| `SHOP` | 4 400 | 3 396 | 5 677 | 0.8× |
| `BLIND_SELECT` | 2 216 | 2 216 | 4 197 | 0.5× |
| `BOOSTER_OPEN` | 2 700 | 2 700 | 4 804 | 0.6× |
| `ROUND_EVAL` | 2 016 | 2 016 | 3 946 | 0.5× |

And measured end to end on the buffer of a real 12-episode MLB self-play run (identical
config, only the encoder / `subsample` flag changed) — the number that actually decides the
checkpoint size:

| run | samples | mean B | median B |
|---|---|---|---|
| v1 flat, Phase 3 shape | 1 451 | **28 597** | 10 476 |
| v2 flat, subsampled | 1 124 | 3 845 | 4 548 |
| v2 **set**, subsampled | 1 342 | **6 563** | 7 092 |

**Honest headline: 11.5× at the leaves that caused the problem, 7.1× on the mean over all
states, not the brief's "~20×".** The 20× estimate came from the flat encoder's 97 KB
`SELECTING_HAND` sample, and subsampling alone does deliver ~17× there (94.6 KB → 5.4 KB).
The set encoder gives some of it back: its observation is a fixed **~5.2 KB dict of 21
arrays** regardless of state, against the flat encoder's 1.8 KB vector, so every small-action
state (`BLIND_SELECT`, `ROUND_EVAL`, most `SHOP`) is now *bigger* than it was. That is the
price of the encoding, it is bounded and known, and it is paid once per sample rather than
once per action.

What it buys in practice, at 6.5 KB/sample:

* a 200 000-sample buffer is **1.30 GB** instead of **9.16 GB**;
* the Phase 3 run's `latest.pt` was **137.8 MB** with 1 472 buffered samples; a set run's
  `latest.pt` with the same buffer cap is **~24 MB**, and a weights-only checkpoint is
  **7.2 MB** against the flat net's 28.9 MB.

### 4.2 What subsampling costs

Every action the search visited is kept, so the policy target's support is complete and the
renormalisation over the kept rows is exact. What is lost is the *negative* signal from
`n_legal - k` unvisited actions that the old target pushed towards 0. With k_unvisited = 8
the net still sees ~8 sampled negatives per state, drawn afresh every time the state is
visited, so over the buffer's lifetime the negatives are covered stochastically rather than
exhaustively — the standard treatment for a large action space (sampled softmax).

The measurable consequence: **the policy loss is no longer comparable across runs with
different `k`**, because `log(k)` replaces `log(n_legal)` as the uniform baseline (~2.6 at
k = 13 against ~6.1 at 436 actions). A Phase 3 run sat at policy loss ~3.1; a subsampled run
starts near 2.0. That is not the policy improving. Do not read the two curves against each
other; the paired eval in §7 is the comparison that means something.

### 4.3 The one non-mechanical change

The trainer's softmax is over the kept rows only, so the gradient is a sampled-softmax
approximation of the full one. That is deliberate and is the trade the decision in the Phase
4 brief §0.2 buys the 20×. It is a *biased* estimator of the full softmax gradient (no
importance correction — the standard AlphaZero-with-subsampling practice), and the bias is
towards under-penalising unvisited actions. If that ever shows up as the policy putting mass
on obviously bad actions, the fix is a log-uniform importance correction on the sampled
negatives, not a bigger k.

---

## 5. Wiring

| file | change |
|---|---|
| `mcts/encoder_set.py` | **new** |
| `mcts/action_features_set.py` | **new** |
| `mcts/model_set.py` | **new** |
| `mcts/policy_set.py` | **new** — `SetNNPolicy` (serial) / `BatchedSetNNPolicy` |
| `mcts/encoder.py` | `+ is_set` on the flat encoders, `ENCODERS["set"]` (lazy import), `get_encoder(name, **kwargs)`, `+ is_set_encoder()` |
| `mcts/policy.py` | `+ make_policy(net, device, encoder, batched)` factory. Existing classes untouched. |
| `mcts/player.py` | `+ build_net(encoder)`; `load_policy` routes through `make_policy`, rebuilds a `SetPolicyValueNet` when the checkpoint says `net_kind="set"` and restores its caps; `make_player` gains `encoder=`. `MCTSPlayer` / `BatchedMCTSPlayerGroup` untouched. |
| `mcts/model.py` | follow-up 2 only: `value_activation` (`sigmoid` default), `apply_value_activation`, `describe`/`from_description`. No state_dict shape change. |
| `search.py`, `batched.py`, `reuse.py`, `node.py`, `outcome.py`, `action.py`, `action_features.py` | **untouched** |
| `train/sample.py` | **new** — `Sample` v2, `SampleBuilder`, `make_sample_v2`, `sample_nbytes`, `subsample_indices` |
| `train/trajectory.py` | `SampleV1` alias; `ReplayBuffer` serialises v1 and v2 |
| `train/trainer.py` | per-sample dispatch (v1 / v2-flat / v2-set); v1 path byte-identical |
| `train/agent.py` | `sample_builder=None` (default) → the Phase 3 path byte-identical; set → v2 samples |
| `train/loop.py` | `TrainConfig.encoder` accepts `"set"`; `+ k_unvisited`, `+ subsample`; net/policy/builder selection |
| `train/checkpoint.py` | `CHECKPOINT_VERSION` 1 → **2**; `net_kind` + `encoder_caps` recorded and checked |
| `scripts/train_cold.py` | `--encoder set`, `--k-unvisited`, `--no-subsample`, `--set-res-blocks`, `--value-activation`; **refuses `--ruleset mlb`** (follow-up 3) |
| `scripts/eval_checkpoint.py` | **new** — an `eval`-schema JSON report for a checkpoint player (`eval` is frozen and its `checkpoint:` spec raises `NotImplementedError` by design) |
| `benchmarks/bench_sample_size.py` | **new** — the §4.1 tables |
| `benchmarks/bench_set_vs_flat.py` | **new** — the §6.2 throughput table (one command, prints the `--device` recommendation) |
| `tests/_states.py` | **new** — the 200-state fixture (§6.1) |
| `tests/test_set_encoder.py` (30), `tests/test_sample_v2.py` (23), `tests/test_followups.py` (18) | **new** |
| `tests/test_checkpoint.py` | the one Phase 3 test that pinned `version == 1` now pins `CHECKPOINT_VERSION` and checks the two new keys |
| **W2's files** (`train/selfplay.py`, `population.py`, `scripts/train_mlb.py`, `tournament/**`) | **untouched by W1.** W2 had already put `"set"` in `train_mlb.py --encoder` and wired `sample_fn = ColdTrainer.sample_builder`, so `train_mlb --encoder set --objective external` works with no further edit. |

**The serial flat path is byte-identical.** `tests/test_batched.py::
test_serial_search_matches_the_pre_w3_implementation` and the whole Phase 3 suite still pass
unchanged; nothing in `search.py` / `batched.py` / `model.py` was edited at all.

---

## 6. Results

### 6.1 The 200 states

`tests/_states.py`. The seeds are the first 40 `kind:"episode"` seeds of
`runs/overnight_2026-08-22/overnight_2026-08-22.jsonl` — read from the file when it is there
and from a copy in the module otherwise, because `agent/runs/` is gitignored and the tests
must run on a clean checkout. Each game is walked and a state is cloned every 7th decision
until 200 are collected.

The trajectories are NOT bit-identical to the logged ones: the run's Gumbel noise stream is
not recoverable from the JSONL. So this is "200 states from the overnight run's seeds", not
"the 200 logged states" — the `latest.pt` buffer holds encoded arrays, not games, so replaying
the exact logged states was not available either. Two drivers:

| driver | used by | `SELECTING_HAND` | `SHOP` | `BLIND_SELECT` | `ROUND_EVAL` | `BOOSTER_OPEN` |
|---|---|---|---|---|---|---|
| seeded uniform legal action | the tests (fast, no torch) | 93 | 57 | 33 | 12 | 5 |
| `ckpt_002072.pt` prior-argmax | `bench_sample_size.py --checkpoint` | 33 | 134 | 18 | 6 | 9 |

The second distribution is the overnight run's own degeneracy showing through — it skips
blinds and lives in the shop. The tests use the first because it exercises the encoder where
the action set is large.

### 6.2 Search throughput, set vs flat

`bench_search`-style: the ante-1 `SELECTING_HAND` demo state with **436 legal actions**,
200 Gumbel simulations, `leaf_batch=16`, best of 2, cold-init nets.

| | flat (`mlb`, 2.41M) | set (1.79M) |
|---|---|---|
| CPU | **449** sims/s | **259** sims/s |
| CUDA | **327** sims/s | **118** sims/s (69 before the transfer packing) |

> **These are PRE-embedding-merge numbers.** Follow-up 1 (§8.1) merged ~25 embedding
> kernels into 7, which is exactly the fix this table's CUDA gap called for, but the
> re-measurement is **PENDING — machine in use** (2026-08-22, lead's instruction). One
> command re-runs all four cells and prints the `--device` recommendation:
>
> ```
> python agent/benchmarks/bench_set_vs_flat.py
> ```
>
> **This is the number that decides `--device` for the first long run.** Both nets were
> faster on CPU than on CUDA before the merge; if that still holds afterwards, the real run
> should not assume `--device cuda`.

At a fixed batch the two are nearly equal — `evaluate_many` on 16 leaves × 436 actions costs
26.5 ms flat vs 28.7 ms set on CUDA, i.e. **8%**. The end-to-end gap is per-CALL overhead:
the set forward is ~60 kernel launches (21 embedding lookups, 5 item MLPs, an attention
block, an FFN, 10 pooling reductions) against the flat net's ~15, and a search issues many
small batches. **That is what follow-up 1 fixed** (§8.1: the embedding lookups are now 7,
not ~25), which is why the table above needs re-running before the `--device` call is made.

Both nets are FASTER ON CPU than on CUDA at this scale — the Phase 3 finding
(AGENT_NOTES §4.4) still holds and the set net makes it starker.

---

## 7. Gates, results and found-not-fixed

### 7.1 Gates (repo root, python 3.13.5, `-p no:cacheprovider`)

| gate | result |
|---|---|
| `pytest agent/tests` | **270 passed / 1 failed** after the §8 follow-ups — the one failure is W2/W3's replay test, see §8.4 (was 253/0 at the first hand-off; 131 at Phase 3 close; W1 adds 71 tests in total, W2 the rest) |
| `pytest engine/tests` | **1614 passed / 10 skipped / 3 xfailed / 0 failed** — unchanged |
| `pytest tests` | **1073 passed / 2 xfailed / 0 failed** — unchanged |
| `pytest tournament/tests eval/tests` | **181 passed / 0 failed** (80 at Phase 3 close; the additions are W2's and W4's) |

W1's own tests: `test_set_encoder.py` **30**, `test_sample_v2.py` **23**, `test_followups.py` **18**. Nothing under
`engine/**`, `rng/**`, `tournament/**` or `eval/**` was edited.

### 7.2 Paired 10-minute comparison

**W2's `--objective external` had landed, so the comparison used it** rather than the plain
(degenerate, free-Nemesis) `MLBOutcome` the brief allowed as a fallback:

```
python agent/scripts/train_mlb.py --minutes 10 --objective external --device cuda        --sims 40 --encoder {v7|set} --seed 0 --run-name w1_cmp_{v7|set}
```

Then each run's `latest.pt` on the same 60 ground-truth seeds, and the frozen harness's own
paired-by-seed `--compare`:

```
python agent/scripts/eval_checkpoint.py --checkpoint agent/runs/w1_cmp_{ENC}/latest.pt        --mode sp_mlb --n-seeds 60 --sims 20 --max-antes 4 --max-steps 1500 --threads 1        --device cpu --out results/w1_cmp_{ENC}.json
python -m eval.eval_harness --compare results/w1_cmp_set.json        results/w1_cmp_v7.json --out results/w1_cmp_set_vs_v7.json
```

**Training (`agent/runs/w1_cmp_{v7,set}/`)**

| | flat `v7` | **set** |
|---|---|---|
| generations completed | 1 | 1 |
| episodes | 16 | 11 |
| samples collected | 8 693 | 4 407 |
| train steps | 271 | 137 |
| wall clock | 604 s | 1 012 s |
| ep/min | 1.59 | 0.65 |
| sims/s | 604 | 172 |
| value-target sd | 0.088 | 0.063 |
| mean ante reached | 2.63 | 1.91 |
| policy loss (end of gen 0) | 3.048 | 3.023 |
| value loss | 0.446 | 0.679 |
| `ckpt_gen0001.pt` | 28.9 MB | **21.6 MB** |

**Paired evaluation, 60 seeds, SP-MLB solo vs `own_big_blind_target(k=1)`, A = set, B = v7**

| metric | mean A (set) | mean B (v7) | A − B | 95% CI |
|---|---|---|---|---|
| furthest ante | 2.283 | 2.233 | **+0.05** | [−0.217, +0.317] |
| lives lost | 3.533 | 3.433 | +0.10 | [−0.167, +0.383] |
| final lives | 0.467 | 0.567 | −0.10 | [−0.383, +0.167] |
| final money | 5.417 | 4.500 | +0.92 | [−1.08, +3.15] |
| steps | 475.7 | 553.8 | −78.1 | [−243.6, +97.7] |

**Every CI straddles zero. No claim, and none is warranted** — as the brief allowed. Four
things make this a shakedown rather than a result, all of them reportable:

1. **Both arms got one generation.** `train_mlb` checks the deadline between episodes, so a
   10-minute budget with `--episodes-per-gen 16` buys exactly one generation and ~150-270
   optimizer steps. Neither net has learned anything a 60-seed eval could resolve; both die
   around ante 2 with 3.4-3.5 of 4 lives lost, which is the cold-start picture.
2. **The set arm got less data in more wall clock** (11 episodes / 1 012 s against 16 / 604 s)
   because its self-play is ~3.5× slower per simulation on CUDA (§6.2, §7.3 item 1). At equal
   wall clock the set encoder is currently the *worse* deal on throughput; the case for it is
   what it can represent, and that needs a real run to show.
3. **The box was shared.** W2's 30-minute `train_mlb` gate ran concurrently for most of both
   arms and W4's transfer-spread before that, so the wall-clock numbers are contended. The
   *paired eval* is unaffected in expectation (same seeds, same driver, sequential).
4. **The eval is truncated at `--max-steps 1500` and `--max-antes 4`** so that a pathological
   seed cannot run away; the same cap applies to both arms and is recorded in the reports.
   `--threads 1` matters: torch's default intra-op parallelism burned 20 873 s of user CPU in
   30 minutes of wall clock on this box (~12 threads spinning over a 2M-param net evaluated
   one leaf at a time) and made an unbounded 126-seed eval take over 40 minutes.

Reports: `results/w1_cmp_v7.json`, `results/w1_cmp_set.json`,
`results/w1_cmp_set_vs_v7.json`.

### 7.3 Found, not fixed

**1. [FIXED 2026-08-22, follow-up 1 — see §8; re-measurement pending]
The set net was launch-bound, and that cost ~2.8× search throughput on CUDA**
(§6.2: 118 vs 327 sims/s; CPU 259 vs 449). At a fixed 16-leaf batch the gap is only 8%, so
this is per-CALL overhead, not arithmetic: `encode_state` issues ~21 embedding lookups
(5 card fields × 3 call sites, plus key / edition / rarity / kind / set), 5 item MLPs, an
attention block, an FFN and 10 pooling reductions — ~60 kernels against the flat net's ~15,
and a search issues many small batches. The transfer half of this I did fix (packing, §3);
the kernel half has now landed too (§8 follow-up 1): the five card fields share one
offset-indexed table, edition/rarity/kind/set share a second, and all four game-key lookups
became one gather — ~25 embedding kernels down to 7, param count 1 793 268 → 1 793 536.
**The re-measurement is pending (machine in use): `python agent/benchmarks/bench_set_vs_flat.py`.** Until it
is run, assume the pre-merge finding still stands: **both nets are faster on CPU than on
CUDA at this scale** (the Phase 3 finding, AGENT_NOTES §4.4), so a self-play-bound run
should not assume `--device cuda` is the fast choice.

**2. The mean sample is 7.1× smaller, not the ~20× the brief projected** (§4.1). Subsampling
alone gives 11.3× on the mean and ~17× at a `SELECTING_HAND` leaf; the set observation is a
fixed ~5.2 KB dict, so every small-action state got bigger. Bounded and known, but it means
a 200k buffer is 1.3 GB rather than the ~0.5 GB a naive reading of "20×" would suggest. If
that ever matters, the cheap next cut is storing `act_sel` / `act_tgt` as sparse index +
weight pairs instead of dense masks (they hold at most 5 non-zeros in 50 columns) — worth
about 200 of the 288 bytes per action row.

**3. [FIXED 2026-08-22, follow-up 3] `--objective external` and `train_cold` do not meet.** The non-degenerate solo objective
lives in W2's `train_mlb.py` / `selfplay.py::play_solo_external_episode`, which drives an
`MCTSPlayer` + `SampleCollector` rather than `SelfPlayAgent`. `train_cold.py --ruleset mlb`
therefore still trains against the degenerate free-Nemesis `MLBOutcome` that the overnight
run exploited. I did not add `--objective` to `train_cold` (it would duplicate W2's driver);
the paired comparison used `train_mlb --objective external` instead. **`train_cold.py` now
REFUSES `--ruleset mlb`** (and refuses to resume an MLB checkpoint, since `--resume` takes
the ruleset from the file), pointing at `train_mlb.py --objective external`. The refusal is
CLI-only: `ColdTrainer(ruleset="mlb")` stays available to W2's `MLBTrainer` and to the tests,
which drive it behind a non-degenerate outcome.

**4. `eval/common.py::parse_player_spec` still raises `NotImplementedError` for
`checkpoint:`.** Its own docstring says W1 owns the loader; `mcts.player.load_policy(path)`
and `make_player(path)` now exist and satisfy the `Player` protocol it describes, so the
wiring is a three-line change in a file that is frozen for me. `scripts/eval_checkpoint.py`
works around it by importing `eval/common.py`'s drivers directly and emitting the
harness's own JSON schema. **W4 (or the lead at phase close) should land the three lines**;
`eval_checkpoint.py` can then be deleted or kept as the batching-aware entry point.

**5. `legal_actions()` enumeration is still inherited** (AGENT_NOTES §8): tarot targets are
enumerated for sizes 0-2 regardless of the card's real target count, and booster picks are
reduced. `encoder_set._tarot_targets` deliberately mirrors that approximation rather than
the real target counts, so the feature is honest about what the action space actually offers;
if the engine's enumeration is ever fixed, that function must be fixed with it.

**Needs engine change: none.** Everything the set encoder reads is a public attribute of
`BalatroGame` / `Card` / `JokerInstance` / `ShopItem` / `BoosterChoice`. The one private call
is `game._can_grant_choice(choice)` (`game.py:2058`), used read-only for the pack items'
`grantable` feature and wrapped in a try/except that degrades to `True`; a public
`can_grant(choice)` would be tidier but nothing is blocked.

**Not attempted:** the `hidden` / `d_item` / `n_heads` / `card_emb` / `aux_emb` sizes are
first guesses, not a sweep. (The unbounded value head that was listed here is **fixed** —
§8 follow-up 2.)

---

## 8. Follow-ups (2026-08-22, lead-requested, before the first long run)

Three changes the lead pulled forward out of §7.3 so they land before a run rather than
during one. `agent/tests/test_followups.py` (**18 tests**) covers all three.

### 8.1 Merged categorical embedding tables

`mcts/model_set.py`. Eleven `nn.Embedding` tables became **three**, addressed by per-field
offsets (`_OffsetEmbedding`):

| | before | after |
|---|---|---|
| card fields (rank/suit/enh/edition/seal) × 3 call sites | 15 gathers, dims (16,8,8,6,6) | **3** gathers, one table, dim 9 each |
| edition / rarity / shelf-kind / pack-set | 6 gathers, 4 tables | **3** gathers, one "aux" table, dim 7 |
| game keys (jokers, consumables, shelf, packs) | 4 gathers | **1** gather + `torch.split` |
| **total embedding kernels per forward** | **~25** | **7** |

The merge is *equivalent*, not an approximation: every (field, value) pair still owns its own
row, and `test_offset_embedding_matches_separate_tables` asserts the merged gather equals a
concatenation of per-field gathers for the same weights. One deliberate semantic change:
`padding_idx=0` is gone from the merged tables, so index 0 ("unknown/pad") now has a learned
row instead of a forced zero — padded item rows are masked out of the attention and the
pooling regardless, and "unknown" is a real value for a face-down card.

**Params 1 793 268 → 1 793 536** (+268). Checkpoints from before this change cannot load into
the new net (the tables have different shapes) — which is why the lead wanted it now.
**Throughput re-measurement is PENDING (machine in use): `python agent/benchmarks/bench_set_vs_flat.py`** and it is
what should decide `--device` for the first long run.

### 8.2 Bounded value heads on BOTH nets

`mcts/model.py` + `mcts/model_set.py`. Every `OutcomeFn` returns a value in [0, 1]
(`VanillaOutcome`, `MLBOutcome`, `ExternalOutcome`, and W2's population rank), but both heads
were a bare `Linear`: a cold-init flat net emitted ~1-2 and the set net ~2.1 against a [0, 1]
target, so the MSE spent its first steps just walking the head into range.

`value_activation` is now a constructor argument and a `TrainConfig` field, default
**`"sigmoid"`** (`"clamp"` available but it kills the gradient outside the interval;
`"linear"` is the old behaviour). At init both heads now sit at exactly 0.5.

* **No state_dict shape change, so `CHECKPOINT_VERSION` stays 2** (per the lead's rule).
* **Old checkpoints keep their semantics**: `describe()` records the activation, both
  `from_description` implementations default to `"linear"` when the key is absent, and
  `ColdTrainer.from_checkpoint` backfills the config field from the net description. A
  pre-follow-up checkpoint therefore resumes unbounded rather than being silently
  reinterpreted through a sigmoid (`test_a_pre_bounded_head_checkpoint_resumes_as_linear`).
* `_check_config` pins `value_activation` across a resume.
* **The bit-exact CPU round-trip is still green**, for the flat net
  (`test_checkpoint_round_trip_stays_bit_exact_with_the_bounded_head`) and for the set net
  (`test_set_encoder_checkpoint_round_trip_is_bit_exact`).

### 8.3 `train_cold.py` refuses `--ruleset mlb`

The overnight shakedown (CAMPAIGN_LOG 07:35) proved that objective is degenerate: with
`pvp_solo=True` the engine resolves the Nemesis at hand exhaustion at no cost, so the agent
learns to skip every blind and coast, and the value-target sd collapses to 0.07. The script
now exits with a message naming `train_mlb.py --objective external`.

Two guards, because `--resume` takes the ruleset from the checkpoint and would otherwise walk
straight past an argument check:

* on the parsed arguments, and
* on `trainer.cfg.ruleset` after a resume has rebuilt the config.

**The refusal is CLI-only.** `ColdTrainer(ruleset="mlb")` is untouched and still drives W2's
`MLBTrainer` and the MLB tests, which supply a non-degenerate outcome
(`test_the_mlb_objective_is_still_reachable_through_the_library`). `--ruleset vanilla` is
unchanged.

### 8.4 One failing test, and it is not W1's

`agent/tests/test_train_mlb.py::test_logged_tournament_trajectories_replay_exactly`
(W2's test, failing inside W3's `replay/replay.py:87`):

```
replay._util.ReplayMismatch: replay divergence at step 40
  (action={'indices': [2], 'type': 'pick_booster'}):
  expected sig 855a5964... got dc7a35a2...
```

**Reproduced with my value-head change reverted** (forcing `value_activation="linear"` on
both nets and re-running that test alone: identical failure), so it is not caused by the
follow-ups. `replay.apply_op` is a straight `game.step(action)` replay compared against a
digest of `game.state_signature()`, so a divergence means the recorded action list is not
sufficient to reproduce the live state across a `pick_booster` — i.e. something mutated the
game outside the logged ops around the booster, or the logged indices are pre-pick while the
engine shrinks `booster_choices` between picks. `replay/**` is W3's and the tournament
logging hook is W2's; flagged, not touched.

The other **270 tests** in `agent/tests` pass.
