# VALUE_NOTES — Phase 5 W1: STATE_SPEC v1 encoder + `SetValueNet` (5M)

**Agent W1, 2026-08-23.** Deliverable: `mp/agent/mcts/encoder_v2.py`, `mp/agent/mcts/value_net.py`,
`tests/test_encoder_v2.py` (23), `tests/test_value_net.py` (12), this file, and one additive change to
`mp/engine/balatro_sim/mlb_match.py` (§6). `encoder_set.py` / `model_set.py` are **byte-for-byte
unchanged**; the v2 encoder subclasses `SetEncoder` and the value net imports `model_set`'s blocks.
Nothing here imports `mp/ev` or `mp/stats`.

---

## 0. Interfaces (as implemented)

```python
# mp/agent/mcts/encoder_v2.py
STATE_SPEC_VERSION = 1
SCALAR_LAYOUT_V2: list[tuple[str, int]]      # 28 blocks, spec order; SCALAR_DIM_V2 = 355
KEY_VOCAB_V2 / KEY_VOCAB_SIZE_V2 = 439       # KEY_IDX_V2, key_index_v2(key), SPARE_KEY_MAP, SPARE_KEY_BASE = 407
ItemCapsV2(hand=16, jokers=12, consumables=6, shelf=8, packs=8, blinds=3)   # DEFAULT_CAPS_V2
def layout_fingerprint(caps=None) -> str     # sha256 over (spec version, scalar layout, item widths, caps, vocab)
def scalar_offsets_v2() -> dict[str, int] ; def scalar_layout_table() -> [(name, width, offset)]

@dataclass class NemesisLive(their_score, their_hands_left, my_score, my_hands_left)
@dataclass class NemesisRecord(ante, their_score, their_hands_used, my_score, outcome(+1/0/-1), early_end)
@dataclass class OpponentView(known=False, lives, skips, dollars, ante, blind_idx, state, pvp_ready,
                              pvp_exhausted, chips_scored, hands_left, comeback_bonus, comeback_pending,
                              current_nemesis: NemesisLive|None, last_nemeses: list[NemesisRecord] (<=4,
                              most recent first), my_last_loss_ante: int|None,
                              sells_per_ante, spent_in_shop, sells_total, spent_total)   # .phase property
NO_OPPONENT = OpponentView()                 # vanilla / solo -> the opponent block is all zeros
def opponent_view(match: MLBMatch, player: int) -> OpponentView   # MLBMatch.player_view() x2 + pvp_detail ONLY

class SetEncoderV2(SetEncoder):              # name = "set_v2", is_set = True, dim = None, .fingerprint
    def __init__(self, caps: ItemCapsV2 | ItemCaps | dict | None = None)
    def __call__(self, game: BalatroGame, opp: OpponentView | None = None) -> Obs
    def describe(self) -> dict ; @classmethod from_description(cls, d)   # raises on fingerprint/version mismatch
    batch(obs_list) -> Obs                   # inherited
def collate(obs_list, device="cpu") -> dict[str, torch.Tensor]   # packed transfer off-CPU (policy_set._stack_obs)

# mp/agent/mcts/value_net.py
@dataclass class ValueNetConfig(d_item=128, n_heads=4, ffn_mult=2, key_emb=64, card_emb=12, aux_emb=8,
                                trunk_width=712, n_res_blocks=3, scalar_hidden=384, caps=dict,
                                scalar_dim=355, key_vocab=439)          # .as_dict() / .from_dict()
class SetValueNet(nn.Module):
    def __init__(self, cfg: ValueNetConfig | dict | None = None)
    def forward(self, batch: dict) -> torch.Tensor      # (B,) LOGITS; .sigmoid() = P(win); .p_win(batch)
    def encode(self, batch) -> (B, W) trunk ; def n_params(self) -> int ; def param_breakdown(self) -> dict
    def describe(self) / from_description(desc)
def save_checkpoint(path, net, encoder, extra: dict | None = None) -> Path
def load_checkpoint(path, device="cpu") -> (net, encoder, extra)        # ValueError on fingerprint / version / kind
def make_value_fn(net, encoder, device="cpu") -> fn(game, opp=None) -> float            # P(win)
def make_values_many(net, encoder, device="cpu", chunk=512) -> fn([(game, opp|None), ...]) -> np.ndarray (N,) float32
```

`Obs` = the 21 Phase-4 arrays (SETENC_NOTES §0.2, unchanged shapes) + 5 blind-offer arrays + `scalars`
of width 355:

| key | shape | dtype | meaning |
|---|---|---|---|
| `blind_key` | (3,) | int16 | Boss slot: boss key / `bl_mp_nemesis` in `KEY_VOCAB_V2`; 0 for Small/Big |
| `blind_tag` | (3,) | int16 | Small/Big: the TAG ON OFFER (`game.blind_tags`) in `KEY_VOCAB_V2`; 0 for Boss |
| `blind_cat` | (3, 2) | int16 | kind (1 Small / 2 Big / 3 Boss), status (1 upcoming / 2 current / 3 done) |
| `blind_num` | (3, 8) | float32 | `log1p(target)/log1p(1e5)`, reward $/8, is_pvp, skip-available, is_current, is_done, is_showdown, disabled |
| `blind_mask` | (3,) | float32 | always 1 (the ante always has three blinds) |
| `scalars` | (355,) | float32 | §1 |

---

## 1. `scalars` layout (SCALAR_LAYOUT_V2, 355)

Generated from the code (`scalar_layout_table()`); `test_spec_version_and_layout_sum` pins the widths and
that the first 16 blocks are `encoder_set.SCALAR_LAYOUT` verbatim (offsets 0..195 are bit-identical to the
Phase 4 encoder — `test_first_196_scalars_match_the_phase4_encoder`).

| block | width | offset | contents (v1 additions; the first 16 are SETENC_NOTES §1.6) |
|---|---|---|---|
| `state` | 7 | 0 | |
| `ante_blind` | 8 | 7 | |
| `blind_kind` | 3 | 15 | |
| `boss` | 28 | 18 | |
| `economy` | 10 | 46 | |
| `capacity` | 9 | 56 | (normalised by the FIXED default caps, see §4.3) |
| `round` | 8 | 65 | |
| `planet_levels` | 12 | 73 | |
| `hand_types_played` | 12 | 85 | |
| `vouchers` | 32 | 97 | |
| `tags` | 24 | 129 | |
| `tag_count` | 1 | 153 | |
| `deck_composition` | 16 | 154 | (the full-deck fractions, kept) |
| `deck_id` | 15 | 170 | |
| `stake` | 1 | 185 | |
| `mlb` | 10 | 186 | |
| **`deck_counts`** | 34 | 196 | REMAINING DRAW PILE (`game.deck`): rank counts 13 (/8), suit counts 4 (/26), enhancement counts 8 Bonus..Lucky (/16), edition counts 5 None(/52) Foil Holo Poly Negative (/16), seal counts 4 Gold Red Blue Purple (/16); all clipped to 1. Counts, so the pile's ORDER cannot leak. |
| **`discard_counts`** | 17 | 230 | `game.discard_pile` (played + discarded this blind): rank 13 (/8), suit 4 (/26) |
| **`ruleset`** | 3 | 247 | one-hot vanilla / mlb / the_order (`queue_scope == "run"`) |
| **`pvp_start_round`** | 1 | 250 | /8, MLB only |
| **`opp_basic`** | 12 | 251 | known, lives/4, skips/8, $/50, log1p($)/log1p(200), ante/8, phase one-hot 6 (selecting / small / big / nemesis / shop / waiting) |
| **`opp_nemesis`** | 4 | 263 | Nemesis in progress: their score (log), their hands left /4, my score (log), my hands left /4 |
| **`opp_history`** | 24 | 267 | last 4 Nemeses, most recent first, 6 each: ante/8, their final score (log), hands they used /4, my score (log), outcome (+1/0/−1), early-end flag |
| **`opp_belief`** | 16 | 291 | RESERVED — zeros until the level-1 belief model |
| **`opp_econ`** | 4 | 307 | sells this ante /5, $ spent in shop this ante /50, log1p(total sells)/log1p(50), log1p(total spent)/log1p(500) |
| **`race`** | 6 | 311 | my lives/4, their lives/4, ante/8, comeback $ pending /16, life lost this round, antes since my last life lost /8 |
| **`money_detail`** | 6 | 317 | interest at current $ (/10), $ to the next interest threshold (/5, 0 when capped), reroll cost now (/10), free rerolls (/3), shop discount, voucher-on-offer cost (`effective_price`, /20) |
| **`reserved`** | 32 | 323 | zeros |

Log scores use `log1p(x)/log1p(1e5)` as the Phase 4 encoder does. Interest uses the engine's own
`INTEREST_RATE` / `game.interest_cap` / `no_interest`; blind targets use `blind_base_chips(ante, idx,
blind_scaling) * ante_scaling * BOSS_CHIP_MULT`, i.e. `_prepare_next_blind`'s formula, and the CURRENT
blind's target is read straight from `current_blind.chips_target` (Chicot / Nemesis-live-score exact).

## 2. `KEY_VOCAB_V2` (439, built from `game_keys` at import — no literal list anywhere)

| segment | n | ids |
|---|---|---|
| `<pad>`, `<unk>` | 2 | 0..1 |
| jokers | 150 | 2..151 |
| tarots | 22 | 152..173 |
| planets | 12 | 174..185 |
| spectrals | 18 | 186..203 |
| vouchers | 32 | 204..235 |
| booster TYPE keys (what the engine / shelf emit) | 15 | 236..250 |
| booster CENTER keys (`p_arcana_normal_1`…, the spec's "32 boosters"; reserved, never emitted today) | 32 | 251..282 |
| tags | 24 | 283..306 |
| blinds (`bl_small`, `bl_big`, 28 bosses, `bl_mp_nemesis`) | 31 | 307..337 |
| card fronts `S_2`..`D_A` (reserved: cards use the 5-column card block) | 52 | 338..389 |
| enhancements `m_*` / editions `e_*` / seals (reserved) | 8 / 5 / 4 | 390..406 |
| `<spare_0..31>` | 32 | 407..438 |

Ids 0..250 ARE `encoder_set.KEY_VOCAB` (asserted at import), so the reused joker / consumable / shelf /
pack encoders index it unchanged. New content goes through `SPARE_KEY_MAP[key] = SPARE_KEY_BASE + i`
(tested) — never by regenerating the list; a pools change that shifts ids changes the fingerprint and
refuses to load old nets, which is the intended failure. The spec's "13 boosters" / "28 blinds" counts are
the engine's 15 type keys and `pools.BLINDS` 30 (+ Nemesis); the tables, not the prose, are authoritative.

Fingerprint of this layout (default caps): `5167cdc1785a4d7b…` (`layout_fingerprint()`).

## 3. `SetValueNet` — 4 996 789 parameters

```
card block (rank,suit,enh,ed,seal @ 12) ⊕ 9 num ────────────── hand_mlp  ─┐
key emb 64 ⊕ aux(edition,rarity @ 8) ⊕ 16 num ──────────────── joker_mlp ─┤
key emb 64 ⊕ 8 num ──────────────────────────────────────────── cons_mlp  ─┤
key emb 64 ⊕ aux(kind,edition) ⊕ card block ⊕ 12 num ────────── shelf_mlp ─┼─ D=128 each, + set-type emb
key emb 64 ⊕ aux(set,edition) ⊕ card block ⊕ 8 num ──────────── pack_mlp  ─┤   + 1 learned global token
boss-key emb 64 ⊕ tag-key emb 64 ⊕ aux(kind,status) ⊕ 8 num ─── blind_mlp ─┘   = 54-slot item sequence
        │
   1 masked MHA block (4 heads, D=128, pre-norm) + FFN(256)        key_padding_mask, global token never padded
        │
   per-set masked mean ⊕ max over 6 sets (12 × 128) ⊕ global (128) = 1664   ⊕ scalar_proj(355 → 384)
        │
   trunk_in Linear(2048 → 712) → 3 × ResidualBlock(712)  → value_head Linear(712 → 1)  → ONE LOGIT
```

| component | params |
|---|---|
| tables (card / aux / key 439×64 / set) | 29 764 |
| item MLPs (6 types) | 188 032 |
| attention block + FFN | 132 480 |
| scalar_proj | 136 704 |
| trunk_in | 1 458 888 |
| res_blocks (3 × 712) | 3 050 208 |
| value_head | 713 |
| **total** | **4 996 789** |

712 is the trunk width that lands 3 res blocks at 5.0M (768 → 5.6M, 704 → 4.9M); it is a multiple of 8.
`forward` returns logits (train with `BCEWithLogitsLoss`); everything downstream (`make_value_fn`,
`make_values_many`, `p_win`) returns the sigmoid. Same per-item encoders, attention, masked pooling and
`ResidualBlock` as `SetPolicyValueNet` (imported from `model_set`), same three-table categorical scheme
(the aux table gains blind kind + blind status), one key gather for all six key columns.

## 4. Results

### 4.1 Gate numbers (this box, shared with Tagg: `torch.set_num_threads(4)`, single-shot, 2026-08-23)

| measurement | value |
|---|---|
| encoder v2, 301 states of a real MLB match with opponent views | **0.120 ms/state** (the Phase-4 `SetEncoder` alone: 0.065 ms; the v1 blocks cost ~0.055 ms, mostly `_count_cards` over the draw pile + `_encode_scalars` being called then copied) |
| `SetValueNet` forward, CPU 4 threads | B=1 **1.99 ms**, B=256 **37.4 ms** |
| `SetValueNet` forward, CUDA (RTX 3080 Ti) | B=1 **3.88 ms**, B=256 **3.40 ms** (10.2 ms including `collate` + host→device) |
| `make_values_many` end to end (encode + collate + forward), 256 states | CUDA **46.5 ms** (0.18 ms/state), CPU **75.5 ms** (0.29 ms/state) |
| CPU vs CUDA logits | max abs diff 2.6e-6 |

The full CUDA sweep was deferred per the lead's instruction (machine in use); these are single-shot numbers.
At B=1 CUDA is launch-bound and slower than CPU, exactly as for the Phase-4 set net (SETENC_NOTES §6.2);
`make_values_many` is the path `EVPlayer` / the label generator should use, in batches.

### 4.2 Tests

| suite | result |
|---|---|
| `python -m pytest mp/agent/tests` | **396 passed / 0 failed** (337 at kickoff + W1's 35: 23 `test_encoder_v2.py` + 12 `test_value_net.py` (1 CUDA test, skipped without a GPU) + W2's 24 `test_determinize_player.py` that landed concurrently; 192 s on the shared box — W1's two files run in 3.5 s) |
| `python -m pytest mp/engine/tests` | **1649 passed / 10 skipped / 3 xfailed** (1614 at kickoff + W2's determinize tests, all green with the `mlb_match.py` change) |
| `python -m pytest mp/tests` | **1073 passed / 2 xfailed** (unchanged) |

What the tests pin: layout widths/offsets/order; vocabulary sizes against the live tables + the Phase-4
prefix + the spare-id promise; fingerprint stable and sensitive to caps / layout / vocab / version;
bit-identical re-encode and clone; **permuting `game.deck` in place leaves the Obs bit-identical**; no
NaN/inf on every state of a full MLB match (both players, with opponent views; GAME_OVER included); the
first 196 scalars equal the Phase-4 encoder; **`opponent_view` is blind to the opponent's hand / jokers /
consumables / shop / deck / discard pile / planet levels / vouchers and sees lives / skips / $**;
vanilla & `opp=None` → all-zero opponent block; blind offers (tags on offer, boss key, Nemesis key under
MLB, targets vs `blind_base_chips`, skip-available only on the current Small/Big at blind select, statuses
incl. the post-boss shop showing the NEXT ante); deck/discard counts vs the live pile; money_detail by
hand; econ counters; last-life-loss tracking on a failed Small; describe/from_description round trip;
**5M budget**; finite logits; **pad-invariance both ways** (appended padded slots with the net's caps
widened, AND a larger-caps encoder + same weights → same logit to 1e-4); within-set permutation
invariance; garbage in padded rows; the all-zero state; **bit-exact checkpoint round trip**
(`torch.equal` on every tensor + equal outputs); refusals for fingerprint / STATE_SPEC_VERSION / file
version / kind; `make_values_many` == forward, order preserved, chunking irrelevant, `opp=None` accepted,
train/eval mode restored; a gradient step moves the loss; CPU == CUDA.

## 5. Decisions and deviations from the spec (flagged for the lead)

1. **`opp_basic` is 12, not "≈10"**: the spec's lives / skips / phase one-hot (6) plus `known`, `$`, `log $`
   and `ante`. `$` is in `PlayerView` (the engine already treats it as public) and the mod's lobby HUD shows
   it; the opponent's ante disambiguates "they are in the shop" (which ante's shop).
2. **`opp_econ` 4 included** (Tagg's open item 2 — "I'd say yes"): sells this ante, $ spent this ante, and
   the two run totals. The engine did not record them, so `MLBMatch` now does (§6).
3. **`pvp_log` was NOT extended in place.** `mp/replay/replay.py:288` and `mp/eval/common.py:323` unpack
   it as exactly four values, so a 7-tuple would have broken frozen readers. A parallel `pvp_detail` list
   carries `(ante, loser, s0, s1, hands_played0, hands_played1, early_end)`; `pvp_log` is untouched.
4. **Blind-offer semantics in the post-boss shop.** `_end_round` advances `ante` on entering the shop while
   `blind_idx` still points at the beaten Boss. The offers encoded there are the NEW ante's three blinds
   (its tags / boss are already drawn), all "upcoming"; a tag pack open at the Boss's blind-select screen is
   distinguished via `_booster_return_state`. Elsewhere "current" = the slot `blind_idx` points at (being
   selected / played / just beaten with its shop open — the `state` one-hot disambiguates).
5. **Caps are transport only, enforced**: position features (`slot / cap`) and `capacity` ratios are
   normalised by the FIXED default caps, not the live caps (`_renormalise_positions`, base scalars encoded
   with `DEFAULT_CAPS`). Caps are nevertheless part of the fingerprint because they are part of the
   checkpoint's transport contract.
6. **Edition count 5 includes "None"** (plain cards /52) so the block is 34 as specified and the plain
   count is there without subtraction.
7. **"Antes since last life lost"** comes from `MLBMatch.last_life_loss_ante` (tracked in `sync()`, so a
   failed Small/Big counts, not only a Nemesis). With no match (`opp=None`) the feature is `ante/8` ("no
   loss known"). The other `race` fields come from the game itself.
8. **Booster vocabulary**: both the 15 TYPE keys (emitted) and the 32 CENTER keys (the spec's "32
   boosters", reserved). Card fronts / enhancements / editions / seals are in the vocabulary as the spec
   lists them but no item emits them today (cards use the card block) — reserved for a future prior.
9. `ValueNetConfig.trunk_width = 712` rather than the brief's "1024 × 3": 1024 × 3 is 8.9M; 712 × 3 is the
   5.0M Tagg asked for with d_item 128 / key_emb 64.
10. `make_value_fn` returns **P(win)** (sigmoid), since `value_fn` is V(state) = P(win) in the brief; the
    raw logit is `net(batch)`.

## 6. The engine change (`mp/engine/balatro_sim/mlb_match.py`, additive only)

* `PlayerView` gains defaulted fields `hands_played`, `sells_per_ante`, `spent_in_shop`, `sells_total`,
  `spent_total`, `last_life_loss_ante` (positional construction unchanged).
* `PlayerEcon` dataclass; `MLBMatch.econ = [PlayerEcon(), PlayerEcon()]`, filled by `_track_econ` in
  `step()` (sell → `sells_per_ante` / `sells_total`; $ leaving the wallet on `buy` / `reroll` → `spent_*`;
  per-ante counters reset when the game's ante moves). A game stepped directly (env_mp) keeps zeros.
* `MLBMatch.pvp_detail` (see §5.3), appended in `_resolve_pvp` next to `pvp_log`.
* `MLBMatch.last_life_loss_ante` per player, updated by `_track_lives()` at the end of `sync()`.
* `MLBMatch.player_view(p)` — the one public accessor `opponent_view` uses; `state()` now calls it.
* `clone()` copies all of the above. `signature()` is unchanged.

## 7. For the lead to wire

* `mcts/encoder.py::get_encoder` does not know `"set_v2"` (file not owned by W1). If you want
  `--encoder set_v2` anywhere, add a lazy `ENCODERS["set_v2"]` pointing at `encoder_v2.SetEncoderV2`.
* W5's trainer: `logits = net(collate(obs, device))`, `loss = BCEWithLogitsLoss()(logits, p_win_labels)`;
  checkpoint with `value_net.save_checkpoint(path, net, encoder, extra={...})` (optimizer state etc. go in
  `extra`, which is `torch.save`d as-is).
* W3/W6: `value_fn = make_value_fn(net, encoder, device)`; for anything that evaluates more than one state
  per decision use `make_values_many` — it is ~10× cheaper per state on CUDA than B=1 calls.
* The opponent block is only non-zero when the caller passes `opponent_view(match, p)`; `EVPlayer(game)`
  alone (no match handle) gets the vanilla/solo encoding. The `Player` protocol in the brief passes only
  the game — if V should see the opponent in tournament/eval play, the player needs the match (or an
  `OpponentView`) handed to it.

## 8. Open issues

* `opp_basic.phase` is None (all-zero) for an opponent at GAME_OVER; `known` stays 1. Fine for V (the
  match is over), noted for completeness.
* `money_detail[5]` is the first unsold voucher on the shelf, 0 when there is none or when it is couponed
  (free) — the shelf set carries the voucher item itself, so "free voucher on offer" is still visible.
* The encoder's `hand_num` inherits the Phase-4 unclipped `base_chips / 11` (a Stone card is 4.5); left
  as-is to keep the first 196 scalars + item blocks identical to Phase 4.
* No `opp_belief` producer exists (by design, level 1 is later); the 16 zeros are pre-reserved.
