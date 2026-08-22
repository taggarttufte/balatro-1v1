"""
model_set.py — `SetPolicyValueNet`, the network for the set-based observation.

    per-type item encoders (weights shared WITHIN a type, across slots)
        card    (rank,suit,enh,ed,seal embeddings + 9 numerics)  -> D   [hand / shelf / pack cards]
        joker   (key emb + edition + rarity + 16 numerics)       -> D
        cons    (key emb + 8 numerics)                           -> D
        shelf   (key emb + kind + edition + card block + 12 num) -> D
        pack    (key emb + set + edition + card block + 8 num)   -> D
            + a per-set type embedding, concatenated into ONE item sequence
            + one always-present learned GLOBAL token (index 0)
        |
    1 masked multi-head self-attention block (pre-norm) + FFN
        |
    per-set masked mean (+) max pooling  ->  5 sets x 2 x D
        |
    concat scalar_proj(scalars) -> trunk Linear -> n_res_blocks x ResidualBlock -> (B, H)
        |
    value head   Linear(H -> 1)
    policy head  pointer-style, per candidate action:
                 act_emb = type_emb + sel_proj(act_sel @ hand_emb)
                                    + tgt_proj(act_tgt @ target_emb)
                                    + num_proj(act_num)
                 logit   = w2( relu( pol_state(trunk) + pol_act(act_emb) ) )

Why attention and not pooling alone
-----------------------------------
Mean+max pooling is invariant and cheap, but Balatro's state is almost entirely
INTERACTION: a Flush-suit card is worth something only next to a suit joker, Blueprint is
worth whatever is to its right, and a shelf joker's value is a function of the board it
would join. Pooling forces every such interaction through the trunk after the item
identities have already been averaged away. One masked attention layer over the union of
all item slots lets a joker attend to the cards in hand and to its neighbours before
pooling, costs (1 + total_slots)^2 attention per state — a rounding error next to the
512-wide trunk — and adds ~30k parameters. Attention is permutation-EQUIVARIANT, so
per-set masked pooling on top is still permutation-INVARIANT; the tests assert exactly that.

Shapes
------
Everything is batched-and-padded. `obs` is the `SetEncoder` dict with a leading batch
dimension; `acts` is the `Acts` dict padded to `(B, max_actions, ...)` with `act_mask`
`(B, max_actions)` marking the live rows. B = 1 for a single leaf. The padded-block shape
is what `_segment_softmax` already materialised in `batched.py`, so this costs nothing new.

The policy head deliberately does NOT concatenate an expanded trunk onto every action row
(`(B, max_actions, H)` is 57 MB at B=64, max_actions=436, H=512). `Linear(H + A, P)` on a
concatenation is algebraically `Linear(H, P) + Linear(A, P, bias=False)`, so the state half
is computed once per state and broadcast.
"""
from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np
import torch
import torch.nn as nn

from .action_features_set import ACT_NUM_DIM, N_ACTION_TYPES_SET
from .model import _check_value_activation, apply_value_activation
from .encoder_set import (
    CARD_CARDINALITIES, CARD_NUM_DIM, CONS_NUM_DIM, ItemCaps, JOKER_CAT_DIM,
    JOKER_NUM_DIM, KEY_VOCAB_SIZE, N_EDITION, N_PACK_SET, N_RARITY, N_SHELF_KIND,
    PACK_NUM_DIM, SCALAR_DIM, SHELF_NUM_DIM,
)

SET_NAMES = ("hand", "jokers", "consumables", "shelf", "packs")


class StateEmbedding(NamedTuple):
    """What `encode_state` produces: the trunk plus the item embeddings the pointer
    policy head pools over."""
    trunk: torch.Tensor        # (B, H)
    hand: torch.Tensor         # (B, caps.hand, D)     — what `act_sel` indexes
    target: torch.Tensor       # (B, caps.target, D)   — what `act_tgt` indexes


class ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + torch.relu(self.fc2(torch.relu(self.fc1(x)))))


class _OffsetEmbedding(nn.Module):
    """Several categorical fields in ONE table, addressed by per-field offsets.

    Five separate `nn.Embedding`s over a (B, S, 5) block cost five gathers and a
    concatenate; one table indexed by `field_value + offset[field]` costs ONE gather and a
    `flatten`. Same expressive power (each field still owns its own rows), one kernel
    instead of five — which is the whole point, because the set net is launch-bound rather
    than arithmetic-bound (SETENC_NOTES 6.2). All fields share one width `dim`.

    Index 0 of every field is "unknown/pad" and now keeps a LEARNED row rather than the
    forced zero `padding_idx` gave it. Padded item rows are masked out of the attention and
    the pooling regardless, and "unknown" is a real value for a face-down card, so a
    learned vector is if anything the more honest encoding.
    """

    def __init__(self, cardinalities, dim: int):
        super().__init__()
        self.cardinalities = tuple(int(c) for c in cardinalities)
        self.dim = int(dim)
        offsets = np.concatenate([[0], np.cumsum(self.cardinalities)[:-1]])
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long),
                             persistent=False)
        self.table = nn.Embedding(int(sum(self.cardinalities)), self.dim)
        self.out_dim = self.dim * len(self.cardinalities)

    def forward(self, cat: torch.Tensor) -> torch.Tensor:
        """(B, S, F) int -> (B, S, F * dim)."""
        return self.table(_ix(cat) + self.offsets).flatten(-2)


class _CardEncoder(nn.Module):
    """The shared playing-card block: rank / suit / enhancement / edition / seal, all in
    one offset-indexed table."""

    def __init__(self, d_item: int, dim: int = 9, num_dim: int = CARD_NUM_DIM):
        super().__init__()
        self.emb = _OffsetEmbedding(CARD_CARDINALITIES, dim)
        self.cat_dim = self.emb.out_dim
        self.num_dim = num_dim
        self.out_dim = d_item

    def embed_cat(self, cat: torch.Tensor) -> torch.Tensor:
        """(B, S, 5) int -> (B, S, cat_dim)."""
        return self.emb(cat)


def _mlp(in_dim: int, d_item: int) -> nn.Sequential:
    m = nn.Sequential(nn.Linear(in_dim, d_item), nn.ReLU(), nn.Linear(d_item, d_item))
    nn.init.orthogonal_(m[0].weight, gain=np.sqrt(2))
    nn.init.constant_(m[0].bias, 0)
    nn.init.orthogonal_(m[2].weight, gain=np.sqrt(2))
    nn.init.constant_(m[2].bias, 0)
    return m


class SetPolicyValueNet(nn.Module):

    def __init__(
        self,
        caps: ItemCaps | dict | None = None,
        d_item: int = 64,
        n_heads: int = 4,
        ffn_mult: int = 2,
        key_emb: int = 48,
        card_emb: int = 9,
        aux_emb: int = 7,
        hidden: int = 512,
        n_res_blocks: int = 2,
        scalar_hidden: int = 256,
        action_hidden: int = 128,
        policy_hidden: int = 128,
        scalar_dim: int = SCALAR_DIM,
        act_num_dim: int = ACT_NUM_DIM,
        value_activation: str = "sigmoid",
    ):
        super().__init__()
        self.caps = caps if isinstance(caps, ItemCaps) else ItemCaps.from_dict(caps)
        self.d_item = d_item
        self.n_heads = n_heads
        self.ffn_mult = ffn_mult
        self.key_emb = key_emb
        self.hidden = hidden
        self.n_res_blocks = n_res_blocks
        self.scalar_hidden = scalar_hidden
        self.action_hidden = action_hidden
        self.policy_hidden = policy_hidden
        self.scalar_dim = scalar_dim
        self.act_num_dim = act_num_dim
        self.value_activation = _check_value_activation(value_activation)
        D = d_item

        # ── shared tables ────────────────────────────────────────────────────
        # Everything categorical lives in THREE tables, not eleven: the card block, one
        # "aux" table holding edition / rarity / shelf-kind / pack-set behind offsets, and
        # the game-key table. Each item type then costs one gather per table it uses.
        self.card_emb = card_emb
        self.aux_emb = aux_emb
        self.card = _CardEncoder(D, dim=card_emb)
        self.key_embed = nn.Embedding(KEY_VOCAB_SIZE, key_emb, padding_idx=0)
        # aux field order: 0 edition, 1 rarity, 2 shelf kind, 3 pack set
        self._aux_cards = (N_EDITION, N_RARITY, N_SHELF_KIND, N_PACK_SET)
        self.aux_table = nn.Embedding(int(sum(self._aux_cards)), aux_emb)
        aux_off = np.concatenate([[0], np.cumsum(self._aux_cards)[:-1]])
        # Offset pairs in the COLUMN ORDER the encoder writes:
        #   joker_cat = [edition, rarity] | shelf_cat = [kind, edition]
        #   pack_cat  = [set, edition]
        for _name, _pair in (("_off_joker", (aux_off[0], aux_off[1])),
                             ("_off_shelf", (aux_off[2], aux_off[0])),
                             ("_off_pack", (aux_off[3], aux_off[0]))):
            self.register_buffer(_name, torch.tensor(_pair, dtype=torch.long),
                                 persistent=False)

        aux2 = 2 * aux_emb
        self.hand_mlp = _mlp(self.card.cat_dim + CARD_NUM_DIM, D)
        self.joker_mlp = _mlp(key_emb + aux2 + JOKER_NUM_DIM, D)
        self.cons_mlp = _mlp(key_emb + CONS_NUM_DIM, D)
        self.shelf_mlp = _mlp(key_emb + aux2 + self.card.cat_dim + SHELF_NUM_DIM, D)
        self.pack_mlp = _mlp(key_emb + aux2 + self.card.cat_dim + PACK_NUM_DIM, D)

        self.set_embed = nn.Embedding(len(SET_NAMES) + 1, D)   # 0 = the global token

        # ── one masked attention block (pre-norm) ────────────────────────────
        self.attn_norm = nn.LayerNorm(D)
        self.attn = nn.MultiheadAttention(D, n_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(D)
        self.ffn = nn.Sequential(nn.Linear(D, ffn_mult * D), nn.ReLU(),
                                 nn.Linear(ffn_mult * D, D))

        # ── trunk ────────────────────────────────────────────────────────────
        self.scalar_proj = nn.Sequential(nn.Linear(scalar_dim, scalar_hidden), nn.ReLU())
        nn.init.orthogonal_(self.scalar_proj[0].weight, gain=np.sqrt(2))
        nn.init.constant_(self.scalar_proj[0].bias, 0)
        pooled = 2 * D * len(SET_NAMES) + D            # + the global token
        self.trunk_in = nn.Sequential(nn.Linear(pooled + scalar_hidden, hidden), nn.ReLU())
        nn.init.orthogonal_(self.trunk_in[0].weight, gain=np.sqrt(2))
        nn.init.constant_(self.trunk_in[0].bias, 0)
        self.res_blocks = nn.Sequential(*[ResidualBlock(hidden) for _ in range(n_res_blocks)])

        self.value_head = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.constant_(self.value_head.bias, 0)

        # ── action embedding + pointer policy head ───────────────────────────
        A = action_hidden
        self.act_type_embed = nn.Embedding(N_ACTION_TYPES_SET, A)
        self.act_sel_proj = nn.Linear(D, A, bias=False)
        self.act_tgt_proj = nn.Linear(D, A, bias=False)
        self.act_num_proj = nn.Linear(act_num_dim, A)
        nn.init.constant_(self.act_num_proj.bias, 0)
        self.act_norm = nn.LayerNorm(A)

        self.pol_state = nn.Linear(hidden, policy_hidden)
        self.pol_act = nn.Linear(A, policy_hidden, bias=False)
        self.pol_out = nn.Linear(policy_hidden, 1)
        nn.init.orthogonal_(self.pol_state.weight, gain=np.sqrt(2))
        nn.init.constant_(self.pol_state.bias, 0)
        nn.init.orthogonal_(self.pol_act.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.pol_out.weight, gain=0.01)   # near-uniform priors at init
        nn.init.constant_(self.pol_out.bias, 0)

    # ── introspection / checkpoint metadata ─────────────────────────────────

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def describe(self) -> dict:
        return {
            "kind": "set",
            "caps": self.caps.as_dict(),
            "d_item": self.d_item, "n_heads": self.n_heads, "ffn_mult": self.ffn_mult,
            "key_emb": self.key_emb, "card_emb": self.card_emb,
            "aux_emb": self.aux_emb, "hidden": self.hidden,
            "n_res_blocks": self.n_res_blocks, "scalar_hidden": self.scalar_hidden,
            "action_hidden": self.action_hidden, "policy_hidden": self.policy_hidden,
            "scalar_dim": self.scalar_dim, "act_num_dim": self.act_num_dim,
            "value_activation": self.value_activation,
        }

    @classmethod
    def from_description(cls, desc: dict) -> "SetPolicyValueNet":
        d = {k: v for k, v in desc.items() if k != "kind"}
        d = {"value_activation": "linear", **d}     # see PolicyValueNet.from_description
        return cls(**d)

    # ── state ────────────────────────────────────────────────────────────────

    def encode_state(self, obs: dict) -> StateEmbedding:
        """`obs`: the `SetEncoder` dict with a leading batch dim, as tensors."""
        caps = self.caps
        hand = self.hand_mlp(torch.cat(
            [self.card.embed_cat(obs["hand_cat"]), obs["hand_num"]], dim=-1))
        # ONE gather for every game key in the state (jokers | consumables | shelf | packs),
        # then split: four `nn.Embedding` calls become one.
        nj, nc, ns, npk = caps.jokers, caps.consumables, caps.shelf, caps.packs
        keys = self.key_embed(_ix(torch.cat(
            [obs["joker_key"], obs["cons_key"], obs["shelf_key"], obs["pack_key"]], dim=1)))
        k_joker, k_cons, k_shelf, k_pack = torch.split(keys, [nj, nc, ns, npk], dim=1)

        joker = self.joker_mlp(torch.cat([
            k_joker,
            self.aux_table(_ix(obs["joker_cat"]) + self._off_joker).flatten(-2),
            obs["joker_num"]], dim=-1))
        cons = self.cons_mlp(torch.cat([k_cons, obs["cons_num"]], dim=-1))
        shelf = self.shelf_mlp(torch.cat([
            k_shelf,
            self.aux_table(_ix(obs["shelf_cat"]) + self._off_shelf).flatten(-2),
            self.card.embed_cat(obs["shelf_card"]),
            obs["shelf_num"]], dim=-1))
        pack = self.pack_mlp(torch.cat([
            k_pack,
            self.aux_table(_ix(obs["pack_cat"]) + self._off_pack).flatten(-2),
            self.card.embed_cat(obs["pack_card"]),
            obs["pack_num"]], dim=-1))

        B = hand.shape[0]
        dev = hand.device
        # per-set type embedding (1..5; 0 is the global token)
        parts = [hand, joker, cons, shelf, pack]
        for i, p in enumerate(parts):
            parts[i] = p + self.set_embed.weight[i + 1]
        glob = self.set_embed.weight[0].expand(B, 1, -1)
        seq = torch.cat([glob] + parts, dim=1)                       # (B, 1+S, D)

        masks = [obs["hand_mask"], obs["joker_mask"], obs["cons_mask"],
                 obs["shelf_mask"], obs["pack_mask"]]
        item_mask = torch.cat(masks, dim=1)                          # (B, S)
        ones = torch.ones(B, 1, device=dev, dtype=item_mask.dtype)
        full_mask = torch.cat([ones, item_mask], dim=1)              # (B, 1+S)
        # The global token is always live, so no row is ever fully masked (which would
        # make the attention softmax NaN) — that matters because a BLIND_SELECT state can
        # legitimately have an empty hand, no jokers, no consumables, no shop and no pack.
        pad = full_mask <= 0

        h = self.attn_norm(seq)
        attended, _ = self.attn(h, h, h, key_padding_mask=pad, need_weights=False)
        seq = seq + attended
        seq = seq + self.ffn(self.ffn_norm(seq))

        glob_out = seq[:, 0]
        items = seq[:, 1:]
        pooled = [glob_out]
        at = 0
        for m in masks:
            w = m.shape[1]
            pooled.append(_masked_mean(items[:, at:at + w], m))
            pooled.append(_masked_max(items[:, at:at + w], m))
            at += w

        trunk = self.trunk_in(torch.cat(
            pooled + [self.scalar_proj(obs["scalars"])], dim=-1))
        trunk = self.res_blocks(trunk)

        hand_emb = items[:, :caps.hand]
        target_emb = items[:, caps.hand:]
        return StateEmbedding(trunk=trunk, hand=hand_emb, target=target_emb)

    def value(self, state: StateEmbedding | torch.Tensor) -> torch.Tensor:
        trunk = state.trunk if isinstance(state, StateEmbedding) else state
        return apply_value_activation(self.value_head(trunk).squeeze(-1),
                                      self.value_activation)

    # ── actions ──────────────────────────────────────────────────────────────

    def action_embedding(self, state: StateEmbedding, acts: dict) -> torch.Tensor:
        """`acts` padded to (B, N, ...) -> (B, N, action_hidden)."""
        e = self.act_type_embed(_ix(acts["act_type"]))
        e = e + self.act_sel_proj(torch.bmm(acts["act_sel"], state.hand))
        e = e + self.act_tgt_proj(torch.bmm(acts["act_tgt"], state.target))
        e = e + self.act_num_proj(acts["act_num"])
        return self.act_norm(e)

    def action_logits(self, state: StateEmbedding, acts: dict) -> torch.Tensor:
        """(B, N) logits. Padded rows are finite but meaningless — mask them before the
        softmax (the policy and the trainer both do)."""
        act_emb = self.action_embedding(state, acts)
        x = torch.relu(self.pol_state(state.trunk).unsqueeze(1) + self.pol_act(act_emb))
        return self.pol_out(x).squeeze(-1)

    def forward(self, obs: dict, acts: dict) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.encode_state(obs)
        return self.action_logits(state, acts), self.value(state)


def _ix(t: torch.Tensor) -> torch.Tensor:
    """Widen a stored int16 categorical to the Long an `nn.Embedding` needs. A no-op when
    the caller already handed over int64."""
    return t if t.dtype == torch.long else t.long()


# ── masked pooling ──────────────────────────────────────────────────────────────

def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1)
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


_NEG = -1e9


def _masked_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1)
    filled = x * m + (1.0 - m) * _NEG
    out = filled.max(dim=1).values
    any_live = (mask.sum(dim=1, keepdim=True) > 0).to(x.dtype)
    return out * any_live          # all-padded set -> 0, not -1e9


__all__ = ["SetPolicyValueNet", "StateEmbedding", "SET_NAMES"]
