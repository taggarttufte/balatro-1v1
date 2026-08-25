"""
value_net.py — `SetValueNet`: V(state) ≈ P(win the MLB match), on the STATE_SPEC v1
observation (`encoder_v2.SetEncoderV2`). Phase 5 W1, 2026-08-23.

    per-type item encoders (weights shared WITHIN a type, across slots)
        card    (rank,suit,enh,ed,seal embeddings + 9 numerics)        -> D   [hand / shelf / pack cards]
        joker   (key emb + edition + rarity + 16 numerics)             -> D
        cons    (key emb + 8 numerics)                                 -> D
        shelf   (key emb + kind + edition + card block + 12 numerics)  -> D
        pack    (key emb + set + edition + card block + 8 numerics)    -> D
        blind   (boss-key emb + tag-key emb + kind + status + 8 num)   -> D   [NEW in v1]
            + a per-set type embedding, concatenated into ONE item sequence
            + one always-present learned GLOBAL token (index 0)
        |
    1 masked multi-head self-attention block (pre-norm) + FFN
        |
    per-set masked mean (+) max pooling  ->  6 sets x 2 x D  (+ the global token)
        |
    concat scalar_proj(scalars 355) -> trunk Linear -> n_res_blocks x ResidualBlock(W) -> (B, W)
        |
    value head Linear(W -> 1)  ->  ONE LOGIT.  `forward` returns logits; `.sigmoid()` = P(win).

Derived from `model_set.SetPolicyValueNet` (same per-item encoders, same attention block,
same masked pooling, same bit-exact checkpoint conventions) with the policy head removed,
the blind-offer set added, and the trunk / item widths raised to land at ≈ 5.0M params
(Tagg's call, STATE_SPEC_v1 §Net). Nothing here imports `mp/ev` or `mp/stats`.

Pad-invariance: padded rows are excluded from the attention (`key_padding_mask`) and from
both poolings, and the global token keeps every attention row non-empty, so appending
padded slots (or an encoder with larger caps) cannot change the logit — pinned by
`tests/test_value_net.py`.

Checkpoint: `save_checkpoint(path, net, encoder, extra)` writes one atomic `torch.save`
payload carrying `STATE_SPEC_VERSION`, `layout_fingerprint()`, the encoder description and
the `ValueNetConfig`; `load_checkpoint(path, device)` refuses a fingerprint mismatch and
rebuilds `(net, encoder, extra)` bit-exactly.

Auxiliary heads (Phase 5 rev 2, W-AUX; `ev/AUX_NOTES.md`): `ValueNetConfig.aux_heads`
(default `{}`) optionally attaches small heads to the SHARED TRUNK, used only by
`forward_with_aux` inside the trainer. With none configured nothing is constructed — same
parameters, same init draws, same `state_dict` — so an old checkpoint (no `aux_heads` key
in its `cfg`) still loads `strict=True`, and `forward` / `p_win` / `make_value_fn` /
`make_values_many` are untouched whether heads exist or not.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from .encoder_set import (
    CARD_NUM_DIM, CONS_NUM_DIM, JOKER_NUM_DIM, N_EDITION, N_PACK_SET,
    N_RARITY, N_SHELF_KIND, PACK_NUM_DIM, SHELF_NUM_DIM,
)
from .encoder_v2 import (
    BLIND_NUM_DIM, DEFAULT_CAPS_V2, ItemCapsV2, KEY_VOCAB_SIZE_V2, N_BLIND_KIND,
    N_BLIND_STATUS, OpponentView, SCALAR_DIM_V2, STATE_SPEC_VERSION, SetEncoderV2, collate,
    layout_fingerprint,
)
from .model_set import ResidualBlock, _CardEncoder, _ix, _masked_max, _masked_mean, _mlp

SET_NAMES_V2 = ("hand", "jokers", "consumables", "shelf", "packs", "blinds")

VALUE_CHECKPOINT_KIND = "mp/agent value_net"
VALUE_CHECKPOINT_VERSION = 1


# ════════════════════════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class ValueNetConfig:
    """Defaults land at 5.0M parameters (see VALUE_NOTES.md for the breakdown)."""
    d_item: int = 128           # item embedding width D (attention width)
    n_heads: int = 4
    ffn_mult: int = 2
    key_emb: int = 64           # game-key table width (KEY_VOCAB_V2 x key_emb)
    card_emb: int = 12          # per-field width of the card block table
    aux_emb: int = 8            # per-field width of the aux table (edition/rarity/kind/set/blind)
    trunk_width: int = 712      # W  (712 x 3 res blocks lands the total at 4,996,789)
    n_res_blocks: int = 3
    scalar_hidden: int = 384
    caps: dict = field(default_factory=lambda: DEFAULT_CAPS_V2.as_dict())
    scalar_dim: int = SCALAR_DIM_V2
    key_vocab: int = KEY_VOCAB_SIZE_V2
    # ── auxiliary prediction heads (Phase 5 rev 2, W-AUX; ev/AUX_NOTES.md) ──
    # {head name: output width}.  EMPTY BY DEFAULT: no modules, no parameters, no state_dict
    # entries, no RNG draws at construction — a net built without them is bit-identical to
    # the pre-W-AUX net, and an old checkpoint (whose `cfg` has no `aux_heads` key at all)
    # rebuilds with `{}` and loads `strict=True` unchanged.  Heads read the SHARED TRUNK
    # (`encode`) and are used only by `forward_with_aux`; `forward` / `p_win` — every
    # play-time path — never touch them.
    aux_heads: dict = field(default_factory=dict)
    aux_hidden: int = 0        # 0 = linear head; > 0 = one hidden layer of this width

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "ValueNetConfig":
        if not d:
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


# ════════════════════════════════════════════════════════════════════════════════
# The net
# ════════════════════════════════════════════════════════════════════════════════

def _aux_head(width: int, dim: int, hidden: int = 0) -> nn.Module:
    """One auxiliary head off the trunk: linear, or ONE hidden layer (brief §6b.2 caps it
    there — an aux head is a probe, not a second network).  Orthogonal init with the same
    conventions as `value_head`, so a fresh head starts near zero output."""
    if hidden and hidden > 0:
        layers = [nn.Linear(width, hidden), nn.ReLU(), nn.Linear(hidden, dim)]
        nn.init.orthogonal_(layers[0].weight, gain=np.sqrt(2))
        nn.init.constant_(layers[0].bias, 0)
        nn.init.orthogonal_(layers[2].weight, gain=1.0)
        nn.init.constant_(layers[2].bias, 0)
        return nn.Sequential(*layers)
    head = nn.Linear(width, dim)
    nn.init.orthogonal_(head.weight, gain=1.0)
    nn.init.constant_(head.bias, 0)
    return head


class SetValueNet(nn.Module):

    def __init__(self, cfg: ValueNetConfig | dict | None = None):
        super().__init__()
        cfg = cfg if isinstance(cfg, ValueNetConfig) else ValueNetConfig.from_dict(cfg)
        self.cfg = cfg
        self.caps = ItemCapsV2.from_dict(cfg.caps)
        D = cfg.d_item
        W = cfg.trunk_width

        # ── shared tables (three, as in SetPolicyValueNet: card block / aux / game keys) ──
        self.card = _CardEncoder(D, dim=cfg.card_emb)
        self.key_embed = nn.Embedding(cfg.key_vocab, cfg.key_emb, padding_idx=0)
        # aux field order: 0 edition, 1 rarity, 2 shelf kind, 3 pack set, 4 blind kind, 5 blind status
        self._aux_cards = (N_EDITION, N_RARITY, N_SHELF_KIND, N_PACK_SET, N_BLIND_KIND, N_BLIND_STATUS)
        self.aux_table = nn.Embedding(int(sum(self._aux_cards)), cfg.aux_emb)
        aux_off = np.concatenate([[0], np.cumsum(self._aux_cards)[:-1]])
        # Offset pairs in the COLUMN ORDER the encoder writes:
        #   joker_cat = [edition, rarity] | shelf_cat = [kind, edition] | pack_cat = [set, edition]
        #   blind_cat = [kind, status]
        for _name, _pair in (("_off_joker", (aux_off[0], aux_off[1])),
                             ("_off_shelf", (aux_off[2], aux_off[0])),
                             ("_off_pack", (aux_off[3], aux_off[0])),
                             ("_off_blind", (aux_off[4], aux_off[5]))):
            self.register_buffer(_name, torch.tensor(_pair, dtype=torch.long), persistent=False)

        aux2 = 2 * cfg.aux_emb
        k = cfg.key_emb
        self.hand_mlp = _mlp(self.card.cat_dim + CARD_NUM_DIM, D)
        self.joker_mlp = _mlp(k + aux2 + JOKER_NUM_DIM, D)
        self.cons_mlp = _mlp(k + CONS_NUM_DIM, D)
        self.shelf_mlp = _mlp(k + aux2 + self.card.cat_dim + SHELF_NUM_DIM, D)
        self.pack_mlp = _mlp(k + aux2 + self.card.cat_dim + PACK_NUM_DIM, D)
        self.blind_mlp = _mlp(2 * k + aux2 + BLIND_NUM_DIM, D)

        self.set_embed = nn.Embedding(len(SET_NAMES_V2) + 1, D)   # 0 = the global token

        # ── one masked attention block (pre-norm) ────────────────────────────
        self.attn_norm = nn.LayerNorm(D)
        self.attn = nn.MultiheadAttention(D, cfg.n_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(D)
        self.ffn = nn.Sequential(nn.Linear(D, cfg.ffn_mult * D), nn.ReLU(),
                                 nn.Linear(cfg.ffn_mult * D, D))

        # ── trunk ────────────────────────────────────────────────────────────
        self.scalar_proj = nn.Sequential(nn.Linear(cfg.scalar_dim, cfg.scalar_hidden), nn.ReLU())
        nn.init.orthogonal_(self.scalar_proj[0].weight, gain=np.sqrt(2))
        nn.init.constant_(self.scalar_proj[0].bias, 0)
        pooled = 2 * D * len(SET_NAMES_V2) + D            # + the global token
        self.trunk_in = nn.Sequential(nn.Linear(pooled + cfg.scalar_hidden, W), nn.ReLU())
        nn.init.orthogonal_(self.trunk_in[0].weight, gain=np.sqrt(2))
        nn.init.constant_(self.trunk_in[0].bias, 0)
        self.res_blocks = nn.Sequential(*[ResidualBlock(W) for _ in range(cfg.n_res_blocks)])

        self.value_head = nn.Linear(W, 1)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.constant_(self.value_head.bias, 0)

        # ── auxiliary heads (W-AUX) — LAST, so that with none configured the module
        # registration order, the parameter order and every init RNG draw above are
        # unchanged from the pre-W-AUX net.  An empty ModuleDict has no parameters and
        # contributes nothing to `state_dict()`.
        self.aux_heads = nn.ModuleDict()
        for name, dim in sorted((cfg.aux_heads or {}).items()):
            self.aux_heads[name] = _aux_head(W, int(dim), int(cfg.aux_hidden or 0))

    # ── introspection ────────────────────────────────────────────────────────

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_breakdown(self) -> dict[str, int]:
        """Parameter count per component (what VALUE_NOTES.md prints)."""
        groups = {
            "tables (card/aux/key/set)": [self.card, self.aux_table, self.key_embed, self.set_embed],
            "item MLPs (6 types)": [self.hand_mlp, self.joker_mlp, self.cons_mlp,
                                    self.shelf_mlp, self.pack_mlp, self.blind_mlp],
            "attention block + FFN": [self.attn_norm, self.attn, self.ffn_norm, self.ffn],
            "scalar_proj": [self.scalar_proj],
            "trunk_in": [self.trunk_in],
            "res_blocks": [self.res_blocks],
            "value_head": [self.value_head],
            "aux_heads": [self.aux_heads],          # 0 unless W-AUX heads are configured
        }
        out = {k: sum(p.numel() for m in ms for p in m.parameters()) for k, ms in groups.items()}
        assert sum(out.values()) == self.n_params()
        return out

    def describe(self) -> dict:
        return {"kind": "value_v1", **self.cfg.as_dict()}

    @classmethod
    def from_description(cls, desc: dict) -> "SetValueNet":
        return cls(ValueNetConfig.from_dict({k: v for k, v in desc.items() if k != "kind"}))

    # ── forward ──────────────────────────────────────────────────────────────

    def encode(self, batch: dict) -> torch.Tensor:
        """`batch`: the `SetEncoderV2` Obs dict with a leading batch dim, as tensors ->
        the trunk (B, W)."""
        caps = self.caps
        hand = self.hand_mlp(torch.cat(
            [self.card.embed_cat(batch["hand_cat"]), batch["hand_num"]], dim=-1))
        # ONE gather for every game key in the state, then split.
        nj, nc, ns, npk = caps.jokers, caps.consumables, caps.shelf, caps.packs
        nb = batch["blind_key"].shape[1]
        keys = self.key_embed(_ix(torch.cat(
            [batch["joker_key"], batch["cons_key"], batch["shelf_key"], batch["pack_key"],
             batch["blind_key"], batch["blind_tag"]], dim=1)))
        k_joker, k_cons, k_shelf, k_pack, k_blind, k_tag = torch.split(
            keys, [nj, nc, ns, npk, nb, nb], dim=1)

        joker = self.joker_mlp(torch.cat([
            k_joker,
            self.aux_table(_ix(batch["joker_cat"]) + self._off_joker).flatten(-2),
            batch["joker_num"]], dim=-1))
        cons = self.cons_mlp(torch.cat([k_cons, batch["cons_num"]], dim=-1))
        shelf = self.shelf_mlp(torch.cat([
            k_shelf,
            self.aux_table(_ix(batch["shelf_cat"]) + self._off_shelf).flatten(-2),
            self.card.embed_cat(batch["shelf_card"]),
            batch["shelf_num"]], dim=-1))
        pack = self.pack_mlp(torch.cat([
            k_pack,
            self.aux_table(_ix(batch["pack_cat"]) + self._off_pack).flatten(-2),
            self.card.embed_cat(batch["pack_card"]),
            batch["pack_num"]], dim=-1))
        blind = self.blind_mlp(torch.cat([
            k_blind, k_tag,
            self.aux_table(_ix(batch["blind_cat"]) + self._off_blind).flatten(-2),
            batch["blind_num"]], dim=-1))

        B = hand.shape[0]
        parts = [hand, joker, cons, shelf, pack, blind]
        for i, p in enumerate(parts):
            parts[i] = p + self.set_embed.weight[i + 1]
        glob = self.set_embed.weight[0].expand(B, 1, -1)
        seq = torch.cat([glob] + parts, dim=1)                       # (B, 1+S, D)

        masks = [batch["hand_mask"], batch["joker_mask"], batch["cons_mask"],
                 batch["shelf_mask"], batch["pack_mask"], batch["blind_mask"]]
        item_mask = torch.cat(masks, dim=1)                          # (B, S)
        ones = torch.ones(B, 1, device=hand.device, dtype=item_mask.dtype)
        pad = torch.cat([ones, item_mask], dim=1) <= 0               # global token never padded

        h = self.attn_norm(seq)
        attended, _ = self.attn(h, h, h, key_padding_mask=pad, need_weights=False)
        seq = seq + attended
        seq = seq + self.ffn(self.ffn_norm(seq))

        items = seq[:, 1:]
        pooled = [seq[:, 0]]
        at = 0
        for m in masks:
            w = m.shape[1]
            pooled.append(_masked_mean(items[:, at:at + w], m))
            pooled.append(_masked_max(items[:, at:at + w], m))
            at += w

        trunk = self.trunk_in(torch.cat(pooled + [self.scalar_proj(batch["scalars"])], dim=-1))
        return self.res_blocks(trunk)

    def forward(self, batch: dict) -> torch.Tensor:
        """(B,) LOGITS — pre-sigmoid. `logits.sigmoid()` is P(win).

        Unchanged by W-AUX: the auxiliary heads are never evaluated here, so play-time
        inference costs exactly what it did before whether or not a checkpoint carries
        them."""
        return self.value_head(self.encode(batch)).squeeze(-1)

    def p_win(self, batch: dict) -> torch.Tensor:
        return self.forward(batch).sigmoid()

    # ── auxiliary heads (W-AUX; trainer graph only) ──────────────────────────

    def aux_head_names(self) -> list[str]:
        return list(self.aux_heads.keys())

    def forward_with_aux(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """`(value logits (B,), {head name: (B, dim) raw output})` from ONE trunk pass.

        The trunk is shared, so aux gradients shape the same representation V reads — the
        whole point (brief §6b). With no heads configured this is `forward` plus an empty
        dict, but the TRAINER still calls `forward` directly in that case so the no-aux
        path stays bit-identical."""
        trunk = self.encode(batch)
        logits = self.value_head(trunk).squeeze(-1)
        return logits, {name: head(trunk) for name, head in self.aux_heads.items()}


# ════════════════════════════════════════════════════════════════════════════════
# Checkpoints
# ════════════════════════════════════════════════════════════════════════════════

def save_checkpoint(path: str | Path, net: SetValueNet, encoder: SetEncoderV2,
                    extra: Optional[dict] = None) -> Path:
    """Atomic `torch.save` of weights + everything needed to rebuild and to REFUSE a
    mismatched input layout (`STATE_SPEC_VERSION`, `layout_fingerprint()`)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": VALUE_CHECKPOINT_KIND,
        "version": VALUE_CHECKPOINT_VERSION,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "state_spec_version": STATE_SPEC_VERSION,
        "fingerprint": encoder.fingerprint,
        "encoder": encoder.describe(),
        "cfg": net.cfg.as_dict(),
        "net": net.describe(),
        "n_params": net.n_params(),
        "state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
        "extra": dict(extra or {}),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu"
                    ) -> tuple[SetValueNet, SetEncoderV2, dict]:
    """`(net, encoder, extra)`. Raises `ValueError` on a foreign file, an unsupported
    version, a `STATE_SPEC_VERSION` mismatch or a layout-fingerprint mismatch."""
    path = Path(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or ckpt.get("kind") != VALUE_CHECKPOINT_KIND:
        raise ValueError(f"{path} is not a {VALUE_CHECKPOINT_KIND} checkpoint")
    if ckpt.get("version") != VALUE_CHECKPOINT_VERSION:
        raise ValueError(f"{path}: checkpoint version {ckpt.get('version')}, "
                         f"this code reads {VALUE_CHECKPOINT_VERSION}")
    if ckpt.get("state_spec_version") != STATE_SPEC_VERSION:
        raise ValueError(f"{path}: STATE_SPEC_VERSION {ckpt.get('state_spec_version')} != "
                         f"{STATE_SPEC_VERSION}")
    desc = dict(ckpt["encoder"])
    want = ckpt.get("fingerprint")
    have = layout_fingerprint(desc.get("caps"))
    if want != have:
        raise ValueError(f"{path}: layout fingerprint mismatch — written against "
                         f"{str(want)[:12]}…, this code's STATE_SPEC v{STATE_SPEC_VERSION} "
                         f"layout is {have[:12]}…  (a FIELD changed: this net cannot be loaded)")
    desc["fingerprint"] = want
    encoder = SetEncoderV2.from_description(desc)
    net = SetValueNet(ValueNetConfig.from_dict(ckpt["cfg"]))
    net.load_state_dict(ckpt["state_dict"], strict=True)
    net.to(torch.device(device))
    net.eval()
    return net, encoder, dict(ckpt.get("extra") or {})


# ════════════════════════════════════════════════════════════════════════════════
# Value functions
# ════════════════════════════════════════════════════════════════════════════════

def make_values_many(net: SetValueNet, encoder: SetEncoderV2, device: str | torch.device = "cpu",
                     chunk: int = 512
                     ) -> Callable[[Sequence[tuple]], np.ndarray]:
    """Batched evaluator: `fn([(game, opp_or_None), ...]) -> np.ndarray (N,)` of P(win),
    float32. This is the one `EVPlayer` / the label generator should use."""
    dev = torch.device(device)
    net = net.to(dev)

    def fn(items: Sequence[tuple]) -> np.ndarray:
        if not items:
            return np.zeros(0, dtype=np.float32)
        was_training = net.training
        net.eval()
        out = []
        try:
            with torch.no_grad():
                for at in range(0, len(items), chunk):
                    part = items[at:at + chunk]
                    obs = [encoder(g, o) for g, o in part]
                    logits = net(collate(obs, dev))
                    out.append(logits.sigmoid().float().cpu().numpy())
        finally:
            if was_training:
                net.train()
        return np.concatenate(out).astype(np.float32, copy=False)

    return fn


def make_value_fn(net: SetValueNet, encoder: SetEncoderV2, device: str | torch.device = "cpu"
                  ) -> Callable[..., float]:
    """Single-state convenience: `fn(game, opp=None) -> float` P(win)."""
    many = make_values_many(net, encoder, device)

    def fn(game, opp: Optional[OpponentView] = None) -> float:
        return float(many([(game, opp)])[0])

    return fn


__all__ = [
    "SetValueNet", "ValueNetConfig", "SET_NAMES_V2",
    "save_checkpoint", "load_checkpoint", "make_value_fn", "make_values_many",
    "VALUE_CHECKPOINT_KIND", "VALUE_CHECKPOINT_VERSION",
]
