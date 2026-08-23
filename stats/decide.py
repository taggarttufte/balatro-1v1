"""
decide.py -- the decision table: one Row per legal SHOP / BOOSTER_OPEN action, with
P(hit), true cost (incl. interest loss), urgency and net EV (Phase 5 rev 2, W4;
mp/docs/PHASE5_BRIEF_2026-08.md §2 interface, §3 gate 3).

``decision_table(game) -> list[Row]`` is the entire public surface W3's ``EVPlayer``
(``mp/ev/player.py``) and W6's advisor need. Everything is read-only on ``game`` -- every
number is either a pure read of scalar state (``economy.py``), a call into
``mp.rng.generate.get_current_pool`` with an explicit (non-rolled) rarity (``hit.py``, zero
RNG consumed), or a dry run through a PRIVATE clone of ``game.run_state.rng``
(``hit.sample_hand_scores`` / ``hit.joker_hit_value``, same pattern as
``card_selection.HypotheticalScorer``). ``tests/test_side_effect_free.py`` pins
``game.state_signature()`` before/after a ``decision_table`` call.

Common unit -- $-equivalent (brief §2)
---------------------------------------
Every ``hit_value`` / ``cost`` / ``true_cost`` / ``net_ev`` is in dollars. Two conversions
feed everything else:

  * A joker's raw score uplift (chips*mult delta from a dry run) is divided by the NEXT
    blind's chip target and multiplied by ``dollars_per_score_ratio`` -- "how many dollars is
    moving 100% of the way to clearing the next blind, in one hand, worth" (``hit.joker_hit_
    value``). Dividing by the (ante-scaled) chip target IS the ante schedule the brief asks
    for: no separate per-ante $/point table is needed because the target already grows with
    ante the same way score needs to.
  * A pool member's (reroll / pack) value never gets a dry run (would mean 150+ dry runs per
    shop visit) -- ``hit.pool_dollar_value`` is ``synergy.estimate_joker_strength`` (1..10) x
    a coherence multiplier x a flat $/point rate. It is only used to (a) classify "hit" by a
    $ threshold and (b) average over the hits that survive -- never to price a KNOWN card
    (those get the precise dry run).

Consumables / cards / vouchers use small flat tables (``hit.tarot_value`` /
``planet_value`` / ``spectral_value`` / ``voucher_value`` / ``card_value``) -- documented,
not derived.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

import _bootstrap  # noqa: F401
from balatro_sim.game import State
from balatro_sim.shop import effective_price
from balatro_sim.jokers.base import joker_sell_value
from balatro_sim import game_keys as _gk

import economy as econ
import hit as hitmod
import urgency as urg

# ══════════════════════════════════════════════════════════════════ config

@dataclass(frozen=True)
class StatsConfig:
    hit: hitmod.StatsConfig = hitmod.DEFAULT
    urgency_gain: float = 1.0           # net_ev's hit_value multiplier = 1 + urgency_gain * urgency
    urgency_hand_samples: int = 5
    horizon_rounds: Optional[int] = None


DEFAULT = StatsConfig()


@dataclass
class Row:
    action: dict
    kind: str
    label: str
    p_hit: float
    hit_value: float
    cost: float
    interest_loss: float
    true_cost: float
    urgency: float
    net_ev: float
    details: dict = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════ decision_table

def decision_table(game, *, horizon_rounds: Optional[int] = None,
                   cfg: StatsConfig = DEFAULT) -> list[Row]:
    """One ``Row`` per ``game.legal_actions()`` entry, sorted by ``net_ev`` descending.
    Empty outside ``SHOP`` / ``BOOSTER_OPEN`` (nothing for those states to compute)."""
    if game.state not in (State.SHOP, State.BOOSTER_OPEN):
        return []
    if horizon_rounds is not None:
        cfg = replace(cfg, horizon_rounds=horizon_rounds)

    ctx = _Context.build(game, cfg)
    rows: list[Row] = []
    for action in game.legal_actions():
        row = _row_for_action(game, action, ctx, cfg)
        if row is not None:
            rows.append(row)
    rows.sort(key=lambda r: -r.net_ev)
    return rows


@dataclass
class _Context:
    """Everything computed ONCE per ``decision_table`` call and shared across rows: the
    urgency read, the shop's slot distribution (for the single ``reroll`` row) and a single
    ``run_state`` clone (reused by every ``buy_pack`` row's ``pack_p_hit`` instead of
    re-cloning per pack) -- the throughput lever that keeps this under the 50ms gate with two
    packs + a reroll on the shelf."""
    urgency: urg.UrgencyResult
    econ: econ.EconomySnapshot
    rs_clone: object
    reroll_slices: Optional[list] = None
    value_cache: dict = field(default_factory=dict)

    @classmethod
    def build(cls, game, cfg: StatsConfig) -> "_Context":
        u = urg.compute(game, cfg.urgency_hand_samples)
        e = econ.EconomySnapshot.build(game, cfg.horizon_rounds)
        rs_clone = game.run_state.clone()
        return cls(urgency=u, econ=e, rs_clone=rs_clone)

    def joker_value(self, game, key: str, edition: str, cfg: hitmod.StatsConfig) -> float:
        ck = (key, edition)
        if ck not in self.value_cache:
            self.value_cache[ck] = hitmod.joker_hit_value(game, key, edition, cfg)
        return self.value_cache[ck]


def _urgency_mult(ctx: _Context, cfg: StatsConfig) -> float:
    return 1.0 + cfg.urgency_gain * ctx.urgency.urgency


def _econ(game, cfg: StatsConfig, cost: float) -> tuple[float, float]:
    """(interest_loss, true_cost) for spending ``cost`` now."""
    il = econ.interest_loss(game, cost, cfg.horizon_rounds)
    return il, cost + il


def _make_row(action: dict, kind: str, label: str, p_hit: float, hit_value: float,
             cost: float, interest_loss_: float, urgency: float, ev_mult: float,
             details: dict) -> Row:
    true_cost = cost + interest_loss_
    net_ev = p_hit * hit_value * ev_mult - true_cost
    return Row(action=action, kind=kind, label=label, p_hit=p_hit, hit_value=hit_value,
              cost=cost, interest_loss=interest_loss_, true_cost=true_cost,
              urgency=urgency, net_ev=net_ev, details=details)


# ══════════════════════════════════════════════════════════════════ per-action-type rows

def _row_for_action(game, action: dict, ctx: _Context, cfg: StatsConfig) -> Optional[Row]:
    t = action["type"]
    mult = _urgency_mult(ctx, cfg)
    u = ctx.urgency.urgency

    if t == "leave_shop":
        return _make_row(action, "leave", "leave shop", 1.0, 0.0, 0.0, 0.0, u, mult,
                         {"reason": "baseline: take no further shop action"})

    if t == "skip_booster":
        return _make_row(action, "skip_pack", "skip pack", 1.0, 0.0, 0.0, 0.0, u, mult,
                         {"reason": "baseline: take nothing from the open pack"})

    if t == "sell_joker":
        return _row_sell(game, action, ctx, cfg, mult, u)

    if t == "reroll":
        return _row_reroll(game, action, ctx, cfg, mult, u)

    if t == "buy":
        return _row_buy(game, action, ctx, cfg, mult, u)

    if t == "use_consumable":
        return _row_use_consumable(game, action, ctx, cfg, mult, u)

    if t == "pick_booster":
        return _row_pick_booster(game, action, ctx, cfg, mult, u)

    return None   # unrecognised action type in this state — decision_table stays silent, not wrong


def _row_sell(game, action, ctx, cfg, mult, u) -> Row:
    idx = action["joker_idx"]
    j = game.jokers[idx]
    sell_dollars = float(joker_sell_value(j))
    ongoing = ctx.joker_value(game, j.key, j.edition, cfg.hit)
    label = f"sell {_gk.JOKER_NAME.get(j.key, j.key)} (+${sell_dollars:.0f})"
    details = {"joker_idx": idx, "key": j.key, "sell_dollars": sell_dollars, "ongoing_value": ongoing}
    # cost is negative (selling is a cash INFLOW); hit_value is negative (you give up the
    # joker's ongoing value); net_ev = sell_dollars - ongoing (positive = sell is correct).
    return _make_row(action, "sell", label, 1.0, -ongoing, -sell_dollars, 0.0, u, mult, details)


def _row_reroll(game, action, ctx, cfg, mult, u) -> Row:
    if ctx.reroll_slices is None:
        ctx.reroll_slices = hitmod.shop_slot_distribution(game, cfg.hit, rs=ctx.rs_clone)
    p_hit, mean_value, details = hitmod.reroll_p_hit(game, cfg.hit, slices=ctx.reroll_slices)
    cost = float(ctx.econ.reroll_cost_now)
    il, _true = _econ(game, cfg, cost)
    label = f"reroll (${cost:.0f}, P(hit)={p_hit:.0%})"
    return _make_row(action, "reroll", label, p_hit, mean_value, cost, il, u, mult, details)


def _row_buy(game, action, ctx, cfg, mult, u) -> Row:
    idx = action["item_idx"]
    item = game.current_shop[idx]
    price = float(effective_price(game, item))
    il, _true = _econ(game, cfg, price)
    details = {"item_idx": idx, "key": item.key, "kind": item.kind, "edition": item.edition}

    if item.kind == "joker":
        value = ctx.joker_value(game, item.key, item.edition, cfg.hit)
        kind = "buy_joker"
        rarity = _gk.JOKER_RARITY.get(item.key, "?")
        label = f"buy {item.name} ({rarity}, ${price:.0f})"
    elif item.kind == "voucher":
        value = hitmod.voucher_value(item.key, cfg.hit)
        kind = "buy_voucher"
        label = f"buy voucher {item.name} (${price:.0f})"
    elif item.kind == "booster":
        return _row_buy_pack(game, action, item, idx, price, il, ctx, cfg, mult, u)
    elif item.kind in ("planet", "tarot", "spectral"):
        value = _consumable_flat_value(game, item.kind, item.key, cfg.hit)
        kind = "buy_consumable"
        label = f"buy {item.name} (${price:.0f})"
    elif item.kind == "card":
        value = hitmod.card_value(item, cfg.hit)
        kind = "buy_card"
        label = f"buy {item.name} (${price:.0f})"
    else:   # pragma: no cover — no other ShopItem.kind exists (shop.py SHELF_KINDS + voucher/booster)
        value = 0.0
        kind = "buy_card"
        label = f"buy {item.name} (${price:.0f})"

    return _make_row(action, kind, label, 1.0, value, price, il, u, mult, details)


def _row_buy_pack(game, action, item, idx, price, il, ctx, cfg, mult, u) -> Row:
    b = _gk.BOOSTER_TYPES.get(item.key, {})
    pack_kind = b.get("kind", "")
    size = int(b.get("cards", 0))
    p_hit, mean_value, pdetails = hitmod.pack_p_hit(game, pack_kind, size, cfg.hit, rs=ctx.rs_clone)
    details = {"item_idx": idx, "key": item.key, "pack_kind": pack_kind, "size": size, **pdetails}
    label = f"buy {item.name} (${price:.0f}, P(hit)={p_hit:.0%})"
    return _make_row(action, "buy_pack", label, p_hit, mean_value, price, il, u, mult, details)


_CONSUMABLE_NAME_TABLES = None   # lazy: {"Tarot": TAROT_NAME, "Planet": PLANET_NAME, "Spectral": SPECTRAL_NAME}


def _row_use_consumable(game, action, ctx, cfg, mult, u) -> Row:
    global _CONSUMABLE_NAME_TABLES
    if _CONSUMABLE_NAME_TABLES is None:
        _CONSUMABLE_NAME_TABLES = {"Tarot": _gk.TAROT_NAME, "Planet": _gk.PLANET_NAME,
                                   "Spectral": _gk.SPECTRAL_NAME}
    idx = action["consumable_idx"]
    key = game.consumable_hand[idx] if idx < len(game.consumable_hand) else None
    value, name = 0.0, (key or "?")
    if key is not None:
        gset = _gk.CONSUMABLE_SET.get(key, "")     # "Tarot" | "Planet" | "Spectral"
        value = _consumable_flat_value(game, gset, key, cfg.hit)
        name = _CONSUMABLE_NAME_TABLES.get(gset, {}).get(key, key)
    details = {"consumable_idx": idx, "key": key}
    label = f"use {name}"
    # Free action (already owned, not sold back): cost 0, so true_cost 0 -- net_ev is pure upside.
    return _make_row(action, "use_consumable", label, 1.0, value, 0.0, 0.0, u, mult, details)


def _row_pick_booster(game, action, ctx, cfg, mult, u) -> Row:
    indices = action["indices"]
    total = 0.0
    names = []
    for i in indices:
        choice = game.booster_choices[i]
        names.append(choice.name)
        if choice.is_joker:
            total += ctx.joker_value(game, choice.key, choice.edition, cfg.hit)
        elif choice.is_consumable:
            total += _consumable_flat_value(game, choice.set.lower(), choice.key, cfg.hit)
        elif choice.is_playing_card:
            total += hitmod.card_value(choice, cfg.hit)
    label = f"pick {' + '.join(names)}"
    details = {"indices": list(indices)}
    # Pack contents are fully revealed at this decision point (no chance left): deterministic.
    return _make_row(action, "pick", label, 1.0, total, 0.0, 0.0, u, mult, details)


def _consumable_flat_value(game, kind: str, key: str, hcfg: hitmod.StatsConfig) -> float:
    kind = kind.lower()
    if kind == "tarot":
        return hitmod.tarot_value(key, hcfg)
    if kind == "planet":
        return hitmod.planet_value(game, key, hcfg)
    if kind == "spectral":
        return hitmod.spectral_value(key, hcfg)
    return 0.0


# ══════════════════════════════════════════════════════════════════ pretty-print

def explain(rows: list[Row]) -> str:
    """A fixed-width table for the advisor CLI (W6). No dependency beyond the stdlib."""
    if not rows:
        return "(no shop/pack rows -- game is not in SHOP or BOOSTER_OPEN)"
    rows = sorted(rows, key=lambda r: -r.net_ev)
    headers = ("kind", "label", "p_hit", "hit_val", "cost", "int_loss", "true_cost", "urgency", "net_ev")
    widths = [len(h) for h in headers]
    fmt_rows = []
    for r in rows:
        cells = (r.kind, r.label, f"{r.p_hit:.2f}", f"{r.hit_value:.1f}", f"{r.cost:.0f}",
                f"{r.interest_loss:.1f}", f"{r.true_cost:.1f}", f"{r.urgency:.2f}", f"{r.net_ev:+.1f}")
        fmt_rows.append(cells)
        widths = [max(w, len(c)) for w, c in zip(widths, cells)]
    lines = []
    lines.append("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    lines.append("  ".join("-" * w for w in widths))
    for cells in fmt_rows:
        lines.append("  ".join(c.ljust(w) for c, w in zip(cells, widths)))
    return "\n".join(lines)


__all__ = ["StatsConfig", "DEFAULT", "Row", "decision_table", "explain"]
