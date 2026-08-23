"""
player.py — ``EVPlayer``: the analytic EV player (Phase 5 rev 2, W3).

The ``Player`` protocol used by the tournament runner, the eval harness, the replay tools
and W5's label rollouts: ``act(game) -> dict`` (always one of ``game.legal_actions()``, or
``no_action`` when there are none), ``reset()``, and ``explain(game)`` for W6's advisor.

Every state is covered:

* ``SELECTING_HAND`` — ``hand.rank_hand_actions`` (fast or full budget; hand.py).
* ``BLIND_SELECT`` — play, or skip Small/Big for a premium tag when the build clears the
  following blind with high probability (rule) / argmax V over sampled worlds (value_fn).
* ``ROUND_EVAL`` — ``advance``.
* ``SHOP`` / ``BOOSTER_OPEN`` — three tiers, in order of preference:
    1. ``stats`` (W4's ``decision_table(game) -> list[Row]``): the max ``net_ev`` row with
       positive EV, else leave / skip;
    2. ``value_fn``: argmax V over the stepped clones, chance actions (reroll, pack
       open, pack pick) averaged over ``n_worlds`` sampled worlds;
    3. built-in rules on the analytic *build proxy* (below).
* ``PVP_WAIT`` / ``GAME_OVER`` — ``no_action``.

The build proxy (the bootstrap valuation until V exists)::

    proxy(g) = P(clear next blind | deck, board) + lam_build * log1p(strength) + lam_money * money(g)

``P(clear next blind)`` = the blind model's tail from a fresh position (``hand.BlindModel``)
with the need divided by the board's exact/cheap ``ratio``; ``strength`` = the model's mean
fresh-hand score times the ratio; ``money`` = dollars + two rounds of the interest they
earn.  Jokers are bought when they raise the proxy net of their price; packs when they
are affordable after the ante's interest floor; vouchers likewise; the shop is rerolled
at most once per visit, never below the interest floor; a joker is sold only to make
room for a shelf joker that is worth more than the weakest owned one.

Determinism: with ``epsilon == 0`` every decision is a function of ``(seed, observable
state)`` — the world sampler is seeded from both (``sampling.world_rng``) and no per-visit
memory is kept (rerolls done are read off ``game.reroll_cost``).  ``epsilon > 0`` mixes
in uniformly random legal actions (W5's self-play diversity) from the same seeded stream.

Side-effect freedom: ``act`` only reads the live game; every evaluation is on a clone.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Optional

import _bootstrap  # noqa: F401
from _bootstrap import State
from balatro_sim.constants import blind_base_chips, INTEREST_RATE, INTEREST_CAP
from balatro_sim.consumables import PLANET_HAND
from balatro_sim.shop import effective_price

import hand as _hand
from hand import HandConfig, DEFAULT_HAND_CONFIG, blind_model_for, board_ratio
from sampling import sample_world, world_rng

__all__ = ["EVPlayer", "PlayerConfig", "DEFAULT_PLAYER_CONFIG", "build_proxy", "PREMIUM_TAGS"]

#: Tags worth skipping a Small/Big blind for (rule tier), by the value of what they give.
PREMIUM_TAGS = frozenset({
    "tag_rare", "tag_uncommon", "tag_buffoon", "tag_foil", "tag_holo", "tag_polychrome",
    "tag_negative", "tag_top_up", "tag_charm", "tag_meteor", "tag_ethereal", "tag_orbital",
    "tag_investment", "tag_coupon",
})

#: Pack type keys in order of preference for the rule tier.
_PACK_PREF = ("buffoon", "celestial", "arcana", "standard", "spectral")

#: Vouchers the rule tier never buys.
_SKIP_VOUCHERS = frozenset({"v_blank", "v_antimatter_placeholder"})


@dataclass(frozen=True)
class PlayerConfig:
    lam_build: float = 0.30        # weight of log1p(strength) in the proxy
    lam_money: float = 0.010       # weight of one $ in the proxy (P(clear) units)
    interest_weight: float = 0.4   # extra worth of a $ that still earns interest (2 rounds)
    skip_min_p_clear: float = 0.85 # skip a blind for a premium tag only above this
    skip_tags: bool = True
    skip_min_ante: int = 2         # never skip at ante 1 (the first shops decide the run)
    pack_floor_ante_step: int = 5  # interest floor for packs/vouchers = min(25, step*ante)
    max_rerolls_per_visit: int = 1
    reroll_floor: int = 25         # never reroll below this many $ after the reroll
    buy_margin: float = 0.0        # proxy gain a purchase must exceed
    sell_margin: float = 0.05      # gain a shelf joker must have over the weakest owned one
    n_worlds: int = 4              # value_fn tier: worlds per chance action
    max_shop_candidates: int = 12  # value_fn tier: actions evaluated per shop state


DEFAULT_PLAYER_CONFIG = PlayerConfig()


# ═══════════════════════════════════════════════════════════════════ the build proxy

def _next_blind_target(game) -> float:
    """Chip target of the NEXT blind to be played from this (non-hand) state."""
    st = game.state
    ante, idx = game.ante, game.blind_idx
    if st in (State.SHOP, State.BOOSTER_OPEN, State.ROUND_EVAL):
        idx += 1
        if idx >= 3:
            idx = 0
            ante += 1
    base = blind_base_chips(ante, idx, getattr(game, "blind_scaling", 1)) * getattr(game, "ante_scaling", 1)
    if idx == 2:
        if game.mlb and ante >= getattr(game, "pvp_start_round", 2):
            return float(base)          # Nemesis: the opponent's score; use the boss-slot base
        from balatro_sim.game import BOSS_CHIP_MULT  # the boss multiplier table
        key = game.boss_blind if ante == game._boss_blind_ante else ""
        return float(base * BOSS_CHIP_MULT.get(key, 1.0))
    return float(base)


def _money_value(game, cfg: PlayerConfig) -> float:
    d = float(game.dollars)
    if getattr(game, "no_interest", False):
        return d
    interest = min(max(0.0, d) // INTEREST_RATE, float(getattr(game, "interest_cap", INTEREST_CAP)))
    return d + 2.0 * cfg.interest_weight * interest


def _auto_use_planets(clone) -> None:
    """Apply held planets on a clone (the rule tier always uses them right away, so a
    proxy measured on a clone should see the levels)."""
    changed = True
    while changed:
        changed = False
        for i, key in enumerate(list(clone.consumable_hand)):
            if key in PLANET_HAND:
                before = len(clone.consumable_hand)
                from balatro_sim.consumables import apply_planet
                if apply_planet(clone, key):
                    clone.consumable_hand.pop(i)
                    changed = True
                    break
                if len(clone.consumable_hand) == before:
                    continue


def build_proxy(game, cfg: PlayerConfig = DEFAULT_PLAYER_CONFIG,
                hcfg: HandConfig = DEFAULT_HAND_CONFIG, *, auto_planets: bool = True) -> dict:
    """The analytic valuation of a non-hand state (see the module docstring).  Returns a
    dict with ``value`` and its parts.  Read-only (works on a clone when planets are held)."""
    g = game
    if auto_planets and any(k in PLANET_HAND for k in g.consumable_hand):
        g = game.clone()
        _auto_use_planets(g)
    model = blind_model_for(g, hcfg)
    ratio = board_ratio(g, cfg=hcfg)
    target = _next_blind_target(g)
    hands = int(g.base_hands) + (3 if any(j.key == "j_burglar" for j in g.jokers) else 0)
    discards = int(g.base_discards)
    p_clear = model.p_clear(target / max(ratio, 1e-6), hands, discards)
    strength = model.mean0 * ratio
    money = _money_value(g, cfg)
    value = p_clear + cfg.lam_build * math.log1p(strength) + cfg.lam_money * money
    return {"value": value, "p_clear": p_clear, "strength": strength, "ratio": ratio,
            "money": money, "target": target}


# ═══════════════════════════════════════════════════════════════════════ the player

class EVPlayer:
    """See the module docstring.  ``budget`` = "fast" | "full" for hand decisions."""

    def __init__(self, value_fn: Optional[Callable] = None, *, stats=None, budget: str = "fast",
                 seed: int = 0, epsilon: float = 0.0, no_action: Optional[dict] = None,
                 cfg: PlayerConfig = DEFAULT_PLAYER_CONFIG, hand_cfg: HandConfig = DEFAULT_HAND_CONFIG,
                 n_worlds: Optional[int] = None, top_k: Optional[int] = None, name: str = "ev"):
        if budget not in ("fast", "full"):
            raise ValueError(f"budget must be 'fast' or 'full', got {budget!r}")
        self.value_fn = value_fn
        self.stats = stats
        self.budget = budget
        self.seed = int(seed)
        self.epsilon = float(epsilon)
        self.no_action = dict(no_action) if no_action is not None else {"type": "advance"}
        self.cfg = cfg
        self.hand_cfg = hand_cfg
        # the shop / pack proxies rebuild the blind model per candidate (a pick changes the
        # deck or the levels): a lighter simulation is plenty for a purchase comparison
        self.proxy_cfg = replace(hand_cfg, model_samples=min(hand_cfg.model_samples, 96))
        self.n_worlds = n_worlds
        self.top_k = top_k
        self.name = name
        self._last_explain: list = []

    # ── protocol ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        self._last_explain = []

    def act(self, game) -> dict:
        legal = game.legal_actions()
        if not legal:
            return dict(self.no_action)
        if self.epsilon > 0.0:
            rng = world_rng(self.seed, game, salt=0xE95)
            if rng.random() < self.epsilon:
                return dict(legal[rng.randrange(len(legal))])
        ranked = self._rank(game, legal, explain=False)
        if ranked:
            return dict(ranked[0][0])
        return dict(legal[0])

    def explain(self, game) -> list:
        """Ranked ``[(action, ev, reason)]`` for the state (W6's advisor prints this)."""
        legal = game.legal_actions()
        if not legal:
            return [(dict(self.no_action), 0.0, "no legal action (waiting / over)")]
        return [(a, ev, r) for a, ev, r in self._rank(game, legal)]

    # ── dispatch ────────────────────────────────────────────────────────────

    def _rank(self, game, legal: list, explain: bool = True) -> list:
        s = game.state
        if s == State.SELECTING_HAND:
            return self._rank_hand(game, legal, explain=explain)
        if s == State.ROUND_EVAL:
            return [({"type": "advance"}, 0.0, "cash out")]
        if s == State.BLIND_SELECT:
            return self._rank_blind_select(game, legal)
        if s in (State.SHOP, State.BOOSTER_OPEN):
            if self.stats is not None:
                r = self._rank_with_stats(game, legal)
                if r:
                    return r
            if self.value_fn is not None:
                return self._rank_with_value(game, legal)
            if s == State.SHOP:
                return self._rank_shop_rules(game, legal)
            return self._rank_booster_rules(game, legal)
        return [(legal[0], 0.0, "fallback: first legal action")]

    # ── hands ───────────────────────────────────────────────────────────────

    def _rank_hand(self, game, legal: list, explain: bool = True) -> list:
        kw = dict(budget=self.budget, cfg=self.hand_cfg, legal=legal)
        if self.budget == "full":
            kw["value_fn"] = self.value_fn
            kw["rng"] = world_rng(self.seed, game, salt=0xF0)
            if self.n_worlds is not None:
                kw["n_worlds"] = self.n_worlds
            if self.top_k is not None:
                kw["top_k"] = self.top_k
        ranked = _hand.rank_hand_actions(game, **kw)
        if not explain:
            return [(a, ev, "") for a, ev in ranked]
        return [(a, ev, self._hand_reason(game, a, ev)) for a, ev in ranked]

    @staticmethod
    def _hand_reason(game, a: dict, ev: float) -> str:
        t = a.get("type")
        if t == "play":
            cards = [game.hand[i] for i in a.get("cards", []) if i < len(game.hand)]
            try:
                ht, _ = _hand.evaluate_hand(cards, **game.hand_eval_flags())
            except Exception:       # noqa: BLE001
                ht = "?"
            return f"play {ht} {' '.join(map(str, cards))} (EV {ev:.3f})"
        if t == "discard":
            cards = [str(game.hand[i]) for i in a.get("cards", []) if i < len(game.hand)]
            keep = [str(c) for i, c in enumerate(game.hand) if i not in set(a.get("cards", []))]
            return f"discard {' '.join(cards)}, keep {' '.join(keep)} (EV {ev:.3f})"
        if t == "use_consumable":
            idx = a.get("consumable_idx", 0)
            key = game.consumable_hand[idx] if idx < len(game.consumable_hand) else "?"
            return f"use {key} on {a.get('target_cards', [])} (EV {ev:.3f})"
        return f"{t} (EV {ev:.3f})"

    # ── blind select ────────────────────────────────────────────────────────

    def _rank_blind_select(self, game, legal: list) -> list:
        keys = [a["type"] for a in legal]
        if self.value_fn is not None:
            return self._rank_with_value(game, legal)
        px = build_proxy(game, self.cfg, self.hand_cfg)
        out = [({"type": "play_blind"}, px["p_clear"],
                f"play {game.current_blind.kind} (P(clear) {px['p_clear']:.2f}, target {px['target']:.0f})")]
        if "skip_blind" in keys and self.cfg.skip_tags and game.ante >= self.cfg.skip_min_ante:
            tag = game.blind_tags.get(game.current_blind.kind, "")
            # the blind faced after the skip
            nxt_idx = game.blind_idx + 1
            nxt_target = blind_base_chips(game.ante, nxt_idx, getattr(game, "blind_scaling", 1)) * getattr(game, "ante_scaling", 1)
            if nxt_idx == 2:
                if game.mlb and game.ante >= getattr(game, "pvp_start_round", 2):
                    p_next = 0.0     # never skip into a Nemesis
                else:
                    from balatro_sim.game import BOSS_CHIP_MULT
                    nxt_target *= BOSS_CHIP_MULT.get(game.boss_blind or "", 1.0)
                    p_next = blind_model_for(game, self.hand_cfg).p_clear(
                        nxt_target / max(px["ratio"], 1e-6), int(game.base_hands), int(game.base_discards))
            else:
                p_next = blind_model_for(game, self.hand_cfg).p_clear(
                    nxt_target / max(px["ratio"], 1e-6), int(game.base_hands), int(game.base_discards))
            premium = tag in PREMIUM_TAGS
            ev = (p_next + 0.02) if (premium and p_next >= self.cfg.skip_min_p_clear) else (p_next - 1.0)
            out.append(({"type": "skip_blind"}, ev,
                        f"skip for {tag or 'no tag'} (P(clear next) {p_next:.2f}{', premium' if premium else ''})"))
        if "reroll_boss" in keys:
            out.append(({"type": "reroll_boss"}, -2.0, "reroll boss (rule: never)"))
        out.sort(key=lambda x: -x[1])
        return out

    # ── W4 stats tier ───────────────────────────────────────────────────────

    def _rank_with_stats(self, game, legal: list) -> list:
        try:
            rows = self.stats.decision_table(game)
        except Exception:           # noqa: BLE001 - W4 not ready: fall through to the rules
            return []
        legal_keys = {_hand._action_sort_key(a) for a in legal}
        out = []
        for row in rows or []:
            action = getattr(row, "action", None) if not isinstance(row, dict) else row.get("action")
            net = getattr(row, "net_ev", None) if not isinstance(row, dict) else row.get("net_ev")
            if not isinstance(action, dict) or net is None:
                continue
            if _hand._action_sort_key(action) not in legal_keys:
                continue
            label = getattr(row, "label", None) if not isinstance(row, dict) else row.get("label")
            out.append((dict(action), float(net), f"stats: {label or action.get('type')} net_ev {float(net):.3f}"))
        if not out:
            return []
        out.sort(key=lambda x: -x[1])
        leave = {"type": "leave_shop"} if game.state == State.SHOP else {"type": "skip_booster"}
        if out[0][1] <= 0.0:
            return [(leave, 0.0, "stats: nothing with positive net EV")] + out
        return out

    # ── V tier ──────────────────────────────────────────────────────────────

    def _rank_with_value(self, game, legal: list) -> list:
        cands = self._shop_candidates(game, legal)
        rng = world_rng(self.seed, game, salt=0x5A)
        nw = self.n_worlds if self.n_worlds is not None else self.cfg.n_worlds
        worlds = None
        out = []
        for a in cands:
            chance = a["type"] in ("reroll", "skip_blind", "play_blind") or (
                a["type"] == "buy" and a.get("item_idx", 0) < len(game.current_shop)
                and game.current_shop[a["item_idx"]].kind == "booster")
            if chance:
                if worlds is None:
                    worlds = [sample_world(game, rng) for _ in range(max(1, nw))]
                vals = []
                for w in worlds:
                    c = w.clone()
                    c.step(a)
                    vals.append(self._v(c))
                ev = sum(vals) / len(vals)
            else:
                c = game.clone()
                c.step(a)
                ev = self._v(c)
            out.append((a, ev, f"V after {a['type']} = {ev:.3f}"))
        out.sort(key=lambda x: (-x[1], _hand._action_sort_key(x[0])))
        return out

    def _v(self, clone) -> float:
        try:
            return float(self.value_fn(clone))
        except Exception:           # noqa: BLE001
            return 0.0

    def _shop_candidates(self, game, legal: list) -> list:
        """Legal actions worth evaluating (caps the enumeration)."""
        out = []
        for a in legal:
            if a["type"] == "use_consumable" and a.get("target_cards"):
                continue
            out.append(a)
        return out[: self.cfg.max_shop_candidates]

    # ── rule tier: shop ─────────────────────────────────────────────────────

    def _interest_floor(self, game) -> int:
        return min(25, self.cfg.pack_floor_ante_step * max(0, game.ante))

    def _rank_shop_rules(self, game, legal: list) -> list:
        cfg = self.cfg
        legal_keys = {_hand._action_sort_key(a): a for a in legal}
        out = []
        # 1. consumables usable here (planets first, then any untargeted use that succeeds)
        for a in legal:
            if a["type"] != "use_consumable" or a.get("target_cards"):
                continue
            key = game.consumable_hand[a["consumable_idx"]] if a["consumable_idx"] < len(game.consumable_hand) else ""
            c = game.clone()
            n0 = len(c.consumable_hand)
            c.step(a)
            if len(c.consumable_hand) < n0 and c.state == State.SHOP:
                bonus = 5.0 if key in PLANET_HAND else 4.0
                out.append((a, bonus, f"use {key} now"))
        base = build_proxy(game, cfg, self.proxy_cfg)
        floor = self._interest_floor(game)
        p_clear = base["p_clear"]
        # 2. shelf purchases
        sells = []
        for a in legal:
            if a["type"] != "buy":
                continue
            item = game.current_shop[a["item_idx"]]
            price = effective_price(game, item)
            if item.kind == "joker":
                c = game.clone()
                c.step(a)
                if len(c.jokers) <= len(game.jokers) and item.edition != "Negative":
                    continue
                px = build_proxy(c, cfg, self.proxy_cfg)
                gain = px["value"] - base["value"]
                out.append((a, gain, f"buy {item.name} ${price}: P(clear) {base['p_clear']:.2f}->{px['p_clear']:.2f}, "
                                     f"strength {base['strength']:.0f}->{px['strength']:.0f} (gain {gain:+.3f})"))
            elif item.kind in ("planet", "tarot", "spectral"):
                c = game.clone()
                c.step(a)
                px = build_proxy(c, cfg, self.proxy_cfg)
                gain = px["value"] - base["value"]
                if item.kind != "planet":
                    # an unvalued tarot/spectral: worth its price only when cheap and early
                    gain = (cfg.lam_money * (3 - price)) if game.dollars - price >= floor else -1.0
                out.append((a, gain, f"buy {item.name} ${price} (gain {gain:+.3f})"))
            elif item.kind == "voucher":
                ok = (item.key not in _SKIP_VOUCHERS) and (game.dollars - price >= floor)
                gain = 0.02 if ok else -1.0
                out.append((a, gain, f"buy voucher {item.name} ${price} ({'affordable after floor' if ok else 'below floor'})"))
            elif item.kind == "booster":
                pref = next((i for i, k in enumerate(_PACK_PREF) if k in item.key), len(_PACK_PREF))
                ok = game.dollars - price >= floor or (p_clear < 0.6 and game.dollars >= price)
                gain = (0.015 - 0.002 * pref) if ok else -1.0
                out.append((a, gain, f"open {item.name} ${price} ({'ok' if ok else 'below floor'})"))
            elif item.kind == "card":
                ok = game.dollars - price >= floor + 5
                gain = 0.005 if ok else -1.0
                out.append((a, gain, f"buy card {item.name} ${price}"))
        # 3. make room: sell the weakest joker when a shelf joker beats it clearly
        if len(game.jokers) >= game.joker_slots and game.jokers:
            shelf_gain = 0.0
            for a in legal:
                if a["type"] == "buy" and game.current_shop[a["item_idx"]].kind == "joker":
                    continue
            # gains of shelf jokers that could not be bought (slots full) -- evaluate on a clone with a slot freed
            worst = None
            for ji in range(len(game.jokers)):
                c = game.clone()
                c.step({"type": "sell_joker", "joker_idx": ji})
                px = build_proxy(c, cfg, self.proxy_cfg)
                loss = base["value"] - px["value"]
                if worst is None or loss < worst[1]:
                    worst = (ji, loss, c)
            if worst is not None:
                ji, loss, c_sold = worst
                best_shelf = None
                for i, item in enumerate(game.current_shop):
                    if item.kind != "joker" or item.sold:
                        continue
                    price = effective_price(game, item)
                    if c_sold.dollars < price:
                        continue
                    c2 = c_sold.clone()
                    c2.step({"type": "buy", "item_idx": i})
                    if len(c2.jokers) <= len(c_sold.jokers):
                        continue
                    px2 = build_proxy(c2, cfg, self.proxy_cfg)
                    g2 = px2["value"] - base["value"]
                    if best_shelf is None or g2 > best_shelf[1]:
                        best_shelf = (item, g2)
                if best_shelf is not None and best_shelf[1] > cfg.sell_margin:
                    out.append(({"type": "sell_joker", "joker_idx": ji}, best_shelf[1],
                                f"sell {game.jokers[ji].key} to buy {best_shelf[0].name} (gain {best_shelf[1]:+.3f})"))
        # 4. reroll: at most N per visit, never below the floor, only when nothing is worth buying
        if _hand._action_sort_key({"type": "reroll"}) in legal_keys:
            done = max(0, int(game.reroll_cost) - 5)
            cost = max(0, game.reroll_cost - game.reroll_discount)
            free = game.free_rerolls_remaining > 0
            best_buy = max((ev for a, ev, _ in out if a["type"] in ("buy", "sell_joker")), default=-1.0)
            ok = (free or (done < cfg.max_rerolls_per_visit and game.dollars - cost >= cfg.reroll_floor)) \
                and best_buy <= cfg.buy_margin
            out.append(({"type": "reroll"}, 0.001 if ok else -1.0,
                        f"reroll ${cost} ({'ok' if ok else 'no'}; {done} done)"))
        out.append(({"type": "leave_shop"}, cfg.buy_margin, f"leave (P(clear next) {p_clear:.2f}, ${game.dollars})"))
        out.sort(key=lambda x: (-x[1], _hand._action_sort_key(x[0])))
        return out

    # ── rule tier: booster ──────────────────────────────────────────────────

    def _rank_booster_rules(self, game, legal: list) -> list:
        cfg = self.cfg
        base = build_proxy(game, cfg, self.proxy_cfg)
        out = []
        for a in legal:
            if a["type"] != "pick_booster":
                continue
            c = game.clone()
            c.step(a)
            px = build_proxy(c, cfg, self.proxy_cfg)
            gain = px["value"] - base["value"]
            names = []
            for i in a.get("indices", []):
                ch = game.booster_choices[i] if i < len(game.booster_choices) else None
                names.append(getattr(ch, "key", None) or (repr(ch.card) if getattr(ch, "card", None) else "?"))
            # a pick that the proxy cannot see (tarot / spectral / unvalued joker) is still
            # free value: prefer taking something over skipping
            gain = max(gain, 1e-4) if gain <= 0 else gain
            out.append((a, gain, f"take {', '.join(names)} (gain {gain:+.3f})"))
        out.append(({"type": "skip_booster"}, 0.0, "skip the pack"))
        out.sort(key=lambda x: (-x[1], _hand._action_sort_key(x[0])))
        return out
