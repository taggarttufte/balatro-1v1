"""
End-to-end ENGINE-vs-ground-truth parity harness (Phase 1, workstream W8).

Drives the REAL engine (``engine/balatro_sim.BalatroGame`` through ``step()``), not
``rng/generate.py``, on each oracle seed with a scripted policy and diffs what the engine
SHOWS (shelf items + editions, pack contents, voucher, boss, tags) against
``oracle/ground_truth/<SEED>.json``.  Its sibling ``parity_check.py`` validates the
generation layer in isolation; this file validates the game-state *plumbing* around it
(run start -> boss/voucher/tags, boss defeat -> next voucher, Cash Out -> tags/boss, shelf
release on leave, reroll = queue advance, pack open/skip release) that neither Phase 0
harness covers.

    python -m oracle.engine_parity --seeds 7I4M53DL,ALEEB --antes 1-3 --variant faithful
    python -m oracle.engine_parity --antes 1-8 --rerolls 5 --quiet
    python -m oracle.engine_parity --seeds ALEEB --reference generate --reroll-every-shop --buy-shelf
    python -m oracle.engine_parity --seeds ALEEB --buy-vouchers
    python -m oracle.engine_parity --probe            # report which target hooks the engine exposes

Scripted policy (``Policy``):
  * clear every blind (``debug_win_blind()`` if the engine has it, else the harness forces
    ``ROUND_EVAL`` -- the policy never needs to score, so the effect-roll streams are untouched);
  * never skip; buy nothing (unless ``--buy-shelf`` / ``--buy-vouchers``);
  * open both packs at every visit IN DISPLAY ORDER, BEFORE any reroll (that is the order the
    ground truth assumes -- a pack opened with a different shelf displayed can legitimately
    contain different cards under the game-faithful ``used_jokers`` rule), then skip the pack;
  * reroll ``--rerolls N`` times at the LAST visit of each ante (again matching the ground-truth
    queue policy; ``--reroll-every-shop`` moves the rerolls to every visit, which is only exact
    against ``--reference generate``).

The policy's own consumption is replayed into the comparison: k rerolls consume queue positions
``[2k, 2k+2)`` of the ante's ``shop_queue``; purchases are replayed through ``generate.RunState``
(``--reference generate``: the oracle-verified generation layer driven with the *same* action
history -- rerolls, purchases, pack opens -- so any policy can be compared, not only the one the
JSON corpus fixed).  ``--buy-vouchers`` uses the JSON's ``voucher_chain_if_bought`` branch.

Exit code 0 when every seed is exact through the requested antes, 1 otherwise, 2 when the engine
cannot even be driven (missing hooks -- see ``--probe`` and tests/HARNESS_NOTES.md).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
MP_ROOT = os.path.dirname(HERE)
ENGINE_ROOT = os.path.join(MP_ROOT, "engine")
for _p in (MP_ROOT,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from oracle import parity_check as PC  # noqa: E402  (loaders, variant overlay, diff printer)
from rng import generate as GEN  # noqa: E402
from rng import pools as P  # noqa: E402


# ============================================================================ engine import

def import_engine():
    """Import the FORK under engine (never any other ``balatro_sim`` on sys.path).

    Mirrors engine/conftest.py: puts engine first on sys.path and refuses to run if a
    different ``balatro_sim`` won."""
    if ENGINE_ROOT in sys.path:
        sys.path.remove(ENGINE_ROOT)
    sys.path.insert(0, ENGINE_ROOT)
    already = sys.modules.get("balatro_sim")
    if already is not None:
        got = os.path.normcase(os.path.abspath(getattr(already, "__file__", "") or ""))
        want = os.path.normcase(os.path.join(ENGINE_ROOT, "balatro_sim", "__init__.py"))
        if got != want:
            raise RuntimeError(
                "a different balatro_sim is already imported in this process:\n"
                f"  got:      {got}\n  expected: {want}\n"
                "Run the harness in a fresh interpreter (python -m oracle.engine_parity / "
                "python -m pytest tests)."
            )
    import balatro_sim  # noqa: WPS433
    got = os.path.normcase(os.path.abspath(balatro_sim.__file__))
    want = os.path.normcase(os.path.join(ENGINE_ROOT, "balatro_sim", "__init__.py"))
    if got != want:
        raise RuntimeError(f"imported the wrong balatro_sim: {got} (expected {want})")
    from balatro_sim import game as game_mod  # noqa: WPS433
    return game_mod


# ============================================================================ hook probing
#
# The target architecture (CAMPAIGN_LOG "Phase 1 -- REVISED architecture") gives the engine a
# ``run_state`` (generate.RunState) and keyed RNG.  Everything below is read defensively so the
# harness runs (and fails informatively) against today's engine; ``probe_hooks`` lists exactly
# which hooks are present so W2/W3 can see what is missing.

HOOKS = [
    # (name, description, how it is detected)
    ("string_seed", "BalatroGame(seed='7I4M53DL') accepts a Balatro seed string (normalize_seed)", "ctor"),
    ("run_state", "game.run_state is a generate.RunState kept in sync with the game", "attr"),
    ("keyed_rng", "game.run_state.rng is a core.PseudoRandom (no game.rng)", "attr"),
    ("boss_blind", "game.boss_blind / run_state.boss_blind: this ante's boss, known at ante start", "attr"),
    ("blind_tags", "game.blind_tags / run_state.blind_tags: {'Small': tag_key, 'Big': tag_key}", "attr"),
    ("tags", "game.tags: list of owned tag keys (skip rewards), W6 wiring", "attr"),
    ("booster_state", "buying a booster enters State.BOOSTER_OPEN with game.booster_choices populated", "dynamic"),
    ("booster_items", "booster_choices entries expose .key (+ .edition/.enhancement/.seal/.front)", "dynamic"),
    ("shelf_game_keys", "current_shop items carry game keys (j_*/c_*/v_*/p_*) and game edition names", "dynamic"),
    ("debug_win_blind", "game.debug_win_blind(): clear the current blind without scoring (harness helper)", "attr"),
    ("debug_add_joker", "game.debug_add_joker(key, edition=None): own a joker AND mark run_state (harness helper)", "attr"),
    ("state_signature", "game.state_signature(): canonical hashable snapshot (determinism tests)", "attr"),
]


def probe_hooks(game_mod=None) -> dict:
    game_mod = game_mod or import_engine()
    out = {}
    g = None
    try:
        g = game_mod.BalatroGame(seed="7I4M53DL")
        out["string_seed"] = True
    except Exception as ex:  # pragma: no cover
        out["string_seed"] = f"raises {type(ex).__name__}: {ex}"
        g = game_mod.BalatroGame(seed=1)
    rs = getattr(g, "run_state", None)
    # duck-typed: W2 may import the module as rng.generate (a distinct class object)
    out["run_state"] = rs is not None and all(hasattr(rs, a) for a in ("used_jokers", "acquire", "release_shop", "rng"))
    out["keyed_rng"] = bool(rs is not None and hasattr(rs, "rng") and hasattr(rs.rng, "pseudorandom")
                            and not hasattr(g, "rng"))
    out["boss_blind"] = hasattr(g, "boss_blind") or (rs is not None and getattr(rs, "boss_blind", None) is not None)
    out["blind_tags"] = hasattr(g, "blind_tags") or (rs is not None and bool(getattr(rs, "blind_tags", None)))
    out["tags"] = hasattr(g, "tags")
    out["debug_win_blind"] = callable(getattr(g, "debug_win_blind", None))
    out["debug_add_joker"] = callable(getattr(g, "debug_add_joker", None))
    out["state_signature"] = callable(getattr(g, "state_signature", None))
    # dynamic: reach the first shop and buy a booster
    try:
        d = EngineDriver(game_mod, "7I4M53DL")
        d.win_blind()
        shelf = d.shelf_items()
        out["shelf_game_keys"] = all(_is_game_key(i["key"]) for i in shelf) and bool(shelf)
        packs = d.pack_slots()
        if packs:
            ok, choices = d.open_pack(packs[0][0])
            out["booster_state"] = ok
            out["booster_items"] = bool(choices) and all(_is_game_key(c.get("key", "")) for c in choices)
            if ok:
                d.skip_pack()
        else:
            out["booster_state"] = "no pack on shelf"
            out["booster_items"] = "no pack on shelf"
    except Exception as ex:
        out["booster_state"] = f"raises {type(ex).__name__}: {ex}"
        out["booster_items"] = out["booster_state"]
        out.setdefault("shelf_game_keys", out["booster_state"])
    return out


_GAME_KEY_RE = re.compile(r"^(j_|c_|v_|p_|m_|tag_|bl_|[HCDS]_[2-9TJQKA]$)")


def _is_game_key(k: str) -> bool:
    return bool(k) and (k in GEN_KEYS or bool(_GAME_KEY_RE.match(k)) and k in GEN_KEYS_PREFIXED)


def _all_game_keys() -> set:
    keys = set()
    for lst in (P.JOKERS, P.TAROTS, P.PLANETS, P.SPECTRALS, P.VOUCHERS, P.BOOSTERS, P.ENHANCEMENTS, P.TAGS, P.BLINDS):
        keys.update(c["key"] for c in lst)
    keys.update(P.PLAYING_CARD_KEYS)
    keys.add("c_base")
    return keys


GEN_KEYS = _all_game_keys()
GEN_KEYS_PREFIXED = GEN_KEYS | {re.sub(r"_\d+$", "", k) for k in GEN_KEYS if k.startswith("p_")}


# ============================================================================ normalisation

_EDITION = {None: None, "None": None, "": None,
            "foil": "e_foil", "holo": "e_holo", "holographic": "e_holo", "polychrome": "e_polychrome",
            "negative": "e_negative", "e_foil": "e_foil", "e_holo": "e_holo", "e_polychrome": "e_polychrome",
            "e_negative": "e_negative", "Foil": "e_foil", "Holographic": "e_holo", "Holo": "e_holo",
            "Polychrome": "e_polychrome", "Negative": "e_negative"}
_ENH = {None: None, "None": None, "": None, "Bonus": "m_bonus", "Mult": "m_mult", "Wild": "m_wild", "Glass": "m_glass",
        "Steel": "m_steel", "Stone": "m_stone", "Gold": "m_gold", "Lucky": "m_lucky"}
_SEAL = {None: None, "None": None, "": None, "Red": "Red", "Blue": "Blue", "Gold": "Gold", "Purple": "Purple"}
_SUIT_LETTER = {"Hearts": "H", "Clubs": "C", "Diamonds": "D", "Spades": "S"}
_RANK_LETTER = {2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10: "T", 11: "J", 12: "Q", 13: "K", 14: "A"}


def norm_edition(e) -> Optional[str]:
    return _EDITION.get(e, e)


def norm_enh(e) -> Optional[str]:
    if e is None:
        return None
    return _ENH.get(e, e if str(e).startswith("m_") else e)


def norm_seal(s) -> Optional[str]:
    return _SEAL.get(s, s)


def set_of_key(key: str) -> str:
    if key.startswith("j_"):
        return "Joker"
    if key.startswith("v_"):
        return "Voucher"
    if key.startswith("p_"):
        return "Booster"
    if key in P.PLAYING_CARD_KEYS or key == "c_base" or key.startswith("m_"):
        return "Base"
    try:
        return GEN.center_set(key)
    except Exception:
        return "?"


def pack_key_base(key: str) -> str:
    """'p_arcana_normal_1' -> 'p_arcana_normal' (art suffix is cosmetic: unseeded math.random)."""
    return re.sub(r"_\d+$", "", key or "")


def card_front(card) -> Optional[str]:
    """engine Card -> G.P_CARDS key ('H_T')."""
    suit = getattr(card, "suit", None)
    rank = getattr(card, "rank", None)
    if suit in _SUIT_LETTER and rank in _RANK_LETTER:
        return f"{_SUIT_LETTER[suit]}_{_RANK_LETTER[rank]}"
    return None


def item_from_engine(obj) -> dict:
    """Normalise whatever the engine puts on a shelf / in a pack into a ground-truth <item>.

    Accepts: ShopItem (kind/key/edition), generate.CardGen, engine Card, ('card', Card) tuples,
    bare key strings, dicts, or any object with .key.  Unknown shapes become
    {'set': '?', 'key': repr(obj)} so the diff shows exactly what the engine produced."""
    if obj is None:
        return {"set": "?", "key": "<none>"}
    if isinstance(obj, tuple) and len(obj) == 2 and obj[0] == "card":
        obj = obj[1]
    if isinstance(obj, str):
        return _item_from_key(obj)
    if isinstance(obj, dict):
        d = dict(obj)
        d.setdefault("set", set_of_key(d.get("key", "")))
        d["edition"] = norm_edition(d.get("edition"))
        return d
    if isinstance(obj, GEN.CardGen):
        return PC._cardgen_item(obj)
    # engine Card (playing card)
    if hasattr(obj, "rank") and hasattr(obj, "suit") and not hasattr(obj, "key"):
        return {"set": "Base", "key": card_front(obj), "enhancement": norm_enh(getattr(obj, "enhancement", None)),
                "edition": norm_edition(getattr(obj, "edition", None)), "seal": norm_seal(getattr(obj, "seal", None))}
    key = getattr(obj, "key", None)
    if key is None:
        return {"set": "?", "key": repr(obj)}
    it = _item_from_key(key)
    ed = getattr(obj, "edition", None)
    if ed is not None:
        it["edition"] = norm_edition(ed)
    if it["set"] == "Base":
        front = getattr(obj, "front", None) or card_front(obj)
        if front:
            it["key"] = front
        it["enhancement"] = norm_enh(getattr(obj, "enhancement", None))
        it["seal"] = norm_seal(getattr(obj, "seal", None))
    if it["set"] == "Joker":
        it.setdefault("edition", None)
    return it


def _item_from_key(key: str) -> dict:
    s = set_of_key(key)
    it: dict = {"set": s, "key": key}
    if s == "Joker":
        it["edition"] = None
        it["rarity"] = P.JOKER_BY_KEY.get(key, {}).get("rarity")
        it["stickers"] = {"eternal": False, "perishable": False, "rental": False}
    elif s == "Base":
        it.update({"enhancement": None, "edition": None, "seal": None})
    return it


# ============================================================================ engine driver

@dataclass
class Visit:
    """What the engine showed at one shop visit plus the actions taken there, in order."""
    index: int                                   # 0-based visit counter across the run
    ante: int                                    # ground-truth ante this visit belongs to
    visit: str                                   # after_small | after_big | after_prev_boss
    shelves: list = field(default_factory=list)  # list of shelf snapshots: [items] (index 0 = as entered, then after each reroll)
    packs: list = field(default_factory=list)    # [{'key': base pack key, 'cards': [items]}] in slot order
    voucher: Optional[str] = None
    actions: list = field(default_factory=list)  # ('reroll',) | ('buy', slot, item) | ('open', pack_slot) | ('buy_voucher', key)


class EngineDriver:
    """Scripted walk through ``BalatroGame`` that records every generation-relevant thing the
    engine displays.  Uses only the public ``step()`` API plus the documented harness hooks;
    where a hook is missing it falls back to the minimal state poke (flagged in ``fallbacks``)."""

    def __init__(self, game_mod, seed: str, money: int = 10 ** 6):
        self.gm = game_mod
        self.State = game_mod.State
        self.seed = seed
        self.money = money
        self.fallbacks: set = set()
        try:
            self.g = game_mod.BalatroGame(seed=seed)
        except TypeError:
            self.fallbacks.add("string_seed")
            self.g = game_mod.BalatroGame(seed=seed)  # re-raise naturally if it truly cannot
        self.visits: list[Visit] = []
        self.ante_info: dict = {}   # gt ante -> {'boss': key|None, 'tags': (small, big)|None}
        self._record_ante_start(1)

    # -- state helpers ---------------------------------------------------------------

    @property
    def run_state(self):
        return getattr(self.g, "run_state", None)

    def boss_now(self) -> Optional[str]:
        rs = self.run_state
        for src in (getattr(self.g, "boss_blind", None), getattr(rs, "boss_blind", None) if rs else None):
            if src:
                return src
        cb = getattr(self.g, "current_blind", None)
        if cb is not None and getattr(cb, "kind", "") == "Boss" and getattr(cb, "boss_key", ""):
            return cb.boss_key
        return None

    def tags_now(self) -> Optional[tuple]:
        rs = self.run_state
        for src in (getattr(self.g, "blind_tags", None), getattr(rs, "blind_tags", None) if rs else None):
            if src and isinstance(src, dict) and src.get("Small") and src.get("Big"):
                return (src["Small"], src["Big"])
            if src and isinstance(src, (tuple, list)) and len(src) == 2:
                return tuple(src)
        return None

    def _record_ante_start(self, gt_ante: int):
        self.ante_info[gt_ante] = {"boss": self.boss_now(), "tags": self.tags_now()}

    def _set_money(self):
        self.g.dollars = max(self.g.dollars, self.money)

    # -- blind ----------------------------------------------------------------------

    def win_blind(self):
        """From BLIND_SELECT: play the blind, clear it without scoring, land in SHOP."""
        g, S = self.g, self.State
        assert g.state == S.BLIND_SELECT, g.state
        boss_before = self.boss_now()
        was_boss = getattr(g.current_blind, "kind", "") == "Boss"
        g.step({"type": "play_blind"})
        assert g.state == S.SELECTING_HAND, g.state
        if boss_before is None and g.current_blind.kind == "Boss":
            # engine has no ante-start boss hook: take it from the blind itself
            self.ante_info[self._gt_ante_of_next_visit()]["boss"] = g.current_blind.boss_key
        win = getattr(g, "debug_win_blind", None)
        if callable(win):
            win()
        else:
            self.fallbacks.add("debug_win_blind")
            g.chips_scored = g.current_blind.chips_target
            g.state = S.ROUND_EVAL
        if g.state == S.ROUND_EVAL:
            g.step({"type": "advance"})
        assert g.state == S.SHOP, f"expected SHOP after clearing the blind, got {g.state}"
        if was_boss:
            # W2: the next ante's voucher/tags/boss are drawn when the boss dies (end_round ->
            # ease_ante -> get_next_voucher_key; Cash Out -> get_next_tag_key x2, reset_blinds ->
            # get_new_boss) -- i.e. NOW, as the post-boss shop (the ground truth's first visit of
            # the next ante) opens.  Record them here; leave_shop() only fills gaps.
            self._record_ante_start(self._gt_ante_of_next_visit())
        self._set_money()

    def _gt_ante_of_next_visit(self) -> int:
        n = len(self.visits)
        return 1 if n < 2 else 2 + (n - 2) // 3

    def _gt_visit_label(self) -> str:
        n = len(self.visits)
        if n < 2:
            return ["after_small", "after_big"][n]
        return ["after_prev_boss", "after_small", "after_big"][(n - 2) % 3]

    # -- shop -----------------------------------------------------------------------

    def shelf_items(self) -> list:
        out = []
        for it in self.g.current_shop:
            kind = getattr(it, "kind", "")
            if kind in ("voucher", "booster") or getattr(it, "sold", False):
                continue
            out.append(item_from_engine(it))
        return out

    def voucher_item(self) -> Optional[str]:
        for it in self.g.current_shop:
            if getattr(it, "kind", "") == "voucher" and not getattr(it, "sold", False):
                return it.key
        return None

    def pack_slots(self) -> list:
        """[(shop_idx, pack_key)] in slot order."""
        return [(i, it.key) for i, it in enumerate(self.g.current_shop)
                if getattr(it, "kind", "") == "booster" and not getattr(it, "sold", False)]

    def open_pack(self, shop_idx: int):
        """Buy the booster at shop index -> expect BOOSTER_OPEN with choices.  Returns (ok, items)."""
        g, S = self.g, self.State
        self._set_money()
        g.step({"type": "buy", "item_idx": shop_idx})
        choices = list(getattr(g, "booster_choices", []) or [])
        items = [item_from_engine(c) for c in choices]
        return (g.state == S.BOOSTER_OPEN), items

    def skip_pack(self):
        g, S = self.g, self.State
        if g.state == S.BOOSTER_OPEN:
            g.step({"type": "skip_booster"})
        else:
            # engine bug §7 item 0: BOOSTER_OPEN never entered -> choices are stale on the game object
            self.fallbacks.add("booster_state")
            g.booster_choices = []
            g.booster_picks_remaining = 0
        assert g.state == S.SHOP, g.state

    def reroll(self) -> list:
        self._set_money()
        self.g.step({"type": "reroll"})
        return self.shelf_items()

    def buy_slot(self, slot: int):
        """Buy the slot-th non-voucher/non-booster shelf card.  Returns the item bought or None."""
        g = self.g
        idx = [i for i, it in enumerate(g.current_shop)
               if getattr(it, "kind", "") not in ("voucher", "booster") and not getattr(it, "sold", False)]
        if slot >= len(idx):
            return None
        it = g.current_shop[idx[slot]]
        item = item_from_engine(it)
        self._set_money()
        g.step({"type": "buy", "item_idx": idx[slot]})
        if not getattr(it, "sold", False):
            return None
        return item

    def buy_voucher(self) -> Optional[str]:
        g = self.g
        for i, it in enumerate(g.current_shop):
            if getattr(it, "kind", "") == "voucher" and not getattr(it, "sold", False):
                self._set_money()
                g.step({"type": "buy", "item_idx": i})
                return it.key if getattr(it, "sold", False) else None
        return None

    def leave_shop(self):
        g, S = self.g, self.State
        assert g.state == S.SHOP, g.state
        ante_before = len(self.visits)
        g.step({"type": "leave_shop"})
        # Fallback for engines without the ante-start hooks: a new ground-truth ante starts
        # when the post-boss shop is left (engine-side ante++).  With the hooks, win_blind()
        # already recorded the ante when the boss died; do not overwrite it.
        if g.state == S.BLIND_SELECT and self._gt_ante_of_next_visit() != self._gt_ante_of_visit(ante_before - 1):
            nxt = self._gt_ante_of_next_visit()
            if nxt not in self.ante_info:
                self._record_ante_start(nxt)

    def _gt_ante_of_visit(self, n: int) -> int:
        return 1 if n < 2 else 2 + (n - 2) // 3

    # -- the scripted visit ---------------------------------------------------------

    def do_visit(self, rerolls: int, open_packs: bool = True, buy_shelf: Optional[int] = None,
                 buy_voucher: bool = False) -> Visit:
        g = self.g
        v = Visit(index=len(self.visits), ante=self._gt_ante_of_next_visit(), visit=self._gt_visit_label())
        v.voucher = self.voucher_item()
        v.shelves.append(self.shelf_items())
        if open_packs:
            for slot, (shop_idx, pkey) in enumerate(self.pack_slots()):
                ok, items = self.open_pack(shop_idx)
                v.packs.append({"key": pack_key_base(pkey), "opened": ok, "cards": items})
                v.actions.append(("open", slot))
                self.skip_pack()
        else:
            for slot, (shop_idx, pkey) in enumerate(self.pack_slots()):
                v.packs.append({"key": pack_key_base(pkey), "opened": False, "cards": None})
        if buy_voucher:
            vk = self.buy_voucher()
            if vk:
                v.actions.append(("buy_voucher", vk))
        if buy_shelf is not None:
            it = self.buy_slot(buy_shelf)
            if it is not None:
                v.actions.append(("buy", buy_shelf, it))
        for _ in range(rerolls):
            v.shelves.append(self.reroll())
            v.actions.append(("reroll",))
        self.visits.append(v)
        self.leave_shop()
        return v

    def run(self, antes: int, policy: "Policy") -> None:
        """Walk ``antes`` antes: 2 visits in ante 1, then 3 per ante."""
        total_visits = 2 + 3 * (antes - 1)
        while len(self.visits) < total_visits and self.g.state != self.State.GAME_OVER:
            n = len(self.visits)
            last_of_ante = (n == 1) or (n >= 2 and (n - 2) % 3 == 2)
            first_of_ante = (n == 0) or (n >= 2 and (n - 2) % 3 == 0)
            self.win_blind()
            rer = policy.rerolls if (policy.reroll_every_shop or last_of_ante) else 0
            self.do_visit(rerolls=rer, open_packs=policy.open_packs,
                          buy_shelf=(policy.buy_shelf_slot if policy.buy_shelf else None),
                          buy_voucher=(policy.buy_vouchers and first_of_ante))

    def observed(self) -> dict:
        """Ground-truth-shaped dict of what the engine showed (for diff_ante)."""
        out: dict = {"seed": self.seed, "antes": {}}
        by_ante: dict = {}
        for v in self.visits:
            by_ante.setdefault(v.ante, []).append(v)
        for a, vs in by_ante.items():
            queue: list = []
            shops = []
            for v in vs:
                for shelf in v.shelves:
                    queue.extend(shelf)
                shops.append({"visit": v.visit, "packs": [{"key": p["key"], "cards": p["cards"]} for p in v.packs],
                              "voucher": v.voucher})
            info = self.ante_info.get(a, {})
            tags = info.get("tags")
            out["antes"][str(a)] = {
                "boss": {"key": info.get("boss")} if info.get("boss") else None,
                "voucher": {"key": vs[0].voucher} if vs[0].voucher else None,
                "tags": {"small": {"key": tags[0]}, "big": {"key": tags[1]}} if tags else None,
                "shop_queue": queue,
                "shops": shops,
            }
        return out


@dataclass
class Policy:
    rerolls: int = 0
    reroll_every_shop: bool = False
    open_packs: bool = True
    buy_shelf: bool = False
    buy_shelf_slot: int = 0
    buy_vouchers: bool = False


# ============================================================================ generate.py replay

def replay_visits(seed: str, visits: list, deck: str = "b_red", stake: str = "stake_white") -> dict:
    """Replay an action history (``EngineDriver.visits``) through the oracle-verified
    ``generate.RunState`` and return the ground-truth-shaped dict the engine SHOULD have shown.

    This is the reference for any policy the JSON corpus does not cover (purchases, rerolls at
    every visit, Showman, ...): the generation layer itself is proven against the Lua
    (tests/test_generate_oracle.py), so disagreement here is an engine plumbing bug."""
    st = GEN.RunState.for_stake(seed, stake=PC._STAKE_N.get(stake, 1))
    rs = GEN.start_run(st, deck)
    out: dict = {"seed": seed, "antes": {}}
    cur_ante = 0
    info = {"boss": rs.boss, "voucher": rs.voucher, "tag_small": rs.tag_small, "tag_big": rs.tag_big}
    for v in visits:
        if v.ante != cur_ante:
            if cur_ante != 0:
                info = GEN.defeat_boss(st)
            cur_ante = v.ante
            out["antes"][str(cur_ante)] = {
                "boss": {"key": info["boss"]}, "voucher": {"key": info["voucher"]},
                "tags": {"small": {"key": info["tag_small"]}, "big": {"key": info["tag_big"]}},
                "shop_queue": [], "shops": [],
            }
        node = out["antes"][str(cur_ante)]
        st.new_round()
        shop = GEN.generate_shop(st)
        node["shop_queue"].extend(PC._cardgen_item(c) for c in shop.cards)
        packs = []
        for act in v.actions:
            if act[0] == "open":
                slot = act[1]
                pk = shop.boosters[slot] if slot < len(shop.boosters) else None
                if pk is None:
                    packs.append({"key": None, "cards": []})
                    continue
                cards = GEN.open_pack(st, pk)
                packs.append({"key": pack_key_base(pk), "cards": [PC._cardgen_item(c) for c in cards]})
                st.release_pack(cards)
                st.used_packs[slot] = "USED"
            elif act[0] == "reroll":
                GEN.reroll_shop(st, shop)
                node["shop_queue"].extend(PC._cardgen_item(c) for c in shop.cards)
            elif act[0] == "buy":
                slot = act[1]
                if slot < len(shop.cards):
                    st.acquire(shop.cards[slot])
            elif act[0] == "buy_voucher":
                st.used_vouchers.add(act[1])
                st.current_round_voucher = None
                shop.voucher = None
                _apply_voucher_rates(st, act[1])
        node["shops"].append({"visit": v.visit, "packs": packs, "voucher": shop.voucher})
        st.release_shop(shop)
    return out


def _apply_voucher_rates(st, key: str) -> None:
    """The generation-relevant effects of redeeming a voucher (card.lua:1890-1909)."""
    rates = P.SHOP_RATE_BY_VOUCHER.get(key)
    if isinstance(rates, dict):
        for k, val in rates.items():
            setattr(st, k, val)
    if key == "v_hone":
        st.edition_rate = 2
    elif key == "v_glow_up":
        st.edition_rate = 4
    elif key in ("v_overstock_norm", "v_overstock_plus"):
        st.shop_joker_max += 1
    elif key in ("v_hieroglyph", "v_petroglyph"):
        st.ante -= 1


# ============================================================================ diffing

def compare_seed(gt: dict, got: dict, antes: list, fields: list, depth: int, flagged: set) -> tuple:
    """Returns (rows, k_exact, absent) using parity_check.diff_ante.  ``got`` may carry None
    for boss/voucher/tags (engine did not produce) -> reported as 'not produced'."""
    rows, absent_all, k_exact, broken = [], set(), 0, False
    for a in antes:
        exp = gt["antes"].get(str(a))
        if exp is None:
            break
        g = got["antes"].get(str(a))
        if g is not None:
            g = {k: (v if v is not None else None) for k, v in g.items()}
            # diff_ante treats a missing dict as 'not produced'; replace None with {} so dig() reports absence
            for k in ("boss", "voucher", "tags"):
                if g.get(k) is None:
                    g[k] = {}
        mism, absent = PC.diff_ante(exp, g, fields, depth)
        absent_all.update(absent)
        if mism:
            for f, e, gg in mism:
                label = f"ante{a}.{f}"
                norm = re.sub(r"shops\[(\d+):[a-z_]+\]", r"shops[\1]", label)
                if norm in flagged:
                    label += "  [analyzer gap: used_jokers]"
                rows.append((label, e, gg))
            broken = True
        elif not broken:
            k_exact = a
    return rows, k_exact, sorted(absent_all)


def compare_voucher_chain(gt: dict, driver: EngineDriver, antes: list) -> list:
    """--buy-vouchers: the engine bought the shelf voucher at the first visit of every ante; the
    JSON's voucher_chain_if_bought says which voucher each later ante should then show."""
    rows = []
    chain = (gt.get("voucher_chain_if_bought") or {}).get("antes") or {}
    by_ante: dict = {}
    for v in driver.visits:
        by_ante.setdefault(v.ante, []).append(v)
    for a in antes:
        exp = (chain.get(str(a)) or {}).get("voucher", {}).get("key")
        vs = by_ante.get(a)
        if exp is None or not vs:
            continue
        got = vs[0].voucher
        if got != exp:
            rows.append((f"ante{a}.voucher[bought-chain]", str(exp), str(got)))
        names = (chain.get(str(a)) or {}).get("shop_queue_first6_names")
        if names:
            shown = []
            for v in vs:
                for it in v.shelves[0]:
                    shown.append(it["key"])
            for i, (n, g) in enumerate(zip(names, shown)):
                try:
                    w = PC.K.key_for(n)
                except KeyError:
                    break   # playing-card names ("Mult 9 of Hearts", Magic Trick) are not in the keymap
                if w != g:
                    rows.append((f"ante{a}.shop_queue[{i}][bought-chain, analyzer-variant]", w, g))
                    break
    return rows


# ============================================================================ main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", help="comma-separated subset (default: all ground-truth seeds)")
    ap.add_argument("--antes", default="1-3", help="e.g. 1-3 or 1,2 (default 1-3)")
    ap.add_argument("--fields", default=",".join(PC.FIELDS_DEFAULT),
                    help="subset of boss,voucher,tags,shop_queue,packs (default: all)")
    ap.add_argument("--variant", choices=["analyzer", "faithful"], default="faithful",
                    help="ground-truth variant (default faithful = the game's used_jokers rule applied)")
    ap.add_argument("--reference", choices=["ground_truth", "generate"], default="ground_truth",
                    help="compare against the JSON corpus (default) or against generate.py replaying the "
                         "engine's own action history (any policy)")
    ap.add_argument("--rerolls", type=int, default=3, help="rerolls at the last visit of each ante (default 3)")
    ap.add_argument("--reroll-every-shop", action="store_true", help="reroll at every visit (exact only vs --reference generate)")
    ap.add_argument("--no-packs", action="store_true", help="do not open packs")
    ap.add_argument("--buy-shelf", action="store_true", help="buy shelf slot 0 at every visit (only vs --reference generate)")
    ap.add_argument("--buy-vouchers", action="store_true", help="buy the voucher at the first visit of each ante; compare to voucher_chain_if_bought")
    ap.add_argument("--probe", action="store_true", help="report which target hooks the engine exposes and exit")
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    try:
        game_mod = import_engine()
    except Exception as ex:
        print(f"[engine] cannot import the engine fork: {type(ex).__name__}: {ex}")
        return 2

    if args.probe:
        hooks = probe_hooks(game_mod)
        print("engine hook probe (target architecture, see tests/HARNESS_NOTES.md):")
        for name, desc, _ in HOOKS:
            v = hooks.get(name)
            mark = "ok " if v is True else "-- "
            print(f"  [{mark}] {name:18s} {desc}" + ("" if v in (True, False) else f"  ({v})"))
        missing = [n for n, _, _ in HOOKS if hooks.get(n) is not True]
        print(f"missing: {len(missing)}/{len(HOOKS)}: {', '.join(missing)}")
        return 0 if not missing else 1

    seeds = [s.strip().upper() for s in args.seeds.split(",")] if args.seeds else None
    gts = PC.load_ground_truth(seeds)
    if not gts:
        print(f"no ground-truth files in {PC.GT_DIR}" + (f" for {seeds}" if seeds else ""))
        return 2
    antes = PC.parse_antes(args.antes)
    fields = [f for f in args.fields.split(",") if f]
    policy = Policy(rerolls=args.rerolls, reroll_every_shop=args.reroll_every_shop, open_packs=not args.no_packs,
                    buy_shelf=args.buy_shelf, buy_vouchers=args.buy_vouchers)
    if args.reference == "ground_truth" and (args.reroll_every_shop or args.buy_shelf):
        print("[policy] --reroll-every-shop / --buy-shelf change pack contents under the faithful rule; "
              "the JSON corpus fixed its policy, so use --reference generate for an exact comparison.")

    print(f"[engine] {game_mod.__file__}")
    print(f"[policy] rerolls={policy.rerolls}{' every shop' if policy.reroll_every_shop else ' at last visit/ante'}; "
          f"packs={'open' if policy.open_packs else 'closed'}; buy_shelf={policy.buy_shelf}; "
          f"buy_vouchers={policy.buy_vouchers}; reference={args.reference}; variant={args.variant}; antes={antes}")

    exact_through: dict = {}
    failures = 0
    fallbacks_seen: set = set()
    for seed, gt0 in gts.items():
        gt = PC.apply_variant(gt0, args.variant)
        flagged = PC.flagged_paths(gt0)
        try:
            d = EngineDriver(game_mod, seed)
            d.run(max(antes), policy)
            got = d.observed()
            fallbacks_seen |= d.fallbacks
            if args.reference == "generate":
                ref = replay_visits(seed, d.visits, gt["deck"], gt["stake"])
                flagged = set()
            else:
                ref = gt
        except Exception as ex:
            failures += 1
            exact_through[seed] = 0
            print(f"\n== {seed}: engine raised {type(ex).__name__}: {ex}")
            if not args.quiet:
                traceback.print_exc()
            continue
        # the engine shows 2 + 3*(antes-1) visits -> shelf depth = 2*(visits_in_ante + rerolls)
        depth_by_ante = {a: 2 * ((2 if a == 1 else 3) + policy.rerolls) for a in antes}
        rows_all: list = []
        k_exact = 0
        broken = False
        absent_all: set = set()
        for a in antes:
            depth = depth_by_ante[a] if not policy.reroll_every_shop else 2 * ((2 if a == 1 else 3) * (1 + policy.rerolls))
            rows, k, absent = compare_seed(ref, got, [a], fields, depth, flagged)
            absent_all.update(absent)
            if rows:
                rows_all.extend(rows)
                broken = True
            elif not broken and str(a) in ref["antes"]:
                k_exact = a
        if policy.buy_vouchers and args.reference == "ground_truth":
            rows_all.extend(compare_voucher_chain(gt, d, antes))
        exact_through[seed] = k_exact
        status = "EXACT" if not rows_all else f"{len(rows_all)} mismatch(es)"
        print(f"\n== {seed}: {status} through ante {k_exact} (checked {antes[0]}-{antes[-1]})"
              + (f"; not produced: {sorted(absent_all)[:6]}" if absent_all else ""))
        if rows_all and not args.quiet:
            PC.print_table(rows_all, limit=args.limit)

    want_k = antes[-1]
    n_exact = sum(1 for k in exact_through.values() if k >= want_k)
    print(f"\nSUMMARY: {n_exact}/{len(gts)} seeds exact through ante {want_k}"
          + (f" ({failures} engine errors)" if failures else ""))
    for k in range(want_k, 0, -1):
        n = sum(1 for v in exact_through.values() if v >= k)
        print(f"  exact through ante {k}: {n}/{len(gts)}")
    if fallbacks_seen:
        print(f"  harness fallbacks used (missing engine hooks): {', '.join(sorted(fallbacks_seen))}")
    return 0 if n_exact == len(gts) else 1


if __name__ == "__main__":
    sys.exit(main())
