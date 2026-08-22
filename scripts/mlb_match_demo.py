"""
mlb_match_demo.py -- play a full Major League Balatro match between two scripted players on
ONE seed through ``MLBMatch`` (the engine's two-player coordinator) and print a readable trace.

Phase 2 W4 (exit gate, 2026-08-21).  This module is also the driver behind
``mp/tests/test_mlb_match_gate.py``: ``ScriptedPlayer`` / ``make_policy`` are the players,
``MatchRecorder`` drives a match step by step and records every gate-relevant event (lives,
cash-out money, Nemesis verdicts, shop visits with shelf / voucher / packs and a snapshot of
the keyed RNG at entry and exit), and ``diff_rng`` / ``classify_key`` / ``key_position`` are
the queue-alignment tools (NOTES: mp/tests/GATE_NOTES.md).

    python mp/scripts/mlb_match_demo.py --seed 7I4M53DL
    python mp/scripts/mlb_match_demo.py --seed 7I4M53DL --deck b_plasma --max-antes 6 --json trace.json
    python mp/scripts/mlb_match_demo.py --seed ALEEB --quiet        # summary only

Players (they deliberately differ so the queue-alignment diff has something to explain):
  P1 "opener"  -- greedy best-hand player; never rerolls; opens pack slot 0 at every shop and
                  picks the first grantable card; buys nothing else.
  P2 "reroller" -- greedy best-hand player; rerolls ONCE per shop when affordable; buys shelf
                  slot 0 when affordable; never opens packs.
Neither skips a blind, sells, or uses a consumable (those are separate streams the gate
classifies but does not need to exercise).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
MP_ROOT = os.path.dirname(HERE)
if MP_ROOT not in sys.path:
    sys.path.insert(0, MP_ROOT)

from oracle.engine_parity import import_engine, item_from_engine  # noqa: E402
from rng import core as RCORE  # noqa: E402

GM = import_engine()                         # the mp/engine fork (refuses the BRL package)
from balatro_sim.mlb_match import MLBMatch  # noqa: E402
from balatro_sim.game import State  # noqa: E402
from balatro_sim.hand_eval import evaluate_hand  # noqa: E402
from balatro_sim.env_v7 import HAND_PRIORITY  # noqa: E402
from balatro_sim.constants import MLB_COMEBACK_PER_LIFE, INTEREST_RATE, HAND_PAYOUT  # noqa: E402
import balatro_sim.jokers  # noqa: E402,F401  (registry)

BalatroGame = GM.BalatroGame
SHELF_KINDS = ("joker", "planet", "tarot", "spectral", "card")


# ============================================================================ players

@dataclass
class ScriptedPlayer:
    """A scripted policy description.  ``hand``: ``greedy`` (best hand type, discard junk
    below Two Pair), ``weak`` (first legal one-card play), ``greedy_until`` (greedy through
    ante ``weak_from_ante - 1``, weak from that ante on -- used to force Nemesis ties until a
    chosen ante).  ``debug_win_regular`` clears every NON-PvP blind through
    ``debug_win_blind()`` (harness helper, touches no stream) so a match can reach any ante."""
    name: str = "player"
    hand: str = "greedy"
    weak_from_ante: int = 99
    rerolls_per_visit: int = 0
    buy_slot0: bool = False
    open_pack_slot: Optional[int] = None    # shop booster slot to buy (0 / 1) or None
    pick_from_pack: bool = True             # pick the first grantable card (else skip the pack)
    buy_voucher: bool = False
    debug_win_regular: bool = False
    rich: bool = False                      # top dollars up to 10**6 at every shop (voucher tests)


def greedy_hand(g, discard: bool = True) -> dict:
    """Best hand type (then most rank chips) over every legal play subset; with ``discard``
    the non-scoring cards are discarded while the best hand is below Two Pair."""
    best = None
    flags = g.hand_eval_flags()
    for a in g.legal_actions():
        if a["type"] != "play":
            continue
        cards = [g.hand[i] for i in a["cards"]]
        ht, scoring = evaluate_hand(cards, **flags)
        key = (HAND_PRIORITY.get(ht, 0), sum(c.rank for c in scoring))
        if best is None or key > best[0]:
            best = (key, a, scoring)
    if best is None:
        return {"type": "play", "cards": [0]}
    (pri, _), a, scoring = best
    if discard and pri < HAND_PRIORITY["Two Pair"] and g.discards_left > 0 and g.hands_left > 1:
        keep = {id(c) for c in scoring}
        junk = [i for i, c in enumerate(g.hand) if id(c) not in keep][:5]
        if junk:
            return {"type": "discard", "cards": junk}
    return a


def weakest_play(g) -> dict:
    return next(a for a in g.legal_actions() if a["type"] == "play")


def shelf_indices(g) -> list:
    return [i for i, it in enumerate(g.current_shop) if it.kind in SHELF_KINDS and not it.sold]


def make_policy(spec: ScriptedPlayer) -> Callable:
    """``policy(match, player, legal_actions) -> action`` for ``MLBMatch.play_out`` and
    ``MatchRecorder``.  Per-visit bookkeeping (rerolls done, pack opened, slot bought) lives
    on the game object under ``_w4_visit`` and is reset by the recorder / on SHOP entry."""

    def visit_state(g) -> dict:
        # (ante, blind_idx) is unique per shop visit: the post-boss shop is (a+1, 2), the
        # next ante's after-Small shop (a+1, 0).
        vs = getattr(g, "_w4_visit", None)
        vid = (g.ante, g.blind_idx)
        if vs is None or vs.get("id") != vid:
            vs = {"id": vid, "rerolls": 0, "opened": False, "bought": False, "voucher": False}
            g._w4_visit = vs
        return vs

    def pol(m, p: int, acts: list) -> dict:
        g = m.games[p]
        s = g.state
        if s == State.BLIND_SELECT:
            return {"type": "play_blind"}
        if s == State.SELECTING_HAND:
            if g.current_blind.is_pvp:
                style = spec.hand
                if style == "greedy_until":
                    style = "weak" if g.ante >= spec.weak_from_ante else "greedy"
            else:
                if spec.debug_win_regular:
                    g.debug_win_blind()
                    return {"type": "advance"}
                style = "weak" if spec.hand == "weak" else "greedy"
            return weakest_play(g) if style == "weak" else greedy_hand(g)
        if s == State.ROUND_EVAL:
            return {"type": "advance"}
        if s == State.BOOSTER_OPEN:
            if spec.pick_from_pack:
                picks = [a for a in acts if a["type"] == "pick_booster"]
                if picks:
                    return picks[0]
            return {"type": "skip_booster"}
        if s == State.SHOP:
            vs = visit_state(g)
            if spec.rich and g.dollars < 10 ** 6:
                g.dollars = 10 ** 6
            legal = {json.dumps(a, sort_keys=True) for a in acts}

            def ok(a):
                return json.dumps(a, sort_keys=True) in legal
            if spec.buy_voucher and not vs["voucher"]:
                vs["voucher"] = True
                for i, it in enumerate(g.current_shop):
                    if it.kind == "voucher" and not it.sold and ok({"type": "buy", "item_idx": i}):
                        return {"type": "buy", "item_idx": i}
            if spec.open_pack_slot is not None and not vs["opened"]:
                vs["opened"] = True
                packs = [i for i, it in enumerate(g.current_shop) if it.kind == "booster" and not it.sold]
                if spec.open_pack_slot < len(packs) and ok({"type": "buy", "item_idx": packs[spec.open_pack_slot]}):
                    return {"type": "buy", "item_idx": packs[spec.open_pack_slot]}
            if vs["rerolls"] < spec.rerolls_per_visit:
                vs["rerolls"] += 1
                if ok({"type": "reroll"}):
                    return {"type": "reroll"}
            if spec.buy_slot0 and not vs["bought"]:
                vs["bought"] = True
                idx = shelf_indices(g)
                if idx and ok({"type": "buy", "item_idx": idx[0]}):
                    return {"type": "buy", "item_idx": idx[0]}
            return {"type": "leave_shop"}
        return acts[0] if acts else {"type": "advance"}

    pol.spec = spec  # type: ignore[attr-defined]
    return pol


OPENER = ScriptedPlayer(name="P1 opener", hand="greedy", open_pack_slot=0, pick_from_pack=True)
REROLLER = ScriptedPlayer(name="P2 reroller", hand="greedy", rerolls_per_visit=1, buy_slot0=True)


# ============================================================================ RNG key tools

class _KeyChain:
    """Lazily extended LCG chain of one (seed, key): value -> number of pseudoseed calls."""
    __slots__ = ("x", "n", "seen")

    def __init__(self, seed: str, key: str):
        self.x = RCORE.pseudohash(key + seed)
        self.n = 0
        self.seen = {}


_CHAINS: dict = {}
KEY_POSITION_MAX = 50_000


def key_position(seed: str, key: str, value: float, limit: int = KEY_POSITION_MAX) -> Optional[int]:
    """How many ``pseudoseed(key)`` calls separate the run-start state from ``value``: the
    per-key state depends ONLY on the call count (core.PseudoRandom.pseudoseed), so this is
    the key's queue position (0 = never called).  None if not reached within ``limit``."""
    ch = _CHAINS.get((seed, key))
    if ch is None:
        ch = _CHAINS[(seed, key)] = _KeyChain(seed, key)
    n = ch.seen.get(value)
    if n is not None:
        return n
    while ch.n < limit:
        ch.x = RCORE.lcg_step(ch.x)
        ch.n += 1
        ch.seen[ch.x] = ch.n
        if ch.x == value:
            return ch.n
    return None


def positions(seed: str, state: dict) -> dict:
    return {k: key_position(seed, k, v) for k, v in state.items()}


# Key classification for the queue-alignment check (GATE_NOTES.md has the table).
#   SHARED        stepped in lockstep by both players (run structure): positions MUST be equal
#                 at the same visit ordinal.
#   OWN_SHOP      shelf generation: stepped per shelf slot -> differs only by own rerolls
#                 (+ Overstock); `cdt<a>` is exactly shop_joker_max x (visits + rerolls).
#   OWN_PACK      pack contents: stepped only when that player opens a pack.
#   OWN_RESAMPLE  `<pool>_resample<it>`: in-place redraw on a slot blocked by that player's own
#                 collection / a ban -> only the player who hit the block steps it.
#   PER_PLAYER    effect rolls / created cards / picks that depend on that player's own play.
#   VOUCHER       the run-global MLB voucher stream: SHARED unless a player owns BOTH tiers
#                 of a pair (then that player redraws -- see GATE_NOTES).
#   UNKNOWN       -> the gate fails and names the key.
_RESAMPLE_RE = re.compile(r"^(?P<pool>.+)_resample(?P<it>\d+)$")
_APPENDED_RE = re.compile(
    r"^(?P<prefix>Tarot|Planet|Spectral|Tarot_Planet|Joker[0-3]|edi|front|Enhanced|rarity\d*)"
    r"(?P<app>sho|ar1|ar2|pl1|spe|buf|sta|8ba|hal|car|vag|sup|sea|sixth|pri|emp|jud|sou|wra|rif|top|rta|uta|fool|blusl)"
    r"(?P<ante>\d*)$")
_ANTE_ONLY_RE = re.compile(r"^(?P<base>[A-Za-z_]+?)(?P<ante>\d+)$")
SHARED_BASES = {"boss", "Tag", "shuffle", "erratic", "nr", "cashout", "idol", "mail", "anc", "cas",
                "orbital", "shop_pack", "Voucher_fromtag", "seed"}
SHOP_APPENDS = {"sho"}
PACK_APPENDS = {"ar1", "ar2", "pl1", "spe", "buf", "sta"}
CREATED_APPENDS = {"8ba", "hal", "car", "vag", "sup", "sea", "sixth", "pri", "emp", "jud", "sou", "wra",
                   "rif", "top", "rta", "uta", "fool", "blusl"}
STD_PACK_BASES = {"stdset", "standard_edition", "stdseal", "stdsealtype"}
EFFECT_KEYS = {
    "lucky_mult", "lucky_money", "glass", "wheel", "wheel_of_fortune", "gros_michel", "cavendish", "8ball",
    "business", "bloodstone", "parking", "space", "misprint", "madness", "aajk", "cerulean_bell", "hook",
    "crimson_heart", "invisible", "perkeo", "to_do", "random_destroy", "marb_fr", "cert_fr", "certsl",
    "spe_card", "illusion", "omen_globe", "hex", "ankh_choice", "ectoplasm", "immolate", "sigil", "ouija",
    "aura", "familiar_create", "grim_create", "incantation_create", "Joker4", "std_create",
}
CLASSES = ("SHARED", "VOUCHER", "OWN_SHOP", "OWN_PACK", "OWN_ANY", "OWN_RESAMPLE", "PER_PLAYER", "UNKNOWN")


def classify_key(key: str) -> str:
    """Name the class of a PseudoRandom key for the queue-alignment check (table in
    GATE_NOTES.md §3).  Anything this cannot name is UNKNOWN and fails the gate."""
    if key == "Voucher0" or re.match(r"^Voucher0\d+$", key):
        return "VOUCHER"
    m = _RESAMPLE_RE.match(key)
    if m:
        base_cls = classify_key(m.group("pool"))
        if base_cls in ("SHARED", "VOUCHER"):
            return "SHARED"            # Tag<a>_resample<it> (tag_boss ban): symmetric for both
        if base_cls in ("OWN_SHOP", "OWN_PACK", "OWN_ANY", "PER_PLAYER"):
            return "OWN_RESAMPLE"
        return "UNKNOWN"
    if key in EFFECT_KEYS:
        return "PER_PLAYER"
    m = _APPENDED_RE.match(key)
    if m:
        app = m.group("app")
        if app in SHOP_APPENDS:
            return "OWN_SHOP"
        if app in PACK_APPENDS:
            return "OWN_PACK"
        if app in CREATED_APPENDS:
            return "PER_PLAYER"
        return "UNKNOWN"
    if re.match(r"^soul_(Tarot|Planet|Spectral)\d+$", key):
        return "OWN_ANY"               # no key_append: shop slots, packs AND created cards step it
    m = _ANTE_ONLY_RE.match(key)
    base = m.group("base") if m else key
    if base in SHARED_BASES or key in SHARED_BASES:
        return "SHARED"
    if base in STD_PACK_BASES:
        return "OWN_PACK"
    if base == "halu":
        return "PER_PLAYER"
    if base in ("cdt", "etperpoll", "ssjr"):
        return "OWN_SHOP"
    if base in ("packetper", "packssjr"):
        return "OWN_PACK"
    return "UNKNOWN"


def diff_rng(seed: str, s0: dict, s1: dict) -> list:
    """Per-key queue-position diff of two ``rng.snapshot()['state']`` dicts.  Returns
    ``[(key, class, pos0, pos1)]`` for every key whose position differs (missing = 0)."""
    out = []
    for k in sorted(set(s0) | set(s1)):
        v0, v1 = s0.get(k), s1.get(k)
        if v0 == v1:
            continue
        p0 = key_position(seed, k, v0) if v0 is not None else 0
        p1 = key_position(seed, k, v1) if v1 is not None else 0
        out.append((k, classify_key(k), p0, p1))
    return out


# ============================================================================ recorder

@dataclass
class ShopVisit:
    player: int
    ordinal: int                 # 0-based visit counter for this player
    ante: int
    after_blind: str             # Small / Big / Boss / Nemesis
    shelves: list = field(default_factory=list)    # [items] at entry, then after each reroll
    voucher: Optional[str] = None
    packs: list = field(default_factory=list)      # pack keys on the shelf
    opened: list = field(default_factory=list)     # [{'key':..., 'cards': [...], 'picked': [...]}]
    bought: list = field(default_factory=list)     # items bought (shelf + voucher)
    rerolls: int = 0
    rng_entry: dict = field(default_factory=dict)  # rng.snapshot()['state'] at entry
    rng_exit: dict = field(default_factory=dict)   # ... at leave_shop
    shop_joker_max: int = 2
    dollars_entry: int = 0
    owned_entry: list = field(default_factory=list)  # owned joker + consumable keys at entry (sorted)
    boss_blind: Optional[str] = None               # this ante's (shadow) boss draw at entry
    blind_tags: dict = field(default_factory=dict) # {'Small': tag, 'Big': tag} at entry


@dataclass
class CashOut:
    player: int
    ante: int
    blind: str
    is_pvp: bool
    won: bool
    dollars_before: int
    dollars_after: int
    hands_left: int
    discards_left: int
    comeback_bonus: int
    comeback_pending: bool
    blind_reward: int
    interest_expected: int
    hand_money_expected: int
    comeback_expected: int
    step: int


@dataclass
class LifeEvent:
    player: int
    ante: int
    blind: str
    is_pvp: bool
    lives_before: int
    lives_after: int
    comeback_bonus_after: int
    step: int
    cause: str                   # 'regular_fail' | 'pvp_loss' | 'deck_out' | '?'


@dataclass
class BlindResult:
    player: int
    ante: int
    blind: str
    boss_key: str
    is_pvp: bool
    target: int
    scored: int
    won: bool
    hands_used: int
    step: int


@dataclass
class PvPResult:
    ante: int
    loser: Optional[int]
    score0: int
    score1: int
    hands_left0: int
    hands_left1: int
    early_end: bool              # the winner still had hands when the server ended it
    tie: bool
    step: int


class MatchRecorder:
    """Drive an ``MLBMatch`` with two policies under the canonical turn order and record
    every gate-relevant event."""

    def __init__(self, seed: str, players, deck_key: str = "b_red", stake=1, lives: int = 4,
                 max_antes: Optional[int] = None, max_steps: int = 200_000):
        self.seed = seed
        self.specs = players
        self.policies = [make_policy(p) for p in players]
        self.m = MLBMatch(seed=seed, deck_key=deck_key, stake=stake, lives=lives)
        self.seed = self.m.seed_str
        self.max_antes = max_antes
        self.max_steps = max_steps
        self.visits: list[list[ShopVisit]] = [[], []]
        self.cashouts: list[CashOut] = []
        self.lives: list[LifeEvent] = []
        self.blinds: list[BlindResult] = []
        self.pvp: list[PvPResult] = []
        self.stopped_by_max_antes = False
        self._open: list[Optional[ShopVisit]] = [None, None]
        self._pack_open: list = [None, None]
        self._last_state = [g.state for g in self.m.games]
        self._last_lives = [g.lives for g in self.m.games]
        self._last_blind = [None, None]   # (ante, kind, is_pvp, boss_key, target) while in a blind

    # -- snapshots -------------------------------------------------------------------

    @staticmethod
    def shelf(g) -> list:
        return [item_from_engine(it) for it in g.current_shop if it.kind in SHELF_KINDS and not it.sold]

    @staticmethod
    def voucher(g) -> Optional[str]:
        for it in g.current_shop:
            if it.kind == "voucher" and not it.sold:
                return it.key
        return None

    @staticmethod
    def packs(g) -> list:
        return [it.key for it in g.current_shop if it.kind == "booster" and not it.sold]

    @staticmethod
    def blind_label(g) -> str:
        return "Nemesis" if g.current_blind.is_pvp else g.current_blind.kind

    # -- driving ---------------------------------------------------------------------

    def run(self) -> "MatchRecorder":
        m = self.m
        while not m.done and m.steps < self.max_steps:
            if self.max_antes is not None and all(g.ante > self.max_antes for g in m.games):
                self.stopped_by_max_antes = True
                break
            p = m.current_player()
            if p is None:
                raise RuntimeError(f"match wedged: nobody can act: {m.state()}")
            g = m.games[p]
            acts = m.legal_actions(p)
            a = self.policies[p](m, p, acts)
            self._before(p, a)
            n_pvp = len(m.pvp_log)
            m.step(p, a)
            self._after(p, a, n_pvp)
        return self

    def _before(self, p: int, a: dict):
        """Called after the policy chose ``a`` (a policy may have side effects such as
        ``debug_win_blind``), before ``match.step``."""
        g = self.m.games[p]
        t = a.get("type")
        self._pending_cashout = None
        self._pending_buy = None
        self._pending_reroll_cost = None
        self._observe(p, g, acting=False)          # absorb policy side effects (debug wins)
        if g.state == State.ROUND_EVAL and t == "advance":
            b = g.current_blind
            won = True if b.is_pvp else g.chips_scored >= b.chips_target
            hand_money = 0 if b.is_pvp else g.hands_left * (g.money_per_hand or HAND_PAYOUT)
            if g.discards_left > 0 and g.money_per_discard:
                hand_money += g.discards_left * g.money_per_discard
            reward = 0 if (b.kind == "Small" and g.no_small_blind_reward) else b.money_reward
            interest = 0 if g.no_interest else min(g.dollars // INTEREST_RATE, g.interest_cap)
            self._pending_cashout = CashOut(
                player=p, ante=g.ante, blind=self.blind_label(g), is_pvp=b.is_pvp, won=won,
                dollars_before=g.dollars, dollars_after=0, hands_left=g.hands_left,
                discards_left=g.discards_left, comeback_bonus=g.comeback_bonus,
                comeback_pending=not g.comeback_bonus_given, blind_reward=reward,
                interest_expected=interest, hand_money_expected=hand_money,
                comeback_expected=(0 if g.comeback_bonus_given else MLB_COMEBACK_PER_LIFE * g.comeback_bonus),
                step=self.m.steps)
        elif g.state == State.SHOP and t == "buy":
            idx = a.get("item_idx", -1)
            it = g.current_shop[idx] if 0 <= idx < len(g.current_shop) else None
            self._pending_buy = (it.kind, it.key, getattr(it, "edition", None), it) if it else None
        elif g.state == State.SHOP and t == "reroll":
            self._pending_reroll_cost = g.reroll_cost

    def _after(self, p: int, a: dict, n_pvp_before: int):
        m = self.m
        # PvP verdicts (affect both players)
        if len(m.pvp_log) > n_pvp_before:
            for entry in m.pvp_log[n_pvp_before:]:
                ante, loser, s0, s1 = entry
                g0, g1 = m.games
                winner = None if loser is None else 1 - loser
                early = winner is not None and m.games[winner].hands_left > 0
                self.pvp.append(PvPResult(ante=ante, loser=loser, score0=s0, score1=s1,
                                          hands_left0=g0.hands_left, hands_left1=g1.hands_left,
                                          early_end=early, tie=(loser is None), step=m.steps))
        for q, g in enumerate(m.games):
            self._observe(q, g, acting=(q == p), action=a)

    def _observe(self, q: int, g, acting: bool, action: Optional[dict] = None):
        """Record what changed for player ``q`` since the last observation."""
        m = self.m
        prev = self._last_state[q]
        cur = g.state
        t = (action or {}).get("type")
        # lives (processed before the blind bookkeeping is cleared)
        if g.lives != self._last_lives[q]:
            lb = self._last_blind[q]
            self.lives.append(LifeEvent(
                player=q, ante=(lb[0] if lb else g.ante), blind=(lb[1] if lb else self.blind_label(g)),
                is_pvp=bool(lb and lb[2]), lives_before=self._last_lives[q], lives_after=g.lives,
                comeback_bonus_after=g.comeback_bonus, step=m.steps,
                cause=("pvp_loss" if (lb and lb[2]) else ("deck_out" if g._hands_played_round == 0 else "regular_fail"))))
            self._last_lives[q] = g.lives
        # blind finished (SELECTING_HAND / PVP_WAIT -> ROUND_EVAL / GAME_OVER)
        if prev in (State.SELECTING_HAND, State.PVP_WAIT) and cur in (State.ROUND_EVAL, State.GAME_OVER):
            lb = self._last_blind[q]
            if cur == State.GAME_OVER and g.lives > 0:
                lb = None          # winGame cut this blind short (the opponent hit 0 lives): not a result
            if lb is not None:
                ante, kind, is_pvp, boss_key, target = lb
                tgt = g.current_blind.chips_target if is_pvp else target
                self.blinds.append(BlindResult(
                    player=q, ante=ante, blind=kind, boss_key=boss_key, is_pvp=is_pvp, target=tgt,
                    scored=g.chips_scored, won=(False if is_pvp else g.chips_scored >= tgt),
                    hands_used=g._hands_played_round, step=m.steps))
            self._last_blind[q] = None
        # a blind started (BLIND_SELECT -> SELECTING_HAND; for the other player too, via startBlind)
        if cur == State.SELECTING_HAND and prev != State.SELECTING_HAND and self._last_blind[q] is None:
            self._last_blind[q] = (g.ante, self.blind_label(g), g.current_blind.is_pvp,
                                   g.current_blind.boss_key, g.current_blind.chips_target)
        if acting:
            # cash out done (ROUND_EVAL -> SHOP): new shop visit
            if prev == State.ROUND_EVAL and cur == State.SHOP and self._pending_cashout is not None:
                co = self._pending_cashout
                co.dollars_after = g.dollars
                self.cashouts.append(co)
                v = ShopVisit(player=q, ordinal=len(self.visits[q]), ante=g.ante, after_blind=co.blind,
                              shelves=[self.shelf(g)], voucher=self.voucher(g), packs=self.packs(g),
                              rng_entry=dict(g.run_state.rng.snapshot()["state"]),
                              shop_joker_max=g.run_state.shop_joker_max, dollars_entry=g.dollars,
                              owned_entry=sorted(set(g.run_state.owned_jokers) | set(g.run_state.owned_consumables)),
                              boss_blind=g.boss_blind, blind_tags=dict(g.blind_tags))
                self.visits[q].append(v)
                self._open[q] = v
            elif self._open[q] is not None:
                v = self._open[q]
                if prev == State.SHOP and t == "reroll" and self._pending_reroll_cost is not None \
                        and g.reroll_cost > self._pending_reroll_cost:
                    v.rerolls += 1
                    v.shelves.append(self.shelf(g))
                elif prev == State.SHOP and t == "buy" and self._pending_buy is not None:
                    kind, key, ed, it = self._pending_buy
                    if cur == State.BOOSTER_OPEN:
                        self._pack_open[q] = {"key": key, "cards": [item_from_engine(c) for c in g.booster_choices],
                                              "picked": []}
                    elif it.sold:
                        v.bought.append({"kind": kind, "key": key, "edition": ed})
                elif prev == State.BOOSTER_OPEN and cur == State.SHOP:
                    po = self._pack_open[q]
                    if po is not None:
                        if t == "pick_booster":
                            po["picked"] = [po["cards"][i] for i in (action or {}).get("indices", []) if i < len(po["cards"])]
                        v.opened.append(po)
                        self._pack_open[q] = None
                elif prev == State.SHOP and cur != State.SHOP and t == "leave_shop":
                    v.rng_exit = dict(g.run_state.rng.snapshot()["state"])
                    self._open[q] = None
        self._last_state[q] = cur

    # -- views -----------------------------------------------------------------------

    def summary(self) -> dict:
        m = self.m
        return {
            "seed": self.seed, "deck": m.games[0].deck_key, "stake": m.games[0].stake,
            "players": [s.name for s in self.specs],
            "done": m.done, "winner": m.winner, "steps": m.steps,
            "stopped_by_max_antes": self.stopped_by_max_antes,
            "final": [{"ante": g.ante, "lives": g.lives, "dollars": g.dollars, "jokers": [j.key for j in g.jokers],
                       "state": g.state.name} for g in m.games],
            "pvp": [asdict(x) for x in self.pvp],
            "lives": [asdict(x) for x in self.lives],
            "cashouts": [asdict(x) for x in self.cashouts],
            "blinds": [asdict(x) for x in self.blinds],
            "visits": [[{k: v for k, v in asdict(x).items() if k not in ("rng_entry", "rng_exit")}
                        for x in vs] for vs in self.visits],
        }


# ============================================================================ trace printing

def _fmt_item(it: dict) -> str:
    k = it.get("key", "?")
    ed = it.get("edition")
    enh = it.get("enhancement")
    seal = it.get("seal")
    extra = ",".join(x for x in (ed, enh, seal) if x)
    return f"{k}[{extra}]" if extra else k


def format_trace(rec: MatchRecorder, quiet: bool = False) -> str:
    m = rec.m
    out = []
    names = [s.name for s in rec.specs]
    out.append(f"MLB match  seed={rec.seed}  deck={m.games[0].deck_key}  stake={m.games[0].stake}  "
               f"lives={m.starting_lives}  players: {names[0]} vs {names[1]}")
    if not quiet:
        antes = sorted({b.ante for b in rec.blinds} | {c.ante for c in rec.cashouts} | {v.ante for vs in rec.visits for v in vs})
        for a in antes:
            out.append(f"\n=== Ante {a} ===")
            for p in (0, 1):
                out.append(f"  -- {names[p]}")
                for b in rec.blinds:
                    if b.ante == a and b.player == p:
                        if b.is_pvp:
                            out.append(f"     Nemesis: scored {b.scored} (target = opp {b.target}), hands used {b.hands_used}")
                        else:
                            res = "WON" if b.won else "LOST"
                            out.append(f"     {b.blind:5s} {b.boss_key or '':14s} {b.scored:>8d}/{b.target:<8d} {res} hands used {b.hands_used}")
                for c in rec.cashouts:
                    if c.ante == a and c.player == p:
                        parts = [f"reward ${c.blind_reward}", f"hands ${c.hand_money_expected}", f"interest ${c.interest_expected}"]
                        if c.comeback_expected:
                            parts.append(f"COMEBACK ${c.comeback_expected}")
                        out.append(f"     cash out after {c.blind}: ${c.dollars_before} -> ${c.dollars_after}  ({', '.join(parts)})")
                for le in rec.lives:
                    if le.ante == a and le.player == p:
                        out.append(f"     LIFE LOST at {le.blind} ({le.cause}): {le.lives_before} -> {le.lives_after}, comeback counter {le.comeback_bonus_after}")
                for v in rec.visits[p]:
                    if v.ante == a:
                        shelf0 = ", ".join(_fmt_item(i) for i in v.shelves[0])
                        out.append(f"     shop #{v.ordinal} (after {v.after_blind}): shelf [{shelf0}]  voucher {v.voucher}  packs {v.packs}")
                        for k, sh in enumerate(v.shelves[1:], 1):
                            out.append(f"        reroll {k}: [{', '.join(_fmt_item(i) for i in sh)}]")
                        for po in v.opened:
                            out.append(f"        opened {po['key']}: [{', '.join(_fmt_item(i) for i in po['cards'])}] picked {[_fmt_item(i) for i in po['picked']]}")
                        for b in v.bought:
                            out.append(f"        bought {b['kind']} {b['key']}")
            for pv in rec.pvp:
                if pv.ante == a:
                    who = "TIE - nobody" if pv.tie else names[pv.loser]
                    out.append(f"  >> Nemesis verdict: {pv.score0} vs {pv.score1} -> {who} loses a life"
                               + ("  (EARLY END: winner had hands left)" if pv.early_end else ""))
    out.append("\n=== Summary ===")
    for p, g in enumerate(m.games):
        out.append(f"  {names[p]}: ante {g.ante}, lives {g.lives}, ${g.dollars}, jokers {[j.key for j in g.jokers]}, state {g.state.name}")
    if m.done:
        out.append(f"  WINNER: {names[m.winner]} (opponent reached 0 lives)  steps={m.steps}")
    elif rec.stopped_by_max_antes:
        out.append(f"  stopped at --max-antes (match not finished)  steps={m.steps}")
    else:
        out.append(f"  stopped at max_steps={rec.max_steps}")
    out.append(f"  Nemesis log: {[(x.ante, x.loser, x.score0, x.score1) for x in rec.pvp]}")
    out.append(f"  lives lost: {[(x.player, x.ante, x.blind, x.cause) for x in rec.lives]}")
    return "\n".join(out)


def alignment_report(rec: MatchRecorder) -> str:
    """Human-readable queue-alignment diff per shop-visit ordinal (entry + exit)."""
    lines = []
    v0s, v1s = rec.visits
    for i in range(min(len(v0s), len(v1s))):
        a, b = v0s[i], v1s[i]
        for tag, s0, s1 in (("entry", a.rng_entry, b.rng_entry), ("exit", a.rng_exit, b.rng_exit)):
            if not s0 or not s1:
                continue
            d = diff_rng(rec.seed, s0, s1)
            lines.append(f"visit #{i} ante {a.ante}/{b.ante} after {a.after_blind}/{b.after_blind} [{tag}]: "
                         f"{len(d)} differing key(s)")
            for k, cls, p0, p1 in d:
                lines.append(f"    {k:28s} {cls:13s} P1@{p0} P2@{p1}")
    return "\n".join(lines)


# ============================================================================ main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", default="7I4M53DL")
    ap.add_argument("--deck", default="b_red")
    ap.add_argument("--stake", default="1")
    ap.add_argument("--lives", type=int, default=4)
    ap.add_argument("--max-antes", type=int, default=None, help="stop once both players are past this ante")
    ap.add_argument("--max-steps", type=int, default=200_000)
    ap.add_argument("--quiet", action="store_true", help="summary only")
    ap.add_argument("--alignment", action="store_true", help="print the per-visit RNG key diff")
    ap.add_argument("--json", help="dump the full trace (events + visits) to this file")
    args = ap.parse_args(argv)
    stake = int(args.stake) if args.stake.isdigit() else args.stake
    rec = MatchRecorder(args.seed, [OPENER, REROLLER], deck_key=args.deck, stake=stake, lives=args.lives,
                        max_antes=args.max_antes, max_steps=args.max_steps).run()
    print(format_trace(rec, quiet=args.quiet))
    if args.alignment:
        print("\n=== Queue alignment (RNG key positions, P1 vs P2) ===")
        print(alignment_report(rec))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rec.summary(), f, indent=1, default=str)
        print(f"\ntrace written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
