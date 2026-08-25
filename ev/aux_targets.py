"""
aux_targets.py — auxiliary prediction targets recorded from rollout intermediates
(Phase 5 rev 2, W-AUX; brief §6b).

Every label costs 8 full-match rollouts and yields ONE bit.  Those rollouts already walk
dense proximal quantities the workers throw away — how much money the player has when the
next shop opens, whether the blind in front of them gets cleared, what the score margin at
the next Nemesis is.  Predicting them as auxiliary heads densifies the signal, shapes the
trunk (UNREAL-style), and turns per-head error into a diagnostic for WHERE V is wrong.

**Light instrumentation only, no extra simulation.**  This module is an *observer* plugged
into the rollout loop that already runs (``labels.rollout(..., observer_factory=...)`` and
``pairs.roll_pair(..., observer_factory=...)``).  It reads public attributes of the two
``BalatroGame``s and of ``MLBMatch.pvp_detail`` after each step; it never steps, clones or
scores anything.  Measured overhead: see AUX_NOTES §5.

Targets are recorded PER PLAYER (a label job needs both perspectives, a pair needs the
actor's) and PER WORLD, then averaged over the shared worlds by ``aggregate``.  A target a
world could not produce (no shop reached, no Nemesis resolved, …) contributes nothing; a
target NO world produced is ``None`` — strict JSON ``null``, never NaN, per PAIRS_NOTES §3.3
— and the trainer masks it.

The stored values are RAW and interpretable (dollars, lives, counts, a log10 ratio); the
trainer applies each spec's ``transform`` (log1p compression / normalisation) on load, so a
shard can always be read back and audited in game units.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import _bootstrap  # noqa: F401  (fork guard, sys.path for the engine)
from _bootstrap import State, game_keys as _gk

__all__ = [
    "AUX_VERSION", "AuxSpec", "AUX_SPECS", "AUX_NAMES", "spec_by_name", "aux_dim",
    "XMULT_JOKERS", "AuxRecorder", "make_recorder_factory", "aggregate", "empty_aux",
    "transform_value", "coverage",
]

#: bumped when the meaning of a recorded field changes (the field NAMES are the schema).
AUX_VERSION = 1

#: lives every MLB match starts with (``mlb_match.DEFAULT_LIVES``); used only to normalise
#: the lives head into [0, 1].  A run with different starting lives still trains — the head
#: just sees a differently-scaled target, which is why the RAW count is what gets stored.
LIVES_SCALE = 4.0
#: log1p denominators for the regression heads (chosen so a typical value lands in ~[0, 1]).
MONEY_SCALE = 200.0
INCOME_SCALE = 50.0
CARDS_SCALE = 20.0
TAROTS_SCALE = 10.0
#: |log10 score ratio| beyond this is clipped (a 10^6 blowout is already saturating).
MARGIN_CLIP = 6.0


# ── the xmult joker set ───────────────────────────────────────────────────────────
#
# The engine has no declarative "this joker is an xMult joker" table: an xMult joker is one
# whose scoring hook multiplies ``ScoreContext.mult_mult``.  This FROZEN list is the set of
# ``JOKER_REGISTRY`` entries whose implementation touches ``mult_mult`` (35 of 150 as of
# 2026-08-25).  ``test_aux.py::test_xmult_joker_set_matches_the_engine`` re-derives it by
# source introspection and fails if the engine drifts — the list is data, the test is the
# guard, and neither costs anything at rollout time.
XMULT_JOKERS = frozenset({
    "j_acrobat", "j_ancient", "j_baron", "j_baseball", "j_blackboard", "j_bloodstone",
    "j_caino", "j_campfire", "j_card_sharp", "j_cavendish", "j_constellation",
    "j_drivers_license", "j_duo", "j_family", "j_flower_pot", "j_glass", "j_hit_the_road",
    "j_hologram", "j_idol", "j_loyalty_card", "j_lucky_cat", "j_madness", "j_obelisk",
    "j_order", "j_photograph", "j_ramen", "j_seeing_double", "j_steel_joker", "j_stencil",
    "j_throwback", "j_tribe", "j_triboulet", "j_trio", "j_vampire", "j_yorick",
})

#: consumable keys that are Tarots (``game_keys.CONSUMABLE_SET[key] == "Tarot"``).
TAROT_KEYS = frozenset(_gk.TAROT_KEYS)

_IN_BLIND = (State.SELECTING_HAND, State.PVP_WAIT)


# ── the spec table ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AuxSpec:
    """One auxiliary head.

    ``kind``      ``"binary"`` (BCE on a soft 0..1 target) or ``"reg"`` (MSE).
    ``dim``       output width (all but ``lives_2antes`` are 1).
    ``transform`` raw stored value -> the value the head regresses (per component).
    ``weight``    default loss weight (brief §6b.2: "defaults ~0.1").
    """
    name: str
    dim: int
    kind: str
    transform: Callable
    weight: float
    doc: str

    @property
    def is_binary(self) -> bool:
        return self.kind == "binary"


def _log1p_scaled(scale: float) -> Callable:
    d = math.log1p(scale)

    def f(x: float) -> float:
        return math.log1p(max(0.0, float(x))) / d
    return f


def _identity(x: float) -> float:
    return float(x)


def _over(scale: float) -> Callable:
    def f(x: float) -> float:
        return float(x) / scale
    return f


AUX_SPECS: tuple = (
    AuxSpec("money_next_shop", 1, "reg", _log1p_scaled(MONEY_SCALE), 0.1,
            "dollars held the first time the player ENTERS a shop after this state "
            "(a shop state's own shop does not count — the next one does)"),
    AuxSpec("lives_2antes", 2, "reg", _over(LIVES_SCALE), 0.1,
            "[own lives, opponent lives] when the player's ante first reaches ante0+2, "
            "or the terminal lives if the match/rollout ends first"),
    AuxSpec("pvp_margin_next", 1, "reg", _over(MARGIN_CLIP), 0.1,
            "log10((1+my score)/(1+their score)) at the next Nemesis resolved after this "
            "state, clipped to +-6"),
    AuxSpec("blind_cleared", 1, "binary", _identity, 0.1,
            "1 if the blind in progress (or the next blind actually played) ends without "
            "this player losing a life"),
    AuxSpec("xmult_by_ante4", 1, "binary", _identity, 0.1,
            "1 if the player holds an xMult joker at any point up to the end of ante 4 "
            "(masked when the state is already past ante 4)"),
    AuxSpec("extract_income", 1, "reg", _log1p_scaled(INCOME_SCALE), 0.1,
            "dollars gained DURING hand play (steps taken from SELECTING_HAND) over the "
            "remainder of the current ante — the proc/sandbag income signal"),
    AuxSpec("cards_modified", 1, "reg", _log1p_scaled(CARDS_SCALE), 0.1,
            "playing cards whose (enhancement, edition, seal) changed, plus cards added to "
            "the deck, over the remainder of the current ante"),
    AuxSpec("tarots_used", 1, "reg", _log1p_scaled(TAROTS_SCALE), 0.1,
            "Tarot consumables used by this player over the remainder of the current ante"),
)

AUX_NAMES: tuple = tuple(s.name for s in AUX_SPECS)
_BY_NAME = {s.name: s for s in AUX_SPECS}


def spec_by_name(name: str) -> AuxSpec:
    return _BY_NAME[name]


def aux_dim(specs: Sequence[AuxSpec] = AUX_SPECS) -> int:
    return sum(s.dim for s in specs)


def transform_value(spec: AuxSpec, raw):
    """Raw stored value -> the head's target (scalar or list of ``spec.dim``)."""
    if spec.dim == 1:
        return spec.transform(raw if not isinstance(raw, (list, tuple)) else raw[0])
    return [spec.transform(v) for v in raw]


def empty_aux() -> dict:
    """Every field masked (``None``) — what a rollout that recorded nothing yields."""
    return {n: None for n in AUX_NAMES}


# ── the observer ──────────────────────────────────────────────────────────────────

class _Track:
    """Per-player accumulator.  Deliberately flat and attribute-cheap: this runs after
    every step of every rollout of every world."""
    __slots__ = ("p", "ante0", "prev_state", "prev_dollars", "prev_cons",
                 "money_shop", "lives2", "margin", "cleared",
                 "xmult_seen", "xmult", "income", "cards", "tarots",
                 "blind_open", "blind_lives0", "ante_open", "deck0")

    def __init__(self, p: int):
        self.p = p
        self.ante0 = 0
        self.prev_state = None
        self.prev_dollars = 0
        self.prev_cons: list = []
        self.money_shop = None
        self.lives2 = None
        self.margin = None
        self.cleared = None
        self.xmult_seen = False
        self.xmult = None
        self.income = 0.0
        self.cards = 0
        self.tarots = 0
        self.blind_open = False
        self.blind_lives0 = 0
        self.ante_open = True
        self.deck0: dict = {}


def _deck_signature(g) -> dict:
    """``{card id: (enhancement, edition, seal)}`` over the full deck (52-ish entries)."""
    return {c.id: (c.enhancement, c.edition, c.seal) for c in g.full_deck}


def _count_modified(before: dict, after: dict) -> int:
    n = 0
    for cid, sig in after.items():
        old = before.get(cid)
        if old is None or old != sig:
            n += 1
    return n


class AuxRecorder:
    """Rollout observer: ``start(match)`` -> ``after(match, actor, action)`` per step ->
    ``finish(match)`` -> ``result()``.

    ``result()`` is ``{player: {spec name: raw value or None}}``.  Nothing here mutates the
    match; every read is a public attribute the encoder already reads."""

    def __init__(self, players: Sequence[int] = (0, 1)):
        self.players = tuple(int(p) for p in players)
        self.tracks = {p: _Track(p) for p in self.players}
        self._pvp0 = 0
        self._started = False

    # ── lifecycle ──
    def start(self, match) -> None:
        self._pvp0 = len(getattr(match, "pvp_detail", ()) or ())
        for p in self.players:
            g = match.games[p]
            t = self.tracks[p]
            t.ante0 = int(g.ante)
            t.prev_state = g.state
            t.prev_dollars = int(g.dollars)
            t.prev_cons = list(g.consumable_hand)
            t.deck0 = _deck_signature(g)
            t.blind_open = g.state in _IN_BLIND
            t.blind_lives0 = int(g.lives)
            t.xmult_seen = t.ante0 <= 4 and any(j.key in XMULT_JOKERS for j in g.jokers)
        self._started = True

    def after(self, match, actor: int, action) -> None:
        pvp = getattr(match, "pvp_detail", None)
        new_pvp = pvp[self._pvp0] if (pvp is not None and len(pvp) > self._pvp0) else None
        for p in self.players:
            self._update(match, p, actor, action, new_pvp)

    def finish(self, match) -> None:
        for p in self.players:
            self._close(match, p)

    # ── per-player update (one step) ──
    def _update(self, match, p: int, actor: int, action, new_pvp) -> None:
        t = self.tracks[p]
        g = match.games[p]
        st = g.state
        dollars = int(g.dollars)
        ante = int(g.ante)

        # 1. money at the next SHOP ENTRY (a transition INTO the shop)
        if t.money_shop is None and st == State.SHOP and t.prev_state != State.SHOP:
            t.money_shop = float(dollars)

        # 2. own + opponent lives at the end of the next 2 antes
        if t.lives2 is None and ante >= t.ante0 + 2:
            t.lives2 = [float(g.lives), float(match.games[1 - p].lives)]

        # 3. log-score margin at the next resolved Nemesis
        if t.margin is None and new_pvp is not None:
            s0, s1 = float(new_pvp[2]), float(new_pvp[3])
            mine, theirs = (s0, s1) if p == 0 else (s1, s0)
            m = math.log10(1.0 + max(0.0, mine)) - math.log10(1.0 + max(0.0, theirs))
            t.margin = max(-MARGIN_CLIP, min(MARGIN_CLIP, m))

        # 4. the current / next blind cleared (a life lost inside it = not cleared)
        in_blind = st in _IN_BLIND
        if t.cleared is None:
            if in_blind and not t.blind_open:
                t.blind_open = True
                t.blind_lives0 = int(g.lives)
            elif t.blind_open and not in_blind:
                t.cleared = 1.0 if int(g.lives) >= t.blind_lives0 else 0.0
                t.blind_open = False

        # 5. an xMult joker owned by the end of ante 4
        if t.xmult is None and t.ante0 <= 4:
            if not t.xmult_seen and ante <= 4 and g.jokers:
                t.xmult_seen = any(j.key in XMULT_JOKERS for j in g.jokers)
            if ante > 4:
                t.xmult = 1.0 if t.xmult_seen else 0.0

        # 6-8. the remainder-of-this-ante window
        if t.ante_open:
            if t.prev_state == State.SELECTING_HAND and dollars > t.prev_dollars:
                t.income += float(dollars - t.prev_dollars)
            if (actor == p and isinstance(action, dict)
                    and action.get("type") == "use_consumable"):
                i = int(action.get("consumable_idx", 0))
                if 0 <= i < len(t.prev_cons) and t.prev_cons[i] in TAROT_KEYS:
                    t.tarots += 1
            if ante > t.ante0:
                t.cards = _count_modified(t.deck0, _deck_signature(g))
                t.ante_open = False

        t.prev_state = st
        t.prev_dollars = dollars
        t.prev_cons = list(g.consumable_hand)

    def _close(self, match, p: int) -> None:
        """Anything still open when the rollout ends takes its terminal value."""
        t = self.tracks[p]
        g = match.games[p]
        if t.lives2 is None:
            t.lives2 = [float(g.lives), float(match.games[1 - p].lives)]
        if t.cleared is None and t.blind_open and (match.done or g.state == State.GAME_OVER):
            t.cleared = 1.0 if int(g.lives) >= t.blind_lives0 else 0.0
        if t.xmult is None and t.ante0 <= 4:
            t.xmult = 1.0 if t.xmult_seen else 0.0
        if t.ante_open:
            t.cards = _count_modified(t.deck0, _deck_signature(g))
            t.ante_open = False

    # ── output ──
    def result(self) -> dict:
        if not self._started:
            return {p: empty_aux() for p in self.players}
        out = {}
        for p in self.players:
            t = self.tracks[p]
            out[p] = {
                "money_next_shop": t.money_shop,
                "lives_2antes": t.lives2,
                "pvp_margin_next": t.margin,
                "blind_cleared": t.cleared,
                "xmult_by_ante4": t.xmult,
                "extract_income": float(t.income),
                "cards_modified": float(t.cards),
                "tarots_used": float(t.tarots),
            }
        return out


def make_recorder_factory(players: Sequence[int] = (0, 1)) -> Callable:
    """``factory() -> AuxRecorder`` — one fresh recorder per rollout, which is what
    ``labels.rollout`` / ``pairs.roll_pair`` want (``observer_factory=``)."""
    ps = tuple(int(p) for p in players)
    return lambda: AuxRecorder(ps)


# ── aggregation over the shared worlds ────────────────────────────────────────────

def aggregate(per_world: Sequence[Optional[dict]], player: int) -> dict:
    """Mean over the worlds that produced each field; ``None`` where none did.

    ``per_world`` is the list of ``AuxRecorder.result()`` dicts, one per rollout (a
    ``None`` entry — a rollout run without the observer — is skipped).  Strict JSON out:
    never NaN, never Inf (PAIRS_NOTES §3.3)."""
    out: dict = {}
    for spec in AUX_SPECS:
        vals = []
        for res in per_world:
            if not res:
                continue
            v = res.get(player, {}).get(spec.name)
            if v is None:
                continue
            vals.append([float(v)] if spec.dim == 1 else [float(x) for x in v])
        if not vals:
            out[spec.name] = None
            continue
        n = len(vals)
        means = [sum(v[i] for v in vals) / n for i in range(spec.dim)]
        if any(not math.isfinite(m) for m in means):
            out[spec.name] = None
            continue
        out[spec.name] = round(means[0], 6) if spec.dim == 1 else [round(m, 6) for m in means]
    return out


def coverage(aux_dicts: Sequence[Optional[dict]]) -> dict:
    """``{name: fraction of records with the field present}`` — the masking diagnostic."""
    n = len(aux_dicts) or 1
    return {s.name: sum(1 for d in aux_dicts if d and d.get(s.name) is not None) / n
            for s in AUX_SPECS}
