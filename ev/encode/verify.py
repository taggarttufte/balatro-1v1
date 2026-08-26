"""verify.py — the empirical harness that decides whether a registry entry is true
(W-ENCODE-POC, 2026-08-26).

THE MARGINAL RULE (the #1 designed-in risk)
-------------------------------------------
Every measurement here is the item's **marginal** realized effect with *everything else
active*.  Two arms are built from the SAME state; they differ only by the presence of the
item; every other joker, voucher, seal and blind reward is live in both.  The measurement
is ``arm_with - arm_without``.

That is not a stylistic choice.  The encode layer sits next to a dry-run scorer that
already prices chips/mult exactly (``HypotheticalScorer``, EV_NOTES §1) and a proc-EV layer
that already prices nine money procs (EXTRACT_NOTES §2).  The failure mode that would
silently wreck the player is an entry that re-prices something already priced -- and a
marginal measurement makes that fail LOUDLY, because the marginal realized dollars of an
already-priced scoring effect are exactly zero.  ``registry.j_joker__doublecount`` is the
standing proof.

REACHABILITY (the A1 lesson)
----------------------------
"The number matched" is not enough: a predictor of 0 matches a hook that never fired.  Every
mode installs a ``ReachProbe`` around the item's engine effect and records how many times
each hook actually ran and how much pending money it wrote.  An entry that predicts a
non-zero value while its hook never fired is rejected with ``reason='unreachable'``,
regardless of the arithmetic.

MODES
-----
``round_end_paired``    two clones of one constructed in-blind state, +/- the item, both run
                        through the real ``BalatroGame._end_round``.  Deterministic items
                        land exactly; the CI is the sample sd over scenarios.
``use_paired``          two clones, one uses the consumable through the real
                        ``use_consumable`` action, the other does not.
``scaling_trajectory``  real ``EVPlayer(budget='fast')`` play from a real seed for a fixed
                        number of SCORING HANDS, then read the joker's internal scaling
                        state.  Also measures the policy rates the entry needs
                        (discards per hand, face-hand rate) -- those rates are the fleet's
                        real deliverable and they cannot be read out of the Lua.
``rollout_paired``      with-vs-without on the SAME ``clone_determinized`` worlds (the
                        repo's CRN machinery), played to a stop ante by ``ev:fast``.
                        Reports realized gross payout (instrumented), realized delta-money
                        and realized delta-ante-progress.
"""
from __future__ import annotations

import math
import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

_HERE = Path(__file__).resolve().parent            # ev/encode
_EV = _HERE.parent                                  # ev
_ROOT = _EV.parent                                  # repo root
for _p in (str(_EV), str(_ROOT), str(_ROOT / "eval"), str(_ROOT / "agent")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _bootstrap  # noqa: E402,F401  (fork guard; raises loudly on the wrong balatro_sim)
from _bootstrap import BalatroGame, State  # noqa: E402
from balatro_sim.jokers.base import JOKER_REGISTRY, JokerInstance  # noqa: E402
from balatro_sim import consumables as _consumables  # noqa: E402

MARGINAL_RULE = (
    "Every number in this module is a MARGINAL measurement: two arms built from one state, "
    "differing only by the item, with every other effect active in both.  Anything the "
    "player already prices therefore measures as zero here, by construction."
)

#: 2x magnitude band, per the brief.  A prediction must sit within [measured/2, measured*2].
BAND = 2.0
#: absolute slack, in the entry's own unit, before the band/CI checks bite.  Dollars and
#: mult are integers in the engine, so this is only float noise.
TOL_ABS = 1e-6
#: default budgets — deliberately modest (a GPU chain and 12-worker tournaments share the box)
DEFAULT_TRAJ_SEEDS = 40
DEFAULT_ROLLOUT_WORLDS = 32
DEFAULT_WORKERS = 4
MAX_WORKERS = 6                  # ops cap for this workstream


# ─────────────────────────────────────────────────────────────── state construction
# The fixture idiom is ev/fixtures/_probe_common.py's, reduced to a solo BalatroGame:
# real engine APIs to reach the state, then direct attribute writes on real Card objects
# pulled out of game.full_deck (so full_deck/hand/deck keep sharing references).

def in_blind(seed: str = "11111111", *, deck_key: str = "b_red", stake: int = 1,
             ruleset: str = "vanilla") -> BalatroGame:
    """A real ``BalatroGame`` stepped through ``play_blind`` — at ``SELECTING_HAND`` with
    the engine's own hands/discards/target."""
    g = BalatroGame(seed=seed, deck_key=deck_key, stake=stake, ruleset=ruleset)
    g.step({"type": "play_blind"})
    if g.state != State.SELECTING_HAND:
        raise RuntimeError(f"in_blind({seed!r}) landed at {g.state}, not SELECTING_HAND")
    return g


def set_hand(g: BalatroGame, specs: Sequence[tuple]) -> BalatroGame:
    """Exactly ``specs`` in hand, taken out of the draw pile (test_extraction.py's
    ``_set_hand``)."""
    pool: dict = {}
    for c in g.full_deck:
        pool.setdefault((c.rank, c.suit), c)
    hand = [pool[s] for s in specs]
    ids = {id(c) for c in hand}
    g.deck = [c for c in g.full_deck if id(c) not in ids]
    g.hand = hand
    g.discard_pile = []
    for c in hand:
        c.face_down = False
        c.debuffed = False
    return g


def add_joker(g: BalatroGame, key: str) -> JokerInstance:
    """``debug_add_joker`` — slot bookkeeping, ``run_state.acquire`` and ``on_init``
    (shop.emplace_joker), i.e. everything a purchase would do."""
    return g.debug_add_joker(key)


def summarize(g: BalatroGame, **extra) -> dict:
    """The canonical state summary every ``Entry.predict`` reads.

    ONE function builds it for every entry, and each entry reads only the fields it needs.
    That is the interface that lets a registry entry stay engine-free: the fleet writes pure
    functions of this dict, and the harness owns the (engine-coupled) job of filling it in.
    ``extra`` carries whatever the scenario adds — the policy-conditional horizons
    (``hands_ahead`` / ``discards_ahead`` / ``face_hand_rate``) have no reading in a static
    state and are supplied by the caller.
    """
    jokers = {j.key: j for j in g.jokers}
    rocket = jokers.get("j_rocket")
    rtb = jokers.get("j_ride_the_bus")
    green = jokers.get("j_green_joker")
    ice = jokers.get("j_ice_cream")
    s = {
        "dollars": int(g.dollars),
        "ante": int(g.ante),
        "blind_is_boss": bool(getattr(g.current_blind, "is_boss", False)),
        "blind_kind": getattr(g.current_blind, "kind", ""),
        "hands_left": int(g.hands_left),
        "discards_left": int(g.discards_left),
        "interest_cap": int(getattr(g, "interest_cap", 5)),
        "deck_nines": sum(1 for c in g.full_deck if c.rank == 9),
        "deck_size": len(g.full_deck),
        "unique_planets_used": len(set(getattr(g, "planets_used", []))),
        "joker_keys": tuple(jokers),
        "rocket_dollars": int(rocket.state.get("bonus", 1)) if rocket else 1,
        "rtb_mult": float(rtb.state.get("mult", 0)) if rtb else 0.0,
        "green_mult": float(green.state.get("mult", 0)) if green else 0.0,
        "ice_chips": float(ice.state.get("chips", 100)) if ice else 100.0,
    }
    s.update(extra)
    return s


def joker_of(g: BalatroGame, key: str) -> Optional[JokerInstance]:
    for j in g.jokers:
        if j.key == key:
            return j
    return None


def set_deck_nines(g: BalatroGame, n: int) -> int:
    """Force the full deck to hold exactly ``n`` cards of rank 9 by re-ranking spare cards.

    Rank is a plain attribute on the shared ``Card`` objects, so this is the same kind of
    direct write the W-PROBE fixtures use for seals and enhancements.  Returns the count.
    """
    nines = [c for c in g.full_deck if c.rank == 9]
    others = [c for c in g.full_deck if c.rank != 9 and c.enhancement != "Stone"]
    while len(nines) > n and nines:
        c = nines.pop()
        c.rank = 8
    i = 0
    while len(nines) < n and i < len(others):
        others[i].rank = 9
        nines.append(others[i])
        i += 1
    got = sum(1 for c in g.full_deck if c.rank == 9)
    if got != n:
        raise RuntimeError(f"set_deck_nines: wanted {n}, got {got}")
    return got


def make_boss_blind(g: BalatroGame) -> None:
    """Mark the current blind a Boss so the end-of-round boss hooks fire.  ``_end_round``
    reads ``blind.is_boss`` and nothing else about bossness for the hook path."""
    g.current_blind.is_boss = True


def use_planets(g: BalatroGame, keys: Sequence[str]) -> None:
    """Use planets through the REAL consumable path (``consumables.apply_planet``), so
    every ``on_planet_used`` hook and ``game.planets_used`` see them."""
    for k in keys:
        if not _consumables.apply_planet(g, k):
            raise RuntimeError(f"apply_planet({k!r}) refused")


# ───────────────────────────────────────────────────────────────── reachability probe

@dataclass
class ReachRecord:
    key: str
    calls: dict = field(default_factory=dict)     # hook name -> times called
    money_written: float = 0.0                    # pending_money the hooks wrote

    def fired(self, hook: Optional[str] = None) -> int:
        if hook is None:
            return sum(self.calls.values())
        return self.calls.get(hook, 0)


class _Proxy:
    """Delegates every hook to the real effect object and records that it ran.

    ``fire_hook`` looks the effect up by key in the global ``JOKER_REGISTRY``
    (jokers/base.py:495), so swapping the entry is the only way to see a hook fire without
    editing the engine — which this workstream is not allowed to do.  Restored in a
    ``finally``; the swap is process-local and lives inside one context manager.
    """

    def __init__(self, real, rec: ReachRecord):
        self._real = real
        self._rec = rec

    def __getattr__(self, name):
        attr = getattr(self._real, name)
        if not callable(attr) or name.startswith("_"):
            return attr

        def wrapped(inst, *a, **kw):
            self._rec.calls[name] = self._rec.calls.get(name, 0) + 1
            before = float(inst.state.get("pending_money", 0) or 0)
            out = attr(inst, *a, **kw)
            after = float(inst.state.get("pending_money", 0) or 0)
            self._rec.money_written += max(0.0, after - before)
            return out

        return wrapped


@contextmanager
def reach_probe(key: str):
    """Instrument one joker key's hooks for the duration of the block.

    ``JOKER_REGISTRY`` is a ``_JokerRegistry`` that refuses a second ``__setitem__`` for a
    key (jokers/base.py:16-28) — a deliberate guard against the pre-Phase-1 "last import
    wins" bug, and one this harness must not weaken.  So the swap goes through
    ``dict.__setitem__`` explicitly: it is a temporary, process-local, ``finally``-restored
    instrument, not a registration, and going around the guard by name makes that visible in
    the diff instead of hiding it behind a relaxed guard.
    """
    rec = ReachRecord(key=key)
    real = JOKER_REGISTRY.get(key)
    if real is None:
        yield rec
        return
    dict.__setitem__(JOKER_REGISTRY, key, _Proxy(real, rec))
    try:
        yield rec
    finally:
        dict.__setitem__(JOKER_REGISTRY, key, real)


# ────────────────────────────────────────────────────────────────────── measurement

@dataclass
class Measurement:
    """One entry x one mode.  ``measured`` is always the MARGINAL realized quantity.

    Three independent gates have to pass, and they catch different lies:

    ``reachable``   the item's hook actually ran.  Catches "the number is right because
                    nothing happened" — the project's A1 lesson.
    ``within_ci``   the mean PER-SCENARIO RESIDUAL (predicted_i - measured_i) is within its
                    own 95% band of zero.  Note this gate is weak on a scenario family that
                    spans a wide range: a 3x error inflates the residual spread as fast as
                    it inflates the residual, so the CI can swallow it.  That is why the
                    band gate exists and is not redundant.
    ``within_band`` the 2x magnitude class on the aggregate.  Catches scale errors and, via
                    the zero-aware case, every double count of an already-priced effect.
    ``exact``       deterministic entries only: EVERY scenario must land on the nose.  A
                    closed form that is only right on average is not a closed form.
    """

    key: str
    mode: str
    n: int
    predicted: float
    measured: float
    ci: float                       # 95% half-width on the mean per-scenario RESIDUAL
    unit: str
    fired: int = 0                  # hook invocations recorded by the probe
    needs_fire: bool = True
    residual: float = 0.0           # mean(predicted_i - measured_i)
    exact: bool = True              # every scenario landed within TOL_ABS
    exactness_required: bool = False
    #: False when the mode produces a measurement the entry makes no claim about (the
    #: empirical buy value of a voucher, say).  An unscored row is INFO, never a REJECT:
    #: reporting "no claim" as a failure would inflate the reject count with noise.
    scored: bool = True
    seconds: float = 0.0
    extra: dict = field(default_factory=dict)
    scenarios: list = field(default_factory=list)   # per-scenario (predicted, measured)

    # ── the accept rule ────────────────────────────────────────────────────────
    @property
    def within_ci(self) -> bool:
        return abs(self.residual) <= self.ci + TOL_ABS

    @property
    def within_band(self) -> bool:
        """The 2x magnitude class the brief asks for, with a zero-aware special case."""
        p, m = self.predicted, self.measured
        if abs(m) <= TOL_ABS:
            return abs(p) <= TOL_ABS
        if abs(p) <= TOL_ABS:
            return abs(m) <= TOL_ABS
        if (p > 0) != (m > 0):
            return False                        # a sign error is never a magnitude class
        r = abs(p) / abs(m)
        return (1.0 / BAND) - 1e-9 <= r <= BAND + 1e-9

    @property
    def reachable(self) -> bool:
        if not self.needs_fire:
            return True
        if abs(self.predicted) <= TOL_ABS and abs(self.measured) <= TOL_ABS:
            return True                          # nothing claimed, nothing needed
        return self.fired > 0

    @property
    def exact_ok(self) -> bool:
        return self.exact or not self.exactness_required

    @property
    def accept(self) -> bool:
        if not self.scored:
            return True
        return self.reachable and self.within_ci and self.within_band and self.exact_ok

    @property
    def reason(self) -> str:
        if not self.scored:
            return "info (no claim to score)"
        if self.accept:
            return "ok"
        bad = []
        if not self.reachable:
            bad.append("unreachable")
        if not self.within_band:
            bad.append("band")
        if not self.within_ci:
            bad.append("ci")
        if not self.exact_ok:
            bad.append("inexact")
        return "+".join(bad)

    def row(self) -> str:
        v = ("INFO  " if not self.scored else ("ACCEPT" if self.accept else "REJECT"))
        return (f"{self.key:<22} {self.mode:<19} n={self.n:<4} "
                f"pred={self.predicted:8.3f} meas={self.measured:8.3f} "
                f"resid={self.residual:+7.3f} +/-{self.ci:6.3f} {self.unit:<7} "
                f"fired={self.fired:<5} {v} ({self.reason})")


def reset_player_caches() -> None:
    """Clear ``ev/hand.py``'s module-level caches before a run.

    **Why this is not paranoia.**  ``hand._RATIO_CACHE`` is a process-global dict keyed by
    ``_board_sig``, which DELIBERATELY omits planet levels and the exact deck composition
    (hand.py:345-358 — a documented speed trade: "a planet pick must not force a ratio
    recompute").  ``board_ratio`` itself samples real hands from the real deck at the run's
    real planet levels, so two states that differ only in those omitted fields share a cache
    entry and the FIRST one computed wins.

    Consequence: ``ev:fast`` is deterministic given ``(seed, budget)`` only within a COLD
    process.  Run several seeds in one worker and a seed's result depends on which seeds
    preceded it — measured at **2 of 24 seeds (8%)** changing trajectory, and isolated to
    ``_RATIO_CACHE`` alone (``_MODEL_CACHE`` shared across runs changes nothing).  Any pool
    that reuses workers therefore gets partition-dependent per-seed numbers.

    A verification harness cannot live with that: a measurement must not depend on what the
    worker did before it.  ``ev/hand.py`` is read-only to this workstream, so the harness
    resets the cache at every run boundary instead.  See POC_NOTES §3.5.
    """
    try:
        import hand as _H
        _H._RATIO_CACHE.clear()
        _H._MODEL_CACHE.clear()
    except Exception:                       # the caches are an implementation detail
        pass


def _counts(it: Iterable) -> dict:
    out: dict = {}
    for x in it:
        out[x] = out.get(x, 0) + 1
    return out


def _mean_ci(xs: Sequence[float]) -> tuple:
    n = len(xs)
    if n == 0:
        return float("nan"), float("inf")
    mean = sum(xs) / n
    if n == 1:
        return mean, 0.0
    sd = statistics.stdev(xs)
    return mean, 1.96 * sd / math.sqrt(n)


# ───────────────────────────────────────────────────────── mode A: round_end_paired

@dataclass
class RoundScenario:
    """One constructed state + the summary the entry's ``predict`` will see."""

    name: str
    build: Callable[[], BalatroGame]           # a fresh in-blind game WITHOUT the item
    install: Callable[[BalatroGame], None]     # add the item to that game
    summarize: Callable[[BalatroGame], dict]   # summary of the WITH arm, post-install
    post_install: Optional[Callable] = None    # anything that must happen AFTER install
    #: reachability signal for items with no joker hook (vouchers, deck modifiers):
    #: ``(with_arm, without_arm) -> int``.  ``None`` falls back to the joker probe.
    reach: Optional[Callable] = None


def measure_round_end(entry, scenarios: Sequence[RoundScenario]) -> Measurement:
    """Paired ``_end_round`` on two identically-built states, +/- the item.

    Both arms pay their blind reward, unused-hand money, gold-card rows and interest; the
    difference is exactly what the item added.  MARGINAL RULE: if the item's effect were
    already priced elsewhere in the engine the difference would be 0 and the entry would be
    rejected — that is the double-count guard, not a side effect of one.
    """
    t0 = time.perf_counter()
    preds, meas, res, rows = [], [], [], []
    fired = 0
    for sc in scenarios:
        with_arm = sc.build()
        sc.install(with_arm)
        if sc.post_install is not None:
            sc.post_install(with_arm)
        summary = sc.summarize(with_arm)
        p = float(entry.predict(summary))

        without_arm = sc.build()               # rebuilt, not cloned: identical construction
        if sc.post_install is not None:
            sc.post_install(without_arm)       # the WITHOUT arm gets everything but the item
        with reach_probe(entry.engine_key) as rec:
            d_with = _end_round_delta(with_arm)
        d_without = _end_round_delta(without_arm)
        fired += rec.fired() if sc.reach is None else int(sc.reach(with_arm, without_arm))
        m = d_with - d_without
        preds.append(p)
        meas.append(m)
        res.append(p - m)
        rows.append({"scenario": sc.name, "predicted": p, "measured": m,
                     "with": d_with, "without": d_without,
                     "hook_calls": dict(rec.calls), "hook_money": rec.money_written})
    return _assemble(entry, "round_end_paired", preds, meas, res, rows, fired,
                     time.perf_counter() - t0)


def _assemble(entry, mode, preds, meas, res, rows, fired, seconds, extra=None) -> Measurement:
    """Common tail of the deterministic modes: residual statistics + the exactness gate."""
    n = len(meas)
    mean_p = sum(preds) / n if n else float("nan")
    mean_m = sum(meas) / n if n else float("nan")
    r_mean, r_ci = _mean_ci(res)
    exact = all(abs(x) <= TOL_ABS for x in res)
    return Measurement(
        key=entry.key, mode=mode, n=n, predicted=mean_p, measured=mean_m,
        ci=r_ci, residual=r_mean, unit=entry.unit, fired=fired,
        exact=exact, exactness_required=(entry.tier == "deterministic"),
        seconds=seconds, scenarios=rows,
        extra={"per_scenario_exact": exact,
               "worst_scenario": (max(rows, key=lambda r: abs(r["predicted"] - r["measured"]))
                                  if rows else None),
               **(extra or {})})


def _end_round_delta(g: BalatroGame) -> float:
    before = g.dollars
    g._end_round()
    return float(g.dollars - before)


# ──────────────────────────────────────────────────────────────── mode B: use_paired

@dataclass
class UseScenario:
    name: str
    build: Callable[[], BalatroGame]
    consumable: str
    summarize: Callable[[BalatroGame], dict]


def measure_use(entry, scenarios: Sequence[UseScenario]) -> Measurement:
    """Two clones of one state; one uses the consumable through the real
    ``use_consumable`` action, the other does not.  The difference is the use's value."""
    t0 = time.perf_counter()
    preds, meas, res, rows = [], [], [], []
    for sc in scenarios:
        g = sc.build()
        control = sc.build()                       # identical state, does NOT use the card
        g.consumable_hand.append(sc.consumable)
        summary = sc.summarize(g)
        p = float(entry.predict(summary))
        idx = len(g.consumable_hand) - 1
        acts = [a for a in g.legal_actions()
                if a.get("type") == "use_consumable" and a.get("consumable_idx") == idx]
        if not acts:
            raise RuntimeError(f"{sc.name}: {sc.consumable} is not usable in this state")
        before = g.dollars
        g.step(acts[0])
        m = float(g.dollars - before) - float(control.dollars - before)   # marginal
        used = sc.consumable in getattr(g, "tarots_used", [])
        preds.append(p)
        meas.append(m)
        res.append(p - m)
        rows.append({"scenario": sc.name, "predicted": p, "measured": m,
                     "used": used, "dollars_before": before, "dollars_after": g.dollars})
    return _assemble(entry, "use_paired", preds, meas, res, rows,
                     sum(1 for r in rows if r["used"]), time.perf_counter() - t0)


# ────────────────────────────────────────────────────── mode C: scaling_trajectory

_SCALE_FIELD = {"j_ride_the_bus": "mult", "j_green_joker": "mult", "j_ice_cream": "chips"}


@dataclass
class TrajResult:
    seed: str
    hands: int
    discards: int
    face_hands: int
    value: float                # the joker's scaling state after `hands` scoring hands
    start: float
    steps: int
    reached: bool
    lost: bool = False          # the joker left the board before the horizon
    lost_at_hand: int = -1
    lost_reason: str = ""       # "sold" | "removed" (Ice Cream's melt would be "removed")
    survived: bool = True


def trajectory(seed: str, joker_key: str, hands_ahead: int, *, budget: str = "fast",
               max_steps: int = 4000) -> TrajResult:
    """Play a real vanilla run from ``seed`` with ``EVPlayer`` until ``hands_ahead``
    SCORING hands have been played, then read the joker's scaling state.

    The joker is installed at the ante-1 blind select through ``debug_add_joker`` (the
    purchase path), so everything downstream — the shop, the packs, the boss — is the real
    policy's, not a fixture's.  Returns the realized policy rates alongside the value: those
    rates are exactly what a Lua reading cannot supply.

    **The joker can leave the board mid-run.**  The current shop rules see no immediate
    strength on a scaling joker (EV_NOTES §8 item 4) and will happily sell it, which is the
    very blind spot this package exists to close.  When that happens the trajectory is
    truncated, the last observed value is kept, and ``lost``/``lost_at_hand`` record it —
    the survival rate is reported rather than silently averaged in.
    """
    from player import EVPlayer                                    # ev/player.py, read-only
    reset_player_caches()
    g = BalatroGame(seed=seed, deck_key="b_red", stake=1, ruleset="vanilla")
    inst = g.debug_add_joker(joker_key)
    fld = _SCALE_FIELD[joker_key]
    start = float(inst.state.get(fld, 0))
    pl = EVPlayer(budget=budget, seed=0)
    hands = discards = face_hands = steps = 0
    value = start
    lost = False
    lost_at = -1
    lost_reason = ""
    while g.state != State.GAME_OVER and steps < max_steps and hands < hands_ahead:
        a = pl.act(g)
        if a.get("type") == "play":
            scoring = _scoring_cards(g, a)
            if any(_is_face(c) for c in scoring):
                face_hands += 1
            hands += 1
        elif a.get("type") == "discard":
            discards += 1
        was_sale = a.get("type") in ("sell_joker", "sell")
        g.step(a)
        steps += 1
        live = joker_of(g, joker_key)
        if live is None:
            lost, lost_at = True, hands
            lost_reason = "sold" if was_sale else "removed"
            break
        value = float(live.state.get(fld, 0))
    return TrajResult(seed=seed, hands=hands, discards=discards, face_hands=face_hands,
                      value=value, start=start, steps=steps,
                      reached=(hands >= hands_ahead), lost=lost, lost_at_hand=lost_at,
                      lost_reason=lost_reason, survived=not lost)


def _scoring_cards(g: BalatroGame, action: dict) -> list:
    """The cards the engine will actually score for ``action``.

    Ride the Bus reads ``context.scoring_hand``, not the played set (card.lua:3526), so a
    face played as a non-scoring kicker must NOT count here.  ``evaluate_hand`` returns
    ``(hand_type, scoring_cards)`` and takes the same Four Fingers / Shortcut / Smeared /
    Pareidolia flags the engine passes it.
    """
    from balatro_sim.hand_eval import evaluate_hand
    cards = [g.hand[i] for i in action.get("cards", []) if 0 <= i < len(g.hand)]
    if not cards:
        return []
    keys = {j.key for j in g.jokers}
    _, scoring = evaluate_hand(
        cards,
        four_fingers="j_four_fingers" in keys,
        shortcut="j_shortcut" in keys,
        smeared="j_smeared" in keys,
        pareidolia="j_pareidolia" in keys,
    )
    return list(scoring)


def _is_face(c) -> bool:
    return getattr(c, "rank", 0) in (11, 12, 13) and not getattr(c, "debuffed", False)


def _traj_job(args):
    seed, key, n, budget = args
    return trajectory(seed, key, n, budget=budget)


def measure_trajectory(entry, seeds: Sequence[str], hands_ahead: int, *,
                       workers: int = DEFAULT_WORKERS, budget: str = "fast",
                       calib_frac: float = 0.5) -> Measurement:
    """Run ``len(seeds)`` real trajectories and score the entry against them.

    Rate handling, and why it is not circular:

    * **Green Joker** takes each seed's OWN realized discard count, so no rate is fitted at
      all — the predictor is tested purely on its mechanism (and on its known floor bias).
    * **Ride the Bus** needs a face-hand RATE, which the Lua cannot supply.  The first
      ``calib_frac`` of the seeds is a CALIBRATION split used only to estimate that rate;
      the entry is then scored on the held-out remainder.  A rate fitted and evaluated on
      the same seeds would make the check vacuous.
    * **Ice Cream** needs no rate: its decay is a deterministic function of hands played.
    """
    t0 = time.perf_counter()
    key = entry.key
    jobs = [(s, key, hands_ahead, budget) for s in seeds]
    raw = _run_jobs(_traj_job, jobs, workers)
    n_raw = len(raw)
    n_lost = sum(1 for r in raw if r.lost)
    n_short = sum(1 for r in raw if not r.reached and not r.lost)
    # Only seeds that reached the horizon WITH the joker still owned are scored.  Seeds
    # where the shop sold it are reported, not averaged in: including them would measure
    # the shop's opinion of the item, not the item.
    results = [r for r in raw if r.reached and r.survived]
    if not results:
        raise RuntimeError(
            f"{key}: no seed reached {hands_ahead} hands still owning the joker "
            f"({n_lost}/{n_raw} sold or removed, {n_short}/{n_raw} died early)")

    n_cal = max(1, int(len(results) * calib_frac)) if key == "j_ride_the_bus" else 0
    cal, ev = results[:n_cal], results[n_cal:]
    if not ev:
        cal, ev = results, results

    face_rate = (sum(r.face_hands for r in cal) / max(1, sum(r.hands for r in cal))) if cal else 0.0
    preds, meas, res, rows = [], [], [], []
    for r in ev:
        s = {"hands_ahead": r.hands, "discards_ahead": r.discards,
             "face_hand_rate": face_rate, "rtb_mult": r.start,
             "green_mult": r.start, "ice_chips": r.start}
        p = float(entry.predict(s))
        preds.append(p)
        meas.append(r.value)
        res.append(p - r.value)
        rows.append({"seed": r.seed, "predicted": p, "measured": r.value,
                     "hands": r.hands, "discards": r.discards, "face_hands": r.face_hands})
    start = results[0].start
    delta = (sum(meas) / len(meas)) - start
    sign_ok = True if not entry.sign_of_delta else (delta * entry.sign_of_delta) > 0
    m = _assemble(entry, "scaling_trajectory", preds, meas, res, rows,
                  sum(r.hands for r in ev), time.perf_counter() - t0,
                  extra={"hands_ahead": hands_ahead, "start": start,
                         "delta": delta, "sign_of_delta_ok": sign_ok,
                         "calibration_seeds": n_cal, "face_hand_rate": face_rate,
                         "discards_per_hand": (sum(r.discards for r in ev)
                                               / max(1, sum(r.hands for r in ev))),
                         "seeds_run": n_raw, "seeds_scored": len(ev),
                         "lost_before_horizon": n_lost,
                         "lost_reasons": _counts(r.lost_reason for r in raw if r.lost),
                         "survival_rate": 1.0 - (n_lost / n_raw) if n_raw else float("nan"),
                         "died_early": n_short,
                         "measured_sd": statistics.stdev(meas) if len(meas) > 1 else 0.0})
    if not sign_ok:                     # a decay predictor that came out flat/rising
        m.exact = False
        m.exactness_required = True
    return m


# ────────────────────────────────────────────────────────── mode D: rollout_paired

@dataclass
class PairedRollout:
    world: int
    d_dollars: float
    d_ante: float
    d_blinds: float
    gross_with: float               # instrumented payout of the item in the WITH arm
    rounds_with: int                # blinds cleared (state-transition count)
    steps: int
    payouts_with: int = 0           # on_round_end firings — the AUTHORITATIVE round count
    nines_start: int = 0            # deck composition drift, the other half of the residual
    nines_end: int = 0


def _paired_job(args):
    seed, key, world, stop_ante, budget, max_steps, install = args
    return _paired_rollout(seed, key, world, stop_ante, budget, max_steps, install)


def _paired_rollout(seed, key, world, stop_ante, budget, max_steps, install):
    base = BalatroGame(seed=seed, deck_key="b_red", stake=1, ruleset="vanilla")
    with_arm = base.clone_determinized(world)
    without_arm = base.clone_determinized(world)
    if install == "joker":
        with_arm.debug_add_joker(key)
    elif install == "voucher":
        _consumables.apply_voucher(with_arm, key)
        with_arm.vouchers.add(key)
    else:
        raise ValueError(install)

    n0 = sum(1 for c in with_arm.full_deck if c.rank == 9)
    with reach_probe(key) as rec:
        a = _play_to(with_arm, stop_ante, budget, max_steps, seed_off=world)
    b = _play_to(without_arm, stop_ante, budget, max_steps, seed_off=world)
    return PairedRollout(world=world,
                         d_dollars=float(a["dollars"] - b["dollars"]),
                         d_ante=float(a["ante"] - b["ante"]),
                         d_blinds=float(a["blinds"] - b["blinds"]),
                         gross_with=rec.money_written,
                         rounds_with=a["blinds"], steps=a["steps"] + b["steps"],
                         payouts_with=rec.calls.get("on_round_end", 0),
                         nines_start=n0,
                         nines_end=sum(1 for c in with_arm.full_deck if c.rank == 9))


def _play_to(g: BalatroGame, stop_ante: int, budget: str, max_steps: int, seed_off: int) -> dict:
    """Play ``g`` forward with ``ev:fast`` until ``stop_ante``.  ``blinds`` counts blinds
    CLEARED: every entry into ROUND_EVAL from a hand decision is one cleared blind (a lost
    blind goes to GAME_OVER in a vanilla solo run).

    Each ARM starts from a cold player cache (``reset_player_caches``) so the WITH arm,
    which runs first, cannot warm a ``_RATIO_CACHE`` entry that the WITHOUT arm then reads —
    a coupling that would leak the item's own effect into its control."""
    from player import EVPlayer
    reset_player_caches()
    pl = EVPlayer(budget=budget, seed=int(seed_off))
    blinds = steps = 0
    while g.state != State.GAME_OVER and g.ante < stop_ante and steps < max_steps:
        st = g.state
        a = pl.act(g)
        g.step(a)
        steps += 1
        if st != State.ROUND_EVAL and g.state in (State.ROUND_EVAL, State.SHOP) and st == State.SELECTING_HAND:
            blinds += 1
    return {"dollars": g.dollars, "ante": g.ante, "blinds": blinds, "steps": steps}


def measure_rollout(entry, seed: str, *, worlds: int = DEFAULT_ROLLOUT_WORLDS,
                    stop_ante: int = 4, budget: str = "fast", workers: int = DEFAULT_WORKERS,
                    max_steps: int = 3000, install: str = "joker",
                    per_round_predict: Optional[Callable] = None) -> Measurement:
    """With-vs-without on the SAME ``clone_determinized`` worlds, played by ``ev:fast``.

    Three numbers come out, and they answer different questions:

    ``gross``    the item's realized GROSS payout in the WITH arm, read off the reachability
                 probe.  This is what the per-round predictor is scored against
                 (predicted x rounds), because it is the only one of the three that the
                 encode layer actually claims.
    ``d_money``  realized final-money difference under CRN.  It is NOT the gross: the extra
                 dollars get SPENT, and the arms diverge as soon as one can afford something
                 the other cannot.  Checked for direction only.
    ``d_ante``   realized ante-progress difference.  Informational; a single item almost
                 never moves it out of the noise at this sample size, and saying so is part
                 of the honest reporting.
    """
    t0 = time.perf_counter()
    jobs = [(seed, entry.engine_key, w, stop_ante, budget, max_steps, install)
            for w in range(worlds)]
    res = _run_jobs(_paired_job, jobs, workers)
    gross = [r.gross_with for r in res]
    dmoney = [r.d_dollars for r in res]
    dante = [r.d_ante for r in res]
    rounds = [r.rounds_with for r in res]
    mean_rounds = sum(rounds) / len(rounds) if rounds else 0.0
    # The AUTHORITATIVE number of payout opportunities is the hook-firing count, not a
    # state-transition tally: a blind counter can drift (a run that ends inside a blind, a
    # boss transition), and scoring a per-round predictor against a drifting round count
    # would blame the entry for the harness's bookkeeping.
    payouts = [r.payouts_with for r in res]
    mean_payouts = sum(payouts) / len(payouts) if payouts else 0.0
    if per_round_predict is not None:
        per_round = float(per_round_predict())
        preds = [per_round * r.payouts_with for r in res]
    else:
        preds = [float("nan")] * len(res)
    resid = [p - g_ for p, g_ in zip(preds, gross)]
    mean_dm, ci_dm = _mean_ci(dmoney)
    mean_da, ci_da = _mean_ci(dante)
    mean_p = (sum(preds) / len(preds)) if per_round_predict is not None else float("nan")
    mean_g, _ = _mean_ci(gross)
    r_mean, r_ci = _mean_ci(resid) if per_round_predict is not None else (0.0, 0.0)
    return Measurement(
        key=entry.key, mode="rollout_paired", n=len(res),
        predicted=mean_p, measured=mean_g, ci=r_ci, residual=r_mean, unit=entry.unit,
        fired=int(sum(1 for g_ in gross if g_ > 0)),
        needs_fire=(per_round_predict is not None),
        exact=True, exactness_required=False,          # a rollout is never exact by design
        scored=(per_round_predict is not None),
        seconds=time.perf_counter() - t0,
        scenarios=[{"world": r.world, "gross": r.gross_with, "d_dollars": r.d_dollars,
                    "d_ante": r.d_ante, "rounds": r.rounds_with, "payouts": r.payouts_with,
                    "nines_start": r.nines_start, "nines_end": r.nines_end} for r in res],
        extra={"mean_rounds": mean_rounds, "mean_payouts": mean_payouts,
               "d_money": mean_dm, "d_money_ci": ci_dm,
               "d_ante": mean_da, "d_ante_ci": ci_da, "stop_ante": stop_ante,
               "worlds": worlds, "seed": seed,
               "d_money_direction_ok": mean_dm > 0,
               "empirical_buy_value_dollars": mean_dm,
               "nines_start": (sum(r.nines_start for r in res) / len(res)) if res else 0.0,
               "nines_end": (sum(r.nines_end for r in res) / len(res)) if res else 0.0,
               "gross_per_payout": (sum(gross) / max(1, sum(payouts))),
               "note": "gross is the item's realized payout (probe); d_money is the CRN "
                       "with-vs-without final-money delta and is SMALLER because the money "
                       "gets spent and the arms diverge."})


# ──────────────────────────────────────────────────────────────────── process pool

def _run_jobs(fn: Callable, jobs: Sequence, workers: int) -> list:
    workers = max(1, min(int(workers), MAX_WORKERS))
    if workers == 1 or len(jobs) <= 1:
        return [fn(j) for j in jobs]
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        return list(pool.map(fn, jobs, chunksize=1))


# ─────────────────────────────────────────────────────────────── empirical fallback

def empirical_fallback(m: Measurement) -> dict:
    """What the encode layer should use when an entry is REJECTED or ambiguous: the
    harness's own measured number, with its provenance.  Part of the design, not a
    consolation prize — a measured constant with a CI beats a wrong closed form."""
    value, ci = m.measured, m.ci
    if m.mode == "rollout_paired":
        # the useful buy-value number from a paired rollout is the CRN money delta, not the
        # item's gross payout (the gross is what the per-round entry claims, and it is
        # already scored above)
        value = m.extra.get("empirical_buy_value_dollars", m.measured)
        ci = m.extra.get("d_money_ci", m.ci)
    return {
        "key": m.key, "mode": m.mode, "unit": m.unit,
        "value": value, "ci95": ci, "n": m.n,
        "source": f"ev/encode/verify.py::{m.mode}",
        "supersedes_predict": m.scored and not m.accept,
        "note": MARGINAL_RULE,
    }
