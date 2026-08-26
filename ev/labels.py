"""
labels.py — P(win) labels for V by determinized rollouts (Phase 5 rev 2, W5).

The target (frozen, STATE_SPEC_v1 §Target):

    y = P(player wins the MLB match | full match state)

estimated by ``n_rollouts`` determinized Monte-Carlo rollouts of the analytic policy on BOTH
sides (symmetric opponent), each from ``match.clone_determinized(seed_i)`` (W2: fresh draw
orders / fresh future streams, the SAME fresh seed for both games = same-seed MP), played
with a fresh ``EVPlayer(budget="fast", epsilon=small)`` per side (W3) until ``match.done``
(outcome 1/0) or ``ante > max_ante`` (closed by ``race.p_win`` on each side's observed
curve + lives).  V is trained on the player's OBSERVATION of the state (W1's encoder v2
with the opponent block), so it learns the expectation over the hidden opponent state.

Where the states come from — decision (2026-08-23): **in-process snapshots**, not log
replay.  ``sample_states(seed)`` self-plays one match (same policy, ``epsilon`` for
diversity) and keeps ``match.clone()`` at reservoir-sampled decision steps, stratified by
``STATE_KINDS``.  Cheaper than logging + exact replay (a clone is 0.14 ms) and needs no log
files; every snapshot is still reproducible from ``(seed, step, selfplay config)`` via
``reconstruct_snapshot`` because the self-play policy is seeded (``policy_seed``) and the
engine is deterministic on a seed.  The tag is stored in every row's meta.

Both players' perspectives of a snapshot are two rows.  ``label_both`` derives them from ONE
rollout set (``y1 = 1 - y0`` per rollout — exact: a finished match has one winner and the
race model is symmetric, ``race.p_win(a, b, ..) + race.p_win(b, a, ..) == 1``).  The
sum-to-one sanity test in ``test_labels.py`` therefore uses INDEPENDENT rollout sets for the
two perspectives (``label_state`` twice) — that is what checks the estimator, not the
identity.

Feature detection (Stage 1 was built before W1/W2/W3 landed):
  * ``MLBMatch.clone_determinized`` (W2) — absent → plain ``clone()`` + a ``determinized=False``
    flag in the result (rollouts then share the true future; fine for plumbing tests, NOT a
    valid label — ``label_job`` refuses unless ``allow_clairvoyant=True``).
  * ``ev.player.EVPlayer`` (W3) — absent → the scripted greedy policy from
    ``scripts/mlb_match_demo`` with epsilon-random actions (``policy="scripted"``).
  * ``mcts.encoder_v2`` (W1) — absent → ``encoder="dummy"`` (16 scalars) for plumbing tests.
"""
from __future__ import annotations

import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import _bootstrap  # noqa: F401  (fork guard, sys.path for mcts / engine)
from _bootstrap import MLBMatch, State, MP_ROOT

import numpy as np

import race as _race
from dataset import LabelRow

__all__ = [
    "STATE_KINDS", "state_kind", "Snapshot", "RolloutResult", "LabelResult",
    "make_policy_factory", "make_encoder", "rollout", "label_state", "label_both",
    "sample_states", "reconstruct_snapshot", "label_job", "wilson_halfwidth",
    "has_determinize", "has_ev_player", "has_encoder_v2",
]

STATE_KINDS = ("blind_select", "hand", "nemesis", "shop", "pack", "other")
DEFAULT_MAX_ANTE = 12
DEFAULT_MAX_STEPS = 40_000


# ── feature detection ─────────────────────────────────────────────────────────────

def has_determinize() -> bool:
    return callable(getattr(MLBMatch, "clone_determinized", None))


def has_ev_player() -> bool:
    try:
        import player as _p  # noqa: F401  (ev/player.py, W3)
        return hasattr(_p, "EVPlayer")
    except Exception:
        return False


def has_encoder_v2() -> bool:
    try:
        import mcts.encoder_v2 as _e  # noqa: F401  (W1)
        return hasattr(_e, "SetEncoderV2") and hasattr(_e, "opponent_view")
    except Exception:
        return False


# ── state kinds ───────────────────────────────────────────────────────────────────

def state_kind(game) -> str:
    """Coarse decision-state class of ONE game (for stratified sampling and reporting)."""
    s = game.state
    if s == State.BLIND_SELECT:
        return "blind_select"
    if s == State.SELECTING_HAND:
        blind = getattr(game, "current_blind", None)
        return "nemesis" if (blind is not None and getattr(blind, "is_pvp", False)) else "hand"
    if s == State.SHOP:
        return "shop"
    if s == State.BOOSTER_OPEN:
        return "pack"
    return "other"


# ── policies ──────────────────────────────────────────────────────────────────────

def _scripted_policy(seed: int, epsilon: float) -> Callable:
    scripts = str(MP_ROOT / "scripts") if hasattr(MP_ROOT, "__truediv__") else f"{MP_ROOT}/scripts"
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import mlb_match_demo as D
    base = D.make_policy(D.ScriptedPlayer(hand="greedy"))
    rng = random.Random(seed)

    def pol(match, p, acts):
        if epsilon > 0 and acts and rng.random() < epsilon:
            return rng.choice(acts)
        return base(match, p, acts)

    pol.reset = lambda: None          # type: ignore[attr-defined]
    return pol


def _with_epsilon(obj, seed: int, epsilon: float) -> Callable:
    """Wrap a ``Player`` (``act(game)``) into the match-policy signature with W5's OWN
    epsilon-random stream (sequential ``random.Random(seed)``, not keyed on the state).

    Why not ``EVPlayer(epsilon=...)``: W3 keys its exploration RNG on the observable state
    (player.py:196-199 via sampling.world_rng, whose key omits shop/consumable contents),
    so a random pick that turns out to be a legal-but-no-op action (e.g. ``use_consumable``
    Wheel of Fortune with no eligible joker) is re-drawn identically forever — the shop
    wedge seen 2026-08-23 (39,952 consecutive shop steps).  A sequential stream cannot
    repeat that way, and ``_Guard`` catches any remaining no-op loop."""
    rng = random.Random(seed)

    def pol(match, p, acts, _obj=obj, _rng=rng):
        if epsilon > 0 and acts and _rng.random() < epsilon:
            return dict(_rng.choice(acts))
        return _obj.act(match.games[p])
    pol.reset = getattr(obj, "reset", lambda: None)             # type: ignore[attr-defined]
    pol.player = obj                                            # type: ignore[attr-defined]
    return pol


def _ev_policy(seed: int, epsilon: float, budget: str, value_fn=None, stats=None) -> Callable:
    import player as P                 # ev/player.py (W3)
    kw = {"budget": budget, "seed": seed, "epsilon": 0.0}
    if stats is not None:
        kw["stats"] = stats
    return _with_epsilon(P.EVPlayer(value_fn, **kw), seed, epsilon)


def load_stats_module():
    """W4's ``stats/decide`` (has ``decision_table(game)``), importable as ``stats=`` for
    ``EVPlayer``.  stats is a flat package with its own bootstrap, so it goes on sys.path."""
    stats_dir = os.path.join(str(MP_ROOT), "stats")
    if stats_dir not in sys.path:
        sys.path.insert(0, stats_dir)
    import decide  # noqa: WPS433
    return decide


def make_policy_factory(policy: str = "auto", *, budget: str = "fast", epsilon: float = 0.02,
                        value_fn=None, stats=None, shop_tier: str = "rules") -> Callable:
    """``factory(seed, player) -> policy(match, p, acts) -> action``.  ``policy``: ``"ev"``
    (W3's EVPlayer), ``"scripted"`` (greedy fallback), ``"auto"`` (ev if importable).

    ``shop_tier``: ``"rules"`` (W3's built-in shop/pack rules — THE label definition, chosen
    2026-08-23: stronger, mean final ante 6.1 vs 3.9 in self-play on 10 seeds) or ``"stats"``
    (W4's ``decision_table``; 4x cheaper, weaker).  An explicit ``stats`` object wins."""
    if policy == "auto":
        policy = "ev" if has_ev_player() else "scripted"
    if stats is None and shop_tier == "stats":
        stats = load_stats_module()
    if policy == "ev":
        return lambda seed, player: _ev_policy(int(seed), epsilon, budget, value_fn, stats)
    if policy == "scripted":
        return lambda seed, player: _scripted_policy(int(seed), epsilon)
    raise ValueError(f"unknown policy {policy!r}")


# ── encoders ──────────────────────────────────────────────────────────────────────

def _dummy_encode(match, player: int) -> dict:
    """16 cheap scalars — plumbing tests and the trainer's smoke test only."""
    g = match.games[player]
    o = match.games[1 - player]
    x = np.zeros(16, dtype=np.float32)
    x[0] = g.ante / 12.0
    x[1] = g.lives / 4.0
    x[2] = o.lives / 4.0
    x[3] = (g.lives - o.lives) / 4.0
    x[4] = min(g.dollars, 100) / 100.0
    x[5] = min(o.dollars, 100) / 100.0
    x[6] = len(g.jokers) / 5.0
    x[7] = len(o.jokers) / 5.0
    x[8] = g.hands_left / 4.0
    x[9] = math.log10(max(g.chips_scored, 1)) / 6.0
    x[10] = float(g.state == State.SHOP)
    x[11] = float(g.state == State.SELECTING_HAND)
    x[12] = float(getattr(g.current_blind, "is_pvp", False))
    x[13] = g.comeback_bonus / 4.0
    x[14] = o.comeback_bonus / 4.0
    x[15] = 1.0
    return {"x": x}


def make_encoder(name: str = "auto") -> Callable:
    """``encode(match, player) -> obs dict`` for V.  ``"v2"`` = W1's
    ``SetEncoderV2(game, opponent_view(match, player))``; ``"dummy"`` = 16 scalars."""
    if name == "auto":
        name = "v2" if has_encoder_v2() else "dummy"
    if name == "dummy":
        return _dummy_encode
    if name == "v2":
        import mcts.encoder_v2 as E
        enc = E.SetEncoderV2()

        def encode(match, player: int, _enc=enc, _E=E):
            return _enc(match.games[player], _E.opponent_view(match, player))
        return encode
    raise ValueError(f"unknown encoder {name!r}")


# ── rollouts ──────────────────────────────────────────────────────────────────────

@dataclass
class RolloutResult:
    p0_win: float                 # 1 / 0, or race.p_win for player 0 at truncation
    winner: Optional[int]
    truncated: bool
    determinized: bool
    steps: int
    decisions: int
    seconds: float
    policy_seconds: float
    antes: tuple
    lives: tuple
    n_nemeses: int
    forced: int = 0               # no-progress guard interventions
    aux: Optional[dict] = None    # W-AUX: the rollout observer's result(), None without one


def _determinize(match, seed: int):
    fn = getattr(match, "clone_determinized", None)
    if callable(fn):
        return fn(seed), True
    return match.clone(), False


def _truncated(m, max_ante: int) -> bool:
    return min(g.ante for g in m.games) > max_ante


def _race_p0(m, max_ante: int, cfg: _race.RaceConfig) -> float:
    ante = max(max_ante + 1, min(g.ante for g in m.games))
    c0 = _race.curve_from_history(m.pvp_log, 0, ante, cfg=cfg)
    c1 = _race.curve_from_history(m.pvp_log, 1, ante, cfg=cfg)
    return _race.p_win(c0, c1, m.games[0].lives, m.games[1].lives, ante, cfg=cfg)


_PROGRESS_PREF = ("leave_shop", "skip_booster", "advance", "play_blind", "select_blind")
NO_PROGRESS_LIMIT = 3


class _Guard:
    """Detects a no-op loop (the match signature unchanged after a step) and, after
    ``NO_PROGRESS_LIMIT`` consecutive no-ops, forces a progress action (``leave_shop`` /
    ``skip_booster`` / ``advance`` / the first legal action that differs from the stuck one).
    Counts how often it fired (``forced``) so the label meta can record it."""
    __slots__ = ("last_sig", "repeats", "forced", "stuck")

    def __init__(self):
        self.last_sig = None
        self.repeats = 0
        self.forced = 0
        self.stuck = None

    def choose(self, m, p, acts, action: dict) -> dict:
        if self.repeats < NO_PROGRESS_LIMIT:
            return action
        self.forced += 1
        for t in _PROGRESS_PREF:
            for a in acts:
                if a["type"] == t and a != self.stuck:
                    return dict(a)
        for a in acts:
            if a != self.stuck:
                return dict(a)
        return action

    def after(self, m, action: dict) -> None:
        sig = m.signature()
        if sig == self.last_sig:
            self.repeats += 1
            self.stuck = action
        else:
            self.repeats = 0
            self.stuck = None
        self.last_sig = sig


def rollout(match, *, seed: int, policy_factory: Callable, max_ante: int = DEFAULT_MAX_ANTE,
            max_steps: int = DEFAULT_MAX_STEPS, race_cfg: _race.RaceConfig = _race.DEFAULT,
            determinize: bool = True, observer_factory: Optional[Callable] = None) -> RolloutResult:
    """One determinized play-out of ``match`` (not modified) with fresh policies on both sides.

    ``observer_factory`` (W-AUX, additive): ``factory() -> observer`` with ``start(m)`` /
    ``after(m, player, action)`` / ``finish(m)`` / ``result()``.  One fresh observer per
    rollout; its ``result()`` lands in ``RolloutResult.aux``.  ``None`` (the default) is a
    single ``is not None`` test per step and leaves every number below bit-identical —
    ``aux_targets.make_recorder_factory()`` is the one this campaign passes."""
    t0 = time.perf_counter()
    m, det = _determinize(match, seed) if determinize else (match.clone(), False)
    if det:
        for g in m.games:                       # belt and braces on W2's contract
            assert getattr(g, "determinized", True), "clone_determinized returned an undetermined game"
    pols = [policy_factory(seed * 2 + p, p) for p in (0, 1)]
    steps = 0
    decisions = 0
    pol_s = 0.0
    guard = _Guard()
    observer = observer_factory() if observer_factory is not None else None
    if observer is not None:
        observer.start(m)
    while not m.done and steps < max_steps and not _truncated(m, max_ante):
        p = m.current_player()
        if p is None:
            raise RuntimeError(f"rollout wedged: nobody can act ({m.state()})")
        acts = m.legal_actions(p)
        t1 = time.perf_counter()
        a = guard.choose(m, p, acts, pols[p](m, p, acts))
        pol_s += time.perf_counter() - t1
        m.step(p, a)
        if observer is not None:
            observer.after(m, p, a)
        guard.after(m, a)
        steps += 1
        decisions += 1
    if observer is not None:
        observer.finish(m)
    if m.done:
        p0 = 1.0 if m.winner == 0 else 0.0
        trunc = False
    else:
        p0 = _race_p0(m, max_ante, race_cfg)
        trunc = True
    return RolloutResult(p0_win=float(p0), winner=m.winner, truncated=trunc, determinized=det,
                         steps=steps, decisions=decisions, seconds=time.perf_counter() - t0,
                         policy_seconds=pol_s, antes=tuple(g.ante for g in m.games),
                         lives=tuple(g.lives for g in m.games), n_nemeses=len(m.pvp_log),
                         forced=guard.forced,
                         aux=(observer.result() if observer is not None else None))


# ── labels ────────────────────────────────────────────────────────────────────────

def wilson_halfwidth(k: float, n: int, z: float = 1.96) -> float:
    """Half-width of the Wilson score interval for ``k`` successes in ``n`` trials."""
    if n <= 0:
        return 1.0
    p = k / n
    denom = 1 + z * z / n
    return (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom


@dataclass
class LabelResult:
    y: float
    ci: float                         # 95% half-width
    n: int
    outcomes: list = field(default_factory=list)
    trunc_frac: float = 0.0
    determinized: bool = True
    seconds: float = 0.0
    decisions: int = 0
    policy_seconds: float = 0.0
    forced: int = 0
    #: W-AUX: ``{player: {aux field: mean over the rollouts or None}}``, empty without an
    #: observer.  It is keyed by PLAYER (not by perspective), so ``flipped()`` carries the
    #: same dict through unchanged — player 1's row reads ``aux_by_player[1]``.
    aux_by_player: dict = field(default_factory=dict)

    def as_tuple(self) -> tuple:
        return self.y, self.ci

    def flipped(self) -> "LabelResult":
        """The same rollouts seen by the other player."""
        return LabelResult(y=1.0 - self.y, ci=self.ci, n=self.n,
                           outcomes=[1.0 - o for o in self.outcomes], trunc_frac=self.trunc_frac,
                           determinized=self.determinized, seconds=self.seconds,
                           decisions=self.decisions, policy_seconds=self.policy_seconds,
                           forced=self.forced, aux_by_player=self.aux_by_player)


def _ci_of(outcomes: Sequence[float]) -> float:
    n = len(outcomes)
    if n == 0:
        return 1.0
    binary = all(o in (0.0, 1.0) for o in outcomes)
    if binary:
        return wilson_halfwidth(sum(outcomes), n)
    mean = sum(outcomes) / n
    var = sum((o - mean) ** 2 for o in outcomes) / max(n - 1, 1)
    return max(1.96 * math.sqrt(var / n), wilson_halfwidth(mean * n, n) * 0.25)


def label_both(match, *, n_rollouts: int = 8, seed: int = 0, policy_factory: Optional[Callable] = None,
               max_ante: int = DEFAULT_MAX_ANTE, max_steps: int = DEFAULT_MAX_STEPS,
               race_cfg: _race.RaceConfig = _race.DEFAULT, determinize: bool = True,
               observer_factory: Optional[Callable] = None) -> tuple:
    """(LabelResult for player 0, LabelResult for player 1) from ONE set of rollouts.

    ``observer_factory`` (W-AUX) is threaded into every rollout and its per-rollout results
    are averaged over the rollout set into ``LabelResult.aux_by_player`` (brief §6b.1: "mean
    over the shared worlds")."""
    if policy_factory is None:
        policy_factory = make_policy_factory()
    t0 = time.perf_counter()
    outs, trunc, det, dec, pol_s, forced = [], 0, True, 0, 0.0, 0
    aux_worlds = []
    for i in range(int(n_rollouts)):
        r = rollout(match, seed=seed * 1_000_003 + i, policy_factory=policy_factory, max_ante=max_ante,
                    max_steps=max_steps, race_cfg=race_cfg, determinize=determinize,
                    observer_factory=observer_factory)
        outs.append(r.p0_win)
        trunc += int(r.truncated)
        det = det and r.determinized
        dec += r.decisions
        pol_s += r.policy_seconds
        forced += r.forced
        if r.aux is not None:
            aux_worlds.append(r.aux)
    n = len(outs)
    aux_by_player = {}
    if aux_worlds:
        import aux_targets as AX
        aux_by_player = {p: AX.aggregate(aux_worlds, p) for p in (0, 1)}
    res0 = LabelResult(y=sum(outs) / n, ci=_ci_of(outs), n=n, outcomes=outs, trunc_frac=trunc / n,
                       determinized=det, seconds=time.perf_counter() - t0, decisions=dec,
                       policy_seconds=pol_s, forced=forced, aux_by_player=aux_by_player)
    return res0, res0.flipped()


def label_state(match, player: int, *, n_rollouts: int = 8, seed: int = 0,
                policy_factory: Optional[Callable] = None, max_ante: int = DEFAULT_MAX_ANTE,
                max_steps: int = DEFAULT_MAX_STEPS, race_cfg: _race.RaceConfig = _race.DEFAULT,
                determinize: bool = True, detail: bool = False):
    """``(p_win, ci_halfwidth)`` for ``player`` (``detail=True`` → the ``LabelResult``)."""
    r0, r1 = label_both(match, n_rollouts=n_rollouts, seed=seed, policy_factory=policy_factory,
                        max_ante=max_ante, max_steps=max_steps, race_cfg=race_cfg,
                        determinize=determinize)
    r = r0 if int(player) == 0 else r1
    return r if detail else r.as_tuple()


# ── snapshots ─────────────────────────────────────────────────────────────────────

@dataclass
class Snapshot:
    seed: str
    step: int                     # match.steps at the snapshot (actions applied so far)
    actor: int                    # the player about to act
    kind: str                     # state_kind(actor's game)
    ante: int
    match: object                 # MLBMatch clone at that step
    selfplay: dict = field(default_factory=dict)

    def tag(self) -> dict:
        return {"seed": self.seed, "step": self.step, "actor": self.actor, "kind": self.kind,
                "ante": self.ante, **{f"sp_{k}": v for k, v in self.selfplay.items()}}


def _selfplay_config(policy: str, budget: str, epsilon: float, policy_seed: int, deck_key: str,
                     stake, lives: int, max_ante: int) -> dict:
    return {"policy": policy, "budget": budget, "epsilon": epsilon, "policy_seed": policy_seed,
            "deck_key": deck_key, "stake": stake, "lives": lives, "max_ante": max_ante}


def _drive_selfplay(seed: str, *, policy_factory: Callable, policy_seed: int, deck_key: str, stake,
                    lives: int, max_ante: int, max_steps: int, on_decision: Optional[Callable] = None):
    m = MLBMatch(seed=seed, deck_key=deck_key, stake=stake, lives=lives)
    pols = [policy_factory(policy_seed * 2 + p, p) for p in (0, 1)]
    guard = _Guard()
    while not m.done and m.steps < max_steps and not _truncated(m, max_ante):
        p = m.current_player()
        if p is None:
            raise RuntimeError(f"self-play wedged: nobody can act ({m.state()})")
        acts = m.legal_actions(p)
        if on_decision is not None and on_decision(m, p) is False:
            break
        a = guard.choose(m, p, acts, pols[p](m, p, acts))
        m.step(p, a)
        guard.after(m, a)
    m._w5_forced = guard.forced
    return m


def sample_states(seed: str, *, n_states: int = 12, per_kind: Optional[dict] = None,
                  policy_factory: Optional[Callable] = None, policy: str = "auto", budget: str = "fast",
                  epsilon: float = 0.1, policy_seed: int = 0, rng_seed: Optional[int] = None,
                  deck_key: str = "b_red", stake=1, lives: int = 4, max_ante: int = DEFAULT_MAX_ANTE,
                  max_steps: int = DEFAULT_MAX_STEPS, min_step: int = 0) -> list:
    """Self-play ONE match on ``seed`` and return reservoir-sampled ``Snapshot``s, stratified
    by ``STATE_KINDS`` (``per_kind`` caps; default ``n_states`` spread evenly over the kinds
    that occur).  ``rng_seed`` (default: hash of seed + policy_seed) seeds the reservoir."""
    if policy_factory is None:
        policy_factory = make_policy_factory(policy, budget=budget, epsilon=epsilon)
    kinds = list(STATE_KINDS)
    if per_kind is None:
        share = max(1, int(math.ceil(n_states / max(len(kinds) - 1, 1))))   # "other" is rare
        per_kind = {k: share for k in kinds}
    # sha1, not hash(): str hash is PYTHONHASHSEED-salted per process, so the default
    # snapshot set differed across workers/runs (found independently by W-PAIRS and the
    # W-ACTIVE POC). Stable across processes; explicit rng_seed still overrides.
    if rng_seed is None:
        import hashlib as _hl
        rng_seed = int.from_bytes(_hl.sha1(f"{seed}:{policy_seed}".encode()).digest()[:4], "big")
    rng = random.Random(rng_seed)
    reservoirs: dict = {k: [] for k in kinds}
    seen: dict = {k: 0 for k in kinds}

    def on_decision(m, p):
        if m.steps < min_step:
            return True
        k = state_kind(m.games[p])
        cap = per_kind.get(k, 0)
        if cap <= 0:
            return True
        seen[k] += 1
        snap = (m.steps, p, k, m.games[p].ante)
        if len(reservoirs[k]) < cap:
            reservoirs[k].append((snap, m.clone()))
        else:
            j = rng.randrange(seen[k])
            if j < cap:
                reservoirs[k][j] = (snap, m.clone())
        return True

    m = _drive_selfplay(seed, policy_factory=policy_factory, policy_seed=policy_seed, deck_key=deck_key,
                        stake=stake, lives=lives, max_ante=max_ante, max_steps=max_steps,
                        on_decision=on_decision)
    sp = _selfplay_config(policy, budget, epsilon, policy_seed, deck_key, stake, lives, max_ante)
    sp.update({"winner": m.winner, "final_antes": [g.ante for g in m.games],
               "final_lives": [g.lives for g in m.games], "steps": m.steps,
               "forced": getattr(m, "_w5_forced", 0), "decisions_seen": dict(seen)})
    out = []
    for k in kinds:
        for (step, p, kind, ante), clone in reservoirs[k]:
            out.append(Snapshot(seed=m.seed_str, step=step, actor=p, kind=kind, ante=ante,
                                match=clone, selfplay=dict(sp)))
    out.sort(key=lambda s: s.step)
    if len(out) > n_states:
        rng.shuffle(out)
        out = sorted(out[:n_states], key=lambda s: s.step)
    return out


def reconstruct_snapshot(seed: str, step: int, *, policy: str = "auto", budget: str = "fast",
                         epsilon: float = 0.1, policy_seed: int = 0, deck_key: str = "b_red", stake=1,
                         lives: int = 4, max_ante: int = DEFAULT_MAX_ANTE,
                         max_steps: int = DEFAULT_MAX_STEPS, policy_factory: Optional[Callable] = None):
    """Re-play the self-play match of ``sample_states`` to ``step`` and return the match
    (the state a ``Snapshot`` with that tag held).  Bit-exact when the policy is seeded."""
    if policy_factory is None:
        policy_factory = make_policy_factory(policy, budget=budget, epsilon=epsilon)
    holder = {}

    def on_decision(m, p):
        if m.steps >= step:
            holder["m"] = m.clone()
            return False
        return True

    m = _drive_selfplay(seed, policy_factory=policy_factory, policy_seed=policy_seed, deck_key=deck_key,
                        stake=stake, lives=lives, max_ante=max_ante, max_steps=max_steps,
                        on_decision=on_decision)
    return holder.get("m", m)


# ── the worker job ────────────────────────────────────────────────────────────────

def label_job(payload: dict) -> dict:
    """One pool job: self-play ``payload["seed"]`` → snapshots → labels → encoded rows.

    payload keys (all optional but ``seed``): ``n_states`` 12, ``n_rollouts`` 8, ``policy``
    "auto", ``budget`` "fast", ``epsilon_selfplay`` 0.1, ``epsilon_rollout`` 0.02,
    ``policy_seed`` 0, ``rollout_seed`` 0, ``encoder`` "auto", ``max_ante`` 12, ``deck_key``
    "b_red", ``stake`` 1, ``lives`` 4, ``allow_clairvoyant`` False (refuse plain-clone rollouts),
    ``independent_perspectives`` False (label player 1 from its OWN rollout set — the
    sum-to-one symmetry check; doubles the cost), ``aux`` False (W-AUX: record the
    auxiliary targets of ``aux_targets.AUX_SPECS`` during the rollouts the job already runs
    and put the per-world means in each row's ``meta["aux"]``; brief §6b).

    Returns ``{"rows": [{"obs", "y", "meta"}...], "timing": {...}, "selfplay": {...}}``.
    """
    seed = str(payload["seed"])
    n_states = int(payload.get("n_states", 12))
    n_rollouts = int(payload.get("n_rollouts", 8))
    policy = payload.get("policy", "auto")
    budget = payload.get("budget", "fast")
    eps_sp = float(payload.get("epsilon_selfplay", 0.1))
    eps_ro = float(payload.get("epsilon_rollout", 0.02))
    policy_seed = int(payload.get("policy_seed", 0))
    rollout_seed = int(payload.get("rollout_seed", 0))
    max_ante = int(payload.get("max_ante", DEFAULT_MAX_ANTE))
    deck_key = payload.get("deck_key", "b_red")
    stake = payload.get("stake", 1)
    lives = int(payload.get("lives", 4))
    allow_clair = bool(payload.get("allow_clairvoyant", False))
    shop_tier = payload.get("shop_tier", "rules")
    aux_on = bool(payload.get("aux", False))          # W-AUX: record auxiliary targets
    if not allow_clair and not has_determinize():
        raise RuntimeError("MLBMatch.clone_determinized (W2) is missing: rollouts would be "
                           "clairvoyant; pass allow_clairvoyant=True for plumbing tests only")
    observer_factory, aux_version = None, None
    if aux_on:
        import aux_targets as AX
        observer_factory = AX.make_recorder_factory((0, 1))
        aux_version = AX.AUX_VERSION
    encode = make_encoder(payload.get("encoder", "auto"))
    sp_factory = make_policy_factory(policy, budget=budget, epsilon=eps_sp, shop_tier=shop_tier)
    ro_factory = make_policy_factory(policy, budget=budget, epsilon=eps_ro, shop_tier=shop_tier)

    t0 = time.perf_counter()
    snaps = sample_states(seed, n_states=n_states, policy_factory=sp_factory, policy=policy, budget=budget,
                          epsilon=eps_sp, policy_seed=policy_seed, deck_key=deck_key, stake=stake,
                          lives=lives, max_ante=max_ante)
    t_sp = time.perf_counter() - t0
    independent = bool(payload.get("independent_perspectives", False))
    rows, t_label, dec, pol_s, n_ro = [], 0.0, 0, 0.0, 0
    for i, s in enumerate(snaps):
        base = rollout_seed * 7919 + s.step * 31 + i
        r0, r1 = label_both(s.match, n_rollouts=n_rollouts, seed=base, policy_factory=ro_factory,
                            max_ante=max_ante, observer_factory=observer_factory)
        t_label += r0.seconds
        dec += r0.decisions
        pol_s += r0.policy_seconds
        n_ro += r0.n
        if independent:
            # the symmetry check: player 1's label from its OWN rollout set
            _, r1 = label_both(s.match, n_rollouts=n_rollouts, seed=base + 500_009,
                               policy_factory=ro_factory, max_ante=max_ante,
                               observer_factory=observer_factory)
            t_label += r1.seconds
            dec += r1.decisions
            pol_s += r1.policy_seconds
            n_ro += r1.n
        for p, r in ((0, r0), (1, r1)):
            meta = {"seed": s.seed, "step": s.step, "player": p, "actor": s.actor, "kind": s.kind,
                    "ante": s.ante, "ci": r.ci, "n_rollouts": r.n, "trunc_frac": r.trunc_frac,
                    "determinized": r.determinized, "lives": list(s.match.games[p].lives for p in (0, 1)),
                    "outcomes": [round(o, 4) for o in r.outcomes],
                    "selfplay": {k: v for k, v in s.selfplay.items() if k != "decisions_seen"},
                    "n_rollouts_cfg": n_rollouts, "epsilon_rollout": eps_ro,
                    "independent": independent, "forced": r.forced, "shop_tier": shop_tier,
                    "budget": budget}
            if r.aux_by_player:
                # W-AUX (brief §6b.3): additive `aux` dict, this player's perspective.
                meta["aux"] = r.aux_by_player.get(p, {})
                meta["aux_version"] = aux_version
            rows.append({"obs": encode(s.match, p), "y": r.y, "meta": meta})
    return {
        "rows": rows,
        "selfplay": snaps[0].selfplay if snaps else {},
        "timing": {"selfplay_s": t_sp, "label_s": t_label, "n_snapshots": len(snaps),
                   "n_rollouts": n_ro, "rollout_decisions": dec, "rollout_policy_s": pol_s,
                   "ms_per_rollout": (1000 * t_label / n_ro) if n_ro else 0.0,
                   "decisions_per_rollout": (dec / n_ro) if n_ro else 0.0,
                   "policy_frac": (pol_s / t_label) if t_label else 0.0,
                   "total_s": time.perf_counter() - t0},
    }


def rows_from_result(result: dict) -> list:
    return [LabelRow(r["obs"], float(r["y"]), r["meta"]) for r in result["rows"]]
