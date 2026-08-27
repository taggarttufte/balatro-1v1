"""
pairs.py — lever (b): PAIRED within-state action labels (Phase 5 rev 2, W-PAIRS).

The measured failure this attacks (brief §0): absolute P(win) labels have a mean CI of
±0.24 at ``n_rollouts = 8``, while the per-action EV gaps that decide a policy are ≪ 0.05.
Resolving an ordering with absolute labels needs n ≈ 500 rollouts per action.  A PAIR
resolves the same ordering far cheaper because both branches are rolled on the SAME
determinized worlds (common random numbers): the shared luck cancels in the difference,

    var(Δ̂) = (σ_a² + σ_b² − 2·cov) / n   vs   (σ_a² + σ_b²) / n  unpaired,

so the variance-reduction factor is 1/(1−ρ) at equal σ.  §4 of this module measures that
factor empirically rather than assuming it (``variance_report``).

**A pair is exactly TRAINV_NOTES §1's label semantics, applied twice on shared worlds:**

1. sample a decision state ``s`` with ``labels.sample_states`` (unchanged machinery);
2. pick two legal actions ``a``, ``b`` for the actor (``choose_pair``, §``PAIR_SOURCES``);
3. ``m_a = s.clone(); m_a.step(actor, a)`` and likewise ``m_b`` — both are REAL states
   (the immediate transition uses the true stream, exactly as it would in the match), so
   ``obs_a`` / ``obs_b`` are honest encoder-v2 observations of them;
4. for each world seed ``w`` in ONE shared list: ``labels.rollout(m_x, seed=w, ...)``, which
   ``clone_determinized(w)``-es both games with the same fresh seed and plays the fast
   policy on both sides to ``match.done`` (1/0) or ``ante > max_ante`` (``race.p_win``).
   The same ``w`` also seeds both branches' rollout policies, so the ε-streams are common
   random numbers too;
5. outcomes are stored from the ACTOR's perspective (V's target for the actor's own
   observation), ``delta = mean(a) − mean(b)``, ``delta_ci`` = 95 % half-width of the
   PAIRED difference (t-free normal approximation on the per-world differences).

Every pair also yields the two absolute rows ``(obs_a, mean(outcomes_a))`` and
``(obs_b, mean(outcomes_b))`` — pairs do double duty as labels (brief §5.3), written to a
separate ``abs_shards/`` directory so ``dataset.LabelDataset.load`` reads them unchanged.

Not owned here and deliberately untouched: ``labels.py``, ``dataset.py``, ``workers.py``,
``player.py``, ``hand.py``.  The pool driver is ``scripts/gen_pairs.py`` (additive).
See ``PAIRS_NOTES.md`` for the decisions, the schema note and the measured numbers.
"""
from __future__ import annotations

import functools
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import _bootstrap  # noqa: F401  (fork guard, sys.path for mcts / engine)
from _bootstrap import MLBMatch, State, MP_ROOT

import numpy as np

import dataset as DS
import labels as L
import race as _race

__all__ = [
    "PAIR_SOURCES", "DEFAULT_MIX", "DEFAULT_N_WORLDS", "DEFAULT_CLOSE_GAP", "PAIR_COLUMNS",
    "player_fingerprint", "source_digest",
    "has_extraction", "extraction_entry_point", "make_extraction_hook",
    "action_key", "rules_ranking", "make_ranker", "mix_sequence", "PairChoice", "choose_pair",
    "PairRollout", "roll_pair", "step_branch", "COUPLINGS", "PairRow", "pair_record",
    "save_pair_shard", "load_pair_shard", "PairShard", "PairDataset",
    "pair_job", "pairs_from_result", "rows_from_result", "variance_report", "mix_report",
]

#: the `pair_source` field of the frozen schema (brief §5.2).
PAIR_SOURCES = ("close_call", "greedy_vs_extract", "random")
#: target mix over pair sources (brief §5.2).  `greedy_vs_extract`'s mass is folded into
#: `close_call` when W-EXTRACT's generator is not importable (feature detection).
DEFAULT_MIX = {"close_call": 0.50, "greedy_vs_extract": 0.40, "random": 0.10}
DEFAULT_N_WORLDS = 8
DEFAULT_CLOSE_GAP = 0.03
#: `other` (ROUND_EVAL / cash-out) states have exactly one legal action, so they cannot host
#: a pair; the sampler is told not to spend snapshots on them.
PAIRABLE_KINDS = tuple(k for k in L.STATE_KINDS if k != "other")

SHARD_VERSION = 1
PAIR_COLUMNS = ("seed", "step", "actor", "state_kind", "ante", "pair_source",
                "player_fingerprint", "n_worlds", "delta", "delta_ci")
_PAIR_COLUMN_DTYPES = {
    "seed": "U16", "step": np.int32, "actor": np.int8, "state_kind": "U16", "ante": np.int16,
    "pair_source": "U24", "player_fingerprint": "U48", "n_worlds": np.int16,
    "delta": np.float32, "delta_ci": np.float32,
}
_PAIR_COLUMN_DEFAULTS = {"seed": "", "step": -1, "actor": -1, "state_kind": "", "ante": -1,
                         "pair_source": "", "player_fingerprint": "", "n_worlds": 0,
                         "delta": float("nan"), "delta_ci": float("nan")}


# ── the label/rollout policy fingerprint (brief §2) ───────────────────────────────
#
# "any change to the fast player changes the label/rollout policy" — so the fingerprint is
# a digest of the fast player's SOURCE plus the knobs that select its behaviour.  It is
# recorded on every pair and every absolute row so the trainer can filter (the old 51k
# corpus predates the field and is therefore identifiable by its absence).

_FINGERPRINT_FILES = ("hand.py", "player.py", "sampling.py")


@functools.lru_cache(maxsize=1)
def source_digest(_stamp: str = "") -> str:
    """sha1 over the fast player's source files (``hand.py``, ``player.py``, ``sampling.py``)."""
    h = hashlib.sha1()
    here = Path(__file__).resolve().parent
    for name in _FINGERPRINT_FILES:
        p = here / name
        h.update(name.encode("utf-8"))
        h.update(p.read_bytes() if p.exists() else b"<missing>")
    return h.hexdigest()


def player_fingerprint(*, policy: str = "ev", budget: str = "fast", shop_tier: str = "rules",
                       epsilon_rollout: float = 0.02, extra: Optional[dict] = None) -> str:
    """``"<policy>-<budget>-<shop_tier>:<12 hex>"`` — changes whenever the fast player's code
    or its selected behaviour changes (W-EXTRACT landing flips it, by design)."""
    cfg = {"policy": policy, "budget": budget, "shop_tier": shop_tier,
           "epsilon_rollout": round(float(epsilon_rollout), 6), "extraction": has_extraction()}
    if extra:
        cfg.update(extra)
    blob = source_digest() + "|" + json.dumps(cfg, sort_keys=True, default=str)
    return f"{policy}-{budget}-{shop_tier}:{hashlib.sha1(blob.encode('utf-8')).hexdigest()[:12]}"


# ── W-EXTRACT feature detection (brief §5.2: "leave the hook + test ready") ───────
#
# W-EXTRACT owns hand.py / player.py and had not landed when this was built, so the exact
# entry-point name is unknown.  Every plausible one is probed, in preference order:
#   1. a GENERATOR of extraction lines           hand.extraction_lines / extraction_candidates
#   2. a generator on the player                 EVPlayer.extraction_actions
#   3. the per-action proc-EV term (brief §3.2)  hand.extraction_ev  → argmax over legal
# A generator may return actions, ``(action, ev)`` pairs or ``(action, ev, reason)`` triples,
# and may or may not take ``legal=``; all forms are accepted.  Anything that raises is
# treated as "not landed" (the campaign must never die on an interface guess).

_GENERATOR_NAMES = ("extraction_lines", "extraction_candidates", "extraction_actions")
_EV_TERM_NAMES = ("extraction_ev",)


def _hand_module():
    try:
        import hand as H
        return H
    except Exception:                      # noqa: BLE001
        return None


def _player_class():
    try:
        import player as P
        return getattr(P, "EVPlayer", None)
    except Exception:                      # noqa: BLE001
        return None


def extraction_entry_point() -> Optional[tuple]:
    """``(kind, owner, name)`` of W-EXTRACT's line generator / EV term, or ``None``.
    ``kind`` is ``"generator"`` or ``"ev_term"``; ``owner`` is ``"hand"`` or ``"player"``."""
    H = _hand_module()
    if H is not None:
        for name in _GENERATOR_NAMES:
            if callable(getattr(H, name, None)):
                return ("generator", "hand", name)
    cls = _player_class()
    if cls is not None:
        for name in _GENERATOR_NAMES:
            if callable(getattr(cls, name, None)):
                return ("generator", "player", name)
    if H is not None:
        for name in _EV_TERM_NAMES:
            if callable(getattr(H, name, None)):
                return ("ev_term", "hand", name)
    return None


def has_extraction() -> bool:
    return extraction_entry_point() is not None


def _try_call(fn, arg_forms):
    """Call ``fn`` with the first argument form that works; ``None`` if none does.  Every
    exception (not just ``TypeError``) moves on to the next form: the brief writes the term
    as ``extraction_ev(action, state)`` while ``hand.py`` landed it as ``(game, action)``,
    and passing a dict where a game is expected raises ``AttributeError``, not ``TypeError``.
    All forms exhausted ⇒ the hook is treated as unavailable, never as a campaign error."""
    for args, kwargs in arg_forms:
        try:
            return fn(*args, **kwargs)
        except Exception:                  # noqa: BLE001 — a raising hook is "not available"
            continue
    return None


def _normalise_lines(out) -> list:
    """A generator's return value → ``[(action_dict, ev_or_None)]``."""
    lines = []
    for item in (out or []):
        if isinstance(item, dict):
            lines.append((item, None))
        elif isinstance(item, (tuple, list)) and item and isinstance(item[0], dict):
            ev = float(item[1]) if len(item) > 1 and isinstance(item[1], (int, float)) else None
            lines.append((dict(item[0]), ev))
    return lines


def _ev_term_lines(H, name: str, game, legal: list, max_scan: int) -> list:
    """Score ``legal`` with W-EXTRACT's per-action proc-EV term.  The module-level
    ``hand.extraction_ev(game, action)`` rebuilds a whole ``HandAnalysis`` per call (~1-3 ms
    x hundreds of legal actions = seconds per decision), so when the same term also exists as
    a ``HandAnalysis`` METHOD one analysis is built and reused — 0.1 ms for the whole scan."""
    cls = getattr(H, "HandAnalysis", None)
    if cls is not None and callable(getattr(cls, name, None)):
        try:
            an = cls(game, legal=legal)
            if not getattr(an, "extract_on", True):
                return []                  # nothing on this board can be extracted
            method = getattr(an, name)
            out = []
            for a in legal[:max_scan]:
                ev = method(a)
                if isinstance(ev, (int, float)) and float(ev) > 0.0:
                    out.append((dict(a), float(ev)))
            return out
        except Exception:                  # noqa: BLE001 — fall back to the module function
            pass
    fn = getattr(H, name)
    out = []
    for a in legal[:max_scan]:
        ev = _try_call(fn, [((game, a), {}), ((a, game), {})])
        if isinstance(ev, (int, float)) and float(ev) > 0.0:
            out.append((dict(a), float(ev)))
    return out


def make_extraction_hook(player_obj=None, *, max_scan: int = 800) -> Optional[Callable]:
    """``hook(game, legal, avoid=None) -> action | None`` — the best extraction line at this
    state, or ``None`` when W-EXTRACT has not landed / offers nothing here."""
    ep = extraction_entry_point()
    if ep is None:
        return None
    kind, owner, name = ep
    H = _hand_module()

    def hook(game, legal, avoid=None):
        legal_keys = {action_key(a): dict(a) for a in legal}
        avoid_k = action_key(avoid) if avoid is not None else None
        if kind == "generator":
            fn = getattr(H, name) if owner == "hand" else getattr(player_obj, name, None)
            if fn is None:
                return None
            raw = _try_call(fn, [((game,), {"legal": legal}), ((game, legal), {}), ((game,), {})])
            lines = _normalise_lines(raw)
        else:
            lines = _ev_term_lines(H, name, game, legal, max_scan)
        cands = [(a, ev) for a, ev in lines
                 if action_key(a) in legal_keys and action_key(a) != avoid_k]
        if not cands:
            return None
        if any(ev is not None for _, ev in cands):
            cands.sort(key=lambda t: -(t[1] if t[1] is not None else -math.inf))
        return dict(legal_keys[action_key(cands[0][0])])

    return hook


# ── action identity ───────────────────────────────────────────────────────────────

def action_key(a: dict) -> str:
    """A FULL action identity.  ``hand._action_sort_key`` deliberately ignores ``item_idx`` /
    ``joker_idx`` / ``indices`` (two different shop buys collapse to one key), which would
    silently make a pair compare an action with itself — so pairs uses its own."""
    return json.dumps(a, sort_keys=True, default=str)


# ── the rules player's ranking ────────────────────────────────────────────────────

def rules_ranking(game, player_obj) -> list:
    """``[(action, ev)]`` from ``EVPlayer.explain``, filtered to ``legal_actions()`` and
    de-duplicated, best first.  The EVs are the FAST player's own units (P(clear) + banking
    at a hand/nemesis state, proxy gain at a shop/pack state) — see PAIRS_NOTES §2."""
    legal = {action_key(a): dict(a) for a in game.legal_actions()}
    out, seen = [], set()
    for item in player_obj.explain(game):
        a, ev = item[0], float(item[1])
        k = action_key(a)
        if k in legal and k not in seen:
            seen.add(k)
            out.append((dict(legal[k]), ev))
    return out


def make_ranker(policy_factory: Optional[Callable] = None, player_obj=None) -> Callable:
    """``ranker(match, actor) -> [(action, ev)]``.  With an ``EVPlayer`` it is the full
    ranking; with only a policy (the scripted fallback) it is the single chosen action at
    ev = 0.0 — enough for ``random`` pairs, never for ``close_call``."""
    if player_obj is not None:
        return lambda match, actor: rules_ranking(match.games[actor], player_obj)
    if policy_factory is None:
        raise ValueError("make_ranker needs a player_obj or a policy_factory")
    pol = policy_factory(0, 0)

    def ranker(match, actor):
        acts = match.legal_actions(actor)
        return [(dict(pol(match, actor, acts)), 0.0)] if acts else []
    return ranker


# ── pair selection ────────────────────────────────────────────────────────────────

def mix_sequence(n: int, rng: random.Random, mix: Optional[dict] = None,
                 extraction: bool = False) -> list:
    """``n`` requested ``pair_source``s in the target proportions, shuffled.  Without
    W-EXTRACT the ``greedy_vs_extract`` mass folds into ``close_call`` (brief §5.2:
    "emit close_call/random only")."""
    mix = dict(mix or DEFAULT_MIX)
    if not extraction:
        mix["close_call"] = mix.get("close_call", 0.0) + mix.pop("greedy_vs_extract", 0.0)
    total = sum(max(0.0, w) for w in mix.values()) or 1.0
    out, acc = [], 0.0
    for src in PAIR_SOURCES:
        w = max(0.0, mix.get(src, 0.0)) / total
        acc += w * n
        while len(out) < min(n, int(round(acc))):
            out.append(src)
    while len(out) < n:
        out.append("close_call" if "close_call" in mix else PAIR_SOURCES[-1])
    rng.shuffle(out)
    return out[:n]


@dataclass
class PairChoice:
    action_a: dict                 # the rules player's own choice (always branch a)
    action_b: dict
    source: str                    # the source actually realised
    requested: str                 # the source the mix asked for
    ev_a: float
    ev_b: float                    # NaN when b is outside the ranked head
    gap: float                     # ev_a - ev_b (NaN when ev_b is unknown)
    n_ranked: int
    n_legal: int


def _uniform_other(legal: list, avoid: dict, rng: random.Random) -> Optional[dict]:
    k = action_key(avoid)
    pool = [a for a in legal if action_key(a) != k]
    return dict(rng.choice(pool)) if pool else None


def choose_pair(match, actor: int, *, ranker: Callable, rng: random.Random, source: str,
                close_gap: float = DEFAULT_CLOSE_GAP, extraction: Optional[Callable] = None,
                cascade: bool = True) -> Optional[PairChoice]:
    """The two actions of one pair (brief §5.2).  Branch ``a`` is ALWAYS the rules player's
    own choice, so ``delta > 0`` means "the rules player was right".

    Fallback rule (``cascade``): a requested source this state cannot supply — a
    `close_call` whose top-2 gap is ≥ ``close_gap``, a `greedy_vs_extract` where the board
    has nothing to extract — tries the OTHER informative source before degrading to
    `random`; a requested `random` stays random (it is the deliberate 10 % of uninformed
    pairs, not a fallback bucket).  The realised source is recorded next to the requested
    one, and the realised mix is what gets reported."""
    game = match.games[actor]
    legal = match.legal_actions(actor)
    if len(legal) < 2:
        return None
    ranked = ranker(match, actor)
    if not ranked:
        ranked = [(dict(legal[0]), float("nan"))]
    a, ev_a = ranked[0]
    n_ranked, n_legal = len(ranked), len(legal)
    ev_by_key = {action_key(x): ev for x, ev in ranked}

    def made(b, src):
        ev_b = ev_by_key.get(action_key(b), float("nan"))
        gap = (ev_a - ev_b) if (ev_b == ev_b and ev_a == ev_a) else float("nan")
        return PairChoice(dict(a), dict(b), src, source, float(ev_a), float(ev_b), float(gap),
                          n_ranked, n_legal)

    order = [source]
    if cascade and source != "random":
        order += [s for s in ("greedy_vs_extract", "close_call") if s != source]
    for src in order:
        if src == "close_call" and n_ranked >= 2:
            b, ev_b = ranked[1]
            if action_key(b) != action_key(a) and (ev_a - ev_b) < close_gap:
                return made(b, "close_call")
        elif src == "greedy_vs_extract" and extraction is not None:
            b = extraction(game, legal, avoid=a)
            if b is not None:
                return made(b, "greedy_vs_extract")
    b = _uniform_other(legal, a, rng)
    if b is None:
        return None
    return made(b, "random")


# ── rolling the pair on shared worlds ─────────────────────────────────────────────

@dataclass
class PairRollout:
    outcomes_a: list                # per shared world, ACTOR's perspective (1/0 or race float)
    outcomes_b: list
    world_seeds: list
    delta: float
    delta_ci: float                 # 95 % half-width of the PAIRED difference
    y_a: float
    y_b: float
    ci_a: float
    ci_b: float
    var_a: float                    # per-world sample variances (ddof=1) — §4's measurement
    var_b: float
    var_d: float                    # sample variance of (a_i - b_i)
    cov: float
    rho: float
    reps: int = 1
    rep_means_a: list = field(default_factory=list)
    rep_means_b: list = field(default_factory=list)
    trunc_frac: float = 0.0
    determinized: bool = True
    seconds: float = 0.0
    decisions: int = 0
    policy_seconds: float = 0.0
    forced: int = 0
    n_rollouts: int = 0
    #: W-AUX: per-branch auxiliary targets, ``{player: {field: mean over worlds or None}}``,
    #: empty without an observer.  BOTH branches are recorded (brief §6b.1) on the SAME
    #: shared worlds, so ``aux_a`` and ``aux_b`` differ only by the branching action.
    aux_a: dict = field(default_factory=dict)
    aux_b: dict = field(default_factory=dict)


def _var(xs: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return sum((x - m) ** 2 for x in xs) / (n - 1)


def _cov(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (n - 1)


def step_branch(match, actor: int, action: dict):
    """The post-action match clone (a REAL state: the immediate transition uses the true
    stream, determinization happens per world inside ``labels.rollout``)."""
    m = match.clone()
    m.step(actor, dict(action))
    return m


#: how the shared world is applied relative to the branching action.
#:   ``step_then_determinize`` (FROZEN, brief §5.1): step a / b on the real state, then
#:       ``clone_determinized(w)`` inside each branch's rollout.  ``obs_a`` / ``obs_b`` are
#:       observations of REAL post-action states — the states V has to value.
#:   ``determinize_then_step``: determinize ONCE per world, then step both branches on
#:       clones of that single world.  Strictly tighter common random numbers (the immediate
#:       draw and the post-action pile are shared too), but a branch state is then a sampled
#:       world — one per world — so there is no single ``obs_a`` to store.  DIAGNOSTIC ONLY:
#:       it separates "the lever is weak" from "the coupling is weak" (PAIRS_NOTES §4).
COUPLINGS = ("step_then_determinize", "determinize_then_step")


def roll_pair(match, actor: int, action_a: dict, action_b: dict, *, n_worlds: int = DEFAULT_N_WORLDS,
              reps: int = 1, seed: int = 0, policy_factory: Optional[Callable] = None,
              max_ante: int = L.DEFAULT_MAX_ANTE, max_steps: int = L.DEFAULT_MAX_STEPS,
              race_cfg: _race.RaceConfig = _race.DEFAULT, determinize: bool = True,
              coupling: str = "step_then_determinize",
              observer_factory: Optional[Callable] = None) -> tuple:
    """``(PairRollout, match_a, match_b)``.  Both branches are rolled on the SAME world-seed
    list — that seed drives ``clone_determinized`` AND both rollout policies' ε-streams
    (``labels.rollout``), so the pairing is common random numbers end to end.

    ``reps > 1`` splits the worlds into ``reps`` DISJOINT blocks of ``n_worlds`` and records
    the per-block means: the direct (replication) variance measurement of §4.  The label
    itself then simply uses all ``reps × n_worlds`` worlds.  ``coupling``: see ``COUPLINGS``
    — the default is the brief's frozen order.

    ``observer_factory`` (W-AUX, additive): threaded into BOTH branches' rollouts, one fresh
    observer per rollout; the per-world results are averaged into ``PairRollout.aux_a`` /
    ``.aux_b``.  ``None`` (the default) leaves every number here bit-identical."""
    if policy_factory is None:
        policy_factory = L.make_policy_factory()
    if coupling not in COUPLINGS:
        raise ValueError(f"coupling must be one of {COUPLINGS}, got {coupling!r}")
    t0 = time.perf_counter()
    pre = coupling == "step_then_determinize"
    ma = step_branch(match, actor, action_a) if pre else None
    mb = step_branch(match, actor, action_b) if pre else None
    total = int(n_worlds) * max(1, int(reps))
    world_seeds = [int(seed) * 1_000_003 + i for i in range(total)]
    outs_a, outs_b = [], []
    aux_worlds_a, aux_worlds_b = [], []
    trunc = dec = forced = 0
    pol_s = 0.0
    det = True
    for w in world_seeds:
        if pre:
            xa, xb, det_roll = ma, mb, determinize
        else:
            world = match.clone_determinized(w) if determinize else match.clone()
            xa = step_branch(world, actor, action_a)
            xb = step_branch(world, actor, action_b)
            det_roll = False                     # the world is already a fresh sample
            if ma is None:
                ma, mb = xa, xb                  # world 0's branches carry the diagnostic obs
        ra = L.rollout(xa, seed=w, policy_factory=policy_factory, max_ante=max_ante,
                       max_steps=max_steps, race_cfg=race_cfg, determinize=det_roll,
                       observer_factory=observer_factory)
        rb = L.rollout(xb, seed=w, policy_factory=policy_factory, max_ante=max_ante,
                       max_steps=max_steps, race_cfg=race_cfg, determinize=det_roll,
                       observer_factory=observer_factory)
        if not pre:
            det = det and bool(determinize)
        if ra.aux is not None:
            aux_worlds_a.append(ra.aux)
        if rb.aux is not None:
            aux_worlds_b.append(rb.aux)
        outs_a.append(ra.p0_win if actor == 0 else 1.0 - ra.p0_win)
        outs_b.append(rb.p0_win if actor == 0 else 1.0 - rb.p0_win)
        trunc += int(ra.truncated) + int(rb.truncated)
        dec += ra.decisions + rb.decisions
        pol_s += ra.policy_seconds + rb.policy_seconds
        forced += ra.forced + rb.forced
        if pre:
            det = det and ra.determinized and rb.determinized
    n = len(outs_a)
    diffs = [x - y for x, y in zip(outs_a, outs_b)]
    var_d = _var(diffs)
    va, vb = _var(outs_a), _var(outs_b)
    cab = _cov(outs_a, outs_b)
    denom = math.sqrt(va * vb)
    rho = (cab / denom) if denom > 0 else 0.0
    delta = (sum(diffs) / n) if n else 0.0
    delta_ci = 1.96 * math.sqrt(var_d / n) if n >= 2 else 1.0
    aux_a, aux_b = {}, {}
    if aux_worlds_a or aux_worlds_b:
        import aux_targets as AX
        aux_a = {p: AX.aggregate(aux_worlds_a, p) for p in (0, 1)}
        aux_b = {p: AX.aggregate(aux_worlds_b, p) for p in (0, 1)}
    rep_a, rep_b = [], []
    if reps > 1:
        for r in range(int(reps)):
            blk = slice(r * n_worlds, (r + 1) * n_worlds)
            rep_a.append(sum(outs_a[blk]) / n_worlds)
            rep_b.append(sum(outs_b[blk]) / n_worlds)
    pr = PairRollout(
        outcomes_a=outs_a, outcomes_b=outs_b, world_seeds=world_seeds,
        delta=float(delta), delta_ci=float(delta_ci),
        y_a=float(sum(outs_a) / n) if n else float("nan"),
        y_b=float(sum(outs_b) / n) if n else float("nan"),
        ci_a=L._ci_of(outs_a), ci_b=L._ci_of(outs_b),
        var_a=float(va), var_b=float(vb), var_d=float(var_d), cov=float(cab), rho=float(rho),
        reps=int(reps), rep_means_a=rep_a, rep_means_b=rep_b,
        trunc_frac=trunc / (2 * n) if n else 0.0, determinized=det,
        seconds=time.perf_counter() - t0, decisions=dec, policy_seconds=pol_s, forced=forced,
        n_rollouts=2 * n, aux_a=aux_a, aux_b=aux_b)
    return pr, ma, mb


# ── the frozen shard record (brief §5.3) ──────────────────────────────────────────

@dataclass
class PairRow:
    """One pair.  ``rec`` is the frozen JSON record; ``obs_a`` / ``obs_b`` are the encoder-v2
    observation dicts, stored as npz arrays exactly as ``dataset.save_shard`` stores ``obs``
    (they are NOT inside the JSON, for the same reason ``meta_json`` does not hold ``obs``)."""
    obs_a: dict
    obs_b: dict
    rec: dict

    def to_json(self) -> str:
        return json.dumps(self.rec, sort_keys=True, default=str)


def _strict_json(x):
    """NaN / ±Inf → ``None`` (a shard must be STRICT JSON: a non-finite float round-trips
    unequal under ``json`` and is rejected outright by other parsers.  The only non-finite
    values here are "unknown" — an ev for an action outside the ranked head)."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _strict_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_strict_json(v) for v in x]
    return x


def pair_record(snapshot, choice: PairChoice, pr: PairRollout, *, fingerprint: str,
                meta: Optional[dict] = None, aux: Optional[dict] = None) -> dict:
    """The frozen §5.3 record (minus ``obs_a`` / ``obs_b``, which are npz arrays).

    ``aux`` (W-AUX, brief §6b.3 — ADDITIVE, the frozen fields are untouched): the auxiliary
    targets of both branches, ``{"a": {...}, "b": {...}}``, in the ACTOR's perspective (the
    perspective ``obs_a`` / ``obs_b`` encode).  Absent entirely when the campaign did not
    record them, so an old shard simply has no ``aux`` key and the trainer masks it."""
    rec = {
        "kind": "pair",
        "seed": snapshot.seed,
        "step": int(snapshot.step),
        "actor": int(snapshot.actor),
        "state_kind": snapshot.kind,
        "ante": int(snapshot.ante),
        "player_fingerprint": fingerprint,
        "pair_source": choice.source,
        "action_a": choice.action_a,
        "action_b": choice.action_b,
        "n_worlds": len(pr.outcomes_a),
        "outcomes_a": [round(o, 6) for o in pr.outcomes_a],
        "outcomes_b": [round(o, 6) for o in pr.outcomes_b],
        "delta": float(pr.delta),
        "delta_ci": float(pr.delta_ci),
        "meta": dict(meta or {}),
    }
    from balatro_sim.behavior_stamp import ENGINE_BEHAVIOR_STAMP
    rec["meta"].setdefault("engine_stamp", ENGINE_BEHAVIOR_STAMP)
    if aux:
        rec["aux"] = dict(aux)
    return _strict_json(rec)


def _pair_meta(snapshot, choice: PairChoice, pr: PairRollout, cfg: dict) -> dict:
    return {
        "y_a": pr.y_a, "y_b": pr.y_b, "ci_a": pr.ci_a, "ci_b": pr.ci_b,
        "var_a": pr.var_a, "var_b": pr.var_b, "var_d": pr.var_d, "cov_ab": pr.cov, "rho": pr.rho,
        "reps": pr.reps, "rep_means_a": [round(x, 6) for x in pr.rep_means_a],
        "rep_means_b": [round(x, 6) for x in pr.rep_means_b],
        "n_worlds_per_rep": cfg.get("n_worlds"),
        "world_seeds": pr.world_seeds,
        "requested_source": choice.requested, "ev_a": choice.ev_a, "ev_b": choice.ev_b,
        "ev_gap": choice.gap, "n_ranked": choice.n_ranked, "n_legal": choice.n_legal,
        "close_gap": cfg.get("close_gap"),
        "trunc_frac": pr.trunc_frac, "determinized": pr.determinized, "forced": pr.forced,
        "lives": [int(g.lives) for g in snapshot.match.games],
        "n_rollouts": pr.n_rollouts,
        "selfplay": {k: v for k, v in snapshot.selfplay.items() if k != "decisions_seen"},
        "rollout": {k: cfg.get(k) for k in ("policy", "budget", "shop_tier", "epsilon_rollout",
                                            "max_ante", "encoder", "deck_key", "stake", "lives",
                                            "coupling")},
    }


# ── pair shards (mirrors dataset.py's conventions) ────────────────────────────────

@dataclass
class PairShard:
    obs_a: dict
    obs_b: dict
    columns: dict
    records: list
    path: Optional[str] = None

    def __len__(self) -> int:
        return len(self.records)


def _pair_columns(records: Sequence[dict]) -> dict:
    cols = {}
    for name in PAIR_COLUMNS:
        cols[name] = np.asarray([r.get(name, _PAIR_COLUMN_DEFAULTS[name]) for r in records],
                                dtype=_PAIR_COLUMN_DTYPES[name])
    return cols


def save_pair_shard(path, rows: Sequence[PairRow]) -> Path:
    """One compressed ``.npz``: ``obs_a__<key>`` / ``obs_b__<key>`` stacked on axis 0, the
    typed columns of ``PAIR_COLUMNS``, and ``pair_json`` (one frozen record per row).
    Written atomically (temp + ``os.replace``), exactly like ``dataset.save_shard``."""
    rows = list(rows)
    if not rows:
        raise ValueError("save_pair_shard: no rows")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].obs_a.keys())
    payload = {"version": np.asarray(SHARD_VERSION), "shard_kind": np.asarray("pair"),
               "obs_keys": np.asarray(keys)}
    for k in keys:
        payload[f"obs_a__{k}"] = np.stack([np.asarray(r.obs_a[k]) for r in rows])
        payload[f"obs_b__{k}"] = np.stack([np.asarray(r.obs_b[k]) for r in rows])
    recs = [dict(r.rec) for r in rows]
    payload.update(_pair_columns(recs))
    payload["pair_json"] = np.asarray(
        [json.dumps(r, sort_keys=True, default=str, allow_nan=False) for r in recs])
    tmp = path.with_suffix(path.suffix + ".tmp.npz")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **payload)
    os.replace(tmp, path)
    return path


def load_pair_shard(path) -> PairShard:
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        keys = [str(k) for k in z["obs_keys"]]
        obs_a = {k: z[f"obs_a__{k}"] for k in keys}
        obs_b = {k: z[f"obs_b__{k}"] for k in keys}
        columns = {n: z[n] for n in PAIR_COLUMNS if n in z.files}
        records = [json.loads(s) for s in z["pair_json"]]
    return PairShard(obs_a=obs_a, obs_b=obs_b, columns=columns, records=records, path=str(path))


class PairDataset:
    """Many pair shards concatenated (the pair-side twin of ``dataset.LabelDataset``)."""

    def __init__(self, obs_a: dict, obs_b: dict, columns: dict, records: list, sources=()):
        self.obs_a, self.obs_b = obs_a, obs_b
        self.columns = columns
        self.records = records
        self.sources = list(sources)

    @classmethod
    def from_shards(cls, shards: Iterable) -> "PairDataset":
        loaded = [s if isinstance(s, PairShard) else load_pair_shard(s) for s in shards]
        loaded = [s for s in loaded if len(s)]
        if not loaded:
            return cls({}, {}, {n: np.zeros((0,), _PAIR_COLUMN_DTYPES[n]) for n in PAIR_COLUMNS}, [])
        keys = list(loaded[0].obs_a.keys())
        for s in loaded[1:]:
            if list(s.obs_a.keys()) != keys:
                raise ValueError(f"pair shard {s.path} has obs keys {list(s.obs_a.keys())} != {keys}")
        obs_a = {k: np.concatenate([s.obs_a[k] for s in loaded]) for k in keys}
        obs_b = {k: np.concatenate([s.obs_b[k] for s in loaded]) for k in keys}
        columns = {n: np.concatenate([s.columns[n] for s in loaded]) for n in PAIR_COLUMNS
                   if all(n in s.columns for s in loaded)}
        records = [r for s in loaded for r in s.records]
        return cls(obs_a, obs_b, columns, records, sources=[s.path for s in loaded])

    @classmethod
    def load(cls, spec) -> "PairDataset":
        return cls.from_shards(DS.list_shards(spec))

    def __len__(self) -> int:
        return len(self.records)

    def seeds(self) -> list:
        return sorted(set(self.columns["seed"].tolist())) if "seed" in self.columns else []


# ── §4: THE MEASUREMENT THAT JUSTIFIES THE LEVER ──────────────────────────────────

def variance_report(records: Sequence[dict], *, n_worlds: Optional[int] = None) -> dict:
    """Empirical variance of the PAIRED estimator vs two INDEPENDENT absolute labels at
    EQUAL rollout budget, from the same shards.

    Two independent measurements, both unbiased and both at ``n`` rollouts per branch:

    * ``crn`` (all pairs, free): per pair, ``s²_d = var(a_i − b_i)`` and ``s²_a + s²_b`` are
      unbiased for ``n·var`` of the paired and the unpaired difference of means respectively
      (the unpaired estimator is exactly "label a on one world list, label b on another" —
      the same ``2n`` rollouts).  Aggregated as a ratio of sums (a variance-weighted mean,
      so a pair whose outcomes are all identical contributes 0/0 = nothing, not a NaN).
    * ``direct`` (``reps ≥ 2`` pairs only): the estimator is REPLICATED on disjoint world
      blocks.  ``var(Δ_r)`` across blocks measures the paired estimator's variance directly;
      ``var(A_r) + var(B_r)`` measures the unpaired one directly (blocks are independent, so
      an ``A`` block and a ``B`` block from different replicates are an independent pair of
      absolute labels).  No modelling assumption at all — this is the audit of ``crn``.
    """
    used, rhos = [], []
    sum_d, sum_ab = 0.0, 0.0
    resolved = 0
    by_source: dict = {}
    by_kind: dict = {}
    dir_p, dir_u = [], []
    n_ref = None
    for r in records:
        meta = r.get("meta", {})
        oa, ob = r.get("outcomes_a", []), r.get("outcomes_b", [])
        n = len(oa)
        if n < 2 or len(ob) != n:
            continue
        va = meta.get("var_a", _var(oa))
        vb = meta.get("var_b", _var(ob))
        vd = meta.get("var_d", _var([x - y for x, y in zip(oa, ob)]))
        n_ref = n_ref or n
        used.append(r)
        sum_d += vd
        sum_ab += va + vb
        if va > 0 and vb > 0:
            rhos.append(meta.get("rho", _cov(oa, ob) / math.sqrt(va * vb)))
        if abs(float(r.get("delta", 0.0))) > float(r.get("delta_ci", 1.0)) > 0:
            resolved += 1
        for bucket, key in ((by_source, r.get("pair_source", "?")), (by_kind, r.get("state_kind", "?"))):
            d = bucket.setdefault(key, {"n": 0, "sum_d": 0.0, "sum_ab": 0.0, "resolved": 0})
            d["n"] += 1
            d["sum_d"] += vd
            d["sum_ab"] += va + vb
            d["resolved"] += int(abs(float(r.get("delta", 0.0))) > float(r.get("delta_ci", 1.0)) > 0)
        ra, rb = meta.get("rep_means_a") or [], meta.get("rep_means_b") or []
        if len(ra) >= 2 and len(ra) == len(rb):
            dir_p.append(_var([x - y for x, y in zip(ra, rb)]))
            dir_u.append(_var(ra) + _var(rb))

    def ratio(num, den):
        return float(num / den) if den > 0 else float("nan")

    n_eff = n_worlds or n_ref or 0
    out = {
        "n_pairs": len(used),
        "n_worlds": n_eff,
        "crn": {
            "var_paired_per_rollout_block": ratio(sum_d, len(used)) if used else float("nan"),
            "var_unpaired_per_rollout_block": ratio(sum_ab, len(used)) if used else float("nan"),
            "var_reduction_factor": ratio(sum_ab, sum_d),
            "mean_rho": float(sum(rhos) / len(rhos)) if rhos else float("nan"),
            "n_with_rho": len(rhos),
            "se_paired_at_n": math.sqrt(ratio(sum_d, len(used)) / n_eff) if (used and n_eff) else float("nan"),
            "se_unpaired_at_n": math.sqrt(ratio(sum_ab, len(used)) / n_eff) if (used and n_eff) else float("nan"),
        },
        "direct": {
            "n_pairs": len(dir_p),
            "var_paired": float(sum(dir_p) / len(dir_p)) if dir_p else float("nan"),
            "var_unpaired": float(sum(dir_u) / len(dir_u)) if dir_u else float("nan"),
            "var_reduction_factor": ratio(sum(dir_u), sum(dir_p)) if dir_p else float("nan"),
        },
        "resolved_frac": (resolved / len(used)) if used else float("nan"),
        "by_source": {k: {"n": v["n"], "var_reduction_factor": ratio(v["sum_ab"], v["sum_d"]),
                          "resolved_frac": v["resolved"] / v["n"]} for k, v in sorted(by_source.items())},
        "by_state_kind": {k: {"n": v["n"], "var_reduction_factor": ratio(v["sum_ab"], v["sum_d"]),
                              "resolved_frac": v["resolved"] / v["n"]} for k, v in sorted(by_kind.items())},
    }
    return out


def mix_report(records: Sequence[dict]) -> dict:
    """Realised ``pair_source`` / ``state_kind`` mix (and requested-vs-realised fallbacks)."""
    n = len(records)
    src, kind, req, cross = {}, {}, {}, {}
    for r in records:
        s = r.get("pair_source", "?")
        k = r.get("state_kind", "?")
        q = r.get("meta", {}).get("requested_source", "?")
        src[s] = src.get(s, 0) + 1
        kind[k] = kind.get(k, 0) + 1
        req[q] = req.get(q, 0) + 1
        cross[f"{q}->{s}"] = cross.get(f"{q}->{s}", 0) + 1
    frac = lambda d: {k: v / n for k, v in sorted(d.items())} if n else {}   # noqa: E731
    return {"n": n, "pair_source": dict(sorted(src.items())), "pair_source_frac": frac(src),
            "state_kind": dict(sorted(kind.items())), "state_kind_frac": frac(kind),
            "requested": dict(sorted(req.items())), "requested_to_realised": dict(sorted(cross.items()))}


# ── the worker job ────────────────────────────────────────────────────────────────

def _rules_player(seed: int, budget: str, shop_tier: str):
    """The SAME analytic player the rollouts use, ε = 0 — its ranking is what a pair is
    selected around (``close_call`` = its top-2)."""
    import player as P
    kw = {"budget": budget, "seed": int(seed), "epsilon": 0.0}
    if shop_tier == "stats":
        kw["stats"] = L.load_stats_module()
    return P.EVPlayer(**kw)


def pair_job(payload: dict) -> dict:
    """One pool job: self-play ``payload["seed"]`` → snapshots → one pair per snapshot.

    payload keys (all optional but ``seed``): ``n_states`` 6, ``n_worlds`` 8, ``reps`` 1
    (the §4 replication probe), ``mix`` (default ``DEFAULT_MIX``), ``close_gap`` 0.03,
    ``policy`` "auto", ``budget`` "fast", ``shop_tier`` "rules", ``epsilon_selfplay`` 0.1,
    ``epsilon_rollout`` 0.02, ``policy_seed`` 0, ``rollout_seed`` 0, ``encoder`` "auto",
    ``max_ante`` 12, ``deck_key`` "b_red", ``stake`` 1, ``lives`` 4, ``allow_clairvoyant``
    False, ``aux`` False (W-AUX: record ``aux_targets.AUX_SPECS`` on BOTH branches of every
    pair from the rollouts this job already runs -> an additive ``aux`` key on the pair
    record and on each absolute row's ``meta``; brief §6b).

    Returns ``{"pairs": [{"obs_a","obs_b","rec"}...], "rows": [{"obs","y","meta"}...],
    "timing": {...}, "selfplay": {...}, "skipped": n}`` — ``rows`` are the two absolute
    label rows every pair also yields (brief §5.3)."""
    seed = str(payload["seed"])
    n_states = int(payload.get("n_states", 6))
    n_worlds = int(payload.get("n_worlds", DEFAULT_N_WORLDS))
    reps = max(1, int(payload.get("reps", 1)))
    close_gap = float(payload.get("close_gap", DEFAULT_CLOSE_GAP))
    mix = payload.get("mix") or DEFAULT_MIX
    policy = payload.get("policy", "auto")
    budget = payload.get("budget", "fast")
    shop_tier = payload.get("shop_tier", "rules")
    eps_sp = float(payload.get("epsilon_selfplay", 0.1))
    eps_ro = float(payload.get("epsilon_rollout", 0.02))
    policy_seed = int(payload.get("policy_seed", 0))
    rollout_seed = int(payload.get("rollout_seed", 0))
    max_ante = int(payload.get("max_ante", L.DEFAULT_MAX_ANTE))
    deck_key = payload.get("deck_key", "b_red")
    stake = payload.get("stake", 1)
    lives = int(payload.get("lives", 4))
    allow_clair = bool(payload.get("allow_clairvoyant", False))
    coupling = payload.get("coupling", "step_then_determinize")
    aux_on = bool(payload.get("aux", False))           # W-AUX: record auxiliary targets
    observer_factory, aux_version = None, None
    if aux_on:
        import aux_targets as AX
        observer_factory = AX.make_recorder_factory((0, 1))
        aux_version = AX.AUX_VERSION
    if not allow_clair and not L.has_determinize():
        raise RuntimeError("MLBMatch.clone_determinized (W2) is missing: paired rollouts would "
                           "be clairvoyant; pass allow_clairvoyant=True for plumbing tests only")
    if policy == "auto":
        policy = "ev" if L.has_ev_player() else "scripted"

    encode = L.make_encoder(payload.get("encoder", "auto"))
    sp_factory = L.make_policy_factory(policy, budget=budget, epsilon=eps_sp, shop_tier=shop_tier)
    ro_factory = L.make_policy_factory(policy, budget=budget, epsilon=eps_ro, shop_tier=shop_tier)
    player_obj = _rules_player(policy_seed * 7919 + 1, budget, shop_tier) if policy == "ev" else None
    ranker = make_ranker(policy_factory=sp_factory, player_obj=player_obj)
    hook = make_extraction_hook(player_obj)
    fingerprint = player_fingerprint(policy=policy, budget=budget, shop_tier=shop_tier,
                                     epsilon_rollout=eps_ro)
    cfg = {"policy": policy, "budget": budget, "shop_tier": shop_tier, "epsilon_rollout": eps_ro,
           "max_ante": max_ante, "encoder": payload.get("encoder", "auto"), "deck_key": deck_key,
           "stake": stake, "lives": lives, "n_worlds": n_worlds, "close_gap": close_gap,
           "reps": reps, "mix": dict(mix), "coupling": coupling}

    t0 = time.perf_counter()
    per_kind = payload.get("per_kind")
    if per_kind is None:
        share = max(1, int(math.ceil(n_states / len(PAIRABLE_KINDS))))
        per_kind = {k: (share if k in PAIRABLE_KINDS else 0) for k in L.STATE_KINDS}
    snaps = L.sample_states(seed, n_states=n_states, per_kind=per_kind, policy_factory=sp_factory,
                            policy=policy, budget=budget, epsilon=eps_sp, policy_seed=policy_seed,
                            deck_key=deck_key, stake=stake, lives=lives, max_ante=max_ante,
                            pvp_protocol=payload.get("pvp_protocol", "canonical"))
    t_sp = time.perf_counter() - t0

    rng = random.Random(f"pairs:{seed}:{policy_seed}:{rollout_seed}")
    wanted = mix_sequence(len(snaps), rng, mix, extraction=hook is not None)
    pairs, rows = [], []
    skipped = 0
    t_pair, n_ro, dec, pol_s = 0.0, 0, 0, 0.0
    for i, s in enumerate(snaps):
        choice = choose_pair(s.match, s.actor, ranker=ranker, rng=rng, source=wanted[i],
                             close_gap=close_gap, extraction=hook)
        if choice is None:
            skipped += 1
            continue
        base = rollout_seed * 7919 + s.step * 31 + i
        pr, ma, mb = roll_pair(s.match, s.actor, choice.action_a, choice.action_b,
                               n_worlds=n_worlds, reps=reps, seed=base, policy_factory=ro_factory,
                               max_ante=max_ante, coupling=coupling,
                               observer_factory=observer_factory)
        t_pair += pr.seconds
        n_ro += pr.n_rollouts
        dec += pr.decisions
        pol_s += pr.policy_seconds
        meta = _pair_meta(s, choice, pr, cfg)
        aux = None
        if pr.aux_a or pr.aux_b:
            aux = {"a": pr.aux_a.get(s.actor, {}), "b": pr.aux_b.get(s.actor, {}),
                   "version": aux_version}
            meta["aux_version"] = aux_version
        rec = pair_record(s, choice, pr, fingerprint=fingerprint, meta=meta, aux=aux)
        obs_a, obs_b = encode(ma, s.actor), encode(mb, s.actor)
        pairs.append({"obs_a": obs_a, "obs_b": obs_b, "rec": rec})
        pair_id = f"{s.seed}:{s.step}:{s.actor}"
        for branch, obs, y, ci, action in (("a", obs_a, pr.y_a, pr.ci_a, choice.action_a),
                                           ("b", obs_b, pr.y_b, pr.ci_b, choice.action_b)):
            row_aux = None
            if aux is not None:
                # the same branch aux the pair record carries, on the absolute row that
                # encodes that branch — the trainer reads absolute-row aux out of `meta`.
                row_aux = _strict_json(aux[branch])
            rows.append({"obs": obs, "y": float(y), "meta": {
                "seed": s.seed, "step": s.step, "player": int(s.actor), "actor": int(s.actor),
                "kind": s.kind, "ante": int(s.ante), "ci": float(ci),
                "n_rollouts": len(pr.outcomes_a), "trunc_frac": pr.trunc_frac,
                "determinized": pr.determinized, "player_fingerprint": fingerprint,
                "from_pair": pair_id, "branch": branch, "pair_source": choice.source,
                "action": action, "post_action": True,
                "outcomes": [round(o, 4) for o in (pr.outcomes_a if branch == "a" else pr.outcomes_b)],
                "lives": [int(g.lives) for g in s.match.games],
                "selfplay": {k: v for k, v in s.selfplay.items() if k != "decisions_seen"},
                "budget": budget, "shop_tier": shop_tier, "epsilon_rollout": eps_ro,
                **({"aux": row_aux, "aux_version": aux_version} if row_aux is not None else {}),
            }})
    total = time.perf_counter() - t0
    return {
        "pairs": pairs,
        "rows": rows,
        "selfplay": snaps[0].selfplay if snaps else {},
        "skipped": skipped,
        "player_fingerprint": fingerprint,
        "timing": {"selfplay_s": t_sp, "pair_s": t_pair, "n_snapshots": len(snaps),
                   "n_pairs": len(pairs), "n_rollouts": n_ro, "rollout_decisions": dec,
                   "rollout_policy_s": pol_s, "total_s": total,
                   "ms_per_rollout": (1000 * t_pair / n_ro) if n_ro else 0.0,
                   "decisions_per_rollout": (dec / n_ro) if n_ro else 0.0,
                   "policy_frac": (pol_s / t_pair) if t_pair else 0.0,
                   "s_per_pair": (t_pair / len(pairs)) if pairs else 0.0},
    }


def pairs_from_result(result: dict) -> list:
    return [PairRow(p["obs_a"], p["obs_b"], p["rec"]) for p in result["pairs"]]


def rows_from_result(result: dict) -> list:
    """The absolute label rows a pair job also yields (``dataset.LabelRow``)."""
    return [DS.LabelRow(r["obs"], float(r["y"]), r["meta"]) for r in result["rows"]]
