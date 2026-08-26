"""
race.py — the MLB "lives race" calculator (Phase 5 rev 2, W5).

Two uses:

1. **Label truncation.**  A determinized label rollout (``labels.py``) stops at
   ``ante > max_ante`` (12).  The rollout's outcome is then *not* 1/0 but
   ``p_win(my_curve, their_curve, my_lives, their_lives, ante)`` — the probability that the
   player wins the remaining race of lives, given each side's observed Nemesis-score curve.
2. **The advisor's "race" number** (W6): the same function on a live match state, with
   curves fitted from ``match.pvp_log`` by ``curve_from_history``.

The model (every assumption is listed; nothing here rolls anything out):

* A player's Nemesis score at ante ``a`` is log-normal: ``log10(score) ~ N(mu(a), sigma(a))``,
  independent across antes and across players.  ``mu`` is LINEAR in the ante beyond the
  last observed point (``Curve.slope``, log10 chips per ante).  Scores are a property of the
  build; the opponent's score does not change mine.
* An ante is Small -> Big -> Boss slot.  Small/Big are regular blinds: ``P(fail) =
  P(score < target)`` under the same distribution (the blind target is
  ``blind_base_chips(ante, idx, stake_scaling) * ante_scaling``), floored at
  ``cfg.blind_fail_floor`` (misplays, boss effects, deck-outs).  Failing a regular blind costs
  ONE life and the run proceeds (MLB rule; the engine's ``_mlb_fail_round``).  The ante-1
  Boss (before ``pvp_start_round``) is a regular blind too.
* From ``pvp_start_round`` the Boss slot is the Nemesis: the lower scorer loses a life, an
  exact tie (``cfg.p_tie``, default 0) costs nobody.  ``P(I lose) = Phi((mu_t - mu_m) /
  sqrt(sigma_m^2 + sigma_t^2))`` (difference of two independent normals).
* Lives are lost at most once per blind (``lose_life``'s round blocker).  When both hit 0 in
  the same blind the match is a coin flip (the engine ends it on whoever steps first).
* Comeback money ($4/life lost, paid at the next cash-out) is NOT modelled as a curve shift;
  neither are skips, tags, vouchers or the boss kind.  The curve is meant to absorb them —
  it is fitted to what the build actually scored.
* The chain is solved by dynamic programming over (ante, my lives, their lives) for
  ``cfg.horizon_antes`` antes, then closed with the exact negative-binomial race at the
  last ante's Nemesis probability (no regular-blind losses past the horizon).  Past ante
  ~12 the endless targets grow super-exponentially while a linear-in-log10 curve does not,
  so regular blinds fail with probability -> 1 for both players and the chain terminates on
  its own well inside the horizon; the closed-form tail is a safety net, not the answer.

Pure Python, no numpy, no engine state: importable from anywhere (``constants`` is the only
engine module touched, for the blind target table).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Callable, Optional, Sequence, Union

try:                                                    # the engine's blind target table
    from balatro_sim.constants import blind_base_chips as _blind_base_chips   # type: ignore
except Exception:                                       # pragma: no cover — standalone import
    import _bootstrap  # noqa: F401  (puts engine on sys.path, fork-guarded)
    from balatro_sim.constants import blind_base_chips as _blind_base_chips   # type: ignore

__all__ = [
    "RaceConfig", "DEFAULT", "Curve", "as_curve", "prior_curve", "curve_from_history",
    "p_win", "p_lose_nemesis", "p_fail_blind", "race_table", "closed_form_race",
    "log10_target",
]


# ── configuration ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RaceConfig:
    horizon_antes: int = 24          # DP depth; the closed-form race closes the tail
    pvp_start_round: int = 2         # Nemesis at the Boss slot from this ante (MLB default)
    regular_blinds: int = 2          # Small + Big per ante
    stake_scaling: int = 1           # 1 = White/Red stake (constants.BLIND_AMOUNTS_BY_SCALING)
    ante_scaling: float = 1.0        # deck ante scaling (Plasma x2)
    blind_fail_floor: float = 0.01   # min P(fail a regular blind) — misplays / boss effects / deck-outs
    blind_fail_ceiling: float = 0.995
    p_tie: float = 0.0               # explicit tie mass at the Nemesis (nobody loses)
    default_slope: float = 0.35      # log10 chips per ante when no slope can be fitted
    slope_min: float = 0.05          # clipping for fitted slopes
    slope_max: float = 1.5
    sigma_prior: float = 0.35        # log10 spread before any evidence (factor ~2.2)
    sigma_prior_weight: float = 2.0  # pseudo-observations for the residual-variance shrinkage
    sigma_floor: float = 0.15
    prior_margin: float = 0.15       # prior curve: log10(Boss target) + margin
    fit_window: int = 4              # Nemeses used by curve_from_history (matches the opp block)


DEFAULT = RaceConfig()


# ── curves ────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Curve:
    """Per-ante (mu, sigma) of log10 Nemesis score, tabulated from ``ante0`` and extended
    linearly with ``slope`` (log10 per ante) beyond the table on both sides.  ``n_obs`` is
    informational (how many observed Nemeses the fit used)."""
    ante0: int
    mu: tuple
    sigma: tuple
    slope: float
    n_obs: int = 0

    def __post_init__(self):
        if len(self.mu) == 0 or len(self.mu) != len(self.sigma):
            raise ValueError("Curve needs >= 1 (mu, sigma) pair, same length")

    def at(self, ante: int) -> tuple:
        i = int(ante) - self.ante0
        n = len(self.mu)
        if 0 <= i < n:
            return float(self.mu[i]), float(self.sigma[i])
        if i >= n:
            return float(self.mu[-1]) + self.slope * (i - n + 1), float(self.sigma[-1])
        return float(self.mu[0]) + self.slope * i, float(self.sigma[0])

    def shifted(self, d_mu: float) -> "Curve":
        return replace(self, mu=tuple(m + d_mu for m in self.mu))

    def describe(self, antes: Sequence[int]) -> list:
        return [(a, *self.at(a)) for a in antes]


CurveLike = Union[Curve, dict, Callable[[int], tuple]]


def as_curve(x: CurveLike, *, cfg: RaceConfig = DEFAULT) -> Curve:
    """Coerce ``Curve`` | ``{ante: (mu, sigma) | mu}`` (contiguous antes; slope from the last
    two points, else ``cfg.default_slope``) | ``callable(ante) -> (mu, sigma)`` (tabulated
    over ``cfg.horizon_antes`` antes from ante 1)."""
    if isinstance(x, Curve):
        return x
    if isinstance(x, dict):
        if not x:
            raise ValueError("empty curve dict")
        antes = sorted(int(a) for a in x)
        if antes != list(range(antes[0], antes[-1] + 1)):
            raise ValueError(f"curve antes must be contiguous, got {antes}")
        mu, sigma = [], []
        for a in antes:
            v = x[a] if a in x else x[str(a)]
            if isinstance(v, (tuple, list)):
                mu.append(float(v[0])); sigma.append(float(v[1]))
            else:
                mu.append(float(v)); sigma.append(cfg.sigma_prior)
        slope = (mu[-1] - mu[-2]) if len(mu) >= 2 else cfg.default_slope
        slope = min(max(slope, cfg.slope_min), cfg.slope_max)
        return Curve(antes[0], tuple(mu), tuple(sigma), slope, n_obs=len(mu))
    if callable(x):
        pts = [x(a) for a in range(1, 1 + cfg.horizon_antes + 16)]
        mu = tuple(float(p[0]) for p in pts)
        sigma = tuple(float(p[1]) for p in pts)
        return Curve(1, mu, sigma, cfg.default_slope)
    raise TypeError(f"cannot interpret {type(x).__name__} as a curve")


def log10_target(ante: int, blind_idx: int, *, cfg: RaceConfig = DEFAULT) -> float:
    """log10 of the regular-blind chip target (0 Small / 1 Big / 2 Boss) at ``ante``."""
    chips = _blind_base_chips(int(ante), int(blind_idx), cfg.stake_scaling) * cfg.ante_scaling
    return math.log10(max(chips, 1.0))


def prior_curve(ante: int, *, cfg: RaceConfig = DEFAULT, margin: Optional[float] = None) -> Curve:
    """The no-evidence curve: a build that clears the Boss target with ``margin`` log10 to
    spare, extended at ``cfg.default_slope``.  Tabulated through ante 8 (the vanilla table),
    linear afterwards — the endless formula is NOT a build's growth law."""
    m = cfg.prior_margin if margin is None else margin
    a0 = max(1, int(ante))
    antes = list(range(a0, max(a0, 8) + 1))
    mu = [log10_target(a, 2, cfg=cfg) + m for a in antes]
    return Curve(a0, tuple(mu), tuple(cfg.sigma_prior for _ in antes), cfg.default_slope, n_obs=0)


def curve_from_history(pvp_log, player: int, ante: int, *, cfg: RaceConfig = DEFAULT,
                       prior: Optional[Curve] = None) -> Curve:
    """Fit a player's curve from their observed Nemesis scores.

    ``pvp_log`` is ``MLBMatch.pvp_log``: ``(ante, loser | None, score0, score1)`` per resolved
    Nemesis.  The last ``cfg.fit_window`` entries are used; a score of 0 (deck-out / no hand
    played) is dropped when any other point exists.  >= 2 points with distinct antes -> OLS
    line in (ante, log10 score), slope clipped to ``[slope_min, slope_max]``; 1 point -> the
    default slope through it; 0 points -> ``prior_curve(ante)``.  Sigma is the residual sd
    shrunk toward ``cfg.sigma_prior`` with ``cfg.sigma_prior_weight`` pseudo-observations and
    floored at ``cfg.sigma_floor``.  The returned curve starts at ``ante`` (the next Nemesis
    the caller cares about).
    """
    pts = []
    for entry in list(pvp_log)[-cfg.fit_window:]:
        a, _loser, s0, s1 = entry[0], entry[1], entry[2], entry[3]
        s = s0 if player == 0 else s1
        pts.append((int(a), float(s)))
    nonzero = [(a, s) for a, s in pts if s > 0]
    if nonzero:
        pts = nonzero
    if not pts:
        return prior if prior is not None else prior_curve(ante, cfg=cfg)
    xs = [a for a, _ in pts]
    ys = [math.log10(max(s, 1.0)) for _, s in pts]
    n = len(pts)
    if n >= 2 and len(set(xs)) >= 2:
        mx = sum(xs) / n
        my = sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        slope = sxy / sxx
        slope = min(max(slope, cfg.slope_min), cfg.slope_max)
        intercept = my - slope * mx
        resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
        dof = max(n - 2, 0)
        ss = sum(r * r for r in resid)
        var = (ss + cfg.sigma_prior_weight * cfg.sigma_prior ** 2) / (dof + cfg.sigma_prior_weight)
        sigma = max(math.sqrt(var), cfg.sigma_floor)
        mu_at = intercept + slope * int(ante)
    else:
        slope = cfg.default_slope
        a_last, y_last = xs[-1], ys[-1]
        mu_at = y_last + slope * (int(ante) - a_last)
        sigma = max(cfg.sigma_prior, cfg.sigma_floor)
    return Curve(int(ante), (mu_at,), (sigma,), slope, n_obs=n)


# ── elementary probabilities ─────────────────────────────────────────────────────

def _phi(z: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def p_fail_blind(curve: Curve, ante: int, blind_idx: int, *, cfg: RaceConfig = DEFAULT) -> float:
    """P(score < target) for a regular blind, clipped to [floor, ceiling]."""
    mu, sigma = curve.at(ante)
    t = log10_target(ante, blind_idx, cfg=cfg)
    if sigma <= 0:
        p = 1.0 if mu < t else 0.0
    else:
        p = _phi((t - mu) / sigma)
    return min(max(p, cfg.blind_fail_floor), cfg.blind_fail_ceiling)


def p_lose_nemesis(my_curve: Curve, their_curve: Curve, ante: int, *,
                   cfg: RaceConfig = DEFAULT) -> tuple:
    """(P(I lose a life), P(they lose a life), P(tie)) at the Nemesis of ``ante``."""
    mu_m, s_m = my_curve.at(ante)
    mu_t, s_t = their_curve.at(ante)
    s = math.sqrt(s_m * s_m + s_t * s_t)
    if s <= 0:
        p_me = 1.0 if mu_m < mu_t else (0.5 if mu_m == mu_t else 0.0)
    else:
        p_me = _phi((mu_t - mu_m) / s)
    tie = min(max(cfg.p_tie, 0.0), 1.0)
    return p_me * (1 - tie), (1 - p_me) * (1 - tie), tie


def closed_form_race(p_me: float, p_them: float, my_lives: int, their_lives: int) -> float:
    """P(they lose ``their_lives`` rounds before I lose ``my_lives``) when every round costs
    exactly one side a life (ties renormalised away): the negative-binomial race
    ``sum_{k<my} C(their-1+k, k) q^their p^k`` with ``p = P(I lose a round)``."""
    if their_lives <= 0 and my_lives <= 0:
        return 0.5
    if their_lives <= 0:
        return 1.0
    if my_lives <= 0:
        return 0.0
    tot = p_me + p_them
    if tot <= 0:
        return 0.5
    p, q = p_me / tot, p_them / tot
    if q <= 0:
        return 0.0
    if p <= 0:
        return 1.0
    return sum(math.comb(their_lives - 1 + k, k) * (q ** their_lives) * (p ** k)
               for k in range(my_lives))


# ── the chain ─────────────────────────────────────────────────────────────────────

def _terminal(my_l: int, their_l: int) -> Optional[float]:
    if my_l <= 0 and their_l <= 0:
        return 0.5
    if their_l <= 0:
        return 1.0
    if my_l <= 0:
        return 0.0
    return None


def p_win(my_curve: CurveLike, their_curve: CurveLike, my_lives: int, their_lives: int,
          ante: int, *, cfg: RaceConfig = DEFAULT, blinds_done: int = 0) -> float:
    """P(I win the lives race) from the start of ``ante`` (``blinds_done`` regular blinds of
    that ante already behind BOTH players, 0..cfg.regular_blinds; the Nemesis of ``ante`` is
    still to be played).  See the module docstring for the model."""
    mine = as_curve(my_curve, cfg=cfg)
    theirs = as_curve(their_curve, cfg=cfg)
    my_lives = int(my_lives)
    their_lives = int(their_lives)
    t = _terminal(my_lives, their_lives)
    if t is not None:
        return t
    ante = int(ante)
    horizon = ante + cfg.horizon_antes

    @lru_cache(maxsize=None)
    def V(a: int, phase: int, ml: int, tl: int) -> float:
        t = _terminal(ml, tl)
        if t is not None:
            return t
        if a >= horizon:
            pm, pt, _ = p_lose_nemesis(mine, theirs, a, cfg=cfg)
            return closed_form_race(pm, pt, ml, tl)
        if phase < cfg.regular_blinds or a < cfg.pvp_start_round:
            # a regular blind (Small / Big, or the pre-PvP Boss), both players independently
            if phase < cfg.regular_blinds:
                idx, nxt = phase, (a, phase + 1)
            else:
                idx, nxt = 2, (a + 1, 0)
            fm = p_fail_blind(mine, a, idx, cfg=cfg)
            ft = p_fail_blind(theirs, a, idx, cfg=cfg)
            return ((1 - fm) * (1 - ft) * V(nxt[0], nxt[1], ml, tl)
                    + fm * (1 - ft) * V(nxt[0], nxt[1], ml - 1, tl)
                    + (1 - fm) * ft * V(nxt[0], nxt[1], ml, tl - 1)
                    + fm * ft * V(nxt[0], nxt[1], ml - 1, tl - 1))
        # the Nemesis
        pm, pt, tie = p_lose_nemesis(mine, theirs, a, cfg=cfg)
        out = pm * V(a + 1, 0, ml - 1, tl) + pt * V(a + 1, 0, ml, tl - 1)
        if tie > 0:
            out += tie * V(a + 1, 0, ml, tl)
        return out

    phase0 = min(max(int(blinds_done), 0), cfg.regular_blinds)
    return float(V(ante, phase0, my_lives, their_lives))


def race_table(my_curve: CurveLike, their_curve: CurveLike, my_lives: int, their_lives: int,
               ante: int, *, cfg: RaceConfig = DEFAULT, n_antes: int = 6) -> list:
    """Per-ante breakdown for the advisor: both curves, blind failure probabilities and the
    Nemesis loss probabilities for the next ``n_antes`` antes, plus ``p_win`` from each."""
    mine = as_curve(my_curve, cfg=cfg)
    theirs = as_curve(their_curve, cfg=cfg)
    rows = []
    for a in range(int(ante), int(ante) + n_antes):
        pm, pt, tie = p_lose_nemesis(mine, theirs, a, cfg=cfg)
        rows.append({
            "ante": a,
            "my_mu": mine.at(a)[0], "my_sigma": mine.at(a)[1],
            "their_mu": theirs.at(a)[0], "their_sigma": theirs.at(a)[1],
            "target_small": log10_target(a, 0, cfg=cfg), "target_big": log10_target(a, 1, cfg=cfg),
            "my_fail_small": p_fail_blind(mine, a, 0, cfg=cfg),
            "my_fail_big": p_fail_blind(mine, a, 1, cfg=cfg),
            "their_fail_small": p_fail_blind(theirs, a, 0, cfg=cfg),
            "their_fail_big": p_fail_blind(theirs, a, 1, cfg=cfg),
            "nemesis": a >= cfg.pvp_start_round,
            "p_i_lose_nemesis": pm, "p_they_lose_nemesis": pt, "p_tie": tie,
            "p_win_from_here": p_win(mine, theirs, my_lives, their_lives, a, cfg=cfg),
        })
    return rows
