"""
mp/eval/common.py -- shared bootstrap, player-spec parsing, solo/1v1 drivers and statistics
helpers for the Phase 3 W4 eval harness (``eval_harness.py``) and rho-decay harness
(``rho_decay.py``).  See mp/eval/EVAL_NOTES.md.

Imports the mp/engine fork (never a different ``balatro_sim``) through the SAME pattern as
``mp/tests/test_mlb_match_gate.py``: put ``mp/`` and ``mp/scripts`` on ``sys.path``, then
``oracle.engine_parity.import_engine()`` puts ``mp/engine`` first and refuses a shadowing
package.  ``mlb_match_demo`` (``ScriptedPlayer`` / ``make_policy`` / ``greedy_hand`` /
``weakest_play`` / ``shelf_indices``) is IMPORTED, not copied, per the Phase 3 brief.

``mp/engine/**`` and ``mp/rng/**`` are frozen and never edited by this package -- every
driver here works entirely through the engine's existing public entry points
(``BalatroGame.step`` / ``set_pvp_info`` / ``lose_life`` / ``legal_actions`` / ``clone``,
``MLBMatch.step`` / ``play_out``).
"""
from __future__ import annotations

import math
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

# ============================================================================ bootstrap

_HERE = Path(__file__).resolve().parent            # mp/eval
MP_ROOT = _HERE.parent                              # mp/
for _p in (str(MP_ROOT), str(MP_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from oracle.engine_parity import import_engine, item_from_engine  # noqa: E402

GM = import_engine()                                # the mp/engine fork (refuses the BRL package)

import mlb_match_demo as D                          # noqa: E402  (ScriptedPlayer, make_policy, ...)
from balatro_sim.game import BalatroGame, State      # noqa: E402
from balatro_sim.mlb_match import MLBMatch           # noqa: E402
from balatro_sim.constants import blind_base_chips, MLB_STARTING_LIVES, MLB_PVP_START_ROUND  # noqa: E402

ScriptedPlayer = D.ScriptedPlayer
make_policy = D.make_policy
greedy_hand = D.greedy_hand
weakest_play = D.weakest_play
shelf_indices = D.shelf_indices

__all__ = [
    "GM", "BalatroGame", "State", "MLBMatch", "ScriptedPlayer", "make_policy", "item_from_engine",
    "MP_ROOT", "GROUND_TRUTH_DIR", "DEFAULT_SEEDS", "RESULTS_DIR",
    "Player", "parse_player_spec", "make_player_policy", "adapt_player",
    "SoloShim", "solo_policy_step", "play_sp_vanilla", "play_sp_mlb", "play_1v1",
    "own_big_blind_target", "external_vanilla_big_blind_target",
    "bootstrap_ci", "paired_bootstrap_ci", "pearson_r", "spearman_r", "bootstrap_corr_ci",
    "unpaired_control_variance", "sample_size_per_arm",
]

GROUND_TRUTH_DIR = MP_ROOT / "oracle" / "ground_truth"
DEFAULT_SEEDS: list = sorted(p.stem for p in GROUND_TRUTH_DIR.glob("*.json"))
RESULTS_DIR = MP_ROOT / "results"

_BLIND_ORDER = {"Small": 0, "Big": 1, "Boss": 2}


# ============================================================================ Player protocol + spec parsing

class Player:
    """Interface a trained/checkpointed player must satisfy so this harness can evaluate it
    once W1 (``mp/agent``) lands a checkpoint loader.  NOT implemented here.

        class MyCheckpointPlayer(Player):
            def act(self, game: "BalatroGame") -> dict:
                ...   # one of game.legal_actions()

    Wrap an instance with ``adapt_player`` to get the ``(match_or_shim, player_idx,
    legal_actions) -> action`` signature ``mlb_match_demo.make_policy`` / ``MLBMatch.play_out``
    / the drivers below all use.
    """

    def act(self, game) -> dict:
        raise NotImplementedError


def adapt_player(player: "Player") -> Callable:
    def pol(m, p, acts):
        return player.act(m.games[p])
    return pol


_BOOL_TRUE = {"1", "true", "t", "yes", "y", "on"}
_SPEC_ALIASES = {
    "reroll": "rerolls_per_visit", "rerolls": "rerolls_per_visit",
    "buy": "buy_slot0",
    "pack": "open_pack_slot",
    "voucher": "buy_voucher",
    "weak_from": "weak_from_ante",
}
_SCRIPTED_FIELDS = set(ScriptedPlayer.__dataclass_fields__)


def _coerce(field_name: str, raw: str):
    default = getattr(ScriptedPlayer, field_name)
    if isinstance(default, bool):
        return raw.strip().lower() in _BOOL_TRUE
    if field_name == "open_pack_slot":
        return None if raw.strip().lower() in ("none", "") else int(raw)
    if isinstance(default, int):
        return int(raw)
    return raw


def parse_scripted_spec(body: str, name: Optional[str] = None) -> "ScriptedPlayer":
    """``"hand=greedy,reroll=1,buy=0"`` -> ``ScriptedPlayer``.  Unknown keys raise ValueError
    so a typo in a spec string fails loudly instead of silently no-op'ing."""
    kwargs: dict = {}
    if body:
        for part in body.split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise ValueError(f"bad scripted player spec fragment {part!r} (want key=value)")
            k, v = part.split("=", 1)
            k = _SPEC_ALIASES.get(k.strip(), k.strip())
            if k not in _SCRIPTED_FIELDS:
                raise ValueError(f"unknown ScriptedPlayer field {k!r} (spec: {body!r})")
            kwargs[k] = _coerce(k, v)
    if name is not None:
        kwargs.setdefault("name", name)
    return ScriptedPlayer(**kwargs)


def parse_player_spec(spec: str):
    """``"scripted:<fields>"`` -> ``(label, ScriptedPlayer)``.  ``"checkpoint:<path>"`` raises
    ``NotImplementedError`` documenting the interface W1's checkpoint loader must provide
    (see ``Player`` above) -- ``mp/agent`` is out of scope for this package."""
    kind, _, body = spec.partition(":")
    kind = kind.strip().lower()
    if kind == "scripted":
        return spec, parse_scripted_spec(body, name=spec)
    if kind == "checkpoint":
        raise NotImplementedError(
            "checkpoint players are not implemented in mp/eval -- W1 (mp/agent) owns the "
            "checkpoint loader. Expected interface: an object implementing "
            "`mp.eval.common.Player` (a `.act(game) -> action` method, `game` = the live "
            "`balatro_sim.game.BalatroGame`); once W1 provides `load_checkpoint(path) -> "
            "Player`, wire it into `parse_player_spec` here. Meanwhile wrap any such object "
            "with `adapt_player(player)` to get a policy usable by every driver in this module "
            f"and by `MLBMatch.play_out`. Got spec: {spec!r}")
    raise ValueError(f"unknown player spec kind {kind!r} in {spec!r} (want 'scripted:' or 'checkpoint:')")


def make_player_policy(spec: str):
    """``"scripted:..."`` / ``"checkpoint:..."`` -> ``(label, policy_fn)``."""
    label, obj = parse_player_spec(spec)
    if isinstance(obj, ScriptedPlayer):
        return label, make_policy(obj)
    return label, adapt_player(obj)


# ============================================================================ solo driving (no MLBMatch)

@dataclass
class SoloShim:
    """Lets a single ``BalatroGame`` be driven by a policy written against the
    ``(match, player_idx, legal_actions)`` signature (``mlb_match_demo.make_policy`` /
    ``adapt_player``): ``m.games[p]`` is all such a policy ever reads off ``m``."""
    games: list


def solo_policy_step(game, policy_fn, player_idx: int = 0) -> dict:
    return policy_fn(SoloShim(games=[game]), player_idx, game.legal_actions())


# ---------------------------------------------------------------------- SP vanilla

def play_sp_vanilla(seed: str, policy_fn, deck_key: str = "b_red", stake=1, max_steps: int = 200_000) -> dict:
    """Play ONE ``BalatroGame(ruleset='vanilla')`` to ``GAME_OVER`` (win at ante 9 or a lost
    blind).  Returns ``won``, ``furthest_ante``, ``furthest_blind``, ``final_ante``,
    ``final_money``, ``steps``."""
    game = BalatroGame(seed=seed, deck_key=deck_key, stake=stake, ruleset="vanilla")
    furthest = (game.ante, _BLIND_ORDER.get(game.current_blind.kind, 0))
    steps = 0
    while game.state != State.GAME_OVER and steps < max_steps:
        key = (game.ante, _BLIND_ORDER.get(game.current_blind.kind, 0))
        if key > furthest:
            furthest = key
        a = solo_policy_step(game, policy_fn)
        game.step(a)
        steps += 1
    key = (game.ante, _BLIND_ORDER.get(game.current_blind.kind, 0))
    if key > furthest:
        furthest = key
    kind_by_idx = {v: k for k, v in _BLIND_ORDER.items()}
    return {
        "seed": game.seed_str,
        "won": bool(game._obs().won),
        "furthest_ante": furthest[0],
        "furthest_blind": kind_by_idx[furthest[1]],
        "final_ante": game.ante,
        "final_money": game.dollars,
        "steps": steps,
    }


# ---------------------------------------------------------------------- SP-MLB solo (phantom Nemesis)

def own_big_blind_target(k: float = 1.0):
    """Default SP-MLB-solo target function: ``k`` times the agent's OWN Big-Blind chip score
    at that same ante, captured live as the agent plays it (``big_blind[ante]``, filled in
    automatically by ``play_sp_mlb`` / ``play_arm_to_horizons`` before each Nemesis).  A
    reproducible, calibration-free "mirror Nemesis": with k=1 the fixed opponent is exactly
    as good as the agent was on the immediately preceding blind, giving a genuinely ~50/50
    target with no historical corpus and no coupling to a second live player.  See
    EVAL_NOTES.md "the default SP-MLB target"."""
    def target(game, big_blind: dict) -> int:
        return int(round(k * big_blind.get(game.ante, 0)))
    target.__name__ = f"own_big_blind_target(k={k})"
    return target


def external_vanilla_big_blind_target(game, big_blind: Optional[dict] = None) -> int:
    """A target that depends on NEITHER paired arm in a rho_decay comparison: the vanilla
    Big-Blind chip requirement for this ante/deck/stake (``blind_base_chips(ante, 1,
    blind_scaling) * ante_scaling``) -- purely a function of (ante, deck, stake), not of
    either arm's realized play.  Default target for ``rho_decay.py``."""
    return int(blind_base_chips(game.ante, 1, game.blind_scaling) * game.ante_scaling)


def play_sp_mlb(seed: str, policy_fn, deck_key: str = "b_red", stake=1, lives: int = MLB_STARTING_LIVES,
                max_antes: int = 8, target_fn=None, max_steps: int = 200_000) -> dict:
    """Drive ONE ``BalatroGame(ruleset='mlb')`` solo (no ``MLBMatch``, no live opponent),
    playing every Nemesis to hand-exhaustion against ``target_fn(game, big_blind) -> int``
    (default ``own_big_blind_target()``).  ``big_blind[ante]`` is filled in automatically the
    moment that ante's Big Blind is cash-out'd -- BEFORE that ante's Nemesis is reached -- so
    a target_fn can read it.  Life loss / game-over are decided the same way ``MLBMatch``
    decides them for a real opponent (``set_pvp_info`` / compare / ``lose_life``), just
    against the external target instead of a second live game.

    Stops at ``GAME_OVER`` (0 lives) or once ``ante > max_antes`` (default: play through the
    ante-8 Nemesis and its cash-out, then stop -- MLB is endless, so this is a harness cutoff,
    not a win condition)."""
    if target_fn is None:
        target_fn = own_big_blind_target()
    game = BalatroGame(seed=seed, deck_key=deck_key, stake=stake, ruleset="mlb")
    game.lives = lives
    big_blind: dict = {}
    nemesis_log: list = []
    money_curve: list = []
    target_set_for_ante: Optional[int] = None
    furthest_ante = game.ante
    steps = 0
    while game.state != State.GAME_OVER and game.ante <= max_antes and steps < max_steps:
        pre_state = game.state
        was_pvp = game.current_blind.is_pvp
        ante_now, kind_now = game.ante, game.current_blind.kind
        if pre_state == State.SELECTING_HAND and was_pvp and target_set_for_ante != ante_now:
            game.set_pvp_info(int(target_fn(game, big_blind)), 0)
            target_set_for_ante = ante_now
        a = solo_policy_step(game, policy_fn)
        game.step(a)
        steps += 1
        cur = game.state
        furthest_ante = max(furthest_ante, game.ante)
        if pre_state == State.SELECTING_HAND and not was_pvp and kind_now == "Big" and cur == State.ROUND_EVAL:
            big_blind[ante_now] = game.chips_scored
        # Nemesis just concluded WITHOUT the engine itself deciding a winner (that only
        # happens against a real second game, e.g. deck-out with 0 hands played, which calls
        # lose_life() unconditionally inside the engine): compare to the external target and
        # decide the life ourselves, exactly what MLBMatch._resolve_pvp does for a real
        # opponent (mlb_match.py) -- but against `target_fn` instead of a live game.
        if pre_state in (State.SELECTING_HAND, State.PVP_WAIT) and was_pvp and cur == State.ROUND_EVAL:
            target = game.current_blind.chips_target
            score = game.chips_scored
            lost = score < target
            if lost:
                game.lose_life()
                if game.lives <= 0:
                    game.state = State.GAME_OVER
            nemesis_log.append({"ante": ante_now, "score": score, "target": target,
                                 "life_lost": lost, "margin": score - target})
        if pre_state == State.ROUND_EVAL and cur == State.SHOP:
            money_curve.append({"ante": game.ante, "dollars": game.dollars})
    # `lives_lost` counts EVERY source of life loss under MLB, not only Nemesis losses: a
    # failed REGULAR blind also costs a life and proceeds (MLB_NOTES.md 1.3d), and a deck-out
    # with 0 hands played costs one unconditionally (1.3e) -- both call the engine's own
    # `lose_life()` directly, so the only correct count is the observed delta.
    return {
        "seed": game.seed_str,
        "furthest_ante": furthest_ante,
        "lives_lost": lives - game.lives,
        "final_lives": game.lives,
        "final_money": game.dollars,
        "money_curve": money_curve,
        "nemesis_log": nemesis_log,
        "steps": steps,
        # game.py:1551/1579/1591 (bl_hook / bl_eye / bl_mouth boss-ability rejection on hand
        # exhaustion) sets GAME_OVER unconditionally, bypassing the mlb-aware _mlb_fail_round
        # a few lines below -- see EVAL_NOTES.md "needs engine change".  This flags exactly
        # that: the run ended at GAME_OVER with lives still remaining (not a real 0-lives loss).
        "ended_early_engine_gap": bool(game.state == State.GAME_OVER and game.lives > 0),
    }


# ---------------------------------------------------------------------- 1v1 (real MLBMatch)

def play_1v1(seed: str, policy_a, policy_b, deck_key: str = "b_red", stake=1,
             lives: int = MLB_STARTING_LIVES, max_steps: int = 200_000) -> dict:
    """Full MLB match through ``MLBMatch`` (canonical alternation, Nemesis-to-exhaustion is
    already the engine's server rule -- see mp/engine/MLB_NOTES.md)."""
    m = MLBMatch(seed=seed, deck_key=deck_key, stake=stake, lives=lives)
    m.play_out([policy_a, policy_b], max_steps=max_steps)
    g0, g1 = m.games
    log_margins = [math.log1p(s0) - math.log1p(s1) for _, _, s0, s1 in m.pvp_log]
    return {
        "seed": m.seed_str,
        "winner": m.winner,
        "lives": [g0.lives, g1.lives],
        "lives_margin": g0.lives - g1.lives,
        "pvp_log": list(m.pvp_log),
        "log_score_margins": log_margins,
        "mean_log_score_margin": (statistics.fmean(log_margins) if log_margins else 0.0),
        "final_ante": [g0.ante, g1.ante],
        "final_money": [g0.dollars, g1.dollars],
        "steps": m.steps,
        "done": m.done,
    }


# ============================================================================ rho-decay: paired arms

def run_until(game, policy_fn, done_fn: Callable, max_steps: int = 20_000) -> int:
    """Step ``game`` with ``policy_fn`` (solo signature) until ``done_fn(game)`` is true.
    Returns the number of steps taken."""
    steps = 0
    while not done_fn(game) and steps < max_steps:
        a = solo_policy_step(game, policy_fn)
        game.step(a)
        steps += 1
    return steps


def at_big_blind_select(game) -> bool:
    return game.state == State.BLIND_SELECT and game.ante == 1 and game.blind_idx == 1


def play_arm_to_horizons(game, policy_fn, target_fn, horizons: Sequence[int],
                         max_steps: int = 300_000) -> dict:
    """Continue playing ``game`` (already at whatever point a perturbation left it) with
    ``policy_fn`` through every Nemesis up to ante ``1 + max(horizons)``, recording per-horizon
    outcomes: own log-score at that ante's Nemesis, dollars at the following Cash Out, and
    cumulative lives lost so far.  ``target_fn(game) -> int`` decides win/loss the same way
    ``play_sp_mlb`` does; the caller is expected to set ``game.lives`` generously (rho_decay
    uses a large number) so a loss never truncates the run before every horizon is reached."""
    horizons = list(horizons)
    last_ante = 1 + max(horizons)
    horizons_set = set(horizons)
    out = {h: {} for h in horizons}
    target_set_for_ante: Optional[int] = None
    starting_lives = game.lives
    pending_money_h: Optional[int] = None
    steps = 0
    while game.ante <= last_ante and game.state != State.GAME_OVER and steps < max_steps:
        pre_state = game.state
        was_pvp = game.current_blind.is_pvp
        ante_now = game.ante
        if pre_state == State.SELECTING_HAND and was_pvp and target_set_for_ante != ante_now:
            game.set_pvp_info(int(target_fn(game)), 0)
            target_set_for_ante = ante_now
        a = solo_policy_step(game, policy_fn)
        game.step(a)
        steps += 1
        cur = game.state
        # `cum_lives_lost` counts EVERY source of life loss under MLB (regular-blind fails
        # and deck-outs call the engine's own `lose_life()` directly, same caveat as
        # `play_sp_mlb`), not only the Nemesis-vs-`target_fn` decision below -- so it is
        # read from `game.lives` itself, not accumulated from the one call this function makes.
        if pre_state in (State.SELECTING_HAND, State.PVP_WAIT) and was_pvp and cur == State.ROUND_EVAL:
            target = game.current_blind.chips_target
            score = game.chips_scored
            lost = score < target
            if lost:
                game.lose_life()
            h = ante_now - 1
            if h in horizons_set:
                out[h].update(ante=ante_now, score=score, target=target,
                              log_score=math.log1p(score), life_lost=lost,
                              cum_lives_lost=starting_lives - game.lives)
                pending_money_h = h
        if pending_money_h is not None and pre_state == State.ROUND_EVAL and cur == State.SHOP:
            out[pending_money_h]["dollars"] = game.dollars
            pending_money_h = None
        if game.ante > last_ante:
            break
    for h in horizons:
        if "score" not in out[h]:
            out[h] = None      # horizon not reached (max_steps hit or GAME_OVER) -- shouldn't
                                 # happen with the generous `lives` rho_decay uses; caller checks
        elif "dollars" not in out[h]:
            out[h]["dollars"] = game.dollars   # last horizon coincided with the loop's end
    return out


# ============================================================================ statistics (pure python + statistics stdlib)

def bootstrap_ci(values: Sequence[float], *, n_boot: int = 2000, alpha: float = 0.05,
                  statistic: Optional[Callable] = None, seed: int = 0) -> dict:
    """Percentile bootstrap CI for ``statistic`` (default: mean) over ``values``."""
    if statistic is None:
        statistic = statistics.fmean
    vals = list(values)
    n = len(vals)
    if n == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    point = statistic(vals)
    if n < 2:
        return {"point": point, "lo": point, "hi": point, "n": n}
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        sample = [vals[rng.randrange(n)] for _ in range(n)]
        boots.append(statistic(sample))
    boots.sort()
    lo_idx = int((alpha / 2) * n_boot)
    hi_idx = int((1 - alpha / 2) * n_boot) - 1
    return {"point": point, "lo": boots[max(0, lo_idx)], "hi": boots[min(n_boot - 1, hi_idx)], "n": n}


def paired_bootstrap_ci(a: Sequence[float], b: Sequence[float], **kw) -> dict:
    """CI of ``mean(a[i] - b[i])`` over the SAME index ``i`` (paired by seed) -- common random
    numbers is the whole point of this harness's ``--compare``."""
    diffs = [x - y for x, y in zip(a, b)]
    return bootstrap_ci(diffs, **kw)


def pearson_r(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def _rank(vals: Sequence[float]) -> list:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman_r(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    return pearson_r(_rank(xs), _rank(ys))


def bootstrap_corr_ci(xs: Sequence[float], ys: Sequence[float], method: str = "pearson",
                      n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> dict:
    fn = pearson_r if method == "pearson" else spearman_r
    n = len(xs)
    point = fn(xs, ys)
    if n < 3:
        return {"point": point, "lo": point, "hi": point, "n": n}
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        idxs = [rng.randrange(n) for _ in range(n)]
        bx = [xs[i] for i in idxs]
        by = [ys[i] for i in idxs]
        r = fn(bx, by)
        if r == r:   # not NaN (a bootstrap resample with zero variance in one arm)
            boots.append(r)
    if not boots:
        return {"point": point, "lo": float("nan"), "hi": float("nan"), "n": n}
    boots.sort()
    lo_idx = int((alpha / 2) * len(boots))
    hi_idx = int((1 - alpha / 2) * len(boots)) - 1
    return {"point": point, "lo": boots[max(0, lo_idx)], "hi": boots[min(len(boots) - 1, hi_idx)], "n": n}


def unpaired_control_variance(xs: Sequence[float], ys: Sequence[float], n_perm: int = 500,
                              seed: int = 0) -> float:
    """Var(A - B) under an UNPAIRED design, estimated by randomly re-pairing the already
    collected same-seed outcomes (``ys[i]`` against ``xs[perm[i]]``) many times and averaging
    the resulting variance.  Statistically equivalent to actually running arm B on unrelated
    seeds (seeds are exchangeable draws), at zero extra simulation cost -- this is the
    "unpaired controls" the Phase 3 brief and MP_TRAINING_DESIGN_2026-08.md §1 ask for."""
    n = len(xs)
    if n < 2:
        return float("nan")
    rng = random.Random(seed)
    idxs = list(range(n))
    variances = []
    for _ in range(n_perm):
        rng.shuffle(idxs)
        diffs = [xs[idxs[i]] - ys[i] for i in range(n)]
        variances.append(statistics.pvariance(diffs))
    return statistics.fmean(variances)


_Z = {0.20: 0.8416, 0.10: 1.2816, 0.05: 1.6449, 0.025: 1.9600, 0.01: 2.3263, 0.005: 2.5758}


def sample_size_per_arm(effect_size_d: float, rho: float = 0.0, alpha: float = 0.05,
                        power: float = 0.8) -> dict:
    """Runs/arm to resolve a standardized effect size ``d`` at ``alpha``/``power``
    (MP_DECISION_ECONOMICS_2026-08.md §2's ``n ~= 2*sigma^2*(z_a/2+z_b)^2 / delta^2``, i.e.
    ``n ~= (z_a/2+z_b)^2 / d^2`` per arm for an UNPAIRED two-sample comparison with equal
    variance; the paired design divides this by ``1/(1-rho)`` -- the variance-reduction
    factor implied by a correlation ``rho`` between the two arms)."""
    z_a = _Z.get(round(alpha / 2, 3), 1.96)
    z_b = _Z.get(round(1 - power, 3), 0.8416)
    n_unpaired = 2 * (z_a + z_b) ** 2 / (effect_size_d ** 2)
    factor = 1.0 / (1.0 - rho) if rho < 1.0 else float("inf")
    n_paired = n_unpaired / factor
    return {"d": effect_size_d, "rho": rho, "n_unpaired": math.ceil(n_unpaired),
            "n_paired": math.ceil(n_paired), "variance_reduction_factor": factor}
