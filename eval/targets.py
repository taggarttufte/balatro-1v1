"""
mp/eval/targets.py -- per-ante EXTERNAL Nemesis targets, ENGINE-ONLY dependencies (no torch,
no mp.eval's heavier bootstrap chain -- mlb_match_demo / oracle.parity_check / rng.generate --
so mp/agent can import this module directly without pulling any of that in).

Why this exists (Phase 4 brief S1 "W4"; CAMPAIGN_LOG.md's 2026-08-22 07:35 overnight
readout): a solo MLB agent whose Nemesis is free (`pvp_solo=True`, no `set_pvp_info` ever
called) learned to skip 15/16 blinds and coast -- the value head collapsed (z sd 0.07) not
because training failed but because the policy reliably attained the objective's max. Phase
4 decision 4: "the Nemesis must cost something." Use:

    game.set_pvp_info(target_fn(game, big_blind), 0)

right when a Nemesis's SELECTING_HAND begins (mirroring what `eval/common.py::play_sp_mlb`
already does with `own_big_blind_target` / `external_vanilla_big_blind_target`) so a solo
agent that skips both regular blinds and builds nothing still risks a life at the Nemesis.
Every target function below follows that SAME `target_fn(game, big_blind) -> int` call
signature -- `game` is a live `balatro_sim.game.BalatroGame`, `big_blind` is the
`{ante: chips_scored}` map `play_sp_mlb` (or an equivalent training-loop driver) fills in as
each ante's own Big Blind concludes, strictly before that ante's Nemesis is reached. This
means every target here drops into `play_sp_mlb(target_fn=...)` and
`play_arm_to_horizons(target_fn=...)` (`eval/common.py`) unmodified, and is exactly what
W2's `train_mlb.py --objective external` is expected to import from here.

`vanilla_boss_target` / `vanilla_boss_target_fn` and `scaled_own_big_blind` are pure Python
plus three tiny reads of `mp/engine/balatro_sim` (`constants.blind_base_chips`,
`decks.deck_spec`, `stakes.stake_spec`) -- no other engine module, no numpy, no torch.
`table_target` additionally reads a JSONL file (stdlib `json` only).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Callable, Optional

# ============================================================================ bootstrap
#
# mp/engine/** is FROZEN and only ever read here. A `balatro_sim` package ALSO lives at the
# repo root (the BRL project's own, unrelated engine) -- it must never win an import race
# against the mp/engine fork. `mp/eval/common.py` (and `mp/tournament/bootstrap.py`) guard
# this the same way, through `oracle.engine_parity.import_engine()`; that entry point is NOT
# used here on purpose -- it is a module-scope import of `mp.oracle.parity_check` +
# `mp.rng.generate` / `mp.rng.pools`, which is exactly the "mp.eval heavy imports" this
# module is supposed to avoid so `mp/agent` can import `targets.py` cheaply. The guard logic
# below is the same check, reimplemented with zero extra imports.

_HERE = Path(__file__).resolve().parent            # mp/eval
_MP_ROOT = _HERE.parent                             # mp/
_ENGINE_ROOT = _MP_ROOT / "engine"


def _import_balatro_sim():
    """Fork-guarded import of mp/engine's balatro_sim package. Puts mp/engine first on
    sys.path, then either confirms an already-imported `balatro_sim` is the fork (a second
    call in the same process -- e.g. after `eval/common.py` already ran its own guarded
    import -- is a cheap no-op) or imports it fresh and checks its `__file__`."""
    engine_root = str(_ENGINE_ROOT)
    if engine_root in sys.path:
        sys.path.remove(engine_root)
    sys.path.insert(0, engine_root)
    want = os.path.normcase(str(_ENGINE_ROOT / "balatro_sim" / "__init__.py"))
    already = sys.modules.get("balatro_sim")
    if already is not None:
        got = os.path.normcase(os.path.abspath(getattr(already, "__file__", "") or ""))
        if got != want:
            raise RuntimeError(
                "mp.eval.targets: a different balatro_sim is already imported in this "
                f"process:\n  got:      {got}\n  expected: {want}\n"
                "Import mp.eval.targets before any other balatro_sim package, or run in a "
                "fresh interpreter (python -m pytest mp/eval/tests / python -m mp.eval....)."
            )
        return already
    import balatro_sim  # noqa: WPS433  (runtime import is the point)
    got = os.path.normcase(os.path.abspath(balatro_sim.__file__))
    if got != want:
        raise RuntimeError(f"mp.eval.targets imported the wrong balatro_sim: {got} (expected {want})")
    return balatro_sim


_import_balatro_sim()

from balatro_sim.constants import blind_base_chips  # noqa: E402
from balatro_sim.decks import deck_spec              # noqa: E402
from balatro_sim.stakes import stake_spec            # noqa: E402

__all__ = [
    "vanilla_boss_target", "vanilla_boss_target_fn",
    "scaled_own_big_blind", "table_target", "get_target", "TARGETS",
]

_BOSS_BLIND_IDX = 2   # BlindInfo.kind index: 0 Small / 1 Big / 2 Boss (constants.BLIND_MULT order)


# ============================================================================ vanilla_boss_target

def vanilla_boss_target(ante: int, deck_key: str = "b_red", stake: "int | str" = 1) -> int:
    """The chip requirement a VANILLA (non-Nemesis) Boss blind would have had at `ante`, for
    `deck_key` at `stake` -- i.e. exactly `game.py::_prepare_next_blind`'s
    ``int(blind_base_chips(ante, 2, blind_scaling) * ante_scaling)`` (game.py:642), before
    any boss-SPECIFIC multiplier.

    Why no boss-specific multiplier: `BOSS_CHIP_MULT` (game.py:193-197) depends on WHICH
    boss a live run's 'boss' RNG stream actually drew, and this function takes no seed --
    it cannot reproduce that draw. Two of the three non-1.0x bosses (`bl_wall` 2x,
    `bl_final_vessel` 3x) are `MLB_BANNED_BLINDS` anyway (constants.py) and never drawn
    under ``ruleset="mlb"``; only `bl_needle` (0.5x, i.e. an EASIER-than-typical boss) is a
    real gap, silently making this target a little harder than that specific boss would
    have been on the ~1/23 regular-boss draws where it comes up (its ante-window is
    ante>=5 per BOSS_MIN_ANTE/MAX_ANTE -- game_keys.py). Documented, not fixed: fixing it
    would require this function to take a live game/boss_key, which the Nemesis's own
    `boss_key` never carries (it is always `MLB_NEMESIS_KEY`), i.e. there is nothing to
    read it FROM inside a real MLB run anyway.

    `deck_key`'s `ante_scaling` (`decks.deck_spec`; 2 for `b_plasma`, 1 for everything else)
    and `stake`'s `blind_scaling` (`stakes.stake_spec`; 2 at Green+, 3 at Purple+, else 1)
    both apply, composed in the same order the engine itself uses
    (`blind_base_chips(...) * ante_scaling`, game.py:642). `ante > 8` falls through to
    `blind_base_chips`'s own endless-formula branch (`constants.get_blind_amount`)
    automatically -- nothing special-cased here.
    """
    scaling = stake_spec(stake).scaling
    ante_scale = deck_spec(deck_key).ante_scaling
    return int(blind_base_chips(int(ante), _BOSS_BLIND_IDX, scaling) * ante_scale)


def vanilla_boss_target_fn(deck_key: Optional[str] = None, stake: "Optional[int | str]" = None) -> Callable:
    """Adapter from `vanilla_boss_target(ante, deck_key, stake)` to the
    `target_fn(game, big_blind) -> int` signature `eval/common.py`'s drivers and W2's
    training-loop Nemesis hook use: reads `ante` / `deck_key` / `stake` off the live `game`
    unless overridden here. This is the function to register as the Nemesis's
    `chips_target` per the module docstring: ``game.set_pvp_info(vanilla_boss_target_fn()
    (game, big_blind), 0)``. `big_blind` is accepted (and ignored) purely for call-signature
    compatibility with `scaled_own_big_blind` / `common.own_big_blind_target`."""
    def target(game, big_blind: Optional[dict] = None) -> int:
        dk = deck_key if deck_key is not None else game.deck_key
        st = stake if stake is not None else game.stake
        return vanilla_boss_target(game.ante, dk, st)
    target.__name__ = f"vanilla_boss_target_fn(deck_key={deck_key!r}, stake={stake!r})"
    return target


# ============================================================================ scaled_own_big_blind

def scaled_own_big_blind(k: float = 1.0) -> Callable:
    """W4-Phase-3's "mirror Nemesis" target (`eval/common.py::own_big_blind_target`),
    reimplemented here with zero extra imports so it is available from this engine-only
    module too: `k` times the agent's OWN Big-Blind chip score that same ante, read out of
    `big_blind[ante]` (filled in by `play_sp_mlb` / an equivalent training-loop driver
    BEFORE that ante's Nemesis is reached). Calibration-free -- no historical corpus, no
    second checkpoint -- and genuinely ~50/50 by construction at `k=1`: a policy that
    improves ante-to-ante faces a target that improves right along with it. `k=1.0` and this
    module's `scaled_own_big_blind` are numerically IDENTICAL to `common.own_big_blind_target`
    for the same `(game, big_blind)` input -- this is a deliberate duplication, not a
    behavioural difference, so `mp/agent` never has to import `mp/eval/common.py`."""
    def target(game, big_blind: Optional[dict]) -> int:
        return int(round(k * (big_blind or {}).get(game.ante, 0)))
    target.__name__ = f"scaled_own_big_blind(k={k})"
    return target


# ============================================================================ table_target

def _load_ante_quantiles(path) -> dict:
    """``{ante: {quantile_str: value}}`` from a tournament run's ``summary.jsonl``
    (``mp/tournament/runs/<run>/summary.jsonl``, `tournament.matrix.write_run`'s format:
    one JSON object per line, keys include ``ante`` and ``quantiles``) -- or a directory
    containing one (``mp/results/<run>/`` works the same way if it holds a
    ``summary.jsonl`` written by the same format)."""
    p = Path(path)
    if p.is_dir():
        p = p / "summary.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"table_target: no such file {p}")
    out: dict = {}
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[int(row["ante"])] = row["quantiles"]
    if not out:
        raise ValueError(f"table_target: {p} has no ante rows")
    return out


def table_target(path, quantile: float = 0.5, fallback: str = "nearest_below") -> Callable:
    """A target read off a tournament run's per-ante score DISTRIBUTION
    (``mp/tournament/runs/*/summary.jsonl`` / ``mp/results/*/summary.jsonl``, written by
    ``tournament.matrix.write_run`` -- ``score_distribution``'s registered quantiles are
    ``0.0/0.1/0.25/0.5/0.75/0.9/1.0``). Median (``quantile=0.5``) by default.

    ``fallback`` decides what happens for an ante missing from the table (a tournament run
    that never reached that ante -- everyone eliminated first, or it was never run that
    far): ``"nearest_below"`` (default) uses the largest tabulated ante <= the requested
    one (a flat extrapolation -- reasonable since blind/Nemesis targets only grow with
    ante, so this is a conservative UNDER-estimate past the table's range, never an
    over-estimate); ``"error"`` raises ``KeyError`` instead."""
    if fallback not in ("nearest_below", "error"):
        raise ValueError(f"table_target: fallback must be 'nearest_below'/'error', got {fallback!r}")
    table = _load_ante_quantiles(path)
    qkey = str(quantile)
    available_antes = sorted(table)
    for a in available_antes:
        if qkey not in table[a]:
            raise KeyError(f"table_target: quantile {qkey!r} not in {path} ante {a} "
                            f"(have: {sorted(table[a])})")

    def target(game, big_blind: Optional[dict] = None) -> int:
        ante = game.ante
        if ante in table:
            return int(round(table[ante][qkey]))
        if fallback == "error":
            raise KeyError(f"table_target: ante {ante} not in {path} (have: {available_antes})")
        below = [a for a in available_antes if a <= ante]
        if not below:
            raise KeyError(f"table_target: ante {ante} is below every tabulated ante in "
                            f"{path} (have: {available_antes})")
        return int(round(table[max(below)][qkey]))
    target.__name__ = f"table_target({path!r}, quantile={quantile})"
    return target


# ============================================================================ registry

TARGETS: dict = {
    "vanilla_boss": vanilla_boss_target_fn,
    "own_big_blind": scaled_own_big_blind,
    "table": table_target,
}


def get_target(name: str, **kw) -> Callable:
    """Tiny registry -- ``get_target("vanilla_boss")``, ``get_target("own_big_blind",
    k=1.5)``, ``get_target("table", path="mp/tournament/runs/foo", quantile=0.9)``. Every
    returned callable follows the shared ``target_fn(game, big_blind=None) -> int``
    signature (see module docstring)."""
    if name not in TARGETS:
        raise ValueError(f"unknown target {name!r} (have: {sorted(TARGETS)})")
    return TARGETS[name](**kw)
