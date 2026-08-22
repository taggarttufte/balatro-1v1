"""
stakes.py — the 8 stakes (Phase 2 W3): catalogue + modifier table + the engine-side hook.

Ground truth: the stake table ``game.lua:253-260`` (``pools.STAKES``, keys
``stake_white`` .. ``stake_gold``, ``stake_level`` 1..8) and the modifiers ``Game:start_run``
sets with ``stake >= N`` (game.lua:2050-2057), i.e. each stake keeps every lower stake's
modifier:

    level  key            adds
    1      stake_white    — (no-op; the vanilla run)
    2      stake_red      modifiers.no_blind_reward.Small = true   (Small Blind pays $0;
                          blind.lua:84 sets blind.dollars = 0, button_callbacks.lua:172 hides it)
    3      stake_green    modifiers.scaling = 2                    (get_blind_amount table 2,
                          misc_functions.lua:933-943)
    4      stake_black    modifiers.enable_eternals_in_shop        (Eternal stickers — sticker
                          system, later phase)
    5      stake_blue     starting_params.discards -= 1
    6      stake_purple   modifiers.scaling = 3                    (table 3, :944-954)
    7      stake_orange   modifiers.enable_perishables_in_shop     (sticker system, later phase)
    8      stake_gold     modifiers.enable_rentals_in_shop         (sticker system, later phase)

The generation side (the three ``enable_*_in_shop`` flags, which gate the ``etperpoll`` /
``stickers`` / ``rental`` rolls) lives in ``generate.RunState.for_stake``; this module owns
the engine side: ``no_small_blind_reward``, ``blind_scaling`` (→ ``constants.blind_base_chips``)
and the discard penalty.  The Eternal / Perishable / Rental EFFECTS (can't sell, debuff after
5 rounds, $3 per round) are NOT implemented — the shelf carries the sticker flags into
``JokerInstance.state`` and nothing reads them (xfail in test_stakes.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from . import game_keys as _gk

if TYPE_CHECKING:  # pragma: no cover
    from .game import BalatroGame

_STAKES = _gk.pools.STAKES


@dataclass(frozen=True)
class StakeSpec:
    key: str
    name: str
    level: int                           # G.GAME.stake (1..8)
    order: int
    # cumulative modifiers at this level (game.lua:2050-2057)
    no_small_blind_reward: bool = False  # level >= 2
    scaling: int = 1                     # 2 at level >= 3, 3 at level >= 6
    eternals_in_shop: bool = False       # level >= 4
    discards: int = 0                    # -1 at level >= 5
    perishables_in_shop: bool = False    # level >= 7
    rentals_in_shop: bool = False        # level >= 8

    @property
    def needs_stickers(self) -> bool:
        return self.eternals_in_shop or self.perishables_in_shop or self.rentals_in_shop


def _spec(entry: dict) -> StakeSpec:
    n = entry["stake_level"]
    spec = StakeSpec(
        key=entry["key"], name=entry["name"], level=n, order=entry["order"],
        no_small_blind_reward=n >= 2,
        scaling=3 if n >= 6 else 2 if n >= 3 else 1,
        eternals_in_shop=n >= 4,
        discards=-1 if n >= 5 else 0,
        perishables_in_shop=n >= 7,
        rentals_in_shop=n >= 8,
    )
    # cross-check against the pools table's own cumulative column
    cum = entry["cumulative"]
    assert spec.no_small_blind_reward == bool(cum.get("no_blind_reward", {}).get("Small")), entry["key"]
    assert spec.scaling == cum.get("scaling", 1), entry["key"]
    assert spec.eternals_in_shop == bool(cum.get("enable_eternals_in_shop")), entry["key"]
    assert spec.discards == cum.get("starting_params.discards", 0), entry["key"]
    assert spec.perishables_in_shop == bool(cum.get("enable_perishables_in_shop")), entry["key"]
    assert spec.rentals_in_shop == bool(cum.get("enable_rentals_in_shop")), entry["key"]
    return spec


STAKES: dict[int, StakeSpec] = {e["stake_level"]: _spec(e) for e in _STAKES}
STAKE_KEYS: list[str] = [e["key"] for e in _STAKES]
STAKE_BY_KEY: dict[str, StakeSpec] = {s.key: s for s in STAKES.values()}
assert list(STAKES) == list(range(1, 9)), list(STAKES)


def stake_spec(stake) -> StakeSpec:
    """Accept a level (1..8) or a key ('stake_white'..'stake_gold')."""
    if isinstance(stake, str):
        if stake in STAKE_BY_KEY:
            return STAKE_BY_KEY[stake]
        raise KeyError(f"unknown stake {stake!r}; valid: {STAKE_KEYS}")
    if stake in STAKES:
        return STAKES[stake]
    raise KeyError(f"unknown stake level {stake!r}; valid: 1..8")


def apply_stake_to_game(game: "BalatroGame") -> StakeSpec:
    """Engine side of game.lua:2050-2057.  Called from ``BalatroGame._init_game_vars``
    BEFORE the deck hook (``Game:start_run`` applies the stake, then
    ``selected_back:apply_to_run()``).  ``run_state`` must already be built with
    ``RunState.for_stake`` for the generation-side flags."""
    spec = stake_spec(game.stake)
    game.stake_key = spec.key
    game.no_small_blind_reward = spec.no_small_blind_reward
    game.blind_scaling = spec.scaling
    game.base_discards += spec.discards
    game.discards_left = game.base_discards
    return spec
