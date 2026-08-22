"""
encoder.py — observation encoding for MCTS, REUSING the fork's env_v7 encoder.

The balatro-mcts original carried a *copy* of V7's `_encode_obs` (434 dims, pre-audit,
15-key boss one-hot, `JOKER_CATALOGUE` names from the old catalogue). Keeping a second
copy is how the two drifted apart in the first place, so this module does not copy
anything: it calls `balatro_sim.env_v7.BalatroV7Env._encode_obs` directly.

That method reads exactly one attribute of `self` — `self.game` — so it is usable as a
pure function of a `BalatroGame` by binding it to a one-slot shim. Verified by
`tests/test_encoder.py::test_encoder_matches_env_v7`, which builds a real `BalatroV7Env`
and asserts byte equality with `encode_obs(env.game)`. If a future env_v7 edit makes
`_encode_obs` touch more of `self`, that test fails loudly rather than silently drifting.

    V7_OBS_DIM   447   = 14 + 8*30 + 5*10 + 7*6 + 12 + 2*8 + (2 + 27 + 28 + 8 + 8)
                         (434 -> 443 -> 447 as the 2026-07-29 audit and W5 widened the
                          boss one-hot to all 23 regular + 5 showdown bosses)

MLB extension (opt-in, `MLB_OBS_DIM = 453`)
-------------------------------------------
The V7 block was written for a vanilla single-player run and is blind to three things
that decide MLB play, so `encode_obs_mlb` appends six features. Justification per feature:

  0 lives / MLB_STARTING_LIVES  the real currency: a lost blind costs a life, not the run
  1 is_pvp                      this blind is the Nemesis (score target is an opponent's
                                live score, every hand is played, $5 win or lose)
  2 pvp_opponent_hands / base   how much the opponent can still add to that target
  3 comeback pending            4 * comeback_bonus is owed at the next Cash Out
  4 pvp_started / waiting       1.0 while the PvP round is live, 0.5 in PVP_WAIT
  5 log-scaled ante             MLB is endless; the V7 `ante/8` scalar saturates at ante 8

`encode_obs` (447) stays the default everywhere: the checkpoint format records which
encoder a run used (`config["encoder"]`), so a net trained on one cannot be silently
loaded into the other.
"""
from __future__ import annotations

import math
from typing import Optional, Protocol

import numpy as np

from balatro_sim.game import BalatroGame, State
from balatro_sim.env_v7 import BalatroV7Env, OBS_DIM as V7_OBS_DIM
from balatro_sim.constants import MLB_STARTING_LIVES

# Back-compat alias for the balatro-mcts call sites (`from mcts.encoder import OBS_DIM`).
OBS_DIM = V7_OBS_DIM


class _GameOnly:
    """Minimal stand-in for BalatroV7Env: `_encode_obs` only ever reads `self.game`."""
    __slots__ = ("game",)

    def __init__(self, game: BalatroGame):
        self.game = game


_v7_encode = BalatroV7Env._encode_obs        # unbound function, not a bound method


def encode_obs(game: BalatroGame) -> np.ndarray:
    """V7's 447-dim observation as a pure function of a BalatroGame."""
    return _v7_encode(_GameOnly(game))


# ── MLB extension ───────────────────────────────────────────────────────────────

MLB_FEATURES = 6
MLB_OBS_DIM = V7_OBS_DIM + MLB_FEATURES


def encode_obs_mlb(game: BalatroGame) -> np.ndarray:
    """V7's 447 dims + six MLB features (see module docstring). Safe on vanilla games:
    every MLB field is inert there, so the block is all zeros except the ante scalar."""
    base = encode_obs(game)
    extra = np.zeros(MLB_FEATURES, dtype=np.float32)
    blind = game.current_blind
    extra[0] = getattr(game, "lives", 0) / max(MLB_STARTING_LIVES, 1)
    extra[1] = float(getattr(blind, "is_pvp", False))
    extra[2] = min(getattr(game, "pvp_opponent_hands", 0) / max(game.base_hands, 1), 2.0)
    if not getattr(game, "comeback_bonus_given", True):
        extra[3] = min(getattr(game, "comeback_bonus", 0) / 4.0, 2.0)
    if game.state is State.PVP_WAIT:
        extra[4] = 0.5
    elif getattr(game, "pvp_started", False):
        extra[4] = 1.0
    extra[5] = math.log1p(max(game.ante, 0)) / math.log1p(16)
    return np.concatenate([base, extra])


# ── Encoder objects (what the model / checkpoint are parameterised by) ──────────

class ObsEncoder(Protocol):
    """A named observation encoder.

    A FLAT encoder (`is_set = False`) returns a 1-D float32 array and `dim` equals its
    length. A SET encoder (`is_set = True`, Phase 4 W1 `encoder_set.SetEncoder`) returns a
    dict of padded arrays + masks and has `dim = None` — callers branch on `is_set`, never
    on `dim` alone.
    """
    name: str
    dim: Optional[int]
    is_set: bool

    def __call__(self, game: BalatroGame): ...


class V7Encoder:
    """The default: V7's 447-dim observation, reused from the fork env."""
    name = "v7"
    dim = V7_OBS_DIM
    is_set = False

    def __call__(self, game: BalatroGame) -> np.ndarray:
        return encode_obs(game)


class MLBEncoder:
    """V7 + the six MLB features. Opt in with `--encoder mlb`."""
    name = "mlb"
    dim = MLB_OBS_DIM
    is_set = False

    def __call__(self, game: BalatroGame) -> np.ndarray:
        return encode_obs_mlb(game)


def _set_encoder_cls():
    """Imported lazily: `encoder_set` pulls in game_keys / consumables / shop tables that
    a caller only wanting the 447-dim encoder should not pay for."""
    from .encoder_set import SetEncoder
    return SetEncoder


ENCODERS: dict[str, object] = {"v7": V7Encoder, "mlb": MLBEncoder, "set": _set_encoder_cls}


def get_encoder(name: str = "v7", **kwargs) -> ObsEncoder:
    """`"v7"` (447) | `"mlb"` (453) | `"set"` (the Phase 4 set encoder).

    `kwargs` reach the constructor — `get_encoder("set", caps=...)` rebuilds an encoder
    with a checkpoint's recorded caps.
    """
    try:
        entry = ENCODERS[name]
    except KeyError:
        raise ValueError(f"unknown encoder {name!r}; known: {sorted(ENCODERS)}") from None
    cls = entry() if entry is _set_encoder_cls else entry
    return cls(**kwargs)


def is_set_encoder(enc) -> bool:
    return bool(getattr(enc, "is_set", False))
