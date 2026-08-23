"""
determinize.py — Phase 5 W2: a non-clairvoyant MCTS player, built without touching
`search.py` / `player.py`.

The problem (PHASE5_BRIEF_2026-08.md §0): every `real1` decision came from a search
whose simulations cloned the TRUE game (`BalatroGame.clone()`, which copies the keyed
RNG and draw-pile order verbatim) — every simulated future saw the actual future draws,
reroll results, pack contents and probability rolls. `mp/engine/balatro_sim/game.py`'s
new `clone_determinized(seed)` fixes the primitive (observed state stays bit-identical;
the draw pile is reshuffled and the keyed RNG is reseeded); this module wires it into
MCTS.

Why no `search.py`/`player.py` edit was needed: read `search.py`'s own module
docstring point 6 and `MCTS._run_sims_iter` — every simulation clones the root game
**exactly once**, at `gen = self._simulate_iter(root, root_game.clone(), ...)`. That is
the ONLY per-simulation clone call in the whole search (SELECT/EXPAND after that just
call `.step()` on the one clone). So handing the search a `root_game` whose *bound*
`.clone()` method is shadowed (instance-level only — the class and every other
`BalatroGame` are untouched) to call `clone_determinized` instead is sufficient to make
every simulation search a freshly-sampled world, with zero changes to search.py/player.py
and the untouched code path staying byte-identical (337 agent tests green).

Two modes (`DeterminizedMCTSPlayer.mode`):

  "per_sim" (default) — PIMC-style: ONE tree is built per decision as usual (one root,
      normal PUCT/Gumbel bookkeeping), but each of its `sims` simulations descends into
      an INDEPENDENTLY resampled world (`make_determinizing_view`). No two simulations
      share a sampled world; the root's Q/visit statistics average over many worlds.
      This is the honest reading of "a determinized search" and the default the brief
      asks to benchmark. Cost: one `clone_determinized` call per simulation instead of
      one `clone()` — see DETERMINIZE_NOTES.md for the measured overhead (within the
      engine's ≤1.5x-of-`clone()` budget per call; the search-level slowdown is that
      factor, since call COUNT is unchanged).
  "per_search" (cheap option) — ONE determinized world is sampled per decision and the
      WHOLE search runs inside it (one `clone_determinized` call total). Cheaper, but
      every simulation in a decision sees the same sampled future — closer to "guess one
      plausible world and search it perfectly" than to PIMC. Useful when `sims` is large
      enough that per-simulation determinization would dominate wall-clock.

Tree reuse (`mcts/reuse.py::TreeCache`) is left exactly as `make_player(...)` configured
it (default on, matching `real1.sh`). Under EITHER mode here a retained subtree from the
previous decision essentially never hits: `TreeCache.store()` computes the next-state
signature by calling `root_game.clone()` too (the SAME shadowed method in "per_sim"
mode), stepping a resampled world, and comparing its `state_signature()` (RNG + deck
order included) against the TRUE next game's signature next decision — those can only
coincide by an astronomically unlikely hash collision, so `TreeCache.take()` reliably
misses and falls back to a fresh search. This is not a bug (see the reasoning in the
class docstring below): `ReuseConfig`'s default `budget_mode="subtract"` counts retained
visits toward the `num_simulations` total, so a miss just means "no visits were free
this decision" — evidence-per-decision (`num_simulations`) is unaffected, only wall
clock is (report `player.inner.reuse_stats` for the always-near-0 hit rate).
"""
from __future__ import annotations

import secrets
import types
from dataclasses import dataclass, field
from typing import Iterator, Optional

from balatro_sim.game import BalatroGame

from .player import MCTSPlayer, make_player

__all__ = ["DeterminizedMCTSPlayer", "make_determinized_player", "make_determinizing_view",
          "seed_stream"]


def seed_stream(base_seed: "int | None") -> Iterator[int]:
    """An endless generator of ints for `clone_determinized(seed=...)`. Deterministic
    (and therefore reproducible across two runs) given `base_seed`; genuinely fresh
    (`secrets`) when `base_seed is None`, exactly like `clone_determinized(seed=None)`
    itself. A tiny counter-based LCG, not `random.Random` — `mp/engine` forbids the
    stdlib `random` module inside `balatro_sim/` (`test_no_random_module_in_engine`),
    and this module intentionally mirrors that discipline even though it lives outside
    the guarded tree, so every draw in the whole determinize path stays traceable to one
    of two primitives: `secrets` (non-reproducible) or an explicit int seed."""
    if base_seed is None:
        while True:
            yield secrets.randbelow(1 << 40)
    # splitmix64-style counter stream: fast, well-mixed, no stdlib `random`.
    state = int(base_seed) & 0xFFFFFFFFFFFFFFFF
    while True:
        state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        z = z ^ (z >> 31)
        yield z & ((1 << 40) - 1)


def make_determinizing_view(game: BalatroGame, seeds: Iterator[int]) -> BalatroGame:
    """A fresh, independent copy of `game` (`game.clone()` — never mutates `game`) whose
    bound `.clone()` method is shadowed, INSTANCE-ONLY, to return
    `self.clone_determinized(next(seeds))` instead. The returned object's own state is a
    faithful (non-determinized) copy of `game` — only what happens when something
    downstream calls `.clone()` ON IT is resampled, which is exactly what MCTS's
    per-simulation `root_game.clone()` does."""
    view = game.clone()

    def _determinized_clone(self: BalatroGame) -> BalatroGame:
        # `clone_determinized` itself calls `self.clone()` internally (to get the plain
        # structural copy it then reseeds/reshuffles) — since `self` IS `view`, that
        # inner call would resolve back to THIS shadowed method and recurse forever.
        # Pop the instance-level shadow for the duration of the call so
        # `clone_determinized`'s own `self.clone()` falls through to the ordinary class
        # method, then restore it so the NEXT simulation's clone is intercepted again.
        shadow = self.__dict__.pop("clone")
        try:
            return self.clone_determinized(next(seeds))
        finally:
            self.clone = shadow

    view.clone = types.MethodType(_determinized_clone, view)   # instance dict only
    return view


@dataclass
class DeterminizedMCTSPlayer:
    """Wraps an `MCTSPlayer` so its search runs against sampled worlds instead of the
    true future. Same minimal `Player` protocol as every other player in this repo
    (`act(game) -> action | None`, `reset()`) — a drop-in substitute for
    `mp/tournament`/`mp/eval` drivers and for `mcts.player.MCTSPlayer` itself.

    `determinize_seed`: seeds the internal `seed_stream` that feeds every
    `clone_determinized` call this player makes, across its whole lifetime (every
    decision, every simulation in "per_sim" mode). `None` (default) draws fresh,
    non-reproducible randomness every call — matching real deployment. A fixed int
    makes an entire playthrough (or an entire batch of them, if the caller advances the
    SAME stream deliberately) bit-reproducible, which is what the invariant tests and
    "same seed -> same reroll" style checks rely on.
    """
    inner: MCTSPlayer
    determinize_seed: Optional[int] = None
    mode: str = "per_sim"          # "per_sim" (PIMC, default) | "per_search" (cheap)
    name: str = "det-mcts"
    _stream: Iterator[int] = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self.mode not in ("per_sim", "per_search"):
            raise ValueError(f"mode must be 'per_sim' or 'per_search', got {self.mode!r}")
        self._stream = seed_stream(self.determinize_seed)

    # ── Player protocol ──────────────────────────────────────────────────────

    def act(self, game: BalatroGame):
        if not game.legal_actions():
            return self.inner.act(game)      # no-action state: nothing to determinize
        if self.mode == "per_search":
            view = game.clone_determinized(next(self._stream))
        else:
            view = make_determinizing_view(game, self._stream)
        return self.inner.act(view)

    def reset(self) -> None:
        self.inner.reset()

    def __repr__(self) -> str:
        seed = "fresh" if self.determinize_seed is None else self.determinize_seed
        return f"DeterminizedMCTSPlayer({self.name!r}, mode={self.mode!r}, seed={seed})"


def make_determinized_player(checkpoint: Optional[str] = None, sims: int = 40,
                             device: str = "cpu", seed: int = 0,
                             determinize_seed: Optional[int] = None,
                             mode: str = "per_sim", **kwargs) -> DeterminizedMCTSPlayer:
    """`mcts.player.make_player(...)`, wrapped for determinized search. Same call
    signature/kwargs (so the clairvoyance measurement's baseline and determinized arms
    differ by exactly this one wrapper call — see `real1.sh`'s heuristic-prior /
    sims / encoder flags, which both arms must share to isolate "sees the future" as
    the only variable). `seed` seeds the INNER player's own rng (Gumbel sampling,
    Dirichlet noise if enabled); `determinize_seed` seeds the WORLD sampling — they are
    deliberately separate knobs.
    """
    inner = make_player(checkpoint=checkpoint, sims=sims, device=device, seed=seed, **kwargs)
    return DeterminizedMCTSPlayer(inner=inner, determinize_seed=determinize_seed, mode=mode)
