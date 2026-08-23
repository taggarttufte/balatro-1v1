"""
sampling.py — draw-world sampling for the EV player (Phase 5 rev 2, W3).

A *world* is a clone of the game in which the hidden information — the ORDER of the draw
pile — has been resampled uniformly.  The composition of the pile (which cards are still
in it) is public: a player can view the remaining deck at any time, so it is never
resampled.  Everything the analytic player computes from a sampled world is therefore an
honest estimate: it can see what a human sees, never the true seed's draw order.

Two paths, chosen at call time:

* ``game.clone_determinized(seed)`` when the engine provides it (W2 adds it concurrently;
  feature-detected with ``hasattr`` on every call, so the module never imports it).
* otherwise ``game.clone()`` + a local ``random.Random`` shuffle of ``deck``.

**Draw-order invariance.**  The live ``game.deck`` is sorted by a canonical card key BEFORE
the shuffle, so two games that differ only by a permutation of the draw pile produce the
SAME sampled world for the same ``rng`` state.  That is what makes the Monte-Carlo
("full") budget's decision a function of the pile's composition, not of its order —
``tests/test_sampling.py`` pins it.

Nothing here touches the original game: ``state_signature()`` is identical before and
after, and ``game.run_state.rng`` is never read except through ``clone()``.
"""
from __future__ import annotations

import random
from typing import Optional

__all__ = ["sample_world", "canonical_card_key", "deck_composition", "world_rng"]


def canonical_card_key(card) -> tuple:
    """Total order on cards by their OBSERVABLE attributes (never ``Card.id``)."""
    return (card.rank, card.suit, card.enhancement, card.edition, card.seal,
            int(getattr(card, "bonus_chips", 0) or 0), bool(card.debuffed),
            bool(getattr(card, "face_down", False)))


def deck_composition(game) -> tuple:
    """Hashable multiset signature of the draw pile (sorted canonical keys)."""
    return tuple(sorted(canonical_card_key(c) for c in game.deck))


def world_rng(seed: int, game, salt: int = 0) -> random.Random:
    """A ``random.Random`` seeded from ``(seed, salt, observable state)`` — so a player
    built with a fixed ``seed`` samples the same worlds at the same observable state,
    whatever the draw pile's order.  Two different live games reaching the same observable
    state sample the same worlds (that is the determinism the brief asks for)."""
    key = (int(seed), int(salt), game.seed_str, game.ante, game.blind_idx,
           int(game.hands_left), int(game.discards_left), int(game.chips_scored),
           tuple(canonical_card_key(c) for c in game.hand), deck_composition(game))
    return random.Random(repr(key))


def sample_world(game, rng: Optional[random.Random] = None):
    """A clone of ``game`` whose draw pile is uniformly reshuffled.

    ``rng`` is a ``random.Random`` (or anything with ``shuffle`` / ``getrandbits``); None
    uses a fresh unseeded generator (only for ad-hoc use — callers wanting determinism
    pass ``world_rng(...)``).  Never mutates ``game``.
    """
    if rng is None:
        rng = random.Random()
    det = getattr(game, "clone_determinized", None)
    if callable(det):
        try:
            w = det(rng.getrandbits(62))
            # W2's API may or may not also canonicalise; re-shuffling from a sorted order
            # with our rng keeps the invariance guarantee independent of its choice.
            w.deck.sort(key=canonical_card_key)
            rng.shuffle(w.deck)
            return w
        except TypeError:
            pass    # unexpected signature: fall through to the local path
    w = game.clone()
    w.deck.sort(key=canonical_card_key)
    rng.shuffle(w.deck)
    return w
