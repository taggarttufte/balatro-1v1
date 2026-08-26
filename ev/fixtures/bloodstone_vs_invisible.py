"""
bloodstone_vs_invisible.py -- Tagg's acceptance-test fixture (PHASE5_BRIEF_2026-08.md section 0 /
gate 5): "the snapshot advisor on a Bloodstone-vs-Invisible-Joker(+Blueprint) state."

``build()`` drives one ``MLBMatch`` (Red deck / White stake, fixed seed) forward with EV
self-play (``EVPlayer(budget="fast", epsilon=0)``, deterministic) until player 0 reaches a
BLIND_SELECT or SHOP state at ante 4 or 5, then hand-edits the halted match:

  * player 0 gets **Bloodstone** (``j_bloodstone``: 1/2 chance of x1.5 Mult per Heart scored)
    plus a **Hearts-leaning full_deck retint** (a fixture-only shortcut, NOT a real tarot
    effect -- see ``lean_suit`` below).
  * player 1 gets **Invisible Joker + Blueprint**, with Blueprint placed immediately to
    Invisible's LEFT so it copies Invisible's hooks (``debug_add_joker`` appends, so adding
    Blueprint first then Invisible puts Blueprint at index 0 / Invisible at index 1;
    ``jokers.misc._Blueprint._get_copy_target`` copies ``ctx.jokers[idx + 1]``).

    **Engine caveat (read the code before trusting this "combo"):** Blueprint only forwards
    ``pre_score`` / ``on_score_card`` / ``on_held_card`` / ``on_hand_scored`` to its target
    (``engine/balatro_sim/jokers/misc.py::_Blueprint``). Invisible Joker implements
    exactly TWO hooks -- ``on_round_end`` (its own rounds-survived counter) and ``on_sell``
    (the duplicate-a-random-joker effect) -- NEITHER of which Blueprint forwards. So in this
    engine (and in real Balatro: Invisible's ability is not a scoring trigger), **Blueprint
    positioned on Invisible is a no-op at scoring time** -- it does not accelerate Invisible's
    counter, does not trigger its sell effect, and contributes nothing to a hand's score.
    This is exactly the kind of thing the advisor should surface as a genuine "this joker
    slot is doing nothing" read, not something the fixture pretends is a strong synergy.

  * plausible $ and lives are set on both sides (3 vs 4) so a life-race question is live.

``build(seed=..., policy_seed=...)`` is deterministic: the same arguments reproduce the
identical ``match.signature()`` (RNG state included) every call -- ``test_advisor.py`` pins
this. Nothing here is imported by any other workstream; this file and the ``fixtures/``
package are W6-owned.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent            # ev/fixtures
_EV = _HERE.parent                                  # ev
for _p in (str(_EV),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _bootstrap  # noqa: E402,F401
from _bootstrap import MLBMatch, State  # noqa: E402

import player as P  # noqa: E402  (ev/player.py, W3)

__all__ = ["FIXTURE_SEED", "TARGET_ANTES", "TARGET_STATES", "build", "lean_suit"]

FIXTURE_SEED = "TAGGADVR"           # fixed, deterministic (no '0' -- normalize_seed maps '0'->'O')
TARGET_ANTES = (4, 5)
TARGET_STATES = (State.BLIND_SELECT, State.SHOP)
DEFAULT_MAX_STEPS = 6000

P0_LIVES = 3
P1_LIVES = 4
P0_MIN_DOLLARS = 12
P1_MIN_DOLLARS = 8
HEARTS_FRACTION = 0.5                # fraction of the 52-card full_deck retinted to Hearts


def _drive_to_target(seed: str, policy_seed: int, max_steps: int) -> MLBMatch:
    m = MLBMatch(seed=seed, deck_key="b_red", stake=1, lives=4)
    pols = [P.EVPlayer(budget="fast", seed=policy_seed, epsilon=0.0),
           P.EVPlayer(budget="fast", seed=policy_seed + 1, epsilon=0.0)]
    steps = 0
    while not m.done and steps < max_steps:
        g0 = m.games[0]
        if g0.ante in TARGET_ANTES and g0.state in TARGET_STATES:
            return m
        p = m.current_player()
        if p is None:
            raise RuntimeError(f"fixture self-play wedged at step {steps} ({m.state()})")
        acts = m.legal_actions(p)
        m.step(p, pols[p].act(m.games[p]))
        steps += 1
    raise RuntimeError(
        f"bloodstone_vs_invisible: did not reach ante in {TARGET_ANTES} at a "
        f"BLIND_SELECT/SHOP state within {max_steps} steps (seed={seed!r}); "
        f"player 0 ended at ante={m.games[0].ante} state={m.games[0].state}")


def lean_suit(game, suit: str = "Hearts", fraction: float = HEARTS_FRACTION) -> int:
    """Fixture-only shortcut: directly retint plain (non-Stone) ``full_deck`` cards to
    ``suit`` until at least ``fraction`` of the 52-card collection is that suit.  ``deck`` /
    ``hand`` / ``discard_pile`` hold REFERENCES into ``full_deck`` (game.py's own comment:
    "mutating a card in hand mutates the permanent deck automatically"), so this is visible
    everywhere immediately -- exactly as if the player had bought that many Hearts, just
    without spending a purchase or consuming a tarot to get there.  Iterates ``full_deck`` in
    its fixed creation order, so this is deterministic given the same starting deck.
    Returns the number of cards changed."""
    have = sum(1 for c in game.full_deck if c.suit == suit)
    target = int(round(fraction * len(game.full_deck)))
    need = max(0, target - have)
    changed = 0
    for c in game.full_deck:
        if changed >= need:
            break
        if c.suit != suit and c.enhancement != "Stone":
            c.suit = suit
            changed += 1
    return changed


def build(seed: str = FIXTURE_SEED, policy_seed: int = 0, max_steps: int = DEFAULT_MAX_STEPS
         ) -> MLBMatch:
    """The finished fixture: an ``MLBMatch`` halted at ante 4-5, player 0 = Bloodstone +
    Hearts-leaning deck, player 1 = Invisible Joker + Blueprint (positioned to copy it)."""
    m = _drive_to_target(seed, policy_seed, max_steps)
    g0, g1 = m.games

    g0.debug_add_joker("j_bloodstone")
    lean_suit(g0, "Hearts", HEARTS_FRACTION)
    g0.dollars = max(g0.dollars, P0_MIN_DOLLARS)
    g0.lives = P0_LIVES

    g1.debug_add_joker("j_blueprint")   # index 0
    g1.debug_add_joker("j_invisible")   # index 1 -- Blueprint's copy target
    g1.dollars = max(g1.dollars, P1_MIN_DOLLARS)
    g1.lives = P1_LIVES

    return m
