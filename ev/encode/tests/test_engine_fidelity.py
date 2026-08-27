"""test_engine_fidelity.py — each POC item's engine implementation against the game's own
Lua in ``_reference/balatro_src/`` (READ-ONLY; cited ``file:line`` in every docstring).

Same discipline as ``ev/tests/test_extraction.py``: drive the real ``BalatroGame``, never a
hand-forged action dict.  Where the engine and the Lua disagree the test **pins the engine's
current behaviour and names the divergence** — fixing the engine is out of this
workstream's scope (brief), and a silently-skipped xfail would lose the finding.
"""
from __future__ import annotations

import pytest

from _bootstrap import BalatroGame, State
from balatro_sim.jokers.base import JokerInstance
import verify as V
import run_poc as P


def _g(seed="11111111", dollars=None, no_interest=True):
    g = V.in_blind(seed)
    V.set_hand(g, P._HAND)
    g.jokers = []
    if dollars is not None:
        g.dollars = dollars
    g.no_interest = no_interest
    return g


# ═══════════════════════════════════════════════════════════ faithful implementations

def test_cloud_9_pays_a_dollar_per_nine_in_the_full_deck():
    """``calculate_dollar_bonus`` returns ``extra * nine_tally`` (card.lua:1661-1663) with
    ``extra = 1`` (game.lua:444); ``nine_tally`` is recomputed over ALL of
    ``G.playing_cards`` in ``Card:update`` (card.lua:4191-4196), i.e. the whole deck, not
    the draw pile.  Engine: jokers/misc.py:219-226 + game.py:2026-2029."""
    for n in (0, 1, 4, 7):
        g = _g()
        V.set_deck_nines(g, n)
        V.add_joker(g, "j_cloud_9")
        base = _g()
        V.set_deck_nines(base, n)
        before, bb = g.dollars, base.dollars
        g._end_round()
        base._end_round()
        assert (g.dollars - before) - (base.dollars - bb) == n


def test_cloud_9_money_does_not_compound_into_the_same_rounds_interest():
    """The joker dollar rows are added by ``evaluate_round`` (state_events.lua:1174-1181)
    AFTER the interest row is computed from ``G.GAME.dollars`` (:1191).  Contrast the GOLD
    ENHANCEMENT rows, which run inside ``end_round()`` and DO feed the interest base
    (EXTRACT_NOTES §1 fix 4) — the two are easy to conflate and the engine gets both right."""
    g = _g(dollars=20, no_interest=False)
    V.set_deck_nines(g, 5)
    V.add_joker(g, "j_cloud_9")
    base = _g(dollars=20, no_interest=False)
    V.set_deck_nines(base, 5)
    before, bb = g.dollars, base.dollars
    g._end_round()
    base._end_round()
    # exactly $5, not $5 + an extra interest step
    assert (g.dollars - before) - (base.dollars - bb) == 5


def test_rocket_upgrades_before_the_boss_round_pays():
    """``end_of_round`` + ``G.GAME.blind.boss`` -> ``extra.dollars += increase``
    (card.lua:2896-2902) fires inside ``end_round()`` (state_events.lua:101), which precedes
    ``evaluate_round``'s ``calculate_dollar_bonus`` sweep (:1174).  So the boss round itself
    pays the upgraded figure.  Engine: game.py fires on_boss_beaten (≈:2021-2023) before
    on_round_end (≈:2032)."""
    def pay(boss, bonus):
        g = _g()
        if boss:
            V.make_boss_blind(g)
        j = V.add_joker(g, "j_rocket")
        j.state["bonus"] = bonus
        base = _g()
        if boss:
            V.make_boss_blind(base)
        b0, bb = g.dollars, base.dollars
        g._end_round()
        base._end_round()
        return (g.dollars - b0) - (base.dollars - bb), j.state["bonus"]

    assert pay(False, 1) == (1, 1)
    assert pay(True, 1) == (3, 3)          # upgraded, then paid — not $1 then $3 next round
    assert pay(True, 3) == (5, 5)


def test_the_hermit_doubles_money_capped_at_a_twenty_dollar_gain():
    """``ease_dollars(math.max(0, math.min(G.GAME.dollars, self.ability.extra)), true)``
    with ``extra = 20`` (card.lua:1385-1391, game.lua:542).  The cap is on the GAIN: $60
    becomes $80, not $40.  Engine: consumables.py:163-169."""
    for start, gain in ((0, 0), (5, 5), (14, 14), (20, 20), (33, 20), (60, 20)):
        g = _g(dollars=start)
        g.consumable_hand.append("c_hermit")
        acts = [a for a in g.legal_actions()
                if a.get("type") == "use_consumable"
                and a.get("consumable_idx") == len(g.consumable_hand) - 1]
        assert acts, "The Hermit must be usable with no target"
        g.step(acts[0])
        assert g.dollars == start + gain, start


def test_seed_money_raises_the_interest_cap_to_ten_dollars_a_round():
    """Lua stores the cap in BALANCE dollars: base ``interest_cap = 25`` (game.lua:1909),
    Seed Money sets it to its ``config.extra = 50`` (game.lua:602 via card.lua:1933), and
    the row is ``interest_amount * min(floor(dollars/5), interest_cap/5)``
    (state_events.lua:1191-1202).  The engine stores the same cap in INTEREST dollars
    (5 -> 10, consumables.py:520-521, game.py:2006) — a units difference, not a defect."""
    for bal, extra in ((0, 0), (24, 0), (27, 0), (30, 1), (40, 3), (55, 5), (500, 5)):
        g = _g(dollars=bal, no_interest=False)
        P._install_seed_money(g)
        base = _g(dollars=bal, no_interest=False)
        b0, bb = g.dollars, base.dollars
        g._end_round()
        base._end_round()
        assert (g.dollars - b0) - (base.dollars - bb) == extra, bal
    assert g.interest_cap == 10


def test_green_joker_loses_one_mult_per_discard_action_not_per_card():
    """The discard hook is gated on ``context.other_card == context.full_hand[#full_hand]``
    (card.lua:2846-2856), i.e. it runs once for the whole discard.  Engine:
    jokers/scaling.py:51-58 — ``on_discard(inst, cards, ctx)`` is called once per action."""
    g = _g()
    j = V.add_joker(g, "j_green_joker")
    j.state["mult"] = 5
    g.step({"type": "discard", "cards": [0, 1, 2, 3]})       # four cards, ONE action
    assert j.state["mult"] == 4


def test_green_joker_floors_at_zero():
    """``math.max(0, self.ability.mult - discard_sub)`` (card.lua:2848)."""
    g = _g()
    j = V.add_joker(g, "j_green_joker")
    j.state["mult"] = 0
    g.step({"type": "discard", "cards": [0]})
    assert j.state["mult"] == 0


def test_green_joker_gains_before_scoring_so_the_triggering_hand_gets_it():
    """The +1 is in ``context.before`` (card.lua:3563-3569) and the mult is applied at
    :4010-4015, so the hand that increments it already scores the new value."""
    g = _g()
    j = V.add_joker(g, "j_green_joker")
    j.state["mult"] = 0
    g.step({"type": "play", "cards": [0]})
    assert j.state["mult"] == 1


def test_ride_the_bus_reads_the_scoring_hand_not_the_played_set():
    """``for i = 1, #context.scoring_hand ... if ...:is_face()`` (card.lua:3526-3528).  A
    face played as a NON-SCORING kicker must not reset the counter — the engine uses
    ``ctx.scoring_cards`` (jokers/scaling.py:349-358), which is the same set."""
    # A pair of Aces plus a King kicker: the King is played but does not score.
    g = V.in_blind()
    V.set_hand(g, [(14, "Spades"), (14, "Hearts"), (13, "Clubs"), (2, "Clubs"),
                   (3, "Clubs"), (4, "Diamonds"), (5, "Diamonds"), (6, "Hearts")])
    g.jokers = []
    j = V.add_joker(g, "j_ride_the_bus")
    j.state["mult"] = 4
    from balatro_sim.hand_eval import evaluate_hand
    _, scoring = evaluate_hand([g.hand[0], g.hand[1], g.hand[2]])
    assert not any(c.rank == 13 for c in scoring), "precondition: the King must not score"
    g.step({"type": "play", "cards": [0, 1, 2]})
    assert j.state["mult"] == 5, "a non-scoring face must not reset Ride the Bus"


def test_ride_the_bus_resets_on_a_scoring_face():
    g = V.in_blind()
    V.set_hand(g, [(13, "Spades"), (13, "Hearts"), (2, "Clubs"), (3, "Clubs"),
                   (4, "Clubs"), (5, "Diamonds"), (6, "Diamonds"), (7, "Hearts")])
    g.jokers = []
    j = V.add_joker(g, "j_ride_the_bus")
    j.state["mult"] = 6
    g.step({"type": "play", "cards": [0, 1]})
    assert j.state["mult"] == 0


def test_satellite_counts_unique_planets_used_after_purchase():
    """``extra * (# distinct G.GAME.consumeable_usage entries with set == 'Planet')``
    (card.lua:1667-1673), ``extra = 1`` (game.lua:515).  Three uses of Mercury is $1."""
    for keys, expect in (((), 0), (("c_mercury",), 1),
                         (("c_mercury", "c_venus", "c_mars"), 3),
                         (("c_mercury", "c_mercury", "c_venus"), 2)):
        g = _g()
        V.add_joker(g, "j_satellite")
        V.use_planets(g, keys)
        base = _g()
        V.use_planets(base, keys)
        b0, bb = g.dollars, base.dollars
        g._end_round()
        base._end_round()
        assert (g.dollars - b0) - (base.dollars - bb) == expect, keys


def test_ice_cream_applies_chips_then_decays_five():
    """``chip_mod = extra.chips`` at joker_main (card.lua:3915-3920), decrement in
    ``context.after`` (card.lua:3593) — so the CURRENT hand scores the pre-decrement value.
    ``extra = {chips = 100, chip_mod = 5}`` (game.lua:420)."""
    g = _g()
    j = V.add_joker(g, "j_ice_cream")
    assert j.state["chips"] == 100, "on_init must seed the 100 chips"
    g.step({"type": "play", "cards": [0]})
    assert j.state["chips"] == 95


# ═════════════════════════════════════════════════ divergences found by this workstream
#
# Each was pinned against the CURRENT (broken) behaviour so a later fix would have to
# update a test rather than pass silently.  Three of the four have since been fixed by
# W-FIX (2026-08-26) and their pins were flipped to assert the Lua's behaviour — that is
# the pattern working as designed, and the flips are listed in engine/FIX_NOTES.md.  The
# fourth (Cloud 9 / Stone) is still open and still pinned as a gap.

def test_FIXED_satellite_pays_for_planets_used_before_it_was_bought():
    """**Was §3.1, FIXED by W-FIX 2026-08-26.**  Lua reads the GLOBAL
    ``G.GAME.consumeable_usage`` (card.lua:1667-1673), so a Satellite bought at ante 4
    after five distinct planets pays $5 on its very first round end.  The engine kept the
    set on the JOKER INSTANCE, populated only by an ``on_planet_used`` sweep, and never
    seeded it from ``game.planets_used`` — so it paid $0 until NEW planets were used,
    which is 100% of the value of a joker that is essentially always bought mid-run.

    Now ``ScoreContext.planets_used`` carries the run-global list (game.py ``_hook_ctx``)
    and ``jokers/misc.py::_Satellite.on_round_end`` counts its distinct entries.  The
    ``j_satellite`` entry, which was always faithful to the Lua, is accepted by
    ``run_poc`` as a result."""
    g = _g()
    V.use_planets(g, ("c_mercury", "c_venus", "c_mars"))     # BEFORE the purchase
    V.add_joker(g, "j_satellite")
    base = _g()
    V.use_planets(base, ("c_mercury", "c_venus", "c_mars"))
    b0, bb = g.dollars, base.dollars
    g._end_round()
    base._end_round()
    assert (g.dollars - b0) - (base.dollars - bb) == 3


def test_FIXED_blueprint_no_longer_double_scales_a_self_mutating_joker():
    """**Was §3.2, FIXED by W-FIX 2026-08-26** — the finding this POC exists to
    demonstrate, because it was found by MEASUREMENT, not by reading: one seed in 40 of
    the Ice Cream trajectory run came back at 10 chips instead of 40, and the trace showed
    the divergence starting on the hand a Brainstorm was bought (seed 7YTVQERM, hand 7).

    Every self-mutating scaling joker guards its state change with ``and not
    context.blueprint`` in the Lua — Ride the Bus (card.lua:3525), Obelisk (:3543), Green
    Joker (:3563), Ice Cream (:3571), and 22 more branches.  The engine's ``_Blueprint`` /
    ``_Brainstorm`` called the target's hook on the target's own instance with no flag, and
    folded "apply the chips" and "decay" into that one hook, so the copy mutated the target
    a second time: Ice Cream melted twice as fast, Green Joker and Ride the Bus scaled
    twice as fast.

    ``jokers/misc.py::_guarded_call`` now sets ``ctx.blueprint`` (card.lua:2310-2312's
    depth counter) and every guarded joker reads it; the ``before`` / ``joker_main`` /
    ``after`` split means the copy still pays the same row the original does.  Engine-side
    coverage — including the structural check that every guarded joker stays guarded —
    lives in ``engine/tests/engine_tests/test_joker_state_fidelity.py``."""
    g = _g()
    ice = V.add_joker(g, "j_ice_cream")
    V.add_joker(g, "j_brainstorm")           # leftmost joker is Ice Cream -> copies it
    g.step({"type": "play", "cards": [0]})
    assert ice.state["chips"] == 95, "one decrement, as the Lua does"


def test_ENGINE_GAP_cloud_9_counts_a_stone_enhanced_nine():
    """**Divergence (minor).**  ``nine_tally`` counts ``v:get_id() == 9``
    (card.lua:4191-4196), and ``Card:get_id`` returns a random NEGATIVE id for a Stone card
    (card.lua:957-962) — so a 9 that has been turned to Stone stops paying.  The engine
    counts ``c.rank == 9`` regardless of enhancement (game.py:2026-2027).

    Materiality: low.  It needs a Stone conversion (The Justice / Midas-style effects) to
    land on a 9 in a Cloud 9 build.  Recorded because the ``j_cloud_9`` entry states the
    assumption explicitly, and an assumption nobody tested is not an assumption."""
    g = _g()
    V.set_deck_nines(g, 4)
    stone = next(c for c in g.full_deck if c.rank == 9)
    stone.enhancement = "Stone"
    V.add_joker(g, "j_cloud_9")
    base = _g()
    V.set_deck_nines(base, 4)
    next(c for c in base.full_deck if c.rank == 9).enhancement = "Stone"
    b0, bb = g.dollars, base.dollars
    g._end_round()
    base._end_round()
    got = (g.dollars - b0) - (base.dollars - bb)
    assert got == 4, "pinning the CURRENT engine behaviour (the Stone 9 still pays)"
    assert got != 3, "the Lua would pay $3 here — see the docstring"


def test_FIXED_ice_cream_melts():
    """**Was §3.3, FIXED by W-FIX 2026-08-26.**  ``if extra.chips - chip_mod <= 0 then ...
    G.jokers:remove_card(self)`` (card.lua:3571-3592): Ice Cream is DESTROYED on the hand
    that would take it to zero, and the joker slot is freed.  The engine floored it at 0
    (``max(0, chips - 5)``) and kept a dead card on the board forever.

    Materiality was low for scoring (a 0-chip joker adds 0 either way) but not zero for
    POLICY — a permanently-occupied joker slot changes every later buy decision, and the
    encode layer's buy value for Ice Cream should price 20 hands of decay followed by a
    freed slot, not 20 hands followed by a blocked one.  That is now what the engine
    does."""
    g = _g()
    j = V.add_joker(g, "j_ice_cream")
    j.state["chips"] = 5                       # one hand from melting
    slots = g.joker_slots
    g.step({"type": "play", "cards": [0]})
    assert V.joker_of(g, "j_ice_cream") is None, "melted"
    assert g.joker_slots == slots and len(g.jokers) == 0, "and the slot is free"
