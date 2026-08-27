"""test_cycle.py — W-CYCLE: the per-target tarot dig (``ev/CYCLE_NOTES.md``).

Three things are pinned here, all on hand-built ``SELECTING_HAND`` states (the idiom
``test_extraction.py`` established):

* **the maths** — ``_target_quality`` is the layer-cake expectation of the k-th best target
  grade, checked against the hypergeometric numbers written out by hand;
* **the economics** — a clearing play banks no dig (the round ends and the cards it draws
  go back into the deck), a throwaway does; the dig lines obey the extraction safety gate
  and never fire while a Nemesis race is live;
* **the flag** — ``tarot_per_target=False`` is EXTRACT_NOTES §6's per-COUNT form
  bit-for-bit, which is what the gate / h2h A/B arm runs.

The acceptance-fixture end of this workstream lives in ``test_probe_fixtures.py``.
"""
from __future__ import annotations

from dataclasses import replace
from math import comb

import pytest

import _bootstrap  # noqa: F401
from _bootstrap import BalatroGame, State

import hand as H

OLD = replace(H.DEFAULT_HAND_CONFIG, tarot_per_target=False)


def _in_blind(seed="11111111", ruleset="vanilla"):
    g = BalatroGame(seed=seed, ruleset=ruleset)
    g.step({"type": "play_blind"})
    assert g.state == State.SELECTING_HAND
    return g


def _set_hand(g, specs):
    pool = {}
    for c in g.full_deck:
        pool.setdefault((c.rank, c.suit), c)
    hand = [pool[s] for s in specs]
    ids = {id(c) for c in hand}
    g.deck = [c for c in g.full_deck if id(c) not in ids]
    g.hand = hand
    g.discard_pile = []
    for c in hand:
        c.face_down = False
        c.debuffed = False
    return g


def _an(g, cfg=H.DEFAULT_HAND_CONFIG, **kw):
    return H.HandAnalysis(g, cfg, legal=g.legal_actions(), **kw)


#: the fixture board, hand-built: a Clubs flush + one Heart + two junk Spades, Sun held
_SUN_HAND = [(3, "Clubs"), (5, "Clubs"), (7, "Clubs"), (9, "Clubs"), (11, "Clubs"),
             (14, "Hearts"), (2, "Spades"), (8, "Spades")]


def _sun_state(target=250, heart=(14, "Hearts")):
    spec = list(_SUN_HAND)
    spec[5] = heart
    g = _set_hand(_in_blind(), spec)
    g.consumable_hand = ["c_sun"]
    g.current_blind.chips_target = target
    g.dollars = max(g.dollars, 10)
    return g


# ══════════════════════════════════════════════════════════════════════ the maths

def test_target_quality_is_the_layer_cake_expectation_of_the_kth_best_grade():
    """``E[X_(k)] = Σ_l (g_l − g_(l+1))·P(#targets of grade ≥ g_l ≥ k)``, written out by
    hand for the Sun board: 44 cards in the pile, 12 Hearts of which 0 are Aces (the Ace of
    Hearts is the one in hand), 4 are ten-or-better and 7 are seven-or-better; the tarot
    needs ``flush_need − 3 = 2`` real Hearts and the kept set already supplies one, at the
    TOP grade tier, so every tier needs exactly one more card."""
    g = _sun_state()
    an = _an(g)
    (want,) = an._tarot_wants
    assert want.k == 2 and want.pile == (0, 4, 7, 12)
    keep = an.full_mask & ~(1 << 6) & ~(1 << 7)          # discard the two junk Spades
    m, N = 2, 44

    def p_at_least_one(K):
        return 1.0 - comb(N - K, m) / comb(N, m)

    tiers = H._GRADE_TIERS
    expect = (0.0                                        # tier 0: no Heart Aces left
              + (tiers[1] - tiers[2]) * p_at_least_one(4)
              + (tiers[2] - tiers[3]) * p_at_least_one(7)
              + tiers[3] * p_at_least_one(12))
    assert an._target_quality(want, keep, m) == pytest.approx(expect)
    # ... and the dollars are that quality times the tarot's value and the cycle fraction
    assert an._cycle_ev(keep, m) == pytest.approx(
        expect * an.cfg.tarot_value_dollars * an.cfg.tarot_cycle_fraction)


def test_target_quality_is_monotone_in_the_draw_and_bounded_by_one():
    g = _sun_state()
    an = _an(g)
    (want,) = an._tarot_wants
    keep = an.full_mask & ~(1 << 6) & ~(1 << 7)
    vals = [an._target_quality(want, keep, m) for m in range(0, 6)]
    assert vals[0] == 0.0                       # no draw, and only one of the two Hearts
    assert all(b >= a for a, b in zip(vals, vals[1:]))
    assert vals[-1] <= 1.0


def test_a_better_held_target_is_worth_more_at_the_same_count_and_draw():
    """The per-target claim in its smallest form: the Ace of Hearts covers every grade tier
    on its own, the Four of Hearts only the bottom one, so the same line drawing the same
    two cards is worth strictly more with the Ace held.  The per-COUNT form cannot see it."""
    line = {"type": "discard", "cards": [6, 7]}
    hi = _an(_sun_state()).extraction_ev(line)
    lo = _an(_sun_state(heart=(4, "Hearts"))).extraction_ev(line)
    assert hi > lo > 0.0
    hi_old = _an(_sun_state(), OLD).extraction_ev(line)
    lo_old = _an(_sun_state(heart=(4, "Hearts")), OLD).extraction_ev(line)
    assert hi_old == pytest.approx(lo_old)


def test_an_enhancement_tarot_wants_one_plain_card_and_stops_wanting_when_it_has_one():
    """The eight enhancement tarots take ``k = 1``: a line that keeps any plain card can use
    the tarot right now, so the dig is worth nothing on it; a line that keeps only enhanced
    cards has to draw one."""
    g = _set_hand(_in_blind(), _SUN_HAND)
    g.consumable_hand = ["c_chariot"]           # Steel, an enhancement tarot
    an = _an(g)
    (want,) = an._tarot_wants
    assert want.k == 1 and want.have_mask == an.full_mask     # every card is plain
    assert an._cycle_ev(an.full_mask & ~0b11, 2) == 0.0       # a plain card is still kept
    for c in g.hand[:6]:
        c.enhancement = "Steel"
    an2 = _an(g)
    keep = an2.full_mask & ~(1 << 6) & ~(1 << 7)              # only enhanced cards kept
    assert an2._cycle_ev(keep, 2) > 0.0


# ══════════════════════════════════════════════════════════════════ the economics

def test_a_clearing_play_banks_no_dig_and_a_throwaway_does():
    """``_play_continues``: the round ends on a clear, and ``game._end_round`` puts the whole
    hand back into the deck, so the five cards a clearing play draws never carry the tarot.
    A play that leaves the blind alive keeps the drawn cards in hand for the next decision."""
    g = _sun_state()
    an = _an(g)
    clear = {"type": "play", "cards": [0, 1, 2, 3, 4]}       # the Clubs flush, 276 >= 250
    throwaway = {"type": "play", "cards": [7]}               # a lone junk Spade
    assert an._exact_of((0, 1, 2, 3, 4)) >= an.need
    assert an.extraction_ev(clear) == 0.0
    assert an.extraction_ev(throwaway) > 0.0
    # ... and it is only the CLEAR that kills it, not the play-ness: raise the target out of
    # reach of one hand and the same flush play banks the dig again
    g2 = _sun_state(target=10 ** 4)
    an2 = _an(g2)
    assert an2.extraction_ev(clear) > 0.0


def test_the_last_hand_never_banks_a_dig():
    g = _sun_state(target=10 ** 4)
    g.hands_left = 1
    an = _an(g)
    assert an._play_continues(0.0) == 0.0
    assert an.extraction_ev({"type": "play", "cards": [7]}) == 0.0


def test_dig_lines_keep_the_targets_and_are_safety_gated():
    g = _sun_state()
    an = _an(g)
    lines = an._dig_lines()
    assert lines and all(5 not in t for t in lines)     # index 5 is the Ace of Hearts
    assert (6, 7) in lines                              # ... and the floor-preserving one
    g.current_blind.chips_target = 10 ** 7              # unreachable: the gate shuts
    assert _an(g)._dig_lines() == []


def test_dig_lines_are_off_at_a_live_nemesis():
    g = _sun_state()
    g.current_blind.is_pvp = True
    an = _an(g)
    assert an.extract_on is False and an._dig_lines() == []
    # with the W-PVP money layer on, still off while the race is LIVE
    cfg = replace(H.DEFAULT_HAND_CONFIG, pvp_extract=True)
    an2 = _an(g, cfg)
    assert an2.pvp_decided() == "" and an2._dig_lines() == []


def test_the_dig_scales_linearly_with_the_measured_tarot_value():
    """``tarot_values`` is W-SHOP's measured per-deck valuation (CYCLE_NOTES.md §2); the dig
    is linear in it, so a $16 Star digs exactly 4x as hard as the flat $4 placeholder."""
    line = {"type": "discard", "cards": [6, 7]}
    flat = _an(_sun_state()).extraction_ev(line)
    g = _sun_state()
    an = H.HandAnalysis(g, H.DEFAULT_HAND_CONFIG, legal=g.legal_actions(),
                        tarot_values={"c_sun": 16.0})
    assert an.extraction_ev(line) == pytest.approx(4.0 * flat)


# ══════════════════════════════════════════════════════════════════════ the flag

def test_tarot_per_target_off_is_the_old_per_count_form():
    """EXTRACT_NOTES §6's formula, written out: ``P(draw the missing targets) · $4 · 0.5``,
    with no grading, no round-continues conditioning and no dig lines."""
    g = _sun_state()
    an = _an(g, OLD)
    (want,) = an._tarot_wants
    keep = an.full_mask & ~(1 << 6) & ~(1 << 7)         # one Heart kept, so one missing
    expect = (H._hyper_tail(44, 12, 2, 1)
              * OLD.tarot_value_dollars * OLD.tarot_cycle_fraction)
    assert an._cycle_ev(keep, 2) == pytest.approx(expect)
    assert an._dig_lines() == []
    assert an.extraction_ev({"type": "play", "cards": [0, 1, 2, 3, 4]}) > 0.0


def test_a_board_with_no_targeted_tarot_is_untouched_by_this_workstream():
    """No held tarot -> no wants, no dig lines, and the ranking is identical with the flag
    either way (this is the 126-seed gate's common case)."""
    for seed in ("11111111", "CHPB293X", "7I4M53DL"):
        g = _in_blind(seed)
        an = _an(g)
        assert an._tarot_wants == [] and an._dig_lines() == []
        a = H.rank_hand_actions(g)
        b = H.rank_hand_actions(g, cfg=OLD)
        assert [H._action_sort_key(x) for x, _ in a] == [H._action_sort_key(x) for x, _ in b]
        assert [pytest.approx(ev) for _, ev in a] == [ev for _, ev in b]


def test_dig_decisions_stay_legal_and_side_effect_free():
    g = _sun_state()
    sig = g.state_signature()
    rng = g.run_state.rng.snapshot()
    ranked = H.rank_hand_actions(g)
    legal = {H._action_sort_key(a) for a in g.legal_actions()}
    assert ranked and all(H._action_sort_key(a) in legal for a, _ in ranked)
    assert g.state_signature() == sig and g.run_state.rng.snapshot() == rng
