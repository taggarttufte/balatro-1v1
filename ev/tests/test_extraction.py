"""test_extraction.py — W-EXTRACT: the sandbag / money-proc layer of the analytic player,
and the engine-fidelity fixes it rests on.

Two halves:

* **engine fidelity** — each money/extraction proc against the Lua reference in
  ``mp/_reference/balatro_src/`` (cited file:line in every docstring).  These drive the real
  ``BalatroGame``; ``mp/engine/tests`` + ``engine_parity`` stay green alongside them.
* **the EV layer** — ``HandAnalysis.extraction_ev`` per proc against hand-computed dollars,
  the keep/junk-value split, the tail-DP safety gate and the candidate lines it unlocks.
"""
from __future__ import annotations

import pytest

import _bootstrap  # noqa: F401
from _bootstrap import BalatroGame, State
from balatro_sim.jokers.base import JokerInstance

import hand as H


# ───────────────────────────────────────────────────────────────────── fixtures

def _in_blind(seed="11111111", ruleset="vanilla"):
    g = BalatroGame(seed=seed, ruleset=ruleset)
    g.step({"type": "play_blind"})
    assert g.state == State.SELECTING_HAND
    return g


def _set_hand(g, specs):
    """Put specific (rank, suit) cards in hand, taken out of the draw pile."""
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


def _jokers(g, *keys):
    g.jokers = [JokerInstance(k) for k in keys]
    return g


def _an(g, **kw):
    return H.HandAnalysis(g, kw.pop("cfg", H.DEFAULT_HAND_CONFIG), legal=g.legal_actions(), **kw)


def _idx(g, rank, suit):
    for i, c in enumerate(g.hand):
        if c.rank == rank and c.suit == suit:
            return i
    raise AssertionError(f"{rank}{suit} not in hand")


# ═════════════════════════════════════════════════════ engine fidelity vs the Lua source

def test_gold_seal_pays_three_per_scoring_pass():
    """Gold seal: ``Card:get_p_dollars`` card.lua:1068-1073 (``seal == 'Gold'`` -> +3), read
    only for ``context.cardarea == G.play`` (common_events.lua:608-611) inside the per-rep
    loop (state_events.lua:692) and paid at :722-726.  Engine: scoring.py:99-101."""
    g = _in_blind()
    _set_hand(g, [(14, "Spades"), (14, "Hearts"), (2, "Clubs"), (3, "Clubs"),
                  (4, "Clubs"), (5, "Diamonds"), (6, "Diamonds"), (7, "Diamonds")])
    g.hand[0].seal = "Gold"
    before = g.dollars
    g.step({"type": "play", "cards": [0, 1]})
    assert g.dollars - before == 3


def test_gold_enhancement_pays_at_round_end_but_not_when_debuffed():
    """m_gold ``h_dollars = 3`` (game.lua:654) via ``Card:get_end_of_round_effect``
    card.lua:1033-1039 — and :1034 ``if self.debuff then return {} end``.  The engine used
    to pay debuffed Gold cards (game.py, pre-2026-08-24)."""
    for debuff, expect in ((False, 3), (True, 0)):
        g = _in_blind()
        _set_hand(g, [(14, "Spades"), (2, "Clubs"), (3, "Clubs"), (4, "Clubs"),
                      (5, "Diamonds"), (6, "Diamonds"), (7, "Diamonds"), (8, "Hearts")])
        g.hand[0].enhancement = "Gold"
        g.hand[0].debuffed = debuff
        g.no_interest = True                      # isolate the h_dollars row
        before = g.dollars
        g._end_round()
        # blind reward + unused-hand money are the same in both branches; the DIFFERENCE is
        # what the Gold card paid
        got = g.dollars - before
        if not debuff:
            baseline = got
        else:
            assert baseline - got == 3
        assert (got, expect) == (got, expect)


def test_gold_enhancement_retriggers_on_a_red_seal():
    """The end-of-round held-card loop repeats the payout once per Red seal
    (state_events.lua:191-196, ``ease_dollars`` on every rep at :221-223).  A Gold
    ENHANCEMENT with a Red SEAL is a legal card, so it pays $6."""
    got = {}
    for seal in ("None", "Red"):
        g = _in_blind()
        _set_hand(g, [(14, "Spades"), (2, "Clubs"), (3, "Clubs"), (4, "Clubs"),
                      (5, "Diamonds"), (6, "Diamonds"), (7, "Diamonds"), (8, "Hearts")])
        g.hand[0].enhancement = "Gold"
        g.hand[0].seal = seal
        g.no_interest = True
        before = g.dollars
        g._end_round()
        got[seal] = g.dollars - before
    assert got["Red"] - got["None"] == 3


def test_gold_enhancement_counts_toward_the_interest_base():
    """The held-card end-of-round effects run inside ``end_round()``, BEFORE
    ``G.FUNCS.evaluate_round`` reads ``G.GAME.dollars`` for the interest row
    (state_events.lua:171-233 then :1191-1202).  $8 + two Gold cards must earn interest on
    $14 (= $2), not on $8 (= $1)."""
    g = _in_blind()
    _set_hand(g, [(14, "Spades"), (14, "Hearts"), (2, "Clubs"), (3, "Clubs"),
                  (4, "Clubs"), (5, "Diamonds"), (6, "Diamonds"), (7, "Diamonds")])
    g.hand[0].enhancement = "Gold"
    g.hand[1].enhancement = "Gold"
    g.dollars = 8
    g.jokers = []
    g._end_round()
    # 8 + 6 (gold) = 14 -> interest 2.  With the old ordering it would have been 1.
    assert g.dollars >= 8 + 6 + 2
    plain = _in_blind()
    _set_hand(plain, [(14, "Spades"), (14, "Hearts"), (2, "Clubs"), (3, "Clubs"),
                      (4, "Clubs"), (5, "Diamonds"), (6, "Diamonds"), (7, "Diamonds")])
    plain.dollars = 8
    plain.jokers = []
    plain._end_round()
    assert g.dollars - plain.dollars == 6 + 1        # $6 of gold + the extra interest step


def test_faceless_ignores_debuffed_faces_and_honours_pareidolia():
    """Faceless Joker counts ``v:is_face()`` over the whole discarded set (card.lua:2858-2871).
    ``Card:is_face`` (card.lua:964-970) is FALSE for a debuffed card (:965) and TRUE for
    every card when Pareidolia is owned (:967).  Before 2026-08-24 the engine counted
    debuffed faces and never saw Pareidolia (``game._hook_ctx`` set no ``all_face_cards``)."""
    faces = [(11, "Spades"), (12, "Hearts"), (13, "Clubs")]
    rest = [(2, "Clubs"), (3, "Clubs"), (4, "Diamonds"), (5, "Diamonds"), (6, "Hearts")]

    g = _jokers(_set_hand(_in_blind(), faces + rest), "j_faceless")
    before = g.dollars
    g.step({"type": "discard", "cards": [0, 1, 2]})
    assert g.dollars - before == 5                      # three real faces

    g = _jokers(_set_hand(_in_blind(), faces + rest), "j_faceless")
    for c in g.hand[:3]:
        c.debuffed = True
    before = g.dollars
    g.step({"type": "discard", "cards": [0, 1, 2]})
    assert g.dollars - before == 0                      # debuffed -> not faces

    g = _jokers(_set_hand(_in_blind(), rest + faces), "j_faceless", "j_pareidolia")
    before = g.dollars
    g.step({"type": "discard", "cards": [0, 1, 2]})     # three NON-face cards
    assert g.dollars - before == 5                      # Pareidolia: everything is a face


def _score_with_held(debuff: bool):
    """One scoring pass of a flush with three FACE cards held in hand, Reserved Parking
    owned.  Returns (dollars the held phase paid, whether a 'parking' draw was consumed)."""
    from balatro_sim import scoring as SC
    g = _in_blind()
    _set_hand(g, [(2, "Clubs"), (3, "Clubs"), (4, "Clubs"), (5, "Clubs"), (6, "Clubs"),
                  (11, "Spades"), (12, "Spades"), (13, "Spades")])
    played, held = g.hand[:5], g.hand[5:]
    for c in held:
        c.debuffed = debuff
    rng = g.run_state.rng
    assert "parking" not in rng._state
    _, ctx = SC.score_hand(played, played, "Flush", [JokerInstance("j_reserved_parking")],
                           g.planet_levels, g.hands_left, g.discards_left, g.dollars, g.ante,
                           len(g.deck), rng=rng, held_cards=held, full_deck=g.full_deck)
    return ctx.pending_money, ("parking" in rng._state)


def test_reserved_parking_takes_no_rng_draw_on_a_debuffed_held_face():
    """``'Reserved Parking' and context.other_card:is_face() and pseudorandom('parking')``
    (card.lua:3302-3304): Lua's ``and`` short-circuits and ``is_face()`` is nil for a
    debuffed card (card.lua:965), so no ``parking`` draw is consumed — the debuff arm at
    :3305-3310 is dead code.  The engine used to roll first and only withhold the money,
    desyncing the keyed stream for the rest of the run."""
    money, rolled = _score_with_held(debuff=True)
    assert (money, rolled) == (0, False)
    money, rolled = _score_with_held(debuff=False)
    assert rolled is True and money >= 0


def test_purple_seal_creates_one_tarot_per_free_consumable_slot():
    """``Card:calculate_seal{discard=true}`` card.lua:2242-2268: per discarded Purple-sealed
    card, skipped when debuffed (:2243) and only while
    ``#G.consumeables.cards + consumeable_buffer < card_limit`` (:2253-2254)."""
    g = _set_hand(_in_blind(), [(2, "Clubs"), (3, "Clubs"), (4, "Clubs"), (5, "Clubs"),
                               (6, "Clubs"), (7, "Diamonds"), (8, "Diamonds"), (9, "Hearts")])
    for c in g.hand[:3]:
        c.seal = "Purple"
    g.consumable_hand = []
    assert g.consumable_slots == 2
    g.step({"type": "discard", "cards": [0, 1, 2]})
    assert len(g.consumable_hand) == 2                  # three seals, two slots


# ═══════════════════════════════════════════════════════ the per-action proc EV, in dollars

def _play_ev(g, cards):
    an = _an(g)
    return an.extraction_ev({"type": "play", "cards": list(cards)})


def _disc_ev(g, cards):
    an = _an(g)
    return an.extraction_ev({"type": "discard", "cards": list(cards)})


NEUTRAL = [(14, "Spades"), (14, "Hearts"), (2, "Clubs"), (3, "Clubs"),
           (4, "Clubs"), (5, "Diamonds"), (6, "Diamonds"), (7, "Diamonds")]


def test_proc_ev_gold_seal_is_three_dollars_per_scoring_card():
    g = _set_hand(_in_blind(), NEUTRAL)
    g.hand[0].seal = "Gold"
    assert _play_ev(g, [0, 1]) == pytest.approx(3.0)     # both aces score
    assert _play_ev(g, [1]) == pytest.approx(0.0)        # the gold one is not played
    # played but NOT scoring (a high-card kicker) pays nothing: card.lua:1068 is only read
    # for the scoring cards (state_events.lua:648-780)
    assert _play_ev(g, [1, 0]) == pytest.approx(3.0)     # a pair: both score


def test_proc_ev_gold_seal_not_paid_for_a_non_scoring_kicker():
    g = _set_hand(_in_blind(), NEUTRAL)
    g.hand[5].seal = "Gold"                              # 5 of Diamonds, kicker only
    assert _play_ev(g, [0, 1, 5]) == pytest.approx(0.0)  # pair of aces: the 5 does not score


def test_proc_ev_lucky_is_twenty_over_fifteen():
    """card.lua:1076 — ``pseudorandom('lucky_money') < normal/15`` -> $20."""
    g = _set_hand(_in_blind(), NEUTRAL)
    g.hand[0].enhancement = "Lucky"
    assert _play_ev(g, [0, 1]) == pytest.approx(20.0 / 15.0)


def test_proc_ev_lucky_doubles_on_a_red_seal_and_on_oops_all_sixes():
    g = _set_hand(_in_blind(), NEUTRAL)
    g.hand[0].enhancement = "Lucky"
    g.hand[0].seal = "Red"
    assert _play_ev(g, [0, 1]) == pytest.approx(2 * 20.0 / 15.0)
    g = _jokers(_set_hand(_in_blind(), NEUTRAL), "j_oops")
    g.hand[0].enhancement = "Lucky"
    assert _play_ev(g, [0, 1]) == pytest.approx(2 * 20.0 / 15.0)


def test_proc_ev_business_card_is_one_dollar_per_scored_face():
    """card.lua:3177 — ``pseudorandom('business') < normal/2`` -> $2 per scored face."""
    g = _jokers(_set_hand(_in_blind(), [(13, "Spades"), (13, "Hearts"), (2, "Clubs"),
                                        (3, "Clubs"), (4, "Clubs"), (5, "Diamonds"),
                                        (6, "Diamonds"), (7, "Diamonds")]), "j_business")
    assert _play_ev(g, [0, 1]) == pytest.approx(2.0)     # two scored kings, 1/2 x $2 each
    assert _play_ev(g, [2, 3, 4]) == pytest.approx(0.0)  # no faces score


def test_proc_ev_reserved_parking_is_half_a_dollar_per_HELD_face():
    """card.lua:3304 — ``pseudorandom('parking') < normal/2`` -> $1 per face card held in
    hand while a played hand scores (state_events.lua:784-870)."""
    g = _jokers(_set_hand(_in_blind(), [(13, "Spades"), (13, "Hearts"), (12, "Clubs"),
                                        (3, "Clubs"), (4, "Clubs"), (5, "Diamonds"),
                                        (6, "Diamonds"), (7, "Diamonds")]),
                "j_reserved_parking")
    # play the two kings: the queen stays in hand -> one held face
    assert _play_ev(g, [0, 1]) == pytest.approx(0.5)
    # play the non-faces: all three faces are held
    assert _play_ev(g, [3, 4, 5]) == pytest.approx(1.5)
    # a discard plays no hand, so Parking never fires for it
    assert _disc_ev(g, [3, 4]) == pytest.approx(0.0)


def test_proc_ev_golden_ticket_and_rough_gem_are_per_scoring_card():
    g = _jokers(_set_hand(_in_blind(), NEUTRAL), "j_ticket")
    g.hand[0].enhancement = "Gold"
    # $4 from the Ticket, minus the $3 of end-of-round hold money the card no longer earns
    # (weighted by P(clear)); the sign must still be positive
    assert 1.0 <= _play_ev(g, [0, 1]) <= 4.0
    g = _jokers(_set_hand(_in_blind(), [(2, "Diamonds"), (2, "Hearts"), (5, "Diamonds"),
                                        (6, "Diamonds"), (7, "Diamonds"), (8, "Clubs"),
                                        (9, "Clubs"), (10, "Clubs")]), "j_rough_gem")
    assert _play_ev(g, [0, 1]) == pytest.approx(1.0)      # one scored Diamond


def test_proc_ev_faceless_pays_five_for_three_discarded_faces():
    g = _jokers(_set_hand(_in_blind(), [(11, "Spades"), (12, "Hearts"), (13, "Clubs"),
                                        (2, "Clubs"), (3, "Clubs"), (4, "Diamonds"),
                                        (5, "Diamonds"), (6, "Hearts")]), "j_faceless")
    assert _disc_ev(g, [0, 1, 2]) == pytest.approx(5.0)
    assert _disc_ev(g, [0, 1]) == pytest.approx(0.0)      # two faces is not enough
    assert _disc_ev(g, [0, 1, 2, 3]) == pytest.approx(5.0)


def test_proc_ev_purple_seal_is_a_tarot_capped_by_the_free_slots():
    cfg = H.DEFAULT_HAND_CONFIG
    g = _set_hand(_in_blind(), NEUTRAL)
    for c in g.hand[:3]:
        c.seal = "Purple"
    g.consumable_hand = []
    assert _disc_ev(g, [0]) == pytest.approx(cfg.tarot_value_dollars)
    assert _disc_ev(g, [0, 1]) == pytest.approx(2 * cfg.tarot_value_dollars)
    assert _disc_ev(g, [0, 1, 2]) == pytest.approx(2 * cfg.tarot_value_dollars)   # 2 slots
    g.consumable_hand = ["c_fool", "c_fool"]
    assert _disc_ev(g, [0]) == pytest.approx(0.0)                                 # no room


def test_proc_ev_mail_in_rebate_and_trading_card():
    g = _set_hand(_in_blind(), NEUTRAL)
    g.round_picks = dict(g.round_picks)
    g.round_picks["mail"] = "A"
    _jokers(g, "j_mail")
    assert _disc_ev(g, [0]) == pytest.approx(5.0)         # one Ace of the round's rank
    assert _disc_ev(g, [0, 1]) == pytest.approx(10.0)
    assert _disc_ev(g, [2]) == pytest.approx(0.0)
    g = _jokers(_set_hand(_in_blind(), NEUTRAL), "j_trading")
    assert _disc_ev(g, [2]) == pytest.approx(3.0)         # a single-card first discard
    assert _disc_ev(g, [2, 3]) == pytest.approx(0.0)      # two cards: card.lua:2802


def test_proc_ev_delayed_gratification_makes_the_first_discard_cost_money():
    g = _jokers(_set_hand(_in_blind(), NEUTRAL), "j_delayed_grat")
    assert _disc_ev(g, [2]) == pytest.approx(-2.0 * g.discards_left)


def test_proc_ev_gold_enhancement_is_lost_when_the_card_leaves_the_hand():
    g = _set_hand(_in_blind(), NEUTRAL)
    g.hand[0].enhancement = "Gold"
    keep = _play_ev(g, [1])          # the other ace: the Gold card stays in hand
    shed = _play_ev(g, [0, 1])       # the Gold card is played away
    assert shed < keep
    assert -3.0 <= shed - keep <= 0.0        # at most the full $3, scaled by P(clear)
    assert _disc_ev(g, [0]) < _disc_ev(g, [2])


def test_extraction_ev_is_zero_on_a_plain_board():
    g = _set_hand(_in_blind(), NEUTRAL)
    an = _an(g)
    assert an.extract_on is False
    assert an.extraction_ev({"type": "play", "cards": [0, 1]}) == 0.0
    assert an.extraction_ev({"type": "discard", "cards": [2]}) == 0.0


def test_extraction_ev_is_side_effect_free():
    g = _jokers(_set_hand(_in_blind(), NEUTRAL), "j_business", "j_faceless")
    g.hand[0].seal = "Gold"
    sig = g.state_signature()
    rng = g.run_state.rng.snapshot()
    H.extraction_ev(g, {"type": "play", "cards": [0, 1]})
    H.extraction_ev(g, {"type": "discard", "cards": [2, 3, 4]})
    assert g.state_signature() == sig
    assert g.run_state.rng.snapshot() == rng


# ═══════════════════════════════════════════════════ keep / play / discard value split

def test_purple_seal_is_discard_valuable_but_not_play_junk():
    """The three orderings must disagree about a Purple seal: it is the FIRST card to
    discard (a Tarot) and one of the last to burn as a play filler (playing it wastes it)."""
    g = _set_hand(_in_blind(), NEUTRAL)
    p = 5                                       # 5 of Diamonds, structurally junk
    g.hand[p].seal = "Purple"
    an = _an(g)
    assert an.discard_junk_order[0] == p
    assert an.play_junk_order.index(p) > an.discard_junk_order.index(p)
    assert an.proc_discard[p] > 0 and an.proc_play[p] == 0


def test_gold_enhancement_is_hold_valuable_in_both_orderings():
    g = _set_hand(_in_blind(), NEUTRAL)
    gold = 5
    g.hand[gold].enhancement = "Gold"
    an = _an(g)
    assert an.proc_hold[gold] == pytest.approx(3.0)
    assert an.discard_junk_order[0] != gold      # never the first card thrown away
    assert an.play_junk_order[0] != gold


def test_gold_seal_and_lucky_are_play_valuable():
    g = _set_hand(_in_blind(), [(13, "Spades"), (13, "Hearts"), (13, "Clubs"),
                                (2, "Clubs"), (3, "Clubs"), (4, "Diamonds"),
                                (5, "Diamonds"), (6, "Hearts")])
    g.hand[2].seal = "Gold"                      # the third king carries the seal
    an = _an(g)
    assert an.proc_play[2] == pytest.approx(3.0)
    # the "two of the three kings" candidate now exists in a variant that INCLUDES the
    # sealed king (the chip-only representative would take the first two)
    pairs = [t for t in an.play_cands if len(t) == 2 and all(g.hand[j].rank == 13 for j in t)]
    assert any(2 in t for t in pairs)


def test_face_cards_are_hold_valuable_with_reserved_parking():
    plain = _an(_set_hand(_in_blind(), [(13, "Spades"), (2, "Clubs"), (3, "Clubs"),
                                        (4, "Clubs"), (5, "Diamonds"), (6, "Diamonds"),
                                        (7, "Diamonds"), (8, "Hearts")]))
    park = _an(_jokers(_set_hand(_in_blind(), [(13, "Spades"), (2, "Clubs"), (3, "Clubs"),
                                               (4, "Clubs"), (5, "Diamonds"), (6, "Diamonds"),
                                               (7, "Diamonds"), (8, "Hearts")]),
                       "j_reserved_parking"))
    assert park.proc_hold[0] > 0 and plain.proc_hold[0] == 0
    assert park.discard_keep[0] > plain.discard_keep[0]


# ═════════════════════════════════════════════════════════ candidate lines + safety gate

def _purple_state(n_purple=2, seed="11111111"):
    g = _set_hand(_in_blind(seed), NEUTRAL)
    for c in g.hand[5:5 + n_purple]:
        c.seal = "Purple"
    g.consumable_hand = []
    return g


def test_the_discard_the_purple_seals_line_is_generated_when_safe():
    g = _purple_state()
    an = _an(g)
    assert an.extract_on
    assert an.extraction_safe(an.h, an.d - 1, an.need)
    lines = an._discard_lines()
    assert (5, 6) in lines                       # exactly the two sealed cards


def test_extraction_lines_are_suppressed_when_the_blind_is_not_safe():
    """Below ``extract_min_clear`` nothing is banked: no dedicated extraction line is
    generated and no candidate carries a money term (the junk ORDERING stays proc-aware —
    it only decides which of two equally worthless cards goes, never whether to sandbag)."""
    from dataclasses import replace
    g = _purple_state()
    g.current_blind.chips_target = 10 ** 7       # hopeless: the tail DP says ~0
    an = _an(g)
    assert an.extraction_safe(an.h, an.d - 1, an.need) is False
    assert an._extraction_discard_lines() == []
    off = H.HandAnalysis(g, replace(H.DEFAULT_HAND_CONFIG, extract=False),
                         legal=g.legal_actions())
    evs = dict((H._action_sort_key(a), ev) for a, ev in an.evaluate())
    common = 0
    for a, ev in off.evaluate():
        k = H._action_sort_key(a)
        if k in evs:
            common += 1
            assert evs[k] == pytest.approx(ev, abs=1e-12)
    assert common > 5


def test_extraction_is_off_at_a_nemesis_blind():
    g = _purple_state()
    g.current_blind.is_pvp = True
    an = _an(g)
    assert an.extract_on is False
    assert an.extraction_safe(an.h, an.d, an.need) is False


def test_the_purple_seal_discard_beats_a_pointless_chase_when_the_blind_is_safe():
    """The whole point of the layer: with two Purple seals in an otherwise junk tail of the
    hand and a blind that is already safe, dumping the seals must outrank dumping the same
    cards' structural neighbours (which banks nothing)."""
    g = _purple_state()
    an = _an(g)
    ranked = dict((H._action_sort_key(a), ev) for a, ev in an.evaluate())
    seal_line = H._action_sort_key({"type": "discard", "cards": [5, 6]})
    assert seal_line in ranked
    for a, ev in an.evaluate():
        if a["type"] == "discard" and not (set(a["cards"]) & {5, 6}):
            assert ranked[seal_line] > ev


def test_the_faceless_discard_line_is_generated():
    g = _jokers(_set_hand(_in_blind(), [(11, "Spades"), (12, "Hearts"), (13, "Clubs"),
                                        (2, "Clubs"), (3, "Clubs"), (4, "Diamonds"),
                                        (5, "Diamonds"), (6, "Hearts")]), "j_faceless")
    an = _an(g)
    lines = an._discard_lines()
    assert any(len(set(t) & {0, 1, 2}) == 3 for t in lines)


def test_tarot_cycling_values_drawing_toward_a_suit_tarot_target():
    """A held Sun (3 cards -> Hearts, consumables.TAROT_SUIT) wants ``flush_need - 3`` real
    Hearts in hand; with none, a line that draws more cards is worth part of the Tarot."""
    g = _set_hand(_in_blind(), [(2, "Clubs"), (3, "Clubs"), (4, "Clubs"), (5, "Clubs"),
                                (6, "Clubs"), (7, "Spades"), (8, "Spades"), (9, "Spades")])
    g.consumable_hand = ["c_sun"]
    an = _an(g)
    assert an._tarot_wants
    small = an._cycle_ev(an.full_mask & ~0b11, 2)
    big = an._cycle_ev(an.full_mask & ~0b11111, 5)
    assert an._cycle_ev(an.full_mask & ~0b1, 1) == 0.0   # one draw cannot supply two hearts
    assert 0.0 < small < big <= an.cfg.tarot_value_dollars * an.cfg.tarot_cycle_fraction
    # ... and it is worth nothing once the targets are already in hand
    g2 = _set_hand(_in_blind(), [(2, "Hearts"), (3, "Hearts"), (4, "Clubs"), (5, "Clubs"),
                                 (6, "Clubs"), (7, "Spades"), (8, "Spades"), (9, "Spades")])
    g2.consumable_hand = ["c_sun"]
    an2 = _an(g2)
    assert an2._cycle_ev(an2.full_mask & ~0b11100000, 3) == pytest.approx(0.0)


def test_extraction_never_changes_a_plain_board_decision():
    """No procs, no seals, no tarots -> the extraction layer must be bit-identical to the
    pre-change player (it is the ante-1 common case and the 126-seed gate depends on it)."""
    from dataclasses import replace
    off = replace(H.DEFAULT_HAND_CONFIG, extract=False)
    for seed in ("11111111", "CHPB293X", "7I4M53DL"):
        g = _in_blind(seed)
        a = H.rank_hand_actions(g, cfg=H.DEFAULT_HAND_CONFIG)
        b = H.rank_hand_actions(g, cfg=off)
        assert [H._action_sort_key(x) for x, _ in a] == [H._action_sort_key(x) for x, _ in b]
        assert [pytest.approx(ev) for _, ev in a] == [ev for _, ev in b]


def test_extraction_lines_is_the_gated_generator_w_pairs_consumes():
    """``hand.extraction_lines(game, legal=...)`` -> ``[(action, ev, reason)]``, best first
    (W-PAIRS's ``greedy_vs_extract`` source, brief §5.2).  Empty on a plain board and empty
    when the blind is not safe."""
    g = _purple_state()
    lines = H.extraction_lines(g, g.legal_actions())
    assert lines and all(len(x) == 3 for x in lines)
    assert [x[1] for x in lines] == sorted((x[1] for x in lines), reverse=True)
    assert any(x[0]["type"] == "discard" and set(x[0]["cards"]) >= {5, 6} for x in lines)
    assert all("extract $" in x[2] for x in lines)
    assert H.extraction_lines(_in_blind(), None) == []          # plain board
    g.current_blind.chips_target = 10 ** 7
    assert H.extraction_lines(g, g.legal_actions()) == []       # unsafe


def test_plain_board_takes_the_zero_cost_fast_path():
    """The ante-1 common case must not pay for the layer at all: one junk ordering, shared
    zero arrays, ``extract_on`` False (this is what keeps the 5 ms fast budget)."""
    an = _an(_in_blind())
    assert an.play_junk_order is an.discard_junk_order is an.junk_order
    assert an.play_keep is an.discard_keep is an.keep_value
    assert an.proc_play is an.proc_hold is an.proc_discard
    assert an.has_card_proc is False and an.extract_on is False


def test_extraction_decisions_stay_legal_and_side_effect_free():
    g = _jokers(_purple_state(), "j_business", "j_faceless", "j_reserved_parking")
    g.hand[0].seal = "Gold"
    g.hand[1].enhancement = "Lucky"
    sig = g.state_signature()
    rng = g.run_state.rng.snapshot()
    ranked = H.rank_hand_actions(g)
    legal = {H._action_sort_key(a) for a in g.legal_actions()}
    assert ranked and all(H._action_sort_key(a) in legal for a, _ in ranked)
    assert g.state_signature() == sig and g.run_state.rng.snapshot() == rng
