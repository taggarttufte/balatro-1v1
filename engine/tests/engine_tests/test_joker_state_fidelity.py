"""test_joker_state_fidelity.py — the STATE half of joker fidelity (W-FIX, 2026-08-26).

Scoring fidelity ("does this joker add the right chips") is covered elsewhere.  This
module covers the three things W-ENCODE-POC found the engine getting wrong about joker
*state*, all of them invisible to a single-joker single-hand test:

1. **What a Blueprint / Brainstorm copy may change.**  card.lua gates 26 joker branches on
   ``and not context.blueprint``: a copy reproduces the SCORE contribution and never the
   self state change.  The engine used to call the target's hook on the target's own
   instance with no flag, so every self-mutating scaling joker advanced twice per hand.
2. **What run-global state a joker reads.**  Satellite reads ``G.GAME.consumeable_usage``
   (card.lua:1667-1673), not a per-joker set, so it pays for planets used before it was
   bought.
3. **The two jokers that consume themselves.**  Ice Cream is DESTROYED when its chips
   would reach zero (card.lua:3571-3592) — it does not clamp at 0 and squat in a slot.

Every test drives the real ``BalatroGame`` through ``step`` and cites ``file:line`` in the
Lua (``_reference/balatro_src/``, read-only).  See ``engine/FIX_NOTES.md``.
"""
from __future__ import annotations

import pytest

from balatro_sim.game import BalatroGame, State
from balatro_sim.consumables import apply_planet
from balatro_sim.jokers.base import JOKER_REGISTRY, JokerInstance, ScoreContext


_HAND = [(14, "Spades"), (2, "Clubs"), (3, "Clubs"), (4, "Clubs"),
         (5, "Diamonds"), (6, "Diamonds"), (7, "Diamonds"), (8, "Hearts")]


def _in_blind(seed: str = "11111111", *, hand=_HAND, no_interest: bool = True) -> BalatroGame:
    g = BalatroGame(seed=seed, deck_key="b_red", stake=1, ruleset="vanilla")
    g.step({"type": "play_blind"})
    assert g.state == State.SELECTING_HAND
    pool: dict = {}
    for c in g.full_deck:
        pool.setdefault((c.rank, c.suit), c)
    cards = [pool[s] for s in hand]
    ids = {id(c) for c in cards}
    g.deck = [c for c in g.full_deck if id(c) not in ids]
    g.hand = cards
    g.discard_pile = []
    for c in cards:
        c.face_down = False
        c.debuffed = False
    g.jokers = []
    g.no_interest = no_interest
    return g


def _j(g, key: str) -> JokerInstance:
    return g.debug_add_joker(key)


def _of(g, key: str):
    return next((j for j in g.jokers if j.key == key), None)


def _play(g, cards=(0,)):
    g.step({"type": "play", "cards": list(cards)})


# ══════════════════════════════════════ 1. what a Blueprint / Brainstorm copy may change
#
# Board shapes: Blueprint copies the joker immediately to its RIGHT (card.lua:2306-2318),
# Brainstorm the LEFTMOST joker (:2320-2332).  So [blueprint, X] and [X, ..., brainstorm]
# both put a second trigger of X's effect on the board.

#: (joker key, state key, a starting state, the value after ONE hand with a copy present)
#: — each row is the Lua's `context.before` / `individual` branch advancing exactly once.
_SELF_MUTATING = [
    # key,                 state field, start,       after one hand, lua
    ("j_green_joker",      "mult",      {"mult": 3},  4,   "card.lua:3563"),
    ("j_ride_the_bus",     "mult",      {"mult": 2},  3,   "card.lua:3525"),
    ("j_ice_cream",        "chips",     {"chips": 60}, 55, "card.lua:3571"),
    ("j_trousers",         "mult",      {"mult": 0},  0,   "card.lua:3412"),  # no Two Pair here
    ("j_runner",           "chips",     {"chips": 7}, 7,   "card.lua:3435"),  # no Straight here
    ("j_selzer",           "hands",     {"hands": 6}, 5,   "card.lua:3601"),
    ("j_loyalty_card",     "count",     {"count": 2}, 3,   "card.lua:3632-3648"),
]


@pytest.mark.parametrize("key,field,start,want,lua", _SELF_MUTATING,
                         ids=[r[0] for r in _SELF_MUTATING])
def test_a_blueprint_copy_does_not_advance_the_joker_it_copies(key, field, start, want, lua):
    """``and not context.blueprint`` ({lua}): the copy pays the joker's score row and
    leaves its state alone.  Before W-FIX the engine re-ran the whole hook on the target's
    own instance, so the state advanced twice per hand — Ice Cream melted at 10 chips a
    hand instead of 5 (W-ENCODE-POC §3.2, found by measurement on seed 7YTVQERM)."""
    g = _in_blind()
    _j(g, "j_blueprint")                 # leftmost -> copies the joker to its right
    j = _j(g, key)
    j.state.update(start)
    _play(g)
    assert j.state.get(field) == want, f"{key}.{field} advanced more than once ({lua})"


@pytest.mark.parametrize("key,field,start,want,lua", _SELF_MUTATING,
                         ids=[r[0] for r in _SELF_MUTATING])
def test_a_brainstorm_copy_does_not_advance_the_joker_it_copies(key, field, start, want, lua):
    """The same guard through the other copier (card.lua:2320-2332 copies ``G.jokers[1]``)."""
    g = _in_blind()
    j = _j(g, key)                       # leftmost -> what Brainstorm copies
    j.state.update(start)
    _j(g, "j_brainstorm")
    _play(g)
    assert j.state.get(field) == want, f"{key}.{field} advanced more than once ({lua})"


def test_a_copy_still_pays_the_score_row_it_copies():
    """The guard must suppress the MUTATION only.  A Blueprint next to Green Joker scores
    the mult twice (card.lua:4010-4015 is outside the ``not context.blueprint`` branch)
    while the counter rises once — that is the whole point of the fix, and a guard that
    also swallowed the payout would pass the tests above and be wrong."""
    g = _in_blind()
    j = _j(g, "j_green_joker")
    j.state["mult"] = 6
    solo = _score_of(g, [0])

    g2 = _in_blind()
    _j(g2, "j_blueprint")
    j2 = _j(g2, "j_green_joker")
    j2.state["mult"] = 6
    doubled = _score_of(g2, [0])
    assert doubled > solo, "the copy must still contribute its mult row"


def _score_of(g, cards) -> int:
    before = g.chips_scored
    _play(g, cards)
    return g.chips_scored - before


def test_the_before_pass_runs_before_every_joker_main_row():
    """card.lua raises Green Joker's counter in ``context.before`` (:3563) and pays it in
    ``joker_main`` (:4010-4015) — two separate passes over the whole board
    (state_events.lua:630, then :876).  So a Blueprint copying it pays the RAISED figure:
    mult 6 -> 7 in the before pass, then TWO rows of +7, not +6 then +7.

    Exact arithmetic on a High Card Ace (base 5 chips / 1 mult, Ace 11 chips = 16 chips):

    ======================================  =========================  =====
    engine                                  mult rows                  score
    ======================================  =========================  =====
    Lua (and this engine after W-FIX)       1 + 7 + 7 = 15             240
    before W-FIX (copy re-ran the gain)     1 + 7 + 8 = 16             256
    a guard applied in place, no before     1 + 6 + 7 = 14             224
    ======================================  =========================  =====

    That is why the fix moved the gain into ``pre_score`` rather than only guarding it."""
    from balatro_sim.card import Card
    from balatro_sim.game_keys import core as _core
    from balatro_sim.scoring import score_hand

    bp, green = JokerInstance("j_blueprint"), JokerInstance("j_green_joker")
    green.state["mult"] = 6
    ace = Card(rank=14, suit="Spades")
    score, _ = score_hand(
        scoring_cards=[ace], all_cards=[ace], hand_type="High Card",
        jokers=[bp, green], planet_levels={"High Card": 1}, hands_left=3,
        discards_left=3, dollars=10, ante=1, deck_remaining=44,
        rng=_core.PseudoRandom("TEST1"))
    assert green.state["mult"] == 7, "the counter advances exactly once"
    assert score == 240, "the copy must pay the POST-gain mult, like the original"


def test_the_blueprint_flag_is_restored_after_the_copy():
    """``ctx.blueprint`` is a depth counter restored in a ``finally`` (card.lua:2310:
    ``context.blueprint = (context.blueprint and (context.blueprint + 1)) or 1``).  If it
    leaked, every joker to the copier's right would stop mutating."""
    g = _in_blind()
    _j(g, "j_blueprint")
    _j(g, "j_green_joker")        # what the Blueprint copies
    tail = _j(g, "j_ride_the_bus")  # to the RIGHT of both: must be unaffected
    tail.state["mult"] = 4
    _play(g)
    assert tail.state["mult"] == 5, "a joker after the copy must still advance"


def test_the_context_carries_no_blueprint_flag_outside_a_copy():
    g = _in_blind()
    ctx = g._hook_ctx()
    assert ctx.blueprint == 0
    assert ScoreContext().blueprint == 0


def test_a_copy_does_not_mutate_the_CARDS_its_target_would_mutate():
    """Two jokers change the played cards themselves rather than their own state, and both
    carry the guard in the Lua: Vampire strips enhancements (card.lua:3465) and Midas Mask
    gilds face cards (:3443).  A copy pays their score row and touches no card."""
    from balatro_sim.card import Card
    for key, hook_kw in (("j_vampire", {"enhancement": "Gold"}),
                         ("j_midas_mask", {})):
        effect = JOKER_REGISTRY[key]
        inst = JokerInstance(key)
        card = Card(13, "Spades")            # a King: a face card, gildable
        for k, v in hook_kw.items():
            setattr(card, k, v)
        before = card.enhancement
        ctx = ScoreContext(scoring_cards=[card], all_cards=[card], jokers=[inst], blueprint=1)
        effect.on_score_card(inst, card, ctx)
        assert card.enhancement == before, f"{key} mutated a card while copied"


def test_hiker_is_deliberately_not_guarded():
    """The guard list is the Lua's, not a rule of thumb.  Hiker's ``perma_bonus`` bump
    (card.lua:3067) has NO ``not context.blueprint``, so a copy really does add another
    +5 per scoring card — and the engine must keep doing that."""
    g = _in_blind()
    _j(g, "j_blueprint")
    _j(g, "j_hiker")
    card = g.hand[0]
    before = getattr(card, "bonus_chips", 0)
    _play(g, [0])
    assert getattr(card, "bonus_chips", 0) == before + 10, "Hiker is copied in full"


# ═══════════════════════════════════════════ 2. run-global state: Satellite's planet tally

def _round_end_delta(build, install) -> int:
    """Marginal end-of-round dollars of ``install`` — two arms from the same builder."""
    a, b = build(), build()
    install(a)
    d0, db = a.dollars, b.dollars
    a._end_round()
    b._end_round()
    return (a.dollars - d0) - (b.dollars - db)


@pytest.mark.parametrize("keys,want", [
    ((), 0),
    (("c_mercury",), 1),
    (("c_mercury", "c_venus", "c_mars"), 3),
    (("c_mercury", "c_mercury", "c_venus"), 2),        # DISTINCT keys, not uses
])
def test_satellite_pays_for_planets_used_before_it_was_bought(keys, want):
    """card.lua:1667-1673 counts distinct ``G.GAME.consumeable_usage`` entries with
    ``set == 'Planet'`` — RUN-GLOBAL state written by ``Card:use_consumeable``
    (card.lua:1093 -> misc_functions.lua:1184-1195), so purchase timing is irrelevant and
    a Satellite bought at ante 4 after three planets pays $3 on its first round end.

    The engine used to keep the set on the joker instance and seed it only from an
    ``on_planet_used`` hook, so it paid $0 for everything used before the purchase — 100%
    of the value of a joker that is essentially always bought mid-run (W-ENCODE-POC §3.1)."""
    def build():
        g = _in_blind()
        for k in keys:
            assert apply_planet(g, k)
        return g
    assert _round_end_delta(build, lambda g: _j(g, "j_satellite")) == want


@pytest.mark.parametrize("keys,want", [
    ((), 0),
    (("c_mercury",), 1),
    (("c_mercury", "c_venus", "c_mars"), 3),
])
def test_satellite_still_pays_for_planets_used_after_it_was_bought(keys, want):
    """The half the engine already had right must not regress."""
    def build():
        return _in_blind()

    def install(g):
        _j(g, "j_satellite")
        for k in keys:
            assert apply_planet(g, k)
    # the WITHOUT arm has to use the planets too, or the comparison prices the planets
    a, b = build(), build()
    install(a)
    for k in keys:
        assert apply_planet(b, k)
    d0, db = a.dollars, b.dollars
    a._end_round()
    b._end_round()
    assert (a.dollars - d0) - (b.dollars - db) == want


def test_satellite_ignores_level_ups_that_are_not_planet_uses():
    """``set_consumeable_usage`` runs from ``Card:use_consumeable`` only (card.lua:1093).
    Black Hole is a SPECTRAL (``set ~= 'Planet'``), and Space Joker / Burnt Joker / the
    Orbital tag call ``level_up_hand`` directly — none of them is a planet use, so none of
    them pays Satellite."""
    from balatro_sim.consumables import apply_spectral
    def build():
        g = _in_blind()
        assert apply_spectral(g, "c_black_hole")
        g.planet_levels["Flush"] = g.planet_levels.get("Flush", 1) + 3   # a tag/Space Joker
        return g
    assert _round_end_delta(build, lambda g: _j(g, "j_satellite")) == 0


def test_satellites_planet_tally_survives_a_clone():
    """``game.planets_used`` is copied by ``BalatroGame.clone`` (game.py:598) and is part
    of ``state_signature`` (:1009), so an MCTS twin / a determinized world sees the same
    Satellite payout as its parent."""
    g = _in_blind()
    for k in ("c_mercury", "c_venus"):
        assert apply_planet(g, k)
    _j(g, "j_satellite")
    c = g.clone()
    assert set(c.planets_used) == {"c_mercury", "c_venus"}
    d0 = c.dollars
    c._end_round()
    base = _in_blind()
    for k in ("c_mercury", "c_venus"):
        apply_planet(base, k)
    db = base.dollars
    base._end_round()
    assert (c.dollars - d0) - (base.dollars - db) == 2


def test_a_clone_does_not_share_the_planet_list_with_its_parent():
    g = _in_blind()
    apply_planet(g, "c_mercury")
    c = g.clone()
    apply_planet(c, "c_venus")
    assert set(g.planets_used) == {"c_mercury"}


# ══════════════════════════════════════════════════ 3. the jokers that consume themselves

def test_ice_cream_melts_when_its_chips_would_reach_zero():
    """``if self.ability.extra.chips - self.ability.extra.chip_mod <= 0 then ...
    G.jokers:remove_card(self); self:remove()`` (card.lua:3571-3592).  The card is
    DESTROYED on the hand that would take it to zero — the engine used to floor it at
    ``max(0, chips - 5)`` and leave a dead 0-chip joker in the slot forever."""
    g = _in_blind()
    j = _j(g, "j_ice_cream")
    j.state["chips"] = 5                       # one hand from melting
    slots, n = g.joker_slots, len(g.jokers)
    _play(g)
    assert _of(g, "j_ice_cream") is None, "Ice Cream must be destroyed, not clamped"
    assert len(g.jokers) == n - 1
    assert g.joker_slots == slots, "the slot is freed, not consumed"


def test_the_melting_hand_still_scores_the_last_chips():
    """joker_main pays ``extra.chips`` (card.lua:3915-3920) and the melt is in the LATER
    ``context.after`` pass (:3570, state_events.lua:1070), so the final hand scores 5."""
    g = _in_blind()
    j = _j(g, "j_ice_cream")
    j.state["chips"] = 5
    with_ice = _score_of(g, [0])
    g2 = _in_blind()
    without = _score_of(g2, [0])
    assert with_ice > without


def test_ice_cream_survives_exactly_twenty_hands():
    """100 chips at 5 a hand: 20 scoring hands (100, 95, ..., 5) and then a free slot."""
    g = _in_blind()
    j = _j(g, "j_ice_cream")
    assert j.state["chips"] == 100
    hands = 0
    for _ in range(30):
        if _of(g, "j_ice_cream") is None:
            break
        if g.state != State.SELECTING_HAND:      # re-enter a blind
            g = _re_enter(g, j)
        _play(g)
        hands += 1
    assert hands == 20, f"melted after {hands} hands, want 20"
    assert _of(g, "j_ice_cream") is None


def _re_enter(g, keep: JokerInstance):
    """A fresh blind carrying one joker instance across — the melt takes more hands than
    a single blind allows and the countdown is per HAND, not per round."""
    new = _in_blind()
    new.jokers = [keep]
    return new


def test_the_slot_ice_cream_frees_can_be_filled():
    g = _in_blind()
    for k in ("j_joker", "j_sly", "j_wily", "j_crafty"):
        _j(g, k)
    j = _j(g, "j_ice_cream")
    j.state["chips"] = 5
    assert len(g.jokers) == g.joker_slots        # full board
    _play(g)
    assert len(g.jokers) == g.joker_slots - 1
    assert _j(g, "j_devious") is not None and len(g.jokers) == g.joker_slots


def test_a_copy_cannot_melt_ice_cream_early():
    """``not context.blueprint`` on the ``context.after`` branch (card.lua:3571): a
    Brainstorm next to a 5-chip Ice Cream must not destroy it twice as fast."""
    g = _in_blind()
    j = _j(g, "j_ice_cream")
    j.state["chips"] = 10
    _j(g, "j_brainstorm")
    _play(g)
    assert _of(g, "j_ice_cream") is not None
    assert j.state["chips"] == 5, "one decrement, not two"


def test_seltzer_still_self_destructs_from_the_after_pass():
    """Seltzer's countdown moved to the same ``context.after`` pass (card.lua:3601) — a
    regression check that the move did not lose the destruction."""
    g = _in_blind()
    j = _j(g, "j_selzer")
    j.state["hands"] = 1
    _play(g)
    assert _of(g, "j_selzer") is None


# ══════════════════════════════════════════════════════════ 4. the guard list is complete

def test_every_hook_a_copy_dispatches_is_guarded_where_the_lua_guards_it():
    """A structural check on the engine, not on one joker: for every joker whose Lua
    branch carries ``and not context.blueprint`` AND whose engine implementation mutates
    ``inst.state`` inside a hook Blueprint dispatches (``pre_score`` / ``on_score_card`` /
    ``on_held_card`` / ``on_hand_scored``), calling that hook with ``ctx.blueprint`` set
    must leave the state untouched.

    This is the test that catches the NEXT joker somebody adds without the guard."""
    from balatro_sim.card import Card
    guarded = ["j_green_joker", "j_ride_the_bus", "j_obelisk", "j_ice_cream", "j_trousers",
               "j_runner", "j_square", "j_vampire", "j_wee", "j_lucky_cat", "j_selzer",
               "j_card_sharp", "j_burnt", "j_loyalty_card"]
    dispatched = ("pre_score", "on_score_card", "on_held_card", "on_hand_scored",
                  "on_hand_after")
    for key in guarded:
        effect = JOKER_REGISTRY[key]
        for hook in dispatched:
            fn = getattr(effect, hook, None)
            if fn is None:
                continue
            inst = JokerInstance(key)
            init = getattr(effect, "on_init", None)
            ctx0 = ScoreContext()
            if init:
                init(inst, ctx0)
            inst.state.setdefault("xmult", 1.0)
            before = {k: (v.copy() if isinstance(v, (dict, set, list)) else v)
                      for k, v in inst.state.items()}
            cards = [Card(2, "Hearts"), Card(2, "Spades"), Card(3, "Clubs"), Card(4, "Clubs")]
            for c in cards:
                c.enhancement = "Gold"
            ctx = ScoreContext(hand_type="Two Pair", scoring_cards=cards, all_cards=cards,
                               held_cards=cards, jokers=[inst], blueprint=1,
                               lucky_trigger=True)
            if hook == "on_score_card":
                fn(inst, cards[0], ctx)
            elif hook == "on_held_card":
                fn(inst, cards[0], ctx)
            else:
                fn(inst, ctx)
            assert inst.state == before, (
                f"{key}.{hook} mutated its own state while ctx.blueprint was set")
