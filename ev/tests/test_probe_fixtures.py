"""
test_probe_fixtures.py -- W-PROBE (Phase 5 rev 2, PHASE5_V2_BRIEF_2026-08.md section 7):
acceptance fixtures for Tagg's six sandbag scenarios, each with a matched "greedy is right"
control, registered in ``fixtures.FIXTURES`` and rendered by ``advisor.advise`` /
``cli.py advise fixture:<name>``.

Every fixture is built through real ``MLBMatch`` / ``BalatroGame`` APIs (see
``fixtures/_probe_common.py``); this file never hand-forges a legal-actions dict.  The
per-fixture ordering assertions read the SAME ``hand.rank_hand_actions`` ranking the fast
(``ev:fast``) advisor uses -- exactly what ``EVPlayer.explain`` and hence the advisor CLI's
"Ranked actions" table print.

Own only: this file, ``ev/fixtures/{purple_seal_discard,faceless_discard,
business_card_board,reserved_parking_hold,gold_seal_weak_play,tarot_target_cycle,
_probe_common}.py``, the additive registration in ``fixtures/__init__.py``, and
``ev/PROBE_NOTES.md``.  ``hand.py`` / ``player.py`` / ``pairs.py`` are read-only here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

import _bootstrap  # noqa: F401
from _bootstrap import State

import fixtures
# W-CYCLE: the tarot fixture's own pinned lines (the package rebinds the NAME
# `fixtures.tarot_target_cycle` to its `build`, so the module has to be reached directly)
from fixtures.tarot_target_cycle import CLEAR_NOW as TC_CLEAR_NOW
from fixtures.tarot_target_cycle import DIG_LINE as TC_DIG_LINE
from fixtures.tarot_target_cycle import CHIPS_TARGET as TC_CHIPS_TARGET
from fixtures.tarot_target_cycle import build_low_grade_target as _tc_low
import hand as H
import player as P
import advisor

_MP_ROOT = str(Path(_bootstrap.MP_ROOT))
if _MP_ROOT not in sys.path:
    sys.path.insert(0, _MP_ROOT)


# Six (sandbag, control) pairs; in every one of them the extraction/dig action and the
# "clear now" action are genuinely DIFFERENT candidates (a real rank swap between fixture
# and control).  tarot_target_cycle used to be the exception -- it was pinned at the weaker
# "same action, more EV" claim because W-EXTRACT's per-COUNT `_cycle_ev` scored every line
# that drew the same number of cards identically (PROBE_NOTES.md section 3.3).  W-CYCLE's
# per-TARGET model (ev/CYCLE_NOTES.md) produces the swap, so it joins the table.
SCENARIOS = [
    ("purple_seal_discard", ("discard", (5, 6)), ("play", (0, 1))),
    ("faceless_discard", ("discard", (2, 3, 4)), ("play", (0, 1))),
    ("business_card_board", ("play", (2,)), ("play", (0, 1))),
    ("reserved_parking_hold", ("play", (7,)), ("play", (0, 1))),
    ("gold_seal_weak_play", ("play", (4,)), ("play", (0, 1))),
    ("tarot_target_cycle", ("discard", (6, 7)), ("play", (0, 1, 2, 3, 4))),
]
ALL_NAMES = [n for n, _, _ in SCENARIOS]


def _ranked_dict(game) -> dict:
    """``{action_sort_key: ev}`` for every candidate at the fast budget -- exactly what the
    advisor's rules-tier ``EVPlayer.explain`` ranks."""
    return {H._action_sort_key(a): ev for a, ev in H.rank_hand_actions(game)}


def _key(kind: str, cards: tuple) -> tuple:
    return H._action_sort_key({"type": kind, "cards": list(cards)})


# ─────────────────────────────────────────────────────────── shape / determinism / registry

@pytest.mark.parametrize("name", ALL_NAMES)
def test_fixture_and_control_are_registered_and_deterministic(name):
    assert name in fixtures.FIXTURES
    assert f"{name}_control" in fixtures.FIXTURES
    for key in (name, f"{name}_control"):
        m1 = fixtures.FIXTURES[key]()
        m2 = fixtures.FIXTURES[key]()
        assert m1.signature() == m2.signature(), f"{key} is not deterministic"


@pytest.mark.parametrize("name", ALL_NAMES)
def test_fixture_and_control_land_at_a_hand_decision(name):
    for key in (name, f"{name}_control"):
        m = fixtures.FIXTURES[key]()
        g0 = m.games[0]
        assert g0.state == State.SELECTING_HAND
        assert g0.ante == 1
        assert g0.current_blind.is_pvp is False


@pytest.mark.parametrize("name", ALL_NAMES)
def test_fixture_decisions_are_side_effect_free_and_legal(name):
    """Matches test_extraction.py's own invariant: ranking a hand never mutates state or
    consumes the RNG stream, and every ranked action is actually legal."""
    for key in (name, f"{name}_control"):
        m = fixtures.FIXTURES[key]()
        g0 = m.games[0]
        sig = g0.state_signature()
        rng = g0.run_state.rng.snapshot()
        ranked = H.rank_hand_actions(g0)
        legal = {H._action_sort_key(a) for a in g0.legal_actions()}
        assert ranked
        assert all(H._action_sort_key(a) in legal for a, _ in ranked)
        assert g0.state_signature() == sig
        assert g0.run_state.rng.snapshot() == rng


# ─────────────────────────────────────────────────────────── the qualitative ordering pin

@pytest.mark.parametrize("name,extract_action,clear_action", SCENARIOS)
def test_sandbag_extraction_line_beats_clear_now(name, extract_action, clear_action):
    """The whole point of the layer (EXTRACT_NOTES.md section 4): in the sandbag fixture,
    the money-bearing line strictly outranks the immediate clear."""
    g0 = fixtures.FIXTURES[name]().games[0]
    evs = _ranked_dict(g0)
    ek, ck = _key(*extract_action), _key(*clear_action)
    assert ek in evs and ck in evs
    assert evs[ek] > evs[ck], (
        f"{name}: extraction line {extract_action} ({evs[ek]:.5f}) did not beat "
        f"clear-now {clear_action} ({evs[ck]:.5f})")
    # ... and it is literally the #1 ranked action (what the advisor prints first).
    ranked = H.rank_hand_actions(g0)
    assert H._action_sort_key(ranked[0][0]) == ek


@pytest.mark.parametrize("name,extract_action,clear_action", SCENARIOS)
def test_control_ordering_reverses(name, extract_action, clear_action):
    """Matched control (EXTRACT_NOTES.md's "procs absent" recipe): same hand shape, no proc
    -- clear-now wins outright and the extraction-flavoured action carries no bonus."""
    g0 = fixtures.FIXTURES[f"{name}_control"]().games[0]
    an = H.HandAnalysis(g0, H.DEFAULT_HAND_CONFIG, legal=g0.legal_actions())
    assert an.extract_on is False, f"{name}_control still has something to extract"
    assert H.extraction_lines(g0, g0.legal_actions()) == []
    evs = _ranked_dict(g0)
    ek, ck = _key(*extract_action), _key(*clear_action)
    assert ck in evs
    if ek in evs:
        # the exact extraction card-set is still a generated candidate (true for the three
        # PLAY-type scenarios, whose single-card representatives always exist per EV_NOTES
        # section 1's "every visible single") -- compare it directly.
        assert evs[ck] >= evs[ek], (
            f"{name}_control: clear-now {clear_action} ({evs[ck]:.5f}) should not rank below "
            f"the (now moneyless) {extract_action} ({evs[ek]:.5f})")
    else:
        # the two DISCARD-type scenarios: without the proc, the junk ordering that singled
        # out exactly those cards (EXTRACT_NOTES.md section 3's discard_keep) no longer
        # exists, so the discard generator does not even propose that exact card set -- the
        # honest, stronger check is that clear-now is the (tied-for-)best action overall.
        assert evs[ck] == pytest.approx(max(evs.values()), abs=1e-4), (
            f"{name}_control: clear-now {clear_action} ({evs[ck]:.5f}) is not the best "
            f"action ({max(evs.values()):.5f})")


def test_tarot_target_cycle_is_a_dig_line_not_a_uniform_bonus():
    """The rank swap of the upgraded fixture, spelled out (ev/CYCLE_NOTES.md section 4).

    ``discard [6,7]`` throws the two junk Spades, keeps the whole Clubs flush (so the floor
    still clears the blind next hand) AND keeps the lone Ace of Hearts the Sun is waiting
    on, and draws 2. It beats ``play [0..4]``, which clears the blind outright -- because a
    clearing play ENDS the round, and the cards it draws go back into the deck without ever
    carrying the tarot (``_play_continues``). Both facts are pinned here: the swap, and the
    fact that the clear carries no dig at all."""
    g_on = fixtures.FIXTURES["tarot_target_cycle"]().games[0]
    evs = _ranked_dict(g_on)
    dig, clear = evs[_key(*TC_DIG_LINE)], evs[_key(*TC_CLEAR_NOW)]
    assert dig > clear, f"dig {dig:.6f} did not beat clear-now {clear:.6f}"
    an = H.HandAnalysis(g_on, H.DEFAULT_HAND_CONFIG, legal=g_on.legal_actions())
    assert an.extraction_ev({"type": "play", "cards": list(TC_CLEAR_NOW[1])}) == 0.0
    assert an.extraction_ev({"type": "discard", "cards": list(TC_DIG_LINE[1])}) > 0.0
    # ... and the dig line only exists because a tarot is held
    assert tuple(TC_DIG_LINE[1]) in an._dig_lines()


def test_tarot_dig_value_depends_on_the_COUNT_of_targets_a_line_keeps():
    """Two discards of the SAME size, from the same hand, leaving the same made-hand floor:
    ``[6,7]`` keeps the Ace of Hearts the Sun is waiting on, ``[5,6]`` bins it (it pairs with
    nothing and sits in no straight window, so it is third in the junk ordering and the very
    next junk-out-k line throws it). Roughly 8x.

    Honest scope: this half of the claim is NOT new -- W-EXTRACT's per-count form already
    subtracted the wanted cards a line keeps. It is here because it is the half that carries
    most of the fixture's margin; the genuinely new half is the next test."""
    g = fixtures.FIXTURES["tarot_target_cycle"]().games[0]
    an = H.HandAnalysis(g, H.DEFAULT_HAND_CONFIG, legal=g.legal_actions())
    keeps_ace = an.extraction_ev({"type": "discard", "cards": [6, 7]})
    bins_ace = an.extraction_ev({"type": "discard", "cards": [5, 6]})
    assert keeps_ace > 4.0 * bins_ace > 0.0


def test_tarot_dig_value_depends_on_WHICH_target_card_the_line_keeps():
    """The per-TARGET claim, isolated (ev/CYCLE_NOTES.md section 1). Same board, same line
    (``discard [6,7]``), same number of Hearts kept, same 2 cards drawn, same draw pile
    depth -- only the RANK of the one held Heart changes, Ace vs Four. The Ace covers the
    top grade tiers on its own, so the second Heart the Sun still needs may be any Heart at
    all; with the Four held, the good tiers still need two more cards.

    W-EXTRACT's per-COUNT ``_cycle_ev`` scored these two boards IDENTICALLY -- it read only
    ``m`` and the COUNT of wanted cards kept, never which ones. That is exactly the
    "depends only on m drawn, not WHICH cards" limitation PROBE_NOTES.md section 3.3
    reported, and it is what this workstream removes."""
    from dataclasses import replace
    line = {"type": "discard", "cards": list(TC_DIG_LINE[1])}
    g_hi = fixtures.FIXTURES["tarot_target_cycle"]().games[0]
    g_lo = _tc_low().games[0]
    hi = H.HandAnalysis(g_hi, H.DEFAULT_HAND_CONFIG, legal=g_hi.legal_actions())
    lo = H.HandAnalysis(g_lo, H.DEFAULT_HAND_CONFIG, legal=g_lo.legal_actions())
    assert hi.extraction_ev(line) > lo.extraction_ev(line) > 0.0
    # ... and the old per-count form cannot tell the two boards apart at all
    old = replace(H.DEFAULT_HAND_CONFIG, tarot_per_target=False)
    hi_old = H.HandAnalysis(g_hi, old, legal=g_hi.legal_actions())
    lo_old = H.HandAnalysis(g_lo, old, legal=g_lo.legal_actions())
    assert hi_old.extraction_ev(line) == pytest.approx(lo_old.extraction_ev(line))


def test_tarot_per_target_off_restores_the_old_ordering():
    """``tarot_per_target=False`` is the h2h / gate A/B arm: with it, the dig is not
    generated, a clearing play banks the cycle bonus again, and the fixture goes back to
    the weaker claim PROBE_NOTES.md section 3.3 pinned (same action, more EV)."""
    from dataclasses import replace
    old = replace(H.DEFAULT_HAND_CONFIG, tarot_per_target=False)
    g_on = fixtures.FIXTURES["tarot_target_cycle"]().games[0]
    g_off = fixtures.FIXTURES["tarot_target_cycle_control"]().games[0]
    an = H.HandAnalysis(g_on, old, legal=g_on.legal_actions())
    assert an._dig_lines() == []
    clear = {"type": "play", "cards": [0, 1, 2, 3, 4]}
    assert an.extraction_ev(clear) > 0.0            # the clear banked it under the old form
    ev_on = dict((H._action_sort_key(a), ev) for a, ev in H.rank_hand_actions(g_on, cfg=old))
    ev_off = dict((H._action_sort_key(a), ev) for a, ev in H.rank_hand_actions(g_off, cfg=old))
    k = _key("play", (0, 1, 2, 3, 4))
    assert ev_on[k] > ev_off[k]


def test_tarot_target_cycle_control_has_nothing_to_dig_toward():
    g_off = fixtures.FIXTURES["tarot_target_cycle_control"]().games[0]
    an = H.HandAnalysis(g_off, H.DEFAULT_HAND_CONFIG, legal=g_off.legal_actions())
    assert an._tarot_wants == [] and an.extract_on is False and an._dig_lines() == []
    assert H.extraction_lines(g_off, g_off.legal_actions()) == []


def test_dig_lines_obey_the_same_safety_gate_as_the_extraction_lines():
    """CYCLE_NOTES.md section 3: a dig costs a discard, so it is only generated while the
    tail DP still clears the blind after spending it -- and never at a LIVE Nemesis."""
    g = fixtures.FIXTURES["tarot_target_cycle"]().games[0]
    an = H.HandAnalysis(g, H.DEFAULT_HAND_CONFIG, legal=g.legal_actions())
    assert an._dig_lines()                                   # safe at the fixture's target
    g.current_blind.chips_target = 10 ** 7                   # unreachable
    an2 = H.HandAnalysis(g, H.DEFAULT_HAND_CONFIG, legal=g.legal_actions())
    assert an2.extract_on is True and an2._dig_lines() == []
    g.current_blind.chips_target = TC_CHIPS_TARGET
    g.current_blind.is_pvp = True
    an3 = H.HandAnalysis(g, H.DEFAULT_HAND_CONFIG, legal=g.legal_actions())
    assert an3.extract_on is False and an3._dig_lines() == []


# ─────────────────────────────────────────────────────────── the safety gate suppresses money

def test_safety_gate_suppresses_extraction_when_p_clear_is_low():
    """Brief section 7's explicit second pin, independent of the "matched control" ask
    above: re-using the purple_seal_discard board (real purple seals still in hand) but with
    an unreachable ``chips_target``, the tail DP's safety gate (EXTRACT_NOTES.md section 4,
    ``extraction_safe`` >= 0.90) must suppress the money -- no extraction line is generated
    and the gated money term drops out of ``evaluate()`` entirely, exactly as
    ``test_extraction.py::test_extraction_lines_are_suppressed_when_the_blind_is_not_safe``
    pins it for a hand-built state."""
    g0 = fixtures.FIXTURES["purple_seal_discard"]().games[0]
    an = H.HandAnalysis(g0, H.DEFAULT_HAND_CONFIG, legal=g0.legal_actions())
    assert an.extraction_safe(an.h, an.d - 1, an.need) is True   # sane at the fixture's own target

    g0.current_blind.chips_target = 10 ** 7        # unreachable -- P(clear) collapses
    an2 = H.HandAnalysis(g0, H.DEFAULT_HAND_CONFIG, legal=g0.legal_actions())
    assert an2.extract_on is True                  # the seals are still there...
    assert an2.extraction_safe(an2.h, an2.d - 1, an2.need) is False   # ...but unsafe to bank
    assert H.extraction_lines(g0, g0.legal_actions()) == []


def test_safety_gate_is_off_at_a_nemesis_even_with_procs_present():
    """EXTRACT_NOTES.md section 4: no unused-hand money at a PvP blind, and every hand is
    played anyway -- extraction is unconditionally off there regardless of P(clear)."""
    g0 = fixtures.FIXTURES["purple_seal_discard"]().games[0]
    g0.current_blind.is_pvp = True
    an = H.HandAnalysis(g0, H.DEFAULT_HAND_CONFIG, legal=g0.legal_actions())
    assert an.extract_on is False
    assert H.extraction_lines(g0, g0.legal_actions()) == []


# ─────────────────────────────────────────────────────────── advisor / CLI integration

@pytest.mark.parametrize("name", ALL_NAMES)
def test_advisor_renders_the_extraction_line_with_its_money_decomposition(name):
    """``python ev/cli.py advise fixture:<name>`` must show the extraction line among the
    ranked actions with its dollar decomposition -- brief section 7's literal ask. This goes
    through ``advisor.advise`` (not ``hand.py`` directly), i.e. the actual CLI path: state
    source loading, ``EVPlayer.explain`` (rules tier, no checkpoint), the money-decomposition
    rendering W-EXTRACT added to ``player.py``'s ``_rank_hand``."""
    m = fixtures.FIXTURES[name]()
    text = advisor.advise(m, 0, n_rollouts=1, rollout_budget="fast", budget="fast")
    assert "Ranked actions" in text
    assert "[extract $+" in text
    # the top-ranked printed line (line "1.") is the extraction line, matching the raw-API
    # assertions above -- what Tagg actually reads at the top of the CLI output.
    top_line = next(ln for ln in text.splitlines() if ln.strip().startswith("1."))
    assert "[extract $+" in top_line


@pytest.mark.parametrize("name", ALL_NAMES)
def test_advisor_control_shows_no_extraction_line(name):
    m = fixtures.FIXTURES[f"{name}_control"]()
    text = advisor.advise(m, 0, n_rollouts=1, rollout_budget="fast", budget="fast")
    assert "[extract $" not in text


@pytest.mark.parametrize("name", ALL_NAMES)
def test_load_state_source_fixture_matches_direct_registry_call(name):
    m, default_player = advisor.load_state_source(f"fixture:{name}")
    assert default_player == 0
    assert m.games[0].state == State.SELECTING_HAND
    m2, _ = advisor.load_state_source(f"fixture:{name}_control")
    assert m2.games[0].state == State.SELECTING_HAND


def test_advise_does_not_mutate_the_fixtures():
    """Same invariant ``test_advisor.py`` pins for bloodstone_vs_invisible: advise() is
    read-only on the match it is handed."""
    for name in ALL_NAMES:
        m = fixtures.FIXTURES[name]()
        sig_before = m.signature()
        advisor.advise(m, 0, n_rollouts=1, rollout_budget="fast", budget="fast")
        assert m.signature() == sig_before
