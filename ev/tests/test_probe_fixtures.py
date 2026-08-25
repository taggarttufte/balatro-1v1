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

Own only: this file, ``mp/ev/fixtures/{purple_seal_discard,faceless_discard,
business_card_board,reserved_parking_hold,gold_seal_weak_play,tarot_target_cycle,
_probe_common}.py``, the additive registration in ``fixtures/__init__.py``, and
``mp/ev/PROBE_NOTES.md``.  ``hand.py`` / ``player.py`` / ``pairs.py`` are read-only here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

import _bootstrap  # noqa: F401
from _bootstrap import State

import fixtures
import hand as H
import player as P
import advisor

_MP_ROOT = str(Path(_bootstrap.MP_ROOT))
if _MP_ROOT not in sys.path:
    sys.path.insert(0, _MP_ROOT)


# Six (sandbag, control) pairs; for five of them the extraction action and the "clear now"
# action are genuinely DIFFERENT candidates (a real rank swap between fixture and control).
# tarot_target_cycle is the one exception -- see its own test below and PROBE_NOTES.md.
SCENARIOS = [
    ("purple_seal_discard", ("discard", (5, 6)), ("play", (0, 1))),
    ("faceless_discard", ("discard", (2, 3, 4)), ("play", (0, 1))),
    ("business_card_board", ("play", (2,)), ("play", (0, 1))),
    ("reserved_parking_hold", ("play", (7,)), ("play", (0, 1))),
    ("gold_seal_weak_play", ("play", (4,)), ("play", (0, 1))),
]
ALL_NAMES = [n for n, _, _ in SCENARIOS] + ["tarot_target_cycle"]


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


def test_tarot_target_cycle_sandbag_beats_the_same_line_without_the_tarot():
    """tarot_target_cycle is the one scenario where the sandbag and control's TOP action is
    literally the same card set (every 5-card clearing play here draws 5 fresh cards, so the
    cycle bonus attaches near-uniformly rather than flipping a rank -- see PROBE_NOTES.md,
    'honest discrepancies'). The claim this test pins is the one that actually holds: the
    SAME action is worth strictly more with the Sun held and no Hearts in hand than without
    any tarot at all, and the bonus is rendered."""
    g_on = fixtures.FIXTURES["tarot_target_cycle"]().games[0]
    g_off = fixtures.FIXTURES["tarot_target_cycle_control"]().games[0]
    action = ("play", (0, 1, 2, 3, 4))
    ev_on = _ranked_dict(g_on)[_key(*action)]
    ev_off = _ranked_dict(g_off)[_key(*action)]
    assert ev_on > ev_off
    lines = H.extraction_lines(g_on, g_on.legal_actions())
    assert lines and all("extract $" in reason for _, _, reason in lines)
    assert H.extraction_lines(g_off, g_off.legal_actions()) == []


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

@pytest.mark.parametrize("name", [n for n, _, _ in SCENARIOS] + ["tarot_target_cycle"])
def test_advisor_renders_the_extraction_line_with_its_money_decomposition(name):
    """``python mp/ev/cli.py advise fixture:<name>`` must show the extraction line among the
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


@pytest.mark.parametrize("name", [n for n, _, _ in SCENARIOS])
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
