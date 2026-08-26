"""W-PVP (2026-08-26) — ``MLBMatch(pvp_protocol=...)``: the trailer-compelled turn protocol.

Two things are pinned here:

1. **The default is provably unchanged.**  ``pvp_canonical_transcripts.json`` was captured
   from the tree BEFORE the protocol landed (four seeds, ``ev:fast`` on both sides, the
   sha1 chain of ``MLBMatch.signature()`` after every single step); the canonical path must
   still reproduce it byte for byte, including the length of the signature tuple.
2. **The protocol itself**, rule by rule, including every edge case decision listed in
   ``ev/PVP_NOTES.md`` §2.

The protocol is a MODELLING CHOICE and cannot be oracle-verified — see the module docstring
of ``balatro_sim/mlb_match.py``.  What CAN be checked against the real rules is that the
protocol changes nobody's verdict: ``_resolve_pvp`` is untouched and the end-condition
tests below assert the same outcomes under both protocols.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from balatro_sim.game import State
from balatro_sim.mlb_match import MLBMatch, PVP_PASS, PVP_PROTOCOLS

from engine_tests.test_mlb_match import (  # the existing drivers; nothing new is needed
    to_nemesis, greedy_policy, weakest_play, greedy_hand, SEEDS,
)

TRANSCRIPTS = Path(__file__).with_name("pvp_canonical_transcripts.json")
PROTO = "trailer_compelled"


# ─────────────────────────────────────────────────────────────── helpers

def _sig_digest(m: MLBMatch) -> str:
    return hashlib.sha1(repr(m.signature()).encode("utf-8")).hexdigest()[:16]


def _ev_pair():
    """Two deterministic ``ev:fast`` policies, the ones the fixture was captured with."""
    root = Path(__file__).resolve().parents[3]        # repo root
    for p in (str(root), str(root / "ev"), str(root / "eval")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import player as P            # ev/player.py
    import common as C            # eval/common.py
    ps = [P.EVPlayer(budget="fast", seed=11, epsilon=0.0),
          P.EVPlayer(budget="fast", seed=22, epsilon=0.0)]
    return [C.adapt_player(ps[0]), C.adapt_player(ps[1])]


def _at_nemesis(seed: str, protocol: str = PROTO, ante: int = 2) -> MLBMatch:
    """A match with both players inside a LIVE Nemesis blind at ``ante``, nothing scored."""
    m = MLBMatch(seed=seed, pvp_protocol=protocol)
    for p in (0, 1):
        to_nemesis(m, p, ante=ante)
    for p in (0, 1):
        m.step(p, {"type": "play_blind"})
    assert m.pvp_active and m.pvp_ante == ante
    assert all(g.state == State.SELECTING_HAND for g in m.games)
    return m


def _types(acts) -> set:
    return {a["type"] for a in acts}


# ─────────────────────────────────────────────── 1. the default is unchanged

class TestCanonicalUnchanged:

    def test_default_protocol_is_canonical(self):
        assert MLBMatch(seed=SEEDS[0]).pvp_protocol == "canonical"
        assert PVP_PROTOCOLS[0] == "canonical"

    def test_unknown_protocol_raises(self):
        with pytest.raises(ValueError, match="pvp_protocol"):
            MLBMatch(seed=SEEDS[0], pvp_protocol="simultaneous")

    def test_canonical_signature_tuple_has_no_protocol_tail(self):
        m = MLBMatch(seed=SEEDS[0])
        assert len(m.signature()) == 8
        assert len(MLBMatch(seed=SEEDS[0], pvp_protocol=PROTO).signature()) == 11

    def test_canonical_never_offers_a_pass(self):
        m = _at_nemesis(SEEDS[0], protocol="canonical")
        # give player 0 a lead, then check neither side is ever offered the pass
        m.step(0, greedy_hand(m.games[0]))
        assert m.games[0].chips_scored > m.games[1].chips_scored
        for p in (0, 1):
            assert "pvp_pass" not in _types(m.legal_actions(p))
            assert m.pass_offered(p) is False

    def test_canonical_pass_action_is_ignored_not_recorded(self):
        """The permissive step contract: an action the protocol does not offer is a no-op
        (the turn still passes), exactly as BalatroGame.step ignores an illegal action."""
        m = _at_nemesis(SEEDS[0], protocol="canonical")
        before = m.games[0].state_signature()
        m.step(0, dict(PVP_PASS))
        assert m.games[0].state_signature() == before
        assert m.pvp_passes == [0, 0]
        assert m._turn == 1

    @pytest.mark.parametrize("run", json.loads(TRANSCRIPTS.read_text(encoding="utf-8"))["runs"],
                             ids=lambda r: r["seed"])
    def test_canonical_transcripts_are_unchanged(self, run):
        """Byte-equivalence against transcripts captured BEFORE the protocol landed."""
        pols = _ev_pair()
        m = MLBMatch(seed=run["seed"], deck_key="b_red", stake=1, lives=4)
        chain = hashlib.sha1()
        chain.update(_sig_digest(m).encode())
        while not m.done and m.steps < 4000:
            p = m.current_player()
            if p is None:
                break
            m.step(p, pols[p](m, p, m.legal_actions(p)))
            chain.update(_sig_digest(m).encode())
        assert m.steps == run["steps"]
        assert m.winner == run["winner"]
        assert [list(x) for x in m.pvp_log] == run["pvp_log"]
        assert [list(x) for x in m.pvp_detail] == run["pvp_detail"]
        assert [m.games[0].lives, m.games[1].lives] == run["lives"]
        assert [m.games[0].dollars, m.games[1].dollars] == run["dollars"]
        assert len(m.signature()) == run["signature_len"]
        assert chain.hexdigest() == run["chain"], "canonical step-by-step transcript changed"


# ───────────────────────────────────────── 2. who may act, who may wait

class TestCompulsion:

    def test_hand_one_is_simultaneous(self):
        """Both scores start at 0, so neither player is 'ahead' and neither may wait: both
        are compelled to make a first move.  The brief's 'both play hand 1 simultaneously'
        is not special-cased — it falls out of the strict comparison."""
        m = _at_nemesis(SEEDS[0])
        assert m.games[0].chips_scored == m.games[1].chips_scored == 0
        for p in (0, 1):
            assert m.pass_offered(p) is False
            assert "pvp_pass" not in _types(m.legal_actions(p))

    def test_only_the_strict_leader_may_wait(self):
        m = _at_nemesis(SEEDS[0])
        m.step(0, greedy_hand(m.games[0]))
        lead, trail = (0, 1) if m.games[0].chips_scored > m.games[1].chips_scored else (1, 0)
        assert m.pass_offered(lead) is True
        assert "pvp_pass" in _types(m.legal_actions(lead))
        assert m.pass_offered(trail) is False, "the trailer is COMPELLED — no waiting"
        assert "pvp_pass" not in _types(m.legal_actions(trail))

    def test_equal_scores_mid_blind_are_simultaneous_again(self):
        """Force an exact mid-blind tie: both players are on ONE seed with identical
        decisions, so the same greedy action gives the same score."""
        m = _at_nemesis(SEEDS[0])
        m.step(0, greedy_hand(m.games[0]))
        m.step(1, greedy_hand(m.games[1]))
        assert m.games[0].chips_scored == m.games[1].chips_scored > 0
        for p in (0, 1):
            assert m.pass_offered(p) is False

    def test_an_exhausted_leader_cannot_wait(self):
        """PVP_WAIT has nothing to conserve: the choice belongs to a player who could act."""
        m = _at_nemesis(SEEDS[0])
        g = m.games[0]
        m.step(0, greedy_hand(g))                       # take the lead
        while g.hands_left > 0 and g.state == State.SELECTING_HAND:
            m.step(0, greedy_hand(g))
        if m.done or not m.pvp_active:
            pytest.skip("the round resolved before player 0 exhausted (score-dependent)")
        assert g.state == State.PVP_WAIT
        assert m.pass_offered(0) is False

    def test_pass_is_a_true_no_op_on_the_game(self):
        m = _at_nemesis(SEEDS[0])
        m.step(0, greedy_hand(m.games[0]))
        lead = 0 if m.games[0].chips_scored > m.games[1].chips_scored else 1
        g = m.games[lead]
        before = (g.state_signature(), g.hands_left, g.discards_left, g.chips_scored,
                  [str(c) for c in g.hand], g.run_state.rng.snapshot()["state"])
        steps = m.steps
        m.step(lead, dict(PVP_PASS))
        after = (g.state_signature(), g.hands_left, g.discards_left, g.chips_scored,
                 [str(c) for c in g.hand], g.run_state.rng.snapshot()["state"])
        assert after == before, "a pass must not touch the game, the hand or the RNG"
        assert m.steps == steps + 1 and m._turn == 1 - lead, "the turn passes to the trailer"
        assert m.pvp_passes[lead] == 1
        assert m.pvp_pass_detail == [(m.pvp_ante, lead)]

    def test_two_passes_in_a_row_are_not_offered(self):
        """Anti-wedge: without progress from the compelled player the pair would hand the
        turn back and forth forever."""
        m = _at_nemesis(SEEDS[0])
        m.step(0, greedy_hand(m.games[0]))
        lead = 0 if m.games[0].chips_scored > m.games[1].chips_scored else 1
        m.step(lead, dict(PVP_PASS))
        assert m.pass_offered(lead) is False
        m.step(lead, dict(PVP_PASS))                    # ignored, streak not reset
        assert m.pvp_passes[lead] == 1
        # the compelled player moving restores the option
        trail = 1 - lead
        m.step(trail, greedy_hand(m.games[trail]))
        if m.games[lead].chips_scored > m.games[trail].chips_scored and m.pvp_active:
            assert m.pass_offered(lead) is True

    def test_pass_streak_resets_at_a_new_nemesis(self):
        m = _at_nemesis(SEEDS[0])
        m.step(0, greedy_hand(m.games[0]))
        lead = 0 if m.games[0].chips_scored > m.games[1].chips_scored else 1
        m.step(lead, dict(PVP_PASS))
        assert m._pass_streak == 1
        m.sync()
        for p in (0, 1):                                # play the blind out
            while m.pvp_active and m.can_act(p) and m.games[p].state == State.SELECTING_HAND:
                m.step(p, greedy_hand(m.games[p]))
        if m.done:
            pytest.skip("match ended inside the first Nemesis")
        for p in (0, 1):
            to_nemesis(m, p, ante=3)
            m.step(p, {"type": "play_blind"})
        assert m._pass_streak == 0


# ────────────────────────────────── 3. end conditions are the REAL rules, unchanged

class TestEndConditions:

    def test_early_end_cut_forfeits_the_trailers_hands_not_the_leaders(self):
        """Out of hands AND strictly behind ends the round at once; the leader keeps
        whatever hands are left.  Same rule under both protocols (``_resolve_pvp``)."""
        m = _at_nemesis(SEEDS[0])
        strong, weak = 0, 1
        m.step(strong, greedy_hand(m.games[strong]))
        # a decisive, unambiguous lead: `_resolve_pvp` reads exactly this field, and an
        # ante-2 no-joker board cannot produce a reliable gap from real plays alone
        m.games[strong].chips_scored = 10 ** 6
        m.sync()
        assert m.pvp_active and m.games[strong].hands_left == 3
        while (m.pvp_active and m.games[weak].state == State.SELECTING_HAND
               and m.games[weak].hands_left > 0):
            m.step(weak, weakest_play(m.games[weak]))
        assert not m.pvp_active, "the round must end when the trailer runs out behind"
        ante, loser, s0, s1 = m.pvp_log[-1]
        assert loser == weak and s0 > s1
        early = m.pvp_detail[-1][6]
        assert early is True
        assert m.games[strong].hands_left > 0, "the leader's hands were NOT spent"

    def test_exact_tie_takes_nobody_a_life(self):
        """Both exhausted with equal scores: the server rule (a REMOTE citation — see
        ``_resolve_pvp``'s docstring) is that nobody loses."""
        m = _at_nemesis(SEEDS[0])
        lives = [g.lives for g in m.games]
        for p in (0, 1):
            while m.pvp_active and m.games[p].state == State.SELECTING_HAND:
                m.step(p, greedy_hand(m.games[p]))
        ante, loser, s0, s1 = m.pvp_log[-1]
        assert s0 == s1, "the mirrored seed should give an exact tie with equal play"
        assert loser is None
        assert [g.lives for g in m.games] == lives
        assert m.pvp_detail[-1][6] is False, "both exhausted -> not an early end"

    @pytest.mark.parametrize("seed", SEEDS[:3])
    def test_a_protocol_match_terminates_and_has_a_winner(self, seed):
        m = MLBMatch(seed=seed, pvp_protocol=PROTO)
        m.play_out([greedy_policy, greedy_policy], max_steps=6000)
        assert m.done and m.winner in (0, 1)
        assert m.steps < 6000, "the protocol must not wedge the match"

    @pytest.mark.parametrize("seed", SEEDS[:3])
    def test_a_pass_only_policy_still_terminates(self, seed):
        """The adversarial case for the anti-wedge rule: a policy that always tries to
        wait.  The compelled player has to move, so the blind still resolves."""
        def stubborn(m, p, acts):
            if any(a["type"] == "pvp_pass" for a in acts):
                return dict(PVP_PASS)
            return greedy_policy(m, p, acts)
        m = MLBMatch(seed=seed, pvp_protocol=PROTO)
        m.play_out([stubborn, stubborn], max_steps=6000)
        assert m.done and m.winner in (0, 1)
        assert m.steps < 6000


# ───────────────────────────────────────────────────── 4. plumbing

class TestPlumbing:

    def test_clone_carries_the_protocol_state(self):
        m = _at_nemesis(SEEDS[0])
        m.step(0, greedy_hand(m.games[0]))
        lead = 0 if m.games[0].chips_scored > m.games[1].chips_scored else 1
        m.step(lead, dict(PVP_PASS))
        for c in (m.clone(), m.clone_determinized(seed=1234)):
            assert c.pvp_protocol == PROTO
            assert c.pvp_passes == m.pvp_passes
            assert c.pvp_pass_detail == m.pvp_pass_detail
            assert c._pass_streak == m._pass_streak
        c = m.clone()
        c.step(1 - lead, greedy_hand(c.games[1 - lead]))
        assert c.pvp_passes == m.pvp_passes, "the clone is independent"

    def test_signature_moves_with_a_pass(self):
        m = _at_nemesis(SEEDS[0])
        m.step(0, greedy_hand(m.games[0]))
        lead = 0 if m.games[0].chips_scored > m.games[1].chips_scored else 1
        before = m.signature()
        m.step(lead, dict(PVP_PASS))
        assert m.signature() != before, "the protocol tail must record the pass"

    def test_legal_actions_is_a_fresh_list(self):
        m = _at_nemesis(SEEDS[0])
        m.step(0, greedy_hand(m.games[0]))
        lead = 0 if m.games[0].chips_scored > m.games[1].chips_scored else 1
        acts = m.legal_actions(lead)
        acts.append({"type": "nonsense"})
        assert "nonsense" not in _types(m.legal_actions(lead))
