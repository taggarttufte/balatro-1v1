"""W-PVP (2026-08-26) — the EV player under ``pvp_protocol="trailer_compelled"``.

Three things:

1. **The default player is untouched.**  Every new knob is a ``HandConfig`` field that
   defaults to OFF, so ``ev:fast`` with ``DEFAULT_HAND_CONFIG`` produces bit-identical
   rankings to the pre-change objective and the extraction layer is still unconditionally
   off at a Nemesis.
2. **The level-1 objective** (``PVP_NOTES.md`` §3): the tie/win mixture reacts to the
   opponent's REVEALED live score, and the leader's PASS is a real, valued candidate.
3. **The extraction pivot** (§5): money at a Nemesis only once the race is DECIDED —
   pinned on the constructed ``nemesis_decided_lost`` fixture and its live-race control.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

import _bootstrap  # noqa: F401
from _bootstrap import MLBMatch, State

import hand as H
import player as P
from hand import DEFAULT_HAND_CONFIG, HandAnalysis, opponent_final_atoms, rank_hand_actions

from fixtures.nemesis_decided_lost import (
    build as build_lost, build_control as build_live, to_nemesis_hand,
    OPP_SCORE_LOST, OPP_SCORE_LIVE,
)

CFG_ON = P.protocol_hand_cfg()                       # level1 + pass + pvp_extract
CFG_NO_EXTRACT = replace(CFG_ON, pvp_extract=False)
CFG_LEVEL0 = P.protocol_hand_cfg(level1=False)       # the attribution arm of the §8 h2h


def _keys(ranked):
    return [(H._action_sort_key(a), round(ev, 12)) for a, ev in ranked]


# ───────────────────────────────────────── 1. the default player is untouched

class TestDefaultsUnchanged:

    def test_new_knobs_all_default_off(self):
        c = DEFAULT_HAND_CONFIG
        assert (c.pvp_level1, c.pvp_pass, c.pvp_extract) == (False, False, False)

    def test_atoms_default_to_level_zero(self):
        m = build_live()
        g = m.games[0]
        model = H.blind_model_for(g, DEFAULT_HAND_CONFIG)
        ratio = HandAnalysis(g, DEFAULT_HAND_CONFIG, lite=True, model=model).ratio
        base = opponent_final_atoms(g, model, ratio)
        assert opponent_final_atoms(g, model, ratio, level1=False) == base
        assert len(base) == 3 and abs(sum(w for w, _ in base) - 1.0) < 1e-12

    def test_extraction_still_off_at_a_nemesis_by_default(self):
        """The pre-change contract: `extract_on` is False at every Nemesis, decided or not."""
        for build in (build_lost, build_live):
            g = build().games[0]
            an = HandAnalysis(g, DEFAULT_HAND_CONFIG, legal=g.legal_actions())
            assert an.pvp is True
            assert an.extract_on is False
            assert an.extraction_safe(an.h, an.d, an.need) is False
            assert an.pvp_decided() == ""
            assert H.extraction_lines(g, cfg=DEFAULT_HAND_CONFIG) == []

    def test_default_ranking_has_no_pass_even_when_leading(self):
        m = to_nemesis_hand()
        g = m.games[0]
        m.games[1].chips_scored = 0
        g.chips_scored = 5000
        m.sync()
        an = HandAnalysis(g, DEFAULT_HAND_CONFIG, legal=g.legal_actions())
        assert an.pvp_leader is True, "leadership is a state fact, not a config one"
        assert an.pass_candidate() is None, "cfg.pvp_pass off => no candidate is generated"
        assert all(a["type"] != "pvp_pass"
                   for a, _ in rank_hand_actions(g, cfg=DEFAULT_HAND_CONFIG, allow_pass=True))


# ────────────────────────────────────── 2. level-1: react to the revealed score

class TestLevelOneAtoms:

    def _atoms(self, g, cfg, **kw):
        model = H.blind_model_for(g, cfg)
        ratio = HandAnalysis(g, cfg, lite=True, model=model).ratio
        return opponent_final_atoms(g, model, ratio, **kw)

    def test_trailer_gets_a_live_atom_at_the_revealed_score(self):
        m = to_nemesis_hand()
        g = m.games[0]
        m.games[1].chips_scored = 900          # I am behind: they may sit on their hands
        m.sync()
        lvl0 = self._atoms(g, CFG_ON)
        lvl1 = self._atoms(g, CFG_ON, level1=True, live_weight=0.5)
        assert len(lvl1) == len(lvl0) + 1
        assert abs(sum(w for w, _ in lvl1) - 1.0) < 1e-12
        w_live, a_live = lvl1[0]
        assert a_live == pytest.approx(900.0)
        assert w_live == pytest.approx(0.5)
        # the projection is untouched apart from the renormalisation
        assert [a for _, a in lvl1[1:]] == [a for _, a in lvl0]
        assert [w for w, _ in lvl1[1:]] == pytest.approx([0.5 * w for w, _ in lvl0])

    def test_the_leader_gets_no_live_atom_the_opponent_is_compelled(self):
        m = to_nemesis_hand()
        g = m.games[0]
        m.games[1].chips_scored = 100
        g.chips_scored = 900                    # I lead: they MUST answer
        m.sync()
        assert self._atoms(g, CFG_ON, level1=True) == self._atoms(g, CFG_ON)

    def test_equal_scores_get_no_live_atom(self):
        m = to_nemesis_hand()
        g = m.games[0]
        m.games[1].chips_scored = 400
        g.chips_scored = 400
        m.sync()
        assert self._atoms(g, CFG_ON, level1=True) == self._atoms(g, CFG_ON)

    def test_an_exhausted_opponent_is_a_point_mass_either_way(self):
        m = to_nemesis_hand()
        g = m.games[0]
        m.games[1].chips_scored = 900
        m.games[1].hands_left = 0
        m.sync()
        assert self._atoms(g, CFG_ON, level1=True) == [(1.0, 900.0)]
        assert self._atoms(g, CFG_ON) == [(1.0, 900.0)]

    def test_live_weight_one_collapses_to_the_revealed_score(self):
        m = to_nemesis_hand()
        g = m.games[0]
        m.games[1].chips_scored = 900
        m.sync()
        assert self._atoms(g, CFG_ON, level1=True, live_weight=1.0) == [(1.0, 900.0)]
        assert self._atoms(g, CFG_ON, level1=True, live_weight=0.0) == self._atoms(g, CFG_ON)

    def test_level_one_raises_the_trailers_estimate_of_the_race(self):
        """Only having to beat what is on the HUD is strictly easier than having to beat
        the HUD plus every hand they still hold — so the trailer's race value goes UP, and
        the decided-LOST gate therefore gets harder to trip, not easier."""
        m = to_nemesis_hand()
        g = m.games[0]
        m.games[1].chips_scored = 900
        m.sync()
        v0 = HandAnalysis(g, CFG_LEVEL0, legal=g.legal_actions()).race_value()
        v1 = HandAnalysis(g, CFG_ON, legal=g.legal_actions()).race_value()
        assert v1 > v0


# ─────────────────────────────────────────────── 3. the leader's PASS

class TestPass:

    def _leading(self, opp=100, mine=90_000):
        m = to_nemesis_hand()
        g = m.games[0]
        m.games[1].chips_scored = opp
        g.chips_scored = mine
        m.sync()
        return m, g

    def test_the_leader_gets_a_pass_candidate(self):
        _m, g = self._leading()
        an = HandAnalysis(g, CFG_ON, legal=g.legal_actions())
        assert an.pvp_leader is True
        act, ev = an.pass_candidate()
        assert act == {"type": "pvp_pass"}
        assert ev == pytest.approx(an.race_value() + CFG_ON.pvp_pass_tiebreak)

    def test_the_trailer_never_gets_one(self):
        m = to_nemesis_hand()
        g = m.games[0]
        m.games[1].chips_scored = 90_000
        m.sync()
        an = HandAnalysis(g, CFG_ON, legal=g.legal_actions())
        assert an.pvp_leader is False and an.pass_candidate() is None

    def test_pass_wins_a_decided_won_race(self):
        """Ahead by an unbridgeable margin: every action is worth 1.0, so the tie-break —
        'waiting spends nothing' — decides, and the player waits."""
        _m, g = self._leading(opp=1, mine=10 ** 7)
        ranked = rank_hand_actions(g, cfg=CFG_NO_EXTRACT, allow_pass=True)
        assert ranked[0][0]["type"] == "pvp_pass"

    def test_pass_loses_when_the_race_is_live_and_a_hand_helps(self):
        """Barely ahead with the opponent holding four hands: waiting is not free, and a
        real play outranks it."""
        _m, g = self._leading(opp=280, mine=300)
        ranked = rank_hand_actions(g, cfg=CFG_ON, allow_pass=True)
        assert ranked[0][0]["type"] != "pvp_pass"

    def test_pass_never_leaks_into_a_path_that_steps_a_bare_game(self):
        """`pvp_pass` is a MATCH action; `BalatroGame.step` does not know it.  The default
        ranking call, the lite/rollout path and the full budget must never emit one."""
        _m, g = self._leading(opp=1, mine=10 ** 7)
        assert all(a["type"] != "pvp_pass" for a, _ in rank_hand_actions(g, cfg=CFG_ON))
        assert all(a["type"] != "pvp_pass"
                   for a, _ in H._hand_ranking_fast(g, CFG_ON, lite=True))
        assert all(a["type"] != "pvp_pass"
                   for a, _ in rank_hand_actions(g, cfg=CFG_ON, budget="full",
                                                 n_worlds=1, top_k=2, allow_pass=True))

    def test_the_match_has_the_last_word_on_legality(self):
        """The analysis derives pass legality from the two scores; if the match did not
        offer it (anti-wedge streak, canonical protocol) `act` must not return one."""
        m, g = self._leading(opp=1, mine=10 ** 7)
        ev = P.EVPlayer(budget="fast", seed=0, epsilon=0.0, hand_cfg=CFG_ON)
        assert ev.act(g)["type"] != "pvp_pass"                    # no extra_actions
        assert ev.act(g, extra_actions=[])["type"] != "pvp_pass"
        offered = [a for a in m.legal_actions(0) if a["type"] == "pvp_pass"]
        assert offered, "the protocol should offer the leader a pass here"
        assert ev.act(g, extra_actions=offered)["type"] == "pvp_pass"

    def test_adapt_match_player_threads_the_pass_through(self):
        m, _g = self._leading(opp=1, mine=10 ** 7)
        pol = P.adapt_match_player(P.EVPlayer(budget="fast", seed=0, epsilon=0.0,
                                              hand_cfg=CFG_ON))
        assert pol(m, 0, m.legal_actions(0))["type"] == "pvp_pass"
        assert pol(m, 1, m.legal_actions(1))["type"] != "pvp_pass"

    def test_explain_renders_the_pass(self):
        m, g = self._leading(opp=1, mine=10 ** 7)
        ev = P.EVPlayer(budget="fast", seed=0, epsilon=0.0, hand_cfg=CFG_ON)
        rows = ev.explain(g, extra_actions=[a for a in m.legal_actions(0)
                                            if a["type"] == "pvp_pass"])
        top = rows[0]
        assert top[0]["type"] == "pvp_pass" and "WAIT" in top[2]


# ───────────────────────────────── 4. the extraction pivot: decided vs live

class TestDecidedRaceExtraction:

    def test_decided_lost_opens_the_gate_and_the_harvest_outranks_the_chase(self):
        g = build_lost().games[0]
        an = HandAnalysis(g, CFG_ON, legal=g.legal_actions())
        assert g.pvp_opponent_score == OPP_SCORE_LOST
        assert an.extract_on is True
        assert an.race_value() <= CFG_ON.pvp_decided_lost_max
        assert an.pvp_decided() == "lost"
        assert an.extraction_safe(an.h, an.d, an.need) is True

        harvest = rank_hand_actions(g, cfg=CFG_ON, allow_pass=True)
        chase = rank_hand_actions(g, cfg=CFG_NO_EXTRACT, allow_pass=True)
        # the futile chase: no line reaches the opponent, every candidate is worth ~0
        assert chase[0][1] <= CFG_ON.pvp_decided_lost_max
        # the harvest: dump the two Purple seals for two Tarots ($8)
        assert harvest[0][0] == {"type": "discard", "cards": [5, 6]}
        assert harvest[0][1] > chase[0][1] + 0.05
        lines = H.extraction_lines(g, cfg=CFG_ON)
        assert lines and lines[0][0] == {"type": "discard", "cards": [5, 6]}
        assert "decided-lost" in lines[0][2]

    def test_a_live_race_keeps_the_gate_shut_bit_for_bit(self):
        g = build_live().games[0]
        an = HandAnalysis(g, CFG_ON, legal=g.legal_actions())
        assert g.pvp_opponent_score == OPP_SCORE_LIVE
        assert an.extract_on is True, "there IS something to extract — the gate is what stops it"
        assert an.pvp_decided() == ""
        assert an.extraction_safe(an.h, an.d, an.need) is False
        assert H.extraction_lines(g, cfg=CFG_ON) == []
        assert _keys(rank_hand_actions(g, cfg=CFG_ON, allow_pass=True)) == \
               _keys(rank_hand_actions(g, cfg=CFG_NO_EXTRACT, allow_pass=True))

    def test_the_last_life_suppresses_the_harvest(self):
        """A decided-lost Nemesis on the LAST life is ``loseGame``: GAME_OVER at once with
        no Cash Out (MLB_NOTES §1.3c), so the dollars are worth nothing."""
        m = build_lost()
        g = m.games[0]
        g.lives = 1
        an = HandAnalysis(g, CFG_ON, legal=g.legal_actions())
        assert an.pvp_decided() == ""
        assert _keys(rank_hand_actions(g, cfg=CFG_ON)) == \
               _keys(rank_hand_actions(g, cfg=CFG_NO_EXTRACT))

    def test_decided_won_opens_the_gate_too(self):
        m = to_nemesis_hand()
        g = m.games[0]
        m.games[1].chips_scored = 1
        g.chips_scored = 10 ** 7
        m.sync()
        an = HandAnalysis(g, CFG_ON, legal=g.legal_actions())
        assert an.race_value() >= CFG_ON.pvp_decided_won_min
        assert an.pvp_decided() == "won"

    def test_a_regular_blind_still_uses_the_tail_dp_gate(self):
        """The pivot must not touch the non-PvP gate: same state, same answer as before."""
        m = MLBMatch(seed="NEMLOST1")
        m.step(0, {"type": "play_blind"})
        g = m.games[0]
        an_new = HandAnalysis(g, CFG_ON, legal=g.legal_actions())
        an_old = HandAnalysis(g, DEFAULT_HAND_CONFIG, legal=g.legal_actions())
        assert an_new.pvp is False
        assert an_new.extract_on == an_old.extract_on
        assert an_new.extraction_safe(an_new.h, an_new.d, an_new.need) == \
               an_old.extraction_safe(an_old.h, an_old.d, an_old.need)
        assert _keys(rank_hand_actions(g, cfg=CFG_ON)) == \
               _keys(rank_hand_actions(g, cfg=DEFAULT_HAND_CONFIG))


# ────────────────────────────────────────────── 5. side-effect freedom

class TestSideEffectFreedom:

    @pytest.mark.parametrize("build", [build_lost, build_live])
    def test_nothing_mutates_the_game(self, build):
        m = build()
        g = m.games[0]
        before = (g.state_signature(), g.run_state.rng.snapshot()["state"],
                  m.signature())
        rank_hand_actions(g, cfg=CFG_ON, allow_pass=True)
        H.extraction_lines(g, cfg=CFG_ON)
        HandAnalysis(g, CFG_ON, legal=g.legal_actions()).pvp_decided()
        after = (g.state_signature(), g.run_state.rng.snapshot()["state"], m.signature())
        assert after == before

    def test_a_full_protocol_match_runs_end_to_end(self):
        pols = [P.adapt_match_player(P.EVPlayer(budget="fast", seed=s, epsilon=0.0,
                                                hand_cfg=CFG_ON)) for s in (11, 22)]
        m = MLBMatch(seed="1558AXDL", lives=4, pvp_protocol="trailer_compelled")
        m.play_out(pols, max_steps=4000)
        assert m.done and m.winner in (0, 1)
        assert sum(m.pvp_passes) == len(m.pvp_pass_detail)
