"""
A boss-rejected hand exhaustion costs a LIFE, not the run — checked through the runner.

Phase 3 shipped `runner._repair_mlb_gameover_bug`, a workaround for `game.py`'s
`bl_hook` / `bl_eye` / `bl_mouth` "hand rejected" branches, which set `State.GAME_OVER` the
instant a rejected hand drove `hands_left` to 0 without checking `self.mlb`.  The lead fixed
the engine at the Phase 3 close (those branches route through `_mlb_fail_round()`;
`engine/tests/.../test_mlb_match.py::TestBossRejectionRespectsMLB`), so Phase 4 W2 deleted
the workaround.  This file is its replacement at the layer that used to need it: drive a
game through the rejection path with the tournament's own `_drive_to_next_nemesis` /
`Tournament` and assert the agent loses ONE life and carries on to the next Nemesis.

Only ante 1's real vanilla Boss can be one of these three bosses — a Nemesis's `boss_key`
is always `MLB_NEMESIS_KEY` — so the scenario is built there, exactly as the engine
regression test builds it.
"""
from __future__ import annotations

import pytest

from tournament.bootstrap import BalatroGame, MLB_PVP_START_ROUND, State
from tournament.runner import Tournament, _drive_to_next_nemesis

SEED = "7I4M53DL"


class RejectionPlayer:
    """Skips Small and Big, then forces `boss_key` at ante 1's Boss and feeds it hands the
    boss must reject until they run out.  After that it plays ordinarily enough to reach the
    ante-2 Nemesis (skip the regular blinds, play the Nemesis to exhaustion)."""

    def __init__(self, boss_key: str = "bl_eye"):
        self.boss_key = boss_key
        self.forced = False

    def act(self, game) -> dict:
        s = game.state
        if s == State.BLIND_SELECT:
            if game.current_blind.kind != "Boss" and not game.current_blind.is_pvp:
                return {"type": "skip_blind"}
            return {"type": "play_blind"}
        if s == State.SELECTING_HAND:
            if game.ante == 1 and game.current_blind.is_boss and not game.current_blind.is_pvp:
                if not self.forced:
                    game.current_blind.boss_key = self.boss_key
                    # bl_eye rejects a REPEATED hand type; bl_mouth rejects anything but the
                    # first.  Either way a single card is a High Card and gets rejected.
                    game.played_hand_types_this_round = (
                        {"High Card"} if self.boss_key == "bl_eye" else {"Flush Five"})
                    self.forced = True
                return {"type": "play", "cards": [0]}
            return {"type": "play", "cards": list(range(min(5, len(game.hand))))}
        if s == State.SHOP:
            return {"type": "leave_shop"}
        if s == State.BOOSTER_OPEN:
            return {"type": "skip_booster"}
        return {"type": "advance"}

    def reset(self) -> None:
        self.forced = False


@pytest.mark.parametrize("boss_key", ["bl_eye", "bl_mouth"])
def test_rejected_exhaustion_costs_one_life_and_the_drive_continues(boss_key):
    game = BalatroGame(seed=SEED, ruleset="mlb")
    lives = game.lives
    status, _ = _drive_to_next_nemesis(game, RejectionPlayer(boss_key))

    assert status == "at_nemesis", "the run must survive a rejected exhaustion"
    assert game.lives == lives - 1, "exactly one life, through _mlb_fail_round"
    assert game.ante == MLB_PVP_START_ROUND
    assert game.state == State.ROUND_EVAL and game.current_blind.is_pvp


def test_rejected_exhaustion_on_the_last_life_ends_the_run():
    game = BalatroGame(seed=SEED, ruleset="mlb")
    game.lives = 1
    status, _ = _drive_to_next_nemesis(game, RejectionPlayer("bl_eye"))

    assert status == "dead"
    assert game.lives == 0 and game.state == State.GAME_OVER


def test_a_tournament_containing_a_rejection_agent_completes():
    """The whole point of removing the repair: nothing in the runner has to notice."""
    players = [RejectionPlayer("bl_eye"), RejectionPlayer("bl_mouth")]
    t = Tournament(seed=SEED, n_agents=2, players=players, life_rule="paired", max_ante=3)
    result = t.run()

    assert [m.ante for m in result.ante_matrices] == [MLB_PVP_START_ROUND, 3]
    # Each agent paid a life at the ante-1 Boss and may have paid more at the Nemeses.
    assert all(v is not None and v < 4 for v in result.final_lives)


def test_no_game_over_with_lives_left_survives_anywhere_in_the_run():
    """`GAME_OVER` with `lives > 0` was the exact signature the deleted repair detected.
    It must not occur any more — if it ever does again, the engine regressed."""
    players = [RejectionPlayer("bl_eye"), RejectionPlayer("bl_mouth")]
    t = Tournament(seed=SEED, n_agents=2, players=players, life_rule="none", max_ante=3)
    t.run()
    for g in t._last_games:
        assert not (g.state == State.GAME_OVER and g.lives > 0)
