"""
test_mlb_agent.py — the agent under Major League Balatro rules.

Nothing here existed in balatro-mcts: the source engine had no MLB. The three things
Phase 3 asks W1 to prove:

  1. MCTS can play a `ruleset="mlb"` game including a Nemesis blind whose target is
     supplied externally (`set_pvp_info`), through to GAME_OVER or a fixed ante cap.
  2. A state with NO legal actions is handled gracefully — under MLB there are two of
     them (`State.PVP_WAIT`, and BLIND_SELECT with `pvp_ready` after readying for the
     Nemesis) and they are normal, not errors.
  3. The terminal/outcome signal is a parameter: `_is_win` / the shaped label must not
     assume an ante-8 win under MLB (the game is endless; the win is `match_won`).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from balatro_sim.game import BalatroGame, State
from mcts import MCTS, MCTSPlayer, NNPolicy, PolicyValueNet, UniformPolicy
from mcts.outcome import ExternalOutcome, MLBOutcome, default_outcome_for, is_stuck_state
from mcts.search import MCTSConfig
from train import SelfPlayAgent

TARGET_SCORE = 50_000        # the "opponent's live score" a driver would relay
TARGET_HANDS = 2


def advance_to_nemesis(g: BalatroGame, max_steps: int = 80) -> BalatroGame:
    """Free-win the way to the first PvP blind (the ante-2 Boss slot under MLB)."""
    for _ in range(max_steps):
        if g.current_blind.is_pvp and g.state in (State.BLIND_SELECT, State.SELECTING_HAND):
            return g
        if g.state is State.BLIND_SELECT:
            g.step({"type": "play_blind"})
        elif g.state is State.SELECTING_HAND:
            g.debug_win_blind()
        elif g.state is State.ROUND_EVAL:
            g.step({"type": "advance"})
        elif g.state is State.SHOP:
            g.step({"type": "leave_shop"})
        elif g.state is State.BOOSTER_OPEN:
            g.step({"type": "skip_booster"})
        else:
            raise AssertionError(f"unexpected state on the way to the Nemesis: {g.state}")
    raise AssertionError("never reached a Nemesis blind")


@pytest.fixture(scope="module")
def policy() -> NNPolicy:
    torch.manual_seed(0)
    return NNPolicy(PolicyValueNet(), device="cpu")


def _cheap_cfg() -> MCTSConfig:
    return MCTSConfig(num_simulations=8, gumbel_max_considered=4)


# ── 1. Playing an MLB game, Nemesis included ────────────────────────────────

def test_mcts_plays_an_mlb_game_to_game_over_or_ante_3(policy):
    """The headline smoke: an MCTS self-play episode on a `ruleset="mlb"` game, with an
    external Nemesis target, runs to GAME_OVER or the ante cap without crashing."""
    calls = []

    def target(_g):
        calls.append(_g.ante)
        return TARGET_SCORE, TARGET_HANDS

    agent = SelfPlayAgent(policy, _cheap_cfg(), rng=np.random.default_rng(0),
                          max_decisions=400, max_antes=3, pvp_target_fn=target)
    game = BalatroGame(seed="7I4M53DL", ruleset="mlb")
    assert game.lives == 4 and game.mlb

    result = agent.play_episode(game)

    assert result.stop_reason in ("game_over", "max_antes", "max_decisions")
    assert len(result.samples) > 0
    assert 0.0 <= result.z <= 1.0
    if result.stop_reason == "game_over":
        assert game.lives <= 0 or game.match_won
    else:
        assert game.ante >= 3
    # Lives are the MLB currency; a random-init policy loses blinds, never gains lives.
    assert game.lives <= 4


def test_mcts_plays_the_nemesis_with_the_supplied_target(policy):
    """At a Nemesis the blind's chips_target IS the opponent's live score, relayed by
    the driver. The agent must apply it and play the blind out (a PvP round never ends
    on `chips >= target` — every hand is played)."""
    game = advance_to_nemesis(BalatroGame(seed="7I4M53DL", ruleset="mlb"))
    seen_targets = []

    def target(g):
        seen_targets.append(g.current_blind.chips_target)
        return TARGET_SCORE, TARGET_HANDS

    agent = SelfPlayAgent(policy, _cheap_cfg(), rng=np.random.default_rng(0),
                          max_decisions=60, pvp_target_fn=target)
    lives_before = game.lives
    result = agent.play_episode(game)

    assert seen_targets, "the PvP target hook never fired at a Nemesis blind"
    # After the first application the game reports back the supplied score.
    assert TARGET_SCORE in seen_targets[1:] or game.pvp_opponent_score == TARGET_SCORE
    assert game.pvp_opponent_hands == TARGET_HANDS
    # A solo MLB game auto-resolves the PvP at exhaustion, so the run moved on.
    assert game.state is not State.PVP_WAIT
    assert game.lives <= lives_before
    assert result.stop_reason in ("game_over", "max_decisions", "max_antes")


def test_mcts_player_acts_on_an_mlb_nemesis(policy):
    """The `Player` shape W2/W4 will plug in: act(game) -> action dict."""
    game = advance_to_nemesis(BalatroGame(seed="7I4M53DL", ruleset="mlb"))
    player = MCTSPlayer(policy, _cheap_cfg(), rng=np.random.default_rng(0))
    for _ in range(12):
        action = player.act(game)
        if action is None:
            break
        game.step(action)
        game.set_pvp_info(TARGET_SCORE, TARGET_HANDS)
    assert player.searches + player.shortcuts > 0


# ── 2. States with no legal actions ─────────────────────────────────────────

def test_pvp_wait_is_a_no_action_state(policy):
    """Hands exhausted at a Nemesis with a match attached (`pvp_solo=False`): the player
    waits. `legal_actions()` is empty and nothing may crash."""
    game = advance_to_nemesis(BalatroGame(seed="7I4M53DL", ruleset="mlb"))
    game.pvp_solo = False
    game.step({"type": "play_blind"})            # -> readied, waiting for startBlind
    assert game.pvp_ready and game.legal_actions() == []
    assert is_stuck_state(game)

    game._start_blind()
    while game.state is State.SELECTING_HAND:
        game.step({"type": "play", "cards": [0, 1, 2, 3, 4]})
    assert game.state is State.PVP_WAIT
    assert game.legal_actions() == []
    assert is_stuck_state(game)

    # Search, player and self-play agent all handle it without raising.
    mcts = MCTS(policy, _cheap_cfg(), rng=np.random.default_rng(0))
    root, visits, chosen = mcts.run_gumbel(game)
    assert visits == {} and chosen is None
    assert root.is_terminal and root.stop_reason == "stuck"
    assert 0.0 <= root.terminal_value <= 1.0

    root2, visits2 = mcts.run(game)
    assert visits2 == {} and root2.stop_reason == "stuck"

    assert MCTSPlayer(policy, _cheap_cfg()).act(game) is None


def test_agent_stops_on_stuck_and_resumes_after_the_driver_resolves(policy):
    """The trajectory is not lost when a Nemesis parks the player: the agent returns
    with stop_reason="stuck", the driver calls end_pvp(), and the SAME episode continues
    (this is how MLBMatch / the tournament runner will drive it)."""
    game = advance_to_nemesis(BalatroGame(seed="7I4M53DL", ruleset="mlb"))
    game.pvp_solo = False
    game._start_blind()
    game.set_pvp_info(TARGET_SCORE, TARGET_HANDS)

    agent = SelfPlayAgent(policy, _cheap_cfg(), rng=np.random.default_rng(0),
                          max_decisions=40)
    first = agent.play_episode(game)
    assert first.stop_reason == "stuck"
    assert game.state is State.PVP_WAIT
    n_first = len(first.samples)
    assert n_first > 0

    game.end_pvp()                      # the "server" resolves the PvP
    assert game.state is State.ROUND_EVAL
    second = agent.resume_episode(game, first)
    assert len(second.samples) > n_first
    assert all(s.z == second.z for s in second.samples)   # one label for the episode


def test_search_on_a_readied_root_is_graceful(policy):
    """BLIND_SELECT + pvp_ready: readied for the Nemesis, waiting for the opponent."""
    game = advance_to_nemesis(BalatroGame(seed="7I4M53DL", ruleset="mlb"))
    game.pvp_solo = False
    game.step({"type": "play_blind"})
    mcts = MCTS(UniformPolicy(), _cheap_cfg(), rng=np.random.default_rng(0))
    root, visits, chosen = mcts.run_gumbel(game)
    assert chosen is None and visits == {} and root.stop_reason == "stuck"


# ── 3. The outcome signal is a parameter ────────────────────────────────────

def test_search_uses_the_mlb_outcome_by_default():
    game = BalatroGame(seed=1, ruleset="mlb")
    mcts = MCTS(UniformPolicy(), _cheap_cfg(), rng=np.random.default_rng(0))
    mcts.run(game)
    assert isinstance(mcts.outcome, MLBOutcome)
    assert default_outcome_for(BalatroGame(seed=1)).name == "vanilla"


def test_mlb_game_over_is_not_valued_as_a_win(policy):
    """An MLB run that dies at ante 12 has passed ante 8 — the vanilla `_is_win` rule
    would call that a WIN and hand the value head a 1.0 label. It must not."""
    game = BalatroGame(seed=1, ruleset="mlb")
    game.ante, game.lives, game.state = 12, 0, State.GAME_OVER
    o = MLBOutcome()
    assert not o.is_win(game)
    assert o.value(game) < 1.0

    mcts = MCTS(UniformPolicy(), _cheap_cfg(), rng=np.random.default_rng(0))
    root, visits = mcts.run(game)
    assert visits == {} and root.stop_reason == "game_over"
    assert root.terminal_value == pytest.approx(o.value(game))


def test_external_outcome_drives_the_backed_up_value():
    """W2/W4 supply the margin; the search must back THAT up, not the engine's guess."""
    game = advance_to_nemesis(BalatroGame(seed="7I4M53DL", ruleset="mlb"))
    game.step({"type": "play_blind"})
    game.set_pvp_info(TARGET_SCORE, TARGET_HANDS)
    outcome = ExternalOutcome(
        value_fn=lambda g: 1.0 if g.chips_scored >= TARGET_SCORE else 0.0,
        terminal_fn=lambda g: g.state is State.GAME_OVER,
    )
    mcts = MCTS(UniformPolicy(), MCTSConfig(num_simulations=20, gumbel_max_considered=4),
                rng=np.random.default_rng(0), outcome=outcome)
    root, visits, chosen = mcts.run_gumbel(game)
    assert chosen is not None and sum(visits.values()) == 20
    assert mcts.outcome is outcome
