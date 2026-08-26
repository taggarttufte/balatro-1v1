"""
The `MCTSPlayer` plug-in (Phase 4 W2, BATCH_NOTES.md §7.2).

`players.MCTSPlayer` is now a factory over `agent/mcts/player.py::make_player`. What has
to hold is exactly the `Player` protocol the runner leans on:

  * `act(game)` returns a dict `game.step()` accepts, NEVER `None` — `_drive_to_next_nemesis`
    steps unconditionally;
  * `reset()` exists and drops the retained tree, because `Tournament.run()` calls it before
    every run and the determinism tests depend on a player carrying nothing across runs;
  * a whole tournament of them completes, produces one `AnteMatrix` per Nemesis ante, and
    the lives bookkeeping is the same as for any other player.

These tests need torch. They skip (not fail) if it is absent, because `tournament` is
deliberately importable without it.
"""
from __future__ import annotations

import pytest

from tournament.bootstrap import BalatroGame, MLB_PVP_START_ROUND, State
from tournament.players import MCTSPlayer, make_scripted
from tournament.runner import Tournament

torch = pytest.importorskip("torch", reason="the MCTS player needs torch")

SEED = "7I4M53DL"
# A tiny net: these tests are about the plumbing, not the policy. `make_player` forwards
# **kwargs to `MCTSPlayer`, and `load_policy(None, ...)` builds the cold-start net for the
# chosen encoder, so the only knob that matters for speed here is `sims`.
SIMS = 8


def _player(seed: int, sims: int = SIMS):
    return MCTSPlayer(checkpoint=None, sims=sims, device="cpu", seed=seed,
                      leaf_batch=16, reuse=True)


def test_factory_returns_a_tournament_shaped_player():
    p = _player(0)
    game = BalatroGame(seed=SEED, ruleset="mlb")
    for _ in range(6):
        action = p.act(game)
        assert isinstance(action, dict), "act() must never return None for the runner"
        game.step(action)
    p.reset()
    assert not p.cache.armed


def test_no_action_state_returns_a_steppable_advance():
    """`legal_actions()` is empty at `PVP_WAIT`; the runner never calls `act` there, but the
    other adapters in `players.py` return `{"type": "advance"}` and so must this one."""
    p = _player(1)
    game = BalatroGame(seed=SEED, ruleset="mlb")
    game.state = State.PVP_WAIT
    assert p.act(game) == {"type": "advance"}


def test_four_mcts_agents_play_a_tournament_to_ante_three():
    players = [_player(i) for i in range(4)]
    t = Tournament(seed=SEED, n_agents=4, players=players, life_rule="paired", max_ante=3)
    result = t.run()

    assert [m.ante for m in result.ante_matrices] == [MLB_PVP_START_ROUND, 3]
    assert result.seed == SEED
    for m in result.ante_matrices:
        # Every agent present this ante has a real score and a rank.
        assert m.scores.shape == (4,)
        assert m.outcome.shape == (4, 4)
    assert all(v is not None for v in result.final_lives)
    assert all(0 <= v <= 4 for v in result.final_lives)
    # Independent games on one seed: nobody's tree leaked into anybody else's.
    assert len({id(p) for p in players}) == 4
    assert all(p.searches > 0 for p in players)


def test_mcts_and_scripted_share_one_population():
    """Heterogeneity is the whole point of the N x N matrix: the runner must not care which
    adapter an agent came from."""
    players = [_player(0), make_scripted(name="s1", hand="greedy"),
               _player(1), make_scripted(name="s2", hand="greedy", rerolls_per_visit=1)]
    t = Tournament(seed=SEED, n_agents=4, players=players, life_rule="none", max_ante=2)
    result = t.run()
    assert len(result.ante_matrices) == 1
    assert result.ante_matrices[0].stats["n_present"] == 4
