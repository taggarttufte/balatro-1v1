"""
The no-progress guard in `_drive_to_next_nemesis` (Phase 4 W2).

Found while running the first MCTS-vs-MCTS tournaments: `game.py`'s SHOP branch of
`legal_actions()` (:1433) enumerates card-targeting `use_consumable` actions against
`self.hand`, which in the SHOP still holds the previous blind's cards, and
`_use_consumable` (:1854) silently no-ops when the application fails.  The result is a
legal action that leaves the game bit-identical, i.e. an infinite loop for any agent that
likes it.  One MCTS agent burned the whole 20 000-step `max_steps_per_drive` budget in a
single shop; under a different noise seed the same pathology cost 55 s of a 60 s generation
and 14 338 training samples.  `mp/engine` is frozen — see `mp/agent/TRAIN_NOTES.md`
"needs engine change" — so the driver breaks the loop instead.

The guard must (a) rescue a wedged agent, (b) be invisible to every agent that makes
progress, and (c) be switchable off.
"""
from __future__ import annotations

import pytest

from tournament.bootstrap import BalatroGame, State
from tournament.runner import (
    NOOP_BUDGET_DEFAULT, Tournament, _drive_to_next_nemesis, _force_progress_action,
)
from tournament.players import default_population, make_scripted

SEED = "7I4M53DL"


class ShopSitter:
    """Plays normally until the first SHOP, then repeats one no-op forever.  The no-op is
    manufactured (a `use_consumable` on an empty consumable hand is a documented early
    return in `_use_consumable`) so the test does not depend on which tarot a seed rolls."""

    def __init__(self):
        self.noops = 0

    def act(self, game) -> dict:
        s = game.state
        if s == State.SHOP:
            self.noops += 1
            return {"type": "use_consumable", "consumable_idx": 99, "target_cards": []}
        if s == State.BLIND_SELECT:
            return ({"type": "skip_blind"} if game.current_blind.kind != "Boss"
                    and not game.current_blind.is_pvp else {"type": "play_blind"})
        if s == State.SELECTING_HAND:
            return {"type": "play", "cards": list(range(min(5, len(game.hand))))}
        if s == State.BOOSTER_OPEN:
            return {"type": "skip_booster"}
        return {"type": "advance"}

    def reset(self) -> None:
        self.noops = 0


def test_the_manufactured_action_really_is_a_no_op():
    """If the engine ever starts rejecting or consuming it, this test tells us before the
    guard tests start passing for the wrong reason."""
    game = BalatroGame(seed=SEED, ruleset="mlb")
    before = game.state_signature()
    game.step({"type": "use_consumable", "consumable_idx": 99, "target_cards": []})
    assert game.state_signature() == before


def test_guard_rescues_a_wedged_agent():
    game = BalatroGame(seed=SEED, ruleset="mlb")
    player = ShopSitter()
    forced = [0]
    status, steps = _drive_to_next_nemesis(game, player, max_steps=5_000, forced=forced)

    assert status in ("at_nemesis", "dead")
    assert steps < 5_000
    assert forced[0] >= 1
    # It kept trying: the guard fires once per shop visit, not once per run.
    assert player.noops >= NOOP_BUDGET_DEFAULT


def test_the_guard_leaves_no_trace_on_the_game():
    """`state_signature()` sweeps up every int/float/str/bool ATTRIBUTE of the game
    (game.py:923), so a diagnostic counter stored on the game silently changes the run's
    signature and breaks trajectory replay. It bit this workstream once; it does not get to
    bite it twice."""
    wedged = BalatroGame(seed=SEED, ruleset="mlb")
    _drive_to_next_nemesis(wedged, ShopSitter(), max_steps=5_000, forced=[0])
    clean = BalatroGame(seed=SEED, ruleset="mlb")
    engine_attrs = set(vars(clean))
    assert set(vars(wedged)) <= engine_attrs, (
        f"the driver added attributes to the game: {set(vars(wedged)) - engine_attrs}")


def test_guard_off_lets_the_wedge_happen():
    game = BalatroGame(seed=SEED, ruleset="mlb")
    with pytest.raises(RuntimeError, match="wedged"):
        _drive_to_next_nemesis(game, ShopSitter(), max_steps=200, noop_budget=0)


def test_guard_never_fires_for_a_population_that_makes_progress():
    """The guard must be a no-op for every agent the tournament already supports; this is
    the property that lets it default to ON."""
    players = default_population(6, base_seed=3)
    t = Tournament(seed=SEED, n_agents=6, players=players, life_rule="paired", max_ante=4)
    result = t.run()
    assert result.forced_progress == [0] * 6


def test_guarded_and_unguarded_runs_agree_when_nothing_wedges():
    """Determinism: computing a signature per step must not perturb the run."""
    def run(noop_budget):
        players = [make_scripted(name=f"s{i}", hand="greedy", rerolls_per_visit=i % 2)
                   for i in range(4)]
        t = Tournament(seed=SEED, n_agents=4, players=players, life_rule="paired",
                       max_ante=4, noop_budget=noop_budget)
        r = t.run()
        return ([g.state_signature() for g in t._last_games], r.final_lives,
                [list(m.scores) for m in r.ante_matrices])

    assert run(0) == run(NOOP_BUDGET_DEFAULT)


@pytest.mark.parametrize("state,expected", [
    (State.SHOP, "leave_shop"),
    (State.BOOSTER_OPEN, "skip_booster"),
    (State.BLIND_SELECT, "play_blind"),
    (State.ROUND_EVAL, "advance"),
])
def test_forced_progress_action_per_state(state, expected):
    game = BalatroGame(seed=SEED, ruleset="mlb")
    game.state = state
    assert _force_progress_action(game)["type"] == expected
