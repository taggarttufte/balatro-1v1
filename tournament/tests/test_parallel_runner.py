"""
Phase 5 W1: ``parallel.py`` — the lockstep form of ``Tournament.run`` and its driver seam.

The load-bearing claim is that this is a REFACTOR, not a rewrite: ``ParallelTournament``
driven by an in-process ``LocalDriver`` must reproduce ``Tournament.run`` exactly — the
same actions in the same per-agent order, the same no-progress guard, the same matrices,
the same lives, the same final ``state_signature()`` for every game, the same trajectory
hooks in the same order.  Only the interleaving across agents changes, and nothing in a
``BalatroGame`` or a ``Player`` is shared between agents.

No torch here: the populations are the scripted / random-legal adapters, so this file runs
wherever the rest of ``mp/tournament`` does.  The MCTS side of the same claim (a lockstep
``decide_many`` that batches leaves) is pinned in ``mp/agent/tests/test_parallel.py``.
"""
import numpy as np
import pytest

from ..bootstrap import State
from ..parallel import (
    AgentDrive, DriveOutcome, LocalDriver, ParallelTournament, drive_many, serial_decide,
)
from ..players import RandomLegalPlayer, default_population, identical_population
from ..runner import Tournament, _drive_to_next_nemesis, clone_games

SEED = "7I4M53DL"


def _assert_same_run(a, games_a, b, games_b):
    assert a.seed == b.seed
    assert a.steps_total == b.steps_total
    assert a.final_lives == b.final_lives
    assert a.alive_at_end == b.alive_at_end
    assert a.forced_progress == b.forced_progress
    assert a.last_score == b.last_score
    assert len(a.ante_matrices) == len(b.ante_matrices)
    for m1, m2 in zip(a.ante_matrices, b.ante_matrices):
        assert m1.ante == m2.ante
        assert np.array_equal(m1.scores, m2.scores, equal_nan=True)
        assert np.array_equal(m1.outcome, m2.outcome, equal_nan=True)
        assert np.array_equal(m1.rank, m2.rank, equal_nan=True)
        assert m1.losers == m2.losers
    assert ([g.state_signature() for g in games_a]
            == [g.state_signature() for g in games_b])


def _both(n, players_fn, **kw):
    serial = Tournament(seed=SEED, n_agents=n, players=players_fn(), **kw)
    r1 = serial.run()
    driver = LocalDriver(players_fn())
    lock = ParallelTournament(seed=SEED, n_agents=n, players=[None] * n, driver=driver, **kw)
    r2 = lock.run()
    return r1, serial._last_games, r2, driver.games


# ══════════════════════════════════════════════════════════ the refactor is a refactor

@pytest.mark.parametrize("life_rule", ["paired", "median", "none"])
def test_lockstep_matches_the_serial_tournament_for_every_life_rule(life_rule):
    _assert_same_run(*_both(8, lambda: default_population(8, base_seed=11),
                            life_rule=life_rule, max_ante=6))


@pytest.mark.parametrize("fanout", ["clone", "construct"])
def test_lockstep_matches_the_serial_tournament_for_both_fanouts(fanout):
    _assert_same_run(*_both(6, lambda: default_population(6, base_seed=3),
                            life_rule="paired", max_ante=5, fanout=fanout))


def test_lockstep_matches_the_serial_tournament_for_an_identical_population():
    """The degenerate case (all ties) exercises the life rule's tie branch and the
    matrix's ``tie_fraction``, both of which live on the main-process side of the seam."""
    _assert_same_run(*_both(4, lambda: identical_population(4), life_rule="paired",
                            max_ante=4))


def test_lockstep_matches_the_serial_tournament_with_an_odd_population():
    """An odd ``n_agents`` has a rotating partner whose opponent depends on the ROUND
    index — a cross-agent decision that has to stay in the main process."""
    _assert_same_run(*_both(7, lambda: default_population(7, base_seed=5),
                            life_rule="paired", max_ante=5))


# ══════════════════════════════════════════════════════════ drive_many vs the serial driver

def test_drive_many_reproduces_drive_to_next_nemesis_for_one_agent():
    games_a, _ = clone_games(SEED, 1, "b_red", 1, 4)
    games_b, _ = clone_games(SEED, 1, "b_red", 1, 4)
    pa, pb = RandomLegalPlayer(seed=9), RandomLegalPlayer(seed=9)

    status_a, steps_a = _drive_to_next_nemesis(games_a[0], pa)
    out = drive_many([AgentDrive(idx=0, game=games_b[0], player=pb)])
    status_b, steps_b, forced_b = out[0]

    assert (status_a, steps_a) == (status_b, steps_b)
    assert games_a[0].state_signature() == games_b[0].state_signature()
    assert forced_b == 0


def test_drive_many_returns_immediately_when_already_at_a_nemesis():
    """Same contract as the serial driver: it must be cashed out first, not driven."""
    games, _ = clone_games(SEED, 1, "b_red", 1, 4)
    game = games[0]
    _drive_to_next_nemesis(game, RandomLegalPlayer(seed=1))
    assert game.state == State.ROUND_EVAL and game.current_blind.is_pvp
    out = drive_many([AgentDrive(idx=0, game=game, player=RandomLegalPlayer(seed=1))])
    assert out[0] == ("at_nemesis", 0, 0)


def test_agent_drive_honours_max_steps():
    games, _ = clone_games(SEED, 1, "b_red", 1, 4)
    d = AgentDrive(idx=0, game=games[0], player=RandomLegalPlayer(seed=1), max_steps=2)
    with pytest.raises(RuntimeError, match="wedged"):
        drive_many([d])


def test_agent_drive_with_the_guard_disabled_takes_no_signatures():
    """``noop_budget=0`` is the documented off switch; it must not silently cost the
    per-step ``state_signature()`` (42 us) anyway."""
    games, _ = clone_games(SEED, 1, "b_red", 1, 4)
    d = AgentDrive(idx=0, game=games[0], player=RandomLegalPlayer(seed=1), noop_budget=0)
    assert d._sig is None
    drive_many([d])
    assert d._sig is None and d.forced == 0


# ══════════════════════════════════════════════════════════ the trajectory hooks

def _record_hooks(n, parallel: bool):
    steps, dones, fanouts = [], [], []
    kw = dict(life_rule="paired", max_ante=5)
    if parallel:
        driver = LocalDriver(default_population(n, base_seed=2),
                             on_step=lambda i, g, a: steps.append((i, a.get("type"))),
                             on_agent_done=lambda i, g, r: dones.append((i, r)),
                             on_fanout=lambda games, s: fanouts.append(s))
        ParallelTournament(seed=SEED, n_agents=n, players=[None] * n, driver=driver,
                           **kw).run()
    else:
        Tournament(seed=SEED, n_agents=n, players=default_population(n, base_seed=2),
                   on_step=lambda i, g, a: steps.append((i, a.get("type"))),
                   on_agent_done=lambda i, g, r: dones.append((i, r)),
                   on_fanout=lambda games, s: fanouts.append(s), **kw).run()
    return steps, dones, fanouts


def test_the_hooks_fire_the_same_way_in_both_paths():
    """W3's ``mp/replay`` logger hangs off these three; a per-agent stream has to be the
    same stream, and every agent must still be closed exactly once."""
    n = 6
    s1, d1, f1 = _record_hooks(n, parallel=False)
    s2, d2, f2 = _record_hooks(n, parallel=True)
    assert f1 == f2
    assert sorted(d1) == sorted(d2)
    assert len(d1) == n and len({i for i, _ in d1}) == n
    for i in range(n):
        assert [t for j, t in s1 if j == i] == [t for j, t in s2 if j == i]
    assert sorted(s1) == sorted(s2)


def test_lose_life_is_reported_to_the_step_hook_in_the_parallel_path():
    """The one life lost without a ``step()`` (REPLAY_NOTES §2.3) still has to appear."""
    from ..runner import OP_LOSE_LIFE
    seen = []
    n = 4
    driver = LocalDriver(default_population(n, base_seed=7),
                         on_step=lambda i, g, a: seen.append(a.get("type")))
    ParallelTournament(seed=SEED, n_agents=n, players=[None] * n, driver=driver,
                       life_rule="paired", max_ante=5).run()
    assert OP_LOSE_LIFE in seen


# ══════════════════════════════════════════════════════════ the driver seam itself

def test_local_driver_can_own_a_subset_of_the_agents():
    """What a worker process does: build only the games it was assigned, and answer about
    them under their GLOBAL indices."""
    players = default_population(6, base_seed=4)
    mine = [1, 3, 5]
    driver = LocalDriver([players[i] for i in mine], indices=mine)
    driver.setup(SEED, 6, "b_red", 1, 4)
    assert sorted(driver.summarize()) == mine
    out = driver.drive([0, 1, 2, 3, 4, 5], 20_000, 8)
    assert sorted(out) == mine                      # it answers only for what it owns
    assert driver.apply([("cash_out", 0)]) == {}    # and ignores what it does not
    assert driver.game(1) is not None and driver.game(0) is None


def test_subset_drivers_together_reproduce_one_driver():
    """Two ``LocalDriver``s splitting the population must play the same tournament a
    single one does — the property the multiprocess driver depends on."""
    n = 6
    players = default_population(n, base_seed=8)
    whole = LocalDriver(default_population(n, base_seed=8))
    r1 = ParallelTournament(seed=SEED, n_agents=n, players=[None] * n, driver=whole,
                            life_rule="paired", max_ante=5).run()

    left, right = [0, 2, 4], [1, 3, 5]
    parts = {"a": LocalDriver([players[i] for i in left], indices=left),
             "b": LocalDriver([players[i] for i in right], indices=right)}
    split = _SplitDriver(list(parts.values()), n)
    r2 = ParallelTournament(seed=SEED, n_agents=n, players=[None] * n, driver=split,
                            life_rule="paired", max_ante=5).run()
    _assert_same_run(r1, whole.games, r2, [split.game(i) for i in range(n)])


class _SplitDriver:
    """The in-process shape of ``mp/agent``'s ``MPDriver``: fan every call out to the
    drivers that own the agents, merge the answers."""

    def __init__(self, drivers, n_agents):
        self.drivers = list(drivers)
        self.n_agents = n_agents

    def setup(self, seed, n_agents, deck_key, stake, lives, ruleset="mlb", fanout="clone"):
        seeds = {d.setup(seed, n_agents, deck_key, stake, lives, ruleset, fanout)
                 for d in self.drivers}
        assert len(seeds) == 1
        return seeds.pop()

    def drive(self, indices, max_steps, noop_budget):
        out = {}
        for d in self.drivers:
            out.update(d.drive(indices, max_steps, noop_budget))
        return out

    def apply(self, ops):
        out = {}
        for d in self.drivers:
            out.update(d.apply(ops))
        return out

    def summarize(self):
        out = {}
        for d in self.drivers:
            out.update(d.summarize())
        return out

    def game(self, idx):
        for d in self.drivers:
            g = d.game(idx)
            if g is not None:
                return g
        return None

    def close(self):
        pass


def test_a_crashed_agent_leaves_the_tournament_and_the_rest_continue():
    """``status="crashed"`` (a worker process died) is handled exactly like a death: the
    agent is out, its last known lives are recorded, the matrix is built from the rest."""
    n = 4
    driver = _CrashingDriver(default_population(n, base_seed=6), crash={2})
    result = ParallelTournament(seed=SEED, n_agents=n, players=[None] * n, driver=driver,
                                life_rule="paired", max_ante=5).run()
    assert result.crashed == [2]
    assert result.final_lives[2] is not None
    assert 2 not in result.alive_at_end
    assert all(np.isnan(m.scores[2]) for m in result.ante_matrices)
    assert any(not np.isnan(m.scores[0]) for m in result.ante_matrices)
    assert ("crashed") in [r for _i, r in driver.finished]


class _CrashingDriver(LocalDriver):
    """A ``LocalDriver`` that reports the chosen agents as ``crashed`` on the first
    drive, without touching their games — what the pool does for a dead worker."""

    def __init__(self, players, crash):
        super().__init__(players, on_agent_done=self._note)
        self.crash = set(crash)
        self.finished = []

    def _note(self, i, game, reason):
        self.finished.append((i, reason))

    def drive(self, indices, max_steps, noop_budget):
        out = super().drive([i for i in indices if i not in self.crash], max_steps,
                            noop_budget)
        for i in indices:
            if i in self.crash:
                out[i] = DriveOutcome(status="crashed", lives=3, ante=1)
        return out


# ══════════════════════════════════════════════════════════ serial_decide is the default

def test_serial_decide_is_the_default_and_asks_every_player_once():
    calls = []

    class Counting(RandomLegalPlayer):
        def act(self, game):
            calls.append(self.seed)
            return super().act(game)

    players = [Counting(seed=i) for i in range(3)]
    games, _ = clone_games(SEED, 3, "b_red", 1, 4)
    items = [(i, games[i], players[i]) for i in range(3)]
    actions = serial_decide(items)
    assert len(actions) == 3 and calls == [0, 1, 2]
