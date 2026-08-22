"""Design doc §6: 'the population must be heterogeneous or the matrix degenerates.'  N
identical deterministic policies on one seed play byte-identical trajectories in every
independent game (same seed, no divergent decisions) -- tie_fraction must hit its ceiling.
A heterogeneous population must show meaningfully more information (lower tie_fraction)."""
from ..players import identical_population, default_population
from ..runner import Tournament


def _tie_fractions(players, n, life_rule="none", max_ante=5, seed="7I4M53DL"):
    t = Tournament(seed=seed, n_agents=n, players=players, life_rule=life_rule, max_ante=max_ante)
    res = t.run()
    return [m.tie_fraction for m in res.ante_matrices]


def test_identical_scripted_population_is_fully_degenerate():
    n = 12
    fracs = _tie_fractions(identical_population(n), n)
    assert fracs, "expected at least one Nemesis"
    # deterministic identical policy on one seed -> every agent's game is bit-identical ->
    # every pairwise comparison at every ante is an exact tie.
    assert all(f == 1.0 for f in fracs)


def test_heterogeneous_population_is_not_degenerate():
    n = 12
    fracs = _tie_fractions(default_population(n, base_seed=7), n)
    assert fracs
    assert all(f < 1.0 for f in fracs)
    # meaningfully below the identical-population ceiling, not just "not exactly 1"
    assert min(fracs) < 0.5


def test_heterogeneous_beats_identical_on_the_same_seed_and_horizon():
    n = 10
    het = _tie_fractions(default_population(n, base_seed=1), n)
    ident = _tie_fractions(identical_population(n), n)
    assert len(het) == len(ident)
    for h, i in zip(het, ident):
        assert h < i
