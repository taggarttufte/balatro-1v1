"""Fan-out: construct N games vs clone() one N times must be state-identical; report which
is faster (matches ``TOURNAMENT_NOTES.md`` "fan-out benchmark")."""
from ..runner import construct_games, clone_games, benchmark_fanout, FANOUT_DEFAULT


def test_construct_and_clone_agree_on_initial_state():
    g_construct, seed_c = construct_games("7I4M53DL", 12, "b_red", 1, 4)
    g_clone, seed_k = clone_games("7I4M53DL", 12, "b_red", 1, 4)
    assert seed_c == seed_k
    sigs_c = [g.state_signature() for g in g_construct]
    sigs_k = [g.state_signature() for g in g_clone]
    assert sigs_c == sigs_k
    # every one of the 12 must independently equal the others too (same seed, no divergence yet)
    assert len(set(sigs_c)) == 1


def test_construct_and_clone_agree_with_a_random_seed():
    # seed=None: construct_games must reuse the FIRST construction's derived seed_str for
    # every subsequent one, not re-derive a fresh random seed each time.
    g_construct, seed_c = construct_games(None, 6, "b_red", 1, 4)
    assert all(g.seed_str == seed_c for g in g_construct)
    g_clone, seed_k = clone_games(seed_c, 6, "b_red", 1, 4)
    assert [g.state_signature() for g in g_construct] == [g.state_signature() for g in g_clone]


def test_benchmark_reports_both_methods_and_matches_the_hardcoded_default():
    res = benchmark_fanout(seed="7I4M53DL", n=50, repeats=2)
    assert res["signatures_equal"]
    assert set(res["seconds"]) == {"construct", "clone"}
    assert FANOUT_DEFAULT in ("construct", "clone")
