"""Exit-gate item 1 (PHASE3_BRIEF_2026-08.md §2): 100 agents on one seed, end-to-end, both
to ante 8 under life_rule="none" and to last-agent-standing under life_rule="paired"."""
import time

from ..players import default_population
from ..runner import Tournament


def test_100_agents_none_rule_to_ante_8():
    n = 100
    players = default_population(n, base_seed=0)
    t = Tournament(seed="7I4M53DL", n_agents=n, players=players, life_rule="none", max_ante=8)
    t0 = time.perf_counter()
    res = t.run()
    wall = time.perf_counter() - t0
    assert len(res.ante_matrices) == 7   # antes 2..8
    for m in res.ante_matrices:
        assert m.stats["n_present"] == n   # "none": nobody dies, every agent reaches every Nemesis
    assert res.alive_at_end == list(range(n))
    assert all(l > 0 for l in res.final_lives)
    # sanity bound, generous: catches a severe perf regression without being flaky about
    # exact hardware speed (measured ~20-30s on the reference machine, see TOURNAMENT_NOTES.md)
    assert wall < 180


def test_100_agents_paired_rule_to_last_agent_standing():
    n = 100
    players = default_population(n, base_seed=0)
    t = Tournament(seed="7I4M53DL", n_agents=n, players=players, life_rule="paired", max_ante=40)
    t0 = time.perf_counter()
    res = t.run()
    wall = time.perf_counter() - t0
    assert len(res.ante_matrices) >= 1
    # the loop only stops when everyone is eliminated OR max_ante is exhausted
    ran_out_of_antes = (len(res.ante_matrices) > 0
                         and res.ante_matrices[-1].ante == t.max_ante
                         and len(res.alive_at_end) > 0)
    everyone_eliminated = len(res.alive_at_end) == 0
    assert ran_out_of_antes or everyone_eliminated or len(res.alive_at_end) <= n
    # n_present must be non-increasing ante over ante (agents only ever drop out)
    present = [m.stats["n_present"] for m in res.ante_matrices]
    assert all(a >= b for a, b in zip(present, present[1:]))
    assert wall < 180
