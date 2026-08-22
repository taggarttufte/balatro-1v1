"""
mp/engine — self-contained fork of the Balatro simulator for the multiplayer (MLB) line.

Contents:
    balatro_sim/   engine fork (balatro-rl @ 4411dbf, branch fix/sim-fidelity-2026-07,
                   plus clone()/legal_actions() ported from balatro-mcts)
    tests/         the engine's test suites, relocated (engine_tests/ = repo tests/,
                   sim_tests/ = balatro_sim/tests/ + the balatro-mcts clone tests)
    benchmarks/    bench_clone_step.py (clone vs deepcopy vs step throughput)
    FORK_NOTES.md  provenance, what was ported, test counts, inherited fidelity issues

Import the fork by putting this directory on sys.path (conftest.py / pytest.ini do
this for the test suite):

    sys.path.insert(0, "<repo>/mp/engine")
    from balatro_sim.game import BalatroGame

Phase 1 of the MP campaign threads keyed RNG through this copy; the top-level
balatro_sim/ (BRL) is left untouched.
"""
