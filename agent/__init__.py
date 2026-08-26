"""
agent — AlphaZero-style MCTS agent layer for the MP (Major League Balatro) line.

Forked 2026-08-21 from ``C:/Users/Taggart/projects/recovered/balatro-mcts`` @ ``ee75d11``
(read-only source) and re-targeted onto the frozen fork engine in ``engine``.

Contents:
    mcts/        search core: action keys, node, encoder, action features, model,
                 policy/value interface (batchable), outcome signals, MCTS, MCTSPlayer
    train/       self-play agent, replay buffer, trainer, cold-start loop, checkpointing
    parallel/    multi-process self-play: worker processes + ONE shared batched evaluator
                 (Phase 5 W1; see PARALLEL_NOTES.md)
    scripts/     mcts_demo.py, smoke_selfplay.py, train_cold.py, train_mlb.py
    benchmarks/  bench_search.py (single-tree sims/sec baseline; W3 extends)
    tests/       the forked MCTS tests + checkpoint round-trip + MLB smoke
    AGENT_NOTES.md  provenance, re-target diff, measured numbers, the interface W3 gets
    PARALLEL_NOTES.md  the multi-process architecture, transport, determinism contract,
                 the throughput benchmark and the swap procedure for a live run

This directory is a sys.path root, not an importable package path: put it AND
``engine`` on sys.path and import ``mcts`` / ``train`` / ``balatro_sim`` top-level
(pytest.ini + conftest.py do this for the suite; scripts/_bootstrap.py for the scripts).
"""
