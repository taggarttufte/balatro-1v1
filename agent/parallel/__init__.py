"""
parallel — multi-process self-play with one shared batched evaluator (Phase 5 infra #1).

    main process                          worker process x N
    ------------------------------        ---------------------------------------
    MLBTrainer (net, optimizer,           its slice of the population: games, MCTS
      buffer, RNG, checkpoints)             trees, rngs, heuristic prior, skip cap,
    ParallelTournament (matrix,             sample collectors, trajectory loggers
      lives, ante lockstep)               NO net
    BatchEvaluator (one thread)           RemotePolicy -> shared memory -> evaluator

Import map:

    layout.py     byte layout of one leaf in a shared-memory arena (+ the measurements
                  that made shared memory the transport)
    channel.py    the arenas, the leaf queue, the reply pipes, the batching policy
    leaf.py       encode_leaf without a net (the worker's per-leaf numpy work)
    forward.py    the torch half, given already-encoded leaves
    remote.py     RemotePolicy: a PolicyValueFn whose net is in another process
    evaluator.py  BatchEvaluator: collects, batches per net, replies
    lockstep.py   LockstepDecider: one decide_many for a whole slice of the population
    worker.py     the worker process
    pool.py       WorkerPool + MPDriver + partition_agents
    protocol.py   the picklable messages

and the trainer that ties them together is ``train/parallel.py::ParallelMLBTrainer``.
See ``mp/agent/PARALLEL_NOTES.md``.
"""

__all__ = []
