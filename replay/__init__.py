"""
replay — trajectory logging + exact replay + tagging + viewer export (Phase 4 W3).

Engine-only package: never imports ``agent`` (torch) or anything from ``tournament``.
``engine/**``, ``rng/**``, ``agent/**``, ``tournament/**`` and ``eval/**`` are
FROZEN for this workstream -- everything here only *reads* the engine through
``oracle.engine_parity.import_engine()`` (the same fork-guarded entry point
``tournament/bootstrap.py`` and ``eval/conftest.py`` use).

See ``REPLAY_NOTES.md`` for the format spec, the hook contract callers wire in, the CLI, tag
definitions, viz-export coverage and the ghost-replay feasibility note.

Modules:
    _bootstrap   sys.path + fork-guarded engine import (BalatroGame, State, MLBMatch, ...)
    _util        shared helpers: per-step summaries, signature digests, synthetic-op dispatch
    log          TrajectoryLogger (single BalatroGame) + MatchLogger (MLBMatch)
    replay       replay() / replay_match() (exact re-run + signature assertion), narrate()
    tags         pure tag functions over a decoded JSONL line + tag_file()
    export_viz   best-effort V7-era `viz/` trajectory.json export (see export_viz.py)
    cli          `python -m replay.cli {show,verify,filter,stats}`
"""
from __future__ import annotations
