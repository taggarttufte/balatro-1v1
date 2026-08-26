"""
tournament — N-agent same-seed tournament runner + N x N outcome matrix.

Phase 3 W2 (2026-08-21).  See TOURNAMENT_NOTES.md for the interleaving contract, how lives
map, file formats and wall-clock numbers.  engine/** and rng/** are FROZEN for this
package: it only imports them (through bootstrap.py's fork-guarded import_engine()), never
edits them.
"""
