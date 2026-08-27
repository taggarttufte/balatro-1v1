"""
ghost — the play-against-the-agent product (G1: static ghost race).

Turns a logged ``MLBMatch`` (replay/log.py's ``kind: "match"`` JSONL line) into a
ghost-replay ``.json`` the installed BalatroMultiplayer mod plays back natively in its
Practice-mode ghost race: same seed, MLB life rules, the agent's Nemesis scores ticking
up on the real HUD.  No Lua is shipped in G1 — the mod's own ghost engine
(``$MOD/lib/ghost_replay.lua``, read-only reference, never copied) is the whole runtime.

    python -m ghost.make --spec ev:fast          # self-play a fresh seed, install the ghost
    python -m ghost.export <log.jsonl> <idx>     # convert an existing logged match

Design brief: docs/GHOST_MOD_BRIEF_2026-08.md.  Notes: ghost/GHOST_NOTES.md.
"""
