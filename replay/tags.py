"""
tags.py — pure functions over a decoded JSONL line (see log.py for the schema) plus
``tag_file()``, which retags a whole log in place.

Tag vocabulary (PHASE4_BRIEF §W3):
    win                  outcome["won"] is truthy (caller-supplied; see note below)
    reached_ante_{k}      k in MILESTONE_ANTES intersected with [1, max_ante_seen], PLUS
                          always the exact max ante reached (so both "give me anything that
                          got past ante 8" and "give me exactly ante 5" queries work)
    skip_heavy           fraction of play_blind/skip_blind decisions that were skips >= 0.5
    no_build             <= 1 joker owned at the final state
    comeback             lives hit 1 at some step, and a LATER step reached ante >= (that
                          step's ante) + 2 -- i.e. survived at least 2 more antes from the brink
    lives_lost_{n}        n = lives_start - final lives (MLB only, n > 0)
    archetype_novel       final joker-set signature's frequency rank (over the whole file,
                          computed by tag_file()) is outside the top ARCHETYPE_TOP_N -- a
                          per-line, single-line tag_episode() call cannot compute this (it is
                          a corpus-level property); it is only ever set by tag_file().
    interest_score       not a tag string but a float alongside tags; see interest_score().

"win" note: TrajectoryLogger.end()/MatchLogger.end() store whatever `outcome` dict the
caller passed verbatim; the `win` tag only fires if that dict has a truthy "won" key
(episode lines) or "winner" equal to the queried player (match lines) -- callers MUST pass
that if they want the tag (see REPLAY_NOTES.md "Hook contract").
"""
from __future__ import annotations

import json
import os
from collections import Counter
from typing import Optional

MILESTONE_ANTES = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 32)
ARCHETYPE_TOP_N = 20


# ============================================================================ episode tags

def _max_ante(line: dict) -> int:
    steps = line.get("steps") or []
    if steps:
        return max(s["ante"] for s in steps)
    return line.get("final_state", {}).get("ante", 0) or 0


def _skip_rate(line: dict) -> Optional[float]:
    actions = line.get("actions") or []
    plays = sum(1 for a in actions if a.get("type") == "play_blind")
    skips = sum(1 for a in actions if a.get("type") == "skip_blind")
    total = plays + skips
    if total == 0:
        return None
    return skips / total


def _comeback(line: dict) -> bool:
    steps = line.get("steps") or []
    for i, s in enumerate(steps):
        if s.get("lives") == 1:
            ante_then = s["ante"]
            for later in steps[i + 1:]:
                if later["ante"] >= ante_then + 2:
                    return True
    return False


def joker_signature(line: dict) -> tuple:
    """Sorted tuple of the final joker keys -- the "build" this episode ended with."""
    fs = line.get("final_state") or {}
    jokers = fs.get("jokers") or []
    return tuple(sorted(jokers))


def tag_episode(line: dict, archetype_rank: Optional[int] = None) -> list:
    """Tags for a ``kind == "episode"`` line (or a synthetic per-player sub-line built by
    ``tag_match``).  ``archetype_rank`` (1 = most common), when given, enables the
    ``archetype_novel`` tag -- pass it from ``tag_file()``'s corpus-wide count; omitted here
    it is simply not evaluated (a single line has no corpus to be novel against)."""
    tags: list = []
    outcome = line.get("outcome") or {}
    if outcome.get("won"):
        tags.append("win")

    max_ante = _max_ante(line)
    if max_ante > 0:
        for k in MILESTONE_ANTES:
            if k <= max_ante:
                tags.append(f"reached_ante_{k}")
        if max_ante not in MILESTONE_ANTES:
            tags.append(f"reached_ante_{max_ante}")

    rate = _skip_rate(line)
    if rate is not None and rate >= 0.5:
        tags.append("skip_heavy")

    fs = line.get("final_state") or {}
    if fs.get("joker_count", len(fs.get("jokers", []))) <= 1:
        tags.append("no_build")

    if _comeback(line):
        tags.append("comeback")

    lives_start = line.get("lives_start") or 0
    if lives_start > 0:
        final_lives = fs.get("lives", lives_start)
        n = lives_start - final_lives
        if n > 0:
            tags.append(f"lives_lost_{n}")

    if archetype_rank is not None and archetype_rank > ARCHETYPE_TOP_N:
        tags.append("archetype_novel")

    return tags


def interest_score(line: dict, tags: Optional[list] = None) -> float:
    """A simple heuristic (not a scientific score, see REPLAY_NOTES.md): rewards depth, a
    win, a comeback, build novelty and build DEPTH (distinct jokers), penalises skip-heavy
    and no-build lines.  Always >= 0."""
    if tags is None:
        tags = tag_episode(line)
    tagset = set(tags)
    max_ante = _max_ante(line)
    fs = line.get("final_state") or {}
    joker_count = fs.get("joker_count", len(fs.get("jokers", [])))

    score = min(max_ante, 24) / 24.0
    if "win" in tagset:
        score += 1.0
    if "comeback" in tagset:
        score += 0.5
    if "archetype_novel" in tagset:
        score += 0.5
    score += 0.3 * min(joker_count, 5) / 5.0
    if "skip_heavy" in tagset:
        score -= 0.3
    if "no_build" in tagset:
        score -= 0.2
    return max(0.0, score)


# ============================================================================ match tags

def _episode_view_of_match(line: dict, player: int) -> dict:
    """Project a ``kind == "match"`` line down to the ``kind == "episode"`` shape
    ``tag_episode``/``interest_score`` expect, from ``player``'s perspective."""
    actions = [op["action"] for op in (line.get("ops") or []) if op["player"] == player]
    steps = [s["players"][player] for s in (line.get("steps") or [])]
    outcome_in = line.get("outcome") or {}
    winner = outcome_in.get("winner")
    outcome = dict(outcome_in)
    outcome["won"] = (winner == player)
    fs_players = (line.get("final_state") or {}).get("players") or [{}, {}]
    fs = dict(fs_players[player]) if player < len(fs_players) else {}
    fs.setdefault("joker_count", len(fs.get("jokers", [])))
    return {
        "kind": "episode",
        "seed": line.get("seed"),
        "deck_key": line.get("deck_key"),
        "stake": line.get("stake"),
        "ruleset": "mlb",
        "lives_start": line.get("lives_start"),
        "actions": actions,
        "steps": steps,
        "outcome": outcome,
        "final_state": fs,
    }


def tag_match(line: dict, player: int, archetype_rank: Optional[int] = None) -> list:
    """Tags for one player's side of a ``kind == "match"`` line."""
    return tag_episode(_episode_view_of_match(line, player), archetype_rank=archetype_rank)


def interest_score_match(line: dict, player: int, tags: Optional[list] = None) -> float:
    return interest_score(_episode_view_of_match(line, player), tags=tags)


# ============================================================================ tag_file

def _line_signature(line: dict) -> tuple:
    if line.get("kind") == "match":
        # both sides' builds count toward the corpus-wide archetype counter
        return None  # handled specially below (two signatures per line)
    return joker_signature(line)


def tag_file(path: str) -> dict:
    """Retag every line in ``path`` IN PLACE (atomic replace via a temp file): recomputes
    every pure tag, builds the corpus-wide joker-archetype frequency table across the whole
    file (both players' builds, for match lines), and sets ``archetype_novel`` +
    ``interest_score`` from it.  Returns ``{"total", "retagged"}``."""
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if raw:
                lines.append(json.loads(raw))

    counter: Counter = Counter()
    for line in lines:
        if line.get("kind") == "match":
            for p in (0, 1):
                counter[joker_signature(_episode_view_of_match(line, p))] += 1
        else:
            counter[joker_signature(line)] += 1

    # rank 1 = most common; ties broken by signature repr for a total, deterministic order
    ranked = sorted(counter.items(), key=lambda kv: (-kv[1], repr(kv[0])))
    rank_of = {sig: i + 1 for i, (sig, _cnt) in enumerate(ranked)}

    out_lines = []
    for line in lines:
        if line.get("kind") == "match":
            tags = {}
            scores = {}
            for p in (0, 1):
                view = _episode_view_of_match(line, p)
                rank = rank_of.get(joker_signature(view))
                t = tag_episode(view, archetype_rank=rank)
                tags[str(p)] = t
                scores[str(p)] = interest_score(view, tags=t)
            line["tags"] = tags
            line["interest_score"] = scores
        else:
            rank = rank_of.get(joker_signature(line))
            t = tag_episode(line, archetype_rank=rank)
            line["tags"] = t
            line["interest_score"] = interest_score(line, tags=t)
        out_lines.append(line)

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for line in out_lines:
            f.write(json.dumps(line, sort_keys=True) + "\n")
    os.replace(tmp, path)
    return {"total": len(out_lines), "retagged": len(out_lines)}
