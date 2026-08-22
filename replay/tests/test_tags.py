"""
tags.py behaviour on constructed lines (pure functions -- no engine needed for most of
these) plus tag_file()'s in-place retagging + corpus-wide archetype novelty, and a couple of
tags computed over a REAL logged+replayed random-legal episode as a sanity cross-check.
"""
from __future__ import annotations

import json

from ..log import TrajectoryLogger
from ..replay import load_lines
from ..tags import (
    ARCHETYPE_TOP_N, interest_score, joker_signature, tag_episode, tag_file, tag_match,
)
from ._helpers import SEEDS, run_logged_episode


def _line(**overrides) -> dict:
    base = {
        "kind": "episode",
        "seed": "TEST0001",
        "deck_key": "b_red",
        "stake": 1,
        "ruleset": "mlb",
        "lives_start": 4,
        "actions": [{"type": "skip_blind"}, {"type": "play_blind"}],
        "steps": [
            {"step": 0, "state": "BLIND_SELECT", "ante": 1, "blind_kind": "Small",
             "is_pvp": False, "money": 4, "lives": 4, "chips_scored": 0,
             "hands_left": 4, "discards_left": 3},
            {"step": 1, "state": "BLIND_SELECT", "ante": 1, "blind_kind": "Big",
             "is_pvp": False, "money": 4, "lives": 4, "chips_scored": 0,
             "hands_left": 4, "discards_left": 3},
        ],
        "outcome": {},
        "final_state": {"state": "GAME_OVER", "ante": 1, "lives": 4, "money": 4,
                        "joker_count": 0, "jokers": [], "consumables": []},
    }
    base.update(overrides)
    return base


# ============================================================================ individual tags

def test_win_tag_requires_outcome_won():
    assert "win" not in tag_episode(_line())
    assert "win" in tag_episode(_line(outcome={"won": True}))
    assert "win" not in tag_episode(_line(outcome={"won": False}))


def test_reached_ante_milestones_and_exact():
    steps = [
        {"step": i, "state": "X", "ante": ante, "blind_kind": "Small", "is_pvp": False,
         "money": 0, "lives": 4, "chips_scored": 0, "hands_left": 4, "discards_left": 3}
        for i, ante in enumerate([1, 2, 3, 5])
    ]
    tags = tag_episode(_line(steps=steps, final_state={"ante": 5, "lives": 4, "joker_count": 0,
                                                        "jokers": []}))
    assert "reached_ante_1" in tags
    assert "reached_ante_2" in tags
    assert "reached_ante_3" in tags
    assert "reached_ante_5" in tags   # exact, not a milestone
    # "reached" means "got at least this far", not "was seen in `steps`" -- ante 4 is a
    # milestone <= max_ante (5) even though this synthetic `steps` list happens to skip it
    # (antes always increment by exactly 1 in the real engine; this is just a fixture).
    assert "reached_ante_4" in tags
    assert "reached_ante_8" not in tags


def test_skip_heavy():
    heavy = _line(actions=[{"type": "skip_blind"}, {"type": "skip_blind"}, {"type": "play_blind"}])
    light = _line(actions=[{"type": "play_blind"}, {"type": "play_blind"}, {"type": "skip_blind"}])
    assert "skip_heavy" in tag_episode(heavy)
    assert "skip_heavy" not in tag_episode(light)


def test_no_build():
    zero = _line(final_state={"ante": 1, "lives": 4, "joker_count": 0, "jokers": []})
    one = _line(final_state={"ante": 1, "lives": 4, "joker_count": 1, "jokers": ["j_joker"]})
    two = _line(final_state={"ante": 1, "lives": 4, "joker_count": 2,
                              "jokers": ["j_joker", "j_greedy_joker"]})
    assert "no_build" in tag_episode(zero)
    assert "no_build" in tag_episode(one)
    assert "no_build" not in tag_episode(two)


def test_comeback_requires_two_more_antes_survived():
    rising = [
        {"step": 0, "state": "X", "ante": 3, "blind_kind": "Small", "is_pvp": False,
         "money": 0, "lives": 1, "chips_scored": 0, "hands_left": 4, "discards_left": 3},
        {"step": 1, "state": "X", "ante": 5, "blind_kind": "Small", "is_pvp": False,
         "money": 0, "lives": 2, "chips_scored": 0, "hands_left": 4, "discards_left": 3},
    ]
    assert "comeback" in tag_episode(_line(steps=rising))

    barely = [
        {"step": 0, "state": "X", "ante": 3, "blind_kind": "Small", "is_pvp": False,
         "money": 0, "lives": 1, "chips_scored": 0, "hands_left": 4, "discards_left": 3},
        {"step": 1, "state": "X", "ante": 4, "blind_kind": "Small", "is_pvp": False,
         "money": 0, "lives": 0, "chips_scored": 0, "hands_left": 4, "discards_left": 3},
    ]
    assert "comeback" not in tag_episode(_line(steps=barely))


def test_lives_lost_n():
    lost_two = _line(lives_start=4, final_state={"ante": 3, "lives": 2, "joker_count": 0,
                                                  "jokers": []})
    lost_none = _line(lives_start=4, final_state={"ante": 3, "lives": 4, "joker_count": 0,
                                                   "jokers": []})
    assert "lives_lost_2" in tag_episode(lost_two)
    assert not any(t.startswith("lives_lost_") for t in tag_episode(lost_none))


def test_vanilla_line_has_no_lives_lost_tag():
    """lives_start == 0 (vanilla) must never emit a lives_lost_* tag."""
    vanilla = _line(ruleset="vanilla", lives_start=0,
                     final_state={"ante": 3, "lives": 0, "joker_count": 0, "jokers": []})
    assert not any(t.startswith("lives_lost_") for t in tag_episode(vanilla))


def test_interest_score_is_nonnegative_and_rewards_win():
    plain = _line()
    win = _line(outcome={"won": True})
    assert interest_score(plain) >= 0.0
    assert interest_score(win) > interest_score(plain)


# ============================================================================ match tags

def test_tag_match_projects_each_players_perspective():
    match_line = {
        "kind": "match",
        "seed": "TEST0001", "deck_key": "b_red", "stake": 1, "lives_start": 4,
        "ops": [{"player": 0, "action": {"type": "skip_blind"}},
                {"player": 1, "action": {"type": "play_blind"}}],
        "steps": [
            {"step": 0, "player": 0, "players": [
                {"step": 0, "state": "X", "ante": 1, "blind_kind": "Small", "is_pvp": False,
                 "money": 4, "lives": 4, "chips_scored": 0, "hands_left": 4, "discards_left": 3},
                {"step": 0, "state": "X", "ante": 1, "blind_kind": "Small", "is_pvp": False,
                 "money": 4, "lives": 4, "chips_scored": 0, "hands_left": 4, "discards_left": 3},
            ]},
        ],
        "outcome": {"winner": 0},
        "final_state": {"winner": 0, "players": [
            {"lives": 4, "ante": 8, "money": 10, "joker_count": 0, "jokers": []},
            {"lives": 0, "ante": 6, "money": 0, "joker_count": 0, "jokers": []},
        ]},
    }
    tags0 = tag_match(match_line, 0)
    tags1 = tag_match(match_line, 1)
    assert "win" in tags0
    assert "win" not in tags1


# ============================================================================ tag_file

def test_tag_file_retags_in_place_and_flags_novelty(tmp_path):
    path = str(tmp_path / "log.jsonl")
    # ARCHETYPE_TOP_N + a few extra distinct single-joker builds so at least one is
    # guaranteed to fall outside the top N by construction.
    n_lines = ARCHETYPE_TOP_N + 5
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n_lines):
            line = _line(final_state={"ante": 1, "lives": 4, "joker_count": 1,
                                       "jokers": [f"j_unique_{i}"]})
            f.write(json.dumps(line) + "\n")

    result = tag_file(path)
    assert result["total"] == n_lines

    lines = load_lines(path)
    assert all(isinstance(l["tags"], list) for l in lines)
    assert all(isinstance(l["interest_score"], float) for l in lines)
    # every signature here is unique (count 1 each) -> a total rank order 1..n_lines ->
    # exactly (n_lines - ARCHETYPE_TOP_N) fall outside the top N and are flagged novel.
    novel_count = sum(1 for l in lines if "archetype_novel" in l["tags"])
    assert novel_count == n_lines - ARCHETYPE_TOP_N


def test_tag_file_common_archetype_is_not_novel(tmp_path):
    """Novelty (rank > ARCHETYPE_TOP_N) only means anything once there are MORE than
    ARCHETYPE_TOP_N distinct archetypes in the file, so this builds exactly that: one very
    common build (rank 1), ARCHETYPE_TOP_N-1 "filler" builds common enough to fill out the
    rest of the top N, and 3 singleton builds guaranteed to rank outside it."""
    path = str(tmp_path / "log.jsonl")
    lines = []
    for _ in range(50):
        lines.append(_line(final_state={"ante": 1, "lives": 4, "joker_count": 1,
                                         "jokers": ["j_joker"]}))
    for i in range(ARCHETYPE_TOP_N - 1):
        for _ in range(5):
            lines.append(_line(final_state={"ante": 1, "lives": 4, "joker_count": 1,
                                             "jokers": [f"j_filler_{i}"]}))
    for i in range(3):
        lines.append(_line(final_state={"ante": 1, "lives": 4, "joker_count": 1,
                                         "jokers": [f"j_rare_{i}"]}))
    with open(path, "w", encoding="utf-8") as f:
        for l in lines:
            f.write(json.dumps(l) + "\n")

    tag_file(path)
    retagged = load_lines(path)
    common = [l for l in retagged if l["final_state"]["jokers"] == ["j_joker"]]
    filler = [l for l in retagged if l["final_state"]["jokers"][0].startswith("j_filler_")]
    rare = [l for l in retagged if l["final_state"]["jokers"][0].startswith("j_rare_")]
    assert all("archetype_novel" not in l["tags"] for l in common)
    assert all("archetype_novel" not in l["tags"] for l in filler)
    assert all("archetype_novel" in l["tags"] for l in rare)


# ============================================================================ real logged episode

def test_tags_over_a_real_logged_episode(tmp_path):
    path = str(tmp_path / "log.jsonl")
    run_logged_episode(path, SEEDS[0], ruleset="mlb", max_steps=500, player_seed=11)
    line = load_lines(path)[0]
    tags = tag_episode(line)
    assert isinstance(tags, list)
    assert isinstance(interest_score(line), float)
    assert isinstance(joker_signature(line), tuple)
