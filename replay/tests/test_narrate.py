"""narrate() must be non-empty and mention every ante actually reached, for both an episode
line and a match line."""
from __future__ import annotations

import re

from ..replay import load_line, load_lines, narrate
from ._helpers import SEEDS, run_logged_episode, run_logged_match


def _antes_mentioned(text: str) -> set:
    return {int(m) for m in re.findall(r"Ante (\d+)", text)}


def test_narrate_episode_nonempty_and_mentions_every_ante(tmp_path):
    path = str(tmp_path / "log.jsonl")
    run_logged_episode(path, SEEDS[1], ruleset="mlb", max_steps=600, player_seed=21)
    line = load_line(path, 0)
    text = narrate(line)
    assert text.strip()
    antes_seen = {s["ante"] for s in line["steps"]}
    assert _antes_mentioned(text) >= antes_seen


def test_narrate_vanilla_episode(tmp_path):
    path = str(tmp_path / "log.jsonl")
    run_logged_episode(path, SEEDS[2], ruleset="vanilla", max_steps=400, player_seed=22)
    line = load_line(path, 0)
    text = narrate(line)
    assert text.strip()
    antes_seen = {s["ante"] for s in line["steps"]}
    assert _antes_mentioned(text) >= antes_seen


def test_narrate_match_nonempty_and_mentions_every_ante(tmp_path):
    path = str(tmp_path / "match.jsonl")
    run_logged_match(path, SEEDS[3], max_steps=1500, player_seeds=(23, 24))
    line = load_line(path, 0)
    text = narrate(line)
    assert text.strip()
    antes_seen = set()
    for s in line["steps"]:
        for pv in s["players"]:
            antes_seen.add(pv["ante"])
    assert _antes_mentioned(text) >= antes_seen
    assert "winner" in text


def test_narrate_zero_action_episode():
    """An episode with no actions at all must still narrate without crashing."""
    line = {
        "kind": "episode", "seed": "7I4M53DL", "deck_key": "b_red", "stake": 1,
        "ruleset": "vanilla", "lives_start": 0, "actions": [], "steps": [],
        "signatures": {}, "outcome": {}, "final_state": {"ante": 1},
    }
    text = narrate(line)
    assert text.strip()
