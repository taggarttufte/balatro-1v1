"""export_viz produces valid JSON with the documented keys, for both episode and match lines,
and export_viz_to_file round-trips through json.load."""
from __future__ import annotations

import json

from .._bootstrap import BalatroGame
from ..export_viz import export_viz, export_viz_match, export_viz_to_file
from ..replay import load_line
from ._helpers import SEEDS, run_logged_episode, run_logged_match

_TOP_KEYS = {"seed", "outcome", "trajectory", "episode_id"}
_OUTCOME_KEYS = {"ante", "reward", "steps", "dollars", "won"}
_STEP_KEYS = {
    "step", "phase", "ante", "blind_idx", "money", "chips_scored", "hands_left",
    "discards_left", "deck_size", "hand_size", "blind", "hand_cards", "jokers", "shop",
    "consumables", "planet_levels", "value_estimate", "action", "top_probs", "reward",
}
_BLIND_KEYS = {"name", "kind", "target", "is_boss", "boss_key"}


def test_export_viz_episode_shape_and_json_valid(tmp_path):
    path = str(tmp_path / "log.jsonl")
    run_logged_episode(path, SEEDS[0], ruleset="mlb", max_steps=400, player_seed=31)
    line = load_line(path, 0)
    doc = export_viz(line, episode_id=7)

    assert set(doc.keys()) == _TOP_KEYS
    assert doc["episode_id"] == 7
    assert set(doc["outcome"].keys()) == _OUTCOME_KEYS
    assert len(doc["trajectory"]) == len(line["actions"])

    # must be actually JSON-serializable (no stray Card/JokerInstance objects leaking through)
    blob = json.dumps(doc)
    reloaded = json.loads(blob)
    assert reloaded["episode_id"] == 7

    for entry in doc["trajectory"]:
        assert _STEP_KEYS <= set(entry.keys())
        assert _BLIND_KEYS <= set(entry["blind"].keys())
        assert isinstance(entry["hand_cards"], list)
        assert isinstance(entry["jokers"], list)
        assert isinstance(entry["shop"], list)
        assert entry["action"]["type"] in ("hand", "phase")


def test_export_viz_pre_action_state_matches_logged_step_summary(tmp_path):
    """Each trajectory entry is the state BEFORE its action (V7 convention, confirmed
    against the shipped viz/trajectory.json): entry[0] must equal the fresh game's own
    starting values, and entry[i] (i>0) must equal the logged POST-state of action i-1
    (line["steps"][i-1]) since that is the same instant."""
    path = str(tmp_path / "log.jsonl")
    run_logged_episode(path, SEEDS[4], ruleset="vanilla", max_steps=300, player_seed=32)
    line = load_line(path, 0)
    doc = export_viz(line)

    fresh = BalatroGame(seed=line["seed"], deck_key=line["deck_key"], stake=line["stake"],
                        ruleset=line["ruleset"])
    first = doc["trajectory"][0]
    assert first["money"] == fresh.dollars
    assert first["chips_scored"] == fresh.chips_scored
    assert first["hands_left"] == fresh.hands_left
    assert first["discards_left"] == fresh.discards_left

    for i in range(1, len(doc["trajectory"])):
        entry = doc["trajectory"][i]
        prior = line["steps"][i - 1]
        assert entry["money"] == prior["money"]
        assert entry["chips_scored"] == prior["chips_scored"]
        assert entry["hands_left"] == prior["hands_left"]
        assert entry["discards_left"] == prior["discards_left"]


def test_export_viz_to_file_writes_valid_json(tmp_path):
    path = str(tmp_path / "log.jsonl")
    run_logged_episode(path, SEEDS[5], ruleset="mlb", max_steps=300, player_seed=33)
    line = load_line(path, 0)
    out = str(tmp_path / "trajectory.json")
    export_viz_to_file(line, out)
    with open(out, "r", encoding="utf-8") as f:
        doc = json.load(f)
    assert set(doc.keys()) == _TOP_KEYS


def test_export_viz_match_per_player(tmp_path):
    path = str(tmp_path / "match.jsonl")
    run_logged_match(path, SEEDS[6], max_steps=1500, player_seeds=(34, 35))
    line = load_line(path, 0)
    doc0 = export_viz_match(line, 0)
    doc1 = export_viz_match(line, 1)
    assert set(doc0.keys()) == _TOP_KEYS
    assert set(doc1.keys()) == _TOP_KEYS
    assert doc0["seed"] != doc1["seed"]   # "<seed>#p0" vs "<seed>#p1"
    json.dumps(doc0)
    json.dumps(doc1)
