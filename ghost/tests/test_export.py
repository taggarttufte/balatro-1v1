"""
test_export.py — the logged-match -> ghost-replay converter.

The fixture is a real random-legal MLB match (the same driver replay's own tests use),
so every assertion runs against genuine engine output, not a hand-built line.
"""
from __future__ import annotations

import json

import pytest

from ._helpers import run_logged_match
from ..export import GhostExportError, GAMEMODE_KEY, RULESET_KEY, ghost_replay

# Oracle ground-truth seeds (any valid seed works; these are real, already-verified ones).
# Random-legal players resolve at least one Nemesis on these before dying — asserted below.
SEEDS = ["7I4M53DL", "11111111", "1558AXDL", "15H9Z3IY", "1KV4W6YS"]


@pytest.fixture(scope="module")
def match_line(tmp_path_factory):
    path = tmp_path_factory.mktemp("ghost") / "match.jsonl"
    for seed in SEEDS:
        line = run_logged_match(str(path), seed)
        if line["pvp_log"]:
            return line
    raise AssertionError("no seed produced a resolved Nemesis round — driver broken?")


@pytest.fixture(scope="module")
def doc(match_line):
    return ghost_replay(match_line, timestamp=0)


# ─────────────────────────────────────────────────────────────────── rejections

def test_episode_line_rejected():
    with pytest.raises(GhostExportError, match="match"):
        ghost_replay({"kind": "episode"})


def test_no_nemesis_rejected(match_line):
    hollow = dict(match_line, pvp_log=[])
    with pytest.raises(GhostExportError, match="Nemesis"):
        ghost_replay(hollow)


def test_bad_seat_rejected(match_line):
    with pytest.raises(GhostExportError, match="agent_seat"):
        ghost_replay(match_line, agent_seat=2)


# ──────────────────────────────────────────────────────────────────── schema

def test_top_level_schema(doc, match_line):
    assert doc["ruleset"] == RULESET_KEY
    assert doc["gamemode"] == GAMEMODE_KEY
    assert doc["seed"] == match_line["seed"]
    # Display-name contract with the mod: get_deck_key_from_name matches
    # G.P_CENTERS[k].name, and the game's b_red center is named "Red Deck".
    assert doc["deck"] == "Red Deck"
    assert doc["stake"] == match_line["stake"]
    assert isinstance(doc["final_ante"], int) and doc["final_ante"] >= 1
    assert doc["winner"] in ("player", "nemesis", "unknown")
    assert doc["player_name"] and doc["nemesis_name"]
    assert json.loads(json.dumps(doc)) == doc          # plain-JSON serialisable


def test_snapshot_keys_are_the_pvp_log_antes(doc, match_line):
    assert set(doc["ante_snapshots"]) == {str(int(a)) for a, *_ in match_line["pvp_log"]}
    assert all(isinstance(k, str) for k in doc["ante_snapshots"])


def test_hands_entries(doc):
    """Format v2: exactly one PRE-EXHAUSTED entry per side, so the mod's exhaustion check
    (game_state.lua:188-198) can never index-lag behind the human's final hand."""
    for ante, snap in doc["ante_snapshots"].items():
        assert snap["result"] in ("win", "loss", "tie")
        for field in ("player_score", "enemy_score"):
            int(snap[field])                            # plain decimal strings
        for field in ("player_lives", "enemy_lives"):
            assert isinstance(snap[field], int) and snap[field] >= 0
        assert [h["side"] for h in snap["hands"]] == ["player", "enemy"]
        for h in snap["hands"]:
            assert h["hands_left"] == 0
            assert isinstance(h["score"], str)
        assert snap["hands"][0]["score"] == snap["player_score"]
        assert snap["hands"][1]["score"] == snap["enemy_score"]
        # the full per-hand progression survives in the mod-ignored field
        enemy_scores = [int(h["score"]) for h in snap["_hand_progression"]
                        if h["side"] == "enemy"]
        # chips are cumulative within a blind: the progression never goes down
        assert enemy_scores == sorted(enemy_scores), f"ante {ante}: {enemy_scores}"


def test_final_hand_matches_resolution(doc, match_line):
    """The last enemy progression entry of each round IS the score pvp_log resolved with —
    the pin that the resolving-play extraction (pre-step detection + pvp_log fallback)
    works."""
    agent_seat = doc["_generator"]["agent_seat"]
    by_ante = {int(a): (s0, s1) for a, _l, s0, s1 in match_line["pvp_log"]}
    saw_enemy_hands = 0
    for ante, snap in doc["ante_snapshots"].items():
        enemy = [h for h in snap["_hand_progression"] if h["side"] == "enemy"]
        if enemy:
            saw_enemy_hands += 1
            assert int(enemy[-1]["score"]) == int(by_ante[int(ante)][agent_seat])
        assert int(snap["enemy_score"]) == int(by_ante[int(ante)][agent_seat])
        assert int(snap["player_score"]) == int(by_ante[int(ante)][1 - agent_seat])
    assert saw_enemy_hands > 0, "no round had any ghost hands — extraction found nothing"


def test_result_mapping(doc, match_line):
    agent_seat = doc["_generator"]["agent_seat"]
    for ante, loser, _s0, _s1 in match_line["pvp_log"]:
        want = ("win" if loser == agent_seat else
                "loss" if loser == (1 - agent_seat) else "tie")
        assert doc["ante_snapshots"][str(int(ante))]["result"] == want


# ─────────────────────────────────────────────────────────── seat selection

def test_default_seat_is_the_winner(doc, match_line):
    winner = match_line["final_state"]["winner"]
    expected = winner if winner in (0, 1) else 0
    assert doc["_generator"]["agent_seat"] == expected
    if winner in (0, 1):
        assert doc["winner"] == "nemesis"       # the ghost is the sim winner


def test_explicit_seats_mirror_each_other(match_line):
    d0 = ghost_replay(match_line, agent_seat=0, timestamp=0)
    d1 = ghost_replay(match_line, agent_seat=1, timestamp=0)
    flip = {"enemy": "player", "player": "enemy"}
    for ante in d0["ante_snapshots"]:
        s0, s1 = d0["ante_snapshots"][ante], d1["ante_snapshots"][ante]
        assert s0["enemy_score"] == s1["player_score"]
        assert s0["player_score"] == s1["enemy_score"]
        assert s0["enemy_lives"] == s1["player_lives"]
        sides0 = [h["side"] for h in s0["_hand_progression"]]
        sides1 = [h["side"] for h in s1["_hand_progression"]]
        assert sides1 == [flip[s] for s in sides0]
