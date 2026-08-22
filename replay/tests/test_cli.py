"""Smoke tests for every mp.replay.cli subcommand, called in-process via main() (no
subprocess: this exercises the same argparse plumbing without a slow process spawn)."""
from __future__ import annotations

import json

from .. import cli
from ._helpers import SEEDS, run_logged_episode, run_logged_match


def test_cli_show(tmp_path, capsys):
    path = str(tmp_path / "log.jsonl")
    run_logged_episode(path, SEEDS[0], ruleset="mlb", max_steps=300, player_seed=41)
    assert cli.main(["show", path, "0"]) == 0
    out = capsys.readouterr().out
    assert "episode seed=" in out


def test_cli_verify_clean(tmp_path, capsys):
    path = str(tmp_path / "log.jsonl")
    run_logged_episode(path, SEEDS[0], ruleset="vanilla", max_steps=300, player_seed=42)
    assert cli.main(["verify", path]) == 0
    out = capsys.readouterr().out
    assert "1/1 lines replay clean" in out


def test_cli_verify_reports_mismatch(tmp_path, capsys):
    path = str(tmp_path / "log.jsonl")
    line = None
    idx = None
    for player_seed in range(43, 43 + 20):
        candidate_path = str(tmp_path / f"log_{player_seed}.jsonl")
        run_logged_episode(candidate_path, SEEDS[0], ruleset="vanilla", max_steps=300,
                            player_seed=player_seed)
        with open(candidate_path, "r", encoding="utf-8") as f:
            candidate = json.loads(f.readline())
        found = next((i for i, a in enumerate(candidate["actions"])
                      if a.get("type") == "skip_blind"), None)
        if found is not None:
            line, idx = candidate, found
            break
    assert line is not None, "no skip_blind found across 20 player seeds"
    line["actions"][idx] = {"type": "play_blind"}
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")
    rc = cli.main(["verify", path])
    assert rc == 1
    out = capsys.readouterr().out
    assert "line 0:" in out


def test_cli_tag_then_filter_and_stats(tmp_path, capsys):
    path = str(tmp_path / "log.jsonl")
    for i, seed in enumerate(SEEDS[:4]):
        run_logged_episode(path, seed, ruleset="mlb", max_steps=300, player_seed=50 + i)

    assert cli.main(["tag", path]) == 0
    capsys.readouterr()

    assert cli.main(["filter", path, "--tag", "reached_ante_1"]) == 0
    out = capsys.readouterr().out
    assert "lines matched" in out

    assert cli.main(["stats", path]) == 0
    out = capsys.readouterr().out
    assert "tag counts:" in out
    assert "ante histogram" in out


def test_cli_export_viz(tmp_path, capsys):
    path = str(tmp_path / "log.jsonl")
    run_logged_episode(path, SEEDS[0], ruleset="mlb", max_steps=300, player_seed=60)
    out_json = str(tmp_path / "trajectory.json")
    assert cli.main(["export-viz", path, "0", out_json]) == 0
    with open(out_json, "r", encoding="utf-8") as f:
        doc = json.load(f)
    assert "trajectory" in doc


def test_cli_export_viz_match(tmp_path, capsys):
    path = str(tmp_path / "match.jsonl")
    run_logged_match(path, SEEDS[0], max_steps=1500, player_seeds=(61, 62))
    out_json = str(tmp_path / "trajectory.json")
    assert cli.main(["export-viz", path, "0", out_json, "--player", "1"]) == 0
    with open(out_json, "r", encoding="utf-8") as f:
        doc = json.load(f)
    assert doc["seed"].endswith("#p1")
