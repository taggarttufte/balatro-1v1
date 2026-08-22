"""
Log -> replay round trip: 20 seeds x {vanilla, MLB solo} with a random-legal player, and with
MLBMatch (two random-legal players).  A deliberately corrupted action list must fail
``verify`` at the right step.  Logger overhead is measured three ways -- see the
"overhead" section below and REPLAY_NOTES.md "Measured: logging overhead" for why a single
flat "<5%" number does not hold with the sig_every=10 default (it is only 5% at 10 for the
bookkeeping-only component; state_signature() capture itself is not cheap).
"""
from __future__ import annotations

import copy
import json
import os
import time

import pytest

from .._bootstrap import BalatroGame, MLBMatch, State
from .._util import summarize
from ..log import MatchLogger, TrajectoryLogger
from ..replay import ReplayMismatch, load_line, load_lines, replay, replay_line, replay_match
from ._helpers import RandomLegalPlayer, SEEDS, run_logged_episode, run_logged_match


# ============================================================================ vanilla round trip

@pytest.mark.parametrize("seed", SEEDS)
def test_episode_round_trip_vanilla(tmp_path, seed):
    path = str(tmp_path / "log.jsonl")
    run_logged_episode(path, seed, ruleset="vanilla", max_steps=400, player_seed=1)
    line = load_line(path, 0)
    assert line["kind"] == "episode"
    assert line["seed"] == seed
    game = replay(line)   # raises ReplayMismatch on any divergence
    assert game.seed_str == seed


@pytest.mark.parametrize("seed", SEEDS)
def test_episode_round_trip_mlb_solo(tmp_path, seed):
    path = str(tmp_path / "log.jsonl")
    run_logged_episode(path, seed, ruleset="mlb", max_steps=400, player_seed=2)
    line = load_line(path, 0)
    assert line["ruleset"] == "mlb"
    replay(line)


# ============================================================================ match round trip

@pytest.mark.parametrize("seed", SEEDS[:8])
def test_match_round_trip(tmp_path, seed):
    path = str(tmp_path / "match.jsonl")
    run_logged_match(path, seed, max_steps=1500, player_seeds=(3, 4))
    line = load_line(path, 0)
    assert line["kind"] == "match"
    match = replay_match(line)
    assert match.seed_str == seed
    assert match.winner == line["outcome"]["winner"]


def test_replay_line_dispatches_on_kind(tmp_path):
    ep_path = str(tmp_path / "ep.jsonl")
    run_logged_episode(ep_path, SEEDS[0], max_steps=100)
    ep_line = load_line(ep_path, 0)
    assert isinstance(replay_line(ep_line), BalatroGame)

    match_path = str(tmp_path / "match.jsonl")
    run_logged_match(match_path, SEEDS[0], max_steps=500)
    match_line = load_line(match_path, 0)
    assert isinstance(replay_line(match_line), MLBMatch)


# ============================================================================ multi-episode file

def test_multiple_episodes_append_to_same_file(tmp_path):
    path = str(tmp_path / "log.jsonl")
    for seed in SEEDS[:5]:
        run_logged_episode(path, seed, max_steps=150)
    lines = load_lines(path)
    assert len(lines) == 5
    for line in lines:
        replay(line)


# ============================================================================ corruption detection

def test_corrupted_action_list_fails_verify_at_right_step(tmp_path):
    path = str(tmp_path / "log.jsonl")
    run_logged_episode(path, SEEDS[0], ruleset="vanilla", max_steps=500, player_seed=5)
    line = load_line(path, 0)

    # A "skip_blind" -> "play_blind" swap always changes behaviour (skip_blind is only ever
    # legal -- hence only ever chosen by the random-legal player -- on Small/Big blinds,
    # where play_blind is unconditionally legal too), so pick the first one in the log.
    bad_idx = next(
        (i for i, a in enumerate(line["actions"]) if a.get("type") == "skip_blind"), None
    )
    assert bad_idx is not None, "no skip_blind in this run -- widen max_steps or change seed"

    corrupt = copy.deepcopy(line)
    corrupt["actions"][bad_idx] = {"type": "play_blind"}

    with pytest.raises(ReplayMismatch) as exc_info:
        replay(corrupt)
    # The mismatch must be reported at or after the corrupted index (never before: nothing
    # upstream of it changed) -- the very next signature checkpoint at/after bad_idx is where
    # it will actually be caught (sig_every default 10 -> checkpoints at 0, 10, 20, ...).
    assert exc_info.value.step >= bad_idx


def test_verify_file_reports_mismatch_and_line_index(tmp_path):
    path = str(tmp_path / "log.jsonl")
    run_logged_episode(path, SEEDS[0], max_steps=200, player_seed=6)
    run_logged_episode(path, SEEDS[1], max_steps=200, player_seed=7)
    lines = load_lines(path)
    corrupt = copy.deepcopy(lines[1])
    corrupt["actions"][5] = {"type": "__does_not_exist__"}
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(lines[0]) + "\n")
        f.write(json.dumps(corrupt) + "\n")

    from ..replay import verify_file
    result = verify_file(path)
    assert result["total"] == 2
    assert result["ok"] == 1
    assert [idx for idx, _ in result["mismatches"]] == [1]


# ============================================================================ overhead
#
# MEASURED (see REPLAY_NOTES.md "Measured: logging overhead"): log.step()'s own bookkeeping
# (copying the action dict, building the 10-field summary, appending to two lists) is cheap
# -- well under 1% of a bare game.step() -- regardless of sig_every.  The other cost,
# state_signature() capture every `sig_every` steps, is NOT cheap: it is a full canonical
# snapshot of the whole run (whole deck, shop, jokers, rng state -- game.py's own docstring
# calls it exactly that), roughly as expensive as a step() itself, sometimes much more once
# the deck/joker/shop have grown.  So the DEFAULT sig_every=10 costs roughly 25-35% wall
# clock on a random-legal MLB episode (measured below); sig_every=100 brings that under 10%.
# Three tests: (1) the ALWAYS-cheap bookkeeping component in isolation, (2) a generous sanity
# ceiling on the full default-tuning overhead (catches a real regression, e.g. an accidental
# per-step signature capture, without asserting a target this design cannot hit), (3) the
# documented throughput tuning (larger sig_every) actually working.

def test_logger_bookkeeping_overhead_excluding_signatures():
    """The part of log.step() that does NOT scale with sig_every (dict copy + summarize +
    buffering) must stay negligible: < 5% of bare game.step() cost, aggregated over several
    natural random-legal episodes."""
    total_bare = 0.0
    total_bookkeeping = 0.0
    for seed in SEEDS[:10]:
        game = BalatroGame(seed=seed, deck_key="b_red", stake=1, ruleset="mlb")
        player = RandomLegalPlayer(seed=1)
        idx = 0
        while game.state != State.GAME_OVER and idx < 400:
            action = player.act(game)
            t0 = time.perf_counter()
            game.step(action)
            total_bare += time.perf_counter() - t0

            t0 = time.perf_counter()
            _ = dict(action)
            _ = summarize(game, idx)
            total_bookkeeping += time.perf_counter() - t0
            idx += 1

    overhead = total_bookkeeping / total_bare
    assert overhead < 0.05, f"bookkeeping overhead {overhead:.1%} >= 5%"


def test_logger_default_overhead_has_a_generous_sanity_ceiling(tmp_path):
    """Full TrajectoryLogger.step() cost (sig_every=10 default) vs bare game.step(),
    aggregated over 10 natural episodes.  This is expected to be well above 5% (see the
    module note) -- the ceiling here (2x) is a regression guard, not a throughput target."""
    path = str(tmp_path / "log.jsonl")
    total_bare = 0.0
    total_logged = 0.0
    for seed in SEEDS[:10]:
        game = BalatroGame(seed=seed, deck_key="b_red", stake=1, ruleset="mlb")
        player = RandomLegalPlayer(seed=1)
        log = TrajectoryLogger(path)
        log.begin(game)
        idx = 0
        while game.state != State.GAME_OVER and idx < 400:
            action = player.act(game)
            t0 = time.perf_counter()
            game.step(action)
            total_bare += time.perf_counter() - t0

            t0 = time.perf_counter()
            log.step(game, action)
            total_logged += time.perf_counter() - t0
            idx += 1
        log.end(game, outcome={"won": False})

    overhead = total_logged / total_bare
    assert overhead < 2.0, f"default-tuning logger overhead {overhead:.1%} -- likely a regression"


def test_larger_sig_every_brings_overhead_down(tmp_path):
    """The documented throughput tuning (REPLAY_NOTES.md) actually works: sig_every=100
    keeps step()-only overhead under 15% on the same workload that costs 25-35% at the
    sig_every=10 default."""
    path = str(tmp_path / "log.jsonl")
    total_bare = 0.0
    total_logged = 0.0
    for seed in SEEDS[:10]:
        game = BalatroGame(seed=seed, deck_key="b_red", stake=1, ruleset="mlb")
        player = RandomLegalPlayer(seed=1)
        log = TrajectoryLogger(path, sig_every=100)
        log.begin(game)
        idx = 0
        while game.state != State.GAME_OVER and idx < 400:
            action = player.act(game)
            t0 = time.perf_counter()
            game.step(action)
            total_bare += time.perf_counter() - t0

            t0 = time.perf_counter()
            log.step(game, action)
            total_logged += time.perf_counter() - t0
            idx += 1
        log.end(game, outcome={"won": False})

    overhead = total_logged / total_bare
    assert overhead < 0.15, f"sig_every=100 overhead {overhead:.1%} >= 15%"


# ============================================================================ bytes / episode

def test_bytes_per_episode_is_a_few_kb(tmp_path):
    path = str(tmp_path / "log.jsonl")
    for seed in SEEDS[:10]:
        run_logged_episode(path, seed, ruleset="mlb", max_steps=500, player_seed=9)
    size = os.path.getsize(path)
    per_episode = size / 10
    # "few KB per episode" (PHASE4_BRIEF §W3) -- generous ceiling, not a tight bound.
    assert per_episode < 40_000, f"{per_episode:.0f} bytes/episode -- log format bloated"
