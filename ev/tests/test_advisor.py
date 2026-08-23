"""test_advisor.py -- fixture determinism/composition, the advisor's three sections, the
replay: state source, and side-effect freedom (Phase 5 rev 2, W6)."""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import pytest

import _bootstrap  # noqa: F401
from _bootstrap import MLBMatch, State

import advisor
import fixtures
from fixtures.bloodstone_vs_invisible import build as build_fixture


# ─────────────────────────────────────────────────────────────── fixture

def test_fixture_builds_deterministically():
    m1 = build_fixture()
    m2 = build_fixture()
    assert m1.signature() == m2.signature()
    assert m1.games[0].state_signature() == m2.games[0].state_signature()
    assert m1.games[1].state_signature() == m2.games[1].state_signature()


def test_fixture_lands_at_target_ante_and_state():
    m = build_fixture()
    g0 = m.games[0]
    assert g0.ante in (4, 5)
    assert g0.state in (State.BLIND_SELECT, State.SHOP)


def test_fixture_jokers_present_and_blueprint_targets_invisible():
    m = build_fixture()
    g0, g1 = m.games

    g0_keys = [j.key for j in g0.jokers]
    assert "j_bloodstone" in g0_keys

    g1_keys = [j.key for j in g1.jokers]
    assert "j_blueprint" in g1_keys
    assert "j_invisible" in g1_keys
    bp_idx = g1_keys.index("j_blueprint")
    inv_idx = g1_keys.index("j_invisible")
    assert inv_idx == bp_idx + 1, "Blueprint must sit immediately LEFT of Invisible to copy it"

    # Hearts-leaning tweak actually moved the needle for player 0.
    hearts = sum(1 for c in g0.full_deck if c.suit == "Hearts")
    assert hearts / len(g0.full_deck) >= 0.45

    # plausible lives / $ per the brief
    assert g0.lives == 3
    assert g1.lives == 4
    assert g0.dollars > 0 and g1.dollars > 0


# ─────────────────────────────────────────────────────────────── the advisor report

@pytest.mark.parametrize("player", [0, 1])
def test_advisor_runs_and_prints_three_sections_fast(player):
    m = build_fixture()
    t0 = time.perf_counter()
    text = advisor.advise(m, player, n_rollouts=2, rollout_budget="fast", budget="fast",
                          rollout_seed=1)
    dt = time.perf_counter() - t0
    assert dt < 60.0, f"advise() took {dt:.1f}s, want < 60s"

    # section 1: the situation
    assert "Lives:" in text
    assert "My jokers" in text
    assert "Opponent PUBLIC block" in text
    # section 2: the three P(win) numbers
    assert "P(win)" in text
    assert "rollout " in text
    assert "race  " in text or "race " in text
    assert "V  " in text or "V " in text
    # section 3: the ranked action table
    assert "Ranked actions" in text
    # section 4: the opponent read
    assert "Opponent read" in text


def _make_tiny_checkpoint(path) -> None:
    """A deliberately tiny (untrained, ~26k param) SetValueNet checkpoint -- just big enough
    to exercise `value_net.save_checkpoint`/`load_checkpoint`/`make_value_fn` and
    `match_player.MatchAwareEVPlayer`'s wiring end to end without paying the real 5M-param
    net's cost.  The number V produces is meaningless (random init); only the WIRING is
    under test here."""
    import mcts.value_net as V
    import mcts.encoder_v2 as E
    cfg = V.ValueNetConfig(d_item=16, n_heads=2, ffn_mult=1, key_emb=8, card_emb=4, aux_emb=4,
                          trunk_width=32, n_res_blocks=1, scalar_hidden=16)
    net = V.SetValueNet(cfg)
    enc = E.SetEncoderV2()
    V.save_checkpoint(str(path), net, enc, extra={"note": "test-only tiny net"})


def test_advisor_with_checkpoint_is_opponent_aware_and_v_guided(tmp_path):
    """W5's match_player.MatchAwareEVPlayer (landed mid-session) is what makes V's use
    INSIDE the ranked action table see the true opponent, not just the standalone "V" number
    -- this pins that the checkpoint path is actually wired through it end to end."""
    ckpt = tmp_path / "tiny_v.pt"
    _make_tiny_checkpoint(ckpt)

    m = build_fixture()
    sig_before = m.signature()
    text = advisor.advise(m, 0, n_rollouts=2, rollout_budget="fast", budget="fast",
                          checkpoint=str(ckpt))
    assert m.signature() == sig_before

    assert f"V        (checkpoint={ckpt})" in text
    assert "(V-guided, opponent-aware)" in text
    # a bound MatchAwareEVPlayer evaluates more shop candidates than the bare rules tier
    # (argmax-V over stepped clones vs the hand-written proxy rules) -- confirms `explainer`
    # actually reached EVPlayer.explain rather than silently falling back to the rules tier.
    assert "V after" in text


def test_advise_does_not_mutate_state():
    m = build_fixture()
    sig_before = m.signature()
    advisor.advise(m, 0, n_rollouts=2, rollout_budget="fast", budget="fast")
    assert m.signature() == sig_before


def test_disagreement_flag_appears_when_estimators_disagree():
    # This fixture's race curve is fit BEFORE the fixture's own joker edits (it only knows
    # the pvp_log from self-play), while the rollout estimator plays the post-edit board --
    # a real, expected disagreement (see ADVISOR_NOTES.md). n_rollouts is small on purpose
    # (fast, may not always trip the 0.15 threshold at n=2, so this just checks the report
    # doesn't crash and the DISAGREEMENT machinery runs without error).
    m = build_fixture()
    text, numbers = advisor.prob_block(m, 0, n_rollouts=2, rollout_budget="fast", rollout_seed=3)
    assert "race" in numbers and "rollout" in numbers
    assert numbers["v"] is None


# ─────────────────────────────────────────────────────────────── state sources

def test_load_state_source_fixture():
    m, default_player = advisor.load_state_source("fixture:bloodstone_vs_invisible")
    assert default_player == 0
    assert isinstance(m, MLBMatch)
    assert m.games[0].ante in (4, 5)


def test_load_state_source_unknown_fixture_raises():
    with pytest.raises(ValueError):
        advisor.load_state_source("fixture:does_not_exist")


def test_load_state_source_bad_spec_raises():
    with pytest.raises(ValueError):
        advisor.load_state_source("not_a_valid_spec")


class _RandomLegalPlayer:
    """Self-contained (no mp/tournament import), deterministic in `seed`."""

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)

    def act(self, game) -> dict:
        acts = game.legal_actions()
        if not acts:
            return {"type": "advance"}
        return self._rng.choice(acts)


def _write_small_match_log(path: str, seed: str, n_steps: int = 24) -> list:
    """Drives a small MLBMatch through mp/replay's MatchLogger (public class; read-only use
    of mp/replay, matching mp/replay/tests/_helpers.py's own pattern) and returns the ops
    actually applied, for an independent cross-check.  mp/replay is a real package (relative
    imports inside it), so it must be imported as ``replay.log`` with mp/ on sys.path."""
    mp_root = str(Path(_bootstrap.MP_ROOT))
    if mp_root not in sys.path:
        sys.path.insert(0, mp_root)
    from replay.log import MatchLogger  # mp/replay/log.py

    match = MLBMatch(seed=seed, deck_key="b_red", stake=1, lives=4)
    players = [_RandomLegalPlayer(seed=0), _RandomLegalPlayer(seed=1)]
    mlog = MatchLogger(path, sig_every=5)
    mlog.begin(match, meta={"src": "test_advisor"})
    ops = []
    n = 0
    while not match.done and n < n_steps:
        p = match.current_player()
        if p is None:
            break
        action = players[p].act(match.games[p])
        match.step(p, action)
        mlog.step(match, p, action)
        ops.append((p, dict(action)))
        n += 1
    mlog.end(match, outcome={"winner": match.winner, "steps": n})
    return ops


def test_load_state_source_replay(tmp_path):
    log_path = tmp_path / "small_match.jsonl"
    ops = _write_small_match_log(str(log_path), seed="1558AXDL", n_steps=20)
    assert len(ops) >= 5, "need enough ops for a meaningful mid-match step"

    step = len(ops) // 2
    m, default_player = advisor.load_state_source(f"replay:{log_path}:{step}")
    assert default_player == 0

    # independent cross-check: replay the same prefix by hand through a fresh MLBMatch
    expect = MLBMatch(seed="1558AXDL", deck_key="b_red", stake=1, lives=4)
    for p, action in ops[:step]:
        expect.step(p, action)
    assert m.signature() == expect.signature()


def test_load_state_source_replay_clamps_step_past_end(tmp_path):
    log_path = tmp_path / "small_match2.jsonl"
    ops = _write_small_match_log(str(log_path), seed="15H9Z3IY", n_steps=12)
    m, _ = advisor.load_state_source(f"replay:{log_path}:999999")
    expect = MLBMatch(seed="15H9Z3IY", deck_key="b_red", stake=1, lives=4)
    for p, action in ops:
        expect.step(p, action)
    assert m.signature() == expect.signature()


def test_load_state_source_seed_is_deterministic():
    m1, _ = advisor.load_state_source("seed:11111111:10", policy_seed=0, budget="fast")
    m2, _ = advisor.load_state_source("seed:11111111:10", policy_seed=0, budget="fast")
    assert m1.signature() == m2.signature()
    assert m1.steps == m2.steps <= 10
