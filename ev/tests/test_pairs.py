"""test_pairs.py — paired within-state action labels (W-PAIRS, lever (b)).

Everything here runs on the scripted fallback policy or on synthetic records except the
two EV-player tests, which skip until W3's ``EVPlayer`` is importable.  The W-EXTRACT hook
is exercised with a FAKE generator / EV term monkeypatched onto ``hand`` — that is the
"hook + test ready" the brief asks for, so the day W-EXTRACT lands only the fake goes away.
"""
from __future__ import annotations

import json
import math
import random

import numpy as np
import pytest

import _bootstrap
from _bootstrap import MLBMatch

import dataset as DS
import hand as H
import labels as L
import pairs as PR

SEED = "7I4M53DL"

FROZEN_KEYS = {"kind", "seed", "step", "actor", "state_kind", "ante", "player_fingerprint",
               "pair_source", "action_a", "action_b", "n_worlds", "outcomes_a", "outcomes_b",
               "delta", "delta_ci", "meta"}


@pytest.fixture(scope="module")
def scripted():
    return L.make_policy_factory("scripted", epsilon=0.1)


@pytest.fixture(scope="module")
def snaps(scripted):
    # rng_seed is pinned: sample_states' default reservoir seed is `hash((seed, policy_seed))`
    # and Python's str hash is PYTHONHASHSEED-salted, so the default draws a DIFFERENT set of
    # snapshots in every process (harmless for the campaign, flaky for tests).
    return L.sample_states(SEED, n_states=8, policy_factory=scripted, policy="scripted",
                           epsilon=0.1, rng_seed=20260825)


# ── action identity ──

def test_action_key_separates_what_hands_sort_key_merges():
    a = {"type": "buy", "item_idx": 0}
    b = {"type": "buy", "item_idx": 3}
    assert H._action_sort_key(a) == H._action_sort_key(b)      # the reason pairs has its own
    assert PR.action_key(a) != PR.action_key(b)
    assert PR.action_key({"type": "play", "cards": [1, 2]}) == PR.action_key({"cards": [1, 2], "type": "play"})


# ── the label-policy fingerprint (brief §2) ──

def test_player_fingerprint_is_stable_and_config_sensitive():
    f = PR.player_fingerprint()
    assert f == PR.player_fingerprint() and f.startswith("ev-fast-rules:")
    assert len(f.split(":")[1]) == 12
    assert PR.player_fingerprint(shop_tier="stats") != f
    assert PR.player_fingerprint(budget="full") != f
    assert PR.player_fingerprint(epsilon_rollout=0.05) != f
    assert PR.player_fingerprint(extra={"extraction_version": 2}) != f


def test_player_fingerprint_tracks_the_fast_players_source(monkeypatch):
    f = PR.player_fingerprint()
    monkeypatch.setattr(PR, "source_digest", lambda *a, **k: "deadbeef" * 5)
    assert PR.player_fingerprint() != f          # W-EXTRACT editing hand.py flips it, by design


def test_source_digest_covers_the_player_files():
    assert set(PR._FINGERPRINT_FILES) == {"hand.py", "player.py", "sampling.py"}
    assert len(PR.source_digest()) == 40


# ── the target mix ──

def test_mix_sequence_matches_the_target_proportions():
    rng = random.Random(0)
    got = PR.mix_sequence(100, rng, extraction=True)
    n = {s: got.count(s) for s in PR.PAIR_SOURCES}
    assert n["close_call"] == 50 and n["greedy_vs_extract"] == 40 and n["random"] == 10
    assert len(got) == 100


def test_mix_sequence_without_extraction_folds_into_close_call():
    got = PR.mix_sequence(100, random.Random(1), extraction=False)
    assert got.count("greedy_vs_extract") == 0
    assert got.count("close_call") == 90 and got.count("random") == 10
    assert PR.mix_sequence(1, random.Random(2), extraction=False) == ["close_call"]
    assert len(PR.mix_sequence(7, random.Random(3), extraction=True)) == 7


# ── pair selection ──

def _ranker(pairs):
    return lambda match, actor: [(dict(a), ev) for a, ev in pairs]


def _no_generator(monkeypatch):
    """Hide every LINE GENERATOR entry point so the per-action EV-term fallback is what is
    under test (W-EXTRACT has since landed ``hand.extraction_lines``, which wins the probe)."""
    import player as _P
    for n in PR._GENERATOR_NAMES:
        monkeypatch.delattr(H, n, raising=False)
        monkeypatch.delattr(_P.EVPlayer, n, raising=False)


def test_choose_pair_close_call_needs_a_small_gap(snaps):
    s = next(x for x in snaps if len(x.match.legal_actions(x.actor)) > 1)
    legal = s.match.legal_actions(s.actor)
    a, b = dict(legal[0]), dict(legal[1])
    rng = random.Random(0)
    close = PR.choose_pair(s.match, s.actor, ranker=_ranker([(a, 0.50), (b, 0.49)]), rng=rng,
                           source="close_call")
    assert close.source == "close_call" and close.requested == "close_call"
    assert close.action_a == a and close.action_b == b
    assert close.gap == pytest.approx(0.01) and close.n_ranked == 2
    wide = PR.choose_pair(s.match, s.actor, ranker=_ranker([(a, 0.9), (b, 0.1)]), rng=rng,
                          source="close_call")
    assert wide.source == "random" and wide.requested == "close_call"   # documented fallback
    assert wide.action_a == a and PR.action_key(wide.action_b) != PR.action_key(a)


def test_choose_pair_random_is_uniform_over_legal_and_never_self(snaps):
    s = next(x for x in snaps if len(x.match.legal_actions(x.actor)) > 3)
    legal = s.match.legal_actions(s.actor)
    a = dict(legal[0])
    seen = set()
    for i in range(30):
        c = PR.choose_pair(s.match, s.actor, ranker=_ranker([(a, 1.0)]), rng=random.Random(i),
                           source="random")
        assert c.source == "random" and PR.action_key(c.action_b) != PR.action_key(a)
        assert c.action_b in legal
        seen.add(PR.action_key(c.action_b))
    assert len(seen) > 1


def test_choose_pair_cascades_between_the_informative_sources(snaps):
    """A requested source the state cannot supply tries the OTHER informative source before
    degrading to random; a requested `random` never cascades (it is the deliberate 10 %)."""
    s = next(x for x in snaps if len(x.match.legal_actions(x.actor)) > 2)
    legal = s.match.legal_actions(s.actor)
    a, b, c = dict(legal[0]), dict(legal[1]), dict(legal[2])
    hook = lambda game, lg, avoid=None: dict(c)          # noqa: E731
    wide = _ranker([(a, 0.9), (b, 0.1)])                 # too wide for a close call
    tight = _ranker([(a, 0.50), (b, 0.49)])

    got = PR.choose_pair(s.match, s.actor, ranker=wide, rng=random.Random(0),
                         source="close_call", extraction=hook)
    assert got.source == "greedy_vs_extract" and got.requested == "close_call"
    got = PR.choose_pair(s.match, s.actor, ranker=tight, rng=random.Random(0),
                         source="greedy_vs_extract", extraction=None)
    assert got.source == "close_call" and got.requested == "greedy_vs_extract"
    got = PR.choose_pair(s.match, s.actor, ranker=tight, rng=random.Random(0),
                         source="random", extraction=hook)
    assert got.source == "random"                        # never cascades away from random
    got = PR.choose_pair(s.match, s.actor, ranker=wide, rng=random.Random(0),
                         source="close_call", extraction=hook, cascade=False)
    assert got.source == "random"


def test_choose_pair_declines_a_single_action_state(snaps):
    m = MLBMatch(seed=SEED)

    class OneAction:
        def legal_actions(self, p):
            return [{"type": "advance"}]
        games = m.games

    assert PR.choose_pair(OneAction(), 0, ranker=_ranker([]), rng=random.Random(0),
                          source="random") is None


# ── W-EXTRACT feature detection: the hook + its tests, ready before the layer lands ──

def test_extraction_feature_detection_matches_reality():
    """Whatever W-EXTRACT has (or has not) landed, the hook returns a LEGAL action or None
    — never an exception and never something outside ``legal_actions``."""
    ep = PR.extraction_entry_point()
    if ep is None:
        assert not PR.has_extraction() and PR.make_extraction_hook() is None
        return
    kind, owner, name = ep
    assert kind in ("generator", "ev_term") and owner in ("hand", "player")
    hook = PR.make_extraction_hook()
    assert callable(hook)
    m = MLBMatch(seed=SEED)
    n_lines = 0
    for _ in range(80):
        p = m.current_player()
        legal = m.legal_actions(p)
        if legal:
            out = hook(m.games[p], legal)
            if out is not None:
                n_lines += 1
                assert PR.action_key(out) in {PR.action_key(a) for a in legal}
            m.step(p, legal[0])
        if m.done:
            break
    assert n_lines >= 0        # a vanilla ante-1 board legitimately has nothing to extract


def test_extraction_hook_uses_a_generator(monkeypatch, snaps):
    s = next(x for x in snaps if len(x.match.legal_actions(x.actor)) > 2)
    legal = s.match.legal_actions(s.actor)

    def fake_extraction_lines(game, legal=None):
        acts = legal if legal is not None else game.legal_actions()
        return [(dict(acts[2]), 0.1), (dict(acts[1]), 0.9)]      # best last: must be sorted by ev

    monkeypatch.setattr(H, "extraction_lines", fake_extraction_lines, raising=False)
    assert PR.has_extraction() and PR.extraction_entry_point() == ("generator", "hand", "extraction_lines")
    hook = PR.make_extraction_hook()
    assert PR.action_key(hook(s.match.games[s.actor], legal)) == PR.action_key(legal[1])
    # avoid= drops the greedy choice so a pair is never an action against itself
    assert PR.action_key(hook(s.match.games[s.actor], legal, avoid=legal[1])) == PR.action_key(legal[2])
    c = PR.choose_pair(s.match, s.actor, ranker=_ranker([(dict(legal[1]), 1.0)]),
                       rng=random.Random(0), source="greedy_vs_extract", extraction=hook)
    assert c.source == "greedy_vs_extract"
    assert PR.action_key(c.action_a) == PR.action_key(legal[1])
    assert PR.action_key(c.action_b) == PR.action_key(legal[2])
    assert PR.mix_sequence(10, random.Random(0), extraction=True).count("greedy_vs_extract") == 4


def test_extraction_hook_uses_one_hand_analysis_for_the_ev_term(monkeypatch, snaps):
    """The landed term is ``hand.extraction_ev(game, action)``, which rebuilds a whole
    ``HandAnalysis`` per call — hundreds of legal actions would cost seconds per decision.
    The hook must build ONE analysis and use its method."""
    _no_generator(monkeypatch)
    s = next(x for x in snaps if len(x.match.legal_actions(x.actor)) > 2)
    legal = s.match.legal_actions(s.actor)
    target = PR.action_key(legal[2])
    built = []

    class FakeAnalysis:
        extract_on = True

        def __init__(self, game, cfg=None, *, legal=None, **kw):
            built.append(game)

        def extraction_ev(self, action):
            return 3.0 if PR.action_key(action) == target else 0.0

    monkeypatch.setattr(H, "HandAnalysis", FakeAnalysis)
    monkeypatch.setattr(H, "extraction_ev", lambda *a, **k: pytest.fail("module fn was used"),
                        raising=False)
    assert PR.extraction_entry_point() == ("ev_term", "hand", "extraction_ev")
    hook = PR.make_extraction_hook()
    assert PR.action_key(hook(s.match.games[s.actor], legal)) == target
    assert len(built) == 1
    assert hook(s.match.games[s.actor], legal, avoid=legal[2]) is None   # nothing else procs
    FakeAnalysis.extract_on = False                                     # nothing to extract here
    assert hook(s.match.games[s.actor], legal) is None


@pytest.mark.parametrize("order", ["game_first", "action_first"])
def test_extraction_hook_falls_back_to_the_module_level_term(monkeypatch, snaps, order):
    """No ``HandAnalysis`` method → the module function, called in whichever argument order
    it accepts (the brief writes ``extraction_ev(action, state)``, hand.py landed
    ``(game, action)``)."""
    _no_generator(monkeypatch)
    s = next(x for x in snaps if len(x.match.legal_actions(x.actor)) > 2)
    legal = s.match.legal_actions(s.actor)
    target = PR.action_key(legal[2])
    monkeypatch.delattr(H.HandAnalysis, "extraction_ev", raising=False)

    if order == "game_first":
        def fake(game, action):
            assert isinstance(action, dict) and hasattr(game, "legal_actions")
            return 3.0 if PR.action_key(action) == target else 0.0
    else:
        def fake(action, game):
            assert isinstance(action, dict) and hasattr(game, "legal_actions")
            return 3.0 if PR.action_key(action) == target else 0.0

    monkeypatch.setattr(H, "extraction_ev", fake, raising=False)
    hook = PR.make_extraction_hook()
    assert PR.action_key(hook(s.match.games[s.actor], legal)) == target
    assert hook(s.match.games[s.actor], legal, avoid=legal[2]) is None


def test_extraction_hook_that_raises_is_treated_as_absent(monkeypatch, snaps):
    def boom(game, legal=None):
        raise RuntimeError("W-EXTRACT mid-refactor")

    monkeypatch.setattr(H, "extraction_lines", boom, raising=False)
    s = next(x for x in snaps if len(x.match.legal_actions(x.actor)) > 1)
    legal = s.match.legal_actions(s.actor)
    hook = PR.make_extraction_hook()
    assert hook is not None and hook(s.match.games[s.actor], legal) is None
    c = PR.choose_pair(s.match, s.actor, ranker=_ranker([(dict(legal[0]), 1.0)]),
                       rng=random.Random(0), source="greedy_vs_extract", extraction=hook)
    assert c.source == "random"          # degrades, never dies


# ── rolling on shared worlds (the common-random-numbers plumbing) ──

def test_identical_actions_give_identical_worlds(snaps, scripted):
    """The sharpest CRN check: if both branches take the SAME action the two rolled-out
    branches are the same state on the same world seeds, so every world must agree
    exactly — delta 0, var(d) 0, rho 1."""
    s = next(x for x in snaps if len(x.match.legal_actions(x.actor)) > 1)
    a = s.match.legal_actions(s.actor)[0]
    pr, ma, mb = PR.roll_pair(s.match, s.actor, a, a, n_worlds=4, seed=11,
                              policy_factory=scripted, determinize=False)
    assert pr.outcomes_a == pr.outcomes_b and len(pr.outcomes_a) == 4
    assert pr.delta == 0.0 and pr.var_d == 0.0 and pr.delta_ci == 0.0
    assert pr.y_a == pr.y_b and pr.n_rollouts == 8
    assert ma.signature() == mb.signature()
    assert len(set(pr.world_seeds)) == 4


def test_roll_pair_is_reproducible_and_leaves_the_snapshot_alone(snaps, scripted):
    s = next(x for x in snaps if len(x.match.legal_actions(x.actor)) > 1)
    legal = s.match.legal_actions(s.actor)
    sig = s.match.signature()
    kw = dict(n_worlds=3, seed=5, policy_factory=scripted, determinize=False)
    p1, _, _ = PR.roll_pair(s.match, s.actor, legal[0], legal[1], **kw)
    p2, _, _ = PR.roll_pair(s.match, s.actor, legal[0], legal[1], **kw)
    assert p1.outcomes_a == p2.outcomes_a and p1.outcomes_b == p2.outcomes_b
    assert p1.world_seeds == p2.world_seeds
    assert s.match.signature() == sig                     # the snapshot is never stepped
    p3, _, _ = PR.roll_pair(s.match, s.actor, legal[0], legal[1], **{**kw, "seed": 6})
    assert p3.world_seeds != p1.world_seeds


def test_outcomes_are_from_the_actors_perspective(snaps, scripted):
    """Rolling the same branch for actor 0 and actor 1 must give complementary outcomes."""
    s = next(x for x in snaps if len(x.match.legal_actions(x.actor)) > 1)
    a = s.match.legal_actions(s.actor)[0]
    pr, _, _ = PR.roll_pair(s.match, s.actor, a, a, n_worlds=3, seed=9, policy_factory=scripted,
                            determinize=False)
    m2 = s.match.clone()
    outs = []
    for w in pr.world_seeds:
        mm = PR.step_branch(m2, s.actor, a)
        r = L.rollout(mm, seed=w, policy_factory=scripted, determinize=False)
        outs.append(r.p0_win)
    expect = outs if s.actor == 0 else [1.0 - o for o in outs]
    assert pr.outcomes_a == pytest.approx(expect)


def test_reps_split_into_disjoint_world_blocks(snaps, scripted):
    s = next(x for x in snaps if len(x.match.legal_actions(x.actor)) > 1)
    legal = s.match.legal_actions(s.actor)
    pr, _, _ = PR.roll_pair(s.match, s.actor, legal[0], legal[1], n_worlds=2, reps=3, seed=4,
                            policy_factory=scripted, determinize=False)
    assert len(pr.outcomes_a) == 6 and len(set(pr.world_seeds)) == 6
    assert len(pr.rep_means_a) == 3 and len(pr.rep_means_b) == 3
    assert pr.rep_means_a[0] == pytest.approx(sum(pr.outcomes_a[:2]) / 2)
    assert pr.delta == pytest.approx(sum(pr.outcomes_a) / 6 - sum(pr.outcomes_b) / 6)


def test_coupling_modes(snaps, scripted):
    s = next(x for x in snaps if len(x.match.legal_actions(x.actor)) > 1)
    legal = s.match.legal_actions(s.actor)
    sig = s.match.signature()
    pr, ma, mb = PR.roll_pair(s.match, s.actor, legal[0], legal[1], n_worlds=3, seed=8,
                              policy_factory=scripted, coupling="determinize_then_step")
    assert len(pr.outcomes_a) == 3 and pr.determinized and s.match.signature() == sig
    assert ma is not None and mb is not None            # world 0's branches carry the obs
    # the frozen mode is the default and a bad name is refused loudly
    with pytest.raises(ValueError, match="coupling"):
        PR.roll_pair(s.match, s.actor, legal[0], legal[1], n_worlds=1, seed=1,
                     policy_factory=scripted, coupling="nonsense")
    res = PR.pair_job({"seed": SEED, "n_states": 2, "n_worlds": 2, "policy": "scripted",
                       "encoder": "dummy", "allow_clairvoyant": True,
                       "coupling": "determinize_then_step"})
    for p in PR.pairs_from_result(res):
        assert p.rec["meta"]["rollout"]["coupling"] == "determinize_then_step"


# ── the frozen schema + shards ──

def test_pair_job_end_to_end_and_shard_round_trip(tmp_path):
    res = PR.pair_job({"seed": SEED, "n_states": 4, "n_worlds": 2, "policy": "scripted",
                       "encoder": "dummy", "allow_clairvoyant": True})
    rows = PR.rows_from_result(res)
    prows = PR.pairs_from_result(res)
    assert prows and len(rows) == 2 * len(prows)
    assert res["timing"]["n_rollouts"] == 4 * len(prows)      # 2 worlds x 2 branches per pair
    fp = res["player_fingerprint"]
    for p in prows:
        rec = p.rec
        assert set(rec) == FROZEN_KEYS, set(rec) ^ FROZEN_KEYS
        assert rec["kind"] == "pair" and rec["seed"] == SEED and rec["player_fingerprint"] == fp
        assert rec["pair_source"] in PR.PAIR_SOURCES and rec["state_kind"] in L.STATE_KINDS
        assert rec["state_kind"] != "other"                  # single-action states cannot pair
        assert rec["n_worlds"] == len(rec["outcomes_a"]) == len(rec["outcomes_b"]) == 2
        assert PR.action_key(rec["action_a"]) != PR.action_key(rec["action_b"])
        assert rec["delta"] == pytest.approx(
            sum(rec["outcomes_a"]) / 2 - sum(rec["outcomes_b"]) / 2, abs=1e-5)
        assert 0.0 <= rec["delta_ci"] <= 2.0 and rec["ante"] >= 1
        assert set(rec["meta"]) >= {"rho", "var_a", "var_b", "var_d", "selfplay", "rollout",
                                    "requested_source", "world_seeds", "y_a", "y_b"}
        assert p.obs_a["x"].shape == (16,) and p.obs_b["x"].shape == (16,)
        # STRICT json: a shard must survive any parser, so no NaN/Inf ever reaches it
        assert json.loads(json.dumps(rec, allow_nan=False)) == rec
    # the two absolute rows every pair also yields
    for r in rows:
        assert 0.0 <= r.y <= 1.0 and r.meta["branch"] in ("a", "b")
        assert r.meta["player"] == r.meta["actor"] and r.meta["player_fingerprint"] == fp
        assert r.meta["post_action"] is True and r.meta["kind"] in L.STATE_KINDS
    by_pair = {}
    for r in rows:
        by_pair.setdefault(r.meta["from_pair"], {})[r.meta["branch"]] = r.y
    assert all(len(v) == 2 for v in by_pair.values())
    for p in prows:
        ys = by_pair[f"{p.rec['seed']}:{p.rec['step']}:{p.rec['actor']}"]
        assert ys["a"] == pytest.approx(sum(p.rec["outcomes_a"]) / 2, abs=1e-5)
        assert ys["b"] == pytest.approx(sum(p.rec["outcomes_b"]) / 2, abs=1e-5)
    # pair shard round trip
    path = PR.save_pair_shard(tmp_path / "p.npz", prows)
    sh = PR.load_pair_shard(path)
    assert len(sh) == len(prows)
    assert np.array_equal(sh.obs_a["x"], np.stack([p.obs_a["x"] for p in prows]))
    assert np.array_equal(sh.obs_b["x"], np.stack([p.obs_b["x"] for p in prows]))
    assert sh.columns["seed"][0] == SEED and sh.columns["pair_source"].dtype.kind == "U"
    assert sh.columns["delta"][0] == pytest.approx(prows[0].rec["delta"], abs=1e-5)
    assert sh.records[0] == prows[0].rec
    pd = PR.PairDataset.from_shards([sh, sh])
    assert len(pd) == 2 * len(prows) and pd.seeds() == [SEED]
    # the absolute rows load through the UNCHANGED label dataset
    ds = DS.LabelDataset.from_shards([DS.load_shard(DS.save_shard(tmp_path / "a.npz", rows))])
    assert len(ds) == len(rows) and set(ds.columns["kind"].tolist()) <= set(L.STATE_KINDS)
    assert ds.summary()["n"] == len(rows)


def test_a_pair_is_reconstructible_from_its_own_meta():
    """``meta.selfplay`` + ``(seed, step)`` must replay to the exact decision state the pair
    branched from, with both stored actions legal there — the audit trail the shard promises."""
    res = PR.pair_job({"seed": SEED, "n_states": 3, "n_worlds": 2, "policy": "scripted",
                       "encoder": "dummy", "allow_clairvoyant": True})
    for p in PR.pairs_from_result(res):
        rec, sp = p.rec, p.rec["meta"]["selfplay"]
        m = L.reconstruct_snapshot(rec["seed"], rec["step"], policy=sp["policy"],
                                   budget=sp["budget"], epsilon=sp["epsilon"],
                                   policy_seed=sp["policy_seed"], deck_key=sp["deck_key"],
                                   stake=sp["stake"], lives=sp["lives"], max_ante=sp["max_ante"])
        assert m.steps == rec["step"] and m.current_player() == rec["actor"]
        assert L.state_kind(m.games[rec["actor"]]) == rec["state_kind"]
        keys = {PR.action_key(a) for a in m.legal_actions(rec["actor"])}
        assert PR.action_key(rec["action_a"]) in keys
        assert PR.action_key(rec["action_b"]) in keys


def test_pair_job_refuses_clairvoyant_rollouts_unless_asked(monkeypatch):
    monkeypatch.setattr(L, "has_determinize", lambda: False)
    with pytest.raises(RuntimeError, match="clone_determinized"):
        PR.pair_job({"seed": SEED, "n_states": 1, "n_worlds": 1, "policy": "scripted",
                     "encoder": "dummy"})


def test_pair_job_probe_mode_records_replicates():
    res = PR.pair_job({"seed": SEED, "n_states": 2, "n_worlds": 2, "reps": 3, "policy": "scripted",
                       "encoder": "dummy", "allow_clairvoyant": True})
    for p in PR.pairs_from_result(res):
        assert p.rec["n_worlds"] == 6
        assert len(p.rec["meta"]["rep_means_a"]) == 3 and p.rec["meta"]["reps"] == 3
        assert p.rec["meta"]["n_worlds_per_rep"] == 2


# ── §4: the measurement that justifies the lever ──

def test_variance_report_recovers_a_known_reduction():
    """Perfectly correlated branches (b = a shifted) → var(d) = 0 → infinite reduction;
    uncorrelated branches → ~1x.  The middle case is checked against hand arithmetic."""
    def rec(oa, ob, **kw):
        d = [x - y for x, y in zip(oa, ob)]
        n = len(oa)
        mean = sum(d) / n
        var_d = sum((x - mean) ** 2 for x in d) / (n - 1)
        r = {"kind": "pair", "pair_source": kw.get("src", "close_call"),
             "state_kind": kw.get("kind", "hand"), "outcomes_a": oa, "outcomes_b": ob,
             "delta": mean, "delta_ci": 1.96 * math.sqrt(var_d / n), "meta": {}}
        return r

    same = [rec([1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]) for _ in range(5)]
    rep = PR.variance_report(same, n_worlds=4)
    assert rep["n_pairs"] == 5 and math.isinf(rep["crn"]["var_reduction_factor"]) or \
        rep["crn"]["var_reduction_factor"] != rep["crn"]["var_reduction_factor"]  # inf or nan: var_d == 0

    anti = [rec([1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0])]
    assert PR.variance_report(anti, n_worlds=4)["crn"]["var_reduction_factor"] < 1.0

    oa = [1.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    ob = [1.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    r = PR.variance_report([rec(oa, ob)], n_worlds=6)["crn"]
    va = PR._var(oa)
    vb = PR._var(ob)
    vd = PR._var([x - y for x, y in zip(oa, ob)])
    assert r["var_reduction_factor"] == pytest.approx((va + vb) / vd)
    assert r["mean_rho"] == pytest.approx(PR._cov(oa, ob) / math.sqrt(va * vb))
    assert r["se_paired_at_n"] == pytest.approx(math.sqrt(vd / 6))
    assert r["se_unpaired_at_n"] == pytest.approx(math.sqrt((va + vb) / 6))


def test_variance_report_direct_branch_uses_replicate_means():
    rec = {"kind": "pair", "pair_source": "close_call", "state_kind": "hand",
           "outcomes_a": [1.0, 0.0, 1.0, 1.0], "outcomes_b": [1.0, 0.0, 0.0, 1.0],
           "delta": 0.25, "delta_ci": 0.5,
           "meta": {"reps": 2, "rep_means_a": [1.0, 0.5, 0.5, 1.0],
                    "rep_means_b": [0.5, 0.5, 0.0, 0.5]}}
    d = PR.variance_report([rec], n_worlds=2)["direct"]
    ra, rb = rec["meta"]["rep_means_a"], rec["meta"]["rep_means_b"]
    assert d["n_pairs"] == 1
    assert d["var_paired"] == pytest.approx(PR._var([x - y for x, y in zip(ra, rb)]))
    assert d["var_unpaired"] == pytest.approx(PR._var(ra) + PR._var(rb))
    assert d["var_reduction_factor"] == pytest.approx(d["var_unpaired"] / d["var_paired"])
    assert math.isnan(PR.variance_report([{**rec, "meta": {}}], n_worlds=2)["direct"]["var_paired"])


def test_variance_report_and_mix_report_bucket_by_source_and_kind():
    def rec(src, kind):
        return {"kind": "pair", "pair_source": src, "state_kind": kind,
                "outcomes_a": [1.0, 0.0, 1.0], "outcomes_b": [1.0, 0.0, 0.0],
                "delta": 0.33, "delta_ci": 0.2, "meta": {"requested_source": "close_call"}}

    recs = [rec("close_call", "hand"), rec("close_call", "shop"), rec("random", "hand")]
    v = PR.variance_report(recs, n_worlds=3)
    assert set(v["by_source"]) == {"close_call", "random"}
    assert v["by_source"]["close_call"]["n"] == 2 and set(v["by_state_kind"]) == {"hand", "shop"}
    assert v["resolved_frac"] == pytest.approx(1.0)          # |delta| 0.33 > ci 0.2
    m = PR.mix_report(recs)
    assert m["n"] == 3 and m["pair_source"] == {"close_call": 2, "random": 1}
    assert m["pair_source_frac"]["random"] == pytest.approx(1 / 3)
    assert m["requested_to_realised"]["close_call->random"] == 1
    assert m["state_kind"] == {"hand": 2, "shop": 1}


def test_variance_report_ignores_degenerate_pairs():
    thin = {"kind": "pair", "pair_source": "random", "state_kind": "hand",
            "outcomes_a": [1.0], "outcomes_b": [0.0], "delta": 1.0, "delta_ci": 1.0, "meta": {}}
    assert PR.variance_report([thin])["n_pairs"] == 0


# ── with the real fast player (W3) ──

@pytest.mark.skipif(not L.has_ev_player(), reason="W3 EVPlayer not landed")
def test_rules_ranking_is_legal_and_ordered():
    import player as P
    m = MLBMatch(seed=SEED)
    pl = P.EVPlayer(budget="fast", seed=0, epsilon=0.0)
    for _ in range(30):
        p = m.current_player()
        legal = m.legal_actions(p)
        if len(legal) > 1:
            ranked = PR.rules_ranking(m.games[p], pl)
            assert ranked and all(a in legal for a, _ in ranked)
            assert [ev for _, ev in ranked] == sorted((ev for _, ev in ranked), reverse=True)
            keys = [PR.action_key(a) for a, _ in ranked]
            assert len(keys) == len(set(keys))
            assert PR.action_key(ranked[0][0]) == PR.action_key(pl.act(m.games[p]))
        m.step(p, m.legal_actions(p)[0])
        if m.done:
            break


@pytest.mark.skipif(not (L.has_ev_player() and L.has_determinize()), reason="W2/W3 not landed")
def test_pair_job_with_the_real_fast_player():
    res = PR.pair_job({"seed": SEED, "n_states": 2, "n_worlds": 2, "policy": "ev",
                       "encoder": "dummy", "max_ante": 3})
    prows = PR.pairs_from_result(res)
    assert prows and res["player_fingerprint"].startswith("ev-fast-rules:")
    for p in prows:
        assert set(p.rec) == FROZEN_KEYS
        assert p.rec["meta"]["determinized"] is True
        assert p.rec["meta"]["n_ranked"] >= 1 and p.rec["meta"]["n_legal"] >= 2
        assert -1.0 <= p.rec["delta"] <= 1.0
    v = PR.variance_report([p.rec for p in prows], n_worlds=2)
    assert v["n_pairs"] == len(prows)
