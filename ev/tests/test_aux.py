"""test_aux.py — W-AUX: auxiliary prediction heads on rollout intermediates (brief §6b).

Three layers:
  1. ``aux_targets.py`` — the spec table, the recorder (cross-checked against an INDEPENDENT
     trace of the same rollout), aggregation/masking, strict JSON.
  2. the producers — ``labels.label_job`` / ``pairs.pair_job`` with ``aux=True``, and the
     shard round trip through both W-PAIRS's and W-RANK's loaders.
  3. the trainer + net — heads off the shared trunk, masked multi-task loss, per-head
     metrics, bit-exact resume, and the two "must be unaffected" guarantees: an aux-less
     trainer is the pre-W-AUX trainer, and old checkpoints (incl. the keeper) still load.
"""
from __future__ import annotations

import inspect
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest
import torch

import _bootstrap  # noqa: F401
from _bootstrap import State

import aux_targets as AX
import dataset as DS
import labels as L
import pairs as PR
import train_v as TV

EV_ROOT = Path(__file__).resolve().parents[1]
KEEPER = EV_ROOT / "runs" / "v_full_best" / "ckpt_0001000.pt"

SMALL_NET = {"d_item": 32, "trunk_width": 64, "n_res_blocks": 1, "scalar_hidden": 32, "key_emb": 8}


# ════════════════════════════════════════════════════════════════════════════════
# 1. aux_targets.py
# ════════════════════════════════════════════════════════════════════════════════

def test_specs_are_wellformed_and_frozen():
    names = [s.name for s in AX.AUX_SPECS]
    assert len(names) == len(set(names)) == 8
    assert names == list(AX.AUX_NAMES)
    for s in AX.AUX_SPECS:
        assert s.kind in ("binary", "reg") and s.dim >= 1 and s.doc
        assert s.weight == pytest.approx(0.1)          # brief §6b.2 "defaults ~0.1"
        assert AX.spec_by_name(s.name) is s
    assert AX.aux_dim() == 9                            # lives_2antes is the only dim-2 head
    # the transforms compress into a sane range and are monotone
    money = AX.spec_by_name("money_next_shop")
    assert 0.0 == money.transform(0) < money.transform(20) < money.transform(200) <= 1.0


def test_xmult_joker_set_matches_the_engine():
    """The frozen ``XMULT_JOKERS`` list is re-derived here from the live registry: an xMult
    joker is one whose scoring implementation multiplies ``ScoreContext.mult_mult``.  If the
    engine gains/loses one, this fails and the list gets updated — the list is data, this is
    its guard (AUX_NOTES §2)."""
    from balatro_sim.jokers.base import JOKER_REGISTRY
    derived = set()
    for key, impl in JOKER_REGISTRY.items():
        try:
            src = inspect.getsource(type(impl))
        except (OSError, TypeError):                     # pragma: no cover - source always available here
            pytest.skip("joker sources are not introspectable in this environment")
        if re.search(r"\bmult_mult\b", src):
            derived.add(key)
    assert derived == set(AX.XMULT_JOKERS), {
        "missing_from_frozen_list": sorted(derived - set(AX.XMULT_JOKERS)),
        "no_longer_xmult": sorted(set(AX.XMULT_JOKERS) - derived),
    }
    assert "j_cavendish" in AX.XMULT_JOKERS and "j_joker" not in AX.XMULT_JOKERS


def test_aggregate_means_over_worlds_and_masks_the_absent():
    worlds = [
        {0: {"money_next_shop": 10.0, "lives_2antes": [4.0, 3.0], "pvp_margin_next": None,
             "blind_cleared": 1.0, "xmult_by_ante4": None, "extract_income": 3.0,
             "cards_modified": 1.0, "tarots_used": 0.0}},
        {0: {"money_next_shop": 20.0, "lives_2antes": [2.0, 3.0], "pvp_margin_next": 0.5,
             "blind_cleared": 0.0, "xmult_by_ante4": None, "extract_income": 0.0,
             "cards_modified": 3.0, "tarots_used": 2.0}},
    ]
    out = AX.aggregate(worlds, 0)
    assert out["money_next_shop"] == pytest.approx(15.0)
    assert out["lives_2antes"] == [pytest.approx(3.0), pytest.approx(3.0)]
    assert out["pvp_margin_next"] == pytest.approx(0.5)      # mean over the ONE world that had it
    assert out["blind_cleared"] == pytest.approx(0.5)        # soft binary target
    assert out["xmult_by_ante4"] is None                     # masked: no world produced it
    assert out["cards_modified"] == pytest.approx(2.0)
    assert set(out) == set(AX.AUX_NAMES)
    # a player nobody recorded, and an empty world list, are fully masked
    assert AX.aggregate(worlds, 1) == AX.empty_aux()
    assert AX.aggregate([], 0) == AX.empty_aux()


def test_aggregate_output_is_strict_json():
    worlds = [{0: {**AX.empty_aux(), "money_next_shop": float("nan"),
                   "extract_income": float("inf"), "blind_cleared": 1.0}}]
    out = AX.aggregate(worlds, 0)
    assert out["money_next_shop"] is None and out["extract_income"] is None
    # strict JSON: NaN/Inf would round-trip unequal and are rejected by other parsers
    assert json.loads(json.dumps(out, allow_nan=False)) == out


def test_coverage_reports_presence_fractions():
    cov = AX.coverage([{"blind_cleared": 1.0}, {"blind_cleared": None}, None])
    assert cov["blind_cleared"] == pytest.approx(1 / 3)
    assert cov["money_next_shop"] == 0.0


# ── the recorder, against an independent trace of the same rollout ────────────────

class _Trace:
    """A second observer that just logs the raw per-step facts.  The test recomputes the
    aux targets from this trace with completely separate code and compares."""

    def __init__(self):
        self.rows = []
        self.start_state = None

    def start(self, m):
        self.start_state = self._snap(m, None, None)
        self.rows.append(self.start_state)

    def after(self, m, p, a):
        self.rows.append(self._snap(m, p, a))

    def finish(self, m):
        pass

    def result(self):
        return None

    @staticmethod
    def _snap(m, p, a):
        return {
            "actor": p, "action": dict(a) if isinstance(a, dict) else None,
            "state": [g.state for g in m.games],
            "dollars": [int(g.dollars) for g in m.games],
            "ante": [int(g.ante) for g in m.games],
            "lives": [int(g.lives) for g in m.games],
            "cons": [list(g.consumable_hand) for g in m.games],
            "n_pvp": len(m.pvp_detail),
            "pvp": list(m.pvp_detail),
            "deck": [{c.id: (c.enhancement, c.edition, c.seal) for c in g.full_deck} for g in m.games],
        }


class _Both:
    """Runs the real recorder and the trace side by side in ONE rollout."""

    def __init__(self):
        self.rec = AX.AuxRecorder((0, 1))
        self.tr = _Trace()

    def start(self, m):
        self.tr.start(m)
        self.rec.start(m)

    def after(self, m, p, a):
        self.tr.after(m, p, a)
        self.rec.after(m, p, a)

    def finish(self, m):
        self.tr.finish(m)
        self.rec.finish(m)

    def result(self):
        return {"aux": self.rec.result(), "trace": self.tr.rows}


def _expected_from_trace(rows: list, p: int) -> dict:
    """Recompute every target from the raw trace — deliberately written differently from
    ``AuxRecorder`` (a straight second pass over the whole history, no incremental state)."""
    ante0 = rows[0]["ante"][p]
    out = dict.fromkeys(AX.AUX_NAMES)

    # money at the next shop ENTRY
    for prev, cur in zip(rows, rows[1:]):
        if cur["state"][p] == State.SHOP and prev["state"][p] != State.SHOP:
            out["money_next_shop"] = float(cur["dollars"][p])
            break

    # own + opponent lives at ante0 + 2 (terminal lives if it never gets there)
    hit = next((r for r in rows[1:] if r["ante"][p] >= ante0 + 2), None)
    src = hit or rows[-1]
    out["lives_2antes"] = [float(src["lives"][p]), float(src["lives"][1 - p])]

    # the next resolved Nemesis
    n0 = rows[0]["n_pvp"]
    for r in rows[1:]:
        if r["n_pvp"] > n0:
            _a, _loser, s0, s1 = r["pvp"][n0][:4]
            mine, theirs = (s0, s1) if p == 0 else (s1, s0)
            m = math.log10(1 + max(0.0, float(mine))) - math.log10(1 + max(0.0, float(theirs)))
            out["pvp_margin_next"] = max(-6.0, min(6.0, m))
            break

    # the first blind that resolves: a life lost inside it = not cleared
    in_blind = [r["state"][p] in (State.SELECTING_HAND, State.PVP_WAIT) for r in rows]
    open_at = None
    for i, flag in enumerate(in_blind):
        if flag and open_at is None:
            open_at = i
        elif open_at is not None and not flag:
            out["blind_cleared"] = 1.0 if rows[i]["lives"][p] >= rows[open_at]["lives"][p] else 0.0
            break

    # the remainder of ante0: proc income, tarots used, cards modified
    end = next((i for i, r in enumerate(rows) if i and r["ante"][p] > ante0), len(rows) - 1)
    income = 0.0
    tarots = 0
    for i in range(1, end + 1):
        prev, cur = rows[i - 1], rows[i]
        if prev["state"][p] == State.SELECTING_HAND and cur["dollars"][p] > prev["dollars"][p]:
            income += float(cur["dollars"][p] - prev["dollars"][p])
        a = cur["action"]
        if cur["actor"] == p and a and a.get("type") == "use_consumable":
            j = int(a.get("consumable_idx", 0))
            hand = prev["cons"][p]
            if 0 <= j < len(hand) and hand[j] in AX.TAROT_KEYS:
                tarots += 1
    out["extract_income"] = income
    out["tarots_used"] = float(tarots)
    before, after = rows[0]["deck"][p], rows[end]["deck"][p]
    out["cards_modified"] = float(sum(1 for cid, sig in after.items()
                                      if before.get(cid) is None or before[cid] != sig))
    return out


@pytest.fixture(scope="module")
def scripted_snaps():
    f = L.make_policy_factory("scripted", epsilon=0.1)
    return L.sample_states("7I4M53DL", n_states=6, policy_factory=f, policy="scripted", rng_seed=11)


def test_recorder_matches_an_independent_trace(scripted_snaps):
    f = L.make_policy_factory("scripted", epsilon=0.05)
    checked = 0
    for i, s in enumerate(scripted_snaps[:4]):
        r = L.rollout(s.match, seed=1000 + i, policy_factory=f, observer_factory=_Both)
        aux, trace = r.aux["aux"], r.aux["trace"]
        for p in (0, 1):
            want = _expected_from_trace(trace, p)
            got = aux[p]
            for name in AX.AUX_NAMES:
                if name == "xmult_by_ante4":
                    continue        # the trace has no joker view; covered separately below
                w, g = want[name], got[name]
                if w is None:
                    assert g is None, (name, p, g)
                elif isinstance(w, list):
                    assert g == pytest.approx(w), (name, p, g, w)
                else:
                    assert g == pytest.approx(w), (name, p, g, w)
            checked += 1
    assert checked == 8


def test_recorder_does_not_change_the_rollout(scripted_snaps):
    """The observer must be side-effect free: same seed, same outcomes, back to back."""
    f = L.make_policy_factory("scripted", epsilon=0.05)
    s = scripted_snaps[1]
    a, _ = L.label_both(s.match, n_rollouts=3, seed=7, policy_factory=f)
    b, _ = L.label_both(s.match, n_rollouts=3, seed=7, policy_factory=f,
                        observer_factory=AX.make_recorder_factory())
    assert a.outcomes == b.outcomes and a.y == b.y and a.ci == b.ci
    assert a.aux_by_player == {} and set(b.aux_by_player) == {0, 1}


def test_recorded_fields_are_sane_and_strict_json(scripted_snaps):
    f = L.make_policy_factory("scripted", epsilon=0.05)
    s = scripted_snaps[0]
    r0, r1 = L.label_both(s.match, n_rollouts=4, seed=3, policy_factory=f,
                          observer_factory=AX.make_recorder_factory())
    assert r1.aux_by_player is r0.aux_by_player          # keyed by player, not perspective
    for p in (0, 1):
        d = r0.aux_by_player[p]
        assert set(d) == set(AX.AUX_NAMES)
        json.loads(json.dumps(d, allow_nan=False))
        assert d["lives_2antes"] is not None and len(d["lives_2antes"]) == 2
        assert all(0.0 <= v <= 4.0 for v in d["lives_2antes"])
        for nm in ("extract_income", "cards_modified", "tarots_used"):
            assert d[nm] is not None and d[nm] >= 0.0
        for nm in ("blind_cleared", "xmult_by_ante4"):
            assert d[nm] is None or 0.0 <= d[nm] <= 1.0


def test_xmult_flag_tracks_the_joker_list(scripted_snaps):
    """A hand-built board: the recorder must see an xMult joker that is already owned, and
    must MASK the target when the state is already past ante 4."""
    from balatro_sim.jokers.base import JokerInstance
    s = scripted_snaps[0]
    m = s.match.clone()
    m.games[0].jokers.append(JokerInstance("j_cavendish", "None"))
    rec = AX.AuxRecorder((0, 1))
    rec.start(m)
    rec.finish(m)
    out = rec.result()
    assert out[0]["xmult_by_ante4"] == 1.0
    assert out[1]["xmult_by_ante4"] == 0.0
    m2 = s.match.clone()
    for g in m2.games:
        g.ante = 7
    rec2 = AX.AuxRecorder((0, 1))
    rec2.start(m2)
    rec2.finish(m2)
    assert rec2.result()[0]["xmult_by_ante4"] is None       # past ante 4 -> masked


# ════════════════════════════════════════════════════════════════════════════════
# 2. the producers: label_job / pair_job / the shard round trip
# ════════════════════════════════════════════════════════════════════════════════

def _job_payload(**kw) -> dict:
    base = dict(seed="7I4M53DL", n_states=2, n_rollouts=2, policy="scripted",
                epsilon_selfplay=0.1, epsilon_rollout=0.05, encoder="dummy",
                allow_clairvoyant=True)
    base.update(kw)
    return base


def test_label_job_emits_aux_and_without_the_flag_emits_nothing():
    on = L.label_job(_job_payload(aux=True))
    off = L.label_job(_job_payload())
    assert on["rows"] and off["rows"]
    for r in on["rows"]:
        assert set(r["meta"]["aux"]) == set(AX.AUX_NAMES)
        assert r["meta"]["aux_version"] == AX.AUX_VERSION
        json.loads(json.dumps(r["meta"], allow_nan=False))
    assert all("aux" not in r["meta"] for r in off["rows"])


def test_label_shard_round_trips_aux(tmp_path):
    res = L.label_job(_job_payload(aux=True))
    DS.save_shard(tmp_path / "s.npz", L.rows_from_result(res))
    ds = DS.LabelDataset.load(tmp_path)
    assert len(ds) == len(res["rows"])
    assert all(set(m["aux"]) == set(AX.AUX_NAMES) for m in ds.meta)
    cov = AX.coverage([m.get("aux") for m in ds.meta])
    assert cov["lives_2antes"] == 1.0 and cov["extract_income"] == 1.0


def _pair_payload(**kw) -> dict:
    base = dict(seed="7I4M53DL", n_states=2, n_worlds=2, policy="scripted",
                epsilon_selfplay=0.1, epsilon_rollout=0.05, encoder="dummy",
                allow_clairvoyant=True)
    base.update(kw)
    return base


def test_pair_job_records_both_branches(tmp_path):
    res = PR.pair_job(_pair_payload(aux=True))
    assert res["pairs"], "the fixture seed produced no pairs"
    for p in res["pairs"]:
        aux = p["rec"]["aux"]
        assert set(aux) == {"a", "b", "version"} and aux["version"] == AX.AUX_VERSION
        for branch in ("a", "b"):
            assert set(aux[branch]) == set(AX.AUX_NAMES)
        json.loads(json.dumps(p["rec"], allow_nan=False))
    # the two absolute rows a pair also yields carry the SAME branch aux
    by_branch = {(r["meta"]["from_pair"], r["meta"]["branch"]): r["meta"]["aux"] for r in res["rows"]}
    for p in res["pairs"]:
        pid = f'{p["rec"]["seed"]}:{p["rec"]["step"]}:{p["rec"]["actor"]}'
        for branch in ("a", "b"):
            assert by_branch[(pid, branch)] == p["rec"]["aux"][branch]
    off = PR.pair_job(_pair_payload())
    assert all("aux" not in p["rec"] for p in off["pairs"])
    assert all("aux" not in r["meta"] for r in off["rows"])


def test_pair_shard_round_trips_aux_through_both_loaders(tmp_path):
    res = PR.pair_job(_pair_payload(aux=True))
    path = PR.save_pair_shard(tmp_path / "pair_0000.npz", PR.pairs_from_result(res))
    # W-PAIRS's own loader
    shard = PR.load_pair_shard(path)
    assert all("aux" in r for r in shard.records)
    # W-RANK's trainer-side loader (the aux dict rides in the frozen record blob)
    pds = TV.PairDataset.load(tmp_path)
    assert len(pds) == len(res["pairs"])
    ad_a = TV.aux_arrays_from_pairs(pds.extra, list(AX.AUX_SPECS), "a")
    ad_b = TV.aux_arrays_from_pairs(pds.extra, list(AX.AUX_SPECS), "b")
    assert ad_a.any_present and ad_b.any_present
    j = list(AX.AUX_NAMES).index("lives_2antes")
    assert ad_a.mask[:, j].all() and ad_b.mask[:, j].all()


# ════════════════════════════════════════════════════════════════════════════════
# 3a. the net: heads are additive and play-time paths are untouched
# ════════════════════════════════════════════════════════════════════════════════

def test_set_value_net_without_heads_is_unchanged():
    from mcts.value_net import SetValueNet, ValueNetConfig
    net = SetValueNet(ValueNetConfig.from_dict(dict(SMALL_NET)))
    assert net.aux_head_names() == []
    assert not any("aux_heads" in k for k in net.state_dict())
    bd = net.param_breakdown()
    assert bd["aux_heads"] == 0 and sum(bd.values()) == net.n_params()
    assert SetValueNet().n_params() == 4_996_789          # the 5M budget, untouched


def test_aux_heads_add_only_head_parameters_and_share_the_trunk():
    from mcts.value_net import SetValueNet, ValueNetConfig
    base = SetValueNet(ValueNetConfig.from_dict(dict(SMALL_NET)))
    cfg = dict(SMALL_NET, aux_heads={"blind_cleared": 1, "lives_2antes": 2}, aux_hidden=0)
    net = SetValueNet(ValueNetConfig.from_dict(cfg))
    W = net.cfg.trunk_width
    assert net.n_params() - base.n_params() == (W + 1) * 1 + (W + 1) * 2
    assert sorted(net.aux_head_names()) == ["blind_cleared", "lives_2antes"]
    hidden = SetValueNet(ValueNetConfig.from_dict(dict(cfg, aux_hidden=16)))
    assert hidden.n_params() > net.n_params()


@pytest.mark.skipif(not KEEPER.exists(), reason="keeper checkpoint not present")
def test_keeper_checkpoint_still_loads_and_plays():
    """brief §6b.2: loading OLD checkpoints must be unaffected."""
    from mcts.value_net import load_checkpoint, make_value_fn
    net, enc, extra = load_checkpoint(KEEPER)
    assert net.n_params() == 4_996_789 and net.aux_head_names() == []
    assert extra["trainer"]["step"] == 1000
    snaps = L.sample_states("7I4M53DL", n_states=1, policy="scripted",
                            policy_factory=L.make_policy_factory("scripted", epsilon=0.1), rng_seed=3)
    import mcts.encoder_v2 as E
    v = make_value_fn(net, enc)(snaps[0].match.games[0], E.opponent_view(snaps[0].match, 0))
    assert 0.0 <= v <= 1.0


def test_checkpoint_saved_with_heads_still_loads_for_play(tmp_path):
    """brief §6b.2: a checkpoint that DOES carry heads must load for play unchanged."""
    from mcts.encoder_v2 import SetEncoderV2, collate, opponent_view
    from mcts.value_net import SetValueNet, ValueNetConfig, load_checkpoint, save_checkpoint
    cfg = dict(SMALL_NET, aux_heads={s.name: s.dim for s in AX.AUX_SPECS}, aux_hidden=8)
    net = SetValueNet(ValueNetConfig.from_dict(cfg))
    enc = SetEncoderV2()
    save_checkpoint(tmp_path / "ck.pt", net, enc, extra={"trainer": {"step": 1}})
    back, enc2, _extra = load_checkpoint(tmp_path / "ck.pt")
    assert sorted(back.aux_head_names()) == sorted(net.aux_head_names())
    assert all(torch.equal(a, b) for a, b in zip(net.state_dict().values(), back.state_dict().values()))
    snaps = L.sample_states("7I4M53DL", n_states=1, policy="scripted",
                            policy_factory=L.make_policy_factory("scripted", epsilon=0.1), rng_seed=3)
    obs = [enc2(snaps[0].match.games[0], opponent_view(snaps[0].match, 0))]
    with torch.no_grad():
        logits = back(collate(obs, "cpu"))               # play-time path: no aux head touched
        both = back.forward_with_aux(collate(obs, "cpu"))
    assert logits.shape == (1,) and torch.equal(logits, both[0])
    assert set(both[1]) == set(AX.AUX_NAMES)


# ════════════════════════════════════════════════════════════════════════════════
# 3b. the trainer
# ════════════════════════════════════════════════════════════════════════════════

def _aux_for(x: np.ndarray, present: bool = True) -> dict:
    """A synthetic aux dict that is a DETERMINISTIC function of the observation, so a head
    that works must reach high R^2 / low Brier."""
    if not present:
        return {}
    money = float(200.0 * (0.5 + 0.5 * x[0]))
    return {
        "money_next_shop": money,
        "lives_2antes": [float(2 + 2 * x[3]), float(2 - 2 * x[3])],
        "pvp_margin_next": None,
        "blind_cleared": 1.0 if x[3] > 0 else 0.0,
        "xmult_by_ante4": None,
        "extract_income": float(10.0 * x[0]),
        "cards_modified": 0.0,
        "tarots_used": 0.0,
    }


def _rows(seed: str, n: int, rng: np.random.Generator, aux: bool = True) -> list:
    out = []
    for i in range(n):
        x = np.zeros(16, np.float32)
        d = rng.integers(-3, 4)
        x[3] = d / 4.0
        x[0] = rng.random()
        x[15] = 1.0
        y = float(np.clip(0.5 + 0.15 * d + rng.normal(0, 0.05), 0, 1))
        meta = {"seed": seed, "step": i, "player": i % 2, "kind": ["hand", "shop", "nemesis"][i % 3],
                "ante": 1 + i % 5, "ci": 0.2, "n_rollouts": 8, "trunc_frac": 0.0}
        if aux:
            meta["aux"] = _aux_for(x)
        out.append(DS.LabelRow({"x": x}, y, meta))
    return out


@pytest.fixture(scope="module")
def aux_shards(tmp_path_factory):
    d = tmp_path_factory.mktemp("aux_shards")
    rng = np.random.default_rng(0)
    seeds = [f"S{i:07d}" for i in range(12)]
    for k in range(3):
        rows = []
        for s in seeds[k * 4:(k + 1) * 4]:
            rows += _rows(s, 60, rng)
        DS.save_shard(d / f"shard_{k:04d}.npz", rows)
    return d


@pytest.fixture(scope="module")
def plain_shards(tmp_path_factory):
    """The SAME data with no ``aux`` key at all — the "old shards" case."""
    d = tmp_path_factory.mktemp("plain_shards")
    rng = np.random.default_rng(0)
    seeds = [f"S{i:07d}" for i in range(12)]
    for k in range(3):
        rows = []
        for s in seeds[k * 4:(k + 1) * 4]:
            rows += _rows(s, 60, rng, aux=False)
        DS.save_shard(d / f"shard_{k:04d}.npz", rows)
    return d


def _cfg(shards, run_dir, **kw) -> TV.TrainVConfig:
    base = dict(shards=[str(shards)], run_dir=str(run_dir), model="dummy", batch_size=32, lr=3e-3,
                warmup_steps=5, max_steps=60, eval_every=20, checkpoint_every=20, device="cpu",
                holdout_frac=0.25, keep=10, seed=1)
    base.update(kw)
    return TV.TrainVConfig(**base)


def test_resolve_aux_specs():
    assert TV.resolve_aux_specs([]) == [] and TV.resolve_aux_specs(None) == []
    assert [s.name for s in TV.resolve_aux_specs(["all"])] == list(AX.AUX_NAMES)
    got = TV.resolve_aux_specs(["tarots_used", "blind_cleared"])
    assert [s.name for s in got] == ["blind_cleared", "tarots_used"]      # AUX_SPECS order
    with pytest.raises(ValueError):
        TV.resolve_aux_specs(["not_a_head"])


def test_aux_arrays_transform_and_mask():
    specs = list(AX.AUX_SPECS)
    dicts = [
        {"money_next_shop": 0.0, "lives_2antes": [4.0, 0.0], "blind_cleared": 1.0},
        {"money_next_shop": None, "extract_income": 0.0},
        None,
        {"lives_2antes": [1.0], "money_next_shop": float("nan")},        # malformed -> masked
    ]
    ad = TV.aux_arrays(dicts, specs)
    j_money = list(AX.AUX_NAMES).index("money_next_shop")
    j_lives = list(AX.AUX_NAMES).index("lives_2antes")
    assert ad.mask[:, j_money].tolist() == [True, False, False, False]
    assert ad.mask[:, j_lives].tolist() == [True, False, False, False]
    sl = ad.slices[j_lives]
    assert ad.values[0, sl].tolist() == [1.0, 0.0]                       # lives / 4
    assert ad.values[0, ad.slices[j_money]].tolist() == [0.0]            # log1p(0) = 0
    assert ad.coverage()["money_next_shop"] == pytest.approx(0.25)
    assert TV.aux_arrays([{}, {}], []).values.shape == (2, 0)


def test_regression_targets_are_standardised_on_the_train_split(aux_shards, tmp_path):
    """AUX_NOTES §3.3: without this a log1p target's sd is ~0.08 and the head never trains at
    the default weight.  Binary heads must NOT be touched (they are BCE targets in [0, 1])."""
    tr = TV.VTrainer(_cfg(aux_shards, tmp_path / "r", aux_heads=["all"]), log=lambda *_: None)
    names = list(AX.AUX_NAMES)
    j_money, j_blind = names.index("money_next_shop"), names.index("blind_cleared")
    c_money = tr.train_aux.slices[j_money].start
    c_blind = tr.train_aux.slices[j_blind].start
    assert c_money in tr.aux_norm and c_blind not in tr.aux_norm
    v = tr.train_aux.values[tr.train_aux.mask[:, j_money], c_money]
    assert v.mean() == pytest.approx(0.0, abs=1e-4) and v.std() == pytest.approx(1.0, abs=1e-4)
    b = tr.train_aux.values[tr.train_aux.mask[:, j_blind], c_blind]
    assert set(np.unique(b)) <= {0.0, 1.0}
    # the holdout is scaled by the TRAIN statistics, not its own
    h = tr.holdout_aux.values[tr.holdout_aux.mask[:, j_money], c_money]
    assert abs(float(h.mean())) < 0.5 and h.std() != pytest.approx(1.0, abs=1e-6)
    # ...and turning it off leaves the raw transformed values alone
    off = TV.VTrainer(_cfg(aux_shards, tmp_path / "r2", aux_heads=["all"], aux_standardize=False),
                      log=lambda *_: None)
    assert off.aux_norm == {}
    raw = off.train_aux.values[off.train_aux.mask[:, j_money], c_money]
    assert 0.0 <= raw.min() and raw.max() <= 1.0 and raw.std() < 0.2


def test_aux_norm_is_carried_across_a_resume(aux_shards, tmp_path):
    run = tmp_path / "run"
    TV.run(_cfg(aux_shards, run, aux_heads=["all"], max_steps=20, eval_every=20, checkpoint_every=20),
           log=lambda *_: None)
    _, _, extra = TV.VTrainer.read_checkpoint(run / "latest.pt")
    saved = extra["trainer"]["aux_norm"]
    assert saved and all(len(v) == 2 for v in saved.values())
    tr = TV.VTrainer.from_checkpoint(run / "latest.pt", overrides={"max_steps": 25}, log=lambda *_: None)
    assert {str(k): list(v) for k, v in tr.aux_norm.items()} == saved
    # a checkpoint whose norm differs (e.g. a different data slice) WINS over a recomputation
    tr2 = TV.VTrainer.from_checkpoint(run / "latest.pt", overrides={"max_steps": 25}, log=lambda *_: None)
    fake = {k: [0.0, 1.0] for k in saved}
    tr2.restore_aux_norm(fake)
    assert all(v == (0.0, 1.0) for v in tr2.aux_norm.values())


def test_no_aux_configured_leaves_the_trainer_untouched(aux_shards, tmp_path):
    """The pre-W-AUX code path: no heads, no aux keys in the records, no aux in the eval."""
    tr = TV.VTrainer(_cfg(aux_shards, tmp_path / "r"), log=lambda *_: None)
    assert tr.aux_specs == [] and tr.aux_on is False and tr.train_aux is None
    assert not any("aux_heads" in k for k in tr.net.state_dict())
    rec = tr.train_step()
    assert not any(k.startswith("aux") for k in rec)
    m = tr.eval()
    assert "aux" not in m


def test_aux_training_reports_per_head_metrics_and_learns(aux_shards, tmp_path):
    # aux_weight 1.0 here (not the 0.1 default): this test asks "can a head learn a
    # learnable target and is it reported", not "is 0.1 the right weight".
    cfg = _cfg(aux_shards, tmp_path / "r", aux_heads=["all"], aux_weight=1.0, max_steps=600,
               eval_every=300, checkpoint_every=600, lr=5e-3)
    summ = TV.run(cfg, log=lambda *_: None)
    a = summ["aux"]
    assert a["source"] == "absolute" and a["n"] > 0
    heads = a["heads"]
    assert set(heads) == set(AX.AUX_NAMES)
    # the two heads whose target is a deterministic function of the input must be learned
    assert heads["money_next_shop"]["r2"] > 0.9, heads["money_next_shop"]
    assert heads["lives_2antes"]["r2"] > 0.9, heads["lives_2antes"]
    assert heads["blind_cleared"]["brier"] < 0.5 * heads["blind_cleared"]["brier_base"]
    # the two the generator never produced are reported as fully masked, not as zeros
    assert heads["pvp_margin_next"]["n"] == 0 and heads["xmult_by_ante4"]["n"] == 0
    assert a["coverage"]["pvp_margin_next"] == 0.0 and a["coverage"]["money_next_shop"] == 1.0
    # and the main metrics are all still there
    fe = summ["final_eval"]
    assert fe["bce"] < fe["const"]["bce"] and fe["auc"] > 0.8 and "ece_guardrail_breached" in fe


def test_old_shards_train_with_aux_muted(plain_shards, tmp_path):
    """Heads configured, data with no aux at all: the run works, every aux term is 0, and
    the per-head metrics say n=0 rather than inventing a target."""
    cfg = _cfg(plain_shards, tmp_path / "r", aux_heads=["all"], max_steps=40, eval_every=40)
    tr = TV.VTrainer(cfg, log=lambda *_: None)
    assert tr.aux_on and not tr.train_aux.any_present
    rec = tr.train_step()
    assert rec["aux_loss"] == 0.0
    assert rec["loss"] == pytest.approx(rec["bce_loss"])
    m = tr.eval()
    assert m["aux"]["n"] == 0


def test_aux_gradients_reach_the_trunk(aux_shards, tmp_path):
    cfg = _cfg(aux_shards, tmp_path / "r", aux_heads=["money_next_shop"], aux_weight=1.0)
    tr = TV.VTrainer(cfg, log=lambda *_: None)
    obs, _y = tr.next_batch()
    batch = TV.to_batch(obs, tr.device)
    _logits, pred = tr.net.forward_with_aux(batch)
    loss, info = tr._aux_loss(pred, tr._last_idx, tr.train_aux)
    tr.net.zero_grad(set_to_none=True)
    loss.backward()
    assert info["aux_loss"] > 0
    grads = {n: (p.grad is not None and float(p.grad.norm()) > 0) for n, p in tr.net.named_parameters()}
    assert grads["aux_heads.money_next_shop.weight"]
    assert grads["net.0.weight"] and grads["net.2.weight"]      # the SHARED trunk
    assert not grads["net.4.weight"]                            # ...but not the value head


def test_aux_on_pairs_uses_both_branches(tmp_path):
    rng = np.random.default_rng(3)
    rows = []
    for i in range(48):
        xa, xb = np.zeros(16, np.float32), np.zeros(16, np.float32)
        xa[3], xb[3] = 0.5, -0.5
        xa[0], xb[0] = rng.random(), rng.random()
        rec_extra = {"kind": "pair", "action_a": {}, "action_b": {}, "outcomes_a": [], "outcomes_b": [],
                     "meta": {}, "aux": {"a": _aux_for(xa), "b": _aux_for(xb)}}
        rows.append(TV.PairRow({"x": xa}, {"x": xb}, 0.3, 0.05,
                               {"seed": f"P{i % 6:07d}", "step": i, "actor": 0, "state_kind": "hand",
                                "ante": 2, "player_fingerprint": "fp", "pair_source": "close_call",
                                "n_worlds": 8},
                               rec_extra))
    TV.save_pair_shard(tmp_path / "pair_0000.npz", rows)
    ds_dir = tmp_path / "abs"
    DS.save_shard(ds_dir / "s.npz", _rows("S0000000", 40, np.random.default_rng(1)))
    cfg = _cfg(ds_dir, tmp_path / "run", aux_heads=["all"], pair_shards=[str(tmp_path)],
               pair_batch_size=16, max_steps=5, eval_every=5, holdout_frac=0.0)
    tr = TV.VTrainer(cfg, log=lambda *_: None)
    assert tr.aux_pairs_on
    rec = tr.train_step()
    assert rec["pair_aux_loss"] > 0 and rec["pair_loss"] >= 0
    # the pair-branch aux term is the MEAN of the two branches, not their sum: they are the
    # two halves of ONE pair batch and no other term counts a batch twice
    tr2 = TV.VTrainer(cfg, log=lambda *_: None)
    obs_a, obs_b, _d, _c = tr2.next_pair_batch()
    ba, bb = TV.to_batch(obs_a, tr2.device), TV.to_batch(obs_b, tr2.device)
    cat = {k: torch.cat([ba[k], bb[k]], dim=0) for k in ba}
    n = len(_d)
    _logits, ap = tr2.net.forward_with_aux(cat)
    aa, ab = tr2.train_pair_aux
    la, _ = tr2._aux_loss({k: v[:n] for k, v in ap.items()}, tr2._last_pair_idx, aa)
    lb, _ = tr2._aux_loss({k: v[n:] for k, v in ap.items()}, tr2._last_pair_idx, ab)
    tr2.pair_epoch, tr2.pair_cursor = 0, 0        # rewind so _pair_loss draws the same batch
    tr2._pair_order_epoch = -1
    _rank, _info = tr2._pair_loss()
    assert float(tr2._last_pair_aux) == pytest.approx(0.5 * float(la + lb), rel=1e-5)
    # switching it off removes the term but keeps the ranking loss
    cfg2 = _cfg(ds_dir, tmp_path / "run2", aux_heads=["all"], pair_shards=[str(tmp_path)],
                pair_batch_size=16, max_steps=5, eval_every=5, holdout_frac=0.0, aux_on_pairs=False)
    tr2 = TV.VTrainer(cfg2, log=lambda *_: None)
    assert not tr2.aux_pairs_on
    assert "pair_aux_loss" not in tr2.train_step()


def test_resume_is_bit_exact_with_aux_state(aux_shards, tmp_path):
    """W-RANK's pinned resume test, extended to aux (brief §6b.4): weights AND every Adam
    moment AND the cursors match a straight run, with heads in the graph."""
    run = tmp_path / "run"
    cfg = _cfg(aux_shards, run, aux_heads=["all"], aux_hidden=8, max_steps=60, eval_every=30,
               checkpoint_every=30)
    TV.run(cfg, log=lambda *_: None)
    netA, _, extraA = TV.VTrainer.read_checkpoint(run / "latest.pt")
    sdA = {k: v.clone() for k, v in netA.state_dict().items()}
    optA = extraA["trainer"]["optimizer"]
    assert extraA["trainer"]["step"] == 60
    assert extraA["trainer"]["aux_heads"] == list(AX.AUX_NAMES)
    assert any("aux_heads" in k for k in sdA)
    TV.run(resume=str(run / "ckpt_0000030.pt"), overrides={"max_steps": 60}, log=lambda *_: None)
    netB, _, extraB = TV.VTrainer.read_checkpoint(run / "latest.pt")
    assert extraB["trainer"]["step"] == 60
    assert all(torch.equal(sdA[k], v) for k, v in netB.state_dict().items())
    optB = extraB["trainer"]["optimizer"]
    assert len(optA["param_groups"][0]["params"]) == len(optB["param_groups"][0]["params"])
    for sa, sb in zip(optA["state"].values(), optB["state"].values()):
        for a, b in zip(sa.values(), sb.values()):
            if isinstance(a, torch.Tensor):
                assert torch.equal(a, b)
    assert [h["step"] for h in extraB["trainer"]["history"]] == [0, 30, 60]


def test_heads_can_be_added_to_a_checkpoint_that_has_none(aux_shards, tmp_path):
    """The "bolt heads onto the keeper" path: fresh-init heads, the trunk carried over
    bit-exactly, the optimizer's existing moments preserved for the old params."""
    run = tmp_path / "run"
    TV.run(_cfg(aux_shards, run, max_steps=20, eval_every=20, checkpoint_every=20),
           log=lambda *_: None)
    netA, _, _ = TV.VTrainer.read_checkpoint(run / "latest.pt")
    tr = TV.VTrainer.from_checkpoint(run / "latest.pt", overrides={"aux_heads": ["all"], "max_steps": 25},
                                     log=lambda *_: None)
    assert tr.aux_on and sorted(tr.net.aux_head_names()) == sorted(AX.AUX_NAMES)
    for k, v in netA.state_dict().items():
        assert torch.equal(v, tr.net.state_dict()[k])       # the trunk is unchanged
    assert tr.step == 20
    rec = tr.train_step()                                   # and it still trains
    assert rec["step"] == 21 and "aux_loss" in rec


def test_pause_and_done_still_work_with_aux(aux_shards, tmp_path):
    run = tmp_path / "run"
    cfg = _cfg(aux_shards, run, aux_heads=["all"], max_steps=100, eval_every=1000,
               checkpoint_every=1000)
    pause = run / TV.PAUSE_FILE
    run.mkdir()
    hit = {"n": 0}

    def stop_check():
        hit["n"] += 1
        if hit["n"] == 8:
            pause.write_text("")
        return False

    summ = TV.run(cfg, log=lambda *_: None, stop_check=stop_check)
    assert summ["stop_reason"] == "PAUSE" and not summ["done"] and pause.exists()
    summ2 = TV.run(resume=str(run / "latest.pt"), overrides={"max_steps": 100}, log=lambda *_: None)
    assert not pause.exists() and summ2["stop_reason"] == "max_steps" and (run / TV.DONE_FILE).exists()


def test_fingerprint_filtering_still_works_with_aux(aux_shards, tmp_path):
    ds = DS.LabelDataset.load(aux_shards)
    for i, m in enumerate(ds.meta):
        m["player_fingerprint"] = "fp-new" if i % 2 == 0 else "fp-old"
    kept = TV.filter_by_fingerprint(ds, "new_only", "fp-new")
    assert len(kept) == len(ds) // 2
    ad = TV.aux_arrays_from_metas(kept.meta, list(AX.AUX_SPECS))
    assert ad.any_present and len(ad) == len(kept)


def test_cli_parses_aux_flags(aux_shards, tmp_path):
    run = tmp_path / "cli"
    rc = TV.main(["--shards", str(aux_shards), "--run-dir", str(run), "--model", "dummy",
                  "--max-steps", "6", "--eval-every", "3", "--checkpoint-every", "6",
                  "--device", "cpu", "--batch-size", "16", "--warmup-steps", "2",
                  "--holdout-frac", "0.25", "--aux-heads", "all", "--aux-weight", "0.25",
                  "--aux-weights", '{"blind_cleared": 0.5}', "--aux-hidden", "8",
                  "--aux-on-pairs", "0"])
    assert rc == 0 and (run / ".DONE").exists()
    _, _, extra = TV.VTrainer.read_checkpoint(run / "latest.pt")
    c = extra["trainer"]["config"]
    assert c["aux_heads"] == ["all"] and c["aux_weight"] == 0.25
    assert c["aux_weights"] == {"blind_cleared": 0.5} and c["aux_hidden"] == 8
    assert c["aux_on_pairs"] is False
    assert extra["trainer"]["aux_heads"] == list(AX.AUX_NAMES)
