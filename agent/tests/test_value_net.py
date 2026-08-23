"""
test_value_net.py — `SetValueNet` (Phase 5 W1).

Pinned: the 5M budget + breakdown; (B,) finite logits; pad-invariance (appended padded
slots AND a larger-caps encoder give the same logit); within-set permutation invariance;
garbage in padded rows is invisible; the empty state does not NaN; bit-exact checkpoint
round trip; fingerprint / version / kind mismatches are refused; `make_value_fn` /
`make_values_many` agree with the raw forward and preserve order; CPU == CUDA; a
gradient step moves the loss.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from balatro_sim.game import BalatroGame, State
from balatro_sim.mlb_match import MLBMatch

from mcts.encoder_v2 import (
    DEFAULT_CAPS_V2, ItemCapsV2, SCALAR_DIM_V2, STATE_SPEC_VERSION, SetEncoderV2, collate,
    opponent_view,
)
from mcts.value_net import (
    SetValueNet, ValueNetConfig, load_checkpoint, make_value_fn, make_values_many,
    save_checkpoint, VALUE_CHECKPOINT_KIND,
)

from _states import collect_states

torch.set_num_threads(min(4, torch.get_num_threads()))     # shared box (lead, 2026-08-23)

SMALL = ValueNetConfig(d_item=32, key_emb=16, card_emb=4, aux_emb=4, trunk_width=64,
                       n_res_blocks=1, scalar_hidden=32)


@pytest.fixture(scope="module")
def enc():
    return SetEncoderV2()


@pytest.fixture(scope="module")
def items():
    """(game, opp) pairs: 60 fixture states (no opponent) + 60 along a real MLB match."""
    out = [(g, None) for g in collect_states(60, ruleset="mlb", rng_seed=5)]
    rng = np.random.default_rng(0)
    m = MLBMatch(seed="7I4M53DL")
    while not m.done and m.steps < 2000 and len(out) < 120:
        p = m.current_player()
        g = m.games[p]
        if g.state is State.SELECTING_HAND and not g.current_blind.is_pvp and rng.random() < 0.7:
            g.debug_win_blind()
            m.sync()
            continue
        acts = m.legal_actions(p)
        if m.steps % 3 == 0:
            out.append((g.clone(), opponent_view(m, p)))
        m.step(p, acts[int(rng.integers(len(acts)))])
    return out


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    n = SetValueNet(ValueNetConfig())
    n.eval()
    return n


@pytest.fixture(scope="module")
def small():
    torch.manual_seed(1)
    n = SetValueNet(SMALL)
    n.eval()
    return n


def _fwd(net, obs_list, device="cpu"):
    with torch.no_grad():
        return net(collate(obs_list, device)).cpu()


# ── budget / shapes ───────────────────────────────────────────────────────────

def test_param_budget_and_breakdown(net):
    n = net.n_params()
    assert 4_500_000 <= n <= 5_500_000, n
    bd = net.param_breakdown()
    assert sum(bd.values()) == n
    assert bd["res_blocks"] > bd["trunk_in"] > bd["item MLPs (6 types)"]
    assert bd["value_head"] == net.cfg.trunk_width + 1
    d = net.describe()
    assert d["kind"] == "value_v1" and d["scalar_dim"] == SCALAR_DIM_V2
    assert SetValueNet.from_description(d).n_params() == n


def test_forward_is_finite_logits(net, enc, items):
    obs = [enc(g, o) for g, o in items]
    logits = _fwd(net, obs)
    assert logits.shape == (len(items),) and logits.dtype == torch.float32
    assert torch.isfinite(logits).all()
    p = logits.sigmoid()
    assert (p > 0).all() and (p < 1).all()
    assert logits.std() > 0                                  # not collapsed at init
    with torch.no_grad():
        assert torch.allclose(net.p_win(collate(obs[:4])), logits[:4].sigmoid())


# ── invariances ───────────────────────────────────────────────────────────────

def _pad_obs(obs, extra):
    """Append `extra` padded rows to every set (the transport-cap change on the wire)."""
    sets = {"hand": ("hand_cat", "hand_num", "hand_mask"),
            "joker": ("joker_key", "joker_cat", "joker_num", "joker_mask"),
            "cons": ("cons_key", "cons_num", "cons_mask"),
            "shelf": ("shelf_key", "shelf_cat", "shelf_card", "shelf_num", "shelf_mask"),
            "pack": ("pack_key", "pack_cat", "pack_card", "pack_num", "pack_mask"),
            "blind": ("blind_key", "blind_tag", "blind_cat", "blind_num", "blind_mask")}
    out = dict(obs)
    for keys in sets.values():
        for k in keys:
            a = obs[k]
            pad = np.zeros((extra,) + a.shape[1:], dtype=a.dtype)
            out[k] = np.concatenate([a, pad], axis=0)
    return out


def test_appending_padded_slots_changes_nothing(net, enc, items):
    obs = [enc(g, o) for g, o in items[:48]]
    base = _fwd(net, obs)
    padded = [_pad_obs(o, 4) for o in obs]
    big_caps = ItemCapsV2(hand=20, jokers=16, consumables=10, shelf=12, packs=12, blinds=7)
    net.caps = big_caps                                      # the net only uses caps to split keys
    try:
        got = _fwd(net, padded)
    finally:
        net.caps = DEFAULT_CAPS_V2
    assert torch.allclose(base, got, atol=1e-4), (base - got).abs().max()


def test_larger_caps_encoder_gives_the_same_value(net, items):
    big_caps = ItemCapsV2(hand=24, jokers=16, consumables=8, shelf=12, packs=10, blinds=5)
    enc_small, enc_big = SetEncoderV2(), SetEncoderV2(big_caps)
    net_big = SetValueNet(ValueNetConfig(caps=big_caps.as_dict()))
    net_big.load_state_dict(net.state_dict())                # same weights, bigger transport
    net_big.eval()
    a = _fwd(net, [enc_small(g, o) for g, o in items[:48]])
    b = _fwd(net_big, [enc_big(g, o) for g, o in items[:48]])
    assert torch.allclose(a, b, atol=1e-4), (a - b).abs().max()


def test_permuting_items_within_a_set_changes_nothing(net, enc, items):
    rng = np.random.default_rng(3)
    obs = [enc(g, o) for g, o in items[:48]]
    base = _fwd(net, obs)
    perm = []
    for o in obs:
        o2 = dict(o)
        for keys in (("hand_cat", "hand_num", "hand_mask"),
                     ("joker_key", "joker_cat", "joker_num", "joker_mask"),
                     ("shelf_key", "shelf_cat", "shelf_card", "shelf_num", "shelf_mask"),
                     ("blind_key", "blind_tag", "blind_cat", "blind_num", "blind_mask")):
            n = o[keys[0]].shape[0]
            p = rng.permutation(n)
            for k in keys:
                o2[k] = o[k][p]
        perm.append(o2)
    got = _fwd(net, perm)
    assert torch.allclose(base, got, atol=1e-4), (base - got).abs().max()


def test_garbage_in_padded_rows_changes_nothing(net, enc, items):
    rng = np.random.default_rng(4)
    obs = [enc(g, o) for g, o in items[:48]]
    base = _fwd(net, obs)
    dirty = []
    for o in obs:
        o2 = {k: v.copy() for k, v in o.items()}
        for mask_key, keys in (("hand_mask", ("hand_cat", "hand_num")),
                               ("joker_mask", ("joker_key", "joker_cat", "joker_num")),
                               ("cons_mask", ("cons_key", "cons_num")),
                               ("shelf_mask", ("shelf_key", "shelf_cat", "shelf_card", "shelf_num")),
                               ("pack_mask", ("pack_key", "pack_cat", "pack_card", "pack_num")),
                               ("blind_mask", ("blind_key", "blind_tag", "blind_cat", "blind_num"))):
            dead = o2[mask_key] <= 0
            for k in keys:
                a = o2[k]
                if a.dtype == np.float32:
                    a[dead] = rng.normal(size=a[dead].shape).astype(np.float32) * 5
                else:
                    a[dead] = rng.integers(0, 4, size=a[dead].shape).astype(a.dtype)
        dirty.append(o2)
    got = _fwd(net, dirty)
    assert torch.allclose(base, got, atol=1e-4), (base - got).abs().max()


def test_empty_state_does_not_nan(net, enc):
    o = enc(BalatroGame(seed="7I4M53DL"))
    for k in o:
        o[k] = np.zeros_like(o[k])
    v = _fwd(net, [o])
    assert v.shape == (1,) and torch.isfinite(v).all()


# ── checkpoints ───────────────────────────────────────────────────────────────

def test_checkpoint_round_trip_is_bit_exact(small, enc, items, tmp_path):
    path = tmp_path / "v.pt"
    extra = {"step": 123, "labels": 4567, "note": "w1"}
    save_checkpoint(path, small, enc, extra)
    assert path.exists() and not path.with_suffix(".pt.tmp").exists()
    net2, enc2, extra2 = load_checkpoint(path)
    assert extra2 == extra and extra2 is not extra
    assert enc2.describe() == enc.describe() and enc2.caps == enc.caps
    assert net2.cfg == small.cfg and not net2.training
    sd1, sd2 = small.state_dict(), net2.state_dict()
    assert sd1.keys() == sd2.keys()
    for k in sd1:
        assert torch.equal(sd1[k], sd2[k]), k
    obs = [enc(g, o) for g, o in items[:16]]
    assert torch.equal(_fwd(small, obs), _fwd(net2, obs))
    payload = torch.load(path, weights_only=False)
    assert payload["kind"] == VALUE_CHECKPOINT_KIND and payload["state_spec_version"] == STATE_SPEC_VERSION
    assert payload["fingerprint"] == enc.fingerprint and payload["n_params"] == small.n_params()


def test_checkpoint_refuses_mismatches(small, enc, tmp_path):
    path = tmp_path / "v.pt"
    save_checkpoint(path, small, enc, {})
    payload = torch.load(path, weights_only=False)

    def write(mod):
        p2 = dict(payload)
        mod(p2)
        bad = tmp_path / "bad.pt"
        torch.save(p2, bad)
        return bad

    with pytest.raises(ValueError, match="fingerprint"):
        load_checkpoint(write(lambda p: p.update(fingerprint="f" * 64)))
    with pytest.raises(ValueError, match="STATE_SPEC_VERSION"):
        load_checkpoint(write(lambda p: p.update(state_spec_version=STATE_SPEC_VERSION + 1)))
    with pytest.raises(ValueError, match="version"):
        load_checkpoint(write(lambda p: p.update(version=99)))
    with pytest.raises(ValueError, match="not a"):
        load_checkpoint(write(lambda p: p.update(kind="something else")))
    # a checkpoint written with DIFFERENT caps carries a different fingerprint and loads
    # back with those caps (caps are part of the fingerprint, so they cannot drift)
    big = SetEncoderV2(ItemCapsV2(hand=20))
    net_big = SetValueNet(ValueNetConfig(**{**SMALL.as_dict(), "caps": big.caps.as_dict()}))
    p3 = tmp_path / "big.pt"
    save_checkpoint(p3, net_big, big, {})
    n3, e3, _ = load_checkpoint(p3)
    assert e3.caps == big.caps and n3.caps == big.caps
    with pytest.raises(ValueError):                         # foreign pickle
        torch.save({"hello": 1}, tmp_path / "x.pt")
        load_checkpoint(tmp_path / "x.pt")


# ── value functions ───────────────────────────────────────────────────────────

def test_value_fns_agree_with_the_forward_and_keep_order(small, enc, items):
    many = make_values_many(small, enc, "cpu", chunk=7)       # chunking must not matter
    one = make_value_fn(small, enc, "cpu")
    batch = items[:23]
    v = many(batch)
    assert v.shape == (23,) and v.dtype == np.float32
    assert np.all((v > 0) & (v < 1))
    ref = _fwd(small, [enc(g, o) for g, o in batch]).sigmoid().numpy()
    assert np.allclose(v, ref, atol=1e-5)
    for i in (0, 5, 22):
        assert one(batch[i][0], batch[i][1]) == pytest.approx(float(v[i]), abs=1e-5)
    g = batch[0][0]
    assert one(g) == one(g, None)                             # opp=None accepted everywhere
    assert many([]).shape == (0,)
    rev = many(batch[::-1])
    assert np.allclose(rev, v[::-1], atol=1e-5)
    assert not small.training                                 # mode restored


def test_gradient_step_moves_the_loss(enc, items):
    torch.manual_seed(2)
    net = SetValueNet(SMALL)
    obs = collate([enc(g, o) for g, o in items[:32]])
    y = torch.tensor([float(i % 2) for i in range(32)])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    l0 = loss_fn(net(obs), y)
    for _ in range(20):
        opt.zero_grad()
        loss = loss_fn(net(obs), y)
        loss.backward()
        opt.step()
    assert loss_fn(net(obs), y) < l0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_cuda_matches_cpu(net, enc, items):
    obs = [enc(g, o) for g, o in items[:32]]
    a = _fwd(net, obs)
    net_c = SetValueNet(net.cfg)
    net_c.load_state_dict(net.state_dict())
    net_c.to("cuda").eval()
    b = _fwd(net_c, obs, "cuda")
    assert torch.allclose(a, b, atol=1e-4), (a - b).abs().max()
    many = make_values_many(net_c, enc, "cuda")
    v = many(items[:32])
    assert np.allclose(v, a.sigmoid().numpy(), atol=1e-4)
