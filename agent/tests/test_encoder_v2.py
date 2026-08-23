"""
test_encoder_v2.py — STATE_SPEC v1 encoder (Phase 5 W1).

What is pinned here:
  * the layout: every spec block present, in order, widths sum to SCALAR_DIM_V2, offsets
    are generated (the VALUE_NOTES table cannot drift from the code);
  * the vocabulary: built from the live game tables (sizes asserted against them, never a
    literal), the Phase 4 prefix is byte-identical, 32 spare ids, the spare-id promise;
  * the fingerprint: stable, sensitive to caps / layout / vocabulary;
  * determinism + order-freedom: same state -> bit-identical Obs; permuting the draw pile
    in place changes nothing;
  * no NaN / inf on every state of a full MLB match (both players, with opponent views);
  * the opponent block is built from PUBLIC information only (mutating the opponent's
    hand / jokers / shop / deck / consumables is invisible; lives / skips / $ are not);
  * blind offers, deck/discard counts, money_detail, race, econ: hand-checked values.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

from balatro_sim import game_keys as gk
from balatro_sim.card import Card
from balatro_sim.constants import INTEREST_CAP, MLB_NEMESIS_KEY, blind_base_chips
from balatro_sim.game import BalatroGame, State
from balatro_sim.mlb_match import MLBMatch, PlayerView

from mcts import encoder_v2 as E
from mcts.encoder_set import KEY_VOCAB, KEY_VOCAB_SIZE, SCALAR_DIM, SCALAR_LAYOUT, SetEncoder
from mcts.encoder_v2 import (
    DEFAULT_CAPS_V2, ItemCapsV2, KEY_VOCAB_SIZE_V2, KEY_VOCAB_V2, NO_OPPONENT, N_SPARE_KEYS,
    NemesisRecord, OpponentView, SCALAR_DIM_V2, SCALAR_LAYOUT_V2, STATE_SPEC_VERSION,
    SetEncoderV2, collate, key_index_v2, layout_fingerprint, opponent_view, scalar_offsets_v2,
)

from _states import collect_states

OFF = scalar_offsets_v2()
WIDTH = dict(SCALAR_LAYOUT_V2)


def _block(obs, name):
    o = OFF[name]
    return obs["scalars"][o:o + WIDTH[name]]


def _obs_equal(a, b) -> bool:
    return a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)


def play_match(seed="7I4M53DL", max_steps=4000, rng_seed=0, win_prob=0.7):
    """Walk a whole MLB match (random legal actions, most regular blinds auto-won so the
    match reaches its Nemeses and an end) yielding (match, player) before every step."""
    rng = np.random.default_rng(rng_seed)
    m = MLBMatch(seed=seed)
    while not m.done and m.steps < max_steps:
        p = m.current_player()
        g = m.games[p]
        if (g.state is State.SELECTING_HAND and not g.current_blind.is_pvp
                and rng.random() < win_prob):
            g.debug_win_blind()
            m.sync()
            continue
        acts = m.legal_actions(p)
        yield m, p
        m.step(p, acts[int(rng.integers(len(acts)))])
    yield m, 0
    yield m, 1


@pytest.fixture(scope="module")
def enc():
    return SetEncoderV2()


@pytest.fixture(scope="module")
def states():
    return collect_states(120, ruleset="mlb", rng_seed=3)


@pytest.fixture(scope="module")
def match_items():
    """(game clone, OpponentView) pairs along a full MLB match, plus the finished match."""
    items = []
    m = None
    for m, p in play_match():
        items.append((m.games[p].clone(), opponent_view(m, p)))
    return items, m


# ── layout ────────────────────────────────────────────────────────────────────

def test_spec_version_and_layout_sum():
    assert STATE_SPEC_VERSION == 1
    assert sum(w for _, w in SCALAR_LAYOUT_V2) == SCALAR_DIM_V2
    assert SCALAR_LAYOUT_V2[:len(SCALAR_LAYOUT)] == list(SCALAR_LAYOUT)
    names = [n for n, _ in SCALAR_LAYOUT_V2]
    assert len(set(names)) == len(names)
    for want in ("deck_counts", "discard_counts", "ruleset", "pvp_start_round", "opp_basic",
                 "opp_nemesis", "opp_history", "opp_belief", "opp_econ", "race",
                 "money_detail", "reserved"):
        assert want in WIDTH
    assert WIDTH["deck_counts"] == 34 and WIDTH["discard_counts"] == 17
    assert WIDTH["opp_belief"] == 16 and WIDTH["reserved"] == 32 and WIDTH["opp_econ"] == 4
    assert WIDTH["opp_history"] == 24 and WIDTH["race"] == 6 and WIDTH["money_detail"] == 6
    assert OFF["deck_counts"] == SCALAR_DIM
    table = E.scalar_layout_table()
    assert [(n, w) for n, w, _ in table] == SCALAR_LAYOUT_V2
    assert all(o == OFF[n] for n, _, o in table)


def test_vocab_is_built_from_the_live_tables():
    assert KEY_VOCAB_V2[:KEY_VOCAB_SIZE] == list(KEY_VOCAB)
    expected = (KEY_VOCAB_SIZE + len(gk.BOOSTER_CENTER_KEYS) + len(gk.TAG_KEYS)
                + len(gk.BLINDS) + 1 + 52 + len(gk.ENHANCEMENT_KEYS) + len(gk.EDITION_KEYS)
                + len(gk.SEAL_KEYS) + N_SPARE_KEYS)
    assert KEY_VOCAB_SIZE_V2 == expected == len(KEY_VOCAB_V2)
    assert len(set(KEY_VOCAB_V2)) == KEY_VOCAB_SIZE_V2
    assert N_SPARE_KEYS == 32
    assert KEY_VOCAB_V2[-N_SPARE_KEYS:] == [f"<spare_{i}>" for i in range(32)]
    for key in gk.JOKER_KEYS + gk.TAROT_KEYS + gk.PLANET_KEYS + gk.SPECTRAL_KEYS \
            + gk.VOUCHER_KEYS + gk.TAG_KEYS + gk.BOSS_KEYS + [MLB_NEMESIS_KEY, "H_A", "S_T"]:
        assert key_index_v2(key) > 1, key
    assert key_index_v2(None) == 0 and key_index_v2("") == 0
    assert key_index_v2("j_not_a_joker") == 1
    assert all(key_index_v2(k) < 512 for k in KEY_VOCAB_V2)     # int16 categoricals
    # the spare-id promise: new content maps to a spare slot without touching the vocabulary
    try:
        E.SPARE_KEY_MAP["j_modded"] = E.SPARE_KEY_BASE + 3
        assert key_index_v2("j_modded") == E.SPARE_KEY_BASE + 3
        assert KEY_VOCAB_V2[key_index_v2("j_modded")] == "<spare_3>"
    finally:
        E.SPARE_KEY_MAP.clear()


def test_fingerprint_is_stable_and_sensitive(monkeypatch):
    a = layout_fingerprint()
    assert a == layout_fingerprint() == layout_fingerprint(DEFAULT_CAPS_V2.as_dict())
    assert len(a) == 64
    assert layout_fingerprint(ItemCapsV2(hand=20)) != a
    monkeypatch.setattr(E, "SCALAR_LAYOUT_V2", list(SCALAR_LAYOUT_V2) + [("x", 1)])
    assert layout_fingerprint() != a
    monkeypatch.setattr(E, "SCALAR_LAYOUT_V2", list(SCALAR_LAYOUT_V2))
    monkeypatch.setattr(E, "KEY_VOCAB_V2", list(KEY_VOCAB_V2) + ["j_extra"])
    assert layout_fingerprint() != a
    monkeypatch.setattr(E, "STATE_SPEC_VERSION", 2)
    monkeypatch.setattr(E, "KEY_VOCAB_V2", list(KEY_VOCAB_V2))
    assert layout_fingerprint() != a


def test_describe_round_trips(enc):
    d = enc.describe()
    assert d["name"] == "set_v2" and d["scalar_dim"] == SCALAR_DIM_V2
    assert d["key_vocab"] == KEY_VOCAB_SIZE_V2 and d["state_spec_version"] == 1
    assert d["fingerprint"] == layout_fingerprint()
    e2 = SetEncoderV2.from_description(d)
    assert e2.caps == enc.caps and e2.describe() == d
    big = SetEncoderV2(ItemCapsV2(hand=20, blinds=4))
    assert SetEncoderV2.from_description(big.describe()).caps == big.caps
    with pytest.raises(ValueError):
        SetEncoderV2.from_description({**d, "fingerprint": "0" * 64})
    with pytest.raises(ValueError):
        SetEncoderV2.from_description({**d, "state_spec_version": 99})
    with pytest.raises(ValueError):
        SetEncoderV2.from_description({**d, "name": "set"})
    assert SetEncoderV2(SetEncoder().caps).caps == DEFAULT_CAPS_V2   # v1 caps promote


# ── shapes / determinism / finiteness ─────────────────────────────────────────

def test_shapes_and_dtypes(enc, states):
    caps = enc.caps
    want = {
        "hand_cat": ((caps.hand, 5), np.int16), "hand_num": ((caps.hand, 9), np.float32),
        "hand_mask": ((caps.hand,), np.float32),
        "joker_key": ((caps.jokers,), np.int16), "joker_cat": ((caps.jokers, 2), np.int16),
        "joker_num": ((caps.jokers, 16), np.float32), "joker_mask": ((caps.jokers,), np.float32),
        "cons_key": ((caps.consumables,), np.int16), "cons_num": ((caps.consumables, 8), np.float32),
        "cons_mask": ((caps.consumables,), np.float32),
        "shelf_key": ((caps.shelf,), np.int16), "shelf_cat": ((caps.shelf, 2), np.int16),
        "shelf_card": ((caps.shelf, 5), np.int16), "shelf_num": ((caps.shelf, 12), np.float32),
        "shelf_mask": ((caps.shelf,), np.float32),
        "pack_key": ((caps.packs,), np.int16), "pack_cat": ((caps.packs, 2), np.int16),
        "pack_card": ((caps.packs, 5), np.int16), "pack_num": ((caps.packs, 8), np.float32),
        "pack_mask": ((caps.packs,), np.float32),
        "blind_key": ((caps.blinds,), np.int16), "blind_tag": ((caps.blinds,), np.int16),
        "blind_cat": ((caps.blinds, 2), np.int16), "blind_num": ((caps.blinds, 8), np.float32),
        "blind_mask": ((caps.blinds,), np.float32),
        "scalars": ((SCALAR_DIM_V2,), np.float32),
    }
    for g in states[:40]:
        o = enc(g)
        assert set(o) == set(want)
        for k, (shape, dt) in want.items():
            assert o[k].shape == shape and o[k].dtype == dt, k
        assert np.all(np.isfinite(o["scalars"]))


def test_deterministic_and_clone_identical(enc, states):
    for g in states[:60]:
        a, b, c = enc(g), enc(g), enc(g.clone())
        assert _obs_equal(a, b) and _obs_equal(a, c)


def test_first_196_scalars_match_the_phase4_encoder(enc, states):
    base = SetEncoder()
    for g in states[:60]:
        assert np.array_equal(enc(g)["scalars"][:SCALAR_DIM], base(g)["scalars"])


def test_permuting_the_draw_pile_changes_nothing(enc, states):
    rng = random.Random(7)
    n_checked = 0
    for g in states:
        if len(g.deck) < 2:
            continue
        before = enc(g)
        rng.shuffle(g.deck)                       # in place, like the engine's own shuffle
        after = enc(g)
        assert _obs_equal(before, after)
        g.deck.reverse()
        assert _obs_equal(before, enc(g))
        n_checked += 1
    assert n_checked > 50


def test_no_nan_inf_on_a_full_mlb_match(enc, match_items):
    items, m = match_items
    assert m.done and m.winner in (0, 1) and len(m.pvp_log) >= 2
    seen = set()
    for g, opp in items:
        o = enc(g, opp)
        seen.add(g.state.name)
        for k, a in o.items():
            if a.dtype == np.float32:
                assert np.all(np.isfinite(a)), k
                assert np.abs(a).max() <= 10.0, k        # (Stone card chips/11 = 4.5 in hand_num)
        s = o["scalars"]
        assert s.min() >= -1.0 and s.max() <= 4.0
        assert _obs_equal(o, enc(g, opp))
    assert {"BLIND_SELECT", "SELECTING_HAND", "SHOP", "GAME_OVER"} <= seen
    assert len(items) > 150


def test_vanilla_game_has_an_all_zero_opponent_block(enc):
    g = BalatroGame(seed="7I4M53DL")
    for opp in (None, NO_OPPONENT, OpponentView()):
        o = enc(g, opp)
        for name in ("opp_basic", "opp_nemesis", "opp_history", "opp_belief", "opp_econ"):
            assert not _block(o, name).any(), name
        assert _block(o, "reserved").sum() == 0.0
        assert _block(o, "ruleset").tolist() == [1.0, 0.0, 0.0]
        assert _block(o, "pvp_start_round")[0] == 0.0
        assert _block(o, "race")[1] == 0.0
    assert _obs_equal(enc(g), enc(g, None))
    mlb = BalatroGame(seed="7I4M53DL", ruleset="mlb")
    o = enc(mlb)
    assert _block(o, "ruleset").tolist() == [0.0, 1.0, 0.0]
    assert _block(o, "pvp_start_round")[0] == pytest.approx(2 / 8)


# ── opponent block: public information only ───────────────────────────────────

def _first_shop_state():
    for m, p in play_match(rng_seed=11):
        other = 1 - p
        og = m.games[other]
        if og.state is State.SHOP and og.current_shop and og.jokers and m.games[p].state is State.SHOP:
            return m.clone(), p
    pytest.skip("no state with both players in the shop")


def test_opponent_view_reads_only_public_fields(enc):
    m, p = _first_shop_state()
    g, og = m.games[p], m.games[1 - p]
    v0 = opponent_view(m, p)
    o0 = enc(g, v0)
    # private mutations: hand, jokers, consumables, shop, deck, discard pile, planet levels
    og.hand.append(Card(rank=14, suit="Spades"))
    og.jokers.pop()
    og.consumable_hand.append("c_fool")
    og.current_shop.pop()
    og.deck.reverse()
    og.deck.pop()
    og.discard_pile.append(Card(rank=2, suit="Hearts"))
    og.planet_levels["Pair"] += 5
    og.vouchers.add("v_overstock_norm")
    assert opponent_view(m, p) == v0
    assert _obs_equal(enc(g, opponent_view(m, p)), o0)
    # public mutations change it
    og.lives -= 1
    v1 = opponent_view(m, p)
    assert v1 != v0 and v1.lives == v0.lives - 1
    assert not _obs_equal(enc(g, v1), o0)
    og.skips += 1
    assert opponent_view(m, p).skips == v0.skips + 1
    og.dollars += 7
    v3 = opponent_view(m, p)
    assert v3.dollars == v0.dollars + 7
    o3 = enc(g, v3)
    b = _block(o3, "opp_basic")
    assert b[0] == 1.0 and b[1] == pytest.approx((v0.lives - 1) / 4)
    assert b[2] == pytest.approx((v0.skips + 1) / 8) and b[3] == pytest.approx((v0.dollars + 7) / 50)
    assert b[6:].tolist() == [0, 0, 0, 0, 1, 0]           # phase: shop
    assert _block(o3, "race")[1] == pytest.approx((v0.lives - 1) / 4)


def test_opponent_view_is_symmetric_and_player_view_is_public():
    m = MLBMatch(seed="7I4M53DL")
    for p in (0, 1):
        v = opponent_view(m, p)
        assert v.known and v.lives == 4 and v.skips == 0 and v.state == "BLIND_SELECT"
        assert v.phase == "selecting" and v.current_nemesis is None and v.last_nemeses == []
        assert v.my_last_loss_ante is None
    pv = m.player_view(0)
    assert isinstance(pv, PlayerView) and pv.hands_played == 0 and pv.sells_per_ante == 0
    assert NO_OPPONENT.phase is None and not NO_OPPONENT.known


def test_nemesis_history_and_race(enc, match_items):
    items, m = match_items
    assert len(m.pvp_detail) == len(m.pvp_log)
    for (a, l, s0, s1), (a2, l2, s02, s12, h0, h1, early) in zip(m.pvp_log, m.pvp_detail):
        assert (a, l, s0, s1) == (a2, l2, s02, s12)
        assert 0 <= h0 <= 8 and 0 <= h1 <= 8 and isinstance(early, bool)
    for p in (0, 1):
        v = opponent_view(m, p)
        assert len(v.last_nemeses) == min(4, len(m.pvp_log))
        recs = v.last_nemeses
        assert [r.ante for r in recs] == [a for a, *_ in m.pvp_log[::-1][:4]]   # most recent first
        for r, (a, loser, s0, s1) in zip(recs, m.pvp_log[::-1]):
            assert isinstance(r, NemesisRecord)
            assert r.my_score == (s0, s1)[p] and r.their_score == (s0, s1)[1 - p]
            assert r.outcome == (0 if loser is None else (1 if loser == 1 - p else -1))
        my_losses = [a for a, loser, *_ in m.pvp_log if loser == p]
        assert v.my_last_loss_ante == m.last_life_loss_ante[p]
        if my_losses:                                     # a Nemesis loss is never missed
            assert v.my_last_loss_ante is not None and v.my_last_loss_ante >= my_losses[-1]
        if m.games[p].lives == 4:
            assert v.my_last_loss_ante is None
        o = enc(m.games[p], v)
        h = _block(o, "opp_history").reshape(4, 6)
        for i, r in enumerate(recs):
            assert h[i, 0] == pytest.approx(min(r.ante / 8, 4))
            assert h[i, 4] == r.outcome and h[i, 5] == float(r.early_end)
            assert h[i, 2] == pytest.approx(r.their_hands_used / 4)
        rc = _block(o, "race")
        assert rc[0] == pytest.approx(m.games[p].lives / 4) and rc[1] == pytest.approx(v.lives / 4)
        assert rc[2] == pytest.approx(min(m.games[p].ante / 8, 4))
    # the loser has 0 lives, the winner does not; the log's last loser is the match loser
    assert m.games[1 - m.winner].lives == 0 and m.games[m.winner].lives > 0


def test_last_life_loss_ante_counts_failed_regular_blinds():
    m = MLBMatch(seed="7I4M53DL")
    g = m.games[0]
    m.step(0, {"type": "play_blind"})
    assert g.state is State.SELECTING_HAND
    while g.state is State.SELECTING_HAND and g.lives == 4:      # play junk until the Small fails
        m.step(0, {"type": "play", "cards": [0]})
    assert g.lives == 3 and not m.pvp_log
    assert m.player_view(0).last_life_loss_ante == 1 and m.player_view(1).last_life_loss_ante is None
    assert opponent_view(m, 0).my_last_loss_ante == 1 and opponent_view(m, 1).my_last_loss_ante is None
    o = SetEncoderV2()(g, opponent_view(m, 0))
    assert _block(o, "race")[5] == 0.0                             # 0 antes since the loss
    assert _block(o, "race")[4] == 1.0                             # lost this round
    c = m.clone()
    assert c.last_life_loss_ante == m.last_life_loss_ante


def test_live_nemesis_block(enc):
    found = False
    for m, p in play_match(rng_seed=5):
        if m.pvp_active:
            v = opponent_view(m, p)
            assert v.current_nemesis is not None
            g, og = m.games[p], m.games[1 - p]
            assert v.current_nemesis.my_score == g.chips_scored
            assert v.current_nemesis.their_score == og.chips_scored
            n = _block(enc(g, v), "opp_nemesis")
            assert n[0] == pytest.approx(np.log1p(og.chips_scored) / np.log1p(1e5))
            assert n[1] == pytest.approx(min(og.hands_left / 4, 2))
            assert n[2] == pytest.approx(np.log1p(g.chips_scored) / np.log1p(1e5))
            assert v.phase in ("nemesis", "waiting")
            found = True
            break
    assert found


def test_econ_counters_follow_public_actions():
    m = MLBMatch(seed="7I4M53DL")
    g = m.games[0]
    g.debug_win_blind() if g.state is State.SELECTING_HAND else None
    # drive player 0 into a shop with a joker to sell
    rng = np.random.default_rng(0)
    while g.state is not State.SHOP:
        if g.state is State.SELECTING_HAND:
            g.debug_win_blind()
            m.sync()
        else:
            acts = m.legal_actions(0)
            m.step(0, acts[0])
    buys = [a for a in m.legal_actions(0) if a["type"] == "buy"]
    before = g.dollars
    if buys:
        m.step(0, buys[0])
        spent = before - g.dollars
        assert m.econ[0].spent_in_shop == spent and m.econ[0].spent_total == spent
        assert opponent_view(m, 1).spent_in_shop == spent
    if g.state is State.BOOSTER_OPEN:
        m.step(0, {"type": "skip_booster"})
    if g.jokers:
        m.step(0, {"type": "sell_joker", "joker_idx": 0})
        assert m.econ[0].sells_per_ante == 1 and m.econ[0].sells_total == 1
        v = opponent_view(m, 1)
        assert v.sells_per_ante == 1 and v.sells_total == 1
        e = _block(SetEncoderV2()(m.games[1], v), "opp_econ")
        assert e[0] == pytest.approx(1 / 5) and e[2] == pytest.approx(np.log1p(1) / np.log1p(50))
    # per-ante counters reset when the ante moves; totals persist
    m.econ[0].ante = g.ante - 1
    assert m.player_view(0).sells_per_ante == 0 and m.player_view(0).sells_total == m.econ[0].sells_total
    c = m.clone()
    assert c.econ[0] == m.econ[0] and c.econ[0] is not m.econ[0]
    assert c.pvp_detail == m.pvp_detail


# ── blind offers ──────────────────────────────────────────────────────────────

def test_blind_offers_at_the_first_blind_select(enc):
    g = BalatroGame(seed="7I4M53DL")
    o = enc(g)
    assert o["blind_mask"].tolist() == [1, 1, 1]
    assert o["blind_cat"][:, 0].tolist() == [1, 2, 3]
    assert o["blind_cat"][:, 1].tolist() == [2, 1, 1]             # current, upcoming, upcoming
    assert o["blind_tag"][0] == key_index_v2(g.blind_tags["Small"]) > 1
    assert o["blind_tag"][1] == key_index_v2(g.blind_tags["Big"]) > 1
    assert o["blind_tag"][2] == 0 and o["blind_key"][0] == 0 and o["blind_key"][1] == 0
    assert o["blind_key"][2] == key_index_v2(g.boss_blind) > 1
    num = o["blind_num"]
    for slot in range(3):
        target = blind_base_chips(1, slot, 1)
        if slot == 2:
            from balatro_sim.game import BOSS_CHIP_MULT
            target = int(target * BOSS_CHIP_MULT.get(g.boss_blind, 1.0))
        assert num[slot, 0] == pytest.approx(np.log1p(target) / np.log1p(1e5), abs=1e-6)
    assert num[0, 0] == pytest.approx(np.log1p(g.current_blind.chips_target) / np.log1p(1e5))
    assert num[:, 1].tolist() == pytest.approx([3 / 8, 4 / 8, 5 / 8])
    assert num[:, 2].tolist() == [0, 0, 0]                          # no pvp in vanilla
    assert num[:, 3].tolist() == [1, 0, 0]                          # skip available: Small only
    assert num[:, 4].tolist() == [1, 0, 0] and num[:, 5].tolist() == [0, 0, 0]
    # skip the Small: slot 0 done, Big current and skippable
    g.step({"type": "skip_blind"})
    o = enc(g)
    assert o["blind_cat"][:, 1].tolist() == [3, 2, 1]
    assert o["blind_num"][:, 3].tolist() == [0, 1, 0] and o["blind_num"][:, 5].tolist() == [1, 0, 0]
    # while playing, nothing is skippable
    g.step({"type": "play_blind"})
    assert g.state is State.SELECTING_HAND
    assert enc(g)["blind_num"][:, 3].tolist() == [0, 0, 0]


def test_blind_offers_show_the_nemesis_under_mlb(enc):
    seen_shop = seen_select = False
    for m, p in play_match(rng_seed=2):
        g = m.games[p]
        if g.ante >= 2:
            o = enc(g, opponent_view(m, p))
            assert o["blind_key"][2] == key_index_v2(MLB_NEMESIS_KEY)
            assert o["blind_num"][2, 2] == 1.0                      # is_pvp
            assert o["blind_num"][2, 1] == pytest.approx(5 / 8)      # Nemesis pays $5
            if g.state is State.SHOP and g.current_blind.is_boss:   # the post-boss shop
                assert o["blind_cat"][:, 1].tolist() == [1, 1, 1]   # next ante: all upcoming
                assert o["blind_num"][:, 4].tolist() == [0, 0, 0]
                assert o["blind_tag"][0] == key_index_v2(g.blind_tags["Small"])
                assert o["blind_num"][0, 0] == pytest.approx(
                    np.log1p(blind_base_chips(g.ante, 0, 1)) / np.log1p(1e5))
                seen_shop = True
            if g.state is State.BLIND_SELECT and g.blind_idx == 2:
                assert o["blind_cat"][:, 1].tolist() == [3, 3, 2] and o["blind_num"][2, 3] == 0.0
                seen_select = True
            if seen_shop and seen_select:
                break
    assert seen_shop and seen_select
    g = BalatroGame(seed="7I4M53DL", ruleset="mlb")
    o = enc(g)                                                      # ante 1: a real boss
    assert o["blind_key"][2] == key_index_v2(g.boss_blind) and o["blind_num"][2, 2] == 0.0


# ── counts / money / caps ─────────────────────────────────────────────────────

def test_deck_and_discard_counts(enc):
    g = BalatroGame(seed="7I4M53DL")
    g.step({"type": "play_blind"})
    assert g.state is State.SELECTING_HAND
    o = enc(g)
    dc = _block(o, "deck_counts")
    ranks = dc[:13] * 8
    suits = dc[13:17] * 26
    assert ranks.sum() == pytest.approx(len(g.deck)) and suits.sum() == pytest.approx(len(g.deck))
    for r in range(2, 15):
        assert ranks[r - 2] == pytest.approx(sum(1 for c in g.deck if c.rank == r))
    assert dc[17:25].sum() == 0.0                                   # no enhancements yet
    assert dc[25] == pytest.approx(len(g.deck) / 52)                # all plain editions
    assert dc[26:].sum() == 0.0
    assert not _block(o, "discard_counts").any()
    g.step({"type": "discard", "cards": [0, 1, 2]})
    o = enc(g)
    disc = _block(o, "discard_counts")
    assert (disc[:13] * 8).sum() == pytest.approx(3) and (disc[13:] * 26).sum() == pytest.approx(3)
    # an enhanced / sealed / edition card in the pile is counted
    c = g.deck[0]
    c.enhancement, c.edition, c.seal = "Glass", "Foil", "Red"
    dc = _block(enc(g), "deck_counts")
    assert dc[17 + 3] == pytest.approx(1 / 16) and dc[25 + 1] == pytest.approx(1 / 16)
    assert dc[30 + 1] == pytest.approx(1 / 16)


def test_money_detail(enc):
    g = BalatroGame(seed="7I4M53DL")
    g.dollars = 13
    m = _block(enc(g), "money_detail")
    assert m[0] == pytest.approx(2 / 10) and m[1] == pytest.approx(2 / 5)   # $2 interest, $2 to $15
    g.dollars = 25 + 3
    m = _block(enc(g), "money_detail")
    assert m[0] == pytest.approx(INTEREST_CAP / 10) and m[1] == 0.0       # capped: nothing to gain
    g.no_interest = True
    m = _block(enc(g), "money_detail")
    assert m[0] == 0.0 and m[1] == 0.0
    g.no_interest = False
    g.reroll_cost, g.reroll_discount, g.free_rerolls_remaining, g.shop_discount = 7, 2, 1, 0.25
    m = _block(enc(g), "money_detail")
    assert m[2] == pytest.approx(0.5) and m[3] == pytest.approx(1 / 3) and m[4] == 0.25
    assert m[5] == 0.0                                                   # no voucher on offer


def test_larger_caps_write_the_same_rows_plus_padding(states):
    small, big = SetEncoderV2(), SetEncoderV2(ItemCapsV2(hand=24, jokers=16, consumables=8,
                                                         shelf=12, packs=10, blinds=5))
    assert small.fingerprint != big.fingerprint
    for g in states[:40]:
        a, b = small(g), big(g)
        assert np.array_equal(a["scalars"], b["scalars"])          # capacity ratios use fixed caps
        for k in a:
            if k == "scalars":
                continue
            n = a[k].shape[0]
            assert np.array_equal(a[k], b[k][:n]), k
            assert not b[k][n:].any(), k


def test_collate_stacks(enc, states):
    obs = [enc(g) for g in states[:5]]
    batch = collate(obs)
    assert batch["scalars"].shape == (5, SCALAR_DIM_V2)
    assert batch["blind_num"].shape == (5, 3, 8) and str(batch["hand_cat"].dtype) == "torch.int16"
    assert np.array_equal(batch["scalars"].numpy(), np.stack([o["scalars"] for o in obs]))


def test_encoder_speed(enc, match_items):
    import time
    items, _ = match_items
    sample = items[:300]
    for g, o in sample[:20]:
        enc(g, o)
    t = time.perf_counter()
    for g, o in sample:
        enc(g, o)
    ms = (time.perf_counter() - t) * 1000 / len(sample)
    assert ms < 2.0, f"{ms:.3f} ms/state"           # gate target is <= 1 ms; CI boxes vary
