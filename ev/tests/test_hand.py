"""hand.py — the analytic hand player: side-effect freedom, legality, draw-order
invariance, the combinatorics, the blind model's sanity, and a few decisions a first-year
player would make."""
from __future__ import annotations

import itertools
import random
from math import comb

import pytest

import _bootstrap  # noqa: F401
from _bootstrap import BalatroGame, State
from balatro_sim.card import Card

import hand as H
import sampling as S


# ───────────────────────────────────────────────────────────────────── fixtures

def _in_blind(seed="11111111", ruleset="vanilla", to_boss=False):
    g = BalatroGame(seed=seed, ruleset=ruleset)
    if to_boss:
        while not (g.current_blind.kind == "Boss" and g.state == State.SELECTING_HAND):
            s = g.state
            if s == State.SELECTING_HAND:
                g.debug_win_blind()
            elif s == State.BLIND_SELECT:
                g.step({"type": "play_blind"})
            elif s == State.ROUND_EVAL:
                g.step({"type": "advance"})
            elif s == State.SHOP:
                g.step({"type": "leave_shop"})
            elif s == State.BOOSTER_OPEN:
                g.step({"type": "skip_booster"})
        return g
    g.step({"type": "play_blind"})
    assert g.state == State.SELECTING_HAND
    return g


def _set_hand(g, specs):
    """Put specific cards (rank, suit) in hand, taken out of the draw pile / full deck."""
    pool = {(c.rank, c.suit): c for c in g.full_deck}
    hand = [pool[s] for s in specs]
    ids = {id(c) for c in hand}
    g.deck = [c for c in g.full_deck if id(c) not in ids]
    g.hand = hand
    g.discard_pile = []
    for c in hand:
        c.face_down = False
    return g


def _legal_keys(g):
    return {H._action_sort_key(a) for a in g.legal_actions()}


# ──────────────────────────────────────────────────────────────── combinatorics

def test_hypergeometric_tail_matches_brute_force():
    N, K, n = 12, 5, 4
    cards = list(range(N))
    for k in range(0, 6):
        hits = sum(1 for draw in itertools.combinations(cards, n) if sum(1 for x in draw if x < K) >= k)
        assert H._hyper_tail(N, K, n, k) == pytest.approx(hits / comb(N, n))


def test_p_all_groups_matches_brute_force():
    N, n = 11, 4
    groups = {0: "a", 1: "a", 2: "b", 3: "b", 4: "b", 5: "c"}
    sizes = (2, 3, 1)
    hits = 0
    for draw in itertools.combinations(range(N), n):
        got = {groups[x] for x in draw if x in groups}
        if got == {"a", "b", "c"}:
            hits += 1
    assert H._p_all_groups(N, n, sizes) == pytest.approx(hits / comb(N, n))
    assert H._p_all_groups(N, 2, (3, 4, 4)) == 0.0      # three groups need three draws


# ──────────────────────────────────────────────────────────── side-effect freedom

@pytest.mark.parametrize("budget", ["fast", "full"])
def test_rank_hand_actions_is_side_effect_free(budget):
    g = _in_blind()
    sig = g.state_signature()
    snap = g.run_state.rng.snapshot()
    ranked = H.rank_hand_actions(g, budget=budget)
    assert ranked
    assert g.state_signature() == sig
    assert g.run_state.rng.snapshot() == snap


def test_hand_ev_is_side_effect_free_and_consistent_with_ranking():
    g = _in_blind()
    sig = g.state_signature()
    ranked = H.rank_hand_actions(g)
    top, ev = ranked[0]
    assert H.hand_ev(g, top) == pytest.approx(ev)
    # a legal action that is not a structural candidate still gets a value
    legal = g.legal_actions()
    cand = {H._action_sort_key(a) for a, _ in ranked}
    other = next(a for a in legal if H._action_sort_key(a) not in cand)
    v = H.hand_ev(g, other)
    assert isinstance(v, float)
    assert H.hand_ev(g, top, budget="full", n_worlds=2) >= 0.0
    assert g.state_signature() == sig


# ───────────────────────────────────────────────────────────────────── legality

@pytest.mark.parametrize("seed", ["11111111", "1558AXDL", "2GHBLJD9"])
def test_every_ranked_action_is_legal(seed):
    g = _in_blind(seed)
    legal = _legal_keys(g)
    for a, ev in H.rank_hand_actions(g):
        assert H._action_sort_key(a) in legal


def test_psychic_boss_only_five_card_plays():
    g = _in_blind()
    g.current_blind.boss_key = "bl_psychic"
    g.current_blind.is_boss = True
    legal = _legal_keys(g)
    ranked = H.rank_hand_actions(g)
    assert ranked
    for a, _ in ranked:
        assert H._action_sort_key(a) in legal
        assert a["type"] == "play" and len(a["cards"]) == 5


# ───────────────────────────────────────────────────────── draw-order invariance

@pytest.mark.parametrize("budget", ["fast", "full"])
def test_decision_is_invariant_under_a_draw_pile_permutation(budget):
    g = _in_blind("1KV4W6YS")
    before = [a for a, _ in H.rank_hand_actions(g, budget=budget)]
    for i in range(3):
        random.Random(i).shuffle(g.deck)
        after = [a for a, _ in H.rank_hand_actions(g, budget=budget)]
        assert after == before


def test_decision_depends_on_the_pile_composition():
    """Removing every remaining heart from the pile must kill a heart-flush chase."""
    g = _in_blind()
    _set_hand(g, [(14, "Hearts"), (7, "Hearts"), (6, "Hearts"), (12, "Hearts"),
                  (11, "Spades"), (13, "Clubs"), (3, "Clubs"), (6, "Diamonds")])
    top, _ = H.rank_hand_actions(g)[0]
    assert top["type"] == "discard"
    kept = {i for i in range(8)} - set(top["cards"])
    assert {0, 1, 2, 3} <= kept, "keep the four hearts"
    g.deck = [c for c in g.deck if c.suit != "Hearts"]
    top2, _ = H.rank_hand_actions(g)[0]
    kept2 = {i for i in range(8)} - set(top2.get("cards", []))
    assert not ({0, 1, 2, 3} <= kept2 and top2["type"] == "discard"), "no hearts left: do not chase"


# ───────────────────────────────────────────────────────────── the blind model

def test_blind_model_is_monotone_and_bounded():
    g = _in_blind()
    m = H.blind_model_for(g)
    ps = [m.p_clear(t, 4, 3) for t in (100, 300, 600, 1200, 5000)]
    assert all(0.0 <= p <= 1.0 for p in ps)
    assert ps == sorted(ps, reverse=True)
    assert m.p_clear(600, 4, 3) >= m.p_clear(600, 3, 3) >= m.p_clear(600, 2, 3)
    assert m.p_clear(600, 4, 3) >= m.p_clear(600, 4, 1) >= m.p_clear(600, 4, 0)
    assert m.value(600, 4, 3) >= m.p_clear(600, 4, 3)
    assert m.p_clear(0, 1, 0) == 1.0 and m.p_clear(10 ** 9, 4, 4) == 0.0
    assert m.p_clear(300, 4, 3) > 0.9        # a plain Red deck clears the ante-1 Small


def test_blind_model_is_cached_and_deterministic():
    g = _in_blind()
    a = H.blind_model_for(g)
    b = H.blind_model_for(g)
    assert a is b
    H._MODEL_CACHE.clear()
    c = H.blind_model_for(g)
    assert c is not a and c.Q == a.Q


def test_estimate_clear_probability_decreases_with_target():
    g = _in_blind()
    p1 = H.estimate_clear_probability(g, 300, 4, 3)
    p2 = H.estimate_clear_probability(g, 3000, 4, 3)
    assert 0.0 <= p2 <= p1 <= 1.0


# ───────────────────────────────────────────────────────────────── decisions

def test_made_flush_that_clears_the_blind_is_played():
    g = _in_blind()
    g.chips_scored = g.current_blind.chips_target - 200
    _set_hand(g, [(14, "Hearts"), (13, "Hearts"), (10, "Hearts"), (9, "Hearts"), (4, "Hearts"),
                  (2, "Clubs"), (3, "Clubs"), (7, "Diamonds")])
    top, _ = H.rank_hand_actions(g)[0]
    assert top["type"] == "play" and set(top["cards"]) >= {0, 1, 2, 3, 4}


def test_four_flush_is_chased_when_nothing_is_made():
    g = _in_blind()
    _set_hand(g, [(14, "Hearts"), (7, "Hearts"), (6, "Hearts"), (12, "Hearts"),
                  (11, "Spades"), (13, "Clubs"), (3, "Clubs"), (6, "Diamonds")])
    top, _ = H.rank_hand_actions(g)[0]
    assert top["type"] == "discard"
    assert not (set(top["cards"]) & {0, 1, 2, 3})


def test_unused_hands_are_worth_money_not_risk():
    """With the blind already nearly cleared the player banks: play, do not discard."""
    g = _in_blind()
    g.chips_scored = g.current_blind.chips_target - 20
    _set_hand(g, [(14, "Hearts"), (14, "Spades"), (6, "Hearts"), (12, "Hearts"),
                  (11, "Spades"), (13, "Clubs"), (3, "Clubs"), (6, "Diamonds")])
    top, ev = H.rank_hand_actions(g)[0]
    assert top["type"] == "play"
    assert ev > 1.0                      # cleared + unused hands bonus


def test_hook_leaves_play_scores_intact():
    """Engine 8d3f0d8: The Hook discards from the UNPLAYED hand only, so a play's score
    is the same as without the boss (only the kept cards are perturbed)."""
    g = _in_blind(to_boss=True)
    base = H.HandAnalysis(g, legal=g.legal_actions())
    g.current_blind.boss_key = "bl_hook"
    g.current_blind.is_boss = True
    hook = H.HandAnalysis(g, legal=g.legal_actions())
    base_scores = {t: sc for t, ht, sc, c in base.plays}
    for t, ht, sc, c in hook.plays:
        if t in base_scores:
            assert sc == pytest.approx(base_scores[t])


def test_eye_boss_forbids_a_repeated_hand_type():
    g = _in_blind(to_boss=True)
    g.current_blind.boss_key = "bl_eye"
    g.played_hand_types_this_round = {"Pair", "High Card"}
    an = H.HandAnalysis(g, legal=g.legal_actions())
    for t, ht, s, c in an.plays:
        if ht in ("Pair", "High Card"):
            assert s == 0.0


# ──────────────────────────────────────────────────────────────────────── PvP

def _nemesis(seed="11111111"):
    g = BalatroGame(seed=seed, ruleset="mlb")
    # walk to the ante-2 Boss slot (the Nemesis) with debug wins
    while not (g.ante == 2 and g.current_blind.is_pvp and g.state == State.SELECTING_HAND):
        s = g.state
        if s == State.SELECTING_HAND:
            g.debug_win_blind()
        elif s == State.BLIND_SELECT:
            g.step({"type": "play_blind"})
        elif s == State.ROUND_EVAL:
            g.step({"type": "advance"})
        elif s == State.SHOP:
            g.step({"type": "leave_shop"})
        elif s == State.BOOSTER_OPEN:
            g.step({"type": "skip_booster"})
        else:
            raise AssertionError(s)
    return g


def test_pvp_objective_values_are_probabilities_and_react_to_the_opponent():
    g = _nemesis()
    g.set_pvp_info(10 ** 7, 0)         # hopelessly behind, opponent out of hands
    lo = H.rank_hand_actions(g)
    assert all(-1e-5 <= ev <= 1.0 + 1e-5 for _, ev in lo)
    assert lo[0][1] < 0.05
    g.set_pvp_info(0, 0)               # opponent out of hands with 0: any play wins
    hi = H.rank_hand_actions(g)
    assert hi[0][1] >= 0.99
    g.set_pvp_info(0, 4)               # opponent still to play: uncertain
    mid = H.rank_hand_actions(g)
    assert 0.0 < mid[0][1] < 1.0
    legal = _legal_keys(g)
    assert all(H._action_sort_key(a) in legal for a, _ in mid)


def test_pvp_ranking_is_side_effect_free():
    g = _nemesis()
    g.set_pvp_info(500, 2)
    sig = g.state_signature()
    H.rank_hand_actions(g)
    H.rank_hand_actions(g, budget="full", n_worlds=2)
    assert g.state_signature() == sig


# ───────────────────────────────────────────────────────────────── full budget

def test_full_budget_is_deterministic_given_rng_seed():
    g = _in_blind("29Y3L4S9")
    a = H.rank_hand_actions(g, budget="full", rng=S.world_rng(0, g))
    b = H.rank_hand_actions(g, budget="full", rng=S.world_rng(0, g))
    assert a == b


def test_full_budget_calls_value_fn_on_post_blind_states():
    g = _in_blind()
    seen = []

    def v(world):
        seen.append(world.state)
        return float(world.dollars)
    ranked = H.rank_hand_actions(g, budget="full", value_fn=v, n_worlds=2, top_k=3)
    assert ranked and seen
    assert all(s in (State.SHOP, State.BLIND_SELECT, State.BOOSTER_OPEN, State.GAME_OVER, State.PVP_WAIT,
                     State.SELECTING_HAND) for s in seen)
    assert State.ROUND_EVAL not in seen        # advanced past the cash-out screen


def test_play_out_blind_ends_the_blind():
    g = _in_blind()
    w = S.sample_world(g, random.Random(0))
    H.play_out_blind(w)
    assert w.state != State.SELECTING_HAND or (w.ante, w.blind_idx) != (g.ante, g.blind_idx)


def test_full_budget_value_fn_exception_propagates():
    g = _in_blind()

    def bad(world):
        raise RuntimeError("broken V")
    with pytest.raises(RuntimeError, match="broken V"):
        H.rank_hand_actions(g, budget="full", value_fn=bad, n_worlds=2, top_k=2)


def test_full_budget_defaults_to_k3_x_8_worlds_with_a_value_fn(monkeypatch):
    """EV_NOTES §8.3 / W-LEAF: a value_fn set (and no explicit top_k/n_worlds) resolves to
    HandConfig.full_top_k_v x full_n_worlds_v = 3 x 8 -- flag-driven on value_fn's presence,
    not a hardcoded change to the no-V default (K=5 x 3 worlds, checked below)."""
    g = _in_blind()
    n_sample_world_calls = [0]
    real_sample_world = H.sample_world

    def counting_sample_world(game, rng):
        n_sample_world_calls[0] += 1
        return real_sample_world(game, rng)

    monkeypatch.setattr(H, "sample_world", counting_sample_world)
    n_value_calls = [0]

    def v(world):
        n_value_calls[0] += 1
        return 0.5

    H.rank_hand_actions(g, budget="full", value_fn=v)
    assert n_sample_world_calls[0] == H.DEFAULT_HAND_CONFIG.full_n_worlds_v == 8
    # worlds are shared across the K rolled-out candidates (common random numbers): the
    # value_fn is called once per (candidate, world) pair, so this count pins K too.
    assert n_value_calls[0] == H.DEFAULT_HAND_CONFIG.full_top_k_v * H.DEFAULT_HAND_CONFIG.full_n_worlds_v == 24


def test_full_budget_default_worlds_unchanged_without_a_value_fn(monkeypatch):
    """The no-V path must be byte-for-byte the original K=5 x 3-world default."""
    g = _in_blind()
    n_sample_world_calls = [0]
    real_sample_world = H.sample_world

    def counting_sample_world(game, rng):
        n_sample_world_calls[0] += 1
        return real_sample_world(game, rng)

    monkeypatch.setattr(H, "sample_world", counting_sample_world)
    H.rank_hand_actions(g, budget="full")
    assert n_sample_world_calls[0] == H.DEFAULT_HAND_CONFIG.full_n_worlds == 3


def test_full_budget_explicit_top_k_and_n_worlds_override_the_value_fn_defaults():
    """An explicit top_k / n_worlds from the caller always wins over the K=3x8 V-defaults
    (this is what EVPlayer(n_worlds=..., top_k=...) relies on)."""
    g = _in_blind()
    n_value_calls = [0]

    def v(world):
        n_value_calls[0] += 1
        return 0.5

    H.rank_hand_actions(g, budget="full", value_fn=v, top_k=2, n_worlds=5)
    assert n_value_calls[0] == 2 * 5 == 10


def test_board_ratio_is_cached_by_board_signature():
    g = _in_blind()
    g.debug_add_joker("j_joker")
    H._RATIO_CACHE.clear()
    r1 = H.board_ratio(g)
    assert len(H._RATIO_CACHE) == 1
    r2 = H.board_ratio(g)
    assert r2 == r1 and len(H._RATIO_CACHE) == 1
    # a changed board (another joker) is a different entry
    g.debug_add_joker("j_greedy_joker")
    H.board_ratio(g)
    assert len(H._RATIO_CACHE) == 2


def test_board_ratio_memoises_into_the_cache_it_is_given():
    """W-FIX: ``cache=`` is the scope boundary.  Two callers with their own dicts cannot
    see each other's entries, which is what makes a run's ratios a function of that run —
    ``_board_sig`` deliberately omits planet levels and the deck composition, so a SHARED
    dict served one run a number computed for another (POC_NOTES §3.5: 8% of seeds)."""
    g = _in_blind()
    g.debug_add_joker("j_joker")
    H._RATIO_CACHE.clear()
    mine: dict = {}
    r = H.board_ratio(g, cache=mine)
    assert len(mine) == 1 and not H._RATIO_CACHE, "nothing leaked into the module cache"
    assert H.board_ratio(g, cache=mine) == r
    yours: dict = {}
    assert H.board_ratio(g, cache=yours) == r      # same board -> same number, cold
    assert len(yours) == 1 and len(mine) == 1


def test_a_private_cache_is_not_served_a_number_computed_for_another_board():
    """The mechanism behind the leak, contained: two boards with an IDENTICAL
    ``_board_sig`` and different planet levels get each other's number when they share a
    dict, and their own when they do not."""
    a = _in_blind()
    a.debug_add_joker("j_joker")
    b = _in_blind()
    b.debug_add_joker("j_joker")
    b.planet_levels["Pair"] = b.planet_levels.get("Pair", 1) + 6
    assert H._board_sig(a) == H._board_sig(b), "the key cannot tell these two apart"

    shared: dict = {}
    ra = H.board_ratio(a, cache=shared)
    assert H.board_ratio(b, cache=shared) == ra, "shared: b is served a's number"
    assert H.board_ratio(b, cache={}) != ra, "private: b computes its own"
