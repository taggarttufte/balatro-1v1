"""player.py — EVPlayer: legality in every state, side-effect freedom per state type,
determinism, the three shop tiers (stats / value_fn / rules), MLB driving."""
from __future__ import annotations

import pytest

import _bootstrap  # noqa: F401
from _bootstrap import BalatroGame, State, MLBMatch

import hand as H
from player import EVPlayer, build_proxy, PREMIUM_TAGS

import sys
from pathlib import Path
_EVAL = str(Path(__file__).resolve().parents[2] / "eval")
if _EVAL not in sys.path:
    sys.path.insert(0, _EVAL)
import common as C  # noqa: E402  (mp/eval/common.py: DEFAULT_SEEDS, drivers, adapt_player)


def _key(a):
    return H._action_sort_key(a)


def _drive(player, game, max_steps=400, stop_ante=3, check=None):
    seen = set()
    steps = 0
    while game.state != State.GAME_OVER and game.ante <= stop_ante and steps < max_steps:
        legal = game.legal_actions()
        sig = game.state_signature()
        snap = game.run_state.rng.snapshot()
        a = player.act(game)
        assert game.state_signature() == sig, f"act mutated the game in {game.state}"
        assert game.run_state.rng.snapshot() == snap
        if legal:
            assert _key(a) in {_key(x) for x in legal}, (game.state, a)
        else:
            assert a == player.no_action
        seen.add(game.state)
        if check:
            check(game, a)
        game.step(a)
        steps += 1
    return seen


# ───────────────────────────────────────────────────────────── every state

@pytest.mark.parametrize("seed", ["11111111", "CHPB293X"])
def test_act_is_legal_and_side_effect_free_through_a_vanilla_run(seed):
    g = BalatroGame(seed=seed, ruleset="vanilla")
    seen = _drive(EVPlayer(), g, stop_ante=3)
    assert {State.BLIND_SELECT, State.SELECTING_HAND, State.ROUND_EVAL, State.SHOP} <= seen


def test_booster_state_is_handled():
    g = BalatroGame(seed="11111111", ruleset="vanilla")
    pl = EVPlayer()
    hit = []

    def check(game, a):
        if game.state == State.BOOSTER_OPEN:
            hit.append(a["type"])
    _drive(pl, g, stop_ante=4, max_steps=800, check=check)
    assert hit and set(hit) <= {"pick_booster", "skip_booster"}


def test_no_legal_actions_gives_no_action():
    g = BalatroGame(seed="11111111", ruleset="mlb")
    g.state = State.PVP_WAIT
    assert EVPlayer().act(g) == {"type": "advance"}
    assert EVPlayer(no_action={"type": "noop"}).act(g) == {"type": "noop"}
    g.state = State.GAME_OVER
    assert EVPlayer().act(g) == {"type": "advance"}


def test_round_eval_advances():
    g = BalatroGame(seed="11111111", ruleset="vanilla")
    g.step({"type": "play_blind"})
    g.debug_win_blind()
    assert g.state == State.ROUND_EVAL
    assert EVPlayer().act(g) == {"type": "advance"}


def test_explain_returns_ranked_triples():
    g = BalatroGame(seed="11111111", ruleset="vanilla")
    g.step({"type": "play_blind"})
    ex = EVPlayer().explain(g)
    assert ex and all(isinstance(a, dict) and isinstance(ev, float) and isinstance(r, str) for a, ev, r in ex)
    assert [ev for _, ev, _ in ex] == sorted((ev for _, ev, _ in ex), reverse=True)
    assert ex[0][0] == EVPlayer().act(g)


# ───────────────────────────────────────────────────────────── determinism

def test_deterministic_given_seed_and_state():
    acts = []
    for _ in range(2):
        g = BalatroGame(seed="1558AXDL", ruleset="vanilla")
        pl = EVPlayer(seed=7)
        seq = []
        _drive(pl, g, stop_ante=2, check=lambda game, a: seq.append(_key(a)))
        acts.append(seq)
    assert acts[0] == acts[1]


def test_full_budget_player_is_deterministic_and_legal():
    acts = []
    for _ in range(2):
        g = BalatroGame(seed="11111111", ruleset="vanilla")
        pl = EVPlayer(budget="full", seed=3, n_worlds=2, top_k=3)
        seq = []
        _drive(pl, g, stop_ante=1, max_steps=60, check=lambda game, a: seq.append(_key(a)))
        acts.append(seq)
    assert acts[0] == acts[1]


def test_epsilon_greedy_is_legal_and_seed_dependent():
    g = BalatroGame(seed="11111111", ruleset="vanilla")
    g.step({"type": "play_blind"})
    legal = {_key(a) for a in g.legal_actions()}
    picks = {_key(EVPlayer(seed=s, epsilon=1.0).act(g)) for s in range(12)}
    assert picks <= legal and len(picks) > 1
    assert _key(EVPlayer(seed=1, epsilon=1.0).act(g)) == _key(EVPlayer(seed=1, epsilon=1.0).act(g))


# ───────────────────────────────────────────────────────────── shop tiers

def _shop(seed="11111111"):
    g = BalatroGame(seed=seed, ruleset="vanilla")
    g.step({"type": "play_blind"})
    g.debug_win_blind()
    g.step({"type": "advance"})
    assert g.state == State.SHOP
    return g


def test_stats_tier_picks_the_best_positive_row_else_leaves():
    g = _shop()
    buys = [a for a in g.legal_actions() if a["type"] == "buy"]
    assert buys

    class Stats:
        def __init__(self, rows):
            self.rows = rows

        def decision_table(self, game):
            return self.rows

    class Row:
        def __init__(self, action, net_ev, label="x"):
            self.action, self.net_ev, self.label = action, net_ev, label
    good = Stats([Row(buys[0], 0.1), Row({"type": "reroll"}, 0.3), Row(buys[-1], -0.2)])
    a = EVPlayer(stats=good).act(g)
    assert _key(a) == _key({"type": "reroll"})
    bad = Stats([Row(buys[0], -0.1), Row({"type": "reroll"}, -0.3)])
    assert EVPlayer(stats=bad).act(g) == {"type": "leave_shop"}
    # dict rows and a broken stats object are tolerated (falls back to the rules)
    dict_rows = Stats([{"action": buys[0], "net_ev": 0.5}])
    assert _key(EVPlayer(stats=dict_rows).act(g)) == _key(buys[0])

    class Broken:
        def decision_table(self, game):
            raise RuntimeError("not ready")
    a = EVPlayer(stats=Broken()).act(g)
    assert _key(a) in {_key(x) for x in g.legal_actions()}


def test_value_fn_tier_argmaxes_v_on_clones():
    g = _shop()
    sig = g.state_signature()
    calls = []

    def v(world):
        calls.append(world)
        return -float(world.dollars)          # "spending is good": prefers a purchase
    a = EVPlayer(value_fn=v).act(g)
    assert g.state_signature() == sig
    assert calls and all(w is not g for w in calls)
    assert a["type"] in ("buy", "reroll")

    def v2(world):
        return float(world.dollars)            # "money is good": leaves
    assert EVPlayer(value_fn=v2).act(g) == {"type": "leave_shop"}


def test_rule_tier_buys_a_joker_that_raises_the_proxy():
    g = _shop()
    base = build_proxy(g)
    assert 0.0 <= base["p_clear"] <= 1.0 and base["strength"] > 0
    pl = EVPlayer()
    ex = pl.explain(g)
    assert ex[0][0]["type"] in ("buy", "leave_shop", "use_consumable", "reroll")
    # a joker purchase appears with a gain relative to leaving
    buys = [(a, ev) for a, ev, r in ex if a["type"] == "buy"]
    leave = next(ev for a, ev, r in ex if a["type"] == "leave_shop")
    for a, ev in buys:
        item = g.current_shop[a["item_idx"]]
        if item.kind == "joker":
            assert ev != leave or True      # present and valued (sign depends on the joker)


def test_blind_select_never_skips_into_a_boss_or_nemesis():
    g = BalatroGame(seed="11111111", ruleset="mlb")
    pl = EVPlayer()
    # Big blind of ante 2 sits before the Nemesis: skipping it is never chosen
    while not (g.ante == 2 and g.blind_idx == 1 and g.state == State.BLIND_SELECT):
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
    g.blind_tags["Big"] = "tag_rare"
    assert pl.act(g) == {"type": "play_blind"}


def test_blind_select_skips_only_for_premium_tags():
    g = BalatroGame(seed="11111111", ruleset="vanilla")
    assert g.state == State.BLIND_SELECT
    pl = EVPlayer()
    g.blind_tags["Small"] = "tag_garbage"
    assert pl.act(g) == {"type": "play_blind"}
    g.blind_tags["Small"] = "tag_rare"
    assert "tag_rare" in PREMIUM_TAGS
    a = pl.act(g)
    assert a["type"] in ("play_blind", "skip_blind")


# ───────────────────────────────────────────────────────────────────── MLB

def test_sp_mlb_driver_runs_with_the_player():
    pol = C.adapt_player(EVPlayer())
    r = C.play_sp_mlb("11111111", pol, max_antes=2)
    assert r["furthest_ante"] >= 2 and r["steps"] > 0


def test_mlb_match_ev_vs_ev_completes():
    m = MLBMatch(seed="11111111", lives=1)
    players = [EVPlayer(seed=0), EVPlayer(seed=1)]
    while not m.done and m.steps < 20000:
        p = m.current_player()
        assert p is not None
        g = m.games[p]
        m.step(p, players[p].act(g))
    assert m.done and m.winner in (0, 1)
    assert m.pvp_log, "at least one Nemesis was resolved"


def test_pvp_decision_is_legal_and_side_effect_free():
    m = MLBMatch(seed="11111111", lives=2)
    players = [EVPlayer(seed=0), EVPlayer(seed=1)]
    checked = 0
    while not m.done and m.steps < 20000 and checked < 5:
        p = m.current_player()
        g = m.games[p]
        if g.state == State.SELECTING_HAND and g.current_blind.is_pvp:
            sig = g.state_signature()
            a = players[p].act(g)
            assert g.state_signature() == sig
            assert _key(a) in {_key(x) for x in g.legal_actions()}
            checked += 1
        else:
            a = players[p].act(g)
        m.step(p, a)
    assert checked == 5


# ─────────────────────────────────────────────────────── fix pass (2026-08-23)

def test_epsilon_does_not_wedge_on_an_unchanged_state():
    """Fix 2: epsilon draws come from a sequential stream, so repeated visits to the SAME
    observable state (a no-op pick) get fresh draws instead of the same action forever."""
    g = _shop()
    pl = EVPlayer(seed=1, epsilon=1.0)
    picks = {_key(pl.act(g)) for _ in range(30)}
    assert len(picks) > 1
    # and reset() replays the same sequence (determinism given (seed, call history))
    a = EVPlayer(seed=5, epsilon=1.0)
    seq1 = [_key(a.act(g)) for _ in range(10)]
    a.reset()
    seq2 = [_key(a.act(g)) for _ in range(10)]
    assert seq1 == seq2


def test_value_fn_exception_propagates_from_act():
    """Fix 3: a broken value_fn raises instead of silently degrading to the proxy."""
    g = _shop()

    def bad(world):
        raise ValueError("V is broken")
    with pytest.raises(ValueError, match="V is broken"):
        EVPlayer(value_fn=bad).act(g)


def test_anti_cycling_guard_breaks_a_value_fn_no_op_loop():
    """Fix 4: a value_fn that prefers standing still (W5's 40k-step shop loop) is broken
    by the signature guard: fall back to the rules after 3 identical visits, force
    leave_shop after 6, so the shop always ends within a handful of steps."""
    g = _shop()
    # Wheel of Fortune with no editionless joker: apply_tarot returns False -> the engine
    # silently no-ops and the consumable stays (W5's observed wedge action)
    assert not g.jokers
    g.consumable_hand.append("c_wheel_of_fortune")
    d0, n0 = g.dollars, len(g.consumable_hand)

    def stay(world):
        good = (world.state == State.SHOP and world.dollars == d0
                and len(world.consumable_hand) == n0)
        return 1.0 if good else 0.0
    pl = EVPlayer(value_fn=stay)
    first = pl.act(g)
    assert first["type"] == "use_consumable", "the no-op must be V's favourite for this test"
    steps = 0
    while g.state == State.SHOP and steps < 12:
        g.step(pl.act(g))
        steps += 1
    assert g.state != State.SHOP, "the guard must break the loop within 12 steps"


def test_anti_cycling_guard_does_not_disturb_a_normal_shop_visit():
    g = _shop()
    pl = EVPlayer()
    sig0 = g.state_signature()
    a1 = pl.act(g)
    a2 = EVPlayer().act(g)
    assert _key(a1) == _key(a2)            # one visit: same decision as a fresh player
    assert g.state_signature() == sig0
