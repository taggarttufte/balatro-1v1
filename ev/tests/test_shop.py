"""W-SHOP: the EV shop economy — the queue arithmetic, the interest-threshold cost, the
pack order statistics, the reroll stopping rule, the cycle-sell toll, the race read and The
Fool sequencing.  See ``ev/SHOP_NOTES.md``.

The load-bearing invariant of the whole workstream is the LAST test in this file: with
``shop_arm_cfgs("old")`` the player is the pre-2026-08-26 player bit-for-bit, because that
is what the h2h's "old" arm is."""
from __future__ import annotations

from dataclasses import replace

import pytest

import _bootstrap  # noqa: F401
from _bootstrap import BalatroGame, State

import hand as H
import player as P
from player import (EVPlayer, DEFAULT_PLAYER_CONFIG, OLD_SHOP_CONFIG, shop_arm_cfgs,
                    shops_left_on_queue, interest_cost, tarot_dollars, _levels_from_atoms,
                    _e_top_k, _standard_card_atoms)


def _new_player(**kw):
    cfg, hcfg = shop_arm_cfgs("new")
    return EVPlayer(budget="fast", seed=0, epsilon=0.0, cfg=replace(cfg, **kw) if kw else cfg,
                    hand_cfg=hcfg)


def _drive_to(game, player, state, max_steps=400, ante_cap=6):
    """Step ``game`` with ``player`` until it is in ``state`` (returns True) or stops."""
    steps = 0
    while game.state != State.GAME_OVER and game.ante <= ante_cap and steps < max_steps:
        if game.state == state:
            return True
        legal = game.legal_actions()
        game.step(player.act(game) if legal else {"type": "advance"})
        steps += 1
    return game.state == state


def _a_shop(seed="11111111", player=None):
    pl = player or _new_player()
    g = BalatroGame(seed=seed, deck_key="b_red", stake=1, ruleset="vanilla")
    assert _drive_to(g, pl, State.SHOP), "no SHOP reached"
    return g, pl


# ══════════════════════════════════════════════════════ the ante shop queue (§2c)

def test_shops_left_on_queue_reads_the_queue_position_off_blind_idx():
    """GENERATION_SPEC §8.3: the ante-N queue is the '...sho<a>' streams, and three shops
    draw from it — the post-Boss shop of ante N-1 (blind_idx 2, ante already bumped), then
    the ones after Small and Big.  The last of the three is the shop before the Boss /
    Nemesis, where depth not bought now is lost."""
    g, _ = _a_shop()
    for idx, expect in ((2, 3), (0, 2), (1, 1)):
        g.blind_idx = idx
        assert shops_left_on_queue(g) == expect
    g.state = State.SELECTING_HAND
    assert shops_left_on_queue(g) == 0


def test_reroll_costs_escalate_and_free_rerolls_come_first():
    g, pl = _a_shop()
    g.reroll_cost, g.reroll_discount, g.free_rerolls_remaining = 5, 0, 0
    g.blind_idx = 1                      # last shop on the queue: no spread term
    costs = pl._reroll_costs(g, 4)
    assert [c[0] for c in costs] == [5.0, 6.0, 7.0, 8.0]
    assert [c[2] for c in costs] == [5, 11, 18, 26]           # cumulative
    g.free_rerolls_remaining = 2
    costs = pl._reroll_costs(g, 3)
    assert [c[0] for c in costs] == [0.0, 0.0, 5.0]


def test_the_spread_term_is_zero_at_the_last_shop_of_the_queue_and_positive_before_it():
    """§2c: while more shops share this ante's queue the $1-per-roll escalation is
    avoidable (the price resets next shop), so it is charged as an opportunity cost; at the
    last shop of the queue it is not."""
    g, pl = _a_shop()
    g.reroll_cost, g.reroll_discount, g.free_rerolls_remaining = 5, 0, 0
    g.dollars = 200                       # above the cap: no interest term either way
    g.blind_idx = 1
    last = pl._reroll_costs(g, 4)
    g.blind_idx = 0
    more = pl._reroll_costs(g, 4)
    assert last[0][1] == pytest.approx(more[0][1])            # the first roll is at base price
    assert more[3][1] > last[3][1]                            # deeper rolls cost more
    assert more[3][1] - last[3][1] == pytest.approx(
        DEFAULT_PLAYER_CONFIG.reroll_defer_delta * 3)


# ══════════════════════════════════════════════════════ the interest threshold (§2b)

def test_interest_cost_is_a_threshold_not_a_slope():
    g, _ = _a_shop()
    cfg = DEFAULT_PLAYER_CONFIG
    per_tier = cfg.interest_rounds * cfg.interest_weight
    g.dollars = 60                                   # far above the $25 cap: nothing forfeited
    assert interest_cost(g, cfg, 5) == pytest.approx(0.0)
    g.dollars = 24                                   # 24 -> 19 crosses the $20 breakpoint
    assert interest_cost(g, cfg, 5) == pytest.approx(per_tier)
    assert interest_cost(g, cfg, 4) == pytest.approx(0.0)     # 24 -> 20: still $4 of interest
    assert interest_cost(g, cfg, 1) == pytest.approx(0.0)     # 24 -> 23: same tier
    g.dollars = 22                                   # 22 -> 12 crosses $20 and $15
    assert interest_cost(g, cfg, 10) == pytest.approx(2 * per_tier)
    g.no_interest = True
    assert interest_cost(g, cfg, 10) == pytest.approx(0.0)


def test_interest_cost_matches_build_proxys_own_money_term():
    """The whole point of not reusing ``economy.interest_loss`` (SHOP_NOTES §2.2): the shop
    tier's money must be the money ``build_proxy`` prices, or a pack row and a joker row are
    compared on different dollars."""
    g, pl = _a_shop()
    cfg = DEFAULT_PLAYER_CONFIG
    g.dollars = 24
    before = P.build_proxy(g, cfg, pl.proxy_cfg, ratio_cache=pl._ratio_cache)["value"]
    spend = 5                                       # 24 -> 19: one interest tier crossed
    g.dollars -= spend
    after = P.build_proxy(g, cfg, pl.proxy_cfg, ratio_cache=pl._ratio_cache)["value"]
    drop = (before - after) / cfg.lam_money
    g.dollars += spend
    assert drop == pytest.approx(spend + interest_cost(g, cfg, spend), abs=1e-6)


# ══════════════════════════════════════════════════════ order statistics (§3)

def test_levels_and_top_k_on_a_degenerate_distribution():
    """A single positive atom: the best of n is that value, and the best k of n is k of it
    (a pack pick is free, so the sum of the top k is what a pack yields)."""
    levels, tails = _levels_from_atoms([(1.0, 6.0)])
    assert tails[-1] == pytest.approx(1.0)
    assert _e_top_k(levels, tails, 3, 1) == pytest.approx(6.0)
    assert _e_top_k(levels, tails, 3, 2) == pytest.approx(12.0)
    assert _e_top_k(levels, tails, 1, 2) == pytest.approx(6.0)     # k is capped at n
    assert _e_top_k(levels, tails, 3, 0) == 0.0


def test_top_k_matches_a_brute_force_expectation():
    import itertools
    atoms = [(0.5, 0.0), (0.3, 4.0), (0.2, 10.0)]
    levels, tails = _levels_from_atoms(atoms, max_levels=64)
    for n, k in ((2, 1), (3, 1), (3, 2), (4, 2)):
        total = 0.0
        for combo in itertools.product(range(len(atoms)), repeat=n):
            w = 1.0
            for i in combo:
                w *= atoms[i][0]
            vals = sorted((max(0.0, atoms[i][1]) for i in combo), reverse=True)
            total += w * sum(vals[:k])
        assert _e_top_k(levels, tails, n, k) == pytest.approx(total, rel=1e-6)


def test_negative_atoms_never_contribute_a_pack_pick_can_be_declined():
    levels, tails = _levels_from_atoms([(0.5, -5.0), (0.5, 4.0)])
    assert _e_top_k(levels, tails, 1, 1) == pytest.approx(0.5 * 4.0)


def test_standard_card_atoms_follow_the_generators_own_recipe():
    """generate.open_pack:1283-1287 — 40% Enhanced, then a seal and an edition roll."""
    atoms = _standard_card_atoms(DEFAULT_PLAYER_CONFIG, {"add_plain": 1.0, "steel1": 4.5})
    assert sum(w for w, _ in atoms) == pytest.approx(1.0)
    plain = atoms[0]
    assert plain[0] == pytest.approx(1.0 - P._STD_P_ENHANCED)
    assert all(v >= plain[1] for _, v in atoms[1:])       # an enhancement is never negative


# ══════════════════════════════════════════════════════ measured deck effects (§3.2)

def test_deck_effects_are_measured_cached_and_lazy():
    g, pl = _a_shop()
    eff = pl._deck_effects(g, ("suit3",))
    assert set(eff) == {"suit3"}                     # only what was asked for
    assert eff["suit3"] >= P._TAROT_SUIT_FLOOR
    eff2 = pl._deck_effects(g, ("suit3", "steel1"))
    assert eff2["suit3"] == eff["suit3"]             # the first measurement is reused
    assert "steel1" in eff2
    sig = g.state_signature()
    pl._deck_effects(g, EVPlayer._EFFECT_KEYS)
    assert g.state_signature() == sig                # measurement is on clones only
    pl.reset()
    assert pl._fix_cache == {}


def test_tarot_dollars_uses_the_measurements_and_the_state():
    g, pl = _a_shop()
    eff = {"suit3": 11.0, "steel1": 4.0, "destroy2": 7.0, "rank2": 2.0}
    cfg = DEFAULT_PLAYER_CONFIG
    assert tarot_dollars(g, "c_star", cfg, eff) == pytest.approx(11.0)
    assert tarot_dollars(g, "c_hanged_man", cfg, eff) == pytest.approx(7.0)
    # Chariot converts 2 cards to Steel; Justice converts 2 to Glass, which the card table
    # prices at 3.0 against Steel's 4.5 -- so the same measurement, scaled by the table.
    assert tarot_dollars(g, "c_chariot", cfg, eff) == pytest.approx(8.0)
    assert tarot_dollars(g, "c_justice", cfg, eff) == pytest.approx(8.0 * 3.0 / 4.5)
    g.dollars = 12
    assert tarot_dollars(g, "c_hermit", cfg, eff) == pytest.approx(12.0)
    g.dollars = 40
    assert tarot_dollars(g, "c_hermit", cfg, eff) == pytest.approx(20.0)   # the game's cap
    g.run_state.last_tarot_planet = None
    assert tarot_dollars(g, "c_fool", cfg, eff) == 0.0                     # unusable
    g.run_state.last_tarot_planet = "c_star"
    assert tarot_dollars(g, "c_fool", cfg, eff) == pytest.approx(11.0)     # it copies


# ══════════════════════════════════════════════════════ the cycle-sell toll (§3.3)

def test_cycle_cost_is_zero_with_a_free_slot_and_a_real_toll_without_one():
    g, pl = _a_shop()
    assert pl._cycle_cost(g) == 0.0 or len(g.jokers) >= g.joker_slots
    from balatro_sim.jokers.base import JokerInstance
    from balatro_sim.shop import emplace_joker
    while len(g.jokers) < g.joker_slots:
        emplace_joker(g, JokerInstance("j_joker", "None"))
    toll = pl._cycle_cost(g)
    assert toll >= 0.0
    # an all-Eternal board cannot be cycled at any price
    for j in g.jokers:
        j.state["eternal"] = True
    assert pl._cycle_cost(g) > 1e6


# ══════════════════════════════════════════════════════ the reroll stopping rule (§2)

def test_the_reroll_plan_is_uncapped_in_depth_and_bounded_by_the_balance():
    g, pl = _a_shop()
    g.dollars = 200
    base = P.build_proxy(g, pl.cfg, pl.proxy_cfg, ratio_cache=pl._ratio_cache)["value"]
    rich = pl._reroll_plan(g, base, -1.0)
    assert rich["depth"] == pl.cfg.reroll_max_depth      # nothing but the loop bound binds
    g.dollars = 7
    pl.reset()
    poor = pl._reroll_plan(g, base, -1.0)
    assert poor["depth"] == 1                            # one $5 roll is all that fits
    g.dollars = 2
    pl.reset()
    broke = pl._reroll_plan(g, base, -1.0)
    assert broke["depth"] == 0 and broke["roll"] is False


def test_the_plan_is_monotone_in_the_hurdle():
    g, pl = _a_shop()
    g.dollars = 60
    base = P.build_proxy(g, pl.cfg, pl.proxy_cfg, ratio_cache=pl._ratio_cache)["value"]
    cheap = pl._reroll_plan(g, base, -1.0)["plan"]
    pl2 = _new_player(reroll_hurdle=8.0)
    dear = pl2._reroll_plan(g, base, -1.0)["plan"]
    assert dear < cheap


def test_fresh_shelf_values_are_side_effect_free_and_seat_symmetric():
    """The sampled worlds are seeded from the OBSERVABLE STATE only, never from
    ``self.seed``: with epsilon 0 the fast player must be a function of the state alone, or
    the two seats of one h2h stop being mirror images (test_h2h.py pins that end to end)."""
    g, pl = _a_shop()
    base = P.build_proxy(g, pl.cfg, pl.proxy_cfg, ratio_cache=pl._ratio_cache)["value"]
    sig, snap = g.state_signature(), g.run_state.rng.snapshot()
    a = pl._fresh_shelf_values(g, base)
    assert g.state_signature() == sig and g.run_state.rng.snapshot() == snap
    other = EVPlayer(budget="fast", seed=999, epsilon=0.0)
    assert other._fresh_shelf_values(g, base) == a
    assert all(x >= 0.0 for x in a)


def test_the_old_arm_keeps_the_one_reroll_cap_and_the_new_arm_does_not():
    """The cap is a hard rule in the old player (``max_rerolls_per_visit``) and is what the
    80-seed baseline's "never more than one reroll in a visit" was."""
    old_cfg, _ = shop_arm_cfgs("old")
    assert old_cfg.max_rerolls_per_visit == 1 and old_cfg.reroll_ev is False
    assert DEFAULT_PLAYER_CONFIG.reroll_ev is True
    g, pl = _a_shop()
    g.dollars = 120
    g.reroll_cost = 12                        # the 8th roll of a visit
    base = P.build_proxy(g, pl.cfg, pl.proxy_cfg, ratio_cache=pl._ratio_cache)["value"]
    plan = pl._reroll_plan(g, base, -1.0)
    assert plan["depth"] >= 1                 # the rule still has an opinion this deep in


# ══════════════════════════════════════════════════════ pack EV (§3)

@pytest.mark.parametrize("kind", ["arcana", "celestial", "buffoon", "standard", "spectral"])
def test_pack_ev_is_finite_and_side_effect_free_for_every_family(kind):
    from balatro_sim.shop import booster_item
    from balatro_sim import game_keys as gk
    g, pl = _a_shop()
    key = next(k for k in gk.BOOSTER_TYPES if kind in k)
    item = booster_item(gk.BOOSTER_TYPES[key]["centers"][0])
    sig = g.state_signature()
    net, why = pl._pack_ev(g, item)
    assert g.state_signature() == sig
    assert isinstance(net, float) and net == net and abs(net) < 1e4
    assert item.name in why


def test_a_pack_that_clears_the_money_guard_outranks_leaving_except_standard():
    """Tagg's prior and the OLD rule's own behaviour: the money guard decides WHETHER, the
    EV decides WHICH.  Standard packs are the one family the measurement may talk down."""
    g, pl = _a_shop()
    g.dollars = 200
    rows = pl._rank_shop_rules(g, g.legal_actions())
    leave = next(ev for a, ev, _ in rows if a["type"] == "leave_shop")
    for a, ev, why in rows:
        if a["type"] != "buy":
            continue
        item = g.current_shop[a["item_idx"]]
        if item.kind == "booster" and "standard" not in item.key:
            assert ev >= min(pl.cfg.pack_take_floor, leave + 1e-9)


# ══════════════════════════════════════════════════════ the race read (§2d)

def test_race_aggression_is_exactly_neutral_in_vanilla():
    g, pl = _a_shop()
    assert getattr(g, "mlb", False) is False
    assert pl._race_aggression(g) == 1.0


def test_race_aggression_rises_when_lives_are_gone_under_mlb():
    from _bootstrap import MLBMatch
    m = MLBMatch(seed="11111111", deck_key="b_red", stake=1, lives=4)
    g = m.games[0]
    pl = _new_player()
    g.state = State.SHOP
    full = pl._race_aggression(g)
    g.lives = 1
    hurt = pl._race_aggression(g)
    assert 1.0 <= full < hurt <= 1.0 + pl.cfg.race_aggression


def test_bind_race_never_raises_and_leaves_vanilla_alone():
    g, pl = _a_shop()

    class _FakeMatch:
        games = [g, g]
        pvp_log = ()
    pl.bind_race(_FakeMatch(), 0)
    assert pl._race_p_win is None                 # vanilla: no race
    pl.bind_race(object(), 0)                     # garbage in, no exception out
    assert pl._race_p_win is None


# ══════════════════════════════════════════════════════ The Fool sequencing (§4)

def _fool_case(evs, keys):
    class _G:
        consumable_hand = list(keys)
    cons = [({"type": "use_consumable", "consumable_idx": i, "target_cards": []}, ev)
            for i, ev in enumerate(evs)]
    return _G(), cons


def test_fool_ordering_permutes_the_batch_so_the_best_use_is_last():
    cfg = H.DEFAULT_HAND_CONFIG
    g, cons = _fool_case([0.9, 0.5, 0.1], ["c_sun", "c_pluto", "c_fool"])
    out = H._fool_ordered(g, cons, cfg)
    ev = {a["consumable_idx"]: e for a, e in out}
    assert ev[1] > ev[0]                       # the weaker planet is used FIRST
    assert ev[2] < min(ev[0], ev[1])           # ... and The Fool waits for the good copy
    # the group's EV multiset over the non-Fool uses is preserved: the rule only reorders
    assert sorted((ev[0], ev[1])) == pytest.approx(sorted([0.9, 0.5]))


def test_fool_ordering_is_inert_without_the_fool_and_when_flagged_off():
    cfg = H.DEFAULT_HAND_CONFIG
    g, cons = _fool_case([0.9, 0.5], ["c_sun", "c_pluto"])
    assert H._fool_ordered(g, cons, cfg) == cons
    g2, cons2 = _fool_case([0.9, 0.5, 0.1], ["c_sun", "c_pluto", "c_fool"])
    assert H._fool_ordered(g2, cons2, replace(cfg, fool_order=False)) == cons2


def test_fool_ordering_does_not_move_the_group_against_plays_or_discards():
    """The rule permutes EVs inside the consumable group, so whether a consumable is used at
    all this decision is bit-identical to the unordered ranking."""
    cfg = H.DEFAULT_HAND_CONFIG
    g, cons = _fool_case([0.9, 0.5, 0.1], ["c_sun", "c_pluto", "c_fool"])
    out = H._fool_ordered(g, cons, cfg)
    assert max(e for _, e in out) == pytest.approx(max(e for _, e in cons))


# ══════════════════════════════════════════════════════ the arms

def test_shop_arm_cfgs_turns_every_w_shop_flag_off_together():
    cfg, hcfg = shop_arm_cfgs("old")
    assert (cfg.reroll_ev, cfg.pack_ev, cfg.fool_order) == (False, False, False)
    assert hcfg.fool_order is False
    assert cfg == OLD_SHOP_CONFIG
    ncfg, nhcfg = shop_arm_cfgs("new")
    assert (ncfg.reroll_ev, ncfg.pack_ev, ncfg.fool_order) == (True, True, True)
    assert nhcfg.fool_order is True
    with pytest.raises(ValueError):
        shop_arm_cfgs("sideways")


def test_the_old_arm_is_the_pre_w_shop_player_bit_for_bit():
    """The h2h's control arm.  Every W-SHOP flag off must reproduce the old rules tier's
    own decisions, so the comparison measures this workstream and nothing else."""
    old_cfg, old_hcfg = shop_arm_cfgs("old")
    pl = EVPlayer(budget="fast", seed=0, epsilon=0.0, cfg=old_cfg, hand_cfg=old_hcfg)
    g = BalatroGame(seed="1558AXDL", deck_key="b_red", stake=1, ruleset="vanilla")
    seen_shop = 0
    steps = 0
    while g.state != State.GAME_OVER and g.ante <= 3 and steps < 400:
        legal = g.legal_actions()
        if g.state == State.SHOP:
            seen_shop += 1
            rows = pl._rank_shop_rules(g, legal)
            for a, ev, why in rows:
                if a["type"] == "reroll":
                    assert "done" in why      # the OLD reason string, not the plan's
                if a["type"] == "buy" and g.current_shop[a["item_idx"]].kind == "booster":
                    assert "below floor" in why or why.endswith("(ok)")
        g.step(pl.act(g) if legal else {"type": "advance"})
        steps += 1
    assert seen_shop >= 2


def test_a_new_arm_run_is_a_function_of_the_seed_alone():
    """W-FIX's property, re-pinned for the two caches this workstream adds: replaying a seed
    on a player that has already played a DIFFERENT seed must give the same trajectory."""
    def run(seed, warm=None):
        pl = _new_player()
        if warm is not None:
            g0 = BalatroGame(seed=warm, deck_key="b_red", stake=1, ruleset="vanilla")
            s = 0
            while g0.state != State.GAME_OVER and g0.ante <= 3 and s < 200:
                la = g0.legal_actions()
                g0.step(pl.act(g0) if la else {"type": "advance"})
                s += 1
            pl.reset()
        g = BalatroGame(seed=seed, deck_key="b_red", stake=1, ruleset="vanilla")
        out = []
        s = 0
        while g.state != State.GAME_OVER and g.ante <= 3 and s < 300:
            la = g.legal_actions()
            a = pl.act(g) if la else {"type": "advance"}
            out.append(H._action_sort_key(a))
            g.step(a)
            s += 1
        return out
    assert run("11111111") == run("11111111", warm="1558AXDL")


def test_act_is_side_effect_free_in_the_shop_for_both_arms():
    for arm in ("old", "new"):
        cfg, hcfg = shop_arm_cfgs(arm)
        pl = EVPlayer(budget="fast", seed=0, epsilon=0.0, cfg=cfg, hand_cfg=hcfg)
        g = BalatroGame(seed="15H9Z3IY", deck_key="b_red", stake=1, ruleset="vanilla")
        steps = 0
        checked = 0
        while g.state != State.GAME_OVER and g.ante <= 3 and steps < 400:
            legal = g.legal_actions()
            if g.state in (State.SHOP, State.BOOSTER_OPEN):
                sig, snap = g.state_signature(), g.run_state.rng.snapshot()
                a = pl.act(g)
                assert g.state_signature() == sig, (arm, g.state)
                assert g.run_state.rng.snapshot() == snap, (arm, g.state)
                assert H._action_sort_key(a) in {H._action_sort_key(x) for x in legal}
                checked += 1
            else:
                a = pl.act(g) if legal else {"type": "advance"}
            g.step(a)
            steps += 1
        assert checked >= 3, arm


def test_stats_pack_slices_is_additive_and_agrees_with_pack_p_hit():
    """``hit.pack_slices`` was split out of ``hit.pack_p_hit`` for this workstream; the
    split must not have changed the older function's numbers."""
    import sys
    from pathlib import Path
    sp = str(Path(__file__).resolve().parents[2] / "stats")
    if sp not in sys.path:
        sys.path.insert(0, sp)
    import hit as hitmod
    g, _ = _a_shop()
    for kind, size in (("Arcana", 3), ("Celestial", 5), ("Buffoon", 2), ("Spectral", 2)):
        slices = hitmod.pack_slices(g, kind)
        assert slices, kind
        p_card, mean_v = hitmod.p_hit_and_value(slices)
        p_pack, mean_hit, _d = hitmod.pack_p_hit(g, kind, size)
        assert p_pack == pytest.approx(1.0 - (1.0 - p_card) ** size)
        assert mean_hit == pytest.approx(mean_v)
    assert hitmod.pack_slices(g, "Standard") == []
    assert hitmod.pack_p_hit(g, "Standard", 3) == (0.0, 0.0, {})
