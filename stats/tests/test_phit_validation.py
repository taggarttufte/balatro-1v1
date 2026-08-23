"""Empirical validation of P(hit): reroll many DETERMINIZED clones of a real shop state and
compare the observed hit frequency to ``hit.reroll_p_hit``'s analytic number, within a
binomial confidence interval. This is "the test that proves known chance" the brief asks
for (gate 3).

Uses ``game.clone_determinized(seed)`` (W2 landed it -- confirmed via ``hasattr``, per the
brief's explicit instruction) so every trial's ``run_state.rng`` is a genuinely fresh,
independent stream; the generation-layer bookkeeping (``used_jokers``, owned lists,
``shop_joker_max``, rates, banned keys) is copied unchanged, exactly what the analytic
computation itself reads.

N is kept small here (CI speed: the full suite must stay under 60s) -- STATS_NOTES.md
reports a larger, ad hoc run (500-2000 trials/state, per the lead's resource note) done via
a one-off script, not this file."""
from __future__ import annotations

import math

import _bootstrap  # noqa: F401
from _bootstrap import BalatroGame, State
from balatro_sim.shop import reroll_shop

import hit as hitmod

N_TRIALS = 150            # keep this file fast; STATS_NOTES.md has the real (500+) numbers
Z_95 = 1.96


def _fresh_shop_game(seed="11111111"):
    g = BalatroGame(seed=seed, deck_key="b_red", stake=1, ruleset="mlb")
    steps = 0
    while g.state != State.SHOP and steps < 4000:
        acts = g.legal_actions()
        if not acts:
            break
        a = next((x for x in acts if x["type"] == "play_blind"), acts[0])
        g.step(a)
        steps += 1
        if g.state == State.SELECTING_HAND:
            acts2 = g.legal_actions()
            play = [x for x in acts2 if x["type"] == "play"]
            if play:
                g.step(play[0])
                steps += 1
    assert g.state == State.SHOP
    return g


def _shelf_has_hit(clone, cfg=hitmod.DEFAULT) -> bool:
    """Same classification ``hit.shop_slot_distribution`` uses per component, applied to an
    ACTUAL post-reroll shelf -- the ground truth this test checks the analytic number
    against."""
    owned = [j.key for j in clone.jokers]
    for it in clone.current_shop:
        if it.kind == "joker" and hitmod.is_hit(it.key, owned, clone.ante, cfg):
            return True
        if it.kind == "tarot" and hitmod.tarot_value(it.key, cfg) > 0:
            return True
        if it.kind == "planet" and hitmod.planet_value(clone, it.key, cfg) > 0:
            return True
        if it.kind == "spectral" and hitmod.spectral_value(it.key, cfg) > 0:
            return True
    return False


def _empirical_reroll_p_hit(game, n_trials: int, cfg=hitmod.DEFAULT) -> tuple[float, float]:
    """Reroll ``n_trials`` DETERMINIZED clones; return (empirical_p, binomial_95%_halfwidth).

    ``balatro_sim.shop.reroll_shop`` (the engine-level wrapper) is money-gated: it silently
    no-ops (returns False, shelf untouched) when the clone cannot afford the reroll cost --
    which is EXACTLY what left an earlier version of this test asserting a state's shelf
    never changed across 500 "trials" (100%/0% empirical rates with zero variance). The
    analytic model has no opinion on affordability (a Row simply would not exist if the
    reroll were illegal -- ``decide.py`` only builds a reroll Row when ``legal_actions()``
    already confirms it is affordable), so the validation must remove that gate rather than
    hit it: top up the clone's dollars before rerolling."""
    assert hasattr(game, "clone_determinized"), \
        "W2's clone_determinized is required for this validation (brief: no rng-state hacks)"
    hits = 0
    for i in range(n_trials):
        clone = game.clone_determinized(seed=1_000_003 * (i + 1) + 17)   # large odd stride, avoid collisions
        clone.dollars = max(clone.dollars, clone.reroll_cost + 1000)     # never let affordability gate the reroll
        assert reroll_shop(clone), "reroll_shop unexpectedly refused on a well-funded clone"
        if _shelf_has_hit(clone, cfg):
            hits += 1
    p = hits / n_trials
    halfwidth = Z_95 * math.sqrt(max(p * (1 - p), 1e-9) / n_trials)
    return p, halfwidth


def _check_one_state(seed: str, n_trials: int = N_TRIALS, slack: float = 0.06):
    game = _fresh_shop_game(seed)
    analytic_p, _mean_val, _details = hitmod.reroll_p_hit(game)
    empirical_p, halfwidth = _empirical_reroll_p_hit(game, n_trials)
    # `slack` absorbs the two known, documented approximations: the independence assumption
    # across shelf slots (dedup makes the true P(>=1 hit) a touch higher) and the small-N
    # binomial noise at N_TRIALS=150 (STATS_NOTES.md's larger run tightens this).
    assert abs(analytic_p - empirical_p) <= halfwidth + slack, (
        f"seed={seed}: analytic={analytic_p:.3f} empirical={empirical_p:.3f}+-{halfwidth:.3f}")
    return analytic_p, empirical_p, halfwidth


def test_phit_validation_ante1_shop():
    _check_one_state("11111111")


def test_phit_validation_second_seed():
    _check_one_state("1558AXDL")


def test_phit_validation_third_seed():
    _check_one_state("15H9Z3IY")
