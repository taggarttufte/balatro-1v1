"""test_race.py — the lives-race calculator (W5).  Pure math, milliseconds."""
from __future__ import annotations

import math

import pytest

import race as R


def _flat(mu: float, sigma: float = 0.35, ante0: int = 3) -> R.Curve:
    return R.Curve(ante0, (mu,), (sigma,), R.DEFAULT.default_slope)


def test_equal_curves_equal_lives_is_a_coin_flip():
    c = R.prior_curve(3)
    for lives in (1, 2, 4):
        assert R.p_win(c, c, lives, lives, 3) == pytest.approx(0.5, abs=1e-9)
    assert R.p_win(R.prior_curve(13), R.prior_curve(13), 4, 4, 13) == pytest.approx(0.5, abs=1e-9)


def test_symmetry_sums_to_one():
    a = R.prior_curve(3).shifted(0.2)
    b = R.prior_curve(3)
    for ml, tl, ante in ((4, 3, 3), (2, 4, 5), (1, 1, 9), (3, 4, 13)):
        assert R.p_win(a, b, ml, tl, ante) + R.p_win(b, a, tl, ml, ante) == pytest.approx(1.0, abs=1e-9)


def test_lives_asymmetry_and_monotonicity():
    c = R.prior_curve(3)
    p43 = R.p_win(c, c, 4, 3, 3)
    p41 = R.p_win(c, c, 4, 1, 3)
    assert 0.6 < p43 < 0.75
    assert p41 > 0.9
    assert R.p_win(c, c, 3, 4, 3) == pytest.approx(1 - p43)
    # more lives for me never hurts; more for them never helps
    prev = 0.0
    for ml in range(1, 6):
        p = R.p_win(c, c, ml, 3, 3)
        assert p >= prev - 1e-12
        prev = p


def test_terminal_states():
    c = R.prior_curve(3)
    assert R.p_win(c, c, 0, 3, 3) == 0.0
    assert R.p_win(c, c, 3, 0, 3) == 1.0
    assert R.p_win(c, c, 0, 0, 3) == 0.5


def test_stronger_curve_wins_more():
    base = R.prior_curve(3)
    ps = [R.p_win(base.shifted(d), base, 4, 4, 3) for d in (-0.3, -0.1, 0.0, 0.1, 0.3)]
    assert ps == sorted(ps)
    assert ps[2] == pytest.approx(0.5)
    assert ps[-1] > 0.85
    # a 2-life deficit can be overcome by a much stronger build
    assert R.p_win(base.shifted(0.3), base, 2, 4, 3) > 0.5


def test_closed_form_race_matches_known_values():
    assert R.closed_form_race(0.5, 0.5, 4, 4) == pytest.approx(0.5)
    assert R.closed_form_race(0.5, 0.5, 4, 3) == pytest.approx(0.65625)      # 21/32
    assert R.closed_form_race(0.5, 0.5, 1, 1) == pytest.approx(0.5)
    assert R.closed_form_race(0.0, 1.0, 1, 4) == pytest.approx(1.0)
    assert R.closed_form_race(1.0, 0.0, 4, 1) == pytest.approx(0.0)


def test_blind_failure_uses_the_target_table():
    weak = _flat(3.0, 0.2, ante0=3)                # 1000 chips at ante 3 (Small is 2000)
    strong = _flat(5.0, 0.2, ante0=3)
    assert R.p_fail_blind(weak, 3, 0) > 0.9
    assert R.p_fail_blind(strong, 3, 0) == R.DEFAULT.blind_fail_floor
    assert R.log10_target(3, 0) == pytest.approx(math.log10(2000))
    assert R.log10_target(12, 2) == pytest.approx(math.log10(600_000_000))


def test_both_collapsing_at_ante_13_is_a_life_count_race():
    w = R.Curve(13, (8.0,), (0.3,), 0.3)           # both far below the ante-13 targets
    assert R.p_win(w, w, 4, 4, 13) == pytest.approx(0.5)
    assert R.p_win(w, w, 4, 3, 13) == pytest.approx(0.75, abs=0.01)
    assert R.p_win(w, w, 4, 2, 13) > 0.99


def test_tie_mass_keeps_symmetry():
    cfg = R.RaceConfig(p_tie=0.2)
    c = R.prior_curve(3)
    assert R.p_win(c, c, 4, 4, 3, cfg=cfg) == pytest.approx(0.5)
    a = c.shifted(0.2)
    assert R.p_win(a, c, 4, 4, 3, cfg=cfg) + R.p_win(c, a, 4, 4, 3, cfg=cfg) == pytest.approx(1.0)


def test_curve_from_history_fits_a_line():
    log = [(2, 1, 1500, 1200), (3, 0, 3000, 4000), (4, 1, 12000, 9000)]
    c0 = R.curve_from_history(log, 0, 5)
    c1 = R.curve_from_history(log, 1, 5)
    assert c0.n_obs == 3 and c1.n_obs == 3
    assert R.DEFAULT.slope_min <= c0.slope <= R.DEFAULT.slope_max
    # player 0 scored more overall -> higher curve at ante 5
    assert c0.at(5)[0] > c1.at(5)[0]
    assert c0.at(5)[1] >= R.DEFAULT.sigma_floor
    # the fit extrapolates through the last point region
    assert 4.0 < c0.at(5)[0] < 5.2


def test_curve_from_history_edge_cases():
    prior = R.curve_from_history([], 0, 5)
    assert prior.n_obs == 0
    assert prior.at(5)[0] == pytest.approx(R.prior_curve(5).at(5)[0])
    one = R.curve_from_history([(2, None, 1000, 1000)], 0, 3)
    assert one.n_obs == 1
    assert one.at(3)[0] == pytest.approx(3.0 + R.DEFAULT.default_slope)
    # zeros (deck-outs) are dropped when other points exist
    z = R.curve_from_history([(2, 0, 0, 900), (3, 1, 5000, 100), (4, 1, 20000, 100)], 0, 5)
    assert z.n_obs == 2
    # same-ante duplicates fall back to the default slope
    d = R.curve_from_history([(2, 0, 1000, 900), (2, 1, 2000, 100)], 0, 3)
    assert d.slope == R.DEFAULT.default_slope


def test_as_curve_accepts_dicts_and_callables():
    d = R.as_curve({3: (4.0, 0.3), 4: (4.4, 0.3), 5: 4.8})
    assert d.at(3) == (4.0, 0.3)
    assert d.at(5)[1] == R.DEFAULT.sigma_prior
    assert d.slope == pytest.approx(0.4)
    assert d.at(7)[0] == pytest.approx(4.8 + 2 * 0.4)
    f = R.as_curve(lambda a: (3.0 + 0.3 * a, 0.2))
    assert f.at(4)[0] == pytest.approx(4.2)
    with pytest.raises(ValueError):
        R.as_curve({3: 4.0, 5: 4.5})
    with pytest.raises(TypeError):
        R.as_curve("nope")


def test_blinds_done_phase_and_table():
    c = R.prior_curve(3)
    a = c.shifted(0.2)
    p0 = R.p_win(a, c, 4, 4, 3, blinds_done=0)
    p2 = R.p_win(a, c, 4, 4, 3, blinds_done=2)
    assert 0 < p0 < 1 and 0 < p2 < 1
    rows = R.race_table(a, c, 4, 4, 3, n_antes=4)
    assert [r["ante"] for r in rows] == [3, 4, 5, 6]
    assert all(r["nemesis"] for r in rows)
    assert rows[0]["p_win_from_here"] == pytest.approx(p0)
    assert rows[0]["p_i_lose_nemesis"] < 0.5


def test_p_win_is_fast():
    import time
    c = R.prior_curve(3)
    a = c.shifted(0.1)
    t = time.perf_counter()
    for _ in range(200):
        R.p_win(a, c, 4, 4, 3)
    assert (time.perf_counter() - t) / 200 < 0.01        # << 10 ms each
