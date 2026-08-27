"""
shop_profile.py — W-SHOP's own measurements: what the player actually DOES in the shop.

Two subcommands, both parallel over a spawn ``multiprocessing.Pool`` exactly like
``gate_ev_player.py`` / ``h2h.py`` / ``extract_dev_slice.py``:

    # 1. the before/after profile table (offers, buys, take rates, reroll depth, money)
    python ev/scripts/shop_profile.py profile --seeds 80 --procs 12 --arms old,new

    # 2. paired h2h, new-shop vs old-shop, both seatings per seed (the thesis test)
    python ev/scripts/shop_profile.py h2h --n-seeds 30 --procs 8 --max-steps 4000

The two arms are the SAME player; only ``player.shop_arm_cfgs`` differs (``PlayerConfig``'s
``reroll_ev`` / ``pack_ev`` / ``fool_order`` and ``HandConfig.fool_order``), so the
comparison is exactly paired — same seeds, same hand player, epsilon 0, and in ``h2h`` both
seatings.

What a "visit" is
-----------------
One SHOP entry through its ``leave_shop``, including any ``BOOSTER_OPEN`` excursion.  An
*offer* is one shelf/voucher/booster ITEM instance the visit put in front of the player:
the ``shop_joker_max`` shelf slots at entry plus ``shop_joker_max`` more per reroll, plus
the voucher slot and the two booster slots (neither of which a reroll touches).  A *take* is
a ``buy`` of that family.  ``take_rate = takes / offers`` per family.

Two honest caveats, both of which the report prints:

* rerolling multiplies the JOKER/consumable/card offer count, so a deep-rolling arm's
  shelf take-rates fall mechanically even if it buys strictly more.  ``buys_per_visit`` and
  the raw counts are printed next to the rates for exactly this reason.  Booster and
  voucher rates are unaffected (a reroll leaves both untouched — shop.py:496).
* ``interest_income`` is ``min(dollars // 5, interest_cap)`` read off the balance in the
  step BEFORE the round-end transition, i.e. the engine's own formula (game.py:2010)
  evaluated one step early: it therefore misses the Gold-ENHANCEMENT held-card row that
  ``_end_round`` pays into the interest base.  It is a lower bound, identical in method
  across arms.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent            # ev/scripts
_EV = _HERE.parent
_MP = _EV.parent
for _p in (str(_EV), str(_MP), str(_MP / "eval"), str(_MP / "agent"), str(_MP / "stats")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _bootstrap  # noqa: E402,F401
from _bootstrap import BalatroGame, State, MLBMatch  # noqa: E402
import common as C  # noqa: E402

#: booster families, in the order the report prints them
PACK_FAMILIES = ("arcana", "celestial", "buffoon", "standard", "spectral")
#: shelf families
SHELF_FAMILIES = ("joker", "tarot", "planet", "spectral", "card")
FAMILIES = SHELF_FAMILIES + ("voucher",) + tuple(f"pack:{k}" for k in PACK_FAMILIES)


def _player(arm: str, seed: int = 0, overrides: tuple = ()):
    """``arm`` = "old" | "new".  ``overrides`` = ``(("field", value), ...)`` applied to the
    ``PlayerConfig`` — how the constant sweeps in SHOP_NOTES.md were run."""
    from dataclasses import replace
    import player as P
    cfg, hcfg = P.shop_arm_cfgs(arm)
    if overrides:
        cfg = replace(cfg, **{k: v for k, v in overrides})
    return P.EVPlayer(budget="fast", seed=seed, epsilon=0.0, cfg=cfg, hand_cfg=hcfg)


def _pack_family(key: str) -> str:
    for k in PACK_FAMILIES:
        if k in key:
            return k
    return "?"


def _item_family(item) -> str:
    if item.kind == "booster":
        return "pack:" + _pack_family(item.key)
    return item.kind


def _shelf_offers(game, counter: Counter, shelf_only: bool = False) -> None:
    """Count every unsold item currently on display.  ``shelf_only``: just the reroll-able
    slots (``shop.SHELF_KINDS``), used after a reroll refills them."""
    for item in game.current_shop:
        if item.sold:
            continue
        fam = _item_family(item)
        if shelf_only and fam not in SHELF_FAMILIES:
            continue
        counter[fam] += 1


def _deck_sculpting(game) -> dict:
    deck = game.full_deck
    return {
        "enhanced": sum(1 for c in deck if c.enhancement != "None"),
        "sealed": sum(1 for c in deck if c.seal != "None"),
        "editioned": sum(1 for c in deck if c.edition != "None"),
        "deck_size": len(deck),
    }


# ════════════════════════════════════════════════════════════════════ one vanilla run

def run_one(args) -> dict:
    """One instrumented vanilla single-player run to ``stop_ante`` (or death)."""
    seed, arm, stop_ante, max_steps, overrides = args
    pl = _player(arm, 0, overrides)
    g = BalatroGame(seed=seed, deck_key="b_red", stake=1, ruleset="vanilla")

    offers: Counter = Counter()
    takes: Counter = Counter()
    pack_opens: Counter = Counter()     # packs actually opened (bought), by family
    pack_picks: Counter = Counter()     # pick_booster actions taken inside them
    pack_skips: Counter = Counter()
    reroll_hist: Counter = Counter()    # rerolls-in-a-visit -> number of visits
    entry_money: list = []
    leave_money: list = []
    reroll_spend = 0
    buy_spend = 0
    sells = 0
    interest_income = 0
    money_at = {}
    blinds_cleared = 0
    shop_ms: list = []
    hand_ms: list = []

    in_visit = False
    visit_rerolls = 0
    cur_pack = None
    steps = 0

    while g.state != State.GAME_OVER and g.ante <= stop_ante and steps < max_steps:
        if g.ante not in money_at:
            money_at[g.ante] = g.dollars
        legal = g.legal_actions()
        if not legal:
            g.step({"type": "advance"})
            steps += 1
            continue
        if g.state == State.SHOP and not in_visit:
            in_visit = True
            visit_rerolls = 0
            entry_money.append(g.dollars)
            _shelf_offers(g, offers)

        t0 = time.perf_counter()
        a = pl.act(g)
        dt = (time.perf_counter() - t0) * 1000.0
        if g.state == State.SELECTING_HAND:
            hand_ms.append(dt)
        elif g.state in (State.SHOP, State.BOOSTER_OPEN):
            shop_ms.append(dt)

        pre_state = g.state
        pre_dollars = g.dollars
        # what the action is about to do, read BEFORE the step
        item = None
        if a.get("type") == "buy" and pre_state == State.SHOP:
            idx = a.get("item_idx", -1)
            if 0 <= idx < len(g.current_shop):
                item = g.current_shop[idx]
        if pre_state == State.BOOSTER_OPEN and cur_pack is None:
            cur_pack = "?"

        g.step(a)
        steps += 1

        t = a.get("type")
        if pre_state == State.SHOP:
            if t == "reroll":
                visit_rerolls += 1
                reroll_spend += max(0, pre_dollars - g.dollars)
                _shelf_offers(g, offers, shelf_only=True)
            elif t == "buy" and item is not None:
                fam = _item_family(item)
                takes[fam] += 1
                buy_spend += max(0, pre_dollars - g.dollars)
                if fam.startswith("pack:"):
                    pack_opens[fam[5:]] += 1
                    cur_pack = fam[5:]
            elif t == "sell_joker":
                sells += 1
            elif t == "leave_shop":
                leave_money.append(pre_dollars)
                reroll_hist[visit_rerolls] += 1
                in_visit = False
        elif pre_state == State.BOOSTER_OPEN:
            fam = cur_pack if cur_pack and cur_pack != "?" else "?"
            if t == "pick_booster":
                pack_picks[fam] += len(a.get("indices", []) or [])
            elif t == "skip_booster":
                pack_skips[fam] += 1
                cur_pack = None
        if pre_state in (State.SELECTING_HAND, State.PVP_WAIT) and g.state == State.ROUND_EVAL:
            interest_income += min(max(0, pre_dollars) // 5, g.interest_cap)
            blinds_cleared += 1
        if g.state == State.SHOP and pre_state not in (State.SHOP, State.BOOSTER_OPEN) and in_visit:
            pass    # a booster excursion returned; the visit is still open

    n_visits = sum(reroll_hist.values())
    hand_ms.sort()
    shop_ms.sort()
    return {
        "seed": seed, "arm": arm,
        "offers": dict(offers), "takes": dict(takes),
        "pack_opens": dict(pack_opens), "pack_picks": dict(pack_picks),
        "pack_skips": dict(pack_skips),
        "reroll_hist": {str(k): v for k, v in reroll_hist.items()},
        "n_visits": n_visits,
        "rerolls_total": sum(int(k) * v for k, v in reroll_hist.items()),
        "visits_with_reroll": sum(v for k, v in reroll_hist.items() if int(k) > 0),
        "entry_money_mean": (sum(entry_money) / len(entry_money)) if entry_money else 0.0,
        "leave_money_mean": (sum(leave_money) / len(leave_money)) if leave_money else 0.0,
        "reroll_spend": reroll_spend, "buy_spend": buy_spend, "sells": sells,
        "interest_income": interest_income,
        "money_at_ante3": money_at.get(3, g.dollars),
        "money_final": g.dollars,
        "final_ante": g.ante, "died": int(g.state == State.GAME_OVER),
        "blinds_cleared": blinds_cleared,
        "jokers": len(g.jokers),
        "tarots_used": len(g.tarots_used), "planets_used": len(g.planets_used),
        "sculpt": _deck_sculpting(g),
        "hand_ms_mean": (sum(hand_ms) / len(hand_ms)) if hand_ms else 0.0,
        "shop_ms_mean": (sum(shop_ms) / len(shop_ms)) if shop_ms else 0.0,
        "shop_ms_p95": shop_ms[int(0.95 * (len(shop_ms) - 1))] if shop_ms else 0.0,
        "n_shop_decisions": len(shop_ms),
        "steps": steps,
    }


# ════════════════════════════════════════════════════════════════════════ paired h2h

def run_match(args) -> list:
    """One seed, BOTH seatings: new-shop fast vs old-shop fast."""
    seed, lives, max_steps, overrides = args
    out = []
    for seating in (0, 1):
        a = _player("new", 1, overrides)
        b = _player("old", 2)
        a.reset()
        b.reset()
        pol_a, pol_b = C.adapt_player(a), C.adapt_player(b)
        policies = [pol_a, pol_b] if seating == 0 else [pol_b, pol_a]
        a_idx = 0 if seating == 0 else 1
        m = MLBMatch(seed=seed, deck_key="b_red", stake=1, lives=lives)
        t0 = time.perf_counter()
        m.play_out(policies, max_steps=max_steps)
        g = m.games
        out.append({
            "seed": str(seed), "seating": seating, "done": bool(m.done), "steps": m.steps,
            "seconds": time.perf_counter() - t0,
            "a_win": (bool(m.winner == a_idx) if m.done else None),
            "lives_margin_a": g[a_idx].lives - g[1 - a_idx].lives,
            "final_ante_a": g[a_idx].ante, "final_ante_b": g[1 - a_idx].ante,
            "money_a": g[a_idx].dollars, "money_b": g[1 - a_idx].dollars,
            "jokers_a": len(g[a_idx].jokers), "jokers_b": len(g[1 - a_idx].jokers),
        })
    return out


# ═══════════════════════════════════════════════════════════════════════════ driving

def _pool_map(fn, jobs, procs):
    if procs <= 1:
        return [fn(j) for j in jobs]
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(processes=procs) as pool:
        return pool.map(fn, jobs, chunksize=1)


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else 0.0


def _agg(rows: list) -> dict:
    """Pooled counters + per-run means for one arm."""
    offers, takes = Counter(), Counter()
    opens, picks, skips, rh = Counter(), Counter(), Counter(), Counter()
    for r in rows:
        offers.update(r["offers"])
        takes.update(r["takes"])
        opens.update(r["pack_opens"])
        picks.update(r["pack_picks"])
        skips.update(r["pack_skips"])
        rh.update({int(k): v for k, v in r["reroll_hist"].items()})
    n_visits = sum(rh.values())
    rates = {}
    for fam in FAMILIES:
        o, t = offers.get(fam, 0), takes.get(fam, 0)
        rates[fam] = {"offers": o, "takes": t, "rate": (t / o) if o else 0.0}
    return {
        "n_runs": len(rows), "n_visits": n_visits,
        "rates": rates,
        "reroll_hist": {str(k): rh.get(k, 0) for k in sorted(rh)},
        "rerolls_per_visit": (sum(k * v for k, v in rh.items()) / n_visits) if n_visits else 0.0,
        "pct_visits_with_reroll": (100.0 * sum(v for k, v in rh.items() if k > 0) / n_visits) if n_visits else 0.0,
        "max_rerolls_in_a_visit": max(rh) if rh else 0,
        "pack_opens": dict(opens), "pack_picks": dict(picks), "pack_skips": dict(skips),
        "entry_money": _mean([r["entry_money_mean"] for r in rows]),
        "leave_money": _mean([r["leave_money_mean"] for r in rows]),
        "interest_income": _mean([r["interest_income"] for r in rows]),
        "reroll_spend": _mean([r["reroll_spend"] for r in rows]),
        "buy_spend": _mean([r["buy_spend"] for r in rows]),
        "sells": _mean([r["sells"] for r in rows]),
        "ante1_clear": _mean([1.0 if r["final_ante"] >= 2 else 0.0 for r in rows]),
        "ante2_clear": _mean([1.0 if r["final_ante"] >= 3 else 0.0 for r in rows]),
        "money_at_ante3": _mean([r["money_at_ante3"] for r in rows]),
        "money_final": _mean([r["money_final"] for r in rows]),
        "final_ante": _mean([r["final_ante"] for r in rows]),
        "blinds_cleared": _mean([r["blinds_cleared"] for r in rows]),
        "died": _mean([r["died"] for r in rows]),
        "jokers": _mean([r["jokers"] for r in rows]),
        "tarots_used": _mean([r["tarots_used"] for r in rows]),
        "planets_used": _mean([r["planets_used"] for r in rows]),
        "enhanced": _mean([r["sculpt"]["enhanced"] for r in rows]),
        "sealed": _mean([r["sculpt"]["sealed"] for r in rows]),
        "shop_ms_mean": _mean([r["shop_ms_mean"] for r in rows]),
        "shop_ms_p95": _mean([r["shop_ms_p95"] for r in rows]),
        "hand_ms_mean": _mean([r["hand_ms_mean"] for r in rows]),
        "n_shop_decisions": _mean([r["n_shop_decisions"] for r in rows]),
    }


_SCALARS = ("ante1_clear", "ante2_clear",
            "entry_money", "leave_money", "interest_income", "reroll_spend", "buy_spend",
            "sells", "money_at_ante3", "money_final", "final_ante", "blinds_cleared",
            "died", "jokers", "tarots_used", "planets_used", "enhanced", "sealed",
            "rerolls_per_visit", "pct_visits_with_reroll", "shop_ms_mean", "shop_ms_p95",
            "hand_ms_mean")


def _print_arms(aggs: dict) -> None:
    arms = list(aggs)
    w = 14
    print("\n=== take rates (takes / offers) " + "=" * 40)
    head = f"{'family':<16}" + "".join(f"{a:>{w}}" for a in arms)
    print(head)
    for fam in FAMILIES:
        cells = ""
        for a in arms:
            r = aggs[a]["rates"][fam]
            cells += f"{100 * r['rate']:>9.1f}% ({r['takes']}/{r['offers']})".rjust(w)
        print(f"{fam:<16}{cells}")
    print("\n=== rerolls per visit " + "=" * 50)
    ks = sorted({int(k) for a in arms for k in aggs[a]["reroll_hist"]})
    print(f"{'rerolls':<16}" + "".join(f"{a:>{w}}" for a in arms))
    for k in ks:
        row = "".join(f"{aggs[a]['reroll_hist'].get(str(k), 0):>{w}}" for a in arms)
        print(f"{k:<16}{row}")
    print(f"{'visits':<16}" + "".join(f"{aggs[a]['n_visits']:>{w}}" for a in arms))
    print(f"{'max in a visit':<16}" + "".join(f"{aggs[a]['max_rerolls_in_a_visit']:>{w}}" for a in arms))
    print("\n=== per-run means " + "=" * 54)
    print(f"{'metric':<22}" + "".join(f"{a:>{w}}" for a in arms))
    for k in _SCALARS:
        print(f"{k:<22}" + "".join(f"{aggs[a][k]:>{w}.3f}" for a in arms))


def _parse_set(text: str):
    k, _, v = text.partition("=")
    try:
        val = int(v)
    except ValueError:
        try:
            val = float(v)
        except ValueError:
            val = v
    return (k, val)


def cmd_profile(a):
    arms = [x for x in a.arms.split(",") if x]
    seeds = C.DEFAULT_SEEDS[a.offset:a.offset + a.seeds]
    ov = tuple(_parse_set(x) for x in (a.set or []))
    jobs = [(s, arm, a.to_ante, a.max_steps, (() if arm == "old" else ov))
            for arm in arms for s in seeds]
    t0 = time.perf_counter()
    rows = _pool_map(run_one, jobs, a.procs)
    aggs = {arm: _agg([r for r in rows if r["arm"] == arm]) for arm in arms}
    print(f"{len(seeds)} seeds x {len(arms)} arm(s), to ante {a.to_ante}, "
          f"{time.perf_counter() - t0:.0f}s on {a.procs} procs")
    _print_arms(aggs)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"aggs": aggs, "rows": rows,
                                           "seeds": seeds, "to_ante": a.to_ante}, indent=1))
        print("\nwrote", a.out)
    return aggs


def cmd_h2h(a):
    seeds = a.seeds.split(",") if a.seeds else C.DEFAULT_SEEDS[a.offset:a.offset + a.n_seeds]
    t0 = time.perf_counter()
    ov = tuple(_parse_set(x) for x in (a.set or []))
    got = _pool_map(run_match, [(s, a.lives, a.max_steps, ov) for s in seeds], a.procs)
    trials = [t for chunk in got for t in chunk]
    dec = [t for t in trials if t["a_win"] is not None]
    wins = sum(1 for t in dec if t["a_win"])
    ci = C.bootstrap_ci([1.0 if t["a_win"] else 0.0 for t in dec])
    print(f"new-shop vs old-shop: {wins}/{len(dec)} decided "
          f"= {100.0 * wins / max(1, len(dec)):.1f}%  CI [{100 * ci['lo']:.1f}, {100 * ci['hi']:.1f}]")
    print(f"  undecided (max-steps): {len(trials) - len(dec)} of {len(trials)}")
    print(f"  mean lives margin (A-B): {_mean([t['lives_margin_a'] for t in trials]):+.3f}")
    print(f"  mean final ante  A {_mean([t['final_ante_a'] for t in trials]):.2f} "
          f"B {_mean([t['final_ante_b'] for t in trials]):.2f}")
    print(f"  mean final money A {_mean([t['money_a'] for t in trials]):.1f} "
          f"B {_mean([t['money_b'] for t in trials]):.1f}")
    print(f"  mean jokers      A {_mean([t['jokers_a'] for t in trials]):.2f} "
          f"B {_mean([t['jokers_b'] for t in trials]):.2f}")
    print(f"  {len(trials)} matches in {time.perf_counter() - t0:.0f}s")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            {"trials": trials, "wins": wins, "decided": len(dec), "ci": ci}, indent=1))
        print("wrote", a.out)
    return trials


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d_procs = max(1, min(8, (os.cpu_count() or 2) // 2))

    s = sub.add_parser("profile")
    s.add_argument("--seeds", type=int, default=80)
    s.add_argument("--offset", type=int, default=0)
    s.add_argument("--arms", default="old,new")
    s.add_argument("--to-ante", type=int, default=8, help="stop at the START of this ante")
    s.add_argument("--max-steps", type=int, default=4000)
    s.add_argument("--procs", type=int, default=d_procs)
    s.add_argument("--out", default=None)
    s.add_argument("--set", action="append", default=None,
                   help="PlayerConfig override for the NEW arm, e.g. --set reroll_hurdle=2.0")
    s.set_defaults(fn=cmd_profile)

    s = sub.add_parser("h2h")
    s.add_argument("--seeds", default=None)
    s.add_argument("--n-seeds", type=int, default=30)
    s.add_argument("--offset", type=int, default=0)
    s.add_argument("--lives", type=int, default=4)
    s.add_argument("--max-steps", type=int, default=4000)
    s.add_argument("--procs", type=int, default=d_procs)
    s.add_argument("--out", default=None)
    s.add_argument("--set", action="append", default=None)
    s.set_defaults(fn=cmd_h2h)

    a = ap.parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
