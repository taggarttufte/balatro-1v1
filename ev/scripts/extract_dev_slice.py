"""
extract_dev_slice.py â€” W-EXTRACT's own measurements (brief Â§3.6 b/d).

Three subcommands, all parallel over a spawn ``multiprocessing.Pool`` exactly like
``gate_ev_player.py`` / ``h2h.py`` (``--procs``, box-shared cap 8):

    # 1. find seeds whose ante-1/2 board actually HAS procs (the dev slice's seed choice)
    python mp/ev/scripts/extract_dev_slice.py scan --seeds 126 --procs 8

    # 2. the 12-seed dev slice: extraction ON vs OFF, same seeds, vanilla single-player
    python mp/ev/scripts/extract_dev_slice.py slice --seeds <s1,s2,...> --procs 8

    # 3. paired h2h, new-fast vs old-fast, both seatings per seed
    python mp/ev/scripts/extract_dev_slice.py h2h --n-seeds 30 --procs 8 --max-steps 4000

"old fast" is the SAME player with ``HandConfig.extract=False`` â€” the only difference is
this workstream's layer, so the comparison is exactly paired (same seeds, same shop rules,
epsilon 0, and in `h2h` both seatings).

Metrics the brief asks for: mean end-of-ante-2 money, tarots used, blinds lost.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

_HERE = Path(__file__).resolve().parent            # mp/ev/scripts
_EV = _HERE.parent
_MP = _EV.parent
for _p in (str(_EV), str(_MP), str(_MP / "eval"), str(_MP / "agent")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _bootstrap  # noqa: E402,F401
from _bootstrap import BalatroGame, State, MLBMatch  # noqa: E402
import common as C  # noqa: E402

#: jokers whose money is a per-ACTION proc the extraction layer prices
PROC_JOKERS = ("j_business", "j_reserved_parking", "j_faceless", "j_ticket", "j_trading",
               "j_rough_gem", "j_mail", "j_delayed_grat", "j_todo_list")


def _cfgs(extract: bool):
    import hand as H
    return replace(H.DEFAULT_HAND_CONFIG, extract=extract)


def _player(extract: bool, seed: int = 0):
    from player import EVPlayer
    return EVPlayer(budget="fast", seed=seed, epsilon=0.0, hand_cfg=_cfgs(extract))


def _proc_exposure(game) -> dict:
    """What the extraction layer could possibly fire on, in THIS state."""
    keys = [j.key for j in game.jokers]
    deck = game.full_deck
    return {
        "jokers": sorted(k for k in keys if k in PROC_JOKERS),
        "gold_seal": sum(1 for c in deck if c.seal == "Gold"),
        "purple_seal": sum(1 for c in deck if c.seal == "Purple"),
        "gold_enh": sum(1 for c in deck if c.enhancement == "Gold"),
        "lucky": sum(1 for c in deck if c.enhancement == "Lucky"),
    }


def _score(ex: dict) -> int:
    return (3 * len(ex["jokers"]) + 2 * ex["purple_seal"] + 2 * ex["gold_seal"]
            + ex["gold_enh"] + ex["lucky"])


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ one vanilla run

def run_one(args) -> dict:
    """One vanilla single-player run up to ``stop_ante`` (or death), instrumented.

    NOTE on "blinds lost": a VANILLA (non-MLB) run has no lives — a failed blind is
    GAME_OVER, so the honest per-run measure is ``died`` (the run ended before
    ``stop_ante``) plus ``blinds_cleared``, not a life counter."""
    seed, extract, stop_ante = args
    pl = _player(extract)
    g = BalatroGame(seed=seed, deck_key="b_red", stake=1, ruleset="vanilla")
    money_at = {}
    exposure_at = {}
    blinds_lost = 0
    blinds_cleared = 0
    lives0 = g.lives
    hand_ms = []
    steps = 0
    while g.state != State.GAME_OVER and g.ante <= stop_ante and steps < 4000:
        if g.ante not in money_at:
            money_at[g.ante] = g.dollars
            exposure_at[g.ante] = _proc_exposure(g)
        legal = g.legal_actions()
        if not legal:
            g.step({"type": "advance"})
            steps += 1
            continue
        t0 = time.perf_counter()
        a = pl.act(g)
        dt = (time.perf_counter() - t0) * 1000.0
        if g.state == State.SELECTING_HAND:
            hand_ms.append(dt)
        before = (g.ante, g.blind_idx, g.lives, g.state)
        g.step(a)
        steps += 1
        if g.lives < before[2]:
            blinds_lost += 1
        if g.state == State.ROUND_EVAL and before[3] != State.ROUND_EVAL:
            blinds_cleared += 1
    ex = _proc_exposure(g)
    ex2 = exposure_at.get(3) or exposure_at.get(2) or ex
    hand_ms.sort()
    return {
        "seed": seed, "extract": extract,
        "money_end_ante2": money_at.get(3, g.dollars),      # $ at the START of ante 3
        "money_at_stop": money_at.get(stop_ante, g.dollars),
        "money_final": g.dollars,
        "tarots_used": len(g.tarots_used),
        "planets_used": len(g.planets_used),
        "blinds_lost": blinds_lost + (lives0 - g.lives - blinds_lost if g.lives < lives0 else 0),
        "lives": g.lives,
        "blinds_cleared": blinds_cleared,
        "died": int(g.state == State.GAME_OVER),
        "final_ante": g.ante,
        "exposure": ex2, "exposure_score": _score(ex2),
        "hand_ms_mean": (sum(hand_ms) / len(hand_ms)) if hand_ms else 0.0,
        "hand_ms_p95": hand_ms[int(0.95 * (len(hand_ms) - 1))] if hand_ms else 0.0,
        "n_hand_decisions": len(hand_ms),
    }


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ paired h2h

def run_match(args) -> list:
    """One seed, BOTH seatings: extraction-ON fast vs extraction-OFF fast."""
    seed, lives, max_steps = args
    out = []
    for seating in (0, 1):
        a = _player(True, seed=1)
        b = _player(False, seed=2)
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
        })
    return out


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ driving

def _pool_map(fn, jobs, procs):
    if procs <= 1:
        return [fn(j) for j in jobs]
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(processes=procs) as pool:
        return pool.map(fn, jobs, chunksize=1)


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else 0.0


def cmd_scan(a):
    seeds = C.DEFAULT_SEEDS[a.offset:a.offset + a.seeds]
    rows = _pool_map(run_one, [(s, False, a.to_ante) for s in seeds], a.procs)
    rows.sort(key=lambda r: (-r["exposure_score"], r["seed"]))
    for r in rows[:a.top]:
        print(f"{r['seed']}  score {r['exposure_score']:>3}  {r['exposure']}")
    print("\nSLICE=" + ",".join(r["seed"] for r in rows[:a.top]))
    return rows


def cmd_slice(a):
    seeds = a.seeds.split(",")
    jobs = [(s, e, a.to_ante) for s in seeds for e in (True, False)]
    rows = _pool_map(run_one, jobs, a.procs)
    on = [r for r in rows if r["extract"]]
    off = [r for r in rows if not r["extract"]]
    on.sort(key=lambda r: r["seed"])
    off.sort(key=lambda r: r["seed"])
    keys = ("money_end_ante2", "money_at_stop", "tarots_used", "planets_used",
            "blinds_cleared", "died", "blinds_lost", "final_ante",
            "money_final", "hand_ms_mean", "hand_ms_p95")
    print(f"{'metric':<20} {'extract ON':>12} {'extract OFF':>12} {'delta':>10}")
    summary = {}
    for k in keys:
        mo, mf = _mean([r[k] for r in on]), _mean([r[k] for r in off])
        summary[k] = {"on": mo, "off": mf, "delta": mo - mf}
        print(f"{k:<20} {mo:>12.3f} {mf:>12.3f} {mo - mf:>+10.3f}")
    print("\nper seed (money_at_stop / tarots_used / blinds_cleared / died):")
    for x, y in zip(on, off):
        print(f"  {x['seed']:<9} {x['money_at_stop']:>4}/{x['tarots_used']:>2}/{x['blinds_cleared']:>2}/{x['died']}"
              f"   vs   {y['money_at_stop']:>4}/{y['tarots_used']:>2}/{y['blinds_cleared']:>2}/{y['died']}")
    if a.out:
        Path(a.out).write_text(json.dumps({"summary": summary, "on": on, "off": off}, indent=1))
        print("wrote", a.out)
    return summary


def cmd_h2h(a):
    seeds = a.seeds.split(",") if a.seeds else C.DEFAULT_SEEDS[a.offset:a.offset + a.n_seeds]
    t0 = time.perf_counter()
    got = _pool_map(run_match, [(s, a.lives, a.max_steps) for s in seeds], a.procs)
    trials = [t for chunk in got for t in chunk]
    dec = [t for t in trials if t["a_win"] is not None]
    wins = sum(1 for t in dec if t["a_win"])
    ci = C.bootstrap_ci([1.0 if t["a_win"] else 0.0 for t in dec])
    print(f"new-fast (extract ON) vs old-fast (extract OFF): {wins}/{len(dec)} decided "
          f"= {100.0 * wins / max(1, len(dec)):.1f}%  CI [{100 * ci['lo']:.1f}, {100 * ci['hi']:.1f}]")
    print(f"  undecided (max-steps): {len(trials) - len(dec)} of {len(trials)}")
    print(f"  mean lives margin (A-B): {_mean([t['lives_margin_a'] for t in trials]):+.3f}")
    print(f"  mean final ante  A {_mean([t['final_ante_a'] for t in trials]):.2f} "
          f"B {_mean([t['final_ante_b'] for t in trials]):.2f}")
    print(f"  mean final money A {_mean([t['money_a'] for t in trials]):.1f} "
          f"B {_mean([t['money_b'] for t in trials]):.1f}")
    print(f"  {len(trials)} matches in {time.perf_counter() - t0:.0f}s")
    if a.out:
        Path(a.out).write_text(json.dumps({"trials": trials, "wins": wins, "decided": len(dec)},
                                          indent=1))
        print("wrote", a.out)
    return trials


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d_procs = max(1, min(8, (os.cpu_count() or 2) // 2))

    s = sub.add_parser("scan")
    s.add_argument("--seeds", type=int, default=126)
    s.add_argument("--offset", type=int, default=0)
    s.add_argument("--top", type=int, default=12)
    s.add_argument("--to-ante", type=int, default=3)
    s.add_argument("--procs", type=int, default=d_procs)
    s.set_defaults(fn=cmd_scan)

    s = sub.add_parser("slice")
    s.add_argument("--seeds", required=True, help="comma-separated seed list")
    s.add_argument("--to-ante", type=int, default=3, help="stop at the START of this ante")
    s.add_argument("--procs", type=int, default=d_procs)
    s.add_argument("--out", default=None)
    s.set_defaults(fn=cmd_slice)

    s = sub.add_parser("h2h")
    s.add_argument("--seeds", default=None)
    s.add_argument("--n-seeds", type=int, default=30)
    s.add_argument("--offset", type=int, default=0)
    s.add_argument("--lives", type=int, default=4)
    s.add_argument("--max-steps", type=int, default=4000)
    s.add_argument("--procs", type=int, default=d_procs)
    s.add_argument("--out", default=None)
    s.set_defaults(fn=cmd_h2h)

    a = ap.parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
