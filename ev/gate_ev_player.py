"""
gate_ev_player.py — Phase 5 gate 2 (W3): the analytic EV player on the 126 ground-truth
seeds, vanilla ruleset, Red deck, White stake, against the scripted greedy baseline.

    python ev/gate_ev_player.py --procs 16                       # the full gate (126 seeds)
    python ev/gate_ev_player.py --seeds 12 --procs 4             # a small subset
    python ev/gate_ev_player.py --seeds 12 --offset 60 --procs 4 --players fast,greedy

Writes ``results/ev_player_gate_<date>.md`` + ``.json`` (``--out`` to override).  Per
seed and player: furthest ante / blind, won, blinds cleared, $ at the start of ante 3, hands
unused per cleared blind, wall-clock per SELECTING_HAND / SHOP decision; in-process
draw-order invariance checks (permute ``game.deck`` -> identical decision); one
``MLBMatch`` EVPlayer-vs-EVPlayer to completion.  Parallel over (seed, player) with
``multiprocessing`` (spawn-safe: the worker re-imports the bootstrap).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent                    # ev
_MP = _HERE.parent
for _p in (str(_HERE), str(_MP), str(_MP / "eval"), str(_MP / "agent")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _bootstrap  # noqa: E402,F401
from _bootstrap import BalatroGame, State, MLBMatch  # noqa: E402
import common as C  # noqa: E402  (eval/common.py)

_BLIND_ORDER = {"Small": 0, "Big": 1, "Boss": 2}
PLAYERS = ("fast", "full", "greedy")


def _make_policy(kind: str):
    if kind == "greedy":
        from mlb_match_demo import ScriptedPlayer, make_policy
        pol = make_policy(ScriptedPlayer(hand="greedy"))
        return lambda g: pol(C.SoloShim(games=[g]), 0, g.legal_actions()), None
    from player import EVPlayer
    pl = EVPlayer(budget=kind)
    return pl.act, pl


def run_seed(args) -> dict:
    """One (seed, player) vanilla run with instrumentation.  Module-level for pickling."""
    seed, kind, n_inv_checks = args
    policy, player = _make_policy(kind)
    game = BalatroGame(seed=seed, deck_key="b_red", stake=1, ruleset="vanilla")
    hand_ms: list = []
    shop_ms: list = []
    other_ms: list = []
    blinds_cleared = 0
    hands_unused = 0
    money_at_ante3 = None
    furthest = (game.ante, _BLIND_ORDER.get(game.current_blind.kind, 0))
    inv_checked = inv_failed = 0
    steps = 0
    t_run = time.perf_counter()
    while game.state != State.GAME_OVER and steps < 200_000:
        key = (game.ante, _BLIND_ORDER.get(game.current_blind.kind, 0))
        if key > furthest:
            furthest = key
        if money_at_ante3 is None and game.ante >= 3:
            money_at_ante3 = game.dollars
        st = game.state
        t = time.perf_counter()
        a = policy(game)
        dt = (time.perf_counter() - t) * 1000.0
        if st == State.SELECTING_HAND:
            hand_ms.append(dt)
            if kind != "greedy" and inv_checked < n_inv_checks and len(hand_ms) % 3 == 0:
                # draw-order invariance: permute the draw pile, decide again
                saved = list(game.deck)
                random.Random(len(hand_ms)).shuffle(game.deck)
                a2 = policy(game)
                game.deck = saved
                inv_checked += 1
                if a2 != a:
                    inv_failed += 1
        elif st in (State.SHOP, State.BOOSTER_OPEN):
            shop_ms.append(dt)
        else:
            other_ms.append(dt)
        was_hand = st == State.SELECTING_HAND
        hands_before = game.hands_left
        game.step(a)
        steps += 1
        if was_hand and game.state == State.ROUND_EVAL:
            blinds_cleared += 1
            hands_unused += max(0, game.hands_left)
    key = (game.ante, _BLIND_ORDER.get(game.current_blind.kind, 0))
    if key > furthest:
        furthest = key
    kind_by_idx = {v: k for k, v in _BLIND_ORDER.items()}
    won = bool(game._obs().won)
    return {
        "seed": seed, "player": kind, "won": won,
        "furthest_ante": furthest[0], "furthest_blind": kind_by_idx[furthest[1]],
        "final_ante": game.ante, "final_money": game.dollars,
        "blinds_cleared": blinds_cleared, "hands_unused": hands_unused,
        "money_at_ante3": money_at_ante3,
        "steps": steps, "seconds": time.perf_counter() - t_run,
        "hand_ms": hand_ms, "shop_ms": shop_ms, "other_ms": other_ms,
        "inv_checked": inv_checked, "inv_failed": inv_failed,
    }


def _pct(xs, q):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def summarise(rows: list) -> dict:
    n = len(rows)
    out = {"n": n}
    for k, fn in (("ante1_clear", lambda r: r["furthest_ante"] >= 2), ("ante2_clear", lambda r: r["furthest_ante"] >= 3),
                  ("ante3_clear", lambda r: r["furthest_ante"] >= 4), ("ante4_clear", lambda r: r["furthest_ante"] >= 5),
                  ("won", lambda r: r["won"])):
        vals = [float(fn(r)) for r in rows]
        out[k] = C.bootstrap_ci(vals)
    out["mean_final_ante"] = C.bootstrap_ci([float(r["furthest_ante"]) for r in rows])
    out["mean_blinds_cleared"] = C.bootstrap_ci([float(r["blinds_cleared"]) for r in rows])
    m3 = [float(r["money_at_ante3"]) for r in rows if r["money_at_ante3"] is not None]
    out["money_at_ante3"] = C.bootstrap_ci(m3) if m3 else {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    tot_cleared = sum(r["blinds_cleared"] for r in rows)
    out["hands_unused_per_cleared_blind"] = (sum(r["hands_unused"] for r in rows) / tot_cleared) if tot_cleared else float("nan")
    hand = [x for r in rows for x in r["hand_ms"]]
    shop = [x for r in rows for x in r["shop_ms"]]
    out["hand_ms"] = {"mean": statistics.fmean(hand) if hand else float("nan"), "p50": _pct(hand, 0.5),
                      "p95": _pct(hand, 0.95), "max": max(hand) if hand else float("nan"), "n": len(hand)}
    out["shop_ms"] = {"mean": statistics.fmean(shop) if shop else float("nan"), "p50": _pct(shop, 0.5),
                      "p95": _pct(shop, 0.95), "max": max(shop) if shop else float("nan"), "n": len(shop)}
    out["inv_checked"] = sum(r["inv_checked"] for r in rows)
    out["inv_failed"] = sum(r["inv_failed"] for r in rows)
    out["seconds_total"] = sum(r["seconds"] for r in rows)
    return out


def paired(rows_a: list, rows_b: list) -> dict:
    by_seed = {r["seed"]: r for r in rows_b}
    pairs = [(r, by_seed[r["seed"]]) for r in rows_a if r["seed"] in by_seed]
    out = {"n": len(pairs)}
    for k, fn in (("ante1_clear", lambda r: float(r["furthest_ante"] >= 2)), ("final_ante", lambda r: float(r["furthest_ante"])),
                  ("blinds_cleared", lambda r: float(r["blinds_cleared"])), ("final_money", lambda r: float(r["final_money"]))):
        out[k] = C.paired_bootstrap_ci([fn(a) for a, _ in pairs], [fn(b) for _, b in pairs])
    return out


def run_mlb_match(seed: str) -> dict:
    from player import EVPlayer
    players = [EVPlayer(seed=0), EVPlayer(seed=1)]
    m = MLBMatch(seed=seed)
    t = time.perf_counter()
    while not m.done and m.steps < 200_000:
        p = m.current_player()
        if p is None:
            return {"seed": seed, "done": False, "error": "wedged"}
        m.step(p, players[p].act(m.games[p]))
    return {"seed": seed, "done": m.done, "winner": m.winner, "steps": m.steps,
            "final_ante": [g.ante for g in m.games], "lives": [g.lives for g in m.games],
            "pvp_log": list(m.pvp_log), "seconds": time.perf_counter() - t}


def _fmt_ci(d, pct=False):
    if not isinstance(d, dict):
        return f"{d:.2f}"
    if pct:
        return f"{100 * d['point']:.1f}% [{100 * d['lo']:.1f}, {100 * d['hi']:.1f}]"
    return f"{d['point']:.2f} [{d['lo']:.2f}, {d['hi']:.2f}]"


def write_report(path_md: Path, path_json: Path, summary: dict, per_seed: dict, deltas: dict, mlb: dict, meta: dict):
    lines = [f"# EV player gate 2 — {meta['date']}", "",
             f"`{meta['command']}`", "",
             f"Seeds: {meta['n_seeds']} of {len(C.DEFAULT_SEEDS)} ground-truth seeds (offset {meta['offset']}); vanilla ruleset, Red deck, "
             f"White stake; {meta['procs']} processes; wall {meta['wall_s']:.0f} s.", "",
             "## Outcomes (bootstrap 95% CI)", "",
             "| metric | " + " | ".join(summary) + " |", "|---|" + "---|" * len(summary)]
    for k, pct in (("ante1_clear", True), ("ante2_clear", True), ("ante3_clear", True), ("ante4_clear", True), ("won", True),
                   ("mean_final_ante", False), ("mean_blinds_cleared", False), ("money_at_ante3", False)):
        lines.append(f"| {k} | " + " | ".join(_fmt_ci(summary[p][k], pct) for p in summary) + " |")
    lines.append("| hands unused / cleared blind | " + " | ".join(f"{summary[p]['hands_unused_per_cleared_blind']:.2f}" for p in summary) + " |")
    lines.append("| $ at ante 3: n reaching | " + " | ".join(str(summary[p]["money_at_ante3"]["n"]) for p in summary) + " |")
    lines += ["", "## Paired-by-seed deltas vs greedy (mean difference, bootstrap 95% CI)", ""]
    for p, d in deltas.items():
        lines.append(f"- **{p} − greedy** (n={d['n']}): ante-1 clear {_fmt_ci(d['ante1_clear'], True)}; final ante {_fmt_ci(d['final_ante'])}; "
                     f"blinds cleared {_fmt_ci(d['blinds_cleared'])}; final $ {_fmt_ci(d['final_money'])}")
    lines += ["", "## Wall-clock per decision (ms)", "", "| player | hand mean | hand p50 | hand p95 | hand max | n | shop mean | shop p95 | n |", "|---|---|---|---|---|---|---|---|---|"]
    for p, s in summary.items():
        h, sh = s["hand_ms"], s["shop_ms"]
        lines.append(f"| {p} | {h['mean']:.2f} | {h['p50']:.2f} | {h['p95']:.2f} | {h['max']:.1f} | {h['n']} | {sh['mean']:.1f} | {sh['p95']:.1f} | {sh['n']} |")
    lines += ["", "Budgets: fast ≤ 5 ms mean, full ≤ 100 ms mean per SELECTING_HAND decision.", "",
              "## Draw-order invariance", ""]
    for p, s in summary.items():
        if p != "greedy":
            lines.append(f"- {p}: {s['inv_checked']} sampled states, `game.deck` permuted → identical decision in "
                         f"{s['inv_checked'] - s['inv_failed']} ({s['inv_failed']} mismatches)")
    lines += ["", "## MLB match (EVPlayer fast vs EVPlayer fast, full MLBMatch)", ""]
    if mlb:
        lines.append(f"- seed {mlb['seed']}: done={mlb.get('done')} winner={mlb.get('winner')} final antes {mlb.get('final_ante')} "
                     f"lives {mlb.get('lives')} steps {mlb.get('steps')} in {mlb.get('seconds', 0):.1f} s; Nemeses (ante, loser, s0, s1): {mlb.get('pvp_log')}")
    lines += ["", "## Per-seed (furthest ante/blind, $)", "", "| seed | " + " | ".join(summary) + " |", "|---|" + "---|" * len(summary)]
    seeds = sorted({r["seed"] for rows in per_seed.values() for r in rows})
    idx = {p: {r["seed"]: r for r in rows} for p, rows in per_seed.items()}
    for sd in seeds:
        cells = []
        for p in summary:
            r = idx[p].get(sd)
            cells.append(f"{r['furthest_ante']} {r['furthest_blind']} ${r['final_money']}" if r else "—")
        lines.append(f"| {sd} | " + " | ".join(cells) + " |")
    path_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    slim = {p: [{k: v for k, v in r.items() if k not in ("hand_ms", "shop_ms", "other_ms")} for r in rows]
            for p, rows in per_seed.items()}
    path_json.write_text(json.dumps({"meta": meta, "summary": summary, "deltas": deltas, "mlb": mlb, "per_seed": slim},
                                    indent=1, default=float), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=0, help="number of seeds (0 = all 126)")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--players", default="fast,full,greedy")
    ap.add_argument("--inv-checks", type=int, default=6, help="draw-order invariance checks per (seed, player)")
    ap.add_argument("--no-mlb", action="store_true")
    ap.add_argument("--out", default=None, help="results stem (default results/ev_player_gate_<date>)")
    args = ap.parse_args(argv)
    seeds = C.DEFAULT_SEEDS[args.offset: args.offset + args.seeds] if args.seeds else C.DEFAULT_SEEDS[args.offset:]
    players = [p.strip() for p in args.players.split(",") if p.strip()]
    for p in players:
        if p not in PLAYERS:
            raise SystemExit(f"unknown player {p!r} (want {PLAYERS})")
    date = _dt.date.today().isoformat()
    stem = Path(args.out) if args.out else (C.RESULTS_DIR / f"ev_player_gate_{date}")
    stem.parent.mkdir(parents=True, exist_ok=True)
    jobs = [(s, p, args.inv_checks) for p in players for s in seeds]
    t0 = time.time()
    results: list = []
    if args.procs > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(args.procs) as pool:
            for i, r in enumerate(pool.imap_unordered(run_seed, jobs, chunksize=1)):
                results.append(r)
                if (i + 1) % 10 == 0 or i + 1 == len(jobs):
                    print(f"  {i + 1}/{len(jobs)} done ({time.time() - t0:.0f} s)", flush=True)
    else:
        for i, job in enumerate(jobs):
            results.append(run_seed(job))
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(jobs)} done ({time.time() - t0:.0f} s)", flush=True)
    per_seed = {p: sorted([r for r in results if r["player"] == p], key=lambda r: r["seed"]) for p in players}
    summary = {p: summarise(per_seed[p]) for p in players}
    deltas = {p: paired(per_seed[p], per_seed["greedy"]) for p in players if p != "greedy" and "greedy" in per_seed}
    mlb = {} if args.no_mlb else run_mlb_match(seeds[0])
    meta = {"date": date, "command": "python ev/gate_ev_player.py " + " ".join(sys.argv[1:]), "n_seeds": len(seeds),
            "offset": args.offset, "procs": args.procs, "players": players, "wall_s": time.time() - t0,
            "cpu_count": os.cpu_count()}
    write_report(stem.with_suffix(".md"), stem.with_suffix(".json"), summary, per_seed, deltas, mlb, meta)
    print(f"\nwrote {stem.with_suffix('.md')} and .json")
    for p, s in summary.items():
        print(f"{p:7s} ante-1 clear {_fmt_ci(s['ante1_clear'], True)}  mean ante {_fmt_ci(s['mean_final_ante'])}  "
              f"hand ms mean {s['hand_ms']['mean']:.2f} p95 {s['hand_ms']['p95']:.2f}  shop ms {s['shop_ms']['mean']:.1f}  "
              f"inv {s['inv_checked'] - s['inv_failed']}/{s['inv_checked']}")
    if mlb:
        print(f"MLB match: done={mlb.get('done')} winner={mlb.get('winner')} antes={mlb.get('final_ante')} {mlb.get('seconds', 0):.0f}s")


if __name__ == "__main__":
    main()
