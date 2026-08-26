"""
pvp_protocol_h2h.py — paired head-to-head for the PvP turn protocol (W-PVP, 2026-08-26).

``ev/h2h.py`` cannot express either arm of this measurement: its ``build_player`` parses
``ev:fast+full+stats`` tokens only (so it cannot vary a ``HandConfig``) and it constructs
``MLBMatch`` without a ``pvp_protocol``.  Rather than edit a driver other workstreams are
using tonight, this is the same design in its own file — both seatings per seed, a spawn
pool, ``eval/common.bootstrap_ci`` — exactly as ``ev/scripts/extract_dev_slice.py h2h``
does for the extraction layer (EXTRACT_NOTES.md §9.1).

Two experiments:

    # (a) the protocol itself: same player both sides, protocol ON vs OFF, same seeds
    python ev/scripts/pvp_protocol_h2h.py protocol --n-seeds 30 --procs 6 --max-steps 4000

    # (b) attribution: protocol ON for BOTH sides, level-1 objective vs level-0
    python ev/scripts/pvp_protocol_h2h.py level --n-seeds 30 --procs 6 --max-steps 4000

``protocol`` is not a head-to-head in the usual sense — both arms are self-play, so the
win rate is 50% by construction.  What it measures is BEHAVIOUR: leader passes per match,
Glass cards still in the deck at the end, dollars extracted at lost Nemeses, hands unspent
when a Nemesis ends, and the early-end cut rate.  It also reports the ON-arm win-rate
symmetry (must sit at ~50% across seatings) as a sanity check that the protocol did not
introduce a seat bias.

``level`` IS a head-to-head: A = level-1 (react to the revealed score), B = level-0 (the
symmetric atoms), both under ``pvp_protocol="trailer_compelled"``, both with PASS and the
decided-race extraction gate on, so the only difference is the objective.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
import zlib
from dataclasses import replace
from pathlib import Path

_HERE = Path(__file__).resolve().parent            # ev/scripts
_EV = _HERE.parent                                  # ev
_ROOT = _EV.parent
for _p in (str(_ROOT), str(_EV), str(_ROOT / "eval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _bootstrap  # noqa: E402,F401
from _bootstrap import MLBMatch, State  # noqa: E402

import common as C  # noqa: E402  (eval/common.py)

__all__ = ["run", "summarise", "write_report"]

PROTO = "trailer_compelled"


def _stable_seed(text: str) -> int:
    return zlib.crc32(text.encode("utf-8"))


def _hand_cfg(arm: str):
    """``arm`` -> the ``HandConfig`` for that side.  Imported lazily so a spawn worker
    builds it in its own process."""
    import player as P
    if arm == "off":
        return P.DEFAULT_HAND_CONFIG          # the pre-W-PVP objective, bit for bit
    if arm == "on":
        return P.protocol_hand_cfg()          # level-1 + PASS + decided-race extraction
    if arm == "level0":
        return P.protocol_hand_cfg(level1=False)
    raise ValueError(f"unknown arm {arm!r}")


def _policy(arm: str, seed: int):
    import player as P
    ev = P.EVPlayer(budget="fast", seed=int(seed), epsilon=0.0, hand_cfg=_hand_cfg(arm),
                    name=f"ev:fast[{arm}]")
    return P.adapt_match_player(ev), ev


# ───────────────────────────────────────────────────────── behaviour metrics

def _glass_alive(game) -> int:
    return sum(1 for c in game.full_deck if getattr(c, "enhancement", "") == "Glass")


def _match_record(m: MLBMatch, a_idx: int, b_idx: int, dollars_at_pvp: dict,
                  nem_discards: dict) -> dict:
    g = m.games
    nem_losses = [sum(1 for (_a, loser, _s0, _s1) in m.pvp_log if loser == p) for p in (0, 1)]
    early = sum(1 for d in m.pvp_detail if d[6])
    n = len(m.pvp_detail) or 1
    # hands actually played per Nemesis (pvp_detail[4], [5]) — the conservation measure:
    # a leader who waits ends the blind having spent fewer hands.
    hands_played = [sum(d[4 + p] for d in m.pvp_detail) / n for p in (0, 1)]
    return {
        "steps": m.steps, "done": bool(m.done),
        "a_win": (bool(m.winner == a_idx) if m.done else None),
        "lives_a": g[a_idx].lives, "lives_b": g[b_idx].lives,
        "lives_margin_a": g[a_idx].lives - g[b_idx].lives,
        "final_ante_a": g[a_idx].ante, "final_ante_b": g[b_idx].ante,
        "final_money_a": g[a_idx].dollars, "final_money_b": g[b_idx].dollars,
        "nem_total": len(m.pvp_log),
        "nem_losses_a": nem_losses[a_idx], "nem_losses_b": nem_losses[b_idx],
        "early_end_cuts": early,
        "passes_a": m.pvp_passes[a_idx], "passes_b": m.pvp_passes[b_idx],
        "hands_per_nemesis_a": hands_played[a_idx], "hands_per_nemesis_b": hands_played[b_idx],
        "nem_discards_a": nem_discards[a_idx], "nem_discards_b": nem_discards[b_idx],
        "glass_alive_a": _glass_alive(g[a_idx]), "glass_alive_b": _glass_alive(g[b_idx]),
        "lost_nemesis_dollars_a": dollars_at_pvp[a_idx],
        "lost_nemesis_dollars_b": dollars_at_pvp[b_idx],
    }


def _play(seed, arm_a: str, arm_b: str, protocol: str, seed_a: int, seed_b: int,
          lives: int, max_steps: int, deck_key: str, stake) -> list:
    """One seed, BOTH seatings."""
    trials = []
    for seating in (0, 1):
        pol_a, obj_a = _policy(arm_a, seed_a)
        pol_b, obj_b = _policy(arm_b, seed_b)
        if seating == 0:
            policies, a_idx, b_idx = [pol_a, pol_b], 0, 1
        else:
            policies, a_idx, b_idx = [pol_b, pol_a], 1, 0
        obj_a.reset()
        obj_b.reset()
        m = MLBMatch(seed=seed, deck_key=deck_key, stake=stake, lives=lives,
                     pvp_protocol=protocol)
        # money banked DURING a Nemesis that this player went on to lose: the whole point
        # of the decided-lost pivot.  Sampled at the blind's start and at its resolution.
        dollars_at_pvp = {0: 0, 1: 0}
        nem_discards = {0: 0, 1: 0}
        start_money = {0: None, 1: None}
        seen_nem = 0
        t0 = time.perf_counter()
        while not m.done and m.steps < max_steps:
            p = m.current_player()
            if p is None:
                break
            if m.pvp_active and start_money[0] is None:
                start_money = {q: m.games[q].dollars for q in (0, 1)}
            act = policies[p](m, p, m.legal_actions(p))
            if m.pvp_active and act.get("type") == "discard":
                nem_discards[p] += 1
            m.step(p, act)
            if len(m.pvp_log) > seen_nem:
                _ante, loser, _s0, _s1 = m.pvp_log[-1]
                if loser is not None and start_money[loser] is not None:
                    dollars_at_pvp[loser] += m.games[loser].dollars - start_money[loser]
                seen_nem = len(m.pvp_log)
                start_money = {0: None, 1: None}
        rec = _match_record(m, a_idx, b_idx, dollars_at_pvp, nem_discards)
        rec.update({"seed": str(seed), "seating": seating, "seconds": time.perf_counter() - t0})
        trials.append(rec)
    return trials


def _worker(job) -> list:
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    return _play(*job)


def run(arm_a: str, arm_b: str, protocol: str, seeds, *, procs: int = 1, lives: int = 4,
        max_steps: int = 4000, deck_key: str = "b_red", stake=1, seed_base: int = 0,
        progress: bool = True) -> dict:
    seeds = list(seeds)
    jobs = [(sd, arm_a, arm_b, protocol, _stable_seed(f"{seed_base}:{sd}:A"),
             _stable_seed(f"{seed_base}:{sd}:B"), lives, max_steps, deck_key, stake)
            for sd in seeds]
    t0 = time.perf_counter()
    trials: list = []
    if procs and procs > 1:
        import multiprocessing as mpc
        with mpc.get_context("spawn").Pool(procs) as pool:
            for i, batch in enumerate(pool.imap_unordered(_worker, jobs, chunksize=1)):
                trials.extend(batch)
                if progress:
                    print(f"  {i + 1}/{len(jobs)} seeds ({time.perf_counter() - t0:.0f}s)",
                          flush=True)
    else:
        for i, job in enumerate(jobs):
            trials.extend(_worker(job))
            if progress:
                print(f"  {i + 1}/{len(jobs)} seeds ({time.perf_counter() - t0:.0f}s)", flush=True)
    return {"arm_a": arm_a, "arm_b": arm_b, "protocol": protocol, "seeds": [str(s) for s in seeds],
            "n_seeds": len(seeds), "lives": lives, "max_steps": max_steps, "deck_key": deck_key,
            "stake": stake, "procs": procs, "seed_base": seed_base,
            "wall_clock_s": time.perf_counter() - t0, "trials": trials,
            "summary": summarise(trials)}


def summarise(trials: list) -> dict:
    def mean(key):
        xs = [t[key] for t in trials]
        return (sum(xs) / len(xs)) if xs else float("nan")

    decided = [t for t in trials if t["a_win"] is not None]
    flags = [1.0 if t["a_win"] else 0.0 for t in decided]
    ci = C.bootstrap_ci(flags) if flags else {"point": float("nan"), "lo": float("nan"),
                                              "hi": float("nan"), "n": 0}
    per_seating = {}
    for s in (0, 1):
        sub = [t for t in decided if t["seating"] == s]
        per_seating[s] = (sum(1 for t in sub if t["a_win"]) / len(sub)) if sub else float("nan")
    nem = sum(t["nem_total"] for t in trials)
    return {
        "n_trials": len(trials), "n_decided": len(decided),
        "a_wins": sum(flags), "undecided": len(trials) - len(decided),
        "win_rate_a": ci, "win_rate_a_by_seating": per_seating,
        "nemeses_total": nem,
        "mean_passes_per_match": mean("passes_a") + mean("passes_b"),
        "mean_passes_a": mean("passes_a"), "mean_passes_b": mean("passes_b"),
        "early_end_cuts_per_match": mean("early_end_cuts"),
        "early_end_rate_per_nemesis": (sum(t["early_end_cuts"] for t in trials) / nem) if nem else float("nan"),
        "mean_hands_per_nemesis_a": mean("hands_per_nemesis_a"),
        "mean_hands_per_nemesis_b": mean("hands_per_nemesis_b"),
        "mean_nemesis_discards_a": mean("nem_discards_a"),
        "mean_nemesis_discards_b": mean("nem_discards_b"),
        "mean_glass_alive_a": mean("glass_alive_a"), "mean_glass_alive_b": mean("glass_alive_b"),
        "mean_lost_nemesis_dollars_a": mean("lost_nemesis_dollars_a"),
        "mean_lost_nemesis_dollars_b": mean("lost_nemesis_dollars_b"),
        "mean_final_ante_a": mean("final_ante_a"), "mean_final_ante_b": mean("final_ante_b"),
        "mean_final_money_a": mean("final_money_a"), "mean_final_money_b": mean("final_money_b"),
        "mean_lives_margin_a": mean("lives_margin_a"),
        "mean_steps": mean("steps"), "mean_seconds_per_match": mean("seconds"),
    }


_ROWS = [
    ("matches (both seatings)", "n_trials", "{:.0f}"),
    ("decided / undecided", None, None),
    ("A win rate", None, None),
    ("A win rate by seating (0 / 1)", None, None),
    ("Nemeses resolved (total)", "nemeses_total", "{:.0f}"),
    ("leader passes / match", "mean_passes_per_match", "{:.2f}"),
    ("early-end cuts / match", "early_end_cuts_per_match", "{:.2f}"),
    ("early-end rate / Nemesis", "early_end_rate_per_nemesis", "{:.3f}"),
    ("hands played per Nemesis (A / B)", None, None),
    ("discards taken at Nemeses / match (A / B)", None, None),
    ("Glass cards alive at the end (A / B)", None, None),
    ("$ banked in a LOST Nemesis (A / B)", None, None),
    ("mean final ante (A / B)", None, None),
    ("mean final $ (A / B)", None, None),
    ("mean lives margin (A - B)", "mean_lives_margin_a", "{:+.3f}"),
    ("mean steps / match", "mean_steps", "{:.0f}"),
    ("seconds / match", "mean_seconds_per_match", "{:.2f}"),
]


def _table(res: dict) -> list:
    s = res["summary"]
    w = s["win_rate_a"]
    out = []
    for label, key, fmt in _ROWS:
        if key is not None:
            out.append(f"| {label} | {fmt.format(s[key])} |")
        elif label.startswith("decided"):
            out.append(f"| {label} | {s['n_decided']} / {s['undecided']} |")
        elif label == "A win rate":
            out.append(f"| {label} | {w['point']:.3f}  CI [{w['lo']:.3f}, {w['hi']:.3f}] |")
        elif label.startswith("A win rate by"):
            b = s["win_rate_a_by_seating"]
            out.append(f"| {label} | {b[0]:.3f} / {b[1]:.3f} |")
        else:
            stem = {"hands played per Nemesis (A / B)": "mean_hands_per_nemesis",
                    "discards taken at Nemeses / match (A / B)": "mean_nemesis_discards",
                    "Glass cards alive at the end (A / B)": "mean_glass_alive",
                    "$ banked in a LOST Nemesis (A / B)": "mean_lost_nemesis_dollars",
                    "mean final ante (A / B)": "mean_final_ante",
                    "mean final $ (A / B)": "mean_final_money"}[label]
            out.append(f"| {label} | {s[stem + '_a']:.2f} / {s[stem + '_b']:.2f} |")
    return out


def write_report(results: list, path_json: str, path_md: str, title: str) -> None:
    Path(path_json).parent.mkdir(parents=True, exist_ok=True)
    Path(path_json).write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
    lines = [f"# {title}", "",
             f"_{_dt.datetime.now().isoformat(timespec='seconds')}_", ""]
    for res in results:
        lines += [f"## {res['label']}", "",
                  f"- A = `{res['arm_a']}` · B = `{res['arm_b']}` · "
                  f"pvp_protocol = `{res['protocol']}`",
                  f"- seeds {res['n_seeds']} x 2 seatings · lives {res['lives']} · "
                  f"max_steps {res['max_steps']} · procs {res['procs']} · "
                  f"wall {res['wall_clock_s']:.0f}s", "",
                  "| metric | value |", "|---|---|"] + _table(res) + [""]
    Path(path_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════ CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("protocol", "level"))
    ap.add_argument("--n-seeds", type=int, default=30)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--lives", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--deck-key", default="b_red")
    ap.add_argument("--stake", default=1)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if args.procs > 6:
        raise SystemExit("--procs is capped at 6 tonight (a pair campaign owns the box)")
    seeds = ([s.strip() for s in args.seeds.split(",") if s.strip()] if args.seeds else
             C.DEFAULT_SEEDS[args.seed_offset: args.seed_offset + args.n_seeds])
    stake = int(args.stake) if str(args.stake).lstrip("-").isdigit() else args.stake
    kw = dict(procs=args.procs, lives=args.lives, max_steps=args.max_steps,
              deck_key=args.deck_key, stake=stake, seed_base=args.seed_base,
              progress=not args.quiet)

    results = []
    if args.mode == "protocol":
        for label, arm, proto in (("protocol ON (both sides)", "on", PROTO),
                                  ("protocol OFF (both sides, the pre-W-PVP player)",
                                   "off", "canonical")):
            print(f"\n=== {label} ===", flush=True)
            r = run(arm, arm, proto, seeds, **kw)
            r["label"] = label
            results.append(r)
        title = "PvP turn protocol: ON vs OFF (paired seeds, self-play both arms)"
        stem = "pvp_protocol_on_vs_off"
    else:
        print("\n=== level-1 vs level-0, protocol ON both sides ===", flush=True)
        r = run("on", "level0", PROTO, seeds, **kw)
        r["label"] = "A = level-1 objective, B = level-0 objective (protocol ON both sides)"
        results.append(r)
        title = "PvP objective: level-1 (react to the revealed score) vs level-0"
        stem = "pvp_level1_vs_level0"

    date = _dt.date.today().isoformat()
    out_json = args.out_json or str(C.RESULTS_DIR / f"{stem}_{date}.json")
    out_md = args.out_md or str(C.RESULTS_DIR / f"{stem}_{date}.md")
    write_report(results, out_json, out_md, title)
    for res in results:
        print(f"\n--- {res['label']} ---")
        for line in _table(res):
            print("   " + line.strip("| ").replace(" | ", ": "))
    print(f"\nwrote {out_json} and {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
