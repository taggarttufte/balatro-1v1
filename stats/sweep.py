"""
sweep.py -- the 126-seed decision-statistics sweep (Phase 5 rev 2, W4, gate 3).

Drives a scripted player (``scripts/mlb_match_demo.py::ScriptedPlayer`` -- the module
``eval/common.py::ScriptedPlayer`` itself re-exports, per that file's own docstring)
through ante 8 on each seed, calling ``decide.decision_table`` at EVERY shop visit and pack
open, and aggregates the result by ante: P(hit) of a reroll, mean net EV of the best row, %
of visits where the best row is ``leave``, interest-loss share of true cost, pack-open EV by
pack kind, voucher EV, and the distribution of urgency.

Multiprocess over seeds (``ProcessPoolExecutor``; each worker independently bootstraps the
engine fork -- required on Windows, which uses ``spawn``, not ``fork``: nothing importable
here is inherited from the parent process).

Usage::

    python stats/sweep.py --out results/stats_sweep_2026-08-23.json
    python stats/sweep.py --seeds 11111111,1558AXDL --processes 2 --out /tmp/smoke.json

CAUTION (2026-08-23 lead note): do not run the full 126-seed / high-process-count sweep
while the box is in interactive use -- smoke-test with ``--n-seeds`` / ``--seeds`` and a
small ``--processes`` first. The exact full-sweep command is documented in STATS_NOTES.md.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent           # stats
_MP_ROOT = _HERE.parent                             # repo root
_SCRIPTS_DIR = _MP_ROOT / "scripts"
_GROUND_TRUTH_DIR = _MP_ROOT / "oracle" / "ground_truth"
_RESULTS_DIR = _MP_ROOT / "results"

for _p in (str(_MP_ROOT), str(_HERE), str(_SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _bootstrap  # noqa: E402  (engine fork guard; also puts agent on sys.path)
from _bootstrap import BalatroGame, State  # noqa: E402

import decide  # noqa: E402

DEFAULT_SEEDS: list = sorted(p.stem for p in _GROUND_TRUTH_DIR.glob("*.json"))
MAX_ANTE = 8
MAX_STEPS = 40_000


# ══════════════════════════════════════════════════════════════════ per-seed driver

class _SoloShim:
    __slots__ = ("games",)

    def __init__(self, game):
        self.games = [game]


def _policy_step(game, policy_fn) -> dict:
    acts = game.legal_actions()
    if not acts:
        return {"type": "advance"}
    return policy_fn(_SoloShim(game), 0, acts)


def _summarize_row(r) -> dict:
    return {"kind": r.kind, "p_hit": r.p_hit, "hit_value": r.hit_value, "cost": r.cost,
           "interest_loss": r.interest_loss, "true_cost": r.true_cost, "net_ev": r.net_ev}


def _visit_record(game, rows: list, elapsed_ms: float) -> dict:
    best = rows[0] if rows else None
    reroll = next((r for r in rows if r.kind == "reroll"), None)
    voucher = next((r for r in rows if r.kind == "buy_voucher"), None)
    packs = [r for r in rows if r.kind == "buy_pack"]
    picks = [r for r in rows if r.kind == "pick"]
    return {
        "ante": game.ante,
        "blind_idx": game.blind_idx,
        "visit_kind": "pack" if game.state == State.BOOSTER_OPEN else "shop",
        "n_rows": len(rows),
        "elapsed_ms": elapsed_ms,
        "urgency": best.urgency if best else None,
        "best_kind": best.kind if best else None,
        "best_net_ev": best.net_ev if best else None,
        "best_is_leave": bool(best and best.kind in ("leave", "skip_pack")),
        "true_cost_of_best": best.true_cost if best else None,
        "interest_loss_of_best": best.interest_loss if best else None,
        "reroll": _summarize_row(reroll) if reroll else None,
        "voucher": _summarize_row(voucher) if voucher else None,
        "packs": [dict(_summarize_row(r), pack_kind=r.details.get("pack_kind", "")) for r in packs],
        "best_pick": _summarize_row(max(picks, key=lambda r: r.net_ev)) if picks else None,
    }


def run_one_seed(seed: str, player_kwargs: dict, max_ante: int = MAX_ANTE,
                 max_steps: int = MAX_STEPS) -> dict:
    """Drive one seed to ``ante > max_ante`` or GAME_OVER; call ``decision_table`` at every
    first-observed SHOP / BOOSTER_OPEN state. Returns ``{"seed", "records": [...], "error"}``
    (``error`` set and ``records`` possibly partial if the run raised -- a sweep over 126
    seeds should not die on one bad seed)."""
    import mlb_match_demo as D   # local: must be (re-)imported inside the worker on Windows spawn

    records: list[dict] = []
    error: Optional[str] = None
    try:
        spec = D.ScriptedPlayer(name=f"sweep-{seed}", **player_kwargs)
        policy = D.make_policy(spec)
        game = BalatroGame(seed=seed, deck_key="b_red", stake=1, ruleset="mlb")
        last_state = None
        steps = 0
        while game.state != State.GAME_OVER and game.ante <= max_ante and steps < max_steps:
            if game.state != last_state and game.state in (State.SHOP, State.BOOSTER_OPEN):
                t0 = time.perf_counter()
                rows = decide.decision_table(game)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                records.append(_visit_record(game, rows, dt_ms))
            last_state = game.state
            game.step(_policy_step(game, policy))
            steps += 1
    except Exception as exc:   # noqa: BLE001 - one bad seed must not kill the sweep
        error = f"{type(exc).__name__}: {exc}"
    return {"seed": seed, "records": records, "error": error}


# ══════════════════════════════════════════════════════════════════ aggregation

def _mean(xs: list) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def aggregate(all_results: list[dict]) -> dict:
    """By-ante tables + a few run-wide numbers. Every seed's records are pooled per ante
    (a visit, not a seed, is the unit of analysis -- an ante is visited up to 3x/seed)."""
    by_ante: dict[int, dict] = {}
    urgencies: list[float] = []
    n_errors = 0
    n_seeds = len(all_results)
    n_visits = 0
    elapsed_all: list[float] = []

    for res in all_results:
        if res.get("error"):
            n_errors += 1
        for rec in res["records"]:
            n_visits += 1
            elapsed_all.append(rec["elapsed_ms"])
            a = rec["ante"]
            bucket = by_ante.setdefault(a, {
                "n_visits": 0, "n_shop_visits": 0, "n_pack_visits": 0,
                "reroll_p_hit": [], "reroll_net_ev": [], "best_net_ev": [], "leave_flags": [],
                "true_cost": [], "interest_loss": [], "urgency": [], "voucher_net_ev": [],
                "pack_net_ev_by_kind": {}, "elapsed_ms": [],
            })
            bucket["n_visits"] += 1
            bucket["elapsed_ms"].append(rec["elapsed_ms"])
            if rec["visit_kind"] == "shop":
                bucket["n_shop_visits"] += 1
            else:
                bucket["n_pack_visits"] += 1
            if rec["urgency"] is not None:
                bucket["urgency"].append(rec["urgency"])
                urgencies.append(rec["urgency"])
            if rec["best_net_ev"] is not None:
                bucket["best_net_ev"].append(rec["best_net_ev"])
            bucket["leave_flags"].append(1.0 if rec["best_is_leave"] else 0.0)
            if rec["true_cost_of_best"]:
                bucket["true_cost"].append(rec["true_cost_of_best"])
                bucket["interest_loss"].append(rec["interest_loss_of_best"] or 0.0)
            if rec["reroll"]:
                bucket["reroll_p_hit"].append(rec["reroll"]["p_hit"])
                bucket["reroll_net_ev"].append(rec["reroll"]["net_ev"])
            if rec["voucher"]:
                bucket["voucher_net_ev"].append(rec["voucher"]["net_ev"])
            for p in rec["packs"]:
                bucket["pack_net_ev_by_kind"].setdefault(p["pack_kind"], []).append(p["net_ev"])

    table = {}
    for a, b in sorted(by_ante.items()):
        true_cost_sum = sum(b["true_cost"]) if b["true_cost"] else 0.0
        int_loss_sum = sum(b["interest_loss"]) if b["interest_loss"] else 0.0
        table[a] = {
            "n_visits": b["n_visits"], "n_shop_visits": b["n_shop_visits"],
            "n_pack_visits": b["n_pack_visits"],
            "reroll_p_hit_mean": _mean(b["reroll_p_hit"]),
            "reroll_net_ev_mean": _mean(b["reroll_net_ev"]),
            "best_net_ev_mean": _mean(b["best_net_ev"]),
            "pct_best_is_leave": _mean(b["leave_flags"]),
            "interest_loss_share_of_true_cost": (int_loss_sum / true_cost_sum) if true_cost_sum else None,
            "urgency_mean": _mean(b["urgency"]),
            "urgency_p10": (sorted(b["urgency"])[max(0, int(0.10 * len(b["urgency"])) - 1)]
                           if b["urgency"] else None),
            "urgency_p90": (sorted(b["urgency"])[min(len(b["urgency"]) - 1, int(0.90 * len(b["urgency"])))]
                           if b["urgency"] else None),
            "voucher_net_ev_mean": _mean(b["voucher_net_ev"]),
            "pack_net_ev_by_kind": {k: _mean(v) for k, v in b["pack_net_ev_by_kind"].items()},
            "decide_ms_mean": _mean(b["elapsed_ms"]),
        }

    elapsed_sorted = sorted(elapsed_all)
    p95 = elapsed_sorted[int(0.95 * (len(elapsed_sorted) - 1))] if elapsed_sorted else None
    return {
        "n_seeds": n_seeds, "n_errors": n_errors, "n_visits": n_visits,
        "decide_ms_mean_overall": _mean(elapsed_all), "decide_ms_p95_overall": p95,
        "urgency_mean_overall": _mean(urgencies),
        "by_ante": table,
    }


# ══════════════════════════════════════════════════════════════════ CLI

_PLAYER_KWARGS = dict(hand="greedy", rerolls_per_visit=1, buy_slot0=False,
                     open_pack_slot=None, buy_voucher=False)


def _parse_player_spec(spec: str) -> dict:
    """``"hand=greedy,reroll=1,buy=0"`` -> ``ScriptedPlayer`` kwargs (same field aliases as
    ``eval/common.py::parse_scripted_spec``, reimplemented here to avoid importing the
    rest of that module's bootstrap for a sweep worker)."""
    import mlb_match_demo as D
    aliases = {"reroll": "rerolls_per_visit", "rerolls": "rerolls_per_visit",
              "buy": "buy_slot0", "pack": "open_pack_slot", "voucher": "buy_voucher",
              "weak_from": "weak_from_ante"}
    fields = set(D.ScriptedPlayer.__dataclass_fields__)
    kwargs = dict(_PLAYER_KWARGS)
    if not spec:
        return kwargs
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        k, _, v = part.partition("=")
        k = aliases.get(k.strip(), k.strip())
        if k not in fields:
            raise ValueError(f"unknown ScriptedPlayer field {k!r} in spec {spec!r}")
        default = getattr(D.ScriptedPlayer, k)
        if isinstance(default, bool):
            kwargs[k] = v.strip().lower() in ("1", "true", "t", "yes", "y", "on")
        elif k == "open_pack_slot":
            kwargs[k] = None if v.strip().lower() in ("none", "") else int(v)
        elif isinstance(default, int):
            kwargs[k] = int(v)
        else:
            kwargs[k] = v
    return kwargs


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="output .json path (a sibling .md summary is also written)")
    ap.add_argument("--seeds", default=None, help="comma-separated explicit seed list (overrides --n-seeds)")
    ap.add_argument("--n-seeds", type=int, default=None,
                    help="use the first N of the 126 ground-truth seeds (default: all 126)")
    ap.add_argument("--processes", type=int, default=4, help="worker processes (default 4; the "
                    "full 126-seed run should use more only when the box is idle -- see STATS_NOTES.md)")
    ap.add_argument("--max-ante", type=int, default=MAX_ANTE)
    ap.add_argument("--player", default="hand=greedy,reroll=1,buy=0",
                    help="ScriptedPlayer spec, e.g. 'hand=greedy,reroll=1,buy=0'")
    args = ap.parse_args(argv)

    if args.seeds:
        seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    elif args.n_seeds:
        seeds = DEFAULT_SEEDS[: args.n_seeds]
    else:
        seeds = DEFAULT_SEEDS

    player_kwargs = _parse_player_spec(args.player)

    t_start = time.time()
    results: list[dict] = []
    if args.processes <= 1:
        for s in seeds:
            results.append(run_one_seed(s, player_kwargs, args.max_ante))
    else:
        with ProcessPoolExecutor(max_workers=args.processes) as ex:
            futs = {ex.submit(run_one_seed, s, player_kwargs, args.max_ante): s for s in seeds}
            for fut in as_completed(futs):
                results.append(fut.result())
    wall_s = time.time() - t_start

    results.sort(key=lambda r: r["seed"])
    agg = aggregate(results)
    payload = {
        "meta": {
            "n_seeds_requested": len(seeds), "processes": args.processes,
            "player_spec": args.player, "max_ante": args.max_ante, "wall_seconds": wall_s,
        },
        "aggregate": agg,
        "per_seed_errors": [{"seed": r["seed"], "error": r["error"]} for r in results if r["error"]],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_path = out_path.with_suffix(".md")
    md_path.write_text(_render_markdown(payload), encoding="utf-8")

    print(f"wrote {out_path} and {md_path}  ({len(seeds)} seeds, {agg['n_visits']} visits, "
         f"{wall_s:.1f}s wall, {agg['n_errors']} errors)")
    return 0


def _render_markdown(payload: dict) -> str:
    meta, agg = payload["meta"], payload["aggregate"]
    lines = [
        "# Stats sweep", "",
        f"- seeds: {meta['n_seeds_requested']}  processes: {meta['processes']}  "
        f"player: `{meta['player_spec']}`  max_ante: {meta['max_ante']}  wall: {meta['wall_seconds']:.1f}s",
        f"- visits: {agg['n_visits']}  errors: {agg['n_errors']}  "
        f"decide_ms mean/p95: {agg['decide_ms_mean_overall']:.3f} / {agg['decide_ms_p95_overall']:.3f}  "
        f"urgency mean: {agg['urgency_mean_overall']:.3f}" if agg['n_visits'] else "- no visits recorded",
        "",
        "| ante | visits | reroll P(hit) | reroll net_ev | best net_ev | % leave best | "
        "int.loss/true_cost | urgency mean | voucher net_ev | decide ms |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for a, b in sorted(agg["by_ante"].items()):
        def f(x, spec=".2f"):
            return format(x, spec) if x is not None else "-"
        lines.append(
            f"| {a} | {b['n_visits']} | {f(b['reroll_p_hit_mean'])} | {f(b['reroll_net_ev_mean'])} | "
            f"{f(b['best_net_ev_mean'])} | {f(b['pct_best_is_leave'], '.0%')} | "
            f"{f(b['interest_loss_share_of_true_cost'], '.0%')} | {f(b['urgency_mean'])} | "
            f"{f(b['voucher_net_ev_mean'])} | {f(b['decide_ms_mean'], '.3f')} |"
        )
    lines += ["", "## Pack net EV by kind, by ante", ""]
    for a, b in sorted(agg["by_ante"].items()):
        if b["pack_net_ev_by_kind"]:
            parts = ", ".join(f"{k}={v:.1f}" for k, v in sorted(b["pack_net_ev_by_kind"].items()))
            lines.append(f"- ante {a}: {parts}")
    if payload["per_seed_errors"]:
        lines += ["", "## Seed errors", ""] + [f"- {e['seed']}: {e['error']}" for e in payload["per_seed_errors"]]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
