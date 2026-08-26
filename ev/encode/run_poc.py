"""run_poc.py — drive every registry entry through the harness and print the table
(W-ENCODE-POC, 2026-08-26).

    python ev/encode/run_poc.py                       # everything, default budgets
    python ev/encode/run_poc.py --fast                # skip the rollout mode (~15 s)
    python ev/encode/run_poc.py --only j_cloud_9,j_rocket
    python ev/encode/run_poc.py --workers 4 --traj-seeds 40 --worlds 32 --json out.json

Ops: never more than ``MAX_WORKERS`` (6) processes, no GPU, no writes outside this
directory unless ``--json`` names one.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_EV = _HERE.parent
_ROOT = _EV.parent
for _p in (str(_HERE), str(_EV), str(_ROOT), str(_ROOT / "eval"), str(_ROOT / "agent")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import registry as R          # noqa: E402  (ev/encode/registry.py)
import verify as V            # noqa: E402


# ══════════════════════════════════════════════════════ scenario builders (mode A/B)
# Each is a small family of REAL constructed states.  They vary the one input the
# predictor reads, so a predictor that ignores its input cannot pass by luck.

_HAND = [(14, "Spades"), (2, "Clubs"), (3, "Clubs"), (4, "Clubs"),
         (5, "Diamonds"), (6, "Diamonds"), (7, "Diamonds"), (8, "Hearts")]


def _base(seed="11111111", dollars=None, no_interest=True):
    g = V.in_blind(seed)
    V.set_hand(g, _HAND)
    g.jokers = []
    if dollars is not None:
        g.dollars = dollars
    g.no_interest = no_interest      # isolate the item's row from the interest step
    return g


def cloud9_scenarios(key="j_cloud_9"):
    out = []
    for n in (0, 1, 4, 7):
        def build(n=n):
            g = _base()
            V.set_deck_nines(g, n)
            return g
        out.append(V.RoundScenario(
            name=f"nines={n}",
            build=build,
            install=lambda g: V.add_joker(g, "j_cloud_9"),
            summarize=V.summarize,
        ))
    return out


def rocket_scenarios():
    out = []
    for boss, bonus in ((False, 1), (True, 1), (False, 3), (True, 3)):
        def build(boss=boss):
            g = _base()
            if boss:
                V.make_boss_blind(g)
            return g

        def install(g, bonus=bonus):
            j = V.add_joker(g, "j_rocket")
            j.state["bonus"] = bonus

        out.append(V.RoundScenario(
            name=f"boss={int(boss)},bonus={bonus}",
            build=build, install=install,
            summarize=V.summarize,
        ))
    return out


def satellite_scenarios(when: str = "before"):
    """``when='before'`` uses the planets BEFORE the joker is installed — the case a
    mid-run purchase actually produces, and the case the Lua pays for
    (``G.GAME.consumeable_usage`` is global run state, card.lua:1667-1673).
    ``when='after'`` uses them after, which is the only case the engine's per-instance
    ``planets_used`` set can see.  Running both is what turns "the entry is wrong" into
    "the ENGINE is wrong, and here is the exact boundary"."""
    plan = [((), "none"), (("c_mercury",), "1 unique"),
            (("c_mercury", "c_venus", "c_mars"), "3 unique"),
            (("c_mercury", "c_mercury", "c_venus"), "2 unique / 3 uses")]
    out = []
    for keys, label in plan:
        def build(keys=keys, when=when):
            g = _base()
            if when == "before":
                V.use_planets(g, keys)
            return g

        def post(g, keys=keys, when=when):
            if when == "after":
                V.use_planets(g, keys)

        out.append(V.RoundScenario(
            name=f"{label} ({when} purchase)", build=build,
            install=lambda g: V.add_joker(g, "j_satellite"),
            post_install=post,
            summarize=V.summarize,
        ))
    return out


def seed_money_scenarios():
    out = []
    for d in (0, 12, 27, 40, 55, 80):
        def build(d=d):
            return _base(dollars=d, no_interest=False)   # interest is the whole point
        out.append(V.RoundScenario(
            name=f"${d}", build=build,
            install=_install_seed_money,
            summarize=V.summarize,
            # a voucher has no joker hook, so reachability is "the cap really differs"
            reach=lambda a, b: int(a.interest_cap != b.interest_cap),
        ))
    return out


def _install_seed_money(g):
    V._consumables.apply_voucher(g, "v_seed_money")
    g.vouchers.add("v_seed_money")


def hermit_scenarios():
    out = []
    for d in (0, 5, 14, 20, 33, 60):
        def build(d=d):
            g = _base(dollars=d)
            return g
        out.append(V.UseScenario(
            name=f"${d}", build=build, consumable="c_hermit",
            summarize=V.summarize,
        ))
    return out


def joker_doublecount_scenarios():
    """The double-count control: the plain Joker is a +4 MULT joker.  Its marginal
    end-of-round dollars are $0 with everything else active, so an entry that prices it as
    $4/round must be rejected.  Nothing about this scenario is special — that is the point:
    the SAME measurement that accepts Cloud 9 rejects this."""
    out = []
    for n in range(3):
        out.append(V.RoundScenario(
            name=f"rep{n}", build=lambda: _base(),
            install=lambda g: V.add_joker(g, "j_joker"),
            summarize=V.summarize,
        ))
    return out


ROUND_SCENARIOS = {
    "j_cloud_9": cloud9_scenarios,
    "j_cloud_9__x3": cloud9_scenarios,
    "j_rocket": rocket_scenarios,
    # both halves: the Lua-faithful "planets used before the purchase" family AND the
    # "used after" family the engine can see.  The entry is scored on the union.
    "j_satellite": lambda: satellite_scenarios("before") + satellite_scenarios("after"),
    "v_seed_money": seed_money_scenarios,
    "j_joker__doublecount": joker_doublecount_scenarios,
}
USE_SCENARIOS = {"c_hermit": hermit_scenarios}
TRAJ_HANDS = {"j_ride_the_bus": 12, "j_green_joker": 12, "j_ice_cream": 12}
#: mode-D targets: one econ joker (full three-number treatment) and the voucher
#: (direction + empirical buy value only — its payout is an interest row, not a hook).
ROLLOUT_SPEC = {
    "j_cloud_9": {"install": "joker", "stop_ante": 4},
    "v_seed_money": {"install": "voucher", "stop_ante": 5},
}


# ══════════════════════════════════════════════════════════════════════════ driver

def run(args) -> dict:
    seeds = _seeds(args.traj_seeds)
    want = set(args.only.split(",")) if args.only else None
    out: list = []
    fallbacks: list = []
    t0 = time.perf_counter()

    for entry in R.entries(include_controls=True):
        if want and entry.key not in want:
            continue
        for mode in entry.modes:
            if mode == "rollout_paired" and (args.fast or entry.key not in ROLLOUT_SPEC):
                continue
            m = _measure(entry, mode, args, seeds)
            if m is None:
                continue
            print(m.row(), flush=True)
            out.append(m)
            # A rejected entry needs a fallback because its closed form is wrong; an
            # UNSCORED row needs one because the entry never made a claim in the first
            # place.  Both hand back the harness's measured number — that is the design.
            if not m.accept or not m.scored:
                fallbacks.append(V.empirical_fallback(m))

    print(f"\n{len(out)} measurements in {time.perf_counter() - t0:.1f}s")
    _summary(out)
    payload = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "marginal_rule": V.MARGINAL_RULE,
        "band": V.BAND,
        "measurements": [_ser(m) for m in out],
        "empirical_fallbacks": fallbacks,
    }
    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"wrote {args.json}")
    return payload


def _measure(entry, mode, args, seeds):
    if mode == "round_end_paired":
        fn = ROUND_SCENARIOS.get(entry.key)
        return V.measure_round_end(entry, fn()) if fn else None
    if mode == "use_paired":
        fn = USE_SCENARIOS.get(entry.key)
        return V.measure_use(entry, fn()) if fn else None
    if mode == "scaling_trajectory":
        n = TRAJ_HANDS.get(entry.key, 12)
        return V.measure_trajectory(entry, seeds, n, workers=args.workers)
    if mode == "rollout_paired":
        spec = ROLLOUT_SPEC[entry.key]
        per_round = None
        if entry.key == "j_cloud_9":
            # the per-round claim, evaluated on the deck the rollout actually starts from
            summary = V.summarize(V.in_blind(args.rollout_seed))
            per_round = lambda: entry.predict(summary)                 # noqa: E731
        return V.measure_rollout(entry, args.rollout_seed, worlds=args.worlds,
                                 stop_ante=spec["stop_ante"], workers=args.workers,
                                 install=spec["install"], per_round_predict=per_round)
    return None


def _seeds(n: int) -> list:
    try:
        import common as C          # eval/common.py — the 126 ground-truth seeds
        return list(C.DEFAULT_SEEDS)[:n]
    except Exception:
        return [str(10_000_000 + i) for i in range(n)]


def _seeds_for_test(n: int) -> list:
    """The same seed source the driver uses, exposed for ``ev/encode/tests``."""
    return _seeds(n)


def _summary(ms) -> None:
    real = [m for m in ms if not R.ALL_ENTRIES[m.key].expect_reject]
    ctrl = [m for m in ms if R.ALL_ENTRIES[m.key].expect_reject]
    print(f"real entries : {sum(1 for m in real if m.accept)}/{len(real)} accepted")
    print(f"controls     : {sum(1 for m in ctrl if not m.accept)}/{len(ctrl)} correctly REJECTED"
          + ("" if all(not m.accept for m in ctrl) else "   <-- HARNESS FAILURE"))


def _ser(m) -> dict:
    return {"key": m.key, "mode": m.mode, "n": m.n, "unit": m.unit,
            "predicted": m.predicted, "measured": m.measured, "ci95": m.ci,
            "fired": m.fired, "accept": m.accept, "reason": m.reason,
            "seconds": round(m.seconds, 2), "extra": m.extra,
            "scenarios": m.scenarios}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=V.DEFAULT_WORKERS)
    ap.add_argument("--traj-seeds", type=int, default=V.DEFAULT_TRAJ_SEEDS)
    ap.add_argument("--worlds", type=int, default=V.DEFAULT_ROLLOUT_WORLDS)
    ap.add_argument("--rollout-seed", default="11111111")
    ap.add_argument("--only", default="")
    ap.add_argument("--fast", action="store_true", help="skip mode D (paired rollouts)")
    ap.add_argument("--json", default="")
    args = ap.parse_args(argv)
    if args.workers > V.MAX_WORKERS:
        print(f"clamping --workers {args.workers} -> {V.MAX_WORKERS} (ops cap)")
        args.workers = V.MAX_WORKERS
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
