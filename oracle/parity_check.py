"""
Parity harness: replay seeds through the Python RNG port (mp.rng.generate) and diff against
the oracle ground truth in mp/oracle/ground_truth/*.json.

    python -m mp.oracle.parity_check --validate-only                 # schema-check the JSONs (works today)
    python -m mp.oracle.parity_check --seeds ALEEB,OG4YQPSI --antes 1-3
    python -m mp.oracle.parity_check --antes 1-8 --queue-depth 20 --fields boss,voucher,tags,shop_queue,packs
    python -m mp.oracle.parity_check --list                          # seeds available + cross-check status

Run from the repo root (mp/ is a namespace package; mp/oracle has __init__.py).

Contract expected from mp/rng/generate.py (first match wins, all called with keyword args):
    generate_ground_truth(seed=, deck=, stake=, antes=, shop_queue_depth=) -> dict
    generate_run(...)  |  analyze_seed(...)  |  generate(...)
The returned dict must follow mp/oracle/schema.md at least for:
    antes[str(n)].boss.key, .voucher.key, .tags.small.key, .tags.big.key,
    antes[str(n)].shop_queue[i] -> {key, edition, stickers?}
    antes[str(n)].shops[j].packs[k] -> {key, cards:[{key, edition?, enhancement?, seal?}]}
Missing sections are reported as "not produced" rather than as mismatches, so a partial port
(e.g. boss/voucher/tags only) can be checked incrementally.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
GT_DIR = os.path.join(HERE, "ground_truth")
sys.path.insert(0, HERE)
import keymap as K  # noqa: E402

FIELDS_DEFAULT = ["boss", "voucher", "tags", "shop_queue", "packs"]
KEY_RE = {
    "boss": re.compile(r"^bl_[a-z_]+$"),
    "voucher": re.compile(r"^v_[a-z_]+$"),
    "tag": re.compile(r"^tag_[a-z_]+$"),
    "pack": re.compile(r"^p_(arcana|celestial|standard|buffoon|spectral)_(normal|jumbo|mega)$"),
}
VALID_ITEM_KEYS = set(K.JOKERS.values()) | set(K.TAROTS.values()) | set(K.PLANETS.values()) | set(K.SPECTRALS.values())
VALID_BASE = {f"{s}_{r}" for s in "HCDS" for r in "23456789TJQKA"}
VALID_ED = {None, "e_foil", "e_holo", "e_polychrome", "e_negative"}
VALID_ENH = {None, "m_bonus", "m_mult", "m_wild", "m_glass", "m_steel", "m_stone", "m_gold", "m_lucky"}
VALID_SEAL = {None, "Red", "Blue", "Gold", "Purple"}


# ----------------------------------------------------------------------------- loading

def load_ground_truth(seeds: list[str] | None = None) -> dict[str, dict]:
    out = {}
    for f in sorted(os.listdir(GT_DIR)):
        if not f.endswith(".json"):
            continue
        seed = f[:-5]
        if seeds and seed not in seeds:
            continue
        with open(os.path.join(GT_DIR, f), encoding="utf-8") as fh:
            out[seed] = json.load(fh)
    return out


def parse_antes(spec: str) -> list[int]:
    if "-" in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",") if x]


# ----------------------------------------------------------------------------- schema validation

def validate(gt: dict) -> list[str]:
    errs = []
    for req in ("schema_version", "seed", "game_version", "deck", "stake", "source", "retrieved", "antes"):
        if req not in gt:
            errs.append(f"missing top-level field {req}")
    if errs:
        return errs
    if gt["deck"] not in K.DECKS.values():
        errs.append(f"unknown deck key {gt['deck']}")
    if gt["stake"] not in K.STAKES.values():
        errs.append(f"unknown stake key {gt['stake']}")
    antes = gt["antes"]
    nums = sorted(int(a) for a in antes)
    if nums != list(range(1, len(nums) + 1)):
        errs.append(f"antes not contiguous from 1: {nums}")
    for a_str, a in antes.items():
        n = int(a_str)
        p = f"ante{n}"
        for req in ("boss", "voucher", "tags", "shop_queue", "shops"):
            if req not in a:
                errs.append(f"{p}: missing {req}")
        if errs:
            continue
        if not KEY_RE["boss"].match(a["boss"]["key"]) or a["boss"]["key"] not in K.BLINDS.values():
            errs.append(f"{p}.boss bad key {a['boss']['key']}")
        if a["boss"]["key"] in ("bl_small", "bl_big"):
            errs.append(f"{p}.boss is not a boss")
        if a["voucher"]["key"] not in K.VOUCHERS.values():
            errs.append(f"{p}.voucher bad key {a['voucher']['key']}")
        for which in ("small", "big"):
            tk = a["tags"][which]["key"]
            if tk not in K.TAGS.values():
                errs.append(f"{p}.tags.{which} bad key {tk}")
            elif n < K.TAG_MIN_ANTE.get(tk, 1):
                errs.append(f"{p}.tags.{which} {tk} not allowed before ante {K.TAG_MIN_ANTE[tk]}")
        if len(a["shop_queue"]) < 10:
            errs.append(f"{p}.shop_queue shorter than 10 ({len(a['shop_queue'])})")
        for i, it in enumerate(a["shop_queue"]):
            errs.extend(f"{p}.shop_queue[{i}]: {e}" for e in _validate_item(it))
        expect_shops = 2 if n == 1 else 3
        if len(a["shops"]) != expect_shops:
            errs.append(f"{p}: expected {expect_shops} shop visits, got {len(a['shops'])}")
        for j, shop in enumerate(a["shops"]):
            if len(shop["packs"]) != 2:
                errs.append(f"{p}.shops[{j}]: expected 2 packs, got {len(shop['packs'])}")
            for k, pk in enumerate(shop["packs"]):
                if not KEY_RE["pack"].match(pk["key"]):
                    errs.append(f"{p}.shops[{j}].packs[{k}] bad key {pk['key']}")
                shape = K.PACK_BY_SHAPE.get((pk["kind"], pk["size"], pk["choices"]))
                if shape != pk["key"]:
                    errs.append(f"{p}.shops[{j}].packs[{k}] shape {pk['kind']}/{pk['size']}/{pk['choices']} != {pk['key']}")
                if len(pk["cards"]) != pk["size"]:
                    errs.append(f"{p}.shops[{j}].packs[{k}] has {len(pk['cards'])} cards, size {pk['size']}")
                for c, it in enumerate(pk["cards"]):
                    errs.extend(f"{p}.shops[{j}].packs[{k}].cards[{c}]: {e}" for e in _validate_item(it))
        if n == 1 and a["shops"] and a["shops"][0]["packs"] and a["shops"][0]["packs"][0]["key"] != "p_buffoon_normal":
            errs.append(f"{p}: first pack of the run should be the forced Buffoon Pack")
    return errs


def _validate_item(it: dict) -> list[str]:
    e = []
    s = it.get("set")
    if s == "Base":
        if it.get("key") not in VALID_BASE:
            e.append(f"bad base card {it.get('key')}")
        if it.get("enhancement") not in VALID_ENH:
            e.append(f"bad enhancement {it.get('enhancement')}")
        if it.get("edition") not in VALID_ED:
            e.append(f"bad edition {it.get('edition')}")
        if it.get("seal") not in VALID_SEAL:
            e.append(f"bad seal {it.get('seal')}")
    elif s in ("Joker", "Tarot", "Planet", "Spectral"):
        if it.get("key") not in VALID_ITEM_KEYS:
            e.append(f"unknown {s} key {it.get('key')}")
        if s == "Joker":
            if it.get("edition") not in VALID_ED:
                e.append(f"bad edition {it.get('edition')}")
            if it.get("rarity") not in (1, 2, 3, 4):
                e.append(f"bad rarity {it.get('rarity')}")
    else:
        e.append(f"unknown set {s}")
    return e


# ----------------------------------------------------------------------------- port adapter

def load_port():
    """Import mp.rng.generate and pick an entry point.  Returns (callable, name) or (None, reason).

    Preference: an explicit one-shot generator matching the documented contract; otherwise the
    RunState engine API (RunState / start_run / defeat_boss / generate_shop / open_pack /
    reroll_shop) driven with the oracle's policy by drive_runstate()."""
    try:
        from mp.rng import generate as gen  # type: ignore
    except ModuleNotFoundError as ex:
        return None, (f"mp.rng.generate is not importable yet ({ex}). "
                      "Agent C's port lives at mp/rng/generate.py; run with --validate-only until it exists.")
    except Exception as ex:  # syntax errors etc.
        return None, f"mp.rng.generate failed to import: {type(ex).__name__}: {ex}\n{traceback.format_exc()}"
    for name in ("generate_ground_truth", "generate_run", "analyze_seed"):
        fn = getattr(gen, name, None)
        if callable(fn):
            return fn, f"mp.rng.generate.{name}"
    if all(hasattr(gen, n) for n in ("RunState", "start_run", "defeat_boss", "generate_shop", "open_pack", "reroll_shop")):
        return (lambda **kw: drive_runstate(gen, **kw)), "RunState engine API via parity_check.drive_runstate"
    return None, ("mp.rng.generate imported but exposes neither generate_ground_truth/generate_run/analyze_seed "
                  "nor the RunState engine API (see contract in parity_check.py docstring).")


_ED = {None: None, "foil": "e_foil", "holo": "e_holo", "polychrome": "e_polychrome", "negative": "e_negative",
       "e_foil": "e_foil", "e_holo": "e_holo", "e_polychrome": "e_polychrome", "e_negative": "e_negative"}
_STAKE_N = {"stake_white": 1, "stake_red": 2, "stake_green": 3, "stake_black": 4, "stake_blue": 5,
            "stake_purple": 6, "stake_orange": 7, "stake_gold": 8}


def _cardgen_item(c) -> dict:
    cset = getattr(c, "set", None)
    if cset in ("Default", "Enhanced"):
        return {"set": "Base", "key": c.front, "enhancement": c.key if str(c.key).startswith("m_") else None,
                "edition": _ED.get(c.edition, c.edition), "seal": c.seal}
    it = {"set": cset, "key": c.key, "edition": _ED.get(c.edition, c.edition)}
    if cset == "Joker":
        it["rarity"] = c.rarity
        it["stickers"] = {"eternal": bool(c.eternal), "perishable": bool(c.perishable), "rental": bool(c.rental)}
    return it


def drive_runstate(gen, seed: str, deck: str, stake: str, antes: int, shop_queue_depth: int) -> dict:
    """Replay the oracle policy on the RunState engine API:
    per ante: (ante>1: defeat_boss) then 2 (ante 1) / 3 shop visits; each visit = new_round,
    generate_shop (2 slots + 2 packs), open both packs in order (released afterwards), leave;
    after the last visit of the ante, reroll until `shop_queue_depth` items were seen."""
    st = gen.RunState.for_stake(seed, stake=_STAKE_N.get(stake, 1)) if hasattr(gen.RunState, "for_stake") else gen.RunState(seed)
    rs = gen.start_run(st, deck)
    out = {"seed": seed, "antes": {}}
    for a in range(1, antes + 1):
        if a == 1:
            boss, voucher, ts, tb = rs.boss, rs.voucher, rs.tag_small, rs.tag_big
        else:
            info = gen.defeat_boss(st)
            boss, voucher, ts, tb = info["boss"], info["voucher"], info["tag_small"], info["tag_big"]
        visits = ["after_small", "after_big"] if a == 1 else ["after_prev_boss", "after_small", "after_big"]
        queue, shops = [], []
        for vi, visit in enumerate(visits):
            st.new_round()
            shop = gen.generate_shop(st)
            queue.extend(_cardgen_item(c) for c in shop.cards)
            packs = []
            for pk in shop.boosters:
                if pk is None:
                    continue
                cards = gen.open_pack(st, pk)
                packs.append({"key": re.sub(r"_\d+$", "", pk), "cards": [_cardgen_item(c) for c in cards]})
                st.release_pack(cards)
            shops.append({"visit": visit, "packs": packs})
            if vi == len(visits) - 1:
                while len(queue) < shop_queue_depth:
                    gen.reroll_shop(st, shop)
                    queue.extend(_cardgen_item(c) for c in shop.cards)
            st.release_shop(shop)
        out["antes"][str(a)] = {
            "boss": {"key": boss}, "voucher": {"key": voucher},
            "tags": {"small": {"key": ts}, "big": {"key": tb}},
            "shop_queue": queue[:shop_queue_depth], "shops": shops,
        }
    return out


def run_port(fn, gt: dict, antes: int, depth: int) -> dict:
    return fn(seed=gt["seed"], deck=gt["deck"], stake=gt["stake"], antes=antes, shop_queue_depth=depth)


# ----------------------------------------------------------------------------- diffing

def item_sig(it: dict | None, with_stickers: bool) -> str:
    if it is None:
        return "<missing>"
    if it.get("set") == "Base":
        return f"{it.get('key')}|{it.get('enhancement')}|{it.get('edition')}|{it.get('seal')}"
    s = f"{it.get('key')}|{it.get('edition')}"
    if with_stickers and it.get("set") == "Joker":
        st = it.get("stickers") or {}
        s += f"|{'E' if st.get('eternal') else ''}{'P' if st.get('perishable') else ''}{'R' if st.get('rental') else ''}"
    return s


def apply_variant(gt: dict, variant: str) -> dict:
    """Return ground truth with the requested variant applied ('analyzer' = as published;
    'faithful' = used_jokers-modelled overrides applied)."""
    if variant == "analyzer":
        return gt
    var = (gt.get("variants") or {}).get("game_faithful_used_jokers")
    if not var:
        return gt
    gt = json.loads(json.dumps(gt))
    for o in var["overrides"]:
        m = re.match(r"antes\.(\d+)\.(.*)$", o["path"])
        a, rest = m.group(1), m.group(2)
        node = gt["antes"][a]
        parts = re.findall(r"([a-z_]+)|\[(\d+)\]", rest)
        for name, idx in parts[:-1]:
            node = node[name] if name else node[int(idx)]
        name, idx = parts[-1]
        if name:
            node[name] = o["value"]
        else:
            node[int(idx)] = o["value"]
    return gt


def flagged_paths(gt: dict) -> set:
    var = (gt.get("variants") or {}).get("game_faithful_used_jokers") or {}
    out = set()
    for o in var.get("overrides", []):
        m = re.match(r"antes\.(\d+)\.(.*)$", o["path"])
        out.add(f"ante{m.group(1)}.{m.group(2)}")
    return out


def diff_ante(exp: dict, got: dict | None, fields: list[str], depth: int) -> tuple[list[tuple[str, str, str]], list[str]]:
    """Returns (mismatches [(field, expected, got)], not_produced [field])."""
    mism, absent = [], []
    if got is None:
        return [("ante", "present", "<not produced>")], []

    def cmp(label, e, g):
        if g is None:
            absent.append(label)
        elif e != g:
            mism.append((label, str(e), str(g)))

    def dig(d, *path):
        for p in path:
            if not isinstance(d, dict) or p not in d:
                return None
            d = d[p]
        return d

    if "boss" in fields:
        cmp("boss", exp["boss"]["key"], dig(got, "boss", "key"))
    if "voucher" in fields:
        cmp("voucher", exp["voucher"]["key"], dig(got, "voucher", "key"))
    if "tags" in fields:
        cmp("tags.small", exp["tags"]["small"]["key"], dig(got, "tags", "small", "key"))
        cmp("tags.big", exp["tags"]["big"]["key"], dig(got, "tags", "big", "key"))
    if "shop_queue" in fields:
        gq = got.get("shop_queue")
        if gq is None:
            absent.append("shop_queue")
        else:
            for i, e in enumerate(exp["shop_queue"][:depth]):
                g = gq[i] if i < len(gq) else None
                cmp(f"shop_queue[{i}]", item_sig(e, True), item_sig(g, True) if g is not None else "<missing>")
    if "packs" in fields:
        gs = got.get("shops")
        if gs is None:
            absent.append("packs")
        else:
            for j, shop in enumerate(exp["shops"]):
                gshop = gs[j] if j < len(gs) else {}
                for k, pk in enumerate(shop["packs"]):
                    gp = (gshop.get("packs") or [None] * 2)[k] if k < len(gshop.get("packs") or []) else None
                    cmp(f"shops[{j}:{shop['visit']}].packs[{k}]", pk["key"], dig(gp, "key") if gp else "<missing>")
                    if gp and gp.get("cards") is not None:
                        for c, ec in enumerate(pk["cards"]):
                            gc = gp["cards"][c] if c < len(gp["cards"]) else None
                            cmp(f"shops[{j}:{shop['visit']}].packs[{k}].cards[{c}]", item_sig(ec, True), item_sig(gc, True))
                    elif gp:
                        absent.append(f"shops[{j}].packs[{k}].cards")
    if "deck_order" in fields:
        for blind in ("small", "big", "boss"):
            e = (exp.get("deck_order_unverified") or {}).get(blind)
            g = (got.get("deck_order") or {}).get(blind) if isinstance(got.get("deck_order"), dict) else None
            if e is not None:
                cmp(f"deck_order.{blind}", ",".join(e[:8]), ",".join(g[:8]) if g else None)
    return mism, absent


def print_table(rows: list[tuple[str, str, str]], highlight_first: bool = True, limit: int = 25):
    if not rows:
        return
    w0 = max(len(r[0]) for r in rows[:limit]) + 2
    w1 = max(len(r[1]) for r in rows[:limit]) + 2
    print(f"    {'field'.ljust(w0)}{'expected'.ljust(w1)}got")
    for i, (f, e, g) in enumerate(rows[:limit]):
        mark = " <-- first mismatch" if (highlight_first and i == 0) else ""
        print(f"    {f.ljust(w0)}{e.ljust(w1)}{g}{mark}")
    if len(rows) > limit:
        print(f"    ... {len(rows) - limit} more")


# ----------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", help="comma-separated subset (default: all ground-truth seeds)")
    ap.add_argument("--antes", default="1-3", help="e.g. 1-3 or 1,2,5 (default 1-3)")
    ap.add_argument("--fields", default=",".join(FIELDS_DEFAULT),
                    help="subset of boss,voucher,tags,shop_queue,packs,deck_order (default: all but deck_order)")
    ap.add_argument("--queue-depth", type=int, default=None, help="compare only the first N shop-queue items")
    ap.add_argument("--validate-only", action="store_true", help="schema-check the ground-truth JSONs and exit")
    ap.add_argument("--list", action="store_true", help="list seeds with cross-check status and exit")
    ap.add_argument("--variant", choices=["analyzer", "faithful"], default="analyzer",
                    help="analyzer = as published by Immolate-lineage tools (default); faithful = with the game's "
                         "used_jokers suppression applied (see schema.md 'variants')")
    ap.add_argument("--limit", type=int, default=25, help="max diff rows printed per seed")
    ap.add_argument("--quiet", action="store_true", help="summary only")
    args = ap.parse_args(argv)

    seeds = [s.strip().upper() for s in args.seeds.split(",")] if args.seeds else None
    gts = load_ground_truth(seeds)
    if not gts:
        print(f"no ground-truth files found in {GT_DIR}" + (f" for {seeds}" if seeds else ""))
        return 2
    antes = parse_antes(args.antes)
    fields = [f for f in args.fields.split(",") if f]

    # --- validation (always runs)
    bad = 0
    for seed, gt in gts.items():
        errs = validate(gt)
        if errs:
            bad += 1
            print(f"[schema] {seed}: {len(errs)} problem(s)")
            for e in errs[:10]:
                print(f"    {e}")
    max_ante = min(len(gt["antes"]) for gt in gts.values())
    cc = {}
    for gt in gts.values():
        for name, c in (gt.get("source", {}).get("cross_checks") or {}).items():
            cc.setdefault(name, {"agree": 0, "disagree": 0})
            cc[name]["agree" if c.get("status") == "agree" else "disagree"] += 1
    print(f"[schema] {len(gts) - bad}/{len(gts)} ground-truth files valid; antes available: 1-{max_ante}; "
          f"cross-checks: " + ", ".join(f"{n} {v['agree']} agree/{v['disagree']} disagree" for n, v in cc.items()))
    if args.list:
        for seed, gt in gts.items():
            ccs = ",".join(f"{n}={c.get('status')}" for n, c in (gt["source"].get("cross_checks") or {}).items())
            print(f"  {seed.ljust(9)} {gt['deck_name']}/{gt['stake_name']} antes={len(gt['antes'])} depth={gt.get('shop_queue_depth')} {ccs}")
        return 0
    if args.validate_only:
        return 1 if bad else 0

    # --- port
    fn, how = load_port()
    if fn is None:
        print(f"[port] {how}")
        print("[port] Nothing to compare; ground truth is ready.  Re-run without --validate-only once mp/rng/generate.py exists.")
        return 2
    print(f"[port] using {how}; fields={fields}; antes={antes}; variant={args.variant}")

    exact_through = {}  # seed -> highest ante k such that antes 1..k are exact
    failures = 0
    for seed, gt0 in gts.items():
        gt = apply_variant(gt0, args.variant)
        flagged = flagged_paths(gt0)
        depth = args.queue_depth or gt.get("shop_queue_depth") or len(gt["antes"]["1"]["shop_queue"])
        try:
            got = run_port(fn, gt, max(antes), depth)
        except Exception as ex:
            failures += 1
            exact_through[seed] = 0
            print(f"\n== {seed}: port raised {type(ex).__name__}: {ex}")
            if not args.quiet:
                traceback.print_exc()
            continue
        got_antes = (got or {}).get("antes") or {}
        k_exact = 0
        all_rows = []
        absent_all = set()
        broken = False
        for a in antes:
            exp = gt["antes"].get(str(a))
            if exp is None:
                break
            g = got_antes.get(str(a)) or got_antes.get(a)
            rows, absent = diff_ante(exp, g, fields, depth)
            absent_all.update(absent)
            if rows:
                for f, e, gg in rows:
                    label = f"ante{a}.{f}"
                    norm = re.sub(r"shops\[(\d+):[a-z_]+\]", r"shops[\1]", label)
                    if norm in flagged:
                        label += "  [analyzer gap: used_jokers]"
                    all_rows.append((label, e, gg))
                broken = True
            elif not broken:
                k_exact = a
        exact_through[seed] = k_exact
        status = "EXACT" if not all_rows else f"{len(all_rows)} mismatch(es)"
        print(f"\n== {seed}: {status} through ante {k_exact} (checked {antes[0]}-{antes[-1]})"
              + (f"; not produced: {sorted(absent_all)[:6]}" if absent_all else ""))
        if all_rows and not args.quiet:
            print_table(all_rows, limit=args.limit)

    want_k = antes[-1]
    n_exact = sum(1 for k in exact_through.values() if k >= want_k)
    print(f"\nSUMMARY: {n_exact}/{len(gts)} seeds exact through ante {want_k}"
          + (f" ({failures} port errors)" if failures else ""))
    for k in range(want_k, 0, -1):
        n = sum(1 for v in exact_through.values() if v >= k)
        print(f"  exact through ante {k}: {n}/{len(gts)}")
    return 0 if n_exact == len(gts) else 1


if __name__ == "__main__":
    sys.exit(main())
