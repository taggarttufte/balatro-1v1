"""
Convert raw analyzer dumps (oracle/blueprint_runner/_raw/<SEED>.{blueprint,thesoul}.json)
into oracle/ground_truth/<SEED>.json in the schema described by oracle/schema.md.

Primary source = Blueprint (TypeScript port).  TheSoul (WASM build of the C++ Immolate) is
an independent implementation; every field it produces is compared and the verdict is stored
in `source.cross_checks.thesoul_wasm`.  A hand-transcribed third-party datum (balatrohq's
server-rendered analysis of ALEEB) is checked when the seed matches.

    python oracle/build_ground_truth.py [--raw DIR] [--out DIR] [--seeds A,B]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import keymap as K  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
VISITS = [("smallBlind", "after_prev_boss"), ("bigBlind", "after_small"), ("bossBlind", "after_big")]

ASSUMPTIONS = [
    "Profile: fully unlocked (every joker/voucher/tag unlockable is unlocked and discovered).",
    "Fresh run: Stone/Steel/Glass Joker, Golden Ticket, Lucky Cat (enhancement_gate), Cavendish "
    "(yes_pool_flag gros_michel_extinct) and Planet X/Ceres/Eris (softlock) are UNAVAILABLE in pools "
    "and are resampled past, exactly as the game does for a deck with no such cards / hands played.",
    "No purchases, no rerolls, no skips, no Showman: the shop queue is the raw per-ante stream "
    "(cdt{ante} / Joker{r}sho{ante} / Tarotsho{ante} / ...); the game fills 2 slots per shop visit "
    "and each reroll draws the next 2.",
    "Every booster pack is opened, in display order, so pack N of a kind within an ante assumes "
    "packs 1..N-1 of that kind were opened (they share the per-ante per-source stream).",
    "Shop visits: ante 1 has two (after Small, after Big); every later ante has three, the first "
    "being the shop entered right after the previous ante's Boss (ease_ante runs before that shop, "
    "so it is keyed with the new ante).  Immolate-lineage UIs label these smallBlind/bigBlind/bossBlind "
    "= the blind you are about to play.",
    "The first pack of the run is the forced Buffoon Pack (get_pack: first_shop_buffoon), consuming "
    "no shop_pack RNG.",
    "Pack art variants (p_arcana_normal_1..4 etc.) are not resolved; keys drop the numeric suffix.",
    "Analyzer version flag 10106 (= 1.0.1f+ pools).  No generation-relevant change is known between "
    "1.0.1f and 1.0.1o; the community uses these analyzers against 1.0.1o and this oracle was built "
    "against a local 1.0.1o install.",
    "Stickers (eternal/perishable/rental) only roll at Black/Orange/Gold stake; at White stake they are all false.",
]


def _set_table(typ: str):
    return {"Joker": K.JOKERS, "Tarot": K.TAROTS, "Planet": K.PLANETS, "Spectral": K.SPECTRALS}[typ]


def conv_item(c: dict) -> dict:
    """Blueprint queue/pack card -> schema item."""
    typ = c["type"]
    if typ == "Standard":
        base = c["base"]
        if isinstance(base, list):
            base = "".join(base)
        suit, rank = base.split("_")
        return {
            "set": "Base",
            "key": base,
            "name": f"{K.RANK_NAMES[rank]} of {K.SUIT_NAMES[suit]}",
            "enhancement": K.ENHANCEMENTS[c.get("enhancements") or "No Enhancement"],
            "edition": K.EDITIONS[c.get("edition") or "No Edition"],
            "seal": K.SEALS[c.get("seal") or "No Seal"],
        }
    name = c["name"]
    if name in ("The Soul", "Black Hole"):
        typ = "Spectral"  # Blueprint types these as Spectral already; be explicit
    table = _set_table(typ)
    key = K.key_for(name, table)
    item = {"set": typ, "key": key, "name": K.key_to_name(key)}
    if typ == "Joker":
        item["edition"] = K.EDITIONS[c.get("edition") or "No Edition"]
        item["rarity"] = c.get("rarity")
        item["stickers"] = {
            "eternal": bool(c.get("isEternal")),
            "perishable": bool(c.get("isPerishable")),
            "rental": bool(c.get("isRental")),
        }
    return item


def conv_pack(p: dict) -> dict:
    kind = p["name"]
    size, choices = p["size"], p["choices"]
    key = K.PACK_BY_SHAPE[(kind, size, choices)]
    name = [n for n, k in K.PACKS.items() if k == key][0]
    return {"key": key, "name": name, "kind": kind, "size": size, "choices": choices,
            "cards": [conv_item(c) for c in p["cards"]]}


def convert_blueprint(raw: dict) -> dict:
    res = raw["result"]
    st = raw["settings"]
    out_antes = {}
    soul_counter = 0
    leg_stream = raw["legendary"]["legendary_stream"]
    for a_str, a in sorted(res["antes"].items(), key=lambda kv: int(kv[0])):
        a_num = int(a_str)
        shops = []
        for blind, visit in VISITS:
            packs = a["blinds"][blind]["packs"]
            if a_num == 1 and blind == "smallBlind":
                continue
            shops.append({
                "visit": visit,
                "analyzer_label": blind,
                "packs": [conv_pack(p) for p in packs],
            })
        souls = []
        for si, shop in enumerate(shops):
            for pi, pack in enumerate(shop["packs"]):
                for ci, card in enumerate(pack["cards"]):
                    if card["key"] == "c_soul":
                        idx = soul_counter
                        soul_counter += 1
                        souls.append({
                            "shop": si, "visit": shop["visit"], "pack": pi, "card": ci,
                            "nth_soul_in_run": soul_counter,
                            "legendary_if_all_prior_souls_used": (
                                K.key_for(leg_stream[idx], K.JOKERS) if idx < len(leg_stream) else None),
                        })
        deck_orders = {}
        for blind, visit in VISITS:
            deck = a["blinds"][blind].get("deck") or []
            if deck:
                deck_orders[blind] = [("".join(c["base"]) if isinstance(c["base"], list) else c["base"]) for c in deck]
        out_antes[a_str] = {
            "boss": {"key": K.key_for(a["boss"], K.BLINDS), "name": a["boss"]},
            "voucher": {"key": K.key_for(a["voucher"], K.VOUCHERS), "name": a["voucher"]},
            "tags": {
                "small": {"key": K.key_for(a["tags"][0], K.TAGS), "name": a["tags"][0]},
                "big": {"key": K.key_for(a["tags"][1], K.TAGS), "name": a["tags"][1]},
            },
            "shop_queue": [conv_item(c) for c in a["queue"]],
            "shops": shops,
            "soul_spawns": souls,
            # Blueprint-only model of G.deck:shuffle('nr'..ante) per blind; NOT cross-checked.
            "deck_order_unverified": {
                "note": "Blueprint model of the per-blind deck shuffle (index 0 = first card drawn). "
                        "No second source; excluded from parity by default.",
                "small": deck_orders.get("smallBlind"),
                "big": deck_orders.get("bigBlind"),
                "boss": deck_orders.get("bossBlind"),
            },
        }
    chain = raw.get("voucher_chain_if_bought")
    chain_out = None
    if chain:
        chain_out = {
            "note": "Branch: the voucher shown in ante N is bought (Blueprint applies the buy after "
                    "ante N's queue, before ante N+1).  Level-2 voucher becomes available; bought one "
                    "leaves the pool.  Shop queue shown for ante N+1 reflects any rate change "
                    "(Tarot/Planet Merchant, Magic Trick...).",
            "antes": {str(k): {"voucher": {"key": K.key_for(v["voucher"], K.VOUCHERS), "name": v["voucher"]},
                               "after_buying": (K.key_for(v["after_buying"], K.VOUCHERS) if v["after_buying"] else None),
                               "shop_queue_first6_names": v["shop_queue_first6"]}
                      for k, v in chain.items()},
        }
    deck_key = K.DECKS[st["deck"]]
    return {
        "schema_version": "1.0",
        "seed": st["seed"],
        "game_version": "1.0.1o",
        "analyzer_version_flag": st["gameVersion"],
        "deck": deck_key, "deck_name": st["deck"],
        "stake": K.STAKES[st["stake"]], "stake_name": st["stake"],
        "profile": "fully_unlocked" if raw.get("unlock_all", True) else "fresh_profile",
        "assumptions": ASSUMPTIONS,
        "shop_queue_depth": st["cardsPerAnte"],
        "source": {
            "primary": "blueprint",
            "primary_detail": {
                "repo": "https://github.com/miaklwalker/Blueprint",
                "commit": raw.get("blueprint_commit"),
                "driver": "oracle/blueprint_runner/run_blueprint.ts",
                "generated_at": raw.get("generated_at"),
            },
            "cross_checks": {},
        },
        "retrieved": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "antes": out_antes,
        "legendary_stream": [K.key_for(n, K.JOKERS) for n in leg_stream],
        "legendary_stream_note": "Raw Joker4 stream (no purchase-locking).  The k-th Soul used in the run "
                                 "yields entry k only if every earlier legendary was used and none is still "
                                 "owned (owned legendaries are resampled past unless Showman).",
        "first_soul_joker_by_ante": {k: {"key": K.key_for(v["name"], K.JOKERS),
                                        "edition": K.EDITIONS[v["edition"] or "No Edition"]}
                                    for k, v in raw["legendary"]["first_soul_joker_by_ante"].items()},
        "voucher_chain_if_bought": chain_out,
    }


# ----------------------------------------------------------------------------- cross-check

def thesoul_item(c: dict) -> dict:
    if c["type"] == "Standard":
        return {"set": "Base", "key": c["base"], "enhancement": K.ENHANCEMENTS[c["enhancement"]],
                "edition": K.EDITIONS[c["edition"]], "seal": K.SEALS[c["seal"]]}
    typ = c["type"]
    if c["name"] in ("The Soul", "Black Hole"):
        typ = "Spectral"
    item = {"set": typ, "key": K.key_for(c["name"], _set_table(typ))}
    if typ == "Joker":
        item["edition"] = K.EDITIONS[c.get("edition") or "No Edition"]
        item["rarity"] = int(c["rarity"]) if c.get("rarity") is not None else None
        s = c.get("stickers") or {}
        item["stickers"] = {"eternal": bool(s.get("eternal")), "perishable": bool(s.get("perishable")), "rental": bool(s.get("rental"))}
    return item


def strip(item: dict) -> dict:
    return {k: v for k, v in item.items() if k != "name"}


def cross_check_thesoul(gt: dict, ts: dict) -> dict:
    mism, n = [], 0

    def cmp(label, exp, got):
        nonlocal n
        n += 1
        if exp != got:
            mism.append({"field": label, "blueprint": exp, "thesoul": got})

    for a_str, ga in gt["antes"].items():
        ta = ts["antes"].get(a_str)
        if ta is None:
            mism.append({"field": f"ante{a_str}", "blueprint": "present", "thesoul": "missing"})
            continue
        cmp(f"ante{a_str}.boss", ga["boss"]["key"], K.key_for(ta["boss"], K.BLINDS))
        cmp(f"ante{a_str}.voucher", ga["voucher"]["key"], K.key_for(ta["voucher"], K.VOUCHERS))
        cmp(f"ante{a_str}.tags.small", ga["tags"]["small"]["key"], K.key_for(ta["tags"][0], K.TAGS))
        cmp(f"ante{a_str}.tags.big", ga["tags"]["big"]["key"], K.key_for(ta["tags"][1], K.TAGS))
        for i, (g, t) in enumerate(zip(ga["shop_queue"], ta["queue"])):
            cmp(f"ante{a_str}.shop_queue[{i}]", strip(g), thesoul_item(t))
        cmp(f"ante{a_str}.shop_queue.len", len(ga["shop_queue"]), len(ta["queue"]))
        flat = [p for s in ga["shops"] for p in s["packs"]]
        cmp(f"ante{a_str}.npacks", len(flat), len(ta["packs"]))
        for pi, (gp, tp) in enumerate(zip(flat, ta["packs"])):
            cmp(f"ante{a_str}.pack[{pi}]", gp["key"], K.PACKS[tp["name"]])
            for ci, (gc, tc) in enumerate(zip(gp["cards"], tp["cards"])):
                cmp(f"ante{a_str}.pack[{pi}].card[{ci}]", strip(gc), thesoul_item(tc))
            cmp(f"ante{a_str}.pack[{pi}].ncards", len(gp["cards"]), len(tp["cards"]))
    return {
        "source": "TheSoul WASM (SpectralPack/TheSoul immolate.wasm, C++ Immolate by MathIsFun0)",
        "driver": "oracle/blueprint_runner/run_thesoul.js (fresh-run locks, fully unlocked profile)",
        "fields_compared": n,
        "status": "agree" if not mism else "DISAGREE",
        "mismatches": mism[:50],
        "generated_at": ts.get("generated_at"),
    }



# ----------------------------------------------------------------------------- game-faithful variant

def _named(item: dict) -> dict:
    if item["set"] == "Base":
        suit, rank = item["key"].split("_")
        item["name"] = f"{K.RANK_NAMES[rank]} of {K.SUIT_NAMES[suit]}"
    else:
        item["name"] = K.key_to_name(item["key"])
    return item


def faithful_variant(gt: dict, fr: dict) -> dict:
    """Diff the used_jokers-modelled run (run_blueprint_faithful.ts) against the primary
    analyzer output.  Returns the variant block and sets `analyzer_gap` flags on primary items."""
    overrides = []
    reasons = {"same_shop_duplicate": 0, "collides_with_displayed_shop": 0, "downstream_resample_shift": 0}
    for a_str, ga in gt["antes"].items():
        fa = fr["antes"][a_str]
        assert ga["boss"]["key"] == K.key_for(fa["boss"], K.BLINDS), (gt["seed"], a_str, "boss")
        assert ga["voucher"]["key"] == K.key_for(fa["voucher"], K.VOUCHERS), (gt["seed"], a_str, "voucher")
        fq = [_named(thesoul_item(c)) for c in fa["queue"]]
        for i, (g, f) in enumerate(zip(ga["shop_queue"], fq)):
            if strip(g) != strip(f):
                if i % 2 == 1 and ga["shop_queue"][i - 1]["key"] == g["key"]:
                    why = "same_shop_duplicate"
                else:
                    why = "downstream_resample_shift"
                reasons[why] += 1
                g["analyzer_gap"] = why
                overrides.append({"path": f"antes.{a_str}.shop_queue[{i}]", "value": f})
        fpacks = [p for bl in ("smallBlind", "bigBlind", "bossBlind") for p in fa["blinds"][bl]["packs"]]
        gpacks = [(si, pi, p) for si, s in enumerate(ga["shops"]) for pi, p in enumerate(s["packs"])]
        assert len(fpacks) == len(gpacks), (gt["seed"], a_str, "npacks")
        for (si, pi, gp), fp in zip(gpacks, fpacks):
            assert gp["kind"] == fp["name"] and gp["size"] == fp["size"], (gt["seed"], a_str, si, pi)
            shown = {c["key"] for c in ga["shop_queue"][2 * si: 2 * si + 2]}
            for ci, (gc, fc) in enumerate(zip(gp["cards"], fp["cards"])):
                f = _named(thesoul_item(fc))
                if strip(gc) != strip(f):
                    why = "collides_with_displayed_shop" if gc["key"] in shown else "downstream_resample_shift"
                    reasons[why] += 1
                    gc["analyzer_gap"] = why
                    overrides.append({"path": f"antes.{a_str}.shops[{si}].packs[{pi}].cards[{ci}]", "value": f})
    return {
        "note": "Same streams as the primary data plus the game's used_jokers semantics (card.lua "
                "Card:set_ability marks every created card; get_current_pool resamples past marked keys): "
                "slot 2 of a shop cannot repeat slot 1, and a pack opened with the shop displayed cannot "
                "contain a displayed card.  Published analyzers omit this.  Primary items that change "
                "carry `analyzer_gap`; apply `overrides` (path -> value) to obtain the faithful sequence. "
                "Boss/voucher/tags/pack kinds never differ.  Policy: packs opened at the visit, before any reroll.",
        "driver": "oracle/blueprint_runner/run_blueprint_faithful.ts",
        "generated_at": fr.get("generated_at"),
        "fields_differing": len(overrides),
        "reasons": reasons,
        "overrides": overrides,
    }


# balatrohq.com/tools/seed-analyzer/ server-renders ONE example seed (ALEEB).  Transcribed
# 2026-08-21 from the HTML ("Baseline: Red Deck - White Stake - fully unlocked").
BALATROHQ_ALEEB = {
    "ante1": {
        "boss": "bl_window", "voucher": "v_magic_trick", "tags": ["tag_skip", "tag_skip"],
        "queue8": ["j_trading", "j_rocket", "c_empress", "j_ceremonial", "j_stencil", "j_raised_fist", "j_selzer", "j_drunkard"],
        "packs": [
            ("p_buffoon_normal", ["j_red_card", "j_riff_raff"]),
            ("p_arcana_normal", ["c_temperance", "c_empress", "c_soul"]),
            ("p_standard_normal", [("C_7", "m_lucky", None), ("C_7", "m_bonus", "e_holo"), ("H_3", None, None)]),
            ("p_arcana_normal", ["c_devil", "c_star", "c_soul"]),
        ],
        "soul_spawns": ["j_caino", "j_triboulet"],
    },
    "prose": "first four antes: Canio, Triboulet, Perkeo appear via Arcana-pack Souls; ante 4 voucher = Blank",
}


def cross_check_balatrohq(gt: dict) -> dict | None:
    if gt["seed"] != "ALEEB":
        return None
    b = BALATROHQ_ALEEB["ante1"]
    a1 = gt["antes"]["1"]
    mism = []

    def cmp(label, exp, got):
        if exp != got:
            mism.append({"field": label, "balatrohq": exp, "blueprint": got})

    cmp("boss", b["boss"], a1["boss"]["key"])
    cmp("voucher", b["voucher"], a1["voucher"]["key"])
    cmp("tags", b["tags"], [a1["tags"]["small"]["key"], a1["tags"]["big"]["key"]])
    cmp("queue[0:8]", b["queue8"], [c["key"] for c in a1["shop_queue"][:8]])
    flat = [p for s in a1["shops"] for p in s["packs"]]
    for i, (pk, cards) in enumerate(b["packs"]):
        cmp(f"pack[{i}].key", pk, flat[i]["key"] if i < len(flat) else None)
        got = []
        for c in flat[i]["cards"]:
            got.append((c["key"], c["enhancement"], c["edition"]) if c["set"] == "Base" else c["key"])
        cmp(f"pack[{i}].cards", [tuple(x) if isinstance(x, tuple) else x for x in cards], got)
    cmp("soul_spawns->legendaries", b["soul_spawns"],
        [s["legendary_if_all_prior_souls_used"] for s in a1["soul_spawns"]])
    cmp("ante4.voucher", "v_blank", gt["antes"]["4"]["voucher"]["key"])
    return {"source": "balatrohq.com/tools/seed-analyzer/ (server-rendered example seed, transcribed 2026-08-21)",
            "status": "agree" if not mism else "DISAGREE", "mismatches": mism}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(HERE, "blueprint_runner", "_raw"))
    ap.add_argument("--out", default=os.path.join(HERE, "ground_truth"))
    ap.add_argument("--seeds", default=None, help="comma-separated subset")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    want = set(args.seeds.split(",")) if args.seeds else None
    files = sorted(f for f in os.listdir(args.raw) if f.endswith(".blueprint.json"))
    agree = disagree = 0
    tot_over = 0
    tot_reasons: dict = {}
    summary = []
    for f in files:
        seed = f[: -len(".blueprint.json")]
        if want and seed not in want:
            continue
        raw = json.load(open(os.path.join(args.raw, f), encoding="utf-8"))
        gt = convert_blueprint(raw)
        tsf = os.path.join(args.raw, f"{seed}.thesoul.json")
        if os.path.exists(tsf):
            cc = cross_check_thesoul(gt, json.load(open(tsf, encoding="utf-8")))
            gt["source"]["cross_checks"]["thesoul_wasm"] = cc
            if cc["status"] == "agree":
                agree += 1
            else:
                disagree += 1
                print(f"  DISAGREE {seed}: {len(cc['mismatches'])} mismatches, first: {cc['mismatches'][0]}")
        ff = os.path.join(args.raw, f"{seed}.faithful.json")
        if os.path.exists(ff):
            var = faithful_variant(gt, json.load(open(ff, encoding="utf-8")))
            gt["variants"] = {"game_faithful_used_jokers": var}
            tot_over += var["fields_differing"]
            for k_, v_ in var["reasons"].items():
                tot_reasons[k_] = tot_reasons.get(k_, 0) + v_
        hq = cross_check_balatrohq(gt)
        if hq:
            gt["source"]["cross_checks"]["balatrohq_ssr"] = hq
            print(f"  balatrohq check for {seed}: {hq['status']} {hq['mismatches']}")
        with open(os.path.join(args.out, f"{seed}.json"), "w", encoding="utf-8", newline="\n") as fh:
            json.dump(gt, fh, indent=1, ensure_ascii=False)
        summary.append(seed)
    print(f"wrote {len(summary)} files to {args.out}; TheSoul cross-check: {agree} agree, {disagree} disagree")
    print(f"game-faithful variant: {tot_over} fields differ from analyzer output across all files; reasons: {tot_reasons}")


if __name__ == "__main__":
    main()
