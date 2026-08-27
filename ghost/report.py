"""
report.py — the post-match report: everything the agent saw and did.

    python -m ghost.report <session-dir> [--spec ev:fast]

The ghost-mode game-over screen has no data source for the opponent's build (end-game
jokers are a network exchange in real MP), so this reconstructs it instead: the mirror
is deterministic in (seed, spec), so replaying the session's outbox reproduces the
agent's ENTIRE run bit-for-bit — every shop it saw, every pack it opened (with the
contents it chose from), every buy/sell/reroll, its Nemesis hands, and the final build.
Writes ``report.md`` into the session dir and prints it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ghost.ipc import iter_session  # noqa: E402
from ghost.mirror import MirrorAgent, MirrorDead  # noqa: E402
from ghost.parity_card import display_name, pack_name  # noqa: E402


def _fmt(n) -> str:
    try:
        return f"{int(float(str(n).replace(',', ''))):,}"
    except (ValueError, TypeError):
        return str(n)


def rebuild(session_dir: str, spec: str = "ev:fast", deck_key: str = "b_red",
            stake: int = 1, lives: int = 4):
    session = Path(session_dir)
    seed = session.name.split("_")[0]
    events = list(iter_session(str(session / "outbox.jsonl")))

    m = MirrorAgent(seed, spec=spec, deck_key=deck_key, stake=stake, lives=lives)
    rounds, dead_at = [], None
    try:
        m.advance_to_nemesis()
        for ev in events:
            if ev["e"] == "pvp_result":
                pend = dict(m._pending)
                res = m.resolve(ev["human_score"], ev.get("human_lives"))
                res["hands"] = pend["hands"]
                rounds.append(res)
                m.advance_to_nemesis()
            elif ev["e"] == "round_fail":
                m.set_opponent_lives(ev["lives"])
    except MirrorDead:
        dead_at = m.ante
    return seed, m, rounds, dead_at


def compose(seed: str, spec: str, m: MirrorAgent, rounds: list, dead_at) -> str:
    by_ante: dict = {}
    for c in m.chronicle:
        by_ante.setdefault(c["ante"], []).append(c)
    round_by_ante = {r["ante"]: r for r in rounds}

    lines = [f"# Agent match report — seed {seed} ({spec})", ""]
    if rounds:
        w = sum(1 for r in rounds if r["loser"] == "human")
        lines.append(f"Nemesis rounds: agent won {w}, human won "
                     f"{sum(1 for r in rounds if r['loser'] == 'agent')}, "
                     f"ties {sum(1 for r in rounds if r['loser'] is None)}."
                     + (f"  Agent DIED at ante {dead_at}." if dead_at else ""))
        lines.append("")

    for ante in sorted(set(by_ante) | set(round_by_ante)):
        lines.append(f"## Ante {ante}")
        for c in by_ante.get(ante, []):
            k = c["kind"]
            if k == "shop":
                items = ", ".join(display_name(i["key"]) + (" (sold)" if i["sold"] else "")
                                  for i in c["items"]) or "(empty)"
                lines.append(f"- shop (${c['money']}): {items}")
            elif k == "buy":
                lines.append(f"- **bought {display_name(c['item'])}** (${c['money']})")
            elif k == "reroll":
                lines.append(f"- rerolled (${c['money']})")
            elif k == "sell":
                lines.append(f"- sold {display_name(c['item'])}")
            elif k == "use":
                lines.append(f"- used {display_name(c['item'])}")
            elif k == "pack_open":
                lines.append(f"- **opened {pack_name(c['pack'])}**: "
                             + ", ".join(display_name(x) for x in c["contents"]))
            elif k == "skip_blind":
                lines.append(f"- skipped the {c['blind']} blind")
        r = round_by_ante.get(ante)
        if r:
            hands = " -> ".join(_fmt(h["score"]) for h in r["hands"])
            verdict = {"human": "HUMAN lost the round", "agent": "AGENT lost the round",
                       None: "exact tie — nobody"}[r["loser"]]
            lines.append(f"- **NEMESIS**: agent {_fmt(r['agent_final'])} ({hands}) "
                         f"vs human {_fmt(r['human_final'])} — {verdict} "
                         f"(agent lives {r['agent_lives']}, ${r['money']})")
        lines.append("")

    lines.append("## Final agent build")
    lines.append("- jokers: " + (", ".join(display_name(j.key) for j in m.game.jokers)
                                 or "(none)"))
    lines.append(f"- money ${m.game.dollars}, lives {m.game.lives}, ante {m.game.ante}")
    if m.game.consumable_hand:
        lines.append("- consumables: "
                     + ", ".join(display_name(k) for k in m.game.consumable_hand))
    lines += ["", "## Agent play stats (chronicle totals)"] + stats_lines(m)
    return "\n".join(lines)


def stats_lines(m: MirrorAgent) -> list:
    """Aggregates for the human-vs-agent comparison table.  The human's column comes
    from the capture layer (G3) — until then, compare against your own in-game stats."""
    from balatro_sim import game_keys as K

    def _cat(key):
        if key in K.TAROT_NAME:
            return "tarot"
        if key in K.PLANET_NAME:
            return "planet"
        if key in K.SPECTRAL_NAME:
            return "spectral"
        if key in K.JOKER_NAME:
            return "joker"
        if key in K.VOUCHER_NAME:
            return "voucher"
        return "other"

    buys = [c for c in m.chronicle if c["kind"] == "buy"]
    uses = [c for c in m.chronicle if c["kind"] == "use"]
    packs = [c for c in m.chronicle if c["kind"] == "pack_open"]
    rerolls = [c for c in m.chronicle if c["kind"] == "reroll"]
    sells = [c for c in m.chronicle if c["kind"] == "sell"]
    skips = [c for c in m.chronicle if c["kind"] == "skip_blind"]
    spent = sum(c.get("price", 0) for c in buys)

    def by_cat(entries, field):
        out: dict = {}
        for c in entries:
            out.setdefault(_cat(c[field]), []).append(display_name(c[field]))
        return out

    lines = [f"- total shop spend: ${spent} over {len(buys)} purchases",
             f"- rerolls: {len(rerolls)} · sells: {len(sells)} · "
             f"blind skips: {len(skips)}"]
    for cat, names in sorted(by_cat(buys, "item").items()):
        lines.append(f"- bought {cat}s ({len(names)}): " + ", ".join(names))
    for cat, names in sorted(by_cat(uses, "item").items()):
        lines.append(f"- used {cat}s ({len(names)}): " + ", ".join(names))
    if packs:
        lines.append(f"- packs opened ({len(packs)}): "
                     + ", ".join(pack_name(p["pack"]) for p in packs))
    deck_delta = len(m.game.full_deck) - 52
    lines.append(f"- final deck size: {len(m.game.full_deck)} "
                 f"({'+' if deck_delta >= 0 else ''}{deck_delta} vs start)")
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ghost.report", description=__doc__)
    ap.add_argument("session_dir")
    ap.add_argument("--spec", default="ev:fast")
    ap.add_argument("--deck-key", default="b_red")
    ap.add_argument("--stake", type=int, default=1)
    ap.add_argument("--lives", type=int, default=4)
    args = ap.parse_args(argv)

    seed, m, rounds, dead_at = rebuild(args.session_dir, spec=args.spec,
                                       deck_key=args.deck_key, stake=args.stake,
                                       lives=args.lives)
    text = compose(seed, args.spec, m, rounds, dead_at)
    out = Path(args.session_dir) / "report.md"
    out.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\n(saved to {out})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
