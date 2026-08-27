"""
parity_card.py — the human-checkable seed-parity card that ships with every ghost.

    python -m ghost.parity_card <seed> [--deck-key b_red]

The race is only meaningful if the real game and the engine agree on the seed's world, so
the card lists facts Tagg can eyeball inside the first two minutes of a run, all produced
by the oracle-verified generator (``rng/generate.py``, 126/126 ground-truth corpus exact
through ante 8):

* run-start facts, fixed before the first action: ante-1 boss, shelf voucher, both skip
  tags, the first 8 cards drawn;
* the ante-1 shops after the Small and Big blinds, BEFORE any reroll.

Deliberately unconditional: no rerolls, no pack opens, nothing that depends on what the
player buys.  The caveats printed on the card are the two ways a player can invalidate a
line (skipping a blind removes its shop; a reroll replaces the card slots).
"""
from __future__ import annotations

import argparse

from . import _bootstrap  # noqa: F401  (fork-guarded engine import, root on sys.path)
from balatro_sim import game_keys as K

from rng import generate as G

_NAME_TABLES = ("JOKER_NAME", "TAROT_NAME", "PLANET_NAME", "SPECTRAL_NAME",
                "VOUCHER_NAME", "BOSS_NAME", "TAG_NAME", "DECK_NAME")
_EDITION = {"holo": "Holographic", "foil": "Foil", "polychrome": "Polychrome",
            "negative": "Negative"}
_PACK_SIZE = {"normal": "", "jumbo": "Jumbo ", "mega": "Mega "}


def display_name(key: str) -> str:
    for table in _NAME_TABLES:
        name = getattr(K, table).get(key)
        if name:
            return name
    return key


def pack_name(pack_key: str) -> str:
    """'p_celestial_jumbo_1' -> 'Jumbo Celestial Pack' (display-only; falls back to the
    raw key rather than guessing)."""
    parts = pack_key.split("_")
    if len(parts) >= 3 and parts[0] == "p" and parts[2] in _PACK_SIZE:
        return f"{_PACK_SIZE[parts[2]]}{parts[1].capitalize()} Pack"
    return pack_key


def card_front(front: str) -> str:
    """'S_3' -> '3S', 'C_T' -> '10C'."""
    suit, rank = front.split("_")
    return ("10" if rank == "T" else rank) + suit


def _shop_lines(shop) -> list:
    lines = []
    for i, c in enumerate(shop.cards):
        name = display_name(c.key)
        if c.front:
            name = f"{name} ({card_front(c.front)})"
        if c.edition:
            name = f"{_EDITION.get(c.edition, c.edition)} {name}"
        lines.append(f"    slot {i + 1}: {name}")
    lines.append(f"    voucher: {display_name(shop.voucher)}")
    lines.append("    boosters: " + ", ".join(pack_name(b) if b else "-" for b in shop.boosters))
    return lines


def parity_card(seed: str, deck_key: str = "b_red") -> str:
    st = G.RunState(seed)
    rs = G.start_run(st, deck_key)
    drawn_first = [card_front(c) for c in rs.deck[-8:][::-1]]

    lines = [
        f"PARITY CARD — seed {seed} ({K.DECK_NAME.get(deck_key, deck_key)}, White Stake)",
        "Check these against the real game. Every line comes from the oracle-verified",
        "generator (126/126 ground-truth seeds exact). If ANY line disagrees, stop the",
        "race and report it — the ghost's scores would be meaningless on that seed.",
        "",
        "Run start (visible before the first hand):",
        f"  ante-1 boss:        {display_name(rs.boss)}",
        f"  shelf voucher:      {display_name(rs.voucher)}",
        f"  Small skip tag:     {display_name(rs.tag_small)}",
        f"  Big skip tag:       {display_name(rs.tag_big)}",
        f"  opening hand (8 cards, the game sorts them): {' '.join(sorted(drawn_first))}",
        f"  ...in draw order: {' '.join(drawn_first)}",
        "",
    ]
    for blind in ("Small", "Big"):
        st.new_round()
        shop = G.generate_shop(st)
        lines.append(f"  shop after the ante-1 {blind} blind (before any reroll):")
        lines.extend(_shop_lines(shop))
        lines.append("")
        st.release_shop(shop)

    lines += [
        "Caveats: play (don't skip) the ante-1 blinds — a skip removes that shop entirely.",
        "The Big-blind shop line additionally assumes ZERO rerolls at the Small shop (a",
        "reroll advances the shared per-ante shop stream) and that you don't already own a",
        "listed item (ownership redraws that slot in place). Under Major League the Nemesis",
        "replaces the boss from ante 2 on, so the ante-1 boss is the last regular boss.",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ghost.parity_card", description=__doc__)
    ap.add_argument("seed")
    ap.add_argument("--deck-key", default="b_red")
    args = ap.parse_args(argv)
    print(parity_card(args.seed.upper(), args.deck_key))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
