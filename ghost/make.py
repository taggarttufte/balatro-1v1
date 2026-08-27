"""
make.py — one command: self-play a seed, convert the winner's side to a ghost replay,
install it where the in-game picker looks.

    python -m ghost.make                          # fresh random seed, ev:fast, install
    python -m ghost.make --seed 7I4M53DL          # a chosen seed
    python -m ghost.make --spec ev:full           # a different ladder rung
    python -m ghost.make --no-install             # just write ghost/runs/, don't install

``--spec`` is ``ev/h2h.py::build_player``'s spec language (``ev:fast``, ``ev:full``,
``ev:full+stats``, ``ev:full+Vleaf``, ``real1:det``, ``scripted:...``) — the same rungs
every h2h measurement uses, so the eventual difficulty ladder (G3) is a spec string here.

The match is played under the CANONICAL PvP protocol — the world every gate and h2h
number so far was measured in.  Ghost-side consequence to know about (GHOST_NOTES.md §4):
a static recording ends each Nemesis round when the SIM race resolved, so a leader whose
sim opponent died early records fewer hands than it would have played against a stronger
human.  That artifact is inherent to G1 (the mirror, G2, is the fix — see
docs/GHOST_MOD_BRIEF_2026-08.md).

To race it in game: Main Menu -> Multiplayer -> Practice, open Ghost Replays, pick the
entry, Play Match.  The mod starts your run on the ghost's seed and deck automatically.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import zlib
from pathlib import Path

_HERE = Path(__file__).resolve().parent            # ghost
_ROOT = _HERE.parent                                # repo root
for _p in (str(_ROOT), str(_ROOT / "ev"), str(_ROOT / "eval"), str(_ROOT / "agent"),
           str(_ROOT / "stats")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ghost._bootstrap import MLBMatch  # noqa: E402  (fork-guarded engine import)
from ghost.export import (  # noqa: E402
    default_filename, ghost_replay, mod_replays_dir, slugify, write_ghost,
)
from replay.log import MatchLogger  # noqa: E402

SEED_ALPHABET = "123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"   # eval/rho_decay.py:67
RUNS_DIR = _HERE / "runs"


def _stable_seed(text: str) -> int:
    return zlib.crc32(text.encode("utf-8"))


def resolve_seed(arg: str | None) -> str:
    if not arg or arg.lower() == "random":
        rng = random.SystemRandom()
        return "".join(rng.choice(SEED_ALPHABET) for _ in range(8))
    seed = arg.upper().replace("0", "O")    # the game has no '0'; matches parse_seeds
    if seed != arg:
        print(f"seed canonicalised: {arg} -> {seed}")
    return seed


def play_logged_match(seed: str, spec: str, *, deck_key: str = "b_red", stake: int = 1,
                      lives: int = 4, max_steps: int = 4000, sims: int = 40,
                      checkpoint: str | None = None, log_path: str = None) -> dict:
    """One canonical-protocol self-play match of ``spec`` vs itself on ``seed``, logged
    through the MatchLogger hook contract.  Returns the logged line."""
    import h2h  # ev/h2h.py (its own header puts ev/eval/agent/stats on sys.path)
    pol_a, obj_a = h2h.build_player(spec, _stable_seed(f"ghost:{seed}:A"),
                                    sims=sims, checkpoint=checkpoint)
    pol_b, obj_b = h2h.build_player(spec, _stable_seed(f"ghost:{seed}:B"),
                                    sims=sims, checkpoint=checkpoint)
    for obj in (obj_a, obj_b):
        if obj is not None and hasattr(obj, "reset"):
            obj.reset()
    policies = [pol_a, pol_b]

    match = MLBMatch(seed=seed, deck_key=deck_key, stake=stake, lives=lives)
    mlog = MatchLogger(log_path, sig_every=50)
    mlog.begin(match, meta={"purpose": "ghost", "spec": spec, "sims": sims,
                            "checkpoint": checkpoint})
    while not match.done and match.steps < max_steps:
        p = match.current_player()
        if p is None:
            break
        act = policies[p](match, p, match.legal_actions(p))
        match.step(p, act)
        mlog.step(match, p, act)
    return mlog.end(match, outcome={"winner": match.winner, "steps": match.steps})


def _print_nemesis_table(doc: dict) -> None:
    print("  Nemesis rounds (ghost = enemy side):")
    for ante in sorted(doc["ante_snapshots"], key=int):
        snap = doc["ante_snapshots"][ante]
        verdict = {"win": "sim opponent won", "loss": "ghost won", "tie": "tie"}[snap["result"]]
        print(f"    A{ante}: ghost {snap['enemy_score']} vs {snap['player_score']} "
              f"({verdict}; {sum(1 for h in snap['hands'] if h['side'] == 'enemy')} ghost hands)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m ghost.make", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", default=None, help="8-char game seed; default: fresh random")
    ap.add_argument("--spec", default="ev:fast", help="ghost player (ev/h2h.py spec language)")
    ap.add_argument("--deck-key", default="b_red")
    ap.add_argument("--stake", type=int, default=1)
    ap.add_argument("--lives", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--sims", type=int, default=40, help="real1:* specs only")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--agent-seat", default="winner", choices=("winner", "0", "1"))
    ap.add_argument("--player-name", default="Tagg")
    ap.add_argument("--nemesis-name", default=None, help="ghost display name (default: --spec)")
    ap.add_argument("--no-install", action="store_true",
                    help="write ghost/runs/ only; skip the mod's replays/ folder")
    args = ap.parse_args(argv)

    seed = resolve_seed(args.seed)
    base = f"{seed}_{slugify(args.spec)}"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RUNS_DIR / f"{base}.jsonl"
    log_path.unlink(missing_ok=True)     # MatchLogger appends; a re-run replaces, not stacks
    log_path = str(log_path)

    print(f"playing {args.spec} vs {args.spec} on seed {seed} "
          f"({args.deck_key}, stake {args.stake}, lives {args.lives})...")
    t0 = time.perf_counter()
    line = play_logged_match(seed, args.spec, deck_key=args.deck_key, stake=args.stake,
                             lives=args.lives, max_steps=args.max_steps, sims=args.sims,
                             checkpoint=args.checkpoint, log_path=log_path)
    dt = time.perf_counter() - t0
    winner = line["final_state"]["winner"]
    antes = [p["ante"] for p in line["final_state"]["players"]]
    print(f"  done in {dt:.0f}s: winner seat {winner}, final antes {antes[0]}/{antes[1]}, "
          f"{len(line['pvp_log'])} Nemesis rounds, {line['outcome']['steps']} steps")
    if winner is None:
        print("  WARNING: match hit --max-steps undecided; ghost defaults to seat 0")

    seat = None if args.agent_seat == "winner" else int(args.agent_seat)
    doc = ghost_replay(line, agent_seat=seat, player_name=args.player_name,
                       nemesis_name=args.nemesis_name or args.spec)
    copy_path = write_ghost(doc, str(RUNS_DIR / f"{base}.ghost.json"))
    print(f"wrote {log_path}")
    print(f"wrote {copy_path}")
    _print_nemesis_table(doc)

    if not args.no_install:
        installed = write_ghost(doc, str(Path(mod_replays_dir()) / default_filename(doc)))
        print(f"installed {installed}")

    from ghost.parity_card import parity_card
    card = parity_card(seed, args.deck_key)
    card_path = RUNS_DIR / f"{base}.parity.txt"
    card_path.write_text(card + "\n", encoding="utf-8")
    print(f"\n{card}\n\n(parity card saved to {card_path})")

    if not args.no_install:
        print("\nTo race it: Balatro -> Multiplayer -> Practice -> Ghost Replays -> "
              f"pick '{doc['nemesis_name']}' (seed {seed}) -> Play Match.")
        print("The run starts on the ghost's seed + deck automatically; MLB life rules apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
