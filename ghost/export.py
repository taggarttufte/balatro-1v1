"""
export.py — convert a logged ``MLBMatch`` line (replay/log.py, ``kind: "match"``) into a
ghost-replay ``.json`` the BalatroMultiplayer mod plays back natively.

    python -m ghost.export <log.jsonl> <idx> [--install | --out PATH]
        [--agent-seat winner|0|1] [--player-name Tagg] [--nemesis-name NAME]

What the mod actually consumes (all citations are files inside the installed mod at
``%APPDATA%/Balatro/Mods/Multiplayer/``, read-only, nothing copied):

* ``lib/ghost_replay.lua::load_json_replay`` accepts any ``.json`` under ``$MOD/replays/``
  with an ``ante_snapshots`` table (string or numeric ante keys — it normalises).
* During play the ONLY load-bearing fields are, per Nemesis ante,
  ``ante_snapshots[ante].hands = [{score, hands_left, side}]`` filtered to
  ``side == "enemy"`` (``ghost_replay.lua::get_enemy_hands``): the human's live score is
  raced against each entry's ``score`` in order (``resolve_pvp_mid_hand`` /
  ``resolve_pvp_hands_exhausted``).  ``score`` is an insane-int string; a plain decimal
  string parses as itself (``lib/insane_int.lua::from_string``), which covers every score
  our engine produces in the ante range that matters.
* **Why we write ONE entry per side per ante (format v2, 2026-08-27):** when the human
  runs out of hands, ``ui/game/game_state.lua:188-198`` synchronously calls
  ``resolve_pvp_hands_exhausted``, which compares chips against the CURRENT index entry
  and awards the round only if playback is already exhausted — but the index advances
  ONLY through the 0.6 s/entry animation (``_start_advance_sequence``), which never runs
  on the human's final hand.  With per-hand entries, overtaking the ghost on your last
  hand leaves the index lagging and takes YOUR life despite beating the final score
  (observed in game, 2026-08-27).  A single pre-exhausted entry per side (final score,
  ``hands_left`` 0) cannot lag: overtaking mid-blind ends the round in the human's favour
  at once, and the exhaustion check compares straight against the final score.  The
  per-hand progression is preserved in each snapshot's ``_hand_progression`` (a field the
  mod ignores) for the G2 mirror and analysis.
* Everything else (``player_score``/``enemy_score``/lives/``result`` per ante, jokers,
  names, ``final_ante``, ``winner``, ``timestamp``) is display-only metadata for the
  replay-picker UI (``ui/main_menu/play_button/ghost_replay_picker.lua``).
* ``ruleset`` must be a ``MP.Rulesets`` key or ``is_ruleset_supported()`` refuses to load
  it; Major League is ``"ruleset_mp_majorleague"`` (used verbatim at ``core.lua:114``) and
  it forces ``gamemode_mp_attrition`` (``rulesets/majorleague.lua:16``).
* On "Play Match" the mod starts the human's run with the REPLAY's seed and deck
  (``lib/practice_mode.lua:32-34`` — ``start_run(e, {seed = r.seed, stake = r.stake})``,
  deck looked up BY DISPLAY NAME via ``MP.UTILS.get_deck_key_from_name``), so the race is
  same-seed by construction — seed parity is what makes the ghost's shops/scores the ones
  the human actually sees.

Both seats are written (agent = ``side "enemy"``, its sim opponent = ``side "player"``),
so the picker's perspective-flip button works.

Extraction subtlety this module owns: the per-step summary in a logged line is captured
AFTER ``match.step()``, and the step that RESOLVES a Nemesis round tears the blind down in
the same call — so the resolving play's post-step summary can show reset chips and a
non-PvP state.  PvP plays are therefore detected from the PRE-step summary (the previous
step's snapshot of that seat), and the final entry of each round takes its score from
``pvp_log`` (the engine's own resolution record) instead of the post-step summary.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Any, Optional

RULESET_KEY = "ruleset_mp_majorleague"
GAMEMODE_KEY = "gamemode_mp_attrition"
GENERATOR_TAG = "balatro-1v1 ghost/export v1"


class GhostExportError(ValueError):
    """A logged line that cannot become a playable ghost (wrong kind, no Nemesis resolved,
    or internally inconsistent ops/steps/pvp_log)."""


# ─────────────────────────────────────────────────────────────── small helpers

def _score_str(v: Any) -> str:
    """Chip score -> the plain-decimal string the mod's insane-int parser reads back
    unchanged.  Non-integral scores would silently corrupt the race target, so they are
    an error, not a rounding."""
    if isinstance(v, bool):
        raise GhostExportError(f"score is a bool: {v!r}")
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v.is_integer():
            return str(int(v))
        raise GhostExportError(f"non-integral chip score {v!r} cannot be exported")
    raise GhostExportError(f"unexpected score type {type(v).__name__}: {v!r}")


def _deck_display_name(deck_key: str) -> str:
    """Engine ``b_*`` key -> the game's display name ("Red Deck"), which is what
    ``MP.UTILS.get_deck_key_from_name`` matches against ``G.P_CENTERS[k].name``."""
    from ._bootstrap import decks
    spec = decks.DECKS.get(deck_key)
    if spec is None:
        raise GhostExportError(f"unknown deck_key {deck_key!r}")
    return spec.name


def slugify(text: str) -> str:
    return re.sub(r"-{2,}", "-", re.sub(r"[^A-Za-z0-9]+", "-", text)).strip("-") or "ghost"


def load_line(path: str, idx: int) -> dict:
    """Line ``idx`` of a replay JSONL file.  Local on purpose — this module stays
    importable without pulling the replay package in."""
    with open(path, "r", encoding="utf-8") as f:
        for i, raw in enumerate(f):
            if i == idx:
                return json.loads(raw)
    raise GhostExportError(f"{path} has no line {idx}")


# ─────────────────────────────────────────────────────────── the converter

def _pvp_hand_entries(line: dict, seat_side: dict) -> dict:
    """ops+steps -> {ante: [ {score, hands_left, side}, ... ]} in chronological order.

    A play by seat ``p`` at op ``i`` is a PvP play iff the PRE-step summary
    (``steps[i-1]["players"][p]``) shows ``is_pvp`` with hands in hand.  Three outcomes:

    * normal: post-step summary still in the same PvP blind with ``hands_left`` down one —
      take score/hands_left from it;
    * ignored: post-step summary identical in (state, chips, hands_left) — the engine's
      documented-permissive ``step()`` dropped an illegal action, no entry;
    * resolving: anything else — the round ended inside this step (post-step state is
      already torn down), so the entry's score is the seat's score in ``pvp_log`` for
      that ante and ``hands_left`` is pre minus one.
    """
    ops, steps = line["ops"], line["steps"]
    if len(ops) != len(steps):
        raise GhostExportError(f"ops/steps length mismatch: {len(ops)} vs {len(steps)}")
    by_ante = {int(a): (loser, s0, s1) for a, loser, s0, s1 in line["pvp_log"]}
    if len(by_ante) != len(line["pvp_log"]):
        raise GhostExportError("pvp_log has two Nemesis rounds in one ante — "
                               "ante_snapshots cannot represent that")

    hands: dict = {}
    for i, (op, st) in enumerate(zip(ops, steps)):
        if op["action"].get("type") != "play" or i == 0:
            continue
        p = op["player"]
        pre = steps[i - 1]["players"][p]
        if not pre["is_pvp"] or pre["hands_left"] <= 0:
            continue
        ante = pre["ante"]
        post = st["players"][p]
        in_same_round = (post["is_pvp"] and post["ante"] == ante
                         and post["hands_left"] == pre["hands_left"] - 1)
        unchanged = (post["state"] == pre["state"]
                     and post["chips_scored"] == pre["chips_scored"]
                     and post["hands_left"] == pre["hands_left"])
        if in_same_round:
            entry = {"score": _score_str(post["chips_scored"]),
                     "hands_left": int(post["hands_left"]), "side": seat_side[p]}
        elif unchanged:
            continue    # documented-permissive step() ignored an illegal action
        else:
            if ante not in by_ante:
                raise GhostExportError(
                    f"op {i}: seat {p}'s play at ante {ante} left the PvP blind but "
                    f"pvp_log has no resolution for that ante")
            _loser, s0, s1 = by_ante[ante]
            entry = {"score": _score_str((s0, s1)[p]),
                     "hands_left": int(pre["hands_left"]) - 1, "side": seat_side[p]}
        hands.setdefault(ante, []).append(entry)
    return hands


def _lives_after_ante(line: dict, seat: int) -> dict:
    """{ante: lives seat had at its first step AFTER that ante}, falling back to the
    final state — a display-only field on the picker's ante breakdown."""
    out = {}
    for st in line["steps"]:
        s = st["players"][seat]
        for ante in list(out):
            if out[ante] is None and s["ante"] > ante:
                out[ante] = s["lives"]
        if s["ante"] not in out:
            out[s["ante"]] = None       # placeholder until a later-ante step is seen
    final = line["final_state"]["players"][seat]["lives"]
    return {a: (v if v is not None else final) for a, v in out.items()}


def ghost_replay(line: dict, *, agent_seat: Optional[int] = None,
                 player_name: str = "Tagg", nemesis_name: Optional[str] = None,
                 timestamp: Optional[int] = None) -> dict:
    """A logged match line -> the mod's ghost-replay JSON table.  ``agent_seat`` is the
    seat the human will race (written as ``side "enemy"``); default = the seat that won
    the sim match (seat 0 if undecided)."""
    if line.get("kind") != "match":
        raise GhostExportError(
            f"kind={line.get('kind')!r}: only kind='match' lines are exportable in G1 "
            "(a solo episode has no opponent seat to hand the human)")
    if not line.get("pvp_log"):
        raise GhostExportError("no Nemesis round was resolved in this match — nothing to race")

    winner = line["final_state"].get("winner")
    if agent_seat is None:
        agent_seat = winner if winner in (0, 1) else 0
    if agent_seat not in (0, 1):
        raise GhostExportError(f"agent_seat must be 0 or 1, got {agent_seat!r}")
    player_seat = 1 - agent_seat
    seat_side = {agent_seat: "enemy", player_seat: "player"}

    progression = _pvp_hand_entries(line, seat_side)
    lives = {seat: _lives_after_ante(line, seat) for seat in (0, 1)}

    snapshots = {}
    for ante, loser, s0, s1 in line["pvp_log"]:
        ante = int(ante)
        scores = (s0, s1)
        prog = progression.get(ante, [])
        enemy_rows = [h for h in prog if h["side"] == "enemy"]
        if enemy_rows and enemy_rows[-1]["score"] != _score_str(scores[agent_seat]):
            raise GhostExportError(
                f"ante {ante}: last enemy hand score {enemy_rows[-1]['score']} != "
                f"pvp_log resolution {scores[agent_seat]} — converter grouping bug")
        snapshots[str(ante)] = {
            "player_score": _score_str(scores[player_seat]),
            "enemy_score": _score_str(scores[agent_seat]),
            "player_lives": lives[player_seat].get(ante,
                                                   line["final_state"]["players"][player_seat]["lives"]),
            "enemy_lives": lives[agent_seat].get(ante,
                                                  line["final_state"]["players"][agent_seat]["lives"]),
            "result": ("win" if loser == agent_seat else
                       "loss" if loser == player_seat else "tie"),
            # one pre-exhausted entry per side — the mod's exhaustion check cannot
            # index-lag against these (module docstring, "format v2")
            "hands": [
                {"score": _score_str(scores[player_seat]), "hands_left": 0, "side": "player"},
                {"score": _score_str(scores[agent_seat]), "hands_left": 0, "side": "enemy"},
            ],
            "_hand_progression": prog,
        }

    fs = line["final_state"]
    nemesis_name = nemesis_name or line.get("meta", {}).get("spec") or "EV agent"
    doc = {
        "gamemode": GAMEMODE_KEY,
        "ruleset": RULESET_KEY,
        "seed": line["seed"],
        "deck": _deck_display_name(line["deck_key"]),
        "stake": line.get("stake", 1),
        "final_ante": fs["players"][agent_seat]["ante"],
        "winner": ("nemesis" if winner == agent_seat else
                   "player" if winner == player_seat else "unknown"),
        "timestamp": int(timestamp if timestamp is not None else time.time()),
        "player_name": player_name,
        "nemesis_name": nemesis_name,
        "player_jokers": [{"key": k} for k in fs["players"][player_seat]["jokers"]],
        "nemesis_jokers": [{"key": k} for k in fs["players"][agent_seat]["jokers"]],
        "ante_snapshots": snapshots,
        "_generator": {
            "tool": GENERATOR_TAG,
            "agent_seat": agent_seat,
            "sim_winner_seat": winner,
            "spec": line.get("meta", {}).get("spec"),
        },
    }
    return doc


# ─────────────────────────────────────────────────────────── writing / install

def mod_replays_dir() -> str:
    appdata = os.environ.get("APPDATA")
    if not appdata:
        raise GhostExportError("APPDATA is not set — cannot locate the Balatro mod folder")
    d = os.path.join(appdata, "Balatro", "Mods", "Multiplayer", "replays")
    if not os.path.isdir(os.path.dirname(d)):
        raise GhostExportError(f"BalatroMultiplayer mod not found at {os.path.dirname(d)}")
    os.makedirs(d, exist_ok=True)
    return d


def default_filename(doc: dict) -> str:
    return f"ghost_{doc['seed']}_{slugify(doc['nemesis_name'])}.json"


def write_ghost(doc: dict, path: str) -> str:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
    return path


# ─────────────────────────────────────────────────────────────────────── CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m ghost.export",
        description="Convert one logged MLBMatch line into a mod-playable ghost replay.")
    ap.add_argument("file", help="replay JSONL file (replay/log.py MatchLogger output)")
    ap.add_argument("idx", type=int, help="line index within the file")
    ap.add_argument("--agent-seat", default="winner", choices=("winner", "0", "1"),
                    help="which seat becomes the ghost (default: the sim-match winner)")
    ap.add_argument("--player-name", default="Tagg")
    ap.add_argument("--nemesis-name", default=None,
                    help="ghost display name (default: the line's meta.spec, else 'EV agent')")
    ap.add_argument("--out", default=None, help="write here instead of installing")
    ap.add_argument("--install", action="store_true",
                    help="write into the mod's replays/ folder so the in-game picker sees it")
    args = ap.parse_args(argv)

    line = load_line(args.file, args.idx)
    seat = None if args.agent_seat == "winner" else int(args.agent_seat)
    doc = ghost_replay(line, agent_seat=seat, player_name=args.player_name,
                       nemesis_name=args.nemesis_name)

    if args.out:
        path = write_ghost(doc, args.out)
    elif args.install:
        path = write_ghost(doc, os.path.join(mod_replays_dir(), default_filename(doc)))
    else:
        path = write_ghost(doc, default_filename(doc))

    n_antes = len(doc["ante_snapshots"])
    print(f"wrote {path}")
    print(f"  seed {doc['seed']} · {doc['deck']} · {n_antes} Nemesis ante(s) · "
          f"ghost = seat {doc['_generator']['agent_seat']} ({doc['nemesis_name']}) · "
          f"sim winner = {doc['winner']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
