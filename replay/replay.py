"""
replay.py — exact re-run of a logged line through the deterministic engine, plus a
human-readable narration.

``replay(line)`` / ``replay_match(line)`` reconstruct the game(s) from ``(seed, deck_key,
stake, ruleset)`` and re-apply the logged actions (or ops, for a match) through the SAME
engine entry points that produced them (``BalatroGame.step()`` / ``MLBMatch.step()``, plus the
synthetic ``__lose_life__`` dispatch for out-of-band mutations -- see ``_util.apply_op``),
asserting every recorded ``state_signature()`` digest matches.  Phase 1 made the engine
deterministic in exactly this sense (same seed/deck/stake/ruleset + same action list =>
identical ``state_signature()`` at every step), so a mismatch means either the log or the
engine changed under it.

``narrate(line)`` drives the same reconstruction and renders a readable line-by-line story
instead of (in addition to) checking signatures.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Iterable, Optional

from ._bootstrap import BalatroGame, MLB_PVP_START_ROUND, MLBMatch, evaluate_hand
from ._util import ReplayMismatch, apply_op, match_sig_digest, sig_digest

__all__ = [
    "ReplayMismatch", "load_lines", "load_line",
    "replay", "replay_match", "replay_line",
    "narrate", "narrate_episode", "narrate_match",
    "verify_file",
]


# ============================================================================ loading

def load_lines(path: str) -> list:
    """Every JSONL line in ``path`` as a decoded dict, in file order."""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if raw:
                out.append(json.loads(raw))
    return out


def load_line(path: str, idx: int) -> dict:
    """The ``idx``-th (0-based) episode/match in ``path``."""
    lines = load_lines(path)
    if not (0 <= idx < len(lines)):
        raise IndexError(f"{path} has {len(lines)} lines, index {idx} out of range")
    return lines[idx]


# ============================================================================ episode replay

def _new_game(line: dict) -> BalatroGame:
    game = BalatroGame(
        seed=line["seed"], deck_key=line.get("deck_key", "b_red"),
        stake=line.get("stake", 1), ruleset=line.get("ruleset", "vanilla"),
    )
    lives_start = line.get("lives_start")
    if lives_start is not None and game.lives != lives_start:
        game.lives = lives_start
    return game


def _drive_episode(
    line: dict,
    on_pre: Optional[Callable[[int, Any, dict], None]] = None,
    on_post: Optional[Callable[[int, Any, dict], None]] = None,
    check: bool = True,
) -> BalatroGame:
    game = _new_game(line)
    sigs = line.get("signatures") or {}
    actions = line.get("actions") or []
    for idx, action in enumerate(actions):
        if on_pre is not None:
            on_pre(idx, game, action)
        apply_op(game, action)
        if on_post is not None:
            on_post(idx, game, action)
        if check:
            key = str(idx)
            if key in sigs:
                got = sig_digest(game)
                if got != sigs[key]:
                    raise ReplayMismatch(idx, action, sigs[key], got)
    if check and "final" in sigs:
        got = sig_digest(game)
        if got != sigs["final"]:
            last = actions[-1] if actions else None
            raise ReplayMismatch(max(len(actions) - 1, 0), last, sigs["final"], got, final=True)
    return game


def replay(line: dict) -> BalatroGame:
    """Re-run a ``kind == "episode"`` line and assert every logged signature matches.
    Raises ``ReplayMismatch`` at the first divergent step.  Returns the final game."""
    return _drive_episode(line)


# ============================================================================ match replay

def _new_match(line: dict) -> MLBMatch:
    return MLBMatch(
        seed=line["seed"], deck_key=line.get("deck_key", "b_red"),
        stake=line.get("stake", 1),
        lives=line.get("lives_start", 4),
        pvp_start_round=line.get("pvp_start_round", MLB_PVP_START_ROUND),
    )


def _drive_match(
    line: dict,
    on_pre: Optional[Callable[[int, Any, int, dict], None]] = None,
    on_post: Optional[Callable[[int, Any, int, dict], None]] = None,
    check: bool = True,
) -> MLBMatch:
    match = _new_match(line)
    sigs = line.get("signatures") or {}
    ops = line.get("ops") or []
    for idx, op in enumerate(ops):
        p, action = op["player"], op["action"]
        if on_pre is not None:
            on_pre(idx, match, p, action)
        match.step(p, action)
        if on_post is not None:
            on_post(idx, match, p, action)
        if check:
            key = str(idx)
            if key in sigs:
                got = match_sig_digest(match)
                if got != sigs[key]:
                    raise ReplayMismatch(idx, op, sigs[key], got)
    if check and "final" in sigs:
        got = match_sig_digest(match)
        if got != sigs["final"]:
            last = ops[-1] if ops else None
            raise ReplayMismatch(max(len(ops) - 1, 0), last, sigs["final"], got, final=True)
    return match


def replay_match(line: dict) -> MLBMatch:
    """Re-run a ``kind == "match"`` line (both players' ops, in their original interleaved
    order) and assert every logged signature matches.  Returns the final match."""
    return _drive_match(line)


def replay_line(line: dict):
    """Dispatch on ``line["kind"]``: episode -> replay(), match -> replay_match()."""
    kind = line.get("kind", "episode")
    if kind == "match":
        return replay_match(line)
    return replay(line)


def verify_file(path: str) -> dict:
    """Replay every line in ``path``.  Returns
    ``{"total", "ok", "mismatches": [(idx, ReplayMismatch), ...]}``."""
    lines = load_lines(path)
    mismatches = []
    for idx, line in enumerate(lines):
        try:
            replay_line(line)
        except ReplayMismatch as e:
            mismatches.append((idx, e))
    return {"total": len(lines), "ok": len(lines) - len(mismatches), "mismatches": mismatches}


# ============================================================================ narration

_BLIND_ACTIONS = {"play_blind", "skip_blind", "reroll_boss"}


def _fmt_money(delta: int) -> str:
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta}"


def narrate_episode(line: dict) -> str:
    out: list = []
    seed = line.get("seed", "?")
    ruleset = line.get("ruleset", "vanilla")
    deck = line.get("deck_key", "?")
    out.append(f"=== episode seed={seed} deck={deck} stake={line.get('stake')} "
               f"ruleset={ruleset} lives_start={line.get('lives_start')} ===")

    seen_ante = {"v": None}
    shop_before: dict = {}
    hand_before: dict = {}

    def on_pre(idx, game, action):
        ante = game.ante
        if ante != seen_ante["v"]:
            blind = game.current_blind
            tag = " (Nemesis)" if blind.is_pvp else ""
            out.append(f"--- Ante {ante}{tag} ---")
            seen_ante["v"] = ante
        atype = action.get("type", "")
        if atype in ("play", "discard"):
            idxs = action.get("cards", [])
            cards = [game.hand[i] for i in idxs if 0 <= i < len(game.hand)]
            hand_before[idx] = (list(cards), game.chips_scored, game.dollars)
        elif atype == "buy":
            i = action.get("item_idx", 0)
            item = game.current_shop[i] if 0 <= i < len(game.current_shop) else None
            shop_before[idx] = (item, game.dollars)
        elif atype in ("reroll", "leave_shop", "sell_joker"):
            shop_before[idx] = (None, game.dollars)

    def on_post(idx, game, action):
        atype = action.get("type", "")
        step = line["steps"][idx] if idx < len(line.get("steps", [])) else None
        if atype == "play":
            cards, chips_pre, money_pre = hand_before.get(idx, ([], 0, 0))
            if cards:
                try:
                    hand_type, _scoring = evaluate_hand(cards)
                except Exception:
                    hand_type = "?"
            else:
                hand_type = "?"
            gained = game.chips_scored - chips_pre
            out.append(f"[{idx}] played {hand_type} ({len(cards)} cards) "
                       f"-> chips {chips_pre} -> {game.chips_scored} "
                       f"(+{gained}, target {game.current_blind.chips_target})")
        elif atype == "discard":
            cards, _, _ = hand_before.get(idx, ([], 0, 0))
            out.append(f"[{idx}] discarded {len(cards)} card(s)")
        elif atype == "play_blind":
            blind = game.current_blind
            out.append(f"[{idx}] played blind {blind.kind}"
                        + (f" (boss {blind.boss_key})" if blind.boss_key else ""))
        elif atype == "skip_blind":
            out.append(f"[{idx}] skipped blind")
        elif atype == "buy":
            item, money_pre = shop_before.get(idx, (None, game.dollars))
            if item is not None:
                out.append(f"[{idx}] bought {item.kind} {item.key} "
                           f"(${money_pre} -> ${game.dollars})")
            else:
                out.append(f"[{idx}] bought item_idx={action.get('item_idx')}")
        elif atype == "reroll":
            _, money_pre = shop_before.get(idx, (None, game.dollars))
            out.append(f"[{idx}] rerolled shop (${money_pre} -> ${game.dollars})")
        elif atype == "leave_shop":
            out.append(f"[{idx}] left shop (${game.dollars})")
        elif atype == "sell_joker":
            out.append(f"[{idx}] sold joker slot {action.get('joker_idx')} "
                       f"(${game.dollars})")
        elif atype == "use_consumable":
            out.append(f"[{idx}] used consumable {action.get('consumable_idx')}")
        elif atype in ("pick_booster", "skip_booster"):
            out.append(f"[{idx}] {atype}")
        elif atype == "advance":
            out.append(f"[{idx}] advance (money ${game.dollars}, lives {game.lives})")
        elif atype == "__lose_life__":
            out.append(f"[{idx}] life lost externally (lives -> {game.lives})")
        if step is not None and step.get("is_pvp") and atype in ("play", "advance"):
            out.append(f"      Nemesis: chips_scored={game.chips_scored} "
                       f"hands_left={game.hands_left} lives={game.lives}")

    _drive_episode(line, on_pre=on_pre, on_post=on_post, check=False)

    outcome = line.get("outcome") or {}
    final_state = line.get("final_state") or {}
    out.append(f"=== outcome: {outcome} final_state: {final_state} ===")
    return "\n".join(out)


def narrate_match(line: dict) -> str:
    out: list = []
    out.append(f"=== match seed={line.get('seed')} deck={line.get('deck_key')} "
               f"stake={line.get('stake')} lives_start={line.get('lives_start')} "
               f"pvp_start_round={line.get('pvp_start_round')} ===")
    seen_ante = {0: None, 1: None}
    pvp_log_seen = {"n": 0}

    def on_post(idx, match, p, action):
        g = match.games[p]
        ante = g.ante
        if ante != seen_ante[p]:
            out.append(f"--- P{p} Ante {ante} ({g.current_blind.kind}) ---")
            seen_ante[p] = ante
        out.append(f"[{idx}] P{p} {action.get('type')} "
                   f"(money ${g.dollars}, lives {g.lives}, chips {g.chips_scored})")
        if len(match.pvp_log) > pvp_log_seen["n"]:
            for ante_, loser, s0, s1 in match.pvp_log[pvp_log_seen["n"]:]:
                verdict = "tie" if loser is None else f"P{loser} loses a life"
                out.append(f"      Nemesis ante {ante_} scores: P0={s0} P1={s1} -> {verdict}")
            pvp_log_seen["n"] = len(match.pvp_log)

    _drive_match(line, on_post=on_post, check=False)

    out.append(f"=== winner: {line.get('outcome', {}).get('winner')} "
               f"final_state: {line.get('final_state')} ===")
    return "\n".join(out)


def narrate(line: dict) -> str:
    """Dispatch on ``line["kind"]``: episode -> narrate_episode(), match -> narrate_match()."""
    kind = line.get("kind", "episode")
    if kind == "match":
        return narrate_match(line)
    return narrate_episode(line)
