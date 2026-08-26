"""
export_viz.py — best-effort export of a logged line into the ``trajectory.json`` shape the
V7-era ``viz/`` app renders.  That app lives in the predecessor repo
(https://github.com/taggarttufte/balatro-rl); nothing here reads it -- only its JSON shape
is targeted.  Confirmed against the shipped ``viz/trajectory.json`` + ``viz/main.js`` (2026-08-22):

    {"seed": ..., "outcome": {"ante", "reward", "steps", "dollars", "won"},
     "episode_id": ..., "trajectory": [ ...one entry per action, PRE-action state... ]}

Each ``trajectory[i]`` is the observation at the decision point (state BEFORE ``action`` is
applied) plus the chosen action -- confirmed by inspecting the shipped file: entry i's
``hand_cards``/``shop``/``jokers`` reflect the state the agent that action was chosen from,
not the result.

COVERAGE (what maps, what does not):
    maps directly:      step, phase, ante, blind_idx, money, chips_scored, hands_left,
                         discards_left, deck_size, hand_size, blind{name,kind,target,
                         is_boss,boss_key}, hand_cards[], jokers[], shop[], consumables[],
                         planet_levels, action (best-effort re-encoding, see below), outcome
    best-effort:         action.name / action.action int -- V7's action space was a fixed
                         56-dim encoding baked into its trained policy; this engine's action
                         dicts are re-encoded into the SAME two action "type"s the viewer
                         switches on ("hand": intent+subset; "phase": name+action-int, where
                         only buy actions get a real int (item_idx+2, matching main.js's
                         `chosenIdx = act - 2` shop highlight -- see renderShop); everything
                         else gets action=None and relies on `action.name` (the viewer's
                         fallback display path already used for reroll/leave_shop/etc.)
    NOT derivable here:  value_estimate, reward (per-step), top_probs -- these come from the
                         agent's MCTS/policy, which an engine-only trajectory log never has.
                         Always written as 0.0 / 0.0 / [] so the viewer still renders (it
                         reads them with `??`/`|| []` fallbacks throughout) but the value
                         panel and probability bars are meaningless for a replayed line.
    unrendered phases:   ROUND_EVAL / BOOSTER_OPEN / PVP_WAIT have no V7 phase and no
                         renderer branch in main.js (it only special-cases hand / shop /
                         blind_select / game_over); they still export with a best-guess
                         phase string ("round_eval" / "booster_open" / "pvp_wait") and the
                         viewer just shows the generic step info (phase pill, probs panel,
                         chosen action) with an empty stage area -- not broken, just plain.
    MLB-only fields (lives, is_pvp, comeback) have no V7 UI at all; not exported (the viewer
    predates MLB). Use ``cli.py show`` / ``narrate()`` for MLB-aware text narration instead.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from . import _bootstrap as _b
from .replay import _drive_episode, _drive_match

__all__ = ["export_viz", "export_viz_match", "export_viz_to_file"]

_PHASE_BY_STATE = {
    "BLIND_SELECT": "blind_select",
    "SELECTING_HAND": "hand",
    "ROUND_EVAL": "round_eval",
    "SHOP": "shop",
    "BOOSTER_OPEN": "booster_open",
    "GAME_OVER": "game_over",
    "PVP_WAIT": "pvp_wait",
}

_NAME_TABLES = {
    "joker": _b.game_keys.JOKER_NAME,
    "tarot": _b.game_keys.TAROT_NAME,
    "planet": _b.game_keys.PLANET_NAME,
    "spectral": _b.game_keys.SPECTRAL_NAME,
    "voucher": _b.game_keys.VOUCHER_NAME,
}


def _card_dict(c) -> dict:
    return {
        "rank": c.rank, "suit": c.suit, "enhancement": c.enhancement,
        "edition": c.edition, "seal": c.seal, "debuffed": bool(c.debuffed),
    }


def _joker_dict(j) -> dict:
    return {
        "key": j.key, "name": _b.game_keys.JOKER_NAME.get(j.key, j.key),
        "edition": j.edition, "state": dict(j.state),
    }


def _shop_dict(item) -> dict:
    table = _NAME_TABLES.get(item.kind, {})
    name = table.get(item.key, item.key)
    return {
        "kind": item.kind, "key": item.key, "name": name, "price": item.price,
        "edition": item.edition, "sold": bool(item.sold),
    }


def _blind_dict(blind, ante: int) -> dict:
    name = f"Ante {ante} {blind.kind}"
    if blind.boss_key:
        boss_name = _b.game_keys.BOSS_NAME.get(blind.boss_key, blind.boss_key)
        name = f"{name} ({boss_name})"
    return {
        "name": name, "kind": blind.kind, "target": blind.chips_target,
        "is_boss": bool(blind.is_boss), "boss_key": blind.boss_key or "",
    }


def _translate_action(game, action: dict) -> dict:
    atype = action.get("type", "")
    if atype in ("play", "discard"):
        return {"type": "hand", "intent": atype, "subset": list(action.get("cards", []))}
    if atype == "buy":
        idx = action.get("item_idx", 0)
        item = game.current_shop[idx] if 0 <= idx < len(game.current_shop) else None
        name = f"buy {item.key}" if item is not None else f"buy item_idx={idx}"
        return {"type": "phase", "action": idx + 2, "name": name}
    if atype == "skip_blind":
        return {"type": "phase", "action": 1, "name": "skip_blind"}
    if atype == "play_blind":
        return {"type": "phase", "action": 0, "name": "select_blind"}
    return {"type": "phase", "action": None, "name": atype or "advance"}


def _v7_step(step_idx: int, game, action: dict) -> dict:
    ante = game.ante
    blind = game.current_blind
    return {
        "step": step_idx,
        "phase": _PHASE_BY_STATE.get(game.state.name, game.state.name.lower()),
        "ante": ante,
        "blind_idx": game.blind_idx,
        "money": game.dollars,
        "chips_scored": game.chips_scored,
        "hands_left": game.hands_left,
        "discards_left": game.discards_left,
        "deck_size": len(game.deck),
        "hand_size": game.hand_size,
        "blind": _blind_dict(blind, ante),
        "hand_cards": [_card_dict(c) for c in game.hand],
        "jokers": [_joker_dict(j) for j in game.jokers],
        "shop": [_shop_dict(it) for it in game.current_shop],
        "consumables": list(game.consumable_hand),
        "planet_levels": dict(game.planet_levels),
        "value_estimate": 0.0,   # not derivable — see module docstring
        "action": _translate_action(game, action),
        "top_probs": [],        # not derivable — see module docstring
        "reward": 0.0,           # not derivable — see module docstring
    }


def export_viz(line: dict, episode_id: int = 0) -> dict:
    """``kind == "episode"`` line -> the V7 ``trajectory.json`` dict.  Drives the engine via
    ``replay._drive_episode`` (signature checking disabled: this is a rendering export, not a
    determinism assertion -- use ``replay.replay()``/``cli.py verify`` for that)."""
    entries: list = []

    def on_pre(idx, game, action):
        entries.append(_v7_step(idx, game, action))

    game = _drive_episode(line, on_pre=on_pre, check=False)
    outcome_in = line.get("outcome") or {}
    fs = line.get("final_state") or {}
    outcome = {
        "ante": fs.get("ante", game.ante),
        "reward": 0.0,
        "steps": len(entries),
        "dollars": fs.get("money", game.dollars),
        "won": bool(outcome_in.get("won", False)),
    }
    return {"seed": line.get("seed"), "outcome": outcome, "trajectory": entries,
            "episode_id": episode_id}


def export_viz_match(line: dict, player: int, episode_id: int = 0) -> dict:
    """``kind == "match"`` line, ``player``'s side only (the viewer has no two-board mode):
    replays the full match but only records an entry at ``player``'s own decision points."""
    entries: list = []

    def on_pre(idx, match, p, action):
        if p != player:
            return
        entries.append(_v7_step(len(entries), match.games[player], action))

    match = _drive_match(line, on_pre=on_pre, check=False)
    fs_players = (line.get("final_state") or {}).get("players") or [{}, {}]
    fs = fs_players[player] if player < len(fs_players) else {}
    outcome_in = line.get("outcome") or {}
    g = match.games[player]
    outcome = {
        "ante": fs.get("ante", g.ante),
        "reward": 0.0,
        "steps": len(entries),
        "dollars": fs.get("money", g.dollars),
        "won": outcome_in.get("winner") == player,
    }
    seed = line.get("seed")
    return {"seed": f"{seed}#p{player}", "outcome": outcome, "trajectory": entries,
            "episode_id": episode_id}


def export_viz_to_file(line: dict, path: str, episode_id: int = 0,
                        player: Optional[int] = None) -> dict:
    """Write the export to ``path`` (the shape the predecessor repo's ``viz/index.html`` loads as
    ``trajectory.json`` -- see replay/REPLAY_NOTES.md for how to point the old viewer at
    it; that app is never modified by this package)."""
    if line.get("kind") == "match":
        p = 0 if player is None else player
        doc = export_viz_match(line, p, episode_id=episode_id)
    else:
        doc = export_viz(line, episode_id=episode_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    return doc
