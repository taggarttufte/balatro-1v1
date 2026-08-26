"""
test_action_space.py — the agent's action vocabulary must match the FORK engine's.

The balatro-mcts action encoding was written against the pre-rekey engine. This file is
the pin that stops the two drifting apart again:

  * every action type the fork's `legal_actions()` can emit is in ACTION_TYPES
  * action_key -> action_from_key round-trips to something `step()` accepts
  * keys are unique per legal action, and so are the feature rows
  * the fork's game keys (`j_*`, `c_*`, `v_*`, `bl_*`) are what the encoder reads

Also asserts the fork guard: `balatro_sim`, `mcts` and `train` must resolve inside
engine and agent, never to any other ``balatro_sim`` on sys.path.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import balatro_sim
import mcts as mcts_pkg
import train as train_pkg
from balatro_sim.game import BalatroGame, State
from mcts.action import action_from_key, action_key
from mcts.action_features import ACTION_TYPE_IDX, featurize_actions

AGENT_ROOT = Path(__file__).resolve().parents[1]
ENGINE_ROOT = AGENT_ROOT.parent / "engine"


# ── Fork guard ──────────────────────────────────────────────────────────────

def test_imports_resolve_to_the_fork():
    assert Path(balatro_sim.__file__).resolve() == ENGINE_ROOT / "balatro_sim" / "__init__.py"
    assert Path(mcts_pkg.__file__).resolve() == AGENT_ROOT / "mcts" / "__init__.py"
    assert Path(train_pkg.__file__).resolve() == AGENT_ROOT / "train" / "__init__.py"


def test_engine_speaks_game_keys():
    """Phase 1 re-keyed the engine (`REKEY_NOTES.md`); the encoder reads the catalogue by
    those keys, so a regression here silently zeroes the joker features."""
    from balatro_sim.shop import JOKER_CATALOGUE
    assert len(JOKER_CATALOGUE) == 150
    assert all(k.startswith("j_") for k in JOKER_CATALOGUE)
    assert "j_joker" in JOKER_CATALOGUE and "j_blueprint" in JOKER_CATALOGUE


# ── Walk real states and check every action ─────────────────────────────────

def _walk_states(game: BalatroGame, max_steps: int = 120):
    """Yield (state, legal_actions) while playing a simple always-win policy, so the
    walk reaches SHOP / BOOSTER_OPEN / ROUND_EVAL and (under MLB) the Nemesis."""
    rng = np.random.default_rng(0)
    for _ in range(max_steps):
        legal = game.legal_actions()
        yield game, legal
        if game.state is State.GAME_OVER or not legal:
            return
        if game.state is State.SELECTING_HAND:
            game.debug_win_blind()
            continue
        if game.state is State.SHOP:
            # Buy something now and then so booster packs / sells become reachable.
            buys = [a for a in legal if a["type"] == "buy"]
            act = buys[int(rng.integers(len(buys)))] if buys and rng.random() < 0.6 \
                else {"type": "leave_shop"}
            game.step(act)
            continue
        game.step(legal[0])


@pytest.mark.parametrize("ruleset", ["vanilla", "mlb"])
def test_every_legal_action_type_is_in_the_vocabulary(ruleset):
    seen = set()
    for seed in ("7I4M53DL", "ALEEB", "1MD1YZ9T"):
        game = BalatroGame(seed=seed, ruleset=ruleset)
        for _g, legal in _walk_states(game):
            for a in legal:
                assert a["type"] in ACTION_TYPE_IDX, f"unknown action type {a['type']!r}"
                seen.add(a["type"])
    # The walk must actually have reached the interesting states.
    assert {"play_blind", "skip_blind", "play", "discard", "use_consumable",
            "buy", "sell_joker", "reroll", "leave_shop", "pick_booster",
            "skip_booster", "advance"} <= seen, seen


def test_reroll_boss_is_reachable_and_covered():
    """`reroll_boss` is the action type the fork added (Directors Cut / Retcon). Seed
    7I4M53DL buys into it during the walk; if the vocabulary lost it, the featurizer
    would emit the all-zero unknown row for a real action."""
    seen = set()
    for _g, legal in _walk_states(BalatroGame(seed="7I4M53DL")):
        seen |= {a["type"] for a in legal}
    assert "reroll_boss" in seen


@pytest.mark.parametrize("ruleset", ["vanilla", "mlb"])
def test_action_keys_round_trip_and_step(ruleset):
    """key(a) -> from_key -> step must be accepted by the engine for EVERY legal action
    (checked on a clone so the walk itself is unaffected)."""
    checked = 0
    game = BalatroGame(seed="7I4M53DL", ruleset=ruleset)
    for g, legal in _walk_states(game, max_steps=40):
        for a in legal[:25]:            # cap: SELECTING_HAND has ~436 of them
            k = action_key(a)
            back = action_from_key(k)
            assert action_key(back) == k
            assert back["type"] == a["type"]
            clone = g.clone()
            clone.step(back)            # must not raise
            checked += 1
    assert checked > 50


def test_keys_are_unique_across_a_legal_action_list():
    """Two legal actions collapsing to one key would silently delete a branch."""
    for seed in ("7I4M53DL", "ALEEB"):
        game = BalatroGame(seed=seed)
        for _g, legal in _walk_states(game, max_steps=30):
            keys = [action_key(a) for a in legal]
            assert len(set(keys)) == len(keys)


def test_feature_rows_are_unique_across_a_legal_action_list():
    for _g, legal in _walk_states(BalatroGame(seed="7I4M53DL"), max_steps=30):
        if not legal:
            continue
        feats = featurize_actions(legal)
        assert len({f.tobytes() for f in feats}) == len(legal)


def test_consumable_target_actions_keep_their_targets():
    """The A1 audit bug (targets dropped, consumable slot 1 unreachable) must not come
    back through the key encoding."""
    a = {"type": "use_consumable", "consumable_idx": 1, "target_cards": [3, 1]}
    k = action_key(a)
    assert k == ("use_consumable", 1, (1, 3))
    back = action_from_key(k)
    assert back["consumable_idx"] == 1 and sorted(back["target_cards"]) == [1, 3]
