"""
test_env_rng_isolation.py — the envs' reward/ranking estimates must not touch
game state, must not touch the process-global `random`, and must be
deterministic (MP_UPDATE_LIST_2026-08 §3, workstream W7 / P1-repro).

Before 2026-08-21:
  * env_v7._best_hand_score scored up to 8 hypothetical hands with `rng=gs.rng`,
    so the seeded stream advanced by a hand-dependent amount BEFORE the real
    play; scaling jokers were bumped per candidate and Space Joker levelled
    hands for free.
  * env_sim / env_v5 _update_play_combos scored all ~218 subsets with NO rng=
    (-> unseeded global `random`) on the live jokers / planet dict.

All three now go through card_selection.HypotheticalScorer.
"""
from __future__ import annotations

import ast
import copy
import random
from pathlib import Path

import numpy as np
import pytest

from balatro_sim.game import BalatroGame, State
from balatro_sim.jokers.base import JokerInstance
from balatro_sim.card_selection import (
    HypotheticalScorer, enumerate_subsets, INTENT_PLAY, INTENT_DISCARD,
)
from balatro_sim.hand_eval import evaluate_hand
from balatro_sim.env_v7 import BalatroV7Env, PHASE_SELECTING_HAND
from balatro_sim.env_sim import BalatroSimEnv
from balatro_sim.env_v5 import BalatroSimEnvV5, SUBSTATE_PACK_OPEN, SUBSTATE_PACK_TARGET


ENGINE_ROOT = Path(__file__).resolve().parents[2]
SIM_DIR = ENGINE_ROOT / "balatro_sim"

# Every stochastic / state-mutating scoring hook the estimate can hit:
#   Bloodstone, Business Card, Misprint, 8 Ball, Space Joker  -> rng rolls
#   Space Joker                                               -> planet_levels write
#   Ride the Bus, Card Sharp (nested set), Vampire (card.enhancement)
#                                                             -> inst.state / card mutation
STOCHASTIC_JOKERS = [
    "j_bloodstone", "j_business", "j_misprint", "j_space", "j_8_ball",
]
MUTATING_JOKERS = ["j_ride_the_bus", "j_card_sharp", "j_vampire"]


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _to_selecting_hand(gs: BalatroGame) -> BalatroGame:
    for _ in range(4):
        if gs.state == State.SELECTING_HAND:
            return gs
        if gs.state == State.BLIND_SELECT:
            gs.step({"type": "play_blind"})
        else:
            raise RuntimeError(f"unexpected state {gs.state}")
    assert gs.state == State.SELECTING_HAND
    return gs


def _rig(gs: BalatroGame) -> BalatroGame:
    """
    Put a game in SELECTING_HAND with a hand and joker board that makes every
    known side effect fire: Lucky cards (rng), Hearts + face cards (Bloodstone,
    Business Card, Vampire), an 8 (8 Ball), and the full stochastic/mutating
    joker set. Deterministic given the game's seed.
    """
    _to_selecting_hand(gs)
    gs.jokers = [JokerInstance(k) for k in STOCHASTIC_JOKERS + MUTATING_JOKERS]
    # Card Sharp keeps a nested set in state — the shallow clone would share it.
    gs.jokers[-2].state["played_hands"] = {"High Card"}
    for i, c in enumerate(gs.hand):
        if i % 2 == 0:
            c.enhancement = "Lucky"
        if i in (1, 3):
            c.rank, c.suit = 13, "Hearts"
        if i == 5:
            c.rank = 8
    return gs


def _prng_state(gs: BalatroGame):
    """Hashable image of the keyed PseudoRandom (W3: game.run_state.rng, no game.rng)."""
    snap = gs.run_state.rng.snapshot()
    return (snap["seed"], tuple(sorted(snap["state"].items())), repr(snap["rng"]))


def _snapshot(gs: BalatroGame) -> dict:
    """Everything a dry-run scoring pass could conceivably touch."""
    return {
        "rng": _prng_state(gs),
        "jokers": [(j.key, j.edition, copy.deepcopy(j.state)) for j in gs.jokers],
        "planet_levels": dict(gs.planet_levels),
        "dollars": gs.dollars,
        "consumable_hand": list(gs.consumable_hand),
        "hand": [(c.id, c.rank, c.suit, c.enhancement, c.edition, c.seal, c.debuffed)
                 for c in gs.hand],
        "full_deck": [(c.id, c.enhancement, c.edition, c.seal, c.debuffed)
                      for c in gs.full_deck],
        "deck_len": len(gs.deck),
        "chips_scored": gs.chips_scored,
        "hands_left": gs.hands_left,
        "discards_left": gs.discards_left,
        "hand_type_counts": dict(gs._hand_type_counts),
    }


def _fingerprint(gs: BalatroGame) -> tuple:
    """Cheap full-state fingerprint for trajectory comparison."""
    return (
        gs.state.name, gs.ante, gs.blind_idx, gs.dollars, gs.chips_scored,
        gs.hands_left, gs.discards_left,
        tuple((j.key, j.edition, repr(sorted(j.state.items(), key=repr))) for j in gs.jokers),
        tuple(sorted(gs.planet_levels.items())),
        tuple(repr(c) for c in gs.hand),
        tuple(repr(c) for c in gs.deck),
        tuple(gs.consumable_hand),
        _prng_state(gs),
    )


# ────────────────────────────────────────────────────────────────────────────
# 1. The estimate path must not mutate anything, and must be deterministic
# ────────────────────────────────────────────────────────────────────────────

def _v7_estimate(env):
    return env._best_hand_score(env.game)


def _combo_estimate(env):
    env._update_play_combos()
    return [tuple(c) for c in env._play_combos]


ENV_CASES = [
    pytest.param(BalatroV7Env, _v7_estimate, id="env_v7"),
    pytest.param(BalatroSimEnv, _combo_estimate, id="env_sim"),
    pytest.param(BalatroSimEnvV5, _combo_estimate, id="env_v5"),
]


@pytest.mark.parametrize("env_cls,estimate", ENV_CASES)
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_estimate_has_no_side_effects(env_cls, estimate, seed):
    env = env_cls(seed=seed)
    env.reset()
    _rig(env.game)

    global_before = random.getstate()
    before = _snapshot(env.game)
    estimate(env)
    after = _snapshot(env.game)
    global_after = random.getstate()

    for key in before:
        assert after[key] == before[key], f"{env_cls.__name__}: {key} changed"
    assert global_after == global_before, \
        f"{env_cls.__name__}: process-global random state advanced"


@pytest.mark.parametrize("env_cls,estimate", ENV_CASES)
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_estimate_is_deterministic(env_cls, estimate, seed):
    env = env_cls(seed=seed)
    env.reset()
    _rig(env.game)
    e1 = estimate(env)
    e2 = estimate(env)
    e3 = estimate(env)
    assert e1 == e2 == e3


def test_v7_estimate_is_positive_with_stochastic_board():
    """Sanity: the rigged board actually exercises the scorer (non-zero)."""
    env = BalatroV7Env(seed=3)
    env.reset()
    _rig(env.game)
    assert env._best_hand_score(env.game) > 0


@pytest.mark.parametrize("seed", [11, 12, 13])
def test_v7_reward_path_does_not_advance_rng_before_real_play(seed):
    """
    The real play must see exactly the stream it would have seen had the
    reward estimate never run: build two identical envs, play the same subset
    on (a) the raw game (no estimate) and (b) via step_hand (estimate first),
    and compare the games afterwards.

    Two independent envs rather than game.clone(): JokerInstance.clone() is a
    shallow state copy, so a clone would SHARE Card Sharp's played_hands set
    with the original and the comparison would be contaminated.
    """
    ref = BalatroV7Env(seed=seed); ref.reset(); _rig(ref.game)
    env = BalatroV7Env(seed=seed); env.reset(); _rig(env.game)
    assert _fingerprint(ref.game) == _fingerprint(env.game)
    subset = (0, 1, 2, 3, 4)

    ref.game.step({"type": "play", "cards": list(subset)})
    ref._auto_advance()

    env.step_hand(INTENT_PLAY, subset)

    assert _fingerprint(env.game) == _fingerprint(ref.game)


# ────────────────────────────────────────────────────────────────────────────
# 2. HypotheticalScorer unit behaviour
# ────────────────────────────────────────────────────────────────────────────

def _scorer_inputs(gs, combo):
    cards = [gs.hand[i] for i in combo]
    ht, sc = evaluate_hand(cards)
    return cards, ht, sc


def test_scorer_isolates_nested_joker_state():
    """Card Sharp mutates a set nested in inst.state; the dry run must not leak."""
    gs = BalatroGame(seed=5)
    gs.reset()
    _rig(gs)
    sharp = next(j for j in gs.jokers if j.key == "j_card_sharp")
    played_before = set(sharp.state["played_hands"])
    scorer = HypotheticalScorer(gs)
    for combo in enumerate_subsets(len(gs.hand))[:40]:
        scorer.score(*_scorer_inputs(gs, combo))
    assert sharp.state["played_hands"] == played_before


def test_scorer_isolates_planet_levels_from_space_joker():
    gs = BalatroGame(seed=9)
    gs.reset()
    _to_selecting_hand(gs)
    gs.jokers = [JokerInstance("j_space")]
    before = dict(gs.planet_levels)
    scorer = HypotheticalScorer(gs)
    # 200 candidates at 1/4 each: a leak would be caught with overwhelming odds.
    for combo in enumerate_subsets(len(gs.hand)):
        scorer.score(*_scorer_inputs(gs, combo))
    assert gs.planet_levels == before


def test_scorer_isolates_card_mutation_from_vampire():
    gs = BalatroGame(seed=9)
    gs.reset()
    _to_selecting_hand(gs)
    gs.jokers = [JokerInstance("j_vampire")]
    for c in gs.hand:
        c.enhancement = "Lucky"
    scorer = HypotheticalScorer(gs, model_held=True)
    for combo in enumerate_subsets(len(gs.hand)):
        scorer.score(*_scorer_inputs(gs, combo))
    assert all(c.enhancement == "Lucky" for c in gs.hand)
    assert all(j.state == {} for j in gs.jokers)


def test_scorer_candidate_order_does_not_matter():
    """Every candidate sees the same snapshot, so scores are order-independent."""
    gs = BalatroGame(seed=21)
    gs.reset()
    _rig(gs)
    combos = enumerate_subsets(len(gs.hand))
    fwd = HypotheticalScorer(gs, model_held=True)
    rev = HypotheticalScorer(gs, model_held=True)
    a = {c: fwd.score(*_scorer_inputs(gs, c)) for c in combos}
    b = {c: rev.score(*_scorer_inputs(gs, c)) for c in reversed(combos)}
    assert a == b


def test_scorer_is_seeded_from_game_rng():
    """Same game state -> same estimate; a different stream position may differ."""
    g1 = BalatroGame(seed=33); g1.reset(); _rig(g1)
    g2 = BalatroGame(seed=33); g2.reset(); _rig(g2)
    combo = enumerate_subsets(len(g1.hand))[-1]
    s1 = HypotheticalScorer(g1, model_held=True).score(*_scorer_inputs(g1, combo))
    s2 = HypotheticalScorer(g2, model_held=True).score(*_scorer_inputs(g2, combo))
    assert s1 == s2
    # Burn g2's 'misprint' stream: the scorer must read the live per-key
    # positions, not a constant, so Misprint's +0..23 (rolled on every candidate)
    # should now differ for at least one of many candidates.
    g2.run_state.rng.pseudorandom("misprint", 0, 23)
    diffs = 0
    for c in enumerate_subsets(len(g1.hand)):
        if (HypotheticalScorer(g1).score(*_scorer_inputs(g1, c))
                != HypotheticalScorer(g2).score(*_scorer_inputs(g2, c))):
            diffs += 1
    assert diffs > 0


# ────────────────────────────────────────────────────────────────────────────
# 3. Same seed + same actions -> identical trajectories
# ────────────────────────────────────────────────────────────────────────────

def _drive_v7(env, script: random.Random, n_steps: int):
    """Scripted policy for env_v7: the script RNG is private to the test."""
    out = []
    obs, _ = env.reset()
    _rig(env.game)
    for _ in range(n_steps):
        if env.game.state == State.GAME_OVER:
            break
        if env.get_phase() == PHASE_SELECTING_HAND:
            subsets = enumerate_subsets(len(env.game.hand))
            subset = script.choice(subsets)
            intent = INTENT_DISCARD if (script.random() < 0.25
                                        and env.game.discards_left > 0) else INTENT_PLAY
            obs, r, term, trunc, info = env.step_hand(intent, subset)
        else:
            mask = env.get_phase_mask()
            valid = np.flatnonzero(mask)
            if len(valid) == 0:
                break
            action = int(script.choice(list(valid)))
            obs, r, term, trunc, info = env.step_phase(action)
        out.append((np.asarray(obs).tobytes(), round(float(r), 9), _fingerprint(env.game)))
        if term or trunc:
            break
    return out


def _drive_sim(env, script: random.Random, n_steps: int):
    out = []
    obs, _ = env.reset()
    _rig(env.game)
    env._update_play_combos()
    for _ in range(n_steps):
        gs = env.game
        if gs.state == State.GAME_OVER:
            break
        if gs.state == State.BLIND_SELECT:
            action = 30
        elif gs.state == State.SELECTING_HAND:
            if script.random() < 0.25 and gs.discards_left > 0 and gs.hand:
                action = 20 + script.randrange(len(gs.hand))
            else:
                action = script.randrange(max(1, min(20, len(env._play_combos))))
        else:  # SHOP
            action = 45 if script.random() < 0.7 else script.choice([32, 33, 44, 45])
        obs, r, term, trunc, info = env.step(action)
        out.append((np.asarray(obs).tobytes(), round(float(r), 9), _fingerprint(env.game)))
        if term or trunc:
            break
    return out


def _drive_v5(env, script: random.Random, n_steps: int):
    out = []
    obs, info = env.reset()
    _rig(env.game)
    env._update_play_combos()
    for _ in range(n_steps):
        gs = env.game
        if gs.state == State.GAME_OVER:
            break
        if info.get("agent") == "play":
            mask = env.get_play_action_mask()
        elif info.get("shop_substate") == SUBSTATE_PACK_OPEN:
            mask = env.get_pack_open_mask()
        elif info.get("shop_substate") == SUBSTATE_PACK_TARGET:
            mask = env.get_pack_target_mask()
        else:
            mask = env.get_shop_action_mask()
        valid = np.flatnonzero(mask)
        if len(valid):
            action = int(script.choice(list(valid)))
        else:
            # Mirrors env_v5's own smoke test: 30 = play_blind/advance, 1 = leave.
            action = 30 if info.get("agent") == "play" else 1
        obs, r, term, trunc, info = env.step(action)
        out.append((np.asarray(obs).tobytes(), round(float(r), 9), _fingerprint(env.game)))
        if term or trunc:
            break
    return out


@pytest.mark.parametrize("env_cls,driver", [
    pytest.param(BalatroV7Env, _drive_v7, id="env_v7"),
    pytest.param(BalatroSimEnv, _drive_sim, id="env_sim"),
    pytest.param(BalatroSimEnvV5, _drive_v5, id="env_v5"),
])
@pytest.mark.parametrize("seed", [0, 42])
def test_same_seed_same_actions_same_trajectory(env_cls, driver, seed):
    """
    Two envs with the same seed, driven by identical scripted action
    sequences, must produce identical (obs, reward, game state) at every step.
    Pre-fix this failed for env_sim/env_v5 because the combo ranking (and thus
    the meaning of each action index) depended on the unseeded global random.
    """
    # Poison the global random differently before each run: a fallback to it
    # would now diverge even more reliably.
    random.seed(1)
    traj_a = driver(env_cls(seed=seed), random.Random(1234), 60)
    random.seed(2)
    traj_b = driver(env_cls(seed=seed), random.Random(1234), 60)
    assert len(traj_a) > 5
    assert len(traj_a) == len(traj_b)
    for i, (a, b) in enumerate(zip(traj_a, traj_b)):
        assert a == b, f"{env_cls.__name__}: trajectories diverge at step {i}"


# ────────────────────────────────────────────────────────────────────────────
# 4. Static guard: every score_hand() call in the env layer passes rng=
# ────────────────────────────────────────────────────────────────────────────

def _score_hand_calls_without_rng(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.id if isinstance(fn, ast.Name) else (
            fn.attr if isinstance(fn, ast.Attribute) else None)
        if name != "score_hand":
            continue
        if not any(kw.arg == "rng" for kw in node.keywords):
            bad.append(node.lineno)
    return bad


@pytest.mark.parametrize("filename", [
    "env_v7.py", "env_sim.py", "env_v5.py", "env_mp.py", "card_selection.py",
])
def test_no_score_hand_call_without_explicit_rng(filename):
    path = SIM_DIR / filename
    assert path.exists(), path
    bad = _score_hand_calls_without_rng(path)
    assert not bad, f"{filename}: score_hand() without rng= at lines {bad}"


def test_env_layer_never_passes_live_game_rng_to_score_hand():
    """
    The only score_hand() call in the env layer lives in HypotheticalScorer and
    passes its private RNG. Guard against someone re-introducing `rng=gs.rng`
    in an env file.
    """
    for filename in ("env_v7.py", "env_sim.py", "env_v5.py", "env_mp.py"):
        src = (SIM_DIR / filename).read_text(encoding="utf-8")
        tree = ast.parse(src)
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and getattr(n.func, "id", getattr(n.func, "attr", None)) == "score_hand"]
        assert calls == [], f"{filename} calls score_hand() directly at " \
                            f"{[c.lineno for c in calls]}; use HypotheticalScorer"


def test_env_layer_does_not_import_global_random_for_game_logic():
    """env_*.py must not use the `random` module outside __main__/benchmark code."""
    for filename in ("env_v7.py", "env_sim.py", "env_v5.py", "env_mp.py"):
        tree = ast.parse((SIM_DIR / filename).read_text(encoding="utf-8"))
        top_level_random = [
            n for n in tree.body
            if isinstance(n, (ast.Import, ast.ImportFrom))
            and any((a.name == "random") for a in n.names)
        ]
        assert not top_level_random, f"{filename} imports random at module level"
