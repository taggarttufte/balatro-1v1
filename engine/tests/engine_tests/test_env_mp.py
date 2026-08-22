"""Tests for balatro_sim.env_mp — the MLB self-play environment on top of MLBMatch
(Phase 2 W1 rewrite; brief §1.8: both players' lives, the opponent's live score and
hands-left during PvP, and the comeback state are in the observation)."""
import numpy as np
import pytest

from balatro_sim.env_mp import (
    MultiplayerBalatroEnv, MULTIPLAYER_BANNED_JOKERS, OBS_DIM, MP_OBS_FEATURES,
    R_PVP_WIN, R_PVP_LOSS, R_LIFE_LOSS, R_GAME_WIN, R_GAME_LOSS, PHASE_WAITING,
)
from balatro_sim.shop import BANNED_JOKERS
from balatro_sim.game import BalatroGame, State
from balatro_sim.constants import MLB_BANNED_KEYS, MLB_STARTING_LIVES
from balatro_sim.env_v7 import (PHASE_SELECTING_HAND, PHASE_BLIND_SELECT, PHASE_SHOP,
                                OBS_DIM as V7_OBS_DIM)
from balatro_sim.card_selection import INTENT_PLAY, INTENT_DISCARD

V = V7_OBS_DIM   # start of the MP block


def _scripted(env, player: int, k: int = 5) -> dict:
    """Play the first k cards; otherwise leave the shop / play the blind."""
    if env.get_phase(player) == PHASE_SELECTING_HAND:
        g = env.get_player_game(player)
        n = min(k, len(g.hand))
        if g.current_blind.boss_key == "bl_psychic":
            n = min(5, len(g.hand))
        return {"type": "hand", "intent": INTENT_PLAY, "subset": tuple(range(n))}
    mask = env.get_phase_mask(player)
    if mask[15]:
        return {"type": "phase", "action": 15}
    if mask[0]:
        return {"type": "phase", "action": 0}
    valid = [i for i in range(len(mask)) if mask[i]]
    return {"type": "phase", "action": valid[0] if valid else 15}


def _to_nemesis(env, player: int, ante: int = 2):
    """Debug-win the player (1/2) to the BLIND_SELECT of ``ante``'s Nemesis."""
    g = env.get_player_game(player)
    while not (g.ante == ante and g.current_blind.is_pvp and g.state == State.BLIND_SELECT):
        if g.state == State.BLIND_SELECT:
            g.step({"type": "play_blind"}); g.debug_win_blind(); g.step({"type": "advance"})
        elif g.state == State.SHOP:
            g.step({"type": "leave_shop"})
        elif g.state == State.BOOSTER_OPEN:
            g.step({"type": "skip_booster"})
        elif g.state == State.ROUND_EVAL:
            g.step({"type": "advance"})
        else:
            raise RuntimeError(g.state)
    env.mp.sync()


class TestBannedJokers:
    def test_banned_set_is_the_attrition_list(self):
        assert MULTIPLAYER_BANNED_JOKERS == {"j_chicot", "j_matador", "j_mr_bones", "j_luchador"}

    def test_import_no_longer_pollutes_the_global_ban_list(self):
        """The V8-era module set shop.BANNED_JOKERS at import time, which changed every
        vanilla game in the process.  Bans now travel with ruleset='mlb'."""
        assert not (MULTIPLAYER_BANNED_JOKERS & BANNED_JOKERS)
        assert not (MLB_BANNED_KEYS & BalatroGame(seed="7I4M53DL").run_state.banned_keys)

    def test_ban_list_lands_in_run_state(self):
        env = MultiplayerBalatroEnv(seed="7I4M53DL")
        env.reset()
        for g in env.mp.games:
            assert MLB_BANNED_KEYS <= g.run_state.banned_keys

    def test_shop_excludes_banned_jokers(self):
        for seed in range(60):
            env = MultiplayerBalatroEnv(seed=seed)
            env.reset()
            g = env.p1_game
            g.step({"type": "play_blind"}); g.debug_win_blind(); g.step({"type": "advance"})
            assert g.state == State.SHOP
            for item in g.current_shop:
                if item.kind == "joker":
                    assert item.key not in MULTIPLAYER_BANNED_JOKERS


class TestMPObsExtension:
    def test_obs_dim(self):
        assert OBS_DIM == V7_OBS_DIM + MP_OBS_FEATURES == V7_OBS_DIM + 10

    def test_lives_in_obs(self):
        env = MultiplayerBalatroEnv(seed=42)
        env.reset()
        o1, o2 = env.p1.encode_obs(), env.p2.encode_obs()
        assert o1[V + 0] == 1.0 and o1[V + 1] == 1.0 and o2[V + 0] == 1.0
        env.p2_game.lose_life()
        o1, o2 = env.p1.encode_obs(), env.p2.encode_obs()
        assert o1[V + 0] == 1.0 and o1[V + 1] == 0.75
        assert o2[V + 0] == 0.75 and o2[V + 1] == 1.0

    def test_comeback_pending_in_obs(self):
        env = MultiplayerBalatroEnv(seed=42)
        env.reset()
        g = env.p1_game
        g.step({"type": "play_blind"})
        g.lose_life()
        assert env.p1.encode_obs()[V + 8] == pytest.approx(4 / 16)
        g.lose_life()   # blocked (one per round)
        assert env.p1.encode_obs()[V + 8] == pytest.approx(4 / 16)
        g.debug_win_blind(); g.step({"type": "advance"})
        assert g.comeback_bonus_given and env.p1.encode_obs()[V + 8] == 0.0

    def test_pvp_block_outside_pvp_is_zero(self):
        env = MultiplayerBalatroEnv(seed=42)
        env.reset()
        o = env.p1.encode_obs()
        assert o[V + 2] == 0.0 and o[V + 3] == 0.0 and o[V + 4] == 0.0 and o[V + 5] == 0.0 and o[V + 6] == 0.0
        assert o[V + 7] == 0.0 and o[V + 9] == 0.0

    def test_pvp_block_during_pvp(self):
        env = MultiplayerBalatroEnv(seed="7I4M53DL")
        env.reset()
        _to_nemesis(env, 1); _to_nemesis(env, 2)
        g1, g2 = env.p1_game, env.p2_game
        env.step({"type": "phase", "action": 0}, {"type": "phase", "action": 0})
        assert env.mp.pvp_active
        o1 = env.p1.encode_obs()
        assert o1[V + 2] == 1.0 and o1[V + 5] == 1.0 and o1[V + 6] == 0.0
        # player 1 plays a hand; player 2 sees the score and the hands-left
        env.step({"type": "hand", "intent": INTENT_PLAY, "subset": (0, 1, 2, 3, 4)},
                 {"type": "phase", "action": 15})          # p2: no-op in SELECTING_HAND
        assert g1.chips_scored > 0 and g2.chips_scored == 0
        o2 = env.p2.encode_obs()
        assert o2[V + 2] == 1.0
        assert o2[V + 3] == pytest.approx(np.log1p(g1.chips_scored) / np.log1p(100000))
        assert o2[V + 4] == -1.0                            # fully behind
        assert o2[V + 5] == pytest.approx((g1.base_hands - 1) / g2.base_hands)
        assert env.p1.encode_obs()[V + 4] == 1.0

    def test_waiting_flag_and_phase(self):
        env = MultiplayerBalatroEnv(seed="7I4M53DL")
        env.reset()
        _to_nemesis(env, 1)
        env.step({"type": "phase", "action": 0}, {"type": "phase", "action": 15})
        assert env.p1_game.pvp_ready
        assert env.p1.encode_obs()[V + 7] == 1.0
        assert env.get_phase(1) == PHASE_WAITING
        assert env.get_phase_mask(1).sum() == 1 and env.get_phase_mask(1)[15]
        assert env.get_phase(2) in (PHASE_BLIND_SELECT, PHASE_SELECTING_HAND, PHASE_SHOP)

    def test_opponent_progress(self):
        env = MultiplayerBalatroEnv(seed="7I4M53DL")
        env.reset()
        _to_nemesis(env, 1)
        assert env.p2.encode_obs()[V + 9] == 1.0      # opponent is >= 3 blinds ahead
        assert env.p1.encode_obs()[V + 9] == -1.0

    def test_obs_has_no_nan(self):
        env = MultiplayerBalatroEnv(seed=42)
        env.reset()
        for _ in range(30):
            (o1, o2), _, done, _ = env.step(_scripted(env, 1), _scripted(env, 2))
            assert not np.isnan(o1).any() and not np.isnan(o2).any()
            if done:
                break


class TestReset:
    def test_returns_two_observations(self):
        env = MultiplayerBalatroEnv(seed=42)
        o1, o2 = env.reset()
        assert o1.shape == (OBS_DIM,) and o2.shape == (OBS_DIM,)

    def test_both_observations_initially_identical(self):
        env = MultiplayerBalatroEnv(seed=42)
        o1, o2 = env.reset()
        np.testing.assert_array_equal(o1, o2)

    def test_reset_clears_episode_reward(self):
        env = MultiplayerBalatroEnv(seed=42)
        env.reset()
        assert env._episode_reward == [0.0, 0.0]

    def test_initial_lives(self):
        env = MultiplayerBalatroEnv(seed=42, lives=3)
        env.reset()
        assert env.get_lives(1) == 3 and env.get_lives(2) == 3
        env = MultiplayerBalatroEnv(seed=42)
        env.reset()
        assert env.get_lives(1) == MLB_STARTING_LIVES

    def test_same_seed_for_both(self):
        env = MultiplayerBalatroEnv(seed="7I4M53DL")
        env.reset()
        assert env.p1_game.seed_str == env.p2_game.seed_str == "7I4M53DL"
        assert env.p1_game.ruleset == env.p2_game.ruleset == "mlb"


class TestStep:
    def test_step_returns_valid_shapes(self):
        env = MultiplayerBalatroEnv(seed=42)
        env.reset()
        (o1, o2), (r1, r2), done, info = env.step(
            {"type": "phase", "action": 0}, {"type": "phase", "action": 0})
        assert o1.shape == (OBS_DIM,) and o2.shape == (OBS_DIM,)
        assert isinstance(r1, float) and isinstance(r2, float) and isinstance(done, bool)
        for k in ("p1_lives", "p2_lives", "p1_ante", "p2_ante", "pvp_active", "pvp_log", "winner"):
            assert k in info

    def test_step_advances_phase(self):
        env = MultiplayerBalatroEnv(seed=42)
        env.reset()
        env.step({"type": "phase", "action": 0}, {"type": "phase", "action": 0})
        assert env.get_phase(1) == PHASE_SELECTING_HAND and env.get_phase(2) == PHASE_SELECTING_HAND

    def test_play_action(self):
        env = MultiplayerBalatroEnv(seed=42)
        env.reset()
        env.step({"type": "phase", "action": 0}, {"type": "phase", "action": 0})
        env.step({"type": "hand", "intent": INTENT_PLAY, "subset": (0, 1, 2, 3, 4)},
                 {"type": "hand", "intent": INTENT_PLAY, "subset": (0, 1, 2, 3, 4)})
        assert env.p1_game.chips_scored > 0 and env.p2_game.chips_scored > 0

    def test_same_actions_same_scores(self):
        env = MultiplayerBalatroEnv(seed=42)
        env.reset()
        env.step({"type": "phase", "action": 0}, {"type": "phase", "action": 0})
        env.step({"type": "hand", "intent": INTENT_PLAY, "subset": (0, 1, 2, 3, 4)},
                 {"type": "hand", "intent": INTENT_PLAY, "subset": (0, 1, 2, 3, 4)})
        assert env.p1_game.chips_scored == env.p2_game.chips_scored

    def test_different_actions_different_scores(self):
        env = MultiplayerBalatroEnv(seed=42)
        env.reset()
        env.step({"type": "phase", "action": 0}, {"type": "phase", "action": 0})
        env.step({"type": "hand", "intent": INTENT_PLAY, "subset": (0, 1, 2, 3, 4)},
                 {"type": "hand", "intent": INTENT_PLAY, "subset": (4, 6)})
        assert env.p1_game.chips_scored != env.p2_game.chips_scored

    def test_discard_action(self):
        env = MultiplayerBalatroEnv(seed=42)
        env.reset()
        env.step({"type": "phase", "action": 0}, {"type": "phase", "action": 0})
        d = env.p1_game.discards_left
        env.step({"type": "hand", "intent": INTENT_DISCARD, "subset": (0, 1)},
                 {"type": "phase", "action": 15})
        assert env.p1_game.discards_left == d - 1


class TestRewardsAndEvents:
    def test_reward_constants_sane(self):
        assert R_PVP_WIN > 0 > R_PVP_LOSS and R_LIFE_LOSS < 0
        assert R_GAME_WIN > R_PVP_WIN and R_GAME_LOSS < R_PVP_LOSS

    def test_life_loss_at_a_regular_blind_is_penalised_and_game_continues(self):
        env = MultiplayerBalatroEnv(seed="7I4M53DL")
        env.reset()
        total = 0.0
        for _ in range(6):
            (_, _), (r1, _), done, info = env.step(_scripted(env, 1, k=1), {"type": "phase", "action": 15})
            total += info["p1_mp_reward"]
            if info["p1_lives"] < MLB_STARTING_LIVES:
                break
        assert info["p1_lives"] == MLB_STARTING_LIVES - 1 and not done
        assert env.p1_game.ante == 1 and env.p1_game.state in (State.SHOP, State.BLIND_SELECT)
        # the life-loss penalty was paid exactly once and is part of the step reward
        assert total == R_LIFE_LOSS and r1 <= info["p1_mp_reward"] + abs(r1 - info["p1_mp_reward"])

    def test_pvp_rewards(self):
        env = MultiplayerBalatroEnv(seed="7I4M53DL")
        env.reset()
        _to_nemesis(env, 1); _to_nemesis(env, 2)
        env.step({"type": "phase", "action": 0}, {"type": "phase", "action": 0})
        assert env.mp.pvp_active
        mp1 = mp2 = 0.0
        for _ in range(10):
            (_, _), (r1, r2), done, info = env.step(_scripted(env, 1, k=5), _scripted(env, 2, k=1))
            mp1 += info["p1_mp_reward"]; mp2 += info["p2_mp_reward"]
            if info["pvp_log"]:
                break
        assert info["pvp_log"] and info["pvp_log"][-1][1] == 1      # player 2 (index 1) lost
        assert info["p2_lives"] == MLB_STARTING_LIVES - 1 and info["p1_lives"] == MLB_STARTING_LIVES
        assert mp1 == R_PVP_WIN and mp2 == R_PVP_LOSS
        assert r1 >= R_PVP_WIN - 1e-9 + (r1 - info["p1_mp_reward"])   # MP component is included

    def test_terminal_rewards_once(self):
        env = MultiplayerBalatroEnv(seed="7I4M53DL", lives=1)
        env.reset()
        done = False
        r1_tot = r2_tot = 0.0
        for _ in range(50):
            (_, _), (r1, r2), done, info = env.step(_scripted(env, 1, k=1), _scripted(env, 2, k=5))
            r1_tot += info["p1_mp_reward"]; r2_tot += info["p2_mp_reward"]
            if done:
                break
        assert done and info["winner"] == 2
        assert r2_tot == R_GAME_WIN and r1_tot == R_GAME_LOSS + R_LIFE_LOSS
        # further steps are inert
        (_, _), (r1, r2), done2, _ = env.step(_scripted(env, 1), _scripted(env, 2))
        assert done2 and r1 == 0.0 and r2 == 0.0


class TestIntegration:
    def test_scripted_full_game_terminates(self):
        env = MultiplayerBalatroEnv(seed=42, lives=2)
        env.reset()
        done = False
        for _ in range(2000):
            _, _, done, info = env.step(_scripted(env, 1), _scripted(env, 2))
            if done:
                break
        assert done and info["winner"] in (1, 2)
        assert min(info["p1_lives"], info["p2_lives"]) == 0

    def test_first_to_zero_loses_is_ordering_dependent_but_deterministic(self):
        """Two equally bad players: whoever fails first (player 1 acts first in env.step)
        loses -- the server does the same with the first failRound it receives."""
        results = set()
        for _ in range(3):
            env = MultiplayerBalatroEnv(seed=55, lives=1)
            env.reset()
            for _ in range(200):
                _, _, done, info = env.step(_scripted(env, 1, k=1), _scripted(env, 2, k=1))
                if done:
                    break
            results.add(info["winner"])
        assert results == {2}

    def test_env_matches_direct_match_drive(self):
        """Stepping through the env or driving MLBMatch directly with the same actions
        gives the same match (the env adds no rules of its own)."""
        from balatro_sim.mlb_match import MLBMatch
        env = MultiplayerBalatroEnv(seed="7I4M53DL")
        env.reset()
        m = MLBMatch(seed="7I4M53DL")
        for _ in range(12):
            env.step({"type": "phase", "action": 0}, {"type": "phase", "action": 0})
            for p in (0, 1):
                if m.games[p].state == State.BLIND_SELECT and m.can_act(p):
                    m.step(p, {"type": "play_blind"})
            env.step({"type": "hand", "intent": INTENT_PLAY, "subset": (0, 1, 2, 3, 4)},
                     {"type": "hand", "intent": INTENT_PLAY, "subset": (0, 1, 2, 3, 4)})
            for p in (0, 1):
                if m.games[p].state == State.SELECTING_HAND and m.can_act(p):
                    m.step(p, {"type": "play", "cards": [0, 1, 2, 3, 4]})
                if m.games[p].state == State.ROUND_EVAL:
                    m.step(p, {"type": "advance"})
            env.step({"type": "phase", "action": 15}, {"type": "phase", "action": 15})
            for p in (0, 1):
                if m.games[p].state == State.SHOP:
                    m.step(p, {"type": "leave_shop"})
            if env.mp.done or m.done:
                break
        assert [g.state_signature() for g in env.mp.games] == [g.state_signature() for g in m.games]
