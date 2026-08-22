"""
env_mp.py — Multiplayer (Major League Balatro) environment on top of ``MLBMatch``.

Phase 2 W1 (2026-08-21) rewrite.  The V8-era version coordinated two games with invented
rules ("regular blind failure = game over", ties, an ante-8 draw); the real rules live in
``mlb_match.py`` / ``game.py`` (``ruleset="mlb"``) and are documented in ``MLB_NOTES.md``.
This module only (a) wraps each player's game in a V7 proxy for the observation / action
encoding and the V7 per-player rewards, (b) adds the MP observation block, and (c) adds
the MP event rewards.

Observation per player = V7 obs (``env_v7.OBS_DIM``) + ``MP_OBS_FEATURES`` (see
``_mp_features``; brief §1.8: both players' lives, the opponent's live score and
hands-left during PvP, the comeback state, the opponent's location between blinds).

Reward per player = V7 reward (proxy) + MP events:
  life lost at a regular blind  R_LIFE_LOSS
  PvP won / lost / tied         R_PVP_WIN / R_PVP_LOSS / 0
  match won / lost              R_GAME_WIN / R_GAME_LOSS

API (unchanged shape):
  env = MultiplayerBalatroEnv(seed=42)
  p1_obs, p2_obs = env.reset()
  (p1_obs, p2_obs), (p1_r, p2_r), done, info = env.step(p1_action, p2_action)
Players are numbered 1 / 2 here (0 / 1 inside ``MLBMatch``).  A player who cannot act
(readied for the Nemesis, out of hands at the Nemesis, match over) has their action
ignored for that step.
"""
from __future__ import annotations
import math
from typing import Optional

import numpy as np

from .mlb_match import MLBMatch, DEFAULT_LIVES
from .game import State
from .constants import MLB_BANNED_JOKERS, MLB_COMEBACK_PER_LIFE
from .env_v7 import (
    BalatroV7Env, OBS_DIM as V7_OBS_DIM, N_PHASE_ACTIONS,
    PHASE_SELECTING_HAND, PHASE_BLIND_SELECT, PHASE_SHOP, PHASE_GAME_OVER,   # re-exported
)


# MP observation block (appended to the V7 obs) — indices relative to V7_OBS_DIM:
#   0 self_lives / 4             1 opp_lives / 4
#   2 is_pvp (Nemesis in progress for me)
#   3 opp_score_log  = log1p(opp score) / log1p(1e5)          (0 outside PvP)
#   4 lead           = clip((mine - opp) / max(mine + opp, 1), -1, 1)   (0 outside PvP)
#   5 opp_hands_left / max(base_hands, 1)                      (0 outside PvP)
#   6 opp_exhausted  (PvP and the opponent has 0 hands left)
#   7 waiting        (readied for the Nemesis, or out of hands at it)
#   8 comeback_pending / 16  ($ owed at my next Cash Out: 4 x cumulative lives lost)
#   9 opp_progress   = clip((opp blind ordinal - my blind ordinal) / 3, -1, 1)
MP_OBS_FEATURES = 10
OBS_DIM = V7_OBS_DIM + MP_OBS_FEATURES

# The Attrition joker bans (attrition.lua:13-18).  Applied per game by
# ``BalatroGame(ruleset="mlb")`` through ``run_state.banned_keys`` — NOT as a
# process-global any more (the V8-era module-import ``set_banned_jokers`` call polluted
# every vanilla game in the same process).
MULTIPLAYER_BANNED_JOKERS = frozenset(MLB_BANNED_JOKERS)

# MP event rewards (on top of the V7 per-player rewards)
R_PVP_WIN     = 10.0
R_PVP_LOSS    = -5.0
R_LIFE_LOSS   = -5.0    # a life lost at a regular (non-PvP) blind
R_GAME_WIN    = 20.0
R_GAME_LOSS   = -10.0

PHASE_WAITING = PHASE_GAME_OVER   # no decision to make (ready-wait / PVP_WAIT / terminal)


class _PlayerEnvProxy:
    """A V7 env pointed at a game owned by the match: reuses V7's obs encoding, masks and
    per-player rewards.  Never constructs its own game."""

    def __init__(self, match: MLBMatch, player: int):
        self.match = match
        self.player = player                  # 1 or 2
        self.idx = player - 1
        self._v7 = BalatroV7Env.__new__(BalatroV7Env)
        # minimal V7 state (BalatroV7Env.__init__ would build a vanilla game we don't want)
        v7 = self._v7
        v7._seed = match.seed_str
        v7.game = match.games[self.idx]
        v7._prev_progress = 0.0
        v7._prev_ante = 1
        v7._prev_blind_idx = 0
        v7._steps = 0
        v7._play_history = []
        v7._episode_reward = 0.0
        v7._joker_acquisition_ante = {}
        v7.observation_space = None       # gym spaces are only used by the training script
        v7.action_space = None

    @property
    def game(self):
        return self.match.games[self.idx]

    @property
    def opponent(self):
        return self.match.games[1 - self.idx]

    def can_act(self) -> bool:
        return self.match.can_act(self.idx)

    def encode_obs(self) -> np.ndarray:
        base = self._v7._encode_obs()
        return np.concatenate([base, self._mp_features()])

    def _mp_features(self) -> np.ndarray:
        g, o = self.game, self.opponent
        lives0 = max(self.match.starting_lives, 1)
        in_pvp = bool(g.pvp_started and g.current_blind.is_pvp
                      and g.state in (State.SELECTING_HAND, State.PVP_WAIT))
        mine, opp = g.chips_scored, g.pvp_opponent_score
        f = np.zeros(MP_OBS_FEATURES, dtype=np.float32)
        f[0] = g.lives / lives0
        f[1] = o.lives / lives0
        f[2] = 1.0 if in_pvp else 0.0
        if in_pvp:
            f[3] = math.log1p(max(opp, 0)) / math.log1p(100000)
            f[4] = max(-1.0, min(1.0, (mine - opp) / max(mine + opp, 1)))
            f[5] = g.pvp_opponent_hands / max(g.base_hands, 1)
            f[6] = 1.0 if g.pvp_opponent_hands <= 0 else 0.0
        f[7] = 1.0 if (g.pvp_ready or g.state == State.PVP_WAIT) else 0.0
        pending = 0 if g.comeback_bonus_given else MLB_COMEBACK_PER_LIFE * g.comeback_bonus
        f[8] = min(pending / 16.0, 2.0)
        mine_ord = g.ante * 3 + g.blind_idx
        opp_ord = o.ante * 3 + o.blind_idx
        f[9] = max(-1.0, min(1.0, (opp_ord - mine_ord) / 3.0))
        return f

    def get_intent_mask(self) -> np.ndarray:
        return self._v7.get_intent_mask()

    def get_phase_mask(self) -> np.ndarray:
        if not self.can_act():
            mask = np.zeros(N_PHASE_ACTIONS, dtype=bool)
            mask[15] = True      # dummy, as V7 does for GAME_OVER
            return mask
        return self._v7.get_phase_mask()

    def get_phase(self) -> int:
        if not self.can_act():
            return PHASE_WAITING
        return self._v7.get_phase()

    def step_hand(self, intent: int, subset: tuple):
        return self._v7.step_hand(intent, subset)

    def step_phase(self, action: int):
        return self._v7.step_phase(action)

    def auto_advance(self):
        self._v7._auto_advance()


class MultiplayerBalatroEnv:
    """Self-play environment: two V7-style agents on one ``MLBMatch``."""

    def __init__(self, seed: Optional[int | str] = None, lives: int = DEFAULT_LIVES,
                 deck_key: str = "b_red", stake: "int | str" = 1):
        self._seed = seed
        self._lives = lives
        self._deck_key = deck_key
        self._stake = stake
        self.mp: Optional[MLBMatch] = None
        self.p1: Optional[_PlayerEnvProxy] = None
        self.p2: Optional[_PlayerEnvProxy] = None
        self._episode_reward = [0.0, 0.0]
        self._terminal_paid = False

    # ── convenience ──────────────────────────────────────────────────────────

    @property
    def p1_game(self):
        return self.mp.games[0]

    @property
    def p2_game(self):
        return self.mp.games[1]

    def get_player_game(self, player: int):
        return self.mp.games[player - 1]

    def get_lives(self, player: int) -> int:
        return self.mp.games[player - 1].lives

    def reset(self, seed: Optional[int | str] = None) -> tuple[np.ndarray, np.ndarray]:
        if seed is not None:
            self._seed = seed
        self.mp = MLBMatch(seed=self._seed, deck_key=self._deck_key, stake=self._stake,
                           lives=self._lives)
        self.p1 = _PlayerEnvProxy(self.mp, 1)
        self.p2 = _PlayerEnvProxy(self.mp, 2)
        self.p1.auto_advance()
        self.p2.auto_advance()
        self.mp.sync()
        self._episode_reward = [0.0, 0.0]
        self._terminal_paid = False
        return self.p1.encode_obs(), self.p2.encode_obs()

    # ── stepping ─────────────────────────────────────────────────────────────

    def step(self, p1_action: dict, p2_action: dict):
        """Apply both players' actions (player 1 first, then 2 — the match's canonical
        alternation is not enforced here; both act every env step), then settle the MP
        events.  Action dicts:
          SELECTING_HAND:  {"type": "hand",  "intent": int, "subset": tuple}
          other phases:    {"type": "phase", "action": int}
        """
        rewards = [0.0, 0.0]
        mp_rewards = [0.0, 0.0]          # the MP-event component alone (diagnostics)
        for proxy, action in ((self.p1, p1_action), (self.p2, p2_action)):
            lives_before = [g.lives for g in self.mp.games]
            pvp_before = len(self.mp.pvp_log)
            rewards[proxy.idx] += self._apply_action(proxy, action)
            self.mp.sync()
            self._settle_events(mp_rewards, lives_before, pvp_before)
        rewards[0] += mp_rewards[0]
        rewards[1] += mp_rewards[1]
        # cash-outs / pack interrupts left pending by a match-driven transition
        for proxy in (self.p1, self.p2):
            if proxy.game.state == State.ROUND_EVAL:
                proxy.auto_advance()
        self.mp.sync()

        self._episode_reward[0] += rewards[0]
        self._episode_reward[1] += rewards[1]
        done = self.mp.done
        g0, g1 = self.mp.games
        info = {
            "p1_lives": g0.lives, "p2_lives": g1.lives,
            "p1_ante": g0.ante, "p2_ante": g1.ante,
            "pvp_active": self.mp.pvp_active,
            "pvp_log": list(self.mp.pvp_log),
            "winner": (self.mp.winner + 1) if self.mp.winner is not None else None,
            "p1_mp_reward": mp_rewards[0], "p2_mp_reward": mp_rewards[1],
            "p1_total_reward": self._episode_reward[0],
            "p2_total_reward": self._episode_reward[1],
        }
        return (self.p1.encode_obs(), self.p2.encode_obs()), (rewards[0], rewards[1]), done, info

    def _apply_action(self, proxy: _PlayerEnvProxy, action: dict) -> float:
        if not proxy.can_act():
            return 0.0
        atype = action.get("type")
        if atype == "hand":
            _, reward, _, _, _ = proxy.step_hand(action["intent"], action["subset"])
        elif atype == "phase":
            _, reward, _, _, _ = proxy.step_phase(action["action"])
        else:
            reward = 0.0
        self.mp.steps += 1
        return float(reward)

    def _settle_events(self, rewards: list, lives_before: list, pvp_before: int) -> None:
        """Translate what ``sync`` just did into MP rewards (once per event)."""
        games = self.mp.games
        new_pvp = self.mp.pvp_log[pvp_before:]
        pvp_losers = {entry[1] for entry in new_pvp if entry[1] is not None}
        for i, g in enumerate(games):
            if g.lives < lives_before[i]:
                if i in pvp_losers:
                    rewards[i] += R_PVP_LOSS
                    rewards[1 - i] += R_PVP_WIN
                else:
                    rewards[i] += R_LIFE_LOSS
        if self.mp.done and not self._terminal_paid:
            w = self.mp.winner
            rewards[w] += R_GAME_WIN
            rewards[1 - w] += R_GAME_LOSS
            self._terminal_paid = True   # pay the terminal reward exactly once

    # ── per-player helpers ───────────────────────────────────────────────────

    def _proxy(self, player: int) -> _PlayerEnvProxy:
        return self.p1 if player == 1 else self.p2

    def get_phase(self, player: int) -> int:
        return self._proxy(player).get_phase()

    def get_intent_mask(self, player: int) -> np.ndarray:
        return self._proxy(player).get_intent_mask()

    def get_phase_mask(self, player: int) -> np.ndarray:
        return self._proxy(player).get_phase_mask()

    def legal_actions(self, player: int) -> list[dict]:
        """Engine-level legal actions for the player (``MLBMatch.legal_actions``)."""
        return self.mp.legal_actions(player - 1)
