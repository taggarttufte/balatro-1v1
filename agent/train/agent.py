"""
agent.py — SelfPlayAgent: plays one episode and returns Samples.

For each decision point:
  1. Encode obs (447,)
  2. Enumerate legal actions and featurize them (N, 56)
  3. Run Gumbel MCTS to get a chosen action and visit counts
  4. Build target_policy = N(a) / sum N(b)  over legal actions
  5. Append a Sample (z=0 placeholder) and step the game

When the episode ends, fill in z on every sample from the OutcomeFn.

Fork changes vs balatro-mcts @ ee75d11
--------------------------------------
* **The outcome is a parameter** (`mcts/outcome.py`). `won` used to be
  `state == GAME_OVER and ante > 8` and `z` used to be `(blinds + chips)/24`; both are
  wrong under MLB (endless; the win is `match_won`; a Nemesis is decided by an external
  score). `play_episode(game, outcome=...)`, defaulting to `default_outcome_for(game)`.
* **States with no legal actions are a normal stop, not an error.** MLB's `PVP_WAIT`
  and "readied at the Nemesis" have no actions; the episode returns with
  `stop_reason="stuck"` so a driver (MLBMatch / the tournament runner) can resolve it
  and — via `resume_episode` — the same agent can continue the same trajectory.
* **`pvp_target_fn` hook**: an external supplier of the Nemesis target (`set_pvp_info`),
  which is what W2/W4 have and the game does not.
* Returns an `EpisodeResult`; it still unpacks as the old
  `(samples, final_ante, won)` 3-tuple.

Single-action states (ROUND_EVAL's dummy `advance`) skip the search — with one child
every simulation lands on the same edge, so it is pure overhead. Their Sample is still
recorded (target_policy = [1.0]) so the value head sees those states.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from balatro_sim.game import BalatroGame
from mcts import MCTS, NNPolicy
from mcts.action import action_key, action_from_key
from mcts.action_features import featurize_actions
from mcts.encoder import ObsEncoder, V7Encoder
from mcts.outcome import OutcomeFn, default_outcome_for
from mcts.search import MCTSConfig

from .trajectory import Sample


@dataclass
class EpisodeResult:
    """What one `play_episode` produced. Unpacks as (samples, final_ante, won) for the
    balatro-mcts call sites."""
    samples: list[Sample]
    final_ante: int
    won: bool
    z: float
    stop_reason: str            # game_over | stuck | no_actions | max_decisions | max_antes
    decisions: int = 0
    searches: int = 0

    def __iter__(self):
        return iter((self.samples, self.final_ante, self.won))


class SelfPlayAgent:
    def __init__(
        self,
        policy: NNPolicy,
        mcts_config: MCTSConfig | None = None,
        rng: np.random.Generator | None = None,
        max_decisions: int = 2000,
        outcome: OutcomeFn | None = None,
        encoder: ObsEncoder | None = None,
        max_antes: Optional[int] = None,
        pvp_target_fn: Optional[Callable[[BalatroGame], tuple[int, int]]] = None,
        shortcut_singletons: bool = True,
    ):
        self.policy = policy
        self.cfg = mcts_config or MCTSConfig()
        self.rng = rng if rng is not None else np.random.default_rng()
        self.max_decisions = max_decisions
        self.max_antes = max_antes
        self.outcome = outcome
        self.encoder: ObsEncoder = encoder or getattr(policy, "encoder", None) or V7Encoder()
        self.pvp_target_fn = pvp_target_fn
        self.shortcut_singletons = shortcut_singletons
        self.mcts = MCTS(policy, self.cfg, rng=self.rng, outcome=outcome)

    # ── Episode ──────────────────────────────────────────────────────────────

    def play_episode(
        self,
        game: BalatroGame,
        outcome: OutcomeFn | None = None,
        samples: Optional[list[Sample]] = None,
    ) -> EpisodeResult:
        """
        Play from the given game state until it stops. Every sample gets the same
        shaped outcome z in [0, 1] from the OutcomeFn:

            VanillaOutcome  fractional progress through the 24 blinds (1.0 = won)
            MLBOutcome      lives left + blind progress    (1.0 = match_won)
            ExternalOutcome whatever the driver supplies (W2/W4)

        `won` is the unshaped win flag and is meant for logging, not for training.

        Pass `samples` to continue an existing trajectory (see `resume_episode`).
        """
        outcome = outcome or self.outcome or default_outcome_for(game)
        self.mcts.outcome = outcome
        samples = samples if samples is not None else []
        stop_reason = "max_decisions"
        decisions = 0
        searches = 0

        for _ in range(self.max_decisions):
            if outcome.is_terminal(game):
                stop_reason = "game_over"
                break
            if self.max_antes is not None and game.ante > self.max_antes:
                stop_reason = "max_antes"
                break
            if outcome.is_stuck(game):
                stop_reason = "stuck"
                break

            # External Nemesis target (`enemyInfo`): the opponent's live score is the
            # blind's chips_target and the game has no way to know it.
            if self.pvp_target_fn is not None and getattr(game.current_blind, "is_pvp", False):
                score, hands = self.pvp_target_fn(game)
                game.set_pvp_info(int(score), int(hands))

            legal = game.legal_actions()
            if not legal:
                stop_reason = "no_actions"
                break

            obs = self.encoder(game)
            action_feats = featurize_actions(legal)
            legal_keys = [action_key(a) for a in legal]

            if self.shortcut_singletons and len(legal) == 1:
                chosen = legal_keys[0]
                target_policy = np.ones(1, dtype=np.float32)
            else:
                searches += 1
                _, visits, chosen = self.mcts.run_gumbel(game)
                if chosen is None:                 # became stuck between the checks
                    stop_reason = "stuck"
                    break
                counts = np.array([visits.get(k, 0) for k in legal_keys], dtype=np.float64)
                total = counts.sum()
                if total > 0:
                    target_policy = (counts / total).astype(np.float32)
                else:
                    target_policy = np.full(
                        len(legal_keys), 1.0 / len(legal_keys), dtype=np.float32
                    )

            samples.append(Sample(
                obs=obs,
                action_features=action_feats,
                target_policy=target_policy,
                z=0.0,  # placeholder, set below
            ))
            decisions += 1
            game.step(action_from_key(chosen))

        won = outcome.is_win(game)
        z = outcome.value(game)
        for s in samples:
            s.z = z

        return EpisodeResult(samples=samples, final_ante=game.ante, won=won, z=z,
                             stop_reason=stop_reason, decisions=decisions,
                             searches=searches)

    def resume_episode(self, game: BalatroGame, prior: EpisodeResult,
                       outcome: OutcomeFn | None = None) -> EpisodeResult:
        """Continue a trajectory that stopped on `stuck` (MLB: the driver resolved the
        PvP / started the blind). Appends to the same sample list, so the final z is
        applied to the whole episode."""
        result = self.play_episode(game, outcome=outcome, samples=prior.samples)
        result.decisions += prior.decisions
        result.searches += prior.searches
        return result


def _shaped_z(game: BalatroGame, won: bool) -> float:
    """Back-compat shim for the balatro-mcts call sites: the VANILLA shaped label.
    Prefer `OutcomeFn.value(game)` — this one assumes a 24-blind, ante-8-win run and is
    wrong under MLB."""
    from mcts.outcome import VanillaOutcome
    if won:
        return 1.0
    return VanillaOutcome().value(game)
