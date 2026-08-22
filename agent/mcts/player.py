"""
player.py — an MCTS `Player`: `act(game) -> action dict | None`.

This is the plug-in shape W2's tournament runner and W4's eval harness expect
(`mp/tournament/players.py`'s `Player` protocol: `act(game) -> action`). Neither
imports `mp/agent` in Phase 3 — this class is the thing that gets wired in when they
do, and it is deliberately the only place that knows how to turn "a game" into "an
action" without a training loop attached.

Semantics
    act(game) -> dict   the chosen action, ready for `game.step(...)`
                -> None when the game has NO legal actions. Under MLB that is a
                   real, expected state: `PVP_WAIT` (hands exhausted at the Nemesis,
                   waiting for the opponent) and BLIND_SELECT with `pvp_ready` (readied,
                   waiting for `startBlind`). The driver — `MLBMatch.sync()` or the
                   tournament runner — is what moves the game on from there.

Single-action states (`ROUND_EVAL`'s dummy `advance`, a BLIND_SELECT with only
`play_blind`) skip the search entirely: with one child every simulation goes to the same
edge, so the search is pure overhead. `--no-shortcut` (shortcut_singletons=False) turns
that off if a caller wants the value estimates anyway.

W3 additions (2026-08-22)
-------------------------
* `reuse=True` keeps the chosen child's subtree for the next `act()` (`mcts/reuse.py`),
  guarded by `state_signature()`. Per-agent state, so one player object per agent —
  `reset()` drops it (the tournament runner calls `reset()` before every run).
* `leaf_batch=L` batches L leaves per forward pass WITHIN one tree using virtual loss.
  That is what a single agent deciding alone can do; it is an approximation of the serial
  search (see BATCH_NOTES.md §3).
* `BatchedMCTSPlayerGroup` decides for N agents at once (`act_many`), batching their
  leaves across trees with no approximation at all. That needs a driver that hands over
  all N games together — see the 10-line `mp/tournament/players.py` diff in
  BATCH_NOTES.md §6.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Optional, Sequence

import numpy as np

from balatro_sim.game import BalatroGame
from .action import ActionKey, action_from_key, action_key
from .batched import BatchedNNPolicy, BatchedSearch, SearchRequest
from .outcome import OutcomeFn
from .policy import PolicyValueFn
from .reuse import ReuseConfig, TreeCache
from .search import MCTS, MCTSConfig


@dataclass
class Decision:
    """What `MCTSPlayer.record_hook` is handed at every decision (W2, Phase 4).

    Deliberately raw: the LIVE game (do not mutate it; encode it and let go), the legal
    action dicts in `legal_actions()` order, their `ActionKey`s in the same order, the
    search's root visit counts, and the chosen key. Turning this into a training `Sample`
    is the caller's job — that is the one place the observation encoding lives, so
    switching encoders is a one-line change in `train/selfplay.py::make_sample`.

    `visits` is `{}` on a forced-action (single legal action) state: no search ran, and
    the policy target for that state is the point mass on `chosen`.
    """
    game: BalatroGame
    legal: list
    legal_keys: list
    visits: dict
    chosen: Optional[ActionKey]
    shortcut: bool
    player: "MCTSPlayer"
    sims: int = 0


@dataclass
class MCTSPlayer:
    """An MCTS-driven player. `strategy`: "gumbel" (default, matches self-play) or
    "puct". `temperature` only applies to "puct" (Gumbel's own selection is the
    improvement operator); 0 means argmax visits."""

    policy: PolicyValueFn
    config: Optional[MCTSConfig] = None
    outcome: Optional[OutcomeFn] = None
    rng: Optional[np.random.Generator] = None
    strategy: str = "gumbel"
    temperature: float = 0.0
    add_noise: bool = False
    shortcut_singletons: bool = True
    name: str = "mcts"
    # W3
    reuse: object = False              # False | True | ReuseConfig
    leaf_batch: int = 1                # >1: virtual-loss leaf batching within this tree
    no_action: Optional[dict] = None   # returned instead of None on a no-action state
    # W2 (Phase 4): an ADDITIVE observation hook. When set, every decision this player
    # makes calls `record_hook(Decision)` just before the action is returned, so a driver
    # that does NOT own the play loop (the tournament runner does) can still collect
    # training samples. `None` (the default) costs nothing — no dict is built, no extra
    # `legal_actions()` call is made. See `train/selfplay.py`.
    record_hook: Optional[Callable[["Decision"], None]] = None
    record_singletons: bool = True     # also record forced-action states (value-head samples)
    # W2 (Phase 4): route a `leaf_batch > 1` search through `BatchedSearch` so its L
    # in-flight leaves reach the net in ONE `evaluate_many` call. `MCTS._drive` answers leaf
    # requests one at a time by design (it is what keeps `run`/`run_gumbel` byte-identical to
    # the pre-W3 implementation), so without this `leaf_batch=16` bought the virtual-loss
    # tree shape and none of the batching. Measured on this box, CUDA, 40 sims, L=16, MLB
    # ante-1 blind: 243 -> 340 sims/s (1.40x). No effect at `leaf_batch == 1`, where the
    # serial path is used unchanged.
    batch_leaf_eval: bool = True
    # W2 (Phase 4): a TRAINING-TIME action constraint. `legal_filter(game, legal) -> legal`
    # returns the subset of `game.legal_actions()` this player may actually take. The
    # search still explores the full set (nothing inside `MCTS` changes), but the choice AND
    # the recorded visit distribution are restricted to the allowed subset and renormalised,
    # so the policy target is "the improved policy under the constraint". Used by
    # `--max-skips-per-ante`; `None` (default) costs nothing. NOT an engine rule — real MLB
    # lets you skip both blinds, and the constraint is annealed away before the run ends.
    legal_filter: Optional[Callable] = None

    def __post_init__(self):
        self.config = self.config or MCTSConfig()
        if self.leaf_batch != self.config.leaf_batch:
            # `replace` so a config object shared with other players is not mutated.
            self.config = replace(self.config, leaf_batch=self.leaf_batch)
        self.leaf_batch = self.config.leaf_batch
        self.rng = self.rng if self.rng is not None else np.random.default_rng()
        self.mcts = MCTS(self.policy, self.config, rng=self.rng, outcome=self.outcome)
        self.cache = TreeCache(_reuse_config(self.reuse))
        self._batched = (
            BatchedSearch(self.policy, self.config, outcome=self.outcome,
                          strategy=self.strategy)
            if (self.batch_leaf_eval and self.config.leaf_batch > 1) else None
        )
        self.searches = 0
        self.shortcuts = 0

    @property
    def reuse_stats(self):
        return self.cache.stats

    # ── Player protocol ──────────────────────────────────────────────────────

    def act(self, game: BalatroGame) -> Optional[dict]:
        key = self.act_key(game)
        if key is None:
            # `no_action` exists for drivers that step unconditionally — the tournament
            # runner does `game.step(player.act(game))`, and its other adapters return
            # `{"type": "advance"}` in this situation. Default stays None (W1's contract,
            # which `MLBMatch` and the self-play agent rely on to detect a stuck state).
            return self.no_action
        return action_from_key(key)

    def reset(self) -> None:
        """Drop the retained tree. The tournament runner calls this before every run;
        counters are kept (they are diagnostics, not state the search reads)."""
        self.cache.clear()
        reset = getattr(self.legal_filter, "reset", None)
        if reset is not None:
            reset()

    def act_key(self, game: BalatroGame) -> Optional[ActionKey]:
        legal = game.legal_actions()
        if not legal:
            self.cache.clear()
            return None
        if self.shortcut_singletons and len(legal) == 1:
            self.shortcuts += 1
            key = self._shortcut(game, action_key(legal[0]))
            if self.record_hook is not None and self.record_singletons:
                self._record(game, legal, [key], {}, key, shortcut=True, sims=0)
            return key

        self.searches += 1
        root = self.cache.take(game)
        sims = self.cache.budget(root, self.config.num_simulations)
        visits: dict = {}
        if self._batched is not None:
            res = self._batched.run_requests([SearchRequest(
                game=game, mcts=self.mcts, strategy=self.strategy, root=root, sims=sims,
                add_noise=self._noise(root), leaf_batch=self.config.leaf_batch,
            )])[0]
            root, visits, chosen = res.root, res.visit_counts, res.chosen
            if chosen is None and visits:
                chosen = MCTS.sample_action(visits, temperature=self.temperature,
                                            rng=self.rng)
        elif self.strategy == "gumbel":
            root, visits, chosen = self.mcts.run_gumbel(game, root=root, sims=sims)
        else:
            root, visits = self.mcts.run(game, add_noise=self._noise(root),
                                         root=root, sims=sims)
            chosen = (MCTS.sample_action(visits, temperature=self.temperature,
                                         rng=self.rng) if visits else None)
        visits = visits or {}
        if self.legal_filter is not None:
            legal, visits, chosen = self._constrain(game, legal, visits, chosen)
        self.cache.store(game, root, chosen)
        if self.record_hook is not None and chosen is not None:
            self._record(game, legal, [action_key(a) for a in legal], visits,
                         chosen, shortcut=False, sims=sims)
        return chosen

    def _constrain(self, game, legal, visits, chosen):
        """Restrict the decision to `legal_filter`'s subset. The search's evidence is kept
        (visit counts on allowed actions are what they were); only the support shrinks."""
        allowed = list(self.legal_filter(game, legal))
        if len(allowed) >= len(legal) or not allowed:
            return legal, visits, chosen
        allowed_keys = [action_key(a) for a in allowed]
        keyset = set(allowed_keys)
        visits = {k: v for k, v in visits.items() if k in keyset}
        if chosen not in keyset:
            chosen = (max(visits, key=visits.get) if visits
                      else allowed_keys[int(self.rng.integers(len(allowed_keys)))])
        return allowed, visits, chosen

    def _record(self, game, legal, legal_keys, visits, chosen, shortcut: bool,
                sims: int) -> None:
        """Hand one `Decision` to `record_hook`. Only ever called when a hook is set."""
        self.record_hook(Decision(game=game, legal=legal, legal_keys=legal_keys,
                                  visits=visits, chosen=chosen, shortcut=shortcut,
                                  player=self, sims=sims))

    # ── Internals ────────────────────────────────────────────────────────────

    def _noise(self, root) -> bool:
        """Root Dirichlet noise. On a REUSED root the children carry bare policy priors
        (noise only ever went into the previous root's own children), so re-noising is
        the same one-shot mix a fresh search does — it cannot compound."""
        if root is None:
            return self.add_noise
        return self.add_noise and self.cache.cfg.renoise

    def _shortcut(self, game: BalatroGame, key: ActionKey) -> ActionKey:
        """A forced action still moves the tree along: if the retained root covers this
        state, keep ITS child rather than throwing the subtree away. `ROUND_EVAL` ->
        `advance` sits between every blind and every shop, so this is the difference
        between a tree surviving a whole ante and one dying at every cash-out."""
        root = self.cache.take(game)
        self.cache.store(game, root, key if root is not None else None)
        return key


def load_policy(checkpoint: Optional[str] = None, device: str = "cpu",
                batched: bool = True, encoder: Optional[str] = None):
    """A `PolicyValueFn` from a `train_cold` checkpoint (or a cold-start net when
    `checkpoint is None`). `batched=True` gives the `evaluate_many` implementation, which
    a single-leaf caller can use unchanged — it is a strict superset of `NNPolicy`.

    The `train` import is function-local on purpose: `train` imports `mcts`, so a
    module-level import here would be circular.
    """
    from .encoder import get_encoder
    from .model import PolicyValueNet
    from .policy import make_policy

    if checkpoint is None:
        enc = get_encoder(encoder or "v7")
        net = build_net(enc)
    else:
        from train.checkpoint import load_checkpoint
        ckpt = load_checkpoint(checkpoint, map_location=device)
        name = encoder or ckpt.get("encoder", "v7")
        # Phase 4 W1: the checkpoint records the SET encoder's caps, so a resume rebuilds
        # exactly the padding the net was trained with rather than the current defaults.
        caps = ckpt.get("encoder_caps")
        enc = get_encoder(name, **({"caps": caps} if caps and name == "set" else {}))
        desc = ckpt["net_desc"]
        if desc.get("kind") == "set" or ckpt.get("net_kind") == "set":
            from .model_set import SetPolicyValueNet
            net = SetPolicyValueNet.from_description(desc)
        else:
            net = PolicyValueNet.from_description(desc)
        net.load_state_dict(ckpt["model"])
    return make_policy(net, device=device, encoder=enc, batched=batched)


def build_net(encoder):
    """A cold-start net matched to `encoder` — the one place that knows which net class
    goes with which encoder."""
    if getattr(encoder, "is_set", False):
        from .model_set import SetPolicyValueNet
        return SetPolicyValueNet(caps=encoder.caps)
    from .model import PolicyValueNet
    return PolicyValueNet(obs_dim=encoder.dim)


def make_player(checkpoint: Optional[str] = None, sims: int = 100, device: str = "cpu",
                seed: int = 0, strategy: str = "gumbel", reuse: object = True,
                leaf_batch: int = 1, temperature: float = 0.0,
                no_action: Optional[dict] = None, encoder: Optional[str] = None,
                **kwargs) -> "MCTSPlayer":
    """One line from "a checkpoint path" to "a `Player` the tournament can call".

    Defaults are the tournament's: tree reuse on, Gumbel selection, argmax at the end,
    and `no_action={"type": "advance"}` so `game.step(player.act(game))` is always safe
    (`mp/tournament/runner.py::_drive_to_next_nemesis` steps unconditionally).
    """
    policy = load_policy(checkpoint, device=device, batched=True, encoder=encoder)
    return MCTSPlayer(
        policy=policy,
        config=MCTSConfig(num_simulations=sims, leaf_batch=leaf_batch),
        # BOTH, deliberately: `__post_init__` lets the FIELD win over the config, so passing
        # only the config silently ran every tournament player at L=1 (W2, Phase 4).
        leaf_batch=leaf_batch,
        strategy=strategy, temperature=temperature, reuse=reuse,
        rng=np.random.default_rng(seed),
        no_action=no_action if no_action is not None else {"type": "advance"},
        name=f"mcts-s{sims}", **kwargs)


def _reuse_config(spec) -> ReuseConfig:
    """`False` -> disabled, `True` -> defaults, a ReuseConfig -> itself."""
    if isinstance(spec, ReuseConfig):
        return spec
    return ReuseConfig(enabled=bool(spec))


class BatchedMCTSPlayerGroup:
    """N `MCTSPlayer`s that decide TOGETHER so their leaves share one forward pass.

    This is the shape the tournament wants: N agents on one seed, all of them needing an
    action at the same moment. `act_many([g0, ..., gN-1])` returns N actions, having run
    N searches in lockstep — every round of the search evaluates N leaves in a single
    `evaluate_many` call instead of N separate ones. Each agent keeps its own rng, its own
    outcome function and its own retained tree, so the population can be heterogeneous
    (different budgets or checkpoints) without breaking the batch.

    Each member is a perfectly ordinary `MCTSPlayer` (`group.players[i].act(game)` still
    works one at a time); the group only adds the batched path.

    Games that are `None`, terminal, or have no legal actions are skipped without cost;
    single-action states are shortcut exactly as `MCTSPlayer.act` does. A tree that
    finishes its budget early leaves the batch (see `BatchedSearch`).
    """

    def __init__(self, n_agents: int, policy: PolicyValueFn,
                 config: Optional[MCTSConfig] = None, strategy: str = "gumbel",
                 outcome: Optional[OutcomeFn] = None, seeds: Optional[Sequence[int]] = None,
                 reuse: object = True, temperature: float = 0.0, add_noise: bool = False,
                 shortcut_singletons: bool = True, name: str = "mcts"):
        self.policy = policy
        self.config = config or MCTSConfig()
        self.strategy = strategy
        self.search = BatchedSearch(policy, self.config, outcome=outcome,
                                    strategy=strategy)
        self.players = [
            MCTSPlayer(policy=policy, config=self.config, outcome=outcome,
                       rng=np.random.default_rng(seeds[i] if seeds is not None else i),
                       strategy=strategy, temperature=temperature, add_noise=add_noise,
                       shortcut_singletons=shortcut_singletons, reuse=reuse,
                       leaf_batch=self.config.leaf_batch, name=f"{name}{i}")
            for i in range(n_agents)
        ]

    def __len__(self) -> int:
        return len(self.players)

    def __getitem__(self, i: int) -> MCTSPlayer:
        return self.players[i]

    def reset(self) -> None:
        for p in self.players:
            p.reset()

    @property
    def stats(self):
        return self.search.stats

    # ── The batched decision ────────────────────────────────────────────────

    def act_many(self, games: Sequence[Optional[BalatroGame]]) -> list[Optional[dict]]:
        keys = self.act_keys(games)
        return [None if k is None else action_from_key(k) for k in keys]

    def act_keys(self, games: Sequence[Optional[BalatroGame]]) -> list[Optional[ActionKey]]:
        if len(games) != len(self.players):
            raise ValueError(f"expected {len(self.players)} games, got {len(games)}")
        out: list[Optional[ActionKey]] = [None] * len(games)
        requests: list[SearchRequest] = []
        owners: list[int] = []

        for i, game in enumerate(games):
            player = self.players[i]
            if game is None:
                player.cache.clear()
                continue
            legal = game.legal_actions()
            if not legal:
                player.cache.clear()
                continue
            if player.shortcut_singletons and len(legal) == 1:
                player.shortcuts += 1
                out[i] = player._shortcut(game, action_key(legal[0]))
                continue
            player.searches += 1
            root = player.cache.take(game)
            sims = player.cache.budget(root, self.config.num_simulations)
            requests.append(SearchRequest(
                game=game, mcts=player.mcts, strategy=player.strategy, root=root,
                sims=sims, add_noise=player._noise(root),
                leaf_batch=player.config.leaf_batch,
            ))
            owners.append(i)

        if not requests:
            return out

        for res, i in zip(self.search.run_requests(requests), owners):
            player = self.players[i]
            chosen = res.chosen
            if chosen is None and res.visit_counts:
                chosen = MCTS.sample_action(res.visit_counts,
                                            temperature=player.temperature,
                                            rng=player.rng)
            player.cache.store(games[i], res.root, chosen)
            out[i] = chosen
        return out


__all__ = ["MCTSPlayer", "Decision", "BatchedMCTSPlayerGroup", "build_net", "load_policy",
           "make_player"]
