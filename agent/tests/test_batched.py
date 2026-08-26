"""
test_batched.py — batched leaf evaluation (Phase 3, W3).

What is pinned here:

  1. `BatchedNNPolicy.evaluate_many` == `NNPolicy.__call__` for the same leaves (float
     tolerance — see `test_batched_policy_matches_single_leaf` for why it is NOT bit
     exact even on CPU), including chunked batches and leaves with no legal actions.
  2. A K-tree batched search reproduces K independent single-tree searches EXACTLY
     (uniform policy) and to within a hair (NN policy, whose priors differ by ~1e-10
     between batch shapes).
  3. Trees that finish early — smaller budgets, terminal roots, MLB no-action roots —
     leave the batch instead of stalling it.
  4. The generator refactor of `search.py` left the serial search byte-identical: a
     verbatim copy of the pre-W3 loops is kept in this file and compared tree-for-tree.
  5. Virtual-loss leaf batching (`MCTSConfig.leaf_batch > 1`) keeps the visit
     bookkeeping exact and collapses to the serial search at L = 1.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from balatro_sim.game import BalatroGame, State
from balatro_sim.jokers.base import JokerInstance
from mcts import (
    MCTS, MCTSConfig, NNPolicy, PolicyValueNet, UniformPolicy,
    BatchedNNPolicy, BatchedSearch, BatchedMCTSPlayerGroup, SearchRequest,
)
from mcts.action import action_from_key, action_key
from mcts.node import Node


# ── Fixtures ────────────────────────────────────────────────────────────────

def _mid_blind(seed: int = 42) -> BalatroGame:
    """SELECTING_HAND with 3 jokers — the demo / benchmark state (~436 actions)."""
    g = BalatroGame(seed=seed)
    g.step({"type": "play_blind"})
    g.dollars = 30
    g.jokers = [
        JokerInstance("j_joker"),
        JokerInstance("j_green_joker"),
        JokerInstance("j_steel_joker"),
    ]
    return g


def _pvp_wait(seed: str = "7I4M53DL") -> BalatroGame:
    """An MLB game parked in `PVP_WAIT`: no legal actions, must never reach the net."""
    g = BalatroGame(seed=seed, ruleset="mlb")
    for _ in range(80):
        if g.current_blind.is_pvp and g.state in (State.BLIND_SELECT, State.SELECTING_HAND):
            break
        if g.state is State.BLIND_SELECT:
            g.step({"type": "play_blind"})
        elif g.state is State.SELECTING_HAND:
            g.debug_win_blind()
        elif g.state is State.ROUND_EVAL:
            g.step({"type": "advance"})
        elif g.state is State.SHOP:
            g.step({"type": "leave_shop"})
        elif g.state is State.BOOSTER_OPEN:
            g.step({"type": "skip_booster"})
        else:
            raise AssertionError(f"unexpected state: {g.state}")
    g.pvp_solo = False
    g.step({"type": "play_blind"})
    g._start_blind()
    while g.state is State.SELECTING_HAND:
        g.step({"type": "play", "cards": [0, 1, 2, 3, 4]})
    assert g.state is State.PVP_WAIT and g.legal_actions() == []
    return g


@pytest.fixture(scope="module")
def net() -> PolicyValueNet:
    torch.manual_seed(0)
    return PolicyValueNet()


@pytest.fixture(scope="module")
def single(net) -> NNPolicy:
    return NNPolicy(net, device="cpu")


@pytest.fixture(scope="module")
def batched(net) -> BatchedNNPolicy:
    return BatchedNNPolicy(net, device="cpu")


def _tv(a: dict, b: dict, n: int) -> float:
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys) / max(1, n)


# ── 1. The policy: batched == single leaf ───────────────────────────────────

def test_batched_policy_matches_single_leaf(single, batched):
    """Same leaves, same priors and values.

    NOT bit-exact, and it cannot be: the trunk of a B-state batch is a (B, 447) matmul
    while a single leaf is a (447,) one, so BLAS blocks the reduction differently. The
    measured gap on this box is ~2e-10 on priors and ~1e-6 on values — far below
    anything the search can see, but `==` would be a false claim.
    """
    games = [_mid_blind(s) for s in (42, 7, 3)]
    want = [single(g) for g in games]
    got = batched.evaluate_many(games)

    assert len(got) == len(want)
    for (pw, vw), (pg, vg) in zip(want, got):
        assert set(pw) == set(pg)
        assert max(abs(pw[k] - pg[k]) for k in pw) < 1e-6
        assert abs(vw - vg) < 1e-4
        assert abs(sum(pg.values()) - 1.0) < 1e-5


def test_batched_policy_chunking_changes_nothing(net):
    """A batch too big for one forward pass is split and stitched, not truncated."""
    games = [_mid_blind(s) for s in (42, 7, 3, 11)]
    whole = BatchedNNPolicy(net, device="cpu")
    chunked = BatchedNNPolicy(net, device="cpu", max_action_rows=500)   # ~1 game/pass

    a = whole.evaluate_many(games)
    b = chunked.evaluate_many(games)
    assert chunked.forwards > whole.forwards >= 1
    for (pa, va), (pb, vb) in zip(a, b):
        assert set(pa) == set(pb)
        assert max(abs(pa[k] - pb[k]) for k in pa) < 1e-6
        assert abs(va - vb) < 1e-4


def test_batched_policy_skips_no_action_games(batched):
    """MLB `PVP_WAIT` in the middle of a batch: ({}, 0.0) and the net never sees it."""
    games = [_mid_blind(42), _pvp_wait(), _mid_blind(7)]
    before = batched.leaves
    out = batched.evaluate_many(games)

    assert out[1] == ({}, 0.0)
    assert out[0][0] and out[2][0]
    assert batched.leaves - before == 2          # only the two real leaves
    assert batched.batch_sizes[-1] == 2


def test_batched_policy_on_an_all_stuck_batch(batched):
    out = batched.evaluate_many([_pvp_wait(), _pvp_wait()])
    assert out == [({}, 0.0), ({}, 0.0)]


# ── 2. K trees batched == K trees alone ─────────────────────────────────────

@pytest.mark.parametrize("strategy", ["gumbel", "puct"])
def test_k_trees_match_k_single_tree_searches(strategy):
    """The scheduler contract, with a policy that has no float wobble at all: K trees in
    lockstep produce exactly the trees K separate searches would."""
    K = 6
    policy = UniformPolicy()
    cfg = MCTSConfig(num_simulations=60)
    games = [_mid_blind(42) for _ in range(K)]

    batch = BatchedSearch(policy, cfg, strategy=strategy)
    got = batch.run_many(games, seeds=list(range(K)), add_noise=False)

    for i in range(K):
        mcts = MCTS(policy, cfg, rng=np.random.default_rng(i))
        if strategy == "gumbel":
            _, visits, chosen = mcts.run_gumbel(_mid_blind(42))
            assert got[i].chosen == chosen
        else:
            _, visits = mcts.run(_mid_blind(42), add_noise=False)
        assert got[i].visit_counts == visits
    assert batch.stats.rounds > 0
    assert max(batch.stats.batch_sizes) == K


@pytest.mark.parametrize("strategy", ["gumbel", "puct"])
def test_k_trees_match_with_the_nn_policy(single, batched, strategy):
    """Same claim with the real net. The tolerance is for the 1e-10 prior differences of
    a different batch shape; it measured 0.0 on this box for every configuration tried."""
    K = 4
    cfg = MCTSConfig(num_simulations=80)
    games = [_mid_blind(42) for _ in range(K)]

    batch = BatchedSearch(batched, cfg, strategy=strategy)
    got = batch.run_many(games, seeds=list(range(K)), add_noise=False)

    for i in range(K):
        mcts = MCTS(single, cfg, rng=np.random.default_rng(i))
        if strategy == "gumbel":
            _, visits, chosen = mcts.run_gumbel(_mid_blind(42))
            assert got[i].chosen == chosen
        else:
            _, visits = mcts.run(_mid_blind(42), add_noise=False)
        assert _tv(got[i].visit_counts, visits, cfg.num_simulations) <= 0.01


def test_heterogeneous_budgets_batch_together(batched):
    """Different agents, different simulation budgets, one batch."""
    cfg = MCTSConfig(num_simulations=40)
    games = [_mid_blind(42), _mid_blind(7), _mid_blind(3)]
    batch = BatchedSearch(batched, cfg, strategy="puct")
    requests = [
        SearchRequest(game=g, mcts=MCTS(batched, cfg, rng=np.random.default_rng(i)),
                      strategy="puct", sims=sims, add_noise=False)
        for i, (g, sims) in enumerate(zip(games, (5, 40, 17)))
    ]
    got = batch.run_requests(requests)
    assert [sum(r.visit_counts.values()) for r in got] == [5, 40, 17]


# ── 3. Early finish ─────────────────────────────────────────────────────────

def test_trees_finishing_early_do_not_stall_the_batch(batched):
    """A batch of very different lengths, plus a root with no legal actions at all.
    Every tree must come back, and the batch must shrink as trees retire — not wait."""
    cfg = MCTSConfig(num_simulations=30)
    games = [_mid_blind(42), _pvp_wait(), _mid_blind(7), _mid_blind(3)]
    sims = [3, None, 30, 1]
    batch = BatchedSearch(batched, cfg, strategy="puct")
    requests = [
        SearchRequest(game=g, mcts=MCTS(batched, cfg, rng=np.random.default_rng(i)),
                      strategy="puct", sims=s, add_noise=False)
        for i, (g, s) in enumerate(zip(games, sims))
    ]
    got = batch.run_requests(requests)

    assert len(got) == 4
    assert got[1].visit_counts == {}          # the PVP_WAIT root: no actions, no crash
    assert sum(got[0].visit_counts.values()) == 3
    assert sum(got[2].visit_counts.values()) == 30
    assert sum(got[3].visit_counts.values()) == 1
    # The first round carries three live trees; the last carries one.
    assert batch.stats.batch_sizes[0] == 3
    assert batch.stats.batch_sizes[-1] == 1
    assert batch.stats.batch_sizes == sorted(batch.stats.batch_sizes, reverse=True)


def test_an_all_terminal_batch_returns_immediately(batched):
    cfg = MCTSConfig(num_simulations=10)
    batch = BatchedSearch(batched, cfg, strategy="gumbel")
    got = batch.run_many([_pvp_wait(), _pvp_wait()], seeds=[0, 1])
    assert all(r.visit_counts == {} and r.chosen is None for r in got)
    assert batch.stats.rounds == 0            # nothing was ever evaluated


# ── 4. Virtual-loss leaf batching within one tree ───────────────────────────

@pytest.mark.parametrize("strategy", ["gumbel", "puct"])
def test_leaf_batch_one_is_the_serial_search(batched, strategy):
    cfg = MCTSConfig(num_simulations=50, leaf_batch=1)
    batch = BatchedSearch(batched, cfg, strategy=strategy)
    got = batch.run_many([_mid_blind(42)], seeds=[0], add_noise=False)[0]

    mcts = MCTS(batched, cfg, rng=np.random.default_rng(0))
    if strategy == "gumbel":
        _, visits, chosen = mcts.run_gumbel(_mid_blind(42))
        assert got.chosen == chosen
    else:
        _, visits = mcts.run(_mid_blind(42), add_noise=False)
    assert got.visit_counts == visits
    assert batch.stats.rounds == batch.stats.leaves      # one leaf per round


@pytest.mark.parametrize("L", [4, 16])
@pytest.mark.parametrize("strategy", ["gumbel", "puct"])
def test_virtual_loss_keeps_the_visit_bookkeeping_exact(batched, strategy, L):
    """L in flight changes WHICH leaves get visited (that is the approximation) but must
    not change how many visits exist, nor leave a node counted twice."""
    sims = 64
    cfg = MCTSConfig(num_simulations=sims, leaf_batch=L)
    batch = BatchedSearch(batched, cfg, strategy=strategy)
    res = batch.run_many([_mid_blind(42)], seeds=[0], add_noise=False)[0]

    assert sum(res.visit_counts.values()) == sims
    assert res.root.visit_count == sims
    assert batch.stats.leaves >= sims                    # + the root expansion
    assert batch.stats.rounds < sims                     # fewer forward passes
    assert batch.stats.mean_batch > 1.0
    stack = [res.root]
    while stack:
        n = stack.pop()
        assert n.visit_count >= 0
        if n.children:
            assert sum(c.visit_count for c in n.children.values()) <= n.visit_count
            stack.extend(n.children.values())


def test_leaf_batch_reaches_the_same_kind_of_answer(batched):
    """Sanity, not equality: L=8 still concentrates its visits (it is a search, not
    noise) and still returns a legal action."""
    game = _mid_blind(42)
    legal = {action_key(a) for a in game.legal_actions()}
    cfg = MCTSConfig(num_simulations=100, leaf_batch=8)
    batch = BatchedSearch(batched, cfg, strategy="gumbel")
    res = batch.run_many([game], seeds=[0])[0]
    assert res.chosen in legal
    assert max(res.visit_counts.values()) >= 100 // 16


# ── 5. The player group ─────────────────────────────────────────────────────

def test_group_act_many_matches_individual_act():
    """N agents decided in one batch == N agents decided one at a time (uniform policy,
    so this is exact), including a stuck agent and a singleton-shortcut agent."""
    policy = UniformPolicy()
    cfg = MCTSConfig(num_simulations=25)
    games = [_mid_blind(42), _pvp_wait(), BalatroGame(seed=42), _mid_blind(7)]

    group = BatchedMCTSPlayerGroup(len(games), policy, cfg, strategy="gumbel",
                                   seeds=list(range(len(games))), reuse=False)
    got = group.act_keys([g.clone() for g in games])

    from mcts import MCTSPlayer
    want = [MCTSPlayer(policy=policy, config=cfg, rng=np.random.default_rng(i),
                       strategy="gumbel", reuse=False).act_key(g.clone())
            for i, g in enumerate(games)]
    assert got == want
    assert got[1] is None                    # PVP_WAIT
    assert got[0] is not None and got[3] is not None


def test_group_actions_are_legal_and_steppable(batched):
    cfg = MCTSConfig(num_simulations=12)
    group = BatchedMCTSPlayerGroup(3, batched, cfg, strategy="gumbel",
                                   seeds=[0, 1, 2], reuse=True)
    games = [BalatroGame(seed=s, ruleset="mlb") for s in ("7I4M53DL", "ABCD1234", "ZZZZ9999")]
    for _ in range(6):
        actions = group.act_many(games)
        for g, a in zip(games, actions):
            if a is None:
                continue
            assert a in g.legal_actions() or len(g.legal_actions()) == 1
            g.step(a)
    assert group.stats.rounds > 0
    assert group.stats.mean_batch > 1.0


def test_group_handles_none_games(batched):
    """A dead agent is passed as None (the tournament drops agents mid-run)."""
    cfg = MCTSConfig(num_simulations=8)
    group = BatchedMCTSPlayerGroup(3, batched, cfg, seeds=[0, 1, 2])
    out = group.act_many([_mid_blind(42), None, _mid_blind(7)])
    assert out[1] is None and out[0] is not None and out[2] is not None


# ── 6. The generator refactor did not change the serial search ──────────────

class _PreW3MCTS(MCTS):
    """A verbatim copy of `search.py`'s run / run_gumbel / _simulate / _expand as they
    stood at W1's hand-off (before leaf evaluation became a generator seam). Kept here so
    "single-tree behaviour is unchanged" is a test, not a claim."""

    def _expand_old(self, node, game):
        outcome = self._outcome_for(game)
        if outcome.is_terminal(game):
            node.is_expanded = True
            return self._mark_stop(node, game, "game_over")
        if outcome.is_stuck(game):
            node.is_expanded = True
            return self._mark_stop(node, game, "stuck")
        priors, value = self._evaluate_leaf(game)
        if not priors:
            node.is_expanded = True
            return self._mark_stop(node, game, "no_actions")
        for k, p in priors.items():
            node.add_child(k, prior=p)
        node.is_expanded = True
        return value

    def _simulate_old(self, root, game, force_first=None):
        outcome = self._outcome_for(game)
        path = [root]
        node = root
        first_step = True
        while node.is_expanded and not node.is_terminal:
            if not node.children:
                self._mark_stop(node, game, "no_actions")
                break
            if first_step and force_first is not None:
                action_k = force_first
                child = node.children[action_k]
            else:
                action_k, child = self._select_child(node)
            first_step = False
            game.step(action_from_key(action_k))
            path.append(child)
            node = child
            if outcome.is_terminal(game):
                self._mark_stop(node, game, "game_over")
                break
            if outcome.is_stuck(game):
                self._mark_stop(node, game, "stuck")
                break
        if not node.is_terminal and not node.is_expanded:
            value = self._expand_old(node, game)
        else:
            value = node.terminal_value
        for n in path:
            n.visit_count += 1
            n.value_sum += value

    def run(self, root_game, add_noise=True, root=None, sims=None):
        self._outcome_for(root_game)
        root = Node()
        self._expand_old(root, root_game)
        if not root.children:
            return root, {}
        if add_noise:
            self._add_dirichlet_noise(root)
        for _ in range(self.cfg.num_simulations):
            self._simulate_old(root, root_game.clone())
        return root, {k: c.visit_count for k, c in root.children.items()}

    def run_gumbel(self, root_game, root=None, sims=None):
        self._outcome_for(root_game)
        root = Node()
        self._expand_old(root, root_game)
        if root.is_terminal or not root.children:
            return root, {}, None
        legal_keys = list(root.children.keys())
        n_legal = len(legal_keys)
        m_init = min(n_legal, self.cfg.gumbel_max_considered)
        priors = np.array([max(root.children[k].prior, 1e-12) for k in legal_keys])
        logits = np.log(priors)
        u = np.maximum(self.rng.random(n_legal), 1e-12)
        gumbel_noise = -np.log(-np.log(u))
        surv = np.argsort(-(gumbel_noise + logits))[:m_init].tolist()
        sims_remaining = self.cfg.num_simulations
        n_phases = max(1, math.ceil(math.log2(max(m_init, 2))))
        sims_per_phase = max(1, self.cfg.num_simulations // n_phases)
        while len(surv) > 1 and sims_remaining > 0:
            per_action = max(1, sims_per_phase // len(surv))
            for ai in surv:
                if sims_remaining <= 0:
                    break
                for _ in range(per_action):
                    if sims_remaining <= 0:
                        break
                    self._simulate_old(root, root_game.clone(),
                                       force_first=legal_keys[ai])
                    sims_remaining -= 1
            scores = self._gumbel_scores(root, legal_keys, surv, gumbel_noise, logits)
            surv = [surv[i] for i in np.argsort(-scores)[:max(1, len(surv) // 2)]]
        while sims_remaining > 0 and surv:
            for ai in surv:
                if sims_remaining <= 0:
                    break
                self._simulate_old(root, root_game.clone(), force_first=legal_keys[ai])
                sims_remaining -= 1
        if len(surv) > 1:
            scores = self._gumbel_scores(root, legal_keys, surv, gumbel_noise, logits)
            chosen_idx = surv[int(np.argmax(scores))]
        else:
            chosen_idx = surv[0]
        return (root, {k: c.visit_count for k, c in root.children.items()},
                legal_keys[chosen_idx])


def _tree_signature(node: Node):
    out = [(node.visit_count, node.value_sum, node.is_terminal, node.stop_reason)]
    for k in sorted(node.children, key=repr):
        out.append((repr(k), node.children[k].prior, _tree_signature(node.children[k])))
    return out


@pytest.mark.parametrize("strategy", ["puct", "gumbel"])
@pytest.mark.parametrize("sims", [37, 120])
def test_serial_search_matches_the_pre_w3_implementation(single, strategy, sims):
    """Whole-tree equality — visits, value sums, priors, stop reasons — against the
    pre-refactor loops. Not "close": identical."""
    cfg = MCTSConfig(num_simulations=sims)
    new = MCTS(single, cfg, rng=np.random.default_rng(3))
    old = _PreW3MCTS(single, cfg, rng=np.random.default_rng(3))
    game = _mid_blind(42)

    if strategy == "puct":
        rn, vn = new.run(game)
        ro, vo = old.run(game)
        cn = co = None
    else:
        rn, vn, cn = new.run_gumbel(game)
        ro, vo, co = old.run_gumbel(game)

    assert vn == vo
    assert cn == co
    assert _tree_signature(rn) == _tree_signature(ro)


# ── 7. The hottest CPU call in the search ───────────────────────────────────

def test_featurize_actions_matches_per_action():
    """`featurize_actions` writes one fancy-indexed block instead of stacking N little
    arrays (W3: it is ~0.3 ms per leaf and runs once per simulation). The block must be
    byte-identical to the per-action function for every legal action of a real game."""
    from mcts.action_features import featurize_action, featurize_actions

    rng = np.random.default_rng(0)
    checked = 0
    for seed, ruleset in ((42, "vanilla"), ("7I4M53DL", "mlb")):
        game = BalatroGame(seed=seed, ruleset=ruleset)
        for _ in range(40):
            legal = game.legal_actions()
            if not legal or game.state is State.GAME_OVER:
                break
            fast = featurize_actions(legal)
            slow = np.stack([featurize_action(a) for a in legal], axis=0)
            assert np.array_equal(fast, slow)
            checked += 1
            game.step(legal[int(rng.integers(len(legal)))])
    assert checked >= 20
    assert featurize_actions([]).shape == (0, slow.shape[1])


# ── 8. The tournament plug-in ───────────────────────────────────────────────

def test_make_player_is_a_tournament_shaped_player():
    """`make_player()` is the whole `tournament/players.py` diff: it must produce
    something whose `act(game)` is always steppable and which has `reset()`."""
    from mcts import make_player

    player = make_player(sims=6, device="cpu", seed=0)
    game = BalatroGame(seed="7I4M53DL", ruleset="mlb")
    for _ in range(6):
        action = player.act(game)
        assert isinstance(action, dict)          # never None: `no_action` is set
        game.step(action)
    player.reset()
    assert not player.cache.armed


def test_no_action_default_is_still_none(batched):
    """W1's contract for `MCTSPlayer` itself is unchanged — only the factory opts in."""
    from mcts import MCTSPlayer

    player = MCTSPlayer(policy=batched, config=MCTSConfig(num_simulations=4))
    assert player.act(_pvp_wait()) is None


def test_load_policy_round_trips_a_checkpoint(tmp_path, net):
    """`load_policy` reads what `train_cold` writes: net description, weights, encoder."""
    from train.checkpoint import save_checkpoint
    from mcts import load_policy

    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, {"config": {}, "net_desc": net.describe(),
                           "encoder": "v7", "model": net.state_dict()})
    policy = load_policy(str(path), device="cpu")
    game = _mid_blind(42)
    priors, value = policy(game)
    want_priors, want_value = NNPolicy(net, device="cpu")(game)
    assert set(priors) == set(want_priors)
    assert value == pytest.approx(want_value, abs=1e-6)
