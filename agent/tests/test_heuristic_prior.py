"""
test_heuristic_prior.py — W0's heuristic hand prior (`mcts/heuristic.py`) and the
search-side candidate mask.

The five claims that have to hold, in the order the brief asks for them:

  1. **Side-effect-free.** Scoring a leaf leaves `state_signature()` bit-identical.
     This is the one that would quietly corrupt a 30-hour run: the pre-2026-08-21 envs
     scored hypotheticals on the LIVE rng and joker state (MP_UPDATE_LIST §3), so a
     reward estimate advanced the seeded stream before the real play.
  2. **It is a distribution, and it favours the dry-run best play.**
  3. **lambda = 0 and no mask reproduce the pre-W0 search byte-identically** — the same
     visit counts, the same chosen action, the same tree.
  4. **The mask keeps the best / the chosen action**, and every pruned action is still
     legal in the engine.
  5. **The checkpoint carries lambda** (and the clear rate that anneals it), so a resume
     does not restart the anneal at lambda0.
"""
from __future__ import annotations

import numpy as np
import pytest

from balatro_sim.card_selection import HypotheticalScorer
from balatro_sim.game import BalatroGame, State
from balatro_sim.hand_eval import evaluate_hand

from mcts import MCTS, MCTSConfig, MCTSPlayer, UniformPolicy
from mcts.action import action_key
from mcts.heuristic import (
    HandHeuristic, HeuristicConfig, hand_action_scores, heuristic_distribution,
    shape_priors,
)


# ── fixtures ────────────────────────────────────────────────────────────────────

def hand_state(seed: int = 12345, ruleset: str = "vanilla") -> BalatroGame:
    """A live `SELECTING_HAND` game — ante 1, Small blind, 8 cards, no jokers."""
    g = BalatroGame(seed=seed, deck_key="b_red", stake=1, ruleset=ruleset)
    for _ in range(30):
        if g.state is State.SELECTING_HAND:
            return g
        legal = g.legal_actions()
        act = next((a for a in legal if a["type"] == "play_blind"), legal[0])
        g.step(act)
    raise AssertionError("never reached SELECTING_HAND")


def uniform_priors(game: BalatroGame) -> dict:
    legal = game.legal_actions()
    return {action_key(a): 1.0 / len(legal) for a in legal}


@pytest.fixture
def game():
    return hand_state()


@pytest.fixture
def priors(game):
    return uniform_priors(game)


# ── 1. side-effect-free ─────────────────────────────────────────────────────────

def test_scoring_does_not_touch_the_game(game, priors):
    before = game.state_signature()
    scores = hand_action_scores(game, priors, HeuristicConfig(exact_top=32))
    assert scores
    assert game.state_signature() == before


def test_shaping_does_not_touch_the_game(game, priors):
    before = game.state_signature()
    shape_priors(game, priors, lam=0.8, max_candidates=32)
    assert game.state_signature() == before


def test_scoring_is_a_pure_function_of_the_state(game, priors):
    cfg = HeuristicConfig(exact_top=32)
    a = hand_action_scores(game, priors, cfg)
    b = hand_action_scores(game, priors, cfg)
    assert a == b


def test_the_exact_tier_does_not_advance_the_run_rng(game, priors):
    """`HypotheticalScorer` clones `run_state.rng`; the live stream must not move."""
    before = game.run_state.rng.snapshot()
    hand_action_scores(game, priors, HeuristicConfig(exact_top=32))
    assert game.run_state.rng.snapshot() == before


# ── 2. a distribution that favours the best play ────────────────────────────────

def test_scores_cover_exactly_the_hand_actions(game, priors):
    scores = hand_action_scores(game, priors)
    hand_keys = {k for k in priors if k[0] in ("play", "discard")}
    assert set(scores) == hand_keys
    assert hand_keys, "the fixture state must have play/discard actions"


def test_prior_is_a_distribution(game, priors):
    out = shape_priors(game, priors, lam=0.8, max_candidates=32)
    assert all(v >= 0.0 for v in out.values())
    assert sum(out.values()) == pytest.approx(1.0)


def test_prior_favours_the_dry_run_best_play(game, priors):
    """Among PLAY actions the highest prior is the highest-scoring dry run."""
    cfg = HeuristicConfig(exact_top=32)
    scores = hand_action_scores(game, priors, cfg)
    out = shape_priors(game, priors, lam=1.0, max_candidates=0, cfg=cfg)
    plays = [k for k in out if k[0] == "play"]
    best_by_prior = max(plays, key=lambda k: out[k])
    assert scores[best_by_prior] == pytest.approx(max(scores[k] for k in plays))
    # ...and it beats a deliberately terrible play (one low card on its own).
    worst = min(plays, key=lambda k: scores[k])
    assert out[best_by_prior] > out[worst]


def test_a_flush_outscores_a_high_card(game):
    """The scores are not just monotone in something — they rank hand TYPES correctly."""
    h = HandHeuristic(game, HeuristicConfig(exact_top=0))
    combos, types = [], []
    for k in range(1, 6):
        for i in range(0, max(1, h.n - k + 1)):
            combos.append(tuple(range(i, i + k)))
    scores = h.play_scores(combos)
    by_type: dict = {}
    for combo, s in zip(combos, scores):
        ht, _ = evaluate_hand([game.hand[j] for j in combo], **h.flags)
        by_type.setdefault(ht, []).append(s)
    assert "High Card" in by_type
    for better in ("Pair", "Two Pair", "Flush", "Straight"):
        if better in by_type:
            assert max(by_type[better]) > max(by_type["High Card"])


def test_cheap_equals_the_dry_run_on_a_plain_board(game):
    """The claim `cheap_is_exact` makes: with no jokers and plain cards, the cheap tier
    IS `score_hand`. If this ever fails the automatic skip must go."""
    h = HandHeuristic(game, HeuristicConfig(exact_top=0))
    assert h.cheap_is_exact, "the ante-1 fixture should be a plain board"
    scorer = HypotheticalScorer(game, model_held=True)
    combos = [(0,), (0, 1), (0, 1, 2), (0, 1, 2, 3), (0, 1, 2, 3, 4)][: max(1, h.n - 3)]
    cheap = h.play_scores(combos)
    for combo, c in zip(combos, cheap):
        cards = [game.hand[j] for j in combo]
        ht, scoring = evaluate_hand(cards, **h.flags)
        assert c == pytest.approx(float(scorer.score(cards, ht, scoring)))


def test_discard_beats_play_when_the_hand_is_junk_but_a_flush_is_live(game):
    """The whole point: a 4-card flush draw must out-rank the best High Card."""
    scores = hand_action_scores(game, uniform_priors(game))
    best_play = max(v for k, v in scores.items() if k[0] == "play")
    best_discard = max(v for k, v in scores.items() if k[0] == "discard")
    # the fixture hand is QH 9S 6D 5C 2C AS 4C 8C — four clubs, nothing made
    assert best_discard > best_play


def test_lambda_only_moves_mass_between_hand_actions(game):
    """Non-hand actions (a Tarot's `use_consumable`) keep exactly the net's prior."""
    priors = uniform_priors(game)
    # a deliberately lopsided "net" so an unchanged value is meaningful
    rng = np.random.default_rng(0)
    w = rng.random(len(priors)) + 0.1
    net = {k: float(v) for k, v in zip(priors, w / w.sum())}
    out = shape_priors(game, net, lam=1.0, max_candidates=0)
    others = [k for k in net if k[0] not in ("play", "discard")]
    for k in others:
        assert out[k] == pytest.approx(net[k])
    # and the hand block keeps its total mass
    hand_mass = sum(net[k] for k in net if k[0] in ("play", "discard"))
    assert sum(out[k] for k in out if k[0] in ("play", "discard")) == pytest.approx(hand_mass)


def test_tau_sharpens(game, priors):
    sharp = shape_priors(game, priors, lam=1.0, cfg=HeuristicConfig(tau=0.25, exact_top=0))
    flat = shape_priors(game, priors, lam=1.0, cfg=HeuristicConfig(tau=2.0, exact_top=0))
    assert max(sharp.values()) > max(flat.values())


def test_lambda_interpolates(game, priors):
    scores = hand_action_scores(game, priors)
    h = heuristic_distribution(priors, scores, tau=0.5)
    mid = shape_priors(game, priors, lam=0.5, max_candidates=0)
    for k in priors:
        assert mid[k] == pytest.approx(0.5 * priors[k] + 0.5 * h[k])


def test_non_hand_states_are_returned_unchanged():
    g = BalatroGame(seed=7, deck_key="b_red", stake=1, ruleset="vanilla")
    assert g.state is State.BLIND_SELECT
    p = uniform_priors(g)
    assert shape_priors(g, p, lam=1.0, max_candidates=4) is p
    assert hand_action_scores(g, p) is None


# ── 3. lambda = 0 reproduces the search ─────────────────────────────────────────

def _search(cfg_kwargs: dict, seed: int = 3):
    policy = UniformPolicy()
    cfg = MCTSConfig(num_simulations=24, **cfg_kwargs)
    mcts = MCTS(policy, cfg, rng=np.random.default_rng(seed))
    return mcts.run_gumbel(hand_state())


def test_lambda_zero_reproduces_the_search():
    """Defaults off == the pre-W0 search, node for node."""
    root_a, visits_a, chosen_a = _search({})
    root_b, visits_b, chosen_b = _search({"heuristic_prior_weight": 0.0,
                                          "max_hand_candidates": 0})
    assert chosen_a == chosen_b
    assert visits_a == visits_b
    assert set(root_a.children) == set(root_b.children)
    for k in root_a.children:
        assert root_a.children[k].prior == root_b.children[k].prior
        assert root_a.children[k].visit_count == root_b.children[k].visit_count


def test_shape_priors_returns_the_same_object_when_inert(game, priors):
    assert shape_priors(game, priors, lam=0.0, max_candidates=0) is priors


def test_the_prior_changes_what_the_search_picks():
    """Sanity: the whole exercise would be pointless if it did not."""
    _, _, cold = _search({})
    _, _, warm = _search({"heuristic_prior_weight": 1.0, "max_hand_candidates": 32})
    assert cold != warm


# ── 4. the mask ─────────────────────────────────────────────────────────────────

def test_mask_keeps_top_k_of_each_kind_plus_everything_else(game, priors):
    k = 12
    out = shape_priors(game, priors, lam=0.8, max_candidates=k)
    plays = [a for a in out if a[0] == "play"]
    discards = [a for a in out if a[0] == "discard"]
    others = [a for a in out if a[0] not in ("play", "discard")]
    assert len(plays) == k
    assert len(discards) == k
    assert others == [a for a in priors if a[0] not in ("play", "discard")]


def test_mask_keeps_the_best_action(game, priors):
    """The best SCORE of each kind survives. Not "the best key": scores tie constantly
    (every subset whose scoring cards are the same Ace scores the same), and which of a
    tied group the sort keeps is arbitrary and does not matter."""
    scores = hand_action_scores(game, priors)
    out = shape_priors(game, priors, lam=0.8, max_candidates=8)
    for atype in ("play", "discard"):
        best = max(scores[k] for k in scores if k[0] == atype)
        kept = max(scores[k] for k in out if k[0] == atype)
        assert kept == pytest.approx(best)


def test_mask_preserves_legal_action_order(game, priors):
    out = shape_priors(game, priors, lam=0.8, max_candidates=16)
    order = [k for k in priors if k in out]
    assert list(out) == order


def test_pruned_actions_are_still_legal_in_the_engine(game, priors):
    """The mask is search-side. Everything it drops must still be steppable."""
    out = shape_priors(game, priors, lam=0.8, max_candidates=4)
    dropped = [k for k in priors if k not in out]
    assert dropped
    legal = {action_key(a) for a in game.legal_actions()}
    assert set(dropped) <= legal


def test_masked_search_expands_only_the_survivors():
    root, visits, chosen = _search({"heuristic_prior_weight": 0.8,
                                    "max_hand_candidates": 16})
    n_hand = sum(1 for k in root.children if k[0] in ("play", "discard"))
    assert n_hand <= 32
    assert chosen in root.children
    assert visits[chosen] > 0


def test_the_chosen_action_is_visited_so_a_sample_keeps_it():
    """`Sample` v2 keeps every VISITED action; the chosen one must be among them or the
    policy target would not carry the decision."""
    root, visits, chosen = _search({"heuristic_prior_weight": 0.8,
                                    "max_hand_candidates": 8})
    assert visits.get(chosen, 0) > 0


def test_mask_bigger_than_the_action_set_is_a_no_op(game, priors):
    out = shape_priors(game, priors, lam=0.0, max_candidates=10_000)
    assert set(out) == set(priors)
    for k in priors:
        assert out[k] == pytest.approx(priors[k])


def test_a_psychic_boss_only_ever_sees_five_card_plays():
    """Keys come FROM the priors, so a boss restriction is respected by construction."""
    g = hand_state()
    g.current_blind.boss_key = "bl_psychic"
    priors = uniform_priors(g)
    assert priors, "psychic still has legal actions"
    scores = hand_action_scores(g, priors)
    for k in scores:
        if k[0] == "play":
            assert len(k[1]) == 5


# ── 5. the player / trainer seams ───────────────────────────────────────────────

def test_player_carries_its_own_lambda():
    p = MCTSPlayer(policy=UniformPolicy(),
                   config=MCTSConfig(num_simulations=4, heuristic_prior_weight=0.3),
                   heuristic_prior=0.9)
    assert p.heuristic_weight == pytest.approx(0.9)
    p.set_heuristic_prior(None)
    assert p.heuristic_weight == pytest.approx(0.3)     # falls back to the config
    p.set_heuristic_prior(0.0)
    assert p.heuristic_weight == 0.0


def test_player_lambda_does_not_leak_into_a_shared_config():
    cfg = MCTSConfig(num_simulations=4, heuristic_prior_weight=0.2)
    a = MCTSPlayer(policy=UniformPolicy(), config=cfg, heuristic_prior=1.0)
    b = MCTSPlayer(policy=UniformPolicy(), config=cfg)
    assert a.heuristic_weight == 1.0
    assert b.heuristic_weight == pytest.approx(0.2)
    assert cfg.heuristic_prior_weight == pytest.approx(0.2)


def test_make_player_wires_the_prior():
    from mcts import make_player
    p = make_player(sims=4, encoder="v7", heuristic_prior=0.7, max_hand_candidates=24)
    assert p.config.heuristic_prior_weight == pytest.approx(0.7)
    assert p.config.max_hand_candidates == 24
    # ...and the default is still OFF for every existing caller
    q = make_player(sims=4, encoder="v7")
    assert q.config.heuristic_prior_weight == 0.0
    assert q.config.max_hand_candidates == 0


def _trainer(**kw):
    from train import ColdTrainer, TrainConfig
    cfg = TrainConfig(seed=1, sims=2, max_considered=2, encoder="v7",
                      ruleset="vanilla", max_decisions=4, min_buffer=10**9, **kw)
    return ColdTrainer(cfg)


def test_trainer_pushes_lambda_into_the_search():
    t = _trainer(heuristic_prior=0.8, max_hand_candidates=16)
    assert t.agent.mcts.heuristic_weight == pytest.approx(0.8)
    assert t.mcts_cfg.max_hand_candidates == 16


def test_episode_anneal_by_episodes():
    t = _trainer(heuristic_prior=1.0, heuristic_prior_floor=0.2,
                 heuristic_prior_anneal="ep:100")
    assert t.annealed_lambda() == pytest.approx(1.0)
    t.counters.episodes = 50
    assert t.annealed_lambda() == pytest.approx(0.6)
    t.counters.episodes = 100
    assert t.annealed_lambda() == pytest.approx(0.2)
    t.counters.episodes = 10_000
    assert t.annealed_lambda() == pytest.approx(0.2)          # never below the floor


def test_episode_anneal_bare_integer_means_episodes():
    t = _trainer(heuristic_prior=1.0, heuristic_prior_floor=0.0,
                 heuristic_prior_anneal="100")
    t.counters.episodes = 25
    assert t.annealed_lambda() == pytest.approx(0.75)


def test_clear_rate_anneal():
    t = _trainer(heuristic_prior=0.8, heuristic_prior_floor=0.1,
                 heuristic_prior_anneal="clear:0.5")
    assert t.annealed_lambda() == pytest.approx(0.8)          # no rate yet
    t.clear_rate_ema = 0.0
    assert t.annealed_lambda() == pytest.approx(0.8)
    t.clear_rate_ema = 0.25
    assert t.annealed_lambda() == pytest.approx(0.1 + 0.7 * 0.5)
    t.clear_rate_ema = 0.9
    assert t.annealed_lambda() == pytest.approx(0.1)


def test_no_anneal_holds_lambda():
    t = _trainer(heuristic_prior=0.8, heuristic_prior_anneal="")
    t.counters.episodes = 10_000
    t.clear_rate_ema = 1.0
    assert t.annealed_lambda() == pytest.approx(0.8)


def test_anneal_is_inert_when_the_prior_is_off():
    t = _trainer(heuristic_prior=0.0, heuristic_prior_anneal="clear:0.5")
    t.clear_rate_ema = 1.0
    assert t.annealed_lambda() == 0.0


def test_clear_rate_ema_moves_with_the_episodes():
    t = _trainer(heuristic_prior=0.5)
    t._update_clear_rate(True)
    assert t.clear_rate_ema == pytest.approx(1.0)             # seeded by the first value
    for _ in range(10):
        t._update_clear_rate(False)
    assert 0.0 < t.clear_rate_ema < 1.0


def test_checkpoint_round_trip_carries_lambda(tmp_path):
    from train import ColdTrainer
    from train.checkpoint import load_checkpoint, save_checkpoint

    t = _trainer(heuristic_prior=0.9, heuristic_prior_floor=0.2,
                 heuristic_prior_anneal="ep:10", max_hand_candidates=16)
    t.counters.episodes = 5
    t.clear_rate_ema = 0.42
    t.heuristic_lambda = t.annealed_lambda()
    assert t.heuristic_lambda == pytest.approx(0.55)

    path = save_checkpoint(tmp_path / "ckpt.pt", t.state_dict(include_buffer=False))
    back = ColdTrainer.from_checkpoint(load_checkpoint(path))
    assert back.heuristic_lambda == pytest.approx(0.55)
    assert back.clear_rate_ema == pytest.approx(0.42)
    assert back.agent.mcts.heuristic_weight == pytest.approx(0.55)
    assert back.cfg.max_hand_candidates == 16
    assert back.cfg.heuristic_prior_anneal == "ep:10"


def test_an_old_checkpoint_without_the_key_still_loads(tmp_path):
    """Phase 4 checkpoints predate W0; they must read as lambda = 0, not crash."""
    from train import ColdTrainer
    from train.checkpoint import load_checkpoint, save_checkpoint

    t = _trainer()
    payload = t.state_dict(include_buffer=False)
    payload.pop("heuristic")
    path = save_checkpoint(tmp_path / "old.pt", payload)
    back = ColdTrainer.from_checkpoint(load_checkpoint(path))
    assert back.heuristic_lambda == 0.0
    assert back.clear_rate_ema is None


def test_episode_record_carries_the_metrics():
    t = _trainer(heuristic_prior=0.5, max_hand_candidates=8)
    rec = t.run_episode()
    assert rec["kind"] in ("episode", "error")
    if rec["kind"] == "episode":
        assert "blinds" in rec and "cleared" in rec
        assert rec["h_lambda"] == pytest.approx(0.5)
        assert rec["clear_rate"] is not None
