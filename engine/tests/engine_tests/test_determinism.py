"""
test_determinism.py — same seed must produce the same episode.

Regression tests for audit finding H3: ~24 stochastic joker sites and the Lucky
Card roll drew from the global `random` module instead of `game.rng`, so a seed
did not determine an episode. Seed control is required for reproducible training
runs and is load-bearing for MCTS state cloning.
"""
import random

import pytest

from balatro_sim.card import Card
from balatro_sim.game import BalatroGame, State
from balatro_sim.jokers.base import JokerInstance, ScoreContext, rng_of, MissingPRNG
from balatro_sim.scoring import score_hand
from balatro_sim.game_keys import core as _core
PseudoRandom = _core.PseudoRandom


def _play_scripted(seed, n_steps=60):
    """Drive a game with a fixed action policy and return a state trace."""
    g = BalatroGame(seed=seed)
    trace = []
    for _ in range(n_steps):
        if g.state == State.GAME_OVER:
            break
        if g.state == State.BLIND_SELECT:
            g.step({"type": "play_blind"})
        elif g.state == State.SELECTING_HAND:
            g.step({"type": "play", "cards": list(range(min(5, len(g.hand))))})
        elif g.state == State.SHOP:
            g.step({"type": "leave_shop"})
        elif g.state == State.ROUND_EVAL:
            g.step({"type": "noop"})
        elif g.state == State.BOOSTER_OPEN:
            g.step({"type": "skip_booster"})
        else:
            break
        trace.append((g.ante, g.blind_idx, g.chips_scored, g.dollars,
                      len(g.hand), len(g.deck), g.state.name))
    return trace


def test_same_seed_same_episode():
    assert _play_scripted(1234) == _play_scripted(1234)


def test_different_seeds_diverge():
    """Guards against the trace being trivially constant."""
    assert _play_scripted(1234) != _play_scripted(9999)


def test_global_random_does_not_affect_episode():
    """
    Perturbing the global RNG between two seeded runs must not change the
    outcome. This is the test that would have caught H3.
    """
    random.seed(0)
    a = _play_scripted(777)
    for _ in range(500):
        random.random()          # churn the global RNG
    random.seed(12345)
    b = _play_scripted(777)
    assert a == b


class TestLuckyCard:
    def test_lucky_uses_context_rng(self):
        """Lucky rolls must come from ctx.prng (the keyed PseudoRandom), not the global module."""
        card = Card(rank=10, suit="Spades", enhancement="Lucky")
        results = []
        for _ in range(2):
            score, _ctx = score_hand(
                scoring_cards=[card], all_cards=[card], hand_type="High Card",
                jokers=[], planet_levels={"High Card": 1}, hands_left=3,
                discards_left=3, dollars=4, ante=1, deck_remaining=40,
                rng=PseudoRandom("SEED42"),
            )
            results.append(score)
        assert results[0] == results[1]

    def test_lucky_mult_rate_is_one_in_five(self):
        """+20 Mult fires on 1 in 5 rolls, not 1 in 4 (audit H1)."""
        card = Card(rank=2, suit="Spades", enhancement="Lucky")
        rng = PseudoRandom("RATE")      # 4000 'lucky_mult' draws on one keyed stream
        trials, hits = 4000, 0
        for _ in range(trials):
            score, _ = score_hand(
                scoring_cards=[card], all_cards=[card], hand_type="High Card",
                jokers=[], planet_levels={"High Card": 1}, hands_left=3,
                discards_left=3, dollars=4, ante=1, deck_remaining=40, rng=rng,
            )
            # High Card base 5/1, card 2 chips -> 7 without the bonus,
            # 7 * 21 with it. Any score above base means the mult roll hit.
            if score > 7:
                hits += 1
        rate = hits / trials
        assert 0.17 < rate < 0.23, f"expected ~0.20, got {rate:.3f}"


def _play_with_joker(seed, joker_key, n_steps=40):
    """
    Same as _play_scripted but with a stochastic joker installed, and tracking
    the state it mutates. Without this, the plain trace exercises only shop/deck
    RNG (which already used game.rng) and would pass even with H3 unfixed.
    """
    g = BalatroGame(seed=seed)
    g.jokers.append(JokerInstance(joker_key))
    trace = []
    for _ in range(n_steps):
        if g.state == State.GAME_OVER:
            break
        if g.state == State.BLIND_SELECT:
            g.step({"type": "play_blind"})
        elif g.state == State.SELECTING_HAND:
            g.step({"type": "play", "cards": list(range(min(5, len(g.hand))))})
        elif g.state == State.SHOP:
            g.step({"type": "leave_shop"})
        elif g.state == State.ROUND_EVAL:
            g.step({"type": "noop"})
        elif g.state == State.BOOSTER_OPEN:
            g.step({"type": "skip_booster"})
        else:
            break
        trace.append((g.chips_scored, g.dollars,
                      tuple(sorted(g.planet_levels.items())),
                      tuple(sorted(g.jokers[0].state.items())) if g.jokers else ()))
    return trace


@pytest.mark.parametrize("joker_key", [
    "j_space",   # 1 in 4 to level up played hand
    "j_8_ball",        # 1 in 4 on each 8 played
    "j_business", # 1 in 2 per face card
    "j_misprint",      # random 0-23 mult every hand
    "j_bloodstone",    # 1 in 2 per Heart
])
def test_stochastic_joker_is_seed_determined(joker_key):
    """A stochastic joker must produce identical results for identical seeds."""
    assert _play_with_joker(1234, joker_key) == _play_with_joker(1234, joker_key)


@pytest.mark.parametrize("joker_key", [
    "j_space", "j_8_ball", "j_business", "j_misprint", "j_bloodstone",
])
def test_stochastic_joker_ignores_global_random(joker_key):
    """
    The real H3 regression test: churn the global RNG between two identically
    seeded runs that each hold a stochastic joker. Before the fix these jokers
    called the global `random` module, so this diverged.
    """
    random.seed(0)
    a = _play_with_joker(2468, joker_key)
    for _ in range(1000):
        random.random()
    random.seed(99999)
    b = _play_with_joker(2468, joker_key)
    assert a == b


def test_rng_of_prefers_context():
    seeded = PseudoRandom("X")
    ctx = ScoreContext(prng=seeded)
    assert rng_of(ctx) is seeded


def test_rng_of_raises_without_context():
    """W3: there is NO unseeded fallback any more. A hook that needs a roll and has
    no PseudoRandom must raise, never silently draw from the global module."""
    with pytest.raises(MissingPRNG):
        rng_of(None)
    with pytest.raises(MissingPRNG):
        rng_of(ScoreContext())
    card = Card(rank=5, suit="Spades", enhancement="Lucky")
    with pytest.raises(MissingPRNG):
        score_hand(scoring_cards=[card], all_cards=[card], hand_type="High Card",
                   jokers=[], planet_levels={"High Card": 1}, hands_left=3,
                   discards_left=3, dollars=4, ante=1, deck_remaining=40)


class TestShopDeterminism:
    """
    shop.py drew every stochastic choice from the global `random` module until
    2026-07-29 — roughly 14 sites, missed by the original H3 audit which only
    grepped scoring.py and jokers/. Shop contents were therefore not
    seed-determined, and in the MCTS repo two clones of one state generated
    different shops, which would corrupt tree search.
    """

    def _first_shop(self, seed):
        g = BalatroGame(seed=seed)
        g.step({"type": "play_blind"})
        g.chips_scored = g.current_blind.chips_target
        g.state = State.ROUND_EVAL
        g._end_round()
        return [(i.kind, i.key, i.price, i.edition) for i in g.current_shop]

    def test_same_seed_same_shop(self):
        assert self._first_shop(4242) == self._first_shop(4242)

    def test_different_seeds_give_different_shops(self):
        assert self._first_shop(1) != self._first_shop(2)

    def test_global_random_does_not_affect_shop(self):
        random.seed(0)
        a = self._first_shop(555)
        for _ in range(1000):
            random.random()
        random.seed(987654)
        b = self._first_shop(555)
        assert a == b

    def test_created_jokers_are_seed_determined(self):
        """W2: joker creation goes through generate.create_card on the run's keyed RNG."""
        from balatro_sim.game_keys import gen
        def draw(seed):
            st = gen.RunState(seed)
            st.showman = True
            return [gen.create_from_spec(st, "judgement").key for _ in range(5)]
        assert draw("ABCDEFGH") == draw("ABCDEFGH")
        assert draw("ABCDEFGH") != draw("HGFEDCBA")
