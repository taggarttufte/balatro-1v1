"""
test_clone_stochastic.py - Verify clones produce identical stochastic event
sequences from the same state.

This catches the bug where shop generation, joker effects, and consumables
fall back to the module-global `random` module instead of `game.rng`. Two
clones from the same state, stepped through identical actions, must produce
byte-identical results.
"""
from balatro_sim.game import BalatroGame, State
from balatro_sim.jokers.base import JokerInstance
from balatro_sim.shop import ShopItem


def _populate_with_jokers_and_shop(seed: int = 42) -> BalatroGame:
    """Fixture: a state where many random sources will fire on the next steps."""
    g = BalatroGame(seed=seed)
    g.reset()
    g.step({"type": "play_blind"})
    g.dollars = 50
    g.jokers = [
        JokerInstance("j_misprint"),       # random chips per hand
        JokerInstance("j_business"),  # random face-card payout
        JokerInstance("j_8_ball"),         # random tarot creation on 8s
        JokerInstance("j_bloodstone"),     # random heart x1.5 mult
    ]
    return g


def test_two_clones_produce_identical_shop_generation():
    """Clones at SHOP-generation step must produce identical shops."""
    from balatro_sim.shop import generate_shop
    g = _populate_with_jokers_and_shop(seed=42)
    a = g.clone()
    b = g.clone()
    shop_a = generate_shop(a)
    shop_b = generate_shop(b)
    # Same kinds, same keys, same prices, same editions
    assert [(s.kind, s.key, s.price, s.edition) for s in shop_a] == \
           [(s.kind, s.key, s.price, s.edition) for s in shop_b]


def test_two_clones_produce_identical_play_outcomes():
    """Clones stepping the same play action must produce identical chip scores."""
    g = _populate_with_jokers_and_shop(seed=42)
    a = g.clone()
    b = g.clone()
    if a.state != State.SELECTING_HAND or len(a.hand) < 3:
        return  # skip if state isn't ripe
    action = {"type": "play", "cards": [0, 1, 2]}
    a.step(action)
    b.step(action)
    # Same scoring (Misprint random chips), same dollars (Business Card random $)
    assert a.chips_scored == b.chips_scored
    assert a.dollars == b.dollars


def test_clones_diverge_after_independent_rng_advance():
    """If the original advances its RNG, the clone's next draw must NOT match."""
    g = _populate_with_jokers_and_shop(seed=42)
    c = g.clone()
    # advance original's keyed stream
    _ = g.run_state.rng.pseudorandom("glass")
    # next draws should diverge
    g_val = g.run_state.rng.pseudorandom("glass")
    c_val = c.run_state.rng.pseudorandom("glass")  # this is the same as g's first draw
    assert g_val != c_val


def test_long_sequence_clones_remain_identical():
    """Walk two clones through an identical 10-step random sequence."""
    g = _populate_with_jokers_and_shop(seed=42)
    a = g.clone()
    b = g.clone()
    seq_a, seq_b = [], []
    for _ in range(10):
        if a.state == State.GAME_OVER or b.state == State.GAME_OVER:
            break
        # Use legal_actions() and pick the first one (deterministic across clones)
        la = a.legal_actions()
        lb = b.legal_actions()
        assert la == lb, "Legal actions diverged between clones"
        if not la:
            break
        action = la[0]
        a.step(action)
        b.step(action)
        seq_a.append((a.state, a.chips_scored, a.dollars, len(a.hand)))
        seq_b.append((b.state, b.chips_scored, b.dollars, len(b.hand)))
    assert seq_a == seq_b


def test_legal_actions_smoke():
    """Sanity: legal_actions() returns plausible action lists for each state."""
    g = BalatroGame(seed=1)
    g.reset()
    # BLIND_SELECT: play or skip
    la = g.legal_actions()
    assert {a["type"] for a in la} == {"play_blind", "skip_blind"}
    # Step into SELECTING_HAND
    g.step({"type": "play_blind"})
    la = g.legal_actions()
    assert g.state == State.SELECTING_HAND
    types = {a["type"] for a in la}
    assert "play" in types
    if g.discards_left > 0:
        assert "discard" in types
    # Should be non-empty and bounded
    assert 1 <= len(la) <= 1024  # 218 plays + 218 discards + few consumables max
