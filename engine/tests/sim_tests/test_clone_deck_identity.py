"""
test_clone_deck_identity.py — clone() must preserve the deck's object identity.

The 2026-07-29 fidelity audit introduced a persistent deck model:

    full_deck                       the permanent collection (source of truth)
    deck / hand / discard_pile      partitions holding REFERENCES to those cards

That aliasing is what makes a tarot applied to a card in hand persist for the
rest of the run. clone() previously copied each collection independently
(`[c.copy() for c in self.deck]`, `[c.copy() for c in self.hand]`), which would
mint several copies of the same card and sever the aliasing — so enhancements
applied after a clone would silently fail to persist, and card counts would drift.

These tests pin the invariant, because it fails silently: nothing raises, MCTS
just searches a subtly wrong game.
"""
import pytest

from balatro_sim.card import Card
from balatro_sim.consumables import apply_tarot
from balatro_sim.game import BalatroGame, State


def _mid_game(seed=42):
    g = BalatroGame(seed=seed)
    g.step({"type": "play_blind"})
    g.step({"type": "play", "cards": [0, 1]})
    g.step({"type": "discard", "cards": [0]})
    return g


class TestPartitionAliasing:
    def test_clone_partitions_alias_into_full_deck(self):
        c = _mid_game().clone()
        ids = {id(x) for x in c.full_deck}
        for name, coll in (("deck", c.deck), ("hand", c.hand),
                           ("discard_pile", c.discard_pile)):
            for card in coll:
                assert id(card) in ids, f"{name} holds a card outside full_deck"

    def test_clone_partitions_exactly_cover_full_deck(self):
        c = _mid_game().clone()
        total = len(c.deck) + len(c.hand) + len(c.discard_pile)
        assert total == len(c.full_deck)

    def test_clone_has_no_duplicate_card_objects(self):
        c = _mid_game().clone()
        all_ids = [id(x) for x in c.deck + c.hand + c.discard_pile]
        assert len(all_ids) == len(set(all_ids)), "a card object appears twice"

    def test_clone_does_not_share_objects_with_the_original(self):
        g = _mid_game()
        c = g.clone()
        orig_ids = {id(x) for x in g.full_deck}
        assert not any(id(x) in orig_ids for x in c.full_deck), \
            "clone shares Card objects with the original"


class TestMutationIsolation:
    def test_enhancing_a_clone_card_reaches_its_full_deck(self):
        c = _mid_game().clone()
        c.hand[0].enhancement = "Steel"
        assert sum(1 for x in c.full_deck if x.enhancement == "Steel") == 1

    def test_enhancing_a_clone_does_not_touch_the_original(self):
        g = _mid_game()
        c = g.clone()
        c.hand[0].enhancement = "Glass"
        assert all(x.enhancement != "Glass" for x in g.full_deck)

    def test_enhancing_the_original_does_not_touch_the_clone(self):
        g = _mid_game()
        c = g.clone()
        g.hand[0].enhancement = "Gold"
        assert all(x.enhancement != "Gold" for x in c.full_deck)

    def test_tarot_on_a_clone_persists_across_blinds(self):
        """The end-to-end property: clone, enhance, advance, still enhanced."""
        c = _mid_game().clone()
        apply_tarot(c, "c_justice", [0])          # -> Glass
        assert sum(1 for x in c.full_deck if x.enhancement == "Glass") == 1

        c.chips_scored = c.current_blind.chips_target
        c.state = State.ROUND_EVAL
        c._end_round()
        c.step({"type": "leave_shop"})
        c.step({"type": "play_blind"})
        dealt = c.deck + c.hand + c.discard_pile
        assert sum(1 for x in dealt if x.enhancement == "Glass") == 1

    def test_destroying_a_card_on_a_clone_is_isolated(self):
        g = _mid_game()
        c = g.clone()
        c.remove_card(c.full_deck[0])
        assert len(c.full_deck) == 51
        assert len(g.full_deck) == 52


class TestNewStateIsCloned:
    @pytest.mark.parametrize("attr", [
        "last_played_hand_type", "_played_this_ante", "_hand_type_counts",
        "_verdant_active", "_disabled_joker_idx", "_forced_card_id",
    ])
    def test_audit_added_state_survives_clone(self, attr):
        """Fields added by the audit must be copied, or search state is wrong.
        Card-id fields (The Pillar's played set, the Cerulean Bell's forced card) are
        compared by the CARDS they name: Card.copy() mints fresh ids, so clone() remaps
        them onto the clone's deck (Phase 2 W1 fix -- they used to keep the stale ids)."""
        g = _mid_game()
        c = g.clone()
        assert hasattr(c, attr), f"clone() dropped {attr}"
        if attr in ("_played_this_ante", "_forced_card_id"):
            def cards(game, ids):
                by_id = {x.id: (x.rank, x.suit) for x in game.full_deck}
                return sorted(by_id[i] for i in ids if i in by_id)
            ids_g = getattr(g, attr); ids_c = getattr(c, attr)
            if attr == "_forced_card_id":
                ids_g, ids_c = [ids_g], [ids_c]
            assert cards(c, ids_c) == cards(g, ids_g)
            assert len(cards(c, ids_c)) == len([i for i in ids_g if i >= 0])
        else:
            assert getattr(c, attr) == getattr(g, attr)

    def test_hand_type_counts_is_a_copy_not_a_reference(self):
        g = _mid_game()
        c = g.clone()
        c._hand_type_counts["Flush"] = 99
        assert g._hand_type_counts.get("Flush") != 99

    def test_played_this_ante_is_a_copy_not_a_reference(self):
        g = _mid_game()
        c = g.clone()
        c._played_this_ante.add(123456)
        assert 123456 not in g._played_this_ante


class TestLockstepStepping:
    def test_two_clones_step_identically_through_a_round(self):
        g = _mid_game()
        a, b = g.clone(), g.clone()
        for _ in range(12):
            if a.state != b.state:
                pytest.fail(f"state diverged: {a.state} vs {b.state}")
            if a.state == State.GAME_OVER:
                break
            if a.state == State.BLIND_SELECT:
                act = {"type": "play_blind"}
            elif a.state == State.SELECTING_HAND:
                act = {"type": "play", "cards": list(range(min(5, len(a.hand))))}
            elif a.state == State.SHOP:
                act = {"type": "leave_shop"}
            elif a.state == State.ROUND_EVAL:
                act = {"type": "noop"}
            else:
                act = {"type": "skip_booster"}
            a.step(act)
            b.step(act)
            assert a.chips_scored == b.chips_scored
            assert a.dollars == b.dollars
            assert len(a.full_deck) == len(b.full_deck)
