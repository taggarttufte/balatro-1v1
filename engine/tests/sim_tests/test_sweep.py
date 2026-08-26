"""
test_sweep.py — Phase 1 W5 (P1-sweep) regressions: play-time boss debuffs + Chicot,
face-down bosses (House / Wheel / Mark / Fish) through Card.face_down, permanent hand-size
deltas (Ouija / Ectoplasm), ante 0 (Hieroglyph at ante 1, boss pinned), Invisible Joker /
Perkeo copies, Negative consumable slot bookkeeping, JokerInstance.clone sort_id,
state_signature, env_mp revival through the round transition, env_v5 BoosterChoice picks.
See engine/SWEEP_NOTES.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from balatro_sim.card import Card
from balatro_sim.game import BalatroGame, State, UNMODELLED_BOSS_BLINDS, REGULAR_BOSS_BLINDS
from balatro_sim.constants import HAND_SIZE, blind_base_chips, get_blind_amount, BLIND_CHIPS
from balatro_sim.game_keys import gen as GEN, core as CORE
from balatro_sim.jokers.base import JokerInstance
from balatro_sim.shop import BoosterChoice, ShopItem
from balatro_sim.consumables import apply_voucher, apply_spectral
import balatro_sim.jokers  # noqa: F401

SEED = "7I4M53DL"
S, H, D, C = "Spades", "Hearts", "Diamonds", "Clubs"


def c(rank, suit=S, enh="None"):
    return Card(rank=rank, suit=suit, enhancement=enh)


def set_hand(g, cards):
    g.full_deck = [x for x in g.full_deck if x not in g.hand] + cards
    g.hand = list(cards)
    return cards


def force_boss(g, key):
    g.blind_idx = 2
    g._prepare_next_blind()
    g.current_blind.boss_key = key
    g.current_blind.is_boss = True
    g.current_blind.kind = "Boss"


def play_crafted(seed, jokers, boss, cards):
    """Craft ``cards`` into the hand AFTER play_blind (bypassing the deck debuff) and play
    them; return the chips scored."""
    g = BalatroGame(seed=seed)
    for k in jokers:
        g.debug_add_joker(k)
    force_boss(g, boss)
    g.step({"type": "play_blind"})
    cards = set_hand(g, cards)
    g.step({"type": "play", "cards": list(range(len(cards)))})
    return g.chips_scored, g


def to_shop(seed=SEED, before=None):
    g = BalatroGame(seed=seed)
    if before:
        before(g)
    g.step({"type": "play_blind"})
    g.debug_win_blind()
    g.step({"type": "advance"})
    assert g.state == State.SHOP
    g.dollars = 10 ** 6
    return g


# ─────────────────────────────────────────────────────────────────────────────
# A4. boss debuffs are re-evaluated at play time; Chicot disables the boss
# ─────────────────────────────────────────────────────────────────────────────

class TestPlayTimeDebuffs:
    FLUSH_S = [c(2, S), c(5, S), c(7, S), c(9, S), c(11, S)]

    def test_goad_debuffs_hand_crafted_spades_at_play(self):
        debuffed, g = play_crafted(SEED, [], "bl_goad", self.FLUSH_S)
        assert all(x.debuffed for x in g.discard_pile[-5:])
        normal, _ = play_crafted(SEED, [], "bl_water", self.FLUSH_S)
        assert debuffed < normal

    def test_chicot_disables_the_boss(self):
        with_chicot, g = play_crafted(SEED, ["j_chicot"], "bl_goad", self.FLUSH_S)
        assert g.current_blind.disabled and g.current_blind.boss_key == ""
        assert not any(x.debuffed for x in g.discard_pile[-5:])
        without, _ = play_crafted(SEED, [], "bl_goad", self.FLUSH_S)
        assert with_chicot > without

    def test_wild_is_debuffed_by_every_suit_boss(self):
        """Card:is_suit(suit, bypass_debuff) is true for a Wild card -> The Head debuffs it."""
        g = BalatroGame(seed=SEED)
        force_boss(g, "bl_head")
        g.step({"type": "play_blind"})
        w = c(9, S, "Wild"); st = c(9, S, "Stone"); sp = c(9, S)
        set_hand(g, [w, st, sp])
        g._refresh_card_debuffs(g.hand)
        assert w.debuffed and not st.debuffed and not sp.debuffed

    def test_smeared_extends_the_suit_boss(self):
        g = BalatroGame(seed=SEED)
        g.debug_add_joker("j_smeared")
        force_boss(g, "bl_head")          # Hearts -> Diamonds too
        g.step({"type": "play_blind"})
        assert all(x.debuffed for x in g.full_deck if x.suit in (H, D))
        assert not any(x.debuffed for x in g.full_deck if x.suit in (S, C))

    def test_pareidolia_makes_the_plant_debuff_everything(self):
        g = BalatroGame(seed=SEED)
        g.debug_add_joker("j_pareidolia")
        force_boss(g, "bl_plant")
        g.step({"type": "play_blind"})
        assert all(x.debuffed for x in g.full_deck)

    def test_non_boss_blind_debuffs_nothing(self):
        g = BalatroGame(seed=SEED)
        g.step({"type": "play_blind"})
        assert not any(x.debuffed for x in g.full_deck)

    def test_verdant_leaf_until_joker_sold(self):
        g = BalatroGame(seed=SEED)
        g.debug_add_joker("j_joker")
        force_boss(g, "bl_final_leaf")
        g.step({"type": "play_blind"})
        assert all(x.debuffed for x in g.hand)
        assert g._boss_debuffs_card(c(2, H))
        g._verdant_active = False
        assert not g._boss_debuffs_card(c(2, H))


# ─────────────────────────────────────────────────────────────────────────────
# B5. face-down bosses
# ─────────────────────────────────────────────────────────────────────────────

class TestFaceDownBosses:
    def test_all_regular_bosses_are_modelled(self):
        assert UNMODELLED_BOSS_BLINDS == []
        for k in ("bl_house", "bl_wheel", "bl_mark", "bl_fish"):
            assert k in REGULAR_BOSS_BLINDS

    def test_house_flips_the_initial_deal_only(self):
        g = BalatroGame(seed=SEED)
        force_boss(g, "bl_house")
        g.step({"type": "play_blind"})
        assert all(x.face_down for x in g.hand)
        g.step({"type": "discard", "cards": [0, 1]})
        assert sum(1 for x in g.hand if not x.face_down) == 2    # the redraw is face up
        assert all(x.face_down is False for x in g.discard_pile)  # revealed on leaving the hand

    def test_fish_flips_the_redraw_after_a_play_not_a_discard(self):
        g = BalatroGame(seed=SEED)
        force_boss(g, "bl_fish")
        g.step({"type": "play_blind"})
        assert not any(x.face_down for x in g.hand)
        g.step({"type": "discard", "cards": [0]})
        assert not any(x.face_down for x in g.hand)
        g.step({"type": "play", "cards": [0]})
        assert sum(1 for x in g.hand if x.face_down) == 1
        assert len(g.hand) == HAND_SIZE                          # no hand-size penalty (old model)

    def test_mark_flips_faces_and_pareidolia_everything(self):
        g = BalatroGame(seed=SEED)
        force_boss(g, "bl_mark")
        g.step({"type": "play_blind"})
        assert all(x.face_down == x.is_face_card for x in g.hand)
        g2 = BalatroGame(seed=SEED)
        g2.debug_add_joker("j_pareidolia")
        force_boss(g2, "bl_mark")
        g2.step({"type": "play_blind"})
        assert all(x.face_down for x in g2.hand)

    def test_wheel_rolls_the_wheel_key_per_card(self):
        g = BalatroGame(seed=SEED)
        force_boss(g, "bl_wheel")
        rng = CORE.PseudoRandom(g.seed_str)
        expect = [rng.pseudorandom("wheel") < 1 / 7 for _ in range(HAND_SIZE)]
        g.step({"type": "play_blind"})
        assert [x.face_down for x in g.hand] == expect
        assert g.run_state.rng.get_key_state("wheel") is not None

    def test_chicot_keeps_everything_face_up(self):
        g = BalatroGame(seed=SEED)
        g.debug_add_joker("j_chicot")
        force_boss(g, "bl_house")
        g.step({"type": "play_blind"})
        assert not any(x.face_down for x in g.hand)

    def test_face_down_cleared_at_next_blind(self):
        g = BalatroGame(seed=SEED)
        force_boss(g, "bl_house")
        g.step({"type": "play_blind"})
        g.debug_win_blind(); g.step({"type": "advance"})
        g.step({"type": "leave_shop"}); g.step({"type": "play_blind"})
        assert not any(x.face_down for x in g.full_deck)

    def test_env_v7_hides_face_down_cards(self):
        from balatro_sim.env_v7 import BalatroV7Env, GAME_SCALARS, CARD_FEATURES, N_BOSS_TYPES
        assert N_BOSS_TYPES == 28
        env = BalatroV7Env(seed=42)
        env.reset()
        g = env.game
        g.step({"type": "play_blind"}) if g.state == State.BLIND_SELECT else None
        for x in g.hand:
            x.face_down = True
        obs = env._encode_obs()
        for slot in range(len(g.hand)):
            block = obs[GAME_SCALARS + slot * CARD_FEATURES: GAME_SCALARS + (slot + 1) * CARD_FEATURES]
            assert block[25] == 1.0 and block.sum() == 1.0, slot
        for x in g.hand:
            x.face_down = False
        obs = env._encode_obs()
        block = obs[GAME_SCALARS: GAME_SCALARS + CARD_FEATURES]
        assert block.sum() > 1.0

    def test_card_copy_carries_face_down(self):
        x = c(5, H); x.face_down = True
        assert x.copy().face_down is True


# ─────────────────────────────────────────────────────────────────────────────
# B7. permanent hand size (Ouija / Ectoplasm)
# ─────────────────────────────────────────────────────────────────────────────

class TestPermanentHandSize:
    def _next_blind(self, g):
        g.debug_win_blind(); g.step({"type": "advance"})
        g.step({"type": "leave_shop"}); g.step({"type": "play_blind"})

    def test_ouija_persists_and_stacks(self):
        g = BalatroGame(seed=SEED)
        g.step({"type": "play_blind"})
        g.consumable_hand = ["c_ouija", "c_ouija"]
        g.step({"type": "use_consumable", "consumable_idx": 0, "target_cards": []})
        assert g.hand_size == HAND_SIZE - 1 and g.hand_size_mod == -1
        g.step({"type": "use_consumable", "consumable_idx": 0, "target_cards": []})
        assert g.hand_size == HAND_SIZE - 2 and g.hand_size_mod == -2
        self._next_blind(g)
        assert g.hand_size == HAND_SIZE - 2 and len(g.hand) == HAND_SIZE - 2

    def test_ectoplasm_persists(self):
        g = BalatroGame(seed=SEED)
        g.debug_add_joker("j_joker")
        g.step({"type": "play_blind"})
        g.consumable_hand = ["c_ectoplasm"]
        g.step({"type": "use_consumable", "consumable_idx": 0, "target_cards": []})
        assert g.jokers[0].edition == "Negative" and g.hand_size_mod == -1
        self._next_blind(g)
        assert g.hand_size == HAND_SIZE - 1

    def test_clone_keeps_the_modifier(self):
        g = BalatroGame(seed=SEED)
        g.hand_size_mod = -1
        assert g.clone().hand_size_mod == -1


# ─────────────────────────────────────────────────────────────────────────────
# B8. Hieroglyph at ante 1 -> ante 0
# ─────────────────────────────────────────────────────────────────────────────

class TestAnteZero:
    def test_blind_table_has_ante_0(self):
        assert BLIND_CHIPS[0] == (100, 150, 200)
        assert get_blind_amount(0) == 100 and get_blind_amount(-1) == 100
        assert blind_base_chips(0, 2) == 200 and blind_base_chips(1, 0) == 300
        assert get_blind_amount(9) == 110000           # endless formula, first step

    def test_hieroglyph_at_ante_1_goes_to_ante_0_and_keeps_the_boss(self):
        g = to_shop()
        boss = g.boss_blind
        g.current_shop.insert(0, ShopItem("voucher", "v_hieroglyph", "v_hieroglyph", 10))
        g.step({"type": "buy", "item_idx": 0})
        assert g.ante == 0 and g.run_state.ante == 0 and g.base_hands == 3
        g.step({"type": "leave_shop"})
        assert g.current_blind.kind == "Big" and g.current_blind.chips_target == 150
        g.step({"type": "play_blind"}); g.debug_win_blind(); g.step({"type": "advance"})
        g.dollars = 10 ** 6
        g.step({"type": "leave_shop"})
        assert g.current_blind.kind == "Boss" and g.boss_blind == boss      # not redrawn
        assert g.current_blind.chips_target == 200
        g.step({"type": "play_blind"}); g.debug_win_blind(); g.step({"type": "advance"})
        assert g.ante == 1 and g.run_state.ante == 1

    def test_petroglyph_pins_the_boss_too(self):
        g = to_shop()
        g.ante = 3; g.run_state.ante = 3
        g._boss_blind_ante = 3
        boss = g.boss_blind
        apply_voucher(g, "v_petroglyph")
        assert g.ante == 2 and g._boss_blind_ante == 2
        g.blind_idx = 2
        g._prepare_next_blind()
        assert g.current_blind.boss_key == boss


# ─────────────────────────────────────────────────────────────────────────────
# B6. Invisible Joker / Perkeo, Negative consumables
# ─────────────────────────────────────────────────────────────────────────────

class TestCopies:
    def test_invisible_copy_acquires_and_strips_negative(self):
        def before(g):
            g.debug_add_joker("j_invisible")
            g.jokers[0].state["rounds"] = 2
            g.debug_add_joker("j_joker", edition="Negative")
        g = to_shop(before=before)
        assert g.run_state.owned_jokers.count("j_joker") == 1
        g.step({"type": "sell_joker", "joker_idx": 0})
        keys = [j.key for j in g.jokers]
        assert keys == ["j_joker", "j_joker"]
        assert [j.edition for j in g.jokers] == ["Negative", "None"]     # copy_card strip_edition
        assert g.run_state.owned_jokers.count("j_joker") == 2
        assert "j_joker" in g.run_state.used_jokers
        assert g.run_state.rng.get_key_state("invisible") is not None

    def test_invisible_needs_two_rounds(self):
        def before(g):
            g.debug_add_joker("j_invisible")
            g.debug_add_joker("j_joker")
        g = to_shop(before=before)
        g.step({"type": "sell_joker", "joker_idx": 0})
        assert [j.key for j in g.jokers] == ["j_joker"]

    def test_invisible_copy_may_take_its_own_slot(self):
        """#G.jokers.cards <= card_limit with the Invisible still on the board."""
        def before(g):
            g.debug_add_joker("j_invisible")
            g.jokers[0].state["rounds"] = 2
            for _ in range(4):
                g.debug_add_joker("j_joker")
        g = to_shop(before=before)
        assert len(g.jokers) == g.joker_slots == 5
        g.step({"type": "sell_joker", "joker_idx": 0})
        assert len(g.jokers) == 5 and all(j.key == "j_joker" for j in g.jokers)

    def test_perkeo_negative_copy_takes_no_slot_and_frees_it_when_used(self):
        def before(g):
            g.debug_add_joker("j_perkeo")
        g = to_shop(before=before)
        g.consumable_hand = ["c_mercury", "c_venus"]
        g.run_state.owned_consumables = list(g.consumable_hand)
        assert len(g.consumable_hand) == g.consumable_slots == 2
        g.step({"type": "leave_shop"})
        assert len(g.consumable_hand) == 3 and g.consumable_slots == 3
        copy = g.consumable_hand[2]
        assert copy in ("c_mercury", "c_venus") and g.negative_consumables == {copy: 1}
        assert g.run_state.rng.get_key_state("perkeo") is not None
        # using the (only) copy of that key drops the extra slot again
        idx = g.consumable_hand.index(copy)
        g.step({"type": "play_blind"})
        g.step({"type": "use_consumable", "consumable_idx": idx, "target_cards": []})
        assert len(g.consumable_hand) == 2 and g.consumable_slots == 2 and g.negative_consumables == {}

    def test_negative_pack_consumable_bookkeeping(self):
        g = BalatroGame(seed=SEED)
        g.consumable_hand = ["c_mercury", "c_venus"]
        ch = BoosterChoice("c_mars", "Planet", "Negative")
        assert g._can_grant_choice(ch) and g._grant_choice(ch)
        assert g.consumable_slots == 3 and g.negative_consumables == {"c_mars": 1}
        g.step({"type": "play_blind"})
        g.step({"type": "use_consumable", "consumable_idx": 2, "target_cards": []})
        assert g.consumable_slots == 2 and g.negative_consumables == {}

    def test_clone_copies_sort_id(self):
        j = JokerInstance("j_joker")
        assert j.clone().sort_id == j.sort_id
        g = BalatroGame(seed=SEED)
        g.debug_add_joker("j_joker"); g.debug_add_joker("j_jolly")
        assert [x.sort_id for x in g.clone().jokers] == [x.sort_id for x in g.jokers]


# ─────────────────────────────────────────────────────────────────────────────
# B11. state_signature
# ─────────────────────────────────────────────────────────────────────────────

class TestStateSignature:
    def test_hashable_deterministic_and_sensitive(self):
        a, b = BalatroGame(seed="ALEEB"), BalatroGame(seed="ALEEB")
        sa = a.state_signature()
        assert hash(sa) == hash(b.state_signature()) and sa == b.state_signature()
        assert sa == a.clone().state_signature()
        assert sa != BalatroGame(seed="7I4M53DL").state_signature()
        a.step({"type": "play_blind"})
        assert a.state_signature() != sa and b.state_signature() == sa
        # an effect roll alone changes the signature (rng digest)
        a.run_state.rng.pseudorandom("lucky_mult")
        s1 = a.state_signature()
        a.run_state.rng.pseudorandom("lucky_mult")
        assert a.state_signature() != s1

    def test_clone_divergence_leaves_original(self):
        g = BalatroGame(seed="ALEEB")
        sig = g.state_signature()
        k = g.clone()
        k.step({"type": "play_blind"}); k.step({"type": "play", "cards": [0, 1]})
        assert g.state_signature() == sig


# ─────────────────────────────────────────────────────────────────────────────
# B10. env_mp revival / env_v5 BoosterChoice
# ─────────────────────────────────────────────────────────────────────────────

def _ground_truth(seed):
    p = Path(__file__).resolve().parents[3] / "oracle" / "ground_truth" / f"{seed}.json"
    if not p.exists():
        pytest.skip("ground truth corpus not present")
    return json.loads(p.read_text())


class TestEnvRevival:
    def test_lost_boss_next_shelf_matches_ground_truth(self):
        """(Phase 2 W1: the env-side "revive" hack is gone -- under ``ruleset="mlb"`` a lost
        blind proceeds natively.)  A player who LOSES the ante-1 boss goes through the normal
        round transition, so the ante-2 shelf is the ground truth's (no double generation).
        The voucher is not compared: MLB vouchers come from the run-global 'Voucher0' path."""
        gt = _ground_truth(SEED)
        g = BalatroGame(seed=SEED, ruleset="mlb")
        # walk to the boss without buying anything
        for _ in range(2):
            if g.state == State.BLIND_SELECT:
                g.step({"type": "play_blind"})
            g.debug_win_blind(); g.step({"type": "advance"})
            g.step({"type": "leave_shop"})
        assert g.current_blind.kind == "Boss"
        g.step({"type": "play_blind"})
        g.hands_left = 1
        g.step({"type": "play", "cards": [0]})
        assert g.state == State.ROUND_EVAL and g.lives == 3
        g.step({"type": "advance"})
        assert g.state == State.SHOP and g.ante == 2
        expect = [it["key"] for it in gt["antes"]["2"]["shop_queue"][:2]]
        got = [it.key for it in g.current_shop if it.kind not in ("voucher", "booster")]
        assert got == expect
        assert g.boss_blind == gt["antes"]["2"]["boss"]["key"]

    def test_env_v5_pack_pick_acquires_and_releases(self):
        from balatro_sim.env_v5 import BalatroSimEnvV5
        env = BalatroSimEnvV5(seed=42)
        env.reset()
        g = env.game
        mk = lambda k: GEN.CardGen(key=k, set="Joker", type_requested="Joker", area="pack")  # noqa: E731
        choices = [BoosterChoice("j_joker", "Joker", gen=mk("j_joker")),
                   BoosterChoice("j_jolly", "Joker", gen=mk("j_jolly"))]
        g.run_state.used_jokers.update({"j_joker", "j_jolly"})
        g.booster_choices = list(choices)
        g.booster_picks_remaining = 1
        env._enter_pack_open(g)
        env._step_pack_open(0)
        assert [j.key for j in g.jokers] == ["j_joker"]
        assert "j_joker" in g.run_state.owned_jokers and "j_joker" in g.run_state.used_jokers
        assert "j_jolly" not in g.run_state.used_jokers       # released on exit


# ─────────────────────────────────────────────────────────────────────────────
# Blind-start order: setting_blind hooks -> 'nr' shuffle -> draw -> first_hand_drawn
# ─────────────────────────────────────────────────────────────────────────────

class TestBlindStartOrder:
    def test_marble_stone_is_in_the_first_shuffle(self):
        """Marble's Stone card enters G.deck in `setting_blind`, BEFORE `nr<ante>` — so the
        deal is the shuffle of the 53-card deck (not 52 + a card stuck at the bottom)."""
        g = BalatroGame(seed=SEED)
        g.debug_add_joker("j_marble")
        g.step({"type": "play_blind"})
        assert len(g.full_deck) == 53
        st = GEN.RunState(SEED); GEN.start_run(st, "b_red")
        st.new_round()
        # replay: marble front first (marb_fr), then the nr1 shuffle over the 53 sorted cards
        front = GEN.marble_joker(st)
        deck = sorted(g.full_deck, key=lambda x: x.id)
        order = GEN.shuffle_deck(st, list(deck), GEN.Keys.new_round_shuffle(1))
        expect = [(x.rank, x.suit, x.enhancement) for x in order[-HAND_SIZE:]][::-1]
        assert [(x.rank, x.suit, x.enhancement) for x in g.hand] == expect

    def test_certificate_joins_the_hand_after_the_draw(self):
        g = BalatroGame(seed=SEED)
        g.debug_add_joker("j_certificate")
        g.step({"type": "play_blind"})
        assert len(g.hand) == HAND_SIZE + 1
        assert sum(1 for x in g.hand if x.seal != "None") >= 1

    def test_negative_created_joker_takes_no_slot(self):
        from balatro_sim.jokers.base import add_joker
        g = BalatroGame(seed=SEED)
        for _ in range(5):
            g.debug_add_joker("j_joker")
        assert not add_joker(g, JokerInstance("j_jolly"))
        assert add_joker(g, JokerInstance("j_jolly", "Negative"))
        assert len(g.jokers) == 6 and g.joker_slots == 6


class TestRetcon:
    def test_retcon_rerolls_without_limit_and_the_boss_changes(self):
        g = BalatroGame(seed=SEED)
        g.dollars = 100
        assert {"type": "reroll_boss"} not in g.legal_actions()
        apply_voucher(g, "v_retcon")
        seen = {g.boss_blind}
        for i in range(3):
            assert {"type": "reroll_boss"} in g.legal_actions()
            g.step({"type": "reroll_boss"})
            seen.add(g.boss_blind)
            assert g.dollars == 100 - 10 * (i + 1)
        assert len(seen) > 1
        g.blind_idx = 2; g._prepare_next_blind()
        assert g.current_blind.boss_key == g.boss_blind
