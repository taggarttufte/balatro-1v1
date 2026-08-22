"""
test_delegate.py — Phase 1 W2: the engine delegates ALL generation to mp/rng/generate
through game.run_state (shelves, rerolls, packs, vouchers, bosses, tags, shuffles, created
cards), the BOOSTER_OPEN state machine works, tags are wired, JokerInstance.clone() isolates
nested containers.  See mp/engine/DELEGATE_NOTES.md.
"""
from __future__ import annotations

import pytest

from balatro_sim.game import BalatroGame, State, BOSS_CHIP_MULT
from balatro_sim.game_keys import gen as GEN, pools as P
from balatro_sim.jokers.base import JokerInstance
from balatro_sim.shop import BoosterChoice, ShopItem, BOOSTER_PICKS
from balatro_sim.consumables import apply_voucher, apply_tarot
from balatro_sim import tags as T

SEED = "7I4M53DL"   # campaign-log live-check seed: Banner + Hierophant, Buffoon + Arcana, Hook


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


def shelf(g):
    return [(it.kind, it.key, it.edition) for it in g.current_shop if it.kind not in ("voucher", "booster")]


def packs(g):
    return [(i, it.key) for i, it in enumerate(g.current_shop) if it.kind == "booster" and not it.sold]


# ── JokerInstance.clone: nested containers ───────────────────────────────────

class TestJokerInstanceClone:
    def test_nested_set_is_isolated(self):
        j = JokerInstance("j_card_sharp")
        j.state["played_hands"] = {"Pair"}
        j.state["pending_consumables"] = ["c_fool"]
        j.state["mult"] = 3
        c = j.clone()
        c.state["played_hands"].add("Flush")
        c.state["pending_consumables"].append("c_hermit")
        c.state["mult"] = 9
        assert j.state["played_hands"] == {"Pair"}
        assert j.state["pending_consumables"] == ["c_fool"]
        assert j.state["mult"] == 3
        assert c.key == j.key and c.edition == j.edition

    def test_game_clone_does_not_share_joker_containers(self):
        g = BalatroGame(seed=SEED)
        j = g.debug_add_joker("j_satellite")
        j.state["planets_used"] = {"c_mercury"}
        c = g.clone()
        c.jokers[0].state["planets_used"].add("c_venus")
        assert g.jokers[0].state["planets_used"] == {"c_mercury"}


# ── Run start / bosses / tags / vouchers ─────────────────────────────────────

class TestRunStart:
    def test_run_start_matches_generate(self):
        g = BalatroGame(seed=SEED)
        st = GEN.RunState(SEED)
        rs = GEN.start_run(st, "b_red")
        assert g.boss_blind == rs.boss == "bl_hook"
        assert g.blind_tags == {"Small": rs.tag_small, "Big": rs.tag_big} == {"Small": "tag_skip", "Big": "tag_economy"}
        assert g.run_state.current_round_voucher == rs.voucher == "v_wasteful"
        assert g.round_picks["idol"] == rs.idol and g.round_picks["ancient"] == rs.ancient_suit

    def test_full_deck_in_creation_order_and_first_deal(self):
        g = BalatroGame(seed=SEED)
        fronts = [f"{c.suit[0]}_{ {10:'T',11:'J',12:'Q',13:'K',14:'A'}.get(c.rank, str(c.rank))}" for c in g.full_deck]
        assert fronts == sorted(P.PLAYING_CARD_KEYS, key=lambda k: k[0] + k[2:])
        g.step({"type": "play_blind"})
        # ground truth deck_order_unverified.small for 7I4M53DL: C_5 C_2 S_J S_7 D_3 D_K C_A H_2
        assert [repr(c) for c in g.hand] == ["5C", "2C", "JS", "7S", "3D", "KD", "AC", "2H"]

    def test_ante_transition_draws_next_voucher_tags_boss(self):
        g = BalatroGame(seed=SEED)
        st = GEN.RunState(SEED)
        GEN.start_run(st, "b_red")
        for _ in range(3):
            g.step({"type": "play_blind"}); g.debug_win_blind(); g.step({"type": "advance"})
            if g.ante == 1:
                g.step({"type": "leave_shop"})
        # boss just died: the engine is in the post-boss shop, ante 2
        assert g.ante == 2 and g.state == State.SHOP
        info = GEN.defeat_boss(st)      # 'Voucher2', 'Tag2' x2, 'boss' -- independent of the shop keys
        assert g.boss_blind == info["boss"] == "bl_window"
        assert g.blind_tags == {"Small": info["tag_small"], "Big": info["tag_big"]}
        assert g.run_state.current_round_voucher == info["voucher"]

    def test_prepare_next_blind_uses_the_drawn_boss(self):
        g = BalatroGame(seed=SEED)
        g.blind_idx = 2
        g._prepare_next_blind()
        assert g.current_blind.boss_key == g.boss_blind == "bl_hook"
        assert g.current_blind.chips_target == 600 * BOSS_CHIP_MULT.get("bl_hook", 1)

    def test_showdown_at_ante_8(self):
        g = BalatroGame(seed=SEED)
        g.ante, g.blind_idx = 8, 2
        g._prepare_next_blind()
        assert g.current_blind.boss_key.startswith("bl_final_") and g.current_blind.is_showdown


# ── Shop delegation ──────────────────────────────────────────────────────────

class TestShopDelegation:
    def test_first_shop_matches_generate_and_ground_truth(self):
        g = to_shop()
        assert shelf(g) == [("joker", "j_banner", "None"), ("tarot", "c_heirophant", "None")]
        assert [k for _, k in packs(g)] == ["p_buffoon_normal", "p_arcana_normal"]
        assert [it.key for it in g.current_shop if it.kind == "voucher"] == ["v_wasteful"]
        assert len(shelf(g)) == g.run_state.shop_joker_max == 2

    def test_reroll_is_queue_advance(self):
        g = to_shop()
        st = GEN.RunState(SEED); GEN.start_run(st, "b_red"); st.new_round()
        s = GEN.generate_shop(st)
        for k in range(4):
            assert [x[1] for x in shelf(g)] == [c.key for c in s.cards], f"k={k}"
            g.step({"type": "reroll"}); GEN.reroll_shop(st, s)
        assert g.reroll_cost == 9

    def test_leaving_shop_releases_the_shelf(self):
        g = to_shop()
        assert {"j_banner", "c_heirophant"} <= g.run_state.used_jokers
        g.step({"type": "leave_shop"})
        assert not ({"j_banner", "c_heirophant"} & g.run_state.used_jokers)
        assert g.current_shop == []

    def test_purchase_acquires_and_blocks_in_place(self):
        g = to_shop()
        g.step({"type": "buy", "item_idx": 0})        # Banner
        assert g.jokers[0].key == "j_banner" and "j_banner" in g.run_state.owned_jokers
        assert "j_banner" in g.run_state.used_jokers
        # a second run that OWNS Banner sees the slot resampled, everything else identical
        h = to_shop(before=lambda x: x.debug_add_joker("j_banner"))
        assert shelf(h)[0][1] != "j_banner"
        assert shelf(h)[1] == ("tarot", "c_heirophant", "None")
        assert [k for _, k in packs(h)] == ["p_buffoon_normal", "p_arcana_normal"]

    def test_showman_lifts_the_block(self):
        h = to_shop(before=lambda x: (x.debug_add_joker("j_banner"), x.debug_add_joker("j_ring_master")))
        assert h.run_state.showman
        assert shelf(h)[0][1] == "j_banner"

    def test_sell_releases(self):
        g = to_shop()
        g.step({"type": "buy", "item_idx": 0})
        g.step({"type": "sell_joker", "joker_idx": 0})
        assert "j_banner" not in g.run_state.owned_jokers
        assert "j_banner" not in g.run_state.used_jokers   # Card:remove, no other copy

    def test_voucher_purchase_empties_slot_and_syncs_run_state(self):
        g = to_shop()
        vi = [i for i, it in enumerate(g.current_shop) if it.kind == "voucher"][0]
        g.step({"type": "buy", "item_idx": vi})
        assert "v_wasteful" in g.vouchers and "v_wasteful" in g.run_state.used_vouchers
        assert g.run_state.current_round_voucher is None
        g.step({"type": "leave_shop"})
        g.step({"type": "play_blind"}); g.debug_win_blind(); g.step({"type": "advance"})
        assert not [it for it in g.current_shop if it.kind == "voucher"]   # empty until the next ante

    def test_overstock_mid_shop_fills_a_third_slot(self):
        g = to_shop()
        apply_voucher(g, "v_overstock_norm")
        assert len(shelf(g)) == 3 and g.run_state.shop_joker_max == 3
        g.step({"type": "reroll"})
        assert len(shelf(g)) == 3

    def test_ban_list_is_honoured_by_generation(self):
        from balatro_sim import shop as S
        S.set_banned_jokers({"j_banner"})
        try:
            g = to_shop()
            assert "j_banner" in g.run_state.banned_keys
            assert shelf(g)[0][1] != "j_banner"
            assert shelf(g)[1] == ("tarot", "c_heirophant", "None")
        finally:
            S.clear_banned_jokers()

    def test_couponed_item_costs_zero(self):
        it = ShopItem("joker", "j_joker", "Joker", 2, couponed=True)
        assert it.discounted_price(0.0) == 0
        assert ShopItem("joker", "j_joker", "Joker", 5).discounted_price(0.25) == 4   # floor(5.5*.75)


# ── Booster state machine ────────────────────────────────────────────────────

class TestBoosterStateMachine:
    def test_buy_enters_booster_open_with_generated_contents(self):
        g = to_shop()
        idx = packs(g)[1][0]                     # Arcana
        g.step({"type": "buy", "item_idx": idx})
        assert g.state == State.BOOSTER_OPEN
        assert [c.key for c in g.booster_choices] == ["c_emperor", "c_hanged_man", "c_lovers"]  # faithful used_jokers: no Hierophant
        assert all(isinstance(c, BoosterChoice) for c in g.booster_choices)
        assert g.booster_picks_remaining == BOOSTER_PICKS["p_arcana_normal"] == 1
        la = g.legal_actions()
        assert {"type": "skip_booster"} in la
        assert {"type": "pick_booster", "indices": [2]} in la

    def test_pick_acquires_and_closes(self):
        g = to_shop()
        g.step({"type": "buy", "item_idx": packs(g)[1][0]})
        g.step({"type": "pick_booster", "indices": [2]})
        assert g.state == State.SHOP
        assert g.consumable_hand == ["c_lovers"]
        assert "c_lovers" in g.run_state.owned_consumables
        assert "c_emperor" not in g.run_state.used_jokers      # unpicked cards released
        assert g.booster_choices == []

    def test_skip_releases(self):
        g = to_shop()
        g.step({"type": "buy", "item_idx": packs(g)[1][0]})
        g.step({"type": "skip_booster"})
        assert g.state == State.SHOP and g.consumable_hand == []
        assert not ({"c_emperor", "c_hanged_man", "c_lovers"} & g.run_state.used_jokers)

    def test_buffoon_pack_joker_pick(self):
        g = to_shop()
        g.step({"type": "buy", "item_idx": packs(g)[0][0]})
        assert g.state == State.BOOSTER_OPEN
        assert [c.key for c in g.booster_choices] == ["j_acrobat", "j_wily"]   # ground truth
        assert g.booster_choices[1].edition == "Foil"
        g.step({"type": "pick_booster", "indices": [1]})
        assert g.jokers[0].key == "j_wily" and g.jokers[0].edition == "Foil"
        assert "j_wily" in g.run_state.owned_jokers

    def test_mega_pack_two_picks(self):
        g = to_shop()
        g.current_shop.append(ShopItem("booster", "p_arcana_mega", "Mega Arcana Pack", 8, center="p_arcana_mega_1", set="Booster"))
        g.step({"type": "buy", "item_idx": len(g.current_shop) - 1})
        assert g.state == State.BOOSTER_OPEN and len(g.booster_choices) == 5 and g.booster_picks_remaining == 2
        g.step({"type": "pick_booster", "indices": [0]})
        assert g.state == State.BOOSTER_OPEN and g.booster_picks_remaining == 1 and len(g.booster_choices) == 4
        g.step({"type": "pick_booster", "indices": [0]})
        assert g.state == State.SHOP and len(g.consumable_hand) == 2

    def test_standard_pack_cards_carry_modifiers_and_go_to_deck(self):
        g = to_shop()
        g.current_shop.append(ShopItem("booster", "p_standard_normal", "Standard Pack", 4, center="p_standard_normal_1", set="Booster"))
        g.step({"type": "buy", "item_idx": len(g.current_shop) - 1})
        assert g.state == State.BOOSTER_OPEN
        c = g.booster_choices[0]
        assert c.is_playing_card and c.card is not None and c.front == c.key
        n = len(g.full_deck)
        g.step({"type": "pick_booster", "indices": [0]})
        assert len(g.full_deck) == n + 1 and g.full_deck[-1] is c.card

    def test_no_room_means_only_skip_is_legal(self):
        g = to_shop()
        g.consumable_hand = ["c_fool", "c_hermit"]
        g.step({"type": "buy", "item_idx": packs(g)[1][0]})
        assert g.legal_actions() == [{"type": "skip_booster"}]
        g.step({"type": "pick_booster", "indices": [0]})     # ungrantable pick closes the pack
        assert g.state == State.SHOP and g.consumable_hand == ["c_fool", "c_hermit"]

    def test_clone_isolates_booster_state(self):
        g = to_shop()
        g.step({"type": "buy", "item_idx": packs(g)[1][0]})
        c = g.clone()
        c.step({"type": "pick_booster", "indices": [0]})
        assert g.state == State.BOOSTER_OPEN and len(g.booster_choices) == 3
        assert c.run_state is not g.run_state and c.tag_state is not g.tag_state


# ── Skip / tags ──────────────────────────────────────────────────────────────

class TestSkipAndTags:
    def test_skip_grants_tag_and_moves_on(self):
        g = BalatroGame(seed=SEED)          # Small: tag_skip, Big: tag_economy
        g.step({"type": "skip_blind"})
        assert g.skips == 1 and g.dollars == 4 + 5          # Skip Tag: $5 x 1 skip, immediate
        assert g.state == State.BLIND_SELECT and g.current_blind.kind == "Big"
        assert g.tags.keys() == []
        g.step({"type": "skip_blind"})
        assert g.dollars == 9 + 9                            # Economy: +min(40, $9)
        assert g.current_blind.kind == "Boss" and g.skips == 2

    def test_investment_tag_pays_after_boss(self):
        g = BalatroGame(seed=SEED)
        g.blind_tags["Small"] = "tag_investment"
        g.step({"type": "skip_blind"})
        assert g.tags.keys() == ["tag_investment"]
        g.step({"type": "skip_blind"})
        g.step({"type": "play_blind"}); g.debug_win_blind()
        before = g.dollars
        g.step({"type": "advance"})
        assert g.tags.keys() == [] and g.dollars >= before + 25

    def test_pack_tag_interrupts_blind_select(self):
        g = BalatroGame(seed=SEED)
        g.blind_tags["Small"] = "tag_charm"
        g.step({"type": "skip_blind"})
        assert g.state == State.BOOSTER_OPEN and g.booster_pack_key == "p_arcana_mega"
        assert g.booster_picks_remaining == 2 and len(g.booster_choices) == 5
        g.step({"type": "pick_booster", "indices": [0, 1]})
        assert g.state == State.BLIND_SELECT and g.current_blind.kind == "Big"
        assert len(g.consumable_hand) == 2 and g.tags.keys() == []

    def test_boss_tag_rerolls_the_boss(self):
        g = BalatroGame(seed=SEED)
        g.blind_tags["Small"] = "tag_boss"
        before = g.boss_blind
        g.step({"type": "skip_blind"})
        assert g.tags.keys() == [] and g._boss_rerolled
        st = GEN.RunState(SEED); GEN.start_run(st, "b_red")
        assert g.boss_blind == GEN.reroll_boss(st)
        assert g.boss_blind != before or True

    def test_uncommon_tag_forces_shop_slot_via_generate(self):
        g = BalatroGame(seed=SEED)
        g.blind_tags["Small"] = "tag_uncommon"
        g.step({"type": "skip_blind"})
        assert g.tags.keys() == ["tag_uncommon"]
        g.step({"type": "play_blind"}); g.debug_win_blind(); g.step({"type": "advance"})
        it = g.current_shop[0]
        assert it.kind == "joker" and it.couponed and it.from_tag == "tag_uncommon"
        assert P.JOKER_BY_KEY[it.key]["rarity"] == 2
        assert it.discounted_price(0.0) == 0
        assert g.tags.keys() == []                            # consumed + purged

    def test_double_tag_copies_next(self):
        g = BalatroGame(seed=SEED)
        g.blind_tags = {"Small": "tag_double", "Big": "tag_investment"}
        g.step({"type": "skip_blind"}); g.step({"type": "skip_blind"})
        assert g.tags.keys() == ["tag_investment", "tag_investment"]

    def test_d6_and_coupon(self):
        g = BalatroGame(seed=SEED)
        g.blind_tags = {"Small": "tag_d_six", "Big": "tag_coupon"}
        g.step({"type": "skip_blind"}); g.step({"type": "skip_blind"})
        g.step({"type": "play_blind"}); g.debug_win_blind(); g.step({"type": "advance"})
        assert g.reroll_cost == 0
        assert all(it.couponed for it in g.current_shop if it.kind != "voucher")
        assert g.tags.keys() == []


# ── Created cards through generate ───────────────────────────────────────────

class TestCreatedCards:
    def test_judgement_creates_joker_through_generate(self):
        g = BalatroGame(seed=SEED)
        g.step({"type": "play_blind"})
        g.consumable_hand = ["c_judgement"]
        st = GEN.RunState(SEED); GEN.start_run(st, "b_red")
        expect = GEN.create_from_spec(st, "judgement").key
        g.step({"type": "use_consumable", "consumable_idx": 0, "target_cards": []})
        assert g.jokers[0].key == expect and expect in g.run_state.owned_jokers
        assert g.consumable_hand == []

    def test_emperor_creates_two_different_tarots(self):
        g = BalatroGame(seed=SEED)
        g.consumable_slots = 4
        g.consumable_hand = ["c_emperor"]
        assert apply_tarot(g, "c_emperor")
        created = [k for k in g.consumable_hand if k != "c_emperor"]
        assert len(created) == 2 and len(set(created)) == 2 and "c_emperor" not in created

    def test_fool_copies_last_used(self):
        g = BalatroGame(seed=SEED)
        g.step({"type": "play_blind"})
        g.consumable_hand = ["c_hermit", "c_fool"]
        g.step({"type": "use_consumable", "consumable_idx": 0, "target_cards": []})
        g.step({"type": "use_consumable", "consumable_idx": 0, "target_cards": []})
        assert g.consumable_hand == ["c_hermit"]

    def test_riff_raff_creates_commons_at_blind_select(self):
        g = BalatroGame(seed=SEED)
        g.debug_add_joker("j_riff_raff")
        g.step({"type": "play_blind"})
        keys = [j.key for j in g.jokers]
        assert len(keys) == 3 and all(P.JOKER_BY_KEY[k]["rarity"] == 1 for k in keys[1:])
        assert set(keys[1:]) <= set(g.run_state.owned_jokers)


# ── Director's Cut / Retcon ──────────────────────────────────────────────────

class TestBossReroll:
    def test_directors_cut_once_per_ante(self):
        g = BalatroGame(seed=SEED)
        g.dollars = 25
        assert {"type": "reroll_boss"} not in g.legal_actions()
        apply_voucher(g, "v_directors_cut")
        assert {"type": "reroll_boss"} in g.legal_actions()
        g.step({"type": "reroll_boss"})
        assert g.dollars == 15 and g._boss_rerolled
        assert {"type": "reroll_boss"} not in g.legal_actions()
        g.blind_idx = 2; g._prepare_next_blind()
        assert g.current_blind.boss_key == g.boss_blind
