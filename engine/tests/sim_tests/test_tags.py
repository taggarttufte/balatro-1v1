"""
test_tags.py -- effects of all 24 Balatro tags (engine/balatro_sim/tags.py) against a fake
TagContext, plus the ordering rules (Double Tag stacking, two Investment tags, Juggle
stacking, one-pack-per-pass at blind select, per-shop guards) and a cross-check that the
tag table equals rng/pools.py.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from balatro_sim.tags import (
    TAG_DEFS, TAG_EDITIONS, TAG_KEYS, TAG_PACKS,
    BlindChoiceOutcome, TagContext, TagHookNotImplemented, TagInstance, TagState, Trigger,
    apply_tag, tag_is_consumed_at,
)


# ---------------------------------------------------------------------------------------
# Fake context
# ---------------------------------------------------------------------------------------

class FakeCard:
    def __init__(self, kind="Joker", edition=None, tag=None):
        self.kind = kind
        self.edition = edition
        self.couponed = False
        self.tag = tag          # (rarity, key_append) when tag-created


class FakeCtx(TagContext):
    """Records every hook call; exposes a few knobs for the effects that branch."""

    def __init__(self, **fields):
        super().__init__(**fields)
        self.calls = []
        self.joker_count = 0
        self.joker_limit = 5
        self.negative_spawn = False     # spawned jokers do not consume a slot
        self.rare_available = True
        self.hand_levels = {}
        self.hand_size = 8
        self.temp_reroll = None
        self.packs = []
        self.boss_rerolls = 0
        self.vouchers_added = 0
        self.shop_free_calls = 0
        self.money_log = []

    def add_dollars(self, amount, source):
        self.calls.append(("add_dollars", amount, source))
        self.money_log.append((source, amount))
        self.dollars += amount

    def joker_slots_free(self):
        return self.joker_limit - self.joker_count

    def spawn_joker_to_slots(self, rarity, key_append):
        self.calls.append(("spawn_joker_to_slots", rarity, key_append))
        if not self.negative_spawn:
            self.joker_count += 1
        return FakeCard("Joker", tag=(rarity, key_append))

    def create_shop_joker(self, rarity, key_append):
        self.calls.append(("create_shop_joker", rarity, key_append))
        return FakeCard("Joker", tag=(rarity, key_append))

    def rare_joker_available(self):
        return self.rare_available

    def card_is_editionless_joker(self, card):
        return card.kind == "Joker" and card.edition is None

    def set_card_edition(self, card, edition):
        self.calls.append(("set_card_edition", edition))
        card.edition = edition

    def mark_card_couponed(self, card):
        self.calls.append(("mark_card_couponed",))
        card.couponed = True

    def level_up_hand(self, hand, levels):
        self.calls.append(("level_up_hand", hand, levels))
        self.hand_levels[hand] = self.hand_levels.get(hand, 1) + levels

    def choose_orbital_hand(self, blind_type):
        self.calls.append(("choose_orbital_hand", blind_type))
        return "Pair"

    def change_hand_size(self, delta):
        self.calls.append(("change_hand_size", delta))
        self.hand_size += delta

    def open_pack(self, pack_key):
        self.calls.append(("open_pack", pack_key))
        self.packs.append(pack_key)

    def reroll_boss(self):
        self.calls.append(("reroll_boss",))
        self.boss_rerolls += 1

    def add_shop_voucher(self):
        self.calls.append(("add_shop_voucher",))
        self.vouchers_added += 1

    def set_temp_reroll_cost(self, cost):
        self.calls.append(("set_temp_reroll_cost", cost))
        self.temp_reroll = cost

    def clear_temp_reroll_cost(self):
        self.calls.append(("clear_temp_reroll_cost",))
        self.temp_reroll = None

    def make_shop_free(self):
        self.calls.append(("make_shop_free",))
        self.shop_free_calls += 1

    def names(self):
        return [c[0] for c in self.calls]


@pytest.fixture
def ctx():
    return FakeCtx()


def state_with(*keys, **kw):
    st = TagState()
    c = FakeCtx()
    for k in keys:
        st.acquire(k, c, **kw)
    return st


# ---------------------------------------------------------------------------------------
# Table / interface
# ---------------------------------------------------------------------------------------

def _load_pools():
    mp_root = Path(__file__).resolve().parents[3]
    pools_py = mp_root / "rng" / "pools.py"
    if not pools_py.exists():
        pytest.skip("rng/pools.py not found at %s" % pools_py)
    if str(mp_root) not in sys.path:
        sys.path.insert(0, str(mp_root))
    spec = importlib.util.spec_from_file_location("rng.pools", pools_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestTable:
    def test_24_tags(self):
        assert len(TAG_DEFS) == 24
        assert [TAG_DEFS[k].order for k in TAG_KEYS] == list(range(1, 25))

    def test_keys_names_configs_match_pools(self):
        P = _load_pools()
        pool_keys = [t["key"] for t in P.TAGS]
        assert TAG_KEYS == pool_keys
        for t in P.TAGS:
            d = TAG_DEFS[t["key"]]
            assert d.name == t["name"]
            assert d.order == t["order"]
            cfg = dict(t["config"])
            assert cfg.pop("type") == d.trigger.value
            assert cfg == d.config

    @pytest.mark.parametrize("key", TAG_KEYS)
    def test_consumed_at(self, key):
        trig = tag_is_consumed_at(key)
        assert isinstance(trig, Trigger)
        assert trig is TAG_DEFS[key].trigger

    def test_trigger_aliases(self):
        assert Trigger.ON_ACQUIRE is Trigger.TAG_ADD
        assert Trigger.ON_BLIND_SELECT is Trigger.NEW_BLIND_CHOICE
        assert Trigger.ON_ROUND_END is Trigger.EVAL
        assert Trigger.ON_ROUND_START is Trigger.ROUND_START_BONUS
        assert Trigger.ON_SHOP_ENTER is Trigger.SHOP_START
        assert Trigger("immediate") is Trigger.IMMEDIATE

    @pytest.mark.parametrize("key", TAG_KEYS)
    def test_wrong_trigger_never_consumes(self, key, ctx):
        mine = tag_is_consumed_at(key)
        for trig in Trigger:
            if trig is mine:
                continue
            t = TagInstance(key, orbital_hand="Pair")
            assert apply_tag(t, trig, ctx, card=FakeCard(), added=TagInstance("tag_handy")) is False
            assert t.triggered is False
        assert ctx.calls == []

    def test_unknown_key_rejected(self, ctx):
        with pytest.raises(KeyError):
            TagState().acquire("tag_bogus", ctx)

    def test_missing_hook_is_loud(self):
        bare = TagContext(dollars=10)
        with pytest.raises(TagHookNotImplemented):
            apply_tag("tag_economy", Trigger.IMMEDIATE, bare)
        with pytest.raises(TypeError):
            TagContext(nope=1)

    def test_clone_is_independent(self):
        st = state_with("tag_handy", "tag_juggle")
        st.shop_free = True
        cl = st.clone()
        cl.tags.pop()
        cl.shop_free = False
        assert st.keys() == ["tag_handy", "tag_juggle"] and st.shop_free is True
        assert cl.keys() == ["tag_handy"]


# ---------------------------------------------------------------------------------------
# 'immediate' tags: Handy, Garbage, Skip, Economy, Top-up, Orbital
# ---------------------------------------------------------------------------------------

class TestImmediate:
    def test_handy_pays_per_hand_played(self):
        c = FakeCtx(hands_played=17)
        assert apply_tag("tag_handy", Trigger.IMMEDIATE, c)
        assert c.money_log == [("tag_handy", 17)]

    def test_handy_zero_hands(self):
        c = FakeCtx(hands_played=0)
        assert apply_tag("tag_handy", Trigger.IMMEDIATE, c)
        assert c.money_log == [("tag_handy", 0)]

    def test_garbage_pays_per_unused_discard(self):
        c = FakeCtx(unused_discards=9)
        assert apply_tag("tag_garbage", Trigger.IMMEDIATE, c)
        assert c.money_log == [("tag_garbage", 9)]

    def test_skip_pays_5_per_skip_including_this_one(self):
        # skips is incremented before add_tag; first skip of the run -> $5
        c = FakeCtx(skips=1)
        assert apply_tag("tag_skip", Trigger.IMMEDIATE, c)
        assert c.money_log == [("tag_skip", 5)]
        c = FakeCtx(skips=4)
        apply_tag("tag_skip", Trigger.IMMEDIATE, c)
        assert c.money_log == [("tag_skip", 20)]

    @pytest.mark.parametrize("dollars,expected", [(0, 0), (-7, 0), (13, 13), (40, 40), (41, 40), (999, 40)])
    def test_economy_doubles_up_to_40(self, dollars, expected):
        c = FakeCtx(dollars=dollars)
        assert apply_tag("tag_economy", Trigger.IMMEDIATE, c)
        assert c.money_log == [("tag_economy", expected)]
        assert c.dollars == dollars + expected

    def test_top_up_spawns_two_commons(self, ctx):
        assert apply_tag("tag_top_up", Trigger.IMMEDIATE, ctx)
        assert ctx.calls == [("spawn_joker_to_slots", 0, "top")] * 2
        assert ctx.joker_count == 2

    def test_top_up_respects_slots(self):
        c = FakeCtx()
        c.joker_count = 4
        apply_tag("tag_top_up", Trigger.IMMEDIATE, c)
        assert c.names().count("spawn_joker_to_slots") == 1
        c = FakeCtx()
        c.joker_count = 5
        assert apply_tag("tag_top_up", Trigger.IMMEDIATE, c)   # consumed even with no room
        assert c.calls == []

    def test_top_up_rechecks_slots_after_each_spawn(self):
        # a Negative joker spawned first leaves the slot free -> second spawn still happens
        c = FakeCtx()
        c.joker_count = 4
        c.negative_spawn = True
        apply_tag("tag_top_up", Trigger.IMMEDIATE, c)
        assert c.names().count("spawn_joker_to_slots") == 2

    def test_orbital_levels_its_hand_3(self, ctx):
        assert apply_tag("tag_orbital", Trigger.IMMEDIATE, ctx, orbital_hand="Flush")
        assert ctx.calls == [("level_up_hand", "Flush", 3)]

    def test_orbital_without_hand_raises(self, ctx):
        with pytest.raises(ValueError):
            apply_tag("tag_orbital", Trigger.IMMEDIATE, ctx)

    def test_orbital_hand_chosen_at_acquire_via_ctx(self, ctx):
        st = TagState()
        st.acquire("tag_orbital", ctx, blind_type="Small")
        assert ("choose_orbital_hand", "Small") in ctx.calls
        assert st.tags[0].orbital_hand == "Pair"
        st.on_immediate(ctx)
        assert ("level_up_hand", "Pair", 3) in ctx.calls

    def test_orbital_explicit_hand_wins_over_ctx(self, ctx):
        st = TagState()
        st.acquire("tag_orbital", ctx, blind_type="Big", orbital_hand="Straight")
        assert "choose_orbital_hand" not in ctx.names()
        assert st.tags[0].orbital_hand == "Straight"

    def test_immediate_fires_all_in_order(self):
        c = FakeCtx(hands_played=3, unused_discards=4, skips=2, dollars=10)
        st = state_with("tag_handy", "tag_garbage", "tag_skip", "tag_economy", "tag_juggle")
        ev = st.on_immediate(c)
        assert [e.key for e in ev] == ["tag_handy", "tag_garbage", "tag_skip", "tag_economy"]
        # Economy's dollars read happens in its own queued event AFTER the earlier tags'
        # queued ease_dollars events (common_events.lua:68-108, tag.lua:181-187): it sees
        # 10 + 3 + 4 + 10 = 27.
        assert c.money_log == [("tag_handy", 3), ("tag_garbage", 4), ("tag_skip", 10), ("tag_economy", 27)]
        assert st.keys() == ["tag_juggle"]

    def test_economy_before_handy_does_not_see_handy_money(self):
        c = FakeCtx(hands_played=3, dollars=10)
        st = state_with("tag_economy", "tag_handy")
        st.on_immediate(c)
        assert c.money_log == [("tag_economy", 10), ("tag_handy", 3)]


# ---------------------------------------------------------------------------------------
# 'eval': Investment
# ---------------------------------------------------------------------------------------

class TestInvestment:
    def test_not_paid_after_non_boss(self):
        c = FakeCtx(last_blind_was_boss=False)
        st = state_with("tag_investment")
        assert st.on_round_eval(c) == 0
        assert st.keys() == ["tag_investment"]
        assert c.calls == []

    def test_paid_after_boss(self):
        c = FakeCtx(last_blind_was_boss=True)
        st = state_with("tag_investment")
        assert st.on_round_eval(c) == 25
        assert c.money_log == [("tag_investment", 25)]
        assert st.keys() == []

    def test_two_investments_both_pay_same_boss(self):
        c = FakeCtx(last_blind_was_boss=True)
        st = state_with("tag_investment", "tag_investment")
        assert st.on_round_eval(c) == 50
        assert c.money_log == [("tag_investment", 25)] * 2
        assert st.keys() == []

    def test_survives_small_and_big_then_pays(self):
        st = state_with("tag_investment")
        assert st.on_round_eval(FakeCtx(last_blind_was_boss=False)) == 0
        assert st.on_round_eval(FakeCtx(last_blind_was_boss=False)) == 0
        assert st.on_round_eval(FakeCtx(last_blind_was_boss=True)) == 25


# ---------------------------------------------------------------------------------------
# 'new_blind_choice': pack tags + Boss Tag
# ---------------------------------------------------------------------------------------

class TestNewBlindChoice:
    @pytest.mark.parametrize("key,pack", sorted(TAG_PACKS.items()))
    def test_pack_tag_opens_right_free_pack(self, key, pack, ctx):
        assert apply_tag(key, Trigger.NEW_BLIND_CHOICE, ctx)
        assert ctx.calls == [("open_pack", pack)]

    def test_pack_keys(self):
        assert TAG_PACKS == {
            "tag_charm": "p_arcana_mega_1", "tag_meteor": "p_celestial_mega_1",
            "tag_ethereal": "p_spectral_normal_1", "tag_standard": "p_standard_mega_1",
            "tag_buffoon": "p_buffoon_mega_1",
        }

    def test_boss_tag_rerolls_boss(self, ctx):
        assert apply_tag("tag_boss", Trigger.NEW_BLIND_CHOICE, ctx)
        assert ctx.calls == [("reroll_boss",)]

    def test_one_pack_per_pass_then_resume(self, ctx):
        st = state_with("tag_charm", "tag_meteor")
        out = st.on_new_blind_choice(ctx)
        assert isinstance(out, BlindChoiceOutcome)
        assert out.pending_pack == "p_arcana_mega_1"
        assert st.keys() == ["tag_meteor"]
        assert ctx.packs == ["p_arcana_mega_1"]
        # engine resolves the pack, then calls again
        out = st.on_new_blind_choice(ctx)
        assert out.pending_pack == "p_celestial_mega_1"
        assert st.keys() == []
        out = st.on_new_blind_choice(ctx)
        assert out.pending_pack is None and out.events == []

    def test_boss_tag_chains_into_next_choice_tag(self, ctx):
        # reroll_boss re-runs the new_blind_choice loop: Boss then Charm in ONE call
        st = state_with("tag_boss", "tag_charm")
        out = st.on_new_blind_choice(ctx)
        assert ctx.boss_rerolls == 1
        assert out.pending_pack == "p_arcana_mega_1"
        assert [e.key for e in out.events] == ["tag_boss", "tag_charm"]
        assert st.keys() == []

    def test_two_boss_tags_reroll_twice(self, ctx):
        st = state_with("tag_boss", "tag_boss")
        out = st.on_new_blind_choice(ctx)
        assert ctx.boss_rerolls == 2
        assert out.pending_pack is None
        assert st.keys() == []

    def test_pack_before_boss_blocks_boss_until_pack_closed(self, ctx):
        st = state_with("tag_standard", "tag_boss")
        out = st.on_new_blind_choice(ctx)
        assert out.pending_pack == "p_standard_mega_1"
        assert ctx.boss_rerolls == 0
        st.on_new_blind_choice(ctx)
        assert ctx.boss_rerolls == 1

    def test_on_blind_select_runs_immediate_then_choice(self):
        c = FakeCtx(hands_played=2)
        st = state_with("tag_buffoon", "tag_handy")
        out = st.on_blind_select(c)
        assert c.money_log == [("tag_handy", 2)]
        assert out.pending_pack == "p_buffoon_mega_1"
        assert [e.key for e in out.events] == ["tag_handy", "tag_buffoon"]


# ---------------------------------------------------------------------------------------
# 'voucher_add', 'shop_start', 'shop_final_pass'
# ---------------------------------------------------------------------------------------

class TestShopPasses:
    def test_voucher_tag_adds_voucher(self, ctx):
        st = state_with("tag_voucher")
        ev = st.on_voucher_add(ctx)
        assert ctx.vouchers_added == 1 and [e.key for e in ev] == ["tag_voucher"]
        assert st.keys() == []

    def test_two_voucher_tags_add_two(self, ctx):
        st = state_with("tag_voucher", "tag_voucher")
        st.on_voucher_add(ctx)
        assert ctx.vouchers_added == 2

    def test_d6_zeroes_reroll_base_once_per_shop(self, ctx):
        st = state_with("tag_d_six", "tag_d_six")
        st.on_cash_out()
        ev = st.on_shop_start(ctx)
        assert [e.key for e in ev] == ["tag_d_six"]
        assert ctx.calls == [("set_temp_reroll_cost", 0)]
        assert st.keys() == ["tag_d_six"]            # second D6 waits
        assert st.on_shop_start(ctx) == []           # re-running the shop build: guarded
        # end of next round: cleanup clears the temp cost
        st.on_round_end_cleanup(ctx)
        assert ctx.calls[-1] == ("clear_temp_reroll_cost",)
        st.on_cash_out()
        ev = st.on_shop_start(ctx)
        assert [e.key for e in ev] == ["tag_d_six"] and st.keys() == []

    def test_coupon_frees_shop_once_per_shop(self, ctx):
        st = state_with("tag_coupon", "tag_coupon")
        st.on_cash_out()
        ev = st.on_shop_final_pass(ctx)
        assert [e.key for e in ev] == ["tag_coupon"]
        assert ctx.shop_free_calls == 1
        assert st.keys() == ["tag_coupon"]
        assert st.on_shop_final_pass(ctx) == []
        st.on_cash_out()
        st.on_shop_final_pass(ctx)
        assert ctx.shop_free_calls == 2 and st.keys() == []

    def test_cleanup_noop_without_d6_or_juggle(self, ctx):
        TagState().on_round_end_cleanup(ctx)
        assert ctx.calls == []


# ---------------------------------------------------------------------------------------
# 'store_joker_create' / 'store_joker_modify'
# ---------------------------------------------------------------------------------------

class TestShopJokerTags:
    def test_uncommon_forces_free_uncommon(self, ctx):
        st = state_with("tag_uncommon")
        card = st.store_joker_create(ctx)
        assert card.tag == (0.9, "uta") and card.couponed
        assert ctx.calls == [("create_shop_joker", 0.9, "uta"), ("mark_card_couponed",)]
        assert st.keys() == []

    def test_rare_forces_free_rare(self, ctx):
        st = state_with("tag_rare")
        card = st.store_joker_create(ctx)
        assert card.tag == (1, "rta") and card.couponed
        assert st.keys() == []

    def test_rare_nopes_when_no_rare_left_and_scan_continues(self):
        c = FakeCtx()
        c.rare_available = False
        st = state_with("tag_rare", "tag_uncommon")
        card = st.store_joker_create(c)
        assert card is not None and card.tag == (0.9, "uta")
        assert c.names().count("create_shop_joker") == 1
        assert st.keys() == []        # Rare consumed by nope, Uncommon consumed by firing

    def test_rare_nope_alone_returns_none(self):
        c = FakeCtx()
        c.rare_available = False
        st = state_with("tag_rare")
        assert st.store_joker_create(c) is None
        assert st.keys() == [] and c.calls == []

    def test_no_create_tags_returns_none(self, ctx):
        st = state_with("tag_foil")
        assert st.store_joker_create(ctx) is None
        assert st.keys() == ["tag_foil"]

    def test_one_forced_card_per_slot(self, ctx):
        st = state_with("tag_uncommon", "tag_uncommon")
        assert st.store_joker_create(ctx) is not None
        assert st.keys() == ["tag_uncommon"]
        assert st.store_joker_create(ctx) is not None
        assert st.keys() == []

    @pytest.mark.parametrize("key,edition", sorted(TAG_EDITIONS.items()))
    def test_edition_tag_on_editionless_joker(self, key, edition, ctx):
        card = FakeCard("Joker")
        assert apply_tag(key, Trigger.STORE_JOKER_MODIFY, ctx, card=card)
        assert card.edition == edition and card.couponed
        assert ctx.calls == [("set_card_edition", edition), ("mark_card_couponed",)]

    def test_edition_tag_skips_non_joker_and_editioned(self, ctx):
        st = state_with("tag_foil")
        assert st.store_joker_modify(ctx, FakeCard("Tarot")) is False
        assert st.store_joker_modify(ctx, FakeCard("Joker", edition="holo")) is False
        assert st.keys() == ["tag_foil"] and ctx.calls == []

    def test_only_first_edition_tag_applies_per_card(self, ctx):
        st = state_with("tag_foil", "tag_holo")
        c1, c2, c3 = FakeCard(), FakeCard(), FakeCard()
        assert st.store_joker_modify(ctx, c1) and c1.edition == "foil"
        assert st.keys() == ["tag_holo"]
        assert st.store_joker_modify(ctx, c2) and c2.edition == "holo"
        assert st.store_joker_modify(ctx, c3) is False and c3.edition is None
        assert st.keys() == []

    def test_forced_card_also_gets_edition(self, ctx):
        st = state_with("tag_negative", "tag_uncommon")
        card = st.store_joker_create(ctx)
        assert card.tag == (0.9, "uta") and card.edition == "negative" and card.couponed
        assert st.keys() == []


# ---------------------------------------------------------------------------------------
# 'round_start_bonus': Juggle
# ---------------------------------------------------------------------------------------

class TestJuggle:
    def test_juggle_plus_3_then_reverted(self, ctx):
        st = state_with("tag_juggle")
        ev = st.on_round_start(ctx)
        assert [e.key for e in ev] == ["tag_juggle"]
        assert ctx.hand_size == 11 and st.temp_handsize == 3 and st.keys() == []
        st.on_round_end_cleanup(ctx)
        assert ctx.hand_size == 8 and st.temp_handsize == 0
        st.on_round_end_cleanup(ctx)       # idempotent
        assert ctx.hand_size == 8

    def test_two_juggles_stack_to_plus_6(self, ctx):
        st = state_with("tag_juggle", "tag_juggle")
        st.on_round_start(ctx)
        assert ctx.hand_size == 14 and st.temp_handsize == 6
        st.on_round_end_cleanup(ctx)
        assert ctx.hand_size == 8

    def test_juggle_does_not_fire_at_immediate(self, ctx):
        st = state_with("tag_juggle")
        st.on_blind_select(ctx)
        assert st.keys() == ["tag_juggle"] and ctx.calls == []


# ---------------------------------------------------------------------------------------
# Double Tag
# ---------------------------------------------------------------------------------------

class TestDoubleTag:
    def test_double_copies_next_tag(self, ctx):
        st = state_with("tag_double")
        ev = st.acquire("tag_handy", ctx)
        assert st.keys() == ["tag_handy", "tag_handy"]
        assert [e.key for e in ev] == ["tag_double"] and ev[0].detail == {"copied": "tag_handy"}

    def test_double_does_not_copy_double(self, ctx):
        st = state_with("tag_double")
        st.acquire("tag_double", ctx)
        assert st.keys() == ["tag_double", "tag_double"]

    def test_two_doubles_make_three(self, ctx):
        st = state_with("tag_double", "tag_double")
        st.acquire("tag_investment", ctx)
        assert st.keys() == ["tag_investment"] * 3

    def test_double_copies_orbital_hand(self, ctx):
        st = state_with("tag_double")
        st.acquire("tag_orbital", ctx, orbital_hand="Two Pair")
        assert [t.orbital_hand for t in st.tags] == ["Two Pair", "Two Pair"]
        st.on_immediate(ctx)
        assert ctx.calls == [("level_up_hand", "Two Pair", 3)] * 2

    def test_double_plus_handy_pays_twice_on_skip(self):
        c = FakeCtx(hands_played=6, skips=2)
        st = state_with("tag_double")
        out = st.skip_blind("tag_handy", c)
        assert c.money_log == [("tag_handy", 6), ("tag_handy", 6)]
        assert [e.key for e in out.events] == ["tag_double", "tag_handy", "tag_handy"]
        assert st.keys() == []

    def test_double_plus_boss_rerolls_twice(self, ctx):
        st = state_with("tag_double")
        st.skip_blind("tag_boss", ctx)
        assert ctx.boss_rerolls == 2 and st.keys() == []

    def test_double_plus_charm_two_packs_one_at_a_time(self, ctx):
        st = state_with("tag_double")
        out = st.skip_blind("tag_charm", ctx)
        assert out.pending_pack == "p_arcana_mega_1" and st.keys() == ["tag_charm"]
        out = st.on_new_blind_choice(ctx)
        assert out.pending_pack == "p_arcana_mega_1" and st.keys() == []

    def test_double_plus_investment_50(self, ctx):
        st = state_with("tag_double")
        st.skip_blind("tag_investment", ctx)
        assert st.keys() == ["tag_investment", "tag_investment"]
        assert st.on_round_eval(FakeCtx(last_blind_was_boss=True)) == 50

    def test_double_acquired_after_tags_only_affects_later_ones(self, ctx):
        st = state_with("tag_handy", "tag_double")
        st.acquire("tag_garbage", ctx)
        assert st.keys() == ["tag_handy", "tag_garbage", "tag_garbage"]

    def test_double_persists_across_other_passes(self, ctx):
        st = state_with("tag_double")
        st.on_blind_select(ctx)
        st.on_round_start(ctx)
        st.on_round_eval(ctx)
        st.on_shop_start(ctx)
        st.on_shop_final_pass(ctx)
        assert st.keys() == ["tag_double"] and ctx.calls == []

    def test_apply_tag_primitive_for_double(self, ctx):
        d = TagInstance("tag_double")
        assert apply_tag(d, Trigger.TAG_ADD, ctx, added=TagInstance("tag_double")) is False
        assert apply_tag(d, Trigger.TAG_ADD, ctx, added=TagInstance("tag_skip")) is True
        assert d.triggered


# ---------------------------------------------------------------------------------------
# Full-cycle ordering
# ---------------------------------------------------------------------------------------

class TestRunOrdering:
    def test_skip_small_skip_big(self):
        """Skip Small (Double), skip Big (Skip Tag -> two Skip Tags, each pays skips*5 = $10)."""
        c = FakeCtx(skips=1)
        st = TagState()
        out = st.skip_blind("tag_double", c)
        assert out.events == [] and st.keys() == ["tag_double"]
        c.skips = 2
        st.skip_blind("tag_skip", c)
        assert c.money_log == [("tag_skip", 10), ("tag_skip", 10)]
        assert st.keys() == []

    def test_shop_cycle_with_every_shop_tag(self):
        c = FakeCtx()
        st = state_with("tag_d_six", "tag_uncommon", "tag_polychrome", "tag_voucher", "tag_coupon")
        st.on_cash_out()
        st.on_shop_start(c)
        slot1 = st.store_joker_create(c)
        assert slot1.edition == "polychrome" and slot1.couponed
        assert st.store_joker_create(c) is None
        natural = FakeCard("Joker")
        assert st.store_joker_modify(c, natural) is False      # Polychrome already spent
        st.on_voucher_add(c)
        st.on_shop_final_pass(c)
        assert c.temp_reroll == 0 and c.vouchers_added == 1 and c.shop_free_calls == 1
        assert st.keys() == []
        st.on_round_end_cleanup(c)
        assert c.temp_reroll is None

    def test_tally_increments_per_instance(self, ctx):
        st = state_with("tag_double")
        st.acquire("tag_handy", ctx)
        assert st.tag_tally == 3
        assert sorted(t.uid for t in st.tags) == [2, 3]
