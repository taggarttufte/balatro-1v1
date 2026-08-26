"""
test_game_keys.py — the engine speaks GAME keys, with rng/pools.py as the
single source of truth (Phase 1 W1, 2026-08-21).

Guards:
  * every pools joker key has exactly one registry implementation, and every
    registry key is a pools key (1:1, asserted both at runtime and by counting
    `JOKER_REGISTRY["key"] =` sites in the source);
  * JOKER_CATALOGUE rarity / cost / name equal pools for all 150;
  * all 22 tarots, 12 planets, 18 spectrals, 32 vouchers, 28 bosses and
    15 booster types are present under game keys and dispatch;
  * tags / decks / stakes catalogues are exposed through constants;
  * no legacy (pre-re-key) key literal survives anywhere under balatro_sim/.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pytest

import balatro_sim
import balatro_sim.jokers  # noqa: F401  (populates the registry)
from balatro_sim import constants, game_keys
from balatro_sim.consumables import (
    ALL_PLANETS, ALL_SPECTRALS, ALL_TAROTS, ALL_VOUCHERS,
    PLANET_HAND, PLANET_NAME, SPECTRAL_NAME, TAROT_NAME, VOUCHER_NAME,
    VOUCHER_REQUIRES, apply_planet, apply_spectral, apply_tarot, apply_voucher,
)
from balatro_sim.game import (
    ALL_REGULAR_BOSS_BLINDS, BOSS_CHIP_MULT, REGULAR_BOSS_BLINDS,
    SHOWDOWN_BOSS_BLINDS, UNMODELLED_BOSS_BLINDS, BalatroGame,
)
from balatro_sim.jokers.base import JOKER_REGISTRY, JokerInstance
from balatro_sim.shop import BOOSTER_CATALOGUE, BOOSTER_PICKS, JOKER_CATALOGUE

pools = game_keys.pools
SIM_ROOT = Path(balatro_sim.__file__).resolve().parent
JOKER_SRC = sorted((SIM_ROOT / "jokers").glob("*.py"))


# ── jokers ───────────────────────────────────────────────────────────────────

class TestJokerRegistryIsOneToOneWithPools:
    def test_pools_has_150_jokers(self):
        assert len(pools.JOKERS) == 150
        assert Counter(j["rarity"] for j in pools.JOKERS) == {1: 61, 2: 64, 3: 20, 4: 5}

    def test_registry_keys_equal_pools_keys(self):
        assert set(JOKER_REGISTRY) == set(game_keys.JOKER_KEYS)

    def test_each_game_key_registered_exactly_once_in_source(self):
        """Count `JOKER_REGISTRY["<key>"] =` sites across jokers/*.py."""
        sites: Counter[str] = Counter()
        pat = re.compile(r'^\s*JOKER_REGISTRY\["([^"]+)"\]\s*=', re.M)
        for f in JOKER_SRC:
            sites.update(pat.findall(f.read_text(encoding="utf-8")))
        dup = {k: n for k, n in sites.items() if n != 1}
        assert dup == {}, f"keys registered != 1 time: {dup}"
        assert set(sites) == set(game_keys.JOKER_KEYS)

    def test_registry_rejects_duplicate_registration(self):
        with pytest.raises(KeyError):
            JOKER_REGISTRY["j_joker"] = object()

    def test_every_implementation_has_at_least_one_hook(self):
        hooks = ("pre_score", "on_score_card", "on_hand_scored", "on_discard",
                 "on_round_end", "on_blind_selected", "on_boss_beaten",
                 "on_planet_used", "on_tarot_used", "on_sell", "on_shop_enter",
                 "on_shop_leave", "on_held_card", "on_card_destroyed",
                 "on_card_added", "on_blind_skipped", "on_reroll", "on_init",
                 "on_booster_opened", "on_boss_ability_triggered", "on_card_sold",
                 "on_first_hand_drawn")
        hookless = sorted(k for k, impl in JOKER_REGISTRY.items()
                          if not any(hasattr(impl, h) for h in hooks))
        # Passives (W3): their effect is read by game.py / shop.py / scoring.py
        # through base.passive_modifiers (hand size, Credit Card debt floor,
        # Chaos free reroll, Astronomer pricing), base.sync_probabilities
        # (Oops! All 6s), run_state.showman, the Chicot boss-disable in
        # _start_blind and the Mime held-card retrigger loop — not through a
        # hook on the effect object.
        assert hookless == ["j_astronomer", "j_chaos", "j_chicot", "j_credit_card",
                            "j_drunkard", "j_juggler", "j_merry_andy", "j_mime",
                            "j_oops", "j_ring_master", "j_troubadour"], hookless

    def test_former_dead_aliases_are_gone_or_real(self):
        for dead in ("j_lucky_joker", "j_oops_all_sixes", "j_showman",
                     "j_space_joker", "j_golden_ticket"):
            assert dead not in JOKER_REGISTRY
        for real in ("j_ring_master", "j_oops", "j_space", "j_ticket", "j_lucky_cat"):
            assert real in JOKER_REGISTRY and real in JOKER_CATALOGUE


class TestJokerCatalogueMatchesPools:
    def test_catalogue_keys_equal_pools_keys(self):
        assert set(JOKER_CATALOGUE) == set(game_keys.JOKER_KEYS)
        assert list(JOKER_CATALOGUE) == game_keys.JOKER_KEYS   # game `order`

    @pytest.mark.parametrize("key", game_keys.JOKER_KEYS)
    def test_name_rarity_cost_equal_pools(self, key):
        j = pools.JOKER_BY_KEY[key]
        c = JOKER_CATALOGUE[key]
        assert c["name"] == j["name"]
        assert c["rarity"] == game_keys.RARITY_NAME[j["rarity"]]
        assert c["rarity_id"] == j["rarity"]
        assert c["price"] == j["cost"]
        assert c["order"] == j["order"]

    def test_spot_checks_against_known_game_values(self):
        # The survey's headline mismatches, now fixed.
        assert JOKER_CATALOGUE["j_joker"]["price"] == 2
        assert JOKER_CATALOGUE["j_credit_card"]["price"] == 1
        assert JOKER_CATALOGUE["j_blueprint"]["price"] == 10
        assert JOKER_CATALOGUE["j_dna"]["rarity"] == "Rare"
        assert JOKER_CATALOGUE["j_stencil"]["rarity"] == "Uncommon"
        assert JOKER_CATALOGUE["j_half"]["rarity"] == "Common"
        assert JOKER_CATALOGUE["j_caino"]["rarity"] == "Legendary"
        assert JOKER_CATALOGUE["j_caino"]["price"] == 20
        assert JOKER_CATALOGUE["j_ring_master"]["name"] == "Showman"
        assert JOKER_CATALOGUE["j_gluttenous_joker"]["name"] == "Gluttonous Joker"

    def test_rarity_buckets_match_pools(self):
        for rarity, keys in game_keys.JOKER_KEYS_BY_RARITY.items():
            cat = [k for k, v in JOKER_CATALOGUE.items() if v["rarity"] == rarity]
            assert cat == keys, rarity

    def test_every_catalogue_joker_instantiates(self):
        for key in JOKER_CATALOGUE:
            assert JokerInstance(key).key == key


# ── consumables ──────────────────────────────────────────────────────────────

class TestConsumablesUseGameKeys:
    def test_counts_and_keys(self):
        assert ALL_TAROTS == [c["key"] for c in pools.TAROTS] and len(ALL_TAROTS) == 22
        assert ALL_PLANETS == [c["key"] for c in pools.PLANETS] and len(ALL_PLANETS) == 12
        assert ALL_SPECTRALS == [c["key"] for c in pools.SPECTRALS] and len(ALL_SPECTRALS) == 18
        assert all(k.startswith("c_") for k in ALL_TAROTS + ALL_PLANETS + ALL_SPECTRALS)

    def test_names_match_pools(self):
        assert TAROT_NAME == {c["key"]: c["name"] for c in pools.TAROTS}
        assert PLANET_NAME == {c["key"]: c["name"] for c in pools.PLANETS}
        assert SPECTRAL_NAME == {c["key"]: c["name"] for c in pools.SPECTRALS}

    def test_planet_hand_types_match_pools(self):
        assert PLANET_HAND == {c["key"]: c["hand_type"] for c in pools.PLANETS}

    def test_hierophant_uses_the_games_misspelling(self):
        assert "c_heirophant" in ALL_TAROTS
        assert "c_hierophant" not in ALL_TAROTS

    @pytest.mark.parametrize("key", [c["key"] for c in pools.TAROTS])
    def test_every_tarot_dispatches(self, key):
        g = BalatroGame(seed=1)
        g.step({"type": "play_blind"})
        if key == "c_wheel_of_fortune":
            # card.lua:1534 — usable only with an editionless joker on the board (W3)
            g.debug_add_joker("j_joker")
        if key == "c_fool":
            # card.lua:1553-1555 — usable only after another Tarot/Planet was used (W2)
            apply_tarot(g, "c_hermit")
        assert apply_tarot(g, key, [0, 1]) is True

    @pytest.mark.parametrize("key", [c["key"] for c in pools.PLANETS])
    def test_every_planet_dispatches(self, key):
        g = BalatroGame(seed=1)
        before = dict(g.planet_levels)
        assert apply_planet(g, key) is True
        assert g.planet_levels[PLANET_HAND[key]] == before[PLANET_HAND[key]] + 1

    @pytest.mark.parametrize("key", [c["key"] for c in pools.SPECTRALS])
    def test_every_spectral_dispatches(self, key):
        g = BalatroGame(seed=1)
        g.step({"type": "play_blind"})
        g.jokers.append(JokerInstance("j_joker"))
        assert apply_spectral(g, key, [0]) is True


# ── vouchers ─────────────────────────────────────────────────────────────────

class TestVouchersUseGameKeys:
    def test_counts_names_requires(self):
        assert ALL_VOUCHERS == [v["key"] for v in pools.VOUCHERS] and len(ALL_VOUCHERS) == 32
        assert VOUCHER_NAME == {v["key"]: v["name"] for v in pools.VOUCHERS}
        assert VOUCHER_REQUIRES == pools.VOUCHER_REQUIRES
        assert "v_overstock_norm" in ALL_VOUCHERS and "v_overstock" not in ALL_VOUCHERS
        for k in ("v_seed_money", "v_money_tree", "v_blank", "v_antimatter", "v_retcon"):
            assert k in ALL_VOUCHERS

    @pytest.mark.parametrize("key", [v["key"] for v in pools.VOUCHERS])
    def test_every_voucher_is_accepted(self, key):
        g = BalatroGame(seed=1)
        assert apply_voucher(g, key) is True
        assert key in g.vouchers
        assert apply_voucher(g, key) is False   # no double-buy

    def test_new_voucher_effects(self):
        g = BalatroGame(seed=1)
        assert g.interest_cap == constants.INTEREST_CAP == 5
        apply_voucher(g, "v_seed_money")
        assert g.interest_cap == 10
        apply_voucher(g, "v_money_tree")
        assert g.interest_cap == 20
        slots = g.joker_slots
        apply_voucher(g, "v_antimatter")
        assert g.joker_slots == slots + 1
        snap = g.clone()
        assert snap.interest_cap == 20 and snap.joker_slots == slots + 1

    def test_interest_cap_is_used_at_cash_out(self):
        g = BalatroGame(seed=1)
        g.dollars = 100
        base = g.clone()
        apply_voucher(g, "v_money_tree")
        # same payout path, only the cap differs: min(100//5, cap) -> 5 vs 20
        assert min(g.dollars // constants.INTEREST_RATE, g.interest_cap) == 20
        assert min(base.dollars // constants.INTEREST_RATE, base.interest_cap) == 5


# ── bosses ───────────────────────────────────────────────────────────────────

class TestBossesUseGameKeys:
    def test_all_28_present_and_partitioned(self):
        game_keys_all = set(pools.BOSS_KEYS_ALPHA)
        assert len(game_keys_all) == 28
        engine_all = set(ALL_REGULAR_BOSS_BLINDS) | set(SHOWDOWN_BOSS_BLINDS)
        assert engine_all == game_keys_all
        assert set(REGULAR_BOSS_BLINDS) | set(UNMODELLED_BOSS_BLINDS) == set(ALL_REGULAR_BOSS_BLINDS)
        assert not set(REGULAR_BOSS_BLINDS) & set(UNMODELLED_BOSS_BLINDS)
        assert len(SHOWDOWN_BOSS_BLINDS) == 5
        assert all(k.startswith("bl_final_") for k in SHOWDOWN_BOSS_BLINDS)

    def test_showdowns_renamed_and_fish_present(self):
        everything = set(ALL_REGULAR_BOSS_BLINDS) | set(SHOWDOWN_BOSS_BLINDS) | set(BOSS_CHIP_MULT)
        for old in ("bl_amber", "bl_cerulean", "bl_crimson", "bl_verdant", "bl_violet"):
            assert old not in everything
        # W5: the face-down bosses are modelled (Card.face_down) -> nothing is unmodelled
        assert "bl_fish" in REGULAR_BOSS_BLINDS and UNMODELLED_BOSS_BLINDS == []
        assert BOSS_CHIP_MULT["bl_final_vessel"] == 3.0

    def test_min_max_ante_carried_from_pools(self):
        assert game_keys.BOSS_MIN_ANTE["bl_ox"] == 6
        assert game_keys.BOSS_MIN_ANTE["bl_final_acorn"] == 10
        assert all(game_keys.BOSS_MAX_ANTE[k] == 10 for k in game_keys.BOSS_KEYS)

    def test_selection_only_draws_modelled_keys(self):
        seen = set()
        for seed in range(300):
            g = BalatroGame(seed=seed)
            g.ante, g.blind_idx = 1, 2
            g._prepare_next_blind()
            seen.add(g.current_blind.boss_key)
        assert seen <= set(REGULAR_BOSS_BLINDS)
        assert not seen & set(UNMODELLED_BOSS_BLINDS)


# ── boosters ─────────────────────────────────────────────────────────────────

class TestBoostersUseGameKeys:
    def test_15_types_collapse_32_centers(self):
        # NOTES_POOLS said "13 types"; it is 5 kinds x 3 sizes = 15 (the sim had 13
        # keys only because p_standard_mega / p_buffoon_mega were missing).
        assert len(pools.BOOSTERS) == 32
        assert set(BOOSTER_CATALOGUE) == {game_keys.booster_type_key(b["key"]) for b in pools.BOOSTERS}
        assert len(BOOSTER_CATALOGUE) == 15
        assert "p_standard_mega" in BOOSTER_CATALOGUE and "p_buffoon_mega" in BOOSTER_CATALOGUE

    def test_cost_cards_picks_match_pools(self):
        for b in pools.BOOSTERS:
            t = game_keys.booster_type_key(b["key"])
            name, price, _kind, n_cards = BOOSTER_CATALOGUE[t]
            assert name == b["name"] and price == b["cost"] and n_cards == b["extra"]
            assert BOOSTER_PICKS[t] == b["choose"]
        assert BOOSTER_PICKS["p_arcana_mega"] == 2 and BOOSTER_PICKS["p_arcana_normal"] == 1

    def test_weights_sum_to_pool_total(self):
        total = sum(v["weight"] for v in game_keys.BOOSTER_TYPES.values())
        assert abs(total - pools.BOOSTER_TOTAL_WEIGHT) < 1e-9


# ── tags / decks / stakes catalogues ─────────────────────────────────────────

class TestCataloguesExposedViaConstants:
    def test_tags(self):
        assert constants.TAGS is pools.TAGS and len(constants.TAG_KEYS) == 24
        assert constants.TAG_NAME["tag_d_six"] == "D6 Tag"

    def test_decks(self):
        assert constants.DECKS is pools.BACKS and len(constants.DECK_KEYS) == 15
        assert constants.DECK_KEYS[0] == "b_red"

    def test_stakes(self):
        assert constants.STAKES is pools.STAKES and len(constants.STAKE_KEYS) == 8
        assert constants.STAKE_BY_LEVEL[1]["key"] == "stake_white"

    def test_modifier_key_maps(self):
        assert set(constants.ENHANCEMENT_KEY.values()) - {None} == set(game_keys.ENHANCEMENT_KEYS)
        assert set(constants.EDITION_KEY.values()) == set(game_keys.EDITION_KEYS)


# ── no legacy key literal survives ───────────────────────────────────────────

LEGACY_KEYS = [
    # jokers (21 renames + 6 dup spellings + dead aliases)
    "j_greedy_mult", "j_lusty_mult", "j_wrathful_mult", "j_gluttonous_mult",
    "j_business_card", "j_space_joker", "j_to_do_list", "j_square_joker", "j_gift_card",
    "j_mail_in_rebate", "j_stone_joker", "j_trading_card", "j_spare_trousers", "j_seltzer",
    "j_golden_ticket", "j_smeared_joker", "j_glass_joker", "j_showman", "j_the_idol",
    "j_invisible_joker", "j_burnt_joker", "j_the_duo", "j_the_trio", "j_the_family",
    "j_the_order", "j_the_tribe", "j_wee_joker", "j_oops_all_sixes", "j_lucky_joker",
    # consumables
    "c_hierophant", "c_wheel",
    # vouchers / bosses / boosters
    "v_overstock", "bl_amber", "bl_cerulean", "bl_crimson", "bl_verdant", "bl_violet",
    "p_arcana", "p_celestial", "p_spectral", "p_standard", "p_buffoon",
]
_LEGACY_LITERAL = re.compile(
    "[\"'](?:%s)[\"']" % "|".join(re.escape(k) for k in LEGACY_KEYS)
)
_LEGACY_PREFIX = re.compile("[\"'](?:pl_|s_)[a-z_]+[\"']")


def test_no_legacy_key_literals_in_engine_source():
    offenders = []
    for f in sorted(SIM_ROOT.rglob("*.py")):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if _LEGACY_LITERAL.search(line) or _LEGACY_PREFIX.search(line):
                offenders.append(f"{f.relative_to(SIM_ROOT)}:{i}: {line.strip()}")
    assert offenders == [], "\n".join(offenders)
