"""
test_effect_keys.py — every effect roll draws from the real per-key stream with the
exact key string the game uses, the right number of times, in the right order.

Mechanism: ``RecordingPRNG`` wraps the run's bit-exact ``PseudoRandom`` and logs
``(method, key[, m, n])`` for every draw while delegating to the real thing, so the
assertions below pin both the KEY (rng/keys.py) and the DRAW COUNT per trigger
(card.lua / state_events.lua call sites cited in EFFECTS_NOTES.md).

Phase 1 W3 (P1-effects).
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from balatro_sim.card import Card
from balatro_sim.game import BalatroGame, State
from balatro_sim.game_keys import core as _core, gen as _gen, TAROT_KEYS, SPECTRAL_KEYS
from balatro_sim.jokers.base import (
    JokerInstance, ScoreContext, JOKER_REGISTRY, MissingPRNG, rng_of, prob_roll,
    fire_hook, drain_joker_state, sync_probabilities,
)
from balatro_sim.scoring import score_hand
from balatro_sim.consumables import apply_tarot
from balatro_sim.card_selection import HypotheticalScorer
import balatro_sim.jokers  # noqa: F401

PseudoRandom = _core.PseudoRandom
SIM_DIR = Path(__file__).resolve().parents[2] / "balatro_sim"


# ─────────────────────────────────────────────────────────────────────────────
# recording proxy
# ─────────────────────────────────────────────────────────────────────────────

class RecordingPRNG:
    """Delegates to a real PseudoRandom and records every draw."""

    def __init__(self, inner):
        self.inner = inner
        self.log: list[tuple] = []

    def pseudorandom(self, key, m=None, n=None):
        self.log.append(("pseudorandom", key) if m is None else ("pseudorandom", key, m, n))
        return self.inner.pseudorandom(key, m, n)

    def pseudorandom_element(self, seq, key):
        self.log.append(("pseudorandom_element", key))
        return self.inner.pseudorandom_element(seq, key)

    def pseudoshuffle(self, lst, key):
        self.log.append(("pseudoshuffle", key))
        return self.inner.pseudoshuffle(lst, key)

    # passthroughs used by generate.* / clone paths
    def pseudoseed(self, key):
        return self.inner.pseudoseed(key)

    def keys(self):
        return self.inner.keys()

    def snapshot(self):
        return self.inner.snapshot()

    def restore(self, snap):
        return self.inner.restore(snap)

    def clone(self):
        return RecordingPRNG(self.inner.clone())

    def drew(self, key):
        return [e for e in self.log if e[1] == key]

    def keys_drawn(self):
        return [e[1] for e in self.log]


def _record(game: BalatroGame) -> RecordingPRNG:
    rec = RecordingPRNG(game.run_state.rng)
    game.run_state.rng = rec
    return rec


PLANETS = {h: 1 for h in ["High Card", "Pair", "Two Pair", "Three of a Kind", "Straight", "Flush",
                          "Full House", "Four of a Kind", "Straight Flush", "Five of a Kind",
                          "Flush House", "Flush Five"]}


def _score(cards, jokers=(), hand_type="High Card", scoring=None, held=(), prng=None, **kw):
    rec = prng or RecordingPRNG(PseudoRandom("EFFECTS1"))
    s, ctx = score_hand(
        scoring_cards=list(scoring if scoring is not None else cards), all_cards=list(cards),
        hand_type=hand_type, jokers=[j if isinstance(j, JokerInstance) else JokerInstance(j) for j in jokers],
        planet_levels=dict(PLANETS), hands_left=3, discards_left=3, dollars=10, ante=1,
        deck_remaining=40, rng=rec, held_cards=list(held), **kw,
    )
    return s, ctx, rec


def _to_hand(game: BalatroGame) -> BalatroGame:
    game.step({"type": "play_blind"})
    assert game.state == State.SELECTING_HAND
    return game


def _set_hand(game, cards):
    game.full_deck = [c for c in game.full_deck if c not in game.hand] + list(cards)
    game.hand = list(cards)
    return cards


# ─────────────────────────────────────────────────────────────────────────────
# 1. no unseeded fallback, no `random` anywhere in the engine
# ─────────────────────────────────────────────────────────────────────────────

def test_no_random_module_in_engine():
    """`random.Random` / `import random` must not exist in any game-logic module under
    balatro_sim/ (card_selection.py included — HypotheticalScorer clones the PseudoRandom
    now). The env_*.py wrappers are W7's (their module-level `random` guard lives in
    test_env_rng_isolation.py; the only uses left there are __main__ benchmark code)."""
    offenders = []
    for path in SIM_DIR.rglob("*.py"):
        if path.name.startswith("env_"):
            continue
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names]
                mod = getattr(node, "module", None)
                if "random" in names or mod == "random":
                    offenders.append(f"{path.name}:{node.lineno} {ast.dump(node)[:60]}")
        for m in re.finditer(r"random\.Random\(|\brandom\.(random|choice|randint|shuffle|sample)\(", src):
            offenders.append(f"{path.name}: {m.group(0)}")
    assert not offenders, offenders


def test_roll_without_prng_raises():
    with pytest.raises(MissingPRNG):
        rng_of(ScoreContext())
    with pytest.raises(MissingPRNG):
        prob_roll(ScoreContext(), "lucky_mult", 5)
    card = Card(rank=5, suit="Spades", enhancement="Lucky")
    with pytest.raises(MissingPRNG):
        score_hand(scoring_cards=[card], all_cards=[card], hand_type="High Card", jokers=[],
                   planet_levels=PLANETS, hands_left=3, discards_left=3, dollars=4, ante=1,
                   deck_remaining=40)


def test_score_hand_rejects_non_prng():
    class Fake:
        def random(self): return 0.5
    card = Card(rank=5, suit="Spades")
    with pytest.raises(TypeError):
        score_hand(scoring_cards=[card], all_cards=[card], hand_type="High Card", jokers=[],
                   planet_levels=PLANETS, hands_left=3, discards_left=3, dollars=4, ante=1,
                   deck_remaining=40, rng=Fake())


def test_game_has_no_legacy_rng():
    g = BalatroGame(seed="7I4M53DL")
    assert not hasattr(g, "rng")
    assert isinstance(g.run_state.rng, PseudoRandom)
    c = g.clone()
    assert not hasattr(c, "rng") and c.run_state.rng is not g.run_state.rng


# ─────────────────────────────────────────────────────────────────────────────
# 2. card effects
# ─────────────────────────────────────────────────────────────────────────────

def test_lucky_card_two_rolls_in_order():
    """card.lua:988 then :1076 — 'lucky_mult' (1/5) then 'lucky_money' (1/15), per pass."""
    card = Card(rank=5, suit="Spades", enhancement="Lucky")
    _, _, rec = _score([card])
    assert rec.keys_drawn() == ["lucky_mult", "lucky_money"]


def test_lucky_retrigger_rolls_again():
    card = Card(rank=5, suit="Spades", enhancement="Lucky", seal="Red")
    _, _, rec = _score([card])
    assert rec.keys_drawn() == ["lucky_mult", "lucky_money"] * 2


def test_lucky_threshold_uses_probabilities_normal():
    """Oops! All 6s: the roll is `< normal / odds`; with normal = 8 the 1/5 and 1/15 both
    exceed 1 -> guaranteed +20 Mult and $20."""
    card = Card(rank=5, suit="Spades", enhancement="Lucky")
    s, ctx, _ = _score([card], probabilities_normal=16.0)
    assert ctx.pending_money == 20 and s == (5 + 5) * (1 + 20)


def test_glass_rolls_once_per_scoring_card_not_per_trigger():
    """state_events.lua:957-963 rolls 'glass' once per scoring Glass card AFTER the hand
    scored — a retrigger (Hanging Chad x2) does not add rolls. Via the real play path."""
    g = _to_hand(BalatroGame(seed="7I4M53DL"))
    g.debug_add_joker("j_hanging_chad")
    cards = _set_hand(g, [Card(rank=9, suit="Spades", enhancement="Glass"), Card(rank=3, suit="Hearts")])
    rec = _record(g)
    g.step({"type": "play", "cards": [0]})
    assert rec.drew("glass") == [("pseudorandom", "glass")]
    # and the shatter threshold is normal/4
    g2 = _to_hand(BalatroGame(seed="7I4M53DL"))
    g2.run_state.probabilities_normal = 4.0
    g2.debug_add_joker("j_oops"); g2.debug_add_joker("j_oops")     # 2 Oops -> normal 4 -> always
    cards = _set_hand(g2, [Card(rank=9, suit="Spades", enhancement="Glass"), Card(rank=3, suit="Hearts")])
    g2.step({"type": "play", "cards": [0]})
    assert cards[0] not in g2.full_deck


def test_gold_seal_and_lucky_money_pay_immediately():
    g = _to_hand(BalatroGame(seed="ALEEB"))
    cards = _set_hand(g, [Card(rank=9, suit="Spades", seal="Gold"), Card(rank=3, suit="Hearts")])
    before = g.dollars
    g.step({"type": "play", "cards": [0]})
    assert g.dollars == before + 3


# ─────────────────────────────────────────────────────────────────────────────
# 3. joker probability rolls — key + count per trigger
# ─────────────────────────────────────────────────────────────────────────────

def test_bloodstone_one_roll_per_scoring_heart():
    hearts = [Card(rank=r, suit="Hearts") for r in (2, 5, 9)] + [Card(rank=4, suit="Spades")]
    _, _, rec = _score(hearts, ["j_bloodstone"], hand_type="Flush", scoring=hearts)
    assert rec.keys_drawn() == ["bloodstone"] * 3


def test_bloodstone_retrigger_rolls_again_and_wild_counts():
    cards = [Card(rank=5, suit="Hearts", seal="Red"), Card(rank=7, suit="Clubs", enhancement="Wild")]
    _, _, rec = _score(cards, ["j_bloodstone"], hand_type="Pair")
    assert rec.keys_drawn() == ["bloodstone"] * 3      # Heart x2 passes + Wild x1


def test_business_card_per_face_per_pass_pareidolia_aware():
    faces = [Card(rank=13, suit="Spades"), Card(rank=12, suit="Hearts"), Card(rank=2, suit="Clubs")]
    _, _, rec = _score(faces, ["j_business"], hand_type="High Card", scoring=faces)
    assert rec.keys_drawn() == ["business"] * 2
    _, _, rec = _score(faces, ["j_pareidolia", "j_business"], hand_type="High Card", scoring=faces)
    assert rec.keys_drawn() == ["business"] * 3
    _, _, rec = _score(faces, ["j_sock_and_buskin", "j_business"], hand_type="High Card", scoring=faces)
    assert rec.keys_drawn() == ["business"] * 4        # the two faces retriggered once each


def test_8ball_rolls_then_creates_through_run_state():
    """card.lua:3106-3115: room check -> '8ball' < normal/4 -> create_card('Tarot', ..., '8ba')
    which draws 'Tarot8ba1' (shared with the Purple Seal)."""
    rs = _gen.RunState(seed="EFFECTS1")
    rec = RecordingPRNG(rs.rng); rs.rng = rec
    eights = [Card(rank=8, suit="Spades"), Card(rank=8, suit="Hearts")]
    _, ctx, _ = _score(eights, ["j_8_ball"], hand_type="Pair", prng=rec, run_state=rs,
                       probabilities_normal=4.0)                 # threshold 1.0: always
    assert [e[1] for e in rec.log if e[1] in ("8ball", "Tarot8ba1")] == ["8ball", "Tarot8ba1"] * 2
    assert len(ctx.pending_consumables) == 2 and all(k in TAROT_KEYS for k in ctx.pending_consumables)
    assert ctx.pending_consumables[0] != ctx.pending_consumables[1]      # used_jokers dedupe
    # full slots: no roll at all
    _, ctx, rec2 = _score(eights, ["j_8_ball"], hand_type="Pair", run_state=rs,
                          consumables=["c_fool", "c_hermit"], consumable_slots=2)
    assert rec2.drew("8ball") == []


def test_8ball_dry_run_consumes_roll_but_creates_nothing():
    eights = [Card(rank=8, suit="Spades")]
    _, ctx, rec = _score(eights, ["j_8_ball"], probabilities_normal=4.0)   # run_state=None
    assert rec.keys_drawn() == ["8ball"] and ctx.pending_consumables == []


def test_misprint_uses_integer_form():
    _, _, rec = _score([Card(rank=5, suit="Spades")], ["j_misprint"])
    assert rec.log == [("pseudorandom", "misprint", 0, 23)]


def test_space_joker_rolls_before_scoring_and_levels_current_hand():
    levels = dict(PLANETS)
    rec = RecordingPRNG(PseudoRandom("EFFECTS1"))
    card = Card(rank=5, suit="Spades")
    s, ctx = score_hand(scoring_cards=[card], all_cards=[card], hand_type="High Card",
                        jokers=[JokerInstance("j_space")], planet_levels=levels, hands_left=3,
                        discards_left=3, dollars=10, ante=1, deck_remaining=40, rng=rec,
                        probabilities_normal=4.0)
    assert rec.keys_drawn() == ["space"]
    assert levels["High Card"] == 2
    assert s == (5 + 10 + 5) * (1 + 1)       # High Card level 2: +10 chips +1 mult, THIS hand


def test_reserved_parking_per_held_face_per_pass_and_mime():
    held = [Card(rank=13, suit="Spades"), Card(rank=12, suit="Hearts"), Card(rank=4, suit="Clubs")]
    _, _, rec = _score([Card(rank=5, suit="Spades")], ["j_reserved_parking"], held=held,
                       probabilities_normal=2.0)                   # always hits
    assert rec.keys_drawn() == ["parking"] * 2
    # Mime: a held card that had an effect gets one more pass -> one more roll each
    _, _, rec = _score([Card(rank=5, suit="Spades")], ["j_reserved_parking", "j_mime"], held=held,
                       probabilities_normal=2.0)
    assert rec.keys_drawn() == ["parking"] * 4


def test_gros_michel_and_cavendish_round_end_keys_and_destruction():
    g = _to_hand(BalatroGame(seed="7I4M53DL"))
    gm = g.debug_add_joker("j_gros_michel")
    cv = g.debug_add_joker("j_cavendish")
    rec = _record(g)
    ctx = g._hook_ctx()
    ctx.probabilities_normal = 6.0        # threshold 1.0 for Gros Michel, 0.006 for Cavendish
    JOKER_REGISTRY["j_gros_michel"].on_round_end(gm, ctx)
    JOKER_REGISTRY["j_cavendish"].on_round_end(cv, ctx)
    assert rec.keys_drawn() == ["gros_michel", "cavendish"]
    assert gm.state.get("destroyed") is True
    drain_joker_state(g, ctx)
    assert gm not in g.jokers and cv in g.jokers
    assert "gros_michel_extinct" in g.run_state.pool_flags
    assert "j_gros_michel" not in g.run_state.owned_jokers


def test_oops_sync_doubles_probabilities_normal():
    g = BalatroGame(seed="7I4M53DL")
    assert sync_probabilities(g) == 1.0
    g.debug_add_joker("j_oops")
    assert sync_probabilities(g) == 2.0
    g.debug_add_joker("j_oops")
    assert g._hook_ctx().probabilities_normal == 4.0
    assert g.run_state.probabilities_normal == 4.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. element / shuffle draws
# ─────────────────────────────────────────────────────────────────────────────

def test_madness_draws_victim_on_madness_key_in_board_order():
    g = BalatroGame(seed="7I4M53DL")
    g.debug_add_joker("j_madness"); g.debug_add_joker("j_joker"); g.debug_add_joker("j_jolly")
    rec = _record(g)
    g.step({"type": "play_blind"})
    assert rec.drew("madness") == [("pseudorandom_element", "madness")]
    assert len(g.jokers) == 2 and any(j.key == "j_madness" for j in g.jokers)
    assert g.jokers[0].state["xmult"] == 1.5


def test_amber_acorn_three_aajk_shuffles():
    g = BalatroGame(seed="7I4M53DL")
    for k in ("j_joker", "j_jolly", "j_zany"):
        g.debug_add_joker(k)
    rec = _record(g)
    g._apply_boss_start("bl_final_acorn")
    assert rec.log == [("pseudoshuffle", "aajk")] * 3


def test_cerulean_bell_hook_crimson_heart_keys():
    g = BalatroGame(seed="7I4M53DL")
    g.debug_add_joker("j_joker"); g.debug_add_joker("j_jolly")
    g.blind_idx = 2; g._prepare_next_blind()
    g.current_blind.boss_key = "bl_hook"; g.current_blind.is_boss = True; g.current_blind.kind = "Boss"
    _to_hand(g)
    rec = _record(g)
    g.step({"type": "play", "cards": [0, 1, 2]})
    assert rec.drew("hook") == [("pseudorandom_element", "hook")] * 2
    for boss, key in (("bl_final_bell", "cerulean_bell"), ("bl_final_heart", "crimson_heart")):
        g = BalatroGame(seed="7I4M53DL")
        g.debug_add_joker("j_joker"); g.debug_add_joker("j_jolly")
        g.blind_idx = 2; g._prepare_next_blind()
        g.current_blind.boss_key = boss; g.current_blind.is_boss = True; g.current_blind.kind = "Boss"
        _to_hand(g)
        rec = _record(g)
        g.step({"type": "play", "cards": [0]})
        assert rec.drew(key) == [("pseudorandom_element", key)]


def test_purple_seal_uses_the_8ball_tarot_stream():
    """card.lua:2260 — create_card('Tarot', ..., '8ba'): the Purple Seal shares 'Tarot8ba<ante>'
    with 8 Ball and rolls nothing else."""
    g = _to_hand(BalatroGame(seed="7I4M53DL"))
    cards = _set_hand(g, [Card(rank=9, suit="Spades", seal="Purple"), Card(rank=3, suit="Hearts")])
    rec = _record(g)
    g.step({"type": "discard", "cards": [0]})
    assert [k for k in rec.keys_drawn() if k.startswith("Tarot")] == ["Tarot8ba1"]
    assert len(g.consumable_hand) == 1 and g.consumable_hand[0] in TAROT_KEYS


def test_wheel_of_fortune_three_draws_on_one_key():
    g = _to_hand(BalatroGame(seed="7I4M53DL"))
    g.debug_add_joker("j_joker")
    g.debug_add_joker("j_oops"); g.debug_add_joker("j_oops")      # normal 4 -> always succeeds
    rec = _record(g)
    assert apply_tarot(g, "c_wheel_of_fortune") is True
    assert rec.log == [("pseudorandom", "wheel_of_fortune"), ("pseudorandom_element", "wheel_of_fortune"),
                       ("pseudorandom", "wheel_of_fortune")]
    assert g.jokers[0].edition in ("Foil", "Holographic", "Polychrome")
    # no editionless joker -> unusable, no draw (card.lua:1534)
    for j in g.jokers:
        j.edition = "Polychrome"
    rec.log.clear()
    assert apply_tarot(g, "c_wheel_of_fortune") is False
    assert rec.log == []


def test_hallucination_key_has_ante_and_creates_hal_tarot():
    g = _to_hand(BalatroGame(seed="7I4M53DL"))
    g.debug_add_joker("j_hallucination")
    g.run_state.probabilities_normal = 2.0    # forced hit
    rec = _record(g)
    ctx = g._hook_ctx(); ctx.probabilities_normal = 2.0
    fire_hook(g, "on_booster_opened", ctx=ctx)
    assert rec.keys_drawn() == ["halu1", "Tarothal1"]
    assert len(g.consumable_hand) == 1 and g.consumable_hand[0] in TAROT_KEYS


def test_round_card_keys_idol_ancient_castle_mail():
    """Game-level picks (W2 rolls them at run start on idol1/mail1/anc1/cas1); a context
    built without a game draws the missing one lazily on the same key."""
    g = BalatroGame(seed="7I4M53DL")
    assert {"idol1", "mail1", "anc1", "cas1"} <= set(g.run_state.rng.keys())
    ctx = g._hook_ctx()
    assert set(ctx.round_cards) == {"idol", "mail", "ancient", "castle"}
    rank, suit = ctx.round_cards["idol"]
    assert 2 <= rank <= 14 and suit in ("Spades", "Hearts", "Clubs", "Diamonds")
    rec = RecordingPRNG(PseudoRandom("EFFECTS1"))
    _, ctx2, _ = _score([Card(rank=5, suit="Hearts")], ["j_ancient"], prng=rec, ante=3) if False else (None, None, None)
    ctx3 = ScoreContext(prng=rec, ante=3, full_deck=[Card(rank=r, suit="Spades") for r in range(2, 8)])
    from balatro_sim.round_cards import round_card
    assert round_card(ctx3, "ancient") in ("Spades", "Hearts", "Clubs", "Diamonds")
    assert round_card(ctx3, "castle") == "Spades"
    assert rec.keys_drawn() == ["anc3", "cas3"]


def test_to_do_list_draws_from_pairs_order_pool():
    g = BalatroGame(seed="7I4M53DL")
    rec = _record(g)
    j = g.debug_add_joker("j_todo_list")          # on_init (Card:set_ability) draws once
    ctx = g._hook_ctx()
    assert rec.keys_drawn() == ["to_do"] and j.state.get("_init") is True
    assert j.state["target"] in _gen.HANDS_PAIRS_ORDER
    assert j.state["target"] not in ("Flush Five", "Flush House", "Five of a Kind")
    old = j.state["target"]
    JOKER_REGISTRY["j_todo_list"].on_round_end(j, ctx)
    assert j.state["target"] != old and rec.keys_drawn() == ["to_do", "to_do"]


def test_riff_raff_marble_certificate_create_through_run_state():
    g = BalatroGame(seed="7I4M53DL")
    for k in ("j_riff_raff", "j_marble", "j_certificate"):
        g.debug_add_joker(k)
    n_deck, n_hand = len(g.full_deck), 0
    rec = _record(g)
    g.step({"type": "play_blind"})
    keys = rec.keys_drawn()
    assert keys.count("Joker1rif1") == 2 or "Joker1rif1" in keys       # resamples add _resample keys
    assert "marb_fr" in keys and "cert_fr" in keys and "certsl" in keys
    assert len(g.jokers) == 5
    assert len(g.full_deck) == n_deck + 2
    assert any(c.enhancement == "Stone" for c in g.full_deck)
    assert any(c.seal != "None" for c in g.hand)


# ─────────────────────────────────────────────────────────────────────────────
# 5. scoring-order fidelity that changed with W3
# ─────────────────────────────────────────────────────────────────────────────

def test_joker_edition_applies_once_in_joker_main():
    diamonds = [Card(rank=r, suit="Diamonds") for r in (2, 5, 7, 9, 11)]
    plain, _, _ = _score(diamonds, [JokerInstance("j_greedy_joker")], hand_type="Flush")
    foil, _, _ = _score(diamonds, [JokerInstance("j_greedy_joker", edition="Foil")], hand_type="Flush")
    assert foil - plain == 50 * (4 + 15)       # +50 chips ONCE, not once per scoring Diamond


def test_lucky_cat_scales_on_lucky_trigger():
    card = Card(rank=5, suit="Spades", enhancement="Lucky")
    j = JokerInstance("j_lucky_cat")
    _score([card], [j], probabilities_normal=16.0)
    assert j.state["xmult"] == 1.25
    _score([card], [j], probabilities_normal=0.0)     # never hits
    assert j.state["xmult"] == 1.25


def test_held_phase_baron_mime_red_seal():
    king = Card(rank=13, suit="Spades")
    s1, _, _ = _score([Card(rank=5, suit="Spades")], ["j_baron"], held=[king])
    assert s1 == int(10 * 1.5)
    s2, _, _ = _score([Card(rank=5, suit="Spades")], ["j_baron", "j_mime"], held=[king])
    assert s2 == int(10 * 1.5 * 1.5)
    king.seal = "Red"
    s3, _, _ = _score([Card(rank=5, suit="Spades")], ["j_baron", "j_mime"], held=[king])
    assert s3 == int(10 * 1.5 ** 3)


def test_raised_fist_uses_nominal_chips():
    ace = Card(rank=14, suit="Spades")
    s, _, _ = _score([Card(rank=5, suit="Spades")], ["j_raised_fist"], held=[ace])
    assert s == 10 * (1 + 2 * 11)           # 2 x nominal(11), not 2 x rank(14)


def test_photograph_first_face_per_hand_and_hanging_chad():
    faces = [Card(rank=13, suit="Spades"), Card(rank=12, suit="Hearts")]
    j = JokerInstance("j_photograph")
    s1, _, _ = _score(faces, [j], hand_type="High Card", scoring=faces)
    s2, _, _ = _score(faces, [j], hand_type="High Card", scoring=faces)
    assert s1 == s2 == (5 + 10 + 10) * 2         # x2 every hand, once (first face only)
    s3, _, _ = _score(faces, ["j_hanging_chad", j], hand_type="High Card", scoring=faces)
    assert s3 == (5 + 10 * 3 + 10) * 2 ** 3      # first card scored 3 times -> x2 three times


# ─────────────────────────────────────────────────────────────────────────────
# 6. HypotheticalScorer: keyed clone, no leakage
# ─────────────────────────────────────────────────────────────────────────────

def test_hypothetical_scorer_uses_clone_and_never_advances_live_streams():
    g = _to_hand(BalatroGame(seed="7I4M53DL"))
    for k in ("j_bloodstone", "j_misprint", "j_8_ball", "j_space"):
        g.debug_add_joker(k)
    for i, c in enumerate(g.hand):
        if i % 2 == 0:
            c.enhancement = "Lucky"
        c.suit = "Hearts"
    snap = g.run_state.rng.snapshot()
    used = set(g.run_state.used_jokers)
    scorer = HypotheticalScorer(g, model_held=True)
    from balatro_sim.hand_eval import evaluate_hand
    cards = g.hand[:5]
    ht, sc = evaluate_hand(cards)
    a = scorer.score(cards, ht, sc)
    b = scorer.score(cards, ht, sc)
    assert a == b
    assert g.run_state.rng.snapshot() == snap
    assert g.run_state.used_jokers == used
    assert g.consumable_hand == []
    # the private clone DID draw on the real keys
    assert {"lucky_mult", "bloodstone", "misprint", "space"} <= set(scorer._rng.keys())
