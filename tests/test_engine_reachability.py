"""Reachability probe: every joker, consumable and voucher in ``rng/pools.py`` must produce a
MEASURABLE state change in the engine when exercised in a minimal state where it should trigger.

Generalises the probe that caught A1 ("28% of episodes used a tarot, 0% ended with a modified
card"): a unit test can bless an effect that the game path never delivers, a reachability probe
cannot.  Each item gets a tiny scripted *scenario* (never a full episode) run twice -- WITH and
WITHOUT the item -- on a real ``BalatroGame`` through ``step()``; the two resulting state
signatures (chips scored, dollars, hand levels, deck composition, consumable slots, other jokers,
hand/discard counts, hand size, slots, tags, ...) must differ.  For items whose effect lives in
their own state (Egg's sell value, Flash's mult), the scenario may instead name a joker-state key
that must change before/after.

Also enforced on the WITH run: no sentinel token may appear in a consumable slot (the nine
``pending_consumables`` producers that emit ``"tarot"`` / ``"double_tag"`` / ... are dead, not
reachable), and a consumable that was used must actually leave the slot.

Items for which no measurable change can be defined are ``xfail``ed with the reason; that list
is itself a finding (see HARNESS_NOTES.md).  Items whose game key the engine does not register
yet are ``xfail``ed at runtime with "not registered" (W1 re-key) -- they turn into real probes
the moment the key exists.

Runtime budget: < 2 min total (tiny states; probabilistic items retry over seeds).

Run:  python -m pytest tests/test_engine_reachability.py -q
      python -m pytest tests/test_engine_reachability.py -q -rx   # list the xfail reasons
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import pytest

HERE = Path(__file__).resolve().parent
MP_ROOT = HERE.parent
if str(MP_ROOT) not in sys.path:
    sys.path.insert(0, str(MP_ROOT))

from oracle import engine_parity as EP  # noqa: E402
from rng import pools as P  # noqa: E402

GM = EP.import_engine()
BalatroGame, State = GM.BalatroGame, GM.State
from balatro_sim.card import Card  # noqa: E402
from balatro_sim.jokers.base import JokerInstance  # noqa: E402
from balatro_sim import consumables as CONS  # noqa: E402
from balatro_sim import shop as SHOP  # noqa: E402

SEEDS = ["7I4M53DL", "ALEEB", "11111111", "1558AXDL", "15H9Z3IY", "1KV4W6YS", "1MD1YZ9T"]
ALL_CONSUMABLE_KEYS = {c["key"] for c in P.TAROTS + P.PLANETS + P.SPECTRALS}
KNOWN_KEYS = EP.GEN_KEYS


# =============================================================================================
# card / scenario DSL
# =============================================================================================

def C(rank, suit, enh="None", seal="None", ed="None"):
    """Card spec (built fresh per run so mutations never leak between runs)."""
    return (rank, suit, enh, seal, ed)


def _mk(spec) -> Card:
    r, s, enh, seal, ed = spec
    return Card(rank=r, suit=s, enhancement=enh, seal=seal, edition=ed)


S, H, D, Cl = "Spades", "Hearts", "Diamonds", "Clubs"
PAIR = [C(14, S), C(14, H), C(5, Cl), C(7, D), C(9, S)]
TRIPS = [C(14, S), C(14, H), C(14, Cl), C(7, D), C(9, S)]
TWO_PAIR = [C(14, S), C(14, H), C(7, Cl), C(7, D), C(9, S)]
STRAIGHT = [C(5, S), C(6, H), C(7, Cl), C(8, D), C(9, S)]
FLUSH_H = [C(2, H), C(5, H), C(7, H), C(9, H), C(11, H)]
FLUSH_S = [C(2, S), C(5, S), C(7, S), C(9, S), C(11, S)]
FLUSH_D = [C(2, D), C(5, D), C(7, D), C(9, D), C(11, D)]
FLUSH_C = [C(2, Cl), C(5, Cl), C(7, Cl), C(9, Cl), C(11, Cl)]
QUADS = [C(14, S), C(14, H), C(14, Cl), C(14, D), C(9, S)]
FULL_HOUSE = [C(14, S), C(14, H), C(14, Cl), C(7, D), C(7, S)]
HIGH_CARD = [C(14, S), C(3, H), C(5, Cl), C(7, D), C(9, S)]
FACES = [C(13, S), C(12, H), C(11, Cl), C(13, D), C(12, S)]
STRAIGHT_FLUSH = [C(5, H), C(6, H), C(7, H), C(8, H), C(9, H)]


@dataclass
class Scenario:
    kind: str = "score"                  # score | discard | round_end | blind_select | shop | sell | skip_then_score | use | voucher | flag
    hand: list = field(default_factory=lambda: list(PAIR))   # cards dealt (crafted) before the action
    play: Optional[list] = None          # indices to play (default: all of `hand`)
    held: list = field(default_factory=list)                 # extra cards in hand that are NOT played
    left: list = field(default_factory=list)                 # extra jokers to the left of the probe
    right: list = field(default_factory=list)                # extra jokers to the right
    joker_state: dict = field(default_factory=dict)          # pre-set state on the probe joker
    consumables: list = field(default_factory=list)          # consumable keys placed in the slots
    pre_use: list = field(default_factory=list)              # [(consumable_idx, target_indices)] used before the action
    pre_discards: list = field(default_factory=list)         # [indices] discards before the play
    dollars: Optional[int] = None
    hands_left: Optional[int] = None
    discards_left: Optional[int] = None
    hand_type_counts: Optional[dict] = None
    planets_used: list = field(default_factory=list)
    deck_mods: Optional[Callable] = None                     # fn(game) mutating full_deck before the blind
    setup: Optional[Callable] = None                         # fn(game) run after the blind starts
    boss: Optional[str] = None                               # force this boss blind
    shop_actions: list = field(default_factory=list)         # actions in the SHOP (kind=shop / sell)
    then_score: bool = False                                 # after shop/skip: play `hand` and measure
    post: Optional[str] = None                               # "next_blind" | "round_end": continue past the hand before measuring
    repeats: int = 1                                         # play the hand this many times
    hands_seq: list = field(default_factory=list)            # per-repeat hands (overrides `hand` for repeat r)
    tries: int = 1                                           # seeds to try (probabilistic effects)
    probe_state_key: Optional[str] = None                    # joker state key that must change before/after
    ignore: tuple = ()                                       # signature fields to ignore
    target_idx: list = field(default_factory=list)           # use: target indices
    flag: Optional[Callable] = None                          # kind=flag: fn(game) -> bool
    xfail: Optional[str] = None                              # no measurable change definable: reason
    note: str = ""


def deck_enhance(n: int, enh: str, suit=None):
    def f(game):
        k = 0
        for c in game.full_deck:
            if suit and c.suit != suit:
                continue
            c.enhancement = enh
            k += 1
            if k >= n:
                break
    return f


def deck_remove(n: int):
    def f(game):
        del game.full_deck[:n]
    return f


# =============================================================================================
# signature
# =============================================================================================

def _card_t(c):
    return (c.rank, c.suit, c.enhancement, c.edition, c.seal)


def signature(game, probe, ignore=()) -> dict:
    sig = {}
    for k, v in vars(game).items():
        if k in ("rng", "_forced_card_id", "_disabled_joker_idx", "_played_this_ante", "reroll_cost",
                 "free_rerolls_remaining", "state"):
            continue
        if isinstance(v, (int, float, str, bool)) or v is None:
            sig[k] = v
    sig["state"] = game.state.name
    sig["planet_levels"] = tuple(sorted(game.planet_levels.items()))
    sig["consumables"] = tuple(game.consumable_hand)
    sig["jokers"] = tuple((j.key, j.edition, j.state.get("sell_value")) for j in game.jokers if j is not probe)
    sig["n_jokers"] = len([j for j in game.jokers if j is not probe])
    sig["full_deck"] = tuple(sorted(map(_card_t, game.full_deck)))
    sig["hand"] = tuple(sorted(map(_card_t, game.hand)))
    sig["n_hand"] = len(game.hand)
    sig["shop"] = tuple((it.kind, it.key, it.price, getattr(it, "sold", False)) for it in game.current_shop)
    tags = getattr(game, "tags", None)
    if tags is not None:
        try:
            sig["tags"] = tuple(tags.keys() if hasattr(tags, "keys") and callable(tags.keys) else tags)
        except Exception:
            sig["tags"] = repr(tags)
    rs = getattr(game, "run_state", None)
    if rs is not None:
        sig["run_state"] = (rs.showman, rs.probabilities_normal, rs.edition_rate, rs.joker_rate, rs.tarot_rate,
                            rs.planet_rate, rs.playing_card_rate, rs.spectral_rate, rs.shop_joker_max, rs.ante,
                            tuple(sorted(rs.used_vouchers)))
    for k in ignore:
        sig.pop(k, None)
    return sig


def diff(a: dict, b: dict) -> list:
    return [k for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]


# =============================================================================================
# runner
# =============================================================================================

def _force_win(game):
    win = getattr(game, "debug_win_blind", None)
    if callable(win):
        win()
    else:
        game.chips_scored = game.current_blind.chips_target
        game.state = State.ROUND_EVAL
    if game.state == State.ROUND_EVAL:
        game.step({"type": "advance"})


def _set_hand(game, specs):
    cards = [_mk(s) for s in specs]
    game.full_deck = [c for c in game.full_deck if c not in game.hand] + cards
    game.hand = list(cards)          # the engine mutates game.hand in place; keep our own list
    return cards


def _add_joker(game, key, state=None):
    fn = getattr(game, "debug_add_joker", None)
    j = fn(key) if callable(fn) else None
    if j is None:
        j = JokerInstance(key)
        game.jokers.append(j)
        rs = getattr(game, "run_state", None)
        if rs is not None:
            rs.acquire(key)
    eff = getattr(game, "_joker_effect", None)
    if state:
        j.state.update(state)
    return j


def run_scenario(sc: Scenario, item_key: Optional[str], seed: str, item_kind: str):
    """Execute one scenario.  ``item_key`` None = baseline run.  Returns (signature, probe, before_state)."""
    g = BalatroGame(seed=seed)
    probe = None
    before_state = None
    if sc.dollars is not None:
        g.dollars = sc.dollars
    if sc.deck_mods:
        sc.deck_mods(g)
    for k in sc.left:
        _add_joker(g, k)
    if item_kind == "joker" and item_key:
        probe = _add_joker(g, item_key, sc.joker_state)
        if sc.probe_state_key:
            before_state = repr(probe.state.get(sc.probe_state_key))
    for k in sc.right:
        _add_joker(g, k)
    if sc.hand_type_counts:
        g._hand_type_counts = dict(sc.hand_type_counts)
    if sc.planets_used:
        g.planets_used = list(sc.planets_used)
    if sc.boss:
        g.blind_idx = 2
        g._prepare_next_blind()
        g.current_blind.boss_key = sc.boss
        g.current_blind.is_boss = True
        g.current_blind.kind = "Boss"

    # -- kinds that act from BLIND_SELECT ---------------------------------------------------
    if sc.kind == "skip_then_score":
        # W2: skipping grants the blind's tag and moves straight to the next blind (real
        # game: no shop after a skip); a pack tag may interrupt with BOOSTER_OPEN.
        g.step({"type": "skip_blind"})
        if g.state == State.BOOSTER_OPEN:
            g.step({"type": "skip_booster"})
        assert g.state == State.BLIND_SELECT, g.state
        g.dollars = 10 ** 4

    if sc.kind == "voucher":
        # reach the shop, inject the voucher item, buy it
        g.step({"type": "play_blind"})
        _force_win(g)
        g.dollars = 10 ** 4
        if sc.setup:
            sc.setup(g)
        if item_key:
            g.current_shop.insert(0, SHOP.ShopItem("voucher", item_key, item_key, 10))
            g.step({"type": "buy", "item_idx": 0})
            g.current_shop = [it for it in g.current_shop if not (it.kind == "voucher" and it.key == item_key)]
        g.step({"type": "leave_shop"})
        g.step({"type": "play_blind"})
        return signature(g, probe, sc.ignore + ("vouchers", "dollars", "shop")), probe, before_state

    g.step({"type": "play_blind"})
    assert g.state == State.SELECTING_HAND, g.state
    if sc.kind == "blind_select":
        if sc.setup:
            sc.setup(g)
        return signature(g, probe, sc.ignore), probe, before_state

    # -- kinds that go through the shop first -----------------------------------------------
    if sc.kind in ("shop", "sell"):
        _force_win(g)
        assert g.state == State.SHOP, g.state
        g.dollars = 10 ** 4 if sc.dollars is None else sc.dollars
        if sc.setup:
            sc.setup(g)
        if sc.kind == "sell" and probe is not None:
            idx = g.jokers.index(probe)
            g.step({"type": "sell_joker", "joker_idx": idx})
        for act in sc.shop_actions:
            if act.get("type") == "buy_first_booster":
                packs = [i for i, it in enumerate(g.current_shop) if it.kind == "booster" and not it.sold]
                if packs:
                    g.step({"type": "buy", "item_idx": packs[0]})
            elif act.get("type") == "sell_other":
                others = [i for i, j in enumerate(g.jokers) if j is not probe]
                if others:
                    g.step({"type": "sell_joker", "joker_idx": others[0]})
            elif act.get("type") == "inject_item":
                g.current_shop.insert(0, SHOP.ShopItem(act["kind"], act["key"], act["key"], act["price"]))
            else:
                g.step(act)
        if not sc.then_score:
            return signature(g, probe, sc.ignore), probe, before_state
        if g.state == State.BOOSTER_OPEN:
            g.step({"type": "skip_booster"})
        g.step({"type": "leave_shop"})
        g.step({"type": "play_blind"})
        assert g.state == State.SELECTING_HAND, g.state

    # -- in SELECTING_HAND --------------------------------------------------------------------
    if sc.hands_left is not None:
        g.hands_left = sc.hands_left
    if sc.discards_left is not None:
        g.discards_left = sc.discards_left
    if sc.dollars is not None:
        g.dollars = sc.dollars
    if sc.setup:
        sc.setup(g)
    g.consumable_hand = list(sc.consumables)
    if sc.kind == "use" and item_key:
        g.consumable_hand.append(item_key)
    cards = _set_hand(g, sc.hand + sc.held)
    for ci, targets in sc.pre_use:
        g.step({"type": "use_consumable", "consumable_idx": ci, "target_cards": targets})
    for idx in sc.pre_discards:
        g.step({"type": "discard", "cards": idx})
        cards = _set_hand(g, sc.hand + sc.held)

    if sc.kind == "use":
        if item_key:
            ci = len(g.consumable_hand) - 1
            g.step({"type": "use_consumable", "consumable_idx": ci, "target_cards": list(sc.target_idx)})
            assert item_key not in g.consumable_hand, f"{item_key} was not consumed (unknown key or apply returned False)"
        return signature(g, probe, sc.ignore + ("dollars",) if "dollars" in sc.ignore else sc.ignore), probe, before_state

    if sc.kind == "discard":
        g.step({"type": "discard", "cards": sc.play if sc.play is not None else list(range(len(sc.hand)))})
        return signature(g, probe, sc.ignore), probe, before_state

    if sc.kind == "round_end":
        _force_win(g)
        return signature(g, probe, sc.ignore), probe, before_state

    # score (possibly repeated)
    play_idx = sc.play if sc.play is not None else list(range(len(sc.hand)))
    scores = []
    for r in range(sc.repeats):
        if r:
            g.state = State.SELECTING_HAND
            g.hands_left = max(g.hands_left, 1)
            g.hand = [c for c in cards if c in g.full_deck]
            g.discard_pile = []
        if sc.hands_seq:
            cards = _set_hand(g, sc.hands_seq[r % len(sc.hands_seq)] + sc.held)
            play_idx = list(range(len(sc.hands_seq[r % len(sc.hands_seq)])))
        before = g.chips_scored
        g.step({"type": "play", "cards": play_idx})
        scores.append(g.chips_scored - before)
    sig = signature(g, probe, sc.ignore)
    sig["scores"] = tuple(scores)
    if sc.post == "round_end":
        _force_win(g)
        sig = signature(g, probe, sc.ignore)
        sig["scores"] = tuple(scores)
    if sc.post == "next_blind":
        _force_win(g)
        g.dollars = 10 ** 4
        g.step({"type": "leave_shop"})
        g.step({"type": "play_blind"})
        sig = signature(g, probe, sc.ignore + ("scores",))
    return sig, probe, before_state


def probe_item(item_key: str, item_kind: str, sc: Scenario):
    if sc.xfail:
        pytest.xfail(f"{item_key}: no measurable change defined -- {sc.xfail}")
    if sc.kind == "flag":
        g = BalatroGame(seed=SEEDS[0])
        _add_joker(g, item_key)
        try:
            ok = sc.flag(g)
        except AttributeError as ex:
            pytest.xfail(f"{item_key}: engine hook missing ({ex})")
        assert ok, f"{item_key}: {sc.note}"
        return
    last = None
    for seed in SEEDS[:sc.tries]:
        with_sig, probe, before_state = run_scenario(sc, item_key, seed, item_kind)
        # sentinel tokens in consumable slots = a dead producer, not a reachable effect
        bad = [c for c in with_sig.get("consumables", ()) if c not in KNOWN_KEYS]
        assert not bad, f"{item_key}: sentinel token(s) in consumable slots: {bad}"
        base_sig, _, _ = run_scenario(sc, None, seed, item_kind)
        changed = diff(with_sig, base_sig)
        if item_kind == "joker" and sc.probe_state_key and probe is not None:
            if repr(probe.state.get(sc.probe_state_key)) != before_state:
                changed.append(f"joker_state[{sc.probe_state_key}]")
        if changed:
            return
        last = (seed, with_sig)
    pytest.fail(f"{item_key}: no measurable state change in {sc.tries} seed(s) (kind={sc.kind}; {sc.note})")


# =============================================================================================
# JOKER SCENARIOS (150, keyed by the game's own key)
# =============================================================================================

def flush_of(suit):
    return [C(2, suit), C(5, suit), C(7, suit), C(9, suit), C(11, suit)]


def sc(**kw) -> Scenario:
    return Scenario(**kw)


JOKER_SCENARIOS: dict = {
    "j_joker": sc(),
    "j_greedy_joker": sc(hand=FLUSH_D),
    "j_lusty_joker": sc(hand=FLUSH_H),
    "j_wrathful_joker": sc(hand=FLUSH_S),
    "j_gluttenous_joker": sc(hand=FLUSH_C),
    "j_jolly": sc(hand=PAIR), "j_zany": sc(hand=TRIPS), "j_mad": sc(hand=TWO_PAIR),
    "j_crazy": sc(hand=STRAIGHT), "j_droll": sc(hand=FLUSH_H),
    "j_sly": sc(hand=PAIR), "j_wily": sc(hand=TRIPS), "j_clever": sc(hand=TWO_PAIR),
    "j_devious": sc(hand=STRAIGHT), "j_crafty": sc(hand=FLUSH_H),
    "j_half": sc(hand=[C(14, S), C(14, H)]),
    "j_stencil": sc(),
    "j_four_fingers": sc(hand=[C(2, H), C(5, H), C(7, H), C(9, H)]),
    "j_mime": sc(hand=PAIR, held=[C(13, S, "Steel")]),
    "j_credit_card": sc(kind="shop", dollars=0, shop_actions=[{"type": "inject_item", "kind": "tarot", "key": "c_hermit", "price": 3},
                                                              {"type": "buy", "item_idx": 0}],
                        note="with Credit Card a $3 purchase at $0 must succeed (debt to -$20)"),
    "j_ceremonial": sc(kind="blind_select", right=["j_joker"], note="joker to the right must be destroyed"),
    "j_banner": sc(discards_left=3),
    "j_mystic_summit": sc(discards_left=0),
    "j_marble": sc(kind="blind_select", note="a Stone card must be added to the deck"),
    "j_loyalty_card": sc(repeats=7, note="every 6th hand x4"),
    "j_8_ball": sc(hand=[C(8, S), C(8, H), C(8, Cl), C(8, D), C(9, S)], tries=7, note="1/4 per 8: Tarot created"),
    "j_misprint": sc(tries=7),
    "j_dusk": sc(hands_left=1),
    "j_raised_fist": sc(hand=PAIR, held=[C(2, Cl)]),
    "j_chaos": sc(kind="shop", dollars=0, shop_actions=[{"type": "reroll"}], note="first reroll free: shelf must change at $0"),
    "j_fibonacci": sc(hand=[C(14, S), C(2, H), C(3, Cl), C(5, D), C(8, S)]),
    "j_steel_joker": sc(deck_mods=deck_enhance(5, "Steel")),
    "j_scary_face": sc(hand=FACES),
    "j_abstract": sc(),
    "j_delayed_grat": sc(kind="round_end", discards_left=3, note="$2 per unused discard when none used"),
    "j_hack": sc(hand=[C(2, S), C(2, H), C(3, Cl), C(4, D), C(9, S)]),
    "j_pareidolia": sc(right=["j_scary_face"], hand=[C(2, S), C(3, H), C(4, Cl), C(5, D), C(9, S)],
                       note="Scary Face must fire on non-faces"),
    "j_gros_michel": sc(),
    "j_even_steven": sc(hand=[C(2, S), C(4, H), C(6, Cl), C(8, D), C(10, S)]),
    "j_odd_todd": sc(hand=[C(14, S), C(3, H), C(5, Cl), C(7, D), C(9, S)]),
    "j_scholar": sc(hand=QUADS),
    "j_business": sc(hand=FACES, tries=7, post="round_end", note="$2 per face at 1/2 (engine pays joker pending_money at round end)"),
    "j_supernova": sc(repeats=2),
    "j_ride_the_bus": sc(hand=[C(2, S), C(4, H), C(6, Cl), C(8, D), C(10, S)], repeats=3),
    # 1/4 per hand on the keyed `space` stream: the FIRST draw is >= 0.25 on all 7 harness
    # seeds (0.911 0.861 0.884 0.374 0.951 0.577 0.359), so play twice (ALEEB fires on
    # the 2nd draw, 0.208) -- not an engine bug (W3/W5).
    "j_space": sc(tries=7, repeats=2, note="1/4: level up played hand"),
    "j_egg": sc(kind="round_end", probe_state_key="sell_value", ignore=("dollars",), note="sell value +$3"),
    "j_burglar": sc(kind="blind_select"),
    "j_blackboard": sc(hand=PAIR, held=[C(2, S), C(3, Cl)]),
    "j_runner": sc(hand=STRAIGHT, repeats=2),
    "j_ice_cream": sc(),
    "j_dna": sc(hand=[C(14, S)], hands_left=4, note="first hand, single card -> copy added to deck"),
    "j_splash": sc(hand=HIGH_CARD),
    "j_blue_joker": sc(),
    "j_sixth_sense": sc(hand=[C(6, S)], hands_left=4, note="single 6 on first hand -> destroyed + Spectral"),
    "j_constellation": sc(consumables=["c_mercury"], pre_use=[(0, [])], hand=PAIR),
    "j_hiker": sc(repeats=2),
    "j_faceless": sc(kind="discard", hand=FACES, note="3+ faces discarded -> $5"),
    "j_green_joker": sc(repeats=2),
    "j_superposition": sc(hand=[C(14, S), C(2, H), C(3, Cl), C(4, D), C(5, S)], note="straight with an Ace -> Tarot"),
    "j_todo_list": sc(hand=HIGH_CARD, tries=7, note="$4 when the listed hand is played (hand is seeded: retries)"),
    "j_cavendish": sc(),
    "j_card_sharp": sc(repeats=2),
    "j_red_card": sc(kind="shop", shop_actions=[{"type": "buy_first_booster"}, {"type": "skip_booster"}], then_score=True,
                     note="+3 Mult per skipped booster"),
    "j_madness": sc(kind="blind_select", right=["j_joker"], note="destroys another joker on Small/Big"),
    "j_square": sc(hand=[C(14, S), C(14, H), C(7, Cl), C(7, D)], repeats=2, note="4-card hand -> +4 chips"),
    "j_seance": sc(hand=STRAIGHT_FLUSH, note="Straight Flush -> Spectral"),
    "j_riff_raff": sc(kind="blind_select", note="2 Common jokers created"),
    "j_vampire": sc(hand=[C(14, S, "Bonus"), C(14, H, "Bonus"), C(5, Cl, "Mult"), C(7, D, "Mult"), C(9, S, "Mult")]),
    "j_shortcut": sc(hand=[C(2, S), C(4, H), C(6, Cl), C(8, D), C(10, S)]),
    "j_hologram": sc(consumables=["c_cryptid"], pre_use=[(0, [0])], hand=PAIR, note="card added to deck -> x0.25"),
    "j_vagabond": sc(dollars=3, note="hand played with <= $4 -> Tarot"),
    "j_baron": sc(hand=PAIR, held=[C(13, S), C(13, H)]),
    "j_cloud_9": sc(kind="round_end", note="$1 per 9 in deck"),
    "j_rocket": sc(kind="round_end"),
    "j_obelisk": sc(hands_seq=[PAIR, PAIR, HIGH_CARD], repeats=3, note="x0.2 per consecutive non-most-played hand"),
    "j_midas_mask": sc(hand=FACES, note="faces become Gold"),
    "j_luchador": sc(xfail="selling a joker during a Boss blind is not an engine action (sell only in SHOP); "
                           "needs sell-anytime + boss-disable hook"),
    "j_photograph": sc(hand=FACES),
    "j_gift": sc(kind="round_end", right=["j_joker"], note="+$1 sell value on the other joker"),
    "j_turtle_bean": sc(kind="blind_select", note="+5 hand size"),
    "j_erosion": sc(deck_mods=deck_remove(4)),
    "j_reserved_parking": sc(hand=PAIR, held=[C(13, S), C(12, H), C(11, Cl)], tries=7, post="round_end"),
    "j_mail": sc(kind="discard", hand=[C(2, S), C(3, H), C(4, Cl), C(5, D), C(6, S), C(7, H), C(8, Cl), C(9, D)],
                 play=list(range(8)), tries=7, note="discard of the rebate rank -> $5 (rank is seeded: retries)"),
    "j_to_the_moon": sc(dollars=25, post="round_end", note="extra interest at round end"),
    "j_hallucination": sc(kind="shop", shop_actions=[{"type": "buy_first_booster"}], tries=7, note="1/2 on pack open: Tarot"),
    "j_fortune_teller": sc(consumables=["c_hermit"], pre_use=[(0, [])], hand=PAIR),
    "j_juggler": sc(kind="blind_select"),
    "j_drunkard": sc(kind="blind_select"),
    "j_stone": sc(deck_mods=deck_enhance(5, "Stone")),
    "j_golden": sc(kind="round_end"),
    "j_lucky_cat": sc(hand=[C(14, S, "Lucky"), C(14, H, "Lucky"), C(5, Cl, "Lucky"), C(7, D, "Lucky"), C(9, S, "Lucky")],
                      repeats=2, tries=7, note="Lucky trigger -> x0.25; on_lucky_trigger has no caller"),
    "j_baseball": sc(right=["j_stencil"], note="x1.5 per Uncommon joker"),
    "j_bull": sc(dollars=10),
    "j_diet_cola": sc(kind="sell", ignore=("dollars",), note="sell -> Double Tag (needs tags)"),
    "j_trading": sc(kind="discard", hand=PAIR, play=[0], discards_left=3, note="first discard of 1 card: destroy + $3"),
    "j_flash": sc(kind="shop", shop_actions=[{"type": "reroll"}], then_score=True, note="+2 Mult per reroll"),
    "j_popcorn": sc(),
    "j_trousers": sc(hand=TWO_PAIR, repeats=2),
    "j_ancient": sc(hand=FLUSH_H, tries=7, note="x1.5 per card of the seeded suit (anc1): retries over seeds"),
    "j_ramen": sc(),
    "j_walkie_talkie": sc(hand=[C(10, S), C(4, H), C(10, Cl), C(4, D), C(9, S)]),
    "j_selzer": sc(),
    "j_castle": sc(pre_discards=[[0, 1, 2, 3, 4]], hand=FLUSH_H, discards_left=3, tries=7,
                   note="discards of the seeded suit (cas1) -> chips: retries over seeds"),
    "j_smiley": sc(hand=FACES),
    "j_campfire": sc(kind="shop", right=["j_joker"], shop_actions=[{"type": "sell_other"}], then_score=True,
                     note="x0.25 per card sold"),
    "j_ticket": sc(hand=[C(14, S, "Gold"), C(14, H, "Gold"), C(5, Cl, "Gold"), C(7, D, "Gold"), C(9, S, "Gold")], post="round_end"),
    "j_mr_bones": sc(hand=PAIR, hands_left=1, setup=lambda g: setattr(g.current_blind, "chips_target", 200),
                     note="losing hand with >= 25% -> saved (state ROUND_EVAL not GAME_OVER)"),
    "j_acrobat": sc(hands_left=1),
    "j_sock_and_buskin": sc(hand=FACES),
    "j_swashbuckler": sc(left=["j_joker"]),
    "j_troubadour": sc(kind="blind_select"),
    "j_certificate": sc(kind="blind_select", note="a sealed playing card is added"),
    "j_smeared": sc(hand=[C(2, H), C(5, D), C(7, H), C(9, D), C(11, H)]),
    "j_throwback": sc(kind="skip_then_score", note="x0.25 per skipped blind"),
    "j_hanging_chad": sc(),
    "j_rough_gem": sc(hand=FLUSH_D),
    "j_bloodstone": sc(hand=FLUSH_H, tries=7),
    "j_arrowhead": sc(hand=FLUSH_S),
    "j_onyx_agate": sc(hand=FLUSH_C),
    "j_glass": sc(hand=[C(14, S, "Glass"), C(14, H, "Glass"), C(5, Cl, "Glass"), C(7, D, "Glass"), C(9, S, "Glass")],
                  repeats=2, tries=7, note="x0.75 per Glass destroyed"),
    "j_ring_master": sc(kind="flag", flag=lambda g: g.run_state.showman, note="run_state.showman must be True when owned"),
    # Reads the SCORING hand (card.lua:3807-3840): a High Card only scores the Jack, so the
    # four suits must all be in the scoring cards -> a Straight (W5 probe fix).
    "j_flower_pot": sc(hand=STRAIGHT),
    "j_blueprint": sc(right=["j_joker"]),
    "j_wee": sc(hand=[C(2, S), C(2, H), C(5, Cl), C(7, D), C(9, S)], repeats=2),
    "j_merry_andy": sc(kind="blind_select"),
    "j_oops": sc(hand=[C(14, S, "Lucky"), C(14, H, "Lucky"), C(5, Cl, "Lucky"), C(7, D, "Lucky"), C(9, S, "Lucky")],
                 tries=7, note="doubled Lucky odds: some seed must differ from the undoubled baseline"),
    "j_idol": sc(hand=[C(14, S), C(14, H), C(14, Cl), C(14, D), C(13, S)], tries=7,
                 note="x2 when the seeded idol card (idol1) is played: retries"),
    "j_seeing_double": sc(hand=[C(14, Cl), C(14, H), C(7, Cl), C(7, D), C(9, S)]),
    "j_matador": sc(boss="bl_hook", hand=PAIR, note="Hook fires -> $8; on_boss_ability_triggered has no caller"),
    "j_hit_the_road": sc(pre_discards=[[0]], hand=[C(11, S), C(14, H), C(5, Cl), C(7, D), C(9, S)], discards_left=3,
                         note="discarded Jack -> x0.5"),
    "j_duo": sc(hand=PAIR), "j_trio": sc(hand=TRIPS), "j_family": sc(hand=QUADS),
    "j_order": sc(hand=STRAIGHT), "j_tribe": sc(hand=FLUSH_H),
    "j_stuntman": sc(),
    "j_invisible": sc(kind="sell", right=["j_joker"], joker_state={"rounds": 2}, ignore=("dollars",),
                      note="sell after 2 rounds -> duplicate a joker"),
    "j_brainstorm": sc(left=["j_joker"]),
    "j_satellite": sc(kind="round_end", consumables=["c_mercury"], pre_use=[(0, [])], note="$1 per unique Planet used"),
    "j_shoot_the_moon": sc(hand=PAIR, held=[C(12, S), C(12, H)]),
    "j_drivers_license": sc(deck_mods=deck_enhance(30, "Bonus")),
    "j_cartomancer": sc(kind="blind_select", note="Tarot created on blind select"),
    "j_astronomer": sc(kind="shop", dollars=0,
                       shop_actions=[{"type": "inject_item", "kind": "planet", "key": "c_mercury", "price": 3},
                                     {"type": "buy", "item_idx": 0}],
                       note="Planets cost $0: a $0 purchase must succeed"),
    "j_burnt": sc(hand=PAIR, post="next_blind", note="most played hand levelled at next blind"),
    "j_bootstraps": sc(dollars=10),
    "j_caino": sc(hand=[C(13, S, "Glass"), C(13, H, "Glass"), C(13, Cl, "Glass"), C(13, D, "Glass"), C(12, S, "Glass")],
                  repeats=2, tries=7, note="face card destroyed (Glass shatter) -> x1"),
    "j_triboulet": sc(hand=FACES),
    "j_yorick": sc(pre_discards=[[0, 1, 2, 3, 4]] * 5, discards_left=5, note="23 discards -> x1"),
    "j_chicot": sc(boss="bl_goad", hand=FLUSH_S, note="boss disabled: Spades must not be debuffed"),
    "j_perkeo": sc(kind="shop", consumables=["c_hermit"], setup=lambda g: g.consumable_hand.append("c_hermit"),
                   shop_actions=[{"type": "leave_shop"}], note="leaving the shop -> Negative copy of a held consumable"),
}

# =============================================================================================
# CONSUMABLE SCENARIOS (52)
# =============================================================================================

def use(**kw):
    kw.setdefault("kind", "use")
    return Scenario(**kw)


ENHANCE_TAROTS = {"c_magician", "c_empress", "c_heirophant", "c_lovers", "c_chariot", "c_justice", "c_devil", "c_tower"}
SUIT_TAROTS = {"c_star", "c_moon", "c_sun", "c_world"}
SEAL_SPECTRALS = {"c_talisman", "c_deja_vu", "c_trance", "c_medium"}

CONSUMABLE_SCENARIOS: dict = {
    "c_fool": use(consumables=["c_hermit"], pre_use=[(0, [])], note="copy of the last Tarot/Planet used"),
    "c_high_priestess": use(),
    "c_emperor": use(),
    "c_hermit": use(dollars=10),
    "c_wheel_of_fortune": use(left=["j_joker"], tries=7, note="1/4: edition on a joker"),
    "c_strength": use(target_idx=[0, 1]),
    "c_hanged_man": use(target_idx=[0, 1]),
    "c_death": use(target_idx=[2, 0]),
    "c_temperance": use(left=["j_joker"], dollars=10),
    "c_judgement": use(),
    "c_familiar": use(target_idx=[0]), "c_grim": use(target_idx=[0]), "c_incantation": use(target_idx=[0]),
    "c_aura": use(left=["j_joker"], target_idx=[0]),
    "c_wraith": use(dollars=10),
    "c_sigil": use(hand=HIGH_CARD), "c_ouija": use(hand=HIGH_CARD),
    "c_ectoplasm": use(left=["j_joker"]),
    "c_immolate": use(dollars=0),
    "c_ankh": use(left=["j_joker", "j_jolly"]),
    "c_hex": use(left=["j_joker", "j_jolly"]),
    "c_cryptid": use(target_idx=[0]),
    "c_soul": use(),
    "c_black_hole": use(),
}
for _k in ENHANCE_TAROTS | SUIT_TAROTS:
    CONSUMABLE_SCENARIOS.setdefault(_k, use(target_idx=[0, 1]))
for _k in SEAL_SPECTRALS:
    CONSUMABLE_SCENARIOS.setdefault(_k, use(target_idx=[0]))
for _p in P.PLANETS:
    CONSUMABLE_SCENARIOS.setdefault(_p["key"], use())

# =============================================================================================
# VOUCHER SCENARIOS (32): buy the voucher, select the next blind, compare everything
# =============================================================================================

VOUCHER_SCENARIOS: dict = {v["key"]: Scenario(kind="voucher") for v in P.VOUCHERS}
VOUCHER_SCENARIOS["v_overstock_plus"] = Scenario(kind="voucher", note="needs v_overstock_norm first; tested standalone")
VOUCHER_SCENARIOS["v_blank"] = Scenario(kind="voucher", xfail="Blank does nothing by design")
VOUCHER_SCENARIOS["v_antimatter"] = Scenario(kind="voucher", note="+1 joker slot")
# W2 added the {"type": "reroll_boss"} action (Director's Cut once per ante, Retcon unlimited);
# the voucher probe sees it through run_state.used_vouchers, the mechanic itself is pinned by
# engine/tests/sim_tests/test_delegate.py::test_directors_cut_once_per_ante and
# test_sweep.py::TestRetcon (W5).
VOUCHER_SCENARIOS["v_retcon"] = Scenario(kind="voucher", note="unlimited boss rerolls ($10 each)")
VOUCHER_SCENARIOS["v_directors_cut"] = Scenario(kind="voucher", note="+1 boss reroll per ante (free_rerolls today)")
for _k in ("v_hieroglyph", "v_petroglyph"):
    VOUCHER_SCENARIOS[_k] = Scenario(kind="voucher", setup=lambda g: setattr(g, "ante", 3), note="-1 ante (from ante 3)")


# =============================================================================================
# tests
# =============================================================================================

def _registered_joker(key: str) -> bool:
    reg = getattr(sys.modules.get("balatro_sim.jokers.base"), "JOKER_REGISTRY", None)
    return reg is None or key in reg


def _registered_consumable(key: str) -> bool:
    known = set(getattr(CONS, "ALL_TAROTS", [])) | set(getattr(CONS, "ALL_PLANETS", [])) | set(getattr(CONS, "ALL_SPECTRALS", []))
    return not known or key in known


def _registered_voucher(key: str) -> bool:
    known = set(getattr(CONS, "ALL_VOUCHERS", []))
    return not known or key in known


@pytest.mark.parametrize("key", [j["key"] for j in P.JOKERS])
def test_joker_reachable(key):
    sc_ = JOKER_SCENARIOS.get(key)
    assert sc_ is not None, f"{key}: no scenario defined in JOKER_SCENARIOS (add one)"
    if not _registered_joker(key):
        pytest.xfail(f"{key}: not registered under its game key yet (W1 re-key)")
    probe_item(key, "joker", sc_)


@pytest.mark.parametrize("key", sorted(ALL_CONSUMABLE_KEYS))
def test_consumable_reachable(key):
    sc_ = CONSUMABLE_SCENARIOS.get(key)
    assert sc_ is not None, f"{key}: no scenario defined in CONSUMABLE_SCENARIOS (add one)"
    if not _registered_consumable(key):
        pytest.xfail(f"{key}: not registered under its game key yet (W1 re-key)")
    probe_item(key, "consumable", sc_)


@pytest.mark.parametrize("key", [v["key"] for v in P.VOUCHERS])
def test_voucher_reachable(key):
    sc_ = VOUCHER_SCENARIOS[key]
    if not _registered_voucher(key):
        pytest.xfail(f"{key}: not registered under its game key yet (W1 re-key)")
    probe_item(key, "voucher", sc_)


def test_every_pool_item_has_a_scenario():
    missing = [j["key"] for j in P.JOKERS if j["key"] not in JOKER_SCENARIOS]
    missing += [k for k in sorted(ALL_CONSUMABLE_KEYS) if k not in CONSUMABLE_SCENARIOS]
    assert not missing, missing
