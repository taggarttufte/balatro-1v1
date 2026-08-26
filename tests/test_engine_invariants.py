"""Stream-independence invariants for the MP engine (Phase 1, W8).

Written against the TARGET architecture (CAMPAIGN_LOG "Phase 1 -- REVISED architecture"): the
engine delegates every *generation* event to ``rng.generate`` through a ``RunState`` it keeps in
sync, and threads keyed RNG (``pseudorandom('lucky_mult')`` ...) through every *effect* roll.
Consequences these tests pin down:

* effect rolls (Lucky / Glass / Bloodstone / 8 Ball / Business Card) never move the next
  shelf, packs, voucher, boss or tags;
* owning a joker only changes the slot that would have shown it (in-place resample from a side
  stream); every other slot / pack / voucher is identical; Showman removes even that;
* k rerolls = queue positions ``[2k, 2k+2)`` of the ante's ``shop_queue`` (ground truth +
  ``generate.py`` agree);
* two players on one seed who play differently differ only by their own rerolls + blocked
  slots -- encoded as: replay each player's action history through ``generate.RunState`` and
  demand equality with what the engine showed;
* determinism: same seed + same actions = identical state; ``clone()`` isolates.

Expected RED against today's engine (single ``random.Random`` stream, no ``run_state``); goes
green as W2/W3 land.  Tests needing a hook that does not exist yet ``xfail`` (non-strict)
instead of erroring -- see tests/HARNESS_NOTES.md for the hook list.

Run:  python -m pytest tests/test_engine_invariants.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
MP_ROOT = HERE.parent
if str(MP_ROOT) not in sys.path:
    sys.path.insert(0, str(MP_ROOT))

from oracle import engine_parity as EP  # noqa: E402  (engine import + driver + replay)
from oracle import parity_check as PC  # noqa: E402
from rng import generate as GEN  # noqa: E402

GM = EP.import_engine()          # raises if the BRL balatro_sim shadows the fork
BalatroGame, State = GM.BalatroGame, GM.State
from balatro_sim.card import Card  # noqa: E402
from balatro_sim.jokers.base import JokerInstance  # noqa: E402

SEED_MAIN = "7I4M53DL"           # live-check seed from the campaign log (Banner + Hierophant shelf)
SEEDS_GT = ["7I4M53DL", "ALEEB", "11111111"]

# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------

def C(rank, suit, enh="None", seal="None", ed="None") -> Card:
    return Card(rank=rank, suit=suit, enhancement=enh, seal=seal, edition=ed)


def sig_items(items) -> list:
    return [PC.item_sig(it, True) for it in items]


def hook(name: str, obj, *attrs):
    """Return the first present attribute or xfail with the hook name (keeps the test red-by-design
    rather than erroring at a missing attribute)."""
    for a in attrs:
        if obj is not None and getattr(obj, a, None) is not None:
            return getattr(obj, a)
    pytest.xfail(f"engine hook missing: {name} (see HARNESS_NOTES.md)")


def run_state_of(game):
    return getattr(game, "run_state", None)


def add_owned_joker(game, key: str, edition=None):
    """Own a joker the way a purchase would: joker slot + run_state bookkeeping (used_jokers,
    showman).  Uses the target hook when present, else does both halves by hand."""
    fn = getattr(game, "debug_add_joker", None)
    if callable(fn):
        return fn(key, edition)
    j = JokerInstance(key, edition or "None")
    game.jokers.append(j)
    rs = run_state_of(game)
    if rs is not None:
        rs.acquire(key)
    return j


def owned_keys(game) -> set:
    keys = {j.key for j in game.jokers} | set(game.consumable_hand)
    rs = run_state_of(game)
    if rs is not None:
        keys |= set(rs.owned_jokers) | set(rs.owned_consumables)
    return keys


def shelves_equal_modulo_ownership(a_items, b_items, a_owned: set, where: str):
    """Slot-by-slot: equal, OR B shows something A owns and A shows something else (in-place
    resample).  Any other difference is a stream shift -> assertion with the slot listed."""
    assert len(a_items) == len(b_items), f"{where}: slot count {len(a_items)} vs {len(b_items)}"
    shifted = []
    for i, (a, b) in enumerate(zip(a_items, b_items)):
        if PC.item_sig(a, True) == PC.item_sig(b, True):
            continue
        if b.get("key") in a_owned and a.get("key") not in a_owned:
            continue   # legitimate in-place block
        shifted.append((i, PC.item_sig(b, True), PC.item_sig(a, True)))
    assert not shifted, f"{where}: slots shifted (slot, B, A): {shifted}"


def to_first_shop(seed: str, before_blind=None, after_blind=None) -> EP.EngineDriver:
    d = EP.EngineDriver(GM, seed)
    if before_blind:
        before_blind(d.g)
    d.g.step({"type": "play_blind"})
    assert d.g.state == State.SELECTING_HAND
    if after_blind:
        after_blind(d.g)
    win = getattr(d.g, "debug_win_blind", None)
    if callable(win):
        win()
    else:
        d.g.chips_scored = d.g.current_blind.chips_target
        d.g.state = State.ROUND_EVAL
    d.g.step({"type": "advance"})
    assert d.g.state == State.SHOP
    d.g.dollars = 10 ** 6
    return d


def play_crafted_hand(game, cards: list, indices=None):
    """Replace the dealt hand with ``cards`` (kept consistent with full_deck) and play them."""
    game.full_deck = list(game.deck) + list(cards)
    game.hand = list(cards)
    game.discard_pile = []
    game.step({"type": "play", "cards": list(range(len(cards))) if indices is None else indices})


def state_signature(game) -> str:
    """Canonical, Card-id-independent snapshot.  Prefers the engine's own hook."""
    fn = getattr(game, "state_signature", None)
    if callable(fn):
        return repr(fn())
    parts = {}
    by_id = {c.id: c for c in game.full_deck}
    card = lambda c: (c.rank, c.suit, c.enhancement, c.edition, c.seal, c.debuffed)  # noqa: E731
    ID_FIELDS = {"_played_this_ante", "_forced_card_id"}   # hold Card ids (a process-global counter)
    for k, v in sorted(vars(game).items()):
        if k in ID_FIELDS:
            ids = v if isinstance(v, (set, list, tuple)) else [v]
            parts[k] = sorted(repr(card(by_id[i])) if i in by_id else "?" for i in ids if i != -1)
        elif isinstance(v, (int, float, str, bool)) or v is None:
            parts[k] = v
        elif isinstance(v, (set, frozenset)):
            parts[k] = sorted(map(repr, v))
        elif isinstance(v, dict) and all(isinstance(x, (int, float, str, bool, type(None))) for x in v.values()):
            parts[k] = sorted(v.items())
    parts["state"] = game.state.name
    parts["blind"] = (game.current_blind.kind, game.current_blind.chips_target, game.current_blind.boss_key)
    parts["jokers"] = [(j.key, j.edition, sorted((k, repr(v)) for k, v in j.state.items())) for j in game.jokers]
    parts["consumables"] = list(game.consumable_hand)
    parts["shop"] = [(it.kind, it.key, getattr(it, "edition", None), it.price, getattr(it, "sold", False))
                     for it in game.current_shop]
    parts["booster"] = [repr(EP.item_from_engine(c)) for c in (getattr(game, "booster_choices", []) or [])]
    parts["full_deck"] = sorted(map(card, game.full_deck))
    parts["deck_order"] = [card(c) for c in game.deck]
    parts["hand"] = [card(c) for c in game.hand]
    parts["discard"] = [card(c) for c in game.discard_pile]
    rs = run_state_of(game)
    if rs is not None:
        parts["run_state"] = (rs.ante, sorted(rs.used_jokers), list(rs.owned_jokers), list(rs.owned_consumables),
                              sorted(rs.used_vouchers), rs.boss_blind, dict(rs.blind_tags), rs.showman,
                              repr(rs.rng.snapshot()) if hasattr(rs.rng, "snapshot") else None)
    if getattr(game, "rng", None) is not None and hasattr(game.rng, "getstate"):
        parts["legacy_rng"] = repr(game.rng.getstate())
    return repr(parts)


def generate_first_shop(seed: str):
    """generate.py view of the first shop (after Small, ante 1) + the RunState to continue."""
    st = GEN.RunState(seed)
    GEN.start_run(st, "b_red")
    st.new_round()
    return st, GEN.generate_shop(st)


# ---------------------------------------------------------------------------------------------
# 1. effect rolls never move generation
# ---------------------------------------------------------------------------------------------

LOADED_HAND = [C(8, "Hearts", "Lucky"), C(8, "Hearts", "Glass"), C(13, "Hearts", "Lucky"),
               C(12, "Hearts", "Glass"), C(11, "Hearts", "Lucky")]          # Flush: all 5 score
QUIET_HAND = [C(2, "Spades"), C(4, "Clubs"), C(6, "Diamonds"), C(9, "Spades"), C(3, "Clubs")]
ROLLING_JOKERS = ["j_bloodstone", "j_8_ball", "j_business"]                 # roll only on hearts / 8s / faces


@pytest.mark.parametrize("seed", SEEDS_GT)
def test_effect_rolls_do_not_move_generation(seed):
    """Same seed, same owned jokers; A plays a hand that fires lucky_mult/lucky_money x3, glass x2,
    bloodstone x5, 8ball x2, business x3; B plays a hand that fires nothing.  Next shelf, packs,
    voucher and boss must be identical (modulo A's 8-Ball tarot, which A now OWNS and which may
    therefore be blocked in place -- the only legitimate difference)."""
    def own(g):
        for k in ROLLING_JOKERS:
            add_owned_joker(g, k)

    A = to_first_shop(seed, before_blind=own, after_blind=lambda g: play_crafted_hand(g, LOADED_HAND))
    B = to_first_shop(seed, before_blind=own, after_blind=lambda g: play_crafted_hand(g, QUIET_HAND))
    a_owned = owned_keys(A.g)

    shelves_equal_modulo_ownership(A.shelf_items(), B.shelf_items(), a_owned, "shelf after loaded hand")
    assert A.voucher_item() == B.voucher_item(), "voucher moved"
    assert [k for _, k in A.pack_slots()] == [k for _, k in B.pack_slots()], "pack kinds moved"

    for (ia, _), (ib, _) in zip(A.pack_slots(), B.pack_slots()):
        oka, ca = A.open_pack(ia)
        okb, cb = B.open_pack(ib)
        assert oka and okb, "buying a booster must enter State.BOOSTER_OPEN (engine bug: §7 item 0)"
        shelves_equal_modulo_ownership(ca, cb, a_owned, f"pack {ia}")
        A.skip_pack()
        B.skip_pack()
    for k in range(3):
        shelves_equal_modulo_ownership(A.reroll(), B.reroll(), a_owned, f"reroll {k + 1}")

    # boss of this ante is a generation event too ('boss' stream), as are the two tags
    A.leave_shop()
    B.leave_shop()
    for D in (A, B):
        D.win_blind()            # Big blind
        D.leave_shop()
    assert A.g.current_blind.kind == "Boss" and B.g.current_blind.kind == "Boss"
    assert A.g.current_blind.boss_key == B.g.current_blind.boss_key, "boss moved"
    ta, tb = A.tags_now(), B.tags_now()
    if ta is None and tb is None:
        pytest.xfail("engine hook missing: blind_tags (W6 wiring)")
    assert ta == tb, "tags moved"


# ---------------------------------------------------------------------------------------------
# 2. ownership blocks in place; Showman lifts the block
# ---------------------------------------------------------------------------------------------

def _seed_with_shelf_joker(candidates=("7I4M53DL", "ALEEB", "11111111", "1558AXDL", "15H9Z3IY")):
    """A ground-truth seed whose first shelf (generate.py) has a Joker; returns (seed, slot, key)."""
    for s in candidates:
        _, shop = generate_first_shop(s)
        for i, c in enumerate(shop.cards):
            if c.is_joker:
                return s, i, c.key
    raise RuntimeError("no candidate seed has a joker on its first shelf")


def test_ownership_blocks_only_that_slot():
    seed, slot, X = _seed_with_shelf_joker()
    B = to_first_shop(seed)
    A = to_first_shop(seed, before_blind=lambda g: add_owned_joker(g, X))
    b_shelf, a_shelf = B.shelf_items(), A.shelf_items()
    assert b_shelf[slot]["key"] == X, (
        f"engine shelf slot {slot} is {b_shelf[slot]['key']}, generate.py/ground truth say {X}: "
        "the engine is not delegating generation to rng.generate yet (W2)")
    assert a_shelf[slot]["key"] != X, "owned joker was shown again (used_jokers block missing)"
    for i, (a, b) in enumerate(zip(a_shelf, b_shelf)):
        if i != slot:
            assert PC.item_sig(a, True) == PC.item_sig(b, True), f"slot {i} shifted by ownership of {X}"
    assert A.voucher_item() == B.voucher_item()
    assert [k for _, k in A.pack_slots()] == [k for _, k in B.pack_slots()]
    for (ia, _), (ib, _) in zip(A.pack_slots(), B.pack_slots()):
        oka, ca = A.open_pack(ia)
        okb, cb = B.open_pack(ib)
        assert oka and okb, "buying a booster must enter State.BOOSTER_OPEN"
        shelves_equal_modulo_ownership(ca, cb, {X}, f"pack {ia}")
        A.skip_pack()
        B.skip_pack()
    for k in range(4):
        shelves_equal_modulo_ownership(A.reroll(), B.reroll(), {X}, f"reroll {k + 1}")


def test_showman_lifts_the_block():
    seed, slot, X = _seed_with_shelf_joker()
    B = to_first_shop(seed)
    Cc = to_first_shop(seed, before_blind=lambda g: (add_owned_joker(g, X), add_owned_joker(g, "j_ring_master")))
    rs = run_state_of(Cc.g)
    if rs is not None:
        assert rs.showman, "run_state.showman not set when Showman is owned"
    b_shelf, c_shelf = B.shelf_items(), Cc.shelf_items()
    assert b_shelf[slot]["key"] == X, "engine is not delegating generation yet (W2)"
    assert sig_items(c_shelf) == sig_items(b_shelf), "with Showman the owned joker must be shown again"
    for k in range(3):
        assert sig_items(Cc.reroll()) == sig_items(B.reroll()), f"reroll {k + 1} differs under Showman"


# ---------------------------------------------------------------------------------------------
# 3. reroll = queue advance
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", SEEDS_GT)
def test_reroll_is_queue_advance(seed):
    gt = PC.apply_variant(PC.load_ground_truth([seed])[seed], "faithful")
    queue = gt["antes"]["1"]["shop_queue"]
    st, gshop = generate_first_shop(seed)
    D = to_first_shop(seed)
    for k in range(5):
        shelf = D.shelf_items() if k == 0 else D.reroll()
        if k:
            GEN.reroll_shop(st, gshop)
        expect_gt = sig_items(queue[2 * k:2 * k + 2])
        expect_gen = sig_items([PC._cardgen_item(c) for c in gshop.cards])
        assert expect_gt == expect_gen, f"ground truth and generate.py disagree at k={k} (oracle problem, not engine)"
        assert sig_items(shelf) == expect_gt, f"after {k} rerolls engine shows {sig_items(shelf)}, queue[{2*k}:{2*k+2}] is {expect_gt}"


# ---------------------------------------------------------------------------------------------
# 4. divergent play stays queue-aligned (replay each history through generate.py)
# ---------------------------------------------------------------------------------------------

def _diff_observed_vs_replay(seed: str, D: EP.EngineDriver, antes: int) -> list:
    got = D.observed()
    ref = EP.replay_visits(seed, D.visits)
    rows = []
    for a in range(1, antes + 1):
        depth = len(ref["antes"][str(a)]["shop_queue"])
        r, _, _ = EP.compare_seed(ref, got, [a], ["boss", "voucher", "tags", "shop_queue", "packs"], depth, set())
        rows.extend(r)
    return rows


@pytest.mark.parametrize("seed", SEEDS_GT[:2])
def test_two_players_divergent_play_stay_queue_aligned(seed):
    A = EP.EngineDriver(GM, seed)
    A.run(2, EP.Policy(rerolls=1, reroll_every_shop=True, buy_shelf=True, buy_shelf_slot=0))
    B = EP.EngineDriver(GM, seed)
    B.run(2, EP.Policy(rerolls=2, reroll_every_shop=True))
    rows_a = _diff_observed_vs_replay(seed, A, 2)
    rows_b = _diff_observed_vs_replay(seed, B, 2)
    msg = []
    for who, rows in (("A", rows_a), ("B", rows_b)):
        if rows:
            msg.append(f"player {who}: {len(rows)} field(s) not explained by own rerolls/purchases; first: {rows[0]}")
    if any("boss" in r[0] or "tags" in r[0] for r in rows_a + rows_b) and not (A.tags_now() or B.tags_now()):
        msg.append("(no blind_tags hook: tags rows are 'not produced')")
    assert not rows_a and not rows_b, "\n".join(msg)
    # sanity: the two histories really did diverge
    assert A.visits[0].actions != B.visits[0].actions


# ---------------------------------------------------------------------------------------------
# 5. determinism / clone isolation
# ---------------------------------------------------------------------------------------------

def _scripted_walk(seed: str, steps: int = 40) -> list:
    g = BalatroGame(seed=seed)
    sigs = []
    for i in range(steps):
        s = g.state
        if s == State.GAME_OVER:
            break
        if s == State.BLIND_SELECT:
            g.step({"type": "play_blind"})
        elif s == State.SELECTING_HAND:
            if g.hands_left <= 1 and g.chips_scored < g.current_blind.chips_target:
                g.chips_scored = g.current_blind.chips_target   # never lose; keep the walk long
                g.state = State.ROUND_EVAL
            else:
                g.step({"type": "play", "cards": list(range(min(5, len(g.hand))))})
        elif s == State.ROUND_EVAL:
            g.step({"type": "advance"})
        elif s == State.SHOP:
            g.dollars = 10 ** 6
            g.step({"type": "reroll"})
            g.step({"type": "buy", "item_idx": 0})
            packs = [i for i, it in enumerate(g.current_shop) if it.kind == "booster" and not it.sold]
            if packs:
                g.step({"type": "buy", "item_idx": packs[0]})
            if g.state == State.BOOSTER_OPEN:
                g.step({"type": "pick_booster", "indices": [0]})
            g.step({"type": "leave_shop"})
        elif s == State.BOOSTER_OPEN:
            g.step({"type": "skip_booster"})
        sigs.append(state_signature(g))
    return sigs


@pytest.mark.parametrize("seed", ["7I4M53DL", "ALEEB"])
def test_same_seed_same_actions_identical_state(seed):
    a, b = _scripted_walk(seed), _scripted_walk(seed)
    assert len(a) == len(b) and len(a) > 10
    for i, (x, y) in enumerate(zip(a, b)):
        assert x == y, f"state diverged at step {i}"


def test_different_seeds_diverge():
    assert _scripted_walk("7I4M53DL") != _scripted_walk("ALEEB")


def test_clone_then_diverge_leaves_original_untouched():
    D = to_first_shop(SEED_MAIN)
    g = D.g
    before = state_signature(g)
    c = g.clone()
    assert state_signature(c) == before, "clone must be an exact copy"
    c.step({"type": "reroll"})
    c.step({"type": "buy", "item_idx": 0})
    c.step({"type": "leave_shop"})
    c.step({"type": "play_blind"})
    c.step({"type": "play", "cards": [0, 1, 2, 3, 4]})
    assert state_signature(g) == before, "stepping the clone mutated the original"
    # and the original still walks exactly like a fresh twin would
    twin = to_first_shop(SEED_MAIN).g
    for x in (g, twin):
        x.step({"type": "reroll"})
    assert state_signature(g) == state_signature(twin)
    rs_g, rs_c = run_state_of(g), run_state_of(c)
    if rs_g is None:
        pytest.xfail("engine hook missing: run_state (clone must deep-copy RunState + PseudoRandom)")
    assert rs_g is not rs_c and rs_g.rng is not rs_c.rng, "clone shares RunState/PseudoRandom with the original"


# ---------------------------------------------------------------------------------------------
# 6. Phase 2 cross-cutting invariants (W4 exit gate, 2026-08-21)
# ---------------------------------------------------------------------------------------------

ENGINE_PKG = MP_ROOT / "engine" / "balatro_sim"
# Not game logic: the V5/V7/MP gym envs and the env_sim rollout helper (documented at
# env_sim.py:663 -- a stand-alone random rollout, not a game draw).
NON_GAME_LOGIC = {"env_sim.py", "env_v5.py", "env_v7.py", "env_mp.py"}


def test_no_python_random_in_game_logic_after_phase_2():
    """Phase 1 deleted ``game.rng``; Phase 2 (MLB rules, The Order, decks, stakes) must not have
    reintroduced an unkeyed stream.  Every roll goes through ``run_state.rng`` (PseudoRandom)."""
    import re as _re
    pat = _re.compile(r"\brandom\.Random\(|^\s*import random\b|^\s*from random import|np\.random\.|random\.(choice|shuffle|randint|random)\(",
                      _re.M)
    hits = []
    for f in sorted(ENGINE_PKG.rglob("*.py")):
        if f.name in NON_GAME_LOGIC:
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        for m in pat.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            src = text.splitlines()[line - 1].strip()
            if src.startswith("#") or "\"\"\"" in src or "The legacy random.Random" in src:
                continue
            hits.append(f"{f.relative_to(ENGINE_PKG)}:{line}: {src}")
    assert not hits, "unkeyed randomness in game logic:\n" + "\n".join(hits)
    # and the seed-less constructor still draws its seed from `secrets`, not `random`
    assert "secrets.randbelow" in (ENGINE_PKG / "game.py").read_text(encoding="utf-8")


def test_ruleset_default_is_vanilla_and_mlb_is_distinct():
    a = BalatroGame(seed=SEED_MAIN)
    b = BalatroGame(seed=SEED_MAIN, ruleset="vanilla")
    c = BalatroGame(seed=SEED_MAIN, ruleset="mlb")
    assert a.state_signature() == b.state_signature()
    assert a.state_signature() != c.state_signature()
    assert not a.run_state.banned_keys & set(GM.MLB_BANNED_KEYS) if hasattr(GM, "MLB_BANNED_KEYS") else True
    assert a.lives == 0 and c.lives == 4 and not a.mlb and c.mlb


MLB_BAN_SEEDS = ["7I4M53DL", "ALEEB", "11111111", "1558AXDL", "15H9Z3IY", "1KV4W6YS", "1MD1YZ9T", "28V7DD4H",
                 "29DAQVG1", "29Y3L4S9", "29ZSW8MY", "2BRGI767", "2CP4KSXZ", "2GHBLJD9", "2H9N3ISZ", "34JCNMPA",
                 "41Y71M6E", "46Y8UZEG", "4H2L46CE", "4K8A9QER"]


@pytest.mark.parametrize("seed", MLB_BAN_SEEDS)
def test_mlb_banned_keys_never_generated(seed):
    """ruleset='mlb' (solo game): over 4 antes with 3 rerolls per ante and every pack opened, no
    banned joker / voucher / tag / blind ever appears on a shelf, in a pack, as the voucher, as a
    tag or as the (shadow) boss draw; the Boss slot from ante 2 is the Nemesis."""
    from balatro_sim.constants import MLB_BANNED_KEYS, MLB_NEMESIS_KEY
    D = EP.EngineDriver(GM, seed)
    D.g = BalatroGame(seed=seed, ruleset="mlb")
    D.ante_info = {}
    D._record_ante_start(1)
    bosses_played = {}
    orig_win = D.win_blind

    def win_blind():
        g = D.g
        if g.current_blind.kind == "Boss":
            bosses_played[g.ante] = (g.current_blind.boss_key, g.current_blind.is_pvp)
        orig_win()
    D.win_blind = win_blind
    D.run(4, EP.Policy(rerolls=3))
    got = D.observed()
    seen = []
    for a, node in got["antes"].items():
        seen += [it["key"] for it in node["shop_queue"]]
        seen += [node["voucher"]["key"]] if node["voucher"] else []
        seen += [node["boss"]["key"]] if node["boss"] else []
        seen += [node["tags"]["small"]["key"], node["tags"]["big"]["key"]] if node["tags"] else []
        for shop in node["shops"]:
            for pk in shop["packs"]:
                seen += [c["key"] for c in (pk["cards"] or [])]
    assert len(seen) > 100
    banned_seen = sorted({k for k in seen if k in MLB_BANNED_KEYS})
    assert not banned_seen, banned_seen
    assert bosses_played[1][1] is False and bosses_played[1][0] not in MLB_BANNED_KEYS
    for a in (2, 3):                       # ante 4's boss is not reached by a 4-ante shop walk
        assert bosses_played[a] == (MLB_NEMESIS_KEY, True), bosses_played
