"""Phase 2 exit gate (W4, 2026-08-21): two scripted players on ONE seed play full MLB matches
through the ENGINE (``balatro_sim.mlb_match.MLBMatch``) and every rule of
``docs/PHASE2_BRIEF_2026-08.md`` "Exit gate" is asserted, not eyeballed.

Items (see tests/GATE_NOTES.md for what each proves and how the key classes are defined):
  1. lives / comeback / endless / early-end / tie          -> class TestLives
  2. money at the Nemesis and at a failed regular blind    -> class TestMoney
  3. queue alignment between the two players (RNG key diff) -> class TestAlignment (+ voucher stream)
  4. vanilla unchanged: engine_parity 126/126; MLB solo run differs from vanilla only in vouchers,
     bans and the ante >= 2 boss slot                        -> class TestVanillaUnchanged
  5. clone() mid-match replays identically                  -> class TestClone

Matches are driven by ``scripts/mlb_match_demo.py`` (``MatchRecorder`` + ``ScriptedPlayer``);
each scenario is run once per (seed, deck) and cached for the whole module.

Run:  python -m pytest tests/test_mlb_match_gate.py -q -rx
"""
from __future__ import annotations

import functools
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
MP_ROOT = HERE.parent
for _p in (str(MP_ROOT), str(MP_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mlb_match_demo as D  # noqa: E402  (imports the engine fork through engine_parity.import_engine)
from oracle import engine_parity as EP  # noqa: E402
from oracle import parity_check as PC  # noqa: E402
from rng import pools as P  # noqa: E402

from balatro_sim.game import State  # noqa: E402
from balatro_sim.mlb_match import MLBMatch  # noqa: E402
from balatro_sim.constants import (  # noqa: E402
    MLB_BANNED_KEYS, MLB_STARTING_LIVES, MLB_COMEBACK_PER_LIFE, MLB_NEMESIS_REWARD, MLB_NEMESIS_KEY,
    STARTING_HANDS, get_blind_amount, blind_base_chips,
)

GM = D.GM

SEEDS = ["7I4M53DL", "ALEEB", "11111111", "1558AXDL", "15H9Z3IY",
         "1KV4W6YS", "1MD1YZ9T", "28V7DD4H", "29DAQVG1", "29Y3L4S9"]      # all in oracle/ground_truth/
DECKS = ["b_red", "b_checkered", "b_plasma"]
CASES = [(s, d) for d in DECKS for s in SEEDS]
IDS = [f"{d[2:]}-{s}" for s, d in CASES]

# ---------------------------------------------------------------------------------------------
# scenarios (players), cached per (seed, deck)
# ---------------------------------------------------------------------------------------------

GREEDY = D.ScriptedPlayer(name="greedy", hand="greedy")
WEAK = D.ScriptedPlayer(name="weak", hand="weak")
# PvP scenario: both clear regular blinds through debug_win_blind (no stream touched); P0 plays
# one-card hands at every Nemesis from ante 2 -> loses it (early-ended: P1 still has hands).
PVP_LOSER = D.ScriptedPlayer(name="weak@nemesis", hand="greedy_until", weak_from_ante=2, debug_win_regular=True)
PVP_GREEDY = D.ScriptedPlayer(name="greedy", hand="greedy", debug_win_regular=True)
# Endless scenario: identical greedy play at every Nemesis (same seed, no purchases -> exact tie)
# through ante 8; from ante 9 P0 plays one-card hands and loses four Nemeses in a row.
TIE_THEN_WEAK = D.ScriptedPlayer(name="tie-then-weak", hand="greedy_until", weak_from_ante=9, debug_win_regular=True)
# Alignment scenario: the demo's two shoppers, regular blinds debug-won so the match lasts.
OPENER = D.ScriptedPlayer(name="opener", hand="greedy", open_pack_slot=0, pick_from_pack=True, debug_win_regular=True)
REROLLER = D.ScriptedPlayer(name="reroller", hand="greedy", rerolls_per_visit=1, buy_slot0=True, debug_win_regular=True)
# Voucher scenario: P0 buys the shelf voucher at every visit it can (unlimited money), P1 never.
BUYER = D.ScriptedPlayer(name="voucher-buyer", hand="greedy", debug_win_regular=True, buy_voucher=True, rich=True)
ABSTAINER = D.ScriptedPlayer(name="abstainer", hand="greedy", debug_win_regular=True, rich=True)


@functools.lru_cache(maxsize=None)
def natural(seed: str, deck: str) -> D.MatchRecorder:
    """greedy vs weak, no shopping: lives go on regular blinds, failed blinds cash out."""
    return D.MatchRecorder(seed, [GREEDY, WEAK], deck_key=deck).run()


@functools.lru_cache(maxsize=None)
def pvp(seed: str, deck: str) -> D.MatchRecorder:
    return D.MatchRecorder(seed, [PVP_LOSER, PVP_GREEDY], deck_key=deck).run()


@functools.lru_cache(maxsize=None)
def endless(seed: str, deck: str) -> D.MatchRecorder:
    return D.MatchRecorder(seed, [TIE_THEN_WEAK, PVP_GREEDY], deck_key=deck).run()


@functools.lru_cache(maxsize=None)
def shoppers(seed: str, deck: str) -> D.MatchRecorder:
    return D.MatchRecorder(seed, [OPENER, REROLLER], deck_key=deck).run()


@functools.lru_cache(maxsize=None)
def voucher_match(seed: str) -> D.MatchRecorder:
    return D.MatchRecorder(seed, [BUYER, ABSTAINER], lives=12, max_antes=16).run()


ALL_SCENARIOS = (natural, pvp, endless, shoppers)


def lives_of(rec: D.MatchRecorder, player: int):
    return [e for e in rec.lives if e.player == player]


# ---------------------------------------------------------------------------------------------
# 1. lives, comeback, endless, early end, tie
# ---------------------------------------------------------------------------------------------

class TestLives:
    @pytest.mark.parametrize("seed,deck", CASES, ids=IDS)
    def test_every_lost_blind_costs_exactly_one_life_of_that_player(self, seed, deck):
        for scen in ALL_SCENARIOS:
            rec = scen(seed, deck)
            # every life event is a decrement of exactly 1
            for e in rec.lives:
                assert e.lives_before - e.lives_after == 1, (scen.__name__, e)
            # at most one life per (player, step) and per (player, ante, blind)
            keys = [(e.player, e.ante, e.blind) for e in rec.lives]
            assert len(keys) == len(set(keys)), (scen.__name__, keys)
            # regular blinds: lost (>= 1 hand played) <-> one life event of THAT player at that step
            for b in rec.blinds:
                if b.is_pvp:
                    continue
                hits = [e for e in rec.lives if e.player == b.player and e.step == b.step]
                others = [e for e in rec.lives if e.player != b.player and e.step == b.step and not e.is_pvp]
                if b.won:
                    assert not hits, (scen.__name__, "won blind cost a life", b, hits)
                else:
                    assert b.hands_used >= 1, (scen.__name__, b)
                    assert len(hits) == 1 and hits[0].cause == "regular_fail" and not hits[0].is_pvp, \
                        (scen.__name__, "lost blind without exactly one life", b, hits)
                assert not others, (scen.__name__, "the OTHER player lost a life at this blind", b, others)
            # Nemesis: loser loses exactly one, the other none; tie -> nobody
            for pv in rec.pvp:
                at = [e for e in rec.lives if e.step == pv.step]
                if pv.tie:
                    assert not at and pv.score0 == pv.score1, (scen.__name__, pv, at)
                else:
                    assert len(at) == 1 and at[0].player == pv.loser and at[0].is_pvp and at[0].cause == "pvp_loss", \
                        (scen.__name__, pv, at)
                    assert (pv.score0 < pv.score1) == (pv.loser == 0), (scen.__name__, pv)
            # bookkeeping: total life events == lost regular blinds + decided Nemeses
            lost_regular = sum(1 for b in rec.blinds if not b.is_pvp and not b.won)
            decided = sum(1 for pv in rec.pvp if not pv.tie)
            assert len(rec.lives) == lost_regular + decided, (scen.__name__, len(rec.lives), lost_regular, decided)

    @pytest.mark.parametrize("seed,deck", CASES, ids=IDS)
    def test_regular_blind_losses_happen(self, seed, deck):
        rec = natural(seed, deck)
        assert any(not e.is_pvp for e in rec.lives), "natural scenario produced no regular-blind life loss"
        assert rec.m.done

    @pytest.mark.parametrize("seed,deck", CASES, ids=IDS)
    def test_comeback_money_lands_at_the_next_cash_out_once_per_loss(self, seed, deck):
        """Every life loss is followed (at that player's next Cash Out) by +4 x cumulative lives
        lost, exactly once; a Cash Out with no new loss pays no comeback."""
        for scen in ALL_SCENARIOS:
            rec = scen(seed, deck)
            for p in (0, 1):
                cos = sorted((c for c in rec.cashouts if c.player == p), key=lambda c: c.step)
                los = sorted(lives_of(rec, p), key=lambda e: e.step)
                paid = []
                last_step = -1
                for c in cos:
                    new_losses = [e for e in los if last_step < e.step <= c.step]
                    cum = sum(1 for e in los if e.step <= c.step)
                    if new_losses:
                        assert c.comeback_pending and c.comeback_expected == MLB_COMEBACK_PER_LIFE * cum, (scen.__name__, p, c, new_losses)
                        assert len(new_losses) == 1, (scen.__name__, "two losses between cash outs", new_losses)
                        paid.append(c.comeback_expected)
                    else:
                        assert not c.comeback_pending and c.comeback_expected == 0, (scen.__name__, p, c)
                    last_step = c.step
                # every loss that was followed by a cash out of this player was paid exactly once
                followed = [e for e in los if any(c.step > e.step for c in cos)]
                assert len(paid) == len(followed), (scen.__name__, p, paid, followed)
                assert paid == [MLB_COMEBACK_PER_LIFE * (i + 1) for i in range(len(paid))], (scen.__name__, p, paid)

    @pytest.mark.parametrize("seed,deck", CASES, ids=IDS)
    def test_match_ends_at_zero_lives_never_at_ante_8(self, seed, deck):
        for scen in ALL_SCENARIOS:
            rec = scen(seed, deck)
            m = rec.m
            assert m.done and m.winner in (0, 1), scen.__name__
            loser, winner = m.games[1 - m.winner], m.games[m.winner]
            assert loser.lives == 0 and winner.lives > 0, (scen.__name__, loser.lives, winner.lives)
            assert all(g.state == State.GAME_OVER for g in m.games) and winner.match_won and not loser.match_won
            # the last life event is the loser's and ends the match (no further steps)
            last = max(rec.lives, key=lambda e: e.step)
            assert last.player == 1 - m.winner and last.lives_after == 0 and last.step == m.steps, (scen.__name__, last, m.steps)
            assert len(lives_of(rec, 1 - m.winner)) == MLB_STARTING_LIVES

    @pytest.mark.parametrize("seed,deck", CASES, ids=IDS)
    def test_endless_past_ante_8_and_tie_costs_nobody(self, seed, deck):
        rec = endless(seed, deck)
        m = rec.m
        assert m.done and m.winner == 1, [(x.ante, x.loser) for x in rec.pvp]
        # antes 2..8: exact ties (identical play on one seed) -> nobody loses, both cash out $5
        ties = [x for x in rec.pvp if x.tie]
        assert [x.ante for x in ties] == list(range(2, 9)), [(x.ante, x.loser, x.score0, x.score1) for x in rec.pvp]
        for x in ties:
            assert x.score0 == x.score1 > 0
            for p in (0, 1):
                co = [c for c in rec.cashouts if c.player == p and c.is_pvp and c.ante == x.ante]
                assert len(co) == 1 and co[0].blind_reward == MLB_NEMESIS_REWARD and co[0].comeback_expected == 0, co
        assert not [e for e in rec.lives if e.ante <= 8], "a life was lost at a tied Nemesis"
        # antes 9..12: P0 loses four Nemeses -> 0 lives at ante 12; the match did NOT end at ante 8
        losses = [x for x in rec.pvp if not x.tie]
        assert [(x.ante, x.loser) for x in losses] == [(9, 0), (10, 0), (11, 0), (12, 0)]
        assert all(g.ante == 12 for g in m.games)
        # blind targets past ante 8 follow the game's endless formula (scaled x2 under Plasma)
        for b in rec.blinds:
            if not b.is_pvp and b.ante >= 9:
                idx = {"Small": 0, "Big": 1}[b.blind]
                want = int(blind_base_chips(b.ante, idx, 1) * (2 if deck == "b_plasma" else 1))
                assert b.target == want and b.target > get_blind_amount(8), (b, want)
        assert any(b.ante >= 9 for b in rec.blinds)

    @pytest.mark.parametrize("seed,deck", CASES, ids=IDS)
    def test_early_end_rule_fires(self, seed, deck):
        """PvP scenario: P0 (one-card hands) exhausts its hands strictly behind while P1 still has
        hands -> the server ends the Nemesis at once, P1's remaining hands are forfeited (never
        played), P0 loses exactly one life."""
        rec = pvp(seed, deck)
        assert rec.pvp and all(x.loser == 0 for x in rec.pvp), [(x.ante, x.loser) for x in rec.pvp]
        early = 0
        for x in rec.pvp:
            assert x.hands_left0 == 0 and x.score0 < x.score1, x
            lb = [b for b in rec.blinds if b.player == 0 and b.is_pvp and b.ante == x.ante]
            wb = [b for b in rec.blinds if b.player == 1 and b.is_pvp and b.ante == x.ante]
            # the verdict step is the step at which the loser played its LAST hand (no gap)
            assert len(lb) == 1 and lb[0].step == x.step and lb[0].hands_used == STARTING_HANDS, (lb, x)
            if x.hands_left1 > 0:
                # EARLY END: the winner still had hands; they are forfeited (its record shows fewer
                # hands used).  Whether P1 still has a hand when P0 exhausts depends on the
                # canonical interleaving (who moved first) and on P1's discards.
                assert x.early_end, x
                early += 1
                if x.step != rec.m.steps:        # (the match-ending verdict cuts the winner's blind)
                    assert len(wb) == 1 and wb[0].step == x.step and wb[0].hands_used < STARTING_HANDS, (wb, x)
            else:
                assert not x.early_end and x.hands_left1 == 0, x
                if x.step != rec.m.steps:
                    assert len(wb) == 1 and wb[0].hands_used == STARTING_HANDS, (wb, x)
        assert early >= 1, [(x.ante, x.hands_left1) for x in rec.pvp]
        assert [(x.ante, x.loser) for x in rec.pvp] == [(2, 0), (3, 0), (4, 0), (5, 0)]
        assert rec.m.done and rec.m.winner == 1


# ---------------------------------------------------------------------------------------------
# 2. money
# ---------------------------------------------------------------------------------------------

class TestMoney:
    @pytest.mark.parametrize("seed,deck", CASES, ids=IDS)
    def test_nemesis_pays_5_win_or_lose_no_hand_money_interest_normal(self, seed, deck):
        """No jokers / tags / enhanced cards in these scenarios, so the Cash Out delta is exactly
        reward + unused-hand money + interest + comeback."""
        seen_winner_with_hands_left = False
        for scen in (pvp, endless):
            rec = scen(seed, deck)
            for c in rec.cashouts:
                if not c.is_pvp:
                    continue
                assert c.blind_reward == MLB_NEMESIS_REWARD, c
                assert c.hand_money_expected == 0, c
                assert c.interest_expected == min(c.dollars_before // 5, 5), c
                assert c.dollars_after - c.dollars_before == c.blind_reward + c.interest_expected + c.comeback_expected, c
                if c.hands_left > 0:
                    seen_winner_with_hands_left = True     # early-ended winner: hands left, NOT paid
        assert seen_winner_with_hands_left

    @pytest.mark.parametrize("seed,deck", CASES, ids=IDS)
    def test_failed_regular_blind_pays_its_reward(self, seed, deck):
        rec = natural(seed, deck)
        failed = [c for c in rec.cashouts if not c.is_pvp and not c.won]
        assert failed, "no failed regular blind reached Cash Out"
        for c in rec.cashouts:
            if c.is_pvp:
                continue
            want = c.blind_reward + c.hand_money_expected + c.interest_expected + c.comeback_expected
            assert c.dollars_after - c.dollars_before == want, c
            if not c.won:
                assert c.hands_left == 0 and c.hand_money_expected == 0 and c.comeback_expected > 0, c
                assert c.blind_reward == {"Small": 3, "Big": 4, "Boss": 5}[c.blind], c


# ---------------------------------------------------------------------------------------------
# 3. queue alignment
# ---------------------------------------------------------------------------------------------

def _ante_of_key(key: str):
    """The ante a key is scoped to: trailing digits, or the digits inside 'rarity<a><app>'."""
    m = D._APPENDED_RE.match(key)
    if m:
        if m.group("ante"):
            return int(m.group("ante"))
        digits = m.group("prefix")[len("rarity"):] if m.group("prefix").startswith("rarity") else ""
        return int(digits) if digits else None
    m = D._ANTE_ONLY_RE.match(key)
    return int(m.group("ante")) if m else None


def _consumption(rec: D.MatchRecorder, p: int, ante: int, upto: int, include_current_rerolls: bool):
    """(shelf slots drawn, packs opened) by player p in ``ante`` through visit ordinal ``upto``."""
    slots = packs = 0
    for v in rec.visits[p]:
        if v.ante != ante or v.ordinal > upto:
            continue
        if v.ordinal == upto and not include_current_rerolls:
            slots += len(v.shelves[0])
        else:
            slots += sum(len(sh) for sh in v.shelves)
            packs += len(v.opened)
    return slots, packs


def alignment_violations(rec: D.MatchRecorder) -> list:
    """Apply the GATE_NOTES §3 rules to every differing RNG key at every shop-visit ordinal
    (entry and exit).  Returns the violations (empty = aligned)."""
    bad = []
    seed = rec.seed
    n = min(len(rec.visits[0]), len(rec.visits[1]))
    for i in range(n):
        va, vb = rec.visits[0][i], rec.visits[1][i]
        assert (va.ante, va.after_blind) == (vb.ante, vb.after_blind), (va.ante, va.after_blind, vb.ante, vb.after_blind)
        for tag, s0, s1, cur in (("entry", va.rng_entry, vb.rng_entry, False), ("exit", va.rng_exit, vb.rng_exit, True)):
            if not s0 or not s1:
                continue
            for key, cls, p0, p1 in D.diff_rng(seed, s0, s1):
                where = f"visit#{i} ante{va.ante} {tag} {key}[{cls}] P1@{p0} P2@{p1}"
                if p0 is None or p1 is None:
                    bad.append(where + " (position not found)")
                    continue
                a = _ante_of_key(key)
                if cls in ("SHARED", "VOUCHER", "UNKNOWN"):
                    bad.append(where)
                elif cls == "OWN_SHOP":
                    c0 = _consumption(rec, 0, a, i, cur)
                    c1 = _consumption(rec, 1, a, i, cur)
                    if key.startswith("cdt"):
                        if (p0, p1) != (c0[0], c1[0]):
                            bad.append(where + f" cdt must equal slots drawn {c0[0]}/{c1[0]}")
                    elif c0[0] == c1[0]:
                        bad.append(where + f" equal slot counts ({c0[0]}) but positions differ")
                    elif (p0 > p1) != (c0[0] > c1[0]):
                        bad.append(where + f" direction contradicts slots drawn {c0[0]}/{c1[0]}")
                elif cls == "OWN_PACK":
                    c0 = _consumption(rec, 0, a, i, cur)
                    c1 = _consumption(rec, 1, a, i, cur)
                    if (c0[1] == 0 and p0 != 0) or (c1[1] == 0 and p1 != 0):
                        bad.append(where + f" pack key stepped without a pack opened ({c0[1]}/{c1[1]})")
                elif cls == "OWN_ANY":
                    c0 = _consumption(rec, 0, a, i, cur)
                    c1 = _consumption(rec, 1, a, i, cur)
                    hi = 0 if p0 > p1 else 1
                    ch, cl = (c0, c1) if hi == 0 else (c1, c0)
                    owned = rec.visits[hi][i].owned_entry
                    if not (ch[0] > cl[0] or ch[1] > 0 or owned):
                        bad.append(where + " no own consumption explains it")
                elif cls == "OWN_RESAMPLE":
                    base = D._RESAMPLE_RE.match(key).group("pool")
                    if D.classify_key(base) not in ("OWN_SHOP", "OWN_PACK", "OWN_ANY", "PER_PLAYER"):
                        bad.append(where + " resample of a shared pool")
                elif cls == "PER_PLAYER":
                    pass
                else:
                    bad.append(where + " unclassified")
    return bad


def _pool_of(item: dict):
    if item.get("set") == "Joker":
        return ("Joker", item.get("rarity"))
    return (item.get("set"),)


def shelf_diff_unexplained(a_items, b_items, a_owned, b_owned) -> list:
    """Slot-by-slot: equal, or the other player's item is owned by this one (in-place resample),
    or -- second order -- an earlier slot on this shelf was such a resample and both items are
    from the same pool (the `_resample` side stream is offset by one step)."""
    out = []
    if len(a_items) != len(b_items):
        return [("slot count", len(a_items), len(b_items))]
    offset = False
    for i, (x, y) in enumerate(zip(a_items, b_items)):
        if PC.item_sig(x, True) == PC.item_sig(y, True):
            continue
        if (y.get("key") in a_owned) or (x.get("key") in b_owned):
            offset = True
            continue
        if offset and _pool_of(x) == _pool_of(y):
            continue
        out.append((i, x.get("key"), y.get("key")))
    return out


class TestAlignment:
    @pytest.mark.parametrize("seed,deck", CASES, ids=IDS)
    def test_every_differing_rng_key_is_explained(self, seed, deck):
        rec = shoppers(seed, deck)
        assert min(len(rec.visits[0]), len(rec.visits[1])) >= 8, "match too short to exercise the shops"
        # the two players really did shop differently
        assert sum(v.rerolls for v in rec.visits[1]) > 0 and sum(len(v.opened) for v in rec.visits[0]) > 0
        assert sum(v.rerolls for v in rec.visits[0]) == 0 and sum(len(v.opened) for v in rec.visits[1]) == 0
        bad = alignment_violations(rec)
        assert not bad, "\n".join(bad[:40])
        # the SHARED streams are present on both sides and at the same position (not just absent)
        for i in range(min(len(rec.visits[0]), len(rec.visits[1]))):
            va, vb = rec.visits[0][i], rec.visits[1][i]
            a = va.ante
            keys = ["boss", "shuffle", "Voucher0", f"Tag{a}", f"cashout{a}", f"shop_pack{a}", f"idol{a}", f"cas{a}"]
            if va.after_blind in ("Small", "Big"):
                keys.append(f"nr{a}")        # the post-boss shop precedes ante a's first blind
            for key in keys:
                assert key in va.rng_exit and key in vb.rng_exit, (i, key)
                assert va.rng_exit[key] == vb.rng_exit[key], (i, key)

    @pytest.mark.parametrize("seed,deck", CASES, ids=IDS)
    def test_classes_actually_exercised(self, seed, deck):
        """Guard: the diff is not vacuous -- own-shop, own-pack and own-resample differences all
        occur, and PER_PLAYER differences only come from named effect keys."""
        rec = shoppers(seed, deck)
        seen = set()
        for i in range(min(len(rec.visits[0]), len(rec.visits[1]))):
            va, vb = rec.visits[0][i], rec.visits[1][i]
            for key, cls, _, _ in D.diff_rng(rec.seed, va.rng_exit, vb.rng_exit):
                seen.add(cls)
        assert {"OWN_SHOP", "OWN_PACK", "OWN_RESAMPLE"} <= seen, seen

    @pytest.mark.parametrize("seed,deck", CASES, ids=IDS)
    def test_first_shelf_of_every_ante_identical_modulo_own_collection(self, seed, deck):
        rec = shoppers(seed, deck)
        by_ante = {}
        for p in (0, 1):
            for v in rec.visits[p]:
                by_ante.setdefault(v.ante, [None, None])
                if by_ante[v.ante][p] is None:
                    by_ante[v.ante][p] = v
        checked = 0
        for ante, (va, vb) in sorted(by_ante.items()):
            if va is None or vb is None:
                continue
            assert va.after_blind == vb.after_blind
            un = shelf_diff_unexplained(va.shelves[0], vb.shelves[0], set(va.owned_entry), set(vb.owned_entry))
            assert not un, (ante, un, va.shelves[0], vb.shelves[0], va.owned_entry, vb.owned_entry)
            assert va.voucher == vb.voucher, (ante, va.voucher, vb.voucher)
            assert va.packs == vb.packs, (ante, va.packs, vb.packs)
            assert va.boss_blind == vb.boss_blind and va.blind_tags == vb.blind_tags, (ante, va.boss_blind, vb.boss_blind)
            checked += 1
        assert checked >= 4
        # and at EVERY visit the voucher / packs / shadow boss / tags agree (SHARED streams)
        for i in range(min(len(rec.visits[0]), len(rec.visits[1]))):
            va, vb = rec.visits[0][i], rec.visits[1][i]
            assert (va.voucher, va.packs, va.boss_blind, va.blind_tags) == (vb.voucher, vb.packs, vb.boss_blind, vb.blind_tags), i

    @pytest.mark.parametrize("seed,deck", CASES, ids=IDS)
    def test_absolute_queue_positions_match_own_consumption(self, seed, deck):
        """Independent of the diff: for BOTH players at every visit, 'cdt<a>' has been stepped
        exactly (shelf slots drawn in ante a) times and 'shop_pack<a>' exactly 2 x (shops visited
        in ante a) times, minus one in ante 1 (the run's first pack is the forced Buffoon that
        consumes nothing)."""
        rec = shoppers(seed, deck)
        checked = 0
        for p in (0, 1):
            for v in rec.visits[p]:
                for st, cur in ((v.rng_entry, False), (v.rng_exit, True)):
                    if not st:
                        continue
                    slots, _ = _consumption(rec, p, v.ante, v.ordinal, cur)
                    visits_in_ante = sum(1 for w in rec.visits[p] if w.ante == v.ante and w.ordinal <= v.ordinal)
                    assert D.key_position(rec.seed, f"cdt{v.ante}", st[f"cdt{v.ante}"]) == slots, (p, v.ordinal, cur)
                    want_packs = 2 * visits_in_ante - (1 if v.ante == 1 else 0)
                    assert D.key_position(rec.seed, f"shop_pack{v.ante}", st[f"shop_pack{v.ante}"]) == want_packs, (p, v.ordinal)
                    checked += 1
        assert checked >= 16

    def test_alignment_check_has_teeth(self):
        """Perturbing one SHARED key / one OWN_SHOP key in a recorded snapshot must be reported."""
        import copy
        rec = shoppers(SEEDS[0], "b_red")
        assert not alignment_violations(rec)
        v = rec.visits[1][3]
        saved = copy.deepcopy(v.rng_exit)
        try:
            v.rng_exit["boss"] = D.RCORE.lcg_step(v.rng_exit["boss"])          # one extra boss draw
            assert any("boss[SHARED]" in x for x in alignment_violations(rec))
            v.rng_exit = copy.deepcopy(saved)
            a = v.ante
            v.rng_exit[f"cdt{a}"] = D.RCORE.lcg_step(v.rng_exit[f"cdt{a}"])  # one phantom shelf slot
            assert any(f"cdt{a}[OWN_SHOP]" in x and "cdt must equal" in x for x in alignment_violations(rec))
            v.rng_exit = copy.deepcopy(saved)
            v.rng_exit["Voucher0"] = D.RCORE.lcg_step(v.rng_exit["Voucher0"])
            assert any("Voucher0[VOUCHER]" in x for x in alignment_violations(rec))
        finally:
            v.rng_exit = saved
        assert not alignment_violations(rec)

    @pytest.mark.parametrize("seed", SEEDS[:5])
    def test_voucher_stream_shared_until_a_pair_is_completed(self, seed):
        """MLB vouchers come from the run-global culled 'Voucher0' stream.  Buying does not step
        it: both players keep the same position and are offered the same voucher PAIR every
        ante (the buyer sees the upgrade where it owns the base).  The stream diverges only
        when a player owns BOTH tiers of a pair: that pair collapses to UNAVAILABLE for them and
        they redraw on the same stream (GATE_NOTES §4)."""
        rec = voucher_match(seed)
        pair = {v["key"]: i // 2 for i, v in enumerate(P.VOUCHERS)}
        n = min(len(rec.visits[0]), len(rec.visits[1]))
        assert n >= 12
        diverged_at = None
        for i in range(n):
            va, vb = rec.visits[0][i], rec.visits[1][i]
            p0 = D.key_position(seed, "Voucher0", va.rng_entry["Voucher0"])
            p1 = D.key_position(seed, "Voucher0", vb.rng_entry["Voucher0"])
            owned = rec.m.games[0].run_state.used_vouchers if i == n - 1 else None
            bought_so_far = {b["key"] for v in rec.visits[0][:i + 1] for b in v.bought if b["kind"] == "voucher"}
            complete_pairs = {pair[k] for k in bought_so_far if all(pair[x] == pair[k] and x in bought_so_far for x in
                                                                      (P.VOUCHERS[2 * pair[k]]["key"], P.VOUCHERS[2 * pair[k] + 1]["key"]))}
            if p0 == p1:
                assert diverged_at is None, (i, p0, p1)
                # same pair offered (or the buyer's slot is empty because it bought this ante's voucher)
                if va.voucher is not None and vb.voucher is not None:
                    assert pair[va.voucher] == pair[vb.voucher], (i, va.voucher, vb.voucher)
                    if va.voucher != vb.voucher:
                        assert vb.voucher in bought_so_far, (i, va.voucher, vb.voucher, bought_so_far)
            else:
                assert p0 > p1, (i, p0, p1)
                if diverged_at is None:
                    diverged_at = i
                    assert complete_pairs, (i, "stream diverged without a completed pair", bought_so_far)
        assert bought_so_far, "the buyer never bought a voucher"


# ---------------------------------------------------------------------------------------------
# 4. vanilla unchanged
# ---------------------------------------------------------------------------------------------

class _RulesetDriver(EP.EngineDriver):
    """engine_parity's scripted driver on a ``ruleset`` game; also records what the Boss slot
    actually shows when it is played."""

    def __init__(self, game_mod, seed: str, ruleset: str):
        super().__init__(game_mod, seed)
        self.g = game_mod.BalatroGame(seed=seed, ruleset=ruleset)
        self.ante_info = {}
        self._record_ante_start(1)
        self.boss_shown = {}

    def win_blind(self):
        g = self.g
        if g.current_blind.kind == "Boss":
            self.boss_shown[g.ante] = (g.current_blind.boss_key, g.current_blind.is_pvp)
        super().win_blind()


def _rank(item: dict):
    return _pool_of(item)


def mlb_vs_vanilla_unexplained(seed: str, antes: int = 8, rerolls: int = 5) -> tuple:
    """Drive the same scripted policy on a vanilla and an MLB single-player game; return the
    differences NOT explained by (a) the voucher (Voucher0 path), (b) a banned item replaced
    in place (+ the resample side stream offset that such a replacement causes later in the
    same ante / area / pool), (c) the ante >= 2 Boss slot."""
    dv = _RulesetDriver(GM, seed, "vanilla")
    dv.run(antes, EP.Policy(rerolls=rerolls))
    dm = _RulesetDriver(GM, seed, "mlb")
    dm.run(antes, EP.Policy(rerolls=rerolls))
    ov, om = dv.observed(), dm.observed()
    bad = []
    for a in range(1, antes + 1):
        ev, em = ov["antes"][str(a)], om["antes"][str(a)]
        # tags: differ only where vanilla drew the banned tag_boss (in-place resample), or the
        # Big tag after a banned Small tag (resample stream offset)
        tv, tm = ev["tags"], em["tags"]
        if tv["small"] != tm["small"] and tv["small"]["key"] not in MLB_BANNED_KEYS:
            bad.append((a, "tag_small", tv["small"], tm["small"]))
        if tv["big"] != tm["big"] and tv["big"]["key"] not in MLB_BANNED_KEYS and tv["small"]["key"] not in MLB_BANNED_KEYS:
            bad.append((a, "tag_big", tv["big"], tm["big"]))
        if a == 1 and ev["boss"] != em["boss"]:
            bad.append((a, "boss", ev["boss"], em["boss"]))
        if em["boss"]["key"] in MLB_BANNED_KEYS:
            bad.append((a, "banned shadow boss", em["boss"]))
        if em["voucher"]["key"] in MLB_BANNED_KEYS:
            bad.append((a, "banned voucher", em["voucher"]))
        # shop queue
        qv, qm = ev["shop_queue"], em["shop_queue"]
        if len(qv) != len(qm):
            bad.append((a, "queue length", len(qv), len(qm)))
        offset_pools = set()
        for i, (x, y) in enumerate(zip(qv, qm)):
            if PC.item_sig(x, True) == PC.item_sig(y, True):
                continue
            if x["key"] in MLB_BANNED_KEYS:
                offset_pools.add(_rank(x))
                continue
            if _rank(x) in offset_pools and _rank(x) == _rank(y):
                continue
            bad.append((a, f"queue[{i}]", x["key"], y["key"]))
        if any(y["key"] in MLB_BANNED_KEYS for y in qm):
            bad.append((a, "banned item on MLB shelf"))
        # packs
        for sv, sm in zip(ev["shops"], em["shops"]):
            if [p["key"] for p in sv["packs"]] != [p["key"] for p in sm["packs"]]:
                bad.append((a, "pack kinds", sv["packs"], sm["packs"]))
                continue
            offset_pools = set()
            for pv, pm in zip(sv["packs"], sm["packs"]):
                for i, (x, y) in enumerate(zip(pv["cards"] or [], pm["cards"] or [])):
                    if PC.item_sig(x, True) == PC.item_sig(y, True):
                        continue
                    if x["key"] in MLB_BANNED_KEYS:
                        offset_pools.add(_rank(x))
                        continue
                    if _rank(x) in offset_pools and _rank(x) == _rank(y):
                        continue
                    bad.append((a, f"pack {pv['key']}[{i}]", x["key"], y["key"]))
                if any(y["key"] in MLB_BANNED_KEYS for y in (pm["cards"] or [])):
                    bad.append((a, "banned item in MLB pack", pm["key"]))
    # the Boss slot: vanilla boss at ante 1, the Nemesis from ante 2; the 'boss' stream position equal
    for a, (k, is_pvp) in dm.boss_shown.items():
        if a == 1 and (is_pvp or k != dv.boss_shown[1][0]):
            bad.append((a, "ante-1 boss", dv.boss_shown[1], (k, is_pvp)))
        if a >= 2 and not (k == MLB_NEMESIS_KEY and is_pvp):
            bad.append((a, "boss slot not the Nemesis", k, is_pvp))
    sv, sm = dv.g.run_state.rng.snapshot()["state"], dm.g.run_state.rng.snapshot()["state"]
    if D.key_position(seed, "boss", sv["boss"]) != D.key_position(seed, "boss", sm["boss"]):
        bad.append(("boss stream position", sv["boss"], sm["boss"]))
    if D.key_position(seed, "shuffle", sv["shuffle"]) != D.key_position(seed, "shuffle", sm["shuffle"]):
        bad.append(("shuffle stream position",))
    n_voucher_diffs = sum(1 for a in range(1, antes + 1)
                          if ov["antes"][str(a)]["voucher"] != om["antes"][str(a)]["voucher"])
    return bad, n_voucher_diffs


class TestVanillaUnchanged:
    def test_engine_parity_126_of_126_through_ante_8(self):
        """The Phase-1 gate, invoked in-process: ruleset='vanilla' (the default) is byte-identical."""
        rc = EP.main(["--antes", "1-8", "--rerolls", "5", "--quiet"])
        assert rc == 0, "engine_parity --antes 1-8 --rerolls 5 is no longer 126/126 (see its output)"

    @pytest.mark.parametrize("seed", sorted(PC.load_ground_truth(None).keys()))
    def test_mlb_solo_differs_from_vanilla_only_in_vouchers_bans_and_the_nemesis(self, seed):
        bad, _ = mlb_vs_vanilla_unexplained(seed)
        assert not bad, bad[:20]

    def test_voucher_path_really_differs(self):
        """Guard that the comparison has teeth: MLB vouchers ('Voucher0') differ from vanilla's
        ('Voucher<ante>') on most seeds."""
        n = sum(mlb_vs_vanilla_unexplained(s)[1] for s in SEEDS)
        assert n >= len(SEEDS) * 4, n


# ---------------------------------------------------------------------------------------------
# 5. clone
# ---------------------------------------------------------------------------------------------

class TestClone:
    @pytest.mark.parametrize("seed,deck", CASES[:12], ids=IDS[:12])
    def test_clone_mid_match_replays_identically(self, seed, deck):
        """Play the shoppers' match until the first Nemesis is live for both, clone, then drive
        both copies with fresh copies of the same policies under the canonical order: the
        signature() trajectories and the verdict logs must be identical step for step."""
        m = MLBMatch(seed=seed, deck_key=deck)
        pols = [D.make_policy(OPENER), D.make_policy(REROLLER)]
        guard = 0
        while not (m.pvp_active and all(g.state == State.SELECTING_HAND for g in m.games)):
            p = m.current_player()
            assert p is not None and guard < 5000
            m.step(p, pols[p](m, p, m.legal_actions(p)))
            guard += 1
        c = m.clone()
        assert c.signature() == m.signature()
        pa = [D.make_policy(OPENER), D.make_policy(REROLLER)]
        pb = [D.make_policy(OPENER), D.make_policy(REROLLER)]
        steps = 0
        while not m.done:
            p, q = m.current_player(), c.current_player()
            assert p == q, (steps, p, q)
            m.step(p, pa[p](m, p, m.legal_actions(p)))
            c.step(q, pb[q](c, q, c.legal_actions(q)))
            assert m.signature() == c.signature(), f"trajectories diverged at step {steps}"
            steps += 1
            assert steps < 20000
        assert c.done and c.winner == m.winner and c.pvp_log == m.pvp_log and steps > 20
        assert m.pvp_log and len(m.pvp_log) >= 2

    def test_clone_is_isolated(self):
        m = MLBMatch(seed="7I4M53DL")
        pols = [D.make_policy(OPENER), D.make_policy(REROLLER)]
        for _ in range(30):
            p = m.current_player()
            m.step(p, pols[p](m, p, m.legal_actions(p)))
        before = m.signature()
        c = m.clone()
        for _ in range(40):
            q = c.current_player()
            if q is None:
                break
            c.step(q, pols[q](c, q, c.legal_actions(q)))
        assert m.signature() == before and c.signature() != before
