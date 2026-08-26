"""Phase 2 W1 — Major League Balatro rules (brief §1.1-1.5) and the two-player lockstep
coordinator (``balatro_sim.mlb_match.MLBMatch``).  Every rule is pinned to the mod /
server source in ``engine/MLB_NOTES.md``."""
from __future__ import annotations

import random

import pytest

from balatro_sim.game import BalatroGame, State, SHOWDOWN_BOSS_BLINDS
from balatro_sim.mlb_match import MLBMatch, DEFAULT_LIVES
from balatro_sim.constants import (
    MLB_BANNED_KEYS, MLB_BANNED_JOKERS, MLB_BANNED_VOUCHERS, MLB_BANNED_TAGS, MLB_BANNED_BLINDS,
    MLB_NEMESIS_KEY, MLB_NEMESIS_REWARD, MLB_COMEBACK_PER_LIFE, MLB_STARTING_LIVES,
    BLIND_REWARD, INTEREST_RATE, INTEREST_CAP, get_blind_amount, blind_base_chips,
)
from balatro_sim.game_keys import gen as GEN
from balatro_sim.hand_eval import evaluate_hand
from balatro_sim.env_v7 import HAND_PRIORITY
import balatro_sim.jokers  # noqa: F401  (registry)

SEED = "7I4M53DL"
SEEDS = ["7I4M53DL", "1558AXDL", "AAAAAAAA", "BALATRO1", "MLBTEST1", "ZZZZ1111"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (test-side drivers; nothing here lives in the engine)
# ─────────────────────────────────────────────────────────────────────────────

def _advance(m: MLBMatch, p: int):
    """Step player p through one non-decision transition (cash out / pack / shop exit)."""
    g = m.games[p]
    if g.state == State.ROUND_EVAL:
        m.step(p, {"type": "advance"})
    elif g.state == State.BOOSTER_OPEN:
        m.step(p, {"type": "skip_booster"})
    elif g.state == State.SHOP:
        m.step(p, {"type": "leave_shop"})
    else:
        raise RuntimeError(f"cannot auto-advance from {g.state}")


def win_blind(m: MLBMatch, p: int):
    """Play the current blind (BLIND_SELECT) and clear it without scoring (debug)."""
    g = m.games[p]
    assert g.state == State.BLIND_SELECT, g.state
    m.step(p, {"type": "play_blind"})
    assert g.state == State.SELECTING_HAND, g.state
    g.debug_win_blind()
    m.sync()
    _advance(m, p)                       # cash out -> shop
    while g.state in (State.BOOSTER_OPEN,):
        _advance(m, p)
    _advance(m, p)                       # leave shop -> next blind select
    while g.state == State.BOOSTER_OPEN:
        _advance(m, p)


def fail_blind(m: MLBMatch, p: int):
    """Play one card per hand: loses any regular blind (and scores very little at a Nemesis)."""
    g = m.games[p]
    if g.state == State.BLIND_SELECT:
        m.step(p, {"type": "play_blind"})
    while g.state == State.SELECTING_HAND and m.can_act(p):
        m.step(p, weakest_play(g))


def weakest_play(g: BalatroGame) -> dict:
    """The first legal play (one card; exactly five at The Psychic)."""
    return next(a for a in g.legal_actions() if a["type"] == "play")


def to_nemesis(m: MLBMatch, p: int, ante: int = 2):
    """Walk player p (debug wins, no shopping) to the BLIND_SELECT of ``ante``'s Nemesis."""
    g = m.games[p]
    while not (g.ante == ante and g.current_blind.is_pvp and g.state == State.BLIND_SELECT):
        if g.state == State.BLIND_SELECT:
            win_blind(m, p)
        else:
            _advance(m, p)
        assert g.ante <= ante, "overshot"


def greedy_hand(g: BalatroGame, discard: bool = False) -> dict:
    """Best hand type (then most chips) over every play subset — a cheap decent player.
    With ``discard=True`` it discards the non-scoring cards while the best hand is below
    Two Pair (and a hand remains)."""
    best = None
    for a in g.legal_actions():
        if a["type"] != "play":
            continue
        cards = [g.hand[i] for i in a["cards"]]
        ht, scoring = evaluate_hand(cards, **g.hand_eval_flags())
        key = (HAND_PRIORITY.get(ht, 0), sum(c.rank for c in scoring))
        if best is None or key > best[0]:
            best = (key, a, scoring)
    (pri, _), a, scoring = best
    if discard and pri < HAND_PRIORITY["Two Pair"] and g.discards_left > 0 and g.hands_left > 1:
        keep = {id(c) for c in scoring}
        junk = [i for i, c in enumerate(g.hand) if id(c) not in keep][:5]
        if junk:
            return {"type": "discard", "cards": junk}
    return a


def greedy_policy(m: MLBMatch, p: int, acts: list) -> dict:
    g = m.games[p]
    if g.state == State.SELECTING_HAND:
        return greedy_hand(g, discard=True)
    if g.state == State.SHOP:
        return {"type": "leave_shop"}
    if g.state == State.BOOSTER_OPEN:
        return {"type": "skip_booster"}
    if g.state == State.BLIND_SELECT:
        return {"type": "play_blind"}
    return acts[0]


def weak_policy(m: MLBMatch, p: int, acts: list) -> dict:
    g = m.games[p]
    if g.state == State.SELECTING_HAND:
        return weakest_play(g)
    return greedy_policy(m, p, acts)


def random_policy_factory(seed: int):
    rng = random.Random(seed)

    def pol(m, p, acts):
        return rng.choice(acts)
    return pol


def play_pvp(m: MLBMatch, policies):
    """Both players are at the Nemesis BLIND_SELECT: ready both, then play it out."""
    for p in (0, 1):
        m.step(p, {"type": "play_blind"})
    assert m.pvp_active
    while m.pvp_active and not m.done:
        p = m.current_player()
        m.step(p, policies[p](m, p, m.legal_actions(p)))


# ─────────────────────────────────────────────────────────────────────────────
# §1.1 ruleset: bans + vanilla untouched
# ─────────────────────────────────────────────────────────────────────────────

class TestRulesetAndBans:
    def test_ruleset_flag(self):
        assert BalatroGame(seed=SEED).ruleset == "vanilla"
        assert BalatroGame(seed=SEED, ruleset="mlb").ruleset == "mlb"
        with pytest.raises(ValueError):
            BalatroGame(seed=SEED, ruleset="blitz")

    def test_vanilla_has_no_mlb_state(self):
        g = BalatroGame(seed=SEED)
        assert not g.mlb and g.lives == 0 and not g.current_blind.is_pvp
        assert not (MLB_BANNED_KEYS & g.run_state.banned_keys)

    def test_attrition_bans_land_in_run_state(self):
        """MP.ApplyBans -> G.GAME.banned_keys: jokers, vouchers, tag_boss, bl_wall, bl_final_vessel."""
        g = BalatroGame(seed=SEED, ruleset="mlb")
        assert MLB_BANNED_KEYS <= g.run_state.banned_keys
        assert set(MLB_BANNED_JOKERS) == {"j_mr_bones", "j_luchador", "j_matador", "j_chicot"}
        assert set(MLB_BANNED_VOUCHERS) == {"v_hieroglyph", "v_petroglyph", "v_directors_cut", "v_retcon"}
        assert set(MLB_BANNED_TAGS) == {"tag_boss"}
        assert set(MLB_BANNED_BLINDS) == {"bl_wall", "bl_final_vessel"}

    def test_bans_are_in_place_resamples(self):
        """Phase-1 invariant: a banned key's slot is 'UNAVAILABLE' and resampled from a side
        stream, so when the vanilla draw is NOT banned the MLB draw is identical (tags, boss).
        Vouchers are NOT compared: under MLB they come from W2's run-global 'Voucher0'
        culled path (brief §1.6), which never yields a banned key either."""
        for seed in SEEDS:
            v = BalatroGame(seed=seed)
            m = BalatroGame(seed=seed, ruleset="mlb")
            assert m.run_state.ruleset == "mlb" and v.run_state.ruleset == "vanilla"
            assert m.run_state.current_round_voucher not in MLB_BANNED_KEYS
            for k in ("Small", "Big"):
                if v.blind_tags[k] != "tag_boss":
                    assert m.blind_tags[k] == v.blind_tags[k]
                else:
                    assert m.blind_tags[k] != "tag_boss"
            if v.boss_blind not in MLB_BANNED_BLINDS:
                assert m.boss_blind == v.boss_blind
            else:
                assert m.boss_blind not in MLB_BANNED_BLINDS

    def test_banned_bosses_leave_the_eligible_set(self):
        st = GEN.RunState(SEED)
        st.banned_keys = set(MLB_BANNED_KEYS)
        assert "bl_wall" not in GEN.eligible_bosses(st)
        st.ante = 8
        elig = GEN.eligible_bosses(st)
        assert "bl_final_vessel" not in elig and elig and all(b in SHOWDOWN_BOSS_BLINDS for b in elig)

    def test_banned_jokers_never_generated(self):
        st = GEN.RunState(SEED)
        st.banned_keys = set(MLB_BANNED_KEYS)
        st.showman = True
        seen = {GEN.create_card(st, "Joker", area="shop", key_append="sho").key for _ in range(2000)}
        assert not (seen & set(MLB_BANNED_JOKERS))

    def test_banned_tag_never_drawn(self):
        st = GEN.RunState(SEED)
        st.banned_keys = set(MLB_BANNED_KEYS)
        assert all(GEN.next_tag(st) != "tag_boss" for _ in range(300))


# ─────────────────────────────────────────────────────────────────────────────
# §1.2 Nemesis blind
# ─────────────────────────────────────────────────────────────────────────────

class TestNemesisBlind:
    def test_ante_1_boss_is_vanilla(self):
        m = MLBMatch(seed=SEED)
        g = m.games[0]
        win_blind(m, 0); win_blind(m, 0)
        assert g.current_blind.kind == "Boss" and not g.current_blind.is_pvp
        assert g.current_blind.boss_key == g.boss_blind and g.boss_blind != MLB_NEMESIS_KEY
        assert g.legal_actions() == [{"type": "play_blind"}]

    def test_boss_slot_is_nemesis_from_ante_2(self):
        m = MLBMatch(seed=SEED)
        to_nemesis(m, 0, 2)
        b = m.games[0].current_blind
        assert b.is_pvp and b.is_boss and b.kind == "Boss" and b.boss_key == MLB_NEMESIS_KEY
        assert b.chips_target == 0 and not b.is_showdown and b.money_reward == MLB_NEMESIS_REWARD
        assert not m.games[0].can_reroll_boss()

    def test_pvp_start_round_is_configurable(self):
        m = MLBMatch(seed=SEED, pvp_start_round=1)
        win_blind(m, 0); win_blind(m, 0)
        assert m.games[0].current_blind.is_pvp and m.games[0].ante == 1

    def test_boss_stream_still_drawn_behind_the_nemesis(self):
        """The mod's reset_blinds runs the vanilla reset_blinds first: the 'boss' stream is
        drawn exactly as often as in single player (the draw itself can differ from ante 2
        on, because bl_wall leaves the candidate list -- that IS the real game's behaviour)."""
        for seed in SEEDS:
            v = BalatroGame(seed=seed)
            m = MLBMatch(seed=seed)
            g = m.games[0]
            assert g.boss_blind == v.boss_blind            # ante 1: bl_wall is not eligible yet
            for _ in range(3):
                v.step({"type": "play_blind"}); v.debug_win_blind(); v.step({"type": "advance"}); v.step({"type": "leave_shop"})
            to_nemesis(m, 0, 2)
            assert v.ante == 2 and g.ante == 2
            assert g.boss_blind not in MLB_BANNED_BLINDS and g.boss_blind != MLB_NEMESIS_KEY
            pos_v = v.run_state.rng.snapshot()["state"].get("boss")
            pos_m = g.run_state.rng.snapshot()["state"].get("boss")
            assert pos_v == pos_m                            # same number of 'boss' draws
            assert sum(g.run_state.bosses_used.values()) == sum(v.run_state.bosses_used.values()) == 2

    def test_nemesis_has_no_boss_effect(self):
        m = MLBMatch(seed=SEED)
        to_nemesis(m, 0, 2); to_nemesis(m, 1, 2)
        m.step(0, {"type": "play_blind"}); m.step(1, {"type": "play_blind"})
        g = m.games[0]
        assert g.state == State.SELECTING_HAND and g.hands_left == g.base_hands
        assert not any(c.debuffed or c.face_down for c in g.hand)
        assert g.discards_left == g.base_discards

    def test_nemesis_pays_5_even_at_a_showdown_ante(self):
        g = BalatroGame(seed=SEED, ruleset="mlb")
        g.ante = 8
        g.blind_idx = 2
        g._prepare_next_blind()
        assert g.current_blind.is_pvp and g.current_blind.money_reward == 5
        assert g.current_blind.is_showdown is False

    def test_target_is_opponent_live_score(self):
        m = MLBMatch(seed=SEED)
        to_nemesis(m, 0, 2); to_nemesis(m, 1, 2)
        m.step(0, {"type": "play_blind"}); m.step(1, {"type": "play_blind"})
        g0, g1 = m.games
        m.step(0, greedy_hand(g0))
        assert g0.chips_scored > 0
        assert g1.current_blind.chips_target == g0.chips_scored
        assert g1.pvp_opponent_hands == g0.hands_left == g0.base_hands - 1
        assert g0.current_blind.chips_target == g1.chips_scored == 0

    def test_no_win_check_at_the_nemesis(self):
        """Scoring above the target does NOT end the round: every hand is played."""
        m = MLBMatch(seed=SEED)
        to_nemesis(m, 0, 2); to_nemesis(m, 1, 2)
        m.step(0, {"type": "play_blind"}); m.step(1, {"type": "play_blind"})
        g0 = m.games[0]
        m.step(0, greedy_hand(g0))
        assert g0.chips_scored > g0.current_blind.chips_target
        assert g0.state == State.SELECTING_HAND and g0.hands_left == g0.base_hands - 1


# ─────────────────────────────────────────────────────────────────────────────
# §1.3 lockstep + PvP resolution (server rule)
# ─────────────────────────────────────────────────────────────────────────────

class TestLockstep:
    def test_independent_until_the_nemesis(self):
        m = MLBMatch(seed=SEED)
        to_nemesis(m, 0, 2)
        g0, g1 = m.games
        assert g0.ante == 2 and g1.ante == 1 and g1.blind_idx == 0
        assert m.current_player() == 0 or m.current_player() == 1
        assert 0 in m.actors() and 1 in m.actors()

    def test_ready_waits_for_the_opponent(self):
        m = MLBMatch(seed=SEED)
        to_nemesis(m, 0, 2)
        m.step(0, {"type": "play_blind"})
        g0 = m.games[0]
        assert g0.pvp_ready and g0.state == State.BLIND_SELECT and not g0.pvp_started
        assert m.legal_actions(0) == [] and not m.can_act(0)
        assert m.current_player() == 1 and m.actors() == [1]
        assert not m.pvp_active

    def test_both_ready_starts_both(self):
        m = MLBMatch(seed=SEED)
        to_nemesis(m, 0, 2)
        m.step(0, {"type": "play_blind"})
        to_nemesis(m, 1, 2)
        assert not m.pvp_active
        m.step(1, {"type": "play_blind"})
        g0, g1 = m.games
        assert m.pvp_active and m.pvp_ante == 2
        assert g0.state == g1.state == State.SELECTING_HAND
        assert g0.pvp_started and g1.pvp_started and not g0.pvp_ready and not g1.pvp_ready
        assert [c.rank for c in g0.hand] == [c.rank for c in g1.hand]   # same seed, same 'nr2' deal

    def test_current_player_alternates_and_skips_waiters(self):
        m = MLBMatch(seed=SEED)
        to_nemesis(m, 0, 2); to_nemesis(m, 1, 2)
        play_pvp_start = lambda: [m.step(p, {"type": "play_blind"}) for p in (0, 1)]  # noqa: E731
        play_pvp_start()
        seen = []
        for _ in range(4):
            p = m.current_player()
            seen.append(p)
            m.step(p, greedy_hand(m.games[p]))
        assert seen == [0, 1, 0, 1]

    def test_nobody_gets_a_nemesis_ahead(self):
        """Player 0 readies at ante 2's Nemesis; however long player 1 takes, player 0 never
        moves and the Nemesis starts the moment player 1 readies at the SAME ante."""
        m = MLBMatch(seed=SEED)
        to_nemesis(m, 0, 2)
        m.step(0, {"type": "play_blind"})
        sig0 = m.games[0].state_signature()
        g1 = m.games[1]
        for _ in range(5):
            assert m.current_player() == 1 and m.games[0].ante == 2
            if g1.current_blind.is_pvp and g1.state == State.BLIND_SELECT:
                break
            win_blind(m, 1)
            assert not m.done and m.games[0].state_signature() == sig0
        assert g1.ante == 2 and m.games[0].state_signature() == sig0
        m.step(1, {"type": "play_blind"})
        assert m.pvp_active and m.pvp_ante == 2


class TestPvPResolution:
    def _setup(self, seed=SEED, lives=DEFAULT_LIVES):
        m = MLBMatch(seed=seed, lives=lives)
        to_nemesis(m, 0, 2); to_nemesis(m, 1, 2)
        return m

    def test_exhausted_player_waits(self):
        m = self._setup()
        m.step(0, {"type": "play_blind"}); m.step(1, {"type": "play_blind"})
        g0 = m.games[0]
        for _ in range(g0.base_hands):
            m.step(0, greedy_hand(g0))
        assert g0.state == State.PVP_WAIT and g0.hands_left == 0
        assert m.legal_actions(0) == [] and m.current_player() == 1
        assert m.pvp_active and g0.hand    # the hand is still held (Gold cards pay at Cash Out)

    def test_lower_score_loses_a_life(self):
        m = self._setup()
        play_pvp(m, [greedy_policy, weak_policy])
        g0, g1 = m.games
        assert g0.chips_scored > g1.chips_scored
        assert (g0.lives, g1.lives) == (DEFAULT_LIVES, DEFAULT_LIVES - 1)
        assert m.pvp_log == [(2, 1, g0.chips_scored, g1.chips_scored)]
        assert g0.state == State.ROUND_EVAL and g1.state == State.ROUND_EVAL
        assert not g0.pvp_started and not g1.pvp_started

    def test_early_end_when_exhausted_and_behind(self):
        """Server: (A.handsLeft < 1 and A.score < B.score) ends the PvP at once; B's
        remaining hands are forfeited (B goes straight to Cash Out)."""
        m = self._setup()
        m.step(0, {"type": "play_blind"}); m.step(1, {"type": "play_blind"})
        g0, g1 = m.games
        # weak player 1 burns all hands first
        for _ in range(g1.base_hands):
            m.step(1, {"type": "play", "cards": [0]})
        assert g1.state == State.PVP_WAIT and m.pvp_active
        # player 0 plays until ahead
        hands = 0
        while m.pvp_active:
            m.step(0, greedy_hand(g0)); hands += 1
        assert g0.chips_scored > g1.chips_scored
        assert g0.hands_left > 0 and hands < g0.base_hands      # early end, hands forfeited
        assert g0.state == State.ROUND_EVAL and g1.state == State.ROUND_EVAL
        assert g1.lives == DEFAULT_LIVES - 1 and g0.lives == DEFAULT_LIVES
        assert m.pvp_log[-1][1] == 1

    def test_no_early_end_while_the_behind_player_has_hands(self):
        m = self._setup()
        m.step(0, {"type": "play_blind"}); m.step(1, {"type": "play_blind"})
        g0, g1 = m.games
        for _ in range(g0.base_hands):
            m.step(0, greedy_hand(g0))
        assert g0.state == State.PVP_WAIT
        m.step(1, {"type": "play", "cards": [0]})
        assert m.pvp_active and g1.state == State.SELECTING_HAND    # behind but still has hands

    def test_tie_loses_nobody(self):
        """Server: `if (!host.score.equalTo(guest.score)) loser.loseLife()` — equal scores
        with both exhausted -> endPvP{lost:false} for both, no life lost."""
        m = self._setup()
        m.step(0, {"type": "play_blind"}); m.step(1, {"type": "play_blind"})
        g0, g1 = m.games
        # identical play on an identical deal -> identical scores
        while m.pvp_active:
            p = m.current_player()
            m.step(p, {"type": "play", "cards": [0, 1, 2, 3, 4]})
        assert g0.chips_scored == g1.chips_scored > 0
        assert g0.lives == g1.lives == DEFAULT_LIVES
        assert m.pvp_log == [(2, None, g0.chips_scored, g1.chips_scored)]
        assert g0.state == g1.state == State.ROUND_EVAL

    def test_outcome_is_independent_of_interleaving(self):
        """The server rule is monotone in the scores: who loses does not depend on the
        order the hands arrive in (only the early-end cut point does)."""
        def run(order_first):
            m = self._setup()
            m.step(0, {"type": "play_blind"}); m.step(1, {"type": "play_blind"})
            g = m.games
            first, second = order_first, 1 - order_first
            while m.pvp_active:
                p = first if m.can_act(first) else second
                if not m.can_act(p):
                    break
                m.step(p, greedy_hand(g[p]) if p == 0 else {"type": "play", "cards": [0]})
            return m.pvp_log[-1][1], m.games[0].lives, m.games[1].lives
        assert run(0) == run(1) == (1, DEFAULT_LIVES, DEFAULT_LIVES - 1)

    def test_zero_lives_at_the_nemesis_ends_the_match(self):
        m = self._setup(lives=1)
        play_pvp(m, [greedy_policy, weak_policy])
        g0, g1 = m.games
        assert m.done and m.winner == 0
        assert g1.lives == 0 and g1.state == State.GAME_OVER and not g1._obs().won
        assert g0.state == State.GAME_OVER and g0.match_won and g0._obs().won
        assert m.legal_actions(0) == [] and m.legal_actions(1) == [] and m.current_player() is None

    def test_step_after_done_is_a_noop(self):
        m = self._setup(lives=1)
        play_pvp(m, [greedy_policy, weak_policy])
        sig = m.signature()
        m.step(0, {"type": "play_blind"})
        assert m.signature() == sig


# ─────────────────────────────────────────────────────────────────────────────
# §1.4 money at blind end + failed-blind-proceeds + lives
# ─────────────────────────────────────────────────────────────────────────────

class TestFailedRegularBlind:
    def test_failed_small_blind_proceeds_and_costs_a_life(self):
        m = MLBMatch(seed=SEED)
        g = m.games[0]
        fail_blind(m, 0)
        assert g.state == State.ROUND_EVAL            # not GAME_OVER: blind.chips = -1 trick
        assert g.chips_scored < blind_base_chips(1, 0)
        assert g.lives == DEFAULT_LIVES - 1 and g.life_lost_this_round
        assert g.comeback_bonus == 1 and not g.comeback_bonus_given
        _advance(m, 0)
        assert g.state == State.SHOP and g.ante == 1
        _advance(m, 0)
        assert g.current_blind.kind == "Big"           # the run proceeds as if defeated

    def test_failed_blind_pays_reward_and_interest_no_hand_money(self):
        m = MLBMatch(seed=SEED)
        g = m.games[0]
        fail_blind(m, 0)
        before = g.dollars
        interest = min(before // INTEREST_RATE, INTEREST_CAP)
        _advance(m, 0)
        # Small $3 + interest + comeback 4 x 1; hands_left == 0 so no hand row
        assert g.dollars == before + BLIND_REWARD["Small"] + interest + MLB_COMEBACK_PER_LIFE

    def test_failed_boss_eases_the_ante(self):
        m = MLBMatch(seed=SEED)
        g = m.games[0]
        win_blind(m, 0); win_blind(m, 0)
        assert g.current_blind.kind == "Boss" and not g.current_blind.is_pvp
        fail_blind(m, 0)
        assert g.state == State.ROUND_EVAL and g.lives == DEFAULT_LIVES - 1
        _advance(m, 0)
        assert g.ante == 2 and g.state == State.SHOP

    def test_vanilla_failed_blind_is_game_over(self):
        g = BalatroGame(seed=SEED)
        g.step({"type": "play_blind"})
        while g.state == State.SELECTING_HAND:
            g.step({"type": "play", "cards": [0]})
        assert g.state == State.GAME_OVER

    def test_one_life_per_round_blocker(self):
        g = BalatroGame(seed=SEED, ruleset="mlb")
        g.step({"type": "play_blind"})
        assert g.lose_life() and g.lives == 3
        assert not g.lose_life() and g.lives == 3           # Client.roundLivesBlocker
        g.debug_win_blind(); g.step({"type": "advance"}); g.step({"type": "leave_shop"})
        g.step({"type": "play_blind"})                      # newRound resets the blocker
        assert g.lose_life() and g.lives == 2

    def test_last_life_lost_at_a_regular_blind_is_game_over_without_cash_out(self):
        m = MLBMatch(seed=SEED, lives=1)
        g0, g1 = m.games
        d = g0.dollars
        fail_blind(m, 0)
        assert g0.lives == 0 and g0.state == State.GAME_OVER and g0.dollars == d
        assert m.done and m.winner == 1 and g1.state == State.GAME_OVER and g1.match_won

    def test_first_to_zero_loses_even_if_both_are_failing(self):
        m = MLBMatch(seed=SEED, lives=1)
        fail_blind(m, 1)
        assert m.done and m.winner == 0

    def test_deck_out_with_no_hand_played_costs_a_life(self):
        """MP.handle_deck_out: 0 hands played + discards used + hand and deck empty ->
        the round ends (reward paid) and fail_round(1) takes a life."""
        g = BalatroGame(seed=SEED, ruleset="mlb")
        g.full_deck = g.full_deck[:8]                       # an 8-card deck: no draw pile
        g.step({"type": "play_blind"})
        assert len(g.hand) == 8 and not g.deck
        g.step({"type": "discard", "cards": [0, 1, 2, 3, 4]})
        assert g.state == State.SELECTING_HAND and len(g.hand) == 3
        g.step({"type": "discard", "cards": [0, 1, 2]})
        assert not g.hand and g.state == State.ROUND_EVAL
        assert g.lives == MLB_STARTING_LIVES - 1 and g.hands_left == 0
        d = g.dollars
        g.step({"type": "advance"})
        assert g.state == State.SHOP and g.dollars == d + BLIND_REWARD["Small"] + min(d // 5, 5) + 4

    def test_deck_out_after_a_hand_ends_the_round(self):
        g = BalatroGame(seed=SEED, ruleset="mlb")
        g.full_deck = g.full_deck[:8]
        g.step({"type": "play_blind"})
        g.step({"type": "play", "cards": [0, 1, 2, 3, 4]})
        assert g.state == State.SELECTING_HAND and len(g.hand) == 3
        g.step({"type": "discard", "cards": [0, 1, 2]})
        assert not g.hand and g.hands_left == 0
        assert g.state == State.ROUND_EVAL and g.lives == MLB_STARTING_LIVES - 1

    def test_vanilla_deck_out_unchanged(self):
        g = BalatroGame(seed=SEED)
        g.full_deck = g.full_deck[:8]
        g.step({"type": "play_blind"})
        g.step({"type": "discard", "cards": [0, 1, 2, 3, 4]})
        g.step({"type": "discard", "cards": [0, 1, 2]})
        assert g.state == State.SELECTING_HAND and g.lives == 0   # untouched pre-W1 behaviour


class TestPvPMoney:
    def _finished_pvp(self, policies, seed=SEED):
        m = MLBMatch(seed=seed)
        to_nemesis(m, 0, 2); to_nemesis(m, 1, 2)
        play_pvp(m, policies)
        return m

    def test_loser_and_winner_both_get_the_blind_reward(self):
        m = self._finished_pvp([greedy_policy, weak_policy])
        for p in (0, 1):
            g = m.games[p]
            assert g.state == State.ROUND_EVAL
            before = g.dollars
            interest = min(before // INTEREST_RATE, INTEREST_CAP)
            comeback = 0 if g.comeback_bonus_given else MLB_COMEBACK_PER_LIFE * g.comeback_bonus
            _advance(m, p)
            assert g.dollars == before + MLB_NEMESIS_REWARD + interest + comeback, p
            assert g.ante == 3 and g.state == State.SHOP

    def test_no_unused_hand_money_at_pvp(self):
        """The early-ended winner still has hands; game.toml:94-99 pays nothing for them."""
        m = MLBMatch(seed=SEED)
        to_nemesis(m, 0, 2); to_nemesis(m, 1, 2)
        m.step(0, {"type": "play_blind"}); m.step(1, {"type": "play_blind"})
        g0, g1 = m.games
        for _ in range(g1.base_hands):
            m.step(1, {"type": "play", "cards": [0]})
        while m.pvp_active:
            m.step(0, greedy_hand(g0))
        assert g0.hands_left > 0
        before = g0.dollars
        interest = min(before // INTEREST_RATE, INTEREST_CAP)
        _advance(m, 0)
        assert g0.dollars == before + MLB_NEMESIS_REWARD + interest

    def test_green_deck_discard_money_still_paid_at_pvp(self):
        """Only the hands row is patched; the Green Deck's money_per_discard row is vanilla."""
        m = MLBMatch(seed=SEED, deck_key="b_green")
        to_nemesis(m, 0, 2); to_nemesis(m, 1, 2)
        play_pvp(m, [greedy_policy, weak_policy])
        g = m.games[0]
        assert g.money_per_discard and g.no_interest
        before, disc = g.dollars, g.discards_left
        assert disc > 0
        _advance(m, 0)
        assert g.dollars == before + MLB_NEMESIS_REWARD + disc * g.money_per_discard


# ─────────────────────────────────────────────────────────────────────────────
# §1.5 comeback money
# ─────────────────────────────────────────────────────────────────────────────

class TestComeback:
    def test_initial_state(self):
        g = BalatroGame(seed=SEED, ruleset="mlb")
        assert g.comeback_bonus == 0 and g.comeback_bonus_given is True

    def test_cumulative_and_paid_once_per_loss(self):
        m = MLBMatch(seed=SEED)
        g = m.games[0]
        fail_blind(m, 0)                                        # Small lost: 1st life
        d = g.dollars; _advance(m, 0)
        assert g.dollars - d == BLIND_REWARD["Small"] + min(d // 5, 5) + 4 * 1
        assert g.comeback_bonus_given
        _advance(m, 0)
        win_blind(m, 0)                                         # Big won: nothing pending
        assert g.current_blind.kind == "Boss"
        d = g.dollars
        fail_blind(m, 0)                                        # Boss lost: 2nd life
        assert g.comeback_bonus == 2 and not g.comeback_bonus_given
        d = g.dollars; _advance(m, 0)
        assert g.dollars - d == BLIND_REWARD["Boss"] + min(d // 5, 5) + 4 * 2
        _advance(m, 0)
        win_blind(m, 0)                                         # ante 2 Small won: no comeback row
        assert g.comeback_bonus_given and g.comeback_bonus == 2

    def test_won_round_pays_no_comeback(self):
        m = MLBMatch(seed=SEED)
        g = m.games[0]
        m.step(0, {"type": "play_blind"}); g.debug_win_blind()
        d = g.dollars; _advance(m, 0)
        assert g.dollars - d == BLIND_REWARD["Small"] + g.hands_left + min(d // 5, 5)

    def test_pvp_loss_pays_at_that_cash_out(self):
        m = MLBMatch(seed=SEED)
        to_nemesis(m, 0, 2); to_nemesis(m, 1, 2)
        play_pvp(m, [greedy_policy, weak_policy])
        g = m.games[1]
        assert g.comeback_bonus == 1 and not g.comeback_bonus_given
        d = g.dollars; _advance(m, 1)
        assert g.dollars - d == MLB_NEMESIS_REWARD + min(d // 5, 5) + 4

    def test_comeback_is_after_interest(self):
        """Interest is computed on the pre-row balance; the comeback row never feeds it."""
        g = BalatroGame(seed=SEED, ruleset="mlb")
        g.dollars = 4
        g.step({"type": "play_blind"})
        g.lose_life()
        g.debug_win_blind()
        g.step({"type": "advance"})
        # 4 + $3 blind + $4 hands + interest(4//5 = 0) + comeback 4 -> 15; if comeback fed
        # interest it would be 16
        assert g.dollars == 4 + 3 + 4 + 0 + 4


class TestSoloMLBGame:
    """``BalatroGame(ruleset="mlb")`` without a match (``pvp_solo``): every client-side rule
    works stand-alone and the Nemesis starts / ends on its own (target 0, no opponent)."""

    def test_nemesis_starts_at_once_and_auto_ends(self):
        g = BalatroGame(seed=SEED, ruleset="mlb")
        assert g.pvp_solo
        while not (g.ante == 2 and g.current_blind.is_pvp):
            g.step({"type": "play_blind"}); g.debug_win_blind(); g.step({"type": "advance"}); g.step({"type": "leave_shop"})
        g.step({"type": "play_blind"})
        assert g.state == State.SELECTING_HAND and g.pvp_started and not g.pvp_ready
        while g.state == State.SELECTING_HAND:
            g.step({"type": "play", "cards": [0, 1, 2, 3, 4]})
        assert g.state == State.ROUND_EVAL and g.lives == MLB_STARTING_LIVES and not g.pvp_started
        d = g.dollars
        g.step({"type": "advance"})
        assert g.state == State.SHOP and g.ante == 3
        assert g.dollars == d + MLB_NEMESIS_REWARD + min(d // 5, 5)      # no hand money

    def test_reset_restores_mlb_state(self):
        g = BalatroGame(seed=SEED, ruleset="mlb")
        g.step({"type": "play_blind"}); g.lose_life()
        g.reset()
        assert g.ruleset == "mlb" and g.lives == MLB_STARTING_LIVES and g.comeback_bonus == 0
        assert g.comeback_bonus_given and MLB_BANNED_KEYS <= g.run_state.banned_keys


# ─────────────────────────────────────────────────────────────────────────────
# Endless
# ─────────────────────────────────────────────────────────────────────────────

class TestEndless:
    def test_no_ante_8_win(self):
        g = BalatroGame(seed=SEED, ruleset="mlb")
        g.ante = 8
        g.blind_idx = 2
        g._prepare_next_blind()
        g.step({"type": "play_blind"})
        g.debug_win_blind()
        g.step({"type": "advance"})
        assert g.state == State.SHOP and g.ante == 9
        g.step({"type": "leave_shop"})
        assert g.state == State.BLIND_SELECT and g.ante == 9 and not g._obs().won
        assert g.current_blind.chips_target == get_blind_amount(9)

    def test_vanilla_still_wins_at_ante_8(self):
        g = BalatroGame(seed=SEED)
        g.ante = 8
        g.blind_idx = 2
        g._prepare_next_blind()
        g.step({"type": "play_blind"})
        g.debug_win_blind()
        g.step({"type": "advance"})
        g.step({"type": "leave_shop"})
        assert g.state == State.GAME_OVER and g._obs().won


# ─────────────────────────────────────────────────────────────────────────────
# clone fidelity + determinism
# ─────────────────────────────────────────────────────────────────────────────

class TestClone:
    @staticmethod
    def _drive(m, n, policies):
        for _ in range(n):
            if m.done:
                break
            p = m.current_player()
            m.step(p, policies[p](m, p, m.legal_actions(p)))

    def test_clone_then_step_both_matches_replay(self):
        pol = [greedy_policy, greedy_policy]
        m = MLBMatch(seed=SEED)
        self._drive(m, 15, pol)
        k = m.clone()
        assert k.signature() == m.signature()
        self._drive(m, 25, pol); self._drive(k, 25, pol)
        assert k.signature() == m.signature()
        # an un-cloned replay from scratch agrees too
        r = MLBMatch(seed=SEED)
        self._drive(r, 40, pol)
        assert r.signature() == m.signature()

    def test_clone_is_isolated(self):
        m = MLBMatch(seed=SEED)
        to_nemesis(m, 0, 2); to_nemesis(m, 1, 2)
        m.step(0, {"type": "play_blind"}); m.step(1, {"type": "play_blind"})
        sig = m.signature()
        k = m.clone()
        assert k.games[0] is not m.games[0] and k.games[0].run_state is not m.games[0].run_state
        play_pvp_rest = [greedy_policy, weak_policy]
        while k.pvp_active:
            p = k.current_player()
            k.step(p, play_pvp_rest[p](k, p, k.legal_actions(p)))
        assert m.signature() == sig and m.pvp_active and not m.pvp_log
        assert k.pvp_log and k.games[1].lives == DEFAULT_LIVES - 1 and m.games[1].lives == DEFAULT_LIVES

    def test_clone_mid_pvp_keeps_targets_and_wait_state(self):
        m = MLBMatch(seed=SEED)
        to_nemesis(m, 0, 2); to_nemesis(m, 1, 2)
        m.step(0, {"type": "play_blind"}); m.step(1, {"type": "play_blind"})
        g0 = m.games[0]
        for _ in range(g0.base_hands):
            m.step(0, greedy_hand(g0))
        k = m.clone()
        assert k.games[0].state == State.PVP_WAIT
        assert k.games[1].current_blind.chips_target == g0.chips_scored
        assert k.pvp_active and k.pvp_ante == 2 and k.current_player() == 1

    def test_game_clone_copies_mlb_scalars(self):
        g = BalatroGame(seed=SEED, ruleset="mlb")
        g.step({"type": "play_blind"}); g.lose_life()
        c = g.clone()
        for f in ("ruleset", "mlb", "lives", "comeback_bonus", "comeback_bonus_given",
                  "life_lost_this_round", "pvp_ready", "pvp_solo", "pvp_started",
                  "pvp_opponent_score", "pvp_opponent_hands", "match_won", "pvp_start_round"):
            assert getattr(c, f) == getattr(g, f), f
        assert c.state_signature() == g.state_signature()


# ─────────────────────────────────────────────────────────────────────────────
# full-match smoke tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFullMatch:
    def test_random_players_terminate_at_zero_lives(self):
        m = MLBMatch(seed=SEED)
        st = m.play_out([random_policy_factory(1), random_policy_factory(2)], max_steps=20000)
        assert st.done and st.winner in (0, 1)
        loser = m.games[1 - st.winner]
        assert loser.lives == 0 and m.games[st.winner].lives > 0
        assert all(g.state == State.GAME_OVER for g in m.games)

    def test_random_match_is_deterministic(self):
        a = MLBMatch(seed=SEED); a.play_out([random_policy_factory(7), random_policy_factory(8)], 20000)
        b = MLBMatch(seed=SEED); b.play_out([random_policy_factory(7), random_policy_factory(8)], 20000)
        assert a.signature() == b.signature() and a.pvp_log == b.pvp_log

    def test_greedy_beats_weak(self):
        """A discard-aware greedy player (no shopping) outlives a one-card player: lives go
        on lost regular blinds, the match ends at 0 lives, never at ante 8."""
        wins = 0
        for seed in ("AAAAAAAA", "BALATRO1", "ZZZZ1111"):
            m = MLBMatch(seed=seed)
            st = m.play_out([greedy_policy, weak_policy], max_steps=20000)
            assert st.done
            assert m.games[1 - st.winner].lives == 0 and m.games[st.winner].lives > 0
            assert all(g.ante <= 8 for g in m.games)
            wins += (st.winner == 0)
        assert wins >= 2

    def test_lives_go_on_regular_blinds_and_pvp(self):
        """Greedy vs greedy on one seed reaches the Nemesis; lives are lost both ways."""
        m = MLBMatch(seed="AAAAAAAA")
        st = m.play_out([greedy_policy, greedy_policy], max_steps=20000)
        assert st.done and m.pvp_log, m.pvp_log
        assert any(e[1] is not None for e in m.pvp_log) or sum(g.lives for g in m.games) < 2 * DEFAULT_LIVES

    def test_shop_queues_stay_aligned_without_purchases(self):
        """Both players on one seed: with no buys / rerolls the shelf, voucher and boss draw of
        every shop are identical (each player's generation is independent but keyed the same)."""
        m = MLBMatch(seed=SEED)
        shelves = {0: [], 1: []}
        for p in (0, 1):
            g = m.games[p]
            for _ in range(4):
                m.step(p, {"type": "play_blind"}); g.debug_win_blind(); m.sync()
                if g.state == State.BLIND_SELECT:      # readied for a nemesis
                    break
                _advance(m, p)
                shelves[p].append(([(it.kind, it.key) for it in g.current_shop], g.boss_blind))
                _advance(m, p)
        assert shelves[0] == shelves[1] and len(shelves[0]) >= 3


# ── Phase 3 close: boss-ability rejection must respect MLB lives ─────────────────
# Found independently by P3-W2 (tournament) and P3-W4 (eval) — game.py's bl_hook /
# bl_eye / bl_mouth "hand rejected" branches set GAME_OVER on exhaustion without the
# `mlb` guard every other exhaustion path has.

class TestBossRejectionRespectsMLB:
    @staticmethod
    def _reject_until_exhausted(g: BalatroGame, boss_key: str):
        g.step({"type": "play_blind"})
        assert g.state == State.SELECTING_HAND
        g.current_blind.boss_key = boss_key
        # bl_eye rejects a repeated hand type; bl_mouth rejects any type but the first.
        g.played_hand_types_this_round = {"High Card"} if boss_key == "bl_eye" else {"Flush Five"}
        while g.state == State.SELECTING_HAND:
            g.step({"type": "play", "cards": [0]})      # single card = High Card, always rejected

    @pytest.mark.parametrize("boss_key", ["bl_eye", "bl_mouth"])
    def test_mlb_rejected_exhaustion_costs_a_life_and_proceeds(self, boss_key):
        g = BalatroGame(seed=SEED, ruleset="mlb")
        lives = g.lives
        self._reject_until_exhausted(g, boss_key)
        assert g.state == State.ROUND_EVAL, g.state
        assert g.lives == lives - 1
        assert g.hands_left == 0

    @pytest.mark.parametrize("boss_key", ["bl_eye", "bl_mouth"])
    def test_vanilla_rejected_exhaustion_is_game_over(self, boss_key):
        g = BalatroGame(seed=SEED)
        self._reject_until_exhausted(g, boss_key)
        assert g.state == State.GAME_OVER

    def test_mlb_rejected_exhaustion_on_last_life_is_game_over(self):
        g = BalatroGame(seed=SEED, ruleset="mlb")
        g.lives = 1
        self._reject_until_exhausted(g, "bl_eye")
        assert g.state == State.GAME_OVER and g.lives == 0


# ── Phase 4 close: no card-target consumable actions in the SHOP ─────────────────
# P4-W2 found `legal_actions()` in SHOP enumerating `use_consumable` against the
# previous blind's hand; `_use_consumable` no-ops on them -> a legal action that
# changes nothing (an MCTS agent looped on one for 20k steps).

class TestShopConsumableActionsHaveNoCardTargets:
    def test_shop_offers_only_target_free_consumable_use(self):
        g = BalatroGame(seed=SEED)
        g.step({"type": "play_blind"}); g.debug_win_blind(); g.step({"type": "advance"})
        assert g.state == State.SHOP
        assert len(g.hand) > 0, "precondition: previous blind's hand is still held"
        g.consumable_hand = ["c_strength", "c_talisman", "pl_mercury"]   # enhancement tarot, spectral, planet
        uses = [a for a in g.legal_actions() if a["type"] == "use_consumable"]
        assert uses, "consumables should still be usable in the shop"
        assert all(a["target_cards"] == [] for a in uses)
        assert len(uses) == 3

    def test_selecting_hand_still_offers_card_targets(self):
        g = BalatroGame(seed=SEED)
        g.step({"type": "play_blind"})
        g.consumable_hand = ["c_strength"]
        uses = [a for a in g.legal_actions() if a["type"] == "use_consumable"]
        assert any(a["target_cards"] for a in uses)
