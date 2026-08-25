"""
hand.py — the analytic hand player (Phase 5 rev 2, W3).

EV of every ``play`` / ``discard`` / ``use_consumable`` action of a ``SELECTING_HAND`` state,
horizon = the end of the current blind.  The maths is in ``EV_NOTES.md``; the short form:

``budget="fast"`` (pure analytic, the rollout policy)
    * **Candidates** are generated structurally (every n-of-a-kind, two pair, full house,
      flush, straight makeable from the hand, each alone and with the least valuable cards
      dumped alongside it; every single; every chase line: keep a suit, keep a straight
      window, keep a rank group, junk out the k worst cards) and filtered through
      ``game.legal_actions()``.  ~40 keep sets instead of the 436 legal subsets.
    * **Plays** are scored cheaply (the hand formula at the run's planet level, W0's
      ``cheap``) and the top ones exactly through the engine's side-effect-free
      ``HypotheticalScorer`` (jokers, editions, enhancements, held-in-hand effects).
    * **Draws** are valued against the REAL draw-pile composition with hypergeometric
      completion probabilities (Flush / Straight / Pair / Trips / Quads / Two Pair / Full
      House).  Only the composition is read — never the order (``tests/test_hand.py``
      permutes ``game.deck`` and pins the decision).
    * **The future** beyond the next draw is a per-blind cached tail ``G(need, h, d)`` =
      P(clear ``need`` chips with ``h`` hands and ``d`` discards) + beta·E[unused hands],
      a dynamic programme over rounds whose per-round score distributions come from
      sampled fresh hands of this deck (``BlindModel``).
    * Objective at a regular blind: P(clear) + beta·(hands unused) + gamma·(discards
      unused) + beta·(dollars the action EXTRACTS).  At a Nemesis (PvP) blind:
      0.5·P(score ≥ opponent) + 0.5·P(score > opponent) with the opponent's unplayed hands
      modelled symmetrically, and no extraction term (no money at a PvP blind).
    * **The extraction layer** (``EXTRACT_NOTES.md``) prices the per-action money procs in
      dollars — Gold seal, Lucky, Business Card, Golden Ticket, Rough Gem, Reserved
      Parking, Faceless, Mail-In Rebate, Trading Card, Delayed Gratification, To Do List,
      Purple seal → Tarot, Gold enhancement held to round end — plus a first-order value
      for cycling toward a held Tarot's targets, and generates the sandbag lines
      (dump the seals, play the proc cards) that the greedy objective could never emit.
      Every dollar is only banked when the tail DP still clears the blind with probability
      ``extract_min_clear`` after the line, so a sandbag never costs a life.

``budget="full"`` (Monte-Carlo expectimax)
    Top-K candidates by the fast scorer; each is stepped on ``n_worlds`` sampled worlds
    (``sampling.sample_world``: draw pile reshuffled, composition kept) and the blind is
    played out with the fast policy; the end-of-blind state is valued by ``value_fn`` (V)
    if given, else the analytic proxy.  Common random numbers: every candidate sees the
    same worlds.  No ``value_fn``: K=``full_top_k`` (5) x ``full_n_worlds`` (3) worlds,
    ~51-74 ms mean (EV_NOTES §2/§6).  A ``value_fn`` given (and no explicit ``top_k``/
    ``n_worlds`` override): K=``full_top_k_v`` (3) x ``full_n_worlds_v`` (8) worlds per
    EV_NOTES §8.3 -- measured ~163 ms mean / 144 ms p50 on this box (LEAF_NOTES.md §2),
    over its ~100 ms target; the V forward pass is ~35-45% of that (24 unbatched
    single-item calls/decision -- batching them is the natural next lever, LEAF_NOTES.md
    §2, not attempted here).

Side-effect freedom: every public function works on clones or the ``HypotheticalScorer``;
``state_signature()`` and ``run_state.rng`` are bit-identical before and after (tests).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from math import comb
from typing import Callable, Optional

import numpy as np

import _bootstrap  # noqa: F401  (fork guard: the mp/engine balatro_sim)
from _bootstrap import State
from balatro_sim.card_selection import HypotheticalScorer
from balatro_sim.constants import HAND_BASE, HAND_LEVEL_CHIPS, HAND_LEVEL_MULT, RANK_CHIPS
from balatro_sim.constants import INTEREST_CAP, INTEREST_RATE
from balatro_sim.hand_eval import evaluate_hand
from balatro_sim.round_cards import from_round_picks

from sampling import sample_world, world_rng, canonical_card_key

__all__ = [
    "HandConfig", "DEFAULT_HAND_CONFIG", "hand_ev", "rank_hand_actions", "best_hand_action",
    "HandAnalysis", "BlindModel", "blind_model_for", "end_of_blind_value", "play_out_blind",
    "estimate_clear_probability", "board_ratio", "opponent_final_atoms", "ProcBoard",
    "extraction_ev", "extraction_lines",
]


# ═══════════════════════════════════════════════════════════════════════════ config

@dataclass(frozen=True)
class HandConfig:
    """Knobs of the analytic hand player (defaults = what EV_NOTES.md recommends)."""
    beta_hand: float = 0.012      # value of one unused hand ($1) in P(clear) units
    gamma_discard: float = 0.002  # value of one unused discard (tie-break: keep resources)
    exact_top: int = 8            # HypotheticalScorer refinements per decision
    model_samples: int = 256      # fresh hands simulated per BlindModel
    model_atoms: int = 6          # atoms per per-round score distribution
    grid_ratio: float = 1.10      # geometric need grid of the tail DP
    max_discard_lines: int = 14   # chase lines considered per decision
    full_top_k: int = 5           # budget="full": candidates rolled out
    full_n_worlds: int = 3        # budget="full": worlds per candidate (common random numbers)
    full_top_k_v: int = 3         # budget="full" WITH value_fn: candidates rolled out (EV_NOTES §8.3)
    full_n_worlds_v: int = 8      # budget="full" WITH value_fn: worlds per candidate (fits the 100 ms budget)
    rollout_lite: bool = True     # rollouts inside "full" use the lite candidate set
    # ── the extraction / sandbag layer (EXTRACT_NOTES.md) ──────────────────────
    extract: bool = True          # price the per-action money procs at all
    tarot_value_dollars: float = 4.0   # what one created Tarot is worth (until V takes over)
    extract_min_clear: float = 0.90    # tail-DP safety gate: only bank money above this
    extract_lines: int = 6        # extra extraction candidate lines per decision (max)
    junk_dollars: float = 0.8     # keep-value units per dollar, for the junk ORDERINGS only
    tarot_cycle_fraction: float = 0.5  # fraction of a Tarot's value a good target unlocks
    interest_bonus: float = 0.16  # extra worth of a $ that still earns interest, see §2


DEFAULT_HAND_CONFIG = HandConfig()

_KIND_NAME = {2: "Pair", 3: "Three of a Kind", 4: "Four of a Kind", 5: "Five of a Kind"}
_STRAIGHT_WINDOWS: tuple = tuple([(14, 2, 3, 4, 5)] + [tuple(range(s, s + 5)) for s in range(2, 11)])
_WINDOW_MASKS: tuple = tuple((w, sum(1 << r for r in w)) for w in _STRAIGHT_WINDOWS)


# ═══════════════════════════════════════════════════════════════════════ combinatorics

@lru_cache(maxsize=65536)
def _hyper_tail(N: int, K: int, n: int, k: int) -> float:
    """P(X >= k) for X ~ Hypergeometric(N, K, n): n draws without replacement from N
    cards of which K are 'useful'."""
    if k <= 0:
        return 1.0
    if N <= 0 or n <= 0 or K <= 0 or k > min(K, n):
        return 0.0
    tot = comb(N, n)
    s = 0
    for j in range(k, min(K, n) + 1):
        s += comb(K, j) * comb(N - K, n - j)
    return s / tot


@lru_cache(maxsize=65536)
def _p_all_groups(N: int, n: int, sizes: tuple) -> float:
    """P(at least one card from EACH of the groups ``sizes``) in ``n`` draws from ``N``
    (inclusion–exclusion over the missed groups; disjoint groups)."""
    if not sizes:
        return 1.0
    if N <= 0 or n < len(sizes) or any(s <= 0 for s in sizes):
        return 0.0
    tot = comb(N, n)
    acc = 0.0
    L = len(sizes)
    for mask in range(1 << L):
        miss = 0
        bits = 0
        for i in range(L):
            if mask >> i & 1:
                miss += sizes[i]
                bits += 1
        rest = N - miss
        term = comb(rest, n) if rest >= n else 0
        acc += term if bits % 2 == 0 else -term
    return max(0.0, min(1.0, acc / tot))


def _hand_base(hand_type: str, levels: dict) -> tuple:
    chips, mult = HAND_BASE.get(hand_type, (5, 1))
    level = levels.get(hand_type, 1) if levels else 1
    if level > 1:
        chips += HAND_LEVEL_CHIPS.get(hand_type, 0) * (level - 1)
        mult += HAND_LEVEL_MULT.get(hand_type, 0) * (level - 1)
    return float(chips), float(mult)


def _card_chips(card) -> float:
    if card.debuffed:
        return 0.0
    return float(card.base_chips + getattr(card, "bonus_chips", 0))


def _popcount(x: int) -> int:
    return bin(x).count("1")


# ═════════════════════════════════════════════════════════════ money procs (extraction)
#
# Every coefficient below is a CLOSED-FORM expectation in dollars, verified against the
# engine's implementation and the Lua reference (see EXTRACT_NOTES.md §1 for the audit
# table with file:line for both sides).  Nothing here rolls RNG or touches ``game``.

_FACE_RANKS = frozenset((11, 12, 13))
#: Tarots that target cards in hand, and how many targets they consume.
_TAROT_TARGETS = {
    "c_magician": 2, "c_empress": 2, "c_heirophant": 2, "c_lovers": 2, "c_chariot": 2,
    "c_justice": 2, "c_devil": 2, "c_tower": 2,          # enhancement tarots
    "c_star": 3, "c_moon": 3, "c_sun": 3, "c_world": 3,  # suit conversion
    "c_strength": 2, "c_hanged_man": 2, "c_death": 2,
}
#: Suit each conversion tarot produces (consumables.TAROT_SUIT).
_TAROT_SUIT = {"c_star": "Diamonds", "c_moon": "Clubs", "c_sun": "Hearts", "c_world": "Spades"}
_TAROT_ENHANCEMENT = frozenset(("c_magician", "c_empress", "c_heirophant", "c_lovers",
                                "c_chariot", "c_justice", "c_devil", "c_tower"))


class ProcBoard:
    """Which per-action money procs THIS board can fire, and with what coefficient.

    Built once per ``HandAnalysis`` from ``game`` — jokers, the round's Mail-In rank, the
    free consumable slots and ``G.GAME.probabilities.normal`` (= ``2 ** owned Oops! All 6s``,
    engine ``jokers/base.sync_probabilities``; computed locally so nothing is mutated)."""

    __slots__ = ("p_business", "p_parking", "faceless", "ticket", "trading", "rough_gem",
                 "mail_rank", "delayed_grat", "todo_hand", "free_slots", "any_card_proc",
                 "any", "p_normal")

    def __init__(self, game):
        jokers = list(getattr(game, "jokers", ()) or ())
        keys = {j.key for j in jokers}
        p = float(2 ** sum(1 for j in jokers if j.key == "j_oops"))
        self.p_normal = p
        # `pseudorandom(k) < normal/odds` -> P = min(1, normal/odds)   (base.prob_roll)
        self.p_business = min(1.0, p / 2.0) if "j_business" in keys else 0.0
        self.p_parking = min(1.0, p / 2.0) if "j_reserved_parking" in keys else 0.0
        self.faceless = "j_faceless" in keys
        self.ticket = "j_ticket" in keys
        self.rough_gem = "j_rough_gem" in keys
        # Trading Card: first discard of the round, exactly 1 card -> $3 and destroy it
        # (card.lua:2802 `discards_used <= 0 and #full_hand == 1`).
        self.trading = any(j.key == "j_trading" and not j.state.get("used") for j in jokers)
        # Mail-In Rebate: $5 per discarded card of the round's rank (card.lua:2825-2835)
        self.mail_rank = 0
        if "j_mail" in keys:
            try:
                self.mail_rank = int(from_round_picks(getattr(game, "round_picks", None)).get("mail") or 0)
            except Exception:               # noqa: BLE001
                self.mail_rank = 0
        # Delayed Gratification: $2 per remaining discard IF none is used this round.
        self.delayed_grat = any(j.key == "j_delayed_grat" and not j.state.get("discarded")
                                for j in jokers)
        # To Do List: $4 when the played hand type matches the listed one (card.lua:2975)
        self.todo_hand = ""
        for j in jokers:
            if j.key == "j_todo_list":
                self.todo_hand = str(j.state.get("target") or "")
        self.free_slots = max(0, int(getattr(game, "consumable_slots", 0))
                              - len(getattr(game, "consumable_hand", ()) or ()))
        #: a per-CARD proc exists (so the keep/junk orderings must be proc-aware)
        self.any_card_proc = bool(self.p_business or self.p_parking or self.ticket
                                  or self.rough_gem or self.mail_rank)
        self.any = bool(self.any_card_proc or self.faceless or self.trading
                        or self.delayed_grat or self.todo_hand)

    @classmethod
    def empty(cls) -> "ProcBoard":
        """A board with no procs at all (``cfg.extract=False``): every coefficient 0."""
        self = cls.__new__(cls)
        for s in cls.__slots__:
            setattr(self, s, 0)
        self.p_normal = 1.0
        self.todo_hand = ""
        self.faceless = self.ticket = self.trading = self.rough_gem = False
        self.delayed_grat = False
        self.any_card_proc = self.any = False
        return self


_NO_PROCS = ProcBoard.empty()


# ═══════════════════════════════════════════════════════════════════════ deck composition

class DeckComp:
    """Composition (never order) of a pile of cards: counts by rank / suit, chip means."""
    __slots__ = ("total", "by_rank", "by_suit", "suit_chips", "avg_chips", "n_active",
                 "n_gold", "n_plain")

    def __init__(self, cards):
        by_rank: dict = {}
        by_suit: dict = {}
        suit_chips: dict = {}
        tot_chips = 0.0
        n_active = 0
        total = 0
        n_gold = 0
        n_plain = 0
        for c in cards:
            total += 1
            if c.enhancement == "Gold":
                n_gold += 1            # hold-to-round-end money (extraction layer)
            elif c.enhancement == "None":
                n_plain += 1           # a targetable card for an enhancement tarot
            if c.enhancement == "Stone":
                continue
            n_active += 1
            ch = _card_chips(c)
            tot_chips += ch
            by_rank[c.rank] = by_rank.get(c.rank, 0) + 1
            if c.enhancement == "Wild" and not c.debuffed:
                for s in ("Spades", "Hearts", "Clubs", "Diamonds"):
                    by_suit[s] = by_suit.get(s, 0) + 1
                    suit_chips[s] = suit_chips.get(s, 0.0) + ch
            else:
                by_suit[c.suit] = by_suit.get(c.suit, 0) + 1
                suit_chips[c.suit] = suit_chips.get(c.suit, 0.0) + ch
        self.total = total
        self.by_rank = by_rank
        self.by_suit = by_suit
        self.suit_chips = suit_chips
        self.n_active = n_active
        self.n_gold = n_gold
        self.n_plain = n_plain
        self.avg_chips = (tot_chips / n_active) if n_active else 0.0

    def suit_avg(self, suit: str) -> float:
        n = self.by_suit.get(suit, 0)
        return (self.suit_chips.get(suit, 0.0) / n) if n else self.avg_chips


# ═══════════════════════════════════════════════════════════════════════ the blind model

def _blind_key(game) -> tuple:
    """Cache key of the per-blind tail model: everything that changes the distribution of a
    fresh hand of THIS deck in cheap units (the full deck's composition, hand size, planet
    levels, Four Fingers, the hand / discard counts) — NOT the draw pile order, NOT the
    current hand, NOT the jokers (their multiplier is applied at lookup as ``ratio``)."""
    flags = game.hand_eval_flags() if hasattr(game, "hand_eval_flags") else {}
    return (tuple(sorted(canonical_card_key(c) for c in game.full_deck)),
            int(game.hand_size), tuple(sorted(game.planet_levels.items())),
            bool(flags.get("four_fingers")), _resource_caps(game))


def _resource_caps(game) -> tuple:
    """(h_max, d_max) of the tail grid: the blind's base counts, widened by 3 only when a
    joker (Burglar / Drunkard / Merry Andy) has pushed the live counts above them."""
    in_blind = game.state == State.SELECTING_HAND
    bh, bd = int(game.base_hands), int(game.base_discards)
    h = bh + 3 if (in_blind and int(game.hands_left) > bh) else bh
    d = bd + 3 if (in_blind and int(game.discards_left) > bd) else bd
    return max(1, h), max(0, d)


_RATIO_CACHE: dict = {}
_RATIO_CACHE_MAX = 256


def _board_sig(game) -> tuple:
    """What ``board_ratio`` may depend on, to cache it: the jokers (with their scaling
    state), Plasma, hand size, planet levels, and coarse deck-modification counts.  The
    exact deck COMPOSITION is deliberately not in the key -- adding one card moves the
    ratio negligibly, and keying on it would recompute for every standard-pack pick."""
    return (tuple((j.key, j.edition, repr(sorted(j.state.items(), key=lambda kv: str(kv[0]))))
                  for j in game.jokers),
            bool(getattr(game, "plasma", False)), int(game.hand_size),
            # planet levels are NOT here: exact and cheap use the same level tables, so
            # the exact/cheap ratio is level-invariant to first order (a planet pick must
            # not force a ratio recompute -- it was 40% of a pack decision)
            sum(1 for c in game.full_deck if c.enhancement != "None"),
            sum(1 for c in game.full_deck if c.edition != "None"),
            sum(1 for c in game.full_deck if c.seal != "None"))


def board_ratio(game, n_hands: int = 4, cfg: HandConfig = DEFAULT_HAND_CONFIG) -> float:
    """Median exact/cheap multiplier of the board (jokers, editions, enhancements) over
    ``n_hands`` deterministic sample hands of the full deck -- the factor that turns the
    cheap-unit blind model into this board's scores.  Read-only (private clone); cached by
    ``_board_sig`` (the shop / pack rule tier calls this once per candidate per act(), and
    the state a purchase produces hits the candidate's cache entry -- fix pass 2026-08-23)."""
    if not game.jokers and not getattr(game, "plasma", False) and all(
            c.enhancement == "None" and c.edition == "None" and c.seal == "None" for c in game.full_deck):
        return 1.0
    sig = (_board_sig(game), n_hands)
    hit = _RATIO_CACHE.get(sig)
    if hit is not None:
        return hit
    import hashlib
    clone = game.clone()
    pool = sorted(clone.full_deck, key=canonical_card_key)
    H = max(1, min(int(game.hand_size), len(pool)))
    seed = int.from_bytes(hashlib.blake2b(repr(_blind_key(game)).encode(), digest_size=4).digest(), "little")
    rng = np.random.RandomState(seed % (2 ** 31))
    ratios = []
    for _ in range(max(1, n_hands)):
        idx = rng.choice(len(pool), size=H, replace=False)
        sample = [pool[i] for i in idx]
        ids = {id(c) for c in sample}
        for c in sample:
            c.face_down = False
        clone.hand = sample
        clone.deck = [c for c in pool if id(c) not in ids]
        an = HandAnalysis(clone, cfg, lite=True, build_model=False)
        ratios.append(an.ratio)
    ratios.sort()
    out = float(ratios[len(ratios) // 2])
    if len(_RATIO_CACHE) >= _RATIO_CACHE_MAX:
        _RATIO_CACHE.pop(next(iter(_RATIO_CACHE)))
    _RATIO_CACHE[sig] = out
    return out


_MODEL_CACHE: dict = {}
_MODEL_CACHE_MAX = 64



# ═══════════════════════════════════════════════════════════════ fresh-hand simulator

_SUIT_IDX = {"Spades": 0, "Hearts": 1, "Clubs": 2, "Diamonds": 3}
_WINDOWS_ARR = np.array([[14, 2, 3, 4, 5]] + [list(range(st, st + 5)) for st in range(2, 11)], dtype=np.int64)
_RANK_CHIPS_ARR = np.array([0, 0] + [RANK_CHIPS.get(r, 0) for r in range(2, 15)], dtype=np.float64)


class _FreshHandSim:
    """Vectorised Monte-Carlo of fresh hands from a deck composition (cheap scoring).

    ``run()`` -> (score_round0, score_round1, score_round2): arrays of the best made-hand
    score of ``n`` sampled hands (a) as dealt, (b) after one chase discard, (c) after two.
    Only the COMPOSITION is used (cards are indexed in their canonical sorted order)."""

    def __init__(self, cards, levels: dict, hand_size: int, *, four_fingers: bool = False,
                 seed: int = 0, n: int = 256):
        self.n = n
        self.hand_size = hand_size
        self.ff = four_fingers
        self.rng = np.random.RandomState(seed)
        cs = sorted(cards, key=canonical_card_key)
        self.D = len(cs)
        self.rank = np.array([c.rank if c.enhancement != "Stone" else 0 for c in cs], dtype=np.int64)
        self.suit = np.array([_SUIT_IDX.get(c.suit, 0) if c.enhancement != "Stone" else -1 for c in cs], dtype=np.int64)
        self.wild = np.array([c.enhancement == "Wild" and not c.debuffed for c in cs], dtype=bool)
        self.chips = np.array([_card_chips(c) for c in cs], dtype=np.float64)
        self.base = {ht: _hand_base(ht, levels) for ht in HAND_BASE}

    def run(self):
        n, H, D = self.n, min(self.hand_size, self.D), self.D
        if D == 0 or H == 0:
            z = np.zeros(n)
            return z, z, z
        perm = np.argsort(self.rng.random_sample((n, D)), axis=1)     # per-sample world
        hand = perm[:, :H].copy()                                    # (n, H) deck indices
        consumed = np.full(n, H, dtype=np.int64)
        s0 = self._score(hand)
        hand, consumed = self._chase(hand, consumed, perm)
        s1 = self._score(hand)
        hand, consumed = self._chase(hand, consumed, perm)
        s2 = self._score(hand)
        return s0, s1, s2

    def _counts(self, hand):
        n, H = hand.shape
        rows = np.repeat(np.arange(n), H)
        r = self.rank[hand].ravel()
        R = np.zeros((n, 15), dtype=np.int64)
        np.add.at(R, (rows, r), 1)
        R[:, 0] = 0
        su = self.suit[hand].ravel()
        w = self.wild[hand].ravel()
        S = np.zeros((n, 4), dtype=np.int64)
        ok = (su >= 0) & ~w
        np.add.at(S, (rows[ok], su[ok]), 1)
        if w.any():
            wc = np.zeros(n, dtype=np.int64)
            np.add.at(wc, rows[w], 1)
            S += wc[:, None]
        return R, S

    def _score(self, hand):
        n, H = hand.shape
        R, S = self._counts(hand)
        rank_chips = _RANK_CHIPS_ARR
        chips_h = self.chips[hand]                                   # (n, H)
        bc, bm = self.base["High Card"]
        best = (bc + chips_h.max(axis=1)) * bm
        Rr = R[:, 2:]
        rc_row = rank_chips[2:][None, :]
        cnt_sorted = np.sort(Rr, axis=1)[:, ::-1]
        top_cnt, second_cnt = cnt_sorted[:, 0], cnt_sorted[:, 1]
        for t, name in ((2, "Pair"), (3, "Three of a Kind"), (4, "Four of a Kind"), (5, "Five of a Kind")):
            have = top_cnt >= t
            if have.any():
                bc, bm = self.base[name]
                rc = np.where(Rr >= t, rc_row, -1.0).max(axis=1)
                sc = (bc + t * np.maximum(rc, 0)) * bm
                best = np.where(have, np.maximum(best, sc), best)
        two_pair = (top_cnt >= 2) & (second_cnt >= 2)
        if two_pair.any():
            bc, bm = self.base["Two Pair"]
            srt = np.sort(np.where(Rr >= 2, rc_row, -1.0), axis=1)[:, ::-1]
            sc = (bc + 2 * np.maximum(srt[:, 0], 0) + 2 * np.maximum(srt[:, 1], 0)) * bm
            best = np.where(two_pair, np.maximum(best, sc), best)
        full = (top_cnt >= 3) & (second_cnt >= 2)
        if full.any():
            bc, bm = self.base["Full House"]
            c3 = np.where(Rr >= 3, rc_row, -1.0).max(axis=1)
            c2 = np.where((Rr >= 2) & (rc_row != c3[:, None]), rc_row, -1.0).max(axis=1)
            c2 = np.where(c2 < 0, c3, c2)
            sc = (bc + 3 * np.maximum(c3, 0) + 2 * np.maximum(c2, 0)) * bm
            best = np.where(full, np.maximum(best, sc), best)
        need = 4 if self.ff else 5
        fs = S.max(axis=1)
        flush = fs >= need
        if flush.any():
            bc, bm = self.base["Flush"]
            bs = S.argmax(axis=1)
            inside = (self.suit[hand] == bs[:, None]) | self.wild[hand]
            top = np.sort(np.where(inside, chips_h, -1.0), axis=1)[:, ::-1][:, :5]
            sc = (bc + np.maximum(top, 0).sum(axis=1)) * bm
            best = np.where(flush, np.maximum(best, sc), best)
        present = R > 0
        bc, bm = self.base["Straight"]
        win_chips = rank_chips[_WINDOWS_ARR].sum(axis=1)
        pw = present[:, _WINDOWS_ARR]
        made = pw.all(axis=2)
        if self.ff:
            made = made | (pw.sum(axis=2) >= 4)
        anyst = made.any(axis=1)
        if anyst.any():
            sc = (bc + np.where(made, win_chips[None, :], 0.0).max(axis=1)) * bm
            best = np.where(anyst, np.maximum(best, sc), best)
        return best

    def _chase(self, hand, consumed, perm):
        """One chase discard per sample: keep the longest suit (3+), else the paired
        ranks, else the three best cards; draw the replacements from the sample's world."""
        n, H = hand.shape
        R, S = self._counts(hand)
        sh = self.suit[hand]
        rh = self.rank[hand]
        wh = self.wild[hand]
        chips_h = self.chips[hand]
        fs = S.max(axis=1)
        bs = S.argmax(axis=1)
        suit_keep = ((sh == bs[:, None]) | wh) & (fs[:, None] >= 3)
        paired = (R[np.arange(n)[:, None], rh] >= 2) & (rh > 0)
        trips = R.max(axis=1) >= 3
        use_suit = (fs >= 4) | ((fs >= 3) & ~trips)
        keep = np.where(use_suit[:, None], suit_keep, paired)
        none = ~keep.any(axis=1)
        if none.any():
            order = np.argsort(-chips_h, axis=1)
            top3 = np.zeros_like(keep)
            rows = np.arange(n)[:, None]
            top3[rows, order[:, :3]] = True
            keep = np.where(none[:, None], top3, keep)
        n_disc = (~keep).sum(axis=1)
        over = n_disc > 5
        if over.any():
            masked = np.where(keep, -1.0, chips_h)
            order = np.argsort(-masked, axis=1)
            for i in np.where(over)[0]:
                extra = n_disc[i] - 5
                keep[i, order[i, :extra]] = True
        new_hand = hand.copy()
        D = perm.shape[1]
        for i in range(n):
            disc = np.where(~keep[i])[0]
            k = len(disc)
            if k == 0:
                continue
            avail = D - consumed[i]
            k = min(k, avail)
            if k <= 0:
                continue
            new_hand[i, disc[:k]] = perm[i, consumed[i]:consumed[i] + k]
            consumed[i] += k
        return new_hand, consumed


class BlindModel:
    """Per-blind tail: per-round score distributions of a fresh hand (0 / 1 / 2 discards)
    + the dynamic programme ``G(need, h, d)`` on a geometric need grid.

    The per-round distributions come from ``cfg.model_samples`` fresh hands of the FULL
    deck's composition simulated in numpy (``_FreshHandSim``): round 0 = the best made
    hand; round 1 / 2 = after one / two chase discards by a simple rule (keep the
    longest suit if it has 3+, else the paired ranks, else the best three cards).  Scores
    are in CHEAP units (the hand formula at the run's planet levels, no jokers); the
    decision divides its ``need`` by the board's exact/cheap ``ratio`` at lookup time.
    Everything is deterministic: the sampler is seeded from the key."""

    def __init__(self, game, cfg: HandConfig = DEFAULT_HAND_CONFIG):
        self.cfg = cfg
        self.key = _blind_key(game)
        self.h_max, self.d_max = _resource_caps(game)
        import hashlib
        digest = hashlib.blake2b(repr(self.key).encode(), digest_size=4).digest()
        seed = int.from_bytes(digest, "little") % (2 ** 31)
        flags = game.hand_eval_flags() if hasattr(game, "hand_eval_flags") else {}
        sim = _FreshHandSim(game.full_deck, dict(game.planet_levels or {}), max(1, int(game.hand_size)),
                            four_fingers=bool(flags.get("four_fingers")), seed=seed,
                            n=max(8, cfg.model_samples))
        s0, s1, s2 = sim.run()
        self.Q = [self._compress([(1.0 / len(arr), float(v)) for v in arr], cfg.model_atoms)
                  for arr in (s0, s1, s2)]
        self.mean0 = float(np.mean(s0))
        self.var0 = float(np.var(s0))
        # per-hand score of a symmetric opponent who spends ~one discard per hand
        self.mean1 = float(np.mean(s1))
        self.var1 = float(np.var(s1))
        self.Qw = [np.array([w for w, _ in q], dtype=np.float64) for q in self.Q]
        self.Qs = [np.array([sc for _, sc in q], dtype=np.float64) for q in self.Q]
        self._build_grid()

    #: cumulative-weight edges of the atoms: the top atoms are deliberately thin so the
    #: big-hit tail (a flush, quads) survives compression
    _EDGES = {3: (0.5, 0.9, 1.0), 4: (0.35, 0.7, 0.92, 1.0), 5: (0.25, 0.5, 0.75, 0.93, 1.0),
              6: (0.2, 0.4, 0.6, 0.8, 0.93, 1.0), 7: (0.15, 0.3, 0.45, 0.6, 0.8, 0.93, 1.0)}

    @staticmethod
    def _compress(atoms: list, k: int) -> list:
        atoms = [(w, s) for w, s in atoms if w > 1e-12]
        if not atoms:
            return [(1.0, 0.0)]
        atoms.sort(key=lambda a: a[1])
        total = sum(w for w, _ in atoms)
        if total <= 0:
            return [(1.0, 0.0)]
        atoms = [(w / total, s) for w, s in atoms]
        if len(atoms) <= k:
            return atoms
        edges = BlindModel._EDGES.get(k) or tuple((i + 1) / k for i in range(k))
        out = []
        acc_w = acc_ws = cum = 0.0
        e = 0
        for w, s in atoms:
            acc_w += w
            acc_ws += w * s
            cum += w
            if e < len(edges) - 1 and cum >= edges[e] - 1e-9:
                out.append((acc_w, acc_ws / acc_w))
                acc_w = acc_ws = 0.0
                e += 1
        if acc_w > 0:
            out.append((acc_w, acc_ws / acc_w))
        return out

    def _build_grid(self) -> None:
        cfg = self.cfg
        r = cfg.grid_ratio
        log_r = math.log(r)
        n_pts = int(math.log(2e7) / log_r) + 2
        grid = np.array([r ** i for i in range(n_pts)], dtype=np.float64)
        self.grid = grid
        self.log_grid = np.log(grid)
        self._log_r = log_r
        H, D = self.h_max + 1, self.d_max + 1
        # A[h, d, :, i] = (P, U, E)[h, d, grid[i]]: P(clear), E[unused hands · 1clear],
        # E[unused discards · 1clear] with h hands / d discards against need grid[i]
        A = np.zeros((H, D, 3, n_pts), dtype=np.float64)
        beta, gamma = cfg.beta_hand, cfg.gamma_discard
        # per-k stencils over the atoms: weights (a,), i0/i1 (a, n), t1/t (a, n), cleared (a, n)
        stencils = []
        for q in self.Q:
            w = np.array([x[0] for x in q], dtype=np.float64)
            sc = np.array([x[1] for x in q], dtype=np.float64)
            rem = grid[None, :] - sc[:, None]
            cleared = rem < 1.0
            x = np.log(np.maximum(rem, 1.0)) / log_r
            i0 = np.floor(x).astype(np.int64)
            t = x - i0
            over = i0 >= n_pts - 1
            i0c = np.minimum(np.maximum(i0, 0), n_pts - 1)
            i1c = np.minimum(i0c + 1, n_pts - 1)
            t1 = 1.0 - t
            t1[over] = 0.0
            t[over] = 0.0
            stencils.append((w, i0c, i1c, t1, t, cleared))
        for h in range(1, H):
            hm1 = float(h - 1)
            for d in range(D):
                bestA = None
                bestV = None
                for k in range(0, min(2, d) + 1):
                    w, i0c, i1c, t1, t, cleared = stencils[k]
                    prev = A[h - 1, d - k]                       # (3, n)
                    vals = prev[:, i0c] * t1[None] + prev[:, i1c] * t[None]   # (3, a, n)
                    vals[0][cleared] = 1.0
                    vals[1][cleared] = hm1
                    vals[2][cleared] = float(d - k)
                    acc = np.einsum("a,can->cn", w, vals)
                    V = acc[0] + beta * acc[1] + gamma * acc[2]
                    if bestV is None:
                        bestA, bestV = acc, V
                    else:
                        better = V > bestV
                        if better.any():
                            bestA = np.where(better[None, :], acc, bestA)
                            bestV = np.maximum(V, bestV)
                A[h, d] = bestA
        self.A = A
        self.P = A[:, :, 0, :]
        self.U = A[:, :, 1, :]
        self.E = A[:, :, 2, :]
        self.V = self.P + beta * self.U + gamma * self.E

    # ── lookups ─────────────────────────────────────────────────────────────

    def _interp(self, arr: np.ndarray, need: float, h: int, d: int) -> float:
        h = min(h, self.h_max)
        d = max(0, min(d, self.d_max))
        x = math.log(need)
        lg = self.log_grid
        if x <= lg[0]:
            return float(arr[h, d, 0])
        if x >= lg[-1]:
            return 0.0
        i = int(x / self._log_r)
        if i >= len(lg) - 1:
            return 0.0
        x0, x1 = lg[i], lg[i + 1]
        y0, y1 = arr[h, d, i], arr[h, d, i + 1]
        t = (x - x0) / (x1 - x0)
        return float(y0 + t * (y1 - y0))

    def value(self, need: float, h: int, d: int, pvp: bool = False) -> float:
        """``G(need, h, d)``: P(clear) (+ beta·E[unused hands] + gamma·E[unused discards]
        unless ``pvp``).  ``need <= 0`` = already cleared."""
        if need <= 0.0:
            if pvp:
                return 1.0
            return 1.0 + self.cfg.beta_hand * max(0, h) + self.cfg.gamma_discard * max(0, d)
        if h <= 0:
            return 0.0
        return self._interp(self.P if pvp else self.V, need, h, d)

    def p_clear(self, need: float, h: int, d: int) -> float:
        return self.value(need, h, d, pvp=True)

    def value_vec(self, needs: np.ndarray, h: int, d: int, pvp: bool = False) -> np.ndarray:
        """Vectorised ``value`` over an array of needs (same (h, d))."""
        term = 1.0 if pvp else 1.0 + self.cfg.beta_hand * max(0, h) + self.cfg.gamma_discard * max(0, d)
        if h <= 0:
            return np.where(needs <= 0.0, term, 0.0)
        row = (self.P if pvp else self.V)[min(h, self.h_max), max(0, min(d, self.d_max))]
        x = np.log(np.maximum(needs, 1e-9))
        v = np.interp(x, self.log_grid, row, left=float(row[0]), right=0.0)
        return np.where(needs <= 0.0, term, v)


def blind_model_for(game, cfg: HandConfig = DEFAULT_HAND_CONFIG) -> BlindModel:
    """The cached per-blind model (LRU of 64 keyed by ``_blind_key``)."""
    key = (_blind_key(game), cfg)
    m = _MODEL_CACHE.get(key)
    if m is None:
        m = BlindModel(game, cfg)
        if len(_MODEL_CACHE) >= _MODEL_CACHE_MAX:
            _MODEL_CACHE.pop(next(iter(_MODEL_CACHE)))
        _MODEL_CACHE[key] = m
    return m


# ═══════════════════════════════════════════════════════════════════ the PvP opponent

def opponent_final_atoms(game, model: Optional["BlindModel"], ratio: float) -> list:
    """[(weight, final score)] of the Nemesis opponent: their live score plus, for each
    hand they still have, a symmetric per-hand score (this deck's one-discard round,
    scaled by OUR board ratio -- level-0 opponent modelling: same deck, same build)."""
    opp = float(getattr(game, "pvp_opponent_score", 0) or 0)
    opp_h = int(getattr(game, "pvp_opponent_hands", 0) or 0)
    if opp_h <= 0 or model is None:
        return [(1.0, opp)]
    mu = opp + opp_h * model.mean1 * ratio
    sd = math.sqrt(max(0.0, opp_h * model.var1)) * ratio
    return [(0.3, max(opp, mu - 0.97 * sd)), (0.4, mu), (0.3, mu + 0.97 * sd)]


# ═══════════════════════════════════════════════════════════════════════ the decision

class HandAnalysis:
    """Everything the fast budget computes for ONE ``SELECTING_HAND`` state.

    ``lite=True`` (rollouts / the blind model): fewer candidates, no exact refinements
    beyond the top 3.  ``legal`` = the legal action list (None = enumerate).  Nothing here
    mutates ``game`` (a private ``HypotheticalScorer``; all reads)."""

    def __init__(self, game, cfg: HandConfig = DEFAULT_HAND_CONFIG, *, lite: bool = False,
                 model: Optional[BlindModel] = None, legal: Optional[list] = None,
                 build_model: bool = True, ratio_hint: Optional[float] = None):
        self.game = game
        self.cfg = cfg
        self.lite = lite
        self.ratio_hint = ratio_hint        # rollouts: reuse the root's board ratio, skip the dry runs
        self.hand = list(game.hand)
        self.n = len(self.hand)
        self.flags = game.hand_eval_flags() if hasattr(game, "hand_eval_flags") else {}
        self.levels = dict(game.planet_levels or {})
        blind = game.current_blind
        self.boss = blind.boss_key if (blind.is_boss and not blind.disabled) else ""
        self.pvp = bool(blind.is_pvp)
        self.h = int(game.hands_left)
        self.d = int(game.discards_left)
        self.scored = float(game.chips_scored)
        self.target = float(blind.chips_target)
        self.need = max(0.0, self.target - self.scored)
        self.deck = DeckComp(game.deck)
        self.played_types = set(game.played_hand_types_this_round or ())
        self.four = bool(self.flags.get("four_fingers"))
        self.flush_need = 4 if self.four else 5
        self.straight_need = 4 if self.four else 5
        self.hand_size = max(1, int(game.hand_size))
        self._base_cache: dict = {}
        self._legal_play: Optional[set] = None
        self._legal_disc: Optional[set] = None
        if legal is not None:
            self._legal_play = {tuple(a["cards"]) for a in legal if a.get("type") == "play"}
            self._legal_disc = {tuple(a["cards"]) for a in legal if a.get("type") == "discard"}
            self.can_discard = bool(self._legal_disc)
        else:
            self.can_discard = self.d > 0 and self.boss != "bl_psychic"
        self.psychic = self.boss == "bl_psychic"
        self.procs = ProcBoard(game) if cfg.extract else _NO_PROCS
        self._prep_cards()
        self._prep_tarot_wants()
        self._gen_play_candidates()
        self._score_plays()
        self.model = model
        if build_model and self.model is None:
            self.model = blind_model_for(game, cfg)
        self._pvp_atoms: Optional[list] = None
        self._targets_cache: dict = {}
        self._floor_cache: dict = {}
        self._tail_cache: dict = {}
        self._chips_cache: dict = {}
        self._gen_cache: dict = {}
        self._safe_cache: dict = {}
        self._cycle_cache: dict = {}
        # the extraction layer is off at a Nemesis (no unused-hand money, every hand is
        # played anyway) and off when there is simply nothing to extract
        self.extract_on = bool(cfg.extract) and not self.pvp and self.model is not None and (
            self.procs.any or self.has_card_proc or bool(self._tarot_wants))

    # ── per-card facts ──────────────────────────────────────────────────────

    def _prep_cards(self):
        hand = self.hand
        self.chips = [_card_chips(c) for c in hand]
        self.hidden = [bool(getattr(c, "face_down", False)) for c in hand]
        self.stone = [c.enhancement == "Stone" for c in hand]
        self.wild = [c.enhancement == "Wild" and not c.debuffed for c in hand]
        rank_bits: dict = {}
        suit_bits: dict = {}
        for j, c in enumerate(hand):
            if self.stone[j] or self.hidden[j]:
                continue
            rank_bits[c.rank] = rank_bits.get(c.rank, 0) | (1 << j)
            if self.wild[j]:
                for s in ("Spades", "Hearts", "Clubs", "Diamonds"):
                    suit_bits[s] = suit_bits.get(s, 0) | (1 << j)
            else:
                suit_bits[c.suit] = suit_bits.get(c.suit, 0) | (1 << j)
        self.rank_bits = rank_bits
        self.suit_bits = suit_bits
        # keep value: how much a card is worth holding on to (orders the junk)
        rank_cnt = {r: _popcount(b) for r, b in rank_bits.items()}
        suit_cnt = {s: _popcount(b) for s, b in suit_bits.items()}
        present = set(rank_bits)
        win_score: dict = {}
        for w in _STRAIGHT_WINDOWS:
            k = sum(1 for r in w if r in present)
            if k >= 3:
                for r in w:
                    win_score[r] = max(win_score.get(r, 0), k)
        kv = []
        for j, c in enumerate(hand):
            if self.hidden[j]:
                kv.append(0.4)
                continue
            if self.stone[j]:
                kv.append(1.0 + 0.01 * self.chips[j])
                continue
            v = 0.0
            rc = rank_cnt.get(c.rank, 1)
            v += 3.0 * (rc - 1)
            sc = max(suit_cnt.get(s, 0) for s in (("Spades", "Hearts", "Clubs", "Diamonds") if self.wild[j] else (c.suit,)))
            if sc >= 3:
                v += 1.2 * (sc - 2)
            ws = win_score.get(c.rank, 0)
            if ws >= 3:
                v += 0.8 * (ws - 2)
            v += 0.02 * self.chips[j]
            # SCORING-relevant modifiers only.  The blanket "+0.5 for any enhancement /
            # seal / edition" used to make a Purple seal (worth nothing when played, a
            # Tarot when discarded) look keep-worthy, so "discard the purple seals" was
            # never generated -- W-EXTRACT replaces that half with the dollar terms below.
            if (c.enhancement in ("Bonus", "Mult", "Wild", "Glass", "Steel", "Lucky")
                    or c.edition != "None" or c.seal in ("Red", "Blue")):
                v += 0.5
            kv.append(v)
        self.keep_value = kv
        self.full_mask = (1 << self.n) - 1
        self._prep_procs()

    def _prep_procs(self):
        """Per-card money-proc value in DOLLARS, in three flavours, and the two junk
        orderings they imply (EXTRACT_NOTES.md §2/§3):

        ``proc_play[j]``    what card ``j`` pays when it is a SCORING card of a played hand
                            (Gold seal $3, Lucky 1/15·$20, Business Card ½·$2 on a face,
                            Golden Ticket $4 on a Gold card, Rough Gem $1 on a Diamond);
        ``proc_hold[j]``    what it pays while it stays in hand (Gold enhancement $3 at
                            round end, Reserved Parking ½·$1 per hand still to be played);
        ``proc_discard[j]`` what it pays when DISCARDED (Purple seal -> a Tarot, Mail-In
                            Rebate $5 on the round's rank).

        ``play_keep`` / ``discard_keep`` fold those into the structural keep value at
        ``cfg.junk_dollars`` keep-units per dollar.  They order JUNK ONLY -- the objective
        prices the money in P(clear) units itself, so the exchange rate here never decides
        a decision, only which cards a generated line throws away."""
        n, hand, pb, cfg = self.n, self.hand, self.procs, self.cfg
        chips = self.chips
        # Fast path: nothing on the board or in the hand can pay, so the three orderings
        # collapse to the structural keep value and NOTHING extra is computed.  This is the
        # ante-1 common case and the whole reason the 5 ms fast budget survives the layer.
        # (``faceless`` needs ``is_face`` even though its payout is set-level; ``trading`` /
        # ``delayed_grat`` / ``todo_list`` are set-level only and stay on the fast path)
        if not (pb.any_card_proc or pb.faceless) and not any(
                c.seal in ("Gold", "Purple") or c.enhancement in ("Gold", "Lucky") for c in hand):
            z = [0.0] * n
            self.proc_play = self.proc_hold = self.proc_discard = z
            self.is_face = self.gold_enh = self.purple = [False] * n
            self.has_card_proc = self.has_play_proc = False
            self.play_keep = self.discard_keep = self.keep_value
            order = sorted(range(n), key=lambda j: (self.keep_value[j], chips[j], j))
            self.play_junk_order = self.discard_junk_order = self.junk_order = order
            return
        self.proc_play = [0.0] * n
        self.proc_hold = [0.0] * n
        self.proc_discard = [0.0] * n
        self.is_face = [False] * n
        self.gold_enh = [False] * n
        self.purple = [False] * n
        pareidolia = bool(self.flags.get("pareidolia"))
        p_lucky_money = min(1.0, pb.p_normal / 15.0)     # card.lua:1076, denominator 15
        h_left = max(0, self.h)
        for j, c in enumerate(hand):
            if self.hidden[j] or c.debuffed:
                continue                 # a face-down card is junk of unknown value; a
                                         # debuffed one is skipped by every proc (Card:is_face
                                         # card.lua:965, scoring.py:257 `continue`)
            face = pareidolia or c.rank in _FACE_RANKS
            self.is_face[j] = face
            passes = 2.0 if c.seal == "Red" else 1.0     # Red seal retriggers the whole pass
            play = 0.0
            if c.seal == "Gold":
                play += 3.0
            if c.enhancement == "Lucky":
                play += 20.0 * p_lucky_money
            if pb.p_business and face:
                play += 2.0 * pb.p_business
            if pb.ticket and c.enhancement == "Gold":
                play += 4.0
            if pb.rough_gem and c.suit == "Diamonds":
                play += 1.0
            self.proc_play[j] = play * passes
            hold = 0.0
            if c.enhancement == "Gold":
                self.gold_enh[j] = True
                hold += 3.0 * passes     # a Red-sealed Gold card pays twice at round end
            if pb.p_parking and face:
                hold += pb.p_parking * h_left
            self.proc_hold[j] = hold
            disc = 0.0
            if c.seal == "Purple":
                self.purple[j] = True
                disc += cfg.tarot_value_dollars
            if pb.mail_rank and c.rank == pb.mail_rank and c.enhancement != "Stone":
                disc += 5.0
            self.proc_discard[j] = disc
        self.has_card_proc = any(self.proc_play) or any(self.proc_hold) or any(self.proc_discard)
        self.has_play_proc = any(self.proc_play)
        w = cfg.junk_dollars
        pp, ph, pd = self.proc_play, self.proc_hold, self.proc_discard
        # burning a card as a non-scoring PLAY filler wastes all three of its uses
        self.play_keep = [kv + w * (pp[j] + ph[j] + pd[j]) for j, kv in enumerate(self.keep_value)]
        # discarding it CASHES proc_discard and forfeits the other two
        self.discard_keep = [kv + w * (pp[j] + ph[j] - pd[j]) for j, kv in enumerate(self.keep_value)]
        self.play_junk_order = sorted(range(n), key=lambda j: (self.play_keep[j], chips[j], j))
        self.discard_junk_order = sorted(range(n), key=lambda j: (self.discard_keep[j], chips[j], j))
        #: back-compat alias: the historic ``junk_order`` was play filler
        self.junk_order = self.play_junk_order

    def _prep_tarot_wants(self):
        """First-order targeting value of the TARGETED tarots already in hand.

        A held tarot is only worth its ``tarot_value_dollars`` if there is something in
        hand worth putting it on.  Each entry is ``(want_mask, need, pile_count)``: the
        hand cards that qualify as targets, how many of them the tarot needs, and how many
        qualifying cards are left in the draw pile.  ``_cycle_ev`` then prices a cycling
        line by the hypergeometric probability that the draw supplies the missing ones.

        Modelled: the four SUIT tarots (Star / Moon / Sun / World -- convert up to 3 cards,
        so a flush wants ``flush_need - 3`` real cards of that suit already in hand) and
        the eight ENHANCEMENT tarots (want a plain card; overwriting an enhancement wastes
        it).  NOT modelled: Strength / Hanged Man / Death targeting, and which SPECIFIC
        card the tarot should land on (see EXTRACT_NOTES.md §6)."""
        out: list = []
        self._tarot_wants = out
        if not self.cfg.extract:
            return
        held = getattr(self.game, "consumable_hand", ()) or ()
        if not held:
            return
        for key in held:
            k = _TAROT_TARGETS.get(key)
            if k is None:
                continue
            if key in _TAROT_SUIT:
                suit = _TAROT_SUIT[key]
                out.append((self.suit_bits.get(suit, 0), max(1, self.flush_need - k),
                            self.deck.by_suit.get(suit, 0)))
            elif key in _TAROT_ENHANCEMENT:
                want = 0
                for j, c in enumerate(self.hand):
                    if not self.hidden[j] and c.enhancement == "None" and not c.debuffed:
                        want |= 1 << j
                out.append((want, 1, self.deck.n_plain))

    # ── candidates ──────────────────────────────────────────────────────────

    def _best_of(self, bits: int, k: int) -> list:
        """The k highest-chip card indices in ``bits``."""
        idx = [j for j in range(self.n) if bits >> j & 1]
        idx.sort(key=lambda j: (-self.chips[j], j))
        return idx[:k]

    def _reps(self, bits: int, k: int) -> list:
        """Representative choices of ``k`` cards out of ``bits`` for one structural hand
        type: the highest-chip pick, plus -- when the board has per-card money procs -- the
        pick that maximises PLAY value instead.  Two same-rank Kings score identically, but
        only one of them carries the Gold seal; without this variant the gold-seal /
        lucky / Business-Card line is never even a candidate."""
        a = tuple(sorted(self._best_of(bits, k)))
        if not self.has_play_proc:
            return [a]
        idx = [j for j in range(self.n) if bits >> j & 1]
        idx.sort(key=lambda j: (-self.proc_play[j], -self.chips[j], j))
        b = tuple(sorted(idx[:k]))
        return [a] if b == a else [a, b]

    def _scoring_sets(self) -> list:
        """Structural scoring sets: (indices tuple, hint) for every hand type makeable."""
        out: list = []
        rb, sb = self.rank_bits, self.suit_bits
        # n of a kind
        for r, bits in rb.items():
            c = _popcount(bits)
            if c >= 2:
                out.extend(self._reps(bits, min(5, c)))
                if c >= 3:
                    out.extend(self._reps(bits, 2))
        # two pair / full house
        pairs = [(r, bits) for r, bits in rb.items() if _popcount(bits) >= 2]
        pairs.sort(key=lambda rb_: -RANK_CHIPS.get(rb_[0], 0))
        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                (r1, b1), (r2, b2) = pairs[i], pairs[j]
                c1, c2 = _popcount(b1), _popcount(b2)
                out.append(tuple(sorted(self._best_of(b1, 2) + self._best_of(b2, 2))))
                if c1 >= 3:
                    out.append(tuple(sorted(self._best_of(b1, 3) + self._best_of(b2, 2))))
                if c2 >= 3:
                    out.append(tuple(sorted(self._best_of(b2, 3) + self._best_of(b1, 2))))
        # flush per suit
        for s, bits in sb.items():
            c = _popcount(bits)
            if c >= self.flush_need:
                out.extend(tuple(sorted(t)) for t in self._reps(bits, 5))
                if c > 5:
                    # the flush that keeps the highest cards for later: lowest 5
                    idx = [j for j in range(self.n) if bits >> j & 1]
                    idx.sort(key=lambda j: (self.chips[j], j))
                    out.append(tuple(sorted(idx[:5])))
                # straight flush inside the suit
                present = {}
                for j in range(self.n):
                    if bits >> j & 1:
                        present.setdefault(self.hand[j].rank, []).append(j)
                for w in _STRAIGHT_WINDOWS:
                    if all(r in present for r in w):
                        out.append(tuple(sorted(max(present[r], key=lambda j: (self.chips[j], -j)) for r in w)))
        # straights
        present = {r: bits for r, bits in rb.items()}
        for w in _STRAIGHT_WINDOWS:
            if all(r in present for r in w):
                out.append(tuple(sorted(self._best_of(present[r], 1)[0] for r in w)))
                if self.has_play_proc:
                    out.append(tuple(sorted(self._reps(present[r], 1)[-1][0] for r in w)))
        # singles: every visible non-stone card, and stones alone
        for j in range(self.n):
            if not self.hidden[j]:
                out.append((j,))
        seen = set()
        uniq = []
        for s in out:
            if s and s not in seen:
                seen.add(s)
                uniq.append(s)
        return uniq

    def _dump_variants(self, s: tuple, limit: int = 5) -> list:
        """``s`` plus the least valuable other cards, up to ``limit`` cards."""
        base = set(s)
        junk = [j for j in self.junk_order if j not in base]
        out = []
        room = limit - len(s)
        if room <= 0:
            return out
        if len(s) == 1:
            sizes = (room,) if not self.psychic else (room,)
        else:
            sizes = {1, room} if not self.psychic else {room}
        for k in sorted(sizes):
            if 1 <= k <= len(junk):
                out.append(tuple(sorted(list(s) + junk[:k])))
        return out

    def _gen_play_candidates(self):
        sets = self._scoring_sets()
        cands: list = []
        seen = set()

        def add(t):
            if t in seen:
                return
            if self._legal_play is not None and t not in self._legal_play:
                return
            if self.psychic and len(t) != 5:
                return
            seen.add(t)
            cands.append(t)

        for s in sets:
            add(s)
            if len(s) < 5:
                for v in self._dump_variants(s):
                    add(v)
        # pure dump: the k worst cards (k = 5, and 1)
        if self.n >= 1:
            add(tuple(sorted(self.junk_order[:min(5, self.n)])))
            add((self.junk_order[0],))
        if self.lite and len(cands) > 24:
            cands = cands[:24]
        self.play_cands = cands

    # ── scoring ─────────────────────────────────────────────────────────────

    def _base(self, ht: str) -> tuple:
        b = self._base_cache.get(ht)
        if b is None:
            b = self._base_cache[ht] = _hand_base(ht, self.levels)
        return b

    def _type_allowed(self, ht: str) -> bool:
        if self.boss == "bl_eye" and ht in self.played_types:
            return False
        if self.boss == "bl_mouth" and self.played_types and ht not in self.played_types:
            return False
        return True

    def _score_plays(self):
        """cheap + exact scores for every play candidate; ``self.plays`` = list of
        (indices, hand_type, score, cheap)."""
        hand, flags = self.hand, self.flags
        evald = []
        cheap = []
        #: indices tuple -> (bitmask of the cards that actually SCORE, hand type).  Money
        #: procs fire per SCORING card, not per played card (state_events.lua:648-780).
        #: Built here from the SAME membership test the chip sum needs -- by identity, not
        #: ``card in scoring``: Card is a dataclass, so ``in`` compares every field.
        self._scoring_of: dict = {}
        for t in self.play_cands:
            cards = [hand[j] for j in t]
            try:
                ht, scoring = evaluate_hand(cards, **flags)
            except Exception:           # noqa: BLE001
                evald.append(None)
                cheap.append(0.0)
                continue
            bc, bm = self._base(ht)
            sc_ids = {id(c) for c in scoring}
            smask = 0
            total = bc
            for j in t:
                if id(hand[j]) in sc_ids:
                    total += self.chips[j]
                    smask |= 1 << j
            self._scoring_of[t] = (smask, ht)
            s = total * bm
            if not self._type_allowed(ht):
                s = 0.0
            evald.append((cards, ht, scoring))
            cheap.append(s)
        # The Hook discards 2 random UNPLAYED cards after a play (engine fix 8d3f0d8, was
        # the whole hand): the play scores in full, only the kept cards are perturbed --
        # modelled as nothing here (second order; the freshness mixture absorbs most of it).
        exact = list(cheap)
        plain = self._cheap_is_exact()
        n_exact = 0
        if not plain and self.play_cands and self.ratio_hint is None:
            n_exact = 3 if self.lite else self.cfg.exact_top
        self.ratio = 1.0 if self.ratio_hint is None else float(self.ratio_hint)
        pick: set = set()
        if n_exact > 0:
            order = sorted(range(len(cheap)), key=lambda i: -cheap[i])
            # always refine the best candidate of every hand type (joker synergies)
            pick = set(order[:n_exact])
            best_by_type: dict = {}
            for i in order:
                e = evald[i]
                if e is None:
                    continue
                if e[1] not in best_by_type:
                    best_by_type[e[1]] = i
            pick.update(best_by_type.values())
            scorer = HypotheticalScorer(self.game, model_held=True)
            ratios = []
            for i in pick:
                e = evald[i]
                if e is None or cheap[i] <= 0.0:
                    continue
                cards, ht, scoring = e
                try:
                    v = float(scorer.score(cards, ht, scoring))
                except Exception:       # noqa: BLE001
                    continue
                if self.boss == "bl_flint":
                    v = float(int(v) // 2)
                exact[i] = v
                if cheap[i] > 0:
                    ratios.append(v / cheap[i])
            if ratios:
                # the board multiplier: median exact/cheap over the top refined plays
                top = sorted(((exact[i], cheap[i]) for i in pick if evald[i] is not None and cheap[i] > 0),
                             key=lambda x: -x[0])[:5]
                rr = sorted(e / c for e, c in top) if top else ratios
                self.ratio = rr[len(rr) // 2]
        elif self.boss == "bl_flint":
            exact = [float(int(v) // 2) for v in exact]
        # non-refined candidates are scaled by the board ratio so the two tiers share a scale
        if self.ratio != 1.0:
            for i in range(len(exact)):
                if exact[i] == cheap[i] and cheap[i] > 0 and i not in pick:
                    exact[i] = cheap[i] * self.ratio
        self.plays = []
        for t, e, s, c in zip(self.play_cands, evald, exact, cheap):
            if e is None:
                continue
            self.plays.append((t, e[1], float(s), float(c)))
        # masks of the scoring sets for the floor computation
        self._play_masks = [(sum(1 << j for j in t), s) for t, _, s, _ in self.plays]

    def _cheap_of(self, t: tuple) -> float:
        if not t:
            return 0.0
        cards = [self.hand[j] for j in t]
        try:
            ht, scoring = evaluate_hand(cards, **self.flags)
        except Exception:               # noqa: BLE001
            return 0.0
        if not self._type_allowed(ht):
            return 0.0
        bc, bm = self._base(ht)
        return (bc + sum(self.chips[j] for j in t if self.hand[j] in scoring)) * bm

    def _cheap_is_exact(self) -> bool:
        g = self.game
        if getattr(g, "jokers", None) or getattr(g, "plasma", False):
            return False
        if self.boss in ("bl_flint",):
            return False
        return all(c.enhancement == "None" and c.edition == "None" and c.seal == "None"
                   for c in self.hand)

    def best_play_score(self) -> float:
        return max((s for _, _, s, _ in self.plays), default=0.0)

    def floor(self, keep_mask: int) -> float:
        """Best made play inside ``keep_mask`` (exact units)."""
        v = self._floor_cache.get(keep_mask)
        if v is not None:
            return v
        best = 0.0
        for m, s in self._play_masks:
            if m & ~keep_mask == 0 and s > best:
                best = s
        self._floor_cache[keep_mask] = best
        return best

    # ── draw targets ────────────────────────────────────────────────────────

    def _mask_chips(self, mask: int) -> float:
        v = self._chips_cache.get(mask)
        if v is None:
            chips = self.chips
            v = 0.0
            mm = mask
            while mm:
                low = mm & -mm
                v += chips[low.bit_length() - 1]
                mm ^= low
            self._chips_cache[mask] = v
        return v

    def _top_chips(self, mask: int, k: int) -> float:
        """Sum of the k highest chip values among the cards in ``mask``."""
        vals = []
        mm = mask
        chips = self.chips
        while mm:
            low = mm & -mm
            vals.append(chips[low.bit_length() - 1])
            mm ^= low
        if len(vals) <= k:
            return float(sum(vals))
        vals.sort(reverse=True)
        return float(sum(vals[:k]))

    def targets(self, keep_mask: int, m: int) -> list:
        """[(p, v)] completion probability and value (exact units) of every draw target
        reachable from ``keep_mask`` with ``m`` fresh cards from the draw pile."""
        if m <= 0:
            return []
        deck = self.deck
        N = deck.total
        if N <= 0:
            return []
        m = min(m, N)
        ck = (keep_mask, m)
        cached = self._targets_cache.get(ck)
        if cached is not None:
            return cached
        ratio = self.ratio
        out: list = []
        self._targets_cache[ck] = out
        rb, sb = self.rank_bits, self.suit_bits
        allowed = self._type_allowed
        by_rank = deck.by_rank
        # flush
        if allowed("Flush"):
            bc, bm = self._base("Flush")
            fn = self.flush_need
            for s, bits in sb.items():
                kb = bits & keep_mask
                if not kb:
                    continue
                have = _popcount(kb)
                need = fn - have
                if need <= 0 or need > m:
                    continue
                K = deck.by_suit.get(s, 0)
                if K < need:
                    continue
                p = _hyper_tail(N, K, m, need)
                if p < 1e-4:
                    continue
                v = (bc + self._top_chips(kb, 5) + need * deck.suit_avg(s)) * bm * ratio
                out.append((p, v))
        # rank groups in the keep set
        groups = []
        present_ranks = 0
        for r, bits in rb.items():
            kb = bits & keep_mask
            if kb:
                groups.append((r, kb, _popcount(kb)))
                present_ranks |= 1 << r
        # straights (windows with <= 2 ranks missing)
        if allowed("Straight") and groups:
            bc, bm = self._base("Straight")
            maxmiss = min(m, 2)
            for w, wmask in _WINDOW_MASKS:
                missing_mask = wmask & ~present_ranks
                nm = _popcount(missing_mask)
                if nm == 0 or nm > maxmiss:
                    continue
                sizes = []
                ok = True
                miss_chips = 0.0
                for r in w:
                    if missing_mask >> r & 1:
                        k = by_rank.get(r, 0)
                        if k == 0:
                            ok = False
                            break
                        sizes.append(k)
                        miss_chips += RANK_CHIPS.get(r, 0)
                if not ok:
                    continue
                p = _p_all_groups(N, m, tuple(sizes))
                if p < 1e-4:
                    continue
                held = 0.0
                for r, kb, c in groups:
                    if wmask >> r & 1:
                        held += self._top_chips(kb, 1)
                out.append((p, (bc + held + miss_chips) * bm * ratio))
        # n of a kind
        for r, kb, c in groups:
            K = by_rank.get(r, 0)
            if K <= 0:
                continue
            held = self._mask_chips(kb)
            rc = RANK_CHIPS.get(r, 0)
            for t in (2, 3, 4):
                need = t - c
                if need <= 0 or need > m or K < need:
                    continue
                name = _KIND_NAME[t]
                if not allowed(name):
                    continue
                p = _hyper_tail(N, K, m, need)
                if p < 1e-4:
                    continue
                bc, bm = self._base(name)
                out.append((p, (bc + held + need * rc) * bm * ratio))
        groups.sort(key=lambda g: (-g[2], -RANK_CHIPS.get(g[0], 0)))
        pairs = [g for g in groups if g[2] >= 2]
        singles = [g for g in groups if g[2] == 1]
        if pairs and allowed("Two Pair"):
            bc, bm = self._base("Two Pair")
            r1, kb1, c1 = pairs[0]
            held1 = self._top_chips(kb1, 2)
            if singles:
                Ksum = sum(by_rank.get(r, 0) for r, _, _ in singles)
                p = _hyper_tail(N, Ksum, m, 1)
                if p >= 1e-4:
                    r2, kb2, _ = singles[0]
                    out.append((p, (bc + held1 + self._mask_chips(kb2) + RANK_CHIPS.get(r2, 0)) * bm * ratio))
            if len(pairs) == 1 and m >= 2:
                Kmax = max(by_rank.values(), default=0)
                p = _hyper_tail(N, Kmax, m, 2)
                if p >= 1e-4:
                    out.append((p, (bc + held1 + 2 * deck.avg_chips) * bm * ratio))
        elif len(singles) >= 2 and m >= 2 and allowed("Two Pair"):
            bc, bm = self._base("Two Pair")
            top = singles[:3]
            for i in range(len(top)):
                for j in range(i + 1, len(top)):
                    (r1, kb1, _), (r2, kb2, _) = top[i], top[j]
                    p = _p_all_groups(N, m, (by_rank.get(r1, 0), by_rank.get(r2, 0)))
                    if p < 1e-4:
                        continue
                    held = self._mask_chips(kb1) + self._mask_chips(kb2)
                    out.append((p, (bc + held + RANK_CHIPS.get(r1, 0) + RANK_CHIPS.get(r2, 0)) * bm * ratio))
        if pairs and allowed("Full House"):
            bc, bm = self._base("Full House")
            trips = [g for g in pairs if g[2] >= 3]
            if trips:
                r1, kb1, c1 = trips[0]
                held1 = self._top_chips(kb1, 3)
                if singles:
                    Ksum = sum(by_rank.get(r, 0) for r, _, _ in singles)
                    p = _hyper_tail(N, Ksum, m, 1)
                    if p >= 1e-4:
                        r2, kb2, _ = singles[0]
                        out.append((p, (bc + held1 + self._mask_chips(kb2) + RANK_CHIPS.get(r2, 0)) * bm * ratio))
            elif len(pairs) >= 2:
                (r1, kb1, _), (r2, kb2, _) = pairs[0], pairs[1]
                Ksum = by_rank.get(r1, 0) + by_rank.get(r2, 0)
                p = _hyper_tail(N, Ksum, m, 1)
                if p >= 1e-4:
                    held = self._mask_chips(kb1) + self._mask_chips(kb2)
                    out.append((p, (bc + held + RANK_CHIPS.get(r1, 0)) * bm * ratio))
            elif singles and m >= 2:
                r1, kb1, _ = pairs[0]
                r2, kb2, _ = singles[0]
                p = _p_all_groups(N, m, (by_rank.get(r1, 0), by_rank.get(r2, 0)))
                if p >= 1e-4:
                    held = self._mask_chips(kb1) + self._mask_chips(kb2)
                    out.append((p, (bc + held + RANK_CHIPS.get(r1, 0) + RANK_CHIPS.get(r2, 0)) * bm * ratio))
        return out

    # ── position value ──────────────────────────────────────────────────────

    def _pvp_needs(self) -> list:
        """[(weight, need_tie, need_win)] atoms of the opponent's final score."""
        if self._pvp_atoms is not None:
            return self._pvp_atoms
        self._pvp_atoms = [(w, a - self.scored, a - self.scored + 1.0)
                           for w, a in opponent_final_atoms(self.game, self.model, self.ratio)]
        return self._pvp_atoms

    def tail(self, need: float, h: int, d: int) -> float:
        """Generic continuation (the blind model, cheap units -> divide by the board's
        exact/cheap ratio), objective-aware."""
        if self.model is None:
            if need <= 0:
                return 1.0 + self.cfg.beta_hand * max(0, h)
            return 0.0
        key = (int(need), h, d)
        v = self._tail_cache.get(key)
        if v is None:
            v = self._tail_cache[key] = self.model.value(need / max(self.ratio, 1e-6), h, d, pvp=self.pvp)
        return v

    def terminal(self, h: int, d: int) -> float:
        if self.pvp:
            return 1.0
        return 1.0 + self.cfg.beta_hand * max(0, h) + self.cfg.gamma_discard * max(0, d)

    def _value_for_need(self, keep_mask: int, m: int, h: int, d: int, need: float) -> float:
        """Value of holding ``keep_mask``, drawing ``m``, with ``h`` hands / ``d`` discards
        and ``need`` chips still required (single-need objective).

        The NEXT hand's score is modelled as a freshness-weighted mixture (w = m / hand
        size) of (a) the specific model — the best made hand in the kept cards (floor), or
        the best draw target completed with its hypergeometric probability, with one
        re-try on a miss when a discard is left — and (b) the generic fresh-hand rounds of
        the blind model lower-bounded by the floor, with 0-2 discards spent on them.
        Everything after that hand is the blind model's tail ``G``."""
        if need <= 0.0:
            return self.terminal(h, d)
        if h <= 0:
            return 0.0
        if self.boss == "bl_serpent":
            m = self.hand_size                      # the whole hand is redrawn
            floor = 0.0
        else:
            floor = self.floor(keep_mask)
        tail = self.tail
        fl_tail = tail(need - floor, h - 1, d) if floor > 0.0 else tail(need, h - 1, d)
        spec = fl_tail
        if m > 0 and self.boss != "bl_serpent":
            for p, v in self.targets(keep_mask, m):
                hit = tail(need - v, h - 1, d)
                miss = fl_tail
                if d > 0:
                    # second attempt at the same target after a miss (one more discard)
                    miss2 = p * tail(need - v, h - 1, d - 1) + (1.0 - p) * (
                        tail(need - floor, h - 1, d - 1) if floor > 0 else tail(need, h - 1, d - 1))
                    if miss2 > miss:
                        miss = miss2
                val = p * hit + (1.0 - p) * miss
                if val > spec:
                    spec = val
        model = self.model
        if m <= 0 or model is None:
            return spec
        w = min(1.0, m / float(self.hand_size))
        gen = self._generic_next(need, h, d, floor)
        if gen > spec:
            return spec + w * (gen - spec)
        return spec

    def _generic_next(self, need: float, h: int, d: int, floor: float) -> float:
        """Next hand ~ the blind model's rounds (0-2 discards), lower-bounded by ``floor``,
        then the tail."""
        key = (int(need), h, d, int(floor))
        v = self._gen_cache.get(key)
        if v is not None:
            return v
        model = self.model
        ratio = self.ratio
        tail = self.tail
        gen = 0.0
        h1 = h - 1
        for k in range(0, min(2, d) + 1):
            acc = 0.0
            dk = d - k
            for q, sc in model.Q[k]:
                e = sc * ratio
                acc += q * tail(need - (e if e > floor else floor), h1, dk)
            if acc > gen:
                gen = acc
        self._gen_cache[key] = gen
        return gen

    def position_value(self, keep_mask: int, m: int, h: int, d: int, need: float) -> float:
        if not self.pvp:
            return self._value_for_need(keep_mask, m, h, d, need)
        total = 0.0
        for w, need_tie, need_win in self._pvp_needs():
            total += w * (0.5 * self._value_for_need(keep_mask, m, h, d, need_tie)
                          + 0.5 * self._value_for_need(keep_mask, m, h, d, need_win))
        return total

    # ── extraction: money procs in dollars ──────────────────────────────────

    def _p_clear_after(self, h: int, d: int, need: float) -> float:
        """P(the blind is still cleared) from a fresh position with ``h`` hands / ``d``
        discards and ``need`` chips outstanding — the tail DP's own pure-probability
        lookup (``BlindModel.p_clear``), which is what the safety gate is defined on."""
        if need <= 0.0:
            return 1.0
        if h <= 0 or self.model is None:
            return 0.0
        key = (int(need), h, d)
        v = self._safe_cache.get(key)
        if v is None:
            v = self._safe_cache[key] = self.model.p_clear(need / max(self.ratio, 1e-6), h, d)
        return v

    def extraction_safe(self, h: int, d: int, need: float) -> bool:
        """The gate of EXTRACT_NOTES §4: money is only worth banking when what is left
        after the line still clears the blind with probability ``cfg.extract_min_clear``.
        Always False at a Nemesis (there is no money at a PvP blind and every hand is
        played anyway — EV_NOTES §3)."""
        if not self.extract_on:
            return False
        return self._p_clear_after(h, d, need) >= self.cfg.extract_min_clear

    def _dollar_rate(self) -> float:
        """$1 in P(clear) units.  ``beta_hand`` IS the rate (an unused hand pays $1), and
        below the interest cap a dollar is worth a bit more because it keeps earning:
        ``player.build_proxy`` values money as ``$ + 0.8·interest`` (PlayerConfig
        interest_weight 0.4, two rounds), whose MARGINAL rate is 1 + 0.8/INTEREST_RATE
        = 1.16.  Same convention, no new model (brief §3.2)."""
        r = self.cfg.beta_hand
        g = self.game
        if not getattr(g, "no_interest", False):
            cap = float(getattr(g, "interest_cap", INTEREST_CAP))
            if float(getattr(g, "dollars", 0)) < cap * INTEREST_RATE:
                r *= 1.0 + self.cfg.interest_bonus
        return r

    def _scoring_mask(self, t: tuple) -> tuple:
        """(scoring bitmask, hand type) of a play, computed on demand for actions that are
        not in the structural candidate set (``hand_ev`` on an arbitrary action)."""
        got = self._scoring_of.get(t)
        if got is not None:
            return got
        cards = [self.hand[j] for j in t if j < self.n]
        try:
            ht, scoring = evaluate_hand(cards, **self.flags)
        except Exception:               # noqa: BLE001
            return (0, "")
        got = (sum(1 << j for j in t if j < self.n and self.hand[j] in scoring), ht)
        self._scoring_of[t] = got
        return got

    def _gold_delta(self, t: tuple, m: int, p_clear: float) -> float:
        """Gold ENHANCEMENT is an end-of-round payout on cards still HELD (game.py
        _end_round, card.lua:1033-1039): a Gold card that leaves the hand now forfeits its
        $3, and the ``m`` fresh cards may bring one in.  Conditioned on ``p_clear`` — a
        failed blind never reaches the payout."""
        if not (self.deck.n_gold or any(self.gold_enh)):
            return 0.0
        lost = 0.0
        for j in t:
            if j < self.n and self.gold_enh[j]:
                lost += 2.0 if self.hand[j].seal == "Red" else 1.0
        gained = (m * self.deck.n_gold / self.deck.total) if self.deck.total else 0.0
        return 3.0 * (gained - lost) * p_clear

    def _play_extraction(self, t: tuple, m: int, p_clear: float) -> float:
        """Dollars a PLAY of ``t`` pays.  Immediate procs are unconditional (``ease_dollars``
        inside the scoring loop); the hold term is conditioned on finishing the round."""
        pb = self.procs
        smask, ht = self._scoring_mask(t)
        money = 0.0
        mm = smask
        pp = self.proc_play
        while mm:
            low = mm & -mm
            money += pp[low.bit_length() - 1]
            mm ^= low
        if pb.todo_hand and ht == pb.todo_hand:
            money += 4.0
        if pb.p_parking:
            # Reserved Parking fires on the cards STILL IN HAND while this hand scores --
            # the played selection has already left G.hand (state_events.lua:478-488, the
            # held phase at :784-870).  Exactly counted for THIS hand; later hands refill
            # to the same deck composition whatever we do now, so they cancel (§5).
            played = set(t)
            money += pb.p_parking * sum(1 for j in range(self.n)
                                        if self.is_face[j] and j not in played)
        return money + self._gold_delta(t, m, p_clear)

    def _discard_extraction(self, t: tuple, m: int, p_clear: float) -> float:
        """Dollars a DISCARD of ``t`` pays (Purple seals -> Tarots, Faceless, Mail-In
        Rebate, Trading Card) net of what the discard costs (Delayed Gratification)."""
        pb = self.procs
        money = 0.0
        n_purple = 0
        for j in t:
            if j >= self.n:
                continue
            money += self.proc_discard[j]
            if self.purple[j]:
                n_purple += 1
        if n_purple > pb.free_slots:
            # game.py:_discard re-checks _free_consumable_slots per card, exactly like the
            # real G.GAME.consumeable_buffer (card.lua:2253-2255): the overflow pays nothing
            money -= (n_purple - pb.free_slots) * self.cfg.tarot_value_dollars
        if pb.faceless and sum(1 for j in t if j < self.n and self.is_face[j]) >= 3:
            money += 5.0
        if pb.trading and len(t) == 1:
            money += 3.0                 # ... and the card is destroyed (not priced, §7)
        if pb.delayed_grat:
            # $2 per discard remaining at round end, but ONLY if none was used: the first
            # discard of the round forfeits the whole row (misc.py:_DelayedGrat)
            money -= 2.0 * max(0, self.d)
        return money + self._gold_delta(t, m, p_clear)

    def _cycle_ev(self, keep_mask: int, m: int) -> float:
        """Tarot-targeting value of drawing ``m`` fresh cards onto ``keep_mask``: for each
        held targeted tarot whose targets are NOT in the kept cards, the hypergeometric
        probability that the draw supplies them, times a fraction of the tarot's value.
        First-order — see ``_prep_tarot_wants`` and EXTRACT_NOTES §6."""
        if not self._tarot_wants or m <= 0:
            return 0.0
        key = (keep_mask, m)
        v = self._cycle_cache.get(key)
        if v is not None:
            return v
        N = self.deck.total
        unit = self.cfg.tarot_value_dollars * self.cfg.tarot_cycle_fraction
        total = 0.0
        for want, need, pile in self._tarot_wants:
            miss = need - _popcount(want & keep_mask)
            if miss <= 0 or pile <= 0 or N <= 0:
                continue
            total += _hyper_tail(N, pile, min(m, N), miss) * unit
        self._cycle_cache[key] = total
        return total

    def extraction_ev(self, action: dict) -> float:
        """Public: expected DOLLARS ``action`` extracts (money procs + tarot cycling).
        UNGATED — this is the raw amount; ``evaluate`` applies the safety gate of §4."""
        if not self.extract_on:
            return 0.0
        t = tuple(action.get("cards", ()))
        mask = sum(1 << j for j in t if j < self.n)
        keep = self.full_mask & ~mask
        m = min(len(t), self.hand_size - _popcount(keep))
        kind = action.get("type")
        if kind == "play":
            pc = self._p_clear_after(self.h - 1, self.d, self.need - self._exact_of(t))
            return self._play_extraction(t, m, pc) + self._cycle_ev(keep, m)
        if kind == "discard":
            pc = self._p_clear_after(self.h, self.d - 1, self.need)
            return self._discard_extraction(t, m, pc) + self._cycle_ev(keep, m)
        return 0.0

    def _exact_of(self, t: tuple) -> float:
        for tt, _, s, _ in self.plays:
            if tt == t:
                return s
        return self._cheap_of(t) * self.ratio

    def _extraction_discard_lines(self) -> list:
        """Discard lines that exist only to EXTRACT: dump the purple seals / Mail-In ranks,
        dump three faces for Faceless, dump one card for Trading Card.  Generated ONLY when
        the tail DP says the blind is still safe after spending the discard (§4)."""
        cfg = self.cfg
        if not self.extract_on or self.d <= 0 or not self.can_discard:
            return []
        if self._p_clear_after(self.h, self.d - 1, self.need) < cfg.extract_min_clear:
            return []
        pb, n = self.procs, self.n
        out: list = []

        def push(core: list, fill_to: int = 5):
            core = sorted(set(core))[:5]
            if not core:
                return
            out.append(tuple(core))
            room = fill_to - len(core)
            if room > 0:
                have = set(core)
                fill = [j for j in self.discard_junk_order if j not in have][:room]
                if fill:
                    out.append(tuple(sorted(core + fill)))

        proc_idx = [j for j in range(n) if self.proc_discard[j] > 0.0]
        if proc_idx:
            proc_idx.sort(key=lambda j: (-self.proc_discard[j], self.discard_keep[j], j))
            # only the seals a free consumable slot can actually absorb are worth dumping
            cap = max(1, pb.free_slots) if any(self.purple[j] for j in proc_idx) else 5
            push(proc_idx[:cap])
        if pb.faceless:
            faces = [j for j in range(n) if self.is_face[j]]
            if len(faces) >= 3:
                faces.sort(key=lambda j: (self.discard_keep[j], self.chips[j], j))
                push(faces[:3], fill_to=5)
        if pb.trading and self.discard_junk_order:
            out.append((self.discard_junk_order[0],))
        cap = 3 if self.lite else cfg.extract_lines
        return out[:cap]

    # ── actions ─────────────────────────────────────────────────────────────

    def _discard_lines(self) -> list:
        """Keep sets worth chasing: (discard indices tuple)."""
        if not self.can_discard or self.n == 0:
            return []
        n = self.n
        full = self.full_mask
        lines: list = []
        seen = set()

        def add_disc(idx: list):
            if not idx:
                return
            if len(idx) > 5:
                # discard only the 5 least valuable of them (DISCARD-value: a purple seal
                # is worth MORE gone, a gold card worth more kept)
                idx = sorted(idx, key=lambda j: (self.discard_keep[j], self.chips[j], j))[:5]
            t = tuple(sorted(idx))
            if t in seen:
                return
            if self._legal_disc is not None and t not in self._legal_disc:
                return
            seen.add(t)
            lines.append(t)

        def add_keep(keep_mask: int):
            disc = full & ~keep_mask
            if disc == 0:
                return
            add_disc([j for j in range(n) if disc >> j & 1])

        # suits
        for s, bits in self.suit_bits.items():
            c = _popcount(bits)
            if c >= 3 and self.flush_need - c <= min(5, n - c):
                add_keep(bits)
        # straight windows with <= 2 missing
        present = self.rank_bits
        for w in _STRAIGHT_WINDOWS:
            have = [r for r in w if r in present]
            if len(have) >= 3:
                keep = 0
                for r in have:
                    # keep one card of the rank: prefer the majority suit, then chips
                    best_j = max((j for j in range(n) if present[r] >> j & 1),
                                 key=lambda j: (self.discard_keep[j], self.chips[j]))
                    keep |= 1 << best_j
                add_keep(keep)
        # rank groups (pair+) and combinations of them
        groups = [(r, bits) for r, bits in present.items() if _popcount(bits) >= 2]
        groups.sort(key=lambda g: (-_popcount(g[1]), -RANK_CHIPS.get(g[0], 0)))
        all_groups = 0
        for r, bits in groups:
            all_groups |= bits
            add_keep(bits)
        if len(groups) >= 2:
            add_keep(all_groups)
        # best made play's scoring set
        if self.plays:
            t, ht, s, _ = max(self.plays, key=lambda p: p[2])
            add_keep(sum(1 << j for j in t))
        # junk-out k worst (keep everything else) -- by DISCARD value
        for k in range(1, min(5, n) + 1):
            m = 0
            for j in self.discard_junk_order[:k]:
                m |= 1 << j
            add_keep(full & ~m)
        lines = lines[:6] if self.lite else lines[: self.cfg.max_discard_lines]
        # the extraction lines are appended AFTER the cap: they are a different KIND of
        # line (bank money / cycle the deck), not a worse chase line, and they are already
        # gated on the tail DP
        seen = set(lines)
        for t in self._extraction_discard_lines():
            if t in seen:
                continue
            if self._legal_disc is not None and t not in self._legal_disc:
                continue
            seen.add(t)
            lines.append(t)
        return lines

    def evaluate(self) -> list:
        """[(action, ev)] for every play / discard candidate.

        The extraction term (EXTRACT_NOTES §2-§4) enters at the objective's own exchange
        rate, $1 = ``beta_hand`` = 0.012 P(clear) units — the same rate that already pays
        for an unused hand — and ONLY for lines the tail DP still calls safe.  That gate is
        what makes sandbagging self-regulating: banking $4 of Tarot is worth 0.056, so it
        only wins against clearing now when the risk it adds is smaller than that."""
        out: list = []
        h, d, need = self.h, self.d, self.need
        full = self.full_mask
        extract = self.extract_on
        rate = self._dollar_rate() if extract else 0.0
        gate = self.cfg.extract_min_clear
        for t, ht, s, _ in self.plays:
            mask = sum(1 << j for j in t)
            keep = full & ~mask
            m = min(len(t), self.hand_size - _popcount(keep))
            if self.pvp:
                ev = self._pvp_play_value(keep, m, s)
            else:
                ev = self._value_for_need(keep, m, h - 1, d, need - s)
            # tie-breaks (bounded by 1e-6): higher score now, fewer cards spent
            ev += 1e-6 * s / (s + max(need, 1.0)) - 1e-9 * len(t)
            if extract:
                pc = self._p_clear_after(h - 1, d, need - s)
                if pc >= gate:
                    ev += rate * (self._play_extraction(t, m, pc) + self._cycle_ev(keep, m))
            out.append(({"type": "play", "cards": list(t)}, ev))
        if d > 0:
            for t in self._discard_lines():
                mask = sum(1 << j for j in t)
                keep = full & ~mask
                m = min(len(t), self.hand_size - _popcount(keep))
                ev = self.position_value(keep, m, h, d - 1, need)
                ev -= 1e-9 * len(t)
                if extract:
                    pc = self._p_clear_after(h, d - 1, need)
                    if pc >= gate:
                        ev += rate * (self._discard_extraction(t, m, pc) + self._cycle_ev(keep, m))
                out.append(({"type": "discard", "cards": list(t)}, ev))
        return out

    def _pvp_play_value(self, keep: int, m: int, s: float) -> float:
        total = 0.0
        for w, need_tie, need_win in self._pvp_needs():
            total += w * (0.5 * self._value_for_need(keep, m, self.h - 1, self.d, need_tie - s)
                          + 0.5 * self._value_for_need(keep, m, self.h - 1, self.d, need_win - s))
        return total

    def best_chase_line(self) -> Optional[tuple]:
        """(p, v, floor) of the single-discard line with the best expected score — used by
        the blind model (expected-score criterion, objective-free)."""
        best = None
        for t in self._discard_lines():
            mask = sum(1 << j for j in t)
            keep = self.full_mask & ~mask
            fl = self.floor(keep)
            for p, v in self.targets(keep, len(t)):
                if v <= fl:
                    continue
                e = p * v + (1 - p) * fl
                if best is None or e > best[0]:
                    best = (e, p, v, fl)
        if best is None:
            return None
        return best[1], best[2], best[3]

    def value_now(self) -> float:
        """Value of the position as it stands (best action's EV)."""
        evs = self.evaluate()
        return max((ev for _, ev in evs), default=0.0)


# ═══════════════════════════════════════════════════════════════════════ consumables

def _consumable_candidates(game, legal: list, analysis: Optional[HandAnalysis]) -> list:
    """A small set of ``use_consumable`` actions worth evaluating: every untargeted use,
    plus targets drawn from the best play's scoring cards (singles and pairs)."""
    uses = [a for a in legal if a.get("type") == "use_consumable"]
    if not uses:
        return []
    best_cards: list = []
    if analysis is not None and analysis.plays:
        t, ht, s, _ = max(analysis.plays, key=lambda p: p[2])
        best_cards = list(t)[:5]
    out = []
    seen = set()
    for a in uses:
        tc = tuple(a.get("target_cards", ()))
        if not tc:
            key = (a["consumable_idx"], tc)
            if key not in seen:
                seen.add(key)
                out.append(a)
            continue
        if all(j in best_cards for j in tc) and len(tc) <= 2:
            key = (a["consumable_idx"], tc)
            if key not in seen and sum(1 for k in seen if k[0] == a["consumable_idx"]) < 3:
                seen.add(key)
                out.append(a)
    return out


def _consumable_ev(game, action: dict, analysis: HandAnalysis, cfg: HandConfig) -> float:
    """EV of using a consumable now = the best hand action's EV in the resulting state
    (on a clone), minus nothing: a use that does not change the position ties with the
    position's own value and is then preferred only if it frees a slot."""
    c = game.clone()
    before = (len(c.consumable_hand), c.state)
    c.step(action)
    if c.state != State.SELECTING_HAND or len(c.consumable_hand) >= before[0]:
        return -1.0                       # failed / no-op use (the engine no-ops silently)
    reuse = analysis.model if (analysis.model is not None and _blind_key(c) == analysis.model.key) else None
    an = HandAnalysis(c, cfg, model=reuse, legal=c.legal_actions(), build_model=True)
    return an.value_now()


# ═══════════════════════════════════════════════════════════════════════ public API

def _hand_ranking_fast(game, cfg: HandConfig, *, legal: Optional[list] = None,
                       with_consumables: bool = True, lite: bool = False,
                       ratio_hint: Optional[float] = None) -> list:
    if legal is None:
        legal = game.legal_actions()
    an = HandAnalysis(game, cfg, lite=lite, legal=legal, ratio_hint=ratio_hint)
    ranked = an.evaluate()
    if with_consumables and not lite:
        base = max((ev for _, ev in ranked), default=0.0)
        for a in _consumable_candidates(game, legal, an):
            ev = _consumable_ev(game, a, an, cfg)
            if ev < 0:
                continue
            # using a consumable that does not hurt is preferred (frees the slot), a
            # targeted one only when it strictly helps
            bonus = 1e-6 if not a.get("target_cards") else -1e-6
            ranked.append((a, ev + bonus))
    ranked.sort(key=lambda x: (-x[1], _action_sort_key(x[0])))
    return ranked


def _action_sort_key(a: dict) -> tuple:
    return (a.get("type", ""), tuple(a.get("cards", ())), a.get("consumable_idx", -1),
            tuple(a.get("target_cards", ())))


def _blind_id(game) -> tuple:
    return (game.ante, game.blind_idx)


def play_out_blind(world, cfg: HandConfig = DEFAULT_HAND_CONFIG, *, lite: bool = True,
                   max_steps: int = 64, ratio_hint: Optional[float] = None) -> None:
    """Drive ``world`` (a clone) with the fast policy until the blind ends (in place).
    ``ratio_hint`` = the root's board ratio (rollouts skip the per-decision dry runs)."""
    bid = _blind_id(world)
    steps = 0
    while world.state == State.SELECTING_HAND and _blind_id(world) == bid and steps < max_steps:
        ranked = _hand_ranking_fast(world, cfg, with_consumables=False, lite=lite, ratio_hint=ratio_hint)
        if not ranked:
            legal = world.legal_actions()
            if not legal:
                break
            world.step(legal[0])
        else:
            world.step(ranked[0][0])
        steps += 1


def end_of_blind_value(world, origin, cfg: HandConfig = DEFAULT_HAND_CONFIG,
                       value_fn: Optional[Callable] = None, model: Optional[BlindModel] = None,
                       ratio: float = 1.0) -> float:
    """Value of ``world`` once its blind has ended (analytic proxy, or ``value_fn``).

    With ``value_fn``: the proxy's terminal cases (GAME_OVER) are kept; otherwise the world
    is advanced out of ROUND_EVAL (``advance`` -> the shop / next blind select) so V sees a
    real decision state, and ``value_fn(world)`` is returned.  ``world`` is consumed."""
    blind = origin.current_blind
    if world.state == State.GAME_OVER and not world.match_won:
        return 0.0
    cleared = world.chips_scored >= world.current_blind.chips_target if not blind.is_pvp else None
    if value_fn is not None:
        if world.state == State.ROUND_EVAL:
            world.step({"type": "advance"})
        # a broken value_fn propagates -- silently degrading to the proxy hid real V bugs
        # from W5's pipeline (lead fix pass, 2026-08-23)
        return float(value_fn(world))
    if blind.is_pvp:
        s = float(world.chips_scored)
        atoms = opponent_final_atoms(origin, model, ratio)
        return sum(w * (0.5 * float(s >= a) + 0.5 * float(s > a)) for w, a in atoms)
    if not cleared:
        return 0.0
    return 1.0 + cfg.beta_hand * max(0, world.hands_left) + cfg.gamma_discard * max(0, world.discards_left)


def _hand_ranking_full(game, cfg: HandConfig, *, value_fn=None, rng=None, n_worlds: int,
                       top_k: Optional[int], legal: Optional[list] = None) -> list:
    if legal is None:
        legal = game.legal_actions()
    fast = _hand_ranking_fast(game, cfg, legal=legal)
    if not fast:
        return fast
    k = top_k if top_k is not None else cfg.full_top_k
    head = fast[:max(1, k)]
    tail_rest = fast[len(head):]
    if rng is None:
        rng = world_rng(0, game)
    model = blind_model_for(game, cfg)
    ratio = HandAnalysis(game, cfg, lite=True, model=model, legal=legal).ratio
    worlds = [sample_world(game, rng) for _ in range(max(1, n_worlds))]
    out = []
    for a, fast_ev in head:
        acc = 0.0
        for w in worlds:
            w2 = w.clone()
            w2.step(a)
            play_out_blind(w2, cfg, lite=cfg.rollout_lite, ratio_hint=ratio)
            acc += end_of_blind_value(w2, game, cfg, value_fn=value_fn, model=model, ratio=ratio)
        ev = acc / len(worlds)
        out.append((a, ev + 1e-9 * fast_ev))
    out.sort(key=lambda x: (-x[1], _action_sort_key(x[0])))
    # the rest keep their fast EV, shifted below the rolled-out head so the order is total
    if tail_rest:
        floor_ev = out[-1][1]
        shift = min(0.0, floor_ev - tail_rest[0][1] - 1e-6)
        out.extend((a, ev + shift) for a, ev in tail_rest)
    return out


def rank_hand_actions(game, *, budget: str = "fast", value_fn=None, rng=None,
                      top_k: Optional[int] = None, n_worlds: Optional[int] = None,
                      cfg: HandConfig = DEFAULT_HAND_CONFIG, legal: Optional[list] = None) -> list:
    """``[(action, ev)]`` sorted descending for a ``SELECTING_HAND`` state.  Side-effect-free."""
    if game.state != State.SELECTING_HAND:
        return []
    if budget == "fast":
        ranked = _hand_ranking_fast(game, cfg, legal=legal)
        return ranked[:top_k] if top_k else ranked
    if budget == "full":
        # W-LEAF / EV_NOTES §8.3: with a value_fn, the leaf is worth more worlds and fewer
        # candidates (K=3 x 8 worlds fits the 100 ms budget) -- flag-driven on value_fn's
        # presence, an explicit top_k/n_worlds from the caller always wins, and the path
        # with no value_fn is byte-for-byte the original K=5 x 3-world default.
        if value_fn is not None:
            nw = n_worlds if n_worlds is not None else cfg.full_n_worlds_v
            tk = top_k if top_k is not None else cfg.full_top_k_v
        else:
            nw = n_worlds if n_worlds is not None else cfg.full_n_worlds
            tk = top_k
        return _hand_ranking_full(game, cfg, value_fn=value_fn, rng=rng, n_worlds=nw,
                                  top_k=tk, legal=legal)
    raise ValueError(f"unknown budget {budget!r} (want 'fast' or 'full')")


def hand_ev(game, action: dict, *, budget: str = "fast", value_fn=None, rng=None,
            n_worlds: int = 8, cfg: HandConfig = DEFAULT_HAND_CONFIG) -> float:
    """EV of one ``SELECTING_HAND`` action, horizon = end of the current blind.

    Exact where enumerable (the current hand's plays, the draw-target probabilities over
    the real pile composition), sampled otherwise (``budget="full"``: ``n_worlds`` draw
    worlds).  Never reads the draw pile's order."""
    if game.state != State.SELECTING_HAND:
        raise ValueError("hand_ev needs a SELECTING_HAND state")
    key = _action_sort_key(action)
    if budget == "fast":
        for a, ev in _hand_ranking_fast(game, cfg):
            if _action_sort_key(a) == key:
                return float(ev)
        # not a candidate: evaluate it directly
        an = HandAnalysis(game, cfg, legal=game.legal_actions())
        if action.get("type") == "play":
            t = tuple(action.get("cards", ()))
            cards = [game.hand[j] for j in t if j < an.n]
            if not cards:
                return 0.0
            ht, scoring = evaluate_hand(cards, **an.flags)
            bc, bm = an._base(ht)
            s = (bc + sum(an.chips[j] for j in t if game.hand[j] in scoring)) * bm * an.ratio
            if not an._type_allowed(ht):
                s = 0.0
            mask = sum(1 << j for j in t)
            return float(an.position_value(an.full_mask & ~mask, len(t), an.h - 1, an.d, an.need - s)
                         if not an.pvp else an._pvp_play_value(an.full_mask & ~mask, len(t), s))
        if action.get("type") == "discard":
            t = tuple(action.get("cards", ()))
            mask = sum(1 << j for j in t)
            return float(an.position_value(an.full_mask & ~mask, len(t), an.h, an.d - 1, an.need))
        if action.get("type") == "use_consumable":
            return float(_consumable_ev(game, action, an, cfg))
        return 0.0
    if budget == "full":
        if rng is None:
            rng = world_rng(0, game)
        model = blind_model_for(game, cfg)
        ratio = HandAnalysis(game, cfg, lite=True, model=model).ratio
        worlds = [sample_world(game, rng) for _ in range(max(1, n_worlds))]
        acc = 0.0
        for w in worlds:
            w2 = w.clone()
            w2.step(action)
            play_out_blind(w2, cfg, lite=cfg.rollout_lite, ratio_hint=ratio)
            acc += end_of_blind_value(w2, game, cfg, value_fn=value_fn, model=model, ratio=ratio)
        return acc / len(worlds)
    raise ValueError(f"unknown budget {budget!r}")


def extraction_ev(game, action: dict, *, cfg: HandConfig = DEFAULT_HAND_CONFIG) -> float:
    """Expected DOLLARS ``action`` extracts from a ``SELECTING_HAND`` state: the money
    procs that fire (Gold seal, Lucky, Business Card, Golden Ticket, Rough Gem, Reserved
    Parking, Faceless, Mail-In Rebate, Trading Card, Purple seal -> Tarot, Gold enhancement
    held to round end) plus the first-order tarot-targeting value of what it cycles.

    UNGATED — the raw dollar amount, for tests / the advisor.  The player only banks it
    when ``HandAnalysis.extraction_safe`` says the blind survives the line."""
    if game.state != State.SELECTING_HAND:
        return 0.0
    return HandAnalysis(game, cfg, legal=game.legal_actions()).extraction_ev(action)


def extraction_lines(game, legal: Optional[list] = None, *,
                     cfg: HandConfig = DEFAULT_HAND_CONFIG) -> list:
    """``[(action, ev, reason)]`` — the SAFETY-GATED sandbag lines at this state, best
    first, or ``[]`` when there is nothing to extract or the tail DP says the blind is not
    safe enough to spend a resource on money.

    A line qualifies when its gated extraction term is non-zero: the dedicated
    ``_extraction_discard_lines`` (dump the Purple seals / three faces for Faceless / one
    card for Trading Card) and every play whose scoring cards carry a proc.  ``ev`` is the
    full objective value, so it is directly comparable with ``rank_hand_actions``.
    (Interface for W-PAIRS's ``greedy_vs_extract`` pair source, brief §5.2.)"""
    if game.state != State.SELECTING_HAND:
        return []
    if legal is None:
        legal = game.legal_actions()
    an = HandAnalysis(game, cfg, legal=legal)
    if not an.extract_on:
        return []
    out = []
    for a, ev in an.evaluate():
        t = tuple(a.get("cards", ()))
        mask = sum(1 << j for j in t if j < an.n)
        keep = an.full_mask & ~mask
        m = min(len(t), an.hand_size - _popcount(keep))
        if a["type"] == "play":
            pc = an._p_clear_after(an.h - 1, an.d, an.need - an._exact_of(t))
            money = an._play_extraction(t, m, pc) + an._cycle_ev(keep, m)
        else:
            pc = an._p_clear_after(an.h, an.d - 1, an.need)
            money = an._discard_extraction(t, m, pc) + an._cycle_ev(keep, m)
        if pc < cfg.extract_min_clear or abs(money) < 1e-9:
            continue
        out.append((a, float(ev), f"extract ${money:.2f} (P(clear after) {pc:.2f})"))
    out.sort(key=lambda x: (-x[1], _action_sort_key(x[0])))
    return out


def best_hand_action(game, **kw) -> dict:
    ranked = rank_hand_actions(game, **kw)
    if ranked:
        return ranked[0][0]
    legal = game.legal_actions()
    return legal[0] if legal else {"type": "advance"}


def estimate_clear_probability(game, target: float, hands: int, discards: int,
                               cfg: HandConfig = DEFAULT_HAND_CONFIG) -> float:
    """P(a fresh blind of ``target`` chips is cleared with ``hands`` / ``discards``) under
    this game's deck + board — the blind model's tail from a fresh position (used by the
    shop / blind-select rules of ``EVPlayer``)."""
    model = blind_model_for(game, cfg)
    ratio = board_ratio(game, cfg=cfg)
    return model.p_clear(float(target) / max(ratio, 1e-6), int(hands), int(discards))
