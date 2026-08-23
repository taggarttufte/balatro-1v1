"""
heuristic.py — a hand-quality PRIOR (and a candidate mask) for `SELECTING_HAND`.

Why this exists
---------------
Stage A of the first real run (`train_cold --ruleset vanilla --encoder set --sims 40`)
played 4 350 episodes and cleared the ante-1 Small blind 46 times (1%); a 200-sim
diagnostic cleared 0/53. So the failure is not the simulation budget. A cold net's prior
over the ~436 legal actions of a `SELECTING_HAND` state is uniform, ~218 of them are
`play` subsets and ~218 are `discard` subsets, and the lines that clear a 300-chip blind
(build a Flush / Straight / Two Pair by discarding, then play it) are a vanishing
fraction of that. The value target is therefore constant (every episode ends at ante 1)
and nothing learns. This module encodes what a first-year Balatro player knows —
*a Flush scores more than a High Card, and three cards of a suit are worth chasing* —
**as a prior, not as a constraint**: the search may still take any legal action, and the
weight `lambda` on the heuristic is annealed to a floor as the net learns.

Everything here is READ-ONLY on the game. Scoring goes through the engine's
`HypotheticalScorer` (`balatro_sim/card_selection.py`), which is the side-effect-free dry
run Phase 1 built for exactly this purpose: it clones `gs.run_state.rng`, copies every
mutable object and never touches `used_jokers`, so `state_signature()` is bit-identical
before and after. `tests/test_heuristic_prior.py` pins that.

The scores
----------
Let `flags = game.hand_eval_flags()` (Four Fingers / Shortcut / Smeared), and for a hand
type `T` let

    base(T)  = HAND_BASE[T] + (level(T) - 1) * (HAND_LEVEL_CHIPS[T], HAND_LEVEL_MULT[T])

be the chips/mult at the run's CURRENT planet level (`game.planet_levels`).

**play(S)** — two tiers, so the common case is cheap and the ones that matter are exact.

    cheap(S) = (base_chips(T) + sum of chips of the scoring cards) * base_mult(T)
               where (T, scoring) = evaluate_hand(S, **flags), and a card's chips are
               `card.base_chips + card.bonus_chips` (0 if debuffed).
    exact(S) = HypotheticalScorer(game, model_held=True).score(S, T, scoring)
               — the real `score_hand`, jokers, editions, enhancements, held-in-hand and
               full-deck effects included, on a private RNG clone.

    play(S)  = exact(S) for the top `exact_top` subsets by cheap(S); cheap(S) otherwise.

    The two tiers share a scale by construction: `cheap` IS `score_hand`'s formula with
    every joker / enhancement / edition contribution dropped, and those are almost always
    non-negative, so refinement moves a candidate UP unless the dry run found a real
    reason it should not (a debuff, a boss). Measured cost on this box at an 8-card hand:
    218 x `evaluate_hand` = 1.35 ms, 32 x `HypotheticalScorer.score` = 0.72 ms.

**discard(D)** — a potential for the position the discard leaves you in. Let `K` be the
kept cards (the hand minus D) and `d = |D|` the number of cards drawn back.

    floor(D) = max{ cheap(S) : S a legal play subset drawn only from K }
               "what I can still play if the draw gives me nothing" — an exact
               max-over-submasks DP over the play candidates already scored.
    draw(D)  = max over draw targets T of  value(T) * P(complete T | K, d)
    discard(D) = discard_bias * max(floor(D), draw(D))

    Draw targets are the ones a 5-card poker hand is actually chased on, each taken at
    its single best completion: the best suit (Flush), the best 5-rank window
    (Straight), and every held rank (Pair / Three / Four of a Kind). `m` = cards still
    needed; `p` = (useful cards left in `game.deck`) / (deck remaining);
    `P(complete) = P(X >= m)`, `X ~ Binom(d, p)` — the standard with-replacement
    approximation to the hypergeometric, which is what makes it one vectorised
    expression instead of 218 loops.

    `value(T) = (base_chips(T) + n_scoring(T) * avg_chips) * base_mult(T)` where
    `avg_chips` is the mean `base_chips` of the cards still in the deck. The drawn cards
    are unknown by definition, so their chip contribution is an expectation; this is the
    one place the heuristic is an estimate of an estimate, and it is deliberately blunt.

**Why the two are on one scale.** `floor(D)` is a `cheap(S)`, `value(T)` is the same
formula with the unknown cards filled in at their mean, and `play(S)` is a `cheap(S)`
possibly refined upward. So a single softmax over play + discard is meaningful:
"discard" wins exactly when the position it leaves is worth more than the best hand on
the table right now, which is the decision a human makes at a 300-chip blind.

The prior
---------
Partition the node's legal actions into H (`play` / `discard`) and O (everything else —
`use_consumable` in `SELECTING_HAND`, and every action of every other state). With
`W_H = sum_{a in H} net(a)`:

    h(a) = W_H * softmax_H( log1p(score(a)) / tau )     a in H
    h(a) = net(a)                                       a in O
    prior(a) = (1 - lambda) * net(a) + lambda * h(a)

Two consequences worth stating. (1) **lambda only ever moves mass BETWEEN hand actions**
— the net keeps its say on how much of the state's probability goes to playing a card at
all versus using a Tarot, and a non-`SELECTING_HAND` state is returned untouched. (2) the
softmax is on `log1p(score)`, so `tau` is scale-free: `tau = 1` is `prior ~ score`,
`tau = 0.5` is `prior ~ score^2`, `tau -> 0` is argmax. Raw scores span 10 (a lone Two)
to 10^4+ (a levelled Flush with jokers), which no additive-temperature softmax survives.

The mask
--------
`max_candidates = K > 0` keeps only the top-K `play` and the top-K `discard` actions by
the same score, plus ALL of O, and renormalises. This is a SEARCH-SIDE mask: every
pruned action is still legal in the engine, still appears in `game.legal_actions()`, and
still shows up (with visit count 0) in the training `Sample`. It is also the per-leaf
throughput lever — featurising 65 actions instead of 436 is most of a leaf evaluation.

Determinism: the returned dict preserves `legal_actions()` order, so root child order,
Dirichlet noise assignment and Gumbel's `legal_keys` are unchanged by the shaping.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from balatro_sim.card_selection import HypotheticalScorer
from balatro_sim.constants import HAND_BASE, HAND_LEVEL_CHIPS, HAND_LEVEL_MULT
from balatro_sim.hand_eval import evaluate_hand

from .action import ActionKey

#: Hands bigger than this fall back to "no heuristic" — the max-over-submasks DP is
#: 2**n and the whole point is to be cheaper than the net. Real hand sizes are 5-12.
MAX_HAND_FOR_HEURISTIC = 14

#: Ace-low first, then 2-6 ... 10-A. Used for the Straight draw target.
_STRAIGHT_WINDOWS: tuple[tuple[int, ...], ...] = tuple(
    [(14, 2, 3, 4, 5)] + [tuple(range(s, s + 5)) for s in range(2, 11)]
)

#: Binomial coefficients C(d, j) for d, j <= 5 (a discard is at most 5 cards).
_BINOM = np.array([[math.comb(d, j) if j <= d else 0 for j in range(6)]
                   for d in range(6)], dtype=np.float64)


@dataclass(frozen=True)
class HeuristicConfig:
    """Knobs for the hand prior. Defaults are what `PRIOR_NOTES.md` recommends."""
    tau: float = 0.5             # softmax temperature on log1p(score); 1.0 = prior ~ score
    exact_top: int = 8           # dry-run `score_hand` refinements per leaf (0 = cheap only)
    discard_bias: float = 1.0    # multiplier on every discard potential
    max_candidates: int = 0      # 0 = no pruning; K = keep top-K play + top-K discard


DEFAULT_CONFIG = HeuristicConfig()


# ══════════════════════════════════════════════════════════════════ hand-type base values

def _hand_base(hand_type: str, planet_levels: dict) -> tuple[float, float]:
    """(chips, mult) for `hand_type` at the run's current planet level."""
    chips, mult = HAND_BASE.get(hand_type, (5, 1))
    level = planet_levels.get(hand_type, 1) if planet_levels else 1
    if level > 1:
        chips += HAND_LEVEL_CHIPS.get(hand_type, 0) * (level - 1)
        mult += HAND_LEVEL_MULT.get(hand_type, 0) * (level - 1)
    return float(chips), float(mult)


def _card_chips(card) -> float:
    if card.debuffed:
        return 0.0
    return float(card.base_chips + getattr(card, "bonus_chips", 0))


# ═══════════════════════════════════════════════════════════════════════ the scorer

class HandHeuristic:
    """Scores the `play` / `discard` actions of ONE `SELECTING_HAND` state.

    Built per leaf (it is cheap — the state-dependent work is the scoring itself) and
    thrown away. Nothing here mutates the game; see the module docstring.
    """

    def __init__(self, game, cfg: HeuristicConfig = DEFAULT_CONFIG):
        self.game = game
        self.cfg = cfg
        self.hand = list(getattr(game, "hand", ()) or ())
        self.n = len(self.hand)
        self.flags = game.hand_eval_flags() if hasattr(game, "hand_eval_flags") else {}
        self.levels = dict(getattr(game, "planet_levels", {}) or {})
        self._cheap: dict = {}
        #: id(card) -> chip contribution, so the inner loop over scoring cards is a dict
        #: hit rather than a property call + getattr per card per subset.
        self._chips = {id(c): _card_chips(c) for c in self.hand}

    @property
    def usable(self) -> bool:
        return 0 < self.n <= MAX_HAND_FOR_HEURISTIC

    @property
    def cheap_is_exact(self) -> bool:
        """True when `cheap(S) == score_hand(S)` for every S, so the dry run is skippable.

        With no jokers, no Plasma balance and every card plain (no enhancement, edition
        or seal), `score_hand` reduces to exactly `(base_chips + sum of card chips) *
        base_mult` — no `pre_score` hooks, no held-in-hand phase, no `joker_main`, no
        retriggers, no probability rolls. That is the whole of ante 1 for a cold run, and
        skipping the refinement there is ~0.4 ms per leaf for zero information.
        `tests/test_heuristic_prior.py::test_cheap_equals_the_dry_run_on_a_plain_board`
        pins the equality rather than trusting this comment.
        """
        g = self.game
        if getattr(g, "jokers", None) or getattr(g, "plasma", False):
            return False
        return all(c.enhancement == "None" and c.edition == "None" and c.seal == "None"
                   for c in self.hand)

    # ── play ────────────────────────────────────────────────────────────────

    def play_scores(self, combos: list) -> list:
        """Two-tier score for each play subset, in the order given."""
        hand, n, flags = self.hand, self.n, self.flags
        levels, chips_of, base_cache = self.levels, self._chips, {}
        evald = []
        cheap = np.zeros(len(combos), dtype=np.float64)
        for i, combo in enumerate(combos):
            cards = [hand[j] for j in combo if j < n]
            if not cards:
                evald.append(None)
                continue
            try:
                ht, scoring = evaluate_hand(cards, **flags)
            except Exception:                       # noqa: BLE001 - a bad subset is worth 0
                evald.append(None)
                continue
            base = base_cache.get(ht)
            if base is None:
                base = base_cache[ht] = _hand_base(ht, levels)
            total = base[0]
            for c in scoring:
                total += chips_of.get(id(c)) or _card_chips(c)
            s = total * base[1]
            evald.append((cards, ht, scoring))
            cheap[i] = s
            self._cheap[combo] = s

        out = cheap.copy()
        n_exact = 0 if self.cheap_is_exact else min(int(self.cfg.exact_top), len(combos))
        if n_exact > 0:
            order = np.argsort(-cheap, kind="stable")[:n_exact]
            scorer = HypotheticalScorer(self.game, model_held=True)
            for i in order:
                item = evald[int(i)]
                if item is None:
                    continue
                cards, ht, scoring = item
                try:
                    out[int(i)] = float(scorer.score(cards, ht, scoring))
                except Exception:                   # noqa: BLE001 - keep the cheap value
                    pass
        return [float(v) for v in out]

    # ── discard ─────────────────────────────────────────────────────────────

    def discard_scores(self, combos: list, play_combos: list) -> list:
        """`discard_bias * max(floor, draw)` for each discard subset, in order."""
        if not combos:
            return []
        n = self.n
        full = (1 << n) - 1
        kept_masks = np.empty(len(combos), dtype=np.int64)
        sizes = np.empty(len(combos), dtype=np.int64)
        for i, combo in enumerate(combos):
            m = 0
            for j in combo:
                if j < n:
                    m |= 1 << j
            kept_masks[i] = full & ~m
            sizes[i] = min(5, bin(m).count("1"))

        floor = self._floor_scores(kept_masks, play_combos, full)
        draw = self._draw_potential(kept_masks, sizes)
        best = np.maximum(floor, draw) * float(self.cfg.discard_bias)
        return [float(v) for v in best]

    def _floor_scores(self, kept_masks: np.ndarray, play_combos: list,
                      full: int) -> np.ndarray:
        """max cheap(S) over legal play subsets S contained in each kept mask.

        Exact max-over-submasks DP: best[m] = max(base[m], max_i best[m without bit i]).
        2**n * n work, with n <= 12 in practice.
        """
        size = full + 1
        best = np.zeros(size, dtype=np.float64)
        for combo in play_combos:
            m = 0
            ok = True
            for j in combo:
                if j >= self.n:
                    ok = False
                    break
                m |= 1 << j
            if not ok:
                continue
            v = self._cheap.get(combo)
            if v is not None and v > best[m]:
                best[m] = v
        idx = np.arange(size)
        for bit in range(self.n):
            has = (idx & (1 << bit)) != 0
            np.maximum(best, np.where(has, best[idx & ~(1 << bit)], 0.0), out=best)
        return best[kept_masks]

    def _draw_potential(self, kept_masks: np.ndarray, sizes: np.ndarray) -> np.ndarray:
        """max over draw targets of value(T) * P(complete T with `sizes` fresh cards)."""
        game = self.game
        deck = list(getattr(game, "deck", ()) or ())
        n_deck = len(deck)
        if n_deck == 0:
            return np.zeros(len(kept_masks), dtype=np.float64)
        avg_chips = float(np.mean([_card_chips(c) for c in deck]))

        four_fingers = bool(self.flags.get("four_fingers"))
        flush_need = 4 if four_fingers else 5
        straight_need = 4 if four_fingers else 5

        # Per-card suit / rank membership as bitmasks over the hand.
        suit_bits: dict = {}
        rank_bits: dict = {}
        for j, c in enumerate(self.hand):
            if c.enhancement == "Stone":
                continue                     # never contributes to a hand type
            suit_bits[c.suit] = suit_bits.get(c.suit, 0) | (1 << j)
            rank_bits[c.rank] = rank_bits.get(c.rank, 0) | (1 << j)

        deck_suits: dict = {}
        deck_ranks: dict = {}
        for c in deck:
            if c.enhancement == "Stone":
                continue
            deck_suits[c.suit] = deck_suits.get(c.suit, 0) + 1
            deck_ranks[c.rank] = deck_ranks.get(c.rank, 0) + 1

        pc = _popcount_table(self.n)
        out = np.zeros(len(kept_masks), dtype=np.float64)
        d_idx = np.clip(sizes, 0, 5)
        tables: dict = {}

        def tail(need: np.ndarray, p: float) -> np.ndarray:
            key = round(p, 6)
            t = tables.get(key)
            if t is None:
                t = tables[key] = _tail_table(key)
            return t[d_idx, np.clip(need, 0, 6)]

        # ── Flush: every suit held, at its own completion count.
        flush_value = self._target_value("Flush", flush_need, avg_chips)
        for suit, bits in suit_bits.items():
            have = pc[kept_masks & bits]
            need = np.maximum(0, flush_need - have)
            p = deck_suits.get(suit, 0) / n_deck
            np.maximum(out, flush_value * tail(need, p), out=out)

        # ── Straight: every 5-rank window, at its own completion count.
        present = {r: ((kept_masks & b) != 0).astype(np.int64)
                   for r, b in rank_bits.items()}
        straight_value = self._target_value("Straight", straight_need, avg_chips)
        for window in _STRAIGHT_WINDOWS:
            have = None
            useful = 0
            for r in window:
                if r in present:
                    have = present[r] if have is None else have + present[r]
                else:
                    useful += deck_ranks.get(r, 0)
            if have is None:
                continue                 # no rank of this window is held: not a draw
            need = np.maximum(0, straight_need - have)
            np.maximum(out, straight_value * tail(need, useful / n_deck), out=out)

        # ── n-of-a-kind: per held rank, each tier up from what is kept.
        tiers = [(t, self._target_value(name, t, avg_chips))
                 for t, name in ((2, "Pair"), (3, "Three of a Kind"),
                                 (4, "Four of a Kind"))]
        for rank, bits in rank_bits.items():
            have = pc[kept_masks & bits]
            p = deck_ranks.get(rank, 0) / n_deck
            for tier, value in tiers:
                np.maximum(out, value * tail(np.maximum(0, tier - have), p), out=out)
        return out

    def _target_value(self, hand_type: str, n_scoring: int, avg_chips: float) -> float:
        base_chips, base_mult = _hand_base(hand_type, self.levels)
        return (base_chips + n_scoring * avg_chips) * base_mult


def _popcount_table(n: int) -> np.ndarray:
    """popcount for every mask over `n` cards. `n` vectorised passes — the obvious
    `pc[1:] = pc[idx >> 1] + (idx & 1)` is WRONG in numpy (the right-hand side is read
    from the all-zero array, not accumulated), which cost an afternoon."""
    size = 1 << n
    idx = np.arange(size, dtype=np.int64)
    pc = np.zeros(size, dtype=np.int64)
    for bit in range(n):
        pc += (idx >> bit) & 1
    return pc


def _tail_table(p: float) -> np.ndarray:
    """`T[d, m] = P(X >= m)` for `X ~ Binom(d, p)`, d in 0..5, m in 0..6.

    Only 42 numbers, and `d` (the discard size) and `m` (the cards still needed) both
    live in that range for every draw target — so one scalar table per target turns 218
    per-discard probabilities into a single fancy-index. This is the difference between
    2.2 ms and 0.2 ms of `_draw_potential`.
    """
    p = min(1.0, max(0.0, float(p)))
    q = 1.0 - p
    t = np.zeros((6, 7), dtype=np.float64)
    for d in range(6):
        pmf = [_BINOM[d, j] * (p ** j) * (q ** (d - j)) for j in range(d + 1)]
        acc = 0.0
        for m in range(d, -1, -1):
            acc += pmf[m]
            t[d, m] = min(1.0, acc)
        t[d, 0] = 1.0
    return t


def _binom_tail(d: np.ndarray, m: np.ndarray, p: float) -> np.ndarray:
    """P(X >= m) for X ~ Binom(d, p), vectorised over per-discard (d, m). d <= 5."""
    return _tail_table(p)[np.clip(d, 0, 5), np.clip(m, 0, 6)]


# ═══════════════════════════════════════════════════════════════════ prior shaping

def hand_action_scores(game, priors: dict, cfg: HeuristicConfig = DEFAULT_CONFIG
                       ) -> Optional[dict]:
    """`{ActionKey: heuristic score}` for the `play` / `discard` keys of `priors`.

    Returns None when the state has no hand actions (every state that is not
    `SELECTING_HAND`, and a `SELECTING_HAND` with an empty or absurd hand).
    Keys are taken FROM `priors`, never re-enumerated, so boss restrictions
    (`bl_psychic`'s 5-cards-only) and `discards_left == 0` are respected by
    construction.
    """
    play = [k for k in priors if k and k[0] == "play"]
    disc = [k for k in priors if k and k[0] == "discard"]
    if not play and not disc:
        return None
    h = HandHeuristic(game, cfg)
    if not h.usable:
        return None
    play_combos = [tuple(k[1]) for k in play]
    disc_combos = [tuple(k[1]) for k in disc]
    scores: dict = {}
    for k, v in zip(play, h.play_scores(play_combos)):
        scores[k] = v
    if disc:
        for k, v in zip(disc, h.discard_scores(disc_combos, play_combos)):
            scores[k] = v
    return scores


def heuristic_distribution(priors: dict, scores: dict, tau: float) -> dict:
    """`h` from the module docstring: the net's hand-action mass, redistributed by score."""
    hand_keys = [k for k in priors if k in scores]
    if not hand_keys:
        return dict(priors)
    w_h = float(sum(priors[k] for k in hand_keys))
    logits = np.array([math.log1p(max(0.0, scores[k])) for k in hand_keys],
                      dtype=np.float64)
    logits /= max(1e-6, float(tau))
    logits -= logits.max()
    w = np.exp(logits)
    total = float(w.sum())
    if not np.isfinite(total) or total <= 0.0:
        w = np.ones(len(hand_keys), dtype=np.float64)
        total = float(len(hand_keys))
    out = dict(priors)
    for k, wi in zip(hand_keys, w):
        out[k] = w_h * float(wi) / total
    return out


def shape_priors(game, priors: dict, lam: float = 0.0, max_candidates: int = 0,
                 cfg: HeuristicConfig = DEFAULT_CONFIG) -> dict:
    """`(1 - lam) * net + lam * heuristic`, optionally pruned to the top candidates.

    Returns `priors` ITSELF (same object) when there is nothing to do, so a search with
    `lam == 0` and `max_candidates == 0` is byte-identical to one built before this
    module existed.
    """
    if not priors or (lam <= 0.0 and max_candidates <= 0):
        return priors
    scores = hand_action_scores(game, priors, cfg)
    if scores is None:
        return priors

    if lam <= 0.0:
        mixed = priors
    else:
        h = heuristic_distribution(priors, scores, cfg.tau)
        mixed = (h if lam >= 1.0 else
                 {k: (1.0 - lam) * priors[k] + lam * h[k] for k in priors})

    keep = _survivors(priors, scores, max_candidates)
    if keep is None:
        keep = set(priors)
    total = float(sum(mixed[k] for k in keep)) if keep else 0.0
    if not keep or not math.isfinite(total) or total <= 0.0:
        # Degenerate (every survivor got zero mass): fall back to a uniform prior over
        # the survivors rather than returning something the search cannot descend into.
        keys = [k for k in priors if k in keep] or list(priors)
        return {k: 1.0 / len(keys) for k in keys}
    return {k: mixed[k] / total for k in priors if k in keep}


def _survivors(priors: dict, scores: dict, max_candidates: int) -> Optional[set]:
    """Top-K `play` + top-K `discard` by score, plus every non-hand action. None = all."""
    if max_candidates <= 0:
        return None
    keep = set()
    for atype in ("play", "discard"):
        cand = [k for k in priors if k and k[0] == atype and k in scores]
        if len(cand) > max_candidates:
            cand.sort(key=lambda k: (-scores[k], k[1]))
            cand = cand[:max_candidates]
        keep.update(cand)
    keep.update(k for k in priors if k not in scores)
    return keep


__all__ = [
    "HeuristicConfig", "DEFAULT_CONFIG", "HandHeuristic", "hand_action_scores",
    "heuristic_distribution", "shape_priors", "MAX_HAND_FOR_HEURISTIC",
    "ActionKey",
]
