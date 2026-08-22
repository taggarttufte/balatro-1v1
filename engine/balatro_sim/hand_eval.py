"""
hand_eval.py — Poker hand evaluation for Balatro.

A port of ``evaluate_poker_hand`` + ``get_X_same`` / ``get_flush`` / ``get_straight`` /
``get_highest`` (functions/misc_functions.lua:376-620), including the joker flags the
real functions read through ``find_joker``:

  four_fingers  Four Fingers  — flushes and straights need 4 cards (get_flush /
                get_straight size floor ``5 - 1``).  The 5th card of a 5-card play is NOT
                part of the scoring hand unless it is a Stone (or Splash adds it).
  shortcut      Shortcut — a straight may skip ONE rank between consecutive present
                ranks (``can_skip`` / ``skipped_rank`` in get_straight); the Ace still
                counts low and high, and there is no wrap-around.
  smeared       Smeared Joker — Hearts≡Diamonds and Spades≡Clubs for ``is_suit(...,
                flush_calc=true)``; only the flush test changes.
  pareidolia    Pareidolia — accepted for API symmetry; face-ness does not influence
                the hand TYPE in the real game either (``is_face`` is not consulted by
                evaluate_poker_hand), so it is a no-op here.  Face-dependent effects read
                ``ScoreContext.is_face_card``.

Stone cards: ``Card:get_id()`` returns a negative random number and ``is_suit(...,
flush_calc)`` is false for them, so they never contribute to any hand type; they are
appended to the scoring cards afterwards (state_events.lua adds every Stone in
``G.play.cards`` to ``scoring_hand``).  Wild cards (not debuffed) match every suit.

Hand type priority (highest to lowest):
  Flush Five > Flush House > Five of a Kind > Straight Flush >
  Four of a Kind > Full House > Flush > Straight >
  Three of a Kind > Two Pair > Pair > High Card
"""
from typing import Optional
from .card import Card


HAND_PRIORITY = [
    "Flush Five",
    "Flush House",
    "Five of a Kind",
    "Straight Flush",
    "Four of a Kind",
    "Full House",
    "Flush",
    "Straight",
    "Three of a Kind",
    "Two Pair",
    "Pair",
    "High Card",
]

# get_flush iterates the suits in this order and returns the FIRST suit that reaches the
# floor (matters only for which cards score when Wilds could complete two suits).
_FLUSH_SUIT_ORDER = ("Spades", "Hearts", "Clubs", "Diamonds")
_RED = frozenset({"Hearts", "Diamonds"})


def _is_suit_flush(card: Card, suit: str, smeared: bool) -> bool:
    """``Card:is_suit(suit, nil, true)`` (card.lua:4064-4076): Stone never, Wild (not
    debuffed) always, Smeared pairs the red and the black suits."""
    if card.enhancement == "Stone":
        return False
    if card.enhancement == "Wild" and not card.debuffed:
        return True
    if smeared and ((card.suit in _RED) == (suit in _RED)):
        return True
    return card.suit == suit


def _get_flush(cards: list[Card], four_fingers: bool, smeared: bool) -> list[Card]:
    """``get_flush`` (misc_functions.lua:522-546): all cards matching the first suit
    (in ``_FLUSH_SUIT_ORDER``) that has at least ``5 - four_fingers`` of them."""
    floor = 4 if four_fingers else 5
    if len(cards) > 5 or len(cards) < floor:
        return []
    for suit in _FLUSH_SUIT_ORDER:
        t = [c for c in cards if _is_suit_flush(c, suit, smeared)]
        if len(t) >= floor:
            return t
    return []


def _get_straight(cards: list[Card], four_fingers: bool, shortcut: bool) -> list[Card]:
    """``get_straight`` (misc_functions.lua:548-590).  Walks j = 1..14 with j == 1
    standing for the Ace (so A-2-3-4-5 and T-J-Q-K-A both work, nothing wraps);
    Shortcut lets ONE absent rank in a row be skipped (never at j == 14); the run
    keeps extending after the floor is reached and stops at the first real gap."""
    floor = 4 if four_fingers else 5
    if len(cards) > 5 or len(cards) < floor:
        return []
    ids: dict[int, list[Card]] = {}
    for c in cards:
        if c.enhancement == "Stone":          # get_id() < 0 for Stone cards
            continue
        if 1 < c.rank < 15:
            ids.setdefault(c.rank, []).append(c)
    t: list[Card] = []
    straight_length = 0
    straight = False
    skipped_rank = False
    for j in range(1, 15):
        rank = 14 if j == 1 else j
        if rank in ids:
            straight_length += 1
            skipped_rank = False
            t.extend(ids[rank])
        elif shortcut and not skipped_rank and j != 14:
            skipped_rank = True
        else:
            straight_length = 0
            skipped_rank = False
            if not straight:
                t = []
            if straight:
                break
        if straight_length >= floor:
            straight = True
    if not straight:
        return []
    return t


def _get_x_same(num: int, cards: list[Card]) -> list[list[Card]]:
    """``get_X_same`` (misc_functions.lua:592-611): rank groups of EXACTLY ``num`` cards,
    highest rank first, each group in play order.  Stone cards never group."""
    groups: dict[int, list[Card]] = {}
    for c in cards:
        if c.enhancement == "Stone":
            continue
        groups.setdefault(c.rank, []).append(c)
    return [g for r, g in sorted(groups.items(), key=lambda kv: -kv[0]) if len(g) == num]


def _get_highest(cards: list[Card]) -> list[Card]:
    """``get_highest`` (misc_functions.lua:613-622): the card with the greatest nominal.
    Stone cards carry ``-1000 * suit_nominal`` so a Stone is only highest when every
    card is a Stone; the first card wins strict-greater ties."""
    best: Optional[Card] = None
    best_nominal = None
    for c in cards:
        nominal = c.rank if c.enhancement != "Stone" else -1000
        if best is None or nominal > best_nominal:
            best, best_nominal = c, nominal
    return [best] if best is not None else []


# Kept for callers / tests that used the old module-level helpers.
def _is_flush(cards: list[Card], four_fingers: bool = False, smeared: bool = False) -> bool:
    return bool(_get_flush(cards, four_fingers, smeared))


def _is_straight(ranks: list[int], four_fingers: bool = False, shortcut: bool = False) -> bool:
    return bool(_get_straight([Card(rank=r, suit="Spades") for r in ranks], four_fingers, shortcut))


def evaluate_hand(cards: list[Card], *, four_fingers: bool = False, shortcut: bool = False,
                  smeared: bool = False, pareidolia: bool = False) -> tuple[str, list[Card]]:
    """
    Evaluate the best hand type from 1-5 cards (``evaluate_poker_hand(...).top``).
    Returns (hand_type, scoring_cards) where scoring_cards are the cards that make up
    the hand type (the game's ``scoring_hand``) followed by every non-debuffed Stone card.

    Keyword flags mirror the jokers the Lua reads through ``find_joker`` — see the
    module docstring.  ``pareidolia`` is accepted but does not change hand types.
    """
    # Stone cards never contribute to the type (get_id < 0, is_suit false); they score
    # afterwards.  Debuffed Stones are left out of the scoring list as before (they would
    # not score anyway).
    stones = [c for c in cards if c.enhancement == "Stone" and not c.debuffed]
    active = [c for c in cards if c.enhancement != "Stone"]

    if not active:
        return "High Card", list(cards)

    # Size checks in get_flush / get_straight use the whole played hand (#hand), so the
    # full card list goes in; the per-card predicates exclude Stones themselves.
    same5 = _get_x_same(5, cards)
    same4 = _get_x_same(4, cards)
    same3 = _get_x_same(3, cards)
    same2 = _get_x_same(2, cards)
    flush = _get_flush(cards, four_fingers, smeared)
    straight = _get_straight(cards, four_fingers, shortcut)

    if same5 and flush:
        return "Flush Five", same5[0] + stones

    if same3 and same2 and flush:
        return "Flush House", same3[0] + same2[0] + stones

    if same5:
        return "Five of a Kind", same5[0] + stones

    if flush and straight:
        # Union, flush cards first (misc_functions.lua:426-437).  With Four Fingers the
        # two 4-card subsets may differ, so all 5 cards can score.
        ret = list(flush)
        for c in straight:
            if c not in ret:
                ret.append(c)
        return "Straight Flush", ret + stones

    if same4:
        return "Four of a Kind", same4[0] + stones

    if same3 and same2:
        return "Full House", same3[0] + same2[0] + stones

    if flush:
        return "Flush", flush + stones

    if straight:
        return "Straight", straight + stones

    if same3:
        return "Three of a Kind", same3[0] + stones

    if len(same2) == 2 or (len(same3) == 1 and len(same2) == 1):
        second = same2[1] if len(same2) > 1 else same3[0]
        return "Two Pair", same2[0] + second + stones

    if same2:
        return "Pair", same2[0] + stones

    return "High Card", _get_highest(active) + stones


def best_hand_from_subset(cards: list[Card], play_count: int = 5, **flags) -> tuple[str, list[Card]]:
    """
    Find the best hand type achievable by playing `play_count` cards
    from the given set. Tries all combinations.  ``flags`` are forwarded to
    ``evaluate_hand`` (four_fingers / shortcut / smeared).
    """
    from itertools import combinations
    n = min(play_count, len(cards))
    best_type = "High Card"
    best_cards = cards[:1]

    for r in range(1, n + 1):
        for combo in combinations(cards, r):
            hand_type, scoring = evaluate_hand(list(combo), **flags)
            if HAND_PRIORITY.index(hand_type) < HAND_PRIORITY.index(best_type):
                best_type = hand_type
                best_cards = list(combo)

    return best_type, best_cards
