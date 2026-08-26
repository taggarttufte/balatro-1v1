"""registry.py — the hand-off format between "an LLM read the Lua" and "the harness
checked it" (W-ENCODE-POC, 2026-08-26).

Every entry is one game item.  ``predict(summary) -> float`` is a pure function of a
``dict`` **state summary** (built by ``verify.summarize`` plus whatever the scenario adds);
it returns a number in ``Entry.unit``.  Nothing here imports the engine — a registry entry
must be readable, diffable and testable without a game running, which is the whole point of
splitting "the knowledge" from "the player".

The three kinds
---------------
``round_econ``  dollars this item pays at ONE end-of-round evaluation.
``use_value``   dollars (or units) one *use* of a consumable yields.
``buy_value``   what owning the item is worth going forward.  For the scaling tier that is
                deliberately **the item's own scaling state at a horizon**, not a
                win-probability: the shop's blind spot is that a Green Joker bought now has
                no immediate strength, and the number the buy logic is missing is "how big
                will it be in N hands".  Converting mult/chips into P(win) is V's job, not
                the encode layer's (brief §1, the ENCODE/LEARN split).

The three tiers
---------------
``deterministic``       the value is a closed form of the summary; the harness should
                        reproduce it exactly.
``stochastic``          the value is an expectation over engine RNG (1-in-k procs).  No POC
                        item is in this tier -- every 1-in-k money proc was already encoded
                        by W-EXTRACT (``ev/EXTRACT_NOTES.md`` §2), and the POC set was
                        chosen to be disjoint from it.  The tier exists so the fleet's
                        first stochastic entry has a slot and so the harness's CI band has
                        a documented consumer.
``policy_conditional``  the value depends on what the POLICY will do (how many hands, how
                        many discards, how often a scoring hand holds a face).  These are
                        the entries the harness exists for: the closed form is only as good
                        as its rate constants, and the harness measures those rates.

Negative controls
-----------------
Two entries at the bottom are deliberately wrong and are expected to be REJECTED.  A POC in
which everything passes proves nothing.  They are excluded from ``REGISTRY`` and live in
``NEGATIVE_CONTROLS``; ``ALL_ENTRIES`` is the union.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

KINDS = ("buy_value", "round_econ", "use_value")
TIERS = ("deterministic", "stochastic", "policy_conditional")
UNITS = ("dollars", "mult", "chips")

#: verification mode each entry asks the harness for (see ``verify.MODES``).
MODES = ("round_end_paired", "use_paired", "scaling_trajectory", "rollout_paired")


@dataclass(frozen=True)
class Entry:
    """One item's analytic value function plus everything needed to audit it."""

    key: str                                  # the game key, e.g. "j_cloud_9"
    name: str                                 # the game's display name
    kind: str                                 # KINDS
    tier: str                                 # TIERS
    unit: str                                 # UNITS
    predict: Callable[[Mapping], float]       # state summary -> value in `unit`
    modes: tuple                              # which verification modes apply, in order
    assumptions: tuple                        # everything the closed form takes on faith
    lua: tuple                                # file:line citations into _reference/balatro_src
    generated_by: str                         # provenance of the closed form
    engine: str = ""                          # where the engine implements it
    sign_of_delta: int = 0                    # +1 grows, -1 decays, 0 no claim (scaling tier)
    expect_reject: bool = False               # negative controls only
    notes: str = ""
    #: the key the ENGINE knows the item by.  Normally ``key``; a negative control names a
    #: different registry key from the item it lies about, and the reachability probe must
    #: follow the real one — otherwise every control would be rejected as "unreachable"
    #: for the trivial reason that its own key is not a joker, which would prove nothing.
    engine_key_override: str = ""

    @property
    def engine_key(self) -> str:
        return self.engine_key_override or self.key

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"{self.key}: kind {self.kind!r} not in {KINDS}")
        if self.tier not in TIERS:
            raise ValueError(f"{self.key}: tier {self.tier!r} not in {TIERS}")
        if self.unit not in UNITS:
            raise ValueError(f"{self.key}: unit {self.unit!r} not in {UNITS}")
        if not self.modes or any(m not in MODES for m in self.modes):
            raise ValueError(f"{self.key}: modes {self.modes!r} not a subset of {MODES}")
        if not self.lua:
            raise ValueError(f"{self.key}: an entry with no Lua citation is not auditable")
        if not self.assumptions:
            raise ValueError(f"{self.key}: an entry with no assumptions is lying")
        if self.sign_of_delta not in (-1, 0, 1):
            raise ValueError(f"{self.key}: sign_of_delta must be -1, 0 or +1")


# ════════════════════════════════════════════════════════ tier 1 — round economy
# All three pay through ``Card:calculate_dollar_bonus`` (card.lua:1655-1683), whose rows are
# added by ``G.FUNCS.evaluate_round`` (state_events.lua:1174-1181) AFTER the interest row is
# computed from ``G.GAME.dollars`` (:1191).  So none of this money compounds into the SAME
# round's interest -- which the harness's paired round-end measurement checks by construction
# (the two arms differ only by the item, and both pay their interest first).

def _predict_cloud_9(s: Mapping) -> float:
    """$1 per 9 in the full deck, at every end of round."""
    return 1.0 * float(s["deck_nines"])


def _predict_rocket(s: Mapping) -> float:
    """$`dollars` at every end of round; the +$2 upgrade lands BEFORE this round pays.

    ``Card:calculate_joker{end_of_round = true}`` runs inside ``end_round()``
    (state_events.lua:101) and ``calculate_dollar_bonus`` inside ``evaluate_round``
    (:1174), so a Boss round pays the *upgraded* figure, not the pre-boss one.
    """
    return float(s["rocket_dollars"]) + (2.0 if s.get("blind_is_boss") else 0.0)


def _predict_satellite(s: Mapping) -> float:
    """$1 per UNIQUE Planet card used this run (the whole run, not since purchase)."""
    return 1.0 * float(s["unique_planets_used"])


# ═══════════════════════════════════════════ tier 2 — policy-conditional scaling
# The shop blind spot: none of these three has any immediate strength, so the dry-run
# scorer prices them at ~0 the moment they are offered.  What the buy logic needs is the
# trajectory, and the trajectory is a function of the POLICY's hand/discard rates.

def _predict_ride_the_bus(s: Mapping) -> float:
    """Expected +Mult after ``hands_ahead`` scoring hands.

    The chain is ``m <- 0`` if the SCORING hand contains a face, else ``m <- m + 1``
    (card.lua:3525-3540) -- so ``m_n`` is the length of the trailing run of face-free hands.
    With per-hand face probability ``p`` and ``q = 1 - p``::

        E[m_n] = q^n (m_0 + n) + p * sum_{k=0}^{n-1} k q^k

    (all ``n`` hands face-free -> ``m_0 + n``; otherwise the last face was ``k`` hands ago).
    """
    n = int(s["hands_ahead"])
    m0 = float(s["rtb_mult"])
    p = float(s["face_hand_rate"])
    if n <= 0:
        return m0
    q = 1.0 - p
    tail = sum(k * (q ** k) for k in range(n))
    return (q ** n) * (m0 + n) + p * tail


def _predict_green_joker(s: Mapping) -> float:
    """+1 Mult per hand played, -1 per discard ACTION, floored at 0.

    ``max(0, m0 + hands - discards)`` is exact **only if the walk never touches the floor**;
    when it does the true value is strictly larger, so this predictor is biased LOW.  The
    assumption is listed and the harness measures the bias.
    """
    return max(0.0, float(s["green_mult"]) + float(s["hands_ahead"]) - float(s["discards_ahead"]))


def _predict_ice_cream(s: Mapping) -> float:
    """The NEGATIVE scaler: +100 chips, -5 per hand played, then the joker melts.

    Returns the chips the joker will still be worth after ``hands_ahead`` hands.  The decay
    is what makes it a buy-value trap -- ``sign_of_delta = -1`` on this entry is checked
    separately by the harness, so a predictor that came out flat or rising would be rejected
    even if its level happened to land inside the band.
    """
    return max(0.0, float(s["ice_chips"]) - 5.0 * float(s["hands_ahead"]))


# ══════════════════════════════════════════════════ tier 3 + 4 — consumable, voucher

def _predict_hermit(s: Mapping) -> float:
    """Doubles money, capped at a +$20 GAIN (not a $20 balance)."""
    return float(max(0, min(int(s["dollars"]), 20)))


def _predict_seed_money(s: Mapping) -> float:
    """Extra interest dollars per round end from raising the cap $25 -> $50 of balance.

    Lua stores the cap in BALANCE dollars (``interest_cap = 25`` base, game.lua:1909; Seed
    Money sets it to its ``config.extra = 50``, game.lua:602 via card.lua:1933) and pays
    ``interest_amount * min(floor(dollars/5), interest_cap/5)``.  The engine stores the same
    cap in INTEREST dollars (5 -> 10); the two agree everywhere.
    """
    steps = int(s["dollars"]) // 5
    return float(min(steps, 10) - min(steps, 5))


REGISTRY: dict = {}


def _add(entry: Entry, into: dict) -> Entry:
    if entry.key in into:
        raise ValueError(f"duplicate registry key {entry.key!r}")
    into[entry.key] = entry
    return entry


_add(Entry(
    key="j_cloud_9", name="Cloud 9", kind="round_econ", tier="deterministic", unit="dollars",
    predict=_predict_cloud_9,
    modes=("round_end_paired", "rollout_paired"),
    assumptions=(
        "nine_tally is the count over the WHOLE deck (G.playing_cards), not the draw pile — "
        "it is recomputed every frame in Card:update, so it tracks cards added/destroyed",
        "Stone-enhanced cards do NOT count: Card:get_id returns a random negative id for "
        "them (card.lua:957-962), so a Stone card that used to be a 9 is invisible",
        "the payout does not compound into the same round's interest (evaluate_round order)",
        "the joker is not debuffed (calculate_dollar_bonus returns early, card.lua:1656)",
    ),
    lua=("card.lua:1661-1663 (calculate_dollar_bonus)",
         "card.lua:4191-4196 (Card:update recomputes nine_tally over G.playing_cards)",
         "card.lua:957-962 (Card:get_id — Stone cards are id-less)",
         "game.lua:444 (config.extra = 1)",
         "functions/state_events.lua:1174-1181 (the joker dollar rows)"),
    engine="engine/balatro_sim/jokers/misc.py:219-226 (_Cloud9), game.py:2026-2032 "
           "(deck_nines precomputed and pushed into every joker's state)",
    generated_by="W-ENCODE-POC (Opus 5) reading card.lua + game.lua, 2026-08-26",
    notes="The simplest possible round_econ entry; used as the negative control's twin.",
), REGISTRY)

_add(Entry(
    key="j_rocket", name="Rocket", kind="round_econ", tier="deterministic", unit="dollars",
    predict=_predict_rocket,
    modes=("round_end_paired",),
    assumptions=(
        "the +$2 upgrade fires on the end_of_round of a BOSS blind and lands BEFORE that "
        "same round's dollar row is read (state_events.lua:101 precedes :1174)",
        "the upgrade is once per boss ROUND, not per boss ability trigger",
        "no cap: the payout grows without bound over a long run",
    ),
    lua=("card.lua:1664-1666 (calculate_dollar_bonus)",
         "card.lua:2896-2902 (end_of_round + G.GAME.blind.boss -> extra.dollars += increase)",
         "game.lua:445 (config.extra = {dollars = 1, increase = 2})",
         "functions/state_events.lua:101 (end_round fires the end_of_round joker context)",
         "functions/state_events.lua:1174-1181 (evaluate_round reads calculate_dollar_bonus)"),
    engine="engine/balatro_sim/jokers/economy.py:43-50 (_Rocket), game.py:2021-2032 "
           "(on_boss_beaten fires before on_round_end — the same order as the Lua)",
    generated_by="W-ENCODE-POC (Opus 5) reading card.lua + state_events.lua, 2026-08-26",
    notes="The ordering claim is the whole content of this entry: a naive reading pays $1 "
          "on the boss round and $3 after, which is off by $2 at every boss.",
), REGISTRY)

_add(Entry(
    key="j_satellite", name="Satellite", kind="round_econ", tier="deterministic", unit="dollars",
    predict=_predict_satellite,
    modes=("round_end_paired",),
    assumptions=(
        "UNIQUE planets: the Lua counts distinct keys in G.GAME.consumeable_usage whose set "
        "is 'Planet', so using Mercury five times is worth $1, not $5",
        "the count is over the WHOLE RUN, including planets used before this joker was "
        "bought — G.GAME.consumeable_usage is global state, not joker state",
        "zero planets pays nothing at all (an explicit `return` with no value, not $0)",
    ),
    lua=("card.lua:1667-1673 (calculate_dollar_bonus — the consumeable_usage sweep)",
         "game.lua:515 (config.extra = 1)"),
    engine="engine/balatro_sim/jokers/misc.py:208-217 (_Satellite), consumables.py:55-59 "
           "(apply_planet fires on_planet_used) — see POC_NOTES §3 for the fidelity gap",
    generated_by="W-ENCODE-POC (Opus 5) reading card.lua + game.lua, 2026-08-26",
    notes="The 'whole run' assumption is the one the engine breaks.",
), REGISTRY)

_add(Entry(
    key="j_ride_the_bus", name="Ride the Bus", kind="buy_value", tier="policy_conditional",
    unit="mult", predict=_predict_ride_the_bus, sign_of_delta=1,
    modes=("scaling_trajectory",),
    assumptions=(
        "faces are counted over context.scoring_hand, NOT the played set — a face played as "
        "a non-scoring kicker does not reset the counter",
        "Card:is_face honours Pareidolia (every card is a face) and returns nil for a "
        "debuffed card (card.lua:964-970)",
        "the increment happens in context.before, so THIS hand already scores the new value",
        "face occurrence is i.i.d. across hands at rate `face_hand_rate` — it is not: the "
        "policy's hand choice correlates with the deck it has left inside a round",
        "`hands_ahead` is a policy rate the harness must supply; the entry does not know it",
    ),
    lua=("card.lua:3525-3540 (context.before — the reset/increment)",
         "card.lua:3992-3997 (the mult_mod that applies it)",
         "card.lua:964-970 (Card:is_face)",
         "game.lua:413 (config.extra = 1)"),
    engine="engine/balatro_sim/jokers/scaling.py:349-358 (_RideTheBus)",
    generated_by="W-ENCODE-POC (Opus 5) reading card.lua, closed form derived by hand "
                 "(geometric reset chain), 2026-08-26",
), REGISTRY)

_add(Entry(
    key="j_green_joker", name="Green Joker", kind="buy_value", tier="policy_conditional",
    unit="mult", predict=_predict_green_joker, sign_of_delta=1,
    modes=("scaling_trajectory",),
    assumptions=(
        "-1 per discard ACTION, not per discarded card: the hook is gated on "
        "`context.other_card == context.full_hand[#context.full_hand]` (card.lua:2846)",
        "+1 in context.before, so the hand that triggers it already scores the new value",
        "the floor at 0 is never active over the horizon — when it is, the true value is "
        "STRICTLY LARGER than this predictor (so the entry is biased low, not high)",
        "`hands_ahead` and `discards_ahead` are policy rates the harness must supply",
    ),
    lua=("card.lua:3563-3569 (context.before — +hand_add)",
         "card.lua:2846-2856 (discard — max(0, mult - discard_sub), once per discard)",
         "card.lua:4010-4015 (the mult_mod that applies it)",
         "game.lua:428 (config.extra = {hand_add = 1, discard_sub = 1})"),
    engine="engine/balatro_sim/jokers/scaling.py:51-58 (_GreenJoker)",
    generated_by="W-ENCODE-POC (Opus 5) reading card.lua, 2026-08-26",
    notes="Deliberately submitted with a KNOWN bias (the floor) so the harness has "
          "something real to measure rather than a tautology.",
), REGISTRY)

_add(Entry(
    key="j_ice_cream", name="Ice Cream", kind="buy_value", tier="policy_conditional",
    unit="chips", predict=_predict_ice_cream, sign_of_delta=-1,
    modes=("scaling_trajectory",),
    assumptions=(
        "the chips are applied at joker_main and decremented in context.after, so the "
        "CURRENT hand gets the pre-decrement value (card.lua:3915-3920 then :3570-3599)",
        "the decrement is per HAND PLAYED, not per round",
        "the joker is DESTROYED when the decrement would take it to <= 0 (card.lua:3571-3592) "
        "— it does not linger at 0 chips occupying a slot",
        "`hands_ahead` is a policy rate the harness must supply",
    ),
    lua=("card.lua:3570-3599 (context.after — decrement, and the melt at <= 0)",
         "card.lua:3915-3920 (chip_mod = extra.chips)",
         "game.lua:420 (config.extra = {chips = 100, chip_mod = 5})"),
    engine="engine/balatro_sim/jokers/scaling.py:305-312 (_IceCream) — floors at 0 instead "
           "of melting; see POC_NOTES §3",
    generated_by="W-ENCODE-POC (Opus 5) reading card.lua + game.lua, 2026-08-26",
), REGISTRY)

_add(Entry(
    key="c_hermit", name="The Hermit", kind="use_value", tier="deterministic", unit="dollars",
    predict=_predict_hermit,
    modes=("use_paired",),
    assumptions=(
        "the GAIN is capped at $20 (config.extra), not the resulting balance — at $30 the "
        "player ends on $50, not $40",
        "negative money yields 0 (the math.max(0, ...) guard), it does not double a debt",
        "no target card is consumed; the tarot is usable with an empty hand",
    ),
    lua=("card.lua:1385-1391 (ease_dollars(math.max(0, math.min(G.GAME.dollars, extra))))",
         "card.lua:1530 (usable with no highlighted cards)",
         "game.lua:542 (config.extra = 20)"),
    engine="engine/balatro_sim/consumables.py:163-169",
    generated_by="W-ENCODE-POC (Opus 5) reading card.lua, 2026-08-26",
    notes="The tarot tier's stand-in.  ev/hand.py currently prices ANY created tarot at a "
          "flat $4 (`tarot_value_dollars`, EXTRACT_NOTES §7) — this entry is worth $20 at "
          "a $20+ balance, i.e. 5x the flat constant, which is the size of the blind spot.",
), REGISTRY)

_add(Entry(
    key="v_seed_money", name="Seed Money", kind="round_econ", tier="deterministic",
    unit="dollars", predict=_predict_seed_money,
    modes=("round_end_paired", "rollout_paired"),
    assumptions=(
        "the cap is on the BALANCE that earns interest ($25 -> $50), which at $1 per $5 is "
        "$5 -> $10 of interest per round",
        "the marginal value is ZERO below a $30 balance and maxes out at $50 — it is a "
        "step function of money held, not the flat +0.02 the shop rules give a voucher",
        "interest is computed on the balance BEFORE the round's payout rows",
        "the Green Deck's no_interest modifier zeroes the whole row",
    ),
    lua=("card.lua:1933 (G.GAME.interest_cap = center_table.extra)",
         "game.lua:602 (v_seed_money config.extra = 50)",
         "game.lua:1909-1910 (base interest_cap = 25, interest_amount = 1)",
         "functions/state_events.lua:1191-1202 (the interest row)"),
    engine="engine/balatro_sim/consumables.py:520-522, game.py:2006 "
           "(cap stored in interest dollars, 5 -> 10 — equivalent)",
    generated_by="W-ENCODE-POC (Opus 5) reading card.lua + state_events.lua, 2026-08-26",
    notes="The voucher tier's stand-in, and the one item whose per-round value is exact but "
          "whose BUY value is not: the harness measures the latter (rollout_paired).",
), REGISTRY)


# ══════════════════════════════════════════════════════════ negative controls
# Required by the brief: a POC where everything passes proves nothing.  Both of these must
# be REJECTED by the harness, for two different reasons.

def _predict_cloud_9_x3(s: Mapping) -> float:
    """Deliberately wrong: 3x the true payout.  A plausible-looking mis-read of
    ``config.extra`` -- the kind of error an LLM makes when it grabs the wrong joker's
    number.  The harness must catch a pure MAGNITUDE error."""
    return 3.0 * float(s["deck_nines"])


def _predict_joker_double_count(s: Mapping) -> float:
    """Deliberately wrong: the plain Joker (+4 Mult) priced as if it ALSO paid $4 a round.

    This is the double-count failure mode the marginal rule exists for.  The +4 Mult is
    already priced by the dry-run scorer (``HypotheticalScorer``, EV_NOTES §1), so any
    encode entry that re-prices it is adding value the player already has.  Because the
    harness measures the MARGINAL realized dollars with everything else active, the entry's
    claim collapses to a measured $0 and the band check rejects it.
    """
    return 4.0


NEGATIVE_CONTROLS: dict = {}

_add(Entry(
    key="j_cloud_9__x3", name="Cloud 9 (x3, WRONG)", kind="round_econ", tier="deterministic",
    unit="dollars", predict=_predict_cloud_9_x3, expect_reject=True,
    engine_key_override="j_cloud_9",
    modes=("round_end_paired",),
    assumptions=("deliberately wrong by a factor of 3 — a magnitude error",),
    lua=("card.lua:1661-1663 (the entry cites the RIGHT Lua and still gets the number wrong "
         "— which is exactly why a citation is not a verification)",),
    engine="engine/balatro_sim/jokers/misc.py:219-226",
    generated_by="W-ENCODE-POC negative control #1 (magnitude)",
    notes="Rejection reason should be BAND (3x > the 2x band) and CI (the paired "
          "round-end measurement has zero variance).",
), NEGATIVE_CONTROLS)

_add(Entry(
    key="j_joker__doublecount", name="Joker (double-counted, WRONG)", kind="round_econ",
    tier="deterministic", unit="dollars", predict=_predict_joker_double_count,
    expect_reject=True, engine_key_override="j_joker",
    modes=("round_end_paired",),
    assumptions=("deliberately wrong in KIND — it re-prices a scoring effect the dry-run "
                 "scorer already handles, and in the wrong currency",),
    lua=("game.lua:368 (j_joker config = {mult = 4}) — a real +4 Mult, and no dollar row "
         "anywhere in Card:calculate_dollar_bonus (card.lua:1655-1683)",),
    engine="engine/balatro_sim/jokers/mult.py:21-24 (_Joker: ctx.mult += 4, no money)",
    generated_by="W-ENCODE-POC negative control #2 (double count)",
    notes="Rejection reason should be BAND with measured == 0: the marginal round-end "
          "delta between owning and not owning j_joker is exactly $0.",
), NEGATIVE_CONTROLS)


ALL_ENTRIES: dict = {**REGISTRY, **NEGATIVE_CONTROLS}


def entries(include_controls: bool = True) -> Sequence[Entry]:
    src = ALL_ENTRIES if include_controls else REGISTRY
    return list(src.values())


def by_mode(mode: str, include_controls: bool = True) -> Sequence[Entry]:
    return [e for e in entries(include_controls) if mode in e.modes]
