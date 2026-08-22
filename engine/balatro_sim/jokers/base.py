"""
base.py — Joker base class, registry, keyed-RNG accessors and hook helpers.

Phase 1 W3 (effect-roll keys): every stochastic joker / card effect draws from
the run's keyed ``PseudoRandom`` (``game.run_state.rng``) through
``ScoreContext.prng`` with the exact key string the real game uses
(mp/rng/keys.py). There is NO unseeded fallback: a hook that needs a roll and
has no context raises. The legacy single-stream ``game.rng`` is gone.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..game import ScoreContext

class _JokerRegistry(dict):
    """
    key -> singleton effect object. Keys are the GAME keys from mp/rng/pools.py
    (`j_joker`, `j_ring_master`, ...). Registering the same key twice is an
    error: before the Phase 1 re-key 36 keys were registered in two modules
    with "last import wins" semantics, and the winner was the wrong
    implementation for several of them (8 Ball, Four Fingers, Satellite, ...).
    """
    def __setitem__(self, key, value):
        if key in self:
            raise KeyError(f"joker {key!r} registered twice (second in {value!r}); "
                           "one implementation per game key")
        super().__setitem__(key, value)


JOKER_REGISTRY: dict[str, object] = _JokerRegistry()


class MissingPRNG(RuntimeError):
    """A keyed effect roll was requested without a PseudoRandom on the context."""


_PRNG_METHODS = ("pseudorandom", "pseudorandom_element", "pseudoshuffle")


def is_prng(obj) -> bool:
    """True if ``obj`` has the three draw methods of ``mp.rng.core.PseudoRandom``."""
    return obj is not None and all(callable(getattr(obj, m, None)) for m in _PRNG_METHODS)


def rng_of(ctx) -> "object":
    """
    Resolve the keyed PseudoRandom from a ScoreContext.

    Every stochastic effect must draw from the run's ``PseudoRandom`` with the
    game's key string (``ctx.prng.pseudorandom('lucky_mult')`` ...). A hook
    called without a context, or with a context that carries no PRNG, raises
    ``MissingPRNG`` — there is deliberately no unseeded fallback any more
    (the pre-W3 fallback to the global ``random`` module hid exactly the class
    of wiring bug this function now surfaces).
    """
    prng = getattr(ctx, "prng", None)
    if prng is None:
        raise MissingPRNG(
            "effect roll requested without a PseudoRandom: pass rng=game.run_state.rng "
            "to score_hand() / build the context with game._hook_ctx()")
    return prng


def prob_roll(ctx, key: str, odds: float) -> bool:
    """The game's probability idiom: ``pseudorandom(key) < G.GAME.probabilities.normal / odds``
    (card.lua:988 etc.). ``probabilities.normal`` is 2**k for k owned Oops! All 6s."""
    return rng_of(ctx).pseudorandom(key) < ctx.probabilities_normal / odds


# poll_edition() in mp/rng/generate.py returns the game's short names.
GEN_EDITION_TO_ENGINE = {"foil": "Foil", "holo": "Holographic", "polychrome": "Polychrome",
                         "negative": "Negative", None: "None"}


_SMEAR = {"Hearts": "Diamonds", "Diamonds": "Hearts", "Spades": "Clubs", "Clubs": "Spades"}


def is_suit(ctx, card, suit: str) -> bool:
    """``Card:is_suit(suit)`` (card.lua:1035-1049): Stone never, Wild always (unless
    debuffed — callers already skip debuffed cards), Smeared Joker pairs
    Hearts/Diamonds and Spades/Clubs."""
    if card.enhancement == "Stone":
        return False
    if card.enhancement == "Wild":
        return True
    if card.suit == suit:
        return True
    smeared = getattr(ctx, "smear_suits", False) or any(j.key == "j_smeared" for j in ctx.jokers)
    return smeared and _SMEAR.get(card.suit) == suit


def sort_id_order(cards):
    """Runtime Cards in ``sort_id`` (creation) order — what ``pseudorandom_element``
    indexes for Card lists. Engine ``Card.id`` is a global creation counter."""
    return sorted(cards, key=lambda c: c.id)


def joker_sell_value(inst) -> int:
    """Sell value of an owned joker: ``state['sell_value']`` (set on purchase) or the
    catalogue default ``max(1, cost // 2)``."""
    v = inst.state.get("sell_value")
    if v is not None:
        return v
    from ..game_keys import JOKER_BY_KEY
    return max(1, int(JOKER_BY_KEY.get(inst.key, {}).get("cost", 4)) // 2)


def register_joker(key: str):
    """Decorator to register a joker effect by its game key (e.g. 'j_joker')."""
    def decorator(cls):
        JOKER_REGISTRY[key] = cls
        cls.key = key
        return cls
    return decorator


@dataclass
class ScoreContext:
    """Passed to joker trigger functions. Mutated as jokers fire."""
    chips: float = 0.0

    # ── Mult accumulation ─────────────────────────────────────────────────────
    # `mult` and `mult_mult` are the PENDING contribution of the contributor
    # currently being processed (one card modifier, or one joker). scoring.py
    # folds them into `running_mult` after each contributor, in order:
    #
    #     running_mult = (running_mult + mult) * mult_mult
    #
    # That fold is what makes joker position matter — a +Mult joker to the LEFT
    # of an xMult joker gets multiplied, one to the right does not. Before the
    # 2026-07-29 audit every +mult was pooled and every xmult applied at the very
    # end, so the engine silently granted the optimal ordering for free and every
    # score this project ever logged was inflated.
    #
    # Jokers should keep using `ctx.mult += x` / `ctx.mult_mult *= x` as before;
    # the fold is scoring.py's job.
    mult: float = 0.0
    mult_mult: float = 1.0
    running_mult: float = 0.0   # set to the hand's base mult by score_hand()
    hand_type: str = ""

    # ── Card collections ──────────────────────────────────────────────────────
    # These are three DIFFERENT things and jokers must read the right one.
    # Before the 2026-07-29 audit only `all_cards` existed and was set to the
    # played selection, while jokers read it as any of the three — so every
    # "held in hand" joker (Baron, Blackboard, Raised Fist, Shoot the Moon,
    # Reserved Parking) and every "in your full deck" joker (Steel Joker, Stone
    # Joker, Driver's License) was reading the wrong collection.
    scoring_cards: list = field(default_factory=list)  # played cards that score
    all_cards: list = field(default_factory=list)      # ALL played cards
    held_cards: list = field(default_factory=list)     # still in hand, not played
    full_deck: list = field(default_factory=list)      # entire permanent deck

    jokers: list = field(default_factory=list)
    hands_left: int = 0
    discards_left: int = 0
    dollars: int = 0
    ante: int = 1
    deck_remaining: int = 0
    planet_levels: dict = field(default_factory=dict)   # hand_type -> level
    hand_type_counts: dict = field(default_factory=dict)  # run totals, for Supernova

    # ── Keyed RNG + run-level state (Phase 1 W3) ─────────────────────────────
    # `prng` is the run's PseudoRandom (game.run_state.rng) or, in a dry run
    # (card_selection.HypotheticalScorer), a private clone of it. Every roll goes
    # through rng_of(ctx) / prob_roll(ctx, key, odds) with the game's key string.
    prng: object = None
    # The live generate.RunState when scoring for real (card creation, pool
    # flags, used_jokers), None in a dry run (effects that CREATE cards are
    # skipped there; their probability roll is still consumed so the estimate
    # and the real play see the same luck).
    run_state: object = None
    probabilities_normal: float = 1.0   # G.GAME.probabilities.normal (2**n Oops! All 6s)
    # G.GAME.current_round.{idol_card, mail_card, ancient_card, castle_card}:
    # {"idol": (rank, suit), "mail": rank, "ancient": suit, "castle": suit}
    # (see round_cards.py). Shared by reference with the game; re-rolled per round.
    round_cards: dict = field(default_factory=dict)
    joker_slots: int = 5
    consumable_slots: int = 2
    consumables: list = field(default_factory=list)   # held consumable keys (G.consumeables order)
    boss_triggered: bool = False       # G.GAME.blind.triggered this hand (Matador)
    lucky_trigger: bool = False        # set while the jokers see a card whose Lucky roll hit
    blind_kind: str = ""               # "Small" | "Big" | "Boss" for the current blind
    hands_played: int = 0              # G.GAME.current_round.hands_played BEFORE this hand

    # ── Pending side-effects produced by hooks and applied by game.py ────────
    pending_jokers: list = field(default_factory=list)      # JokerInstance to add (Riff-raff, Invisible)
    pending_cards: list = field(default_factory=list)       # (Card, where) to add to the deck ("deck"|"hand")
    pending_destroy: list = field(default_factory=list)     # Cards to destroy (Trading Card, Sixth Sense)

    # ── Retrigger system ──────────────────────────────────────────────────────
    # Maps scoring_card index -> extra retrigger count (0 = score once, 1 = twice, etc.)
    card_retriggers: dict = field(default_factory=dict)

    # ── Hand eval modification flags (set by jokers before scoring) ───────────
    all_face_cards: bool = False        # Pareidolia: treat all cards as face cards
    four_finger_mode: bool = False      # FourFingers: Flush/Straight valid with 4 cards
    smear_suits: bool = False           # SmearedJoker: Hearts=Diamonds, Spades=Clubs
    all_scoring_mode: bool = False      # Splash: all played cards score

    # ── Pending side-effects (collected, applied post-score) ─────────────────
    pending_money: int = 0             # dollars to award after round
    prevent_loss: bool = False         # Mr. Bones
    pending_consumables: list = field(default_factory=list)  # created tarots/planets
    glass_scored: list = field(default_factory=list)   # Glass cards that scored;
    # the caller rolls 1-in-4 destruction on each of these once scoring finishes

    @property
    def n_jokers(self) -> int:
        return len(self.jokers)

    def is_face_card(self, card) -> bool:
        """Respects Pareidolia flag."""
        return self.all_face_cards or card.is_face_card

    def fold_mult(self):
        """
        Commit the pending contribution into the running mult and reset it.

        Called by scoring.py after each individual contributor so that additive
        and multiplicative effects interleave in the real game's order.
        """
        self.running_mult = (self.running_mult + self.mult) * self.mult_mult
        self.mult = 0.0
        self.mult_mult = 1.0


class JokerInstance:
    """
    A joker in the player's joker slots.
    Holds the joker key, runtime state (ability.mult, extra, etc.), and edition.
    """
    _sort_counter = 0

    def __init__(self, key: str, edition: str = "None"):
        self.key = key
        self.edition = edition
        self.state: dict = {}   # runtime state (e.g. {"mult": 0} for scaling jokers)
        # Card.sort_id: creation order, what pseudorandom_element / pseudoshuffle
        # over G.jokers.cards index (Amber Acorn, Crimson Heart, Invisible, Ankh).
        JokerInstance._sort_counter += 1
        self.sort_id = JokerInstance._sort_counter

    def on_score_card(self, card, ctx: ScoreContext):
        """Fires for each scoring card."""
        effect = JOKER_REGISTRY.get(self.key)
        if effect and hasattr(effect, "on_score_card"):
            effect.on_score_card(self, card, ctx)

    def on_hand_scored(self, ctx: ScoreContext):
        """Fires after all scoring cards processed."""
        effect = JOKER_REGISTRY.get(self.key)
        if effect and hasattr(effect, "on_hand_scored"):
            effect.on_hand_scored(self, ctx)

    def on_discard(self, cards, ctx: ScoreContext):
        effect = JOKER_REGISTRY.get(self.key)
        if effect and hasattr(effect, "on_discard"):
            effect.on_discard(self, cards, ctx)

    def on_round_end(self, ctx: ScoreContext):
        effect = JOKER_REGISTRY.get(self.key)
        if effect and hasattr(effect, "on_round_end"):
            effect.on_round_end(self, ctx)

    def clone(self) -> "JokerInstance":
        """Fast structured copy — avoids deepcopy overhead.

        State is a dict of primitives plus a few flat containers (Card Sharp
        ``played_hands`` set, Satellite ``planets_used``, ``pending_consumables``
        lists). Those are copied one level deep: a shallow ``state.copy()`` shared
        them between the clone and the original, so MCTS siblings cross-contaminated
        (W7 finding, 2026-08-21). Same pattern as
        ``card_selection._clone_joker_for_dry_run``.
        """
        new = JokerInstance.__new__(JokerInstance)
        new.key = self.key
        new.edition = self.edition
        # W5: the clone is the SAME card (MCTS twin), so it keeps the creation id the
        # sort_id-ordered draws (Amber Acorn, Crimson Heart, Invisible, Ankh) index.
        new.sort_id = getattr(self, "sort_id", 0)
        new.state = {
            k: (v.copy() if isinstance(v, (list, set, dict)) else v)
            for k, v in self.state.items()
        }
        return new

    def on_held_card(self, card, ctx: ScoreContext) -> bool:
        """Held-in-hand individual effect (Baron, Shoot the Moon, Raised Fist,
        Reserved Parking). Returns True if the joker produced an effect for this
        card — the real game only grants Red-seal / Mime repetitions to held
        cards that had SOME effect (state_events.lua:812-825)."""
        effect = JOKER_REGISTRY.get(self.key)
        if effect and hasattr(effect, "on_held_card"):
            return bool(effect.on_held_card(self, card, ctx))
        return False

    def __repr__(self):
        return f"Joker({self.key}, state={self.state}, ed={self.edition})"


# ════════════════════════════════════════════════════════════════════════════
# Game-facing helpers (W3). game.py / shop.py call these; they are the ONLY
# places that turn hook output (pending_*) into game state, so every call site
# drains the same way.
# ════════════════════════════════════════════════════════════════════════════

def n_owned(jokers, key: str) -> int:
    return sum(1 for j in jokers if j.key == key)


def sync_probabilities(game) -> float:
    """``G.GAME.probabilities.normal`` = 2 ** (owned Oops! All 6s). The real game
    doubles on ``add_to_deck`` and halves on ``remove_from_deck`` (card.lua:608,665);
    recomputing from ownership is equivalent and robust to every way a joker can
    enter the board (shop, pack, Judgement, Riff-raff, test injection)."""
    rs = game.run_state
    rs.probabilities_normal = float(2 ** n_owned(game.jokers, "j_oops"))
    return rs.probabilities_normal


def passive_modifiers(jokers) -> dict:
    """Passive effects the real game applies in add_to_deck / remove_from_deck and
    that live OUTSIDE scoring. Readers: game._start_blind (hand size), shop.py
    (Credit Card debt floor, Astronomer free planets, Chaos free reroll)."""
    keys = [j.key for j in jokers]
    hand_size = 0
    for j in jokers:
        if j.key == "j_juggler":      hand_size += 1
        elif j.key == "j_troubadour": hand_size += 2
        elif j.key == "j_merry_andy": hand_size -= 1
        elif j.key == "j_stuntman":   hand_size -= 2
        elif j.key == "j_turtle_bean": hand_size += j.state.get("h_size", 5)
    return {
        "hand_size": hand_size,
        "bankrupt_at": -20 * keys.count("j_credit_card"),   # G.GAME.bankrupt_at
        "free_rerolls": keys.count("j_chaos"),              # current_round.free_rerolls
        "free_planets": "j_astronomer" in keys,             # planet/celestial cost 0
        "showman": "j_ring_master" in keys,
    }


HAND_EVAL_FLAG_JOKERS = {"j_four_fingers": "four_fingers", "j_shortcut": "shortcut",
                         "j_smeared": "smeared", "j_pareidolia": "pareidolia"}


def hand_eval_flags(jokers) -> dict:
    """The ``find_joker`` flags ``evaluate_poker_hand`` reads (misc_functions.lua:524,
    550, 568; card.lua:4071): Four Fingers / Shortcut / Smeared Joker (+ Pareidolia for
    ``is_face``).  Pass ``**hand_eval_flags(game.jokers)`` to ``hand_eval.evaluate_hand``
    BEFORE scoring — the flags are a property of the board, not of the scoring pass.
    ``find_joker`` skips debuffed jokers, so pass the active board (Crimson Heart)."""
    flags = {"four_fingers": False, "shortcut": False, "smeared": False, "pareidolia": False}
    for j in jokers:
        name = HAND_EVAL_FLAG_JOKERS.get(j.key)
        if name:
            flags[name] = True
    return flags


def init_joker(inst: JokerInstance, ctx: ScoreContext) -> None:
    """``Card:set_ability`` for a joker entering the game: run the effect's
    ``on_init`` (To Do List draws its first hand here, Popcorn/Ramen/Ice Cream
    set their starting values). Idempotent — guarded by ``state['_init']``."""
    if inst.state.get("_init"):
        return
    inst.state["_init"] = True
    effect = JOKER_REGISTRY.get(inst.key)
    if effect and hasattr(effect, "on_init"):
        effect.on_init(inst, ctx)


def add_joker(game, inst: JokerInstance, ctx: Optional[ScoreContext] = None) -> bool:
    """Put a joker on the board: slot check, ``run_state.acquire``, ``on_init``,
    Oops! sync. Returns False if there is no room."""
    if len(game.jokers) >= game.joker_slots and inst.edition != "Negative":
        return False
    game.jokers.append(inst)
    if inst.edition == "Negative":          # Negative takes no slot (card_limit + 1) — W5
        game.joker_slots += 1
    rs = getattr(game, "run_state", None)
    if rs is not None:
        rs.acquire(inst.key)
    init_joker(inst, ctx or game._hook_ctx())
    sync_probabilities(game)
    return True


def remove_joker(game, inst: JokerInstance) -> None:
    """Take a joker off the board (destroyed / eaten / sold) with the
    ``run_state`` bookkeeping (``remove_owned`` -> ``used_jokers`` release)."""
    if inst in game.jokers:
        game.jokers.remove(inst)
    rs = getattr(game, "run_state", None)
    if rs is not None:
        rs.remove_owned(inst.key)
    sync_probabilities(game)


def drain_joker_state(game, ctx: Optional[ScoreContext] = None) -> None:
    """Apply everything hooks left pending, in the order the game resolves it:

    * per-joker ``state['pending_money']`` -> dollars;
    * per-joker ``state['pending_consumables']`` (real ``c_*`` keys) -> consumable slots;
    * ``ctx.pending_money`` / ``ctx.pending_consumables`` / ``ctx.pending_jokers`` /
      ``ctx.pending_cards`` / ``ctx.pending_destroy``;
    * per-joker ``state['destroyed']`` -> joker removed (Gros Michel sets the
      ``gros_michel_extinct`` pool flag so Cavendish enters the pool — card.lua:3020-3040).
    """
    rs = getattr(game, "run_state", None)
    for j in list(game.jokers):
        game.dollars += j.state.pop("pending_money", 0)
        for key in j.state.pop("pending_consumables", []):
            _grant_consumable(game, key)
    if ctx is not None:
        game.dollars += ctx.pending_money
        ctx.pending_money = 0
        for key in ctx.pending_consumables:
            _grant_consumable(game, key)
        ctx.pending_consumables = []
        for inst in ctx.pending_jokers:
            add_joker(game, inst, ctx)
        ctx.pending_jokers = []
        for card, where in ctx.pending_cards:
            game.add_card(card, to_draw_pile=(where != "hand"))
            if where == "hand":
                game.hand.append(card)
        ctx.pending_cards = []
        for card in ctx.pending_destroy:
            game.destroy_card(card)
        ctx.pending_destroy = []
    for j in list(game.jokers):
        if j.state.pop("destroyed", False):
            if j.key == "j_gros_michel" and rs is not None:
                rs.pool_flags.add("gros_michel_extinct")
            remove_joker(game, j)


def _grant_consumable(game, key: str) -> bool:
    """Emplace a created consumable (already drawn through ``run_state``). Delegates
    to ``game._materialize`` (W2) when present so every created-card token is
    resolved by one routine."""
    mat = getattr(game, "_materialize", None)
    if callable(mat):
        return bool(mat(key))
    if len(game.consumable_hand) >= game.consumable_slots:
        return False
    game.consumable_hand.append(key)
    rs = getattr(game, "run_state", None)
    if rs is not None:
        rs.acquire(key)
    return True


def has_consumable_room(ctx: ScoreContext) -> bool:
    """``#G.consumeables.cards + G.GAME.consumeable_buffer < card_limit`` — the gate
    every consumable-creating joker checks BEFORE rolling (card.lua:3106, 2337, ...)."""
    return len(ctx.consumables) + len(ctx.pending_consumables) < ctx.consumable_slots


def create_consumable(ctx: ScoreContext, spec_name: str) -> Optional[str]:
    """Create a consumable through the oracle-verified generation layer
    (``generate.CREATE_SPECS[spec_name]`` -> ``create_card``) and queue its real
    key. Returns None in a dry run (no ``run_state``)."""
    rs = ctx.run_state
    if rs is None:
        return None
    from ..game_keys import gen
    card = gen.create_from_spec(rs, spec_name)
    ctx.pending_consumables.append(card.key)
    return card.key


def fire_hook(game, hook_name: str, *args, ctx: Optional[ScoreContext] = None,
              jokers=None, drain: bool = True) -> ScoreContext:
    """Fire ``hook_name`` on every owned joker (left to right) with one shared
    game context, then drain pending state. This is the single call shape for
    the non-scoring hooks: ``fire_hook(game, 'on_reroll')``,
    ``fire_hook(game, 'on_booster_opened')``, ``fire_hook(game, 'on_shop_enter')``,
    ``fire_hook(game, 'on_shop_leave')``, ``fire_hook(game, 'on_card_sold')``,
    ``fire_hook(game, 'on_boss_ability_triggered')``, ``fire_hook(game, 'on_card_added')``."""
    ctx = ctx or game._hook_ctx()
    for j in list(jokers if jokers is not None else game.jokers):
        effect = JOKER_REGISTRY.get(j.key)
        if effect and hasattr(effect, hook_name):
            getattr(effect, hook_name)(j, *args, ctx)
    if drain:
        drain_joker_state(game, ctx)
    return ctx


def sell_hooks(game, sold: JokerInstance) -> None:
    """What ``Card:sell_card`` triggers for a joker that has just been removed from
    ``game.jokers``: the sold joker's own ``on_sell`` (Invisible Joker copy,
    Diet Cola tag, Luchador boss-disable), then ``selling_card`` on every
    remaining joker (Campfire), then the drain."""
    ctx = game._hook_ctx()
    effect = JOKER_REGISTRY.get(sold.key)
    if effect and hasattr(effect, "on_sell"):
        effect.on_sell(sold, ctx)
    # pending state the sold joker produced lives on the (popped) instance
    for key in sold.state.pop("pending_consumables", []):
        _grant_consumable(game, key)
    game.dollars += sold.state.pop("pending_money", 0)
    fire_hook(game, "on_card_sold", ctx=ctx)
