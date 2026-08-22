"""
game.py — Top-level Balatro game state machine.

States:
  BLIND_SELECT   -> agent chooses to play or skip a blind
  SELECTING_HAND -> agent plays or discards cards
  ROUND_EVAL     -> end-of-round payout (auto-advances)
  SHOP           -> agent buys, sells, uses consumables, rerolls, then leaves
  BOOSTER_OPEN   -> agent picks from opened booster pack
  GAME_OVER      -> terminal state
  PVP_WAIT       -> (ruleset="mlb" only) out of hands at the Nemesis blind; waits for the
                    opponent / MLBMatch (no actions) — see mlb_match.py, MLB_NOTES.md

Actions (passed as dict to game.step()):
  BLIND_SELECT:
    {"type": "play_blind"}
    {"type": "skip_blind"}

  SELECTING_HAND:
    {"type": "play",    "cards": [0, 2, 4]}
    {"type": "discard", "cards": [1, 3]}
    {"type": "use_consumable", "consumable_idx": 0, "target_cards": [0, 1]}

  SHOP:
    {"type": "buy",          "item_idx": 0}
    {"type": "sell_joker",   "joker_idx": 1}
    {"type": "use_consumable","consumable_idx": 0, "target_cards": [2]}
    {"type": "reroll"}
    {"type": "leave_shop"}

  BOOSTER_OPEN:
    {"type": "pick_booster", "indices": [0, 2]}  # which items to keep
    {"type": "skip_booster"}
"""
from __future__ import annotations
import secrets
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from .card import Card, make_standard_deck
from .hand_eval import evaluate_hand
from .scoring import score_hand
from .constants import (
    BLIND_CHIPS, blind_base_chips, STARTING_HANDS, STARTING_DISCARDS, HAND_SIZE,
    INTEREST_RATE, INTEREST_CAP, HAND_PAYOUT, STARTING_MONEY,
    BLIND_REWARD, SHOWDOWN_BLIND_REWARD, ENHANCEMENT_KEY,
    MLB_NEMESIS_KEY, MLB_NEMESIS_REWARD, MLB_STARTING_LIVES, MLB_PVP_START_ROUND,
    MLB_COMEBACK_PER_LIFE, MLB_BANNED_KEYS,    # Phase 2 W1 (MLB_NOTES.md)
)
from .jokers.base import (
    JokerInstance, JOKER_REGISTRY, ScoreContext,
    fire_hook, drain_joker_state, sync_probabilities, passive_modifiers, sort_id_order,   # W3
    hand_eval_flags,   # W5
)
from .round_cards import from_round_picks   # W3: current_round.{idol,mail,ancient,castle} for the hooks
from .consumables import (
    apply_planet, apply_tarot, apply_spectral,
    PLANET_HAND, ALL_TAROTS, ALL_PLANETS, ALL_SPECTRALS,
    TAROT_NAME, PLANET_NAME, SPECTRAL_NAME,
)
from .shop import (
    ShopItem, BoosterChoice, generate_shop, buy_item, sell_joker, reroll_shop,
    booster_contents, effective_price, can_afford, emplace_joker, BOOSTER_PICKS,
    EDITION_FROM_GEN, front_to_rank_suit, SHELF_KINDS,
)
from . import game_keys as _gk   # exposes gen (mp.rng.generate), core, normalize_seed, seed_from_int — Phase 1 seam
from . import tags as _tags      # W6 tag effects; wired here (W2), see DELEGATE_NOTES.md
from . import decks as _decks    # Phase 2 W3: deck catalogue + engine-side deck hooks (DECKS_NOTES.md)
from . import stakes as _stakes  # Phase 2 W3: stake catalogue + engine-side stake modifiers

_gen = _gk.gen
_Keys = _gen.Keys


class State(Enum):
    BLIND_SELECT   = auto()
    SELECTING_HAND = auto()
    ROUND_EVAL     = auto()
    SHOP           = auto()
    BOOSTER_OPEN   = auto()
    GAME_OVER      = auto()
    # MLB only (Phase 2 W1): hands exhausted at the Nemesis (PvP) blind — the player
    # waits for the opponent / the match's resolution ($MOD/ui/game/game_state.lua:190-213,
    # "k_wait_enemy").  No legal actions; the match moves the game on via end_pvp().
    PVP_WAIT       = auto()


@dataclass
class BlindInfo:
    name: str
    kind: str           # "Small" | "Big" | "Boss"
    chips_target: int
    is_boss: bool = False
    boss_key: str = ""
    is_showdown: bool = False   # finisher boss (ante 8/16/...) — pays $8, not $5
    disabled: bool = False      # Chicot / Luchador: boss effect neutralised (W3)
    is_pvp: bool = False        # MLB Nemesis blind (W1): chips_target = opponent's live score

    @property
    def money_reward(self) -> int:
        """Base $ awarded for defeating this blind."""
        if self.is_pvp:
            return MLB_NEMESIS_REWARD   # nemesis.lua: dollars = 5, never the showdown $8
        if self.is_showdown:
            return SHOWDOWN_BLIND_REWARD
        return BLIND_REWARD.get(self.kind, 0)


@dataclass
class GameState:
    """Full observable game state snapshot."""
    state: State
    ante: int
    blind_kind: str
    chips_target: int
    chips_scored: int
    hands_left: int
    discards_left: int
    dollars: int
    hand: list[Card]
    deck_remaining: int
    jokers: list[JokerInstance]
    consumable_hand: list[str]      # list of consumable keys held
    planet_levels: dict[str, int]
    shop_items: list[ShopItem]
    hand_type: str = ""
    done: bool = False
    won: bool = False
    info: dict = field(default_factory=dict)


# ════════════════════════════════════════════════════════════════════════════
# Boss blinds
# ════════════════════════════════════════════════════════════════════════════
# Effects verified against https://balatrowiki.org/w/Blinds on 2026-07-29. Several
# of the old comments here described the wrong mechanic entirely (bl_goad was
# documented and implemented as debuffing Clubs when the real Goad debuffs Spades;
# bl_wall as "+100 chips" rather than 4x base; bl_house, bl_ox, bl_final_leaf,
# bl_final_acorn and bl_final_bell were all wrong), and the showdown bosses were mixed
# into the regular pool.
#
# Bosses whose real effect is "drawn face down" (The House, The Wheel, The Mark,
# The Fish) are modelled since W5 through ``Card.face_down`` (``Blind:stay_flipped``,
# blind.lua:605-622): the card is hidden from the observation (env_v7 zeroes its
# features) and revealed when played / discarded / at round end.

# Keys are the game's `bl_*` keys from mp/rng/pools.py (the five showdown
# bosses were `bl_final_acorn/cerulean/crimson/verdant/violet` before the Phase 1
# re-key; the game calls them `bl_final_acorn/bell/heart/leaf/vessel`).
# `BOSS_MIN_ANTE` / `BOSS_MAX_ANTE` carry the game's `boss.min`/`boss.max`
# ante gates (The Ox min 6, Serpent 5, Plant 4, Eye/Tooth 3, ...) for W2's
# selection; the selection here is still a flat uniform draw (W2 delegates it
# to generate.next_boss).
#
# Effects (verified against https://balatrowiki.org/w/Blinds on 2026-07-29):
#   bl_hook     discards 2 random cards from hand after each played hand
#   bl_club     all Club cards debuffed          bl_goad    all Spade cards debuffed
#   bl_window   all Diamond cards debuffed       bl_head    all Heart cards debuffed
#   bl_plant    all face cards debuffed          bl_manacle -1 hand size
#   bl_eye      no repeat hand types this round  bl_mouth   only one hand type playable
#   bl_needle   play only 1 hand (1x base chips) bl_water   start with 0 discards
#   bl_tooth    lose $1 per card played          bl_wall    requires 4x base chips
#   bl_flint    base chips and mult halved       bl_psychic must play exactly 5 cards
#   bl_serpent  always draw 3 after play/discard bl_pillar  cards played this ante debuffed
#   bl_house    first hand drawn face down      bl_wheel   1 in 7 cards drawn face down ('wheel')
#   bl_mark     face cards drawn face down      bl_fish    cards drawn face down after each play
#   bl_ox       most-used hand type sets $ to 0  bl_arm     played hand level -1
#   bl_final_vessel  6x base chips               bl_final_leaf   all debuffed until a Joker is sold
#   bl_final_heart   random Joker disabled/hand  bl_final_acorn  flips + shuffles Jokers
#   bl_final_bell    one card always selected
from .game_keys import (
    BOSS_KEYS_REGULAR as _POOL_BOSS_KEYS_REGULAR,
    BOSS_KEYS_SHOWDOWN as _POOL_BOSS_KEYS_SHOWDOWN,
    BOSS_MIN_ANTE, BOSS_MAX_ANTE, BOSS_NAME,
)

# Kept (empty) for callers that partition the pool: since W2 the boss is drawn by
# `generate.next_boss` (the full game pool, parity-verified) and since W5 every regular
# boss has an engine effect — the four face-down bosses via ``Card.face_down``:
#   bl_house  first hand drawn face down      bl_wheel  1 in 7 cards drawn face down
#   bl_mark   face cards drawn face down      bl_fish   cards drawn face down after a play
UNMODELLED_BOSS_BLINDS: list = []

REGULAR_BOSS_BLINDS = [k for k in _POOL_BOSS_KEYS_REGULAR if k not in UNMODELLED_BOSS_BLINDS]   # 19
ALL_REGULAR_BOSS_BLINDS = list(_POOL_BOSS_KEYS_REGULAR)                                          # 23

# Finisher bosses — ante 8, 16, 24, ... They pay $8 instead of $5.
SHOWDOWN_BOSS_BLINDS = list(_POOL_BOSS_KEYS_SHOWDOWN)   # acorn, leaf, vessel, heart, bell (pool order)

# Chip target multiplier RELATIVE to a standard boss blind, whose target is
# already 2x the ante's small blind in BLIND_CHIPS.
BOSS_CHIP_MULT = {
    "bl_wall":         2.0,   # 4x base  = 2x a normal boss
    "bl_needle":       0.5,   # 1x base  = half a normal boss
    "bl_final_vessel": 3.0,   # 6x base  = 3x a normal boss
}

# Kept as an alias so older callers/tests that import BOSS_BLINDS keep working.
BOSS_BLINDS = REGULAR_BOSS_BLINDS


class BalatroGame:
    """
    Full stateful Balatro game engine.

    Usage:
        game = BalatroGame(seed=42)
        obs = game.reset()
        while not obs.done:
            action = agent.act(obs)
            obs = game.step(action)
    """

    def __init__(self, seed: "Optional[int | str]" = None, deck_key: str = "b_red",
                 stake: "int | str" = 1, ruleset: str = "vanilla"):
        # Phase 1 seam: one keyed generation state per run. `run_state.rng` is the
        # real-Balatro PseudoRandom; W2 delegates all generation through run_state
        # (rebuilt in _init_game_vars so reset() starts a fresh run), W3 routes all
        # effect rolls through run_state.rng, then `self.rng` is deleted.
        if seed is None:
            seed = secrets.randbelow(1 << 40)
        # Phase 2 W1: "vanilla" (default; byte-identical to the single-player engine) or
        # "mlb" (Major League Balatro = the Multiplayer mod's Attrition gamemode, MLB_NOTES.md):
        # Attrition bans, Nemesis blind at the Boss slot from ante `pvp_start_round`, lives,
        # failed blinds proceed + cost a life, comeback money, endless (no ante-8 win).
        if ruleset not in ("vanilla", "mlb"):
            raise ValueError(f"ruleset must be 'vanilla' or 'mlb', got {ruleset!r}")
        self.ruleset = ruleset
        self.seed_str = _gk.normalize_seed(seed) if isinstance(seed, str) else _gk.seed_from_int(seed)
        self.deck_key = _decks.deck_spec(deck_key).key   # any of the 15 b_* keys (decks.DECK_STATUS)
        # G.GAME.stake, 1..8 (or 'stake_white'..'stake_gold'); White = the vanilla run.
        self.stake = _stakes.stake_spec(stake).level
        self.run_state = _gk.gen.RunState.for_stake(self.seed_str, stake=self.stake)
        # W3: there is no `self.rng` any more — every roll is `run_state.rng.pseudorandom(<key>)`.
        self._init_game_vars()

    # ── Initialization ───────────────────────────────────────────────────────

    def _init_game_vars(self):
        """Set all mutable game state to starting values and perform the run-start draws
        (GENERATION_SPEC §16.2) through ``generate.start_run``."""
        # One keyed generation state per run (generate.RunState + the bit-exact PseudoRandom).
        # `for_stake` sets the generation-side stake flags (enable_*_in_shop); stake 1 == RunState(seed).
        self.run_state = _gen.RunState.for_stake(self.seed_str, stake=self.stake)
        from . import shop as _shop   # read the module attribute: set_banned_jokers() rebinds it
        self.run_state.banned_keys = set(_shop.BANNED_JOKERS)     # MP ruleset bans -> G.GAME.banned_keys
        # ── Major League Balatro (Phase 2 W1, MLB_NOTES.md) ──────────────────
        # Attrition bans go into G.GAME.banned_keys exactly as MP.ApplyBans does
        # ($MOD/rulesets/_rulesets.lua:198-229, run from Game:start_run BEFORE the first
        # 'boss'/'Voucher1'/'Tag1' draws — game.lua:2170 vs 2177), so the Phase-1 in-place
        # resample invariant (UNAVAILABLE slot + side stream) holds for jokers, vouchers
        # and tags, and the two banned bosses leave `get_new_boss`'s eligible set.
        self.mlb = (getattr(self, "ruleset", "vanilla") == "mlb")
        if self.mlb:
            self.run_state.banned_keys |= set(MLB_BANNED_KEYS)
        # Generation-layer switches (W2, generate.RunState doc / NOTES_ORDER.md), set BEFORE
        # the run-start draws below: `ruleset = "mlb"` routes the shop voucher AND the
        # Voucher Tag through the mod's culled-pool 'Voucher0' run-global stream (brief
        # §1.6, TheOrder.lua:481-525 -- on under MLB even with The Order off); `key_scope`
        # is The Order switch (W2 owns its wiring; "ante" is a no-op).
        self.run_state.ruleset = self.ruleset
        # W2 (2026-08-21): RunState.key_scope exists; "run" re-seeds the RNG with '*'..seed
        # (the mod prefixes the seed) and must therefore be set before any draw -- here.
        # The Order additionally needs RunState.blind_key / blind_type maintained at blind
        # start and cash-out (NOTES_ORDER.md §5); not wired (MLB forces The Order off).
        self.run_state.key_scope = getattr(self, "queue_scope", "ante")
        # MP.GAME client state ($MOD/core.lua:205-260) + the server's per-player state
        # (BalatroMultiplayerAPI-Server src/Client.ts): lives, comeback counter, the
        # once-per-round life-loss blocker, PvP bookkeeping.  All vanilla-inert.
        self.pvp_start_round = MLB_PVP_START_ROUND
        self.lives = MLB_STARTING_LIVES if self.mlb else 0
        self.comeback_bonus = 0               # MP.GAME.comeback_bonus: cumulative lives lost
        self.comeback_bonus_given = True      # MP.GAME.comeback_bonus_given (starts true)
        self.life_lost_this_round = False     # Client.roundLivesBlocker (reset on newRound)
        self.pvp_ready = False                # readyBlind sent, waiting for the opponent
        self.pvp_solo = True                  # no MLBMatch attached: play_blind starts the Nemesis at once
        self.pvp_started = False              # a Nemesis blind is in progress (startBlind received)
        self.pvp_opponent_score = 0           # MP.GAME.enemy.score  (live, from enemyInfo)
        self.pvp_opponent_hands = 0           # MP.GAME.enemy.hands
        self.match_won = False                # winGame received (the opponent hit 0 lives)
        self.ante = 1
        self.blind_idx = 0                  # 0=Small, 1=Big, 2=Boss
        self.dollars = STARTING_MONEY
        self.jokers: list[JokerInstance] = []
        self.joker_slots = 5
        self.consumable_hand: list[str] = []  # held consumable keys
        self.consumable_slots = 2
        self.planet_levels: dict[str, int] = {h: 1 for h in [
            "High Card", "Pair", "Two Pair", "Three of a Kind",
            "Straight", "Flush", "Full House", "Four of a Kind",
            "Straight Flush", "Five of a Kind", "Flush House", "Flush Five",
        ]}
        self.vouchers: set[str] = set()
        self.interest_cap = INTEREST_CAP    # raised by v_seed_money / v_money_tree
        self.planets_used: list[str] = []
        self.tarots_used: list[str] = []

        # Hand / discard / hand-size settings
        self.base_hands = STARTING_HANDS
        self.base_discards = STARTING_DISCARDS
        self.hand_size = HAND_SIZE
        # Permanent `G.hand:change_size` deltas (Ouija / Ectoplasm -1 each; stack for the
        # run) — applied on top of HAND_SIZE at every blind start (W5).
        self.hand_size_mod = 0
        # Negative consumables take no slot: `consumable_slots` is bumped on acquire and
        # must drop again when THAT card is used (Card:remove_from_deck).  Consumables are
        # bare keys, so the negative copies are tracked as a key multiset (W5).
        self.negative_consumables: dict = {}

        # Shop settings. The shelf has `run_state.shop_joker_max` slots (2, +1 per
        # Overstock tier) shared by jokers / consumables / playing cards — the real
        # shop has no separate "card row" (GENERATION_SPEC §8.1).
        self.shop_discount = 0.0
        self.reroll_cost = 5
        self.reroll_discount = 0
        self.free_rerolls_per_round = 0
        self.free_rerolls_remaining = 0
        self.current_shop: list[ShopItem] = []
        self._shop_gen = None               # generate.ShopContents mirror of current_shop (for release/reroll)

        # Booster state (State.BOOSTER_OPEN)
        self.booster_choices: list = []      # BoosterChoice entries (.key/.edition/.enhancement/.seal/.front)
        self.booster_picks_remaining: int = 0
        self.booster_pack_key: Optional[str] = None
        self._booster_return_state: Optional[State] = None   # SHOP, or BLIND_SELECT for tag packs
        self._booster_free: bool = False

        # Tags (W6 tags.py) + the G.GAME counters the tag effects read
        self.tag_state = _tags.TagState()
        self.skips = 0
        self.unused_discards = 0
        self._orbital_choices: dict = {}    # (ante, blind_type) -> hand ('orbital' draws at blind select)
        self._boss_rerolled = False         # round_resets.boss_rerolled (Director's Cut once per ante)
        # MLB "The Order" switch: "ante" = vanilla/MLB ante-suffixed keys ('Joker1sho<ante>');
        # "run" = the same keys without the ante suffix (one queue for the whole run).
        # NO-OP until generate.py exposes the key-suffix hook -- see DELEGATE_NOTES.md §The Order.
        self.queue_scope = "ante"

        # Deck / stake modifiers (G.GAME.modifiers + starting_params, Phase 2 W3) — the
        # vanilla defaults; overwritten by stakes.apply_stake_to_game / decks.apply_deck_to_game
        # after the run-start draws below (Game:start_run order: stake, then Back:apply_to_run).
        self.stake_key = "stake_white"
        self.no_small_blind_reward = False  # modifiers.no_blind_reward.Small (stake >= 2)
        self.blind_scaling = 1              # modifiers.scaling: get_blind_amount table (stake >= 3 / >= 6)
        self.ante_scaling = 1               # starting_params.ante_scaling (Plasma 2) — blind targets
        self.no_interest = False            # modifiers.no_interest (Green Deck)
        self.money_per_hand = 1             # modifiers.money_per_hand (Green Deck 2)
        self.money_per_discard = 0          # modifiers.money_per_discard (Green Deck 1)
        self.plasma = False                 # Plasma Deck: balance at final_scoring_step
        self.anaglyph = False               # Anaglyph Deck: Double Tag after each boss

        # Blind state
        self.current_blind: BlindInfo = BlindInfo("", "Small", 0)
        self.chips_scored = 0
        self.hands_left = self.base_hands
        self.discards_left = self.base_discards

        # ── Deck model ────────────────────────────────────────────────────────
        # full_deck is the player's PERMANENT collection and the single source of
        # truth for deck composition. Enhancements, seals and editions applied by
        # tarots/spectrals live on these Card objects and persist for the run.
        #
        # deck / hand / discard_pile are per-blind partitions holding references
        # to those same objects, so mutating a card in hand mutates the permanent
        # deck automatically. Never rebuild full_deck mid-run, and never add or
        # remove cards except through add_card() / remove_card().
        #
        # Before 2026-07-29 there was no full_deck: _init_deck() built a fresh
        # vanilla 52-card deck every blind, silently discarding every permanent
        # card modification the player had bought.
        #
        # Phase 1 W2: the cards are created in the game's creation order (sorted
        # `suit..rank` string: C2..C9,CA,CJ,CK,CQ,CT,D2,... — game.lua:2326-2378), so
        # `Card.id` order == the game's `sort_id` order, which every deck shuffle
        # (`pseudoshuffle`) sorts by first.
        self.played_hand_types_this_round: set[str] = set()
        self.last_played_hand_type: str = ""   # for Blue Seal at end of round

        # Boss-blind bookkeeping
        self._played_this_ante: set[int] = set()   # card ids, for The Pillar
        self._hand_type_counts: dict[str, int] = {}  # run totals, for The Ox
        self._verdant_active: bool = False         # cleared when a Joker is sold
        self._disabled_joker_idx: int = -1         # The Crimson Heart, per hand
        self._forced_card_id: int = -1             # The Cerulean Bell
        self._hands_played_round = 0               # current_round.hands_played (reset per blind)
        self._discards_used_round = 0              # current_round.discards_used (reset per blind)

        # ── Run start (Game:start_run, GENERATION_SPEC §16.2) ─────────────────
        # deck effects -> 'boss' -> 'Voucher1' -> 'Tag1' x2 -> deck creation ->
        # 'shuffle' -> 'idol1'/'mail1'/'anc1'/'cas1'; then the blind-select screen
        # ('orbital' x3, tag passes) in _enter_blind_select().
        rs = self.run_state
        start = _gen.start_run(rs, self.deck_key)
        by_key: dict[str, list] = {}
        self.full_deck: list[Card] = []
        # creation order (sort_id): the pre-Checkered-swap `suit..rank` sort (decks.creation_order)
        for front in _decks.creation_order(self.deck_key, start.deck):
            rank, suit = front_to_rank_suit(front)
            c = Card(rank=rank, suit=suit)
            self.full_deck.append(c)
            by_key.setdefault(front, []).append(c)
        # draw pile in the 'shuffle' order (top of deck = last); redone with 'nr<ante>' at blind start
        self.deck: list[Card] = [by_key[k].pop(0) for k in start.deck]
        self.hand: list[Card] = []
        self.discard_pile: list[Card] = []  # played/discarded this blind
        # deck-granted consumables / vouchers (Magic, Nebula, Ghost, Zodiac decks)
        self.consumable_hand = list(rs.owned_consumables)
        self.vouchers |= set(rs.used_vouchers)
        self.boss_blind: Optional[str] = start.boss          # this ante's boss, known at ante start
        self._boss_blind_ante = 1
        self.blind_tags: dict = dict(rs.blind_tags)           # {'Small': tag_key, 'Big': tag_key}
        # G.GAME.current_round.{idol_card, mail_card.rank, ancient_card.suit, castle_card.suit}
        # in generation-layer keys: idol/castle 'S_A', mail rank char 'A', ancient suit name
        self.round_picks = {"idol": start.idol, "mail": (start.mail[2:] if start.mail else None),
                            "ancient": start.ancient_suit, "castle": start.castle}
        # Engine side of the stake modifiers (game.lua:2050-2057) then Back:apply_to_run
        # (back.lua:173-288): hands / discards / $ / slots / hand size / ante_scaling /
        # money modifiers.  No RNG (the generation side ran inside start_run).  W3.
        _stakes.apply_stake_to_game(self)
        _decks.apply_deck_to_game(self)

        self.state = State.BLIND_SELECT
        self._prepare_next_blind()
        self._enter_blind_select()

    def reset(self) -> GameState:
        self._init_game_vars()
        return self._obs()

    def _hook_ctx(self) -> ScoreContext:
        """
        Minimal ScoreContext for non-scoring joker hooks (on_round_end,
        on_discard, on_boss_beaten, ...).

        These used to be called with `None`, which meant any stochastic joker
        firing outside the scoring loop silently fell back to the global `random`
        module and escaped seed control. Always pass this instead of None.
        """
        return ScoreContext(
            hands_left=self.hands_left,
            discards_left=self.discards_left,
            dollars=self.dollars,
            ante=self.ante,
            deck_remaining=len(self.deck),
            planet_levels=self.planet_levels,
            jokers=self.jokers,
            held_cards=list(self.hand),
            full_deck=self.full_deck,
            hand_type_counts=self._hand_type_counts,
            # W3: keyed RNG + run-level state every effect hook needs
            prng=self.run_state.rng,
            run_state=self.run_state,
            probabilities_normal=sync_probabilities(self),
            round_cards=from_round_picks(getattr(self, "round_picks", None)),
            joker_slots=self.joker_slots,
            consumable_slots=self.consumable_slots,
            consumables=self.consumable_hand,
            blind_kind=self.current_blind.kind,
            hands_played=getattr(self, "_hands_played_round", 0),
        )

    # Alias kept for MCTS-side callers that used the earlier name.
    _bare_ctx = _hook_ctx

    # ── Cloning (for MCTS) ───────────────────────────────────────────────────

    def clone(self) -> "BalatroGame":
        """
        Fast structured copy of the full game state for MCTS tree expansion.

        Skips __init__ via __new__ and copies each attribute by hand:
          - Primitives: assignment
          - Jokers: list comp using JokerInstance.clone()
          - Shop items: dataclass.replace (all-primitive fields)
          - RNG: new Random with setstate to preserve stream position
          - Sets/dicts: shallow .copy()

        IMPORTANT — deck identity. full_deck is the permanent collection and
        deck/hand/discard_pile hold REFERENCES to those same Card objects, which
        is what lets a tarot applied to a card in hand persist for the run. A
        naive `[c.copy() for c in self.deck]` per collection would clone each card
        several times over and sever that relationship, so enhancements applied
        after cloning would silently fail to persist. full_deck is therefore
        copied once and the partitions are rebuilt as references into it, keyed by
        card id.
        """
        from dataclasses import replace as _dc_replace
        from .card import Card as _Card

        new = BalatroGame.__new__(BalatroGame)

        # RNG: the keyed PseudoRandom lives in run_state (cloned below); W3 removed game.rng.
        new._hands_played_round = getattr(self, "_hands_played_round", 0)
        new._discards_used_round = getattr(self, "_discards_used_round", 0)
        new.seed_str = self.seed_str
        new.deck_key = self.deck_key
        new.queue_scope = self.queue_scope
        new.run_state = _fast_clone_run_state(self.run_state)
        # MLB (Phase 2 W1): ruleset + the MP.GAME / server-side per-player scalars
        new.ruleset = self.ruleset
        new.mlb = self.mlb
        new.pvp_start_round = self.pvp_start_round
        new.lives = self.lives
        new.comeback_bonus = self.comeback_bonus
        new.comeback_bonus_given = self.comeback_bonus_given
        new.life_lost_this_round = self.life_lost_this_round
        new.pvp_ready = self.pvp_ready
        new.pvp_solo = self.pvp_solo
        new.pvp_started = self.pvp_started
        new.pvp_opponent_score = self.pvp_opponent_score
        new.pvp_opponent_hands = self.pvp_opponent_hands
        new.match_won = self.match_won
        # Deck / stake modifiers (W3) — all immutable per run
        new.stake = self.stake
        new.stake_key = self.stake_key
        new.no_small_blind_reward = self.no_small_blind_reward
        new.blind_scaling = self.blind_scaling
        new.ante_scaling = self.ante_scaling
        new.no_interest = self.no_interest
        new.money_per_hand = self.money_per_hand
        new.money_per_discard = self.money_per_discard
        new.plasma = self.plasma
        new.anaglyph = self.anaglyph

        # W2 fields: generation / tags / booster state machine
        new.tag_state = _fast_clone_tag_state(self.tag_state)
        new.skips = self.skips
        new.unused_discards = self.unused_discards
        new._orbital_choices = dict(self._orbital_choices)
        new._boss_rerolled = self._boss_rerolled
        new.boss_blind = self.boss_blind
        new._boss_blind_ante = self._boss_blind_ante
        new.blind_tags = dict(self.blind_tags)
        new.round_picks = dict(self.round_picks)
        new.booster_pack_key = self.booster_pack_key
        new._booster_return_state = self._booster_return_state
        new._booster_free = self._booster_free
        if self._shop_gen is None:
            new._shop_gen = None
        else:   # CardGens are never mutated after creation -> shallow container copies suffice
            sg = self._shop_gen
            new._shop_gen = _gen.ShopContents(cards=list(sg.cards), voucher=sg.voucher,
                                              tag_vouchers=list(sg.tag_vouchers),
                                              boosters=list(sg.boosters), rerolls=sg.rerolls)

        # State enum + scalars
        new.state = self.state
        new.ante = self.ante
        new.blind_idx = self.blind_idx
        new.dollars = self.dollars
        new.joker_slots = self.joker_slots
        new.interest_cap = self.interest_cap
        new.consumable_slots = self.consumable_slots
        new.base_hands = self.base_hands
        new.base_discards = self.base_discards
        new.hand_size = self.hand_size
        new.hand_size_mod = getattr(self, "hand_size_mod", 0)
        new.negative_consumables = dict(getattr(self, "negative_consumables", {}))
        new.shop_discount = self.shop_discount
        new.reroll_cost = self.reroll_cost
        new.reroll_discount = self.reroll_discount
        new.free_rerolls_per_round = self.free_rerolls_per_round
        new.free_rerolls_remaining = self.free_rerolls_remaining
        new.booster_picks_remaining = self.booster_picks_remaining
        new.chips_scored = self.chips_scored
        new.hands_left = self.hands_left
        new.discards_left = self.discards_left

        # Card collections — copy the permanent deck once, then alias into it.
        new.full_deck = [c.copy() for c in self.full_deck]
        by_id = {orig.id: copy
                 for orig, copy in zip(self.full_deck, new.full_deck)}
        # Card.copy() mints a fresh id, so map through the ORIGINAL ids.
        def _alias(cards):
            out = []
            for c in cards:
                mapped = by_id.get(c.id)
                out.append(mapped if mapped is not None else c.copy())
            return out
        new.deck = _alias(self.deck)
        new.hand = _alias(self.hand)
        new.discard_pile = _alias(self.discard_pile)

        # Jokers
        new.jokers = [j.clone() for j in self.jokers]

        # Consumables / planets / vouchers / used-lists
        new.consumable_hand = list(self.consumable_hand)
        new.planet_levels = self.planet_levels.copy()
        new.vouchers = self.vouchers.copy()
        new.planets_used = list(self.planets_used)
        new.tarots_used = list(self.tarots_used)

        # Shop items — all-primitive dataclass
        new.current_shop = [_dc_replace(item) for item in self.current_shop]

        # Booster choices: BoosterChoice (W2) — legacy str / ("card", Card) tolerated
        new_choices = []
        for x in self.booster_choices:
            if isinstance(x, BoosterChoice):
                new_choices.append(x.clone())
            elif isinstance(x, tuple) and len(x) == 2 and isinstance(x[1], _Card):
                new_choices.append((x[0], x[1].copy()))
            else:
                new_choices.append(x)
        new.booster_choices = new_choices

        # Blind info + per-round / per-ante bookkeeping
        new.current_blind = _dc_replace(self.current_blind)
        new.played_hand_types_this_round = self.played_hand_types_this_round.copy()
        new.last_played_hand_type = self.last_played_hand_type
        # Card ids are minted fresh by Card.copy(): remap the Pillar's played-card ids and
        # the Cerulean Bell's forced card onto the clone's cards (W1 fix -- they used to
        # point at the ORIGINAL ids, so a clone silently lost both effects).
        new._played_this_ante = {by_id[i].id for i in self._played_this_ante if i in by_id}
        new._hand_type_counts = self._hand_type_counts.copy()
        new._verdant_active = self._verdant_active
        new._disabled_joker_idx = self._disabled_joker_idx
        new._forced_card_id = (by_id[self._forced_card_id].id
                               if self._forced_card_id in by_id else self._forced_card_id)

        return new

    # ── Blind setup ──────────────────────────────────────────────────────────

    def _prepare_next_blind(self):
        """Set up current_blind without starting play yet.

        The boss is NOT drawn here: the game draws it at run start and at Cash Out after
        a boss (``get_new_boss``, 'boss' stream, alphabetical eligible list with the
        min-usage "exhaustion" filter — GENERATION_SPEC §11) into ``self.boss_blind``.
        If a caller pokes ``self.ante`` and asks for a boss the run never drew (tests),
        one is drawn now for that ante through the same generate call.
        """
        kind = ["Small", "Big", "Boss"][self.blind_idx]
        # get_blind_amount(ante) * blind mult (1 / 1.5 / 2): ante < 1 -> 100 base (Hieroglyph /
        # Petroglyph at ante 1 really take the run to ante 0), ante > 8 -> the game's formula.
        # blind.lua:107: get_blind_amount(ante) [stake scaling table] * mult * ante_scaling
        # (Plasma Deck 2) — the deck factor is applied here so every boss multiplier below
        # composes with it (all factors commute).  W3.
        chips = int(blind_base_chips(self.ante, self.blind_idx, self.blind_scaling) * self.ante_scaling)
        boss_key = ""
        is_showdown = False
        if kind == "Boss":
            if self.boss_blind is None or self._boss_blind_ante != self.ante:
                self.run_state.ante = self.ante
                self.boss_blind = _gen.next_boss(self.run_state)
                self.run_state.boss_blind = self.boss_blind
                self._boss_blind_ante = self.ante
            boss_key = self.boss_blind
            # Showdown (finisher) bosses pay $8; the pool membership is the game's
            # (ante % win_ante == 0 draws only from the five showdown blinds).
            is_showdown = boss_key in SHOWDOWN_BOSS_BLINDS
            chips = int(chips * BOSS_CHIP_MULT.get(boss_key, 1.0))
        is_pvp = False
        if self.mlb and kind == "Boss" and self.ante >= self.pvp_start_round:
            # MLB Nemesis ($MOD/gamemodes/attrition.lua:3-11 via ui/game/round.lua:54-66):
            # the mod's reset_blinds runs the VANILLA reset_blinds first (so the 'boss'
            # stream is drawn exactly as in single player -- `self.boss_blind` keeps that
            # "shadow" draw and bosses_used keeps counting) and then overwrites the Boss
            # choice with bl_mp_nemesis: no boss effect, mult 1, $5, and the target is the
            # opponent's live score (enemyInfo; 0 until the opponent scores).
            is_pvp = True
            boss_key = MLB_NEMESIS_KEY
            is_showdown = False
            # enemy.score / enemy.hands are reset at startBlind (action_handlers.lua:332-338);
            # the target is 0 until the opponent scores (set_pvp_info from the match).
            self.pvp_opponent_score = 0
            self.pvp_opponent_hands = 0
            chips = 0
        self.current_blind = BlindInfo(
            name=f"Ante {self.ante} {'Nemesis' if is_pvp else kind}",
            kind=kind,
            chips_target=chips,
            is_boss=(kind == "Boss"),
            boss_key=boss_key,
            is_showdown=is_showdown,
            is_pvp=is_pvp,
        )
        self.state = State.BLIND_SELECT

    # ── Generation-state sync / tag context (W2) ─────────────────────────────

    def _playing_cards_sorted(self) -> list:
        """``G.playing_cards`` in ``sort_id`` (creation) order — what every pseudoshuffle /
        pseudorandom_element over the deck indexes."""
        return sorted(self.full_deck, key=lambda c: c.id)

    def _sync_run_state(self):
        """Rebuild the ``RunState`` ownership view from the game's own collections.

        ``used_jokers`` = centers of every Card that currently exists anywhere (owned
        jokers + consumables + unsold shelf cards + cards of an open pack) — exactly
        ``Card:set_ability`` / ``Card:remove`` semantics (GENERATION_SPEC §6).  Called
        before every generation call so hand-edited game state (tests, envs) can never
        desync the incremental acquire/release bookkeeping.
        """
        rs = self.run_state
        rs.ante = self.ante
        rs.owned_jokers = [j.key for j in self.jokers]
        rs.owned_consumables = [k for k in self.consumable_hand if isinstance(k, str)]
        rs.showman = any(j.key == "j_ring_master" for j in self.jokers)
        shelf = set()
        vouchers_shown = []
        for it in self.current_shop:
            if it.sold:
                continue
            if it.kind in ("joker", "planet", "tarot", "spectral"):
                shelf.add(it.key)
            elif it.kind == "voucher":
                vouchers_shown.append(it.key)
        pack = {c.key for c in self.booster_choices
                if isinstance(c, BoosterChoice) and not c.is_playing_card}
        rs.used_jokers = set(rs.owned_jokers) | set(rs.owned_consumables) | shelf | pack
        rs.shop_voucher_keys = vouchers_shown
        rs.used_vouchers |= set(self.vouchers)
        rs.deck_enhancements = {ENHANCEMENT_KEY[c.enhancement] for c in self.full_deck
                                if c.enhancement in ENHANCEMENT_KEY and ENHANCEMENT_KEY[c.enhancement]}
        rs.hands_played = {h: self._hand_type_counts.get(h, 0) for h in rs.hands_played}
        for h, n in self._hand_type_counts.items():
            rs.hands_played[h] = n

    def _absorb_tag_triggers(self):
        """Option B of TAGS_NOTES §2: ``generate.generate_shop``/``reroll_shop`` applied the
        store_joker_create / store_joker_modify / voucher_add tags themselves against
        ``run_state.tags`` (indices into ``tag_state.keys()``); mark + purge them here."""
        rs = self.run_state
        for i in sorted(rs.triggered_tags):
            if i < len(self.tag_state.tags):
                self.tag_state.tags[i].triggered = True
        self.tag_state._purge()
        rs.tags = self.tag_state.keys()
        rs.triggered_tags = set()

    @property
    def tags(self) -> "_tags.TagState":
        """``G.GAME.tags`` (TAGS_NOTES: ``game.tags = TagState()``); ``.keys()`` = owned tag keys."""
        return self.tag_state

    def _tag_ctx(self, last_blind_was_boss: Optional[bool] = None) -> "_GameTagContext":
        return _GameTagContext(self, last_blind_was_boss)

    def _visible_hands(self) -> set:
        """``G.GAME.hands[h].visible``: the 9 base hands plus any hidden hand once played."""
        vis = {h["name"] for h in _gk.pools.POKER_HANDS if h["visible"]}
        vis |= {h for h, n in self._hand_type_counts.items() if n > 0}
        return vis

    def _enter_blind_select(self):
        """The blind-select screen is shown (game.lua:3290-3295 / UI_definitions.lua:1506-1516):
        'orbital' draws for Small/Big/Boss once per ante, then the tags' ``immediate`` and
        ``new_blind_choice`` passes (a pack tag opens a free pack as an interrupt)."""
        self.state = State.BLIND_SELECT
        for bt in ("Small", "Big", "Boss"):
            if (self.ante, bt) not in self._orbital_choices:
                self._orbital_choices[(self.ante, bt)] = _gen.orbital_hand(self.run_state, self._visible_hands())
        out = self.tag_state.on_blind_select(self._tag_ctx())
        self._handle_blind_choice(out)

    def _handle_blind_choice(self, out):
        """Open a ``new_blind_choice`` tag's pack as an interrupt; when it closes the pass is
        re-run (``_close_booster``) until no pack tag is left (button_callbacks.lua:2617-2619)."""
        if out is not None and out.pending_pack:
            self._open_booster(out.pending_pack, free=True, return_state=State.BLIND_SELECT)

    def _round_end_resets(self):
        """``reset_idol_card`` / ``reset_mail_rank`` / ``reset_ancient_card`` / ``reset_castle_card``
        (common_events.lua:2271-2324): run at run start and after every round end, with the
        current (post-ease_ante) ante.  Stored in ``round_picks`` for The Idol / Mail-In
        Rebate / Ancient Joker / Castle."""
        rs = self.run_state
        rs.ante = self.ante
        cards = [c for c in self._playing_cards_sorted() if c.enhancement != "Stone"]
        ante = self.ante
        if cards:
            idol, _ = rs.rng.pseudorandom_element(cards, _Keys.idol(ante))
            mail, _ = rs.rng.pseudorandom_element(cards, _Keys.mail(ante))
        else:
            idol = mail = None
        suits = [s for s in ("Spades", "Hearts", "Clubs", "Diamonds") if s != self.round_picks.get("ancient")]
        ancient, _ = rs.rng.pseudorandom_element(suits, _Keys.ancient(ante))
        castle, _ = rs.rng.pseudorandom_element(cards, _Keys.castle(ante)) if cards else (None, None)
        self.round_picks = {
            "idol": (f"{idol.suit[0]}_{_front_rank(idol.rank)}" if idol is not None else None),
            "mail": (_front_rank(mail.rank) if mail is not None else None),
            "ancient": ancient,
            "castle": (f"{castle.suit[0]}_{_front_rank(castle.rank)}" if castle is not None else None),
        }

    # ── Created cards (consumables / jokers / tags -> generate.create_card) ──

    def _free_consumable_slots(self) -> int:
        return self.consumable_slots - len(self.consumable_hand)

    def grant_created(self, spec: str, to_hand: bool = True):
        """Create a card exactly as the named game site does (``generate.CREATE_SPECS``:
        'judgement', 'soul', 'wraith', 'emperor', 'high_priestess', '8_ball', 'purple_seal',
        'hallucination', 'riff_raff', 'cartomancer', 'sixth_sense', 'vagabond',
        'superposition', 'seance', 'top_up_tag') and add it to the owned area.
        Returns the created key (or None when there is no room)."""
        self._sync_run_state()
        kw = _gen.CREATE_SPECS[spec]
        if kw["_type"] == "Joker":
            if len(self.jokers) >= self.joker_slots:
                return None
            c = _gen.create_card(self.run_state, **kw)
            j = JokerInstance(c.key, EDITION_FROM_GEN.get(c.edition, "None"))
            emplace_joker(self, j)
            return c.key
        if self._free_consumable_slots() <= 0:
            return None
        c = _gen.create_card(self.run_state, **kw)
        self.consumable_hand.append(c.key)
        self.run_state.acquire(c.key)
        return c.key

    def add_negative_consumable(self, key: str) -> None:
        """Emplace a Negative consumable (Perkeo's copy, a Negative pack/shelf card):
        no slot needed — ``card_limit`` grows by one while the card exists and shrinks
        again when it is used (``Card:remove_from_deck``), tracked per key."""
        self.consumable_hand.append(key)
        self.consumable_slots += 1
        self.negative_consumables[key] = self.negative_consumables.get(key, 0) + 1
        self.run_state.acquire(key)

    def _consumable_removed(self, key: str) -> None:
        """A consumable left the owned area (used): drop a Negative copy's slot if one of
        that key was Negative (the game tracks it per card; keys are the best we have)."""
        n = self.negative_consumables.get(key, 0)
        if n > 0:
            self.negative_consumables[key] = n - 1
            if n - 1 == 0:
                del self.negative_consumables[key]
            self.consumable_slots = max(1, self.consumable_slots - 1)

    def _materialize(self, item) -> bool:
        """Resolve one queued "created card" token from a joker/tag hook into real state.

        Accepted: a game consumable key (``c_*``) -> consumable slot; ``"create:<spec>"`` ->
        ``grant_created(spec)``; ``"double_tag"`` (Diet Cola) -> Double Tag;
        ``"common_joker"`` (Riff-raff) -> 'Joker1rif<ante>'; ``"stone_card"`` (Marble) ->
        'marb_fr'; ``"random_enhanced_card"`` (Certificate) -> 'cert_fr'+'certsl';
        ``"copy_card:<rank>:<suit>"`` (DNA).  Anything else is pushed through unchanged
        (legacy sentinel -- W5 worklist).  Returns True when handled."""
        rs = self.run_state
        if isinstance(item, str) and item.startswith("create:"):
            self.grant_created(item[len("create:"):])
            return True
        if isinstance(item, str) and item.startswith("negative:"):     # Perkeo (W5)
            self.add_negative_consumable(item[len("negative:"):])
            return True
        if item == "double_tag":
            self.tag_state.acquire("tag_double", self._tag_ctx())
            return True
        if item == "common_joker":
            self.grant_created("riff_raff")
            return True
        if item == "stone_card":
            self._sync_run_state()
            front = _gen.marble_joker(rs)
            rank, suit = front_to_rank_suit(front)
            c = Card(rank=rank, suit=suit)
            c.enhancement = "Stone"
            self.add_card(c)
            return True
        if item == "random_enhanced_card":
            self._sync_run_state()
            front, seal = _gen.certificate(rs)
            rank, suit = front_to_rank_suit(front)
            c = Card(rank=rank, suit=suit)
            c.seal = seal or "None"
            self.add_card(c)
            return True
        if isinstance(item, str) and item.startswith("copy_card:"):
            _, rank, suit = item.split(":")
            self.add_card(Card(rank=int(rank), suit=suit))
            return True
        if isinstance(item, str) and (item in TAROT_NAME or item in PLANET_NAME or item in SPECTRAL_NAME):
            if self._free_consumable_slots() > 0:
                self.consumable_hand.append(item)
                rs.acquire(item)
            return True
        # unknown token: every producer now emits a real key (W3/W5); never park a
        # non-game string in a consumable slot
        return False

    def _materialize_pending(self, j: JokerInstance):
        for item in j.state.pop("pending_consumables", []):
            self._materialize(item)

    # ── Harness helpers (HARNESS_NOTES hooks 10/11) ──────────────────────────

    def state_signature(self) -> tuple:
        """Canonical, hashable, Card-id-independent snapshot of the whole run (harness hook
        12 — determinism / clone-isolation tests).  Two games with equal signatures will
        produce identical futures: it covers every scalar, the blind, hand levels, jokers
        (key + edition + state), consumables (+ Negative bookkeeping), the deck composition
        and the per-blind partitions, the shop / open pack, tags, and a hash of the keyed
        PseudoRandom's full state (every per-key position + the global LuaJIT state)."""
        def card_sig(c):
            return (c.rank, c.suit, c.enhancement, c.edition, c.seal, c.debuffed,
                    c.bonus_chips, getattr(c, "face_down", False))
        def frz(v):
            if isinstance(v, dict):
                return tuple(sorted((str(k), frz(x)) for k, x in v.items()))
            if isinstance(v, (set, frozenset)):
                return tuple(sorted(map(repr, v)))
            if isinstance(v, (list, tuple)):
                return tuple(frz(x) for x in v)
            if isinstance(v, (int, float, str, bool)) or v is None:
                return v
            return repr(v)
        by_id = {c.id: c for c in self.full_deck}
        id_cards = lambda ids: tuple(sorted(card_sig(by_id[i]) for i in ids if i in by_id))  # noqa: E731
        rs = self.run_state
        snap = rs.rng.snapshot()
        # process-independent digest (hash() of str is salted per process)
        import hashlib
        rng_hash = hashlib.blake2b(
            repr((snap["seed"], sorted(snap["state"].items()), snap["rng"])).encode(),
            digest_size=16).hexdigest()
        scalars = tuple(sorted(
            (k, v) for k, v in vars(self).items()
            if (isinstance(v, (int, float, str, bool)) or v is None)
            and k not in ("_forced_card_id", "_disabled_joker_idx")))
        return (
            self.seed_str, self.state.name, scalars,
            (self.current_blind.kind, self.current_blind.chips_target, self.current_blind.boss_key,
             self.current_blind.disabled, self.current_blind.is_showdown),
            tuple(sorted(self.planet_levels.items())),
            tuple((j.key, j.edition, frz(j.state)) for j in self.jokers),
            tuple(self.consumable_hand), frz(self.negative_consumables),
            tuple(sorted(self.vouchers)), tuple(self.planets_used), tuple(self.tarots_used),
            tuple(sorted(card_sig(c) for c in self.full_deck)),
            tuple(card_sig(c) for c in self.deck), tuple(card_sig(c) for c in self.hand),
            tuple(card_sig(c) for c in self.discard_pile),
            id_cards(self._played_this_ante),
            id_cards([self._forced_card_id]) if self._forced_card_id >= 0 else (),
            tuple(sorted(self._hand_type_counts.items())),
            tuple((it.kind, it.key, it.edition, it.price, it.sold) for it in self.current_shop),
            tuple((getattr(c, "key", repr(c)), getattr(c, "edition", None), getattr(c, "seal", None))
                  for c in self.booster_choices),
            tuple(self.tag_state.keys()), frz(self.blind_tags), self.boss_blind,
            tuple(sorted(self.round_picks.items())),
            (rs.ante, tuple(sorted(rs.used_jokers)), tuple(rs.owned_jokers), tuple(rs.owned_consumables),
             tuple(sorted(rs.used_vouchers)), rs.boss_blind, rs.showman, rs.shop_joker_max,
             tuple(sorted(rs.pool_flags)) if hasattr(rs, "pool_flags") else ()),
            rng_hash,
        )

    def debug_win_blind(self):
        """Clear the current blind without scoring (effect streams untouched)."""
        if self.state == State.SELECTING_HAND:
            self.chips_scored = self.current_blind.chips_target
            self.state = State.ROUND_EVAL

    def debug_add_joker(self, key: str, edition=None) -> JokerInstance:
        """Own a joker AND do the run_state bookkeeping a purchase would."""
        j = JokerInstance(key, edition or "None")
        return emplace_joker(self, j)

    # ── Boss reroll (Director's Cut / Retcon; Boss Tag goes through the TagContext) ──

    def can_reroll_boss(self) -> bool:
        if self.state != State.BLIND_SELECT or self.blind_idx > 2:
            return False
        if self.current_blind.is_pvp:
            return False      # MLB: the Nemesis is fixed (Director's Cut / Retcon / Boss Tag are banned anyway)
        if not can_afford(self, 10):
            return False
        return "v_retcon" in self.vouchers or ("v_directors_cut" in self.vouchers and not self._boss_rerolled)

    def _reroll_boss(self, from_tag: bool = False):
        """button_callbacks.lua:2800-2848: $10 unless from a Boss Tag; sets ``boss_rerolled``;
        ``get_new_boss()`` on the same 'boss' stream; re-runs the ``new_blind_choice`` pass."""
        if not from_tag:
            self.dollars -= 10
        self._boss_rerolled = True
        self.run_state.ante = self.ante
        self.boss_blind = _gen.reroll_boss(self.run_state)
        self._boss_blind_ante = self.ante
        if self.current_blind.kind == "Boss":
            self._prepare_next_blind()
        if not from_tag:
            self._handle_blind_choice(self.tag_state.on_new_blind_choice(self._tag_ctx()))

    def _start_blind(self):
        """Begin playing the current blind."""
        self.chips_scored = 0
        self.hands_left = self.base_hands
        self.discards_left = self.base_discards
        self.hand_size = HAND_SIZE + self.hand_size_mod  # base + permanent Ouija/Ectoplasm deltas
        self.played_hand_types_this_round = set()
        self.last_played_hand_type = ""   # stale value would mislead Blue Seal
        self._forced_card_id = -1         # The Cerulean Bell picks a fresh card
        self._hands_played_round = 0      # G.GAME.current_round.hands_played (Sixth Sense, DNA)
        self._discards_used_round = 0     # G.GAME.current_round.discards_used (The House)
        # MLB: G.FUNCS.select_blind sends `newRound` ($MOD/ui/game/functions.lua:75) ->
        # the server's once-per-round life-loss blocker resets (Client.ts resetBlocker);
        # a Nemesis blind is now in progress (startBlind) with the opponent at 0 / full hands.
        self.life_lost_this_round = False
        self.pvp_ready = False
        if self.current_blind.is_pvp:
            self.pvp_started = True
            self.pvp_opponent_score = 0
            self.pvp_opponent_hands = 0
            self.current_blind.chips_target = 0

        # Apply passive joker effects (constant while owned, not cumulative).
        # Hand size: Juggler +1, Troubadour +2, Merry Andy -1, Stuntman -2,
        # Turtle Bean +h_size (5, -1 per round) — base.passive_modifiers (W3).
        joker_keys = {j.key for j in self.jokers}
        self.hand_size = max(1, self.hand_size + passive_modifiers(self.jokers)["hand_size"])
        if "j_drunkard" in joker_keys:
            self.discards_left += 1
        if "j_troubadour" in joker_keys:
            self.hands_left = max(1, self.hands_left - 1)
        if "j_merry_andy" in joker_keys:
            self.discards_left += 3

        # Apply voucher hand size adjustments
        if "v_paint_brush" in self.vouchers:
            self.hand_size += 1
        if "v_palette" in self.vouchers:
            self.hand_size += 1

        # Order (state_events.lua:290-353 new_round, game.lua:3207-3222 DRAW_TO_HAND):
        #   set_blind (boss start effects, debuffs)  ->  `setting_blind` joker hooks
        #   (Chicot disable, Riff-raff, Marble's Stone INTO THE DECK, Cartomancer, Madness,
        #   Burglar, Dagger)  ->  'nr<ante>' shuffle (so Marble's card is in it)  ->
        #   round_start_bonus tags (Juggle)  ->  draw  ->  `first_hand_drawn` (Certificate).
        # Before W5 the engine drew the hand first and fired the hooks afterwards.
        self.run_state.new_round()
        # Chicot (card.lua:596 / :2492): the boss is disabled — no start-of-blind effect,
        # no play-time effect, target back to the plain boss amount (Blind:disable,
        # blind.lua:233-260).
        # (Blind:disable is a no-op at a PvP blind -- $MOD/ui/game/blind_hud.lua:190-195.)
        if self.current_blind.is_boss and not self.current_blind.is_pvp            and any(j.key == "j_chicot" for j in self.jokers):
            self.current_blind.chips_target = int(
                self.current_blind.chips_target / BOSS_CHIP_MULT.get(self.current_blind.boss_key, 1.0))
            self.current_blind.boss_key = ""
            self.current_blind.disabled = True
        if self.current_blind.is_boss:
            self._apply_boss_start(self.current_blind.boss_key)
        # Blind:set_blind ends with `debuff_card` over G.playing_cards (blind.lua:208-210):
        # one predicate (`_boss_debuffs_card`) evaluated here and re-evaluated at play
        # time, so cards that enter the hand later (created / hand-edited) are covered.
        self._refresh_card_debuffs(self.full_deck)
        # Fire setting_blind joker hooks (one shared context, W3 base.fire_hook):
        # created cards (Riff-raff 'Joker1rif<ante>', Cartomancer 'Tarotcar<ante>',
        # Marble 'marb_fr', Certificate 'cert_fr'/'certsl') are drawn through
        # run_state inside the hooks and materialised by drain_joker_state.
        fire_hook(self, "on_blind_selected")
        for j in list(self.jokers):
            if "planet_upgrade" in j.state:
                ht = j.state.pop("planet_upgrade")
                self.planet_levels[ht] = self.planet_levels.get(ht, 1) + 1
            # Event-based game-state modifiers (Burglar — fires once per blind)
            extra_hands = j.state.pop("extra_hands", 0)
            if extra_hands:
                self.hands_left += extra_hands
            if j.state.pop("zero_discards", False):
                self.discards_left = 0
        # Ceremonial Dagger: destroy joker to the right, gain 2x sell value as mult
        to_destroy = []
        for i, j in enumerate(self.jokers):
            if j.state.pop("destroy_right", False) and i + 1 < len(self.jokers):
                target = self.jokers[i + 1]
                from .jokers.base import joker_sell_value, remove_joker
                j.state["mult"] = j.state.get("mult", 0) + joker_sell_value(target) * 2
                to_destroy.append(target)
        for target in to_destroy:
            remove_joker(self, target)
        # Madness (card.lua:2503-2520): destroy a random OTHER joker on Small/Big —
        # `pseudorandom_element(destructable_jokers, 'madness')`, board order.
        for j in list(self.jokers):
            if j.state.pop("destroy_random", False):
                others = [k for k in self.jokers if k is not j]
                if others:
                    victim, _ = self.run_state.rng.pseudorandom_element(others, "madness")
                    from .jokers.base import remove_joker
                    remove_joker(self, victim)
        # 'nr<ante>' shuffle of the (possibly grown) deck, Juggle, then the first draw
        self._init_deck()
        self.tag_state.on_round_start(self._tag_ctx())
        self._draw_to_full()
        # `first_hand_drawn` (card.lua:2462): Certificate's sealed card joins the hand
        fire_hook(self, "on_first_hand_drawn")
        self.state = State.SELECTING_HAND

    # Suit debuffed by each suit-targeting boss. bl_goad debuffed Clubs before the
    # 2026-07-29 audit — that is The Club's effect; the real Goad targets Spades.
    BOSS_DEBUFF_SUIT = {
        "bl_goad":   "Spades",
        "bl_club":   "Clubs",
        "bl_window": "Diamonds",
        "bl_head":   "Hearts",
    }

    def _boss_debuffs_card(self, card: Card) -> bool:
        """``Blind:debuff_card`` (blind.lua:624-648) for a playing card, as a predicate:
        suit bosses use ``is_suit(suit, bypass_debuff)`` (Wild cards match every suit,
        Smeared pairs the suits, Stone never), The Plant uses ``is_face(true)`` (so
        Pareidolia makes EVERY card a face), The Pillar reads ``played_this_ante``,
        Verdant Leaf debuffs everything until a Joker is sold.  A disabled boss
        (Chicot / Luchador) and non-boss blinds debuff nothing."""
        blind = self.current_blind
        if not blind.is_boss or blind.disabled:
            return False
        boss_key = blind.boss_key
        if boss_key == "bl_final_leaf":
            return self._verdant_active
        suit = self.BOSS_DEBUFF_SUIT.get(boss_key)
        if suit is not None:
            if card.enhancement == "Stone":
                return False
            if card.enhancement == "Wild":
                return True
            if card.suit == suit:
                return True
            if any(j.key == "j_smeared" for j in self.jokers):
                red = {"Hearts", "Diamonds"}
                return (card.suit in red) == (suit in red)
            return False
        if boss_key == "bl_plant":
            return card.is_face_card or any(j.key == "j_pareidolia" for j in self.jokers)
        if boss_key == "bl_pillar":
            return card.id in self._played_this_ante
        return False

    def _refresh_card_debuffs(self, cards) -> None:
        """Re-evaluate the boss debuff on ``cards`` (the game does this on every
        ``set_base`` / ``set_ability`` / blind change — card.lua:143, 365, 561)."""
        for c in cards:
            c.debuffed = self._boss_debuffs_card(c)

    def hand_eval_flags(self, jokers=None) -> dict:
        """Flags for ``hand_eval.evaluate_hand`` from the (active) board — Four Fingers,
        Shortcut, Smeared Joker, Pareidolia (``base.hand_eval_flags``)."""
        return hand_eval_flags(self.jokers if jokers is None else jokers)

    def _apply_boss_start(self, boss_key: str):
        """Apply start-of-blind boss effects (card debuffs are applied by
        ``_refresh_card_debuffs`` right after this, from ``_boss_debuffs_card``)."""
        if boss_key in self.BOSS_DEBUFF_SUIT:
            pass   # -> _boss_debuffs_card
        elif boss_key == "bl_manacle":
            self.hand_size = max(1, self.hand_size - 1)
        elif boss_key == "bl_needle":
            self.hands_left = 1
        elif boss_key == "bl_water":
            self.discards_left = 0
        elif boss_key in ("bl_plant", "bl_pillar"):
            pass   # -> _boss_debuffs_card
        elif boss_key == "bl_final_leaf":
            # All cards debuffed until one Joker is sold (see sell handling).
            self._verdant_active = True
        elif boss_key == "bl_final_acorn":
            # Flips and shuffles all Jokers (blind.lua:190-205): THREE
            # `G.jokers:shuffle('aajk')` in a row, each a pseudoshuffle that first
            # re-sorts by sort_id — so the result is the third shuffle of the
            # creation-ordered board and three 'aajk' draws are consumed.
            if len(self.jokers) > 1:
                for _ in range(3):
                    board = sorted(self.jokers, key=lambda j: getattr(j, "sort_id", 0))
                    self.run_state.rng.pseudoshuffle(board, "aajk")
                    self.jokers = board
        elif boss_key in ("bl_fish", "bl_psychic", "bl_eye", "bl_mouth",
                          "bl_tooth", "bl_flint", "bl_serpent", "bl_hook",
                          "bl_wall", "bl_final_vessel", "bl_ox", "bl_arm",
                          "bl_final_heart", "bl_final_bell"):
            pass  # handled elsewhere: draw, play validation, scoring, or target

    def _undo_boss_debuffs(self, boss_key: str):
        """Re-enable cards after a boss blind ends."""
        if boss_key in self.BOSS_DEBUFF_SUIT or boss_key in (
                "bl_plant", "bl_pillar", "bl_final_leaf"):
            for c in self.full_deck:
                c.debuffed = False
            self._verdant_active = False

    def _init_deck(self):
        """
        Shuffle the permanent deck into a fresh draw pile for a new blind.

        This does NOT create new cards — it reshuffles the player's existing
        collection, so enhancements/seals/editions survive across blinds and
        across antes.
        """
        # CardArea:shuffle('nr'..ante) at every blind start (state_events.lua:344):
        # pseudoshuffle sorts by sort_id (creation order) first, so the result depends
        # only on the deck's composition + the 'nr<ante>' stream. Dealt from the END.
        self.run_state.ante = self.ante
        self.deck = _gen.shuffle_deck(self.run_state, self._playing_cards_sorted(),
                                      _Keys.new_round_shuffle(self.ante))
        for c in self.full_deck:
            c.face_down = False      # cards in the deck area flip on emplace into the hand
        self.hand = []
        self.discard_pile = []

    # ── Permanent deck mutation ──────────────────────────────────────────────

    def add_card(self, card: Card, to_draw_pile: bool = True):
        """Add a card to the permanent deck (and optionally the current draw pile).
        Fires the `playing_card_added` joker context (Hologram) — W3."""
        self.full_deck.append(card)
        if to_draw_pile:
            self.deck.insert(0, card)
        if self.jokers:
            fire_hook(self, "on_card_added", drain=False)

    def remove_card(self, card: Card) -> bool:
        """
        Permanently destroy a card, removing it from the deck and every
        per-blind partition. Returns True if the card was found.

        Prefer destroy_card() for in-game destruction — it also notifies jokers.
        """
        found = False
        for coll in (self.full_deck, self.deck, self.hand, self.discard_pile):
            if card in coll:
                coll.remove(card)
                found = True
        return found

    def destroy_card(self, card: Card) -> bool:
        """
        Destroy a card and fire on_card_destroyed joker hooks.

        Jokers that scale on destruction (Glass Joker gains x0.75 Mult per Glass
        card destroyed; Caino gains x0.1 per face card destroyed) depend on this
        notification, so any destruction that should feed them must come through
        here rather than remove_card().
        """
        removed = self.remove_card(card)
        ctx = self._hook_ctx()
        for j in self.jokers:
            effect = JOKER_REGISTRY.get(j.key)
            if effect and hasattr(effect, "on_card_destroyed"):
                effect.on_card_destroyed(j, card, ctx)
        return removed

    def _stay_flipped(self, card: Card, after_play: bool) -> bool:
        """``Blind:stay_flipped(G.hand, card)`` (blind.lua:605-622): The Wheel rolls
        ``'wheel' < normal/7`` per drawn card, The House flips the initial deal (no hand
        played, no discard used yet), The Mark flips faces (``is_face(true)``: Pareidolia
        counts), The Fish flips the redraw after a PLAYED hand (``prepped`` is set by
        ``press_play`` and cleared by ``drawn_to_hand``).  Disabled boss: nothing."""
        blind = self.current_blind
        if not blind.is_boss or blind.disabled:
            return False
        key = blind.boss_key
        if key == "bl_wheel":
            return self.run_state.rng.pseudorandom("wheel") < self.run_state.probabilities_normal / 7
        if key == "bl_house":
            return self._hands_played_round == 0 and self._discards_used_round == 0
        if key == "bl_mark":
            return card.is_face_card or any(j.key == "j_pareidolia" for j in self.jokers)
        if key == "bl_fish":
            return after_play
        return False

    def _draw_to_full(self, after_play: bool = False):
        """Draw up to ``hand_size``; each drawn card gets its boss debuff and, for the
        face-down bosses, its ``face_down`` flag (``after_play`` = the redraw that
        follows a played hand, which is what The Fish hides)."""
        target = self.hand_size
        while len(self.hand) < target and self.deck:
            card = self.deck.pop()
            card.debuffed = self._boss_debuffs_card(card)   # Blind:debuff_card on draw
            card.face_down = self._stay_flipped(card, after_play)
            self.hand.append(card)

    # ── Main step ────────────────────────────────────────────────────────────

    def step(self, action: dict) -> GameState:
        atype = action.get("type", "")

        if self.state == State.BLIND_SELECT:
            if atype == "play_blind":
                if self.mlb and self.current_blind.is_pvp and not self.pvp_solo:
                    # Nemesis: the button is "Ready" (G.FUNCS.mp_toggle_ready -> readyBlind);
                    # the blind starts for both players when the server sends startBlind,
                    # i.e. when MLBMatch sees both ready.  Solo (no match): start at once.
                    self.pvp_ready = True
                else:
                    self._start_blind()
            elif atype == "skip_blind":
                self._skip_blind()
            elif atype == "reroll_boss":
                if self.can_reroll_boss():
                    self._reroll_boss()

        elif self.state == State.SELECTING_HAND:
            if atype == "play":
                self._play_hand(action.get("cards", []))
            elif atype == "discard":
                self._discard(action.get("cards", []))
            elif atype == "use_consumable":
                self._use_consumable(
                    action.get("consumable_idx", 0),
                    action.get("target_cards", [])
                )

        elif self.state == State.ROUND_EVAL:
            self._end_round()

        elif self.state == State.SHOP:
            if atype == "buy":
                idx = action.get("item_idx", 0)
                if idx < len(self.current_shop):
                    item = self.current_shop[idx]
                    if buy_item(self, item) and item.kind == "booster" and self.booster_choices:
                        # §7 item 0 fix: a bought pack is opened — the agent picks from it
                        self.state = State.BOOSTER_OPEN
            elif atype == "sell_joker":
                sell_joker(self, action.get("joker_idx", 0))
                # The Verdant Leaf debuffs every card until a Joker is sold.
                if self._verdant_active:
                    self._verdant_active = False
                    for c in self.full_deck:
                        c.debuffed = False
            elif atype == "use_consumable":
                self._use_consumable(
                    action.get("consumable_idx", 0),
                    action.get("target_cards", [])
                )
            elif atype == "reroll":
                reroll_shop(self)
            elif atype == "leave_shop":
                self._end_shop()

        elif self.state == State.BOOSTER_OPEN:
            if atype == "pick_booster":
                self._pick_booster(action.get("indices", []))
            elif atype == "skip_booster":
                self._skip_booster()

        return self._obs()

    def legal_actions(self) -> list[dict]:
        """
        Return all syntactically legal actions in the current state.

        Used by MCTS to expand children. Card subsets are enumerated
        combinatorially: a 5-card hand has C(5,1)+...+C(5,5) = 31 play
        subsets. An 8-card hand has 218 play subsets (V7's action space).

        Notes:
        - Booster pick selection is reduced to "pick first valid combination
          per pick count" rather than enumerating all combinations of
          booster_choices, since most packs are pick-1 anyway. Expand later
          if MCTS needs richer pack reasoning.
        - Consumable target selection enumerates per consumable: tarots
          targeting 1-2 cards expand combinatorially; planets/spectrals with
          no targets are a single action each.
        """
        from itertools import combinations

        s = self.state
        if s == State.GAME_OVER or s == State.PVP_WAIT:
            return []

        if s == State.BLIND_SELECT:
            if self.pvp_ready:
                return []          # MLB: readied for the Nemesis, waiting for the opponent
            actions = [{"type": "play_blind"}]
            if self.current_blind.kind != "Boss":
                actions.append({"type": "skip_blind"})
            if self.can_reroll_boss():
                actions.append({"type": "reroll_boss"})
            return actions

        if s == State.SELECTING_HAND:
            actions: list[dict] = []
            n = len(self.hand)
            if n == 0:
                return actions

            # Boss psychic: only 5-card plays
            psychic = self.current_blind.boss_key == "bl_psychic"

            # Play subsets — sizes 1..5 (or exactly 5 for psychic)
            sizes = [5] if psychic else range(1, min(5, n) + 1)
            for k in sizes:
                if k > n:
                    continue
                for combo in combinations(range(n), k):
                    actions.append({"type": "play", "cards": list(combo)})

            # Discard subsets — only if discards remain (and not psychic)
            if self.discards_left > 0 and not psychic:
                for k in range(1, min(5, n) + 1):
                    for combo in combinations(range(n), k):
                        actions.append({"type": "discard", "cards": list(combo)})

            # Consumable use — enumerate per consumable
            for ci, key in enumerate(self.consumable_hand):
                actions.extend(self._consumable_target_actions(ci, key, n))

            return actions

        if s == State.ROUND_EVAL:
            # Auto-advance state — represent as a single dummy action
            return [{"type": "advance"}]

        if s == State.SHOP:
            actions: list[dict] = [{"type": "leave_shop"}]
            # Buy each affordable, slot-available item
            for i, item in enumerate(self.current_shop):
                if item.sold:
                    continue
                price = effective_price(self, item)
                if not can_afford(self, price):
                    continue
                if item.kind == "joker" and len(self.jokers) >= self.joker_slots \
                   and item.edition != "Negative":
                    continue
                if item.kind in ("planet", "tarot", "spectral") and \
                   len(self.consumable_hand) >= self.consumable_slots and item.edition != "Negative":
                    continue
                actions.append({"type": "buy", "item_idx": i})
            # Sell each owned joker
            for ji in range(len(self.jokers)):
                actions.append({"type": "sell_joker", "joker_idx": ji})
            # Reroll if affordable
            cost = max(0, self.reroll_cost - self.reroll_discount)
            if self.free_rerolls_remaining > 0 or can_afford(self, cost):
                actions.append({"type": "reroll"})
            # Use consumable
            for ci, key in enumerate(self.consumable_hand):
                actions.extend(self._consumable_target_actions(ci, key, len(self.hand)))
            return actions

        if s == State.BOOSTER_OPEN:
            actions = [{"type": "skip_booster"}]
            picks = self.booster_picks_remaining
            grantable = [i for i, c in enumerate(self.booster_choices) if self._can_grant_choice(c)]
            if not grantable or picks == 0:
                return actions
            # Enumerate index combinations of size 1..picks over the grantable cards
            for k in range(1, min(picks, len(grantable)) + 1):
                for combo in combinations(grantable, k):
                    actions.append({"type": "pick_booster", "indices": list(combo)})
            return actions

        return []

    def _consumable_target_actions(
        self, consumable_idx: int, key: str, n_hand_cards: int
    ) -> list[dict]:
        """
        Return legal use_consumable actions for a single consumable.

        Tarots that target 1 or 2 cards expand combinatorially. Planets and
        spectrals with no card targets are a single action.
        """
        from itertools import combinations
        from .consumables import (
            ALL_PLANETS, ALL_TAROTS, ALL_SPECTRALS, PLANET_HAND,
        )

        # Planets — no card target
        if key in PLANET_HAND or key in ALL_PLANETS:
            return [{"type": "use_consumable", "consumable_idx": consumable_idx,
                     "target_cards": []}]

        # Tarots that target N cards (N=1 or 2 typically). Enumerate sizes 0..2.
        # The sim accepts any size and clips internally.
        if key in ALL_TAROTS:
            out = [{"type": "use_consumable", "consumable_idx": consumable_idx,
                    "target_cards": []}]
            for k in (1, 2):
                if k > n_hand_cards:
                    break
                for combo in combinations(range(n_hand_cards), k):
                    out.append({"type": "use_consumable",
                                "consumable_idx": consumable_idx,
                                "target_cards": list(combo)})
            return out

        # Spectrals — most are no-target; a few target 1 card
        if key in ALL_SPECTRALS:
            out = [{"type": "use_consumable", "consumable_idx": consumable_idx,
                    "target_cards": []}]
            for k in (1,):
                if k > n_hand_cards:
                    break
                for combo in combinations(range(n_hand_cards), k):
                    out.append({"type": "use_consumable",
                                "consumable_idx": consumable_idx,
                                "target_cards": list(combo)})
            return out

        return [{"type": "use_consumable", "consumable_idx": consumable_idx,
                 "target_cards": []}]

    # ── Play ─────────────────────────────────────────────────────────────────

    def _play_hand(self, card_indices: list[int]):
        selected = [self.hand[i] for i in card_indices if i < len(self.hand)]
        if not selected:
            return

        boss_key = self.current_blind.boss_key
        boss_triggered = False            # G.GAME.blind.triggered (Matador) — W3
        prng = self.run_state.rng

        # Boss: cerulean bell — one card is forced into every played hand
        # (blind.lua:583 `pseudorandom_element(G.hand.cards, 'cerulean_bell')`, sort_id order)
        if boss_key == "bl_final_bell":
            if self._forced_card_id < 0 and self.hand:
                pick, _ = prng.pseudorandom_element(sort_id_order(self.hand), "cerulean_bell")
                self._forced_card_id = pick.id
            forced = next((c for c in self.hand if c.id == self._forced_card_id), None)
            if forced is not None and forced not in selected:
                if len(selected) >= 5:
                    selected[-1] = forced
                else:
                    selected.append(forced)

        # Boss: psychic — must play exactly 5 (rejected hand: `debuffed_hand`, Matador pays)
        if boss_key == "bl_psychic" and len(selected) != 5:
            fire_hook(self, "on_boss_ability_triggered")
            return

        # Boss: hook — 2 random cards of the hand are discarded after the play
        # (blind.lua:470-484: two 'hook' draws over a sort_id-ordered copy of
        # G.hand.cards, the first pick removed before the second; triggered = true).
        # The engine resolves the picks before scoring; picked cards leave both
        # the selection (if played) and the hand.
        if boss_key == "bl_hook":
            pool = sort_id_order(self.hand)
            boss_triggered = True
            for _ in range(2):
                if not pool:
                    break
                pick, idx = prng.pseudorandom_element(pool, "hook")
                pool.pop(idx)
                if pick in selected:
                    selected.remove(pick)
                if pick in self.hand:
                    self.hand.remove(pick)
                    self.discard_pile.append(pick)
            if not selected:
                self._draw_to_full(after_play=True)
                self.hands_left -= 1
                self._hands_played_round += 1
                if self.hands_left <= 0:
                    # Rejected hand exhausts the round: under MLB a lost regular blind
                    # costs a life and proceeds (MLB_NOTES.md §2 1.3d), same as the
                    # normal exhaustion path below.  (P3 close; found by W2 + W4.)
                    if self.mlb:
                        self._mlb_fail_round()
                    else:
                        self.state = State.GAME_OVER
                return

        # Boss: crimson heart — one random Joker is disabled for this hand
        # (blind.lua:594 `pseudorandom_element(non-debuffed jokers, 'crimson_heart')`,
        # sort_id order; the game rolls on every draw-to-hand, the engine per play).
        # Rolled BEFORE evaluation: a disabled Four Fingers / Shortcut / Smeared is not
        # found by `find_joker` (W5).
        active_jokers = self.jokers
        if boss_key == "bl_final_heart" and len(self.jokers) > 0:
            board = sorted(self.jokers, key=lambda j: getattr(j, "sort_id", 0))
            disabled, _ = prng.pseudorandom_element(board, "crimson_heart")
            active_jokers = [j for j in self.jokers if j is not disabled]
            boss_triggered = True

        # Boss debuffs are a property of the blind, re-checked at play time
        # (Blind:debuff_card runs on every card change; hand-edited cards included).
        self._refresh_card_debuffs(self.hand)
        hand_type, scoring_cards = evaluate_hand(selected, **self.hand_eval_flags(active_jokers))

        # Boss: eye — can't play same hand type twice
        if boss_key == "bl_eye":
            if hand_type in self.played_hand_types_this_round:
                # Rejected — still costs a hand to prevent infinite loops
                fire_hook(self, "on_boss_ability_triggered")
                self.hands_left -= 1
                self._hands_played_round += 1
                if self.hands_left <= 0:
                    # Rejected hand exhausts the round: under MLB a lost regular blind
                    # costs a life and proceeds (MLB_NOTES.md §2 1.3d), same as the
                    # normal exhaustion path below.  (P3 close; found by W2 + W4.)
                    if self.mlb:
                        self._mlb_fail_round()
                    else:
                        self.state = State.GAME_OVER
                return

        # Boss: mouth — can only play first hand type used
        if boss_key == "bl_mouth":
            if self.played_hand_types_this_round and \
               hand_type not in self.played_hand_types_this_round:
                # Rejected — still costs a hand to prevent infinite loops
                fire_hook(self, "on_boss_ability_triggered")
                self.hands_left -= 1
                self._hands_played_round += 1
                if self.hands_left <= 0:
                    # Rejected hand exhausts the round: under MLB a lost regular blind
                    # costs a life and proceeds (MLB_NOTES.md §2 1.3d), same as the
                    # normal exhaustion path below.  (P3 close; found by W2 + W4.)
                    if self.mlb:
                        self._mlb_fail_round()
                    else:
                        self.state = State.GAME_OVER
                return

        self.played_hand_types_this_round.add(hand_type)
        # Tracked for Blue Seal, which creates the Planet card for the FINAL
        # played hand type of the round.
        self.last_played_hand_type = hand_type
        # Boss: ox — playing your MOST-USED hand type sets money to $0.
        # Evaluated against the run totals as they stand BEFORE this hand.
        if boss_key == "bl_ox" and self._hand_type_counts:
            top = max(self._hand_type_counts.values())
            most_used = {h for h, n in self._hand_type_counts.items() if n == top}
            if hand_type in most_used:
                self.dollars = 0
                boss_triggered = True
        if boss_key == "bl_arm" and self.planet_levels.get(hand_type, 1) > 1:
            boss_triggered = True                      # blind.lua:550-558
        if boss_key in ("bl_tooth", "bl_flint"):
            boss_triggered = True                      # blind.lua:496-515

        # Run totals for The Ox ("most-used hand type"), and card ids for The
        # Pillar ("cards played this ante are debuffed at the boss").
        self._hand_type_counts[hand_type] = self._hand_type_counts.get(hand_type, 0) + 1
        if not self.current_blind.is_boss:
            for c in selected:
                self._played_this_ante.add(c.id)

        score, ctx = score_hand(
            scoring_cards=scoring_cards,
            all_cards=selected,
            hand_type=hand_type,
            jokers=active_jokers,
            planet_levels=self.planet_levels,
            hands_left=self.hands_left - 1,
            discards_left=self.discards_left,
            dollars=self.dollars,
            ante=self.ante,
            deck_remaining=len(self.deck),
            rng=prng,                                   # W3: keyed PseudoRandom
            # Played cards are still in self.hand at this point (they move to the
            # discard pile after scoring), so held = hand minus the selection.
            held_cards=[c for c in self.hand if c not in selected],
            full_deck=self.full_deck,
            hand_type_counts=self._hand_type_counts,
            run_state=self.run_state,
            probabilities_normal=sync_probabilities(self),
            round_cards=from_round_picks(getattr(self, "round_picks", None)),
            joker_slots=self.joker_slots,
            consumable_slots=self.consumable_slots,
            consumables=self.consumable_hand,
            boss_triggered=boss_triggered,
            hands_played=self._hands_played_round,
            plasma=self.plasma,                        # W3: Plasma Deck final_scoring_step balance
        )
        self._hands_played_round += 1

        # Boss: flint — halve chips and mult (approximate: halve score)
        if self.current_blind.boss_key == "bl_flint":
            score = score // 2

        # Boss: tooth — lose $1 per card played
        if self.current_blind.boss_key == "bl_tooth":
            self.dollars = max(0, self.dollars - len(selected))

        # Boss: arm — permanently reduce the played hand's level by 1
        if self.current_blind.boss_key == "bl_arm":
            self.planet_levels[hand_type] = max(
                1, self.planet_levels.get(hand_type, 1) - 1)

        self.chips_scored += score
        self.hands_left -= 1

        # Apply pending side-effects from scoring: money, created consumables
        # (real keys drawn through run_state), DNA's copy into hand, Sixth
        # Sense's destroyed 6, joker self-destructs — base.drain_joker_state (W3).
        drain_joker_state(self, ctx)

        # Move played cards from hand to the discard pile (they stay in full_deck
        # and return to the draw pile at the start of the next blind)
        for c in selected:
            c.face_down = False      # G.play:emplace flips a face-down card (cardarea.lua:38)
            if c in self.hand:
                self.hand.remove(c)
                self.discard_pile.append(c)

        # Glass cards shatter AFTER all scoring: one 'glass' roll per scoring
        # Glass card (not per retrigger), `< probabilities.normal / 4`, in
        # scoring order (state_events.lua:951-963).
        normal = self.run_state.probabilities_normal
        for glass_card in ctx.glass_scored:
            if glass_card not in self.full_deck:
                continue
            if prng.pseudorandom("glass") < normal / 4:
                self.destroy_card(glass_card)

        # Boss: serpent — discard remaining hand after play, redraw
        if self.current_blind.boss_key == "bl_serpent":
            self.hand = []

        self._draw_to_full(after_play=True)     # The Fish: this redraw is face down

        # Blue Seal fires at END OF ROUND for cards held in hand, and Purple Seal
        # fires on DISCARD — neither belongs here. Both were wired to the play
        # path before the 2026-07-29 audit. See _end_round and _discard.

        # Check win / loss
        if self.mlb and self.current_blind.is_pvp:
            # Nemesis: the mod's Game:update_hand_played ($MOD/ui/game/game_state.lua:163-236)
            # never ends the round on `chips >= blind.chips` -- every hand is played unless
            # the server ends the PvP early (end_pvp); out of hands -> wait for the enemy.
            if self.hands_left <= 0:
                self.state = State.PVP_WAIT
                if self.pvp_solo:
                    self.end_pvp()        # no opponent / no match: nothing to wait for
            else:
                self._mlb_check_deck_out()
        elif ctx.prevent_loss and self.chips_scored >= self.current_blind.chips_target * 0.25:
            # Mr. Bones: prevent death if >= 25% reached
            self.chips_scored = self.current_blind.chips_target
            self.state = State.ROUND_EVAL
        elif self.chips_scored >= self.current_blind.chips_target:
            self.state = State.ROUND_EVAL
        elif self.hands_left <= 0:
            if self.mlb:
                self._mlb_fail_round()
            else:
                self.state = State.GAME_OVER
        elif self.mlb:
            self._mlb_check_deck_out()

    # ── MLB round outcomes (Phase 2 W1, MLB_NOTES.md §2) ─────────────────────

    def lose_life(self) -> bool:
        """The server's ``Client.loseLife('round')`` + the client's ``playerInfo`` handler
        ($MOD/networking/action_handlers.lua:510-523): at most ONE life per round
        (``roundLivesBlocker``, reset by ``newRound`` at every blind start), ``lives -= 1``,
        and with ``gold_on_life_loss`` the comeback counter bumps and the next Cash Out
        owes ``4 * comeback_bonus``.  Returns True when a life was actually lost."""
        if self.life_lost_this_round:
            return False
        self.life_lost_this_round = True
        self.lives -= 1
        self.comeback_bonus += 1
        self.comeback_bonus_given = False
        return True

    def _mlb_fail_round(self, hands_used: Optional[int] = None):
        """A NON-PvP blind is lost ($MOD/ui/game/game_state.lua:245-261): the mod sets
        ``blind.chips = -1`` so vanilla ``end_round`` treats the round as won (the run
        proceeds, the blind's reward is paid, a Boss still eases the ante) and sends
        ``failRound`` -> ``death_on_round_loss`` -> a life, unless ``hands_used == 0``
        (action_handlers.lua:1143-1149).  0 lives -> ``loseGame`` -> GAME_OVER at once."""
        if hands_used is None:
            hands_used = self._hands_played_round
        if hands_used > 0:
            self.lose_life()
        if self.lives <= 0:
            self.state = State.GAME_OVER
        else:
            self.state = State.ROUND_EVAL

    def _mlb_check_deck_out(self):
        """Hand AND draw pile empty mid-blind (MLB only).  Two mod paths:
        (a) >= 1 hand played ($MOD/ui/game/game_state.lua:287-311): ``hands_left = 0``; a
            regular blind ends (won or failed on chips), a Nemesis reports ``playHand(chips, 0)``
            and waits;
        (b) no hand played, discards used (``MP.handle_deck_out``, game_state.lua:446-459):
            the round ends as "defeated", ``fail_round(1)`` costs a life, a Nemesis reports
            ``playHand(0, 0)``.
        Vanilla's behaviour (``end_round()`` straight from update_selecting_hand) is untouched."""
        if self.state != State.SELECTING_HAND or self.hand or self.deck:
            return
        blind = self.current_blind
        if self._hands_played_round == 0:
            if self._discards_used_round == 0:
                return
            self.hands_left = 0
            if blind.is_pvp:
                self.lose_life()             # fail_round(1); the PvP resolution is blocked (one life/round)
                self.state = State.PVP_WAIT if self.lives > 0 else State.GAME_OVER
                if self.pvp_solo:
                    self.end_pvp()
            else:
                self._mlb_fail_round(hands_used=1)
            return
        self.hands_left = 0
        if blind.is_pvp:
            self.state = State.PVP_WAIT
            if self.pvp_solo:
                self.end_pvp()
        elif self.chips_scored >= blind.chips_target:
            self.state = State.ROUND_EVAL
        else:
            self._mlb_fail_round()

    def set_pvp_info(self, score: int, hands_left: int) -> None:
        """``enemyInfo`` (action_handlers.lua:349-459): the opponent's live score becomes the
        Nemesis target; their hands-left is shown next to it."""
        self.pvp_opponent_score = int(score)
        self.pvp_opponent_hands = int(hands_left)
        if self.current_blind.is_pvp:
            self.current_blind.chips_target = int(score)

    def end_pvp(self) -> None:
        """``endPvP`` (action_handlers.lua:472-508 -> game_state.lua:229-233 / 313-318): the
        Nemesis round is over for this player -- whatever hands remain are forfeited (no
        unused-hand money at a PvP blind anyway) and the round goes to Cash Out.  The life,
        if any, was already taken by ``lose_life`` (playerInfo precedes endPvP)."""
        if not self.pvp_started:
            return
        self.pvp_started = False
        if self.state in (State.SELECTING_HAND, State.PVP_WAIT):
            self.state = State.GAME_OVER if self.lives <= 0 else State.ROUND_EVAL

    def _discard(self, card_indices: list[int]):
        if self.discards_left <= 0:
            return
        selected = [self.hand[i] for i in card_indices if i < len(self.hand)]
        if not selected:
            return

        # Purple Seal (card.lua:2255-2268, `calculate_seal{discard}` runs BEFORE the
        # jokers' discard context, state_events.lua:400-404): per discarded
        # Purple-sealed card, if there is a free slot, create_card('Tarot', ...,
        # '8ba') — the stream shared with 8 Ball. Debuffed cards are skipped.
        for c in selected:
            if c.seal == "Purple" and not c.debuffed and \
               len(self.consumable_hand) < self.consumable_slots:
                self.grant_created("purple_seal")

        # Fire on_discard joker hooks (one shared context; Trading Card's destroy
        # and money, Castle / Mail-In Rebate round cards) and drain — W3.
        fire_hook(self, "on_discard", selected)

        for c in selected:
            c.face_down = False
            if c in self.hand:
                self.hand.remove(c)
                self.discard_pile.append(c)
        self.discards_left -= 1
        self._discards_used_round += 1
        self._draw_to_full()
        if self.mlb:
            self._mlb_check_deck_out()

    def _use_consumable(self, consumable_idx: int, target_cards: list[int]):
        if consumable_idx >= len(self.consumable_hand):
            return
        key = self.consumable_hand[consumable_idx]
        success = False

        if key in PLANET_HAND:
            success = apply_planet(self, key)
        elif key in {t for t in ALL_TAROTS}:
            success = apply_tarot(self, key, target_cards)
        elif key in {s for s in ALL_SPECTRALS}:
            success = apply_spectral(self, key, target_cards)

        if success:
            self.consumable_hand.pop(consumable_idx)
            self._consumable_removed(key)
            if self.state == State.SELECTING_HAND:
                # set_base / set_ability re-run Blind:debuff_card (card.lua:143, 365)
                self._refresh_card_debuffs(self.hand)

    # ── Round end / shop ─────────────────────────────────────────────────────

    def _end_round(self):
        """Round won: payout rows, end_round bookkeeping, the ante transition after a boss,
        Cash Out, then the shop (state_events.lua:87-288, button_callbacks.lua:2912-2957).

        Generation-relevant order (GENERATION_SPEC §16.3): after a BOSS, ``ease_ante(1)``
        runs first, then 'Voucher<new ante>' (end_round), then at Cash Out the deck shuffle
        'cashout<ante>', 'Tag<ante>' x2 and 'boss' (``reset_blinds``).  ``generate.defeat_boss``
        performs voucher -> tags -> boss; the post-boss shop already belongs to the new ante.
        """
        blind = self.current_blind
        was_boss = blind.is_boss
        # state_events.lua:124 — discards left over at the end of every PLAYED round (Garbage Tag)
        self.unused_discards += max(0, self.discards_left)

        # Payout: base blind reward + $1 per unused hand + (joker $ rows) + tag rows + interest.
        # Interest is computed on the balance BEFORE any row is paid (state_events.lua:1191).
        # Deck / stake modifiers (W3): Red stake+ pays $0 for the Small Blind (blind.lua:84);
        # Green Deck pays money_per_hand ($2) per hand, money_per_discard ($1) per discard and
        # no interest (state_events.lua:1166-1173, :1191).
        # MLB ($MOD/lovely/game.toml:146-154): the blind reward is paid at a PvP blind won
        # OR lost, and a failed regular blind reaches this point as "won" (blind.chips = -1,
        # game_state.lua:250) so it is paid too.  game.toml:93-100: NO unused-hand money at a
        # PvP blind (hands left when the server ends the PvP early are simply forfeited).
        blind_money = 0 if (blind.kind == "Small" and self.no_small_blind_reward) else blind.money_reward
        earnings = self.hands_left * (self.money_per_hand or HAND_PAYOUT)
        if self.mlb and blind.is_pvp:
            earnings = 0      # only the `hands_left > 0` row is patched (game.toml:93-100) --
        # -- the Green Deck `money_per_discard` row (state_events.lua:1170-1174) is NOT,
        # so discard money is still paid at a PvP blind.
        if self.discards_left > 0 and self.money_per_discard:
            earnings += self.discards_left * self.money_per_discard
        interest = 0 if self.no_interest else min(self.dollars // INTEREST_RATE, self.interest_cap)
        self.dollars += blind_money + earnings
        # Investment Tag rows (tag.lua:117-130) — paid through ctx.add_dollars
        self.tag_state.on_round_eval(self._tag_ctx(last_blind_was_boss=was_boss))
        # Back:trigger_effect{context='eval'} (state_events.lua:1163): Anaglyph's Double Tag
        # after a boss; queued as an event, so it lands after the synchronous tag `eval` loop.
        _decks.on_round_eval(self, was_boss)
        self.dollars += interest
        # MLB comeback money ($MOD/lovely/game.toml:15-49, patched in right AFTER the interest
        # row of G.FUNCS.evaluate_round, outside the `dollars >= 5` guard): once per life-loss
        # event, 4 x cumulative lives lost (MP.GAME.comeback_bonus), then `given = true`.
        if self.mlb and not self.comeback_bonus_given:
            self.comeback_bonus_given = True
            self.dollars += MLB_COMEBACK_PER_LIFE * self.comeback_bonus

        # Boss blind beaten: fire on_boss_beaten hooks
        if was_boss:
            fire_hook(self, "on_boss_beaten")
            self._undo_boss_debuffs(blind.boss_key)

        # Pre-compute deck stats for jokers that need them (e.g. Cloud 9)
        deck_nines = sum(1 for c in self.full_deck if c.rank == 9)
        for j in self.jokers:
            j.state["deck_nines"] = deck_nines  # for Cloud 9
        # Fire on_round_end hooks (one shared context); pending money / created cards /
        # self-destructs are drained by base.drain_joker_state (created cards -> _materialize)
        fire_hook(self, "on_round_end")

        # Gold enhancement: "$3 if this card is held in hand at end of round."
        # (Gold SEAL is different — "$3 when played and scores" — and now lives
        # in scoring.py. The two were conflated before the 2026-07-29 audit.)
        for c in self.hand:
            if c.enhancement == "Gold":
                self.dollars += 3

        # Blue Seal: "Creates the Planet card for the final played poker hand of
        # the round if this card is held in hand." End-of-round, held-in-hand.
        # Forced key (no pool draw), append 'blusl' — generate.blue_seal.
        if self.last_played_hand_type:
            for c in self.hand:
                if c.seal == "Blue" and len(self.consumable_hand) < self.consumable_slots:
                    self._sync_run_state()
                    cg = _gen.blue_seal(self.run_state, self.last_played_hand_type)
                    if cg.key in PLANET_HAND:
                        self.consumable_hand.append(cg.key)
                        self.run_state.acquire(cg.key)

        # end_round cleanup (state_events.lua:270-271): Juggle hand size revert, D6 base clear
        self.tag_state.on_round_end_cleanup(self._tag_ctx())

        # Reset hand size mods from boss (bl_manacle)
        if blind.boss_key == "bl_manacle":
            self.hand_size = HAND_SIZE + self.hand_size_mod  # restore (voucher adjustments persist)

        # ── Ante transition (boss defeated): ease_ante, then the new ante's draws ──
        if was_boss:
            self._played_this_ante.clear()      # played_this_ante = nil (state_events.lua:262)
            self.ante += 1
            self.run_state.ante = self.ante - 1
            info = _gen.defeat_boss(self.run_state)   # ante += 1; 'Voucher<a>', 'Tag<a>' x2, 'boss'
            assert self.run_state.ante == self.ante
            self.boss_blind = info["boss"]
            self._boss_blind_ante = self.ante
            self.blind_tags = {"Small": info["tag_small"], "Big": info["tag_big"]}
            self._boss_rerolled = False
        self._round_end_resets()                 # 'idol<a>' / 'mail<a>' / 'anc<a>' / 'cas<a>'

        # ── Cash Out ──
        self.run_state.ante = self.ante
        _gen.shuffle_deck(self.run_state, self._playing_cards_sorted(), _Keys.cashout_shuffle(self.ante))
        self.tag_state.on_cash_out()             # per-shop D6 / Coupon guards reset
        self._end_blind_and_enter_shop()

    def _end_shop(self):
        """Leave the shop: every unbought shelf card is ``Card:remove``d (released from
        ``used_jokers``), Perkeo's ``on_shop_leave`` fires, then the next blind is selected."""
        fire_hook(self, "on_shop_leave")          # Perkeo
        if self._shop_gen is not None:
            self.run_state.release_shop(self._shop_gen)
        self._shop_gen = None
        self.current_shop = []
        self.run_state.shop_voucher_keys = []
        self._advance_blind()

    def _advance_blind(self):
        """Move to the next blind on the blind-select screen (the ante itself advances at
        boss defeat, see ``_end_round``)."""
        self.blind_idx += 1
        if self.blind_idx >= 3:
            self.blind_idx = 0
            # The Pillar only debuffs cards played during THIS ante's earlier blinds
            self._played_this_ante.clear()
            # MLB: `G.GAME.win_ante = 999` while the round ends and `game_won = nil`
            # ($MOD/ui/game/game_state.lua:264-276, lovely/end_round.toml:30-43): there is
            # no ante-8 win, the match runs until a player has 0 lives (endless scaling).
            if self.ante > 8 and not self.mlb:
                self.state = State.GAME_OVER
                return
        self._prepare_next_blind()
        self._enter_blind_select()

    def _skip_blind(self):
        """Skip a non-Boss blind (Boss can't be skipped) — G.FUNCS.skip_blind
        (button_callbacks.lua:2740-2782): ``skips += 1``, joker ``skip_blind`` hooks, the
        blind's tag is acquired (``add_tag``) and the ``immediate`` + ``new_blind_choice``
        passes run.  There is NO shop after a skip: play moves straight to the next blind.
        """
        if self.current_blind.kind == "Boss":
            return
        kind = self.current_blind.kind
        self.skips += 1
        # Fire blind_skipped joker hooks (Throwback)
        fire_hook(self, "on_blind_skipped")
        # blind_on_deck advances before the tag events
        self.blind_idx += 1
        self._prepare_next_blind()
        tag_key = self.blind_tags.get(kind)
        if tag_key:
            out = self.tag_state.skip_blind(
                tag_key, self._tag_ctx(), blind_type=kind,
                orbital_hand=self._orbital_choices.get((self.ante, kind)))
            self._handle_blind_choice(out)

    def _end_blind_and_enter_shop(self):
        self.reroll_cost = 5
        # current_round.free_rerolls: one free reroll per owned Chaos the Clown (card.lua:601)
        self.free_rerolls_remaining = self.free_rerolls_per_round + passive_modifiers(self.jokers)["free_rerolls"]
        generate_shop(self)        # sets current_shop (+ _shop_gen)
        self.state = State.SHOP

    # ── Booster state machine (State.BOOSTER_OPEN) ───────────────────────────

    def _open_booster(self, pack_key: str, free: bool = False, return_state: Optional[State] = None):
        """Open a pack: ``generate.open_pack`` creates the cards (each marks ``used_jokers``
        as created; the shelf on display is excluded), the agent then picks
        ``BOOSTER_PICKS`` of them.  ``return_state`` is where the game goes when the pack
        closes (SHOP for a bought pack; BLIND_SELECT for a tag pack, which also re-runs
        the ``new_blind_choice`` pass).  Fires ``on_booster_opened`` (Hallucination)."""
        enter_now = return_state is not None
        rs_state = return_state or self.state
        choices = booster_contents(self, pack_key)
        tkey = _gk.booster_type_key(pack_key)
        self.booster_choices = choices
        self.booster_picks_remaining = BOOSTER_PICKS.get(tkey, 1)
        self.booster_pack_key = tkey
        self._booster_return_state = rs_state if rs_state != State.BOOSTER_OPEN else State.SHOP
        self._booster_free = free
        if enter_now:
            self.state = State.BOOSTER_OPEN
        fire_hook(self, "on_booster_opened")      # Hallucination ('halu<ante>' then 'Tarothal<ante>')

    def _can_grant_choice(self, choice) -> bool:
        if isinstance(choice, BoosterChoice):
            if choice.is_joker:
                return len(self.jokers) < self.joker_slots or choice.edition == "Negative"
            if choice.is_consumable:
                return len(self.consumable_hand) < self.consumable_slots or choice.edition == "Negative"
            return True
        if isinstance(choice, str):
            return len(self.consumable_hand) < self.consumable_slots or \
                (choice in JOKER_REGISTRY and len(self.jokers) < self.joker_slots)
        return isinstance(choice, tuple) and choice[0] == "card"

    def _grant_choice(self, choice) -> bool:
        """Take one card from the open pack into the owned area (``run_state.acquire``)."""
        if isinstance(choice, BoosterChoice):
            if choice.is_joker:
                if not self._can_grant_choice(choice):
                    return False
                j = JokerInstance(choice.key, choice.edition)
                for flag in ("eternal", "perishable", "rental"):
                    if getattr(choice, flag):
                        j.state[flag] = True
                emplace_joker(self, j)
                return True
            if choice.is_consumable:
                if not self._can_grant_choice(choice):
                    return False
                if choice.edition == "Negative":
                    self.add_negative_consumable(choice.key)
                    return True
                self.consumable_hand.append(choice.key)
                self.run_state.acquire(choice.key)
                return True
            if choice.card is not None:
                self.add_card(choice.card)
                return True
            return False
        # legacy shapes
        if isinstance(choice, str):
            if choice in JOKER_REGISTRY:
                if len(self.jokers) < self.joker_slots:
                    emplace_joker(self, JokerInstance(choice))
                    return True
                return False
            if len(self.consumable_hand) < self.consumable_slots:
                self.consumable_hand.append(choice)
                self.run_state.acquire(choice)
                return True
            return False
        if isinstance(choice, tuple) and choice[0] == "card":
            self.add_card(choice[1])
            return True
        return False

    def _pick_booster(self, indices: list[int]):
        """Pick up to ``booster_picks_remaining`` cards (indices into ``booster_choices``).
        Ungrantable picks (no slot) are ignored; a pick call that grants nothing closes the
        pack like a skip.  When the picks are used up the pack closes."""
        picked = []
        for idx in indices:
            if self.booster_picks_remaining <= 0:
                break
            if 0 <= idx < len(self.booster_choices) and idx not in picked:
                if self._grant_choice(self.booster_choices[idx]):
                    picked.append(idx)
                    self.booster_picks_remaining -= 1
        remaining = [c for i, c in enumerate(self.booster_choices) if i not in picked]
        self.booster_choices = remaining
        if not picked or self.booster_picks_remaining <= 0 or not remaining \
           or not any(self._can_grant_choice(c) for c in remaining):
            self._close_booster()

    def _skip_booster(self):
        """Close the pack without (further) picks; fires ``on_booster_skipped`` (Red Card)."""
        fire_hook(self, "on_booster_skipped")     # Red Card
        self._close_booster()

    def _close_booster(self):
        """``end_consumeable`` / ``skip_booster``: the unchosen cards are ``Card:remove``d
        (``run_state.release_pack``); return to the shop, or to blind select where the
        ``new_blind_choice`` pass is re-run (button_callbacks.lua:2617-2619)."""
        gens = [c.gen for c in self.booster_choices if isinstance(c, BoosterChoice) and c.gen is not None]
        self.run_state.release_pack(gens)
        self.booster_choices = []
        self.booster_picks_remaining = 0
        self.booster_pack_key = None
        ret = self._booster_return_state or State.SHOP
        self._booster_return_state = None
        self._booster_free = False
        self.state = ret
        if ret == State.BLIND_SELECT:
            self._handle_blind_choice(self.tag_state.on_new_blind_choice(self._tag_ctx()))

    # ── Observation ──────────────────────────────────────────────────────────

    def _obs(self) -> GameState:
        return GameState(
            state=self.state,
            ante=self.ante,
            blind_kind=self.current_blind.kind,
            chips_target=self.current_blind.chips_target,
            chips_scored=self.chips_scored,
            hands_left=self.hands_left,
            discards_left=self.discards_left,
            dollars=self.dollars,
            hand=list(self.hand),
            deck_remaining=len(self.deck),
            jokers=list(self.jokers),
            consumable_hand=list(self.consumable_hand),
            planet_levels=dict(self.planet_levels),
            shop_items=list(self.current_shop),
            done=(self.state == State.GAME_OVER),
            won=(self.match_won if self.mlb else (self.ante > 8 and self.state == State.GAME_OVER)),
            info={
                "boss_key": self.current_blind.boss_key,
                "vouchers": list(self.vouchers),
                "booster_choices": list(self.booster_choices),
                # MLB (W1): MP.GAME mirror -- zeros / inert in vanilla
                "lives": self.lives,
                "is_pvp": self.current_blind.is_pvp,
                "pvp_started": self.pvp_started,
                "opponent_score": self.pvp_opponent_score,
                "opponent_hands": self.pvp_opponent_hands,
                "comeback_bonus": self.comeback_bonus,
                "comeback_bonus_given": self.comeback_bonus_given,
            }
        )


# ════════════════════════════════════════════════════════════════════════════
# W2 helpers: TagContext against the engine (tags.py hooks, TAGS_NOTES §3)
# ════════════════════════════════════════════════════════════════════════════

def _fast_clone_run_state(rs):
    """``RunState.clone()`` without deepcopy: every RunState field is a primitive, a FLAT
    container (set/list/dict of primitives) or the PseudoRandom -- a shallow copy plus
    one-level container copies is exact and ~5x cheaper (MCTS clone budget)."""
    import copy as _copy
    new = _copy.copy(rs)
    for k, v in vars(rs).items():
        if k == "rng":
            new.rng = v.clone()
        elif isinstance(v, (set, list, dict)):
            setattr(new, k, v.copy())
    return new


def _fast_clone_tag_state(ts):
    """``TagState.clone()`` without deepcopy (TagInstance is a flat dataclass)."""
    import copy as _copy
    from dataclasses import replace as _replace
    new = _copy.copy(ts)
    new.tags = [_replace(t) for t in ts.tags]
    return new


_FRONT_RANK = {2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10: "T",
               11: "J", 12: "Q", 13: "K", 14: "A"}


def _front_rank(rank: int) -> str:
    return _FRONT_RANK.get(rank, str(rank))


class _GameTagContext(_tags.TagContext):
    """``TagContext`` over a ``BalatroGame``.  Read-only fields are live properties; the hooks
    the shop-time tags would use (``create_shop_joker`` / ``store_joker_modify`` family) are
    implemented but unused — under option B (TAGS_NOTES §2) ``generate.generate_shop`` applies
    those tags itself, see ``BalatroGame._absorb_tag_triggers``."""

    def __init__(self, game: "BalatroGame", last_blind_was_boss: Optional[bool] = None):
        self.game = game
        self._last_boss = last_blind_was_boss

    # -- fields (G.GAME.*) ----------------------------------------------------------------
    @property
    def dollars(self) -> int:
        return self.game.dollars

    @property
    def skips(self) -> int:
        return self.game.skips

    @property
    def hands_played(self) -> int:           # G.GAME.hands_played: run total (state_events.lua:523)
        return sum(self.game._hand_type_counts.values())

    @property
    def unused_discards(self) -> int:
        return self.game.unused_discards

    @property
    def last_blind_was_boss(self) -> bool:
        if self._last_boss is not None:
            return self._last_boss
        return self.game.current_blind.is_boss

    # -- money ---------------------------------------------------------------------------
    def add_dollars(self, amount: int, source: str) -> None:
        self.game.dollars += amount

    # -- jokers --------------------------------------------------------------------------
    def joker_slots_free(self) -> int:
        return self.game.joker_slots - len(self.game.jokers)

    def spawn_joker_to_slots(self, rarity: float, key_append: str):
        g = self.game
        g._sync_run_state()
        c = _gen.create_card(g.run_state, "Joker", area="jokers", rarity=rarity, key_append=key_append)
        j = JokerInstance(c.key, EDITION_FROM_GEN.get(c.edition, "None"))
        return emplace_joker(g, j)

    def create_shop_joker(self, rarity: float, key_append: str):
        g = self.game
        g._sync_run_state()
        c = _gen.create_card(g.run_state, "Joker", area="shop", rarity=rarity, key_append=key_append)
        from .shop import shop_item_from_gen
        return shop_item_from_gen(c)

    def rare_joker_available(self) -> bool:
        owned_rares = {j.key for j in self.game.jokers if _gk.JOKER_RARITY.get(j.key) == "Rare"}
        return len(_gk.pools.JOKER_POOL_RARITY_3) > len(owned_rares)

    def card_is_editionless_joker(self, card) -> bool:
        return getattr(card, "kind", "") == "joker" and getattr(card, "edition", "None") in (None, "None")

    def set_card_edition(self, card, edition: str) -> None:
        card.edition = EDITION_FROM_GEN.get(edition, edition)

    def mark_card_couponed(self, card) -> None:
        card.couponed = True

    # -- hands ---------------------------------------------------------------------------
    def level_up_hand(self, hand: str, levels: int) -> None:
        g = self.game
        g.planet_levels[hand] = g.planet_levels.get(hand, 1) + levels

    def choose_orbital_hand(self, blind_type: Optional[str]) -> str:
        g = self.game
        key = (g.ante, blind_type)
        if key not in g._orbital_choices:
            g._orbital_choices[key] = _gen.orbital_hand(g.run_state, g._visible_hands())
        return g._orbital_choices[key]

    def change_hand_size(self, delta: int) -> None:
        self.game.hand_size = max(1, self.game.hand_size + delta)

    # -- blind select --------------------------------------------------------------------
    def open_pack(self, pack_key: str) -> None:
        self.game._open_booster(pack_key, free=True, return_state=State.BLIND_SELECT)

    def reroll_boss(self) -> None:
        self.game._reroll_boss(from_tag=True)

    # -- shop ----------------------------------------------------------------------------
    def add_shop_voucher(self) -> None:     # unused under option B (generate handles tag_voucher)
        g = self.game
        from .shop import voucher_item
        g._sync_run_state()
        vk = _gen.next_voucher(g.run_state, from_tag=True)
        g.run_state.shop_voucher_keys.append(vk)
        g.current_shop.append(voucher_item(vk, from_tag=True))

    def set_temp_reroll_cost(self, cost: int) -> None:
        self.game.reroll_cost = cost

    def clear_temp_reroll_cost(self) -> None:
        self.game.reroll_cost = 5

    def make_shop_free(self) -> None:
        for it in self.game.current_shop:
            if it.kind != "voucher":
                it.couponed = True
