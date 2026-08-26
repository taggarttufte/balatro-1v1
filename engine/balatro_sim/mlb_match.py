"""
mlb_match.py — Major League Balatro: two ``BalatroGame``s on ONE seed in ante lockstep.

Phase 2 W1 (2026-08-21).  Rules are ported from the installed BalatroMultiplayer mod
(v0.5.2, ``$MOD``) and its server (``BalatroMultiplayerAPI-Server/src/actionHandlers.ts``,
``src/Client.ts``); every rule's source line is listed in ``engine/MLB_NOTES.md``.
Nothing here copies mod code.

The split of responsibilities mirrors the real system:

* ``BalatroGame(ruleset="mlb")`` is the CLIENT: bans, Nemesis blind at the Boss slot from
  ante ``pvp_start_round``, lives / comeback counters, failed-blind-proceeds, Cash Out
  money, ``PVP_WAIT`` when out of hands at the Nemesis, ``lose_life`` / ``end_pvp`` /
  ``set_pvp_info`` entry points (the ``playerInfo`` / ``endPvP`` / ``enemyInfo`` messages).
* ``MLBMatch`` is the SERVER: it starts the Nemesis blind when both players are ready
  (``readyBlind`` -> ``startBlind``), relays live scores (``enemyInfo``), applies the PvP
  end rule after every hand (``playHand`` handler), takes lives, and ends the match at 0
  lives (``winGame`` / ``loseGame``).

Lockstep: Small and Big (and the ante-1 vanilla Boss) are played independently -- each
player has their own shops, rerolls and skips on the shared seed.  The only
synchronisation point is the Nemesis blind: ``play_blind`` there marks the player ready;
the blind starts for BOTH when both are ready; a ready player has no legal actions until
then.  Nobody can get more than one Nemesis ahead, because the Nemesis can't be skipped
and only the match can end it.

Step API (usable by an env and by a scripted driver):

    m = MLBMatch(seed="7I4M53DL")
    while not m.done:
        p = m.current_player()            # canonical turn order (alternation; PvP too)
        a = choose(m.legal_actions(p))    # m.legal_actions(q) for ANY q that can act
        m.step(p, a)
    m.winner  ->  0 | 1

``step(player, action)`` steps that player's game and then ``sync()``s the match; ``sync()``
is idempotent and may also be called after a game was stepped directly (env_mp does that).
``clone()`` composes with ``BalatroGame.clone()`` (MCTS snapshot machinery).
"""
from __future__ import annotations
import secrets
from dataclasses import dataclass
from typing import Optional

from .constants import MLB_STARTING_LIVES, MLB_PVP_START_ROUND, MLB_COMEBACK_PER_LIFE
from .game import BalatroGame, State

__all__ = ["MLBMatch", "MLBMatchState", "PlayerView", "PlayerEcon", "DEFAULT_LIVES",
           "COMEBACK_MONEY_PER_LIFE"]

DEFAULT_LIVES = MLB_STARTING_LIVES
COMEBACK_MONEY_PER_LIFE = MLB_COMEBACK_PER_LIFE


@dataclass
class PlayerView:
    """Per-player public state (what the opponent's ``enemyInfo`` / lobby HUD reveals)."""
    ante: int
    blind_idx: int
    state: str
    lives: int
    skips: int
    dollars: int
    chips_scored: int
    hands_left: int
    pvp_ready: bool
    pvp_exhausted: bool
    comeback_bonus: int
    comeback_pending: int      # $ owed at the next Cash Out (0 when nothing is pending)
    # Phase 5 W1 (2026-08-23, additive, defaults keep positional construction working):
    # the rest of what the mod's ``enemyInfo`` / lobby HUD broadcasts about a player.
    hands_played: int = 0      # hands played this round (MP.GAME.enemy.hands is hands LEFT)
    sells_per_ante: int = 0    # MP.GAME.enemy.sells_per_ante — jokers sold this ante
    spent_in_shop: int = 0     # MP.GAME.enemy.spent_in_shop — $ spent on buys/rerolls this ante
    sells_total: int = 0       # run totals of the two above
    spent_total: int = 0
    last_life_loss_ante: Optional[int] = None   # ante of the most recent life lost (any blind)


@dataclass
class PlayerEcon:
    """Per-player public shop economics (Phase 5 W1).  The mod broadcasts
    ``sells_per_ante`` / ``spent_in_shop`` with every ``playerInfo``; the match tracks them
    from the actions it steps (``MLBMatch.step``), so a game stepped directly (env_mp) keeps
    zeros.  ``ante`` is the ante the per-ante counters belong to; they reset when it moves."""
    ante: int = 1
    sells_per_ante: int = 0
    spent_in_shop: int = 0
    sells_total: int = 0
    spent_total: int = 0

    def copy(self) -> "PlayerEcon":
        return PlayerEcon(self.ante, self.sells_per_ante, self.spent_in_shop,
                          self.sells_total, self.spent_total)


@dataclass
class MLBMatchState:
    seed: str
    players: tuple            # (PlayerView, PlayerView)
    pvp_active: bool          # both players are inside a Nemesis blind
    pvp_ante: int             # ante of the Nemesis in progress (0 when none)
    done: bool
    winner: Optional[int]     # 0 / 1 when done, else None
    current_player: Optional[int]
    steps: int


class MLBMatch:
    """Two-player MLB match coordinator (see module docstring)."""

    def __init__(self, seed=None, deck_key: str = "b_red", stake: "int | str" = 1,
                 lives: int = DEFAULT_LIVES, pvp_start_round: int = MLB_PVP_START_ROUND):
        # different_seeds = false (core.lua:173): both games on ONE seed, one deck, one stake.
        g0 = BalatroGame(seed=seed, deck_key=deck_key, stake=stake, ruleset="mlb")
        g1 = BalatroGame(seed=g0.seed_str, deck_key=deck_key, stake=stake, ruleset="mlb")
        self.seed_str = g0.seed_str
        self.games: list[BalatroGame] = [g0, g1]
        for g in self.games:
            g.pvp_solo = False             # the match (server) starts the Nemesis
            g.lives = lives
            g.pvp_start_round = pvp_start_round   # read when the Boss slot is prepared
        self.starting_lives = lives
        self.pvp_start_round = pvp_start_round
        self.pvp_active = False
        self.pvp_ante = 0
        self.done = False
        self.winner: Optional[int] = None
        self.steps = 0
        self._turn = 0                     # alternation pointer for current_player()
        self.pvp_log: list = []            # (ante, loser|None, score0, score1) per resolved Nemesis
        # Phase 5 W1 (additive): the SAME Nemeses with the rest of the public record —
        # (ante, loser|None, score0, score1, hands_played0, hands_played1, early_end).
        # `pvp_log`'s 4-tuple is unpacked as exactly four by frozen readers
        # (replay/replay.py, eval/common.py), so it is left untouched.
        self.pvp_detail: list = []
        self.econ: list = [PlayerEcon(), PlayerEcon()]   # public shop economics per player
        self._lives_seen: list = [lives, lives]          # for last_life_loss_ante (sync)
        self.last_life_loss_ante: list = [None, None]    # per player; a failed Small/Big counts too
        self.sync()

    # ── queries ──────────────────────────────────────────────────────────────

    def game(self, player: int) -> BalatroGame:
        return self.games[player]

    @staticmethod
    def _other(player: int) -> int:
        return 1 - player

    def _in_pvp(self, g: BalatroGame) -> bool:
        return g.pvp_started and g.current_blind.is_pvp and g.state in (State.SELECTING_HAND, State.PVP_WAIT)

    def can_act(self, player: int) -> bool:
        return not self.done and bool(self.legal_actions(player))

    def legal_actions(self, player: int) -> list[dict]:
        """The player's game's legal actions, minus what the match forbids: nothing once the
        match is over, nothing while readied for / waiting at the Nemesis (``PVP_WAIT``,
        ``pvp_ready`` -> the game already returns [])."""
        if self.done:
            return []
        return self.games[player].legal_actions()

    def actors(self) -> list[int]:
        """Every player who can act right now (both, in the independent phases)."""
        return [p for p in (0, 1) if self.can_act(p)]

    def current_player(self) -> Optional[int]:
        """Canonical turn order: players alternate action by action (``_turn`` flips after
        every ``step``); a player with no legal actions is skipped.  In the real game both
        play in real time -- the outcome of a Nemesis does not depend on the interleaving
        (MLB_NOTES.md §3), only the early-end cut point does, and this makes it
        deterministic for MCTS / replay."""
        if self.done:
            return None
        for p in (self._turn, self._other(self._turn)):
            if self.can_act(p):
                return p
        return None

    def player_view(self, player: int) -> PlayerView:
        """The PUBLIC view of one player (what ``enemyInfo`` / the lobby HUD reveal).  This
        is the only engine-side reader the opponent-modelling features are allowed to use
        (Phase 5 W1, `mcts/encoder_v2.opponent_view`): nothing here touches the player's
        hand, jokers, consumables, deck or shop."""
        g = self.games[player]
        e = self.econ[player]
        return PlayerView(
            ante=g.ante, blind_idx=g.blind_idx, state=g.state.name, lives=g.lives,
            skips=g.skips, dollars=g.dollars, chips_scored=g.chips_scored,
            hands_left=g.hands_left, pvp_ready=g.pvp_ready,
            pvp_exhausted=(g.state == State.PVP_WAIT),
            comeback_bonus=g.comeback_bonus,
            comeback_pending=(0 if g.comeback_bonus_given else COMEBACK_MONEY_PER_LIFE * g.comeback_bonus),
            hands_played=g._hands_played_round,
            sells_per_ante=(e.sells_per_ante if e.ante == g.ante else 0),
            spent_in_shop=(e.spent_in_shop if e.ante == g.ante else 0),
            sells_total=e.sells_total, spent_total=e.spent_total,
            last_life_loss_ante=self.last_life_loss_ante[player],
        )

    def state(self) -> MLBMatchState:
        views = [self.player_view(0), self.player_view(1)]
        return MLBMatchState(seed=self.seed_str, players=(views[0], views[1]),
                             pvp_active=self.pvp_active, pvp_ante=self.pvp_ante, done=self.done,
                             winner=self.winner, current_player=self.current_player(), steps=self.steps)

    def signature(self) -> tuple:
        """Hashable snapshot of the whole match (both games' ``state_signature`` + the
        match scalars) -- clone-fidelity / determinism tests."""
        return (self.games[0].state_signature(), self.games[1].state_signature(),
                self.pvp_active, self.pvp_ante, self.done, self.winner, self._turn, tuple(self.pvp_log))

    # ── stepping ─────────────────────────────────────────────────────────────

    def step(self, player: int, action: dict) -> MLBMatchState:
        """Apply ``action`` for ``player`` (any player who can act; ``current_player()`` is
        the canonical one), then run the server-side bookkeeping.  Illegal / no-op actions
        are ignored exactly as ``BalatroGame.step`` ignores them."""
        if self.done:
            return self.state()
        g = self.games[player]
        if g.state not in (State.PVP_WAIT, State.GAME_OVER):
            dollars_before = g.dollars
            g.step(action)
            self._track_econ(player, action, dollars_before - g.dollars)
        self.steps += 1
        self._turn = self._other(player)
        self.sync()
        return self.state()

    def _track_econ(self, player: int, action, spent: int) -> None:
        """Public shop economics (Phase 5 W1): a sold joker bumps ``sells_per_ante``; $ that
        left the wallet on a ``buy`` / ``reroll`` goes to ``spent_in_shop``.  The per-ante
        counters belong to the game's current ante and reset when it moves."""
        e = self.econ[player]
        ante = self.games[player].ante
        if e.ante != ante:
            e.ante, e.sells_per_ante, e.spent_in_shop = ante, 0, 0
        kind = action.get("type") if isinstance(action, dict) else None
        if kind == "sell_joker":
            e.sells_per_ante += 1
            e.sells_total += 1
        elif kind in ("buy", "reroll") and spent > 0:
            e.spent_in_shop += spent
            e.spent_total += spent

    def sync(self) -> None:
        """Server-side rules, idempotent: startBlind when both are ready, enemyInfo relay,
        the ``playHand`` end-of-PvP check, ``winGame`` / ``loseGame``."""
        if self.done:
            return
        g0, g1 = self.games
        # 1. readyBlind x2 -> startBlind (actionHandlers.ts:173-215): both Nemesis blinds
        #    start together; scores / hands are reset by each game's _start_blind.
        if g0.pvp_ready and g1.pvp_ready:
            assert g0.ante == g1.ante, "lockstep broken: players readied different antes"
            for g in self.games:
                g._start_blind()
            self.pvp_active = True
            self.pvp_ante = g0.ante
        # 2. enemyInfo relay + 3. the end check, while both are in the Nemesis
        if self._in_pvp(g0) and self._in_pvp(g1):
            self.pvp_active = True
            g0.set_pvp_info(g1.chips_scored, g1.hands_left)
            g1.set_pvp_info(g0.chips_scored, g0.hands_left)
            self._resolve_pvp()
        elif self.pvp_active and not (self._in_pvp(g0) or self._in_pvp(g1)):
            self.pvp_active = False
        # 4. 0 lives anywhere -> the match is over (failRound / playHand handlers)
        self._check_game_over()
        self._track_lives()

    def _track_lives(self) -> None:
        """Phase 5 W1 (additive): remember the ante at which each player last lost a life
        (a failed Small / Big costs one too, not only a Nemesis) — public, it is the lives
        counter on the HUD moving."""
        for p, g in enumerate(self.games):
            if g.lives < self._lives_seen[p]:
                self.last_life_loss_ante[p] = g.ante
            self._lives_seen[p] = g.lives

    def _resolve_pvp(self) -> None:
        """``playHandAction`` (actionHandlers.ts:221-345), evaluated on the live state: the
        PvP ends when a player is out of hands AND strictly behind, or when both are out of
        hands.  Strictly-lower score loses a life; an exact tie loses nobody (both get
        ``endPvP{lost=false}``).  A player's remaining hands are forfeited on an early end."""
        g0, g1 = self.games
        ex0 = g0.state == State.PVP_WAIT or g0.hands_left < 1
        ex1 = g1.state == State.PVP_WAIT or g1.hands_left < 1
        s0, s1 = g0.chips_scored, g1.chips_scored
        if not ((ex0 and s0 < s1) or (ex1 and s1 < s0) or (ex0 and ex1)):
            return
        loser: Optional[int] = None
        if s0 != s1:
            loser = 0 if s0 < s1 else 1
            self.games[loser].lose_life()          # Client.loseLife -> playerInfo (comeback bump)
        self.pvp_log.append((self.pvp_ante, loser, s0, s1))
        self.pvp_detail.append((self.pvp_ante, loser, s0, s1,
                                g0._hands_played_round, g1._hands_played_round,
                                not (ex0 and ex1)))        # early_end: a hand was forfeited
        for g in self.games:
            g.end_pvp()                            # endPvP -> Cash Out (or GAME_OVER at 0 lives)
        self.pvp_active = False

    def _check_game_over(self) -> None:
        for p, g in enumerate(self.games):
            if g.lives <= 0:
                self.done = True
                self.winner = self._other(p)
                g.state = State.GAME_OVER                   # loseGame
                w = self.games[self.winner]
                w.match_won = True                          # winGame -> win_game()
                w.state = State.GAME_OVER
                return

    # ── cloning ──────────────────────────────────────────────────────────────

    def clone(self) -> "MLBMatch":
        new = MLBMatch.__new__(MLBMatch)
        new.seed_str = self.seed_str
        new.games = [g.clone() for g in self.games]
        new.starting_lives = self.starting_lives
        new.pvp_start_round = self.pvp_start_round
        new.pvp_active = self.pvp_active
        new.pvp_ante = self.pvp_ante
        new.done = self.done
        new.winner = self.winner
        new.steps = self.steps
        new._turn = self._turn
        new.pvp_log = list(self.pvp_log)
        new.pvp_detail = list(self.pvp_detail)
        new.econ = [e.copy() for e in self.econ]
        new._lives_seen = list(self._lives_seen)
        new.last_life_loss_ante = list(self.last_life_loss_ante)
        return new

    def clone_determinized(self, seed=None) -> "MLBMatch":
        """Phase 5 W2 (DETERMINIZE_NOTES.md): both games determinized with the SAME fresh
        seed — real-game behaviour is one seed shared by both players (``different_seeds =
        false``, see ``__init__``/module docstring), and that correlation must survive
        determinization: the sampled WORLD is "what if the run had actually started on
        this other seed", not two independent worlds. `seed` is resolved to a concrete
        value ONCE (``secrets``-random when ``None``) and handed to both
        ``BalatroGame.clone_determinized`` calls, so both draw-pile reshuffles and both
        future keyed-RNG streams derive from the identical fresh seed string — exactly
        `MLBMatch.__init__`'s own ``g1 = BalatroGame(seed=g0.seed_str, ...)`` pattern, just
        with a resampled seed instead of the true one. Never touches ``self`` or
        ``self.games`` (same independence contract as ``clone()``); ``pvp_log`` is copied,
        never mutated."""
        if seed is None:
            seed = secrets.randbelow(1 << 40)
        new = MLBMatch.__new__(MLBMatch)
        new.seed_str = self.seed_str
        new.games = [g.clone_determinized(seed) for g in self.games]
        new.starting_lives = self.starting_lives
        new.pvp_start_round = self.pvp_start_round
        new.pvp_active = self.pvp_active
        new.pvp_ante = self.pvp_ante
        new.done = self.done
        new.winner = self.winner
        new.steps = self.steps
        new._turn = self._turn
        new.pvp_log = list(self.pvp_log)
        new.pvp_detail = list(self.pvp_detail)
        new.econ = [e.copy() for e in self.econ]
        new._lives_seen = list(self._lives_seen)
        new.last_life_loss_ante = list(self.last_life_loss_ante)
        return new

    # ── convenience driver ───────────────────────────────────────────────────

    def play_out(self, policies, max_steps: int = 100_000) -> MLBMatchState:
        """Drive the match with ``policies[p](match, player, legal_actions) -> action`` until
        it ends or ``max_steps`` is hit.  Uses ``current_player()``; a deterministic policy
        on a fixed seed gives a reproducible match."""
        while not self.done and self.steps < max_steps:
            p = self.current_player()
            if p is None:
                raise RuntimeError(f"MLBMatch wedged: nobody can act ({self.state()})")
            acts = self.legal_actions(p)
            self.step(p, policies[p](self, p, acts))
        return self.state()
