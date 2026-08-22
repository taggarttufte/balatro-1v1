"""
mlb_match.py — Major League Balatro: two ``BalatroGame``s on ONE seed in ante lockstep.

Phase 2 W1 (2026-08-21).  Rules are ported from the installed BalatroMultiplayer mod
(v0.5.2, ``$MOD``) and its server (``BalatroMultiplayerAPI-Server/src/actionHandlers.ts``,
``src/Client.ts``); every rule's source line is listed in ``mp/engine/MLB_NOTES.md``.
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
from dataclasses import dataclass
from typing import Optional

from .constants import MLB_STARTING_LIVES, MLB_PVP_START_ROUND, MLB_COMEBACK_PER_LIFE
from .game import BalatroGame, State

__all__ = ["MLBMatch", "MLBMatchState", "PlayerView", "DEFAULT_LIVES", "COMEBACK_MONEY_PER_LIFE"]

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

    def state(self) -> MLBMatchState:
        views = []
        for g in self.games:
            views.append(PlayerView(
                ante=g.ante, blind_idx=g.blind_idx, state=g.state.name, lives=g.lives,
                skips=g.skips, dollars=g.dollars, chips_scored=g.chips_scored,
                hands_left=g.hands_left, pvp_ready=g.pvp_ready,
                pvp_exhausted=(g.state == State.PVP_WAIT),
                comeback_bonus=g.comeback_bonus,
                comeback_pending=(0 if g.comeback_bonus_given else COMEBACK_MONEY_PER_LIFE * g.comeback_bonus),
            ))
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
            g.step(action)
        self.steps += 1
        self._turn = self._other(player)
        self.sync()
        return self.state()

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
