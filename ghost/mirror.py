"""
mirror.py — the G2 mirror: ONE solo-MLB ``BalatroGame`` playing the agent's run on the
session seed, with the human as the EXTERNAL opponent.

Mechanism (engine recon 2026-08-27, all cited in G2_DESIGN.md §4):

* ``pvp_solo = False`` — the sidecar IS the server for this game: the Nemesis parks in
  ``State.PVP_WAIT`` (``legal_actions() == []``) until we resolve it, exactly like a
  match-attached game (template: ``agent/tests/test_mlb_agent.py:131-181``).
* ``step({"type": "play_blind"})`` at a Nemesis only sets ``pvp_ready`` — we call
  ``game._start_blind()`` ourselves (what ``MLBMatch.sync`` does, ``mlb_match.py:373-374``).
  ``_start_blind`` ZEROES the opponent fields, so ``set_pvp_info`` is re-applied before
  EVERY decision (footgun documented in ``agent/TRAIN_NOTES.md:122-125``).
* ``set_pvp_info(score, hands)`` is load-bearing for EVPlayer's Nemesis objective
  (``ev/hand.py:857-858`` reads ``game.pvp_opponent_score/hands``): v1 computes the
  agent's round AHEAD of the human, so it plays against ``(0, opponent_hands_estimate)``
  — the honest state of a simultaneous round's opening.
* Resolution is DRIVER-side, server rule: strict ``<`` loses a life, exact tie takes
  nobody (``mlb_match.py:427-431``); ``lose_life()`` then ``end_pvp()`` (which routes to
  ``ROUND_EVAL`` or ``GAME_OVER``); the comeback payout ($4 x cumulative) happens inside
  the next Cash Out automatically (``game.py:2018-2023``) because ``lose_life`` runs
  before the ``advance``.
* The agent's own regular-blind failures cost lives inside the engine (MLB ruleset);
  the mirror only OBSERVES those via lives deltas.
* The human's lives reach the agent's play through ``EVPlayer.bind_race`` fed a
  match-shaped shim (``ev/player.py:589-615`` wants ``.games[1-p].lives`` +
  ``.pvp_log``) — without it the shop tier's race aggression is neutral.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent            # ghost
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_ROOT / "ev"), str(_ROOT / "eval"), str(_ROOT / "agent"),
           str(_ROOT / "stats")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ghost._bootstrap import BalatroGame, State  # noqa: E402

DEFAULT_OPP_HANDS_ESTIMATE = 4      # Red deck hands/round; only a prior — the human's
                                    # real hands-left arrives with their events


class _OppStub:
    """The opponent's side of the race shim: just the fields bind_race reads."""

    def __init__(self, lives: int, ante: int):
        self.lives = lives
        self.ante = ante


class _RaceShim:
    """Match-shaped duck type for ``EVPlayer.bind_race`` (ev/player.py:589-615):
    ``.games = [agent_game, opponent_stub]`` + ``.pvp_log`` in MLBMatch format
    ``[(ante, loser|None, agent_score, human_score), ...]``."""

    def __init__(self, game, opp: _OppStub):
        self.games = [game, opp]
        self.pvp_log = []


class MirrorDead(RuntimeError):
    """The agent's run ended (0 lives) — the human wins the match."""


class MirrorAgent:
    """The agent's half of the live match.  Drive it with:

        m = MirrorAgent(seed)
        rec = m.advance_to_nemesis()      # plays blinds/shops, then its Nemesis round;
                                          # returns {"ante", "hands": [...], "final"} or
                                          # raises MirrorDead
        ...human plays their round...
        res = m.resolve(human_final)      # server rule; returns the result record
        rec = m.advance_to_nemesis()      # next ante
    """

    MAX_STEPS_PER_PHASE = 2000
    NO_PROGRESS_LIMIT = 40

    def __init__(self, seed: str, spec: str = "ev:fast", deck_key: str = "b_red",
                 stake: int = 1, lives: int = 4, player_seed: int = 0,
                 opp_hands_estimate: int = DEFAULT_OPP_HANDS_ESTIMATE):
        import h2h  # ev/h2h.py — build_player's spec language = the difficulty ladder
        self.game = BalatroGame(seed=seed, deck_key=deck_key, stake=stake, ruleset="mlb")
        self.game.lives = lives
        self.game.pvp_solo = False          # the sidecar is the server for this game
        self.spec = spec
        self.opp_hands_estimate = int(opp_hands_estimate)
        _policy, obj = h2h.build_player(spec, player_seed)
        if obj is None:
            raise ValueError(f"spec {spec!r} gives no .act(game) player object; "
                             "the mirror needs one (use ev:*/real1:* specs)")
        obj.reset()
        self.player = obj
        self.opp = _OppStub(lives=lives, ante=self.game.ante)
        self.shim = _RaceShim(self.game, self.opp)
        self.pvp_log = self.shim.pvp_log    # sidecar-side nemesis log (no engine pvp_log
                                            # on a solo game — recon §6)
        self._pending: Optional[dict] = None   # the un-resolved Nemesis round record
        self._opp_score_now = 0
        self._opp_hands_now = self.opp_hands_estimate
        # the chronicle: everything the agent SAW and DID outside hand play — shops,
        # packs (with contents), buys, rerolls, sells, consumables, skips.  Feeds the
        # post-match report (ghost/report.py); the human never sees the agent's shops
        # in game, and the game-over screen has no data source for them in ghost mode.
        self.chronicle: list = []
        self._last_shop_sig = None
        self._last_pack_sig = None

    # ── observation ────────────────────────────────────────────────────────────

    @property
    def lives(self) -> int:
        return self.game.lives

    @property
    def ante(self) -> int:
        return self.game.ante

    @property
    def money(self) -> int:
        return self.game.dollars

    @property
    def dead(self) -> bool:
        return self.game.state is State.GAME_OVER or self.game.lives <= 0

    @property
    def awaiting_final(self) -> bool:
        return self._pending is not None

    def set_opponent_lives(self, lives: int) -> None:
        self.opp.lives = int(lives)

    # ── internals ──────────────────────────────────────────────────────────────

    def _stuck(self) -> bool:
        """agent/mcts/outcome.py::is_stuck_state, implemented locally to keep ghost/
        torch-free: PVP_WAIT, or BLIND_SELECT with pvp_ready set."""
        g = self.game
        return (g.state is State.PVP_WAIT
                or (g.state is State.BLIND_SELECT and g.pvp_ready))

    def _act(self) -> dict:
        if hasattr(self.player, "bind_race"):
            try:
                self.player.bind_race(self.shim, 0)
            except Exception:
                pass                      # race curve is an enhancement, never a blocker
        return self.player.act(self.game)

    def _note(self, kind: str, **detail) -> None:
        self.chronicle.append({"ante": self.game.ante, "kind": kind, **detail})

    @staticmethod
    def _item_key(item) -> str:
        return getattr(item, "key", None) or str(item)

    def _chronicle_pre(self, action: dict) -> None:
        """Record what the agent sees/does at shops and packs, BEFORE the step applies.
        Purely observational — never touches the game."""
        g = self.game
        at = action.get("type")
        if g.state is State.SHOP:
            sig = tuple((self._item_key(i), bool(i.sold)) for i in g.current_shop)
            if sig != self._last_shop_sig:
                self._last_shop_sig = sig
                self._note("shop", money=g.dollars,
                           items=[{"key": self._item_key(i),
                                   "kind": getattr(i, "kind", "?"),
                                   "sold": bool(i.sold)} for i in g.current_shop])
            if at == "buy":
                idx = action.get("item_idx")
                if isinstance(idx, int) and 0 <= idx < len(g.current_shop):
                    from balatro_sim.shop import effective_price
                    item = g.current_shop[idx]
                    self._note("buy", item=self._item_key(item),
                               kind_bought=getattr(item, "kind", "?"),
                               price=effective_price(g, item), money=g.dollars)
            elif at == "reroll":
                self._note("reroll", money=g.dollars)
            elif at == "sell_joker":
                ji = action.get("joker_idx")
                if isinstance(ji, int) and 0 <= ji < len(g.jokers):
                    self._note("sell", item=g.jokers[ji].key)
            elif at == "use_consumable":
                ci = action.get("consumable_idx", action.get("idx"))
                if isinstance(ci, int) and 0 <= ci < len(g.consumable_hand):
                    self._note("use", item=g.consumable_hand[ci])
        elif g.state is State.BOOSTER_OPEN:
            contents = [self._item_key(c) for c in (g.booster_choices or [])]
            sig = (g.booster_pack_key, tuple(contents))
            if g.booster_pack_key and self._last_pack_sig != sig \
               and (self._last_pack_sig is None
                    or self._last_pack_sig[0] != g.booster_pack_key
                    or len(contents) >= len(self._last_pack_sig[1])):
                self._note("pack_open", pack=g.booster_pack_key, contents=contents)
            self._last_pack_sig = sig
            self._note("pack_action", action=dict(action), remaining=contents)
        elif g.state is State.BLIND_SELECT and at == "skip_blind":
            self._note("skip_blind", blind=getattr(g.current_blind, "kind", "?"))

    def _progress_key(self) -> tuple:
        g = self.game
        return (g.state.name, g.ante, g.dollars, g.chips_scored, g.hands_left,
                g.discards_left, len(g.jokers), len(g.consumable_hand))

    def _step_until(self, done) -> None:
        """Step the agent until ``done(game)`` or a stuck/game-over state, with the
        no-progress guard the recon prescribes for the frozen engine's silent no-op
        corners (a repeated identical no-op action forces an advance)."""
        last_key, stall = None, 0
        for _ in range(self.MAX_STEPS_PER_PHASE):
            g = self.game
            if done(g) or self._stuck() or g.state is State.GAME_OVER:
                return
            key = self._progress_key()
            stall = stall + 1 if key == last_key else 0
            last_key = key
            if g.current_blind.is_pvp and g.state is State.SELECTING_HAND:
                g.set_pvp_info(self._opp_score_now, self._opp_hands_now)
            action = self._act() if stall < self.NO_PROGRESS_LIMIT else {"type": "advance"}
            self._chronicle_pre(action)
            g.step(action)
        raise RuntimeError(
            f"mirror exceeded {self.MAX_STEPS_PER_PHASE} steps in one phase "
            f"(state {self.game.state.name}, ante {self.game.ante})")

    # ── the two public moves ───────────────────────────────────────────────────

    def advance_to_nemesis(self) -> dict:
        """Play regular blinds/shops to the next Nemesis, then play the agent's whole
        round against the injected pre-round opponent state (score 0, estimated hands).
        Returns {"ante", "hands": [{"score", "hands_left"}...], "final"} and leaves the
        game parked in PVP_WAIT until ``resolve()``."""
        if self._pending is not None:
            raise RuntimeError("resolve() the pending Nemesis before advancing")
        if self.dead:
            raise MirrorDead(f"agent dead at ante {self.game.ante}")

        # regular blinds / shops until the Nemesis is offered
        self._step_until(lambda g: g.state is State.BLIND_SELECT
                         and g.current_blind.is_pvp)
        if self.dead:
            raise MirrorDead(f"agent dead at ante {self.game.ante}")

        g = self.game
        nemesis_ante = g.ante
        g.step({"type": "play_blind"})     # -> pvp_ready, legal_actions() == []
        g._start_blind()                    # mlb_match.py:373-374; zeroes opponent info
        self._opp_score_now, self._opp_hands_now = 0, self.opp_hands_estimate

        hands = []
        for _ in range(self.MAX_STEPS_PER_PHASE):
            if g.state is not State.SELECTING_HAND:
                break
            g.set_pvp_info(self._opp_score_now, self._opp_hands_now)
            before = g.hands_left
            g.step(self._act())
            if g.hands_left == before - 1:
                hands.append({"score": int(g.chips_scored),
                              "hands_left": int(g.hands_left)})
        else:
            raise RuntimeError(f"mirror exceeded {self.MAX_STEPS_PER_PHASE} steps "
                               f"inside the ante-{nemesis_ante} Nemesis")
        # deck-out edge: _mlb_check_deck_out may have taken a life already
        self._pending = {"ante": nemesis_ante, "hands": hands,
                         "final": int(g.chips_scored)}
        return dict(self._pending)

    def resolve(self, human_final: int, human_lives: Optional[int] = None) -> dict:
        """``human_lives`` is the human's lives AFTER this round's resolution, as the
        mod reports it (the mod decrements before emitting ``pvp_result``); omitted, the
        mirror derives the loss itself.

        Server rule: strictly smaller final loses a life; exact tie takes nobody.
        Applies the life BEFORE the cash-out advance so the engine pays comeback money
        in the same Cash Out, then plays through ROUND_EVAL.  Returns the result record
        (also appended to ``self.pvp_log`` in MLBMatch format, agent = seat 0)."""
        if self._pending is None:
            raise RuntimeError("no pending Nemesis to resolve")
        pend, self._pending = self._pending, None
        g = self.game
        # The mod sends tostring(chips); with Talisman installed that can be a
        # comma-grouped Big ("1,073") or scientific notation — normalise all of them.
        human_final = int(float(str(human_final).replace(",", "").strip()))
        if human_lives is not None:
            self.set_opponent_lives(human_lives)

        g.set_pvp_info(human_final, 0)
        agent_final = pend["final"]
        loser = None
        if agent_final != human_final:
            loser = 0 if agent_final < human_final else 1
        if loser == 0:
            g.lose_life()                   # may return False after a deck-out life
        g.end_pvp()                          # -> ROUND_EVAL, or GAME_OVER at 0 lives
        self.pvp_log.append((pend["ante"], loser, agent_final, human_final))
        if loser == 1 and human_lives is None:
            # no reported value to trust: derive the human's life loss ourselves
            self.opp.lives = max(0, self.opp.lives - 1)

        if g.state is State.ROUND_EVAL:
            g.step({"type": "advance"})      # cash out: $5 Nemesis + interest + comeback
        result = {"ante": pend["ante"], "agent_final": agent_final,
                  "human_final": human_final,
                  "loser": {0: "agent", 1: "human", None: None}[loser],
                  "agent_lives": g.lives, "money": g.dollars}
        return result
