"""test_mod_lua.py — execute the REAL GhostRace main.lua in LuaJIT (lupa) against a
stubbed game environment, then close the loop against the REAL Python sidecar over
REAL IPC files.  This is the whole G2 protocol end-to-end, minus only Balatro itself —
the same execute-the-actual-Lua technique Phase 2 used on the MP mod's patches.

The stubs implement exactly the surface the mod-side recon documented (G2_DESIGN.md §5):
MP.GHOST originals, MP.GAME/LOBBY state, NFS (bridged to real Python file IO), love.timer
(a controllable clock), to_big, win_game.
"""
from __future__ import annotations

import json as pyjson
from pathlib import Path

import pytest

lupa = pytest.importorskip("lupa")

from ..ipc import append_event, iter_session
from ..live import Sidecar, launcher_doc

MAIN_LUA = Path(__file__).resolve().parents[1] / "mod" / "GhostRace" / "main.lua"

PRELUDE = r"""
MP = { GHOST = {}, GAME = {}, UI = {},
       LOBBY = { config = { gold_on_life_loss = true, no_gold_on_round_loss = false,
                            death_on_round_loss = true, starting_lives = 4 } } }
MP.GHOST.active = false; MP.GHOST.replay = nil
MP.GHOST._hands = {}; MP.GHOST._hand_idx = 0; MP.GHOST._advancing = false
function MP.GHOST.load(replay)
  MP.GHOST.active = true; MP.GHOST.replay = replay
  MP.GHOST._hands = {}; MP.GHOST._hand_idx = 0; MP.GHOST._advancing = false
end
function MP.GHOST.clear()
  MP.GHOST.active = false; MP.GHOST.replay = nil
  MP.GHOST._hands = {}; MP.GHOST._hand_idx = 0
end
function MP.GHOST.is_active() return MP.GHOST.active and MP.GHOST.replay ~= nil end
function MP.GHOST.get_enemy_hands(ante) return {} end
function MP.GHOST.init_playback(ante)
  local hands = MP.GHOST.get_enemy_hands(ante)   -- dynamic: resolves the WRAPPED one
  MP.GHOST._hands = hands; MP.GHOST._hand_idx = 0
  if #hands > 0 then
    MP.GHOST._hand_idx = 1
    MP.GAME.enemy.score = hands[1].score
    MP.GAME.enemy.hands = hands[1].hands_left or 0
    return true
  end
  return false
end
function MP.GHOST.advance_hand()
  if MP.GHOST._hand_idx >= #MP.GHOST._hands then return false end
  MP.GHOST._hand_idx = MP.GHOST._hand_idx + 1
  local e = MP.GHOST._hands[MP.GHOST._hand_idx]
  MP.GAME.enemy.score = e.score; MP.GAME.enemy.hands = e.hands_left or 0
  return true
end
function MP.GHOST.resolve_pvp_mid_hand(chips) return false end
function MP.GHOST.resolve_pvp_hands_exhausted(chips) return "continue" end
function MP.GHOST.resolve_round_fail()
  if MP.LOBBY.config.death_on_round_loss then
    MP.GAME.lives = MP.GAME.lives - 1
    if MP.GAME.lives <= 0 then return "game_over" end
  end
  return nil
end
function MP.GHOST._start_advance_sequence() MP.GHOST._advancing = true end
function MP.is_pvp_boss() return (G.GAME.blind and G.GAME.blind.pvp) or false end
MP.UI.ease_lives = function(mod) EASE_LIVES = (EASE_LIVES or 0) + 1 end
function MP.reset_game_states()
  MP.GAME = { lives = 4, enemy = { lives = 4, score = "0", hands = 4 },
              end_pvp = false, won = false,
              comeback_bonus = 0, comeback_bonus_given = true }
end
MP.reset_game_states()

function to_big(x) return tonumber(x) end
WIN_GAME_CALLED = 0
function win_game() WIN_GAME_CALLED = WIN_GAME_CALLED + 1 end
WARNINGS = 0
function sendDebugMessage(m, t) end
function sendWarnMessage(m, t) WARNINGS = WARNINGS + 1; LAST_WARN = m end

G = { STATE = 1, STATE_COMPLETE = true,
      STATES = { SELECTING_HAND = 1, DRAW_TO_HAND = 2, HAND_PLAYED = 3,
                 NEW_ROUND = 4, GAME_OVER = 5, ROUND_EVAL = 6 },
      STAGE = 1, STAGES = { RUN = 1 },
      GAME = { round_resets = { ante = 2 },
               current_round = { hands_left = 4 },
               blind = { pvp = true, dollars = 5 }, chips = 0 } }
Game = {}
function Game.update(self, dt) end

NFS = { getInfo = function(p) return __py_nfs_getinfo(p) end,
        read = function(p) return __py_nfs_read(p) end,
        append = function(p, d) return __py_nfs_append(p, d) end }
love = { timer = { getTime = function() return __py_time() end } }
function require(name)
  if name == "json" then
    return { encode = function(t) return __py_json_encode(t) end,
             decode = function(s) return __py_json_decode(s) end }
  end
  error("unexpected require: " .. tostring(name))
end
"""


# ─────────────────────────────────────────────────── python<->lua bridging

def to_lua(lua, obj):
    if isinstance(obj, dict):
        t = lua.table()
        for k, v in obj.items():
            t[k] = to_lua(lua, v)
        return t
    if isinstance(obj, list):
        t = lua.table()
        for i, v in enumerate(obj, 1):
            t[i] = to_lua(lua, v)
        return t
    return obj


def from_lua(obj):
    if lupa.lua_type(obj) == "table":
        keys = list(obj.keys())
        if keys and all(isinstance(k, int) for k in keys):
            return [from_lua(obj[i]) for i in range(1, len(keys) + 1)]
        return {k: from_lua(obj[k]) for k in keys}
    return obj


class LuaMod:
    """The real main.lua running in LuaJIT over the stub environment."""

    def __init__(self):
        self.lua = lupa.LuaRuntime(unpack_returned_tuples=True)
        self.clock = 0.0
        g = self.lua.globals()
        g["__py_time"] = lambda: self.clock
        g["__py_json_encode"] = lambda t: pyjson.dumps(from_lua(t))
        g["__py_json_decode"] = lambda s: to_lua(self.lua, pyjson.loads(s))
        g["__py_nfs_getinfo"] = self._nfs_getinfo
        g["__py_nfs_read"] = self._nfs_read
        g["__py_nfs_append"] = self._nfs_append
        self.lua.execute(PRELUDE)
        self.lua.execute(MAIN_LUA.read_text(encoding="utf-8"))
        self.g = g

    def _nfs_getinfo(self, path):
        p = Path(path)
        if not p.exists():
            return None
        return to_lua(self.lua, {"type": "file", "size": p.stat().st_size})

    def _nfs_read(self, path):
        p = Path(path)
        return p.read_text(encoding="utf-8") if p.exists() else None

    def _nfs_append(self, path, data):
        with open(path, "a", encoding="utf-8") as f:
            f.write(data)
        return True

    # ── driving helpers ───────────────────────────────────────────────────────

    def load_replay(self, doc: dict):
        self.lua.globals().MP.GHOST.load(to_lua(self.lua, doc))

    def tick(self, dt: float = 0.4):
        self.clock += dt
        game = self.lua.globals().Game
        game.update(game, dt)

    def eval(self, expr: str):
        return from_lua(self.lua.eval(expr))

    def set(self, stmt: str):
        self.lua.execute(stmt)

    def mid_hand(self, chips, hands_left):
        self.set(f"G.GAME.current_round.hands_left = {hands_left}")
        return self.eval(f"MP.GHOST.resolve_pvp_mid_hand({chips})")

    def exhausted(self, chips):
        self.set("G.GAME.current_round.hands_left = 0")
        return self.eval(f"MP.GHOST.resolve_pvp_hands_exhausted({chips})")


def _launcher(outbox, inbox, bootstrap=None):
    return launcher_doc("TESTSEED", "b_red", 1, "ev:fast", str(outbox), str(inbox),
                        bootstrap=bootstrap)


@pytest.fixture()
def rig(tmp_path):
    outbox, inbox = tmp_path / "outbox.jsonl", tmp_path / "inbox.jsonl"
    outbox.touch()
    return LuaMod(), outbox, inbox


BOOT = {"ante": 2, "hands": [{"score": 100, "hands_left": 3},
                             {"score": 250, "hands_left": 2},
                             {"score": 400, "hands_left": 0}], "final": 400}


# ─────────────────────────────────────────────────────── scripted scenarios

def test_load_arms_and_emits_session_start(rig):
    mod, outbox, inbox = rig
    mod.load_replay(_launcher(outbox, inbox, BOOT))
    assert mod.eval("GR.live") is True
    ev = list(iter_session(str(outbox)))
    assert ev[0]["e"] == "session_start" and ev[0]["seed"] == "TESTSEED"
    assert mod.eval("GR.buffer[2].final") == 400


def test_nemesis_flow_human_wins_round(rig):
    mod, outbox, inbox = rig
    mod.load_replay(_launcher(outbox, inbox, BOOT))
    mod.tick()                                          # detects the Nemesis, inits
    assert mod.eval("MP.GHOST._hand_idx") == 1
    assert mod.eval("MP.GAME.enemy.score") == "100"     # first hand displayed

    assert mod.mid_hand(90, 3) is False                 # behind: race continues
    events = list(iter_session(str(outbox)))
    assert [e["e"] for e in events][:3] == ["session_start", "nemesis_start", "pvp_hand"]

    mod.tick(3.0)                                       # paced reveal advances
    assert mod.eval("MP.GHOST._hand_idx") == 2
    mod.tick(3.0)
    assert mod.eval("MP.GHOST._hand_idx") == 3          # fully revealed

    # the cut: agent exhausted (revealed) and strictly behind mid-hand
    assert mod.mid_hand(500, 2) is True
    assert mod.eval("MP.GAME.enemy.lives") == 3
    assert mod.eval("MP.GAME.end_pvp") is True
    res = [e for e in iter_session(str(outbox)) if e["e"] == "pvp_result"][-1]
    assert res["loser"] == "agent" and res["human_lives"] == 4
    assert float(res["human_score"]) == 500 and float(res["agent_score"]) == 400


def test_exhausted_loss_tie_and_match_end(rig):
    mod, outbox, inbox = rig
    mod.load_replay(_launcher(outbox, inbox, BOOT))
    mod.tick()

    # human exhausts BELOW the agent's final: loses a life + comeback bookkeeping
    assert mod.exhausted(399) == "continue"
    assert mod.eval("MP.GAME.lives") == 3
    assert mod.eval("MP.GAME.comeback_bonus") == 1
    assert mod.eval("MP.GAME.comeback_bonus_given") is False
    res = [e for e in iter_session(str(outbox)) if e["e"] == "pvp_result"][-1]
    assert res["loser"] == "human" and res["human_lives"] == 3

    # exact tie: nobody loses (server rule)
    mod.set("MP.GAME.end_pvp = false")
    assert mod.exhausted(400) == "continue"
    assert mod.eval("MP.GAME.lives") == 3 and mod.eval("MP.GAME.enemy.lives") == 4
    res = [e for e in iter_session(str(outbox)) if e["e"] == "pvp_result"][-1]
    assert "loser" not in res or res["loser"] is None

    # grind the human to 0: the last loss returns game_over + emits match_end
    mod.set("MP.GAME.lives = 1")
    assert mod.exhausted(1) == "game_over"
    ends = [e for e in iter_session(str(outbox)) if e["e"] == "match_end"]
    assert ends and ends[-1]["winner"] == "agent"


def test_agent_death_via_inbox_wins_the_match(rig):
    mod, outbox, inbox = rig
    mod.load_replay(_launcher(outbox, inbox, BOOT))
    append_event(str(inbox), "agent_state", ante=3, lives=1, money=10)
    mod.tick()
    assert mod.eval("MP.GAME.enemy.lives") == 1
    append_event(str(inbox), "agent_dead", ante=4)
    mod.tick()
    assert mod.eval("MP.GAME.won") is True
    assert mod.eval("WIN_GAME_CALLED") == 1
    ends = [e for e in iter_session(str(outbox)) if e["e"] == "match_end"]
    assert ends[-1]["winner"] == "human"


def test_round_fail_wrap_reports(rig):
    mod, outbox, inbox = rig
    mod.load_replay(_launcher(outbox, inbox, BOOT))
    assert mod.eval("MP.GHOST.resolve_round_fail()") is None
    ev = [e for e in iter_session(str(outbox)) if e["e"] == "round_fail"]
    assert ev and ev[-1]["lives"] == 3


def test_replay_mode_untouched(rig):
    mod, outbox, inbox = rig
    doc = _launcher(outbox, inbox, BOOT)
    doc.pop("_live")                                    # a plain G1 replay
    mod.load_replay(doc)
    assert mod.eval("GR.live") is False
    assert mod.eval("MP.GHOST.resolve_pvp_hands_exhausted(999)") == "continue"
    assert not list(iter_session(str(outbox)))          # nothing emitted


# ────────────────────────────── the real loop: Lua mod <-> real sidecar over files

def test_lua_mod_against_real_sidecar(tmp_path):
    outbox, inbox = tmp_path / "outbox.jsonl", tmp_path / "inbox.jsonl"
    outbox.touch()
    car = Sidecar(str(outbox), str(inbox), seed="7I4M53DL", log=lambda *_: None)
    car.start()
    boot = car.pending
    first_ante = boot["ante"]

    mod = LuaMod()
    mod.load_replay(launcher_doc("7I4M53DL", "b_red", 1, "ev:fast",
                                 str(outbox), str(inbox), bootstrap=boot))
    car.pump()                                          # session_start consumed
    mod.set(f"G.GAME.round_resets.ante = {first_ante}")
    mod.tick()
    assert mod.eval("GR.buffer[%d].final" % first_ante) == boot["final"]

    # the human plays the first Nemesis and loses it (one chip short)
    mod.exhausted(boot["final"] - 1)
    car.pump()                                          # sidecar resolves + publishes
    assert car.mirror.pvp_log[-1] == (first_ante, 1, boot["final"], boot["final"] - 1)
    assert car.mirror.opp.lives == 3                    # from the mod's human_lives
    next_ante = car.pending["ante"]
    assert next_ante > first_ante

    # the mod's poller buffers the next round and the agent's lives stay synced
    mod.tick()
    assert mod.eval("GR.buffer[%d] ~= nil" % next_ante) is True
    assert mod.eval("MP.GAME.enemy.lives") == car.mirror.lives == 4

    # next ante: the human WINS the round; both sides agree on the ledger
    mod.set(f"G.GAME.round_resets.ante = {next_ante}")
    mod.set("MP.GAME.end_pvp = false")
    agent_final = car.pending["final"]
    mod.exhausted(agent_final + 100)
    car.pump()
    assert car.mirror.pvp_log[-1] == (next_ante, 0, agent_final, agent_final + 100)
    assert car.mirror.lives == 3
    mod.tick()
    assert mod.eval("MP.GAME.enemy.lives") == 3
    assert not car.done and not mod.eval("GR.match_over")
