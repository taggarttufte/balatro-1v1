"""
live.py — the G2 sidecar: run the agent's mirror beside the real game.

    python -m ghost.live [--seed X] [--spec ev:fast]

Bootstraps a session (G2_DESIGN.md §1): creates the session dir + IPC files, installs
the LIVE launcher ghost into the mod's replays folder, pre-plays the agent to its first
Nemesis, then services the protocol (§2) until the match ends: every ``pvp_result`` from
the mod resolves the mirror's parked Nemesis and advances it to the next; ``round_fail``
keeps the human's lives current (the agent's race calculus reads them); ``agent_nemesis``
messages carry the agent's full round to the mod as soon as each is computed — a full
ante ahead of the human, so the mod never waits.

Crash recovery: the mirror is deterministic in (seed, spec), and the outbox is
append-only — ``--resume <session-dir>`` rebuilds the mirror by replaying the recorded
resolutions and carries on.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent            # ghost
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ghost.export import (  # noqa: E402
    GAMEMODE_KEY, RULESET_KEY, _deck_display_name, mod_replays_dir, slugify, write_ghost,
)
from ghost.ipc import PROTOCOL_VERSION, JsonlTail, append_event, iter_session  # noqa: E402
from ghost.make import resolve_seed  # noqa: E402
from ghost.mirror import MirrorAgent, MirrorDead  # noqa: E402

LIVE_RUNS_DIR = _HERE / "runs" / "live"


def launcher_doc(seed: str, deck_key: str, stake: int, spec: str,
                 outbox: str, inbox: str, player_name: str = "Tagg",
                 bootstrap: dict = None) -> dict:
    """The G1-shaped ghost JSON the picker previews + the ``_live`` marker our Lua mod
    switches on.  ``bootstrap`` (the agent's already-computed first Nemesis round) rides
    along twice: in ``_live.bootstrap`` for the live mod (data before any IPC), and as a
    G1-v2 single-entry snapshot so the file degrades to a partial STATIC ghost if the
    GhostRace mod isn't installed."""
    snapshots = {}
    if bootstrap:
        snapshots[str(bootstrap["ante"])] = {
            "player_score": "0",
            "enemy_score": str(bootstrap["final"]),
            "result": "loss",
            "hands": [{"score": str(bootstrap["final"]), "hands_left": 0,
                       "side": "enemy"}],
        }
    doc = {
        "gamemode": GAMEMODE_KEY,
        "ruleset": RULESET_KEY,
        "seed": seed,
        "deck": _deck_display_name(deck_key),
        "stake": stake,
        "winner": "unknown",
        "timestamp": int(time.time()),      # newest-first: the LIVE entry sorts on top
        "player_name": player_name,
        "nemesis_name": f"{spec} LIVE",
        "ante_snapshots": snapshots,
        "_live": {"protocol": PROTOCOL_VERSION, "outbox": outbox, "inbox": inbox},
    }
    if bootstrap:
        doc["_live"]["bootstrap"] = {"ante": bootstrap["ante"],
                                     "hands": bootstrap["hands"],
                                     "final": bootstrap["final"]}
    return doc


class Sidecar:
    """The protocol loop, IPC-file-driven and free of any CLI/thread machinery so tests
    drive it synchronously: call ``start()`` once, then ``pump()`` per poll tick."""

    def __init__(self, outbox: str, inbox: str, *, seed: str, spec: str = "ev:fast",
                 deck_key: str = "b_red", stake: int = 1, lives: int = 4, log=print):
        self.outbox_path, self.inbox_path = str(outbox), str(inbox)
        self.seed, self.spec = seed, spec
        self.deck_key, self.stake, self.lives = deck_key, stake, lives
        self.log = log
        self.tail = JsonlTail(self.outbox_path, fresh=True)
        self.mirror: MirrorAgent = None
        self.done = False
        self.winner = None

    @property
    def pending(self):
        """The agent's computed-but-unresolved Nemesis round (launcher bootstrap)."""
        return None if self.mirror is None else self.mirror._pending

    def pvp_log_snapshot(self) -> list:
        return [] if self.mirror is None else [tuple(t) for t in self.mirror.pvp_log]

    # ── outbound ──────────────────────────────────────────────────────────────

    def _emit(self, event: str, **fields) -> None:
        append_event(self.inbox_path, event, **fields)

    def _emit_state(self) -> None:
        m = self.mirror
        self._emit("agent_state", ante=m.ante, lives=m.lives, money=m.money)

    def _advance_and_publish(self) -> None:
        """Advance the mirror to its next Nemesis and publish the round; on death the
        human wins the match."""
        try:
            rec = self.mirror.advance_to_nemesis()
        except MirrorDead:
            self._emit("agent_dead", ante=self.mirror.ante)
            self._emit_state()
            self._finish("human")
            return
        self._emit("agent_nemesis", ante=rec["ante"], hands=rec["hands"],
                   final=rec["final"])
        self._emit_state()
        self.log(f"[sidecar] agent ready at ante {rec['ante']}: "
                 f"final {rec['final']} over {len(rec['hands'])} hands "
                 f"(lives {self.mirror.lives}, ${self.mirror.money})")

    def _finish(self, winner: str) -> None:
        self.done, self.winner = True, winner
        self.log(f"[sidecar] match over: {winner} wins")

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        self.mirror = MirrorAgent(self.seed, spec=self.spec, deck_key=self.deck_key,
                                  stake=self.stake, lives=self.lives)
        self._emit("hello", protocol=PROTOCOL_VERSION, spec=self.spec,
                   agent_name=f"{self.spec} LIVE")
        self._advance_and_publish()

    def recover(self) -> None:
        """Rebuild from the outbox: fresh mirror + replay every recorded resolution
        (determinism in (seed, spec) makes this exact), then republish the pending
        round so the mod's buffer is complete even if the inbox was lost."""
        self.mirror = MirrorAgent(self.seed, spec=self.spec, deck_key=self.deck_key,
                                  stake=self.stake, lives=self.lives)
        events = list(iter_session(self.outbox_path))
        self._emit("hello", protocol=PROTOCOL_VERSION, spec=self.spec,
                   agent_name=f"{self.spec} LIVE")
        try:
            self.mirror.advance_to_nemesis()
            for ev in events:
                if ev["e"] == "pvp_result":
                    self.mirror.resolve(ev["human_score"], ev.get("human_lives"))
                    self.mirror.advance_to_nemesis()
                elif ev["e"] == "round_fail":
                    self.mirror.set_opponent_lives(ev["lives"])
                elif ev["e"] == "match_end":
                    self._finish(ev.get("winner", "abandoned"))
                    return
        except MirrorDead:
            self._emit("agent_dead", ante=self.mirror.ante)
            self._finish("human")
            return
        pend = self.mirror._pending
        self._emit("agent_nemesis", ante=pend["ante"], hands=pend["hands"],
                   final=pend["final"])
        self._emit_state()
        self.log(f"[sidecar] recovered: {len(events)} outbox events replayed, "
                 f"agent at ante {pend['ante']}")

    # ── inbound ───────────────────────────────────────────────────────────────

    def pump(self) -> int:
        """One poll: handle every new outbox event.  Returns how many were handled."""
        events = self.tail.poll()
        for ev in events:
            self._handle(ev)
            if self.done:
                break
        return len(events)

    def _handle(self, ev: dict) -> None:
        e = ev.get("e")
        if e == "session_start":
            self.log(f"[sidecar] mod connected (seed {ev.get('seed')})")
            if ev.get("seed") not in (None, self.seed):
                self.log(f"[sidecar] WARNING: mod seed {ev['seed']!r} != "
                         f"session seed {self.seed!r}")
        elif e == "nemesis_start":
            pend = self.mirror._pending
            if pend is None or pend["ante"] != ev.get("ante"):
                self.log(f"[sidecar] WARNING: human at Nemesis {ev.get('ante')} but "
                         f"agent pending is {pend and pend['ante']}")
        elif e == "pvp_hand":
            pass                                    # v1: informational (session.log only)
        elif e == "pvp_result":
            res = self.mirror.resolve(ev["human_score"], ev.get("human_lives"))
            self.log(f"[sidecar] ante {res['ante']} resolved: "
                     f"agent {res['agent_final']} vs human {res['human_final']} -> "
                     f"loser {res['loser']} (agent lives {res['agent_lives']})")
            if self.mirror.dead:
                # confirmation only — the mod's own resolution already decided this
                self._emit("agent_dead", ante=self.mirror.ante)
                self._emit_state()
                self._finish("human")
                return
            self._advance_and_publish()
        elif e == "round_fail":
            self.mirror.set_opponent_lives(ev["lives"])
            if ev["lives"] <= 0:
                self._finish("agent")
        elif e == "match_end":
            self._finish(ev.get("winner", "abandoned"))
        elif e == "_corrupt":
            self.log(f"[sidecar] WARNING: corrupt outbox line: {ev.get('raw')!r}")

    def run(self, interval: float = 0.25, timeout: float = None) -> None:
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self.done:
            self.pump()
            if deadline is not None and time.monotonic() >= deadline:
                self.log("[sidecar] timeout — stopping")
                return
            time.sleep(interval)


# ─────────────────────────────────────────────────────────────────────── CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ghost.live", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", default=None)
    ap.add_argument("--spec", default="ev:fast")
    ap.add_argument("--deck-key", default="b_red")
    ap.add_argument("--stake", type=int, default=1)
    ap.add_argument("--lives", type=int, default=4)
    ap.add_argument("--player-name", default="Tagg")
    ap.add_argument("--resume", default=None, metavar="SESSION_DIR",
                    help="rebuild from an existing session's outbox and carry on")
    ap.add_argument("--no-install", action="store_true",
                    help="skip writing the launcher into the mod's replays folder")
    args = ap.parse_args(argv)

    if args.resume:
        session = Path(args.resume)
        seed = session.name.split("_")[0]
    else:
        seed = resolve_seed(args.seed)
        session = LIVE_RUNS_DIR / f"{seed}_{int(time.time())}"
        session.mkdir(parents=True, exist_ok=True)
    outbox = str(session / "outbox.jsonl")
    inbox = str(session / "inbox.jsonl")
    Path(outbox).touch()

    log_file = open(session / "session.log", "a", encoding="utf-8")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    car = Sidecar(outbox, inbox, seed=seed, spec=args.spec, deck_key=args.deck_key,
                  stake=args.stake, lives=args.lives, log=log)

    log(f"[sidecar] session {session}")
    log(f"[sidecar] computing the agent's opening (seed {seed}, {args.spec})...")
    if args.resume:
        car.recover()
    else:
        car.start()

    if not args.no_install:
        doc = launcher_doc(seed, args.deck_key, args.stake, args.spec, outbox, inbox,
                           player_name=args.player_name, bootstrap=car.pending)
        path = write_ghost(doc, str(Path(mod_replays_dir()) /
                                    f"live_{seed}_{slugify(args.spec)}.json"))
        log(f"[sidecar] launcher installed: {path}")

    if not car.done:
        log("[sidecar] ready — in Balatro: Practice -> Match Replays -> "
            f"'{args.spec} LIVE' -> Play Match.  Ctrl+C to stop.")
        try:
            car.run()
        except KeyboardInterrupt:
            log("[sidecar] stopped by user")
    log_file.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
