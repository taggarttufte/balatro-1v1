"""test_live.py — the G2 sidecar over real IPC files, driven by a scripted fake mod.

The fake mod behaves like the GhostRace Lua mod: it appends protocol events to the
outbox and reads the sidecar's inbox.  Real engine, real EVPlayer, no threads — the
test calls ``pump()`` synchronously after each batch of mod events.
"""
from __future__ import annotations

import pytest

from ..ipc import append_event, iter_session
from ..live import Sidecar, launcher_doc

SEED = "7I4M53DL"


def _events(path):
    return list(iter_session(str(path)))


def _by_kind(events, kind):
    return [e for e in events if e["e"] == kind]


@pytest.fixture()
def session(tmp_path):
    outbox = tmp_path / "outbox.jsonl"
    inbox = tmp_path / "inbox.jsonl"
    outbox.touch()
    car = Sidecar(str(outbox), str(inbox), seed=SEED, log=lambda *_: None)
    return car, outbox, inbox


def test_start_publishes_opening(session):
    car, outbox, inbox = session
    car.start()
    ev = _events(inbox)
    assert [e["e"] for e in ev[:2]] == ["hello", "agent_nemesis"]
    nem = ev[1]
    assert nem["ante"] >= 2 and nem["hands"] and nem["final"] > 0
    assert _by_kind(ev, "agent_state")[-1]["lives"] == 4
    assert car.pending["ante"] == nem["ante"]


def test_full_match_flow(session):
    car, outbox, inbox = session
    car.start()
    nem = _by_kind(_events(inbox), "agent_nemesis")[-1]

    # the mod connects and the human plays the first Nemesis... and loses it
    append_event(str(outbox), "session_start", seed=SEED)
    append_event(str(outbox), "nemesis_start", ante=nem["ante"], lives=4)
    append_event(str(outbox), "pvp_hand", ante=nem["ante"],
                 score=nem["final"] - 10, hands_left=0)
    append_event(str(outbox), "pvp_result", ante=nem["ante"],
                 human_score=nem["final"] - 10, human_lives=3, loser="human")
    car.pump()
    ev = _events(inbox)
    nems = _by_kind(ev, "agent_nemesis")
    assert len(nems) == 2 and nems[-1]["ante"] > nem["ante"]     # next round published
    assert car.mirror.opp.lives == 3
    assert not car.done

    # a regular-blind failure costs the human another life
    append_event(str(outbox), "round_fail", ante=nems[-1]["ante"], lives=2)
    car.pump()
    assert car.mirror.opp.lives == 2

    # the human wins every remaining Nemesis: the agent must die and concede
    for _ in range(10):
        if car.done:
            break
        nem = _by_kind(_events(inbox), "agent_nemesis")[-1]
        append_event(str(outbox), "pvp_result", ante=nem["ante"],
                     human_score=nem["final"] + 1, human_lives=2, loser="agent")
        car.pump()
    assert car.done and car.winner == "human"
    assert _by_kind(_events(inbox), "agent_dead")
    assert _by_kind(_events(inbox), "agent_state")[-1]["lives"] == 0


def test_human_death_ends_match(session):
    car, outbox, inbox = session
    car.start()
    append_event(str(outbox), "round_fail", ante=2, lives=0)
    car.pump()
    assert car.done and car.winner == "agent"


def test_match_end_event_stops(session):
    car, outbox, inbox = session
    car.start()
    append_event(str(outbox), "match_end", winner="abandoned")
    car.pump()
    assert car.done and car.winner == "abandoned"


def test_recover_replays_outbox(tmp_path):
    outbox = tmp_path / "outbox.jsonl"
    inbox = tmp_path / "inbox.jsonl"
    outbox.touch()
    car = Sidecar(str(outbox), str(inbox), seed=SEED, log=lambda *_: None)
    car.start()
    nem = _by_kind(_events(inbox), "agent_nemesis")[-1]
    append_event(str(outbox), "pvp_result", ante=nem["ante"],
                 human_score=nem["final"] - 10, human_lives=3, loser="human")
    car.pump()
    state = (car.mirror.ante, car.mirror.lives, car.mirror.money,
             car.pending["ante"], tuple(car.pvp_log_snapshot()))

    # "crash": a brand-new sidecar on the same files rebuilds the identical mirror
    car2 = Sidecar(str(outbox), str(inbox), seed=SEED, log=lambda *_: None)
    car2.recover()
    state2 = (car2.mirror.ante, car2.mirror.lives, car2.mirror.money,
              car2.pending["ante"], tuple(car2.pvp_log_snapshot()))
    assert state2 == state
    # and it republished the pending round for the mod's buffer
    assert _by_kind(_events(inbox), "agent_nemesis")[-1]["ante"] == state[3]

    # THE 2026-08-27 live-session bug: pumping after recovery must be a strict no-op —
    # the replayed events are consumed through the SAME tail, never applied twice
    car2.pump()
    state3 = (car2.mirror.ante, car2.mirror.lives, car2.mirror.money,
              car2.pending["ante"], tuple(car2.pvp_log_snapshot()))
    assert state3 == state
    assert not car2.done


def test_mid_match_reload_resets_the_mirror(session):
    car, outbox, inbox = session
    car.start()
    first = _by_kind(_events(inbox), "agent_nemesis")[-1]
    append_event(str(outbox), "session_start", seed=SEED)
    append_event(str(outbox), "pvp_result", ante=first["ante"],
                 human_score=first["final"] - 5, human_lives=3, loser="human")
    car.pump()
    assert car.pvp_log_snapshot()                       # one resolved round

    # the human reloads the launcher: a fresh run begins — the mirror restarts
    append_event(str(outbox), "session_start", seed=SEED)
    car.pump()
    assert car.pvp_log_snapshot() == []
    fresh = _by_kind(_events(inbox), "agent_nemesis")[-1]
    assert fresh["ante"] == first["ante"] and fresh["final"] == first["final"]
    assert not car.done


def test_resolve_parses_formatted_scores(session):
    """Talisman's Big tostring comma-groups: '1,073' must parse (the first live
    session's crash)."""
    car, outbox, inbox = session
    car.start()
    nem = _by_kind(_events(inbox), "agent_nemesis")[-1]
    append_event(str(outbox), "pvp_result", ante=nem["ante"],
                 human_score=f"{nem['final'] + 1000:,}", human_lives=4, loser="agent")
    car.pump()
    log = car.pvp_log_snapshot()[-1]
    assert log[3] == nem["final"] + 1000 and log[1] == 0    # agent lost, parsed clean


def test_launcher_doc_carries_bootstrap(session):
    car, outbox, inbox = session
    car.start()
    doc = launcher_doc(SEED, "b_red", 1, "ev:fast", str(outbox), str(inbox),
                       bootstrap=car.pending)
    live = doc["_live"]
    assert live["protocol"] == 1 and live["outbox"] == str(outbox)
    assert live["bootstrap"]["ante"] == car.pending["ante"]
    # graceful degrade: a G1-v2 single-entry snapshot for mod-less racing
    snap = doc["ante_snapshots"][str(car.pending["ante"])]
    assert snap["hands"] == [{"score": str(car.pending["final"]), "hands_left": 0,
                              "side": "enemy"}]
