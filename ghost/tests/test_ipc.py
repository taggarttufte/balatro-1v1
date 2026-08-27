"""test_ipc.py — the append-only JSONL transport (G2_DESIGN.md §1-2)."""
from __future__ import annotations

import json

from ..ipc import JsonlTail, append_event, iter_session


def test_append_creates_and_stamps(tmp_path):
    p = tmp_path / "out" / "outbox.jsonl"
    obj = append_event(str(p), "session_start", seed="Q3YSA2CC", lives=4)
    assert p.exists()
    on_disk = json.loads(p.read_text().strip())
    assert on_disk == obj
    assert on_disk["e"] == "session_start" and on_disk["seed"] == "Q3YSA2CC"
    assert isinstance(on_disk["ts"], float)


def test_tail_incremental(tmp_path):
    p = str(tmp_path / "box.jsonl")
    tail = JsonlTail(p)
    assert tail.poll() == []                      # file may not exist yet: normal
    append_event(p, "a")
    append_event(p, "b")
    got = tail.poll()
    assert [x["e"] for x in got] == ["a", "b"]
    assert tail.poll() == []                      # nothing new
    append_event(p, "c")
    assert [x["e"] for x in tail.poll()] == ["c"]


def test_partial_line_not_consumed(tmp_path):
    p = tmp_path / "box.jsonl"
    tail = JsonlTail(str(p))
    p.write_text('{"e": "a", "ts": 1}\n{"e": "b", "ts"', encoding="utf-8")
    assert [x["e"] for x in tail.poll()] == ["a"]
    with open(p, "a", encoding="utf-8") as f:      # the write completes
        f.write(': 2}\n')
    assert [x["e"] for x in tail.poll()] == ["b"]


def test_corrupt_line_reported_not_raised(tmp_path):
    p = tmp_path / "box.jsonl"
    p.write_text('not json at all\n{"e": "ok", "ts": 1}\n', encoding="utf-8")
    got = JsonlTail(str(p)).poll()
    assert got[0]["e"] == "_corrupt" and got[1]["e"] == "ok"


def test_fresh_false_skips_history(tmp_path):
    p = str(tmp_path / "box.jsonl")
    append_event(p, "old")
    tail = JsonlTail(p, fresh=False)
    append_event(p, "new")
    assert [x["e"] for x in tail.poll()] == ["new"]


def test_iter_session_recovers_everything(tmp_path):
    p = str(tmp_path / "box.jsonl")
    for e in ("a", "b", "c"):
        append_event(p, e)
    assert [x["e"] for x in iter_session(p)] == ["a", "b", "c"]


def test_wait_for_sees_intermediate_events(tmp_path):
    p = str(tmp_path / "box.jsonl")
    append_event(p, "x")
    append_event(p, "target")
    seen = []
    tail = JsonlTail(p)
    hit = tail.wait_for(lambda ev: ev["e"] == "target", timeout=2.0,
                        on_event=seen.append)
    assert hit is not None and hit["e"] == "target"
    assert [x["e"] for x in seen] == ["x", "target"]


def test_wait_for_timeout(tmp_path):
    p = str(tmp_path / "box.jsonl")
    tail = JsonlTail(p)
    assert tail.wait_for(lambda ev: False, timeout=0.3, interval=0.05) is None
