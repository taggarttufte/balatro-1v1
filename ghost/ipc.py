"""
ipc.py — the G2 file-based transport: append-only JSONL in both directions.

Design (G2_DESIGN.md §1-2): the Lua mod and the Python sidecar each append one JSON
object per line to their own file and tail the other's.  Append-only + idempotent
messages means either side can recover from a crash by re-reading from byte 0 —
``JsonlTail`` supports exactly that (`fresh=True`), and otherwise remembers its offset
between polls.

Windows note: the writer (Lua/nativefs or Python) and the reader hold the file at the
same time, so everything here opens with read-share semantics (plain ``open`` on CPython/
Windows shares fine for read; the writer only ever appends).  A partially-flushed last
line (no trailing newline yet) is left in the file for the NEXT poll rather than parsed.
"""
from __future__ import annotations

import json
import os
import time
from typing import Iterator, Optional

PROTOCOL_VERSION = 1


def append_event(path: str, event: str, **fields) -> dict:
    """Append one event object (adds ``e`` and ``ts``).  Creates the file/dirs."""
    obj = {"e": event, "ts": round(time.time(), 3), **fields}
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return obj


class JsonlTail:
    """Incremental reader of an append-only JSONL file.

    ``poll()`` returns every COMPLETE new line since the last poll, parsed; a trailing
    line without ``\\n`` (a write in flight) is not consumed.  A line that is complete
    but unparsable is returned as ``{"e": "_corrupt", "raw": ...}`` rather than raised —
    the stream must survive a torn write, and the caller decides how loud to be.
    The file not existing yet is a normal state (the other side hasn't started)."""

    def __init__(self, path: str, fresh: bool = True):
        self.path = str(path)
        self._offset = 0
        if not fresh and os.path.exists(self.path):
            self._offset = os.path.getsize(self.path)

    def poll(self) -> list:
        try:
            size = os.path.getsize(self.path)
        except OSError:
            return []
        if size < self._offset:          # file replaced/truncated: start over
            self._offset = 0
        if size == self._offset:
            return []
        with open(self.path, "r", encoding="utf-8", newline="") as f:
            f.seek(self._offset)
            chunk = f.read()
        events = []
        consumed = 0
        for line in chunk.splitlines(keepends=True):
            if not line.endswith("\n"):
                break                     # in-flight partial line: next poll
            consumed += len(line.encode("utf-8"))
            text = line.strip()
            if not text:
                continue
            try:
                events.append(json.loads(text))
            except ValueError:
                events.append({"e": "_corrupt", "raw": text})
        self._offset += consumed
        return events

    def wait_for(self, predicate, timeout: float = 60.0, interval: float = 0.2,
                 on_event=None) -> Optional[dict]:
        """Poll until an event satisfies ``predicate``; every event seen along the way is
        passed to ``on_event`` (so nothing is dropped while waiting).  None on timeout."""
        deadline = time.monotonic() + timeout
        while True:
            for ev in self.poll():
                if on_event is not None:
                    on_event(ev)
                if predicate(ev):
                    return ev
            if time.monotonic() >= deadline:
                return None
            time.sleep(interval)


def iter_session(path: str) -> Iterator[dict]:
    """All complete events currently in a file, from byte 0 (crash recovery / tests)."""
    tail = JsonlTail(path, fresh=True)
    yield from tail.poll()
