"""Where does the time go? Every run writes logs/trace_YYYYmmdd_HHMMSS.jsonl; read it with

    python tools/trace_report.py            # newest run: a table of every step, slowest first

One line per camera frame, with the time each step took on that frame:

    cable_ms      phone capture -> frame arrived here, above the best case seen (queueing, hiccups)
    wait_ms       frame arrived -> the loop picked it up (it sat waiting while the loop was busy)
    skipped       camera frames that arrived and were never used, since the previous one
    spans         stream.prepare, perception.detector, perception.hand, perception.depth, guidance
    loop_ms       picked up -> the loop came back for the next frame (everything, incl. view and logs)
    idle_ms       the loop then waited this long for the camera (slack: the camera is the limit)
    e2e_ms        frame arrived -> the sound was updated

and one line per slow or occasional call: model loads, Cosmos requests, Whisper, audio underruns.

In code:

    with tracing.span("perception.detector"):      # a step inside the current frame
        ...
    tracing.event("cosmos", ms=612, target="mug")  # something slow that is not once per frame

TRACE=0 switches it off. TRACE_DIR changes the folder (default logs/).
"""
from __future__ import annotations

import atexit
import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path


class Tracer:
    def __init__(self, directory: str | Path | None = None, enabled: bool | None = None):
        self.enabled = os.environ.get("TRACE", "1") != "0" if enabled is None else enabled
        self.dir = Path(directory or os.environ.get("TRACE_DIR") or "logs")
        self.path: Path | None = None
        self._f = None
        self._lock = threading.Lock()
        self._frame = None           # the record being filled in
        self._key = None
        self._start = 0.0            # perf_counter when the current frame was picked up
        self._returned = None        # perf_counter when the loop came back asking for the next one
        self._n = 0
        self._meta = {}

    # --- file ------------------------------------------------------------------
    def _write(self, record: dict):
        with self._lock:
            if self._f is None:
                self.dir.mkdir(parents=True, exist_ok=True)
                self.path = self.dir / time.strftime("trace_%Y%m%d_%H%M%S.jsonl")
                self._f = open(self.path, "a")
                self._f.write(json.dumps({"type": "meta", "started": time.time(), "pid": os.getpid(),
                                          **self._meta}) + "\n")
                atexit.register(self.close)
            self._f.write(json.dumps(record, separators=(",", ":")) + "\n")
            if record["type"] != "frame" or self._n % 30 == 0:
                self._f.flush()

    def meta(self, **fields):
        """Facts about the run (source, models, audio latency). Written once the file exists."""
        if not self.enabled:
            return
        self._meta.update(fields)
        if self._f is not None:
            self._write({"type": "meta", **fields})

    # --- per frame -------------------------------------------------------------
    def begin_frame(self, key, **fields):
        """Start the record for a new camera frame (and finish the previous one). Calling it
        again with the same key is a no-op, so every lane can call it without coordinating."""
        if not self.enabled or key == self._key:
            if self.enabled and fields and self._frame is not None:
                self._frame.update(fields)
            return
        now = time.perf_counter()
        self._finish(now)
        self._key, self._start, self._returned = key, now, None
        self._n += 1
        self._frame = {"type": "frame", "i": self._n, "t": round(time.time(), 4), "spans": {}, **fields}

    def loop_returned(self):
        """The loop has come back for another frame: everything for the current one is done."""
        if self.enabled and self._frame is not None and self._returned is None:
            self._returned = time.perf_counter()

    def _finish(self, now: float):
        fr = self._frame
        if fr is None:
            return
        end = self._returned if self._returned is not None else now
        fr["loop_ms"] = round((end - self._start) * 1000, 3)
        fr["idle_ms"] = round((now - end) * 1000, 3)
        fr["spans"] = {k: round(v, 3) for k, v in fr["spans"].items()}
        self._frame = None
        self._write(fr)

    def add(self, name: str, ms: float):
        """Time spent on a named step of the current frame. Repeats add up."""
        if self.enabled and self._frame is not None:
            self._frame["spans"][name] = self._frame["spans"].get(name, 0.0) + ms

    def set(self, **fields):
        if self.enabled and self._frame is not None:
            self._frame.update(fields)

    @contextmanager
    def span(self, name: str):
        t = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, (time.perf_counter() - t) * 1000)

    # --- everything else -------------------------------------------------------
    def event(self, name: str, ms: float | None = None, **fields):
        """A slow or occasional call. Safe from any thread."""
        if not self.enabled:
            return
        rec = {"type": "event", "t": round(time.time(), 4), "name": name, **fields}
        if ms is not None:
            rec["ms"] = round(ms, 3)
        self._write(rec)

    @contextmanager
    def timed(self, name: str, **fields):
        """`with tracing.timed("load.detector"):` records an event with how long the block took."""
        t = time.perf_counter()
        try:
            yield
        finally:
            self.event(name, (time.perf_counter() - t) * 1000, **fields)

    def close(self):
        if self._frame is not None:
            self._finish(time.perf_counter())
        with self._lock:
            if self._f is not None:
                self._f.close()
                self._f = None


_default = Tracer()


# Module-level shortcuts. They look the tracer up on every call so tests can swap it.
def begin_frame(key, **fields):
    _default.begin_frame(key, **fields)


def loop_returned():
    _default.loop_returned()


def add(name: str, ms: float):
    _default.add(name, ms)


def put(**fields):
    """Set plain fields on the current frame's record, e.g. tracing.put(state="guiding")."""
    _default.set(**fields)


def span(name: str):
    return _default.span(name)


def event(name: str, ms: float | None = None, **fields):
    _default.event(name, ms, **fields)


def timed(name: str, **fields):
    return _default.timed(name, **fields)


def meta(**fields):
    _default.meta(**fields)


def close():
    _default.close()


def path() -> Path | None:
    return _default.path
