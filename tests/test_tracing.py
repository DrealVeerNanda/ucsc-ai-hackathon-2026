"""Tracing: per-frame records, the report built from them, and the camera latency measurement."""
import json
import time

import numpy as np
import pytest

import guidance
import perception
import stream
import tracing
from interfaces import Frame
from tools import measure_camera_latency as mcl
from tools import trace_report
from tracing import Tracer


def read(tracer):
    tracer.close()
    return [json.loads(line) for line in open(tracer.path)]


def frames_of(records):
    return [r for r in records if r["type"] == "frame"]


# --- tracer --------------------------------------------------------------------

def test_one_record_per_frame_with_steps_and_loop_time(tmp_path):
    tr = Tracer(tmp_path, enabled=True)
    for key in (1.0, 2.0):
        tr.begin_frame(key, wait_ms=1.5)
        with tr.span("perception.detector"):
            time.sleep(0.02)
        tr.add("perception.hand", 3.0)
        tr.add("perception.hand", 2.0)            # repeats add up
        tr.begin_frame(key)                       # another lane announcing the same frame: no new record
        tr.set(state="guiding")
        time.sleep(0.01)                          # view, logging...
        tr.loop_returned()
        time.sleep(0.02)                          # ...then waiting for the camera
    recs = read(tr)
    assert recs[0]["type"] == "meta"
    a, b = frames_of(recs)
    assert a["i"] == 1 and b["i"] == 2 and a["state"] == "guiding" and a["wait_ms"] == 1.5
    assert a["spans"]["perception.hand"] == 5.0 and 18 < a["spans"]["perception.detector"] < 40
    assert 28 < a["loop_ms"] < 60 and 18 < a["idle_ms"] < 45
    assert a["loop_ms"] >= a["spans"]["perception.detector"]


def test_events_and_meta_are_written_from_anywhere(tmp_path):
    tr = Tracer(tmp_path, enabled=True)
    tr.meta(source="pathon")
    tr.event("cosmos", 612.3456, target="mug")
    with tr.timed("load.detector", kind="yoloworld"):
        time.sleep(0.01)
    tr.meta(audio_out_ms=5.3)                      # learned later: still recorded
    recs = read(tr)
    assert recs[0]["source"] == "pathon"
    ev = {r["name"]: r for r in recs if r["type"] == "event"}
    assert ev["cosmos"]["ms"] == 612.346 and ev["cosmos"]["target"] == "mug"
    assert ev["load.detector"]["ms"] >= 10 and ev["load.detector"]["kind"] == "yoloworld"
    assert any(r.get("audio_out_ms") == 5.3 for r in recs)


def test_switched_off_writes_nothing(tmp_path):
    tr = Tracer(tmp_path, enabled=False)
    tr.begin_frame(1.0)
    tr.add("x", 1.0)
    tr.event("cosmos", 5)
    tr.close()
    assert tr.path is None and list(tmp_path.iterdir()) == []


def test_steps_outside_a_frame_are_ignored(tmp_path):
    tr = Tracer(tmp_path, enabled=True)
    tr.add("x", 1.0)
    tr.set(state="idle")
    tr.loop_returned()
    tr.close()
    assert tr.path is None


# --- the lanes report into it --------------------------------------------------

def _recording(folder, n=8, dt=0.03):
    rec = stream.Recorder(folder)
    K = np.array([[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]])
    for i in range(n):
        rgb = np.full((240, 320, 3), 60, np.uint8)
        rgb[100:130, 150:180] = (40, 220, 60)                       # a green sticker to find
        rec.add(stream.RawFrame(rgb, np.full((48, 64), 0.5, np.float32), K, None, "record3d"), 10.0 + i * dt)
    rec.close()


def test_stream_perception_and_guidance_fill_one_record_per_frame(tmp_path, monkeypatch):
    tr = Tracer(tmp_path / "logs", enabled=True)
    monkeypatch.setattr(tracing, "_default", tr)
    monkeypatch.setenv("GUIDANCE_AUDIO", "off")
    monkeypatch.setattr(guidance, "_default", None)
    _recording(tmp_path / "rec")
    p = perception.Perception(None, perception.StickerHand())
    s = stream.Stream(str(tmp_path / "rec"), loop=False).start()
    guidance.reset("mug")
    n = 0
    try:
        while (fr := s.get_frame(wait=0.5)) is not None:
            assert isinstance(fr, Frame)
            guidance.update(p.locate(fr, "mug"))
            n += 1
        s.get_frame()                                               # the loop comes back once more
    finally:
        s.close()
    recs = read(tr)
    frames = frames_of(recs)
    assert len(frames) == n >= 3 and recs[0]["source"].endswith("rec")
    for f in frames:
        assert {"stream.prepare", "perception.hand", "perception.depth", "guidance"} <= set(f["spans"])
        assert f["state"] == "searching" and f["wait_ms"] >= 0 and f["skipped"] >= 0
        assert 0 <= f["e2e_ms"] < 500 and f["loop_ms"] >= sum(f["spans"].values()) - 0.01
        assert "cable_ms" not in f                                  # a recording has no phone clock


def test_frames_the_loop_never_took_are_counted(tmp_path, monkeypatch):
    tr = Tracer(tmp_path / "logs", enabled=True)
    monkeypatch.setattr(tracing, "_default", tr)
    _recording(tmp_path / "rec", n=12, dt=0.02)
    s = stream.Stream(str(tmp_path / "rec"), loop=False).start()
    try:
        s.get_frame(wait=0.5)
        time.sleep(0.15)                                            # a slow loop: frames pile up
        s.get_frame(wait=0.5)
    finally:
        s.close()
    assert frames_of(read(tr))[1]["skipped"] >= 3


def test_delay_over_the_quickest_frame_needs_a_sender_clock():
    s = stream.Stream("record3d")
    raw = lambda t_src: stream.RawFrame(None, None, None, None, "pathon", t_src)   # noqa: E731
    assert s._delay_ms(raw(None), 100.0) is None
    # sender clock runs 1000 s behind ours; frames take 20, 20, 65, 20 ms to arrive
    got = [s._delay_ms(raw(t), t + 1000.0 + d) for t, d in ((1.0, 0.020), (1.1, 0.020), (1.2, 0.065), (1.3, 0.020))]
    assert got == pytest.approx([0.0, 0.0, 45.0, 0.0], abs=0.01)


def test_relay_message_carries_the_send_time():
    raw = stream.RawFrame(np.zeros((24, 32, 3), np.uint8), np.ones((12, 16), np.float32), np.eye(3), None, "pathon")
    msg = stream.encode_raw(raw)
    _, a, b, c = stream._HEAD.unpack(msg[:stream._HEAD.size])
    body = msg[stream._HEAD.size:]
    out = stream.decode_raw(body[:a], body[a:a + b], body[a + b:a + b + c])
    assert abs(out.t_src - time.time()) < 5


# --- report --------------------------------------------------------------------

def _trace(n=100, detector=20.0, hand=10.0, idle=1.0, **extra):
    frames = [{"i": i + 1, "t": 1000.0 + 0.04 * i, "wait_ms": 2.0, "skipped": 1, "idle_ms": idle,
               "spans": {"stream.prepare": 1.0, "perception.detector": detector, "perception.hand": hand,
                         "guidance": 0.1}, "loop_ms": detector + hand + 1.1 + 4.0, "e2e_ms": detector + hand + 3.0,
               **extra} for i in range(n)]
    events = [{"name": "cosmos", "ms": 600.0}, {"name": "cosmos", "ms": 2000.0},
              {"name": "audio", "underruns": 2, "render_max_ms": 0.4}]
    return {"source": "pathon", "detector": "yoloworld", "audio_out_ms": 5.0}, frames, events


def test_report_ranks_steps_and_names_the_bottleneck():
    s = trace_report.summarize(*_trace())
    assert s["frames"] == 100 and s["fps"] == pytest.approx(25.0, abs=0.1) and s["unused"] == pytest.approx(0.5)
    assert [st["name"] for st in s["steps"]][:2] == ["perception.detector", "perception.hand"]
    assert s["bottleneck"]["name"] == "perception.detector" and s["bottleneck"]["share"] == pytest.approx(20 / 35.1, abs=0.01)
    rest = next(st for st in s["steps"] if st["name"] == trace_report.REST)
    assert rest["median"] == pytest.approx(4.0)                       # loop time no step accounts for
    assert sum(st["share"] for st in s["steps"]) == pytest.approx(1.0)
    assert s["events"][0]["name"] == "cosmos" and s["events"][0]["n"] == 2 and s["events"][0]["max"] == 2000
    assert s["audio"] == {"underruns": 2, "render_max_ms": 0.4}
    text = trace_report.render(s, "x.jsonl")
    assert "Biggest step: perception.detector" in text and "set by processing" in text
    assert "2 underruns" in text and "no timestamps" in text


def test_report_says_when_the_camera_is_the_limit():
    s = trace_report.summarize(*_trace(detector=5.0, hand=3.0, idle=30.0, cable_ms=4.0))
    text = trace_report.render(s)
    assert "set by the camera" in text and "no timestamps" not in text and s["cable"]["median"] == 4.0


def test_report_skips_warm_up_and_survives_an_empty_trace():
    meta, frames, events = _trace(n=10)
    frames[0]["loop_ms"] = 5000.0                                      # model start-up
    assert trace_report.summarize(meta, frames, events, skip=1)["loop"]["max"] < 100
    assert "No frames" in trace_report.render(trace_report.summarize({}, [], []))


def test_report_reads_a_file_and_writes_csv(tmp_path):
    meta, frames, events = _trace(n=5)
    p = tmp_path / "trace_x.jsonl"
    lines = [{"type": "meta", **meta}] + [{"type": "frame", **f} for f in frames] + [{"type": "event", **e} for e in events]
    p.write_text("\n".join(json.dumps(x) for x in lines) + '\n{"type":"frame","i":6,"sp')   # a cut-off last line
    m, f, e = trace_report.load(p)
    assert m["source"] == "pathon" and len(f) == 5 and len(e) == 3
    trace_report.write_csv(f, tmp_path / "out.csv")
    rows = (tmp_path / "out.csv").read_text().splitlines()
    assert rows[0].startswith("i,t,state,cable_ms,wait_ms,skipped,guidance,perception.detector") and len(rows) == 6


# --- camera latency ------------------------------------------------------------

def test_latency_is_the_time_to_the_first_frame_that_shows_the_flip():
    flips = [(1.000, True), (2.000, False), (3.000, True)]
    samples = []
    for i in range(130):                                               # a frame every 25 ms
        t = 0.5 + 0.025 * i
        lit = (1.080 <= t < 2.080) or t >= 3.080                       # the camera sees each flip 80 ms late
        samples.append((t, 200.0 if lit else 40.0))
    got = mcl.latencies(flips, samples, contrast=160.0)
    assert len(got) == 3 and all(80 <= d <= 105 for d in got)          # within one frame interval


def test_a_flip_nobody_saw_is_left_out():
    samples = [(0.5 + 0.025 * i, 40.0) for i in range(60)]
    assert mcl.latencies([(1.0, True)], samples, contrast=160.0) == []


class _DelayedCamera:
    """A camera pointed at the screen: its brightness follows what was shown, `delay` seconds late."""

    def __init__(self, delay=0.06, fps=40, bright=200.0, dark=40.0):
        self.delay, self.period, self.levels = delay, 1.0 / fps, (dark, bright)
        self.shown = [(time.monotonic(), 0)]
        self.next = time.monotonic()

    def show(self, level):
        self.shown.append((time.monotonic(), level))

    def get_frame(self, wait=0.0):
        now = time.monotonic()
        if now < self.next:
            time.sleep(min(wait, self.next - now))
            return None
        self.next = now + self.period
        level = [lv for t, lv in self.shown if t <= now - self.delay][-1:] or [0]
        rgb = np.full((40, 40, 3), self.levels[level[0] == 255], np.uint8)
        return Frame(rgb, None, None, None, now, t_recv=now)

    def status(self):
        return {"fps": 1.0 / self.period}


def test_measured_latency_matches_a_known_camera_delay():
    cam = _DelayedCamera(delay=0.06)
    delays, contrast, fps = mcl.run(cam, cam.show, flips_wanted=5, settle_s=0.3)
    assert contrast == pytest.approx(160.0) and len(delays) >= 4
    assert 55 <= np.median(delays) <= 100                              # 60 ms plus up to one frame


def test_dark_screen_is_reported_not_measured():
    cam = _DelayedCamera(bright=45.0)                                  # the phone is not looking at the screen
    delays, contrast, _ = mcl.run(cam, cam.show, flips_wanted=3, settle_s=0.2)
    assert delays == [] and contrast < 25
