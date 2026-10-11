"""The soak and fault-drill tools themselves: the fake camera, the misbehaving model server,
the trace reader and the arithmetic behind the report. No GPU, no pipeline."""
import json
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import fault_drill  # noqa: E402
import soak  # noqa: E402
import stream  # noqa: E402


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# --- requests ----------------------------------------------------------------

def test_request_script_repeats_with_the_same_seed_and_keeps_demo_pace():
    def take(seed):
        it = soak.request_script(seed)
        return [next(it) for _ in range(200)]
    a, b = take(3), take(3)
    assert a == b
    assert a != take(4)
    assert all(20.0 <= wait <= 60.0 for wait, _, _ in a)
    assert {kind for _, kind, _ in a} == {"there", "repeat", "not_there", "indirect", "list", "option", "nonsense"}


def test_a_repeat_asks_for_the_last_object_again():
    it = soak.request_script(1)
    last = soak.THERE[0]
    for _ in range(300):
        _, kind, text = next(it)
        if kind == "repeat":
            assert text == last
        elif kind in ("there", "not_there"):
            last = text


# --- camera ------------------------------------------------------------------

def test_control_file_survives_being_missing_or_half_written(tmp_path):
    path = tmp_path / "camera.json"
    assert soak.read_control(path) == soak.DEFAULT_CONTROL
    path.write_text('{"fps": 3, "pause": tr')
    assert soak.read_control(path, dict(soak.DEFAULT_CONTROL, fps=9.0))["fps"] == 9.0
    path.write_text('{"fps": 3, "pause": true, "unknown": 1}')
    got = soak.read_control(path)
    assert got["fps"] == 3 and got["pause"] is True and "unknown" not in got


def test_set_camera_changes_one_thing_and_keeps_the_rest(tmp_path):
    (tmp_path / "camera.json").write_text(json.dumps(dict(soak.DEFAULT_CONTROL, fps=12.0)))
    soak.set_camera(tmp_path, depth="zero")
    got = soak.read_control(tmp_path / "camera.json")
    assert got["depth"] == "zero" and got["fps"] == 12.0


def test_shape_depth():
    d = np.full((4, 5), 0.8, np.float32)
    assert soak.shape_depth(d, "real") is d
    assert not soak.shape_depth(d, "zero").any()
    assert np.allclose(soak.shape_depth(d, "flat:0.45"), 0.45)


def test_a_frame_message_is_what_the_relay_source_reads():
    rgb = np.random.default_rng(0).integers(0, 255, (48, 64, 3), np.uint8)
    depth = np.full((12, 16), 0.512, np.float32)
    K = np.array([[50.0, 0, 32], [0, 50, 24], [0, 0, 1]])
    msg = soak.frame_message(*soak.encode_pictures(rgb, depth), K, np.eye(4), "pathon")
    magic, n_meta, n_jpg, n_png = stream._HEAD.unpack(msg[:stream._HEAD.size])
    assert magic == stream._MAGIC and len(msg) == stream._HEAD.size + n_meta + n_jpg + n_png
    body = msg[stream._HEAD.size:]
    raw = stream.decode_raw(body[:n_meta], body[n_meta:n_meta + n_jpg], body[n_meta + n_jpg:])
    assert raw.kind == "pathon" and raw.rgb.shape == rgb.shape
    assert np.allclose(raw.depth, 0.512) and np.allclose(raw.K, K) and np.allclose(raw.pose, np.eye(4))
    assert abs(raw.t_src - time.time()) < 5


def synthetic_frames():
    rgb = np.zeros((48, 64, 3), np.uint8)
    depth = np.full((12, 16), 0.6, np.float32)
    K = np.array([[50.0, 0, 32], [0, 50, 24], [0, 0, 1]])
    while True:
        yield rgb, depth, K, np.eye(4), "record3d", 0.05


def test_the_camera_feeds_a_real_stream_and_obeys_its_control_file(tmp_path):
    port = free_port()
    control = tmp_path / "camera.json"
    control.write_text(json.dumps(dict(soak.DEFAULT_CONTROL, fps=40.0)))
    halt = threading.Event()
    cam = threading.Thread(target=soak.run_camera, args=([], port, str(control)),
                           kwargs={"fps": 40.0, "frames": synthetic_frames(), "stop": halt}, daemon=True)
    cam.start()
    s = stream.Stream(f"relay:{port}", max_side=None)
    try:
        s.start(timeout=10)
        raw, _ = s.get_raw(wait=2.0)
        assert raw.depth.max() == pytest.approx(0.6, abs=1e-3) and raw.pose is not None

        soak.set_camera(tmp_path, depth="zero", pose="none")
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            got = s.get_raw(wait=0.5)
            if got is not None and got[0].depth.max() == 0 and got[0].pose is None:
                break
        else:
            pytest.fail("the camera never switched to frames without depth and pose")

        soak.set_camera(tmp_path, pause=True)
        time.sleep(0.3)
        while s.get_raw(wait=0.2) is not None:      # whatever was already on its way
            pass
        assert s.get_raw(wait=0.5) is None

        soak.set_camera(tmp_path, pause=False)
        assert s.get_raw(wait=3.0) is not None
    finally:
        halt.set()
        s.close()
        cam.join(5)
    assert not cam.is_alive()


def test_the_camera_reconnects_after_the_other_side_goes_away():
    port = free_port()
    halt = threading.Event()
    cam = threading.Thread(target=soak.run_camera, args=([], port, None),
                           kwargs={"fps": 50.0, "frames": synthetic_frames(), "stop": halt}, daemon=True)
    cam.start()
    try:
        for _ in range(2):
            s = stream.Stream(f"relay:{port}", max_side=None).start(timeout=10)
            assert s.get_raw(wait=2.0) is not None
            s.close()
    finally:
        halt.set()
        cam.join(5)


# --- sound that goes nowhere -------------------------------------------------

def test_the_silent_card_plays_a_clip_in_real_time():
    import audio_out
    sr = 48000
    clip = np.zeros(int(0.3 * sr), np.float32)
    real, audio_out.Mixer = audio_out.Mixer, lambda cfg=None: real(cfg, clips={"grab": clip, "looking": clip})
    try:
        sink = soak.silent_sink()
    finally:
        audio_out.Mixer = real
    t = time.monotonic()
    sink.say("looking")
    assert sink.saying()
    sink.wait_quiet(3.0)
    took = time.monotonic() - t
    assert not sink.saying() and 0.2 <= took <= 1.5


# --- the trace ---------------------------------------------------------------

def test_trace_poll_returns_whole_lines_only_and_follows_the_newest_file(tmp_path):
    first = tmp_path / "trace_1.jsonl"
    first.write_text('{"type":"frame","t":1.0}\n{"type":"ev')
    tr = soak.Trace(tmp_path)
    assert [r["t"] for r in tr.poll()] == [1.0]
    with open(first, "a") as f:
        f.write('ent","t":2.0,"name":"say"}\nnot json\n')
    assert [r["t"] for r in tr.poll()] == [2.0]
    assert tr.poll() == []
    time.sleep(0.02)
    (tmp_path / "trace_2.jsonl").write_text('{"type":"frame","t":3.0}\n')
    assert [r["t"] for r in tr.poll()] == [3.0]


def test_trace_wait_for_ignores_what_came_before(tmp_path):
    path = tmp_path / "trace_1.jsonl"
    path.write_text('{"type":"event","t":5.0,"name":"say","text":"Looking"}\n'
                    '{"type":"event","t":9.0,"name":"say","text":"Grab"}\n')
    seen = []
    tr = soak.Trace(tmp_path)
    got = tr.wait_for(soak.event("say"), 0.5, since=6.0, seen=seen)
    assert got["text"] == "Grab" and len(seen) == 2
    assert soak.Trace(tmp_path).wait_for(soak.event("say", text="Nope"), 0.2) is None


def frames_at(times, loop_ms=20.0):
    return [{"type": "frame", "t": t, "loop_ms": loop_ms, "e2e_ms": loop_ms + 2, "skipped": 0} for t in times]


def test_frame_stats():
    st = soak.frame_stats(frames_at(np.arange(0, 10, 0.04)))
    assert st["fps"] == pytest.approx(25.0, rel=0.01)
    assert st["loop_p50"] == 20.0 and st["gap_max_ms"] == pytest.approx(40.0, abs=0.01)
    assert soak.frame_stats([])["fps"] == 0.0


def test_a_pause_is_a_stall_only_when_nothing_explains_it():
    frames = frames_at([0.0, 0.04, 5.0, 5.04, 6.0, 6.04, 9.0, 9.04])
    events = [{"type": "event", "t": 4.9, "name": "nemotron"},          # the wearer asked for something
              {"type": "event", "t": 9.05, "name": "detector.set_target"}]   # logged just after the pause it caused
    found = soak.stalls(frames, events)
    assert [round(t, 2) for t, _ in found] == [5.04]
    assert found[0][1] == pytest.approx(960.0)


def test_slope_per_hour():
    xs = np.arange(0, 3600, 10.0)
    assert soak.slope_per_hour(xs, 100 + xs * (50 / 3600)) == pytest.approx(50.0)
    assert soak.slope_per_hour([0, 1], [1, 2]) == 0.0


def write_soak(folder: Path, minutes: float, rss_per_hour: float = 0.0, die_at: float | None = None):
    (folder / "logs").mkdir(parents=True)
    t0 = 1000.0
    rows = []
    for i in range(int(minutes * 6)):
        el = (i + 1) * 10.0
        alive = int(die_at is None or el < die_at)
        rows.append({"t": t0 + el, "elapsed_s": el, "alive": alive, "rss_mb": 3000 + rss_per_hour * el / 3600,
                     "gpu_mb": 4000, "threads": 110, "fds": 60, "cpu_pct": 50, "dash_events_s": 25, "dash_mb_s": 1.0,
                     "dash_reconnects": 0, "logs_mb": el / 100, "trace_mb": el / 200, "gpu_busy_pct": 30, "load1": 2})
    import csv
    with open(folder / "samples.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=soak.FIELDS)
        w.writeheader()
        w.writerows(rows)
    with open(folder / "requests.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "kind", "text", "http_ms", "heard", "error"])
        w.writerow([t0 + 30, "there", "cup", 4, "cup", ""])
        w.writerow([t0 + 90, "list", "what's on the table", 4, "what's on the table", ""])
    with open(folder / "logs" / "trace_1.jsonl", "w") as f:
        for t in np.arange(t0, t0 + minutes * 60, 0.04):
            f.write(json.dumps({"type": "frame", "t": float(t), "loop_ms": 20.0, "e2e_ms": 22.0, "state": "searching"}) + "\n")
        f.write(json.dumps({"type": "event", "t": t0 + 30.2, "name": "say", "text": "Looking"}) + "\n")
        f.write(json.dumps({"type": "event", "t": t0 + 92.5, "name": "cosmos.list", "ms": 2000.0, "objects": ["cup"]}) + "\n")


def test_report_passes_a_steady_run_and_reads_the_requests(tmp_path):
    write_soak(tmp_path, 30)
    text = soak.report(tmp_path)
    assert "FAIL" not in text and "PASS  the pipeline ran to the end" in text
    assert "PASS  rss_mb grows" in text and "0.0 MB an hour" in text
    done = soak.request_outcomes(soak._read_csv(tmp_path / "requests.csv"), soak._load_trace(tmp_path)[1])
    assert done[0]["did"] == "say" and done[0]["react_ms"] == pytest.approx(200, abs=1)
    assert done[1]["did"] == "cosmos.list" and done[1]["react_ms"] == pytest.approx(500, abs=1)   # minus the 2 s Cosmos took


def test_report_fails_a_leak_and_a_pipeline_that_stopped(tmp_path):
    write_soak(tmp_path / "leak", 40, rss_per_hour=400)
    assert "FAIL  rss_mb grows +400.0 MB an hour" in soak.report(tmp_path / "leak")
    write_soak(tmp_path / "dead", 40, die_at=900)
    assert "FAIL  THE PIPELINE STOPPED" in soak.report(tmp_path / "dead")


def test_report_of_an_empty_folder_says_so(tmp_path):
    assert "nothing measured yet" in soak.report(tmp_path)


# --- the model server that misbehaves ----------------------------------------

@pytest.fixture
def upstream():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, body):
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send(b'{"status": "ok"}')

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self._send(json.dumps({"choices": [{"message": {"content": '{"intent": "pick", "target": "mug"}'}}]}).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def ask(port, timeout=5.0):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", b"{}", {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def test_proxy_passes_answers_through_and_can_break_them(upstream):
    proxy = fault_drill.ModelProxy(free_port(), upstream)
    try:
        assert json.loads(ask(proxy.port)["choices"][0]["message"]["content"])["target"] == "mug"
        with urllib.request.urlopen(f"http://127.0.0.1:{proxy.port}/health", timeout=5) as r:
            assert r.status == 200

        proxy.set("truncated")
        with pytest.raises(ValueError):
            json.loads(ask(proxy.port)["choices"][0]["message"]["content"])

        proxy.set("garbage")
        with pytest.raises(ValueError):
            ask(proxy.port)

        proxy.set("empty")
        assert ask(proxy.port) == {}

        proxy.set("http500")
        with pytest.raises(urllib.error.HTTPError):
            ask(proxy.port)

        proxy.set("slow", 0.4)
        t = time.monotonic()
        ask(proxy.port)
        assert time.monotonic() - t >= 0.4

        proxy.set("hang")
        with pytest.raises(OSError):
            ask(proxy.port, timeout=0.5)

        proxy.set("down")
        with pytest.raises(OSError):
            ask(proxy.port, timeout=1.0)

        proxy.set("pass")
        assert "choices" in ask(proxy.port)
    finally:
        proxy.close()


def test_voice_fallback_is_what_the_nemotron_drills_expect(upstream, monkeypatch):
    """The drills tell "Nemotron read it" from "the simple rule read it" by the target that
    comes out. Pin both readings of the same sentence."""
    import voice
    proxy = fault_drill.ModelProxy(free_port(), upstream)
    monkeypatch.setattr(voice, "_llm_ok", None)
    monkeypatch.setattr(voice, "_llm_retry_at", 0.0, raising=False)     # whichever way voice.py remembers a failure
    monkeypatch.setattr(voice, "last_objects", [])
    monkeypatch.setenv("VOICE_LLM_URL", f"http://127.0.0.1:{proxy.port}/v1")
    try:
        assert voice.interpret("I am thirsty") == ("pick", "mug")
        proxy.set("http500")
        assert voice.interpret("I am thirsty") == ("pick", "i am thirsty")
    finally:
        proxy.close()


# --- drill bookkeeping -------------------------------------------------------

def test_every_drill_is_listed_once_and_described():
    names = [*fault_drill.SHARED, *fault_drill.OWN]
    assert len(names) == len(set(names)) and len(names) >= 40
    assert all((fn.__doc__ or "").strip() for fn in {**fault_drill.SHARED, **fault_drill.OWN}.values())
    assert fault_drill.SLOW <= set(names)
    assert set(fault_drill.QUICK) <= set(names) - fault_drill.SLOW


def test_table_counts_passes_failures_and_information():
    R = fault_drill.Result
    text = fault_drill.table([R("a", True, 0.2, True, "fine"), R("b", False, None, False, "broke"), R("c", None, None, None, "numbers")])
    assert "1 passed, 1 failed, 1 for information" in text
    assert [line.split()[1] for line in text.splitlines()[1:4]] == ["PASS", "FAIL", "INFO"]


def test_port_free():
    with socket.socket() as s:
        s.bind(("0.0.0.0", 0))
        s.listen(1)
        assert not soak.port_free(s.getsockname()[1])
    assert soak.port_free(free_port())
