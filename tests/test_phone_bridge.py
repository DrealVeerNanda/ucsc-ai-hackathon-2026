"""The phone bridge (tools/phone_bridge.py) against a pretend phone (tools/fake_phone.py): frames
reach readers intact, readers and bridge may each die without hurting the other, every way the
phone can misbehave shows up in out/status/phone.json with words a person can act on."""
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import stream
from stream import FrameBoard, SourceError, Stream
from tools import fake_phone, phone_bridge
from tools.fake_phone import FakePhone
from tools.phone_bridge import Bridge, Packets, decode

ROOT = Path(__file__).resolve().parent.parent
FRAMES = fake_phone.synthetic_frames(6, (320, 240), (64, 48))


@pytest.fixture
def home(monkeypatch):
    """The bridge's folder. Short on purpose: the path of a local socket may be ~100 characters."""
    d = Path(tempfile.mkdtemp(prefix="pb", dir="/tmp"))
    monkeypatch.setenv("PHONE_BRIDGE_DIR", str(d))
    monkeypatch.setenv("PHONE_BRIDGE_SYSFS", str(d / "no-usb-list"))      # as on a Mac: ask idevice_id
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def phone():
    p = FakePhone(FRAMES, fps=30).start()
    yield p
    p.stop()


@pytest.fixture
def bridges(home):
    """Starts bridges on threads and stops whatever is still running at the end."""
    running = []

    def start(phone=None, wait="streaming", **kw):
        if phone is not None:
            kw = {"port": phone.port, "phone_ip": "127.0.0.1", "usb": False, **kw}
        b = Bridge(home, record_dir=home / "recordings", **kw)
        th = threading.Thread(target=b.run, daemon=True)
        th.start()
        running.append((b, th))
        if wait:
            _wait_state(home, wait, 5.0)
        return b

    def stop(b):
        b.stop()
        for other, th in running:
            if other is b:
                th.join(timeout=10.0)
                assert not th.is_alive()

    start.stop = stop
    yield start
    for b, th in running:
        b.stop()
        th.join(timeout=10.0)


def _status(home):
    try:
        return json.loads((home / "status" / "phone.json").read_text())
    except (OSError, ValueError):
        return {}


def _wait(cond, seconds, what="it"):
    """Seconds until cond() is true; fails the test if it takes longer than `seconds`."""
    t = time.monotonic()
    while time.monotonic() - t < seconds:
        if cond():
            return time.monotonic() - t
        time.sleep(0.01)
    raise AssertionError(f"{what} did not happen within {seconds} s")


def _wait_state(home, state, seconds=2.0):
    return _wait(lambda: _status(home).get("state") == state, seconds, f"state {state} (now: {_status(home)})")


def _frames(s, n=3, seconds=3.0):
    out, end = [], time.monotonic() + seconds
    while len(out) < n and time.monotonic() < end:
        got = s.get_raw(wait=0.2)
        if got is not None:
            out.append(got[0])
    assert len(out) == n, f"only {len(out)} of {n} frames: {s.status()}"
    return out


# --- the wire ------------------------------------------------------------------------

def _packet(i=0, t=1.5):
    f = FRAMES[i]
    return fake_phone.pack(i, t, f["K"], f["pose"], f["depth"], f["jpeg"], f["conf"], f["rgb_size"])


def test_a_packet_cut_anywhere_comes_out_whole_and_exact():
    p, data = Packets(), _packet(2, t=12.25) + _packet(3)
    got = []
    for at in range(0, len(data), 997):                 # as a socket delivers it: in arbitrary pieces
        got += p.feed(data[at:at + 997])
    assert [h["frame_id"] for h, _ in got] == [2, 3] and p.bad == 0 and got[0][0]["t"] == 12.25
    rgb, depth, K, pose = decode(*got[0])
    assert rgb.shape == (240, 320, 3) and rgb.dtype == np.uint8
    assert np.array_equal(depth, FRAMES[2]["depth"]) and (depth == 0).any()        # lidar holes stay holes
    assert np.allclose(K, FRAMES[2]["K"]) and np.allclose(pose, FRAMES[2]["pose"])


def test_low_confidence_depth_is_dropped_like_the_direct_reader_does():
    f = FRAMES[0]
    conf = np.full(f["depth"].shape, 2, np.uint8)
    conf[:10] = 0
    head, packet = Packets().feed(fake_phone.pack(0, 0.0, f["K"], f["pose"], np.ones_like(f["depth"]), f["jpeg"], conf))[0]
    depth = decode(head, packet)[1]
    assert (depth[:10] == 0).all() and (depth[10:] == 1).all()


def test_garbage_and_impossible_headers_are_skipped_and_counted():
    p = Packets()
    broken = _packet(1)
    broken = broken[:22] + (0x7FFFFFFF).to_bytes(4, "little") + broken[26:]
    got = p.feed(b"\x00" * 5000 + _packet(0) + broken + b"REC" + _packet(2))
    assert [h["frame_id"] for h, _ in got] == [0, 2]
    assert p.bad == 2                                     # the junk before, and the broken one
    f = FRAMES[0]
    head, packet = Packets().feed(fake_phone.pack(0, 0.0, f["K"], f["pose"], f["depth"], b"not a jpeg" * 50))[0]
    assert decode(head, packet) is None


def test_the_real_sdk_reads_the_pretend_phone():
    sdk = pytest.importorskip("sdk")
    f = FRAMES[4]
    fr = sdk.protocol.parse_frame(_packet(4, t=7.5))
    assert fr.frame_id == 4 and fr.timestamp == 7.5 and fr.color.shape == (240, 320, 3)
    assert np.array_equal(fr.depth, f["depth"]) and np.array_equal(fr.confidence, f["conf"])
    assert np.isclose(fr.intrinsics.fx, f["K"][0, 0]) and np.isclose(fr.intrinsics.ppy, f["K"][1, 2])
    assert np.allclose(fr.transform, f["pose"]) and fr.imu is not None


# --- the board -----------------------------------------------------------------------

def test_board_keeps_the_newest_frame_exactly():
    board = FrameBoard.create(rgb_cap=64 * 48 * 3, depth_cap=16 * 12 * 4)
    reader = FrameBoard.attach(board.name)
    try:
        assert reader.read() is None
        rng = np.random.default_rng(0)
        for i in range(1, 8):                             # more frames than slots: they are reused
            rgb = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
            depth = rng.random((12, 16), dtype=np.float32)
            pose = None if i % 2 else np.eye(4) * i
            assert board.write(rgb, depth, np.eye(3) * i, pose, t_recv=10.0 + i, t_src=None if i == 7 else 3.0 * i) == i
            seq, raw = reader.read(after=i - 1)
            assert seq == i and np.array_equal(raw.rgb, rgb) and np.array_equal(raw.depth, depth)
            assert np.array_equal(raw.K, np.eye(3) * i) and raw.t_recv == 10.0 + i and raw.kind == "pathon"
            assert (raw.pose is None) == bool(i % 2) and raw.t_src == (None if i == 7 else 3.0 * i)
            assert reader.read(after=i) is None           # nothing newer
        raw.rgb[:] = 0                                    # a reader's frame is its own copy
        assert reader.read()[1].rgb.any()
        assert board.fits(rgb, depth) and not board.fits(np.zeros((480, 640, 3), np.uint8), depth)
    finally:
        reader.close()
        board.close()
    with pytest.raises(SourceError):
        FrameBoard.attach(board.name)                     # the bridge took it away when it closed


def test_a_reader_that_exits_does_not_take_the_board_with_it():
    """Before Python 3.13, a program that merely opened shared memory removed it on exit."""
    board = FrameBoard.create(rgb_cap=64, depth_cap=64)
    try:
        code = ("import sys; sys.path.insert(0, sys.argv[1]); import stream;"
                "b = stream.FrameBoard.attach(sys.argv[2]); print('attached', flush=True)")
        for kill in (False, True):
            p = subprocess.Popen([sys.executable, "-c", code + ("; import time; time.sleep(30)" if kill else ""),
                                  str(ROOT), board.name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            assert p.stdout.readline().strip() == b"attached", p.stderr.read()
            if kill:
                p.kill()
            p.wait(timeout=10)
            time.sleep(0.3)                               # whatever would clean up after it has had its chance
            FrameBoard.attach(board.name).close()
    finally:
        board.close()


# --- frames through the bridge ---------------------------------------------------------

def test_frames_reach_a_reader_through_the_bridge(home, phone, bridges):
    b = bridges(phone)
    s = Stream("bridge").start()
    try:
        raws = _frames(s, 5)
        raw = raws[-1]
        src = FRAMES[[np.array_equal(raw.depth, f["depth"]) and np.allclose(raw.pose, f["pose"]) for f in FRAMES].index(True)]
        assert raw.rgb.shape == (240, 320, 3) and raw.kind == "pathon" and np.allclose(raw.K, src["K"])
        # the picture went through JPEG once, on the phone, and not again
        import cv2
        sent = cv2.cvtColor(cv2.imdecode(np.frombuffer(src["jpeg"], np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        assert np.abs(raw.rgb.astype(int) - sent).max() <= 2
        # stamped by the bridge on arrival, on the clock every program here shares
        assert 0 <= time.monotonic() - raw.t_recv < 0.5 and 0 <= raw.t_recv - raw.t_src < 0.2
        fr = s.get_frame(wait=1.0)
        assert fr is not None and fr.rgb.shape == (320, 240, 3) and fr.t_recv <= time.monotonic()
        st = s.status()
        assert st["state"] == "streaming" and not st["stalled"] and st["message"] is None
        assert st["bridge"]["state"] == "streaming" and st["bridge"]["via"] == "wifi"
    finally:
        s.close()
    assert phone.stats()["connections"] == 1 and b.frames > 5


def test_restarting_the_reader_never_disconnects_the_phone(home, phone, bridges):
    bridges(phone)
    for _ in range(5):
        s = Stream("bridge").start()
        _frames(s, 2)
        s.close()
    st = phone.stats()
    assert st["connections"] == 1 and st["disconnects"] == 0
    _wait(lambda: _status(home)["readers"] == 0, 2.0, "the bridge noticing the readers left")


def test_several_readers_at_once_and_one_killed_dead(home, phone, bridges):
    b = bridges(phone)
    code = ("import sys, time; sys.path.insert(0, sys.argv[1]); import stream;"
            "s = stream.Stream('bridge').start(); print('reading', flush=True)\n"
            "while True: s.get_raw(wait=0.5)")
    victim = subprocess.Popen([sys.executable, "-c", code, str(ROOT)], stdout=subprocess.PIPE, env=os.environ)
    a, c = Stream("bridge").start(), Stream("bridge").start()
    try:
        assert victim.stdout.readline().strip() == b"reading"
        _wait(lambda: _status(home)["readers"] == 3, 2.0, "three readers")
        victim.send_signal(signal.SIGKILL)
        victim.wait(timeout=5)
        _wait(lambda: _status(home)["readers"] == 2, 2.0, "the bridge dropping the dead reader")
        n = b.frames
        _frames(a, 3), _frames(c, 3)
        assert b.frames > n and _status(home)["state"] == "streaming" and phone.stats()["connections"] == 1
    finally:
        victim.kill()
        a.close(), c.close()


def test_a_reader_notices_a_dead_bridge_within_a_second_and_comes_back(home, phone, bridges):
    first = bridges(phone)
    s = Stream("bridge").start()
    try:
        _frames(s, 2)
        bridges.stop(first)
        took = _wait(lambda: s.status()["state"] == "reconnecting", 1.0, "the reader noticing")
        assert took < 1.0 and "bridge" in s.status()["error"]
        assert not stream.bridge_running(home)
        bridges(phone)                                    # the launcher starts another
        _wait(lambda: s.status()["state"] == "streaming", 3.0, "the reader attaching to the new bridge")
        _frames(s, 2)
    finally:
        s.close()


def _bridge_process(home, phone, *args):
    return subprocess.Popen([sys.executable, str(ROOT / "tools" / "phone_bridge.py"), "run", "--no-usb", "--phone-ip",
                             "127.0.0.1", "--port", str(phone.port), "--record-dir", str(home / "recordings"), *args],
                            env=os.environ, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def test_a_bridge_killed_dead_is_noticed_and_its_successor_cleans_up(home, phone, bridges):
    proc = _bridge_process(home, phone)
    try:
        _wait_state(home, "streaming", 15.0)
        board = _status(home)["shm"]
        s = Stream("auto").start()                        # auto: there is a bridge, so that is where frames come from
        try:
            _frames(s, 2)
            assert s.status()["bridge"]["pid"] == proc.pid
            proc.kill()
            took = _wait(lambda: s.status()["state"] == "reconnecting", 1.0, "the reader noticing")
            assert took < 1.0
            proc.wait(timeout=5)
            assert not stream.bridge_running(home)
            bridges(phone)
            _wait(lambda: s.status()["state"] == "streaming", 3.0, "the reader attaching to the new bridge")
            assert _status(home)["shm"] != board
            with pytest.raises(SourceError):
                FrameBoard.attach(board)                  # the dead bridge's memory was removed
        finally:
            s.close()
    finally:
        proc.kill()


def test_only_one_bridge_at_a_time(home, phone, bridges):
    bridges(phone)
    assert stream.bridge_running(home)
    with pytest.raises(phone_bridge.AlreadyRunning):
        Bridge(home, port=phone.port, phone_ip="127.0.0.1", usb=False).run()
    assert _status(home)["state"] == "streaming"          # and the first is untouched


def test_auto_reads_the_phone_itself_when_no_bridge_is_running(home, monkeypatch):
    assert not stream.bridge_running(home)
    monkeypatch.setattr(stream, "_pathon_frames", lambda *a, **k: iter([stream.RawFrame(
        np.zeros((24, 32, 3), np.uint8), np.ones((12, 16), np.float32), np.eye(3), None, "pathon")]))
    assert next(stream._auto_frames()).rgb.shape == (24, 32, 3)


# --- what the status file says ---------------------------------------------------------

def test_status_file_follows_the_contract_and_is_rewritten_every_second(home, phone, bridges):
    bridges(phone)
    time.sleep(1.2)
    st = _status(home)
    assert st["state"] == "streaming" and st["ok"] is True and abs(st["t"] - time.time()) < 2.0
    assert "Streaming over Wi-Fi at" in st["message"] and st["via"] == "wifi"
    assert 20 < st["fps"] < 40 and st["frame_age_s"] < 0.5 and st["frames"] > 10 and st["reconnects"] == 0
    assert 0.9 < st["depth_valid"] < 1.0                  # the made-up depth has one hole
    assert st["rgb"] == [240, 320] and st["depth"] == [48, 64] and st["bad_frames"] == 0
    assert st["recording"] is False and st["disk_free_gb"] > 0 and st["usb_reattaches"] == 0
    t = st["t"]
    _wait(lambda: _status(home).get("t", t) > t, 1.5, "the next report")
    assert stream.bridge_status(home)["state"] == "streaming"


def test_a_stopped_bridge_says_so_and_cleans_up(home, phone, bridges):
    b = bridges(phone)
    board = b.board.name
    bridges.stop(b)
    st = _status(home)
    assert st["state"] == "stopped" and st["ok"] is False and not (home / "phone_bridge.sock").exists()
    with pytest.raises(SourceError):
        FrameBoard.attach(board)
    with pytest.raises(SourceError, match="not running"):
        Stream("bridge").start()


@pytest.mark.parametrize("mode, state, words", [
    ("refuse", "app_not_streaming", "tap Start Streaming"),
    ("drop", "app_not_streaming", "tap Start Streaming"),
    ("stall", "stalled", "sent no picture"),
])
def test_each_way_of_failing_is_named_within_two_seconds(home, phone, bridges, mode, state, words):
    b = bridges(phone, redial_s=0)
    phone.set_mode(mode)
    if mode != "stall":
        phone.restart(0.1)                                # the app stops: everyone is hung up on
    took = _wait_state(home, state, 2.0)
    st = _status(home)
    assert took < 2.0 and st["ok"] is False and words in st["message"], st
    phone.set_mode("stream")                              # someone taps Start
    assert _wait_state(home, "streaming", 2.0) < 2.0
    assert b.reconnects == (0 if mode == "stall" else 1)   # a stall that ends needs no new connection


def test_a_stall_is_redialled_after_a_while(home, phone, bridges):
    b = bridges(phone, redial_s=1.5)
    phone.set_mode("stall")
    _wait_state(home, "stalled", 2.0)
    _wait(lambda: phone.stats()["connections"] >= 2, 3.0, "the bridge dialling again")
    phone.set_mode("stream")
    _wait_state(home, "streaming", 2.0)
    assert b.reconnects >= 1


def test_a_slow_stream_is_reported_as_slow(home, phone, bridges):
    bridges(phone)
    phone.fps = 5.0
    took = _wait(lambda: 3 <= _status(home).get("fps", 99) <= 7, 3.0, "the rate showing")
    assert took < 3.0 and _status(home)["state"] == "streaming"
    _wait(lambda: _status(home).get("slow") and "slow" in _status(home)["message"], 4.0, "the warning")


@pytest.mark.parametrize("kind", ["jpeg", "garbage", "header"])
def test_a_corrupt_frame_is_counted_and_the_stream_carries_on(home, phone, bridges, kind):
    b = bridges(phone)
    phone.corrupt(kind)
    _wait(lambda: b.bad_frames >= 1, 2.0, "the bad frame being counted")
    n = b.frames
    _wait(lambda: b.frames > n + 5, 2.0, "frames carrying on")
    assert _status(home)["state"] == "streaming" and b.reconnects == 0 and phone.stats()["connections"] == 1


def test_a_phone_that_restarts_is_picked_up_again(home, phone, bridges):
    b = bridges(phone)
    s = Stream("bridge").start()
    try:
        _frames(s, 2)
        phone.restart(0.8)
        _wait_state(home, "app_not_streaming", 2.0)
        assert s.status()["message"] and "Start Streaming" in s.status()["message"]     # the reader passes it on
        _wait_state(home, "streaming", 3.0)
        _frames(s, 2)
        assert b.reconnects == 1 and s.reconnects == 0    # the reader never let go of the bridge
    finally:
        s.close()


def test_an_app_that_wants_a_tap_is_reported_and_picked_up_when_tapped(home, bridges):
    phone = FakePhone(FRAMES, fps=30, close_on_disconnect=True).start()
    try:
        b = bridges(phone)
        bridges.stop(b)                                   # the only client leaves: the app stops, as PathOn does
        _wait(lambda: phone.stats()["needs_tap"], 2.0, "the app stopping")
        bridges(phone, wait="app_not_streaming")
        with pytest.raises(SourceError, match="tap Start Streaming"):
            Stream("bridge").start()                      # a reader is told what to do, at once
        assert "tap Start Streaming" in stream.status()["message"]
        phone.tap()
        assert _wait_state(home, "streaming", 1.5) < 1.5
    finally:
        phone.stop()


def test_a_blip_shorter_than_the_apps_patience_needs_no_tap(home, bridges):
    """What tonight's logs suggest PathOn does: it keeps its door open for a few seconds."""
    phone = FakePhone(FRAMES, fps=30, close_on_disconnect=True, grace_s=1.0).start()
    try:
        b = bridges(phone)
        for conn in list(phone._clients):
            phone._hang_up(conn)                          # the link breaks, the app is still there
        _wait(lambda: b.reconnects == 1, 1.0, "the bridge noticing")
        assert _wait_state(home, "streaming", 1.0) < 1.0 and not phone.stats()["needs_tap"]
    finally:
        phone.stop()


# --- the USB path, with stand-ins for idevice_id, idevicepair and iproxy ------------------

@pytest.fixture
def usb(home, phone, monkeypatch):
    state = home / "usb.json"
    for name in ("idevice_id", "idevicepair", "iproxy"):
        monkeypatch.setenv(f"PHONE_BRIDGE_{name.upper()}", f"{sys.executable} {ROOT / 'tools' / 'fake_phone.py'} {name} {state}")

    def set_usb(word):
        tmp = state.with_suffix(".tmp")
        tmp.write_text(json.dumps({"usb": word, "port": phone.port}))
        os.replace(tmp, state)
    set_usb("ok")
    return set_usb


def test_usb_states_are_told_apart(home, phone, bridges, usb):
    b = bridges(usb=True)
    st = _status(home)
    assert st["via"] == "usb" and "USB cable" in st["message"]
    for word, state, words in [("absent", "no_phone", "Plug it in"), ("untrusted", "untrusted", "tap Trust"),
                               ("locked", "untrusted", "Unlock it"), ("ok", "streaming", "Streaming over the USB cable")]:
        usb(word)
        if word != "ok":
            phone.restart(0.1)                            # whatever happened, the stream is gone
            b._pairing_at = 0.0
        _wait(lambda: _status(home)["state"] == state and words.lower() in _status(home)["message"].lower(), 4.0,
              f"{word} -> {state} saying {words!r} (now: {_status(home).get('state')}: {_status(home).get('message')})")
    assert _status(home)["ok"] is True


def test_usb_with_the_app_not_streaming_says_tap_start_and_then_streams(home, phone, bridges, usb):
    phone.set_mode("refuse")                              # the phone is on the cable, PathOn is not streaming
    bridges(usb=True, wait="app_not_streaming")
    st = _status(home)
    assert "choose USB and tap Start Streaming" in st["message"] and st["via"] is None
    assert phone.stats()["connections"] == 0
    phone.set_mode("stream")
    assert _wait_state(home, "streaming", 2.0) < 2.0 and _status(home)["via"] == "usb"


def test_wifi_takes_over_when_the_cable_has_no_phone(home, phone, bridges, usb):
    usb("absent")
    bridges(usb=True, phone_ip="127.0.0.1", port=phone.port)
    assert _status(home)["via"] == "wifi"


def test_the_usb_port_is_watched_and_blips_are_counted(home, monkeypatch):
    tree = home / "sys"
    monkeypatch.setenv("PHONE_BRIDGE_SYSFS", str(tree))
    (tree / "5-1.2").mkdir(parents=True)
    (tree / "5-1.2" / "idVendor").write_text("413c\n")     # a keyboard

    def plug(devnum):
        d = tree / "5-1.4"
        d.mkdir()
        for name, text in (("idVendor", "05ac"), ("product", "iPhone"), ("devnum", str(devnum)), ("speed", "480")):
            (d / name).write_text(text + "\n")
    w = phone_bridge.UsbWatch()
    assert w.poll() is False and (w.detaches, w.reattaches) == (0, 0)
    plug(10)
    assert w.poll() is True and (w.detaches, w.reattaches) == (0, 0)       # plugged in for the first time
    shutil.rmtree(tree / "5-1.4")
    assert w.poll() is False and (w.detaches, w.reattaches) == (1, 0)
    plug(11)
    assert w.poll() is True and (w.detaches, w.reattaches) == (1, 1)
    shutil.rmtree(tree / "5-1.4")
    plug(12)                                              # gone and back between two looks: the number changed
    assert w.poll() is True and (w.detaches, w.reattaches) == (2, 2)


def test_no_phone_on_the_port_is_known_without_asking_any_helper(home, bridges, monkeypatch):
    (home / "sys").mkdir()
    monkeypatch.setenv("PHONE_BRIDGE_SYSFS", str(home / "sys"))
    monkeypatch.setenv("PHONE_BRIDGE_IDEVICE_ID", "/nonexistent/idevice_id")
    bridges(usb=True, wait="no_phone")
    st = _status(home)
    assert st["usb_present"] is False and "No iPhone on the USB cable" in st["message"]


# --- recording -------------------------------------------------------------------------

def _folders(home):
    return sorted((home / "recordings").glob("demo_*"))


def test_recording_is_switched_while_streaming_and_replays(home, phone, bridges):
    b = bridges(phone)
    s = Stream("bridge").start()
    try:
        assert phone_bridge.main(["record", "on"]) == 0
        _wait(lambda: _status(home)["recording"], 3.0, "recording starting")
        st = _status(home)
        assert st["record_folder"] == str(_folders(home)[0]) and st["record_error"] is None
        _wait(lambda: _status(home)["record_frames"] >= 8, 3.0, "frames being saved")
        _frames(s, 3)                                     # readers are not disturbed
        assert phone_bridge.main(["record", "off"]) == 0
        _wait(lambda: not _status(home)["recording"], 3.0, "recording stopping")
        folder = _folders(home)[0]
        _wait(lambda: (folder / "meta.json").exists(), 5.0, "the folder being finished")
        meta = json.loads((folder / "meta.json").read_text())
        assert 8 <= meta["frames"] <= 40 and 7 < meta["fps"] < 13 and meta["kind"] == "pathon"
        got = list(stream._replay_frames(folder, loop=False, paced=False))
        assert len(got) == meta["frames"] and got[0].rgb.shape == (240, 320, 3) and got[0].depth.shape == (48, 64)
        assert phone.stats()["connections"] == 1 and b.reconnects == 0
    finally:
        s.close()


def test_recording_rolls_to_a_new_folder(home, phone, bridges, monkeypatch):
    names = iter(f"demo_20261011_10000{i}" for i in range(10))
    real = time.strftime
    monkeypatch.setattr(phone_bridge.time, "strftime", lambda fmt, *a: next(names) if fmt.startswith("demo_") else real(fmt, *a))
    bridges(phone, record=True, record_minutes=1.0 / 60)   # a new folder every second
    _wait(lambda: len(_folders(home)) >= 3, 5.0, "three folders")
    first = _folders(home)[0]
    _wait(lambda: (first / "meta.json").exists(), 5.0, "the first folder being finished")
    assert 5 <= len(list(stream._replay_frames(first, loop=False, paced=False))) <= 15


def test_no_recording_when_the_disk_is_nearly_full(home, phone, bridges):
    bridges(phone, record=True, min_free_gb=10 ** 6)
    _wait(lambda: _status(home).get("record_error"), 3.0, "the refusal")
    st = _status(home)
    assert st["recording"] is False and st["record_wanted"] is True and st["ok"] is True
    assert "NOT recording" in st["message"] and "GB free" in st["message"] and not _folders(home)


def test_the_switch_file_wins_over_the_flag_and_survives_a_restart(home, phone, bridges):
    (home / "status").mkdir()
    (home / "status" / "phone_record").write_text("off\n")
    b = bridges(phone, record=True)
    time.sleep(1.2)
    assert _status(home)["recording"] is False and not _folders(home)
    (home / "status" / "phone_record").write_text("on\n")
    _wait(lambda: _status(home)["recording"], 3.0, "recording starting")
    bridges.stop(b)
    bridges(phone)                                        # restarted without the flag: the file still says on
    _wait(lambda: _status(home)["recording"], 3.0, "recording starting again")


def test_a_recording_survives_the_bridge_being_killed(home, bridges):
    # Pictures of the size that is really recorded: the video reaches the disk a few tens of
    # kilobytes at a time, so with tiny ones nothing would be there yet.
    phone = FakePhone(fake_phone.synthetic_frames(4, (960, 720), (64, 48)), fps=30).start()
    try:
        proc = _bridge_process(home, phone, "--record")
        try:
            _wait(lambda: _status(home).get("record_frames", 0) >= 30, 20.0, "frames being saved")
        finally:
            proc.kill()
            proc.wait(timeout=5)
        folder = _folders(home)[0]
        assert (folder / "meta.part").exists() and not (folder / "frames.npz").exists()
        bridges(phone)                                    # the next bridge finishes what the dead one left
        _wait(lambda: (folder / "meta.json").exists(), 5.0, "the folder being finished")
        got = list(stream._replay_frames(folder, loop=False, paced=False))
        # all but the last second or two: what had not reached the disk when it died
        assert len(got) >= 5 and got[0].depth.shape == (48, 64) and got[-1].rgb.any()
    finally:
        phone.stop()


# --- the command line ------------------------------------------------------------------

def test_start_status_stop_from_the_command_line(home, phone):
    tool = [sys.executable, str(ROOT / "tools" / "phone_bridge.py")]
    opts = ["--no-usb", "--phone-ip", "127.0.0.1", "--port", str(phone.port), "--record-dir", str(home / "recordings")]
    assert subprocess.run(tool + ["status"], capture_output=True, text=True).returncode == 2      # not running
    try:
        r = subprocess.run(tool + ["start"] + opts, capture_output=True, text=True, timeout=30)
        assert r.returncode == 0 and "streaming" in r.stdout, r.stdout + r.stderr
        r = subprocess.run(tool + ["start"] + opts, capture_output=True, text=True, timeout=30)
        assert r.returncode == 0 and "already running" in r.stdout
        r = subprocess.run(tool + ["status", "--json"], capture_output=True, text=True)
        assert r.returncode == 0 and json.loads(r.stdout)["state"] == "streaming"
        assert phone.stats()["connections"] == 1
        assert (home / "phone_bridge.log").read_text().count("phone bridge up") == 1
    finally:
        r = subprocess.run(tool + ["stop"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "stopped" in r.stdout and not stream.bridge_running(home)
    assert _status(home)["state"] == "stopped"
    _wait(lambda: phone.stats()["clients"] == 0, 2.0, "the phone seeing the bridge leave")


# --- nothing spins -----------------------------------------------------------------------

def test_a_reader_waiting_on_a_silent_bridge_uses_no_cpu(home, phone, bridges):
    """The class of bug that starved Whisper: a wait that returns at once, in a loop."""
    bridges(phone, redial_s=0)
    s = Stream("bridge").start()
    try:
        _frames(s, 2)
        phone.set_mode("stall")
        _wait_state(home, "stalled", 2.0)
        cpu, late = time.process_time(), 0.0
        for _ in range(300):                              # the stall meter: how late does a 1 ms sleep wake?
            t = time.perf_counter()
            time.sleep(0.001)
            late = max(late, time.perf_counter() - t - 0.001)
        cpu = time.process_time() - cpu                   # this process holds reader, bridge and pretend phone
        assert cpu < 0.25 and late < 0.05, (cpu, late)
        assert s.status()["stalled"] and s.status()["state"] == "streaming"
        assert "sent no picture" in s.status()["message"]
    finally:
        s.close()


# --- the direct reader's own patience ------------------------------------------------------

def test_the_direct_reader_gives_up_on_a_silent_phone_after_two_seconds(monkeypatch):
    class Silent:
        is_connected = True

        def __init__(self, *a, **kw):
            pass

        def start(self):
            return True

        def wait_for_frame(self, timeout=1.0):
            time.sleep(timeout)
            return None

        def stop(self):
            pass
    monkeypatch.setitem(sys.modules, "sdk", SimpleNamespace(IPhoneSensorClient=Silent))
    assert stream.PATHON_GIVE_UP_S == {"usb": 2.0, "wifi": 4.0} and stream.STALL_S == 1.0
    t = time.perf_counter()
    with pytest.raises(SourceError, match="no frame for"):
        next(stream._pathon_frames("192.0.2.1", give_up_s=0.5))
    assert 0.5 <= time.perf_counter() - t < 1.0


def test_the_direct_reader_reads_the_pretend_phone_with_the_real_sdk(phone, monkeypatch):
    pytest.importorskip("sdk")
    phone.port and monkeypatch.setenv("STREAM_PHONE_IP", "127.0.0.1")
    gen = stream._pathon_frames("127.0.0.1", port=phone.port)
    try:
        raw = next(gen)
        assert raw.rgb.shape == (240, 320, 3) and raw.depth.shape == (48, 64) and raw.kind == "pathon"
        phone.set_mode("stall")
        t = time.perf_counter()
        with pytest.raises(SourceError, match="no frame for 4 s"):
            for raw in gen:
                pass
        assert 3.5 < time.perf_counter() - t < 5.5
    finally:
        gen.close()
