"""Lane 1 stream: frame preparation, rotation maths, record/replay, reconnect, relay,
and the Record3D source failing cleanly without a phone."""
import socket
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

import stream
from interfaces import Frame
from stream import RawFrame, Recorder, SourceError, Stream, prepare

K0 = np.array([[500.0, 0, 170.0], [0, 510.0, 110.0], [0, 0, 1]])


@pytest.fixture(autouse=True)
def _no_phone_bridge(tmp_path, monkeypatch):
    """A phone bridge running on this machine must not be picked up by these tests."""
    monkeypatch.setenv("PHONE_BRIDGE_DIR", str(tmp_path / "no_bridge"))
    monkeypatch.setattr(stream, "_bridge_used", False)


def _pose(tx=0.1, ty=-0.2, tz=0.3, yaw=0.3):
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    T[:3, 3] = [tx, ty, tz]
    return T


def _raw(i=0, size=(320, 240), dsize=(64, 48), pose=True, kind="pathon"):
    w, h = size
    rgb = np.full((h, w, 3), 40 + i, np.uint8)
    depth = np.full((dsize[1], dsize[0]), 0.5 + 0.01 * i, np.float32)
    return RawFrame(rgb, depth, K0.copy(), _pose(tx=0.01 * i) if pose else None, kind)


def _xyz(u, v, fr):
    d = float(fr.depth[v, u])
    return np.array([(u - fr.K[0, 2]) * d / fr.K[0, 0], (v - fr.K[1, 2]) * d / fr.K[1, 1], d])


# --- prepare -------------------------------------------------------------------

def test_depth_is_resized_to_rgb_and_holes_stay_holes():
    raw = _raw()
    raw.depth[10:20, 10:20] = 0.0      # no lidar reading
    raw.depth[30:, :] = 2.0
    fr = prepare(raw, t=1.5, rotate=0)
    assert fr.rgb.shape == (240, 320, 3) and fr.rgb.dtype == np.uint8
    assert fr.depth.shape == (240, 320) and fr.depth.dtype == np.float32
    assert set(np.unique(fr.depth)) == {0.0, 0.5, 2.0}   # nearest neighbour: no blended values
    assert fr.depth[75, 75] == 0.0 and fr.depth[200, 100] == 2.0
    assert fr.t == 1.5 and fr.t_recv == 1.5 and np.allclose(fr.K, K0)


def test_big_frames_are_downscaled_and_K_follows():
    raw = _raw(size=(1920, 1440), dsize=(256, 192))
    raw.K = np.array([[1400.0, 0, 960.0], [0, 1400.0, 720.0], [0, 0, 1]])
    fr = prepare(raw, t=0, rotate=0, max_side=960)
    assert fr.rgb.shape == (720, 960, 3) and fr.depth.shape == (720, 960)
    assert np.allclose(fr.K, [[700, 0, 480], [0, 700, 360], [0, 0, 1]])


def test_rotation_keeps_rgb_and_depth_aligned():
    raw = _raw(size=(160, 120), dsize=(64, 48))
    raw.rgb[:] = 0
    raw.rgb[0:10, 0:20] = 255           # marker at top-left
    raw.depth[:] = 0
    raw.depth[0:4, 0:8] = 1.0           # same marker in depth
    fr = prepare(raw, t=0, rotate=90)
    assert fr.rgb.shape == (160, 120, 3) and fr.depth.shape == (160, 120)
    # clockwise: top-left goes to top-right, in both images
    assert fr.rgb[0:20, -10:].mean() == 255 and fr.depth[0:20, -10:].mean() == 1.0
    assert fr.depth[40:, :].max() == 0.0


@pytest.mark.parametrize("deg", [90, 180, 270])
def test_a_pixel_maps_to_the_same_world_point_after_rotation(deg):
    w, h = 320, 240
    raw = _raw(size=(w, h), dsize=(w, h))
    raw.depth = np.random.default_rng(0).uniform(0.4, 2.0, (h, w)).astype(np.float32)
    fix = np.eye(3)
    a = prepare(raw, t=0, rotate=0, pose_fix=fix)
    b = prepare(raw, t=0, rotate=deg, pose_fix=fix)
    for u, v in [(10, 20), (300, 30), (170, 110), (250, 230)]:
        u2, v2 = {90: (h - 1 - v, u), 180: (w - 1 - u, h - 1 - v), 270: (v, w - 1 - u)}[deg]
        assert b.depth[v2, u2] == a.depth[v, u]
        pa, pb = _xyz(u, v, a), _xyz(u2, v2, b)
        assert pb[2] == pytest.approx(pa[2])
        assert np.allclose(a.pose[:3, :3] @ pa + a.pose[:3, 3], b.pose[:3, :3] @ pb + b.pose[:3, 3], atol=1e-6)


def test_rotation_and_pose_fix_default_to_the_app():
    for kind in ("record3d", "pathon"):
        raw = _raw(kind=kind)
        fr = prepare(raw, t=0)
        sideways = stream.ROTATE[kind] in (90, 270)
        assert fr.rgb.shape[:2] == ((320, 240) if sideways else (240, 320))
        assert np.linalg.det(fr.pose[:3, :3]) == pytest.approx(1.0)   # still a rotation
        assert np.allclose(fr.pose[:3, 3], raw.pose[:3, 3])


def test_missing_pose_stays_none():
    assert prepare(_raw(pose=False), t=0).pose is None
    assert stream._valid_pose(np.zeros((4, 4))) is None
    assert stream._valid_pose(np.full((4, 4), np.nan)) is None


# --- record and replay ---------------------------------------------------------

def _record(folder, n=12, dt=0.05, **kw):
    rec = Recorder(folder)
    raws = [_raw(i, **kw) for i in range(n)]
    for i, r in enumerate(raws):
        rec.add(r, 100.0 + i * dt)
    return raws, rec.close()


def test_recording_replays_with_exact_depth_K_and_pose(tmp_path):
    raws, meta = _record(tmp_path / "rec")
    assert meta["frames"] == 12 and meta["has_pose"] and meta["kind"] == "pathon"
    assert meta["fps"] == pytest.approx(20.0, abs=0.1)
    got = list(stream._replay_frames(tmp_path / "rec", loop=False, paced=False))
    assert len(got) == 12
    for a, b in zip(raws, got):
        assert b.kind == "pathon" and b.rgb.shape == a.rgb.shape
        assert np.abs(b.depth - a.depth).max() < 0.0006          # whole millimetres
        assert np.allclose(b.K, a.K)
        assert abs(int(b.rgb[0, 0, 0]) - int(a.rgb[0, 0, 0])) <= 8   # colour is lossy, slightly
    assert np.allclose(got[5].pose, _pose(tx=0.05))


def test_a_run_killed_before_saving_is_packed_when_the_folder_is_next_opened(tmp_path):
    rec = Recorder(tmp_path / "rec")
    raws = [_raw(i) for i in range(8)]
    for i, r in enumerate(raws):
        rec.add(r, 100.0 + i * 0.05)
    rec.writer.release()                                   # the video survives a Ctrl-C; close() never ran
    rec._depth.flush()
    rec._index.flush()
    with open(tmp_path / "rec" / "depth.part", "ab") as f:
        f.write(b"\x01" * 1000)                            # and the kill came half way through a frame
    assert not (tmp_path / "rec" / "frames.npz").exists()
    got = list(stream._replay_frames(tmp_path / "rec", loop=False, paced=False))
    assert len(got) == 8 and np.abs(got[7].depth - raws[7].depth).max() < 0.0006 and np.allclose(got[3].K, raws[3].K)
    assert (tmp_path / "rec" / "meta.json").exists() and not (tmp_path / "rec" / "depth.part").exists()


def test_recording_keeps_frames_without_pose(tmp_path):
    rec = Recorder(tmp_path / "rec")
    rec.add(_raw(0, pose=False), 0.0)
    rec.add(_raw(1), 0.1)
    assert rec.close()["has_pose"]
    a, b = list(stream._replay_frames(tmp_path / "rec", loop=False, paced=False))
    assert a.pose is None and b.pose is not None


def test_replay_is_paced_like_the_recording(tmp_path):
    _record(tmp_path / "rec", n=11, dt=0.05)   # 0.5 s
    t = time.perf_counter()
    n = sum(1 for _ in stream._replay_frames(tmp_path / "rec", loop=False))
    assert n == 11 and 0.45 < time.perf_counter() - t < 0.9


# --- Stream --------------------------------------------------------------------

def test_get_frame_returns_each_frame_once_then_none(tmp_path):
    _record(tmp_path / "rec", n=6, dt=0.05)
    s = Stream(str(tmp_path / "rec"), loop=False).start()
    try:
        fr = s.get_frame(wait=1.0)
        assert isinstance(fr, Frame) and fr.depth.shape == fr.rgb.shape[:2]
        assert 0 <= time.monotonic() - fr.t_recv < 1.0 and fr.t == fr.t_recv
        assert s.get_frame() is None                 # nothing new yet: does not block
        seen = 1
        while (fr := s.get_frame(wait=0.5)) is not None:
            seen += 1
        assert 2 <= seen <= 6 and s.status()["state"] == "ended"
    finally:
        s.close()


def test_replay_loops_forever_by_default(tmp_path):
    _record(tmp_path / "rec", n=4, dt=0.02)
    s = Stream(str(tmp_path / "rec")).start()
    try:
        time.sleep(0.5)
        st = s.status()
        assert st["state"] == "streaming" and st["frames"] > 8 and st["fps"] > 10
        assert st["age_s"] < 0.3 and st["raw_depth"] == (48, 64)
    finally:
        s.close()


def test_module_get_frame_uses_the_environment(tmp_path, monkeypatch):
    _record(tmp_path / "rec", n=4, dt=0.02)
    monkeypatch.setenv("STREAM_SOURCE", str(tmp_path / "rec"))
    monkeypatch.setenv("STREAM_ROTATE", "180")
    try:
        fr = stream.get_frame(wait=1.0)
        assert fr is not None and fr.rgb.shape == (240, 320, 3)
        assert np.allclose(fr.K[:2, 2], [319 - 170.0, 239 - 110.0])   # rotated 180
        assert stream.status()["state"] == "streaming"
    finally:
        stream.stop()
    assert stream.status()["state"] == "closed"


def test_missing_recording_is_a_clear_error(tmp_path):
    with pytest.raises(SourceError, match="Not a recording"):
        Stream(str(tmp_path / "nope")).start()


def test_stream_reconnects_after_the_source_drops(monkeypatch):
    calls = []

    def frames(self):
        calls.append(1)

        def gen():
            if len(calls) == 2:
                raise SourceError("still unplugged")
            for i in range(3):
                yield _raw(i)
                time.sleep(0.01)
            raise SourceError("cable bumped")
        return gen()

    monkeypatch.setattr(Stream, "_frames", frames)
    monkeypatch.setattr(threading.Event, "wait", lambda self, timeout=None: time.sleep(0.02))
    s = Stream("record3d").start()
    try:
        deadline = time.time() + 3
        while len(calls) < 4 and time.time() < deadline:
            time.sleep(0.01)
        assert len(calls) >= 4, "kept retrying through a failed reconnect"
        assert s.reconnects >= 2 and s.get_frame(wait=0.5) is not None
    finally:
        s._stop.set()


def test_no_reconnect_before_the_first_frame(monkeypatch):
    def frames(self):
        raise SourceError("no phone")
        yield

    monkeypatch.setattr(Stream, "_frames", frames)
    with pytest.raises(SourceError, match="no phone"):
        Stream("record3d").start()


# --- Record3D ------------------------------------------------------------------

class _FakeRecord3D:
    """Stands in for record3d.Record3DStream: three frames, then the stream stops."""
    devices = [SimpleNamespace(product_id=1, udid="x")]

    def __init__(self):
        self.on_new_frame = self.on_stream_stopped = None
        self.disconnected = False

    @staticmethod
    def get_connected_devices():
        return _FakeRecord3D.devices

    def connect(self, dev):
        def feed():
            for _ in range(3):
                time.sleep(0.02)
                self.on_new_frame()
            time.sleep(0.02)
            self.on_stream_stopped()
        threading.Thread(target=feed, daemon=True).start()
        return True

    def disconnect(self):
        self.disconnected = True

    def get_device_type(self):
        return 1

    def get_rgb_frame(self):
        return np.full((960, 720, 3), 7, np.uint8)

    def get_depth_frame(self):
        d = np.full((256, 192), 0.75, np.float32)
        d[0, 0] = np.nan
        return d

    def get_confidence_frame(self):
        c = np.full((256, 192), 2, np.uint8)
        c[5, 5] = 0
        return c

    def get_intrinsic_mat(self):
        return SimpleNamespace(fx=700.0, fy=701.0, tx=360.0, ty=480.0)

    def get_camera_pose(self):
        return SimpleNamespace(qx=0.0, qy=0.0, qz=0.0, qw=1.0, tx=0.1, ty=0.2, tz=0.3)


def test_record3d_frames_are_parsed(monkeypatch):
    monkeypatch.setitem(sys.modules, "record3d", SimpleNamespace(Record3DStream=_FakeRecord3D))
    got = []
    with pytest.raises(SourceError, match="stream stopped"):
        for raw in stream._record3d_frames():
            got.append(raw)
    assert len(got) >= 1
    raw = got[0]
    assert raw.kind == "record3d" and raw.rgb.shape == (960, 720, 3)
    assert raw.depth[0, 0] == 0.0 and raw.depth[5, 5] == 0.0 and raw.depth[100, 100] == 0.75
    assert np.allclose(raw.K, [[700, 0, 360], [0, 701, 480], [0, 0, 1]])
    assert np.allclose(raw.pose[:3, :3], np.eye(3)) and np.allclose(raw.pose[:3, 3], [0.1, 0.2, 0.3])
    fr = prepare(raw, t=0)
    assert fr.depth.shape == fr.rgb.shape[:2]


def test_record3d_without_phone_raises_source_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "record3d", SimpleNamespace(Record3DStream=_FakeRecord3D))
    monkeypatch.setattr(_FakeRecord3D, "devices", [])
    with pytest.raises(SourceError, match="No Record3D device"):
        Stream("record3d").start()


def test_pathon_without_phone_raises_source_error(monkeypatch):
    import subprocess
    monkeypatch.setitem(sys.modules, "sdk", SimpleNamespace(IPhoneSensorClient=object))
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/" + name)
    # simulate "no iPhone on USB" regardless of what is plugged in right now
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=""))
    with pytest.raises(SourceError, match="no iPhone"):
        Stream("pathon").start()


# --- relay ---------------------------------------------------------------------

def test_relay_message_roundtrip_keeps_depth_exact():
    raw = _raw(3)
    raw.depth = np.random.default_rng(1).uniform(0, 4, raw.depth.shape).astype(np.float32)
    msg = stream.encode_raw(raw)
    magic, a, b, c = stream._HEAD.unpack(msg[:stream._HEAD.size])
    body = msg[stream._HEAD.size:]
    out = stream.decode_raw(body[:a], body[a:a + b], body[a + b:a + b + c])
    assert magic == stream._MAGIC and out.kind == "pathon" and out.rgb.shape == raw.rgb.shape
    assert np.abs(out.depth - raw.depth).max() < 0.0006
    assert np.allclose(out.K, raw.K) and np.allclose(out.pose, raw.pose)


def test_relay_delivers_frames_over_a_socket():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    def send():
        for _ in range(50):
            try:
                c = socket.create_connection(("127.0.0.1", port), timeout=1)
                break
            except OSError:
                time.sleep(0.05)
        with c:
            for i in range(5):
                c.sendall(stream.encode_raw(_raw(i, pose=False)))
                time.sleep(0.03)

    threading.Thread(target=send, daemon=True).start()
    s = Stream(f"relay:{port}", reconnect=False).start()
    try:
        fr = s.get_frame(wait=1.0)
        assert fr is not None and fr.pose is None and fr.depth.shape == fr.rgb.shape[:2]
        deadline = time.time() + 5
        while s.status()["state"] == "streaming" and time.time() < deadline:
            time.sleep(0.05)
        assert s.status()["state"] == "failed" and "disconnected" in s.status()["error"]
    finally:
        s.close()


def test_auto_uses_whichever_app_is_streaming(monkeypatch):
    # PathOn is not streaming (no phone answers), Record3D is
    import subprocess
    monkeypatch.setitem(sys.modules, "sdk", SimpleNamespace(IPhoneSensorClient=object))
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=""))
    monkeypatch.setitem(sys.modules, "record3d", SimpleNamespace(Record3DStream=_FakeRecord3D))
    s = Stream("auto", reconnect=False).start()
    try:
        fr = s.get_frame(wait=1.0)
        assert fr is not None and fr.rgb.shape == (960, 720, 3)
    finally:
        s.close()


def test_auto_reports_both_reasons_when_nothing_streams(monkeypatch):
    import subprocess
    monkeypatch.setitem(sys.modules, "sdk", SimpleNamespace(IPhoneSensorClient=object))
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=""))
    monkeypatch.setitem(sys.modules, "record3d", SimpleNamespace(Record3DStream=_FakeRecord3D))
    monkeypatch.setattr(_FakeRecord3D, "devices", [])
    with pytest.raises(SourceError, match="no iPhone.*No Record3D device"):
        Stream("auto").start()


def test_get_frame_waits_for_a_camera_that_is_not_there_yet(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("STREAM_SOURCE", str(tmp_path / "rec"))        # does not exist yet
    monkeypatch.setattr(stream, "_retry_at", 0.0)
    monkeypatch.setattr(stream, "_complained", None)
    try:
        assert stream.get_frame() is None and stream.get_frame() is None     # no exception: the loop lives
        assert capsys.readouterr().err.count("Not a recording") == 1          # said once, not every frame
        _record(tmp_path / "rec", n=4, dt=0.02)                               # the camera appears
        monkeypatch.setattr(stream, "_retry_at", 0.0)                         # (two seconds later)
        assert stream.get_frame(wait=1.0) is not None
    finally:
        stream.stop()


def test_a_live_session_can_be_saved_while_it_is_used(tmp_path, monkeypatch):
    def frames(self):
        def gen():
            for i in range(40):
                yield _raw(i)
                time.sleep(0.02)
        return gen()
    monkeypatch.setattr(Stream, "_frames", frames)
    monkeypatch.setenv("STREAM_RECORD", str(tmp_path / "session"))
    s = Stream("pathon", reconnect=False).start()
    try:
        seen = 0
        while s.status()["state"] == "streaming":
            seen += s.get_frame(wait=0.1) is not None
    finally:
        s.close()
    assert seen > 5                                             # the pipeline still got its frames
    got = list(stream._replay_frames(tmp_path / "session", loop=False, paced=False))
    assert 5 <= len(got) <= 12                                  # about 10 a second for 0.8 s
    assert got[0].depth.shape == (48, 64) and got[0].kind == "pathon"


def test_finishing_a_recording_never_holds_up_the_frames(tmp_path, monkeypatch):
    """Packing a long recording takes many seconds. It used to happen in the reader, with the
    lock held, so the loop got no frame until it was done."""
    def frames(self):
        def gen():
            for i in range(120):
                yield _raw(i)
                time.sleep(0.02)
        return gen()
    real_pack, packing = stream.pack, threading.Event()

    def slow_pack(folder):
        packing.set()
        time.sleep(1.5)
        return real_pack(folder)
    monkeypatch.setattr(Stream, "_frames", frames)
    monkeypatch.setattr(stream, "pack", slow_pack)
    monkeypatch.setenv("STREAM_RECORD", str(tmp_path / "session"))
    monkeypatch.setenv("STREAM_RECORD_SECONDS", "0.3")
    monkeypatch.setenv("STREAM_PACK_WAIT_S", "0.1")
    s = Stream("pathon", reconnect=False).start()
    try:
        assert packing.wait(2.0), "the recording should have ended by itself"
        gaps, last = [], time.perf_counter()
        while time.perf_counter() - last < 0.5 and len(gaps) < 25:        # while it is being packed
            if s.get_frame(wait=0.2) is not None:
                gaps.append(time.perf_counter() - last)
                last = time.perf_counter()
        assert len(gaps) == 25 and max(gaps) < 0.2, gaps
    finally:
        t = time.perf_counter()
        s.close()
    assert time.perf_counter() - t < 1.0                    # closing does not sit through the packing either
    assert (tmp_path / "session" / "meta.part").exists()    # abandoned half way, and nothing is lost:
    time.sleep(1.6)
    assert len(list(stream._replay_frames(tmp_path / "session", loop=False, paced=False))) >= 2


def test_replaying_a_recording_is_never_recorded_again(tmp_path, monkeypatch):
    _record(tmp_path / "rec", n=4, dt=0.02)
    monkeypatch.setenv("STREAM_RECORD", str(tmp_path / "copy"))
    s = Stream(str(tmp_path / "rec"), loop=False).start()
    s.get_frame(wait=0.5)
    s.close()
    assert not (tmp_path / "copy" / "frames.npz").exists()


def test_pathon_gives_up_at_once_when_the_connection_has_closed(monkeypatch):
    """iproxy accepts the connection and drops it when the app is not streaming. The SDK's
    wait_for_frame() then returns None without waiting: the reader must not spin on it."""
    class Closed:
        is_connected = False
        calls = 0

        def __init__(self, *a, **kw):
            pass

        def start(self):
            return True

        def wait_for_frame(self, timeout=1.0):
            Closed.calls += 1
            return None

        def stop(self):
            pass
    monkeypatch.setitem(sys.modules, "sdk", SimpleNamespace(IPhoneSensorClient=Closed))
    t = time.perf_counter()
    with pytest.raises(SourceError, match="closed the connection"):
        next(stream._pathon_frames("192.0.2.1"))
    assert time.perf_counter() - t < 0.5 and Closed.calls == 1
