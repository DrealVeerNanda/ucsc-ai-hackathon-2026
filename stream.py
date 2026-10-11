"""Lane 1: frames from the iPhone (colour + lidar depth) for the rest of the pipeline.

    import stream
    frame = stream.get_frame()      # newest Frame, or None if nothing new since the last call

Where frames come from is set by stream.start(SOURCE) or the STREAM_SOURCE environment variable:

    auto                   whichever phone app is streaming: PathOn over USB, PathOn over Wi-Fi if
                           STREAM_PHONE_IP gives the phone's address, then Record3D (default)
    record3d               Record3D app over USB
    pathon | pathon:IP     free "Record3DStream - PathOn" app over USB | Wi-Fi
    relay | relay:PORT     frames forwarded from a Mac by tools/relay_send.py
    recordings/NAME        a saved recording, looped and paced like a live camera

Frames arrive upright, with depth resized to the rgb size and K matching it, so nothing
downstream has to think about resolution or rotation. Depth 0 means no lidar reading there.
Camera axes are x right, y down, z forward.

Frame.t_recv is time.monotonic() when the frame reached this process. Record3D sends no
capture timestamp, so Frame.t is the same value: use it for durations, not as a phone clock.
"""
from __future__ import annotations

import atexit
import json
import os
import socket
import struct
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

import tracing
from interfaces import Frame


@dataclass
class RawFrame:
    """A frame as the app sent it: not rotated yet, depth still at its native size."""
    rgb: np.ndarray
    depth: np.ndarray            # float32 metres, 0 = no reading
    K: np.ndarray                # 3 x 3, for the rgb size
    pose: np.ndarray | None      # 4 x 4, as the app reports it
    kind: str = "record3d"       # which app sent it: picks the entry in ROTATE and POSE_FIX
    t_src: float | None = None   # seconds on the sender's clock (phone, or the relaying Mac), if it sends one


class SourceError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Rotation and pose conventions

_CV_ROTATE = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
# Camera coordinates after the image is rotated clockwise -> camera coordinates before.
_ROT = {
    0: np.eye(3),
    90: np.array([[0.0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
    180: np.diag([-1.0, -1, 1]),
    270: np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]]),
}
_FLIP_YZ = np.diag([1.0, -1.0, -1.0])

# Clockwise rotation that makes each app's frames upright on a portrait chest mount.
# Check with `python tools/probe_stream.py --show` (r cycles the rotation).
ROTATE = {"record3d": 0, "pathon": 90}
# Turns the pose each app reports (ARKit camera axes: x right, y up, z backward, on the
# landscape sensor image) into camera-to-world for our axes on the frame as the app sent it.
# These are derived from the ARKit convention, not yet measured on a phone:
# `python tools/check_pose.py` measures it and prints the matrix that belongs here.
POSE_FIX = {"record3d": _FLIP_YZ @ _ROT[90], "pathon": _FLIP_YZ}


def rotate_K(K: np.ndarray, w: int, h: int, degrees: int) -> np.ndarray:
    """Intrinsics after rotating a w x h image clockwise by 0/90/180/270 degrees."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if degrees == 90:
        fx, fy, cx, cy = fy, fx, (h - 1) - cy, cx
    elif degrees == 180:
        cx, cy = (w - 1) - cx, (h - 1) - cy
    elif degrees == 270:
        fx, fy, cx, cy = fy, fx, cy, (w - 1) - cx
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def shrink(raw: RawFrame, max_side: int | None) -> RawFrame:
    """Downscale rgb (and K with it) so its longer side is at most max_side."""
    h, w = raw.rgb.shape[:2]
    if not max_side or max(h, w) <= max_side:
        return raw
    s = max_side / max(h, w)
    nw, nh = int(round(w * s)), int(round(h * s))
    K = np.array(raw.K, dtype=np.float64)
    K[0] *= nw / w
    K[1] *= nh / h
    rgb = cv2.resize(raw.rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    return RawFrame(rgb, raw.depth, K, raw.pose, raw.kind, raw.t_src)


def prepare(raw: RawFrame, t: float, rotate: int | None = None, max_side: int | None = 960,
            pose_fix: np.ndarray | None = None) -> Frame:
    """RawFrame -> Frame: depth resized to rgb (nearest neighbour), both rotated upright,
    K and pose adjusted to match. rotate/pose_fix default to the app's entry above."""
    if rotate is None:
        rotate = ROTATE.get(raw.kind, 0)
    if pose_fix is None:
        pose_fix = POSE_FIX.get(raw.kind, _FLIP_YZ)
    raw = shrink(raw, max_side)
    rgb, depth = raw.rgb, raw.depth
    K = np.array(raw.K, dtype=np.float64)
    h, w = rgb.shape[:2]
    if depth.shape[:2] != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
    pose = None
    if raw.pose is not None:
        pose = np.array(raw.pose, dtype=np.float64)
        pose[:3, :3] = pose[:3, :3] @ pose_fix @ _ROT[rotate]
    if rotate:
        rgb = cv2.rotate(rgb, _CV_ROTATE[rotate])
        depth = cv2.rotate(depth, _CV_ROTATE[rotate])
        K = rotate_K(K, w, h, rotate)
    return Frame(np.ascontiguousarray(rgb), np.ascontiguousarray(depth, dtype=np.float32), K, pose, t, t_recv=t)


def _clean_depth(depth, conf=None, min_conf: int = 1) -> np.ndarray:
    depth = np.array(depth, dtype=np.float32, copy=True)
    if conf is not None and conf.shape == depth.shape and conf.size:
        depth[conf < min_conf] = 0.0
    depth[~np.isfinite(depth)] = 0.0
    return depth


def _valid_pose(pose) -> np.ndarray | None:
    """None for the all-zero or non-finite matrices apps send before tracking starts."""
    if pose is None:
        return None
    pose = np.array(pose, dtype=np.float64).reshape(4, 4)
    if not np.isfinite(pose).all() or abs(np.linalg.det(pose[:3, :3]) - 1.0) > 0.05:
        return None
    return pose


# ---------------------------------------------------------------------------
# Sources. Each is a generator of RawFrame that raises SourceError with a readable message.

def _record3d_frames(device_index: int = 0, timeout_s: float = 3.0) -> Iterator[RawFrame]:
    """iPhone RGB + lidar over USB via the Record3D app (adapted from the record3d demo).

    Needs: Record3D app on the phone with "USB Streaming mode" on, the rear lidar camera
    selected, the red button pressed, and the phone plugged in and trusted.
    """
    try:
        from record3d import Record3DStream
    except Exception as e:
        raise SourceError(f"record3d package not importable ({e}). On the GX10 build it from source "
                          "(see README); on a Mac: brew install cmake && pip install record3d") from e
    try:
        devs = Record3DStream.get_connected_devices()
    except Exception as e:
        raise SourceError(f"record3d could not list devices: {e}") from e
    if len(devs) <= device_index:
        raise SourceError(
            f"No Record3D device at index {device_index} ({len(devs)} found). Check: phone plugged in "
            "by USB and trusted, Record3D app open, Settings > 'USB Streaming mode' on, and the red "
            "record button pressed to start streaming.")
    event = threading.Event()
    stopped = []

    def on_stopped():
        stopped.append(True)
        event.set()

    session = Record3DStream()
    session.on_new_frame = event.set
    session.on_stream_stopped = on_stopped
    if not session.connect(devs[device_index]):
        raise SourceError("Record3D: the phone is on USB but the app is not streaming. Open Record3D, turn on "
                          "Settings > 'USB Streaming mode', keep the phone unlocked and press the red button. "
                          "(Or another program is already connected to it.)")
    try:
        truedepth = session.get_device_type() == 0  # TrueDepth = 0, LiDAR = 1
        if truedepth:
            print("[stream] Record3D is using the front TrueDepth camera; select the rear lidar "
                  "camera in the app.", file=sys.stderr)
        while True:
            if not event.wait(timeout_s):
                raise SourceError(f"Record3D: no frame for {timeout_s:.0f} s. Is streaming started in the app?")
            event.clear()
            if stopped:
                raise SourceError("Record3D: stream stopped (cable unplugged or app closed).")
            rgb = np.array(session.get_rgb_frame(), copy=True)  # already RGB order
            try:
                conf = np.asarray(session.get_confidence_frame())
            except Exception:
                conf = None
            depth = _clean_depth(session.get_depth_frame(), conf)
            if truedepth:
                depth, rgb = cv2.flip(depth, 1), cv2.flip(rgb, 1)
            c = session.get_intrinsic_mat()
            K = np.array([[c.fx, 0, c.tx], [0, c.fy, c.ty], [0, 0, 1]], dtype=np.float64)
            yield RawFrame(rgb, depth, K, _pose_from_quat(session.get_camera_pose()), "record3d")
    finally:
        try:
            session.disconnect()
        except Exception:
            pass


def _pose_from_quat(p) -> np.ndarray | None:
    """Record3D CameraPose (qx, qy, qz, qw, tx, ty, tz) -> 4 x 4, None if tracking has no pose."""
    x, y, z, w = p.qx, p.qy, p.qz, p.qw
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if not np.isfinite(n) or n < 1e-6:
        return None
    x, y, z, w = x / n, y / n, z / n, w / n
    T = np.eye(4)
    T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    T[:3, 3] = [p.tx, p.ty, p.tz]
    return _valid_pose(T)


def _kill_orphan_iproxy():
    """The PathOn SDK starts an `iproxy` tunnel and stops it when Python exits normally. If the
    program is killed instead, the tunnel is left holding the port and the next run cannot
    connect. Stop any such tunnel that no longer has a parent program."""
    import subprocess
    try:
        rows = subprocess.run(["ps", "-eo", "pid=,ppid=,comm="], capture_output=True, text=True).stdout.splitlines()
    except Exception:
        return
    procs = [r.split(None, 2) for r in rows]
    names = {p[0]: p[2].strip() for p in procs if len(p) == 3}
    killed = False
    for pid, ppid, name in (p for p in procs if len(p) == 3):
        # adopted by init or by systemd: whoever started it is gone
        if name.strip() == "iproxy" and (ppid == "1" or names.get(ppid) == "systemd"):
            try:
                os.kill(int(pid), 15)
                killed = True
            except OSError:
                pass
    if killed:
        time.sleep(0.3)


def _pathon_frames(host: str | None = None, port: int = 8888, timeout_s: float = 5.0) -> Iterator[RawFrame]:
    """iPhone RGB + lidar via the free "Record3DStream - PathOn" app (USB, or Wi-Fi with host).

    Uses that app's Python SDK, which is pure Python (no ARM build). Its licence allows personal,
    non-commercial use and forbids redistribution, so it is NOT copied into this repo.
    USB mode tunnels TCP through `iproxy` (libimobiledevice).
    """
    try:
        from sdk import IPhoneSensorClient
    except Exception as e:
        raise SourceError(f"PathOn SDK not importable ({e}). Install: pip install "
                          "'git+https://github.com/YoshiRi/Record3DStream#subdirectory=sdk'") from e
    usb = host is None
    if usb:
        # iproxy accepts local connections even with no phone attached, so the SDK would
        # report "Connected". Check that a phone is actually on USB first.
        import shutil
        import subprocess
        if shutil.which("idevice_id") is None:
            raise SourceError("PathOn USB needs libimobiledevice (macOS: brew install libimobiledevice; "
                              "Ubuntu: sudo apt install libimobiledevice-utils libusbmuxd-tools)")
        ids = subprocess.run(["idevice_id", "-l"], capture_output=True, text=True).stdout.split()
        if not ids:
            raise SourceError("PathOn: no iPhone on USB (`idevice_id -l` is empty). Plug in with a data "
                              "cable, unlock the phone and tap Trust.")
        _kill_orphan_iproxy()
    client = IPhoneSensorClient(host or "localhost", port, timeout=timeout_s, usb=usb)
    try:
        try:
            ok, err = client.start(), None
        except Exception as e:
            ok, err = False, e
        if not ok:
            how = ("USB: phone plugged in, unlocked and trusted, app open and streaming"
                   if usb else f"Wi-Fi: same network, IP {host}")
            raise SourceError(f"PathOn: could not connect ({err or 'see message above'}). Check {how}.")
        last = time.perf_counter()
        while True:
            fr = client.wait_for_frame(timeout=1.0)
            if fr is None:
                # A TCP connection is not proof of a stream: require real frames.
                if time.perf_counter() - last > timeout_s:
                    raise SourceError(f"PathOn: no frame for {timeout_s:.0f} s. Is the app streaming?")
                continue
            last = time.perf_counter()
            i = fr.intrinsics
            K = np.array([[i.fx, 0, i.ppx], [0, i.fy, i.ppy], [0, 0, 1]], dtype=np.float64)
            rgb = cv2.cvtColor(fr.color, cv2.COLOR_BGR2RGB)
            depth = _clean_depth(fr.depth, getattr(fr, "confidence", None))
            yield RawFrame(rgb, depth, K, _valid_pose(fr.transform), "pathon", getattr(fr, "timestamp", None))
    finally:
        try:
            client.stop()
        except Exception:
            pass


def _auto_frames() -> Iterator[RawFrame]:
    """Whichever app is streaming. The two are easy to mix up: the free one is called
    "Record3DStream - PathOn", the paid one "Record3D"."""
    errors = []
    wifi = os.environ.get("STREAM_PHONE_IP")          # PathOn streams over the cable or Wi-Fi, whichever
    sources = [_pathon_frames] + ([lambda: _pathon_frames(wifi)] if wifi else []) + [_record3d_frames]   # mode the app is in
    for source in sources:
        it = source()
        try:
            first = next(it)
        except SourceError as e:
            errors.append(str(e))
            continue
        yield first
        yield from it
        return
    raise SourceError("No phone app is streaming. " + " ALSO: ".join(errors))


def _replay_frames(folder: str | Path, loop: bool = True, paced: bool = True) -> Iterator[RawFrame]:
    """Replays a folder written by Recorder at the speed it was recorded."""
    folder = Path(folder)
    video, side = folder / "rgb.mp4", folder / "frames.npz"
    if not video.exists() or not side.exists():
        raise SourceError(f"Not a recording: {folder} (expected rgb.mp4 and frames.npz). "
                          "Make one with: python tools/record.py NAME")
    data = np.load(side)
    depth_mm, Ks, poses, ts = data["depth_mm"], data["K"], data["pose"], data["t"]
    meta = folder / "meta.json"
    kind = json.loads(meta.read_text()).get("kind", "record3d") if meta.exists() else "record3d"
    while True:
        cap = cv2.VideoCapture(str(video))
        start = time.perf_counter()
        try:
            for i in range(len(ts)):
                ok, bgr = cap.read()
                if not ok:
                    break
                if paced:
                    wait = start + float(ts[i] - ts[0]) - time.perf_counter()
                    if wait > 0:
                        time.sleep(wait)
                yield RawFrame(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), depth_mm[i].astype(np.float32) / 1000.0,
                               Ks[i], _valid_pose(poses[i]), kind)
        finally:
            cap.release()
        if not loop:
            return


# ---------------------------------------------------------------------------
# Mac relay (fallback if record3d will not build on the GX10): tools/relay_send.py on the Mac
# connects to the GX10 and sends one message per frame. Colour is JPEG; depth is a 16-bit PNG
# in millimetres, which is lossless.

RELAY_PORT = 9099
_HEAD = struct.Struct("<4sIII")  # magic, json bytes, jpeg bytes, png bytes
_MAGIC = b"BFP1"


def _to_mm(depth: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(depth * 1000.0), 0, 65535).astype(np.uint16)


def encode_raw(raw: RawFrame, jpeg_quality: int = 85) -> bytes:
    meta = json.dumps({"kind": raw.kind, "t_send": time.time(), "K": np.asarray(raw.K, dtype=np.float64).ravel().tolist(),
                       "pose": None if raw.pose is None else np.asarray(raw.pose, dtype=np.float64).ravel().tolist()}).encode()
    _, jpg = cv2.imencode(".jpg", cv2.cvtColor(raw.rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    _, png = cv2.imencode(".png", _to_mm(raw.depth), [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return _HEAD.pack(_MAGIC, len(meta), len(jpg), len(png)) + meta + jpg.tobytes() + png.tobytes()


def decode_raw(meta: bytes, jpg: bytes, png: bytes) -> RawFrame:
    m = json.loads(meta)
    bgr = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    mm = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
    if bgr is None or mm is None:
        raise SourceError("relay: could not decode a frame")
    pose = None if m["pose"] is None else np.array(m["pose"], dtype=np.float64).reshape(4, 4)
    return RawFrame(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), mm.astype(np.float32) / 1000.0,
                    np.array(m["K"], dtype=np.float64).reshape(3, 3), pose, m.get("kind", "record3d"),
                    m.get("t_send"))


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        try:
            chunk = conn.recv(min(1 << 20, n - len(buf)))
        except socket.timeout as e:
            raise SourceError("relay: no data from the Mac for 3 s") from e
        except OSError as e:
            raise SourceError(f"relay: connection error ({e})") from e
        if not chunk:
            raise SourceError("relay: the Mac disconnected")
        buf += chunk
    return bytes(buf)


def _relay_frames(port: int, stop: threading.Event, wait_s: float = 60.0) -> Iterator[RawFrame]:
    try:
        srv = socket.create_server(("", port))
    except OSError as e:
        raise SourceError(f"relay: cannot listen on port {port} ({e})") from e
    conn = None
    try:
        srv.settimeout(0.5)
        print(f"[stream] waiting for tools/relay_send.py to connect on port {port}", file=sys.stderr)
        deadline = time.time() + wait_s
        while conn is None:
            if stop.is_set() or time.time() > deadline:
                raise SourceError(f"relay: nothing connected on port {port} in {wait_s:.0f} s. On the Mac run: "
                                  "python tools/relay_send.py GX10_IP")
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                pass
        conn.settimeout(3.0)
        while True:
            magic, n_meta, n_jpg, n_png = _HEAD.unpack(_recv_exact(conn, _HEAD.size))
            if magic != _MAGIC:
                raise SourceError("relay: bad data on the port (is something else connected to it?)")
            yield decode_raw(_recv_exact(conn, n_meta), _recv_exact(conn, n_jpg), _recv_exact(conn, n_png))
    finally:
        if conn is not None:
            conn.close()
        srv.close()


# ---------------------------------------------------------------------------
class Recorder:
    """Writes raw frames to FOLDER/rgb.mp4 (colour) and FOLDER/frames.npz (depth in whole
    millimetres at its native size, plus K, pose and time for every frame), so the stream
    can replay them. Frames are stored before rotation, so the rotation can still be changed."""

    def __init__(self, folder: str | Path, fps: float = 20.0, max_side: int | None = 960):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.fps, self.max_side = fps, max_side
        self.writer = None
        self.kind = None
        self.depth, self.K, self.pose, self.t = [], [], [], []

    def add(self, raw: RawFrame, t: float):
        raw = shrink(raw, self.max_side)
        if self.writer is None:
            h, w = raw.rgb.shape[:2]
            self.writer = cv2.VideoWriter(str(self.folder / "rgb.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
            if not self.writer.isOpened():
                raise SourceError(f"could not open {self.folder / 'rgb.mp4'} for writing")
            self.kind = raw.kind
        self.writer.write(cv2.cvtColor(raw.rgb, cv2.COLOR_RGB2BGR))
        self.depth.append(_to_mm(raw.depth))
        self.K.append(np.array(raw.K, dtype=np.float64))
        self.pose.append(np.full((4, 4), np.nan) if raw.pose is None else np.array(raw.pose, dtype=np.float64))
        self.t.append(t)

    def close(self) -> dict:
        if self.writer is None:
            return {"frames": 0}
        self.writer.release()
        np.savez_compressed(self.folder / "frames.npz", depth_mm=np.stack(self.depth), K=np.stack(self.K),
                            pose=np.stack(self.pose), t=np.array(self.t))
        seconds = self.t[-1] - self.t[0]
        meta = {"kind": self.kind, "frames": len(self.t), "seconds": round(seconds, 2),
                "fps": round((len(self.t) - 1) / seconds, 1) if seconds > 0 else 0.0,
                "depth_size": list(self.depth[0].shape[::-1]),
                "has_pose": bool(np.isfinite(np.stack(self.pose)).all(axis=(1, 2)).any())}
        (self.folder / "meta.json").write_text(json.dumps(meta, indent=2))
        return meta


# ---------------------------------------------------------------------------
class Stream:
    """Reads a source on a background thread and keeps only the newest frame, so a slow loop
    never works on an old one. Once frames have flowed, a lost source (cable bump, app closed)
    is retried every second until it comes back."""

    def __init__(self, source: str = "auto", rotate: int | None = None, max_side: int | None = 960,
                 reconnect: bool = True, loop: bool = True):
        self.source = str(source)
        self.rotate, self.max_side = rotate, max_side
        self.reconnect, self.loop = reconnect, loop
        self.reconnects = 0
        self._cond = threading.Condition()
        self._latest = None  # (RawFrame, time.monotonic() at arrival)
        self._seq = self._taken = 0
        self._offsets = deque(maxlen=600)   # arrival time minus the sender's timestamp, recent frames
        self._cable_ms = None
        self._arrivals = deque(maxlen=60)
        self._state, self._error = "connecting", None
        self._stop = threading.Event()
        self._thread = None

    def _frames(self) -> Iterator[RawFrame]:
        kind, _, arg = self.source.partition(":")
        if kind == "auto":
            return _auto_frames()
        if kind == "record3d":
            return _record3d_frames(int(arg or 0))
        if kind == "pathon":
            return _pathon_frames(arg or None)
        if kind == "relay":
            return _relay_frames(int(arg or RELAY_PORT), self._stop)
        return _replay_frames(self.source, loop=self.loop)

    def _set(self, state, error=None):
        with self._cond:
            self._state, self._error = state, error
            self._cond.notify_all()

    def _run(self):
        while not self._stop.is_set():
            gen = self._frames()
            try:
                for raw in gen:
                    if self._stop.is_set():
                        break
                    now = time.monotonic()
                    with self._cond:
                        if self._state == "reconnecting":
                            print(f"[stream] {self.source} is back", file=sys.stderr)
                        self._latest = (raw, now)
                        self._cable_ms = self._delay_ms(raw, now)
                        self._seq += 1
                        self._arrivals.append(now)
                        self._state, self._error = "streaming", None
                        self._cond.notify_all()
                else:
                    self._set("ended")
                    return
            except Exception as e:
                msg = str(e) if isinstance(e, SourceError) else f"{self.source}: {e!r}"
                if self._seq == 0 or not self.reconnect:
                    self._set("failed", msg)
                    return
                if self._state == "streaming":
                    print(f"[stream] lost: {msg} Retrying.", file=sys.stderr)
                    self.reconnects += 1
                self._set("reconnecting", msg)
            finally:
                gen.close()
            self._stop.wait(1.0)

    def _delay_ms(self, raw: RawFrame, now: float):
        """How much longer this frame took to get here than the quickest recent one. The two
        clocks are not synchronised, so the absolute delay is unknown: this is the part of it
        that varies (queueing, a busy phone, a slow link). tools/measure_camera_latency.py
        measures the absolute number."""
        if raw.t_src is None:
            return None
        self._offsets.append(now - raw.t_src)
        return (self._offsets[-1] - min(self._offsets)) * 1000.0

    def start(self, timeout: float | None = None) -> "Stream":
        """Blocks until the first frame. Raises SourceError if the source cannot be opened."""
        if timeout is None:
            timeout = 65.0 if self.source.startswith("relay") else 10.0
        self._thread = threading.Thread(target=self._run, daemon=True, name="stream")
        self._thread.start()
        deadline = time.time() + timeout
        with self._cond:
            while self._seq == 0 and self._state == "connecting":
                left = deadline - time.time()
                if left <= 0:
                    break
                self._cond.wait(left)
            ok, err = self._seq > 0, self._error
        if not ok:
            self.close()
            raise SourceError(err or f"{self.source}: no frame within {timeout:.0f} s")
        tracing.meta(source=self.source)
        return self

    def get_raw(self, wait: float = 0.0):
        """Newest (RawFrame, arrival time) not handed out yet, or None. Waits up to `wait` seconds."""
        with self._cond:
            if self._seq == self._taken and wait > 0:
                self._cond.wait(wait)
            if self._seq == self._taken:
                return None
            self._skipped = self._seq - self._taken - 1    # arrived and were overwritten unseen
            self._taken = self._seq
            self._taken_cable_ms = self._cable_ms
            return self._latest

    def get_frame(self, wait: float = 0.0) -> Frame | None:
        tracing.loop_returned()
        got = self.get_raw(wait)
        if got is None:
            return None
        raw, t = got
        tracing.begin_frame(t, wait_ms=round((time.monotonic() - t) * 1000, 3), skipped=self._skipped,
                            **({} if self._taken_cable_ms is None else {"cable_ms": round(self._taken_cable_ms, 3)}))
        with tracing.span("stream.prepare"):
            return prepare(raw, t, rotate=self.rotate, max_side=self.max_side)

    def status(self) -> dict:
        """state is connecting, streaming, reconnecting, ended or failed. age_s is how old the
        newest frame is: show it, and distrust anything computed from a frame older than ~0.3 s."""
        with self._cond:
            a, latest = list(self._arrivals), self._latest
            fps = (len(a) - 1) / (a[-1] - a[0]) if len(a) > 1 and a[-1] > a[0] else 0.0
            return {"source": self.source, "state": self._state, "error": self._error, "fps": round(fps, 1),
                    "age_s": None if latest is None else time.monotonic() - latest[1],
                    "frames": self._seq, "reconnects": self.reconnects,
                    "raw_rgb": None if latest is None else latest[0].rgb.shape[:2],
                    "raw_depth": None if latest is None else latest[0].depth.shape[:2]}

    def close(self):
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=5.0)


# ---------------------------------------------------------------------------
# The interface the other lanes use.

_default: Stream | None = None


def start(source: str | None = None, **kw) -> Stream:
    """Open the stream. Without arguments: STREAM_SOURCE (default auto) and STREAM_ROTATE."""
    global _default
    stop()
    if "rotate" not in kw and os.environ.get("STREAM_ROTATE"):
        kw["rotate"] = int(os.environ["STREAM_ROTATE"])
    _default = Stream(source or os.environ.get("STREAM_SOURCE", "auto"), **kw).start()
    return _default


# Join the reader thread before Python shuts down. Left running, it can be torn down in the
# middle of a decode, which aborts the process on Linux ("FATAL: exception not rethrown").
atexit.register(lambda: stop())


_retry_at = 0.0
_complained = None


def get_frame(wait: float = 0.0) -> Frame | None:
    """Latest frame, or None if nothing new. `wait` (seconds) blocks for the next frame
    instead of returning None at once, so a loop does not have to spin.

    If the camera cannot be opened (the phone app is not streaming yet), this says so once and
    returns None, then tries again every two seconds: the loop keeps running and picks the
    stream up when it appears."""
    global _retry_at, _complained
    if _default is None:
        if time.monotonic() < _retry_at:
            return None
        try:
            start()
            _complained = None
        except SourceError as e:
            _retry_at = time.monotonic() + 2.0
            if str(e) != _complained:
                _complained = str(e)
                print(f"[stream] {e} Waiting for it.", file=sys.stderr)
            return None
    return _default.get_frame(wait)


def status() -> dict:
    return _default.status() if _default is not None else {"state": "closed"}


def stop():
    global _default
    if _default is not None:
        _default.close()
        _default = None
