"""A stand-in for the iPhone's PathOn app: speaks its stream well enough for the real SDK client
and for tools/phone_bridge.py, and can be told to misbehave. No phone, no cable.

    python tools/fake_phone.py recordings/live_2025 --port 8888      # a recording, looped
    python tools/fake_phone.py synthetic --port 9888 --fps 30         # made-up frames
        --size 1920x1440      send pictures of this size (a recording is stored at 960 wide)
        --control 9887        take commands on this port while running (see below)
        --close-on-disconnect stop listening when the last client leaves, until `tap`
        --grace 3             ... but only after this many seconds without a client
        --one-client          turn away a second client

Then read it like the phone on Wi-Fi: STREAM_SOURCE=pathon:127.0.0.1 (the SDK's port is 8888), or
through the bridge: python tools/phone_bridge.py run --no-usb --phone-ip 127.0.0.1 --port 9888

Commands, one per line on the control port (python tools/fake_phone.py ctl 9887 stall):

    stream | refuse | drop | stall     send frames | close the port | accept then hang up | stay
                                       connected and send nothing
    fps N                              change the frame rate
    corrupt [jpeg|garbage|header]      send one broken frame, then carry on
    restart [SECONDS]                  hang up on everyone and close the port for a while
    tap                                start streaming again (after --close-on-disconnect)
    stats                              connections so far, clients now, frames sent

Every frame carries time.monotonic() at the moment it was sent as its timestamp. On one machine
that clock is shared by all programs, so a reader can tell exactly how long a frame took to arrive.

Stand-ins for the USB helper programs, so the bridge's USB path can be exercised without a phone
(a JSON file gives the pretend state: {"usb": "ok" | "absent" | "untrusted" | "locked", "port": N}):

    PHONE_BRIDGE_IDEVICE_ID="python tools/fake_phone.py idevice_id STATE.json"
    PHONE_BRIDGE_IDEVICEPAIR="python tools/fake_phone.py idevicepair STATE.json"
    PHONE_BRIDGE_IPROXY="python tools/fake_phone.py iproxy STATE.json"
"""
from __future__ import annotations

import json
import select
import socket
import struct
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MAGIC = b"REC3D"
_HEAD_V2 = struct.Struct("<5sBIdHHIIHHII6x")     # the 48-byte header of protocol version 2


def pack(frame_id: int, t: float, K, pose, depth, jpeg: bytes, conf=None, rgb_size=(0, 0)) -> bytes:
    """One frame as PathOn sends it. depth is float32 metres, conf one byte per depth pixel
    (0, 1, 2), K and pose row by row as float32, rgb_size is (width, height)."""
    import numpy as np
    depth = np.ascontiguousarray(depth, dtype=np.float32)
    conf = b"" if conf is None else np.ascontiguousarray(conf, dtype=np.uint8).tobytes()
    imu = struct.pack("<7d", 0.0, -9.81, 0.0, 0.0, 0.0, 0.0, t)
    head = _HEAD_V2.pack(MAGIC, 2, frame_id & 0xFFFFFFFF, t, depth.shape[1], depth.shape[0], depth.nbytes,
                         len(jpeg), rgb_size[0], rgb_size[1], len(conf), len(imu))
    pose = np.zeros((4, 4), np.float32) if pose is None else np.asarray(pose, dtype=np.float32)
    return b"".join((head, np.asarray(K, dtype=np.float32).tobytes(), pose.tobytes(), depth.tobytes(), jpeg, conf, imu))


def body_of(rgb, depth, K, pose, quality: int = 80) -> dict:
    """Everything of a frame that does not change from one sending to the next, encoded once."""
    import cv2
    import numpy as np
    ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    depth = np.ascontiguousarray(depth, dtype=np.float32)
    return {"K": np.asarray(K, dtype=np.float32), "pose": pose, "depth": depth, "jpeg": jpeg.tobytes(),
            "conf": np.where(depth > 0, 2, 0).astype(np.uint8), "rgb_size": (rgb.shape[1], rgb.shape[0])}


def synthetic_frames(n: int = 30, size=(1920, 1440), depth_size=(256, 192)) -> list:
    """Made-up frames with enough detail that the JPEG is as heavy as a camera's: a textured
    wall, a box that moves, the frame number as a row of bright pixels. Depth is a tilted
    plane with a hole in it (no lidar reading there)."""
    import cv2
    import numpy as np
    w, h = size
    dw, dh = depth_size
    rng = np.random.default_rng(1)
    wall = cv2.resize(rng.integers(40, 215, (h // 8, w // 8, 3), dtype=np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
    K = np.array([[w * 0.74, 0, w / 2], [0, w * 0.74, h / 2], [0, 0, 1]])
    out = []
    for i in range(n):
        rgb = wall.copy()
        x = int((w - w // 5) * (0.5 + 0.4 * np.sin(2 * np.pi * i / n)))
        rgb[h // 3:h // 3 + h // 5, x:x + w // 5] = (230, 60, 40)
        rgb[:8, :8 * (i + 1)] = 255
        depth = np.tile(np.linspace(0.6, 2.4, dh, dtype=np.float32)[:, None], (1, dw))
        depth[dh // 4:dh // 2, dw // 8:dw // 4] = 0.0
        pose = np.eye(4)
        pose[0, 3] = 0.01 * i
        out.append(body_of(rgb, depth, K, pose))
    return out


def recording_frames(folder, n: int = 60, size=None) -> list:
    """The first n frames of a recording, scaled to `size` (width, height) if given: a
    recording is stored at 960 wide, the phone sends 1920 x 1440."""
    import cv2
    import numpy as np
    sys.path.insert(0, str(ROOT))
    import stream
    out = []
    for raw in stream._replay_frames(folder, loop=False, paced=False):
        rgb, K = raw.rgb, np.array(raw.K, dtype=np.float64)
        if size and (rgb.shape[1], rgb.shape[0]) != tuple(size):
            K[0] *= size[0] / rgb.shape[1]
            K[1] *= size[1] / rgb.shape[0]
            rgb = cv2.resize(rgb, tuple(size), interpolation=cv2.INTER_LINEAR)
        out.append(body_of(rgb, raw.depth, K, raw.pose))
        if len(out) >= n:
            break
    if not out:
        raise SystemExit(f"no frames in {folder}")
    return out


class FakePhone:
    """The server. One thread sends frames to every client at the chosen rate, takes new
    clients and notices the ones that left."""

    def __init__(self, frames: list, port: int = 0, host: str = "127.0.0.1", fps: float = 30.0,
                 close_on_disconnect: bool = False, grace_s: float = 0.0, one_client: bool = False):
        self.frames, self.host, self.fps = frames, host, fps
        self.close_on_disconnect, self.grace_s, self.one_client = close_on_disconnect, grace_s, one_client
        self.mode = "stream"
        self.connections = self.turned_away = self.frames_sent = self.disconnects = 0
        self.needs_tap = False
        self._served = False                 # a client has connected since the last tap
        self._corrupt = None
        self._down_until = 0.0
        self._empty_since = None
        self._clients: list[socket.socket] = []
        self._listener = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.port = port
        self._open()

    # --- switches ----------------------------------------------------------------
    def set_mode(self, mode: str):
        assert mode in ("stream", "refuse", "drop", "stall"), mode
        self.mode = mode

    def corrupt(self, kind: str = "jpeg"):
        assert kind in ("jpeg", "garbage", "header"), kind
        self._corrupt = kind

    def restart(self, down_s: float = 1.0):
        self._down_until = time.monotonic() + down_s

    def tap(self):
        """Someone taps Start Streaming."""
        self.needs_tap = self._served = False

    def stats(self) -> dict:
        return {"mode": self.mode, "fps": self.fps, "port": self.port, "connections": self.connections,
                "clients": len(self._clients), "disconnects": self.disconnects, "turned_away": self.turned_away,
                "frames_sent": self.frames_sent, "listening": self._listener is not None, "needs_tap": self.needs_tap}

    # --- the server --------------------------------------------------------------
    def _open(self):
        if self._listener is None:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.host, self.port))
            s.listen(8)
            self.port = s.getsockname()[1]
            self._listener = s

    def _close(self):
        if self._listener is not None:
            self._listener.close()
            self._listener = None

    def _hang_up(self, conn, left: bool = True):
        with self._lock:
            if conn in self._clients:
                self._clients.remove(conn)
                self.disconnects += left
        try:
            conn.close()
        except OSError:
            pass

    def start(self) -> "FakePhone":
        self._thread = threading.Thread(target=self._run, daemon=True, name="fake-phone")
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        for conn in list(self._clients):
            self._hang_up(conn, left=False)
        self._close()

    def _packet(self, i: int) -> bytes:
        f = self.frames[i % len(self.frames)]
        kind, self._corrupt = self._corrupt, None
        if kind == "garbage":
            return bytes(range(256)) * 40                       # not a packet at all
        jpeg = f["jpeg"]
        if kind == "jpeg":
            jpeg = jpeg[:200] + bytes(len(jpeg) - 200)           # right size, no picture in it
        data = pack(i, time.monotonic(), f["K"], f["pose"], f["depth"], jpeg, f["conf"], f["rgb_size"])
        if kind == "header":
            data = data[:22] + struct.pack("<I", 0x7FFFFFFF) + data[26:]     # a depth size that cannot be true
        return data

    def _run(self):
        i, due = 0, time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            down = now < self._down_until or self.mode == "refuse" or self.needs_tap
            if down:
                self._close()
                if now < self._down_until:
                    for conn in list(self._clients):
                        self._hang_up(conn, left=False)
            else:
                self._open()
            watch = ([self._listener] if self._listener is not None else []) + list(self._clients)
            wait = max(0.0, min(due - now, 0.05)) if self.mode == "stream" and self._clients else 0.02
            try:
                ready, _, _ = select.select(watch, [], [], wait)
            except (OSError, ValueError):
                ready = []
            for s in ready:
                if s is self._listener:
                    try:
                        conn, _ = s.accept()
                    except OSError:
                        continue
                    if self.mode == "drop" or (self.one_client and self._clients):
                        self.turned_away += 1
                        conn.close()
                        continue
                    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    conn.settimeout(2.0)
                    self.connections += 1
                    self._served = True
                    with self._lock:
                        self._clients.append(conn)
                else:
                    try:
                        gone = not s.recv(4096)                  # clients send nothing: readable means it left
                    except OSError:
                        gone = True
                    if gone:
                        self._hang_up(s)
            now = time.monotonic()
            # Like the real app (as far as its behaviour could be seen): once the last client
            # has been gone for a moment it stops streaming, and wants a tap.
            if self._clients or not self._served:
                self._empty_since = None
            elif self.close_on_disconnect and not self.needs_tap:
                self._empty_since = self._empty_since or now
                if now - self._empty_since >= self.grace_s:
                    self.needs_tap, self._empty_since = True, None
            if self.mode != "stream" or not self._clients:
                due = max(due, now)
                continue
            if now < due:
                continue
            data = self._packet(i)
            for conn in list(self._clients):
                try:
                    conn.sendall(data)
                except OSError:
                    self._hang_up(conn)
            i += 1
            self.frames_sent += 1
            due = max(due + 1.0 / self.fps, now - 0.1)           # keep the rate, without a burst after a pause

    # --- commands ----------------------------------------------------------------
    def command(self, line: str) -> dict:
        word, *rest = line.split() or ["stats"]
        if word in ("stream", "refuse", "drop", "stall"):
            self.set_mode(word)
        elif word == "fps":
            self.fps = float(rest[0])
        elif word == "corrupt":
            self.corrupt(rest[0] if rest else "jpeg")
        elif word == "restart":
            self.restart(float(rest[0]) if rest else 1.0)
        elif word == "tap":
            self.tap()
        elif word != "stats":
            return {"error": f"unknown command: {line}"}
        return self.stats()

    def serve_commands(self, port: int):
        srv = socket.create_server(("127.0.0.1", port))
        srv.settimeout(0.5)
        while not self._stop.is_set():
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            with conn:
                conn.settimeout(2.0)
                try:
                    for line in conn.makefile():
                        conn.sendall(json.dumps(self.command(line.strip())).encode() + b"\n")
                except OSError:
                    pass
        srv.close()


def ctl(port: int, line: str) -> dict:
    """Send one command to a running fake phone and return its answer."""
    with socket.create_connection(("127.0.0.1", port), timeout=3.0) as s:
        s.sendall(line.encode() + b"\n")
        return json.loads(s.makefile().readline())


# ---------------------------------------------------------------------------
# Stand-ins for idevice_id, idevicepair and iproxy. They import nothing heavy: the bridge
# starts them often.

def _usb_state(path: str) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {"usb": "absent"}


def _fake_iproxy(state_file: str, local: int):
    """Like iproxy: takes every connection, and hangs up at once when the phone is not listening."""
    def pump(a, b):
        try:
            while data := a.recv(1 << 16):
                b.sendall(data)
        except OSError:
            pass
        for s in (a, b):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    srv = socket.create_server(("127.0.0.1", local))
    print(f"Creating listening port {local} for device port 8888", flush=True)
    while True:
        conn, _ = srv.accept()
        st = _usb_state(state_file)
        try:
            if st.get("usb") != "ok":
                raise OSError("no device")
            phone = socket.create_connection(("127.0.0.1", int(st["port"])), timeout=1.0)
        except OSError:
            print("Error connecting to device: Success", flush=True)
            conn.close()
            continue
        phone.settimeout(None)
        threading.Thread(target=pump, args=(phone, conn), daemon=True).start()
        threading.Thread(target=pump, args=(conn, phone), daemon=True).start()


def _shim(argv: list[str]) -> int:
    tool, state_file, *rest = argv
    st = _usb_state(state_file)
    usb = st.get("usb", "absent")
    if tool == "idevice_id":
        if usb == "absent":
            print("ERROR: Unable to retrieve device list!", file=sys.stderr)
            return 255
        if usb != "unlisted":
            print("00008130-000000000000FAKE")
        return 0
    if tool == "idevicepair":
        udid = "00008130-000000000000FAKE"
        print({"ok": f"SUCCESS: Validated pairing with device {udid}",
               "locked": f"ERROR: Could not validate with device {udid} because a passcode is set. "
                         "Please enter the passcode on the device and retry.",
               "untrusted": f"ERROR: Please accept the trust dialog on the screen of device {udid}, "
                            "then attempt to pair again."}.get(usb, "No device found."))
        return 0 if usb == "ok" else 1
    _fake_iproxy(state_file, int(rest[0]))
    return 0


def main(argv: list[str]) -> int:
    if argv and argv[0] in ("idevice_id", "idevicepair", "iproxy"):
        return _shim(argv)
    if argv and argv[0] == "ctl":
        print(json.dumps(ctl(int(argv[1]), " ".join(argv[2:]))))
        return 0
    import argparse
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="a recording folder, or: synthetic")
    p.add_argument("--port", type=int, default=8888)
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to be reachable from other machines")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--size", default=None, help="WIDTHxHEIGHT of the pictures sent, e.g. 1920x1440")
    p.add_argument("--frames", type=int, default=60, help="how many different frames to cycle through")
    p.add_argument("--control", type=int, default=None, help="port for commands")
    p.add_argument("--close-on-disconnect", action="store_true")
    p.add_argument("--grace", type=float, default=0.0)
    p.add_argument("--one-client", action="store_true")
    a = p.parse_args(argv)
    size = tuple(int(v) for v in a.size.lower().split("x")) if a.size else None
    frames = (synthetic_frames(a.frames, size or (1920, 1440)) if a.source == "synthetic"
              else recording_frames(a.source, a.frames, size))
    phone = FakePhone(frames, a.port, a.host, a.fps, a.close_on_disconnect, a.grace, a.one_client).start()
    w, h = frames[0]["rgb_size"]
    print(f"[fake phone] {len(frames)} frames of {w}x{h}, {len(frames[0]['jpeg']) // 1024} KB each, "
          f"on {a.host}:{phone.port} at {a.fps:g} a second", flush=True)
    try:
        if a.control:
            phone.serve_commands(a.control)
        else:
            while True:
                time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        phone.stop()
    return 0


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    sys.exit(main(sys.argv[1:]))
