"""Hold the connection to the iPhone's PathOn app, and never let go of it.

PathOn stops streaming when the program reading it goes away, and someone then has to tap Start
on the phone again. So the pipeline, which is restarted often, should not be that program. This
one is: it does nothing but read the phone, and hands each frame to whoever asks through shared
memory (stream.py reads it with STREAM_SOURCE=bridge, or auto while a bridge is running).

    python tools/phone_bridge.py start          # start it in the background, wait until it is up
    python tools/phone_bridge.py status         # one line: what the phone link is doing (--json for all)
    python tools/phone_bridge.py stop           # stop it cleanly
    python tools/phone_bridge.py run            # run it here in the foreground (Ctrl-C stops it)
    python tools/phone_bridge.py record on|off  # start or stop saving what the phone sends

It reaches the phone over the USB cable (it runs its own `iproxy`), and over Wi-Fi when the cable
does not answer and STREAM_PHONE_IP (or --phone-ip) gives the phone's address. When the link
breaks it dials again by itself, and it says what is wrong in words a person can act on.

Once a second, and at once when something changes, it writes out/status/phone.json:

    t, state, ok, message        the contract every long-lived part follows
    state                        starting | no_phone | untrusted | app_not_streaming | connecting |
                                 streaming | stalled | stopped
    via                          usb | wifi (null when not connected)
    fps, frame_age_s             frames a second over the last two seconds; age of the newest frame
    depth_valid                  share of the newest depth image that has a lidar reading
    frames, reconnects           frames since it started; times a working stream was lost
    bad_frames, skipped          frames that could not be decoded; frames dropped to stay current
    readers                      programs reading frames from it now
    usb_present                  an iPhone is on a USB port, by the kernel's own list (null: cannot tell here)
    usb_detaches, usb_reattaches times it left the port, and times it came back, since the bridge started:
                                 a cable that is being tugged shows up here even when the stream recovers
    recording, record_folder, record_frames, record_error, disk_free_gb

Recording: with --record (or PHONE_BRIDGE_RECORD=1) it saves about ten frames a second into
recordings/demo_YYYYMMDD_HHMMSS/, a new folder every 15 minutes (--record-minutes), each one a
recording stream.py can replay (STREAM_SOURCE=recordings/demo_...). `record on|off` switches it
while it runs by writing "on" or "off" to out/status/phone_record, which anything else may write
too; that file, when present, wins over the flag, also after a restart. It will not record with
less than 5 GB free on the disk, and says so in the status message. The video is written as
rgb.mkv, which stays readable if the bridge is killed, and the depth of a folder cut short is
packed when the bridge next starts or the folder is next replayed.

Other files, all in out/ (PHONE_BRIDGE_DIR or --dir moves them): phone_bridge.log (rotated at
1 MB, three old ones kept), phone_bridge.lock (one bridge at a time), phone_bridge.sock.
"""
from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import os
import queue
import select
import selectors
import shlex
import shutil
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("TRACE", "0")          # the pipeline's per-frame trace is not for this program
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import cv2  # noqa: E402
import numpy as np  # noqa: E402

import stream  # noqa: E402

log = logging.getLogger("phone_bridge")

# ---------------------------------------------------------------------------
# PathOn's stream: one packet per frame, little-endian (the layout is in its SDK's protocol.py).
#   header     "REC3D", version, frame number, phone time, depth width and height, depth bytes,
#              jpeg bytes; version 2 adds picture width and height, confidence bytes, imu bytes
#   then       K (3 x 3 float32), pose (4 x 4 float32), depth (float32 metres), the picture
#              (JPEG), confidence (one byte per depth pixel), imu
MAGIC = b"REC3D"
_HEAD = struct.Struct("<5sBIdHHII")
_HEAD_V2 = struct.Struct("<HHII")
_K_BYTES, _POSE_BYTES = 36, 64
_MAX_JPEG, _MAX_DEPTH_SIDE, _MAX_IMU = 8 << 20, 1024, 1024
_AS_RGB = getattr(cv2, "IMREAD_COLOR_RGB", None)        # OpenCV 4.10+: decode straight to RGB


class Packets:
    """Cuts the bytes from the app into whole packets. Bytes that are not a packet (a corrupt
    frame, a lost start) are skipped up to the next "REC3D" and counted in `bad`."""

    def __init__(self):
        self.buf = bytearray()
        self.bad = 0
        self._lost = False

    def _skip(self, n: int):
        del self.buf[:n]
        if not self._lost:
            self._lost = True
            self.bad += 1

    def feed(self, data: bytes) -> list:
        """Add bytes; returns the (header, packet) pairs they complete, oldest first."""
        buf = self.buf
        buf += data
        out = []
        while len(buf) >= 6:
            if buf[:5] != MAGIC:
                at = buf.find(MAGIC, 1)
                self._skip(at if at > 0 else max(len(buf) - 4, 0))     # "REC3" may be the start of the next one
                if at < 0:
                    break
                continue
            size = 48 if buf[5] >= 2 else 32
            if len(buf) < size:
                break
            _, version, frame_id, t, dw, dh, n_depth, n_jpeg = _HEAD.unpack_from(buf)
            w = h = n_conf = n_imu = 0
            if version >= 2:
                w, h, n_conf, n_imu = _HEAD_V2.unpack_from(buf, _HEAD.size)
            if (n_depth != dw * dh * 4 or max(dw, dh) > _MAX_DEPTH_SIDE or not 0 < n_jpeg <= _MAX_JPEG
                    or n_conf not in (0, dw * dh) or n_imu > _MAX_IMU):
                self._skip(1)                                           # a header that cannot be true
                continue
            total = size + _K_BYTES + _POSE_BYTES + n_depth + n_jpeg + n_conf + n_imu
            if len(buf) < total:
                break
            head = {"frame_id": frame_id, "t": t, "at": size, "depth": (dh, dw), "rgb": (h, w),
                    "n_jpeg": n_jpeg, "n_conf": n_conf}
            out.append((head, bytes(buf[:total])))
            del buf[:total]
            self._lost = False
        return out


def decode(head: dict, packet: bytes):
    """(rgb, depth, K, pose) as stream.RawFrame wants them, or None if the picture is broken.
    Depth and pose are cleaned exactly as the direct reader in stream.py cleans them."""
    at = head["at"]
    k = np.frombuffer(packet, np.float32, 9, at).reshape(3, 3)
    pose = np.frombuffer(packet, np.float32, 16, at + _K_BYTES).reshape(4, 4)
    at += _K_BYTES + _POSE_BYTES
    dh, dw = head["depth"]
    depth = np.frombuffer(packet, np.float32, dh * dw, at).reshape(dh, dw)
    at += dh * dw * 4
    jpeg = np.frombuffer(packet, np.uint8, head["n_jpeg"], at)
    at += head["n_jpeg"]
    conf = np.frombuffer(packet, np.uint8, dh * dw, at).reshape(dh, dw) if head["n_conf"] else None
    if _AS_RGB is not None:
        rgb = cv2.imdecode(jpeg, _AS_RGB)
    else:
        rgb = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
        rgb = None if rgb is None else cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    if rgb is None or (head["rgb"][1] and rgb.shape[:2] != head["rgb"]):
        return None
    K = np.array([[k[0, 0], 0, k[0, 2]], [0, k[1, 1], k[1, 2]], [0, 0, 1]], dtype=np.float64)
    return rgb, stream._clean_depth(depth, conf), K, stream._valid_pose(pose)


# ---------------------------------------------------------------------------
class DemoRecorder:
    """Saves what the phone sends, about ten frames a second, on its own thread so that a slow
    disk never delays a frame on its way to the readers. A new folder is started every
    `minutes`: no folder grows without end, and a kill costs at most the tail of the newest."""

    def __init__(self, root: Path, minutes: float = 15.0, fps: float = 10.0, min_free_gb: float = 5.0):
        self.root, self.minutes, self.fps, self.min_free_gb = Path(root), minutes, fps, min_free_gb
        self.want = False
        self.folder: Path | None = None      # the folder being written, None when not recording
        self.frames = 0                      # in that folder
        self.error: str | None = None
        self._rec = None
        self._t0 = self._last = 0.0
        self._queue: queue.Queue = queue.Queue(maxsize=4)
        self._stop = threading.Event()
        self._closing: list[threading.Thread] = []
        self._thread = threading.Thread(target=self._run, daemon=True, name="recorder")
        self._thread.start()

    def free_gb(self) -> float | None:
        try:
            at = self.root if self.root.exists() else self.root.parent
            return shutil.disk_usage(at).free / 2 ** 30
        except OSError:
            return None

    def offer(self, rgb, depth, K, pose, t_src, t: float):
        """Called for every frame; takes the ones that are due. The arrays must not change later."""
        if not self.want or t - self._last < 1.0 / self.fps - 0.005:
            return
        try:
            self._queue.put_nowait((stream.RawFrame(rgb, depth, K, pose, "pathon", t_src), t))
            self._last = t
        except queue.Full:
            pass                             # the disk is behind: skip this one rather than queue up

    def _room(self) -> bool:
        free = self.free_gb()
        if free is not None and free < self.min_free_gb:
            if self.error is None:
                log.warning("not recording: %.1f GB free, %.0f needed", free, self.min_free_gb)
            self.error = (f"NOT recording: only {free:.1f} GB free on the disk "
                          f"(it needs {self.min_free_gb:.0f}). Delete old folders in {self.root.name}/.")
            return False
        self.error = None
        return True

    def _run(self):
        while not self._stop.is_set():
            try:
                raw, t = self._queue.get(timeout=0.2)
            except queue.Empty:
                raw = None
            if not self.want:
                self._finish()
                continue
            if raw is None:
                continue
            try:
                if self._rec is None or t - self._t0 >= self.minutes * 60 or self.frames % 50 == 49:
                    if not self._room():
                        self._finish()
                        continue
                if self._rec is not None and t - self._t0 >= self.minutes * 60:
                    self._finish()
                if self._rec is None:
                    folder = self.root / time.strftime("demo_%Y%m%d_%H%M%S")
                    self._rec = stream.Recorder(folder, fps=self.fps, video="rgb.mkv")
                    self._t0, self.frames, self.folder = t, 0, folder
                    log.info("recording to %s", folder)
                self._rec.add(raw, t)
                self.frames = self._rec.frames
            except Exception as e:               # a full disk, a folder that cannot be made
                log.error("recording failed: %r", e)
                self.error = f"NOT recording: {e}"
                self._finish()
                self._stop.wait(5.0)
        self._finish()

    def _finish(self):
        """Close the folder being written. Packing its depth takes a while for a long take, so
        that happens on the side while the next folder is already being filled."""
        rec, self._rec, self.folder = self._rec, None, None
        if rec is not None:
            th = threading.Thread(target=_close_recording, args=(rec,), daemon=True, name="pack")
            th.start()
            self._closing = [c for c in self._closing if c.is_alive()] + [th]

    def pack_leftovers(self):
        """Finish the folders an earlier bridge left half done (it was killed while recording)."""
        for part in sorted(self.root.glob("demo_*/meta.part")):
            th = threading.Thread(target=_pack_folder, args=(part.parent,), daemon=True, name="pack")
            th.start()
            self._closing.append(th)

    def close(self, wait_s: float = 30.0):
        self.want = False
        self._stop.set()
        self._thread.join(timeout=5.0)
        end = time.monotonic() + wait_s          # cut short, a pack is finished at the next start
        for th in self._closing:
            th.join(timeout=max(0.0, end - time.monotonic()))


def _close_recording(rec):
    try:
        meta = rec.close()
        if meta.get("frames"):
            log.info("saved %d frames (%.0f s) to %s", meta["frames"], meta["seconds"], rec.folder)
    except Exception as e:
        log.error("could not finish %s: %r", rec.folder, e)


def _pack_folder(folder: Path):
    try:
        meta = stream.pack(folder)
        log.info("finished %s, cut short by a kill: %d frames", folder, meta["frames"])
    except Exception as e:
        log.warning("nothing to recover in %s: %s", folder, e)


# ---------------------------------------------------------------------------
def _command(env: str, default: str) -> list[str]:
    """A helper program's command line. The environment can replace it (tests use stand-ins)."""
    return shlex.split(os.environ.get(env) or default)


def _die_with_parent():
    """In the child, on Linux: be sent SIGTERM when the bridge dies, however it dies, so that
    no iproxy is ever left behind holding a connection to the phone."""
    try:
        import ctypes
        ctypes.CDLL(None).prctl(1, signal.SIGTERM)       # PR_SET_PDEATHSIG
    except Exception:
        pass


class UsbWatch:
    """Is an iPhone on a USB port? Read from the kernel's own list, which needs no helper
    program and no special rights, so it can be asked four times a second. On the evening
    before the demo the phone left the port six times; twice it was back within four seconds
    (a connector moving on the body), and every such blip is counted here."""

    def __init__(self):
        self.root = Path(os.environ.get("PHONE_BRIDGE_SYSFS") or "/sys/bus/usb/devices")
        self.present: bool | None = None         # None: this system has no such list (a Mac)
        self.detaches = self.reattaches = 0
        self._dev = None

    def poll(self) -> bool | None:
        if not self.root.is_dir():
            return None
        found = None
        for vendor in self.root.glob("*/idVendor"):
            try:
                dev = vendor.parent
                if vendor.read_text().strip() == "05ac" and (dev / "product").read_text().strip() in ("iPhone", "iPad"):
                    found = (dev.name, (dev / "devnum").read_text().strip(), (dev / "speed").read_text().strip())
                    break
            except OSError:
                continue                         # it left while we were reading
        if found != self._dev:
            if self._dev is not None:
                self.detaches += 1
                log.warning("the iPhone left USB port %s (%d times now)", self._dev[0], self.detaches)
            if found is not None:
                self.reattaches += self.detaches > 0
                log.info("an iPhone is on USB port %s at %s Mb/s", found[0], found[2])
            self._dev = found
        self.present = found is not None
        return self.present


class AlreadyRunning(RuntimeError):
    pass


class Bridge:
    """Dials the phone, reads it, decodes each frame once, puts it on the shared-memory board,
    tells the readers, and reports. run() does not return until stop() is called."""

    # Between attempts while the phone is not answering. Tonight's logs suggest PathOn keeps
    # its door open for only about three seconds after its client is gone (a new client 2.7 s
    # later was served, one 3.8 s later was refused), so after a blip every moment counts.
    RETRY_S = 0.25

    def __init__(self, folder: str | Path | None = None, port: int = 8888, phone_ip: str | None = None,
                 usb: bool = True, stall_s: float = stream.STALL_S, redial_s: float = 10.0,
                 record: bool = False, record_dir: str | Path | None = None, record_minutes: float = 15.0,
                 min_free_gb: float = 5.0):
        self.files = stream.bridge_files(folder)
        self.port, self.phone_ip, self.usb = port, phone_ip, usb
        # stall_s: no frame for this long is reported as stalled. redial_s: after this long the
        # silent connection is given up and dialled again (0: never), which is what brought the
        # stream back by itself once on the evening before the demo.
        self.stall_s, self.redial_s = stall_s, redial_s
        self.state, self.message, self.via = "starting", "The phone bridge is starting.", None
        self.frames = self.reconnects = self.connections = self.bad_frames = self.skipped = 0
        self.started = time.monotonic()
        self._since = self.started
        self._arrivals: deque = deque(maxlen=200)
        self._last_frame = None              # (time, depth, rgb shape) of the newest frame
        self._decode_ms = 0.0
        self._stop = threading.Event()
        self._changed = threading.Event()
        self._lock_fd = None
        self.board = None
        self._server = None
        self._readers: list[socket.socket] = []
        self._readers_lock = threading.Lock()
        self._iproxy = None
        self._iproxy_port = 0
        self._pairing_at, self._pairing = 0.0, None
        self.usb_watch = UsbWatch()
        self._unlisted_since = None
        self._record_flag = record
        self.recorder = DemoRecorder(Path(record_dir or ROOT / "recordings"), record_minutes, min_free_gb=min_free_gb)
        self._too_big = False

    # --- state -----------------------------------------------------------------
    def _set(self, state: str, message: str = "", via: str | None = None):
        if (state, message, via) != (self.state, self.message, self.via):
            if state != self.state:
                log.info("%s%s: %s", state, f" ({via})" if via else "", message)
                self._since = time.monotonic()
            self.state, self.message, self.via = state, message, via
            self._changed.set()

    def status(self) -> dict:
        now = time.monotonic()
        state, message, via, last = self.state, self.message, self.via, self._last_frame
        age = None if last is None else now - last[0]
        recent = [t for t in list(self._arrivals) if now - t <= 2.0]
        if state != "streaming" or not recent:
            fps = 0.0
        elif len(recent) > 1 and now - recent[0] < 1.9:          # the stream is younger than the two seconds
            fps = (len(recent) - 1) / (recent[-1] - recent[0])
        else:
            fps = len(recent) / 2.0
        slow = state == "streaming" and now - self._since > 3.0 and fps < 15.0
        if state == "streaming":
            message = f"Streaming over {'the USB cable' if via == 'usb' else 'Wi-Fi'} at {fps:.0f} frames a second."
            if slow:
                message += " That is slow: the sound will lag behind the hand." + (
                    " Use the USB cable." if via == "wifi" else "")
        elif state == "stalled":
            message = (f"The iPhone is connected but has sent no picture for {now - self._since + self.stall_s:.0f} s. "
                       "Look at the phone: PathOn must be open, on the screen and streaming.")
        rec = self.recorder
        if rec.want and rec.error:
            message += " " + rec.error
        return {"t": time.time(), "state": state, "ok": state == "streaming", "message": message,
                "via": via, "fps": round(fps, 1), "frame_age_s": None if age is None else round(age, 3),
                "depth_valid": None if last is None else round(float(np.count_nonzero(last[1])) / last[1].size, 3),
                "frames": self.frames, "reconnects": self.reconnects, "connections": self.connections,
                "bad_frames": self.bad_frames, "skipped": self.skipped, "slow": slow,
                "readers": len(self._readers), "since_s": round(now - self._since, 1),
                "usb_present": self.usb_watch.present, "usb_detaches": self.usb_watch.detaches,
                "usb_reattaches": self.usb_watch.reattaches,
                "uptime_s": round(now - self.started, 1), "decode_ms": round(self._decode_ms, 2),
                "rgb": None if last is None else list(last[2][:2]),
                "depth": None if last is None else list(last[1].shape),
                "recording": rec.folder is not None, "record_wanted": rec.want,
                "record_folder": None if rec.folder is None else str(rec.folder),
                "record_frames": rec.frames if rec.folder is not None else 0, "record_error": rec.error,
                "disk_free_gb": None if (free := rec.free_gb()) is None else round(free, 1),
                "pid": os.getpid(), "shm": None if self.board is None else self.board.name}

    def _report(self):
        """Once a second, and at once on a change: the status file, and the same line to readers."""
        path = self.files["status"]
        tmp = path.with_name(path.name + ".tmp")
        while True:
            self._changed.wait(1.0)
            self._changed.clear()
            self._read_record_switch()
            st = self.status()
            if self._stop.is_set():
                st.update(state="stopped", ok=False, message="The phone bridge is not running.", via=None)
            line = json.dumps(st)
            try:
                tmp.write_text(line)
                os.replace(tmp, path)
            except OSError as e:
                log.error("cannot write %s: %s", path, e)
            self._tell(line.encode() + b"\n")
            if self._stop.is_set():
                return

    def _read_record_switch(self):
        """out/status/phone_record says on or off; without the file the start flag decides."""
        try:
            word = self.files["record"].read_text().split()[0].lower()
            want = word in ("on", "1", "yes", "true")
        except (OSError, IndexError):
            want = self._record_flag
        if want != self.recorder.want:
            log.info("recording switched %s", "on" if want else "off")
            self.recorder.want = want

    # --- readers ---------------------------------------------------------------
    def _tell(self, data: bytes):
        """Send to every reader without ever waiting for one. A reader that cannot take it has
        stopped reading for good (its buffer holds an hour of frames): it is let go, and
        attaches again if it is still alive."""
        with self._readers_lock:
            readers = list(self._readers)
        for conn in readers:
            try:
                if conn.send(data) != len(data):
                    raise BlockingIOError
            except OSError:
                self._drop_reader(conn)

    def _drop_reader(self, conn):
        with self._readers_lock:
            if conn not in self._readers:
                return
            self._readers.remove(conn)
        try:
            conn.shutdown(socket.SHUT_RDWR)      # the serving thread sees this and closes it
        except OSError:
            pass
        log.info("a reader left (%d now)", len(self._readers))
        self._changed.set()

    def _serve(self):
        sel = selectors.DefaultSelector()
        sel.register(self._server, selectors.EVENT_READ)
        while not self._stop.is_set():
            for key, _ in sel.select(0.2):
                conn = key.fileobj
                if conn is self._server:
                    try:
                        conn, _ = self._server.accept()
                    except OSError:
                        continue
                    conn.setblocking(False)
                    try:
                        conn.send(json.dumps(self.status()).encode() + b"\n")   # where the board is, and how things are
                    except OSError:
                        conn.close()
                        continue
                    sel.register(conn, selectors.EVENT_READ)
                    with self._readers_lock:
                        self._readers.append(conn)
                    log.info("a reader attached (%d now)", len(self._readers))
                    self._changed.set()
                    continue
                try:
                    gone = not conn.recv(4096)       # readers send nothing: readable means closed
                except BlockingIOError:
                    continue
                except OSError:
                    gone = True
                if gone:
                    sel.unregister(conn)
                    self._drop_reader(conn)
                    conn.close()
        for key in list(sel.get_map().values()):
            key.fileobj.close()
        sel.close()

    # --- the phone -------------------------------------------------------------
    def _usb_devices(self) -> list[str]:
        """UDIDs of the iPhones on USB. Empty also when usbmuxd is not running, which on Ubuntu
        means the same thing: it is started when an iPhone is plugged in."""
        try:
            r = subprocess.run(_command("PHONE_BRIDGE_IDEVICE_ID", "idevice_id") + ["-l"],
                               capture_output=True, text=True, timeout=3.0)
            return r.stdout.split() if r.returncode == 0 else []
        except (OSError, subprocess.TimeoutExpired):
            return []

    def _pairing_problem(self):
        """(state, message) if this computer may not talk to the phone, else None. Asked at most
        every five seconds, and only while the app is not answering."""
        if time.monotonic() - self._pairing_at < 5.0:
            return self._pairing
        self._pairing_at = time.monotonic()
        try:
            r = subprocess.run(_command("PHONE_BRIDGE_IDEVICEPAIR", "idevicepair") + ["validate"],
                               capture_output=True, text=True, timeout=3.0)
            said = (r.stdout + r.stderr).lower()
        except (OSError, subprocess.TimeoutExpired):
            said = ""
        if "passcode" in said:
            p = ("untrusted", "The iPhone is locked. Unlock it with its passcode.")
        elif "trust dialog" in said and "denied" not in said:
            p = ("untrusted", "The iPhone is asking whether to trust this computer. Unlock it and tap Trust.")
        elif "denied" in said or "not paired" in said:
            p = ("untrusted", "The iPhone does not trust this computer. Unplug the cable, plug it in again, "
                              "unlock the phone and tap Trust.")
        else:
            p = None
        self._pairing = p
        return p

    def _stop_iproxy(self):
        proc, self._iproxy = self._iproxy, None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()

    def _start_iproxy(self):
        self._stop_iproxy()
        with socket.socket() as s:                   # a port of our own: nothing else can be squatting on it
            s.bind(("127.0.0.1", 0))
            self._iproxy_port = s.getsockname()[1]
        cmd = _command("PHONE_BRIDGE_IPROXY", "iproxy") + [str(self._iproxy_port), str(self.port)]
        self._iproxy = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        preexec_fn=_die_with_parent if sys.platform == "linux" else None)
        threading.Thread(target=self._drain, args=(self._iproxy,), daemon=True, name="iproxy-log").start()
        log.info("iproxy started on port %d (pid %d)", self._iproxy_port, self._iproxy.pid)

    def _drain(self, proc):
        """iproxy prints a line for every connection. Read them, or its pipe fills up and it
        hangs; keep the ones that say something new."""
        last, at = None, 0.0
        for line in proc.stdout:
            line = line.decode(errors="replace").strip()
            if line and (line != last or time.monotonic() - at > 60.0):
                log.info("iproxy: %s", line)
                last, at = line, time.monotonic()

    def _watch_usb(self):
        while not self._stop.wait(0.25):
            self.usb_watch.poll()

    def _dial_usb(self):
        """(socket, None) or (None, (state, message)). Nothing here opens a connection to the
        app only to close it again: the app may take any client leaving as the end of the session."""
        absent = ("no_phone", "No iPhone on the USB cable. Plug it into the GX10 with the data cable and unlock it.")
        if self.usb_watch.present is False:
            self._unlisted_since = None
            return None, absent
        if not self._usb_devices():
            if not self.usb_watch.present:
                return None, absent
            # On the port, but the phone has not let this computer in. For the first moments
            # after plugging in that is normal.
            self._unlisted_since = self._unlisted_since or time.monotonic()
            if time.monotonic() - self._unlisted_since < 2.0:
                return None, ("connecting", "The iPhone has just been plugged in.")
            return None, ("untrusted", "The iPhone is on the cable but does not answer this computer. "
                                       "Unlock it, and tap Trust if it asks.")
        self._unlisted_since = None
        try:
            if self._iproxy is None or self._iproxy.poll() is not None:
                self._start_iproxy()
        except OSError as e:
            return None, ("no_phone", f"Cannot run iproxy ({e}). Install libusbmuxd-tools.")
        sock, end = None, time.monotonic() + 2.0
        while sock is None and not self._stop.is_set():
            try:
                sock = socket.create_connection(("127.0.0.1", self._iproxy_port), timeout=2.0)
            except OSError:                          # iproxy is not listening yet, or has died
                if time.monotonic() > end or self._iproxy.poll() is not None:
                    self._stop_iproxy()              # a fresh one next time round
                    return None, ("connecting", "Starting the USB link to the iPhone.")
                time.sleep(0.05)
        if sock is None:
            return None, ("connecting", "")
        if self._hung_up(sock):
            return None, self._pairing_problem() or (
                "app_not_streaming", "The iPhone is on the cable but PathOn is not streaming. "
                                     "Open PathOn on the phone, choose USB and tap Start Streaming.")
        return sock, None

    @staticmethod
    def _hung_up(sock: socket.socket) -> bool:
        """iproxy takes a connection whether or not the app is listening, and drops it at once
        if it is not. Only the app's own bytes, or half a second of the line staying open, show
        that somebody is there. Closes the socket if nobody is."""
        try:
            ready, _, _ = select.select([sock], [], [], 0.5)
            gone = bool(ready) and sock.recv(1, socket.MSG_PEEK) == b""
        except OSError:
            gone = True
        if gone:
            sock.close()
        return gone

    def _dial_wifi(self):
        closed = ("app_not_streaming", "The iPhone answers on Wi-Fi but PathOn is not streaming. "
                                       "Open PathOn on the phone, choose WiFi and tap Start Streaming.")
        try:
            sock = socket.create_connection((self.phone_ip, self.port), timeout=1.5)
        except ConnectionRefusedError:
            return None, closed
        except OSError:
            return None, ("no_phone", f"The iPhone does not answer on Wi-Fi at {self.phone_ip}. "
                                      "Wake it, and check it is on the same network.")
        return (None, closed) if self._hung_up(sock) else (sock, None)

    def _dial(self):
        """(socket, via, when it was dialled), or None with the state saying why. The cable first, always."""
        problem = None
        if self.usb:
            t = time.monotonic()
            sock, problem = self._dial_usb()
            if sock is not None:
                return sock, "usb", t
        if self.phone_ip and not self._stop.is_set():
            t = time.monotonic()
            sock, wifi_problem = self._dial_wifi()
            if sock is not None:
                return sock, "wifi", t
            problem = problem or wifi_problem
        if problem is None:
            problem = ("no_phone", "Nothing to connect to: USB is switched off and no Wi-Fi address is given.")
        if problem[1]:
            self._set(*problem)
        return None

    def _receive(self, sock: socket.socket, via: str, last: float):
        """Read one connection until it ends. Returns why it ended and how many frames it gave.
        `last` is when it was dialled: silence is counted from there."""
        for opt, value in ((socket.SO_RCVBUF, 4 << 20), (socket.SO_KEEPALIVE, 1)):
            try:
                sock.setsockopt(socket.SOL_SOCKET, opt, value)
            except OSError:
                pass
        # Over Wi-Fi a phone that walks out of range says nothing: probes show it within seconds.
        for name, value in (("TCP_KEEPIDLE", 2), ("TCP_KEEPALIVE", 2), ("TCP_KEEPINTVL", 1), ("TCP_KEEPCNT", 3)):
            if hasattr(socket, name):
                try:
                    sock.setsockopt(socket.IPPROTO_TCP, getattr(socket, name), value)
                except OSError:
                    pass
        sock.settimeout(0.2)                         # how often the clock is looked at: a wait, never a spin
        packets = Packets()
        got = 0
        if self.state != "app_not_streaming":        # (it may be about to hang up, as it did a moment ago)
            self._set("connecting", f"Connected over {'the USB cable' if via == 'usb' else 'Wi-Fi'}, "
                                    "waiting for the first picture.", via)
        try:
            while not self._stop.is_set():
                try:
                    data = sock.recv(1 << 20)
                except socket.timeout:
                    data = None
                except OSError as e:
                    return f"the connection broke ({e})", got
                now = time.monotonic()
                if data == b"":
                    return "the phone closed the connection", got
                if data:
                    whole = packets.feed(data)
                    self.bad_frames += packets.bad
                    packets.bad = 0
                    if whole:
                        self.skipped += len(whole) - 1               # behind: only the newest matters
                        if self._publish(*whole[-1], now):
                            got += 1
                            last = now
                            if self.state != "streaming":
                                self._arrivals.clear()
                                self._arrivals.append(now)
                                self.connections += got == 1
                                self._set("streaming", "", via)
                quiet = now - last
                if self.redial_s and quiet > self.redial_s:
                    return f"no picture for {quiet:.0f} s", got
                if quiet > self.stall_s and self.state != "stalled":
                    self._set("stalled", "", via)
            return "stopping", got
        finally:
            sock.close()

    def _publish(self, head: dict, packet: bytes, now: float) -> bool:
        t = time.perf_counter()
        frame = decode(head, packet)
        if frame is None:
            self.bad_frames += 1
            return False
        rgb, depth, K, pose = frame
        if not self.board.fits(rgb, depth):
            if not self._too_big:
                self._too_big = True
                log.error("a %s picture does not fit the board (%d bytes): frames are being dropped",
                          rgb.shape, self.board.rgb_cap)
            self.bad_frames += 1
            return False
        self.board.write(rgb, depth, K, pose, now, head["t"], head["frame_id"])
        self._tell(b"f\n")
        self._decode_ms += 0.05 * ((time.perf_counter() - t) * 1000 - self._decode_ms)
        self.frames += 1
        self._arrivals.append(now)
        self._last_frame = (now, depth, rgb.shape)
        self.recorder.offer(rgb, depth, K, pose, head["t"], now)
        return True

    # --- life ------------------------------------------------------------------
    def _take_lock(self):
        import fcntl
        self.files["dir"].mkdir(parents=True, exist_ok=True)
        self.files["status"].parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.files["lock"], os.O_RDWR | os.O_CREAT, 0o644)
        end = time.monotonic() + 1.0                 # a reader asking "is it running?" holds it for an instant
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() > end:
                    os.close(fd)
                    raise AlreadyRunning(f"A phone bridge is already running here ({self.files['lock']}).")
                time.sleep(0.05)
        # The bridge before this one may have been killed with its board still in memory.
        try:
            old = json.loads(os.pread(fd, 4096, 0) or b"{}")
            if old.get("shm"):
                import _posixshmem
                _posixshmem.shm_unlink("/" + old["shm"].lstrip("/"))
                log.info("removed the board a killed bridge left behind (%s)", old["shm"])
        except (ImportError, OSError, ValueError):
            pass
        self._lock_fd = fd

    def run(self):
        self._take_lock()
        threads = []
        try:
            self.board = stream.FrameBoard.create()
            os.ftruncate(self._lock_fd, 0)
            os.pwrite(self._lock_fd, json.dumps({"pid": os.getpid(), "shm": self.board.name}).encode(), 0)
            sock_path = self.files["sock"]
            sock_path.unlink(missing_ok=True)        # left by a bridge that was killed
            self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._server.bind(str(sock_path))
            self._server.listen(16)
            if self.usb and not os.environ.get("PHONE_BRIDGE_IPROXY"):
                stream._kill_orphan_iproxy()         # one left by a pipeline that was killed
            self._read_record_switch()
            self.recorder.pack_leftovers()
            self.usb_watch.poll()
            threads = [threading.Thread(target=self._serve, daemon=True, name="readers"),
                       threading.Thread(target=self._report, daemon=True, name="status"),
                       threading.Thread(target=self._watch_usb, daemon=True, name="usb")]
            for th in threads:
                th.start()
            log.info("phone bridge up: pid %d, board %s, port %d, usb %s, wifi %s", os.getpid(), self.board.name,
                     self.port, "on" if self.usb else "off", self.phone_ip or "off")
            while not self._stop.is_set():
                conn = self._dial()
                if conn is None:
                    self._stop.wait(self.RETRY_S)
                    continue
                why, got = self._receive(*conn)
                if self._stop.is_set():
                    break
                if got:
                    self.reconnects += 1
                    log.warning("lost the stream after %d frames: %s. Dialling again.", got, why)
                    self._set("connecting", f"The stream from the iPhone stopped ({why}). Connecting again.")
                else:
                    if "closed" in why:
                        self._set("app_not_streaming", "The iPhone hangs up as soon as it is called: PathOn is not "
                                                       "streaming. Open PathOn on the phone and tap Start Streaming.")
                    self._stop.wait(self.RETRY_S)    # connected, and not one picture: do not hammer it
        finally:
            self._stop.set()
            self._changed.set()
            for th in threads:
                th.join(timeout=3.0)
            self._stop_iproxy()
            self.recorder.close()
            if self._server is not None:
                self._server.close()
                self.files["sock"].unlink(missing_ok=True)
            if self.board is not None:
                self.board.close()
            try:
                os.ftruncate(self._lock_fd, 0)
            except OSError:
                pass
            os.close(self._lock_fd)
            log.info("phone bridge stopped")

    def stop(self):
        self._stop.set()
        self._changed.set()


# ---------------------------------------------------------------------------
# The command line.

def _lock_holder(files: dict) -> int | None:
    """The running bridge's pid, or None."""
    if not stream.bridge_running(files["dir"]):
        return None
    try:
        return int(json.loads(files["lock"].read_text())["pid"])
    except (OSError, ValueError, KeyError):
        return 0                                     # running, but it has not written its pid yet


def _wait(cond, seconds: float) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.1)
    return cond()


def _print_status(files: dict, as_json: bool = False) -> int:
    st = stream.bridge_status(files["dir"])
    if _lock_holder(files) is None or st is None:
        print("{}" if as_json else "phone bridge: not running. Start it with: python tools/phone_bridge.py start")
        return 2
    if as_json:
        print(json.dumps(st))
    else:
        rec = f", recording to {st['record_folder']}" if st.get("recording") else ""
        print(f"phone bridge: {st['state']}{rec}. {st['message']}")
    return 0 if st["ok"] else 1


def _start(a, files: dict, argv: list[str]) -> int:
    if _lock_holder(files) is not None:
        print("phone bridge: already running.")
        return _print_status(files) and 0
    files["dir"].mkdir(parents=True, exist_ok=True)
    # Its own session, with no terminal: closing the window that started it does not reach it.
    with open(files["dir"] / "phone_bridge.out", "wb") as out:
        subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "run", *argv], stdin=subprocess.DEVNULL,
                         stdout=out, stderr=out, start_new_session=True, cwd=str(ROOT))
    if not _wait(lambda: _lock_holder(files) and stream.bridge_status(files["dir"]) is not None, 15.0):
        print(f"phone bridge: did not start. See {files['log']} and {files['dir'] / 'phone_bridge.out'}", file=sys.stderr)
        return 2
    _wait(lambda: (stream.bridge_status(files["dir"]) or {}).get("state") not in ("starting", "connecting"), 3.0)
    _print_status(files)
    return 0


def _stop(a, files: dict) -> int:
    pid = _lock_holder(files)
    if pid is None:
        print("phone bridge: not running.")
        return 0
    if not pid:
        print("phone bridge: running, but its pid is not known yet. Try again.", file=sys.stderr)
        return 1
    os.kill(pid, signal.SIGTERM)
    if not _wait(lambda: _lock_holder(files) is None, a.wait):      # it may be finishing a recording
        print(f"phone bridge: still stopping after {a.wait:.0f} s; killing it (the recording is finished at the next start).",
              file=sys.stderr)
        os.kill(pid, signal.SIGKILL)
        _wait(lambda: _lock_holder(files) is None, 5.0)
    print("phone bridge: stopped.")
    return 0


def _record(a, files: dict) -> int:
    files["record"].parent.mkdir(parents=True, exist_ok=True)
    tmp = files["record"].with_name(files["record"].name + ".tmp")
    tmp.write_text(a.switch + "\n")
    os.replace(tmp, files["record"])
    print(f"phone bridge: recording switched {a.switch}" +
          ("." if _lock_holder(files) is not None else " for when the bridge is started (it is not running)."))
    return 0


def _run(a, files: dict) -> int:
    files["dir"].mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(files["log"], maxBytes=1_000_000, backupCount=3)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler] + ([logging.StreamHandler()] if sys.stderr.isatty() else []))
    bridge = Bridge(a.dir, a.port, a.phone_ip, not a.no_usb, a.stall, a.redial, a.record, a.record_dir,
                    a.record_minutes, a.min_free_gb)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: bridge.stop())
    signal.signal(signal.SIGHUP, signal.SIG_IGN)     # the terminal closing is not a reason to drop the phone
    try:
        bridge.run()
    except AlreadyRunning as e:
        print(f"phone bridge: {e}", file=sys.stderr)
        return 3
    except Exception:
        log.exception("phone bridge crashed")
        raise
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    env = os.environ.get
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", nargs="?", default="run", choices=["run", "start", "stop", "status", "record"])
    p.add_argument("switch", nargs="?", choices=["on", "off"], help="for record")
    p.add_argument("--dir", default=None, help="where its files go (default out/, or PHONE_BRIDGE_DIR)")
    p.add_argument("--port", type=int, default=int(env("PHONE_BRIDGE_PORT", "8888")), help="PathOn's port on the phone")
    p.add_argument("--phone-ip", default=env("STREAM_PHONE_IP"), help="the phone's Wi-Fi address, tried when the cable does not answer")
    p.add_argument("--no-usb", action="store_true", default=env("PHONE_BRIDGE_USB") == "0", help="Wi-Fi only")
    p.add_argument("--stall", type=float, default=stream.STALL_S, help="seconds without a frame that count as stalled")
    p.add_argument("--redial", type=float, default=float(env("PHONE_BRIDGE_REDIAL_S", "10")),
                   help="seconds of silence after which the connection is dropped and dialled again (0: never)")
    p.add_argument("--record", action="store_true", default=env("PHONE_BRIDGE_RECORD", "0") not in ("0", ""),
                   help="save what the phone sends (out/status/phone_record overrides this)")
    p.add_argument("--record-dir", default=env("PHONE_BRIDGE_RECORD_DIR"), help="default recordings/")
    p.add_argument("--record-minutes", type=float, default=float(env("PHONE_BRIDGE_RECORD_MINUTES", "15")))
    p.add_argument("--min-free-gb", type=float, default=5.0, help="do not record with less disk space than this")
    p.add_argument("--wait", type=float, default=45.0, help="for stop: seconds to wait before killing it")
    p.add_argument("--json", action="store_true", help="for status: print all of out/status/phone.json")
    a = p.parse_args(argv)
    files = stream.bridge_files(a.dir)
    if a.command == "status":
        return _print_status(files, a.json)
    if a.command == "stop":
        return _stop(a, files)
    if a.command == "record":
        if a.switch is None:
            p.error("record needs on or off")
        return _record(a, files)
    if a.command == "start":
        return _start(a, files, [x for x in argv if x != "start"])
    return _run(a, files)


if __name__ == "__main__":
    sys.exit(main())
