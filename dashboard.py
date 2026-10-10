"""Live web dashboard for judges, served by main.py from the GX10 itself.

Standard library only and no CDN, so it keeps working with the network unplugged.

    GET  /          the page (dashboard/index.html)
    GET  /events    server-sent events: one JSON snapshot per frame, JPEG inlined
    POST /key       {"key": "left"} from the page, for the fake hand and retargeting

main.py calls publish() every frame and say() whenever the wearer hears a phrase.
"""
from __future__ import annotations

import base64
import collections
import json
import mimetypes
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

import trials
from interfaces import hand_error, plan_audio

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard")
MAX_IMG_H = 720
DEPTH_H = 240
# Depth picture in the page's own blue (hue 255): near is pale, far is dark. Yellow and
# green stay reserved for the fingertip-to-object line and the grab. BGR.
DEPTH_FAR = np.array([40, 22, 12], np.float32)
# 3D view: about this many lidar points, sent every few frames (int16 mm xyz + uint8 rgb).
CLOUD_POINTS = 9000
CLOUD_EVERY = 3
CLOUD_MAX_M = 1.6
DEPTH_NEAR = np.array([255, 228, 196], np.float32)
MAX_FPS = 30.0
# Host probed to tell whether the internet is reachable. Point it at an unroutable address
# (DASHBOARD_NET_CHECK_HOST=10.255.255.1) to see the offline screen without unplugging.
NET_CHECK_HOST = os.environ.get("DASHBOARD_NET_CHECK_HOST", "1.1.1.1")


def _num(v, nd=4):
    return None if v is None else round(float(v), nd)


def _vec(v, nd=4):
    return None if v is None else [round(float(x), nd) for x in v]


def _trial_stats():
    rows = trials.read_rows()
    out = {}
    for cond in ("baseline", "guided"):
        rs = [r for r in rows if r.get("condition") == cond]
        ok = [float(r["time_s"]) for r in rs if r.get("success") == "1"]
        out[cond] = {"n": len(rs), "successes": len(ok),
                     "median_s": float(np.median(ok)) if ok else None}
    return out


class Dashboard:
    def __init__(self, host: str = "0.0.0.0", port: int = 8642):
        self.host, self.port = host, port
        self._lock = threading.Lock()
        self._new = threading.Event()
        self._cond = threading.Condition()
        self._snap = None             # latest (frame, scene, info) from main
        self._msg = b""               # latest encoded SSE payload
        self._cloud_msg = b""         # latest point cloud, its own SSE event so new pages get it at once
        self._cloud_seq = 0
        self._seq = 0
        self._img_cache = (None, None, (0, 0))
        self._depth_cache = (None, None)
        self._phrases = collections.deque(maxlen=8)
        self._keys = collections.deque(maxlen=64)
        self._online = None
        self._trials = _trial_stats()
        self._host = socket.gethostname().split(".")[0]

    # main loop side ---------------------------------------------------------

    def start(self) -> str:
        handler = self._handler()
        self._server = ThreadingHTTPServer((self.host, self.port), handler)
        self._server.daemon_threads = True
        for target in (self._server.serve_forever, self._encode_loop, self._background_loop):
            threading.Thread(target=target, daemon=True).start()
        return f"http://localhost:{self.port}/"

    def publish(self, frame, scene, info: dict) -> None:
        with self._lock:
            self._snap = (frame, scene, info)
        self._new.set()

    def say(self, text: str) -> None:
        """Record a phrase the wearer heard, for the caption bar."""
        with self._lock:
            self._phrases.append({"t": time.time(), "text": text})

    def pop_keys(self) -> list[str]:
        with self._lock:
            keys = list(self._keys)
            self._keys.clear()
        return keys

    def stop(self) -> None:
        if hasattr(self, "_server"):
            self._server.shutdown()

    # encoding ---------------------------------------------------------------

    def _encode_loop(self):
        min_dt = 1.0 / MAX_FPS
        last = 0.0
        while True:
            self._new.wait()
            self._new.clear()
            wait = min_dt - (time.monotonic() - last)
            if wait > 0:
                time.sleep(wait)
            last = time.monotonic()
            with self._lock:
                snap = self._snap
                phrases = list(self._phrases)
            if snap is None:
                continue
            try:
                body = json.dumps(self._snapshot(*snap, phrases), separators=(",", ":"))
                cloud = None
                if self._seq % CLOUD_EVERY == 0:
                    cloud = json.dumps(self._cloud(snap[0]), separators=(",", ":"))
            except Exception as e:  # never let one bad frame kill the stream
                print(f"[dashboard] encode failed: {e!r}")
                continue
            with self._cond:
                self._seq += 1
                self._msg = f"id: {self._seq}\ndata: {body}\n\n".encode()
                if cloud:
                    self._cloud_seq += 1
                    self._cloud_msg = f"event: cloud\ndata: {cloud}\n\n".encode()
                self._cond.notify_all()

    def _image(self, frame):
        key = id(frame.rgb), frame.t
        if self._img_cache[0] != key:
            h, w = frame.rgb.shape[:2]
            s = min(1.0, MAX_IMG_H / h)
            bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
            if s < 1.0:
                bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            ok, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 82])
            uri = "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode() if ok else None
            self._img_cache = (key, uri, (w, h))
        return self._img_cache[1], self._img_cache[2]

    def _depth(self, frame):
        key = id(frame.depth), frame.t
        due = self._depth_cache[0] is None or self._seq % 3 == 0
        if due and self._depth_cache[0] != key:
            d = frame.depth
            h, w = d.shape[:2]
            small = cv2.resize(d, (max(1, int(w * DEPTH_H / h)), DEPTH_H), interpolation=cv2.INTER_NEAREST)
            near = np.clip(1 - (small - 0.2) / 1.8, 0, 1)[..., None]
            img = (DEPTH_FAR + (DEPTH_NEAR - DEPTH_FAR) * near).astype(np.uint8)
            img[(small <= 0) | ~np.isfinite(small)] = 0
            ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
            uri = "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode() if ok else None
            self._depth_cache = (key, uri)
        return self._depth_cache[1]

    def _cloud(self, frame):
        """A thinned, coloured point cloud in camera coordinates."""
        d = frame.depth
        h, w = d.shape[:2]
        step = max(1, int(np.ceil(np.sqrt(h * w / CLOUD_POINTS))))
        ds = d[::step, ::step]
        vs, us = np.mgrid[0:h:step, 0:w:step]
        ok = np.isfinite(ds) & (ds > 0.15) & (ds < CLOUD_MAX_M)
        z = ds[ok]
        x = (us[ok] - frame.K[0, 2]) * z / frame.K[0, 0]
        y = (vs[ok] - frame.K[1, 2]) * z / frame.K[1, 1]
        xyz = np.round(np.stack([x, y, z], axis=1) * 1000).astype("<i2")
        rgb = frame.rgb[::step, ::step][ok].astype(np.uint8)
        return {"n": int(z.size), "b64": base64.b64encode(xyz.tobytes() + rgb.tobytes()).decode()}

    def _snapshot(self, frame, scene, info, phrases):
        img, (w, h) = self._image(frame)
        err = hand_error(scene)
        box = None
        if scene.object_box is not None:
            x1, y1, x2, y2 = scene.object_box
            box = [x1 / w, y1 / h, x2 / w, y2 / h]
        obj_px = None
        if box is not None:
            obj_px = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
        elif scene.object_xyz is not None and scene.object_xyz[2] > 0:
            x, y, z = scene.object_xyz
            obj_px = [(frame.K[0, 0] * x / z + frame.K[0, 2]) / w, (frame.K[1, 1] * y / z + frame.K[1, 2]) / h]
        hand_px = None if scene.hand_px is None else [scene.hand_px[0] / w, scene.hand_px[1] / h]
        audio = info.get("audio") or plan_audio(err, info.get("state"))
        return {
            "seq": self._seq + 1, "sent": round(time.time(), 3),
            "img": img, "aspect": w / h, "depth": self._depth(frame),
            "target": info.get("target"), "heard": info.get("heard"), "state": info.get("state"),
            "object": {"box": box, "px": obj_px, "xyz": _vec(scene.object_xyz), "visible": bool(scene.object_visible)},
            "hand": {"px": hand_px, "xyz": _vec(scene.hand_xyz), "visible": bool(scene.hand_visible)},
            "err": _vec(err), "dist": _num(np.linalg.norm(err)) if err is not None else None,
            "audio": {k: (_num(v, 3) if isinstance(v, float) else v) for k, v in audio.items()},
            "audio_source": "guidance" if info.get("audio") else "derived",
            "phrases": phrases,
            "stages": {k: _num(v, 2) for k, v in (info.get("stages") or {}).items()},
            "fps": _num(info.get("fps"), 1), "age_ms": _num(info.get("age_ms"), 1),
            "e2e_ms": _num(info.get("e2e_ms"), 1), "e2e_label": info.get("e2e_label"),
            "run": info.get("run"), "trials": self._trials,
            "online": self._online, "modes": info.get("modes_map"),
            "models": info.get("models"), "host": self._host,
        }

    # background checks ------------------------------------------------------

    def _background_loop(self):
        while True:
            try:
                socket.create_connection((NET_CHECK_HOST, 53), timeout=0.6).close()
                self._online = True
            except OSError:
                self._online = False
            try:
                self._trials = _trial_stats()
            except Exception:
                pass
            time.sleep(3.0)

    # http -------------------------------------------------------------------

    def _handler(self):
        dash = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_GET(self):
                path = self.path.split("?")[0]
                if path == "/events":
                    return self._events()
                if path == "/":
                    path = "/index.html"
                full = os.path.normpath(os.path.join(STATIC_DIR, path.lstrip("/")))
                if not full.startswith(STATIC_DIR + os.sep) or not os.path.isfile(full):
                    self.send_error(404)
                    return
                with open(full, "rb") as f:
                    data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", mimetypes.guess_type(full)[0] or "application/octet-stream")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                if self.path != "/key":
                    self.send_error(404)
                    return
                n = int(self.headers.get("Content-Length") or 0)
                try:
                    key = json.loads(self.rfile.read(n) or b"{}").get("key")
                except ValueError:
                    key = None
                if isinstance(key, str) and len(key) <= 12:
                    with dash._lock:
                        dash._keys.append(key)
                self.send_response(204)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def _events(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                seen = -1
                seen_cloud = -1
                try:
                    while True:
                        with dash._cond:
                            dash._cond.wait_for(lambda: dash._seq != seen, timeout=2.0)
                            seq, msg = dash._seq, dash._msg
                            cloud_seq, cloud_msg = dash._cloud_seq, dash._cloud_msg
                        if seq == seen or not msg:
                            self.wfile.write(b": keepalive\n\n")
                        else:
                            self.wfile.write(msg)
                            seen = seq
                        if cloud_msg and cloud_seq != seen_cloud:
                            self.wfile.write(cloud_msg)
                            seen_cloud = cloud_seq
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return

        return Handler
