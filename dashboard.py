"""Live web dashboard for judges, served by main.py from the GX10 itself.

Standard library only and no CDN, so it keeps working with the network unplugged.

    GET  /          the page (dashboard/index.html)
    GET  /deck/     the pitch deck (deck/index.html), which embeds the dashboard for the demo
    GET  /trials.json  baseline and guided trial stats from trials.csv
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

import tracing
import trials
from interfaces import hand_error

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard")
DECK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deck")
MAX_IMG_H = 720
DEPTH_H = 240
# Depth picture in the page's own blue (hue 255): near is pale, far is dark. Yellow and
# green stay reserved for the fingertip-to-object line and the grab. BGR.
DEPTH_FAR = np.array([40, 22, 12], np.float32)
# 3D view: about this many lidar points, sent every few frames (int16 mm xyz + uint8 rgb).
CLOUD_POINTS = 9000
CLOUD_EVERY = 3
CLOUD_MAX_M = 1.6
# Level the 3D view on the tabletop: RANSAC plane fit on the cloud. Without a table in view,
# assume the planned chest mount (phone tilted down 30 degrees, table 45 cm below the lens).
ASSUMED_TILT_DEG = 30.0
ASSUMED_TABLE_M = 0.45
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


# What the wearer can switch between while it runs: (family, heading, module, the module's list,
# how to read the current one, how to set it, what voice calls it). A family whose functions do
# not exist yet is simply left out, so this works with whatever the modules offer.
_FAMILIES = (("guidance", "Guidance sound", "guidance", "OPTIONS", "option", "set_option", "option"),
             ("hand", "Hand tracking", "perception", "HAND_OPTIONS", "hand_option", "set_hand_option", "hand option"),
             ("grab", "When it says grab", "guidance", "GRAB_OPTIONS", "grab_option", "set_grab_option", "grab option"))


def page_version() -> int:
    """Changes whenever the page's files do: a tab left open from before an update reloads itself."""
    try:
        return int(max(os.path.getmtime(os.path.join(STATIC_DIR, n)) for n in os.listdir(STATIC_DIR)))
    except (OSError, ValueError):
        return 0


def _running(module) -> bool:
    """Is the real module in use? Asking guidance for its option would otherwise start it,
    and open the sound device, in a run that uses the fake one."""
    return getattr(module, "_default", None) is not None


def option_families() -> list[dict]:
    """Every family of options with its choices and the one in use (None when not running)."""
    import importlib
    out = []
    for family, heading, mod, names, current, _setter, say in _FAMILIES:
        try:
            module = importlib.import_module(mod)
            options = [{"n": i + 1, "key": o[0], "title": o[1], "about": o[2] if len(o) > 2 else ""}
                       for i, o in enumerate(getattr(module, names))]
            now = getattr(module, current)()[0] if _running(module) else None
        except Exception:
            continue
        out.append({"family": family, "heading": heading, "say": say, "current": now, "options": options})
    return out


def set_option(family: str, n: int) -> str:
    """Switch one family to option n and tell the wearer. Returns the option's title."""
    import importlib
    for fam, _heading, mod, _names, _current, setter, say in _FAMILIES:
        if fam == family:
            module = importlib.import_module(mod)
            if not _running(module):
                raise RuntimeError("not running yet")
            title = getattr(module, setter)(n)
            try:
                import guidance
                if _running(guidance):
                    guidance.say_text(f"{say.capitalize()} {n}: {title}.")
            except Exception:
                pass
            return title
    raise ValueError(f"no such family: {family}")


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
        self._phrases = collections.deque(maxlen=12)
        self._keys = collections.deque(maxlen=64)
        self._online = None
        self._trials = _trial_stats()
        self._host = socket.gethostname().split(".")[0]
        self._rng = np.random.default_rng(0)
        t = np.radians(ASSUMED_TILT_DEG)
        self._up = np.array([0.0, -np.cos(t), -np.sin(t)])  # table normal in camera axes
        self._table_h = ASSUMED_TABLE_M                      # lens height above the table

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
        pts = np.stack([x, y, z], axis=1)
        found = self._fit_table(pts)
        up = self._up
        fwd = np.array([0.0, 0.0, 1.0]) - up * up[2]
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, up)
        xyz = np.round(pts * 1000).astype("<i2")
        rgb = frame.rgb[::step, ::step][ok].astype(np.uint8)
        return {"n": int(z.size), "b64": base64.b64encode(xyz.tobytes() + rgb.tobytes()).decode(),
                # rows map camera axes to a level frame: x right, y up (off the table), z forward
                "R": [_vec(right, 5), _vec(up, 5), _vec(fwd, 5)], "h": _num(self._table_h, 4),
                "table": found}

    def _fit_table(self, pts) -> bool:
        """Find the tabletop: the biggest roughly level plane below the lens. Smoothed over time."""
        if len(pts) < 200:
            return False
        rng = self._rng
        sample = pts[rng.choice(len(pts), min(len(pts), 2500), replace=False)]
        best = (0, None, None)
        for _ in range(60):
            a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
            n = np.cross(b - a, c - a)
            norm = np.linalg.norm(n)
            if norm < 1e-9:
                continue
            n /= norm
            if n[1] > 0:          # face up, toward the lens (camera y points down)
                n = -n
            dist = -float(n @ a)  # lens height above this plane
            if -n[1] < 0.5 or dist < 0.1:
                continue
            count = int((np.abs(sample @ n + dist) < 0.015).sum())
            if count > best[0]:
                best = (count, n, dist)
        count, n, dist = best
        if n is None or count < 0.25 * len(sample):
            return False
        inl = pts[np.abs(pts @ n + dist) < 0.015]
        centre = inl.mean(axis=0)
        n = np.linalg.svd(inl - centre, full_matrices=False)[2][2]
        if n[1] > 0:
            n = -n
        dist = -float(n @ centre)
        self._up = self._up * 0.7 + n * 0.3
        self._up /= np.linalg.norm(self._up)
        self._table_h = self._table_h * 0.7 + dist * 0.3
        return True

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
        return {
            "seq": self._seq + 1, "sent": round(time.time(), 3),
            "img": img, "aspect": w / h, "depth": self._depth(frame),
            "target": info.get("target"), "heard": info.get("heard"), "state": info.get("state"),
            "object": {"box": box, "px": obj_px, "xyz": _vec(scene.object_xyz), "visible": bool(scene.object_visible)},
            "hand": {"px": hand_px, "xyz": _vec(scene.hand_xyz), "visible": bool(scene.hand_visible)},
            "err": _vec(err), "dist": _num(np.linalg.norm(err)) if err is not None else None,
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
                if path == "/log":
                    # live activity for the logs-only view: what was heard, asked, answered and said
                    try:
                        after = int(self.path.partition("after=")[2].split("&")[0] or 0)
                    except ValueError:
                        after = 0
                    events, nxt = tracing.events_since(after)
                    lines = [{"t": e.get("t"), "who": d[0], "took": d[1], "what": d[2]}
                             for e in events for d in [tracing.describe(e)] if d]
                    return self._send(json.dumps({"next": nxt, "lines": lines}).encode(), "application/json")
                if path == "/options":
                    return self._send(json.dumps({"families": option_families(), "version": page_version()}).encode(),
                                      "application/json")
                if path == "/trials.json":
                    # readable from a deck served elsewhere (?dash=...): read-only, local numbers
                    return self._send(json.dumps(_trial_stats()).encode(), "application/json", cors=True)
                if path == "/deck":
                    self.send_response(301)
                    self.send_header("Location", "/deck/")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                root = STATIC_DIR
                if path.startswith("/deck/"):
                    root, path = DECK_DIR, path[len("/deck"):]
                if path.endswith("/"):
                    path += "index.html"
                full = os.path.normpath(os.path.join(root, path.lstrip("/")))
                if not full.startswith(root + os.sep) or not os.path.isfile(full):
                    self.send_error(404)
                    return
                with open(full, "rb") as f:
                    data = f.read()
                ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
                if full.endswith((".js", ".mjs")):
                    ctype = "text/javascript"
                self._send(data, ctype)

            def _send(self, data, ctype, cors=False):
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                if cors:
                    self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                if self.path not in ("/key", "/option"):
                    self.send_error(404)
                    return
                try:
                    n = int(self.headers.get("Content-Length") or 0)
                    body = json.loads(self.rfile.read(n) or b"{}")
                except ValueError:
                    body = {}
                if not isinstance(body, dict):
                    body = {}
                if self.path == "/option":
                    try:
                        reply = {"ok": True, "title": set_option(str(body.get("family")), int(body.get("n")))}
                    except Exception as e:
                        reply = {"ok": False, "error": str(e)}
                    return self._send(json.dumps(reply).encode(), "application/json")
                key = body.get("key")
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
                            # "and dash._msg": before the first frame there is nothing to send. Without
                            # it this never waits until then (seq 0 != seen -1) and every open page
                            # spins here writing keepalives, fighting Whisper for the GIL.
                            dash._cond.wait_for(lambda: dash._seq != seen and dash._msg, timeout=2.0)
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
