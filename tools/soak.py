"""Soak test: run the whole pipeline for hours with no phone, no sound and no window, drive it
like a demo, and show what grows, drifts or stops.

  python tools/soak.py run --minutes 130       # pipeline + camera + requests + measurements, report at the end
  python tools/soak.py run --minutes 5 --dash-port 8656 --voice-port 8666     # a quick look
  python tools/soak.py report out/soak/NAME    # the table again (also works while a run is going)

What stands in for the hardware:

  camera    `soak.py camera` replays recordings into stream.py's relay port at the live frame rate
            (the recordings hold 8 frames a second, the phone sends 26), so the reader thread, the
            reconnect logic and the per-frame work all carry the load of a live run. A recording
            that lost its depth (live_2148) gets a flat one, which is enough for its three reaches
            to be followed all the way to "grab". A control file changes it while it runs: pause,
            3 frames a second, no depth, no poses. tools/fault_drill.py uses that.
  sound     `soak.py pipeline` is main.py with a sound card that plays to nowhere: the real mixer,
            Piper and the clips all run, and a sentence takes as long as it takes to say, but
            nothing reaches a loudspeaker.
  requests  typed text posted to the voice page (/text), a new one every 20 to 60 s: things that
            are there, things that are not, questions, option switches and nonsense.

Every 10 s it notes the process's memory, GPU memory, threads and open files, the dashboard's
event rate and the size of the logs. The frame times come from the run's own trace.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import signal
import socket
import ssl
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SAMPLE_S = 10.0
RELAY_PORT = 9156            # not stream.RELAY_PORT: a teammate's relay may be on that one

# What the driver asks for. The names in THERE are what Cosmos lists in recordings/live_2025 and
# live_2148; the rest are the ways a request goes wrong on stage.
THERE = ["water bottle", "blue cup", "orange", "laptop", "cup", "bottle"]
NOT_THERE = ["banana", "scissors", "red stapler", "tennis ball", "umbrella"]
INDIRECT = ["I'm thirsty", "can you find my water bottle please", "I want something to drink from",
            "where is the blue cup", "grab the orange for me"]
LISTS = ["what's on the table", "what can I pick up", "what do you see"]
OPTIONS = ["option 2", "option 3", "next option", "option 5", "option 3", "option 1", "option 3"]
NONSENSE = ["asdf qwer zxcv", "thank you", "the the the the the the the the the", "no", "hmm okay so",
            "the second one", "yes"]


def request_script(seed: int = 0):
    """An endless, repeatable stream of (seconds to wait, kind, text)."""
    rng = random.Random(seed)
    kinds = [("there", THERE, 6), ("repeat", None, 2), ("not_there", NOT_THERE, 2), ("indirect", INDIRECT, 3),
             ("list", LISTS, 2), ("option", OPTIONS, 2), ("nonsense", NONSENSE, 2)]
    last = THERE[0]
    while True:
        kind, pool, _ = rng.choices(kinds, weights=[k[2] for k in kinds])[0]
        text = last if pool is None else rng.choice(pool)
        if kind in ("there", "not_there"):
            last = text
        yield rng.uniform(20.0, 60.0), kind, text


# ---------------------------------------------------------------------------
# Sound card that plays to nowhere

def silent_sink(mode: str = "auto", device=None, cfg=None):
    """audio_out.open_sink for a machine that must stay quiet: the real Mixer, rendered at the
    speed a sound card would ask for it and thrown away."""
    import audio_out
    mixer = audio_out.Mixer(cfg)
    block = 256
    dt = block / mixer.cfg.sr

    def play():
        import tracing
        due = reported = time.monotonic()
        blocks = late = 0
        worst = 0.0
        while True:
            t = time.perf_counter()
            mixer.render(block)
            worst = max(worst, (time.perf_counter() - t) * 1000)
            blocks += 1
            due += dt
            now = time.monotonic()
            # A real card holds about 10 ms of sound (10.7 ms in the live runs). A block that is
            # ready later than that would have been a gap in the wearer's ears.
            late += now - due > 0.010
            if now - reported >= 5.0:         # the same summary AudioEngine writes, so reports read alike
                tracing.event("audio", render_max_ms=round(worst, 3), blocks=blocks, underruns=late)
                blocks = late = 0
                worst, reported = 0.0, now
            wait = due - now
            if wait > 0:
                time.sleep(wait)
            elif wait < -0.2:
                due = now                     # the process was stalled: carry on from now, like a card would

    threading.Thread(target=play, daemon=True, name="silent-card").start()
    return mixer


def run_pipeline(argv: list[str]):
    """main.py, in this process, with the sound going nowhere."""
    os.chdir(ROOT)
    # When the soak itself was started with nohup or &, Ctrl-C is ignored all the way down, and
    # the run could only be killed. (What main.py does by itself in that case is its own drill.)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    if os.environ.get("SOAK_FILE_LIMIT_KB"):          # a full disk, without filling one: no file may grow past this
        import resource
        limit = int(os.environ["SOAK_FILE_LIMIT_KB"]) * 1024
        resource.setrlimit(resource.RLIMIT_FSIZE, (limit, limit))
    if os.environ.get("SOAK_AUDIO", "silent") == "silent":
        import audio_out
        audio_out.open_sink = silent_sink     # before guidance imports it
    import main
    sys.argv = ["main.py", *argv]
    main.main()


# ---------------------------------------------------------------------------
# Camera

# restart: change the number to start again at recording number `recording`, `seek` seconds in.
DEFAULT_CONTROL = {"fps": 26.0, "pause": False, "depth": "real", "pose": "real", "drop": False,
                   "restart": 0, "recording": 0, "seek": 0.0}


def read_control(path, previous: dict | None = None) -> dict:
    """The camera's settings from its control file. A file that is missing or half written
    leaves them as they were."""
    out = dict(previous or DEFAULT_CONTROL)
    try:
        out.update({k: v for k, v in json.loads(Path(path).read_text()).items() if k in DEFAULT_CONTROL})
    except (OSError, ValueError, AttributeError, TypeError):
        pass
    return out


def shape_depth(depth_m: np.ndarray, how: str) -> np.ndarray:
    """real | zero (no lidar reading anywhere) | flat:METRES (a wall at that distance)."""
    if how == "zero":
        return np.zeros_like(depth_m)
    if how.startswith("flat:"):
        return np.full_like(depth_m, float(how.split(":", 1)[1]))
    return depth_m


def frame_message(jpg: bytes, png: bytes, K, pose, kind: str) -> bytes:
    """One frame in the format stream.py's relay source reads."""
    import stream
    meta = json.dumps({"kind": kind, "t_send": time.time(), "K": np.asarray(K, dtype=np.float64).ravel().tolist(),
                       "pose": None if pose is None else np.asarray(pose, dtype=np.float64).ravel().tolist()}).encode()
    return stream._HEAD.pack(stream._MAGIC, len(meta), len(jpg), len(png)) + meta + jpg + png


def encode_pictures(rgb: np.ndarray, depth_m: np.ndarray, quality: int = 85) -> tuple[bytes, bytes]:
    import cv2
    import stream
    _, jpg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    _, png = cv2.imencode(".png", stream._to_mm(depth_m), [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return jpg.tobytes(), png.tobytes()


def recorded_frames(folders: list, loop: bool = True, first: int = 0, seek_s: float = 0.0):
    """(rgb, depth in metres, K, pose or None, kind, seconds until the next frame) from each
    recording in turn, for ever, starting `seek_s` into recording number `first`. A recording
    without depth gets a flat one at 45 cm."""
    import cv2
    import stream
    folders = list(folders)
    order = folders[first % len(folders):] + folders[:first % len(folders)]
    while True:
        for folder in order:
            folder = Path(folder)
            data = np.load(folder / "frames.npz")
            depth_mm, Ks, poses, ts = data["depth_mm"], data["K"], data["pose"], data["t"]
            meta = folder / "meta.json"
            kind = json.loads(meta.read_text()).get("kind", "record3d") if meta.exists() else "record3d"
            flat = float((depth_mm[::max(1, len(ts) // 40)] > 0).mean()) < 0.2
            cap = cv2.VideoCapture(str(folder / "rgb.mp4"))
            skip, seek_s = int(np.searchsorted(ts - ts[0], seek_s)), 0.0
            try:
                for i in range(len(ts)):
                    ok, bgr = cap.read() if i >= skip else (cap.grab(), None)
                    if not ok:
                        break
                    if i < skip:
                        continue
                    depth = depth_mm[i].astype(np.float32) / 1000.0
                    if flat:
                        depth = np.full_like(depth, 0.45)
                    gap = float(ts[i + 1] - ts[i]) if i + 1 < len(ts) else 0.1
                    yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), depth, Ks[i], stream._valid_pose(poses[i]), kind, gap
            finally:
                cap.release()
        if not loop:
            return


def run_camera(folders: list, port: int, control: str | None, host: str = "127.0.0.1", fps: float = 26.0,
               frames=None, stop: threading.Event | None = None):
    """Send recorded frames to the pipeline's relay port, at `fps` whatever the recording's own
    rate was: each recorded picture is repeated until the next one is due."""
    settings = dict(DEFAULT_CONTROL, fps=fps)
    if control:
        settings = read_control(control, settings)
    source = frames if frames is not None else recorded_frames(folders, True, int(settings["recording"]), float(settings["seek"]))
    restart = settings["restart"]
    conn = None
    current = next(source)
    shown_until = time.monotonic() + current[5]
    encoded = (None, None)       # (the settings it was encoded for, (jpg, png))
    due = time.monotonic()
    sent = 0
    while stop is None or not stop.is_set():
        if control:
            settings = read_control(control, settings)
        now = time.monotonic()
        if settings["restart"] != restart and frames is None:      # a cut: another recording, or back to the start
            restart = settings["restart"]
            source = recorded_frames(folders, True, int(settings["recording"]), float(settings["seek"]))
            shown_until = now
        while now >= shown_until:                     # the recording moves on in real time, paused or not
            current = next(source)
            shown_until += current[5]
            encoded = (None, None)
        due += 1.0 / max(0.2, float(settings["fps"]))
        if due < now - 1.0:
            due = now
        wait = due - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        if settings["drop"] and conn is not None:
            conn.close()
            conn = None
        if settings["pause"] or settings["drop"]:
            continue
        if conn is None:
            try:
                conn = socket.create_connection((host, port), timeout=3)
                conn.settimeout(30)                   # a pipeline frozen for a few seconds is not a lost connection
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                print(f"[camera] connected to {host}:{port}", flush=True)
            except OSError:
                time.sleep(0.3)
                continue
        if encoded[0] != settings["depth"]:
            encoded = (settings["depth"], encode_pictures(current[0], shape_depth(current[1], settings["depth"])))
        pose = current[3] if settings["pose"] == "real" else None
        try:
            conn.sendall(frame_message(*encoded[1], current[2], pose, current[4]))
            sent += 1
        except OSError as e:
            print(f"[camera] connection lost ({e}); reconnecting", flush=True)
            conn.close()
            conn = None
    if conn is not None:
        conn.close()
    return sent


# ---------------------------------------------------------------------------
# Starting and stopping the pieces

def port_free(port: int) -> bool:
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("0.0.0.0", port))
            return True
        except OSError:
            return False


def python() -> str:
    return sys.executable


def start_camera(out: Path, recordings: list, relay_port: int, fps: float = 26.0, **control) -> subprocess.Popen:
    control_file = out / "camera.json"
    control_file.write_text(json.dumps(dict(DEFAULT_CONTROL, fps=fps, **control)))
    log = open(out / "camera.log", "ab")
    return subprocess.Popen([python(), str(ROOT / "tools" / "soak.py"), "camera", "--port", str(relay_port),
                             "--control", str(control_file), "--fps", str(fps), *map(str, recordings)],
                            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)


def set_camera(out: Path, **changes):
    """Change the camera while it runs, e.g. set_camera(out, pause=True)."""
    path = out / "camera.json"
    settings = read_control(path)
    settings.update(changes)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(settings))
    os.replace(tmp, path)


def pipeline_env(out: Path, dash_port: int, voice_port: int, source: str, extra: dict | None = None) -> dict:
    env = dict(os.environ)
    env.update({"STREAM_SOURCE": source, "GUIDANCE_AUDIO": "off",      # off: if the silent card is ever not
                "VOICE_MODE": "web", "VOICE_PORT": str(voice_port),     # put in, it prints rather than plays
                "DASHBOARD_PORT": str(dash_port), "TRACE_DIR": str(out / "logs"), "PYTHONUNBUFFERED": "1"})
    env.setdefault("GUIDANCE_MODE", "3")              # the option the live runs use; without it main.py starts in "words"
    env.update(extra or {})
    return {k: v for k, v in env.items() if v is not None}


def start_pipeline(out: Path, dash_port: int, voice_port: int, source: str, extra_env: dict | None = None,
                   extra_args: tuple = (), stdout=None, ui: str = "web") -> subprocess.Popen:
    """main.py with every real module, the way tools/live.sh starts it, minus the window."""
    (out / "logs").mkdir(parents=True, exist_ok=True)
    log = stdout if stdout is not None else open(out / "pipeline.log", "ab")
    args = ["--stream", "real", "--perception", "real", "--guidance", "real", "--voice", "real", "--ui", ui,
            "--no-open", "--port", str(dash_port), "--log-dir", str(out / "logs"), *extra_args]
    return subprocess.Popen([python(), str(ROOT / "tools" / "soak.py"), "pipeline", "--", *args], cwd=ROOT,
                            env=pipeline_env(out, dash_port, voice_port, source, extra_env),
                            stdout=log, stderr=subprocess.STDOUT if stdout is None else None,
                            stdin=subprocess.DEVNULL, start_new_session=True)


def stop(proc: subprocess.Popen | None, sig=signal.SIGINT, wait_s: float = 30.0) -> float | None:
    """Ask a process to stop and wait. Returns how long it took, or None if it had to be killed."""
    if proc is None or proc.poll() is not None:
        return 0.0
    t = time.monotonic()
    try:
        proc.send_signal(sig)
        proc.wait(wait_s)
        return time.monotonic() - t
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            proc.kill()
        proc.wait(10)
        return None


_TLS = ssl.create_default_context()
_TLS.check_hostname = False
_TLS.verify_mode = ssl.CERT_NONE         # the voice page's certificate is made on the spot


def post_text(voice_port: int, text, timeout_s: float = 10.0, raw: bytes | None = None) -> tuple[float, str]:
    """Type a request into the voice page. Returns (seconds it took, what the page says it heard)."""
    body = raw if raw is not None else json.dumps({"text": text}).encode()
    req = urllib.request.Request(f"https://127.0.0.1:{voice_port}/text", body, {"Content-Type": "application/json"})
    t = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout_s, context=_TLS) as r:
        heard = json.load(r).get("heard", "")
    return time.monotonic() - t, heard


def voice_status(voice_port: int, timeout_s: float = 2.0) -> dict | None:
    try:
        with urllib.request.urlopen(f"https://127.0.0.1:{voice_port}/status", timeout=timeout_s, context=_TLS) as r:
            return json.load(r)
    except (OSError, ValueError):
        return None


def wait_ready(proc: subprocess.Popen, voice_port: int, timeout_s: float = 180.0) -> float | None:
    """Seconds until the voice page answers (Whisper is loaded and it is waiting for a request),
    or None if the pipeline died or never got there."""
    t = time.monotonic()
    while time.monotonic() - t < timeout_s:
        if proc.poll() is not None:
            return None
        if voice_status(voice_port, 1.0) is not None:
            return time.monotonic() - t
        time.sleep(0.5)
    return None


# ---------------------------------------------------------------------------
# Reading the run's own trace

class Trace:
    """Follows the newest trace file in a folder as it grows. poll() returns the new records."""

    def __init__(self, folder: Path):
        self.folder, self.path, self._pos, self._rest = Path(folder), None, 0, b""

    def poll(self) -> list[dict]:
        newest = max(self.folder.glob("trace_*.jsonl"), key=lambda p: p.stat().st_mtime, default=None)
        if newest is None:
            return []
        if newest != self.path:
            self.path, self._pos, self._rest = newest, 0, b""
        with open(self.path, "rb") as f:
            f.seek(self._pos)
            data = f.read()
        self._pos += len(data)
        lines = (self._rest + data).split(b"\n")
        self._rest = lines.pop()                      # a line still being written
        out = []
        for line in lines:
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
        return out

    def wait_for(self, want, timeout_s: float, since: float = 0.0, seen: list | None = None):
        """The first record for which want(record) is true and that is not older than `since`
        (epoch seconds), or None. Records read on the way are appended to `seen`."""
        end = time.monotonic() + timeout_s
        while True:
            for rec in self.poll():
                if seen is not None:
                    seen.append(rec)
                if rec.get("t", 0) >= since and want(rec):
                    return rec
            if time.monotonic() >= end:
                return None
            time.sleep(0.1)


def event(name: str, **fields):
    """A test for Trace.wait_for: an event with this name and these fields."""
    return lambda r: r.get("type") == "event" and r.get("name") == name and all(r.get(k) == v for k, v in fields.items())


def frame_stats(frames: list[dict]) -> dict:
    """Frame rate and loop times over a stretch of trace."""
    if len(frames) < 2:
        return {"frames": len(frames), "fps": 0.0}
    loop = np.array([f.get("loop_ms", 0.0) for f in frames])
    e2e = np.array([f["e2e_ms"] for f in frames if f.get("e2e_ms") is not None] or [0.0])
    gaps = np.diff([f["t"] for f in frames])
    seconds = frames[-1]["t"] - frames[0]["t"]
    return {"frames": len(frames), "fps": (len(frames) - 1) / seconds if seconds > 0 else 0.0,
            "loop_p50": float(np.percentile(loop, 50)), "loop_p95": float(np.percentile(loop, 95)),
            "loop_max": float(loop.max()), "e2e_p50": float(np.percentile(e2e, 50)),
            "e2e_p95": float(np.percentile(e2e, 95)), "gap_max_ms": float(gaps.max() * 1000),
            "skipped": int(sum(f.get("skipped", 0) for f in frames))}


# ---------------------------------------------------------------------------
# Measuring

def gpu_memory_mb(pid: int) -> float | None:
    try:
        rows = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=5).stdout.splitlines()
        for row in rows:
            p, _, mb = row.partition(",")
            if p.strip() == str(pid):
                return float(mb)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return None


def gpu_busy() -> float | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.split()
        return float(out[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def folder_mb(folder: Path) -> float:
    return sum(p.stat().st_size for p in Path(folder).rglob("*") if p.is_file()) / 1e6


class EventsClient:
    """Stands in for the browser: reads the dashboard's /events and counts what arrives."""

    def __init__(self, port: int):
        self.port, self.events, self.bytes, self.reconnects = port, 0, 0, 0
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True, name="events-client").start()

    def _run(self):
        while not self._stop.is_set():
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/events", timeout=10) as r:
                    for line in r:
                        self.bytes += len(line)
                        self.events += line.startswith(b"id:")
                        if self._stop.is_set():
                            return
            except (OSError, ValueError):
                pass
            self.reconnects += 1
            self._stop.wait(1.0)

    def close(self):
        self._stop.set()


FIELDS = ["t", "elapsed_s", "alive", "rss_mb", "gpu_mb", "threads", "fds", "cpu_pct", "dash_events_s",
          "dash_mb_s", "dash_reconnects", "logs_mb", "trace_mb", "gpu_busy_pct", "load1"]


def sample(proc, ps, client: EventsClient, out: Path, t0: float, last: dict) -> dict:
    now = time.time()
    row = {"t": round(now, 2), "elapsed_s": round(now - t0, 1), "alive": int(proc.poll() is None)}
    try:
        with ps.oneshot():
            row.update(rss_mb=round(ps.memory_info().rss / 1e6, 1), threads=ps.num_threads(), fds=ps.num_fds(),
                       cpu_pct=round(ps.cpu_percent(None), 1))
    except Exception:                                 # the process has gone
        row["alive"] = 0
    row["gpu_mb"] = gpu_memory_mb(proc.pid)
    dt = now - last.get("t", t0) or 1.0
    row["dash_events_s"] = round((client.events - last.get("events", 0)) / dt, 2)
    row["dash_mb_s"] = round((client.bytes - last.get("bytes", 0)) / dt / 1e6, 3)
    row["dash_reconnects"] = client.reconnects
    last.update(t=now, events=client.events, bytes=client.bytes)
    row["logs_mb"] = round(folder_mb(out / "logs"), 2)
    traces = sorted((out / "logs").glob("trace_*.jsonl"))
    row["trace_mb"] = round(traces[-1].stat().st_size / 1e6, 2) if traces else 0.0
    row["gpu_busy_pct"] = gpu_busy()
    row["load1"] = round(os.getloadavg()[0], 2)
    return row


def drive(voice_port: int, out: Path, stop_at: float, seed: int, halt: threading.Event):
    """Type requests the way a demo would, and note each with the time it was sent."""
    with open(out / "requests.csv", "a", newline="") as f:
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(["t", "kind", "text", "http_ms", "heard", "error"])
        for wait, kind, text in request_script(seed):
            if halt.wait(wait) or time.time() + 1 > stop_at:
                return
            t = time.time()
            try:
                seconds, heard = post_text(voice_port, text, timeout_s=20.0)
                w.writerow([f"{t:.3f}", kind, text, f"{seconds * 1000:.0f}", heard, ""])
            except Exception as e:
                w.writerow([f"{t:.3f}", kind, text, "", "", repr(e)])
            f.flush()


def run(a) -> int:
    import psutil
    out = Path(a.out or ROOT / "out" / "soak" / time.strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    for port in (a.dash_port, a.voice_port, a.relay_port):
        if not port_free(port):
            sys.exit(f"[soak] port {port} is taken; pass other ports")
    recordings = [ROOT / r for r in a.recordings]
    if a.camera == "relay":
        camera = start_camera(out, recordings, a.relay_port, a.fps)
        source = f"relay:{a.relay_port}"
    else:
        camera, source = None, str(a.recordings[0])
    proc = start_pipeline(out, a.dash_port, a.voice_port, source)
    (out / "run.json").write_text(json.dumps({"started": time.time(), "pid": proc.pid, "minutes": a.minutes,
                                              "source": source, "fps": a.fps, "seed": a.seed,
                                              "recordings": [str(r) for r in a.recordings]}))
    print(f"[soak] pipeline pid {proc.pid}, writing to {out}", flush=True)
    halt = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: halt.set())
    signal.signal(signal.SIGINT, lambda *_: halt.set())
    code = 1
    client = None
    try:
        ready = wait_ready(proc, a.voice_port)
        if ready is None:
            print(f"[soak] the pipeline did not come up; see {out / 'pipeline.log'}", flush=True)
            return 1
        print(f"[soak] ready after {ready:.0f} s", flush=True)
        client = EventsClient(a.dash_port)
        t0 = time.time()
        stop_at = t0 + a.minutes * 60
        threading.Thread(target=drive, args=(a.voice_port, out, stop_at, a.seed, halt), daemon=True).start()
        ps = psutil.Process(proc.pid)
        ps.cpu_percent(None)
        last = {}
        with open(out / "samples.csv", "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            while time.time() < stop_at and not halt.is_set():
                halt.wait(SAMPLE_S)
                row = sample(proc, ps, client, out, t0, last)
                w.writerow(row)
                f.flush()
                if not row["alive"]:
                    print(f"[soak] THE PIPELINE STOPPED after {row['elapsed_s']:.0f} s (exit code {proc.poll()})", flush=True)
                    break
        code = 0 if proc.poll() is None else 1
    finally:
        halt.set()
        if client:
            client.close()
        took = stop(proc)
        stop(camera, signal.SIGTERM, 5)
        (out / "stopped.json").write_text(json.dumps({"t": time.time(), "alive_at_end": code == 0, "stop_s": took}))
    print(report(out), flush=True)
    return code


# ---------------------------------------------------------------------------
# The table

def _load_trace(folder: Path):
    frames, events = [], []
    for path in sorted((folder / "logs").glob("trace_*.jsonl")):
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                (frames if rec.get("type") == "frame" else events if rec.get("type") == "event" else []).append(rec)
    return frames, events


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _f(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# A gap between two frames with one of these inside it is the loop doing what it was asked:
# waiting for a request, or loading a model. A gap with none is the loop hanging.
EXPECTED_PAUSES = ("say", "nemotron", "cosmos.list", "option", "whisper", "detector.set_target")


def stalls(frames: list[dict], events: list[dict], longer_ms: float = 250.0) -> list[tuple[float, float]]:
    """(when, how long in ms) for every pause in the loop that nothing in the trace explains."""
    import bisect
    marks = sorted(e["t"] for e in events if e["name"] in EXPECTED_PAUSES or e["name"].startswith("load."))
    out = []
    for a, b in zip(frames, frames[1:]):
        gap = (b["t"] - a["t"]) * 1000
        if gap <= longer_ms:
            continue
        # a model call is logged when it ends and a retarget during the frame after the pause
        i = bisect.bisect_left(marks, a["t"] - 0.05)
        if i < len(marks) and marks[i] <= b["t"] + min(2.0, gap / 1000):
            continue
        out.append((a["t"], gap))
    return out


def slope_per_hour(xs, ys) -> float:
    """Straight-line trend of ys against xs (seconds), per hour."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if xs.size < 3 or np.ptp(xs) == 0:
        return 0.0
    return float(np.polyfit(xs, ys, 1)[0] * 3600.0)


def buckets(samples: list[dict], frames: list[dict], events: list[dict], minutes: float) -> list[dict]:
    """The run cut into stretches of `minutes`, each with its averages."""
    if not samples:
        return []
    t0 = _f(samples[0]["t"]) - _f(samples[0]["elapsed_s"])
    width = minutes * 60
    rows = []
    n = int((_f(samples[-1]["elapsed_s"]) - 1) // width) + 1
    for b in range(n):
        lo, hi = t0 + b * width, t0 + (b + 1) * width
        ss = [s for s in samples if lo <= _f(s["t"]) < hi]
        if not ss:
            continue
        fr = [f for f in frames if lo <= f["t"] < hi]
        ev = [e for e in events if lo <= e["t"] < hi]

        def mean(key):
            v = [_f(s[key]) for s in ss if _f(s.get(key)) is not None]
            return float(np.mean(v)) if v else None

        def ms(name, q=50):
            v = [e["ms"] for e in ev if e["name"] == name and e.get("ms") is not None]
            return float(np.percentile(v, q)) if v else None
        row = {"from_min": b * minutes, "rss_mb": mean("rss_mb"), "gpu_mb": mean("gpu_mb"), "threads": mean("threads"),
               "fds": mean("fds"), "cpu_pct": mean("cpu_pct"), "dash_events_s": mean("dash_events_s"),
               "logs_mb": _f(ss[-1]["logs_mb"]), "gpu_busy_pct": mean("gpu_busy_pct"),
               "nemotron_p50": ms("nemotron"), "nemotron_max": ms("nemotron", 100),
               "cosmos_list_p50": ms("cosmos.list"), "cosmos_list_max": ms("cosmos.list", 100),
               "cosmos_find_p50": ms("cosmos.find"), "set_target_p50": ms("detector.set_target"),
               "requests": sum(1 for e in ev if e["name"] == "nemotron") + sum(1 for e in ev if e["name"] == "cosmos.list")}
        row.update(frame_stats(fr))
        hangs = stalls(fr, ev)
        row["stalls"], row["stall_max"] = len(hangs), max((g for _, g in hangs), default=0.0)
        rows.append(row)
    return rows


def request_outcomes(requests: list[dict], events: list[dict]) -> list[dict]:
    """For each typed request: how long until the pipeline did something about it (the first
    thing said or the first model call after it), and what that was."""
    out = []
    events = sorted(events, key=lambda e: e["t"])
    for i, r in enumerate(requests):
        t = _f(r["t"])
        nxt = _f(requests[i + 1]["t"]) if i + 1 < len(requests) else t + 120
        after = [e for e in events if t <= e["t"] < nxt and e["name"] in
                 ("nemotron", "cosmos.list", "say", "option", "detector.set_target", "state")]
        first = after[0] if after else None
        out.append({**r, "react_ms": None if first is None else (first["t"] - t) * 1000 - (first.get("ms") or 0.0),
                    "did": None if first is None else first["name"],
                    "said": next((e.get("text") for e in after if e["name"] == "say"), None)})
    return out


def verdicts(samples: list[dict], rows: list[dict], warm_s: float = 600.0) -> list[tuple[bool, str]]:
    """What must hold for the run to count as steady. The first ten minutes are warm-up."""
    out = []
    alive = bool(samples) and all(s["alive"] == "1" for s in samples)
    out.append((alive, "the pipeline ran to the end" if alive else "THE PIPELINE STOPPED during the run"))
    steady = [s for s in samples if _f(s["elapsed_s"], 0) >= warm_s and s["alive"] == "1"]
    if len(steady) < 30:
        out.append((True, "too short to judge growth (needs 15 minutes or more)"))
        return out
    xs = [_f(s["elapsed_s"]) for s in steady]
    for key, limit, unit in (("rss_mb", 100.0, "MB"), ("gpu_mb", 100.0, "MB"), ("threads", 2.0, ""), ("fds", 2.0, "")):
        pts = [(x, _f(s[key])) for x, s in zip(xs, steady) if _f(s.get(key)) is not None]
        if len(pts) < 30:
            continue
        slope = slope_per_hour(*zip(*pts))
        out.append((slope <= limit, f"{key} grows {slope:+.1f} {unit} an hour after warm-up (limit {limit:g})"))
    later = [r for r in rows if r["from_min"] * 60 >= warm_s and r.get("loop_p50")]
    if len(later) >= 2:
        first, last = later[0], later[-1]
        drift = last["loop_p50"] / first["loop_p50"] - 1 if first["loop_p50"] else 0.0
        out.append((drift <= 0.15, f"loop time at the end is {drift:+.0%} against the first steady stretch"))
    worst = max((r.get("stall_max", 0.0) for r in rows), default=0.0)
    out.append((worst <= 1000.0, f"longest pause in the loop that nothing explains: {worst:.0f} ms (limit 1000)"))
    return out


def report(folder, minutes: float | None = None) -> str:
    folder = Path(folder)
    samples = _read_csv(folder / "samples.csv")
    requests = _read_csv(folder / "requests.csv")
    frames, events = _load_trace(folder)
    if not samples:
        return f"[soak] nothing measured yet in {folder}"
    total_min = _f(samples[-1]["elapsed_s"]) / 60
    minutes = minutes or (10 if total_min > 40 else max(1, round(total_min / 6)))
    rows = buckets(samples, frames, events, minutes)

    def cell(v, fmt="{:.0f}"):
        return "-" if v is None else fmt.format(v)
    cols = [("min", "from_min", "{:.0f}"), ("RSS MB", "rss_mb", "{:.0f}"), ("GPU MB", "gpu_mb", "{:.0f}"),
            ("thr", "threads", "{:.0f}"), ("fds", "fds", "{:.0f}"), ("cpu%", "cpu_pct", "{:.0f}"),
            ("fps", "fps", "{:.1f}"), ("loop p50", "loop_p50", "{:.1f}"), ("p95", "loop_p95", "{:.1f}"),
            ("stalls", "stalls", "{:.0f}"), ("longest", "stall_max", "{:.0f}"), ("e2e p95", "e2e_p95", "{:.0f}"),
            ("dash/s", "dash_events_s", "{:.1f}"), ("logs MB", "logs_mb", "{:.1f}"), ("nemo p50", "nemotron_p50", "{:.0f}"),
            ("max", "nemotron_max", "{:.0f}"), ("list p50", "cosmos_list_p50", "{:.0f}"), ("max", "cosmos_list_max", "{:.0f}"),
            ("find p50", "cosmos_find_p50", "{:.0f}"), ("retarget", "set_target_p50", "{:.0f}"), ("GPU busy%", "gpu_busy_pct", "{:.0f}")]
    lines = [f"Soak {folder}: {total_min:.0f} minutes, {len(frames)} frames, {len(requests)} requests. "
             f"Times in ms; each row is the {minutes:g} minutes from 'min'. fps counts the time spent waiting for a "
             "request too; stalls are pauses over 250 ms that no request or model load explains.",
             "  ".join(f"{name:>8}" for name, _, _ in cols)]
    for r in rows:
        lines.append("  ".join(f"{cell(r.get(key), fmt):>8}" for _, key, fmt in cols))

    done = request_outcomes(requests, events)
    if done:
        lines += ["", "Requests by kind: count, no reaction within the gap to the next one, reaction time median / max (ms)"]
        for kind in sorted({d["kind"] for d in done}):
            ds = [d for d in done if d["kind"] == kind]
            react = [d["react_ms"] for d in ds if d["react_ms"] is not None]
            lines.append(f"  {kind:<10} {len(ds):4d}   ignored {sum(d['react_ms'] is None for d in ds):3d}   "
                         + (f"{np.median(react):7.0f} / {max(react):7.0f}" if react else "      - /       -")
                         + f"   errors {sum(bool(d['error']) for d in ds)}")
    said = {}
    for e in events:
        if e["name"] == "say":
            said[e.get("text", "")] = said.get(e.get("text", ""), 0) + 1
    if said:
        top = sorted(said.items(), key=lambda kv: -kv[1])[:8]
        lines += ["", "Said most often: " + "; ".join(f'"{t[:40]}" x{n}' for t, n in top)]
    states = {}
    for f in frames:
        states[f.get("state")] = states.get(f.get("state"), 0) + 1
    if frames:
        lines.append("Frames by state: " + ", ".join(f"{k} {v / len(frames):.0%}" for k, v in sorted(states.items(), key=lambda kv: -kv[1])))
    lines.append("")
    results = verdicts(samples, rows)
    lines += [("PASS  " if ok else "FAIL  ") + text for ok, text in results]
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="soak the pipeline and report")
    r.add_argument("--minutes", type=float, default=130.0)
    r.add_argument("--dash-port", type=int, default=8656)
    r.add_argument("--voice-port", type=int, default=8666)
    r.add_argument("--relay-port", type=int, default=RELAY_PORT)
    r.add_argument("--camera", choices=("relay", "replay"), default="relay",
                   help="relay: the fake camera at the live frame rate. replay: stream.py's own replay of the first recording")
    r.add_argument("--fps", type=float, default=26.0, help="what the fake camera sends (the phone: 26)")
    r.add_argument("--recordings", nargs="+", default=["recordings/live_2025", "recordings/live_2148"])
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", help="folder for the results (default out/soak/DATE_TIME)")
    p = sub.add_parser("report", help="the table for a finished or running soak")
    p.add_argument("folder")
    p.add_argument("--minutes", type=float, help="width of each row")
    c = sub.add_parser("camera", help="the fake camera alone")
    c.add_argument("recordings", nargs="+")
    c.add_argument("--port", type=int, default=RELAY_PORT)
    c.add_argument("--host", default="127.0.0.1")
    c.add_argument("--fps", type=float, default=26.0)
    c.add_argument("--control", help="JSON file read every frame: fps, pause, depth, pose, drop")
    m = sub.add_parser("pipeline", help="main.py with a sound card that plays to nowhere")
    m.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    if a.cmd == "run":
        sys.exit(run(a))
    elif a.cmd == "report":
        print(report(a.folder, a.minutes))
    elif a.cmd == "camera":
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
        try:
            run_camera(a.recordings, a.port, a.control, a.host, a.fps)
        except KeyboardInterrupt:
            pass
    elif a.cmd == "pipeline":
        run_pipeline([x for x in a.rest if x != "--"])
