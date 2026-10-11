"""One button for the demo. Double-click the Proprio icon on the GX10 (tools/install_launcher.sh
puts it on the desktop) and a page opens with everything that has to be true, green or red,
each line saying what to do. When it is green, press Start. From then on this keeps it going:
it restarts the pipeline if it crashes or hangs, reconnects the AirPods, restarts a model
server that died, and only stops when Stop is pressed on the page.

    python launcher.py            # what the icon does: start the keeper if needed, show the page
    python launcher.py status     # the page's state as JSON
    python launcher.py stop       # stop the pipeline; the keeper and the page stay
    python launcher.py quit       # stop the pipeline and the phone bridge, and end the keeper
    python launcher.py serve      # the keeper itself, in this terminal (the icon runs it detached)

The page is http://localhost:8640 (PROPRIO_PORT), on this machine only unless PROPRIO_HOST=0.0.0.0;
even then its buttons only work from the GX10 itself. The keeper is one detached process, logging to
out/supervisor.log; starting it twice, closing the browser or refreshing the page changes nothing.
It runs as a service of the desktop session (systemd-run --user), which starts it again if it is
killed; the new one carries on with the pipeline and phone bridge that are already running.
The pipeline is started through tools/live.sh, so a run from the button and a run by hand are
the same thing; its output goes to out/pipeline.log.

Settings come from the environment, then from .env.launcher next to this file (KEY=VALUE lines),
then from DEFAULTS below. The checks themselves are in launcher_checks.py.
"""
from __future__ import annotations

import fcntl
import json
import mimetypes
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import launcher_checks as checks
from launcher_checks import AMBER, GREEN, RED, line

ROOT = Path(__file__).resolve().parent
PAGE_DIR = ROOT / "launcher"
PORT = 8640

# The settings of the run that worked, for every start from the button. tools/live.sh has the
# same list for a start by hand (tests/test_launcher.py checks the two agree).
DEFAULTS = {
    "UI": "web",                         # the dashboard in the browser, no OpenCV window
    "VIEW": "camera",                    # the dashboard opens on the camera, not on the logs
    "GUIDANCE_AUDIO": "on",              # sound through the sound card or not at all: never a silent stand-in
    "GUIDANCE_MODE": "3",                # option 3, the combined spoken directions
    "VOICE_MODE": "web",                 # the earbuds (or the microphone page), never "press Enter"
    "TRANSFORMERS_VERBOSITY": "error",   # these two keep the log readable
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "HOST": "127.0.0.1",                 # the dashboard answers this machine only: the hall's network is public.
                                         # HOST=0.0.0.0 in .env.launcher to watch it from a laptop
}
RECORD_S = 12 * 3600    # see Supervisor.pipeline_env: one recording for the whole run, closed when it ends

# How the pipeline is judged (its own words for its state are in main.py's Heartbeat).
BOOT_S = 60.0           # no heartbeat this long after it was started: it never came up
FROZEN_S = 10.0         # a heartbeat this old: the whole process is frozen. It writes one a second; the
                        # longest gap measured on the GX10 through a whole start (Whisper, YOLO-World and
                        # CLIP loading, two other pipelines sharing the machine) was 1.2 s
FAILING_S = 20.0        # a step failing on every frame for this long will not heal by itself
BACKOFF_S = (1, 2, 5, 10, 20, 30)   # the wait before each restart in a row; back to the first after a good run
SETTLED_S = 60.0        # a run this long was a good run
STOP_WAIT_S = 20.0      # after Ctrl-C the pipeline closes its recording; give it time before killing it


def load_env(root: Path = ROOT, base: dict | None = None) -> dict:
    """The environment for the keeper and everything it starts."""
    env = dict(os.environ if base is None else base)
    try:
        for row in (root / ".env.launcher").read_text().splitlines():
            key, eq, value = row.strip().partition("=")
            if eq and key and not key.startswith("#"):
                env.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    except OSError:
        pass
    if sys.platform.startswith("linux"):
        # Started from the desktop these are already set; over SSH they are not, and without them
        # there is no screen to open the page on and no sound server to ask about the AirPods.
        uid = os.getuid()
        env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{uid}")
        env.setdefault("DISPLAY", ":1")
        env.setdefault("XAUTHORITY", f"/run/user/{uid}/gdm/Xauthority")
        env.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path=/run/user/{uid}/bus")
    for key, value in DEFAULTS.items():
        env.setdefault(key, value)
    return env


def tail(path: Path, n: int = 200, max_bytes: int = 200_000) -> list[str]:
    """The last n lines of a log, without reading all of it."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            text = f.read().decode("utf-8", "replace")
    except OSError:
        return []
    rows = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text).replace("\r", "\n").splitlines()
    return [r for r in rows if r.strip()][-n:]


def judge(alive: bool, hb: dict | None, started: float, now: float, boot_s: float = BOOT_S,
          frozen_s: float = FROZEN_S, failing_s: float = FAILING_S) -> tuple[str, str]:
    """Is the pipeline all right? ("ok" | "dead" | "hung", why). `hb` is its pipeline.json, or
    None if it has not written one of its own yet; `started` is when it was started.

    Waiting for the wearer to speak is never a hang, however long: main.py reports that as
    "listening". A hang is a step that does not end ("stuck", decided in main.py, which knows
    how long each step may take), a step that fails on every frame, or a process so frozen that
    even its heartbeat thread has stopped."""
    if not alive:
        return "dead", "it exited"
    if hb is None:
        late = now - started
        return ("hung", f"it had not reported {late:.0f} s after starting") if late > boot_s else ("ok", "starting")
    age = now - float(hb.get("t") or 0)
    if age > frozen_s:
        return "hung", f"it stopped reporting {age:.0f} s ago (frozen)"
    if hb.get("state") == "stuck":
        return "hung", str(hb.get("message") or "a step never finished")
    failing = max((hb.get("failing") or {}).values(), default=0)
    if failing > failing_s:
        return "hung", str(hb.get("message") or "a step kept failing")
    return "ok", str(hb.get("message") or "")


class Proc:
    """A long-running program in its own session, with its output in a file. It is found again
    by its pid file, so a keeper that is restarted carries on with the same pipeline and the same
    phone bridge instead of starting second ones."""

    def __init__(self, name: str, argv: list[str], cwd: Path, out: Path, match: str):
        self.name, self.argv, self.cwd, self.match = name, argv, Path(cwd), match
        self.log = out / f"{name}.log"
        self.pid_file = out / "status" / f"{name}.pid"
        self.popen = None
        self.pid: int | None = None
        self.started = 0.0

    def start(self, env: dict):
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)
        if self.log.exists() and self.log.stat().st_size > 20e6:
            os.replace(self.log, self.log.with_suffix(".log.1"))
        with open(self.log, "ab") as f:
            f.write(f"\n--- {self.name} started {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n".encode())
            f.flush()
            self.popen = subprocess.Popen(self.argv, cwd=self.cwd, env=env, stdin=subprocess.DEVNULL, stdout=f,
                                          stderr=subprocess.STDOUT, start_new_session=True)
        self.pid, self.started = self.popen.pid, time.time()
        self.pid_file.write_text(json.dumps({"pid": self.pid, "started": self.started}))

    def adopt(self) -> bool:
        """Carry on with the one a previous keeper started, if it is still running."""
        got = checks.read_status(self.pid_file)
        try:
            pid = int(got["pid"])
            os.kill(pid, 0)
            args = subprocess.run(["ps", "-o", "args=", "-p", str(pid)], capture_output=True, text=True, timeout=3).stdout
        except (TypeError, KeyError, ValueError, OSError, subprocess.SubprocessError):
            return False
        if self.match not in args:             # the number has been given to some other program since
            return False
        self.popen, self.pid, self.started = None, pid, float(got.get("started") or time.time())
        return True

    def alive(self) -> bool:
        if self.pid is None:
            return False
        if self.popen is not None:
            return self.popen.poll() is None
        try:
            os.kill(self.pid, 0)
            return True
        except OSError:
            return False

    def exit_code(self) -> int | None:
        return self.popen.returncode if self.popen is not None else None

    def stop(self, sig: int = signal.SIGINT, wait_s: float = STOP_WAIT_S):
        """Ask it to stop, wait, then kill whatever is left of it and of anything it started."""
        if self.pid is None:
            return
        pid = self.pid
        try:
            os.kill(pid, sig)
        except OSError:
            pass
        end = time.monotonic() + wait_s
        while self.alive() and time.monotonic() < end:
            time.sleep(0.1)
        try:
            os.killpg(pid, signal.SIGKILL)     # also what a frozen (stopped) process needs
        except OSError:
            pass
        end = time.monotonic() + 3.0
        while self.alive() and time.monotonic() < end:
            time.sleep(0.05)
        self.forget()

    def forget(self):
        self.popen = self.pid = None
        try:
            self.pid_file.unlink()
        except OSError:
            pass


class Supervisor:
    """The keeper: gathers the checklist, runs the buttons, and once started keeps everything alive."""

    GROUPS = (("phone", 1.0), ("airpods", 2.0), ("servers", 2.0), ("gpu", 3.0), ("models", 10.0), ("machine", 3.0),
              ("room", 5.0))

    def __init__(self, root: Path = ROOT, env: dict | None = None, port: int = PORT, run=checks.run,
                 pipeline_argv: list[str] | None = None, home: Path | None = None):
        self.root, self.port = Path(root), port
        self.env = load_env(self.root) if env is None else env
        self.s = checks.Setup(root=self.root, env=self.env, run=run, home=home or Path.home())
        self.out = self.root / "out"
        self.status_dir = self.out / "status"
        self.status_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline = Proc("pipeline", pipeline_argv or ["bash", "tools/live.sh"], self.root, self.out, "main.py")
        # "bridge", not "phone_bridge": out/phone_bridge.log is the bridge's own log
        self.bridge = Proc("bridge", [sys.executable, "tools/phone_bridge.py", "run"], self.root, self.out, "phone_bridge.py")
        self.boot_s, self.frozen_s, self.failing_s, self.backoff_s = BOOT_S, FROZEN_S, FAILING_S, BACKOFF_S
        self.settled_s, self.stop_wait_s = SETTLED_S, STOP_WAIT_S
        self.lock = threading.RLock()
        self.mode = "checklist"            # "checklist" | "running" | "stopping"
        self.since: float | None = None
        self.restarts = 0
        self.in_a_row = 0                  # restarts since the last good run
        self.next_start = 0.0              # time.time() before which the pipeline is not started again
        self.trouble = ""                  # why it was last restarted, in plain words
        self.events: deque = deque(maxlen=60)
        self.lines: dict = {}              # group -> its lines, as last looked at
        self.notes: dict = {}              # line id -> the outcome of its button
        self.busy: dict = {}               # action -> when it began
        self.ear: dict = {}                # "sound_check" | "mic_check" -> its last result
        self.record = self.env.get("PROPRIO_RECORD", "1") != "0"
        self.record_dir: Path | None = None
        self.proof = checks.read_status(self.status_dir / "models_proof.json") or {}
        self.gpu_busy: list = []
        self.dash_shown = False
        self.mode_lost = 0.0               # since when the AirPods have been missing a mode
        self.next_try: dict = {}           # what -> time.time() before which it is not tried again
        self.tries: dict = {}
        self.stopped = threading.Event()
        self.server = None

    # --- what happened, for the page and the log --------------------------------------

    def event(self, text: str):
        print(f"{time.strftime('%H:%M:%S')} {text}", flush=True)
        self.events.append({"t": time.time(), "text": text})

    def note(self, id: str, ok: bool, message: str):
        before = self.notes.get(id) or {}
        self.notes[id] = {"t": time.time(), "ok": ok, "message": message}
        if before.get("message") != message or time.time() - before.get("said", 0) > 300:
            self.notes[id]["said"] = time.time()     # the same outcome again and again is said once in five minutes
            self.event(message)
        else:
            self.notes[id]["said"] = before.get("said", 0)

    # --- the checklist -------------------------------------------------------------------

    def look(self, group: str) -> list[dict]:
        running = self.mode != "checklist" and self.pipeline.alive()
        if group == "phone":
            note = "" if self.bridge.alive() else "The bridge is not running; it is started again every few seconds."
            lines = checks.phone(self.s, note)
            stops = self.tries.get("bridge", 0)
            for l in lines:
                if l["id"] == "phone_stream" and l["level"] != GREEN and stops >= 3:
                    l["message"] += f" The bridge has stopped {stops} times in a row: its last words are in out/phone_bridge.log."
            return lines
        if group == "airpods":
            return checks.airpods(self.s)
        if group == "servers":
            return checks.servers(self.s)
        if group == "gpu":
            ours = {p for p in (self.pipeline.pid, self.busy.get("pid")) if p}
            return checks.gpu(self.s, ours, running, self.gpu_busy)
        if group == "models":
            return checks.models(self.s, {"running": True} if "prove" in self.busy else self.proof)
        if group == "room":
            return checks.room(self.s)
        return checks.machine(self.s, running)

    def watch(self, group: str, every_s: float):
        while not self.stopped.is_set():
            try:
                self.lines[group] = self.look(group)
            except Exception:
                self.event(f"The {group} check failed:\n{traceback.format_exc()}")
                self.lines[group] = [line(group, group.capitalize(), RED, "This check itself failed; see out/supervisor.log.")]
            self.stopped.wait(every_s)

    def ear_lines(self) -> list[dict]:
        """The two checks only a person can judge: can you hear it, and can it hear you."""
        running = self.mode != "checklist"
        silent = self.env.get("GUIDANCE_AUDIO") == "off"
        pods = {l["id"]: l for l in self.lines.get("airpods", [])}
        through = silent or (pods.get("airpods_output") or {}).get("level") == GREEN
        out = []
        for id, title, label, ask in (("sound_check", "Sound check", "Play a test phrase", "Press the button and listen: you should hear "
                                       "a sentence in the AirPods."),
                                      ("mic_check", "Microphone check", "Test the microphone", "Press the button, wait for the beep, say "
                                       "\"blue cup\", and listen to what it heard.")):
            last = self.ear.get(id)
            if id in self.busy:
                out.append(line(id, title, AMBER, "Running now: listen.", required=False))
            elif running:
                out.append(line(id, title, last["level"] if last else "none", (last["message"] + " " if last else "")
                                + "Stop the demo to test again: the pipeline is using the AirPods.", required=False))
            elif not through:
                out.append(line(id, title, "none", "Waiting for the AirPods to be where the sound goes.", required=False, blocked=True))
            else:
                out.append(line(id, title, last["level"] if last else "none", last["message"] if last else ask,
                                required=False, fix=(id, label)))
        return out

    def pipeline_line(self, now: float) -> dict:
        title = "Pipeline"
        p = self.pipeline
        if self.mode == "stopping":
            return line("pipeline", title, AMBER, "Stopping: closing the recording and the logs.", required=False)
        if not p.alive():
            left = max(0.0, self.next_start - now)
            why = f" It was restarted because {self.trouble}." if self.trouble else ""
            return line("pipeline", title, RED if self.in_a_row > 1 else AMBER,
                        (f"Starting again in {left:.0f} s." if left >= 1 else "Starting.") + why, required=False)
        hb = self.heartbeat()
        verdict, why = judge(True, hb, p.started, now, self.boot_s, self.frozen_s, self.failing_s)
        if hb is None:
            return line("pipeline", title, AMBER, "Starting: Python and the models are loading.", required=False)
        quiet = now - float(hb.get("t") or 0)
        if verdict == "ok" and quiet > 3.0:
            return line("pipeline", title, AMBER, f"No word from the pipeline for {quiet:.0f} s. At {self.frozen_s:.0f} s it is "
                        "taken to be frozen and started again.", required=False)
        level = GREEN if hb.get("ok") else AMBER if hb.get("state") == "starting" else RED
        return line("pipeline", title, level if verdict == "ok" else RED, str(hb.get("message") or why), required=False)

    def tell_bridge_to_record(self):
        """The phone bridge does the recording when there is one (a new folder every 15 minutes):
        it is told through out/status/phone_record, and records only between Start and Stop."""
        if self.s.bridge.exists() and not self.s.replay:
            want = "on" if self.record and self.mode == "running" else "off"
            switch = self.status_dir / "phone_record"
            try:
                if not switch.exists() or switch.read_text().strip() != want:
                    switch.write_text(want + "\n")
            except OSError as e:
                self.event(f"Could not write {switch}: {e}")

    def record_line(self) -> dict | None:
        if self.s.replay:
            return line("record", "Recording", "none", "Nothing to record: the camera is a replayed recording.", required=False,
                        minor=True)
        free = checks.shutil.disk_usage(self.root).free / 1e9
        if not self.record:
            return line("record", "Recording", AMBER, "Off: this session is not being saved.", required=False,
                        fix=("toggle_record", "Record"))
        if self.s.bridge.exists() and self.mode != "checklist":
            st = checks.read_status(self.status_dir / "phone.json") or {}
            where = str(st.get("record_folder") or "").replace(str(self.root) + "/", "")
            if st.get("record_error"):
                return line("record", "Recording", RED, f"Not recording: {st['record_error']}", required=False)
            if st.get("recording"):
                return line("record", "Recording", GREEN, f"Into {where}: {st.get('record_frames', 0)} frames so far, a new folder "
                            f"every 15 minutes. {free:.0f} GB left on the disk.", required=False, fix=("toggle_record", "Stop recording"))
            return line("record", "Recording", AMBER, "Waiting for the phone's first frame to start the recording. "
                        f"{free:.0f} GB left on the disk.", required=False, fix=("toggle_record", "Do not record"))
        folder = self.record_dir
        if self.mode == "checklist" or folder is None:
            return line("record", "Recording", GREEN, f"On: colour and depth from Start to Stop, into recordings/. {free:.0f} GB left "
                        "on the disk.", required=False, fix=("toggle_record", "Do not record"))
        size = sum(f.stat().st_size for f in folder.glob("*") if f.is_file()) / 1e6 if folder.exists() else 0.0
        if size == 0 and time.time() - self.pipeline.started > 90 and (self.heartbeat() or {}).get("frames"):
            return line("record", "Recording", RED, f"Nothing has been written to {folder.relative_to(self.root)} although frames "
                        "are arriving.", required=False)
        return line("record", "Recording", GREEN, f"Into {folder.relative_to(self.root)}: {size:.0f} MB so far, {free:.0f} GB left "
                    "on the disk.", required=False, fix=("toggle_record", "Stop recording at the next start"))

    def all_lines(self, now: float | None = None) -> list[dict]:
        now = now or time.time()
        got = {g: list(self.lines.get(g) or [line(g, g.capitalize(), AMBER, "Looking…")]) for g, _ in self.GROUPS}
        out = got["phone"] + got["airpods"] + self.ear_lines() + got["servers"] + got["gpu"] + got["models"] + got["machine"]
        rec = self.record_line()
        out += ([rec] if rec else []) + got["room"]
        if self.mode != "checklist":
            out.insert(0, self.pipeline_line(now))
        for l in out:
            note = self.notes.get(l["id"])
            if note and now - note["t"] < 20 and l["level"] != GREEN:
                l["note"] = note["message"]
        return out

    def heartbeat(self) -> dict | None:
        """The pipeline's own status file, if this pipeline wrote it."""
        hb = checks.read_status(self.status_dir / "pipeline.json")
        return hb if hb and hb.get("pid") == self.pipeline.pid and hb.get("state") != "stopped" else None

    def state(self, can_control: bool = True) -> dict:
        now = time.time()
        lines = self.all_lines(now)
        hb = self.heartbeat() if self.mode != "checklist" else None
        return {"t": now, "mode": self.mode, "ready": checks.ready(lines), "lines": lines, "since": self.since,
                "restarts": self.restarts, "heard": (hb or {}).get("heard"), "said": (hb or {}).get("said"),
                "target": (hb or {}).get("target"), "fps": (hb or {}).get("fps"), "pipeline": hb,
                "events": list(self.events)[-12:], "busy": sorted(k for k in self.busy if k != "pid"),
                "dash_port": self.s.dash_port, "voice_port": self.s.voice_port, "host": socket.gethostname(),
                "can_control": can_control, "replay": self.s.source if self.s.replay else None,
                "settings": {k: self.env.get(k) for k in ("GUIDANCE_MODE", "GUIDANCE_AUDIO", "VOICE_MODE", "STREAM_SOURCE") if self.env.get(k)},
                "offline": bool(checks.offline_env(self.s, self.proof)),
                "log": tail(self.pipeline.log, 14) if self.mode != "checklist" else []}

    # --- starting and stopping -----------------------------------------------------------

    def pipeline_env(self) -> dict:
        env = dict(self.env)
        env.update(OPEN="0", PORT=str(self.s.dash_port), PYTHONUNBUFFERED="1")
        env.update(checks.offline_env(self.s, self.proof))
        self.record_dir = None
        if self.record and not self.s.replay and "STREAM_RECORD" not in self.env and not self.s.bridge.exists():
            # One folder for this run of the pipeline, closed when it ends. stream.py can also
            # stop after a set time, but it packs the recording right then, inside the camera
            # thread: a quarter of an hour of depth is 900 MB and guidance would stall while it
            # is compressed. So the limit is set far beyond any demo and nothing is packed
            # mid-run; a pipeline that is killed leaves .part files that stream.pack() finishes.
            self.record_dir = self.root / "recordings" / time.strftime("demo_%Y%m%d_%H%M%S")
            env.update(STREAM_RECORD=str(self.record_dir.relative_to(self.root)), STREAM_RECORD_SECONDS=str(RECORD_S))
        return env

    def start(self, anyway: bool = False) -> tuple[bool, str]:
        with self.lock:
            if self.mode != "checklist":
                return True, "Already running."
            waiting = [l["title"] for l in self.all_lines() if l["required"] and l["level"] != GREEN]
            if waiting and not anyway:
                return False, "Not ready: " + ", ".join(waiting) + "."
            self.mode, self.since, self.restarts, self.in_a_row = "running", time.time(), 0, 0
            self.next_start, self.trouble, self.dash_shown = 0.0, "", False
            self.event("Started" + (f" anyway, without: {', '.join(waiting)}" if waiting else "") + ".")
            self.spawn()
        return True, "Started."

    def spawn(self):
        self.pipeline.start(self.pipeline_env())
        self.event(f"Pipeline started (pid {self.pipeline.pid})"
                   + (f", recording into {self.record_dir.relative_to(self.root)}." if self.record_dir else "."))

    def stop(self) -> tuple[bool, str]:
        with self.lock:
            if self.mode != "running":
                return True, "Not running."
            self.mode = "stopping"
        self.event("Stop was pressed.")

        def work():
            self.pipeline.stop(signal.SIGINT, self.stop_wait_s)
            self.mode, self.since = "checklist", None
            self.event("Pipeline stopped.")
        threading.Thread(target=work, daemon=True, name="stop").start()
        return True, "Stopping."

    def restart_pipeline(self, why: str, now: float):
        """It died or hung: clear it away and set when to start it again."""
        ran = now - self.pipeline.started
        code = self.pipeline.exit_code()
        if self.pipeline.alive():
            self.pipeline.stop(signal.SIGTERM, 1.0)        # a hung program rarely hears this; it is killed a second later
        else:
            self.pipeline.forget()
        self.in_a_row = 0 if ran >= self.settled_s else self.in_a_row + 1
        wait = self.backoff_s[min(self.in_a_row, len(self.backoff_s) - 1)]
        self.next_start = time.time() + wait
        self.trouble = why + (f" (exit code {code})" if code not in (None, 0) else "")
        self.event(f"Pipeline: {self.trouble} after {ran:.0f} s. Starting it again in {wait} s.")

    # --- keeping things alive --------------------------------------------------------------

    def tick(self, now: float | None = None):
        """Once a second: look at everything this is responsible for and put right what is not."""
        now = now or time.time()
        with self.lock:
            if self.mode == "running":
                p = self.pipeline
                if p.pid is None:
                    if now >= self.next_start:
                        self.restarts += 1
                        self.spawn()
                else:
                    verdict, why = judge(p.alive(), self.heartbeat(), p.started, now, self.boot_s, self.frozen_s, self.failing_s)
                    if verdict != "ok":
                        self.restart_pipeline(why, now)
        if self.mode == "running":
            self.keep_servers(now)
            self.keep_airpods(now)
            if not self.dash_shown and self.pipeline.alive() and checks.listening(self.s.dash_port):
                self.dash_shown = True
                show_window(self.env, "Proprio, live", f"http://localhost:{self.s.dash_port}/#{self.env.get('VIEW', 'camera')}")
        self.keep_bridge(now)
        self.tell_bridge_to_record()
        self.write_status(now)

    def due(self, what: str, now: float, every_s: float, later_s: float | None = None, after: int = 6) -> bool:
        """True at most once every `every_s` (every `later_s` once it has been tried `after` times):
        things are retried politely, not in a tight loop."""
        if now < self.next_try.get(what, 0.0):
            return False
        n = self.tries[what] = self.tries.get(what, 0) + 1
        self.next_try[what] = now + (later_s if later_s and n > after else every_s)
        return True

    def keep_bridge(self, now: float):
        """The phone bridge runs from the moment the page is up, so the checklist can see the
        stream, and is never stopped by Stop: the app stops streaming when its reader goes away."""
        if self.s.replay or not self.s.bridge.exists():
            return
        if self.bridge.alive():
            if now - self.bridge.started > 30:
                self.tries.pop("bridge", None)       # it has settled: the next time it dies is a first time again
            return
        if self.bridge.pid is None and self.bridge.adopt():
            return self.event(f"Carrying on with the phone bridge that is already running (pid {self.bridge.pid}).")
        if checks.bridge_running(self.s):
            return                   # one started by hand: leave it, a second would only fight it for the phone
        if self.due("bridge", now, 3.0, 15.0):
            self.bridge.forget()
            self.bridge.start(dict(self.env, PYTHONUNBUFFERED="1"))
            self.event(f"Phone bridge started (pid {self.bridge.pid}).")

    def keep_servers(self, now: float):
        for l in self.lines.get("servers", []):
            if l["level"] == RED and l["fix"] and self.due(l["id"], now, 60.0):
                self.event(f"{l['title']} is not running.")
                self.start_server(l["id"])

    def keep_airpods(self, now: float):
        first = (self.lines.get("airpods") or [{}])[0]
        if first.get("level") == GREEN:
            self.tries.pop("airpods", None)
        elif first.get("fix") and "connect_airpods" not in self.busy and self.due("airpods", now, 10.0, 30.0):
            self.do("connect_airpods")
        # Connected, but a mode has gone (stereo, three times in one evening): left alone they take
        # one to two and a half minutes to recover; disconnecting and connecting again takes five seconds.
        modes = next((l for l in self.lines.get("airpods", []) if l["id"] == "airpods_mic"), {})
        if modes.get("level") == RED and not modes.get("blocked"):
            self.mode_lost = self.mode_lost or now
            if now - self.mode_lost >= 10.0 and "reconnect_airpods" not in self.busy and self.due("airpods_modes", now, 90.0):
                self.event("The AirPods have lost a mode: disconnecting and connecting them again.")
                self.do("reconnect_airpods")
        else:
            self.mode_lost = 0.0

    def write_status(self, now: float):
        ok = self.mode == "checklist" or judge(self.pipeline.alive(), self.heartbeat(), self.pipeline.started, now,
                                               self.boot_s, self.frozen_s, self.failing_s)[0] == "ok"
        message = {"checklist": "Waiting for Start.", "stopping": "Stopping."}.get(
            self.mode, f"Running, {self.restarts} restart{'' if self.restarts == 1 else 's'}.")
        status = {"t": now, "state": self.mode, "ok": ok, "message": message, "pid": os.getpid(), "port": self.port,
                  "since": self.since, "restarts": self.restarts, "record": self.record}
        tmp = self.status_dir / "supervisor.tmp"
        tmp.write_text(json.dumps(status))
        os.replace(tmp, self.status_dir / "supervisor.json")

    def resume(self):
        """A keeper that was killed while the demo was running carries on with that run."""
        before = checks.read_status(self.status_dir / "supervisor.json") or {}
        adopted = self.pipeline.adopt()
        # Killed together with its pipeline (someone's `pkill python`): the demo was running a
        # moment ago, so it is started again. An old "running" left by a power cut is not.
        just_now = before.get("state") == "running" and (checks.age_of(before) or 1e9) < 120
        if adopted or just_now:
            self.mode, self.since = "running", before.get("since") or self.pipeline.started or time.time()
            self.restarts, self.dash_shown = int(before.get("restarts") or 0), True
            self.record = bool(before.get("record", self.record))
            self.event(f"Carrying on with the pipeline that is already running (pid {self.pipeline.pid})." if adopted
                       else "The demo was running when the keeper was killed: starting the pipeline again.")

    # --- the buttons ---------------------------------------------------------------------------

    def do(self, action: str) -> tuple[bool, str]:
        quick = {"start": self.start, "start_anyway": lambda: self.start(anyway=True), "stop": self.stop,
                 "toggle_record": self.toggle_record, "show_dashboard": self.show_dashboard, "quit": self.quit}
        slow = {"connect_airpods": self.connect_airpods, "reconnect_airpods": lambda: self.connect_airpods(again=True),
                "use_airpods_output": self.use_airpods_output, "pair_phone": self.pair_phone,
                "start_nemotron": lambda: self.start_server("nemotron"), "start_cosmos": lambda: self.start_server("cosmos"),
                "restart_bridge": self.restart_bridge, "stop_old_run": self.stop_old_run, "recheck_models": self.prove,
                "quiet_bluetooth": self.quiet_bluetooth, "quiet_desktop": self.quiet_desktop,
                "sound_check": lambda: self.ear_check("sound_check"), "mic_check": lambda: self.ear_check("mic_check")}
        if action in quick:
            return quick[action]()
        if action not in slow:
            return False, f"No such button: {action}"
        if action in self.busy:
            return True, "Already at it."
        if action in ("sound_check", "mic_check") and self.mode != "checklist":
            return False, "Stop the demo first: the pipeline is using the AirPods."
        self.busy[action] = time.time()

        def work():
            try:
                slow[action]()
            except Exception as e:
                self.event(f"{action} failed: {e!r}\n{traceback.format_exc()}")
            finally:
                self.busy.pop(action, None)
        threading.Thread(target=work, daemon=True, name=action).start()
        return True, "Working on it."

    def toggle_record(self) -> tuple[bool, str]:
        self.record = not self.record
        when = " from the next start" if self.mode != "checklist" else ""
        self.event(f"Recording switched {'on' if self.record else 'off'}{when}.")
        return True, "Recording " + ("on" if self.record else "off") + when + "."

    def show_dashboard(self) -> tuple[bool, str]:
        show_window(self.env, "Proprio, live", f"http://localhost:{self.s.dash_port}/#{self.env.get('VIEW', 'camera')}")
        return True, "Opened."

    def quit(self) -> tuple[bool, str]:
        def work():
            self.event("Quitting: stopping the pipeline and the phone bridge.")
            self.mode = "stopping"
            self.pipeline.stop(signal.SIGINT, self.stop_wait_s)
            self.stop_bridge()
            self.mode = "checklist"
            self.write_status(time.time())
            # As a service this would be started again the moment it ended: tell the session it is meant.
            try:
                subprocess.run(["systemctl", "--user", "stop", unit_name(self.port)], env=self.env, capture_output=True, timeout=10)
            except (OSError, subprocess.SubprocessError):
                pass                                      # not a service here: simply end
            self.stopped.set()
            if self.server is not None:
                self.server.shutdown()
        threading.Thread(target=work, daemon=True, name="quit").start()
        return True, "Quitting."

    def connect_airpods(self, again: bool = False):
        pods = checks.find_headset(self.s)
        if again:
            self.s.run(["bluetoothctl", "disconnect", pods["addr"]], 10.0)
            time.sleep(2.0)
        # With the AirPods out of reach, Bluetooth itself takes about 50 s to give up (measured on the GX10).
        rc, out = self.s.run(["bluetoothctl", "connect", pods["addr"]], 60.0)
        ok = rc == 0 and "successful" in out.lower()
        self.next_try["airpods"] = time.time() + 10.0      # the next try counts from the end of this one
        if ok and again:
            self.s.reconnected = time.time()       # mode trouble from before this is no longer news
        self.note("airpods", ok, f"{pods['name']}: connected." if ok else
                  f"{pods['name']} did not connect ({(out.splitlines() or ['no answer'])[-1].strip()}). Are they out of the case and near?")

    def quiet_bluetooth(self):
        """Stop the radio looking around: close the Settings window that keeps it scanning, and stop announcing."""
        self.s.run(["pkill", "-x", "gnome-control-c"])                  # the name the kernel keeps: 15 letters
        rc, out = self.s.run(["bluetoothctl", "discoverable", "off"], 8.0)
        self.note("bluetooth", rc == 0, "Closed Settings and stopped Bluetooth announcing itself." if rc == 0
                  else f"Could not quiet Bluetooth ({(out.splitlines() or ['no answer'])[-1].strip()}).")

    def quiet_desktop(self, say: bool = True):
        """No banners, alert sounds or Files windows on the projector."""
        bad = [f"{schema} {key}" for schema, key, value, _ in checks.QUIET
               if self.s.run(["gsettings", "get", schema, key])[1].strip() != value
               and self.s.run(["gsettings", "set", schema, key, value])[0] != 0]
        if say or bad:
            self.note("desktop", not bad, "The desktop is quiet." if not bad else "Could not set: " + ", ".join(bad) + ".")

    def use_airpods_output(self):
        sink = checks.bluez_sink(self.s)
        if sink is None:
            return self.note("airpods_output", False, "The sound server has no AirPods output yet. Press Reconnect on the line below.")
        rc, out = self.s.run(["wpctl", "set-default", str(sink)])
        self.note("airpods_output", rc == 0, "Sound now goes to the AirPods." if rc == 0 else f"Could not switch the sound ({out}).")

    def pair_phone(self):
        rc, out = self.s.run(["idevicepair", "pair"], 15.0)
        self.note("phone_trust", rc == 0, (out.splitlines() or ["No answer from the phone."])[-1].replace("SUCCESS: ", "").replace("ERROR: ", ""))

    def start_server(self, id: str):
        """Start a model server detached, with its output where tools/show_models.sh looks for it."""
        url = self.s.nemotron_url if id == "nemotron" else self.s.cosmos_url
        script, log_name, port_var = (("run_nemotron.sh", "nemotron.log", "NEMOTRON_PORT") if id == "nemotron"
                                      else ("run_cosmos.sh", "cosmos_8b.log", "COSMOS_PORT"))
        logs = Path(self.env.get("PROPRIO_SERVER_LOGS") or self.s.home / "models")
        logs.mkdir(parents=True, exist_ok=True)
        with open(logs / log_name, "ab") as f:
            p = subprocess.Popen(["bash", str(self.root / "tools" / script)], cwd=self.root, stdin=subprocess.DEVNULL, stdout=f,
                                 stderr=subprocess.STDOUT, start_new_session=True,
                                 env=dict(self.env, **{port_var: str(checks.port_of(url) or "")}))
        self.note(id, True, f"{id.capitalize()} is starting (pid {p.pid}); its model takes about half a minute to load.")

    def stop_bridge(self):
        """Its own `stop` lets go of the phone cleanly, whoever started it; then whatever is left."""
        if self.s.bridge.exists():
            self.s.run([sys.executable, str(self.s.bridge), "stop"], 60.0)
        self.bridge.stop(signal.SIGINT, 5.0)

    def restart_bridge(self):
        self.stop_bridge()
        self.next_try.pop("bridge", None)
        self.note("phone_stream", True, "The phone bridge was restarted. If the app shows Start again, tap it.")

    def stop_old_run(self):
        for port in (self.s.dash_port, self.s.voice_port):
            pid, args = checks.holder(self.s, port)
            if pid and "main.py" in args and pid != self.pipeline.pid:
                os.kill(pid, signal.SIGINT)           # one Ctrl-C, as tools/stop_live.sh does: it is saving
                return self.note("ports", True, f"Asked the old run (pid {pid}) to stop; it closes its recording first.")
        self.note("ports", False, "No old run of main.py is holding the ports.")

    def prove(self):
        """Load every model with the network blocked, in a separate Python so this one stays light."""
        key = checks.files_key(checks.model_files(self.s))
        self.busy["prove"] = time.time()
        try:
            p = subprocess.run(["nice", sys.executable, str(self.root / "launcher_checks.py"), "--prove"], cwd=self.root,
                               env=dict(self.env, TRACE="0"), capture_output=True, text=True, timeout=600,
                               stdin=subprocess.DEVNULL)
            proof = json.loads(p.stdout.strip().splitlines()[-1])
        except Exception as e:
            proof = {"ok": False, "results": [], "note": f"the test itself failed: {e!r}"}
        finally:
            self.busy.pop("prove", None)
        self.proof = dict(proof, key=key)
        (self.status_dir / "models_proof.json").write_text(json.dumps(self.proof))
        self.event("Models loaded with the network blocked: " + ("all of them." if proof.get("ok") else
                   "not all. " + "; ".join(f"{r['name']}: {r['note']}" for r in proof.get("results", []) if not r["ok"])))
        self.lines["models"] = self.look("models")

    def ear_check(self, id: str):
        """Run tools/ear_check.py and keep its last line, which is its verdict."""
        which = "sound" if id == "sound_check" else "mic"
        try:
            p = subprocess.Popen([sys.executable, str(self.root / "tools" / "ear_check.py"), which], cwd=self.root,
                                 env=dict(self.env, TRACE="0"), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL, text=True)
            self.busy["pid"] = p.pid
            out, _ = p.communicate(timeout=120)
            got = json.loads(out.strip().splitlines()[-1])
        except Exception as e:
            got = {"ok": False, "message": f"The test could not run ({e!r})."}
        finally:
            self.busy.pop("pid", None)
        level = "none" if got.get("ok") is None else GREEN if got.get("ok") else RED
        self.ear[id] = {"level": level, "message": str(got.get("message", ""))}
        self.event(f"{'Sound' if which == 'sound' else 'Microphone'} check: {got.get('message')}")

    # --- running -----------------------------------------------------------------------------------

    def run_forever(self):
        self.resume()
        for group, every_s in self.GROUPS:
            threading.Thread(target=self.watch, args=(group, every_s), daemon=True, name=f"check-{group}").start()
        files = checks.model_files(self.s)
        if all(f["ok"] for f in files) and self.proof.get("key") != checks.files_key(files):
            threading.Thread(target=self.prove, daemon=True, name="prove").start()
        threading.Thread(target=self.quiet_desktop, args=(False,), daemon=True, name="quiet").start()
        self.server = ThreadingHTTPServer((self.env.get("PROPRIO_HOST", "127.0.0.1"), self.port), make_handler(self))
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True, name="page").start()
        self.event(f"Keeper up (pid {os.getpid()}): http://localhost:{self.port}/")
        while not self.stopped.is_set():
            try:
                self.tick()
            except Exception:
                self.event(f"The keeper stumbled and carries on:\n{traceback.format_exc()}")
            self.stopped.wait(1.0)


def make_handler(sup: Supervisor):
    remote_ok = sup.env.get("PROPRIO_REMOTE") == "1"

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def local(self) -> bool:
            return remote_ok or self.client_address[0] in ("127.0.0.1", "::1", "::ffff:127.0.0.1")

        def send(self, code: int, body: bytes, kind: str):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def json(self, obj, code: int = 200):
            self.send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/api/state":
                return self.json(sup.state(self.local()))
            if path == "/api/log":
                which = {"keeper": sup.out / "supervisor.log", "bridge": sup.bridge.log}.get(
                    self.path.partition("which=")[2], sup.pipeline.log)
                return self.json({"lines": tail(which, 200)})
            # The page's own three files, and the dashboard's fonts and icon so both look alike.
            if path == "/":
                path = "/index.html"
            folder = ROOT / "dashboard" if path.startswith("/fonts/") or path == "/icon.svg" else PAGE_DIR
            full = (folder / path.lstrip("/")).resolve()
            if folder.resolve() not in full.parents or not full.is_file():
                return self.send(404, b"not found", "text/plain")
            self.send(200, full.read_bytes(), mimetypes.guess_type(str(full))[0] or "application/octet-stream")

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(min(n, 10_000))
            if self.path != "/api/do":
                return self.send(404, b"not found", "text/plain")
            # Only from this machine's own screen, and only from this page: the header cannot be
            # sent by another site's page, and a stranger on the hall's Wi-Fi cannot press Stop.
            if not self.local() or self.headers.get("X-Proprio") != "1":
                return self.json({"ok": False, "message": "The buttons work on the GX10's own screen."}, 403)
            try:
                action = str(json.loads(body or b"{}").get("action", ""))
            except ValueError:
                action = ""
            ok, message = sup.do(action)
            self.json({"ok": ok, "message": message})

    return Handler


# ---------------------------------------------------------------------------
# The browser window on the GX10's screen

def show_window(env: dict, title: str, url: str, wait_s: float = 12.0) -> bool:
    """Bring the Firefox window whose title starts with `title` to the front, or open one at
    `url` if there is none. Never a second window for the same page."""
    if env.get("PROPRIO_OPEN", "1") == "0" or not sys.platform.startswith("linux"):
        return False

    def find() -> str | None:
        try:
            got = subprocess.run(["xdotool", "search", "--name", f"^{re.escape(title)}"], env=env, capture_output=True,
                                 text=True, timeout=5).stdout.split()
        except (OSError, subprocess.SubprocessError):
            return None
        return got[-1] if got else None

    window = find()
    if window is None:
        try:
            subprocess.Popen(["firefox", "--new-window", url], env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True)
        except OSError as e:
            print(f"could not open Firefox ({e}); the page is {url}", file=sys.stderr)
            return False
        end = time.monotonic() + wait_s
        while window is None and time.monotonic() < end:
            time.sleep(0.5)
            window = find()
    if window is not None:
        subprocess.run(["xdotool", "windowactivate", window], env=env, capture_output=True, timeout=5)
    return window is not None


# ---------------------------------------------------------------------------
# The command line

def ask(port: int, action: str | None = None, timeout_s: float = 3.0):
    """The keeper's state, or press one of its buttons; None if no keeper answers."""
    url = f"http://127.0.0.1:{port}/api/" + ("do" if action else "state")
    req = urllib.request.Request(url, json.dumps({"action": action}).encode() if action else None,
                                 {"Content-Type": "application/json", "X-Proprio": "1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            return json.loads(r.read())
    except (OSError, ValueError):
        return None


def unit_name(port: int) -> str:
    return f"proprio-keeper-{port}.service"


def start_keeper(port: int, env: dict, log: Path):
    """Start the keeper detached from this program. As a service of the desktop session when that
    is possible, because then it is started again if something kills it; KillMode=process so that
    restarting the keeper never takes the pipeline or the phone bridge down with it."""
    me = [sys.executable, str(ROOT / "launcher.py"), "serve"]
    unit = unit_name(port)
    try:
        subprocess.run(["systemctl", "--user", "reset-failed", unit], env=env, capture_output=True, timeout=10)
        got = subprocess.run(["systemd-run", "--user", "--quiet", "--collect", f"--unit={unit}",
                              "--description=Proprio keeper (launcher.py)", f"--working-directory={ROOT}",
                              "-p", "Restart=always", "-p", "RestartSec=3", "-p", "KillMode=process",
                              "-p", f"StandardOutput=append:{log}", "-p", f"StandardError=append:{log}",
                              *[f"--setenv={k}={env[k]}" for k in ("DISPLAY", "XAUTHORITY", "DBUS_SESSION_BUS_ADDRESS") if k in env],
                              *me], env=env, capture_output=True, text=True, timeout=15)
        if got.returncode == 0:
            return
        print(f"not as a service ({got.stderr.strip() or got.returncode}); starting it plainly", file=sys.stderr)
    except (OSError, subprocess.SubprocessError) as e:
        print(f"not as a service ({e}); starting it plainly", file=sys.stderr)
    with open(log, "ab") as f:
        subprocess.Popen(me, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)


def serve(port: int):
    """The keeper, in this process. Only one at a time: a second one leaves at once."""
    out = ROOT / "out"
    out.mkdir(exist_ok=True)
    held = open(out / "supervisor.lock", "w")
    try:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("a keeper is already running", file=sys.stderr)
        return
    env = load_env()
    os.environ.update(env)                 # the checks' own commands (wpctl, bluetoothctl) need it too
    signal.signal(signal.SIGHUP, signal.SIG_IGN)      # a terminal closing is not a reason to stop
    sup = Supervisor(env=env, port=port)
    signal.signal(signal.SIGTERM, lambda *_: sup.stopped.set())
    try:
        sup.run_forever()
    except KeyboardInterrupt:
        pass
    # Ending the keeper leaves the pipeline and the phone bridge running: the next keeper carries
    # on with them. `python launcher.py quit` is the way to stop everything.
    sup.event("Keeper ending.")


def open_page(port: int) -> int:
    """What the icon does."""
    out = ROOT / "out"
    out.mkdir(exist_ok=True)
    env = load_env()
    with open(out / "open.lock", "w") as one_at_a_time:      # two double-clicks: the second waits for the first
        fcntl.flock(one_at_a_time, fcntl.LOCK_EX)
        if ask(port) is None:
            start_keeper(port, env, out / "supervisor.log")
            end = time.monotonic() + 15.0
            while ask(port) is None and time.monotonic() < end:
                time.sleep(0.2)
        if ask(port) is None:
            print(f"the keeper did not come up; see {out / 'supervisor.log'}", file=sys.stderr)
            return 1
        show_window(env, "Proprio, start", f"http://localhost:{port}/")
    print(f"http://localhost:{port}/")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    port = int(load_env().get("PROPRIO_PORT", PORT))
    what = argv[0] if argv else "open"
    if what == "open":
        return open_page(port)
    if what == "serve":
        serve(port)
        return 0
    if what == "status":
        state = ask(port)
        print(json.dumps(state, indent=1) if state else "no keeper is running")
        return 0 if state else 1
    if what in ("stop", "quit", "start", "start_anyway"):
        got = ask(port, what, timeout_s=10.0)
        print(got["message"] if got else "no keeper is running")
        return 0 if got and got.get("ok") else 1
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
