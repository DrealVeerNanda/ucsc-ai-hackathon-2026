"""Proprio main loop.

Every lane's module can be fake or real, so pieces swap in one at a time:

    python main.py                                   # all fake
    python main.py --guidance real                   # lane 3 tunes audio on fake scenes
    python main.py --stream real --perception real   # lanes 1 + 2 live, printed audio
    python main.py --all-real                        # the demo

The judge dashboard opens in the browser at http://localhost:8642 (--ui web, the default).
--ui cv shows the OpenCV debug window instead, --ui both shows both, --ui none neither.
Keys (dashboard or window): q / esc quit, t ask for a new target. Fake perception adds
hand keys (see fakes.py). Each run writes a per-frame CSV to logs/ and prints a latency
summary on exit.

Once started it keeps going: an exception in the camera, perception, guidance, voice or the
views is printed and the loop carries on. Once a second it writes what it is doing to
out/status/pipeline.json, which is how launcher.py tells a run that is waiting for the wearer
to speak from one that is stuck.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import json
import os
import signal
import sys
import threading
import time
import traceback
import webbrowser
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import debug_view
import fakes
import tracing
from dashboard import Dashboard
from interfaces import hand_error

LANES = ("stream", "perception", "guidance", "voice")
STAGES = ("stream", "perception", "guidance", "view")
FAKE_KEYS_HINT = "arrows/w/s move hand   o occlude   h hide hand   n jitter   r reset"
# What guidance says on entering each state, per the team plan, for guidance modules
# that neither report phrases nor let us listen to their sink.
STATE_PHRASES = {"waiting_hand": "show_me_your_hand", "grab": "grab"}
# Clip names (audio/*.wav) as the wearer hears them; words not listed are capitalised.
CLIP_TEXT = {"looking": "Looking", "found_it": "Found it", "show_me_your_hand": "Show me your hand",
             "grab": "Grab", "i_lost_it": "I lost it", "i_cant_see_that": "I can't see that"}


STATUS_DIR = Path(__file__).resolve().parent / "out" / "status"
# A step other than waiting for the wearer that has not come back for this long is stuck. The
# slowest honest ones are opening the camera when the phone is not streaming (10 s, in stream.py)
# and a new target's first frame (3 s in the last live run).
STUCK_S = 30.0
STARTING_S = 120.0      # before the first frame is through: importing torch and loading the models, cold
NO_CAMERA_S = 2.0       # this long without a frame while it should be guiding: say so
FAILING_S = 5.0         # a step that has failed on every try for this long is not a blip
CAMERA_LOST_S = 1.5      # no picture for this long in the middle of a reach: tell the wearer
RESUME_S = 90.0         # a run that died mid-reach at most this long ago is carried on with, not asked again


def clip_text(name: str) -> str:
    return CLIP_TEXT.get(name, name.replace("_", " ").capitalize())


class Faults:
    """What went wrong, per step. The first failure of a step is printed whole; while it keeps
    failing, one line every five seconds, so a broken step cannot bury the log."""

    def __init__(self, every_s: float = 5.0):
        self.counts: dict = {}
        self.since: dict = {}        # step -> time.monotonic() of the first failure in a row
        self.last: str | None = None
        self._every_s = every_s
        self._printed: dict = {}

    def failed(self, step: str, e: BaseException):
        now = time.monotonic()
        self.counts[step] = self.counts.get(step, 0) + 1
        self.since.setdefault(step, now)
        self.last = f"{step}: {e.__class__.__name__}: {e}"
        if step not in self._printed:
            print(f"[main] {step} failed, carrying on:\n{traceback.format_exc()}", file=sys.stderr, flush=True)
        elif now - self._printed[step] >= self._every_s:
            print(f"[main] {step} has failed {self.counts[step]} times, last: {e.__class__.__name__}: {e}",
                  file=sys.stderr, flush=True)
        else:
            return
        self._printed[step] = now

    def ok(self, step: str):
        self.since.pop(step, None)

    def failing(self, now: float) -> dict:
        """Steps that are failing right now -> for how many seconds."""
        return {step: round(now - t, 1) for step, t in list(self.since.items())}


class Heartbeat:
    """Writes pipeline.json once a second from its own thread: what the loop is doing, for how
    long, and how the run is going. The loop only says which step it is on (`beat.now("perception")`),
    so a loop that is stuck shows up as a step that never ends, and a frozen process as a file
    that stops changing. Waiting for the wearer to speak can take minutes and is never "stuck"."""

    def __init__(self, folder: str | Path | None, faults: Faults | None = None, every_s: float = 1.0):
        self.path = None if not folder or str(folder) == "off" else Path(folder) / "pipeline.json"
        self.previous = self._read()                 # what the run before this one last reported
        self.faults = faults or Faults()
        self.every_s = every_s
        self.started = time.time()
        self.at = ("starting", time.monotonic())     # (step, since when), replaced whole
        self.warm = False                            # a frame has been through every step
        self.frames, self.fps, self.last_frame = 0, 0.0, None
        self.target = self.guidance = self.heard = self.said = None
        self.voice = None
        self._seen = 0
        self._stop = threading.Event()
        self._thread = None

    def _read(self) -> dict:
        try:
            got = json.loads(self.path.read_text())
            return got if isinstance(got, dict) else {}
        except Exception:
            return {}

    def now(self, step: str):
        self.at = (step, time.monotonic())

    def start(self) -> "Heartbeat":
        if self.path is not None and self._thread is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._thread = threading.Thread(target=self._run, daemon=True, name="heartbeat")
            self._thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            try:
                self.write(self.snapshot())
            except Exception as e:                   # a full disk must not end the run
                print(f"[main] could not write {self.path}: {e}", file=sys.stderr)
            self._stop.wait(self.every_s)

    def snapshot(self) -> dict:
        now = time.monotonic()
        step, since = self.at
        for e in tracing.events_since(self._seen)[0]:
            if e.get("name") == "whisper" and e.get("text"):
                self.heard = e["text"]
            elif e.get("name") == "say" and e.get("text"):
                self.said = e["text"]
        self._seen = tracing.events_since(self._seen)[1]
        heard = self.heard or getattr(self.voice, "last_transcript", None)
        age = None if self.last_frame is None else now - self.last_frame
        # While it waits for the wearer no other step runs, so none of them can be "still failing".
        failing = {k: v for k, v in self.faults.failing(now).items() if v >= FAILING_S and (step != "listening" or k == "voice")}
        what = f"the {self.target}" if self.target else "nothing yet"
        if failing:
            worst = max(failing, key=failing.get)
            state, message = "failing", f"{worst.capitalize()} has been failing for {failing[worst]:.0f} s ({self.faults.last})."
        elif step == "listening":
            state, message = "listening", "Waiting for the wearer to say what to pick up."
        elif now - since > (STUCK_S if self.warm else STARTING_S):
            state, message = "stuck", f"Stuck in {step} for {now - since:.0f} s."
        elif not self.warm:
            state, message = "starting", "Starting: loading the models and waiting for the first frame."
        elif age is not None and age > NO_CAMERA_S:
            state, message = "no_camera", f"No camera frame for {age:.0f} s."
        else:
            doing = str(self.guidance or "starting").replace("_", " ")
            state, message = "running", f"Looking for {what}: {doing}, {self.fps:.0f} frames a second."
        return {"t": round(time.time(), 3), "state": state, "ok": state in ("listening", "running"), "message": message,
                "pid": os.getpid(), "started": round(self.started, 3), "doing": step, "doing_s": round(now - since, 2),
                "warm": self.warm, "target": self.target, "guidance": self.guidance, "frames": self.frames,
                "fps": round(self.fps, 1), "last_frame_age_s": None if age is None else round(age, 2),
                "heard": heard, "said": self.said, "errors": dict(self.faults.counts),
                "failing": failing, "last_error": self.faults.last}

    def write(self, status: dict):
        if self.path is None:
            return
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(status))
        os.replace(tmp, self.path)                   # readers never see half a file

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            try:
                self.write(dict(self.snapshot(), state="stopped", ok=False, message="Stopped."))
            except Exception:
                pass


def patient(get_frame, wait_s: float = 0.05):
    """(get_frame that waits for the next frame, can it wait?). stream.get_frame can block until
    a frame arrives; the fakes cannot, and return None at once."""
    try:
        waits = "wait" in inspect.signature(get_frame).parameters
    except (TypeError, ValueError):
        waits = False
    return ((lambda: get_frame(wait=wait_s)) if waits else get_frame), waits


def listen_to_guidance(g):
    """Record every clip guidance plays, so the dashboard shows exactly what the wearer hears.
    Wraps the sink of guidance.py's default instance; a public pop_phrases() would replace this."""
    spoken = []
    inst = g._get()
    play = inst.sink.say

    def say(name, now=False):
        spoken.append(name)
        return play(name, now=now)

    inst.sink.say = say

    def pop():
        out = spoken[:]
        spoken.clear()
        return out

    return pop


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for lane in LANES:
        ap.add_argument(f"--{lane}", choices=("fake", "real"), default="fake")
    ap.add_argument("--all-real", action="store_true", help="use every real module")
    ap.add_argument("--tones", action="store_true", help="real guidance plays tones instead of spoken words")
    ap.add_argument("--target", default="mug", help="target for fake voice")
    ap.add_argument("--ask", action="store_true", help="fake voice: type the target in the terminal")
    ap.add_argument("--image", help="fake stream: replay this image instead of the synthetic table")
    ap.add_argument("--fps", type=float, default=30.0, help="fake stream frame rate")
    ap.add_argument("--ui", choices=("web", "cv", "both", "none"), default="web",
                    help="judge dashboard in the browser, OpenCV debug window, both, or neither")
    ap.add_argument("--host", default="0.0.0.0", help="dashboard bind address")
    ap.add_argument("--port", type=int, default=8642, help="dashboard port")
    ap.add_argument("--no-open", action="store_true", help="do not open the dashboard in a browser")
    ap.add_argument("--no-view", action="store_true", help="same as --ui none")
    ap.add_argument("--fullscreen", action="store_true", help="OpenCV window fullscreen")
    ap.add_argument("--frames", type=int, default=0, help="stop after N frames (smoke tests)")
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--status-dir", default=str(STATUS_DIR),
                    help="where pipeline.json is written once a second; \"off\" for nowhere")
    args = ap.parse_args(argv)
    if args.all_real:
        for lane in LANES:
            setattr(args, lane, "real")
    if args.no_view:
        args.ui = "none"
    return args


def load_modules(args) -> SimpleNamespace:
    m = SimpleNamespace(fake_locate=None, reset=None)
    if args.stream == "real":
        m.get_frame = importlib.import_module("stream").get_frame
    else:
        m.get_frame = fakes.FakeStream(args.image, args.fps).get_frame
    if args.perception == "real":
        perception = importlib.import_module("perception")
        perception.in_use = True     # so an option picked before the first request is taken, not refused
        m.locate = perception.locate
    else:
        m.fake_locate = fakes.FakeLocate()
        m.locate = m.fake_locate.locate
    if args.guidance == "real":
        m.guidance = importlib.import_module("guidance")
    else:
        m.guidance = fakes.FakeGuidance()
    m.update, m.reset = m.guidance.update, getattr(m.guidance, "reset", None)
    m.pop_phrases = getattr(m.guidance, "pop_phrases", None)
    if m.pop_phrases is None and hasattr(m.guidance, "_get"):
        m.pop_phrases = listen_to_guidance(m.guidance)
    if args.voice == "real":
        m.voice = importlib.import_module("voice")
    else:
        m.voice = fakes.FakeVoice(args.target, args.ask)
    m.listen = m.voice.listen_for_target
    return m


class RunLog:
    FIELDS = ["t_wall", "t_recv", "frame_t", "age_ms", "e2e_ms", *[f"{s}_ms" for s in STAGES],
              "target", "state", "obj_vis", "hand_vis", "ox", "oy", "oz", "hx", "hy", "hz", "dist_cm"]

    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, time.strftime("run_%Y%m%d_%H%M%S.csv"))
        self.f = open(self.path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(self.FIELDS)
        self.n = 0

    def row(self, frame, scene, t_recv, age_ms, e2e_ms, stages, target, state):
        o = scene.object_xyz if scene.object_xyz is not None else (None,) * 3
        h = scene.hand_xyz if scene.hand_xyz is not None else (None,) * 3
        err = hand_error(scene)
        self.w.writerow([
            f"{time.time():.3f}", f"{t_recv:.4f}", f"{frame.t:.4f}", f"{age_ms:.1f}", f"{e2e_ms:.1f}",
            *[f"{stages.get(s, 0):.2f}" for s in STAGES],
            target, state, int(scene.object_visible), int(scene.hand_visible),
            *[_fmt(v) for v in o], *[_fmt(v) for v in h],
            _fmt(np.linalg.norm(err) * 100 if err is not None else None),
        ])
        self.n += 1
        if self.n % 30 == 0:
            self.f.flush()

    def close(self):
        self.f.close()


def _fmt(v):
    return "" if v is None else f"{float(v):.4f}"


class Stats:
    def __init__(self, keep: int = 20000):
        self.series = {k: deque(maxlen=keep) for k in (*STAGES, "e2e", "loop")}
        self.fps = 0.0
        self.last_t = None

    def add(self, stages: dict, e2e_ms: float, now: float):
        for k, v in stages.items():
            self.series[k].append(v)
        self.series["e2e"].append(e2e_ms)
        if self.last_t is not None:
            dt = now - self.last_t
            self.series["loop"].append(dt * 1000)
            if dt > 0:
                self.fps = 1 / dt if self.fps == 0 else 0.9 * self.fps + 0.1 / dt
        self.last_t = now

    def report(self, e2e_label: str) -> str:
        lines = [f"{'stage':<12}{'p50 ms':>9}{'p95 ms':>9}{'max ms':>9}   n"]
        for k in (*STAGES, "e2e", "loop"):
            v = np.array(self.series[k])
            if v.size:
                name = e2e_label if k == "e2e" else k
                lines.append(f"{name:<12}{np.percentile(v, 50):9.1f}{np.percentile(v, 95):9.1f}{v.max():9.1f}   {v.size}")
        loop = np.array(self.series["loop"])
        if loop.size:
            lines.append(f"mean loop rate {1000 / loop.mean():.1f} fps")
        return "\n".join(lines)


FAILED = object()        # what attempt() returns when the step raised


def main(argv=None):
    args = parse_args(argv)
    # Spoken words (left, right, up, down, forward, back) are the default guidance.
    os.environ.setdefault("GUIDANCE_WORDS", "0" if args.tones else "1")
    # A plain `kill` ends Python without the summary below or the last status; make it a normal exit.
    if signal.getsignal(signal.SIGTERM) is signal.SIG_DFL:
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    if signal.getsignal(signal.SIGINT) is signal.SIG_IGN:
        # Started with & or nohup by a script: the shell told us to ignore Ctrl-C, and
        # tools/stop_live.sh, which sends it, would wait for nothing. Take it as usual.
        signal.signal(signal.SIGINT, signal.default_int_handler)
    beat = Heartbeat(args.status_dir).start()        # before the modules load: that can take a while
    try:
        mods = load_modules(args)
    except BaseException:
        beat.stop()
        raise
    run(args, mods, beat)


def run(args, mods, beat: Heartbeat | None = None):
    """The loop. Returns when asked to quit (q, esc, Ctrl-C, --frames); nothing else ends it."""
    beat = (beat or Heartbeat(args.status_dir)).start()
    faults = beat.faults
    beat.voice = getattr(mods, "voice", None)
    view = args.ui in ("cv", "both")
    modes = "  ".join(f"{lane}:{getattr(args, lane)}" for lane in LANES)
    modes_map = {lane: getattr(args, lane) for lane in LANES}
    print(f"[main] {modes}")

    log = RunLog(args.log_dir)
    stats = Stats()
    if view:
        debug_view.open_window(args.fullscreen)
    dash = None
    if args.ui in ("web", "both"):
        dash = Dashboard(args.host, args.port)
        url = dash.start()
        os.environ["DASHBOARD_PORT"] = str(args.port)      # voice.py presses "new target" through it
        print(f"[main] dashboard at {url}")
        if not args.no_open:
            webbrowser.open(url)

    next_frame, waits = patient(mods.get_frame)
    pop_phrases = mods.pop_phrases
    run = {"status": "idle", "started": None, "elapsed_s": None, "target": None}
    last = None              # (frame, scene, info) for redraws while no frames arrive
    last_shown = 0.0
    t_recv_src = None
    prev_state = None
    target = None
    n = 0

    def attempt(step: str, fn, *a, default=None):
        """fn(*a), or `default` if it raised: one bad frame or a module having a moment must not
        end the run. Ctrl-C and sys.exit are not caught."""
        try:
            out = fn(*a)
        except Exception as e:
            faults.failed(step, e)
            return default
        faults.ok(step)
        return out

    def show(frame, scene, info) -> str | None:
        """Both views; the key pressed in the OpenCV window, if any."""
        if dash:
            attempt("dashboard", dash.publish, frame, scene, info)
        return attempt("view", debug_view.show, frame, scene, info) if view else None

    class Camera:
        """Tells the wearer when the picture has gone and when it is back. Mid-reach, silence
        after the camera drops is the worst thing to hear: they cannot know to hold still."""
        since, told, had, told_t = None, False, False, 0.0

        def missing(self, now: float):
            self.since = now if self.since is None else self.since
            say = getattr(mods.guidance, "say_text", None)
            if self.had and target and say and now - self.since >= CAMERA_LOST_S and (not self.told or now - self.told_t >= 15.0):
                tracing.event("camera", lost=True, after_s=round(now - self.since, 1))
                attempt("guidance", say, "Still no camera. Hold still." if self.told else "I have lost the camera. Hold still.")
                self.told, self.told_t = True, now

        def arrived(self):
            if self.told:
                say = getattr(mods.guidance, "say_text", None)
                tracing.event("camera", lost=False)
                if say:
                    attempt("guidance", say, "The camera is back.")
            self.since, self.told, self.had = None, False, True

    camera = Camera()

    def resume_target() -> str | None:
        """The object the run before this one was guiding to, if that run died in the middle of
        a reach moments ago and the keeper has started this one in its place. A blindfolded
        wearer then hears the reach carry on, not "what do you want to grab?" again."""
        prev = beat.previous
        if os.environ.get("PROPRIO_KEEPER") != "1" or not prev.get("target"):
            return None
        try:
            age = time.time() - float(prev.get("t") or 0)
        except (TypeError, ValueError):
            return None
        if prev.get("state") in ("stopped", "listening") or not 0 <= age <= RESUME_S:
            return None                              # it was stopped on purpose, or was not reaching, or is old news
        if prev.get("guidance") not in ("searching", "found", "waiting_hand", "guiding"):
            return None
        target = str(prev["target"])
        print(f"[main] carrying on with {target!r}: the run before this one stopped {age:.0f} s ago in the middle of it")
        tracing.event("resume", target=target, after_s=round(age, 1))
        say = getattr(mods.guidance, "say_text", None)
        if mods.reset:
            attempt("guidance", mods.reset, target)
        if say:                                      # after reset: its "Looking" cuts off whatever is being said
            attempt("guidance", say, f"Still looking for the {target}.")
        run.update(status="running", started=time.monotonic(), elapsed_s=0.0, target=target)
        beat.target = target
        beat.now("loop")
        return target

    def request_target(prev: str | None = None) -> str:
        beat.now("listening")
        if view:
            attempt("view", debug_view.show_message, "LISTENING...")
        if dash and last is not None:
            f, s, info = last
            attempt("dashboard", dash.publish, f, s, dict(info, state="listening"))
        target = None
        while not target:
            t = time.monotonic()
            target = attempt("voice", mods.listen, default=FAILED)
            # Asked again at once, a voice module that fails or returns nothing straight away would spin.
            pause = 1.0 if target is FAILED else 0.2
            if not target or target is FAILED:
                target = None
                time.sleep(max(0.0, pause - (time.monotonic() - t)))
        print(f"[main] target: {target!r}" + (f" (was {prev!r})" if prev else ""))
        if dash:
            dash.pop_keys()                          # "new target" pressed while it was asking: this answer is that target
        if mods.reset:
            attempt("guidance", mods.reset, target)
        if mods.fake_locate:
            mods.fake_locate.reset_hand()
        run.update(status="running", started=time.monotonic(), elapsed_s=0.0, target=target)
        beat.target = target
        beat.now("loop")
        return target

    def handle_key(key: str | None) -> bool:
        """Returns False to quit."""
        nonlocal target
        if key in ("q", "esc"):
            return False
        if key == "t":
            target = request_target(target)
        elif mods.fake_locate is not None:
            mods.fake_locate.handle_key(key)
        return True

    def handle_web_keys() -> bool:
        keys = dash.pop_keys() if dash else []
        return all(handle_key(k) for k in dict.fromkeys(keys))      # "new target" pressed twice is asked once

    def redraw() -> str | None:
        """No new picture this turn: keep the views alive with the last one."""
        nonlocal last_shown
        now = time.monotonic()
        if last is None or now - last_shown <= 0.1:
            return None
        f, s, info = last
        info["age_ms"] = (now - info["t_recv"]) * 1000
        last_shown = now
        return show(f, s, info)

    try:
        target = resume_target() or request_target()
        while True:
            t0 = time.perf_counter()
            beat.now("stream")
            frame = attempt("stream", mods.get_frame)
            if frame is None and waits:
                # Nothing new yet: wait for the next frame inside the stream instead of asking
                # again and again. The wait is not the stream's own work, so it is left out of
                # its time: that starts when the frame arrived.
                frame = attempt("stream", next_frame)
                if frame is not None and getattr(frame, "t_recv", None) is not None:
                    t0 = time.perf_counter() - (time.monotonic() - frame.t_recv)
            if frame is None:
                camera.missing(time.monotonic())
                # A source that cannot wait (the fakes, a camera that is not open yet) came back
                # at once: sleep here, so a missing frame never spins the loop.
                spent = time.perf_counter() - t0
                time.sleep(max(0.002, 0.02 - spent) if waits else 0.002)
                if not handle_key(redraw()) or not handle_web_keys():
                    break
                continue
            camera.arrived()
            t1 = time.perf_counter()

            # Prefer lane 1's arrival stamp; otherwise stamp it here (misses transport time).
            if getattr(frame, "t_recv", None) is not None:
                t_recv, src = frame.t_recv, "e2e"
            else:
                t_recv, src = time.monotonic(), "proc"
            if src != t_recv_src:
                t_recv_src = src
                if src == "proc":
                    print("[main] frames carry no t_recv; latency covers processing only")

            beat.now("perception")
            scene = attempt("perception", mods.locate, frame, target, default=FAILED)
            t2 = time.perf_counter()
            beat.now("guidance")
            state = FAILED if scene is FAILED else attempt("guidance", mods.update, scene, default=FAILED)
            if state is FAILED:
                # Nothing to guide on this frame. The sound goes quiet by itself when it stops
                # being updated; the views and the keys carry on.
                beat.last_frame = time.monotonic()
                if not handle_key(redraw()) or not handle_web_keys():
                    break
                continue
            t3 = time.perf_counter()
            now = time.monotonic()
            e2e_ms = (now - t_recv) * 1000

            stages = {"stream": (t1 - t0) * 1000, "perception": (t2 - t1) * 1000,
                      "guidance": (t3 - t2) * 1000}

            if pop_phrases:
                phrases = attempt("guidance", pop_phrases, default=[])
            else:
                phrases = [STATE_PHRASES[state]] if state != prev_state and state in STATE_PHRASES else []
            if dash:
                for name in phrases:
                    attempt("dashboard", dash.say, clip_text(name))
            if run["status"] == "running":
                run["elapsed_s"] = now - run["started"]
                if state == "grab":
                    run["status"] = "done"

            info = {
                "target": target, "state": state, "fps": stats.fps,
                "age_ms": e2e_ms, "e2e_ms": e2e_ms, "e2e_label": t_recv_src,
                "stages": dict(stages, view=stats.series["view"][-1] if stats.series["view"] else 0.0),
                "modes": modes, "modes_map": modes_map, "t_recv": t_recv,
                "keys_hint": FAKE_KEYS_HINT if mods.fake_locate else None,
                "heard": getattr(mods.voice, "last_transcript", None),
                "models": getattr(sys.modules.get("perception"), "MODEL_NAMES", None),
                "run": {k: run[k] for k in ("status", "elapsed_s", "target")},
            }
            beat.now("view")
            key = show(frame, scene, info)
            last, last_shown = (frame, scene, info), now
            stages["view"] = (time.perf_counter() - t3) * 1000

            stats.add(stages, e2e_ms, now)
            attempt("log", log.row, frame, scene, t_recv, info["age_ms"], e2e_ms, stages, target, state)
            n += 1
            beat.frames, beat.fps, beat.last_frame, beat.guidance, beat.warm = n, stats.fps, now, state, True

            if not handle_key(key) or not handle_web_keys():
                break
            if prev_state not in (None, "idle") and state == "idle":
                target = request_target(target)
            prev_state = state

            if args.frames and n >= args.frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        beat.stop()
        log.close()
        if view:
            debug_view.close()
        if dash:
            dash.stop()
        report = stats.report(t_recv_src or "e2e")
        print(f"\n[main] {n} frames, log: {log.path}\n{report}")
        with open(log.path.replace(".csv", "_summary.txt"), "w") as f:
            f.write(f"{modes}\n{n} frames\n{report}\n")


if __name__ == "__main__":
    main()
