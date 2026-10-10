"""Blindfold pickup main loop.

Every lane's module can be fake or real, so pieces swap in one at a time:

    python main.py                                   # all fake
    python main.py --guidance real                   # lane 3 tunes audio on fake scenes
    python main.py --stream real --perception real   # lanes 1 + 2 live, printed audio
    python main.py --all-real --fullscreen           # the demo

Keys: q / esc quit, t ask for a new target. Fake perception adds hand keys (see fakes.py).
Each run writes a per-frame CSV to logs/ and prints a latency summary on exit.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import os
import time
from collections import deque
from types import SimpleNamespace

import numpy as np

import debug_view
import fakes
from interfaces import hand_error

LANES = ("stream", "perception", "guidance", "voice")
STAGES = ("stream", "perception", "guidance", "view")
FAKE_KEYS_HINT = "arrows/w/s move hand   o occlude   h hide hand   n jitter   r reset"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for lane in LANES:
        ap.add_argument(f"--{lane}", choices=("fake", "real"), default="fake")
    ap.add_argument("--all-real", action="store_true", help="use every real module")
    ap.add_argument("--target", default="mug", help="target for fake voice")
    ap.add_argument("--ask", action="store_true", help="fake voice: type the target in the terminal")
    ap.add_argument("--image", help="fake stream: replay this image instead of the synthetic table")
    ap.add_argument("--fps", type=float, default=30.0, help="fake stream frame rate")
    ap.add_argument("--no-view", action="store_true", help="run without the debug window")
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--frames", type=int, default=0, help="stop after N frames (smoke tests)")
    ap.add_argument("--log-dir", default="logs")
    args = ap.parse_args()
    if args.all_real:
        for lane in LANES:
            setattr(args, lane, "real")
    return args


def load_modules(args) -> SimpleNamespace:
    m = SimpleNamespace(fake_locate=None, reset=None)
    if args.stream == "real":
        m.get_frame = importlib.import_module("stream").get_frame
    else:
        m.get_frame = fakes.FakeStream(args.image, args.fps).get_frame
    if args.perception == "real":
        m.locate = importlib.import_module("perception").locate
    else:
        m.fake_locate = fakes.FakeLocate()
        m.locate = m.fake_locate.locate
    if args.guidance == "real":
        g = importlib.import_module("guidance")
        m.update, m.reset = g.update, getattr(g, "reset", None)
    else:
        g = fakes.FakeGuidance()
        m.update, m.reset = g.update, g.reset
    if args.voice == "real":
        m.listen = importlib.import_module("voice").listen_for_target
    else:
        m.listen = fakes.FakeVoice(args.target, args.ask).listen_for_target
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


def main():
    args = parse_args()
    mods = load_modules(args)
    view = not args.no_view
    modes = "  ".join(f"{lane}:{getattr(args, lane)}" for lane in LANES)
    print(f"[main] {modes}")

    log = RunLog(args.log_dir)
    stats = Stats()
    if view:
        debug_view.open_window(args.fullscreen)

    def request_target(prev: str | None = None) -> str:
        if view:
            debug_view.show_message("LISTENING...")
        target = None
        while not target:
            target = mods.listen()
        print(f"[main] target: {target!r}" + (f" (was {prev!r})" if prev else ""))
        if mods.reset:
            mods.reset(target)
        if mods.fake_locate:
            mods.fake_locate.reset_hand()
        return target

    target = request_target()
    prev_state = None
    last = None              # (frame, scene, info) for redraws while no frames arrive
    last_shown = 0.0
    t_recv_src = None
    n = 0

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

    try:
        while True:
            t0 = time.perf_counter()
            frame = mods.get_frame()
            if frame is None:
                time.sleep(0.002)
                now = time.monotonic()
                if view and last is not None and now - last_shown > 0.1:
                    f, s, info = last
                    info["age_ms"] = (now - info["t_recv"]) * 1000
                    last_shown = now
                    if not handle_key(debug_view.show(f, s, info)):
                        break
                continue
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

            scene = mods.locate(frame, target)
            t2 = time.perf_counter()
            state = mods.update(scene)
            t3 = time.perf_counter()
            now = time.monotonic()
            e2e_ms = (now - t_recv) * 1000

            stages = {"stream": (t1 - t0) * 1000, "perception": (t2 - t1) * 1000,
                      "guidance": (t3 - t2) * 1000}
            info = {
                "target": target, "state": state, "fps": stats.fps,
                "age_ms": e2e_ms, "e2e_ms": e2e_ms, "e2e_label": t_recv_src,
                "stages": dict(stages, view=stats.series["view"][-1] if stats.series["view"] else 0.0),
                "modes": modes, "t_recv": t_recv,
                "keys_hint": FAKE_KEYS_HINT if mods.fake_locate else None,
            }
            key = None
            if view:
                key = debug_view.show(frame, scene, info)
                last, last_shown = (frame, scene, info), now
            stages["view"] = (time.perf_counter() - t3) * 1000

            stats.add(stages, e2e_ms, now)
            log.row(frame, scene, t_recv, info["age_ms"], e2e_ms, stages, target, state)

            if not handle_key(key):
                break
            if prev_state not in (None, "idle") and state == "idle":
                target = request_target(target)
            prev_state = state

            n += 1
            if args.frames and n >= args.frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        log.close()
        if view:
            debug_view.close()
        report = stats.report(t_recv_src or "e2e")
        print(f"\n[main] {n} frames, log: {log.path}\n{report}")
        with open(log.path.replace(".csv", "_summary.txt"), "w") as f:
            f.write(f"{modes}\n{n} frames\n{report}\n")


if __name__ == "__main__":
    main()
