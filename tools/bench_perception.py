"""How long does perception.locate() take, and does a change alter what it finds?

    python tools/bench_perception.py IMAGES... --target "water bottle"
    python tools/bench_perception.py frames/ hands.jpg --frames 300 --save out/bench_before.json
    python tools/bench_perception.py frames/ hands.jpg --frames 300 --compare out/bench_before.json
    HAND_OPTION=3 python tools/bench_perception.py frames/ hands.jpg      # another hand tracker

Runs the real models (the detector and hand tracker perception.build() picks, Cosmos off) on
the given images, cycled as a video, with a flat depth map at 50 cm. Depth reading is a
fraction of a millisecond, so the models are what is timed. The first frames are warm-up.

Prints p50 / p95 of the whole locate() call and of each model. --save writes the box and
fingertip found on every frame; --compare reads such a file and reports how far today's
boxes and fingertips are from it, in pixels, so a speed-up can be checked for side effects.
Other programs on the GX10 (a running demo, Cosmos) share the GPU and CPU: compare runs made
back to back.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TRACE", "0")
os.environ.setdefault("PERCEPTION_HELPER", "off")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import perception  # noqa: E402
from interfaces import Frame  # noqa: E402

WARMUP = 20


def load_images(paths) -> list[np.ndarray]:
    files = []
    for p in map(Path, paths):
        files += sorted(f for f in p.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")) if p.is_dir() else [p]
    imgs = [cv2.imread(str(f)) for f in files]
    return [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in imgs if i is not None]


class Timed:
    """Wraps a model and records how long each call takes."""

    def __init__(self, model, name):
        self.model, self.name, self.ms = model, name, []

    def __getattr__(self, attr):
        return getattr(self.model, attr)

    def detect(self, *a, **kw):
        t = time.perf_counter()
        out = self.model.detect(*a, **kw)
        self.ms.append((time.perf_counter() - t) * 1000)
        return out


def pct(v, q):
    return float(np.percentile(v, q)) if len(v) else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", help="image files or folders")
    ap.add_argument("--target", default="water bottle")
    ap.add_argument("--frames", type=int, default=300, help="frames to time, after the warm-up")
    ap.add_argument("--fps", type=float, default=30.0, help="frame clock for the hand tracker's video mode")
    ap.add_argument("--save", help="write per-frame boxes and fingertips here (JSON)")
    ap.add_argument("--compare", help="a file written by --save: report how far today's results are from it")
    args = ap.parse_args()

    imgs = load_images(args.images)
    if not imgs:
        sys.exit("no images")
    p = perception.build()
    det = Timed(p.detector, "detector") if p.detector is not None else None
    hands = Timed(p.hands, "hand") if p.hands is not None else None
    p.detector, p.hands = det, hands

    total, rows = [], []
    for i in range(WARMUP + args.frames):
        rgb = imgs[i % len(imgs)]
        h, w = rgb.shape[:2]
        f = 0.9 * max(h, w)
        K = np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1]])
        frame = Frame(rgb, np.full((h, w), 0.5, np.float32), K, None, i / args.fps, t_recv=i / args.fps)
        t = time.perf_counter()
        scene = p.locate(frame, args.target)
        ms = (time.perf_counter() - t) * 1000
        if i == WARMUP - 1:
            for m in (det, hands):
                if m is not None:
                    m.ms.clear()
        if i >= WARMUP:
            total.append(ms)
            rows.append({"img": i % len(imgs), "box": scene.object_box, "tip": scene.hand_px})

    found_box = sum(r["box"] is not None for r in rows)
    found_tip = sum(r["tip"] is not None for r in rows)
    print(f"{len(imgs)} images, {args.frames} frames, target {args.target!r}")
    print(f"{'step':10s} {'p50 ms':>8s} {'p95 ms':>8s} {'mean ms':>8s}")
    for name, v in [("locate", total)] + [(m.name, m.ms) for m in (det, hands) if m is not None]:
        print(f"{name:10s} {pct(v, 50):8.2f} {pct(v, 95):8.2f} {np.mean(v):8.2f}")
    print(f"object box on {found_box}/{len(rows)} frames, fingertip on {found_tip}/{len(rows)} frames")

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(json.dumps({"target": args.target, "rows": rows,
                                               "locate_p50": pct(total, 50), "locate_p95": pct(total, 95)}))
    if args.compare:
        ref = json.loads(Path(args.compare).read_text())["rows"]
        box_d, tip_d, box_flip, tip_flip = [], [], 0, 0
        for a, b in zip(rows, ref):
            if (a["box"] is None) != (b["box"] is None):
                box_flip += 1
            elif a["box"] is not None:
                box_d.append(float(np.max(np.abs(np.subtract(a["box"], b["box"])))))
            if (a["tip"] is None) != (b["tip"] is None):
                tip_flip += 1
            elif a["tip"] is not None:
                tip_d.append(float(np.hypot(*np.subtract(a["tip"], b["tip"]))))
        print(f"vs {args.compare}: box found/lost differs on {box_flip} frames, box edges move "
              f"p50 {pct(box_d, 50):.1f} / max {max(box_d, default=0):.1f} px; fingertip found/lost differs on "
              f"{tip_flip} frames, moves p50 {pct(tip_d, 50):.1f} / max {max(tip_d, default=0):.1f} px")


if __name__ == "__main__":
    main()
