"""Run the real detector and hand tracker on a stream and report what they find and how fast.
Works over SSH (no window): it prints numbers and saves a picture.

  python tools/check_perception.py recordings/NAME mug        # a saved recording
  python tools/check_perception.py pathon mug --seconds 10    # the phone, live
  PERCEPTION_HAND=sticker python tools/check_perception.py record3d bottle
Saves out/perception.jpg: the last frame with the box, the fingertip and the distance drawn on.

Lane 2 is done when the distance printed here matches a ruler within about 3 cm.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import perception  # noqa: E402
import stream  # noqa: E402
from interfaces import hand_error  # noqa: E402


def draw(frame, scene, target):
    img = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
    if scene.object_box is not None:
        x1, y1, x2, y2 = scene.object_box
        colour = (0, 255, 255) if scene.object_visible else (0, 140, 255)   # orange = remembered
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(img, target, (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    if scene.hand_px is not None:
        cv2.circle(img, scene.hand_px, 8, (255, 0, 255), -1)
        if scene.object_box is not None:
            x1, y1, x2, y2 = scene.object_box
            cv2.line(img, scene.hand_px, ((x1 + x2) // 2, (y1 + y2) // 2), (255, 255, 255), 1)
    err = hand_error(scene)
    text = "no distance" if err is None else f"{np.linalg.norm(err) * 100:.0f} cm"
    cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
    cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return img


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="record3d | pathon[:IP] | relay[:PORT] | recordings/NAME")
    ap.add_argument("target", help='what to find, e.g. "mug"')
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--save", default=str(ROOT / "out" / "perception.jpg"))
    a = ap.parse_args()

    try:
        s = stream.start(a.source)
    except stream.SourceError as e:
        sys.exit(f"[check] {e}")
    print("[check] loading models (the first run downloads weights)...")
    t = time.perf_counter()
    first = None
    while first is None:
        first = s.get_frame(wait=1.0)
    perception.locate(first, a.target)          # warm-up: model load and CUDA start-up are not per-frame cost
    print(f"[check] ready after {time.perf_counter() - t:.1f} s; measuring for {a.seconds:.0f} s")

    ms, dists, n_obj, n_hand, n_both, last = [], [], 0, 0, 0, None
    t0 = time.time()
    try:
        while time.time() - t0 < a.seconds:
            frame = s.get_frame(wait=0.5)
            if frame is None:
                continue
            t = time.perf_counter()
            scene = perception.locate(frame, a.target)
            ms.append((time.perf_counter() - t) * 1000)
            n_obj += scene.object_visible
            n_hand += scene.hand_xyz is not None
            err = hand_error(scene)
            if err is not None:
                n_both += 1
                dists.append(float(np.linalg.norm(err)) * 100)
            last = (frame, scene)
    finally:
        stream.stop()
    if not ms:
        sys.exit("[check] no frames arrived")

    n = len(ms)
    print(f"frames:    {n} in {a.seconds:.0f} s = {n / a.seconds:.1f} fps "
          f"(stream delivers {s.status()['fps']:.0f}; need 15)")
    print(f"locate():  median {np.median(ms):.0f} ms, 95% under {np.percentile(ms, 95):.0f} ms")
    print(f"object:    seen in {n_obj / n:.0%} of frames")
    print(f"hand:      positioned in {n_hand / n:.0%} of frames")
    if dists:
        print(f"distance:  median {np.median(dists):.1f} cm, spread {np.percentile(dists, 5):.1f}-"
              f"{np.percentile(dists, 95):.1f} cm, in {n_both / n:.0%} of frames")
    else:
        print("distance:  never had both the object and the hand in the same frame")
    frame, scene = last
    if scene.object_xyz is not None:
        print(f"object at: {np.round(scene.object_xyz, 3)} m (x right, y down, z forward)")
    Path(a.save).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(a.save, draw(frame, scene, a.target))
    print(f"saved {a.save}")
