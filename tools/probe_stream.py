"""First contact with the phone: what arrives, how fast, and is it upright?

  python tools/probe_stream.py                   # Record3D over USB: 3 s, then a report
  python tools/probe_stream.py pathon            # the free PathOn app instead
  python tools/probe_stream.py recordings/NAME   # a saved recording
  python tools/probe_stream.py --show            # live rgb | depth window: r rotates, s saves, q quits
Saves out/probe.jpg (left: rgb, right: depth 0-3 m).
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import stream  # noqa: E402


def side_by_side(fr, lines=()):
    vis = cv2.applyColorMap((np.clip(fr.depth / 3.0, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    vis[fr.depth <= 0] = 0
    img = np.hstack([cv2.cvtColor(fr.rgb, cv2.COLOR_RGB2BGR), vis])
    for i, text in enumerate(lines):
        cv2.putText(img, text, (10, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(img, text, (10, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return img


def centre_depth(fr):
    h, w = fr.depth.shape
    c = fr.depth[h // 2 - 4:h // 2 + 5, w // 2 - 4:w // 2 + 5]
    c = c[c > 0]
    return float(np.median(c)) if c.size else float("nan")


def save(fr):
    out = ROOT / "out" / "probe.jpg"
    out.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out), side_by_side(fr))
    print(f"saved {out.relative_to(ROOT)} (left: rgb, right: depth 0-3 m, black = no reading)")


def show(s):
    rotations = [0, 90, 180, 270]
    last = None
    print("r: rotate   s: save out/probe.jpg   q: quit")
    while True:
        fr = s.get_frame(wait=0.05)
        st = s.status()
        if fr is not None:
            last = fr
            rot = stream.ROTATE.get(s.source.partition(":")[0], 0) if s.rotate is None else s.rotate
            pose = "no pose" if fr.pose is None else "pose xyz " + " ".join(f"{v:+.2f}" for v in fr.pose[:3, 3])
            cv2.imshow("stream", side_by_side(fr, [
                f"{st['state']} {st['fps']:.0f} fps  rotate {rot}  {fr.rgb.shape[1]}x{fr.rgb.shape[0]}",
                f"centre depth {centre_depth(fr):.2f} m  {pose}"]))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            cur = rot if last is not None else 0
            s.rotate = rotations[(rotations.index(cur) + 1) % 4]
            print(f"rotate = {s.rotate}")
        if key == ord("s") and last is not None:
            save(last)
    cv2.destroyAllWindows()


def report(s, seconds):
    t0, n, last, poses = time.time(), 0, None, 0
    while time.time() - t0 < seconds:
        fr = s.get_frame(wait=0.2)
        if fr is None:
            continue
        n, last = n + 1, fr
        poses += fr.pose is not None
    st = s.status()
    if last is None:
        sys.exit(f"[probe] connected but no frames in {seconds:.0f} s ({st['error'] or st['state']})")
    d = last.depth
    valid = d[d > 0]
    h, w = d.shape
    print(f"source:    {st['source']}  ({st['state']})")
    print(f"rate:      {st['fps']:.1f} fps arriving, {n / seconds:.1f} fps read by this loop"
          f"  -> {'OK' if st['fps'] >= 15 else 'TOO SLOW'} (need 15)")
    print(f"from app:  rgb {st['raw_rgb'][1]}x{st['raw_rgb'][0]}  depth {st['raw_depth'][1]}x{st['raw_depth'][0]}")
    print(f"delivered: rgb {w}x{h} {last.rgb.dtype}  depth {w}x{h} {d.dtype}  "
          f"({'portrait' if h > w else 'landscape'}: check it is upright with --show)")
    if valid.size:
        print(f"depth:     {valid.size / d.size:.0%} valid, {valid.min():.2f}-{valid.max():.2f} m, "
              f"centre {centre_depth(last):.3f} m")
    else:
        print("depth:     NO valid pixels: is the rear lidar camera selected?")
    print(f"K:         fx {last.K[0, 0]:.1f}  fy {last.K[1, 1]:.1f}  cx {last.K[0, 2]:.1f}  cy {last.K[1, 2]:.1f}")
    if abs(last.K[0, 2] - w / 2) > 0.2 * w or abs(last.K[1, 2] - h / 2) > 0.2 * h:
        print("           WARNING: principal point is far from the image centre; K may not match this size")
    if poses:
        print(f"pose:      yes, on {poses}/{n} frames; position {np.round(last.pose[:3, 3], 3)} "
              "(run tools/check_pose.py to see if it can be trusted)")
    else:
        print("pose:      NOT available")
    save(last)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", nargs="?", default=None, help="record3d | pathon[:IP] | relay[:PORT] | recordings/NAME")
    p.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=None, help="clockwise; default: per app")
    p.add_argument("--seconds", type=float, default=3.0)
    p.add_argument("--show", action="store_true", help="live window instead of a report")
    a = p.parse_args()
    kw = {} if a.rotate is None else {"rotate": a.rotate}
    try:
        s = stream.start(a.source, **kw)
    except stream.SourceError as e:
        sys.exit(f"[probe] {e}")
    try:
        show(s) if a.show else report(s, a.seconds)
    finally:
        stream.stop()
