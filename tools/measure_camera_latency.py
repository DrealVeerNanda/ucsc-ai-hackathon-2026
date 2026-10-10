"""Measure the real delay from light reaching the phone's camera to the frame arriving here:
sensor, the app's encoding, the USB cable (or the network, with a relay) and decoding.

  python tools/measure_camera_latency.py            # Record3D over USB
  python tools/measure_camera_latency.py pathon
  python tools/measure_camera_latency.py relay      # includes the Mac and the network link

Run it on the machine with the screen (the GX10's own monitor, not over SSH). Point the phone
at the screen so the window fills most of its view. The window flips between black and white
about 20 times; each flip is timed until a camera frame shows it.

The answer includes the monitor's own delay (roughly 10-20 ms) and is only as fine as the
camera's frame interval, so read it as "about this much", not to the millisecond.
"""
import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import stream  # noqa: E402
import tracing  # noqa: E402


def brightness(rgb: np.ndarray) -> float:
    """Mean of the middle of the image: the part most likely to be looking at the screen."""
    h, w = rgb.shape[:2]
    return float(rgb[h // 4:3 * h // 4, w // 4:3 * w // 4].mean())


def latencies(flips, samples, contrast: float, share: float = 0.35):
    """flips: [(time, went_bright)], samples: [(arrival time, brightness)], both in time order.
    For each flip, the delay until the first frame that has moved `share` of the way to the new
    level, compared with the last frame before the flip. Flips nobody saw are left out."""
    out = []
    times = np.array([t for t, _ in samples])
    for k, (t_flip, bright) in enumerate(flips):
        t_end = flips[k + 1][0] if k + 1 < len(flips) else np.inf
        before = np.searchsorted(times, t_flip) - 1
        if before < 0:
            continue
        base = samples[before][1]
        sign = 1.0 if bright else -1.0
        for t, b in samples[before + 1:]:
            if t >= t_end:
                break
            if (b - base) * sign > share * contrast:
                out.append((t - t_flip) * 1000.0)
                break
    return out


def run(s, show, flips_wanted: int = 20, settle_s: float = 1.2):
    """show(level) paints the screen 0 (black) or 255 (white). Returns (delays ms, contrast, fps)."""
    def watch(seconds):
        got, end = [], time.monotonic() + seconds
        while time.monotonic() < end:
            fr = s.get_frame(wait=0.05)
            if fr is not None:
                got.append((fr.t_recv, brightness(fr.rgb)))
        return got

    show(0)
    dark = np.median([b for _, b in watch(settle_s)[-10:]] or [0])
    show(255)
    light = np.median([b for _, b in watch(settle_s)[-10:]] or [0])
    contrast = float(light - dark)
    if contrast < 25:
        return [], contrast, s.status()["fps"]

    flips, samples, level = [], [], 255
    for _ in range(flips_wanted):
        level = 255 - level
        show(level)
        flips.append((time.monotonic(), level == 255))
        samples += watch(random.uniform(0.5, 0.9))      # uneven gaps, so flips do not line up with frames
    return latencies(flips, samples, contrast), contrast, s.status()["fps"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="?", default=None, help="record3d | pathon[:IP] | relay[:PORT]")
    ap.add_argument("--flips", type=int, default=20)
    a = ap.parse_args()

    import cv2
    try:
        s = stream.start(a.source)
    except stream.SourceError as e:
        sys.exit(f"[latency] {e}")
    try:
        cv2.namedWindow("latency", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("latency", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except cv2.error as e:
        stream.stop()
        sys.exit(f"[latency] cannot open a window ({e}). Run this at the machine's own screen, not over SSH.")

    def show(level):
        cv2.imshow("latency", np.full((720, 1280, 3), level, np.uint8))
        cv2.waitKey(1)

    print("[latency] point the phone at this screen; flashing for about 20 s")
    try:
        delays, contrast, fps = run(s, show, a.flips)
    finally:
        cv2.destroyAllWindows()
        stream.stop()
    if len(delays) < 5:
        sys.exit(f"[latency] the camera did not see the flashes (brightness changed by only {contrast:.0f} of 255, "
                 f"{len(delays)} flips caught). Move the phone closer so the screen fills its view.")
    frame_ms = 1000.0 / fps if fps else float("nan")
    med = float(np.median(delays))
    print(f"camera to here: about {med:.0f} ms  (median of {len(delays)} flips; quickest {min(delays):.0f}, "
          f"slowest {max(delays):.0f})")
    print(f"frames arrive every {frame_ms:.0f} ms, so the true delay is roughly {max(0.0, med - frame_ms):.0f}-{med:.0f} ms, "
          "of which the monitor is about 10-20 ms")
    tracing.event("camera_latency", med, source=s.source, flips=len(delays), fastest=round(min(delays), 1),
                  slowest=round(max(delays), 1), frame_interval_ms=round(frame_ms, 1))
    if tracing.path():
        print(f"recorded in {tracing.path()}")
