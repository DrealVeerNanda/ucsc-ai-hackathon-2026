"""Ask the local Cosmos Reason server where an object is in one image, and time it.

  bash tools/run_cosmos.sh 2b &                         # start the server first
  python tools/ask_cosmos.py photo.jpg mug              # prints the box and the seconds taken
  python tools/ask_cosmos.py recordings/NAME mug        # first frame of a recording
  python tools/ask_cosmos.py photo.jpg mug --save out/cosmos.jpg --repeat 5
  python tools/ask_cosmos.py photo.jpg --list           # what could be picked up in this picture

The requests themselves live in cosmos.py.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cosmos  # noqa: E402

URL = cosmos.URL
ask = cosmos.find       # perception.py and voice.py use cosmos.py directly; this name is kept for scripts


def load_rgb(path: str) -> np.ndarray:
    p = Path(path)
    if p.is_dir():  # a recording: take its first frame, upright
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        import stream
        raw = next(stream._replay_frames(p, loop=False, paced=False))
        return stream.prepare(raw, 0.0).rgb
    bgr = cv2.imread(str(p))
    if bgr is None:
        sys.exit(f"[cosmos] cannot read image {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", help="image file or recordings/NAME")
    ap.add_argument("target", nargs="?", help='what to find, e.g. "mug"')
    ap.add_argument("--list", action="store_true", help="list what could be picked up instead of finding one thing")
    ap.add_argument("--url", default=URL)
    ap.add_argument("--repeat", type=int, default=1, help="ask N times to see the steady-state time")
    ap.add_argument("--save", help="write the image with the box drawn to this file")
    a = ap.parse_args()
    rgb = load_rgb(a.image)
    if a.list or not a.target:
        try:
            print(cosmos.list_objects(rgb, url=a.url))
        except OSError as e:
            sys.exit(f"[cosmos] no answer from {a.url} ({e}). Start it with: bash tools/run_cosmos.sh")
        sys.exit(0)
    out = None
    for i in range(a.repeat):
        try:
            out = ask(rgb, a.target, a.url)
        except OSError as e:
            sys.exit(f"[cosmos] no answer from {a.url} ({e}). Start it with: bash tools/run_cosmos.sh")
        box = None if out["box"] is None else [round(v) for v in out["box"]]
        print(f"{out['seconds']:.2f} s  box {box}  grasp: {out['grasp']}  tokens {out['usage'].get('completion_tokens')}")
    if a.save and out is not None:
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if out["box"] is not None:
            x1, y1, x2, y2 = map(int, out["box"])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 3)
        Path(a.save).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(a.save, img)
        print(f"saved {a.save}")
