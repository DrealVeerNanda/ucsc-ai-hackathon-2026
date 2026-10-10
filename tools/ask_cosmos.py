"""Ask the local Cosmos Reason server where an object is in one image, and time it.

  bash tools/run_cosmos.sh 2b &                         # start the server first
  python tools/ask_cosmos.py photo.jpg mug              # prints the box and the seconds taken
  python tools/ask_cosmos.py recordings/NAME mug        # first frame of a recording
  python tools/ask_cosmos.py photo.jpg mug --save out/cosmos.jpg --repeat 5

The request format (box first, 0-1000 coordinates, JSON schema) is the one that worked best in
the RTHACK tests. `ask()` can be imported by perception.py.
"""
import argparse
import base64
import json
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

URL = "http://127.0.0.1:8001/v1"
PROMPT = ('Is there a {target} in this image? Reply with JSON only: {{"visible": true or false, '
          '"box": [x1, y1, x2, y2] bounding box of the {target} with coordinates normalized to 0-1000, '
          'or null if it is not visible, "grasp": "<where to hold it, a few words>"}}')
SCHEMA = {
    "type": "object",
    "properties": {
        # Asked first and on its own: without it both models drew a box for objects that were not there.
        "visible": {"type": "boolean"},
        "box": {"anyOf": [{"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                          {"type": "null"}]},
        "grasp": {"type": "string", "maxLength": 60},
    },
    "required": ["visible", "box", "grasp"],
    "additionalProperties": False,
}


def ask(rgb: np.ndarray, target: str, url: str = URL, max_side: int = 640, timeout_s: float = 30.0) -> dict:
    """Returns {"box": (x1, y1, x2, y2) in pixels of rgb or None, "grasp": str, "seconds": float}."""
    h, w = rgb.shape[:2]
    s = min(1.0, max_side / max(h, w))
    small = rgb if s == 1.0 else cv2.resize(rgb, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    _, jpg = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    body = {
        "model": "cosmos", "temperature": 0.0, "max_tokens": 120,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()}},
            {"type": "text", "text": PROMPT.format(target=target)}]}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "find", "schema": SCHEMA, "strict": True}},
    }
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        reply = json.load(r)
    seconds = time.perf_counter() - t0
    d = json.loads(reply["choices"][0]["message"]["content"])
    box = d.get("box") if d.get("visible", True) else None
    if box is not None:
        x1, x2 = sorted((box[0] * w / 1000.0, box[2] * w / 1000.0))
        y1, y2 = sorted((box[1] * h / 1000.0, box[3] * h / 1000.0))
        box = (max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2))
        if box[2] - box[0] < 2 or box[3] - box[1] < 2:
            box = None
    return {"box": box, "grasp": d.get("grasp", ""), "seconds": seconds, "usage": reply.get("usage", {})}


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
    ap.add_argument("target", help='what to find, e.g. "mug"')
    ap.add_argument("--url", default=URL)
    ap.add_argument("--repeat", type=int, default=1, help="ask N times to see the steady-state time")
    ap.add_argument("--save", help="write the image with the box drawn to this file")
    a = ap.parse_args()
    rgb = load_rgb(a.image)
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
