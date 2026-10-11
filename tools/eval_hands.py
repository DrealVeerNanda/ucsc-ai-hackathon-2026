"""How well does each hand option follow the wearer's hand on a recording?

    python tools/eval_hands.py recordings/live_2148 --option 3
    python tools/eval_hands.py recordings/live_2148 --all --out out/hands      # every option, one table
    python tools/eval_hands.py recordings/live_2148 --option 5 --point palm --filter kalman

Runs the real perception (detector and hand tracker; Cosmos off) over the recording, frame
by frame at the recording's own clock, and scores the hand against tools/hand_labels_NAME.json:
frames somebody looked at and marked (is the wearer's hand in view, roughly where is its index
fingertip), the frame ranges of the reaches, a stretch with no hand in view, and a stretch
where the hand holds still. There is no other ground truth, so read the columns for what
they are:

    found      share of frames inside the reaches where the hand has a 3D position
    gaps       how many times it was lost inside the reaches, and the longest, in seconds
    tip        labelled frames where the reported point is within 120 px of the marked fingertip
    wrong      labelled frames where a hand is reported far from the marked one (a bystander's)
    partial    frames with the hand in view but its fingertip not: is any hand reported
    false      frames marked "no wearer's hand" where one is reported, and the same for the no-hand stretch
    jitter     median frame-to-frame move of the point while the hand holds still, px and mm
    dz         frames with lidar: 95th percentile and largest frame-to-frame change of the hand's depth, cm
    ms         the hand tracker alone, median and 95th percentile

--out also writes NAME_OPTION.jsonl (every frame) and NAME_OPTION.jpg: the same 12 labelled
frames for every option, with what it reported drawn on. Look at the pictures before
believing the numbers. The recordings run at 8 to 10 frames a second, a third of the live
rate: a tracker that follows the hand from frame to frame (MediaPipe) has a harder time here
than live; one that looks at each frame afresh (YOLO) does not care.
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
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import perception  # noqa: E402
import stream  # noqa: E402

HIT_PX = 120          # the labels are good to about 50 px, and which finger is the index is not always clear
SHEET = 12


class Timed:
    """Wraps a tracker and records how long each call takes."""

    def __init__(self, model):
        self.model, self.ms = model, []

    def __getattr__(self, attr):
        return getattr(self.model, attr)

    def detect(self, *a, **kw):
        t = time.perf_counter()
        out = self.model.detect(*a, **kw)
        self.ms.append((time.perf_counter() - t) * 1000)
        return out


def load(folder: Path, a: int, b: int):
    """(index, Frame) for frames a to b, with the recording's own times."""
    ts = np.load(folder / "frames.npz")["t"]
    for i, raw in enumerate(stream._replay_frames(folder, loop=False, paced=False)):
        if i >= b:
            break
        if i >= a:
            yield i, stream.prepare(raw, float(ts[i] - ts[0]))


def run(frames, number: int, target: str, detector: str, point: str | None, smoothing: str | None, sheet_frames=()):
    """Per-frame rows for one hand option, and the pictures of the frames wanted for the contact sheet."""
    key = perception.HAND_OPTIONS[number - 1][0]
    tracker = perception._hand_tracker(perception._HAND_SETUPS[key][0])
    settings = perception._hand_settings(key)
    if settings["hand_depth"] == "fused":
        settings.update({k: v for k, v in (("hand_point", point), ("hand_filter", smoothing)) if v})
    p = run.shared
    if p is None:
        det = None if detector == "none" else perception.OpenVocabDetector(detector)
        p = run.shared = perception.Perception(det, None, perception.PerceptionConfig())
    timed = Timed(tracker)
    p._set_hands(timed, settings)
    p._hand_depth = perception.trackers.HandDepth()      # every option starts without knowing the wearer's hand size
    p._hand_scale.clear()
    p.target = None
    rows, keep = [], {}
    for i, f in frames:
        if i in sheet_frames:
            keep[i] = f.rgb
        t = time.perf_counter()
        s = p.locate(f, target)
        ms = (time.perf_counter() - t) * 1000
        d = s.hand
        rows.append({"i": i, "t": round(f.t, 3), "ms": round(ms, 2), "hand_ms": round(timed.ms[-1], 2),
                     "seen": bool(s.hand_visible), "px": s.hand_px,
                     "xyz": None if s.hand_xyz is None else [round(float(v), 4) for v in s.hand_xyz],
                     "depth_from": None if d is None else d.depth_from, "tracker": None if d is None else d.tracker,
                     "conf": None if d is None else round(d.confidence, 2),
                     "box": None if d is None or d.box is None else [round(float(v)) for v in d.box],
                     "kp": None if d is None or d.keypoints_px is None else np.round(d.keypoints_px).astype(int).tolist(),
                     "obj": s.object_box, "obj_seen": bool(s.object_visible),
                     "obj_xyz": None if s.object_xyz is None else [round(float(v), 4) for v in s.object_xyz],
                     "has_depth": bool(np.any(f.depth > 0))})
    return key, rows, keep


run.shared = None


def score(rows, labels) -> dict:
    by = {r["i"]: r for r in rows}
    out = {}
    inside = [by[i] for a, b in labels["reaches"] for i in range(a, b) if i in by]
    out["found"] = float(np.mean([r["xyz"] is not None for r in inside])) if inside else float("nan")
    gaps, longest, start = 0, 0.0, None
    for a, b in labels["reaches"]:
        seq = [by[i] for i in range(a, b) if i in by]
        had = False
        for r in seq + [None]:
            lost = r is not None and r["xyz"] is None
            if lost and had and start is None:
                start = r["t"]
                gaps += 1
            if start is not None and not lost:
                end = r["t"] if r is not None else seq[-1]["t"]
                longest, start = max(longest, end - start), None
            had = had or (r is not None and r["xyz"] is not None)
    out["gaps"], out["longest_s"] = gaps, longest
    hit = wrong = n = part = part_n = neg = neg_n = 0
    dist = []
    for k, v in labels["frames"].items():
        r = by.get(int(k))
        if r is None:
            continue
        if not v["hand"]:
            neg_n += 1
            neg += r["seen"]
        elif v.get("tip") is None:
            part_n += 1
            part += r["seen"]
        else:
            n += 1
            if r["px"] is not None:
                d = float(np.hypot(r["px"][0] - v["tip"][0], r["px"][1] - v["tip"][1]))
                # Any point of the reported hand near the marked fingertip: the right hand, another point of it.
                near = d if r["kp"] is None else float(np.min(np.linalg.norm(np.array(r["kp"]) - v["tip"], axis=1)))
                hit += d <= HIT_PX
                wrong += min(d, near) > 2 * HIT_PX
                dist.append(d)
    out.update(tip=f"{hit}/{n}", wrong=wrong, partial=f"{part}/{part_n}", false_labelled=f"{neg}/{neg_n}",
               tip_px=float(np.median(dist)) if dist else float("nan"))
    a, b = labels["no_hand"]
    quiet = [by[i]["seen"] for i in range(a, b) if i in by]
    out["false_quiet"] = f"{sum(quiet)}/{len(quiet)}"
    a, b = labels["still"]
    px = [np.hypot(*np.subtract(by[i]["px"], by[i - 1]["px"])) for i in range(a + 1, b)
          if i in by and i - 1 in by and by[i]["px"] and by[i - 1]["px"]]
    mm = [1000 * np.linalg.norm(np.subtract(by[i]["xyz"], by[i - 1]["xyz"])) for i in range(a + 1, b)
          if i in by and i - 1 in by and by[i]["xyz"] and by[i - 1]["xyz"]]
    out["jitter_px"] = float(np.median(px)) if px else float("nan")
    out["jitter_mm"] = float(np.median(mm)) if mm else float("nan")
    dz = [100 * abs(r["xyz"][2] - q["xyz"][2]) for q, r in zip(inside, inside[1:])
          if r["i"] == q["i"] + 1 and r["has_depth"] and q["has_depth"] and r["xyz"] and q["xyz"]]
    out["dz95_cm"] = float(np.percentile(dz, 95)) if dz else float("nan")
    out["dzmax_cm"] = float(np.max(dz)) if dz else float("nan")
    ms = [r["hand_ms"] for r in rows[5:]]
    out["ms50"], out["ms95"] = float(np.percentile(ms, 50)), float(np.percentile(ms, 95))
    return out


def sheet(pics: dict, rows, labels, title: str, path: Path):
    """The labelled frames picked for the contact sheet, with what the option reported drawn on."""
    by, tiles = {r["i"]: r for r in rows}, []
    for i in labels["sheet"]:
        if i not in pics:
            continue
        img, r = cv2.cvtColor(pics[i], cv2.COLOR_RGB2BGR), by[i]
        lab = labels["frames"].get(str(i), {})
        if lab.get("tip"):
            cv2.drawMarker(img, tuple(lab["tip"]), (255, 255, 255), cv2.MARKER_CROSS, 44, 3)      # the marked fingertip
        if r["obj"] is not None:
            cv2.rectangle(img, tuple(r["obj"][:2]), tuple(r["obj"][2:]), (0, 255, 255) if r["obj_seen"] else (0, 140, 255), 2)
        if r["box"] is not None:
            cv2.rectangle(img, tuple(r["box"][:2]), tuple(r["box"][2:]), (0, 255, 0), 2)
        for q in r["kp"] or []:
            cv2.circle(img, tuple(q), 4, (0, 255, 0), -1)
        if r["px"] is not None:
            cv2.circle(img, tuple(r["px"]), 14, (255, 0, 255), -1 if r["xyz"] else 3)
        z = "" if r["xyz"] is None else f" {r['xyz'][2] * 100:.0f}cm"
        text = f"{i} " + ("no hand" if not r["seen"] else f"{(r['tracker'] or '')[:12]} {r['depth_from']}{z}")
        cv2.putText(img, text, (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 7)
        cv2.putText(img, text, (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
        tiles.append(cv2.resize(img, (360, 480)))
    if not tiles:
        return
    blank = np.zeros_like(tiles[0])
    grid = [np.hstack((tiles[k:k + 4] + [blank] * 4)[:4]) for k in range(0, len(tiles), 4)]
    out = np.vstack(grid)
    bar = np.zeros((44, out.shape[1], 3), np.uint8)
    cv2.putText(bar, f"{title}   cross = marked fingertip, dot = reported point (ring: no depth), green = its hand",
                (8, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite(str(path), np.vstack([bar, out]), [cv2.IMWRITE_JPEG_QUALITY, 80])


COLUMNS = ("found", "gaps", "tip", "wrong", "partial", "false", "jitter", "dz", "ms")


def line(name: str, s: dict) -> str:
    return (f"{name:<28}{s['found']:>6.0%} {s['gaps']:>3} / {s['longest_s']:>4.1f}s  {s['tip']:>6} {s['wrong']:>5} "
            f"{s['partial']:>7}  {s['false_labelled']:>4} {s['false_quiet']:>8}  {s['jitter_px']:>5.1f}px "
            f"{s['jitter_mm']:>5.1f}mm  {s['dz95_cm']:>4.1f} / {s['dzmax_cm']:>4.1f}cm  {s['ms50']:>5.1f} / {s['ms95']:>5.1f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording", help="a folder written by tools/record.py or STREAM_RECORD")
    ap.add_argument("--option", action="append", help="hand option, number or key; may be given several times")
    ap.add_argument("--all", action="store_true", help="every hand option")
    ap.add_argument("--labels", help="default: tools/hand_labels_NAME.json for the recording's name")
    ap.add_argument("--target", default="water bottle")
    ap.add_argument("--detector", default="yoloworld", help="yoloworld, or none to leave the object out")
    ap.add_argument("--frames", help="A:B, default: from just before the first reach to just after the last")
    ap.add_argument("--point", choices=perception.trackers.POINTS, help="override the option's hand point")
    ap.add_argument("--filter", choices=("average", "one_euro", "kalman"), help="override the option's smoothing")
    ap.add_argument("--out", help="folder for the per-frame files and contact sheets")
    args = ap.parse_args()

    folder = Path(args.recording)
    name = folder.name
    labels = json.loads(Path(args.labels or ROOT / "tools" / f"hand_labels_{name}.json").read_text())
    if args.frames:
        a, b = (int(v) for v in args.frames.split(":"))
    else:
        a = min(labels["no_hand"][0], labels["reaches"][0][0] - 30)
        b = labels["reaches"][-1][1] + 30
    numbers = list(range(1, len(perception.HAND_OPTIONS) + 1)) if args.all else [
        perception._hand_index(o) + 1 for o in (args.option or ["1"])]
    print(f"{folder}: frames {a} to {b}")
    print(f"{'option':<28}{'found':>6} {'gaps / longest':>12}  {'tip':>6} {'wrong':>5} {'partial':>7}  "
          f"{'false':>4} {'(quiet)':>8}  {'jitter':>16}  {'dz 95% / max':>14}  {'ms 50% / 95%':>13}")
    for n in numbers:
        key, rows, pics = run(load(folder, a, b), n, args.target, args.detector, args.point, args.filter,
                              set(labels["sheet"]) if args.out else ())
        s = score(rows, labels)
        tag = key + "".join(f"_{v}" for v in (args.point, args.filter) if v)
        print(line(f"{n} {tag}", s), flush=True)
        if args.out:
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{name}_{tag}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            (out / f"{name}_{tag}.json").write_text(json.dumps(s))
            sheet(pics, rows, labels, f"hand option {n}: {tag}", out / f"{name}_{tag}.jpg")


if __name__ == "__main__":
    main()
