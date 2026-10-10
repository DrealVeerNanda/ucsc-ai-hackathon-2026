"""Lane 2: where are the object and the fingertip, in metres?

    scene = perception.locate(frame, target)

Every frame: a detector finds the object's box, a hand tracker finds the index fingertip, and
the lidar depth turns both pixels into 3D points in the camera frame (x right, y down,
z forward). No model ever sees the depth; it is a lookup table for the code below.

When the arm covers the object, the last position is remembered: for 2 s in the camera frame
(stand still), or in world coordinates if the pose can be trusted, so it stays right when the
wearer moves.

Settings by environment variable:
    PERCEPTION_DETECTOR=yoloworld|yoloe|coco|none   open-vocabulary (default) or fixed COCO classes
    PERCEPTION_HAND=mediapipe|sticker|none          hand model (default) or a coloured sticker
    PERCEPTION_STICKER_HSV=35,80,80,85,255,255      sticker colour range (default: green)
    PERCEPTION_USE_POSE=1       remember the object in world coordinates. Only switch this on
                                after tools/check_pose.py says the pose is usable.
    PERCEPTION_DETECT_EVERY=1   run the detector on every Nth frame
    PERCEPTION_DEVICE=cuda|cpu  torch device for the detector (default: ultralytics picks)
    MODELS_DIR=PATH             where weights are kept (default: models/ in this repo)

The detector and tracker wrappers are adapted from the RTHACK handguide target and hand modules.
"""
from __future__ import annotations

import os
import sys
import time
import urllib.request
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import tracing
from interfaces import Frame, Scene

MODELS_DIR = Path(os.environ.get("MODELS_DIR") or Path(__file__).resolve().parent / "models")
HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/"
                  "float16/latest/hand_landmarker.task")
DEPTH_MIN, DEPTH_MAX = 0.15, 4.0     # lidar readings outside this are not trusted


# ---------------------------------------------------------------------------
# Pixels to metres

def to_xyz(u: float, v: float, depth: np.ndarray, K: np.ndarray, r: int = 4, nearest: bool = False):
    """3D point for pixel (u, v), from the depth in a small patch around it, ignoring holes.
    nearest=True takes the near side of the patch (25th percentile) instead of the median:
    a fingertip is narrower than the patch, and what is behind it must not win."""
    h, w = depth.shape[:2]
    u, v = int(round(u)), int(round(v))
    if not (0 <= u < w and 0 <= v < h):
        return None
    patch = depth[max(0, v - r):v + r + 1, max(0, u - r):u + r + 1]
    vals = patch[(patch > DEPTH_MIN) & (patch < DEPTH_MAX)]
    # A patch that is mostly holes (a hand too close for lidar) must not be read from the few
    # background pixels around its edge.
    if vals.size < max(5, patch.size // 4):
        return None
    d = float(np.percentile(vals, 25) if nearest else np.median(vals))
    return back_project(u, v, d, K)


def back_project(u: float, v: float, d: float, K: np.ndarray) -> np.ndarray:
    return np.array([(u - K[0, 2]) * d / K[0, 0], (v - K[1, 2]) * d / K[1, 1], d])


def box_xyz(box, depth: np.ndarray, K: np.ndarray, exclude: np.ndarray | None = None):
    """3D point of an object: its box centre, at the median depth of the box's centre third.
    With `exclude` (a mask of pixels that are hand, not object) the whole box is used minus
    those pixels, leaning to the near side so the background behind the object does not win."""
    x1, y1, x2, y2 = box
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    part = 2.0 if exclude is not None else 6.0
    hw, hh = max(2.0, (x2 - x1) / part), max(2.0, (y2 - y1) / part)
    h, w = depth.shape[:2]
    xa, xb = int(max(0, cx - hw)), int(min(w, cx + hw + 1))
    ya, yb = int(max(0, cy - hh)), int(min(h, cy + hh + 1))
    if xb <= xa or yb <= ya:
        return None
    vals = depth[ya:yb, xa:xb]
    ok = (vals > DEPTH_MIN) & (vals < DEPTH_MAX)
    if exclude is not None:
        ok &= ~exclude[ya:yb, xa:xb].astype(bool)
    vals = vals[ok]
    if vals.size < (30 if exclude is not None else 5):
        return None
    d = np.percentile(vals, 40) if exclude is not None else np.median(vals)
    return back_project(cx, cy, float(d), K)


def hand_mask(shape, points: np.ndarray, grow: float = 0.25, min_px: int = 41) -> np.ndarray:
    """Pixels covered by the hand: the hull of its known points, grown to take in finger width."""
    mask = np.zeros(shape[:2], np.uint8)
    pts = np.round(points).astype(np.int32)
    cv2.fillConvexPoly(mask, cv2.convexHull(pts), 1)
    size = float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])))
    k = max(min_px, int(grow * size)) | 1
    return cv2.dilate(mask, np.ones((k, k), np.uint8))


def iou(a, b) -> float:
    if a is None or b is None:
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Object detectors. Each has set_target(name) and detect(rgb) -> [Detection], best first.

@dataclass
class Detection:
    box: tuple  # x1, y1, x2, y2 px
    score: float
    label: str


@contextmanager
def _in_models_dir():
    """ultralytics downloads weights and text encoders into the CWD; keep them in MODELS_DIR."""
    old = os.getcwd()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(MODELS_DIR)
    try:
        yield
    finally:
        os.chdir(old)


def _detections(res, names) -> list[Detection]:
    if res.boxes is None:
        return []
    out = [Detection(tuple(float(v) for v in b), float(s), str(names[int(c)]))
           for b, s, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(), res.boxes.cls.cpu().numpy())]
    out.sort(key=lambda d: -d.score)
    return out


class OpenVocabDetector:
    """YOLO-World or YOLOE through ultralytics: finds whatever the text prompt names."""

    WEIGHTS = {"yoloworld": "yolov8s-worldv2.pt", "yoloworld-m": "yolov8m-worldv2.pt",
               "yoloe": "yoloe-11s-seg.pt", "yoloe-m": "yoloe-11m-seg.pt"}
    # Extra prompt words that helped in the RTHACK photo benchmark.
    PROMPTS = {"cup": ["cup", "mug"], "mug": ["mug", "cup"], "ball": ["ball", "sports ball"]}

    def __init__(self, kind: str = "yoloworld", device: str | None = None, conf: float = 0.1, imgsz: int = 640):
        from ultralytics import YOLOE, YOLOWorld

        weights = self.WEIGHTS.get(kind, kind)
        self.is_yoloe = "yoloe" in weights
        with _in_models_dir():
            self.model = (YOLOE if self.is_yoloe else YOLOWorld)(weights)
        self.device, self.conf, self.imgsz = device, conf, imgsz
        self.names: list[str] = []

    def set_target(self, target: str):
        key = target.lower().strip()
        names = list(dict.fromkeys(self.PROMPTS.get(key, [key])))
        if names == self.names:
            return
        with _in_models_dir():
            if self.is_yoloe:
                self.model.set_classes(names, self.model.get_text_pe(names))
            else:
                self.model.set_classes(names)
        self.names = names

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])  # ultralytics expects BGR numpy arrays
        res = self.model.predict(bgr, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        return _detections(res, self.names)


class CocoDetector:
    """Fallback: a standard YOLO with the 80 COCO classes (cup, banana, bottle, sports ball...)."""

    ALIASES = {"mug": "cup", "glass": "cup", "ball": "sports ball", "tennis ball": "sports ball",
               "phone": "cell phone", "water bottle": "bottle", "remote control": "remote"}

    def __init__(self, weights: str = "yolo11s.pt", device: str | None = None, conf: float = 0.2, imgsz: int = 640):
        from ultralytics import YOLO

        with _in_models_dir():
            self.model = YOLO(weights)
        self.device, self.conf, self.imgsz = device, conf, imgsz
        self.classes: list[int] = []

    def set_target(self, target: str):
        key = target.lower().strip()
        key = self.ALIASES.get(key, key)
        self.classes = [i for i, n in self.model.names.items() if n == key]
        if not self.classes:
            print(f"[perception] {target!r} is not a COCO class; try one of: cup, bottle, banana, apple, "
                  "sports ball, cell phone, book, scissors, remote", file=sys.stderr)

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        if not self.classes:
            return []
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        res = self.model.predict(bgr, conf=self.conf, imgsz=self.imgsz, device=self.device,
                                 classes=self.classes, verbose=False)[0]
        return _detections(res, self.model.names)


# ---------------------------------------------------------------------------
# Hand trackers. Each has detect(rgb, t_ms) -> [Hand].

@dataclass
class Hand:
    tip: tuple                 # (u, v) of the point to guide: the index fingertip
    points: np.ndarray         # N x 2 px, everything known to be hand (for "is it covering the object")
    samples: list              # (u, v) pixels to read depth at, best first
    score: float = 1.0
    px_per_m: float | None = None   # how big the hand looks: pixels per metre of real hand, if known


class MediaPipeHand:
    """MediaPipe hand landmarker. The guided point is the index fingertip (landmark 8); depth is
    read there first, then further down the finger if the tip has no valid lidar reading."""

    def __init__(self, model_path: str | Path | None = None, num_hands: int = 2, min_conf: float = 0.5):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        path = Path(model_path) if model_path else MODELS_DIR / "hand_landmarker.task"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[perception] downloading the hand model to {path}", file=sys.stderr)
            urllib.request.urlretrieve(HAND_MODEL_URL, path)
        self._mp = mp
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path), delegate=BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO, num_hands=num_hands,
            min_hand_detection_confidence=min_conf, min_hand_presence_confidence=min_conf,
            min_tracking_confidence=min_conf)
        self.landmarker = vision.HandLandmarker.create_from_options(opts)
        self._last_ts = -1

    def detect(self, rgb: np.ndarray, t_ms: int) -> list[Hand]:
        h, w = rgb.shape[:2]
        ts = max(int(t_ms), self._last_ts + 1)   # video mode needs strictly increasing timestamps
        self._last_ts = ts
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self.landmarker.detect_for_video(img, ts)
        hands = []
        for lms, hd in zip(res.hand_landmarks, res.handedness):
            pts = np.array([[p.x * w, p.y * h] for p in lms], dtype=np.float32)
            # Wrist to middle knuckle is about 9.5 cm and the knuckle row about 8 cm. One of the two
            # faces the camera whichever way the hand points, so the larger gives the scale.
            scale = max(np.linalg.norm(pts[0] - pts[9]) / 0.095, np.linalg.norm(pts[5] - pts[17]) / 0.08)
            hands.append(Hand(tip=tuple(pts[8]), points=pts, samples=[tuple(pts[i]) for i in (8, 7, 6, 5)],
                              score=float(hd[0].score) if hd else 0.0, px_per_m=float(scale)))
        return hands


class StickerHand:
    """Fallback: a bright sticker or glove on the reaching hand, found by colour. Installs
    nowhere and runs at any frame rate. Default range is green; set PERCEPTION_STICKER_HSV."""

    def __init__(self, lo=(35, 80, 80), hi=(85, 255, 255), min_area: int = 120):
        self.lo, self.hi, self.min_area = np.array(lo, np.uint8), np.array(hi, np.uint8), min_area

    def detect(self, rgb: np.ndarray, t_ms: int = 0) -> list[Hand]:
        mask = cv2.inRange(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV), self.lo, self.hi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
        if n < 2:
            return []
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[i, cv2.CC_STAT_AREA] < self.min_area:
            return []
        c = (float(cents[i][0]), float(cents[i][1]))
        return [Hand(tip=c, points=np.array([c], np.float32), samples=[c])]


# ---------------------------------------------------------------------------
@dataclass
class PerceptionConfig:
    detect_every: int = 1        # run the detector on every Nth frame, reuse its box in between
    hold_s: float = 2.0          # no pose: keep the last object position this long while it is hidden
    hold_covered_s: float = 4.0  # ...a little longer when the hand is what is hiding it
    hold_pose_s: float = 15.0    # with pose: the position is anchored to the room, so keep it longer
    use_pose: bool = False       # anchor the object in world coordinates (needs a checked pose)
    smooth: int = 3              # moving-average length for both positions
    jump_m: float = 0.15         # a move larger than this restarts the average instead of blending
    box_margin: float = 0.10     # hand inside the box grown by this fraction: the box shows the hand
    hand_patch: int = 6          # half-size in px of the depth patch at the fingertip
    max_hand_m: float = 0.90     # a hand farther from the camera than this is somebody else's
    size_depth: bool = True      # no lidar on the hand (closer than about 20 cm): estimate its depth from its size


class _Average:
    """Light moving average: jitter in the positions becomes jitter in the audio."""

    def __init__(self, n: int, jump_m: float):
        self.buf = deque(maxlen=max(1, n))
        self.jump_m = jump_m

    def add(self, p: np.ndarray) -> np.ndarray:
        if self.buf and np.linalg.norm(p - self.buf[-1]) > self.jump_m:
            self.buf.clear()
        self.buf.append(np.asarray(p, float))
        return np.mean(self.buf, axis=0)

    def clear(self):
        self.buf.clear()


class Perception:
    def __init__(self, detector=None, hands=None, cfg: PerceptionConfig | None = None):
        self.detector, self.hands = detector, hands
        self.cfg = cfg or PerceptionConfig()
        self.target = None
        self._obj_avg = _Average(self.cfg.smooth, self.cfg.jump_m)
        self._hand_avg = _Average(self.cfg.smooth, self.cfg.jump_m)
        self._hand_scale = deque(maxlen=90)   # lidar depth / size-based depth, to fit the size estimate to this hand
        self._forget()

    def _forget(self):
        self._box = None
        self._seen = False           # did the detector find the object the last time it ran?
        self._anchor = None          # remembered object position, world frame if anchored else camera
        self._anchor_t = 0.0
        self._anchored = False       # is _anchor in world coordinates?
        self._i = 0
        self._obj_avg.clear()
        self._hand_avg.clear()

    def _pick(self, dets: list[Detection]):
        if not dets:
            return None
        if self._box is not None:    # prefer the one we were already following
            return max(dets, key=lambda d: d.score + 0.5 * iou(d.box, self._box))
        return dets[0]

    def _covered(self, box, hand: Hand | None) -> bool:
        """Is any part of the hand over the object's box? Then the box's depth is the hand's."""
        if box is None or hand is None:
            return False
        mx, my = self.cfg.box_margin * (box[2] - box[0]), self.cfg.box_margin * (box[3] - box[1])
        p = hand.points
        return bool(np.any((p[:, 0] >= box[0] - mx) & (p[:, 0] <= box[2] + mx) &
                           (p[:, 1] >= box[1] - my) & (p[:, 1] <= box[3] + my)))

    def _hand_xyz(self, hand: Hand, frame: Frame):
        """(fingertip position or None, where its depth came from, in reach?).

        Depth is lidar at the fingertip, or further down the finger if the tip has no reading.
        Lidar reads nothing closer than about 20 cm, so a hand near the chest has none: then
        the depth is estimated from how big the hand looks, fitted to this wearer's hand
        whenever lidar and size are both available. Coarser, but it keeps the guidance going."""
        cfg = self.cfg
        z_lidar = None
        for u, v in hand.samples:
            p = to_xyz(u, v, frame.depth, frame.K, r=cfg.hand_patch, nearest=True)
            if p is not None:
                z_lidar = float(p[2])
                break
        z_size = None
        if hand.px_per_m:
            raw = frame.K[0, 0] / hand.px_per_m
            if z_lidar is not None and 0.25 <= z_lidar <= 0.8 and 0.6 < z_lidar / raw < 1.6:
                self._hand_scale.append(z_lidar / raw)
            z_size = raw * (float(np.median(self._hand_scale)) if len(self._hand_scale) >= 10 else 1.0)
            if z_lidar is not None and z_size < 0.35 and z_lidar > 1.8 * z_size:
                z_lidar = None       # the hand is plainly close, yet lidar reports far: it is reading past it
        if z_lidar is not None:
            z, src, reach = z_lidar, "lidar", z_lidar <= cfg.max_hand_m
        elif z_size is not None:
            reach = z_size <= 1.3 * cfg.max_hand_m       # a hand across the room is small in the picture
            z, src = (z_size, "size") if cfg.size_depth else (None, "none")
        else:
            return None, "none", True                    # nothing to judge by: may be ours, held too close
        xyz = None if z is None else back_project(hand.tip[0], hand.tip[1], z, frame.K)
        return xyz, src, reach

    def locate(self, frame: Frame, target: str) -> Scene:
        cfg = self.cfg
        tracing.begin_frame(frame.t)
        t_start = time.perf_counter()
        t_models = 0.0               # time inside the models, the rest of locate() is our own maths
        if target != self.target:
            self.target = target
            self._forget()
            if self.detector is not None:
                t = time.perf_counter()
                self.detector.set_target(target)     # encodes the text prompt: slow, once per target
                t_models += time.perf_counter() - t
                tracing.event("detector.set_target", (time.perf_counter() - t) * 1000, target=target)

        # Object box: fresh from the detector, or reused for a few frames.
        if self.detector is not None:
            if self._i % cfg.detect_every == 0 or not self._seen:
                t = time.perf_counter()
                dets = self.detector.detect(frame.rgb)
                t_models += time.perf_counter() - t
                tracing.add("perception.detector", (time.perf_counter() - t) * 1000)
                det = self._pick(dets)
                if det is not None:
                    self._box = det.box
                self._seen = det is not None
            self._i += 1
        seen = self._seen

        # Hand: only one within arm's reach can be the wearer's. Of those, the one nearest the object.
        hands = []
        if self.hands is not None:
            t = time.perf_counter()
            hands = self.hands.detect(frame.rgb, int(frame.t * 1000))
            t_models += time.perf_counter() - t
            tracing.add("perception.hand", (time.perf_counter() - t) * 1000)
        hand, hand_raw = None, None
        reach = [(k, p, src) for k, (p, src, ok) in ((k, self._hand_xyz(k, frame)) for k in hands) if ok]
        if reach:
            if self._box is not None:
                cx, cy = 0.5 * (self._box[0] + self._box[2]), 0.5 * (self._box[1] + self._box[3])
                hand, hand_raw, src = min(reach, key=lambda kp: (kp[0].tip[0] - cx) ** 2 + (kp[0].tip[1] - cy) ** 2)
            else:
                hand, hand_raw, src = max(reach, key=lambda kp: kp[0].score)
            tracing.put(hand_depth=src)

        # Object position. Measured only when the detector sees it and the hand is not over it.
        pose = frame.pose if (cfg.use_pose and frame.pose is not None) else None
        if (pose is not None) != self._anchored and self._anchor is not None:
            self._anchor = None      # pose came or went: the remembered point is in the wrong frame
            self._obj_avg.clear()
        self._anchored = pose is not None
        covered = self._covered(self._box, hand)
        measured = box_xyz(self._box, frame.depth, frame.K) if seen and not covered else None
        if measured is None and seen and covered and self._anchor is None:
            # First sight of the object with the hand already on it: nothing is remembered yet,
            # so read the depth from the part of the box the hand does not cover.
            measured = box_xyz(self._box, frame.depth, frame.K, exclude=hand_mask(frame.depth.shape, hand.points))
        visible = measured is not None
        if measured is not None:
            p = measured if pose is None else pose[:3, :3] @ measured + pose[:3, 3]
            self._anchor = self._obj_avg.add(p)
            self._anchor_t = frame.t
        elif self._anchor is not None and seen and covered:
            # Still detected under the hand, but the depth there is the hand's: keep the
            # remembered depth and only follow the box sideways.
            if not self._anchored:
                b = self._box
                self._anchor = self._obj_avg.add(
                    back_project(0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3]), self._anchor[2], frame.K))
            self._anchor_t = frame.t
            visible = True
        elif self._anchor is not None:
            limit = cfg.hold_pose_s if self._anchored else (cfg.hold_covered_s if covered else cfg.hold_s)
            if frame.t - self._anchor_t > limit:
                self._anchor = None      # hidden for too long to trust
                self._obj_avg.clear()
                self._box = None
        object_xyz = None
        if self._anchor is not None:
            object_xyz = self._anchor if pose is None else pose[:3, :3].T @ (self._anchor - pose[:3, 3])

        hand_xyz = None if hand_raw is None else self._hand_avg.add(hand_raw)
        if hand_xyz is None:
            self._hand_avg.clear()

        tracing.add("perception.depth", (time.perf_counter() - t_start - t_models) * 1000)
        return Scene(
            object_xyz=object_xyz, hand_xyz=hand_xyz,
            object_visible=visible, hand_visible=hand is not None,
            object_box=None if self._box is None else tuple(int(round(v)) for v in self._box),
            hand_px=None if hand is None else (int(round(hand.tip[0])), int(round(hand.tip[1]))),
            t=frame.t)


# ---------------------------------------------------------------------------
# The interface main.py uses.

_default: Perception | None = None


def build(detector: str | None = None, hand: str | None = None) -> Perception:
    """Perception from the environment variables in the module docstring."""
    detector = detector or os.environ.get("PERCEPTION_DETECTOR", "yoloworld")
    hand = hand or os.environ.get("PERCEPTION_HAND", "mediapipe")
    device = os.environ.get("PERCEPTION_DEVICE") or None
    tracing.meta(detector=detector, hand=hand)
    with tracing.timed("load.detector", kind=detector):
        if detector == "none":
            det = None
        elif detector == "coco":
            det = CocoDetector(device=device)
        else:
            det = OpenVocabDetector(detector, device=device)
    with tracing.timed("load.hand", kind=hand):
        if hand == "none":
            hands = None
        elif hand == "sticker":
            hsv = os.environ.get("PERCEPTION_STICKER_HSV")
            v = [int(x) for x in hsv.split(",")] if hsv else [35, 80, 80, 85, 255, 255]
            hands = StickerHand(v[:3], v[3:])
        else:
            hands = MediaPipeHand()
    cfg = PerceptionConfig(use_pose=os.environ.get("PERCEPTION_USE_POSE", "") not in ("", "0"),
                           detect_every=int(os.environ.get("PERCEPTION_DETECT_EVERY", "1")))
    return Perception(det, hands, cfg)


def locate(frame: Frame, target: str) -> Scene:
    global _default
    if _default is None:
        _default = build()
    return _default.locate(frame, target)
