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
    HAND_OPTION=1..7 or a name  which of HAND_OPTIONS below finds the hand (default 1, the one the
                                demo was built on). It can be changed while running: say "hand option 3".
    HAND_POINT=index_tip|palm|knuckles|nearest      which point of the hand is guided to the object
                                (default: whatever the hand option says). Options 2 and up only.
    HAND_FILTER=average|one_euro|kalman             smoothing of that point (default: the option's)
    PERCEPTION_USE_POSE=1       remember the object in world coordinates. Only switch this on
                                after tools/check_pose.py says the pose is usable.
    PERCEPTION_DETECT_EVERY=1   run the detector on every Nth frame
    PERCEPTION_DEVICE=cuda|cpu  torch device for the detector (default: ultralytics picks)
    PERCEPTION_PARALLEL=0       run the hand tracker after the detector instead of alongside it
    PERCEPTION_FP16=0           run the detector in full precision (default: half precision on the GPU)
    PERCEPTION_FAST_CORES=0     let the hand tracker run on any core (default: only the fast ones)
    PERCEPTION_HELPER=auto|off  when the detector cannot find the object, ask Cosmos for its box
                                in the background (auto: only if tools/run_cosmos.sh is running)
    MODELS_DIR=PATH             where weights are kept (default: models/ in this repo)

The detector and tracker wrappers are adapted from the RTHACK handguide target and hand modules.
"""
from __future__ import annotations

import math
import os
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass

import cv2
import numpy as np

import hands as trackers
import tracing
from hands import (DEPTH_MAX, DEPTH_MIN, MODELS_DIR, Hand, MediaPipeHand, StickerHand,  # noqa: F401
                   back_project, fast_cores)
from hands import fp16 as _fp16
from interfaces import Frame, Scene

# Ways of finding the wearer's hand, to try against each other. (key, name said aloud, what it is)
# Each is a tracker from hands.py plus the settings in _HAND_SETUPS that go with it.
HAND_OPTIONS = [
    ("mediapipe", "MediaPipe", "The tracker the demo was built on, unchanged: MediaPipe on the CPU, depth "
                               "from lidar at the fingertip or else from the hand's size."),
    ("mediapipe_steady", "MediaPipe, steady", "The same model looking a little wider and letting go later, with one "
                                              "continuous depth for the hand, a filter that does not lag, and "
                                              "bystanders' hands rejected by the way their arm comes in."),
    ("yolo", "YOLO hand", "A keypoint model on the GPU that looks at every frame afresh, 3 ms instead of 15. "
                          "Trained on hands seen from behind."),
    ("yolo_medium", "YOLO hand, large", "The larger version of the same model: a little slower, holds on to "
                                        "hands that are partly out of the picture."),
    ("hybrid", "YOLO hand with depth", "YOLO hand, and when it loses the hand the lidar picture of the arm carries "
                                       "on from where it was: for the hand too close, blurred or round the object."),
    ("arm", "depth only", "No hand model at all: the arm is the near thing that comes in over the edge of the "
                          "lidar picture, the hand is its far end. Works with gloves; fooled by a knee."),
    ("hybrid_palm", "YOLO hand with depth, palm", "Option 5 guiding the middle of the palm instead of the index "
                                                  "fingertip: steadier, and the hand arrives ready to close."),
]


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


def surface_depth(vals: np.ndarray, share: float = 0.12, least: int = 5) -> float:
    """The depth of the nearest surface of some size among these lidar readings.

    Readings of one surface lie within a few centimetres of each other. Going from the
    nearest reading outward, the first such group that holds `share` of all the readings is
    the object's front; its median is the answer. A solid object filling the patch gives the
    plain median. A clear bottle gives the label and the water, not the table and wall seen
    through it; a few stray near readings (a finger, lidar noise at an edge) are passed over."""
    v = np.sort(np.asarray(vals, np.float64))
    end = np.searchsorted(v, v + np.maximum(0.03, 0.06 * v), side="right")
    count = end - np.arange(len(v))
    big = np.nonzero(count >= max(float(least), share * len(v)))[0]
    i = int(big[0]) if len(big) else int(np.argmax(count))
    return float(np.median(v[i:end[i]]))


def box_xyz(box, depth: np.ndarray, K: np.ndarray, exclude: np.ndarray | None = None, loose: bool = True):
    """3D point of an object: its box centre, at the depth of its front surface.

    The surface is looked for in the box's centre third, where a solid object is all there
    is to see. Lidar goes through clear plastic and glass, so a water bottle can have no
    reading at all there: then (unless loose=False) the whole box is used, where the label,
    the water or the foot of the bottle still give one. That reading is less sure: now and
    then it is the table seen through the bottle. With `exclude` (a mask of pixels that are
    hand, not object) the whole box minus those pixels is used from the start."""
    x1, y1, x2, y2 = box
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    h, w = depth.shape[:2]

    def readings(part: float):
        hw, hh = max(2.0, (x2 - x1) / part), max(2.0, (y2 - y1) / part)
        xa, xb = int(max(0, cx - hw)), int(min(w, cx + hw + 1))
        ya, yb = int(max(0, cy - hh)), int(min(h, cy + hh + 1))
        if xb <= xa or yb <= ya:
            return np.empty(0), 1
        vals = depth[ya:yb, xa:xb]
        ok = (vals > DEPTH_MIN) & (vals < DEPTH_MAX)
        if exclude is not None:
            ok &= ~exclude[ya:yb, xa:xb].astype(bool)
        return vals[ok], vals.size

    if exclude is None:
        vals, area = readings(6.0)
        if vals.size >= max(5, 0.3 * area):      # the middle is mostly there: a solid object
            return back_project(cx, cy, surface_depth(vals, share=0.12), K)
        if not loose:
            return None
    vals, area = readings(2.2)                   # a little inside the box: its edge is background
    if vals.size < 30:
        return None
    # One lidar reading is a dozen pixels here, so 40 pixels are three readings that agree.
    return back_project(cx, cy, surface_depth(vals, share=0.10, least=40), K)


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
        self.name = "YOLOE" if self.is_yoloe else "YOLO-World"
        with _in_models_dir():
            self.model = (YOLOE if self.is_yoloe else YOLOWorld)(weights)
        self.device, self.conf, self.imgsz = device, conf, imgsz
        self.quantize = 16 if _fp16(device) else None
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
        res = self.model.predict(bgr, conf=self.conf, imgsz=self.imgsz, device=self.device,
                                 quantize=self.quantize, verbose=False)[0]
        return _detections(res, self.names)


class CocoDetector:
    """Fallback: a standard YOLO with the 80 COCO classes (cup, banana, bottle, sports ball...)."""

    ALIASES = {"mug": "cup", "glass": "cup", "ball": "sports ball", "tennis ball": "sports ball",
               "phone": "cell phone", "water bottle": "bottle", "remote control": "remote"}

    def __init__(self, weights: str = "yolo11s.pt", device: str | None = None, conf: float = 0.2, imgsz: int = 640):
        from ultralytics import YOLO

        self.name = "YOLO"
        with _in_models_dir():
            self.model = YOLO(weights)
        self.device, self.conf, self.imgsz = device, conf, imgsz
        self.quantize = 16 if _fp16(device) else None
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
                                 classes=self.classes, quantize=self.quantize, verbose=False)[0]
        return _detections(res, self.model.names)


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
    shadow_s: float = 0.5        # the hand tracker lost the hand less than this ago: it is still over the object
    leap_m: float = 0.10         # the object's depth coming nearer by more than this in one frame, to where the
                                 # hand is, is the hand being read
    pick_reach_m: float = 1.1    # of several things the detector takes for the target, one nearer than this wins
    reach_m: float = 0.75        # farther from the camera than this, a seated wearer cannot reach it
    aim_inner: float = 0.7       # the hand is guided to the nearest point of the middle 70% of the
                                 # object, so any part of it counts, not only its centre. 0 = centre only
    helper_after_s: float = 1.0  # detector has found nothing for this long: ask the helper (Cosmos)
    helper_every_s: float = 3.0  # ...and not more often than this; doubles after each "not there".
                                 # Each call loads the GPU for 0.6-2.6 s and slows the loop while it runs
    helper_hold_s: float = 5.0   # trust a position from the helper this long without a newer one
    size_depth: bool = True      # no lidar on the hand (closer than about 20 cm): estimate its depth from its size
    parallel: bool = True        # hand tracker (CPU) runs while the detector (GPU) does, not after it
    # The hand, for hand options 2 and up. The defaults below are option 1: what the demo was built on.
    hand_depth: str = "switch"   # "switch": lidar at the fingertip, or else the hand's size. "fused": one
                                 # continuous depth from lidar over the palm and the hand's shape
    hand_point: str = "index_tip"  # which point of the hand is guided: index_tip, palm, knuckles, nearest
    hand_filter: str = "average"   # smoothing of that point: average (of `smooth` frames), one_euro, kalman
    own_arm: bool = False        # a hand whose forearm leaves by the top of the picture is somebody else's
    hand_coast_s: float = 0.0    # hand not found for less than this: carry on along its path (kalman only)
    stick_s: float = 0.5         # stay with the hand that was followed this recently: another one, elsewhere
    hand_speed_px_s: float = 2500.0   # in the picture, is not it. Elsewhere = farther than its own size plus
                                 # this speed (a fast reach at 30 cm crosses the picture in a third of a second)
    lidar_from_m: float = 0.20   # lidar reads a surface as broad as a palm from this distance out


# Settings that go with each of HAND_OPTIONS: (tracker in hands.make, config).
_FUSED = dict(hand_depth="fused", hand_filter="one_euro", own_arm=True)
_HAND_SETUPS = {
    "mediapipe": ("mediapipe", {}),
    "mediapipe_steady": ("mediapipe_wide", _FUSED),
    "yolo": ("yolo", _FUSED),
    "yolo_medium": ("yolo_medium", _FUSED),
    "hybrid": ("hybrid", dict(_FUSED, hand_filter="kalman", hand_coast_s=0.25)),
    "arm": ("arm", dict(_FUSED, hand_filter="kalman")),
    "hybrid_palm": ("hybrid", dict(_FUSED, hand_filter="kalman", hand_coast_s=0.25, hand_point="palm")),
    "sticker": ("sticker", {}),
}
_HAND_DEFAULTS = dict(hand_depth="switch", hand_point="index_tip", hand_filter="average", own_arm=False,
                      hand_coast_s=0.0)


@dataclass
class HandDetails:
    """Everything known about the wearer's hand in one frame, for whoever decides "grab".
    Positions are metres in the camera frame; None where the tracker cannot tell."""
    xyz: np.ndarray | None        # the guided point, smoothed: the same as Scene.hand_xyz
    px: tuple                     # ...and where it is in the picture
    point: str                    # which point that is: index_tip, palm, knuckles, nearest
    confidence: float             # 0..1, the tracker's own
    tracker: str                  # which tracker found it, e.g. "YOLO hand (small)" or "Depth arm"
    depth_from: str               # lidar, size, blend, or none
    box: tuple | None             # x1, y1, x2, y2 px
    keypoints_px: np.ndarray | None = None    # 21 x 2
    keypoints_xyz: np.ndarray | None = None   # 21 x 3, this frame, not smoothed
    palm_xyz: np.ndarray | None = None        # middle of the palm
    index_tip_xyz: np.ndarray | None = None
    nearest_m: float | None = None            # from the nearest fingertip or knuckle to the object's nearest point
    coasting: bool = False                    # not seen in this frame: carried on along its path


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


class CosmosHelper:
    """Asks Cosmos where the object is when the detector cannot find it. One question at a time,
    on its own thread, so the loop never waits for the 0.6 to 2 s answer."""

    def __init__(self, find=None):
        if find is None:
            import cosmos
            find = cosmos.find
        self._find = find
        self._lock = threading.Lock()
        self._busy = False
        self._result = None

    def idle(self) -> bool:
        with self._lock:
            return not self._busy

    def request(self, frame: Frame, target: str) -> bool:
        with self._lock:
            if self._busy:
                return False
            self._busy = True
        threading.Thread(target=self._run, args=(frame, target), daemon=True, name="cosmos-helper").start()
        return True

    def _run(self, frame: Frame, target: str):
        try:
            box = self._find(frame.rgb, target)["box"]
        except Exception as e:
            print(f"[perception] Cosmos helper failed: {e}", file=sys.stderr)
            box = None
        with self._lock:
            self._result = (frame, target, box)
            self._busy = False

    def poll(self):
        """(the frame that was asked about, target, box or None) once per answer, else None."""
        with self._lock:
            out, self._result = self._result, None
            return out


class Perception:
    def __init__(self, detector=None, hands=None, cfg: PerceptionConfig | None = None, helper=None):
        self.detector, self.hands, self.helper = detector, hands, helper
        self.cfg = cfg or PerceptionConfig()
        self.target = None
        self._obj_avg = _Average(self.cfg.smooth, self.cfg.jump_m)
        self._hand_avg = _Average(self.cfg.smooth, self.cfg.jump_m)
        self._hand_scale = deque(maxlen=90)   # lidar depth / size-based depth, to fit the size estimate to this hand
        self._hand_depth = trackers.HandDepth()     # the same for hand_depth="fused"; it outlives a change of target
        self._hand_filter = None
        self._hand_last = None       # (t, u, v) of the hand last taken as the wearer's
        self._next_hands = None      # (tracker, settings) waiting to be swapped in between two frames
        self._hand_pool = None
        self.hand = None             # HandDetails for the last frame, or None
        self._set_hands(hands, {})
        self._forget()

    def use_hands(self, tracker, **settings):
        """Change the hand tracker and the settings that go with it. Safe from any thread and in
        the middle of a reach: the swap happens between two frames."""
        self._next_hands = (tracker, settings)

    def _set_hands(self, tracker, settings: dict):
        self.hands = tracker
        for name, value in settings.items():
            setattr(self.cfg, name, value)
        self._hand_filter = None if self.cfg.hand_filter == "average" else trackers.make_filter(self.cfg.hand_filter)
        self._hand_avg.clear()
        self._hand_depth.reset()
        self._hand_last = None
        if self._hand_pool is None and self.cfg.parallel and tracker is not None:
            # One worker, so the hand tracker still sees frames one at a time and in order.
            cores = fast_cores()
            self._hand_pool = ThreadPoolExecutor(1, thread_name_prefix="hand",
                                                 initializer=lambda: cores and os.sched_setaffinity(0, cores))

    def _forget(self):
        self._box = None
        self._seen = False           # did the detector find the object the last time it ran?
        self._anchor = None          # remembered object position, world frame if anchored else camera
        self._anchor_t = 0.0
        self._anchored = False       # is _anchor in world coordinates?
        self._from_helper = False    # did _anchor come from the helper rather than the detector?
        self._unseen_t = None        # since when the detector has found nothing
        self._blocked_t = None       # since when something nearer has stood in front of the object
        self._shadow = None          # (the wearer's hand, when) the last time the tracker had it
        self._other = False          # the detector has moved to a different thing than the one remembered
        self._hand_z = None          # (when, depth) of the wearer's hand, the last time it had one
        self._helper_t = -1e9        # when the helper was last asked
        self._helper_gap = self.cfg.helper_every_s
        self._score = 0.0
        self._said_seen = None       # what the log last said about the object and the hand,
        self._said_hand = None       # so it reports changes, not every frame
        self._said_t = -1e9
        self._i = 0
        self._goal = None            # where the hand was last being sent, for hand_point="nearest"
        self._obj_avg.clear()
        self._hand_avg.clear()

    def _track_hands(self, frame: Frame):
        t = time.perf_counter()
        if getattr(self.hands, "uses_depth", False):
            hands = self.hands.detect(frame.rgb, int(frame.t * 1000), depth=frame.depth, K=frame.K)
        else:
            hands = self.hands.detect(frame.rgb, int(frame.t * 1000))
        return hands, (time.perf_counter() - t) * 1000

    def _pick(self, dets: list[Detection], frame: Frame | None = None):
        """The detection to follow. One the wearer can reach always wins over one they cannot:
        there may be another bottle on a desk across the room, and the detector likes that one
        just as much. Among those, the one already being followed, then the best score."""
        if not dets:
            return None

        def within_reach(d: Detection) -> int:
            if frame is None:
                return 1
            p = box_xyz(d.box, frame.depth, frame.K)
            return 1 if p is None else (2 if p[2] <= self.cfg.pick_reach_m else 0)     # no reading: cannot rule it out

        def rank(d: Detection):
            return within_reach(d), d.score + (0.5 * iou(d.box, self._box) if self._box is not None else 0.0)

        best = max(dets[:6], key=rank)
        # While an object within reach is remembered, a different one out of reach is not "it":
        # the near one is only missed for a moment (a clear bottle often is).
        near = self._anchor is not None and not self._anchored and self._anchor[2] <= self.cfg.pick_reach_m
        if near and within_reach(best) == 0 and iou(best.box, self._box) == 0.0:
            return None
        return best

    def _covered(self, box, hand: Hand | None) -> bool:
        """Is any part of the hand over the object's box? Then the box's depth is the hand's."""
        if box is None or hand is None:
            return False
        mx, my = self.cfg.box_margin * (box[2] - box[0]), self.cfg.box_margin * (box[3] - box[1])
        p = hand.points
        return bool(np.any((p[:, 0] >= box[0] - mx) & (p[:, 0] <= box[2] + mx) &
                           (p[:, 1] >= box[1] - my) & (p[:, 1] <= box[3] + my)))

    def _blocked(self, box, z: float, depth: np.ndarray) -> bool:
        """Is something nearer than the object standing in front of its box? An arm reaching
        for it, whether or not the hand tracker has found the hand. Then what lidar reads there
        is the arm, and the detector not seeing the object means hidden, not gone."""
        if box is None:
            return False
        h, w = depth.shape[:2]

        def part(grow: float):
            gx, gy = grow * (box[2] - box[0]), grow * (box[3] - box[1])
            xa, xb = int(max(0, box[0] - gx)), int(min(w, box[2] + gx + 1))
            ya, yb = int(max(0, box[1] - gy)), int(min(h, box[3] + gy + 1))
            return depth[ya:yb:2, xa:xb:2]

        vals = part(0.0)
        if vals.size == 0:
            return False
        if float(np.mean((vals > DEPTH_MIN) & (vals < z - 0.06))) >= 0.3:
            return True
        # A hand closer than lidar ranges leaves a hole, and a wide one: over the box and well
        # past it. (A clear bottle is a hole too, but only as big as itself.)
        return float(np.mean(part(0.5) <= DEPTH_MIN)) >= 0.8

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
            if z_lidar is not None:
                # Both readings side by side in the trace: this is how far off a camera-only
                # estimate of the hand would be, measured against lidar.
                tracing.put(hand_z_lidar=round(z_lidar, 3), hand_z_size=round(z_size, 3), hand_z_size_raw=round(raw, 3))
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

    def _measure(self, hand: Hand, frame: Frame):
        """hand_depth="fused", one candidate hand: (lidar depth or None, depth from its shape for
        an average hand or None, its 21 points from the shape or None, could it be the wearer's?)."""
        cfg, hd = self.cfg, self._hand_depth
        pts = trackers.hand_points_3d(hand, frame.K)
        z_shape = None if pts is None else float(np.mean(pts[list(trackers.PALM), 2]))
        if hand.z is not None:
            z_lidar = hand.z
        elif hand.kp is not None:
            z_lidar, _ = trackers.palm_depth(hand, frame.depth)
        elif hand.box is not None:
            z_lidar = trackers.box_depth(hand.box, frame.depth)
        else:
            z_lidar = next((float(p[2]) for p in (to_xyz(u, v, frame.depth, frame.K, r=cfg.hand_patch, nearest=True)
                                                  for u, v in hand.samples) if p is not None), None)
        z_sized = hd.sized(z_shape)
        if hand.ours is not None:
            return z_lidar, z_shape, pts, hand.ours
        if cfg.own_arm and hand.kp is not None and (hand.sure is None or hand.sure[0]) and trackers.arm_from_above(hand.kp):
            return z_lidar, z_shape, pts, False
        # Lidar far beyond what the hand's size allows is reading past the hand, not the hand.
        past = z_lidar is not None and z_sized is not None and z_lidar > 1.5 * z_sized + 0.05
        if z_lidar is not None and not past:
            return z_lidar, z_shape, pts, z_lidar <= cfg.max_hand_m
        if z_sized is not None:
            return z_lidar, z_shape, pts, z_sized <= cfg.max_hand_m
        return z_lidar, z_shape, pts, True               # nothing to judge by: may be ours, held too close

    @staticmethod
    def _palm_is_hole(hand: Hand, depth: np.ndarray) -> bool:
        h, w = depth.shape[:2]
        palm = np.round(hand.kp[list(trackers.PALM)]).astype(int)
        palm = palm[(palm[:, 0] >= 0) & (palm[:, 0] < w) & (palm[:, 1] >= 0) & (palm[:, 1] < h)]
        if len(palm) < 2 or not np.any(depth[::16, ::16] > DEPTH_MIN):
            return False
        return bool(np.all(depth[palm[:, 1], palm[:, 0]] <= DEPTH_MIN))

    def _fused_hand(self, hands: list, frame: Frame):
        """hand_depth="fused": (the wearer's hand, its guided point in metres before smoothing,
        that point in the picture, where the depth came from, its 21 points in metres).

        The depth is one estimate from lidar over the palm and the hand's shape (see
        hands.HandDepth). Each keypoint is then placed on its own pixel's ray, at that depth
        plus how much nearer or farther the shape says the keypoint is than the palm."""
        cfg, hd = self.cfg, self._hand_depth
        cands = [(k, m) for k, m in ((k, self._measure(k, frame)) for k in hands) if m[3]]
        # A hand the model sees beats one that is only being followed in the depth picture.
        cands = [c for c in cands if c[0].kp is not None] or cands
        if not cands:
            return None, None, None, "none", None
        last = self._hand_last
        if last is not None and 0 <= frame.t - last[0] <= cfg.stick_s:
            # Stay with the hand that was being followed. A second hand coming nearer the object
            # (someone handing it over) must not take the guidance away, and if the wearer's own
            # hand is missed for a frame, the hand that is found somewhere else is not it.
            dt = frame.t - last[0]
            reach_px, reach_m = last[3] + cfg.hand_speed_px_s * dt, 0.15 + 1.5 * dt

            def same(c) -> bool:
                k, (z_lidar, z_shape, _, _) = c
                z = z_lidar if hd.agrees(z_lidar, hd.sized(z_shape)) or z_shape is None else hd.sized(z_shape)
                return (math.hypot(k.tip[0] - last[1], k.tip[1] - last[2]) <= reach_px
                        and (z is None or last[4] is None or abs(z - last[4]) <= reach_m))

            cands = [c for c in cands if same(c)]
            if not cands:
                return None, None, None, "none", None
            hand, m = min(cands, key=lambda c: (c[0].tip[0] - last[1]) ** 2 + (c[0].tip[1] - last[2]) ** 2)
        elif self._box is not None:
            cx, cy = 0.5 * (self._box[0] + self._box[2]), 0.5 * (self._box[1] + self._box[3])
            hand, m = min(cands, key=lambda c: (c[0].tip[0] - cx) ** 2 + (c[0].tip[1] - cy) ** 2)
        else:
            hand, m = max(cands, key=lambda c: c[0].score)
        z_lidar, z_shape, pts, _ = m
        hd.learn(z_lidar, z_shape)
        # Lidar reads a palm from 20 cm out. No reading on it while there are readings elsewhere
        # means it is nearer than that, whatever its size suggests: a hand that close is cut by
        # the edge of the picture and looks smaller than it is.
        near = z_lidar is None and hand.kp is not None and self._palm_is_hole(hand, frame.depth)
        z, lidar_share = hd.fuse(z_lidar, z_shape, frame.t, at_most=cfg.lidar_from_m if near else None)
        tracing.put(hand_z=None if z is None else round(z, 3), hand_lidar_share=round(lidar_share, 2),
                    hand_z_lidar=None if z_lidar is None else round(z_lidar, 3),
                    hand_z_size=None if z_shape is None else round(hd.sized(z_shape), 3))
        if z is None:
            return hand, None, hand.tip, "none", None
        src = "lidar" if lidar_share > 0.7 or z_shape is None else ("size" if lidar_share < 0.3 else "blend")
        if pts is None:
            return hand, back_project(hand.tip[0], hand.tip[1], z, frame.K), hand.tip, src, None
        K, kp = frame.K, np.asarray(hand.kp, float)
        zs = pts[:, 2] * (z / z_shape)
        xyz = np.column_stack([(kp[:, 0] - K[0, 2]) * zs / K[0, 0], (kp[:, 1] - K[1, 2]) * zs / K[1, 1], zs])
        uv, which = trackers.guided_point(cfg.hand_point, kp, xyz, self._goal)
        return hand, xyz[which].mean(0), (float(uv[0]), float(uv[1])), src, xyz

    def _aim(self, point: np.ndarray, object_xyz: np.ndarray, K: np.ndarray) -> np.ndarray:
        """Where to send a point of the hand: the nearest point of the object's middle region,
        at the object's depth. Reaching any part of the object is a success, not only its centre."""
        b, z = self._box, object_xyz[2]
        half = self.cfg.aim_inner * 0.5 * np.array([(b[2] - b[0]) * z / K[0, 0], (b[3] - b[1]) * z / K[1, 1]])
        off = np.clip(point[:2] - object_xyz[:2], -half, half)
        return np.array([object_xyz[0] + off[0], object_xyz[1] + off[1], z])

    def _ask_helper(self, frame: Frame, target: str, seen: bool, pose):
        """Take in an answer from the helper if one has arrived, and ask again if the detector
        is still blind. The answer is about an older frame: the position is read from that
        frame's depth, and carried into the present through the poses when they are in use.
        Without poses it is taken as it stands, which is right while the wearer stands still."""
        cfg = self.cfg
        if seen:
            self._unseen_t = None
        elif self._unseen_t is None:
            self._unseen_t = frame.t
        answer = self.helper.poll()
        if answer is not None:
            old, asked, box = answer
            if asked == self.target and not seen:
                p = box_xyz(box, old.depth, old.K) if box is not None else None
                if p is not None and pose is not None:
                    p = None if old.pose is None else old.pose[:3, :3] @ p + old.pose[:3, 3]
                if p is not None:
                    self._obj_avg.clear()
                    self._anchor = self._obj_avg.add(p)
                    self._anchor_t = frame.t
                    self._from_helper = True
                    self._box = box
                    self._helper_gap = cfg.helper_every_s
                else:
                    self._helper_gap = min(2 * self._helper_gap, 8.0)   # not there: ask less often
        if (not seen and frame.t - self._unseen_t >= cfg.helper_after_s
                and frame.t - self._helper_t >= self._helper_gap and self.helper.idle()):
            self._helper_t = frame.t
            self.helper.request(frame, target)

    def locate(self, frame: Frame, target: str) -> Scene:
        cfg = self.cfg
        tracing.begin_frame(frame.t)
        t_start = time.perf_counter()
        t_models = 0.0               # time inside the models, the rest of locate() is our own maths
        if self._next_hands is not None:
            (tracker, settings), self._next_hands = self._next_hands, None
            self._set_hands(tracker, settings)
        if target != self.target:
            self.target = target
            self._forget()
            if self.detector is not None:
                t = time.perf_counter()
                self.detector.set_target(target)     # encodes the text prompt: slow, once per target
                t_models += time.perf_counter() - t
                tracing.event("detector.set_target", (time.perf_counter() - t) * 1000, target=target,
                              model=getattr(self.detector, "name", "detector"))

        # Hand tracker starts first and runs alongside the detector: they share nothing.
        t_par = time.perf_counter()
        hand_job = None
        if self._hand_pool is not None:
            hand_job = self._hand_pool.submit(self._track_hands, frame)

        # Object box: fresh from the detector, or reused for a few frames.
        if self.detector is not None:
            if self._i % cfg.detect_every == 0 or not self._seen:
                t = time.perf_counter()
                dets = self.detector.detect(frame.rgb)
                if hand_job is None:
                    t_models += time.perf_counter() - t
                tracing.add("perception.detector", (time.perf_counter() - t) * 1000)
                det = self._pick(dets, frame)
                if det is not None:
                    # A box that shares nothing with the last one is another thing altogether (the
                    # bottle on this desk instead of the one across the room): start afresh on it.
                    self._other = self._box is not None and iou(det.box, self._box) == 0.0
                    self._box = det.box
                    self._score = det.score
                self._seen = det is not None
            self._i += 1
        seen = self._seen

        # Hand: only one within arm's reach can be the wearer's. Of those, the one nearest the object.
        hands = []
        if hand_job is not None:
            hands, ms = hand_job.result()
            t_models += time.perf_counter() - t_par     # both models, overlapped
            tracing.add("perception.hand", ms)
        elif self.hands is not None:
            t = time.perf_counter()
            hands, ms = self._track_hands(frame)
            t_models += time.perf_counter() - t
            tracing.add("perception.hand", ms)
        hand, hand_raw, hand_uv, src, hand_points = None, None, None, "none", None
        if cfg.hand_depth == "fused":
            hand, hand_raw, hand_uv, src, hand_points = self._fused_hand(hands, frame)
        else:
            reach = [(k, p, src) for k, (p, src, ok) in ((k, self._hand_xyz(k, frame)) for k in hands) if ok]
            if reach:
                if self._box is not None:
                    cx, cy = 0.5 * (self._box[0] + self._box[2]), 0.5 * (self._box[1] + self._box[3])
                    hand, hand_raw, src = min(reach, key=lambda kp: (kp[0].tip[0] - cx) ** 2 + (kp[0].tip[1] - cy) ** 2)
                else:
                    hand, hand_raw, src = max(reach, key=lambda kp: kp[0].score)
                hand_uv = hand.tip
        if hand_raw is not None:
            self._hand_z = (frame.t, float(hand_raw[2]))
        if hand is not None:
            tracing.put(hand_depth=src)
            size = 150.0 if hand.box is None else max(hand.box[2] - hand.box[0], hand.box[3] - hand.box[1])
            self._hand_last = (frame.t, hand.tip[0], hand.tip[1], size, None if hand_raw is None else float(hand_raw[2]))
        if hasattr(self.hands, "chosen"):
            self.hands.chosen(hand, int(frame.t * 1000), None if hand_raw is None else float(hand_raw[2]))

        # Object position. Measured only when the detector sees it and the hand is not over it.
        pose = frame.pose if (cfg.use_pose and frame.pose is not None) else None
        if (pose is not None) != self._anchored and self._anchor is not None:
            self._anchor = None      # pose came or went: the remembered point is in the wrong frame
            self._obj_avg.clear()
        self._anchored = pose is not None
        if self.helper is not None:
            self._ask_helper(frame, target, seen, pose)
        # The hand tracker blinks, a frame or two at a time, just when the hand is on the object.
        # A hand seen a moment ago is still there for the question "whose depth is in this box".
        if hand is not None:
            self._shadow = (hand, frame.t)
        shadow = hand if hand is not None else (
            self._shadow[0] if self._shadow is not None and 0 <= frame.t - self._shadow[1] < cfg.shadow_s else None)
        covered = self._covered(self._box, shadow)
        z_was = None if self._anchor is None or self._other else float(
            self._anchor[2] if pose is None else (pose[:3, :3].T @ (self._anchor - pose[:3, 3]))[2])
        # Something in front of the object is believed for hold_covered_s at most. Past that it
        # is not an arm passing: the object itself has come nearer, and is read again.
        believe = self._blocked_t is None or frame.t - self._blocked_t <= cfg.hold_covered_s
        measured, in_front = None, False
        if seen and not covered:
            measured = box_xyz(self._box, frame.depth, frame.K, loose=False)
            if measured is None:
                # Nothing to read in the middle of the box (a clear bottle). The reading from the
                # whole box is good enough to start from, or to confirm what is remembered, but
                # not to move the object somewhere else.
                measured = box_xyz(self._box, frame.depth, frame.K)
                if measured is not None and z_was is not None and abs(measured[2] - z_was) > max(0.05, 0.10 * z_was):
                    measured = None
            hand_z = self._hand_z[1] if self._hand_z is not None and 0 <= frame.t - self._hand_z[0] < 1.5 else None
            if (measured is not None and z_was is not None and hand_z is not None and believe
                    and measured[2] < z_was - cfg.leap_m and abs(measured[2] - hand_z) < 0.08):
                # The object has leapt toward the camera, to where the hand is: what was read is
                # the hand, in a frame where the tracker did not place it over the box.
                in_front, measured = True, None
        elif not seen and not covered and z_was is not None:
            in_front = self._blocked(self._box, z_was, frame.depth)      # hidden behind an arm, not gone
        self._blocked_t = (self._blocked_t if self._blocked_t is not None else frame.t) if in_front else None
        covered = covered or (in_front and believe)
        if measured is None and seen and covered and (self._anchor is None or self._other) and shadow is not None:
            # First sight of the object with the hand already on it: nothing is remembered yet,
            # so read the depth from the part of the box the hand does not cover.
            measured = box_xyz(self._box, frame.depth, frame.K, exclude=hand_mask(frame.depth.shape, shadow.points))
        visible = measured is not None
        if measured is not None:
            p = measured if pose is None else pose[:3, :3] @ measured + pose[:3, 3]
            if self._other:
                self._obj_avg.clear()
                self._other = False
            self._anchor = self._obj_avg.add(p)
            self._anchor_t = frame.t
            self._from_helper = False
        elif self._anchor is not None and seen and not self._other:
            # Still detected, but its depth cannot be read: the hand is over it and the depth
            # there is the hand's, or it is a clear bottle and lidar goes through it. Keep the
            # remembered depth and only follow the box sideways. It is not lost: it is in view.
            if not self._anchored:
                b = self._box
                self._anchor = self._obj_avg.add(
                    back_project(0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3]), self._anchor[2], frame.K))
            self._anchor_t = frame.t
            visible = True
        elif self._anchor is not None:
            limit = cfg.hold_pose_s if self._anchored else (cfg.hold_covered_s if covered else cfg.hold_s)
            if self._from_helper:
                limit = max(limit, cfg.helper_hold_s)
            if frame.t - self._anchor_t > limit:
                self._anchor = None      # hidden for too long to trust
                self._from_helper = False
                self._obj_avg.clear()
                self._box = None
        tracing.put(object_from="detector" if measured is not None else
                    ("cosmos" if self._from_helper and self._anchor is not None else
                     ("memory" if self._anchor is not None else "nothing")))
        object_xyz = None
        if self._anchor is not None:
            object_xyz = self._anchor if pose is None else pose[:3, :3].T @ (self._anchor - pose[:3, 3])

        coasting = False
        if hand_raw is None:
            hand_xyz = None
            self._hand_avg.clear()
            if hand is None and cfg.hand_coast_s > 0 and isinstance(self._hand_filter, trackers.Kalman):
                # The tracker blinked. For a few frames the hand is where its path was taking it.
                hand_xyz = self._hand_filter.ahead(frame.t, cfg.hand_coast_s)
                coasting = hand_xyz is not None
        elif self._hand_filter is None:
            hand_xyz = self._hand_avg.add(hand_raw)
        else:
            hand_xyz = self._hand_filter.add(hand_raw, frame.t)

        # Where to send the hand: the nearest point of the object's middle region.
        aim = None
        can_aim = object_xyz is not None and self._box is not None and cfg.aim_inner > 0
        if can_aim and hand_xyz is not None:
            aim = self._aim(hand_xyz, object_xyz, frame.K)
        self._goal = aim if aim is not None else object_xyz

        self.hand = None
        if hand is not None or coasting:
            nearest = None
            if hand_points is not None and can_aim:
                ends = hand_points[list(trackers.TIPS + trackers.KNUCKLES)]
                nearest = min(float(np.linalg.norm(self._aim(p, object_xyz, frame.K) - p)) for p in ends)
            self.hand = HandDetails(
                xyz=hand_xyz, px=None if hand_uv is None else (float(hand_uv[0]), float(hand_uv[1])),
                point=cfg.hand_point if hand_points is not None else "index_tip",
                confidence=0.0 if hand is None else float(hand.score),
                tracker=getattr(self.hands, "name", "") if hand is None else (hand.source or getattr(self.hands, "name", "")),
                depth_from=src, box=None if hand is None else hand.box,
                keypoints_px=None if hand is None else hand.kp, keypoints_xyz=hand_points,
                palm_xyz=None if hand_points is None else hand_points[list(trackers.PALM)].mean(0),
                index_tip_xyz=None if hand_points is None else hand_points[trackers.INDEX_TIP],
                nearest_m=nearest, coasting=coasting)

        self._log_changes(frame.t, target, seen, hand is not None)
        tracing.add("perception.depth", (time.perf_counter() - t_start - t_models) * 1000)
        scene = Scene(
            object_xyz=object_xyz, hand_xyz=hand_xyz,
            object_visible=visible, hand_visible=hand is not None,
            object_box=None if self._box is None else tuple(int(round(v)) for v in self._box),
            hand_px=None if hand is None else (int(round(hand_uv[0])), int(round(hand_uv[1]))),
            t=frame.t, aim_xyz=aim)
        # Not in the agreed Scene yet: read these with getattr(scene, name, None).
        scene.hand = self.hand
        scene.object_out_of_reach = object_xyz is not None and float(np.linalg.norm(object_xyz)) > cfg.reach_m
        return scene

    def _log_changes(self, t: float, target: str, seen: bool, hand: bool):
        """One log line when the detector finds or loses the object, or the hand tracker the hand."""
        if t - self._said_t < 0.7:           # a flicker is not news
            return
        if self.detector is not None and seen != self._said_seen:
            self._said_seen, self._said_t = seen, t
            tracing.event("detector", found=seen, target=target, model=getattr(self.detector, "name", "detector"),
                          score=round(self._score, 2) if seen else None)
        if self.hands is not None and hand != self._said_hand:
            self._said_hand, self._said_t = hand, t
            tracing.event("hand", seen=hand, model=getattr(self.hands, "name", "hand tracker"))


# ---------------------------------------------------------------------------
# The interface main.py uses.

_default: Perception | None = None
_hand_lock = threading.Lock()
_hand_loading = threading.Lock()     # one model loads at a time
_hand_number: int | None = None      # the hand option in use or on its way in; None until someone asks
_hand_request = 0                    # counts the requests, so a slow load that was overtaken is dropped
_hand_trackers: dict = {}            # loaded trackers by their hands.make name: switching back is instant


def _hand_index(which) -> int | None:
    """A number from 1 or a key from HAND_OPTIONS -> its index, or None."""
    which = str(which).strip().lower()
    keys = [k for k, _, _ in HAND_OPTIONS]
    if which.isdigit() and 1 <= int(which) <= len(keys):
        return int(which) - 1
    return keys.index(which) if which in keys else None


def _hand_settings(key: str) -> dict:
    settings = dict(_HAND_DEFAULTS, **_HAND_SETUPS[key][1])
    if settings["hand_depth"] == "fused":            # option 1 stays exactly as it was
        point, smoothing = os.environ.get("HAND_POINT"), os.environ.get("HAND_FILTER")
        if point in trackers.POINTS:
            settings["hand_point"] = point
        if smoothing in ("average", "one_euro", "kalman"):
            settings["hand_filter"] = smoothing
    return settings


def _hand_tracker(kind: str):
    """The tracker of that name, loaded and warmed up once."""
    with _hand_loading:
        if kind not in _hand_trackers:
            device = os.environ.get("PERCEPTION_DEVICE") or None
            fetch = _default is None     # weights are only ever downloaded before the run, not during it
            if kind == "hybrid":         # shares its model with the plain YOLO option
                yolo = _hand_trackers.get("yolo") or trackers.make("yolo", device, fetch)
                _hand_trackers["yolo"] = yolo
                tracker = trackers.Hybrid(yolo)
            else:
                tracker = trackers.make(kind, device, fetch)
            # The first frame through a GPU model takes a second or more. Spend it here, not in the loop.
            tracker.detect(np.zeros((960, 720, 3), np.uint8), 0)
            _hand_trackers[kind] = tracker
        return _hand_trackers[kind]


def _load_hand_option(number: int):
    """(option number that loaded, tracker, settings). An option whose model cannot be loaded
    (weights missing and no network, no GPU) gives way to option 1, and says why."""
    key = HAND_OPTIONS[number - 1][0]
    try:
        return number, _hand_tracker(_HAND_SETUPS[key][0]), _hand_settings(key)
    except Exception as e:
        if number == 1:
            raise
        print(f"[perception] hand option {number} ({key}) could not load: {e}. Staying with option 1.", file=sys.stderr)
        tracing.event("hand_option.failed", option=key, number=number, reason=str(e)[:200])
        return 1, _hand_tracker(_HAND_SETUPS[HAND_OPTIONS[0][0]][0]), _hand_settings(HAND_OPTIONS[0][0])


def _switch_hands(number: int, request: int):
    global _hand_number
    loaded, tracker, settings = _load_hand_option(number)
    with _hand_lock:
        if request != _hand_request:
            return                       # another option was asked for while this one loaded
        _hand_number = loaded
    if _default is not None:
        _default.use_hands(tracker, **settings)
        tracing.event("hand_option.ready", option=HAND_OPTIONS[loaded - 1][0], number=loaded)


def set_hand_option(which) -> str:
    """Switch the way the hand is found while running. Returns its spoken name at once. The
    model loads on a helper thread (a second or two the first time) while the old tracker
    carries on, and takes over between two frames. If it cannot load, option 1 is used and
    the reason is logged."""
    global _hand_number, _hand_request
    i = _hand_index(which)
    if i is None:
        raise ValueError(f"no hand option {which!r}")
    with _hand_lock:
        _hand_number = i + 1
        _hand_request += 1
        request = _hand_request
    key, title, _ = HAND_OPTIONS[i]
    tracing.event("hand_option", option=key, number=i + 1, title=title)
    if _default is not None:             # before the first frame, build() picks the choice up
        threading.Thread(target=_switch_hands, args=(i + 1, request), daemon=True, name="hand-option").start()
    return title


def hand_option() -> tuple[int, str, str]:
    """(number, key, name) of the hand option in use, or on its way in."""
    number = _hand_number or (_hand_index(os.environ.get("HAND_OPTION", "1")) or 0) + 1
    key, title, _ = HAND_OPTIONS[number - 1]
    return number, key, title


def set_hand_point(kind: str) -> str:
    """Which point of the hand is guided to the object: index_tip, palm, knuckles or nearest.
    Takes effect on the next frame, for hand options 2 and up."""
    if kind not in trackers.POINTS:
        raise ValueError(f"no hand point {kind!r}; one of: " + ", ".join(trackers.POINTS))
    os.environ["HAND_POINT"] = kind      # so the next option switched to keeps it
    if _default is not None and _default.cfg.hand_depth == "fused":
        _default.cfg.hand_point = kind
    tracing.event("hand_point", point=kind)
    return kind


def hand_details() -> HandDetails | None:
    """Everything known about the wearer's hand in the last frame (also on the Scene as .hand)."""
    return None if _default is None else _default.hand


def build(detector: str | None = None, hand: str | None = None) -> Perception:
    """Perception from the environment variables in the module docstring."""
    global _hand_number
    detector = detector or os.environ.get("PERCEPTION_DETECTOR", "yoloworld")
    hand = hand or os.environ.get("PERCEPTION_HAND", "mediapipe")
    device = os.environ.get("PERCEPTION_DEVICE") or None
    with tracing.timed("load.detector", kind=detector):
        if detector == "none":
            det = None
        elif detector == "coco":
            det = CocoDetector(device=device)
        else:
            det = OpenVocabDetector(detector, device=device)
    settings = {}
    with tracing.timed("load.hand", kind=hand):
        if hand == "none":
            hands = None
        elif hand == "sticker":
            hands = trackers.make("sticker")
        else:
            if _hand_index(os.environ.get("HAND_OPTION", "1")) is None:
                print(f"[perception] HAND_OPTION={os.environ['HAND_OPTION']!r} is not one of the hand options; "
                      "using option 1", file=sys.stderr)
            number, hands, settings = _load_hand_option(hand_option()[0])
            with _hand_lock:
                _hand_number = number
            hand = HAND_OPTIONS[number - 1][0]
    tracing.meta(detector=detector, hand=hand)
    cfg = PerceptionConfig(use_pose=os.environ.get("PERCEPTION_USE_POSE", "") not in ("", "0"),
                           detect_every=int(os.environ.get("PERCEPTION_DETECT_EVERY", "1")),
                           parallel=os.environ.get("PERCEPTION_PARALLEL", "1") != "0", **settings)
    helper = None
    if os.environ.get("PERCEPTION_HELPER", "auto") != "off":
        import cosmos
        if cosmos.available():
            helper = CosmosHelper()
            print("[perception] Cosmos is running: it will be asked when the detector cannot find the object")
    tracing.meta(helper="cosmos" if helper is not None else "none")
    return Perception(det, hands, cfg, helper)


def locate(frame: Frame, target: str) -> Scene:
    global _default
    if _default is None:
        _default = build()
    return _default.locate(frame, target)
