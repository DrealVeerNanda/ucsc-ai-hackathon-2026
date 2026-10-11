"""Ways of finding the wearer's hand, to try against each other.

    tracker = hands.make("yolo")            # or mediapipe, mediapipe_wide, yolo_medium, arm, hybrid, sticker
    for hand in tracker.detect(frame.rgb, t_ms, depth=frame.depth, K=frame.K): ...

Every tracker has detect(rgb, t_ms, depth=None, K=None) -> [Hand] and a name. A Hand is pixels
only: perception.py turns it into metres, decides whether it is the wearer's, and picks the
point to guide. The helpers for that live here too, so they can be tested without a model:

    hand_points_3d   the 21 keypoints in the camera frame, from the hand's shape alone
    palm_depth       lidar depth of the palm, read over the whole palm rather than one fingertip
    HandDepth        the two fused into one depth that never jumps when the lidar comes or goes
    arm_exit         which edge of the picture the forearm leaves by (the top: not the wearer's)
    guided_point     which point of the hand is "the hand" (index tip, palm, knuckles, nearest)
    OneEuro, Kalman  smoothing for the guided point

The camera is on the chest, so the wearer's own hand is seen from behind, comes in over the
bottom or a side edge, is often cut by the edge, and at arm's length or closer. A bystander's
hand is across the table and hangs from the top of the picture.

Weights (all in MODELS_DIR, fetched on first use; ~/.cache/proprio is looked in as well):
    hand_landmarker.task     MediaPipe, 8 MB
    wilor_hand_small.pt      YOLOv8n-pose with 21 hand keypoints, 7 MB   } the hand detector of WiLoR
    wilor_hand_medium.pt     YOLO11m-pose, 42 MB                          } (CC BY-NC: not for sale)
"""
from __future__ import annotations

import math
import os
import sys
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MODELS_DIR = Path(os.environ.get("MODELS_DIR") or Path(__file__).resolve().parent / "models")
HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/"
                  "float16/latest/hand_landmarker.task")
YOLO_HAND = {"small": ("wilor_hand_small.pt", "https://huggingface.co/rolpotamias/WiLoR/resolve/main/detector_small.pt"),
             "medium": ("wilor_hand_medium.pt",
                        "https://huggingface.co/rolpotamias/WiLoR/resolve/main/detector_medium_v2.pt")}
DEPTH_MIN, DEPTH_MAX = 0.15, 4.0     # lidar readings outside this are not trusted

# Keypoint numbers, the same in MediaPipe and the YOLO hand models: 0 wrist, 1-4 thumb,
# 5-8 index (5 its knuckle, 8 its tip), 9-12 middle, 13-16 ring, 17-20 little finger.
WRIST, INDEX_TIP = 0, 8
KNUCKLES = (5, 9, 13, 17)
PALM = (0, 5, 9, 13, 17)            # the part of the hand that does not bend
TIPS = (4, 8, 12, 16, 20)
POINTS = ("index_tip", "palm", "knuckles", "nearest")

# The palm of an average adult hand, metres, flat in its own plane: x across the knuckles
# toward the little finger, y from the wrist toward the fingers. For trackers that give
# keypoints in the picture but no 3D shape. The knuckle row is 6 cm from joint to joint.
PALM_SHAPE = np.array([[0.0, 0.0, 0.0], [-0.022, 0.089, 0.0], [0.0, 0.091, 0.0],
                       [0.019, 0.085, 0.0], [0.035, 0.074, 0.0]])


def weights_path(name: str, url: str | None = None) -> Path:
    """Where a weights file is, downloading it into MODELS_DIR if it is nowhere yet."""
    for folder in (MODELS_DIR, Path.home() / ".cache" / "proprio"):
        if (folder / name).exists():
            return folder / name
    if url is None:
        raise FileNotFoundError(f"{name} is not in {MODELS_DIR}")
    path = MODELS_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[hands] downloading {name} to {path}", file=sys.stderr)
    part = path.with_suffix(path.suffix + ".part")     # a download cut short must not pass for the file
    urllib.request.urlretrieve(url, part)
    os.replace(part, path)
    return path


def fast_cores() -> set[int] | None:
    """The CPU's fast cores, if it has two kinds. The GX10 has 10 Cortex-X925 at 3.9 GHz and
    10 Cortex-A725 at 2.8 GHz; MediaPipe takes 2.5x as long on the slow ones (24 ms instead of
    9.5 ms a frame with no hand, 41 instead of 15 with two), and Linux moves it between them."""
    if os.environ.get("PERCEPTION_FAST_CORES", "1") == "0" or not hasattr(os, "sched_setaffinity"):
        return None
    freq = {}
    for c in os.sched_getaffinity(0):
        try:
            freq[c] = int(Path(f"/sys/devices/system/cpu/cpu{c}/cpufreq/cpuinfo_max_freq").read_text())
        except (OSError, ValueError):
            return None
    top = max(freq.values(), default=0)
    fast = {c for c, f in freq.items() if f == top}
    return fast if 0 < len(fast) < len(freq) else None


@contextmanager
def _on_cores(cores: set[int] | None):
    """Run the block on these cores; threads it starts keep them. Linux sets this per thread."""
    if not cores:
        yield
        return
    old = os.sched_getaffinity(0)
    os.sched_setaffinity(0, cores)
    try:
        yield
    finally:
        os.sched_setaffinity(0, old)


def fp16(device) -> bool:
    """Half precision on the GPU: about 25% faster on the GX10, boxes within a pixel."""
    if os.environ.get("PERCEPTION_FP16", "1") == "0" or (device and not str(device).startswith("cuda")):
        return False
    import torch
    return torch.cuda.is_available()


@dataclass
class Hand:
    tip: tuple                 # (u, v) of the point to guide: the index fingertip
    points: np.ndarray         # N x 2 px, everything known to be hand (for "is it covering the object")
    samples: list              # (u, v) pixels to read depth at, best first
    score: float = 1.0         # how sure the tracker is that this is a hand, 0 to 1
    px_per_m: float | None = None   # how big the hand looks: pixels per metre of real hand, if known
    kp: np.ndarray | None = None    # 21 x 2 px keypoints, numbered as above, if the tracker gives them
    world: np.ndarray | None = None  # 21 x 3 m, the hand's 3D shape about its own centre (MediaPipe)
    box: tuple | None = None        # x1, y1, x2, y2 px around the hand
    z: float | None = None          # depth of `tip` in metres, when the tracker measured it itself
    source: str = ""                # which tracker found it
    ours: bool | None = None        # the tracker already knows whose hand it is (None: perception decides)


def _box_of(pts: np.ndarray, shape) -> tuple:
    h, w = shape[:2]
    x1, y1 = np.clip(pts.min(0), 0, (w - 1, h - 1))
    x2, y2 = np.clip(pts.max(0), 0, (w - 1, h - 1))
    return float(x1), float(y1), float(x2), float(y2)


def _size_scale(pts: np.ndarray) -> float:
    # Wrist to middle knuckle is about 9.5 cm and the knuckle row about 8 cm. One of the two
    # faces the camera whichever way the hand points, so the larger gives the scale.
    return float(max(np.linalg.norm(pts[0] - pts[9]) / 0.095, np.linalg.norm(pts[5] - pts[17]) / 0.08))


# ---------------------------------------------------------------------------
# Trackers.

class MediaPipeHand:
    """MediaPipe hand landmarker. The guided point is the index fingertip (landmark 8); depth is
    read there first, then further down the finger if the tip has no valid lidar reading.

    pad > 0 adds a border of repeated edge pixels before the model looks: its palm detector
    wants to see the whole palm, and the wearer's hand is often cut by the edge of the picture.
    track_conf below min_conf keeps following a blurred hand the detector would not find anew."""

    def __init__(self, model_path: str | Path | None = None, num_hands: int = 2, min_conf: float = 0.5,
                 track_conf: float | None = None, pad: int = 0, name: str = "MediaPipe"):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        path = Path(model_path) if model_path else weights_path("hand_landmarker.task", HAND_MODEL_URL)
        self._mp = mp
        track = min_conf if track_conf is None else track_conf
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path), delegate=BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO, num_hands=num_hands,
            min_hand_detection_confidence=min_conf, min_hand_presence_confidence=track,
            min_tracking_confidence=track)
        with _on_cores(fast_cores()):   # MediaPipe's own threads start here and keep these cores
            self.landmarker = vision.HandLandmarker.create_from_options(opts)
        self.name, self.pad = name, int(pad)
        self._last_ts = -1

    def detect(self, rgb: np.ndarray, t_ms: int, depth=None, K=None) -> list[Hand]:
        h, w = rgb.shape[:2]
        ts = max(int(t_ms), self._last_ts + 1)   # video mode needs strictly increasing timestamps
        self._last_ts = ts
        pad = self.pad
        if pad:
            rgb = cv2.copyMakeBorder(rgb, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self.landmarker.detect_for_video(img, ts)
        worlds = getattr(res, "hand_world_landmarks", None) or [None] * len(res.hand_landmarks)
        hands = []
        for lms, hd, wl in zip(res.hand_landmarks, res.handedness, worlds):
            pts = np.array([[p.x * (w + 2 * pad) - pad, p.y * (h + 2 * pad) - pad] for p in lms], dtype=np.float32)
            world = None if wl is None else np.array([[p.x, p.y, p.z] for p in wl], dtype=np.float64)
            hands.append(Hand(tip=tuple(pts[8]), points=pts, samples=[tuple(pts[i]) for i in (8, 7, 6, 5)],
                              score=float(hd[0].score) if hd else 0.0, px_per_m=_size_scale(pts),
                              kp=pts, world=world, box=_box_of(pts, (h, w)), source=self.name))
        return hands

    def close(self):
        self.landmarker.close()


class StickerHand:
    """Fallback: a bright sticker or glove on the reaching hand, found by colour. Installs
    nowhere and runs at any frame rate. Default range is green; set PERCEPTION_STICKER_HSV."""

    name = "Sticker"

    def __init__(self, lo=(35, 80, 80), hi=(85, 255, 255), min_area: int = 120):
        self.lo, self.hi, self.min_area = np.array(lo, np.uint8), np.array(hi, np.uint8), min_area

    def detect(self, rgb: np.ndarray, t_ms: int = 0, depth=None, K=None) -> list[Hand]:
        mask = cv2.inRange(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV), self.lo, self.hi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
        if n < 2:
            return []
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[i, cv2.CC_STAT_AREA] < self.min_area:
            return []
        c = (float(cents[i][0]), float(cents[i][1]))
        return [Hand(tip=c, points=np.array([c], np.float32), samples=[c], source=self.name)]


class YoloHand:
    """A YOLO pose model on the GPU: every hand in the picture with its 21 keypoints, in one
    pass of 3 to 6 ms. It looks at each frame afresh, so a fast or blurred hand is not "lost"
    the way a frame-to-frame tracker loses it, and it was trained on hands seen from behind.
    It also finds the small far hands of bystanders: perception.py has to reject those."""

    def __init__(self, size: str = "small", device: str | None = None, conf: float = 0.3, imgsz: int = 512,
                 kp_conf: float = 0.3):
        from ultralytics import YOLO

        self.model = YOLO(str(weights_path(*YOLO_HAND[size])))
        self.device, self.conf, self.imgsz, self.kp_conf = device, conf, imgsz, kp_conf
        self.quantize = 16 if fp16(device) else None
        self.name = f"YOLO hand ({size})"

    def detect(self, rgb: np.ndarray, t_ms: int = 0, depth=None, K=None) -> list[Hand]:
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        res = self.model.predict(bgr, conf=self.conf, imgsz=self.imgsz, device=self.device,
                                 quantize=self.quantize, verbose=False)[0]
        if res.keypoints is None or res.boxes is None or len(res.boxes) == 0:
            return []
        kps = res.keypoints.data.float().cpu().numpy()          # N x 21 x (x, y, confidence)
        boxes, scores = res.boxes.xyxy.float().cpu().numpy(), res.boxes.conf.float().cpu().numpy()
        hands = []
        for kp, box, score in zip(kps, boxes, scores):
            pts, sure = kp[:, :2].astype(np.float32), kp[:, 2] >= self.kp_conf
            if not sure[list(PALM)].all():
                # The palm is what size, depth and "whose hand" are judged by. A hand with its
                # palm out of the picture is reported, but as a blob: a place, not a pose.
                c = (float(box[0] + box[2]) / 2, float(box[1] + box[3]) / 2)
                known = pts[sure] if sure.any() else np.array([c], np.float32)
                hands.append(Hand(tip=c, points=known, samples=[c], score=float(score),
                                  box=tuple(float(v) for v in box), source=self.name))
                continue
            order = [i for i in (8, 7, 6, 5) if sure[i]]
            hands.append(Hand(tip=tuple(pts[8]), points=pts, samples=[tuple(pts[i]) for i in order],
                              score=float(score), px_per_m=_size_scale(pts), kp=pts,
                              box=tuple(float(v) for v in box), source=self.name))
        return hands


def skin(rgb: np.ndarray) -> np.ndarray:
    """Pixels that could be skin, any skin tone, by colour alone (red-difference and
    blue-difference ranges, as in the classic face detectors). Wood and cardboard pass too:
    this is only good for telling an arm from the holes the lidar leaves in far, dark things."""
    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    return cv2.inRange(ycc, (40, 135, 80), (255, 180, 130)) > 0


class DepthArm:
    """No learned model: the wearer's arm is the near thing in the lidar picture that comes in
    over the bottom or a side edge, and the hand is its far end. Works for any hand shape,
    glove or grip, and when the hand is too close or too blurred for a keypoint model.

    A hand closer than the lidar can range (about 20 cm) is a hole in the depth, as is
    anything far and dark; the hole counts as arm only where the colour could be skin.
    What it cannot tell apart from an arm: anything else near that crosses the bottom or a
    side edge with a free end, such as the wearer's knee when sitting. So on its own it is
    an option to try, and its main use is inside Hybrid, which only asks it where the hand it
    was already following has gone."""

    name = "Depth arm"
    uses_depth = True                # perception.py hands detect() the depth picture as well
    STEP = 4                         # the lidar picture is 256 x 192; the frame is 3.75x that

    def __init__(self, near_m: float = 0.75, enter_m: float = 0.45, edge_m: float = 0.04, min_frac: float = 0.006,
                 too_near_m: float = 0.17):
        self.near_m, self.enter_m, self.edge_m = near_m, enter_m, edge_m
        self.min_frac, self.too_near_m = min_frac, too_near_m

    def blobs(self, rgb: np.ndarray, depth: np.ndarray):
        """(labels, [blob]) at 1/STEP size. A blob is a dict: label, area, where it enters."""
        s = self.STEP
        d = depth[s // 2::s, s // 2::s]
        valid = (d > DEPTH_MIN) & (d < DEPTH_MAX)
        hole = (d <= DEPTH_MIN) & skin(rgb[s // 2::s, s // 2::s])     # too close to range, and skin-coloured
        near = (valid & (d < self.near_m)) | hole
        # Cut the near region wherever the depth steps: an arm over a table is nearer than the
        # table by more than a few centimetres all the way round, and that step is its outline.
        dz = np.where(valid, d, 0.0).astype(np.float32)
        step = np.zeros(d.shape, bool)
        gx = np.abs(dz[:, 1:] - dz[:, :-1]) > self.edge_m
        gy = np.abs(dz[1:, :] - dz[:-1, :]) > self.edge_m
        both_x, both_y = valid[:, 1:] & valid[:, :-1], valid[1:, :] & valid[:-1, :]
        step[:, 1:] |= gx & both_x & (dz[:, 1:] > dz[:, :-1])       # mark the far side of each step,
        step[:, :-1] |= gx & both_x & (dz[:, :-1] > dz[:, 1:])      # so the near thing keeps its edge
        step[1:, :] |= gy & both_y & (dz[1:, :] > dz[:-1, :])
        step[:-1, :] |= gy & both_y & (dz[:-1, :] > dz[1:, :])
        n, labels, stats, _ = cv2.connectedComponentsWithStats((near & ~step).astype(np.uint8), connectivity=4)
        h, w = d.shape
        out = []
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < self.min_frac * h * w:
                continue
            x, y, bw, bh = (int(stats[i, k]) for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH,
                                                       cv2.CC_STAT_HEIGHT))
            if not (x == 0 or x + bw == w or y + bh == h):          # must reach a side or the bottom edge
                continue
            m = labels == i
            edge = np.zeros_like(m)
            edge[:, 0], edge[:, -1], edge[-1, :] = m[:, 0], m[:, -1], m[-1, :]
            ev, eu = np.nonzero(edge)
            de = d[ev, eu]
            de = de[de > DEPTH_MIN]
            # Where it comes in it is forearm or elbow, close to the body. A table edge or the
            # floor crossing the border further away is not an arm.
            if de.size and float(np.median(de)) > self.enter_m:
                continue
            out.append({"label": i, "area": area, "enter": (float(eu.mean()), float(ev.mean())), "mask": m})
        return labels, d, out

    def _hand(self, blob, d, shape) -> Hand:
        s = self.STEP
        m = blob["mask"]
        vv, uu = np.nonzero(m)
        eu, ev = blob["enter"]
        far = (uu - eu) ** 2 + (vv - ev) ** 2
        k = int(np.argmax(far))
        tu, tv = float(uu[k]), float(vv[k])
        # The end of the arm: the part of the blob within a hand's length of its far point.
        length = math.sqrt(float(far[k]))
        end = (uu - tu) ** 2 + (vv - tv) ** 2 <= max(6.0, 0.35 * length) ** 2
        zs = d[vv[end], uu[end]]
        zs = zs[zs > DEPTH_MIN]
        # Mostly hole at the hand: it is closer than the lidar reaches. Say so with the nearest
        # depth lidar can give, which is roughly right and keeps the hand "seen".
        z = float(np.median(zs)) if zs.size >= max(4, 0.3 * int(end.sum())) else self.too_near_m
        pts = np.stack([uu[end], vv[end]], 1).astype(np.float32)
        hull = cv2.convexHull(pts).reshape(-1, 2) if len(pts) >= 3 else pts
        to_px = lambda p: (np.asarray(p, np.float32) * s + s / 2)      # noqa: E731
        tip = to_px((tu, tv))
        h, w = shape[:2]
        solid = min(1.0, blob["area"] / (0.03 * m.size)) * (1.0 if zs.size else 0.6)
        return Hand(tip=(float(tip[0]), float(tip[1])), points=to_px(hull), samples=[(float(tip[0]), float(tip[1]))],
                    score=round(0.3 + 0.5 * solid, 2), box=_box_of(to_px(pts), (h, w)), z=z, source=self.name,
                    ours=True)

    def detect(self, rgb: np.ndarray, t_ms: int = 0, depth=None, K=None, near=None) -> list[Hand]:
        """`near` = (u, v, radius) px: only the arm whose hand end is within that circle."""
        if depth is None or not np.any(depth > DEPTH_MIN):
            return []                                    # no lidar at all: nothing to go by
        _, d, blobs = self.blobs(rgb, depth)
        hands = [self._hand(b, d, rgb.shape) for b in sorted(blobs, key=lambda b: -b["area"])]
        if near is not None:
            u, v, r = near
            hands = [k for k in hands if (k.tip[0] - u) ** 2 + (k.tip[1] - v) ** 2 <= r * r
                     or np.min(np.sum((k.points - (u, v)) ** 2, axis=1)) <= r * r]
        return hands[:1]


class Hybrid:
    """A keypoint model while it sees the hand; when it does not, the lidar picture carries on
    from where the hand was. The keypoint model gives the pose and tells the wearer's hand from
    anyone else's; the depth blob needs neither a sharp picture nor a whole hand, so it covers
    exactly the moments the model drops out: the hand too close, smeared, or wrapped round the
    object. It is only asked about the arm that ends where the hand was last seen, for
    bridge_s at most, so the knee and the table never start a track."""

    uses_depth = True

    def __init__(self, learned, arm: DepthArm | None = None, bridge_s: float = 1.5, grow_px_s: float = 900.0):
        self.learned, self.arm = learned, arm or DepthArm()
        self.bridge_s, self.grow_px_s = bridge_s, grow_px_s
        self.name = f"{learned.name} + depth arm"
        self._last = None            # (t_ms, u, v, size px) of the wearer's hand when the model last had it
        self._offset = np.zeros(2)   # from the arm's far end to the guided point, learnt while both are seen

    def chosen(self, hand: Hand | None, t_ms: int):
        """perception.py reports which hand it took as the wearer's (None: none of them)."""
        if hand is None or hand.source == self.arm.name:
            return
        size = 120.0 if hand.box is None else max(hand.box[2] - hand.box[0], hand.box[3] - hand.box[1])
        self._last = (t_ms, float(hand.tip[0]), float(hand.tip[1]), float(size))

    def detect(self, rgb: np.ndarray, t_ms: int, depth=None, K=None) -> list[Hand]:
        hands = self.learned.detect(rgb, t_ms, depth=depth, K=K)
        if self._last is None or depth is None:
            return hands
        t0, u, v, size = self._last
        age = (t_ms - t0) / 1000.0
        if not 0 <= age <= self.bridge_s:
            self._last = None
            return hands
        radius = size + self.grow_px_s * age                 # a hand moves; look further the longer it is gone
        here = [k for k in hands if (k.tip[0] - u) ** 2 + (k.tip[1] - v) ** 2 <= radius * radius and k.kp is not None]
        arm = self.arm.detect(rgb, t_ms, depth=depth, near=(u, v, radius))
        if here:
            if arm:      # both see it: remember where the guided point sits relative to the arm's end
                k = min(here, key=lambda k: (k.tip[0] - u) ** 2 + (k.tip[1] - v) ** 2)
                self._offset = 0.7 * self._offset + 0.3 * (np.array(k.tip) - np.array(arm[0].tip))
            return hands
        for k in arm:
            tip = np.array(k.tip) + self._offset
            k.tip, k.samples = (float(tip[0]), float(tip[1])), [k.tip]
        return hands + arm


# ---------------------------------------------------------------------------
# From keypoints to metres.

def back_project(u: float, v: float, d: float, K: np.ndarray) -> np.ndarray:
    return np.array([(u - K[0, 2]) * d / K[0, 0], (v - K[1, 2]) * d / K[1, 1], d])


def hand_points_3d(hand: Hand, K: np.ndarray) -> np.ndarray | None:
    """The 21 keypoints in the camera frame, metres, for an average-sized hand, from the hand's
    shape alone (no lidar). None if the hand has no keypoints or the fit fails.

    With MediaPipe's 3D shape of the hand: the pose of that shape which puts its 21 points on
    the 21 pixels. Without it: the same for the palm, which is flat and does not bend, and the
    fingers are taken to lie in the palm's plane. Either way a tilted hand comes out at the
    right distance, which measuring one span in the picture does not give."""
    if hand.kp is None:
        return None
    kp = np.asarray(hand.kp, np.float64)
    Kd = np.asarray(K, np.float64)
    if hand.world is not None:
        try:
            ok, rvec, tvec = cv2.solvePnP(np.asarray(hand.world, np.float64), kp, Kd, None, flags=cv2.SOLVEPNP_SQPNP)
        except cv2.error:
            return None
        if not ok or not np.all(np.isfinite(tvec)) or tvec[2, 0] <= 0.02:
            return None
        R, _ = cv2.Rodrigues(rvec)
        return np.asarray(hand.world, np.float64) @ R.T + tvec.ravel()
    try:
        n, rvecs, tvecs, _ = cv2.solvePnPGeneric(PALM_SHAPE, kp[list(PALM)], Kd, None, flags=cv2.SOLVEPNP_IPPE)
    except cv2.error:
        return None
    best = None
    for rvec, tvec in zip(rvecs, tvecs):
        if not np.all(np.isfinite(tvec)) or tvec[2, 0] <= 0.02:
            continue
        R, _ = cv2.Rodrigues(rvec)
        # A flat shape fits two ways, tilted toward the camera or away. The wearer's fingers
        # point away from the chest, so take the fit whose fingers are the farther end.
        if best is None or R[2, 1] > best[0][2, 1]:
            best = (R, tvec.ravel())
    if best is None:
        return None
    R, t = best
    normal = R[:, 2]
    rays = np.column_stack([(kp[:, 0] - Kd[0, 2]) / Kd[0, 0], (kp[:, 1] - Kd[1, 2]) / Kd[1, 1], np.ones(len(kp))])
    along = rays @ normal
    z_palm = float(np.mean((PALM_SHAPE @ R.T + t)[:, 2]))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(np.abs(along) > 0.15, float(normal @ t) / along, z_palm)
    # A palm seen nearly edge-on makes the plane's far end run away: fingers are within a
    # hand's length of the palm, whatever the fit says.
    z = np.clip(z, z_palm - 0.06, z_palm + 0.12)
    return rays * z[:, None]


def palm_depth(hand: Hand, depth: np.ndarray, min_valid: float = 0.4):
    """(lidar depth of the palm in metres or None, share of the palm with a reading).

    Read at a spread of points over the palm instead of a patch at one fingertip: the palm is
    the widest part of the hand, so the reading is the hand's and not what is behind a finger,
    and one landmark sliding off the hand does not change it."""
    if hand.kp is None:
        return None, 0.0
    kp = np.asarray(hand.kp, np.float32)
    palm = kp[list(PALM)]
    c = palm.mean(0)
    pts = np.concatenate([[c], 0.5 * (palm + c), 0.75 * palm + 0.25 * c])
    h, w = depth.shape[:2]
    u, v = np.round(pts[:, 0]).astype(int), np.round(pts[:, 1]).astype(int)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if inside.sum() < 4:
        return None, 0.0
    vals = depth[v[inside], u[inside]]
    vals = vals[(vals > DEPTH_MIN) & (vals < DEPTH_MAX)]
    share = vals.size / len(pts)
    if share < min_valid:
        return None, share
    return float(np.median(vals)), share


class HandDepth:
    """One depth for the hand out of two that each fail somewhere.

    Lidar is right to a centimetre but reads nothing closer than about 20 cm, reads the
    background between spread fingers, and drops out as the hand turns. The hand's size in
    the picture always gives a depth, but for an average hand: off by this wearer's hand size.

    So the size-based depth carries the estimate from frame to frame, scaled to this wearer
    (the scale is learnt whenever both are valid), and lidar pulls it into place through a
    correction that follows lidar within a few frames while it is there and fades over half a
    second when it is not. The reported depth therefore never steps when lidar comes or goes;
    it slides. With lidar and no keypoints it is plain lidar."""

    def __init__(self, follow_s: float = 0.12, fade_s: float = 0.6, lidar_min: float = 0.22, lidar_max: float = 0.9):
        self.follow_s, self.fade_s = follow_s, fade_s
        self.lidar_min, self.lidar_max = lidar_min, lidar_max
        self.ratios = []             # lidar / size-based depth, for this wearer's hand size
        self._scale = 1.0
        self.reset()

    def reset(self):
        self._corr, self._t, self._weight = 0.0, None, 0.0

    @property
    def scale(self) -> float:
        return self._scale

    def learn(self, z_lidar: float | None, z_shape: float | None):
        """Both valid on the wearer's hand: one more sample of how its size compares to average."""
        if z_lidar is None or z_shape is None or not (0.25 <= z_lidar <= 0.8) or not (0.6 < z_lidar / z_shape < 1.6):
            return
        self.ratios.append(z_lidar / z_shape)
        del self.ratios[:-90]
        if len(self.ratios) >= 10:
            self._scale = float(np.median(self.ratios))

    def sized(self, z_shape: float | None) -> float | None:
        return None if z_shape is None else z_shape * self._scale

    def agrees(self, z_lidar: float | None, z_sized: float | None) -> bool:
        """Is this lidar reading the hand? Not when it is out of lidar's range, or far from
        what the hand's size says (then it is the table or the wall seen past the hand)."""
        if z_lidar is None or not (self.lidar_min <= z_lidar <= self.lidar_max):
            return False
        return z_sized is None or abs(z_lidar - z_sized) <= max(0.08, 0.3 * z_sized)

    def fuse(self, z_lidar: float | None, z_shape: float | None, t: float):
        """(depth or None, how much of it is lidar 0..1). Call once a frame for the chosen hand."""
        z_sized = self.sized(z_shape)
        ok = self.agrees(z_lidar, z_sized)
        if z_sized is None:
            self.reset()
            return (z_lidar, 1.0) if ok or (z_lidar is not None and z_lidar <= self.lidar_max) else (None, 0.0)
        dt = 1.0 if self._t is None else t - self._t
        fresh = self._t is None or dt > 0.5 or dt < 0      # first sight of the hand: nothing to stay close to
        self._t = t
        if ok:
            target = z_lidar - z_sized
            a = 1.0 if fresh else 1.0 - math.exp(-dt / self.follow_s)
            self._corr += a * (target - self._corr)
            self._weight += a * (1.0 - self._weight)
        else:
            k = 0.0 if fresh else math.exp(-dt / self.fade_s)
            self._corr *= k
            self._weight *= k
        return z_sized + self._corr, self._weight


def arm_exit(kp: np.ndarray, shape) -> str:
    """Which edge of the picture the forearm leaves by: "bottom", "left", "right" or "top".
    The forearm runs from the knuckles through the wrist and on. A chest camera sees the
    wearer's own arm come in over the bottom or a side; an arm hanging from the top of the
    picture belongs to someone facing the wearer."""
    h, w = shape[:2]
    wrist = np.asarray(kp[WRIST], float)
    d = wrist - np.asarray(kp[9], float)
    n = float(np.linalg.norm(d))
    if n < 1e-6:
        return "bottom"
    d /= n
    best, edge = math.inf, "bottom"
    for name, axis, limit in (("left", 0, 0.0), ("right", 0, w - 1.0), ("top", 1, 0.0), ("bottom", 1, h - 1.0)):
        if abs(d[axis]) < 1e-6:
            continue
        s = (limit - wrist[axis]) / d[axis]
        if 0 <= s < best:
            best, edge = s, name
    if not (0 <= wrist[0] < w and 0 <= wrist[1] < h):     # the wrist itself is already outside
        over = {"left": -wrist[0], "right": wrist[0] - (w - 1), "top": -wrist[1], "bottom": wrist[1] - (h - 1)}
        edge = max(over, key=over.get)
    return edge


def guided_point(kind: str, kp: np.ndarray, xyz: np.ndarray | None = None, goal: np.ndarray | None = None):
    """(u, v) and index into the keypoints (or None for a mean) of the point that stands for
    the hand. kp is 21 x 2 px; "nearest" also needs the keypoints (xyz) and the goal in metres."""
    if kind == "palm":
        return kp[list(PALM)].mean(0), list(PALM)
    if kind == "knuckles":
        return kp[list(KNUCKLES)].mean(0), list(KNUCKLES)
    if kind == "nearest" and xyz is not None and goal is not None:
        cand = list(TIPS) + list(KNUCKLES)
        i = cand[int(np.argmin(np.linalg.norm(xyz[cand] - np.asarray(goal, float), axis=1)))]
        return kp[i], [i]
    return kp[INDEX_TIP], [INDEX_TIP]


# ---------------------------------------------------------------------------
# Smoothing the guided point. Each has add(p, t) -> p and clear().

class OneEuro:
    """The One Euro filter: smooths hard while the hand is still, where jitter is all there is
    to hear, and lets go as it speeds up, where lag is what hurts. A plain average cannot do
    both. min_cutoff (Hz) sets the smoothing at rest, beta how fast it opens up with speed."""

    def __init__(self, min_cutoff: float = 1.2, beta: float = 12.0, d_cutoff: float = 1.5, gap_s: float = 0.5):
        self.min_cutoff, self.beta, self.d_cutoff, self.gap_s = min_cutoff, beta, d_cutoff, gap_s
        self.clear()

    def clear(self):
        self._x = self._dx = self._t = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def add(self, p: np.ndarray, t: float) -> np.ndarray:
        p = np.asarray(p, float)
        if self._x is None or not (0 < t - self._t <= self.gap_s):
            self._x, self._dx, self._t = p.copy(), np.zeros_like(p), t
            return p.copy()
        dt = t - self._t
        self._t = t
        dx = (p - self._x) / dt
        a = self._alpha(self.d_cutoff, dt)
        self._dx = a * dx + (1 - a) * self._dx
        a = self._alpha(self.min_cutoff + self.beta * float(np.linalg.norm(self._dx)), dt)
        self._x = a * p + (1 - a) * self._x
        return self._x.copy()


class Kalman:
    """Constant-velocity Kalman filter on the 3D point: no lag at a steady speed, and it can
    say where the hand should be for the few frames a tracker drops out (ahead()). It
    overshoots when the hand stops dead. accel (m/s^2) is how hard a hand is expected to
    change speed, noise_m the tracker's jitter."""

    def __init__(self, accel: float = 6.0, noise_m: float = 0.006, gap_s: float = 0.5):
        self.accel, self.noise_m, self.gap_s = accel, noise_m, gap_s
        self.clear()

    def clear(self):
        self._x = self._v = self._t = None
        self._P = np.eye(2)

    def _predict(self, dt: float):
        F = np.array([[1.0, dt], [0.0, 1.0]])
        q = self.accel ** 2
        Q = q * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]])
        return self._x + self._v * dt, F @ self._P @ F.T + Q

    def add(self, p: np.ndarray, t: float) -> np.ndarray:
        p = np.asarray(p, float)
        if self._x is None or not (0 < t - self._t <= self.gap_s):
            self._x, self._v, self._t = p.copy(), np.zeros_like(p), t
            self._P = np.diag([self.noise_m ** 2, 1.0])
            return p.copy()
        x, P = self._predict(t - self._t)
        self._t = t
        gain = P[:, 0] / (P[0, 0] + self.noise_m ** 2)
        resid = p - x
        self._x, self._v = x + gain[0] * resid, self._v + gain[1] * resid
        self._P = P - np.outer(gain, P[0, :])
        return self._x.copy()

    def ahead(self, t: float, limit_s: float = 0.3) -> np.ndarray | None:
        """Where the hand should be at time t, without a measurement; None once that is a guess."""
        if self._x is None or not (0 <= t - self._t <= limit_s):
            return None
        return self._x + self._v * (t - self._t)


def make_filter(kind: str):
    if kind == "one_euro":
        return OneEuro()
    if kind == "kalman":
        return Kalman()
    raise ValueError(f"no hand filter {kind!r}")


# ---------------------------------------------------------------------------

def make(kind: str, device: str | None = None):
    """A tracker by name. Loads its model, so call it off the frame loop."""
    if kind == "mediapipe":
        return MediaPipeHand()
    if kind == "mediapipe_wide":
        return MediaPipeHand(min_conf=0.3, track_conf=0.3, pad=160, name="MediaPipe (wide)")
    if kind in ("yolo", "yolo_small"):
        return YoloHand("small", device=device)
    if kind == "yolo_medium":
        return YoloHand("medium", device=device)
    if kind == "arm":
        return DepthArm()
    if kind == "hybrid":
        return Hybrid(YoloHand("small", device=device))
    if kind == "hybrid_mediapipe":
        return Hybrid(MediaPipeHand(min_conf=0.3, track_conf=0.3, pad=160, name="MediaPipe (wide)"))
    if kind == "sticker":
        hsv = os.environ.get("PERCEPTION_STICKER_HSV")
        v = [int(x) for x in hsv.split(",")] if hsv else [35, 80, 80, 85, 255, 255]
        return StickerHand(v[:3], v[3:])
    raise ValueError(f"no hand tracker {kind!r}")
