"""Fake versions of every lane's module, so main.py runs before anything real exists.

    python main.py                    # everything fake
    python main.py --guidance real    # lane 3 tunes audio against fake scenes

Keys (debug window focused):
    arrows    move hand left / right / up / down, 1 cm
    w / s     move hand forward / back, 1 cm
    o         toggle occlusion (arm covers the object; its position is remembered)
    h         toggle hand visible
    n         toggle +-5 mm jitter on both positions
    r         reset hand to its start position
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from interfaces import Frame, Scene, hand_error

W, H = 480, 640                              # portrait, upright
OBJECT_XYZ = np.array([0.05, 0.12, 0.60])    # meters, camera frame
OBJECT_SIZE = (0.08, 0.10)                   # mug width, height in meters
HAND_START = np.array([-0.15, 0.05, 0.40])
LIDAR_MIN = 0.25                             # lidar reads nothing closer than this
TILT_DEG = 30.0                              # phone tilted down toward the table
TABLE_BELOW = 0.45                           # table surface below the lens, meters


def make_K(w: int, h: int, hfov_deg: float = 63.0) -> np.ndarray:
    fx = (w / 2) / np.tan(np.radians(hfov_deg) / 2)
    return np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1]], dtype=np.float64)


def project(xyz, K) -> tuple[float, float]:
    x, y, z = xyz
    return K[0, 0] * x / z + K[0, 2], K[1, 1] * y / z + K[1, 2]


def object_box(K, xyz=OBJECT_XYZ, size=OBJECT_SIZE) -> tuple[int, int, int, int]:
    u, v = project(xyz, K)
    hw = K[0, 0] * size[0] / 2 / xyz[2]
    hh = K[1, 1] * size[1] / 2 / xyz[2]
    return int(u - hw), int(v - hh), int(u + hw), int(v + hh)


def synth_scene(K, w: int = W, h: int = H) -> tuple[np.ndarray, np.ndarray]:
    """A table seen from a chest-mounted phone, with a red mug at OBJECT_XYZ."""
    rng = np.random.default_rng(0)
    yn = (np.arange(h, dtype=np.float32)[:, None] - K[1, 2]) / K[1, 1]
    t = np.radians(TILT_DEG)
    denom = np.cos(t) * yn + np.sin(t)
    with np.errstate(divide="ignore"):
        table_z = np.where(denom > 0, TABLE_BELOW / denom, np.inf)
    on_table = np.broadcast_to(table_z < 1.3, (h, w))
    depth = np.where(on_table, np.broadcast_to(table_z, (h, w)), 2.0).astype(np.float32)

    rgb = np.empty((h, w, 3), np.uint8)
    rgb[:] = (205, 205, 200)
    shade = np.clip(1.15 - 0.35 * (np.broadcast_to(table_z, (h, w)) - 0.4), 0.6, 1.1)
    wood = (np.array([196, 160, 118])[None, None, :] * shade[..., None]).astype(np.uint8)
    rgb[on_table] = wood[on_table]
    rgb = np.clip(rgb.astype(np.int16) + rng.integers(-6, 7, (h, w, 1)), 0, 255).astype(np.uint8)

    x1, y1, x2, y2 = object_box(K)
    cv2.rectangle(rgb, (x1, y1), (x2, y2), (190, 50, 45), -1)
    hx = x2 + (x2 - x1) // 5
    cv2.ellipse(rgb, (x2, (y1 + y2) // 2), ((x2 - x1) // 4, (y2 - y1) // 4), 0, -90, 90,
                (190, 50, 45), max(3, (x2 - x1) // 10))
    cv2.rectangle(rgb, (x1, y1), (x2, y1 + 4), (150, 35, 30), -1)
    depth[max(y1, 0):y2, max(x1, 0):max(hx, x2)] = OBJECT_XYZ[2]
    return rgb, depth


class FakeStream:
    """Replays one image (or a synthetic table) at a fixed rate."""

    def __init__(self, image_path: str | None = None, fps: float = 30.0):
        if image_path:
            bgr = cv2.imread(image_path)
            if bgr is None:
                raise FileNotFoundError(image_path)
            self.rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = self.rgb.shape[:2]
            self.K = make_K(w, h)
            self.depth = np.full((h, w), OBJECT_XYZ[2], np.float32)
        else:
            self.K = make_K(W, H)
            self.rgb, self.depth = synth_scene(self.K)
        self.period = 1.0 / fps
        self.t0 = time.monotonic()
        self.next_t = 0.0

    def get_frame(self) -> Frame | None:
        now = time.monotonic()
        if now < self.next_t:
            return None
        self.next_t = now + self.period
        return Frame(rgb=self.rgb, depth=self.depth, K=self.K, pose=None,
                     t=now - self.t0, t_recv=now)


class FakeLocate:
    """Object fixed in place, hand driven by the keyboard."""

    MOVES = {"left": (0, -1), "right": (0, 1), "up": (1, -1), "down": (1, 1),
             "w": (2, 1), "s": (2, -1)}

    def __init__(self, object_xyz=OBJECT_XYZ, hand_xyz=HAND_START, step: float = 0.01):
        self.object_xyz = np.array(object_xyz, float)
        self.hand_start = np.array(hand_xyz, float)
        self.hand_xyz = self.hand_start.copy()
        self.step = step
        self.occluded = False
        self.hand_hidden = False
        self.noise = False
        self.rng = np.random.default_rng()

    def reset_hand(self) -> None:
        self.hand_xyz = self.hand_start.copy()

    def handle_key(self, key: str | None) -> bool:
        if key in self.MOVES:
            axis, sign = self.MOVES[key]
            self.hand_xyz[axis] += sign * self.step
        elif key == "o":
            self.occluded = not self.occluded
        elif key == "h":
            self.hand_hidden = not self.hand_hidden
        elif key == "n":
            self.noise = not self.noise
        elif key == "r":
            self.reset_hand()
        else:
            return False
        return True

    def locate(self, frame: Frame, target: str) -> Scene:
        h, w = frame.depth.shape[:2]
        obj = self.object_xyz.copy()
        hand = self.hand_xyz.copy()
        if self.noise:
            obj += self.rng.normal(0, 0.005, 3)
            hand += self.rng.normal(0, 0.005, 3)

        hand_px = None
        if not self.hand_hidden and hand[2] > 0.05:
            u, v = project(hand, frame.K)
            if 0 <= u < w and 0 <= v < h:
                hand_px = (int(u), int(v))
        hand_visible = hand_px is not None
        has_depth = hand[2] >= LIDAR_MIN

        return Scene(
            object_xyz=obj,
            hand_xyz=hand if hand_visible and has_depth else None,
            object_visible=not self.occluded,
            hand_visible=hand_visible,
            object_box=object_box(frame.K, self.object_xyz),
            hand_px=hand_px,
            t=frame.t,
        )


class FakeGuidance:
    """Prints what lane 3's audio would say, and runs the same state machine."""

    GRAB_M = 0.05
    DEAD_M = 0.03

    def __init__(self, print_hz: float = 4.0):
        self.state = "idle"
        self.grab_t = 0.0
        self.print_every = 1.0 / print_hz
        self.last_print = 0.0

    def reset(self, target: str) -> None:
        self.state = "searching"

    def update(self, scene: Scene) -> str:
        now = time.monotonic()
        if self.state == "grab":
            if now - self.grab_t > 1.5:
                self.state = "idle"
            return self.state
        if self.state == "idle":
            return self.state

        err = hand_error(scene)
        if scene.object_xyz is None:
            self.state = "searching"
        elif scene.hand_xyz is None:
            self.state = "waiting_hand"
        elif np.linalg.norm(err) < self.GRAB_M:
            self.state, self.grab_t = "grab", now
            print("[guide] GRAB")
            return self.state
        else:
            self.state = "guiding"

        if now - self.last_print >= self.print_every:
            self.last_print = now
            if err is None:
                print(f"[guide] {self.state}")
            else:
                x, y, z = err
                words = [
                    ("right" if x > 0 else "left", x),
                    ("down" if y > 0 else "up", y),
                    ("forward" if z > 0 else "back", z),
                ]
                says = "  ".join(f"{d} {abs(v) * 100:.0f}" for d, v in words if abs(v) > self.DEAD_M)
                print(f"[guide] {self.state}  {says or 'centered'}  "
                      f"dist {np.linalg.norm(err) * 100:.0f} cm")
        return self.state


class FakeVoice:
    def __init__(self, target: str = "mug", ask: bool = False):
        self.target = target
        self.ask = ask

    def listen_for_target(self) -> str:
        if self.ask:
            return input("target> ").strip() or self.target
        return self.target
