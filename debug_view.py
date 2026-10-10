"""On-screen overlay. Judges cannot hear the earbuds, so this screen is the demo for them.

Left: camera image with object box, fingertip dot, a line between them, distance.
Right: target, state, distance, which way the hand has to move, timing, lidar depth.
"""
from __future__ import annotations

import cv2
import numpy as np

from interfaces import Frame, Scene, hand_error

WIN = "blindfold-pickup"
VIEW_H = 900
PANEL_W = 600
FONT = cv2.FONT_HERSHEY_DUPLEX
STALE_MS = 200
DEAD_M = 0.03
GRAB_M = 0.05

# BGR
BG = (28, 26, 24)
TEXT = (240, 240, 240)
DIM = (150, 150, 150)
OK = (90, 205, 90)
WARN = (0, 175, 255)
BAD = (60, 60, 230)
HAND = (255, 215, 0)
OBJ = (80, 225, 80)
MEM = (170, 170, 170)

STATE_STYLE = {
    "idle": ("IDLE", DIM),
    "searching": ("SEARCHING", WARN),
    "waiting_hand": ("SHOW YOUR HAND", WARN),
    "guiding": ("GUIDING", OK),
    "grab": ("GRAB", OK),
    "listening": ("LISTENING", WARN),
}

_last_canvas: np.ndarray | None = None


def open_window(fullscreen: bool = False) -> None:
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    if fullscreen:
        cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def close() -> None:
    cv2.destroyAllWindows()


def normalize_key(k: int) -> str | None:
    """waitKeyEx codes differ per OS; return 'left', 'q', 'esc', ... or None."""
    if k is None or k < 0:
        return None
    arrows = {0xFF51: "left", 0xFF52: "up", 0xFF53: "right", 0xFF54: "down",   # Linux
              0xF702: "left", 0xF700: "up", 0xF703: "right", 0xF701: "down"}   # macOS
    if (k & 0xFFFF) in arrows:
        return arrows[k & 0xFFFF]
    windows = {0x25: "left", 0x26: "up", 0x27: "right", 0x28: "down"}
    if (k & 0xFFFF) == 0 and (k >> 16) & 0xFF in windows:
        return windows[(k >> 16) & 0xFF]
    c = k & 0xFF
    if c == 27:
        return "esc"
    if 32 <= c < 127:
        return chr(c).lower()
    return None


def show(frame: Frame, scene: Scene, info: dict) -> str | None:
    global _last_canvas
    _last_canvas = render(frame, scene, info)
    cv2.imshow(WIN, _last_canvas)
    return normalize_key(cv2.waitKeyEx(1))


def show_message(text: str) -> str | None:
    """Dim the last screen and print a message over it (used while blocking on voice)."""
    canvas = _last_canvas.copy() if _last_canvas is not None else np.full((VIEW_H, 1200, 3), BG, np.uint8)
    canvas = (canvas * 0.35).astype(np.uint8)
    (tw, th), _ = cv2.getTextSize(text, FONT, 2.0, 3)
    _put(canvas, text, ((canvas.shape[1] - tw) // 2, (canvas.shape[0] + th) // 2), 2.0, TEXT, 3)
    cv2.imshow(WIN, canvas)
    return normalize_key(cv2.waitKeyEx(1))


def render(frame: Frame, scene: Scene, info: dict) -> np.ndarray:
    h, w = frame.rgb.shape[:2]
    s = VIEW_H / h
    size = (int(round(w * s)), VIEW_H)
    img = cv2.resize(cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR), size,
                     interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    err = hand_error(scene)
    _draw_image(img, frame, scene, info, s, err)
    panel = np.full((VIEW_H, PANEL_W, 3), BG, np.uint8)
    _draw_panel(panel, frame, scene, info, err)
    return np.hstack([img, panel])


def _put(img, text, org, scale, color, thick=1):
    cv2.putText(img, text, (int(org[0]), int(org[1])), FONT, scale, color, thick, cv2.LINE_AA)


def _label(img, text, org, scale, fg, bg, thick=1):
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thick)
    x, y = int(org[0]), int(org[1])
    cv2.rectangle(img, (x - 6, y - th - 8), (x + tw + 6, y + base + 2), bg, -1)
    _put(img, text, (x, y), scale, fg, thick)


def _draw_image(img, frame, scene, info, s, err):
    target = info.get("target") or "object"

    obj_c = None
    if scene.object_box is not None:
        x1, y1, x2, y2 = (int(round(c * s)) for c in scene.object_box)
        color = OBJ if scene.object_visible else MEM
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3 if scene.object_visible else 2, cv2.LINE_AA)
        name = target if scene.object_visible else f"{target} (remembered)"
        _label(img, name, (x1, max(y1 - 10, 30)), 0.8, (0, 0, 0), color)
        obj_c = ((x1 + x2) // 2, (y1 + y2) // 2)
    elif scene.object_xyz is not None and scene.object_xyz[2] > 0:
        x, y, z = scene.object_xyz
        u = (frame.K[0, 0] * x / z + frame.K[0, 2]) * s
        v = (frame.K[1, 1] * y / z + frame.K[1, 2]) * s
        obj_c = (int(u), int(v))
        cv2.drawMarker(img, obj_c, MEM, cv2.MARKER_CROSS, 40, 3, cv2.LINE_AA)
        _label(img, f"{target} (remembered)", (obj_c[0] + 24, obj_c[1] - 24), 0.8, (0, 0, 0), MEM)

    hand_c = None
    if scene.hand_px is not None:
        hand_c = (int(round(scene.hand_px[0] * s)), int(round(scene.hand_px[1] * s)))
        if scene.hand_xyz is not None:
            cv2.circle(img, hand_c, 13, HAND, -1, cv2.LINE_AA)
            cv2.circle(img, hand_c, 13, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.circle(img, hand_c, 13, HAND, 3, cv2.LINE_AA)
            _label(img, "no depth", (hand_c[0] + 20, hand_c[1] + 8), 0.7, (0, 0, 0), HAND)

    if obj_c is not None and hand_c is not None and err is not None:
        d = float(np.linalg.norm(err))
        color = OK if d < GRAB_M else WARN
        cv2.line(img, hand_c, obj_c, color, 3, cv2.LINE_AA)
        at = (int(hand_c[0] + 0.35 * (obj_c[0] - hand_c[0])) - 30,
              int(hand_c[1] + 0.35 * (obj_c[1] - hand_c[1])) + 52)
        _label(img, f"{d * 100:.0f} cm", at, 1.0, (0, 0, 0), color, 2)

    age = info.get("age_ms")
    if age is not None and age > STALE_MS:
        cv2.rectangle(img, (0, 0), (img.shape[1], 56), BAD, -1)
        _put(img, f"NO NEW FRAMES  {age / 1000:.1f} s", (16, 40), 1.0, TEXT, 2)


def _draw_panel(p, frame, scene, info, err):
    x = 32
    y = 56
    _put(p, "LOOKING FOR", (x, y), 0.65, DIM)
    y += 52
    _put(p, info.get("target") or "-", (x, y), 1.6, TEXT, 2)

    y += 40
    state = info.get("state")
    label, color = STATE_STYLE.get(state, (str(state).upper() if state else "-", DIM))
    (tw, th), _ = cv2.getTextSize(label, FONT, 1.1, 2)
    cv2.rectangle(p, (x, y), (x + tw + 32, y + th + 28), color, -1)
    _put(p, label, (x + 16, y + th + 14), 1.1, (0, 0, 0), 2)

    y += th + 28 + 56
    _put(p, "DISTANCE", (x, y), 0.65, DIM)
    y += 112
    if err is not None:
        d = float(np.linalg.norm(err))
        _put(p, f"{d * 100:.0f} cm", (x, y), 3.6, OK if d < GRAB_M else TEXT, 6)
    else:
        _put(p, "--", (x, y), 3.6, DIM, 6)

    y += 56
    _put(p, "MOVE HAND", (x, y), 0.65, DIM)
    rows = [("SIDE", "RIGHT", "LEFT", 0), ("HEIGHT", "DOWN", "UP", 1), ("REACH", "FORWARD", "BACK", 2)]
    for name, pos, neg, axis in rows:
        y += 44
        _put(p, name, (x, y), 0.7, DIM)
        if err is None:
            _put(p, "-", (x + 130, y), 1.0, DIM, 2)
            continue
        v = float(err[axis])
        if abs(v) <= DEAD_M:
            _put(p, "ok", (x + 130, y), 1.0, OK, 2)
        else:
            _put(p, f"{pos if v > 0 else neg}  {abs(v) * 100:.0f} cm", (x + 130, y), 1.0, TEXT, 2)

    y += 34
    cv2.line(p, (x, y), (PANEL_W - x, y), (70, 70, 70), 1)

    inset_w = 150
    inset = _depth_inset(frame.depth, inset_w)
    ih = inset.shape[0]
    iy = VIEW_H - ih - 84
    p[iy:iy + ih, PANEL_W - x - inset_w:PANEL_W - x] = inset
    _put(p, "lidar depth", (PANEL_W - x - inset_w, iy - 10), 0.5, DIM)

    y += 36
    age = info.get("age_ms")
    stats = [
        ("loop", f"{info.get('fps', 0):.1f} fps"),
        ("frame age", f"{age:.0f} ms" if age is not None else "-"),
        (info.get("e2e_label", "e2e"), f"{info['e2e_ms']:.0f} ms" if info.get("e2e_ms") is not None else "-"),
    ]
    for k, v in stats:
        _put(p, k, (x, y), 0.6, DIM)
        _put(p, v, (x + 130, y), 0.6, TEXT)
        y += 30
    st = info.get("stages") or {}
    for k in ("stream", "perception", "guidance", "view"):
        if k in st:
            _put(p, k, (x, y), 0.55, DIM)
            _put(p, f"{st[k]:.1f} ms", (x + 130, y), 0.55, TEXT)
            y += 26

    modes = info.get("modes")
    if modes:
        _put(p, modes, (x, VIEW_H - 24), 0.5, DIM)
    if info.get("keys_hint"):
        _put(p, info["keys_hint"], (x, VIEW_H - 50), 0.45, DIM)


def _depth_inset(depth: np.ndarray, width: int) -> np.ndarray:
    h, w = depth.shape[:2]
    height = int(round(width * h / w))
    d = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
    norm = (np.clip(d / 2.0, 0, 1) * 255).astype(np.uint8)
    img = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    img[(d <= 0) | ~np.isfinite(d)] = 0
    return img
