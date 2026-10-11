"""The two agreed interfaces. Do not change them without telling everyone.

stream.py and perception.py can import these instead of redefining them:

    from interfaces import Frame, Scene

Camera frame axes: x right, y down, z forward, as seen by the wearer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# Lane 1 -> lane 2
@dataclass
class Frame:
    rgb: np.ndarray                 # H x W x 3, uint8, upright (not sideways)
    depth: np.ndarray               # H x W, float32, meters, same size as rgb
    K: np.ndarray                   # 3 x 3 intrinsics matching rgb size
    pose: Optional[np.ndarray]      # 4 x 4 camera-to-world, None if unavailable
    t: float                        # capture time, seconds (phone clock)
    t_recv: Optional[float] = None  # GX10 time.monotonic() when the frame arrived


# Lane 2 -> lane 3
@dataclass
class Scene:
    object_xyz: Optional[np.ndarray]  # meters, camera frame
    hand_xyz: Optional[np.ndarray]    # meters, camera frame
    object_visible: bool              # seen in this frame, not remembered
    hand_visible: bool
    object_box: Optional[tuple]       # x1, y1, x2, y2 for the debug view
    hand_px: Optional[tuple]          # u, v for the debug view
    t: float
    aim_xyz: Optional[np.ndarray] = None  # where the fingertip should go: the nearest part of the
                                          # object, not its middle. None means the centre, object_xyz.
    hand: Optional[object] = None         # perception.HandDetails: xyz, px, point, confidence, tracker,
                                          # depth_from, box, keypoints_px (21x2), keypoints_xyz (21x3),
                                          # palm_xyz, index_tip_xyz, nearest_m, coasting
    object_out_of_reach: bool = False     # farther than a seated wearer's arm (0.75 m)


def hand_error(scene: Scene) -> Optional[np.ndarray]:
    """Where the hand has to move, in meters, or None. Toward the nearest part of the object
    when lane 2 provides it (aim_xyz), otherwise toward its centre."""
    if scene.object_xyz is None or scene.hand_xyz is None:
        return None
    goal = scene.aim_xyz if getattr(scene, "aim_xyz", None) is not None else scene.object_xyz
    return np.asarray(goal, float) - np.asarray(scene.hand_xyz, float)

