"""tools/check_pose.py on a synthetic room (floor, wall, slanted panel) seen by a moving camera."""
import numpy as np
import pytest

import stream
from stream import RawFrame
from tools.check_pose import evaluate

W, H = 256, 192
K = np.array([[210.0, 0, 127.5], [0, 210.0, 95.5], [0, 0, 1]])
PLANES = [((0, 1, 0), 1.0), ((0, 0, 1), 3.0), ((0.7071, 0, 0.7071), 2.5)]   # y is down


def _camera(i, still=False):
    """Camera-to-world in our axes: looking 25 degrees down, drifting and tilting with i."""
    k = 0.0 if still else 1.0
    a, b = np.radians(-25) + k * 0.12 * np.sin(0.7 * i), k * 0.15 * np.sin(0.5 * i)
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    T = np.eye(4)
    T[:3, :3] = Ry @ Rx
    T[:3, 3] = k * np.array([0.15 * np.sin(0.4 * i), 0.08 * np.sin(0.9 * i), 0.10 * np.sin(0.3 * i)])
    return T


def _depth(T):
    v, u = np.mgrid[0:H, 0:W]
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones((H, W))], axis=-1) @ T[:3, :3].T
    depth = np.full((H, W), np.inf)
    for n, c in PLANES:
        n = np.array(n)
        s = (c - n @ T[:3, 3]) / (rays @ n)
        depth = np.where((s > 0) & (s < depth), s, depth)
    return np.where(np.isfinite(depth), depth, 0).astype(np.float32)


def _session(fix, n=24, still=False, inverted=False, kind="record3d", pose=True):
    """What an app whose pose uses the convention `fix` would send over n frames, 0.5 s apart."""
    raws = []
    for i in range(n):
        T = _camera(i, still)
        dev = T.copy()
        dev[:3, :3] = T[:3, :3] @ fix.T
        if inverted:
            dev = np.linalg.inv(dev)
        raws.append(RawFrame(np.zeros((H, W, 3), np.uint8), _depth(T), K.copy(), dev if pose else None, kind))
    return raws, [0.5 * i for i in range(n)]


@pytest.mark.parametrize("kind", ["record3d", "pathon"])
def test_finds_the_convention_stream_already_uses(kind):
    r = evaluate(*_session(stream.POSE_FIX[kind], kind=kind))
    assert r["verdict"] == "yes" and r["matches_stream"]
    assert r["best"]["residual"] < 0.01 < 0.03 < r["ignoring_pose"]
    assert r["ranked"][1][0] > 3 * r["ranked"][0][0]        # the runner-up is clearly worse


def test_reports_a_different_convention_with_its_matrix():
    fix = np.diag([1.0, -1.0, -1.0]) @ stream._ROT[270]
    r = evaluate(*_session(fix))
    assert r["verdict"] == "yes" and not r["matches_stream"]
    assert np.allclose(r["best"]["fix"], fix) and not r["best"]["inverted"]


def test_notices_a_world_to_camera_pose():
    r = evaluate(*_session(stream.POSE_FIX["record3d"], inverted=True))
    assert r["verdict"] == "yes" and r["best"]["inverted"] and not r["matches_stream"]


def test_still_phone_is_unclear_not_yes():
    r = evaluate(*_session(stream.POSE_FIX["record3d"], still=True))
    assert r["verdict"] == "unclear" and "barely moved" in r["why"]


def test_no_pose_is_a_no():
    r = evaluate(*_session(np.eye(3), pose=False))
    assert r["verdict"] == "no" and r["with_pose"] == 0


def test_wrong_poses_are_a_no():
    raws, times = _session(stream.POSE_FIX["record3d"])
    for i, raw in enumerate(raws):          # poses that have nothing to do with the real motion
        raw.pose = _camera(7 * i + 3)
    assert evaluate(raws, times)["verdict"] == "no"


def test_covered_lens_is_unclear_not_a_no():
    raws, times = _session(stream.POSE_FIX["record3d"])
    for raw in raws:
        raw.depth[:] = 0.0                    # phone face down on the table: no depth at all
    r = evaluate(raws, times)
    assert r["verdict"] == "unclear" and "valid depth" in r["why"]
