"""Lane 2: pixels to metres, remembering the object while the arm covers it, choosing and
smoothing the hand. The detector and hand tracker are faked; the real models are exercised by
tools/check_perception.py."""
import numpy as np
import pytest

import perception
from fakes import make_K, object_box, project
from interfaces import Frame
from perception import Detection, Hand, Perception, PerceptionConfig, StickerHand, box_xyz, to_xyz

W, H = 480, 640
K = make_K(W, H)
OBJ = np.array([0.05, 0.12, 0.60])
HAND = np.array([-0.15, 0.05, 0.40])
TABLE = 0.90


def frame(t=0.0, obj=OBJ, hands=(), pose=None):
    """A flat background at TABLE metres, the object's box at its depth, and a disc per hand."""
    depth = np.full((H, W), TABLE, np.float32)
    rgb = np.full((H, W, 3), 90, np.uint8)
    if obj is not None:
        x1, y1, x2, y2 = object_box(K, obj)
        depth[max(y1, 0):y2, max(x1, 0):x2] = obj[2]
    vv, uu = np.mgrid[0:H, 0:W]
    for hand in hands:
        u, v = project(hand, K)
        disc = (uu - u) ** 2 + (vv - v) ** 2 <= 14 ** 2
        depth[disc] = hand[2]
        rgb[disc] = (40, 220, 60)        # a green sticker
    return Frame(rgb, depth, K, pose, t, t_recv=t)


class FakeDetector:
    def __init__(self, obj=OBJ):
        self.obj, self.calls, self.targets = obj, 0, []

    def set_target(self, target):
        self.targets.append(target)

    def detect(self, rgb):
        self.calls += 1
        return [] if self.obj is None else [Detection(object_box(K, self.obj), 0.9, "mug")]


class FakeHands:
    def __init__(self, *hands):
        self.hands = list(hands)

    def detect(self, rgb, t_ms):
        out = []
        for h in self.hands:
            u, v = project(h, K)
            tip = (u, v)
            out.append(Hand(tip=tip, points=np.array([tip], np.float32), samples=[tip, (u, v + 12)]))
        return out


def make(obj=OBJ, hands=(HAND,), **cfg):
    cfg.setdefault("smooth", 1)
    return Perception(FakeDetector(obj), FakeHands(*hands), PerceptionConfig(**cfg))


# --- pixels to metres ----------------------------------------------------------

def test_to_xyz_uses_the_median_and_ignores_holes():
    depth = np.full((H, W), 0.8, np.float32)
    depth[300:320, 200:220] = 0.0                       # a hole right at the pixel
    depth[296:300, 196:224] = 0.5
    u, v = 210, 302
    p = to_xyz(u, v, depth, K, r=6)
    assert p[2] == pytest.approx(0.5)
    assert p[0] == pytest.approx((u - K[0, 2]) * 0.5 / K[0, 0]) and p[1] == pytest.approx((v - K[1, 2]) * 0.5 / K[1, 1])
    assert to_xyz(210, 310, depth, K, r=2) is None      # nothing valid nearby
    assert to_xyz(-5, 10, depth, K) is None and to_xyz(10, H + 3, depth, K) is None


def test_to_xyz_nearest_keeps_a_thin_finger_in_front_of_the_table():
    depth = np.full((H, W), 0.8, np.float32)
    depth[:, 238:243] = 0.4                             # a finger 5 px wide in a 13 px patch
    assert to_xyz(240, 300, depth, K, r=6)[2] == pytest.approx(0.8)             # median: the table wins
    assert to_xyz(240, 300, depth, K, r=6, nearest=True)[2] == pytest.approx(0.4)


def test_readings_outside_the_lidar_range_are_ignored():
    depth = np.full((H, W), 0.10, np.float32)           # closer than lidar can measure
    assert to_xyz(240, 300, depth, K) is None
    depth[:] = 6.0
    assert to_xyz(240, 300, depth, K) is None


def test_box_xyz_samples_the_centre_third():
    depth = np.full((H, W), TABLE, np.float32)
    box = (200, 300, 290, 420)
    depth[340:380, 230:260] = 0.55                      # only the middle of the box is the object
    p = box_xyz(box, depth, K)
    assert p[2] == pytest.approx(0.55)
    assert p[0] == pytest.approx((245 - K[0, 2]) * 0.55 / K[0, 0])
    assert box_xyz(box, np.zeros((H, W), np.float32), K) is None


# --- locate --------------------------------------------------------------------

def test_object_and_hand_come_out_in_metres():
    s = make().locate(frame(hands=[HAND]), "mug")
    assert s.object_visible and s.hand_visible
    assert np.allclose(s.object_xyz, OBJ, atol=0.004) and np.allclose(s.hand_xyz, HAND, atol=0.004)
    assert np.linalg.norm(s.object_xyz - s.hand_xyz) == pytest.approx(np.linalg.norm(OBJ - HAND), abs=0.005)
    assert s.object_box == object_box(K, OBJ) and s.hand_px == tuple(int(round(v)) for v in project(HAND, K))


def test_nothing_found_gives_none():
    s = make(obj=None, hands=()).locate(frame(obj=None), "mug")
    assert s.object_xyz is None and s.hand_xyz is None and not s.object_visible and not s.hand_visible
    assert s.object_box is None and s.hand_px is None


def test_new_target_resets_memory_and_tells_the_detector():
    p = make(hands=())
    assert p.locate(frame(t=0.0), "mug").object_xyz is not None
    p.detector.obj = None
    s = p.locate(frame(t=0.1, obj=None), "bottle")
    assert p.detector.targets == ["mug", "bottle"] and s.object_xyz is None


def test_hand_over_the_object_does_not_drag_the_object_to_the_hand():
    over = OBJ + np.array([0.0, 0.0, -0.12])            # fingertip 12 cm in front of the object
    p = make(hands=())
    p.locate(frame(t=0.0), "mug")
    p.hands = FakeHands(over)
    s = p.locate(frame(t=0.1, hands=[over]), "mug")
    assert s.object_xyz[2] == pytest.approx(OBJ[2], abs=0.004)      # remembered depth, not the hand's
    assert s.hand_xyz[2] == pytest.approx(over[2], abs=0.004)
    assert s.object_visible                                           # the detector still sees it


def test_covered_object_still_follows_the_box_sideways():
    over = OBJ + np.array([0.0, 0.0, -0.12])
    p = make(hands=())
    p.locate(frame(t=0.0), "mug")
    p.hands = FakeHands(over + np.array([0.03, 0, 0]))
    moved = OBJ + np.array([0.03, 0.0, 0.0])
    p.detector.obj = moved
    s = p.locate(frame(t=0.1, obj=moved, hands=[over + np.array([0.03, 0, 0])]), "mug")
    assert s.object_xyz[0] == pytest.approx(moved[0], abs=0.004) and s.object_xyz[2] == pytest.approx(OBJ[2], abs=0.004)


def test_hidden_object_is_remembered_for_two_seconds_without_pose():
    p = make(hands=())
    p.locate(frame(t=0.0), "mug")
    p.detector.obj = None                                # the arm hides it completely
    s = p.locate(frame(t=1.9, obj=None), "mug")
    assert np.allclose(s.object_xyz, OBJ, atol=0.004) and not s.object_visible
    s = p.locate(frame(t=2.1, obj=None), "mug")
    assert s.object_xyz is None and s.object_box is None


def test_hidden_by_the_hand_is_remembered_a_little_longer():
    over = OBJ + np.array([0.0, 0.0, -0.10])
    p = make(hands=())
    p.locate(frame(t=0.0), "mug")
    p.detector.obj = None
    p.hands = FakeHands(over)
    assert p.locate(frame(t=3.0, obj=None, hands=[over]), "mug").object_xyz is not None
    assert p.locate(frame(t=4.1, obj=None, hands=[over]), "mug").object_xyz is None


def _pose(tx=0.0, yaw=0.0):
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    T[0, 3] = tx
    return T


def test_with_pose_the_remembered_object_stays_put_when_the_wearer_moves():
    p = make(hands=(), use_pose=True)
    p.locate(frame(t=0.0, pose=_pose()), "mug")
    p.detector.obj = None
    moved = _pose(tx=0.10, yaw=0.1)                      # wearer steps 10 cm right and turns a little
    s = p.locate(frame(t=5.0, obj=None, pose=moved), "mug")
    expected = moved[:3, :3].T @ (OBJ - moved[:3, 3])
    assert np.allclose(s.object_xyz, expected, atol=0.004) and not np.allclose(s.object_xyz, OBJ, atol=0.02)
    assert p.locate(frame(t=16.0, obj=None, pose=moved), "mug").object_xyz is None   # not forever


def test_pose_is_ignored_unless_switched_on():
    p = make(hands=())
    p.locate(frame(t=0.0, pose=_pose()), "mug")
    p.detector.obj = None
    s = p.locate(frame(t=1.0, obj=None, pose=_pose(tx=0.10)), "mug")
    assert np.allclose(s.object_xyz, OBJ, atol=0.004)


def test_losing_the_pose_drops_the_world_anchor():
    p = make(hands=(), use_pose=True)
    p.locate(frame(t=0.0, pose=_pose(tx=0.3)), "mug")
    p.detector.obj = None
    assert p.locate(frame(t=0.1, obj=None, pose=None), "mug").object_xyz is None


def test_with_two_hands_the_one_nearer_the_object_is_used():
    far = np.array([-0.20, -0.10, 0.45])
    close = np.array([0.02, 0.02, 0.50])
    s = make(hands=(far, close)).locate(frame(hands=[far, close]), "mug")
    assert np.allclose(s.hand_xyz, close, atol=0.004)


def test_fingertip_without_depth_uses_the_next_point_down_the_finger():
    f = frame(hands=[HAND])
    u, v = (int(round(x)) for x in project(HAND, K))
    f.depth[v - 6:v + 7, u - 6:u + 7] = 0.0              # no lidar return at the tip
    s = make().locate(f, "mug")
    assert s.hand_visible and s.hand_xyz[2] == pytest.approx(HAND[2], abs=0.004)   # depth from further down
    assert s.hand_px == (u, v)                           # still the tip's pixel
    assert s.hand_xyz[0] == pytest.approx(HAND[0], abs=0.004)


def test_hand_seen_but_too_close_for_lidar_has_no_position():
    near = np.array([0.0, 0.05, 0.10])
    f = frame(hands=[near])
    f.depth[:] = 0.0
    s = make(hands=(near,)).locate(f, "mug")
    assert s.hand_visible and s.hand_xyz is None


def test_positions_are_averaged_but_a_jump_is_not_blended():
    p = make(smooth=3)
    for i, dx in enumerate((0.00, 0.01, 0.02)):
        h = HAND + np.array([dx, 0, 0])
        p.hands = FakeHands(h)
        s = p.locate(frame(t=0.03 * i, hands=[h]), "mug")
    assert s.hand_xyz[0] == pytest.approx(HAND[0] + 0.01, abs=0.004)     # mean of the three
    h = HAND + np.array([0.25, 0, 0])
    p.hands = FakeHands(h)
    s = p.locate(frame(t=0.1, hands=[h]), "mug")
    assert s.hand_xyz[0] == pytest.approx(h[0], abs=0.004)               # a real move, not smeared


def test_detector_can_run_on_every_nth_frame():
    p = make(hands=(), detect_every=3)
    for i in range(9):
        s = p.locate(frame(t=0.03 * i), "mug")
    assert p.detector.calls == 3 and s.object_visible
    p.detector.obj = None
    for i in range(9, 14):
        p.locate(frame(t=0.03 * i, obj=None), "mug")
    assert p.detector.calls >= 6                         # once lost, it looks on every frame


def test_sticker_tracker_finds_the_green_blob():
    f = frame(hands=[HAND])
    hands = StickerHand().detect(f.rgb)
    assert len(hands) == 1
    assert np.allclose(hands[0].tip, project(HAND, K), atol=1.5)
    assert StickerHand().detect(frame().rgb) == []


def test_module_locate_runs_without_any_model(monkeypatch):
    monkeypatch.setenv("PERCEPTION_DETECTOR", "none")
    monkeypatch.setenv("PERCEPTION_HAND", "sticker")
    monkeypatch.setattr(perception, "_default", None)
    s = perception.locate(frame(hands=[HAND]), "mug")
    assert s.object_xyz is None and np.allclose(s.hand_xyz, HAND, atol=0.004)


def test_object_first_seen_with_the_hand_already_on_it_still_gets_a_position():
    over = OBJ + np.array([0.0, 0.0, -0.08])            # fingertip resting 8 cm in front of the object centre
    p = make(hands=(over,))
    s = p.locate(frame(t=0.0, hands=[over]), "mug")     # no earlier frame without the hand
    assert s.object_visible and s.object_xyz is not None
    assert s.object_xyz[2] == pytest.approx(OBJ[2], abs=0.01)       # the object's depth, not the hand's
    assert s.hand_xyz[2] == pytest.approx(over[2], abs=0.004)
    assert np.linalg.norm(s.object_xyz - s.hand_xyz) == pytest.approx(0.08, abs=0.015)


def test_hand_mask_covers_the_hand_and_little_else():
    pts = np.array([[200, 300], [260, 300], [260, 420], [200, 420]], np.float32)
    m = perception.hand_mask((H, W), pts)
    assert m[360, 230] == 1 and m[300, 190] == 1         # inside, and just outside the hull (finger width)
    assert m[360, 400] == 0 and m[100, 230] == 0
    one = perception.hand_mask((H, W), np.array([[240, 320]], np.float32))
    assert one[320, 240] == 1 and one[320, 258] == 1 and one[320, 300] == 0
