"""The hand options: metres from keypoints, one continuous depth, whose hand it is, which
point is guided, smoothing, switching while running, and the object staying put when lidar
cannot see it. Models are faked; tools/eval_hands.py runs the real ones on a recording."""
import math
import time

import numpy as np
import pytest

import hands
import perception
from fakes import make_K, object_box, project
from hands import Hand, HandDepth, Kalman, OneEuro
from interfaces import Frame
from perception import Detection, Perception, PerceptionConfig, box_xyz, surface_depth

W, H = 480, 640
K = make_K(W, H)
OBJ = np.array([0.05, 0.12, 0.60])
TABLE = 0.90
FUSED = dict(hand_depth="fused", hand_filter="one_euro", own_arm=True)


def flat_hand() -> np.ndarray:
    """21 keypoints of an open hand lying in its own plane, metres: x across the knuckles, y toward the fingers."""
    pts = np.zeros((21, 3))
    for i, xy in hands._PALM_XY.items():
        pts[i, :2] = xy
    pts[1, :2] = (-0.03, 0.03)
    for chain, bones in hands.FINGERS:
        a = chain[0]
        d = np.array([-0.7, 0.7, 0.0]) if a == 1 else pts[a] / np.linalg.norm(pts[a])
        for prev, nxt, length in zip(chain, chain[1:], bones):
            pts[nxt] = pts[prev] + d * length
    return pts


def place(at, tilt_deg=0.0, turn_deg=0.0, size=1.0) -> np.ndarray:
    """The hand in the camera frame with its palm's middle at `at`: fingers up in the picture,
    tilted away from the camera by tilt_deg, turned in the picture by turn_deg."""
    pts = flat_hand() * size
    pts[:, 1] *= -1                                     # camera y points down; fingers point up
    t, r = math.radians(tilt_deg), math.radians(turn_deg)
    tilt = np.array([[1, 0, 0], [0, math.cos(t), math.sin(t)], [0, -math.sin(t), math.cos(t)]])
    turn = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    pts = pts @ tilt.T @ turn.T
    return pts - pts[list(hands.PALM)].mean(0) + np.asarray(at, float)


def seen(xyz: np.ndarray, source="fake") -> Hand:
    kp = np.array([project(p, K) for p in xyz], np.float32)
    return Hand(tip=tuple(kp[8]), points=kp, samples=[tuple(kp[i]) for i in (8, 7, 6, 5)], score=0.9,
                px_per_m=hands._size_scale(kp), kp=kp, box=(*kp.min(0), *kp.max(0)), source=source)


def frame(t=0.0, obj=OBJ, hand_xyz=(), lidar=True, rgb=90):
    """A table at TABLE metres, the object's box at its depth, and each hand's palm and fingers at theirs."""
    depth = np.full((H, W), TABLE, np.float32)
    img = np.full((H, W, 3), rgb, np.uint8)
    if obj is not None:
        x1, y1, x2, y2 = object_box(K, obj)
        depth[max(y1, 0):y2, max(x1, 0):x2] = obj[2]
    if lidar:
        import cv2
        for xyz in hand_xyz:
            kp = np.array([project(p, K) for p in xyz], np.int32)
            mask = np.zeros((H, W), np.uint8)
            cv2.fillConvexPoly(mask, cv2.convexHull(kp), 1)
            depth[mask > 0] = float(xyz[list(hands.PALM), 2].mean())
            img[mask > 0] = (200, 150, 120)              # skin
    return Frame(img, depth, K, None, t, t_recv=t)


class FakeDetector:
    def __init__(self, obj=OBJ):
        self.obj = obj

    def set_target(self, target):
        pass

    def detect(self, rgb):
        return [] if self.obj is None else [Detection(object_box(K, self.obj), 0.9, "mug")]


class FakeHands:
    name = "fake"

    def __init__(self, *xyz):
        self.xyz = list(xyz)

    def detect(self, rgb, t_ms, depth=None, K=None):
        return [seen(x) for x in self.xyz]


def make(*hand_xyz, obj=OBJ, **cfg):
    cfg = dict(FUSED, **cfg)
    return Perception(FakeDetector(obj), FakeHands(*hand_xyz), PerceptionConfig(**cfg))


# --- metres from keypoints -------------------------------------------------------

@pytest.mark.parametrize("tilt", [0, 30, 55])
@pytest.mark.parametrize("turn", [0, 90, 200])
def test_distance_from_keypoints_survives_a_tilted_or_turned_hand(tilt, turn):
    xyz = place((0.03, 0.02, 0.45), tilt, turn)
    pts = hands.hand_points_3d(seen(xyz), K)
    z = pts[list(hands.PALM), 2].mean()
    assert z == pytest.approx(0.45, rel=0.12)


def test_the_old_two_span_size_is_a_quarter_off_when_the_knuckle_row_faces_the_camera():
    # This is why lidar and size disagreed in the live run: 8 cm was assumed for a 6 cm knuckle row.
    xyz = place((0.0, 0.0, 0.45), tilt_deg=70)
    old = K[0, 0] / seen(xyz).px_per_m
    assert old / 0.45 > 1.2
    assert hands.hand_points_3d(seen(xyz), K)[list(hands.PALM), 2].mean() == pytest.approx(0.45, rel=0.12)


def test_fingers_pointing_away_come_out_farther_than_the_palm():
    xyz = place((0.0, 0.0, 0.45), tilt_deg=50)
    pts = hands.hand_points_3d(seen(xyz), K)
    palm = pts[list(hands.PALM), 2].mean()
    assert 0.03 < pts[8, 2] - palm < 0.12 and pts[0, 2] < palm
    flat = hands.hand_points_3d(seen(place((0.0, 0.0, 0.45))), K)
    assert np.ptp(flat[:, 2]) < 0.03                    # a hand square to the camera is all at one depth


def test_palm_depth_is_the_hand_not_what_is_behind_a_finger():
    xyz = place((0.0, 0.0, 0.40), tilt_deg=20)
    f = frame(hand_xyz=[xyz])
    z, share = hands.palm_depth(seen(xyz), f.depth)
    assert z == pytest.approx(xyz[list(hands.PALM), 2].mean(), abs=0.005) and share > 0.9
    f.depth[:] = 0.0                                    # closer than lidar can range: no reading, not a wrong one
    assert hands.palm_depth(seen(xyz), f.depth) == (None, 0.0)


# --- one depth out of two ----------------------------------------------------------

def test_depth_does_not_jump_when_lidar_goes_and_comes_back():
    hd, dt = HandDepth(), 1 / 30
    out = []
    for i in range(90):
        lidar = 0.45 if i < 30 or i >= 60 else None      # lidar drops out for a second
        out.append(hd.fuse(lidar, 0.40, i * dt)[0])      # size says 5 cm nearer than lidar
    steps = np.abs(np.diff(out))
    assert out[29] == pytest.approx(0.45, abs=0.002)     # with lidar: lidar
    assert steps.max() < 0.01                            # never a centimetre in one frame
    assert out[59] == pytest.approx(0.40, abs=0.012)     # without it, it has slid to the size estimate
    assert out[89] == pytest.approx(0.45, abs=0.005)     # and back


def test_lidar_reading_past_the_hand_is_not_believed():
    hd = HandDepth()
    assert hd.fuse(0.40, 0.40, 0.0)[0] == pytest.approx(0.40)
    z, share = hd.fuse(1.10, 0.40, 1 / 30)               # the wall between the fingers
    assert z == pytest.approx(0.40, abs=0.01) and share < 1.0
    assert not hd.agrees(0.18, 0.18)                     # nearer than lidar ranges reliably


def test_the_size_estimate_is_fitted_to_this_wearers_hand():
    hd = HandDepth()
    for i in range(12):                                  # a small hand: looks 20% farther than it is
        hd.learn(0.40, 0.50)
    assert hd.scale == pytest.approx(0.8) and hd.sized(0.50) == pytest.approx(0.40)
    hd.learn(0.40, 0.10)                                 # a nonsense pair is not learnt from
    assert hd.scale == pytest.approx(0.8)
    assert hd.fuse(None, 0.50, 0.0)[0] == pytest.approx(0.40)


def test_without_keypoints_the_depth_is_plain_lidar():
    hd = HandDepth()
    assert hd.fuse(0.33, None, 0.0) == (0.33, 1.0)
    assert hd.fuse(None, None, 0.1) == (None, 0.0)


# --- whose hand ----------------------------------------------------------------------

def test_an_arm_hanging_from_the_top_of_the_picture_is_not_the_wearers():
    ours = seen(place((0.0, 0.05, 0.45), tilt_deg=30))
    theirs = seen(place((0.0, -0.05, 0.45), tilt_deg=30, turn_deg=180))     # fingers down, wrist above
    side = seen(place((0.0, 0.0, 0.45), tilt_deg=30, turn_deg=-90))         # in from the right edge
    raised = seen(place((0.0, 0.0, 0.45), tilt_deg=30, turn_deg=-115))      # from the right, arm raised
    assert not hands.arm_from_above(ours.kp) and not hands.arm_from_above(side.kp)
    assert not hands.arm_from_above(raised.kp) and hands.arm_from_above(theirs.kp)


def test_the_bystander_is_rejected_and_the_wearer_followed():
    ours = place((-0.10, 0.08, 0.40), tilt_deg=30)
    far = place((0.10, -0.10, 1.20), tilt_deg=10)                        # across the table
    reaching = place((0.05, 0.02, 0.55), tilt_deg=20, turn_deg=180)      # handing the object over, from above
    f = frame(hand_xyz=[ours, far, reaching])
    p = make(far, reaching, ours)
    s = p.locate(f, "mug")
    assert s.hand_visible and np.allclose(s.hand_xyz[:2], ours[8, :2], atol=0.02)
    p.hands.xyz = [far, reaching]
    s = p.locate(frame(t=0.1, hand_xyz=[far, reaching]), "mug")
    assert not s.hand_visible and s.hand_xyz is None and s.hand is None


def test_a_far_hand_is_rejected_by_size_alone_when_lidar_has_nothing():
    far = place((0.05, -0.05, 1.20))
    p = make(far)
    f = frame(lidar=False)
    f.depth[:] = 0.0
    assert not p.locate(f, "mug").hand_visible
    near = place((0.05, 0.05, 0.18))                      # ours, closer than lidar ranges
    p.hands.xyz = [near]
    f2 = frame(t=0.1, lidar=False)
    f2.depth[:] = 0.0
    s = p.locate(f2, "mug")
    assert s.hand_visible and s.hand_xyz[2] == pytest.approx(0.18, abs=0.04) and s.hand.depth_from == "size"


def test_it_stays_with_the_hand_it_was_following():
    a, b = place((-0.12, 0.10, 0.40)), place((0.05, 0.10, 0.58))         # b is the one nearer the object
    p = make(a)
    p.locate(frame(hand_xyz=[a]), "mug")
    p.hands.xyz = [b, a]
    s = p.locate(frame(t=0.1, hand_xyz=[a, b]), "mug")
    assert np.allclose(s.hand_xyz[:2], a[8, :2], atol=0.02)


# --- which point, and what the grab rule is told ---------------------------------------

@pytest.mark.parametrize("point", hands.POINTS)
def test_each_hand_point_is_where_it_says(point):
    xyz = place((-0.10, 0.08, 0.42), tilt_deg=35)
    p = make(xyz, hand_point=point)
    s = p.locate(frame(hand_xyz=[xyz]), "mug")
    want = {"index_tip": xyz[8], "palm": xyz[list(hands.PALM)].mean(0), "knuckles": xyz[list(hands.KNUCKLES)].mean(0)}
    if point in want:
        assert np.allclose(s.hand_xyz, want[point], atol=0.035)
        assert np.allclose(s.hand_px, project(want[point], K), atol=2)
    else:                                                 # nearest: one of the fingertips or knuckles
        d = np.linalg.norm(xyz[list(hands.TIPS + hands.KNUCKLES)] - s.hand_xyz, axis=1)
        assert d.min() < 0.035
    assert s.hand.point == point


def test_the_scene_carries_the_whole_hand_for_the_grab_rule():
    xyz = place((0.02, 0.10, 0.52), tilt_deg=30)
    p = make(xyz)
    s = p.locate(frame(hand_xyz=[xyz]), "mug")
    d = s.hand
    assert isinstance(d, perception.HandDetails)
    assert d.keypoints_xyz.shape == (21, 3) and d.keypoints_px.shape == (21, 2) and len(d.box) == 4
    assert np.allclose(d.keypoints_xyz[list(hands.PALM)].mean(0), d.palm_xyz)
    assert np.allclose(d.index_tip_xyz, xyz[8], atol=0.035) and 0 < d.confidence <= 1
    assert d.tracker == "fake" and d.depth_from == "lidar" and not d.coasting
    assert d.nearest_m is not None and d.nearest_m <= np.linalg.norm(s.aim_xyz - s.hand_xyz) + 1e-9
    assert np.allclose(d.xyz, s.hand_xyz)


def test_option_one_is_untouched_and_still_reports_the_hand():
    xyz = place((-0.10, 0.08, 0.40))
    p = Perception(FakeDetector(), FakeHands(xyz), PerceptionConfig(smooth=1))
    s = p.locate(frame(hand_xyz=[xyz]), "mug")
    assert s.hand_px == tuple(int(round(v)) for v in project(xyz[8], K))
    assert s.hand_xyz[2] == pytest.approx(xyz[list(hands.PALM), 2].mean(), abs=0.01)     # lidar under the fingertip
    assert s.hand.point == "index_tip" and s.hand.keypoints_xyz is None and s.hand.depth_from == "lidar"


# --- smoothing -------------------------------------------------------------------------

def reach(n=60, fps=30.0, noise=0.004, seed=0):
    """A hand held still for a second, then a 40 cm reach in a second (smooth start and stop), with tracker jitter."""
    rng = np.random.default_rng(seed)
    t = np.arange(n * 2) / fps
    u = np.clip((t - 1.0) / 1.0, 0, 1)
    x = 0.40 * (10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5)
    true = np.stack([x, np.zeros_like(x), 0.3 + 0 * x], 1)
    return t, true, true + rng.normal(0, noise, true.shape)


@pytest.mark.parametrize("make_filter", [OneEuro, Kalman])
def test_filters_calm_a_still_hand_without_falling_behind_a_moving_one(make_filter):
    t, true, noisy = reach()
    f = make_filter()
    out = np.array([f.add(p, ti) for p, ti in zip(noisy, t)])
    still = slice(10, 30)
    assert np.linalg.norm(np.diff(out[still], axis=0), axis=1).mean() < 0.6 * np.linalg.norm(
        np.diff(noisy[still], axis=0), axis=1).mean()
    moving = (t > 1.2) & (t < 1.8)
    lag_m = np.abs(out[moving, 0] - true[moving, 0]).mean()
    speed = np.abs(np.gradient(true[:, 0], t))[moving].mean()
    assert lag_m / speed < 0.05                           # less than 50 ms behind


def test_filters_start_again_after_a_gap():
    for f in (OneEuro(), Kalman()):
        f.add(np.array([0.0, 0.0, 0.3]), 0.0)
        assert np.allclose(f.add(np.array([0.3, 0.0, 0.5]), 2.0), [0.3, 0.0, 0.5])


def test_kalman_carries_on_along_the_path_for_a_few_frames():
    k = Kalman()
    for i in range(20):
        k.add(np.array([0.02 * i, 0.0, 0.4]), i / 30)                    # 0.6 m/s to the right
    ahead = k.ahead(19 / 30 + 0.1)
    assert ahead[0] == pytest.approx(0.02 * 19 + 0.06, abs=0.015)
    assert k.ahead(19 / 30 + 1.0) is None                                # a second on, it would be a guess


def test_a_blink_of_the_tracker_is_coasted_through_only_when_asked():
    xyz = place((-0.10, 0.08, 0.40))
    for coast, expect in ((0.25, True), (0.0, False)):
        p = make(xyz, hand_filter="kalman", hand_coast_s=coast)
        for i in range(5):
            p.locate(frame(t=i / 30, hand_xyz=[xyz]), "mug")
        p.hands.xyz = []
        s = p.locate(frame(t=5 / 30), "mug")
        assert (s.hand_xyz is not None) == expect and not s.hand_visible
        if expect:
            assert s.hand.coasting and np.allclose(s.hand_xyz, xyz[8], atol=0.04)
            assert p.locate(frame(t=1.0), "mug").hand_xyz is None        # not for long


# --- trackers that need no model ----------------------------------------------------------

def arm_scene(tip_v=300, z=0.35):
    """An arm coming up from the bottom edge over a far table, in depth and in skin colour."""
    depth = np.full((960, 720), 0.9, np.float32)
    rgb = np.full((960, 720, 3), 60, np.uint8)
    depth[tip_v:, 300:400] = z
    rgb[tip_v:, 300:400] = (200, 150, 120)
    return rgb, depth


def test_depth_arm_finds_the_far_end_of_the_arm_and_nothing_without_lidar():
    rgb, depth = arm_scene()
    arm = hands.DepthArm()
    (hand,) = arm.detect(rgb, 0, depth=depth)
    assert abs(hand.tip[1] - 300) < 12 and 300 <= hand.tip[0] <= 400 and hand.z == pytest.approx(0.35)
    assert hand.ours and hand.kp is None
    assert arm.detect(rgb, 0, depth=np.zeros_like(depth)) == [] and arm.detect(rgb, 0) == []
    table = np.full((960, 720), 2.0, np.float32)
    table[300:, :] = np.linspace(1.2, 0.3, 660)[:, None]                  # a table top running on out of reach
    assert arm.detect(rgb, 0, depth=table) == []


class Blinking:
    """A keypoint model that sees the hand only while `on`."""
    name = "blinking"

    def __init__(self):
        self.on, self.hand = True, None

    def detect(self, rgb, t_ms, depth=None, K=None):
        return [self.hand] if self.on else []


def test_hybrid_keeps_hold_of_the_hand_in_the_depth_picture_when_the_model_loses_it():
    model = Blinking()
    hy = hands.Hybrid(model, bridge_s=1.0)
    rgb, depth = arm_scene(tip_v=300)
    model.hand = Hand(tip=(350.0, 310.0), points=np.array([[350, 310]], np.float32), samples=[(350.0, 310.0)],
                      box=(300.0, 300.0, 400.0, 420.0), kp=np.zeros((21, 2), np.float32), source="blinking")
    assert hy.detect(rgb, 0, depth=depth) == [model.hand]
    hy.chosen(model.hand, 0, 0.35)
    model.on = False
    rgb2, depth2 = arm_scene(tip_v=260)                                   # the hand has moved up 40 px
    (kept,) = hy.detect(rgb2, 100, depth=depth2)
    assert kept.source == "Depth follow" and kept.ours and kept.z == pytest.approx(0.35)
    assert 250 < kept.tip[1] < 310                                        # it followed the hand up
    gone = np.full_like(depth2, 0.9)                                      # the arm has left
    assert hy.detect(np.full_like(rgb2, 60), 200, depth=gone) == []
    hy.chosen(model.hand, 0, 0.35)
    assert hy.detect(rgb2, 5000, depth=depth2) == []                      # too long ago to carry on from


# --- switching while running ----------------------------------------------------------------

@pytest.fixture
def running(monkeypatch):
    """perception with a default instance and fake models, as in the middle of a run."""
    made = []

    def fake_make(kind, device=None):
        if kind == "yolo_medium":
            raise RuntimeError("weights missing")
        made.append(kind)
        t = FakeHands(place((-0.10, 0.08, 0.40)))
        t.name = kind
        return t

    monkeypatch.setattr(hands, "make", fake_make)
    monkeypatch.setattr(perception, "_hand_trackers", {})
    monkeypatch.setattr(perception, "_hand_number", None)
    monkeypatch.setattr(perception, "_default", Perception(FakeDetector(), FakeHands(), PerceptionConfig()))
    monkeypatch.delenv("HAND_OPTION", raising=False)
    monkeypatch.delenv("HAND_POINT", raising=False)
    monkeypatch.delenv("HAND_FILTER", raising=False)
    return made


def settle(p, timeout=3.0):
    t = time.time()
    while p._next_hands is None and time.time() - t < timeout:
        time.sleep(0.01)
    return p.locate(frame(), "mug")


def test_hand_options_follow_the_same_pattern_as_guidance_options(running):
    assert perception.hand_option() == (1, "mediapipe", "MediaPipe")
    assert len(perception.HAND_OPTIONS) >= 5 and all(len(o) == 3 for o in perception.HAND_OPTIONS)
    assert set(k for k, _, _ in perception.HAND_OPTIONS) <= set(perception._HAND_SETUPS)
    with pytest.raises(ValueError):
        perception.set_hand_option(99)


def test_switching_mid_run_takes_effect_between_frames(running):
    p = perception._default
    assert perception.set_hand_option(3) == "YOLO hand"
    assert perception.hand_option()[:2] == (3, "yolo")
    s = settle(p)
    assert p.hands.name == "yolo" and p.cfg.hand_depth == "fused" and p.cfg.own_arm
    assert s.hand is None or s.hand.tracker                                   # the frame went through
    perception.set_hand_option("mediapipe")
    settle(p)
    assert p.hands.name == "mediapipe" and p.cfg.hand_depth == "switch" and p.cfg.hand_filter == "average"


def test_an_option_that_cannot_load_falls_back_to_option_one(running, capsys):
    p = perception._default
    perception.set_hand_option(4)                         # yolo_medium: the fake refuses to load it
    settle(p)
    assert p.hands.name == "mediapipe" and perception.hand_option()[0] == 1
    assert "could not load" in capsys.readouterr().err


def test_the_hand_point_can_be_changed_while_running(running):
    p = perception._default
    perception.set_hand_option(3)
    settle(p)
    assert perception.set_hand_point("palm") == "palm" and p.cfg.hand_point == "palm"
    with pytest.raises(ValueError):
        perception.set_hand_point("elbow")
    perception.set_hand_option(5)                         # the next option keeps the choice
    settle(p)
    assert p.cfg.hand_point == "palm"


def test_the_environment_picks_the_starting_option(running, monkeypatch):
    monkeypatch.setenv("HAND_OPTION", "yolo")
    assert perception.hand_option()[:2] == (3, "yolo")
    monkeypatch.setenv("HAND_OPTION", "nonsense")
    assert perception.hand_option()[0] == 1


# --- the object ---------------------------------------------------------------------------------

def test_surface_depth_is_the_nearest_surface_of_some_size():
    solid = np.full(200, 0.60)
    assert surface_depth(solid) == pytest.approx(0.60)
    bottle = np.concatenate([np.full(60, 0.42), np.full(140, 1.3)])       # label and water; the table seen through
    assert surface_depth(bottle) == pytest.approx(0.42)
    stray = np.concatenate([np.full(3, 0.20), np.full(200, 0.60)])        # a few near readings are passed over
    assert surface_depth(stray) == pytest.approx(0.60)


def see_through(t=0.0, z=0.40):
    """A clear bottle: nothing readable in the middle of its box, only a strip of label and its foot."""
    f = frame(t=t, obj=None)
    x1, y1, x2, y2 = object_box(K, np.array([OBJ[0], OBJ[1], z]))
    f.depth[y1:y2, x1:x2] = 0.0
    f.depth[y2 - (y2 - y1) // 5:y2, x1:x2] = z
    return f, (x1, y1, x2, y2)


def test_a_see_through_object_is_read_from_the_rest_of_its_box():
    f, box = see_through()
    assert box_xyz(box, f.depth, K, loose=False) is None                  # what happened in the live run
    assert box_xyz(box, f.depth, K)[2] == pytest.approx(0.40)


def test_the_object_is_not_lost_while_the_detector_still_sees_it():
    near = np.array([OBJ[0], OBJ[1], 0.40])
    p = Perception(FakeDetector(near), None, PerceptionConfig())
    assert p.locate(see_through()[0], "bottle").object_xyz[2] == pytest.approx(0.40)
    for i in range(1, 100):                               # ten seconds with no reading at all in its box
        f, box = see_through(t=i * 0.1)
        f.depth[box[1]:box[3], box[0]:box[2]] = 0.0
        s = p.locate(f, "bottle")
    assert s.object_xyz is not None and s.object_visible and s.object_xyz[2] == pytest.approx(0.40, abs=0.01)


def test_a_weak_reading_does_not_move_a_remembered_object():
    near = np.array([OBJ[0], OBJ[1], 0.40])
    p = Perception(FakeDetector(near), None, PerceptionConfig())
    p.locate(see_through()[0], "bottle")
    f, box = see_through(t=0.1)
    f.depth[box[1]:box[3], box[0]:box[2]] = 0.0
    f.depth[box[3] - 20:box[3], box[0]:box[2]] = 1.3                      # only the table seen through it
    assert p.locate(f, "bottle").object_xyz[2] == pytest.approx(0.40, abs=0.01)


def test_one_missed_frame_does_not_lose_the_object_but_two_seconds_do():
    p = Perception(FakeDetector(), None, PerceptionConfig())
    p.locate(frame(t=0.0), "mug")
    p.detector.obj = None
    assert p.locate(frame(t=0.03, obj=None), "mug").object_xyz is not None
    assert p.locate(frame(t=1.9, obj=None), "mug").object_xyz is not None
    assert p.locate(frame(t=2.2, obj=None), "mug").object_xyz is None


def test_an_arm_in_front_keeps_the_object_longer_even_if_the_hand_tracker_missed_the_hand():
    p = Perception(FakeDetector(), None, PerceptionConfig())
    p.locate(frame(t=0.0), "mug")
    p.detector.obj = None

    def hidden(t):
        f = frame(t=t, obj=None)
        x1, y1, x2, y2 = object_box(K, OBJ)
        f.depth[y1 - 20:y2 + 20, x1 - 20:x2 + 20] = 0.35                  # an arm, 25 cm nearer than the object
        return f

    for t in (0.5, 1.5, 2.5, 3.5):
        assert p.locate(hidden(t), "mug").object_xyz is not None          # past the 2 s of an open view
    assert p.locate(hidden(4.6), "mug").object_xyz is None                # but not for ever


def test_a_hand_too_close_for_lidar_also_counts_as_hiding_the_object():
    p = Perception(FakeDetector(), None, PerceptionConfig())
    p.locate(frame(t=0.0), "mug")
    p.detector.obj = None
    f = frame(t=3.0, obj=None)
    f.depth[:] = 0.0                                      # the hand fills the view: no reading anywhere
    assert p.locate(f, "mug").object_xyz is not None      # after 3 s it is still remembered
    bottle = frame(t=3.0, obj=None)
    x1, y1, x2, y2 = object_box(K, OBJ)
    bottle.depth[y1:y2, x1:x2] = 0.0                      # a hole only as big as the object is not a hand
    q = Perception(FakeDetector(), None, PerceptionConfig())
    q.locate(frame(t=0.0), "mug")
    q.detector.obj = None
    assert q.locate(bottle, "mug").object_xyz is None
