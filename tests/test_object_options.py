"""The ways of keeping track of the object (perception.OBJECT_OPTIONS): watched as it always
was, remembered in the room, or found once and then not looked for again."""
import numpy as np
import pytest

import perception
import tracing
from fakes import object_box
from test_perception import HAND, K, OBJ, FakeDetector, FakeHands, _pose, frame, make

FPS = 30


def looks(p, n=20, t0=0.0, pose=None, **kw):
    """n frames with the object in clear view; returns the last scene and the time after it."""
    s = None
    for i in range(n):
        s = p.locate(frame(t=t0 + i / FPS, pose=_pose() if pose is None else pose, **kw), "mug")
    return s, t0 + n / FPS


def test_the_options_are_listed_and_the_first_is_what_there_always_was():
    assert [k for k, _, _ in perception.OBJECT_OPTIONS] == ["watch", "room", "locked"]
    assert perception.PerceptionConfig().object_mode == "watch" and not perception.PerceptionConfig().use_pose


def test_watched_takes_no_notice_of_the_pose():
    p = make(hands=())
    looks(p)
    assert p._lock is None
    p.detector.obj = None
    s = p.locate(frame(t=1.0, obj=None, pose=_pose(tx=0.10)), "mug")
    assert np.allclose(s.object_xyz, OBJ, atol=0.004)       # where it was in the picture, as before
    assert p.locate(frame(t=3.5, obj=None, pose=_pose(tx=0.10)), "mug").object_xyz is None


def test_remembered_in_the_room_is_the_old_pose_setting_by_another_name():
    p = make(hands=(), object_mode="room")
    looks(p)
    calls = p.detector.calls
    moved = _pose(tx=0.10, yaw=0.1)
    p.detector.obj = None
    s = p.locate(frame(t=5.0, obj=None, pose=moved), "mug")
    assert np.allclose(s.object_xyz, moved[:3, :3].T @ (OBJ - moved[:3, 3]), atol=0.004)
    assert p.detector.calls == calls + 1 and p._lock is None         # still looked for in every frame
    assert p.locate(frame(t=21.0, obj=None, pose=moved), "mug").object_xyz is None


def test_found_once_is_fixed_after_enough_clear_looks_and_not_looked_for_again():
    p = make(hands=(), object_mode="locked")
    for i in range(9):
        p.locate(frame(t=i / FPS, pose=_pose()), "mug")
    assert p._lock is None                                  # nine looks, a third of a second: not yet
    _, t = looks(p, n=8, t0=9 / FPS)
    assert p._lock is not None and np.allclose(p._lock[0], OBJ, atol=0.004)
    calls = p.detector.calls
    moved = _pose(tx=0.10, yaw=0.1)
    for i in range(60):                                     # two seconds, the object nowhere to be seen
        s = p.locate(frame(t=t + i / FPS, obj=None, pose=moved), "mug")
    assert p.detector.calls == calls
    assert np.allclose(s.object_xyz, moved[:3, :3].T @ (OBJ - moved[:3, 3]), atol=0.004)
    s = p.locate(frame(t=t + 2.0 + 1.5, obj=None, pose=moved), "mug")
    assert s.object_xyz is not None                         # and it does not run out


def test_found_once_gives_the_box_where_the_object_is_in_the_picture_now():
    p = make(hands=(), object_mode="locked")
    _, t = looks(p)
    moved = _pose(tx=0.10)
    s = p.locate(frame(t=t, obj=None, pose=moved), "mug")
    assert np.allclose(s.object_box, object_box(K, OBJ - np.array([0.10, 0, 0])), atol=3)
    assert not s.object_visible                             # it is not being looked at: see the test after a grab
    turned = _pose(yaw=np.pi)                               # the wearer has turned right round
    s = p.locate(frame(t=t + 0.1, obj=None, pose=turned), "mug")
    assert s.object_box is None and s.object_xyz[2] < 0


def test_a_hand_in_front_of_the_object_cannot_move_it_once_found():
    over = np.array([OBJ[0], OBJ[1], 0.35])                 # 25 cm short of the object, right in front of it
    p = perception.Perception(FakeDetector(OBJ), FakeHands(), perception.PerceptionConfig(smooth=1, object_mode="locked"))
    _, t = looks(p)
    p.hands = FakeHands(over)
    p.detector.obj = over                                   # ...and the detector would now take the hand for it
    for i in range(90):
        s = p.locate(frame(t=t + i / FPS, obj=None, hands=[over], pose=_pose()), "mug")
    assert np.allclose(s.object_xyz, OBJ, atol=0.004) and np.allclose(s.hand_xyz, over, atol=0.01)


def test_looks_that_disagree_do_not_fix_anything():
    p = make(hands=(), object_mode="locked")
    for i in range(40):
        obj = OBJ + np.array([0.0, 0.0, 0.08 * (i % 2)])    # 8 cm nearer and farther, frame about
        p.detector.obj = obj
        p.locate(frame(t=i / FPS, obj=obj, pose=_pose()), "mug")
    assert p._lock is None and p.detector.calls == 40


def test_asked_again_after_a_pause_it_is_looked_for_again():
    p = make(hands=(), object_mode="locked")
    _, t = looks(p)
    assert p._lock is not None
    there = OBJ + np.array([0.15, 0.0, 0.0])                # the wearer picked it up and put it down elsewhere
    p.detector.obj = there
    s, _ = looks(p, n=1, t0=t + 5.0, obj=there)             # five seconds of being listened to in between
    assert p._lock is None and np.allclose(s.object_xyz, there, atol=0.004)
    looks(p, n=20, t0=t + 5.0 + 1 / FPS, obj=there)
    assert np.allclose(p._lock[0], there, atol=0.004)


def test_a_new_target_is_looked_for_afresh():
    p = make(hands=(), object_mode="locked")
    _, t = looks(p)
    p.locate(frame(t=t, pose=_pose()), "cup")
    assert p._lock is None and p.detector.targets == ["mug", "cup"]


@pytest.mark.parametrize("pose, why", [(None, "no pose"), (_pose(tx=0.6), "the phone's tracking started afresh")])
def test_when_the_phone_loses_its_place_the_object_is_looked_for_again(tmp_path, monkeypatch, pose, why):
    monkeypatch.setattr(tracing, "_trace", None, raising=False)
    events = []
    monkeypatch.setattr(tracing, "event", lambda name, *a, **kw: events.append((name, kw)))
    p = make(hands=(), object_mode="locked")
    _, t = looks(p)
    calls = p.detector.calls
    p.detector.obj = None
    s = p.locate(frame(t=t, obj=None, pose=pose), "mug")
    assert s.object_xyz is None and p._lock is None and p.detector.calls == calls + 1
    assert ("object.unlocked", {"target": "mug", "why": why}) in events
    assert [n for n, _ in events].count("object.locked") == 1


def test_changing_the_option_lets_go_of_the_fixed_place():
    p = make(hands=(), object_mode="locked")
    _, t = looks(p)
    p.cfg.object_mode = "watch"
    calls = p.detector.calls
    s = p.locate(frame(t=t, pose=_pose(tx=0.10)), "mug")
    assert p._lock is None and p.detector.calls == calls + 1
    assert np.allclose(s.object_xyz, OBJ, atol=0.004)       # as the camera sees it in this frame, whatever the pose says


def test_the_option_is_chosen_by_setting_by_name_and_while_running(monkeypatch):
    monkeypatch.setattr(perception, "_default", None)
    monkeypatch.delenv("OBJECT_OPTION", raising=False)
    monkeypatch.delenv("PERCEPTION_USE_POSE", raising=False)
    assert perception.object_option() == (1, "watch", "watched")
    monkeypatch.setenv("PERCEPTION_USE_POSE", "1")          # the setting from before there were options
    assert perception.object_option()[:2] == (2, "room")
    monkeypatch.setenv("OBJECT_OPTION", "3")
    assert perception.object_option() == (3, "locked", "found once")
    monkeypatch.setenv("PERCEPTION_DETECTOR", "none")
    monkeypatch.setenv("PERCEPTION_HAND", "sticker")
    perception.locate(frame(hands=[HAND]), "mug")
    assert perception._default.cfg.object_mode == "locked"
    assert perception.set_object_option("watch") == "watched" and perception._default.cfg.object_mode == "watch"
    assert perception.set_object_option(2) == "remembered in the room" and perception.object_option()[0] == 2
    with pytest.raises(ValueError):
        perception.set_object_option(9)


def test_the_pages_and_the_voice_know_the_new_family(monkeypatch):
    import dashboard
    import launcher
    import voice
    families = {f["family"]: f for f in dashboard.option_families()}
    assert [o["key"] for o in families["object"]["options"]] == ["watch", "room", "locked"]
    assert families["object"]["say"] == "object option" and launcher.OPTION_ENV["object"] == "OBJECT_OPTION"
    monkeypatch.setenv("PERCEPTION_DETECTOR", "none")
    monkeypatch.setenv("PERCEPTION_HAND", "sticker")
    monkeypatch.setenv("GUIDANCE_AUDIO", "off")
    monkeypatch.setattr(perception, "_default", None)
    perception.locate(frame(hands=[HAND]), "mug")
    said = []
    monkeypatch.setattr(voice, "_say", said.append)
    voice.switch_option("object option three")
    assert perception.object_option()[1] == "locked" and said == ["Object option 3: found once."]
    assert voice.option_families()["options"]["object"]["now"] == 3


def test_found_once_an_object_taken_away_after_grab_is_not_a_miss():
    # Guidance says "Try again" when the hand is away from an object that is *seen* still there.
    # A fixed place is not a sighting: the wearer walking off with the bottle is not sent back.
    from audio_out import PrintSink
    from guidance import Guidance, GuidanceConfig
    p = make(hands=(HAND,), object_mode="locked")
    _, t = looks(p, hands=[HAND])
    g = Guidance(PrintSink(quiet=True), GuidanceConfig())
    g.reset("mug", now=t)
    at = OBJ - np.array([0.0, 0.0, 0.02])
    p.hands = FakeHands(at)
    state = None
    for i in range(45):
        state = g.update(p.locate(frame(t=t + i / FPS, obj=None, hands=[at], pose=_pose()), "mug"), now=t + i / FPS)
        if state == "grab":
            break
    assert state == "grab"
    t += (i + 1) / FPS
    gone = OBJ - np.array([0.30, 0.0, 0.25])                # the hand, with the object in it, 40 cm away
    p.hands = FakeHands(gone)
    for i in range(90):
        state = g.update(p.locate(frame(t=t + i / FPS, obj=None, hands=[gone], pose=_pose()), "mug"), now=t + i / FPS)
    assert state == "idle" and "Try again" not in g.sink.spoken


def test_an_option_clicked_between_start_and_the_first_request_is_not_lost(monkeypatch):
    # The models load when the first object is asked for. Until then the page said "ok" and the
    # choice was only kept for the next Start: the run in progress began on the old one.
    import dashboard
    monkeypatch.setattr(perception, "_default", None)
    monkeypatch.setattr(perception, "_hand_number", None)
    monkeypatch.delenv("OBJECT_OPTION", raising=False)
    monkeypatch.delenv("HAND_OPTION", raising=False)
    monkeypatch.setenv("GUIDANCE_AUDIO", "off")
    with pytest.raises(RuntimeError, match="not running yet"):
        dashboard.set_option("object", 3)                   # a run on the fake perception: nothing to switch
    monkeypatch.setattr(perception, "in_use", True)
    assert dashboard.set_option("object", 3) == "found once"
    assert dashboard.set_option("hand", 2) == "MediaPipe, steady"
    now = {f["family"]: f["current"] for f in dashboard.option_families()}
    assert now["object"] == 3 and now["hand"] == 2
    assert perception.OBJECT_OPTIONS[perception._object_wanted()][0] == "locked" and perception.hand_option()[0] == 2
