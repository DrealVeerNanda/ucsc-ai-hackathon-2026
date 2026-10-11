"""An object the arm cannot reach is named as such, not steered at forever."""
import test_guidance as T


def far(scene):
    scene.object_out_of_reach = True
    return scene


def test_an_object_out_of_reach_is_said_so_once_and_then_now_and_then():
    g = T.dyn()
    for i in range(120):                                    # 12 s lined up with a bottle 80 cm away
        g.update(far(T.near(dx=0.0, dy=0.0, dz=0.30)), now=1.0 + 0.1 * i)
    said = [w for w in T.said(g) if "too far" in str(w).lower()]
    assert 1 <= len(said) <= 2


def test_nothing_is_said_about_reach_for_an_object_within_it_or_a_blip():
    g = T.dyn()
    for i in range(50):
        g.update(T.near(dx=0.10, dy=0.0, dz=0.25), now=1.0 + 0.1 * i)
    g.update(far(T.near(dx=0.10, dy=0.0, dz=0.25)), now=6.1)        # one frame of a bad reading
    g.update(T.near(dx=0.10, dy=0.0, dz=0.25), now=6.2)
    assert not [w for w in T.said(g) if "too far" in str(w).lower()]


def test_an_object_that_never_shows_is_given_up_on_so_the_wearer_is_asked_again():
    g = T.guide()
    states = [g.update(T.scene(obj=None, hand=None), now=float(t)) for t in range(1, 26)]
    assert states[:19] == ["searching"] * 19 and states[-1] == "idle"      # 20 s, after "I can't see that" at 8 s
    assert g.sink.spoken.count("i_cant_see_that") == 1


def test_an_object_that_was_seen_and_then_lost_gets_its_own_twenty_seconds():
    g = T.guide()
    g.update(T.scene(hand=None), now=1.0)                                   # found
    states = [g.update(T.scene(obj=None, hand=None), now=2.0 + t) for t in range(30)]
    assert states[10] == "searching" and states[-1] == "idle"               # "I lost it", a fresh search, then ask again


def test_grab_does_not_wait_for_ever_on_a_sound_card_that_has_stopped():
    g = T.guide()
    g.sink.saying = lambda: True                                            # nothing is being played, so nothing ever ends
    g.state, g._grab_t = "grab", 10.0
    assert g.update(T.near(), now=10.0 + 2 * g.cfg.grab_state_s) == "grab"
    assert g.update(T.near(), now=10.0 + 4.1 * g.cfg.grab_state_s) == "idle"


def test_a_hand_that_never_comes_is_given_up_on_too():
    g = T.guide()
    states = [g.update(T.scene(hand=None), now=1.0 + t) for t in range(60)]       # the object is "seen", nobody reaches
    assert states[25] == "waiting_hand" and states[-1] == "idle"
    assert 4 <= g.sink.spoken.count("show_me_your_hand") <= 7


def test_a_hand_that_comes_and_goes_is_not_given_up_on():
    g = T.guide()
    for t in range(120):                                                            # a slow, real reach with pauses
        state = g.update(T.scene(hand=None) if t % 30 < 20 else T.near(dx=0.2, dz=0.3), now=1.0 + t)
    assert state != "idle"
