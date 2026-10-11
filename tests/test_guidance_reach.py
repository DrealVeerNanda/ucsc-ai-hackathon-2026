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
