"""Lane 3, second half: the rules for saying "Grab", the further ways of guiding, the fixes
for what the live run of 10 Oct showed, what guidance writes to the trace, and the tool that
renders every option to listen to."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

import audio_out
import guidance
import tracing
from audio_out import Mixer, PrintSink, ToneConfig, ToneSynth
from guidance import Guidance, GuidanceConfig
from interfaces import Scene

OBJ = np.array([0.05, 0.12, 0.60])
BOX = (380, 420, 460, 640)                   # the object in the picture, 80 px wide
FPS = 30
PATIENT = ("zone", "over", "arrive")
ALL_GRABS = [k for k, _, _ in guidance.GRAB_OPTIONS]


def scene(hand=(-0.15, 0.05, 0.40), obj=OBJ, t=0.0, px=None, box=None, seen=True):
    hand = None if hand is None else np.array(hand, float)
    return Scene(object_xyz=None if obj is None else np.array(obj, float), hand_xyz=hand,
                 object_visible=obj is not None and seen, hand_visible=hand is not None,
                 object_box=box, hand_px=px, t=t)


def near(dx=0.0, dy=0.0, dz=0.0, **kw):
    """A hand at the object, offset by the given error (object minus hand)."""
    return scene(hand=OBJ - np.array([dx, dy, dz]), **kw)


def over(dz=0.0, outside_px=0.0, **kw):
    """The same with the picture: the fingertip over the object's box, or that many pixels beside it."""
    return near(dz=dz, px=(BOX[2] + outside_px if outside_px else 420, 500), box=BOX, **kw)


def guide(grab="over", option="combined", hand=True, **kw):
    g = Guidance(PrintSink(quiet=True), GuidanceConfig(**kw))
    g.set_option(option)
    g.set_grab_option(grab)
    g.reset("mug", now=0.0)
    g.update(scene(hand=None), now=0.05)                   # object found, hand not shown yet
    if hand:
        g.update(scene(), now=0.1)                         # the hand comes into view, well away
    g.sink.spoken.clear()
    return g


def run(g, frames, t0=1.0, fps=FPS):
    """Feed scenes (or callables of time) one per frame. Returns the time of "grab", or None."""
    for i, f in enumerate(frames):
        t = t0 + i / fps
        if g.update(f(t) if callable(f) else f, now=t) == "grab":
            return round(t - t0, 3)
    return None


def said(g):
    return [w for w in g.sink.spoken if w not in ("looking", "found_it", "show_me_your_hand")]


# --- the rules for "Grab": contract ------------------------------------------------

def test_grab_rules_are_listed_and_the_old_one_is_still_among_them():
    keys = [k for k, _, _ in guidance.GRAB_OPTIONS]
    assert len(keys) >= 4 and len(set(keys)) == len(keys) and keys[0] == "fingertip"
    assert all(title and text.endswith(".") for _, title, text in guidance.GRAB_OPTIONS)
    assert GuidanceConfig().grab_option == "over"          # the default changed: Sasha asked for it


def test_grab_rule_is_switchable_while_running(monkeypatch):
    monkeypatch.setenv("GUIDANCE_AUDIO", "off")
    monkeypatch.delenv("GUIDANCE_GRAB", raising=False)
    monkeypatch.setattr(guidance, "_default", None)
    assert guidance.grab_option() == (3, "over", "over the object")
    assert guidance.set_grab_option(1) == "fingertip touch" and guidance.grab_option()[:2] == (1, "fingertip")
    assert guidance.set_grab_option("zone") == "grabbing distance" and guidance.grab_option()[0] == 2
    with pytest.raises(ValueError):
        guidance.set_grab_option(9)
    monkeypatch.setenv("GUIDANCE_GRAB", "arrive")
    monkeypatch.setattr(guidance, "_default", None)
    assert guidance.grab_option()[1] == "arrive"
    monkeypatch.setenv("GUIDANCE_GRAB", "nonsense")
    monkeypatch.setattr(guidance, "_default", None)
    with pytest.raises(ValueError):
        guidance.grab_option()
    monkeypatch.setattr(guidance, "_default", None)


def test_switching_the_rule_mid_reach_starts_its_count_again():
    g = guide("zone")
    run(g, [near(dz=0.08)] * 6)
    assert g._near_t is not None
    g.set_grab_option("fingertip")
    assert g._near_t is None and run(g, [near(dz=0.08)] * 30, t0=2.0) is None     # 8 cm is not a touch


# --- the old rule, kept to compare ---------------------------------------------------

def test_the_old_rule_needs_five_centimetres_in_every_frame():
    assert run(guide("fingertip"), [near(dz=0.08)] * 90) is None            # a hand's width away: never
    assert run(guide("fingertip"), [near(dz=0.03)] * 30) == pytest.approx(0.3, abs=0.04)
    blink = [near(dz=0.03) if i % 6 else scene(hand=None) for i in range(90)]     # the tracker misses every sixth frame
    assert run(guide("fingertip"), blink) is None                           # ...and each miss starts the count again


def test_the_wide_circle_is_the_old_rule_with_a_bigger_radius():
    assert run(guide("near"), [near(dz=0.08)] * 30) == pytest.approx(0.3, abs=0.04)
    assert run(guide("near"), [near(dz=0.12)] * 60) is None


# --- grabbing distance ------------------------------------------------------------------

@pytest.mark.parametrize("rule", PATIENT)
def test_a_hand_a_few_centimetres_short_is_in_grabbing_distance(rule):
    assert run(guide(rule), [near(dz=0.08)] * 30) == pytest.approx(0.3, abs=0.04)
    assert run(guide(rule), [near(dx=0.04, dz=0.05)] * 30) == pytest.approx(0.3, abs=0.04)


def test_depth_is_allowed_more_than_sideways():
    assert run(guide("zone"), [near(dz=0.09)] * 30) is not None             # 9 cm short in depth: the rough axis
    assert run(guide("zone"), [near(dx=0.09)] * 60) is None                 # 9 cm to the side is simply beside it
    assert run(guide("zone"), [near(dy=-0.09)] * 60) is None
    assert run(guide("zone"), [near(dz=0.14)] * 60) is None                 # and there is a limit in depth too


@pytest.mark.parametrize("rule", PATIENT)
def test_tracker_blinks_do_not_restart_the_count(rule):
    blink = [near(dz=0.06) if i % 6 else scene(hand=None) for i in range(40)]
    assert run(guide(rule), blink) is not None and run(guide(rule), blink) < 0.6


@pytest.mark.parametrize("rule", PATIENT)
def test_time_out_of_sight_is_not_time_spent_there(rule):
    frames = [near(dz=0.06)] * 3 + [scene(hand=None)] * 12 + [near(dz=0.06)] * 3
    assert run(guide(rule), frames) is None                 # seen there for six frames in all, not for a third of a second
    assert run(guide(rule), frames + [near(dz=0.06)] * 8) is not None


@pytest.mark.parametrize("rule", ("zone", "over"))
def test_a_hand_gone_for_long_starts_again(rule):
    g = guide(rule)
    run(g, [near(dz=0.06)] * 6 + [scene(hand=None)] * 20)                  # two thirds of a second unseen
    assert g._near_t is None


@pytest.mark.parametrize("rule", PATIENT)
def test_one_bad_frame_inside_the_zone_is_forgiven_but_leaving_is_not(rule):
    one_bad = [near(dz=0.06)] * 5 + [near(dx=0.10, dz=0.06)] * 2 + [near(dz=0.06)] * 8
    assert run(guide(rule), one_bad) is not None
    left = [near(dz=0.06)] * 5 + [near(dx=0.12, dz=0.06)] * 8 + [near(dz=0.06)] * 6
    assert run(guide(rule), left) is None                   # away for a quarter of a second: the count starts over


@pytest.mark.parametrize("rule", PATIENT)
def test_a_hand_sweeping_past_is_not_an_arrival(rule):
    def sweep(t):                                           # 40 cm/s across the front of the object, 4 cm short of it
        return near(dx=0.3 - 0.4 * (t - 1.0), dz=0.04)
    assert run(guide(rule), [sweep] * 45) is None
    slow = [lambda t: near(dx=0.10 - 0.08 * min(t - 1.0, 1.25), dz=0.04)] * 60     # the same path at 8 cm/s, stopping on it
    assert run(guide(rule), slow) is not None


# --- what a hand cannot do ---------------------------------------------------------------

@pytest.mark.parametrize("rule", PATIENT)
def test_depth_that_jumps_onto_the_object_is_not_believed(rule):
    # 10 Oct, 324.4 s: the hand 24 cm short of the bottle, lined up with it, and for a third of a
    # second lidar read past the finger onto the bottle.
    short, on_it = over(dz=0.24), over(dz=0.01)
    frames = [short] * 15 + [on_it] * 10 + [short] * 15 + [on_it] * 12 + [short] * 10
    assert run(guide(rule), frames) is None
    assert run(guide("near"), frames) is not None           # the plain circle falls for it


@pytest.mark.parametrize("rule", PATIENT)
def test_a_hand_that_really_goes_forward_is_believed_soon_after(rule):
    frames = [lambda t: over(dz=max(0.03, 0.24 - 0.5 * (t - 1.0)))] * 60    # a brisk 50 cm/s reach, then on it
    t = run(guide(rule), frames)
    assert t is not None and t < 1.5


@pytest.mark.parametrize("rule", PATIENT)
def test_an_object_that_jumps_onto_the_hand_is_not_an_arrival(rule):
    # 10 Oct, 326.8 s: the bottle 0.74 m away, the hand at 0.50 m in front of it, the tracker
    # blinked, and the bottle's depth was measured on the hand. "Grab", 24 cm short.
    hand = OBJ - np.array([0.0, 0.0, 0.24])
    stolen = np.array([OBJ[0], OBJ[1], hand[2] + 0.01])
    frames = [scene(hand=hand)] * 20 + [scene(hand=None, obj=stolen)] * 2 + [scene(hand=hand, obj=stolen)] * 60
    assert run(guide(rule), frames) is None
    assert run(guide("fingertip"), frames) is not None      # as it did that night
    # Seen clearly apart again (the hand moves off), an arrival counts once more.
    back = frames + [scene(hand=hand + np.array([0.2, 0, 0]), obj=stolen)] * 10 + \
        [lambda t: scene(hand=hand + np.array([max(0.0, 0.2 - 0.4 * (t - 4.07)), 0, 0]), obj=stolen)] * 60
    assert run(guide(rule), back) is not None


def test_an_object_found_with_the_hand_already_beside_it_can_be_grabbed():
    g = Guidance(PrintSink(quiet=True), GuidanceConfig())
    g.reset("mug", now=0.0)
    assert run(g, [near(dz=0.06)] * 30) is not None         # nothing jumped: it was simply found there


def test_an_object_that_jumps_somewhere_else_changes_nothing():
    elsewhere = OBJ + np.array([0.3, 0.0, 0.2])
    frames = [near(dz=0.3)] * 10 + [scene(hand=OBJ - [0, 0, 0.3], obj=elsewhere)] * 10 + \
        [lambda t: near(dz=max(0.05, 0.3 - 0.3 * (t - 1.67)))] * 60
    assert run(guide("over"), frames) is not None


# --- over the object in the picture ---------------------------------------------------------

def test_over_the_object_takes_what_a_hand_on_it_measures():
    # A hand wrapped round the bottle measured 6 to 15 cm short of it, for seconds.
    assert run(guide("over"), [over(dz=0.08)] * 30) == pytest.approx(0.3, abs=0.04)
    slow = run(guide("over"), [over(dz=0.13)] * 60)
    assert slow == pytest.approx(0.8, abs=0.05)             # believed, but only after most of a second
    assert run(guide("zone"), [over(dz=0.13)] * 60) is None
    assert run(guide("over"), [over(dz=0.20)] * 120) is None                # out of reach stays out of reach


def test_over_the_object_means_over_it():
    assert run(guide("over"), [over(dz=0.05, outside_px=15)] * 30) is not None      # a fifth of its width beside it
    assert run(guide("over"), [over(dz=0.05, outside_px=60)] * 60) is None          # most of its width beside it
    assert run(guide("over"), [over(dz=-0.06)] * 30) is not None            # a little past its front: round the side
    assert run(guide("over"), [over(dz=-0.12)] * 60) is None


def test_a_hand_coming_in_through_the_wide_band_is_not_called_early():
    # 6 cm/s forward, over the object all the way: "Grab" comes near 10 cm, not at 15.
    g = guide("over")
    t = run(g, [lambda t: over(dz=0.20 - 0.06 * (t - 1.0))] * 120)
    assert t is not None and 0.20 - 0.06 * t < 0.115


def test_without_the_picture_over_falls_back_to_grabbing_distance():
    assert run(guide("over"), [near(dz=0.13)] * 60) is None
    assert run(guide("over"), [near(dz=0.08)] * 30) is not None


# --- arriving by disappearing ------------------------------------------------------------------

def closing(t):
    return over(dz=max(0.11, 0.25 - 0.2 * (t - 1.0)))      # in to 11 cm short, slowing


def test_a_hand_that_closes_in_and_vanishes_has_arrived():
    frames = [closing] * 24 + [scene(hand=None, box=BOX)] * 12
    assert run(guide("arrive"), frames) == pytest.approx(0.8 + 0.25, abs=0.07)
    assert run(guide("over"), frames) is None and run(guide("zone"), frames) is None


def test_vanishing_is_only_an_arrival_close_to_the_object_and_slowly():
    far = [over(dz=0.30)] * 24 + [scene(hand=None, box=BOX)] * 12
    assert run(guide("arrive"), far) is None
    fast = [lambda t: near(dx=0.5 - 0.6 * (t - 1.0), dz=0.03)] * 24 + [scene(hand=None)] * 12     # crossed and left the view
    assert run(guide("arrive"), fast) is None
    stray = [closing] * 24 + [over(dz=0.35)] * 2 + [scene(hand=None, box=BOX)] * 12   # last seen well away
    assert run(guide("arrive"), stray) is None


# --- after "Grab" --------------------------------------------------------------------------------

@pytest.mark.parametrize("rule", ALL_GRABS)
def test_grab_is_said_once_and_then_it_is_idle(rule):
    g = guide(rule)
    assert run(g, [near(dz=0.02)] * 30) is not None
    for i in range(90):
        state = g.update(near(dz=0.02), now=3.0 + i / FPS)
    assert state == "idle" and g.sink.spoken.count("grab") == 1


def test_a_hand_that_comes_away_empty_is_guided_again():
    g = guide("over")
    t = run(g, [near(dz=0.02)] * 30)
    away = scene(hand=OBJ - np.array([0.0, 0.25, 0.20]))                    # pulled back, bottle seen still standing
    for i in range(60):
        state = g.update(away, now=1.0 + t + 0.5 + i / FPS)
    assert state == "guiding" and "Try again" in g.sink.spoken
    assert run(g, [lambda t: near(dy=0.0, dz=max(0.03, 0.2 - 0.2 * (t - 5.0)))] * 90, t0=5.0) is not None
    assert g.sink.spoken.count("grab") == 2


def test_a_hand_that_takes_the_object_away_ends_the_reach():
    g = guide("over")
    t = run(g, [near(dz=0.02)] * 30)
    gone = scene(hand=OBJ - np.array([0.0, 0.25, 0.20]), seen=False)        # the object is only remembered now
    for i in range(60):
        state = g.update(gone, now=1.0 + t + 0.5 + i / FPS)
    assert state == "idle" and "Try again" not in g.sink.spoken
    g2 = guide("over")
    t = run(g2, [near(dz=0.02)] * 30)
    for i in range(60):
        state = g2.update(scene(hand=None), now=1.0 + t + 0.5 + i / FPS)    # hand and bottle both out of view
    assert state == "idle"


# --- the further ways of guiding: contract ----------------------------------------------------------

NEW = ("numbers", "echo", "hotcold", "spearcons", "beam", "twostage", "detector")


def test_twelve_options_each_with_a_title_and_one_plain_description():
    keys = [k for k, _, _ in guidance.OPTIONS]
    assert keys[:5] == ["tones", "steps", "combined", "combined_beeps", "clock"]       # the first five are as they were
    assert tuple(keys[5:]) == NEW and len(set(keys)) == 12
    for _, title, text in guidance.OPTIONS:
        assert title and 40 < len(text) < 260 and text.endswith(".")


def test_new_options_are_reached_by_number_and_by_name(monkeypatch):
    monkeypatch.setenv("GUIDANCE_AUDIO", "off")
    monkeypatch.setenv("GUIDANCE_MODE", "3")
    monkeypatch.setattr(guidance, "_default", None)
    for n, key in enumerate(NEW, start=6):
        title = guidance.set_option(n)
        assert guidance.option() == (n, key, title)
        assert guidance.set_option(key) == title
    with pytest.raises(ValueError):
        guidance.set_option(13)
    monkeypatch.setenv("GUIDANCE_MODE", "12")
    monkeypatch.setattr(guidance, "_default", None)
    assert guidance.option()[1] == "detector"
    monkeypatch.setattr(guidance, "_default", None)


@pytest.mark.parametrize("option", [k for k, _, _ in guidance.OPTIONS])
@pytest.mark.parametrize("rule", ALL_GRABS)
def test_every_option_reaches_grab_under_every_rule(option, rule):
    g = guide(rule, option)
    assert run(g, [lambda t: near(dx=max(0.0, 0.1 - 0.1 * (t - 1.0)), dz=max(0.02, 0.2 - 0.1 * (t - 1.0)))] * 120) is not None
    assert g.sink.spoken[-1] == "grab" and g.sink.tone[3] == "silent"


@pytest.mark.parametrize("option", NEW)
def test_switching_to_a_new_option_mid_reach_and_back_is_safe(option):
    g = guide()
    g.update(near(dx=0.15, dy=0.1, dz=0.25), now=1.0)
    g.set_option(option)
    for i in range(30):
        assert g.update(near(dx=0.15, dy=0.1, dz=0.25), now=1.1 + i / FPS) == "guiding"
    g.set_option("combined")
    g.update(near(dx=0.15, dy=0.1, dz=0.25), now=2.2)
    assert g.sink.tone[3] == "silent"


@pytest.mark.parametrize("option", NEW)
def test_a_hand_blink_and_a_lost_hand_are_handled_in_every_new_option(option):
    g = guide(option=option)
    g.update(near(dx=0.15, dz=0.25), now=1.0)
    tone = g.sink.tone
    assert g.update(scene(hand=None), now=1.3) == "guiding" and g.sink.tone == tone     # a blink: the sound holds
    assert g.update(scene(hand=None), now=1.7) == "guiding" and g.sink.tone[3] == "silent"
    assert g.update(scene(hand=None), now=2.2) == "waiting_hand" and g.sink.spoken[-1] == "show_me_your_hand"


# --- 6: measured words -----------------------------------------------------------------------------

def test_measured_words_say_how_far_in_steps_a_hand_can_make():
    assert [guidance._amount(m) for m in (0.01, 0.07, 0.13, 0.22, 0.31, 0.44, 0.9)] == \
        ["five", "five", "fifteen", "twenty", "thirty", "forty", "sixty"]
    g = guide(option="numbers", hand=False)
    g.update(near(dx=0.21, dy=-0.09, dz=0.25), now=1.0)
    assert said(g) == ["up ten, right twenty"] and g.sink.tone[3] == "silent"


def test_measured_words_give_one_instruction_for_each_move():
    g = guide(option="numbers", hand=False)
    g.update(near(dx=0.20, dz=0.25), now=1.0)
    for i in range(45):                                     # the hand makes its move: 20 cm in a second and a half
        g.update(near(dx=0.20 - 0.14 * i / 45, dz=0.25), now=1.03 + i / FPS)
    assert said(g) == ["right twenty"]                      # nothing new while it is on its way
    for i in range(20):                                     # it has stopped, 6 cm short
        g.update(near(dx=0.06, dz=0.25), now=2.55 + i / FPS)
    assert said(g) == ["right twenty", "right five"]
    for i in range(30):
        g.update(near(dx=0.0, dz=0.25), now=3.3 + i / FPS)
    assert said(g)[2:] == ["aligned", "forward twenty five"]


def test_measured_words_repeat_for_a_hand_that_does_not_move():
    g = guide(option="numbers", hand=False)
    for i in range(100):
        g.update(near(dx=0.20, dz=0.25), now=1.0 + i / FPS)
    assert said(g) == ["right twenty", "right twenty"]      # once more after two and a half seconds


# --- 7: ping and echo ---------------------------------------------------------------------------------

def test_the_echo_sounds_where_the_object_is():
    g = guide(option="echo")
    g.update(near(dx=0.15, dy=-0.10, dz=0.3), now=1.0)      # object to the right and above
    pan, freq, rate, mode = g.sink.tone
    assert mode == "echo" and pan > 0.5 and freq > g.tone.base_hz and said(g) == []
    g.update(near(dx=-0.15, dy=0.10, dz=0.3), now=1.1)
    assert g.sink.tone[0] < -0.5 and g.sink.tone[1] < g.tone.base_hz


def test_the_echo_follows_sooner_as_the_hand_closes_in_and_matches_the_ping_when_lined_up():
    g = guide(option="echo")
    gaps = []
    for i, d in enumerate((0.5, 0.35, 0.2, 0.12)):
        for j in range(30):
            g.update(near(dz=d), now=1.0 + i + j / FPS)
        gaps.append(g.sink.gap)
        assert g.sink.tone[:2] == (0.0, g.tone.base_hz)     # in line: the same note in the middle of the head
    assert gaps == sorted(gaps, reverse=True) and gaps[0] > 2 * gaps[-1]
    assert all(1.0 / g.sink.tone[2] > gap for gap in gaps[-1:])             # the echo fits inside its cycle


def test_echo_tone_is_two_notes_the_second_at_the_given_side_and_pitch():
    s = ToneSynth()
    s.set(1.0, 2 * s.cfg.base_hz, 1.5, "echo", gap=0.3)
    out = np.concatenate([s.render(256) for _ in range(375)])               # 2 s: three cycles
    loud = np.abs(out).max(axis=1) > 0.02
    env = np.convolve(loud.astype(float), np.ones(300) / 300, "same") > 0.05
    assert np.count_nonzero(env[1:] & ~env[:-1]) == 6                       # a ping and an echo in each
    cycle = out[64000:96000]                                # the third cycle, after the pan has settled
    ping, echo = cycle[:int(0.07 * 48000)], cycle[int(0.3 * 48000):int(0.37 * 48000)]
    assert np.abs(ping[:, 0]).max() == pytest.approx(np.abs(ping[:, 1]).max(), rel=0.05)    # the ping is in the middle
    assert np.abs(echo[:, 1]).max() > 0.1 and np.abs(echo[:, 0]).max() < 0.01               # the echo in the right ear
    crossings = lambda x: np.count_nonzero(np.diff(np.signbit(x)))          # noqa: E731
    assert crossings(echo[:, 1]) > 1.7 * crossings(ping[:, 1])              # and an octave higher


# --- 8: warmer and colder -----------------------------------------------------------------------------

def test_the_note_rises_all_the_way_in_and_leans_to_the_side():
    g = guide(option="hotcold")
    notes = []
    for i, d in enumerate((0.6, 0.5, 0.4, 0.3, 0.2, 0.12)):
        for j in range(30):
            g.update(near(dx=d / 2, dz=d), now=1.0 + i + j / FPS)
        notes.append(g.sink.tone[1])
        assert g.sink.tone[3] == "steady" and g.sink.tone[0] > 0
    assert notes[0] == notes[1] == pytest.approx(220.0, rel=0.02)           # as cold as it gets from half a metre
    assert notes[1:] == sorted(notes[1:]) and notes[-1] > 2.5 * notes[1] and said(g) == []
    g.update(near(dx=-0.2, dz=0.3), now=9.0)
    assert g.sink.tone[0] < 0


def test_the_note_does_not_warble_with_the_depth_flipping_four_centimetres():
    g = guide(option="hotcold")
    notes = []
    for i in range(90):
        g.update(near(dz=0.30 + (0.04 if (i // 5) % 2 else 0.0)), now=1.0 + i / FPS)
        notes.append(g.sink.tone[1])
    swing = 12 * np.log2(max(notes[30:]) / min(notes[30:]))
    assert swing < 1.2                                      # semitones: the raw flip would be more than two


def test_continuous_notes_are_quieter_than_beeps():
    g = guide(option="hotcold")
    quiet = g.tone.volume
    g.set_option("tones")
    assert quiet < g.tone.volume
    g.set_option("combined_beeps")
    assert g.tone.volume == 0.16


# --- 9: fast words ---------------------------------------------------------------------------------------

def test_fast_words_repeat_faster_when_closer_and_take_turns_on_a_diagonal():
    g = guide(option="spearcons")
    for i in range(90):
        g.update(near(dx=0.25, dy=-0.25, dz=0.3), now=1.0 + i / FPS)
    far = said(g)
    assert set(far) == {"fast_up", "fast_right"} and all(a != b for a, b in zip(far, far[1:]))
    g2 = guide(option="spearcons")
    for i in range(90):
        g2.update(near(dx=0.07, dy=-0.07, dz=0.02), now=1.0 + i / FPS)
    assert len(said(g2)) > 1.5 * len(far)


def test_fast_words_chime_when_lined_up_and_then_say_forward():
    g = guide(option="spearcons")
    for i in range(60):
        g.update(near(dx=0.01, dz=0.3), now=1.0 + i / FPS)
    assert said(g)[0] == "aligned" and set(said(g)[1:]) == {"fast_forward"}


def test_fast_words_are_one_word_and_short():
    words = audio_out.short_words(48000)
    assert {f"{kind}_{w}" for kind in ("word", "fast") for w in audio_out.DIRECTIONS} <= set(words)
    for w in audio_out.DIRECTIONS:
        assert len(words[f"fast_{w}"]) < 0.75 * len(words[f"word_{w}"]) or len(words[f"word_{w}"]) < 0.25 * 48000
        assert len(words[f"word_{w}"]) / 48000 < 0.7        # not the 1.4 s sentences of the newest clip set
        assert len(words[f"fast_{w}"]) / 48000 < 0.4
    assert audio_out.CLIP_TEXT["word_left"] == "left"


def test_speeding_up_keeps_the_pitch_and_the_level():
    sr = 48000
    t = np.arange(sr) / sr
    x = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    y = audio_out.speed_up(x, 2.0, sr)
    assert 0.45 * len(x) < len(y) < 0.55 * len(x)
    mid = y[4800:-4800]
    assert np.abs(mid).max() == pytest.approx(0.3, rel=0.25)
    crossings = np.count_nonzero(np.diff(np.signbit(mid)))
    assert crossings / (len(mid) / sr) == pytest.approx(400, rel=0.1)       # still 200 Hz
    assert audio_out.speed_up(x[:1000], 2.0, sr).size == 1000               # too short to cut: left alone


# --- 10: stay on the beam -----------------------------------------------------------------------------------

def test_on_the_beam_it_is_quiet_and_a_drift_is_ticks_not_words():
    g = guide(option="beam", hand=False, lead_s=0.0)
    g.update(near(dx=0.15, dy=0.12, dz=0.25), now=1.0)
    assert said(g) == ["down and right"] and g.sink.tone[3] == "silent"
    for i in range(30):
        g.update(near(dx=0.01, dy=0.0, dz=0.25), now=2.0 + i / FPS)
    assert said(g)[1:] == ["aligned", "forward"] and g.sink.tone[3] == "silent"
    n = len(said(g))
    for i in range(30):
        g.update(near(dx=0.07, dy=0.0, dz=0.25), now=3.5 + i / FPS)         # drifted 7 cm: object now to the right
    pan, freq, rate, mode = g.sink.tone
    assert mode == "beep" and pan > 0.5 and freq == g.tone.base_hz and len(said(g)) == n
    g.update(near(dx=0.0, dy=-0.07, dz=0.25), now=5.0)                      # too low: high ticks say go up
    assert g.sink.tone[0] == 0.0 and g.sink.tone[1] == 2 * g.tone.base_hz
    g.update(near(dx=0.0, dy=0.07, dz=0.25), now=5.1)
    assert g.sink.tone[1] == g.tone.base_hz / 2


def test_far_off_the_beam_it_goes_back_to_words():
    g = guide(option="beam")
    for i in range(30):
        g.update(near(dx=0.01, dz=0.25), now=1.0 + i / FPS)
    for i in range(30):
        g.update(near(dx=0.2, dz=0.25), now=3.0 + i / FPS)
    assert said(g)[-1] == "right" and g.sink.tone[3] == "silent"


def test_ticks_come_faster_the_further_off_the_beam():
    g = guide(option="beam", lead_s=0.0)
    for i in range(30):
        g.update(near(dx=0.01, dz=0.25), now=1.0 + i / FPS)
    g.update(near(dx=0.06, dz=0.25), now=3.0)
    a = g.sink.tone[2]
    g.update(near(dx=0.10, dz=0.25), now=3.1)
    assert g.sink.tone[2] > a


# --- 11: words then a note ------------------------------------------------------------------------------------

def test_words_for_the_big_move_then_the_note_for_the_last_hand_width():
    g = guide(option="twostage", hand=False)
    g.update(near(dx=0.2, dz=0.25), now=1.0)
    assert said(g) == ["right twenty"] and g.sink.tone[3] == "silent"
    for i in range(20):
        g.update(near(dx=0.03, dz=0.115), now=2.0 + i / FPS)
    assert g.sink.tone[3] == "steady" and len(said(g)) == 1
    low = g.sink.tone[1]
    for i in range(20):
        g.update(near(dx=0.03, dz=0.115 - 0.0005 * i), now=2.7 + i / FPS)
    assert g.sink.tone[1] > low
    g.update(near(dx=0.03, dz=0.15), now=3.5)               # a little further out again: the note stays
    assert g.sink.tone[3] == "steady"
    g.update(near(dx=0.03, dz=0.25), now=3.6)
    assert g.sink.tone[3] == "silent"


# --- 12: metal detector ---------------------------------------------------------------------------------------

def test_the_hum_rises_as_the_hand_lines_up_and_becomes_beeps_on_the_line():
    g = guide(option="detector")
    hum = []
    for i, d in enumerate((0.30, 0.20, 0.12, 0.07)):
        for j in range(30):
            g.update(near(dx=d, dz=0.25), now=1.0 + i + j / FPS)
        hum.append(g.sink.tone[1])
        assert g.sink.tone[3] == "steady" and g.sink.tone[0] > 0
    assert hum == sorted(hum) and hum[-1] > 2 * hum[0] and said(g) == []
    for j in range(15):
        g.update(near(dx=0.01, dz=0.25), now=6.0 + j / FPS)
    far = g.sink.tone
    assert far[3] == "beep" and far[0] == 0.0 and far[1] > hum[-1]
    g.update(near(dx=0.01, dz=0.08), now=7.0)
    assert g.sink.tone[2] > far[2]                          # faster as the hand pushes forward


# --- what the live run showed ----------------------------------------------------------------------------------

def test_a_detector_blink_is_not_i_lost_it_found_it():
    g = guide()
    g.update(near(dx=0.15, dz=0.25), now=1.0)
    assert g.update(scene(obj=None, hand=None), now=1.03) == "guiding"      # 10 Oct: lost and found 0.03 s apart
    assert g.update(near(dx=0.15, dz=0.25), now=1.07) == "guiding"
    assert "i_lost_it" not in g.sink.spoken and "found_it" not in g.sink.spoken
    for i in range(15):
        state = g.update(scene(obj=None, hand=None), now=2.0 + i / FPS)
    assert state == "searching" and g.sink.spoken.count("i_lost_it") == 1


def test_show_me_your_hand_is_dropped_if_the_hand_shows_before_it_is_said():
    speech = np.full(48000, 0.2, np.float32)                # one-second clips
    m = Mixer(ToneConfig(), clips={"looking": speech, "found_it": speech, "show_me_your_hand": speech}, stale_s=1e9)
    g = Guidance(m, GuidanceConfig(), m.cfg)
    g.reset("mug", now=0.0)
    m.render(4800)
    g.update(scene(hand=None), now=0.1)                     # "found it" cuts in, "show me your hand" waits its turn
    assert m.spoken == ["looking", "found_it"] and m._next == "show_me_your_hand"
    m.render(4800)
    g.update(scene(), now=0.2)                              # and there is the hand
    for _ in range(30):
        m.render(4800)
    assert "show_me_your_hand" not in m.spoken


def test_cancel_drops_only_a_phrase_that_is_still_waiting():
    m = Mixer(ToneConfig(), clips={"a": np.full(4800, 0.2, np.float32), "b": np.full(4800, 0.2, np.float32)})
    m.say("a")
    m.say("b")
    m.cancel("a")                                           # already playing: left to finish
    assert m._clip is not None and m._next == "b"
    m.cancel("b")
    assert m._next is None
    while m.saying():
        m.render(256)
    assert m.spoken == ["a"]


def test_grab_does_not_chime_again_right_after_the_lined_up_chime():
    tone = ToneConfig()
    speech = np.full(4800, 0.3, np.float32)
    m = Mixer(tone, clips={"grab": speech})
    m.say("aligned", now=True)
    m.render(int(0.4 * tone.sr))                            # 10 Oct: chime, and 0.4 s later the chime again, then "Grab"
    m.say("grab", now=True)
    assert len(m._clip) == len(speech)
    m2 = Mixer(tone, clips={"grab": speech})
    m2.say("aligned", now=True)
    m2.render(int(2.0 * tone.sr))
    m2.say("grab", now=True)
    assert len(m2._clip) == len(audio_out.success_chime(tone.sr)) + len(speech)
    m3 = Mixer(tone, clips={"grab": speech})
    m3.say("grab", now=True)                                # no chime before it at all: it brings its own
    assert len(m3._clip) == len(audio_out.success_chime(tone.sr)) + len(speech)


class Talking(PrintSink):
    """A sink that is busy for a while after each phrase, like the real one."""

    def __init__(self, seconds=0.6):
        super().__init__(quiet=True)
        self.seconds, self.until, self.now = seconds, -1.0, 0.0

    def speak(self, text, now=False):
        super().speak(text, now)
        self.until = self.now + self.seconds

    def saying(self):
        return self.now < self.until


def talk(g, frames, t0=1.0):
    out = []
    for i, f in enumerate(frames):
        g.sink.now = t = t0 + i / FPS
        n = len(g.sink.spoken)
        g.update(f(t) if callable(f) else f, now=t)
        out += [(round(t - t0, 2), w) for w in g.sink.spoken[n:]]
    return out


def test_a_new_phrase_waits_for_a_breath_after_the_last_one():
    g = Guidance(Talking(0.6), GuidanceConfig())
    g.set_option("combined")
    g.reset("mug", now=0.0)
    g.update(scene(hand=None), now=0.05)
    out = talk(g, [near(dx=0.15, dz=0.25)] * 3 + [near(dx=0.15, dy=0.15, dz=0.25)] * 40)
    assert [w for _, w in out] == ["right", "down and right"]
    assert out[1][0] - out[0][0] >= 0.6 + 0.3 - 0.04       # the phrase itself, then a third of a second of quiet


def test_left_straight_after_right_is_jitter_unless_the_hand_is_well_past():
    def wobble(t):                                          # 6 cm either side of the line, twice a second
        return near(dx=0.06 if int(t * 4) % 2 == 0 else -0.06, dz=0.25)
    g = guide(lead_s=0.0)
    words = [w for _, w in talk(g, [wobble] * 90)]
    turns = sum(1 for a, b in zip(words, words[1:]) if {a, b} == {"left", "right"})
    assert turns <= 2                                       # 10 Oct: ten such turns under a second apart
    g2 = guide(lead_s=0.0)
    out = talk(g2, [near(dx=0.15, dz=0.25)] * 3 + [near(dx=-0.15, dz=0.25)] * 30)
    assert [w for _, w in out] == ["right", "left"] and out[1][0] < 1.0     # a real overshoot is corrected at once


def test_forward_with_no_progress_becomes_lean_forward():
    # 10 Oct, first reach: the bottle 24 cm beyond a fully stretched arm, and "forward" seven times.
    g = guide(lead_s=0.0)
    words = [w for _, w in talk(g, [near(dz=0.24)] * (9 * FPS))]
    assert words[:2] == ["aligned", "forward"] and "lean forward" in words
    assert words.index("lean forward") <= 3
    g2 = guide(lead_s=0.0)
    words = [w for _, w in talk(g2, [lambda t: near(dz=0.60 - 0.03 * (t - 1.0))] * (9 * FPS))]
    assert words == ["aligned", "forward"]                  # a hand that is getting there is left alone: no reminders


def test_a_depth_that_jumps_is_not_steered_by():
    g = guide(lead_s=0.2)
    frames = [over(dz=0.24)] * 30 + [over(dz=-0.12)] * 8 + [over(dz=0.24)] * 30
    words = [w for _, w in talk(g, frames)]
    assert "back" not in words                              # lidar read past the finger: the hand did not move


@pytest.mark.parametrize("seed", range(3))
def test_default_guidance_is_calm_on_a_jittery_slow_reach(seed):
    rng = np.random.default_rng(seed)
    g = Guidance(Talking(0.6), GuidanceConfig())
    g.set_option("combined")
    g.reset("mug", now=0.0)
    g.update(scene(hand=None), now=0.05)
    recent = []

    def frame(t):
        nonlocal recent
        recent = (recent + [rng.normal(0.0, 0.006, 2)])[-3:]
        jx, jy = np.mean(recent, axis=0)
        return near(dx=max(0.035, 0.25 - 0.08 * (t - 1.0)) + jx, dy=jy, dz=0.25 + (0.04 if rng.random() < 0.3 else 0.0))
    out = talk(g, [frame] * (8 * FPS))
    starts = [t for t, _ in out]
    assert len(out) <= 8                                    # at most a phrase a second
    assert sum(b - a < 0.8 for a, b in zip(starts, starts[1:])) <= 1


# --- what guidance writes to the trace ----------------------------------------------------------------------------

def test_every_guided_frame_records_what_explains_a_grab(tmp_path, monkeypatch):
    tr = tracing.Tracer(tmp_path, enabled=True)
    monkeypatch.setattr(tracing, "_default", tr)
    monkeypatch.setenv("GUIDANCE_AUDIO", "off")
    monkeypatch.setenv("GUIDANCE_MODE", "3")
    monkeypatch.delenv("GUIDANCE_GRAB", raising=False)
    monkeypatch.setattr(guidance, "_default", None)
    monkeypatch.setattr(guidance, "_logged_state", None)
    guidance.reset("mug")
    clock = [100.0]
    monkeypatch.setattr(guidance.time, "monotonic", lambda: clock[0])
    for i in range(40):                                     # in line and 30 cm short, then 8 cm short
        clock[0] = 100.0 + i / FPS
        guidance.update(near(dx=0.01, dy=0.0, dz=0.30 if i < 6 else 0.08, t=float(i)))
    guidance.set_grab_option(2)
    tr.close()
    monkeypatch.setattr(guidance, "_default", None)
    records = [json.loads(line) for line in open(tr.path)]
    frames = [r for r in records if r["type"] == "frame"]
    assert frames[3]["lined"] == 3 and frames[3]["zone"] > 1 and frames[3]["near_s"] is None
    last = frames[26]                                       # the jump forward is not believed for half a second
    assert frames[8]["trusted"] == 0 and frames[8]["near_s"] is None
    assert last["option"] == "combined" and last["grab_option"] == "over" and last["state"] == "guiding"
    assert last["err"] == [10, 0, 80] and len(last["lead"]) == 3            # millimetres
    assert 0 < last["zone"] < 1 and last["trusted"] == 1 and last["near_s"] > 0.05
    assert "speed" in last
    assert any(r["state"] == "grab" for r in frames)
    assert len(json.dumps(last)) < 400                      # a few numbers a frame, not a dump
    events = [r for r in records if r["type"] == "event"]
    assert any(e["name"] == "state" and e.get("grab_option") == "over" for e in events)
    assert any(e["name"] == "grab_option" and e["option"] == "zone" and e["number"] == 2 for e in events)
    meta = {k: v for r in records if r["type"] == "meta" for k, v in r.items()}
    assert meta["option"] == "combined" and meta["grab_option"] == "over"


def test_frames_without_a_hand_record_only_the_state_and_the_options(tmp_path, monkeypatch):
    tr = tracing.Tracer(tmp_path, enabled=True)
    monkeypatch.setattr(tracing, "_default", tr)
    monkeypatch.setenv("GUIDANCE_AUDIO", "off")
    monkeypatch.setattr(guidance, "_default", None)
    guidance.reset("mug")
    guidance.update(scene(hand=None, t=1.0))
    guidance.update(scene(hand=None, t=2.0))
    tr.close()
    monkeypatch.setattr(guidance, "_default", None)
    frame = [json.loads(line) for line in open(tr.path) if '"frame"' in line][0]
    assert frame["state"] == "waiting_hand" and "err" not in frame and "grab_option" in frame


# --- measured phrases are built from halves ------------------------------------------------------------------------

def test_a_measured_pair_is_made_from_its_two_halves(monkeypatch):
    made = []

    def fake(text, sr):
        made.append(text)
        return np.full(1000, 0.1, np.float32)
    monkeypatch.setattr(audio_out, "_synthesize", fake)
    monkeypatch.setattr(audio_out, "_spoken_cache", {})
    a = audio_out.synthesize("right twenty, up ten", 48000)
    assert made == ["right twenty", "up ten"] and len(a) == 2000 + int(0.07 * 48000)
    audio_out.synthesize("up ten, right twenty", 48000)
    assert len(made) == 2                                   # both halves were kept: nothing new to make
    audio_out.synthesize("I can see a mug, a cup and a box.", 48000)
    assert made[-1] == "I can see a mug, a cup and a box."  # an ordinary sentence is left whole


# --- the tool that renders them to listen to --------------------------------------------------------------------------

def demo():
    spec = importlib.util.spec_from_file_location("audio_demo", Path(__file__).resolve().parent.parent / "tools" / "audio_demo.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_made_up_reach_is_as_rough_as_the_live_one_and_still_ends_in_grab():
    d = demo()
    seen = d.Seen(0)
    scenes = [seen.scene(d.scripted(i / 30)[0], i / 30, d.scripted(i / 30)[1]) for i in range(100, 360)]
    hands = [s.hand_xyz for s in scenes]
    assert 3 < sum(h is None for h in hands) < 60           # the tracker loses the hand now and then
    steps = [abs(b[2] - a[2]) for a, b in zip(hands, hands[1:]) if a is not None and b is not None]
    assert max(steps) > 0.02 and np.median(steps) < 0.01    # depth flips between its two sources
    for key, _, _ in guidance.OPTIONS:
        _, said_, states, grab_t = d.render(key, sound=False)
        assert grab_t is not None and 9.0 < grab_t < 13.5, key
    assert set(d.CUES) == {k for k, _, _ in guidance.OPTIONS} and all(len(v) == 4 for v in d.CUES.values())


def test_the_old_rule_is_later_than_the_new_ones_on_the_made_up_reach():
    d = demo()
    when = {key: [d.render("combined", key, seed, sound=False)[3] for seed in range(4)] for key in ALL_GRABS}
    assert all(t is not None for key in PATIENT for t in when[key])
    old = [t for t in when["fingertip"] if t is not None]
    assert not old or np.median(old) > np.median(when["over"])


def test_the_page_lists_every_option_with_a_player(tmp_path):
    d = demo()
    rows = [(i + 1, title, text, d.CUES[key], f"{i + 1:02d}_{key}.wav") for i, (key, title, text) in enumerate(guidance.OPTIONS)]
    grabs = [(i + 1, title, text, None, f"grab_{i + 1}_{key}.wav") for i, (key, title, text) in enumerate(guidance.GRAB_OPTIONS)]
    d.page(tmp_path, rows, grabs, "SIMULATION: table <here>")
    page = (tmp_path / "index.html").read_text()
    assert page.count("<audio") == len(guidance.OPTIONS) + len(guidance.GRAB_OPTIONS)
    assert "07_echo.wav" in page and "grab_3_over.wav" in page and "table &lt;here&gt;" in page
    assert "ping and echo" in page and "left / right" in page
