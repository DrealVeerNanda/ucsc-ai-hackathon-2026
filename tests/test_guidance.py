"""Lane 3: the sound mapping, the guidance states and their timeouts, the audio mixer, and
pulling the object name out of a spoken request."""
import numpy as np
import pytest

import audio_out
import guidance
import voice
from audio_out import Mixer, PrintSink, ToneConfig, ToneSynth
from guidance import Guidance, GuidanceConfig, tone_for, word_for
from interfaces import Scene

CFG, TONE = GuidanceConfig(), ToneConfig()
OBJ = np.array([0.05, 0.12, 0.60])


def scene(hand=(-0.15, 0.05, 0.40), obj=OBJ, t=0.0):
    hand = None if hand is None else np.array(hand, float)
    return Scene(object_xyz=None if obj is None else np.array(obj, float), hand_xyz=hand,
                 object_visible=obj is not None, hand_visible=hand is not None,
                 object_box=None, hand_px=None, t=t)


def near(dx=0.0, dy=0.0, dz=0.0):
    """A hand at the object, offset by the given error (object minus hand)."""
    return scene(hand=OBJ - np.array([dx, dy, dz]))


def guide(**kw):
    g = Guidance(PrintSink(quiet=True), GuidanceConfig(**kw))
    g.reset("mug", now=0.0)
    return g


# --- sound mapping -------------------------------------------------------------

def test_pan_follows_the_side_the_hand_should_move_to():
    assert tone_for((0.15, 0, 0.2), CFG, TONE)[0] > 0.5        # object to the right -> right ear
    assert tone_for((-0.15, 0, 0.2), CFG, TONE)[0] < -0.5
    assert tone_for((0.5, 0, 0), CFG, TONE)[0] == 1.0          # never past full pan


def test_pitch_rises_for_up_and_falls_for_down():
    level = tone_for((0, 0, 0.3), CFG, TONE)[1]
    up = tone_for((0, -0.15, 0.3), CFG, TONE)[1]               # camera y is down: object above = negative y
    down = tone_for((0, 0.15, 0.3), CFG, TONE)[1]
    assert level == pytest.approx(TONE.base_hz) and up > level > down
    assert tone_for((0, -1.0, 0), CFG, TONE)[1] == pytest.approx(2 * TONE.base_hz)   # one octave at most


def test_beeps_get_faster_as_the_hand_gets_closer():
    rates = [tone_for((0, 0, d), CFG, TONE)[2] for d in (0.8, 0.5, 0.3, 0.1, 0.05)]
    assert rates == sorted(rates) and rates[0] == rates[1] == TONE.rate_far
    assert rates[-1] == pytest.approx(TONE.rate_near)


def test_dead_zone_is_silent_about_small_errors_and_has_no_jump():
    assert tone_for((0.029, -0.029, 0.3), CFG, TONE)[:2] == (0.0, TONE.base_hz)
    assert 0 < tone_for((0.031, 0, 0.3), CFG, TONE)[0] < 0.02   # ramps up from zero at the edge


def test_fallback_word_names_the_biggest_error():
    assert word_for(np.array([0.10, 0.02, 0.05]), CFG) == "right"
    assert word_for(np.array([-0.10, 0.02, 0.05]), CFG) == "left"
    assert word_for(np.array([0.01, -0.12, 0.05]), CFG) == "up"
    assert word_for(np.array([0.01, 0.12, 0.05]), CFG) == "down"
    assert word_for(np.array([0.01, 0.02, 0.20]), CFG) == "forward"
    assert word_for(np.array([0.01, 0.02, -0.20]), CFG) == "back"
    assert word_for(np.array([0.01, 0.02, 0.02]), CFG) is None


# --- states --------------------------------------------------------------------

def test_idle_until_a_target_is_set():
    g = Guidance(PrintSink(quiet=True))
    assert g.update(scene()) == "idle" and g.sink.tone[3] == "silent"
    g.reset("mug", now=0.0)
    assert g.sink.spoken == ["looking"]


def test_searching_then_found_then_waiting_for_the_hand():
    g = guide()
    assert g.update(scene(obj=None, hand=None), now=0.1) == "searching"
    assert g.update(scene(hand=None), now=0.2) == "waiting_hand"
    assert g.sink.spoken == ["looking", "found_it", "show_me_your_hand"]
    assert g.sink.tone[3] == "silent"
    g.update(scene(hand=None), now=3.0)
    assert g.sink.spoken.count("show_me_your_hand") == 1        # not nagging
    g.update(scene(hand=None), now=6.5)
    assert g.sink.spoken.count("show_me_your_hand") == 2        # but it does remind


def test_says_it_cannot_see_the_object_once():
    g = guide()
    for t in (1.0, 5.0, 9.0, 12.0, 30.0):
        assert g.update(scene(obj=None, hand=None), now=t) == "searching"
    assert g.sink.spoken == ["looking", "i_cant_see_that"]


def test_guiding_plays_a_tone_toward_the_object():
    g = guide()
    assert g.update(scene(), now=0.1) == "guiding"
    pan, freq, rate, mode = g.sink.tone
    assert mode == "beep" and pan == 1.0 and freq < TONE.base_hz     # object is right of and below the hand


def test_grab_needs_the_hand_to_stay_close_then_goes_idle():
    g = guide()
    g.update(scene(), now=0.1)
    assert g.update(near(dz=0.03), now=1.00) == "guiding"
    assert g.sink.tone[3] == "steady" and g.sink.tone[1] == TONE.steady_hz
    assert g.update(near(dz=0.03), now=1.20) == "guiding"          # not held long enough yet
    assert g.update(near(dz=0.03), now=1.35) == "grab"
    assert g.sink.spoken[-1] == "grab" and g.sink.tone[3] == "silent"
    assert g.update(near(dz=0.03), now=2.0) == "grab"
    assert g.update(near(dz=0.03), now=2.9) == "idle"              # main.py now asks for the next target
    assert g.update(near(dz=0.03), now=3.0) == "idle" and g.sink.spoken.count("grab") == 1


def test_one_close_frame_does_not_trigger_grab():
    g = guide()
    g.update(near(dz=0.03), now=1.0)
    g.update(near(dz=0.20), now=1.1)                                # jumped away again
    assert g.update(near(dz=0.03), now=1.35) == "guiding"           # the clock restarted
    assert "grab" not in g.sink.spoken


def test_jitter_at_the_edge_of_the_grab_radius_still_grabs():
    g = guide()
    for i, d in enumerate((0.048, 0.055, 0.047, 0.060, 0.049)):     # in and slightly out
        state = g.update(near(dz=d), now=1.0 + 0.1 * i)
    assert state == "grab"


def test_hand_blink_holds_the_sound_then_mutes_then_waits():
    g = guide()
    g.update(scene(), now=1.0)
    tone = g.sink.tone
    assert g.update(scene(hand=None), now=1.3) == "guiding" and g.sink.tone == tone     # under 0.5 s: hold
    assert g.update(scene(hand=None), now=1.7) == "guiding" and g.sink.tone[3] == "silent"   # stale: mute
    assert g.update(scene(hand=None), now=2.1) == "waiting_hand"
    assert g.sink.spoken[-1] == "show_me_your_hand"
    assert g.update(scene(), now=2.2) == "guiding" and g.sink.tone[3] == "beep"


def test_losing_the_object_says_so_and_searches_again():
    g = guide()
    g.update(scene(), now=1.0)
    assert g.update(scene(obj=None), now=1.1) == "searching"
    assert g.sink.spoken[-1] == "i_lost_it" and g.sink.tone[3] == "silent"
    assert g.update(scene(), now=1.2) == "guiding" and g.sink.spoken[-1] == "found_it"


def test_words_mode_speaks_at_most_twice_a_second_and_plays_no_tone():
    g = guide(words=True)
    for i in range(11):                                             # 1 s of frames
        g.update(scene(), now=1.0 + 0.1 * i)
    words = [w for w in g.sink.spoken if w in ("left", "right", "up", "down", "forward", "back")]
    assert words == ["right", "right", "right"] and g.sink.tone[3] == "silent"


def test_reset_starts_a_fresh_search():
    g = guide()
    g.update(scene(), now=1.0)
    g.reset("bottle", now=2.0)
    assert g.state == "searching" and g.target == "bottle" and g.sink.spoken[-1] == "looking"
    assert g.update(scene(), now=2.1) == "guiding" and g.sink.spoken[-1] == "found_it"


def test_module_interface_matches_main(monkeypatch):
    monkeypatch.setenv("GUIDANCE_AUDIO", "off")
    monkeypatch.setattr(guidance, "_default", None)
    assert guidance.update(scene()) == "idle"
    guidance.reset("mug")
    assert guidance.update(scene()) == "guiding"


# --- audio ---------------------------------------------------------------------

def _mixer(**clips):
    return Mixer(ToneConfig(), clips={k: np.full(v, 0.5, np.float32) for k, v in clips.items()})


def test_tone_is_click_free_and_pans():
    s = ToneSynth()
    s.set(1.0, 600, 8, "beep")
    out = np.concatenate([s.render(256) for _ in range(200)])       # about 1 s
    assert np.abs(np.diff(out, axis=0)).max() < 0.05                # no jumps between blocks
    settled = out[24000:]                                           # the pan glides there, it does not jump
    assert np.abs(settled[:, 1]).max() > 0.2 and np.abs(settled[:, 0]).max() < 0.005   # right ear only
    s.set(0.0, 880, 8, "steady")
    out = np.concatenate([s.render(256) for _ in range(100)])
    assert np.abs(out[-4800:, 0]).max() == pytest.approx(np.abs(out[-4800:, 1]).max(), rel=0.01)


def test_beep_rate_matches_the_request():
    s = ToneSynth()
    s.set(0.0, 500, 10, "beep")
    out = np.concatenate([s.render(256) for _ in range(375)])       # 2 s
    loud = np.abs(out[:, 0]) > 0.02
    env = np.convolve(loud.astype(float), np.ones(200) / 200, "same") > 0.05
    starts = np.count_nonzero(env[1:] & ~env[:-1])
    assert 18 <= starts <= 21                                       # 10 beeps a second for 2 s


def test_silent_means_silent():
    m = _mixer()
    m.set_tone(0, 500, 5, "silent")
    assert np.abs(np.concatenate([m.render(256) for _ in range(50)])).max() < 1e-3


def test_tone_goes_quiet_when_nobody_updates_it(monkeypatch):
    m = _mixer()
    m.set_tone(0, 500, 5, "steady")
    for _ in range(40):
        m.render(256)
    assert np.abs(m.render(256)).max() > 0.1
    t = audio_out.time.monotonic()
    monkeypatch.setattr(audio_out.time, "monotonic", lambda: t + 2.0)   # the loop has stalled for 2 s
    out = np.concatenate([m.render(256) for _ in range(80)])
    assert np.abs(out[-2000:]).max() < 1e-3


def test_clips_play_in_order_and_now_cuts_in():
    m = _mixer(a=1000, b=1000, c=1000)
    m.say("a")
    m.say("b")                # waits for a
    m.say("c")                # replaces b before it started
    assert m.saying()
    while m.saying():
        m.render(256)
    assert m.spoken == ["a", "c"]
    m.say("a")
    m.render(256)
    m.say("b", now=True)      # cuts a off
    assert m.spoken[-2:] == ["a", "b"]


def test_clip_is_heard_over_a_ducked_tone():
    m = _mixer(a=4800)
    m.set_tone(0, 500, 5, "steady")
    for _ in range(40):
        m.render(256)
    before = np.abs(m.render(256)).max()
    m.say("a")
    during = np.concatenate([m.render(256) for _ in range(10)])
    assert during.max() > before                                    # the clip is added on top
    assert m.cfg.duck < 1.0


def test_missing_clip_does_not_break_playback(capsys):
    m = _mixer()
    m.say("nope")
    m.render(256)
    assert not m.saying() and "no clip" in capsys.readouterr().err


def test_real_clips_exist_for_every_phrase_guidance_uses():
    clips = audio_out.load_clips()
    needed = {"looking", "found_it", "show_me_your_hand", "grab", "i_lost_it", "i_cant_see_that",
              "left", "right", "up", "down", "forward", "back"}
    assert needed <= set(clips)
    assert all(0.2 < len(c) / 48000 < 2.0 for c in clips.values())


# --- voice ---------------------------------------------------------------------

@pytest.mark.parametrize("said, target", [
    ("Find my mug.", "mug"),
    ("find the red mug please", "red mug"),
    ("Where's my water bottle?", "water bottle"),
    ("Can you grab me a banana, please?", "banana"),
    ("pick up the tennis ball for me", "tennis ball"),
    ("I need my phone now please", "phone"),
    ("mug", "mug"),
    ("  The Cup  ", "cup"),
    ("", ""),
    ("please", ""),
])
def test_extract_target(said, target):
    assert voice.extract_target(said) == target


def test_typed_target_goes_through_the_same_cleanup(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt="": "find my Mug")
    monkeypatch.setenv("VOICE_MODE", "type")
    monkeypatch.setattr(voice, "_mode", None)
    assert voice.listen_for_target() == "mug"
