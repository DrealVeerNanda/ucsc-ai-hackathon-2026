"""Voice on stage: what a blindfolded wearer hears when the table looks empty or nobody speaks, the
words that have to just work, telling the wearer from the room, and every failure it must live
through (run with: python -m pytest tests/test_voice_night.py)."""
import json
import socket
import ssl
import sys
import threading
import time
import urllib.request
import wave
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import cosmos
import guidance
import stream
import voice
from interfaces import Frame

SR = voice.SR


@pytest.fixture
def quiet(monkeypatch, tmp_path):
    monkeypatch.setenv("VOICE_LLM_URL", "off")
    monkeypatch.setenv("VOICE_AUTO_LIST", "1")
    monkeypatch.setenv("GUIDANCE_AUDIO", "off")
    monkeypatch.setenv("VOICE_STATUS", str(tmp_path / "status" / "voice.json"))
    for name in ("GUIDANCE_MODE", "VOICE_WEARER_LEVEL", "VOICE_REQUEST_S", "VOICE_HEADSET_MODE", "VOICE_HEADSET_VOLUME",
                 "VOICE_DOWNLOAD", "VOICE_STT", "VOICE_REMIND", "VOICE_MODE", "VOICE_DEVICE", "VOICE_HEADSET"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(guidance, "_default", None)
    for name, value in (("_ask_next", True), ("last_objects", []), ("_llm_ok", None), ("_llm_again_t", 0.0), ("_stopped", False),
                        ("_told_no_cosmos", False), ("_said_t", 0.0), ("_reminders", 0), ("_last_said", ""), ("_trouble", {}),
                        ("_problems", {}), ("_switch_t", -1e9), ("_repair_t", -1e9), ("_stereo_volume", None), ("_listener", None),
                        ("_mode", None), ("_failures", 0), ("_no_keyboard", False), ("_status_thread", object()), ("_left_open", [])):
        monkeypatch.setattr(voice, name, value)
    monkeypatch.setattr(voice, "_status", dict(voice._status, state="starting", heard="", meant="", error="", error_t=None))
    monkeypatch.setattr(voice, "_poke_main", lambda: None)
    monkeypatch.setattr(voice, "_beep", lambda: beeps.append(len(guidance._get().sink.log)))
    monkeypatch.setattr(voice, "FIND_EARBUDS_S", 0.05)
    beeps = []
    monkeypatch.beeps = beeps
    yield monkeypatch
    for mic in _mics:                                       # a test that failed half way: end its wait before the fakes go away
        mic.submit("the mug")
    for thread in _threads:
        thread.join(3)
    del _mics[:], _threads[:]


def said():
    return list(guidance._get().sink.log)


class Camera:
    """The stream main.py uses: always a fresh picture, with or without depth."""

    def __init__(self, depth=None, dark=0):
        self.depth, self.dark = depth, dark              # dark: how many calls give no picture first

    def get_frame(self, wait=0.0):
        if self.dark > 0:
            self.dark -= 1
            return None
        now = time.monotonic()
        return Frame(np.zeros((64, 48, 3), np.uint8), self.depth, None, None, now, t_recv=now)


def looking(monkeypatch, *lists, camera=None):
    """Cosmos answers with each list in turn, then keeps giving the last one. Returns the calls made."""
    calls, answers = [], list(lists)
    monkeypatch.setattr(stream, "_default", camera or Camera())
    monkeypatch.setattr(cosmos, "available", lambda *a, **k: True)

    def list_objects(rgb, **k):
        calls.append(time.monotonic())
        answer = answers.pop(0) if len(answers) > 1 else answers[0]
        if isinstance(answer, Exception):
            raise answer
        return list(answer)
    monkeypatch.setattr(cosmos, "list_objects", list_objects)
    return calls


def pace(monkeypatch, look=0.05, nothing=(60, 60, 60), silence=(60, 60, 60), patience=2.0):
    monkeypatch.setattr(voice, "LOOK_EVERY_S", look)
    monkeypatch.setattr(voice, "LOOK_SLOWER_S", look)
    monkeypatch.setattr(voice, "LOOK_PATIENCE_S", patience)
    monkeypatch.setattr(voice, "NOTHING_REMINDERS_S", nothing)
    monkeypatch.setattr(voice, "SILENCE_REMINDERS_S", silence)


class Buds:
    """Earbuds that follow a script: each record() gives the next step. None is nobody speaking,
    an exception is raised, ("words", level, seconds) is a recording that Whisper hears as those words."""
    name, device, headset, stereo = "AirPods Pro", 61, 261, 5

    def __init__(self, *script):
        self.script, self.modes, self.calls, self.words = list(script), [], 0, {}

    def open(self):
        self.modes.append("headset")

    def close(self):
        self.modes.append("stereo")

    def mode(self):
        return self.headset

    def record(self, should_stop=None, abandon=None):
        self.calls += 1
        step = self.script.pop(0) if self.script else None
        if isinstance(step, Exception):
            raise step
        if step is None:                                  # wait like the real one, until there is a reason to stop
            end = time.monotonic() + 0.1
            while time.monotonic() < end and not (should_stop and should_stop()):
                time.sleep(0.005)
            return None
        text, level, seconds = (step + (0.05, 1.5))[:3] if len(step) < 3 else step
        audio = np.full(int(seconds * SR), level, np.float32)
        self.words[audio.size, round(float(level), 4)] = text
        return audio

    def hear(self, audio):
        return self.words[audio.size, round(float(audio[0]), 4)]


_mics, _threads = [], []


def web_mic(tmp_path, buds=None, hear=None, **more):
    mic = voice.WebMic(port=0, transcribe=lambda data: "", cert_dir=tmp_path / "cert", typed=False, headset=buds,
                       hear=hear or (buds.hear if buds is not None else None), **more)
    _mics.append(mic)
    return mic


def listen_in_background(mic):
    out = {}
    thread = threading.Thread(target=lambda: out.setdefault("target", mic.listen()), daemon=True)
    thread.start()
    _threads.append(thread)
    return out, thread


def until(condition, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.01)
    return False


# --- nothing on the table ----------------------------------------------------------

EMPTY = "I can't see anything to pick up. Turn towards the table, or step back a little."


def test_an_empty_table_is_explained_then_watched_until_something_is_there(quiet, tmp_path):
    pace(quiet)
    calls = looking(quiet, [], [], [], ["mug", "phone"])
    buds = Buds()
    mic = web_mic(tmp_path, buds)
    try:
        out, thread = listen_in_background(mic)
        assert until(lambda: len(said()) >= 2)
        # it said why and what to do, looked again by itself, and spoke the list the moment there was one
        assert said() == [EMPTY + " I'll keep looking.", "I can see mug and phone. What do you want to grab?"]
        assert len(calls) == 4 and buds.calls >= 1          # the microphone was open while it looked
        assert until(lambda: len(quiet.beeps) == 2) and quiet.beeps == [1, 2]      # "your turn" after each thing said, and only then
        mic.submit("the phone")
        thread.join(3)
        assert out["target"] == "phone"
        time.sleep(0.2)
        assert len(calls) == 4                              # found: it stopped looking
    finally:
        mic.close()


def test_reminders_about_an_empty_table_are_few_and_far_between(quiet, tmp_path):
    pace(quiet, look=0.02, nothing=(0.15, 0.3, 60))
    calls = looking(quiet, [])
    mic = web_mic(tmp_path, Buds())
    try:
        out, thread = listen_in_background(mic)
        time.sleep(1.2)
        assert said() == [EMPTY + " I'll keep looking.",
                          "Still nothing to pick up. Turn towards the table or step back, or tell me what you want.",
                          "Still looking."]                 # three sentences in all, however often it looked
        assert len(calls) > 10
        mic.submit("water bottle")                          # naming something always works, list or no list
        thread.join(3)
        assert out["target"] == "water bottle" and len(said()) == 3
        looks = len(calls)
        time.sleep(0.2)
        assert len(calls) <= looks + 1                      # ...and ends the looking
    finally:
        mic.close()


def test_something_in_front_of_the_camera_is_said_as_such(quiet, tmp_path):
    pace(quiet)
    depth = np.full((64, 48), 1.2, np.float32)
    depth[:16] = 0.15                                       # a forearm across a quarter of the picture
    looking(quiet, [], camera=Camera(depth))
    mic = web_mic(tmp_path, Buds())
    try:
        out, thread = listen_in_background(mic)
        assert until(lambda: said())
        assert said()[0] == "Something is right in front of the camera. Move your arm away, or step back a little. I'll keep looking."
        mic.submit("mug")
        thread.join(3)
    finally:
        mic.close()


def test_only_measured_closeness_counts_as_covered():
    far = np.full((192, 256), 1.5, np.float32)
    assert not voice.covered(None) and not voice.covered(far)
    assert not voice.covered(np.zeros((192, 256), np.float32))      # no reading at all: a far wall does that too
    a_little = far.copy()
    a_little[:8] = 0.2                                              # 4%: the table's near edge
    assert not voice.covered(a_little)
    arm = far.copy()
    arm[:40] = 0.2                                                  # 21%
    assert voice.covered(arm)
    arm[:40] = np.nan
    assert not voice.covered(arm)


def test_no_picture_yet_is_said_once_and_the_list_follows_when_the_camera_wakes(quiet, tmp_path):
    pace(quiet)
    looking(quiet, ["mug", "phone"], camera=Camera(dark=30))
    quiet.setattr(voice, "_camera_frame", lambda wait_s=2.0: stream._default.get_frame())      # without the two-second wait
    mic = web_mic(tmp_path, Buds())
    try:
        out, thread = listen_in_background(mic)
        assert until(lambda: len(said()) >= 2)
        assert said() == ["I have no picture from the camera yet. I'll keep looking.", "I can see mug and phone. What do you want to grab?"]
        mic.submit("mug")
        thread.join(3)
        assert out["target"] == "mug"
    finally:
        mic.close()


def test_cosmos_down_is_said_once_and_then_it_simply_asks(quiet, tmp_path):
    pace(quiet)
    quiet.setattr(stream, "_default", Camera())
    quiet.setattr(cosmos, "available", lambda *a, **k: False)
    mic = web_mic(tmp_path, Buds())
    try:
        for want in ("I can't look around right now. Tell me what you want to grab.", "What do you want to grab?"):
            out, thread = listen_in_background(mic)
            assert until(lambda: said() and said()[-1] == want)
            mic.submit("the mug")
            thread.join(3)
            assert out["target"] == "mug"
        assert len(said()) == 2
    finally:
        mic.close()


@pytest.mark.parametrize("fault", [TimeoutError("timed out"), OSError("connection refused"), ValueError("bad json")])
def test_cosmos_failing_does_not_stop_the_listening(quiet, tmp_path, fault):
    pace(quiet)
    looking(quiet, fault)
    mic = web_mic(tmp_path, Buds(None, ("water bottle",)))
    try:
        assert mic.listen() == "water bottle"
        assert said() == ["I can't look around right now. Tell me what you want to grab."]
        assert "cosmos" in voice.status()["error"]
    finally:
        mic.close()


def test_a_slow_look_does_not_leave_the_wearer_in_silence(quiet, tmp_path):
    pace(quiet, patience=0.1)
    quiet.setattr(stream, "_default", Camera())
    quiet.setattr(cosmos, "available", lambda *a, **k: True)
    quiet.setattr(cosmos, "list_objects", lambda rgb, **k: time.sleep(0.5) or ["mug", "phone"])
    buds = Buds()
    mic = web_mic(tmp_path, buds)
    try:
        out, thread = listen_in_background(mic)
        assert until(lambda: len(said()) >= 2)
        assert said() == ["What do you want to grab?", "I can see mug and phone. What do you want to grab?"]
        assert buds.calls >= 1                              # it was already listening while Cosmos was thinking
        mic.submit("mug")
        thread.join(3)
    finally:
        mic.close()


def test_asking_what_is_there_restarts_the_watching(quiet, tmp_path):
    pace(quiet)
    quiet.setenv("VOICE_AUTO_LIST", "0")                    # nothing said first: the wearer asks
    calls = looking(quiet, [], [], ["mug", "phone"])
    quiet.setenv("VOICE_AUTO_LIST", "1")
    quiet.setattr(voice, "_ask_next", False)
    mic = web_mic(tmp_path, Buds(("what is on the table",)))
    try:
        out, thread = listen_in_background(mic)
        assert until(lambda: len(said()) >= 2)
        assert said() == [EMPTY, "I can see mug and phone. What do you want to grab?"] and len(calls) == 3
        mic.submit("mug")
        thread.join(3)
    finally:
        mic.close()


# --- nobody says anything -----------------------------------------------------------

def test_a_long_silence_gets_a_short_reminder_not_the_list_again(quiet, tmp_path):
    pace(quiet, silence=(0.15, 0.3, 60))
    calls = looking(quiet, ["mug", "phone"])
    mic = web_mic(tmp_path, Buds())
    try:
        out, thread = listen_in_background(mic)
        time.sleep(1.2)
        assert said() == ["I can see mug and phone. What do you want to grab?",
                          "I'm listening. Say what you want to grab, or ask what is on the table.", "Still listening."]
        assert len(calls) == 1 and quiet.beeps == [1, 2, 3]
        mic.submit("again")                                 # "again" is the list, not the reminder
        assert until(lambda: len(said()) == 4) and said()[-1] == said()[0] and len(calls) == 1
        mic.submit("the phone")
        thread.join(3)
        assert out["target"] == "phone"
    finally:
        mic.close()


def test_reminders_can_be_switched_off(quiet, tmp_path):
    pace(quiet, silence=(0.05, 0.05, 0.05))
    quiet.setenv("VOICE_REMIND", "0")
    looking(quiet, ["mug", "phone"])
    mic = web_mic(tmp_path, Buds())
    try:
        out, thread = listen_in_background(mic)
        time.sleep(0.5)
        assert len(said()) == 1
        mic.submit("mug")
        thread.join(3)
    finally:
        mic.close()


# --- words that have to just work ------------------------------------------------------

def test_stop_ends_the_reach_and_goes_back_to_asking(quiet):
    guidance.reset("mug")
    guidance.pop_phrases()
    assert voice.respond("Stop.") == ""
    assert guidance.pop_phrases() == ["Stopped. What do you want to grab?"]
    assert voice.respond("option 2") == ""                  # the stopped reach does not come back with the next remark
    guidance.pop_phrases()
    assert voice.respond("the phone") == "phone"
    assert voice.respond("never mind") == "" and guidance.pop_phrases() == ["Stopped. What do you want to grab?"]


@pytest.mark.parametrize("words", ["cancel", "Never mind.", "okay stop", "stop it please", "forget it"])
def test_other_ways_of_saying_stop(quiet, words):
    assert voice.respond(words) == "" and guidance.pop_phrases() == ["Okay. What do you want to grab?"]


@pytest.mark.parametrize("words", ["we should stop by the store later", "please stop the car now we are done", "don't stop me now",
                                   "Thank you.", "you", "Thanks for watching!", "Oh.", "I'm sorry.", "Oh my God.", "Bye."])
def test_a_sentence_with_stop_in_it_and_whispers_noise_are_not_commands(quiet, words):
    guidance.reset("mug")
    guidance.pop_phrases()
    clean = voice._plain(words)
    assert not voice._STOP.match(clean) and not voice._AGAIN.match(clean) and not voice._HELP.match(clean)
    assert voice.switch_option(words) is None and guidance.pop_phrases() == []


def test_whispers_inventions_on_noise_never_reach_a_request():
    for words in ("Thank you.", " you", "Thanks for watching!", "Thank you for watching.", "Oh.", "I'm sorry.", "Oh my God.",
                  "Bye.", "So", "Subtitles by the Amara.org community", "Okay."):
        assert not voice.worth_hearing(words), words
    for words in ("stop", "again", "help", "blue cup", "option three"):
        assert voice.worth_hearing(words), words


def test_the_rule_does_not_take_a_whole_sentence_for_an_objects_name(quiet):
    assert voice.interpret("please stop the car now we are done") == ("none", "")       # Nemotron is off here
    assert voice.interpret("find my blue water bottle please") == ("pick", "blue water bottle")


def test_again_repeats_the_last_thing_said_and_help_explains(quiet):
    looking(quiet, ["mug", "phone"])
    assert voice.respond("what's on the table") == ""
    assert voice.respond("Again.") == "" and voice.respond("say that again please") == ""
    assert guidance.pop_phrases() == ["I can see mug and phone. Which one?"] * 3
    assert voice.respond("help") == "" and guidance.pop_phrases() == [voice.HELP]
    assert voice.HELP.count(".") == 2
    guidance.reset("mug")
    guidance.pop_phrases()
    assert voice.respond("repeat") == "mug"                 # in the middle of a reach it carries on with the same thing


# --- options -----------------------------------------------------------------------

@pytest.fixture
def families(quiet):
    hands = [(f"h{i}", f"hand way {i}", "") for i in range(1, 13)]
    grabs = [("near", "when it is close", ""), ("touch", "when it touches", "")]
    state = {"hand": 1, "grab": 1, "calls": []}

    def set_hand_option(n):
        state["calls"].append(("hand", n))
        state["hand"] = int(n)
        return hands[int(n) - 1][1]

    def set_grab_option(n):
        state["calls"].append(("grab", n))
        state["grab"] = int(n)                              # one that forgets to return the title

    quiet.setitem(sys.modules, "perception", SimpleNamespace(
        HAND_OPTIONS=hands, set_hand_option=set_hand_option, hand_option=lambda: (state["hand"], hands[state["hand"] - 1][0], hands[state["hand"] - 1][1])))
    quiet.setattr(guidance, "GRAB_OPTIONS", grabs, raising=False)
    quiet.setattr(guidance, "set_grab_option", set_grab_option, raising=False)
    quiet.setattr(guidance, "grab_option", lambda: (state["grab"], grabs[state["grab"] - 1][0], grabs[state["grab"] - 1][1]), raising=False)
    return state


@pytest.mark.parametrize("words, number", [("hand option two", 2), ("Hand option 12.", 12), ("hand option twelve", 12),
                                           ("hand option number seven", 7), ("switch to hand option 10 please", 10), ("hand option for", 4)])
def test_hand_options_are_switched_by_voice_up_to_twelve(families, words, number):
    assert voice.respond(words) == ""
    assert families["calls"] == [("hand", number)] and guidance.pop_phrases() == [f"Hand option {number}: hand way {number}."]


def test_grab_options_and_asking_which_is_on(families):
    assert voice.respond("grab option two") == "" and families["calls"] == [("grab", 2)]
    assert guidance.pop_phrases() == ["Grab option 2: when it touches."]
    for words, answer in (("what option is this", "Option 1: tones."), ("Which option?", "Option 1: tones."),
                          ("which hand option is this", "Hand option 1: hand way 1."), ("what's the grab option", "Grab option 2: when it touches.")):
        assert voice.respond(words) == "" and guidance.pop_phrases() == [answer]
    assert voice.respond("next hand option") == "" and voice.respond("previous hand option") == ""
    assert voice.respond("previous hand option") == "" and families["hand"] == 12           # wraps round
    assert voice.respond("previous option") == "" and guidance.option()[0] == 5


def test_a_number_that_does_not_exist_is_answered_plainly(families):
    assert voice.respond("option nine") == "" and guidance.pop_phrases() == ["There are 5 options."]
    assert voice.respond("grab option 3") == "" and guidance.pop_phrases() == ["There are 2 grab options."]
    assert voice.respond("hand option 13") == "" and guidance.pop_phrases() == ["There are 12 hand options."]
    assert voice.respond("option 0") == "" and guidance.option()[0] == 1 and families["calls"] == []


def test_options_that_do_not_exist_yet_or_break_do_not_break_voice(quiet):
    quiet.delitem(sys.modules, "perception", raising=False)
    quiet.delattr(guidance, "GRAB_OPTIONS", raising=False)
    assert voice.respond("hand option 2") == "" and guidance.pop_phrases() == ["Hand options are not available here."]
    assert voice.respond("grab option 2") == "" and guidance.pop_phrases() == ["Grab options are not available here."]

    def broken(n):
        raise RuntimeError("camera busy")
    quiet.setitem(sys.modules, "perception", SimpleNamespace(HAND_OPTIONS=[("a", "one", "")], set_hand_option=broken, hand_option=lambda: (1, "a", "one")))
    assert voice.respond("hand option 1") == "" and guidance.pop_phrases() == ["I could not switch that."]
    assert "option" in voice.status()["error"]


def test_the_page_is_told_every_family_of_options(families):
    told = voice.option_families()
    assert told["option"] == "1: tones" and told["options"]["guide"]["titles"][2] == "combined words"
    assert told["options"]["hand"] == {"now": 1, "titles": [f"hand way {i}" for i in range(1, 13)]}
    assert told["options"]["grab"]["titles"] == ["when it is close", "when it touches"]


# --- the wearer and the room ----------------------------------------------------------

def test_a_soft_voice_counts_only_when_it_answers_what_was_asked(quiet):
    quiet.setattr(voice, "last_objects", ["water bottle", "blue cup", "laptop"])
    soft, normal = 0.0054, 0.0317                           # the soft "blue cup" recording, and the quietest real request
    for words in ("Blue cup.", "the bottle", "the second one", "what is on the table"):
        assert voice.from_the_wearer(words, soft), words
    for words in ("You're welcome.", "How are you feeling?", "stop", "option two", "I've got to get on the phone."):
        assert not voice.from_the_wearer(words, soft), words
        assert voice.from_the_wearer(words, normal), words
    quiet.setattr(voice, "last_objects", [])                # nothing was listed: every voice is heard, as before
    assert voice.from_the_wearer("the green can", soft)
    quiet.setattr(voice, "last_objects", ["laptop"])
    quiet.setenv("VOICE_WEARER_LEVEL", "0")
    assert voice.from_the_wearer("You're welcome.", soft)


def test_long_talk_and_distant_talk_never_reach_whisper_or_nemotron(quiet, tmp_path):
    pace(quiet)
    looking(quiet, ["water bottle", "blue cup"])
    heard, read = [], []
    buds = Buds(("he sent an email that I did not spend a week on", 0.03, 12.0),       # a conversation next to the wearer
                ("you're welcome", 0.0124, 1.86),                                       # somebody further off
                ("blue cup", 0.0054, 3.2))                                              # the wearer, softly
    quiet.setattr(voice, "interpret", lambda text: read.append(text) or ("pick", "blue cup"))
    mic = web_mic(tmp_path, buds, hear=lambda audio: heard.append(audio.size / SR) or buds.hear(audio))
    try:
        assert mic.listen() == "blue cup"
        assert heard == [1.86, 3.2] and read == ["blue cup"]
        assert quiet.beeps == [1] and voice.status()["heard"] == "blue cup"
    finally:
        mic.close()


def _noise(*parts, seed=0, block_s=0.03):
    """parts: (seconds, level). 30 ms blocks of noise at each level, one after the other."""
    rng = np.random.default_rng(seed)
    n = int(SR * block_s)
    return [(rng.standard_normal(n) * level).astype(np.float32) for seconds, level in parts for _ in range(int(round(seconds / block_s)))]


def test_a_loud_moment_at_the_start_does_not_deafen_the_seconds_after_it():
    # The hall's speakers say a word just as the microphone opens; then the wearer, at the quietest level measured.
    blocks = _noise((0.3, 0.02), (1.5, 0.0005), (1.0, 0.04), (2.0, 0.0005))
    audio = voice.cut_speech(iter(blocks))
    assert audio is not None and 1.9 < audio.size / SR < 2.4
    history = deque(maxlen=333)                             # with the room's level carried over, even more so
    assert voice.cut_speech(iter(_noise((3.0, 0.0005))), history=history) is None
    assert voice.cut_speech(iter(_noise((0.6, 0.02), (1.0, 0.04), (2.0, 0.0005), seed=1)), history=history) is not None


def test_the_wearer_is_cut_out_of_the_talk_around_them():
    info = {}
    talk = (0.4, 0.012), (0.15, 0.002)                      # somebody nearby: words and short gaps, for a long time
    blocks = _noise(*(talk * 6), (1.0, 0.06), *(talk * 12))
    audio = voice.cut_speech(iter(blocks), info=info)
    assert audio is not None and 1.9 < audio.size / SR < 2.5 and info["ended"] == "quiet" and info["skipped_s"] > 2.5
    assert voice.loudness(audio) > 0.035                    # it is the wearer's second, not the 13 s of talk around it
    alone = voice.cut_speech(iter(_noise(*(talk * 18))), info=info)
    assert alone.size / SR > voice.REQUEST_MAX_S and info["ended"] == "max"     # the talk alone runs on, and is dropped unheard


def test_no_recording_runs_past_its_limit_and_the_hall_becomes_the_room(quiet):
    info, history = {}, deque(maxlen=333)
    hall = _noise((0.5, 0.0005), (30.0, 0.05))              # the speakers come on and stay on
    audio = voice.cut_speech(iter(hall), history=history, info=info)
    assert info["ended"] == "max" and voice.CLIP_MAX_S < audio.size / SR < voice.CLIP_MAX_S + 0.5
    assert voice.voice_level(history) > 0.05                # ...so the next window does not start on it again
    assert voice.cut_speech(iter(_noise((7.0, 0.05), seed=2)), history=history) is None
    assert voice.cut_speech(iter(_noise((2.0, 0.05), (1.0, 0.2), (2.0, 0.05), seed=3)), history=history) is not None   # a shout still gets through


def test_a_typed_request_does_not_wait_for_the_room_to_finish():
    seen = []
    out = voice.cut_speech(iter(_noise((0.5, 0.0005), (20.0, 0.05))), abandon=lambda: seen.append(1) or len(seen) > 30, info=(info := {}))
    assert out is None and info["ended"] == "abandoned" and len(seen) < 40


def test_how_loud_a_voice_must_be_follows_the_room():
    assert voice.voice_level([]) == 0.005 and voice.voice_level([0.0003] * 100) == 0.005        # earbuds in a quiet room
    assert voice.voice_level([0.004] * 100) == pytest.approx(0.010)                               # 2.5 times the room
    assert voice.voice_level([0.012] * 100) == pytest.approx(0.025)                               # a loud room: never beyond the wearer's own voice
    assert voice.voice_level([0.03] * 100) == pytest.approx(0.045)                                # louder still: 1.5 times
    assert voice.voice_level([0.0003] * 80 + [0.05] * 20) == 0.005                                # a loud word does not move it


MIC = [p for p in (Path(voice.ROOT) / "out",) if all((p / f"mic_test_{i}.wav").exists() for i in (1, 2, 3))]


@pytest.mark.skipif(not MIC, reason="the AirPods recordings are not in out/ (tools/mic_test.py makes them)")
def test_the_real_recordings_are_still_cut_right_also_under_hall_noise():
    def blocks(i, noise=0.0, seed=0):
        with wave.open(str(MIC[0] / f"mic_test_{i}.wav")) as w:
            a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
        a = np.concatenate([np.zeros(SR // 2, np.float32), a, np.zeros(SR, np.float32)])
        if noise:                                          # speakers in the hall: bursts of words over everything
            rng = np.random.default_rng(seed)
            gate = np.repeat(rng.random(a.size // 4800 + 1) < 0.7, 4800)[:a.size]
            a = a + (rng.standard_normal(a.size) * noise * gate).astype(np.float32)
        return [a[k:k + 480] for k in range(0, a.size - 479, 480)]
    soft, loud = voice.cut_speech(iter(blocks(1))), voice.cut_speech(iter(blocks(2)))
    assert soft is not None and 2.5 < soft.size / SR < 4.5                  # the soft "blue cup": all of it
    assert loud is not None and 2.3 < loud.size / SR < 3.2 and voice.loudness(loud) > 0.03     # without the murmur after it
    quiet_room = voice.cut_speech(iter(blocks(3)))
    assert quiet_room is None or voice.loudness(quiet_room) < voice.WEARER_LEVEL            # distant voices: not the wearer
    for seed in range(5):
        under_pa = voice.cut_speech(iter(blocks(2, noise=0.01, seed=seed)), info=(info := {}))
        assert under_pa is not None and info["ended"] == "quiet" and 2.3 < under_pa.size / SR < 3.6, (seed, info)
        assert voice.loudness(under_pa) > 0.03


# --- it never stops -------------------------------------------------------------------

def test_earbuds_that_drop_out_are_looked_for_and_used_again(quiet, tmp_path):
    pace(quiet)
    looking(quiet, ["mug", "phone"])
    first, second = Buds(voice.MicLost("no sound is arriving from the earbuds' microphone")), Buds(None, ("the mug",))
    finds = []
    quiet.setattr(voice.Headset, "find", staticmethod(lambda: finds.append(1) or (second if len(finds) > 3 else None)))
    mic = web_mic(tmp_path, first, hear=lambda audio: second.hear(audio))
    mic._find_headset = True
    try:
        out, thread = listen_in_background(mic)
        assert until(lambda: mic.headset is None)
        assert voice.status()["ok"] is False and "earbuds" in voice.status()["message"].lower()
        thread.join(5)
        assert out["target"] == "mug" and second.modes == ["headset", "stereo"]
        assert voice.status()["ok"] is True and "no sound is arriving" in voice.status()["error"]     # the last error stays readable
        assert quiet.beeps == [1, 1]                        # the beep again when they came back: the microphone is open
        assert 3 < len(finds) < 12                          # looked for now and then, not in a spin
    finally:
        mic.close()


def test_a_microphone_that_keeps_failing_is_tried_less_and_less_often(quiet, tmp_path):
    pace(quiet)
    quiet.setenv("VOICE_AUTO_LIST", "0")
    buds = Buds(*[RuntimeError("PortAudioError: Error opening InputStream")] * 50)
    quiet.setattr(voice.Headset, "find", staticmethod(lambda: buds))
    mic = web_mic(tmp_path, buds)
    mic._find_headset = True
    try:
        out, thread = listen_in_background(mic)
        time.sleep(1.0)
        assert 3 <= buds.calls <= 6                         # 0.05, 0.1, 0.2, 0.4 s apart...: never a busy loop
        assert voice.status()["ok"] is False
        mic.submit("the mug")                               # ...and typing still works meanwhile
        thread.join(3)
        assert out["target"] == "mug"
    finally:
        mic.close()


def test_earbuds_that_came_back_under_a_new_number_or_in_stereo_are_put_right(quiet, tmp_path):
    quiet.setattr(voice, "CHECK_EARBUDS_S", 0.0)
    old, new = Buds(), Buds()
    new.device = 77                                         # the sound server restarted: same earbuds, new number
    quiet.setattr(voice.Headset, "find", staticmethod(lambda: new))
    mic = web_mic(tmp_path, old)
    mic._find_headset = True
    try:
        mic._check()
        assert mic.headset is new and new.modes == ["headset"]
        new.mode = lambda: new.stereo                       # they reconnected and came back in stereo
        mic._check()
        assert new.modes == ["headset", "headset"]
        quiet.setattr(voice.Headset, "find", staticmethod(lambda: None))
        with pytest.raises(voice.MicLost):
            mic._check()
    finally:
        mic.close()


def test_whisper_failing_on_one_recording_is_said_and_survived(quiet, tmp_path):
    pace(quiet)
    quiet.setenv("VOICE_AUTO_LIST", "0")
    buds = Buds(("first", 0.05, 1.5), ("second", 0.05, 1.6), ("the mug", 0.05, 1.7))

    def hear(audio):
        if audio.size < 1.65 * SR:
            raise RuntimeError("CUDA out of memory")
        return buds.hear(audio)
    mic = web_mic(tmp_path, buds, hear=hear)
    try:
        assert mic.listen() == "mug"
        assert said() == ["I did not catch that. Say it again."]              # once, not after every failure
        assert "whisper: RuntimeError: CUDA out of memory" in voice.status()["error"] and voice.status()["ok"] is True
    finally:
        mic.close()


class _Reply:
    def __init__(self, content):
        self.body = json.dumps({"choices": [{"message": {"content": content}}]}).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, *a):
        return self.body


def test_nemotron_timing_out_is_left_alone_for_a_while_and_then_tried_again(quiet):
    quiet.delenv("VOICE_LLM_URL")
    calls, answer = [], [socket.timeout("timed out")]

    def urlopen(req, timeout=None):
        calls.append(1)
        if isinstance(answer[0], Exception):
            raise answer[0]
        return _Reply(answer[0])
    quiet.setattr(voice.urllib.request, "urlopen", urlopen)
    assert voice.interpret("find my mug") == ("pick", "mug") and len(calls) == 1       # the rule read it
    assert voice.interpret("find my phone") == ("pick", "phone") and len(calls) == 1   # not asked again at once
    assert "nemotron" in voice.status()["error"]
    quiet.setattr(voice, "_llm_again_t", 0.0)                                           # ...half a minute later
    answer[0] = json.dumps({"intent": "pick", "target": "water bottle"})
    assert voice.interpret("I am really thirsty") == ("pick", "water bottle") and len(calls) == 2


@pytest.mark.parametrize("content", ["not json at all", "[1, 2]", '{"intent": "dance", "target": "mug"}', '{"target": 7}'])
def test_a_garbled_answer_from_nemotron_costs_one_request_not_the_run(quiet, content):
    quiet.delenv("VOICE_LLM_URL")
    answers = [content, json.dumps({"intent": "none", "target": ""})]
    quiet.setattr(voice.urllib.request, "urlopen", lambda req, timeout=None: _Reply(answers.pop(0)))
    expect = ("pick", "mug") if content != '{"target": 7}' else ("pick", "7")
    assert voice.interpret("find my mug") == expect                                     # the rule, or a reading that still parsed
    assert voice.interpret("how are you today") == ("none", "")                         # Nemotron again, straight away


def test_an_answer_that_blows_up_does_not_end_the_wait(quiet, tmp_path):
    pace(quiet)
    quiet.setenv("VOICE_AUTO_LIST", "0")
    real, calls = voice.respond, []

    def respond(text):
        calls.append(text)
        if len(calls) == 1:
            raise KeyError("something new in guidance")
        return real(text)
    quiet.setattr(voice, "respond", respond)
    mic = web_mic(tmp_path, Buds(("option three",), ("the mug",)))
    try:
        assert mic.listen() == "mug" and calls == ["option three", "the mug"]
        assert "answering: KeyError" in voice.status()["error"]
    finally:
        mic.close()


def _https(url, data=None, kind=None, timeout=5):
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, data, {"Content-Type": kind} if kind else {})
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
        return r.status, r.read()


def test_a_taken_port_moves_the_page_and_no_port_at_all_leaves_the_earbuds(quiet, tmp_path):
    quiet.setenv("VOICE_AUTO_LIST", "0")
    taken = [socket.socket() for _ in range(5)]
    try:
        taken[0].bind(("0.0.0.0", 0))
        taken[0].listen(1)
        port = taken[0].getsockname()[1]
        mic = voice.WebMic(port=port, transcribe=lambda data: "", cert_dir=tmp_path / "cert", typed=False)
        try:
            assert port < mic.port <= port + 4 and _https(f"https://127.0.0.1:{mic.port}/")[0] == 200
        finally:
            mic.close()
        quiet.setattr(voice, "ThreadingHTTPServer", None)   # every port refused
        buds = Buds(("the mug",))
        mic = web_mic(tmp_path, buds)
        assert mic.url is None and mic.listen() == "mug"
        assert "no microphone page" in voice.status()["error"]
        mic.close()
    finally:
        for s in taken:
            s.close()


def test_without_a_certificate_the_page_is_served_plainly_for_typing(quiet, tmp_path):
    def no_openssl(*a, **k):
        raise FileNotFoundError("openssl")
    quiet.setattr(voice, "_tls_context", no_openssl)
    mic = web_mic(tmp_path)
    try:
        assert mic.url.startswith("http://")
        with urllib.request.urlopen(f"http://127.0.0.1:{mic.port}/", timeout=5) as r:
            assert r.status == 200 and b"isSecureContext" in r.read()       # the page says why it has no microphone
        req = urllib.request.Request(f"http://127.0.0.1:{mic.port}/text", json.dumps({"text": "the mug"}).encode())
        urllib.request.urlopen(req, timeout=5).close()
        assert mic.listen() == "mug"
    finally:
        mic.close()


def test_a_connection_that_never_says_anything_does_not_hold_up_the_page(quiet, tmp_path):
    mic = web_mic(tmp_path)
    try:
        silent = socket.create_connection(("127.0.0.1", mic.port))          # a browser's spare connection: opened, nothing sent
        try:
            t = time.monotonic()
            assert _https(f"https://127.0.0.1:{mic.port}/", timeout=4)[0] == 200
            assert time.monotonic() - t < 3
        finally:
            silent.close()
    finally:
        mic.close()


def test_typing_http_by_mistake_is_sent_to_https(quiet, tmp_path):
    mic = web_mic(tmp_path)
    try:
        with socket.create_connection(("127.0.0.1", mic.port)) as s:
            s.sendall(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n")
            reply = s.recv(500).decode()
        assert reply.startswith("HTTP/1.0 301") and f"Location: https://127.0.0.1:{mic.port}/" in reply
    finally:
        mic.close()


def test_the_certificate_names_this_machine_and_is_kept(quiet, tmp_path):
    voice._tls_context(tmp_path, "10.1.2.3")
    made = (tmp_path / "cert.pem").stat().st_mtime_ns
    text = voice.subprocess.run(["openssl", "x509", "-in", str(tmp_path / "cert.pem"), "-noout", "-text"], capture_output=True, text=True).stdout
    assert "IP Address:10.1.2.3" in text and "TLS Web Server Authentication" in text
    voice._tls_context(tmp_path, "10.1.2.3")
    assert (tmp_path / "cert.pem").stat().st_mtime_ns == made                # the phone does not have to accept a new one each run
    voice._tls_context(tmp_path, "10.9.9.9")
    assert (tmp_path / "cert.pem").stat().st_mtime_ns != made                # ...until the address changes


def test_the_page_reports_its_own_failures(quiet, tmp_path, capsys):
    page = voice.PAGE.read_text()
    for needed in ("isSecureContext", "MediaRecorder.isTypeSupported", "audio/mp4", 'fetch("log"', "NotAllowedError", "pointercancel",
                   "AbortController", 'id="log"'):
        assert needed in page, needed

    def unreadable(data):
        raise ValueError("no audio stream in 2212 bytes")
    mic = voice.WebMic(port=0, transcribe=unreadable, cert_dir=tmp_path / "cert", typed=False)
    try:
        base = f"https://127.0.0.1:{mic.port}"
        reply = json.loads(_https(base + "/say", b"x" * 2212, "audio/mp4")[1])
        assert reply["heard"] == "" and reply["bytes"] == 2212 and "no audio stream" in reply["error"]
        _https(base + "/log", b"iPhone Safari | The recorder failed (NotSupportedError).", "text/plain")
        err = capsys.readouterr().err
        assert "page: iPhone Safari | The recorder failed" in err and "2212 bytes, audio/mp4" in err
    finally:
        mic.close()


def test_listen_for_target_never_raises_and_never_spins(quiet):
    naps = []
    quiet.setattr(voice.time, "sleep", lambda s: naps.append(s))

    class Broken:
        def listen(self):
            raise RuntimeError("anything at all")
    quiet.setattr(voice, "_listener", Broken())
    quiet.setattr(voice, "_mode", "web")
    assert [voice.listen_for_target() for _ in range(4)] == [""] * 4
    assert naps == [0.5, 1.0, 1.5, 2.0] and "listening: RuntimeError: anything at all" in voice.status()["error"]


def test_no_terminal_to_type_into_is_said_once_and_waited_out(quiet):
    naps = []
    quiet.setenv("VOICE_MODE", "type")
    quiet.setenv("VOICE_AUTO_LIST", "0")
    quiet.setattr(voice.time, "sleep", lambda s: naps.append(s))

    def closed(prompt=""):
        raise EOFError
    quiet.setattr("builtins.input", closed)
    assert [voice.listen_for_target() for _ in range(3)] == [""] * 3 and naps == [1.0] * 3
    assert voice.status()["ok"] is False and "no keyboard" in voice.status()["message"]


def test_whatever_is_missing_at_the_start_something_is_left_that_works(quiet):
    quiet.setenv("VOICE_MODE", "speech")                    # asked for the wired microphone...

    def nothing(*a, **k):
        raise OSError("no such device")
    quiet.setattr(voice, "Listener", nothing)               # ...there is none...
    quiet.setattr(voice, "WebMic", nothing)                 # ...and the page cannot start either
    quiet.setattr("builtins.input", lambda prompt="": "the mug")
    quiet.setenv("VOICE_AUTO_LIST", "0")
    assert voice.listen_for_target() == "mug" and voice._mode == "type"


def test_without_whisper_typing_and_the_page_still_work(quiet, tmp_path):
    quiet.setenv("VOICE_AUTO_LIST", "0")

    def missing(model=None):
        raise RuntimeError("Whisper large-v3 is not on this machine")
    quiet.setattr(voice, "load_whisper", missing)
    quiet.setattr(voice.Headset, "find", staticmethod(lambda: Buds()))
    mic = voice.WebMic(port=0, cert_dir=tmp_path / "cert", typed=False)
    try:
        assert mic.headset is None                          # earbuds are no use without it
        assert voice.status()["ok"] is False and "Whisper could not be loaded" in voice.status()["message"]
        reply = json.loads(_https(f"https://127.0.0.1:{mic.port}/say", b"RECORDING", "audio/webm")[1])
        assert "Whisper is not loaded" in reply["error"]    # the page says so instead of "I did not catch that"
        mic.submit("the mug")
        assert mic.listen() == "mug"
    finally:
        mic.close()


# --- Whisper without a network ------------------------------------------------------------

def test_whisper_is_loaded_from_this_machine_only_and_says_what_is_missing(quiet):
    asked = []

    class WhisperModel:
        def __init__(self, name, **kw):
            asked.append((name, kw.get("local_files_only")))
            if name == "missing.en":
                raise ValueError("Cannot find an appropriate cached snapshot folder")
    quiet.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=WhisperModel))
    quiet.setenv("VOICE_STT", "cpu")
    assert voice.load_whisper("small.en").on == "cpu" and asked == [("small.en", True)]
    assert "whisper model" not in voice._trouble            # asked for the CPU: nothing to warn about
    with pytest.raises(RuntimeError, match="not on this machine.*VOICE_DOWNLOAD=1"):
        voice.load_whisper("missing.en")
    quiet.setenv("VOICE_DOWNLOAD", "1")
    voice.load_whisper("small.en")
    assert asked[-1] == ("small.en", False)


def test_falling_back_to_the_cpu_is_loud(quiet, capsys):
    tries = []

    def no_gpu(model):
        tries.append(model)
        raise RuntimeError("CUDA error: out of memory")
    quiet.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=lambda name, **kw: SimpleNamespace()))
    quiet.setattr(voice, "_gpu_whisper", no_gpu)
    quiet.setattr(voice.time, "sleep", lambda s: None)
    assert voice.load_whisper().on == "cpu" and tries == ["large-v3", "large-v3"]       # one more try before giving in
    now = voice.status()
    assert now["ok"] is False and "on the CPU" in now["message"] and now["whisper"] == "small.en on the CPU"
    assert "could not start on the GPU (RuntimeError: CUDA error: out of memory)" in capsys.readouterr().err


def test_the_checklist_names_every_part_and_never_raises(quiet):
    parts = voice.ready()
    assert [what for what, ok, detail in parts] == ["Whisper", "speech detector", "Piper voice", "Nemotron", "Cosmos", "earbuds", "openssl"]
    assert all(isinstance(ok, bool) and detail for what, ok, detail in parts)
    assert dict((what, ok) for what, ok, detail in parts)["Nemotron"] is True           # switched off in this test: said, not failed


# --- the status file ----------------------------------------------------------------------

def test_the_status_file_says_what_it_is_doing(quiet, tmp_path):
    pace(quiet)
    looking(quiet, ["mug", "phone"])
    states = []
    buds = Buds(None, ("the phone",))
    record = buds.record

    def watched(should_stop=None, abandon=None):
        states.append(voice.status())
        return record(should_stop, abandon)
    buds.record = watched
    mic = web_mic(tmp_path, buds)
    try:
        assert mic.listen() == "phone"
    finally:
        mic.close()
    assert states[0]["state"] == "listening" and states[0]["mic"] == "AirPods Pro" and states[0]["ok"] is True
    voice._write_status()
    now = json.loads((tmp_path / "status" / "voice.json").read_text())
    assert abs(now["t"] - time.time()) < 2 and now["state"] == "reaching" and now["message"] == "Guiding to phone."
    assert now["heard"] == "the phone" and now["meant"] == "reach for phone" and now["ok"] is True and now["error"] == ""
    voice._problem("earbuds", voice.MicLost("the earbuds are gone"), "The earbuds' microphone is not working.")
    now = voice.status()
    assert now["ok"] is False and now["message"] == "The earbuds' microphone is not working." and "earbuds are gone" in now["error"]
    voice._fine("earbuds")
    assert voice.status()["ok"] is True
    voice._now("listening")
    quiet.setattr(voice, "_alive_t", time.monotonic() - 120)                 # a wait that has stopped moving shows
    assert voice.status()["ok"] is False and "not moved for 120 s" in voice.status()["message"]


# --- the earbuds' two modes -------------------------------------------------------------------

class PipeWire:
    """The sound server as Headset sees it through wpctl, pw-cli, pw-dump and bluetoothctl."""

    def __init__(self, profile=5, takes_s=0.2):
        self.profile, self.takes_s, self.stuck, self.blind = profile, takes_s, False, False
        self.device, self.ready_t, self.commands, self.volumes = 61, 0.0, [], {}

    def nodes(self):
        if time.monotonic() < self.ready_t:
            return {}                                       # the old mode's nodes are gone, the new ones not yet up
        return {"Audio/Sink": 90, "Audio/Source": 91} if self.profile == 261 else {"Audio/Sink": 80}

    def run(self, cmd, **k):
        self.commands.append((time.monotonic(), cmd))
        out = ""
        if cmd[:2] == ["wpctl", "set-profile"]:
            if not (self.stuck and int(cmd[3]) == 5):       # stuck: "trying to set invalid profile", as on 10 October
                self.profile, self.ready_t = int(cmd[3]), time.monotonic() + self.takes_s
        elif cmd[:2] == ["pw-cli", "enum-params"] and cmd[3] == "Profile":
            out = f"  Object: size 328\n    Prop: key Spa:Pod:Object:Param:Profile:index (1), flags 00000000\n      Int {self.profile}\n"
        elif cmd[0] == "pw-dump":
            if self.blind:
                raise FileNotFoundError("pw-dump")
            out = json.dumps([{"id": i, "type": "PipeWire:Interface:Node", "info": {"props": {"device.id": self.device, "media.class": c}}}
                              for c, i in self.nodes().items()] + [{"id": 3, "type": "PipeWire:Interface:Core"}])
        elif cmd[:2] == ["wpctl", "get-volume"]:
            out = "Volume: 0.53\n"
        elif cmd[:2] == ["wpctl", "set-volume"]:
            self.volumes[int(cmd[2])] = float(cmd[3])
        elif cmd[:2] == ["pw-cli", "info"]:
            out = '\t\tapi.bluez5.address = "18:3F:70:53:BD:95"\n\t\tdevice.name = "bluez_card.18_3F_70_53_BD_95"\n'
        elif cmd[:2] == ["bluetoothctl", "connect"]:
            self.stuck, self.profile, self.device, self.ready_t = False, 5, 62, 0.0     # back in stereo, under a new number
        return SimpleNamespace(stdout=out, returncode=0)

    def switches(self):
        return [(t, int(cmd[3])) for t, cmd in self.commands if cmd[:2] == ["wpctl", "set-profile"]]


@pytest.fixture
def pipewire(quiet):
    pw = PipeWire()
    quiet.setattr(voice.subprocess, "run", pw.run)
    quiet.setattr(voice, "SWITCH_GAP_S", 0.4)
    quiet.setattr(voice, "SWITCH_WAIT_S", 0.6)
    return pw


def test_a_switch_waits_for_the_mode_to_be_really_up_and_matches_the_volume(pipewire):
    buds = voice.Headset(61, 261, 5, "AirPods Pro")
    t = time.monotonic()
    buds.open()
    assert 0.2 <= time.monotonic() - t < 1.2 and pipewire.nodes() == {"Audio/Sink": 90, "Audio/Source": 91}
    assert pipewire.volumes == {90: 0.53}                   # the question is as loud as the guidance was in stereo
    buds.open()
    assert len(pipewire.switches()) == 1                    # already there: no second switch to go wrong


def test_two_switches_are_never_close_together(pipewire):
    buds = voice.Headset(61, 261, 5, "AirPods Pro")
    pipewire.takes_s = 0.0
    buds.open()
    buds.close()                                            # a request that was already waiting: back at once
    (t1, to1), (t2, to2) = pipewire.switches()
    assert (to1, to2) == (261, 5) and t2 - t1 >= 0.4


def test_stereo_that_does_not_come_back_is_repaired_by_reconnecting(pipewire, quiet):
    found = voice.Headset(62, 261, 5, "AirPods Pro")
    pipewire.takes_s = 0.0
    buds = voice.Headset(61, 261, 5, "AirPods Pro")
    buds.open()
    pipewire.stuck = True
    quiet.setattr(voice.Headset, "find", staticmethod(lambda: found))
    buds.close()
    blue = [cmd for t, cmd in pipewire.commands if cmd[0] == "bluetoothctl"]
    assert blue == [["bluetoothctl", "disconnect", "18:3F:70:53:BD:95"], ["bluetoothctl", "connect", "18:3F:70:53:BD:95"]]
    assert buds.device == 62 and pipewire.profile == 5 and voice.status()["ok"] is True
    pipewire.device = 62
    buds.open()
    pipewire.stuck = True
    pipewire.run = lambda cmd, **k: PipeWire.run(pipewire, cmd, **k) if cmd[:2] != ["bluetoothctl", "connect"] else SimpleNamespace(stdout="")
    buds.close()                                            # a minute has not passed: not reconnected again, but said
    assert len([cmd for t, cmd in pipewire.commands if cmd[0] == "bluetoothctl"]) == 2
    assert voice.status()["ok"] is False and "stuck in headset mode" in voice.status()["message"]


def test_a_microphone_that_does_not_come_up_is_reported_not_recorded_from(pipewire, quiet, tmp_path):
    pipewire.takes_s = 5.0                                  # the headset mode never comes up in time
    buds = voice.Headset(61, 261, 5, "AirPods Pro")
    with pytest.raises(voice.MicLost, match="did not go into headset mode"):
        buds.open()
    mic = web_mic(tmp_path, buds, hear=lambda audio: "")
    try:
        pipewire.ready_t = time.monotonic() + 5
        pipewire.profile = 5
        assert mic._open(buds) is False and mic.headset is None and voice.status()["ok"] is False
    finally:
        mic.close()


def test_where_the_mode_cannot_be_read_it_waits_the_fixed_moment_as_before(pipewire, quiet):
    pipewire.blind = True
    naps = []
    real = time.sleep
    quiet.setattr(voice.time, "sleep", lambda s: naps.append(s) or real(0))
    buds = voice.Headset(61, 261, 5, "AirPods Pro")
    buds.open()
    buds.close()
    assert 0.5 in naps and 1.0 in naps and not [cmd for t, cmd in pipewire.commands if cmd[0] == "bluetoothctl"]


def test_staying_in_headset_mode_never_switches_back(quiet, tmp_path):
    quiet.setenv("VOICE_HEADSET_MODE", "stay")
    quiet.setenv("VOICE_AUTO_LIST", "0")
    registered = []
    quiet.setattr(voice.atexit, "register", lambda f: registered.append(f))
    buds = Buds(("the mug",), ("the phone",))
    mic = web_mic(tmp_path, buds)
    try:
        assert mic.listen() == "mug" and mic.listen() == "phone"
        assert "stereo" not in buds.modes and registered == [buds.close]    # stereo again only when the run ends
    finally:
        mic.close()


# --- the earbuds' microphone itself -------------------------------------------------------

class FakeSound:
    """sounddevice with a microphone that delivers the given blocks as fast as they are taken."""

    def __init__(self, blocks):
        self.blocks = blocks

    def InputStream(self, callback=None, **k):
        sound = self

        class Stream:
            def __enter__(self):
                self.on = True
                self.thread = threading.Thread(target=self.feed, daemon=True)
                self.thread.start()

            def feed(self):
                for block in sound.blocks:
                    if not self.on:
                        return
                    callback(np.asarray(block, np.float32).reshape(-1, 1), len(block), None, None)
                    time.sleep(0.0002)

            def __exit__(self, *a):
                self.on = False
                self.thread.join(1)
        return Stream()


def test_a_microphone_that_gives_no_sound_or_only_zeros_counts_as_lost(quiet):
    buds = voice.Headset(61, 261, 5)
    quiet.setitem(sys.modules, "sounddevice", FakeSound([]))
    with pytest.raises(voice.MicLost, match="no sound is arriving"):
        buds.record()
    zeros = [np.zeros(480, np.float32)] * 250
    quiet.setitem(sys.modules, "sounddevice", FakeSound(zeros))
    assert buds.record() is None and buds.record() is None                  # a quiet moment proves nothing...
    with pytest.raises(voice.MicLost, match="only silence"):
        buds.record()                                                       # ...fifteen seconds of exact zeros does
    quiet.setitem(sys.modules, "sounddevice", FakeSound(_noise((7.0, 0.0003))))
    assert buds.record() is None and buds._zeros_s == 0.0                   # a real, quiet room is fine


def test_the_tail_of_talk_that_ran_on_is_not_taken_for_a_short_sentence(quiet):
    buds = voice.Headset(61, 261, 5)
    monologue = _noise((0.5, 0.0005), (8.5, 0.03))
    quiet.setitem(sys.modules, "sounddevice", FakeSound(monologue))
    first = buds.record()
    assert first is not None and first.size / SR > voice.REQUEST_MAX_S and buds._ran_on     # too long: dropped by the caller
    quiet.setitem(sys.modules, "sounddevice", FakeSound(_noise((3.0, 0.07), (2.0, 0.0005), seed=4)))
    assert buds.record() is None and not buds._ran_on                       # three more seconds of the same talk: not a request
    quiet.setitem(sys.modules, "sounddevice", FakeSound(_noise((1.0, 0.0005), (1.0, 0.07), (2.0, 0.0005), seed=5)))
    assert buds.record() is not None                                        # after a pause, a voice is a voice again
