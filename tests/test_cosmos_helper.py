"""Cosmos as a slow helper: listing what is on the table when asked, and finding the object
when the detector cannot. The Cosmos server, Nemotron and the camera are all faked."""
import io
import json
import time

import numpy as np
import pytest

import audio_out
import cosmos
import guidance
import perception
import stream
import voice
from audio_out import Mixer, PrintSink, ToneConfig
from fakes import make_K, object_box
from interfaces import Frame
from perception import CosmosHelper, Detection, Perception, PerceptionConfig

W, H = 480, 640
K = make_K(W, H)
OBJ = np.array([0.05, 0.12, 0.60])


def frame(t=0.0, obj=OBJ, pose=None):
    depth = np.full((H, W), 0.90, np.float32)
    if obj is not None:
        x1, y1, x2, y2 = object_box(K, obj)
        depth[y1:y2, x1:x2] = obj[2]
    return Frame(np.full((H, W, 3), 90, np.uint8), depth, K, pose, t, t_recv=t)


class _Reply:
    def __init__(self, content):
        self.body = json.dumps({"choices": [{"message": {"content": json.dumps(content)}}],
                                "usage": {"completion_tokens": 20}}).encode()

    def __enter__(self):
        return io.BytesIO(self.body)

    def __exit__(self, *a):
        return False


def _server(monkeypatch, content, seen=None):
    def urlopen(req, timeout=None):
        if seen is not None:
            seen.append(json.loads(req.data))
        return _Reply(content)
    monkeypatch.setattr(cosmos.urllib.request, "urlopen", urlopen)


# --- cosmos.py -----------------------------------------------------------------

def test_find_scales_the_box_to_the_picture(monkeypatch):
    sent = []
    _server(monkeypatch, {"visible": True, "box": [250, 500, 750, 1000], "grasp": "the handle"}, sent)
    out = cosmos.find(np.zeros((640, 480, 3), np.uint8), "mug")
    assert out["box"] == (120.0, 320.0, 360.0, 640.0) and out["grasp"] == "the handle"
    said = sent[0]["messages"][0]["content"]
    assert "mug" in said[1]["text"] and said[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_find_says_none_when_cosmos_does_not_see_it(monkeypatch):
    _server(monkeypatch, {"visible": False, "box": [10, 10, 500, 500], "grasp": ""})    # a box it should not have drawn
    assert cosmos.find(np.zeros((64, 64, 3), np.uint8), "banana")["box"] is None
    _server(monkeypatch, {"visible": True, "box": [500, 500, 500, 500], "grasp": ""})   # an empty box
    assert cosmos.find(np.zeros((64, 64, 3), np.uint8), "banana")["box"] is None


def test_list_objects_gives_short_names_without_repeats(monkeypatch):
    _server(monkeypatch, {"objects": ["Water Bottle.", "laptop  computer", "water bottle", "", "mug", "pen", "book", "cap"]})
    assert cosmos.list_objects(np.zeros((64, 64, 3), np.uint8), n=4) == ["water bottle", "laptop computer", "mug", "pen"]


def test_available_is_false_when_nothing_answers(monkeypatch):
    def refuse(req, timeout=None):
        raise OSError("connection refused")
    monkeypatch.setattr(cosmos.urllib.request, "urlopen", refuse)
    assert cosmos.available("http://127.0.0.1:9/v1") is False and cosmos.available("off") is False


# --- the detector cannot find it: ask Cosmos ------------------------------------

class Blind:
    """A detector that never finds anything."""
    def __init__(self):
        self.obj = None

    def set_target(self, target):
        pass

    def detect(self, rgb):
        return [] if self.obj is None else [Detection(object_box(K, self.obj), 0.9, "x")]


class Helper:
    """Stands in for CosmosHelper: answers on the next poll after being asked."""
    def __init__(self, box="object"):
        self.box, self.asked, self.pending = box, [], None

    def idle(self):
        return self.pending is None

    def request(self, frame, target):
        self.asked.append((frame.t, target))
        box = object_box(K, OBJ) if self.box == "object" else self.box
        self.pending = (frame, target, box)
        return True

    def poll(self):
        out, self.pending = self.pending, None
        return out


def make(helper, **cfg):
    cfg.setdefault("smooth", 1)
    return Perception(Blind(), None, PerceptionConfig(**cfg), helper)


def test_cosmos_is_asked_only_after_the_detector_has_failed_for_a_while():
    h = Helper()
    p = make(h)
    assert p.locate(frame(t=0.0), "can").object_xyz is None and h.asked == []
    p.locate(frame(t=0.5), "can")
    assert h.asked == []                                    # under a second: the detector may still find it
    p.locate(frame(t=1.1), "can")
    assert h.asked == [(1.1, "can")]


def test_cosmos_box_gives_the_object_a_position_from_the_frame_it_saw():
    h = Helper()
    p = make(h)
    for t in (0.0, 1.1):
        p.locate(frame(t=t), "can")
    s = p.locate(frame(t=1.2, obj=None), "can")             # the answer arrives; this frame's own depth is useless
    assert np.allclose(s.object_xyz, OBJ, atol=0.004)
    assert s.object_box == object_box(K, OBJ)
    assert not s.object_visible                             # the detector did not see it in this frame


def test_cosmos_position_is_kept_for_a_few_seconds_then_dropped():
    h = Helper()
    p = make(h)
    for t in (0.0, 1.1, 1.2):
        p.locate(frame(t=t), "can")
    h.box = None                                            # from now on Cosmos does not see it either
    assert p.locate(frame(t=5.0), "can").object_xyz is not None     # 3.8 s old: past hold_s, within helper_hold_s
    assert p.locate(frame(t=12.5), "can").object_xyz is None


def test_cosmos_is_asked_less_often_after_it_says_not_there():
    h = Helper(box=None)
    p = make(h)
    t = 0.0
    while t < 20.0:
        p.locate(frame(t=t, obj=None), "unicorn")
        t += 0.1
    gaps = np.diff([a[0] for a in h.asked])
    assert 2 <= len(h.asked) <= 4 and gaps[0] >= 6.0 and gaps[-1] >= gaps[0]    # backs off instead of hammering the GPU


def test_an_answer_about_the_previous_target_is_ignored():
    h = Helper()
    p = make(h)
    for t in (0.0, 1.1):
        p.locate(frame(t=t), "can")
    s = p.locate(frame(t=1.2), "phone")                     # target changed while Cosmos was thinking
    assert s.object_xyz is None


def test_detector_takes_over_again_when_it_finds_the_object():
    h = Helper()
    p = make(h)
    for t in (0.0, 1.1, 1.2):
        p.locate(frame(t=t), "can")
    moved = OBJ + np.array([0.10, 0.0, 0.0])
    p.detector.obj = moved
    asked = len(h.asked)
    s = p.locate(frame(t=1.3, obj=moved), "can")
    assert s.object_visible and np.allclose(s.object_xyz, moved, atol=0.004)
    for t in (2.0, 3.0, 4.0):
        p.locate(frame(t=t, obj=moved), "can")
    assert len(h.asked) == asked                            # no more questions while the detector sees it


def test_with_poses_the_cosmos_answer_is_carried_to_where_the_wearer_is_now():
    def pose(tx):
        T = np.eye(4)
        T[0, 3] = tx
        return T
    h = Helper()
    p = make(h, use_pose=True)
    for t in (0.0, 1.1):
        p.locate(frame(t=t, pose=pose(0.0)), "can")
    s = p.locate(frame(t=1.2, obj=None, pose=pose(0.10)), "can")    # stepped 10 cm right while Cosmos was thinking
    assert np.allclose(s.object_xyz, OBJ - np.array([0.10, 0, 0]), atol=0.004)


def test_helper_thread_delivers_one_answer_and_survives_errors():
    calls = []

    def find(rgb, target):
        calls.append(target)
        time.sleep(0.05)
        if target == "boom":
            raise OSError("server went away")
        return {"box": (1.0, 2.0, 30.0, 40.0)}

    h = CosmosHelper(find)
    f = frame()
    assert h.idle() and h.request(f, "can") and not h.request(f, "can")     # one question at a time
    deadline = time.time() + 2
    while not h.idle() and time.time() < deadline:
        time.sleep(0.01)
    assert h.poll() == (f, "can", (1.0, 2.0, 30.0, 40.0)) and h.poll() is None
    h.request(f, "boom")
    while not h.idle() and time.time() < deadline:
        time.sleep(0.01)
    assert h.poll() == (f, "boom", None) and calls == ["can", "boom"]


# --- "what's on the table?" ----------------------------------------------------

@pytest.fixture
def quiet_voice(monkeypatch):
    monkeypatch.setenv("VOICE_LLM_URL", "off")
    monkeypatch.setenv("VOICE_AUTO_LIST", "0")
    monkeypatch.setenv("GUIDANCE_AUDIO", "off")
    monkeypatch.delenv("GUIDANCE_MODE", raising=False)
    monkeypatch.setattr(voice, "_ask_next", True)
    monkeypatch.setattr(guidance, "_default", None)
    monkeypatch.setattr(voice, "last_objects", [])
    monkeypatch.setattr(voice, "_llm_ok", None)
    monkeypatch.setattr(voice, "_stopped", False)
    monkeypatch.setattr(voice, "_told_no_cosmos", False)
    return monkeypatch


@pytest.mark.parametrize("said", ["What's on the table?", "what is in front of me", "What can I pick up?",
                                  "what do you see", "describe what is here"])
def test_asking_what_is_there_is_a_list_request(quiet_voice, said):
    assert voice.interpret(said) == ("list", "")


@pytest.mark.parametrize("said, target", [("find my mug", "mug"), ("where's my phone", "phone"), ("can", "can")])
def test_naming_an_object_is_still_a_pick(quiet_voice, said, target):
    assert voice.interpret(said) == ("pick", target)


class _Camera:
    def get_frame(self, wait=0.0):
        now = time.monotonic()
        return Frame(np.zeros((64, 48, 3), np.uint8), None, None, None, now, t_recv=now)


def test_listing_speaks_what_cosmos_sees_then_the_wearer_picks(quiet_voice):
    quiet_voice.setattr(stream, "_default", _Camera())
    quiet_voice.setattr(cosmos, "available", lambda *a, **k: True)
    quiet_voice.setattr(cosmos, "list_objects", lambda rgb, **k: ["water bottle", "laptop computer", "mug"])
    assert voice.respond("what's on the table?") == ""              # nothing to look for yet: main.py asks again
    assert guidance.pop_phrases()[-1] == "I can see water bottle, laptop computer and mug. Which one?"
    assert voice.respond("the mug") == "mug"
    assert voice.respond("the second one") == "laptop computer" and voice.respond("number one") == "water bottle"
    assert voice.respond("the last one") == "mug"


def test_a_single_object_can_be_accepted_with_yes(quiet_voice):
    quiet_voice.setattr(stream, "_default", _Camera())
    quiet_voice.setattr(cosmos, "available", lambda *a, **k: True)
    quiet_voice.setattr(cosmos, "list_objects", lambda rgb, **k: ["coffee cup"])
    voice.respond("what can I pick up")
    assert "one thing: coffee cup" in guidance.pop_phrases()[-1]
    assert voice.respond("yes") == "coffee cup"


def test_listing_explains_itself_when_it_cannot_look(quiet_voice):
    def no_phone(*a, **k):
        raise stream.SourceError("No phone app is streaming.")
    quiet_voice.setattr(stream, "_default", None)
    quiet_voice.setattr(stream, "start", no_phone)                  # the camera cannot be opened
    voice.respond("what's on the table?")
    assert guidance.pop_phrases()[-1] == "I have no picture from the camera yet."
    quiet_voice.setattr(stream, "_default", _Camera())
    quiet_voice.setattr(cosmos, "available", lambda *a, **k: False)
    voice.respond("what's on the table?")
    assert guidance.pop_phrases()[-1] == "I can't look around right now."
    quiet_voice.setattr(cosmos, "available", lambda *a, **k: True)
    quiet_voice.setattr(cosmos, "list_objects", lambda rgb, **k: [])
    voice.respond("what's on the table?")
    assert guidance.pop_phrases()[-1] == "I can't see anything to pick up. Turn towards the table, or step back a little."


def test_nemotron_can_decide_it_is_a_list_request(quiet_voice):
    quiet_voice.delenv("VOICE_LLM_URL")

    def urlopen(req, timeout=None):
        said = json.loads(req.data)["messages"][-1]["content"]
        answer = {"anything I could grab from here?": {"intent": "list", "target": ""},
                  "I am really thirsty": {"intent": "pick", "target": "water bottle"},
                  "how are you today": {"intent": "none", "target": ""}}[said]
        return _Reply(answer)
    quiet_voice.setattr(voice.urllib.request, "urlopen", urlopen)
    assert voice.interpret("anything I could grab from here?") == ("list", "")
    assert voice.interpret("I am really thirsty") == ("pick", "water bottle")
    assert voice.interpret("how are you today") == ("none", "") and voice.understand("how are you today") == ""


# --- saying any sentence -------------------------------------------------------

def test_speak_plays_synthesised_speech_and_logs_the_words(monkeypatch):
    monkeypatch.setattr(audio_out, "synthesize", lambda text, sr: np.full(2400, 0.5, np.float32))
    m = Mixer(ToneConfig(), clips={"grab": np.full(1200, 0.5, np.float32)})
    m.speak("I can see a mug.")
    assert m.saying() and np.abs(m.render(256)).max() > 0.3
    m.say("grab")                                           # waits its turn
    m.wait_quiet(0.01)                                      # returns even though it is still speaking
    while m.saying():
        m.render(256)
    assert m.spoken == ["I can see a mug.", "grab"] and m.log == ["I can see a mug.", "Grab"]


def test_speak_without_any_voice_still_reaches_the_log(monkeypatch):
    monkeypatch.setattr(audio_out, "synthesize", lambda text, sr: None)
    m = Mixer(ToneConfig(), clips={})
    m.speak("I can see a mug.")
    m.render(256)
    assert not m.saying() and m.log == ["I can see a mug."]


def test_guidance_reports_what_was_said_once(monkeypatch):
    monkeypatch.setenv("GUIDANCE_AUDIO", "off")
    monkeypatch.setattr(guidance, "_default", None)
    guidance.reset("mug")
    guidance.say_text("I can see a mug.")
    assert guidance.pop_phrases() == ["Looking", "I can see a mug."] and guidance.pop_phrases() == []
    assert isinstance(guidance._get().sink, PrintSink)


def test_the_first_question_opens_the_camera_itself(quiet_voice):
    # main.py asks for the first target before it has read a single frame
    opened = []

    def start(*a, **k):
        opened.append(1)
        stream._default = _Camera()
    quiet_voice.setattr(stream, "_default", None)
    quiet_voice.setattr(stream, "start", start)
    quiet_voice.setattr(cosmos, "available", lambda *a, **k: True)
    quiet_voice.setattr(cosmos, "list_objects", lambda rgb, **k: ["mug", "phone"])
    voice.respond("what's on the table?")
    assert opened == [1] and guidance.pop_phrases()[-1] == "I can see mug and phone. Which one?"
    quiet_voice.setattr(stream, "_default", None)


# --- a phone or laptop as the microphone ----------------------------------------

import ssl
import urllib.request


def _https(url, data=None, kind=None):
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE                              # the page's certificate is made on the spot
    req = urllib.request.Request(url, data, {"Content-Type": kind} if kind else {})
    with urllib.request.urlopen(req, timeout=5, context=ctx) as r:
        return r.status, r.read()


@pytest.fixture
def mic(quiet_voice, tmp_path):
    poked = []
    quiet_voice.setattr(voice, "_poke_main", lambda: poked.append(1))
    heard = {b"RECORDING-1": "find my mug", b"RECORDING-2": "", b"BROKEN": None}

    def transcribe(data):
        if heard[data] is None:
            raise ValueError("not audio")
        return heard[data]
    m = voice.WebMic(port=0, transcribe=transcribe, cert_dir=tmp_path / "cert", typed=False)
    m.base = f"https://127.0.0.1:{m.port}"
    m.poked = poked
    yield m
    m.close()


def test_microphone_page_is_served_over_https(mic):
    status, body = _https(mic.base + "/")
    assert status == 200 and b"Hold to talk" in body and b"getUserMedia" in body
    assert json.loads(_https(mic.base + "/status")[1]) == {"heard": "", "target": ""}


def test_a_recording_from_the_page_becomes_the_target(mic):
    out = {}
    t = __import__("threading").Thread(target=lambda: out.setdefault("target", mic.listen()))
    t.start()
    time.sleep(0.1)                                              # main.py is waiting for a request
    status, body = _https(mic.base + "/say", b"RECORDING-1", "audio/webm")
    t.join(3)
    assert status == 200 and json.loads(body)["heard"] == "find my mug"
    assert out["target"] == "mug" and mic.poked == []            # it was waiting: nothing to interrupt
    status = json.loads(_https(mic.base + "/status")[1])
    assert status["heard"] == "find my mug" and status["target"] == "mug" and status["option"] == "1: tones"


def test_a_request_during_a_reach_interrupts_it(mic):
    _https(mic.base + "/text", json.dumps({"text": "the phone"}).encode(), "application/json")
    assert mic.poked == [1]                                      # main.py is told to ask again...
    assert mic.listen() == "phone"                               # ...and gets the request that was waiting


def test_silence_or_a_bad_recording_asks_for_nothing(mic):
    assert json.loads(_https(mic.base + "/say", b"RECORDING-2", "audio/webm")[1]) == {"heard": "", "bytes": 11}
    broken = json.loads(_https(mic.base + "/say", b"BROKEN", "audio/webm")[1])
    assert broken["heard"] == "" and "could not read that recording (ValueError: not audio)" in broken["error"]   # the page shows why
    assert mic._requests.empty() and mic.poked == []


def test_a_sound_server_with_nothing_to_record_from_is_not_a_microphone(quiet_voice):
    import shutil
    from types import SimpleNamespace
    quiet_voice.setattr(shutil, "which", lambda name: "/usr/bin/pactl")
    monitors = "55\talsa_output.hdmi.monitor\tPipeWire\ts32le 2ch 48000Hz\tSUSPENDED\n"
    quiet_voice.setattr(voice.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=monitors))
    assert voice.has_microphone() is False                     # only the speakers' loopback
    real = monitors + "60\tbluez_input.AA_BB.0\tPipeWire\ts16le 1ch 16000Hz\tSUSPENDED\n"
    quiet_voice.setattr(voice.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=real))
    assert voice.has_microphone() is True                      # earbuds with a microphone


def test_pipewire_source_list_decides_when_there_is_no_pactl(quiet_voice):
    import shutil
    from types import SimpleNamespace
    quiet_voice.setattr(shutil, "which", lambda name: "/usr/bin/wpctl" if name == "wpctl" else None)
    empty = "Audio\n \u251c\u2500 Sinks:\n \u2502  *   55. Built-in Audio Digital Stereo (HDMI) [vol: 0.40]\n \u251c\u2500 Sources:\n \u2502  \n \u251c\u2500 Source endpoints:\n \u2514\u2500 Streams:\nVideo\n"
    quiet_voice.setattr(voice.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=empty))
    assert voice.has_microphone() is False
    paired = empty.replace("Sources:\n \u2502  \n", "Sources:\n \u2502  *   71. AirPods Pro [vol: 1.00]\n")
    quiet_voice.setattr(voice.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=paired))
    assert voice.has_microphone() is True


# --- speaking up after a grab, and switching the way of guiding ---------------------

def _looking(quiet_voice, objects):
    quiet_voice.setenv("VOICE_AUTO_LIST", "1")
    quiet_voice.setattr(stream, "_default", _Camera())
    quiet_voice.setattr(cosmos, "available", lambda *a, **k: True)
    quiet_voice.setattr(cosmos, "list_objects", lambda rgb, **k: list(objects))


def test_it_looks_and_asks_by_itself_at_the_start_and_after_each_grab(quiet_voice):
    _looking(quiet_voice, ["mug", "phone"])
    answers = iter(["the mug", "um", "the second one"])
    quiet_voice.setattr("builtins.input", lambda prompt="": next(answers))
    assert voice.typed() == "mug"                                   # start: it speaks first
    assert guidance.pop_phrases() == ["I can see mug and phone. What do you want to grab?"]
    # ...the reach happens, grab, main.py asks again...
    assert voice.typed() == ""                                      # an answer it could not use
    assert guidance.pop_phrases() == ["I can see mug and phone. What do you want to grab?"]
    assert voice.typed() == "phone"                                 # asked again: it does not repeat the list
    assert guidance.pop_phrases() == []


def test_a_request_already_waiting_is_not_talked_over(quiet_voice, mic):
    _looking(quiet_voice, ["mug", "phone"])
    mic.submit("find the phone")                                    # said during the reach
    assert mic.listen() == "phone" and guidance.pop_phrases() == []


def test_without_cosmos_it_still_asks(quiet_voice):
    quiet_voice.setenv("VOICE_AUTO_LIST", "1")
    quiet_voice.setattr(stream, "_default", _Camera())
    quiet_voice.setattr(cosmos, "available", lambda *a, **k: False)
    quiet_voice.setattr("builtins.input", lambda prompt="": "mug")
    assert voice.typed() == "mug" and guidance.pop_phrases() == ["I can't look around right now. Tell me what you want to grab."]
    assert voice.respond("") == "" and voice.typed() == "mug"       # said once: after the next grab it simply asks
    assert guidance.pop_phrases() == ["What do you want to grab?"]


@pytest.mark.parametrize("said, number, name", [("option three", 3, "combined words"), ("Option 5.", 5, "clock face"),
                                               ("switch to mode two please", 2, "one step at a time")])
def test_saying_an_option_switches_the_way_of_guiding(quiet_voice, said, number, name):
    assert voice.respond(said) == ""                                # nothing being looked for: main.py asks again
    assert guidance.option()[0] == number and guidance.pop_phrases() == [f"Option {number}: {name}."]


def test_switching_option_mid_reach_keeps_the_target(quiet_voice):
    guidance.reset("mug")
    guidance.pop_phrases()
    assert voice.respond("option 4") == "mug"                       # main.py restarts guidance on the same object
    assert guidance.option()[1] == "combined_beeps"
    assert voice.respond("next option") == "mug" and guidance.option()[1] == "clock"
    assert voice.respond("next option") == "mug" and guidance.option()[1] == "tones"      # wraps round


# --- earbuds as the microphone ------------------------------------------------

def _blocks(*parts, sr=16000, block_s=0.03):
    """parts: (seconds, level). Returns a generator of 30 ms blocks of noise at each level."""
    rng = np.random.default_rng(0)
    n = int(sr * block_s)
    for seconds, level in parts:
        for _ in range(int(round(seconds / block_s))):
            yield (rng.standard_normal(n) * level).astype(np.float32)


def test_the_spoken_part_is_cut_out_of_the_recording():
    audio = voice.cut_speech(_blocks((1.0, 0.004), (1.5, 0.08), (2.0, 0.004)))      # quiet, voice, quiet
    assert audio is not None and 1.6 < audio.size / 16000 < 2.9                       # the voice plus the pause that ended it
    assert np.sqrt((audio[4800:24000] ** 2).mean()) > 0.05


def test_nobody_speaking_gives_nothing():
    assert voice.cut_speech(_blocks((7.0, 0.004))) is None
    assert voice.cut_speech(_blocks((7.0, 0.03))) is None             # a steadily noisy hall is not a voice
    assert voice.cut_speech(_blocks((1.0, 0.004), (0.06, 0.2), (7.0, 0.004))) is None     # a click


def test_a_soft_voice_through_earbuds_is_still_caught():
    # Measured through AirPods in the hall: room 0.0003, a soft "blue cup" 0.006 to 0.010, people nearby 0.003.
    assert voice.cut_speech(_blocks((1.0, 0.0003), (0.6, 0.008), (2.0, 0.0003))) is not None
    assert voice.cut_speech(_blocks((1.0, 0.0003), (0.6, 0.003), (6.0, 0.0003))) is None


def test_whisper_is_told_the_names_that_were_just_listed(monkeypatch):
    monkeypatch.setattr(voice, "last_objects", ["water bottle", "blue cup"])
    assert voice.expected_words() == "water bottle, blue cup"
    monkeypatch.setattr(voice, "last_objects", [])
    assert voice.expected_words() is None


def test_a_listed_name_is_taken_as_said_with_its_colour(monkeypatch):
    monkeypatch.setenv("VOICE_LLM_URL", "off")
    monkeypatch.setattr(voice, "last_objects", ["water bottle", "blue cup", "orange", "cup"])
    assert voice.interpret("Blue cup.") == ("pick", "blue cup")              # not the plain cup beside it
    assert voice.interpret("I want the cup") == ("pick", "cup")
    assert voice.interpret("blue cup, blue cup, what is on the table?") == ("list", "")


def test_nonsense_from_whisper_is_not_a_request():
    assert not voice.plausible("F" * 100) and not voice.worth_hearing("F" * 100)          # seen live on the GX10
    assert not voice.plausible("the the the the the the the the the the the the the the the the the the the the")
    assert voice.plausible("blue cup, blue cup, what is on the table?") and voice.plausible("blue cup")
    assert voice.plausible("so the phone streams color and depth to the box, the detector finds the object")


def test_the_listening_beep_is_short_and_not_the_chime():
    m = Mixer(ToneConfig(), clips={})
    assert 0.1 < len(m.clips["listen"]) / m.cfg.sr < 0.4 and len(m.clips["listen"]) < len(m.clips["aligned"])
    m.say("listen")
    assert m.log == ["(beep: listening)"]


def test_listening_stops_when_a_typed_request_arrives():
    seen = []
    out = voice.cut_speech(_blocks((5.0, 0.004)), should_stop=lambda: seen.append(1) or len(seen) > 20)
    assert out is None and len(seen) < 40


def test_room_noise_transcripts_are_ignored():
    assert not voice.worth_hearing("Thank you.") and not voice.worth_hearing(" you ") and not voice.worth_hearing("")
    assert voice.worth_hearing("the green can") and voice.worth_hearing("option three")


def test_earbuds_are_found_with_their_two_modes(quiet_voice):
    import shutil
    from types import SimpleNamespace
    status = ("Audio\n ├─ Devices:\n │      42. Built-in Audio                      [alsa]\n"
              " │      61. AirPods Pro                         [bluez5]\n ├─ Sinks:\n │  *   59. AirPods Pro [vol: 0.5]\nVideo\n")
    profiles = "".join(f'  Object: size 1\n    Prop: key Spa:Pod:Object:Param:Profile:index (1), flags 0\n      Int {i}\n'
                       f'    Prop: key Spa:Pod:Object:Param:Profile:name (2), flags 0\n      String "{n}"\n'
                       for i, n in ((0, "off"), (3, "headset-head-unit"), (5, "a2dp-sink"), (261, "headset-head-unit-msbc")))

    def run(cmd, **k):
        return SimpleNamespace(stdout=status if cmd[0] == "wpctl" else profiles)
    quiet_voice.setattr(shutil, "which", lambda name: "/usr/bin/" + name)
    quiet_voice.setattr(voice.subprocess, "run", run)
    h = voice.Headset.find()
    assert (h.device, h.headset, h.stereo, h.name) == (61, 261, 5, "AirPods Pro")     # wideband mode preferred
    quiet_voice.setenv("VOICE_HEADSET", "off")
    assert voice.Headset.find() is None


class _Earbuds:
    """Hears nothing twice, then a sentence."""
    name = "AirPods"

    def __init__(self):
        self.calls = 0
        self.modes = []

    def open(self):
        self.modes.append("headset")

    def close(self):
        self.modes.append("stereo")

    def record(self, should_stop=None):
        self.calls += 1
        return None if self.calls < 3 else np.full(16000, 0.1, np.float32)


def test_a_spoken_answer_through_the_earbuds_becomes_the_target(quiet_voice, tmp_path):
    quiet_voice.setattr(voice, "_poke_main", lambda: None)
    buds = _Earbuds()
    mic = voice.WebMic(port=0, transcribe=lambda data: "", cert_dir=tmp_path / "c", typed=False,
                       headset=buds, hear=lambda audio: "the green can please")
    try:
        assert mic.listen() == "green can" and buds.calls == 3 and mic.status["heard"] == "the green can please"
        assert buds.modes == ["headset", "stereo"]            # one switch each way, however long the wait
    finally:
        mic.close()


def test_stray_talk_does_not_beep_again_or_switch_the_earbuds(quiet_voice, tmp_path):
    quiet_voice.setattr(voice, "_poke_main", lambda: None)
    beeps = []
    quiet_voice.setattr(voice, "_beep", lambda: beeps.append(1))
    heard = iter(["okay guys i am going", "each processor has its own memory", "the green can please"])
    quiet_voice.setattr(voice, "interpret", lambda text: ("pick", "green can") if "green" in text else ("none", ""))
    buds = _Earbuds()
    buds.record = lambda should_stop=None: np.full(16000, 0.1, np.float32)
    mic = voice.WebMic(port=0, transcribe=lambda data: "", cert_dir=tmp_path / "c", typed=False,
                       headset=buds, hear=lambda audio: next(heard))
    try:
        assert mic.listen() == "green can"
        assert beeps == [1] and buds.modes == ["headset", "stereo"]      # two sentences from the room changed nothing
    finally:
        mic.close()


def test_typing_still_wins_while_the_earbuds_listen(quiet_voice, tmp_path):
    quiet_voice.setattr(voice, "_poke_main", lambda: None)
    mic = voice.WebMic(port=0, transcribe=lambda data: "", cert_dir=tmp_path / "c", typed=False,
                       headset=_Earbuds(), hear=lambda audio: "thank you")      # only room noise is heard
    try:
        mic.submit("find the phone")
        assert mic.listen() == "phone"
    finally:
        mic.close()
