"""Lane 3: what does the wearer want to pick up?

    target = voice.listen_for_target()     # blocks, returns e.g. "mug"

Push-to-talk with Whisper running on the GX10 (faster-whisper), or a typed request as the backup
for a noisy hall. Either way the words then become an object name: Nemotron does it when it is
running (bash tools/run_nemotron.sh), so "I'm thirsty" becomes "water bottle"; a simple rule does
it otherwise, so "find my mug" still becomes "mug". Nothing leaves the machine.

Asking "what's on the table?" (or "what can I pick up?") makes Cosmos look at the camera picture
and say what it sees; the wearer then names one, or says "the second one". That needs the Cosmos
server (bash tools/run_cosmos.sh) and the real stream. It also does this by itself at the start and
after every grab ("I can see ... What do you want to grab?"), so the wearer is never left in
silence wondering whether it stopped. VOICE_AUTO_LIST=0 switches that off.

When it sees nothing to pick up it says why and what to do ("Turn towards the table, or step back
a little"), keeps looking by itself, and speaks the list as soon as there is one. Naming an object
always works, list or no list. After a long silence it says "I'm listening" (VOICE_REMIND=0
switches that off).

Other things that can be said: "stop" / "cancel" / "never mind", "again" / "repeat", "help",
"which option". "Option 3" (or "next option") changes the way guidance sounds without stopping
(guidance.OPTIONS); "hand option 2" changes the hand tracker, "grab option 2" the grab rule and
"object option 2" how the object is kept track of.

    VOICE_MODE=auto      a microphone on this machine if there is one; otherwise the web page below
                         if faster-whisper is installed; otherwise typed (default)
    VOICE_MODE=web       any phone or laptop is the microphone: it opens https://THIS_MACHINE:8644,
                         and you hold the button and speak. Typing in the terminal still works too.
    VOICE_MODE=type      always typed
    VOICE_MODE=speech    always the microphone on this machine
    VOICE_MODEL=base.en  Whisper model size (tiny.en is faster, small.en is more accurate)
    VOICE_DEVICE=NAME    sounddevice input device (the earbuds' microphone)
    VOICE_LLM_URL=URL    Nemotron's address (default http://127.0.0.1:8000/v1); "off" uses the rule only
    VOICE_HEADSET=off    do not use Bluetooth earbuds as the microphone
    VOICE_HEADSET_MODE=stay   keep the earbuds in headset mode for the whole run instead of switching to
                         stereo for every reach: nothing to go wrong between reaches, but everything is
                         heard in mono at telephone quality, so left and right tones mean nothing (default: switch)
    VOICE_HEADSET_VOLUME=0.6  how loud the earbuds are in headset mode (wpctl's scale); "match" (default)
                         copies the stereo volume, "keep" leaves it alone
    PROPRIO_KEEPER=1     set by the launcher: its keeper reconnects earbuds that lost a mode, so voice does not
    VOICE_REACH_LISTEN=1 with VOICE_HEADSET_MODE=stay: keep listening during a reach too, for "stop", "again",
                         "help" and the options, said by the wearer and nothing else. Without it those
                         words work during a reach only from the page or the keyboard: in stereo mode the
                         earbuds have no microphone.
    VOICE_DOWNLOAD=1     let Whisper be fetched from the internet; otherwise only what is on this machine is used
    VOICE_WEARER_LEVEL=0.015   how loud the wearer's own voice is in the earbuds (0 trusts every voice)
    VOICE_REQUEST_S=6    anything longer than this is a conversation, not a request

Bluetooth earbuds (AirPods) are used as the microphone by themselves when they are connected. They
cannot play stereo and record at once, so they are switched to headset mode only while it is
listening, which is whenever it is waiting for a request, and back to stereo for the guidance.
No button: it has just asked "What do you want to grab?", so you simply answer. If they drop out
it keeps looking for them and carries on through them when they are back.

Speech: press Enter, say "find my mug", press Enter again. A Bluetooth clicker that sends
Enter works as the button.

What it is doing is written once a second to out/status/voice.json. `python voice.py --check`
says whether everything it needs is on this machine, without loading anything.
"""
from __future__ import annotations

import atexit
import io
import json
import os
import queue
import re
import socket
import ssl
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

import tracing

SR = 16000          # what Whisper expects
MAX_S = 8.0         # stop recording after this even without the second Enter
ROOT = Path(__file__).resolve().parent

# How the wait for a request is paced. A blindfolded wearer has to hear that it is alive and what
# to do, but while it speaks it is not listening, and the same sentence over and over is noise.
# So each thing is said at once, again after a short gap, again after a longer one, and from
# then on only every three minutes.
LOOK_EVERY_S = 5.0                          # nothing to pick up, or no picture: look again this often (a look takes 1 to 2 s)...
LOOK_SLOWER_S = 15.0                        # ...and this often once a minute has gone by
LOOK_PATIENCE_S = 6.0                       # the first look is waited for this long before asking without a list
NOTHING_REMINDERS_S = (15.0, 40.0, 180.0)   # gaps before "still nothing"; the last gap repeats
SILENCE_REMINDERS_S = (20.0, 60.0, 180.0)   # gaps before "I'm listening"; the last gap repeats

# Telling the wearer from the room, measured through AirPods in the hall on 10 October: the wearer's
# requests came in at 0.03 to 0.09 (RMS of the cut-out recording) and lasted 2 to 4 s; people standing
# further off 0.006 to 0.012; a conversation next to the wearer was as loud as a request but 5 to 14 s long.
REQUEST_MAX_S = 7.5         # longer than this is not a request: it is dropped before Whisper hears it. A hesitant
                            # "um, okay, can you please find me the... blue cup" is 6.4 s; talk in the hall ran 7.7 s and up
CLIP_MAX_S = 8.0            # no recording runs longer than this, however long the hall goes on
LOUD_ROOM = 0.025           # a voice has to be 2.5 times the room; in a loud room, no more than this (or 1.5 times the room)
WEARER_LEVEL = 0.015        # quieter than this is trusted only when it names something that was just listed
COVERED_NEAR_M = 0.30       # the camera counts as covered when this much of the depth picture is nearer than this.
COVERED_SHARE = 0.05        # (in recordings/live_2025 a clear view of the table never had more than 2% that near)
FIND_EARBUDS_S = 3.0        # without earbuds, look for them this often; twice as long after each failure, up to...
FIND_EARBUDS_MAX_S = 30.0
CHECK_EARBUDS_S = 10.0      # while listening, make sure they are still there and still in headset mode
REACH_WORDS_S = 3.0         # during a reach only a word or two is listened for ("stop" came to 1.6 s with its pauses)
SWITCH_GAP_S = 3.0          # never two changes of the earbuds' mode closer than this: one started while the last was
                            # still coming up left them without stereo for minutes, three times on 10 October
SWITCH_WAIT_S = 6.0         # a mode that is not up after this has failed
KEEPER_WAIT_S = 20.0        # with the launcher's keeper running: how long to wait for it to bring stereo back
                            # (it acts after a mode has been missing for 10 s, and a reconnect takes a few more)

_LEAD = re.compile(
    r"^(?:(?:please|hey|ok|okay|can you|could you|i want you to|i want to|i want|i need|help me)\s+)*"
    r"(?:(?:find|get|grab|fetch|locate|pick up|pick|take|reach for|reach|look for|search for|"
    r"where is|where's|where are|give me|hand me|bring me|show me)\s+)?"
    r"(?:me\s+)?(?:(?:my|the|a|an|that|this|some|one of the)\s+)?")
_FILLER = {"please", "thanks", "thank you", "ok", "okay", "hey", "um", "uh", "it", "that", "this",
           "yes", "yeah", "yep", "sure", "no", "nope", "nothing", "never mind"}
_TAIL = re.compile(r"\s+(?:please|for me|now|thanks|thank you|on the table|on the desk|in front of me)$")


def extract_target(text: str) -> str:
    """ "Find my mug, please." -> "mug". Returns "" if nothing is left."""
    t = re.sub(r"[^a-z0-9' -]+", " ", text.lower())
    t = re.sub(r"\s+", " ", t).strip()
    t = _LEAD.sub("", t, count=1)
    prev = None
    while prev != t:                     # "mug please now" has two tails
        prev = t
        t = _TAIL.sub("", t)
    t = t.strip(" '-")
    return "" if t in _FILLER else t


def _plain(text: str) -> str:
    """ "Option 3, please." -> "option 3 please": what was said, without the punctuation."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]+", " ", text.lower())).strip()


# --- what it is doing, for out/status/voice.json and the log ----------------------

_DOING = {"starting": "Starting.", "listing": "Looking at what is on the table.", "asking": "Speaking to the wearer.",
          "listening": "Listening for a request.", "thinking": "Working out what was said.",
          "waiting": "Waiting for a request from the page or the keyboard.", "reaching": "A reach is going on."}
_status = {"state": "starting", "message": _DOING["starting"], "mic": "", "heard": "", "meant": "", "error": "", "error_t": None}
_status_lock = threading.Lock()
_trouble: dict = {}             # what is wrong right now, by part ("earbuds", "whisper"); the status is ok while this is empty
_problems: dict = {}            # the last fault of each part, so the same one is not printed again and again
_status_thread: threading.Thread | None = None
_alive_t = time.monotonic()     # when the wait for a request last moved


def _now(state: str | None = None, **more):
    """Note what it is doing."""
    global _alive_t
    _alive_t = time.monotonic()
    with _status_lock:
        if state is not None:
            _status.update(state=state, message=_DOING.get(state, state))
        _status.update(more)


def _problem(where: str, e, trouble: str | None = None):
    """Something failed and it carries on: one plain line in the log, a trace event, and the
    status file. `trouble` is a sentence for a person, and marks the status not ok until _fine()."""
    text = " ".join((e if isinstance(e, str) else f"{e.__class__.__name__}: {e}").split())[:240]
    if _problems.get(where) != text:
        print(f"[voice] {where}: {text}", file=sys.stderr)
    _problems[where] = text
    try:
        tracing.event("voice.error", where=where, error=text)
    except Exception:
        pass
    with _status_lock:
        _status.update(error=f"{where}: {text}", error_t=round(time.time(), 1))
        if trouble:
            _trouble[where] = trouble


def _fine(where: str):
    """That part works (again)."""
    _problems.pop(where, None)
    with _status_lock:
        was = _trouble.pop(where, None)
    if was:
        print(f"[voice] {where}: working again", file=sys.stderr)
        tracing.event("voice.recovered", where=where)


def status() -> dict:
    """What out/status/voice.json holds."""
    with _status_lock:
        s, trouble = dict(_status), list(_trouble.values())
    stuck = time.monotonic() - _alive_t
    if s["state"] in ("listing", "asking", "listening", "thinking", "waiting") and stuck > 60:
        trouble.append(f"Voice has not moved for {stuck:.0f} s.")      # a wait moves at least every few seconds
    return {"t": round(time.time(), 2), "state": s.pop("state"), "ok": not trouble,
            "message": " ".join(trouble) or s.pop("message"), **{k: v for k, v in s.items() if k != "message"}}


def _write_status(path: str | Path | None = None):
    path = Path(path or os.environ.get("VOICE_STATUS") or ROOT / "out" / "status" / "voice.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(status()))
    os.replace(tmp, path)                 # whoever reads it never sees half a file


def _keep_status():
    while True:
        try:
            _write_status()
        except Exception:
            pass
        time.sleep(1.0)


LLM_PROMPT = ('You work out what a blind person wearing a camera wants. Reply with JSON only: '
              '{"intent": "pick" or "list" or "none", "target": "<object>"}. '
              'intent is "list" when they ask what is there, what you can see, or what they could pick up. '
              'intent is "pick" when they want an object: then target is that object as 1 to 3 plain words an object '
              'detector can find, such as mug, water bottle, phone. Keep a colour or other word they used to tell it '
              'apart: blue cup stays blue cup, red phone stays red phone. Name the physical thing, not the wish: thirsty '
              'means water bottle. intent is "none" for anything else. target is an empty string unless intent is "pick".')
LLM_SCHEMA = {"type": "object",
              "properties": {"intent": {"type": "string", "enum": ["pick", "list", "none"]},
                             "target": {"type": "string", "maxLength": 30}},
              "required": ["intent", "target"], "additionalProperties": False}
_LIST = re.compile(r"\b(?:what(?:'s| is| are)?\b.*\b(?:on|in front|here|there|around|see|table|desk|pick up|grab|reach)"
                   r"|what do you see|what can you see|what can i|look around|describe|list (?:the )?(?:objects|things))\b")
_YES = re.compile(r"^(?:yes|yeah|yep|sure|ok|okay|please|that one|i want it|yes please|yes i do|get it)$")
_NUMBER = {w: i + 1 for i, w in enumerate("one two three four five six seven eight nine ten eleven twelve".split())}
_NUMBER_HEARD = {"won": 1, "to": 2, "too": 2, "for": 4, "ate": 8}       # what Whisper may write for a number said by itself
_OPTION = re.compile(rf"\b(?:(hand|grab|object) )?(?:option|mode|version|style|scheme) (?:number )?(\d{{1,2}}|{'|'.join(_NUMBER)})\b")
_OPTION_HEARD = re.compile(rf"^(?:(hand|grab|object) )?option (?:number )?({'|'.join(_NUMBER_HEARD)})$")
_OPTION_STEP = re.compile(r"^(next|previous) (?:(hand|grab|object) )?(?:option|one|mode|version|style)$")
_OPTION_WHICH = re.compile(r"^(?:what|which|what's)\b.{0,20}\b(?:option|mode)\b.{0,12}$")
# The whole of what was said has to be one of these, so a sentence from the room with "stop" in it is not a command.
_STOP = re.compile(r"^(?:(?:please|ok|okay|no) )*(?:stop|stop it|stop that|cancel|cancel it|cancel that|never mind|nevermind|"
                   r"forget it)(?: please| now)?$")
_AGAIN = re.compile(r"^(?:(?:please|can you|could you) )*(?:again|repeat|repeat that|say again|say that again|say it again|"
                    r"one more time|come again|what did you say)(?: please)?$")
_HELP = re.compile(r"^(?:help|help me|help please|what can i say|what can you do|how does this work)$")
HELP = ("Say the name of something to reach for it, or ask what is on the table. "
        "You can also say stop, again, or option and a number.")
_ORDINAL = re.compile(r"^(?:the |number )?(first|second|third|fourth|fifth|last|one|two|three|four|five|[1-5])(?: one)?$")
_ORDER = {"first": 0, "one": 0, "1": 0, "second": 1, "two": 1, "2": 1, "third": 2, "three": 2, "3": 2,
          "fourth": 3, "four": 3, "4": 3, "fifth": 4, "five": 4, "5": 4, "last": -1}
last_transcript: str | None = None      # what was last said or typed, for the dashboard
last_objects: list[str] = []            # what Cosmos last listed, so "the second one" means something
_llm_ok: bool | None = None             # None until the first request shows whether Nemotron is there
_llm_again_t = 0.0                      # after it failed to answer: when to try it again
LLM_PAUSE_S = 30.0                      # how long the simple rule stands in before Nemotron is tried again


def ask_llm(text: str, url: str, timeout_s: float = 4.0) -> tuple[str, str]:
    """Nemotron's reading of the request: (intent, target). Raises if the server is not there."""
    body = {"model": "nemotron", "temperature": 0.0, "max_tokens": 40,
            "chat_template_kwargs": {"enable_thinking": False},      # the answer is one word: no need to reason aloud
            "messages": [{"role": "system", "content": LLM_PROMPT}, {"role": "user", "content": text}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "target", "schema": LLM_SCHEMA, "strict": True}}}
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        reply = json.load(r)
    d = json.loads(reply["choices"][0]["message"]["content"])
    intent = d.get("intent", "pick")
    if intent not in ("pick", "list", "none"):
        raise ValueError(f"unknown intent {intent!r}")
    return intent, (extract_target(str(d.get("target", ""))) if intent == "pick" else "")


def interpret(text: str) -> tuple[str, str]:
    """Words -> (intent, target). intent is "pick" (target names the object), "list" (they asked
    what is there) or "none". Short requests go by rule; longer or indirect ones ("I'm thirsty")
    go to Nemotron if it is running, and the rule steps in when it is not."""
    global _llm_ok, _llm_again_t, last_transcript
    last_transcript = text
    clean = re.sub(r"[^a-z0-9' ]+", " ", text.lower()).strip()
    m = _ORDINAL.match(clean)
    if m and last_objects:                                   # "the second one", after a list
        i = _ORDER[m.group(1)]
        if -1 <= i < len(last_objects):
            return "pick", last_objects[i]
    listed = [o for o in last_objects if re.search(rf"\b{re.escape(o)}\b", clean)]
    if listed and not _LIST.search(clean):                   # they said one of the names they were just told
        return "pick", max(listed, key=len)
    by_rule = ("list", "") if _LIST.search(clean) else ("pick", extract_target(text))
    if len(by_rule[1].split()) > 4:
        by_rule = ("none", "")                               # the rule cannot read a whole sentence, and it is no object's name
    url = os.environ.get("VOICE_LLM_URL", "http://127.0.0.1:8000/v1")
    if url == "off" or len(clean.split()) < 2:               # one word is already a name
        return by_rule
    if _llm_ok is False and time.monotonic() < _llm_again_t:
        return by_rule                                       # it did not answer a moment ago: do not make every request wait for it
    t = time.perf_counter()
    try:
        intent, target = ask_llm(text, url)
    except Exception as e:
        # Not there, too slow, or an answer that makes no sense: the rule reads this request. Only a
        # server that does not answer is left alone for a while; it is tried again after that, so one
        # slow answer does not switch Nemotron off for the rest of the run.
        late = isinstance(e, TimeoutError) or isinstance(getattr(e, "reason", None), TimeoutError) or "timed out" in str(e)
        if late or isinstance(e, urllib.error.HTTPError):
            # It is there, but was too slow this once (the GPU was busy) or answered with an error:
            # ask it again next time.
            _problem("nemotron", f"no usable answer this time ({e.__class__.__name__}: {e}); the simple rule read this request")
        elif isinstance(e, OSError):
            _llm_ok, _llm_again_t = False, time.monotonic() + LLM_PAUSE_S
            _problem("nemotron", f"not answering at {url} ({e.__class__.__name__}); using the simple rule and trying again "
                                 f"in {LLM_PAUSE_S:.0f} s. Start it with: bash tools/run_nemotron.sh")
        else:
            _problem("nemotron", f"an answer that could not be read ({e.__class__.__name__}: {e}); the simple rule read this request")
        return by_rule
    if _llm_ok is False:
        _fine("nemotron")
    _llm_ok = True
    _problems.pop("nemotron", None)
    tracing.event("nemotron", (time.perf_counter() - t) * 1000, text=text, intent=intent, target=target)
    if by_rule[0] == "list" and intent != "pick":            # the rule is sure about this one
        return "list", ""
    return intent, target


def understand(text: str) -> str:
    """Words -> object name, or "" if they did not ask for one."""
    intent, target = interpret(text)
    return target if intent == "pick" else ""


_said = 0               # how many sentences have been spoken, so the listener knows when it has just said something
_said_t = 0.0           # when it last finished speaking: the reminders count from here
_last_said = ""         # the last thing it told the wearer, for "again"
_reminders = 0          # reminders since it last said anything else


def _say(text: str, reminder: bool = False):
    global _said, _said_t, _last_said, _reminders
    _said += 1
    _reminders = _reminders + 1 if reminder else 0
    if not reminder:
        _last_said = text
    print(f"[voice] {text}")
    _now("asking")
    try:
        import guidance
        guidance.say_text(text, wait=True)       # finish speaking before the microphone opens again
    except Exception as e:
        print(f"[voice] could not speak ({e})", file=sys.stderr)
    _said_t = time.monotonic()


def _beep():
    """The short sound that means "I am listening now"."""
    try:
        import guidance
        guidance.say_clip("listen", wait=True)
    except Exception:
        pass


def _camera_frame(wait_s: float = 2.0):
    """A fresh frame from the stream main.py is using, or None if it is not the real stream."""
    module = sys.modules.get("stream")
    if module is None:
        return None                      # main.py is running on the fake stream
    try:
        if module._default is None:
            # The first question comes before main.py has asked for a single frame, so the stream
            # has not been opened yet. Open it now; main.py will carry on with the same one.
            module.start()
        s = module._default
        end = time.monotonic() + wait_s
        frame = None
        while time.monotonic() < end:
            got = s.get_frame(wait=0.3)
            if got is not None:
                frame = got
                if time.monotonic() - got.t_recv < 0.2:
                    break
        return frame
    except Exception as e:
        _problem("camera", e)
        return None


def covered(depth) -> bool:
    """Is something right in front of the camera (the wearer's own arm, a laptop lid)? Only depth
    that was measured counts: no reading at all is what a far wall gives too."""
    if depth is None:
        return False
    d = np.nan_to_num(np.asarray(depth, np.float32)[::4, ::4])
    return d.size > 0 and float(np.mean((d > 0) & (d < COVERED_NEAR_M))) >= COVERED_SHARE


def look() -> tuple[str, list[str]]:
    """Look through the camera with Cosmos: ("ok", names), or why there is nothing to tell.
    "nothing": Cosmos sees nothing to pick up; in the recordings that was always a camera not
    pointing at the table. "covered": the same, with something right in front of the camera.
    "no_camera": no picture has arrived. "no_cosmos": Cosmos is not answering. "no_stream": this
    run has no real camera at all."""
    if sys.modules.get("stream") is None:
        return "no_stream", []
    frame = _camera_frame()
    if frame is None:
        return "no_camera", []
    try:
        import cosmos
        if not cosmos.available():
            _problem("cosmos", "not answering. Start it with: bash tools/run_cosmos.sh")
            return "no_cosmos", []
        objects = cosmos.list_objects(frame.rgb)
    except Exception as e:
        _problem("cosmos", e)
        return "no_cosmos", []
    _problems.pop("cosmos", None)
    if objects:
        return "ok", list(objects)
    return ("covered" if covered(getattr(frame, "depth", None)) else "nothing"), []


_WHY = {"nothing": "I can't see anything to pick up. Turn towards the table, or step back a little.",
        "covered": "Something is right in front of the camera. Move your arm away, or step back a little.",
        "no_camera": "I have no picture from the camera yet."}
_told_no_cosmos = False     # "I can't look around" has been said: it is not said again before every question
_looked = 0                 # how many looks have been told to the wearer
_seen = ""                  # what the last of them found


def _tell(seen: str, objects: list[str], asked: bool, keeps_looking: bool = False):
    """Say what a look found, or why it found nothing and what the wearer can do about it.
    asked=False is the unprompted version at the start and after a grab."""
    global last_objects, _told_no_cosmos, _looked, _seen
    _looked, _seen = _looked + 1, seen
    tracing.event("voice.look", seen=seen, objects=objects, asked=asked)
    if seen == "ok":
        last_objects, _told_no_cosmos = objects, False
        if len(objects) == 1:
            _say(f"I can see one thing: {objects[0]}. Do you want it?")
        else:
            _say("I can see " + ", ".join(objects[:-1]) + " and " + objects[-1] + ". "
                 + ("Which one?" if asked else "What do you want to grab?"))
    elif seen == "no_stream":
        if asked:
            _say("I need the camera for that.")
    elif seen == "no_cosmos":
        if asked:
            _say("I can't look around right now.")
        else:                                    # said once; after that it simply asks
            _say("What do you want to grab?" if _told_no_cosmos else "I can't look around right now. Tell me what you want to grab.")
            _told_no_cosmos = True
    else:
        if seen != "no_camera":
            last_objects = []
        _say(_WHY[seen] if asked else _WHY[seen] + (" I'll keep looking." if keeps_looking else " What do you want?"))


def describe(asked: bool = True) -> list[str]:
    """Look through the camera with Cosmos and tell the wearer what could be picked up.
    asked=False is the unprompted version at the start and after a grab: it ends with
    "What do you want to grab?"."""
    _now("listing")
    seen, objects = look()
    _tell(seen, objects, asked)
    return objects


_ask_next = True        # speak up before the next wait: at the start, and after each target (so after each grab)
_stopped = False        # the wearer said "stop": the reach that was going on is over, whatever guidance still holds


def invite(pending: bool = False):
    """Called before waiting for a request. If the wearer has just finished (or we have just
    started) and has not already said something, look around and ask what they want."""
    global _ask_next
    if _ask_next and not pending and os.environ.get("VOICE_AUTO_LIST", "1") != "0":
        _ask_next = False
        describe(asked=False)


def _carry_on() -> str:
    """What main.py should look for after a remark that is not a new request: the same thing
    again if a reach is going on, so it carries on."""
    if _stopped:
        return ""
    try:
        import guidance
        return guidance.current_target() or ""
    except Exception:
        return ""


def _family(kind: str | None):
    """One kind of option the wearer can switch: (what it is called aloud, its options, the
    function that chooses one, the function that tells which is on). The hand and grab options
    belong to perception.py and guidance.py; where they do not exist the functions are None."""
    import guidance
    if kind == "hand":
        module = sys.modules.get("perception")           # never loaded just for this: with fake perception there is no hand tracker to switch
        return ("hand option", getattr(module, "HAND_OPTIONS", None), getattr(module, "set_hand_option", None),
                getattr(module, "hand_option", None))
    if kind == "object":
        module = sys.modules.get("perception")
        return ("object option", getattr(module, "OBJECT_OPTIONS", None), getattr(module, "set_object_option", None),
                getattr(module, "object_option", None))
    if kind == "grab":
        return ("grab option", getattr(guidance, "GRAB_OPTIONS", None), getattr(guidance, "set_grab_option", None),
                getattr(guidance, "grab_option", None))
    return "option", guidance.OPTIONS, guidance.set_option, guidance.option


def switch_option(text: str) -> str | None:
    """ "option 3", "next option", "hand option 2", "grab option 2", "which option": change how
    guidance sounds, how the hand is tracked or when it says grab, or say which is on. Returns
    what to look for next (the same target, so a reach in progress carries on), or None if this
    was not such a request."""
    clean = _plain(text)
    words = clean.split()
    m = _OPTION_STEP.match(clean) or _OPTION_HEARD.match(clean) or (_OPTION.search(clean) if len(words) <= 7 else None)
    which = not m and len(words) <= 7 and _OPTION_WHICH.match(clean)
    if not m and not which:
        return None
    kind = (m.group(2 if m.re is _OPTION_STEP else 1) if m else None) or next((k for k in ("hand", "grab", "object") if k in words), None)
    name, options, choose, current = _family(kind)
    said = name.capitalize()
    try:
        count = len(options) if options else 0
        if not count or choose is None:
            _say(f"{said}s are not available here.")
        elif which:
            number, _, title = current()
            _say(f"{said} {number}: {title}." if number else f"This is {title}, which is not one of the {count} {name}s.")
        else:
            if m.re is _OPTION_STEP:
                number = (current()[0] - 1 + (1 if m.group(1) == "next" else -1)) % count + 1
            else:
                number = int(m.group(2)) if m.group(2).isdigit() else (_NUMBER.get(m.group(2)) or _NUMBER_HEARD[m.group(2)])
            if not 1 <= number <= count:
                _say(f"There are {count} {name}s.")
            else:
                title = choose(number)
                if not isinstance(title, str) or not title:
                    title = current()[2]
                _say(f"{said} {number}: {title}.")
    except Exception as e:                       # the other modules' option code is new tonight
        _problem("option", e)
        _say("I could not switch that.")
    return _carry_on()


def switch_option_asked(clean: str) -> bool:
    """Is this (already without punctuation) about the options?"""
    return bool(_OPTION_STEP.match(clean) or _OPTION_HEARD.match(clean) or _OPTION.search(clean) or _OPTION_WHICH.match(clean))


def respond(text: str) -> str:
    """What to do with what was said: the target to look for, or "" after answering a question
    (main.py then asks again, and the wearer names their pick)."""
    global last_transcript, _ask_next, _stopped
    clean = _plain(text)
    switched = switch_option(text)
    if switched is not None:
        last_transcript = text
        _now(heard=text, meant="an option")
        return switched
    if _STOP.match(clean):
        last_transcript = text
        reaching = bool(_carry_on())
        _stopped, _ask_next = True, False                    # straight back to the question; they know what is there
        _now(heard=text, meant="stop")
        _say("Stopped. What do you want to grab?" if reaching else "Okay. What do you want to grab?")
        return ""
    if _AGAIN.match(clean) or _HELP.match(clean):
        last_transcript = text
        again = bool(_AGAIN.match(clean)) and bool(_last_said)
        _now(heard=text, meant="say it again" if again else "help")
        _say(_last_said if again else HELP)
        return _carry_on()
    if last_objects and _YES.match(re.sub(r"[^a-z' ]+", " ", text.lower()).strip()):
        last_transcript = text
        _ask_next, _stopped = True, False
        _now(heard=text, meant=f"reach for {last_objects[0]}")
        return last_objects[0]                               # "yes" to "Do you want it?"
    intent, target = interpret(text)
    if intent == "list":
        _now(heard=text, meant="what is on the table")
        describe()
        return ""
    if intent == "pick" and target:
        _ask_next, _stopped = True, False                    # after this reach, speak up again
        _now(heard=text, meant=f"reach for {target}")
        return target
    _now(heard=text, meant="nothing for me")
    return ""


_no_keyboard = False


def typed() -> str:
    global _no_keyboard
    invite()
    _now("waiting", mic="keyboard")
    try:
        return respond(input("what do you want to pick up?> "))
    except (EOFError, OSError):
        if not _no_keyboard:
            _no_keyboard = True
            _problem("keyboard", "nothing can be typed here (no terminal), and there is no microphone",
                     "There is no microphone and no keyboard: nothing can be asked for.")
        time.sleep(1.0)                  # main.py asks again at once; without this it would spin
        return ""


def _offline() -> bool:
    """Only what is already on this machine may be loaded: no download in the middle of a demo."""
    return os.environ.get("VOICE_DOWNLOAD", "0") != "1"


def _not_here(what: str, e: Exception) -> RuntimeError:
    return RuntimeError(f"{what} is not on this machine ({e.__class__.__name__}: {' '.join(str(e).split())[:120]}). "
                        "Fetch it once while online: VOICE_DOWNLOAD=1 python -c 'import voice; voice.load_whisper()'")


_whisper_lock = threading.Lock()         # the page and the earbuds may hand it a recording at the same moment


def load_whisper(model: str | None = None):
    """Whisper as a function: audio (mono float32 at 16 kHz) -> text. On the GPU through PyTorch
    when there is one (about 0.05 s a request on the GX10). Otherwise faster-whisper on the CPU
    (about 1.2 s): its build for the GX10's processor has no GPU support. VOICE_STT=cpu or gpu
    forces one. Either way the stretches without speech are cut out first, because Whisper
    writes sentences of its own when handed silence. With no model named it is large-v3 on the
    GPU (12 of 12 test clips right, 0.18 s) and small.en on the CPU. The model has to be on this
    machine already (VOICE_DOWNLOAD=1 allows fetching it): the venue's network may be down, and
    a download that hangs is worse than a clear message at the start."""
    want = os.environ.get("VOICE_STT", "auto")
    why = None
    for attempt in (1, 2):
        if want == "cpu":
            break
        try:
            hear = _gpu_whisper(model or "large-v3")
            _now(whisper=f"{model or 'large-v3'} on the GPU")
            return hear
        except Exception as e:
            if want == "gpu":
                raise
            why = e
            if attempt == 2 or not isinstance(e, RuntimeError) or "not on this machine" in str(e) or "no GPU" in str(e):
                break
            time.sleep(3.0)               # a GPU busy loading the other models is worth one more try
    from faster_whisper import WhisperModel
    try:
        whisper = WhisperModel(model or "small.en", device="cpu", compute_type="int8", local_files_only=_offline())
    except Exception as e:
        raise _not_here(f"Whisper {model or 'small.en'} for the CPU" + (f" (and the GPU one failed: {why})" if why else ""), e) from e
    _now(whisper=f"{model or 'small.en'} on the CPU")
    if why is not None:
        # Not the same thing more slowly: the small model hears a soft "blue cup" as "look up". Say so loudly.
        _problem("whisper model", f"Whisper could not start on the GPU ({why.__class__.__name__}: {why}); running "
                                  f"{model or 'small.en'} on the CPU instead, which hears worse and takes about a second",
                 f"Whisper is on the CPU with the small model, not on the GPU: it will mishear more ({why.__class__.__name__}).")
        tracing.event("voice.whisper_on_cpu", why=f"{why.__class__.__name__}: {why}"[:200])

    def hear(audio: np.ndarray) -> str:
        with _whisper_lock:
            segments, _ = whisper.transcribe(audio, language="en", beam_size=1, vad_filter=True, hotwords=expected_words())
            return " ".join(s.text for s in segments).strip()
    hear.on = "cpu"
    return hear


def _whisper_straight_to_gpu(name: str):
    """Whisper with its weights read from the file directly onto the GPU: 0.7 s on the GX10,
    against 13 to 17 s for loading them into memory first and moving them (large-v3). That is
    most of the time the pipeline takes to come back after a restart. None if anything about
    it is not exactly as expected; the caller then loads the ordinary way."""
    try:
        import torch
        from accelerate import init_empty_weights
        from safetensors.torch import load_file
        from transformers import GenerationConfig, WhisperConfig, WhisperForConditionalGeneration
        from transformers.utils import cached_file
        path = cached_file(name, "model.safetensors", local_files_only=True)
        with init_empty_weights():
            net = WhisperForConditionalGeneration(WhisperConfig.from_pretrained(name, local_files_only=True))
        weights = {k: v.to(torch.float16) for k, v in load_file(path, device="cuda").items()}
        missing, unexpected = net.load_state_dict(weights, strict=False, assign=True)
        net.tie_weights()
        # The output layer shares the word table's weights and so is not in the file. Anything
        # else missing, or anything left over, means this is not the model we think it is.
        tied = net.proj_out.weight is net.model.decoder.embed_tokens.weight
        if set(missing) - {"proj_out.weight"} or unexpected or not tied:
            raise ValueError(f"missing {missing[:3]}, unexpected {unexpected[:3]}, tied {tied}")
        net.generation_config = GenerationConfig.from_pretrained(name, local_files_only=True)
        return net
    except Exception as e:
        print(f"[voice] Whisper will load the slow way ({e.__class__.__name__}: {e})", file=sys.stderr)
        return None


def _gpu_whisper(model: str):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("no GPU here")
    from faster_whisper.vad import get_speech_timestamps
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    name = model if "/" in model else f"openai/whisper-{model}"
    try:
        proc = WhisperProcessor.from_pretrained(name, local_files_only=_offline())
        net = _whisper_straight_to_gpu(name)
        if net is None:
            net = WhisperForConditionalGeneration.from_pretrained(name, torch_dtype=torch.float16, local_files_only=_offline())
            net = net.to("cuda")
    except OSError as e:
        raise _not_here(f"Whisper {model}", e) from e
    net = net.eval()
    fixed = {} if name.endswith(".en") else {"language": "en", "task": "transcribe"}

    def write(audio: np.ndarray, hint: str | None) -> str:
        kw = dict(fixed)
        if hint:
            kw["prompt_ids"] = proc.get_prompt_ids(hint, return_tensors="pt").to("cuda")
        feats = proc(audio, sampling_rate=SR, return_tensors="pt").input_features.to("cuda", torch.float16)
        with torch.no_grad():
            ids = net.generate(feats, max_new_tokens=60, num_beams=1, do_sample=False, **kw)
        text = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
        return text[len(hint):].strip() if hint and text.startswith(hint) else text

    def hear(audio: np.ndarray) -> str:
        speech = get_speech_timestamps(audio)
        if not speech:
            return ""
        spoken = np.concatenate([audio[c["start"]:c["end"]] for c in speech])
        # The names help with a short answer ("blue cup"). On a long sentence they pull the
        # whole transcript towards themselves, so there Whisper is left alone.
        with _whisper_lock:
            try:
                text = write(spoken, expected_words() if spoken.size <= 4 * SR else None)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()                  # give back what the failed attempt took, then say what happened
                raise
        if not plausible(text):
            _keep_rejected(audio, text)
            return ""
        return text
    write(np.zeros(SR, np.float32), None)                 # the first request is slow; get it over with now
    hear.on = "gpu"
    return hear


class Listener:
    """Records between two presses of Enter and transcribes with Whisper."""

    def __init__(self, model: str | None = None, device=None):
        import sounddevice as sd

        self.sd = sd
        self.device = device
        sd.check_input_settings(device=device, samplerate=SR, channels=1)   # fails if there is no microphone
        with tracing.timed("load.whisper", model=model):
            self.whisper = load_whisper(model)

    def record(self) -> np.ndarray:
        chunks = []
        stop = threading.Event()
        input("[voice] press Enter, then speak> ")
        with self.sd.InputStream(samplerate=SR, channels=1, dtype="float32", device=self.device,
                                 callback=lambda data, n, t, status: chunks.append(data[:, 0].copy())):
            threading.Thread(target=lambda: (input("[voice] listening... press Enter when done> "), stop.set()),
                             daemon=True).start()
            stop.wait(MAX_S)
        return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size < SR // 4:
            return ""
        return self.whisper(audio)

    def listen(self) -> str:
        invite()
        _now("listening", mic="a microphone on this machine")
        audio = self.record()
        t = time.perf_counter()
        text = self.transcribe(audio)
        tracing.event("whisper", (time.perf_counter() - t) * 1000, audio_s=round(audio.size / SR, 2), text=text)
        target = respond(text)
        print(f"[voice] heard {text!r} -> target {target!r}")
        return target


PAGE = ROOT / "voice_page.html"


def _poke_main():
    """A request arrived while a reach was in progress: press main.py's "new target" key through
    the dashboard, so the wearer can change their mind without waiting for a grab."""
    port = os.environ.get("DASHBOARD_PORT") or (sys.argv[sys.argv.index("--port") + 1] if "--port" in sys.argv[:-1] else "8642")
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/key", json.dumps({"key": "t"}).encode(),
                                     {"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=0.5).close()
    except Exception:
        pass                             # no dashboard: the request is picked up after the grab


def _my_address() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))   # sends nothing; just asks which interface would be used
            return s.getsockname()[0]
    except OSError:
        return socket.gethostname()


# Things Whisper writes when it is handed room noise instead of speech.
# The second line is the rest of the most common ones measured for large-v3 (arXiv 2501.11378).
_NOT_SPEECH = {"", "you", "thank you", "thanks", "thanks for watching", "bye", "okay", "ok", "so", "the", "uh", "um", "hmm",
               "thank you for watching", "oh", "im sorry", "oh my god", "bye bye", "thank you very much",
               "subtitles by the amaraorg community"}


def expected_words() -> str | None:
    """The names Whisper should expect: what was just listed. Without them the small models
    hear "blue cup" as "look up"."""
    return ", ".join(last_objects) if last_objects else None


def plausible(text: str) -> bool:
    """False for what Whisper writes when it loses the thread: one letter or phrase repeated
    ("FFFFFFFF", "the the the the"). Such text packs down far better than speech does."""
    import zlib
    raw = text.strip().encode()
    if len(raw) < 20:
        return True
    return max(len(w) for w in text.split()) <= 25 and len(raw) / len(zlib.compress(raw)) <= 2.4


def _keep_rejected(audio: np.ndarray, text: str):
    """The last recording that came out as nonsense, kept so it can be listened to."""
    try:
        import wave
        path = ROOT / "out" / "heard_rejected.wav"
        path.parent.mkdir(exist_ok=True)
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
        tracing.event("whisper.rejected", text=text[:60], audio_s=round(audio.size / SR, 2))
    except Exception:
        pass


# Words that name nothing. On 11 October a stray "and" from the room was taken for an object
# and searched for for forty seconds.
_NO_NAME = {"and", "a", "an", "the", "but", "or", "so", "well", "right", "it", "is", "its", "this", "that", "i", "to", "of",
            "in", "on", "uh", "um", "hmm", "mm", "mhm", "huh", "oh", "wow", "hi", "hey", "hello", "alright", "then", "now"}


def worth_hearing(text: str) -> bool:
    clean = re.sub(r"[^a-z ]+", "", text.lower()).strip()
    return plausible(text) and clean not in _NOT_SPEECH and not all(w in _NO_NAME for w in clean.split())


def loudness(audio: np.ndarray) -> float:
    """How loud a cut-out recording is: the same number the trace keeps as "level"."""
    return float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0


def from_the_wearer(text: str, level: float) -> bool:
    """Was this said by the wearer? A voice as loud as the wearer's own in the earbuds is taken
    at its word. A quieter one is somebody further off, or the wearer speaking softly: it counts
    only when it answers what was just asked, by naming something from the list ("blue cup",
    "the bottle", "the second one") or asking for the list. With nothing listed there is nothing
    to check it against, and every voice is heard as before."""
    if level >= float(os.environ.get("VOICE_WEARER_LEVEL", WEARER_LEVEL)) or not last_objects:
        return True
    clean = _plain(text)
    if _ORDINAL.match(clean) or _LIST.search(clean):
        return True
    if len(last_objects) == 1 and _YES.match(clean):         # "Do you want it?" was the question: "yes" answers it
        return True
    said = set(clean.split())
    return any(said & set(name.split()) for name in last_objects)


def voice_level(history, min_level: float = 0.005) -> float:
    """How loud a block has to be to count as a voice, given the levels of the room lately. The
    room is the quiet fifth of the last ten seconds, so a loud word from the hall's speakers at
    the wrong moment does not deafen the next few seconds. A voice is 2.5 times the room; in a
    loud room that would ask for more than the wearer's own voice gives (0.03 and up), so there
    it is LOUD_ROOM, or 1.5 times the room once the room itself is louder than that."""
    room = float(np.percentile(history, 20)) if len(history) else 0.0
    return max(min_level, min(2.5 * room, max(LOUD_ROOM, 1.5 * room)))


def cut_speech(blocks, sr: int = SR, block_s: float = 0.03, start_wait_s: float = 6.0, max_s: float = CLIP_MAX_S,
               quiet_s: float = 0.9, should_stop=None, min_level: float = 0.005, own_share: float = 0.5,
               history=None, abandon=None, info: dict | None = None) -> np.ndarray | None:
    """Take audio blocks as they arrive and return the stretch that holds speech: from a little
    before the voice starts until it has been quiet for `quiet_s`, or `max_s` after it started.
    None if nobody spoke within `start_wait_s`, or `should_stop()` said to give up, or
    `abandon()` did in the middle of a sentence. The level that counts as a voice comes from
    the room's own level (see voice_level; `history` carries it from one call to the next), and
    is never below `min_level`: earbuds silence the room, and a soft voice measured 0.006 to
    0.010 through them. Once somebody speaks, "quiet" is measured against that voice: anything
    under `own_share` of the loudest level it held is the room again. So the wearer's request ends when
    the wearer stops, though people further off go on talking, and what they said before the
    wearer started is cut off the front. `info` is filled with how it ended."""
    history = deque(maxlen=333) if history is None else history      # ten seconds of block levels
    info = {} if info is None else info
    kept, levels, lead, lead_levels = [], [], [], []
    loud, quiet, started, t, t_voice, top, threshold = 0, 0.0, False, 0.0, 0.0, 0.0, min_level
    info.update(ended="nobody spoke")
    for block in blocks:
        t += block_s
        level = float(np.sqrt(np.mean(np.square(block)))) if len(block) else 0.0
        if not started:
            history.append(level)
            lead = (lead + [block])[-10:]                 # keep 0.3 s from before the voice
            lead_levels = (lead_levels + [level])[-10:]
            if t <= 0.3:
                continue                                  # always a moment of the room first
            threshold = voice_level(history, min_level)
            loud = loud + 1 if level > threshold else 0
            if loud >= 4:                                 # 120 ms of voice, not a click
                started, kept, levels, t_voice = True, list(lead), list(lead_levels), t
                for _ in range(4):
                    history.pop()                         # those four were the voice, not the room
            elif t > start_wait_s or (should_stop is not None and should_stop()):
                return None
            continue
        kept.append(block)
        levels.append(level)
        top = max(top, float(np.median(levels[-10:])))    # the loudest level held for a third of a second: a voice,
                                                          # not a click or a hand brushing the earbud
        quiet = quiet + block_s if level <= max(threshold, own_share * top) else 0.0
        if quiet >= quiet_s:
            info.update(ended="quiet")
            break
        if t - t_voice > max_s:
            # Never a pause: the hall itself is this loud. Let that count as the room from now
            # on, or every next recording would start at once and run its full length too.
            info.update(ended="max")
            history.extend(levels)
            break
        if abandon is not None and abandon():
            info.update(ended="abandoned")
            return None
    else:
        info.update(ended="the sound stopped" if started else info["ended"])
    if not started or not kept:
        return None
    bar = max(threshold, own_share * top)
    first = max(0, next((i for i, level in enumerate(levels) if level > bar), 0) - 10)
    info.update(voice_at=round(t_voice - 4 * block_s, 2), skipped_s=round(first * block_s, 2), threshold=round(threshold, 4))
    return np.concatenate(kept[first:])


class MicLost(Exception):
    """The earbuds' microphone has stopped giving sound."""


_switch_lock = threading.RLock()         # one change of the earbuds' mode at a time, anywhere in this process
_switch_t = -1e9                         # when the last one was asked for
_repair_t = -1e9
_stereo_volume: float | None = None      # how loud the wearer has the earbuds in stereo, read before leaving it
_nodes_trusted = True                    # the sound nodes look the way _is_up expects for these earbuds


def _wants_earbuds() -> bool:
    """Would this machine use Bluetooth earbuds as the microphone if there were any?"""
    import shutil
    return os.environ.get("VOICE_HEADSET", "auto") != "off" and bool(shutil.which("wpctl") and shutil.which("pw-cli"))


def _stays() -> bool:
    return os.environ.get("VOICE_HEADSET_MODE", "switch") == "stay"


class Headset:
    """Bluetooth earbuds as the microphone, through PipeWire. In stereo mode they have no
    microphone at all; in headset mode they have one but the sound is mono and coarse. So the
    mode is switched just for the moments it listens (or, with VOICE_HEADSET_MODE=stay, once)."""

    def __init__(self, device: int, headset: int, stereo: int, name: str = "earbuds"):
        self.device, self.headset, self.stereo, self.name = device, headset, stereo, name
        self.history: deque = deque(maxlen=333)       # the room's level lately, for cut_speech
        self._zeros_s = 0.0               # how long nothing but zeros has come in
        self._ran_on = False              # the last recording ended at its full length, with the talk still going

    @staticmethod
    def find():
        """The connected earbuds, or None. Reads PipeWire's device list and their profiles."""
        import shutil
        if os.environ.get("VOICE_HEADSET", "auto") == "off" or not (shutil.which("wpctl") and shutil.which("pw-cli")):
            return None
        try:
            status = subprocess.run(["wpctl", "status"], capture_output=True, text=True, timeout=3).stdout
            devices = status.split("Audio", 1)[1].split("Sinks:", 1)[0]
            found = list(re.finditer(r"(\d+)\.\s+(.+?)\s+\[bluez5\]", devices))
            if not found:
                return None
            m = found[0]
            wanted = os.environ.get("PROPRIO_HEADSET", "").strip().upper()
            if wanted and len(found) > 1:
                # Two headsets are connected (somebody's earbuds came out of their case): the one
                # named in the settings is the one. If it is not among them, the first will do.
                for other in found:
                    about = subprocess.run(["wpctl", "inspect", other.group(1)], capture_output=True, text=True, timeout=3).stdout
                    if wanted in about.upper() or wanted.replace(":", "_") in about.upper():
                        m = other
                        break
            params = subprocess.run(["pw-cli", "enum-params", m.group(1), "EnumProfile"], capture_output=True,
                                    text=True, timeout=3).stdout
        except Exception:
            return None
        found = dict(reversed(pair) for pair in re.findall(r'Profile:index.*?Int (\d+).*?Profile:name.*?String "([^"]+)"', params, re.S))
        headset = found.get("headset-head-unit-msbc") or found.get("headset-head-unit")
        stereo = found.get("a2dp-sink")
        if headset is None or stereo is None:
            return None
        return Headset(int(m.group(1)), int(headset), int(stereo), m.group(2).strip())

    def _profile(self, index: int):
        subprocess.run(["wpctl", "set-profile", str(self.device), str(index)], capture_output=True, timeout=3)

    def mode(self) -> int | None:
        """The profile they are in right now, or None if that cannot be read."""
        try:
            out = subprocess.run(["pw-cli", "enum-params", str(self.device), "Profile"], capture_output=True,
                                 text=True, timeout=3).stdout
        except Exception:
            return None
        m = re.search(r"Profile:index.*?Int (\d+)", out, re.S)
        return int(m.group(1)) if m else None

    def _nodes(self) -> dict | None:
        """The earbuds' sound nodes as PipeWire has them right now: {"Audio/Sink": id, "Audio/Source": id},
        with the source only in headset mode. None if that cannot be read."""
        try:
            dump = json.loads(subprocess.run(["pw-dump"], capture_output=True, text=True, timeout=3).stdout)
            out = {}
            for thing in dump:
                props = (thing.get("info") or {}).get("props") or {}
                if props.get("device.id") == self.device and props.get("media.class") in ("Audio/Sink", "Audio/Source"):
                    out[props["media.class"]] = thing["id"]
            return out
        except Exception:
            return None

    def _is_up(self, index: int) -> bool | None:
        """Are they in that mode, with its sound nodes really there? None if it cannot be told."""
        mode = self.mode()
        if mode is None:
            return None
        if mode != index:
            return False
        nodes = self._nodes() if _nodes_trusted else None
        if nodes is None:
            return None
        return "Audio/Sink" in nodes and ("Audio/Source" in nodes) == (index == self.headset)

    def _switch(self, index: int) -> bool | None:
        """Change mode and wait until it has really happened, rather than for a fixed moment:
        True when it is up, False when it did not come up in SWITCH_WAIT_S, None when this
        machine cannot tell (then it waits the fixed moment, as before)."""
        global _switch_t, _stereo_volume, _nodes_trusted
        to = "headset" if index == self.headset else "stereo"
        with _switch_lock:
            t0 = time.monotonic()
            if self._is_up(index):
                return True                               # already there: no switch to get wrong
            if index == self.headset and self.mode() == self.stereo:
                _stereo_volume = self._volume() or _stereo_volume
            time.sleep(max(0.0, _switch_t + SWITCH_GAP_S - time.monotonic()))
            self._profile(index)
            _switch_t = time.monotonic()
            while True:
                up = self._is_up(index)
                if up is not False or time.monotonic() - _switch_t >= SWITCH_WAIT_S:
                    break
                time.sleep(0.15)
            if up is False and self.mode() == index:
                # The mode is right; it is the nodes that do not look as expected. Earbuds that show
                # them differently must not be taken for broken (and reconnected) at every switch:
                # from here on only the mode is checked, with the fixed wait as before.
                _nodes_trusted, up = False, None
                tracing.event("headset", to=to, nodes="not as expected", seen=self._nodes())
            # Up is not yet playing: give the sound server a moment to move the streams over, so
            # the next thing said does not start on the wrong speaker.
            time.sleep(0.3 if up else 0.0 if up is False else 0.5 if index == self.headset else 1.0)
            tracing.event("headset", to=to, ok=up, ms=round((time.monotonic() - t0) * 1000))
            return up

    def _volume(self) -> float | None:
        """How loud their output is set, on wpctl's scale."""
        try:
            sink = (self._nodes() or {}).get("Audio/Sink")
            out = subprocess.run(["wpctl", "get-volume", str(sink)], capture_output=True, text=True, timeout=3).stdout
            return float(re.search(r"Volume:\s*([0-9.]+)", out).group(1))
        except Exception:
            return None

    def _match_volume(self):
        """Headset mode keeps its own volume, and the sound server starts it far below the stereo
        one (0.40 against 0.53 on the slider), so the beep and the question came out much quieter
        than the guidance. Make it as loud as stereo, or what VOICE_HEADSET_VOLUME says."""
        want = os.environ.get("VOICE_HEADSET_VOLUME", "match")
        try:
            volume = _stereo_volume if want == "match" else None if want == "keep" else float(want)
            sink = (self._nodes() or {}).get("Audio/Sink")
            if volume is None or sink is None:
                return
            subprocess.run(["wpctl", "set-volume", str(sink), f"{min(1.0, volume):.3f}"], capture_output=True, timeout=3)
            tracing.event("headset", volume=round(volume, 3))
        except Exception as e:
            _problem("earbuds volume", e)

    def open(self):
        """Headset mode: the microphone comes up, the sound drops to mono. Stay here for the whole
        wait rather than flipping back and forth, which the earbuds do not take well."""
        if self._switch(self.headset) is False:
            raise MicLost("the earbuds did not go into headset mode")
        self._match_volume()
        self._ran_on = False

    def close(self):
        """Stereo again, for the guidance. If it does not come back, reconnect them."""
        if self._switch(self.stereo) is False and not self.reconnect():
            _problem("earbuds stereo", "the earbuds did not go back to stereo; the guidance is in mono until they do",
                     "The earbuds are stuck in headset mode: the guidance sounds mono. Take them out and put them back in.")
        else:
            _fine("earbuds stereo")

    def _address(self) -> str | None:
        """Their Bluetooth address, for bluetoothctl."""
        try:
            out = subprocess.run(["pw-cli", "info", str(self.device)], capture_output=True, text=True, timeout=3).stdout
            m = re.search(r'api\.bluez5\.address = "([0-9A-Fa-f:]{17})"', out) or re.search(r"bluez_card\.([0-9A-Fa-f_]{17})", out)
            if m:
                return m.group(1).replace("_", ":")
            out = subprocess.run(["bluetoothctl", "devices"], capture_output=True, text=True, timeout=5).stdout
            m = re.search(rf"Device ([0-9A-F:]{{17}}) {re.escape(self.name)}\s*$", out, re.M)
            return m.group(1) if m else None
        except Exception:
            return None

    def reconnect(self) -> bool:
        """Stereo did not come back by itself (on 10 October it stayed away for up to two and a
        half minutes). Disconnecting and connecting them again is what brings it back by hand."""
        global _repair_t, _switch_t
        if os.environ.get("PROPRIO_KEEPER") == "1":
            return self._wait_for_keeper()
        with _switch_lock:
            if time.monotonic() - _repair_t < 60:
                return False                              # tried a moment ago: do not keep cutting them off
            _repair_t, t0, address, found = time.monotonic(), time.monotonic(), self._address(), None
            try:
                if address is None:
                    raise RuntimeError("their Bluetooth address could not be read")
                subprocess.run(["bluetoothctl", "disconnect", address], capture_output=True, timeout=10)
                time.sleep(2.0)
                subprocess.run(["bluetoothctl", "connect", address], capture_output=True, timeout=15)
                end = time.monotonic() + 10
                while found is None and time.monotonic() < end:
                    time.sleep(0.5)
                    found = Headset.find()                # they come back under a new number
                if found is not None:
                    self.device, self.headset, self.stereo = found.device, found.headset, found.stereo
                    _switch_t = time.monotonic()
            except Exception as e:
                _problem("earbuds", f"could not reconnect them ({e})")
            ok = found is not None and self._switch(self.stereo) is not False
            tracing.event("headset", to="stereo", reconnected=True, ok=ok, ms=round((time.monotonic() - t0) * 1000))
            print(f"[voice] the earbuds did not go back to stereo; reconnected them ({'ok' if ok else 'still not right'})", file=sys.stderr)
            return ok

    def _wait_for_keeper(self) -> bool:
        """The launcher's keeper is watching the earbuds and reconnects them itself when a mode
        has been missing for ten seconds. Two of us cutting them off would undo each other's
        work, so here it only waits, for a bounded time, until stereo is back."""
        global _switch_t
        t0, ok, gone = time.monotonic(), False, False
        tracing.event("headset", to="stereo", waiting_on="keeper")
        print("[voice] the earbuds did not go back to stereo; waiting for the launcher's keeper to reconnect them", file=sys.stderr)
        while not ok and time.monotonic() - t0 < KEEPER_WAIT_S:
            time.sleep(0.5)
            found = Headset.find()
            if found is None:
                gone = True
                continue                                  # the keeper has just disconnected them
            if gone or found.device != self.device:       # ...and they are back, perhaps under a new number
                self.device, self.headset, self.stereo = found.device, found.headset, found.stereo
                _switch_t = time.monotonic()              # no switch of ours while they are still coming up
                gone = False
                ok = self._switch(self.stereo) is not False    # they can come back in headset mode: ask for stereo
                continue
            ok = self.mode() == self.stereo
        tracing.event("headset", to="stereo", keeper=True, ok=ok, ms=round((time.monotonic() - t0) * 1000))
        return ok

    def record(self, should_stop=None, abandon=None) -> np.ndarray | None:
        """Wait for someone to speak and return what they said (None if nobody did). Call
        between open() and close(). Raises MicLost when no sound arrives at all: the earbuds
        went away, or dropped out of headset mode."""
        import sounddevice as sd
        blocks: queue.Queue = queue.Queue()
        n = int(SR * 0.03)
        lost, got, peak, info = [0], [0], [0.0], {}

        def arrived(data, frames, t, status):
            lost[0] += bool(status)                       # the sound server dropped some on the way in
            blocks.put(data[:, 0].copy())
        with sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=n, callback=arrived):
            def incoming():
                while True:
                    try:
                        block = blocks.get(timeout=1.0)
                    except queue.Empty:
                        lost[0] += 1                      # a whole second without sound
                        return
                    got[0] += 1
                    if block.size:
                        peak[0] = max(peak[0], float(np.abs(block).max()))
                    yield block
            audio = cut_speech(incoming(), should_stop=should_stop, abandon=abandon, history=self.history, info=info)
        if lost[0]:
            tracing.event("mic", lost=lost[0], heard=audio is not None)
        if not got[0]:
            raise MicLost("no sound is arriving from the earbuds' microphone")
        # A real microphone is never exactly silent (the room measured 0.0003 through AirPods);
        # only zeros means the sound server is feeding us a source with nothing behind it.
        self._zeros_s = self._zeros_s + got[0] * 0.03 if peak[0] == 0.0 else 0.0
        if self._zeros_s >= 15.0:
            self._zeros_s = 0.0
            raise MicLost("the earbuds' microphone gives only silence")
        # Talk that filled the last recording and is loud again from the first moment of this one
        # is the same talk going on, not a request: its tail would otherwise pass for a short sentence.
        goes_on = self._ran_on and audio is not None and info.get("voice_at", 9.0) < 0.5 and info.get("skipped_s", 9.0) < 0.3
        self._ran_on = info.get("ended") == "max"
        if goes_on:
            tracing.event("voice.dropped", why="goes on", audio_s=round(audio.size / SR, 2), level=round(loudness(audio), 4))
            return None
        return audio


def decode_recording(data: bytes) -> np.ndarray:
    """A recording as the browser sent it (webm/opus, mp4/aac or wav) -> mono float32 at 16 kHz.
    Done with PyAV directly: faster-whisper's own decoder does not work with the PyAV installed
    on the GX10."""
    import av
    chunks = []
    with av.open(io.BytesIO(data), mode="r") as container:
        resampler = av.AudioResampler(format="s16", layout="mono", rate=SR)
        for frame in container.decode(audio=0):
            chunks += [f.to_ndarray().reshape(-1) for f in resampler.resample(frame)]
        chunks += [f.to_ndarray().reshape(-1) for f in resampler.resample(None)]      # flush
    if not chunks:
        return np.zeros(0, np.float32)
    return np.concatenate(chunks).astype(np.float32) / 32768.0


class _Wait:
    """One wait for a request: what the wearer has been told, and when to speak again. With
    nothing to pick up (or no picture) it keeps looking on a thread of its own, so the
    microphone stays open meanwhile, and the list is spoken as soon as there is one."""
    AGAIN = ("nothing", "covered", "no_camera", "no_cosmos")    # worth another look by itself

    def __init__(self):
        global _reminders
        _reminders = 0
        self.t0 = time.monotonic()
        self.remind = os.environ.get("VOICE_AUTO_LIST", "1") != "0" and os.environ.get("VOICE_REMIND", "1") != "0"
        self.seen: str | None = None      # what the last look the wearer heard about found
        self._found = None                # a look that has not been spoken yet: (seen, names)
        self._thread = None
        self._over = threading.Event()
        self._first = False               # the first look has been waited for
        self._looked = _looked

    def look(self, after_s: float = 0.0):
        """Start looking, and keep looking until there is something to pick up."""
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._watch, args=(after_s,), daemon=True, name="voice-look")
            self._thread.start()

    def _watch(self, after_s: float):
        if after_s:
            self._over.wait(after_s)
        while not self._over.is_set():
            try:
                found = look()
            except Exception as e:
                _problem("looking", e)
                found = ("no_cosmos", [])
            self._found = found
            if found[0] not in self.AGAIN:
                return
            self._over.wait(LOOK_EVERY_S if time.monotonic() - self.t0 < 60 else LOOK_SLOWER_S)

    def stop(self):
        self._over.set()

    def _remind_t(self) -> float:
        gaps = NOTHING_REMINDERS_S if self.seen in ("nothing", "covered", "no_camera") else SILENCE_REMINDERS_S
        return max(self.t0, _said_t) + gaps[min(_reminders, len(gaps) - 1)]

    def due(self) -> bool:
        """Is there something to say? Asked between sentences, never in the middle of one."""
        found = self._found
        return (found is not None and (found[0] == "ok" or self.seen is None)) or (self.remind and time.monotonic() >= self._remind_t())

    def speak(self):
        """Say whatever is due: the list once it is there, why there is none, or a reminder."""
        if self._thread is not None and not self._first:
            self._first = True
            _now("listing")
            end = time.monotonic() + LOOK_PATIENCE_S
            while self._found is None and time.monotonic() < end:
                time.sleep(0.05)
            if self._found is None:                       # a slow look must not leave them in silence; the list follows when it comes
                _say("What do you want to grab?")
        if _looked != self._looked:                       # they asked "what's on the table" meanwhile and were answered
            self._looked, self.seen, self._found = _looked, _seen, None
            if _seen in self.AGAIN:
                self.look(LOOK_EVERY_S)
        found, self._found = self._found, None
        if found is not None:
            seen, names = found
            if seen == "ok" and self.seen != "ok" or self.seen is None:
                _tell(seen, names, asked=False, keeps_looking=True)
                self.seen, self._looked = seen, _looked
            elif seen != "ok" and self.seen != "ok":
                self.seen = seen                          # the reason changed (an arm moved away): the next reminder says the new one
        if self.remind and time.monotonic() >= self._remind_t():
            first = _reminders == 0
            if self.seen in ("nothing", "covered", "no_camera"):
                _say({"nothing": "Still nothing to pick up. Turn towards the table or step back, or tell me what you want.",
                      "covered": "Something is still in front of the camera. Move your arm away or step back, or tell me what you want.",
                      "no_camera": "Still no picture from the camera. You can tell me what you want."}[self.seen]
                     if first else "Still looking.", reminder=True)
            else:
                _say("I'm listening. Say what you want to grab, or ask what is on the table." if first else "Still listening.",
                     reminder=True)


_left_open: list = []           # earbuds kept in headset mode for the whole run


class WebMic:
    """Lets any phone or laptop be the microphone and the push-to-talk button. It opens a page
    served from here, records while the button is held, and sends the recording to this machine,
    where Whisper transcribes it. Browsers only allow the microphone over HTTPS, so the page is
    served with a certificate made on the spot; the browser asks once whether to trust it."""

    def __init__(self, port: int = 8644, transcribe=None, model: str | None = None, cert_dir: str | Path = "out/voice_cert",
                 typed: bool = True, headset=None, hear=None):
        self.port = port
        self.status = {"heard": "", "target": ""}
        self._requests: queue.Queue = queue.Queue()
        self._waiting = False
        self.headset, self._hear = headset, hear          # earbuds as the microphone; audio -> text
        self._find_headset = transcribe is None           # keep looking: earbuds may connect after the start
        self._find_t, self._find_gap, self._check_t, self._sorry_t = 0.0, FIND_EARBUDS_S, 0.0, -1e9
        self.last_recording = {}                          # what the page last sent, for the page's own log
        self._reach = None                                # the thread listening during a reach, and what ends it
        self._device = None                               # the sound server's number for the earbuds last used
        if transcribe is None:
            try:
                with tracing.timed("load.whisper", model=model):
                    whisper = load_whisper(model)
            except Exception as e:
                # Without Whisper nothing spoken can be understood, but typing and the page's
                # text box still work, and saying so here beats stopping the whole run.
                whisper = None
                _problem("whisper", e, "Whisper could not be loaded: speech is not understood. Type the request instead.")

            def hear(audio: np.ndarray) -> str:
                if audio.size < SR // 4:
                    return ""
                t = time.perf_counter()
                text = whisper(audio)
                tracing.event("whisper", (time.perf_counter() - t) * 1000, audio_s=round(audio.size / SR, 2), text=text,
                              level=round(loudness(audio), 4), on=whisper.on)
                return text

            def transcribe(data: bytes) -> str:
                if whisper is None:
                    raise RuntimeError("Whisper is not loaded on the GX10; type the request instead")
                audio = decode_recording(data)
                self.last_recording = {"seconds": round(audio.size / SR, 2), "level": round(loudness(audio), 4)}
                return hear(audio)
            self._hear = hear if whisper is not None else None
            self._look_for_headset()
        self._transcribe = transcribe
        self._tls, self._server, self.url = None, None, None
        try:
            self._server = self._serve(Path(cert_dir))
            self.url = f"{'https' if self._tls else 'http'}://{_my_address()}:{self.port}/"
            print(f"[voice] Microphone page: open {self.url} on a phone or laptop on the same network, accept the "
                  "certificate warning once, then hold the button and speak. You can also type here.")
        except Exception as e:                            # the page is the backup microphone: the earbuds and the keyboard do not need it
            _problem("page", f"no microphone page ({e.__class__.__name__}: {e}); the earbuds and the keyboard still work")
        if typed and sys.stdin is not None and sys.stdin.isatty():
            threading.Thread(target=self._read_typed, daemon=True, name="voice-typed").start()

    def _read_typed(self):
        while True:
            try:
                line = input()
            except (EOFError, OSError):
                return
            if line.strip():
                self.submit(line.strip())

    def submit(self, text: str):
        """A request has arrived, spoken or typed."""
        self.status["heard"] = text
        self._requests.put(text)
        if not self._waiting:
            _poke_main()

    def _look_for_headset(self):
        if self.headset is None and self._find_headset and self._hear is not None:
            self.headset = Headset.find()
            if self.headset is not None:
                global _switch_t
                if self._device not in (None, self.headset.device):
                    _switch_t = time.monotonic()          # reconnected: let them finish coming up before a switch
                self._device = self.headset.device
                print(f"[voice] Using {self.headset.name} as the microphone: answer after the beep.")
            elif "earbuds" not in _trouble and _wants_earbuds():
                with _status_lock:                        # quietly: this is asked every few seconds
                    _trouble["earbuds"] = "No earbuds are connected: only the page and the keyboard can ask for something."
        return self.headset

    def _open(self, buds) -> bool:
        """Earbuds to headset mode. False (and no earbuds any more) if that did not work."""
        if buds is None:
            return False
        try:
            buds.open()
            self._check_t = time.monotonic() + CHECK_EARBUDS_S
            return True
        except Exception as e:
            self._lost(e)
            return False

    def _lost(self, e):
        """The earbuds stopped working as the microphone. Carry on with the page and the keyboard,
        and look for them again: soon at first, less often while it keeps failing, never in a spin."""
        _problem("earbuds", e, "The earbuds' microphone is not working; looking for it again. The page and the keyboard still work.")
        self.headset = None
        self._find_t = time.monotonic() + self._find_gap
        self._find_gap = min(FIND_EARBUDS_MAX_S, self._find_gap * 2)

    def _check(self):
        """While listening: are the earbuds still there, and still in headset mode? If they
        reconnected, or the sound server restarted, they come back in stereo with a new number."""
        if not self._find_headset or time.monotonic() < self._check_t:
            return
        self._check_t = time.monotonic() + CHECK_EARBUDS_S
        found = Headset.find()
        if found is None:
            raise MicLost("the earbuds are gone")
        if found.device != self.headset.device:
            tracing.event("voice.earbuds", change="new device", device=found.device)
            self.headset = found
            found.open()
        elif found.mode() not in (None, found.headset):
            tracing.event("voice.earbuds", change="back to headset mode")
            found.open()

    def _answer(self, text: str) -> str:
        try:
            return respond(text)
        except Exception as e:
            _problem("answering", e)
            return ""
        finally:
            self.status.update(option_families())         # the page shows which options are on

    def _hear_commands(self, buds, over: threading.Event):
        """During a reach, with the earbuds kept in headset mode: listen for the few words that
        make sense then, and hand them over as if they had been typed. Only a word or two, only as
        loud as the wearer, and only if it is exactly a command: the hall says "stop" too."""
        while not over.is_set():
            try:
                audio = buds.record(should_stop=over.is_set, abandon=over.is_set)
                if audio is None or audio.size / SR > REACH_WORDS_S:
                    continue
                if loudness(audio) < float(os.environ.get("VOICE_WEARER_LEVEL", WEARER_LEVEL)):
                    continue
                text = self._hear(audio)
            except Exception as e:
                _problem("earbuds", f"stopped listening during the reach ({e.__class__.__name__}: {e})")
                return
            clean = _plain(text)
            if over.is_set() or not (_STOP.match(clean) or _AGAIN.match(clean) or _HELP.match(clean)
                                     or (len(clean.split()) <= 5 and switch_option_asked(clean))):
                continue
            tracing.event("voice.reach_command", text=text, level=round(loudness(audio), 4))
            self.submit(text)
            return                                        # main.py now asks; the next wait takes it from here

    def listen(self) -> str:
        global _ask_next
        if self._reach is not None:                       # the reach is over: one listener at a time
            self._reach[1].set()
            self._reach[0].join(3)
            self._reach = None
        wait = _Wait()
        text, target = "", ""
        buds = None
        try:
            self._waiting = True
            if _ask_next and self._requests.empty() and os.environ.get("VOICE_AUTO_LIST", "1") != "0":
                _ask_next = False
                wait.look()               # on its own thread, while the earbuds change mode
            buds = self._look_for_headset()
            if buds is not None and self._find_headset and buds.mode() is None:
                self.headset = None       # they reconnected during the reach: the number held here is nobody's now
                buds = self._look_for_headset()
            self._open(buds)              # before the question: an answer given at once must not fall in the switch
            self.status.update(option_families())
            beeped = -1
            while not target:
                wait.speak()
                # The beep means "your turn", so it comes once at the start and after anything
                # said to the wearer, not after every stray sentence from the room.
                heard = self._next_request(beep=beeped != _said, due=wait.due)
                beeped = _said
                if heard is not None:
                    text = heard
                    target = self._answer(text)   # stay here, in headset mode, until there is something to reach for
        finally:
            self._waiting = False
            wait.stop()
            for b in {buds, self.headset} - {None}:       # the one opened here, or one that connected during the wait
                try:
                    if not _stays():
                        b.close()
                    elif b not in _left_open:
                        _left_open.append(b)
                        atexit.register(b.close)          # stereo again when the run ends, for whoever uses them next
                except Exception as e:
                    _problem("earbuds", e)
        self.status["target"] = target or self.status["target"]
        self.status.update(option_families())
        if target and self.headset is not None and _stays() and os.environ.get("VOICE_REACH_LISTEN", "0") == "1":
            over = threading.Event()
            thread = threading.Thread(target=self._hear_commands, args=(self.headset, over), daemon=True, name="voice-reach")
            self._reach = (thread, over)
            thread.start()
        _now("reaching", message=f"Guiding to {target}.")
        print(f"[voice] heard {text!r} -> target {target!r}")
        return target

    def _next_request(self, beep: bool = True, due=lambda: False) -> str | None:
        """Whatever comes first: something typed or sent from the page, or, with earbuds as the
        microphone, something the wearer said aloud. None when `due()` says there is something
        to tell the wearer instead."""
        while True:
            if due():
                return None
            buds = self.headset
            _now("listening" if buds is not None else "waiting", mic=buds.name if buds is not None else "page or keyboard")
            try:
                text = self._requests.get(timeout=0.05 if buds is not None else 0.25)
                while True:                               # several typed in a row: the newest is the one meant,
                    try:                                  # and none may be left to be taken for the next request
                        text = self._requests.get_nowait()
                    except queue.Empty:
                        return text
            except queue.Empty:
                pass
            if buds is None:
                if self._find_headset and self._hear is not None and time.monotonic() >= self._find_t:
                    self._find_t = time.monotonic() + self._find_gap
                    if self._look_for_headset() is not None and self._open(self.headset):
                        beep = True                       # they are back: the beep says the microphone is open again
                continue
            try:
                if beep:
                    _beep()                               # "your turn": the microphone is open from here
                    beep = False
                self._check()
                buds = self.headset
                audio = buds.record(should_stop=lambda: not self._requests.empty() or due(),
                                    abandon=lambda: not self._requests.empty())
            except Exception as e:
                self._lost(e)
                continue
            if self._find_gap != FIND_EARBUDS_S or "earbuds" in _trouble:
                self._find_gap = FIND_EARBUDS_S
                _fine("earbuds")
            if audio is None:
                continue
            text = self._heard(audio)
            if text:
                self.status["heard"] = text
                return text

    def _heard(self, audio: np.ndarray) -> str:
        """What the wearer said in this recording, or "" if it was not the wearer or not a request."""
        seconds, level = audio.size / SR, loudness(audio)
        if seconds > float(os.environ.get("VOICE_REQUEST_S", REQUEST_MAX_S)):
            tracing.event("voice.dropped", why="long", audio_s=round(seconds, 2), level=round(level, 4))
            _now(meant=f"not a request: {seconds:.0f} s of talk")
            return ""
        _now("thinking")
        try:
            text = self._hear(audio)
        except Exception as e:
            _problem("whisper", e, "Whisper failed on the last thing said.")
            if level >= WEARER_LEVEL and time.monotonic() - self._sorry_t > 30:       # tell the wearer, but not over and over
                self._sorry_t = time.monotonic()
                _say("I did not catch that. Say it again.")
            return ""
        _fine("whisper")
        if not text or not worth_hearing(text):
            return ""
        if not from_the_wearer(text, level):
            tracing.event("voice.dropped", why="soft", audio_s=round(seconds, 2), level=round(level, 4), text=text[:80])
            _now(heard=text, meant="not the wearer: too quiet")
            return ""
        return text

    def _serve(self, cert_dir: Path):
        mic = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def setup(self):
                # The TLS handshake is done here, on this connection's own thread and with a time
                # limit. Done while accepting, one phone that connects and then says nothing (a
                # browser's spare connection, a screen that locked) holds up every other request.
                self.request.settimeout(15)
                if mic._tls is not None:
                    if self.request.recv(1, socket.MSG_PEEK) not in (b"\x16", b""):      # not TLS: somebody typed http://
                        host = self.request.getsockname()[0]
                        self.request.sendall(f"HTTP/1.0 301 Moved\r\nLocation: https://{host}:{mic.port}/\r\n"
                                             "Content-Length: 0\r\n\r\n".encode())
                        raise ConnectionError("plain http")
                    self.request = mic._tls.wrap_socket(self.request, server_side=True)
                super().setup()

            def finish(self):
                super().finish()
                try:
                    self.request.close()
                except OSError:
                    pass

            def _send(self, code: int, body: bytes, kind: str):
                self.send_response(code)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path = self.path.split("?")[0]
                if path == "/":
                    self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
                elif path == "/status":
                    self._send(200, json.dumps(mic.status).encode(), "application/json")
                else:
                    self._send(404, b"not found", "text/plain")

            def do_POST(self):
                data = self.rfile.read(min(int(self.headers.get("Content-Length") or 0), 20_000_000))
                path = self.path.split("?")[0]
                reply = {}
                if path == "/say":
                    mic.last_recording = {}
                    try:
                        text = mic._transcribe(data)
                    except Exception as e:
                        _problem("page", f"could not read a recording of {len(data)} bytes, {self.headers.get('Content-Type')} "
                                         f"({e.__class__.__name__}: {e})")
                        text, reply = "", {"error": f"The GX10 could not read that recording ({e.__class__.__name__}: {e})."}
                    reply = dict(reply, bytes=len(data), **mic.last_recording)
                elif path == "/text":
                    try:
                        text = json.loads(data or b"{}").get("text", "")
                    except (ValueError, AttributeError):
                        text = ""
                    # Only words count: a number or a list in that place is somebody's mistake, and
                    # taking it for a request would stop the reach in progress.
                    text = text.strip()[:200] if isinstance(text, str) else ""
                elif path == "/log":                      # the page telling us what went wrong on its side
                    print(f"[voice] page: {data[:300].decode('utf-8', 'replace')}", file=sys.stderr)
                    tracing.event("voice.page", text=data[:300].decode("utf-8", "replace"))
                    self._send(200, b"{}", "application/json")
                    return
                else:
                    self._send(404, b"not found", "text/plain")
                    return
                if text:
                    mic.submit(text)
                self._send(200, json.dumps(dict(reply, heard=text)).encode(), "application/json")

        class Server(ThreadingHTTPServer):
            daemon_threads = True

            def handle_error(self, request, client_address):
                pass                                      # a refused certificate or a dropped phone is not news

        try:
            self._tls = _tls_context(cert_dir, _my_address())
        except Exception as e:
            # No certificate: the page is still served, plainly. Browsers will not give it the
            # microphone, but its text box and option buttons work, and it says why.
            _problem("page", f"no certificate ({e.__class__.__name__}: {e}); the page is served over http, for typing only")
        server = None
        for port in ([self.port] if self.port == 0 else range(self.port, self.port + 5)):
            try:
                server = Server(("0.0.0.0", port), Handler)
                break
            except OSError as e:                          # taken, most likely by a run that has not quite gone yet
                taken = e
        if server is None:
            raise taken
        if server.server_address[1] != self.port and self.port:
            print(f"[voice] port {self.port} is taken; the microphone page is on {server.server_address[1]}", file=sys.stderr)
        self.port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True, name="voice-web").start()
        return server

    def close(self):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()


def _tls_context(cert_dir: Path, address: str) -> ssl.SSLContext:
    """A certificate for the microphone page, made once and kept. It names this machine's
    address and says it is for a server: iPhones want both before they will trust one."""
    cert_dir.mkdir(parents=True, exist_ok=True)
    crt, key, made_for = cert_dir / "cert.pem", cert_dir / "key.pem", cert_dir / "address.txt"
    fresh = (crt.exists() and key.exists() and made_for.exists() and made_for.read_text() == address
             and time.time() - crt.stat().st_mtime < 20 * 86400)
    if not fresh:
        base = ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", str(key), "-out", str(crt),
                "-days", "30", "-subj", "/CN=proprio"]
        numeric = re.fullmatch(r"[0-9.]+", address) is not None
        names = "subjectAltName=DNS:localhost,IP:127.0.0.1," + ("IP:" if numeric else "DNS:") + address
        try:
            subprocess.run(base + ["-addext", names, "-addext", "extendedKeyUsage=serverAuth"], check=True,
                           capture_output=True, timeout=30)
        except subprocess.CalledProcessError:             # an openssl too old for -addext
            subprocess.run(base, check=True, capture_output=True, timeout=30)
        made_for.write_text(address)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(str(crt), str(key))
    return ctx


def option_families() -> dict:
    """What can be switched and what is on, for the page's buttons."""
    out = {}
    for kind in ("", "hand", "grab", "object"):
        try:
            name, options, choose, current = _family(kind)
            if options and choose is not None:
                number, _, title = current()
                out[kind or "guide"] = {"now": number, "titles": [o[1] if isinstance(o, (tuple, list)) else str(o) for o in options]}
                if not kind:
                    out["option"] = f"{number}: {title}" if number else title      # 0: a way of guiding from before the options
        except Exception:
            pass
    return {"option": out.pop("option", ""), "options": out}


def has_microphone() -> bool:
    """Is a real microphone plugged in or paired? The sound server always offers a default
    input, even with nothing to record from, so that alone proves nothing."""
    import shutil
    if shutil.which("pactl"):
        try:
            rows = subprocess.run(["pactl", "list", "short", "sources"], capture_output=True, text=True, timeout=3).stdout
            return any(r.split("\t")[1:2] and not r.split("\t")[1].endswith(".monitor") for r in rows.splitlines() if r.strip())
        except Exception:
            pass
    if shutil.which("wpctl"):            # PipeWire without the pulse tools: read its "Sources:" list
        try:
            out = subprocess.run(["wpctl", "status"], capture_output=True, text=True, timeout=3).stdout
            audio = out.split("Audio", 1)[1].split("Video", 1)[0]
            sources = audio.split("Sources:", 1)[1].split("Source endpoints:", 1)[0]
            return bool(re.search(r"\d+\.\s+\S", sources))
        except Exception:
            pass
    try:
        import sounddevice as sd
        return any(d["max_input_channels"] > 0 and d["name"] not in ("default", "pipewire", "pulse", "sysdefault")
                   for d in sd.query_devices())
    except Exception:
        return False


_listener = None                # Listener (local microphone) or WebMic (a phone or laptop as the microphone)
_mode: str | None = None
_failures = 0                   # times in a row the wait for a request has failed


def _start():
    """Choose how to listen, once. Whatever is missing, something is left that works: the wired
    microphone, then earbuds and the page, then the keyboard."""
    global _listener, _mode, _status_thread
    if _status_thread is None:
        _status_thread = threading.Thread(target=_keep_status, daemon=True, name="voice-status")
        _status_thread.start()
    want = os.environ.get("VOICE_MODE", "auto")
    model = os.environ.get("VOICE_MODEL") or None
    _mode = "type"
    why = "none is plugged in or paired"
    buds = want == "auto" and not os.environ.get("VOICE_DEVICE") and Headset.find() is not None
    # Earbuds are handled by WebMic (it listens by itself); the Enter-to-talk Listener is for a wired microphone.
    if want == "speech" or (want == "auto" and not buds and (os.environ.get("VOICE_DEVICE") or has_microphone())):
        try:
            _listener = Listener(model, os.environ.get("VOICE_DEVICE") or None)
            _mode = "speech"
        except Exception as e:
            why = e
            if want == "speech":
                _problem("microphone", f"VOICE_MODE=speech, but the microphone on this machine cannot be used ({e}); trying the page")
    if _mode == "type" and want in ("auto", "web", "speech"):
        try:
            _listener = WebMic(int(os.environ.get("VOICE_PORT", "8644")), model=model)
            _mode = "web"
        except Exception as e:
            _problem("microphone", f"no microphone here ({why}) and no microphone page ({e}); type the target instead")


def listen_for_target() -> str:
    """Blocks until the wearer has named something. May return "" (main.py asks again). It never
    raises: whatever goes wrong while listening is logged in one line, and the next call
    listens again, after a pause so that a fault which repeats cannot become a busy loop."""
    global _failures
    try:
        if _mode is None:
            _start()
        target = _listener.listen() if _mode in ("speech", "web") else typed()
        _failures = 0
        if target:
            _now("reaching", message=f"Guiding to {target}.")
        return target
    except Exception as e:
        _failures += 1
        _problem("listening", e)
        time.sleep(min(5.0, 0.5 * _failures))
        return ""


def ready() -> list[tuple[str, bool, str]]:
    """Is everything voice needs on this machine and answering? (what, ok, detail) for each part,
    without loading a model: for a checklist before the run, and `python voice.py --check`."""
    import shutil
    out = []

    def check(what: str, test):
        try:
            detail = test()
            out.append((what, True, detail or "ok"))
        except Exception as e:
            out.append((what, False, " ".join(str(e).split())[:160] or e.__class__.__name__))

    def whisper():
        from huggingface_hub import snapshot_download
        model = os.environ.get("VOICE_MODEL") or "large-v3"
        folder = Path(snapshot_download(model if "/" in model else f"openai/whisper-{model}", local_files_only=True))
        if not any(folder.glob("*.safetensors")) and not any(folder.glob("*.bin")):
            raise RuntimeError(f"{folder} has no weights in it")
        return f"{model} is on this machine"

    def vad():
        from faster_whisper.utils import get_assets_path
        if not any(Path(get_assets_path()).glob("silero_vad*.onnx")):
            raise RuntimeError("faster-whisper's speech detector file is missing")

    def piper():
        import audio_out
        if not audio_out.PIPER_VOICE.exists():
            raise RuntimeError(f"{audio_out.PIPER_VOICE} is missing: bash tools/fetch_models.sh piper")

    def nemotron():
        url = os.environ.get("VOICE_LLM_URL", "http://127.0.0.1:8000/v1")
        if url == "off":
            return "switched off; the simple rule reads requests"
        with urllib.request.urlopen(url.rsplit("/v1", 1)[0] + "/health", timeout=1.0):
            pass

    def cosmos_up():
        import cosmos
        if not cosmos.available():
            raise RuntimeError("not answering: bash tools/run_cosmos.sh")

    def earbuds():
        buds = Headset.find()
        if buds is None:
            raise RuntimeError("no Bluetooth earbuds connected")
        return buds.name

    def openssl():
        if not shutil.which("openssl"):
            raise RuntimeError("openssl is missing: no certificate for the microphone page")

    for what, test in (("Whisper", whisper), ("speech detector", vad), ("Piper voice", piper), ("Nemotron", nemotron),
                       ("Cosmos", cosmos_up), ("earbuds", earbuds), ("openssl", openssl)):
        check(what, test)
    return out


if __name__ == "__main__":
    if "--check" in sys.argv:
        results = ready()
        for what, ok, detail in results:
            print(f"{'ok ' if ok else 'NO '} {what}: {detail}")
        sys.exit(0 if all(ok for _, ok, _ in results) else 1)
    print(__doc__)
