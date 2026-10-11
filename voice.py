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

"Option 3" (or "next option") changes the way guidance sounds without stopping; see guidance.OPTIONS.

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

Bluetooth earbuds (AirPods) are used as the microphone by themselves when they are connected. They
cannot play stereo and record at once, so they are switched to headset mode only while it is
listening, which is whenever it is waiting for a request, and back to stereo for the guidance.
No button: it has just asked "What do you want to grab?", so you simply answer.

Speech: press Enter, say "find my mug", press Enter again. A Bluetooth clicker that sends
Enter works as the button.
"""
from __future__ import annotations

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
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

import tracing

SR = 16000          # what Whisper expects
MAX_S = 8.0         # stop recording after this even without the second Enter

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
_OPTION = re.compile(r"\b(?:option|mode|version|style|scheme)\s+(?:number\s+)?(one|two|three|four|five|[1-5])\b"
                     r"|^(next) (?:option|one|mode|version|style)$")
_ORDINAL = re.compile(r"^(?:the |number )?(first|second|third|fourth|fifth|last|one|two|three|four|five|[1-5])(?: one)?$")
_ORDER = {"first": 0, "one": 0, "1": 0, "second": 1, "two": 1, "2": 1, "third": 2, "three": 2, "3": 2,
          "fourth": 3, "four": 3, "4": 3, "fifth": 4, "five": 4, "5": 4, "last": -1}
last_transcript: str | None = None      # what was last said or typed, for the dashboard
last_objects: list[str] = []            # what Cosmos last listed, so "the second one" means something
_llm_ok: bool | None = None             # None until the first request shows whether Nemotron is there


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
    return intent, (extract_target(str(d.get("target", ""))) if intent == "pick" else "")


def interpret(text: str) -> tuple[str, str]:
    """Words -> (intent, target). intent is "pick" (target names the object), "list" (they asked
    what is there) or "none". Short requests go by rule; longer or indirect ones ("I'm thirsty")
    go to Nemotron if it is running, and the rule steps in when it is not."""
    global _llm_ok, last_transcript
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
    url = os.environ.get("VOICE_LLM_URL", "http://127.0.0.1:8000/v1")
    if url == "off" or _llm_ok is False or len(clean.split()) < 2:     # one word is already a name
        return by_rule
    t = time.perf_counter()
    try:
        intent, target = ask_llm(text, url)
    except Exception as e:
        if _llm_ok is None:
            print(f"[voice] Nemotron is not answering at {url} ({e.__class__.__name__}); using the simple rule. "
                  "Start it with: bash tools/run_nemotron.sh", file=sys.stderr)
        _llm_ok = False
        return by_rule
    _llm_ok = True
    tracing.event("nemotron", (time.perf_counter() - t) * 1000, text=text, intent=intent, target=target)
    if by_rule[0] == "list" and intent != "pick":            # the rule is sure about this one
        return "list", ""
    return intent, target


def understand(text: str) -> str:
    """Words -> object name, or "" if they did not ask for one."""
    intent, target = interpret(text)
    return target if intent == "pick" else ""


def _say(text: str):
    print(f"[voice] {text}")
    try:
        import guidance
        guidance.say_text(text, wait=True)       # finish speaking before the microphone opens again
    except Exception as e:
        print(f"[voice] could not speak ({e})", file=sys.stderr)


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
    if module._default is None:
        # The first question comes before main.py has asked for a single frame, so the stream
        # has not been opened yet. Open it now; main.py will carry on with the same one.
        try:
            module.start()
        except Exception as e:
            print(f"[voice] no camera: {e}", file=sys.stderr)
            return None
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


def describe(asked: bool = True) -> list[str]:
    """Look through the camera with Cosmos and tell the wearer what could be picked up.
    asked=False is the unprompted version at the start and after a grab: it ends with
    "What do you want to grab?" and stays quiet about why if it cannot look."""
    global last_objects
    import cosmos
    frame = _camera_frame()
    if frame is None:
        if asked:
            _say("I need the camera for that.")
        return []
    if not cosmos.available():
        if asked:
            print("[voice] Cosmos is not running. Start it with: bash tools/run_cosmos.sh", file=sys.stderr)
            _say("I can't look around right now.")
        else:
            _say("What do you want to grab?")
        return []
    try:
        objects = cosmos.list_objects(frame.rgb)
    except Exception as e:
        print(f"[voice] Cosmos failed ({e})", file=sys.stderr)
        _say("I can't look around right now.")
        return []
    last_objects = objects
    if not objects:
        _say("I don't see anything to pick up." if asked else "I don't see anything to pick up. What do you want?")
    elif len(objects) == 1:
        _say(f"I can see one thing: {objects[0]}. Do you want it?")
    else:
        _say("I can see " + ", ".join(objects[:-1]) + " and " + objects[-1] + ". "
             + ("Which one?" if asked else "What do you want to grab?"))
    return objects


_ask_next = True        # speak up before the next wait: at the start, and after each target (so after each grab)


def invite(pending: bool = False):
    """Called before waiting for a request. If the wearer has just finished (or we have just
    started) and has not already said something, look around and ask what they want."""
    global _ask_next
    if _ask_next and not pending and os.environ.get("VOICE_AUTO_LIST", "1") != "0":
        _ask_next = False
        describe(asked=False)


def switch_option(text: str) -> str | None:
    """ "option 3" or "next option": change how guidance sounds. Returns what to look for next
    (the same target, so a reach in progress carries on), or None if this was not such a request."""
    m = _OPTION.search(re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip())
    if not m:
        return None
    import guidance
    if m.group(2):
        number = guidance.option()[0] % len(guidance.OPTIONS) + 1
    else:
        number = _ORDER[m.group(1)] + 1
    title = guidance.set_option(number)
    _say(f"Option {number}: {title}.")
    return guidance.current_target() or ""


def respond(text: str) -> str:
    """What to do with what was said: the target to look for, or "" after answering a question
    (main.py then asks again, and the wearer names their pick)."""
    global last_transcript, _ask_next
    switched = switch_option(text)
    if switched is not None:
        last_transcript = text
        return switched
    if last_objects and _YES.match(re.sub(r"[^a-z' ]+", " ", text.lower()).strip()):
        last_transcript = text
        _ask_next = True
        return last_objects[0]                               # "yes" to "Do you want it?"
    intent, target = interpret(text)
    if intent == "list":
        describe()
        return ""
    if intent == "pick" and target:
        _ask_next = True                                     # after this reach, speak up again
        return target
    return ""


def typed() -> str:
    invite()
    try:
        return respond(input("what do you want to pick up?> "))
    except EOFError:
        return ""


def load_whisper(model: str = "small.en"):
    """Whisper as a function: audio (mono float32 at 16 kHz) -> text. On the GPU through PyTorch
    when there is one (about 0.05 s a request on the GX10). Otherwise faster-whisper on the CPU
    (about 1.2 s): its build for the GX10's processor has no GPU support. VOICE_STT=cpu or gpu
    forces one. Either way the stretches without speech are cut out first, because Whisper
    writes sentences of its own when handed silence."""
    want = os.environ.get("VOICE_STT", "auto")
    if want != "cpu":
        try:
            return _gpu_whisper(model)
        except Exception as e:
            if want == "gpu":
                raise
            print(f"[voice] Whisper on the CPU ({e.__class__.__name__}: {e})", file=sys.stderr)
    from faster_whisper import WhisperModel
    whisper = WhisperModel(model, device="cpu", compute_type="int8")

    def hear(audio: np.ndarray) -> str:
        segments, _ = whisper.transcribe(audio, language="en", beam_size=1, vad_filter=True, hotwords=expected_words())
        return " ".join(s.text for s in segments).strip()
    hear.on = "cpu"
    return hear


def _gpu_whisper(model: str):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("no GPU here")
    from faster_whisper.vad import get_speech_timestamps
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    name = model if "/" in model else f"openai/whisper-{model}"
    proc = WhisperProcessor.from_pretrained(name)
    net = WhisperForConditionalGeneration.from_pretrained(name, torch_dtype=torch.float16).to("cuda").eval()
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
        return write(np.concatenate([audio[c["start"]:c["end"]] for c in speech]), expected_words())
    write(np.zeros(SR, np.float32), None)                 # the first request is slow; get it over with now
    hear.on = "gpu"
    return hear


class Listener:
    """Records between two presses of Enter and transcribes with Whisper."""

    def __init__(self, model: str = "small.en", device=None):
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
        audio = self.record()
        t = time.perf_counter()
        text = self.transcribe(audio)
        tracing.event("whisper", (time.perf_counter() - t) * 1000, audio_s=round(audio.size / SR, 2), text=text)
        target = respond(text)
        print(f"[voice] heard {text!r} -> target {target!r}")
        return target


PAGE = Path(__file__).resolve().parent / "voice_page.html"


def _poke_main():
    """A request arrived while a reach was in progress: press main.py's "new target" key through
    the dashboard, so the wearer can change their mind without waiting for a grab."""
    port = os.environ.get("DASHBOARD_PORT", "8642")
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
_NOT_SPEECH = {"", "you", "thank you", "thanks", "thanks for watching", "bye", "okay", "ok", "so", "the", "uh", "um", "hmm"}


def expected_words() -> str | None:
    """The names Whisper should expect: what was just listed. Without them the small models
    hear "blue cup" as "look up"."""
    return ", ".join(last_objects) if last_objects else None


def worth_hearing(text: str) -> bool:
    return re.sub(r"[^a-z ]+", "", text.lower()).strip() not in _NOT_SPEECH


def cut_speech(blocks, sr: int = SR, block_s: float = 0.03, start_wait_s: float = 6.0, max_s: float = 8.0,
               quiet_s: float = 0.9, should_stop=None, min_level: float = 0.005) -> np.ndarray | None:
    """Take audio blocks as they arrive and return the stretch that holds speech: from a little
    before the voice starts until it has been quiet for `quiet_s`. None if nobody spoke within
    `start_wait_s`, or `should_stop()` said to give up. The level that counts as a voice is set
    from the first third of a second, so it adapts to a noisy hall, and is never below
    `min_level`: earbuds silence the room, and a soft voice measured 0.006 to 0.010 through them."""
    kept, lead, floor, loud, quiet, started, t = [], [], [], 0, 0.0, False, 0.0
    for block in blocks:
        t += block_s
        level = float(np.sqrt(np.mean(np.square(block)))) if len(block) else 0.0
        if t <= 0.3:
            floor.append(level)
            lead.append(block)
            continue
        threshold = max(min_level, 2.5 * (float(np.median(floor)) if floor else 0.0))
        if not started:
            lead = (lead + [block])[-10:]                 # keep 0.3 s from before the voice
            loud = loud + 1 if level > threshold else 0
            if loud >= 4:                                 # 120 ms of voice, not a click
                started, kept = True, list(lead)
            elif t > start_wait_s or (should_stop is not None and should_stop()):
                return None
            continue
        kept.append(block)
        quiet = quiet + block_s if level <= threshold else 0.0
        if quiet >= quiet_s or t > start_wait_s + max_s:
            break
    return np.concatenate(kept) if started and kept else None


class Headset:
    """Bluetooth earbuds as the microphone, through PipeWire. In stereo mode they have no
    microphone at all; in headset mode they have one but the sound is mono and coarse. So the
    mode is switched just for the moments it listens."""

    def __init__(self, device: int, headset: int, stereo: int, name: str = "earbuds"):
        self.device, self.headset, self.stereo, self.name = device, headset, stereo, name

    @staticmethod
    def find():
        """The connected earbuds, or None. Reads PipeWire's device list and their profiles."""
        import shutil
        if os.environ.get("VOICE_HEADSET", "auto") == "off" or not (shutil.which("wpctl") and shutil.which("pw-cli")):
            return None
        try:
            status = subprocess.run(["wpctl", "status"], capture_output=True, text=True, timeout=3).stdout
            devices = status.split("Audio", 1)[1].split("Sinks:", 1)[0]
            m = re.search(r"(\d+)\.\s+(.+?)\s+\[bluez5\]", devices)
            if not m:
                return None
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

    def open(self):
        """Headset mode: the microphone comes up, the sound drops to mono. Stay here for the whole
        wait rather than flipping back and forth, which the earbuds do not take well."""
        self._profile(self.headset)
        time.sleep(0.5)

    def close(self):
        """Stereo again, for the guidance. Give the earbuds a moment to come back as the output,
        so the next thing said does not start on the wrong speaker."""
        self._profile(self.stereo)
        time.sleep(1.0)

    def record(self, should_stop=None) -> np.ndarray | None:
        """Wait for someone to speak and return what they said (None if nobody did). Call
        between open() and close()."""
        import sounddevice as sd
        blocks: queue.Queue = queue.Queue()
        n = int(SR * 0.03)
        lost = [0]

        def arrived(data, frames, t, status):
            lost[0] += bool(status)                       # the sound server dropped some on the way in
            blocks.put(data[:, 0].copy())
        with sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=n, callback=arrived):
            def incoming():
                while True:
                    try:
                        yield blocks.get(timeout=1.0)
                    except queue.Empty:
                        lost[0] += 1                      # a whole second without sound
                        return
            audio = cut_speech(incoming(), should_stop=should_stop)
        if lost[0]:
            tracing.event("mic", lost=lost[0], heard=audio is not None)
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


class WebMic:
    """Lets any phone or laptop be the microphone and the push-to-talk button. It opens a page
    served from here, records while the button is held, and sends the recording to this machine,
    where Whisper transcribes it. Browsers only allow the microphone over HTTPS, so the page is
    served with a certificate made on the spot; the browser asks once whether to trust it."""

    def __init__(self, port: int = 8644, transcribe=None, model: str = "small.en", cert_dir: str | Path = "out/voice_cert",
                 typed: bool = True, headset=None, hear=None):
        self.port = port
        self.status = {"heard": "", "target": ""}
        self._requests: queue.Queue = queue.Queue()
        self._waiting = False
        self.headset, self._hear = headset, hear          # earbuds as the microphone; audio -> text
        self._find_headset = transcribe is None           # keep looking: earbuds may connect after the start
        if transcribe is None:
            with tracing.timed("load.whisper", model=model):
                whisper = load_whisper(model)

            def hear(audio: np.ndarray) -> str:
                if audio.size < SR // 4:
                    return ""
                t = time.perf_counter()
                text = whisper(audio)
                tracing.event("whisper", (time.perf_counter() - t) * 1000, audio_s=round(audio.size / SR, 2), text=text,
                              level=round(float(np.sqrt(np.mean(np.square(audio)))), 4), on=whisper.on)
                return text

            def transcribe(data: bytes) -> str:
                return hear(decode_recording(data))
            self._hear = hear
            self._look_for_headset()
        self._transcribe = transcribe
        self._server = self._serve(Path(cert_dir))
        self.url = f"https://{_my_address()}:{self.port}/"
        print(f"[voice] Microphone page: open {self.url} on a phone or laptop on the same network, accept the "
              "certificate warning once, then hold the button and speak. You can also type here.")
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
        if self.headset is None and self._find_headset:
            self.headset = Headset.find()
            if self.headset is not None:
                print(f"[voice] Using {self.headset.name} as the microphone: answer after the beep.")
        return self.headset

    def listen(self) -> str:
        buds = self._look_for_headset()
        try:
            if buds is not None:
                buds.open()               # before the question: an answer given at once must not fall in the switch
            invite(pending=not self._requests.empty())
            self._waiting = True
            text = self._next_request()
        finally:
            self._waiting = False
            for b in {buds, self.headset} - {None}:       # the one opened here, or one that connected during the wait
                try:
                    b.close()
                except Exception:
                    pass
        target = respond(text)
        self.status["target"] = target or self.status["target"]
        try:
            import guidance
            self.status["option"] = "%d: %s" % guidance.option()[::2]
        except Exception:
            pass
        print(f"[voice] heard {text!r} -> target {target!r}")
        return target

    def _next_request(self) -> str:
        """Whatever comes first: something typed or sent from the page, or, with earbuds as the
        microphone, something said aloud."""
        while self.headset is None:
            try:
                return self._requests.get(timeout=3.0)
            except queue.Empty:
                if self._look_for_headset() is not None:
                    self.headset.open()
        buds = self.headset
        try:
            _beep()                                       # "your turn": the microphone is open from here
            while True:
                try:
                    return self._requests.get(timeout=0.05)
                except queue.Empty:
                    pass
                audio = buds.record(should_stop=lambda: not self._requests.empty())
                text = self._hear(audio) if audio is not None else ""
                if text and worth_hearing(text):
                    self.status["heard"] = text
                    return text
        except Exception as e:
            print(f"[voice] the earbuds' microphone failed ({e}); use the page or type", file=sys.stderr)
            self.headset = None
            return self._requests.get()

    def _serve(self, cert_dir: Path):
        mic = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
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
                if path == "/say":
                    try:
                        text = mic._transcribe(data)
                    except Exception as e:
                        print(f"[voice] could not read that recording ({e})", file=sys.stderr)
                        text = ""
                elif path == "/text":
                    try:
                        text = str(json.loads(data or b"{}").get("text", "")).strip()[:200]
                    except ValueError:
                        text = ""
                else:
                    self._send(404, b"not found", "text/plain")
                    return
                if text:
                    mic.submit(text)
                self._send(200, json.dumps({"heard": text}).encode(), "application/json")

        cert_dir.mkdir(parents=True, exist_ok=True)
        crt, key = cert_dir / "cert.pem", cert_dir / "key.pem"
        if not crt.exists() or not key.exists():
            subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", str(key), "-out", str(crt),
                            "-days", "30", "-subj", "/CN=blindfold-pickup"], check=True, capture_output=True)
        server = ThreadingHTTPServer(("0.0.0.0", self.port), Handler)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(str(crt), str(key))
        server.socket = ctx.wrap_socket(server.socket, server_side=True)
        self.port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True, name="voice-web").start()
        return server

    def close(self):
        self._server.shutdown()
        self._server.server_close()


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


def listen_for_target() -> str:
    """Blocks until the wearer has named something. May return "" (main.py asks again)."""
    global _listener, _mode
    if _mode is None:
        want = os.environ.get("VOICE_MODE", "auto")
        model = os.environ.get("VOICE_MODEL", "small.en")
        _mode = "type"
        why = "none is plugged in or paired"
        if want == "speech" or (want == "auto" and (os.environ.get("VOICE_DEVICE") or has_microphone())):
            try:
                _listener = Listener(model, os.environ.get("VOICE_DEVICE") or None)
                _mode = "speech"
            except Exception as e:
                if want == "speech":
                    raise
                why = e
        if _mode == "type" and want in ("auto", "web"):
            try:
                _listener = WebMic(int(os.environ.get("VOICE_PORT", "8644")), model=model)
                _mode = "web"
            except Exception as e:
                if want == "web":
                    raise
                print(f"[voice] no microphone here ({why}) and no microphone page ({e}); type the target instead",
                      file=sys.stderr)
    return _listener.listen() if _mode in ("speech", "web") else typed()
