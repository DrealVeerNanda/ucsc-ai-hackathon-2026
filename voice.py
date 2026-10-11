"""Lane 3: what does the wearer want to pick up?

    target = voice.listen_for_target()     # blocks, returns e.g. "mug"

Push-to-talk with Whisper running on the GX10 (faster-whisper), or a typed request as the backup
for a noisy hall. Either way the words then become an object name: Nemotron does it when it is
running (bash tools/run_nemotron.sh), so "I'm thirsty" becomes "water bottle"; a simple rule does
it otherwise, so "find my mug" still becomes "mug". Nothing leaves the machine.

Asking "what's on the table?" (or "what can I pick up?") makes Cosmos look at the camera picture
and say what it sees; the wearer then names one, or says "the second one". That needs the Cosmos
server (bash tools/run_cosmos.sh) and the real stream.

    VOICE_MODE=auto      a microphone on this machine if there is one; otherwise the web page below
                         if faster-whisper is installed; otherwise typed (default)
    VOICE_MODE=web       any phone or laptop is the microphone: it opens https://THIS_MACHINE:8644,
                         and you hold the button and speak. Typing in the terminal still works too.
    VOICE_MODE=type      always typed
    VOICE_MODE=speech    always the microphone on this machine
    VOICE_MODEL=base.en  Whisper model size (tiny.en is faster, small.en is more accurate)
    VOICE_DEVICE=NAME    sounddevice input device (the earbuds' microphone)
    VOICE_LLM_URL=URL    Nemotron's address (default http://127.0.0.1:8000/v1); "off" uses the rule only

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
              'detector can find, such as mug, water bottle, phone. Name the physical thing, not the wish: thirsty '
              'means water bottle. intent is "none" for anything else. target is an empty string unless intent is "pick".')
LLM_SCHEMA = {"type": "object",
              "properties": {"intent": {"type": "string", "enum": ["pick", "list", "none"]},
                             "target": {"type": "string", "maxLength": 30}},
              "required": ["intent", "target"], "additionalProperties": False}
_LIST = re.compile(r"\b(?:what(?:'s| is| are)?\b.*\b(?:on|in front|here|there|around|see|table|desk|pick up|grab|reach)"
                   r"|what do you see|what can you see|what can i|look around|describe|list (?:the )?(?:objects|things))\b")
_YES = re.compile(r"^(?:yes|yeah|yep|sure|ok|okay|please|that one|i want it|yes please|yes i do|get it)$")
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


def describe() -> list[str]:
    """Look through the camera with Cosmos and tell the wearer what could be picked up."""
    global last_objects
    import cosmos
    frame = _camera_frame()
    if frame is None:
        _say("I need the camera for that.")
        return []
    if not cosmos.available():
        print("[voice] Cosmos is not running. Start it with: bash tools/run_cosmos.sh", file=sys.stderr)
        _say("I can't look around right now.")
        return []
    try:
        objects = cosmos.list_objects(frame.rgb)
    except Exception as e:
        print(f"[voice] Cosmos failed ({e})", file=sys.stderr)
        _say("I can't look around right now.")
        return []
    last_objects = objects
    if not objects:
        _say("I don't see anything to pick up.")
    elif len(objects) == 1:
        _say(f"I can see one thing: {objects[0]}. Do you want it?")
    else:
        _say("I can see " + ", ".join(objects[:-1]) + " and " + objects[-1] + ". Which one?")
    return objects


def respond(text: str) -> str:
    """What to do with what was said: the target to look for, or "" after answering a question
    (main.py then asks again, and the wearer names their pick)."""
    global last_transcript
    if last_objects and _YES.match(re.sub(r"[^a-z' ]+", " ", text.lower()).strip()):
        last_transcript = text
        return last_objects[0]                               # "yes" to "Do you want it?"
    intent, target = interpret(text)
    if intent == "list":
        describe()
        return ""
    return target if intent == "pick" else ""


def typed() -> str:
    try:
        return respond(input("what do you want to pick up?> "))
    except EOFError:
        return ""


class Listener:
    """Records between two presses of Enter and transcribes with faster-whisper."""

    def __init__(self, model: str = "base.en", device=None):
        import sounddevice as sd
        from faster_whisper import WhisperModel

        self.sd = sd
        self.device = device
        sd.check_input_settings(device=device, samplerate=SR, channels=1)   # fails if there is no microphone
        with tracing.timed("load.whisper", model=model):
            self.whisper = WhisperModel(model, device="cpu", compute_type="int8")

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
        segments, _ = self.whisper.transcribe(audio, language="en", beam_size=1, vad_filter=True)
        return " ".join(s.text for s in segments).strip()

    def listen(self) -> str:
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


class WebMic:
    """Lets any phone or laptop be the microphone and the push-to-talk button. It opens a page
    served from here, records while the button is held, and sends the recording to this machine,
    where Whisper transcribes it. Browsers only allow the microphone over HTTPS, so the page is
    served with a certificate made on the spot; the browser asks once whether to trust it."""

    def __init__(self, port: int = 8644, transcribe=None, model: str = "base.en", cert_dir: str | Path = "out/voice_cert",
                 typed: bool = True):
        self.port = port
        self.status = {"heard": "", "target": ""}
        self._requests: queue.Queue = queue.Queue()
        self._waiting = False
        if transcribe is None:
            from faster_whisper import WhisperModel
            from faster_whisper.audio import decode_audio
            with tracing.timed("load.whisper", model=model):
                whisper = WhisperModel(model, device="cpu", compute_type="int8")

            def transcribe(data: bytes) -> str:
                audio = decode_audio(io.BytesIO(data), sampling_rate=SR)     # webm, mp4 or wav: whatever the browser sent
                if audio.size < SR // 4:
                    return ""
                t = time.perf_counter()
                segments, _ = whisper.transcribe(audio, language="en", beam_size=1, vad_filter=True)
                text = " ".join(s.text for s in segments).strip()
                tracing.event("whisper", (time.perf_counter() - t) * 1000, audio_s=round(audio.size / SR, 2), text=text)
                return text
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

    def listen(self) -> str:
        self._waiting = True
        try:
            text = self._requests.get()
        finally:
            self._waiting = False
        target = respond(text)
        self.status["target"] = target or self.status["target"]
        print(f"[voice] heard {text!r} -> target {target!r}")
        return target

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
        model = os.environ.get("VOICE_MODEL", "base.en")
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
