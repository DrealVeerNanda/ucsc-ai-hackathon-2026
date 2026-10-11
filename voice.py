"""Lane 3: what does the wearer want to pick up?

    target = voice.listen_for_target()     # blocks, returns e.g. "mug"

Push-to-talk with Whisper running on the GX10 (faster-whisper), or a typed request as the backup
for a noisy hall. Either way the words then become an object name: Nemotron does it when it is
running (bash tools/run_nemotron.sh), so "I'm thirsty" becomes "water bottle"; a simple rule does
it otherwise, so "find my mug" still becomes "mug". Nothing leaves the machine.

    VOICE_MODE=auto      speech if a microphone and faster-whisper are there, else typed (default)
    VOICE_MODE=type      always typed
    VOICE_MODE=speech    always speech
    VOICE_MODEL=base.en  Whisper model size (tiny.en is faster, small.en is more accurate)
    VOICE_DEVICE=NAME    sounddevice input device (the earbuds' microphone)
    VOICE_LLM_URL=URL    Nemotron's address (default http://127.0.0.1:8000/v1); "off" uses the rule only

Speech: press Enter, say "find my mug", press Enter again. A Bluetooth clicker that sends
Enter works as the button.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
import urllib.request

import numpy as np

import tracing

SR = 16000          # what Whisper expects
MAX_S = 8.0         # stop recording after this even without the second Enter

_LEAD = re.compile(
    r"^(?:(?:please|hey|ok|okay|can you|could you|i want you to|i want to|i want|i need|help me)\s+)*"
    r"(?:(?:find|get|grab|fetch|locate|pick up|pick|take|reach for|reach|look for|search for|"
    r"where is|where's|where are|give me|hand me|bring me|show me)\s+)?"
    r"(?:me\s+)?(?:(?:my|the|a|an|that|this|some|one of the)\s+)?")
_FILLER = {"please", "thanks", "thank you", "ok", "okay", "hey", "um", "uh", "it", "that", "this"}
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


LLM_PROMPT = ('You turn what a blind person says into the one object they want to pick up. Reply with JSON only: '
              '{"target": "<the object as 1 to 3 plain words an object detector can find, such as mug, water bottle, phone>"}. '
              'Name the physical thing, not the wish: thirsty means water bottle. If no object is named or implied, '
              'use an empty string.')
LLM_SCHEMA = {"type": "object", "properties": {"target": {"type": "string", "maxLength": 30}},
              "required": ["target"], "additionalProperties": False}
last_transcript: str | None = None      # what was last said or typed, for the dashboard
_llm_ok: bool | None = None             # None until the first request shows whether Nemotron is there


def ask_llm(text: str, url: str, timeout_s: float = 4.0) -> str:
    """Nemotron's reading of the request. Raises if the server is not there."""
    body = {"model": "nemotron", "temperature": 0.0, "max_tokens": 40,
            "chat_template_kwargs": {"enable_thinking": False},      # the answer is one word: no need to reason aloud
            "messages": [{"role": "system", "content": LLM_PROMPT}, {"role": "user", "content": text}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "target", "schema": LLM_SCHEMA, "strict": True}}}
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        reply = json.load(r)
    return extract_target(str(json.loads(reply["choices"][0]["message"]["content"]).get("target", "")))


def understand(text: str) -> str:
    """Words -> object name. Short requests ("mug", "find my mug") go by the rule; longer or
    indirect ones ("I'm thirsty") go to Nemotron if it is running. If Nemotron says no object
    was asked for, the answer is "" and main.py asks again; the rule only steps in when Nemotron
    is not there."""
    global _llm_ok, last_transcript
    last_transcript = text
    rule = extract_target(text)
    url = os.environ.get("VOICE_LLM_URL", "http://127.0.0.1:8000/v1")
    if url == "off" or _llm_ok is False or len(text.split()) < 3:
        return rule
    t = time.perf_counter()
    try:
        target = ask_llm(text, url)
    except Exception as e:
        if _llm_ok is None:
            print(f"[voice] Nemotron is not answering at {url} ({e.__class__.__name__}); using the simple rule. "
                  "Start it with: bash tools/run_nemotron.sh", file=sys.stderr)
        _llm_ok = False
        return rule
    _llm_ok = True
    tracing.event("nemotron", (time.perf_counter() - t) * 1000, text=text, target=target)
    return target


def typed() -> str:
    try:
        return understand(input("what do you want to pick up?> "))
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
        target = understand(text)
        print(f"[voice] heard {text!r} -> target {target!r}")
        return target


_listener: Listener | None = None
_mode: str | None = None


def listen_for_target() -> str:
    """Blocks until the wearer has named something. May return "" (main.py asks again)."""
    global _listener, _mode
    if _mode is None:
        _mode = os.environ.get("VOICE_MODE", "auto")
        if _mode != "type":
            try:
                _listener = Listener(os.environ.get("VOICE_MODEL", "base.en"), os.environ.get("VOICE_DEVICE") or None)
                _mode = "speech"
            except Exception as e:
                if _mode == "speech":
                    raise
                print(f"[voice] speech input not available ({e}); type the target instead", file=sys.stderr)
                _mode = "type"
    return _listener.listen() if _mode == "speech" else typed()
