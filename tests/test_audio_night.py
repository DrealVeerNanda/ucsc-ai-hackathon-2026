"""The sound output and the voice recover by themselves."""
import sys
import time
from types import SimpleNamespace

import numpy as np

import audio_out
from audio_out import AudioEngine, ToneConfig


class _Card:
    """Stands in for sounddevice: streams that can be made to go quiet."""
    opened = []

    class OutputStream:
        def __init__(self, callback=None, **kw):
            self.callback, self.latency, self.device, self.closed = callback, 0.01, 0, False
            _Card.opened.append(self)

        def start(self):
            pass

        def stop(self):
            pass

        def abort(self):
            pass

        def close(self):
            self.closed = True

    @staticmethod
    def query_devices(i):
        return {"name": "fake"}


def test_a_sound_output_that_stops_is_opened_again(monkeypatch):
    _Card.opened = []
    monkeypatch.setitem(sys.modules, "sounddevice", _Card)
    eng = AudioEngine(ToneConfig(), clips={})
    first = eng.stream
    out = np.zeros((256, 2), np.float32)
    first.callback(out, 256, None, None)                    # the card is asking for sound: all is well
    eng.set_tone(0.0, 500.0, 2.0, "beep")
    assert eng.stream is first and eng.reopens == 0
    eng._heard -= 3.0                                       # ...then it has not asked for three seconds
    eng._reopened -= 10.0
    eng.say("listen")
    assert eng.stream is not first and first.closed and eng.reopens == 1
    eng.set_tone(0.0, 500.0, 2.0, "beep")                   # and not again straight away
    assert eng.reopens == 1 and len(_Card.opened) == 2


def test_one_failed_sentence_does_not_silence_the_voice(monkeypatch, tmp_path):
    voice_file = tmp_path / "voice.onnx"
    voice_file.write_bytes(b"x")
    calls = []

    class Voice:
        config = SimpleNamespace(sample_rate=22050)

        def synthesize(self, text):
            calls.append(text)
            if len(calls) == 1:
                raise RuntimeError("out of memory")
            return [SimpleNamespace(audio_float_array=np.full(2205, 0.1, np.float32))]
    monkeypatch.setattr(audio_out, "PIPER_VOICE", voice_file)
    monkeypatch.setattr(audio_out, "_piper", None)
    monkeypatch.setattr(audio_out, "_piper_retry", 0.0)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setitem(sys.modules, "piper", SimpleNamespace(PiperVoice=SimpleNamespace(load=lambda path: Voice())))
    assert audio_out._synthesize("first", 48000) is None    # this one is lost
    assert audio_out._synthesize("too soon", 48000) is None and calls == ["first"]      # no reload storm
    monkeypatch.setattr(audio_out, "_piper_retry", time.monotonic() - 1.0)
    assert audio_out._synthesize("later", 48000) is not None and calls == ["first", "later"]
