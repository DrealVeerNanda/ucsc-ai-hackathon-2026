"""Lane 3: everything the wearer hears. A continuous stereo tone that can change every frame
without gaps or clicks, and short spoken clips from audio/ mixed over it.

    sink = open_sink()                       # sound card if there is one, printed text if not
    sink.set_tone(pan, freq_hz, beeps_per_s, "beep")   # "beep", "steady" or "silent"
    sink.say("grab")                         # plays audio/grab.wav

The tone generator is adapted from the RTHACK handguide synth: the oscillator phase is
continuous, every parameter is ramped across each audio block, and beeps have fixed-length
edges. `Mixer.render()` is pure numpy so it can be tested and rendered to a WAV offline.
"""
from __future__ import annotations

import math
import sys
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import tracing

CLIPS_DIR = Path(__file__).resolve().parent / "audio"


@dataclass
class ToneConfig:
    sr: int = 48000
    base_hz: float = 523.25      # C5 when the hand is level with the object
    octaves: float = 1.0         # full "up" or "down" shifts the pitch by this much
    rate_far: float = 2.0        # beeps per second when far
    rate_near: float = 12.0      # beeps per second when almost there
    steady_hz: float = 880.0     # the "you are on it" tone
    beep_on_s: float = 0.045
    edge_s: float = 0.008        # attack / release
    smooth_s: float = 0.04       # parameter smoothing time constant
    volume: float = 0.25
    clip_volume: float = 0.9
    duck: float = 0.3            # tone level while a clip is speaking


class ToneSynth:
    def __init__(self, cfg: ToneConfig | None = None):
        self.cfg = cfg or ToneConfig()
        c = self.cfg
        self._target = (0.0, c.base_hz, c.rate_far, "silent")
        self.pan = 0.0
        self.freq = c.base_hz
        self.gain = 0.0          # master gain (0 when silent)
        self.steady = 0.0        # 0 = beeping, 1 = steady tone
        self.phase = 0.0
        self.t_cycle = 0.0

    def set(self, pan: float, freq: float, rate: float, mode: str):
        """pan -1 (left ear) .. +1 (right ear); mode is beep, steady or silent."""
        self._target = (float(pan), float(freq), float(rate), mode)

    def _approach(self, cur, tgt, n):
        k = 1.0 - math.exp(-n / (self.cfg.smooth_s * self.cfg.sr))
        return cur + (tgt - cur) * k

    def render(self, n: int) -> np.ndarray:
        c = self.cfg
        pan_t, freq_t, rate_t, mode = self._target
        gain_t = 0.0 if mode == "silent" else 1.0
        steady_t = 1.0 if mode == "steady" else 0.0

        ramp = np.arange(1, n + 1, dtype=np.float64) / n
        pan_n = self._approach(self.pan, pan_t, n)
        # smooth pitch in log space so glides sound even
        freq_n = math.exp(self._approach(math.log(self.freq), math.log(freq_t), n))
        gain_n = self._approach(self.gain, gain_t, n)
        steady_n = self._approach(self.steady, steady_t, n)

        pan = self.pan + (pan_n - self.pan) * ramp
        freq = self.freq + (freq_n - self.freq) * ramp
        gain = self.gain + (gain_n - self.gain) * ramp
        steady = self.steady + (steady_n - self.steady) * ramp

        # continuous-phase oscillator
        ph = self.phase + np.cumsum(2.0 * math.pi * freq / c.sr)
        tone = np.sin(ph)
        self.phase = float(ph[-1] % (2.0 * math.pi))

        # beep envelope from time-in-cycle with fixed on-length
        period = max(1.0 / max(rate_t, 0.1), c.beep_on_s + 2 * c.edge_s)
        if self.t_cycle >= period:  # period shrank; we were in the silent gap, so env was 0
            self.t_cycle = 0.0
        t = (self.t_cycle + np.arange(n) / c.sr) % period
        self.t_cycle = float((self.t_cycle + n / c.sr) % period)
        env = np.clip(t / c.edge_s, 0, 1) * np.clip((c.beep_on_s - t) / c.edge_s, 0, 1)
        env = np.where(t <= c.beep_on_s, env, 0.0)

        amp = c.volume * gain * (env * (1.0 - steady) + steady)
        theta = (pan + 1.0) * (math.pi / 4.0)  # equal-power pan
        out = np.empty((n, 2), dtype=np.float32)
        out[:, 0] = tone * amp * np.cos(theta)
        out[:, 1] = tone * amp * np.sin(theta)

        self.pan, self.freq, self.gain, self.steady = pan_n, freq_n, gain_n, steady_n
        return out


def load_clip(path: str | Path, sr: int) -> np.ndarray:
    """16-bit PCM WAV -> mono float32 at the engine's sample rate."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16-bit PCM")
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        if w.getnchannels() > 1:
            x = x.reshape(-1, w.getnchannels()).mean(axis=1)
        src = w.getframerate()
    if src != sr and x.size:
        n = int(round(x.size * sr / src))
        x = np.interp(np.arange(n) * (src / sr), np.arange(x.size), x).astype(np.float32)
    return x


def load_clips(folder: str | Path = CLIPS_DIR, sr: int = 48000) -> dict:
    """Every WAV in the folder, keyed by file name without .wav (grab.wav -> "grab")."""
    clips = {}
    for p in sorted(Path(folder).glob("*.wav")):
        try:
            clips[p.stem] = load_clip(p, sr)
        except Exception as e:
            print(f"[audio] skipping {p.name}: {e}", file=sys.stderr)
    return clips


class Mixer:
    """Tone plus clips, rendered block by block. Holds no sound device."""

    def __init__(self, cfg: ToneConfig | None = None, clips: dict | None = None, stale_s: float = 0.5):
        self.cfg = cfg or ToneConfig()
        self.synth = ToneSynth(self.cfg)
        self.clips = clips if clips is not None else load_clips(sr=self.cfg.sr)
        self.stale_s = stale_s
        self.spoken: list[str] = []          # names actually started, for tests and logs
        self._lock = threading.Lock()
        self._clip = None                    # samples of the clip being played
        self._pos = 0
        self._next = None                    # name waiting for the current clip to end
        self._duck = 1.0
        self._last_set = time.monotonic()

    def set_tone(self, pan: float, freq: float, rate: float, mode: str):
        self._last_set = time.monotonic()
        self.synth.set(pan, freq, rate, mode)

    def say(self, name: str, now: bool = False):
        """Play a clip after the current one. now=True cuts the current one off, because a
        stale phrase is worse than a clipped one. A phrase still waiting is replaced."""
        with self._lock:
            if now or self._clip is None:
                self._start(name)
                self._next = None
            else:
                self._next = name

    def _start(self, name: str):
        self.spoken.append(name)
        clip = self.clips.get(name)
        if clip is None:
            print(f"[audio] no clip audio/{name}.wav", file=sys.stderr)
            self._clip = None
        else:
            self._clip, self._pos = clip, 0

    def saying(self) -> bool:
        with self._lock:
            return self._clip is not None or self._next is not None

    def render(self, n: int) -> np.ndarray:
        # If nobody has updated the tone for a while (stream stalled, loop crashed), go quiet:
        # a tone computed from an old position must never keep steering.
        if time.monotonic() - self._last_set > self.stale_s:
            self.synth.set(0.0, self.cfg.base_hz, self.cfg.rate_far, "silent")
        out = self.synth.render(n)
        with self._lock:
            duck_t = self.cfg.duck if self._clip is not None else 1.0
            out *= np.linspace(self._duck, duck_t, n, dtype=np.float32)[:, None]
            self._duck = duck_t
            done = 0
            while self._clip is not None and done < n:
                seg = self._clip[self._pos:self._pos + (n - done)]
                out[done:done + seg.size] += (seg * self.cfg.clip_volume)[:, None]
                self._pos += seg.size
                done += seg.size
                if self._pos >= self._clip.size:
                    self._clip = None
                    if self._next is not None:
                        name, self._next = self._next, None
                        self._start(name)
        return out

    def close(self):
        pass


class AudioEngine(Mixer):
    """Mixer played through a sound card with sounddevice."""

    def __init__(self, cfg: ToneConfig | None = None, clips: dict | None = None, device=None,
                 blocksize: int = 256):
        import sounddevice as sd

        super().__init__(cfg, clips)
        self.stream = sd.OutputStream(samplerate=self.cfg.sr, channels=2, dtype="float32", device=device,
                                      blocksize=blocksize, callback=self._cb, latency="low")
        self.stream.start()
        self._blocks = self._underruns = 0
        self._render_max = 0.0
        self._reported = time.monotonic()
        tracing.meta(audio_out_ms=round(float(self.stream.latency) * 1000, 1), audio_block=blocksize,
                     audio_device=str(sd.query_devices(self.stream.device).get("name", "")))

    def _cb(self, outdata, frames, time_info, status):
        t = time.perf_counter()
        outdata[:] = self.render(frames)
        self._render_max = max(self._render_max, (time.perf_counter() - t) * 1000)
        self._blocks += 1
        self._underruns += bool(status and status.output_underflow)
        now = time.monotonic()
        if now - self._reported >= 5.0:      # a summary now and then, not a line per 5 ms block
            tracing.event("audio", render_max_ms=round(self._render_max, 3), blocks=self._blocks,
                          underruns=self._underruns)
            self._blocks = self._underruns = 0
            self._render_max, self._reported = 0.0, now

    def close(self):
        self.synth.set(0.0, self.cfg.base_hz, 1.0, "silent")
        time.sleep(0.1)
        self.stream.stop()
        self.stream.close()


class PrintSink:
    """Stands in for the sound card: prints what would be heard, a few times a second."""

    def __init__(self, print_hz: float = 4.0, quiet: bool = False):
        self.every = 1.0 / print_hz
        self.quiet = quiet
        self.spoken: list[str] = []
        self.tone = (0.0, 0.0, 0.0, "silent")
        self._last = 0.0

    def set_tone(self, pan: float, freq: float, rate: float, mode: str):
        self.tone = (pan, freq, rate, mode)
        now = time.monotonic()
        if not self.quiet and mode != "silent" and now - self._last >= self.every:
            self._last = now
            side = "centre" if abs(pan) < 0.05 else f"{'right' if pan > 0 else 'left'} {abs(pan):.1f}"
            print(f"[audio] {mode:6s} {side:10s} {freq:4.0f} Hz  {rate:4.1f} beeps/s")

    def say(self, name: str, now: bool = False):
        self.spoken.append(name)
        if not self.quiet:
            print(f"[audio] says: {name.replace('_', ' ')}")

    def saying(self) -> bool:
        return False

    def close(self):
        pass


def open_sink(mode: str = "auto", device=None, cfg: ToneConfig | None = None):
    """mode: "on" needs a sound card, "off" prints, "auto" uses the card if it opens."""
    if mode != "off":
        try:
            return AudioEngine(cfg, device=device)
        except Exception as e:
            if mode == "on":
                raise
            print(f"[audio] no sound output ({e}); printing cues instead", file=sys.stderr)
    return PrintSink()


def write_wav(path: str | Path, stereo: np.ndarray, sr: int):
    data = (np.clip(stereo, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data.tobytes())
