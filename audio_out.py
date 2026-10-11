"""Lane 3: everything the wearer hears. A continuous stereo tone that can change every frame
without gaps or clicks, and short spoken clips from audio/ mixed over it.

    sink = open_sink()                       # sound card if there is one, printed text if not
    sink.set_tone(pan, freq_hz, beeps_per_s, "beep")   # "beep", "steady", "echo" or "silent"
    sink.say("grab")                         # plays audio/grab.wav
    sink.speak("I can see a mug and a phone.")   # any sentence, with a local voice (Piper)

The tone generator is adapted from the RTHACK handguide synth: the oscillator phase is
continuous, every parameter is ramped across each audio block, and beeps have fixed-length
edges. `Mixer.render()` is pure numpy so it can be tested and rendered to a WAV offline.
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import tracing

CLIPS_DIR = Path(__file__).resolve().parent / "audio"
PIPER_VOICE = Path(os.environ.get("PIPER_VOICE") or
                   Path(__file__).resolve().parent / "models" / "piper" / "en_US-lessac-medium.onnx")
# What each clip says, for the dashboard and the printed log.
CLIP_TEXT = {"listen": "(beep: listening)", "looking": "Looking", "found_it": "Found it", "show_me_your_hand": "Show me your hand", "grab": "Grab",
             "i_lost_it": "I lost it", "i_cant_see_that": "I can't see that"}
DIRECTIONS = ("left", "right", "up", "down", "forward", "back")
CLIP_TEXT.update({f"word_{w}": w for w in DIRECTIONS})
CLIP_TEXT.update({f"fast_{w}": f"{w} (fast)" for w in DIRECTIONS})


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
        self._gap = 0.0          # "echo" mode: seconds from the ping to its echo
        self.pan = 0.0
        self.freq = c.base_hz
        self.gain = 0.0          # master gain (0 when silent)
        self.steady = 0.0        # 0 = beeping, 1 = steady tone
        self.phase = 0.0
        self.t_cycle = 0.0

    def set(self, pan: float, freq: float, rate: float, mode: str, gap: float = 0.0):
        """pan -1 (left ear) .. +1 (right ear); mode is beep, steady, echo or silent.

        "echo" plays two notes each cycle: a ping in the middle of the head at the base pitch,
        then, `gap` seconds later, an echo at this pan and pitch. Only the difference between
        the two has to be heard, so nobody needs to know what "level" sounds like."""
        self._target = (float(pan), float(freq), float(rate), mode)
        self._gap = float(gap)

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

        # beep envelope from time-in-cycle with fixed on-length
        period = max(1.0 / max(rate_t, 0.1), c.beep_on_s + 2 * c.edge_s)
        if self.t_cycle >= period:  # period shrank; we were in the silent gap, so env was 0
            self.t_cycle = 0.0
        t = (self.t_cycle + np.arange(n) / c.sr) % period
        self.t_cycle = float((self.t_cycle + n / c.sr) % period)
        env = np.clip(t / c.edge_s, 0, 1) * np.clip((c.beep_on_s - t) / c.edge_s, 0, 1)
        env = np.where(t <= c.beep_on_s, env, 0.0)
        if mode == "echo":
            # The echo is a second beep later in the same cycle. Pitch and side switch to the
            # echo's in the silence between the two, so the switch itself is never heard.
            gap = min(max(self._gap, c.beep_on_s + c.edge_s), period - c.beep_on_s - c.edge_s)
            te = t - gap
            echo = np.clip(te / c.edge_s, 0, 1) * np.clip((c.beep_on_s - te) / c.edge_s, 0, 1)
            env = env + np.where((te >= 0) & (te <= c.beep_on_s), echo, 0.0)
            second = t >= 0.5 * (c.beep_on_s + gap)
            freq = np.where(second, freq, c.base_hz)
            pan = np.where(second, pan, 0.0)

        # continuous-phase oscillator
        ph = self.phase + np.cumsum(2.0 * math.pi * freq / c.sr)
        tone = np.sin(ph)
        self.phase = float(ph[-1] % (2.0 * math.pi))

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


def _resample(x: np.ndarray, src: int, sr: int) -> np.ndarray:
    if src == sr or not x.size:
        return x.astype(np.float32)
    n = int(round(x.size * sr / src))
    return np.interp(np.arange(n) * (src / sr), np.arange(x.size), x).astype(np.float32)


_piper = None
_piper_retry = 0.0             # after a failure, when Piper may be loaded again
_spoken_cache: dict = {}


def synthesize(text: str, sr: int) -> np.ndarray | None:
    """A sentence as mono float32 samples, made on this machine: Piper if its voice file is
    here (about 0.1 s on the GX10), the macOS voice on a Mac, otherwise None. Short phrases are
    kept, since guidance says the same few again and again."""
    key = (text, sr)
    parts = text.split(", ")
    if len(parts) > 1 and all(p.split()[0] in DIRECTIONS for p in parts):
        # "right twenty, up ten": made from its halves, which are kept, so that the hundreds of
        # pairs a reach can ask for cost the loop nothing once each half has been said.
        made = [synthesize(p, sr) for p in parts]
        if any(m is None for m in made):
            return None
        pause = np.zeros(int(0.07 * sr), np.float32)
        return np.concatenate([x for m in made for x in (m, pause)][:-1])
    if key not in _spoken_cache:
        out = _synthesize(text, sr)
        if out is None or len(text) > 40 or len(_spoken_cache) > 300:
            return out
        _spoken_cache[key] = out
    return _spoken_cache[key]


def _synthesize(text: str, sr: int) -> np.ndarray | None:
    global _piper, _piper_retry
    if PIPER_VOICE.exists() and _piper is not False and time.monotonic() >= _piper_retry:
        try:
            if _piper is None:
                from piper import PiperVoice
                _piper = PiperVoice.load(str(PIPER_VOICE))
            chunks = [np.asarray(c.audio_float_array, dtype=np.float32) for c in _piper.synthesize(text)]
            if chunks:
                return _resample(np.concatenate(chunks) * 0.8, _piper.config.sample_rate, sr)
        except Exception as e:
            # One bad sentence must not silence every sentence after it: load the voice afresh
            # next time, but not more than once every ten seconds if it keeps failing.
            print(f"[audio] Piper could not speak ({e}); will load it again", file=sys.stderr)
            _piper, _piper_retry = None, time.monotonic() + 10.0
    if sys.platform == "darwin" and shutil.which("say") and shutil.which("afconvert"):
        with tempfile.TemporaryDirectory() as d:
            aiff, wav = Path(d) / "s.aiff", Path(d) / "s.wav"
            try:
                subprocess.run(["say", "-o", str(aiff), text], check=True, capture_output=True, timeout=20)
                subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@24000", "-c", "1", str(aiff), str(wav)],
                               check=True, capture_output=True, timeout=20)
                return load_clip(wav, sr)
            except Exception:
                return None
    return None


def load_clips(folder: str | Path | None = None, sr: int = 48000) -> dict:
    """Every WAV in the folder, keyed by file name without .wav (grab.wav -> "grab")."""
    if folder is None:
        # Original clips remain the fallback if the optional calm set is incomplete.
        clips = load_clips(CLIPS_DIR, sr)
        clips.update(load_clips(CLIPS_DIR / "calm", sr))
        clips.update(load_clips(CLIPS_DIR / "elevenlabs", sr))
        return clips
    clips = {}
    for p in sorted(Path(folder).glob("*.wav")):
        try:
            clips[p.stem] = load_clip(p, sr)
        except Exception as e:
            print(f"[audio] skipping {p.name}: {e}", file=sys.stderr)
    return clips


def speed_up(x: np.ndarray, factor: float, sr: int) -> np.ndarray:
    """The same words in less time at the same pitch (a "spearcon"). Overlapping 30 ms pieces
    are taken from the clip `factor` times as far apart as they are laid down, each nudged by
    up to 5 ms to where it lines up best with what is already there."""
    win, hop, seek = int(0.030 * sr), int(0.010 * sr), int(0.005 * sr)
    if factor <= 1.0 or x.size < 3 * win:
        return x.astype(np.float32)
    w = np.hanning(win).astype(np.float32)
    n = int((x.size - win - seek) / (hop * factor))
    out = np.zeros(n * hop + win, np.float32)
    norm = np.zeros_like(out)
    for i in range(n):
        at = int(i * hop * factor)
        if i:
            want = out[i * hop:i * hop + hop]            # the tail the next piece has to continue
            best = max(range(0, seek + 1, 8), key=lambda d: float(np.dot(want, x[at + d:at + d + hop])))
            at += best
        out[i * hop:i * hop + win] += x[at:at + win] * w
        norm[i * hop:i * hop + win] += w
    return out / np.maximum(norm, 0.1)


def short_words(sr: int, fast: float = 1.7) -> dict:
    """One-word directions for the options that repeat them quickly: "word_left" is the
    shortest recording of "left" there is (the sets in elevenlabs/ and calm/ say whole
    sentences, 1.4 s each), and "fast_left" is that word again in about half the time."""
    out = {}
    for name, clip in load_clips(CLIPS_DIR, sr).items():
        if name in DIRECTIONS:
            loud = np.flatnonzero(np.abs(clip) > 0.02)       # the recordings start and end with silence
            if loud.size:
                clip = clip[max(0, loud[0] - sr // 100):loud[-1] + sr // 50]
            out[f"word_{name}"] = clip
            out[f"fast_{name}"] = speed_up(clip, fast, sr) if clip.size > 0.3 * sr else clip    # "up" is short enough as it is
    return out


def success_chime(sr: int) -> np.ndarray:
    """Quiet two-note confirmation with soft edges and a pause before 'grab'."""
    out = np.zeros(round(0.48 * sr), dtype=np.float32)
    for start_s, freq, gain in ((0.0, 659.25, 0.16), (0.15, 880.0, 0.14)):
        t = np.arange(round(0.30 * sr)) / sr
        envelope = np.minimum(t / 0.018, 1.0) * np.exp(-t / 0.085)
        envelope *= np.clip((0.30 - t) / 0.035, 0.0, 1.0)
        note = gain * envelope * (np.sin(2 * np.pi * freq * t) +
                                  0.15 * np.sin(4 * np.pi * freq * t))
        start = round(start_s * sr)
        out[start:start + len(note)] += note.astype(np.float32)
    return np.concatenate((out, np.zeros(round(0.12 * sr), dtype=np.float32)))


def listen_beep(sr: int) -> np.ndarray:
    """One short high note: the microphone is open, speak now. Unlike the two-note chime."""
    t = np.arange(round(0.16 * sr)) / sr
    envelope = np.minimum(t / 0.01, 1.0) * np.clip((0.16 - t) / 0.03, 0.0, 1.0)
    return np.concatenate(((0.22 * envelope * np.sin(2 * np.pi * 1175.0 * t)).astype(np.float32),
                           np.zeros(round(0.08 * sr), dtype=np.float32)))


class Mixer:
    """Tone plus clips, rendered block by block. Holds no sound device."""

    def __init__(self, cfg: ToneConfig | None = None, clips: dict | None = None, stale_s: float = 0.5):
        self.cfg = cfg or ToneConfig()
        self.synth = ToneSynth(self.cfg)
        self.clips = dict(clips) if clips is not None else load_clips(sr=self.cfg.sr)
        self.clips["aligned"] = success_chime(self.cfg.sr)
        self.clips["listen"] = listen_beep(self.cfg.sr)
        if clips is None:
            self.clips.update(short_words(self.cfg.sr))
        self._grab_clip = (np.concatenate((success_chime(self.cfg.sr), self.clips["grab"]))
                           if "grab" in self.clips else None)
        self.stale_s = stale_s
        self.rechime_s = 1.5                 # the lined-up chime this recently: "Grab" comes without its own
        self.spoken: list[str] = []          # names actually started, for tests and logs
        self.log: list[str] = []             # the words of everything said, for the dashboard
        self._lock = threading.Lock()
        self._clip = None                    # samples of the clip being played
        self._pos = 0
        self._next = None                    # name waiting for the current clip to end
        self._duck = 1.0
        self._last_set = time.monotonic()
        self._played = 0                     # samples rendered so far: the mixer's own clock
        self._chimed = None                  # ...and its reading when the lined-up chime last started

    def set_tone(self, pan: float, freq: float, rate: float, mode: str, gap: float = 0.0):
        self._last_set = time.monotonic()
        self.synth.set(pan, freq, rate, mode, gap)

    def cancel(self, name: str):
        """Drop a phrase that is still waiting its turn and is no longer true, such as "show
        me your hand" once the hand is in view. One already playing is left to finish."""
        with self._lock:
            if self._next == name:
                self._next = None

    def say(self, name: str, now: bool = False):
        """Play a clip after the current one. now=True cuts the current one off, because a
        stale phrase is worse than a clipped one. A phrase still waiting is replaced."""
        with self._lock:
            if now or self._clip is None:
                self._start(name)
                self._next = None
            else:
                self._next = name

    def _start(self, name):
        if isinstance(name, tuple):          # (text, samples) from speak()
            text, clip = name
            self.spoken.append(text)
        else:
            text = CLIP_TEXT.get(name, name.replace("_", " "))
            self.spoken.append(name)
            clip = self.clips.get(name)
            if name == "aligned":
                self._chimed = self._played
            elif name == "grab" and self._grab_clip is not None:
                # "Grab" opens with the chime, unless the same chime has only just been heard
                # for lining up: twice in a second sounds like a stutter.
                recent = self._chimed is not None and self._played - self._chimed < self.rechime_s * self.cfg.sr
                clip = clip if recent else self._grab_clip
            if clip is None:
                print(f"[audio] no clip audio/{name}.wav", file=sys.stderr)
        self.log.append(text)
        tracing.event("say", text=text)
        if clip is None:
            self._clip = None
        else:
            self._clip, self._pos = clip, 0

    def speak(self, text: str, now: bool = False):
        """Say any sentence. It is synthesised here first (a fraction of a second), then played
        like a clip. With no voice available the words still reach the log and the dashboard."""
        samples = synthesize(text, self.cfg.sr)
        with self._lock:
            if now or self._clip is None:
                self._start((text, samples))
                self._next = None
            else:
                self._next = (text, samples)

    def saying(self) -> bool:
        with self._lock:
            return self._clip is not None or self._next is not None

    def wait_quiet(self, timeout_s: float = 15.0):
        """Block until nothing is being said, e.g. before opening the microphone."""
        end = time.monotonic() + timeout_s
        while self.saying() and time.monotonic() < end:
            time.sleep(0.05)

    def render(self, n: int) -> np.ndarray:
        # If nobody has updated the tone for a while (stream stalled, loop crashed), go quiet:
        # a tone computed from an old position must never keep steering.
        if time.monotonic() - self._last_set > self.stale_s:
            self.synth.set(0.0, self.cfg.base_hz, self.cfg.rate_far, "silent")
        out = self.synth.render(n)
        with self._lock:
            self._played += n
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
        self._device, self._blocksize = device, blocksize
        self._blocks = self._underruns = 0
        self._render_max = 0.0
        self._reported = self._heard = self._reopened = time.monotonic()
        self.reopens = 0
        self.stream = self._open()
        tracing.meta(audio_out_ms=round(float(self.stream.latency) * 1000, 1), audio_block=blocksize,
                     audio_device=str(sd.query_devices(self.stream.device).get("name", "")))

    def _open(self):
        import sounddevice as sd
        stream = sd.OutputStream(samplerate=self.cfg.sr, channels=2, dtype="float32", device=self._device,
                                 blocksize=self._blocksize, callback=self._cb, latency="low")
        stream.start()
        return stream

    def _keep_alive(self):
        """If the sound card has stopped asking for sound (the sound server restarted, or the
        device went away with nothing to fall back to), open the output again. Without this a
        dead stream means silence for the rest of the run, and every sentence waits out its
        full time-out because nothing ever finishes playing. Called from the pipeline's side,
        never from the sound callback."""
        now = time.monotonic()
        if now - self._heard < 2.0 or now - self._reopened < 5.0:
            return
        self._reopened = now
        try:
            old, self.stream = self.stream, None
            try:
                old.abort()
                old.close()
            except Exception:
                pass
            self.stream = self._open()
            self._heard = time.monotonic()
            self.reopens += 1
            tracing.event("audio.reopened", count=self.reopens)
            print("[audio] the sound output had stopped; opened it again", file=sys.stderr)
        except Exception as e:
            tracing.event("audio.reopen_failed", error=str(e)[:120])
            print(f"[audio] the sound output has stopped and will not reopen yet ({e})", file=sys.stderr)

    def set_tone(self, *tone):
        self._keep_alive()
        super().set_tone(*tone)                  # whatever the mixer takes: some styles send a fifth value

    def say(self, name: str, now: bool = False):
        self._keep_alive()
        super().say(name, now)

    def speak(self, text: str, now: bool = False):
        self._keep_alive()
        super().speak(text, now)

    def _cb(self, outdata, frames, time_info, status):
        t = time.perf_counter()
        self._heard = time.monotonic()
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
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()


class PrintSink:
    """Stands in for the sound card: prints what would be heard, a few times a second."""

    def __init__(self, print_hz: float = 4.0, quiet: bool = False):
        self.every = 1.0 / print_hz
        self.quiet = quiet
        self.spoken: list[str] = []
        self.log: list[str] = []
        self.tone = (0.0, 0.0, 0.0, "silent")
        self.gap = 0.0
        self._last = 0.0

    def set_tone(self, pan: float, freq: float, rate: float, mode: str, gap: float = 0.0):
        self.tone = (pan, freq, rate, mode)
        self.gap = gap
        now = time.monotonic()
        if not self.quiet and mode != "silent" and now - self._last >= self.every:
            self._last = now
            side = "centre" if abs(pan) < 0.05 else f"{'right' if pan > 0 else 'left'} {abs(pan):.1f}"
            print(f"[audio] {mode:6s} {side:10s} {freq:4.0f} Hz  {rate:4.1f} beeps/s")

    def say(self, name: str, now: bool = False):
        self.spoken.append(name)
        self.log.append(CLIP_TEXT.get(name, name.replace("_", " ")))
        tracing.event("say", text=self.log[-1])
        if not self.quiet:
            print(f"[audio] says: {name.replace('_', ' ')}")

    def speak(self, text: str, now: bool = False):
        self.spoken.append(text)
        self.log.append(text)
        tracing.event("say", text=text)
        if not self.quiet:
            print(f"[audio] says: {text}")

    def cancel(self, name: str):
        pass

    def saying(self) -> bool:
        return False

    def wait_quiet(self, timeout_s: float = 15.0):
        pass

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
