"""Lane 3: turns a Scene into what the wearer hears, and decides when to say "grab".

    state = guidance.update(scene)     # every frame; returns the state name
    guidance.reset(target)             # after every new target

Sound mapping (the error is object minus hand, in metres, camera axes x right, y down, z forward):

    left / right     the tone pans toward the side the hand should move to
    up / down        pitch rises when the hand should go up, falls for down
    total distance   beeps get faster as the hand gets closer
    within 5 cm      steady tone, then the "grab" clip

States: idle, searching, waiting_hand, guiding, grab. After grab it returns to idle, which
is main.py's cue to ask for the next target.

Settings by environment variable:
    GUIDANCE_AUDIO=auto|on|off   sound card if there is one (default), required, or print only
    GUIDANCE_DEVICE=NAME         sounddevice output device (earbuds)
    GUIDANCE_WORDS=1             fallback: spoken left/right/up/down/forward instead of tones
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

import numpy as np

import tracing
from audio_out import ToneConfig, open_sink
from interfaces import Scene, hand_error

IDLE, SEARCHING, WAITING_HAND, GUIDING, GRAB = "idle", "searching", "waiting_hand", "guiding", "grab"


@dataclass
class GuidanceConfig:
    grab_m: float = 0.05         # closer than this: steady tone, then "grab"
    grab_hold_s: float = 0.3     # ...for this long, so one noisy frame cannot trigger it
    grab_state_s: float = 1.5    # stay in grab this long, then go idle
    dead_m: float = 0.03         # per-axis error inside this is treated as zero
    pan_full_m: float = 0.20     # sideways error that pans fully to one ear
    pitch_full_m: float = 0.20   # vertical error that shifts the pitch by a full octave
    far_m: float = 0.50          # at this distance or more, beeps are at their slowest
    hold_s: float = 0.5          # hand missing for less than this: keep the last sound
    lost_s: float = 1.0          # hand missing for longer than this: back to waiting_hand
    remind_s: float = 6.0        # repeat "show me your hand" this often while waiting
    cant_see_s: float = 8.0      # searching for this long: say "I can't see that"
    words: bool = False          # fallback: spoken single words instead of tones
    word_gap_s: float = 0.5      # at most two words a second


def _dead(v: float, dead: float, full: float) -> float:
    """0 inside +/-dead, then a straight ramp to +/-1 at `full`. No jump at the edge, so the
    tone settles instead of flickering."""
    a = abs(v)
    if a <= dead:
        return 0.0
    return math.copysign(min(1.0, (a - dead) / (full - dead)), v)


def tone_for(err, cfg: GuidanceConfig, tone: ToneConfig) -> tuple[float, float, float, str]:
    """Error in metres -> (pan, freq_hz, beeps_per_s, mode)."""
    pan = _dead(err[0], cfg.dead_m, cfg.pan_full_m)        # object to the right -> right ear
    up = _dead(-err[1], cfg.dead_m, cfg.pitch_full_m)      # camera y points down
    freq = tone.base_hz * 2.0 ** (up * tone.octaves)
    dist = float(np.linalg.norm(err))
    closeness = 1.0 - min(1.0, max(0.0, (dist - cfg.grab_m) / (cfg.far_m - cfg.grab_m)))
    return pan, freq, tone.rate_far + (tone.rate_near - tone.rate_far) * closeness, "beep"


def word_for(err, cfg: GuidanceConfig) -> str | None:
    """The one word that helps most: the axis with the largest error outside the dead zone."""
    axis = int(np.argmax(np.abs(err)))
    if abs(err[axis]) <= cfg.dead_m:
        return None
    return (("left", "right"), ("up", "down"), ("back", "forward"))[axis][int(err[axis] > 0)]


class Guidance:
    def __init__(self, sink=None, cfg: GuidanceConfig | None = None, tone: ToneConfig | None = None):
        self.cfg = cfg or GuidanceConfig()
        self.tone = tone or ToneConfig()
        self.sink = sink if sink is not None else open_sink("off")
        self.state = IDLE
        self.target = None
        self._silent = (0.0, self.tone.base_hz, self.tone.rate_far, "silent")
        self._last_tone = self._silent
        self._clear(0.0)

    def _clear(self, now: float):
        self._had_object = False
        self._search_t = now
        self._said_cant_see = False
        self._last_hand_t = None
        self._remind_t = None
        self._near_t = None
        self._grab_t = None
        self._word_t = -1e9

    def _tone(self, params):
        self._last_tone = params
        self.sink.set_tone(*params)

    def reset(self, target: str, now: float | None = None):
        now = time.monotonic() if now is None else now
        self.target = target
        self.state = SEARCHING
        self._clear(now)
        self._tone(self._silent)
        self.sink.say("looking", now=True)

    def update(self, scene: Scene, now: float | None = None) -> str:
        now = time.monotonic() if now is None else now
        cfg = self.cfg
        if self.state == IDLE:
            self._tone(self._silent)
            return self.state
        if self.state == GRAB:
            self._tone(self._silent)
            if now - self._grab_t >= cfg.grab_state_s:
                self.state = IDLE
            return self.state

        if scene.object_xyz is None:
            if self._had_object:
                self._clear(now)
                self.sink.say("i_lost_it", now=True)
            elif not self._said_cant_see and now - self._search_t > cfg.cant_see_s:
                self._said_cant_see = True
                self.sink.say("i_cant_see_that")
            self.state = SEARCHING
            self._tone(self._silent)
            return self.state
        if not self._had_object:
            self._had_object = True
            self.sink.say("found_it", now=True)

        if scene.hand_xyz is None:
            self._near_t = None
            missing = math.inf if self._last_hand_t is None else now - self._last_hand_t
            if self.state == GUIDING and missing < cfg.hold_s:
                self._tone(self._last_tone)          # a blink: keep the last sound
            elif self.state == GUIDING and missing < cfg.lost_s:
                self._tone(self._silent)             # too old to steer by, not yet given up
            else:
                if self.state != WAITING_HAND or now - self._remind_t >= cfg.remind_s:
                    self._remind_t = now
                    self.sink.say("show_me_your_hand")
                self.state = WAITING_HAND
                self._tone(self._silent)
            return self.state

        self._last_hand_t = now
        self.state = GUIDING
        err = hand_error(scene)
        dist = float(np.linalg.norm(err))
        # Once inside, allow a little slack before counting it as outside again.
        if dist < cfg.grab_m or (self._near_t is not None and dist < 1.3 * cfg.grab_m):
            if self._near_t is None:
                self._near_t = now
            if now - self._near_t >= cfg.grab_hold_s:
                self.state = GRAB
                self._grab_t = now
                self._tone(self._silent)
                self.sink.say("grab", now=True)
                return self.state
            self._tone((0.0, self.tone.steady_hz, self.tone.rate_near, "steady"))
        else:
            self._near_t = None
            if cfg.words:
                self._tone(self._silent)
                word = word_for(err, cfg)
                if word and now - self._word_t >= cfg.word_gap_s and not self.sink.saying():
                    self._word_t = now
                    self.sink.say(word)
            else:
                self._tone(tone_for(err, cfg, self.tone))
        return self.state


# ---------------------------------------------------------------------------
# The interface main.py uses.

_default: Guidance | None = None


def _get() -> Guidance:
    global _default
    if _default is None:
        tone = ToneConfig()
        sink = open_sink(os.environ.get("GUIDANCE_AUDIO", "auto"), os.environ.get("GUIDANCE_DEVICE") or None, tone)
        cfg = GuidanceConfig(words=os.environ.get("GUIDANCE_WORDS", "") not in ("", "0"))
        _default = Guidance(sink, cfg, tone)
    return _default


def update(scene: Scene) -> str:
    tracing.begin_frame(scene.t)
    with tracing.span("guidance"):
        state = _get().update(scene)
    age_ms = (time.monotonic() - scene.t) * 1000      # only meaningful when scene.t is an arrival time
    tracing.put(state=state, **({"e2e_ms": round(age_ms, 3)} if 0 <= age_ms < 10000 else {}))
    return state


def reset(target: str) -> None:
    _get().reset(target)


def say_text(text: str, wait: bool = False) -> None:
    """Say any sentence to the wearer (used by voice.py for what Cosmos sees)."""
    sink = _get().sink
    sink.speak(text)
    if wait:
        sink.wait_quiet()


def pop_phrases() -> list:
    """Everything said since the last call, as words. main.py shows these on the dashboard."""
    log = _get().sink.log
    out = list(log)
    del log[:len(out)]
    return out
