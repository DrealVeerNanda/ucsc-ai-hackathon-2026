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
    GUIDANCE_MODE=tones|words|hybrid  explicit mode, overrides GUIDANCE_WORDS
    GUIDANCE_MODE=1..5 or a name      one of OPTIONS below, to compare ways of guiding. It can be
                                      changed while running: say or type "option 3".
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

# Ways of guiding, to try against each other. (key, name said aloud, what the wearer hears)
OPTIONS = [
    ("tones", "tones", "Everything at once in sound: left and right is the ear, up and down is the pitch, "
                       "faster beeps are closer."),
    ("steps", "one step at a time", "One direction is spoken, beeps speed up until that one is lined up, a chime, "
                                    "then the next direction."),
    ("combined", "combined words", "The whole direction at once, like 'down and right', spoken again only when it "
                                   "changes. A chime when lined up, then 'forward'."),
    ("combined_beeps", "combined words with beeps", "Combined words, plus quiet beeps that speed up as the hand "
                                                    "gets closer and lean to the side it should move."),
    ("clock", "clock face", "'Two o'clock' for which way to move, a chime when lined up, then 'forward', "
                            "with the same quiet beeps."),
]
DYNAMIC = ("combined", "combined_beeps", "clock")
HOURS = ("twelve", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven")


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
    hybrid: bool = False         # say one direction, beep along that axis, chime, next axis
    step_m: float = 0.025        # axis is aligned within 2.5 cm
    step_hold_s: float = 0.2     # alignment must persist before the step chime
    step_pause_s: float = 0.6    # leave time for the step chime before the next instruction
    word_gap_s: float = 1.0      # minimum pause before a different direction
    word_repeat_s: float = 2.5   # gentle reminder if the direction stays the same
    option: str = "tones"        # which of OPTIONS is in use
    lead_s: float = 0.2          # judge the direction from where the hand will be this much later: the
                                 # sound reaches the ears late, and a moving hand has overshot by then
    line_in_m: float = 0.025     # sideways or vertical error below this: lined up...
    line_out_m: float = 0.05     # ...and stays lined up until it grows past this, so it cannot flip-flop
    reach_m: float = 0.03        # lined up, and the object is this much further away: "forward"
    back_m: float = 0.07         # only say "back" when the hand is clearly past the object
    phrase_gap_s: float = 0.7    # minimum pause before a different phrase


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
        self._last_word = None
        self._word_axis = None
        self._step_axis = None
        self._step_word = None
        self._step_near_t = None
        self._step_pause_until = 0.0
        self._prev_hand = None
        self._prev_t = 0.0
        self._vel = None
        self._lined = [False, False]
        self._was_lined = False
        self._phrase = None
        self._phrase_t = -1e9

    def set_option(self, key: str):
        """Change how guidance sounds. Safe in the middle of a reach."""
        if key not in [k for k, _, _ in OPTIONS] + ["words"]:
            raise ValueError(f"unknown guidance option {key!r}")
        self.cfg.option = key
        self.cfg.hybrid, self.cfg.words = key == "steps", key == "words"
        self.tone.volume = 0.25 if key == "tones" else 0.16      # beeps under speech are kept quiet
        self._step_axis = self._step_word = self._phrase = None
        self._lined, self._was_lined = [False, False], False
        self._tone(self._silent)

    def _phrase_for(self, lead) -> str | None:
        """What to say for this error, for the options that give the whole direction at once."""
        cfg = self.cfg
        for a in (0, 1):
            self._lined[a] = abs(lead[a]) < (cfg.line_out_m if self._lined[a] else cfg.line_in_m)
        if all(self._lined):
            if lead[2] > cfg.reach_m:
                return "forward"
            return "back" if lead[2] < -cfg.back_m else None
        if cfg.option == "clock":
            hour = int(round(math.degrees(math.atan2(lead[0], -lead[1])) / 30.0)) % 12    # 12 is up, 3 is right
            return f"{HOURS[hour]} o'clock"
        words = []
        if not self._lined[1]:
            words.append("down" if lead[1] > 0 else "up")
        if not self._lined[0]:
            words.append("right" if lead[0] > 0 else "left")
        return " and ".join(words)

    def _guide_dynamic(self, err, hand, now: float):
        cfg = self.cfg
        if self._prev_hand is not None and 0 < now - self._prev_t < 0.5:
            v = (hand - self._prev_hand) / (now - self._prev_t)
            self._vel = v if self._vel is None else 0.5 * self._vel + 0.5 * v
        else:
            self._vel = None
        self._prev_hand, self._prev_t = hand.copy(), now
        lead = err if self._vel is None else err - self._vel * cfg.lead_s

        phrase = self._phrase_for(lead)
        lined = all(self._lined)
        if lined and not self._was_lined:
            self.sink.say("aligned", now=True)           # lined up: from here it is only forward
            self._phrase, self._phrase_t = None, now
        self._was_lined = lined
        gap = cfg.word_repeat_s if phrase == self._phrase else cfg.phrase_gap_s
        if phrase and now - self._phrase_t >= gap and not self.sink.saying():
            self._phrase, self._phrase_t = phrase, now
            self.sink.speak(phrase)

        if cfg.option == "combined":
            self._tone(self._silent)
        else:
            dist = float(np.linalg.norm(err))
            closeness = 1.0 - min(1.0, max(0.0, (dist - cfg.grab_m) / (cfg.far_m - cfg.grab_m)))
            rate = self.tone.rate_far + (self.tone.rate_near - self.tone.rate_far) * closeness
            pan = 0.0 if self._lined[0] else 0.6 * math.copysign(1.0, lead[0])
            self._tone((pan, self.tone.base_hz, rate, "beep"))

    def _guide_step(self, err, now: float):
        cfg = self.cfg
        self._tone(self._silent)
        if now < self._step_pause_until or self.sink.saying():
            return
        if self._step_axis is None:
            # Sideways, then height, then depth. Skip axes already aligned.
            self._step_axis = next((a for a in (0, 1, 2) if abs(err[a]) > cfg.step_m), None)
            if self._step_axis is None:
                return
        axis = self._step_axis
        distance = abs(err[axis])
        if distance <= cfg.step_m:
            if self._step_near_t is None:
                self._step_near_t = now
            if now - self._step_near_t >= cfg.step_hold_s:
                self.sink.say("aligned", now=True)
                self._step_axis = None
                self._step_word = None
                self._step_near_t = None
                self._step_pause_until = now + cfg.step_pause_s
            return
        self._step_near_t = None
        word = (("left", "right"), ("up", "down"), ("back", "forward"))[axis][int(err[axis] > 0)]
        if word != self._step_word:
            # Announce once; repeat only if overshoot requires the opposite direction.
            self._step_word = word
            self.sink.say(word, now=True)
            return
        closeness = 1.0 - np.clip((distance - cfg.step_m) / (cfg.far_m - cfg.step_m), 0, 1)
        rate = self.tone.rate_far + (self.tone.rate_near - self.tone.rate_far) * closeness
        pan = float(np.sign(err[0])) * 0.7 if axis == 0 else 0.0
        # Fixed pitch: beep speed now measures progress on this step alone.
        self._tone((pan, self.tone.base_hz, float(rate), "beep"))

    def _speak_direction(self, err, now: float):
        # Finish one axis before switching: similar errors must not alternate commands.
        axis = self._word_axis
        if axis is None or abs(err[axis]) <= self.cfg.dead_m:
            axis = int(np.argmax(np.abs(err)))
        if abs(err[axis]) <= self.cfg.dead_m:
            return
        word = (("left", "right"), ("up", "down"), ("back", "forward"))[axis][int(err[axis] > 0)]
        gap = self.cfg.word_repeat_s if word == self._last_word else self.cfg.word_gap_s
        if now - self._word_t >= gap and not self.sink.saying():
            self._word_axis = axis
            self._last_word = word
            self._word_t = now
            self.sink.say(word)

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
            if now - self._grab_t >= cfg.grab_state_s and not self.sink.saying():
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
            self._step_near_t = None
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
                self._step_word = None  # re-announce when the hand returns after a loss
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
            self._tone(self._silent if cfg.words or cfg.hybrid or cfg.option in DYNAMIC else
                       (0.0, self.tone.steady_hz, self.tone.rate_near, "steady"))
        else:
            self._near_t = None
            if cfg.hybrid:
                self._guide_step(err, now)
            elif cfg.words:
                self._tone(self._silent)
                self._speak_direction(err, now)
            elif cfg.option in DYNAMIC:
                self._guide_dynamic(err, np.asarray(scene.hand_xyz, float), now)
            else:
                self._tone(tone_for(err, cfg, self.tone))
        return self.state


# ---------------------------------------------------------------------------
# The interface main.py uses.

_default: Guidance | None = None
_logged_state: str | None = None


def _get() -> Guidance:
    global _default
    if _default is None:
        mode = os.environ.get("GUIDANCE_MODE") or (
            "words" if os.environ.get("GUIDANCE_WORDS", "") not in ("", "0") else "tones")
        key = _option_key(mode)
        if key is None:
            raise ValueError("GUIDANCE_MODE must be tones, words, hybrid, a number 1-5, or one of: "
                             + ", ".join(k for k, _, _ in OPTIONS))
        tone = ToneConfig()
        sink = open_sink(os.environ.get("GUIDANCE_AUDIO", "auto"), os.environ.get("GUIDANCE_DEVICE") or None, tone)
        _default = Guidance(sink, GuidanceConfig(), tone)
        _default.set_option(key)
        tracing.meta(option=key)
    return _default


def _option_key(which) -> str | None:
    """A number 1-5, a key from OPTIONS, or one of the older names -> the option's key."""
    which = str(which).strip().lower()
    keys = [k for k, _, _ in OPTIONS]
    if which.isdigit() and 1 <= int(which) <= len(keys):
        return keys[int(which) - 1]
    which = {"hybrid": "steps"}.get(which, which)
    return which if which in keys + ["words"] else None


def set_option(which) -> str:
    """Switch the way of guiding while running. Returns its spoken name, e.g. "combined words"."""
    key = _option_key(which)
    if key is None:
        raise ValueError(f"no guidance option {which!r}")
    _get().set_option(key)
    keys = [k for k, _, _ in OPTIONS]
    number = keys.index(key) + 1 if key in keys else 0
    title = OPTIONS[number - 1][1] if number else key
    tracing.event("option", option=key, number=number, title=title)
    return title


def option() -> tuple[int, str, str]:
    """(number, key, name) of the option in use."""
    key = _get().cfg.option
    keys = [k for k, _, _ in OPTIONS]
    number = keys.index(key) + 1 if key in keys else 0
    return number, key, OPTIONS[number - 1][1] if number else key


def current_target() -> str | None:
    """What is being looked for right now, or None between requests."""
    g = _get()
    return g.target if g.state not in (IDLE, GRAB) else None


def update(scene: Scene) -> str:
    global _logged_state
    tracing.begin_frame(scene.t)
    with tracing.span("guidance"):
        state = _get().update(scene)
    if state != _logged_state:
        _logged_state = state
        tracing.event("state", state=state, target=_get().target, option=_get().cfg.option)
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


def say_clip(name: str, wait: bool = False) -> None:
    """Play one of the short sounds outside a reach (voice.py's "listening" beep)."""
    sink = _get().sink
    sink.say(name)
    if wait:
        sink.wait_quiet()


def pop_phrases() -> list:
    """Everything said since the last call, as words. main.py shows these on the dashboard."""
    log = _get().sink.log
    out = list(log)
    del log[:len(out)]
    return out
