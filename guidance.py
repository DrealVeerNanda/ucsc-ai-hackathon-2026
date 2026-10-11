"""Lane 3: turns a Scene into what the wearer hears, and decides when to say "grab".

    state = guidance.update(scene)     # every frame; returns the state name
    guidance.reset(target)             # after every new target

Sound mapping (the error is object minus hand, in metres, camera axes x right, y down, z forward):

    left / right     the tone pans toward the side the hand should move to
    up / down        pitch rises when the hand should go up, falls for down
    total distance   beeps get faster as the hand gets closer
    in grabbing distance   steady tone, then the "grab" clip (see GRAB_OPTIONS for what counts)

States: idle, searching, waiting_hand, guiding, grab. After grab it returns to idle, which
is main.py's cue to ask for the next target, unless the hand is plainly seen to have come
away without the object: then it guides the same reach again.

Settings by environment variable:
    GUIDANCE_AUDIO=auto|on|off   sound card if there is one (default), required, or print only
    GUIDANCE_DEVICE=NAME         sounddevice output device (earbuds)
    GUIDANCE_WORDS=1             fallback: spoken left/right/up/down/forward instead of tones
    GUIDANCE_MODE=tones|words|hybrid  explicit mode, overrides GUIDANCE_WORDS
    GUIDANCE_MODE=1..12 or a name     one of OPTIONS below, to compare ways of guiding. It can be
                                      changed while running: say or type "option 3".
    GUIDANCE_GRAB=1..5 or a name      one of GRAB_OPTIONS below: when "Grab" is said. Also changeable
                                      while running: "grab option 2".
"""
from __future__ import annotations

import math
import os
import time
from collections import deque
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
    ("numbers", "measured words", "How far as well as which way, in centimetres: 'right twenty, up ten'. One "
                                  "instruction for each move, a chime when lined up, then 'forward fifteen'."),
    ("echo", "ping and echo", "Two notes. The first is your hand, the second is the object: it sounds on the side "
                              "the object is on, higher if it is above you, and it follows sooner the closer you "
                              "are. Move until the two are the same."),
    ("hotcold", "warmer and colder", "One steady note that rises as your fingertip gets closer and falls when it "
                                     "moves away, in the ear on the side to move to. Make the note go up."),
    ("spearcons", "fast words", "The direction as one very short word, said again and again like a parking "
                                "sensor: the faster it repeats, the closer you are."),
    ("beam", "stay on the beam", "Words to get lined up, then quiet while you stay on the line. Ticks only when "
                                 "you drift: in the ear on the side to come back to, high to go up, low to go down."),
    ("twostage", "words then a note", "Measured words for the big move, then the rising note of 'warmer and "
                                      "colder' for the last hand's width."),
    ("detector", "metal detector", "A hum that rises as your hand lines up with the object: sweep to find its "
                                   "highest point. On it, the hum turns into beeps that speed up as you push forward."),
]
DYNAMIC = ("combined", "combined_beeps", "clock")
TONAL = ("tones", "echo", "hotcold", "twostage", "detector")     # the tone is the guidance: full volume, and a steady note on arrival
HOURS = ("twelve", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven")
OPPOSITE = {"left": "right", "right": "left", "up": "down", "down": "up", "forward": "back", "back": "forward"}
AMOUNTS = {5: "five", 10: "ten", 15: "fifteen", 20: "twenty", 25: "twenty five", 30: "thirty", 40: "forty",
           50: "fifty", 60: "sixty"}

# When to say "Grab". (key, name said aloud, what it means)
GRAB_OPTIONS = [
    ("fingertip", "fingertip touch", "The first rule: the index fingertip within 5 cm of the object, in every "
                                     "frame for a third of a second."),
    ("zone", "grabbing distance", "The fingertip within grabbing distance of the nearest part of the object: 6 cm "
                                  "to the side, 10 cm in depth, where the measurement is rougher. Blinks of the "
                                  "hand tracker do not restart the count, and readings no hand could make are "
                                  "left out."),
    ("over", "over the object", "The fingertip over the object in the picture, or within a quarter of its width "
                                "of it, and within 10 cm of it in depth for a third of a second, or up to 15 cm "
                                "short for most of a second: what a hand resting on the object measures."),
    ("arrive", "arrive or cover", "Grabbing distance, and also when a hand that was closing in from a hand's "
                                  "width away disappears at the object: wrapped around it, the tracker loses it."),
    ("near", "wide circle", "The first rule with the circle widened to 9 cm, to compare."),
]
PATIENT = ("zone", "over", "arrive")     # rules that sit through blinks and refuse what a hand cannot do


@dataclass
class GuidanceConfig:
    grab_m: float = 0.05         # "fingertip" rule: closer than this, steady tone, then "grab"
    grab_hold_s: float = 0.3     # ...for this long, so one noisy frame cannot trigger it
    grab_state_s: float = 1.5    # stay in grab this long, then go idle
    grab_option: str = "over"    # which of GRAB_OPTIONS is in use
    # Grabbing distance. "Grab" is heard about a second after the hand gets there (0.3 s of
    # watching, then the chime), and a hand homing in at 5 to 10 cm/s covers that much meanwhile;
    # fingers then close through another 7 cm. Depth is the rough number: in the run of 10 Oct a
    # hand wrapped round the bottle measured 6 to 15 cm short of it for seconds (the camera sees
    # the back of the hand, and lidar reads low that close), while a hand truly out of reach,
    # over the bottle in the picture, measured 23 cm short and more.
    grab_side_m: float = 0.06    # "zone": this far across the camera's view, where pixels are exact
    grab_depth_m: float = 0.10   # ...and this far along it
    over_short_m: float = 0.15   # "over": the hand may measure this far short of the object...
    over_past_m: float = 0.08    # ...or this far past its front...
    over_hold_s: float = 0.8     # ...if it stays this long: a hand still on its way in is through that band sooner
    depth_jump_m: float = 0.12   # the hand's depth changing this much in a tenth of a second is lidar reading past
    doubt_s: float = 0.5         # ...the finger onto what is behind it: not believed for this long
    goal_jump_m: float = 0.10    # the object moving this much in one frame was measured on something else.
    landed_m: float = 0.12       # ...If that puts it this close to the hand, it was measured on the hand:
    apart_m: float = 0.15        # no "grab" until the two have been seen this far apart again. See _trust
    grab_near_m: float = 0.09    # the "near" rule's radius
    grab_slack: float = 1.3      # once inside, still counted as inside out to this many times the zone
    grab_forgive_s: float = 0.15  # outside for less than this is a bad frame, not a hand that left
    grab_max_mps: float = 0.30   # a hand crossing the view faster than this is passing, not arriving
    grab_box_grow: float = 0.25  # "over": this fraction of the object's width around its box still counts
    arrive_gone_s: float = 0.25  # "arrive" rule: the hand unseen this long, having been...
    arrive_zone: float = 1.3     # ...within this many times the zone, and closing in: it is at the object
    miss_m: float = 0.25         # after "grab": hand seen this far from an object that is seen still there...
    miss_s: float = 0.4          # ...for this long: it came away empty, guide again
    object_blink_s: float = 0.4  # object gone for less than this: the detector blinked, say nothing
    dead_m: float = 0.03         # per-axis error inside this is treated as zero
    pan_full_m: float = 0.20     # sideways error that pans fully to one ear
    pitch_full_m: float = 0.20   # vertical error that shifts the pitch by a full octave
    far_m: float = 0.50          # at this distance or more, beeps are at their slowest
    hold_s: float = 0.5          # hand missing for less than this: keep the last sound
    lost_s: float = 1.0          # hand missing for longer than this: back to waiting_hand
    remind_s: float = 6.0        # repeat "show me your hand" this often while waiting
    cant_see_s: float = 8.0      # searching for this long: say "I can't see that"
    give_up_s: float = 20.0      # searching this long without ever seeing it: go idle, so the wearer is asked
                                 # again. Misheard or simply not there, they could otherwise say nothing more:
                                 # the microphone is only open while a request is awaited. 0 = search for ever
    hand_give_up_s: float = 30.0 # waiting for a hand this long: go idle too, for the same reason. 0 = wait for ever
    far_hold_s: float = 2.5      # the object has been out of arm's reach this long: say so. Not sooner: a clear
                                 # bottle entering the picture reads as the wall behind it for a moment
    far_repeat_s: float = 8.0    # ...and again this often while it stays there
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
    vel_window_s: float = 0.25   # the hand's speed is measured over this long, so tracking jitter does not pass for motion
    vel_min_s: float = 0.09      # ...and not at all until the hand has been watched this long
    still_mps: float = 0.06      # slower than this counts as holding still
    line_in_m: float = 0.025     # sideways or vertical error below this: lined up...
    line_out_m: float = 0.05     # ...and stays lined up until it grows past this, so it cannot flip-flop
    rechime_s: float = 2.0       # back on the line after less than this off it: no second chime
    reach_m: float = 0.03        # lined up, and the object is this much further away: "forward"
    back_m: float = 0.07         # only say "back" when the hand is clearly past the object
    phrase_gap_s: float = 0.7    # minimum pause before a different phrase
    breath_s: float = 0.3        # ...and this much silence after the last one ended
    reverse_s: float = 1.5       # "left" this soon after "right" is jitter, unless...
    reverse_m: float = 0.08      # ...the hand is this far past the line: then it really overshot
    progress_m: float = 0.03     # a hand this much nearer than when it was told is following: no reminder
    rest_s: float = 0.3          # "numbers": the hand has finished its move when it has been still this long
    stall_s: float = 4.0         # told "forward" this long with the hand no deeper: the object is out of reach,
    stall_m: float = 0.12        # ...if it is still this much further away. Say "lean forward" instead
    beam_m: float = 0.12         # "beam": drift up to this far is corrected with ticks, beyond it with words
    fine_m: float = 0.12         # "twostage": closer than this, the note takes over from the words
    sweep_m: float = 0.25        # "detector": the hum starts to rise this far off the line
    hot_low_hz: float = 220.0    # "hotcold": the note when far; two octaves higher on arrival


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
        self.trace = {}              # this frame's numbers for the trace: see update() at the bottom
        self._silent = (0.0, self.tone.base_hz, self.tone.rate_far, "silent")
        self._last_tone = self._silent
        self._clear(0.0)

    def _clear(self, now: float):
        self._had_object = False
        self._search_t = now
        self._said_cant_see = False
        self._far_t, self._far_said_t = None, -1e9
        self._object_gone_t = None
        self._last_hand_t = None
        self._unseen_t = None        # since when the hand has been out of sight
        self._waiting_t = now        # since when it has been waiting for a hand
        self._remind_t = None
        self._near_t = None          # since when the hand has been in grabbing distance
        self._out_t = None           # ...and since when it has been out of it again
        self._wide_t = None          # ...and when it was last inside by the wide depth limit only
        self._wide = False
        self._zone_log = deque(maxlen=40)    # (time, zone distance) of the last frames with a hand
        self._before = None          # (time, object position, hand depth) of the frame before
        self._doubt_t = -1e9         # when the hand's depth last jumped
        self._depth = None           # (time, depth still to go) when it was last believed
        self._unearned = False       # the object jumped onto the hand, and has not been seen apart from it since
        self._grab_t = None
        self._miss_t = None
        self._word_t = -1e9
        self._last_word = None
        self._word_axis = None
        self._step_axis = None
        self._step_word = None
        self._step_near_t = None
        self._step_pause_until = 0.0
        self._track = []             # (time, hand position) over the last moments, for its speed
        self._vel = None
        self._speed = None           # across the camera's view, m/s; None until it can be measured
        self._lined = [False, False]
        self._was_lined = False
        self._lined_t = None
        self._phrase = None
        self._phrase_t = -1e9
        self._phrase_lead = None     # where the hand had to go when the phrase was said
        self._heard_t = -1e9         # when speech was last playing
        self._word_times = {}        # direction word -> when it was last spoken
        self._stall = None           # (time, hand depth) since which "forward" has brought no progress
        self._moved = False          # has the hand moved since the last measured instruction?
        self._still_t = None         # ...and since when has it been holding still
        self._on_beam = False
        self._fine = False
        self._smooth = None          # (time, value): the distance behind the note, without its jitter
        self._fast_t = -1e9
        self._fast_i = 0

    def set_option(self, key: str):
        """Change how guidance sounds. Safe in the middle of a reach."""
        if key not in [k for k, _, _ in OPTIONS] + ["words"]:
            raise ValueError(f"unknown guidance option {key!r}")
        self.cfg.option = key
        self.cfg.hybrid, self.cfg.words = key == "steps", key == "words"
        # Beeps that are the guidance are loud, beeps under speech quiet, and a note that never stops quieter still.
        self.tone.volume = 0.14 if key in ("hotcold", "twostage", "detector") else (0.25 if key in TONAL else 0.16)
        self._step_axis = self._step_word = self._phrase = None
        self._lined, self._was_lined, self._lined_t = [False, False], False, None
        self._on_beam = self._fine = self._moved = False
        self._smooth = self._stall = None
        self._tone(self._silent)

    def set_grab_option(self, key: str):
        """Change the rule for saying "Grab". Safe in the middle of a reach."""
        if key not in [k for k, _, _ in GRAB_OPTIONS]:
            raise ValueError(f"unknown grab option {key!r}")
        self.cfg.grab_option = key
        self._near_t = self._out_t = None
        self._zone_log.clear()

    # --- what the hand is doing ---------------------------------------------------

    def _watch(self, hand, now: float):
        """The hand's speed, from where it was a quarter of a second ago rather than one frame
        ago: 6 mm of tracking jitter between two frames reads as 0.25 m/s, which the lead
        would turn into 5 cm of error, wider than the whole lined-up band."""
        cfg = self.cfg
        if self._track and now - self._track[-1][0] >= 0.5:
            self._track = []                              # the hand was away: its old path says nothing
        self._track = [(t, h) for t, h in self._track if now - t <= cfg.vel_window_s] + [(now, hand.copy())]
        t0, h0 = self._track[0]
        self._vel = (hand - h0) / (now - t0) if now - t0 >= cfg.vel_min_s else None
        # Across the view only: the depth flips between lidar and the size estimate, 4 cm at a time.
        self._speed = None if self._vel is None else float(np.hypot(self._vel[0], self._vel[1]))
        if self._vel is not None and float(np.linalg.norm(self._vel)) < cfg.still_mps:
            self._vel = None                              # a hand held still is where it is: nothing to anticipate

    def _line(self, lead) -> bool:
        """Is the hand lined up with the object, sideways and in height? Each axis stays lined
        up until it clearly drifts, so it cannot flip-flop at the edge."""
        cfg = self.cfg
        for a in (0, 1):
            self._lined[a] = abs(lead[a]) < (cfg.line_out_m if self._lined[a] else cfg.line_in_m)
        return all(self._lined)

    def _chime(self, now: float) -> bool:
        """The lined-up chime, once. Returns whether the hand is lined up."""
        lined = all(self._lined)
        if lined and not self._was_lined:
            # A real hand trembles on and off the line several times a second. The chime is for
            # lining up after a clear spell away, not for each of those returns.
            if self._lined_t is None or now - self._lined_t >= self.cfg.rechime_s:
                self.sink.say("aligned", now=True)       # lined up: from here it is only forward
                self._phrase, self._phrase_t = None, now
        if lined:
            self._lined_t = now
        self._was_lined = lined
        return lined

    def _stalled(self, lead, hand, now: float) -> bool:
        """Told "forward" for seconds and the hand is no deeper: it is at the end of the arm."""
        cfg = self.cfg
        z = float(np.median([h[2] for _, h in self._track])) if self._track else float(hand[2])
        if lead[2] < cfg.stall_m or self._stall is None or z - self._stall[1] >= 0.05:
            self._stall = (now, z)
        return now - self._stall[0] >= cfg.stall_s

    def _phrase_for(self, lead, hand=None, now: float = 0.0, amounts=None) -> str | None:
        """What to say for this error, for the options that give the whole direction at once.
        With `amounts` (the measured error) each word is followed by how far, in centimetres."""
        cfg = self.cfg
        how_far = (lambda a: "") if amounts is None else (lambda a: " " + _amount(amounts[a]))
        if self._line(lead):
            if lead[2] > cfg.reach_m:
                if hand is not None and self._stalled(lead, hand, now):
                    return "lean forward"
                return "forward" + how_far(2)
            self._stall = None
            return "back" + how_far(2) if lead[2] < -cfg.back_m else None
        self._stall = None
        if cfg.option == "clock":
            hour = int(round(math.degrees(math.atan2(lead[0], -lead[1])) / 30.0)) % 12    # 12 is up, 3 is right
            return f"{HOURS[hour]} o'clock"
        words = []
        if not self._lined[1]:
            words.append(("down" if lead[1] > 0 else "up") + how_far(1))
        if not self._lined[0]:
            words.append(("right" if lead[0] > 0 else "left") + how_far(0))
        return (", " if amounts is not None else " and ").join(words)

    def _speak(self, phrase: str | None, now: float, lead=None, repeat_s: float | None = None) -> bool:
        """Say a direction if it is time to: not over other speech, not straight after it, not
        the same thing again too soon, and not the opposite of what was just said unless the
        hand has clearly gone past."""
        cfg = self.cfg
        if not phrase or self.sink.saying():
            return False
        # "lean forward" is "forward" said another way: it takes the reminder's turn, not a turn of its own.
        same = self._phrase is not None and phrase.replace("lean ", "") == self._phrase.replace("lean ", "")
        if now - self._phrase_t < ((repeat_s or cfg.word_repeat_s) if same else cfg.phrase_gap_s):
            return False
        words = [w for w in phrase.replace(",", " ").split() if w in OPPOSITE]
        axes = sorted({("left", "right", "up", "down", "back", "forward").index(w) // 2 for w in words}) or [0, 1]
        if same and lead is not None and self._phrase_lead is not None and phrase == self._phrase:
            # A reminder is for a hand that is not getting anywhere. One that is following the
            # last instruction is left to it.
            gained = [abs(self._phrase_lead[a]) - abs(lead[a]) for a in axes]
            if max(gained) >= cfg.progress_m and min(gained) > -cfg.progress_m:
                self._phrase_t, self._phrase_lead = now, np.array(lead, float)
                return False
        if not same:
            if now - self._heard_t < cfg.breath_s:
                return False
            for w in words:
                axis = ("left", "right", "up", "down", "back", "forward").index(w) // 2
                small = lead is not None and abs(lead[axis]) < cfg.reverse_m
                if small and now - self._word_times.get(OPPOSITE[w], -1e9) < cfg.reverse_s:
                    return False
        self._phrase, self._phrase_t = phrase, now
        self._phrase_lead = None if lead is None else np.array(lead, float)
        self._moved = False
        for w in words:
            self._word_times[w] = now
        self.sink.speak(phrase)
        return True

    def _eased(self, value: float, now: float, tau: float = 0.15) -> float:
        """The value with its frame-to-frame jitter taken out, so a note made from it does not warble."""
        if self._smooth is None or now - self._smooth[0] > 0.5:
            self._smooth = (now, value)
        else:
            t, v = self._smooth
            self._smooth = (now, v + (value - v) * (1.0 - math.exp(-(now - t) / tau)))
        return self._smooth[1]

    def _closeness(self, dist: float, near: float | None = None) -> float:
        """0 at far_m or more, 1 at grabbing distance."""
        near = self.cfg.grab_m if near is None else near
        return 1.0 - min(1.0, max(0.0, (dist - near) / (self.cfg.far_m - near)))

    def _rate(self, closeness: float) -> float:
        return self.tone.rate_far + (self.tone.rate_near - self.tone.rate_far) * closeness

    # --- the ways of guiding ----------------------------------------------------------

    def _guide_dynamic(self, err, lead, hand, now: float):
        cfg = self.cfg
        phrase = self._phrase_for(lead, hand, now)
        self._chime(now)
        self._speak(phrase, now, lead)
        if cfg.option == "combined":
            self._tone(self._silent)
        else:
            rate = self._rate(self._closeness(float(np.linalg.norm(err))))
            pan = 0.0 if self._lined[0] else 0.6 * math.copysign(1.0, lead[0])
            self._tone((pan, self.tone.base_hz, rate, "beep"))

    def _guide_numbers(self, err, lead, hand, now: float):
        """One measured instruction for each move: said, then nothing until the hand has made
        its move and stopped, or has not moved at all for a while."""
        cfg = self.cfg
        phrase = self._phrase_for(lead, hand, now, amounts=err)
        self._chime(now)
        self._tone(self._silent)
        if self._speed is not None and self._speed >= cfg.still_mps + 0.02:
            self._moved = True
        self._still_t = (self._still_t if self._still_t is not None else now) if self._vel is None else None
        rested = self._moved and self._still_t is not None and now - self._still_t >= cfg.rest_s
        # The reminder counts from the end of the phrase: a measured pair takes a second and a half to say.
        if self._phrase is None or rested or now - max(self._phrase_t, self._heard_t) >= cfg.word_repeat_s:
            if phrase == self._phrase:
                self._phrase = None                      # the same words after a move are news: it fell short
            self._speak(phrase, now, lead)

    def _guide_echo(self, err, lead, now: float):
        cfg = self.cfg
        pan = _dead(lead[0], cfg.dead_m, cfg.pan_full_m)
        up = _dead(-lead[1], cfg.dead_m, cfg.pitch_full_m)
        far = 1.0 - self._closeness(self._eased(float(np.linalg.norm(err)), now))
        gap = 0.10 + 0.32 * far                           # the echo follows sooner as the hand closes in
        self._tone((pan, self.tone.base_hz * 2.0 ** (up * self.tone.octaves), 1.0 / (gap + 0.28), "echo", gap))

    def _hot_note(self, err, lead, now: float):
        cfg = self.cfg
        warm = self._closeness(self._eased(float(np.linalg.norm(err)), now), cfg.grab_side_m)
        pan = 0.8 * _dead(lead[0], cfg.dead_m, cfg.pan_full_m)
        self._tone((pan, cfg.hot_low_hz * 2.0 ** (2.0 * warm), self.tone.rate_near, "steady"))

    def _guide_spearcons(self, err, lead, now: float):
        cfg = self.cfg
        self._tone(self._silent)
        if self._chime_line(lead, now):
            words = ["forward"] if lead[2] > cfg.reach_m else (["back"] if lead[2] < -cfg.back_m else [])
        else:
            words = ([] if self._lined[1] else ["down" if lead[1] > 0 else "up"]) + \
                    ([] if self._lined[0] else ["right" if lead[0] > 0 else "left"])
        every = 0.95 - 0.55 * self._closeness(float(np.linalg.norm(err)))
        if words and now - self._fast_t >= every and not self.sink.saying():
            self._fast_t = now
            self._fast_i += 1
            self.sink.say("fast_" + words[self._fast_i % len(words)])      # a diagonal alternates its two words

    def _chime_line(self, lead, now: float) -> bool:
        self._line(lead)
        return self._chime(now)

    def _guide_beam(self, err, lead, hand, now: float):
        """Like a radio beam for landing: quiet on course, a sound only for being off it."""
        cfg = self.cfg
        phrase = self._phrase_for(lead, hand, now)
        off = float(np.hypot(lead[0], lead[1]))
        if self._chime(now):
            self._on_beam = True
            self._tone(self._silent)
            self._speak(phrase, now, lead, repeat_s=4.0)
        elif self._on_beam and off < cfg.beam_m:
            pan = 0.0 if self._lined[0] else 0.9 * math.copysign(1.0, lead[0])
            octave = 0.0 if self._lined[1] else (-1.0 if lead[1] > 0 else 1.0)       # low: go down. high: go up
            self._tone((pan, self.tone.base_hz * 2.0 ** octave, 3.0 + 6.0 * off / cfg.beam_m, "beep"))
        else:
            self._on_beam = False
            self._tone(self._silent)
            self._speak(phrase, now, lead)

    def _guide_twostage(self, err, lead, hand, now: float):
        cfg = self.cfg
        dist = float(np.linalg.norm(err))
        self._fine = dist < (1.5 * cfg.fine_m if self._fine else cfg.fine_m)
        if self._fine:
            self._line(lead)
            self._was_lined = all(self._lined)
            self._hot_note(err, lead, now)
        else:
            self._guide_numbers(err, lead, hand, now)

    def _guide_detector(self, err, lead, now: float):
        cfg = self.cfg
        off = self._eased(float(np.hypot(lead[0], lead[1])), now)
        if self._line(lead):
            ahead = self._closeness(max(0.0, float(err[2])), cfg.reach_m)
            self._tone((0.0, 2 * self.tone.base_hz, self._rate(ahead), "beep"))
        else:
            on = 1.0 - min(1.0, off / cfg.sweep_m)
            pan = 0.8 * _dead(lead[0], cfg.dead_m, cfg.pan_full_m)
            self._tone((pan, 196.0 * 2.0 ** (2.0 * on), self.tone.rate_near, "steady"))

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

    def _guide(self, err, lead, hand, now: float):
        cfg = self.cfg
        if cfg.hybrid:
            self._guide_step(err, now)
        elif cfg.words:
            self._tone(self._silent)
            self._speak_direction(err, now)
        elif cfg.option in DYNAMIC:
            self._guide_dynamic(err, lead, hand, now)
        elif cfg.option == "numbers":
            self._guide_numbers(err, lead, hand, now)
        elif cfg.option == "echo":
            self._guide_echo(err, lead, now)
        elif cfg.option == "hotcold":
            self._hot_note(err, lead, now)
        elif cfg.option == "spearcons":
            self._guide_spearcons(err, lead, now)
        elif cfg.option == "beam":
            self._guide_beam(err, lead, hand, now)
        elif cfg.option == "twostage":
            self._guide_twostage(err, lead, hand, now)
        elif cfg.option == "detector":
            self._guide_detector(err, lead, now)
        else:
            self._tone(tone_for(err, cfg, self.tone))

    def _tone(self, params):
        self._last_tone = params
        self.sink.set_tone(*params)

    # --- "grab" -------------------------------------------------------------------------

    def _zone(self, scene: Scene, err) -> float:
        """How far the fingertip is from grabbing distance, as a fraction: under 1 is inside."""
        cfg = self.cfg
        key = cfg.grab_option
        self._wide = False           # inside only by the wider depth limit of "over"
        if key in ("fingertip", "near"):
            return float(np.linalg.norm(err)) / (cfg.grab_m if key == "fingertip" else cfg.grab_near_m)
        side = float(np.hypot(err[0], err[1])) / cfg.grab_side_m
        if key != "zone" and scene.hand_px is not None and scene.object_box is not None:
            # In the picture a centimetre is a centimetre whatever the depth says, so with the
            # fingertip seen over the object the depth only has to be roughly right.
            x1, y1, x2, y2 = scene.object_box
            u, v = scene.hand_px
            outside = max(x1 - u, u - x2, y1 - v, v - y2, 0.0)
            side = outside / max(8.0, cfg.grab_box_grow * min(x2 - x1, y2 - y1))
            self._wide = float(np.hypot(side, err[2] / (cfg.over_short_m if err[2] > 0 else cfg.over_past_m))) < 1.0
        return float(np.hypot(side, err[2] / cfg.grab_depth_m))

    def _trust(self, scene: Scene, hand, err, now: float) -> bool:
        """Can this frame count towards "Grab"? Not if it shows something a hand or a bottle
        cannot do. Both happened in the run of 10 Oct, and both said "the hand is on it":

        The hand's depth jumped 20 cm in one frame: lidar had read past the finger onto the
        object behind it, for a third of a second at a time.

        The object jumped onto the hand: with the hand in front of it and the tracker blinking,
        its depth was measured on the hand. An arrival is the hand moving, never the object, so
        after such a jump the two have to be seen clearly apart before the hand can arrive."""
        cfg = self.cfg
        before, self._before = self._before, (now, np.asarray(scene.object_xyz, float).copy(), float(hand[2]))
        dist = float(np.linalg.norm(err))
        # Against every sighting of the last tenth of a second, not only the last one: the
        # camera's own smoothing spreads a jump over three frames.
        if any(now - t <= 0.12 and abs(hand[2] - h[2]) > cfg.depth_jump_m for t, h in self._track[:-1]):
            self._doubt_t = now
        # Compared with the last frame that had a hand in it: the object jumps while the tracker blinks.
        if before is not None and now - before[0] < 1.0 and \
                float(np.linalg.norm(self._before[1] - before[1])) > cfg.goal_jump_m:
            self._unearned = dist < cfg.landed_m
        believed = now - self._doubt_t >= cfg.doubt_s
        if believed and dist >= cfg.apart_m:
            self._unearned = False
        return believed and not self._unearned

    def _arrived(self, zone: float, trusted: bool, now: float) -> bool:
        """Has the hand been in grabbing distance long enough to say so?"""
        cfg = self.cfg
        patient = cfg.grab_option in PATIENT
        if patient and not trusted:
            self._near_t = self._out_t = None
            self._zone_log.clear()
            return False
        forgive = cfg.grab_forgive_s if patient else 0.0
        slack = cfg.grab_slack
        self._zone_log.append((now, zone))
        inside = zone < 1.0 or self._wide
        if inside or (self._near_t is not None and zone < slack):
            # Once inside, allow a little slack before counting it as outside again.
            if self._out_t is not None and now - self._out_t >= forgive:
                self._near_t = None                      # it was away for longer than a bad frame lasts
            self._out_t = None
            if self._near_t is None and inside:
                self._near_t = now
            if zone >= 1.0:
                self._wide_t = now
        elif self._near_t is not None:
            if self._out_t is None:
                self._out_t = now
            if now - self._out_t >= forgive:
                self._near_t = self._out_t = None
        passing = patient and self._speed is not None and self._speed > cfg.grab_max_mps
        if self._near_t is None or self._out_t is not None or passing:
            return False
        # Inside by the wide depth limit only: that takes longer to believe.
        surely = self._near_t if self._wide_t is None else max(self._near_t, self._wide_t)
        return now - surely >= cfg.grab_hold_s or now - self._near_t >= cfg.over_hold_s

    def _covered(self, now: float) -> bool:
        """The hand has gone from view. Was it closing in on the object, a hand's width away?
        Then it has not left: it is on the object, where the tracker cannot make it out."""
        cfg = self.cfg
        log = [(t, z) for t, z in self._zone_log if self._last_hand_t - t <= 0.5]
        if cfg.grab_option != "arrive" or len(log) < 4 or log[-1][0] - log[0][0] < 0.1:
            return False
        if self._speed is not None and self._speed > cfg.grab_max_mps:
            return False                                 # it was crossing the view: it may simply have left it
        last = max(z for _, z in log[-3:])               # every one of its last sightings, not most of them
        closing = self._near_t is not None or last < float(np.median([z for _, z in log[:3]])) - 0.2
        return closing and last < cfg.arrive_zone and now - self._last_hand_t >= cfg.arrive_gone_s

    def _say_grab(self, now: float) -> str:
        self.state = GRAB
        self._grab_t = now
        self._miss_t = None
        self._tone(self._silent)
        self.sink.say("grab", now=True)
        return self.state

    def _after_grab(self, scene: Scene, now: float) -> str:
        """Idle when "grab" has been said, which is main.py's cue to ask what next. Unless the
        hand is seen well away from an object that is seen still standing there: it came away
        empty, so the same reach is guided again and nobody has to ask twice."""
        cfg = self.cfg
        self._tone(self._silent)
        err = hand_error(scene)
        away = err is not None and scene.object_visible and float(np.linalg.norm(err)) > cfg.miss_m
        self._miss_t = (self._miss_t if self._miss_t is not None else now) if away else None
        # "Grab" takes about a second to say. A sink that still claims to be speaking long after
        # is not being played (the sound card has gone): move on rather than wait for ever.
        if now - self._grab_t >= cfg.grab_state_s and (not self.sink.saying() or now - self._grab_t >= 4 * cfg.grab_state_s):
            if self._miss_t is not None and now - self._miss_t >= cfg.miss_s:
                self._near_t = self._out_t = self._phrase = None
                self._zone_log.clear()
                self.state = GUIDING
                self.sink.speak("Try again")
            else:
                self.state = IDLE
        return self.state

    # --- every frame ----------------------------------------------------------------------

    def reset(self, target: str, now: float | None = None):
        now = time.monotonic() if now is None else now
        self.target = target
        self.state = SEARCHING
        self._clear(now)
        self._tone(self._silent)
        self.sink.say("looking", now=True)

    def _too_far(self, scene: Scene, now: float):
        """No direction brings a hand to something farther away than the arm is long: in one
        live reach the bottle stood 80 cm off and "forward" was said seven times. Say what the
        wearer can do about it, once it has been so for a second, and again now and then."""
        cfg = self.cfg
        if not getattr(scene, "object_out_of_reach", False):
            self._far_t = None
            return
        if self._far_t is None:
            self._far_t = now
        if now - self._far_t >= cfg.far_hold_s and now - self._far_said_t >= cfg.far_repeat_s and not self.sink.saying():
            self._far_said_t = now
            self.sink.speak("It is too far to reach. Move closer.")

    def update(self, scene: Scene, now: float | None = None) -> str:
        now = time.monotonic() if now is None else now
        cfg = self.cfg
        self.trace = {}
        if self.sink.saying():
            self._heard_t = now
        if self.state == IDLE:
            self._tone(self._silent)
            return self.state
        if self.state == GRAB:
            return self._after_grab(scene, now)

        if scene.object_xyz is None:
            if self._had_object:
                # The detector misses the odd frame just as its memory of the object runs out,
                # and has it again in the next one: that is not worth "I lost it", "Found it".
                if self._object_gone_t is None:
                    self._object_gone_t = now
                if now - self._object_gone_t < cfg.object_blink_s:
                    self._tone(self._silent)
                    return self.state
                self._clear(now)
                self.sink.say("i_lost_it", now=True)
            elif not self._said_cant_see and now - self._search_t > cfg.cant_see_s:
                self._said_cant_see = True
                self.sink.say("i_cant_see_that")
            self.state = SEARCHING
            self._tone(self._silent)
            if cfg.give_up_s and not self._had_object and now - self._search_t > cfg.give_up_s and not self.sink.saying():
                self.state = IDLE
            return self.state
        self._object_gone_t = None
        if not self._had_object:
            self._had_object = True
            self.sink.say("found_it", now=True)
        self._too_far(scene, now)

        patient = cfg.grab_option in PATIENT
        if scene.hand_xyz is None:
            self._unseen_t = now if self._unseen_t is None else self._unseen_t
            self._step_near_t = None
            missing = math.inf if self._last_hand_t is None else now - self._last_hand_t
            if not patient or missing >= cfg.hold_s:
                self._near_t = self._out_t = None        # the patient rules sit through a blink
            if self.state == GUIDING and missing < cfg.lost_s and self._covered(now):
                return self._say_grab(now)
            if self.state == GUIDING and missing < cfg.hold_s:
                self._tone(self._last_tone)          # a blink: keep the last sound
            elif self.state == GUIDING and missing < cfg.lost_s:
                self._tone(self._silent)             # too old to steer by, not yet given up
            else:
                if self.state != WAITING_HAND:
                    self._waiting_t = now
                elif cfg.hand_give_up_s and now - self._waiting_t > cfg.hand_give_up_s and not self.sink.saying():
                    # Nobody is reaching. The object may be one the detector made up for a word
                    # it misheard, and "show me your hand" for ever leaves the wearer no way to
                    # say so: go idle, so they are asked what they want again.
                    self.state = IDLE
                    self._tone(self._silent)
                    return self.state
                if self.state != WAITING_HAND or now - self._remind_t >= cfg.remind_s:
                    self._remind_t = now
                    self.sink.say("show_me_your_hand")
                self.state = WAITING_HAND
                self._step_word = None  # re-announce when the hand returns after a loss
                self._tone(self._silent)
            return self.state

        if self.state == WAITING_HAND:
            self.sink.cancel("show_me_your_hand")        # still waiting its turn, and no longer true
        if self._unseen_t is not None and self._near_t is not None:
            self._near_t += now - self._unseen_t         # time unseen is not time spent there
        self._unseen_t = None
        self._last_hand_t = now
        self.state = GUIDING
        err = hand_error(scene)
        hand = np.asarray(scene.hand_xyz, float)
        self._watch(hand, now)
        zone = self._zone(scene, err)
        trusted = self._trust(scene, hand, err, now)
        arrived = self._arrived(zone, trusted, now)
        vel, measured = self._vel, err
        if now - self._doubt_t < cfg.doubt_s and self._depth is not None and now - self._depth[0] < 1.0:
            # A depth that has just jumped is not steered by either: "back" for a hand that
            # never moved is worse than saying nothing new for half a second.
            err = np.array([err[0], err[1], self._depth[1]])
            vel = None if vel is None else np.array([vel[0], vel[1], 0.0])
        else:
            self._depth = (now, float(err[2]))
        lead = err if vel is None else err - vel * cfg.lead_s
        self.trace = {"err": [int(round(1000 * v)) for v in measured], "lead": [int(round(1000 * v)) for v in lead],
                      "zone": round(zone, 2), "trusted": int(trusted),
                      "near_s": None if self._near_t is None else round(now - self._near_t, 2),
                      "speed": None if self._speed is None else round(self._speed, 2)}
        if arrived:
            return self._say_grab(now)
        if self._near_t is not None and self._out_t is None:
            tonal = cfg.option in TONAL and not (cfg.words or cfg.hybrid)
            self._tone((0.0, self.tone.steady_hz, self.tone.rate_near, "steady") if tonal else self._silent)
        else:
            self._guide(err, lead, hand, now)
        self.trace["lined"] = int(self._lined[0]) + 2 * int(self._lined[1])
        return self.state


def _amount(metres: float) -> str:
    """A distance as spoken centimetres, in steps a hand can actually make blind: fives up to
    thirty, then tens."""
    cm = abs(metres) * 100
    step = 5 if cm < 32.5 else 10
    return AMOUNTS[int(min(60, max(5, step * round(cm / step))))]


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
            raise ValueError(f"GUIDANCE_MODE must be tones, words, hybrid, a number 1-{len(OPTIONS)}, or one of: "
                             + ", ".join(k for k, _, _ in OPTIONS))
        grab = _grab_key(os.environ.get("GUIDANCE_GRAB") or GuidanceConfig.grab_option)
        if grab is None:
            raise ValueError(f"GUIDANCE_GRAB must be a number 1-{len(GRAB_OPTIONS)} or one of: "
                             + ", ".join(k for k, _, _ in GRAB_OPTIONS))
        tone = ToneConfig()
        sink = open_sink(os.environ.get("GUIDANCE_AUDIO", "auto"), os.environ.get("GUIDANCE_DEVICE") or None, tone)
        cfg = GuidanceConfig()
        if os.environ.get("GUIDANCE_GIVE_UP_S"):
            cfg.give_up_s = float(os.environ["GUIDANCE_GIVE_UP_S"])
        if os.environ.get("GUIDANCE_HAND_GIVE_UP_S"):
            cfg.hand_give_up_s = float(os.environ["GUIDANCE_HAND_GIVE_UP_S"])
        _default = Guidance(sink, cfg, tone)
        _default.set_option(key)
        _default.set_grab_option(grab)
        tracing.meta(option=key, grab_option=grab)
    return _default


def _option_key(which) -> str | None:
    """A number, a key from OPTIONS, or one of the older names -> the option's key."""
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


def _grab_key(which) -> str | None:
    which = str(which).strip().lower()
    keys = [k for k, _, _ in GRAB_OPTIONS]
    if which.isdigit() and 1 <= int(which) <= len(keys):
        return keys[int(which) - 1]
    return which if which in keys else None


def set_grab_option(which) -> str:
    """Switch the rule for saying "Grab" while running. Returns its spoken name."""
    key = _grab_key(which)
    if key is None:
        raise ValueError(f"no grab option {which!r}")
    _get().set_grab_option(key)
    number, _, title = grab_option()
    tracing.event("grab_option", option=key, number=number, title=title)
    return title


def grab_option() -> tuple[int, str, str]:
    """(number, key, name) of the grab rule in use."""
    key = _get().cfg.grab_option
    number = [k for k, _, _ in GRAB_OPTIONS].index(key) + 1
    return number, key, GRAB_OPTIONS[number - 1][1]


def current_target() -> str | None:
    """What is being looked for right now, or None between requests."""
    g = _get()
    return g.target if g.state not in (IDLE, GRAB) else None


def update(scene: Scene) -> str:
    global _logged_state
    tracing.begin_frame(scene.t)
    g = _get()
    with tracing.span("guidance"):
        state = g.update(scene)
    if state != _logged_state:
        _logged_state = state
        tracing.event("state", state=state, target=g.target, option=g.cfg.option, grab_option=g.cfg.grab_option)
    age_ms = (time.monotonic() - scene.t) * 1000      # only meaningful when scene.t is an arrival time
    # What a run needs afterwards to see why "Grab" came when it did: where the hand had to go
    # (err as measured, and lead: what was steered by, judged from where the moving hand will be
    # and without a depth that has just jumped), in millimetres, how far
    # it was from grabbing distance (zone: under 1 is inside), whether that frame was believed
    # (trusted), for how long it has been inside (near_s), its speed across the view, which axes
    # were lined up (1 sideways, 2 height, 3 both), and the options.
    tracing.put(state=state, option=g.cfg.option, grab_option=g.cfg.grab_option, **g.trace,
                **({"e2e_ms": round(age_ms, 3)} if 0 <= age_ms < 10000 else {}))
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
