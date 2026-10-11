"""Hear the guidance sounds without a camera, a phone or a sound card on this machine.

  python tools/audio_demo.py                # writes out/audio_demo.wav (about 20 s): play it with headphones
  python tools/audio_demo.py --option 7     # the same reach in another of guidance.OPTIONS (number or name)
  python tools/audio_demo.py --grab 1       # ...or with another of guidance.GRAB_OPTIONS
  python tools/audio_demo.py --play         # also play it now through the default output
  python tools/audio_demo.py --all          # every option and every grab rule into out/options/, with a
                                            # page to play them from (index.html) and the tables below
  python tools/audio_demo.py --table        # only the tables, no sound

It runs the real guidance.py and the real mixer on a made-up reach for a bottle: the hand comes
into view low and to the left, moves up and right, overshoots, comes back, pushes forward,
hesitates a hand's width short, and takes hold. What the camera would report of it is made as
rough as the live run of 10 Oct was: 6 mm of fingertip jitter, the hand's depth flipping
between lidar and the size estimate 4 cm apart, the tracker losing the hand for a frame now and
then and for longer at the object, depth read past the finger onto the bottle, and a hand on
the bottle measuring short of it.

The tables are SIMULATION. In the first, a stand-in wearer follows each option's cues: it
understands every cue exactly as meant and reacts a quarter of a second after hearing it. So
the table shows how long each way of guiding takes to say what it has to say, not which one a
person finds easy. In the second, the hand does four set things and each grab rule is judged
on when it says "Grab".
"""
import argparse
import html
import sys
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import guidance  # noqa: E402
from audio_out import Mixer, PrintSink, ToneConfig, write_wav  # noqa: E402
from guidance import Guidance, GuidanceConfig  # noqa: E402
from interfaces import Scene  # noqa: E402

FPS, SR = 30, 48000
BLOCK = SR // FPS
FX, CX, CY = 682.0, 360.0, 480.0             # the phone's camera at the size the loop uses (720 x 960)
OBJ = np.array([0.05, 0.10, 0.45])           # middle of the bottle's front, metres, camera axes
SIZE = np.array([0.065, 0.20])               # the bottle: width and height
START = np.array([-0.17, 0.30, 0.24])        # hand: left of, below and nearer than the bottle

# What carries each piece of information, for the page and for whoever explains it on stage.
CUES = {
    "tones": ("which ear the beeps are in", "pitch above or below a middle note", "beeps speed up", "steady high note, chime, Grab"),
    "steps": ("the spoken word", "the spoken word, after sideways is done", "the word, then beeps speed up", "chime per step, then Grab"),
    "combined": ("spoken, with up or down", "spoken, with left or right", "'forward' after the chime", "chime when lined up, then Grab"),
    "combined_beeps": ("spoken, and the ear the beeps lean to", "spoken", "'forward', and beeps speed up", "chime, then Grab"),
    "clock": ("the hour: three is right, nine is left", "the hour: twelve is up, six is down", "'forward', beeps speed up", "chime, then Grab"),
    "numbers": ("spoken with centimetres: 'right twenty'", "spoken with centimetres: 'up ten'", "'forward fifteen'", "chime when lined up, then Grab"),
    "echo": ("the ear the second note is in", "second note higher or lower than the first", "second note follows sooner", "the two notes become one, chime, Grab"),
    "hotcold": ("the ear the note is in", "found by moving: the note rises when you are right", "the note rises", "the note reaches its top, chime, Grab"),
    "spearcons": ("fast word 'left' or 'right'", "fast word 'up' or 'down', taking turns with the other", "fast word 'forward'; all repeat faster when closer", "chime when lined up, then Grab"),
    "beam": ("spoken once; after that ticks in the ear to come back to", "spoken once; after that high ticks for up, low for down", "'forward', then quiet means carry on", "chime on the beam, then Grab"),
    "twostage": ("spoken with centimetres, then the ear the note is in", "spoken with centimetres, then by the rising note", "spoken, then the note rises", "the note reaches its top, chime, Grab"),
    "detector": ("the ear the hum is in; it rises as you line up", "the hum rises as you line up", "beeps speed up once on the object", "hum turns to beeps on the line; chime, Grab"),
}


# ---------------------------------------------------------------------------
# What the camera reports of a hand, as rough as it was live.

class Seen:
    def __init__(self, seed: int = 0, clean: bool = False):
        self.rng = np.random.default_rng(seed)
        self.clean = clean
        self.recent = deque(maxlen=3)        # perception averages three frames
        self.by_size = False
        self.size_off = 0.04
        self.gone = 0                        # frames the tracker still has to miss
        self.past = 0                        # frames lidar still reads past the finger
        self.obj = OBJ.copy()                # where perception believes the object is

    def box(self, z=None):
        z = OBJ[2] if z is None else z
        u, v = CX + FX * OBJ[0] / OBJ[2], CY + FX * OBJ[1] / OBJ[2]
        w, h = FX * SIZE / OBJ[2]
        return (u - w / 2, v - h / 2, u + w / 2, v + h / 2)

    def scene(self, hand, t: float, lose: bool = False) -> Scene:
        """`hand` is where the fingertip really is, or None when it is out of view."""
        rng, box = self.rng, self.box()
        if hand is None or lose:
            self.recent.clear()
            return Scene(self.obj, None, True, False, box, None, t)
        gap = OBJ - hand
        at_object = abs(gap[0]) < SIZE[0] / 2 + 0.02 and abs(gap[1]) < SIZE[1] / 2 + 0.02 and gap[2] < 0.12
        seen = np.array(hand, float)
        if not self.clean:
            if self.gone == 0 and rng.random() < (0.05 if at_object else 0.02):
                self.gone = int(rng.integers(3, 9)) if at_object and rng.random() < 0.25 else 1
            if self.gone:
                self.gone -= 1
                self.recent.clear()
                return Scene(self.obj, None, True, False, box, None, t)
            seen[:2] += rng.normal(0.0, 0.006, 2)
            if rng.random() < 0.11:                                   # 249 flips in 2296 frames
                self.by_size = not self.by_size
                self.size_off = 0.04 * rng.choice((-1.0, 1.0)) + rng.normal(0.0, 0.015)
            seen[2] += (self.size_off + rng.normal(0.0, 0.012)) if self.by_size else rng.normal(0.0, 0.006)
            if at_object:
                seen[2] -= 0.06 + 0.04 * rng.random()                 # a hand on the bottle measured 6-15 cm short
            over = abs(gap[0]) < SIZE[0] / 2 and abs(gap[1]) < SIZE[1] / 2 and gap[2] > 0.12
            if over and self.past == 0 and rng.random() < 0.012:
                self.past = int(rng.integers(3, 12))
            if self.past:
                self.past -= 1
                seen[2] = OBJ[2] + rng.normal(0.0, 0.04)              # lidar read the bottle behind the finger
        if self.recent and float(np.linalg.norm(seen - self.recent[-1])) > 0.15:
            self.recent.clear()                                       # perception restarts its average on a jump
        self.recent.append(seen)
        seen = np.mean(self.recent, axis=0)
        half = 0.35 * SIZE * self.obj[2] / OBJ[2]
        aim = np.array([*(self.obj[:2] + np.clip(seen[:2] - self.obj[:2], -half, half)), self.obj[2]])
        px = (CX + FX * seen[0] / seen[2], CY + FX * seen[1] / seen[2])
        return Scene(self.obj, seen, True, True, box, px, t, aim_xyz=aim)


def ease(a: float) -> float:
    a = min(1.0, max(0.0, a))
    return a * a * (3 - 2 * a)


def scripted(t: float):
    """Where the hand really is at time t in the set reach, or None while it is out of view.
    Returns (position, tracker loses it now)."""
    if t < 3.2:
        return None, False                                           # looking, found it, show me your hand
    p = START.copy()
    side = OBJ[0] - START[0]
    p[0] += side * 1.3 * ease((t - 3.2) / 2.2) - side * 0.3 * ease((t - 6.0) / 1.2)     # overshoots 30%, comes back
    p[1] += (OBJ[1] - START[1]) * ease((t - 3.6) / 2.6)
    depth = OBJ[2] - START[2]
    p[2] += depth * 0.62 * ease((t - 7.6) / 2.2) + depth * 0.38 * ease((t - 11.0) / 1.6)   # hesitates 8 cm short
    return p, 8.6 < t < 9.0                                          # and the tracker drops it for 0.4 s on the way


def render(option="combined", grab=None, seed=0, seconds=20.0, sound=True, script=scripted, seen=None):
    """Run guidance over the set reach. Returns (audio or None, what was said with times,
    state changes, time of "Grab" or None)."""
    tone = ToneConfig()
    sink = Mixer(tone, stale_s=1e9) if sound else PrintSink(quiet=True)   # rendered offline, faster than real time
    g = Guidance(sink, GuidanceConfig(), tone)
    g.set_option(option)
    if grab:
        g.set_grab_option(grab)
    seen = seen or Seen(seed)
    blocks, said, states, grab_t, idle_t, n = [], [], [], None, None, 0
    g.reset("bottle", now=0.0)
    for i in range(int(seconds * FPS)):
        t = i / FPS
        hand, lose = script(t)
        sc = seen.scene(hand, t, lose) if t >= 1.5 else Scene(None, None, False, False, None, None, t)   # found after 1.5 s
        state = g.update(sc, now=t)
        said += [(t, w) for w in sink.spoken[n:]]
        n = len(sink.spoken)
        if not states or states[-1][1] != state:
            states.append((t, state))
        if state == "grab" and grab_t is None:
            grab_t = t
        if sound:
            blocks.append(sink.render(BLOCK))
        if state == "idle":
            idle_t = idle_t if idle_t is not None else t
            if t - idle_t > 0.5:
                break
    return (np.concatenate(blocks) if sound else None), said, states, grab_t


# ---------------------------------------------------------------------------
# The stand-in wearer: understands every cue as it was meant, a moment after hearing it.

WORDS = {"left": (0, -1), "right": (0, 1), "up": (1, -1), "down": (1, 1), "back": (2, -1), "forward": (2, 1)}
CM = {v: k for k, v in guidance.AMOUNTS.items()}


class Wearer:
    SPEED, PUSH, REACT = 0.16, 0.11, 0.25      # m/s sideways, m/s forward, s from hearing to moving

    def __init__(self, rng):
        self.rng = rng
        self.p = START + rng.normal(0.0, 0.03, 3)
        self.v = np.zeros(3)
        self.left = [None, None, None]         # metres still to go on each axis, when the cue said how far
        self.todo = []                         # (time it takes effect, what was heard)
        self.tones = deque()                   # (time, tone) so the tone is acted on late too
        self.care = np.ones(3)                 # each overshoot makes the next move on that axis half as fast
        self.went = np.zeros(3)                # the way it was last sent on each axis
        self.probe = None                      # for the notes that only say warmer or colder
        self.done = False
        self.ticking = False                   # moving because of ticks rather than words

    def hear(self, t: float, text: str, ends: float):
        """A word is understood when it ends, a chime as soon as it starts."""
        self.todo.append(((t if text == "aligned" else ends) + self.REACT, text))

    def _words(self, text: str):
        text = text.replace("fast_", "").replace("word_", "").replace(" (fast)", "")
        if text in ("grab", "Grab"):
            self.v[:] = 0
            self.done = True
        elif text == "aligned":
            self.v[:2] = 0
            self.left[0] = self.left[1] = None
        elif "o'clock" in text:
            hour = guidance.HOURS.index(text.split()[0])
            a = np.radians(30.0 * hour)
            self.v = np.array([np.sin(a), -np.cos(a), 0.0]) * self.SPEED
        else:
            parts = [p.split() for p in text.replace(" and ", ",").replace("lean ", "").split(",")]
            named = [p for p in parts if p and p[0] in WORDS]
            if not named:
                return
            sideways = any(WORDS[p[0]][0] < 2 for p in named)
            self.v[:] = 0
            self.left = [None, None, None]
            for p in named:
                axis, sign = WORDS[p[0]]
                if axis == 2 and sideways:
                    continue
                if self.went[axis] == -sign:
                    self.care[axis] = max(0.25, 0.5 * self.care[axis])
                self.went[axis] = sign
                self.v[axis] = sign * (self.PUSH if axis == 2 else self.SPEED) * self.care[axis]
                if len(p) > 1 and " ".join(p[1:]) in CM:
                    self.left[axis] = CM[" ".join(p[1:])] / 100.0

    def _tone(self, option: str, tone, t: float):
        pan, freq, rate, mode = tone[:4]
        base = ToneConfig().base_hz
        if mode == "silent":
            if self.ticking:                                         # the ticks stopped: back on the beam
                self.v[:2], self.ticking = 0.0, False
            return
        if mode == "beep" and option in ("steps", "combined_beeps", "clock"):
            moving = float(np.linalg.norm(self.v))                   # faster beeps: nearly there, slow down
            if moving > 0:
                self.v *= min(1.0, self.SPEED * float(np.clip(1.3 - rate / ToneConfig().rate_near, 0.25, 1.0)) / moving)
            return
        if mode == "steady" and freq == ToneConfig().steady_hz:
            self.v[:] = 0                                            # the "you are on it" note: hold still
        elif option in ("tones", "echo"):
            up = np.log2(freq / base)
            self.v = np.array([self.SPEED * np.clip(2 * pan, -1, 1), -self.SPEED * np.clip(4 * up, -1, 1), 0.0])
            if abs(pan) < 0.08 and abs(up) < 0.04:
                self.v[2] = self.PUSH
        elif option == "beam" and mode == "beep":
            self.v[:2] = 0.5 * self.SPEED * np.sign(pan), -0.5 * self.SPEED * np.sign(np.log2(freq / base))
            self.ticking = True
        elif option == "detector" and mode == "beep":
            self.v = np.array([0.0, 0.0, self.PUSH])
        elif mode == "steady":
            # Only "warmer" or "colder": keep going while the note rises, try the next way when it falls.
            ways = [np.array([np.sign(pan) or 1.0, 0, 0]), np.array([0, -1.0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])]
            if abs(pan) < 0.05:
                ways = ways[1:]
            if self.probe is None or t - self.probe[0] >= 0.45:
                rising = self.probe is not None and freq > self.probe[1] * 1.004
                i = 0 if self.probe is None else (self.probe[2] if rising else self.probe[2] + 1)
                if abs(pan) >= 0.05 and not rising:
                    i = 0                                            # the ear always says which side
                self.probe = (t, freq, i % len(ways))
            self.v = ways[self.probe[2] % len(ways)] * 0.12

    def step(self, t: float, option: str, tone, dt: float):
        self.tones.append((t, tone))
        for due, text in [x for x in self.todo if x[0] <= t]:
            self._words(text)
            self.todo.remove((due, text))
        late = None
        while self.tones and self.tones[0][0] <= t - self.REACT:
            late = self.tones.popleft()[1]
        if late is not None and not self.done:
            self._tone(option, late, t)
        move = self.v * dt
        for a in range(3):
            if self.left[a] is not None:                             # told how far: make that move and stop
                self.left[a] -= abs(move[a])
                if self.left[a] <= 0:
                    self.v[a], self.left[a] = 0.0, None
        self.p = self.p + move + self.rng.normal(0.0, 0.0006, 3)     # an unsteady hand
        self.p[2] = min(self.p[2], OBJ[2] + 0.01)                    # the bottle is in the way


def follow(option: str, seed: int, grab=None, seconds: float = 40.0) -> dict:
    """One reach by the stand-in wearer. Real guidance, real mixer (for how long each phrase takes)."""
    rng = np.random.default_rng(1000 + seed)
    tone = ToneConfig()
    sink = Mixer(tone, stale_s=1e9)
    g = Guidance(sink, GuidanceConfig(), tone)
    g.set_option(option)
    if grab:
        g.set_grab_option(grab)
    seen, who = Seen(seed), Wearer(rng)
    g.reset("bottle", now=0.0)
    said, talking, grab_t, t, n = [], 0, None, 0.0, 0
    for i in range(int(seconds * FPS)):
        t = i / FPS
        state = g.update(seen.scene(who.p if t >= 1.0 else None, t), now=t)
        for w in sink.spoken[n:]:
            length = (sink._clip.size / SR) if sink._clip is not None else 0.3
            said.append((t, w))
            who.hear(t, w, t + length)
            talking += length if w not in ("looking", "found_it", "show_me_your_hand", "grab") else 0.0
        n = len(sink.spoken)
        sink.render(BLOCK)
        if state == "grab":
            grab_t = t
            break
        who.step(t, option, getattr(sink.synth, "_target", (0, 0, 0, "silent")), 1.0 / FPS)
    words = [w for _, w in said if w not in ("looking", "found_it", "show_me_your_hand", "aligned", "grab")]
    gap = OBJ - who.p
    return dict(time=None if grab_t is None else grab_t - 1.0, words=len(words), chimes=sum(w == "aligned" for _, w in said),
                reversals=reversals(said), talking=talking / max(0.1, t - 1.0),
                miss_cm=100 * float(np.linalg.norm([max(0.0, abs(gap[0]) - SIZE[0] / 2), max(0.0, abs(gap[1]) - SIZE[1] / 2), gap[2]])))


def reversals(said, within: float = 1.0) -> int:
    """Times a direction was followed by its opposite in under a second."""
    last, n = {}, 0
    for t, w in said:
        for part in w.replace("fast_", "").replace(",", " ").split():
            if part in guidance.OPPOSITE and part not in ("forward", "back"):
                n += t - last.get(guidance.OPPOSITE[part], -1e9) < within
                last[part] = t
    return n


def phrase_seconds(option: str) -> str:
    """How long the option's typical direction takes to say: nothing can be corrected faster."""
    m = Mixer(ToneConfig(), stale_s=1e9)
    if option in ("tones", "echo", "hotcold", "detector"):
        return "no words"
    if option in ("steps",):
        clips = [m.clips[w].size / SR for w in ("left", "right", "up", "down", "forward")]
    elif option == "spearcons":
        clips = [m.clips["fast_" + w].size / SR for w in ("left", "right", "up", "down", "forward") if "fast_" + w in m.clips]
    else:
        from audio_out import synthesize
        texts = {"clock": ["two o'clock", "eleven o'clock"], "numbers": ["right twenty, up ten", "forward fifteen"],
                 "twostage": ["right twenty, up ten", "forward fifteen"]}.get(option, ["up and left", "forward", "down and right"])
        clips = [s.size / SR for s in (synthesize(x, SR) for x in texts) if s is not None]
    return f"{min(clips):.1f}-{max(clips):.1f} s" if clips else "no voice here"


def options_table(trials: int) -> str:
    lines = ["SIMULATION: a stand-in wearer that understands every cue and reacts after 0.25 s. Not a measurement of people.",
             "median s: hand in view to Grab. speech: share of that time a direction or chime was sounding. off by: how far",
             "the fingertip really was from the bottle at Grab.",
             f"{'option':<32}{'reached':>8}{'median s':>9}{'slowest':>8}{'words':>6}{'chimes':>7}{'reversals':>10}{'speech':>8}{'off by':>8}   one direction takes"]
    for i, (key, title, _) in enumerate(guidance.OPTIONS):
        runs = [follow(key, s) for s in range(trials)]
        done = [r for r in runs if r["time"] is not None]
        if done:
            t = np.array([r["time"] for r in done])
            lines.append(f"{i + 1:>2} {title:<29}{len(done):>4}/{trials:<3}{np.median(t):>9.1f}{t.max():>8.1f}"
                         f"{np.median([r['words'] for r in done]):>6.0f}{np.median([r['chimes'] for r in done]):>7.0f}"
                         f"{np.mean([r['reversals'] for r in done]):>10.1f}{100 * np.mean([r['talking'] for r in done]):>7.0f}%"
                         f"{np.median([r['miss_cm'] for r in done]):>6.1f}cm   {phrase_seconds(key)}")
        else:
            lines.append(f"{i + 1:>2} {title:<29}{0:>4}/{trials:<3}   never reached in 40 s                                        {phrase_seconds(key)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The grab rules on four set situations.

def _touch(t):
    p, lose = scripted(t)
    return p, lose


def _pass_by(t):
    """A hand sweeping across in front of the bottle at 35 cm/s, 4 cm short of it: never stopping."""
    if t < 3.0:
        return None, False
    x = -0.30 + 0.35 * (t - 3.0)
    return (np.array([x, OBJ[1], OBJ[2] - 0.04]) if x < 0.40 else None), False


def _out_of_reach(t):
    """Over the bottle in the picture but 22 cm short of it for 8 s, the arm at full stretch."""
    if t < 3.0:
        return None, False
    p = START + (OBJ - np.array([0.0, 0.0, 0.22]) - START) * ease((t - 3.0) / 3.0)
    return p, False


class _Stolen(Seen):
    """The same, and at 8 s the tracker blinks and the bottle's depth is measured on the hand."""
    def scene(self, hand, t, lose=False):
        if 8.0 <= t < 8.1:
            lose = True
            self.obj = np.array([OBJ[0], OBJ[1], OBJ[2] - 0.22])
        return super().scene(hand, t, lose)


def grab_table(trials: int) -> str:
    lines = ["SIMULATION: the same four hand movements for every rule, 'Grab' judged against where the hand really was.",
             f"{'rule':<24}{'takes hold: Grab after':>24}{'never':>7}{'sweeps past':>13}{'out of reach':>14}{'object read on the hand':>25}"]
    arrive = 11.0 + 1.6 * 0.75                                        # in the set reach the fingertip is within 3 cm from here
    for i, (key, title, _) in enumerate(guidance.GRAB_OPTIONS):
        late, never, false = [], 0, [0, 0, 0]
        for s in range(trials):
            g = render("combined", key, s, sound=False)[3]
            if g is None:
                never += 1
            else:
                late.append(g - arrive)
            for j, (script, seen) in enumerate(((_pass_by, Seen(s)), (_out_of_reach, Seen(s)), (_out_of_reach, _Stolen(s)))):
                false[j] += render("combined", key, s, 14.0, False, script, seen)[3] is not None
        took = f"{np.median(late):+.1f} s ({min(late):+.1f} to {max(late):+.1f})" if late else "-"
        lines.append(f"{i + 1} {title:<22}{took:>24}{never:>4}/{trials:<2}" + "".join(f"{f:>{w}}/{trials}" for f, w in zip(false, (10, 11, 22))))
    lines.append("Columns 3-5 are false Grabs (lower is better). 'Grab after' is seconds after the fingertip is really within 3 cm; "
                 "negative means it was said on the way in.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------

def page(folder: Path, rows, grabs, tables: str):
    def item(n, title, text, cues, wav):
        cue = "" if cues is None else ("<dl>" + "".join(f"<dt>{k}</dt><dd>{html.escape(v)}</dd>" for k, v in
                                                         zip(("left / right", "up / down", "forward", "arrived"), cues)) + "</dl>")
        return (f"<section><h2><span>{n}</span> {html.escape(title)}</h2><p>{html.escape(text)}</p>{cue}"
                f"<audio controls preload='none' src='{wav}'></audio></section>")
    body = "".join(item(*r) for r in rows)
    grab = "".join(item(*r) for r in grabs)
    (folder / "index.html").write_text(f"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Proprio: ways of guiding</title>
<style>
:root{{--bg:#f7f6f2;--ink:#1c1b19;--dim:#66625a;--line:#d9d5cb;--card:#fff;--mark:#0b6e69}}
@media (prefers-color-scheme:dark){{:root{{--bg:#161512;--ink:#eeebe3;--dim:#a39e92;--line:#34312b;--card:#1f1d19;--mark:#58c9c2}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,sans-serif}}
main{{max-width:880px;margin:0 auto;padding:32px 16px 64px}}
h1{{font-size:28px;margin:0 0 8px}} h2{{font-size:19px;margin:0 0 4px}} h2 span{{color:var(--mark);font-variant-numeric:tabular-nums}}
h3{{margin:40px 0 12px;font-size:14px;letter-spacing:.08em;text-transform:uppercase;color:var(--dim)}}
p{{margin:0 0 8px}} .lead{{color:var(--dim);max-width:64ch}}
section{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:12px 0}}
dl{{display:grid;grid-template-columns:max-content 1fr;gap:2px 16px;margin:8px 0 12px;font-size:14px}}
dt{{color:var(--dim)}} dd{{margin:0}} audio{{width:100%}}
pre{{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;font-size:12.5px;line-height:1.45}}
</style><main>
<h1>Twelve ways of guiding a hand</h1>
<p class="lead">The same made-up reach for a bottle in every clip: the hand comes in low and to the left, goes up and right,
overshoots, comes back, pushes forward, hesitates a hand's width short, and takes hold. Use headphones: left and right matter.
Say "option six" during a run to switch, or "grab option two" for the rule below.</p>
<h3>Guidance options</h3>{body}
<h3>When "Grab" is said (all with option 3)</h3>{grab}
<h3>Simulation, not measurement</h3><pre>{html.escape(tables)}</pre>
</main>""")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument("--words", action="store_true")
    modes.add_argument("--hybrid", action="store_true")
    modes.add_argument("--option", help="one of guidance.OPTIONS: its number or its name")
    ap.add_argument("--grab", help="one of guidance.GRAB_OPTIONS: its number or its name")
    ap.add_argument("--all", action="store_true", help="every option and grab rule into out/options/, with index.html")
    ap.add_argument("--table", action="store_true", help="print the simulated comparisons only")
    ap.add_argument("--trials", type=int, default=20, help="reaches per row of the tables")
    ap.add_argument("--play", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "out" / "audio_demo.wav"))
    a = ap.parse_args()

    if a.all or a.table:
        tables = options_table(a.trials) + "\n\n" + grab_table(a.trials)
        if a.all:
            folder = ROOT / "out" / "options"
            folder.mkdir(parents=True, exist_ok=True)
            rows, grabs = [], []
            for i, (key, title, text) in enumerate(guidance.OPTIONS):
                audio, said, _, grab_t = render(key)
                name = f"{i + 1:02d}_{key}.wav"
                write_wav(folder / name, audio, SR)
                rows.append((i + 1, title, text, CUES.get(key), name))
                print(f"{name:<24}{len(audio) / SR:5.1f} s   Grab at {grab_t if grab_t is None else round(grab_t, 1)} s")
            for i, (key, title, text) in enumerate(guidance.GRAB_OPTIONS):
                audio, said, _, grab_t = render("combined", key)
                name = f"grab_{i + 1}_{key}.wav"
                write_wav(folder / name, audio, SR)
                grabs.append((i + 1, title, text, None, name))
                print(f"{name:<24}{len(audio) / SR:5.1f} s   Grab at {grab_t if grab_t is None else round(grab_t, 1)} s")
            page(folder, rows, grabs, tables)
            print(f"\nopen {folder / 'index.html'}\n")
        print(tables)
        sys.exit(0)

    option = guidance._option_key(a.option) if a.option else ("words" if a.words else "steps" if a.hybrid else "combined")
    if option is None:
        ap.error(f"no guidance option {a.option!r}")
    grab = guidance._grab_key(a.grab) if a.grab else None
    if a.grab and grab is None:
        ap.error(f"no grab option {a.grab!r}")
    audio, said, states, _ = render(option, grab)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    write_wav(a.out, audio, SR)
    print("states:  " + "  ".join(f"{t:.1f}s {s}" for t, s in states))
    print("spoken:  " + ", ".join(w for _, w in said))
    print(f"wrote {a.out} ({len(audio) / SR:.1f} s)")
    if a.play:
        import sounddevice as sd
        sd.play(audio, SR)
        sd.wait()
