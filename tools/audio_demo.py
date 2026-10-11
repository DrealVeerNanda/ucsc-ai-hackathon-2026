"""Hear the guidance sounds without a camera, a phone or a sound card on this machine.

  python tools/audio_demo.py                # writes out/audio_demo.wav (about 20 s): play it with headphones
  python tools/audio_demo.py --words        # the spoken-words fallback instead of tones
  python tools/audio_demo.py --hybrid       # direction once, progress beeps, chime, next step
  python tools/audio_demo.py --play         # also play it now through the default output

It runs the real guidance.py on a made-up reach: search, find, show the hand, move right,
then up, then forward, hold at the object, grab.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from audio_out import Mixer, ToneConfig, write_wav  # noqa: E402
from guidance import Guidance, GuidanceConfig  # noqa: E402
from interfaces import Scene  # noqa: E402

OBJ = np.array([0.05, 0.12, 0.60])
START = np.array([-0.20, 0.30, 0.35])       # hand: left of, below and nearer than the object
FPS, BLOCK = 30, 1600                        # 48000 / 30


def hand_at(t: float):
    """Where the hand is at time t, or None while it is out of view."""
    if t < 5.0:
        return None                                              # looking, found it, show me your hand
    legs = [(5.0, 0), (8.0, 1), (11.0, 2)]                       # fix one axis at a time, 3 s each
    hand = START.copy()
    for t0, axis in legs:
        hand[axis] += (OBJ[axis] - START[axis]) * min(1.0, max(0.0, (t - t0) / 2.5))
    if 12.0 < t < 12.4:
        return None                                              # a blink: the sound should hold
    return hand


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument("--words", action="store_true")
    modes.add_argument("--hybrid", action="store_true")
    ap.add_argument("--play", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "out" / "audio_demo.wav"))
    a = ap.parse_args()

    tone = ToneConfig(volume=0.16) if a.hybrid else ToneConfig()
    mixer = Mixer(tone, stale_s=1e9)                             # rendered offline, faster than real time
    g = Guidance(mixer, GuidanceConfig(words=a.words, hybrid=a.hybrid), tone)
    blocks, states = [], []
    g.reset("mug", now=0.0)
    for i in range(int(20 * FPS)):
        t = i / FPS
        hand = hand_at(t)
        obj = OBJ if t >= 2.0 else None                          # found after 2 s
        states.append(g.update(Scene(obj, hand, obj is not None, hand is not None, None, None, t), now=t))
        blocks.append(mixer.render(BLOCK))
        if states[-1] == "idle":
            break
    audio = np.concatenate(blocks)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    write_wav(a.out, audio, tone.sr)
    changes = [(i / FPS, s) for i, s in enumerate(states) if i == 0 or s != states[i - 1]]
    print("states:  " + "  ".join(f"{t:.1f}s {s}" for t, s in changes))
    print("spoken:  " + ", ".join(mixer.spoken))
    print(f"wrote {a.out} ({len(audio) / tone.sr:.1f} s)")
    if a.play:
        import sounddevice as sd
        sd.play(audio, tone.sr)
        sd.wait()
