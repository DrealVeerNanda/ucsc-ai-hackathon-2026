"""Generate slower Daniel clips on a Mac; playback on the GX10 stays offline.

Run: python tools/make_calm_audio.py
Original audio/*.wav files are preserved. audio/calm/*.wav takes precedence.
"""
from pathlib import Path
import argparse
import subprocess
import tempfile
import wave

ROOT = Path(__file__).resolve().parent.parent
PHRASES = {
    "looking": "Looking.", "found_it": "Found it.",
    "show_me_your_hand": "Show me your hand.", "grab": "Grab.",
    "i_lost_it": "I lost it.", "i_cant_see_that": "I can't see that.",
    **{word: word.capitalize() + "." for word in
       ("left", "right", "up", "down", "forward", "back")},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice", default="Daniel", help="installed macOS voice")
    parser.add_argument("--rate", type=int, default=135, help="words per minute")
    args = parser.parse_args()
    if not 80 <= args.rate <= 250:
        parser.error("--rate must be between 80 and 250")
    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        for name, phrase in PHRASES.items():
            aiff, wav = scratch / (name + ".aiff"), scratch / (name + ".wav")
            subprocess.run(["say", "-v", args.voice, "-r", str(args.rate), "-o", str(aiff), phrase], check=True)
            subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@48000", str(aiff), str(wav)], check=True)
            with wave.open(str(wav)) as clip:
                if clip.getnframes() == 0:
                    raise RuntimeError("Speech generation produced empty audio; no clips installed.")
        destination = ROOT / "audio" / "calm"
        destination.mkdir(parents=True, exist_ok=True)
        for wav in scratch.glob("*.wav"):
            (destination / wav.name).write_bytes(wav.read_bytes())
    print(f"Saved {len(PHRASES)} calm clips to {destination}")


if __name__ == "__main__":
    main()
