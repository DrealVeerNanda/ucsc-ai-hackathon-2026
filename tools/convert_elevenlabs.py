"""Convert the downloaded Brian MP3 set to offline guidance clips.

Run in the project's environment (PyAV is installed with faster-whisper):
    python tools/convert_elevenlabs.py
"""
from pathlib import Path
import wave

import av
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "audio" / "eleven_labs_raw"
DESTINATION = ROOT / "audio" / "elevenlabs"
SR = 48000
LABELS = {
    "looking": "Looking.", "found_it": "Found it.",
    "show_me_your_hand": "Show me your ha", "right": "Move your hand ",
    "left": "Moce Hand Left", "up": "Move Hand up", "down": "Move Hand Down",
    "forward": "Move Hand Forward", "back": "Move Hand Back toward you",
    "grab": "Brian - Grab", "i_lost_it": "I Lost the object",
    "i_cant_see_that": "I cant see that object",
}


def convert(path: Path) -> np.ndarray:
    resampler = av.AudioResampler(format="s16", layout="mono", rate=SR)
    chunks = []
    with av.open(str(path)) as container:
        for frame in container.decode(audio=0):
            chunks.extend(f.to_ndarray().reshape(-1) for f in resampler.resample(frame))
        chunks.extend(f.to_ndarray().reshape(-1) for f in resampler.resample(None))
    if not chunks:
        raise ValueError(f"No audio in {path.name}")
    samples = np.concatenate(chunks).astype(np.float32) / 32768
    active = np.flatnonzero(np.abs(samples) > 0.008)
    if not active.size:
        raise ValueError(f"No audible speech in {path.name}")
    padding = round(0.06 * SR)
    samples = samples[max(0, active[0] - padding):min(len(samples), active[-1] + padding + 1)]
    voiced = samples[np.abs(samples) > 0.008]
    rms = float(np.sqrt(np.mean(voiced ** 2)))
    # Match speech levels, with capped gain and enough peak headroom for the mixer.
    samples *= min(3.0, 0.13 / rms, 0.85 / float(np.max(np.abs(samples))))
    return samples


def save(path: Path, samples: np.ndarray):
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 2, SR, 0, "NONE", "not compressed"))
        output.writeframes(np.round(samples * 32767).astype(np.int16).tobytes())


def main():
    clips = {}
    for name, label in LABELS.items():
        matches = [path for path in SOURCE.glob("*.mp3") if label.lower() in path.name.lower()]
        if len(matches) != 1:
            raise ValueError(f"Expected one MP3 for {name}, found {len(matches)}")
        clips[name] = convert(matches[0])
    # Validate the entire set before installing any replacements.
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for name, samples in clips.items():
        save(DESTINATION / (name + ".wav"), samples)
        print(f"{name}.wav: {len(samples) / SR:.2f} s")
    preview = ROOT / "out" / "elevenlabs_preview_all.wav"
    preview.parent.mkdir(exist_ok=True)
    save(preview, np.concatenate([np.concatenate((samples, np.zeros(round(0.8 * SR), np.float32)))
                                 for samples in clips.values()]))
    print(f"Converted {len(clips)} clips. Preview: {preview}")


if __name__ == "__main__":
    main()
