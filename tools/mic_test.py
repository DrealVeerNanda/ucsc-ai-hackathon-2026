"""Check the microphone by ear, with nobody at the screen: it tells you what to say, beeps,
records, and then tells you what it heard. Stop the live run first; both want the microphone.

  python tools/mic_test.py                  # three rounds through the earbuds (or the default microphone)
  python tools/mic_test.py --rounds 1 --seconds 6
  python tools/mic_test.py --silent         # no prompts, no beep: just record and measure

Each round prints how loud the voice was against the room, whether any sound was lost on the
way in, whether the live run's own speech detector would have caught it, and what Whisper
made of it. The recordings are kept in out/mic_test_N.wav and the numbers in out/mic_test.json.
"""
import argparse
import json
import re
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import voice  # noqa: E402
from audio_out import synthesize  # noqa: E402

SR, BLOCK_S = voice.SR, 0.03
PROMPTS = [("After the beep, say: blue cup.", "blue cup"),
           ("After the beep, say: what is on the table.", "what is on the table"),
           ("After the beep, stay silent.", "")]


def sources() -> list[str]:
    """The microphones the sound server has right now."""
    try:
        out = subprocess.run(["wpctl", "status"], capture_output=True, text=True, timeout=3).stdout
        part = out.split("Audio", 1)[1].split("Video", 1)[0].split("Sources:", 1)[1].split("Source endpoints:", 1)[0]
        return re.findall(r"\d+\.\s+(.+?)\s+\[vol", part)
    except Exception:
        return []


def wait_for_microphone(name: str, timeout_s: float = 8.0) -> float | None:
    """Seconds until the earbuds' microphone exists after switching to headset mode."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        if any(name in s for s in sources()):
            return time.perf_counter() - t0
        time.sleep(0.1)
    return None


def play(samples: np.ndarray | None, sr: int = 48000):
    if samples is None or not len(samples):
        return
    import sounddevice as sd
    sd.play(np.concatenate([samples, np.zeros(int(sr * 0.25), np.float32)]), sr, blocking=True)


def say(text: str):
    print(f"  says: {text}", flush=True)
    play(synthesize(text, 48000))


def beep(sr: int = 48000) -> np.ndarray:
    t = np.arange(int(sr * 0.18)) / sr
    return (0.4 * np.sin(2 * np.pi * 880 * t) * np.minimum(1, 40 * np.minimum(t, t[-1] - t))).astype(np.float32)


def record(seconds: float) -> tuple[list, int]:
    """Blocks as they came in, and how many times the sound server said some were lost."""
    import sounddevice as sd
    blocks, lost = [], [0]

    def cb(data, frames, t, status):
        if status:
            lost[0] += 1
        blocks.append(data[:, 0].copy())
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=int(SR * BLOCK_S), callback=cb):
        time.sleep(seconds)
    return blocks, lost[0]


def measure(blocks: list, seconds: float) -> dict:
    """What a recording says about the microphone: did the sound arrive whole, how loud was the
    voice against the room, and would voice.cut_speech have started on it."""
    if not blocks:
        return {"arrived_s": 0.0, "wanted_s": seconds, "floor": 0.0, "voice": 0.0, "peak": 0.0,
                "threshold": 0.012, "caught": False, "caught_s": 0.0}
    level = np.array([float(np.sqrt(np.mean(np.square(b)))) for b in blocks])
    first = level[: max(1, int(round(0.3 / BLOCK_S)))]
    cut = voice.cut_speech(iter(blocks), start_wait_s=seconds, max_s=seconds)
    return {"arrived_s": round(sum(len(b) for b in blocks) / SR, 2), "wanted_s": seconds,
            "floor": round(float(np.percentile(level, 10)), 4), "voice": round(float(np.percentile(level, 95)), 4),
            "peak": round(float(max(np.max(np.abs(b)) for b in blocks)), 3),
            "threshold": round(max(0.012, 2.5 * float(np.median(first))), 4),
            "caught": cut is not None, "caught_s": round(len(cut) / SR, 2) if cut is not None else 0.0}


def verdict(m: dict, lost: int, text: str, want: str) -> str:
    if m["arrived_s"] < 0.9 * m["wanted_s"] or lost:
        return f"sound was lost on the way in ({m['arrived_s']} s of {m['wanted_s']} s arrived, {lost} warnings)"
    if m["peak"] < 0.003:
        return "the microphone is silent: nothing but zeros came in"
    if not want:
        return "stayed quiet, as asked" if not m["caught"] else "the room alone set off the speech detector"
    if not m["caught"]:
        return f"too quiet for the speech detector (loudest {m['voice']}, it needs {m['threshold']})"
    words = lambda s: re.sub(r"[^a-z ]+", "", s.lower()).split()
    return "heard correctly" if want in " ".join(words(text)) else "heard, but the words are wrong"


def save_wav(path: Path, samples: np.ndarray):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=4.0, help="how long to record after the beep")
    ap.add_argument("--silent", action="store_true", help="no prompts and no beep")
    ap.add_argument("--model", default="small.en")
    a = ap.parse_args()

    whisper = voice.load_whisper(a.model)                 # the same one the live run uses
    print(f"whisper {a.model} on the {whisper.on.upper()}", flush=True)
    out = ROOT / "out"
    out.mkdir(exist_ok=True)
    report = {"microphones_before": sources(), "rounds": []}

    buds = voice.Headset.find()
    if buds is not None:
        buds._profile(buds.headset)
        report["earbuds"] = buds.name
        report["microphone_ready_s"] = wait_for_microphone(buds.name)
        print(f"earbuds: {buds.name}; microphone ready after {report['microphone_ready_s']} s", flush=True)
        time.sleep(0.5)
    else:
        print(f"no earbuds found; using the default microphone ({report['microphones_before'] or 'none listed'})", flush=True)
    try:
        if not a.silent:
            say("Microphone test.")
        for i in range(a.rounds):
            prompt, want = PROMPTS[i % len(PROMPTS)]
            print(f"round {i + 1}", flush=True)
            if not a.silent:
                say(prompt)
                play(beep())
            blocks, lost = record(a.seconds)
            audio = np.concatenate(blocks) if blocks else np.zeros(0, np.float32)
            save_wav(out / f"mic_test_{i + 1}.wav", audio)
            m = measure(blocks, a.seconds)
            t = time.perf_counter()
            text = whisper(audio) if audio.size > SR // 4 else ""
            took = time.perf_counter() - t
            result = verdict(m, lost, text, want)
            report["rounds"].append({"asked": want, **m, "lost_warnings": lost, "heard": text, "whisper_s": round(took, 3),
                                     "verdict": result})
            print(f"  arrived {m['arrived_s']} of {m['wanted_s']} s, lost warnings {lost}; room {m['floor']}, voice {m['voice']}, "
                  f"peak {m['peak']}, detector needs {m['threshold']} -> {'caught' if m['caught'] else 'missed'}", flush=True)
            print(f"  whisper: {text!r} in {took:.2f} s", flush=True)
            print(f"  => {result}", flush=True)
            if not a.silent:
                say(f"I heard: {text}" if text else "I heard nothing.")
        if not a.silent:
            say("Test finished.")
    finally:
        if buds is not None:
            buds.close()
        (out / "mic_test.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
