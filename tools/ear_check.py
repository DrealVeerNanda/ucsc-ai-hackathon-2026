"""The two checks only the wearer can judge, run by the buttons on the launcher's page. Stop
the live run first; both want the AirPods.

  python tools/ear_check.py sound     # says one sentence through the default output
  python tools/ear_check.py mic       # says what to say, beeps, records 3 s, says back what it heard

The last line printed is the verdict as JSON, {"ok": true | false | null, "message": "..."}; null
means "played, only your ears can tell". With GUIDANCE_AUDIO=off nothing is played: the sentence
is written to out/ear_check.wav instead, and the microphone is recorded without the prompts.
Built from the pieces of tools/mic_test.py, which is the longer version for the terminal.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TRACE", "0")
import voice  # noqa: E402
from audio_out import synthesize, write_wav  # noqa: E402
from tools import mic_test  # noqa: E402

SOUND_ON = os.environ.get("GUIDANCE_AUDIO", "on") != "off"
SENTENCE = "This is Proprio. If you can hear me in both ears, the sound works."
SECONDS = 3.0


def say(text: str) -> bool:
    samples = synthesize(text, 48000)
    if samples is None:
        return False
    if SOUND_ON:
        mic_test.play(samples)
    else:
        (ROOT / "out").mkdir(exist_ok=True)
        write_wav(ROOT / "out" / "ear_check.wav", np.stack([samples, samples], axis=1), 48000)
    return True


def sound() -> dict:
    if not say(SENTENCE):
        return {"ok": False, "message": "The voice could not speak: the Piper voice file is missing (bash tools/fetch_models.sh piper)."}
    if not SOUND_ON:
        return {"ok": None, "message": "Sound is switched off (GUIDANCE_AUDIO=off): the sentence was written to out/ear_check.wav, not played."}
    return {"ok": None, "message": f"Played: \"{SENTENCE}\" Did you hear it in both ears? If not, check the two lines above."}


def mic() -> dict:
    buds = voice.Headset.find()
    if buds is None and not mic_test.sources():
        return {"ok": False, "message": "There is no microphone: the AirPods are not connected, or have no microphone mode."}
    whisper = voice.load_whisper()                    # the same one the live run uses
    try:
        if buds is not None:
            buds._profile(buds.headset)
            if mic_test.wait_for_microphone(buds.name) is None:
                return {"ok": False, "message": f"{buds.name} switched to microphone mode but no microphone appeared. Press Reconnect."}
            time.sleep(0.5)
        if SOUND_ON:
            say("After the beep, say: blue cup.")
            mic_test.play(mic_test.beep())
        blocks, lost = mic_test.record(SECONDS)
        audio = np.concatenate(blocks) if blocks else np.zeros(0, np.float32)
        m = mic_test.measure(blocks, SECONDS)
        text = whisper(audio) if audio.size > voice.SR // 4 else ""
        result = mic_test.verdict(m, lost, text, "blue cup")
        if SOUND_ON:
            say(f"I heard: {text}" if text else "I heard nothing.")
    finally:
        if buds is not None:
            buds.close()
    heard = f"Heard \"{text}\"" if text else "Heard nothing"
    return {"ok": result == "heard correctly", "heard": text, "message": f"{heard}: {result}."}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    if which not in ("sound", "mic"):
        sys.exit(__doc__)
    try:
        got = sound() if which == "sound" else mic()
    except Exception as e:
        got = {"ok": False, "message": f"The test failed ({e.__class__.__name__}: {e})."}
    print(json.dumps(got), flush=True)


if __name__ == "__main__":
    main()
