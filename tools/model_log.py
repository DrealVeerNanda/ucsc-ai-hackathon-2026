"""Watch the models work: one line for every request to Whisper, Nemotron and Cosmos, with what
was asked, what came back and how long it took. It follows the newest trace in logs/ and moves
to the next run by itself.

  python tools/model_log.py            # follow live
  python tools/model_log.py --all      # start from the beginning of the newest run

The servers' own logs (tokens, tokens per second) are in ~/models/nemotron.log and
~/models/cosmos_*.log; bash tools/show_models.sh puts all three on the GX10's screen.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
COLOUR = {"WHISPER": "\033[36m", "NEMOTRON": "\033[32m", "COSMOS": "\033[35m", "DETECTOR": "\033[33m", "LOAD": "\033[2m"}


def line(e: dict) -> str | None:
    """One trace event -> one readable line, or None for events that are not model calls."""
    name, ms = e.get("name", ""), e.get("ms")
    took = "" if ms is None else (f"{ms / 1000:5.2f} s" if ms >= 100 else f"{ms:4.0f} ms")
    if name == "whisper":
        who, what = "WHISPER", f'heard "{e.get("text", "")}"  ({e.get("audio_s", "?")} s of speech)'
    elif name == "nemotron":
        intent, target = e.get("intent", "pick"), e.get("target", "")
        answer = {"pick": f"wants: {target}" if target else "wants: (nothing named)", "list": "asks what is there",
                  "none": "not a request"}.get(intent, intent)
        who, what = "NEMOTRON", f'"{e.get("text", "")}"  ->  {answer}'
    elif name == "cosmos.list":
        objs = e.get("objects") or []
        who, what = "COSMOS", "looked at the camera  ->  " + (", ".join(objs) if objs else "nothing to pick up")
    elif name == "cosmos.find":
        who, what = "COSMOS", f'where is the {e.get("target", "?")}?  ->  ' + ("found, box returned" if e.get("found") else "not visible")
    elif name == "detector.set_target":
        who, what = "DETECTOR", f'now searching for "{e.get("target", "")}"'
    elif name.startswith("load."):
        who, what = "LOAD", f'{name[5:]} ({e.get("kind") or e.get("model") or ""}) ready'
    else:
        return None
    stamp = time.strftime("%H:%M:%S", time.localtime(e.get("t", time.time())))
    return f"{DIM}{stamp}{RESET}  {COLOUR[who]}{BOLD}{who:<9}{RESET} {took:>8}  {what}"


def newest(directory: Path):
    files = sorted(directory.glob("trace_*.jsonl"))
    return files[-1] if files else None


def follow(directory: Path, from_start: bool):
    path, f = None, None
    while True:
        latest = newest(directory)
        if latest is not None and latest != path:
            if f:
                f.close()
            path, f = latest, open(latest)
            if not from_start:
                f.seek(0, 2)                       # only what happens from now on
            from_start = True                      # later runs are shown in full
            print(f"{DIM}--- following {path.name} ---{RESET}", flush=True)
        text = f.readline() if f else ""
        if not text or not text.endswith("\n"):
            time.sleep(0.2)
            continue
        if '"type":"event"' not in text and '"type": "event"' not in text:
            continue
        try:
            out = line(json.loads(text))
        except ValueError:
            continue
        if out:
            print(out, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="show the newest run from its start")
    ap.add_argument("--dir", default=str(ROOT / "logs"))
    a = ap.parse_args()
    print(f"{BOLD}What each model was asked and answered{RESET}  {DIM}(whisper = speech to text, "
          f"nemotron = understands the request, cosmos = looks at the picture){RESET}", flush=True)
    try:
        follow(Path(a.dir), a.all)
    except KeyboardInterrupt:
        sys.exit(0)
