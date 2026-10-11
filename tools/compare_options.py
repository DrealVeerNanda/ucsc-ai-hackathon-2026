"""Which way of guiding works best? Reads the traces in logs/ and, for every completed reach,
takes the option that was in use, how long it took from the first guidance to "grab", and how
many directions had to be spoken.

  python tools/compare_options.py                 # every trace in logs/
  python tools/compare_options.py --since 19:30   # only runs started after that time today
  python tools/compare_options.py logs/trace_X.jsonl

A reach starts when guidance first has both the object and the hand, and ends at "grab". Reaches
abandoned for a new target are counted as unfinished.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import guidance  # noqa: E402

DIRECTIONS = {"left", "right", "up", "down", "forward", "back"}


def is_direction(text: str) -> bool:
    words = set(text.lower().replace(" and ", " ").split())
    return bool(words) and (words <= DIRECTIONS or text.endswith("o'clock"))


def reaches(events):
    """[(option, seconds or None if unfinished, directions spoken, target)] from one run's events."""
    out, start, said, option, target = [], None, 0, "tones", None
    for e in events:
        name = e.get("name")
        if name == "option":
            option = e.get("option", option)
        elif name == "say" and start is not None and is_direction(e.get("text", "")):
            said += 1
        elif name == "state":
            option = e.get("option", option)
            state = e.get("state")
            if state == "searching":                      # a new target: anything in progress was abandoned
                if start is not None:
                    out.append((option, None, said, target))
                start, said, target = None, 0, e.get("target")
            elif state == "guiding" and start is None:
                start, said, target = e["t"], 0, e.get("target", target)
            elif state == "grab" and start is not None:
                out.append((option, e["t"] - start, said, target))
                start, said = None, 0
    return out


def load(paths):
    rows = []
    for p in paths:
        events = []
        for line in open(p):
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("type") == "event":
                events.append(rec)
        rows += reaches(events)
    return rows


def table(rows) -> str:
    keys = [k for k, _, _ in guidance.OPTIONS]
    names = {k: f"{i + 1} {title}" for i, (k, title, _) in enumerate(guidance.OPTIONS)}
    lines = [f"{'option':<30}{'reaches':>8}{'unfinished':>11}{'median':>9}{'fastest':>9}{'slowest':>9}{'directions spoken':>19}"]
    for key in keys + sorted({r[0] for r in rows} - set(keys)):
        mine = [r for r in rows if r[0] == key]
        if not mine:
            continue
        done = np.array([r[1] for r in mine if r[1] is not None])
        said = np.array([r[2] for r in mine if r[1] is not None])
        if done.size:
            lines.append(f"{names.get(key, key):<30}{done.size:>8}{len(mine) - done.size:>11}{np.median(done):>7.1f} s{done.min():>7.1f} s"
                         f"{done.max():>7.1f} s{np.median(said):>12.0f} a reach")
        else:
            lines.append(f"{names.get(key, key):<30}{0:>8}{len(mine):>11}")
    if len(lines) == 1:
        return "No completed reaches in these traces yet."
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="*", help="trace files (default: all in logs/)")
    ap.add_argument("--since", help="HH:MM today: ignore runs that started earlier")
    a = ap.parse_args()
    paths = [Path(p) for p in a.traces] or sorted((ROOT / "logs").glob("trace_*.jsonl"))
    if a.since:
        stamp = time.strftime("%Y%m%d_") + a.since.replace(":", "") + "00"
        paths = [p for p in paths if p.stem.replace("trace_", "") >= stamp]
    print(table(load(paths)))
