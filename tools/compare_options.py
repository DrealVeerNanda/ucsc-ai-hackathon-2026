"""Which way of guiding works best? Reads the traces in logs/ and, for every completed reach,
takes the option that was in use, how long it took from the first guidance to "grab", and how
many directions had to be spoken.

  python tools/compare_options.py                 # every trace in logs/
  python tools/compare_options.py --since 19:30   # only runs started after that time today
  python tools/compare_options.py logs/trace_X.jsonl

A reach starts when guidance first has both the object and the hand, and ends at "grab". Reaches
abandoned for a new target are counted as unfinished.

A second table is about "Grab" itself, by the rule in use (guidance.GRAB_OPTIONS): how long the
hand had already been in grabbing distance when it was said. Long waits are what "I was touching
it and it said nothing" looks like in the numbers. It needs the per-frame fields guidance writes
(zone, trusted), so runs from before 11 Oct are left out of it.
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


def grabs(frames):
    """[(grab rule, seconds the hand had been in grabbing distance when "Grab" came, seconds of
    those in which the reading was not believed)] from one run's frames."""
    out, since, doubted, prev = [], None, 0.0, None
    for f in frames:
        state, zone = f.get("state"), f.get("zone")
        if state == "guiding" and zone is not None:
            if zone < 1.0:
                since = f["t"] if since is None else since
                doubted += min(0.2, f["t"] - prev["t"]) if prev is not None and not f.get("trusted", 1) else 0.0
            elif zone > 2.0:
                since, doubted = None, 0.0                # well away again: the wait starts over
        elif state == "grab" and prev is not None and prev.get("state") != "grab":
            if "grab_option" in f:
                out.append((f["grab_option"], 0.0 if since is None else f["t"] - since, doubted))
            since, doubted = None, 0.0
        elif state in ("searching", "idle"):
            since, doubted = None, 0.0
        prev = f
    return out


def load(paths, frames_too=False):
    rows, waits = [], []
    for p in paths:
        events, frames = [], []
        for line in open(p):
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("type") == "event":
                events.append(rec)
            elif rec.get("type") == "frame" and "state" in rec:
                frames.append(rec)
        rows += reaches(events)
        waits += grabs(frames)
    return (rows, waits) if frames_too else rows


def grab_table(waits) -> str:
    if not waits:
        return ""
    names = {k: f"{i + 1} {title}" for i, (k, title, _) in enumerate(guidance.GRAB_OPTIONS)}
    lines = [f"{'grab rule':<24}{'grabs':>6}{'in grabbing distance before Grab: median':>42}{'longest':>9}{'of it not believed':>20}"]
    for key in [k for k, _, _ in guidance.GRAB_OPTIONS]:
        mine = [w for w in waits if w[0] == key]
        if mine:
            t = np.array([w[1] for w in mine])
            lines.append(f"{names[key]:<24}{len(mine):>6}{np.median(t):>40.1f} s{t.max():>7.1f} s{np.median([w[2] for w in mine]):>18.1f} s")
    return "\n".join(lines)


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
    rows, waits = load(paths, frames_too=True)
    print(table(rows))
    if waits:
        print("\n" + grab_table(waits))
