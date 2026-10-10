"""Read a trace written by tracing.py and show where the time goes, slowest step first.

  python tools/trace_report.py                    # the newest run in logs/
  python tools/trace_report.py logs/trace_X.jsonl
  python tools/trace_report.py --list             # every run, newest first
  python tools/trace_report.py --csv out/trace.csv   # one row per frame, for a spreadsheet
  python tools/trace_report.py --skip 30          # ignore the first 30 frames (warm-up)

What the trace cannot see is listed at the bottom of the report: the fixed part of the camera
and cable delay (measure it with tools/measure_camera_latency.py) and Bluetooth earbud delay.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REST = "rest of the loop (view, dashboard, logs)"


def load(path):
    meta, frames, events = {}, [], []
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue                      # a half-written last line of a run that was killed
            kind = rec.pop("type", None)
            if kind == "meta":
                meta.update(rec)
            elif kind == "frame":
                frames.append(rec)
            elif kind == "event":
                events.append(rec)
    return meta, frames, events


def _stats(values):
    v = np.array([x for x in values if x is not None], dtype=float)
    if v.size == 0:
        return None
    return {"n": int(v.size), "median": float(np.median(v)), "p95": float(np.percentile(v, 95)),
            "max": float(v.max()), "total": float(v.sum())}


def summarize(meta, frames, events, skip: int = 0):
    frames = frames[skip:]
    out = {"meta": meta, "frames": len(frames), "steps": [], "events": [], "audio": None}
    if not frames:
        return out
    seconds = frames[-1]["t"] - frames[0]["t"]
    out["seconds"] = seconds
    out["fps"] = (len(frames) - 1) / seconds if seconds > 0 else 0.0
    skipped = sum(f.get("skipped", 0) for f in frames)
    out["unused"] = skipped / (skipped + len(frames))

    loop = _stats(f.get("loop_ms") for f in frames)
    names = sorted({k for f in frames for k in f.get("spans", {})})
    spans = {n: _stats(f.get("spans", {}).get(n, 0.0) for f in frames) for n in names}
    rest = _stats(max(0.0, f.get("loop_ms", 0.0) - sum(f.get("spans", {}).values())) for f in frames)
    for name, st in [*spans.items(), (REST, rest)]:
        out["steps"].append({"name": name, **st, "share": st["total"] / loop["total"] if loop["total"] else 0.0})
    out["steps"].sort(key=lambda s: -s["total"])
    out["loop"] = loop
    out["idle"] = _stats(f.get("idle_ms") for f in frames)
    out["idle_share"] = out["idle"]["total"] / (out["idle"]["total"] + loop["total"])
    out["wait"] = _stats(f.get("wait_ms") for f in frames)
    out["cable"] = _stats(f.get("cable_ms") for f in frames)
    out["e2e"] = _stats(f.get("e2e_ms") for f in frames)

    by_name = {}
    for e in events:
        by_name.setdefault(e["name"], []).append(e)
    for name, evs in sorted(by_name.items()):
        if name == "audio":
            out["audio"] = {"underruns": sum(e.get("underruns", 0) for e in evs),
                            "render_max_ms": max(e.get("render_max_ms", 0.0) for e in evs)}
        else:
            st = _stats(e.get("ms") for e in evs)
            if st:
                out["events"].append({"name": name, **st})
    out["bottleneck"] = out["steps"][0] if out["steps"] else None
    return out


def render(s, path="") -> str:
    m = s["meta"]
    lines = [f"Trace {path}".rstrip()]
    facts = [f"{k} {m[k]}" for k in ("source", "detector", "hand") if m.get(k)]
    if m.get("audio_out_ms") is not None:
        facts.append(f"audio buffer {m['audio_out_ms']:.0f} ms ({m.get('audio_device', '?')})")
    if facts:
        lines.append("   ".join(facts))
    if not s["frames"]:
        return "\n".join(lines + ["No frames in this trace."])
    lines.append(f"{s['frames']} frames in {s['seconds']:.1f} s: the loop ran at {s['fps']:.1f} fps. "
                 f"Camera frames that arrived and were never used: {s['unused']:.0%}.")

    def row(label, st, share=None):
        if st is None:
            return f"  {label:<46}{'not measured':>14}"
        bar = "" if share is None else f"{share:6.0%}  " + "#" * int(round(share * 30))
        return f"  {label:<46}{st['median']:8.1f}{st['p95']:8.1f}{st['max']:9.1f}  {bar}"

    lines += ["", f"  {'Each frame, in milliseconds':<46}{'median':>8}{'95%':>8}{'max':>9}   share of the loop"]
    lines.append(row("phone to GX10, extra over the quickest frame", s["cable"]))
    lines.append(row("arrived, waiting for the loop to pick it up", s["wait"]))
    for st in s["steps"]:
        lines.append(row(st["name"], st, st["share"]))
    lines.append(row("LOOP TOTAL", s["loop"]))
    lines.append(row("then idle, waiting for the next camera frame", s["idle"]))
    lines.append(row("ARRIVED TO SOUND UPDATED", s["e2e"]))

    if s["events"] or s["audio"]:
        lines += ["", f"  {'Not once per frame':<46}{'count':>8}{'median':>10}{'max':>10}"]
        for e in s["events"]:
            lines.append(f"  {e['name']:<46}{e['n']:8d}{e['median']:8.0f} ms{e['max']:7.0f} ms")
        if s["audio"]:
            a = s["audio"]
            lines.append(f"  audio: {a['underruns']} underruns, slowest block took {a['render_max_ms']:.2f} ms to make")

    b = s["bottleneck"]
    lines += ["", f"Biggest step: {b['name']}, {b['share']:.0%} of the loop ({b['median']:.1f} ms a frame)."]
    if s["idle_share"] > 0.25:
        lines.append(f"Frame rate is set by the camera: the loop sits idle {s['idle_share']:.0%} of the time, "
                     "so faster code would not raise it.")
    else:
        lines.append(f"Frame rate is set by processing: the loop is idle only {s['idle_share']:.0%} of the time. "
                     "Speeding up the biggest step raises the frame rate.")
    if s["cable"] is None:
        lines.append("This source sends no timestamps, so the phone-to-GX10 delay is not in the trace.")
    lines.append("Not in any trace: the fixed camera and cable delay (tools/measure_camera_latency.py measures it) "
                 "and Bluetooth earbuds (typically 150-250 ms; wired adds almost nothing).")
    return "\n".join(lines)


def write_csv(frames, path):
    names = sorted({k for f in frames for k in f.get("spans", {})})
    cols = ["i", "t", "state", "cable_ms", "wait_ms", "skipped", *names, "loop_ms", "idle_ms", "e2e_ms"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for fr in frames:
            w.writerow([fr.get("spans", {}).get(c, "") if c in names else fr.get(c, "") for c in cols])


def traces(directory=ROOT / "logs"):
    return sorted(Path(directory).glob("trace_*.jsonl"), reverse=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", nargs="?", help="a trace file (default: the newest in logs/)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--skip", type=int, default=0, help="ignore this many frames at the start")
    ap.add_argument("--csv", help="also write one row per frame to this file")
    a = ap.parse_args()
    found = traces()
    if a.list:
        for p in found:
            meta, frames, events = load(p)
            print(f"{p.relative_to(ROOT)}   {len(frames):6d} frames   {meta.get('source', '')}")
        sys.exit(0)
    path = Path(a.trace) if a.trace else (found[0] if found else None)
    if path is None or not path.exists():
        sys.exit("No trace found. Run main.py (or any tool that reads the stream) first; traces go to logs/.")
    meta, frames, events = load(path)
    print(render(summarize(meta, frames, events, a.skip), path))
    if a.csv:
        write_csv(frames[a.skip:], a.csv)
        print(f"wrote {a.csv}")
