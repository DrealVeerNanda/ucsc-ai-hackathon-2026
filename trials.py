"""Time blindfolded trials into trials.csv.

    python trials.py baseline --tester veer --object mug    # no help, do these tonight
    python trials.py guided --tester sam --object mug        # with the system
    python trials.py summary                                 # baseline vs guided

Enter starts the clock, Enter again when the object is in hand, x if they gave up.
Anything over --cap seconds counts as a failure.
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import time

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trials.csv")
FIELDS = ["trial", "condition", "tester", "object", "start_spot", "time_s", "success", "notes", "timestamp"]


def read_rows() -> list[dict]:
    if not os.path.exists(PATH):
        return []
    with open(PATH, newline="") as f:
        return list(csv.DictReader(f))


def append(row: dict) -> None:
    new = not os.path.exists(PATH)
    with open(PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def summary() -> None:
    rows = read_rows()
    if not rows:
        print("no trials yet")
        return
    for cond in sorted({r["condition"] for r in rows}):
        rs = [r for r in rows if r["condition"] == cond]
        ok = [float(r["time_s"]) for r in rs if r["success"] == "1"]
        rate = len(ok) / len(rs)
        med = f"{statistics.median(ok):.1f} s" if ok else "-"
        print(f"{cond:<10} n={len(rs):<3} success {rate:4.0%}   median time (successes) {med}")


def run(args) -> None:
    n = sum(1 for r in read_rows() if r["condition"] == args.condition)
    print(f"{args.condition} trials, tester {args.tester}, object {args.object}, cap {args.cap:.0f} s")
    while True:
        cmd = input(f"\ntrial {n + 1}: Enter to start, q to stop > ").strip().lower()
        if cmd == "q":
            break
        t0 = time.monotonic()
        end = input("  timing... Enter when grabbed, x if gave up > ").strip().lower()
        dt = time.monotonic() - t0
        success = end != "x" and dt <= args.cap
        notes = input(f"  {dt:.1f} s, {'success' if success else 'FAIL'}. notes (optional) > ").strip()
        n += 1
        append({"trial": n, "condition": args.condition, "tester": args.tester, "object": args.object,
                "start_spot": args.spot, "time_s": f"{dt:.2f}", "success": int(success),
                "notes": notes, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")})
    summary()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("condition", help="baseline, guided, or summary")
    ap.add_argument("--tester", default="")
    ap.add_argument("--object", default="mug")
    ap.add_argument("--spot", default="A", help="which taped start spot")
    ap.add_argument("--cap", type=float, default=60.0)
    args = ap.parse_args()
    summary() if args.condition == "summary" else run(args)


if __name__ == "__main__":
    main()
