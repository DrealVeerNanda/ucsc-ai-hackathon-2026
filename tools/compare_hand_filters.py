"""What does each way of smoothing the hand's position cost in delay, and buy in steadiness?

    python tools/compare_hand_filters.py [--fps 30] [--noise-mm 4]

A simulated hand: still for a second, a 40 cm reach in one second with a smooth start and
stop, still again; the tracker adds jitter to every frame. For each filter:

    jitter   frame-to-frame movement of a hand that is not moving, as a share of the tracker's
    delay    how far behind the true position the output runs in the middle of the reach, in ms
    past     how far it carries on past the place the hand stops, in mm

"average" is what perception.py has always used (the mean of the last 3 frames).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

import hands  # noqa: E402
from perception import _Average  # noqa: E402


def reach(fps: float, noise_m: float, seed: int):
    rng = np.random.default_rng(seed)
    t = np.arange(int(3 * fps)) / fps
    u = np.clip(t - 1.0, 0, 1)
    x = 0.40 * (10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5)
    true = np.stack([x, 0 * x, 0.3 + 0 * x], 1)
    return t, true, true + rng.normal(0, noise_m, true.shape)


def measure(make, fps: float, noise_m: float, runs: int = 30):
    jitter, delay, past = [], [], []
    for seed in range(runs):
        t, true, noisy = reach(fps, noise_m, seed)
        f = make()
        out = np.array([f.add(p, ti) if not isinstance(f, _Average) else f.add(p) for p, ti in zip(noisy, t)])
        still = t < 0.95
        jitter.append(np.linalg.norm(np.diff(out[still], axis=0), axis=1).mean()
                      / np.linalg.norm(np.diff(noisy[still], axis=0), axis=1).mean())
        mid = (t > 1.25) & (t < 1.75)
        speed = np.gradient(true[:, 0], t)
        delay.append(1000 * np.mean((true[mid, 0] - out[mid, 0]) / speed[mid]))
        after = t > 2.0
        past.append(1000 * max(0.0, float(np.max(out[after, 0] - 0.40))))
    return np.mean(jitter), np.mean(delay), np.mean(past)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--noise-mm", type=float, default=4.0, help="the tracker's jitter per axis")
    a = ap.parse_args()
    print(f"{a.fps:.0f} frames a second, {a.noise_mm:.0f} mm of tracker jitter")
    print(f"{'filter':<12}{'jitter':>8}{'delay ms':>10}{'past mm':>9}")
    for name, make in (("none", lambda: _Average(1, 0.15)), ("average", lambda: _Average(3, 0.15)),
                       ("one_euro", hands.OneEuro), ("kalman", hands.Kalman)):
        j, d, p = measure(make, a.fps, a.noise_mm / 1000)
        print(f"{name:<12}{j:>7.0%}{d:>10.0f}{p:>9.1f}")


if __name__ == "__main__":
    main()
