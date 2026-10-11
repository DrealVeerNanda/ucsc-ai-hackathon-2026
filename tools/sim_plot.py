"""Figure for tools/sim_lidar_vs_camera.py: what lidar buys, and which camera error costs it."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
LIDAR, CAMERA = "#2a78d6", "#eb6834"      # categorical slots 1 and 2
SUCCESS_CM = 6.0


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _bars(ax, rows, title):
    names = [r[0] for r in rows][::-1]
    vals = [r[1] * 100 for r in rows][::-1]
    cols = [r[2] for r in rows][::-1]
    ci = [r[3] * 100 for r in rows][::-1]
    y = np.arange(len(rows))
    ax.barh(y, vals, height=0.55, color=cols, edgecolor=SURFACE, linewidth=2)
    ax.errorbar(vals, y, xerr=ci, fmt="none", ecolor=INK2, elinewidth=1, capsize=2)
    for yi, v, c in zip(y, vals, ci):
        ax.text(v + c + 1.5, yi, f"{v:.0f}%", va="center", ha="left", fontsize=9, color=INK)
    ax.set_yticks(y, names, fontsize=9, color=INK)
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    _style(ax)


def plot(summaries, sweep, path: Path):
    by = {s["condition"]: s for s in summaries}
    fig = plt.figure(figsize=(12, 8.2), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.42, left=0.17, right=0.97, top=0.84, bottom=0.08)
    fig.suptitle("Blindfold reach, simulated: lidar + camera against camera only",
                 x=0.03, ha="left", fontsize=14, color=INK, fontweight="bold")
    n = summaries[0]["trials"]
    fig.text(0.03, 0.915, f"{n} reaches per condition, the same mugs, starts and wearer in every condition. The real perception.py "
             f"and guidance.py decide every cue;\nonly the depth sensor is simulated. Success: the real fingertip is within "
             f"{SUCCESS_CM:.0f} cm of the mug when guidance says \"grab\". Bars show 95% intervals.",
             fontsize=9, color=INK2, linespacing=1.5)

    def row(name, label, color):
        s = by[name]
        return label, s["success"], color, s["success_ci95"]

    ax = fig.add_subplot(gs[0, 0])
    _bars(ax, [row("lidar + camera", "Lidar + camera", LIDAR), row("lidar, 3x noise", "Lidar, 3x the noise", LIDAR),
               row("camera only", "Camera only", CAMERA), row("camera, no error", "Camera, perfect depth", MUTED)],
          "Grab success")

    ax = fig.add_subplot(gs[0, 1])
    labels = {"scale": "Overall scale", "flicker": "Scale flicker", "relative": "Hand vs mug offset",
              "bleed": "Edge bleed", "pixel": "Pixel noise"}
    rows = [row(f"camera, {k} only", v, CAMERA) for k, v in labels.items()]
    rows += [row("camera, all but relative", "All but hand vs mug", CAMERA), row("camera only", "All together", CAMERA)]
    _bars(ax, rows, "One camera error at a time (dashed: lidar)")
    ax.axvline(by["lidar + camera"]["success"] * 100, color=LIDAR, linewidth=1.5, linestyle=(0, (4, 3)))

    ax = fig.add_subplot(gs[1, 0])
    xs = [0.0] + [f for f in sweep]
    ys = [by["camera, no error"]["success"]] + [by["camera only" if f == 1.0 else f"camera, errors x{f}"]["success"]
                                                 for f in sweep]
    ax.plot(xs, np.array(ys) * 100, color=CAMERA, linewidth=2, marker="o", markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=2, label="Camera only")
    ax.axhline(by["lidar + camera"]["success"] * 100, color=LIDAR, linewidth=2, label="Lidar + camera")
    ax.set_xlabel("Camera depth error, times the assumed size", fontsize=9, color=INK2)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
    ax.set_title("Grab success as camera depth gets better or worse", loc="left", fontsize=11, color=INK,
                 fontweight="bold")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower left")
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)

    ax = fig.add_subplot(gs[1, 1])
    bins = np.arange(0, 26, 1.0)
    for name, label, color in (("lidar + camera", "Lidar + camera", LIDAR), ("camera only", "Camera only", CAMERA)):
        m = np.clip(np.array(by[name]["misses_m"]) * 100, 0, 25)
        ax.hist(m, bins=bins, color=color, histtype="step", linewidth=2, label=label)
    ax.axvline(SUCCESS_CM, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(SUCCESS_CM + 0.4, ax.get_ylim()[1] * 0.55, "6 cm: the hand\ncloses on the mug", fontsize=8, color=INK2)
    ax.set_xlabel("Real fingertip-to-mug distance at \"grab\", cm (last bin: 25 or more)", fontsize=9,
                  color=INK2)
    ax.set_ylabel("Reaches", fontsize=9, color=INK2)
    ax.set_title("Where the hand actually is at \"grab\"", loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper right")
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
