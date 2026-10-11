"""Simulated blindfold reaches: lidar + camera against camera only (depth estimated from colour).

    python tools/sim_lidar_vs_camera.py                 # 300 reaches per condition, about a minute
    python tools/sim_lidar_vs_camera.py --trials 1000 --plot out/sim/lidar_vs_camera.png

Only the depth sensor is simulated. Everything that decides what the wearer hears is the real
code: perception.Perception (box depth, fingertip depth, covered-object memory, size fallback,
smoothing) and guidance.Guidance (the 5 cm / 0.3 s grab rule). Detector and hand tracker are
stand-ins that report the true box and fingertip with a few pixels of noise, the same in every
condition, so any difference comes from the depth.

Each trial: a mug stands on the table 40-65 cm in front of a chest camera tilted down 30
degrees; the hand starts near the chest; a simulated wearer moves it the way the cues say,
with a 0.2 s reaction delay and some motor noise, until guidance says "grab" or 20 s pass.
A grab is a success when the real fingertip is within 6 cm of the mug's front face.

Depth models:

  lidar     the true depth seen through the iPhone lidar: 256 x 192 grid (edge pixels blend
            foreground and background), about 1 cm noise at 1 m, no reading under 20 cm.
  camera    the true depth through the error sources of monocular depth estimation:
              scale      one wrong overall scale per session (the camera cannot know the size of the room)
              flicker    that scale wobbles from frame to frame
              relative   the hand and the object are each off by their own amount, after the
                         overall scale: a close, moving hand gives the model little to go on
              bleed      depth is smooth across edges, so a hand next to the mug takes on
                         some of the mug's depth and the other way round
            The sizes are assumptions, set in CameraErrors below. Not yet measured: compare a depth
            model with the lidar on the same phone frames, and pass the sizes as JSON with --camera-errors.

The camera condition is also run with one error source at a time, and with the relative error
scaled from 0 to 2x, to show which source costs what.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, replace
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("TRACE", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import guidance  # noqa: E402
import perception  # noqa: E402
from audio_out import PrintSink  # noqa: E402
from interfaces import Frame  # noqa: E402

W, H = 360, 480                 # portrait camera image
LIDAR_W, LIDAR_H = 192, 256     # iPhone lidar depth grid, portrait
FPS = 15.0
DT = 1.0 / FPS
TILT = math.radians(30.0)       # chest mount tilted down toward the table
TABLE_BELOW = 0.45              # table surface below the lens, metres
FAR = 3.0
MUG_W, MUG_H = 0.08, 0.10
SUCCESS_M = 0.06                # real fingertip within this of the mug at "grab": the hand closes on it
TIMEOUT_S = 20.0


def make_K(w=W, h=H, hfov_deg=63.0):
    f = (w / 2) / math.tan(math.radians(hfov_deg) / 2)
    return np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1]], dtype=np.float64)


K = make_K()


def project(p):
    return K[0, 0] * p[0] / p[2] + K[0, 2], K[1, 1] * p[1] / p[2] + K[1, 2]


def table_depth():
    yn = (np.arange(H, dtype=np.float64)[:, None] - K[1, 2]) / K[1, 1]
    denom = math.cos(TILT) * yn + math.sin(TILT)
    with np.errstate(divide="ignore"):
        z = np.where(denom > 0, TABLE_BELOW / denom, FAR)
    return np.broadcast_to(np.minimum(z, FAR), (H, W)).astype(np.float32).copy()


TABLE = table_depth()
UU, VV = np.meshgrid(np.arange(W), np.arange(H))


# ---------------------------------------------------------------------------
# Sensor models

@dataclass
class LidarModel:
    min_m: float = 0.20          # no reading closer than this
    noise_m: float = 0.003       # noise sigma = noise_m + noise_per_m * depth
    noise_per_m: float = 0.007


@dataclass
class CameraErrors:
    scale: float = 0.15          # sigma of log overall scale, per session
    flicker: float = 0.03        # sigma of log overall scale wobble, frame to frame (AR(1))
    rel_obj: float = 0.05        # sigma of log error of the object relative to the scene, per session
    rel_hand: float = 0.08       # same for hand and arm
    hand_flicker: float = 0.03   # hand's relative error wobble, frame to frame (AR(1))
    bleed_px: float = 4.0        # Gaussian smoothing of depth across edges, pixels at 360 px wide
    pixel: float = 0.01          # per-pixel noise, fraction of depth
    rho: float = 0.8             # frame-to-frame correlation of the wobbles

    def only(self, *keep):
        """These error sources at their size, the rest switched off."""
        groups = {"scale": ("scale",), "flicker": ("flicker",), "relative": ("rel_obj", "rel_hand", "hand_flicker"),
                  "bleed": ("bleed_px",), "pixel": ("pixel",)}
        names = {n for k in keep for n in groups[k]}
        return replace(self, **{f: (getattr(self, f) if f in names else 0.0)
                                for g in groups.values() for f in g})

    def scaled(self, k: float):
        """Every error source k times its size."""
        return replace(self, **{f: getattr(self, f) * k for f in
                                ("scale", "flicker", "rel_obj", "rel_hand", "hand_flicker", "bleed_px", "pixel")})


# ---------------------------------------------------------------------------
# The world

@dataclass
class World:
    mug: np.ndarray          # centre of the front face, camera frame
    hand0: np.ndarray        # fingertip start


def make_world(rng) -> World:
    """A mug on the table and a start for the hand. A start where the arm hides the mug is drawn
    again: the camera could not find the mug from there with either sensor."""
    while True:
        z = rng.uniform(0.40, 0.65)
        x = rng.uniform(-0.15, 0.15)
        y_bottom = (TABLE_BELOW - math.sin(TILT) * z) / math.cos(TILT)    # where the table is at that depth
        mug = np.array([x, y_bottom - MUG_H / 2, z])
        hand0 = np.array([x + rng.uniform(0.0, 0.15), rng.uniform(0.05, 0.12), rng.uniform(0.22, 0.30)])
        _, ids, box, _ = render(mug, hand0)
        inside = ids[max(0, int(box[1])):max(0, int(box[3])), max(0, int(box[0])):max(0, int(box[2]))]
        if inside.size and np.mean(inside == 2) < 0.2:
            return World(mug, hand0)


SHOULDER = np.array([0.18, 0.10, -0.05])   # right shoulder, beside and behind the chest camera


def hand_points(tip):
    """3D points of the reaching hand, from the fingertip back along the arm to the shoulder."""
    tip = np.asarray(tip, float)
    d = tip - SHOULDER
    d /= np.linalg.norm(d)
    f = d + np.array([0.0, 0.7, 0.0])         # the finger bends down toward the table
    f /= np.linalg.norm(f)
    side = np.cross(d, [0.0, -1.0, 0.0])
    side /= np.linalg.norm(side)
    knuckle = tip - 0.08 * f
    wrist = knuckle - 0.09 * d
    return {"tip": tip, "dip": tip - 0.025 * f, "pip": tip - 0.05 * f, "knuckle": knuckle,
            "k_in": knuckle - 0.04 * side, "k_out": knuckle + 0.04 * side, "wrist": wrist,
            "palm_end": knuckle - 0.035 * d,
            "elbow": SHOULDER + 0.5 * (wrist - SHOULDER) + np.array([0.0, 0.10, 0.0])}


def _paint_disc(z, ids, u, v, r, depth, ident):
    u0, u1 = max(0, int(u - r)), min(W, int(u + r) + 2)
    v0, v1 = max(0, int(v - r)), min(H, int(v + r) + 2)
    if u0 >= u1 or v0 >= v1:
        return
    m = (UU[v0:v1, u0:u1] - u) ** 2 + (VV[v0:v1, u0:u1] - v) ** 2 <= r * r
    sub = z[v0:v1, u0:u1]
    m &= depth < sub
    sub[m] = depth
    ids[v0:v1, u0:u1][m] = ident


def _paint_limb(z, ids, a, b, width_m, ident, n=14):
    for s in np.linspace(0, 1, n):
        p = a + s * (b - a)
        if p[2] <= 0.03:
            continue
        u, v = project(p)
        _paint_disc(z, ids, u, v, K[0, 0] * width_m / 2 / p[2], p[2], ident)


def render(mug, tip):
    """True depth (metres) and which thing each pixel shows: 0 table, 1 mug, 2 hand or arm."""
    z = TABLE.copy()
    ids = np.zeros((H, W), np.uint8)
    u0, v0 = project(mug + [-MUG_W / 2, -MUG_H / 2, 0])
    u1, v1 = project(mug + [MUG_W / 2, MUG_H / 2, 0])
    box = (u0, v0, u1, v1)
    ua, ub, va, vb = max(0, int(u0)), min(W, int(u1)), max(0, int(v0)), min(H, int(v1))
    sub = z[va:vb, ua:ub]
    m = mug[2] < sub
    sub[m] = mug[2]
    ids[va:vb, ua:ub][m] = 1
    p = hand_points(tip)
    _paint_limb(z, ids, p["elbow"], p["wrist"], 0.07, 2)            # forearm
    _paint_limb(z, ids, p["wrist"], p["palm_end"], 0.06, 2)         # palm, ending short of the knuckles
    _paint_limb(z, ids, p["k_in"], p["k_out"], 0.02, 2, n=8)        # knuckle row
    _paint_limb(z, ids, p["knuckle"], p["tip"], 0.018, 2)           # index finger
    return z, ids, box, p


def lidar_depth(z_true, model: LidarModel, rng):
    small = cv2.resize(z_true, (LIDAR_W, LIDAR_H), interpolation=cv2.INTER_AREA)   # edge pixels blend
    small = small + rng.normal(0, 1, small.shape).astype(np.float32) * (model.noise_m + model.noise_per_m * small)
    small[small < model.min_m] = 0.0
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)   # as stream.prepare does


class CameraDepth:
    """Monocular estimate of one session: fixed biases drawn once, wobbles evolve per frame."""

    def __init__(self, e: CameraErrors, rng):
        self.e, self.rng = e, rng
        self.log_scale = rng.normal(0, e.scale)
        self.rel_obj = rng.normal(0, e.rel_obj)
        self.rel_hand = rng.normal(0, e.rel_hand)
        self.f_scale = rng.normal(0, e.flicker)
        self.f_hand = rng.normal(0, e.hand_flicker)

    def __call__(self, z_true, ids):
        e, rng, k = self.e, self.rng, math.sqrt(1 - self.e.rho ** 2)
        self.f_scale = e.rho * self.f_scale + k * rng.normal(0, e.flicker)
        self.f_hand = e.rho * self.f_hand + k * rng.normal(0, e.hand_flicker)
        log_err = np.full(z_true.shape, self.log_scale + self.f_scale, np.float32)
        log_err[ids == 1] += self.rel_obj
        log_err[ids == 2] += self.rel_hand + self.f_hand
        d = z_true * np.exp(log_err)
        if e.bleed_px > 0:
            d = cv2.GaussianBlur(d, (0, 0), e.bleed_px)
        if e.pixel > 0:
            d = d * (1 + rng.normal(0, e.pixel, d.shape).astype(np.float32))
        return d.astype(np.float32)


# ---------------------------------------------------------------------------
# Stand-ins for the detector and hand tracker: true box and landmarks plus a few pixels of noise

class SimDetector:
    def __init__(self, rng):
        self.rng, self.box, self.covered = rng, None, 0.0

    def set_target(self, name):
        pass

    def detect(self, rgb):
        if self.covered > 0.5:                       # mostly behind the hand: the detector loses it
            return []
        b = np.array(self.box) + self.rng.normal(0, 2.0, 4)
        return [perception.Detection(tuple(float(v) for v in b), 0.8, "mug")]


class SimHands:
    def __init__(self, rng):
        self.rng, self.p = rng, None

    def detect(self, rgb, t_ms):
        n = lambda: self.rng.normal(0, 2.0, 2)      # noqa: E731  landmark jitter, pixels
        px = {k: np.array(project(v)) + n() for k, v in self.p.items() if k not in ("elbow", "palm_end") and v[2] > 0.03}
        if "tip" not in px:
            return []
        pts = np.array(list(px.values()), np.float32)
        scale = K[0, 0] / self.p["knuckle"][2] * (1 + self.rng.normal(0, 0.05))   # knuckle row width
        samples = [tuple(px[k]) for k in ("tip", "dip", "pip", "knuckle") if k in px]
        return [perception.Hand(tip=tuple(px["tip"]), points=pts, samples=samples, px_per_m=float(scale))]


# ---------------------------------------------------------------------------
# One reach

def deadzone(err, dead=0.025):
    return np.where(np.abs(err) > dead, err - np.sign(err) * dead, 0.0)


def run_trial(seed: int, sensor: str, cam: CameraErrors | None = None, lidar: LidarModel | None = None) -> dict:
    world = make_world(np.random.default_rng(seed))
    user_rng = np.random.default_rng([seed, 1])      # the wearer moves the same way in every condition
    sens_rng = np.random.default_rng([seed, 2])
    det, hands = SimDetector(sens_rng), SimHands(sens_rng)
    perc = perception.Perception(det, hands, perception.PerceptionConfig())
    guide = guidance.Guidance(PrintSink(quiet=True))
    guide.set_grab_option("fingertip")      # the 5 cm rule this study was run with; the live default is now "over"
    guide.reset("mug", now=0.0)
    cam_depth = CameraDepth(cam, sens_rng) if sensor == "camera" else None
    lidar = lidar or LidarModel()
    rgb = np.zeros((H, W, 3), np.uint8)
    tip = world.hand0.copy()
    cues = []                                         # perceived error, for the reaction delay
    delay = int(round(0.2 * FPS))
    err_rows = []
    t = 0.0
    while t < TIMEOUT_S:
        z, ids, box, pts = render(world.mug, tip)
        det.box = box
        bx = (slice(max(0, int(box[1])), max(0, int(box[3]))), slice(max(0, int(box[0])), max(0, int(box[2]))))
        det.covered = float(np.mean(ids[bx] == 2)) if ids[bx].size else 0.0
        hands.p = pts
        depth = lidar_depth(z, lidar, sens_rng) if sensor == "lidar" else cam_depth(z, ids)
        scene = perc.locate(Frame(rgb, depth, K, None, t, t_recv=t), "mug")
        state = guide.update(scene, now=t)
        true_err = world.mug - tip
        perceived = None
        if scene.object_xyz is not None and scene.hand_xyz is not None:
            perceived = np.asarray(scene.object_xyz) - np.asarray(scene.hand_xyz)
            err_rows.append(perceived - true_err)
        if state == guidance.GRAB:
            miss = float(np.linalg.norm(true_err))
            return {"grab": True, "success": miss <= SUCCESS_M, "time_s": t, "miss_m": miss,
                    "miss_z_m": float(true_err[2]), "err_rms": _rms(err_rows)}
        cues.append(perceived)
        cue = cues[-1 - delay] if len(cues) > delay else None
        if cue is not None:
            v = 1.6 * deadzone(cue)
            speed = float(np.linalg.norm(v))
            if speed > 0.25:
                v *= 0.25 / speed
            tip = tip + (v + user_rng.normal(0, 0.015, 3)) * DT
        t += DT
    miss = float(np.linalg.norm(world.mug - tip))
    return {"grab": False, "success": False, "time_s": TIMEOUT_S, "miss_m": miss,
            "miss_z_m": float((world.mug - tip)[2]), "err_rms": _rms(err_rows)}


def _rms(rows):
    if not rows:
        return [None, None, None]
    a = np.array(rows)
    return [float(v) for v in np.sqrt(np.mean(a ** 2, axis=0))]


def _job(args):
    return run_trial(*args)


# ---------------------------------------------------------------------------
def summarize(name, results) -> dict:
    grabs = [r for r in results if r["grab"]]
    ok = [r for r in results if r["success"]]
    rms = np.array([r["err_rms"] for r in results if r["err_rms"][0] is not None])
    n = len(results)
    p = len(ok) / n
    return {
        "condition": name, "trials": n,
        "success": p, "success_ci95": 1.96 * math.sqrt(max(p * (1 - p), 1e-9) / n),
        "grab_rate": len(grabs) / n,
        "wrong_grab": sum(1 for r in grabs if not r["success"]) / n,
        "timeout": sum(1 for r in results if not r["grab"]) / n,
        "median_time_s": float(np.median([r["time_s"] for r in ok])) if ok else None,
        "median_miss_cm": float(np.median([r["miss_m"] for r in grabs]) * 100) if grabs else None,
        "p90_miss_cm": float(np.percentile([r["miss_m"] for r in grabs], 90) * 100) if grabs else None,
        "perceived_err_rms_cm": [float(v) * 100 for v in np.median(rms, axis=0)] if len(rms) else None,
        "misses_m": [r["miss_m"] for r in grabs],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--camera-errors", help="JSON with CameraErrors fields, e.g. measured on the phone")
    ap.add_argument("--out", default="out/sim/lidar_vs_camera.json")
    ap.add_argument("--plot", help="also save a figure here (PNG)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    cam = CameraErrors()
    if args.camera_errors:
        cam = replace(cam, **{k: v for k, v in json.loads(Path(args.camera_errors).read_text()).items()
                              if k in asdict(cam)})
    conditions = [("lidar + camera", "lidar", None),
                  ("lidar, 3x noise", "lidar", None, LidarModel(noise_m=0.009, noise_per_m=0.021)),
                  ("camera only", "camera", cam),
                  ("camera, no error", "camera", cam.only())]
    conditions += [(f"camera, {k} only", "camera", cam.only(k)) for k in ("scale", "flicker", "relative", "bleed", "pixel")]
    conditions += [("camera, all but relative", "camera", cam.only("scale", "flicker", "bleed", "pixel"))]
    sweep = [0.25, 0.5, 0.75, 1.0, 1.5]
    for f in sweep:
        if f != 1.0:
            conditions.append((f"camera, errors x{f}", "camera", cam.scaled(f)))
    conditions = [c if len(c) == 4 else (*c, None) for c in conditions]

    jobs = [(seed, sensor, c, lid) for _, sensor, c, lid in conditions for seed in range(args.trials)]
    print(f"[sim] {len(conditions)} conditions x {args.trials} reaches on {args.workers} workers", file=sys.stderr)
    with Pool(args.workers) as pool:
        flat = pool.map(_job, jobs, chunksize=8)
    summaries = []
    for i, (name, _, c, _) in enumerate(conditions):
        s = summarize(name, flat[i * args.trials:(i + 1) * args.trials])
        s["camera_errors"] = asdict(c) if c else None
        summaries.append(s)

    print(f"\n{'condition':32s} {'success':>12s} {'wrong grab':>10s} {'timeout':>8s} {'time s':>7s} "
          f"{'miss cm':>8s} {'p90 cm':>7s}  perceived error rms x/y/z cm")
    for s in summaries:
        rms = "/".join(f"{v:.1f}" for v in s["perceived_err_rms_cm"]) if s["perceived_err_rms_cm"] else "-"
        fmt = lambda v, f="{:.1f}": "-" if v is None else f.format(v)   # noqa: E731
        print(f"{s['condition']:32s} {s['success']*100:6.1f}% ±{s['success_ci95']*100:4.1f} "
              f"{s['wrong_grab']*100:9.1f}% {s['timeout']*100:7.1f}% {fmt(s['median_time_s']):>7s} "
              f"{fmt(s['median_miss_cm']):>8s} {fmt(s['p90_miss_cm']):>7s}  {rms}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"trials": args.trials, "success_m": SUCCESS_M, "lidar": asdict(LidarModel()),
                               "camera_errors": asdict(cam), "conditions": summaries}, indent=1))
    print(f"\n[sim] results in {out}", file=sys.stderr)
    if args.plot:
        plot(summaries, sweep, Path(args.plot))
        print(f"[sim] figure in {args.plot}", file=sys.stderr)


def plot(summaries, sweep, path):
    from sim_plot import plot as _plot
    _plot(summaries, sweep, path)


if __name__ == "__main__":
    main()
