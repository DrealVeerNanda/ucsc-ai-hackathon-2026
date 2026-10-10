"""Can lane 2 trust the camera pose? Answers yes or no, with no ruler needed.

  python tools/check_pose.py                    # Record3D over USB
  python tools/check_pose.py pathon
  python tools/check_pose.py recordings/NAME    # a recording made while moving the phone

Point the phone at a table or shelf where nothing moves, start this, then move the phone
slowly for the 10 s: about 20 cm side to side and up and down, with some tilting.

How it works: lidar points seen in one frame are carried through the two poses into a frame one
second later and compared with the depth measured there. With a correct pose they agree to a
centimetre or two however the phone moved. Every plausible axis convention is tried, so this
also says which matrix belongs in POSE_FIX in stream.py.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import stream  # noqa: E402
from stream import RawFrame, prepare  # noqa: E402

GOOD_M = 0.025    # median depth disagreement that still counts as a correct pose
MOVED_M = 0.03    # disagreement with the pose ignored, below which the phone barely moved


def residual(a, b, use_pose=True, step=4):
    """Median depth disagreement in metres after moving frame a's lidar points into frame b,
    or None if too few of them land on valid depth in b."""
    h, w = a.depth.shape
    v, u = np.mgrid[step // 2:h:step, step // 2:w:step]
    z = a.depth[v, u]
    ok = (z > 0.15) & (z < 5.0)
    u, v, z = u[ok], v[ok], z[ok]
    if z.size < 50:
        return None
    pts = np.stack([(u - a.K[0, 2]) * z / a.K[0, 0], (v - a.K[1, 2]) * z / a.K[1, 1], z, np.ones_like(z)])
    if use_pose:
        pts = np.linalg.inv(b.pose) @ a.pose @ pts
    zb = pts[2]
    front = zb > 0.1
    ub = np.full(zb.shape, -1)
    vb = np.full(zb.shape, -1)
    ub[front] = np.rint(pts[0, front] / zb[front] * b.K[0, 0] + b.K[0, 2]).astype(int)
    vb[front] = np.rint(pts[1, front] / zb[front] * b.K[1, 1] + b.K[1, 2]).astype(int)
    hb, wb = b.depth.shape
    inside = front & (ub >= 0) & (ub < wb) & (vb >= 0) & (vb < hb)
    seen = b.depth[vb[inside], ub[inside]]
    good = seen > 0
    if good.sum() < 50:
        return None
    return float(np.median(np.abs(seen[good] - zb[inside][good])))


def candidates():
    """(name, pose is world-to-camera, 3 x 3 fix) for every plausible convention."""
    out = []
    for inverted in (False, True):
        for flip, F in (("flip y,z", np.diag([1.0, -1.0, -1.0])), ("keep axes", np.eye(3))):
            for deg in (0, 90, 180, 270):
                name = f"{flip}, turn {deg}" + (", world-to-camera" if inverted else "")
                out.append((name, inverted, F @ stream._ROT[deg]))
    return out


def evaluate(raws, times, spacing=0.5):
    """raws: RawFrames with poses, times: their arrival times. Returns a dict with the verdict."""
    keys, last = [], -1e9
    for raw, t in zip(raws, times):
        if raw.pose is not None and t - last >= spacing:
            # depth-sized colour keeps this fast: only depth, K and pose are used
            keys.append(stream.shrink(raw, max(raw.depth.shape)))
            last = t
    kind = raws[0].kind if raws else "record3d"
    out = {"kind": kind, "frames": len(raws), "with_pose": sum(r.pose is not None for r in raws),
           "verdict": "no", "best": None, "ranked": []}
    if out["with_pose"] == 0:
        out["why"] = "the app sends no pose"
        return out
    pairs = [(i, i + 2) for i in range(len(keys) - 2)]
    if len(pairs) < 3:
        out["verdict"], out["why"] = "unclear", "too few frames with a pose; record for longer"
        return out

    def frames(inverted, F):
        return [prepare(RawFrame(k.rgb, k.depth, k.K, np.linalg.inv(k.pose) if inverted else k.pose, k.kind),
                        0.0, rotate=0, max_side=None, pose_fix=F) for k in keys]

    def score(fs, use_pose=True):
        r = [residual(fs[i], fs[j], use_pose) for i, j in pairs]
        r = [x for x in r if x is not None]
        return float(np.median(r)) if len(r) >= 3 else float("inf")

    base = frames(False, np.eye(3))
    out["ignoring_pose"] = score(base, use_pose=False)
    pos = np.array([k.pose[:3, 3] for k in keys])
    out["moved_m"] = float(np.linalg.norm(pos.max(0) - pos.min(0)))
    out["biggest_jump_m"] = float(np.linalg.norm(np.diff(pos, axis=0), axis=1).max())
    ranked = sorted(((score(frames(inv, F)), name, inv, F) for name, inv, F in candidates()), key=lambda c: c[0])
    out["ranked"] = [(r, name) for r, name, _, _ in ranked]
    best_r, best_name, best_inv, best_F = ranked[0]
    out["best"] = {"residual": best_r, "name": best_name, "inverted": best_inv, "fix": best_F}
    cur = stream.POSE_FIX.get(kind)
    out["matches_stream"] = cur is not None and not best_inv and np.allclose(cur, best_F)
    if not np.isfinite(out["ignoring_pose"]):
        out["verdict"] = "unclear"
        out["why"] = "the camera saw almost no valid depth; is the lens covered, or too close to something?"
    elif out["ignoring_pose"] < MOVED_M:
        out["verdict"], out["why"] = "unclear", "the phone barely moved; move it about 20 cm and tilt it"
    elif best_r < GOOD_M and best_r < 0.5 * out["ignoring_pose"]:
        out["verdict"], out["why"] = "yes", "depth agrees across frames once the pose is applied"
        if out["biggest_jump_m"] > 0.5:
            out["why"] += f", but the position jumped {out['biggest_jump_m']:.2f} m at one point (tracking reset)"
    else:
        out["why"] = "a pose arrives but no convention makes depth agree across frames"
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", nargs="?", default=None, help="record3d | pathon[:IP] | relay[:PORT] | recordings/NAME")
    p.add_argument("--seconds", type=float, default=10.0)
    a = p.parse_args()
    try:
        s = stream.start(a.source, loop=False)
    except stream.SourceError as e:
        sys.exit(f"[pose] {e}")
    print(f"[pose] collecting for {a.seconds:.0f} s: move the phone slowly, about 20 cm each way, and tilt it")
    raws, times, t0 = [], [], time.time()
    try:
        while time.time() - t0 < a.seconds and s.status()["state"] != "ended":
            got = s.get_raw(wait=0.5)
            if got is not None and (not times or got[1] - times[-1] >= 0.1):
                raws.append(got[0])
                times.append(got[1])
    finally:
        stream.stop()
    if not raws:
        sys.exit("[pose] no frames arrived")

    r = evaluate(raws, times)
    print(f"\nframes: {r['frames']}, with a pose: {r['with_pose']}")
    if "ignoring_pose" in r:
        print(f"phone moved about {r['moved_m'] * 100:.0f} cm according to the pose")
        print(f"depth disagreement ignoring the pose: {r['ignoring_pose'] * 100:.1f} cm")
        print("depth disagreement using the pose, best conventions first:")
        for res, name in r["ranked"][:4]:
            print(f"  {res * 100:6.1f} cm   {name}")
    print(f"\nPOSE USABLE: {r['verdict'].upper()} ({r['why']})")
    if r["verdict"] == "yes":
        if r["matches_stream"]:
            print(f"stream.py already uses this convention for {r['kind']}: Frame.pose is good to use.")
        elif r["best"]["inverted"]:
            print("The app sends world-to-camera, which stream.py does not expect: tell lane 1 before using Frame.pose.")
        else:
            rows = ", ".join("[" + ", ".join(f"{v:.0f}" for v in row) + "]" for row in r["best"]["fix"])
            print(f'stream.py has a different convention. Set this in stream.py, then Frame.pose is good to use:\n'
                  f'  POSE_FIX["{r["kind"]}"] = np.array([{rows}])')
    elif r["verdict"] == "no":
        print("Tell lane 2 to use the fallback: hold the last camera-frame position for up to 2 s.")
    return r


if __name__ == "__main__":
    main()
