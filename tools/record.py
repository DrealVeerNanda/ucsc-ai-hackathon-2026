"""Save a recording that stream.py can replay, so the other lanes can work without the phone.

  python tools/record.py object_on_table                 # 30 s from Record3D -> recordings/object_on_table
  python tools/record.py hand_reaching --source pathon --seconds 20
Replay it anywhere (no phone, no record3d needed):
  python tools/probe_stream.py recordings/object_on_table --show
  STREAM_SOURCE=recordings/object_on_table python main.py
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import stream  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name", help="folder name under recordings/")
    p.add_argument("--source", default=None, help="record3d (default) | pathon[:IP] | relay[:PORT]")
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--fps", type=float, default=20.0, help="keep at most this many frames per second")
    p.add_argument("--countdown", type=int, default=3)
    p.add_argument("--force", action="store_true", help="overwrite an existing recording")
    a = p.parse_args()

    folder = ROOT / "recordings" / a.name
    if folder.exists():
        if not a.force:
            sys.exit(f"[record] {folder.relative_to(ROOT)} already exists; pick another name or pass --force")
        shutil.rmtree(folder)
    try:
        s = stream.start(a.source)
    except stream.SourceError as e:
        sys.exit(f"[record] {e}")
    for i in range(a.countdown, 0, -1):
        print(f"[record] starting in {i}...")
        time.sleep(1)

    rec = stream.Recorder(folder, fps=a.fps)
    print(f"[record] recording {a.seconds:.0f} s to {folder.relative_to(ROOT)}. Ctrl-C stops early and still saves.")
    t0 = last = shown = 0.0
    try:
        while True:
            got = s.get_raw(wait=1.0)
            if got is None:
                if s.status()["state"] != "streaming":
                    print(f"[record] stream {s.status()['state']}: {s.status()['error']}")
                continue
            raw, t = got
            t0 = t0 or t
            if t - t0 > a.seconds:
                break
            if t - last < 1.0 / a.fps - 0.002:
                continue
            rec.add(raw, t)
            last = t
            if t - shown >= 5.0:
                shown = t
                print(f"[record] {t - t0:4.0f} s  {len(rec.t)} frames")
    except KeyboardInterrupt:
        pass
    finally:
        meta = rec.close()
        stream.stop()
    if not meta["frames"]:
        sys.exit("[record] no frames were saved")
    mb = sum(f.stat().st_size for f in folder.iterdir()) / 1e6
    print(f"[record] saved {meta['frames']} frames, {meta['seconds']} s at {meta['fps']} fps, "
          f"pose {'yes' if meta['has_pose'] else 'NO'}, {mb:.0f} MB")
