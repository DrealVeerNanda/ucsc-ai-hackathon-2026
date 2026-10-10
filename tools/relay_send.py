"""Fallback if record3d will not build on the GX10: read the phone on a Mac and forward the
frames to the GX10 over Ethernet. Run this on the Mac; all models still run on the GX10.

  On the GX10:  STREAM_SOURCE=relay python main.py          (or: python tools/probe_stream.py relay)
  On the Mac:   python tools/relay_send.py GX10_IP           # Record3D over USB
                python tools/relay_send.py GX10_IP --source pathon

Colour goes as JPEG, depth as a 16-bit PNG in millimetres (lossless). Only the newest frame is
sent, so a slow link lowers the frame rate instead of adding delay. Either side can be
restarted; the other one waits and carries on.
"""
import argparse
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import stream  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("host", help="GX10 address, optionally HOST:PORT")
    p.add_argument("--source", default=None, help="record3d (default) | pathon[:IP] | recordings/NAME")
    p.add_argument("--max-side", type=int, default=960, help="downscale colour before sending")
    p.add_argument("--quality", type=int, default=85, help="JPEG quality for colour")
    a = p.parse_args()
    host, _, port = a.host.partition(":")
    port = int(port or stream.RELAY_PORT)

    try:
        s = stream.start(a.source)
    except stream.SourceError as e:
        sys.exit(f"[relay] {e}")
    try:
        while True:
            try:
                conn = socket.create_connection((host, port), timeout=3)
            except OSError as e:
                print(f"[relay] {host}:{port} not reachable ({e}); retrying. Is the GX10 side running?")
                time.sleep(1)
                continue
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"[relay] connected to {host}:{port}")
            n, sent, shown = 0, 0, time.time()
            try:
                with conn:
                    while True:
                        got = s.get_raw(wait=1.0)
                        if got is None:
                            continue
                        msg = stream.encode_raw(stream.shrink(got[0], a.max_side), a.quality)
                        conn.sendall(msg)
                        n, sent = n + 1, sent + len(msg)
                        if time.time() - shown >= 5.0:
                            dt = time.time() - shown
                            print(f"[relay] {n / dt:.1f} fps, {sent * 8 / dt / 1e6:.1f} Mbit/s, phone {s.status()['state']}")
                            n, sent, shown = 0, 0, time.time()
            except OSError as e:
                print(f"[relay] connection lost ({e}); reconnecting")
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
