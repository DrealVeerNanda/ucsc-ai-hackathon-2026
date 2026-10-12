"""Bring the demo back after the GX10 has been switched off (a move, a power cut), with nobody at
the keyboard: the keeper and its phone bridge, both model servers, and Start as soon as every
line on the checklist is green. Nobody has to be logged in on the screen; logging in is only for
seeing the pages (double-click Proprio).

  python tools/boot_start.py              # what the proprio-boot service runs when the GX10 starts
  bash tools/install_launcher.sh --boot   # install that service (no sudo; needs "linger", which the GX10 has)

It never starts with a red line: with no headphones the sound would come out of the screen.
PROPRIO_BOOT_START=0 in .env.launcher leaves Start to a person; PROPRIO_BOOT_WAIT_S (1800) is how
long it waits for the lines to turn green before leaving it at the checklist.

One thing does need a person, once, at the GX10's keyboard: while its login screen is up, the
login screen's own sound server holds Bluetooth, and the headphones get no microphone (11 Oct
2026). Log in, or press Ctrl+Alt+F3 and log in there. This script then restarts our sound manager
so it can claim the headphones, and reconnects them.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import launcher  # noqa: E402


def say(text: str):
    print(time.strftime("%H:%M:%S"), text, flush=True)


def answering(url: str) -> bool:
    try:
        with urllib.request.urlopen(url.rstrip("/").removesuffix("/v1") + "/health", timeout=2.0) as r:
            return r.status == 200
    except Exception:
        return False


def login_screen_up() -> bool:
    """Is the login screen the one in front? Then its sound server, not ours, has Bluetooth."""
    try:
        out = subprocess.run(["loginctl", "list-sessions", "--no-legend"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return any(" gdm " in f" {row} " and " active " in f" {row} " for row in out.splitlines())


def take_headphones(port: int):
    """The login screen has just stepped aside: our sound manager was refused the microphone side
    while it was there, and only asks again when it starts."""
    subprocess.run(["systemctl", "--user", "restart", "wireplumber"], capture_output=True, timeout=20)
    time.sleep(5.0)
    lines = {l["id"]: l["level"] for l in (launcher.ask(port) or {}).get("lines", [])}
    got = launcher.ask(port, "reconnect_airpods" if lines.get("airpods") == "green" else "connect_airpods", timeout_s=20.0)
    say(f"sound manager restarted; headphones: {(got or {}).get('message', 'the keeper did not answer')}")


def main() -> int:
    uid = os.getuid()
    os.environ.setdefault("XDG_RUNTIME_DIR", f"/run/user/{uid}")
    os.environ.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path=/run/user/{uid}/bus")
    env = launcher.load_env()
    port = int(env.get("PROPRIO_PORT", launcher.PORT))
    if launcher.ask(port) is None:
        got = subprocess.run([sys.executable, str(ROOT / "launcher.py")], env=dict(os.environ, PROPRIO_OPEN="0"),
                             capture_output=True, text=True, timeout=60)
        if launcher.ask(port) is None:
            say(f"the keeper did not come up: {(got.stderr or got.stdout).strip()[-300:]}")
            return 1
        say("keeper started")
    else:
        say("keeper already running")
    for name, url in (("nemotron", env.get("VOICE_LLM_URL", "http://127.0.0.1:8000/v1")),
                      ("cosmos", env.get("COSMOS_URL", "http://127.0.0.1:8001/v1"))):
        if url != "off" and not answering(url):
            got = launcher.ask(port, f"start_{name}", timeout_s=15.0)
            say(f"{name}: {(got or {}).get('message', 'the keeper did not answer')}")
    if env.get("PROPRIO_BOOT_START", "1") == "0":
        say("Start is left to a person (PROPRIO_BOOT_START=0)")
        return 0
    end, reds = time.monotonic() + float(env.get("PROPRIO_BOOT_WAIT_S", "1800")), None
    held = login_screen_up()             # was Bluetooth the login screen's when this began?
    while time.monotonic() < end:
        state = launcher.ask(port) or {}
        if state.get("mode") in ("running", "stopping"):
            say("it is running")
            return 0
        if state.get("ready"):
            got = launcher.ask(port, "start", timeout_s=15.0)
            say(f"every line is green: {(got or {}).get('message', 'the keeper did not answer')}")
            return 0
        now = [l["title"] for l in state.get("lines", []) if l.get("level") == "red"]
        pods_red = any(l.get("level") == "red" for l in state.get("lines", []) if str(l.get("id", "")).startswith("airpods"))
        if held and not login_screen_up():
            held = False
            if pods_red:
                say("the login screen has stepped aside")
                take_headphones(port)
                continue
        if now != reds:
            reds = now
            say("waiting for: " + (", ".join(now) or "the checklist")
                + (". Log in at the GX10's keyboard (or Ctrl+Alt+F3 and log in): its login screen is holding Bluetooth."
                   if held and pods_red else ""))
        time.sleep(2.0)
    say("still not green: left at the checklist. Press Start (or Start anyway) on the page.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
