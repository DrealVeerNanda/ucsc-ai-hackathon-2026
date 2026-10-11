"""The checklist launcher.py shows before the demo starts: everything that has to be true, each
as one line a person can read from across the table.

    line = {"id": "airpods", "title": "AirPods connected", "level": "green" | "amber" | "red",
            "message": "one plain sentence: what is wrong and what to do",
            "required": True,            # False: shown, but never holds up Start
            "blocked": False,            # True: cannot be judged until a line above is fixed
            "fix": {"action": "connect_airpods", "label": "Connect AirPods"} or None}

Each group below looks at the machine and returns its lines. Nothing here changes anything: the
buttons' actions live in launcher.py. Standard library only, so it starts in a blink and cannot
be broken by a Python package.

The phone's own port is never touched from here. The app stops streaming when a program
connects to it and goes away again, so whether it is streaming is read from the status file the
phone bridge writes (out/status/phone.json).

    python launcher_checks.py              # print every line once
    python launcher_checks.py --prove      # load every model with the network blocked
    python launcher_checks.py --offline    # the settings that keep the libraries off the network, if that is safe
"""
from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

GREEN, AMBER, RED = "green", "amber", "red"
ROOT = Path(__file__).resolve().parent
AIRPODS = "18:3F:70:53:BD:95"      # the pair that lives with the GX10; any connected headset is used first
STALE_S = 5.0                      # a status file older than this means its writer is dead
# Bluetooth services that make a device something to listen through and speak into.
AUDIO_SINK, HANDSFREE = "0000110b", "0000111e"


def line(id: str, title: str, level: str, message: str, *, required: bool = True, blocked: bool = False,
         fix: tuple | None = None, detail: list | None = None, minor: bool = False) -> dict:
    return {"id": id, "title": title, "level": level, "message": message, "required": required, "minor": minor,
            "blocked": blocked, "fix": {"action": fix[0], "label": fix[1]} if fix else None, "detail": detail}


def advice(id: str, title: str, level: str, message: str, **kw) -> dict:
    """A line worth knowing about that never holds up Start, drawn small while it is green."""
    return line(id, title, level, message, required=False, minor=True, **kw)


def run(cmd: list[str], timeout: float = 4.0) -> tuple[int, str]:
    """(exit code, everything it printed). 127 if the program is not installed, 124 if it hung."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
        return p.returncode, (p.stdout + p.stderr).strip()
    except FileNotFoundError:
        return 127, f"{cmd[0]}: not installed"
    except subprocess.TimeoutExpired:
        return 124, f"{' '.join(cmd)}: no answer in {timeout:.0f} s"
    except OSError as e:
        return 126, str(e)


def read_status(path: Path) -> dict | None:
    """A status file as written by the long-lived parts, or None if it is missing or half-written."""
    try:
        got = json.loads(Path(path).read_text())
        return got if isinstance(got, dict) else None
    except (OSError, ValueError):
        return None


def age_of(status: dict | None, now: float | None = None) -> float | None:
    try:
        return (now or time.time()) - float(status["t"])
    except (TypeError, KeyError, ValueError):
        return None


@dataclass
class Setup:
    """Where things are and how to ask the machine. Tests hand in their own `run` and folders."""
    root: Path = ROOT
    env: dict = field(default_factory=lambda: dict(os.environ))
    run: object = run
    home: Path = field(default_factory=Path.home)
    reconnected: float = 0.0     # time.time() of the last AirPods reconnect: trouble before it is history

    @property
    def status_dir(self) -> Path:
        return self.root / "out" / "status"

    @property
    def source(self) -> str:
        return self.env.get("STREAM_SOURCE", "auto")

    @property
    def replay(self) -> bool:
        """The camera is a saved recording, not the phone (stream.py treats any other name as a folder)."""
        return self.source.partition(":")[0] not in ("auto", "pathon", "record3d", "relay", "bridge")

    @property
    def bridge(self) -> Path:
        return self.root / "tools" / "phone_bridge.py"

    @property
    def nemotron_url(self) -> str:
        return self.env.get("VOICE_LLM_URL", "http://127.0.0.1:8000/v1")      # the same variable voice.py reads

    @property
    def cosmos_url(self) -> str:
        return self.env.get("COSMOS_URL", "http://127.0.0.1:8001/v1")         # and cosmos.py

    @property
    def dash_port(self) -> int:
        return int(self.env.get("PROPRIO_DASH_PORT", "8642"))

    @property
    def voice_port(self) -> int:
        return int(self.env.get("VOICE_PORT", "8644"))


# ---------------------------------------------------------------------------
# The phone

def phone(s: Setup, bridge_note: str = "") -> list[dict]:
    if s.replay:
        return [line("phone_stream", "Camera", AMBER, f"Replaying {s.source} instead of the phone: a rehearsal, not the demo.",
                     required=False)]
    st = read_status(s.status_dir / "phone.json")
    age = age_of(st)
    fresh = age is not None and age <= STALE_S
    streaming = bool(fresh and st.get("ok"))

    rc, out = s.run(["idevice_id", "-l"])
    ids = [w for w in out.split() if re.fullmatch(r"[0-9A-Fa-f-]{20,}", w)] if rc == 0 else []
    if rc == 127:
        usb = line("phone_usb", "Phone on the cable", RED, "The iPhone tools are not installed here (libimobiledevice-utils).")
    elif ids:
        usb = line("phone_usb", "Phone on the cable", GREEN, f"iPhone connected by cable ({ids[0][:8]}).")
    elif streaming:
        usb = line("phone_usb", "Phone on the cable", AMBER, "No iPhone on the cable, but frames are arriving: it is streaming "
                   "over Wi-Fi, which is slower.", required=False)
    else:
        usb = line("phone_usb", "Phone on the cable", RED, "No iPhone on the cable. Plug the phone into the GX10 with the data "
                   "cable and unlock it.")

    if streaming:
        # Frames are arriving, so it trusts us. Not asking the phone saves it a question a second.
        trust = line("phone_trust", "Phone trusts this computer", GREEN, "Trusted: frames are arriving.", required=bool(ids))
    elif not ids:
        trust = line("phone_trust", "Phone trusts this computer", RED, "Waiting for the phone on the cable.", blocked=True)
    else:
        rc, out = s.run(["idevicepair", "validate"], 6.0)
        low = out.lower()
        if rc == 0 and "success" in low:
            trust = line("phone_trust", "Phone trusts this computer", GREEN, "Trusted and paired.")
        elif "passcode" in low:
            trust = line("phone_trust", "Phone trusts this computer", RED, "The phone is locked. Unlock it with its passcode.")
        elif "trust dialog" in low:
            trust = line("phone_trust", "Phone trusts this computer", RED, "The phone is asking whether to trust this computer. "
                         "Tap Trust on its screen.", fix=("pair_phone", "Pair"))
        else:
            trust = line("phone_trust", "Phone trusts this computer", RED, "The phone has not been paired with this computer. "
                         "Unlock it, press Pair, and tap Trust on its screen.", fix=("pair_phone", "Pair"))

    title = "Phone is streaming"
    if not s.bridge.exists():
        stream = line("phone_stream", title, AMBER, "This checkout has no phone bridge (tools/phone_bridge.py), so the stream cannot "
                      "be checked before the start. Open PathOn on the phone and tap Start; the pipeline connects by itself.",
                      required=False)
    elif st is None:
        stream = line("phone_stream", title, RED, "The phone bridge has not reported yet (out/status/phone.json is missing). "
                      + (bridge_note or "It is started with this page; give it a few seconds."))
    elif not fresh:
        stream = line("phone_stream", title, RED, f"The phone bridge stopped reporting {_ago(age)} ago. " + bridge_note,
                      fix=("restart_bridge", "Restart the phone bridge"))
    elif not st.get("ok"):
        stream = line("phone_stream", title, RED, str(st.get("message") or f"The phone is not streaming ({st.get('state', 'no state')}). "
                                                      "Open PathOn on the phone and tap Start."))
    else:
        fps, frame_age = _number(st, "fps"), _number(st, "frame_age_s", "last_frame_age_s", "age_s")
        if frame_age is None and _number(st, "age_ms") is not None:
            frame_age = _number(st, "age_ms") / 1000.0
        facts = ", ".join(f for f in (None if fps is None else f"{fps:.0f} frames a second",
                                       None if frame_age is None else f"newest frame {frame_age * 1000:.0f} ms old") if f)
        slow = (fps is not None and fps < 15) or (frame_age is not None and frame_age > 0.5)
        stream = line(title=title, id="phone_stream", level=AMBER if slow else GREEN,
                      message=("Streaming, but slowly: " if slow else "Streaming: ") + (facts or str(st.get("message", "ok"))) + ".")
    return [usb, trust, stream, cable(s)]


def cable(s: Setup) -> dict:
    """Has anything on USB dropped out in the last five minutes? The phone did six times in one
    evening, twice coming back within seconds: a connector being tugged by the strap."""
    rc, out = s.run(["journalctl", "-k", "--since", "-5min", "--no-pager", "-q", "-g", "USB disconnect"], 5.0)
    drops = len([r for r in out.splitlines() if "USB disconnect" in r]) if rc in (0, 1) else None
    if drops is None:
        return advice("phone_cable", "Phone cable steady", "none", "Cannot read the kernel log here to see whether the cable drops out.")
    if drops:
        return advice("phone_cable", "Phone cable steady", AMBER, f"A USB cable dropped out {drops} time{'' if drops == 1 else 's'} in the "
                      "last five minutes. Tape the cable to the strap with a slack loop.")
    return advice("phone_cable", "Phone cable steady", GREEN, "No USB drop-outs in the last five minutes.")


def _number(d: dict, *keys):
    for k in keys:
        try:
            if d.get(k) is not None:
                return float(d[k])
        except (TypeError, ValueError):
            pass
    return None


def _ago(seconds: float) -> str:
    return f"{seconds:.0f} s" if seconds < 120 else f"{seconds / 60:.0f} min" if seconds < 7200 else f"{seconds / 3600:.0f} h"


# ---------------------------------------------------------------------------
# The earbuds: connected, the place sound goes, and able to be the microphone

def bt_info(s: Setup, addr: str) -> dict:
    """What bluetoothctl knows about one device: name, paired, connected, and its services."""
    rc, out = s.run(["bluetoothctl", "info", addr])
    if rc != 0 or "not available" in out:
        return {"addr": addr, "known": False, "name": "AirPods", "connected": False, "paired": False, "audio": False}
    name = (re.search(r"^\s*(?:Alias|Name): (.+)$", out, re.M) or [None, addr])[1].strip()
    return {"addr": addr, "known": True, "name": name,
            "connected": bool(re.search(r"Connected: yes", out)), "paired": bool(re.search(r"Paired: yes", out)),
            "audio": AUDIO_SINK in out or HANDSFREE in out}


def find_headset(s: Setup) -> dict:
    """The earbuds to use: the one named in PROPRIO_HEADSET, else a headset that is connected right
    now (whoever's it is), else the paired AirPods, so the Connect button has something to connect."""
    wanted = s.env.get("PROPRIO_HEADSET")
    if wanted:
        return bt_info(s, wanted)
    rc, out = s.run(["bluetoothctl", "devices", "Connected"])
    connected = re.findall(r"^Device (\S+) ", out, re.M) if rc == 0 else []
    for addr in sorted(connected, key=lambda a: a != AIRPODS):
        info = bt_info(s, addr)
        if info["connected"] and info["audio"]:
            return info
    rc, out = s.run(["bluetoothctl", "devices", "Paired"])
    paired = re.findall(r"^Device (\S+) (.+)$", out, re.M) if rc == 0 else []
    pods = [a for a, n in paired if "airpods" in n.lower()]
    return bt_info(s, AIRPODS if AIRPODS in [a for a, _ in paired] or not pods else pods[0])


def default_sink(s: Setup) -> dict:
    """Where sound goes right now: {"name": PipeWire's node name, "says": its name for people}."""
    rc, out = s.run(["wpctl", "inspect", "@DEFAULT_AUDIO_SINK@"])
    if rc != 0:
        return {"name": None, "says": None, "error": out.splitlines()[-1] if out else "wpctl failed"}
    get = lambda key: (re.search(rf'{re.escape(key)} = "([^"]*)"', out) or [None, None])[1]
    return {"name": get("node.name"), "says": get("node.description") or get("node.nick") or get("node.name")}


def bluez_device(s: Setup) -> tuple[int | None, list[str]]:
    """(PipeWire's number for the Bluetooth earbuds, the modes they offer), as voice.Headset.find reads them."""
    rc, out = s.run(["wpctl", "status"])
    try:
        devices = out.split("Audio", 1)[1].split("Sinks:", 1)[0]
    except IndexError:
        return None, []
    m = re.search(r"(\d+)\.\s+(.+?)\s+\[bluez5\]", devices)
    if rc != 0 or not m:
        return None, []
    rc, params = s.run(["pw-cli", "enum-params", m.group(1), "EnumProfile"])
    return int(m.group(1)), re.findall(r'Profile:name.*?String "([^"]+)"', params, re.S)


def bluez_sink(s: Setup) -> int | None:
    """PipeWire's number for the earbuds as a place to send sound, for making it the default."""
    rc, out = s.run(["wpctl", "status"])
    try:
        sinks = out.split("Audio", 1)[1].split("Sinks:", 1)[1].split("Sink endpoints:", 1)[0]
    except IndexError:
        return None
    for num in re.findall(r"(\d+)\.\s", sinks):
        rc, about = s.run(["wpctl", "inspect", num])
        if rc == 0 and re.search(r'node\.name = "bluez_output\.', about):
            return int(num)
    return None


def airpods(s: Setup) -> list[dict]:
    pods = find_headset(s)
    name = pods["name"]
    are = "are" if name.lower().rstrip().endswith(("s", "pro")) else "is"
    t1, t2, t3 = "AirPods connected", "Sound goes to the AirPods", "AirPods microphone"
    if not pods["known"]:
        first = line("airpods", t1, RED, "No AirPods are paired with this computer. Pair them in Settings, Bluetooth, "
                     "with the case open and its button held.")
    elif not pods["connected"]:
        first = line("airpods", t1, RED, f"{name} {are} not connected. Put them in your ears (or open the case next to the GX10), "
                     "then press Connect.", fix=("connect_airpods", "Connect AirPods"))
    else:
        first = line("airpods", t1, GREEN, f"{name} {are} connected.")
    if not pods["connected"]:
        wait = "Waiting for the AirPods."
        return [first, line("airpods_output", t2, RED, wait, blocked=True), line("airpods_mic", t3, RED, wait, blocked=True),
                bluetooth(s)]

    sink = default_sink(s)
    if sink["name"] and sink["name"].startswith("bluez_output."):
        second = line("airpods_output", t2, GREEN, f"Sound goes to {sink['says']}.")
    elif sink["name"]:
        second = line("airpods_output", t2, RED, f"Sound is going to {sink['says']}, not the AirPods.",
                      fix=("use_airpods_output", "Send sound to the AirPods"))
    else:
        second = line("airpods_output", t2, RED, f"The sound server is not answering ({sink.get('error')}).")

    device, modes = bluez_device(s)
    headset = next((m for m in ("headset-head-unit-msbc", "headset-head-unit") if m in modes), None)
    if device is None:
        third = line("airpods_mic", t3, RED, "The sound server does not see the AirPods yet. Press Reconnect.",
                     fix=("reconnect_airpods", "Reconnect"))
    elif headset is None or "a2dp-sink" not in modes:
        third = line("airpods_mic", t3, RED, "The AirPods are offering no microphone mode. Press Reconnect.",
                     fix=("reconnect_airpods", "Reconnect"))
    elif lost_modes(s):
        third = line("airpods_mic", t3, AMBER, f"The AirPods refused a mode switch {lost_modes(s)} time(s) in the last ten minutes "
                     "(\"invalid profile\"); they can take two minutes to recover by themselves. Press Reconnect.",
                     fix=("reconnect_airpods", "Reconnect AirPods"))
    else:
        third = line("airpods_mic", t3, GREEN, f"Stereo and microphone modes are both there (a2dp-sink, {headset}).")
    return [first, second, third, bluetooth(s)]


def lost_modes(s: Setup) -> int:
    """How often the sound server failed to switch the earbuds' mode in the last ten minutes,
    not counting what happened before the last reconnect (which is the repair)."""
    since = max(time.time() - 600, s.reconnected)
    rc, out = s.run(["journalctl", "--user", "-u", "wireplumber", "--since", f"@{since:.0f}", "--no-pager", "-q",
                     "-g", "invalid profile"], 5.0)
    return len([r for r in out.splitlines() if "invalid profile" in r]) if rc == 0 else 0


def bluetooth(s: Setup) -> dict:
    """Is the radio busy looking for other devices? A Settings window left on its Bluetooth page
    keeps it scanning for hours, and the AirPods' sound pays for it."""
    rc, out = s.run(["bluetoothctl", "show"])
    flag = lambda name: bool(re.search(rf"{name}: yes", out))
    if rc != 0 or "Powered" not in out:
        return line("bluetooth", "Bluetooth quiet", RED, "Bluetooth is not answering. Switch it on in Settings.")
    if not flag("Powered"):
        return line("bluetooth", "Bluetooth quiet", RED, "Bluetooth is switched off. Switch it on in Settings.")
    if flag("Discovering") or flag("Discoverable"):
        doing = "scanning for devices" if flag("Discovering") else "announcing itself to other devices"
        return line("bluetooth", "Bluetooth quiet", AMBER, f"Bluetooth is {doing}, which costs the AirPods sound quality. "
                    "It is usually a Settings window left open on Bluetooth.", fix=("quiet_bluetooth", "Close Settings and stop it"), minor=True)
    return line("bluetooth", "Bluetooth quiet", GREEN, "Not scanning and not announcing itself.", minor=True)


# ---------------------------------------------------------------------------
# The two model servers

def health(url: str, timeout_s: float = 1.5) -> tuple[str, str]:
    """("up" | "loading" | "down" | "silent" | "off", why) for a llama.cpp server at .../v1."""
    if url == "off":
        return "off", "switched off"
    try:
        with urllib.request.urlopen(url.rsplit("/v1", 1)[0] + "/health", timeout=timeout_s) as r:
            return ("up", "") if r.status == 200 else ("loading", f"HTTP {r.status}")
    except urllib.error.HTTPError as e:
        return "loading", f"HTTP {e.code}"                 # llama.cpp answers 503 while the model loads
    except urllib.error.URLError as e:
        refused = isinstance(e.reason, ConnectionRefusedError)
        return ("down", "nothing is listening") if refused else ("silent", str(e.reason))
    except (OSError, ValueError) as e:
        return "silent", str(e)


def port_of(url: str) -> int | None:
    m = re.search(r":(\d+)(?:/|$)", url.split("//", 1)[-1])
    return int(m.group(1)) if m else None


def server(s: Setup, id: str, name: str, url: str, does: str) -> dict:
    state, why = health(url)
    port = port_of(url)
    if state == "up":
        return line(id, name, GREEN, f"{name} is answering on port {port}.")
    if state == "loading":
        return line(id, name, AMBER, f"{name} is loading its model ({why}). It takes about half a minute.")
    if state == "off":
        return line(id, name, AMBER, f"{name} is switched off in the settings: {does} will not work.", required=False)
    if state == "silent":
        return line(id, name, AMBER, f"{name} is running but not answering ({why}).")
    return line(id, name, RED, f"{name} is not running, so {does} will not work.", fix=(f"start_{id}", f"Start {name}"))


def servers(s: Setup) -> list[dict]:
    return [server(s, "nemotron", "Nemotron", s.nemotron_url, "understanding what the wearer asks for"),
            server(s, "cosmos", "Cosmos", s.cosmos_url, "listing what is on the table")]


# ---------------------------------------------------------------------------
# The GPU, which teammates share

def gpu(s: Setup, ours: set[int] | None = None, running: bool = False, busy: list | None = None) -> list[dict]:
    """`ours`: programs this launcher started. `busy`: recent load readings, kept by the caller."""
    title = "GPU free for the demo"
    rc, out = s.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"], 6.0)
    if rc != 0 or not out.strip():
        return [line("gpu", title, RED, "The GPU is not answering (nvidia-smi failed): "
                     + (out.splitlines()[-1] if out else "no output") + ". Restart the GX10.")]
    name, load, temp = ([p.strip() for p in out.splitlines()[0].split(",")] + ["", ""])[:3]
    rc, apps = s.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], 6.0)
    ports = {port_of(s.nemotron_url), port_of(s.cosmos_url)}
    others, mine = [], []
    for row in apps.splitlines() if rc == 0 else []:
        try:
            pid, mem = [p.strip() for p in row.split(",")[:2]]
            pid = int(pid)
        except ValueError:
            continue
        rc2, who = s.run(["ps", "-o", "user=,etimes=,args=", "-p", str(pid)])
        user, took, args = (who.split(None, 2) + ["", "", ""])[:3] if rc2 == 0 and who else ("?", "0", "a program that has ended")
        served = re.search(r"llama-server.*--port (\d+)", args)
        if pid in (ours or set()):
            mine.append("this demo")
        elif served and int(served.group(1)) in ports:
            mine.append({port_of(s.nemotron_url): "Nemotron", port_of(s.cosmos_url): "Cosmos"}[int(served.group(1))])
        else:
            gb = f"{float(mem) / 1024:.1f} GB" if mem.replace(".", "").isdigit() else "memory unknown"
            others.append(f"{_short(args)} ({user}, {gb}, running for {_ago(float(took or 0))})")
    try:
        load_now = float(load)
    except ValueError:
        load_now = 0.0
    if busy is not None:
        busy.append(load_now)
        del busy[:-5]
    facts = f"{name}, {temp} °C, {load_now:.0f}% busy"
    if others:
        return [line("gpu", title, AMBER, ("Someone else is using the GPU: " + "; ".join(others) + ". It will slow the demo "
                                           "(a job like this took a run from 26 to 10 frames a second). Ask them to stop it."))]
    if not running and busy and len(busy) >= 3 and min(busy[-3:]) >= 50:
        return [line("gpu", title, AMBER, f"The GPU is {load_now:.0f}% busy although the demo is not running: someone is "
                     "using Nemotron or Cosmos right now. Wait for it to finish.")]
    using = " and ".join(sorted(set(mine))) or "nothing"
    return [line("gpu", title, GREEN, f"Only {using} on it ({facts}).")]


def _short(args: str, n: int = 60) -> str:
    """A command line as a person would name it: no folders, not too long."""
    words = [w.rsplit("/", 1)[-1] if w.startswith("/") else w for w in args.split()]
    out = " ".join(words)
    return out if len(out) <= n else out[: n - 1] + "…"


# ---------------------------------------------------------------------------
# Every model on disk, so nothing is fetched when the network is unplugged

WHISPER = "openai/whisper-large-v3"           # voice.load_whisper's choice on the GPU
WHISPER_FILES = ("config.json", "generation_config.json", "preprocessor_config.json", "tokenizer.json",
                 "tokenizer_config.json", "vocab.json", "merges.txt", "normalizer.json", "added_tokens.json",
                 "special_tokens_map.json", "model.safetensors")
YOLO_WEIGHTS = "yolov8s-worldv2.pt"           # perception.OpenVocabDetector.WEIGHTS["yoloworld"]


def model_files(s: Setup) -> list[dict]:
    """Every file the pipeline and the two servers load, where the code looks for it, and whether it is there."""
    env, home = s.env, s.home
    models = Path(env.get("MODELS_DIR") or s.root / "models")                # perception.MODELS_DIR
    try:    # ultralytics keeps its text encoder where its own settings say, not next to the weights
        weights = Path(json.loads((home / ".config" / "Ultralytics" / "settings.json").read_text())["weights_dir"])
    except (OSError, ValueError, KeyError):
        weights = s.root / "weights"
    hub = Path(env.get("HF_HUB_CACHE") or Path(env.get("HF_HOME") or home / ".cache" / "huggingface") / "hub")
    snaps = sorted((hub / ("models--" + WHISPER.replace("/", "--")) / "snapshots").glob("*"))
    snap = snaps[-1] if snaps else hub / ("models--" + WHISPER.replace("/", "--")) / "snapshots" / "none"
    piper = Path(env.get("PIPER_VOICE") or s.root / "models" / "piper" / "en_US-lessac-medium.onnx")   # audio_out.PIPER_VOICE
    nemotron = Path(env.get("NEMOTRON_MODEL") or home / "unsloth" / "NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF"
                    / "NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-Q4_K_XL.gguf")                 # tools/run_nemotron.sh
    cosmos = home / "models" / "Cosmos-Reason2-8B-GGUF"                                                # tools/run_cosmos.sh
    want = [("YOLO-World weights", models / YOLO_WEIGHTS, 20e6, "start the pipeline once with the network on"),
            ("YOLO-World's CLIP text encoder", weights / "clip" / "ViT-B-32.pt", 300e6, "start the pipeline once with the network on"),
            ("MediaPipe hand model", models / "hand_landmarker.task", 5e6, "start the pipeline once with the network on"),
            ("Piper voice", piper, 50e6, "bash tools/fetch_models.sh piper"),
            ("Piper voice settings", Path(str(piper) + ".json"), 1e3, "bash tools/fetch_models.sh piper"),
            ("Nemotron (GGUF)", nemotron, 1e9, "download it again into ~/unsloth"),
            ("Cosmos Reason 2 (GGUF)", cosmos / "Cosmos-Reason2-8B.Q8_0.gguf", 1e9, "bash tools/fetch_models.sh 8b"),
            ("Cosmos Reason 2 vision part (GGUF)", cosmos / "Cosmos-Reason2-8B.mmproj-f16.gguf", 1e8, "bash tools/fetch_models.sh 8b")]
    want += [(f"Whisper large-v3 {name}", snap / name, 3e9 if name.endswith(".safetensors") else 1,
              "run the pipeline once with the network on") for name in WHISPER_FILES]
    out = []
    for name, path, at_least, get in want:
        try:
            st = path.stat()                                  # follows the cache's symlinks to the real file
            size, mtime = st.st_size, int(st.st_mtime)
        except OSError:
            size, mtime = None, None
        ok = size is not None and size >= at_least
        note = "" if ok else ("missing" if size is None else f"only {size / 1e6:.0f} MB: cut short") + f". To get it: {get}"
        out.append({"name": name, "path": str(path), "ok": ok, "size": size, "mtime": mtime, "note": note})
    return out


def files_key(files: list[dict]) -> str:
    """Changes when any model file does, so a proof made for other files is not believed."""
    return json.dumps([[f["path"], f["size"], f["mtime"]] for f in files])


def models(s: Setup, proof: dict | None = None) -> list[dict]:
    """`proof`: what prove() last found, {"key": files_key, "ok": bool, "results": [...]}, or
    {"running": True} while it is at work."""
    title = "Models on disk"
    files = model_files(s)
    missing = [f for f in files if not f["ok"]]
    detail = [{"name": f["name"], "ok": f["ok"], "note": f["note"] or f["path"]} for f in files]
    if missing:
        names = ", ".join(f["name"] for f in missing[:3]) + (f" and {len(missing) - 3} more" if len(missing) > 3 else "")
        return [line("models", title, RED, f"Not on this disk: {names}. With the network unplugged the demo would not start. "
                     f"{missing[0]['note'].split('. ', 1)[-1]}.", detail=detail)]
    if proof and proof.get("key") == files_key(files) and "ok" in proof:
        detail += [{"name": "Loads with the network blocked: " + r["name"], "ok": r["ok"], "note": r.get("note", "")}
                   for r in proof.get("results", [])]
        if proof["ok"]:
            return [line("models", title, GREEN, f"All {len(files)} files are here and every model loads with the network "
                         "blocked. The pipeline starts offline.", detail=detail)]
        bad = next((r for r in proof.get("results", []) if not r["ok"]), {"name": "a model", "note": proof.get("note", "")})
        return [line("models", title, AMBER, f"The files are here, but {bad['name']} did not load with the network blocked "
                     f"({bad.get('note', '')[:120]}). The pipeline will start with the network allowed.",
                     fix=("recheck_models", "Check again"), detail=detail)]
    return [line("models", title, AMBER, "The files are here. Loading each model once with the network blocked, to be sure: "
                 "about a minute.", detail=detail)]


class NoNetwork:
    """While this is active, a connection to anywhere but this machine fails at once. Anything
    that would have gone to the internet for a file shows itself as an error instead of working
    today and failing on stage."""

    def __enter__(self):
        self._connect = socket.socket.connect
        self._getaddrinfo = socket.getaddrinfo
        real = self._connect

        def connect(sock, address):
            host = address[0] if isinstance(address, tuple) else ""
            if sock.family in (socket.AF_INET, socket.AF_INET6) and host not in ("127.0.0.1", "::1", "localhost"):
                raise OSError(f"network blocked for this test: tried to reach {host}")
            return real(sock, address)

        def getaddrinfo(host, *a, **kw):
            if host not in ("127.0.0.1", "::1", "localhost", None, ""):
                raise socket.gaierror(f"network blocked for this test: tried to look up {host}")
            return self._getaddrinfo(host, *a, **kw)

        socket.socket.connect, socket.getaddrinfo = connect, getaddrinfo
        return self

    def __exit__(self, *exc):
        socket.socket.connect, socket.getaddrinfo = self._connect, self._getaddrinfo


def prove() -> dict:
    """Load every model the pipeline loads, the way it loads them, with the network blocked and
    the libraries told to stay offline. On the CPU, so it does not disturb a run. About a minute."""
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", YOLO_OFFLINE="true", TRACE="0",
                      TRANSFORMERS_VERBOSITY="error", HF_HUB_DISABLE_PROGRESS_BARS="1")
    sys.path.insert(0, str(ROOT))
    results = []

    def step(name, fn):
        t = time.perf_counter()
        try:
            fn()
            results.append({"name": name, "ok": True, "note": f"{time.perf_counter() - t:.1f} s"})
        except BaseException as e:
            results.append({"name": name, "ok": False, "note": f"{e.__class__.__name__}: {e}"[:300]})
        print(f"[prove] {results[-1]}", file=sys.stderr, flush=True)

    def yolo():
        import perception
        det = perception.OpenVocabDetector("yoloworld", device="cpu")
        det.set_target("cup")                         # this is what loads the CLIP text encoder

    def hand():
        import perception
        perception.MediaPipeHand()

    def whisper():
        import torch
        from faster_whisper.vad import get_speech_timestamps
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        WhisperProcessor.from_pretrained(WHISPER)
        WhisperForConditionalGeneration.from_pretrained(WHISPER, torch_dtype=torch.float16)
        import numpy as np
        get_speech_timestamps(np.zeros(16000, np.float32))      # the speech detector's own file

    def piper():
        import audio_out
        if audio_out.synthesize("ready", 48000) is None:
            raise RuntimeError(f"no voice at {audio_out.PIPER_VOICE}")

    with NoNetwork():
        step("YOLO-World and CLIP", yolo)
        step("MediaPipe hands", hand)
        step("Whisper large-v3", whisper)
        step("Piper voice", piper)
    return {"ok": all(r["ok"] for r in results), "results": results, "t": time.time()}


def offline_env(s: Setup, proof: dict | None) -> dict:
    """The settings that keep every library off the network, if and only if every file is on
    disk and the proof above passed for exactly these files. Otherwise nothing: a missing file
    can then still be fetched."""
    files = model_files(s)
    if all(f["ok"] for f in files) and proof and proof.get("ok") and proof.get("key") == files_key(files):
        return {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "YOLO_OFFLINE": "true"}
    return {}


# ---------------------------------------------------------------------------
# The machine itself

def listening(port: int, timeout_s: float = 0.3) -> bool:
    try:
        socket.create_connection(("127.0.0.1", port), timeout=timeout_s).close()
        return True
    except OSError:
        return False


def holder(s: Setup, port: int) -> tuple[int | None, str]:
    """(pid, command line) of whatever is listening on a port, if it is ours to see."""
    rc, out = s.run(["ss", "-ltnpH", f"sport = :{port}"])
    m = re.search(r"pid=(\d+)", out) if rc == 0 else None
    if not m:
        return None, ""
    rc, args = s.run(["ps", "-o", "args=", "-p", m.group(1)])
    return int(m.group(1)), args.strip()


def machine(s: Setup, running: bool = False) -> list[dict]:
    free = shutil.disk_usage(s.root).free / 1e9
    if free < 5:
        disk = line("disk", "Disk space", RED, f"Only {free:.0f} GB free. Recordings and logs will fill it during the demo: "
                    "delete old folders in recordings/ and logs/.")
    elif free < 20:
        disk = line("disk", "Disk space", AMBER, f"{free:.0f} GB free: enough for today, but clear old recordings soon.")
    else:
        disk = line("disk", "Disk space", GREEN, f"{free:.0f} GB free.")

    names = {s.dash_port: "the dashboard", s.voice_port: "the microphone page"}
    if running:
        return [disk, line("ports", "Ports free", GREEN, f"Ports {s.dash_port} and {s.voice_port} are in use by this run.")]
    taken = [p for p in names if listening(p)]
    if not taken:
        return [disk, line("ports", "Ports free", GREEN, f"Ports {s.dash_port} (dashboard) and {s.voice_port} (microphone page) are free.")]
    pid, args = holder(s, taken[0])
    who = f"{_short(args)} (pid {pid})" if pid else "another program"
    old_run = pid is not None and "main.py" in args
    return [disk, line("ports", "Ports free", RED, f"Port {taken[0]}, which {names[taken[0]]} needs, is taken by {who}"
                       + (": an old run is still going." if old_run else "."),
                       fix=("stop_old_run", "Stop the old run") if old_run else None)]


# What the desktop must not do while it is on the projector: (schema, key, value, what it stops).
QUIET = (("org.gnome.desktop.notifications", "show-banners", "false", "notification banners"),
         ("org.gnome.desktop.sound", "event-sounds", "false", "alert sounds"),
         ("org.gnome.desktop.media-handling", "automount", "false", "mounting the phone as a disk"),
         ("org.gnome.desktop.media-handling", "automount-open", "false", "a Files window when the phone is plugged in"))


def room(s: Setup) -> list[dict]:
    """Things around the demo that have spoiled a run before. None of them holds up Start."""
    temps = []
    for zone in sorted(Path(s.env.get("PROPRIO_THERMAL", "/sys/class/thermal")).glob("thermal_zone*/temp")):
        try:
            temps.append(int(zone.read_text()) / 1000.0)
        except (OSError, ValueError):
            pass
    if not temps:
        heat = advice("heat", "Temperature", "none", "No temperature sensors to read here.")
    elif max(temps) >= 80:
        heat = advice("heat", "Temperature", AMBER, f"The GX10 is at {max(temps):.0f} °C and will slow itself down. Give it air: "
                      "nothing on top of it or against its vents.")
    else:
        heat = advice("heat", "Temperature", GREEN, f"{max(temps):.0f} °C.")

    rc, out = s.run(["iw", "dev"])
    dev = re.search(r"Interface (\S+)", out) if rc == 0 else None
    rc, link = s.run(["iw", "dev", dev.group(1), "link"]) if dev else (1, "")
    freq = re.search(r"freq: (\d+)", link)
    if not freq:
        wifi = advice("wifi", "Wi-Fi band", GREEN, "Wi-Fi is not connected, so it cannot crowd the AirPods.")
    elif int(freq.group(1)) < 5000:
        wifi = advice("wifi", "Wi-Fi band", AMBER, f"Wi-Fi is on 2.4 GHz ({freq.group(1)} MHz), the AirPods' own band, and one chip "
                      "does both. Join a 5 GHz network or switch Wi-Fi off.")
    else:
        wifi = advice("wifi", "Wi-Fi band", GREEN, f"On 5 GHz ({freq.group(1)} MHz), clear of the AirPods.")

    loud = [what for schema, key, value, what in QUIET if s.run(["gsettings", "get", schema, key])[1].strip() != value]
    if loud:
        desk = advice("desktop", "Quiet desktop", AMBER, "The desktop can still interrupt the projector with: " + ", ".join(loud) + ".",
                      fix=("quiet_desktop", "Quiet the desktop"))
    else:
        desk = advice("desktop", "Quiet desktop", GREEN, "No banners, no alert sounds, no Files window when the phone is plugged in.")
    return [heat, wifi, desk]


def ready(lines: list[dict]) -> bool:
    return all(l["level"] == GREEN for l in lines if l["required"])


def main():
    if "--prove" in sys.argv:
        print(json.dumps(prove()))
        return
    s = Setup()
    if "--offline" in sys.argv:           # for tools/live.sh: `eval` what this prints
        for key, value in offline_env(s, read_status(s.status_dir / "models_proof.json")).items():
            print(f"export {key}={value}")
        return
    for l in phone(s) + airpods(s) + servers(s) + gpu(s) + models(s) + machine(s) + room(s):
        print(f"{l['level']:<6} {l['title']:<28} {l['message']}")


if __name__ == "__main__":
    main()
