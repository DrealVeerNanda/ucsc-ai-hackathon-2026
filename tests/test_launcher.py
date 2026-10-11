"""The one-button start: a main loop that nothing stops, the checklist's verdicts, and the keeper
that restarts what dies or hangs."""
import json
import os
import re
import signal
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

import fakes
import launcher
import launcher_checks as checks
import main
from launcher_checks import AMBER, GREEN, RED

ROOT = Path(__file__).resolve().parent.parent


# --- main.py: the loop carries on ---------------------------------------------------------

def loop_args(tmp_path, *extra):
    return main.parse_args(["--ui", "none", "--log-dir", str(tmp_path / "logs"), "--status-dir", str(tmp_path / "status"),
                            "--fps", "200", *extra])


def fake_mods(**swap):
    stream, locate, guidance, voice = fakes.FakeStream(None, 200.0), fakes.FakeLocate(), fakes.FakeGuidance(), fakes.FakeVoice("mug")
    guidance.say = lambda clip: guidance.phrases.append(clip)       # no printing in tests
    m = SimpleNamespace(get_frame=stream.get_frame, locate=locate.locate, fake_locate=locate, guidance=guidance,
                        update=guidance.update, reset=guidance.reset, pop_phrases=guidance.pop_phrases, voice=voice,
                        listen=voice.listen_for_target)
    for name, fn in swap.items():
        setattr(m, name, fn)
    return m


def breaks_sometimes(fn, when=lambda n: n % 3 == 0):
    """fn, except that some calls raise."""
    calls = [0]

    def wrapped(*a, **kw):
        calls[0] += 1
        if when(calls[0]):
            raise RuntimeError(f"injected failure on call {calls[0]}")
        return fn(*a, **kw)
    wrapped.calls = calls
    return wrapped


@pytest.mark.parametrize("step, attr", [("stream", "get_frame"), ("perception", "locate"), ("guidance", "update"),
                                        ("guidance", "pop_phrases"), ("voice", "listen")])
def test_an_exception_in_any_step_is_survived(tmp_path, capsys, step, attr):
    mods = fake_mods()
    setattr(mods, attr, breaks_sometimes(getattr(mods, attr), when=lambda n: n % 3 == 1))    # the very first call fails too
    beat = main.Heartbeat(tmp_path / "status")
    main.run(loop_args(tmp_path, "--frames", "12"), mods, beat)
    assert beat.frames == 12
    assert beat.faults.counts[step] >= 1
    assert f"{step} failed, carrying on" in capsys.readouterr().err
    status = json.loads((tmp_path / "status" / "pipeline.json").read_text())
    assert status["state"] == "stopped" and status["frames"] == 12 and status["errors"][step] >= 1


class FakeDash:
    """A dashboard whose publish and say always fail, and where someone presses q after a while."""
    instances = []

    def __init__(self, host, port):
        self.turns = 0
        FakeDash.instances.append(self)

    def start(self):
        return "http://fake/"

    def publish(self, *a):
        raise RuntimeError("dashboard is broken")

    say = publish

    def pop_keys(self):
        self.turns += 1
        return ["q"] if self.turns == 40 else []

    def stop(self):
        pass


def test_a_broken_dashboard_does_not_end_the_run_and_q_still_quits(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "Dashboard", FakeDash)
    monkeypatch.setattr(main.webbrowser, "open", lambda url: None)
    beat = main.Heartbeat("off")
    args = main.parse_args(["--ui", "web", "--no-open", "--log-dir", str(tmp_path), "--status-dir", "off", "--fps", "200"])
    main.run(args, fake_mods(), beat)
    assert beat.frames > 3 and beat.faults.counts["dashboard"] >= beat.frames
    assert FakeDash.instances[-1].turns == 40


def test_q_quits_even_when_perception_fails_on_every_frame(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "Dashboard", FakeDash)
    FakeDash.publish = lambda self, *a: None
    def never(frame, target):
        raise RuntimeError("no detector")
    beat = main.Heartbeat("off")
    args = main.parse_args(["--ui", "web", "--no-open", "--log-dir", str(tmp_path), "--status-dir", "off"])
    main.run(args, fake_mods(locate=never), beat)
    assert beat.frames == 0 and beat.faults.counts["perception"] >= 1


def test_no_frame_does_not_spin_the_loop(tmp_path):
    """A camera that has nothing yet: the loop must sleep between asks, not hammer it."""
    calls, t0, stream = [0], time.monotonic(), fakes.FakeStream(None, 200.0)

    def get_frame(wait: float = 0.0):
        calls[0] += 1
        return None if time.monotonic() - t0 < 0.4 else stream.get_frame()
    main.run(loop_args(tmp_path, "--frames", "1"), fake_mods(get_frame=get_frame), main.Heartbeat("off"))
    assert calls[0] < 40            # 0.4 s at one ask per 20 ms; a spinning loop makes thousands


def test_a_camera_that_can_wait_is_asked_to_when_it_has_nothing_new(tmp_path):
    waits, stream = [], fakes.FakeStream(None, 50.0)

    def get_frame(wait: float = 0.0):
        waits.append(wait)
        if wait:
            time.sleep(0.02)                 # stream.py blocks here until the next frame arrives
        return stream.get_frame()
    args = loop_args(tmp_path, "--frames", "3")
    main.run(args, fake_mods(get_frame=get_frame), main.Heartbeat("off"))
    assert 0.0 in waits and max(waits) > 0                # asked plainly first, then asked to wait
    rows = (tmp_path / "logs").glob("run_*.csv").__next__().read_text().splitlines()
    stream_ms = [float(r.split(",")[5]) for r in rows[1:]]
    assert max(stream_ms) < 15                             # the wait itself is not counted as the stream's time


def test_a_voice_module_that_returns_nothing_at_once_does_not_spin(tmp_path):
    answers = iter(["", "", "", "mug"])
    asked = []

    def listen():
        asked.append(time.monotonic())
        return next(answers)
    main.run(loop_args(tmp_path, "--frames", "1"), fake_mods(listen=listen), main.Heartbeat("off"))
    assert len(asked) == 4 and asked[-1] - asked[0] >= 0.55


def test_heartbeat_tells_waiting_for_speech_from_being_stuck(tmp_path, monkeypatch):
    beat = main.Heartbeat(tmp_path)
    beat.warm = True
    beat.at = ("listening", time.monotonic() - 600)             # ten minutes waiting for the wearer
    s = beat.snapshot()
    assert s["state"] == "listening" and s["ok"]
    beat.at = ("perception", time.monotonic() - main.STUCK_S - 1)
    s = beat.snapshot()
    assert s["state"] == "stuck" and not s["ok"] and "perception" in s["message"]
    beat.warm = False                                            # still loading: allowed much longer
    assert beat.snapshot()["state"] == "starting"


def test_heartbeat_reports_a_step_that_keeps_failing_and_a_missing_camera(tmp_path):
    beat = main.Heartbeat(tmp_path)
    beat.warm, beat.last_frame = True, time.monotonic()
    assert beat.snapshot()["state"] == "running"
    beat.last_frame = time.monotonic() - 10
    assert beat.snapshot()["state"] == "no_camera"
    beat.last_frame = time.monotonic()
    try:
        raise RuntimeError("cuda gone")
    except RuntimeError as e:
        beat.faults.failed("perception", e)
    beat.faults.since["perception"] = time.monotonic() - 30
    s = beat.snapshot()
    assert s["state"] == "failing" and s["failing"]["perception"] >= 30 and "cuda gone" in s["message"]
    beat.at = ("listening", time.monotonic())          # no frame is looked at while it waits for the wearer, so
    assert beat.snapshot()["state"] == "listening"     # perception cannot be "still failing": that would get it restarted
    beat.faults.failed("voice", RuntimeError("no microphone"))
    beat.faults.since["voice"] = time.monotonic() - 30
    assert beat.snapshot()["state"] == "failing"       # a voice module that keeps failing is not "waiting for the wearer"
    beat.faults.ok("voice")
    assert beat.snapshot()["state"] == "listening"


def test_heartbeat_file_is_written_while_the_loop_waits_for_the_wearer(tmp_path):
    release = threading.Event()

    def listen():
        release.wait(5)
        return "mug"
    beat = main.Heartbeat(tmp_path / "status", every_s=0.05).start()
    t = threading.Thread(target=main.run, args=(loop_args(tmp_path, "--frames", "1"), fake_mods(listen=listen), beat))
    t.start()
    try:
        time.sleep(0.4)
        status = json.loads((tmp_path / "status" / "pipeline.json").read_text())
        assert status["state"] == "listening" and status["ok"] and status["pid"] == os.getpid()
        assert time.time() - status["t"] < 0.3
    finally:
        release.set()
        t.join(5)


# --- the checklist ----------------------------------------------------------------------------

BT_INFO = """Device 18:3F:70:53:BD:95 (public)
\tName: AirPods Pro
\tAlias: AirPods Pro
\tPaired: yes
\tConnected: {connected}
\tUUID: Audio Sink                (0000110b-0000-1000-8000-00805f9b34fb)
\tUUID: Handsfree                 (0000111e-0000-1000-8000-00805f9b34fb)"""
WPCTL = """PipeWire 'pipewire-0' [1.0.5, asus@7AHF, cookie:5299301]
Audio
 ├─ Devices:
 │      42. Built-in Audio                      [alsa]
 │      77. AirPods Pro                         [bluez5]
 │
 ├─ Sinks:
 │      55. Built-in Audio Digital Stereo (HDMI) [vol: 0.40]
 │  *   81. AirPods Pro                         [vol: 0.60]
 │
 ├─ Sink endpoints:
 │
 ├─ Sources:
 │
 └─ Streams:
Video
"""
PROFILES = """  Object: size 88, type Spa:Pod:Object:Param:Profile (262147), id Spa:Enum:ParamId:EnumProfile (8)
    Prop: key Spa:Pod:Object:Param:Profile:index (1), flags 00000000
      Int 0
    Prop: key Spa:Pod:Object:Param:Profile:name (2), flags 00000000
      String "off"
    Prop: key Spa:Pod:Object:Param:Profile:index (1), flags 00000000
      Int 1
    Prop: key Spa:Pod:Object:Param:Profile:name (2), flags 00000000
      String "a2dp-sink"
    Prop: key Spa:Pod:Object:Param:Profile:index (1), flags 00000000
      Int 2
    Prop: key Spa:Pod:Object:Param:Profile:name (2), flags 00000000
      String "headset-head-unit-msbc"
"""


class Machine:
    """Answers the commands the checks run, from a table: {"command start": (exit code, output)}."""

    def __init__(self, **answers):
        self.answers = {"journalctl": (1, ""), "gsettings get": (0, "false"), "iw dev": (0, ""),
                        "bluetoothctl show": (0, "Powered: yes\nDiscoverable: no\nDiscovering: no"), **answers}
        self.asked = []

    def __call__(self, cmd, timeout=4.0):
        text = " ".join(cmd)
        self.asked.append(text)
        for start in sorted(self.answers, key=len, reverse=True):
            if text.startswith(start):
                return self.answers[start]
        return 127, f"{cmd[0]}: not installed"


def setup(tmp_path, machine=None, **env):
    (tmp_path / "out" / "status").mkdir(parents=True, exist_ok=True)
    return checks.Setup(root=tmp_path, env=env, run=machine or Machine(), home=tmp_path / "home")


def by_id(lines):
    return {l["id"]: l for l in lines}


def write_phone(tmp_path, **status):
    (tmp_path / "out" / "status").mkdir(parents=True, exist_ok=True)
    (tmp_path / "out" / "status" / "phone.json").write_text(json.dumps({"t": time.time(), **status}))


def with_bridge(tmp_path):
    (tmp_path / "tools").mkdir(exist_ok=True)
    (tmp_path / "tools" / "phone_bridge.py").write_text("")


def test_no_phone_on_the_cable_is_one_red_line_and_the_rest_wait(tmp_path):
    with_bridge(tmp_path)
    got = by_id(checks.phone(setup(tmp_path, Machine(**{"idevice_id": (0, "")}))))
    assert got["phone_usb"]["level"] == RED and "Plug the phone" in got["phone_usb"]["message"]
    assert got["phone_trust"]["level"] == RED and got["phone_trust"]["blocked"]
    assert got["phone_stream"]["level"] == RED and "missing" in got["phone_stream"]["message"]


def test_a_phone_that_does_not_trust_us_says_what_to_tap(tmp_path):
    with_bridge(tmp_path)
    m = Machine(**{"idevice_id": (0, "00008130-001A2B3C4D5E6F78"),
                   "idevicepair validate": (1, "ERROR: Please accept the trust dialog on the screen of device 0000, then attempt to pair again.")})
    got = by_id(checks.phone(setup(tmp_path, m)))
    assert got["phone_usb"]["level"] == GREEN
    assert got["phone_trust"]["level"] == RED and "Tap Trust" in got["phone_trust"]["message"]
    assert got["phone_trust"]["fix"]["action"] == "pair_phone"


def test_streaming_phone_is_green_with_its_numbers_and_is_not_asked_about_trust(tmp_path):
    with_bridge(tmp_path)
    write_phone(tmp_path, state="streaming", ok=True, message="ok", fps=29.8, frame_age_s=0.031)
    m = Machine(**{"idevice_id": (0, "00008130-001A2B3C4D5E6F78")})
    got = by_id(checks.phone(setup(tmp_path, m)))
    assert [got[k]["level"] for k in ("phone_usb", "phone_trust", "phone_stream")] == [GREEN] * 3
    assert "30 frames a second" in got["phone_stream"]["message"] and "31 ms" in got["phone_stream"]["message"]
    assert not any(a.startswith("idevicepair") for a in m.asked)
    assert not any("8888" in a or "iproxy" in a for a in m.asked)       # the phone's own port is never touched


def test_phone_status_that_is_stale_or_not_ok_is_red_with_the_reason(tmp_path):
    with_bridge(tmp_path)
    m = Machine(**{"idevice_id": (0, "00008130-001A2B3C4D5E6F78"), "idevicepair validate": (0, "SUCCESS: Validated pairing with device 0000")})
    write_phone(tmp_path, state="waiting", ok=False, message="Open PathOn on the phone and tap Start.")
    stream = by_id(checks.phone(setup(tmp_path, m)))["phone_stream"]
    assert stream["level"] == RED and stream["message"] == "Open PathOn on the phone and tap Start."
    write_phone(tmp_path, ok=True, state="streaming", t=time.time() - 42)
    stream = by_id(checks.phone(setup(tmp_path, m)))["phone_stream"]
    assert stream["level"] == RED and "42 s ago" in stream["message"] and stream["fix"]["action"] == "restart_bridge"


def test_slow_stream_is_amber_and_a_missing_bridge_does_not_hold_up_start(tmp_path):
    m = Machine(**{"idevice_id": (0, "00008130-001A2B3C4D5E6F78"), "idevicepair validate": (0, "SUCCESS: Validated pairing")})
    got = by_id(checks.phone(setup(tmp_path, m)))
    assert got["phone_stream"]["level"] == AMBER and not got["phone_stream"]["required"]
    with_bridge(tmp_path)
    write_phone(tmp_path, state="streaming", ok=True, fps=9.0, frame_age_s=0.05)
    assert by_id(checks.phone(setup(tmp_path, m)))["phone_stream"]["level"] == AMBER


def test_replaying_a_recording_needs_no_phone(tmp_path):
    lines = checks.phone(setup(tmp_path, STREAM_SOURCE="recordings/live_2025"))
    assert [l["id"] for l in lines] == ["phone_stream"] and not lines[0]["required"]


def test_usb_dropouts_are_advice_about_the_cable(tmp_path):
    drop = "Oct 10 21:02:11 7AHF kernel: usb 1-1: USB disconnect, device number 12"
    line = checks.cable(setup(tmp_path, Machine(journalctl=(0, drop + "\n" + drop))))
    assert line["level"] == AMBER and "2 times" in line["message"] and "slack loop" in line["message"] and not line["required"]
    assert checks.cable(setup(tmp_path))["level"] == GREEN


def pods_machine(connected="yes", sink="bluez_output.18_3F_70_53_BD_95.1", profiles=PROFILES, **more):
    return Machine(**{"bluetoothctl devices Connected": (0, "Device 18:3F:70:53:BD:95 AirPods Pro" if connected == "yes" else ""),
                      "bluetoothctl devices Paired": (0, "Device 18:3F:70:53:BD:95 AirPods Pro"),
                      "bluetoothctl info": (0, BT_INFO.format(connected=connected)),
                      "wpctl inspect @DEFAULT_AUDIO_SINK@": (0, f'  * node.name = "{sink}"\n    node.description = "AirPods Pro"'),
                      "wpctl inspect 81": (0, 'node.name = "bluez_output.18_3F_70_53_BD_95.1"'),
                      "wpctl inspect 55": (0, 'node.name = "alsa_output.hdmi"'),
                      "wpctl status": (0, WPCTL), "pw-cli enum-params 77": (0, profiles), **more})


def test_airpods_connected_default_output_and_both_modes_are_green(tmp_path):
    got = by_id(checks.airpods(setup(tmp_path, pods_machine())))
    assert [got[k]["level"] for k in ("airpods", "airpods_output", "airpods_mic", "bluetooth")] == [GREEN] * 4
    assert "headset-head-unit-msbc" in got["airpods_mic"]["message"]


def test_airpods_not_connected_offers_connect_and_the_rest_wait(tmp_path):
    got = by_id(checks.airpods(setup(tmp_path, pods_machine(connected="no"))))
    assert got["airpods"]["level"] == RED and got["airpods"]["fix"]["action"] == "connect_airpods"
    assert got["airpods_output"]["blocked"] and got["airpods_mic"]["blocked"]


def test_sound_going_to_the_monitor_is_red_with_a_button(tmp_path):
    m = pods_machine(sink="alsa_output.platform-NVDA2014_00.hdmi-stereo")
    out = by_id(checks.airpods(setup(tmp_path, m)))["airpods_output"]
    assert out["level"] == RED and out["fix"]["action"] == "use_airpods_output"
    assert checks.bluez_sink(setup(tmp_path, m)) == 81


def test_a_missing_or_refused_mode_asks_for_a_reconnect(tmp_path):
    no_stereo = PROFILES.replace("a2dp-sink", "a2dp-source")
    mic = by_id(checks.airpods(setup(tmp_path, pods_machine(profiles=no_stereo))))["airpods_mic"]
    assert mic["level"] == RED and mic["fix"]["action"] == "reconnect_airpods"
    refused = pods_machine(journalctl=(0, "wireplumber[2571]: failed: invalid profile\nwireplumber[2571]: failed: invalid profile"))
    s = setup(tmp_path, refused)
    mic = by_id(checks.airpods(s))["airpods_mic"]
    assert mic["level"] == AMBER and "2 time" in mic["message"] and mic["fix"]["action"] == "reconnect_airpods"
    s.reconnected = time.time()                    # the journal is then only asked about what came after
    checks.airpods(s)
    assert f"@{s.reconnected:.0f}" in " ".join(a for a in refused.asked if a.startswith("journalctl --user"))


def test_any_connected_headset_is_used_before_the_default_airpods(tmp_path):
    other = "AA:BB:CC:DD:EE:FF"
    m = Machine(**{"bluetoothctl devices Connected": (0, f"Device {other} Someone's Buds"),
                   f"bluetoothctl info {other}": (0, BT_INFO.format(connected="yes").replace("AirPods Pro", "Someone's Buds")),
                   "bluetoothctl info": (0, BT_INFO.format(connected="no"))})
    assert checks.find_headset(setup(tmp_path, m))["addr"] == other
    nothing = Machine(**{"bluetoothctl devices": (0, ""), "bluetoothctl info": (0, BT_INFO.format(connected="no"))})
    assert checks.find_headset(setup(tmp_path, nothing))["addr"] == checks.AIRPODS
    assert checks.find_headset(setup(tmp_path, nothing, PROPRIO_HEADSET=other))["addr"] == other


def test_scanning_bluetooth_is_called_out(tmp_path):
    line = checks.bluetooth(setup(tmp_path, Machine(**{"bluetoothctl show": (0, "Powered: yes\nDiscoverable: no\nDiscovering: yes")})))
    assert line["level"] == AMBER and line["fix"]["action"] == "quiet_bluetooth"


@pytest.fixture
def health_server():
    code = [200]

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(code[0])
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")
    server = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_address[1], code
    server.shutdown()
    server.server_close()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_model_servers_up_loading_and_down(tmp_path, health_server):
    port, code = health_server
    s = setup(tmp_path, VOICE_LLM_URL=f"http://127.0.0.1:{port}/v1", COSMOS_URL=f"http://127.0.0.1:{free_port()}/v1")
    got = by_id(checks.servers(s))
    assert got["nemotron"]["level"] == GREEN and str(port) in got["nemotron"]["message"]
    assert got["cosmos"]["level"] == RED and got["cosmos"]["fix"]["action"] == "start_cosmos"
    code[0] = 503
    assert by_id(checks.servers(s))["nemotron"]["level"] == AMBER


def gpu_machine(apps: str, who: dict):
    answers = {"nvidia-smi --query-gpu": (0, "NVIDIA GB10, 12, 51"), "nvidia-smi --query-compute-apps": (0, apps)}
    answers.update({f"ps -o user=,etimes=,args= -p {pid}": (0, row) for pid, row in who.items()})
    return Machine(**answers)


def test_gpu_with_only_our_servers_is_green_and_a_teammates_job_is_named(tmp_path):
    ours = {264384: "asus 24616 /home/asus/llama.cpp/llama-server --model n.gguf --alias nemotron --host 127.0.0.1 --port 8000 -ngl 99",
            272120: "asus 22000 /home/asus/llama.cpp/llama-server --model c.gguf --alias cosmos --host 127.0.0.1 --port 8001"}
    line = checks.gpu(setup(tmp_path, gpu_machine("264384, 23142\n272120, 10869", ours)))[0]
    assert line["level"] == GREEN and "Cosmos and Nemotron" in line["message"]
    theirs = {**ours, 633161: "asus 7200 /home/asus/x/.venv/bin/python train.py --epochs 50"}
    line = checks.gpu(setup(tmp_path, gpu_machine("264384, 23142\n272120, 10869\n633161, 3714", theirs)))[0]
    assert line["level"] == AMBER and "python train.py" in line["message"] and "3.6 GB" in line["message"] and "2 h" in line["message"]
    line = checks.gpu(setup(tmp_path, gpu_machine("633161, 3714", theirs)), ours={633161})[0]
    assert line["level"] == GREEN and "this demo" in line["message"]


def test_gpu_missing_is_red_and_busy_servers_before_the_start_are_amber(tmp_path):
    assert checks.gpu(setup(tmp_path, Machine()))[0]["level"] == RED
    busy = Machine(**{"nvidia-smi --query-gpu": (0, "NVIDIA GB10, 95, 56"), "nvidia-smi --query-compute-apps": (0, "")})
    history = []
    levels = [checks.gpu(setup(tmp_path, busy), busy=history)[0]["level"] for _ in range(3)]
    assert levels == [GREEN, GREEN, AMBER]
    assert checks.gpu(setup(tmp_path, busy), running=True, busy=history)[0]["level"] == GREEN


def put_models(tmp_path, skip=()):
    """Every file the checklist looks for, the right size but empty (sparse)."""
    s = setup(tmp_path)
    snap = s.home / ".cache" / "huggingface" / "hub" / "models--openai--whisper-large-v3" / "snapshots" / "abc"
    snap.mkdir(parents=True, exist_ok=True)
    for f in checks.model_files(s):
        if f["name"] in skip:
            continue
        path = Path(f["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            fh.truncate(int(4e9) if path.suffix in (".safetensors", ".gguf") else int(4e8))
    return s


def test_models_missing_is_red_and_names_how_to_get_it(tmp_path):
    s = put_models(tmp_path, skip=("Piper voice",))
    line = checks.models(s)[0]
    assert line["level"] == RED and "Piper voice" in line["message"] and "fetch_models.sh piper" in line["message"]
    assert checks.offline_env(s, {"ok": True, "key": checks.files_key(checks.model_files(s))}) == {}


def test_offline_only_when_every_file_is_there_and_was_loaded_with_the_network_blocked(tmp_path):
    s = put_models(tmp_path)
    key = checks.files_key(checks.model_files(s))
    assert checks.models(s)[0]["level"] == AMBER and checks.offline_env(s, None) == {}
    proof = {"key": key, "ok": True, "results": [{"name": "Whisper large-v3", "ok": True, "note": "1.1 s"}]}
    assert checks.models(s, proof)[0]["level"] == GREEN
    assert checks.offline_env(s, proof) == {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "YOLO_OFFLINE": "true"}
    failed = {"key": key, "ok": False, "results": [{"name": "Whisper large-v3", "ok": False, "note": "OSError: not cached"}]}
    assert checks.models(s, failed)[0]["level"] == AMBER and checks.offline_env(s, failed) == {}
    (s.root / "models" / checks.YOLO_WEIGHTS).write_bytes(b"x" * int(21e6))       # a file changed: the old proof is void
    assert checks.offline_env(s, proof) == {} and checks.models(s, proof)[0]["level"] == AMBER


def test_a_cut_short_download_counts_as_missing(tmp_path):
    s = put_models(tmp_path)
    (s.root / "models" / checks.YOLO_WEIGHTS).write_bytes(b"x" * 1000)
    line = checks.models(s)[0]
    assert line["level"] == RED and "YOLO-World weights" in line["message"]


def test_no_network_blocks_the_internet_but_not_this_machine(health_server):
    port, _ = health_server
    with checks.NoNetwork():
        with pytest.raises(OSError, match="network blocked"):
            socket.create_connection(("93.184.216.34", 80), timeout=1)
        with pytest.raises(urllib.error.URLError, match="network blocked"):
            urllib.request.urlopen("https://huggingface.co/openai/whisper-large-v3", timeout=1)
        assert urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).status == 200
    assert socket.getaddrinfo("localhost", 80)


def test_ports_taken_by_an_old_run_is_red_with_a_stop_button(tmp_path, health_server):
    port, _ = health_server
    m = Machine(**{"ss": (0, f'LISTEN 0 5 0.0.0.0:{port} 0.0.0.0:* users:(("python",pid=4242,fd=5))'),
                   "ps -o args= -p 4242": (0, "python main.py --stream real --perception real")})
    s = setup(tmp_path, m, PROPRIO_DASH_PORT=str(port), VOICE_PORT=str(free_port()))
    ports = by_id(checks.machine(s))["ports"]
    assert ports["level"] == RED and "pid 4242" in ports["message"] and ports["fix"]["action"] == "stop_old_run"
    assert by_id(checks.machine(s, running=True))["ports"]["level"] == GREEN
    free = setup(tmp_path, m, PROPRIO_DASH_PORT=str(free_port()), VOICE_PORT=str(free_port()))
    assert by_id(checks.machine(free))["ports"]["level"] == GREEN


def test_room_lines_never_hold_up_start(tmp_path):
    zone = tmp_path / "thermal" / "thermal_zone0"
    zone.mkdir(parents=True)
    (zone / "temp").write_text("86000\n")
    m = Machine(**{"iw dev wlan0 link": (0, "Connected to aa:bb\n\tSSID: hall\n\tfreq: 2437.0"), "iw dev": (0, "phy#0\n\tInterface wlan0"),
                   "gsettings get org.gnome.desktop.notifications": (0, "true")})
    got = by_id(checks.room(setup(tmp_path, m, PROPRIO_THERMAL=str(tmp_path / "thermal"))))
    assert [got[k]["level"] for k in ("heat", "wifi", "desktop")] == [AMBER] * 3 and got["phone_cable"]["level"] == GREEN
    assert "86" in got["heat"]["message"] and "2.4 GHz" in got["wifi"]["message"] and "notification banners" in got["desktop"]["message"]
    assert checks.ready(list(got.values()))
    (zone / "temp").write_text("47800\n")
    m.answers.update({"iw dev wlan0 link": (0, "freq: 5660.0"), "gsettings get org.gnome.desktop.notifications": (0, "false")})
    assert all(l["level"] == GREEN for l in checks.room(setup(tmp_path, m, PROPRIO_THERMAL=str(tmp_path / "thermal"))))


def test_ready_needs_every_required_line_green():
    ok = checks.line("a", "A", GREEN, "")
    assert checks.ready([ok, checks.advice("b", "B", AMBER, "")])
    assert not checks.ready([ok, checks.line("c", "C", AMBER, "")])


# --- the keeper ---------------------------------------------------------------------------------

def test_judge_knows_waiting_from_hanging():
    now = 1000.0
    hb = lambda **kw: {"t": now - 0.5, "state": "running", "ok": True, "message": "fine", "failing": {}, **kw}
    assert launcher.judge(False, hb(), now - 100, now)[0] == "dead"
    assert launcher.judge(True, None, now - 5, now) == ("ok", "starting")
    assert launcher.judge(True, None, now - launcher.BOOT_S - 1, now)[0] == "hung"
    assert launcher.judge(True, hb(), now - 100, now)[0] == "ok"
    # ten minutes of silence from the wearer is fine as long as the heartbeat keeps coming
    assert launcher.judge(True, hb(state="listening", doing="listening", doing_s=600), now - 700, now)[0] == "ok"
    assert launcher.judge(True, hb(t=now - launcher.FROZEN_S - 1), now - 100, now)[0] == "hung"
    assert launcher.judge(True, hb(state="stuck", ok=False, message="Stuck in perception for 31 s."), now - 100, now) == \
        ("hung", "Stuck in perception for 31 s.")
    assert launcher.judge(True, hb(state="failing", failing={"perception": 6.0}), now - 100, now)[0] == "ok"
    assert launcher.judge(True, hb(state="failing", failing={"perception": 21.0}), now - 100, now)[0] == "hung"
    assert launcher.judge(True, hb(state="no_camera", ok=False), now - 100, now)[0] == "ok"      # the phone's trouble, not its own


FAKE_PIPELINE = '''
import json, os, signal, sys, time
status = os.path.join(os.getcwd(), "out", "status", "pipeline.json")
mode = open("mode.txt").read().strip() if os.path.exists("mode.txt") else "ok"
if mode == "crash":
    sys.exit(3)
t0 = time.time()
while True:
    state = "stuck" if mode == "stuck" and time.time() - t0 > 0.3 else "listening"
    with open(status + ".tmp", "w") as f:
        json.dump({"t": time.time(), "pid": os.getpid(), "state": state, "ok": state != "stuck", "message": "fake " + state,
                   "heard": "blue cup", "said": "Looking", "target": "cup", "failing": {}}, f)
    os.replace(status + ".tmp", status)
    time.sleep(0.05)
'''


@pytest.fixture
def keeper(tmp_path):
    (tmp_path / "fake_main.py").write_text(FAKE_PIPELINE)
    sup = launcher.Supervisor(root=tmp_path, env={"PROPRIO_OPEN": "0", "PROPRIO_DASH_PORT": str(free_port()), "PATH": os.environ["PATH"]},
                              run=Machine(), pipeline_argv=[sys.executable, "fake_main.py"], home=tmp_path / "home")
    sup.frozen_s, sup.boot_s, sup.backoff_s, sup.settled_s, sup.stop_wait_s = 0.6, 3.0, (0.1, 0.2, 0.4, 5.0), 60.0, 3.0
    yield sup
    sup.pipeline.stop(signal.SIGKILL, 0.5)


def tick_until(sup, condition, timeout_s=8.0):
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        sup.tick()
        if condition():
            return True
        time.sleep(0.05)
    return False


def test_start_needs_green_lines_or_start_anyway(keeper):
    ok, message = keeper.start()
    assert not ok and "Not ready" in message and keeper.mode == "checklist"
    assert keeper.start(anyway=True)[0] and keeper.mode == "running" and keeper.pipeline.alive()
    first = keeper.pipeline.pid
    assert keeper.start(anyway=True) == (True, "Already running.") and keeper.pipeline.pid == first     # a second click changes nothing


def test_a_killed_pipeline_is_restarted_within_a_couple_of_seconds(keeper):
    keeper.start(anyway=True)
    assert tick_until(keeper, lambda: keeper.heartbeat() is not None)
    first = keeper.pipeline.pid
    os.kill(first, signal.SIGKILL)
    t = time.monotonic()
    assert tick_until(keeper, lambda: keeper.pipeline.pid not in (None, first) and keeper.heartbeat() is not None)
    assert time.monotonic() - t < 2.5
    assert keeper.restarts == 1 and "exit" in keeper.trouble
    state = keeper.state()
    assert state["mode"] == "running" and state["heard"] == "blue cup" and state["said"] == "Looking"
    assert state["lines"][0]["id"] == "pipeline" and state["lines"][0]["level"] == GREEN


def test_a_frozen_pipeline_is_noticed_and_restarted(keeper):
    keeper.start(anyway=True)
    assert tick_until(keeper, lambda: keeper.heartbeat() is not None)
    first = keeper.pipeline.pid
    os.kill(first, signal.SIGSTOP)                       # alive, but not even its heartbeat moves
    time.sleep(0.3)
    keeper.frozen_s = 30.0                               # (long enough to look at the page before it acts)
    line = keeper.pipeline_line(time.time() + 4)
    assert line["level"] == AMBER and "No word from the pipeline for 4 s" in line["message"]
    keeper.frozen_s = 0.6
    assert tick_until(keeper, lambda: keeper.pipeline.pid not in (None, first) and keeper.heartbeat() is not None)
    assert "frozen" in keeper.trouble and keeper.restarts == 1
    with pytest.raises(OSError):
        os.kill(first, 0)                                # the frozen one is really gone


def test_a_pipeline_that_reports_being_stuck_is_restarted(keeper):
    (keeper.root / "mode.txt").write_text("stuck")
    keeper.start(anyway=True)
    first = keeper.pipeline.pid
    assert tick_until(keeper, lambda: keeper.restarts >= 1)
    assert keeper.trouble.startswith("fake stuck") and keeper.pipeline.pid != first


def test_a_pipeline_that_keeps_crashing_is_restarted_more_and_more_slowly(keeper):
    (keeper.root / "mode.txt").write_text("crash")
    keeper.start(anyway=True)
    waits = []
    end = time.monotonic() + 6
    while len(waits) < 3 and time.monotonic() < end:
        before = keeper.in_a_row
        keeper.tick()
        if keeper.in_a_row != before:
            waits.append(round(keeper.next_start - time.time(), 1))
        time.sleep(0.02)
    assert waits == [0.2, 0.4, 5.0]                      # the first restart of a good run would be 0.1
    assert "exit code 3" in keeper.trouble
    assert keeper.state()["lines"][0]["level"] == RED
    (keeper.root / "mode.txt").write_text("ok")          # and it comes back by itself once the cause is gone
    keeper.next_start = 0.0
    assert tick_until(keeper, lambda: keeper.heartbeat() is not None)


def test_stop_ends_the_pipeline_and_returns_to_the_checklist(keeper):
    keeper.start(anyway=True)
    assert tick_until(keeper, lambda: keeper.heartbeat() is not None)
    pid = keeper.pipeline.pid
    keeper.stop()
    assert tick_until(keeper, lambda: keeper.mode == "checklist")
    with pytest.raises(OSError):
        os.kill(pid, 0)
    keeper.tick()
    assert keeper.pipeline.pid is None                   # and it is not started again


def test_a_new_keeper_carries_on_with_the_running_pipeline(keeper, tmp_path):
    keeper.start(anyway=True)
    assert tick_until(keeper, lambda: keeper.heartbeat() is not None)
    pid = keeper.pipeline.pid
    second = launcher.Supervisor(root=tmp_path, env=keeper.env, run=Machine(), pipeline_argv=keeper.pipeline.argv, home=tmp_path / "home")
    second.frozen_s, second.backoff_s = 0.6, (0.1,)
    second.resume()
    assert second.mode == "running" and second.pipeline.pid == pid and second.heartbeat() is not None
    second.tick()
    assert second.pipeline.pid == pid                    # no second pipeline
    os.kill(pid, signal.SIGKILL)                         # it is not this keeper's child, and is still watched
    keeper.pipeline.popen.wait()                         # (in real life its parent is gone and init does this)
    assert tick_until(second, lambda: second.pipeline.pid not in (None, pid) and second.heartbeat() is not None)
    second.pipeline.stop(signal.SIGKILL, 0.5)
    keeper.pipeline.forget()


def test_a_run_killed_together_with_its_keeper_is_started_again_but_an_old_one_is_not(keeper, tmp_path):
    keeper.start(anyway=True)
    assert tick_until(keeper, lambda: keeper.heartbeat() is not None)
    os.kill(keeper.pipeline.pid, signal.SIGKILL)         # someone's `pkill python` took everything
    keeper.pipeline.popen.wait()

    def new_keeper():
        k = launcher.Supervisor(root=tmp_path, env=keeper.env, run=Machine(), pipeline_argv=keeper.pipeline.argv, home=tmp_path / "home")
        k.resume()
        return k
    second = new_keeper()
    assert second.mode == "running" and second.pipeline.pid is None
    assert tick_until(second, lambda: second.heartbeat() is not None) and second.restarts == 1
    second.pipeline.stop(signal.SIGKILL, 0.5)
    status = tmp_path / "out" / "status" / "supervisor.json"
    status.write_text(json.dumps(dict(json.loads(status.read_text()), t=time.time() - 3600)))     # the power went an hour ago
    assert new_keeper().mode == "checklist"
    keeper.pipeline.forget()


def test_recording_is_on_by_default_into_a_dated_folder_and_can_be_switched_off(keeper):
    env = keeper.pipeline_env()
    assert re.fullmatch(r"recordings/demo_\d{8}_\d{6}", env["STREAM_RECORD"]) and int(env["STREAM_RECORD_SECONDS"]) >= 3600
    assert env["OPEN"] == "0"
    keeper.toggle_record()
    assert "STREAM_RECORD" not in keeper.pipeline_env() and keeper.record_line()["level"] == AMBER
    keeper.toggle_record()
    keeper.s.env["STREAM_SOURCE"] = "recordings/live_2025"           # a replay is not recorded again
    assert "STREAM_RECORD" not in keeper.pipeline_env()


def test_the_pipeline_is_started_offline_only_with_proof(keeper):
    assert "HF_HUB_OFFLINE" not in keeper.pipeline_env()
    put_models(keeper.root)
    keeper.proof = {"ok": True, "key": checks.files_key(checks.model_files(keeper.s))}
    env = keeper.pipeline_env()
    assert env["HF_HUB_OFFLINE"] == "1" and env["YOLO_OFFLINE"] == "true"


def test_dropped_airpods_are_reconnected_politely(keeper):
    asked = keeper.s.run.asked
    keeper.s.run.answers.update({"bluetoothctl info": (0, BT_INFO.format(connected="no")), "bluetoothctl devices": (0, ""),
                                 "bluetoothctl connect": (1, "Failed to connect: org.bluez.Error.Failed br-connection-page-timeout")})
    keeper.lines["airpods"] = checks.airpods(keeper.s)
    connects = lambda: sum(a.startswith("bluetoothctl connect") for a in asked)
    now = time.time()
    for i in range(50):                                   # five seconds of ticks
        keeper.keep_airpods(now + i * 0.1)
        time.sleep(0.01)
    time.sleep(0.2)
    assert connects() == 1                                # not once per tick
    keeper.keep_airpods(now + 11)
    time.sleep(0.2)
    assert connects() == 2
    assert "did not connect" in keeper.notes["airpods"]["message"]


def test_a_dead_model_server_is_started_again_but_not_in_a_loop(keeper, monkeypatch):
    started = []
    monkeypatch.setattr(keeper, "start_server", started.append)
    keeper.lines["servers"] = [checks.line("cosmos", "Cosmos", RED, "not running", fix=("start_cosmos", "Start Cosmos")),
                               checks.line("nemotron", "Nemotron", GREEN, "up")]
    now = time.time()
    for i in range(30):
        keeper.keep_servers(now + i)
    assert started == ["cosmos"]
    keeper.keep_servers(now + 61)
    assert started == ["cosmos", "cosmos"]


def test_ear_checks_refuse_to_run_while_the_pipeline_has_the_airpods(keeper):
    keeper.start(anyway=True)
    ok, message = keeper.do("mic_check")
    assert not ok and "Stop the demo first" in message
    lines = by_id(keeper.all_lines())
    assert lines["sound_check"]["fix"] is None and lines["mic_check"]["fix"] is None


def test_page_and_buttons_over_http(keeper):
    server = ThreadingHTTPServer(("127.0.0.1", 0), launcher.make_handler(keeper))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert b"<title>Proprio, start</title>" in urllib.request.urlopen(base + "/").read()
        assert urllib.request.urlopen(base + "/fonts/barlow-400.woff2").status == 200
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + "/../launcher.py")
        state = json.loads(urllib.request.urlopen(base + "/api/state").read())
        assert state["mode"] == "checklist" and not state["ready"] and state["can_control"]
        # a page from somewhere else cannot press the buttons: it cannot send this header
        bare = urllib.request.Request(base + "/api/do", json.dumps({"action": "start_anyway"}).encode())
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(bare)
        assert e.value.code == 403 and keeper.mode == "checklist"
        port = server.server_address[1]
        assert launcher.ask(port, "start")["ok"] is False
        assert launcher.ask(port, "start_anyway")["ok"] and launcher.ask(port)["mode"] == "running"
        assert launcher.ask(port, "no_such_button")["ok"] is False
    finally:
        server.shutdown()
        server.server_close()


def test_settings_come_from_the_environment_then_the_file_then_the_defaults(tmp_path):
    (tmp_path / ".env.launcher").write_text("# rehearsal\nGUIDANCE_AUDIO=off\nSTREAM_SOURCE='recordings/live_2025'\nGUIDANCE_MODE=2\n")
    env = launcher.load_env(tmp_path, {"GUIDANCE_MODE": "5"})
    assert env["GUIDANCE_MODE"] == "5" and env["GUIDANCE_AUDIO"] == "off" and env["STREAM_SOURCE"] == "recordings/live_2025"
    assert env["VOICE_MODE"] == "web" and env["HOST"] == "127.0.0.1"
    assert launcher.load_env(tmp_path / "nowhere", {})["GUIDANCE_MODE"] == "3"


def test_live_sh_and_the_launcher_agree_on_the_settings_that_worked():
    script = (ROOT / "tools" / "live.sh").read_text()
    for key, value in launcher.DEFAULTS.items():
        assert re.search(rf'{key}(="\${{{key}:-|:=){re.escape(value)}}}', script), f"tools/live.sh does not default {key} to {value}"


def test_tail_gives_the_last_lines_without_colour_codes(tmp_path):
    log = tmp_path / "pipeline.log"
    log.write_text("".join(f"line {i}\n" for i in range(500)) + "\x1b[1;32mgreen\x1b[0m\n\n")
    got = launcher.tail(log, 200)
    assert len(got) == 200 and got[-1] == "green" and got[0] == "line 301"
    assert launcher.tail(tmp_path / "missing.log") == []


def test_recording_goes_through_the_phone_bridge_when_there_is_one(keeper):
    with_bridge(keeper.root)
    switch = keeper.root / "out" / "status" / "phone_record"
    assert "STREAM_RECORD" not in keeper.pipeline_env()           # the pipeline does not record a second copy
    keeper.tell_bridge_to_record()
    assert switch.read_text().strip() == "off"                    # nothing is saved while the checklist is up
    keeper.mode = "running"
    keeper.tell_bridge_to_record()
    assert switch.read_text().strip() == "on"
    write_phone(keeper.root, state="streaming", ok=True, recording=True, record_frames=120,
                record_folder=str(keeper.root / "recordings" / "demo_20261011_090000"))
    line = keeper.record_line()
    assert line["level"] == GREEN and "recordings/demo_20261011_090000" in line["message"] and "120 frames" in line["message"]
    write_phone(keeper.root, state="streaming", ok=True, recording=False, record_error="less than 5 GB free on the disk")
    assert keeper.record_line()["level"] == RED
    keeper.toggle_record()
    keeper.tell_bridge_to_record()
    assert switch.read_text().strip() == "off"
    keeper.mode = "checklist"


def test_a_phone_bridge_started_by_hand_is_left_alone(keeper):
    import fcntl
    with_bridge(keeper.root)
    held = open(keeper.root / "out" / "phone_bridge.lock", "w")
    fcntl.flock(held, fcntl.LOCK_EX)
    assert checks.bridge_running(keeper.s)
    keeper.keep_bridge(time.time())
    assert keeper.bridge.pid is None                              # no second bridge to fight it for the phone
    held.close()
    assert not checks.bridge_running(keeper.s)
    keeper.keep_bridge(time.time())
    assert keeper.bridge.pid is not None and keeper.bridge.argv[-2:] == ["tools/phone_bridge.py", "run"]
    keeper.bridge.stop(signal.SIGKILL, 0.2)


def test_a_bridge_that_keeps_stopping_is_named_on_the_page_and_started_less_often(keeper):
    with_bridge(keeper.root)                                      # an empty program: it ends at once, every time
    now, starts = time.time(), 0
    for second in range(0, 60, 3):
        before = keeper.bridge.pid
        keeper.keep_bridge(now + second)
        starts += keeper.bridge.pid != before
        time.sleep(0.1)
    assert starts == 9                                            # every 3 s six times, then every 15 s
    line = by_id(keeper.look("phone"))["phone_stream"]
    assert line["level"] == RED and "stopped 9 times in a row" in line["message"] and "phone_bridge.log" in line["message"]


def test_the_bridge_says_whether_the_phone_trusts_us(tmp_path):
    with_bridge(tmp_path)
    m = Machine(**{"idevice_id": (0, "00008130-001A2B3C4D5E6F78")})
    write_phone(tmp_path, state="untrusted", ok=False, message="Unlock the phone and tap Trust.")
    trust = by_id(checks.phone(setup(tmp_path, m)))["phone_trust"]
    assert trust["level"] == RED and trust["message"] == "Unlock the phone and tap Trust."
    write_phone(tmp_path, state="app_not_streaming", ok=False, message="Open PathOn on the phone and tap Start.")
    got = by_id(checks.phone(setup(tmp_path, m)))
    assert got["phone_trust"]["level"] == GREEN and got["phone_stream"]["level"] == RED
    assert not any(a.startswith("idevicepair") for a in m.asked)


def test_new_request_presses_t_on_the_running_pipeline_and_stops_nothing(keeper):
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    got = []

    class Dash(BaseHTTPRequestHandler):
        def do_POST(self):
            got.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *a):
            pass

    assert keeper.do("new_request") == (False, "Nothing is running: press Start.")
    keeper.start(anyway=True)
    assert tick_until(keeper, lambda: keeper.heartbeat() is not None)
    pid = keeper.pipeline.pid
    ok, message = keeper.do("new_request")              # the pipeline's own page is not up: said, not hidden
    assert not ok and "did not answer" in message
    server = HTTPServer(("127.0.0.1", keeper.s.dash_port), Dash)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        assert keeper.do("new_request") == (True, "Asking what to grab.")
    finally:
        server.shutdown()
        server.server_close()
    assert got == [("/key", {"key": "t"})]
    assert keeper.mode == "running" and keeper.pipeline.pid == pid and keeper.pipeline.alive()


def test_both_pages_have_the_new_request_button():
    root = Path(launcher.__file__).parent
    assert 'id="again"' in (root / "launcher" / "index.html").read_text()
    assert 'act("new_request")' in (root / "launcher" / "app.js").read_text()
    assert 'id="new-btn"' in (root / "dashboard" / "index.html").read_text()
    assert '$("new-btn").addEventListener("click", () => sendKey("t"))' in (root / "dashboard" / "app.js").read_text()
