# UCSC AI Hackathon 2026

Our team's project for the UCSC AI Hackathon, October 10–11, 2026, at UC Santa Cruz Silicon Valley (Santa Clara).

## Team

- Veer Nanda (@DrealVeerNanda)
- _add yourself_

## Project: Blindfold pickup (ASUS Local AI track)

A person wears an iPhone on their chest, says "find my mug," and picks it up blindfolded.
Audio tones steer their hand to the object. Every model runs on the ASUS Ascent GX10.

| File | Lane | What |
|---|---|---|
| `interfaces.py` | all | `Frame` and `Scene`, the two agreed interfaces. Import them, don't redefine them. |
| `stream.py` | 1 | `get_frame() -> Frame \| None` |
| `perception.py` | 2 | `locate(frame, target) -> Scene` |
| `guidance.py` | 3 | `update(scene) -> str` (state), optional `reset(target)` |
| `voice.py` | 3 | `listen_for_target() -> str` |
| `main.py` | 4 | the loop; every module can be fake or real |
| `fakes.py` | 4 | fake versions of all of the above |
| `debug_view.py` | 4 | the on-screen overlay judges watch |
| `trials.py` | 4 | timed baseline / guided trials into `trials.csv` |

### Run it

```bash
pip install -r requirements.txt
python main.py                                   # everything fake
python main.py --guidance real                   # lane 3: tune audio against fake scenes
python main.py --stream real --perception real   # lanes 1 + 2 live
python main.py --all-real --fullscreen           # the demo
```

Fake perception keys (debug window focused): arrows and `w`/`s` move the hand 1 cm,
`o` occludes the object, `h` hides the hand, `n` adds jitter, `r` resets. `t` asks for a
new target, `q` quits. Each run writes a per-frame CSV to `logs/` and prints p50/p95
latency per stage on exit.

### Interface notes

- `guidance.update(scene)` returns its state as a string (`idle`, `searching`,
  `waiting_hand`, `guiding`, `grab`). The debug view shows it, and when it goes back to
  `idle` main calls `listen_for_target()` again. If `guidance.reset(target)` exists, main
  calls it after every new target.
- `Frame.t_recv` is `time.monotonic()` on the GX10 when the frame arrived. Without it the
  latency number only covers processing, not the trip from the phone.

## Getting started

```bash
git clone https://github.com/DrealVeerNanda/ucsc-ai-hackathon-2026.git
cd ucsc-ai-hackathon-2026
```

## Stream (lane 1)

`stream.py` gives everyone the iPhone's colour and lidar depth:

```python
import stream
frame = stream.get_frame()   # newest Frame, or None if nothing new since the last call
```

`Frame` has `rgb`, `depth`, `K`, `pose`, `t`, as agreed. What you can rely on:

- `rgb` and `depth` are upright and the same size. `K` matches that size.
- `depth` is in meters. `0` means no lidar reading at that pixel.
- `pose` is camera-to-world, or `None`. Do not build on it until lane 1 says it passed `tools/check_pose.py`.
- `t` is `time.time()` when the frame arrived, so `time.time() - frame.t` is the frame age.
- `get_frame()` never blocks. `get_frame(wait=0.1)` waits up to 0.1 s for a new frame, so a loop does not spin.
- `stream.status()` gives `state`, `fps` and `age_s` for the debug view. After a cable bump the stream reconnects by itself.

### Work without the phone

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
STREAM_SOURCE=recordings/NAME python main.py              # replays a recording, looped, at real speed
python tools/probe_stream.py recordings/NAME --show       # look at it
```

### Pick the source

Set `STREAM_SOURCE`, or call `stream.start(SOURCE)` once before the loop.

| `STREAM_SOURCE` | Frames come from |
|---|---|
| `record3d` (default) | Record3D app over USB |
| `pathon` | Free "Record3DStream - PathOn" app over USB. See `requirements.txt`. |
| `relay` | A Mac running `tools/relay_send.py` |
| `recordings/NAME` | A saved recording |

`STREAM_ROTATE=0|90|180|270` turns the image clockwise if it is not upright.

### Phone on the GX10

```bash
sudo apt install usbmuxd libusbmuxd-dev cmake build-essential
git clone https://github.com/marek-simonik/record3d && cd record3d && pip install .
```

In Record3D: turn on USB Streaming mode in Settings, select the rear lidar camera, press the red button. Plug the phone into a data port, unlock it, tap Trust. Then:

```bash
python tools/probe_stream.py            # frame rate, sizes, K, depth range, pose yes or no
python tools/probe_stream.py --show     # live rgb and depth. r rotates, s saves, q quits
python tools/check_pose.py              # move the phone for 10 s. Says if lane 2 can use the pose
python tools/record.py object_on_table  # 30 s into recordings/object_on_table
```

### If record3d will not build on the GX10

Read the phone on a Mac and forward frames over Ethernet. Models still run on the GX10.

```bash
STREAM_SOURCE=relay python main.py              # on the GX10
python tools/relay_send.py GX10_IP              # on the Mac
```

Tests: `python -m pytest -q`

## Workflow

- Pull before you start: `git pull --rebase`
- Small commits, push often
- Don't commit secrets. Put API keys in `.env` (it's gitignored)
