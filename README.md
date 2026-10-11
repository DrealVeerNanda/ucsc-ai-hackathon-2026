# UCSC AI Hackathon 2026

Our team's project for the UCSC AI Hackathon, October 10–11, 2026, at UC Santa Cruz Silicon Valley (Santa Clara).

## Team

- Veer Nanda (@DrealVeerNanda)
- Gavin Wang (gavinyhwang@gmail.com)
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
| `dashboard.py`, `dashboard/` | 4 | the judge dashboard in the browser, served by `main.py` (offline, no CDN) |
| `debug_view.py` | 4 | OpenCV debug window (`--ui cv`) |
| `trials.py` | 4 | timed baseline / guided trials into `trials.csv` |

### Run it

```bash
pip install -r requirements.txt
python main.py                                   # everything fake
python main.py --guidance real                   # lane 3: tune audio against fake scenes
python main.py --stream real --perception real   # lanes 1 + 2 live
python main.py --all-real                        # the demo
```

`main.py` serves the judge dashboard at http://localhost:8642 and opens it. From another
machine on the same network use `http://<gx10-ip>:8642`. `--ui cv` shows the OpenCV window
instead, `--ui both` shows both, `--no-open` skips opening the browser.

What modules can report to the dashboard (all optional, picked up if present):

- `guidance.last_audio`: `{"tone": bool, "pan": -1..1, "pitch": -1..1, "beep_hz": float, "steady": bool}`.
  Without it the dashboard shows the plan's mapping and says it is estimated.
- `guidance.pop_phrases()`: list of phrases spoken since the last call, for the captions.
- `voice.last_transcript`: what the wearer said, shown as a quote.
- `perception.MODEL_NAMES`: `{"object": "YOLO-World", "hand": "MediaPipe"}`, shown in the pipeline steps.

Fake perception keys (dashboard or debug window focused): arrows and `w`/`s` move the hand 1 cm,
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
- `t_recv` is `time.monotonic()` when the frame arrived, so `time.monotonic() - frame.t_recv` is the frame age. Record3D sends no capture timestamp, so `t` is the same value: fine for durations, not a phone clock.
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
| `auto` (default) | Whichever phone app is streaming over USB: PathOn first, then Record3D |
| `pathon` | Free "Record3DStream - PathOn" app over USB. This is what we use. See `requirements.txt`. |
| `record3d` | Paid Record3D app over USB |
| `relay` | A Mac running `tools/relay_send.py` |
| `recordings/NAME` | A saved recording |

`STREAM_ROTATE=0|90|180|270` turns the image clockwise if it is not upright.

### Phone on the GX10

This is already set up on our GX10 in `~/ucsc-ai-hackathon-2026` (`source .venv/bin/activate`). To redo it:

```bash
# The GX10 has no Python headers. Either `sudo apt install python3-dev`, or make the venv
# from a uv Python, which brings its own and needs no sudo:
pip install uv && uv venv --seed --managed-python --python 3.12 .venv && source .venv/bin/activate
pip install -r requirements.txt
# record3d has no ARM wheel and its source misses an include that GCC 13 insists on:
CXXFLAGS="-include string -include cstdint" pip install record3d
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

## Perception (lane 2)

`perception.py` is a working first version, ported from an earlier project so lane 2 does not start from zero. Change anything.

It finds the object's box (YOLO-World, any name you type) and the index fingertip (MediaPipe Hands), reads lidar depth at both, and returns their positions in meters. When the hand covers the object it keeps the object's remembered depth.

```bash
python tools/check_perception.py recordings/NAME mug    # no window: prints fps, hit rates, distance. Saves out/perception.jpg
python main.py --stream real --perception real          # with the debug view
```

| Setting | Values |
|---|---|
| `PERCEPTION_DETECTOR` | `yoloworld` (default), `yoloe`, `coco` (fixed classes: cup, bottle, banana...), `none` |
| `PERCEPTION_HAND` | `mediapipe` (default), `sticker` (a green sticker on the hand), `none` |
| `PERCEPTION_USE_POSE=1` | Remember the object in room coordinates. Only after `tools/check_pose.py` says yes. |
| `PERCEPTION_DETECT_EVERY=2` | Run the detector on every second frame if the frame rate is short |

On the GX10 it runs at about 35 ms per frame. Not yet checked: real lidar depth against a ruler (needs the phone), and the tuning in `PerceptionConfig`.

## Guidance and voice (lane 3)

`guidance.py`, `audio_out.py` and `voice.py` are also a working first version. Change anything.

| You hear | It means |
|---|---|
| Tone in the right or left ear | Move the hand that way |
| Pitch higher or lower | Move up or down |
| Faster beeps | Closer |
| Steady tone, then "grab" | Within 5 cm |

```bash
python tools/audio_demo.py --play            # hear a made-up reach. Without --play it writes out/audio_demo.wav
python main.py --guidance real               # tune by ear: move the fake hand with the arrow keys
GUIDANCE_WORDS=1 python main.py --guidance real   # fallback: spoken left, right, up, down, forward
```

- Distances and timings to tune are in `GuidanceConfig` in `guidance.py`; the sound itself is `ToneConfig` in `audio_out.py`.
- The clips in `audio/` are placeholders made with the macOS voice. Replace them with ElevenLabs clips of the same file names.
- If the loop stalls for half a second the tone goes quiet, so an old position never keeps steering.
- `voice.py` asks for a typed target unless a microphone is present. Then it is push-to-talk: Enter, speak, Enter. `VOICE_MODE=type` forces typing.
- The GX10 has no speakers or microphone of its own, only HDMI. Pair the earbuds, then set `GUIDANCE_DEVICE` and `VOICE_DEVICE` if they are not the default.

## The language models (helpers, not in the loop)

Two models run next to the pipeline and are only asked now and then. Both stay on the GX10.

```bash
bash tools/run_nemotron.sh     # port 8000: turns what the wearer says into an object name
bash tools/run_cosmos.sh       # port 8001: looks at one picture (8b by default, `2b` is faster)
bash tools/fetch_models.sh piper   # once: the voice that speaks sentences
```

| The wearer says | What happens |
|---|---|
| "I want to grab a can", "I'm thirsty" | Nemotron names the object (`can`, `water bottle`), about 0.3 s |
| "What's on the table?" | Cosmos looks through the camera and lists what could be picked up; it is spoken, and the wearer answers "the mug" or "the second one" |
| A name the detector cannot find | After one second of searching, Cosmos is asked for the object's box in the background (0.6 to 2 s) and its answer positions the object until the detector takes over |

If a server is not running, the pipeline carries on without it: a simple rule reads plain requests, and the detector works alone. `python tools/ask_cosmos.py photo.jpg --list` tries the listing on any picture.

## Where the time goes

Every run writes a trace to `logs/trace_*.jsonl` by itself: one line per camera frame with the time each step took, plus a line for every slow call (model loads, Cosmos, Whisper, audio underruns).

```bash
python tools/trace_report.py                       # newest run: every step, slowest first, and what limits the frame rate
python tools/trace_report.py --skip 30 --csv out/trace.csv   # drop warm-up frames; one row per frame for a spreadsheet
python tools/measure_camera_latency.py pathon      # at the GX10's own screen: real camera-to-GX10 delay, by flashing the screen
```

| In the trace | Meaning |
|---|---|
| `cable_ms` | Phone to GX10, extra over the quickest recent frame. Only sources that send a timestamp (PathOn, relay). |
| `wait_ms` | The frame had arrived and sat waiting for the loop |
| `skipped` | Camera frames that arrived and were never used |
| `stream.prepare`, `perception.detector`, `perception.hand`, `perception.depth`, `guidance` | Time inside each step |
| `loop_ms`, `idle_ms` | The whole loop for that frame, then how long it waited for the camera |
| `e2e_ms` | Frame arrived to sound updated |

To time something new: `with tracing.span("my.step"):` inside the loop, or `tracing.event("my.call", ms=...)` for anything slower than a frame. `TRACE=0` switches tracing off.

The trace cannot see two things: the fixed camera and cable delay (the flash tool measures it) and Bluetooth earbud delay.

## Workflow

- Pull before you start: `git pull --rebase`
- Small commits, push often
- Don't commit secrets. Put API keys in `.env` (it's gitignored)

### Pitch deck

Plain static files in `deck/` (three.js and the fonts are inside it), so any web server works:

```bash
python -m http.server 8650   # from the repo root, then open http://localhost:8650/deck/
```

`main.py` also serves it at http://localhost:8642/deck/. Then the live-demo slide embeds the
dashboard and the results slide reads the trial numbers from it. From another machine, add
`?dash=http://<gx10-ip>:8642`. With no dashboard reachable, the demo slide plays
`deck/media/demo-run.mp4` if it exists, and the results slide reads `trials.csv` directly.

Keys: arrows, space or click to advance; F fullscreen; N speaker notes with a timer.
`?auto` plays it on a timer (for recording a video); `?still` skips animation (for screenshots).
