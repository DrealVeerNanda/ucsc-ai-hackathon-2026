# UCSC AI Hackathon 2026

Our team's project for the UCSC AI Hackathon, October 10–11, 2026, at UC Santa Cruz Silicon Valley (Santa Clara).

## Team

- Veer Nanda (@DrealVeerNanda)
- Gavin Wang (gavinyhwang@gmail.com)
- Jake Pong (jzcpong@gmail.com)
- _add yourself_

## Project: Proprio (ASUS Local AI track)

A person wears an iPhone on their chest, says "find my mug," and picks it up blindfolded.
Spoken directions and tones in their earbuds steer the hand to the object. Every model runs on the
ASUS Ascent GX10; nothing goes to the cloud.

| File | Lane | What |
|---|---|---|
| `interfaces.py` | all | `Frame` and `Scene`, the two agreed interfaces. Import them, don't redefine them. |
| `stream.py` | 1 | `get_frame() -> Frame \| None` |
| `perception.py`, `hands.py` | 2 | `locate(frame, target) -> Scene`; the hand trackers to choose from |
| `cosmos.py` | 2 | the slow helper: lists what is on the table, gives a box when the detector fails |
| `guidance.py`, `audio_out.py` | 3 | `update(scene) -> str` (state), optional `reset(target)`; the sound |
| `voice.py` | 3 | `listen_for_target() -> str` |
| `main.py` | 4 | the loop; every module can be fake or real |
| `launcher.py`, `launcher/` | 4 | the one-button start: checklist page, and a keeper that restarts what dies |
| `tools/phone_bridge.py` | 1 | holds the phone's connection so restarting the pipeline never drops it |
| `tracing.py` | all | per-frame timings and events into `logs/trace_*.jsonl` |
| `fakes.py` | 4 | fake versions of all of the above |
| `dashboard.py`, `dashboard/` | 4 | the judge dashboard in the browser, served by `main.py` (offline, no CDN) |
| `debug_view.py` | 4 | OpenCV debug window (`--ui cv`) |
| `trials.py` | 4 | timed baseline / guided trials into `trials.csv` |

### Start the demo (on the GX10)

Double-click **Proprio** on the GX10's desktop (`bash tools/install_launcher.sh` puts it there). A page
opens at http://localhost:8640 with one line per thing that has to be true (phone on the cable and
streaming, AirPods connected, model servers up, models on disk), green or red, each saying what to do.
When it is green press **Start**; **Start anyway** is there for when you decide to go regardless.

From then on a keeper keeps it running: it restarts the pipeline if it crashes or hangs (a reach in
progress carries on), holds the phone connection in its own process so PathOn does not need another tap,
reconnects the AirPods, and records every session into `recordings/demo_*/`. **Stop** on the page ends
the pipeline; `python launcher.py quit` ends everything. Settings are in `launcher.DEFAULTS`, and
`.env.launcher` (KEY=VALUE lines, not in git) overrides them, e.g. `HOST=0.0.0.0` to watch the
dashboard from a laptop: by default it answers on the GX10 only.

The **Options to try** card at the top of that page switches, with one click, between the ways of
guiding (`guidance.OPTIONS`, 12 of them), of tracking the hand (`perception.HAND_OPTIONS`, 7), of
deciding "Grab" (`guidance.GRAB_OPTIONS`, 5) and of keeping track of the object
(`perception.OBJECT_OPTIONS`, 3). Before Start a click sets what the run starts with;
while it runs it switches at once and the wearer hears the name. The choice is remembered (also
across a restart of the pipeline); **Back to the usual ones** undoes it. The same switches work by
voice ("option 3", "hand option 5", "grab option 2", "object option 3") and from the live dashboard's **Options**
button. Hear every option without the rig: `python tools/audio_demo.py --all`, then open
`out/options/index.html`.

If the new parts and the real hardware do not get on, `.env.launcher` has a way back for each:
`PROPRIO_BRIDGE=0` (the pipeline reads the phone itself), `VOICE_HEADSET_MODE=stay` (the earbuds
never switch mode: mono sound), `VOICE_WEARER_LEVEL=0` and `VOICE_REQUEST_S=12` (hear everything).
The code of the first live run that worked is the tag `worked-live-2026-10-10`.

### Run it by hand

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

`bash tools/live.sh` is the real pipeline with the settings of the run that worked (guidance
option 3, the dashboard in the browser, earbuds as the microphone). It and the launcher bind the
dashboard to the GX10 only; `HOST=0.0.0.0 bash tools/live.sh` opens it to the network.
`bash tools/stop_live.sh` stops a run and waits until its recording is saved.

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
  calls it after every new target. It goes `idle` after "Grab", and also when it gives up:
  20 s without ever seeing the object, 30 s with no hand shown, or 2 minutes without a "Grab".
- `Scene` has two optional extras from lane 2: `hand` (keypoints, palm, confidence, which
  tracker) and `object_out_of_reach` (farther than 0.75 m: guidance says "too far, move closer").
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
| `auto` (default) | A running phone bridge if there is one; otherwise whichever phone app is streaming: PathOn over USB, PathOn over Wi-Fi if `STREAM_PHONE_IP` is set, then Record3D |
| `bridge` | `tools/phone_bridge.py`, which holds the phone and hands frames on through shared memory. The launcher uses this. |
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

In PathOn (the app we use): set its WiFi | USB switch to **USB**, then tap Start Streaming. Unlock the
phone before plugging it into a data port, and tap Trust. On the phone: Auto-Lock Never, Low Power Mode
off. (In Record3D instead: turn on USB Streaming mode in Settings, select the rear lidar camera, press
the red button.)

PathOn stops streaming when the program reading it goes away, and someone then has to tap Start again.
`python tools/phone_bridge.py start` holds the connection in a process of its own so the pipeline can be
restarted freely (`status`, `stop`, `record on|off`; `probe` measures, with the real phone, how long a gap
the app forgives). `python tools/fake_phone.py recordings/NAME --port 9888` stands in for the app when
there is no phone. Then:

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

Without the phone or the earbuds the whole pipeline can still be exercised on the GX10:

```bash
python tools/fault_drill.py --quick     # 10 min: break one thing at a time (camera, model servers, earbuds) and see it recover
python tools/fault_drill.py             # all 46 drills, about 37 min
python tools/soak.py run --minutes 100  # run it like a demo from recordings and report what grows, drifts or stops
```

## Perception (lane 2)

`perception.py` is a working first version, ported from an earlier project so lane 2 does not start from zero. Change anything.

It finds the object's box (YOLO-World, any name you type) and the index fingertip (MediaPipe Hands by default), reads lidar depth at both, and returns their positions in meters. When the hand covers the object it keeps the object's remembered depth; a clear bottle, which the lidar sees through, takes its depth from the nearest surface in its box; of two matching objects the one within reach wins.

Seven hand trackers can be switched while it runs (`HAND_OPTION=1..7`, `perception.HAND_OPTIONS`): 1 is MediaPipe as the demo was built on; 3 and 4 are a YOLO hand-keypoint model on the GPU; 5 adds a depth-and-skin follower that carries on when the model loses the hand (found on 94% of frames of a recorded reach against 81% for option 1); 7 guides the palm, not the fingertip. Compare them on a recording with `python tools/eval_hands.py`.

The object can be kept track of in three ways (`OBJECT_OPTION=1..3`, `perception.OBJECT_OPTIONS`), switched while it runs:

1. **watched** (default, as it always was): the detector finds it in every frame; behind the hand it is remembered for 2 to 4 s where it was in the picture.
2. **remembered in the room**: the detector still looks in every frame, but the place is kept in room coordinates from the phone's own motion tracking (ARKit's pose), so a hidden object stays put for 15 s, also when the wearer leans or turns.
3. **found once**: after ten clear looks that agree to 3 cm (about a third of a second live), the place is fixed and the detector is not run again for that reach. Only the hand and the camera's motion are followed, so an arm in front of the object cannot move it or lend it its depth. The place is let go when the object is asked for again, the target changes, or the phone's tracking starts afresh. If the object itself is moved during a reach, it is not noticed: ask for it again, or click the option again on the Options card, and it is looked for afresh.

Options 2 and 3 depend on the pose. On the reaches recorded on 11 October it carried lidar points from one frame to one a second later to within 0.4 cm (`python tools/check_pose.py recordings/NAME --seconds 100`), against 2.5 cm with the motion ignored. Played back over those reaches, "found once" kept the box on the bottle while the camera moved, read 0.68 m where "watched" read the arm in front of it at 0.48 m, and took 17 ms a frame against 28.

```bash
python tools/check_perception.py recordings/NAME mug    # no window: prints fps, hit rates, distance. Saves out/perception.jpg
python main.py --stream real --perception real          # with the debug view
```

| Setting | Values |
|---|---|
| `PERCEPTION_DETECTOR` | `yoloworld` (default), `yoloe`, `coco` (fixed classes: cup, bottle, banana...), `none` |
| `PERCEPTION_HAND` | `mediapipe` (default), `sticker` (a green sticker on the hand), `none` |
| `HAND_OPTION` | `1` (default) to `7`, or a key from `perception.HAND_OPTIONS` |
| `OBJECT_OPTION` | `1` (default) to `3`, or a key from `perception.OBJECT_OPTIONS` |
| `PERCEPTION_USE_POSE=1` | The same as `OBJECT_OPTION=2`, from before there were object options |
| `PERCEPTION_DETECT_EVERY=2` | Run the detector on every second frame if the frame rate is short |

On the GX10 it takes about 20 ms per frame. Not yet checked: real lidar depth against a ruler (needs the phone), and the tuning in `PerceptionConfig`.

## Guidance and voice (lane 3)

`guidance.py`, `audio_out.py` and `voice.py` are also a working first version. Change anything.

There are twelve ways of guiding (`GUIDANCE_MODE=1..12`, `guidance.OPTIONS`), switchable while it runs.
The demo uses option 3, "combined words": the whole direction at once ("up and left"), a chime when
lined up, "forward", then "Grab". Options 6 to 12 are newer ideas (amounts in centimetres, two notes to
match, a rising note, fast repeated words, quiet-unless-drifting, words then a note, a hum that turns
to beeps) that nobody has followed with a real hand yet.

"Grab" is said by one of five rules (`GUIDANCE_GRAB=1..5`, `guidance.GRAB_OPTIONS`). The default,
"over the object", fires when the fingertip is over the object in the picture and within grabbing
distance in depth; rule 1 is the first one, the fingertip within 5 cm.

Option 1, tones only:

| You hear | It means |
|---|---|
| Tone in the right or left ear | Move the hand that way |
| Pitch higher or lower | Move up or down |
| Faster beeps | Closer |
| Steady tone, then "grab" | Within grabbing distance (the rule is switchable: `guidance.GRAB_OPTIONS`) |

```bash
python tools/audio_demo.py --play            # hear a made-up reach. Without --play it writes out/audio_demo.wav
python main.py --guidance real               # tune by ear: move the fake hand with the arrow keys
GUIDANCE_WORDS=1 python main.py --guidance real   # fallback: spoken left, right, up, down, forward
GUIDANCE_MODE=hybrid python main.py --guidance real # direction once, progress beeps, chime, next step
GUIDANCE_MODE=tones python main.py --guidance real  # original spatial beeps, no spoken directions
```

- Distances and timings to tune are in `GuidanceConfig` in `guidance.py`; the sound itself is `ToneConfig` in `audio_out.py`.
- Spoken mode follows one axis until aligned, pauses at least 1 second before a new direction,
  and repeats an unchanged direction only every 2.5 seconds. It stays silent near the target until "grab".
  Tune `word_gap_s` and `word_repeat_s` in `GuidanceConfig` after testing with the real hand tracker.
- Slower, lower Daniel clips in `audio/calm/` override the original clips when present. Generate them
  on a Mac with `python tools/make_calm_audio.py`; copy the WAVs to the GX10 for offline playback.
  Use `--voice Samantha --rate 140` to regenerate the earlier voice instead.
- The complete Brian voice set in `audio/elevenlabs/` takes precedence over both older sets.
  These are local 48 kHz mono 16-bit WAVs: no ElevenLabs account, API key, or network is needed
  for playback. Missing or unreadable clips fall back to `audio/calm/`, then `audio/`.
  To reconvert the downloaded MP3s in `audio/eleven_labs_raw/`, run
  `python tools/convert_elevenlabs.py` (needs PyAV, installed with faster-whisper).
  It trims excess silence and matches speech levels, and writes a combined preview to
  `out/elevenlabs_preview_all.wav`. Arbitrary sentences still use local Piper/macOS speech.
- Hybrid mode guides one hand movement at a time: sideways, then up/down, then forward/back.
  It says the direction once, then uses fixed-pitch beeps that speed up as that axis gets aligned.
  Beeps stop and a chime plays once after staying within 2.5 cm on that axis for 0.2 seconds.
  After a pause it announces the next direction, skipping axes already aligned.
  If the hand overshoots, it announces the corrective direction once. Completed axes are checked
  again before choosing the next movement. Tune `step_m`, `step_hold_s`, and `step_pause_s`.
  `GUIDANCE_MODE` overrides the older `GUIDANCE_WORDS` flag.
  Preview without a camera: `python tools/audio_demo.py --hybrid --play`.
- At the target, after 0.3 seconds near it, a soft two-note chime plays once followed by "grab".
  Hybrid and words modes are quiet during that short confirmation wait; tones mode uses a steady tone.
  This means the fingertip is near the target; it doesn't detect an actual pickup.
  The next request waits for the spoken audio to finish.
- The clips in `audio/` remain the original macOS voice fallback.
- If the loop stalls for half a second the tone goes quiet, so an old position never keeps steering.
- With Bluetooth earbuds paired to the GX10, `voice.py` uses them as the microphone and listens by itself:
  it says what it can see, asks "What do you want to grab?", beeps, and takes the answer (Whisper large-v3
  on the GPU, about 0.2 s). The earbuds are switched to headset mode to listen and back to stereo to guide,
  so the wearer cannot be heard during a reach. Besides an object's name it understands "what's on the
  table", "the second one", "yes", "stop", "again", "help", "option N", "hand option N", "grab option N", "object option N".
  With a wired microphone it is push-to-talk (Enter, speak, Enter); `VOICE_MODE=type` forces typing.
- When it sees nothing to pick up it says why (turn towards the table, or something is in front of the
  camera), looks again every few seconds, and reminds the wearer that it is listening.
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
| "I want to grab a can", "I'm thirsty" | Nemotron names the object (`can`, `water bottle`), about 0.4 s. A name it has just listed is taken as said, without asking Nemotron. |
| "What's on the table?" | Cosmos looks through the camera and lists what could be picked up; it is spoken, and the wearer answers "the mug" or "the second one" |
| A name the detector cannot find | After one second of searching, Cosmos is asked for the object's box in the background (0.6 to 2 s) and its answer positions the object until the detector takes over |

### Talking to it

The earbuds are the microphone (see above). The backup is a page any phone or laptop can open: a run
without a target (`bash tools/live.sh`, or the launcher) serves `https://GX10_IP:8644`. Open it on a
device on the same network, accept the certificate warning once, then hold the button (or the space
bar) and speak, or type in the box; it also has Stop, Again and option buttons. On the GX10 itself the
launcher's running page links to it as **Type a request**. A new request at any time replaces the
target, even in the middle of a reach.

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

`main.py` also serves it at http://localhost:8642/deck/, and then the live-demo slide embeds the
dashboard. From another machine, add `?dash=http://<gx10-ip>:8642`. With no dashboard reachable,
the demo slide plays `deck/media/demo-run.mp4` if it exists.

Online copy: https://drealveernanda.github.io/ucsc-ai-hackathon-2026/ (`deck/`, on the `gh-pages`
branch), and this README at .../readme/ without the email addresses. After changing the deck or
this README, run `bash deck/deploy.sh` to update them.

Keys: arrows, space or click to advance; F fullscreen; N speaker notes with a timer.
`?auto` plays it on a timer (for recording a video); `?still` skips animation (for screenshots).
