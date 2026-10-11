# Benchmark: perception speed

`perception.locate()` is now 1.4–2.8× faster on the GX10, and its slow frames are gone. Boxes and
fingertips are identical to before (fp16 moves boxes by up to 1 px).

| Frames | Before: p50 / p95 | After: p50 / p95 |
|---|---|---|
| Live phone frames, no hand in view | 27–31 / 32–41 ms | **9.8–9.9 / 10.5–10.8 ms** |
| Photo with two hands | 22.3 / 23.5–23.8 ms | **15.6 / 16.3–16.5 ms** |

Three alternating before/after runs of 300 frames each, on 2026-10-10. Another demo, Cosmos and
Nemotron were running on the GX10 at the same time. "Before" is the same code with the three
switches below turned off, which is exactly the old behaviour.

Before these changes, live traces (about 38,000 frames from today's runs) showed perception at
roughly 25 ms of a 34 ms loop. The loop never waited for the camera, so perception set the frame rate.

## What changed

| Change | Effect | Switch to undo |
|---|---|---|
| Hand tracker runs alongside the detector | They share nothing: MediaPipe runs on the CPU, YOLO-World on the GPU. Each frame now costs about the slower of the two, not their sum: 6–7 ms saved | `PERCEPTION_PARALLEL=0` |
| Detector in half precision on the GPU | Detector 6.3 → 5.4 ms. Whole frames aren't faster, because the detector now runs inside the hand tracker's time; this leaves more of the GPU to Cosmos and Nemotron | `PERCEPTION_FP16=0` |
| Hand tracker kept on the fast CPU cores | The GX10 has 10 fast cores (Cortex-X925, 3.9 GHz) and 10 slow ones (Cortex-A725, 2.8 GHz). MediaPipe takes 2.5× as long on the slow ones, and Linux moved it between them, so whole stretches of frames ran at 24–41 ms. Only the hand tracker's threads are pinned | `PERCEPTION_FAST_CORES=0` |

MediaPipe on each kind of core, measured with `taskset`:

| Cores | No hand: p50 / p95 | Two hands: p50 / p95 |
|---|---|---|
| Fast (5–9, 15–19) | 9.5 / 10.2 ms | 14.7 / 15.6 ms |
| Slow (0–4, 10–14) | 24.0 / 24.5 ms | 41.1 / 41.9 ms |

## Tried and not kept

- **Smaller image for MediaPipe** (640, 480 or 360 px instead of full size): no faster, since its
  time is in its models, not in handling the image. The fingertip moved by up to 1.6 px.
- **Tracking at most one hand:** two hands in view took 14.4 → 9.1 ms. Not the default, because
  perception chooses among all the hands in reach, and with only one tracked it could lock onto
  a bystander's hand instead of the wearer's.

## What is left

- The hand tracker is now the limit: about 10 ms a frame with no hand in view and 15 ms with two.
- The rest of the loop, outside perception, takes about 9 ms (dashboard JPEG, depth picture and
  3D cloud). That is the next thing to look at for frame rate.
- When Cosmos is answering a request, it shares the GPU and the detector can stall (up to 1.3 s
  in the live traces). Half precision eases this but does not remove it.

## Reproduce

```bash
source .venv/bin/activate
# images: folders of saved frames and/or single photos
python tools/bench_perception.py FRAMES_DIR HAND_PHOTO.jpg --frames 300 --target person --save before.json
PERCEPTION_PARALLEL=0 PERCEPTION_FP16=0 PERCEPTION_FAST_CORES=0 python tools/bench_perception.py FRAMES_DIR --frames 300   # the old behaviour
python tools/bench_perception.py FRAMES_DIR HAND_PHOTO.jpg --frames 300 --target person --compare before.json  # same results?
```

The test frames are not in the repo. The live frames show people at the venue, and the hand
photo is MediaPipe's sample image (`storage.googleapis.com/mediapipe-tasks/hand_landmarker/woman_hands.jpg`).
