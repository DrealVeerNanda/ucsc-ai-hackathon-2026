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

## Workflow

- Pull before you start: `git pull --rebase`
- Small commits, push often
- Don't commit secrets. Put API keys in `.env` (it's gitignored)
