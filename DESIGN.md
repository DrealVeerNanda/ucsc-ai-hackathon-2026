---
name: Blindfold pickup, live
description: Live broadcast graphics over a chest camera, so judges can follow a blindfolded grab they cannot hear.
colors:
  ground: "oklch(0.165 0.022 255)"
  plate: "oklch(0.215 0.026 255)"
  plate-hi: "oklch(0.265 0.03 255)"
  rule: "oklch(0.33 0.03 255)"
  text: "oklch(0.965 0.005 255)"
  muted: "oklch(0.76 0.02 255)"
  faint: "oklch(0.6 0.022 255)"
  scrim: "oklch(0.12 0.02 255 / 0.82)"
  video-well: "oklch(0.1 0.015 255)"
  telestrator-yellow: "oklch(0.88 0.165 96)"
  on-yellow: "oklch(0.21 0.04 96)"
  grab-green: "oklch(0.82 0.16 152)"
  on-green: "oklch(0.2 0.04 152)"
  warn-amber: "oklch(0.8 0.13 65)"
  alert-red: "oklch(0.62 0.19 27)"
  depth-far: "#0c1628"
  depth-near: "#c4e4ff"
typography:
  display:
    fontFamily: "system-ui, -apple-system, \"Segoe UI\", Ubuntu, Cantarell, \"Noto Sans\", sans-serif"
    fontSize: "min(4.3rem, 15cqi)"
    fontWeight: 700
    lineHeight: 1.02
    letterSpacing: "-0.025em"
    fontFeature: "\"tnum\""
  display-number:
    fontFamily: "system-ui, -apple-system, \"Segoe UI\", Ubuntu, Cantarell, \"Noto Sans\", sans-serif"
    fontSize: "min(5.16rem, 17cqi)"
    fontWeight: 700
    lineHeight: 1
    letterSpacing: "-0.03em"
    fontFeature: "\"tnum\""
  headline:
    fontFamily: "system-ui, -apple-system, \"Segoe UI\", Ubuntu, Cantarell, \"Noto Sans\", sans-serif"
    fontSize: "2.074rem"
    fontWeight: 700
    lineHeight: 1.1
    letterSpacing: "-0.01em"
    fontFeature: "\"tnum\""
  title:
    fontFamily: "system-ui, -apple-system, \"Segoe UI\", Ubuntu, Cantarell, \"Noto Sans\", sans-serif"
    fontSize: "1.728rem"
    fontWeight: 600
    lineHeight: 1.25
    fontFeature: "\"tnum\""
  section:
    fontFamily: "system-ui, -apple-system, \"Segoe UI\", Ubuntu, Cantarell, \"Noto Sans\", sans-serif"
    fontSize: "1.2rem"
    fontWeight: 600
    lineHeight: 1.45
  body:
    fontFamily: "system-ui, -apple-system, \"Segoe UI\", Ubuntu, Cantarell, \"Noto Sans\", sans-serif"
    fontSize: "1rem"
    fontWeight: 400
    lineHeight: 1.45
    fontFeature: "\"tnum\""
  label:
    fontFamily: "system-ui, -apple-system, \"Segoe UI\", Ubuntu, Cantarell, \"Noto Sans\", sans-serif"
    fontSize: "0.833rem"
    fontWeight: 400
    lineHeight: 1.45
    fontFeature: "\"tnum\""
rounded:
  plate: "3px"
  dot: "50%"
spacing:
  sp1: "4px"
  sp2: "8px"
  sp3: "12px"
  sp4: "16px"
  sp5: "24px"
  sp6: "32px"
  sp7: "48px"
components:
  tag-object:
    backgroundColor: "{colors.text}"
    textColor: "{colors.ground}"
    typography: "{typography.body}"
    rounded: "{rounded.plate}"
    padding: "2px 8px"
  tag-leg:
    backgroundColor: "{colors.telestrator-yellow}"
    textColor: "{colors.on-yellow}"
    rounded: "{rounded.plate}"
    padding: "2px 12px"
  tag-leg-close:
    backgroundColor: "{colors.grab-green}"
    textColor: "{colors.on-green}"
    rounded: "{rounded.plate}"
    padding: "2px 12px"
  tag-hand-warn:
    backgroundColor: "{colors.warn-amber}"
    textColor: "{colors.on-yellow}"
    rounded: "{rounded.plate}"
    padding: "2px 8px"
  clock-bug:
    backgroundColor: "{colors.scrim}"
    textColor: "{colors.text}"
    typography: "{typography.headline}"
    rounded: "{rounded.plate}"
    padding: "8px 16px 12px"
  caption-now:
    backgroundColor: "{colors.scrim}"
    textColor: "{colors.text}"
    typography: "{typography.title}"
    padding: "12px 16px"
  caption-tag:
    backgroundColor: "{colors.text}"
    textColor: "{colors.ground}"
    typography: "{typography.label}"
    rounded: "{rounded.plate}"
    padding: "2px 8px"
  stale-banner:
    backgroundColor: "{colors.alert-red}"
    textColor: "{colors.text}"
    rounded: "{rounded.plate}"
    padding: "8px 16px"
  notice-plate:
    backgroundColor: "{colors.plate}"
    textColor: "{colors.muted}"
    rounded: "{rounded.plate}"
    padding: "8px 16px"
  offline-bug:
    backgroundColor: "{colors.text}"
    textColor: "{colors.ground}"
    rounded: "{rounded.plate}"
    padding: "4px 16px"
---

# Design System: Blindfold pickup, live

## Overview

**Creative North Star: "The Telestrator Booth"**

The screen is a live sports broadcast of one event: a blindfolded hand reaching an object. Judges stand 1 to 2 m away and cannot hear the earbuds, so every graphic is a broadcast element laid over or beside live video: a clock bug, a lower-third caption, a telestrator line drawn from fingertip to object, a picture-in-picture depth inset. It is an Operate surface (a live monitor), also used by the team while tuning, so density stays honest: real frames, real millimetres, real milliseconds, and empty states where nothing has been measured yet.

Graphics are flat plates on a blue-black ground. The video is always the brightest thing on screen; the surround is dark by decision so it never competes. One colour, telestrator yellow, is reserved for the active leg; green means success and nothing else. Type is a single white sans with tabular figures, set big enough that the state word and the distance read from across the table. Motion is almost absent: values snap like a broadcast data feed, and only new caption phrases wipe in.

Reference studied for the overlay grammar (annotations drawn over a camera image, depth beside colour): https://docs.foxglove.dev/docs/visualization/panels/image. The world is Live Broadcast Graphics (seed 5260f28d, code-led); it rejects the docked-panel telemetry grid.

**Key Characteristics:**
- Dark only; the live video is the brightest element on screen.
- One reserved accent (telestrator yellow) for the active leg; green only for success.
- Flat broadcast plates with 3px corners, no shadows.
- System sans with tabular numerals everywhere; poster-size state and distance read from 2 m.
- Values snap; motion only for new caption phrases and the state-carrying beep dot.

## Colors

A restrained broadcast palette: a cool blue-black ground (hue 255) in four tonal steps, white text in three tones, and four signal colours whose meanings never overlap.

### Primary
- **Telestrator Yellow** (oklch(0.88 0.165 96), about #f8d63b): the active leg and only the active leg. The fingertip-to-object line and its arrowhead on camera, the leg's distance tag, the poster distance number in the guidance column, and the leg on the top-down map. Text on it is **Yellow Ink** (oklch(0.21 0.04 96)).

### Secondary
- **Grab Green** (oklch(0.82 0.16 152), about #67e18f): success states only. The GRAB state word, the leg and its tag once within reach, "On target" move values, the steady-tone beep dot, the finished clock time, and the 6px green frame around the video at grab. Text on it is **Green Ink** (oklch(0.2 0.04 152)).

### Tertiary
- **Warn Amber** (oklch(0.8 0.13 65), about #f7ab5d): "too close for lidar" and "covered by the arm": the dashed hand ring when the fingertip has no depth, its tag, and warning pipeline step statuses.
- **Alert Red** (oklch(0.62 0.19 27), about #e24942): the stale-frames banner only.

### Neutral
- **Blue-Black Ground** (oklch(0.165 0.022 255)): page background.
- **Video Well** (oklch(0.1 0.015 255)): behind the camera image, darker than the ground so letterboxing reads as the frame's edge.
- **Plate** (oklch(0.215 0.026 255)): solid informational plates (the "waiting for first frame" notice).
- **Raised Plate** (oklch(0.265 0.03 255)): meter tracks, the lidar-blind zone on the map.
- **Rule** (oklch(0.33 0.03 255)): 1px hairlines between move rows and pipeline steps, the meter centre tick, map range rings.
- **Scrim** (oklch(0.12 0.02 255 / 0.82)): translucent plates sitting over video (clock bug, captions, depth inset, simulated-lanes note).
- **Broadcast White** (oklch(0.965 0.005 255)): primary text, object brackets, the hand marker, meter marks.
- **Muted** (oklch(0.76 0.02 255)): labels, secondary text, remembered (out-of-view) objects drawn dashed.
- **Faint** (oklch(0.6 0.022 255)): empty values ("–"), step numbers, map annotations, the idle beep dot.

### Depth Ramp
- **Depth Far to Depth Near** (#0c1628 to #c4e4ff, set in `dashboard.py` as BGR): the lidar inset is a navy-to-pale ramp in the ground's blue family, so nearer reads lighter without spending yellow or green.

### Named Rules
**The Telestrator Rule.** Yellow is the active leg. If an element is not the fingertip-to-object line, its distance, or that leg on the map, it is not yellow.

**The Green Means Grab Rule.** Green appears only when something succeeded. Never as decoration, a status-OK dot, or a brand colour.

**The Brightest Thing Rule.** Nothing on screen may outshine the video. Plates over video use the scrim; the surround stays at ground level. This is why the surface is dark only (doctrine rule 29 exception, recorded here).

**The Depth Keeps Its Own Hue Rule.** Data visualisations (the depth inset) ramp within the ground's blue family; they never borrow a signal colour.

## Typography

**Display Font:** System UI stack (system-ui, -apple-system, Segoe UI, Ubuntu, Cantarell, Noto Sans, sans-serif)
**Body Font:** the same stack
**Label/Mono Font:** ui-monospace stack for inline commands only

**Character:** one neutral white sans, as in a broadcast data feed, with tabular numerals on everything so ticking values never jitter. Hierarchy comes from weight (400, 500, 600, 700) and the three text tones, then size.

The build ships no web fonts because the demo must run with the network unplugged. This is provisional, not a rule future surfaces inherit: a face self-hosted from the GX10 would also run offline and would give the poster-size state word its own voice.

### Hierarchy
- **Display** (700, min(4.3rem, 15cqi), 1.02, -0.025em): the guidance state word (Searching, Guiding, GRAB), read from 2 m.
- **Display Number** (700, min(5.16rem, 17cqi), 1, -0.03em): the fingertip-to-object distance in telestrator yellow.
- **Headline** (700, 2.074rem, 1.1): the clock bug time; the end-to-end latency figure runs one step down at 1.728rem.
- **Title** (600, 1.728rem, 1.25): the current caption phrase and the move-hand values (Side, Height, Reach).
- **Section** (600, 1.2rem to 1.44rem): column headings ("In the earbuds" at 1.44rem; rail headings at 1.2rem). Also the distance tag on camera (1.44rem).
- **Body** (400, 1rem, 1.45): labels, rail text, object tags (600). Notes cap at 40 to 46ch.
- **Label** (400 to 700, 0.833rem): clock sub-line, step statuses, meter ends, inset caption, the "Wearer hears" caption tag (700).

Scale is fixed rem at a 1.2 ratio (0.833, 1, 1.2, 1.44, 1.728, 2.074, 2.986, 4.3, 5.16). Root size is 16px, 18px from 1600px wide, 20px from 1880px wide, so the whole screen scales for a projector or large monitor without reflowing.

### Named Rules
**The Tabular Everything Rule.** `font-variant-numeric: tabular-nums` is set on the body; every number on screen holds its width while it changes.

**The Two Metre Rule.** The state word and the distance are deliberately larger than doctrine rule 15's Operate cap of 2x body (4.3x and 5.16x). They are read by judges from 1 to 2 m, the broadcast "poster" raise in the direction contract. No other text exceeds 2.074rem.

## Layout

Three fixed columns at desktop: the portrait camera at full viewport height (aspect 3/4 by default, set live from the stream), a flexible guidance column, and a system rail of 18 to 22rem. Gap 32px, page padding 24px. Overlays are absolutely placed on the video: clock bug top left, depth inset top right, captions as a full-width lower third, status banners centred.

Cells are fixed and never reflow as values change: move rows use a 5.5rem label column, pipeline steps a numbered grid with a right-aligned ms column, and the state and distance size against their column (container query units) rather than their content.

Responsive: below 1280px the system rail drops beneath as an auto-fit row of 16rem cells; below 900px everything stacks to one column, the video is sized to min(100%, 82dvh x aspect) and centred, padding tightens to 16px, the clock time steps down to 1.728rem and captions to 1.2rem. Spacing is a 4px base scale (4, 8, 12, 16, 24, 32, 48).

## Elevation & Depth

Flat. There are no shadows anywhere. Depth comes from tone: the video well is darker than the ground, solid plates step up one tone, and plates that sit over the video are a translucent scrim. The only emphatic edge is the 6px Grab Green frame drawn inside the video at the grab moment (an inset outline, not a shadow).

### Named Rules
**The Flat Plate Rule.** Graphics are flat plates. Separate with tone, scrim, or a 1px rule, never a shadow or a glow.

## Shapes

Every plate, tag, banner, and the video itself takes the same small 3px corner. Circles are reserved for dots and markers (the hand dot and ring, meter marks, the beep dot, map hand). Meter tracks are 10px rounded bars. On camera, the object is marked with square-capped corner brackets (5px stroke, non-scaling), the leg with a round-capped 5px line and a solid arrowhead; remembered objects and depth-less hands switch to dashes.

## Components

### Overlay Tags
Small solid plates pinned to points in the camera image.
- **Object tag:** white plate, ground-coloured text, 600; muted plate when the object is remembered rather than seen.
- **Leg tag:** yellow plate with yellow ink at 1.44rem, padding 2px 12px; turns green with green ink within reach.
- **Hand tag:** amber plate with dark ink, 500, only when the fingertip lacks depth.

### Clock Bug
Scrim plate at top left of the video, min width 11.5rem: elapsed time in Headline, a Label sub-line, and a 4px budget bar (white fill on rule track, scaled with transform). Time turns green when the grab completes.

### Lower-Third Captions
Full-width scrim strip at the bottom of the video showing what the wearer hears. The previous phrase sits above in muted body text; the current phrase is Title weight with a white "Wearer hears" tag at its left. When nothing is being said it drops to muted 1.2rem.
- **Motion:** a new phrase fades in over 220ms (ease cubic-bezier(0.22, 1, 0.36, 1)); with `prefers-reduced-motion: no-preference` it also slides 12px from the left. Repeated phrases do not re-animate.

### Status Banners and Plates
- **Stale banner:** Alert Red plate, white 1.2rem 600, centred at 30% height; appears only when frames stop arriving.
- **Notice plate:** solid Plate, muted text, centred; used for "waiting for the first frame" and lost connection.
- **Offline bug:** white plate, ground text, 1.44rem 700, top centre of the video when the network is unplugged.
- **Simulated note:** scrim plate under the clock bug in muted Label text, naming which lanes are fake.

### Guidance Column
State word (Display), distance (Display Number, yellow; green within reach; faint "–" when either point is missing), then three move rows between 1px rules: muted axis label, Title-weight value with an SVG arrow that snaps to direction. "On target" rows turn green and hide the arrow.

### Tone Meters
10px Raised Plate track with a 2px centre tick and a 22px white circular mark positioned along it; ends labelled in Label text. The mark fades to 25% opacity when the tone is off (200ms opacity only); position changes snap.

### Beep Indicator
14px dot plus text. Faint when silent, white and blinking at the real beep period while beeping (stepped opacity, no easing), solid green for the steady tone. The blink carries state, so it is kept under reduced motion. An "estimated" note appears when guidance does not report what it plays.

### Pipeline Steps
Numbered list in the rail: step number (faint), name (600), live ms (muted, right aligned), and a Label status line that turns white when OK and amber on warning. End-to-end latency follows as a 1.728rem figure with fps beneath.

### Top-Down Map
Inline SVG: range rings in rule colour, the lidar-blind half disc in Raised Plate, the wearer as a muted bar, the object as a white square outline (dashed muted when remembered), the hand as a white dot, and the leg in yellow.

## Do's and Don'ts

### Do:
- **Do** reserve Telestrator Yellow for the active leg: the line, its tag, the distance number, the map leg.
- **Do** use Grab Green only for success: grab state, within reach, On target, steady tone, finished clock, the green video frame.
- **Do** place plates over video on the scrim, and keep every plate to the 3px corner.
- **Do** keep tabular numerals on every number and size live values against their cell, not their content.
- **Do** show an honest empty state ("–", "No timed trials yet", "Waiting for the first frame") instead of a placeholder number.
- **Do** label simulated input on screen whenever a lane is fake.
- **Do** let values snap; animate only new caption phrases (opacity, plus 12px slide when motion is allowed) and the state-carrying beep blink.

### Don't:
- **Don't** add a light scheme or lift the surround; nothing may be brighter than the video.
- **Don't** colour the depth inset or any chart with yellow, green, amber, or red; ramp it within the ground's blue family.
- **Don't** use shadows or glows to separate plates.
- **Don't** set any text other than the state word and the distance above 2.074rem.
- **Don't** use Alert Red for anything but stale frames.
