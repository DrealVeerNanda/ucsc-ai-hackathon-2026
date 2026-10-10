---
name: Blindfold pickup, live
description: Live broadcast graphics over a chest camera, so judges can follow a blindfolded grab they cannot hear.
colors:
  ground: "oklch(0.165 0.022 255)"
  plate: "oklch(0.215 0.026 255)"
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
    fontFamily: "\"Barlow Semi Condensed\", \"Barlow Display Fallback\", sans-serif"
    fontSize: "min(4.3rem, 15cqi)"
    fontWeight: 700
    lineHeight: 1.02
    fontFeature: "\"tnum\""
  display-number:
    fontFamily: "\"Barlow Semi Condensed\", \"Barlow Display Fallback\", sans-serif"
    fontSize: "min(5.16rem, 17cqi)"
    fontWeight: 700
    lineHeight: 1
    fontFeature: "\"tnum\""
  spoken:
    fontFamily: "\"Barlow Semi Condensed\", \"Barlow Display Fallback\", sans-serif"
    fontSize: "2.986rem"
    fontWeight: 700
    lineHeight: 1.1
  headline:
    fontFamily: "\"Barlow Semi Condensed\", \"Barlow Display Fallback\", sans-serif"
    fontSize: "2.074rem"
    fontWeight: 700
    lineHeight: 1.1
    fontFeature: "\"tnum\""
  title:
    fontFamily: "Barlow, \"Barlow Fallback\", sans-serif"
    fontSize: "1.728rem"
    fontWeight: 600
    lineHeight: 1.25
    fontFeature: "\"tnum\""
  section:
    fontFamily: "Barlow, \"Barlow Fallback\", sans-serif"
    fontSize: "1.2rem"
    fontWeight: 600
    lineHeight: 1.45
  body:
    fontFamily: "Barlow, \"Barlow Fallback\", sans-serif"
    fontSize: "1rem"
    fontWeight: 400
    lineHeight: 1.45
    fontFeature: "\"tnum\""
  label:
    fontFamily: "Barlow, \"Barlow Fallback\", sans-serif"
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
  said-chip:
    backgroundColor: "{colors.plate}"
    textColor: "{colors.muted}"
    typography: "{typography.body}"
    rounded: "{rounded.plate}"
    padding: "2px 8px"
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

Graphics are flat plates on a blue-black ground. The video is always the brightest thing on screen; the surround is dark by decision so it never competes. One colour, telestrator yellow, is reserved for the active leg; green means success and nothing else. Type is Barlow, a signage grotesk drawn from California plates and highway signs, in white with tabular figures: a semi-condensed bold cut for the broadcast figures and words, regular width for everything else, set big enough that the state word, the distance and the spoken instruction read from across the table. Motion is almost absent: values snap like a broadcast data feed, and only new caption phrases wipe in.

Reference studied for the overlay grammar (annotations drawn over a camera image, depth beside colour): https://docs.foxglove.dev/docs/visualization/panels/image. The world is Live Broadcast Graphics (seed 5260f28d, code-led); it rejects the docked-panel telemetry grid.

**Key Characteristics:**
- Dark only; the live video is the brightest element on screen.
- One reserved accent (telestrator yellow) for the active leg; green only for success.
- Flat broadcast plates with 3px corners, no shadows.
- Self-hosted Barlow (Semi Condensed 700 for figures, 400 to 600 for text) with tabular numerals everywhere; poster-size state, distance and spoken word read from 2 m.
- Values snap; motion only for new caption phrases.

## Colors

A restrained broadcast palette: a cool blue-black ground (hue 255) in four tonal steps, white text in three tones, and four signal colours whose meanings never overlap.

### Primary
- **Telestrator Yellow** (oklch(0.88 0.165 96), about #f8d63b): the active leg and only the active leg. The fingertip-to-object line and its arrowhead on camera, the leg's distance tag, the poster distance number in the guidance column, and the leg in the 3D scene. Text on it is **Yellow Ink** (oklch(0.21 0.04 96)).

### Secondary
- **Grab Green** (oklch(0.82 0.16 152), about #67e18f): success states only. The GRAB state word, the leg and its tag once within reach, "On target" move values, the spoken "Grab", the finished clock time, and the 6px green frame around the video at grab. Text on it is **Green Ink** (oklch(0.2 0.04 152)).

### Tertiary
- **Warn Amber** (oklch(0.8 0.13 65), about #f7ab5d): "too close for lidar" and "covered by the arm": the dashed hand ring when the fingertip has no depth, its tag, and warning pipeline step statuses.
- **Alert Red** (oklch(0.62 0.19 27), about #e24942): the stale-frames banner only.

### Neutral
- **Blue-Black Ground** (oklch(0.165 0.022 255)): page background.
- **Video Well** (oklch(0.1 0.015 255)): behind the camera image, darker than the ground so letterboxing reads as the frame's edge.
- **Plate** (oklch(0.215 0.026 255)): solid informational plates (the "waiting for first frame" notice), the earlier-instruction chips, and the 3D scene's ground.
- **Rule** (oklch(0.33 0.03 255)): 1px hairlines between move rows and pipeline steps, the 10 cm grid in the 3D scene.
- **Scrim** (oklch(0.12 0.02 255 / 0.82)): translucent plates sitting over video (clock bug, captions, depth inset, simulated-lanes note).
- **Broadcast White** (oklch(0.965 0.005 255)): primary text, object brackets, the hand marker.
- **Muted** (oklch(0.76 0.02 255)): labels, secondary text, remembered (out-of-view) objects drawn dashed.
- **Faint** (oklch(0.6 0.022 255)): empty values ("–"), step numbers, "Quiet" between instructions.

### Depth Ramp
- **Depth Far to Depth Near** (#0c1628 to #c4e4ff, set in `dashboard.py` as BGR): the lidar inset is a navy-to-pale ramp in the ground's blue family, so nearer reads lighter without spending yellow or green.

### Named Rules
**The Telestrator Rule.** Yellow is the active leg. If an element is not the fingertip-to-object line, its distance, or that leg in the 3D scene, it is not yellow.

**The Green Means Grab Rule.** Green appears only when something succeeded. Never as decoration, a status-OK dot, or a brand colour.

**The Brightest Thing Rule.** Nothing on screen may outshine the video. Plates over video use the scrim; the surround stays at ground level. This is why the surface is dark only (doctrine rule 29 exception, recorded here).

**The Depth Keeps Its Own Hue Rule.** Data visualisations (the depth inset) ramp within the ground's blue family; they never borrow a signal colour.

## Typography

**Display Font:** Barlow Semi Condensed 700, with "Barlow Display Fallback" (Arial Narrow, size-adjust 88%)
**Body Font:** Barlow 400, 500, 600, with "Barlow Fallback" (Arial, size-adjust 93%)
**Label/Mono Font:** ui-monospace stack for inline commands only (`python trials.py baseline`)

**Character:** Barlow by Jeremy Tribby (OFL 1.1), a slightly rounded, low-contrast grotesk drawn from California's car plates, highway signs, buses and trains (source: https://github.com/jpt/barlow). Signage lettering fits a screen that issues directions and is read from 2 m. The semi-condensed bold cut (the display token) carries the broadcast figures and words (clock, distance, state word, spoken instruction, end-to-end figure, offline plate and line, camera distance tag), the way a score bug sets its numbers; regular-width Barlow carries everything else. Tabular numerals on everything so ticking values never jitter.

Self-hosted from `dashboard/fonts/` (four latin-subset woff2 files, about 16 KB each, plus OFL.txt), so the page runs with the network unplugged. The display cut is preloaded. Chosen 2026-10-10 at Veer's request to replace the platform UI font.

### Hierarchy
- **Display** (Barlow Semi Condensed 700, min(4.3rem, 15cqi), 1.02, no tracking): the guidance state word (Looking, Guiding, Grab), read from 2 m.
- **Display Number** (Barlow Semi Condensed 700, min(5.16rem, 17cqi), 1): the fingertip-to-object distance in telestrator yellow.
- **Spoken Word** (Barlow Semi Condensed 700, 2.986rem, 1.1): the instruction being spoken now, with its arrow.
- **Headline** (Barlow Semi Condensed 700, 2.074rem, 1.1): the clock bug time. The end-to-end latency figure uses the same cut one step down at 1.728rem; the leg distance tag and the offline plate use it at 1.44rem.
- **Title** (Barlow 600, 1.728rem, 1.25): the current caption phrase and the move-hand values (Side, Height, Reach).
- **Section** (Barlow 600, 1.2rem to 1.44rem): column headings ("In the earbuds" at 1.44rem; rail headings at 1.2rem).
- **Body** (Barlow 400, 1rem, 1.45): labels, rail text, the "why" line, earlier-instruction chips, object tags (600). Notes cap at 40ch.
- **Label** (Barlow 400 to 600, 0.833rem): clock sub-line, step statuses, inset caption, fps, 3D note, simulated note, the "Wearer hears" caption tag (600).

Scale is fixed rem at a 1.2 ratio (0.833, 1, 1.2, 1.44, 1.728, 2.074, 2.986, 4.3, 5.16). Root size is 16px, 18px from 1600px wide, 20px from 1880px wide, so the whole screen scales for a projector or large monitor without reflowing.

### Named Rules
**The Tabular Everything Rule.** `font-variant-numeric: tabular-nums` is set on the body; every number on screen holds its width while it changes.

**The Two Metre Rule.** The state word, the distance and the spoken instruction are deliberately larger than doctrine rule 15's Operate cap of 2x body (4.3x, 5.16x and 2.986x). They are read by judges from 1 to 2 m, the broadcast "poster" raise in the direction contract. No other text exceeds 2.074rem.

**The Display Cut Rule.** Barlow Semi Condensed 700 (the display token) is for broadcast figures and the words judges read from 2 m; it never sets running text or labels. It takes no negative tracking. Regular-width Barlow tops out at 600.

## Layout

Three fixed columns at desktop: the portrait camera at full viewport height (aspect 3/4 by default, set live from the stream), a flexible guidance column, and a system rail of 18 to 22rem. Gap 32px, page padding 24px. Overlays are absolutely placed on the video: clock bug top left, depth inset top right, captions as a full-width lower third, status banners centred.

Cells are fixed and never reflow as values change: move rows use a 5.5rem label column, pipeline steps a numbered grid with a right-aligned ms column, and the state and distance size against their column (container query units) rather than their content.

Responsive: below 1280px the system rail drops beneath as an auto-fit row of 16rem cells; below 900px everything stacks to one column, the video is sized to min(100%, 82dvh x aspect) and centred, padding tightens to 16px, the clock time steps down to 1.728rem and captions to 1.2rem. Spacing is a 4px base scale (4, 8, 12, 16, 24, 32, 48).

## Elevation & Depth

Flat. There are no shadows anywhere. Depth comes from tone: the video well is darker than the ground, solid plates step up one tone, and plates that sit over the video are a translucent scrim. The only emphatic edge is the 6px Grab Green frame drawn inside the video at the grab moment (an inset outline, not a shadow).

### Named Rules
**The Flat Plate Rule.** Graphics are flat plates. Separate with tone, scrim, or a 1px rule, never a shadow or a glow.

## Shapes

Every plate, tag, banner, and the video itself takes the same small 3px corner. Circles are reserved for dots and markers (the hand dot and ring, the 3D fingertip). On camera, the object is marked with square-capped corner brackets (5px stroke, non-scaling), the leg with a round-capped 5px line and a solid arrowhead; remembered objects and depth-less hands switch to dashes.

## Components

### Overlay Tags
Small solid plates pinned to points in the camera image.
- **Object tag:** white plate, ground-coloured text, 600; muted plate when the object is remembered rather than seen.
- **Leg tag:** yellow plate with yellow ink, display cut 700 at 1.44rem, padding 2px 12px; turns green with green ink within reach.
- **Hand tag:** amber plate with dark ink, 500, only when the fingertip lacks depth.

### Clock Bug
Scrim plate at top left of the video, min width 11.5rem: elapsed time in Headline, a Label sub-line, and a 4px budget bar (white fill on rule track, scaled with transform). Time turns green when the grab completes.

### Lower-Third Captions
Full-width scrim strip at the bottom of the video showing what the wearer hears. The previous phrase sits above in muted body text; the current phrase is Title weight with a white "Wearer hears" tag at its left. When nothing is being said it drops to muted 1.2rem.
- **Motion:** a new phrase fades in over 220ms (ease cubic-bezier(0.22, 1, 0.36, 1)); with `prefers-reduced-motion: no-preference` it also slides 12px from the left. Repeated phrases do not re-animate.

### Status Banners and Plates
- **Stale banner:** Alert Red plate, white 1.2rem 600, centred at 30% height; appears only when frames stop arriving.
- **Notice plate:** solid Plate, muted text, centred; used for "waiting for the first frame" and lost connection.
- **Offline bug:** white plate, ground text, display cut 1.44rem 700, top centre of the video when the network is unplugged.
- **Simulated note:** scrim plate under the clock bug in muted Label text, naming which lanes are fake.

### Guidance Column
State word (Display), distance (Display Number, yellow; green within reach; faint "–" when either point is missing), then three move rows between 1px rules: muted axis label, Title-weight value with an SVG arrow that snaps to direction. "On target" rows turn green and hide the arrow.

### Spoken Instruction
Guidance speaks single words (left, right, up, down, forward, back) at most twice a second, plus the fixed phrases (Looking, Found it, Show me your hand, Grab, I lost it, I can't see that). Under "In the earbuds": the word spoken in the last 2.5 s in the Spoken Word style with its direction arrow (no arrow for phrases; green for Grab; Faint "Quiet" between instructions), then one Muted line saying why, computed from the measured positions ("Hand is 20 cm left of the mug"), then up to five earlier instructions as Plate chips, newest first and white, repeats collapsed ("Right ×5"). Values snap; no motion. `main.py` records exactly what guidance plays by listening to its sound sink, so the panel never guesses. Replaced the tone meters and beep dot on 2026-10-10 when the team settled on spoken words.

### Pipeline Steps
Numbered list in the rail: step number (faint), name (600), live ms (muted, right aligned; "Simulated" for a fake lane), and a short Label status line that turns white when OK and amber on warning. End-to-end latency follows as a 1.728rem display figure labelled "camera to earbuds" ("processing only" when that is all that is measured), with fps beneath as "28.9 fps". While any lane is fake the figure reads "Simulated" in Muted, never a number.

### 3D Scene
A canvas in the rail (13rem tall, Plate ground, 3px corner) that answers "where is my hand relative to the mug" in one look.
- **Level, not tilted:** the server fits the tabletop plane (RANSAC on the cloud, smoothed) and sends the rotation, so the table is flat and up is up. With no table in view it assumes the planned 30° chest tilt and says so under the view.
- **Fixed third-person view:** seen from behind and above the wearer (about 34° down), like a game camera. No automatic motion. Drag to look around; double-click returns to the default view. The view recentres smoothly on the reach area.
- **Scale and height cues:** a 10 cm grid in Rule colour on the tabletop; the hand casts a ground-coloured shadow disc with a dashed drop line, so its height reads without stereo.
- **Points:** about 5,000 lidar points in real camera colour, splatted with a depth buffer into an offscreen canvas (no WebGL, no library, works offline). Tabletop points are dimmed to 38% so anything standing on the table stays bright.
- **Labelled marks:** "You" (muted dot at the chest camera), the object as a white wireframe box resting on the table with its name tag (dashed muted with "last seen" when remembered), "Hand" as a white dot (label hidden at grab range so it never covers the object name), and the leg as a 3.5px round yellow line with a yellow distance tag (green within reach).
- **Transport:** the cloud travels as its own `cloud` SSE event every third frame (int16 mm xyz plus uint8 rgb, base64, plus the level rotation and table height), so a newly opened page gets one at once.
- **Note:** one muted Label line under it: "View from behind you, 10 cm grid." (plus the assumed-tilt sentence when no table is found).
Replaced the flat top-down map on 2026-10-10 at Veer's request, then reworked the same day for readability (levelled, labelled, no sway).

## Do's and Don'ts

### Do:
- **Do** reserve Telestrator Yellow for the active leg: the line, its tag, the distance number, the 3D leg.
- **Do** use Grab Green only for success: grab state, within reach, On target, the spoken Grab, finished clock, the green video frame.
- **Do** place plates over video on the scrim, and keep every plate to the 3px corner.
- **Do** keep tabular numerals on every number and size live values against their cell, not their content.
- **Do** show an honest empty state ("–", "None yet", "Waiting for the first frame") instead of a placeholder number.
- **Do** label simulated input on screen whenever a lane is fake.
- **Do** let values snap; animate only new caption phrases (opacity, plus 12px slide when motion is allowed).

### Don't:
- **Don't** add a light scheme or lift the surround; nothing may be brighter than the video.
- **Don't** colour the depth inset or any chart with yellow, green, amber, or red; ramp it within the ground's blue family.
- **Don't** use shadows or glows to separate plates.
- **Don't** set any text other than the state word, the distance and the spoken instruction above 2.074rem.
- **Don't** set running text or labels in the semi-condensed display cut, track it negatively, or push regular-width Barlow past 600.
- **Don't** fall back to the platform UI stack; text names Barlow then its size-adjusted fallback.
- **Don't** use Alert Red for anything but stale frames.
