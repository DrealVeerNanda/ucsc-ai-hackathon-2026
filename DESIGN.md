---
name: Blindfold pickup, live
description: Broadcast graphics over a luminous point-cloud world, so judges can follow a blindfolded grab they cannot hear; one world shared by the live dashboard and the pitch deck.
colors:
  ground: "oklch(0.165 0.022 255)"
  deep: "oklch(0.13 0.02 255)"
  video-well: "oklch(0.1 0.015 255)"
  plate: "oklch(0.215 0.026 255)"
  rule: "oklch(0.33 0.03 255)"
  scrim: "oklch(0.12 0.02 255 / 0.82)"
  text: "oklch(0.965 0.005 255)"
  muted: "oklch(0.76 0.02 255)"
  faint: "oklch(0.6 0.022 255)"
  telestrator-yellow: "oklch(0.88 0.165 96)"
  on-yellow: "oklch(0.21 0.04 96)"
  grab-green: "oklch(0.82 0.16 152)"
  on-green: "oklch(0.2 0.04 152)"
  warn-amber: "oklch(0.8 0.13 65)"
  alert-red: "oklch(0.62 0.19 27)"
  point-near: "#c4e4ff"
  point-far: "#162e54"
  point-standing: "#ecf6ff"
typography:
  display:
    fontFamily: "\"Barlow Semi Condensed\", \"Barlow Display Fallback\", sans-serif"
    fontSize: "min(7.5rem, 13cqi)"
    fontWeight: 700
    lineHeight: 0.95
    fontFeature: "\"tnum\""
  display-number:
    fontFamily: "\"Barlow Semi Condensed\", \"Barlow Display Fallback\", sans-serif"
    fontSize: "min(4.3rem, 8cqi)"
    fontWeight: 700
    lineHeight: 1
    fontFeature: "\"tnum\""
  headline:
    fontFamily: "\"Barlow Semi Condensed\", \"Barlow Display Fallback\", sans-serif"
    fontSize: "1.728rem"
    fontWeight: 700
    lineHeight: 1
    fontFeature: "\"tnum\""
  title:
    fontFamily: "\"Barlow Semi Condensed\", \"Barlow Display Fallback\", sans-serif"
    fontSize: "1.44rem"
    fontWeight: 700
    lineHeight: 1
    fontFeature: "\"tnum\""
  reason:
    fontFamily: "Barlow, \"Barlow Fallback\", sans-serif"
    fontSize: "1.44rem"
    fontWeight: 400
    lineHeight: 1.45
  sub:
    fontFamily: "Barlow, \"Barlow Fallback\", sans-serif"
    fontSize: "1.2rem"
    fontWeight: 400
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
  deck-poster:
    fontFamily: "\"Barlow Semi Condensed\", \"Barlow Display Fallback\", sans-serif"
    fontSize: "min(11vw, 19vh)"
    fontWeight: 700
    lineHeight: 0.95
  deck-head:
    fontFamily: "\"Barlow Semi Condensed\", \"Barlow Display Fallback\", sans-serif"
    fontSize: "min(5.4vw, 9.5vh)"
    fontWeight: 700
    lineHeight: 1
  deck-stat:
    fontFamily: "\"Barlow Semi Condensed\", \"Barlow Display Fallback\", sans-serif"
    fontSize: "min(6.2vw, 11vh)"
    fontWeight: 700
    lineHeight: 0.95
    fontFeature: "\"tnum\""
  deck-lede:
    fontFamily: "Barlow, \"Barlow Fallback\", sans-serif"
    fontSize: "min(2.5vw, 4.4vh)"
    fontWeight: 400
    lineHeight: 1.3
  deck-body:
    fontFamily: "Barlow, \"Barlow Fallback\", sans-serif"
    fontSize: "min(2vw, 3.6vh)"
    fontWeight: 400
    lineHeight: 1.35
  deck-small:
    fontFamily: "Barlow, \"Barlow Fallback\", sans-serif"
    fontSize: "min(1.35vw, 2.4vh)"
    fontWeight: 600
    lineHeight: 1.25
  deck-fine:
    fontFamily: "Barlow, \"Barlow Fallback\", sans-serif"
    fontSize: "min(0.95vw, 1.7vh)"
    fontWeight: 400
    lineHeight: 1.4
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
    typography: "{typography.title}"
    rounded: "{rounded.plate}"
    padding: "2px 12px"
  tag-leg-close:
    backgroundColor: "{colors.grab-green}"
    textColor: "{colors.on-green}"
    typography: "{typography.title}"
    rounded: "{rounded.plate}"
    padding: "2px 12px"
  tag-hand-warn:
    backgroundColor: "{colors.warn-amber}"
    textColor: "{colors.on-yellow}"
    rounded: "{rounded.plate}"
    padding: "2px 8px"
  key-over-video:
    backgroundColor: "{colors.scrim}"
    textColor: "{colors.muted}"
    typography: "{typography.label}"
    rounded: "{rounded.plate}"
    padding: "2px 8px"
  said-chip:
    backgroundColor: "{colors.scrim}"
    textColor: "{colors.muted}"
    typography: "{typography.body}"
    rounded: "{rounded.plate}"
    padding: "2px 8px"
  sim-chip:
    textColor: "{colors.muted}"
    typography: "{typography.label}"
    rounded: "{rounded.plate}"
    padding: "4px 12px"
  stale-banner:
    backgroundColor: "{colors.alert-red}"
    textColor: "{colors.text}"
    typography: "{typography.sub}"
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
    typography: "{typography.title}"
    rounded: "{rounded.plate}"
    padding: "8px 16px"
  hero-canvas:
    backgroundColor: "{colors.deep}"
    rounded: "{rounded.plate}"
  deck-label-object:
    backgroundColor: "{colors.text}"
    textColor: "{colors.ground}"
    typography: "{typography.deck-small}"
    rounded: "{rounded.plate}"
    padding: "2px 8px"
  deck-label-quiet:
    backgroundColor: "{colors.scrim}"
    textColor: "{colors.muted}"
    typography: "{typography.deck-small}"
    rounded: "{rounded.plate}"
    padding: "2px 8px"
  deck-compare-own-row:
    backgroundColor: "{colors.grab-green}"
    textColor: "{colors.ground}"
    typography: "{typography.deck-small}"
  deck-placeholder:
    textColor: "{colors.faint}"
    typography: "{typography.deck-small}"
    rounded: "{rounded.plate}"
    width: "min(30vw, 52vh)"
---

# Design System: Blindfold pickup, live

## Overview

**Creative North Star: "The Telestrator Booth"**

The product is shown as a live broadcast of one event: a blindfolded hand reaching an object. Judges stand 1 to 2 m away and cannot hear the earbuds, so every graphic is a broadcast element laid over a picture: a status strip, a telestrator line from fingertip to object, tags pinned to what the system sees. Since the 2026-10-10 redesign the main picture is a perception view: the table drawn as luminous points on deep navy, objects as labelled outlines, the way a self-driving car renders its world (reference fetched for it: Waymo Driver perception view, https://waymo.com/waymo-driver/). The chest camera remains beside it as evidence. The original overlay grammar came from https://docs.foxglove.dev/docs/visualization/panels/image.

Two surfaces share this world. The **dashboard** (`dashboard/`) is an Operate surface, a live monitor: real frames, real millimetres, real milliseconds, empty states where nothing is measured, and nothing about how the system works. The **pitch deck** (`deck/`) is a Persuade surface: a left text column over a persistent 3D stage that re-stages the same world (points, mug, fingertip, leg) with a camera shot per build.

Graphics are flat plates on a blue-black ground. Pictures (video, point cloud) are the brightest things on screen. One colour, telestrator yellow, belongs to the fingertip-to-object leg; green means grab and nothing else. Type is Barlow, a signage grotesk drawn from California plates and highway signs, with a semi-condensed bold cut for the words and figures read from 2 m. Values snap like a data feed; only the deck glides.

**Key Characteristics:**
- Dark only; the picture (video or point cloud) is the brightest element on screen.
- A perception world: blue near-to-far points on deep navy, things standing on the table in near-white, objects as outlines with name tags.
- One reserved accent (telestrator yellow) for the active leg; green only for the grab.
- Flat broadcast plates with 3px corners, no shadows.
- Self-hosted Barlow (Semi Condensed 700 for poster words and figures, 400 to 600 for text) with tabular numerals.
- Dashboard values snap; the deck's camera and builds ease.

## Colors

A restrained broadcast palette: a cool blue-black family (hue 255) in five tonal steps, white text in three tones, a blue point-cloud ramp, and four signal colours whose meanings never overlap.

### Primary
- **Telestrator Yellow** (about #f8d63b): the active leg and only the active leg. On camera, the fingertip-to-object line, its arrowhead and its distance tag; in the hero, the poster distance figure and the 3D leg (3.5px round line) with its tag; in the deck, the 3D leg. Text on it is **Yellow Ink**.

### Secondary
- **Grab Green** (about #67e18f): the grab. The state word and finished time in the status bar, the spoken "Grab", the leg and its tag within reach, the distance figure within reach, and the 6px frame around both the camera and the hero at grab. In the deck: the leg and word label at the grab shot, the closing "Grab." poster, and the product's own row in the competitor table (see Pitch Deck). Text on it is **Green Ink** or the ground.

### Tertiary
- **Warn Amber** (about #f7ab5d): "too close for lidar" and "covered by the arm": the dashed hand ring when the fingertip has no depth, and its tag.
- **Alert Red** (about #e24942): the stale-frames banner only.

### Neutral
- **Blue-Black Ground**: page background on both surfaces; the deck's 3D scene background.
- **Deep Navy**: the hero perception canvas, one step below the ground so the points glow.
- **Video Well**: behind the camera image, so letterboxing reads as the frame's edge.
- **Plate**: solid informational plates (the "waiting for the first frame" notice).
- **Rule**: 1px hairlines (status bar base, competitor table rows, the simulated-input chip outline), the 10 cm grid in both 3D views, the dashed placeholder frames.
- **Scrim**: translucent plates over pictures: the "Chest camera" key, earlier-instruction chips, deck labels for the camera and cloud, the competitor table body.
- **Broadcast White**: primary text, object brackets and 3D outlines, the hand marker.
- **Muted**: keys and secondary text, the reason line, remembered (out-of-view) objects drawn dashed.
- **Faint**: empty values ("–"), "Quiet" between instructions, the scene note, deck sources and slide counter.

### Point Cloud
- **Point Near to Point Far**: tabletop points ramp from pale blue near to navy far (rgb 196,228,255 to 22,46,84, 64 steps, in `app.js`).
- **Point Standing**: points of anything standing on the table (rgb 236,246,255), so objects lift off the surface. The deck's stage uses the same idea: table points lerp within the navy-to-pale family (from #0c1628), standing points in Broadcast White.

### Named Rules
**The Telestrator Rule.** Yellow is the active leg. If an element is not the fingertip-to-object line, its distance, or that leg in 3D, it is not yellow. This holds on both surfaces.

**The Green Means Grab Rule.** Green appears only for the grab and things within reach of it. Never as decoration, a status-OK dot, or a link colour. The one bounded exception is the deck's competitor table, where the product's own row is green.

**The Brightest Thing Rule.** Nothing may outshine the picture. Plates over video or points use the scrim; the surround stays at ground level; the hero sits on Deep Navy so the cloud is the light source. This is why both surfaces are dark only (doctrine rule 29 exception, recorded here).

**The Points Keep Their Own Hue Rule.** Point clouds ramp within the ground's blue family and near-white; they never borrow a signal colour.

## Typography

**Display Font:** Barlow Semi Condensed 700, with "Barlow Display Fallback" (Arial Narrow, size-adjust 88%)
**Body Font:** Barlow 400, 500, 600, with "Barlow Fallback" (Arial, size-adjust 93%)

**Character:** Barlow by Jeremy Tribby (OFL 1.1), a slightly rounded, low-contrast grotesk drawn from California's plates and highway signs (https://github.com/jpt/barlow). Signage lettering fits a screen that issues directions and is read from 2 m. The semi-condensed bold cut sets the poster words and figures, the way a score bug sets its numbers; regular-width Barlow sets everything else. Self-hosted from `dashboard/fonts/` (latin-subset woff2, plus OFL.txt; the deck loads the same files from `/fonts/`), display cut preloaded, so both surfaces run with the network unplugged.

### Hierarchy: dashboard
- **Display** (Semi Condensed 700, min(7.5rem, 13cqi), 0.95): the spoken word over the hero, with its direction arrow (0.8em). Faint "Quiet" between instructions; green "Grab".
- **Display Number** (Semi Condensed 700, min(4.3rem, 8cqi), 1): the fingertip-to-object distance in yellow, followed by "to the mug" in Sub, muted.
- **Headline** (Semi Condensed 700, 1.728rem, 1): the state word at the left of the status bar (Muted when quiet, green at grab).
- **Title** (Semi Condensed 700, 1.44rem, 1): status-bar values, the leg distance tag on camera, the Offline plate.
- **Reason** (Barlow 400, 1.44rem, Muted): the one line under the spoken word saying why ("Hand is 20 cm left of the mug").
- **Sub** (Barlow 400 to 600, 1.2rem): "to the mug", the stale banner (600).
- **Body** (Barlow 400 to 600, 1rem, 1.45): object tags (600), earlier-instruction chips.
- **Label** (Barlow 400, 0.833rem): status-bar keys (Muted, above each value), "Chest camera" key, simulated-input chip, scene note. 3D-view tags are 15px canvas text, 600.

Fixed rem at a 1.2 ratio (0.833, 1, 1.2, 1.44, 1.728) for everything except the two hero figures, which size against the hero container (cqi). Root is 16px, 18px from 1600px, 20px from 1880px.

### Hierarchy: pitch deck
Viewport-based so it fills whatever screen the GX10 drives: each step is min(vw, vh).
- **Poster** (Semi Condensed 700, min(11vw, 19vh), 0.95): the title slide; the closing "Grab." steps down to min(9vw, 15vh) in green.
- **Head** (Semi Condensed 700, min(5.4vw, 9.5vh), 1): slide headings.
- **Stat** (Semi Condensed 700, min(6.2vw, 11vh), 0.95): problem figures and results.
- **Lede** (Barlow 400, min(2.5vw, 4.4vh), 1.3, max 34ch): the one sentence under a heading. Big point lists use the display cut at this size.
- **Body** (Barlow 400 to 500, min(2vw, 3.6vh), 1.35, max 30 to 34ch): points, stat labels, flow steps.
- **Small** (Barlow 600, min(1.35vw, 2.4vh)): 3D labels, the competitor table, placeholder text.
- **Fine** (Barlow 400, min(0.95vw, 1.7vh), Faint, max 70ch): sources and the slide counter.

### Named Rules
**The Tabular Everything Rule.** `font-variant-numeric: tabular-nums` is set on the body of both surfaces; every number holds its width while it changes.

**The Two Metre Rule.** On the dashboard only the spoken word and the distance exceed the Operate cap of 2x body; they are read by judges from 1 to 2 m. No other dashboard text exceeds 1.728rem.

**The Display Cut Rule.** Barlow Semi Condensed 700 sets poster words, figures and single-line claims; never paragraphs, keys or labels. No negative tracking. Regular-width Barlow tops out at 600.

## Layout

**Dashboard.** A full-viewport grid (100dvh, no page scroll at desktop): a thin status bar, then the screen. The status bar is one wrapping row (gap 12px by 48px, padding 12px 24px, 1px Rule beneath): the state word, then Finding / Time (or "Grabbed in") / Camera to earbuds / Internet as Label keys over Title values, and the simulated-input chip pushed right. Below it, two columns with 24px gap and padding: the portrait chest camera at full height on the left (aspect from the stream, 3/4 by default), and the hero filling the rest. The hero is full bleed: the canvas covers it, and the HUD floats top left (32px in, 24px down), the earlier-instruction chips bottom left (max 60% wide), the scene note bottom right. The detector's cramped-padding rule is ignored on `dashboard/index.html` for this reason (recorded in `.impeccable/config.json`).

Under 900px the page scrolls and stacks: hero first at min(72dvh, 125vw), insets tighten to 16px, the scene note hides, and the camera follows at min(100%, 82dvh x aspect), centred.

**Pitch deck.** A fixed full-screen 3D stage behind everything; slides are absolutely placed text columns on the left (padding 7vh by 5vw, column min(46vw, 82vh), vertically centred, 3.2vh gap). Each shot pushes its subject right by 16 to 20% of the width (a camera view offset) so text and scene never overlap. Wide content (the competitor table) uses a full-width slide with the stage receded and unshifted. The live-demo slide is the dashboard full screen in an iframe.

Spacing on the dashboard is a 4px base scale (4, 8, 12, 16, 24, 32, 48); the deck spaces in vh and vw.

## Elevation & Depth

Flat. No box shadows on either surface. Depth comes from tone: Video Well and Deep Navy sit below the ground, solid plates step one tone up, plates over pictures are the scrim. Inside the 3D views, height is shown by scene cues, not UI elevation: a dark disc on the table under the hand and a dashed drop line. The only emphatic UI edge is the 6px Grab Green frame at grab: an inset outline on the camera, and an `::after` border on the hero so the canvas cannot cover it (fades in over 200ms).

### Named Rules
**The Flat Plate Rule.** Graphics are flat plates. Separate with tone, scrim, or a 1px rule, never a shadow or a glow.

## Shapes

Every plate, tag, chip, banner, panel and placeholder takes the same 3px corner. Circles are reserved for markers (the hand dot and ring, the 3D fingertip, deck list bullets). On camera, the object is marked with square-capped corner brackets (5px, non-scaling) and the leg with a round-capped 5px line and solid arrowhead; in 3D the object is a white wireframe box and the leg a round 3.5px line. Remembered objects and depth-less hands switch to dashes; so do placeholders.

## Components

### Status Bar
One thin strip that answers "what is happening" at a glance: state word, target, time, latency, internet. Keys are Muted Label above Title values; empty values read "–" or "nothing yet". While any lane is fake, an outlined chip at the right (1px Rule border, Muted Label, 4px 12px) says "Simulated input", and the latency value never shows a fake number. Offline turns the Internet value to plain white text; the camera also gets the Offline plate.

### Overlay Tags (camera)
Small solid plates pinned to points in the camera image.
- **Object tag:** white plate, ground text, 600; Muted plate when remembered.
- **Leg tag:** yellow plate, Yellow Ink, Title cut, 2px 12px; green with Green Ink within reach.
- **Hand tag:** amber plate, dark ink, 500, only when the fingertip lacks depth.
- **Camera key:** "Chest camera" on a scrim plate, top left, Muted Label.

### Status Plates
- **Stale banner:** Alert Red, white Sub 600, centred at 30% height; only when frames stop arriving.
- **Notice plate:** solid Plate, Muted 500, centred; "waiting for the first frame" and lost connection.
- **Offline plate:** white plate, ground text, Title cut, top centre of the video.

### Perception Hero (signature)
The 3D view as the hero: where the hand is relative to the object, in one look.
- **Canvas:** Deep Navy, 3px corner, full bleed. Points are splatted with a depth buffer into an offscreen canvas (no WebGL, no library, offline). Tabletop points use the blue near-to-far ramp; anything standing on the table is near-white.
- **Level, third-person view:** the server fits the table plane and sends the rotation; the view sits behind and above the wearer (about 34° down). A 10 cm Rule grid gives scale. Drag turns the scene with the pointer (grab / grabbing cursor).
- **Marks:** the object as a white wireframe box resting on the table with its name tag (dashed Muted and "last seen" when remembered); the hand as a white dot with a ground ring, a "Hand" tag (hidden at grab range), a dark disc and dashed drop line to the table; the leg in yellow with its distance tag, green within reach. Tags are 15px plates.
- **HUD:** spoken word (Display) with arrow, Reason line, distance (Display Number) with "to the mug". Values snap.
- **Earlier instructions:** up to five scrim chips, newest first and white, repeats collapsed ("Right ×5").
- **Note:** Faint Label bottom right ("View from behind you, 10 cm grid.", plus the assumed-tilt sentence when no table is found).

### Pitch Deck (second surface)
Same world, staged, and extremely minimal: eleven slides (title, problem, the ask, competitors, why not vibration, how it works, core loop, live demo, results, why local, close), one line or one number per build. `deck/index.html`, `deck.css`, `deck.js`, `stage.js`; three.js r169, the fonts and the icon live inside `deck/`, so any static server runs it offline. Mechanics informed by the standardphysics deck (github.com/Imhaohao/standardphysics, apps/web/src/components/deck).
- **Persistent stage:** one three.js scene behind every slide: table points, the mug, the fingertip, the yellow leg, plus props raised per shot (camera frustum, sweep plane, axis split of the leg, object box). Each slide lists a camera shot per build (`data-shots`); camera, look-at and every prop level ease toward the shot each frame, so moves between slides glide. Cameras sit just behind the wearer's right shoulder and the scene is mirrored once, so "right" is always screen right.
- **Clear text column:** on slides with left-hand text, points fade toward 4% brightness across the left 40-62% of the width, so type never sits on busy points. Text is full white; fine print is Muted, never Faint.
- **Tracking labels:** HTML plates that follow projected 3D points, positioned with left/top (never transform, so the pop and buzz scales stay in place) and hidden while the camera is still moving: the mug in white, axis labels on scrim, and the spoken word in a white speech bubble with a tail toward the hand (green on Grab), popping on each new word. Yellow stays the leg alone.
- **Builds:** `data-step` builds accumulate (rise 1.2vh and fade, 420ms); `data-only` builds (0-based) replace each other in place, so every click changes what is on screen.
- **Transitions, one per slide** (Veer: "change it up significantly"): a curtain in the ground colour reveals each slide its own way: lights-out fade for the problem, an iris opening on the scene for the ask, a horizontal wipe for competitors, closing blinds for how it works, doors splitting top and bottom for the core loop, the live view growing out of a small screen, a curtain lifting for results, a diagonal sweep for why local, vertical shutters for why not vibration, and a centre-out reveal for the close. Text has its own effects: count-ups on every number, a typewriter on "Find my mug.", a decode on the title and one-word reasons, rows that slide in with crosses that draw themselves and our row landing last, words that roll like a counter, equations written on left to right. Reduced motion turns every one into a short fade.
- **One look per slide:** the problem is a fog of scattered points with a wandering hand; the ask has no text at all: a low shot of a point-cloud hand (palm down, index out, the tracked fingertip at its tip) reaching one axis at a time, with a speech bubble per word; the guided reach only starts once the camera settles, and each word holds for about two seconds; how it works scans a bright line across the table, then shows the point-cloud hand with a "Fingertip" tag beside the boxed mug, splits the leg quickly (this shot eases 2.6x faster), splits the leg into right/down/forward, then shows a large speech bubble; the core loop is straight overhead with an x/y/z gizmo at the fingertip; results look down on the mug with a 6 cm ring and landing dots (green for lidar, muted for camera only, an illustration of the simulated medians); why not vibration rings the fingertip with four motor circles that buzz one at a time, then all at once for the overload build; the problem shows a fog, then a "?" plate over a half-formed mug, then the chest camera; why local raises a wireframe privacy dome, the GX10 box, a faster guided reach, and the GX10's internet cable parting.
- **Big type:** stat figures at min(14vw, 24vh), single words at min(8vw, 14vh), the quote at min(8.5vw, 15vh), all in the display cut.
- **Competitors:** one tool per Enter: name in the display cut, what it does in Muted Small, its gap in white 600 Small, and a cross that draws itself as the row arrives. The last Enter brings the product's own row, a Grab Green plate with green ink and a check: the bounded exception to the Green Means Grab Rule.
- **Core loop slide:** four equations typeset with KaTeX 0.16.11 (vendored in `deck/vendor/katex`, MIT, its own math fonts; the one place a second family appears, and only for mathematics), each with a Muted one-line gloss, written on one after another.
- **Modes:** `?still` snaps every level, camera and animation (screenshots, PDF); `?auto` advances on each slide's `data-dur` (recording); `?dash=` points the live slide at a dashboard on another machine; with none reachable the live slide plays `media/demo-run.mp4` if present. The cursor hides after 1.4s idle; slide counter bottom right.
- **Speaker notes:** N opens `notes.html`, synced over BroadcastChannel: slide title, an elapsed timer that appends "N s over" past the summed budget, the notes, the next slide, and Back / Next buttons.

## Do's and Don'ts

### Do:
- **Do** reserve Telestrator Yellow for the active leg: the line, its tag, the distance figure, the 3D leg.
- **Do** use Grab Green only for the grab and within-reach states, plus the green frame on both panels at grab.
- **Do** put the perception canvas on Deep Navy and keep point clouds in the blue ramp and near-white.
- **Do** place plates over pictures on the scrim, and keep every plate to the 3px corner.
- **Do** keep tabular numerals on every number and size live values against their container, not their content.
- **Do** show an honest empty state ("–", "nothing yet", "Waiting for the first frame") instead of a placeholder number.
- **Do** label simulated input on screen whenever a lane is fake.
- **Do** mark missing deck assets as dashed placeholder frames that name what belongs there.
- **Do** let dashboard values snap; let only the deck's camera, labels and builds ease, and honour `?still` and reduced motion.

### Don't:
- **Don't** add a light scheme or lift the surround; nothing may be brighter than the picture.
- **Don't** colour point clouds or charts with yellow, green, amber, or red.
- **Don't** use shadows or glows to separate plates.
- **Don't** set any dashboard text other than the spoken word and the distance above 1.728rem.
- **Don't** set paragraphs, keys or labels in the semi-condensed display cut, track it negatively, or push regular-width Barlow past 600.
- **Don't** fall back to the platform UI stack; text names Barlow then its size-adjusted fallback.
- **Don't** use Alert Red for anything but stale frames.
- **Don't** explain the pipeline on the dashboard; how it works belongs in the deck.
