---
version: 1
slug: "dashboard-index-html"
primary_target: "dashboard/index.html"
related_targets: ["dashboard/style.css","dashboard/app.js"]
---

# Judge dashboard

Mode: Operate (live monitor read by standing judges at 1 to 2 m; the team also tunes with it).
Job: let judges follow, in seconds, what the system sees, decides and says while a blindfolded wearer reaches for an object.
First read: the hand being steered to the object on the live camera.
Constraints: served by main.py from the GX10, works with the network unplugged (no CDN; fonts self-hosted from dashboard/fonts), fed by /events at up to 30 fps. Branding deferred.
Unresolved: final detector and hand tracker names (shown only when perception reports MODEL_NAMES), display hardware (assumed GX10 monitor, landscape 1920x1080).

## Direction contract

THESIS: The grab is a live event broadcast to the judges, not a debug panel; refuses the docked-panel telemetry grid.
OWN-WORLD: Flat broadcast plates on a blue-black ground over live video; one telestrator yellow reserved for the active leg (fingertip to object) on camera and map; grab green only for success; white system sans with tabular figures; fixed cells that never reflow.
STORY: Judges see the object found, the fingertip found, the yellow leg shrink, hear-by-caption what the wearer hears, then GRAB and the clock beating the unaided baseline.
FIRST VIEWPORT: Portrait live camera full height on the left with the yellow leg and its distance; clock bug top-left of the video; captions as a lower third on the video; centre column with the state as a poster word, distance, move-hand directions and tone meters; right rail with the pipeline steps and live ms, top-down map, internet status.
FORM: Live Broadcast Graphics, position 4 of 7, seed 5260f28d.
FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
