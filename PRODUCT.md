# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Plain HTML, CSS and JavaScript served by `main.py` from the GX10 itself (Python standard library HTTP server, no build step, no CDN). Must work with the network unplugged. Inferred from the offline requirement; not separately confirmed.

## Users

- **Wearer:** a blindfolded person with an iPhone on their chest who asks for an object by voice and is steered to it by audio. They never see the screen.
- **Judges:** ASUS Local AI track judges at the UCSC AI Hackathon (October 10 to 11, 2026), standing at the demo table. They cannot hear the earbuds, so the screen is how they follow what the system sees, decides and says.
- **Team:** uses the same screen while integrating and tuning (frame age, per-stage latency, which modules are fake).

## Product Purpose

A wearer says "find my mug" and picks it up blindfolded. The iPhone sends color and lidar depth to the ASUS Ascent GX10; the GX10 finds the object and the fingertip, turns both into 3D positions, and plays tones that steer the hand, then says "grab." Success is a blindfolded grab that is faster and more reliable than searching unaided, with every model running on the GX10.

## Positioning

Hands-free (phone on the chest, not held), objects named by voice rather than pre-scanned, and all inference on local hardware. Prior work credited: AIGuide (Penn State, 2022) guided a hand with a handheld phone and pre-scanned objects.

## Operating Context

Live demo at a hackathon table: plain table, three or four matte objects, taped standing spot, blindfold, wired or Bluetooth earbuds. Three-minute demo: baseline time, live blindfolded run, move the object, pull the network cable and run again, numbers, why local. Judges score: solves a real problem, innovation and novel use of local hardware, value of local AI, inspire and engage.

## Capabilities and Constraints

- Camera frame axes: x right, y down, z forward, as seen by the wearer.
- Lidar reads nothing closer than about 20 to 30 cm; glass, shiny metal and black objects give bad depth.
- Guidance states: idle, searching, waiting for hand, guiding, grab.
- Fixed phrases: looking, found it, show me your hand, grab, I lost it, I can't see that.
- Audio mapping (first version, may change): pan toward the side the hand should move, pitch up/down for height, faster beeps when closer, steady tone within 5 cm, 3 cm dead zone per axis.
- Open: final detector and hand tracker (YOLO-World / MediaPipe planned, fallbacks exist), whether camera pose is available, project branding.

## Evidence on Hand

- Live frames, positions, states and per-stage latency from `main.py` (fake modules until each lane lands).
- `trials.csv`: baseline and guided timings, filled during the event. No numbers exist yet; never display invented ones.

## Product Principles

1. The screen exists because judges cannot hear the earbuds: always show what the wearer is hearing.
2. Show the mechanism, not claims: real frames, real positions, real timings.
3. Local is the point: make it visible that everything runs on the GX10 and works offline.
4. Honest numbers only: empty states instead of invented results.

## Accessibility & Inclusion

The product is assistive technology for finding objects without sight. The dashboard itself is for sighted observers and must be readable from 1 to 2 m.
