# Proprio: the 2-minute pitch

The pitch is 2:00 across the 11 deck slides, with about 40 seconds live: one presenter talks, one teammate wears the blindfold. Clicks (→) are deck builds. Read the words aloud at a normal pace; the times are where each slide should start.

| Time | Slide | Say this |
|---|---|---|
| 0:00 | 1 Proprio | "Hi, we're Veer, Gavin, Jake and Sasha, and this is Proprio." Then straight into the problem, no description yet. |
| 0:05 | 2 Problem (→ →) | "43 million people are blind." → "Their number one everyday challenge: finding a dropped or misplaced item." → "And 68% of their requests for help need an answer within 10 minutes." |
| 0:18 | 3 Ask for it | "With Proprio you just say *find my mug*, and your earbuds walk your hand there, one word at a time: right, down, forward, grab. It remembers the room, so a mug under your hand isn't lost." |
| 0:29 | 4 What exists today (→ ×5) | "Today's tools stop short." → "Meta glasses need internet and a volunteer." → "Be My AI only describes." → "Seeing AI gets you near the item, not onto it." → "AIGuide means holding the phone and scanning first." → "Proprio: hands-free, offline, any item." |
| 0:41 | 5 Why not vibration | "Why not vibration? 74% of overlapping buzzes are misread. A word needs no decoding." |
| 0:47 | 6 How it works (→ ×3) | "The iPhone streams color and lidar." → "Models find the mug and your fingertip." → "Lidar puts both in 3D." → "The biggest gap becomes one word." |
| 0:56 | 7 Core loop | "The math is one subtraction and a 5-centimeter rule." Click on. |
| 1:00 | 8 Live demo | "She's blindfolded, and the internet is unplugged." Wearer: "Find my mug." Talk over the screen (run-of-show below). If the GX10 isn't connected, this slide plays the demo video. |
| 1:38 | 9 Results (→) | "Depth is what makes it work. At six known distances, a camera alone was 52% accurate." → "Lidar, 99%." |
| 1:46 | 10 Why local (→ ×3) | "It all runs on this one GX10. Private: video of your home never leaves the room." → "128 gigabytes holds every model, even a 30-billion-parameter Nemotron." → "35 milliseconds a frame." → "And you just watched it work offline." |
| 1:57 | 11 Proprio | "Proprio. Thank you." |

Over time at rehearsal? Cut slide 7's line first (click straight past it), then say only the first and last builds on slide 4.

## Live demo run-of-show

About 40 seconds on slide 8, which shows the dashboard in demo mode full screen. Unplug the GX10's Ethernet before the pitch starts, so one grab shows both: blindfolded and offline.

1. **Before judges arrive:** run `python main.py --all-real`, open the deck at slide 1, press F. Mug on the taped spot, wearer on the floor tape, earbuds in, blindfold on. Check the dashboard says *Connected* and shows lidar points, then pull the Ethernet cable and check the top bar says *Unplugged, still running*.
2. **0:00** Presenter holds up the loose cable: "She's blindfolded, and the internet is unplugged." Wearer: "Find my mug."
3. **0:05 to 0:30** Presenter narrates the screen, not the person: "Top left is what she hears: right, down, forward. The yellow line is the gap between her fingertip and the mug, measured in 3D." (Adjust the pronoun to whoever wears it.)
4. **About 0:30** Grab: the screen turns green. Presenter: "Grab. About X seconds, blindfolded and offline." (Read the time off the top bar.) Back to the deck.

### If something fails

| Failure | Do this |
|---|---|
| Phone stream drops (frozen image, red banner) | Unplug and replug the USB-C; say "one second, reconnecting". If not back in 5 s, switch to the demo video. |
| Mug not found | Move the mug fully into view, matte side to the camera. Glass and black objects read badly. |
| Words don't play | Check the earbuds are the output device; the screen still shows every word, so narrate from it. |
| Anything else, or past 45 s | Play the [demo video](https://www.youtube.com/watch?v=wdUEhJDfMpk), kept open in a second full-screen tab, and say "here's a recorded run". Slide 8 also plays it by itself whenever the dashboard isn't reachable. |

## Q&A prep

Judges will test how the numbers were measured and why this needs a GX10; answer in one or two sentences, then stop.

| Likely question | Answer |
|---|---|
| How did you get 99% vs 52%? | "We measured depth to an item at six known distances, from 45 cm to 2 m. A camera alone was off by 48% on average: it read 37 to 72 cm at every distance, so the farther the item, the worse it got, 64% off at 2 m. Lidar stayed within 1%. Accuracy is 100% minus that average error." |
| Why does camera-only fail? | "A camera guesses depth per object, so the hand and the mug are each wrong by a different amount. That gap doesn't shrink as the hand closes in, so it says grab too early or too late. Lidar measures every point to about a centimeter." |
| Have you beaten Meta glasses? | "We haven't tested against them ourselves. A 2025 study measured Meta glasses at 56% retrieval accuracy for blind users; that's the bar we'd test against next." |
| Why a GX10 and not a laptop? | "Privacy first: it's a camera in someone's home, so nothing leaves the room. And 128 GB holds the detector, hand tracker, Whisper, the 30B Nemotron language model and an 8B vision-language model together, with a steady 35 ms loop and no internet." |
| Isn't the GX10 expensive? | "It's one box per home that serves the phone, and the person only wears a phone and earbuds they likely own. It replaces a person on a video call, not a phone." |
| Why words, not tones or vibration? | "Anyone understands 'right, down, forward' with no training. Buzzers need a motor per direction and get misread when they overlap." |
| What does "forward" mean? | "Reach further out from your chest, along the camera's line of sight. We line your hand up side to side and in height first; it says aligned, and from there it's only forward until grab. If it says forward and your hand stops getting closer, the item is past arm's reach, so it says lean forward." |
| What does "remembers the room" mean? | "The phone tracks its own motion, so we store where the item is in the room, not where it is in the picture. Today, when your hand covers the mug or you lean or turn, it keeps the mug's place for 15 seconds. The product keeps a map of every item it has seen, so you can ask for something you walked past earlier." |
| What about glass or shiny objects? | "Lidar reads them badly; that's our biggest limit. Matte objects work today." |
| What if the hand is too close for lidar? | "Lidar reads nothing under about 20 cm, so we estimate the hand's depth from its size in the image." |
| What did you build vs. reuse? | "Reused: YOLO-World, MediaPipe, Whisper, Nemotron, Cosmos Reason, and Record3D for the phone stream; the voice clips were made with ElevenLabs and play offline. Built: the 3D measuring, the room memory, the guidance state machine, the spoken cues, the dashboard and the depth test." |
| Prior work? | "AIGuide from Penn State guided a hand with a handheld phone and pre-scanned objects. We're hands-free and work on any item you name." |
| What's next? | "A memory beyond one room: the places you go, the items you use most and where each was last seen. Then trials with blind testers, glass and shiny items, and a more compact device." |

## Numbers and sources

Every number said aloud traces to a published source or to our own depth test.

| Number | What it is | Say it as | Source |
|---|---|---|---|
| 43 million | People who are blind worldwide (61 million by 2050) | "43 million people are blind" | [Lancet Global Health, 2020](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7820390/) |
| #1 | Finding dropped or misplaced items, described as the most common challenge | "the number one everyday challenge" (a stated finding, not a survey percentage) | [Yi et al., 2013](https://pmc.ncbi.nlm.nih.gov/articles/PMC3719410/) |
| 68% | VizWiz Social questions from blind users that needed an answer within 10 minutes | "68% of requests need an answer within 10 minutes" | [Brady et al., CHI 2013](https://www.cs.rochester.edu/hci/pubs/pdfs/chi2013-vizwiz.pdf) |
| 74% | Overlapping vibration buzzes misread (26% read correctly vs 93% one at a time; forearm study) | "74% of overlapping buzzes are misread", never "74% of people" | [Yeganeh, University of Iceland, 2026](https://opinvisindi.is/items/720e9be4-8063-445e-a987-e028461f877a/full) |
| 56% | Meta glasses retrieval accuracy for blind users (Be My AI 85.7%) | Only in Q&A, as a published bar, not our result | [Sridhar et al., ASSETS 2025](https://arxiv.org/html/2509.18672v1) |
| 99% vs 52% | Depth accuracy at six known distances, 45 cm to 2 m: 100% minus the average error (camera only 48%, lidar 1%) | "52% accurate with a camera alone, 99% with lidar" | Our depth test (45, 83, 100, 120, 150, 200 cm) |
| 128 GB | GX10 unified memory | "holds every model at once" | ASUS Ascent GX10 spec |
| 30B | Nemotron 3 Nano's size: 30 billion parameters, about 3 billion active per word | "a 30-billion-parameter Nemotron" | The model in `tools/run_nemotron.sh` (Nemotron-3-Nano-Omni-30B-A3B) |
| 35 ms | Perception time per frame on the GX10 | "35 milliseconds a frame" | Our measurement (README) |
| 5 cm, 0.3 s | Grab rule: fingertip within 5 cm of the item for 0.3 s | As stated | `guidance.py` |

Competitor facts on slide 4: [Be My Eyes on Ray-Ban Meta](https://bemyeyes.com/blog/be-my-eyes-on-ray-ban-meta), [Microsoft Research, Find My Things](https://www.microsoft.com/en-us/research/story/find-my-things/), [AIGuide, ACM TACCESS 2022](https://dl.acm.org/doi/10.1145/3508501). The Devpost text is on the [write-up page](../writeup/).
