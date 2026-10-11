For blind people, finding a dropped or misplaced item is the most common everyday challenge. Proprio guides their hand onto anything they ask for, one spoken word at a time.

## Inspiration
At a coffee shop, one of us watched a blind customer drop something and sweep the floor with their hands, trying to find it. Being able to see shouldn't decide whether you can find your own things.

Most requests for help from blind people need an answer within minutes, yet today's tools only describe what a camera sees or call a volunteer over the internet. None of them walk your hand onto the item, hands-free, without the internet.

## What it does
You wear an iPhone on your chest and say "find my mug." Proprio finds the mug and your fingertip in 3D and speaks one word at a time into your earbuds: _right_, _down_, _forward_, then _grab_. It works for any item you can name, with no setup or pre-scanning. It remembers where items are in the room, so it can still guide you to one your hand is covering or the camera can't see. Every model runs on one ASUS Ascent GX10, so no video leaves the room and it keeps working with the internet unplugged.

## How we built it
- **See:** the iPhone streams color and lidar depth to the GX10.
- **Ask:** Whisper hears the request, and Nemotron 3 Nano (30B) turns it into an item name, so "I'm thirsty" becomes "water bottle".
- **Find:** YOLO-World finds the item by name; MediaPipe tracks the fingertip.
- **Measure:** lidar depth turns both into 3D points.
- **Remember:** the phone's own motion tracking (ARKit pose) keeps the item's place in room coordinates, so it holds steady when the hand covers it or the wearer leans or turns.
- **Speak:** the axis with the biggest gap becomes one word, decided from the last 0.8 s of frames so one noisy frame can't change it.
- **Show:** a live dashboard draws the lidar scene in 3D with the hand, the item and the word being spoken.
- **Test:** we checked depth to an item at six known distances, from 45 cm to 2 m, with a camera alone and with lidar.

Mathematically:

$$\mathbf{p} = d\,K^{-1}\begin{bmatrix}u & v & 1\end{bmatrix}^{\top} \qquad \text{(pixel and lidar depth to 3D)}$$

$$\mathbf{e} = \mathbf{p}\_{\text{item}} - \mathbf{p}\_{\text{finger}} \qquad \text{(where the hand must go)}$$

$$k = \arg\max\_{i}\,\lvert e\_i\rvert \quad \text{if } \lvert e\_k\rvert > 3\,\text{cm} \qquad \text{(one word: right, down, forward)}$$

$$\lVert \mathbf{e} \rVert < 5\,\text{cm for } 0.3\,\text{s} \;\Rightarrow\; \text{grab}$$

## Challenges we ran into
- Getting the iPhone's lidar stream onto an ARM box (the GX10) took a custom build.
- Lidar reads nothing under about 20 cm, so a hand near the chest has no depth; we estimate it from the hand's size in the image.
- A forearm in front of the item can lend it the wrong depth, so we keep the item's place in the room instead of trusting every frame.
- Making guidance usable: tones confused testers, so we switched to single spoken words.

## Accomplishments that we're proud of
- A full loop from voice request to grab, running locally on one box.
- Lidar depth was 99% accurate from 45 cm to 2 m (1% average error), against 52% for a camera alone (48% average error).
- It works offline: we unplug the internet mid-demo.
- The first four people from outside our team to try it all grabbed the item with no issues.

## What we learned
Cues have to be easy to read, not just present. Vibration and other cues made people misjudge where the item was, or overloaded them with too much to decode at once: in published studies, 1 in 10 wrist buzzes is felt in the wrong spot, and 74% of overlapping buzzes are misread. A single plain word in the ear needs no decoding.

Depth precision decides whether hand guidance works. In our test a camera alone misjudged depth by 48% on average, and by 64% at 2 m; lidar stayed within 1%. A camera's depth errors are different for the hand and the item, so the guidance is poor.

## What's next for Proprio
A memory that reaches beyond one room. Proprio would learn the places you go most, the items you use most, and where each was last seen, so when you forget where you left your keys it already knows, even if they're in another room, and guides you there.

Alongside that: trials with blind and low-vision testers, glass and shiny items, and making it into a more compact device.
