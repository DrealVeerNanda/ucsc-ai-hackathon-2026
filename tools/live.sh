#!/usr/bin/env bash
# Run the real pipeline with the live view on the GX10's own screen: the camera with the box,
# fingertip and distance on the left; state, distance, which way to move and timings on the right.
#
#   bash tools/live.sh mug        # look for a mug
#   bash tools/live.sh            # ask by voice or by typing in this terminal; a new request any time
#                                 # replaces the target. With no microphone on the GX10 it prints the
#                                 # address of a page that turns a phone or laptop into the microphone.
#
#   UI=web bash tools/live.sh mug # fastest: no window, only the browser dashboard (it opens in
#                                 # Firefox on the GX10's screen). The window costs about 9 ms a frame.
#
# Works at the GX10's terminal or over SSH (the window still opens on the GX10's screen).
# Sound follows GUIDANCE_AUDIO (auto, on, off). The web dashboard is at http://GX10_IP:8642 too.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export DISPLAY="${DISPLAY:-:1}" XAUTHORITY="${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}"

if [ $# -ge 1 ]; then
    args=(--target "$1")
else
    args=(--voice real)     # asks by voice: a microphone here, else the microphone page, else typed
fi

# The window opens tiny: give it its full size on the left monitor once it exists.
if command -v xdotool >/dev/null; then
    (for _ in $(seq 1 60); do
        w=$(xdotool search --name '^blindfold-pickup$' 2>/dev/null | tail -1 || true)
        if [ -n "$w" ]; then xdotool windowsize "$w" 1275 900 windowmove "$w" 20 40; break; fi
        sleep 0.5
    done) &
fi
UI="${UI:-both}"
if [ "$UI" = web ] && command -v firefox >/dev/null; then
    export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=/run/user/$(id -u)/bus}"
    (sleep 6; firefox --new-window http://localhost:8642/ >/dev/null 2>&1 || true) &
fi
exec python main.py --stream real --perception real --guidance real "${args[@]}" --ui "$UI" --no-open
