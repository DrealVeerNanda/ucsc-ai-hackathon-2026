#!/usr/bin/env bash
# Run the real pipeline with the live view on the GX10's own screen: the camera with the box,
# fingertip and distance on the left; state, distance, which way to move and timings on the right.
#
#   bash tools/live.sh mug        # look for a mug
#   bash tools/live.sh            # type the target in this terminal; press t in the window to change it
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
    args=(--voice real)
    export VOICE_MODE="${VOICE_MODE:-type}"
fi

# The window opens tiny: give it its full size on the left monitor once it exists.
if command -v xdotool >/dev/null; then
    (for _ in $(seq 1 60); do
        w=$(xdotool search --name '^blindfold-pickup$' 2>/dev/null | tail -1 || true)
        if [ -n "$w" ]; then xdotool windowsize "$w" 1275 900 windowmove "$w" 20 40; break; fi
        sleep 0.5
    done) &
fi
exec python main.py --stream real --perception real --guidance real "${args[@]}" --ui both --no-open
