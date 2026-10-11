#!/usr/bin/env bash
# Run the real pipeline with the dashboard in Firefox on the GX10's own screen: the camera with the
# box, fingertip and distance on the left; state, distance, which way to move and timings on the right.
# The Proprio icon on the desktop (launcher.py) starts the pipeline through this script, so a run
# from the button and a run by hand are the same thing.
#
#   bash tools/live.sh            # ask by voice (earbuds, or the microphone page it prints the address
#                                 # of); a new request any time replaces the target
#   bash tools/live.sh mug        # look for a mug, no voice
#
#   UI=both bash tools/live.sh    # also the OpenCV window. It costs about 9 ms a frame.
#   VIEW=logs bash tools/live.sh  # the dashboard opens showing only the live log, no camera: for
#                                 # testing by ear. The button on the page (or L) switches back.
#   HOST=0.0.0.0 bash tools/live.sh   # let other machines open the dashboard at http://GX10_IP:8642
#   OPEN=0 bash tools/live.sh     # do not open Firefox (the launcher looks after the windows itself)
#
# Works at the GX10's terminal or over SSH (the browser still opens on the GX10's screen).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

# The settings of the run that worked. launcher.py's DEFAULTS is the same list; change both.
: "${UI:=web}"                           # the dashboard in the browser, no OpenCV window
: "${VIEW:=camera}"                      # the dashboard opens on the camera, not on the logs
export GUIDANCE_AUDIO="${GUIDANCE_AUDIO:-on}"    # sound through the sound card or not at all (off: printed)
export GUIDANCE_MODE="${GUIDANCE_MODE:-3}"       # option 3, the combined spoken directions
export VOICE_MODE="${VOICE_MODE:-web}"           # the earbuds (or the microphone page), never "press Enter"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}" HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
: "${HOST:=127.0.0.1}"                   # the dashboard answers this machine only: the hall's network is public
: "${PORT:=8642}"

# A detached start or an SSH session has no desktop around it: say where the screen and the sound server are.
export DISPLAY="${DISPLAY:-:1}" XAUTHORITY="${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=/run/user/$(id -u)/bus}"
# The phone's address on the Wi-Fi, tried when the app is not sending over the cable.
[ -f .phone_ip ] && export STREAM_PHONE_IP="${STREAM_PHONE_IP:-$(cat .phone_ip)}"
# Keep every library off the network, if and only if every model is on disk and has been loaded
# with the network blocked (the launcher's "Models on disk" line does that and remembers it).
eval "$(python launcher_checks.py --offline 2>/dev/null || true)"

if [ $# -ge 1 ]; then
    args=(--target "$1")
else
    args=(--voice real)     # asks by voice: a microphone here, else the microphone page, else typed
fi

# The OpenCV window opens tiny: give it its full size on the left monitor once it exists.
if [ "$UI" != web ] && command -v xdotool >/dev/null; then
    (for _ in $(seq 1 60); do
        w=$(xdotool search --name '^blindfold-pickup$' 2>/dev/null | tail -1 || true)
        if [ -n "$w" ]; then xdotool windowsize "$w" 1275 900 windowmove "$w" 20 40; break; fi
        sleep 0.5
    done) &
fi
if [ "$UI" = web ] && [ "${OPEN:-1}" != 0 ] && command -v firefox >/dev/null; then
    # A dashboard page left open from the last run reconnects by itself: open another only if none is there.
    (sleep 6; xdotool search --name "^Proprio, live" >/dev/null 2>&1 || firefox --new-window "http://localhost:$PORT/#$VIEW" >/dev/null 2>&1 || true) &
fi
exec python main.py --stream real --perception real --guidance real "${args[@]}" --ui "$UI" --no-open --host "$HOST" --port "$PORT"
