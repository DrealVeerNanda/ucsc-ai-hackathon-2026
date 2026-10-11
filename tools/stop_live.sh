#!/usr/bin/env bash
# Stop the live run and wait until it has finished saving. One Ctrl-C is sent, never a second:
# a second one lands in the middle of writing the recording.
#
#   bash tools/stop_live.sh          # waits up to two minutes
set -uo pipefail
cd "$(dirname "$0")/.."
# Started from the Proprio icon? Then its keeper would start the pipeline again a second later:
# ask the keeper to stop it instead.
if [ -x .venv/bin/python ] && .venv/bin/python launcher.py status >/dev/null 2>&1; then
    exec .venv/bin/python launcher.py stop
fi
pid=$(pgrep -f "[m]ain.py --stream" | head -1)
[ -z "$pid" ] && { echo "no live run"; exit 0; }
kill -INT "$pid"
for _ in $(seq 1 120); do
    kill -0 "$pid" 2>/dev/null || { echo "live run stopped"; tmux kill-session -t live 2>/dev/null; exit 0; }
    sleep 1
done
echo "still running after two minutes (pid $pid): look before killing it harder" >&2
exit 1
