#!/usr/bin/env bash
# Put the models' real logs on the GX10's screen, in one terminal with three parts:
#   top     Nemotron's own server log (requests, tokens, tokens per second)
#   middle  Cosmos's own server log
#   bottom  what each model was asked and what it answered (from the pipeline's trace)
#
#   bash tools/show_models.sh           # at the GX10, or over SSH (it still opens on the GX10's screen)
#
# Needs the servers started by tools/run_nemotron.sh and tools/run_cosmos.sh with their output
# in ~/models/nemotron.log and ~/models/cosmos_*.log.
set -euo pipefail
cd "$(dirname "$0")/.."
export DISPLAY="${DISPLAY:-:1}" XAUTHORITY="${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}"
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=/run/user/$(id -u)/bus}"

NEMO="${NEMOTRON_LOG:-$HOME/models/nemotron.log}"
COSMOS="${COSMOS_LOG:-$(ls -t "$HOME"/models/cosmos_*.log 2>/dev/null | head -1)}"
# The lines of a llama.cpp server log that show a request arriving and what it cost.
KEEP='POST /|prompt eval time|       eval time|total time|launch_slot|slot release|n_tokens'

tmux kill-session -t models 2>/dev/null || true
tmux new-session -d -s models -x 118 -y 54 "printf '\033[1;32mNEMOTRON server log\033[0m  %s\n' '$NEMO'; tail -n 400 -F '$NEMO' | grep --line-buffered -E '$KEEP'"
tmux split-window -v -t models "printf '\033[1;35mCOSMOS server log\033[0m  %s\n' '$COSMOS'; tail -n 400 -F '$COSMOS' | grep --line-buffered -E '$KEEP'"
tmux split-window -v -t models "source .venv/bin/activate && python tools/model_log.py --all"
tmux select-layout -t models even-vertical
tmux set-option -t models -g mouse on >/dev/null
tmux set-option -t models status off >/dev/null

gnome-terminal --title="Models: real logs" --geometry=118x54+985+30 -- tmux attach -t models
echo "opened on the GX10's screen. Also: tmux attach -t models"
