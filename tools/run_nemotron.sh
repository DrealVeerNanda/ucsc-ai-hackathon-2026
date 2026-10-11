#!/usr/bin/env bash
# Serve Nemotron on the GX10 with llama.cpp, for turning what the wearer says into an object name
# ("I'm thirsty" -> "water bottle"). OpenAI-compatible, local only: http://127.0.0.1:8000/v1
#
#   bash tools/run_nemotron.sh          # leave it running; voice.py finds it by itself
#
# Text only (the vision/audio projector is not loaded). About 23 GB of GPU memory, 16 s to load,
# 0.3 s per request.
set -euo pipefail
MODEL="${NEMOTRON_MODEL:-$HOME/unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-Q4_K_XL.gguf}"
SERVER="${LLAMA_SERVER:-$HOME/llama.cpp/llama-server}"
[ -f "$MODEL" ] || { echo "missing $MODEL" >&2; exit 1; }
exec "$SERVER" --model "$MODEL" --alias nemotron --host 127.0.0.1 --port "${NEMOTRON_PORT:-8000}" -ngl 99 -c 4096 --jinja
