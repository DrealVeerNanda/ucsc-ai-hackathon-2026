#!/usr/bin/env bash
# Serve Cosmos Reason 2 on the GX10 with llama.cpp. OpenAI-compatible, local only:
#   http://127.0.0.1:8001/v1   (model name: cosmos)
#
#   bash tools/run_cosmos.sh        # 8B
#   bash tools/run_cosmos.sh 2b     # faster, weaker
#
# Port 8001 so it can run next to the models the other run_*.sh scripts put on 8000.
# Get the weights first with: bash tools/fetch_models.sh
set -euo pipefail
DIR="${MODELS_DIR:-$HOME/models}"
SERVER="${LLAMA_SERVER:-$HOME/llama.cpp/llama-server}"

case "${1:-8b}" in
    8b) MODEL="$DIR/Cosmos-Reason2-8B-GGUF/Cosmos-Reason2-8B.Q8_0.gguf"
        MMPROJ="$DIR/Cosmos-Reason2-8B-GGUF/Cosmos-Reason2-8B.mmproj-f16.gguf" ;;
    2b) MODEL="$DIR/Cosmos-Reason2-2B-GGUF/Cosmos-Reason2-2B-Q8_0.gguf"
        MMPROJ="$DIR/Cosmos-Reason2-2B-GGUF/mmproj-Cosmos-Reason2-2B-F16.gguf" ;;
    *)  echo "usage: $0 [8b|2b]" >&2; exit 1 ;;
esac
for f in "$MODEL" "$MMPROJ"; do
    [ -f "$f" ] || { echo "missing $f: run bash tools/fetch_models.sh" >&2; exit 1; }
done

exec "$SERVER" --model "$MODEL" --mmproj "$MMPROJ" --alias cosmos \
    --host 127.0.0.1 --port "${COSMOS_PORT:-8001}" -ngl 99 -c 8192 --jinja
