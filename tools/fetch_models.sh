#!/usr/bin/env bash
# Download model files onto the GX10 so the demo still runs with the network unplugged.
# Safe to re-run: finished files are skipped and interrupted ones resume.
#
#   bash tools/fetch_models.sh          # Cosmos Reason 2: 8B (better boxes) and 2B (faster)
#   bash tools/fetch_models.sh 8b       # only one of them
#   bash tools/fetch_models.sh piper    # the voice that speaks sentences (63 MB), into models/piper here
#
# Files go to ~/models (override with MODELS_DIR). These are third-party GGUF conversions of
# nvidia/Cosmos-Reason2-*, under the NVIDIA Open Model License, for llama.cpp.
set -euo pipefail
DIR="${MODELS_DIR:-$HOME/models}"

get() {  # get HF_REPO FILE
    local out="$DIR/${1#*/}/$2"
    if [ -f "$out" ]; then
        echo "have   $out"
        return
    fi
    mkdir -p "$(dirname "$out")"
    echo "fetch  $out"
    curl -L --fail --retry 30 --retry-delay 5 --retry-all-errors -C - -o "$out.part" \
        "https://huggingface.co/$1/resolve/main/$2"
    mv "$out.part" "$out"
}

for m in "${@:-8b 2b}"; do
    for name in $m; do
        case "$name" in
            8b) get mradermacher/Cosmos-Reason2-8B-GGUF Cosmos-Reason2-8B.Q8_0.gguf
                get mradermacher/Cosmos-Reason2-8B-GGUF Cosmos-Reason2-8B.mmproj-f16.gguf ;;
            2b) get apolo13x/Cosmos-Reason2-2B-GGUF Cosmos-Reason2-2B-Q8_0.gguf
                get apolo13x/Cosmos-Reason2-2B-GGUF mmproj-Cosmos-Reason2-2B-F16.gguf ;;
            piper)
                V="$(cd "$(dirname "$0")/.." && pwd)/models/piper"
                mkdir -p "$V"
                for f in en_US-lessac-medium.onnx en_US-lessac-medium.onnx.json; do
                    [ -s "$V/$f" ] && { echo "have   $V/$f"; continue; }
                    echo "fetch  $V/$f"
                    curl -L --fail --retry 30 --retry-delay 5 -o "$V/$f" \
                        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/$f"
                done ;;
            *)  echo "unknown model: $name (use 8b, 2b or piper)" >&2; exit 1 ;;
        esac
    done
done
du -sh "$DIR"/* 2>/dev/null
