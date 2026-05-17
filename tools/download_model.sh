#!/bin/bash
# Download Voxtral Realtime 4B GGUF model from HuggingFace
#
# Usage: ./download_model.sh [QUANT] [--dir DIR]
#   QUANT       Quantization precision (default: Q4_0)
#   --dir DIR   Download to DIR (default: models/voxtral)

set -e

MODEL_ID="andrijdavid/Voxtral-Mini-4B-Realtime-2602-GGUF"
MODEL_DIR="models/voxtral"
QUANT="Q4_0"

# Simple argument parsing
while [[ $# -gt 0 ]]; do
    case $1 in
        --dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        *)
            if [[ $1 != --* ]]; then
                QUANT="$1"
                shift
            else
                echo "Unknown option: $1"
                exit 1
            fi
            ;;
    esac
done

# Standardize quant naming (if user passes q4_0 instead of Q4_0)
QUANT=$(echo "$QUANT" | tr '[:lower:]' '[:upper:]')

FILE_NAME="${QUANT}.gguf"
echo "Downloading Voxtral Realtime 4B GGUF to ${MODEL_DIR}/"
echo "Model: ${MODEL_ID}"
echo "Quantization: ${QUANT}"
echo ""

mkdir -p "${MODEL_DIR}"

BASE_URL="https://huggingface.co/${MODEL_ID}/resolve/main"
DEST="${MODEL_DIR}/${FILE_NAME}"

if [ -f "${DEST}" ]; then
    echo "  [skip] ${FILE_NAME} (already exists)"
else
    echo "  [download] ${FILE_NAME}..."
    # Check if the file exists on HF first.
    # HF returns 302 (Redirect) for existing files when using HEAD requests (-I).
    HTTP_CODE=$(curl -s -o /dev/null -I -L -w "%{http_code}" "${BASE_URL}/${FILE_NAME}")
    if [[ "$HTTP_CODE" -ne 200 && "$HTTP_CODE" -ne 302 ]]; then
        echo "Error: Quantization '${QUANT}' not found in repository (HTTP ${HTTP_CODE})."
        echo "Available quants include: Q2_K, Q4_0, Q4_1, Q5_0, Q5_1"
        exit 1
    fi
    
    curl -L -o "${DEST}" "${BASE_URL}/${FILE_NAME}" --progress-bar
    echo "  [done] ${FILE_NAME}"
fi

echo ""
echo "Download complete. Model file: ${DEST}"
ls -lh "${DEST}"