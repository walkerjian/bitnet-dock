#!/usr/bin/env bash
set -e

CHECKPOINT_DIR="/checkpoints"
MODEL_NAME="bitnet-b1.58-2B-4T-bf16"
BF16_DIR="${CHECKPOINT_DIR}/${MODEL_NAME}"
CONVERTED="${CHECKPOINT_DIR}/model_weights.pt"

# ── Download + convert weights if not done ───────────────────────────────────
if [ ! -f "${CONVERTED}" ]; then
    echo "==> Downloading bf16 checkpoint for GPU inference ..."
    huggingface-cli download "microsoft/${MODEL_NAME}" \
        --local-dir "${BF16_DIR}"

    echo "==> Converting safetensors → checkpoint ..."
    python3 ./convert_safetensors.py \
        --safetensors_file "${BF16_DIR}/model.safetensors" \
        --output "${CHECKPOINT_DIR}/model_state.pt" \
        --model_name 2B

    python3 ./convert_checkpoint.py \
        --input "${CHECKPOINT_DIR}/model_state.pt"

    # keep the final weights, drop the intermediate
    rm -f "${CHECKPOINT_DIR}/model_state.pt"
    echo "==> Conversion done."
else
    echo "==> Checkpoint found, skipping conversion."
fi

# ── Run ───────────────────────────────────────────────────────────────────────
case "${BITNET_MODE}" in

  chat)
    echo "==> Starting GPU interactive chat ..."
    exec python3 ./generate.py "${CHECKPOINT_DIR}" \
        --interactive --chat_format
    ;;

  benchmark)
    echo "==> Running GPU benchmark ..."
    exec python3 ./generate.py "${CHECKPOINT_DIR}" \
        --benchmark
    ;;

  *)
    echo "Unknown BITNET_MODE '${BITNET_MODE}'. Use: chat | benchmark"
    exit 1
    ;;
esac
