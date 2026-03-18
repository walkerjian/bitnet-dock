#!/usr/bin/env bash
set -e

MODEL_DIR="/models/${BITNET_MODEL}"
GGUF_FILE="${MODEL_DIR}/ggml-model-${BITNET_QUANT}.gguf"

# ── Download model if not present ────────────────────────────────────────────
if [ ! -f "${GGUF_FILE}" ]; then
    echo "==> Model not found, downloading microsoft/${BITNET_MODEL} ..."
    huggingface-cli download "microsoft/${BITNET_MODEL}" \
        --local-dir "${MODEL_DIR}"
    echo "==> Building quantised model (quant=${BITNET_QUANT}) ..."
    python3 setup_env.py -md "${MODEL_DIR}" -q "${BITNET_QUANT}"
else
    echo "==> Model found at ${GGUF_FILE}, skipping download."
fi

# ── Run ───────────────────────────────────────────────────────────────────────
case "${BITNET_MODE}" in

  server)
    echo "==> Starting inference server on :8080 ..."
    exec python3 run_inference_server.py \
        -m "${GGUF_FILE}" \
        -t "${N_THREADS:-8}" \
        -c "${CTX_SIZE:-2048}" \
        --host 0.0.0.0 --port 8080
    ;;

  chat)
    echo "==> Starting interactive chat ..."
    exec python3 run_inference.py \
        -m "${GGUF_FILE}" \
        -p "You are a helpful assistant" \
        -t "${N_THREADS:-8}" \
        -c "${CTX_SIZE:-2048}" \
        -cnv
    ;;

  benchmark)
    echo "==> Running benchmark ..."
    exec python3 utils/e2e_benchmark.py \
        -m "${GGUF_FILE}" \
        -t "${N_THREADS:-8}" \
        -n 200 -p 512
    ;;

  *)
    echo "Unknown BITNET_MODE '${BITNET_MODE}'. Use: server | chat | benchmark"
    exit 1
    ;;
esac
