#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${GENEAKV_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/models/Qwen3-0.6B}"
DEVICE_ID="${DEVICE_ID:-0}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.10}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"

export ASCEND_RT_VISIBLE_DEVICES="${DEVICE_ID}"
export VLLM_USE_V1=1
export PYTHONUNBUFFERED=1

exec vllm serve "${MODEL_DIR}" \
  --served-model-name geneakv-qwen3-0.6b \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --tensor-parallel-size 1 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-num-seqs 8 \
  --enable-prefix-caching \
  --enforce-eager
