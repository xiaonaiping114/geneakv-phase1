#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${GENEAKV_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/models/Qwen3-0.6B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-geneakv-qwen3-0.6b}"
DEVICE_ID="${DEVICE_ID:-1}"
PORT="${PORT:-8001}"
CPU_BYTES="${CPU_BYTES:-4294967296}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.10}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
HF_OVERRIDES="${HF_OVERRIDES:-}"
NUM_GPU_BLOCKS_OVERRIDE="${NUM_GPU_BLOCKS_OVERRIDE:-}"

if [[ -n "${ASCEND_TOOLKIT_ENV:-}" ]]; then
  # shellcheck disable=SC1090
  source "${ASCEND_TOOLKIT_ENV}"
fi
export ASCEND_RT_VISIBLE_DEVICES="${DEVICE_ID}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

KV_CONFIG=$(printf '{"kv_connector":"GeneaKVConnector","kv_role":"kv_both","kv_connector_module_path":"geneakv.connector","kv_connector_extra_config":{"cpu_bytes_to_use":%s,"lazy_offload":false}}' "${CPU_BYTES}")

EXTRA_ARGS=()
if [[ -n "${HF_OVERRIDES}" ]]; then
  EXTRA_ARGS+=(--hf-overrides "${HF_OVERRIDES}")
fi
if [[ -n "${NUM_GPU_BLOCKS_OVERRIDE}" ]]; then
  EXTRA_ARGS+=(--num-gpu-blocks-override "${NUM_GPU_BLOCKS_OVERRIDE}")
fi

exec vllm serve "${MODEL_DIR}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --tensor-parallel-size 1 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --enable-prefix-caching \
  --enforce-eager \
  --kv-transfer-config "${KV_CONFIG}" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
