#!/usr/bin/env bash
set -euo pipefail

# Sequential six-run scouting matrix for the deterministic interleaved replay.
# A fresh vLLM process is used for every run, so HBM and CPU KV start cold.

ROOT_DIR="${GENEAKV_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
TRACE_DIR="${TRACE_DIR:-${ROOT_DIR}/datasets/kv-cache-tester-subsets/trace-20}"
MANIFEST="${MANIFEST:-${ROOT_DIR}/datasets/interleaved-20x20-64k.json}"
MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/models/Qwen3-8B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-geneakv-qwen3-8b}"
RUN_ID="${RUN_ID:-interleaved-20x20-$(date +%Y%m%d-%H%M%S)}"
RESULT_DIR="${RESULT_DIR:-${ROOT_DIR}/results/${RUN_ID}}"
DEVICE_ID="${DEVICE_ID:-0}"
PORT="${PORT:-8000}"
CAPACITIES_GIB="${CAPACITIES_GIB:-8 16 24}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.65}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
NUM_GPU_BLOCKS_OVERRIDE="${NUM_GPU_BLOCKS_OVERRIDE:-1024}"
HF_OVERRIDES="${HF_OVERRIDES:-}"
if [[ -z "${HF_OVERRIDES}" ]]; then
  # Base64 avoids nested shell-quote corruption when this script is launched
  # through SSH or another wrapper.
  HF_OVERRIDES="$(printf '%s' 'eyJyb3BlX3NjYWxpbmciOnsicm9wZV90eXBlIjoieWFybiIsImZhY3RvciI6NC4wLCJvcmlnaW5hbF9tYXhfcG9zaXRpb25fZW1iZWRkaW5ncyI6MzI3Njh9fQ==' | base64 -d)"
fi
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-1200}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "missing manifest: ${MANIFEST}" >&2
  exit 2
fi
if [[ ! -d "${TRACE_DIR}" ]]; then
  echo "missing trace directory: ${TRACE_DIR}" >&2
  exit 2
fi
if ! vllm serve --help=num-gpu-blocks-override 2>&1 \
  | grep -q -- "--num-gpu-blocks-override"; then
  echo "this vLLM CLI does not expose --num-gpu-blocks-override" >&2
  exit 2
fi
if [[ -e "${RESULT_DIR}" ]]; then
  echo "refusing to overwrite existing result directory: ${RESULT_DIR}" >&2
  exit 2
fi
mkdir -p "${RESULT_DIR}/logs"

PYTHONPATH="${ROOT_DIR}" python3 "${ROOT_DIR}/scripts/benchmark_interleaved_trace.py" \
  --manifest "${MANIFEST}" \
  --trace-dir "${TRACE_DIR}" \
  --mode geneakv-r0 \
  --dry-run >"${RESULT_DIR}/manifest-dry-run.json"

active_pid=""
cleanup_server() {
  if [[ -n "${active_pid}" ]] && kill -0 "${active_pid}" 2>/dev/null; then
    kill -TERM "${active_pid}" 2>/dev/null || true
    wait "${active_pid}" 2>/dev/null || true
  fi
  active_pid=""
}
trap cleanup_server EXIT INT TERM

wait_for_health() {
  local deadline=$((SECONDS + SERVER_START_TIMEOUT))
  while (( SECONDS < deadline )); do
    if curl --fail --silent --max-time 3 "http://127.0.0.1:${PORT}/health" >/dev/null; then
      return 0
    fi
    if [[ -n "${active_pid}" ]] && ! kill -0 "${active_pid}" 2>/dev/null; then
      return 1
    fi
    sleep 5
  done
  return 1
}

run_one() {
  local mode="$1"
  local capacity_gib="$2"
  local cpu_bytes=$((capacity_gib * 1024 * 1024 * 1024))
  local tag="${mode}-cpu${capacity_gib}g"
  local server_log="${RESULT_DIR}/logs/${tag}-server.log"
  local client_log="${RESULT_DIR}/logs/${tag}-client.log"
  local output="${RESULT_DIR}/${tag}.json"
  local launcher

  if [[ "${mode}" == "eager" ]]; then
    launcher="${ROOT_DIR}/scripts/run_offload.sh"
  else
    launcher="${ROOT_DIR}/scripts/run_geneakv.sh"
  fi

  echo "[START] ${tag} device=${DEVICE_ID} gpu_blocks=${NUM_GPU_BLOCKS_OVERRIDE}"
  env \
    GENEAKV_ROOT="${ROOT_DIR}" \
    MODEL_DIR="${MODEL_DIR}" \
    SERVED_MODEL_NAME="${SERVED_MODEL_NAME}" \
    DEVICE_ID="${DEVICE_ID}" \
    PORT="${PORT}" \
    CPU_BYTES="${cpu_bytes}" \
    LAZY_OFFLOAD=false \
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
    MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
    MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
    NUM_GPU_BLOCKS_OVERRIDE="${NUM_GPU_BLOCKS_OVERRIDE}" \
    HF_OVERRIDES="${HF_OVERRIDES}" \
    bash "${launcher}" >"${server_log}" 2>&1 &
  active_pid=$!

  if ! wait_for_health; then
    echo "server failed to become healthy; inspect ${server_log}" >&2
    cleanup_server
    return 1
  fi

  PYTHONPATH="${ROOT_DIR}" python3 "${ROOT_DIR}/scripts/benchmark_interleaved_trace.py" \
    --manifest "${MANIFEST}" \
    --trace-dir "${TRACE_DIR}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --mode "${mode}" \
    --model "${SERVED_MODEL_NAME}" \
    --max-tokens 2 \
    --vllm-block-size 128 \
    --output "${output}" 2>&1 | tee "${client_log}"

  cleanup_server
  echo "[DONE] ${tag}"
}

python3 - "${MANIFEST}" "${RESULT_DIR}/experiment-config.json" \
  "${CAPACITIES_GIB}" "${DEVICE_ID}" "${NUM_GPU_BLOCKS_OVERRIDE}" <<'PY'
import json, os, pathlib, sys
manifest_path = pathlib.Path(sys.argv[1])
output_path = pathlib.Path(sys.argv[2])
manifest = json.loads(manifest_path.read_text())
payload = {
    "manifest": str(manifest_path),
    "manifest_summary": manifest["summary"],
    "capacities_gib": sys.argv[3].split(),
    "device_id": sys.argv[4],
    "num_gpu_blocks_override": sys.argv[5],
    "execution": "sequential; fresh vLLM process for every policy/capacity run",
}
output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
PY

for capacity_gib in ${CAPACITIES_GIB}; do
  run_one eager "${capacity_gib}"
  run_one geneakv-r0 "${capacity_gib}"
done

python3 "${ROOT_DIR}/scripts/summarize_interleaved_matrix.py" "${RESULT_DIR}"

trap - EXIT INT TERM
echo "matrix complete: ${RESULT_DIR}"
