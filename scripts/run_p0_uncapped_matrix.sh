#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${GENEAKV_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
ROUNDS="${ROUNDS:-3}"
CAPACITIES_GIB="${CAPACITIES_GIB:-8 16}"
RESULT_DIR="${ROOT_DIR}/results/p0-uncapped"
LOG_DIR="${ROOT_DIR}/logs/p0-uncapped"
mkdir -p "${RESULT_DIR}" "${LOG_DIR}"

eager_pid=""
geneakv_pid=""

stop_pair() {
  if [[ -n "${eager_pid}" ]] && kill -0 "${eager_pid}" 2>/dev/null; then
    kill -TERM "${eager_pid}"
    wait "${eager_pid}" 2>/dev/null || true
  fi
  if [[ -n "${geneakv_pid}" ]] && kill -0 "${geneakv_pid}" 2>/dev/null; then
    kill -TERM "${geneakv_pid}"
    wait "${geneakv_pid}" 2>/dev/null || true
  fi
  eager_pid=""
  geneakv_pid=""
}

trap stop_pair EXIT INT TERM

wait_ready() {
  local port="$1"
  local log_file="$2"
  for _ in $(seq 1 300); do
    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  tail -n 120 "${log_file}" || true
  return 1
}

for capacity_gib in ${CAPACITIES_GIB}; do
  cpu_bytes=$((capacity_gib * 1024 * 1024 * 1024))
  for round in $(seq 1 "${ROUNDS}"); do
    if (( round == ROUNDS )); then
      eager_device=1
      geneakv_device=0
    else
      eager_device=0
      geneakv_device=1
    fi

    tag="cap${capacity_gib}-r${round}"
    eager_log="${LOG_DIR}/eager-${tag}.log"
    geneakv_log="${LOG_DIR}/geneakv-${tag}.log"

    env DEVICE_ID="${eager_device}" PORT=8000 CPU_BYTES="${cpu_bytes}" \
      bash "${ROOT_DIR}/scripts/run_offload.sh" >"${eager_log}" 2>&1 &
    eager_pid=$!

    env DEVICE_ID="${geneakv_device}" PORT=8001 CPU_BYTES="${cpu_bytes}" \
      bash "${ROOT_DIR}/scripts/run_geneakv.sh" >"${geneakv_log}" 2>&1 &
    geneakv_pid=$!

    wait_ready 8000 "${eager_log}"
    wait_ready 8001 "${geneakv_log}"

    python "${ROOT_DIR}/scripts/benchmark_agent_workload.py" \
      --port 8000 \
      --mode eager \
      --max-tokens 2 \
      --run-id "p0-${tag}" \
      --output "${RESULT_DIR}/eager-${tag}.json"

    python "${ROOT_DIR}/scripts/benchmark_agent_workload.py" \
      --port 8001 \
      --mode geneakv \
      --max-tokens 2 \
      --run-id "p0-${tag}" \
      --output "${RESULT_DIR}/geneakv-${tag}.json"

    stop_pair
    sleep 5
  done
done

python - "${RESULT_DIR}" <<'PY'
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*.json")):
    data = json.loads(path.read_text())
    metrics = data["cache_metrics"]
    resume = data["phases"]["resume_high"]["summary"]
    rows.append({
        "file": path.name,
        "mode": data["mode"],
        "capacity_gib": int(path.stem.split("cap", 1)[1].split("-", 1)[0]),
        "external_hits": metrics["external_hits"],
        "effective_hit_rate": metrics["effective_hit_rate"],
        "resume_mean_ms": resume["mean_seconds"] * 1000,
        "resume_p95_ms": resume["p95_seconds"] * 1000,
    })

groups = {}
for row in rows:
    groups.setdefault((row["mode"], row["capacity_gib"]), []).append(row)

summary = {"runs": rows, "groups": []}
for (mode, capacity), items in sorted(groups.items()):
    summary["groups"].append({
        "mode": mode,
        "capacity_gib": capacity,
        "rounds": len(items),
        "external_hits_mean": statistics.fmean(x["external_hits"] for x in items),
        "effective_hit_rate_mean": statistics.fmean(
            x["effective_hit_rate"] for x in items
        ),
        "resume_mean_ms": statistics.fmean(x["resume_mean_ms"] for x in items),
        "resume_p95_ms": statistics.fmean(x["resume_p95_ms"] for x in items),
    })

(root / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
)
print(json.dumps(summary["groups"], ensure_ascii=False, indent=2))
PY
