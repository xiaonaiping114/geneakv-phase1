#!/usr/bin/env python3
"""End-to-end A/B: HIGH survives HBM reset via CPU; LOW recomputes."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig


ROOT = os.environ.get("GENEAKV_ROOT", str(Path(__file__).resolve().parents[1]))
MODEL = os.environ.get("MODEL_DIR", f"{ROOT}/models/Qwen3-0.6B")
CPU_BYTES = int(os.environ.get("CPU_BYTES", str(4 * 1024**3)))
DECISIONS = os.environ.get(
    "GENEAKV_DECISIONS", f"{ROOT}/results/lineage-policy-registry-decisions.jsonl"
)


def fallback_policy(value_class: str, node_id: str, prefix_tokens: int) -> dict:
    return {
        "kv_transfer_params": {
            "geneakv": {
                "schema_version": 1,
                "session_id": "selective-ab",
                "node_id": node_id,
                "event_seq": 1,
                "epoch": 1,
                "value_class": value_class,
                "protected_prefix_tokens": prefix_tokens,
                "reason_codes": [
                    "TOOL_RESUME_EXPECTED"
                    if value_class == "HIGH"
                    else "ONE_SHOT_BRANCH"
                ],
                "ttl_seconds": 300,
            }
        }
    }


def load_trace_policies() -> tuple[dict, dict]:
    high = low = None
    if os.path.exists(DECISIONS):
        with open(DECISIONS, encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                params = record["kv_transfer_params"]
                value_class = params["geneakv"]["value_class"]
                if value_class == "HIGH" and high is None:
                    high = params
                elif value_class == "LOW" and low is None:
                    low = params
    return (
        high or fallback_policy("HIGH", "high-node", 8192)["kv_transfer_params"],
        low or fallback_policy("LOW", "low-node", 0)["kv_transfer_params"],
    )


def run(llm: LLM, prompt: str, kv_transfer_params: dict):
    params = SamplingParams(
        max_tokens=8,
        temperature=0,
        seed=0,
        extra_args={"kv_transfer_params": kv_transfer_params},
    )
    start = time.perf_counter()
    output = llm.generate(prompt, params, use_tqdm=False)[0]
    return time.perf_counter() - start, output.outputs[0].token_ids


def reset_hbm(llm: LLM) -> None:
    time.sleep(2)
    if not llm.reset_prefix_cache():
        raise RuntimeError("HBM cache reset failed while transfers were pending")


def main() -> None:
    connector = KVTransferConfig(
        kv_connector="GeneaKVConnector",
        kv_role="kv_both",
        kv_connector_module_path="geneakv.connector",
        kv_connector_extra_config={
            "cpu_bytes_to_use": CPU_BYTES,
            "lazy_offload": False,
        },
    )
    llm = LLM(
        model=MODEL,
        max_model_len=16384,
        gpu_memory_utilization=0.10,
        enable_prefix_caching=True,
        kv_transfer_config=connector,
        enforce_eager=True,
    )

    high_prompt = ("High value tool resume shared ancestor. " * 700) + "Answer READY."
    low_prompt = ("Low value one shot abandoned branch. " * 700) + "Answer READY."
    high_policy, low_policy = load_trace_policies()

    high_first_s, high_first_tokens = run(llm, high_prompt, high_policy)
    reset_hbm(llm)
    high_resume_s, high_resume_tokens = run(llm, high_prompt, high_policy)

    low_first_s, low_first_tokens = run(llm, low_prompt, low_policy)
    reset_hbm(llm)
    low_again_s, low_again_tokens = run(llm, low_prompt, low_policy)

    result = {
        "policies_source": DECISIONS,
        "high_policy": high_policy["geneakv"],
        "low_policy": low_policy["geneakv"],
        "high": {
            "first_seconds": round(high_first_s, 6),
            "after_hbm_reset_seconds": round(high_resume_s, 6),
            "tokens_equal": high_first_tokens == high_resume_tokens,
            "expected_path": "CPU_RESTORE",
        },
        "low": {
            "first_seconds": round(low_first_s, 6),
            "after_hbm_reset_seconds": round(low_again_s, 6),
            "tokens_equal": low_first_tokens == low_again_tokens,
            "expected_path": "RECOMPUTE",
        },
    }
    print("GENEAKV_RESULT=" + json.dumps(result, ensure_ascii=False))
    if not result["high"]["tokens_equal"] or not result["low"]["tokens_equal"]:
        raise AssertionError("Output token mismatch in selective admission A/B")


if __name__ == "__main__":
    main()
