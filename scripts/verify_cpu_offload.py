#!/usr/bin/env python3
"""Prove that a prefix is restored from CPU after clearing only the HBM cache."""

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


def main() -> None:
    connector = KVTransferConfig(
        kv_connector="SimpleCPUOffloadConnector",
        kv_role="kv_both",
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
    params = SamplingParams(max_tokens=8, temperature=0, seed=0)
    prompt = ("GeneaKV CPU restore validation stable prefix. " * 700) + "Answer READY."

    t0 = time.perf_counter()
    cold = llm.generate(prompt, params, use_tqdm=False)[0]
    cold_s = time.perf_counter() - t0
    cold_tokens = cold.outputs[0].token_ids

    time.sleep(2)
    reset_ok = llm.reset_prefix_cache()
    if not reset_ok:
        raise RuntimeError("reset_prefix_cache returned False; pending requests or transfers remain")

    t1 = time.perf_counter()
    restored = llm.generate(prompt, params, use_tqdm=False)[0]
    restored_s = time.perf_counter() - t1
    restored_tokens = restored.outputs[0].token_ids

    result = {
        "cold_seconds": round(cold_s, 6),
        "cpu_restore_seconds": round(restored_s, 6),
        "hbm_prefix_cache_reset": reset_ok,
        "tokens_equal": cold_tokens == restored_tokens,
        "cold_token_ids": cold_tokens,
        "restored_token_ids": restored_tokens,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["tokens_equal"]:
        raise AssertionError("CPU-restored output differs from cold output")


if __name__ == "__main__":
    main()
