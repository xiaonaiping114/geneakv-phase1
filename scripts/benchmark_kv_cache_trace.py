#!/usr/bin/env python3
"""Replay a small kv-cache-tester trace against an OpenAI-compatible vLLM.

Source 64-token hash blocks are deterministically represented by fixed-size
token-id segments. This preserves prefix identity without requiring private
prompt text. Low-value interference requests create natural HBM/CPU pressure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any


TRACKED_METRICS = (
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:external_prefix_cache_queries_total",
    "vllm:external_prefix_cache_hits_total",
    "vllm:prompt_tokens_by_source_total",
)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(len(values) - 1, round((len(values) - 1) * q))]


def source_block_tokens(block_hash: int, tokens_per_source_block: int) -> list[int]:
    """Map one anonymized source block to stable, valid ordinary token IDs."""
    output: list[int] = []
    counter = 0
    while len(output) < tokens_per_source_block:
        digest = hashlib.sha256(f"geneakv:{block_hash}:{counter}".encode()).digest()
        for offset in range(0, len(digest), 2):
            value = int.from_bytes(digest[offset : offset + 2], "big")
            output.append(1000 + value % 48000)
            if len(output) == tokens_per_source_block:
                break
        counter += 1
    return output


def trace_prompt(hash_ids: list[int], tokens_per_source_block: int) -> list[int]:
    prompt: list[int] = []
    for block_hash in hash_ids:
        prompt.extend(source_block_tokens(int(block_hash), tokens_per_source_block))
    return prompt


def low_prompt(round_index: int, request_index: int, tokens: int) -> list[int]:
    seed = 700000000 + round_index * 1000 + request_index
    blocks = (tokens + 31) // 32
    prompt: list[int] = []
    for block_index in range(blocks):
        prompt.extend(source_block_tokens(seed + block_index, 32))
    return prompt[:tokens]


def empty_trace_prompt(session_id: str, request_index: int) -> list[int]:
    """Return a unique non-empty API placeholder without creating reuse.

    A few public trace events contain no anonymous ``hash_ids``.  vLLM's
    completions endpoint rejects an empty prompt, so map that event to four
    deterministic tokens scoped to the session and request index.
    """
    digest = hashlib.sha256(
        f"geneakv-empty:{session_id}:{request_index}".encode()
    ).digest()
    seed = int.from_bytes(digest[:8], "big")
    return source_block_tokens(seed, 4)


def policy(
    value_class: str,
    session_id: str,
    node_id: str,
    event_seq: int,
    protected_prefix_tokens: int,
) -> dict[str, Any]:
    return {
        "geneakv": {
            "schema_version": 1,
            "session_id": session_id,
            "node_id": node_id,
            "event_seq": event_seq,
            "epoch": 1,
            "value_class": value_class,
            "protected_prefix_tokens": protected_prefix_tokens,
            "reason_codes": [
                "TOOL_RESUME_EXPECTED"
                if value_class == "HIGH"
                else "ONE_SHOT_BRANCH"
            ],
            "ttl_seconds": 600,
        }
    }


def send(
    endpoint: str,
    model: str,
    prompt: list[int],
    kv_policy: dict[str, Any] | None,
    max_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if kv_policy is not None:
        body["kv_transfer_params"] = kv_policy
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    return {
        "elapsed_seconds": time.perf_counter() - started,
        "prompt_tokens": int(payload.get("usage", {}).get("prompt_tokens", 0)),
        "completion_tokens": int(payload.get("usage", {}).get("completion_tokens", 0)),
        "text": payload["choices"][0].get("text", ""),
    }


def read_metrics(host: str, port: int, timeout: int) -> dict[str, float]:
    with urllib.request.urlopen(
        f"http://{host}:{port}/metrics", timeout=timeout
    ) as response:
        text = response.read().decode()
    values: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#") or not line.startswith(TRACKED_METRICS):
            continue
        metric_key, raw_value = line.rsplit(" ", 1)
        values[metric_key] = float(raw_value)
    return values


def metric_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {
        key: after.get(key, 0.0) - before.get(key, 0.0)
        for key in sorted(before.keys() | after.keys())
    }


def metric_sum(delta: dict[str, float], prefix: str) -> float:
    return sum(value for key, value in delta.items() if key.startswith(prefix))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--mode", choices=("native", "eager", "lazy", "geneakv"), required=True
    )
    parser.add_argument("--model", default="geneakv-qwen3-0.6b")
    parser.add_argument("--requests", type=int, default=5)
    parser.add_argument("--tokens-per-source-block", type=int, default=32)
    parser.add_argument("--low-per-gap", type=int, default=10)
    parser.add_argument("--low-tokens", type=int, default=4096)
    parser.add_argument("--protected-prefix-tokens", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=2)
    parser.add_argument("--max-prompt-tokens", type=int, default=40000)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    trace = json.loads(args.trace.read_text(encoding="utf-8"))
    requests = trace["requests"][: args.requests]
    session_id = str(trace.get("id") or args.trace.stem)
    endpoint = f"http://{args.host}:{args.port}/v1/completions"

    # Short warmup does not materially consume KV capacity.
    send(endpoint, args.model, [1001, 1002, 1003, 1004], None, args.max_tokens, args.timeout)
    metrics_before = read_metrics(args.host, args.port, args.timeout)
    trace_rows: list[dict[str, Any]] = []
    low_rows: list[dict[str, Any]] = []

    previous_hashes: list[int] = []
    for index, source_request in enumerate(requests):
        hashes = [int(value) for value in source_request.get("hash_ids", [])]
        prompt = trace_prompt(hashes, args.tokens_per_source_block)
        source_empty_prompt = not prompt
        if source_empty_prompt:
            prompt = empty_trace_prompt(session_id, index)
        if len(prompt) > args.max_prompt_tokens:
            raise ValueError(
                f"request {index} maps to {len(prompt)} tokens, exceeding "
                f"--max-prompt-tokens={args.max_prompt_tokens}; lower "
                "--tokens-per-source-block or --requests"
            )
        lcp = 0
        for left, right in zip(previous_hashes, hashes):
            if left != right:
                break
            lcp += 1
        value_class = "HIGH" if source_request.get("stop") == "tool_use" else "NORMAL"
        kv_policy = None
        if args.mode == "geneakv":
            kv_policy = policy(
                value_class,
                session_id,
                f"request-{index:04d}",
                index + 1,
                args.protected_prefix_tokens if value_class == "HIGH" else 0,
            )
        row = send(
            endpoint,
            args.model,
            prompt,
            kv_policy,
            args.max_tokens,
            args.timeout,
        )
        row.update(
            {
                "source_request_index": index,
                "source_input_tokens": source_request.get("in"),
                "source_hash_blocks": len(hashes),
                "source_parent_lcp_blocks": lcp,
                "source_stop": source_request.get("stop"),
                "source_input_types": source_request.get("input_types", []),
                "source_empty_prompt": source_empty_prompt,
                "online_value_class": value_class,
            }
        )
        trace_rows.append(row)
        previous_hashes = hashes

        if index + 1 == len(requests):
            continue
        for low_index in range(args.low_per_gap):
            low_policy = None
            if args.mode == "geneakv":
                low_policy = policy(
                    "LOW",
                    f"interference-{index}-{low_index}",
                    "one-shot",
                    1,
                    0,
                )
            low_rows.append(
                send(
                    endpoint,
                    args.model,
                    low_prompt(index, low_index, args.low_tokens),
                    low_policy,
                    args.max_tokens,
                    args.timeout,
                )
            )

    metrics_after = read_metrics(args.host, args.port, args.timeout)
    delta = metric_delta(metrics_before, metrics_after)
    local_queries = metric_sum(delta, "vllm:prefix_cache_queries_total")
    local_hits = metric_sum(delta, "vllm:prefix_cache_hits_total")
    external_hits = metric_sum(delta, "vllm:external_prefix_cache_hits_total")
    resume_latencies = [row["elapsed_seconds"] for row in trace_rows[1:]]
    result = {
        "source": "callanjfox/kv-cache-tester",
        "trace": str(args.trace),
        "trace_id": session_id,
        "mode": args.mode,
        "config": {
            "requests": len(requests),
            "tokens_per_source_block": args.tokens_per_source_block,
            "low_per_gap": args.low_per_gap,
            "low_tokens": args.low_tokens,
            "protected_prefix_tokens": args.protected_prefix_tokens,
            "max_tokens": args.max_tokens,
            "max_prompt_tokens": args.max_prompt_tokens,
        },
        "trace_requests": trace_rows,
        "low_requests": len(low_rows),
        "summary": {
            "resume_requests": len(resume_latencies),
            "resume_mean_seconds": statistics.fmean(resume_latencies),
            "resume_p50_seconds": percentile(resume_latencies, 0.50),
            "resume_p95_seconds": percentile(resume_latencies, 0.95),
            "low_mean_seconds": statistics.fmean(
                row["elapsed_seconds"] for row in low_rows
            ),
        },
        "cache_metrics": {
            "delta": delta,
            "local_queries": local_queries,
            "local_hits": local_hits,
            "external_hits": external_hits,
            "effective_hit_rate": (
                (local_hits + external_hits) / local_queries if local_queries else 0.0
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "summary": result["summary"],
                "cache_metrics": result["cache_metrics"],
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
