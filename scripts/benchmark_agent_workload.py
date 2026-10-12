#!/usr/bin/env python3
"""Natural-eviction Agent workload for Native/Eager/Lazy/GeneaKV comparison."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[idx]


def policy(
    value_class: str,
    session_id: str,
    event_seq: int,
    protected_prefix_tokens: int | None,
) -> dict[str, Any]:
    return {
        "geneakv": {
            "schema_version": 1,
            "session_id": session_id,
            "node_id": "shared-root",
            "event_seq": event_seq,
            "epoch": 1,
            "value_class": value_class,
            "protected_prefix_tokens": (
                protected_prefix_tokens if value_class == "HIGH" else 0
            ),
            "reason_codes": [
                "TOOL_RESUME_EXPECTED"
                if value_class == "HIGH"
                else "ONE_SHOT_BRANCH"
            ],
            "ttl_seconds": 600,
        }
    }


TRACKED_METRICS = (
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:external_prefix_cache_queries_total",
    "vllm:external_prefix_cache_hits_total",
    "vllm:prompt_tokens_by_source_total",
)


def make_prompt(kind: str, index: int, repeat: int, run_id: str) -> str:
    # Put uniqueness in the first block so different sessions cannot
    # accidentally share a long APC prefix.
    header = (
        f"GeneaKV benchmark run {run_id}, {kind} session {index:04d}. "
        f"Unique root {run_id}-{kind}-{index}. "
    )
    body = (
        "This is a deterministic long Agent context used to create real KV pressure. "
        * repeat
    )
    return header + body + "Return the single word READY."


def read_metrics(host: str, port: int, timeout: int) -> dict[str, float]:
    url = f"http://{host}:{port}/metrics"
    with urllib.request.urlopen(url, timeout=timeout) as response:
        text = response.read().decode()
    values: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        if not line.startswith(TRACKED_METRICS):
            continue
        metric_key, raw_value = line.rsplit(" ", 1)
        values[metric_key] = float(raw_value)
    return values


def metric_delta(
    before: dict[str, float], after: dict[str, float]
) -> dict[str, float]:
    keys = before.keys() | after.keys()
    return {key: after.get(key, 0.0) - before.get(key, 0.0) for key in sorted(keys)}


def metric_sum(delta: dict[str, float], prefix: str) -> float:
    return sum(value for key, value in delta.items() if key.startswith(prefix))


def send(
    endpoint: str,
    model: str,
    prompt: str,
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
    elapsed = time.perf_counter() - started
    return {
        "elapsed_seconds": elapsed,
        "prompt_tokens": int(payload.get("usage", {}).get("prompt_tokens", 0)),
        "completion_tokens": int(
            payload.get("usage", {}).get("completion_tokens", 0)
        ),
        "text": payload["choices"][0].get("text", ""),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(row["elapsed_seconds"]) for row in rows]
    return {
        "requests": len(rows),
        "prompt_tokens": sum(int(row["prompt_tokens"]) for row in rows),
        "mean_seconds": statistics.fmean(latencies) if latencies else 0.0,
        "p50_seconds": percentile(latencies, 0.50),
        "p95_seconds": percentile(latencies, 0.95),
        "max_seconds": max(latencies, default=0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--mode", choices=("native", "eager", "lazy", "geneakv"), required=True
    )
    parser.add_argument("--model", default="geneakv-qwen3-0.6b")
    parser.add_argument("--high-sessions", type=int, default=8)
    parser.add_argument("--flood-requests", type=int, default=12)
    parser.add_argument("--repeat", type=int, default=400)
    parser.add_argument("--run-id", default="round1")
    parser.add_argument("--store-wait", type=float, default=2.0)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument(
        "--protected-prefix-tokens",
        type=int,
        default=None,
        help=(
            "Optional explicit HIGH prefix cap. Omit it to store every "
            "confirmed, cacheable HIGH block."
        ),
    )
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    endpoint = f"http://{args.host}:{args.port}/v1/completions"
    result: dict[str, Any] = {
        "mode": args.mode,
        "endpoint": endpoint,
        "config": {
            "high_sessions": args.high_sessions,
            "flood_requests": args.flood_requests,
            "repeat": args.repeat,
            "store_wait": args.store_wait,
            "max_tokens": args.max_tokens,
            "protected_prefix_tokens": args.protected_prefix_tokens,
            "run_id": args.run_id,
        },
        "phases": {},
    }

    # Warm up model execution without consuming meaningful KV capacity.
    send(
        endpoint,
        args.model,
        f"Warm up for {args.run_id}. Reply READY.",
        None,
        args.max_tokens,
        args.timeout,
    )
    metrics_before = read_metrics(args.host, args.port, args.timeout)

    high_prompts = [
        make_prompt("high-tool-wait", index, args.repeat, args.run_id)
        for index in range(args.high_sessions)
    ]
    flood_prompts = [
        make_prompt("low-one-shot", index, args.repeat, args.run_id)
        for index in range(args.flood_requests)
    ]

    build_rows = []
    for index, prompt in enumerate(high_prompts):
        kv_policy = (
            policy(
                "HIGH", f"high-{index}", 1, args.protected_prefix_tokens
            )
            if args.mode == "geneakv"
            else None
        )
        build_rows.append(
            send(endpoint, args.model, prompt, kv_policy, args.max_tokens, args.timeout)
        )

    time.sleep(args.store_wait)

    flood_rows = []
    for index, prompt in enumerate(flood_prompts):
        kv_policy = (
            policy("LOW", f"low-{index}", 1, args.protected_prefix_tokens)
            if args.mode == "geneakv"
            else None
        )
        flood_rows.append(
            send(endpoint, args.model, prompt, kv_policy, args.max_tokens, args.timeout)
        )

    resume_rows = []
    for index, prompt in enumerate(high_prompts):
        kv_policy = (
            policy(
                "HIGH", f"high-{index}", 2, args.protected_prefix_tokens
            )
            if args.mode == "geneakv"
            else None
        )
        resume_rows.append(
            send(endpoint, args.model, prompt, kv_policy, args.max_tokens, args.timeout)
        )

    result["phases"] = {
        "build_high": {"summary": summarize(build_rows), "requests": build_rows},
        "flood_low": {"summary": summarize(flood_rows), "requests": flood_rows},
        "resume_high": {"summary": summarize(resume_rows), "requests": resume_rows},
    }
    metrics_after = read_metrics(args.host, args.port, args.timeout)
    delta = metric_delta(metrics_before, metrics_after)
    local_queries = metric_sum(delta, "vllm:prefix_cache_queries_total")
    local_hits = metric_sum(delta, "vllm:prefix_cache_hits_total")
    external_queries = metric_sum(delta, "vllm:external_prefix_cache_queries_total")
    external_hits = metric_sum(delta, "vllm:external_prefix_cache_hits_total")
    result["cache_metrics"] = {
        "delta": delta,
        "local_queries": local_queries,
        "local_hits": local_hits,
        "external_queries": external_queries,
        "external_hits": external_hits,
        "local_hit_rate": local_hits / local_queries if local_queries else 0.0,
        "external_hit_rate": (
            external_hits / external_queries if external_queries else 0.0
        ),
        "effective_hit_rate": (
            (local_hits + external_hits) / local_queries if local_queries else 0.0
        ),
    }
    result["derived"] = {
        "resume_vs_build_mean_ratio": (
            result["phases"]["resume_high"]["summary"]["mean_seconds"]
            / result["phases"]["build_high"]["summary"]["mean_seconds"]
        ),
        "all_outputs_nonempty": all(
            row["text"] for phase in (build_rows, flood_rows, resume_rows) for row in phase
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "phases": {
                    key: value["summary"] for key, value in result["phases"].items()
                },
                "cache_metrics": result["cache_metrics"],
            },
            indent=2,
        )
    )
    print(f"result={args.output}")


if __name__ == "__main__":
    main()
