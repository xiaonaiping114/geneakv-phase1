#!/usr/bin/env python3
"""Replay a fixed real-trace manifest with deterministic session interleaving.

The scheduler uses round-robin barrier batches.  A batch contains at most
``max_concurrency`` different sessions and completes before the next batch is
submitted.  This preserves source order within every session, creates genuine
cross-session cache pressure, and keeps the logical submission stream identical
across Eager and GeneaKV-R0 runs.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

try:
    from scripts.interleaved_replay import (
        MANIFEST_SCHEMA_VERSION,
        load_and_validate_sources,
        reconstruct_prompt,
    )
except ModuleNotFoundError:  # Direct execution: python scripts/benchmark_*.py
    from interleaved_replay import (
        MANIFEST_SCHEMA_VERSION,
        load_and_validate_sources,
        reconstruct_prompt,
    )


TRACKED_METRIC_PREFIXES = (
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:external_prefix_cache_queries_total",
    "vllm:external_prefix_cache_hits_total",
    "vllm:prompt_tokens_by_source_total",
    "geneakv_",
)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * q))]


def policy(
    value_class: str,
    session_id: str,
    node_id: str,
    event_seq: int,
    prompt_tokens: int,
    ttl_seconds: int,
) -> dict[str, Any]:
    if value_class == "HIGH":
        reason_codes = ["TOOL_RESUME_EXPECTED"]
        protected_prefix_tokens = prompt_tokens
    elif value_class == "LOW":
        reason_codes = ["SESSION_END_OR_ONE_SHOT"]
        protected_prefix_tokens = 0
    else:
        reason_codes = ["NO_HIGH_VALUE_SIGNAL"]
        protected_prefix_tokens = 0
    return {
        "geneakv": {
            "schema_version": 1,
            "session_id": session_id,
            "node_id": node_id,
            "event_seq": event_seq,
            "epoch": 1,
            "value_class": value_class,
            "protected_prefix_tokens": protected_prefix_tokens,
            "reason_codes": reason_codes,
            "ttl_seconds": ttl_seconds,
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
    completed = time.perf_counter()
    return {
        "started_monotonic": started,
        "completed_monotonic": completed,
        "elapsed_seconds": completed - started,
        "prompt_tokens_reported": int(
            payload.get("usage", {}).get("prompt_tokens", 0)
        ),
        "completion_tokens": int(
            payload.get("usage", {}).get("completion_tokens", 0)
        ),
    }


def read_metrics(host: str, port: int, timeout: int) -> dict[str, float]:
    with urllib.request.urlopen(
        f"http://{host}:{port}/metrics", timeout=timeout
    ) as response:
        text = response.read().decode()
    values: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        if not line.startswith(TRACKED_METRIC_PREFIXES):
            continue
        metric_key, raw_value = line.rsplit(" ", 1)
        try:
            values[metric_key] = float(raw_value)
        except ValueError:
            continue
    return values


def metric_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {
        key: after.get(key, 0.0) - before.get(key, 0.0)
        for key in sorted(before.keys() | after.keys())
    }


def metric_sum(delta: dict[str, float], prefix: str) -> float:
    return sum(value for key, value in delta.items() if key.startswith(prefix))


def prepare_events(
    manifest: dict[str, Any],
    sources: dict[str, dict[str, Any]],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, dict[str, Any]]]:
    sessions = {session["session_id"]: session for session in manifest["sessions"]}
    events: dict[tuple[str, str], dict[str, Any]] = {}
    for session_id, session in sessions.items():
        for node in session["nodes"]:
            prompt = reconstruct_prompt(session, node, sources[session_id])
            events[(session_id, node["node_id"])] = {
                "session": session,
                "node": node,
                "prompt": prompt,
            }
    return events, sessions


def dry_run_summary(
    manifest: dict[str, Any], events: dict[tuple[str, str], dict[str, Any]]
) -> dict[str, Any]:
    prompts = [event["prompt"] for event in events.values()]
    return {
        "status": "DRY_RUN_OK",
        "manifest_summary": manifest["summary"],
        "validated_prompts": len(prompts),
        "total_mapped_prompt_tokens": sum(map(len, prompts)),
        "max_mapped_prompt_tokens": max(map(len, prompts), default=0),
    }


def run_batches(
    manifest: dict[str, Any],
    events: dict[tuple[str, str], dict[str, Any]],
    *,
    mode: str,
    endpoint: str,
    model: str,
    max_tokens: int,
    timeout: int,
    ttl_seconds: int,
    request_sender=send,
    progress: bool = False,
) -> tuple[list[dict[str, Any]], float]:
    """Execute deterministic barrier batches with an injectable sender."""
    rows: list[dict[str, Any]] = []
    batches: dict[int, list[dict[str, Any]]] = {}
    for entry in manifest["dispatch_plan"]:
        batches.setdefault(int(entry["batch_index"]), []).append(entry)

    benchmark_started = time.perf_counter()
    max_concurrency = int(manifest["config"]["max_concurrency"])
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        ordered_batch_indices = sorted(batches)
        for completed_batch_count, batch_index in enumerate(ordered_batch_indices, 1):
            entries = sorted(
                batches[batch_index], key=lambda item: item["dispatch_ordinal"]
            )
            futures = []
            for entry in entries:
                event = events[(entry["session_id"], entry["node_id"])]
                node = event["node"]
                kv_policy = None
                if mode == "geneakv-r0":
                    kv_policy = policy(
                        str(node["online_value_class"]),
                        str(entry["session_id"]),
                        str(entry["node_id"]),
                        int(node["source_request_index"]) + 1,
                        len(event["prompt"]),
                        ttl_seconds,
                    )
                future = executor.submit(
                    request_sender,
                    endpoint,
                    model,
                    event["prompt"],
                    kv_policy,
                    max_tokens,
                    timeout,
                )
                futures.append((entry, event, future))

            # Barrier: every request in this batch completes before the next
            # batch, keeping the logical schedule reproducible across modes.
            for entry, event, future in futures:
                response = future.result()
                node = event["node"]
                rows.append(
                    {
                        **entry,
                        **response,
                        "source_request_index": node["source_request_index"],
                        "source_stop": node["source_stop"],
                        "online_value_class": node["online_value_class"],
                        "resume_from_high": node["resume_from_high"],
                        "mapped_prompt_tokens": node["prompt_tokens"],
                        "parent_lcp_tokens": node["parent_lcp_tokens"],
                        "prompt_sha256": node["prompt_sha256"],
                    }
                )
            if progress and (
                completed_batch_count == 1
                or completed_batch_count % 5 == 0
                or completed_batch_count == len(ordered_batch_indices)
            ):
                print(
                    "[REPLAY] "
                    f"batches={completed_batch_count}/{len(ordered_batch_indices)} "
                    f"requests={len(rows)}/{len(manifest['dispatch_plan'])}",
                    flush=True,
                )

    rows.sort(key=lambda row: int(row["dispatch_ordinal"]))
    return rows, time.perf_counter() - benchmark_started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--mode", choices=("eager", "geneakv-r0"), required=True)
    parser.add_argument("--model", default="geneakv-qwen3-8b")
    parser.add_argument("--max-tokens", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--ttl-seconds", type=int, default=600)
    parser.add_argument("--vllm-block-size", type=int, default=128)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported replay manifest schema")
    sources = load_and_validate_sources(manifest, args.trace_dir)
    events, _ = prepare_events(manifest, sources)

    if args.dry_run:
        summary = dry_run_summary(manifest, events)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    endpoint = f"http://{args.host}:{args.port}/v1/completions"
    # Warm-up is outside the measured interval and intentionally tiny.
    send(endpoint, args.model, [1001, 1002, 1003, 1004], None, args.max_tokens, args.timeout)
    metrics_before = read_metrics(args.host, args.port, args.timeout)
    rows, benchmark_elapsed = run_batches(
        manifest,
        events,
        mode=args.mode,
        endpoint=endpoint,
        model=args.model,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        ttl_seconds=args.ttl_seconds,
        progress=True,
    )
    metrics_after = read_metrics(args.host, args.port, args.timeout)
    delta = metric_delta(metrics_before, metrics_after)

    local_queries = metric_sum(delta, "vllm:prefix_cache_queries_total")
    local_hits = metric_sum(delta, "vllm:prefix_cache_hits_total")
    external_queries = metric_sum(
        delta, "vllm:external_prefix_cache_queries_total"
    )
    external_hits = metric_sum(delta, "vllm:external_prefix_cache_hits_total")
    resume_rows = [row for row in rows if row["resume_from_high"]]
    resume_latencies = [float(row["elapsed_seconds"]) for row in resume_rows]
    all_latencies = [float(row["elapsed_seconds"]) for row in rows]
    block_size = max(1, args.vllm_block_size)
    expected_high_resume_tokens = sum(
        (int(row["parent_lcp_tokens"]) // block_size) * block_size
        for row in resume_rows
    )

    summary = {
        "mode": args.mode,
        "model": args.model,
        "requests": len(rows),
        "high_store_candidates": sum(
            row["online_value_class"] == "HIGH" for row in rows
        ),
        "high_resume_requests": len(resume_rows),
        "benchmark_elapsed_seconds": benchmark_elapsed,
        "throughput_requests_per_second": (
            len(rows) / benchmark_elapsed if benchmark_elapsed else 0.0
        ),
        "request_mean_seconds": statistics.fmean(all_latencies),
        "request_p95_seconds": percentile(all_latencies, 0.95),
        "resume_mean_seconds": (
            statistics.fmean(resume_latencies) if resume_latencies else 0.0
        ),
        "resume_p50_seconds": percentile(resume_latencies, 0.50),
        "resume_p95_seconds": percentile(resume_latencies, 0.95),
        "resume_p99_seconds": percentile(resume_latencies, 0.99),
        "expected_high_resume_cacheable_tokens": expected_high_resume_tokens,
        "local_queries": local_queries,
        "local_hits": local_hits,
        "external_queries": external_queries,
        "external_hits": external_hits,
        "aggregate_effective_hit_rate": (
            (local_hits + external_hits) / local_queries if local_queries else 0.0
        ),
        "external_restore_coverage_proxy": (
            external_hits / expected_high_resume_tokens
            if expected_high_resume_tokens
            else 0.0
        ),
    }
    result = {
        "source": "GeneaKV deterministic interleaved online NPU replay",
        "manifest": str(args.manifest),
        "manifest_config": manifest["config"],
        "mode": args.mode,
        "run_config": {
            "host": args.host,
            "port": args.port,
            "model": args.model,
            "max_tokens": args.max_tokens,
            "ttl_seconds": args.ttl_seconds,
            "vllm_block_size": args.vllm_block_size,
        },
        "summary": summary,
        "cache_metrics_delta": delta,
        "requests": rows,
        "limitations": [
            "Anonymous source blocks are mapped to deterministic token IDs.",
            "The mapper is trace/session scoped because source hash IDs are local.",
            "Non-streaming elapsed time with max_tokens=2 is a Resume latency proxy, not wire-level TTFT.",
            "External restore coverage is an aggregate proxy because Prometheus cache counters are not request-scoped.",
        ],
    }
    if args.output is None:
        raise ValueError("--output is required unless --dry-run is used")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
