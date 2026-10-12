#!/usr/bin/env python3
"""Normalize kv-cache-tester traces into GeneaKV AgentEvent JSONL.

The online label uses only information available when a request completes.
The oracle label inspects the following request and is emitted strictly for
evaluation; it must never be consumed by the online policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def longest_common_prefix(left: list[int], right: list[int]) -> int:
    count = 0
    for left_id, right_id in zip(left, right):
        if left_id != right_id:
            break
        count += 1
    return count


def aligned_prefix(tokens: int, boundary: int, block_size: int) -> int:
    return max(0, min(tokens, boundary) // block_size * block_size)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_trace(
    path: Path,
    protect_tokens: int,
    engine_block_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trace = json.loads(path.read_text(encoding="utf-8"))
    requests = trace["requests"]
    source_block_size = int(trace.get("block_size", 64))
    session_id = str(trace.get("id") or path.stem)
    events: list[dict[str, Any]] = []
    confusion = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}

    for index, request in enumerate(requests):
        hashes = [int(value) for value in request.get("hash_ids", [])]
        next_request = requests[index + 1] if index + 1 < len(requests) else None
        next_hashes = (
            [int(value) for value in next_request.get("hash_ids", [])]
            if next_request is not None
            else []
        )
        next_lcp_blocks = longest_common_prefix(hashes, next_hashes)
        next_input_types = set(next_request.get("input_types", [])) if next_request else set()

        # Online policy: a tool call means the same Agent branch is expected to
        # resume with a tool result. This uses no future request information.
        online_high = request.get("stop") == "tool_use"
        is_last = next_request is None
        online_class = "HIGH" if online_high else ("LOW" if is_last else "NORMAL")

        # Offline oracle: the next invocation really arrived with a tool result
        # and reused at least 80% of this node's block prefix.
        oracle_high = bool(
            next_request is not None
            and "tool_result" in next_input_types
            and hashes
            and next_lcp_blocks / len(hashes) >= 0.80
        )
        if online_high and oracle_high:
            confusion["tp"] += 1
        elif online_high:
            confusion["fp"] += 1
        elif oracle_high:
            confusion["fn"] += 1
        else:
            confusion["tn"] += 1

        if online_high:
            event_type = "TOOL_CALL"
        elif is_last:
            event_type = "SESSION_END"
        else:
            event_type = "MODEL_RESPONSE"

        prefix_tokens = int(request.get("in") or len(hashes) * source_block_size)
        event = {
            "session_id": session_id,
            "node_id": f"request-{index:04d}",
            "parent_id": f"request-{index - 1:04d}" if index else None,
            "event_type": event_type,
            "event_seq": index + 1,
            "epoch": 1,
            "prefix_tokens": prefix_tokens,
            "branch_id": "main",
            "metadata": {
                "source": "callanjfox/kv-cache-tester",
                "source_file": path.name,
                "source_request_index": index,
                "timestamp_seconds": float(request.get("t", 0.0)),
                "input_types": list(request.get("input_types", [])),
                "output_types": list(request.get("output_types", [])),
                "stop_reason": str(request.get("stop", "")),
                "source_hash_blocks": len(hashes),
                "source_block_size": source_block_size,
                "next_lcp_blocks": next_lcp_blocks,
                "next_lcp_tokens": next_lcp_blocks * source_block_size,
                "next_resume_seconds": (
                    float(next_request.get("t", 0.0)) - float(request.get("t", 0.0))
                    if next_request is not None
                    else None
                ),
                "online_value_class": online_class,
                "online_protected_prefix_tokens": (
                    aligned_prefix(prefix_tokens, protect_tokens, engine_block_size)
                    if online_high
                    else 0
                ),
                "oracle_future_resume": oracle_high,
            },
        }
        events.append(event)

    tp = confusion["tp"]
    fp = confusion["fp"]
    fn = confusion["fn"]
    stats = {
        "source_file": str(path),
        "sha256": sha256(path),
        "session_id": session_id,
        "requests": len(requests),
        "source_block_size": source_block_size,
        "online_high": sum(
            event["metadata"]["online_value_class"] == "HIGH" for event in events
        ),
        "oracle_high": sum(
            bool(event["metadata"]["oracle_future_resume"]) for event in events
        ),
        "confusion": confusion,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
    }
    return events, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--protect-tokens", type=int, default=4096)
    parser.add_argument("--engine-block-size", type=int, default=128)
    args = parser.parse_args()

    all_events: list[dict[str, Any]] = []
    trace_stats: list[dict[str, Any]] = []
    for path in args.traces:
        events, stats = normalize_trace(
            path,
            protect_tokens=args.protect_tokens,
            engine_block_size=args.engine_block_size,
        )
        all_events.extend(events)
        trace_stats.append(stats)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for event in all_events:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    aggregate = {
        "source": "https://github.com/callanjfox/kv-cache-tester",
        "source_revision": args.source_revision,
        "protect_tokens": args.protect_tokens,
        "engine_block_size": args.engine_block_size,
        "events": len(all_events),
        "traces": trace_stats,
        "output": str(args.output),
    }
    args.manifest.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
