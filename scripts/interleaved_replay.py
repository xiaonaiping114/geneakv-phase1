#!/usr/bin/env python3
"""Shared helpers for deterministic, interleaved real-trace replay.

The public traces use anonymous hash IDs whose scope is local to each trace.
The token mapper therefore includes ``session_id`` in its hash input.  This is
important: mapping ``hash_id=1`` identically across unrelated traces would
manufacture cross-session prefix-cache hits that do not exist in the source
data.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any


MANIFEST_SCHEMA_VERSION = 1
SCHEDULER_NAME = "round_robin_barrier_batches"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=262_144)
def scoped_source_block_tokens(
    session_id: str,
    block_hash: int,
    tokens_per_source_block: int,
) -> tuple[int, ...]:
    """Map one trace-local anonymous block to stable ordinary token IDs."""
    output: list[int] = []
    counter = 0
    while len(output) < tokens_per_source_block:
        material = f"geneakv:{session_id}:{block_hash}:{counter}".encode()
        digest = hashlib.sha256(material).digest()
        for offset in range(0, len(digest), 2):
            value = int.from_bytes(digest[offset : offset + 2], "big")
            # Qwen3 has a much larger vocabulary; this deliberately stays in
            # the ordinary-token range already used by the phase-1 harness.
            output.append(1000 + value % 48000)
            if len(output) == tokens_per_source_block:
                break
        counter += 1
    return tuple(output)


def empty_trace_prompt(session_id: str, request_index: int) -> list[int]:
    """Create a deterministic non-empty prompt for an empty trace event."""
    digest = hashlib.sha256(
        f"geneakv-empty:{session_id}:{request_index}".encode()
    ).digest()
    seed = int.from_bytes(digest[:8], "big")
    return list(scoped_source_block_tokens(session_id, seed, 4))


def trace_prompt(
    session_id: str,
    hash_ids: list[int],
    tokens_per_source_block: int,
    request_index: int,
) -> list[int]:
    if not hash_ids:
        return empty_trace_prompt(session_id, request_index)
    prompt: list[int] = []
    for block_hash in hash_ids:
        prompt.extend(
            scoped_source_block_tokens(
                session_id, int(block_hash), tokens_per_source_block
            )
        )
    return prompt


def prompt_sha256(tokens: list[int]) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(int(token).to_bytes(4, "big", signed=False))
    return digest.hexdigest()


def lcp_blocks(left: list[int], right: list[int]) -> int:
    count = 0
    for left_value, right_value in zip(left, right):
        if left_value != right_value:
            break
        count += 1
    return count


def value_class(stop: str, is_last: bool) -> str:
    if stop == "tool_use":
        return "HIGH"
    if is_last or stop == "end_turn":
        return "LOW"
    return "NORMAL"


def build_manifest(
    trace_dir: Path,
    *,
    max_traces: int = 20,
    requests_per_trace: int = 20,
    target_prompt_tokens: int = 64_000,
    max_concurrency: int = 8,
) -> dict[str, Any]:
    """Build an immutable workload manifest and deterministic dispatch plan."""
    subset_manifest_path = trace_dir / "manifest.json"
    trace_paths: list[Path]
    if subset_manifest_path.exists():
        subset_manifest = json.loads(subset_manifest_path.read_text(encoding="utf-8"))
        ranked_paths = [
            trace_dir / str(row["relative_path"])
            for row in subset_manifest.get("traces", [])
            if row.get("relative_path")
        ]
        trace_paths = ranked_paths or sorted(trace_dir.glob("trace_*.json"))
    else:
        trace_paths = sorted(trace_dir.glob("trace_*.json"))
    if max_traces:
        trace_paths = trace_paths[:max_traces]
    if not trace_paths:
        raise ValueError(f"no trace_*.json files under {trace_dir}")
    if requests_per_trace <= 0:
        raise ValueError("requests_per_trace must be positive")
    if target_prompt_tokens <= 0:
        raise ValueError("target_prompt_tokens must be positive")
    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be positive")

    sessions: list[dict[str, Any]] = []
    for trace_path in trace_paths:
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        requests = list(trace.get("requests") or [])[:requests_per_trace]
        if not requests:
            continue
        trace_id = str(trace.get("id") or trace_path.stem)
        session_id = f"{trace_id}/main"
        max_blocks = max(
            (len(request.get("hash_ids") or []) for request in requests),
            default=1,
        )
        tokens_per_source_block = max(
            1, min(16, target_prompt_tokens // max(1, max_blocks))
        )
        if max_blocks * tokens_per_source_block > target_prompt_tokens:
            raise ValueError(
                f"{trace_id}: at least {max_blocks} mapped tokens are required, "
                f"exceeding target_prompt_tokens={target_prompt_tokens}"
            )

        nodes: list[dict[str, Any]] = []
        previous_hashes: list[int] = []
        previous_stop = ""
        for request_index, request in enumerate(requests):
            hashes = [int(value) for value in request.get("hash_ids") or []]
            prompt = trace_prompt(
                session_id, hashes, tokens_per_source_block, request_index
            )
            if len(prompt) > target_prompt_tokens:
                raise ValueError(
                    f"{trace_id} request {request_index}: mapped prompt has "
                    f"{len(prompt)} tokens, limit={target_prompt_tokens}"
                )
            lcp = lcp_blocks(previous_hashes, hashes)
            stop = str(request.get("stop") or "")
            node_value = value_class(stop, request_index + 1 == len(requests))
            nodes.append(
                {
                    "node_id": f"request-{request_index:04d}",
                    "source_request_index": request_index,
                    "source_time_seconds": float(request.get("t") or 0.0),
                    "source_stop": stop,
                    "source_input_types": list(request.get("input_types") or []),
                    "source_hash_blocks": len(hashes),
                    "tokens_per_source_block": tokens_per_source_block,
                    "prompt_tokens": len(prompt),
                    "prompt_sha256": prompt_sha256(prompt),
                    "parent_lcp_blocks": lcp,
                    "parent_lcp_tokens": lcp * tokens_per_source_block,
                    "online_value_class": node_value,
                    "resume_from_high": request_index > 0
                    and previous_stop == "tool_use",
                }
            )
            previous_hashes = hashes
            previous_stop = stop

        sessions.append(
            {
                "session_id": session_id,
                "trace_id": trace_id,
                "trace_relative_path": str(trace_path.relative_to(trace_dir)),
                "trace_sha256": file_sha256(trace_path),
                "source_block_size": int(trace.get("block_size") or 0),
                "source_hash_id_scope": str(
                    trace.get("hash_id_scope") or "unknown"
                ),
                "tokens_per_source_block": tokens_per_source_block,
                "nodes": nodes,
            }
        )

    if not sessions:
        raise ValueError("selected traces contain no requests")

    dispatch_plan: list[dict[str, Any]] = []
    ordinal = 0
    batch_index = 0
    max_rounds = max(len(session["nodes"]) for session in sessions)
    for round_index in range(max_rounds):
        round_entries: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for session in sessions:
            if round_index < len(session["nodes"]):
                round_entries.append((session, session["nodes"][round_index]))
        for round_offset in range(0, len(round_entries), max_concurrency):
            for batch_slot, (session, node) in enumerate(
                round_entries[round_offset : round_offset + max_concurrency]
            ):
                dispatch_plan.append(
                    {
                        "dispatch_ordinal": ordinal,
                        "round_index": round_index,
                        "batch_index": batch_index,
                        "batch_slot": batch_slot,
                        "session_id": session["session_id"],
                        "node_id": node["node_id"],
                    }
                )
                ordinal += 1
            batch_index += 1

    high_nodes = sum(
        node["online_value_class"] == "HIGH"
        for session in sessions
        for node in session["nodes"]
    )
    high_resumes = sum(
        bool(node["resume_from_high"])
        for session in sessions
        for node in session["nodes"]
    )
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source": "callanjfox/kv-cache-tester real trace subset",
        "config": {
            "scheduler": SCHEDULER_NAME,
            "trace_count_limit": max_traces,
            "requests_per_trace_limit": requests_per_trace,
            "target_prompt_tokens": target_prompt_tokens,
            "max_concurrency": max_concurrency,
            "token_mapping_scope": "session_id + trace-local block_hash",
            "artificial_low_requests": 0,
            "trace_order": (
                "subset_manifest_selection_rank"
                if subset_manifest_path.exists() and ranked_paths
                else "lexicographic_filename"
            ),
        },
        "summary": {
            "sessions": len(sessions),
            "requests": len(dispatch_plan),
            "rounds": max_rounds,
            "batches": 1
            + max(entry["batch_index"] for entry in dispatch_plan),
            "r0_high_store_candidates": high_nodes,
            "resume_from_r0_high_requests": high_resumes,
        },
        "sessions": sessions,
        "dispatch_plan": dispatch_plan,
    }


def load_and_validate_sources(
    manifest: dict[str, Any], trace_dir: Path
) -> dict[str, dict[str, Any]]:
    """Load source traces and verify every content digest in the manifest."""
    sources: dict[str, dict[str, Any]] = {}
    for session in manifest["sessions"]:
        path = trace_dir / session["trace_relative_path"]
        actual_sha = file_sha256(path)
        if actual_sha != session["trace_sha256"]:
            raise ValueError(
                f"trace digest mismatch for {path}: "
                f"manifest={session['trace_sha256']} actual={actual_sha}"
            )
        trace = json.loads(path.read_text(encoding="utf-8"))
        sources[session["session_id"]] = trace
    return sources


def reconstruct_prompt(
    session: dict[str, Any],
    node: dict[str, Any],
    source_trace: dict[str, Any],
) -> list[int]:
    request_index = int(node["source_request_index"])
    source_request = source_trace["requests"][request_index]
    hashes = [int(value) for value in source_request.get("hash_ids") or []]
    prompt = trace_prompt(
        session["session_id"],
        hashes,
        int(session["tokens_per_source_block"]),
        request_index,
    )
    actual_sha = prompt_sha256(prompt)
    if actual_sha != node["prompt_sha256"]:
        raise ValueError(
            f"prompt digest mismatch for {session['session_id']}/{node['node_id']}"
        )
    return prompt
