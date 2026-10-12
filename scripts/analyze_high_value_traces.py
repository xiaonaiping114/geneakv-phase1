#!/usr/bin/env python3
"""Evaluate online HIGH rules against actual next-turn prefix reuse.

The public traces contain anonymized hash blocks rather than prompt text.  This
tool deliberately keeps online decisions and offline labels separate:

* R0 uses only the current response's stop reason (tool_use => HIGH).
* The offline label uses the next request's actual contiguous hash-prefix reuse.

No fixed percentage of nodes is marked HIGH.  Instead, the report sweeps reuse
block and time-horizon thresholds so a system-calibrated threshold can be chosen
later from measured Store/Load versus Recompute cost.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Stream:
    trace_id: str
    stream_id: str
    kind: str
    start_time: float
    block_size: int
    requests: list[dict[str, Any]]


def lcp_blocks(left: list[Any], right: list[Any]) -> int:
    count = 0
    for lhs, rhs in zip(left, right):
        if lhs != rhs:
            break
        count += 1
    return count


def iter_streams(trace: dict[str, Any]) -> Iterable[Stream]:
    trace_id = str(trace.get("id", "unknown"))
    block_size = int(trace.get("block_size", 64))
    main_requests: list[dict[str, Any]] = []
    for index, entry in enumerate(trace.get("requests", [])):
        nested = entry.get("requests")
        if isinstance(nested, list):
            agent_id = str(entry.get("agent_id", f"subagent-{index}"))
            yield Stream(
                trace_id=trace_id,
                stream_id=f"{trace_id}/{agent_id}",
                kind="subagent",
                start_time=float(entry.get("t", 0.0)),
                block_size=block_size,
                requests=[r for r in nested if isinstance(r, dict)],
            )
        elif isinstance(entry, dict) and isinstance(entry.get("hash_ids"), list):
            main_requests.append(entry)
    yield Stream(
        trace_id=trace_id,
        stream_id=f"{trace_id}/main",
        kind="main",
        start_time=0.0,
        block_size=block_size,
        requests=main_requests,
    )


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def classification_metrics(rows: list[dict[str, Any]], predicate) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    predicted_blocks = useful_blocks = false_positive_blocks = missed_blocks = 0
    for row in rows:
        predicted = bool(row["r0_high"])
        actual = bool(predicate(row))
        reusable = int(row["next_lcp_blocks"])
        current = int(row["current_blocks"])
        if predicted:
            predicted_blocks += current
        if actual:
            useful_blocks += reusable
        if predicted and actual:
            tp += 1
        elif predicted:
            fp += 1
            false_positive_blocks += current
        elif actual:
            fn += 1
            missed_blocks += reusable
        else:
            tn += 1
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": safe_div(tp, tp + fp),
        "recall": safe_div(tp, tp + fn),
        "f1": safe_div(2 * tp, 2 * tp + fp + fn),
        "predicted_store_blocks": predicted_blocks,
        "oracle_reuse_blocks": useful_blocks,
        "false_positive_store_blocks": false_positive_blocks,
        "false_negative_reuse_blocks": missed_blocks,
        "token_weighted_recall": safe_div(
            useful_blocks - missed_blocks, useful_blocks
        ),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--threshold-blocks",
        type=int,
        nargs="+",
        default=[1, 8, 16, 32, 64, 128],
    )
    parser.add_argument(
        "--horizons-seconds",
        type=float,
        nargs="+",
        default=[60, 300, 600, 3600],
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace_files = sorted(args.trace_dir.rglob("*.json"))
    rows: list[dict[str, Any]] = []
    stream_counts: Counter[str] = Counter()
    stop_counts: Counter[str] = Counter()

    for trace_path in trace_files:
        trace = json.loads(trace_path.read_text())
        for stream in iter_streams(trace):
            stream_counts[stream.kind] += 1
            requests = stream.requests
            for index, current in enumerate(requests):
                hashes = list(current.get("hash_ids", []))
                next_request = requests[index + 1] if index + 1 < len(requests) else None
                next_hashes = (
                    list(next_request.get("hash_ids", []))
                    if next_request is not None
                    else []
                )
                lcp = lcp_blocks(hashes, next_hashes)
                current_time = stream.start_time + float(current.get("t", 0.0))
                next_time = (
                    stream.start_time + float(next_request.get("t", 0.0))
                    if next_request is not None
                    else None
                )
                delay = (
                    max(0.0, next_time - current_time)
                    if next_time is not None
                    else None
                )
                stop = str(current.get("stop") or "")
                stop_counts[stop or "EMPTY"] += 1
                next_input_types = (
                    list(next_request.get("input_types", []))
                    if next_request is not None
                    else []
                )
                rows.append(
                    {
                        "trace_id": stream.trace_id,
                        "stream_id": stream.stream_id,
                        "stream_kind": stream.kind,
                        "request_index": index,
                        "absolute_time": current_time,
                        "stop": stop,
                        "input_types": list(current.get("input_types", [])),
                        "current_blocks": len(hashes),
                        "current_tokens": len(hashes) * stream.block_size,
                        "r0_high": stop == "tool_use",
                        "has_next": next_request is not None,
                        "next_has_tool_result": "tool_result" in next_input_types,
                        "next_lcp_blocks": lcp,
                        "next_lcp_tokens": lcp * stream.block_size,
                        "next_lcp_ratio": safe_div(lcp, len(hashes)),
                        "next_delay_seconds": delay,
                    }
                )

    decisions_path = args.output_dir / "high-value-decisions.jsonl"
    with decisions_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    sweeps = []
    for min_blocks in args.threshold_blocks:
        for horizon in args.horizons_seconds:
            def oracle(row, mb=min_blocks, h=horizon):
                delay = row["next_delay_seconds"]
                return (
                    delay is not None
                    and delay <= h
                    and row["next_lcp_blocks"] >= mb
                )

            metrics = classification_metrics(rows, oracle)
            metrics.update(
                {
                    "min_reuse_blocks": min_blocks,
                    "horizon_seconds": horizon,
                }
            )
            sweeps.append(metrics)

    tool_resume_metrics = classification_metrics(
        rows,
        lambda row: bool(
            row["next_has_tool_result"] and row["next_lcp_blocks"] > 0
        ),
    )
    r0_count = sum(bool(row["r0_high"]) for row in rows)
    nested_requests = sum(row["stream_kind"] == "subagent" for row in rows)
    summary = {
        "dataset": {
            "trace_dir": str(args.trace_dir),
            "trace_files": len(trace_files),
            "trace_file_sha256_manifest": "trace-sha256.json",
            "streams": dict(stream_counts),
            "requests": len(rows),
            "main_requests": len(rows) - nested_requests,
            "subagent_requests": nested_requests,
            "r0_high": r0_count,
            "r0_high_ratio": safe_div(r0_count, len(rows)),
            "stop_counts": dict(stop_counts),
        },
        "r0_vs_next_tool_result_reuse": tool_resume_metrics,
        "threshold_sweep": sweeps,
        "notes": [
            "R0 uses current stop reason only and does not inspect future data.",
            "Offline labels use actual next-turn contiguous hash-prefix reuse.",
            "Block/time thresholds are swept rather than selecting a fixed HIGH ratio.",
            "Restore-versus-recompute cost must be calibrated with NPU measurements.",
        ],
    }
    (args.output_dir / "high-value-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    sha_manifest = {
        str(path.relative_to(args.trace_dir)): file_sha256(path)
        for path in trace_files
    }
    (args.output_dir / "trace-sha256.json").write_text(
        json.dumps(sha_manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
