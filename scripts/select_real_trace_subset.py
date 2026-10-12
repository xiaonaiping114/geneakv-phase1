#!/usr/bin/env python3
"""Select deterministic, diverse real-trace subsets without using cache outcome.

Selection uses workload-shape features only: request count, Tool ratio, maximum
prefix length, duration and Sub-agent count.  It never uses future KV hits or a
policy's measured benefit, avoiding cherry-picking profitable traces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from statistics import fmean
from typing import Any

from scripts.analyze_high_value_traces import iter_streams


FEATURES = (
    "log_requests",
    "tool_ratio",
    "log_max_blocks",
    "log_duration",
    "log_subagents",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe(path: Path) -> dict[str, Any] | None:
    try:
        trace = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(trace, dict) or not isinstance(trace.get("requests"), list):
        return None
    requests = []
    subagents = 0
    for stream in iter_streams(trace):
        requests.extend(stream.requests)
        if stream.kind == "subagent":
            subagents += 1
    if not requests:
        return None
    tool_calls = sum(str(row.get("stop") or "") == "tool_use" for row in requests)
    max_blocks = max(len(row.get("hash_ids", [])) for row in requests)
    times = [float(row.get("t", 0.0)) for row in requests]
    duration = max(times) - min(times) if times else 0.0
    return {
        "trace_id": str(trace.get("id", path.stem)),
        "relative_path": path.name,
        "path": path,
        "sha256": sha256(path),
        "requests": len(requests),
        "tool_calls": tool_calls,
        "tool_ratio": tool_calls / len(requests),
        "max_blocks": max_blocks,
        "duration_seconds": duration,
        "subagents": subagents,
        "log_requests": math.log1p(len(requests)),
        "log_max_blocks": math.log1p(max_blocks),
        "log_duration": math.log1p(duration),
        "log_subagents": math.log1p(subagents),
    }


def normalize(rows: list[dict[str, Any]]) -> None:
    for feature in FEATURES:
        values = [float(row[feature]) for row in rows]
        low, high = min(values), max(values)
        for row in rows:
            row[f"norm_{feature}"] = (
                (float(row[feature]) - low) / (high - low) if high > low else 0.0
            )


def vector(row: dict[str, Any]) -> tuple[float, ...]:
    return tuple(float(row[f"norm_{feature}"]) for feature in FEATURES)


def distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def diverse_order(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Greedy farthest-point traversal, deterministic on trace_id ties."""
    normalize(rows)
    center = tuple(fmean(row[f"norm_{feature}"] for row in rows) for feature in FEATURES)
    first = min(rows, key=lambda row: (distance(vector(row), center), row["trace_id"]))
    selected = [first]
    remaining = {row["trace_id"]: row for row in rows if row is not first}
    while remaining:
        candidate = max(
            remaining.values(),
            key=lambda row: (
                min(distance(vector(row), vector(chosen)) for chosen in selected),
                row["trace_id"],
            ),
        )
        selected.append(candidate)
        del remaining[candidate["trace_id"]]
    return selected


def export_subset(
    ordered: list[dict[str, Any]], count: int, output_root: Path
) -> None:
    output = output_root / f"trace-{count}"
    output.mkdir(parents=True, exist_ok=True)
    chosen = ordered[:count]
    manifest_rows = []
    for rank, row in enumerate(chosen, start=1):
        target = output / row["relative_path"]
        shutil.copy2(row["path"], target)
        manifest_rows.append(
            {
                key: value
                for key, value in row.items()
                if key != "path" and not key.startswith("norm_")
            }
            | {"selection_rank": rank}
        )
    manifest = {
        "method": "deterministic greedy farthest-point workload-shape sampling",
        "outcome_features_used": False,
        "features": list(FEATURES),
        "source_population": len(ordered),
        "selected": count,
        "nested_property": "trace-20 is the first 20 members of trace-50",
        "traces": manifest_rows,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--counts", type=int, nargs="+", default=[20, 50])
    args = parser.parse_args()
    rows = []
    for path in sorted(args.trace_dir.rglob("*.json")):
        row = describe(path)
        if row is not None:
            rows.append(row)
    maximum = max(args.counts)
    if maximum > len(rows):
        raise SystemExit(f"requested {maximum} traces, only {len(rows)} available")
    ordered = diverse_order(rows)
    for count in sorted(set(args.counts)):
        export_subset(ordered, count, args.output_root)
    print(
        json.dumps(
            {
                "population": len(rows),
                "counts": sorted(set(args.counts)),
                "first_ids": [row["trace_id"] for row in ordered[:maximum]],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
