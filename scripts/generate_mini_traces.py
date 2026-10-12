#!/usr/bin/env python3
"""Generate deterministic Agent events for Lineage/DAG functional tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def event(session: str, seq: int, node: str, kind: str, **fields: object) -> dict:
    row = {
        "session_id": session,
        "event_seq": seq,
        "timestamp_ms": seq * 1000,
        "request_id": f"{session}-req-{seq}",
        "node_id": node,
        "event_type": kind,
    }
    row.update(fields)
    return row


def build() -> list[dict]:
    rows: list[dict] = []
    rows += [
        event("tool-resume", 1, "N1", "REQUEST_END", branch_id="main", input_tokens=4096, prefix_tokens=4096, state="COMMITTED"),
        event("tool-resume", 2, "N1", "TOOL_CALL", branch_id="main", tool_id="search-1", tool_eta_ms=10000, state="TOOL_WAIT"),
        event("tool-resume", 3, "N2", "TOOL_RESULT", parent_node_id="N1", branch_id="main", tool_id="search-1", prefix_tokens=4096, state="ACTIVE"),
        event("tool-resume", 4, "N2", "RESUME", parent_node_id="N1", branch_id="main", prefix_tokens=4096, state="COMMITTED"),
    ]
    rows += [
        event("fork-commit", 1, "N1", "REQUEST_END", branch_id="main", input_tokens=6144, prefix_tokens=6144, state="COMMITTED"),
        event("fork-commit", 2, "N1", "FORK", branch_id="main", child_branch_ids=["A", "B"], prefix_tokens=6144, state="ACTIVE"),
        event("fork-commit", 3, "N2", "BRANCH_STEP", parent_node_id="N1", branch_id="A", prefix_tokens=6144, state="SPECULATIVE"),
        event("fork-commit", 4, "N3", "BRANCH_STEP", parent_node_id="N1", branch_id="B", prefix_tokens=6144, state="SPECULATIVE"),
        event("fork-commit", 5, "N2", "COMMIT", parent_node_id="N1", branch_id="A", prefix_tokens=6144, state="COMMITTED"),
    ]
    rows += [
        event("fork-abort", 1, "N1", "FORK", branch_id="main", child_branch_ids=["A", "B"], prefix_tokens=4096, state="ACTIVE"),
        event("fork-abort", 2, "N2", "BRANCH_STEP", parent_node_id="N1", branch_id="B", prefix_tokens=5120, state="SPECULATIVE"),
        event("fork-abort", 3, "N2", "ABORT", parent_node_id="N1", branch_id="B", prefix_tokens=5120, state="ABORTED"),
    ]
    rows += [
        event("retry", 1, "N1", "REQUEST_END", branch_id="main", input_tokens=4096, prefix_tokens=4096, state="COMMITTED"),
        event("retry", 2, "N2", "ERROR", parent_node_id="N1", branch_id="attempt-1", prefix_tokens=4608, state="FAILED"),
        event("retry", 3, "N3", "RETRY", parent_node_id="N1", branch_id="attempt-2", prefix_tokens=4096, state="ACTIVE"),
    ]
    rows += [
        event("one-shot", 1, "N1", "REQUEST_END", branch_id="main", input_tokens=4096, prefix_tokens=4096, state="FINISHED"),
        event("one-shot", 2, "N1", "EXPIRE", branch_id="main", prefix_tokens=4096, state="EXPIRED"),
    ]
    rows += [
        event("tool-expire", 1, "N1", "TOOL_CALL", branch_id="main", prefix_tokens=4096, tool_id="slow-1", tool_eta_ms=60000, state="TOOL_WAIT"),
        event("tool-expire", 2, "N1", "EXPIRE", branch_id="main", prefix_tokens=4096, tool_id="slow-1", state="EXPIRED"),
    ]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "datasets/synthetic/agent_events.jsonl",
    )
    output = parser.parse_args().output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in build():
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(build())} events to {output}")


if __name__ == "__main__":
    main()
