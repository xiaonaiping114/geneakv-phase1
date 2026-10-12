#!/usr/bin/env python3
"""Replay an Agent event JSONL and emit the final GeneaKV policy registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from geneakv import AgentEvent, LineageGraph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decisions-output", type=Path)
    parser.add_argument("--block-size", type=int, default=128)
    args = parser.parse_args()

    graph = LineageGraph(block_size=args.block_size)
    event_count = 0
    decisions = []
    with args.trace.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            event = AgentEvent.from_dict(json.loads(line))
            policy = graph.apply(event)
            decisions.append(
                {
                    "session_id": event.session_id,
                    "node_id": event.node_id,
                    "event_seq": event.event_seq,
                    "event_type": event.event_type.value,
                    "kv_transfer_params": policy.to_kv_transfer_params(),
                }
            )
            event_count += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    decisions_output = args.decisions_output or args.output.with_name(
        args.output.stem + "-decisions.jsonl"
    )
    with decisions_output.open("w", encoding="utf-8") as stream:
        for decision in decisions:
            stream.write(json.dumps(decision, ensure_ascii=False) + "\n")
    counts = {"HIGH": 0, "NORMAL": 0, "LOW": 0}
    with args.output.open("w", encoding="utf-8") as stream:
        for key in sorted(graph.nodes):
            node = graph.nodes[key]
            policy = graph.policy_for(node)
            counts[policy.value_class.value] += 1
            record = {
                "session_id": node.session_id,
                "node_id": node.node_id,
                "parent_id": node.parent_id,
                "branch_id": node.branch_id,
                "prefix_tokens": node.prefix_tokens,
                "children": sorted(node.children),
                "live_branches": sorted(node.live_children),
                "kv_transfer_params": policy.to_kv_transfer_params(),
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "events": event_count,
                "nodes": len(graph.nodes),
                "classes": counts,
                "output": str(args.output),
                "decisions_output": str(decisions_output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
