#!/usr/bin/env python3
"""Run a deterministic online NPU benchmark over a real trace subset.

Each source trace stores anonymous hash blocks rather than prompt text.  The
driver therefore selects the largest deterministic token expansion in [1, 16]
that keeps the first N requests under ``target_prompt_tokens``.  The selected
expansion is workload-only and is identical for every cache policy.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * q))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--mode", choices=("eager", "geneakv"), required=True)
    parser.add_argument("--model", default="geneakv-qwen3-0.6b")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--requests-per-trace", type=int, default=5)
    parser.add_argument("--low-per-gap", type=int, default=4)
    parser.add_argument("--low-tokens", type=int, default=4096)
    parser.add_argument("--target-prompt-tokens", type=int, default=40000)
    parser.add_argument("--max-traces", type=int, default=0)
    args = parser.parse_args()

    traces = sorted(args.trace_dir.glob("trace_*.json"))
    if args.max_traces:
        traces = traces[: args.max_traces]
    if not traces:
        raise SystemExit(f"no traces found under {args.trace_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    benchmark = Path(__file__).with_name("benchmark_kv_cache_trace.py")

    for index, trace_path in enumerate(traces, 1):
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        requests = trace.get("requests", [])[: args.requests_per_trace]
        max_blocks = max(
            (len(request.get("hash_ids", [])) for request in requests),
            default=1,
        )
        tokens_per_block = max(
            1, min(16, args.target_prompt_tokens // max(1, max_blocks))
        )
        protected_tokens = min(
            args.target_prompt_tokens, max_blocks * tokens_per_block
        )
        output_path = args.output_dir / f"{trace_path.stem}-{args.mode}.json"
        command = [
            sys.executable,
            str(benchmark),
            str(trace_path),
            "--port",
            str(args.port),
            "--mode",
            args.mode,
            "--model",
            args.model,
            "--requests",
            str(args.requests_per_trace),
            "--tokens-per-source-block",
            str(tokens_per_block),
            "--low-per-gap",
            str(args.low_per_gap),
            "--low-tokens",
            str(args.low_tokens),
            "--protected-prefix-tokens",
            str(protected_tokens),
            "--max-prompt-tokens",
            str(args.target_prompt_tokens),
            "--output",
            str(output_path),
        ]
        print(
            f"[{index}/{len(traces)}] {trace_path.stem} mode={args.mode} "
            f"requests={len(requests)} max_blocks={max_blocks} "
            f"tokens_per_block={tokens_per_block}",
            flush=True,
        )
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        result = json.loads(output_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "trace_id": result["trace_id"],
                "mode": args.mode,
                "requests": result["config"]["requests"],
                "resume_requests": result["summary"]["resume_requests"],
                "tokens_per_source_block": tokens_per_block,
                "max_source_blocks": max_blocks,
                "resume_mean_seconds": result["summary"]["resume_mean_seconds"],
                "resume_p95_seconds": result["summary"]["resume_p95_seconds"],
                "effective_hit_rate": result["cache_metrics"]["effective_hit_rate"],
                "local_queries": result["cache_metrics"]["local_queries"],
                "local_hits": result["cache_metrics"]["local_hits"],
                "external_hits": result["cache_metrics"]["external_hits"],
            }
        )

    resume_means = [row["resume_mean_seconds"] for row in rows]
    total_queries = sum(row["local_queries"] for row in rows)
    total_hits = sum(row["local_hits"] + row["external_hits"] for row in rows)
    summary = {
        "source": "callanjfox/kv-cache-tester real trace subset",
        "mode": args.mode,
        "model": args.model,
        "trace_count": len(rows),
        "requests_per_trace_limit": args.requests_per_trace,
        "low_per_gap": args.low_per_gap,
        "low_tokens": args.low_tokens,
        "target_prompt_tokens": args.target_prompt_tokens,
        "aggregate_effective_hit_rate": total_hits / total_queries
        if total_queries
        else 0.0,
        "trace_resume_mean_seconds": statistics.fmean(resume_means),
        "trace_resume_p95_seconds": percentile(resume_means, 0.95),
        "total_local_queries": total_queries,
        "total_cache_hits": total_hits,
        "rows": rows,
        "limitations": [
            "Anonymous source hash blocks are mapped to deterministic token IDs.",
            "Token expansion is selected from workload shape only and shared by policies.",
            "This is an online NPU replay, but it is not the original private prompt text.",
        ],
    }
    summary_path = args.output_dir / f"suite-{args.mode}-summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
