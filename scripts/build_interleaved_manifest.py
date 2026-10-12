#!/usr/bin/env python3
"""Build a deterministic real-trace replay manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.interleaved_replay import build_manifest
except ModuleNotFoundError:  # Direct execution: python scripts/build_*.py
    from interleaved_replay import build_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-traces", type=int, default=20)
    parser.add_argument("--requests-per-trace", type=int, default=20)
    parser.add_argument("--target-prompt-tokens", type=int, default=64_000)
    parser.add_argument("--max-concurrency", type=int, default=8)
    args = parser.parse_args()

    manifest = build_manifest(
        args.trace_dir,
        max_traces=args.max_traces,
        requests_per_trace=args.requests_per_trace,
        target_prompt_tokens=args.target_prompt_tokens,
        max_concurrency=args.max_concurrency,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), **manifest["summary"]}, indent=2))


if __name__ == "__main__":
    main()
