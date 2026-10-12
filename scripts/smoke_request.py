#!/usr/bin/env python3
"""Send a deterministic long-prefix request to an OpenAI-compatible server."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--repeat", type=int, default=700)
    parser.add_argument("--suffix", default="Answer with the single word READY.")
    args = parser.parse_args()

    prompt = ("GeneaKV stable shared prefix for cache validation. " * args.repeat) + args.suffix
    body = {
        "model": "geneakv-qwen3-0.6b",
        "prompt": prompt,
        "temperature": 0,
        "max_tokens": 8,
        "stream": False,
    }
    request = urllib.request.Request(
        f"http://{args.host}:{args.port}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )

    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=300) as response:
        payload = json.load(response)
    elapsed = time.perf_counter() - started

    output = {
        "elapsed_seconds": round(elapsed, 6),
        "usage": payload.get("usage"),
        "text": payload["choices"][0].get("text", ""),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
