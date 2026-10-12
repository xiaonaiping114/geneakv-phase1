#!/usr/bin/env python3
"""Dynamic HBM/CPU KV-cache replay over real Coding Agent traces.

The public traces contain real request timing and anonymous prefix hash IDs but
not prompt text.  This simulator therefore replays the real lineage/timing at a
fixed chunk granularity.  It models QUERY, response completion/STORE, HBM LRU,
CPU TTL, CPU LRU or value-aware admission/eviction, and CPU-to-HBM promotion.

It is intentionally an offline cache-lifecycle simulator.  Transfer bandwidth,
NPU execution and queueing latency are measured in the later online replay.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from scripts.analyze_high_value_traces import iter_streams, lcp_blocks


@dataclass
class CacheEntry:
    size: int
    last_access: float
    priority: float
    expires_at: float
    version: int = 0
    ever_hit: bool = False


class TierCache:
    def __init__(self, capacity_blocks: int, eviction: str, ttl_seconds: float):
        self.capacity = max(1, capacity_blocks)
        self.eviction = eviction
        self.ttl = ttl_seconds
        self.entries: dict[tuple[Any, ...], CacheEntry] = {}
        self.used = self.peak = 0
        self.evicted_blocks = self.expired_blocks = 0
        self.unused_evicted_blocks = 0
        self._lru_heap: list[tuple[float, int, tuple[Any, ...]]] = []
        self._value_heap: list[tuple[float, float, int, tuple[Any, ...]]] = []
        self._expiry_heap: list[tuple[float, int, tuple[Any, ...]]] = []

    def _push(self, key: tuple[Any, ...], entry: CacheEntry) -> None:
        heapq.heappush(self._lru_heap, (entry.last_access, entry.version, key))
        heapq.heappush(
            self._value_heap,
            (entry.priority, entry.last_access, entry.version, key),
        )
        if math.isfinite(entry.expires_at):
            heapq.heappush(
                self._expiry_heap, (entry.expires_at, entry.version, key)
            )

    def _remove(self, key: tuple[Any, ...], reason: str) -> int:
        entry = self.entries.pop(key, None)
        if entry is None:
            return 0
        self.used -= entry.size
        if reason == "ttl":
            self.expired_blocks += entry.size
        else:
            self.evicted_blocks += entry.size
        if not entry.ever_hit:
            self.unused_evicted_blocks += entry.size
        return entry.size

    def expire(self, now: float) -> None:
        while self._expiry_heap and self._expiry_heap[0][0] <= now:
            expires_at, version, key = heapq.heappop(self._expiry_heap)
            entry = self.entries.get(key)
            if (
                entry is not None
                and entry.version == version
                and entry.expires_at == expires_at
            ):
                self._remove(key, "ttl")

    def contains(self, key: tuple[Any, ...], now: float, touch: bool = True) -> bool:
        entry = self.entries.get(key)
        if entry is None:
            return False
        if entry.expires_at <= now:
            self._remove(key, "ttl")
            return False
        if touch:
            entry.ever_hit = True
            entry.last_access = now
            entry.version += 1
            self._push(key, entry)
        return True

    def insert(
        self,
        key: tuple[Any, ...],
        size: int,
        now: float,
        priority: float,
    ) -> tuple[int, int]:
        """Return (newly admitted blocks, immediately rejected blocks)."""
        self.expire(now)
        entry = self.entries.get(key)
        expires_at = now + self.ttl if self.ttl > 0 else math.inf
        if entry is not None:
            entry.last_access = now
            entry.expires_at = expires_at
            entry.priority = max(entry.priority, priority)
            entry.version += 1
            self._push(key, entry)
            return 0, 0

        entry = CacheEntry(size, now, priority, expires_at)
        self.entries[key] = entry
        self.used += size
        self._push(key, entry)
        self._evict_to_capacity()
        self.peak = max(self.peak, self.used)
        if key in self.entries:
            return size, 0
        return 0, size

    def _evict_to_capacity(self) -> None:
        heap = self._lru_heap if self.eviction == "lru" else self._value_heap
        while self.used > self.capacity and heap:
            item = heapq.heappop(heap)
            if self.eviction == "lru":
                _, version, key = item
            else:
                _, _, version, key = item
            entry = self.entries.get(key)
            if entry is None or entry.version != version:
                continue
            self._remove(key, "capacity")

    def unused_resident_blocks(self) -> int:
        return sum(entry.size for entry in self.entries.values() if not entry.ever_hit)


@dataclass(frozen=True)
class Event:
    time: float
    order: int
    kind: str
    chunks: tuple[tuple[tuple[Any, ...], int], ...]
    prompt_blocks: int
    stop: str
    input_types: tuple[str, ...]
    r1_score: float
    oracle_priority: float


def chunk_hashes(
    trace_id: str, hashes: list[Any], chunk_blocks: int
) -> tuple[tuple[tuple[Any, ...], int], ...]:
    chunks = []
    for start in range(0, len(hashes), chunk_blocks):
        values = hashes[start : start + chunk_blocks]
        if not values:
            continue
        # hash_id_scope is local to a trace.  Position plus the final anonymous
        # block ID provides a compact, stable identity for a prefix chunk.
        key = (trace_id, start // chunk_blocks, values[-1], len(values))
        chunks.append((key, len(values)))
    return tuple(chunks)


def online_r1_score(
    current_blocks: int,
    input_types: Iterable[str],
    reuse_ema: float | None,
    delay_ema: float | None,
) -> float:
    historical_reuse = 0.75 if reuse_ema is None else reuse_ema
    historical_delay = 60.0 if delay_ema is None else delay_ema
    urgency = math.exp(-min(historical_delay, 3600.0) / 900.0)
    tool_chain = 1.10 if "tool_result" in set(input_types) else 1.0
    size_tiebreaker = 1.0 + 0.02 * math.log2(1.0 + max(0, current_blocks))
    return (
        (0.35 + 0.65 * historical_reuse)
        * urgency
        * tool_chain
        * size_tiebreaker
    )


def build_trace_events(
    trace: dict[str, Any],
    start_time: float,
    chunk_blocks: int,
    oracle_horizon: float,
) -> tuple[list[Event], int, float]:
    events: list[Event] = []
    unique_chunks: dict[tuple[Any, ...], int] = {}
    max_time = start_time
    for stream in iter_streams(trace):
        reuse_ema: float | None = None
        delay_ema: float | None = None
        requests = stream.requests
        for index, current in enumerate(requests):
            hashes = list(current.get("hash_ids", []))
            if index:
                previous = requests[index - 1]
                previous_hashes = list(previous.get("hash_ids", []))
                ratio = lcp_blocks(previous_hashes, hashes) / max(1, len(previous_hashes))
                previous_time = float(previous.get("t", 0.0))
                current_time = float(current.get("t", 0.0))
                observed_delay = max(0.0, current_time - previous_time)
                reuse_ema = ratio if reuse_ema is None else 0.7 * reuse_ema + 0.3 * ratio
                delay_ema = (
                    observed_delay
                    if delay_ema is None
                    else 0.7 * delay_ema + 0.3 * observed_delay
                )
            chunks = chunk_hashes(stream.trace_id, hashes, chunk_blocks)
            for key, size in chunks:
                unique_chunks[key] = size
            request_time = start_time + stream.start_time + float(current.get("t", 0.0))
            response_time = request_time + max(0.0, float(current.get("api_time", 0.0)))
            r1_score = online_r1_score(
                len(hashes), current.get("input_types", []), reuse_ema, delay_ema
            )

            next_request = requests[index + 1] if index + 1 < len(requests) else None
            oracle_blocks = 0
            if next_request is not None:
                delay = max(
                    0.0,
                    float(next_request.get("t", 0.0))
                    - float(current.get("t", 0.0)),
                )
                if delay <= oracle_horizon:
                    oracle_blocks = lcp_blocks(
                        hashes, list(next_request.get("hash_ids", []))
                    )
            oracle_priority = oracle_blocks / max(1, len(hashes))
            common = dict(
                chunks=chunks,
                prompt_blocks=len(hashes),
                stop=str(current.get("stop") or ""),
                input_types=tuple(current.get("input_types", [])),
                r1_score=r1_score,
                oracle_priority=oracle_priority,
            )
            # A request must query the cache before its own response can Store.
            events.append(Event(request_time, 0, "query", **common))
            events.append(Event(response_time, 1, "store", **common))
            max_time = max(max_time, response_time)
    return events, sum(unique_chunks.values()), max_time


class PolicySimulation:
    def __init__(
        self,
        name: str,
        mode: str,
        hbm_capacity: int,
        cpu_capacity: int,
        ttl_seconds: float,
    ):
        self.name = name
        self.mode = mode
        self.hbm = TierCache(hbm_capacity, "lru", 0)
        cpu_eviction = "value" if mode in {"r1", "future"} else "lru"
        self.cpu = TierCache(cpu_capacity, cpu_eviction, ttl_seconds)
        self.stats: defaultdict[str, float] = defaultdict(float)

    def process(self, event: Event) -> None:
        if event.kind == "query":
            self._query(event)
        else:
            self._store(event)

    def _query(self, event: Event) -> None:
        self.stats["requests"] += 1
        self.stats["prompt_blocks"] += event.prompt_blocks
        is_resume = "tool_result" in event.input_types
        if is_resume:
            self.stats["resume_requests"] += 1
            self.stats["resume_prompt_blocks"] += event.prompt_blocks

        hbm_hits = cpu_hits = 0
        prefix_open = True
        for key, size in event.chunks:
            if not prefix_open:
                break
            if self.hbm.contains(key, event.time):
                hbm_hits += size
            elif self.cpu.contains(key, event.time):
                cpu_hits += size
                self.hbm.insert(key, size, event.time, 0.0)
            else:
                prefix_open = False
        compute = max(0, event.prompt_blocks - hbm_hits - cpu_hits)
        self.stats["hbm_hit_blocks"] += hbm_hits
        self.stats["cpu_hit_blocks"] += cpu_hits
        self.stats["compute_blocks"] += compute
        if is_resume:
            self.stats["resume_hbm_hit_blocks"] += hbm_hits
            self.stats["resume_cpu_hit_blocks"] += cpu_hits
            self.stats["resume_compute_blocks"] += compute

    def _store(self, event: Event) -> None:
        for key, size in event.chunks:
            self.hbm.insert(key, size, event.time, 0.0)

        should_store = False
        priority = 0.0
        if self.mode == "eager":
            should_store = True
        elif self.mode == "r0":
            should_store = event.stop == "tool_use"
        elif self.mode == "r1":
            should_store = event.stop == "tool_use"
            priority = event.r1_score
        elif self.mode == "future":
            should_store = event.stop == "tool_use" and event.oracle_priority > 0
            priority = event.oracle_priority
        if not should_store:
            return

        for position, (key, size) in enumerate(event.chunks):
            # Small prefix bonus makes value ties deterministic and preserves
            # shared ancestors before request-specific suffixes.
            prefix_bonus = 0.02 * (1.0 - position / max(1, len(event.chunks)))
            self.stats["cpu_store_requested_blocks"] += size
            admitted, rejected = self.cpu.insert(
                key, size, event.time, priority + prefix_bonus
            )
            self.stats["cpu_store_admitted_blocks"] += admitted
            self.stats["cpu_store_rejected_blocks"] += rejected

    def result(self, capacity_fraction: float) -> dict[str, Any]:
        s = self.stats
        return {
            "policy": self.name,
            "capacity_fraction": capacity_fraction,
            "hbm_capacity_blocks": self.hbm.capacity,
            "cpu_capacity_blocks": self.cpu.capacity,
            **{key: value for key, value in sorted(s.items())},
            "overall_cpu_hit_rate": s["cpu_hit_blocks"] / max(1, s["prompt_blocks"]),
            "resume_cpu_coverage": s["resume_cpu_hit_blocks"] / max(1, s["resume_prompt_blocks"]),
            "resume_total_coverage": (
                s["resume_cpu_hit_blocks"] + s["resume_hbm_hit_blocks"]
            ) / max(1, s["resume_prompt_blocks"]),
            "cpu_peak_blocks": self.cpu.peak,
            "cpu_evicted_blocks": self.cpu.evicted_blocks,
            "cpu_expired_blocks": self.cpu.expired_blocks,
            "cpu_unused_evicted_blocks": self.cpu.unused_evicted_blocks,
            "cpu_unused_resident_blocks": self.cpu.unused_resident_blocks(),
        }


def valid_trace_paths(trace_dir: Path) -> list[Path]:
    paths = []
    for path in sorted(trace_dir.rglob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and isinstance(data.get("requests"), list):
            paths.append(path)
    return paths


def load_batch(
    paths: list[Path],
    base_time: float,
    chunk_blocks: int,
    oracle_horizon: float,
) -> tuple[list[Event], int, float]:
    events: list[Event] = []
    unique: dict[tuple[Any, ...], int] = {}
    max_time = base_time
    for path in paths:
        trace = json.loads(path.read_text())
        trace_events, _, trace_max = build_trace_events(
            trace, base_time, chunk_blocks, oracle_horizon
        )
        events.extend(trace_events)
        for event in trace_events:
            if event.kind == "query":
                for key, size in event.chunks:
                    unique[key] = size
        max_time = max(max_time, trace_max)
    events.sort(key=lambda event: (event.time, event.order))
    return events, sum(unique.values()), max_time


def calculate_working_set(
    paths: list[Path], concurrency: int, chunk_blocks: int, horizon: float
) -> int:
    maximum = 1
    for start in range(0, len(paths), concurrency):
        _, blocks, _ = load_batch(
            paths[start : start + concurrency], 0.0, chunk_blocks, horizon
        )
        maximum = max(maximum, blocks)
    return maximum


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = valid_trace_paths(args.trace_dir)
    working_set = calculate_working_set(
        paths, args.concurrency, args.chunk_blocks, args.oracle_horizon_seconds
    )
    hbm_capacity = max(1, int(working_set * args.hbm_fraction))
    simulations: list[tuple[float, PolicySimulation]] = []
    for fraction in args.cpu_fractions:
        cpu_capacity = max(1, int(working_set * fraction))
        for name, mode in (
            ("Eager-LRU", "eager"),
            ("R0-LRU", "r0"),
            ("R1-Value", "r1"),
            ("Future-NextReuse", "future"),
        ):
            simulations.append(
                (
                    fraction,
                    PolicySimulation(
                        name,
                        mode,
                        hbm_capacity,
                        cpu_capacity,
                        args.ttl_seconds,
                    ),
                )
            )

    base_time = 0.0
    for start in range(0, len(paths), args.concurrency):
        batch = paths[start : start + args.concurrency]
        events, _, max_time = load_batch(
            batch,
            base_time,
            args.chunk_blocks,
            args.oracle_horizon_seconds,
        )
        for event in events:
            for _, simulation in simulations:
                simulation.process(event)
        base_time = max_time + args.batch_gap_seconds

    return {
        "configuration": {
            "trace_dir": str(args.trace_dir),
            "trace_files": len(paths),
            "concurrency": args.concurrency,
            "chunk_blocks": args.chunk_blocks,
            "hbm_fraction": args.hbm_fraction,
            "cpu_fractions": args.cpu_fractions,
            "ttl_seconds": args.ttl_seconds,
            "oracle_horizon_seconds": args.oracle_horizon_seconds,
            "batch_gap_seconds": args.batch_gap_seconds,
            "max_batch_unique_blocks": working_set,
        },
        "results": [
            simulation.result(fraction) for fraction, simulation in simulations
        ],
        "limitations": [
            "Real per-session timing/lineage is preserved; cross-session starts are controlled batching because global arrival timestamps are unavailable.",
            "Anonymous hash IDs are replayed at fixed chunk granularity; original prompt text is unavailable.",
            "The simulator models cache blocks, TTL and eviction, not D2H/H2D bandwidth or NPU queue latency.",
        ],
    }


def render_html(summary: dict[str, Any]) -> str:
    results = summary["results"]
    policies = ["Eager-LRU", "R0-LRU", "R1-Value", "Future-NextReuse"]
    colors = {
        "Eager-LRU": "#f59e0b",
        "R0-LRU": "#94a3b8",
        "R1-Value": "#0f9f85",
        "Future-NextReuse": "#6366f1",
    }

    def svg(metric: str, title: str) -> str:
        width, height = 850, 330
        left, top, cw, ch = 65, 35, 720, 230
        fractions = sorted({float(row["capacity_fraction"]) for row in results})
        max_value = max(1e-12, max(float(row[metric]) for row in results))
        parts = [f'<h3>{title}</h3><svg viewBox="0 0 {width} {height}">']
        for tick in range(6):
            y = top + ch * (1 - tick / 5)
            value = max_value * tick / 5
            parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+cw}" y2="{y:.1f}" stroke="#dbe4ee"/>')
            parts.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end">{value:.1%}</text>')
        for index, fraction in enumerate(fractions):
            x = left if len(fractions) == 1 else left + cw * index / (len(fractions)-1)
            parts.append(f'<text x="{x:.1f}" y="{top+ch+25}" text-anchor="middle">{fraction:.0%}</text>')
        for policy in policies:
            rows = sorted((r for r in results if r["policy"] == policy), key=lambda r:r["capacity_fraction"])
            pts=[]
            for index,row in enumerate(rows):
                x = left if len(rows)==1 else left+cw*index/(len(rows)-1)
                y = top+ch*(1-float(row[metric])/max_value)
                pts.append((x,y))
            parts.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x,y in pts)}" fill="none" stroke="{colors[policy]}" stroke-width="4"/>')
            for x,y in pts: parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{colors[policy]}"/>')
        parts.append('</svg>')
        return ''.join(parts)

    rows_html = ''.join(
        f"<tr><td>{r['policy']}</td><td>{r['capacity_fraction']:.0%}</td>"
        f"<td>{r['resume_cpu_coverage']:.3%}</td><td>{r['overall_cpu_hit_rate']:.3%}</td>"
        f"<td>{r['compute_blocks']:.0f}</td><td>{r['cpu_unused_evicted_blocks']:.0f}</td>"
        f"<td>{r['cpu_peak_blocks']:.0f}</td></tr>" for r in results
    )
    legend=''.join(f'<span><i style="background:{colors[p]}"></i>{p}</span>' for p in policies)
    cfg=summary['configuration']
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>GeneaKV真实Trace动态缓存实验</title><style>
body{{margin:0;background:#f5f8fb;color:#172033;font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}main{{max-width:1240px;margin:28px auto;padding:0 22px}}.card{{background:#fff;border:1px solid #dbe4ee;border-radius:16px;padding:22px;margin:16px 0;box-shadow:0 8px 24px #1e293b0d}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}.note{{background:#eff6ff;border-left:4px solid #3b82f6;padding:12px 16px}}.legend span{{margin-right:18px}}.legend i{{display:inline-block;width:17px;height:4px;margin:0 6px 3px 0}}svg{{width:100%;height:auto}}svg text{{font-size:12px;fill:#526070}}table{{width:100%;border-collapse:collapse}}th,td{{padding:9px 10px;border-bottom:1px solid #e5edf5;text-align:right}}th:first-child,td:first-child{{text-align:left}}@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main><section class="card"><h1>GeneaKV真实Trace动态缓存实验</h1><p>Trace={cfg['trace_files']}，并发={cfg['concurrency']}，Chunk={cfg['chunk_blocks']} Blocks，TTL={cfg['ttl_seconds']}s，最大批次工作集={cfg['max_batch_unique_blocks']} Blocks。</p><div class="note">真实Session内部时间、Tool链、Sub-agent与Hash前缀保持不变；跨Session并发使用固定批次，因为数据集没有全局到达时间。</div><p class="legend">{legend}</p></section><section class="grid"><div class="card">{svg('resume_cpu_coverage','Tool Resume CPU Coverage')}</div><div class="card">{svg('overall_cpu_hit_rate','Overall CPU Hit Rate')}</div></section><section class="card"><table><thead><tr><th>策略</th><th>CPU容量/工作集</th><th>Resume CPU Coverage</th><th>Overall CPU Hit</th><th>重算Blocks</th><th>无效驱逐Blocks</th><th>CPU峰值Blocks</th></tr></thead><tbody>{rows_html}</tbody></table></section></main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--chunk-blocks", type=int, default=32)
    parser.add_argument("--hbm-fraction", type=float, default=0.02)
    parser.add_argument("--cpu-fractions", type=float, nargs="+", default=[0.05,0.1,0.2,0.4])
    parser.add_argument("--ttl-seconds", type=float, default=600.0)
    parser.add_argument("--oracle-horizon-seconds", type=float, default=600.0)
    parser.add_argument("--batch-gap-seconds", type=float, default=60.0)
    args = parser.parse_args()
    summary = run(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "dynamic-cache-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
    with (args.output_dir / "dynamic-cache-results.csv").open("w", newline="") as handle:
        writer=csv.DictWriter(handle, fieldnames=list(summary["results"][0]))
        writer.writeheader(); writer.writerows(summary["results"])
    (args.output_dir / "GeneaKV真实Trace动态缓存实验.html").write_text(render_html(summary))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
