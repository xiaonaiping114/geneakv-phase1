#!/usr/bin/env python3
"""Aggregate the controlled capacity sweep and render a self-contained HTML."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def cacheable_resume_tokens(data: dict[str, Any], block_size: int) -> int:
    requests = data["phases"]["resume_high"]["requests"]
    return sum((int(row["prompt_tokens"]) // block_size) * block_size for row in requests)


def aggregate(result_dir: Path, block_size: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, float]]] = defaultdict(list)
    for path in sorted(result_dir.glob("*.json")):
        if path.name == "summary.json":
            continue
        data = json.loads(path.read_text())
        stem = path.stem
        if "cap" not in stem:
            continue
        capacity = int(stem.split("cap", 1)[1].split("-", 1)[0])
        metrics = data["cache_metrics"]
        resume = data["phases"]["resume_high"]["summary"]
        target = cacheable_resume_tokens(data, block_size)
        external_hits = float(metrics["external_hits"])
        groups[(str(data["mode"]), capacity)].append(
            {
                "external_hits": external_hits,
                "resume_coverage": external_hits / target if target else 0.0,
                "overall_hit_rate": float(metrics["effective_hit_rate"]),
                "resume_mean_ms": float(resume["mean_seconds"]) * 1000,
                "resume_p95_ms": float(resume["p95_seconds"]) * 1000,
            }
        )
    rows = []
    for (mode, capacity), items in sorted(groups.items(), key=lambda item: (item[0][1], item[0][0])):
        rows.append(
            {
                "mode": mode,
                "capacity_gib": capacity,
                "rounds": len(items),
                **{
                    key: statistics.fmean(float(item[key]) for item in items)
                    for key in items[0]
                },
            }
        )
    return rows


def chart(rows: list[dict[str, Any]], metric: str, title: str, percent: bool) -> str:
    width, height = 800, 310
    left, top, chart_w, chart_h = 62, 34, 680, 220
    capacities = sorted({int(row["capacity_gib"]) for row in rows})
    maximum = max((float(row[metric]) for row in rows), default=1.0) or 1.0
    if percent:
        maximum = max(1.0, maximum)
    colors = {"eager": "#f59e0b", "geneakv": "#0f9f85"}
    parts = [f'<h3>{title}</h3><svg viewBox="0 0 {width} {height}">']
    for tick in range(6):
        value = maximum * tick / 5
        y = top + chart_h * (1 - tick / 5)
        label = f"{value:.0%}" if percent else f"{value:.0f}"
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+chart_w}" y2="{y:.1f}" stroke="#dbe4ee"/>')
        parts.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end">{label}</text>')
    for index, capacity in enumerate(capacities):
        x = left if len(capacities) == 1 else left + chart_w * index / (len(capacities) - 1)
        parts.append(f'<text x="{x:.1f}" y="{top+chart_h+25}" text-anchor="middle">{capacity}G</text>')
    for mode in ("eager", "geneakv"):
        values = {int(row["capacity_gib"]): float(row[metric]) for row in rows if row["mode"] == mode}
        points = []
        for index, capacity in enumerate(capacities):
            if capacity not in values:
                continue
            x = left if len(capacities) == 1 else left + chart_w * index / (len(capacities) - 1)
            y = top + chart_h * (1 - values[capacity] / maximum)
            points.append((x, y))
        parts.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x,y in points)}" fill="none" stroke="{colors[mode]}" stroke-width="4"/>')
        for x, y in points:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{colors[mode]}"/>')
    parts.append('</svg>')
    return "".join(parts)


def render(rows: list[dict[str, Any]]) -> str:
    body = "".join(
        f"<tr><td>{r['mode']}</td><td>{r['capacity_gib']} GiB</td><td>{r['rounds']}</td>"
        f"<td>{r['external_hits']:.0f}</td><td>{r['resume_coverage']:.2%}</td>"
        f"<td>{r['overall_hit_rate']:.3%}</td><td>{r['resume_mean_ms']:.3f} ms</td>"
        f"<td>{r['resume_p95_ms']:.3f} ms</td></tr>" for r in rows
    )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>GeneaKV CPU KV 容量收益曲线</title><style>
body{{margin:0;background:#f5f8fb;color:#172033;font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}main{{max-width:1200px;margin:28px auto;padding:0 22px}}.card{{background:#fff;border:1px solid #dbe4ee;border-radius:16px;padding:22px;margin:16px 0;box-shadow:0 8px 24px #1e293b0d}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}h1,h3{{margin-top:0}}svg{{width:100%;height:auto}}svg text{{fill:#526070;font-size:12px}}.legend span{{margin-right:24px}}.dot{{display:inline-block;width:12px;height:12px;border-radius:50%;margin-right:6px}}table{{width:100%;border-collapse:collapse}}th,td{{padding:9px 10px;border-bottom:1px solid #e5edf5;text-align:right}}th:first-child,td:first-child{{text-align:left}}.note{{border-left:4px solid #0f9f85;background:#ecfdf5;padding:12px 16px}}@media(max-width:850px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main><section class="card"><h1>GeneaKV CPU KV 容量收益曲线</h1><p>自然驱逐工作负载：8 HIGH → 12 LOW → Resume 8 HIGH。</p><div class="note">Resume Cache Coverage 仅以恢复阶段可缓存的完整 Block 为分母，避免不同 run_id 的少量 Token 差异污染结论。</div><p class="legend"><span><i class="dot" style="background:#f59e0b"></i>Eager</span><span><i class="dot" style="background:#0f9f85"></i>GeneaKV</span></p></section><section class="grid"><div class="card">{chart(rows,'resume_coverage','HIGH Resume Cache Coverage',True)}</div><div class="card">{chart(rows,'resume_mean_ms','Resume Mean Latency (ms)',False)}</div></section><section class="card"><table><thead><tr><th>策略</th><th>CPU KV</th><th>轮数</th><th>命中 Token</th><th>Resume Coverage</th><th>全流程命中率</th><th>Resume Mean</th><th>Resume P95</th></tr></thead><tbody>{body}</tbody></table></section></main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=128)
    args = parser.parse_args()
    rows = aggregate(args.result_dir, args.block_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "capacity-summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    with (args.output_dir / "capacity-summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (args.output_dir / "GeneaKV-CPU容量收益曲线.html").write_text(render(rows))
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
