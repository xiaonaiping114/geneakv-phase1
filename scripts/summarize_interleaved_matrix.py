#!/usr/bin/env python3
"""Summarize the six-run interleaved capacity matrix as JSON, CSV and HTML."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from pathlib import Path
from typing import Any


NAME_RE = re.compile(r"^(eager|geneakv-r0)-cpu(\d+)g\.json$")
COLORS = {"eager": "#ef4444", "geneakv-r0": "#2563eb"}
LABELS = {"eager": "Eager", "geneakv-r0": "GeneaKV-R0"}


def load_rows(result_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(result_dir.glob("*.json")):
        match = NAME_RE.match(path.name)
        if not match:
            continue
        mode, capacity = match.groups()
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload["summary"]
        rows.append(
            {
                "mode": mode,
                "capacity_gib": int(capacity),
                "requests": int(summary["requests"]),
                "high_store_candidates": int(summary["high_store_candidates"]),
                "high_resume_requests": int(summary["high_resume_requests"]),
                "effective_hit_rate": float(summary["aggregate_effective_hit_rate"]),
                "local_hits": float(summary["local_hits"]),
                "external_hits": float(summary["external_hits"]),
                "external_restore_coverage_proxy": float(
                    summary["external_restore_coverage_proxy"]
                ),
                "benchmark_elapsed_seconds": float(
                    summary["benchmark_elapsed_seconds"]
                ),
                "request_mean_ms": float(summary["request_mean_seconds"]) * 1000,
                "request_p95_ms": float(summary["request_p95_seconds"]) * 1000,
                "resume_mean_ms": float(summary["resume_mean_seconds"]) * 1000,
                "resume_p50_ms": float(summary["resume_p50_seconds"]) * 1000,
                "resume_p95_ms": float(summary["resume_p95_seconds"]) * 1000,
                "resume_p99_ms": float(summary["resume_p99_seconds"]) * 1000,
                "throughput_rps": float(summary["throughput_requests_per_second"]),
                "source_file": path.name,
            }
        )
    rows.sort(key=lambda row: (row["capacity_gib"], row["mode"]))
    return rows


def comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["capacity_gib"], row["mode"]): row for row in rows}
    output = []
    for capacity in sorted({row["capacity_gib"] for row in rows}):
        eager = by_key.get((capacity, "eager"))
        geneakv = by_key.get((capacity, "geneakv-r0"))
        if not eager or not geneakv:
            continue
        def gain_percent(base: float, candidate: float) -> float | None:
            if base == 0:
                return None
            return 100 * (candidate - base) / base

        def reduction_percent(base: float, candidate: float) -> float:
            return 100 * (base - candidate) / max(base, 1e-12)

        output.append(
            {
                "capacity_gib": capacity,
                "eager_hit_rate": eager["effective_hit_rate"],
                "geneakv_hit_rate": geneakv["effective_hit_rate"],
                "hit_rate_lift_pp": 100
                * (geneakv["effective_hit_rate"] - eager["effective_hit_rate"]),
                "hit_rate_relative_gain_percent": gain_percent(
                    eager["effective_hit_rate"], geneakv["effective_hit_rate"]
                ),
                "eager_external_hits": eager["external_hits"],
                "geneakv_external_hits": geneakv["external_hits"],
                "external_hit_relative_gain_percent": gain_percent(
                    eager["external_hits"], geneakv["external_hits"]
                ),
                "external_hit_delta": geneakv["external_hits"]
                - eager["external_hits"],
                "eager_restore_coverage": eager["external_restore_coverage_proxy"],
                "geneakv_restore_coverage": geneakv[
                    "external_restore_coverage_proxy"
                ],
                "eager_resume_mean_ms": eager["resume_mean_ms"],
                "geneakv_resume_mean_ms": geneakv["resume_mean_ms"],
                "resume_mean_reduction_percent": reduction_percent(
                    eager["resume_mean_ms"], geneakv["resume_mean_ms"]
                ),
                "eager_resume_p50_ms": eager["resume_p50_ms"],
                "geneakv_resume_p50_ms": geneakv["resume_p50_ms"],
                "resume_p50_reduction_percent": reduction_percent(
                    eager["resume_p50_ms"], geneakv["resume_p50_ms"]
                ),
                "eager_resume_p95_ms": eager["resume_p95_ms"],
                "geneakv_resume_p95_ms": geneakv["resume_p95_ms"],
                "resume_p95_reduction_percent": reduction_percent(
                    eager["resume_p95_ms"], geneakv["resume_p95_ms"]
                ),
                "eager_resume_p99_ms": eager["resume_p99_ms"],
                "geneakv_resume_p99_ms": geneakv["resume_p99_ms"],
                "resume_p99_reduction_percent": reduction_percent(
                    eager["resume_p99_ms"], geneakv["resume_p99_ms"]
                ),
                "eager_elapsed_seconds": eager["benchmark_elapsed_seconds"],
                "geneakv_elapsed_seconds": geneakv["benchmark_elapsed_seconds"],
                "elapsed_reduction_percent": reduction_percent(
                    eager["benchmark_elapsed_seconds"],
                    geneakv["benchmark_elapsed_seconds"],
                ),
                "eager_throughput_rps": eager["throughput_rps"],
                "geneakv_throughput_rps": geneakv["throughput_rps"],
                "throughput_gain_percent": gain_percent(
                    eager["throughput_rps"], geneakv["throughput_rps"]
                ),
                "eager_request_mean_ms": eager["request_mean_ms"],
                "geneakv_request_mean_ms": geneakv["request_mean_ms"],
                "request_mean_reduction_percent": reduction_percent(
                    eager["request_mean_ms"], geneakv["request_mean_ms"]
                ),
                "eager_request_p95_ms": eager["request_p95_ms"],
                "geneakv_request_p95_ms": geneakv["request_p95_ms"],
                "request_p95_reduction_percent": reduction_percent(
                    eager["request_p95_ms"], geneakv["request_p95_ms"]
                ),
            }
        )
    return output


def svg_chart(
    rows: list[dict[str, Any]],
    metric: str,
    title: str,
    *,
    percent: bool = False,
) -> str:
    width, height = 720, 300
    left, top, chart_w, chart_h = 70, 35, 610, 210
    capacities = sorted({row["capacity_gib"] for row in rows})
    values = [float(row[metric]) for row in rows]
    maximum = max(values, default=1.0)
    if percent:
        maximum = max(maximum, 0.01)
    elif maximum <= 0:
        maximum = 1.0

    def x_pos(capacity: int) -> float:
        if len(capacities) <= 1:
            return left + chart_w / 2
        return left + capacities.index(capacity) * chart_w / (len(capacities) - 1)

    def y_pos(value: float) -> float:
        return top + chart_h * (1 - value / maximum)

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img">']
    parts.append(f'<text class="chart-title" x="{left}" y="20">{html.escape(title)}</text>')
    for tick in range(5):
        value = maximum * tick / 4
        y = y_pos(value)
        label = f"{100*value:.1f}%" if percent else f"{value:.1f}"
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+chart_w}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end">{label}</text>')
    for capacity in capacities:
        x = x_pos(capacity)
        parts.append(f'<text x="{x:.1f}" y="{top+chart_h+25}" text-anchor="middle">{capacity} GiB</text>')
    for mode in ("eager", "geneakv-r0"):
        series = [row for row in rows if row["mode"] == mode]
        series.sort(key=lambda row: row["capacity_gib"])
        points = " ".join(
            f'{x_pos(row["capacity_gib"]):.1f},{y_pos(float(row[metric])):.1f}'
            for row in series
        )
        parts.append(f'<polyline points="{points}" fill="none" stroke="{COLORS[mode]}" stroke-width="4"/>')
        for row in series:
            x = x_pos(row["capacity_gib"])
            y = y_pos(float(row[metric]))
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{COLORS[mode]}"/>')
    parts.append("</svg>")
    return "".join(parts)


def render_html(rows: list[dict[str, Any]], deltas: list[dict[str, Any]]) -> str:
    def cls(value: float, *, neutral_threshold: float = 0.01) -> str:
        if abs(value) < neutral_threshold:
            return "neutral"
        return "good" if value > 0 else "bad"

    table_rows = []
    for row in rows:
        table_rows.append(
            "<tr>"
            f"<td>{LABELS[row['mode']]}</td><td>{row['capacity_gib']} GiB</td>"
            f"<td>{100*row['effective_hit_rate']:.3f}%</td>"
            f"<td>{row['local_hits']:.0f}</td><td>{row['external_hits']:.0f}</td>"
            f"<td>{100*row['external_restore_coverage_proxy']:.3f}%</td>"
            f"<td>{row['resume_mean_ms']:.2f}</td>"
            f"<td>{row['resume_p95_ms']:.2f}</td>"
            f"<td>{row['throughput_rps']:.3f}</td></tr>"
        )
    cache_delta_rows = []
    resume_delta_rows = []
    efficiency_delta_rows = []
    for row in deltas:
        relative_hit = row["hit_rate_relative_gain_percent"]
        external_gain = row["external_hit_relative_gain_percent"]
        external_change = (
            f"新增 {row['external_hit_delta']:.0f}"
            if external_gain is None
            else f"{external_gain:+.2f}%（约 {row['geneakv_external_hits']/row['eager_external_hits']:.2f}×）"
        )
        cache_delta_rows.append(
            "<tr>"
            f"<td>{row['capacity_gib']} GiB</td>"
            f"<td>{100*row['eager_hit_rate']:.3f}%</td>"
            f"<td>{100*row['geneakv_hit_rate']:.3f}%</td>"
            f"<td class=\"good\">{row['hit_rate_lift_pp']:+.3f} pp</td>"
            f"<td class=\"good\">{relative_hit:+.2f}%（约 {row['geneakv_hit_rate']/row['eager_hit_rate']:.2f}×）</td>"
            f"<td>{row['eager_external_hits']:.0f}</td>"
            f"<td>{row['geneakv_external_hits']:.0f}</td>"
            f"<td class=\"good\">{external_change}</td>"
            f"<td>{100*row['eager_restore_coverage']:.3f}%</td>"
            f"<td class=\"good\">{100*row['geneakv_restore_coverage']:.3f}%</td></tr>"
        )
        resume_delta_rows.append(
            "<tr>"
            f"<td>{row['capacity_gib']} GiB</td>"
            f"<td>{row['eager_resume_mean_ms']/1000:.3f} s</td><td>{row['geneakv_resume_mean_ms']/1000:.3f} s</td>"
            f"<td class=\"{cls(row['resume_mean_reduction_percent'])}\">{row['resume_mean_reduction_percent']:+.2f}%</td>"
            f"<td>{row['eager_resume_p50_ms']/1000:.3f} s</td><td>{row['geneakv_resume_p50_ms']/1000:.3f} s</td>"
            f"<td class=\"{cls(row['resume_p50_reduction_percent'])}\">{row['resume_p50_reduction_percent']:+.2f}%</td>"
            f"<td>{row['eager_resume_p95_ms']/1000:.3f} s</td><td>{row['geneakv_resume_p95_ms']/1000:.3f} s</td>"
            f"<td class=\"{cls(row['resume_p95_reduction_percent'])}\">{row['resume_p95_reduction_percent']:+.2f}%</td>"
            f"<td>{row['eager_resume_p99_ms']/1000:.3f} s</td><td>{row['geneakv_resume_p99_ms']/1000:.3f} s</td>"
            f"<td class=\"{cls(row['resume_p99_reduction_percent'])}\">{row['resume_p99_reduction_percent']:+.2f}%</td></tr>"
        )
        efficiency_delta_rows.append(
            "<tr>"
            f"<td>{row['capacity_gib']} GiB</td>"
            f"<td>{row['eager_elapsed_seconds']:.1f} s</td><td>{row['geneakv_elapsed_seconds']:.1f} s</td>"
            f"<td class=\"{cls(row['elapsed_reduction_percent'])}\">{row['elapsed_reduction_percent']:+.2f}%</td>"
            f"<td>{row['eager_throughput_rps']:.4f}</td><td>{row['geneakv_throughput_rps']:.4f}</td>"
            f"<td class=\"{cls(row['throughput_gain_percent'])}\">{row['throughput_gain_percent']:+.2f}%</td>"
            f"<td>{row['eager_request_mean_ms']/1000:.3f} s</td><td>{row['geneakv_request_mean_ms']/1000:.3f} s</td>"
            f"<td class=\"{cls(row['request_mean_reduction_percent'])}\">{row['request_mean_reduction_percent']:+.2f}%</td>"
            f"<td>{row['eager_request_p95_ms']/1000:.3f} s</td><td>{row['geneakv_request_p95_ms']/1000:.3f} s</td>"
            f"<td class=\"{cls(row['request_p95_reduction_percent'])}\">{row['request_p95_reduction_percent']:+.2f}%</td></tr>"
        )
    legend = "".join(
        f'<span><i style="background:{COLORS[mode]}"></i>{LABELS[mode]}</span>'
        for mode in ("eager", "geneakv-r0")
    )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>GeneaKV 真实多 Session 交错容量实验</title><style>
body{{margin:0;background:#f5f7fb;color:#172033;font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1240px;margin:28px auto;padding:0 22px}}.card{{background:#fff;border:1px solid #dbe4ee;border-radius:16px;padding:22px;margin:16px 0;box-shadow:0 8px 24px #1e293b0d}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}h1,h2{{margin-top:0}}.note{{border-left:4px solid #2563eb;background:#eff6ff;padding:12px 16px}}
.legend span{{margin-right:24px}}.legend i{{display:inline-block;width:18px;height:5px;margin:0 7px 3px 0}}svg{{width:100%;height:auto}}svg text{{fill:#526070;font-size:12px}}svg .chart-title{{font-size:15px;font-weight:700;fill:#172033}}svg .grid{{stroke:#dbe4ee;stroke-width:1}}
table{{width:100%;border-collapse:collapse}}th,td{{padding:9px 10px;border-bottom:1px solid #e5edf5;text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}.table-wrap{{overflow-x:auto}}.metric-note{{color:#526070;font-size:14px;margin:4px 0 14px}}.good{{color:#047857;font-weight:700}}.bad{{color:#dc2626;font-weight:700}}.neutral{{color:#64748b}}@media(max-width:900px){{.grid2{{grid-template-columns:1fr}}}}
</style></head><body><main><section class="card"><h1>GeneaKV 真实多 Session 交错容量实验</h1>
<p>20 条真实 Agent Trace，每条最多 20 节点；Round-Robin 屏障批次；全局并发 8；不插入人工 LOW；单次运行中不主动清空缓存。</p>
<p class="legend">{legend}</p><div class="note">当前在线 GeneaKV 使用 R0（stop == tool_use）；Resume 时延是 max_tokens=2 的非流式端到端代理值，不冒充 wire-level TTFT。</div></section>
<section class="grid2"><div class="card">{svg_chart(rows,'effective_hit_rate','全局有效命中率',percent=True)}</div>
<div class="card">{svg_chart(rows,'resume_p95_ms','HIGH Resume P95 延迟代理（ms）')}</div></section>
<section class="grid2"><div class="card">{svg_chart(rows,'external_restore_coverage_proxy','CPU Restore Coverage Proxy',percent=True)}</div>
<div class="card">{svg_chart(rows,'throughput_rps','吞吐（requests/s）')}</div></section>
<section class="card"><h2>原始结果</h2><table><thead><tr><th>策略</th><th>CPU KV</th><th>命中率</th><th>HBM Hits</th><th>CPU Hits</th><th>CPU Coverage</th><th>Resume Mean ms</th><th>Resume P95 ms</th><th>Req/s</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></section>
<section class="card"><h2>同容量完整对比：命中与 CPU Restore</h2><p class="metric-note">命中率相对提升 =（GeneaKV 命中率 − Eager 命中率）/ Eager 命中率；pp 表示百分点。Eager 为 0 时，CPU Restore 不计算倍数，直接列新增 Token。</p><div class="table-wrap"><table><thead><tr><th>CPU KV</th><th>Eager 命中率</th><th>GeneaKV 命中率</th><th>命中率提升</th><th>命中率相对提升</th><th>Eager CPU Restore</th><th>GeneaKV CPU Restore</th><th>Restore 变化</th><th>Eager HIGH 覆盖率</th><th>GeneaKV HIGH 覆盖率</th></tr></thead><tbody>{''.join(cache_delta_rows)}</tbody></table></div></section>
<section class="card"><h2>同容量完整对比：HIGH Resume 延迟</h2><p class="metric-note">Resume 提升 =（Eager 延迟 − GeneaKV 延迟）/ Eager 延迟。正数表示 GeneaKV 更快，负数表示回退。这里是 max_tokens=2 的非流式端到端 Resume 延迟代理，不等同于 wire-level TTFT。</p><div class="table-wrap"><table><thead><tr><th>CPU KV</th><th>Eager Mean</th><th>GeneaKV Mean</th><th>Mean 提升</th><th>Eager P50</th><th>GeneaKV P50</th><th>P50 提升</th><th>Eager P95</th><th>GeneaKV P95</th><th>P95 提升</th><th>Eager P99</th><th>GeneaKV P99</th><th>P99 提升</th></tr></thead><tbody>{''.join(resume_delta_rows)}</tbody></table></div></section>
<section class="card"><h2>同容量完整对比：端到端效率</h2><p class="metric-note">总耗时与请求延迟的“提升”为降低比例；吞吐提升为增加比例。正数均表示 GeneaKV 更优。</p><div class="table-wrap"><table><thead><tr><th>CPU KV</th><th>Eager 总耗时</th><th>GeneaKV 总耗时</th><th>总耗时提升</th><th>Eager 吞吐</th><th>GeneaKV 吞吐</th><th>吞吐提升</th><th>Eager 请求 Mean</th><th>GeneaKV 请求 Mean</th><th>请求 Mean 提升</th><th>Eager 请求 P95</th><th>GeneaKV 请求 P95</th><th>请求 P95 提升</th></tr></thead><tbody>{''.join(efficiency_delta_rows)}</tbody></table></div></section>
</main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    rows = load_rows(args.result_dir)
    if not rows:
        raise SystemExit(f"no matrix result JSON files under {args.result_dir}")
    deltas = comparisons(rows)
    (args.result_dir / "matrix-summary.json").write_text(
        json.dumps({"rows": rows, "comparisons": deltas}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    with (args.result_dir / "matrix-summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.result_dir / "GeneaKV真实多Session交错容量实验.html").write_text(
        render_html(rows, deltas), encoding="utf-8"
    )
    print(json.dumps({"result_dir": str(args.result_dir), "runs": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
