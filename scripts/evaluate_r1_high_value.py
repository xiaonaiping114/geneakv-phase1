#!/usr/bin/env python3
"""Evaluate a rule-based R1 ranking over R0 HIGH candidates.

The input is ``high-value-decisions.jsonl`` produced by
``analyze_high_value_traces.py``.  R1 deliberately uses only fields visible at
the current decision plus outcomes of *earlier* transitions in the same
stream.  The next transition of the current row is reserved for offline labels.

This is an admission-ranking experiment, not a CPU-cache residency simulator.
It answers: with the same cumulative Store block budget, which online ranking
retains the most near-term reusable blocks?
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


DEFAULT_BUDGETS = (0.1, 0.2, 0.4, 0.6, 0.8, 1.0)


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def enrich_online_features(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach R1 scores without reading the current row's future label.

    Rows emitted by the R0 analyzer are ordered within each stream.  When row
    ``i`` is scored, row ``i-1``'s transition to ``i`` has already happened and
    is therefore legitimate online history.
    """

    by_stream: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_stream[str(row["stream_id"])].append(row)

    enriched: list[dict[str, Any]] = []
    for stream_rows in by_stream.values():
        stream_rows.sort(key=lambda row: int(row["request_index"]))
        reuse_ema: float | None = None
        delay_ema: float | None = None
        for index, original in enumerate(stream_rows):
            # At request i, the transition i-1 -> i is already observable.
            if index:
                previous = stream_rows[index - 1]
                observed_ratio = float(previous.get("next_lcp_ratio") or 0.0)
                observed_delay = previous.get("next_delay_seconds")
                reuse_ema = (
                    observed_ratio
                    if reuse_ema is None
                    else 0.7 * reuse_ema + 0.3 * observed_ratio
                )
                if observed_delay is not None:
                    observed_delay = max(0.0, float(observed_delay))
                    delay_ema = (
                        observed_delay
                        if delay_ema is None
                        else 0.7 * delay_ema + 0.3 * observed_delay
                    )

            row = dict(original)
            current_blocks = max(0, int(row.get("current_blocks") or 0))
            input_types = set(row.get("input_types") or [])

            # Conservative cold-start priors; no fitted model is involved.
            historical_reuse = 0.75 if reuse_ema is None else reuse_ema
            historical_delay = 60.0 if delay_ema is None else delay_ema
            urgency = math.exp(-min(historical_delay, 3600.0) / 900.0)
            tool_chain_bonus = 1.10 if "tool_result" in input_types else 1.0
            size_tiebreaker = 1.0 + 0.02 * math.log2(1.0 + current_blocks)

            # A value-density score: predicted reusable fraction × urgency.
            # It ranks candidates under an equal cumulative Store-block budget.
            score = (
                (0.35 + 0.65 * historical_reuse)
                * urgency
                * tool_chain_bonus
                * size_tiebreaker
            )
            row.update(
                {
                    "r1_score": score,
                    "r1_history_reuse_ema": historical_reuse,
                    "r1_history_delay_ema": historical_delay,
                    "r1_tool_chain": "tool_result" in input_types,
                }
            )
            enriched.append(row)
    return enriched


def actual_utility(row: dict[str, Any], horizon_seconds: float) -> int:
    delay = row.get("next_delay_seconds")
    if delay is None or float(delay) > horizon_seconds:
        return 0
    return max(0, int(row.get("next_lcp_blocks") or 0))


def select_under_budget(
    candidates: list[dict[str, Any]],
    budget_blocks: int,
    key,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used = 0
    for row in sorted(candidates, key=key, reverse=True):
        cost = max(1, int(row["current_blocks"]))
        if used + cost > budget_blocks:
            continue
        selected.append(row)
        used += cost
    return selected


def summarize_selection(
    selected: list[dict[str, Any]],
    all_candidates: list[dict[str, Any]],
    horizon_seconds: float,
    policy: str,
    budget_fraction: float,
    budget_blocks: int,
) -> dict[str, Any]:
    stored = sum(max(1, int(row["current_blocks"])) for row in selected)
    useful = sum(actual_utility(row, horizon_seconds) for row in selected)
    oracle_total = sum(actual_utility(row, horizon_seconds) for row in all_candidates)
    useful_nodes = sum(actual_utility(row, horizon_seconds) > 0 for row in selected)
    return {
        "policy": policy,
        "budget_fraction": budget_fraction,
        "budget_blocks": budget_blocks,
        "selected_nodes": len(selected),
        "stored_blocks": stored,
        "useful_reuse_blocks": useful,
        "useful_block_recall": safe_div(useful, oracle_total),
        "store_efficiency": safe_div(useful, stored),
        "useful_node_precision": safe_div(useful_nodes, len(selected)),
    }


def evaluate(
    rows: list[dict[str, Any]],
    horizon_seconds: float,
    budget_fractions: tuple[float, ...],
    random_seeds: int,
) -> dict[str, Any]:
    enriched = enrich_online_features(rows)
    candidates = [row for row in enriched if bool(row.get("r0_high"))]
    total_store_blocks = sum(max(1, int(row["current_blocks"])) for row in candidates)
    oracle_total = sum(actual_utility(row, horizon_seconds) for row in candidates)
    result_rows: list[dict[str, Any]] = []

    for fraction in budget_fractions:
        budget = max(1, int(total_store_blocks * fraction))
        policies = {
            "R1-online": lambda row: (
                float(row["r1_score"]),
                int(row["current_blocks"]),
                str(row["stream_id"]),
            ),
            "Size-only": lambda row: (
                int(row["current_blocks"]),
                str(row["stream_id"]),
            ),
            "Oracle": lambda row: (
                safe_div(actual_utility(row, horizon_seconds), max(1, int(row["current_blocks"]))),
                actual_utility(row, horizon_seconds),
            ),
        }
        for name, key in policies.items():
            selected = select_under_budget(candidates, budget, key)
            result_rows.append(
                summarize_selection(
                    selected, candidates, horizon_seconds, name, fraction, budget
                )
            )

        random_metrics: list[dict[str, Any]] = []
        for seed in range(random_seeds):
            shuffled = list(candidates)
            random.Random(seed).shuffle(shuffled)
            rank = {id(row): -index for index, row in enumerate(shuffled)}
            selected = select_under_budget(
                candidates, budget, lambda row, ranks=rank: ranks[id(row)]
            )
            random_metrics.append(
                summarize_selection(
                    selected,
                    candidates,
                    horizon_seconds,
                    "R0-random",
                    fraction,
                    budget,
                )
            )
        averaged = {
            key: fmean(float(item[key]) for item in random_metrics)
            for key in (
                "selected_nodes",
                "stored_blocks",
                "useful_reuse_blocks",
                "useful_block_recall",
                "store_efficiency",
                "useful_node_precision",
            )
        }
        averaged.update(
            {
                "policy": "R0-random",
                "budget_fraction": fraction,
                "budget_blocks": budget,
            }
        )
        result_rows.append(averaged)

    return {
        "configuration": {
            "horizon_seconds": horizon_seconds,
            "budget_fractions": list(budget_fractions),
            "random_seeds": random_seeds,
            "online_features": [
                "current_blocks",
                "current_input_types",
                "past_transition_lcp_ratio_ema",
                "past_transition_delay_ema",
            ],
            "offline_label": "current row next_lcp_blocks within horizon",
        },
        "population": {
            "all_rows": len(rows),
            "r0_candidates": len(candidates),
            "candidate_store_blocks": total_store_blocks,
            "oracle_reuse_blocks": oracle_total,
        },
        "results": result_rows,
        "notes": [
            "R1 never reads the current row's next_* fields while scoring it.",
            "Oracle reads future reuse and is an unreachable upper bound.",
            "This is an equal Store-block admission-budget experiment, not a residency simulation.",
        ],
    }


def render_html(summary: dict[str, Any]) -> str:
    rows = summary["results"]
    policies = ["R0-random", "Size-only", "R1-online", "Oracle"]
    colors = {
        "R0-random": "#94a3b8",
        "Size-only": "#f59e0b",
        "R1-online": "#0f9f85",
        "Oracle": "#6366f1",
    }
    width, height = 920, 410
    left, top, chart_w, chart_h = 70, 35, 790, 300
    svg = [f'<svg viewBox="0 0 {width} {height}" role="img">']
    for tick in range(0, 6):
        value = tick / 5
        y = top + chart_h * (1 - value)
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+chart_w}" y2="{y:.1f}" stroke="#dbe4ee"/>')
        svg.append(f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end">{value:.0%}</text>')
    for policy in policies:
        points = []
        for item in sorted((r for r in rows if r["policy"] == policy), key=lambda r: r["budget_fraction"]):
            x = left + chart_w * float(item["budget_fraction"])
            y = top + chart_h * (1 - float(item["useful_block_recall"]))
            points.append(f"{x:.1f},{y:.1f}")
        svg.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[policy]}" stroke-width="4"/>')
    svg.append(f'<text x="{left+chart_w/2}" y="385" text-anchor="middle">Store Block Budget / R0 全量候选</text>')
    svg.append(f'<text x="18" y="{top+chart_h/2}" transform="rotate(-90 18 {top+chart_h/2})" text-anchor="middle">有效复用 Block Recall</text>')
    svg.append('</svg>')
    legend = "".join(
        f'<span><i style="background:{colors[p]}"></i>{p}</span>' for p in policies
    )
    body_rows = "".join(
        "<tr>" + "".join(
            f"<td>{value}</td>" for value in (
                row["policy"],
                f'{row["budget_fraction"]:.0%}',
                f'{row["stored_blocks"]:.0f}',
                f'{row["useful_reuse_blocks"]:.0f}',
                f'{row["useful_block_recall"]:.2%}',
                f'{row["store_efficiency"]:.3f}',
            )
        ) + "</tr>" for row in rows
    )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>GeneaKV R1 HIGH_VALUE 离线排序实验</title><style>
body{{margin:0;background:#f6f8fb;color:#172033;font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1180px;margin:30px auto;padding:0 24px}} .card{{background:#fff;border:1px solid #dbe4ee;border-radius:16px;padding:24px;margin:18px 0;box-shadow:0 8px 26px #1e293b0d}}
h1{{margin:0 0 8px}} .legend span{{margin-right:22px}} .legend i{{display:inline-block;width:18px;height:4px;margin:0 7px 3px 0}}
svg{{width:100%;height:auto}} svg text{{font-size:13px;fill:#526070}} table{{width:100%;border-collapse:collapse}} th,td{{padding:9px 12px;border-bottom:1px solid #e6edf4;text-align:right}} th:first-child,td:first-child{{text-align:left}}
.note{{background:#ecfdf5;border-left:4px solid #0f9f85;padding:12px 16px}}
</style></head><body><main><div class="card"><h1>GeneaKV R1 HIGH_VALUE 离线排序实验</h1>
<p>相同累计 Store Block 预算下，比较 R0 随机准入、仅按大小、只用在线历史特征的 R1，以及使用未来真实复用信息的 Oracle 上界。</p>
<div class="note">R1 打分不读取当前节点的 next_* 字段；未来复用只用于离线验收。</div></div>
<div class="card"><div class="legend">{legend}</div>{''.join(svg)}</div>
<div class="card"><table><thead><tr><th>策略</th><th>预算</th><th>Store Blocks</th><th>有效复用 Blocks</th><th>复用 Recall</th><th>Store 效率</th></tr></thead><tbody>{body_rows}</tbody></table></div>
</main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon-seconds", type=float, default=600.0)
    parser.add_argument("--budget-fractions", type=float, nargs="+", default=DEFAULT_BUDGETS)
    parser.add_argument("--random-seeds", type=int, default=20)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.decisions.read_text().splitlines() if line.strip()]
    summary = evaluate(
        rows,
        horizon_seconds=args.horizon_seconds,
        budget_fractions=tuple(args.budget_fractions),
        random_seeds=args.random_seeds,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "r1-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    with (args.output_dir / "r1-results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary["results"][0]))
        writer.writeheader()
        writer.writerows(summary["results"])
    (args.output_dir / "GeneaKV-R1离线排序实验.html").write_text(
        render_html(summary)
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
