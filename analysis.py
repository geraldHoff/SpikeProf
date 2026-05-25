"""
Analysis Engine
===============

Produces human-readable reports and machine-readable exports from
profiling results. All analysis functions are pure functions operating
on lists of immutable ``SpikeProfileResult`` records.
"""

from __future__ import annotations

import csv
import io
import json
import math
import statistics
from dataclasses import asdict, fields
from typing import Any, Callable, Mapping, Sequence

from models import SpikeProfileResult, results_to_csv, results_to_json
from typedecs import (
    FractionZeroOne,
    KernelCategory,
    Percentage,
)

# ── Ranked Table ────────────────────────────────────────────────────


def rank_results(
    results: Sequence[SpikeProfileResult],
    *,
    sort_by: str = "time_ms",
    ascending: bool = True,
    top_n: int = 20,
) -> list[SpikeProfileResult]:
    """Sort results by a metric and return top N.

    Excludes results with errors or NaN values in the sort field.
    Pure function.
    """
    # Filter out errored results
    valid = [
        r for r in results
        if not r.error_message and not math.isnan(getattr(r, sort_by, float("nan")))
    ]

    valid.sort(
        key=lambda r: getattr(r, sort_by, 0.0),
        reverse=not ascending,
    )

    return valid[:top_n]


# ── Parameter Effect Report ─────────────────────────────────────────


def analyze_parameter_effect(
    results: Sequence[SpikeProfileResult],
    *,
    param: str,
    metric: str,
) -> list[dict[str, Any]]:
    """Analyze the marginal effect of one parameter on a metric.

    Groups results by ``param`` and reports statistics of ``metric``
    across all other parameter combinations.

    Pure function.

    Returns:
        List of dicts with keys: param_value, mean, min, max, std, count.
    """
    # Group by parameter value
    groups: dict[Any, list[float]] = {}
    for r in results:
        if r.error_message:
            continue
        pval = getattr(r, param, None)
        mval = getattr(r, metric, None)
        if pval is None or mval is None:
            continue
        if isinstance(mval, float) and math.isnan(mval):
            continue
        groups.setdefault(pval, []).append(float(mval))

    report: list[dict[str, Any]] = []
    for pval in sorted(groups.keys()):
        vals = groups[pval]
        report.append(
            {
                "param_value": pval,
                "mean": statistics.mean(vals),
                "min": min(vals),
                "max": max(vals),
                "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                "count": len(vals),
            }
        )

    return report


# ── Component Breakdown ─────────────────────────────────────────────


def compute_component_breakdown(
    results: Sequence[SpikeProfileResult],
) -> dict[str, FractionZeroOne]:
    """Compute time fraction by kernel category.

    Pure function.
    """
    total_time = sum(r.time_ms for r in results if not r.error_message)
    if total_time <= 0.0:
        return {}

    categories: dict[str, float] = {}
    for r in results:
        if r.error_message:
            continue
        cat = r.kernel_category
        categories[cat] = categories.get(cat, 0.0) + r.time_ms

    return {
        cat: FractionZeroOne(t / total_time) for cat, t in categories.items()
    }


# ── Roofline Summary ───────────────────────────────────────────────


def roofline_summary(
    results: Sequence[SpikeProfileResult],
) -> list[dict[str, Any]]:
    """Produce roofline classification summary for all results.

    Pure function.
    """
    summary: list[dict[str, Any]] = []
    for r in results:
        if r.error_message:
            continue
        summary.append(
            {
                "kernel_name": r.kernel_name,
                "config": f"B{r.batch_size}_W{r.num_warps}_sp{r.sparsity}",
                "arithmetic_intensity": r.arithmetic_intensity,
                "roofline_class": r.roofline_class,
                "dram_total_gbs": float(r.dram_total_gbs),
                "bw_util_pct": float(r.bw_util_pct),
                "validated": r.roofline_validated,
            }
        )
    return summary


# ── Best Configuration Finder ───────────────────────────────────────


def find_best(
    results: Sequence[SpikeProfileResult],
    *,
    metric: str = "time_ms",
    minimize: bool = True,
    constraints: dict[str, tuple[str, float]] | None = None,
) -> SpikeProfileResult | None:
    """Find the best configuration by a metric, optionally with constraints.

    Args:
        results: List of profiling results.
        metric: Field name to optimize.
        minimize: True to minimize, False to maximize.
        constraints: Dict of {field: (op, value)} where op is '>=' or '<='.
            Example: {'bw_util_pct': ('>=', 60), 'sm_occupancy_pct': ('>=', 50)}

    Pure function.
    """
    valid = [r for r in results if not r.error_message]

    if constraints:
        for field_name, (op, threshold) in constraints.items():
            filtered = []
            for r in valid:
                val = getattr(r, field_name, None)
                if val is None:
                    continue
                if op == ">=" and float(val) >= threshold:
                    filtered.append(r)
                elif op == "<=" and float(val) <= threshold:
                    filtered.append(r)
            valid = filtered

    if not valid:
        return None

    return (min if minimize else max)(
        valid,
        key=lambda r: getattr(r, metric, float("inf") if minimize else float("-inf")),
    )


# ── Report Printing ────────────────────────────────────────────────


_REPORT_COLUMNS = [
    "kernel_name",
    "batch_size",
    "num_warps",
    "block_l",
    "sparsity",
    "time_ms",
    "analytical_bw_gbs",
    "bw_util_pct",
    "arithmetic_intensity",
    "roofline_class",
    "sm_occupancy_pct",
    "edp_estimate_uj_ms",
]


def format_report(
    results: Sequence[SpikeProfileResult],
    *,
    top_n: int = 20,
    sort_by: str = "time_ms",
    columns: Sequence[str] | None = None,
) -> str:
    """Format a ranked results table as a string.

    Pure function.
    """
    ranked = rank_results(results, sort_by=sort_by, top_n=top_n)
    cols = list(columns) if columns else _REPORT_COLUMNS

    if not ranked:
        return "No valid results to display."

    # Compute column widths
    widths: dict[str, int] = {}
    for col in cols:
        widths[col] = max(len(col), 12)

    # Header
    lines: list[str] = []
    header = " | ".join(col.ljust(widths[col]) for col in cols)
    lines.append(header)
    lines.append("-" * len(header))

    # Rows
    for r in ranked:
        row_parts: list[str] = []
        for col in cols:
            val = getattr(r, col, "")
            if isinstance(val, float):
                formatted = f"{val:.4f}"
            else:
                formatted = str(val)
            row_parts.append(formatted.ljust(widths[col]))
        lines.append(" | ".join(row_parts))

    return "\n".join(lines)


def print_report(
    results: Sequence[SpikeProfileResult],
    *,
    top_n: int = 20,
    sort_by: str = "time_ms",
) -> None:
    """Print a formatted report to stdout."""
    print(format_report(results, top_n=top_n, sort_by=sort_by))


# ── Export Functions ────────────────────────────────────────────────


def export_csv(
    results: Sequence[SpikeProfileResult],
    *,
    path: str,
) -> None:
    """Export results to a CSV file."""
    content = results_to_csv(list(results))
    with open(path, "w", newline="") as f:
        f.write(content)


def export_json(
    results: Sequence[SpikeProfileResult],
    *,
    path: str,
    group_by_kernel: bool = False,
) -> None:
    """Export results to a JSON file.

    If group_by_kernel=True, groups results hierarchically by kernel name.
    """
    if group_by_kernel:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for r in results:
            grouped.setdefault(r.kernel_name, []).append(r.to_dict())
        content = json.dumps(grouped, indent=2, default=str)
    else:
        content = results_to_json(list(results))

    with open(path, "w") as f:
        f.write(content)
