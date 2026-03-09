"""
Chart Comparison Tool

Generates comparison charts from benchmark results in all_results.csv.
Allows comparing specific runs or test configurations side by side.

Usage:
    python chart_compare.py --list                           # List available runs
    python chart_compare.py --runs RUN_ID1 RUN_ID2           # Compare specific runs
    python chart_compare.py --tests TEST1 TEST2              # Compare latest runs by test name
    python chart_compare.py --tests TEST1 TEST2 --sb 700GB   # Filter by shared_buffers
"""

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MARKERS = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
LINE_STYLES = ['-', '--', '-.', ':']


def load_csv(results_dir: str = "./results") -> list[dict]:
    """Load all results from consolidated CSV."""
    filepath = Path(results_dir) / "all_results.csv"
    if not filepath.exists():
        print(f"No results found at {filepath}")
        return []

    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def list_runs(rows: list[dict]):
    """List all unique runs in the CSV."""
    runs = {}
    for row in rows:
        run_id = row["run_id"]
        if run_id not in runs:
            runs[run_id] = {
                "test_name": row["test_name"],
                "suite_type": row["suite_type"],
                "dataset": row["dataset"],
                "shared_buffers": row.get("shared_buffers", "N/A"),
                "fs_cache": row.get("fs_cache", "True"),
                "benchmarks": 0,
            }
        runs[run_id]["benchmarks"] += 1

    print(f"\n{'Idx':<4} {'Run ID':<16} {'Test Name':<40} {'Suite':<12} {'SB':<8} {'Cache':<6} {'Benchmarks'}")
    print("-" * 110)
    for i, (run_id, info) in enumerate(sorted(runs.items()), 1):
        cache = "yes" if info["fs_cache"] == "True" else "no"
        print(f"{i:<4} {run_id:<16} {info['test_name']:<40} {info['suite_type']:<12} "
              f"{info['shared_buffers']:<8} {cache:<6} {info['benchmarks']}")
    print()


def get_series_data(rows: list[dict], run_id: str) -> dict:
    """Extract recall/QPS/latency data points for a run."""
    points = []
    meta = {}
    for row in rows:
        if row["run_id"] != run_id:
            continue
        if not meta:
            meta = {
                "test_name": row["test_name"],
                "suite_type": row["suite_type"],
                "shared_buffers": row.get("shared_buffers", "N/A"),
                "fs_cache": row.get("fs_cache", "True"),
                "m": row.get("m", "N/A"),
                "ef_construction": row.get("ef_construction", "N/A"),
                "lists": row.get("lists", "N/A"),
            }

        try:
            recall = float(row["recall"])
            qps = float(row["qps"])
            p99 = float(row["p99_latency_ms"])
        except (ValueError, KeyError):
            continue

        # Build point label from search params
        if row.get("ef_search") not in ("N/A", "", None):
            label = f"ef={row['ef_search']}"
        elif row.get("nprob") not in ("N/A", "", None):
            eps = row.get("epsilon", "")
            label = f"{row['nprob']} e{eps}"
        else:
            label = row.get("benchmark_name", "")

        points.append({"recall": recall, "qps": qps, "p99": p99, "label": label})

    # Sort by recall
    points.sort(key=lambda p: p["recall"])
    return {"meta": meta, "points": points}


def make_series_label(meta: dict) -> str:
    """Create a human-readable label for a series."""
    suite = meta.get("suite_type", "")
    sb = meta.get("shared_buffers", "")
    cache = "cache" if meta.get("fs_cache", "True") == "True" else "no-cache"

    if suite == "pgvector":
        m = meta.get("m", "?")
        efc = meta.get("ef_construction", "?")
        return f"pgvector m={m} ef_c={efc} (sb={sb}, {cache})"
    else:
        lists = meta.get("lists", "?")
        return f"{suite} lists={lists} (sb={sb}, {cache})"


def find_latest_run_id(rows: list[dict], test_name: str, sb: str = None, cache_mode: str = None) -> str:
    """Find the latest run_id for a test_name with optional filters."""
    candidates = []
    for row in rows:
        if row["test_name"] != test_name:
            continue
        if sb and row.get("shared_buffers") != sb:
            continue
        if cache_mode == "with" and row.get("fs_cache") != "True":
            continue
        if cache_mode == "without" and row.get("fs_cache") != "False":
            continue
        candidates.append(row["run_id"])

    if not candidates:
        return None
    return max(set(candidates))


def plot_comparison(series_list: list[dict], output: Path, chart_type: str = "qps"):
    """Generate a comparison chart with multiple series."""
    fig, ax = plt.subplots(figsize=(12, 8))

    y_key = "qps" if chart_type == "qps" else "p99"
    y_label = "QPS" if chart_type == "qps" else "P99 Latency (ms)"
    title_suffix = "Recall vs QPS" if chart_type == "qps" else "Recall vs P99 Latency"

    for i, series in enumerate(series_list):
        meta = series["meta"]
        points = series["points"]
        if not points:
            continue

        color = COLORS[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]
        linestyle = LINE_STYLES[i % len(LINE_STYLES)]
        label = make_series_label(meta)

        recalls = [p["recall"] for p in points]
        y_vals = [p[y_key] for p in points]

        ax.plot(recalls, y_vals, color=color, marker=marker, linestyle=linestyle,
                linewidth=2, markersize=8, label=label, zorder=3)

        # Add point labels with alternating positions to avoid overlap
        for j, point in enumerate(points):
            # Alternate above/below for different series, offset for same series
            y_offset = 12 if i % 2 == 0 else -16
            x_offset = 5

            ax.annotate(
                point["label"],
                (point["recall"], point[y_key]),
                textcoords="offset points",
                xytext=(x_offset, y_offset),
                fontsize=7,
                color=color,
                alpha=0.8,
                ha="left",
                va="bottom" if y_offset > 0 else "top",
            )

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title_suffix, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"Chart saved: {output}")


def main():
    parser = argparse.ArgumentParser(description="Compare benchmark runs with charts")
    parser.add_argument("--list", action="store_true", help="List available runs")
    parser.add_argument("--runs", nargs="+", help="Run IDs to compare")
    parser.add_argument("--tests", nargs="+", help="Test names to compare (uses latest run for each)")
    parser.add_argument("--sb", type=str, help="Filter by shared_buffers size (e.g., '700GB')")
    parser.add_argument("--cache-mode", choices=["with", "without"], help="Filter by cache mode")
    parser.add_argument("--output", type=str, default="./results/comparisons",
                        help="Output directory for charts")
    parser.add_argument("--results-dir", type=str, default="./results",
                        help="Results directory containing all_results.csv")
    args = parser.parse_args()

    rows = load_csv(args.results_dir)
    if not rows:
        return

    if args.list:
        list_runs(rows)
        return

    # Determine which run_ids to compare
    run_ids = []
    if args.runs:
        run_ids = args.runs
    elif args.tests:
        for test_name in args.tests:
            rid = find_latest_run_id(rows, test_name, sb=args.sb, cache_mode=args.cache_mode)
            if rid:
                run_ids.append(rid)
                print(f"Using run {rid} for {test_name}")
            else:
                print(f"No matching run found for {test_name}")
    else:
        parser.print_help()
        return

    if len(run_ids) < 2:
        print("Need at least 2 runs to compare.")
        return

    # Build series data
    series_list = []
    for rid in run_ids:
        data = get_series_data(rows, rid)
        if data["points"]:
            series_list.append(data)
        else:
            print(f"No benchmark data found for run {rid}")

    if len(series_list) < 2:
        print("Need at least 2 series with data to compare.")
        return

    # Generate charts
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    plot_comparison(series_list, output_dir / f"recall_vs_qps_{timestamp}.png", chart_type="qps")
    plot_comparison(series_list, output_dir / f"recall_vs_p99_{timestamp}.png", chart_type="p99")


if __name__ == "__main__":
    main()
