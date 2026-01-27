#!/usr/bin/env python3
"""
Compare Runs - Historical Benchmark Comparison Utility

Compare benchmark results between different runs to analyze performance
changes over time or across configurations.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from tabulate import tabulate


def load_raw_result(filepath: Path) -> dict:
    """Load a raw JSON result file."""
    with open(filepath) as f:
        return json.load(f)


def find_raw_files(results_dir: Path) -> list[Path]:
    """Find all raw result JSON files."""
    raw_dir = results_dir / "raw"
    if not raw_dir.exists():
        return []
    return sorted(raw_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)


def list_runs(results_dir: Path):
    """List all available benchmark runs."""
    raw_files = find_raw_files(results_dir)

    if not raw_files:
        print("No benchmark runs found.")
        return

    print(f"\nFound {len(raw_files)} benchmark run(s):\n")

    rows = []
    for i, filepath in enumerate(raw_files, 1):
        try:
            data = load_raw_result(filepath)
            suite_name = list(data.keys())[0] if data else "unknown"
            suite_data = data.get(suite_name, {})

            # Extract key info
            dataset = suite_data.get("dataset", "unknown")
            timestamp = filepath.stem.split("_")[-1] if "_" in filepath.stem else "unknown"
            num_benchmarks = len(suite_data.get("benchmarks", {}))

            rows.append([i, filepath.name, suite_name, dataset, num_benchmarks, timestamp])
        except Exception as e:
            rows.append([i, filepath.name, "error", str(e)[:30], "-", "-"])

    headers = ["#", "Filename", "Suite", "Dataset", "Benchmarks", "Timestamp"]
    print(tabulate(rows, headers=headers, tablefmt="simple"))
    print()


def get_run_by_index(results_dir: Path, index: int) -> Optional[Path]:
    """Get a raw result file by its index (1-based)."""
    raw_files = find_raw_files(results_dir)
    if 1 <= index <= len(raw_files):
        return raw_files[index - 1]
    return None


def extract_benchmark_metrics(data: dict) -> pd.DataFrame:
    """Extract benchmark metrics into a DataFrame."""
    rows = []

    for suite_name, suite_data in data.items():
        benchmarks = suite_data.get("benchmarks", {})

        for bench_name, bench_data in benchmarks.items():
            row = {
                "suite": suite_name,
                "benchmark": bench_name,
                "recall": bench_data.get("recall"),
                "qps": bench_data.get("qps"),
                "p50_ms": bench_data.get("p50_ms"),
                "p99_ms": bench_data.get("p99_ms"),
            }
            rows.append(row)

    return pd.DataFrame(rows)


def compare_runs(results_dir: Path, run1_idx: int, run2_idx: int, output_format: str = "table"):
    """Compare two benchmark runs."""
    run1_path = get_run_by_index(results_dir, run1_idx)
    run2_path = get_run_by_index(results_dir, run2_idx)

    if not run1_path:
        print(f"Error: Run #{run1_idx} not found.")
        return
    if not run2_path:
        print(f"Error: Run #{run2_idx} not found.")
        return

    print(f"\nComparing runs:")
    print(f"  Run A (#{run1_idx}): {run1_path.name}")
    print(f"  Run B (#{run2_idx}): {run2_path.name}")
    print()

    data1 = load_raw_result(run1_path)
    data2 = load_raw_result(run2_path)

    df1 = extract_benchmark_metrics(data1)
    df2 = extract_benchmark_metrics(data2)

    if df1.empty or df2.empty:
        print("Error: One or both runs have no benchmark data.")
        return

    # Merge on benchmark name
    merged = pd.merge(
        df1, df2,
        on=["suite", "benchmark"],
        suffixes=("_A", "_B"),
        how="outer"
    )

    # Calculate deltas
    comparison_rows = []
    for _, row in merged.iterrows():
        bench_name = row["benchmark"]

        def calc_delta(col_a, col_b, higher_is_better=True):
            val_a = row.get(col_a)
            val_b = row.get(col_b)
            if pd.isna(val_a) or pd.isna(val_b):
                return None, None, None
            delta = val_b - val_a
            pct = (delta / val_a * 100) if val_a != 0 else 0
            if higher_is_better:
                indicator = "+" if delta > 0 else ("-" if delta < 0 else "=")
            else:
                indicator = "-" if delta > 0 else ("+" if delta < 0 else "=")
            return delta, pct, indicator

        recall_delta, recall_pct, recall_ind = calc_delta("recall_A", "recall_B", higher_is_better=True)
        qps_delta, qps_pct, qps_ind = calc_delta("qps_A", "qps_B", higher_is_better=True)
        p50_delta, p50_pct, p50_ind = calc_delta("p50_ms_A", "p50_ms_B", higher_is_better=False)
        p99_delta, p99_pct, p99_ind = calc_delta("p99_ms_A", "p99_ms_B", higher_is_better=False)

        comparison_rows.append({
            "Benchmark": bench_name,
            "Recall A": f"{row.get('recall_A', 0):.4f}" if pd.notna(row.get('recall_A')) else "-",
            "Recall B": f"{row.get('recall_B', 0):.4f}" if pd.notna(row.get('recall_B')) else "-",
            "Recall Δ": f"{recall_ind}{abs(recall_pct):.1f}%" if recall_pct is not None else "-",
            "QPS A": f"{row.get('qps_A', 0):.0f}" if pd.notna(row.get('qps_A')) else "-",
            "QPS B": f"{row.get('qps_B', 0):.0f}" if pd.notna(row.get('qps_B')) else "-",
            "QPS Δ": f"{qps_ind}{abs(qps_pct):.1f}%" if qps_pct is not None else "-",
            "P50 A": f"{row.get('p50_ms_A', 0):.2f}" if pd.notna(row.get('p50_ms_A')) else "-",
            "P50 B": f"{row.get('p50_ms_B', 0):.2f}" if pd.notna(row.get('p50_ms_B')) else "-",
            "P50 Δ": f"{p50_ind}{abs(p50_pct):.1f}%" if p50_pct is not None else "-",
        })

    comparison_df = pd.DataFrame(comparison_rows)

    if output_format == "table":
        print("Benchmark Comparison (A → B):")
        print("  + = improvement, - = regression, = = no change\n")
        print(tabulate(comparison_df, headers="keys", tablefmt="simple", showindex=False))
    elif output_format == "csv":
        print(comparison_df.to_csv(index=False))
    elif output_format == "json":
        print(comparison_df.to_json(orient="records", indent=2))

    # Summary statistics
    print("\n" + "=" * 60)
    print("Summary:")

    def summarize_metric(name, col_a, col_b, higher_is_better=True):
        vals_a = merged[col_a].dropna()
        vals_b = merged[col_b].dropna()
        if vals_a.empty or vals_b.empty:
            return

        avg_a = vals_a.mean()
        avg_b = vals_b.mean()
        delta_pct = ((avg_b - avg_a) / avg_a * 100) if avg_a != 0 else 0

        if higher_is_better:
            status = "improved" if delta_pct > 1 else ("regressed" if delta_pct < -1 else "unchanged")
        else:
            status = "improved" if delta_pct < -1 else ("regressed" if delta_pct > 1 else "unchanged")

        print(f"  {name}: {avg_a:.4f} → {avg_b:.4f} ({delta_pct:+.1f}%) [{status}]")

    summarize_metric("Avg Recall", "recall_A", "recall_B", higher_is_better=True)
    summarize_metric("Avg QPS", "qps_A", "qps_B", higher_is_better=True)
    summarize_metric("Avg P50 (ms)", "p50_ms_A", "p50_ms_B", higher_is_better=False)
    summarize_metric("Avg P99 (ms)", "p99_ms_A", "p99_ms_B", higher_is_better=False)
    print()


def extract_suite_summary(data: dict) -> dict:
    """Extract summary metrics from a run for cross-suite comparison."""
    summary = {}

    for suite_name, suite_data in data.items():
        results = suite_data.get("results", suite_data)  # Handle both formats
        benchmarks = results.get("benchmarks", suite_data.get("benchmarks", {}))

        # Find best QPS at recall >= 95%
        best_qps_95 = None
        best_recall_95 = None
        best_p50 = None
        best_p99 = None

        for bench_name, bench_data in benchmarks.items():
            recall = bench_data.get("recall", 0)
            qps = bench_data.get("qps", 0)
            p50 = bench_data.get("p50_ms") or bench_data.get("p50_latency", 0)
            p99 = bench_data.get("p99_ms") or bench_data.get("p99_latency", 0)

            # Best QPS at recall >= 95%
            if recall >= 0.95:
                if best_qps_95 is None or qps > best_qps_95:
                    best_qps_95 = qps
                    best_recall_95 = recall

            # Track best (lowest) latencies
            if p50 and (best_p50 is None or p50 < best_p50):
                best_p50 = p50
            if p99 and (best_p99 is None or p99 < best_p99):
                best_p99 = p99

        summary[suite_name] = {
            "suite_type": "pgvector" if "pgvector" in suite_name.lower() else
                         "pgpu" if "pgpu" in suite_name.lower() else
                         "vectorchord",
            "index_build_time": results.get("index_build_time"),
            "load_time": results.get("load_time"),
            "index_size": results.get("index_size"),
            "best_qps_95": best_qps_95,
            "best_recall_95": best_recall_95,
            "best_p50": best_p50,
            "best_p99": best_p99,
            "num_benchmarks": len(benchmarks),
        }

    return summary


def compare_runs_cross_suite(results_dir: Path, run_indices: list[int]):
    """Compare multiple runs from different suite types."""
    # Load all runs
    runs = []
    for idx in run_indices:
        path = get_run_by_index(results_dir, idx)
        if not path:
            print(f"Error: Run #{idx} not found.")
            return
        data = load_raw_result(path)
        summary = extract_suite_summary(data)
        suite_name = list(summary.keys())[0]
        runs.append({
            "idx": idx,
            "path": path,
            "suite_name": suite_name,
            "summary": summary[suite_name],
        })

    # Helper functions
    def fmt_time(val):
        if val is None:
            return "-"
        return f"{val}s" if isinstance(val, (int, float)) else str(val)

    def fmt_qps(val):
        return f"{val:.0f}" if val else "-"

    def fmt_recall(val):
        return f"{val:.4f}" if val else "-"

    def fmt_latency(val):
        return f"{val:.2f}" if val else "-"

    # Print header
    print(f"\n{'=' * 80}")
    print("CROSS-SUITE COMPARISON")
    print(f"{'=' * 80}\n")

    # List runs
    for i, run in enumerate(runs):
        label = chr(65 + i)  # A, B, C, ...
        print(f"  Run {label} (#{run['idx']}): {run['suite_name']}")
    print()

    # Build column headers
    headers = ["Metric"] + [chr(65 + i) for i in range(len(runs))]

    # Build metrics table
    print(f"{'─' * 80}")
    print("BUILD METRICS")
    print(f"{'─' * 80}")

    build_rows = []
    build_rows.append(["Index Build Time"] + [fmt_time(r["summary"].get("index_build_time")) for r in runs])
    build_rows.append(["Load Time"] + [fmt_time(r["summary"].get("load_time")) for r in runs])
    build_rows.append(["Index Size"] + [r["summary"].get("index_size") or "-" for r in runs])

    print(tabulate(build_rows, headers=headers, tablefmt="simple"))
    print()

    # Query performance table
    print(f"{'─' * 80}")
    print("QUERY PERFORMANCE (best results)")
    print(f"{'─' * 80}")

    perf_rows = []
    perf_rows.append(["Best QPS (recall≥95%)"] + [fmt_qps(r["summary"].get("best_qps_95")) for r in runs])
    perf_rows.append(["Recall at best QPS"] + [fmt_recall(r["summary"].get("best_recall_95")) for r in runs])
    perf_rows.append(["Best P50 (ms)"] + [fmt_latency(r["summary"].get("best_p50")) for r in runs])
    perf_rows.append(["Best P99 (ms)"] + [fmt_latency(r["summary"].get("best_p99")) for r in runs])

    print(tabulate(perf_rows, headers=headers, tablefmt="simple"))
    print()

    print(f"{'=' * 80}\n")


def show_run_details(results_dir: Path, run_idx: int):
    """Show detailed information about a specific run."""
    run_path = get_run_by_index(results_dir, run_idx)
    if not run_path:
        print(f"Error: Run #{run_idx} not found.")
        return

    print(f"\nDetails for run #{run_idx}: {run_path.name}\n")

    data = load_raw_result(run_path)

    for suite_name, suite_data in data.items():
        print(f"Suite: {suite_name}")
        print("-" * 40)

        # Configuration info
        config_keys = ["dataset", "lists", "m", "efConstruction", "index_build_time", "load_time"]
        for key in config_keys:
            if key in suite_data:
                print(f"  {key}: {suite_data[key]}")

        print()

        # Benchmark results
        benchmarks = suite_data.get("benchmarks", {})
        if benchmarks:
            rows = []
            for bench_name, bench_data in benchmarks.items():
                rows.append({
                    "Benchmark": bench_name,
                    "Recall": f"{bench_data.get('recall', 0):.4f}",
                    "QPS": f"{bench_data.get('qps', 0):.0f}",
                    "P50 (ms)": f"{bench_data.get('p50_ms', 0):.2f}",
                    "P99 (ms)": f"{bench_data.get('p99_ms', 0):.2f}",
                })

            print("Benchmark Results:")
            print(tabulate(rows, headers="keys", tablefmt="simple", showindex=False))
            print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare benchmark results between different runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available runs
  python compare_runs.py --list

  # Show details for run #5
  python compare_runs.py --show 5

  # Compare run #3 with run #7 (same suite type)
  python compare_runs.py --compare 3 7

  # Compare runs from different suites (pgvector vs vectorchord vs pgpu)
  python compare_runs.py --cross-suite 3 7 12

  # Compare runs and output as CSV
  python compare_runs.py --compare 3 7 --format csv
        """
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("./results"),
        help="Results directory (default: ./results)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available benchmark runs"
    )
    parser.add_argument(
        "--show", "-s",
        type=int,
        metavar="N",
        help="Show details for run #N"
    )
    parser.add_argument(
        "--compare", "-c",
        type=int,
        nargs=2,
        metavar=("A", "B"),
        help="Compare run #A with run #B (same suite type)"
    )
    parser.add_argument(
        "--cross-suite", "-x",
        type=int,
        nargs="+",
        metavar="N",
        help="Compare multiple runs from different suite types (e.g., pgvector vs vectorchord vs pgpu)"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["table", "csv", "json"],
        default="table",
        help="Output format for comparison (default: table)"
    )

    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"Error: Results directory not found: {args.results_dir}")
        sys.exit(1)

    if args.list:
        list_runs(args.results_dir)
    elif args.show:
        show_run_details(args.results_dir, args.show)
    elif args.compare:
        compare_runs(args.results_dir, args.compare[0], args.compare[1], args.format)
    elif args.cross_suite:
        compare_runs_cross_suite(args.results_dir, args.cross_suite)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
