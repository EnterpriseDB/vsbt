"""
Results Management Module

Handles saving, consolidating, and visualizing benchmark results.
Includes system metrics and PostgreSQL statistics integration.
"""

import csv
import json
import socket
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def format_markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Format a markdown table with proper column alignment."""
    # Calculate max width for each column
    num_cols = len(headers)
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < num_cols:
                widths[i] = max(widths[i], len(str(cell)))

    # Build formatted lines
    lines = []
    # Header row
    header_cells = [h.ljust(widths[i]) for i, h in enumerate(headers)]
    lines.append("| " + " | ".join(header_cells) + " |")
    # Separator row
    sep_cells = ["-" * widths[i] for i in range(num_cols)]
    lines.append("|-" + "-|-".join(sep_cells) + "-|")
    # Data rows
    for row in rows:
        data_cells = [str(cell).ljust(widths[i]) for i, cell in enumerate(row)]
        lines.append("| " + " | ".join(data_cells) + " |")

    return lines


class ResultsManager:
    """
    Manages benchmark results storage, consolidation, and visualization.

    Handles:
    - Raw results storage (JSON per run)
    - Consolidated results (CSV append)
    - Chart generation (recall vs QPS, build times)
    - Markdown report generation
    """

    def __init__(self, base_dir: str = "./results"):
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.consolidated_dir = self.base_dir / "consolidated"
        self.reports_dir = self.base_dir / "reports"
        self.charts_dir = self.reports_dir / "charts"

        # Create directories
        for directory in [self.raw_dir, self.consolidated_dir, self.reports_dir, self.charts_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        self.hostname = socket.gethostname()
        self._current_run_id = None  # Set per-test in process_suite_results

    def _generate_run_id(self) -> str:
        """Generate a new run ID (timestamp) for a test."""
        return datetime.now().strftime("%Y%m%d%H%M%S")

    def save_raw_results(self, test_name: str, config: dict, results: dict) -> Path:
        """Save raw results as JSON for a single run."""
        filename = f"{test_name}_{self._current_run_id}.json"
        filepath = self.raw_dir / filename

        raw_data = {
            "metadata": {
                "run_id": self._current_run_id,
                "test_name": test_name,
                "hostname": self.hostname,
            },
            "config": config,
            "results": results,
        }

        with open(filepath, "w") as f:
            json.dump(raw_data, f, indent=2, default=str)

        return filepath

    def append_to_consolidated(
        self,
        suite_type: str,
        test_name: str,
        config: dict,
        results: dict,
        benchmark_name: str,
        benchmark_config: dict,
    ) -> Path:
        """Append benchmark results to consolidated CSV."""
        filepath = self.consolidated_dir / "all_results.csv"
        file_exists = filepath.exists()

        # Build row data
        row = {
            "run_id": self._current_run_id,
            "hostname": self.hostname,
            "suite_type": suite_type,
            "test_name": test_name,
            "benchmark_name": benchmark_name,
            "dataset": config.get("dataset", "N/A"),
            "metric": config.get("metric", "N/A"),
            "pg_parallel_workers": config.get("pg_parallel_workers", "N/A"),
            "top": config.get("top", "N/A"),
            # Index config (varies by suite type)
            "m": config.get("m", "N/A"),
            "ef_construction": config.get("efConstruction", "N/A"),
            "ef_search": benchmark_config.get("efSearch", "N/A"),
            "lists": str(config.get("lists", results.get("lists", "N/A"))),
            "sampling_factor": config.get("samplingFactor", "N/A"),
            "nprob": benchmark_config.get("nprob", "N/A"),
            "epsilon": benchmark_config.get("epsilon", "N/A"),
            "residual_quantization": config.get("residual_quantization", "N/A"),
            "build_threads": results.get("build_threads", "N/A"),
            # Timing metrics
            "load_time_s": results.get("load_time", "N/A"),
            "clustering_time": results.get("clustering_time", "N/A"),
            "index_build_time_s": results.get("index_build_time", "N/A"),
            "index_size": results.get("index_size", "N/A"),
            # Benchmark results
            "recall": results.get(benchmark_name, {}).get("recall", "N/A"),
            "qps": results.get(benchmark_name, {}).get("qps", "N/A"),
            "p50_latency_ms": results.get(benchmark_name, {}).get("p50_latency", "N/A"),
            "p99_latency_ms": results.get(benchmark_name, {}).get("p99_latency", "N/A"),
        }

        with open(filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        return filepath

    def generate_recall_vs_qps_chart(
        self,
        test_name: str,
        results: dict,
        config: dict,
    ) -> Path:
        """Generate a recall vs QPS scatter plot."""
        filepath = self.charts_dir / f"{test_name}_recall_vs_qps.png"

        benchmarks = config.get("benchmarks", {})
        if not benchmarks:
            return None

        recalls = []
        qps_values = []
        labels = []

        for bench_name in benchmarks.keys():
            if bench_name in results:
                recalls.append(results[bench_name].get("recall", 0))
                qps_values.append(results[bench_name].get("qps", 0))
                labels.append(bench_name)

        if not recalls:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot points
        scatter = ax.scatter(recalls, qps_values, s=100, c=range(len(recalls)), cmap="viridis", edgecolors="black")

        # Add labels to points
        for i, label in enumerate(labels):
            ax.annotate(
                label,
                (recalls[i], qps_values[i]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("QPS", fontsize=12)
        ax.set_title(f"Recall vs QPS - {test_name}", fontsize=14)
        ax.grid(True, alpha=0.3)

        # Set axis limits with padding
        if recalls:
            ax.set_xlim(min(recalls) * 0.95, min(max(recalls) * 1.02, 1.0))
        if qps_values:
            ax.set_ylim(0, max(qps_values) * 1.1)

        plt.tight_layout()
        plt.savefig(filepath, dpi=150)
        plt.close()

        return filepath

    def generate_latency_chart(
        self,
        test_name: str,
        results: dict,
        config: dict,
    ) -> Path:
        """Generate a latency comparison bar chart."""
        filepath = self.charts_dir / f"{test_name}_latency.png"

        benchmarks = config.get("benchmarks", {})
        if not benchmarks:
            return None

        bench_names = []
        p50_values = []
        p99_values = []

        for bench_name in benchmarks.keys():
            if bench_name in results:
                bench_names.append(bench_name)
                p50_values.append(results[bench_name].get("p50_latency", 0))
                p99_values.append(results[bench_name].get("p99_latency", 0))

        if not bench_names:
            return None

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(bench_names))
        width = 0.35

        bars1 = ax.bar(x - width / 2, p50_values, width, label="P50", color="#2ecc71")
        bars2 = ax.bar(x + width / 2, p99_values, width, label="P99", color="#e74c3c")

        ax.set_xlabel("Benchmark Configuration", fontsize=12)
        ax.set_ylabel("Latency (ms)", fontsize=12)
        ax.set_title(f"Query Latency - {test_name}", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(bench_names, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.tight_layout()
        plt.savefig(filepath, dpi=150)
        plt.close()

        return filepath

    def generate_build_time_chart(
        self,
        test_name: str,
        results: dict,
        config: dict,
    ) -> Path:
        """Generate a build time breakdown chart."""
        filepath = self.charts_dir / f"{test_name}_build_times.png"

        load_time = results.get("load_time", 0) or 0
        clustering_time_str = results.get("clustering_time", "0")
        index_build_time = results.get("index_build_time", 0) or 0

        # Parse clustering time (might be string like "123.45s")
        if isinstance(clustering_time_str, str):
            clustering_time = float(clustering_time_str.replace("s", "").strip()) if clustering_time_str else 0
        else:
            clustering_time = float(clustering_time_str) if clustering_time_str else 0

        # Calculate index-only time (excluding clustering if we have it)
        if clustering_time > 0:
            index_only_time = max(0, index_build_time - clustering_time)
        else:
            index_only_time = index_build_time
            clustering_time = 0

        fig, ax = plt.subplots(figsize=(10, 6))

        categories = ["Load Data", "Clustering", "Index Build"]
        times = [load_time, clustering_time, index_only_time]
        colors = ["#3498db", "#f39c12", "#9b59b6"]

        bars = ax.barh(categories, times, color=colors, edgecolor="black")

        ax.set_xlabel("Time (seconds)", fontsize=12)
        ax.set_title(f"Build Time Breakdown - {test_name}", fontsize=14)
        ax.grid(True, alpha=0.3, axis="x")

        # Add value labels
        for bar, time_val in zip(bars, times):
            width = bar.get_width()
            if time_val > 0:
                ax.annotate(
                    f"{time_val:.1f}s",
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                )

        # Add total time annotation
        total_time = load_time + clustering_time + index_only_time
        ax.annotate(
            f"Total: {total_time:.1f}s",
            xy=(0.98, 0.02),
            xycoords="axes fraction",
            ha="right",
            fontsize=12,
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()
        plt.savefig(filepath, dpi=150)
        plt.close()

        return filepath

    def generate_markdown_report(
        self,
        suite_type: str,
        test_name: str,
        config: dict,
        results: dict,
        query_clients: int = 1,
        system_metrics: Optional[str] = None,
        pg_stats: Optional[str] = None,
        system_dashboard_path: Optional[Path] = None,
    ) -> Path:
        """
        Generate a comprehensive markdown report with embedded charts.

        Args:
            suite_type: Type of benchmark suite (pgvector, vectorchord, pgpu)
            test_name: Name of the benchmark suite
            config: Suite configuration dictionary
            results: Benchmark results dictionary
            query_clients: Number of parallel query clients used
            system_metrics: Pre-formatted markdown string with system metrics
            pg_stats: Pre-formatted markdown string with PostgreSQL statistics
            system_dashboard_path: Path to system metrics dashboard image
        """
        filepath = self.reports_dir / f"{test_name}_report.md"

        # Generate charts first
        recall_qps_chart = self.generate_recall_vs_qps_chart(test_name, results, config)
        latency_chart = self.generate_latency_chart(test_name, results, config)
        build_time_chart = self.generate_build_time_chart(test_name, results, config)

        lines = [
            f"# Benchmark Report: {test_name}",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Host:** {self.hostname}",
            f"**Suite Type:** {suite_type}",
            "",
            "---",
            "",
            "## Configuration",
            "",
        ]

        # Build configuration table rows
        config_rows = [
            ["Dataset", str(config.get("dataset", "N/A"))],
            ["Metric", str(config.get("metric", "N/A"))],
            ["PG Parallel Workers", str(config.get("pg_parallel_workers", "N/A"))],
            ["Query Clients", str(query_clients)],
            ["Top-K", str(config.get("top", "N/A"))],
        ]

        # Add suite-specific config
        if suite_type == "pgvector":
            config_rows.extend([
                ["M", str(config.get("m", "N/A"))],
                ["EF Construction", str(config.get("efConstruction", "N/A"))],
            ])
        elif suite_type in ("vectorchord", "pgpu"):
            config_rows.extend([
                ["Lists", str(config.get("lists", results.get("lists", "N/A")))],
                ["Sampling Factor", str(config.get("samplingFactor", "N/A"))],
                ["Residual Quantization", str(config.get("residual_quantization", "N/A"))],
            ])
            if suite_type == "vectorchord":
                config_rows.extend([
                    ["Build Threads", str(results.get("build_threads", "N/A"))],
                    ["K-means Hierarchical", str(config.get("kmeans_hierarchical", "N/A"))],
                ])

        lines.extend(format_markdown_table(["Parameter", "Value"], config_rows))

        # Build metrics table
        build_rows = [
            ["Load Time", f"{results.get('load_time', 'N/A')}s"],
        ]
        if results.get("clustering_time"):
            build_rows.append(["Clustering Time", str(results.get("clustering_time"))])
        build_rows.extend([
            ["Index Build Time", f"{results.get('index_build_time', 'N/A')}s"],
            ["Index Size", str(results.get("index_size", "N/A"))],
        ])

        lines.extend([
            "",
            "---",
            "",
            "## Build Metrics",
            "",
        ])
        lines.extend(format_markdown_table(["Metric", "Value"], build_rows))

        # Add build time chart
        if build_time_chart:
            lines.extend([
                "",
                f"![Build Time Breakdown](charts/{build_time_chart.name})",
            ])

        lines.extend([
            "",
            "---",
            "",
            "## Benchmark Results",
            "",
        ])

        # Build benchmark results table
        bench_rows = []
        for bench_name, bench_config in config.get("benchmarks", {}).items():
            if bench_name in results:
                bench_results = results[bench_name]
                recall = bench_results.get("recall", 0)
                qps = bench_results.get("qps", 0)
                p50 = bench_results.get("p50_latency", 0)
                p99 = bench_results.get("p99_latency", 0)

                if suite_type == "pgvector":
                    bench_rows.append([
                        str(bench_config.get("efSearch", "N/A")),
                        f"{recall:.4f}",
                        f"{qps:.2f}",
                        f"{p50:.2f}",
                        f"{p99:.2f}",
                    ])
                else:
                    bench_rows.append([
                        str(bench_config.get("nprob", "N/A")),
                        str(bench_config.get("epsilon", "N/A")),
                        f"{recall:.4f}",
                        f"{qps:.2f}",
                        f"{p50:.2f}",
                        f"{p99:.2f}",
                    ])

        if suite_type == "pgvector":
            lines.extend(format_markdown_table(
                ["EF Search", "Recall", "QPS", "P50 (ms)", "P99 (ms)"],
                bench_rows
            ))
        else:
            lines.extend(format_markdown_table(
                ["nprob", "epsilon", "Recall", "QPS", "P50 (ms)", "P99 (ms)"],
                bench_rows
            ))

        # Add charts
        lines.extend([
            "",
            "---",
            "",
            "## Charts",
            "",
        ])

        if recall_qps_chart:
            lines.extend([
                "### Recall vs QPS",
                "",
                f"![Recall vs QPS](charts/{recall_qps_chart.name})",
                "",
            ])

        if latency_chart:
            lines.extend([
                "### Query Latency",
                "",
                f"![Query Latency](charts/{latency_chart.name})",
                "",
            ])

        # Add system metrics section if provided
        if system_metrics:
            lines.extend([
                "---",
                "",
                system_metrics,
            ])

            # Add system dashboard if available
            if system_dashboard_path and system_dashboard_path.exists():
                lines.extend([
                    "",
                    f"![System Dashboard](charts/{system_dashboard_path.name})",
                    "",
                ])

        # Add PostgreSQL stats section if provided
        if pg_stats:
            lines.extend([
                "---",
                "",
                pg_stats,
            ])

        # Write file
        with open(filepath, "w") as f:
            f.write("\n".join(lines))

        return filepath

    def process_suite_results(
        self,
        suite_type: str,
        config: dict,
        results: dict,
        query_clients: int = 1,
        system_metrics: Optional[str] = None,
        pg_stats: Optional[str] = None,
        system_dashboard_path: Optional[Path] = None,
    ):
        """
        Process and save all results for a benchmark suite.

        This is the main entry point for saving results after a benchmark run.

        Args:
            suite_type: Type of benchmark suite (pgvector, vectorchord, pgpu)
            config: Suite configuration dictionary
            results: Benchmark results dictionary
            query_clients: Number of parallel query clients used
            system_metrics: Pre-formatted markdown string with system metrics
            pg_stats: Pre-formatted markdown string with PostgreSQL statistics
            system_dashboard_path: Path to system metrics dashboard image
        """
        for test_name, suite_config in config.items():
            # Generate unique run_id for each test
            self._current_run_id = self._generate_run_id()

            suite_results = results.get(test_name, {})

            # Save raw results
            self.save_raw_results(test_name, suite_config, suite_results)

            # Append each benchmark to consolidated CSV
            for bench_name, bench_config in suite_config.get("benchmarks", {}).items():
                self.append_to_consolidated(
                    suite_type=suite_type,
                    test_name=test_name,
                    config=suite_config,
                    results=suite_results,
                    benchmark_name=bench_name,
                    benchmark_config=bench_config,
                )

            # Copy system dashboard to charts directory if provided
            dashboard_in_charts = None
            if system_dashboard_path and system_dashboard_path.exists():
                import shutil
                dashboard_in_charts = self.charts_dir / system_dashboard_path.name
                shutil.copy(system_dashboard_path, dashboard_in_charts)

            # Generate report with charts
            self.generate_markdown_report(
                suite_type=suite_type,
                test_name=test_name,
                config=suite_config,
                results=suite_results,
                query_clients=query_clients,
                system_metrics=system_metrics,
                pg_stats=pg_stats,
                system_dashboard_path=dashboard_in_charts,
            )

        print(f"\n📁 Results available in {self.reports_dir.parent}/ → raw, reports, consolidated")
