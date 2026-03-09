"""
Results Management Module

Handles saving, consolidating, and visualizing benchmark results.
Results are organized per test_name with incremental reports across runs.

Directory structure:
    results/
    ├── all_results.csv                    # Global CSV (append-only)
    ├── {test_name}/                       # One folder per YAML test name
    │   ├── report.md                      # Incremental report (all runs)
    │   ├── runs/                          # Raw data per run
    │   │   └── {run_id}.json
    │   ├── charts/                        # Charts for this test
    │   │   ├── recall_vs_qps.png
    │   │   ├── latency.png
    │   │   ├── build_times.png
    │   │   └── system_dashboard.png
    │   └── index_build/                   # Index build monitoring
    └── comparisons/                       # Cross-test comparison charts
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
    num_cols = len(headers)
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < num_cols:
                widths[i] = max(widths[i], len(str(cell)))

    lines = []
    header_cells = [h.ljust(widths[i]) for i, h in enumerate(headers)]
    lines.append("| " + " | ".join(header_cells) + " |")
    sep_cells = ["-" * widths[i] for i in range(num_cols)]
    lines.append("|-" + "-|-".join(sep_cells) + "-|")
    for row in rows:
        data_cells = [str(cell).ljust(widths[i]) for i, cell in enumerate(row)]
        lines.append("| " + " | ".join(data_cells) + " |")

    return lines


class ResultsManager:
    """
    Manages benchmark results storage, consolidation, and visualization.

    Results are organized per test_name with incremental reports that
    accumulate across runs (different shared_buffers, fs_cache variants).
    """

    def __init__(self, base_dir: str = "./results"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.hostname = socket.gethostname()
        self._current_run_id = None

    def _test_dir(self, test_name: str) -> Path:
        """Get the directory for a specific test."""
        d = self.base_dir / test_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _runs_dir(self, test_name: str) -> Path:
        d = self._test_dir(test_name) / "runs"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _charts_dir(self, test_name: str) -> Path:
        d = self._test_dir(test_name) / "charts"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _generate_run_id(self) -> str:
        return datetime.now().strftime("%Y%m%d%H%M%S")

    def save_raw_results(self, test_name: str, config: dict, results: dict) -> Path:
        """Save raw results as JSON for a single run."""
        filepath = self._runs_dir(test_name) / f"{self._current_run_id}.json"

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
        filepath = self.base_dir / "all_results.csv"
        file_exists = filepath.exists()

        row = {
            "run_id": self._current_run_id,
            "hostname": self.hostname,
            "suite_type": suite_type,
            "test_name": test_name,
            "benchmark_name": benchmark_name,
            "shared_buffers": results.get("shared_buffers", "N/A"),
            "maintenance_work_mem": results.get("maintenance_work_mem", "N/A"),
            "fs_cache": str(results.get("fs_cache", True)),
            "dataset": config.get("dataset", "N/A"),
            "metric": config.get("metric", "N/A"),
            "pg_parallel_workers": config.get("pg_parallel_workers", "N/A"),
            "top": config.get("top", "N/A"),
            "m": config.get("m", "N/A"),
            "ef_construction": config.get("efConstruction", "N/A"),
            "ef_search": benchmark_config.get("efSearch", "N/A"),
            "lists": str(config.get("lists", results.get("lists", "N/A"))),
            "sampling_factor": config.get("samplingFactor", "N/A"),
            "nprob": benchmark_config.get("nprob", "N/A"),
            "epsilon": benchmark_config.get("epsilon", "N/A"),
            "residual_quantization": config.get("residual_quantization", "N/A"),
            "build_threads": results.get("build_threads", "N/A"),
            "load_time_s": results.get("load_time", "N/A"),
            "clustering_time": results.get("clustering_time", "N/A"),
            "index_build_time_s": results.get("index_build_time", "N/A"),
            "index_size": results.get("index_size", "N/A"),
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

    # --- Chart Generation ---

    def generate_recall_vs_qps_chart(self, test_name: str, results: dict, config: dict) -> Optional[Path]:
        """Generate a recall vs QPS scatter plot."""
        charts_dir = self._charts_dir(test_name)
        filepath = charts_dir / "recall_vs_qps.png"

        benchmarks = config.get("benchmarks", {})
        if not benchmarks:
            return None

        recalls, qps_values, labels = [], [], []
        for bench_name in benchmarks.keys():
            if bench_name in results:
                recalls.append(results[bench_name].get("recall", 0))
                qps_values.append(results[bench_name].get("qps", 0))
                labels.append(bench_name)

        if not recalls:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(recalls, qps_values, s=100, c=range(len(recalls)), cmap="viridis", edgecolors="black")

        for i, label in enumerate(labels):
            ax.annotate(label, (recalls[i], qps_values[i]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)

        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("QPS", fontsize=12)
        ax.set_title(f"Recall vs QPS - {test_name}", fontsize=14)
        ax.grid(True, alpha=0.3)
        if recalls:
            ax.set_xlim(min(recalls) * 0.95, min(max(recalls) * 1.02, 1.0))
        if qps_values:
            ax.set_ylim(0, max(qps_values) * 1.1)

        plt.tight_layout()
        plt.savefig(filepath, dpi=150)
        plt.close()
        return filepath

    def generate_latency_chart(self, test_name: str, results: dict, config: dict) -> Optional[Path]:
        """Generate a latency comparison bar chart."""
        charts_dir = self._charts_dir(test_name)
        filepath = charts_dir / "latency.png"

        benchmarks = config.get("benchmarks", {})
        if not benchmarks:
            return None

        bench_names, p50_values, p99_values = [], [], []
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

        for bar in list(bars1) + list(bars2):
            height = bar.get_height()
            ax.annotate(f"{height:.1f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        plt.savefig(filepath, dpi=150)
        plt.close()
        return filepath

    def generate_build_time_chart(self, test_name: str, results: dict, config: dict) -> Optional[Path]:
        """Generate a build time breakdown chart."""
        charts_dir = self._charts_dir(test_name)
        filepath = charts_dir / "build_times.png"

        load_time = results.get("load_time", 0) or 0
        clustering_time_str = results.get("clustering_time", "0")
        index_build_time = results.get("index_build_time", 0) or 0

        if isinstance(clustering_time_str, str):
            clustering_time = float(clustering_time_str.replace("s", "").strip()) if clustering_time_str else 0
        else:
            clustering_time = float(clustering_time_str) if clustering_time_str else 0

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

        for bar, time_val in zip(bars, times):
            if time_val > 0:
                ax.annotate(f"{time_val:.1f}s",
                            xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                            xytext=(5, 0), textcoords="offset points",
                            ha="left", va="center", fontsize=10, fontweight="bold")

        total_time = load_time + clustering_time + index_only_time
        ax.annotate(f"Total: {total_time:.1f}s", xy=(0.98, 0.02), xycoords="axes fraction",
                    ha="right", fontsize=12, fontweight="bold",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        plt.tight_layout()
        plt.savefig(filepath, dpi=150)
        plt.close()
        return filepath

    # --- Report Generation ---

    def _load_all_runs(self, test_name: str) -> list[dict]:
        """Load all previous run JSON files for a test, sorted by run_id."""
        runs_dir = self._runs_dir(test_name)
        runs = []
        for f in sorted(runs_dir.glob("*.json")):
            try:
                with open(f) as fh:
                    runs.append(json.load(fh))
            except (json.JSONDecodeError, IOError):
                continue
        return runs

    def _build_run_section(self, suite_type: str, run_data: dict) -> list[str]:
        """Build markdown lines for a single run's benchmark results."""
        results = run_data.get("results", {})
        config = run_data.get("config", {})
        metadata = run_data.get("metadata", {})
        run_id = metadata.get("run_id", "unknown")

        sb = results.get("shared_buffers", "N/A")
        mwm = results.get("maintenance_work_mem", "N/A")
        fs_cache = results.get("fs_cache", True)
        cache_str = "with page cache" if fs_cache else "no page cache"

        # Parse date from run_id
        try:
            run_date = datetime.strptime(run_id, "%Y%m%d%H%M%S").strftime("%Y-%m-%d %H:%M")
        except ValueError:
            run_date = run_id

        lines = [
            f"### Run: {run_date} (sb={sb}, mwm={mwm}, {cache_str})",
            "",
        ]

        # Build benchmark results table
        benchmarks = config.get("benchmarks", {})
        if not benchmarks:
            lines.append("*No benchmark results (build-only mode)*")
            lines.append("")
            return lines

        bench_rows = []
        for bench_name, bench_config in benchmarks.items():
            if bench_name in results and isinstance(results[bench_name], dict) and "recall" in results[bench_name]:
                br = results[bench_name]
                if suite_type == "pgvector":
                    bench_rows.append([
                        str(bench_config.get("efSearch", "N/A")),
                        f"{br['recall']:.4f}",
                        f"{br['qps']:.2f}",
                        f"{br['p50_latency']:.2f}",
                        f"{br['p99_latency']:.2f}",
                    ])
                else:
                    bench_rows.append([
                        str(bench_config.get("nprob", "N/A")),
                        str(bench_config.get("epsilon", "N/A")),
                        f"{br['recall']:.4f}",
                        f"{br['qps']:.2f}",
                        f"{br['p50_latency']:.2f}",
                        f"{br['p99_latency']:.2f}",
                    ])

        if bench_rows:
            if suite_type == "pgvector":
                lines.extend(format_markdown_table(
                    ["EF Search", "Recall", "QPS", "P50 (ms)", "P99 (ms)"], bench_rows))
            else:
                lines.extend(format_markdown_table(
                    ["nprob", "epsilon", "Recall", "QPS", "P50 (ms)", "P99 (ms)"], bench_rows))
        else:
            lines.append("*No benchmark results*")

        lines.append("")
        return lines

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
        Generate an incremental markdown report that includes all runs for this test.

        The report is regenerated from all stored run JSON files each time,
        so it always reflects the complete history.
        """
        test_dir = self._test_dir(test_name)
        filepath = test_dir / "report.md"

        # Generate charts for the current run
        self.generate_recall_vs_qps_chart(test_name, results, config)
        self.generate_latency_chart(test_name, results, config)
        self.generate_build_time_chart(test_name, results, config)

        # Load all runs for this test
        all_runs = self._load_all_runs(test_name)

        # --- Header ---
        lines = [
            f"# Benchmark Report: {test_name}",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Host:** {self.hostname}",
            f"**Suite Type:** {suite_type}",
            f"**Total Runs:** {len(all_runs)}",
            "",
        ]

        # --- System Information ---
        system_report = results.get("system_report")
        if system_report:
            lines.extend([
                "---",
                "",
                "## System Information",
                "",
                "```",
                system_report,
                "```",
                "",
            ])

        # --- Configuration ---
        lines.extend(["---", "", "## Configuration", ""])

        config_rows = [
            ["Dataset", str(config.get("dataset", "N/A"))],
            ["Metric", str(config.get("metric", "N/A"))],
            ["PG Parallel Workers", str(config.get("pg_parallel_workers", "N/A"))],
            ["Query Clients", str(query_clients)],
            ["Top-K", str(config.get("top", "N/A"))],
        ]

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

        # --- Build Metrics (across all runs) ---
        lines.extend(["", "---", "", "## Build Metrics", ""])

        build_rows = []
        for run_data in all_runs:
            r = run_data.get("results", {})
            m = run_data.get("metadata", {})
            run_id = m.get("run_id", "?")
            try:
                run_date = datetime.strptime(run_id, "%Y%m%d%H%M%S").strftime("%Y-%m-%d %H:%M")
            except ValueError:
                run_date = run_id

            build_rows.append([
                run_date,
                f"{r.get('load_time', 'N/A')}s" if r.get('load_time') else "N/A",
                f"{r.get('index_build_time', 'N/A')}s" if r.get('index_build_time') else "N/A",
                str(r.get("index_size", "N/A")),
                str(r.get("shared_buffers", "N/A")),
                str(r.get("maintenance_work_mem", "N/A")),
                "yes" if r.get("fs_cache", True) else "no",
            ])

        if build_rows:
            lines.extend(format_markdown_table(
                ["Date", "Load Time", "Build Time", "Index Size", "shared_buffers", "maint_work_mem", "FS Cache"],
                build_rows))

        # --- Build time chart (latest run) ---
        build_chart = self._charts_dir(test_name) / "build_times.png"
        if build_chart.exists():
            lines.extend(["", f"![Build Time Breakdown](charts/{build_chart.name})", ""])

        # --- Benchmark Results (each run as a section, newest first) ---
        lines.extend(["", "---", "", "## Benchmark Results", ""])

        for run_data in reversed(all_runs):
            lines.extend(self._build_run_section(suite_type, run_data))

        # --- Charts (latest run) ---
        lines.extend(["---", "", "## Charts", ""])

        recall_chart = self._charts_dir(test_name) / "recall_vs_qps.png"
        if recall_chart.exists():
            lines.extend(["### Recall vs QPS", "", f"![Recall vs QPS](charts/{recall_chart.name})", ""])

        latency_chart = self._charts_dir(test_name) / "latency.png"
        if latency_chart.exists():
            lines.extend(["### Query Latency", "", f"![Query Latency](charts/{latency_chart.name})", ""])

        # --- System metrics ---
        if system_metrics:
            lines.extend(["---", "", system_metrics])
            dashboard = self._charts_dir(test_name) / "system_dashboard.png"
            if dashboard.exists():
                lines.extend(["", f"![System Dashboard](charts/{dashboard.name})", ""])

        if pg_stats:
            lines.extend(["---", "", pg_stats])

        with open(filepath, "w") as f:
            f.write("\n".join(lines))

        return filepath

    # --- Main Entry Point ---

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
        """
        for test_name, suite_config in config.items():
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

            # Copy system dashboard to test charts directory
            if system_dashboard_path and system_dashboard_path.exists():
                import shutil
                dest = self._charts_dir(test_name) / "system_dashboard.png"
                shutil.copy(system_dashboard_path, dest)

            # Generate incremental report
            self.generate_markdown_report(
                suite_type=suite_type,
                test_name=test_name,
                config=suite_config,
                results=suite_results,
                query_clients=query_clients,
                system_metrics=system_metrics,
                pg_stats=pg_stats,
                system_dashboard_path=system_dashboard_path,
            )

        print(f"\n Results available in {self.base_dir}/")
