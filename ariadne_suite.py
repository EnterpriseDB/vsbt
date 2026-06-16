"""
Ariadne Benchmark Suite

Benchmarks vector search using the Ariadne extension (CrackIVF +
Extended-RaBitQ) for PostgreSQL.

Build options are passed per-index via WITH (k_outer, k_leaves).
Query-time knobs are session GUCs:
  - ariadne.probes         (total leaves probed per query, VChord nprobe-like)
  - ariadne.probes_outer   (outers expanded to select leaves)
  - ariadne.rerank_k       (exact-distance rerank pass, 0 = disabled)
  - ariadne.residual_quant (quantize residuals at build time)
  - ariadne.bits_per_dim   (Extended-RaBitQ bits, 0 = auto)
  - ariadne.sampling_factor
"""

import argparse
import time

import psycopg
import pgvector.psycopg

import common
from monitor import PgBuildMemoryTracker, format_bytes
from results import ResultsManager


def build_arg_parse():
    """Build argument parser for Ariadne benchmark suite."""
    parser = argparse.ArgumentParser(description="Ariadne Benchmark Suite")
    common.build_arg_parse(parser)
    return parser


class TestSuite(common.TestSuite):
    """
    Test suite for Ariadne (CrackIVF + Extended-RaBitQ).

    Convergence: Ariadne typically converges with far fewer probes than
    VectorChord on the same dataset — keep probe ladders short.
    """

    METRIC_OPS_OPCLASS = {
        "l2": "vector_l2_ops",
        "euclidean": "vector_l2_ops",
        "cos": "vector_cosine_ops",
        "angular": "vector_cosine_ops",
        "ip": "vector_ip_ops",
        "dot": "vector_ip_ops",
    }

    METRIC_OPS = {
        "l2": "<->",
        "euclidean": "<->",
        "cos": "<=>",
        "angular": "<=>",
        "dot": "<#>",
        "ip": "<#>",
    }

    @staticmethod
    def _parse_probe_pair(value) -> tuple[int, int]:
        """Parse 'probes_outer,probes' string (or list/tuple) into ints."""
        if isinstance(value, str):
            parts = value.split(",")
        else:
            parts = list(value)
        if len(parts) != 2:
            raise ValueError(
                f"probes must be 'probes_outer,probes' pair, got: {value!r}"
            )
        return int(parts[0]), int(parts[1])

    def _apply_query_gucs(self, conn, benchmark: dict, suite_config: dict):
        """Apply Ariadne session GUCs for a single benchmark run.

        Build-side knobs (bits_per_dim, residual_quant, etc.) are baked into
        the index at CREATE INDEX time via WITH-clause reloptions; the scan
        path reads them from the meta page, so we do NOT set them here."""
        probes_outer, probes = self._parse_probe_pair(benchmark["probes"])
        rerank_k = benchmark.get("rerank_k", suite_config.get("rerank_k", 0))
        query_bitquant = benchmark.get(
            "query_bitquant", suite_config.get("query_bitquant", False)
        )

        conn.execute("SET jit=false")
        conn.execute(f"SET ariadne.probes_outer = {probes_outer}")
        conn.execute(f"SET ariadne.probes = {probes}")
        conn.execute(f"SET ariadne.rerank_k = {rerank_k}")
        conn.execute(
            f"SET ariadne.query_bitquant = {'on' if query_bitquant else 'off'}"
        )

    @staticmethod
    def process_batch(args):
        """Process a batch of queries in parallel."""
        (test, answer, top, metric_ops, url, table_name,
         probes_outer, probes, rerank_k, query_bitquant) = args

        conn = psycopg.connect(url)
        pgvector.psycopg.register_vector(conn)
        conn.execute("SET jit=false")
        conn.execute(f"SET ariadne.probes_outer = {probes_outer}")
        conn.execute(f"SET ariadne.probes = {probes}")
        conn.execute(f"SET ariadne.rerank_k = {rerank_k}")
        conn.execute(
            f"SET ariadne.query_bitquant = {'on' if query_bitquant else 'off'}"
        )

        query_sql = f"SELECT id FROM {table_name} ORDER BY embedding {metric_ops} %s LIMIT {top}"

        results = []
        cursor = conn.cursor()
        for query, ground_truth in zip(test, answer):
            start = time.perf_counter()
            cursor.execute(query_sql, (query,))
            result = cursor.fetchall()
            end = time.perf_counter()

            result_ids = {p[0] for p in result[:top]}
            gt_ids = ground_truth[:top]
            ground_truth_ids = set(gt_ids.tolist() if hasattr(gt_ids, "tolist") else gt_ids)
            hit = len(result_ids & ground_truth_ids)
            results.append((hit, (start, end)))

        cursor.close()
        conn.close()
        return results

    def make_batch_args(self, test, answer, top, metric, table_name, benchmark):
        """Prepare arguments for parallel batch processing."""
        metric_ops = self._get_metric_operator(metric)
        # Resolve suite-level defaults from the current suite config.
        suite_config = next(iter(self.config.values()))
        probes_outer, probes = self._parse_probe_pair(benchmark["probes"])
        rerank_k = benchmark.get("rerank_k", suite_config.get("rerank_k", 0))
        query_bitquant = benchmark.get(
            "query_bitquant", suite_config.get("query_bitquant", False)
        )
        return (
            test,
            answer,
            top,
            metric_ops,
            self.url,
            table_name,
            probes_outer,
            probes,
            rerank_k,
            query_bitquant,
        )

    @classmethod
    def _get_metric_operator(cls, metric: str) -> str:
        """Convert metric name to PostgreSQL operator."""
        if metric not in cls.METRIC_OPS:
            raise ValueError(f"Unsupported metric type: {metric}")
        return cls.METRIC_OPS[metric]

    def create_connection(self):
        """Create a database connection with pgvector support."""
        conn = super().create_connection()
        pgvector.psycopg.register_vector(conn)
        return conn

    def init_ext(self, suite_name: str = None):
        """Initialize required PostgreSQL extensions."""
        conn = super().create_connection()
        conn.execute("CREATE EXTENSION IF NOT EXISTS ariadne CASCADE")
        conn.close()
        self.debug_log("Extensions initialized successfully.")

    def prewarm_index(self, table_name: str):
        """Prewarm the index into memory using pg_prewarm."""
        index_name = f"{table_name}_embedding_idx"
        conn = self.create_connection()
        self.check_index_fits_shared_buffers(conn, index_name, table_name)
        print("Prewarming the index...", end="", flush=True)
        try:
            conn.execute("CREATE EXTENSION IF NOT EXISTS pg_prewarm")
            prewarm_start = time.perf_counter()
            conn.execute(f"SELECT pg_prewarm('{index_name}'::regclass)")
            prewarm_time = time.perf_counter() - prewarm_start
            print(f" done! ({prewarm_time:.1f}s)")
        except psycopg.Error as e:
            print(f" failed! ({e.diag.message_primary})")
            self.debug_log(f"Prewarm failed: {e}")
        finally:
            conn.close()

    def create_index(self, suite_name: str, table_name: str, dataset: dict):
        """Create an Ariadne index."""
        event, index_monitor_thread = super().create_index(
            suite_name, table_name, dataset
        )

        config = self.config[suite_name]
        pg_parallel_workers = config["pg_parallel_workers"]
        k_outer = config.get("k_outer", 0)
        k_leaves = config.get("k_leaves", 0)
        sampling_factor = config.get("samplingFactor",
                                     config.get("sampling_factor"))
        residual_quant = config.get(
            "residual_quantization", config.get("residual_quant", False)
        )
        bits_per_dim = config.get("bits_per_dim", 0)
        kmeans_iters = config.get("kmeans_iters")
        # Build-side params promoted to index reloptions in the Ariadne
        # storage-param refactor; pulled from YAML and pushed through
        # CREATE INDEX ... WITH (...). None = leave at the extension default
        # registered in src/index/am.c.
        colocate_fp = config.get("colocate_fp")
        routing_bits_per_dim = config.get("routing_bits_per_dim")
        parallel_sample = config.get("parallel_sample")
        build_threads = config.get("build_threads")
        debug_build = config.get("debug_build")
        metric = dataset["metric"]

        if self.debug:
            print(f"\n🔧 Index Configuration (Ariadne):")
            print(f"    • k_outer:         {k_outer}")
            print(f"    • k_leaves:        {k_leaves}")
            print(f"    • Sampling Factor: {sampling_factor}")
            print(f"    • bits_per_dim:    {bits_per_dim}")
            print(f"    • Residual Quant:  {residual_quant}")
            if kmeans_iters is not None:
                print(f"    • Lloyd iters:     {kmeans_iters}")
            if colocate_fp is not None:
                print(f"    • colocate_fp:     {colocate_fp}")
            if routing_bits_per_dim is not None:
                print(f"    • routing_bits:    {routing_bits_per_dim}")
            print()

        self.results[suite_name]["k_outer"] = k_outer
        self.results[suite_name]["k_leaves"] = k_leaves
        self.results[suite_name]["bits_per_dim"] = bits_per_dim
        # Mirror VectorChord results schema so the ResultsManager
        # "vectorchord" code path can render Ariadne runs unchanged.
        self.results[suite_name]["lists"] = [k_outer, k_leaves]

        opclass = self.METRIC_OPS_OPCLASS[metric]

        conn = self.create_connection()
        start_time = time.perf_counter()

        conn.execute(f"SET max_parallel_maintenance_workers TO {pg_parallel_workers}")
        conn.execute(f"SET max_parallel_workers TO {pg_parallel_workers}")

        # Every build-side knob lives in the CREATE INDEX WITH clause.
        # Skip keys the YAML didn't set — the C-side default kicks in.
        with_opts = []
        if k_outer:
            with_opts.append(f"k_outer = {k_outer}")
        if k_leaves:
            with_opts.append(f"k_leaves = {k_leaves}")
        if bits_per_dim is not None:
            with_opts.append(f"bits_per_dim = {bits_per_dim}")
        if sampling_factor is not None:
            with_opts.append(f"sampling_factor = {sampling_factor}")
        if kmeans_iters is not None:
            with_opts.append(f"kmeans_iters = {kmeans_iters}")
        with_opts.append(
            f"residual_quant = {'true' if residual_quant else 'false'}"
        )
        if colocate_fp is not None:
            with_opts.append(f"colocate_fp = '{colocate_fp}'")
        if routing_bits_per_dim is not None:
            with_opts.append(f"routing_bits_per_dim = {routing_bits_per_dim}")
        if parallel_sample is not None:
            with_opts.append(
                f"parallel_sample = {'true' if parallel_sample else 'false'}"
            )
        if build_threads is not None:
            with_opts.append(f"build_threads = {build_threads}")
        if debug_build is not None:
            with_opts.append(
                f"debug_build = {'true' if debug_build else 'false'}"
            )
        with_clause = f" WITH ({', '.join(with_opts)})" if with_opts else ""

        with PgBuildMemoryTracker(conn, self.url) as mem_tracker:
            conn.execute(
                f"CREATE INDEX {table_name}_embedding_idx ON {table_name} "
                f"USING ariadne (embedding {opclass}){with_clause}"
            )

        build_time = int(round(time.perf_counter() - start_time))
        self.results[suite_name]["index_build_time"] = build_time
        self.results[suite_name]["index_build_peak_heap_bytes"] = mem_tracker.peak_bytes

        event.set()
        index_monitor_thread.join()

        print(f"Index build time: {build_time}s")
        if mem_tracker.peak_bytes is not None:
            print(
                f"Index build peak private heap: {format_bytes(mem_tracker.peak_bytes)} "
                f"(leader {format_bytes(mem_tracker.leader_peak_bytes)} + "
                f"{mem_tracker.peak_worker_count} workers "
                f"{format_bytes(mem_tracker.workers_peak_bytes)}, "
                f"{mem_tracker.samples} samples)"
            )
        elif mem_tracker.reason_skipped:
            print(f"Index build peak private heap: not tracked ({mem_tracker.reason_skipped})")

        conn.execute("CHECKPOINT")
        conn.close()
        print("Index built successfully.")

    def sequential_bench(
        self,
        name: str,
        table_name: str,
        conn: psycopg.Connection,
        metric: str,
        top: int,
        benchmark: dict,
        dataset: dict,
    ) -> tuple[list[tuple[int, float]], str]:
        """Run sequential benchmark queries."""
        suite_config = next(iter(self.config.values()))
        self._apply_query_gucs(conn, benchmark, suite_config)

        metric_ops = self._get_metric_operator(metric)

        self.debug_log(
            f"Benchmark config: probes={benchmark['probes']}, "
            f"rerank_k={benchmark.get('rerank_k', suite_config.get('rerank_k', 0))}, "
            f"metric={metric}, metric_ops={metric_ops}"
        )

        return super().sequential_bench(
            name, table_name, conn, metric_ops, top, benchmark, dataset
        )

    def generate_markdown_result(self):
        """Generate benchmark results.

        Reuses the VectorChord report path (nprob/epsilon columns) so we
        get a basic report without touching results.py yet — probes_outer,
        probes pair lands in the "nprob" column, rerank_k in "epsilon".
        """
        self.debug_log(f"Results: {self.results}")

        results_manager = ResultsManager()

        # Massage benchmark configs so the vectorchord renderer prints
        # something useful for Ariadne runs.
        for suite_name, suite_config in self.config.items():
            patched_benchmarks = {}
            for bench_name, bench_config in suite_config.get("benchmarks", {}).items():
                patched = dict(bench_config)
                if "probes" in patched and "nprob" not in patched:
                    patched["nprob"] = patched["probes"]
                if "rerank_k" in patched and "epsilon" not in patched:
                    patched["epsilon"] = patched["rerank_k"]
                patched_benchmarks[bench_name] = patched
            shim_config = dict(suite_config)
            shim_config["benchmarks"] = patched_benchmarks

            system_metrics, pg_stats, dashboard_path = self.get_monitoring_data(suite_name)

            results_manager.process_suite_results(
                suite_type="vectorchord",
                config={suite_name: shim_config},
                results={suite_name: self.results.get(suite_name, {})},
                query_clients=self.query_clients,
                system_metrics=system_metrics,
                pg_stats=pg_stats,
                system_dashboard_path=dashboard_path,
            )


def main():
    """Main entry point for Ariadne benchmark suite."""
    parser = build_arg_parse()
    args = parser.parse_args()

    test_suite = TestSuite(
        suite_file=args.suite,
        url=args.url,
        devices=args.devices,
        chunk_size=args.chunk_size,
        skip_add_embeddings=args.skip_add_embeddings,
        centroids=args.centroids_file,
        centroids_table=args.centroids_table,
        skip_index_creation=args.skip_index_creation,
        query_clients=args.query_clients,
        max_load_threads=args.max_load_threads,
        debug=args.debug,
        overwrite_table=args.overwrite_table,
        debug_single_query=args.debug_single_query,
        build_only=args.build_only,
        max_queries=args.max_queries,
    )

    test_suite.run()
    print("Test suite completed.")


if __name__ == "__main__":
    main()
