"""
edb_vectorplus Benchmark Suite

Benchmarks vector search using the edb_vectorplus extension's `ivfplus`
access method for PostgreSQL.

edb_vectorplus builds on pgvector's `vector` type (the extension is
installed with CASCADE) but ships its own access method and its own
`ivfplus.*` GUC namespace, so it lives in its own suite module alongside
pgvector_suite.py / vectorchord_suite.py / pgpu_suite.py.

ivfplus indexes pgvector `vector` or `halfvec` columns; the YAML
`vectorType` key (default `vector`) picks which one the table is loaded as
and which opclass family is used. ivfplus has no `bit` opclass.
"""

import argparse
import math
import time
from typing import Optional

import psycopg
import pgvector.psycopg

import common
from results import ResultsManager

# Operator + opclass per metric. ivfplus opclass names mirror pgvector's:
# `<vectorType>_<suffix>`, e.g. vector_cosine_ops / halfvec_cosine_ops.
_METRIC_OPS = {
    "l2": "<->", "euclidean": "<->",
    "cos": "<=>", "angular": "<=>",
    "dot": "<#>", "ip": "<#>",
}
_METRIC_OPCLASS_SUFFIX = {
    "l2": "l2_ops", "euclidean": "l2_ops",
    "cos": "cosine_ops",
    "ip": "ip_ops", "dot": "ip_ops",
}

# Markdown report columns consumed by ResultsManager. `config_columns` are
# (label, extractor(config, results) -> str) rows for the per-run
# "Configuration" table; `bench_columns` are (benchmark key, header) pairs
# prepended to the benchmark results table.
CONFIG_COLUMNS = (
    ("Lists", lambda c, r: str(c.get("lists", r.get("lists", "N/A")))),
    ("Vector Type", lambda c, r: c.get("vectorType", "vector")),
)
BENCH_COLUMNS = (("probes", "Probes"),)


def metric_operator(metric: str) -> str:
    """Distance operator for the ORDER BY clause."""
    if metric not in _METRIC_OPS:
        raise ValueError(f"Unsupported metric type: {metric}")
    return _METRIC_OPS[metric]


def metric_opclass(metric: str, vector_type: str = "vector") -> str:
    """Operator class for CREATE INDEX, for a `vector` or `halfvec` column."""
    if metric not in _METRIC_OPCLASS_SUFFIX:
        raise ValueError(f"Unsupported metric type: {metric}")
    return f"{vector_type}_{_METRIC_OPCLASS_SUFFIX[metric]}"


def resolve_lists(config: dict, dataset: dict) -> int:
    """Resolve the `lists` build parameter; `auto` means sqrt(num vectors)."""
    lists = config["lists"]
    if isinstance(lists, str) and lists.strip().lower() == "auto":
        return max(1, int(math.sqrt(dataset["num"])))
    if not isinstance(lists, int) or lists < 1:
        raise ValueError(
            f"lists must be a positive integer or 'auto'; got {lists!r}"
        )
    return lists


def probe_gucs(benchmark: dict) -> list[str]:
    """Per-benchmark session GUCs -- the swept `probes` value."""
    return [
        f"SET ivfplus.probes = {benchmark['probes']}",
        "SET enable_seqscan = off",
    ]


def create_index_sql(table_name: str, config: dict, dataset: dict) -> str:
    """CREATE INDEX statement for the ivfplus access method."""
    lists = resolve_lists(config, dataset)
    opclass = metric_opclass(dataset["metric"], dataset.get("vector_type", "vector"))
    return (
        f"CREATE INDEX {table_name}_embedding_idx ON {table_name} "
        f"USING ivfplus (embedding {opclass}) "
        f"WITH (lists = {lists})"
    )


def print_index_config(config: dict, dataset: dict) -> None:
    """Debug banner printed before the index build."""
    print(f"\n🔧 Index Configuration (ivfplus):")
    print(f"    • Lists:           {resolve_lists(config, dataset)}")
    print(f"    • Metric Function: "
          f"{metric_opclass(dataset['metric'], dataset.get('vector_type', 'vector'))}")
    print()


def search_query_sql(table_name: str, metric_ops: str, top: int) -> str:
    """Single-stage KNN query. Matches common.TestSuite.warmup_query so the
    warmup, EXPLAIN and measurement loops all exercise the same plan."""
    return (
        f"SELECT id FROM {table_name} ORDER BY embedding {metric_ops} %s "
        f"LIMIT {top}"
    )


def build_arg_parse():
    """Build argument parser for the edb_vectorplus benchmark suite."""
    parser = argparse.ArgumentParser(description="edb_vectorplus Benchmark Suite")
    common.build_arg_parse(parser)
    return parser


class TestSuite(common.TestSuite):
    """
    Test suite for edb_vectorplus ivfplus indexing.

    Sweeps `ivfplus.probes` per benchmark point over an IVF index built
    with a fixed `lists` count.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        wrong = sorted({
            cfg["indexType"] for cfg in self.config.values()
            if cfg.get("indexType", "ivfplus") != "ivfplus"
        })
        if wrong:
            raise ValueError(
                f"edb_vectorplus_suite.py only runs indexType 'ivfplus'; "
                f"got {wrong}. Use pgvector_suite.py for hnsw / ivfflat / "
                f"ivfflat_bq_rerank configs."
            )

    # --- connection / extension -------------------------------------------

    def create_connection(self):
        """Create a database connection with pgvector type support."""
        conn = super().create_connection()
        pgvector.psycopg.register_vector(conn)
        return conn

    def init_ext(self, suite_name: Optional[str] = None):
        """Initialize required PostgreSQL extensions."""
        conn = super().create_connection()
        # Surface server NOTICEs in vsbt output
        conn.add_notice_handler(common.psql_log_handler)
        # CASCADE pulls in pgvector's `vector` extension automatically.
        conn.execute("CREATE EXTENSION IF NOT EXISTS edb_vectorplus CASCADE")
        conn.execute("CREATE EXTENSION IF NOT EXISTS pg_prewarm")
        conn.close()
        self.debug_log("Extensions initialized successfully.")

    def prewarm_index(self, table_name: str):
        """Prewarm the index into memory for consistent benchmarking."""
        index_name = self.index_name(table_name)
        conn = self.create_connection()
        self.check_index_fits_shared_buffers(conn, index_name, table_name)
        print("Prewarming the index into shared_buffers...", end="", flush=True)
        try:
            prewarm_start = time.perf_counter()
            conn.execute(f"SELECT pg_prewarm('{index_name}')")
            prewarm_time = time.perf_counter() - prewarm_start
            print(f" done! ({prewarm_time:.1f}s)")
        except psycopg.Error as e:
            print(f" failed! ({e.diag.message_primary})")
            self.debug_log(f"Prewarm failed: {e}")
        finally:
            conn.close()

    # --- metric hooks used by common.TestSuite ----------------------------

    @staticmethod
    def _get_metric_operator(metric: str) -> str:
        return metric_operator(metric)

    @staticmethod
    def _get_metric_func(metric: str) -> str:
        return metric_opclass(metric)

    # --- query execution --------------------------------------------------

    def apply_session_guc(self, conn, benchmark):
        for stmt in probe_gucs(benchmark):
            conn.execute(stmt)

    @staticmethod
    def process_batch(args):
        """Run one worker's batch of queries. Runs in an mp.Pool worker, so
        every element of `args` must be picklable (no closures, no self)."""
        test, answer, top, metric_ops, url, table_name, gucs, warmup_n = args

        conn = psycopg.connect(url)
        pgvector.psycopg.register_vector(conn)
        for stmt in gucs:
            conn.execute(stmt)

        query_sql = search_query_sql(table_name, metric_ops, top)
        cursor = conn.cursor()

        # Per-worker warmup; latencies/results discarded.
        if warmup_n:
            n_test = len(test)
            for j in range(warmup_n):
                cursor.execute(query_sql, (test[j % n_test],))
                cursor.fetchall()

        results = []
        for query, ground_truth in zip(test, answer):
            start = time.perf_counter()
            cursor.execute(query_sql, (query,))
            result = cursor.fetchall()
            end = time.perf_counter()

            result_ids = {p[0] for p in result[:top]}
            gt_ids = ground_truth[:top]
            ground_truth_ids = set(
                gt_ids.tolist() if hasattr(gt_ids, "tolist") else gt_ids
            )
            hit = len(result_ids & ground_truth_ids)
            results.append((hit, (start, end)))

        cursor.close()
        conn.close()
        return results

    def make_batch_args(self, test, answer, top, metric, table_name, benchmark,
                        warmup_n=0):
        """Prepare arguments for parallel batch processing."""
        return (
            test,
            answer,
            top,
            metric_operator(metric),
            self.url,
            table_name,
            probe_gucs(benchmark),
            warmup_n,
        )

    # --- index build ------------------------------------------------------

    def create_index(self, suite_name: str, table_name: str, dataset: dict) -> None:
        """Create the ivfplus index."""
        event, index_monitor_thread = super().create_index(
            suite_name, table_name, dataset
        )

        config = self.config[suite_name]
        pg_parallel_workers = config["pg_parallel_workers"]
        maintenance_work_mem = config.get("maintenance_work_mem")

        self.results[suite_name]["lists"] = resolve_lists(config, dataset)

        if self.debug:
            print_index_config(config, dataset)

        conn = self.create_connection()
        start_time = time.perf_counter()

        if maintenance_work_mem:
            conn.execute(f"SET maintenance_work_mem TO '{maintenance_work_mem}'")
        conn.execute(f"SET max_parallel_maintenance_workers TO {pg_parallel_workers}")
        conn.execute(f"SET max_parallel_workers TO {pg_parallel_workers}")
        conn.execute(create_index_sql(table_name, config, dataset))

        build_time = int(round(time.perf_counter() - start_time))
        self.results[suite_name]["index_build_time"] = build_time

        event.set()
        index_monitor_thread.join()

        print(f"Index build time: {build_time}s")

        conn.execute("CHECKPOINT")
        conn.close()
        print("Index built successfully.")

    # --- benchmark --------------------------------------------------------

    def sequential_bench(self, name, table_name, conn, metric, top, benchmark, dataset):
        self.apply_session_guc(conn, benchmark)
        metric_ops = metric_operator(metric)

        self.debug_log(
            f"Benchmark config: {benchmark}, metric={metric}, "
            f"metric_ops={metric_ops}"
        )

        self.warmup_for_benchmark(
            conn, table_name, dataset, metric_ops, top, name,
            benchmark=benchmark,
        )

        return super().sequential_bench(
            name, table_name, conn, metric_ops, top, benchmark, dataset
        )

    # --- reporting --------------------------------------------------------

    def print_summary_table(self, suite_name: str):
        """Print a Probes-keyed summary table. common.print_summary_table
        only knows the efSearch / nprob shapes."""
        benchmarks = self.config[suite_name].get("benchmarks", {})
        results = self.results.get(suite_name, {})
        if not benchmarks:
            return

        header = "| Probes    | Recall | QPS    | P50 (ms) | P99 (ms) |"
        sep =    "|-----------|--------|--------|----------|----------|"

        sb = results.get("shared_buffers", "N/A")
        idx_size = results.get("index_size", "N/A")
        qc = results.get("query_clients", 1)

        print(f"\n{'=' * len(sep)}")
        print(f"  Results Summary: {suite_name}")
        print(f"  shared_buffers: {sb} | clients: {qc} | index_size: {idx_size}")
        print(f"{'=' * len(sep)}")
        print(header)
        print(sep)

        for name, benchmark in benchmarks.items():
            r = results.get(name, {})
            if "recall" not in r:
                continue
            print(f"| {benchmark.get('probes', 'N/A'):<9} "
                  f"| {r['recall']:.4f} "
                  f"| {r['qps']:>6.2f} "
                  f"| {r['p50_latency']:>8.2f} "
                  f"| {r['p99_latency']:>8.2f} |")

        print()

    def generate_markdown_result(self):
        """Generate benchmark results with charts and consolidated CSV."""
        self.debug_log(f"Results: {self.results}")

        results_manager = ResultsManager()

        for suite_name in self.config:
            system_metrics, pg_stats, dashboard_path = self.get_monitoring_data(suite_name)

            results_manager.process_suite_results(
                suite_type="edb_vectorplus",
                config={suite_name: self.config[suite_name]},
                results={suite_name: self.results.get(suite_name, {})},
                query_clients=self.query_clients,
                system_metrics=system_metrics,
                pg_stats=pg_stats,
                system_dashboard_path=dashboard_path,
                config_columns=list(CONFIG_COLUMNS),
                bench_columns=list(BENCH_COLUMNS),
            )


def main():
    """Main entry point for the edb_vectorplus benchmark suite."""
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
        warmup=args.warmup,
    )

    test_suite.run()
    print("Test suite completed.")


if __name__ == "__main__":
    main()
