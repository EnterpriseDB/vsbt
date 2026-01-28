"""
pgvector Benchmark Suite

Benchmarks vector search using the pgvector extension with HNSW indexes
for PostgreSQL.
"""

import argparse
import time

import psycopg
import pgvector.psycopg

import common
from results import ResultsManager


def build_arg_parse():
    """Build argument parser for pgvector benchmark suite."""
    parser = argparse.ArgumentParser(description="pgvector Benchmark Suite")
    common.build_arg_parse(parser)
    return parser


class TestSuite(common.TestSuite):
    """
    Test suite for pgvector HNSW indexing.

    Uses the pgvector extension to build HNSW indexes and perform
    approximate nearest neighbor searches.
    """

    @staticmethod
    def process_batch(args):
        """Process a batch of queries in parallel."""
        test, answer, top, metric_ops, url, table_name, ef_search = args

        conn = psycopg.connect(url)
        pgvector.psycopg.register_vector(conn)
        conn.execute(f"SET hnsw.ef_search={ef_search}")

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
        return (
            test,
            answer,
            top,
            metric_ops,
            self.url,
            table_name,
            benchmark["efSearch"],
        )

    @staticmethod
    def _get_metric_operator(metric: str) -> str:
        """Convert metric name to PostgreSQL operator."""
        operators = {
            "l2": "<->",
            "euclidean": "<->",
            "cos": "<=>",
            "angular": "<=>",
            "dot": "<#>",
            "ip": "<#>",
        }
        if metric not in operators:
            raise ValueError(f"Unsupported metric type: {metric}")
        return operators[metric]

    @staticmethod
    def _get_metric_func(metric: str) -> str:
        """Convert metric name to pgvector operator class."""
        funcs = {
            "l2": "vector_l2_ops",
            "euclidean": "vector_l2_ops",
            "cos": "vector_cosine_ops",
            "ip": "vector_ip_ops",
            "dot": "vector_ip_ops",
        }
        if metric not in funcs:
            raise ValueError(f"Unsupported metric type: {metric}")
        return funcs[metric]

    def create_connection(self):
        """Create a database connection with pgvector support."""
        conn = super().create_connection()
        pgvector.psycopg.register_vector(conn)
        return conn

    def init_ext(self, suite_name: str = None):
        """Initialize required PostgreSQL extensions."""
        conn = super().create_connection()
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute("CREATE EXTENSION IF NOT EXISTS pg_prewarm")
        conn.close()
        self.debug_log("Extensions initialized successfully.")

    def prewarm_index(self, table_name: str):
        """Prewarm the index into memory for consistent benchmarking."""
        conn = self.create_connection()
        print("Prewarming the index...", end="", flush=True)
        try:
            conn.execute(f"SELECT pg_prewarm('{table_name}_embedding_idx')")
            print(" done!")
        except psycopg.Error as e:
            print(f" failed! ({e.diag.message_primary})")
            self.debug_log(f"Prewarm failed: {e}")
        finally:
            conn.close()

    def create_index(self, suite_name: str, table_name: str, dataset: dict):
        """Create an HNSW index using pgvector."""
        event, index_monitor_thread = super().create_index(
            suite_name, table_name, dataset
        )

        config = self.config[suite_name]
        pg_parallel_workers = config["pg_parallel_workers"]
        m = config["m"]
        ef_construction = config["efConstruction"]
        metric = dataset["metric"]
        metric_func = self._get_metric_func(metric)

        if self.debug:
            print(f"\n🔧 Index Configuration (HNSW):")
            print(f"    • M:               {m}")
            print(f"    • EF Construction: {ef_construction}")
            print(f"    • Metric Function: {metric_func}")
            print()

        conn = self.create_connection()
        start_time = time.perf_counter()

        conn.execute(f"SET max_parallel_maintenance_workers TO {pg_parallel_workers}")
        conn.execute(f"SET max_parallel_workers TO {pg_parallel_workers}")
        conn.execute(
            f"CREATE INDEX {table_name}_embedding_idx ON {table_name} "
            f"USING hnsw (embedding {metric_func}) WITH (m = {m}, ef_construction = {ef_construction})"
        )

        build_time = int(round(time.perf_counter() - start_time))
        self.results[suite_name]["index_build_time"] = build_time
        print(f"Index build time: {build_time}s")

        conn.close()
        print("Index built successfully.")

        event.set()
        index_monitor_thread.join()

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
        conn.execute(f"SET hnsw.ef_search={benchmark['efSearch']}")

        metric_ops = self._get_metric_operator(metric)

        self.debug_log(
            f"Benchmark config: ef_search={benchmark['efSearch']}, "
            f"metric={metric}, metric_ops={metric_ops}"
        )

        return super().sequential_bench(
            name, table_name, conn, metric_ops, top, benchmark, dataset
        )

    def generate_markdown_result(self):
        """Generate benchmark results with charts and consolidated CSV."""
        self.debug_log(f"Results: {self.results}")

        results_manager = ResultsManager()

        # Get monitoring data for each suite
        for suite_name in self.config:
            system_metrics, pg_stats, dashboard_path = self.get_monitoring_data(suite_name)

            results_manager.process_suite_results(
                suite_type="pgvector",
                config={suite_name: self.config[suite_name]},
                results={suite_name: self.results.get(suite_name, {})},
                query_clients=self.query_clients,
                system_metrics=system_metrics,
                pg_stats=pg_stats,
                system_dashboard_path=dashboard_path,
            )


def main():
    """Main entry point for pgvector benchmark suite."""
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
    )

    test_suite.run()
    print("Test suite completed.")


if __name__ == "__main__":
    main()
